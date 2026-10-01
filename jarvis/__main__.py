"""Entry point: `python -m jarvis` or `jarvis`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import fastpath
from .agent import Agent, Confirmer
from .bus import EventBus
from .config import Config, load_config, repo_root
from .llm import LLMError, OllamaClient
from .tools import load_all
from .tools.context import set_context

log = logging.getLogger("jarvis")


@dataclass
class Request:
    kind: str  # "text" | "activate"
    text: str = ""
    source: str = "hotkey"  # activate only: "wake" (audio after the wake word is queued) | "hotkey"

BANNER = "Jarvis ready. Say 'Hey Jarvis' or press Ctrl+Alt+J."
SORRY = "Sorry sir, I didn't catch that."
APOLOGY = "Apologies sir, something went wrong. Check the console."
NO_OLLAMA = "I cannot reach my language model, sir. Is Ollama running?"


class Assistant:
    def __init__(self, cfg: Config, voice: bool = True, debug_audio: bool = False) -> None:
        self.cfg = cfg
        self.voice = voice
        self.bus = EventBus(cfg.bus.host, cfg.bus.port)
        self.requests: asyncio.Queue[Request] = asyncio.Queue()
        self.turn_lock = asyncio.Lock()
        self.wake: Any = None
        self.recorder: Any = None
        self.stt: Any = None
        self.speaker: Any = None
        self.mic: Any = None
        self.llm_ok = False
        self._stt_secs = 0.0
        self._first_audio: float | None = None

        if voice:
            from .audio.mic import MicStream
            from .audio.recorder import Recorder
            from .audio.stt import Transcriber
            from .audio.tts import KokoroSpeaker
            from .audio.wakeword import WakeWordDetector

            a = cfg.audio
            self.mic = MicStream(a.input_device)
            self.wake = WakeWordDetector(cfg.wakeword.model, cfg.wakeword.threshold, self.mic, debug=debug_audio)
            self.recorder = Recorder(self.mic, a.silence_seconds, a.max_record_seconds, a.no_speech_timeout)
            w = cfg.whisper
            self.stt = Transcriber(w.model, w.device, w.compute_type, w.fallback_model, w.language)
            self.speaker = KokoroSpeaker(self.bus, cfg.tts.voice, cfg.tts.lang_code, cfg.tts.speed, a.output_device)
        else:
            from .audio.tts import ConsoleSpeaker

            self.speaker = ConsoleSpeaker(self.bus)

        self.llm = OllamaClient(cfg.ollama.url, cfg.ollama.model, cfg.ollama.think,
                                cfg.ollama.timeout, cfg.ollama.num_ctx, cfg.ollama.keep_alive,
                                cfg.ollama.max_reply_tokens)
        self.confirmer = Confirmer(self.bus, self.speaker.speak, cfg.safety.confirm_timeout,
                                   listen=self.listen_text if voice else None)
        set_context(cfg, self.bus, self.announce)
        self.agent = Agent(self.llm, load_all(), self.bus, self.confirmer,
                           cfg.agent.max_steps, cfg.agent.max_history_messages, cfg.agent.stream_replies)

    # ---- audio helpers ---------------------------------------------------
    async def warm_up(self) -> None:
        """Load models and run one dummy pass of each (CUDA kernels, VRAM) in parallel."""
        t_all = time.perf_counter()
        if self.voice:
            log.info("loading models (first run downloads them)...")

        async def timed(label: str, fn: Any) -> None:
            t0 = time.perf_counter()
            await asyncio.to_thread(fn)
            log.info("%s (%.1f s)", label, time.perf_counter() - t0)

        async def warm(label: str, load: Any, warm: Any) -> None:
            await timed(label + " loaded", load)
            try:
                await timed(label + " warm-up", warm)
            except Exception:  # noqa: BLE001 - warm-up is best effort
                log.warning("%s warm-up failed", label, exc_info=True)

        async def warm_llm() -> None:
            try:
                secs = await self.llm.warm_up()
                self.llm_ok = True
                log.info("ollama model %s loaded in %.1f s", self.cfg.ollama.model, secs)
            except LLMError as exc:
                log.warning("Ollama not reachable at %s (%s). Start Ollama and run `ollama pull %s`.",
                            self.cfg.ollama.url, exc, self.cfg.ollama.model)

        jobs = [warm_llm()]
        if self.voice:
            jobs += [timed("wake word", self.wake.load),
                     warm("whisper", self.stt.load, self.stt.warm_up),
                     warm("tts", self.speaker.load, self.speaker.warm_up)]
        await asyncio.gather(*jobs)
        log.info("startup warm-up done in %.1f s", time.perf_counter() - t_all)

    async def record_text(self, cancel: threading.Event | None = None, flush: bool = False,
                          before: Any = None) -> str:
        audio = await asyncio.to_thread(
            lambda: self.recorder.record_blocking(
                lambda lvl: self.bus.emit_nowait("level", rms=lvl), cancel, flush, before))
        self.bus.emit_nowait("level", rms=0.0)
        if audio.size == 0:
            log.warning("no speech heard (recording ended: %s after %.1f s)",
                        self.recorder.last_reason, self.recorder.last_seconds)
            return ""
        t0 = time.perf_counter()
        text = (await asyncio.to_thread(self.stt.transcribe, audio)).strip()
        self._stt_secs = time.perf_counter() - t0
        log.info("heard: '%s' (stt %.1f s, audio %.1f s)", text, self._stt_secs, audio.size / 16000)
        return text

    def _chime(self) -> None:
        from .audio.chime import play_chime
        from .audio.mic import parse_device

        play_chime(parse_device(self.cfg.audio.output_device), blocking=True)

    async def listen_text(self) -> str:
        """Record one utterance and transcribe (used for spoken yes/no). Cancel-safe."""
        cancel = threading.Event()
        self.bus.emit_nowait("state", state="listening")
        task = asyncio.ensure_future(self.record_text(cancel, flush=True))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self.bus.emit_nowait("state", state="thinking")

    async def say_safe(self, text: str) -> None:
        """Speak without ever raising (used for error and status messages)."""
        try:
            self.bus.emit_nowait("state", state="speaking")
            await self.speaker.speak(text)
        except Exception:  # noqa: BLE001
            log.exception("could not speak %r", text)

    # ---- speaking out of turn -------------------------------------------
    async def announce(self, text: str) -> None:
        """Speak a background announcement (e.g. Claude job finished) once the current turn ends."""
        async with self.turn_lock:
            self.bus.emit_nowait("state", state="speaking")
            try:
                await self.speaker.speak(text)
            finally:
                self.bus.emit_nowait("state", state="idle")

    # ---- one turn --------------------------------------------------------
    async def run_fast(self, fp: fastpath.FastPath, text: str) -> None:
        """Answer a fast-path command without the LLM (see fastpath.py)."""
        if fp.stop_speaking:
            self.speaker.stop()
            return
        self.agent.add_exchange(text, fp.reply)
        self.bus.emit_nowait("state", state="speaking")
        action = asyncio.create_task(self._fast_action(fp)) if fp.action else None
        stream = self.speaker.start_stream()
        stream.push(fp.reply)
        self._first_audio = stream.first_audio_at
        await stream.finish()
        self._first_audio = stream.first_audio_at
        if action is not None:
            err = await action
            if err:
                await self.say_safe(f"I am afraid that failed, sir. {err}")

    async def _fast_action(self, fp: fastpath.FastPath) -> str:
        """Run the side effect of a fast path. Returns an error text, or '' on success."""
        from .tools import system

        kind = fp.action[0]
        try:
            if kind == "volume":
                await asyncio.to_thread(system.press_volume_key, fp.action[1])
                return ""
            args = {"name": fp.action[1]} if kind == "open_app" else {}
            result = await self.agent.registry.call(kind, args)
            return result[:120] if result.startswith("Error:") else ""
        except Exception as exc:  # noqa: BLE001
            log.exception("fast path action failed")
            return str(exc)[:120]

    def _log_latency(self, t_heard: float, t_llm_start: float | None, first_token: float | None,
                     first_audio: float | None) -> None:
        now = time.perf_counter()
        t_heard -= self._stt_secs  # measure from the end of speech: transcription is part of the wait
        parts = [f"stt {self._stt_secs:.1f} s"]
        if t_llm_start is not None and first_token is not None:
            parts.append(f"llm first token {first_token - t_llm_start:.1f} s")
        if first_audio is not None:
            parts.append(f"first audio {first_audio - t_heard:.1f} s")
        parts.append(f"total {now - t_heard:.1f} s")
        log.info("latency: %s", ", ".join(parts))

    async def run_turn(self, req: Request) -> None:
        async with self.turn_lock:
            try:
                self._stt_secs = 0.0
                self._first_audio = None
                text = req.text
                if req.kind == "activate":
                    if not self.voice:
                        return
                    self.bus.emit_nowait("state", state="listening")
                    text = await self.record_text(flush=req.source != "wake",
                                                  before=self._chime if self.cfg.audio.chime else None)
                    if not text:
                        await self.say_safe(SORRY)
                        return
                if not text:
                    return
                t_heard = time.perf_counter()
                self.bus.emit_nowait("transcript", text=text, final=True)
                self.bus.emit_nowait("state", state="thinking")
                fp = fastpath.match(text) if self.cfg.agent.fast_paths else None
                if fp is not None:
                    log.info("fast path: %s", fp.kind)
                    await self.run_fast(fp, text)
                    self._log_latency(t_heard, None, None, self._first_audio)
                    return
                self.bus.emit_nowait("state", state="speaking")  # sentences stream out as they finish
                reply = await self.agent.handle(text, speaker=self.speaker)
                log.info("reply: '%s'", reply)
                if not self.agent.streamed:
                    await self.speaker.speak(reply)
                self._log_latency(t_heard, self.agent.timing.get("start"), self.agent.timing.get("first_token"),
                                  self.agent.first_audio_at)
            except Exception:  # noqa: BLE001
                log.exception("turn failed")
                await self.say_safe(APOLOGY)
            finally:
                self.bus.emit_nowait("state", state="idle")

    def submit_text(self, text: str) -> None:
        """Typed or bus text: answers a pending confirmation, else becomes a request."""
        text = text.strip()
        if not text:
            return
        if self.confirmer.pending and self.confirmer.resolve_text(text):
            return
        self.requests.put_nowait(Request("text", text))

    async def dispatch_bus(self) -> None:
        while True:
            msg = await self.bus.inbound.get()
            kind = msg.get("type")
            if kind == "confirm_response":
                self.confirmer.resolve(str(msg.get("id", "")), bool(msg.get("approved")))
            elif kind == "text_input":
                self.submit_text(str(msg.get("text", "")))
            elif kind == "activate":
                if self.turn_lock.locked():
                    self.speaker.stop()  # interrupt speech; ignore if mid-recording
                else:
                    self.requests.put_nowait(Request("activate"))

    async def stdin_repl(self) -> None:
        loop = asyncio.get_running_loop()
        print("Jarvis text mode. Type a request, or 'quit'.", flush=True)
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line or line.strip().lower() in ("quit", "exit"):  # EOF or quit
                await asyncio.sleep(0.3)
                while not self.requests.empty() or self.turn_lock.locked():
                    await asyncio.sleep(0.2)  # let queued turns finish
                raise SystemExit(0)
            line = line.strip()
            self.submit_text(line)

    # ---- main loop -------------------------------------------------------
    async def next_request(self) -> Request:
        """Wake word or a queued request, whichever comes first."""
        if not self.voice:
            return await self.requests.get()
        dropped = self.mic.flush()  # stale audio, e.g. our own TTS reply
        log.debug("flushed %.1f s of queued mic audio", dropped)
        stop = threading.Event()
        wake_task = asyncio.ensure_future(asyncio.to_thread(self.wake.wait_blocking, stop))
        req_task = asyncio.ensure_future(self.requests.get())
        try:
            done, _ = await asyncio.wait({wake_task, req_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            stop.set()
            raise
        stop.set()
        await asyncio.gather(wake_task, return_exceptions=True)
        if req_task in done:
            return req_task.result()
        req_task.cancel()
        woke = wake_task.result()
        return Request("activate", source="wake") if woke else await self.requests.get()

    async def run(self, repl: bool = False) -> None:
        try:
            await self.bus.start()
        except OSError as exc:
            log.warning("event bus unavailable (%s); running without the orb", exc)
        if self.mic is not None:
            try:
                self.mic.start()
            except Exception as exc:  # noqa: BLE001
                log.error("cannot open the microphone (%s); check Windows sound settings or "
                          "audio.input_device in config.toml", exc)
                raise
        await self.warm_up()
        bg = [asyncio.create_task(self.dispatch_bus())]
        if repl:
            bg.append(asyncio.create_task(self.stdin_repl()))
        log.info("model %s, config: %s", self.cfg.ollama.model, self.cfg.source)
        log.info(BANNER if self.voice else "Jarvis ready (no voice).")
        try:
            if self.voice and self.cfg.audio.announce_ready:
                await self.say_safe("Online, sir.")
                if not self.llm_ok:
                    await self.say_safe(NO_OLLAMA)
                self.bus.emit_nowait("state", state="idle")
            while True:
                done_bg = [t for t in bg if t.done()]
                for t in done_bg:
                    t.result()  # propagate SystemExit / errors
                req = await self.next_request_or_bg(bg)
                if req is not None:
                    await self.run_turn(req)
        finally:
            for t in bg:
                t.cancel()
            if self.mic is not None:
                self.mic.stop()
            await self.bus.stop()
            await self.llm.aclose()

    async def next_request_or_bg(self, bg: list[asyncio.Task]) -> Request | None:
        """next_request, but wake up if a background task (REPL quit) finishes."""
        main = asyncio.ensure_future(self.next_request())
        try:
            done, _ = await asyncio.wait({main, *bg}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            main.cancel()
            raise
        if main in done:
            return main.result()
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)
        for t in done:
            t.result()
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis", description="Jarvis local voice assistant")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--text", action="store_true", help="keyboard REPL, no audio (implies --no-voice)")
    p.add_argument("--no-voice", action="store_true", help="do not load audio models; bus and text input only")
    p.add_argument("--debug-audio", action="store_true",
                   help="print mic level and wake word score twice a second")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup_logging(verbose: bool = False) -> None:
    """Console plus rotating file logs/jarvis.log (1 MB x 3) in the repo root."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        log_dir = repo_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / "jarvis.log", maxBytes=1_000_000, backupCount=3,
                                                  encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:
        log.warning("file logging disabled: %s", exc)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    voice = not (args.text or args.no_voice)

    async def runner() -> None:
        assistant = Assistant(cfg, voice=voice, debug_audio=args.debug_audio)
        await assistant.run(repl=args.text)

    try:
        asyncio.run(runner())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
