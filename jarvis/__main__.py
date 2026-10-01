"""Entry point: `python -m jarvis` or `jarvis`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any

from .agent import Agent, Confirmer
from .bus import EventBus
from .config import Config, load_config
from .llm import OllamaClient
from .tools import load_all
from .tools.context import set_context

log = logging.getLogger("jarvis")


@dataclass
class Request:
    kind: str  # "text" | "activate"
    text: str = ""


class Assistant:
    def __init__(self, cfg: Config, voice: bool = True) -> None:
        self.cfg = cfg
        self.voice = voice
        self.bus = EventBus(cfg.bus.host, cfg.bus.port)
        self.requests: asyncio.Queue[Request] = asyncio.Queue()
        self.turn_lock = asyncio.Lock()
        self.wake: Any = None
        self.recorder: Any = None
        self.stt: Any = None
        self.speaker: Any = None

        if voice:
            from .audio.recorder import Recorder
            from .audio.stt import Transcriber
            from .audio.tts import KokoroSpeaker
            from .audio.wakeword import WakeWordDetector

            a = cfg.audio
            self.wake = WakeWordDetector(cfg.wakeword.model, cfg.wakeword.threshold, a.input_device)
            self.recorder = Recorder(a.silence_seconds, a.max_record_seconds, a.no_speech_timeout, a.input_device)
            w = cfg.whisper
            self.stt = Transcriber(w.model, w.device, w.compute_type, w.fallback_model, w.language)
            self.speaker = KokoroSpeaker(self.bus, cfg.tts.voice, cfg.tts.lang_code, cfg.tts.speed, a.output_device)
        else:
            from .audio.tts import ConsoleSpeaker

            self.speaker = ConsoleSpeaker(self.bus)

        self.llm = OllamaClient(cfg.ollama.url, cfg.ollama.model, cfg.ollama.think,
                                cfg.ollama.timeout, cfg.ollama.num_ctx, cfg.ollama.keep_alive)
        self.confirmer = Confirmer(self.bus, self.speaker.speak, cfg.safety.confirm_timeout,
                                   listen=self.listen_text if voice else None)
        set_context(cfg, self.bus, self.announce)
        self.agent = Agent(self.llm, load_all(), self.bus, self.confirmer,
                           cfg.agent.max_steps, cfg.agent.max_history_messages)

    # ---- audio helpers ---------------------------------------------------
    async def load_models(self) -> None:
        if not self.voice:
            return
        log.info("loading models (first run downloads them)...")
        await asyncio.gather(
            asyncio.to_thread(self.wake.load),
            asyncio.to_thread(self.stt.load),
            asyncio.to_thread(self.speaker.load),
        )

    async def record_text(self, cancel: threading.Event | None = None) -> str:
        audio = await asyncio.to_thread(
            self.recorder.record_blocking, lambda lvl: self.bus.emit_nowait("level", rms=lvl), cancel)
        self.bus.emit_nowait("level", rms=0.0)
        if audio.size == 0:
            return ""
        return (await asyncio.to_thread(self.stt.transcribe, audio)).strip()

    async def listen_text(self) -> str:
        """Record one utterance and transcribe (used for spoken yes/no). Cancel-safe."""
        cancel = threading.Event()
        self.bus.emit_nowait("state", state="listening")
        task = asyncio.ensure_future(self.record_text(cancel))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self.bus.emit_nowait("state", state="thinking")

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
    async def run_turn(self, req: Request) -> None:
        async with self.turn_lock:
            try:
                text = req.text
                if req.kind == "activate":
                    if not self.voice:
                        return
                    self.bus.emit_nowait("state", state="listening")
                    text = await self.record_text()
                if not text:
                    return
                self.bus.emit_nowait("transcript", text=text, final=True)
                self.bus.emit_nowait("state", state="thinking")
                reply = await self.agent.handle(text)
                self.bus.emit_nowait("state", state="speaking")
                await self.speaker.speak(reply)
            except Exception:  # noqa: BLE001
                log.exception("turn failed")
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
        return Request("activate") if woke else await self.requests.get()

    async def run(self, repl: bool = False) -> None:
        try:
            await self.bus.start()
        except OSError as exc:
            log.warning("event bus unavailable (%s); running without the orb", exc)
        await self.load_models()
        bg = [asyncio.create_task(self.dispatch_bus())]
        if repl:
            bg.append(asyncio.create_task(self.stdin_repl()))
        log.info("Jarvis ready (model %s, config: %s)", self.cfg.ollama.model, self.cfg.source)
        try:
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
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    voice = not (args.text or args.no_voice)

    async def runner() -> None:
        assistant = Assistant(cfg, voice=voice)
        await assistant.run(repl=args.text)

    try:
        asyncio.run(runner())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
