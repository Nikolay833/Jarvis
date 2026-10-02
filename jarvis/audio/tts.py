"""Text to speech: Kokoro synthesis with sentence-by-sentence playback."""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from typing import Any

import numpy as np

from .bargein import PlaybackTap
from .recorder import display_level, rms_of
from .mic import parse_device
from .voicefx import apply_fx

SAMPLE_RATE = 24000
PLAY_BLOCK = 800  # 33 ms -> about 30 level events per second
FADE_SAMPLES = 720  # 30 ms fade-out when interrupted (fits in one play block)

log = logging.getLogger("jarvis.tts")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# A "sentence" that is only a list number ("2.") or ends in a common abbreviation is not a
# sentence: speaking it alone puts the pause in the wrong place ("...ago two. <pause> Settings").
_NOT_AN_END = re.compile(r"(?:^|\s)(?:\d{1,3}|[A-Za-z]|mr|mrs|ms|dr|st|vs|etc|e\.g|i\.e|no)\.$", re.IGNORECASE)


def _merge(parts: list[str]) -> list[str]:
    out: list[str] = []
    carry = ""
    for p in parts:
        p = (carry + " " + p).strip() if carry else p.strip()
        carry = ""
        if not p:
            continue
        if _NOT_AN_END.search(p):
            carry = p
        else:
            out.append(p)
    if carry:
        out.append(carry)
    return out


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental synthesis."""
    text = re.sub(r"\s+", " ", text).strip()
    return _merge(_SENTENCE_END.split(text))


class SentenceBuffer:
    """Turns streamed text deltas into complete sentences (a sentence ends at .!? plus whitespace)."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        parts = _SENTENCE_END.split(self._buf)
        if len(parts) < 2:
            return []
        done = [re.sub(r"\s+", " ", p).strip() for p in parts[:-1] if p.strip()]
        tail = parts[-1]
        # A trailing list number or abbreviation waits for the words that follow it.
        while done and _NOT_AN_END.search(done[-1]):
            tail = done.pop() + " " + tail
        self._buf = tail
        return _merge(done)

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return split_sentences(rest)

    def discard(self) -> None:
        self._buf = ""


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    """"bm_george" -> [("bm_george", 1.0)]; "bm_george:0.6,bm_lewis:0.4" -> normalised weights."""
    out: list[tuple[str, float]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        name = name.strip()
        try:
            weight = float(w) if w.strip() else 1.0
        except ValueError:
            raise ValueError(f"bad voice weight in {part!r}") from None
        if not name or weight < 0:
            raise ValueError(f"bad voice spec {part!r}")
        out.append((name, weight))
    total = sum(w for _, w in out)
    if not out or total <= 0:
        raise ValueError(f"empty voice spec {spec!r}")
    return [(n, w / total) for n, w in out]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def fade_out(block: np.ndarray, n: int = FADE_SAMPLES) -> np.ndarray:
    """Linear fade to silence over the first n samples of `block`, silence after (no click on interrupt)."""
    out = np.zeros(len(block), dtype=np.float32)
    n = min(n, len(block))
    if n:
        out[:n] = block[:n].astype(np.float32) * np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def truncate_words(sentence: str, fraction: float) -> str:
    """The first `fraction` of the words of a sentence (what was spoken before an interruption)."""
    words = sentence.split()
    return " ".join(words[:int(len(words) * min(1.0, max(0.0, fraction)))])


class ConsoleStream:
    """Streaming API of the console speaker: prints each pushed sentence at once."""

    def __init__(self, speaker: "ConsoleSpeaker") -> None:
        self._speaker = speaker
        self.first_audio_at: float | None = None
        self.pushed = 0

    def push(self, sentence: str) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
        self.pushed += 1
        self._speaker.say_sentence(sentence)

    async def finish(self) -> None:
        return None


class ConsoleSpeaker:
    """No-audio speaker: prints each sentence and emits `reply` events."""

    def __init__(self, bus: Any, prefix: str = "Jarvis: ") -> None:
        self.bus = bus
        self.prefix = prefix

    def say_sentence(self, sentence: str) -> None:
        self.bus.emit_nowait("reply", text=sentence)
        print(f"{self.prefix}{sentence}", flush=True)

    async def speak(self, text: str, force: bool = False) -> None:
        for sentence in split_sentences(text):
            self.say_sentence(sentence)

    def start_stream(self, force: bool = False) -> ConsoleStream:
        return ConsoleStream(self)

    def begin_turn(self) -> None:
        pass

    def stop(self) -> None:
        pass


class KokoroStream:
    """Incremental utterance: `push(sentence)` as text arrives, `await finish()` when done.

    Synthesis of sentence N+1 overlaps playback of sentence N (see KokoroSpeaker._speak_blocking).
    """

    def __init__(self, speaker: "KokoroSpeaker", force: bool = False) -> None:
        self._speaker = speaker
        self._q: queue.Queue[str | None] = queue.Queue()
        self.first_audio_at: float | None = None
        self.pushed = 0
        self.force = force  # speak even after a barge-in muted the speaker (confirmation prompts)
        self.epoch = speaker.epoch
        self._task = asyncio.get_running_loop().create_task(self._run())

    @property
    def cancelled(self) -> bool:
        """True once a barge-in cut the reply this stream belongs to: it drops everything further."""
        sp = self._speaker
        return not self.force and (sp.muted or self.epoch != sp.epoch)

    async def _run(self) -> None:
        sp = self._speaker
        async with sp._lock:
            if self.cancelled:
                return
            sp._stop.clear()
            sp._interrupt.clear()
            await asyncio.to_thread(sp._speak_blocking, self._q, self)

    def push(self, sentence: str) -> None:
        if sentence.strip() and not self.cancelled:
            self.pushed += 1
            self._q.put_nowait(sentence)

    async def finish(self) -> None:
        self._q.put_nowait(None)
        await self._task


class KokoroSpeaker:
    def __init__(self, bus: Any, voice: str = "bm_george", lang_code: str = "b",
                 speed: float = 1.0, device: str = "", fx: str = "none", fx_amount: float = 0.35) -> None:
        self.bus = bus
        self.voice = voice
        self.fx = fx
        self.fx_amount = fx_amount
        self._voice_arg: Any = None
        self.lang_code = lang_code
        self.speed = speed
        self.device = parse_device(device)
        self._pipeline: Any = None
        self._stop = threading.Event()
        self._lock = asyncio.Lock()  # one utterance at a time
        # barge-in
        self._interrupt = threading.Event()  # fade out and stop the current utterance
        self.muted = False  # after an interrupt: later sentences of the cut reply are dropped
        self.epoch = 0
        self.spoken: list[str] = []  # sentences (the last one possibly cut) of the latest utterance
        self.tap = PlaybackTap()  # what is playing, for the echo gate
        self.output_latency: Any = None  # sounddevice latency ("low" keeps the interrupt fade tight)
        self.on_playback_start: Any = None  # hooks: the barge-in monitor
        self.on_playback_end: Any = None

    def load(self) -> None:
        if self._pipeline is None:
            from kokoro import KPipeline

            device = "cuda" if _cuda_available() else "cpu"
            try:
                self._pipeline = KPipeline(lang_code=self.lang_code, device=device)
            except TypeError:  # older kokoro without the `device` argument (auto-selects)
                self._pipeline = KPipeline(lang_code=self.lang_code)
            except Exception:  # noqa: BLE001 - e.g. CUDA init failure: retry on CPU
                if device == "cpu":
                    raise
                log.warning("Kokoro failed on %s, falling back to cpu", device, exc_info=True)
                self._pipeline = KPipeline(lang_code=self.lang_code, device="cpu")
            try:
                log.info("kokoro device: %s", self._pipeline.model.device)
            except Exception:  # noqa: BLE001
                log.info("kokoro device: %s (requested)", device)

    def _voice(self) -> Any:
        """What KPipeline gets as `voice`: a name, or for a weighted blend a tensor (cached)."""
        if self._voice_arg is None:
            spec = parse_voice_spec(self.voice)
            if len(spec) == 1:
                self._voice_arg = spec[0][0]
            else:
                packs = [self._pipeline.load_voice(n) * w for n, w in spec]
                blend = packs[0]
                for p in packs[1:]:
                    blend = blend + p
                self._voice_arg = blend  # float32 tensor; load_voice() passes it through
        return self._voice_arg

    def synth(self, sentence: str) -> np.ndarray:
        """Blocking synthesis of one sentence to float32 24 kHz mono, with the voice fx applied."""
        self.load()
        parts: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipeline(sentence, voice=self._voice(), speed=self.speed):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        try:
            return apply_fx(out, self.fx, self.fx_amount)
        except Exception:  # noqa: BLE001 - never lose speech to an fx bug
            log.exception("voice fx failed; playing dry")
            return out

    def warm_up(self) -> None:
        """Synthesize a short phrase so kernels are compiled before the first real reply."""
        self.synth("Ready.")

    def stop(self) -> None:
        """Interrupt current speech (thread-safe)."""
        self._stop.set()

    def interrupt(self, mute: bool = True) -> None:
        """Barge-in (thread-safe): fade out ~30 ms and stop. With `mute` the rest of the reply being
        generated is dropped too (until `begin_turn`); streams opened with force=True still speak."""
        if mute:
            self.muted = True
            self.epoch += 1
        self._interrupt.set()

    def begin_turn(self) -> None:
        """A new turn starts: speak again."""
        self.muted = False

    async def speak(self, text: str, force: bool = False) -> None:
        sentences = split_sentences(text)
        if not sentences:
            return
        stream = self.start_stream(force)
        for s in sentences:
            stream.push(s)
        await stream.finish()

    def start_stream(self, force: bool = False) -> KokoroStream:
        """Begin an utterance fed sentence by sentence. Must be called inside the event loop."""
        return KokoroStream(self, force)

    @staticmethod
    def _notify(hook: Any) -> None:
        if hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001 - the barge-in monitor must never break speech
                log.exception("playback hook failed")

    def _speak_blocking(self, src: "queue.Queue[str | None]", stream: KokoroStream | None = None) -> None:
        """Play sentences taken from `src` (None ends the utterance) while synthesizing the next."""
        import sounddevice as sd

        q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=2)

        def producer() -> None:
            try:
                while not self._stop.is_set():
                    try:
                        s = src.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if s is None:
                        break
                    audio = self.synth(s)
                    while not self._stop.is_set():
                        try:
                            q.put((s, audio), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception:  # noqa: BLE001
                log.exception("TTS synthesis failed")
            finally:
                while True:
                    try:
                        q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if self._stop.is_set():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass

        t0 = time.perf_counter()
        first_audio = True
        self.spoken = []
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        kw: dict[str, Any] = {} if self.output_latency is None else {"latency": self.output_latency}
        try:
            with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self.device,
                                 **kw) as out:
                try:
                    self.tap.latency = min(0.5, max(0.0, float(getattr(out, "latency", 0.0))))
                except (TypeError, ValueError):
                    self.tap.latency = 0.0
                self._notify(self.on_playback_start)
                faded = False
                while not self._stop.is_set() and not self._interrupt.is_set():
                    try:
                        item = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    sentence, audio = item
                    if first_audio:
                        first_audio = False
                        now = time.perf_counter()
                        if stream is not None:
                            stream.first_audio_at = now
                        log.info("tts first audio after %.1f s", now - t0)
                    self.bus.emit_nowait("reply", text=sentence)
                    self.spoken.append(sentence)
                    for i in range(0, len(audio), PLAY_BLOCK):
                        if self._stop.is_set():
                            break
                        block = audio[i:i + PLAY_BLOCK]
                        if self._interrupt.is_set():  # barge-in: one last block, faded out
                            block = fade_out(block)
                            faded = True
                        self.bus.emit_nowait("level", rms=display_level(rms_of(block), gain=4.0))
                        self.tap.note(block, SAMPLE_RATE)
                        out.write(block.reshape(-1, 1))
                        if faded:
                            kept = truncate_words(sentence, i / max(1, len(audio)))
                            if kept:
                                self.spoken[-1] = kept
                            else:
                                self.spoken.pop()
                            break
                    if faded:
                        break
        finally:
            self._stop.set()  # lets the producer exit if we left early
            t.join(timeout=2)
            self._stop.clear()
            self._interrupt.clear()
            self._notify(self.on_playback_end)
            self.bus.emit_nowait("level", rms=0.0)
