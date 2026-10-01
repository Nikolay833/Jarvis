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

from .recorder import display_level, rms_of
from .mic import parse_device

SAMPLE_RATE = 24000
PLAY_BLOCK = 800  # 33 ms -> about 30 level events per second

log = logging.getLogger("jarvis.tts")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental synthesis."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class SentenceBuffer:
    """Turns streamed text deltas into complete sentences (a sentence ends at .!? plus whitespace)."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        parts = _SENTENCE_END.split(self._buf)
        if len(parts) < 2:
            return []
        self._buf = parts[-1]
        return [re.sub(r"\s+", " ", p).strip() for p in parts[:-1] if p.strip()]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return split_sentences(rest)

    def discard(self) -> None:
        self._buf = ""


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


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

    async def speak(self, text: str) -> None:
        for sentence in split_sentences(text):
            self.say_sentence(sentence)

    def start_stream(self) -> ConsoleStream:
        return ConsoleStream(self)

    def stop(self) -> None:
        pass


class KokoroStream:
    """Incremental utterance: `push(sentence)` as text arrives, `await finish()` when done.

    Synthesis of sentence N+1 overlaps playback of sentence N (see KokoroSpeaker._speak_blocking).
    """

    def __init__(self, speaker: "KokoroSpeaker") -> None:
        self._speaker = speaker
        self._q: queue.Queue[str | None] = queue.Queue()
        self.first_audio_at: float | None = None
        self.pushed = 0
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        sp = self._speaker
        async with sp._lock:
            sp._stop.clear()
            await asyncio.to_thread(sp._speak_blocking, self._q, self)

    def push(self, sentence: str) -> None:
        if sentence.strip():
            self.pushed += 1
            self._q.put_nowait(sentence)

    async def finish(self) -> None:
        self._q.put_nowait(None)
        await self._task


class KokoroSpeaker:
    def __init__(self, bus: Any, voice: str = "bm_george", lang_code: str = "b",
                 speed: float = 1.0, device: str = "") -> None:
        self.bus = bus
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.device = parse_device(device)
        self._pipeline: Any = None
        self._stop = threading.Event()
        self._lock = asyncio.Lock()  # one utterance at a time

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

    def synth(self, sentence: str) -> np.ndarray:
        """Blocking synthesis of one sentence to float32 24 kHz mono."""
        self.load()
        parts: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipeline(sentence, voice=self.voice, speed=self.speed):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def warm_up(self) -> None:
        """Synthesize a short phrase so kernels are compiled before the first real reply."""
        self.synth("Ready.")

    def stop(self) -> None:
        """Interrupt current speech (thread-safe)."""
        self._stop.set()

    async def speak(self, text: str) -> None:
        sentences = split_sentences(text)
        if not sentences:
            return
        stream = self.start_stream()
        for s in sentences:
            stream.push(s)
        await stream.finish()

    def start_stream(self) -> KokoroStream:
        """Begin an utterance fed sentence by sentence. Must be called inside the event loop."""
        return KokoroStream(self)

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
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        try:
            with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self.device) as out:
                while not self._stop.is_set():
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
                    for i in range(0, len(audio), PLAY_BLOCK):
                        if self._stop.is_set():
                            break
                        block = audio[i:i + PLAY_BLOCK]
                        self.bus.emit_nowait("level", rms=display_level(rms_of(block), gain=4.0))
                        out.write(block.reshape(-1, 1))
        finally:
            self._stop.set()  # lets the producer exit if we left early
            t.join(timeout=2)
            self._stop.clear()
            self.bus.emit_nowait("level", rms=0.0)
