"""Text to speech: Kokoro synthesis with sentence-by-sentence playback."""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from typing import Any

import numpy as np

from .recorder import display_level, rms_of
from .wakeword import parse_device

SAMPLE_RATE = 24000
PLAY_BLOCK = 800  # 33 ms -> about 30 level events per second

log = logging.getLogger("jarvis.tts")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental synthesis."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class ConsoleSpeaker:
    """No-audio speaker: prints each sentence and emits `reply` events."""

    def __init__(self, bus: Any, prefix: str = "Jarvis: ") -> None:
        self.bus = bus
        self.prefix = prefix

    async def speak(self, text: str) -> None:
        for sentence in split_sentences(text):
            self.bus.emit_nowait("reply", text=sentence)
            print(f"{self.prefix}{sentence}", flush=True)

    def stop(self) -> None:
        pass


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

            self._pipeline = KPipeline(lang_code=self.lang_code)

    def synth(self, sentence: str) -> np.ndarray:
        """Blocking synthesis of one sentence to float32 24 kHz mono."""
        self.load()
        parts: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipeline(sentence, voice=self.voice, speed=self.speed):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def stop(self) -> None:
        """Interrupt current speech (thread-safe)."""
        self._stop.set()

    async def speak(self, text: str) -> None:
        sentences = split_sentences(text)
        if not sentences:
            return
        async with self._lock:
            self._stop.clear()
            await asyncio.to_thread(self._speak_blocking, sentences)

    def _speak_blocking(self, sentences: list[str]) -> None:
        import sounddevice as sd

        q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=2)

        def producer() -> None:
            try:
                for s in sentences:
                    if self._stop.is_set():
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
