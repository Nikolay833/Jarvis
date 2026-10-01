"""One persistent microphone stream shared by wake word detection and recording.

The stream is opened once at startup. The PortAudio callback pushes int16 blocks into a
thread-safe queue and a ~1.5 s pre-roll ring buffer. Consumers (wake word, recorder) read
from the queue one after the other, so no audio is lost between them.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 512  # callback block size (32 ms)

log = logging.getLogger("jarvis.mic")


def parse_device(value: str) -> int | str | None:
    """Config device string -> sounddevice device (index, name or default)."""
    value = (value or "").strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


class MicStream:
    def __init__(self, device: str = "", sample_rate: int = SAMPLE_RATE,
                 preroll_seconds: float = 1.5, block: int = BLOCK) -> None:
        self.device = parse_device(device)
        self.sample_rate = sample_rate
        self.block = block
        self._max_ring = int(preroll_seconds * sample_rate)
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._pending = np.zeros(0, dtype=np.int16)  # leftover samples of a partly consumed block
        self._read_lock = threading.Lock()
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._ring_lock = threading.Lock()
        self._stream: Any = None
        self.last_block_time = 0.0
        self.overflows = 0

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                                blocksize=self.block, device=self.device, callback=self._callback)
        stream.start()
        self._stream = stream
        self.last_block_time = time.monotonic()
        log.info("microphone open (device %s, %d Hz)", self.device if self.device is not None else "default",
                 self.sample_rate)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                log.debug("closing microphone failed", exc_info=True)

    @property
    def running(self) -> bool:
        return self._stream is not None

    # ---- producer side ---------------------------------------------------
    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:  # PortAudio thread
        if status:
            self.overflows += 1
        self.feed(indata[:, 0])

    def feed(self, block: Any) -> None:
        """Push one block of int16 mono samples (called by the callback; tests call it directly)."""
        data = np.array(block, dtype=np.int16, copy=True).reshape(-1)
        self.last_block_time = time.monotonic()
        with self._ring_lock:
            self._ring.append(data)
            self._ring_samples += len(data)
            while len(self._ring) > 1 and self._ring_samples - len(self._ring[0]) >= self._max_ring:
                self._ring_samples -= len(self._ring.popleft())
        self._q.put(data)

    # ---- consumer side ---------------------------------------------------
    def read(self, n: int, timeout: float = 0.1) -> np.ndarray | None:
        """Exactly `n` int16 samples, continuing where the previous read stopped; None on timeout."""
        deadline = time.monotonic() + timeout
        with self._read_lock:
            while len(self._pending) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    blk = self._q.get(timeout=remaining)
                except queue.Empty:
                    return None
                self._pending = np.concatenate((self._pending, blk))
            out = self._pending[:n].copy()
            self._pending = self._pending[n:]
            return out

    def flush(self) -> float:
        """Drop all queued audio (stale TTS echo etc.). Returns seconds dropped."""
        with self._read_lock:
            dropped = len(self._pending)
            self._pending = np.zeros(0, dtype=np.int16)
            while True:
                try:
                    dropped += len(self._q.get_nowait())
                except queue.Empty:
                    break
        return dropped / self.sample_rate

    def preroll(self) -> np.ndarray:
        """Copy of the most recent ~1.5 s of audio (int16)."""
        with self._ring_lock:
            if not self._ring:
                return np.zeros(0, dtype=np.int16)
            return np.concatenate(list(self._ring))

    def seconds_since_audio(self) -> float:
        return time.monotonic() - self.last_block_time
