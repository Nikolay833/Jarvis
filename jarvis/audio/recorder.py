"""Record speech after the wake word until about one second of silence."""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from .wakeword import parse_device

SAMPLE_RATE = 16000
BLOCK = 512  # 32 ms -> about 31 level events per second


def rms_of(block: np.ndarray) -> float:
    """RMS of an int16 or float32 block, normalised to 0..1."""
    if block.size == 0:
        return 0.0
    x = block.astype(np.float32)
    if block.dtype == np.int16:
        x = x / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def display_level(rms: float, gain: float = 6.0) -> float:
    """Scale raw RMS to a 0..1 value for the orb."""
    return float(min(1.0, max(0.0, rms * gain)))


class EndpointDetector:
    """Energy based end-of-speech detection. Pure; fed one RMS value per block."""

    def __init__(self, silence_seconds: float = 1.0, max_seconds: float = 20.0,
                 no_speech_timeout: float = 6.0, min_threshold: float = 0.012,
                 calibration_seconds: float = 0.25, speech_ratio: float = 3.0) -> None:
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.no_speech_timeout = no_speech_timeout
        self.min_threshold = min_threshold
        self.calibration_seconds = calibration_seconds
        self.speech_ratio = speech_ratio
        self.elapsed = 0.0
        self.silence = 0.0
        self.speech_started = False
        self.reason = ""
        self._noise: list[float] = []
        self._noise_floor = 0.0

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self._noise_floor * self.speech_ratio)

    def feed(self, rms: float, dt: float) -> bool:
        """Returns True when recording should stop (see `reason`)."""
        self.elapsed += dt
        if self.elapsed <= self.calibration_seconds and not self.speech_started:
            self._noise.append(rms)
            self._noise_floor = min(0.02, sum(self._noise) / len(self._noise))
        elif rms >= self.threshold:
            self.speech_started = True
            self.silence = 0.0
        elif self.speech_started:
            self.silence += dt
        if self.elapsed >= self.max_seconds:
            self.reason = "max"
        elif self.speech_started and self.silence >= self.silence_seconds:
            self.reason = "silence"
        elif not self.speech_started and self.elapsed >= self.no_speech_timeout:
            self.reason = "no_speech"
        return bool(self.reason)


class Recorder:
    def __init__(self, silence_seconds: float = 1.0, max_seconds: float = 20.0,
                 no_speech_timeout: float = 6.0, device: str = "") -> None:
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.no_speech_timeout = no_speech_timeout
        self.device = parse_device(device)

    def record_blocking(self, on_level: Callable[[float], None] | None = None,
                        cancel: threading.Event | None = None) -> np.ndarray:
        """Record one utterance. Returns float32 mono 16 kHz; empty array if nobody spoke."""
        import sounddevice as sd

        det = EndpointDetector(self.silence_seconds, self.max_seconds, self.no_speech_timeout)
        chunks: list[np.ndarray] = []
        dt = BLOCK / SAMPLE_RATE
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=BLOCK, device=self.device) as stream:
            while not (cancel and cancel.is_set()):
                data, _ = stream.read(BLOCK)
                block: Any = data[:, 0]
                chunks.append(block.copy())
                level = rms_of(block)
                if on_level:
                    on_level(display_level(level))
                if det.feed(level, dt):
                    break
        if not det.speech_started or not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0
