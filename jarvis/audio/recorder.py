"""Record speech after the wake word until about one second of silence."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .mic import MicStream

SAMPLE_RATE = 16000
BLOCK = 512  # 32 ms -> about 31 level events per second
MAX_AFTER_WAKE_FLOOR = 0.015  # cap for a pre-roll noise estimate (may contain speech)
STALL_SECONDS = 3.0  # give up if the mic delivers nothing for this long

log = logging.getLogger("jarvis.recorder")


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


def estimate_noise_floor(samples: np.ndarray, block: int = BLOCK) -> float:
    """Noise floor from pre-roll audio: 10th percentile of per-block RMS (quiet blocks are noise)."""
    n = len(samples) // block
    if n == 0:
        return 0.0
    levels = [rms_of(samples[i * block:(i + 1) * block]) for i in range(n)]
    return float(np.percentile(levels, 10))


class EndpointDetector:
    """Energy based end-of-speech detection. Pure; fed one RMS value per block.

    Default mode calibrates the noise floor from the first `calibration_seconds`.
    `after_wake=True` is for recording right after the wake word: the user may already be
    speaking, so nothing is calibrated from the first blocks. The threshold is `min_threshold`
    or, when `noise_floor` (e.g. from the pre-roll) is given, noise_floor * speech_ratio.
    `ignore_seconds` blanks the first blocks for speech detection (own chime).
    """

    def __init__(self, silence_seconds: float = 1.0, max_seconds: float = 20.0,
                 no_speech_timeout: float = 6.0, min_threshold: float = 0.012,
                 calibration_seconds: float = 0.25, speech_ratio: float = 3.0,
                 after_wake: bool = False, noise_floor: float | None = None,
                 ignore_seconds: float = 0.0) -> None:
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.no_speech_timeout = no_speech_timeout
        self.min_threshold = min_threshold
        self.calibration_seconds = calibration_seconds
        self.speech_ratio = speech_ratio
        self.after_wake = after_wake
        self.ignore_seconds = ignore_seconds
        self.elapsed = 0.0
        self.silence = 0.0
        self.speech_started = False
        self.reason = ""
        self._noise: list[float] = []
        self._noise_floor = 0.0
        if after_wake and noise_floor is not None:
            self._noise_floor = min(MAX_AFTER_WAKE_FLOOR, max(0.0, noise_floor))

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self._noise_floor * self.speech_ratio)

    def feed(self, rms: float, dt: float) -> bool:
        """Returns True when recording should stop (see `reason`)."""
        self.elapsed += dt
        if not self.after_wake and self.elapsed <= self.calibration_seconds and not self.speech_started:
            self._noise.append(rms)
            self._noise_floor = min(0.02, sum(self._noise) / len(self._noise))
        else:
            if self.elapsed <= self.ignore_seconds:
                rms = 0.0
            if rms >= self.threshold:
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
    def __init__(self, mic: MicStream, silence_seconds: float = 1.0, max_seconds: float = 20.0,
                 no_speech_timeout: float = 6.0) -> None:
        self.mic = mic
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.no_speech_timeout = no_speech_timeout
        self.last_reason = ""
        self.last_seconds = 0.0

    def record_blocking(self, on_level: Callable[[float], None] | None = None,
                        cancel: threading.Event | None = None, flush: bool = False,
                        before: Callable[[], None] | None = None,
                        prefix: np.ndarray | None = None) -> np.ndarray:
        """Record one utterance from the shared mic. Float32 mono 16 kHz; empty if nobody spoke.

        Recording starts with the audio queued right now. After a wake word that is the audio
        immediately following the detection frame, so pass flush=False. Pass flush=True when
        starting cold (hotkey, spoken confirmation) to drop stale queued audio.
        `before` (e.g. the chime) runs first; the mic keeps queueing meanwhile and the energy it
        produced is ignored for endpointing, but the audio is kept in case the user spoke over it.
        `prefix` (int16, barge-in): speech already in progress. It starts the recording and counts as
        speech, so the endpointer only waits for the silence after it.
        """
        floor = estimate_noise_floor(self.mic.preroll())  # taken before flush: ambient noise
        if prefix is not None:
            floor = 0.0  # the pre-roll holds Jarvis's own voice; do not raise the threshold with it
        if flush:
            self.mic.flush()
        ignore = 0.0
        if before is not None:
            t0 = time.monotonic()
            before()
            ignore = time.monotonic() - t0 + 0.05
        det = EndpointDetector(self.silence_seconds, self.max_seconds, self.no_speech_timeout,
                               after_wake=True, noise_floor=floor, ignore_seconds=ignore)
        chunks: list[np.ndarray] = []
        if prefix is not None and len(prefix):
            chunks.append(np.asarray(prefix, dtype=np.int16))
            det.speech_started = True
        dt = BLOCK / SAMPLE_RATE
        stalled = 0.0
        reason = "cancelled"
        while not (cancel and cancel.is_set()):
            block = self.mic.read(BLOCK, timeout=0.2)
            if block is None:
                stalled += 0.2
                if stalled >= STALL_SECONDS:
                    log.warning("no audio from the microphone while recording")
                    reason = "mic_stalled"
                    break
                continue
            stalled = 0.0
            chunks.append(block)
            level = rms_of(block)
            if on_level:
                on_level(display_level(level))
            if det.feed(level, dt):
                reason = det.reason
                break
        self.last_reason = reason
        self.last_seconds = det.elapsed
        log.debug("recording ended (%s) after %.1f s, noise floor %.4f, threshold %.4f",
                  reason, det.elapsed, floor, det.threshold)
        if not det.speech_started or not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0
