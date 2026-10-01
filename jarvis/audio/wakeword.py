"""Wake word detection with openWakeWord ("hey_jarvis")."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .mic import MicStream, parse_device  # noqa: F401 - parse_device re-exported for older imports

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms, what openWakeWord expects
NO_AUDIO_WARN_SECONDS = 3.0
DEBUG_INTERVAL = 0.5

log = logging.getLogger("jarvis.wakeword")


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5, mic: MicStream | None = None,
                 debug: bool = False) -> None:
        self.model_name = model
        self.threshold = threshold
        self.mic = mic
        self.debug = debug  # print mic level and wake score twice a second
        self.last_score = 0.0
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import openwakeword
        from openwakeword.model import Model

        try:
            self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except Exception:  # noqa: BLE001 - models not downloaded yet
            openwakeword.utils.download_models([self.model_name])
            self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")

    def score(self, frame_int16: Any) -> float:
        """Feed one 1280-sample int16 frame; return the best wake score."""
        self.load()
        preds = self._model.predict(frame_int16)
        return float(max(preds.values())) if preds else 0.0

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def wait_blocking(self, stop: threading.Event) -> bool:
        """Listen on the shared mic until the wake word (True) or `stop` is set (False). Runs in a thread.

        Reading stops right after the detection frame, so the audio that follows is still queued
        in the mic for the recorder.
        """
        assert self.mic is not None, "WakeWordDetector needs a MicStream"
        self.load()
        self.reset()
        warned = False
        window_start = time.monotonic()
        peak_rms = peak_score = 0.0
        while not stop.is_set():
            frame = self.mic.read(FRAME, timeout=0.1)
            if frame is None:
                if not warned and self.mic.seconds_since_audio() > NO_AUDIO_WARN_SECONDS:
                    log.warning("no audio from the microphone for %.0f s; check Windows sound settings "
                                "or audio.input_device", NO_AUDIO_WARN_SECONDS)
                    warned = True
                continue
            warned = False
            score = self.score(frame)
            if self.debug:
                x = frame.astype(np.float32) / 32768.0
                peak_rms = max(peak_rms, float(np.sqrt(np.mean(x * x))))
                peak_score = max(peak_score, score)
                if time.monotonic() - window_start >= DEBUG_INTERVAL:
                    bar = "#" * min(40, int(peak_rms * 200))
                    print(f"[debug-audio] mic rms {peak_rms:.4f} |{bar:<40}| wake score {peak_score:.2f} "
                          f"(threshold {self.threshold:.2f})", flush=True)
                    window_start = time.monotonic()
                    peak_rms = peak_score = 0.0
            if score >= self.threshold:
                self.last_score = score
                log.info("wake word detected (score %.2f)", score)
                return True
        return False
