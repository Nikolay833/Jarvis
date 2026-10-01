"""Wake word detection with openWakeWord ("hey_jarvis")."""

from __future__ import annotations

import logging
import threading
from typing import Any

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms, what openWakeWord expects

log = logging.getLogger("jarvis.wakeword")


def parse_device(value: str) -> int | str | None:
    """Config device string -> sounddevice device (index, name or default)."""
    value = (value or "").strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5, device: str = "") -> None:
        self.model_name = model
        self.threshold = threshold
        self.device = parse_device(device)
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
        """Listen on the mic until the wake word (True) or `stop` is set (False). Runs in a thread."""
        import sounddevice as sd

        self.load()
        self.reset()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME, device=self.device) as stream:
            while not stop.is_set():
                data, _overflow = stream.read(FRAME)
                if self.score(data[:, 0]) >= self.threshold:
                    log.info("wake word detected")
                    return True
        return False
