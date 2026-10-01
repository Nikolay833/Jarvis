"""Speech to text with faster-whisper."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.stt")


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "float16", fallback_model: str = "small",
                 language: str = "en") -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.fallback_model = fallback_model
        self.language = language or None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            log.info("whisper %s on %s", self.model_name, self.device)
        except Exception as exc:  # noqa: BLE001 - CUDA/cuDNN missing etc.
            log.warning("whisper load on %s failed (%s); falling back to CPU %s",
                        self.device, exc, self.fallback_model)
            self._model = WhisperModel(self.fallback_model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz. Blocking."""
        if audio.size == 0:
            return ""
        self.load()
        segments, _info = self._model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
