"""Speech to text. Pluggable engines behind one interface: load(), transcribe(audio), warm_up().

Engines: "whisper" (faster-whisper, the default) and "parakeet" (NVIDIA Parakeet TDT 0.6B via
the optional `onnx-asr` package). `make_transcriber(cfg)` picks one from `[stt] engine`.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.stt")

ENGINES = ("whisper", "parakeet")
PARAKEET_INSTALL = 'pip install "onnx-asr[gpu,hub]"'


class Transcriber:
    """faster-whisper engine. Positional args kept compatible with the old signature."""

    def __init__(self, model: str = "large-v3", device: str = "cuda",
                 compute_type: str = "float16", fallback_model: str = "small",
                 language: str = "en", vocabulary: list[str] | None = None, *,
                 beam_size: int = 5, best_of: int = 5,
                 temperature: tuple[float, ...] | list[float] | float = (0.0, 0.2, 0.4),
                 vad_filter: bool = True, vad_min_silence_ms: int = 500, vad_speech_pad_ms: int = 300,
                 condition_on_previous_text: bool = False, no_speech_threshold: float = 0.6,
                 log_prob_threshold: float = -1.0, use_hotwords: bool = True) -> None:
        self.model_name = model
        # Names Whisper should expect; it otherwise hears "Claude" as "cloud" or "Clyde".
        self.prompt = ", ".join(vocabulary or []) + "." if vocabulary else None
        self.hotwords = ", ".join(vocabulary) if vocabulary and use_hotwords else None
        self.device = device
        self.compute_type = compute_type
        self.fallback_model = fallback_model
        self.language = language or None
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = tuple(temperature) if isinstance(temperature, (list, tuple)) else temperature
        self.vad_filter = vad_filter
        self.vad_parameters = {"min_silence_duration_ms": vad_min_silence_ms, "speech_pad_ms": vad_speech_pad_ms}
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            log.info("whisper %s on %s (beam %d)", self.model_name, self.device, self.beam_size)
        except Exception as exc:  # noqa: BLE001 - CUDA/cuDNN missing etc.
            log.warning("whisper load on %s failed (%s); falling back to CPU %s",
                        self.device, exc, self.fallback_model)
            self._model = WhisperModel(self.fallback_model, device="cpu", compute_type="int8")

    def transcribe_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for WhisperModel.transcribe; hotwords only if this version has it."""
        kw: dict[str, Any] = {
            "language": self.language, "beam_size": self.beam_size, "best_of": self.best_of,
            "temperature": self.temperature, "vad_filter": self.vad_filter,
            "vad_parameters": dict(self.vad_parameters),
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self.prompt, "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
        }
        if self.hotwords and self._supports("hotwords"):
            kw["hotwords"] = self.hotwords
        return kw

    def _supports(self, name: str) -> bool:
        try:
            return name in inspect.signature(self._model.transcribe).parameters
        except (TypeError, ValueError):
            return False

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz. Blocking."""
        if audio.size == 0:
            return ""
        self.load()
        segments, _info = self._model.transcribe(audio, **self.transcribe_kwargs())
        return " ".join(s.text.strip() for s in segments).strip()

    def warm_up(self) -> None:
        """Transcribe 1 s of silence without VAD so CUDA kernels are compiled up front."""
        self.load()
        segments, _info = self._model.transcribe(np.zeros(16000, dtype=np.float32), language=self.language,
                                                 beam_size=self.beam_size, vad_filter=False)
        for _ in segments:
            pass

    def unload(self) -> None:
        self._model = None
        free_gpu_memory()


class ParakeetTranscriber:
    """NVIDIA Parakeet TDT 0.6B through onnx-asr. No prompt or vocabulary: fix_names corrects names."""

    def __init__(self, model: str = "nemo-parakeet-tdt-0.6b-v2", device: str = "cuda") -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError(
                f"The parakeet STT engine needs the onnx-asr package. Install it with: {PARAKEET_INSTALL}"
            ) from exc
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda"
                     else ["CPUExecutionProvider"])
        try:
            self._model = onnx_asr.load_model(self.model_name, providers=providers)
        except Exception as exc:  # noqa: BLE001 - GPU provider problems
            if self.device != "cuda":
                raise
            log.warning("parakeet load with CUDA failed (%s); retrying on CPU", exc)
            self._model = onnx_asr.load_model(self.model_name, providers=["CPUExecutionProvider"])
        log.info("parakeet %s (%s)", self.model_name, self.device)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz. Blocking."""
        if audio.size == 0:
            return ""
        self.load()
        return str(self._model.recognize(np.asarray(audio, dtype=np.float32), sample_rate=16000)).strip()

    def warm_up(self) -> None:
        self.load()
        self._model.recognize(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def unload(self) -> None:
        self._model = None
        free_gpu_memory()


def free_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch optional
        pass


def whisper_from_config(w: Any) -> Transcriber:
    return Transcriber(
        w.model, w.device, w.compute_type, w.fallback_model, w.language, w.vocabulary,
        beam_size=w.beam_size, best_of=w.best_of, temperature=w.temperature, vad_filter=w.vad_filter,
        vad_min_silence_ms=w.vad_min_silence_ms, vad_speech_pad_ms=w.vad_speech_pad_ms,
        condition_on_previous_text=w.condition_on_previous_text, no_speech_threshold=w.no_speech_threshold,
        log_prob_threshold=w.log_prob_threshold, use_hotwords=w.use_hotwords)


def make_transcriber(cfg: Any) -> Transcriber | ParakeetTranscriber:
    """Engine from `[stt] engine` ("whisper" or "parakeet")."""
    engine = (cfg.stt.engine or "whisper").strip().lower()
    if engine == "parakeet":
        import importlib.util

        if importlib.util.find_spec("onnx_asr") is not None:
            return ParakeetTranscriber(cfg.parakeet.model, cfg.parakeet.device)
        log.warning('stt engine "parakeet" needs: pip install "onnx-asr[gpu,hub]"; using whisper for now')
    if engine != "whisper":
        log.warning("unknown stt engine '%s'; using whisper", engine)
    return whisper_from_config(cfg.whisper)
