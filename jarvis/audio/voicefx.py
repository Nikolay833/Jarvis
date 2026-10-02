"""Subtle "AI butler" processing for a mono float32 24 kHz clip. Pure numpy; scipy optional.

Chain: high-pass 90 Hz, presence boost 3-5 kHz, tiny room (early reflections), faint modulated comb
(digital sheen), soft limiter, loudness normalisation. Stateless per clip; a short tail is appended so
the reflections are not cut off. Without scipy the filters fall back to FFT shaping.
"""

from __future__ import annotations

import numpy as np

try:  # optional
    from scipy import signal as _signal
except Exception:  # noqa: BLE001
    _signal = None

SAMPLE_RATE = 24000
TAIL_SECONDS = 0.020
TARGET_RMS = 0.10  # about -20 dBFS
PEAK_CEILING = 0.97


def extra_samples(fx: str, amount: float, sr: int = SAMPLE_RATE) -> int:
    return 0 if _is_off(fx, amount) else int(TAIL_SECONDS * sr)


def _is_off(fx: str, amount: float) -> bool:
    return str(fx).strip().lower() in ("", "none", "off") or amount <= 0.0


def _filters(x: np.ndarray, amount: float, sr: int) -> np.ndarray:
    gain_db = 4.0 * amount  # presence boost
    if _signal is not None:
        hp = _signal.butter(2, 90.0 / (sr / 2), btype="highpass", output="sos")
        x = _signal.sosfilt(hp, x)
        bp = _signal.butter(2, [3000.0 / (sr / 2), 5000.0 / (sr / 2)], btype="bandpass", output="sos")
        band = _signal.sosfilt(bp, x)
        return x + band * (10 ** (gain_db / 20) - 1.0)
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    h = 1.0 / np.sqrt(1.0 + (90.0 / np.maximum(f, 1e-3)) ** 4)
    pres = np.exp(-0.5 * ((f - 4000.0) / 1000.0) ** 2) * (10 ** (gain_db / 20) - 1.0)
    return np.fft.irfft(spec * h * (1.0 + pres), n).astype(np.float32)


def _room(x: np.ndarray, amount: float, sr: int) -> np.ndarray:
    y = x.copy()
    for ms, g in ((13.0, 0.10), (21.0, 0.06)):
        d = int(ms * sr / 1000)
        y[d:] += g * amount * x[: len(x) - d]
    return y


def _sheen(x: np.ndarray, amount: float, sr: int) -> np.ndarray:
    """Comb with 0.3-0.6 ms delay swept by a slow LFO (linear interpolation)."""
    n = len(x)
    t = np.arange(n, dtype=np.float32) / sr
    delay = (0.45 + 0.15 * np.sin(2 * np.pi * 0.7 * t)) * 1e-3 * sr  # samples
    pos = np.arange(n, dtype=np.float32) - delay
    pos = np.clip(pos, 0, n - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    fr = pos - i0
    delayed = x[i0] * (1 - fr) + x[i1] * fr
    return x + 0.22 * amount * delayed


def _soft_limit(x: np.ndarray) -> np.ndarray:
    return (PEAK_CEILING * np.tanh(x / PEAK_CEILING)).astype(np.float32)


def apply_fx(audio: np.ndarray, fx: str = "jarvis", amount: float = 0.35, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return processed copy of `audio` (+20 ms tail), or `audio` itself unchanged when fx is off."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    amount = float(min(max(amount, 0.0), 1.0))
    if _is_off(fx, amount) or x.size == 0:
        return audio
    in_rms = float(np.sqrt(np.mean(x * x)))
    x = np.concatenate([x, np.zeros(int(TAIL_SECONDS * sr), dtype=np.float32)])
    y = _filters(x, amount, sr)
    y = _room(y, amount, sr)
    y = _sheen(y, amount, sr)
    out_rms = float(np.sqrt(np.mean(y * y)))
    if out_rms > 1e-9 and in_rms > 1e-9:
        # keep the loudness the voice had, nudged toward the target by amount
        want = in_rms * (TARGET_RMS / in_rms) ** (0.5 * amount)
        y = y * (want / out_rms)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return _soft_limit(y).astype(np.float32)
