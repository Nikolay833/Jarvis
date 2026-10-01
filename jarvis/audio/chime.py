"""Short soft two-tone chime that tells the user Jarvis is listening."""

from __future__ import annotations

import logging

import numpy as np

SAMPLE_RATE = 24000

log = logging.getLogger("jarvis.chime")


def make_chime(sample_rate: int = SAMPLE_RATE, volume: float = 0.12) -> np.ndarray:
    """Two rising tones (660 Hz, 880 Hz), 60 ms each, soft edges. float32 mono, about 120 ms."""
    parts = []
    for freq in (660.0, 880.0):
        n = int(0.06 * sample_rate)
        t = np.arange(n) / sample_rate
        env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.008)  # 8 ms fade in and out
        parts.append(np.sin(2 * np.pi * freq * t) * env)
    return (np.concatenate(parts) * volume).astype(np.float32)


def play_chime(device: int | str | None = None, blocking: bool = True) -> None:
    """Play the chime on the output device. Never raises."""
    try:
        import sounddevice as sd

        sd.play(make_chime(), SAMPLE_RATE, device=device)
        if blocking:
            sd.wait()
    except Exception:  # noqa: BLE001 - no audio output must not break a turn
        log.debug("chime failed", exc_info=True)
