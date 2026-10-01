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
        self._in_rate = sample_rate
        self._resample_pos = 0.0
        self._resample_tail = np.zeros(0, dtype=np.float32)
        self.last_block_time = 0.0
        self.overflows = 0

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Open the microphone, trying fallbacks until one works.

        Windows drivers often refuse 16 kHz mono with misleading errors such as
        "Insufficient memory [-9992]". So we try, in order: the device at 16 kHz,
        the same device at its native rate (resampled here), the same device under
        other host APIs (WASAPI, DirectSound, MME), then every other input device.
        """
        if self._stream is not None:
            return
        import sounddevice as sd

        errors: list[str] = []
        for dev, rate, label in self._candidates(sd):
            try:
                stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                                        blocksize=int(self.block * rate / self.sample_rate),
                                        device=dev, callback=self._callback)
                stream.start()
            except Exception as exc:  # noqa: BLE001 - PortAudio errors vary by driver
                errors.append(f"{label} @ {rate} Hz: {exc}")
                log.debug("mic candidate failed: %s @ %s Hz: %s", label, rate, exc)
                continue
            self._stream = stream
            self._in_rate = int(rate)
            self._resample_pos = 0.0
            self._resample_tail = np.zeros(0, dtype=np.float32)
            self.last_block_time = time.monotonic()
            log.info("microphone open: %s at %d Hz%s", label, int(rate),
                     "" if int(rate) == self.sample_rate else f" (resampled to {self.sample_rate} Hz)")
            return
        detail = "\n  ".join(errors[-12:]) or "no input devices found"
        raise RuntimeError("could not open any microphone. Tried:\n  " + detail
                           + "\nRun `python -m jarvis --list-devices` and set audio.input_device in config.toml.")

    def _candidates(self, sd: Any) -> list[tuple[Any, float, str]]:
        """(device, samplerate, label) to try, best first."""
        try:
            devices = list(sd.query_devices())
            apis = list(sd.query_hostapis())
        except Exception:  # noqa: BLE001
            return [(self.device, self.sample_rate, "default")]

        def label(i: int) -> str:
            d = devices[i]
            return f"#{i} {d['name']} [{apis[d['hostapi']]['name']}]"

        inputs = [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
        if isinstance(self.device, int):
            first = self.device
        elif isinstance(self.device, str):
            first = next((i for i in inputs if self.device.lower() in devices[i]["name"].lower()), None)
        else:
            try:
                first = int(sd.default.device[0])
            except Exception:  # noqa: BLE001
                first = None
            if first is not None and first < 0:
                first = None
        # Same physical mic under other host APIs (names are truncated differently, so match a prefix).
        api_rank = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2, "Windows WDM-KS": 3}
        order: list[int] = []
        if first is not None and first in inputs:
            order.append(first)
            stem = devices[first]["name"][:20].lower()
            same = [i for i in inputs if i != first and devices[i]["name"][:20].lower() == stem]
            order += sorted(same, key=lambda i: api_rank.get(apis[devices[i]["hostapi"]]["name"], 9))
        rest = [i for i in inputs if i not in order]
        order += sorted(rest, key=lambda i: api_rank.get(apis[devices[i]["hostapi"]]["name"], 9))

        out: list[tuple[Any, float, str]] = []
        for i in order:
            native = float(devices[i].get("default_samplerate") or 48000)
            for rate in dict.fromkeys([float(self.sample_rate), native, 48000.0, 44100.0]):
                out.append((i, rate, label(i)))
        return out or [(None, float(self.sample_rate), "default")]
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
        block = indata[:, 0]
        if self._in_rate != self.sample_rate:
            block = self._resample(block)
            if block.size == 0:
                return
        self.feed(block)

    def _resample(self, block: Any) -> np.ndarray:
        """Linear-interpolation resample of int16 input to self.sample_rate, continuous across blocks."""
        x = np.concatenate((self._resample_tail, np.asarray(block, dtype=np.float32)))
        step = self._in_rate / self.sample_rate
        pos = np.arange(self._resample_pos, len(x) - 1, step)
        if pos.size == 0:
            self._resample_tail = x
            return np.zeros(0, dtype=np.int16)
        idx = pos.astype(np.int64)
        frac = pos - idx
        y = x[idx] * (1.0 - frac) + x[idx + 1] * frac
        nxt = pos[-1] + step
        keep = int(nxt)
        self._resample_tail = x[keep:]
        self._resample_pos = nxt - keep
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

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


def list_devices() -> str:
    """Human-readable list of audio devices, for `python -m jarvis --list-devices`."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    lines = []
    for i, d in enumerate(sd.query_devices()):
        kind = "/".join(k for k, n in (("in", d["max_input_channels"]), ("out", d["max_output_channels"])) if n > 0)
        lines.append(f"#{i:<3} {kind:<6} {int(d['default_samplerate']):>6} Hz  {d['name']}  [{apis[d['hostapi']]['name']}]")
    try:
        lines.append(f"default input #{sd.default.device[0]}, default output #{sd.default.device[1]}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)
