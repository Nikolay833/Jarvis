"""Barge-in: let the user talk over Jarvis.

While Jarvis speaks, a `BargeInMonitor` listens to the shared mic (through `MicStream.subscribe`, so it never
steals audio from the wake word or recorder) and decides whether the *user* is talking, not Jarvis's own voice
coming out of the speakers into the mic (no headset assumed).

Decision (`EchoGate`, pure and unit tested): a mic frame counts as user speech when
  * the VAD probability is high (Silero via onnxruntime, else an energy fallback), AND
  * the mic RMS exceeds the expected echo by a margin. Expected echo = coupling * max playback RMS over the last
    ~150 ms (the TTS knows exactly what it plays, see `PlaybackTap`), coupling = mic RMS / playback RMS learned
    from non-speech frames (`EchoCalibrator`, so the startup "Online, sir." calibrates it),
and this holds for `min_ms` continuously (short gaps tolerated). The wake word "Hey Jarvis" during playback
always counts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms, a frame size Silero accepts (verified: 480 and 512 work, 160 fails)
WAKE_FRAME = 1280  # 80 ms, what openWakeWord expects
PREROLL_SECONDS = 0.3  # audio kept from before the speech was first detected
MIN_PLAY_RMS = 0.02  # playback quieter than this teaches nothing about the echo path
DEFAULT_COUPLING = 0.8  # conservative until calibrated (mic RMS / playback RMS)
COUPLING_RANGE = (0.02, 2.0)
SPEECH_FLOOR_RMS = 0.015  # mic RMS below this is never speech, whatever the echo estimate says

log = logging.getLogger("jarvis.bargein")


def rms(block: np.ndarray) -> float:
    """RMS of an int16 (normalised to 0..1) or float block."""
    if block.size == 0:
        return 0.0
    x = block.astype(np.float32)
    if block.dtype == np.int16:
        x = x / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def gate_params(sensitivity: float, min_ms: float = 250.0) -> dict[str, float]:
    """Sensitivity 0..1 (0.5 default) -> echo margin, duration and VAD threshold.

    0 = hard to trigger (margin 3.5x, 1.5x min_ms, vad 0.75); 0.5 = margin 2.5x, min_ms, vad 0.60;
    1 = easy (margin 1.5x, 0.5x min_ms, vad 0.45).
    """
    s = min(1.0, max(0.0, float(sensitivity)))
    return {"margin": 3.5 - 2.0 * s, "min_ms": float(min_ms) * (1.5 - s), "vad_threshold": 0.75 - 0.3 * s}


class EchoCalibrator:
    """Learns the coupling factor mic RMS / playback RMS from frames without user speech.

    Keeps the last `keep` ratios; the estimate is their 90th percentile (the echo path is bursty, and
    underestimating it makes Jarvis interrupt himself), clamped to COUPLING_RANGE. None until `min_samples`.
    """

    def __init__(self, keep: int = 150, min_samples: int = 10, percentile: float = 90.0) -> None:
        self._ratios: deque[float] = deque(maxlen=keep)
        self.min_samples = min_samples
        self.percentile = percentile

    def add(self, mic_rms: float, play_rms: float) -> None:
        if play_rms >= MIN_PLAY_RMS:
            self._ratios.append(mic_rms / play_rms)

    @property
    def samples(self) -> int:
        return len(self._ratios)

    def value(self) -> float | None:
        if len(self._ratios) < self.min_samples:
            return None
        v = float(np.percentile(list(self._ratios), self.percentile))
        return min(COUPLING_RANGE[1], max(COUPLING_RANGE[0], v))


class EchoGate:
    """Pure barge-in decision. Feed one mic frame at a time; `feed` returns True once when the user barges in."""

    def __init__(self, margin: float = 2.5, min_ms: float = 250.0, vad_threshold: float = 0.6,
                 max_gap_ms: float = 100.0, min_rms: float = SPEECH_FLOOR_RMS, window_ms: float = 150.0,
                 coupling: float = DEFAULT_COUPLING, calibrator: EchoCalibrator | None = None) -> None:
        self.margin = margin
        self.min_s = min_ms / 1000.0
        self.vad_threshold = vad_threshold
        self.max_gap_s = max_gap_ms / 1000.0
        self.min_rms = min_rms
        self.window_s = window_ms / 1000.0
        self.default_coupling = coupling
        self.calibrator = calibrator if calibrator is not None else EchoCalibrator()
        self._play: deque[tuple[float, float]] = deque()
        self._t = 0.0
        self.reset()

    @classmethod
    def from_sensitivity(cls, sensitivity: float = 0.5, min_ms: float = 250.0, **kw: Any) -> "EchoGate":
        p = gate_params(sensitivity, min_ms)
        return cls(margin=p["margin"], min_ms=p["min_ms"], vad_threshold=p["vad_threshold"], **kw)

    @property
    def coupling(self) -> float:
        v = self.calibrator.value()
        return self.default_coupling if v is None else v

    def reset(self) -> None:
        """Forget the current candidate (not the learned coupling)."""
        self._play.clear()
        self._t = 0.0
        self._active = 0.0  # seconds of speech-like frames in the current run
        self._gap = 0.0
        self.run_seconds = 0.0  # since the run's first speech-like frame, gaps included
        self.peak_vad = 0.0
        self.peak_ratio = 0.0
        self.ratio = 0.0
        self.expected_echo = 0.0
        self.suppressed = False  # last frame: VAD says speech but the energy is explained by the echo
        self.triggered = False

    def feed(self, mic_rms: float, play_rms: float, vad_prob: float, dt: float) -> bool:
        self._t += dt
        self._play.append((self._t, play_rms))
        while self._play and self._play[0][0] < self._t - self.window_s:
            self._play.popleft()
        window_max = max(v for _, v in self._play)
        voiced = vad_prob >= self.vad_threshold
        if vad_prob < 0.35:  # clearly not speech: what the mic hears now is echo (or noise)
            self.calibrator.add(mic_rms, window_max)
        self.expected_echo = self.coupling * window_max
        need = max(self.margin * self.expected_echo, self.min_rms)
        # >= margin exactly when mic_rms >= need (the echo term is floored at min_rms / margin)
        self.ratio = mic_rms / max(self.expected_echo, self.min_rms / self.margin)
        loud = mic_rms >= need
        self.suppressed = voiced and not loud
        if self.triggered:
            return False
        if voiced and loud:
            self._active += dt
            self._gap = 0.0
            self.run_seconds += dt
            self.peak_vad = max(self.peak_vad, vad_prob)
            self.peak_ratio = max(self.peak_ratio, self.ratio)
        elif self._active > 0.0:
            self._gap += dt
            self.run_seconds += dt
            if self._gap > self.max_gap_s:  # too long a break: a click or a burst of echo, start over
                self._active = self._gap = self.run_seconds = 0.0
                self.peak_vad = self.peak_ratio = 0.0
        if self._active >= self.min_s:
            self.triggered = True
            return True
        return False


class PlaybackTap:
    """What the speaker is playing and when it is audible, so the monitor can estimate the echo.

    `note(block)` is called right before each `OutputStream.write`. Blocks are written ahead of real time
    (the device buffers them), so each one gets an audible time slot: max(now + device latency, end of the
    previous block). `level_at(t)` is the RMS of the block audible at time t (0 if silence).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, keep_seconds: float = 3.0) -> None:
        self.clock = clock
        self.keep = keep_seconds
        self.latency = 0.0
        self._items: deque[tuple[float, float, float]] = deque()  # (start, end, rms)
        self._end = 0.0
        self._lock = threading.Lock()

    def note(self, block: np.ndarray, sample_rate: int) -> None:
        dur = len(block) / sample_rate
        level = rms(np.asarray(block, dtype=np.float32))
        with self._lock:
            start = max(self.clock() + self.latency, self._end)
            self._end = start + dur
            self._items.append((start, self._end, level))
            while self._items and self._items[0][1] < self.clock() - self.keep:
                self._items.popleft()

    def level_at(self, t: float) -> float:
        with self._lock:
            for start, end, level in reversed(self._items):
                if start <= t < end:
                    return level
                if end <= t:
                    break
        return 0.0


class EnergyVAD:
    """Fallback when Silero is unavailable: a frame is speech if clearly above a slowly rising noise floor."""

    def __init__(self) -> None:
        self._floor = 0.01

    def reset(self) -> None:
        self._floor = 0.01

    def prob(self, frame: np.ndarray) -> float:
        level = rms(frame)
        # the floor follows quiet frames down at once and creeps up towards louder ones
        self._floor = max(0.002, level if level < self._floor else min(level, self._floor * 1.01))
        return 0.9 if level >= max(SPEECH_FLOOR_RMS, 3.0 * self._floor) else 0.1


class SileroVAD:
    """Silero VAD through onnxruntime, using the model openWakeWord ships.

    File: <openwakeword>/resources/models/silero_vad.onnx (openWakeWord's `download_models` fetches it; it is
    not inside the wheel). Inputs: input float32 [1, N] (N = 512 at 16 kHz), sr int64 scalar, h and c float32
    [2, 1, 64] LSTM state. Outputs: speech probability [1, 1], new h, new c.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._sess: Any = None
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    @staticmethod
    def find_model(download: bool = True) -> Path | None:
        try:
            import openwakeword

            path = Path(openwakeword.__file__).resolve().parent / "resources" / "models" / "silero_vad.onnx"
            if path.is_file():
                return path
            if download:
                from openwakeword import utils

                url = openwakeword.VAD_MODELS["silero_vad"]["download_url"]
                log.info("downloading silero_vad.onnx for barge-in")
                utils.download_file(url, str(path.parent))
                if path.is_file():
                    return path
        except Exception:  # noqa: BLE001
            log.debug("silero model lookup failed", exc_info=True)
        return None

    def load(self) -> bool:
        if self._sess is not None:
            return True
        try:
            import onnxruntime as ort

            path = self.model_path or self.find_model()
            if path is None:
                return False
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = opts.intra_op_num_threads = 1
            self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            return True
        except Exception:  # noqa: BLE001
            log.debug("silero load failed", exc_info=True)
            return False

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = (frame.astype(np.float32) / 32768.0)[None, :]
        out, self._h, self._c = self._sess.run(None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c})
        return float(out[0][0])


@dataclass
class BargeIn:
    """The user barged in. Positions are absolute MicStream sample positions."""

    onset: int  # where the recording should start (speech start minus the pre-roll)
    trigger: int  # end of the frame that triggered
    reason: str  # "voice" | "wake"
    vad: float
    ratio: float
    at: float  # time.monotonic()


class BargeInMonitor:
    """Runs a thread during playback (`start()` / `stop()` are the speaker's playback hooks)."""

    def __init__(self, mic: Any, tap: PlaybackTap, sensitivity: float = 0.5, min_ms: float = 250.0,
                 wake: Any = None, wake_ok: Callable[[], bool] = lambda: True,
                 on_barge: Callable[[BargeIn], None] | None = None, vad: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.mic = mic
        self.tap = tap
        self.wake = wake  # WakeWordDetector (score(frame), threshold, reset()); None disables the wake path
        self.wake_ok = wake_ok  # False while the wake thread uses the same model
        self.on_barge = on_barge
        self.clock = clock
        self.vad: Any = vad
        self.calibrator = EchoCalibrator()
        self.gate = EchoGate.from_sensitivity(sensitivity, min_ms, calibrator=self.calibrator)
        self._loaded = vad is not None
        self._load_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        """Load Silero (blocking, idempotent). Falls back to the energy VAD."""
        with self._load_lock:
            if self._loaded:
                return
            silero = SileroVAD()
            if silero.load():
                self.vad = silero
                log.info("barge-in VAD: silero (%s)", silero.model_path or "openwakeword resources")
            else:
                self.vad = EnergyVAD()
                log.warning("barge-in: silero_vad.onnx/onnxruntime unavailable, using the energy VAD "
                            "(less robust against Jarvis's own voice)")
            self._loaded = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._loaded:
            self.load()
        self._stop.clear()
        sub = self.mic.subscribe()  # a fresh queue: no stale audio
        self._thread = threading.Thread(target=self._run, args=(sub,), daemon=True, name="barge-in")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    # ---- the loop --------------------------------------------------------
    def _run(self, sub: Any) -> None:
        try:
            self._loop(sub)
        except Exception:  # noqa: BLE001 - never let the monitor break playback
            log.exception("barge-in monitor failed")
        finally:
            self.mic.unsubscribe(sub)

    def _loop(self, sub: Any) -> None:
        import queue as _q

        self.gate.reset()
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        use_wake = self.wake is not None
        if use_wake and self.wake_ok():
            try:
                self.wake.reset()
            except Exception:  # noqa: BLE001
                use_wake = False
        buf = np.zeros(0, dtype=np.int16)
        buf_end = 0  # absolute position of the end of buf
        wake_buf = np.zeros(0, dtype=np.int16)
        t_arrival = self.clock()
        dt = FRAME / SAMPLE_RATE
        last_log = 0.0
        while not self._stop.is_set():
            try:
                data, end, t_arrival = sub.get(timeout=0.1)
            except _q.Empty:
                continue
            buf = np.concatenate((buf, data))
            buf_end = end
            while len(buf) >= FRAME and not self._stop.is_set():
                frame, buf = buf[:FRAME], buf[FRAME:]
                frame_end = buf_end - len(buf)
                # arrival time is the end of the newest block: the frame ended len(buf) samples earlier
                t_frame = t_arrival - len(buf) / SAMPLE_RATE
                prob = self.vad.prob(frame)
                play = self.tap.level_at(t_frame)
                fired = self.gate.feed(rms(frame), play, prob, dt)
                g = self.gate
                if g.suppressed and t_frame - last_log > 0.5:
                    last_log = t_frame
                    log.debug("barge-in suppressed as echo (vad %.2f, mic/echo ratio %.1f, coupling %.2f)",
                              prob, g.ratio, g.coupling)
                if fired:
                    onset = frame_end - int(g.run_seconds * SAMPLE_RATE) - int(PREROLL_SECONDS * SAMPLE_RATE)
                    self._fire(BargeIn(onset, frame_end, "voice", g.peak_vad, g.peak_ratio, self.clock()))
                    return
                if use_wake and self.wake_ok():
                    wake_buf = np.concatenate((wake_buf, frame))
                    if len(wake_buf) >= WAKE_FRAME:
                        chunk, wake_buf = wake_buf[:WAKE_FRAME], wake_buf[WAKE_FRAME:]
                        try:
                            score = float(self.wake.score(chunk))
                        except Exception:  # noqa: BLE001
                            log.debug("wake score failed during playback", exc_info=True)
                            use_wake = False
                            continue
                        if score >= self.wake.threshold:
                            onset = frame_end - int(PREROLL_SECONDS * SAMPLE_RATE) - WAKE_FRAME
                            self._fire(BargeIn(onset, frame_end, "wake", prob, score, self.clock()))
                            return

    def _fire(self, info: BargeIn) -> None:
        if info.reason == "wake":
            log.info("barge-in detected (wake word, score %.2f)", info.ratio)
        else:
            log.info("barge-in detected (vad %.2f, mic/echo ratio %.1f)", info.vad, info.ratio)
        if self.on_barge is not None:
            self.on_barge(info)
