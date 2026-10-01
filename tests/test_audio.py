import threading

import numpy as np

from jarvis.audio.chime import make_chime
from jarvis.audio.mic import MicStream
from jarvis.audio.recorder import BLOCK, EndpointDetector, Recorder, estimate_noise_floor
from jarvis.audio.wakeword import FRAME, WakeWordDetector

DT = BLOCK / 16000


def tone(n, amp=3000):
    return (np.sin(np.arange(n) / 5.0) * amp).astype(np.int16)


def test_mic_rechunk_and_continuity():
    mic = MicStream()
    mic.feed(np.arange(512, dtype=np.int16))
    mic.feed(np.arange(512, 1024, dtype=np.int16))
    a = mic.read(700, timeout=0.05)
    b = mic.read(324, timeout=0.05)
    assert a is not None and b is not None
    assert np.array_equal(np.concatenate([a, b]), np.arange(1024))
    assert mic.read(10, timeout=0.01) is None


def test_mic_flush_and_preroll():
    mic = MicStream(preroll_seconds=1.5)
    for _ in range(100):  # 3.2 s
        mic.feed(np.zeros(512, dtype=np.int16))
    pre = mic.preroll()
    assert 24000 <= len(pre) < 24000 + 512
    assert mic.flush() > 3.0
    assert mic.read(512, timeout=0.01) is None
    assert len(mic.preroll()) == len(pre)  # preroll survives flush


def test_wake_leaves_following_audio_queued():
    mic = MicStream()
    for i in range(10):
        mic.feed(np.full(FRAME, i, dtype=np.int16))

    class Model:
        def __init__(self):
            self.n = 0

        def predict(self, frame):
            self.n += 1
            return {"hey_jarvis": 0.9 if self.n == 3 else 0.0}

        def reset(self):
            pass

    wake = WakeWordDetector(threshold=0.5, mic=mic)
    wake._model = Model()
    assert wake.wait_blocking(threading.Event())
    assert abs(wake.last_score - 0.9) < 1e-6
    nxt = mic.read(FRAME, timeout=0.01)
    assert nxt is not None and nxt[0] == 3  # frame right after the detection frame (index 2)


def test_wake_stop():
    mic = MicStream()
    wake = WakeWordDetector(mic=mic)
    wake._model = type("M", (), {"predict": lambda s, f: {}, "reset": lambda s: None})()
    stop = threading.Event()
    stop.set()
    assert wake.wait_blocking(stop) is False


def test_endpoint_after_wake_already_speaking():
    det = EndpointDetector(after_wake=True)
    for _ in range(20):  # loud from the very first block: no calibration swallows it
        assert not det.feed(0.1, DT)
    assert det.speech_started
    done = False
    for _ in range(50):
        if det.feed(0.002, DT):
            done = True
            break
    assert done and det.reason == "silence"


def test_endpoint_after_wake_noise_floor_and_ignore():
    det = EndpointDetector(after_wake=True, noise_floor=0.01)
    assert abs(det.threshold - 0.03) < 1e-9
    assert EndpointDetector(after_wake=True, noise_floor=0.5).threshold <= 0.045 + 1e-9
    assert EndpointDetector(after_wake=True).threshold == det.min_threshold
    det = EndpointDetector(after_wake=True, ignore_seconds=0.2)
    for _ in range(6):  # 0.19 s of loud chime echo is ignored
        det.feed(0.5, DT)
    assert not det.speech_started
    det.feed(0.5, DT)
    assert det.speech_started


def test_noise_floor_estimate():
    quiet = tone(BLOCK * 10, 100)
    loud = tone(BLOCK * 20, 8000)
    assert estimate_noise_floor(np.concatenate([quiet, loud])) < 0.01
    assert estimate_noise_floor(np.zeros(10, dtype=np.int16)) == 0.0


def test_recorder_records_audio_following_wake():
    mic = MicStream()
    for _ in range(30):  # speech already in progress when recording starts
        mic.feed(tone(BLOCK))
    for _ in range(50):
        mic.feed(np.zeros(BLOCK, dtype=np.int16))
    rec = Recorder(mic, silence_seconds=0.5)
    audio = rec.record_blocking()
    assert rec.last_reason == "silence"
    assert audio.dtype == np.float32 and audio.size >= 30 * BLOCK
    assert np.abs(audio[: 30 * BLOCK]).max() > 0.05  # speech kept from the first block


def test_recorder_no_speech_and_flush():
    mic = MicStream()
    for _ in range(40):
        mic.feed(tone(BLOCK))  # stale audio
    rec = Recorder(mic, no_speech_timeout=0.5)
    for _ in range(40):
        pass
    done = threading.Event()

    def fill():
        while not done.is_set():
            mic.feed(np.zeros(BLOCK, dtype=np.int16))
            done.wait(0.002)

    t = threading.Thread(target=fill, daemon=True)
    t.start()
    audio = rec.record_blocking(flush=True)
    done.set()
    t.join()
    assert audio.size == 0 and rec.last_reason == "no_speech"


def test_chime():
    c = make_chime()
    assert c.dtype == np.float32 and 0.10 < len(c) / 24000 < 0.14
    assert np.abs(c).max() <= 0.2
