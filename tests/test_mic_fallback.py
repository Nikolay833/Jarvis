import numpy as np

from jarvis.audio.mic import MicStream


class FakeSD:
    """Minimal sounddevice stand-in: device 1 (MME) refuses 16 kHz, WASAPI twin works at 48 kHz."""

    def __init__(self, ok):
        self.ok = ok  # set of (device, rate) that open
        self.opened = []
        self.default = type("D", (), {"device": [1, 3]})()

    def query_hostapis(self):
        return [{"name": "MME"}, {"name": "Windows WASAPI"}]

    def query_devices(self):
        return [
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Microphone (USB Audio Device)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Microphone (USB Audio Device)", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
            {"name": "Stereo Mix", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100},
        ]

    def InputStream(self, samplerate, device, **kw):  # noqa: N802
        if (device, samplerate) not in self.ok:
            raise RuntimeError("Error opening InputStream: Insufficient memory [PaErrorCode -9992]")
        sd = self

        class S:
            def start(self):
                sd.opened.append((device, samplerate))

        return S()


def test_falls_back_to_same_mic_on_other_api_and_native_rate(monkeypatch):
    import sys

    fake = FakeSD({(2, 48000.0)})
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    mic = MicStream()
    mic.start()
    assert fake.opened == [(2, 48000.0)]
    assert mic._in_rate == 48000


def test_raises_clear_error_when_nothing_opens(monkeypatch):
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD(set()))
    with pytest.raises(RuntimeError, match="could not open any microphone"):
        MicStream().start()


def test_resample_48k_to_16k_is_continuous():
    mic = MicStream()
    mic._in_rate = 48000
    t = np.arange(48000) / 48000
    sig = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    out = np.concatenate([mic._resample(sig[i:i + 1536]) for i in range(0, len(sig), 1536)])
    assert abs(len(out) - 16000) <= 2
    ref = 8000 * np.sin(2 * np.pi * 440 * np.arange(len(out)) / 16000)
    assert np.max(np.abs(out.astype(np.float64) - ref)) < 400
