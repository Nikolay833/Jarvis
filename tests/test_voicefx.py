import sys
import time

import numpy as np
import pytest

from jarvis.audio import voicefx
from jarvis.audio.tts import KokoroSpeaker, parse_voice_spec

SR = 24000


def sig(sec=2.0):
    t = np.arange(int(sec * SR)) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    x = 0.4 * env * (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 420 * t) + 0.2 * np.sin(2 * np.pi * 3500 * t))
    return x.astype(np.float32)


def test_fx_none_identity():
    x = sig()
    assert voicefx.apply_fx(x, "none", 0.5) is x or np.array_equal(voicefx.apply_fx(x, "none", 0.5), x)


def test_amount_zero_identity():
    x = sig()
    assert np.allclose(voicefx.apply_fx(x, "jarvis", 0.0), x, atol=1e-6)


@pytest.mark.parametrize("amount", [0.2, 0.35, 1.0])
def test_jarvis_properties(amount):
    x = sig()
    y = voicefx.apply_fx(x, "jarvis", amount)
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert len(y) == len(x) + int(0.020 * SR)
    assert np.abs(y).max() <= 1.0


def test_loud_input_limited_and_silence_ok():
    assert np.abs(voicefx.apply_fx(sig() * 20, "jarvis", 1.0)).max() <= 1.0
    z = voicefx.apply_fx(np.zeros(1000, np.float32), "jarvis", 0.5)
    assert np.isfinite(z).all() and not z.any()
    assert voicefx.apply_fx(np.zeros(0, np.float32), "jarvis", 0.5).size == 0


def test_without_scipy(monkeypatch):
    monkeypatch.setattr(voicefx, "_signal", None)
    y = voicefx.apply_fx(sig(), "jarvis", 0.35)
    assert np.isfinite(y).all() and np.abs(y).max() <= 1.0


def test_cheap():
    x = sig(5.0)
    voicefx.apply_fx(x, "jarvis", 0.35)
    t = time.perf_counter()
    voicefx.apply_fx(x, "jarvis", 0.35)
    assert (time.perf_counter() - t) / 5.0 < 0.05  # generous; target is < 5 ms/s


def test_parse_voice_spec():
    assert parse_voice_spec("bm_george") == [("bm_george", 1.0)]
    s = parse_voice_spec("bm_george:0.6,bm_lewis:0.4")
    assert [n for n, _ in s] == ["bm_george", "bm_lewis"]
    assert [round(w, 3) for _, w in s] == [0.6, 0.4]
    assert [w for _, w in parse_voice_spec("a,b")] == [0.5, 0.5]
    for bad in ("", "a:x", "a:-1"):
        with pytest.raises(ValueError):
            parse_voice_spec(bad)


def test_synth_applies_fx_and_blend():
    calls = {}

    class P:
        def load_voice(self, n):
            return {"a": 1.0, "b": 3.0}[n]

        def __call__(self, text, voice, speed):
            calls["voice"], calls["speed"] = voice, speed
            yield "g", "p", sig(0.5)

    class Bus:
        def emit_nowait(self, *a, **k):
            pass

    sp = KokoroSpeaker(Bus(), voice="a:0.5,b:0.5", speed=1.05, fx="jarvis", fx_amount=0.35)
    sp._pipeline = P()
    y = sp.synth("hi")
    assert calls["voice"] == 2.0 and calls["speed"] == 1.05
    assert len(y) == int(0.5 * SR) + int(0.020 * SR)
    sp2 = KokoroSpeaker(Bus(), voice="a", fx="none")
    sp2._pipeline = P()
    assert len(sp2.synth("hi")) == int(0.5 * SR) and calls["voice"] == "a"


def test_voices_cli_presets():
    from jarvis import voices

    assert len(voices.PRESETS) >= 6
    assert any("," in p[1] for p in voices.PRESETS)
