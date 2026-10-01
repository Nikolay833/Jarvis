import sys
import types

import numpy as np
import pytest

from jarvis import stt_bench
from jarvis.audio import stt
from jarvis.audio.stt import ParakeetTranscriber, Transcriber, make_transcriber
from jarvis.config import config_from_dict


class FakeSeg:
    def __init__(self, text):
        self.text = text


class NewModel:
    def __init__(self):
        self.kw = None

    def transcribe(self, audio, language=None, beam_size=1, best_of=1, temperature=0.0, vad_filter=False,
                   vad_parameters=None, condition_on_previous_text=True, initial_prompt=None,
                   no_speech_threshold=0.6, log_prob_threshold=-1.0, hotwords=None):
        self.kw = dict(language=language, beam_size=beam_size, vad_filter=vad_filter, vad_parameters=vad_parameters,
                       hotwords=hotwords, temperature=temperature, initial_prompt=initial_prompt,
                       condition_on_previous_text=condition_on_previous_text)
        return iter([FakeSeg(" hello "), FakeSeg("world ")]), None


class OldModel:
    def transcribe(self, audio, language=None, beam_size=1, best_of=1, temperature=0.0, vad_filter=False,
                   vad_parameters=None, condition_on_previous_text=True, initial_prompt=None,
                   no_speech_threshold=0.6, log_prob_threshold=-1.0):
        return iter([FakeSeg("x")]), None


AUDIO = np.ones(16000, dtype=np.float32)


def test_engine_selection(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    assert isinstance(make_transcriber(config_from_dict({})), Transcriber)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "onnx_asr" else real(n, *a))
    assert isinstance(make_transcriber(config_from_dict({"stt": {"engine": "parakeet"}})), ParakeetTranscriber)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "onnx_asr" else real(n, *a))
    assert isinstance(make_transcriber(config_from_dict({"stt": {"engine": "parakeet"}})), Transcriber)
    assert isinstance(make_transcriber(config_from_dict({"stt": {"engine": "bogus"}})), Transcriber)


def test_whisper_defaults_from_config():
    t = make_transcriber(config_from_dict({"whisper": {"beam_size": 2}}))
    assert t.model_name == "large-v3" and t.beam_size == 2
    assert t.temperature == (0.0, 0.2, 0.4)
    assert t.vad_parameters == {"min_silence_duration_ms": 500, "speech_pad_ms": 300}
    assert t.condition_on_previous_text is False


def test_whisper_kwargs_with_hotwords():
    t = Transcriber(vocabulary=["Jarvis", "Claude"])
    t._model = NewModel()
    assert t.transcribe(AUDIO) == "hello world"
    kw = t._model.kw
    assert kw["beam_size"] == 5 and kw["vad_filter"] is True and kw["condition_on_previous_text"] is False
    assert kw["vad_parameters"]["speech_pad_ms"] == 300
    assert kw["hotwords"] == "Jarvis, Claude" and kw["initial_prompt"] == "Jarvis, Claude."
    assert kw["temperature"] == (0.0, 0.2, 0.4)


def test_whisper_no_hotwords_when_unsupported_or_disabled():
    t = Transcriber(vocabulary=["Jarvis"])
    t._model = OldModel()
    assert "hotwords" not in t.transcribe_kwargs()
    t2 = Transcriber(vocabulary=["Jarvis"], use_hotwords=False)
    t2._model = NewModel()
    assert "hotwords" not in t2.transcribe_kwargs()


def test_empty_audio():
    assert Transcriber().transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert ParakeetTranscriber().transcribe(np.zeros(0, dtype=np.float32)) == ""


def _fake_onnx_asr(monkeypatch, fail_cuda=False):
    calls = []

    class M:
        def recognize(self, wav, sample_rate=16000):
            calls.append(("recognize", wav.dtype, sample_rate))
            return " open claude "

    def load_model(name, providers=None):
        calls.append(("load", name, tuple(providers)))
        if fail_cuda and "CUDAExecutionProvider" in providers:
            raise RuntimeError("no cuda")
        return M()

    mod = types.ModuleType("onnx_asr")
    mod.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnx_asr", mod)
    return calls


def test_parakeet_adapter(monkeypatch):
    calls = _fake_onnx_asr(monkeypatch)
    p = ParakeetTranscriber("nemo-parakeet-tdt-0.6b-v2", "cuda")
    assert p.transcribe(AUDIO) == "open claude"
    assert calls[0] == ("load", "nemo-parakeet-tdt-0.6b-v2", ("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert calls[1] == ("recognize", np.float32, 16000)
    p.warm_up()


def test_parakeet_cpu_fallback(monkeypatch):
    calls = _fake_onnx_asr(monkeypatch, fail_cuda=True)
    p = ParakeetTranscriber()
    p.load()
    assert calls[-1][2] == ("CPUExecutionProvider",)


def test_parakeet_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx_asr", None)  # import raises ImportError
    with pytest.raises(RuntimeError, match=r'pip install "onnx-asr\[gpu,hub\]"'):
        ParakeetTranscriber().load()


def test_bench_table_and_wav(tmp_path):
    out = stt_bench.format_table([("whisper large-v3 beam5", 1.234, "Open Claude"), ("parakeet", 0.2, "open cloud")])
    lines = out.splitlines()
    assert lines[0].startswith("engine") and "time" in lines[0] and "transcript" in lines[0]
    assert "1.23s" in lines[2] and lines[2].endswith("Open Claude")
    assert len({l.index("|") for l in lines if "|" in l}) == 1  # aligned columns
    assert stt_bench.format_table([]) == ""
    p = tmp_path / "a.wav"
    stt_bench.write_wav(p, AUDIO * 0.5)
    back = stt_bench.read_wav(p)
    assert back.size == 16000 and abs(float(back[0]) - 0.5) < 1e-3


def test_bench_run_skips_failing_engine():
    class E:
        def load(self): pass
        def warm_up(self): pass
        def transcribe(self, a): return "hi"

    def boom():
        raise RuntimeError("nope")

    res = stt_bench.run_bench([("c.wav", AUDIO)], [("ok", E), ("bad", boom)])
    assert [r[0] for r in res["c.wav"]] == ["ok"] and res["c.wav"][0][2] == "hi"
