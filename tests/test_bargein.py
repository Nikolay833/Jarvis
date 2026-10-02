"""Barge-in: echo gate, calibration, speaker cancel, monitor and an end-to-end turn with fake audio."""
import asyncio
import sys
import threading
import time
import types

import numpy as np

from jarvis.__main__ import Assistant, Request
from jarvis.audio import tts
from jarvis.audio.bargein import (BargeIn, BargeInMonitor, EchoCalibrator, EchoGate, EnergyVAD, PlaybackTap,
                                  gate_params)
from jarvis.audio.mic import MicStream
from jarvis.audio.recorder import BLOCK, Recorder
from jarvis.audio.tts import KokoroSpeaker, fade_out, truncate_words
from jarvis.config import config_from_dict

DT = 512 / 16000  # 32 ms


def run_gate(gate, frames):
    """frames: (mic_rms, play_rms, vad). Returns the index of the frame that triggered, or None."""
    for i, (m, p, v) in enumerate(frames):
        if gate.feed(m, p, v, DT):
            return i
    return None


# ---- echo gate (pure) --------------------------------------------------------------------------
def test_echo_only_does_not_trigger_even_when_vad_is_fooled():
    # Jarvis loud (0.1), coupled into the mic at ~0.6; Silero thinks it is speech (0.9)
    g = EchoGate()
    assert run_gate(g, [(0.06, 0.1, 0.9)] * 200) is None
    assert g.suppressed  # logged as "suppressed as echo"


def test_loud_user_over_echo_triggers_after_min_duration():
    g = EchoGate()  # 250 ms: 8 frames of 32 ms (7 frames = 224 ms is not enough)
    idx = run_gate(g, [(0.06, 0.1, 0.9)] * 20 + [(0.5, 0.1, 0.95)] * 20)
    assert idx == 20 + 7
    assert abs(g.run_seconds - 8 * DT) < 1e-9 and g.peak_ratio > 2.5 and g.peak_vad == 0.95


def test_short_click_and_non_speech_noise_do_not_trigger():
    g = EchoGate()
    assert run_gate(g, [(0.06, 0.1, 0.9)] * 10 + [(0.8, 0.1, 0.1)] * 3 + [(0.06, 0.1, 0.9)] * 20) is None  # click, VAD low
    assert run_gate(g, [(0.8, 0.1, 0.95)] * 4 + [(0.06, 0.1, 0.05)] * 10) is None  # 128 ms of loud voice only
    assert run_gate(g, [(0.8, 0.1, 0.95)] * 4 + [(0.06, 0.1, 0.05)] * 4 + [(0.8, 0.1, 0.95)] * 4) is None  # gap too long


def test_short_gap_is_tolerated_and_counts_in_run_time():
    g = EchoGate()
    frames = [(0.5, 0.1, 0.9)] * 5 + [(0.06, 0.1, 0.2)] * 2 + [(0.5, 0.1, 0.9)] * 3
    assert run_gate(g, frames) == 9
    assert abs(g.run_seconds - 10 * DT) < 1e-9


def test_no_playback_needs_only_vad_and_absolute_floor():
    assert run_gate(EchoGate(), [(0.05, 0.0, 0.9)] * 12) == 7
    assert run_gate(EchoGate(), [(0.005, 0.0, 0.9)] * 40) is None  # whisper-quiet: below the floor
    assert run_gate(EchoGate(), [(0.05, 0.0, 0.3)] * 40) is None  # loud but VAD says no


def test_echo_window_covers_latency():
    # playback just stopped (0.1 -> 0): the echo is still arriving for ~150 ms, so it still must not trigger
    g = EchoGate()
    frames = [(0.07, 0.1, 0.9)] * 3 + [(0.07, 0.0, 0.9)] * 4 + [(0.0, 0.0, 0.0)] * 5
    assert run_gate(g, frames) is None


def test_sensitivity_mapping():
    mid = gate_params(0.5, 250)
    assert abs(mid["margin"] - 2.5) < 1e-9 and abs(mid["min_ms"] - 250) < 1e-9
    assert abs(mid["vad_threshold"] - 0.6) < 1e-9
    lo, hi = gate_params(0.0, 250), gate_params(1.0, 250)
    assert lo["margin"] > mid["margin"] > hi["margin"] and lo["min_ms"] > hi["min_ms"]
    assert gate_params(5)["margin"] == gate_params(1)["margin"]  # clamped
    assert EchoGate.from_sensitivity(1.0).margin < EchoGate.from_sensitivity(0.0).margin
    easy = EchoGate.from_sensitivity(1.0)  # 125 ms, margin 1.5: a quieter user triggers
    assert run_gate(easy, [(0.2, 0.1, 0.9)] * 10) is not None
    assert run_gate(EchoGate.from_sensitivity(0.0), [(0.2, 0.1, 0.9)] * 10) is None


# ---- calibration math --------------------------------------------------------------------------
def test_calibrator_percentile_clamp_and_minimum():
    c = EchoCalibrator(min_samples=10)
    assert c.value() is None
    for _ in range(9):
        c.add(0.05, 0.1)
    assert c.value() is None
    c.add(0.05, 0.1)
    assert abs(c.value() - 0.5) < 1e-9
    c.add(0.01, 0.005)  # playback too quiet to teach anything
    assert c.samples == 10
    c = EchoCalibrator()
    for r in [0.4] * 90 + [0.8] * 10:  # bursty echo path: p90 leans to the high side
        c.add(r * 0.1, 0.1)
    assert 0.4 < c.value() <= 0.8
    c = EchoCalibrator()
    for _ in range(20):
        c.add(0.0001, 0.1)
    assert c.value() == 0.02  # clamped low
    for _ in range(200):
        c.add(5.0, 0.1)
    assert c.value() == 2.0  # clamped high, and the ring forgets old samples


def test_gate_learns_coupling_from_non_speech_frames():
    g = EchoGate()
    assert g.coupling == g.default_coupling
    run_gate(g, [(0.03, 0.1, 0.05)] * 30)  # "Online, sir.": mic hears 0.3x the playback
    assert abs(g.coupling - 0.3) < 1e-6
    # with the learned coupling, a user at 0.12 over playback 0.1 (margin 2.5 * 0.03 = 0.075) is caught
    assert run_gate(g, [(0.12, 0.1, 0.9)] * 10) is not None


# ---- small pieces ------------------------------------------------------------------------------
def test_playback_tap_levels_and_latency():
    now = [100.0]
    tap = PlaybackTap(clock=lambda: now[0])
    tap.latency = 0.1
    tap.note(np.full(2400, 0.5, dtype=np.float32), 24000)  # 100 ms, audible 100.1 - 100.2
    tap.note(np.full(2400, 0.2, dtype=np.float32), 24000)  # queued behind it: 100.2 - 100.3
    assert tap.level_at(100.05) == 0.0
    assert abs(tap.level_at(100.15) - 0.5) < 1e-6 and abs(tap.level_at(100.25) - 0.2) < 1e-6
    assert tap.level_at(100.5) == 0.0


def test_fade_out_and_truncate():
    f = fade_out(np.ones(800, dtype=np.float32))
    assert f[0] == 1.0 and abs(f[719]) < 1e-6 and not f[720:].any() and np.all(np.diff(f[:720]) <= 0)
    assert truncate_words("one two three four", 0.5) == "one two"
    assert truncate_words("one two three four", 0.0) == "" and truncate_words("one two", 5) == "one two"


def test_energy_vad_fallback():
    v = EnergyVAD()
    quiet = (np.random.RandomState(0).randn(512) * 30).astype(np.int16)
    loud = (np.sin(np.arange(512) / 5.0) * 8000).astype(np.int16)
    assert all(v.prob(quiet) < 0.5 for _ in range(20))
    assert v.prob(loud) > 0.5


def test_mic_subscribe_and_grab_since():
    mic = MicStream()
    sub = mic.subscribe()
    for i in range(10):
        mic.feed(np.full(512, i + 1, dtype=np.int16))
    assert mic.position() == 5120
    blk, end, _t = sub.get_nowait()
    assert end == 512 and blk[0] == 1
    assert mic.read(512, timeout=0.01)[0] == 1  # subscribers do not take audio from read()
    got = mic.grab_since(5120 - 1024)
    assert len(got) == 1024 and got[0] == 9 and got[-1] == 10
    assert mic.read(512, timeout=0.01) is None  # queue dropped up to "now"
    mic.feed(np.full(512, 11, dtype=np.int16))
    assert mic.read(512, timeout=0.01)[0] == 11  # reading continues right after the grabbed audio
    assert len(mic.grab_since(-10**9)) == len(mic.preroll())  # clamped to the ring
    n = sub.qsize()
    mic.unsubscribe(sub)
    mic.feed(np.zeros(512, dtype=np.int16))
    assert sub.qsize() == n  # nothing delivered after unsubscribe


def test_recorder_prefix_counts_as_speech():
    mic = MicStream()
    for _ in range(30):
        mic.feed(np.zeros(BLOCK, dtype=np.int16))
    rec = Recorder(mic, silence_seconds=0.3)
    prefix = (np.sin(np.arange(4000) / 5.0) * 6000).astype(np.int16)
    audio = rec.record_blocking(prefix=prefix)
    assert rec.last_reason == "silence" and audio.size >= 4000
    assert np.abs(audio[:4000]).max() > 0.1  # the barge-in audio leads the recording
    mic2 = MicStream()
    for _ in range(30):
        mic2.feed(np.zeros(BLOCK, dtype=np.int16))
    assert Recorder(mic2, silence_seconds=0.3, no_speech_timeout=0.3).record_blocking().size == 0


# ---- speaker cancel ----------------------------------------------------------------------------
class FakeBus:
    def emit_nowait(self, *a, **k):
        pass


def make_speaker(monkeypatch, on_write=None):
    writes = []

    class FakeOut:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, block):
            writes.append((FakeOut.cur, block.copy()))
            if on_write:
                on_write(len(writes))

        cur = None

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=FakeOut))
    sp = KokoroSpeaker(FakeBus())
    sp._pipeline = object()
    synthed = []

    def synth(sentence):
        synthed.append(sentence)
        FakeOut.cur = sentence
        return np.full(4000, float(len(synthed)), dtype=np.float32)  # 5 play blocks; value = sentence number

    monkeypatch.setattr(sp, "synth", synth)
    return sp, writes, synthed


def test_speaker_interrupt_fades_drops_rest_and_resumes_after_begin_turn(monkeypatch):
    sp = None

    def on_write(n):
        if n == 7:  # during the second sentence: the user barges in
            sp.interrupt()

    sp, writes, synthed = make_speaker(monkeypatch, on_write)
    hooks = []
    sp.on_playback_start = lambda: hooks.append("start")
    sp.on_playback_end = lambda: hooks.append("end")

    async def run():
        s = sp.start_stream()
        s.push("First one here.")
        s.push("Second sentence is long.")
        s.push("Third never plays.")
        await asyncio.sleep(0.3)
        s.push("Late fourth.")  # still being generated
        await s.finish()
        return s

    s = asyncio.run(run())
    assert len(writes) == 8  # 5 + 2 + one faded block, then nothing
    assert [float(w[1].max()) for w in writes] == [1.0] * 5 + [2.0] * 3  # sentences 3 and 4 never reach the device
    last = writes[-1][1].reshape(-1)
    assert abs(last[719]) < 1e-6 and not last[720:].any() and last[0] > 0.9  # faded out, no click
    assert sp.spoken[0] == "First one here." and len(sp.spoken) == 2 and sp.spoken[1] in ("Second", "Second sentence")
    assert sp.muted and s.cancelled and hooks == ["start", "end"]
    assert synthed[:2] == ["First one here.", "Second sentence is long."]

    async def after():
        n = len(writes)
        await sp.speak("Muted, dropped.")  # same reply keeps being generated: silent
        assert len(writes) == n
        await sp.speak("Confirm me.", force=True)  # prompts still speak
        assert len(writes) == n + 5
        sp.begin_turn()
        await sp.speak("Next turn.")
        assert len(writes) == n + 10

    asyncio.run(after())


def test_speaker_interrupt_without_mute_keeps_later_speech(monkeypatch):
    sp = None

    def on_write(n):
        if n == 2:
            sp.interrupt(mute=False)

    sp, writes, _ = make_speaker(monkeypatch, on_write)
    asyncio.run(sp.speak("One two three four. Dropped with the utterance."))
    assert len(writes) == 3 and not sp.muted
    n = len(writes)
    asyncio.run(sp.speak("Answer this."))
    assert len(writes) == n + 5  # a confirmation re-prompt after an unmuted interrupt plays


# ---- monitor with fake mic ---------------------------------------------------------------------
class FakeVAD:
    def prob(self, frame):
        return 0.95 if np.abs(frame).mean() > 800 else 0.02


def const(level, n=512):
    return np.full(n, int(level * 32768), dtype=np.int16)


def make_monitor(mic, tap, **kw):
    found = []
    ev = threading.Event()

    def cb(info):
        found.append(info)
        ev.set()

    return BargeInMonitor(mic, tap, vad=FakeVAD(), on_barge=cb, **kw), found, ev


def test_monitor_ignores_echo_then_triggers_with_preroll():
    mic = MicStream()
    tap = PlaybackTap()
    mon, found, ev = make_monitor(mic, tap)
    mon.start()
    tap.note(np.full(24000, 0.3, dtype=np.float32), 24000)  # Jarvis plays a loud second
    for _ in range(20):  # his echo in the mic: 0.1 vs expected 0.8 * 0.3, VAD fooled
        mic.feed(const(0.1))
    time.sleep(0.15)
    assert not ev.is_set()
    start = mic.position()
    for _ in range(20):  # the user: 0.9
        mic.feed(const(0.9))
    assert ev.wait(2)
    mon.stop()
    info = found[0]
    assert info.reason == "voice" and info.vad > 0.9 and info.ratio > 2.5
    assert start < info.trigger <= start + 9 * 512  # after 8 loud frames (250 ms), not before
    assert info.onset == info.trigger - 8 * 512 - 4800  # 8 speech frames + 300 ms pre-roll
    got = mic.grab_since(info.onset)
    assert len(got) >= 4800 + 8 * 512 and np.abs(got).max() > 20000 and np.abs(got[:4000]).max() < 5000


def test_monitor_wake_word_always_counts():
    mic = MicStream()

    class Wake:
        threshold = 0.5
        n = 0

        def reset(self):
            pass

        def score(self, frame):
            self.n += 1
            assert len(frame) == 1280
            return 0.9 if self.n == 3 else 0.0

    mon, found, ev = make_monitor(mic, PlaybackTap(), wake=Wake())
    mon.start()
    for _ in range(20):
        mic.feed(const(0.001))  # nothing the VAD would call speech
    assert ev.wait(2)
    mon.stop()
    assert found[0].reason == "wake"


# ---- end to end: barge-in leads to a new turn with the pre-roll audio ---------------------------
class FakeStream:
    """Plays pushed sentences as they arrive (0.15 s each), like the real speaker."""

    def __init__(self, sp):
        self.sp = sp
        self.sentences = []
        self.first_audio_at = None
        self.pushed = 0
        self.closed = False
        self._task = asyncio.get_running_loop().create_task(self._play())

    def push(self, sentence):
        if not self.sp.muted:
            self.pushed += 1
            self.sentences.append(sentence)

    async def _play(self):
        i = 0
        while not self.sp.interrupted:
            if i < len(self.sentences):
                self.sp.spoken.append(self.sentences[i])
                self.sp.played.append(self.sentences[i])
                i += 1
                await asyncio.sleep(0.15)
            elif self.closed:
                return
            else:
                await asyncio.sleep(0.01)

    async def finish(self):
        self.closed = True
        await self._task


class FakeSpeaker:
    def __init__(self):
        self.muted = False
        self.interrupted = False
        self.spoken = []
        self.played = []
        self.interrupts = []

    def start_stream(self, force=False):
        self.interrupted = False
        self.spoken = []
        return FakeStream(self)

    async def speak(self, text, force=False):
        s = self.start_stream(force)
        s.push(text)
        await s.finish()

    def interrupt(self, mute=True):
        self.interrupts.append(mute)
        self.interrupted = True
        self.muted = self.muted or mute

    def begin_turn(self):
        self.muted = False

    def stop(self):
        pass


class FakeAgent:
    def __init__(self):
        self.msgs = []
        self.calls = []
        self.streamed = False
        self.first_audio_at = None
        self.timing = {}

    def history(self, session="default"):
        return self.msgs

    async def handle(self, text, session="default", speaker=None):
        self.calls.append(text)
        self.msgs.append({"role": "user", "content": text})
        stream = speaker.start_stream()
        if len(self.calls) == 1:
            reply = "Once upon a time. There was a dragon. It slept."
            stream.push("Once upon a time.")
            await asyncio.sleep(0.45)  # the LLM is still generating while the user barges in
            stream.push("There was a dragon.")
            stream.push("It slept.")
        else:
            reply = "Rain, sir."
            stream.push(reply)
        await stream.finish()
        self.msgs.append({"role": "assistant", "content": reply})
        self.streamed = True
        return reply


def test_barge_in_leads_to_new_turn_with_preroll_audio():
    cfg = config_from_dict({"audio": {"conversation": False, "follow_up": False}})
    a = Assistant(cfg, voice=False)
    a.voice = True
    a.mic = MicStream(preroll_seconds=5.0)
    a.recorder = Recorder(a.mic, silence_seconds=0.3, no_speech_timeout=1.0)
    heard = []

    class STT:
        def transcribe(self, audio):
            heard.append(audio)
            return "what about the weather"

    a.stt = STT()
    a.speaker = FakeSpeaker()
    a.barge = object()  # enables the interruptible path
    a.agent = FakeAgent()

    stop = threading.Event()
    pause = threading.Event()

    def mic_feeder():  # a live mic: quiet room
        while not stop.is_set():
            if not pause.is_set():
                a.mic.feed(const(0.0005))
            time.sleep(0.002)

    async def main():
        a._loop = asyncio.get_running_loop()
        feeder = threading.Thread(target=mic_feeder, daemon=True)
        feeder.start()
        turn = asyncio.create_task(a.run_turn(Request("text", "tell me a story")))
        await asyncio.sleep(0.25)  # sentence one is playing, the agent is still generating

        def user_barges_in():
            pause.set()
            time.sleep(0.02)
            for _ in range(10):  # the user starts talking over him
                a.mic.feed(const(0.4))
            end = a.mic.position()
            info = BargeIn(end - 5120 - 4800, end, "voice", 0.93, 4.1, time.monotonic())
            a._barge_cb(info)  # what the monitor thread does
            for _ in range(5):  # keeps talking, then stops
                a.mic.feed(const(0.4))
            pause.clear()  # the room goes quiet again

        await asyncio.to_thread(user_barges_in)
        await asyncio.wait_for(turn, 10)
        stop.set()

    asyncio.run(main())
    sp, ag = a.speaker, a.agent
    assert sp.interrupts == [True]
    assert ag.calls == ["tell me a story", "what about the weather"]  # next turn, no wake word
    assert "There was a dragon." not in sp.played and "It slept." not in sp.played  # silenced
    assert sp.played[-1] == "Rain, sir."  # and he answers the new question
    assert len(heard) == 1
    audio = heard[0]
    assert audio.size >= (5120 + 4800 + 5 * 512) / 16000 * 16000 * 0.95  # pre-roll + speech + what followed
    assert audio.max() > 0.3 and np.abs(audio[:4800]).max() < 0.01  # starts 300 ms before the speech
    # history stays honest: only what was spoken, marked as interrupted
    assert ag.msgs[1] == {"role": "assistant", "content": "Once upon a time. [interrupted by user]"}
    assert ag.msgs[3]["content"] == "Rain, sir."


def test_barge_during_confirmation_keeps_the_turn_and_does_not_mute():
    a = Assistant(config_from_dict({}), voice=False)
    a.speaker = FakeSpeaker()

    async def main():
        a._loop = asyncio.get_running_loop()
        a.confirmer._pending["x"] = asyncio.get_running_loop().create_future()  # a confirmation is waiting
        a._barge_cb(BargeIn(0, 100, "voice", 0.9, 3.0, time.monotonic()))
        await asyncio.sleep(0.01)

    asyncio.run(main())
    assert a.speaker.interrupts == [False] and not a.speaker.muted and not a._interrupted
    assert a._barge_pending()  # the confirmation's voice loop picks the audio up via record_text
