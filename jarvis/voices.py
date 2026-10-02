"""Audition Jarvis-style voices (Kokoro British male voices + a subtle AI processing chain).

    python -m jarvis.voices                      # play all presets, one after another
    python -m jarvis.voices --save out_dir       # also write wavs (add --no-play to only write)
    python -m jarvis.voices --voice bm_lewis --fx jarvis --amount 0.4 --speed 1.05
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .audio.tts import SAMPLE_RATE, KokoroSpeaker

LINE = "Good evening, sir. All systems are online, and I've taken the liberty of queuing your playlist."

# (name, voice, fx, amount, speed)
PRESETS: list[tuple[str, str, str, float, float]] = [
    ("bm_george plain", "bm_george", "none", 0.0, 1.0),
    ("bm_george + jarvis fx", "bm_george", "jarvis", 0.35, 1.05),
    ("bm_lewis + jarvis fx", "bm_lewis", "jarvis", 0.35, 1.05),
    ("bm_daniel + jarvis fx", "bm_daniel", "jarvis", 0.35, 1.05),
    ("bm_fable + jarvis fx", "bm_fable", "jarvis", 0.35, 1.05),
    ("blend george 0.6 / lewis 0.4 + jarvis fx", "bm_george:0.6,bm_lewis:0.4", "jarvis", 0.35, 1.05),
    ("bm_george + jarvis fx, speed 1.0", "bm_george", "jarvis", 0.35, 1.0),
    ("bm_george + jarvis fx, speed 1.05", "bm_george", "jarvis", 0.35, 1.05),
]


class _NullBus:
    def emit_nowait(self, *a, **k) -> None:
        pass


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _play(audio: np.ndarray) -> None:
    import sounddevice as sd

    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jarvis.voices", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", help='voice or blend, e.g. "bm_george:0.6,bm_lewis:0.4"')
    ap.add_argument("--fx", default="jarvis", choices=["jarvis", "none"])
    ap.add_argument("--amount", type=float, default=0.35)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--text", default=LINE)
    ap.add_argument("--lang", default="b")
    ap.add_argument("--save", metavar="DIR", help="write wavs to DIR")
    ap.add_argument("--no-play", action="store_true", help="do not play (use with --save)")
    args = ap.parse_args(argv)

    presets = PRESETS
    if args.voice:
        presets = [(f"custom: {args.voice} fx={args.fx} amount={args.amount} speed={args.speed}",
                    args.voice, args.fx, args.amount, args.speed)]
    outdir = Path(args.save) if args.save else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    sp = KokoroSpeaker(_NullBus(), lang_code=args.lang)
    print("Loading Kokoro...", flush=True)
    sp.load()
    for i, (name, voice, fx, amount, speed) in enumerate(presets, 1):
        print(f"[{i}/{len(presets)}] {name}", flush=True)
        sp.voice, sp.fx, sp.fx_amount, sp.speed, sp._voice_arg = voice, fx, amount, speed, None
        audio = sp.synth(args.text)
        if outdir:
            write_wav(outdir / f"{i:02d}_{_slug(name)}.wav", audio)
        if not args.no_play:
            _play(audio)
            time.sleep(0.6)

    last = presets[-1]
    print("\nTo use one, put this in config.toml (no command edits [tts]; edit the file by hand):\n")
    if args.voice:
        _, voice, fx, amount, speed = last
    else:
        voice, fx, amount, speed = "bm_george", "jarvis", 0.35, 1.05
        print("# example for 'bm_george + jarvis fx'; swap voice/fx_amount/speed for your favourite preset")
    print("[tts]")
    print(f'voice = "{voice}"')
    print('lang_code = "b"')
    print(f"speed = {speed}")
    print(f'fx = "{fx}"')
    print(f"fx_amount = {amount}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
