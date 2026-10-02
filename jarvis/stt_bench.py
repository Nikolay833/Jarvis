"""Compare STT engines on your own voice and mic.  python -m jarvis.stt_bench [-n 4] [--files a.wav ...]"""

from __future__ import annotations

import argparse
import glob
import logging
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio.stt import ParakeetTranscriber, Transcriber, free_gpu_memory
from .config import Config, load_config

SAMPLE_RATE = 16000
OUT_DIR = Path("logs") / "stt_bench"
SCRIPT = [
    "Open Claude in the Jarvis project",
    "Play Numb by Linkin Park on Spotify",
    "Create a folder called testing on my desktop",
    "What did Claude say in the AI on PC session",
]


def read_wav(path: str | Path) -> np.ndarray:
    """16-bit PCM wav -> float32 mono 16 kHz (linear resample if needed)."""
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE and x.size:
        n = int(x.size * SAMPLE_RATE / rate)
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return x


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def record_utterances(n: int, device: str, cfg: Config) -> list[np.ndarray]:
    """Press Enter, speak, recording stops on silence (same endpointing as Jarvis)."""
    from .audio.mic import MicStream
    from .audio.recorder import Recorder

    mic = MicStream(device)
    mic.start()
    rec = Recorder(mic, cfg.audio.silence_seconds, cfg.audio.max_record_seconds, 8.0)
    out: list[np.ndarray] = []
    try:
        for i in range(n):
            hint = SCRIPT[i % len(SCRIPT)]
            input(f"[{i + 1}/{n}] Press Enter, then say: \"{hint}\" (or anything) ")
            audio = rec.record_blocking(flush=True)
            if audio.size == 0:
                print("  nothing heard, skipped")
                continue
            print(f"  got {audio.size / SAMPLE_RATE:.1f} s")
            out.append(audio)
    finally:
        mic.stop()
    return out


def build_engines(cfg: Config) -> list[tuple[str, Callable[[], Any]]]:
    """(label, factory) pairs; one model is built, used and freed at a time."""
    w = cfg.whisper
    kw = dict(device=w.device, compute_type=w.compute_type, fallback_model=w.fallback_model,
              language=w.language, vocabulary=w.vocabulary)

    def whisper(model: str, beam: int) -> Callable[[], Any]:
        return lambda: Transcriber(model, beam_size=beam, **kw)

    engines: list[tuple[str, Callable[[], Any]]] = [
        ("whisper large-v3-turbo beam1 (old)", whisper("large-v3-turbo", 1)),
        ("whisper large-v3 beam5", whisper("large-v3", 5)),
        ("whisper large-v3-turbo beam5", whisper("large-v3-turbo", 5)),
    ]
    try:
        import onnx_asr  # noqa: F401
        engines.append((f"parakeet {cfg.parakeet.model}", lambda: ParakeetTranscriber(cfg.parakeet.model, cfg.parakeet.device)))
    except ImportError:
        print('parakeet skipped (install with: pip install "onnx-asr[gpu,hub]")')
    return engines


def format_table(rows: list[tuple[str, float, str]]) -> str:
    """rows: (engine, seconds, transcript)."""
    if not rows:
        return ""
    w = max(len("engine"), *(len(r[0]) for r in rows))
    lines = [f"{'engine'.ljust(w)} | {'time':>6} | transcript", f"{'-' * w}-+-{'-' * 6}-+-{'-' * 30}"]
    for name, secs, text in rows:
        lines.append(f"{name.ljust(w)} | {secs:5.2f}s | {text}")
    return "\n".join(lines)


def run_bench(clips: list[tuple[str, np.ndarray]], engines: list[tuple[str, Callable[[], Any]]]) -> dict[str, list[tuple[str, float, str]]]:
    results: dict[str, list[tuple[str, float, str]]] = {name: [] for name, _ in clips}
    for label, factory in engines:
        print(f"== {label}: loading...", flush=True)
        try:
            eng = factory()
            eng.load()
            eng.warm_up()
        except Exception as exc:  # noqa: BLE001
            print(f"   skipped: {exc}")
            continue
        for name, audio in clips:
            t0 = time.perf_counter()
            try:
                text = eng.transcribe(audio)
            except Exception as exc:  # noqa: BLE001
                text = f"<error: {exc}>"
            results[name].append((label, time.perf_counter() - t0, text))
        del eng
        free_gpu_memory()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare speech-to-text engines on your own recordings.")
    ap.add_argument("-n", type=int, default=4, help="utterances to record (default 4)")
    ap.add_argument("--files", nargs="+", help="wav files to use instead of recording")
    ap.add_argument("--last", action="store_true", help="reuse the most recent set of recordings")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--device", help="mic name or index (default: [audio] input_device)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config)

    clips: list[tuple[str, np.ndarray]] = []
    if args.last:
        args.files = [str(OUT_DIR / "*.wav")]
    if args.files:
        # cmd.exe does not expand wildcards, so do it here.
        paths = sorted({p for f in args.files for p in (glob.glob(f) or [f])})
        if args.last:
            stamps = sorted({Path(p).name.rsplit("-", 1)[0] for p in paths})
            paths = [p for p in paths if stamps and Path(p).name.startswith(stamps[-1] + "-")]
        if not paths:
            print("No recordings found; run without --files/--last to record some.")
            return 1
        clips = [(Path(f).name, read_wav(f)) for f in paths]
    else:
        print("Say these (or your own):")
        for line in SCRIPT:
            print(f'  - "{line}"')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, audio in enumerate(record_utterances(args.n, args.device or cfg.audio.input_device, cfg), 1):
            path = OUT_DIR / f"{stamp}-{i}.wav"
            write_wav(path, audio)
            clips.append((path.name, audio))
        print(f"saved to {OUT_DIR}")
    if not clips:
        print("no audio")
        return 1

    results = run_bench(clips, build_engines(cfg))
    for name, audio in clips:
        print(f"\n### {name} ({audio.size / SAMPLE_RATE:.1f} s)")
        print(format_table(results[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
