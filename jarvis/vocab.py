"""Edit the speech-recognition word list in config.toml from the command line.

    python -m jarvis.vocab add "Linkin Park" "Imagine Dragons"
    python -m jarvis.vocab remove "Discord"
    python -m jarvis.vocab list
    python -m jarvis.vocab model large-v3

Only the [whisper] lines it touches are rewritten; the rest of the file is kept as is.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from .config import DEFAULT_VOCABULARY

ROOT = Path(__file__).resolve().parent.parent
_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")


def _whisper_span(lines: list[str]) -> tuple[int, int] | None:
    """(header index, end index) of the [whisper] section, or None."""
    start = None
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m and start is None and m.group(1).strip() == "whisper":
            start = i
        elif m and start is not None:
            return start, i
    return (start, len(lines)) if start is not None else None


def _find_key(lines: list[str], span: tuple[int, int], key: str) -> int | None:
    rx = re.compile(rf"^\s*{key}\s*=")
    return next((i for i in range(span[0] + 1, span[1]) if rx.match(lines[i])), None)


def read_vocab(text: str) -> list[str]:
    lines = text.splitlines()
    span = _whisper_span(lines)
    i = _find_key(lines, span, "vocabulary") if span else None
    if i is None:
        return list(DEFAULT_VOCABULARY)
    m = re.search(r"\[(.*)\]", lines[i])
    return json.loads("[" + m.group(1) + "]") if m else list(DEFAULT_VOCABULARY)


def set_key(text: str, key: str, value_toml: str) -> str:
    lines = text.splitlines()
    span = _whisper_span(lines)
    if span is None:
        lines += ["", "[whisper]"]
        span = (len(lines) - 1, len(lines))
    i = _find_key(lines, span, key)
    new = f"{key} = {value_toml}"
    if i is None:
        lines.insert(span[1] if span[1] == len(lines) else span[0] + 1, new)
    else:
        lines[i] = new
    return "\n".join(lines) + "\n"


def write_vocab(text: str, words: list[str]) -> str:
    return set_key(text, "vocabulary", json.dumps(words, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jarvis.vocab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["add", "remove", "list", "model"])
    ap.add_argument("values", nargs="*")
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    args = ap.parse_args(argv)
    path = Path(args.config)
    if not path.exists():
        shutil.copy(ROOT / "config.example.toml", path)
    text = path.read_text(encoding="utf-8")
    words = read_vocab(text)
    if args.action == "list":
        print("\n".join(words))
        return 0
    if args.action == "model":
        if len(args.values) != 1:
            print("give one model name, e.g. large-v3")
            return 2
        text = set_key(text, "model", json.dumps(args.values[0]))
        print(f"whisper model set to {args.values[0]}")
    elif args.action == "add":
        for w in args.values:
            if w.strip() and w.lower() not in (x.lower() for x in words):
                words.append(w.strip())
        text = write_vocab(text, words)
        print("vocabulary: " + ", ".join(words))
    else:
        drop = {w.lower() for w in args.values}
        words = [w for w in words if w.lower() not in drop]
        text = write_vocab(text, words)
        print("vocabulary: " + ", ".join(words))
    shutil.copy(path, path.with_suffix(".toml.bak"))
    path.write_text(text, encoding="utf-8")
    print(f"saved {path} (backup: {path.with_suffix('.toml.bak').name}). Restart Jarvis to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
