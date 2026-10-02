"""File tools. Optional `files.allowed_roots` restricts where they may act."""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from pathlib import Path

from ..paths import resolve_path
from .context import IS_WINDOWS, ctx
from .registry import ToolError, tool

MAX_READ_CHARS = 6000
MAX_LIST = 200
MAX_SEARCH_RESULTS = 50
SEARCH_TIME_LIMIT = 15.0


def check_allowed(p: Path) -> Path:
    roots = [resolve_path(r) for r in ctx.config.files.allowed_roots]
    if roots and not any(p == r or r in p.parents for r in roots):
        raise ToolError(f"{p} is outside the allowed folders")
    return p


def _p(path: str) -> Path:
    return check_allowed(resolve_path(path))


@tool("List the files and folders in a directory.")
def list_dir(path: str) -> str:
    """List a directory.

    Args:
        path: Folder path. Use ~ for the home folder.
    """
    p = _p(path)
    if not p.is_dir():
        raise ToolError(f"{p} is not a folder")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    names = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_LIST]]
    more = f"\n...and {len(entries) - MAX_LIST} more" if len(entries) > MAX_LIST else ""
    return (f"{p} ({len(entries)} items)\n" + "\n".join(names) + more) if names else f"{p} is empty"


@tool("Read a text file (truncated if long).")
def read_file(path: str) -> str:
    """Read a text file.

    Args:
        path: File path.
    """
    p = _p(path)
    if not p.is_file():
        raise ToolError(f"{p} is not a file")
    with p.open("rb") as fh:
        data = fh.read(MAX_READ_CHARS * 4 + 1)
    if b"\x00" in data[:2048]:
        raise ToolError("file looks binary")
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n...[truncated]"
    return text or "(empty file)"


@tool("Create a folder (and missing parent folders). Safe.")
def create_folder(path: str) -> str:
    """Create a folder.

    Args:
        path: Folder to create. Can be like "desktop/new folder" or a full path.
    """
    p = _p(path)
    if p.is_dir():
        return f"{p} already exists"
    if p.exists():
        raise ToolError(f"{p} exists and is not a folder")
    p.mkdir(parents=True, exist_ok=True)
    return f"Created folder {p}"


@tool("Write text to a file. Creating a new file is safe; overwriting an existing file or writing a "
      "program or script file needs approval.")
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write a text file.

    Args:
        path: File path, e.g. "desktop/notes.txt".
        content: Full text to write.
        overwrite: Set true to replace an existing file.
    """
    p = _p(path)
    if p.is_dir():
        raise ToolError(f"{p} is a folder")
    if p.exists() and not overwrite:
        raise ToolError(f"{p} already exists; call again with overwrite=true to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if existed else 'Created'} {p} ({len(content)} characters)"


@tool("Search for files by name pattern (wildcards like *.pdf or report*) under a folder.")
def search_files(root: str, pattern: str) -> str:
    """Search files by name.

    Args:
        root: Folder to search in.
        pattern: Case-insensitive wildcard pattern, e.g. *.docx. Plain words match as *word*.
    """
    r = _p(root)
    if not r.is_dir():
        raise ToolError(f"{r} is not a folder")
    pat = pattern if any(c in pattern for c in "*?[") else f"*{pattern}*"
    pat = pat.lower()
    found: list[str] = []
    deadline = time.monotonic() + SEARCH_TIME_LIMIT
    for dirpath, dirnames, filenames in os.walk(r):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
        for name in filenames + dirnames:
            if fnmatch.fnmatch(name.lower(), pat):
                found.append(os.path.join(dirpath, name))
                if len(found) >= MAX_SEARCH_RESULTS:
                    return "\n".join(found) + "\n...[more results omitted]"
        if time.monotonic() > deadline:
            found.append("...[search time limit reached]")
            break
    return "\n".join(found) if found else "No matches"


_SKIP_DIRS = {"node_modules", "__pycache__", "appdata", "$recycle.bin", "windows", "program files",
              "program files (x86)", "site-packages", ".git", ".venv", "venv"}
_last_found: list[str] = []
DETAILS_MARK = "\n[details, not spoken]"


def _words(text: str) -> list[str]:
    import re

    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _matches(name: str, words: list[str]) -> bool:
    flat = " ".join(_words(name))
    joined = flat.replace(" ", "")
    return all(w in flat or w in joined for w in words)


def _spoken_place(path: str) -> str:
    parent = Path(path).parent
    return f"in {parent.name}" if parent.name else ""


@tool("Find files or folders on the PC by (part of) their name and give a short spoken answer. Use for "
      "'find X on my PC', 'search my documents for X', 'where is my X file'. Words match loosely: 'tax report' "
      "finds Tax_Report_2025.pdf. Afterwards open_path can open one of the results.")
def find_on_pc(name: str, where: str = "~") -> str:
    """Find files and folders by name.

    Args:
        name: Words from the file or folder name, e.g. "tax report".
        where: Folder to search: "~" (whole user folder), "desktop", "documents", "downloads", or a path.
    """
    root = _p(where or "~")
    if not root.is_dir():
        raise ToolError(f"{root} is not a folder")
    words = [w for w in _words(name) if w not in ("a", "the", "my", "file", "folder", "called", "named")]
    if not words:
        raise ToolError("tell me part of the name to look for")
    hits: list[tuple[int, float, str]] = []
    deadline = time.monotonic() + SEARCH_TIME_LIMIT
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d.lower() not in _SKIP_DIRS]
        for n in dirnames + filenames:
            if _matches(n, words):
                full = os.path.join(dirpath, n)
                exact = 0 if " ".join(_words(Path(n).stem)) == " ".join(words) else 1
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    mtime = 0.0
                hits.append((exact, -mtime, full))
        if time.monotonic() > deadline or len(hits) > 500:
            break
    hits.sort()
    _last_found[:] = [h[2] for h in hits[:10]]
    if not hits:
        return f"I couldn't find anything called {name} there."
    top = _last_found[:3]
    kinds = ["folder" if os.path.isdir(p) else "file" for p in top]
    if len(hits) == 1:
        return (f"Found it: the {kinds[0]} {Path(top[0]).name}, {_spoken_place(top[0])}. Shall I open it?"
                + DETAILS_MARK + " " + top[0])
    parts = [f"{Path(p).name} {_spoken_place(p)}" for p in top]
    more = f" and {len(hits) - 3} more" if len(hits) > 3 else ""
    return (f"I found {len(hits)} matches. The best are {', '.join(parts)}{more}. Which one should I open?"
            + DETAILS_MARK + " " + "; ".join(top))


@tool("Open a file or folder with its default application.")
def open_path(path: str) -> str:
    """Open a file or folder.

    Args:
        path: File or folder to open.
    """
    p = _p(path)
    if not p.exists():
        raise ToolError(f"{p} does not exist")
    if not IS_WINDOWS:
        raise ToolError("open_path is only supported on Windows")
    os.startfile(str(p))  # type: ignore[attr-defined]
    return f"Opened {p}"


@tool("Permanently delete a file or folder. Needs approval.", risk="risky")
def delete_path(path: str) -> str:
    """Delete a file or folder permanently.

    Args:
        path: File or folder to delete.
    """
    p = _p(path)
    if not p.exists():
        raise ToolError(f"{p} does not exist")
    if p == Path(p.anchor) or p == Path.home():
        raise ToolError("refusing to delete a drive root or the home folder")
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"Deleted {p}"


@tool("Move or rename a file or folder. Needs approval.", risk="risky")
def move_path(src: str, dst: str) -> str:
    """Move or rename.

    Args:
        src: Existing file or folder.
        dst: New location or name.
    """
    s, d = _p(src), _p(dst)
    if not s.exists():
        raise ToolError(f"{s} does not exist")
    shutil.move(str(s), str(d))
    return f"Moved {s} to {d}"
