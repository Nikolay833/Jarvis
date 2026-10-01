"""File tools. Optional `files.allowed_roots` restricts where they may act."""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from pathlib import Path

from .context import IS_WINDOWS, ctx
from .registry import ToolError, tool

MAX_READ_CHARS = 6000
MAX_LIST = 200
MAX_SEARCH_RESULTS = 50
SEARCH_TIME_LIMIT = 15.0


def resolve_path(path: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(path.strip().strip('"'))))
    if not p.is_absolute():
        p = Path.home() / p
    return p.resolve()


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
