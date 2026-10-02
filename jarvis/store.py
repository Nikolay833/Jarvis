"""Small JSON files in the Jarvis app folder (%APPDATA%/Jarvis, else ~/.local/share/Jarvis)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.store")


def app_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "Jarvis"


def read_json(path: Path, default: Any) -> Any:
    """Parsed JSON, or `default` if the file is missing or broken (never raises)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        log.warning("could not read %s", path, exc_info=True)
        return default


def write_json(path: Path, data: Any) -> bool:
    """Atomic write (temp file + replace). Returns False on failure instead of raising."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except OSError:
        log.warning("could not write %s", path, exc_info=True)
        return False


def state_path() -> Path:
    return app_dir() / "state.json"


def load_state() -> dict[str, Any]:
    data = read_json(state_path(), {})
    return data if isinstance(data, dict) else {}


def update_state(**fields: Any) -> None:
    state = load_state()
    state.update(fields)
    write_json(state_path(), state)
