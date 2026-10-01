"""User folder resolution: ~, env vars, "desktop"/"documents"/... names, OneDrive-redirected folders."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

FOLDER_NAMES = ("desktop", "documents", "downloads", "pictures", "music", "videos")

# Known folder GUIDs (SHGetKnownFolderPath) and User Shell Folders registry value names.
_GUIDS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}
_REG_VALUES = {
    "desktop": "Desktop",
    "documents": "Personal",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "My Pictures",
    "music": "My Music",
    "videos": "My Video",
}
_NAME_RE = re.compile(r"^(?:(?:my|the)\s+)?(desktop|documents|downloads|pictures|music|videos)(?=$|[\\/])",
                      re.IGNORECASE)


def _shell_known_folder(name: str) -> Path | None:
    """SHGetKnownFolderPath via ctypes (Windows only)."""
    import ctypes
    from ctypes import wintypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    u = uuid.UUID(_GUIDS[name])
    guid = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8).from_buffer_copy(u.bytes[8:]))
    buf = ctypes.c_wchar_p()
    fn = ctypes.windll.shell32.SHGetKnownFolderPath  # type: ignore[attr-defined]
    fn.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
    if fn(ctypes.byref(guid), 0, None, ctypes.byref(buf)) != 0 or not buf.value:
        return None
    path = buf.value
    ctypes.windll.ole32.CoTaskMemFree(buf)  # type: ignore[attr-defined]
    return Path(path)


def _registry_folder(name: str) -> Path | None:
    import winreg  # type: ignore[import-not-found]

    key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:  # type: ignore[attr-defined]
        value, _ = winreg.QueryValueEx(k, _REG_VALUES[name])  # type: ignore[attr-defined]
    value = os.path.expandvars(str(value))
    return Path(value) if value and "%" not in value else None


def _windows_folder(name: str) -> Path | None:
    for getter in (_shell_known_folder, _registry_folder):
        try:
            p = getter(name)
        except Exception:  # noqa: BLE001  (any ctypes/registry failure: try the next source)
            p = None
        if p is not None:
            return p
    return None


@lru_cache(maxsize=None)
def known_folder(name: str) -> Path:
    """Real location of a user folder ("desktop", ...). Falls back to <home>/<Name>."""
    name = name.lower()
    if sys.platform == "win32":
        p = _windows_folder(name)
        if p is not None:
            return p
    return Path.home() / name.capitalize()


def clear_cache() -> None:
    known_folder.cache_clear()


def known_folders() -> dict[str, Path]:
    return {n: known_folder(n) for n in FOLDER_NAMES}


def resolve_path(path: str) -> Path:
    """Turn what the user or model said into an absolute path (not required to exist)."""
    raw = path.strip().strip('"').strip("'")
    m = _NAME_RE.match(raw)
    if m:
        p = known_folder(m.group(1)) / raw[m.end():].replace("\\", "/").lstrip("/")
    else:
        p = Path(os.path.expandvars(os.path.expanduser(raw)))
        if not p.is_absolute():
            p = Path.home() / p
        home = Path.home()
        try:
            parts = p.relative_to(home).parts
        except ValueError:
            parts = ()
        if parts and parts[0].lower() in FOLDER_NAMES and not (home / parts[0]).exists():
            real = known_folder(parts[0].lower())  # e.g. ~/Desktop on a OneDrive-redirected PC
            if real.exists():
                p = real.joinpath(*parts[1:])
    return p.resolve()


def folder_hint() -> str:
    """One static prompt line: home and resolved user folders."""
    f = known_folders()
    return (f"The user's home folder is {Path.home()}. Desktop is {f['desktop']}, Documents is {f['documents']}, "
            f"Downloads is {f['downloads']}. Build full paths from these.")
