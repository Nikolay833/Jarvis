"""Window management: list, minimize, maximize, restore, focus, close (Windows only, ctypes).

Matching is pure (`match_windows`) so it is tested on fake window lists. The Win32 calls are only
made inside functions guarded by IS_WINDOWS. Close is graceful (WM_CLOSE), never a kill.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from .apps import APP_MAP
from .context import IS_WINDOWS
from .registry import ToolError, tool

SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE = 6, 3, 9
WM_CLOSE = 0x0010
ACTIONS = ("minimize", "maximize", "restore", "focus", "close")

# apps.py Start-Process target -> process exe name, where it is not just "<target>.exe"
_EXE_FOR_TARGET = {"code": "code.exe", "wt": "windowsterminal.exe"}
# extra spoken names that are not in APP_MAP
_EXTRA_EXE = {"discord": "discord.exe", "task manager": "taskmgr.exe", "settings": "systemsettings.exe"}
# Jarvis's own overlay (Tauri) and shell windows are never touched
_SKIP_TITLES = {"jarvis orb", "program manager"}
_SKIP_EXES = {"jarvis-orb.exe", "jarvis orb.exe", "orb.exe"}
_SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


@dataclass(frozen=True)
class Win:
    hwnd: int
    title: str
    exe: str  # lowercase process image name, e.g. "chrome.exe"


def alias_exe(app: str) -> str | None:
    """Process exe for a spoken app name using the apps.py aliases, or None if unknown."""
    key = " ".join(app.lower().split())
    if key in APP_MAP:
        target = APP_MAP[key]
        return _EXE_FOR_TARGET.get(target, f"{target}.exe").lower()
    return _EXTRA_EXE.get(key)


def match_windows(windows: list[Win], app: str) -> list[Win]:
    """Windows matching `app`, best tier only, order kept (EnumWindows gives top of z-order first).

    Tier 1: process name equals the alias exe. Tier 2: exe name contains the query. Tier 3: title
    contains the query. All case-insensitive.
    """
    q = " ".join(app.lower().split())
    if not q:
        return []
    q_exe = q.removesuffix(".exe")
    want = alias_exe(q) or (q if q.endswith(".exe") else None)
    if want:
        hits = [w for w in windows if w.exe.lower() == want]
        if hits:
            return hits
    hits = [w for w in windows if q_exe and q_exe in w.exe.lower().removesuffix(".exe")]
    if hits:
        return hits
    return [w for w in windows if q in w.title.lower()]


# ---- Win32 plumbing (Windows only) ----------------------------------------
def _user32():
    import ctypes
    from ctypes import wintypes

    u = ctypes.windll.user32  # type: ignore[attr-defined]
    u.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    u.IsIconic.argtypes = [wintypes.HWND]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    return u


def _exe_name(pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value).lower()
        return ""
    finally:
        k32.CloseHandle(h)


def _cloaked(hwnd: int) -> bool:
    import ctypes
    from ctypes import wintypes

    try:
        val = wintypes.DWORD(0)
        dwm = ctypes.windll.dwmapi  # type: ignore[attr-defined]
        dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), 14, ctypes.byref(val), ctypes.sizeof(val))  # DWMWA_CLOAKED
        return bool(val.value)
    except Exception:  # noqa: BLE001
        return False


def enum_windows() -> list[Win]:
    """Visible, titled, non-cloaked, non-tool top-level windows, top of z-order first."""
    if not IS_WINDOWS:
        raise ToolError("window control is only supported on Windows")
    import ctypes
    from ctypes import wintypes

    u = _user32()
    own_pid = os.getpid()
    out: list[Win] = []

    def cb(hwnd, _lparam):
        try:
            if not u.IsWindowVisible(hwnd):
                return True
            n = u.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            if not title or title.lower() in _SKIP_TITLES:
                return True
            ex = u.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF  # GWL_EXSTYLE
            if ex & 0x80 and not ex & 0x40000:  # WS_EX_TOOLWINDOW without WS_EX_APPWINDOW
                return True
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            if cls.value in _SKIP_CLASSES or _cloaked(hwnd):
                return True
            pid = wintypes.DWORD(0)
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == own_pid:
                return True
            exe = _exe_name(pid.value)
            if exe in _SKIP_EXES:
                return True
            out.append(Win(int(hwnd), title, exe))
        except Exception:  # noqa: BLE001
            pass
        return True

    u.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb), 0)
    return out


def _press(vk: int, up: bool) -> None:
    import ctypes

    ctypes.windll.user32.keybd_event(vk, 0, 2 if up else 0, 0)  # type: ignore[attr-defined]


def focus_hwnd(hwnd: int) -> bool:
    """Bring a window to the front (restores if minimized). Uses the Alt-key trick to pass the
    foreground lock. Returns True if it is the foreground window afterwards."""
    u = _user32()
    u.AllowSetForegroundWindow(-1)  # ASFW_ANY
    if u.IsIconic(hwnd):
        u.ShowWindow(hwnd, SW_RESTORE)
    _press(0x12, False)  # VK_MENU down: lets SetForegroundWindow through
    try:
        u.SetForegroundWindow(hwnd)
    finally:
        _press(0x12, True)
    time.sleep(0.05)
    return int(u.GetForegroundWindow()) == hwnd


def _label(w: Win) -> str:
    return w.exe.removesuffix(".exe") or w.title[:30]


@tool("List the visible application windows on the PC (title and program name).")
def list_windows() -> str:
    wins = enum_windows()
    if not wins:
        return "No visible windows"
    return "\n".join(f"{w.exe or '?'}: {w.title[:80]}" for w in wins[:40])


@tool("Minimize, maximize, restore, bring to front (focus) or close an application's windows, e.g. "
      "'minimize chrome', 'focus spotify', 'close notepad'. app is a program or window name. "
      "Closing needs approval and is graceful (the app may ask to save).")
def window_action(app: str, action: str) -> str:
    """Control app windows.

    Args:
        app: Program or window title text, e.g. chrome, spotify, vscode, explorer.
        action: One of minimize, maximize, restore, focus, close.
    """
    act = action.strip().lower()
    act = {"minimise": "minimize", "maximise": "maximize", "front": "focus", "foreground": "focus",
           "show": "focus", "switch": "focus"}.get(act, act)
    if act not in ACTIONS:
        raise ToolError(f"action must be one of {', '.join(ACTIONS)}")
    hits = match_windows(enum_windows(), app)
    if not hits:
        raise ToolError(f"no open window found for '{app}'")
    u = _user32()
    if act == "focus":
        ok = focus_hwnd(hits[0].hwnd)
        return f"Brought {_label(hits[0])} to the front" if ok else \
            f"Asked Windows to show {_label(hits[0])}, but it may still be behind another window"
    if act == "maximize" or act == "restore":
        hits = hits[:1]  # one window: the top match
    for w in hits:
        if act == "close":
            u.PostMessageW(w.hwnd, WM_CLOSE, 0, 0)
        else:
            u.ShowWindow(w.hwnd, {"minimize": SW_MINIMIZE, "maximize": SW_MAXIMIZE, "restore": SW_RESTORE}[act])
        if act == "maximize":
            focus_hwnd(w.hwnd)
    past = {"minimize": "Minimized", "maximize": "Maximized", "restore": "Restored", "close": "Asked to close"}[act]
    n = len(hits)
    return f"{past} {n} {_label(hits[0])} window{'s' if n != 1 else ''}"


@tool("Minimize every window and show the desktop.")
def minimize_all() -> str:
    if not IS_WINDOWS:
        raise ToolError("window control is only supported on Windows")
    _press(0x5B, False)  # Win+M: minimize all (not a toggle)
    _press(0x4D, False)
    _press(0x4D, True)
    _press(0x5B, True)
    return "Minimized all windows"


def foreground_hwnd() -> int:
    """Handle of the current foreground window (0 if unknown or not on Windows)."""
    if sys.platform != "win32":
        return 0
    import ctypes

    return int(ctypes.windll.user32.GetForegroundWindow())
