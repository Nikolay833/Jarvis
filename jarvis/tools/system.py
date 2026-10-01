"""System tools: PowerShell, system info, lock."""

from __future__ import annotations

import asyncio
import ctypes
import datetime as dt
import os
import platform
import shutil

from .context import IS_WINDOWS, NO_WINDOW
from .registry import ToolError, tool

MAX_OUTPUT = 4000
POWERSHELL_TIMEOUT = 60


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _powershell_exe() -> str | None:
    for name in ("powershell", "pwsh"):
        path = shutil.which(name)
        if path:
            return path
    return None


@tool("Run a PowerShell command on the user's Windows PC and return its output. "
      "Risky commands (deleting, stopping processes, installing, registry) need approval.")
async def run_powershell(command: str) -> str:
    """Run a PowerShell command.

    Args:
        command: The PowerShell command line to run.
    """
    exe = _powershell_exe()
    if exe is None:
        raise ToolError("PowerShell is not available on this machine")
    proc = await asyncio.create_subprocess_exec(
        exe, "-NoProfile", "-NonInteractive", "-Command", command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        creationflags=NO_WINDOW,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=POWERSHELL_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"command timed out after {POWERSHELL_TIMEOUT} s")
    text = out.decode("utf-8", errors="replace").strip()
    prefix = "" if proc.returncode == 0 else f"[exit code {proc.returncode}]\n"
    return prefix + truncate(text or "(no output)")


def _memory_gb() -> tuple[float, float] | None:
    """(total, available) GB, or None if unknown."""
    try:
        if IS_WINDOWS:
            class MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return st.ullTotalPhys / 2**30, st.ullAvailPhys / 2**30
        info: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0])
        return info["MemTotal"] / 2**20, info.get("MemAvailable", info["MemFree"]) / 2**20
    except Exception:  # noqa: BLE001
        return None


@tool("Get CPU, memory, disk usage and the current date and time of the PC.")
def system_info() -> str:
    lines = [
        f"Time: {dt.datetime.now().strftime('%A %d %B %Y, %H:%M')}",
        f"OS: {platform.system()} {platform.release()}",
        f"CPU: {platform.processor() or 'unknown'}, {os.cpu_count()} logical cores",
    ]
    try:
        load = os.getloadavg()[0]  # not on Windows
        lines.append(f"Load average: {load:.2f}")
    except (AttributeError, OSError):
        pass
    mem = _memory_gb()
    if mem:
        lines.append(f"RAM: {mem[0] - mem[1]:.1f} GB used of {mem[0]:.1f} GB")
    root = os.path.abspath(os.sep) if not IS_WINDOWS else os.environ.get("SystemDrive", "C:") + "\\"
    try:
        du = shutil.disk_usage(root)
        lines.append(f"Disk {root}: {du.free / 2**30:.0f} GB free of {du.total / 2**30:.0f} GB")
    except OSError:
        pass
    return "\n".join(lines)


@tool("Lock the Windows PC (show the lock screen).")
def lock_pc() -> str:
    if not IS_WINDOWS:
        raise ToolError("locking is only supported on Windows")
    ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
    return "PC locked"
