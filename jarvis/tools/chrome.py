"""Chrome profiles and search: read profiles from Local State, launch Chrome in a chosen profile."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from .context import IS_WINDOWS, NO_WINDOW
from .registry import ToolError, tool

SEARCH_URL = "https://www.google.com/search?q="


def local_state_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Google" / "Chrome" / "User Data" / "Local State"


def parse_local_state(data: dict[str, Any]) -> list[dict[str, str]]:
    """profile.info_cache -> [{dir, name, gaia_name, user_name}] (Default first, then by dir)."""
    cache = ((data.get("profile") or {}).get("info_cache")) or {}
    out = []
    for d, info in cache.items():
        info = info if isinstance(info, dict) else {}
        out.append({"dir": str(d), "name": str(info.get("name") or ""),
                    "gaia_name": str(info.get("gaia_name") or ""), "user_name": str(info.get("user_name") or "")})
    out.sort(key=lambda p: (p["dir"] != "Default", p["dir"].lower()))
    return out


def load_profiles() -> list[dict[str, str]]:
    path = local_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read Chrome profiles from {path}: {exc}")
    return parse_local_state(data if isinstance(data, dict) else {})


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@. ]+", "", s.lower()).strip()


def match_profile(profiles: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    """Profiles matching `query` (best tier only; several means ambiguous, empty means none).

    Tiers: exact on directory/name/gaia name/email (or email local part); substring on any of
    them; close fuzzy match on name or gaia name.
    """
    q = _norm(query)
    if not q:
        return []
    fields = lambda p: [_norm(p["dir"]), _norm(p["name"]), _norm(p["gaia_name"]), _norm(p["user_name"])]  # noqa: E731
    exact = [p for p in profiles
             if q in fields(p) or (p["user_name"] and q == _norm(p["user_name"].split("@")[0]))]
    if exact:
        return exact
    sub = [p for p in profiles if any(q in f for f in fields(p) if f)]
    if sub:
        return sub
    scored = []
    for p in profiles:
        best = max((difflib.SequenceMatcher(None, q, f).ratio() for f in fields(p)[1:3] if f), default=0.0)
        if best >= 0.75:
            scored.append((best, p))
    if scored:
        top = max(s for s, _ in scored)
        return [p for s, p in scored if s == top]
    return []


def describe_profile(p: dict[str, str]) -> str:
    bits = [p["name"] or p["dir"]]
    extra = [x for x in (p["gaia_name"], p["user_name"]) if x and x != p["name"]]
    return bits[0] + (f" ({', '.join(extra)})" if extra else "")


def find_chrome() -> str:
    if IS_WINDOWS:
        try:
            import winreg

            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as k:
                        path = winreg.QueryValue(k, None)
                        if path and os.path.isfile(path):
                            return path
                except OSError:
                    continue
        except ImportError:
            pass
        for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                p = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
                if os.path.isfile(p):
                    return p
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    raise ToolError("Chrome does not seem to be installed")


def build_launch_args(chrome: str, profile_dir: str = "", url: str = "", search: str = "") -> list[str]:
    args = [chrome]
    if profile_dir:
        args.append(f"--profile-directory={profile_dir}")
    if search.strip():
        args.append(SEARCH_URL + quote_plus(search.strip()))
    elif url.strip():
        u = url.strip()
        if "://" not in u:
            u = "https://" + u
        if not u.lower().startswith(("http://", "https://")):
            raise ToolError("only http and https addresses can be opened")
        args.append(u)
    return args


@tool("List the Chrome profiles on this PC (display name, Google account name and email).")
def chrome_profiles() -> str:
    profiles = load_profiles()
    if not profiles:
        return "No Chrome profiles found"
    return "\n".join(f"{describe_profile(p)} [folder {p['dir']}]" for p in profiles)


@tool("Open Chrome, optionally in a named profile (display name, Google name, email or folder, e.g. 'work', "
      "'Nikolay'), and optionally on a web address or a Google search. Use this for 'search for X in Chrome' "
      "and 'open Chrome with my work profile'.")
def open_chrome(profile: str = "", url: str = "", search: str = "") -> str:
    """Open Chrome.

    Args:
        profile: Profile name, Google account name, email part or folder. Empty means Chrome's default.
        url: Optional web address to open.
        search: Optional text to search on Google.
    """
    profile_dir, label = "", "Chrome"
    if profile.strip():
        profiles = load_profiles()
        hits = match_profile(profiles, profile)
        if not hits:
            names = ", ".join(describe_profile(p) for p in profiles) or "none"
            raise ToolError(f"no Chrome profile matches '{profile}'. Profiles: {names}")
        if len(hits) > 1:
            raise ToolError(f"'{profile}' matches several profiles: {', '.join(describe_profile(p) for p in hits)}")
        profile_dir, label = hits[0]["dir"], f"Chrome ({hits[0]['name'] or hits[0]['dir']})"
    if not IS_WINDOWS and not shutil.which("chrome") and not shutil.which("google-chrome"):
        raise ToolError("open_chrome is only supported on Windows")
    args = build_launch_args(find_chrome(), profile_dir, url, search)
    subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=NO_WINDOW)
    what = f" and searched for {search.strip()}" if search.strip() else (" on " + url.strip() if url.strip() else "")
    return f"Opened {label}{what}"
