"""Spotify (desktop app, works on the FREE plan): media keys, now playing, best-effort play by name.

Free plan limits: there is no API to start a chosen track without Premium. `spotify_play` opens the
track (or a search) in the desktop app via a spotify: URI, then tries to press Play by UI Automation
(optional pywinauto) or, failing that, one Enter key. It verifies by watching the window title.
"""

from __future__ import annotations

import base64
import os
import time
from urllib.parse import quote

from .context import IS_WINDOWS, ctx
from .registry import ToolError, tool
from . import system, windows

VK_MEDIA = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}
IDLE_TITLES = {"spotify", "spotify free", "spotify premium", "spotify - web player", "advertisement", "spotify - advertisement"}
WAIT_WINDOW = 8.0
VERIFY_SECS = 4.0


def parse_now_playing(title: str) -> tuple[str, str] | None:
    """Spotify's window title is 'Artist - Song' while playing, 'Spotify [Free|Premium]' otherwise."""
    t = " ".join(title.split())
    if not t or t.lower() in IDLE_TITLES or " - " not in t:
        return None
    artist, song = t.split(" - ", 1)
    return (artist.strip(), song.strip()) if artist.strip() and song.strip() else None


def _spotify_windows() -> list[windows.Win]:
    return [w for w in windows.enum_windows() if w.exe == "spotify.exe"]


def now_playing_from(wins: list[windows.Win]) -> tuple[str, str] | None:
    for w in wins:
        np = parse_now_playing(w.title)
        if np:
            return np
    return None


def _spoken(np: tuple[str, str]) -> str:
    return f"{np[1]} by {np[0]}"


@tool("Control music playback with media keys: play_pause, next, previous, stop. Works for Spotify and any "
      "media player. Use for 'pause the music', 'skip this song', 'resume'.")
def music_control(action: str) -> str:
    """Media key.

    Args:
        action: One of play_pause, next, previous, stop.
    """
    a = action.strip().lower().replace(" ", "_").replace("-", "_")
    a = {"play": "play_pause", "pause": "play_pause", "resume": "play_pause", "toggle": "play_pause",
         "skip": "next", "prev": "previous", "back": "previous"}.get(a, a)
    if a not in VK_MEDIA:
        raise ToolError("action must be one of play_pause, next, previous, stop")
    system.press_key(VK_MEDIA[a])
    return {"play_pause": "Toggled play/pause", "next": "Skipped to the next track",
            "previous": "Went to the previous track", "stop": "Stopped playback"}[a]


@tool("Say what Spotify is playing right now (song and artist).")
def spotify_now_playing() -> str:
    if not IS_WINDOWS:
        raise ToolError("Spotify control is only supported on Windows")
    wins = _spotify_windows()
    if not wins:
        return "Spotify is not open (or is minimised to the tray)"
    np = now_playing_from(wins)
    return f"Playing {_spoken(np)}" if np else "Nothing is playing in Spotify right now"


def find_track(query: str, client_id: str, client_secret: str) -> tuple[str, str, str] | None:
    """Spotify Web API (Client Credentials, no user login or Premium needed) -> (id, name, artist)."""
    import httpx

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    with httpx.Client(timeout=10) as c:
        r = c.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                   headers={"Authorization": f"Basic {auth}"})
        r.raise_for_status()
        token = r.json()["access_token"]
        r = c.get("https://api.spotify.com/v1/search", params={"q": query, "type": "track", "limit": 1},
                  headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        items = (r.json().get("tracks") or {}).get("items") or []
    if not items:
        return None
    t = items[0]
    return t["id"], t.get("name", ""), ", ".join(a.get("name", "") for a in t.get("artists", [])[:2])


def _uia_press_play(hwnd: int, track: str) -> bool:
    """Optional pywinauto: invoke a 'Play' button (prefer one naming the track). Best effort."""
    try:
        from pywinauto import Application  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        win = Application(backend="uia").connect(handle=hwnd, timeout=3).window(handle=hwnd)
        buttons = [(b, (b.element_info.name or "").strip()) for b in win.descendants(control_type="Button")]
        track_l = track.lower()
        pick = next((b for b, n in buttons if track_l and n.lower().startswith("play") and track_l in n.lower()), None)
        pick = pick or next((b for b, n in buttons if n.lower() == "play"), None)
        if pick is None:
            return False
        pick.invoke()
        return True
    except Exception:  # noqa: BLE001
        return False


@tool("Play a song in Spotify by name (best effort on the free plan). Use for 'play <song> on Spotify'. "
      "Report what actually started.")
def spotify_play(query: str) -> str:
    """Play a song in Spotify.

    Args:
        query: Song title, optionally with the artist.
    """
    if not IS_WINDOWS:
        raise ToolError("Spotify control is only supported on Windows")
    q = query.strip()
    if not q:
        raise ToolError("which song?")
    cfg = ctx.config.spotify
    track_id, track_name, label = "", "", q
    if cfg.client_id and cfg.client_secret:
        try:
            found = find_track(q, cfg.client_id, cfg.client_secret)
        except Exception as exc:  # noqa: BLE001
            found = None
            label = f"{q} (Spotify search API failed: {type(exc).__name__})"
        if found:
            track_id, track_name, artist = found
            label = f"{track_name} by {artist}"
    uri = f"spotify:track:{track_id}" if track_id else f"spotify:search:{quote(q)}"
    before = now_playing_from(_spotify_windows())
    os.startfile(uri)  # type: ignore[attr-defined]  # Windows only
    deadline = time.time() + WAIT_WINDOW
    wins: list[windows.Win] = []
    while time.time() < deadline and not wins:
        time.sleep(0.4)
        wins = _spotify_windows()
    if not wins:
        return f"Opened {label} in Spotify, but its window did not appear"
    time.sleep(1.5)  # let the page load
    wins = _spotify_windows() or wins
    focused = windows.focus_hwnd(wins[0].hwnd)
    pressed = _uia_press_play(wins[0].hwnd, track_name)
    if not pressed and focused:
        time.sleep(0.3)
        # Re-check right before the blind keypress: Enter sent to any other window
        # could submit a chat message or a form there.
        if windows.foreground_hwnd() == wins[0].hwnd:
            system.press_key(0x0D)  # Enter: starts the focused Play button/row on the opened page (unverified)
    end = time.time() + VERIFY_SECS
    now = None
    while time.time() < end:
        time.sleep(0.5)
        now = now_playing_from(_spotify_windows())
        if now and (now != before or (track_name and track_name.lower() in now[1].lower())):
            return f"Now playing {_spoken(now)}"
    kind = "track" if track_id else "search results"
    return (f"Opened {label} ({kind}) in Spotify; playback may need a click on the free plan"
            + (f". Still playing {_spoken(now)}." if now else "."))
