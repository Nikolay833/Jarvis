"""Map tools: directions, show a place, close the map, open transit in Google Maps, where am I (see geo.py).

The map itself is drawn by the orb app's full-screen "map" window from the `map_show` / `map_hide` bus events.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import webbrowser
from typing import Any

from .. import geo
from .context import ctx, emit
from .registry import ToolError, tool

log = logging.getLogger("jarvis.maps")

_last_transit_url: str = ""

_SOURCE_WORDS = {"windows": "from Windows location services", "home": "from your saved home address",
                 "ip": "roughly, from your internet connection"}


def last_transit_url() -> str:
    return _last_transit_url


def _open_url(url: str) -> None:
    if sys.platform == "win32":
        import os

        os.startfile(url)  # type: ignore[attr-defined]  # noqa: S606 - default browser
    else:
        webbrowser.open(url)


def _origin_dict(fix: tuple[float, float, float, str]) -> dict[str, Any]:
    lat, lon, acc, source = fix
    return {"lat": round(lat, 6), "lon": round(lon, 6), "label": "You are here", "accuracy": int(round(acc)),
            "source": source}


def build_map_event(fix: tuple[float, float, float, str], dest: geo.Place, routes: list[dict[str, Any]],
                    focus: str | None = None) -> dict[str, Any]:
    """Fields of the `map_show` bus event (see docs/ARCHITECTURE.md)."""
    show_transit = ctx.config.maps.show_transit_link
    return {
        "origin": _origin_dict(fix),
        "destination": {"lat": round(dest.lat, 6), "lon": round(dest.lon, 6), "label": dest.label},
        "routes": routes,
        "transit_url": geo.transit_url((fix[0], fix[1]), (dest.lat, dest.lon)) if show_transit else "",
        "focus": focus or "",
    }


def spoken_summary(routes: list[dict[str, Any]], transit: bool) -> str:
    by = {r["mode"]: r for r in routes}
    ref = by.get("car") or by.get("walk")
    if ref is None:
        return "I've put the destination on the map."
    parts = []
    if "car" in by:
        parts.append(f"By car roughly {geo.spoken_travel(by['car']['duration_s'])}")
    if "walk" in by:
        walk = geo.spoken_travel(by["walk"]["duration_s"])
        parts.append(f"on foot about {walk}" if parts else f"On foot about {walk}")
    text = f"It's about {geo.spoken_distance(ref['distance_m'])}. " + ", ".join(parts)
    if transit:
        text += "; for public transport I've put a Google Maps link on screen."
    else:
        text += "."
    return text


def _modes(mode: str) -> list[str]:
    m = (mode or "all").strip().lower()
    if m in ("car", "drive", "driving"):
        return ["car"]
    if m in ("walk", "walking", "foot", "on foot"):
        return ["walk"]
    return ["car", "walk"]


@tool("Show a full-screen map with directions to a destination: car and walking routes plus a Google Maps "
      "public transport link. Use for 'I want to go to X', 'how do I get to X', 'directions to X', 'navigate to X'. "
      "Say the returned text as given.")
async def directions(destination: str, mode: str = "all") -> str:
    """Directions from the user's current location.

    Args:
        destination: place name or address, e.g. "Sofia Airport"; "home" and "work" use the saved addresses
        mode: "all" (default), "car", "walk" or "transit" when the user asks for just one way of travelling
    """
    cfg = ctx.config
    try:
        fix = await geo.locate(cfg)
        origin = (fix[0], fix[1])
        dest = await geo.geocode(cfg, destination, origin)
        transit_only = (mode or "").strip().lower() in ("transit", "bus", "public transport", "metro", "tram")
        modes = [] if transit_only else _modes(mode)
        results = await asyncio.gather(*(geo.route(cfg, m, origin, (dest.lat, dest.lon)) for m in modes),
                                       return_exceptions=True)
    except geo.GeoError as exc:
        raise ToolError(str(exc)) from None
    routes = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]
    if modes and not routes:
        why = str(errors[0]) if errors and isinstance(errors[0], geo.GeoError) else "No route found."
        raise ToolError(f"I found {dest.label}, but couldn't get a route there. {why}")
    focus = "transit" if transit_only else (modes[0] if len(modes) == 1 else None)
    event = build_map_event(fix, dest, routes, focus)
    global _last_transit_url
    _last_transit_url = event["transit_url"]
    emit("map_show", **event)
    log.info("directions to %s: %s", dest.label, [(r["mode"], r["distance_m"], r["duration_s"]) for r in routes])
    if transit_only:
        return ("I've put the destination on the map with a Google Maps link for public transport, "
                "since I have no live timetables.")
    return spoken_summary(routes, bool(event["transit_url"]))


@tool("Show a place on a full-screen map without directions. Use for 'show me X on the map', 'where is X'.")
async def show_on_map(place: str) -> str:
    """Show a place on the map.

    Args:
        place: place name or address
    """
    cfg = ctx.config
    try:
        try:
            fix = await geo.locate(cfg)
        except geo.GeoError:
            fix = None  # a map of a place needs no origin
        dest = await geo.geocode(cfg, place, (fix[0], fix[1]) if fix else None)
    except geo.GeoError as exc:
        raise ToolError(str(exc)) from None
    origin = _origin_dict(fix) if fix else None
    global _last_transit_url
    _last_transit_url = ""
    emit("map_show", origin=origin,
         destination={"lat": round(dest.lat, 6), "lon": round(dest.lon, 6), "label": dest.label},
         routes=[], transit_url="", focus="")
    return f"Here is {dest.label} on the map."


@tool("Close the full-screen map.")
def close_map() -> str:
    emit("map_hide")
    return "Map closed"


@tool("Open the last public transport directions in Google Maps in the default browser. Use for 'open the "
      "transit directions' or 'show me the bus route'.")
def open_transit_directions() -> str:
    if not _last_transit_url:
        raise ToolError("There are no directions to open yet. Ask me for directions first.")
    try:
        _open_url(_last_transit_url)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"I couldn't open the browser: {exc}") from None
    return "Opening public transport directions in Google Maps"


@tool("Say roughly where the user is right now (street and town) using location services.")
async def where_am_i() -> str:
    cfg = ctx.config
    try:
        lat, lon, acc, source = await geo.locate(cfg)
        try:
            near = await geo.reverse_geocode(cfg, lat, lon)
        except geo.GeoError:
            near = ""
    except geo.GeoError as exc:
        raise ToolError(str(exc)) from None
    how = _SOURCE_WORDS.get(source, "")
    where = f"You're near {near}" if near else f"You're at {lat:.3f}, {lon:.3f}"
    if source == "windows":
        return f"{where}, {how}, accurate to about {geo.spoken_distance(max(acc, 50))}"
    return f"{where}, but that's only {how}"
