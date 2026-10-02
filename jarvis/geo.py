"""Location, geocoding and routing for the map feature.

Location order (`locate`): Windows location services (PowerShell GeoCoordinateWatcher), then the saved home
address (config or a "home is ..." memory fact, geocoded once and cached in state.json), then IP geolocation.
Geocoding is Nominatim (1 request per second, descriptive User-Agent). Routing is FOSSGIS OSRM (car, foot).
Free OSRM has no timetables, so public transport is only a Google Maps link, never an estimate.

Network functions raise `GeoError` whose message is fit to be spoken. Parsing and text helpers are pure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from . import store
from .config import Config

log = logging.getLogger("jarvis.geo")

USER_AGENT = "Jarvis-voice-assistant/0.1 (personal use)"
IS_WINDOWS = sys.platform == "win32"
CACHE_SECONDS = 600.0
OFFLINE_MSG = "I can't reach the map services right now. Please check the internet connection."


class GeoError(Exception):
    """Expected failure; the message can be spoken."""


@dataclass(frozen=True)
class Fix:
    lat: float
    lon: float
    accuracy_m: float
    source: str  # "windows" | "home" | "ip"

    def as_tuple(self) -> tuple[float, float, float, str]:
        return self.lat, self.lon, self.accuracy_m, self.source


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float
    label: str
    full: str = ""


# ---- pure helpers ----------------------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def viewbox(lat: float, lon: float, km: float = 50.0) -> str:
    """Nominatim viewbox 'lon_min,lat_max,lon_max,lat_min' around a point."""
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
    return f"{lon - dlon:.5f},{lat + dlat:.5f},{lon + dlon:.5f},{lat - dlat:.5f}"


def transit_url(origin: tuple[float, float], dest: tuple[float, float]) -> str:
    return ("https://www.google.com/maps/dir/?api=1"
            f"&origin={origin[0]:.6f},{origin[1]:.6f}&destination={dest[0]:.6f},{dest[1]:.6f}&travelmode=transit")


def short_label(item: dict[str, Any]) -> str:
    """Short name of a Nominatim result: its own name, else the first part of display_name."""
    name = str(item.get("name") or "").strip()
    if name:
        return name
    return str(item.get("display_name") or "").split(",")[0].strip()


def pick_best(results: list[dict[str, Any]], origin: tuple[float, float] | None) -> dict[str, Any] | None:
    """Best Nominatim result. With an origin, the nearest one among the reasonably important results."""
    items = [r for r in results if isinstance(r, dict) and "lat" in r and "lon" in r]
    if not items:
        return None
    if origin is None or len(items) == 1:
        return items[0]  # Nominatim already ranks by importance
    top = max(float(r.get("importance") or 0.0) for r in items)
    keep = [r for r in items if float(r.get("importance") or 0.0) >= top * 0.5] or items
    return min(keep, key=lambda r: haversine_m(origin[0], origin[1], float(r["lat"]), float(r["lon"])))


def parse_places(results: Any) -> list[Place]:
    out = []
    for r in results if isinstance(results, list) else []:
        try:
            out.append(Place(float(r["lat"]), float(r["lon"]), short_label(r), str(r.get("display_name") or "")))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_windows_location(stdout: str) -> Fix | None:
    """Last JSON line of the PowerShell output -> Fix, or None (disabled, unknown, garbage)."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(re.sub(r"\bNaN\b", "null", line))
        except ValueError:
            return None
        try:
            lat, lon = float(data["lat"]), float(data["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            return None
        acc = data.get("acc")
        return Fix(lat, lon, float(acc) if isinstance(acc, (int, float)) and acc > 0 else 100.0, "windows")
    return None


# ---- spoken text -----------------------------------------------------------------------------------------
def spoken_distance(m: float) -> str:
    if m < 950:
        n = max(50, int(round(m / 50.0)) * 50)
        return f"{n} metres"
    km = m / 1000.0
    if km < 10:
        r = round(km * 2) / 2 if km >= 3 else round(km, 1)
        return f"{int(r)} kilometre{'' if r == 1 else 's'}" if r == int(r) else f"{r:g} kilometres"
    return f"{int(round(km))} kilometres"


def spoken_travel(seconds: float) -> str:
    """Rounded, natural: 840 -> '14 minutes', 4200 -> 'an hour and 10 minutes', 9000 -> '2 and a half hours'."""
    if seconds < 45:
        return "under a minute"
    mins = seconds / 60.0
    if mins < 60:
        n = int(round(mins)) if mins < 20 else int(round(mins / 5.0)) * 5
        n = min(max(n, 1), 60)
        if n == 60:
            return "an hour"
        return "a minute" if n == 1 else f"{n} minutes"
    total = int(round(mins / 5.0)) * 5
    h, m = divmod(total, 60)
    if h >= 3:
        h, m = (h + 1, 0) if m >= 45 else (h, 30 if 15 <= m < 45 else 0)
        if m == 30:
            return f"{h} and a half hours"
        return f"{h} hours"
    hours = "an hour" if h == 1 else f"{h} hours"
    if m == 0:
        return hours
    if h == 1:
        return f"an hour and {m} minutes"
    return f"{hours} and {m} minutes"


# ---- route steps -----------------------------------------------------------------------------------------
_DIRS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def _compass(bearing: Any) -> str:
    try:
        return _DIRS[int(round(float(bearing) / 45.0)) % 8]
    except (TypeError, ValueError):
        return ""


def _turn_words(modifier: str) -> str:
    return {"left": "Turn left", "right": "Turn right", "slight left": "Bear left", "slight right": "Bear right",
            "sharp left": "Turn sharp left", "sharp right": "Turn sharp right", "uturn": "Make a U-turn",
            "straight": "Continue straight"}.get(modifier, "Continue")


def step_text(step: dict[str, Any]) -> str:
    """One OSRM step -> short human instruction ('Turn left onto Vitosha Boulevard')."""
    man = step.get("maneuver") or {}
    typ = str(man.get("type") or "")
    mod = str(man.get("modifier") or "")
    name = str(step.get("name") or "").strip() or str(step.get("ref") or "").strip()
    onto = f" onto {name}" if name else ""
    side = f" on the {mod}" if mod in ("left", "right") else ""
    if typ == "depart":
        d = _compass(man.get("bearing_after"))
        return f"Head {d}" + (f" on {name}" if name else "") if d else f"Start{(' on ' + name) if name else ''}"
    if typ == "arrive":
        return f"Arrive at your destination{side}"
    if typ in ("roundabout", "rotary", "roundabout turn"):
        if typ == "roundabout turn":
            return f"At the roundabout {_turn_words(mod).lower()}{onto}"
        n = man.get("exit")
        ex = _ORD.get(n) if isinstance(n, int) else None
        return f"At the roundabout take the {ex} exit{onto}" if ex else f"Enter the roundabout{onto}"
    if typ in ("exit roundabout", "exit rotary"):
        return f"Exit the roundabout{onto}"
    if typ == "fork":
        d = "left" if "left" in mod else "right" if "right" in mod else ""
        return f"Keep {d} at the fork{onto}" if d else f"Continue at the fork{onto}"
    if typ == "merge":
        d = f" {mod}" if mod in ("left", "right") else ""
        return f"Merge{d}{onto}"
    if typ == "on ramp":
        return f"Take the ramp{onto}"
    if typ == "off ramp":
        return f"Take the exit{onto}"
    if typ == "end of road":
        w = _turn_words(mod).lower() if mod else "turn"
        return f"At the end of the road {w}{onto}"
    if mod == "uturn":
        return f"Make a U-turn{(' on ' + name) if name else ''}"
    if typ in ("new name", "continue") and mod in ("", "straight"):
        return f"Continue{onto}"
    if typ in ("turn", "continue", "new name", "use lane", "notification"):
        base = _turn_words(mod) if mod else "Continue"
        return f"{base}{onto}"
    return f"Continue{onto}"


def parse_route(data: Any, mode: str) -> dict[str, Any]:
    """OSRM route response -> {mode, distance_m, duration_s, geometry, steps[{text, distance_m}]}."""
    try:
        if data.get("code") != "Ok":
            raise GeoError("No route found." if data.get("code") == "NoRoute" else "Routing failed.")
        r = data["routes"][0]
        geometry = r["geometry"]
        steps: list[dict[str, Any]] = []
        for leg in r.get("legs", []):
            for st in leg.get("steps", []):
                text = step_text(st)
                dist = float(st.get("distance") or 0.0)
                if dist <= 0 and (st.get("maneuver") or {}).get("type") != "arrive":
                    continue
                steps.append({"text": text, "distance_m": int(round(dist))})
        coords = [[round(float(x), 5), round(float(y), 5)] for x, y in geometry.get("coordinates", [])]
        return {"mode": mode, "distance_m": int(round(float(r["distance"]))),
                "duration_s": int(round(float(r["duration"]))),
                "geometry": {"type": "LineString", "coordinates": coords}, "steps": steps}
    except GeoError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise GeoError("The route service sent something I couldn't read.") from None


# ---- HTTP ------------------------------------------------------------------------------------------------
async def _get_json(url: str, params: dict[str, Any] | None = None, *, timeout: float = 6.0) -> Any:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=3.0),
                                     headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning("http %s from %s", exc.response.status_code, url)
        raise GeoError("The map service refused the request. Please try again in a moment.") from None
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("request to %s failed: %s", url, exc)
        raise GeoError(OFFLINE_MSG) from None


_nominatim_lock: asyncio.Lock | None = None  # created lazily (needs a running loop)
_nominatim_loop: Any = None
_nominatim_last = 0.0


async def _nominatim(cfg: Config, path: str, params: dict[str, Any]) -> Any:
    """Nominatim call, at most one per second (usage policy)."""
    global _nominatim_lock, _nominatim_loop, _nominatim_last
    loop = asyncio.get_running_loop()
    if _nominatim_lock is None or _nominatim_loop is not loop:
        _nominatim_lock, _nominatim_loop = asyncio.Lock(), loop
    async with _nominatim_lock:
        wait = 1.05 - (time.monotonic() - _nominatim_last)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            return await _get_json(cfg.maps.geocoder_url.rstrip("/") + path, params)
        finally:
            _nominatim_last = time.monotonic()


# ---- geocoding -------------------------------------------------------------------------------------------
_SAVED = {
    "home": [re.compile(r"^(?:my |the )?home(?: address)?\s*(?:is|=|:)\s*(?:at |in |on )?(?P<a>.+)$", re.I),
             re.compile(r"^(?:i|user) lives? (?:at|in|on) (?P<a>.+)$", re.I)],
    "work": [re.compile(r"^(?:my |the )?(?:work|office|workplace)(?: address)?\s*(?:is|=|:)\s*(?:at |in |on )?(?P<a>.+)$",
                        re.I),
             re.compile(r"^(?:i|user) works? (?:at|in) (?P<a>.+)$", re.I)],
}


def saved_address(kind: str, cfg: Config) -> str:
    """Saved address text for 'home' or 'work': config first, then a memory fact. '' if none."""
    configured = (cfg.maps.home_address if kind == "home" else cfg.maps.work_address).strip()
    if configured:
        return configured
    try:
        from .memory import default_memory

        facts = sorted(default_memory().facts(), key=lambda f: -f["created"])
    except Exception:  # noqa: BLE001
        return ""
    for f in facts:
        text = f["text"].strip().rstrip(".")
        for rx in _SAVED[kind]:
            m = rx.match(text)
            if m:
                return m.group("a").strip()
    return ""


def _cache_get(address: str) -> Place | None:
    entry = (store.load_state().get("geo_cache") or {}).get(address.lower())
    try:
        return Place(float(entry["lat"]), float(entry["lon"]), str(entry.get("label") or address))
    except (KeyError, TypeError, ValueError):
        return None


def _cache_put(address: str, place: Place) -> None:
    cache = store.load_state().get("geo_cache")
    cache = dict(cache) if isinstance(cache, dict) else {}
    cache[address.lower()] = {"lat": place.lat, "lon": place.lon, "label": place.label}
    store.update_state(geo_cache=cache)


async def geocode(cfg: Config, query: str, origin: tuple[float, float] | None = None) -> Place:
    """Find a place. 'home' / 'work' use the saved addresses. Raises GeoError (spoken) when not found."""
    q = " ".join((query or "").split())
    if not q:
        raise GeoError("Where to, exactly?")
    kind = {"home": "home", "my home": "home", "work": "work", "my work": "work", "the office": "work",
            "my office": "work", "office": "work"}.get(q.lower())
    if kind:
        addr = saved_address(kind, cfg)
        if not addr:
            raise GeoError(f"I don't know where your {kind} is yet. Tell me to remember that {kind} is at an address, "
                           f"or set it in the config.")
        cached = _cache_get(addr)
        if cached:
            return Place(cached.lat, cached.lon, kind.capitalize(), cached.label)
        place = await geocode(cfg, addr, origin)
        _cache_put(addr, place)
        return Place(place.lat, place.lon, kind.capitalize(), place.full or place.label)
    queries = [q]
    city = cfg.maps.default_city.strip()
    if city and city.lower() not in q.lower():
        queries.append(f"{q}, {city}")
    for text in queries:
        params: dict[str, Any] = {"q": text, "format": "jsonv2", "limit": 5, "bounded": 0}
        if origin is not None:
            params["viewbox"] = viewbox(*origin)
        best = pick_best(await _nominatim(cfg, "/search", params), origin)
        if best is not None:
            return Place(float(best["lat"]), float(best["lon"]), short_label(best), str(best.get("display_name") or ""))
    raise GeoError(f"I couldn't find {q} on the map.")


async def reverse_geocode(cfg: Config, lat: float, lon: float) -> str:
    """Short spoken place: 'Vitosha Boulevard in Sofia'. '' when nothing useful."""
    data = await _nominatim(cfg, "/reverse", {"lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "format": "jsonv2", "zoom": 17})
    return describe_address(data)


def describe_address(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    a = data.get("address") or {}
    street = a.get("road") or a.get("pedestrian") or a.get("footway") or a.get("neighbourhood") or a.get("suburb") or ""
    if street and a.get("house_number"):
        street = f"{street} {a['house_number']}"
    town = a.get("city") or a.get("town") or a.get("village") or a.get("municipality") or a.get("county") or ""
    if street and town:
        return f"{street} in {town}"
    return street or town or short_label(data)


# ---- routing ---------------------------------------------------------------------------------------------
_PROFILES = {"car": ("routed-car", "driving"), "walk": ("routed-foot", "foot")}


async def route(cfg: Config, mode: str, origin: tuple[float, float], dest: tuple[float, float]) -> dict[str, Any]:
    server, profile = _PROFILES[mode]
    url = (f"{cfg.maps.routing_url.rstrip('/')}/{server}/route/v1/{profile}/"
           f"{origin[1]:.6f},{origin[0]:.6f};{dest[1]:.6f},{dest[0]:.6f}")
    data = await _get_json(url, {"overview": "full", "geometries": "geojson", "steps": "true"}, timeout=10.0)
    return parse_route(data, mode)


# ---- location --------------------------------------------------------------------------------------------
WINDOWS_SCRIPT = r"""
Add-Type -AssemblyName System.Device
$w = New-Object System.Device.Location.GeoCoordinateWatcher
$w.Start()
$n = 0
while ($n -lt 50 -and $w.Status -ne 'Ready' -and $w.Permission -ne 'Denied') { Start-Sleep -Milliseconds 100; $n++ }
$loc = $w.Position.Location
if ($w.Status -eq 'Ready' -and -not $loc.IsUnknown) {
  $acc = $loc.HorizontalAccuracy
  if ([double]::IsNaN($acc)) { $acc = 0 }
  @{ lat = $loc.Latitude; lon = $loc.Longitude; acc = $acc } | ConvertTo-Json -Compress
} else {
  @{ error = ([string]$w.Permission + '/' + [string]$w.Status) } | ConvertTo-Json -Compress
}
$w.Stop()
"""


async def _windows_fix() -> Fix | None:
    if not IS_WINDOWS:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", WINDOWS_SCRIPT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, creationflags=0x08000000)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        log.info("windows location timed out")
        return None
    except Exception as exc:  # noqa: BLE001
        log.info("windows location unavailable: %s", exc)
        return None
    fix = parse_windows_location(out.decode("utf-8", "replace"))
    if fix is None:
        log.info("windows location off or unknown: %s", out.decode("utf-8", "replace").strip()[:120])
    return fix


async def _home_fix(cfg: Config) -> Fix | None:
    addr = saved_address("home", cfg)
    if not addr:
        return None
    try:
        place = _cache_get(addr)
        if place is None:
            place = await geocode(cfg, addr, None)
            _cache_put(addr, place)
    except GeoError as exc:
        log.info("home address not geocoded: %s", exc)
        return None
    return Fix(place.lat, place.lon, 100.0, "home")


async def _ip_fix() -> Fix | None:
    for url, parse in (("https://ipapi.co/json/", lambda d: (d["latitude"], d["longitude"])),
                       ("http://ip-api.com/json", lambda d: (d["lat"], d["lon"]))):
        try:
            data = await _get_json(url, timeout=4.0)
            lat, lon = parse(data)
            return Fix(float(lat), float(lon), 5000.0, "ip")
        except (GeoError, KeyError, TypeError, ValueError):
            continue
    return None


_fix_cache: tuple[float, Fix] | None = None


def clear_cache() -> None:
    global _fix_cache
    _fix_cache = None


async def locate(cfg: Config, *, use_cache: bool = True) -> tuple[float, float, float, str]:
    """(lat, lon, accuracy_m, source). Windows location, then saved home, then IP. Cached ~10 minutes."""
    global _fix_cache
    if use_cache and _fix_cache and time.monotonic() - _fix_cache[0] < CACHE_SECONDS:
        return _fix_cache[1].as_tuple()
    fix = await _windows_fix() or await _home_fix(cfg) or await _ip_fix()
    if fix is None:
        raise GeoError("I can't work out where you are. Turn on Windows location services, "
                       "or set a home address in the config.")
    log.info("location: %.5f, %.5f (+-%d m) from %s", fix.lat, fix.lon, fix.accuracy_m, fix.source)
    _fix_cache = (time.monotonic(), fix)
    return fix.as_tuple()
