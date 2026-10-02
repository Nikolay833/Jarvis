import asyncio
import json

import pytest

from jarvis import geo, store
from jarvis.config import Config, config_from_dict
from jarvis.fastpath import match
from jarvis.memory import default_memory
from jarvis.tools import load_all
from jarvis.tools import maps_tools as mt
from jarvis.tools.context import ctx, set_context

ORIGIN = (42.6847, 23.3188)

NOMINATIM_FIXTURE = [
    {"lat": "42.6966", "lon": "23.4114", "name": "Sofia Airport", "display_name": "Sofia Airport, Sofia, Bulgaria",
     "importance": 0.55},
    {"lat": "42.5000", "lon": "27.4000", "name": "Sofia Airport", "display_name": "Sofia Airport, Elsewhere",
     "importance": 0.60},
    {"lat": "10.0", "lon": "10.0", "name": "Tiny", "display_name": "Tiny", "importance": 0.05},
]


def osrm(steps, distance=6000.0, duration=840.0):
    return {"code": "Ok", "routes": [{"distance": distance, "duration": duration,
                                      "geometry": {"type": "LineString", "coordinates": [[23.3188, 42.6847], [23.4114, 42.6966]]},
                                      "legs": [{"steps": steps}]}]}


def step(typ, mod="", name="", dist=100.0, **man):
    return {"distance": dist, "name": name, "maneuver": {"type": typ, "modifier": mod, **man}}


@pytest.fixture(autouse=True)
def _fresh():
    geo.clear_cache()
    cfg = Config()
    cfg.maps.default_city = ""
    events = []

    class Bus:
        def emit_nowait(self, type, **f):
            events.append({"type": type, **f})

    set_context(cfg, Bus())
    yield events
    set_context(Config(), None)
    geo.clear_cache()


# ---- parsing and text ------------------------------------------------------------------------------------
def test_pick_best_prefers_nearest_among_important():
    best = geo.pick_best(NOMINATIM_FIXTURE, ORIGIN)
    assert best["lon"] == "23.4114"
    assert geo.pick_best(NOMINATIM_FIXTURE, None)["lon"] == "23.4114"  # rank order without origin
    assert geo.pick_best([], ORIGIN) is None


def test_parse_places_and_labels():
    places = geo.parse_places(NOMINATIM_FIXTURE + [{"lat": "x"}])
    assert len(places) == 3 and places[0].label == "Sofia Airport"
    assert geo.short_label({"display_name": "Vitosha Blvd, Sofia"}) == "Vitosha Blvd"


def test_step_texts():
    assert geo.step_text(step("depart", "", "Vitosha Boulevard", bearing_after=92)) == "Head east on Vitosha Boulevard"
    assert geo.step_text(step("turn", "left", "Vitosha Boulevard")) == "Turn left onto Vitosha Boulevard"
    assert geo.step_text(step("turn", "slight right", "")) == "Bear right"
    assert geo.step_text(step("end of road", "right", "Rakovski Street")) == "At the end of the road turn right onto Rakovski Street"
    assert geo.step_text(step("roundabout", "right", "Cherni Vrah", **{"exit": 2})) == "At the roundabout take the second exit onto Cherni Vrah"
    assert geo.step_text(step("new name", "straight", "Tsarigradsko Shose")) == "Continue onto Tsarigradsko Shose"
    assert geo.step_text(step("fork", "slight left", "A1")) == "Keep left at the fork onto A1"
    assert geo.step_text(step("arrive", "left")) == "Arrive at your destination on the left"
    assert geo.step_text(step("continue", "uturn", "Main")) == "Make a U-turn on Main"
    assert geo.step_text({"name": "", "ref": "E79", "maneuver": {"type": "on ramp"}}) == "Take the ramp onto E79"


def test_parse_route():
    data = osrm([step("depart", "", "A", 50, bearing_after=0), step("turn", "left", "B", 0.0),
                 step("turn", "right", "C", 250.4), step("arrive", "", "", 0.0)])
    r = geo.parse_route(data, "car")
    assert r["mode"] == "car" and r["distance_m"] == 6000 and r["duration_s"] == 840
    assert r["geometry"]["type"] == "LineString" and len(r["geometry"]["coordinates"]) == 2
    assert [s["text"] for s in r["steps"]] == ["Head north on A", "Turn right onto C", "Arrive at your destination"]
    assert r["steps"][1]["distance_m"] == 250
    with pytest.raises(geo.GeoError):
        geo.parse_route({"code": "NoRoute"}, "car")
    with pytest.raises(geo.GeoError):
        geo.parse_route({"code": "Ok", "routes": []}, "car")


def test_spoken_rounding():
    assert geo.spoken_distance(420) == "400 metres"
    assert geo.spoken_distance(5960) == "6 kilometres"
    assert geo.spoken_distance(6400) == "6.5 kilometres"
    assert geo.spoken_distance(1200) == "1.2 kilometres"
    assert geo.spoken_distance(23400) == "23 kilometres"
    assert geo.spoken_travel(20) == "under a minute"
    assert geo.spoken_travel(14 * 60 + 10) == "14 minutes"
    assert geo.spoken_travel(33 * 60) == "35 minutes"
    assert geo.spoken_travel(70 * 60) == "an hour and 10 minutes"
    assert geo.spoken_travel(3600) == "an hour"
    assert geo.spoken_travel(2 * 3600 + 20 * 60) == "2 hours and 20 minutes"
    assert geo.spoken_travel(4 * 3600 + 35 * 60) == "4 and a half hours"


def test_transit_url_and_viewbox():
    url = geo.transit_url(ORIGIN, (42.6966, 23.4114))
    assert url == ("https://www.google.com/maps/dir/?api=1&origin=42.684700,23.318800"
                   "&destination=42.696600,23.411400&travelmode=transit")
    left, top, right, bottom = map(float, geo.viewbox(*ORIGIN).split(","))
    assert left < ORIGIN[1] < right and bottom < ORIGIN[0] < top
    assert 95 < geo.haversine_m(0, 0, 0, 1) / 1000 < 112


def test_parse_windows_location():
    f = geo.parse_windows_location('noise\n{"lat":42.68,"lon":23.31,"acc":35.0}\n')
    assert f and (f.lat, f.lon, f.accuracy_m, f.source) == (42.68, 23.31, 35.0, "windows")
    assert geo.parse_windows_location('{"lat":42.68,"lon":23.31,"acc":NaN}').accuracy_m == 100.0
    assert geo.parse_windows_location('{"error":"Denied/Disabled"}') is None
    assert geo.parse_windows_location("") is None
    assert geo.parse_windows_location('{"lat":0,"lon":0,"acc":1}') is None


# ---- locate fallback order -------------------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


def test_locate_order(monkeypatch):
    cfg = Config()
    calls = []

    async def win(): calls.append("win"); return None
    async def home(c): calls.append("home"); return None
    async def ip(): calls.append("ip"); return geo.Fix(1.0, 2.0, 5000.0, "ip")

    monkeypatch.setattr(geo, "_windows_fix", win)
    monkeypatch.setattr(geo, "_home_fix", home)
    monkeypatch.setattr(geo, "_ip_fix", ip)
    assert _run(geo.locate(cfg)) == (1.0, 2.0, 5000.0, "ip")
    assert calls == ["win", "home", "ip"]
    calls.clear()
    assert _run(geo.locate(cfg))[3] == "ip" and calls == []  # cached
    geo.clear_cache()

    async def home2(c): return geo.Fix(3.0, 4.0, 100.0, "home")
    monkeypatch.setattr(geo, "_home_fix", home2)
    assert _run(geo.locate(cfg))[3] == "home"
    geo.clear_cache()

    async def win2(): return geo.Fix(5.0, 6.0, 20.0, "windows")
    monkeypatch.setattr(geo, "_windows_fix", win2)
    assert _run(geo.locate(cfg)) == (5.0, 6.0, 20.0, "windows")


def test_locate_nothing_is_spoken_error(monkeypatch):
    async def none(*a): return None
    monkeypatch.setattr(geo, "_windows_fix", none)
    monkeypatch.setattr(geo, "_home_fix", none)
    monkeypatch.setattr(geo, "_ip_fix", none)
    with pytest.raises(geo.GeoError, match="location"):
        _run(geo.locate(Config()))


def test_windows_fix_skipped_off_windows(monkeypatch):
    monkeypatch.setattr(geo, "IS_WINDOWS", False)
    assert _run(geo._windows_fix()) is None


def test_home_from_config_geocoded_once_and_cached(monkeypatch):
    cfg = config_from_dict({"maps": {"home_address": "12 Vitosha Blvd, Sofia"}})
    n = []

    async def nom(c, path, params):
        n.append(params["q"])
        return [{"lat": "42.69", "lon": "23.32", "name": "12", "display_name": "12 Vitosha Blvd"}]

    monkeypatch.setattr(geo, "_nominatim", nom)
    fix = _run(geo._home_fix(cfg))
    assert fix and fix.source == "home" and (fix.lat, fix.lon) == (42.69, 23.32)
    assert _run(geo._home_fix(cfg)).lat == 42.69 and len(n) == 1  # second time from state.json
    assert "12 vitosha blvd, sofia" in store.load_state()["geo_cache"]


def test_home_from_memory_fact():
    cfg = Config()
    assert geo.saved_address("home", cfg) == ""
    default_memory().add("Home is 5 Graf Ignatiev Street, Sofia")
    default_memory().add("Work is at Tsarigradsko Shose 7, Sofia")
    assert geo.saved_address("home", cfg) == "5 Graf Ignatiev Street, Sofia"
    assert geo.saved_address("work", cfg) == "Tsarigradsko Shose 7, Sofia"


def test_ip_fix_fallback_url(monkeypatch):
    seen = []

    async def get(url, params=None, timeout=6.0):
        seen.append(url)
        if "ipapi" in url:
            raise geo.GeoError("offline")
        return {"lat": 42.7, "lon": 23.3}

    monkeypatch.setattr(geo, "_get_json", get)
    fix = _run(geo._ip_fix())
    assert fix.source == "ip" and fix.accuracy_m == 5000.0 and "ip-api.com" in seen[-1]


# ---- geocode / route requests ----------------------------------------------------------------------------
def test_geocode_request_and_saved_places(monkeypatch):
    cfg = Config()
    cfg.maps.default_city = "Sofia"
    reqs = []

    async def nom(c, path, params):
        reqs.append((path, params))
        return [] if "Sofia" not in params["q"] else NOMINATIM_FIXTURE

    monkeypatch.setattr(geo, "_nominatim", nom)
    place = _run(geo.geocode(cfg, "airport", ORIGIN))
    assert place.label == "Sofia Airport" and place.lon == 23.4114
    assert reqs[0][1]["format"] == "jsonv2" and reqs[0][1]["limit"] == 5 and reqs[0][1]["bounded"] == 0
    assert "viewbox" in reqs[0][1] and reqs[1][1]["q"] == "airport, Sofia"  # retry with the default city
    with pytest.raises(geo.GeoError, match="home"):
        _run(geo.geocode(cfg, "home", ORIGIN))
    cfg.maps.work_address = "Tsarigradsko 7"
    assert _run(geo.geocode(cfg, "work", ORIGIN)).label == "Work"


def test_route_url(monkeypatch):
    seen = []

    async def get(url, params=None, timeout=6.0):
        seen.append((url, params))
        return osrm([step("arrive")])

    monkeypatch.setattr(geo, "_get_json", get)
    _run(geo.route(Config(), "walk", ORIGIN, (42.6966, 23.4114)))
    assert seen[0][0] == ("https://routing.openstreetmap.de/routed-foot/route/v1/foot/"
                          "23.318800,42.684700;23.411400,42.696600")
    assert seen[0][1] == {"overview": "full", "geometries": "geojson", "steps": "true"}
    _run(geo.route(Config(), "car", ORIGIN, (42.6966, 23.4114)))
    assert "/routed-car/route/v1/driving/" in seen[1][0]


def test_describe_address():
    d = {"address": {"road": "Vitosha Boulevard", "house_number": "12", "city": "Sofia"}}
    assert geo.describe_address(d) == "Vitosha Boulevard 12 in Sofia"
    assert geo.describe_address({"address": {"town": "Pernik"}}) == "Pernik"


# ---- tools -----------------------------------------------------------------------------------------------
def _patch_geo(monkeypatch, car_ok=True):
    async def locate(cfg, **kw): return (ORIGIN[0], ORIGIN[1], 30.0, "windows")
    async def geocode(cfg, q, origin=None): return geo.Place(42.6966, 23.4114, "Sofia Airport")
    async def route(cfg, mode, o, d):
        if mode == "car" and not car_ok:
            raise geo.GeoError("No route found.")
        return geo.parse_route(osrm([step("arrive")], 6000.0, 840.0 if mode == "car" else 4200.0), mode)
    monkeypatch.setattr(geo, "locate", locate)
    monkeypatch.setattr(geo, "geocode", geocode)
    monkeypatch.setattr(geo, "route", route)


def test_directions_event_and_summary(monkeypatch, _fresh):
    _patch_geo(monkeypatch)
    reg = load_all()
    out = _run(reg.call("directions", {"destination": "Sofia Airport"}))
    assert out == ("It's about 6 kilometres. By car roughly 14 minutes, on foot about an hour and 10 minutes; "
                   "for public transport I've put a Google Maps link on screen.")
    ev = _fresh[-1]
    assert ev["type"] == "map_show"
    assert ev["origin"] == {"lat": 42.6847, "lon": 23.3188, "label": "You are here", "accuracy": 30, "source": "windows"}
    assert ev["destination"] == {"lat": 42.6966, "lon": 23.4114, "label": "Sofia Airport"}
    assert [r["mode"] for r in ev["routes"]] == ["car", "walk"]
    assert set(ev["routes"][0]) == {"mode", "distance_m", "duration_s", "geometry", "steps"}
    assert ev["transit_url"].endswith("travelmode=transit")
    json.dumps(ev)
    assert mt.last_transit_url() == ev["transit_url"]


def test_directions_one_mode_and_failures(monkeypatch, _fresh):
    _patch_geo(monkeypatch, car_ok=False)
    reg = load_all()
    out = _run(reg.call("directions", {"destination": "x"}))  # walk still works
    assert "On foot" in out and [r["mode"] for r in _fresh[-1]["routes"]] == ["walk"]
    out = _run(reg.call("directions", {"destination": "x", "mode": "car"}))
    assert out.startswith("Error: I found Sofia Airport, but couldn't get a route")
    out = _run(reg.call("directions", {"destination": "x", "mode": "transit"}))
    assert "Google Maps" in out and _fresh[-1]["routes"] == []


def test_directions_offline(monkeypatch):
    async def locate(cfg, **kw): raise geo.GeoError(geo.OFFLINE_MSG)
    monkeypatch.setattr(geo, "locate", locate)
    assert _run(load_all().call("directions", {"destination": "x"})).startswith("Error: I can't reach the map")


def test_show_close_open_where(monkeypatch, _fresh):
    _patch_geo(monkeypatch)
    reg = load_all()
    assert "Sofia Airport" in _run(reg.call("show_on_map", {"place": "airport"}))
    assert _fresh[-1]["routes"] == [] and _fresh[-1]["type"] == "map_show"
    _run(reg.call("close_map", {}))
    assert _fresh[-1] == {"type": "map_hide"}
    assert _run(reg.call("open_transit_directions", {})).startswith("Error: There are no directions")
    opened = []
    monkeypatch.setattr(mt, "_open_url", opened.append)
    _run(reg.call("directions", {"destination": "x"}))
    assert _run(reg.call("open_transit_directions", {})).startswith("Opening")
    assert opened and opened[0].startswith("https://www.google.com/maps/dir/")

    async def rev(cfg, lat, lon): return "Vitosha Boulevard in Sofia"
    monkeypatch.setattr(geo, "reverse_geocode", rev)
    out = _run(reg.call("where_am_i", {}))
    assert out.startswith("You're near Vitosha Boulevard in Sofia, from Windows location services, accurate to about 50 metres")


def test_transit_link_can_be_disabled(monkeypatch, _fresh):
    _patch_geo(monkeypatch)
    ctx.config.maps.show_transit_link = False
    out = _run(load_all().call("directions", {"destination": "x"}))
    assert "Google Maps" not in out and _fresh[-1]["transit_url"] == ""


def test_registration_and_safety():
    reg = load_all()
    for name in ("directions", "show_on_map", "close_map", "open_transit_directions", "where_am_i"):
        t = reg.get(name)
        assert t is not None and t.risk == "safe"
    assert reg.get("directions").parameters["required"] == ["destination"]


# ---- fast paths ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["close the map", "Hide the map.", "Jarvis, close map please", "dismiss the map"])
def test_fastpath_close_map(text):
    fp = match(text)
    assert fp and fp.kind == "map" and fp.action[:2] == ("call", "close_map")


@pytest.mark.parametrize("text", ["where am I", "Where am I?", "what's my location", "Where am I right now"])
def test_fastpath_where(text):
    fp = match(text)
    assert fp and fp.kind == "where_am_i" and fp.speak_result and fp.action[1] == "where_am_i"


@pytest.mark.parametrize("text", ["how do I get to the airport", "I want to go to Sofia Airport", "directions to the mall",
                                  "show me the airport on the map", "close the map app now"])
def test_no_fastpath_for_directions(text):
    assert match(text) is None


def test_system_prompt_mentions_maps():
    from jarvis.agent import SYSTEM_PROMPT

    assert "directions(destination=X)" in SYSTEM_PROMPT and "close_map" in SYSTEM_PROMPT
