import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "maplibre-gl/dist/maplibre-gl.css";
import "./map.css";
import { Map as MapLibreMap, Marker, setWorkerUrl } from "maplibre-gl";
import type { GeoJSONSource, LngLatBoundsLike } from "maplibre-gl";
import { Bus } from "./bus";
import { fmtDistance, fmtDuration } from "./format";
import { hudStyle, offlineStyle, STYLE_URL } from "./map-style";
import type { CoreMessage, MapControl, MapRoute, MapShow } from "./protocol";
import { setMapVisible } from "./tauri";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
// maplibre 6 looks for its worker file next to its own module, which a bundle does not keep: vite.config.ts
// serves/emits it (and its shared chunk) under /maplibre/.
setWorkerUrl(new URL("./maplibre/maplibre-gl-worker.mjs", location.href).href);
const params = new URLSearchParams(location.search);
const reduceQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

const ROUTE_COLOR = { car: "#5ff0ff", walk: "#8fb2ff" } as const;
const MODE_LABEL = { car: "Car", walk: "Walk" } as const;
const DRAW_MS = 1000;

// ---- state --------------------------------------------------------------------------------------------
let isOpen = false;
let map: MapLibreMap | null = null;
let mapReady: Promise<MapLibreMap> | null = null;
let basemap = true;
let current: MapShow | null = null;
let selected: "car" | "walk" = "car";
let markers: Marker[] = [];
let drawRaf = 0;

const bus = new Bus(params.get("ws") ?? "ws://127.0.0.1:8765", handle, () => {});

function handle(m: CoreMessage): void {
  switch (m.type) {
    case "map_show":
      void show(m);
      break;
    case "map_hide":
      void close(false);
      break;
    case "map_control":
      control(m);
      break;
  }
}

// ---- map instance -------------------------------------------------------------------------------------
async function loadStyle(): Promise<ReturnType<typeof offlineStyle>> {
  if (params.get("basemap") === "off") return offlineStyle();
  try {
    const ctl = new AbortController();
    const timer = window.setTimeout(() => ctl.abort(), 6000);
    const res = await fetch(params.get("style") ?? STYLE_URL, { signal: ctl.signal }); // ?style= is for offline previews
    window.clearTimeout(timer);
    if (!res.ok) throw new Error(`style ${res.status}`);
    return hudStyle(await res.json());
  } catch (e) {
    console.warn("basemap unavailable, using the offline HUD style", e);
    return offlineStyle();
  }
}

function ensureMap(): Promise<MapLibreMap> {
  mapReady ??= (async () => {
    const style = await loadStyle();
    basemap = Object.keys(style.sources ?? {}).length > 0;
    $("attrib").textContent = basemap ? "© OpenStreetMap contributors © CARTO" : "Basemap unavailable offline";
    const m = new MapLibreMap({
      container: "map",
      style,
      center: [23.3219, 42.6977],
      zoom: 11,
      attributionControl: false,
      pitchWithRotate: false,
      dragRotate: false,
      fadeDuration: reduceQuery.matches ? 0 : 200,
    });
    m.touchZoomRotate.disableRotation();
    await new Promise<void>((resolve) => {
      if (m.isStyleLoaded()) return resolve();
      m.once("style.load", () => resolve());
      window.setTimeout(resolve, 8000);
    });
    map = m;
    if (params.has("debug")) (window as unknown as { __map?: MapLibreMap }).__map = m; // for screenshots/tests
    return m;
  })();
  return mapReady;
}

// ---- drawing ------------------------------------------------------------------------------------------
// Minimal GeoJSON shapes (no @types/geojson dependency).
type Feature = { type: "Feature"; properties: Record<string, never>; geometry: { type: string; coordinates: unknown } };
type FeatureCollection = { type: "FeatureCollection"; features: Feature[] };

function circlePolygon(lat: number, lon: number, radiusM: number): Feature {
  const ring: [number, number][] = [];
  const dLat = radiusM / 111_320;
  const dLon = radiusM / (111_320 * Math.max(Math.cos((lat * Math.PI) / 180), 0.05));
  for (let i = 0; i <= 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    ring.push([lon + Math.cos(a) * dLon, lat + Math.sin(a) * dLat]);
  }
  return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } };
}

function line(coords: [number, number][]): Feature {
  return { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } };
}

const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

function clearOverlays(m: MapLibreMap): void {
  cancelAnimationFrame(drawRaf);
  for (const mk of markers) mk.remove();
  markers = [];
  for (const id of ["car-glow", "car-core", "walk-glow", "walk-core", "accuracy-fill", "accuracy-line"]) {
    if (m.getLayer(id)) m.removeLayer(id);
  }
  for (const id of ["car", "walk", "accuracy"]) if (m.getSource(id)) m.removeSource(id);
}

function addRouteLayers(m: MapLibreMap, mode: "car" | "walk"): void {
  const color = ROUTE_COLOR[mode];
  m.addSource(mode, { type: "geojson", data: EMPTY });
  // two stacked lines: a wide soft one for the glow, a thin bright one on top
  m.addLayer({
    id: `${mode}-glow`,
    type: "line",
    source: mode,
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": color, "line-width": mode === "car" ? 11 : 7, "line-blur": mode === "car" ? 7 : 5, "line-opacity": 0.38 },
  });
  m.addLayer({
    id: `${mode}-core`,
    type: "line",
    source: mode,
    layout: { "line-cap": mode === "car" ? "round" : "butt", "line-join": "round" },
    paint: {
      "line-color": mode === "car" ? "#b9fbff" : color,
      "line-width": mode === "car" ? 2.6 : 2.2,
      "line-opacity": 1,
      ...(mode === "walk" ? { "line-dasharray": [1.6, 1.6] } : {}),
    },
  });
}

function markerEl(cls: string, html: string): HTMLElement {
  const el = document.createElement("div");
  el.className = cls;
  el.innerHTML = html;
  return el;
}
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);

function highlight(m: MapLibreMap): void {
  for (const mode of ["car", "walk"] as const) {
    if (!m.getLayer(`${mode}-core`)) continue;
    const on = mode === selected;
    m.setPaintProperty(`${mode}-core`, "line-opacity", on ? 1 : 0.3);
    m.setPaintProperty(`${mode}-glow`, "line-opacity", on ? 0.4 : 0.08);
    if (on) {
      m.moveLayer(`${mode}-glow`);
      m.moveLayer(`${mode}-core`);
    }
  }
}

function animateRoutes(m: MapLibreMap, routes: MapRoute[]): void {
  const full = () => {
    for (const r of routes) (m.getSource(r.mode) as GeoJSONSource | undefined)?.setData(line(r.geometry.coordinates));
  };
  if (reduceQuery.matches) return full();
  const t0 = performance.now();
  const step = (now: number) => {
    const t = Math.min(1, (now - t0) / DRAW_MS);
    const k = 1 - Math.pow(1 - t, 3); // ease-out
    for (const r of routes) {
      const n = Math.max(2, Math.ceil(r.geometry.coordinates.length * k));
      (m.getSource(r.mode) as GeoJSONSource | undefined)?.setData(line(r.geometry.coordinates.slice(0, n)));
    }
    if (t < 1) drawRaf = requestAnimationFrame(step);
  };
  drawRaf = requestAnimationFrame(step);
}

/** Keep the route/place clear of the strip (bottom-left) and the faded edges. */
function fitPadding(): { top: number; bottom: number; left: number; right: number } {
  const h = window.innerHeight;
  const w = window.innerWidth;
  return { top: Math.round(h * 0.2), bottom: Math.round(h * 0.3), left: Math.round(w * 0.14), right: Math.round(w * 0.14) };
}

function fit(m: MapLibreMap, msg: MapShow): void {
  const padding = fitPadding();
  const duration = reduceQuery.matches ? 0 : 1300;
  const d = msg.destination;
  if (msg.routes.length === 0 && !msg.origin) {
    // a place: fly to it and zoom to its own extent (country ~5, city ~10, street/POI ~16)
    if (d.bbox) {
      const [s, n, w, e] = d.bbox;
      // fit the extent, then come in a little: an administrative bbox is generous (London fits at ~9, a city looks right at ~10)
      const cam = m.cameraForBounds([[w, s], [e, n]] as LngLatBoundsLike, { padding, maxZoom: 16.5 });
      if (cam?.center && cam.zoom !== undefined) {
        m.easeTo({ center: cam.center, zoom: Math.min(cam.zoom + 1, 16.5), duration });
      } else {
        m.flyTo({ center: [d.lon, d.lat], zoom: 12, duration });
      }
    } else {
      m.flyTo({ center: [d.lon, d.lat], zoom: 14.5, duration });
    }
    return;
  }
  const pts: [number, number][] = [[d.lon, d.lat]];
  if (msg.origin) pts.push([msg.origin.lon, msg.origin.lat]);
  for (const r of msg.routes) pts.push(...r.geometry.coordinates);
  const b = pts.reduce(
    (acc, p) => [Math.min(acc[0], p[0]), Math.min(acc[1], p[1]), Math.max(acc[2], p[0]), Math.max(acc[3], p[1])],
    [180, 90, -180, -90],
  );
  m.fitBounds([[b[0], b[1]], [b[2], b[3]]] as LngLatBoundsLike, { padding, duration, maxZoom: 16 });
}

/** Geographic point at the middle of the free (unpadded) area: zooming around it keeps the shown place in view. */
function viewCenter(m: MapLibreMap) {
  const p = fitPadding();
  return m.unproject([(p.left + window.innerWidth - p.right) / 2, (p.top + window.innerHeight - p.bottom) / 2]);
}

/** Voice zoom/pan/reset from the core (tool map_control). */
function control(c: MapControl): void {
  if (!map || !isOpen) return;
  const duration = reduceQuery.matches ? 0 : 600;
  switch (c.action) {
    case "zoom_in":
      map.easeTo({ zoom: map.getZoom() + c.amount, around: viewCenter(map), duration });
      break;
    case "zoom_out":
      map.easeTo({ zoom: Math.max(map.getZoom() - c.amount, 1), around: viewCenter(map), duration });
      break;
    case "pan": {
      const dx = c.direction === "east" ? 1 : c.direction === "west" ? -1 : 0;
      const dy = c.direction === "south" ? 1 : c.direction === "north" ? -1 : 0;
      const k = 0.4 * c.amount;
      map.panBy([dx * k * window.innerWidth, dy * k * window.innerHeight], { duration });
      break;
    }
    case "reset":
      if (current) fit(map, current);
      break;
  }
}

// ---- strip --------------------------------------------------------------------------------------
function renderPanel(msg: MapShow): void {
  $("dest").textContent = msg.destination.label || "Destination";
  $("dest").title = msg.destination.label;
  $("note").hidden = msg.origin?.source !== "ip"; // only worth saying when the start point is a rough guess

  const modes = $("modes");
  modes.replaceChildren();
  for (const r of msg.routes) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "opt";
    b.dataset.mode = r.mode;
    b.setAttribute("role", "radio");
    b.innerHTML =
      `<span class="m">${MODE_LABEL[r.mode]}</span><span class="t">${esc(fmtDuration(r.duration_s))}</span>` +
      `<span class="d">${esc(fmtDistance(r.distance_m))}</span>`;
    b.addEventListener("click", () => select(r.mode));
    modes.append(b);
  }
  $("open-transit").hidden = !msg.transit_url;
  $("row").hidden = msg.routes.length === 0 && !msg.transit_url;
  $("steps-toggle").hidden = msg.routes.length === 0;
  setSteps(false);

  const first = msg.routes.find((r) => r.mode === msg.focus) ?? msg.routes.find((r) => r.mode === "car") ?? msg.routes[0];
  if (first) select(first.mode);
  else renderSteps(null);
}

function select(mode: "car" | "walk"): void {
  selected = mode;
  for (const b of document.querySelectorAll<HTMLButtonElement>("#modes .opt")) {
    const on = b.dataset.mode === mode;
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
  }
  renderSteps(current?.routes.find((r) => r.mode === mode) ?? null);
  if (map) highlight(map);
}

function renderSteps(r: MapRoute | null): void {
  const list = $("steps");
  list.replaceChildren();
  if (!r) return;
  r.steps.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="txt" title="${esc(s.text)}">${esc(s.text)}</span><span class="dist">${s.distance_m > 0 ? esc(fmtDistance(s.distance_m)) : ""}</span>`;
    list.append(li);
  });
  list.scrollTop = 0;
}

/** Steps are collapsed by default; the "Steps" text toggles them. */
function setSteps(open: boolean): void {
  $("steps").hidden = !open;
  const t = $("steps-toggle");
  t.setAttribute("aria-expanded", String(open));
  t.textContent = open ? "Hide steps" : "Steps";
}

// ---- show / close -------------------------------------------------------------------------------------
async function show(msg: MapShow & { type?: string }): Promise<void> {
  current = msg;
  isOpen = true;
  document.body.classList.add("open");
  renderPanel(msg);
  await setMapVisible(true);
  const m = await ensureMap();
  if (!isOpen || current !== msg) return; // closed or replaced while the style was loading
  m.resize();
  clearOverlays(m);

  if (msg.origin && msg.origin.accuracy > 40 && msg.origin.accuracy < 20_000) {
    m.addSource("accuracy", { type: "geojson", data: circlePolygon(msg.origin.lat, msg.origin.lon, msg.origin.accuracy) });
    m.addLayer({ id: "accuracy-fill", type: "fill", source: "accuracy", paint: { "fill-color": "#4a86ff", "fill-opacity": 0.1 } });
    m.addLayer({ id: "accuracy-line", type: "line", source: "accuracy", paint: { "line-color": "#4a86ff", "line-opacity": 0.45, "line-width": 1 } });
  }
  for (const r of msg.routes) addRouteLayers(m, r.mode);
  highlight(m);

  if (msg.origin) {
    markers.push(
      new Marker({ element: markerEl("mk-origin", '<i class="pulse"></i><i class="dot"></i>'), anchor: "center" })
        .setLngLat([msg.origin.lon, msg.origin.lat])
        .addTo(m),
    );
  }
  markers.push(
    new Marker({
      element: markerEl("mk-dest", `<span class="label">${esc(msg.destination.label)}</span><i class="ring"></i><i class="dot"></i>`),
      anchor: "center",
    })
      .setLngLat([msg.destination.lon, msg.destination.lat])
      .addTo(m),
  );
  fit(m, msg);
  animateRoutes(m, msg.routes);
}

async function close(notifyCore: boolean): Promise<void> {
  if (!isOpen) return;
  isOpen = false;
  current = null;
  document.body.classList.remove("open");
  cancelAnimationFrame(drawRaf);
  if (notifyCore) bus.send({ type: "map_closed" });
  await setMapVisible(false);
}

// ---- wiring -------------------------------------------------------------------------------------------
$("close").addEventListener("click", () => void close(true));
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isOpen) {
    e.preventDefault();
    void close(true);
  }
});
$("steps-toggle").addEventListener("click", () => setSteps($("steps").hidden));
$("open-transit").addEventListener("click", () => bus.send({ type: "map_open_transit" }));

// roving radio group: arrows switch between Car and Walk
$("modes").addEventListener("keydown", (e) => {
  if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key) || !current) return;
  const modes = current.routes.map((r) => r.mode);
  if (modes.length < 2) return;
  e.preventDefault();
  const next = modes[(modes.indexOf(selected) + 1) % modes.length];
  select(next);
  document.querySelector<HTMLButtonElement>(`#modes .opt[data-mode="${next}"]`)?.focus();
});

bus.start();
