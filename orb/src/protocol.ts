// WebSocket event protocol, see docs/ARCHITECTURE.md ("Event bus protocol").

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

export interface MapPoint {
  lat: number;
  lon: number;
  label: string;
  /** Nominatim bounding box [south, north, west, east]: how far to zoom for a place (city, country, street). */
  bbox?: [number, number, number, number];
}
export type MapControl =
  | { action: "zoom_in" | "zoom_out"; amount: number }
  | { action: "pan"; direction: "north" | "south" | "east" | "west"; amount: number }
  | { action: "reset" };
export interface MapOrigin extends MapPoint {
  accuracy: number; // metres
  source: "windows" | "home" | "ip" | string;
}
export interface MapStep {
  text: string;
  distance_m: number;
}
export interface MapRoute {
  mode: "car" | "walk";
  distance_m: number;
  duration_s: number;
  geometry: { type: "LineString"; coordinates: [number, number][] }; // [lon, lat]
  steps: MapStep[];
}
export interface MapShow {
  origin: MapOrigin | null;
  destination: MapPoint;
  routes: MapRoute[];
  transit_url: string; // "" = no public transport card
  focus: "car" | "walk" | "transit" | "";
}

export type CoreMessage =
  | { type: "state"; state: OrbState }
  | { type: "level"; rms: number }
  | { type: "transcript"; text: string; final: boolean }
  | { type: "reply"; text: string }
  | { type: "confirm"; id: string; summary: string }
  | { type: "confirm_resolved"; id: string; approved: boolean }
  | ({ type: "map_show" } & MapShow)
  | { type: "map_hide" }
  | ({ type: "map_control" } & MapControl)
  | { type: "job"; id: string; kind: string; status: "running" | "done" | "failed"; summary: string };

export type OrbMessage =
  | { type: "confirm_response"; id: string; approved: boolean }
  | { type: "text_input"; text: string }
  | { type: "activate" }
  | { type: "map_closed" }
  | { type: "map_open_transit" };

const STATES: readonly string[] = ["idle", "listening", "thinking", "speaking"];

/** Parse and loosely validate a core message. Returns null for anything unusable. */
export function parseCoreMessage(raw: unknown): CoreMessage | null {
  if (typeof raw !== "string") return null;
  let m: Record<string, unknown>;
  try {
    const v: unknown = JSON.parse(raw);
    if (typeof v !== "object" || v === null) return null;
    m = v as Record<string, unknown>;
  } catch {
    return null;
  }
  switch (m.type) {
    case "state":
      return typeof m.state === "string" && STATES.includes(m.state)
        ? { type: "state", state: m.state as OrbState }
        : null;
    case "level":
      return typeof m.rms === "number" && Number.isFinite(m.rms) ? { type: "level", rms: m.rms } : null;
    case "transcript":
      return typeof m.text === "string" ? { type: "transcript", text: m.text, final: m.final === true } : null;
    case "reply":
      return typeof m.text === "string" ? { type: "reply", text: m.text } : null;
    case "confirm":
      return typeof m.id === "string" && typeof m.summary === "string"
        ? { type: "confirm", id: m.id, summary: m.summary }
        : null;
    case "confirm_resolved":
      return typeof m.id === "string" ? { type: "confirm_resolved", id: m.id, approved: m.approved === true } : null;
    case "map_show":
      return parseMapShow(m);
    case "map_hide":
      return { type: "map_hide" };
    case "map_control":
      return parseMapControl(m);
    case "job":
      return m as unknown as CoreMessage;
    default:
      return null;
  }
}

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function point(v: unknown): MapPoint | null {
  if (typeof v !== "object" || v === null) return null;
  const p = v as Record<string, unknown>;
  if (!num(p.lat) || !num(p.lon) || Math.abs(p.lat) > 90 || Math.abs(p.lon) > 180) return null;
  const b = p.bbox;
  const bbox =
    Array.isArray(b) && b.length === 4 && b.every(num) && (b[0] as number) <= (b[1] as number)
      ? (b as [number, number, number, number])
      : undefined;
  return { lat: p.lat, lon: p.lon, label: typeof p.label === "string" ? p.label : "", ...(bbox ? { bbox } : {}) };
}

function parseMapControl(m: Record<string, unknown>): CoreMessage | null {
  const amount = num(m.amount) && m.amount > 0 ? Math.min(m.amount, 10) : 1;
  switch (m.action) {
    case "zoom_in":
    case "zoom_out":
      return { type: "map_control", action: m.action, amount };
    case "pan":
      return m.direction === "north" || m.direction === "south" || m.direction === "east" || m.direction === "west"
        ? { type: "map_control", action: "pan", direction: m.direction, amount }
        : null;
    case "reset":
      return { type: "map_control", action: "reset" };
    default:
      return null;
  }
}

function route(v: unknown): MapRoute | null {
  if (typeof v !== "object" || v === null) return null;
  const r = v as Record<string, unknown>;
  const g = r.geometry as { coordinates?: unknown } | undefined;
  if ((r.mode !== "car" && r.mode !== "walk") || !num(r.distance_m) || !num(r.duration_s)) return null;
  if (!Array.isArray(g?.coordinates) || g.coordinates.length < 2) return null;
  const coords = (g.coordinates as unknown[]).filter(
    (c): c is [number, number] => Array.isArray(c) && num(c[0]) && num(c[1]),
  );
  const steps = (Array.isArray(r.steps) ? r.steps : []).flatMap((s): MapStep[] => {
    const o = s as Record<string, unknown>;
    return typeof o?.text === "string" ? [{ text: o.text, distance_m: num(o.distance_m) ? o.distance_m : 0 }] : [];
  });
  return {
    mode: r.mode,
    distance_m: r.distance_m,
    duration_s: r.duration_s,
    geometry: { type: "LineString", coordinates: coords },
    steps,
  };
}

function parseMapShow(m: Record<string, unknown>): CoreMessage | null {
  const destination = point(m.destination);
  if (!destination) return null;
  const o = point(m.origin);
  const acc = (m.origin as Record<string, unknown> | null)?.accuracy;
  const src = (m.origin as Record<string, unknown> | null)?.source;
  const origin: MapOrigin | null = o ? { ...o, accuracy: num(acc) ? acc : 0, source: typeof src === "string" ? src : "" } : null;
  const routes = (Array.isArray(m.routes) ? m.routes : []).map(route).filter((r): r is MapRoute => r !== null);
  const focus = m.focus === "car" || m.focus === "walk" || m.focus === "transit" ? m.focus : "";
  return {
    type: "map_show",
    origin,
    destination,
    routes,
    transit_url: typeof m.transit_url === "string" ? m.transit_url : "",
    focus,
  };
}
