// HUD restyle of the CARTO dark-matter vector style, plus an offline "no basemap" style.
import type { LayerSpecification, StyleSpecification } from "maplibre-gl";

export const STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export const PALETTE = {
  bg: "#05070f", // page / sea of the HUD
  land: "#0c1d31", // dark teal-navy
  park: "#0a2331",
  water: "#040a19", // deep navy, darker than land so coasts read
  building: "#10304d",
  roadMinor: "rgba(88, 150, 255, 0.28)",
  roadMid: "rgba(112, 178, 255, 0.55)",
  roadMajor: "rgba(128, 214, 255, 0.85)",
  boundary: "rgba(98, 150, 240, 0.30)",
  labelMajor: "#a9c4e6",
  labelMinor: "#6f8db5",
};

const num = (stops: [number, number][]): unknown => [
  "interpolate",
  ["exponential", 1.5],
  ["zoom"],
  ...stops.flat(),
];
const W_MINOR = num([[10, 0.25], [13, 0.5], [16, 1.1], [19, 2.4]]);
const W_MID = num([[8, 0.3], [11, 0.6], [14, 1.0], [17, 2], [20, 4]]);
const W_MAJOR = num([[5, 0.4], [9, 0.8], [12, 1.3], [15, 2.2], [18, 4.5]]);

type Loose = {
  id: string;
  type: string;
  paint?: Record<string, unknown>;
  layout?: Record<string, unknown>;
};

const ROADISH = /road|street|bridge|tunnel|motorway|trunk|primary|secondary|tertiary|path|pedestrian|service|track|link|ramp|highway/i;

function restyleLayer(l: Loose): void {
  const id = l.id.toLowerCase();
  l.paint = { ...(l.paint ?? {}) };
  l.layout = { ...(l.layout ?? {}) };
  const hide = () => (l.layout!.visibility = "none");
  switch (l.type) {
    case "background":
      l.paint["background-color"] = PALETTE.land;
      break;
    case "fill":
      delete l.paint["fill-pattern"];
      l.paint["fill-antialias"] = true;
      if (/water/.test(id)) {
        l.paint["fill-color"] = PALETTE.water;
        l.paint["fill-opacity"] = 1;
      } else if (/building/.test(id)) {
        l.paint["fill-color"] = PALETTE.building;
        l.paint["fill-opacity"] = 0.45;
      } else if (/park|wood|forest|grass|landcover|nature|reserve/.test(id)) {
        l.paint["fill-color"] = PALETTE.park;
        l.paint["fill-opacity"] = 0.55;
      } else {
        l.paint["fill-color"] = PALETTE.land;
        l.paint["fill-opacity"] = 0.6;
      }
      l.paint["fill-outline-color"] = "rgba(0,0,0,0)";
      break;
    case "fill-extrusion":
      l.paint["fill-extrusion-color"] = PALETTE.building;
      l.paint["fill-extrusion-opacity"] = 0.45;
      break;
    case "line":
      if (/case|casing/.test(id)) {
        l.paint["line-opacity"] = 0; // casings are clutter on a HUD
      } else if (/boundary|admin/.test(id)) {
        l.paint["line-color"] = PALETTE.boundary;
        l.paint["line-opacity"] = 1;
      } else if (/waterway/.test(id)) {
        l.paint["line-color"] = PALETTE.water;
        l.paint["line-opacity"] = 0.9;
      } else if (/rail|transit|aeroway|ferry/.test(id)) {
        l.paint["line-color"] = PALETTE.boundary;
        l.paint["line-opacity"] = 0.5;
      } else if (ROADISH.test(id)) {
        const major = /motorway|trunk|primary/.test(id);
        const mid = /secondary|tertiary/.test(id);
        l.paint["line-color"] = major ? PALETTE.roadMajor : mid ? PALETTE.roadMid : PALETTE.roadMinor;
        l.paint["line-width"] = major ? W_MAJOR : mid ? W_MID : W_MINOR;
        l.paint["line-opacity"] = 1;
        l.paint["line-blur"] = major ? 0.4 : 0;
      } else {
        l.paint["line-color"] = PALETTE.boundary;
        l.paint["line-opacity"] = 0.6;
      }
      break;
    case "symbol": {
      const hasText = l.layout["text-field"] !== undefined;
      if (!hasText || /poi|housenumber|house_number|transit|rail|mountain_peak|aerodrome/.test(id)) {
        hide(); // icons, pois, house numbers: noise
        break;
      }
      const major = /country|state|city|capital|water|ocean|sea/.test(id);
      l.paint["text-color"] = major ? PALETTE.labelMajor : PALETTE.labelMinor;
      l.paint["text-halo-color"] = PALETTE.bg;
      l.paint["text-halo-width"] = 1.2;
      l.paint["text-halo-blur"] = 0.5;
      l.paint["text-opacity"] = major ? 0.95 : 0.8;
      break;
    }
    case "raster":
    case "hillshade":
    case "heatmap":
      hide();
      break;
  }
}

/** Recolor a fetched CARTO dark-matter style to the HUD palette (returns a copy). */
export function hudStyle(style: StyleSpecification): StyleSpecification {
  const copy = JSON.parse(JSON.stringify(style)) as StyleSpecification;
  copy.layers = (copy.layers as LayerSpecification[]).map((layer) => {
    restyleLayer(layer as unknown as Loose);
    return layer;
  });
  delete (copy as { sky?: unknown }).sky;
  return copy;
}

/** No tiles, no glyphs: just the HUD background. The grid, route and markers still draw on top. */
export function offlineStyle(): StyleSpecification {
  return {
    version: 8,
    name: "jarvis-hud-offline",
    sources: {},
    layers: [{ id: "background", type: "background", paint: { "background-color": PALETTE.land } }],
  };
}
