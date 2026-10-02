// Display formatting for the map panel (pure).

export function fmtDuration(seconds: number): string {
  const mins = Math.round(seconds / 60);
  if (mins < 1) return "1 min";
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

export function fmtDistance(meters: number): string {
  if (meters < 1000) return `${Math.max(10, Math.round(meters / 10) * 10)} m`;
  const km = meters / 1000;
  return km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`;
}

export function fmtCoords(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(4)}°${ns}  ${Math.abs(lon).toFixed(4)}°${ew}`;
}

const SOURCES: Record<string, string> = {
  windows: "your current location",
  home: "your saved home address",
  ip: "your approximate location",
};

/** "From your current location, about 30 m". */
export function fmtFrom(source: string, accuracyM: number): string {
  const what = SOURCES[source] ?? "your location";
  const acc = accuracyM >= 1 ? `, accurate to ${accuracyM < 1000 ? Math.round(accuracyM) + " m" : Math.round(accuracyM / 1000) + " km"}` : "";
  return `From ${what}${acc}`;
}
