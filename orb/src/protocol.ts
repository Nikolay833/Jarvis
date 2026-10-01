// WebSocket event protocol, see docs/ARCHITECTURE.md ("Event bus protocol").

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

export type CoreMessage =
  | { type: "state"; state: OrbState }
  | { type: "level"; rms: number }
  | { type: "transcript"; text: string; final: boolean }
  | { type: "reply"; text: string }
  | { type: "confirm"; id: string; summary: string }
  | { type: "confirm_resolved"; id: string; approved: boolean }
  | { type: "job"; id: string; kind: string; status: "running" | "done" | "failed"; summary: string };

export type OrbMessage =
  | { type: "confirm_response"; id: string; approved: boolean }
  | { type: "text_input"; text: string }
  | { type: "activate" };

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
    case "job":
      return m as unknown as CoreMessage;
    default:
      return null;
  }
}
