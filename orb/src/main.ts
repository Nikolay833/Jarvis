import "@fontsource-variable/geist";
import "./style.css";
import { Bus } from "./bus";
import { Orb } from "./orb";
import type { CoreMessage, OrbState } from "./protocol";
import { onHotkey, setClickthrough } from "./tauri";
import { Caption, ConfirmPills } from "./ui";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const orb = new Orb($<HTMLCanvasElement>("orb"));
const caption = new Caption($("caption"));

let state: OrbState = "idle";

const bus = new Bus(
  new URLSearchParams(location.search).get("ws") ?? "ws://127.0.0.1:8765",
  handle,
  () => {
    // Core went away: drop everything and go invisible.
    pills.hide();
    setState("idle");
  },
);

const pills = new ConfirmPills(
  $("actions"),
  $<HTMLButtonElement>("approve"),
  $<HTMLButtonElement>("deny"),
  (id, approved) => bus.send({ type: "confirm_response", id, approved }),
  (visible) => {
    orb.setForceVisible(visible);
    void setClickthrough(!visible); // window takes clicks only while pills show
    if (!visible && state === "idle") caption.fadeOut();
  },
);

function setState(next: OrbState): void {
  const prev = state;
  state = next;
  orb.setState(next);
  if (next === "listening" && prev !== "listening" && !pills.pending) caption.clear();
  if (next === "thinking") caption.dim(true);
  else caption.dim(false);
  if (next === "idle" && !pills.pending) caption.fadeOut();
}

function handle(m: CoreMessage): void {
  switch (m.type) {
    case "state":
      setState(m.state);
      break;
    case "level":
      orb.setLevel(m.rms);
      break;
    case "transcript":
      if (state !== "speaking") caption.transcript(m.text, m.final);
      break;
    case "reply":
      caption.reply(m.text);
      break;
    case "confirm":
      caption.summary(m.summary);
      pills.show(m.id);
      break;
    case "confirm_resolved":
      pills.resolved(m.id);
      break;
    case "map_show":
    case "map_hide":
    case "map_control":
      break; // drawn by the separate map window
    case "job":
      break; // background jobs are spoken by the core; nothing to draw
  }
}

onHotkey(() => bus.send({ type: "activate" }));
bus.start();
