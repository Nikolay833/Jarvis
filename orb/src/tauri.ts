// Thin wrappers so the same frontend also runs in a plain browser (dev, tests).

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export const inTauri = "__TAURI_INTERNALS__" in window;

/** true = clicks pass through the window; false = window takes clicks. */
export async function setClickthrough(ignore: boolean): Promise<void> {
  if (!inTauri) return;
  try {
    await invoke("set_clickthrough", { ignore });
  } catch (e) {
    console.warn("set_clickthrough failed", e);
  }
}

/** Called when the global hotkey (Ctrl+Alt+J) fires. */
export function onHotkey(cb: () => void): void {
  if (!inTauri) return;
  void listen("hotkey", cb);
}

/** Show/hide the transparent map overlay window (Rust sizes and centers it; the orb window stays). */
export async function setMapVisible(visible: boolean): Promise<void> {
  if (!inTauri) return;
  try {
    await invoke("set_map_visible", { visible });
  } catch (e) {
    console.warn("set_map_visible failed", e);
  }
}
