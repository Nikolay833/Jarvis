# Jarvis orb

Desktop overlay for Jarvis: a glowing orb at the bottom-center of the screen,
one caption line under it, and Approve / Deny pills for risky actions. Tauri v2,
vanilla TypeScript, Vite. It is a client of the core's WebSocket event bus
(`ws://127.0.0.1:8765`, see `../docs/ARCHITECTURE.md`).

- Transparent, frameless, always on top, hidden from the taskbar, click-through.
- Invisible while `idle`; fades in on `listening`. No animation loop while hidden.
- The window accepts clicks only while a `confirm` is pending.
- `Ctrl+Alt+J` sends `{"type":"activate"}`.
- Reconnects to the core every 2 s, silently.
- Honors `prefers-reduced-motion` (no pulsing or orbiting; loudness becomes brightness).

## Map window

"Jarvis, I want to go to X" opens a full-screen HUD map (second Tauri window `map`, `map.html` + `src/map.ts`):
dark MapLibre basemap (CARTO dark-matter, recolored in `src/map-style.ts`), glowing car route (cyan) and walking
route (dashed blue), a side panel with Car / Walk / Public transport cards and turn-by-turn steps, and the orb
shrunk into the bottom-right corner. Esc or Close hides it. Public transport is only a Google Maps link (free
routing has no timetables). While it is open the small orb window is hidden (`set_map_visible` in `lib.rs`).
Events: `map_show` / `map_hide` from the core, `map_closed` / `map_open_transit` back (see the protocol table).

Without Rust: `node mock/mock-core.mjs --map`, then open `http://localhost:1420/map.html` (npm run dev) or the
preview build. Add `?basemap=off` for the no-tiles HUD, `?style=<url>` for another style JSON. `GET
http://127.0.0.1:8766/map?mode=all|none|transit` and `/maphide` replay the scenario (Sofia, NDK to the airport).

## Prerequisites

- Rust (stable, MSVC toolchain on Windows): https://rustup.rs
- Node.js 20 or newer
- Microsoft WebView2 runtime (preinstalled on Windows 11, and on current Windows 10)
- Visual Studio Build Tools with "Desktop development with C++" (needed by Rust on Windows)

## Run

```
cd orb
npm install
npm run tauri dev
```

## Build

```
npm run tauri build
```

The installer lands in `src-tauri/target/release/bundle/nsis/`.

## Frontend only (no Rust)

```
npm run build        # tsc + vite build
npm run dev          # http://localhost:1420 in a normal browser
node mock/mock-core.mjs   # fake core on ws://127.0.0.1:8765, loops through all states
```

Open the page with a dark background behind it (the page itself is transparent).
Pass `?ws=ws://host:port` to point at another bus.

## Layout

- `src/main.ts` wires bus events to the orb, caption and pills.
- `src/orb.ts` canvas renderer (eased level pulse, per-state looks).
- `src/ui.ts` caption line and confirm pills.
- `src/bus.ts`, `src/protocol.ts` WebSocket client and message types.
- `src/map.ts`, `src/map-style.ts`, `src/map.css`, `src/format.ts` the map window.
- `src-tauri/src/lib.rs` window placement (above the taskbar), commands `set_clickthrough` and
  `set_map_visible`, global hotkey.
- `mock/mock-core.mjs` dependency-free fake core for development.
