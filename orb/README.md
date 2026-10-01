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
- `src-tauri/src/lib.rs` window placement (above the taskbar), click-through
  command `set_clickthrough`, global hotkey.
- `mock/mock-core.mjs` dependency-free fake core for development.
