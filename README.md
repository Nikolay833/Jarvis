# Jarvis

Local voice assistant for Windows. Say "Hey Jarvis", a small glowing orb
appears, and Jarvis acts on your PC: opens apps and folders, reads files,
runs PowerShell, and drives Claude Code headlessly. Everything runs locally
(Ollama, faster-whisper, Kokoro) on your GPU.

Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup (Windows)

Prerequisites: Python 3.11+, [Ollama](https://ollama.com), NVIDIA driver
with CUDA 12.8+ support, and for the orb Rust, Node 20+ and WebView2.
Claude Code (`claude` on PATH) is optional, for Claude Code jobs.

```powershell
.\scripts\setup_windows.ps1          # venv, torch cu128, deps, pulls qwen3:14b
copy config.example.toml config.toml # then edit if needed
cd orb; npm install; npm run tauri build; cd ..
.\scripts\start_jarvis.ps1           # core + orb
.\scripts\start_jarvis.ps1 -RegisterAutostart   # start at login
```

Try it without a microphone: `python -m jarvis --text`.
Hotkey `Ctrl+Alt+J` starts listening without the wake word.

Jarvis plays a soft chime when it starts listening, says "Online, sir." when ready
(`audio.announce_ready`), and logs every stage with timings to the console and
`logs/jarvis.log`. If nothing happens, run `python -m jarvis --debug-audio` to see the
live mic level and wake score (twice a second).

## Speed

Replies stream into speech sentence by sentence, simple requests (time, date, "open chrome",
"lock the pc", "stop", volume) skip the LLM, and each turn logs a `latency:` line. `setup_windows.ps1`
sets the user environment variables `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`;
restart Ollama (quit it from the tray and reopen) for them to apply. Tunables are in
`config.example.toml` (`ollama.max_reply_tokens`, `ollama.keep_alive`, `audio.silence_seconds`,
`agent.fast_paths`, `agent.stream_replies`).

## Safety

Reading, listing and opening things run at once. Deleting, moving, installing,
registry or system changes, and starting Claude Code jobs need your approval
(say "yes" or click Approve). No answer in 30 s means no.

## Roadmap

1. Desktop core and orb (this milestone).
2. Phone: ntfy push alerts and a voice web app over Tailscale.
3. Raspberry Pi Wake-on-LAN relay and Windows auto-login.
