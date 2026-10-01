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

## Safety

Reading, listing and opening things run at once. Deleting, moving, installing,
registry or system changes, and starting Claude Code jobs need your approval
(say "yes" or click Approve). No answer in 30 s means no.

## Roadmap

1. Desktop core and orb (this milestone).
2. Phone: ntfy push alerts and a voice web app over Tailscale.
3. Raspberry Pi Wake-on-LAN relay and Windows auto-login.
