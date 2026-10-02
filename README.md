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

## Talking over Jarvis (barge-in)

You can interrupt him mid-sentence: start talking while he speaks and he fades out, drops the rest of
the reply and listens, keeping the words you had already begun. No headset needed: the mic level is
compared with what he is playing, so his own voice from the speakers does not interrupt him (it learns
the speaker-to-mic coupling from "Online, sir."). Saying "Hey Jarvis" also interrupts. Config `[audio]`:
`barge_in`, `barge_in_sensitivity` (0..1; raise it if he does not stop, lower it if he stops himself),
`barge_in_min_ms`, `barge_in_wake`. Uses Silero VAD (`onnxruntime`, model file from openWakeWord);
`logs/jarvis.log` shows `barge-in detected (vad .., mic/echo ratio ..)`, and with `-v` also the
suppressed-as-echo frames.

## Speed

Replies stream into speech sentence by sentence, simple requests (time, date, "open chrome",
"lock the pc", "stop", volume) skip the LLM, and each turn logs a `latency:` line. `setup_windows.ps1`
sets the user environment variables `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`;
restart Ollama (quit it from the tray and reopen) for them to apply. Tunables are in
`config.example.toml` (`ollama.max_reply_tokens`, `ollama.keep_alive`, `audio.silence_seconds`,
`agent.fast_paths`, `agent.stream_replies`).

## Speech to text

Default: faster-whisper `large-v3`, beam 5, VAD with padding, vocabulary as prompt and hotwords
(`[whisper]` in config). Faster but less accurate: `model = "large-v3-turbo"`, `beam_size = 1`.
Alternative: NVIDIA Parakeet TDT 0.6B. Install with `pip install "onnx-asr[gpu,hub]"` (or run
`setup_windows.ps1 -Parakeet`), then set `[stt] engine = "parakeet"`.
Compare engines on your own voice and mic: `python -m jarvis.stt_bench` (records 4 phrases to
`logs/stt_bench/`, then prints engine / time / transcript per phrase) or
`python -m jarvis.stt_bench --files a.wav b.wav`.

## Voice

Jarvis speaks with a Jarvis-style British voice: Kokoro's own male voices plus a subtle processing chain
(light room, presence, faint digital sheen). No voice cloning. Settings in `[tts]`:

```toml
[tts]
voice = "bm_george"          # bm_george, bm_lewis, bm_daniel, bm_fable, or "bm_george:0.6,bm_lewis:0.4"
speed = 1.05
fx = "jarvis"                # or "none"
fx_amount = 0.35             # 0..1
```

Audition presets with `python -m jarvis.voices` (`--save DIR` writes wavs;
`--voice bm_lewis --fx jarvis --amount 0.4 --speed 1.05` tries a custom one).

## More commands

- Windows: "minimise Chrome", "maximise VS Code", "show the desktop", "focus Spotify", "close Notepad"
  (closing asks for approval and is graceful).
- Chrome: "search for best pizza", "google python tutorials", "open Chrome with my work profile" (profiles are
  read from Chrome's Local State; matched by name, Google name, email or folder).
- Music (Spotify desktop, free plan): "pause the music", "resume", "skip song", "previous song",
  "what's playing", "play <song> on Spotify". Free Spotify cannot be forced to play a chosen track, so
  "play <song>" is best effort: Jarvis opens the track, tries to press Play and tells you if it needs a click.
  Optional, for exact song lookup: create a free app at https://developer.spotify.com/dashboard (any redirect
  URI, no Premium) and put its client id and secret in the `[spotify]` section of `config.toml`.
  Optional `pip install pywinauto` (`pip install -e .[ui]`) lets Jarvis press Play itself.
- Claude chat: "ask Claude why the sky is blue", "new Claude chat about dinner", "what were my Claude chats".
  Uses your Claude subscription through the `claude` CLI in an empty folder, without file or shell tools.
  Coding in a project folder is still "Claude Code" (`claude_code_run`).
- Claude Code sessions (projects can be anywhere; Jarvis learns them from Claude's own history plus your Desktop and
  Documents folders): "open the login bug session in Jarvis", "continue where I left off in Jarvis", "new Claude
  session for the Jarvis project: fix the flaky test", "list my Claude sessions", "what is Claude doing?",
  "is Claude done?", "ask that session whether the tests pass and tell me". If several sessions match, Jarvis reads
  the top three and you answer "two".
- Claude announcements: Jarvis says "Sir, Claude finished in <project>: ..." or "Sir, Claude needs your permission
  in <project>." Install the hooks once (`setup_windows.ps1` does it):
  `.venv\Scripts\python scripts\install_claude_hooks.py` and remove them with `... --uninstall`. They edit
  `~/.claude/settings.json` (a timestamped backup is written first) and only send events to the local Jarvis.
  Tune in `[claude_watch]` in `config.toml` (`min_turn_seconds` skips short turns).
- Memory: "remember that my main project is Jarvis", "what do you remember", "forget my favourite colour". Jarvis also
  saves lasting preferences you state clearly ("I prefer Spotify", "call me Nikolay") and says so; he never stores
  passwords or keys. Facts live in `%APPDATA%\Jarvis\memory.json` (edit or delete it freely) and the relevant ones
  are shown to the model with each request.
- Timers and reminders: "set a timer for 10 minutes", "remind me in 20 minutes to call mom", "remind me to pay rent
  tomorrow at 9", "what reminders do I have", "cancel the timer". They survive restarts; anything that came due while
  Jarvis was off is announced when he starts. Settings in `[reminders]`.
- Briefing: "give me my briefing" (or "what's my day"): greeting, date, weather, today's reminders and what Claude Code
  did lately. It is also spoken automatically the first time you wake Jarvis each day after 05:00. Weather comes from
  Open-Meteo (free, no key) for the city in `[briefing]` (default Sofia; set `city`, `latitude`, `longitude`).
- Maps: "I want to go to Sofia Airport", "how do I get to X", "directions to X", "show me X on the map" open a
  full-screen HUD map with car and walking routes (time, distance, steps) and a "Public transport" card that opens
  Google Maps (no free timetable data exists, so Jarvis never guesses bus times). "Close the map" or Esc closes it;
  "where am I" says your location. Your position comes from Windows location services first (turn on Settings >
  Privacy & security > Location, including "Let desktop apps access your location"), then the saved home address
  (`[maps] home_address` or "remember that home is ..."), then your IP (city level). Data: OpenStreetMap
  (Nominatim, FOSSGIS OSRM) and CARTO basemap tiles; needs internet. See `orb/README.md`.

## Safety

Reading, listing and opening things run at once. Deleting, moving, installing,
registry or system changes, and starting Claude Code jobs need your approval
(say "yes" or click Approve). No answer in 30 s means no.

## Roadmap

1. Desktop core and orb (this milestone).
2. Phone: ntfy push alerts and a voice web app over Tailscale.
3. Raspberry Pi Wake-on-LAN relay and Windows auto-login.
