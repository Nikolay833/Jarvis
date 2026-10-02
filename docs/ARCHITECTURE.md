# Jarvis architecture

Jarvis is a local voice assistant for a Windows 11/10 PC. It listens for
"Hey Jarvis", shows a small glowing orb, understands the request with a local
LLM, and acts on the PC through tools. It can also drive Claude Code headlessly.

Milestone 1 (this code) covers the desktop only. Phone alerts (ntfy), the
phone voice web app (over Tailscale) and the Raspberry Pi Wake-on-LAN relay
come in later milestones. Keep module boundaries so they can plug in.

## Hardware target

- Windows 11/10, NVIDIA RTX 5070 Ti (16 GB VRAM).
- Everything runs locally. No paid APIs except Claude Code itself.

## Components

```
microphone ──► wake word ──► record until silence ──► STT ──► agent ──► TTS ──► speakers
                (openWakeWord   (webrtcvad/energy)   (faster-   (Ollama   (Kokoro,
                 "hey_jarvis")                        whisper)   tools)    bm_george)
                                       │
                                       ▼
                         WebSocket event bus ws://127.0.0.1:8765
                                       │
                                       ▼
                         orb/ (Tauri overlay window)
```

| Part | Library | Notes |
|---|---|---|
| Wake word | `openwakeword` | Pretrained `hey_jarvis` model. CPU. |
| Speech to text | `faster-whisper` (default) or `onnx-asr` Parakeet | `[stt] engine`. Whisper `large-v3`, beam 5, float16 on CUDA; fallback `small` on CPU. Parakeet TDT 0.6B v2 via onnxruntime (optional). Engines in `audio/stt.py`; `python -m jarvis.stt_bench` compares them. |
| LLM | Ollama HTTP API (`/api/chat`) | Default model `qwen3:14b`. Native tool calling. |
| Text to speech | `kokoro` | Voice `bm_george` (British male), 24 kHz. |
| Audio I/O | `sounddevice` | One persistent 16 kHz mono `MicStream` (`audio/mic.py`) opened at startup; wake word and recorder both read from it, with a 1.5 s pre-roll ring. |
| Event bus | `websockets` | Python core is the server. Orb is the client. |
| Overlay | Tauri v2 | Transparent, frameless, always on top, click-through when idle. |

## Repository layout

```
jarvis/                 Python package (the core)
  __main__.py           `python -m jarvis` entry point
  config.py             Loads config.toml (falls back to defaults)
  bus.py                WebSocket event server
  audio/                wakeword.py, recorder.py, stt.py (whisper + parakeet engines), tts.py, bargein.py
  stt_bench.py          `python -m jarvis.stt_bench`: compare STT engines on your recordings
  llm.py                Ollama chat client with tool calling (non-streaming `chat`, streaming `chat_stream`)
  fastpath.py           LLM-free answers for time/date/open app/lock/stop/volume, minimise/maximise, music keys, "play X on spotify", "search for X" (strict anchored regexes; "close X" is never a fast path)
  agent.py              Conversation loop: messages, tool calls, confirmations
  safety.py             Risk classification of tool calls
  paths.py              Path resolution (~, env vars, Desktop/Documents/... incl. OneDrive-redirected)
  claude_sessions.py    Claude Code sessions from ~/.claude/projects transcripts (parse, project resolve, rank)
  claude_watch.py       Hook events -> spoken announcements (throttle, pending permission)
  claude_hook.py        Stdlib-only Claude Code hook: forwards Stop/Notification to the bus
  store.py              %APPDATA%/Jarvis JSON files (atomic writes) and state.json
  memory.py             Long-term facts about the user, dedupe, relevance, "[Known about the user: ...]" block
  timeparse.py          Natural durations and times ("in 20 minutes", "tomorrow at 9")
  reminders.py          Persistent reminders/timers + asyncio scheduler (missed ones announced at start)
  briefing.py           Morning briefing (Open-Meteo weather, reminders, Claude sessions), first-wake-of-day logic
  geo.py                Location (Windows location -> saved home -> IP), Nominatim geocoding, OSRM routing, step texts, spoken rounding
  extras.py             Glue: memory block into the agent, scheduler start, automatic briefing hook
  tools/                Tool registry + tools (system, files, apps, windows, chrome, spotify, claude_code,
                        claude_history, claude_chat, claude_sessions_tools, claude_terminal, memory_tools,
                        reminder_tools, briefing_tool, maps_tools)
scripts/                setup_windows.ps1, start_jarvis.ps1, install_claude_hooks.py
tests/                  pytest, no hardware or network needed
orb/                    Tauri v2 overlay app
config.example.toml     Copy to config.toml
```

## Event bus protocol

JSON messages over `ws://127.0.0.1:8765`. Every message has a `type` field.

Core to orb:

| type | fields | meaning |
|---|---|---|
| `state` | `state`: `idle` \| `listening` \| `thinking` \| `speaking` | Orb shows/hides and animates. `idle` means fade out. |
| `level` | `rms`: float 0..1 | Mic or TTS loudness, about 30 per second, drives orb pulse. |
| `transcript` | `text`, `final`: bool | Live user words. |
| `reply` | `text` | Jarvis reply sentence (shown as one caption line). |
| `confirm` | `id`, `summary` | Risky action waiting for approval. |
| `confirm_resolved` | `id`, `approved`: bool | Approval answered (by voice or click). |
| `job` | `id`, `kind`, `status`: `running` \| `done` \| `failed`, `summary` | Background job update (Claude Code runs). |
| `map_show` | `origin`: `{lat, lon, label, accuracy, source: windows\|home\|ip}` or null, `destination`: `{lat, lon, label}`, `routes`: `[{mode: car\|walk, distance_m, duration_s, geometry: GeoJSON LineString [lon,lat], steps: [{text, distance_m}]}]`, `transit_url` ("" = no card), `focus`: `car\|walk\|transit\|""` | Open the full-screen map window. `routes` empty = just a place. |
| `map_hide` | none | Close the map window (also echoed by the core after `map_closed`, so the orb window learns the map is gone). |

Orb to core:

| type | fields | meaning |
|---|---|---|
| `confirm_response` | `id`, `approved`: bool | User clicked approve/deny. |
| `text_input` | `text` | Typed request instead of voice. |
| `activate` | none | Hotkey/click: start listening without wake word. |
| `map_closed` | none | The map window closed itself (Esc or Close button). |
| `map_open_transit` | none | "Open in Google Maps" button: the core opens the last transit URL in the default browser. |
| `claude_event` | `event` (`Stop`/`Notification`), `session_id`, `cwd`, `notification_type`, `message`, `last_assistant_message`, `transcript_path` | From `jarvis.claude_hook` (any local client): Jarvis announces it. |

## Audio flow and feedback

- The mic stream never closes between wake word and recording, so speech right after
  "Hey Jarvis" is kept. Recording starts with the audio after the detection frame; the
  `EndpointDetector` runs in `after_wake` mode (no calibration from the first blocks, noise
  floor from pre-roll, capped). Queued audio is flushed before wake listening resumes
  (no self-hearing of TTS) and before hotkey/confirmation recordings.
- Feedback: chime on wake/hotkey (its energy is ignored for endpointing), "Online, sir." at
  startup, spoken apologies for empty transcripts ("Sorry sir, I didn't catch that.") and failed turns.
- Startup warm-up in parallel: Ollama model load (empty `/api/chat`, same `num_ctx`/`keep_alive`),
  the STT engine on 1 s silence, Kokoro "Ready.". Ollama unreachable gives a WARNING and a spoken hint.
- Barge-in (`audio/bargein.py`, `[audio] barge_in`): the user can talk over Jarvis. During playback the speaker's
  hooks start a `BargeInMonitor` thread that reads the mic through `MicStream.subscribe()` (a side queue, so the wake
  thread and recorder are unaffected). Per 32 ms frame: Silero VAD probability (onnxruntime, the `silero_vad.onnx`
  in openwakeword's `resources/models`; inputs `input` [1,512] f32, `sr` i64, `h`/`c` [2,1,64]; energy VAD
  fallback) and the echo gate (`EchoGate`, pure): mic RMS must exceed `margin` (2.5x at sensitivity 0.5) times the
  expected echo = learned coupling * max playback RMS over the last 150 ms (`PlaybackTap`: the speaker reports each
  block it writes, with its audible time = write time + device latency). Coupling = p90 of mic/playback RMS over
  non-speech frames (`EchoCalibrator`; the startup "Online, sir." calibrates it, it keeps adapting). VAD must stay
  high for `barge_in_min_ms` (gaps up to 100 ms tolerated). "Hey Jarvis" during playback always counts
  (`barge_in_wake`, only while the wake thread is not using the model).
  On trigger (`Assistant._barge_cb`, monitor thread): `speaker.interrupt()` fades out 30 ms and stops, the speaker
  drops the rest of the reply (muted until `begin_turn`; confirmation prompts use `force=True` and still speak), the
  agent turn keeps running silently and is awaited before the next turn (`_settle_interrupted`), which also rewrites
  the last assistant message in history to the words actually spoken + `[interrupted by user]`. The next recording
  starts from `MicStream.grab_since(onset)` = the pre-roll ring from ~300 ms before the speech was first detected,
  passed to `Recorder.record_blocking(prefix=...)` (counts as speech), then the usual endpointing, STT and the next
  turn in the same conversation, no wake word. A barge-in while a confirmation is pending does not mute: the
  confirmation's own voice loop records it and the transcript resolves it. Barge-in during an out-of-turn
  announcement queues an `activate` request (source `barge`). The chime and Jarvis's own voice never trigger it
  (the monitor only runs while TTS plays; the echo gate handles the speaker-to-mic leak).
- Logs: INFO per stage with timings (wake score, stt, llm steps, reply, tts first audio) to the
  console and `logs/jarvis.log` (rotating 1 MB x 3). `--debug-audio` prints mic RMS and wake score.
- Latency (target: first audio < 1.5 s after end of speech for simple requests):
  - The system prompt is fully static so Ollama's KV cache keeps system prompt + tool schemas
    between turns. The date/time is a `[Now: ...]` line prefixed to the newest user message on the
    wire only; history is stored without it.
  - `OllamaClient.chat_stream` (NDJSON, `<think>` stripped across chunk boundaries). The agent pushes each
    finished sentence of a plain reply to `speaker.start_stream()` (`push(sentence)`, `await finish()`);
    Kokoro synthesizes sentence N+1 while N plays. Tool steps are never spoken (unfinished text is
    dropped). Streaming errors fall back to non-streaming `chat`.
  - `fastpath.match` runs before the agent (`agent.fast_paths`); fast-path turns are still added to history.
  - Replies are short by prompt and capped by `ollama.max_reply_tokens` (num_predict); `keep_alive = "-1"`
    keeps the model loaded; `audio.silence_seconds = 0.7` ends recording sooner.
  - Each turn logs `latency: stt .. s, llm first token .. s, first audio .. s, total .. s`
    (measured from the end of speech, after endpointing silence).

## Tools and tool-call robustness

Files: `list_dir`, `read_file`, `search_files`, `open_path`, `create_folder` (safe), `write_file` (safe when
creating a new file; risky when overwriting or writing a program/script extension), `delete_path`, `move_path`.
All file tools resolve paths with `paths.resolve_path`. `claude_code_history(folder, count)` reads the local
Claude Code transcripts in `~/.claude/projects` (safe).

- Tool calls written as text (`<tool_call>{...}</tool_call>`, fenced or bare JSON with name+arguments) are
  recovered in `llm.finalize_response` / `Agent._recover_calls` (known tool names only). While streaming,
  `ToolCallStreamFilter` keeps such text from being spoken.
- Promise guard: a reply with no tool call that announces an action ("I'll check ...") gets a hidden
  `(system)` nudge and the loop continues (max 2 per turn). Nudge exchanges are dropped from history.
- Each turn logs the tools it executed.

## Windows, Chrome, Spotify, Claude chat tools

- `tools/windows.py`: `list_windows`, `window_action(app, action)` (minimize|maximize|restore|focus|close),
  `minimize_all` (Win+M). ctypes user32 only (EnumWindows, ShowWindow, SetForegroundWindow with the Alt-key
  trick, PostMessage WM_CLOSE: graceful, never a kill). Skips invisible, cloaked, tool, shell and Jarvis's own
  windows. `match_windows` is pure: tier 1 process name equals the apps.py alias exe, tier 2 exe name contains the
  query, tier 3 title contains it. `close` is risky ("close all Chrome windows" confirmation); the rest are safe.
- `tools/chrome.py`: `chrome_profiles` reads `%LOCALAPPDATA%\Google\Chrome\User Data\Local State`
  (`profile.info_cache`); `open_chrome(profile, url, search)` fuzzy-matches profile by name, Google name, email or
  folder, finds chrome.exe (registry App Paths, then Program Files) and runs it with `--profile-directory`.
  Unknown profile: the error lists the profiles that exist.
- `tools/spotify.py` (Spotify FREE, desktop app): `music_control` sends media keys (play/pause, next, previous,
  stop), `spotify_now_playing` reads the Spotify window title ("Artist - Song"), `spotify_play(query)` opens
  `spotify:track:<id>` (id from the Web API Client Credentials search when `[spotify]` client id/secret are set,
  no Premium or user login needed) or `spotify:search:<query>`, focuses Spotify, tries to press Play through UI
  Automation (optional `pywinauto`, extra `ui`) else one Enter key, then watches the title for ~4 s. Free plan
  limit: Spotify offers no supported way to force playback of a chosen track, so this is best effort and the reply
  says when "playback may need a click".
- `tools/claude_chat.py`: normal chat with Claude via `claude -p --output-format json` (subscription, not coding).
  Runs in the empty folder `%APPDATA%\Jarvis\claude-chat`, message on stdin, `--disallowedTools
  Bash,Edit,Write,MultiEdit,NotebookEdit`, `--allowedTools WebSearch,WebFetch`, a short "voice assistant" appended
  system prompt. Sessions in `%APPDATA%\Jarvis\claude_chats.json` (name -> session_id, created, last_used, cwd,
  started); no name continues the chat used in the last 30 min, else starts a new one. New chat uses
  `--session-id <uuid>`, later messages `--resume <id>`. Q/A is appended to `claude_chats/<name>.md`.
  `claude_chat_new`, `claude_chat_list`, `claude_chat_history`, `claude_chat_delete` (risky). The agent speaks
  "Asking Claude, sir." (`SLOW_TOOL_NOTICE`) and shows the thinking state while the tool runs.
- Fast-path tool actions use `("call", tool, json_args)`; `speak_result` fast paths (what's playing, play X on
  Spotify) speak the tool result instead of a canned reply.

## Memory, reminders and briefing

All three keep small JSON files in `%APPDATA%/Jarvis` (Linux fallback `~/.local/share/Jarvis`, see `store.py`:
atomic writes, a broken file reads as empty).

- **Memory** (`memory.py`, tools `remember` / `recall` / `forget` in `tools/memory_tools.py`). `memory.json` is a list of
  `{id, text, created, source: "explicit"|"inferred"}`, capped at 200 (oldest inferred is dropped first). A new fact
  that matches an existing one (same text, Jaccard >= 0.7 on stemmed content words, or a 3+ word subset) updates it
  instead; negations are content words, so "I don't like tea" is not a duplicate of "I like tea". Passwords, keys,
  PINs and similar are refused. Relevance is token overlap (4+ letter prefixes count). **Injection:** the static system
  prompt only carries the rules; `Agent.handle` adds a `[Known about the user: a; b; c]` line after `[Now: ...]` on the
  newest user message (`Agent.memory_block`, set by `extras.Extras`). At most 8 facts: everything when 8 or fewer are
  stored, else up to 3 core facts (name, preferences, "main", family...) plus the best matches for the user's words.
  It is added to the per-call copy of the messages only, never stored in history, so the KV cache of system prompt +
  tool schemas stays valid. `recall` and `forget` speak facts back in the second person ("Your main project is Jarvis").
- **Time phrases** (`timeparse.py`, pure): `parse_duration("an hour and a half")`, `parse_when(text, now)` for "in 20
  minutes", "at 6pm", "at 18:30", "tomorrow at 9", "friday at 3pm", "tonight at 9", "day after tomorrow", noon,
  midnight, "half past 7". The result is always in the future. A bare "at 9" means the next time that clock reading
  happens; with an explicit day, 1-6 reads as pm and 7-12 as am. `strict=True` requires the whole text to be a time
  expression (used to split "call mom at the office at 6pm").
- **Reminders and timers** (`reminders.py`, tools in `tools/reminder_tools.py`): `set_timer`, `set_reminder(text,
  when)`, `list_reminders`, `cancel_reminder(query)`. `reminders.json` holds pending items `{id, kind, text, label,
  due (epoch), created, seconds}`. `Scheduler` runs as a background task started by `Extras.start` once Jarvis is
  online and ticks every second; due items are removed from the store and announced through
  `Assistant.announce` ("Sir, reminder: ..." / "Your 10 minute timer is done."), which waits for the current turn.
  `reminders.chime` plays a gentle three-note descending chime first (when nobody is mid-turn). Items that came due
  while Jarvis was off are announced once at start ("Sir, while you were away: ...") or, while the automatic briefing
  is still due today, held and read as part of it.
- **Briefing** (`briefing.py`, tool `briefing` in `tools/briefing_tool.py`). `compose()` is pure: greeting by time of
  day (+ name from memory), date, weather for the configured city, today's remaining reminders, missed items, and
  Claude Code sessions with activity in the last 24 h ("Claude finished work on <title> in <project> last night",
  "still working", "waiting for your permission"). Weather is Open-Meteo (`api.open-meteo.com/v1/forecast`, 3 s timeout,
  WMO code -> words, skipped silently on any failure); weather and the session scan run in parallel.
  **Automatic:** `Extras.first_wake_briefing` runs in `run_turn` after the user's request was recorded, for wake-word
  and hotkey activations only (not typed text or barge-in): if `due_today` (enabled, `auto_first_wake`, hour >=
  `after_hour`, and `state.json` `last_briefing` is not today) it speaks the briefing, records it in history, writes
  today's date to `state.json`, then the request is handled normally. Asking for the briefing yourself skips the
  automatic one and counts for the day.
- **Fast paths** (`fastpath.py`): "remember (that) X", "what do you remember (about me)", "forget (that|about) X", "set a
  timer for 10 minutes" / "timer 5 minutes" / "set a 10 minute timer", "remind me in 20 minutes to X" / "remind me to
  X at 6pm" / "set a reminder for 6pm to X", "what reminders do I have", "cancel the timer", "give me my briefing" /
  "morning briefing" / "what's my day". All run the tool and speak its result. Anything unclear (no time, no timer
  length, "remember to ...") goes to the model.
- `extras.py` keeps the glue out of `__main__.py`: `Extras(assistant)` in `Assistant.__init__`, `extras.start(bg)` after
  the "Online" announcement, `extras.first_wake_briefing(req, text)` in `run_turn`. Config: `[memory]`, `[reminders]`,
  `[briefing]`. Tests use an autouse fixture (`tests/conftest.py`) that points `APPDATA` at a temp folder.

## Safety model

Every tool declares a base risk: `safe` or `risky`. `safety.py` can raise a
call to `risky` from its arguments (for example a PowerShell command that
contains `Remove-Item`, `del`, `format`, `Stop-Process`, registry writes,
`Invoke-WebRequest | iex`, installs).

- `safe` runs at once (list folders, read files, open apps or paths, system info).
- `risky` needs approval. Jarvis says "Sir, I'm about to <summary>. Shall I proceed?"
  and sends a `confirm` event. A spoken "yes"/"no" or an orb click resolves it.
  Timeout of 30 s counts as "no".

## Claude Code integration

Tool `claude_code_run(folder, prompt)` starts
`claude -p <prompt> --output-format stream-json --verbose` in `folder` as a
background job. It streams events, keeps the final result text, and emits
`job` events. On failure Jarvis speaks a short alert. Milestone 2 will also
send this alert to the phone through ntfy (`notify.py` hook point).
Starting a run counts as `risky` (it edits files), so it needs approval.

## Claude Code sessions (voice control)

- `claude_sessions.py` learns sessions from Claude's own transcripts `~/.claude/projects/<encoded-cwd>/<id>.jsonl`
  (`CLAUDE_CONFIG_DIR` honoured). The line format is internal and version dependent, so parsing is defensive:
  every line is tried as JSON; `user`/`assistant` lines with string or block content count; `isMeta`,
  `isSidechain`, tool-result-only lines and command noise (`<command-name>`, `Caveat:`) are skipped; titles come
  from `custom-title` > `ai-title` > `summary` lines, else the first real user message. Only the first 64 KB and
  last 256 KB of each file are read; results are cached by (path, mtime, size); the 300 newest files are parsed per
  scan. `Session`: id, path, project_dir (the `cwd` field, else a lossy decode of the folder name), project_name,
  title, first_message, last_activity (mtime), last_assistant_text, last_tool, message_count (extrapolated for
  huge files), last_kind, last_user_ts.
- `projects()` = project folders from sessions plus immediate subfolders of Desktop and Documents (`paths.known_folder`).
  `resolve_project(spoken)` tokenizes (filler words and camelCase handled), fuzzy matches the folder name, its parent
  and grandparent (so "ai on pc" and "jarvis pc assistant" find `...\Ai on pc\Jarvis`; a bare container folder
  resolves to the newest project inside it) and prefers projects with history.
- `find_sessions(query, project, limit)` scores token overlap with title (1.0), first message (0.6), project
  name (0.5), last reply (0.3) plus a recency boost (up to 0.2, 7-day decay); `is_ambiguous` = top two within 0.12.
- Tools (`tools/claude_sessions_tools.py`, all safe): `claude_sessions` (spoken list "1. Login bug, Jarvis, 2 hours
  ago"), `claude_open_session` (`--resume <id>` in the session's folder; ambiguous topic returns "I found 3
  sessions: ... Which one, sir?", the follow-up listening captures the answer and the model calls again with
  `choice=N`; the offered ids are remembered for 5 minutes), `claude_continue` (`--continue`), `claude_new_session`
  (`--name`, derived from the prompt when not given), `claude_status` (busy = file modified < 20 s ago and last
  line mid-turn; idle otherwise; last reply in <= 2 sentences, last tool, pending permission),
  `claude_ask_session` (headless `claude -p --resume <id> --output-format json`, question on stdin, 180 s).
  Terminals are opened by `claude_terminal.open_claude_terminal(folder, prompt, args)`.
- Headless runs started by Jarvis (`claude_chat`, `claude_code_run`, `claude_ask_session`) set `JARVIS_HEADLESS=1`
  so the hook ignores them.

### Hooks: Jarvis speaks when Claude finishes or needs you

`scripts/install_claude_hooks.py` merges into `~/.claude/settings.json` a `Stop` hook and `Notification` hooks
(one entry each for `permission_prompt`, `agent_needs_input`, `elicitation_dialog`) running
`"<repo>\.venv\Scripts\python.exe" -m jarvis.claude_hook` (timeout 10). It writes a timestamped backup first,
replaces older Jarvis entries (idempotent), leaves all other settings and hooks alone; `--uninstall` removes only
its entries; `--settings` and `--python` override the paths. `setup_windows.ps1` runs it.

`jarvis.claude_hook` reads the hook JSON from stdin and sends one `claude_event` over a raw WebSocket to the bus
(port from config, default 8765, 2 s total budget). Stdlib only, all errors swallowed, exit code always 0 (exit 2
from a Stop hook would block Claude). Nothing is sent when `JARVIS_HEADLESS=1`.

`Assistant.on_claude_event` -> `claude_watch.ClaudeWatch.handle`: Stop -> "Sir, Claude finished in <project>:
<first sentence, <= 25 words>"; permission_prompt -> "Sir, Claude needs your permission in <project>."
(remembered per session for `claude_status`, cleared by the next Stop); agent_needs_input / elicitation ->
"Sir, Claude has a question for you in <project>."; idle_prompt ignored. Same kind for the same session is
throttled for 30 s. Spoken through `announce()`, which waits for the current turn. Config `[claude_watch]`:
`enabled`, `announce_finish`, `announce_permission`, `min_turn_seconds` (a "finished" announcement needs the turn
to have lasted that long, measured from the last real user message timestamp in the transcript to now; if
unavailable it announces anyway).

## Maps and directions

"I want to go to X", "how do I get to X", "directions to X" go to the model, whose static prompt routes them to the
`directions` tool (`tools/maps_tools.py`, all tools safe). `show_on_map(place)`, `close_map`, `open_transit_directions`
and `where_am_i` complete the set; "close the map" and "where am I" are fast paths.

- **Location** (`geo.locate`, cached 10 min, source and accuracy logged): 1. Windows location services via a hidden
  `powershell.exe -NoProfile` running `System.Device.Location.GeoCoordinateWatcher` (waits up to ~5 s, 8 s timeout;
  needs Settings > Privacy & security > Location on, including "Let desktop apps access your location"; if off or
  denied it falls through). 2. The saved home: `[maps] home_address`, else a memory fact like "Home is 5 Graf Ignatiev
  Street, Sofia" (`work` likewise); geocoded once and cached in `state.json` (`geo_cache`). 3. IP geolocation
  (ipapi.co, fallback ip-api.com; accuracy 5 km). "home" / "work" also work as destinations.
- **Geocoding**: Nominatim `/search` (`jsonv2`, limit 5, `viewbox` +-50 km around the origin, `bounded=0`, User-Agent
  `Jarvis-voice-assistant/0.1 (personal use)`, at most 1 request per second). Among results at least half as important
  as the best, the nearest to the origin wins. A miss is retried with `, <default_city>` appended.
- **Routing**: FOSSGIS OSRM `routed-car` and `routed-foot` (`overview=full&geometries=geojson&steps=true`), fetched in
  parallel. `step_text` turns maneuver type/modifier + road name into "Turn left onto Vitosha Boulevard".
- **Public transport**: free OSRM has no timetables and Jarvis never invents times. The `transit_url`
  (`google.com/maps/dir/?api=1&origin=..&destination=..&travelmode=transit`) is shown as a card with an "Open in Google
  Maps" button; the spoken summary says so ("for public transport I've put a Google Maps link on screen").
- **Map window** (`orb/map.html`, `orb/src/map.ts`): a second Tauri window `map`, created hidden from `tauri.conf.json`
  (fullscreen, frameless, always on top, NOT click-through, focusable). The Rust command `set_map_visible` shows it and
  hides the small orb window (the map page has its own orb with a glowing ring in the bottom-right corner, plus
  caption and Approve/Deny pills), or reverses that. The map page has its own WebSocket to the core. Esc or the Close
  button hides it and sends `map_closed`. Basemap: MapLibre GL JS (bundled) with the CARTO dark-matter style, recolored
  at load to the HUD palette (`map-style.ts`); if the style cannot be fetched it falls back to a no-basemap HUD
  (grid, route, markers). CSP allows `basemaps.cartocdn.com` and `worker-src blob:`.
  `vite.config.ts` copies MapLibre's worker files to `/maplibre/` (maplibre 6 loads `maplibre-gl-worker.mjs` by
  relative URL, which a bundle would break).
- Errors are spoken: offline gives "I can't reach the map services right now. Please check the internet connection."
- Config `[maps]`: `home_address`, `work_address`, `default_city`, `geocoder_url`, `routing_url`, `show_transit_link`.
