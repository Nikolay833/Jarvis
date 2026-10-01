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
| Speech to text | `faster-whisper` | `large-v3-turbo` on CUDA, float16. Fallback `small` on CPU. |
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
  audio/                wakeword.py, recorder.py, stt.py, tts.py
  llm.py                Ollama chat client with tool calling (non-streaming `chat`, streaming `chat_stream`)
  fastpath.py           LLM-free answers for time/date/open app/lock/stop/volume, minimise/maximise, music keys, "play X on spotify", "search for X" (strict anchored regexes; "close X" is never a fast path)
  agent.py              Conversation loop: messages, tool calls, confirmations
  safety.py             Risk classification of tool calls
  paths.py              Path resolution (~, env vars, Desktop/Documents/... incl. OneDrive-redirected)
  tools/                Tool registry + tools (system, files, apps, windows, chrome, spotify, claude_code,
                        claude_history, claude_chat)
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

Orb to core:

| type | fields | meaning |
|---|---|---|
| `confirm_response` | `id`, `approved`: bool | User clicked approve/deny. |
| `text_input` | `text` | Typed request instead of voice. |
| `activate` | none | Hotkey/click: start listening without wake word. |

## Audio flow and feedback

- The mic stream never closes between wake word and recording, so speech right after
  "Hey Jarvis" is kept. Recording starts with the audio after the detection frame; the
  `EndpointDetector` runs in `after_wake` mode (no calibration from the first blocks, noise
  floor from pre-roll, capped). Queued audio is flushed before wake listening resumes
  (no self-hearing of TTS) and before hotkey/confirmation recordings.
- Feedback: chime on wake/hotkey (its energy is ignored for endpointing), "Online, sir." at
  startup, spoken apologies for empty transcripts ("Sorry sir, I didn't catch that.") and failed turns.
- Startup warm-up in parallel: Ollama model load (empty `/api/chat`, same `num_ctx`/`keep_alive`),
  Whisper on 1 s silence, Kokoro "Ready.". Ollama unreachable gives a WARNING and a spoken hint.
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
