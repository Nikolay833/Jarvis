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
  audio/                wakeword.py, recorder.py, stt.py (whisper + parakeet engines), tts.py
  stt_bench.py          `python -m jarvis.stt_bench`: compare STT engines on your recordings
  llm.py                Ollama chat client with tool calling (non-streaming `chat`, streaming `chat_stream`)
  fastpath.py           LLM-free answers for time/date/open app/lock/stop/volume, minimise/maximise, music keys, "play X on spotify", "search for X" (strict anchored regexes; "close X" is never a fast path)
  agent.py              Conversation loop: messages, tool calls, confirmations
  safety.py             Risk classification of tool calls
  paths.py              Path resolution (~, env vars, Desktop/Documents/... incl. OneDrive-redirected)
  claude_sessions.py    Claude Code sessions from ~/.claude/projects transcripts (parse, project resolve, rank)
  claude_watch.py       Hook events -> spoken announcements (throttle, pending permission)
  claude_hook.py        Stdlib-only Claude Code hook: forwards Stop/Notification to the bus
  tools/                Tool registry + tools (system, files, apps, windows, chrome, spotify, claude_code,
                        claude_history, claude_chat, claude_sessions_tools, claude_terminal)
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

Orb to core:

| type | fields | meaning |
|---|---|---|
| `confirm_response` | `id`, `approved`: bool | User clicked approve/deny. |
| `text_input` | `text` | Typed request instead of voice. |
| `activate` | none | Hotkey/click: start listening without wake word. |
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
