"""Configuration loading. Falls back to defaults when no config.toml exists."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class OllamaConfig:
    url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:14b"
    think: bool = False
    timeout: float = 120.0
    num_ctx: int = 8192
    keep_alive: str = "-1"  # "-1" = keep the model loaded forever (sent as int -1); or e.g. "30m"
    max_reply_tokens: int = 450  # num_predict per LLM step; 0 = no limit


# Words Whisper should expect to hear (names it otherwise mishears).
DEFAULT_VOCABULARY = ["Jarvis", "Claude", "Claude Code", "Spotify", "Chrome", "Discord", "VS Code", "PowerShell"]


@dataclass
class WhisperConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    fallback_model: str = "small"
    language: str = "en"
    vocabulary: list[str] = field(default_factory=lambda: list(DEFAULT_VOCABULARY))
    beam_size: int = 5  # 1 = greedy (fastest), 5 = noticeably more accurate
    best_of: int = 5  # candidates when the temperature fallback samples
    temperature: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])  # retry ladder on bad decodes
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 300  # padding keeps VAD from clipping word starts
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    use_hotwords: bool = True  # also pass the vocabulary as hotwords (faster-whisper >= 1.0)


@dataclass
class STTConfig:
    engine: str = "whisper"  # "whisper" or "parakeet"


@dataclass
class ParakeetConfig:
    model: str = "nemo-parakeet-tdt-0.6b-v2"  # v2 = English, "nemo-parakeet-tdt-0.6b-v3" = multilingual
    device: str = "cuda"  # "cuda" (falls back to CPU) or "cpu"


@dataclass
class TTSConfig:
    voice: str = "bm_george"  # or a blend: "bm_george:0.6,bm_lewis:0.4"
    lang_code: str = "b"
    speed: float = 1.05
    fx: str = "jarvis"  # "jarvis" | "none"
    fx_amount: float = 0.35  # 0..1


@dataclass
class WakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5


@dataclass
class AudioConfig:
    input_device: str = ""
    output_device: str = ""
    silence_seconds: float = 1.2
    max_record_seconds: float = 20.0
    no_speech_timeout: float = 6.0
    announce_ready: bool = True  # say "Online, sir." after startup
    chime: bool = True  # soft chime when listening starts
    follow_up: bool = True  # after Jarvis asks a question, listen for the answer without the wake word
    follow_up_seconds: float = 6.0  # how long to wait for that answer
    conversation: bool = True  # keep listening after every reply; Jarvis goes away once you stop talking
    conversation_seconds: float = 5.0  # silence that ends the conversation
    barge_in: bool = True  # talk over Jarvis to interrupt him (speakers + mic; echo is filtered out)
    barge_in_sensitivity: float = 0.5  # 0..1, higher interrupts more easily (also more false interrupts)
    barge_in_min_ms: float = 250.0  # speech must last this long (at sensitivity 0.5) to count
    barge_in_wake: bool = True  # "Hey Jarvis" during speech always interrupts


@dataclass
class BusConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class SafetyConfig:
    confirm_timeout: float = 30.0


@dataclass
class AgentConfig:
    max_steps: int = 8
    max_history_messages: int = 30
    fast_paths: bool = True  # answer time/date/open app/lock/stop without the LLM
    stream_replies: bool = True  # speak sentences while the LLM is still generating


@dataclass
class ClaudeCodeConfig:
    binary: str = "claude"
    permission_mode: str = "acceptEdits"


@dataclass
class ClaudeWatchConfig:
    enabled: bool = True  # react to events sent by the Claude Code hooks (scripts/install_claude_hooks.py)
    announce_finish: bool = True  # "Sir, Claude finished in <project>: ..."
    announce_permission: bool = True  # "Sir, Claude needs your permission in <project>."
    min_turn_seconds: float = 20.0  # announce "finished" only if the turn took at least this long
    voice_approval: bool = True  # ask by voice "Shall I allow it?" for Claude permission prompts (PermissionRequest hook)


@dataclass
class MemoryConfig:
    enabled: bool = True  # remember facts about you and show the relevant ones to the model each turn
    max_facts: int = 200


@dataclass
class RemindersConfig:
    enabled: bool = True  # timers and reminders (announced out loud when due)
    chime: bool = True  # gentle chime before the announcement


@dataclass
class BriefingConfig:
    enabled: bool = True  # "give me my briefing" and the automatic morning one
    auto_first_wake: bool = True  # speak it on the first activation of the day
    after_hour: int = 5  # ...but only after this hour (0-23), so a 1 a.m. wake does not use up the day
    city: str = "Sofia"
    latitude: float = 42.6977
    longitude: float = 23.3219
    include_weather: bool = True  # Open-Meteo (free, no key); skipped silently when offline
    include_reminders: bool = True
    include_claude: bool = True  # Claude Code sessions active since yesterday


@dataclass
class SpotifyConfig:
    client_id: str = ""      # optional Spotify developer app (free) for song search
    client_secret: str = ""


@dataclass
class MapsConfig:
    home_address: str = ""  # "home" for directions and the fallback location; empty = a "home is ..." memory fact
    work_address: str = ""  # "work" for directions; empty = a "work is ..." memory fact
    default_city: str = "Sofia"  # retried with this appended when a place is not found
    geocoder_url: str = "https://nominatim.openstreetmap.org"
    routing_url: str = "https://routing.openstreetmap.de"
    show_transit_link: bool = True  # Public transport card with a Google Maps link


@dataclass
class VisionConfig:
    """Computer vision: look_at_screen / read_screen_text (screen) and look_through_webcam (camera)."""
    model: str = "qwen2.5vl:3b"  # small VLM; the 7b does not fit next to qwen3:14b + Whisper on 16 GB
    url: str = ""  # Ollama URL for vision; empty = [ollama] url
    keep_alive: str = "2m"  # unload soon after use so the main model keeps its VRAM ("-1" = keep loaded)
    max_side: int = 1280  # longest image side sent to the model, in pixels
    timeout: float = 90.0  # seconds per vision request (the first call also loads the model)
    max_reply_tokens: int = 400
    webcam_enabled: bool = False  # privacy: the camera is never used unless this is true AND you ask
    webcam_index: int = 0


@dataclass
class FilesConfig:
    allowed_roots: list[str] = field(default_factory=list)


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    claude_watch: ClaudeWatchConfig = field(default_factory=ClaudeWatchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reminders: RemindersConfig = field(default_factory=RemindersConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    spotify: SpotifyConfig = field(default_factory=SpotifyConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    source: str = "defaults"


def _coerce(value: Any, default: Any) -> Any:
    """Coerce a TOML value to the type of the default; keep value if impossible."""
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
        if isinstance(default, list):
            return list(value)
    except (TypeError, ValueError):
        return default
    return value


def _apply(obj: Any, data: dict[str, Any]) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        current = getattr(obj, f.name)
        if is_dataclass(current):
            if isinstance(data[f.name], dict):
                _apply(current, data[f.name])
        else:
            setattr(obj, f.name, _coerce(data[f.name], current))


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    _apply(cfg, data)
    return cfg


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("JARVIS_CONFIG")
    if env:
        paths.append(Path(env))
    paths.append(repo_root() / "config.toml")
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Jarvis" / "config.toml")
    return paths


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path`, else the first existing candidate, else defaults."""
    candidates = [Path(path)] if path else candidate_paths()
    for p in candidates:
        if p.is_file():
            with p.open("rb") as fh:
                data = tomllib.load(fh)
            cfg = config_from_dict(data)
            cfg.source = str(p)
            return cfg
    return Config()
