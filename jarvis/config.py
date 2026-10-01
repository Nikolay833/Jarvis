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
    max_reply_tokens: int = 200  # num_predict per LLM step; 0 = no limit


@dataclass
class WhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    fallback_model: str = "small"
    language: str = "en"


@dataclass
class TTSConfig:
    voice: str = "bm_george"
    lang_code: str = "b"
    speed: float = 1.0


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
class SpotifyConfig:
    client_id: str = ""      # optional Spotify developer app (free) for song search
    client_secret: str = ""


@dataclass
class FilesConfig:
    allowed_roots: list[str] = field(default_factory=list)


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    spotify: SpotifyConfig = field(default_factory=SpotifyConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
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
