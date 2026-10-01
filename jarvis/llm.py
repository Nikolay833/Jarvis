"""Ollama /api/chat client with tool calling (non-streaming)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("jarvis.llm")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove qwen3 <think> blocks (also an unclosed one, or a stray closing tag)."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLM(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=str(fn.get("name", "")), arguments=args))
    return calls


class OllamaClient:
    def __init__(self, url: str, model: str, think: bool | None = False,
                 timeout: float = 120.0, num_ctx: int = 8192, keep_alive: str = "30m") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.think = think
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._supports_think = True
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.4},
        }
        if tools:
            payload["tools"] = tools
        if self.think is not None and self._supports_think:
            payload["think"] = self.think
        return payload

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        import httpx

        payload = self.build_payload(messages, tools)
        try:
            resp = await self._http().post("/api/chat", json=payload)
            if resp.status_code == 400 and "think" in payload and "think" in resp.text.lower():
                # Model or Ollama version does not know `think`: retry without it.
                self._supports_think = False
                payload.pop("think")
                resp = await self._http().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        message = resp.json().get("message", {})
        return LLMResponse(
            content=strip_think(message.get("content") or ""),
            tool_calls=parse_tool_calls(message),
            raw_tool_calls=list(message.get("tool_calls") or []),
        )
