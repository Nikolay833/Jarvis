"""Ollama /api/chat client with tool calling (non-streaming)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

log = logging.getLogger("jarvis.llm")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove qwen3 <think> blocks (also an unclosed one, or a stray closing tag)."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class ThinkStripper:
    """Incremental `strip_think`: feed raw deltas, get visible text. Tags may split across chunks."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            low = self._buf.lower()
            if self._inside:
                i = low.find(self.CLOSE)
                if i < 0:
                    # drop thinking text, keep a possible partial closing tag
                    self._buf = self._buf[len(self._buf) - self._partial(low, self.CLOSE):]
                    break
                self._buf = self._buf[i + len(self.CLOSE):]
                self._inside = False
                continue
            o, c = low.find(self.OPEN), low.find(self.CLOSE)
            hits = [i for i in (o, c) if i >= 0]
            if not hits:
                keep = max(self._partial(low, self.OPEN), self._partial(low, self.CLOSE))
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            i = min(hits)
            out.append(self._buf[:i])
            if i == o:
                self._buf = self._buf[i + len(self.OPEN):]
                self._inside = True
            else:  # stray closing tag: ignore it
                self._buf = self._buf[i + len(self.CLOSE):]
        return "".join(out)

    @staticmethod
    def _partial(low: str, tag: str) -> int:
        """Length of the longest suffix of `low` that is a proper prefix of `tag`."""
        for n in range(min(len(tag) - 1, len(low)), 0, -1):
            if tag.startswith(low[-n:]):
                return n
        return 0

    def flush(self) -> str:
        """End of stream: release held-back text unless it is inside an unclosed think block."""
        rest = "" if self._inside else self._buf
        self._buf = ""
        self._inside = False
        return rest


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

    # Optional: chat_stream(messages, tools) -> async iterator of str (visible text deltas),
    # ending with one LLMResponse (the complete step). The agent falls back to `chat` without it.


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


def normalize_keep_alive(value: int | str) -> int | str:
    """Ollama takes a number of seconds (-1 = forever) or a duration string such as "30m".

    A bare numeric string ("-1", "300") must become an int: Ollama rejects "-1" without a unit.
    """
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return v
    return value


class OllamaClient:
    def __init__(self, url: str, model: str, think: bool | None = False,
                 timeout: float = 120.0, num_ctx: int = 8192, keep_alive: int | str = -1,
                 max_reply_tokens: int = 0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.think = think
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = normalize_keep_alive(keep_alive)
        self.max_reply_tokens = max_reply_tokens
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
        if self.max_reply_tokens > 0:
            payload["options"]["num_predict"] = self.max_reply_tokens
        if tools:
            payload["tools"] = tools
        if self.think is not None and self._supports_think:
            payload["think"] = self.think
        return payload

    async def warm_up(self) -> float:
        """Load the model into VRAM with an empty chat request. Returns seconds taken."""
        import httpx

        payload = {"model": self.model, "messages": [], "stream": False, "keep_alive": self.keep_alive,
                   "options": {"num_ctx": self.num_ctx}}  # same num_ctx, or Ollama reloads the model
        t0 = time.perf_counter()
        try:
            resp = await self._http().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc!r}") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return time.perf_counter() - t0

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

    async def chat_stream(self, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]] | None = None) -> AsyncIterator[str | LLMResponse]:
        """Stream one step. Yields visible text deltas (think blocks removed), then one LLMResponse."""
        import httpx

        payload = self.build_payload(messages, tools)
        payload["stream"] = True
        stripper = ThinkStripper()
        visible: list[str] = []
        raw_calls: list[dict[str, Any]] = []
        try:
            for attempt in (1, 2):
                async with self._http().stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if (attempt == 1 and resp.status_code == 400 and "think" in payload
                                and "think" in body.lower()):
                            self._supports_think = False
                            payload.pop("think")
                            continue
                        raise LLMError(f"Ollama error {resp.status_code}: {body[:300]}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError as exc:
                            raise LLMError(f"bad stream line from Ollama: {line[:100]!r}") from exc
                        if chunk.get("error"):
                            raise LLMError(f"Ollama error: {str(chunk['error'])[:300]}")
                        message = chunk.get("message") or {}
                        raw_calls.extend(message.get("tool_calls") or [])
                        delta = message.get("content") or ""
                        if delta:
                            text = stripper.feed(delta)
                            if text:
                                visible.append(text)
                                yield text
                        if chunk.get("done"):
                            break
                    break
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc}") from exc
        tail = stripper.flush()
        if tail:
            visible.append(tail)
            yield tail
        yield LLMResponse(
            content="".join(visible).strip(),
            tool_calls=parse_tool_calls({"tool_calls": raw_calls}),
            raw_tool_calls=raw_calls,
        )
