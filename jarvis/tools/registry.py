"""Tool registry. `@tool(...)` turns a function into an Ollama tool schema."""

from __future__ import annotations

import asyncio
import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

SAFE = "safe"
RISKY = "risky"


class ToolError(Exception):
    """Expected failure; message is returned to the model as the tool result."""


@dataclass
class Tool:
    name: str
    description: str
    risk: str
    func: Callable[..., Any]
    parameters: dict[str, Any]
    is_async: bool = field(default=False)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _json_type(ann: Any) -> dict[str, Any]:
    origin = typing.get_origin(ann)
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    if origin in (typing.Union, types.UnionType):
        return _json_type(args[0]) if args else {"type": "string"}
    if origin in (list, tuple, set):
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if ann is bool:
        return {"type": "boolean"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is dict:
        return {"type": "object"}
    return {"type": "string"}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    doc = inspect.cleandoc(doc or "")
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        if re.match(r"^\s*(Args|Arguments|Parameters):\s*$", line):
            in_args = True
            continue
        if in_args:
            m = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?:\s*(.+)$", line)
            if m:
                params[m.group(1)] = m.group(2).strip()
            elif line.strip() and not line.startswith((" ", "\t")):
                in_args = False
        if not in_args and not re.match(r"^\s*(Returns|Raises):", line):
            summary_lines.append(line)
    summary = " ".join(" ".join(summary_lines).split())
    return summary, params


def build_parameters(func: Callable[..., Any], param_docs: dict[str, str] | None = None) -> dict[str, Any]:
    hints = typing.get_type_hints(func)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in inspect.signature(func).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = _json_type(hints.get(pname, str))
        if param_docs and pname in param_docs:
            prop["description"] = param_docs[pname]
        props[pname] = prop
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    try:
        if t == "integer" and not isinstance(value, bool):
            return int(value)
        if t == "number":
            return float(value)
        if t == "boolean" and isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        if t == "string" and not isinstance(value, str):
            return str(value)
    except (TypeError, ValueError):
        raise ToolError(f"Invalid value {value!r}, expected {t}")
    return value


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, func: Callable[..., Any], *, name: str | None = None,
                 description: str | None = None, risk: str = SAFE) -> Tool:
        if risk not in (SAFE, RISKY):
            raise ValueError(f"bad risk {risk!r}")
        summary, param_docs = _parse_docstring(func.__doc__ or "")
        t = Tool(
            name=name or func.__name__,
            description=description or summary or func.__name__,
            risk=risk,
            func=func,
            parameters=build_parameters(func, param_docs),
            is_async=inspect.iscoroutinefunction(func),
        )
        self._tools[t.name] = t
        return t

    def tool(self, description: str | None = None, *, risk: str = SAFE, name: str | None = None):
        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(func, name=name, description=description, risk=risk)
            return func

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool; always returns a string (errors included)."""
        t = self._tools.get(name)
        if t is None:
            return f"Error: unknown tool '{name}'"
        props = t.parameters["properties"]
        clean: dict[str, Any] = {}
        try:
            for k, v in (args or {}).items():
                if k in props:
                    clean[k] = _coerce(v, props[k])
            missing = [r for r in t.parameters["required"] if r not in clean]
            if missing:
                return f"Error: missing argument(s): {', '.join(missing)}"
            result = await t.func(**clean) if t.is_async else await asyncio.to_thread(t.func, **clean)
        except ToolError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else str(result)


registry = Registry()
tool = registry.tool
