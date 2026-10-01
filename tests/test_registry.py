import asyncio
from typing import Optional

from jarvis.tools.registry import Registry, ToolError


def make():
    reg = Registry()

    @reg.tool(risk="risky")
    def add(a: int, b: float = 1.5, label: Optional[str] = None, flag: bool = False, names: list[str] = None):
        """Add numbers.

        Args:
            a: first number
            b: second number
        """
        return a + b

    @reg.tool("Async thing")
    async def aecho(text: str) -> str:
        return text.upper()

    @reg.tool()
    def boom() -> str:
        raise ToolError("nope")

    return reg


def test_schema():
    reg = make()
    t = reg.get("add")
    assert t.risk == "risky"
    s = t.schema()
    assert s["type"] == "function"
    fn = s["function"]
    assert fn["name"] == "add" and fn["description"] == "Add numbers."
    p = fn["parameters"]
    assert p["required"] == ["a"]
    assert p["properties"]["a"] == {"type": "integer", "description": "first number"}
    assert p["properties"]["b"]["type"] == "number"
    assert p["properties"]["label"]["type"] == "string"
    assert p["properties"]["flag"]["type"] == "boolean"
    assert p["properties"]["names"] == {"type": "array", "items": {"type": "string"}}
    assert reg.get("aecho").description == "Async thing"


def test_call_coerces_and_errors():
    reg = make()
    assert asyncio.run(reg.call("add", {"a": "2", "b": 3})) == "5.0"
    assert asyncio.run(reg.call("aecho", {"text": "hi"})) == "HI"
    assert "missing" in asyncio.run(reg.call("add", {}))
    assert asyncio.run(reg.call("boom", {})) == "Error: nope"
    assert "unknown tool" in asyncio.run(reg.call("zzz", {}))
    assert asyncio.run(reg.call("add", {"a": "x"})).startswith("Error")


def test_real_tools_register():
    from jarvis.tools import load_all

    reg = load_all()
    names = set(reg.names())
    expected = {"run_powershell", "system_info", "lock_pc", "list_dir", "read_file", "search_files",
                "open_path", "delete_path", "move_path", "open_app", "open_url",
                "claude_code_run", "claude_code_status", "claude_code_cancel"}
    assert expected <= names
    assert reg.get("delete_path").risk == "risky"
    assert reg.get("claude_code_run").risk == "risky"
    assert reg.get("list_dir").risk == "safe"
    for s in reg.schemas():
        assert s["function"]["parameters"]["type"] == "object"
