import asyncio
import json

import numpy as np

from jarvis.audio.recorder import EndpointDetector, display_level, rms_of
from jarvis.audio.tts import split_sentences
from jarvis.bus import EventBus
from jarvis.config import config_from_dict, load_config
from jarvis.llm import OllamaClient, parse_tool_calls
from jarvis.tools.apps import build_open_app_command


def test_config_defaults_and_override(tmp_path):
    cfg = config_from_dict({"ollama": {"model": "x"}, "bus": {"port": "9000"}, "files": {"allowed_roots": ["a"]},
                            "bogus": {}})
    assert cfg.ollama.model == "x" and cfg.bus.port == 9000 and cfg.files.allowed_roots == ["a"]
    assert cfg.tts.voice == "bm_george" and cfg.claude_code.permission_mode == "acceptEdits"
    p = tmp_path / "c.toml"
    p.write_text('[safety]\nconfirm_timeout = 5\n')
    assert load_config(p).safety.confirm_timeout == 5.0


def test_example_config_matches_defaults():
    from pathlib import Path

    from jarvis.config import Config

    path = Path(__file__).resolve().parent.parent / "config.example.toml"
    assert load_config(path) == _with_source(Config(), str(path))


def _with_source(cfg, src):
    cfg.source = src
    return cfg


def test_endpoint_detector():
    det = EndpointDetector(silence_seconds=1.0, max_seconds=20, no_speech_timeout=6)
    dt = 0.032
    t = 0
    for _ in range(10):
        assert not det.feed(0.002, dt)
    for _ in range(30):
        assert not det.feed(0.1, dt)
    assert det.speech_started
    done = False
    for _ in range(40):
        t += dt
        if det.feed(0.002, dt):
            done = True
            break
    assert done and det.reason == "silence" and 0.9 < t < 1.2


def test_endpoint_no_speech_and_max():
    det = EndpointDetector(no_speech_timeout=1.0)
    while not det.feed(0.001, 0.1):
        pass
    assert det.reason == "no_speech"
    det = EndpointDetector(max_seconds=2.0)
    while not det.feed(0.2, 0.1):
        pass
    assert det.reason == "max"


def test_levels():
    assert rms_of(np.zeros(10, dtype=np.int16)) == 0.0
    assert abs(rms_of(np.full(10, 16384, dtype=np.int16)) - 0.5) < 1e-3
    assert display_level(1.0) == 1.0 and display_level(0.0) == 0.0


def test_split_sentences():
    assert split_sentences("Hello sir.  Very good! Shall we? ok") == ["Hello sir.", "Very good!", "Shall we?", "ok"]
    assert split_sentences("  ") == []


def test_open_app_command():
    assert build_open_app_command("VS Code") == "Start-Process 'code'"
    assert "Discord" in build_open_app_command("discord")
    cmd = build_open_app_command("Foo's App")
    assert "Get-StartApps" in cmd and "Foo''s App" in cmd


def test_parse_tool_calls_and_payload():
    msg = {"tool_calls": [{"function": {"name": "a", "arguments": '{"x": 1}'}},
                          {"function": {"name": "b", "arguments": {"y": 2}}}]}
    calls = parse_tool_calls(msg)
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2})]
    cli = OllamaClient("http://x", "qwen3:14b", think=False)
    p = cli.build_payload([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert p["think"] is False and p["stream"] is False and p["tools"]


def test_bus_websocket_roundtrip():
    import websockets

    async def run():
        bus = EventBus("127.0.0.1", 0)
        await bus.start()
        port = bus._server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            hello = json.loads(await ws.recv())
            assert hello == {"type": "state", "state": "idle"}
            bus.emit_nowait("reply", text="hi")
            assert json.loads(await asyncio.wait_for(ws.recv(), 2)) == {"type": "reply", "text": "hi"}
            await ws.send(json.dumps({"type": "activate"}))
            assert (await asyncio.wait_for(bus.inbound.get(), 2))["type"] == "activate"
            await ws.send("garbage")
        await bus.stop()

    asyncio.run(run())
