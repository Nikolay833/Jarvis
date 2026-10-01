import asyncio
import importlib.util
import io
import json
import socket
import sys
import time
from pathlib import Path

import pytest
from claude_fixtures import A, T, U, home, make, proj  # noqa: F401

from jarvis import claude_hook, claude_sessions as cs
from jarvis.claude_watch import ClaudeWatch
from jarvis.config import ClaudeWatchConfig, Config, config_from_dict
from jarvis.tools import claude_sessions_tools as cst
from jarvis.tools.registry import ToolError


@pytest.fixture
def opened(monkeypatch):
    calls = []

    def fake(folder, prompt="", args=None):
        calls.append({"folder": str(folder), "prompt": prompt, "args": list(args or [])})
        return Path(folder) if folder else Path.home()

    monkeypatch.setattr(cst, "open_claude_terminal", fake)
    cst._offer["ids"] = []
    return calls


@pytest.fixture
def world(home):  # noqa: F811
    jarvis = proj(home, "Documents", "Ai on pc", "Jarvis")
    site = proj(home, "Desktop", "site")
    now = time.time()
    make(home, jarvis, "j-login", [U("fix the login bug", cwd=str(jarvis)), A([T("Fixed it.")])], now - 7200)
    make(home, jarvis, "j-dark", [U("add dark mode", cwd=str(jarvis)), A([T("Done.")])], now - 3600)
    make(home, site, "s-login", [U("login page broken", cwd=str(site)), A([T("Looking.")])], now - 900)
    return jarvis, site


def test_tools_registered_and_safe():
    from jarvis.tools import load_all

    reg = load_all()
    for n in ("claude_sessions", "claude_open_session", "claude_continue", "claude_new_session", "claude_status",
              "claude_ask_session", "claude_code_history", "claude_terminal"):
        assert reg.get(n) is not None and reg.get(n).risk == "safe", n


def test_claude_sessions_lists_spoken(world):
    out = cst.claude_sessions()
    lines = out.splitlines()
    assert lines[1].startswith("1. Looking") or lines[1].startswith("1. login page broken")
    assert "Jarvis" in out and "hours ago" in out
    jar = cst.claude_sessions(project="the jarvis project", count=1)
    assert "dark mode" in jar and "login" not in jar
    with pytest.raises(ToolError):
        cst.claude_sessions(project="nonexistent thing")


def test_open_session_unique_match_resumes(world, opened):
    jarvis, _ = world
    out = cst.claude_open_session(topic="dark mode", project="jarvis", prompt="continue please")
    assert opened == [{"folder": str(jarvis), "prompt": "continue please", "args": ["--resume", "j-dark"]}]
    assert "Opened" in out and "add dark mode" in out


def test_open_session_ambiguous_then_choice(world, opened):
    out = cst.claude_open_session(topic="login")
    assert opened == []
    assert out.startswith("I found 2 sessions: 1. ") and out.endswith("Which one, sir?")
    assert "2. " in out
    cst.claude_open_session(topic="", project="", choice=2)  # follow-up: uses the offered list
    assert len(opened) == 1 and opened[0]["args"][0] == "--resume"
    assert opened[0]["args"][1] in ("j-login", "s-login")


def test_open_session_project_filter_disambiguates(world, opened):
    jarvis, _ = world
    cst.claude_open_session(topic="login", project="Jarvis")
    assert opened[0]["args"] == ["--resume", "j-login"] and opened[0]["folder"] == str(jarvis)


def test_open_session_empty_topic_most_recent(world, opened):
    jarvis, site = world
    cst.claude_open_session()
    assert opened[-1]["args"] == ["--resume", "s-login"]
    cst.claude_open_session(project="jarvis")
    assert opened[-1]["args"] == ["--resume", "j-dark"]


def test_open_session_errors(world, opened):
    with pytest.raises(ToolError):
        cst.claude_open_session(topic="quantum fridge")
    with pytest.raises(ToolError):
        cst.claude_open_session(topic="login", project="nowhere at all")
    with pytest.raises(ToolError):
        cst.claude_open_session(topic="login", choice=5)
    assert opened == []


def test_continue_and_new_session(world, opened):
    jarvis, site = world
    cst.claude_continue(project="jarvis", prompt="go on")
    assert opened[-1] == {"folder": str(jarvis), "prompt": "go on", "args": ["--continue"]}
    cst.claude_continue()
    assert opened[-1]["folder"] == str(site)  # most recently active project
    cst.claude_new_session(project="jarvis", prompt="Please fix the flaky audio test in the recorder")
    assert opened[-1]["args"] == ["--name", "fix flaky audio test"]
    cst.claude_new_session(project="site", prompt="x", name="My Name")
    assert opened[-1]["args"] == ["--name", "My Name"]
    cst.claude_new_session()
    assert opened[-1]["args"] == [] and opened[-1]["folder"] == ""


def test_derive_name():
    assert cst.derive_name("Can you please fix the login bug now") == "fix login bug"
    assert cst.derive_name("") == ""
    assert cst.derive_name("refactor") == "refactor"


def test_status_idle_busy_and_pending(world):
    from jarvis.claude_watch import watch

    out = cst.claude_status(project="jarvis")
    assert "Jarvis" in out and "idle" in out and "Last said: Done." in out
    f = Path(cs.scan_sessions()[0].path)  # s-login: make it look busy
    with f.open("a") as fh:
        fh.write(json.dumps(A([{"type": "tool_use", "name": "Bash", "input": {}}])) + "\n")
    out = cst.claude_status()
    assert "busy" in out and "Last tool: Bash" in out
    watch.handle({"event": "Notification", "notification_type": "permission_prompt", "session_id": "s-login",
                  "cwd": "/x/site", "message": "Claude needs permission to use Bash"})
    try:
        out = cst.claude_status()
        assert "waiting for your permission" in out and "Permission request: Claude needs permission" in out
    finally:
        watch.pending.clear()
    with pytest.raises(ToolError):
        cst.claude_status(project="nope nope")


def test_ask_session_headless(world, monkeypatch):
    jarvis, _ = world
    seen = {}

    async def fake_run(cmd, cwd, message, timeout=0):
        seen.update(cmd=cmd, cwd=cwd, message=message, timeout=timeout)
        return 0, json.dumps({"type": "result", "result": "It is **fixed**.", "session_id": "j-dark"}), ""

    monkeypatch.setattr(cst.claude_chat, "run_cli", fake_run)
    monkeypatch.setattr(cst, "resolve_binary", lambda b: "claude")
    out = asyncio.run(cst.claude_ask_session(topic="dark mode", project="jarvis", question="is it done?"))
    assert out.endswith("It is fixed.")
    assert seen["cmd"] == ["claude", "-p", "--resume", "j-dark", "--output-format", "json"]
    assert seen["cwd"] == str(jarvis) and seen["message"] == "is it done?" and seen["timeout"] == 180
    with pytest.raises(ToolError):
        asyncio.run(cst.claude_ask_session(topic="dark", question=""))


def test_headless_env_marks_subprocess():
    from jarvis.tools.claude_code import headless_env

    assert headless_env()["JARVIS_HEADLESS"] == "1"


# ---- hook ---------------------------------------------------------------------------------------------------------
STOP = {"hook_event_name": "Stop", "session_id": "abc", "cwd": "C:\\p\\Jarvis", "transcript_path": "t.jsonl",
        "last_assistant_message": "x" * 3000, "stop_reason": "end_turn"}


def run_hook(monkeypatch, data, env=None):
    sent = []
    monkeypatch.setattr(claude_hook, "send_event", lambda payload, port, timeout=2.0: sent.append((payload, port)))
    monkeypatch.setattr(claude_hook, "bus_port", lambda: 9999)
    monkeypatch.delenv("JARVIS_HEADLESS", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "stdin", io.StringIO(data if isinstance(data, str) else json.dumps(data)))
    return claude_hook.main(), sent


def test_hook_forwards_payload(monkeypatch):
    code, sent = run_hook(monkeypatch, STOP)
    assert code == 0 and len(sent) == 1
    payload, port = sent[0]
    assert port == 9999
    assert payload["type"] == "claude_event" and payload["event"] == "Stop" and payload["session_id"] == "abc"
    assert payload["cwd"] == "C:\\p\\Jarvis" and payload["transcript_path"] == "t.jsonl"
    assert len(payload["last_assistant_message"]) == 2000 and payload["notification_type"] == ""
    code, sent = run_hook(monkeypatch, {"hook_event_name": "Notification", "notification_type": "permission_prompt",
                                        "message": "ok?", "session_id": "s"})
    assert sent[0][0]["notification_type"] == "permission_prompt" and sent[0][0]["message"] == "ok?"


def test_hook_skips_headless_and_never_fails(monkeypatch):
    code, sent = run_hook(monkeypatch, STOP, env={"JARVIS_HEADLESS": "1"})
    assert code == 0 and sent == []
    for bad in ("", "not json", "[1]"):
        code, sent = run_hook(monkeypatch, bad)
        assert code == 0 and sent == []
    monkeypatch.setattr(claude_hook, "send_event", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(STOP)))
    assert claude_hook.main() == 0


def test_hook_unreachable_bus_is_quick_and_silent():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t0 = time.monotonic()
    with pytest.raises(OSError):
        claude_hook.send_event({"type": "claude_event"}, port)
    assert time.monotonic() - t0 < 2.5


def test_hook_reaches_real_bus():
    from jarvis.bus import EventBus

    async def go():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        bus = EventBus("127.0.0.1", port)
        await bus.start()
        try:
            payload = claude_hook.build_payload(STOP)
            await asyncio.to_thread(claude_hook.send_event, payload, port)
            return await asyncio.wait_for(bus.inbound.get(), 3)
        finally:
            await bus.stop()

    msg = asyncio.run(go())
    assert msg["type"] == "claude_event" and msg["event"] == "Stop" and msg["session_id"] == "abc"


# ---- watch / dispatch -----------------------------------------------------------------------------------------------
class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def mkwatch(turn=None, **cfg):
    clock = Clock()
    return ClaudeWatch(ClaudeWatchConfig(**cfg), clock=clock, turn_seconds_fn=lambda p: turn), clock


def stop_msg(text="Fixed the login bug. Also tidied imports.", sid="s1"):
    return {"event": "Stop", "session_id": sid, "cwd": "C:\\p\\Jarvis", "last_assistant_message": text,
            "transcript_path": "t"}


def notif(ntype, sid="s1"):
    return {"event": "Notification", "session_id": sid, "cwd": "C:\\p\\Jarvis", "notification_type": ntype,
            "message": "Claude needs your permission to use Bash"}


def test_watch_stop_announcement_first_sentence_25_words():
    w, _ = mkwatch()
    assert w.handle(stop_msg()) == "Sir, Claude finished in Jarvis: Fixed the login bug."
    w2, _ = mkwatch()
    long = " ".join(f"w{i}" for i in range(60)) + "."
    said = w2.handle(stop_msg(long))
    assert said.split(": ", 1)[1].count("w") == 25 and said.endswith("...")
    w3, _ = mkwatch()
    assert w3.handle(stop_msg("")) == "Sir, Claude finished in Jarvis."


def test_watch_notifications():
    w, _ = mkwatch()
    assert w.handle(notif("permission_prompt")) == "Sir, Claude needs your permission in Jarvis."
    assert w.handle(notif("agent_needs_input")) == "Sir, Claude has a question for you in Jarvis."
    assert w.handle(notif("elicitation_dialog", sid="s2")) == "Sir, Claude has a question for you in Jarvis."
    assert w.handle(notif("idle_prompt")) is None
    assert w.handle(notif("auth_success")) is None
    assert w.handle({"event": "SessionStart"}) is None


def test_watch_throttle_per_session_and_kind():
    w, clock = mkwatch()
    assert w.handle(stop_msg()) is not None
    clock.t += 10
    assert w.handle(stop_msg()) is None  # same session, same kind, within 30 s
    assert w.handle(stop_msg(sid="other")) is not None
    assert w.handle(notif("permission_prompt")) is not None  # different kind
    clock.t += 25
    assert w.handle(stop_msg()) is not None  # window elapsed


def test_watch_pending_permission_cleared_on_stop():
    w, _ = mkwatch()
    w.handle(notif("permission_prompt"))
    assert w.pending_for("s1").message.startswith("Claude needs")
    w.handle(stop_msg())
    assert w.pending_for("s1") is None


def test_watch_min_turn_seconds_and_unknown_duration():
    w, _ = mkwatch(turn=5.0, min_turn_seconds=20.0)
    assert w.handle(stop_msg()) is None  # short turn
    w, _ = mkwatch(turn=45.0, min_turn_seconds=20.0)
    assert w.handle(stop_msg()) is not None
    w, _ = mkwatch(turn=None, min_turn_seconds=20.0)
    assert w.handle(stop_msg()) is not None  # unknown duration: announce


def test_watch_config_switches():
    w, _ = mkwatch(enabled=False)
    assert w.handle(stop_msg()) is None and w.handle(notif("permission_prompt")) is None
    assert w.pending_for("s1") is not None  # still recorded for claude_status
    w, _ = mkwatch(announce_finish=False)
    assert w.handle(stop_msg()) is None and w.handle(notif("permission_prompt")) is not None
    w, _ = mkwatch(announce_permission=False)
    assert w.handle(notif("permission_prompt")) is None and w.handle(stop_msg()) is not None


def test_watch_config_loaded_from_toml_dict():
    cfg = config_from_dict({"claude_watch": {"min_turn_seconds": 5, "announce_finish": False}})
    assert cfg.claude_watch.min_turn_seconds == 5.0 and cfg.claude_watch.announce_finish is False
    assert Config().claude_watch.enabled is True


def test_dispatch_bus_announces_claude_events(monkeypatch):
    from jarvis.__main__ import Assistant
    from jarvis.claude_watch import watch

    spoken = []

    async def go():
        a = Assistant(Config(), voice=False)
        watch.cfg = ClaudeWatchConfig(min_turn_seconds=0)
        watch._turn_seconds = lambda p: None
        watch._last.clear()

        async def fake_announce(text):
            spoken.append(text)

        a.announce = fake_announce
        task = asyncio.create_task(a.dispatch_bus())
        a.bus.inbound.put_nowait({"type": "claude_event", **stop_msg(sid="d1")})
        a.bus.inbound.put_nowait({"type": "claude_event", **stop_msg(sid="d1")})  # duplicate: throttled
        a.bus.inbound.put_nowait({"type": "claude_event", **notif("permission_prompt", sid="d1")})
        a.bus.inbound.put_nowait({"type": "claude_event", **notif("idle_prompt", sid="d1")})
        await asyncio.sleep(0.1)
        task.cancel()
        watch.pending.clear()

    asyncio.run(go())
    assert spoken == ["Sir, Claude finished in Jarvis: Fixed the login bug.",
                      "Sir, Claude needs your permission in Jarvis."]


# ---- installer ------------------------------------------------------------------------------------------------------
def load_installer():
    path = Path(__file__).resolve().parent.parent / "scripts" / "install_claude_hooks.py"
    spec = importlib.util.spec_from_file_location("install_claude_hooks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PY = "C:\\Users\\Nikolay\\Documents\\Ai on pc\\Jarvis\\.venv\\Scripts\\python.exe"


def test_installer_command_quotes_path():
    ih = load_installer()
    cmd = ih.build_command(PY)
    assert cmd == '"' + PY.replace("\\", "/") + '" -m jarvis.claude_hook'
    assert "\\" not in cmd


def test_installer_merge_into_empty():
    ih = load_installer()
    out = ih.merge({}, ih.build_command(PY))
    stop = out["hooks"]["Stop"]
    assert stop == [{"matcher": "", "hooks": [{"type": "command", "command": ih.build_command(PY), "timeout": 10}]}]
    notes = out["hooks"]["Notification"]
    assert [n["matcher"] for n in notes] == ["permission_prompt", "agent_needs_input", "elicitation_dialog"]
    assert all(n["hooks"][0]["command"] == ih.build_command(PY) for n in notes)


def test_installer_keeps_other_settings_and_hooks_and_is_idempotent():
    ih = load_installer()
    other = {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
    base = {"model": "opus", "permissions": {"allow": ["Bash(ls)"]},
            "hooks": {"PreToolUse": [other], "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "notify-send done"}]}]}}
    once = ih.merge(base, ih.build_command(PY))
    assert once["model"] == "opus" and once["permissions"] == base["permissions"]
    assert once["hooks"]["PreToolUse"] == [other]
    assert once["hooks"]["Stop"][0]["hooks"][0]["command"] == "notify-send done"
    assert len(once["hooks"]["Stop"]) == 2
    twice = ih.merge(once, ih.build_command(PY))
    assert twice == once
    moved = ih.merge(once, ih.build_command("D:\\other path\\python.exe"))  # replaced, not duplicated
    assert len(moved["hooks"]["Stop"]) == 2 and len(moved["hooks"]["Notification"]) == 3
    assert "other path" in moved["hooks"]["Stop"][1]["hooks"][0]["command"]
    assert base["hooks"]["Stop"][0]["hooks"][0]["command"] == "notify-send done"  # input not mutated


def test_installer_uninstall_removes_only_ours():
    ih = load_installer()
    base = {"theme": "dark", "hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "keep me"}]}]}}
    installed = ih.merge(base, ih.build_command(PY))
    assert ih.remove(installed) == base
    assert ih.remove(ih.merge({"theme": "dark"}, ih.build_command(PY))) == {"theme": "dark"}
    shared = {"hooks": {"Stop": [{"matcher": "", "hooks": [
        {"type": "command", "command": "keep me"}, {"type": "command", "command": '"py" -m jarvis.claude_hook'}]}]}}
    assert ih.remove(shared)["hooks"]["Stop"][0]["hooks"] == [{"type": "command", "command": "keep me"}]


def test_installer_run_writes_backup_and_roundtrips(tmp_path):
    ih = load_installer()
    p = tmp_path / ".claude" / "settings.json"
    assert "Installed" in ih.run(p, PY)  # creates the file and folder
    assert json.loads(p.read_text())["hooks"]["Stop"]
    assert not list(p.parent.glob("settings.json.bak-*"))
    assert "No change" in ih.run(p, PY)
    p.write_text(json.dumps({"theme": "dark"}))
    ih.run(p, PY)
    backups = list(p.parent.glob("settings.json.bak-*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == {"theme": "dark"}
    assert "Removed" in ih.run(p, PY, uninstall=True)
    assert json.loads(p.read_text()) == {"theme": "dark"}


def test_installer_refuses_invalid_json(tmp_path):
    ih = load_installer()
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    with pytest.raises(SystemExit):
        ih.run(p, PY)
    assert p.read_text() == "{not json"


def test_system_prompt_routes_every_session_tool():
    from jarvis.agent import SYSTEM_PROMPT

    for name in ("claude_open_session", "claude_continue", "claude_new_session", "claude_status",
                 "claude_sessions", "claude_ask_session", "claude_terminal", "choice=N"):
        assert name in SYSTEM_PROMPT, name
