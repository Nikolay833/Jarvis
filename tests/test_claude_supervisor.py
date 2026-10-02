"""Claude supervisor: progress, change summary, voice approval of permission prompts, hooks installer, fast paths."""

import asyncio
import importlib.util
import io
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claude_fixtures import A, T, U, home, make, proj  # noqa: F401
from jarvis import claude_hook, claude_progress as cp
from jarvis.claude_watch import ASK, ALLOW, DENY, ClaudeWatch, permission_texts
from jarvis.config import ClaudeWatchConfig
from jarvis.safety import classify_shell
from jarvis.tools import claude_sessions_tools as cst
from jarvis.tools.registry import ToolError


def use(i, name, **inp):
    return A([{"type": "tool_use", "id": i, "name": name, "input": inp}])


def res(i, text, err=False):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": i, "content": text, "is_error": err}]}}


def ts(n):
    return f"2026-01-01T10:{n // 60:02d}:{n % 60:02d}Z"


def session_entries(cwd, fail=False):
    return [
        U("fix the map", cwd=str(cwd), timestamp=ts(0)),
        {**use("1", "Read", file_path="/p/map.ts"), "timestamp": ts(5)}, res("1", "..."),
        {**use("2", "Edit", file_path="/p/map.ts"), "timestamp": ts(20)}, res("2", "ok"),
        {**use("3", "Write", file_path="/p/map.css"), "timestamp": ts(40)}, res("3", "File created successfully at: /p/map.css"),
        {**use("4", "Bash", command="pytest -q"), "timestamp": ts(60)},
        res("4", "2 failed, 5 passed in 1s" if fail else "7 passed in 1s", err=fail),
        {**use("5", "Bash", command="git status"), "timestamp": ts(70)}, res("5", "clean"),
        {**use("6", "Edit", file_path="/p/map.ts"), "timestamp": ts(80)},  # still running
    ]


# ---- progress extraction ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cmd,yes", [("pytest -q", 1), ("python -m pytest tests/", 1), ("npm test", 1), ("npm run test", 1),
                                     ("cargo test --release", 1), ("go test ./...", 1), ("npx vitest run", 1),
                                     ("git status", 0), ("npm install", 0), ("ls tests", 0)])
def test_is_test_command(cmd, yes):
    assert cp.is_test_command(cmd) == bool(yes)


@pytest.mark.parametrize("out,err,want", [
    ("===== 7 passed in 1.2s =====", False, cp.PASS), ("2 failed, 5 passed", False, cp.FAIL),
    ("test result: ok. 3 passed", False, cp.PASS), ("test result: FAILED. 1 passed; 1 failed", False, cp.FAIL),
    ("Tests: 1 failed, 4 passed, 5 total", False, cp.FAIL), ("Tests: 5 passed, 5 total", False, cp.PASS),
    ("ok  \texample.com/pkg\t0.2s", False, cp.PASS), ("--- FAIL: TestX", False, cp.FAIL),
    ("", False, cp.UNKNOWN), ("boom", True, cp.FAIL), ("something odd", False, cp.UNKNOWN)])
def test_outcome(out, err, want):
    assert cp.test_outcome(out, err) == want


def test_transcript_progress_and_sentence(home):
    cwd = proj(home, "Jarvis")
    f = make(home, cwd, "s1", session_entries(cwd))
    recs, ts0 = cp.turn_records(str(f))
    p = cp.summarize(recs, ts0)
    assert [basename for basename in map(lambda x: x.rsplit("/", 1)[1], p.files)] == ["map.ts", "map.css"]
    assert p.created == ["/p/map.css"] and p.commands == 1 and p.tests == [cp.PASS]
    assert p.current is not None and p.current.name == "Edit"
    said = cp.progress_sentence(p, "Jarvis", busy=True, now=ts0 + 360)
    assert said == ("Claude has been at it for 6 minutes in Jarvis: two files edited, one of them new, "
                    "one shell command run, tests ran once, passing. He's currently editing map.ts.")


def test_progress_turn_resets_at_new_user_message(home):
    cwd = proj(home, "Jarvis")
    entries = session_entries(cwd) + [U("now the docs", cwd=str(cwd), timestamp=ts(100)),
                                      {**use("7", "Edit", file_path="/p/README.md"), "timestamp": ts(110)}, res("7", "ok")]
    recs, _ = cp.turn_records(str(make(home, cwd, "s1", entries)))
    assert [r.file for r in recs] == ["/p/README.md"]


def test_progress_failing_tests_and_multiple_runs(home):
    p = cp.summarize([cp.ToolRec("Bash", command="pytest", result="1 failed", is_error=True),
                      cp.ToolRec("Bash", command="npm test", result="Tests: 4 passed")])
    assert cp.tests_clause(p) == "tests ran twice, last run passing"
    p = cp.summarize([cp.ToolRec("Bash", command="pytest", result="1 passed"),
                      cp.ToolRec("Bash", command="pytest", result="3 failed")])
    assert cp.tests_clause(p) == "tests ran twice, last run failing" and cp.tests_clause(p, short=True) == "tests failing"
    assert cp.progress_sentence(cp.Progress(), "X", busy=False).startswith("Claude has nothing to report")


def test_hook_event_records_and_stop_announcement():
    w = ClaudeWatch(ClaudeWatchConfig(min_turn_seconds=0), clock=lambda: 1000.0, turn_seconds_fn=lambda p: None,
                    records_fn=lambda p: [])
    base = {"session_id": "s1", "cwd": "C:\\p\\Jarvis"}
    for ev, tool, extra in [("PostToolUse", "Edit", {"file_path": "C:\\p\\a.py"}), ("PostToolUse", "Edit", {"file_path": "C:\\p\\a.py"}),
                            ("PostToolUse", "Write", {"file_path": "C:\\p\\b.py", "tool_result": "File created successfully"}),
                            ("PostToolUse", "Bash", {"command": "pytest", "tool_result": "4 passed in 1s"}),
                            ("PostToolUseFailure", "Bash", {"command": "pytest", "tool_result": "boom"}),
                            ("PostToolUse", "Edit", {"file_path": "C:\\p\\c.py"})]:
        assert w.handle({**base, "event": ev, "tool_name": tool, **extra}) is None
    assert len(w.live_records("s1")) == 6
    out = w.handle({**base, "event": "Stop", "last_assistant_message": "All done. Really."})
    assert out == "Sir, Claude has finished in Jarvis: three files changed, tests failing. He says: All done."
    assert w.live_records("s1") == []


def test_stop_falls_back_to_transcript_records():
    recs = [cp.ToolRec("Edit", file="/x/a.py"), cp.ToolRec("Bash", command="pytest", result="2 passed")]
    w = ClaudeWatch(ClaudeWatchConfig(min_turn_seconds=0), turn_seconds_fn=lambda p: None, records_fn=lambda p: recs)
    out = w.handle({"event": "Stop", "session_id": "q", "cwd": "/x/Jarvis", "last_assistant_message": "Done.",
                    "transcript_path": "t"})
    assert out == "Sir, Claude has finished in Jarvis: one file changed, tests passing. He says: Done."


def test_hook_build_payload_for_post_tool_use():
    data = {"hook_event_name": "PostToolUse", "session_id": "s", "cwd": "/p", "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"}, "tool_response": {"type": "text", "text": "x" * 4000 + "7 passed"}}
    pl = claude_hook.build_payload(data)
    assert pl["tool_name"] == "Bash" and pl["command"] == "pytest -q" and pl["tool_result"].endswith("7 passed")
    assert len(pl["tool_result"]) == 1500
    fail = claude_hook.build_payload({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
                                      "tool_input": {"command": "x"}, "error": "exit 1"})
    assert fail["tool_result"] == "exit 1"
    assert "tool_name" not in claude_hook.build_payload({"hook_event_name": "Stop"})


def test_progress_tool(home):
    cwd = proj(home, "Jarvis")
    make(home, cwd, "s1", session_entries(cwd), mtime=time.time())
    said = cst.claude_progress(project="jarvis")
    assert "in Jarvis" in said and "two files edited" in said and "tests ran once, passing" in said
    assert "currently editing map.ts" in said
    with pytest.raises(ToolError):
        cst.claude_progress(project="nonexistent project")


# ---- change summary --------------------------------------------------------------------------------------------
def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.email=a@b.c", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def test_changes_with_git_repo(home):
    cwd = proj(home, "Jarvis")
    git(cwd, "init", "-q")
    (cwd / "map.ts").write_text("a\nb\nc\nd\n")
    (cwd / "other.txt").write_text("o\n")
    git(cwd, "add", ".")
    git(cwd, "commit", "-qm", "init")
    (cwd / "map.ts").write_text("a\nB\nc\nd\ne\nf\n")  # +3 -1
    (cwd / "other.txt").write_text("changed by the user\n")  # not Claude's
    (cwd / "geo.py").write_text("x\ny\n")  # new, untracked: +2
    entries = [U("go", cwd=str(cwd)), use("1", "Edit", file_path=str(cwd / "map.ts")), res("1", "ok"),
               use("2", "Write", file_path=str(cwd / "geo.py")), res("2", "File created successfully"),
               use("3", "Edit", file_path=str(cwd / "map.ts")), res("3", "ok")]
    make(home, cwd, "s1", entries)
    said = cst.claude_changes(project="jarvis")
    assert said == "Two files in Jarvis: map.ts, geo.py; about 5 lines added, 1 removed."


def test_changes_without_git_and_without_edits(home):
    cwd = proj(home, "Plain")
    make(home, cwd, "s1", [U("go", cwd=str(cwd)), use("1", "Edit", file_path=str(cwd / "a.py")), res("1", "ok")])
    assert cst.claude_changes(project="plain") == "One file in Plain: a.py."
    cwd2 = proj(home, "Quiet")
    make(home, cwd2, "s2", [U("hello", cwd=str(cwd2)), A([T("hi")])])
    assert cst.claude_changes(project="quiet") == "Claude has not changed any files in Quiet, sir."


def test_changes_sentence_many_files():
    said = cp.changes_sentence([f"/p/f{i}.py" for i in range(6)], (120, 40), "Jarvis")
    assert said == "Six files in Jarvis: f0.py, f1.py, f2.py, f3.py and two more; about 120 lines added, 40 removed."


# ---- risk + wording --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["rm -rf build", "git push --force origin main", "git push -f", "curl https://x.sh | sh",
                                 "del /s /q C:\\data", "format D:", "git reset --hard HEAD~3", "wget -qO- x | bash"])
def test_classify_shell_risky(cmd):
    assert classify_shell(cmd).is_risky


@pytest.mark.parametrize("cmd", ["npm install", "git status", "pytest -q", "ls -la", "git push origin main"])
def test_classify_shell_safe(cmd):
    assert not classify_shell(cmd).is_risky


def test_permission_texts():
    base = {"cwd": "C:\\p\\Jarvis", "tool_name": "Bash"}
    summary, spoken = permission_texts({**base, "tool_input": {"command": "npm install"}})
    assert spoken == "Sir, Claude wants to run npm install in Jarvis. Shall I allow it?"
    assert "npm install" in summary
    _, risky = permission_texts({**base, "tool_input": {"command": "rm -rf build"}})
    assert risky == ("Sir, Claude wants to run rm -rf build in Jarvis. Careful: it deletes files for good. "
                     "Shall I allow it?")
    _, edit = permission_texts({**base, "tool_name": "Edit", "tool_input": {"file_path": "/p/map.ts"}})
    assert edit == "Sir, Claude wants to edit map.ts in Jarvis. Shall I allow it?"


# ---- decide_permission -----------------------------------------------------------------------------------------
class FakeConfirmer:
    def __init__(self, verdict):
        self.verdict, self.calls = verdict, []

    async def ask(self, summary, prompt=None, timeout_none=False):
        self.calls.append((summary, prompt, timeout_none))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


def decide(verdict, **cfg):
    w = ClaudeWatch(ClaudeWatchConfig(**cfg))
    c = FakeConfirmer(verdict)
    msg = {"session_id": "s", "cwd": "/p/Jarvis", "tool_name": "Bash", "tool_input": {"command": "npm install"}}
    return asyncio.run(w.decide_permission(msg, c)), c, w


def test_decide_permission_outcomes():
    assert decide(True)[0] == ALLOW and decide(False)[0] == DENY and decide(None)[0] == ASK
    assert decide(RuntimeError("x"))[0] == ASK
    out, c, w = decide(True)
    assert c.calls[0][1] == "Sir, Claude wants to run npm install in Jarvis. Shall I allow it?" and c.calls[0][2] is True
    assert not w.asking
    out, c, _ = decide(True, voice_approval=False)
    assert out == ASK and c.calls == []  # disabled: never even asks
    assert decide(True, enabled=False)[0] == ASK


def test_permission_notification_not_announced_while_asking():
    w = ClaudeWatch(ClaudeWatchConfig(), clock=lambda: 5.0)
    w.asking.add("s1")
    msg = {"event": "Notification", "session_id": "s1", "cwd": "/p/Jarvis", "notification_type": "permission_prompt"}
    assert w.handle(msg) is None and w.pending_for("s1") is not None


# ---- hook round trip -------------------------------------------------------------------------------------------
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PERM = {"hook_event_name": "PermissionRequest", "session_id": "s1", "cwd": "/p/Jarvis", "tool_name": "Bash",
        "tool_input": {"command": "npm install"}}


def test_permission_round_trip_over_real_bus():
    from jarvis.bus import EventBus

    async def go(verdict):
        port = free_port()
        bus = EventBus("127.0.0.1", port)
        await bus.start()
        w = ClaudeWatch(ClaudeWatchConfig())
        conf = FakeConfirmer(verdict)

        async def serve():
            msg = await bus.inbound.get()
            assert msg["type"] == "claude_permission" and msg["tool_name"] == "Bash"
            d = await w.decide_permission(msg, conf)
            bus.emit_nowait("claude_permission_result", id=msg["id"], decision=d)

        task = asyncio.create_task(serve())
        try:
            req = claude_hook.build_permission_request(PERM, "rid1")
            got = await asyncio.to_thread(claude_hook.request_decision, req, port, 5.0)
            await task
            return got, conf.calls
        finally:
            await bus.stop()

    got, calls = asyncio.run(go(True))
    assert got == "allow" and "npm install" in calls[0][1]
    assert asyncio.run(go(False))[0] == "deny"
    assert asyncio.run(go(None))[0] == "ask"


def test_permission_ignores_other_ids_and_times_out_to_ask():
    from jarvis.bus import EventBus

    async def go():
        port = free_port()
        bus = EventBus("127.0.0.1", port)
        await bus.start()
        try:
            async def noise():
                await bus.inbound.get()
                bus.emit_nowait("claude_permission_result", id="someone-else", decision="allow")
                bus.emit_nowait("level", rms=0.1)

            t = asyncio.create_task(noise())
            t0 = time.monotonic()
            got = await asyncio.to_thread(claude_hook.request_decision,
                                          claude_hook.build_permission_request(PERM, "mine"), port, 1.0)
            await t
            return got, time.monotonic() - t0
        finally:
            await bus.stop()

    got, took = asyncio.run(go())
    assert got == "ask" and 0.9 < took < 3


def test_permission_unreachable_bus_is_ask_quickly():
    t0 = time.monotonic()
    assert claude_hook.request_decision(claude_hook.build_permission_request(PERM, "x"), free_port(), 5.0) == "ask"
    assert time.monotonic() - t0 < 2.5


def test_decision_output_json():
    assert json.loads(claude_hook.decision_output("allow")) == {
        "hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}}
    assert json.loads(claude_hook.decision_output("deny"))["hookSpecificOutput"]["decision"] == {"behavior": "deny"}
    assert claude_hook.decision_output("ask") == "" and claude_hook.decision_output("garbage") == ""


def run_perm_hook(monkeypatch, capsys, decision, env=None, enabled=True):
    monkeypatch.setattr(claude_hook, "request_decision", lambda payload, port, timeout=45.0: decision)
    monkeypatch.setattr(claude_hook, "bus_port", lambda: 9999)
    monkeypatch.setattr(claude_hook, "voice_approval_enabled", lambda: enabled)
    monkeypatch.delenv("JARVIS_HEADLESS", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PERM)))
    code = claude_hook.main()
    return code, capsys.readouterr().out


def test_permission_hook_main_outputs(monkeypatch, capsys):
    code, out = run_perm_hook(monkeypatch, capsys, "allow")
    assert code == 0 and json.loads(out)["hookSpecificOutput"]["decision"]["behavior"] == "allow"
    assert run_perm_hook(monkeypatch, capsys, "deny")[1].count("deny") == 1
    for fallback in ("ask", "weird"):  # no decision: Claude prompts on screen
        assert run_perm_hook(monkeypatch, capsys, fallback) == (0, "")
    assert run_perm_hook(monkeypatch, capsys, "allow", enabled=False) == (0, "")  # voice_approval = false
    assert run_perm_hook(monkeypatch, capsys, "allow", env={"JARVIS_HEADLESS": "1"}) == (0, "")


def test_permission_hook_main_never_raises_or_allows_on_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(claude_hook, "request_decision", boom)
    monkeypatch.setattr(claude_hook, "voice_approval_enabled", lambda: True)
    monkeypatch.delenv("JARVIS_HEADLESS", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PERM)))
    assert claude_hook.main() == 0 and capsys.readouterr().out == ""


def test_config_voice_approval_default_and_toml():
    from jarvis.config import config_from_dict

    assert config_from_dict({}).claude_watch.voice_approval is True
    assert config_from_dict({"claude_watch": {"voice_approval": False}}).claude_watch.voice_approval is False


# ---- installer ---------------------------------------------------------------------------------------------------
def load_installer():
    path = Path(__file__).resolve().parent.parent / "scripts" / "install_claude_hooks.py"
    spec = importlib.util.spec_from_file_location("install_claude_hooks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PY = "/opt/jarvis/.venv/bin/python"


def test_installer_adds_supervisor_hooks():
    ih = load_installer()
    cmd = ih.build_command(PY)
    out = ih.merge({}, cmd)["hooks"]
    assert set(out) == {"Stop", "Notification", "PostToolUse", "PostToolUseFailure", "PermissionRequest"}
    perm = out["PermissionRequest"]
    assert perm == [{"matcher": "", "hooks": [{"type": "command", "command": cmd, "timeout": 60}]}]
    assert out["PostToolUse"][0]["matcher"] == ih.TOOL_MATCHER and "Bash" in ih.TOOL_MATCHER and "Edit" in ih.TOOL_MATCHER


def test_installer_idempotent_and_uninstall_keeps_foreign_hooks():
    ih = load_installer()
    mine = {"matcher": "Bash", "hooks": [{"type": "command", "command": "audit.sh"}]}
    base = {"theme": "dark", "hooks": {"PermissionRequest": [mine], "PostToolUse": [mine]}}
    once = ih.merge(base, ih.build_command(PY))
    assert ih.merge(once, ih.build_command(PY)) == once
    assert len(once["hooks"]["PermissionRequest"]) == 2 and len(once["hooks"]["PostToolUse"]) == 2
    assert ih.remove(once) == base
    assert ih.remove(ih.remove(once)) == base
    old = {"hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": ih.build_command(PY)}]}]}}
    upgraded = ih.merge(old, ih.build_command(PY))  # an old install gains the new events, Stop is not duplicated
    assert len(upgraded["hooks"]["Stop"]) == 1 and "PermissionRequest" in upgraded["hooks"]


def test_installer_run_backup_uninstall(tmp_path):
    ih = load_installer()
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"theme": "dark"}))
    assert "Installed" in ih.run(p, PY)
    assert len(list(tmp_path.glob("settings.json.bak-*"))) == 1
    assert "No change" in ih.run(p, PY)
    assert "Removed" in ih.run(p, PY, uninstall=True)
    assert json.loads(p.read_text()) == {"theme": "dark"}


# ---- fast paths + prompt -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,tool,project", [
    ("How's Claude doing?", "claude_progress", ""), ("how is claude doing in the website project", "claude_progress", "website"),
    ("Claude progress", "claude_progress", ""), ("Jarvis, give me a Claude progress report", "claude_progress", ""),
    ("how long has Claude been at it", "claude_progress", ""),
    ("what has Claude changed", "claude_changes", ""), ("what has claude changed in the website project", "claude_changes", "website"),
    ("which files did Claude touch", "claude_changes", ""), ("what did Claude change?", "claude_changes", ""),
    ("what is Claude doing", "claude_status", ""), ("is Claude done", "claude_status", "")])
def test_fast_paths(text, tool, project):
    from jarvis import fastpath

    fp = fastpath.match(text)
    assert fp is not None
    assert tool in repr(fp.action) and fp.speak_result
    if project:
        assert project in repr(fp.action)


def test_fast_path_does_not_hijack_other_questions():
    from jarvis import fastpath

    for text in ("how is claude different from chatgpt", "what is the weather", "how are you doing"):
        fp = fastpath.match(text)
        assert fp is None or "claude_progress" not in repr(fp.action)


def test_system_prompt_mentions_supervisor_tools():
    from jarvis.agent import SYSTEM_PROMPT

    assert "claude_progress" in SYSTEM_PROMPT and "claude_changes" in SYSTEM_PROMPT


def test_confirmer_custom_prompt_and_timeout_none():
    from jarvis.agent import Confirmer

    class Bus:
        async def emit(self, *a, **k):
            pass

    spoken = []

    async def speak(t):
        spoken.append(t)

    async def go():
        c = Confirmer(Bus(), speak, timeout=0.05)
        return await c.ask("do x", prompt="Custom?", timeout_none=True), await c.ask("do y")

    assert asyncio.run(go()) == (None, False)
    assert spoken[0] == "Custom?" and spoken[1].startswith("Sir, I'm about to do y")
