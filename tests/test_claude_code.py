import json

from jarvis.tools.claude_code import Job, JobManager, build_announcement, build_command, parse_line, shorten

LINES = [
    json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
    "",
    "not json",
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Looking at the code."}]}}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {}}]}}),
    json.dumps({"type": "user", "message": {"content": []}}),
    json.dumps({"type": "result", "subtype": "success", "is_error": False,
                "result": "Fixed the **bug** in main.py. Tests pass.", "total_cost_usd": 0.12, "session_id": "s1"}),
]


def run(lines):
    job = Job(id="a", folder="C:\\proj\\site", prompt="p")
    for line in lines:
        ev = parse_line(line)
        if ev:
            job.apply_event(ev)
    return job


def test_parse_line():
    assert parse_line("") is None
    assert parse_line("garbage") is None
    assert parse_line("[1]") is None
    assert parse_line('{"type":"x"}') == {"type": "x"}


def test_success_flow():
    job = run(LINES)
    assert job.session_id == "s1"
    assert job.last_text == "Looking at the code."
    assert job.last_tool == "Edit"
    assert job.has_result and not job.is_error
    assert job.cost_usd == 0.12
    JobManager.finish(job, 0)
    assert job.status == "done"
    msg = build_announcement(job)
    assert msg.startswith("Sir, the Claude job in site finished:")
    assert "**" not in msg and "Fixed the bug" in msg


def test_error_result():
    job = run([json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": ""})])
    JobManager.finish(job, 1)
    assert job.status == "failed"
    assert "failed" in build_announcement(job)


def test_no_result_is_failure():
    job = run(LINES[:4])
    job.stderr_tail = "boom"
    JobManager.finish(job, 1)
    assert job.status == "failed"
    assert "boom" not in build_announcement(job) or "failed" in build_announcement(job)


def test_cancel():
    job = run(LINES[:4])
    job.cancelled = True
    JobManager.finish(job, -15)
    assert job.status == "failed" and job.result == "Cancelled"


def test_command_and_shorten():
    cmd = build_command("claude", "do it", "acceptEdits")
    assert cmd[:2] == ["claude", "-p"] and "--output-format" in cmd and "stream-json" in cmd
    assert cmd[-2:] == ["--permission-mode", "acceptEdits"] and "--verbose" in cmd
    long = "One sentence here. " * 40
    out = shorten(long, 100)
    assert len(out) <= 100 and out.endswith(".")
