"""Drive Claude Code headlessly as background jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..notify import notify
from .context import NO_WINDOW, ctx, emit
from .registry import ToolError, tool

log = logging.getLogger("jarvis.claude_code")

STDOUT_LIMIT = 16 * 1024 * 1024  # stream-json lines can be large


def parse_line(line: str) -> dict[str, Any] | None:
    """Parse one stream-json line; None for blank or non-JSON lines."""
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


@dataclass
class Job:
    id: str
    folder: str
    prompt: str
    status: str = "running"  # running | done | failed
    last_text: str = ""
    last_tool: str = ""
    result: str = ""
    is_error: bool = False
    session_id: str = ""
    cost_usd: float | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancelled: bool = False
    stderr_tail: str = ""
    proc: Any = field(default=None, repr=False)
    has_result: bool = False

    def apply_event(self, ev: dict[str, Any]) -> bool:
        """Update from a stream-json event. Returns True if a visible change happened."""
        t = ev.get("type")
        if ev.get("session_id"):
            self.session_id = str(ev["session_id"])
        if t == "assistant":
            changed = False
            for block in (ev.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    self.last_text = block["text"].strip()
                    changed = True
                elif block.get("type") == "tool_use":
                    self.last_tool = str(block.get("name", ""))
                    changed = True
            return changed
        if t == "result":
            self.has_result = True
            self.is_error = bool(ev.get("is_error")) or str(ev.get("subtype", "success")) != "success"
            self.result = str(ev.get("result") or "").strip()
            if ev.get("total_cost_usd") is not None:
                self.cost_usd = float(ev["total_cost_usd"])
            return True
        return False

    @property
    def folder_name(self) -> str:
        parts = [p for p in re.split(r"[\\/]", self.folder) if p]
        return parts[-1] if parts else self.folder

    def summary(self) -> str:
        if self.status == "running":
            if self.last_tool:
                return f"working, last used {self.last_tool}"
            return self.last_text[:200] or "working"
        text = self.result or self.last_text or self.stderr_tail
        return shorten(text, 400) or ("finished" if self.status == "done" else "failed")


_MD = re.compile(r"[*_`#>]+")


def shorten(text: str, limit: int = 280) -> str:
    """Plain, speakable, at most `limit` chars, cut at a sentence end when possible."""
    text = _MD.sub("", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > limit // 3:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def build_announcement(job: Job) -> str:
    if job.status == "done":
        return f"Sir, the Claude job in {job.folder_name} finished: {shorten(job.result or job.last_text or 'no summary given')}"
    return f"Sir, the Claude job in {job.folder_name} failed: {shorten(job.result or job.stderr_tail or job.last_text or 'no details available')}"


def build_command(binary: str, prompt: str, permission_mode: str) -> list[str]:
    return [binary, "-p", prompt, "--output-format", "stream-json", "--verbose",
            "--permission-mode", permission_mode]


def resolve_binary(configured: str) -> str:
    """Path of the claude executable (PATH lookup finds the Windows claude.cmd shim), or ToolError."""
    binary = shutil.which(configured) or (configured if os.path.isfile(configured) else None)
    if not binary:
        raise ToolError(f"Claude Code binary '{configured}' not found; set claude_code.binary in config.toml")
    return binary


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def _emit(self, job: Job) -> None:
        emit("job", id=job.id, kind="claude_code", status=job.status, summary=job.summary())

    async def start(self, folder: str, prompt: str) -> Job:
        cfg = ctx.config.claude_code
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            raise ToolError(f"folder does not exist: {folder}")
        binary = resolve_binary(cfg.binary)
        proc = await asyncio.create_subprocess_exec(
            *build_command(binary, prompt, cfg.permission_mode),
            cwd=folder,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STDOUT_LIMIT,
            creationflags=NO_WINDOW,
        )
        job = Job(id=uuid.uuid4().hex[:6], folder=os.path.abspath(folder), prompt=prompt, proc=proc)
        self.jobs[job.id] = job
        self._emit(job)
        task = asyncio.create_task(self._watch(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _read_stderr(self, job: Job) -> None:
        assert job.proc and job.proc.stderr
        data = await job.proc.stderr.read()
        job.stderr_tail = data.decode("utf-8", errors="replace").strip()[-500:]

    async def _watch(self, job: Job) -> None:
        proc = job.proc
        err_task = asyncio.create_task(self._read_stderr(job))
        try:
            assert proc.stdout
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                ev = parse_line(raw.decode("utf-8", errors="replace"))
                if ev and job.apply_event(ev) and ev.get("type") == "assistant":
                    self._emit(job)
            code = await proc.wait()
            await err_task
        except Exception as exc:  # noqa: BLE001
            log.exception("claude job watcher failed")
            job.stderr_tail = str(exc)
            code = -1
        self.finish(job, code)
        self._emit(job)
        if job.cancelled:
            return
        message = build_announcement(job)
        try:
            await ctx.speak(message)
        except Exception:  # noqa: BLE001
            log.exception("could not announce job")
        await notify(f"Claude job {job.status}", message)

    @staticmethod
    def finish(job: Job, returncode: int) -> None:
        job.finished = time.time()
        if job.cancelled:
            job.status, job.result = "failed", job.result or "Cancelled"
        elif job.has_result and not job.is_error:
            job.status = "done"  # result event is authoritative even if exit code is odd
        else:
            job.status = "failed"
            job.is_error = True

    async def cancel(self, job: Job) -> None:
        job.cancelled = True
        if job.proc and job.proc.returncode is None:
            job.proc.terminate()
            try:
                await asyncio.wait_for(job.proc.wait(), 5)
            except asyncio.TimeoutError:
                job.proc.kill()


jobs = JobManager()


def _describe(job: Job) -> str:
    age = int((job.finished or time.time()) - job.started)
    return f"[{job.id}] {job.folder_name}: {job.status} ({age}s) - {job.summary()}"


@tool("Start a Claude Code agent in a project folder to carry out a coding task in the background. "
      "Needs approval. Jarvis announces the result when it finishes.", risk="risky")
async def claude_code_run(folder: str, prompt: str) -> str:
    """Start a Claude Code job.

    Args:
        folder: Project folder to work in.
        prompt: The task for Claude Code, in full.
    """
    job = await jobs.start(folder, prompt)
    return f"Started Claude Code job {job.id} in {job.folder_name}. I will report when it finishes."


@tool("Report the status of Claude Code jobs (all recent jobs, or one by id).")
def claude_code_status(job_id: str | None = None) -> str:
    """Claude Code job status.

    Args:
        job_id: Optional job id; omit to list all jobs.
    """
    if job_id:
        job = jobs.jobs.get(job_id)
        if not job:
            raise ToolError(f"no job with id {job_id}")
        return _describe(job)
    if not jobs.jobs:
        return "No Claude Code jobs"
    return "\n".join(_describe(j) for j in list(jobs.jobs.values())[-10:])


@tool("Cancel a running Claude Code job by id.")
async def claude_code_cancel(job_id: str) -> str:
    """Cancel a Claude Code job.

    Args:
        job_id: Id of the job to cancel.
    """
    job = jobs.jobs.get(job_id)
    if not job:
        raise ToolError(f"no job with id {job_id}")
    if job.status != "running":
        return f"Job {job_id} already {job.status}"
    await jobs.cancel(job)
    return f"Cancelled job {job_id}"
