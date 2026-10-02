"""Claude Code hook: forward Stop / Notification / PostToolUse events to the running Jarvis over its event bus,
and (PermissionRequest) let Jarvis ask the user by voice.

Run as `<venv python> -m jarvis.claude_hook` (installed by scripts/install_claude_hooks.py). Claude Code
passes the event as JSON on stdin. This must be fast and must never disturb Claude: stdlib only, a raw
WebSocket client (no `websockets` import), total budget about 2 seconds, every error swallowed, and the
exit code is ALWAYS 0 (exit code 2 from a Stop hook would stop Claude from finishing).

Observational events (everything but PermissionRequest) are fire-and-forget. For PermissionRequest the hook
sends a `claude_permission` request, WAITS up to PERMISSION_WAIT seconds for Jarvis's `claude_permission_result`
and prints the allow/deny decision JSON. Jarvis not running, a timeout, a disabled setting or any error means
no output at all, so Claude shows its normal on-screen prompt. It never allows anything on a failure.

Runs started by Jarvis itself (env JARVIS_HEADLESS=1) are not forwarded.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import time
import uuid

DEFAULT_PORT = 8765
TOTAL_TIMEOUT = 2.0
MAX_MESSAGE = 2000
PERMISSION_WAIT = 45.0
MAX_RESULT = 1500


def _result_text(data: dict) -> str:
    """Text of a tool result (PostToolUse tool_response / PostToolUseFailure error); the tail is what matters."""
    parts: list[str] = []
    resp = data.get("tool_response")
    if isinstance(resp, str):
        parts.append(resp)
    elif isinstance(resp, dict):
        for key in ("text", "stdout", "stderr", "output", "content", "error"):
            val = resp.get(key)
            if isinstance(val, str):
                parts.append(val)
    err = data.get("error")
    if isinstance(err, str):
        parts.append(err)
    return "\n".join(parts)[-MAX_RESULT:]


def build_payload(data: dict) -> dict:
    tool_input = data.get("tool_input") if isinstance(data.get("tool_input"), dict) else {}
    resp = data.get("tool_response")
    extra = {
        "tool_name": str(data.get("tool_name") or ""),
        "file_path": str(tool_input.get("file_path") or tool_input.get("notebook_path") or ""),
        "command": str(tool_input.get("command") or "")[:MAX_MESSAGE],
        "tool_result": _result_text(data),
        "is_error": bool(isinstance(resp, dict) and resp.get("is_error")),
    } if str(data.get("hook_event_name") or "").startswith("PostToolUse") else {}
    return {
        "type": "claude_event",
        "event": str(data.get("hook_event_name") or ""),
        "session_id": str(data.get("session_id") or ""),
        "cwd": str(data.get("cwd") or ""),
        "notification_type": str(data.get("notification_type") or ""),
        "message": str(data.get("message") or "")[:MAX_MESSAGE],
        "last_assistant_message": str(data.get("last_assistant_message") or "")[:MAX_MESSAGE],
        "transcript_path": str(data.get("transcript_path") or ""),
        **extra,
    }


def bus_port() -> int:
    try:
        from .config import load_config

        return int(load_config().bus.port)
    except Exception:  # noqa: BLE001
        return DEFAULT_PORT


def _frame(opcode: int, payload: bytes) -> bytes:
    """One masked client frame (RFC 6455)."""
    head = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    mask = os.urandom(4)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _connect(port: int, deadline: float) -> tuple[socket.socket, bytes]:
    """TCP connect plus WebSocket handshake: (socket, bytes already read after the handshake). Raises on failure."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=max(0.05, deadline - time.monotonic()))
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            chunk = sock.recv(4096)
            if not chunk:
                raise OSError("connection closed during handshake")
            buf += chunk
        if b" 101 " not in buf.split(b"\r\n", 1)[0] + b" ":
            raise OSError("websocket handshake refused")
        return sock, buf.split(b"\r\n\r\n", 1)[1]  # (the server may send a frame right after the handshake)
    except BaseException:
        sock.close()
        raise


def send_event(payload: dict, port: int, timeout: float = TOTAL_TIMEOUT) -> None:
    """Connect, WebSocket handshake, send one text frame, close. Raises on any failure."""
    deadline = time.monotonic() + timeout
    sock, _rest = _connect(port, deadline)
    try:
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        sock.sendall(_frame(0x1, json.dumps(payload).encode("utf-8")))
        sock.sendall(_frame(0x8, struct.pack(">H", 1000)))
    finally:
        try:
            sock.close()
        except OSError:
            pass


class _Reader:
    """Reads unmasked server frames from a socket (text, ping, close; fragments are joined)."""

    def __init__(self, sock: socket.socket, buf: bytes = b"") -> None:
        self.sock = sock
        self.buf = buf

    def _need(self, n: int, deadline: float) -> bytes:
        while len(self.buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("no answer in time")
            self.sock.settimeout(max(0.05, left))
            chunk = self.sock.recv(65536)
            if not chunk:
                raise OSError("connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def frame(self, deadline: float) -> tuple[int, bool, bytes]:
        b0, b1 = self._need(2, deadline)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._need(2, deadline))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._need(8, deadline))[0]
        if n > 4_000_000:
            raise OSError("frame too large")
        mask = self._need(4, deadline) if b1 & 0x80 else b""
        data = self._need(n, deadline)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return b0 & 0x0F, bool(b0 & 0x80), data


def request_decision(payload: dict, port: int, timeout: float = PERMISSION_WAIT) -> str:
    """Send a `claude_permission` request and wait for the matching `claude_permission_result`.

    Returns "allow", "deny" or "ask". Anything else (refused connection, timeout, closed socket, garbage) is "ask"."""
    deadline = time.monotonic() + timeout
    try:
        sock, rest = _connect(port, min(deadline, time.monotonic() + TOTAL_TIMEOUT))
    except BaseException:  # noqa: BLE001
        return "ask"
    try:
        reader = _Reader(sock, rest)
        sock.settimeout(TOTAL_TIMEOUT)
        sock.sendall(_frame(0x1, json.dumps(payload).encode("utf-8")))
        text = b""
        while True:
            op, fin, data = reader.frame(deadline)
            if op == 0x8:
                return "ask"
            if op == 0x9:  # ping: answer, or the server drops the connection
                sock.sendall(_frame(0xA, data))
                continue
            if op in (0x1, 0x0):
                text += data
                if not fin:
                    continue
                msg, text = text, b""
                try:
                    obj = json.loads(msg.decode("utf-8"))
                except ValueError:
                    continue
                if (isinstance(obj, dict) and obj.get("type") == "claude_permission_result"
                        and obj.get("id") == payload.get("id")):
                    return obj["decision"] if obj.get("decision") in ("allow", "deny") else "ask"
    except BaseException:  # noqa: BLE001
        return "ask"
    finally:
        try:
            sock.sendall(_frame(0x8, struct.pack(">H", 1000)))
            sock.close()
        except OSError:
            pass


def build_permission_request(data: dict, request_id: str) -> dict:
    ti = data.get("tool_input") if isinstance(data.get("tool_input"), dict) else {}
    slim = {k: str(ti[k])[:MAX_MESSAGE] for k in ("command", "file_path", "notebook_path", "url", "description")
            if k in ti}
    return {"type": "claude_permission", "id": request_id, "timeout": PERMISSION_WAIT,
            "session_id": str(data.get("session_id") or ""), "cwd": str(data.get("cwd") or ""),
            "tool_name": str(data.get("tool_name") or ""), "tool_input": slim}


def decision_output(decision: str) -> str:
    """The hook's stdout for a decision; empty (no decision) for anything but allow / deny."""
    if decision not in ("allow", "deny"):
        return ""
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                              "decision": {"behavior": decision}}})


def voice_approval_enabled() -> bool:
    try:
        from .config import load_config

        cfg = load_config().claude_watch
        return bool(cfg.enabled and cfg.voice_approval)
    except Exception:  # noqa: BLE001
        return True


def handle_permission(data: dict) -> None:
    if not voice_approval_enabled():
        return
    out = decision_output(request_decision(build_permission_request(data, uuid.uuid4().hex), bus_port()))
    if out:
        sys.stdout.write(out)
        sys.stdout.flush()


def main() -> int:
    try:
        if os.environ.get("JARVIS_HEADLESS") == "1":
            return 0
        data = json.loads(sys.stdin.read() or "{}")
        if not isinstance(data, dict) or not data.get("hook_event_name"):
            return 0
        if data.get("hook_event_name") == "PermissionRequest":
            handle_permission(data)
            return 0
        send_event(build_payload(data), bus_port())
    except BaseException:  # noqa: BLE001 - never disturb Claude
        pass
    return 0


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        pass
    sys.exit(0)
