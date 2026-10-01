"""Claude Code hook: forward Stop / Notification events to the running Jarvis over its event bus.

Run as `<venv python> -m jarvis.claude_hook` (installed by scripts/install_claude_hooks.py). Claude Code
passes the event as JSON on stdin. This must be fast and must never disturb Claude: stdlib only, a raw
WebSocket client (no `websockets` import), total budget about 2 seconds, every error swallowed, and the
exit code is ALWAYS 0 (exit code 2 from a Stop hook would stop Claude from finishing).

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

DEFAULT_PORT = 8765
TOTAL_TIMEOUT = 2.0
MAX_MESSAGE = 2000


def build_payload(data: dict) -> dict:
    return {
        "type": "claude_event",
        "event": str(data.get("hook_event_name") or ""),
        "session_id": str(data.get("session_id") or ""),
        "cwd": str(data.get("cwd") or ""),
        "notification_type": str(data.get("notification_type") or ""),
        "message": str(data.get("message") or "")[:MAX_MESSAGE],
        "last_assistant_message": str(data.get("last_assistant_message") or "")[:MAX_MESSAGE],
        "transcript_path": str(data.get("transcript_path") or ""),
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


def send_event(payload: dict, port: int, timeout: float = TOTAL_TIMEOUT) -> None:
    """Connect, WebSocket handshake, send one text frame, close. Raises on any failure."""
    deadline = time.monotonic() + timeout
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
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
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        sock.sendall(_frame(0x1, json.dumps(payload).encode("utf-8")))
        sock.sendall(_frame(0x8, struct.pack(">H", 1000)))
    finally:
        try:
            sock.close()
        except OSError:
            pass


def main() -> int:
    try:
        if os.environ.get("JARVIS_HEADLESS") == "1":
            return 0
        data = json.loads(sys.stdin.read() or "{}")
        if not isinstance(data, dict) or not data.get("hook_event_name"):
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
