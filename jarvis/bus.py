"""WebSocket event bus. Core is the server, the orb (and later phone) are clients.

Outgoing events go through an ordered queue so `emit_nowait` is safe from any
thread. Incoming client messages land in `inbound`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

log = logging.getLogger("jarvis.bus")


class EventBus:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.last_state: str = "idle"
        self._clients: set[Any] = set()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._out: asyncio.Queue[dict[str, Any]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._pump_task: asyncio.Task | None = None

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._out = asyncio.Queue()
        self._server = await serve(self._handle, self.host, self.port)
        self._pump_task = asyncio.create_task(self._pump())
        log.info("event bus on ws://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._pump_task:
            self._pump_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._server = None

    # ---- outgoing --------------------------------------------------------
    def add_listener(self, cb: Callable[[dict[str, Any]], None]) -> None:
        """Local synchronous observer of every emitted event (tests, logging)."""
        self._listeners.append(cb)

    def emit_nowait(self, type: str, **fields: Any) -> None:
        """Thread-safe emit. Dropped (but still observed) if the server is not running."""
        msg = {"type": type, **fields}
        if type == "state":
            self.last_state = fields.get("state", self.last_state)
        for cb in self._listeners:
            try:
                cb(msg)
            except Exception:  # noqa: BLE001
                log.exception("bus listener failed")
        if self._loop is None or self._out is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._out.put_nowait(msg)
        else:
            self._loop.call_soon_threadsafe(self._out.put_nowait, msg)

    async def emit(self, type: str, **fields: Any) -> None:
        self.emit_nowait(type, **fields)

    async def broadcast(self, msg: dict[str, Any]) -> None:
        if not self._clients:
            return
        text = json.dumps(msg)
        clients = list(self._clients)
        results = await asyncio.gather(*(c.send(text) for c in clients), return_exceptions=True)
        for c, r in zip(clients, results):
            if isinstance(r, Exception):
                self._clients.discard(c)

    async def _pump(self) -> None:
        assert self._out is not None
        while True:
            msg = await self._out.get()
            try:
                await self.broadcast(msg)
            except Exception:  # noqa: BLE001
                log.exception("broadcast failed")

    # ---- incoming --------------------------------------------------------
    async def _handle(self, ws: Any) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "state", "state": self.last_state}))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(msg, dict) and isinstance(msg.get("type"), str):
                    self.inbound.put_nowait(msg)
        except Exception:  # noqa: BLE001 - connection closed
            pass
        finally:
            self._clients.discard(ws)
