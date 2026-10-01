"""Out-of-band notifications (phone). Stub for milestone 1."""

from __future__ import annotations

import logging

log = logging.getLogger("jarvis.notify")


async def notify(title: str, body: str) -> None:
    """Send a notification outside the desktop.

    TODO(milestone 2): POST to an ntfy topic (httpx), e.g.
    httpx.post(f"{ntfy_url}/{topic}", content=body, headers={"Title": title}).
    """
    log.info("notify (stub): %s - %s", title, body)
