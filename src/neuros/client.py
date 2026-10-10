"""NeurOS Python client (the SDK's Python side, D-014): receive intents from the local engine.

    import asyncio
    from neuros.client import intents

    async def main():
        async for event in intents():            # ws://127.0.0.1:8765 by default
            if event["type"] == "intent" and event["command"] != "HOLD":
                print(event["command"], event["confidence"])

    asyncio.run(main())

Messages follow schema ``neuros.intent.v0`` (architecture §7.3): one ``hello`` on connect, an
``intent`` per decision (8 per second) and a ``status`` per engine state change. The client
reconnects by itself (with a growing pause) when the engine restarts, and skips messages that
aren't valid ``neuros.intent.v0`` JSON instead of raising (logged).

Needs the ``websockets`` package (``uv sync --extra engine``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from neuros.engine.events import SCHEMA
from neuros.errors import NeurOSConfigError

log = logging.getLogger(__name__)

DEFAULT_URL = "ws://127.0.0.1:8765"
MESSAGE_TYPES = ("hello", "intent", "status")


def parse(message: str | bytes) -> dict[str, Any] | None:
    """A ``neuros.intent.v0`` message as a dict, or None if it isn't one (logged)."""
    try:
        event = json.loads(message)
    except (ValueError, TypeError):
        log.warning("skipping a message that isn't JSON")
        return None
    if not isinstance(event, dict) or event.get("schema") != SCHEMA:
        log.warning("skipping a message with another schema: %r", event)
        return None
    if event.get("type") not in MESSAGE_TYPES:
        log.warning("skipping a message of unknown type: %r", event.get("type"))
        return None
    return event


async def intents(
    url: str = DEFAULT_URL, *, reconnect: bool = True, max_delay_s: float = 5.0
) -> AsyncIterator[dict[str, Any]]:
    """Every message from the engine at ``url``, as dicts, forever (or until the connection
    closes, with ``reconnect=False``).

    Raises:
        NeurOSConfigError: on a URL that isn't ws:// or wss://, or a non-positive delay cap.
    """
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    if not url.startswith(("ws://", "wss://")):
        raise NeurOSConfigError(f"the engine's URL must be ws:// or wss://, got {url!r}")
    if not max_delay_s > 0:
        raise NeurOSConfigError(f"max_delay_s must be > 0, got {max_delay_s}")
    delay = 0.25
    while True:
        try:
            async with connect(url) as connection:
                delay = 0.25
                async for message in connection:
                    event = parse(message)
                    if event is not None:
                        yield event
        except (OSError, ConnectionClosed) as error:
            if not reconnect:
                return
            log.info("engine unreachable (%s); retrying in %.2f s", error, delay)
        if not reconnect:
            return
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay_s)
