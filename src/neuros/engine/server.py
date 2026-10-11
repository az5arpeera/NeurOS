"""The engine's output to apps: a WebSocket broadcast of ``neuros.intent.v0`` events (§7.1, 9).

Every connected app receives every event as one JSON text message. On connecting, an app
first gets a ``hello`` message (schema, the enabled commands, the engine's current state), so
a page opened mid-session knows where things stand.

**Local only (architecture §10).** The server binds to a loopback address by default and
refuses any other host unless ``allow_remote=True`` is passed explicitly: brain data and the
intents derived from it never leave the machine by accident.

The server never blocks the engine: a slow or vanished app is dropped, not waited for.

**Requests from apps.** An app may send ``{"type": "request", "request": NAME}`` (optionally
with ``"schema": "neuros.intent.v0"``), where NAME is one the hub advertised in ``hello``
(``requests``): with the guided calibration (D-040), ``calibrate``, ``cancel_calibration``
and ``reset_calibration``; when the session is recorded, ``mark`` with a ``"label"`` among
the commands (e.g. the door a game asked for, saved with the session as ground truth).
Requests are queued for the engine's loop (``take_requests``);
anything else (not JSON, too long, unknown, more than ``MAX_PENDING_REQUESTS`` waiting) is
logged and ignored, never answered with an error that could stall the app.

**The demo page.** Plain HTTP requests to the same port get the JavaScript client's demo page
(``clients/js``): open ``http://127.0.0.1:8765/`` in a browser. Browsers block JavaScript
modules on pages opened from disk, so the engine serves them; only the files listed in
``STATIC_FILES`` are served, from that one folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuros.engine.events import SCHEMA, Event, StatusEvent
from neuros.errors import NeurOSConfigError

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# A message an app hasn't taken within this many seconds drops that app.
SEND_TIMEOUT_S = 1.0
# The demo page and its client (repo folder clients/js), by URL path.
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("demo.html", "text/html; charset=utf-8"),
    "/demo.html": ("demo.html", "text/html; charset=utf-8"),
    "/neuros-client.js": ("neuros-client.js", "text/javascript; charset=utf-8"),
    "/stage": ("stage.html", "text/html; charset=utf-8"),
    "/stage.html": ("stage.html", "text/html; charset=utf-8"),
}
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[3] / "clients" / "js"
# Requests are tiny; anything longer isn't one.
MAX_REQUEST_CHARS = 1024
MAX_PENDING_REQUESTS = 16
NOT_FOUND = b"not found\n"


@dataclass(frozen=True)
class Request:
    """An app's request; ``label`` only for ``mark``."""

    name: str
    label: str | None = None


def check_host(host: str, *, allow_remote: bool) -> None:
    """Refuse a non-loopback bind address unless remote access is explicitly allowed.

    Raises:
        NeurOSConfigError: on a non-loopback host without ``allow_remote``.
    """
    if allow_remote:
        return
    if host == "localhost":
        return
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise NeurOSConfigError(
            f"refusing to serve intents on {host!r}: the engine is local-only (§10). Use "
            "127.0.0.1, or pass allow_remote explicitly"
        )


class EventHub:
    """Tracks connected apps and broadcasts events to them (one asyncio loop)."""

    def __init__(
        self,
        commands: Sequence[str],
        *,
        requests: Sequence[str] = (),
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            requests: the requests apps may send (advertised in ``hello``); none by default.
            info: labels for displays, sent in ``hello.info`` (``telemetry.stage_info``);
                must be JSON-serialisable.

        Raises:
            NeurOSConfigError: if ``info`` can't be sent as JSON.
        """
        self.commands = tuple(commands)
        self.requests = tuple(requests)
        self.info = dict(info or {})
        try:
            json.dumps(self.info, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise NeurOSConfigError(f"hello info must be plain JSON: {error}") from error
        self._pending: list[Request] = []
        self._clients: set[Any] = set()
        self._state: str = "disconnected"
        self.sent = 0
        self.dropped = 0

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def hello(self) -> str:
        """The first message an app gets. ``info`` (labels for displays) appears only when
        set, so apps that never asked for telemetry see the same hello as before."""
        body: dict[str, Any] = {
            "type": "hello",
            "schema": SCHEMA,
            "commands": list(self.commands),
            "state": self._state,
            "requests": list(self.requests),
        }
        if self.info:
            body["info"] = self.info
        return json.dumps(body, separators=(",", ":"))

    async def handler(self, connection: Any) -> None:
        """One app's connection: hello, then events until it goes away."""
        self._clients.add(connection)
        log.info("app connected (%d now)", len(self._clients))
        try:
            await connection.send(self.hello())
            try:
                async for message in connection:  # ends when the app closes
                    self.receive(message)
            except Exception as error:  # an abrupt close is an app going away, nothing more
                log.info("app connection ended: %s", error)
        finally:
            self._clients.discard(connection)
            log.info("app disconnected (%d left)", len(self._clients))

    def receive(self, message: str | bytes) -> bool:
        """Queue an app's request if it is valid and advertised; True if queued."""
        if isinstance(message, bytes) or len(message) > MAX_REQUEST_CHARS:
            log.info("ignoring a binary or oversized message from an app")
            return False
        try:
            request = json.loads(message)
        except ValueError:
            log.info("ignoring a message that isn't JSON")
            return False
        if (
            not isinstance(request, dict)
            or request.get("type") != "request"
            or request.get("schema", SCHEMA) != SCHEMA
        ):
            log.info("ignoring an app message that isn't a request: %r", request)
            return False
        name = request.get("request")
        if name not in self.requests:
            log.info("ignoring request %r (this engine accepts %s)", name, list(self.requests))
            return False
        label = None
        if name == "mark":
            label = request.get("label")
            if label not in self.commands:
                log.info("ignoring a mark whose label %r isn't a command", label)
                return False
        if len(self._pending) >= MAX_PENDING_REQUESTS:
            log.warning("ignoring request %r: %d already waiting", name, len(self._pending))
            return False
        self._pending.append(Request(str(name), label))
        return True

    def take_requests(self) -> list[Request]:
        """The queued requests, oldest first (the queue is emptied)."""
        taken, self._pending = self._pending, []
        return taken

    async def broadcast(self, event: Event) -> None:
        """Send ``event`` to every app; drop any that fail or are too slow."""
        if isinstance(event, StatusEvent):
            self._state = event.state
        if not self._clients:
            return
        message = event.to_json()
        for connection in list(self._clients):
            try:
                await asyncio.wait_for(connection.send(message), timeout=SEND_TIMEOUT_S)
                self.sent += 1
            except Exception as error:  # a vanished or stuck app never stalls the engine
                log.info("dropping an app: %s", error)
                self._clients.discard(connection)
                self.dropped += 1
                with contextlib.suppress(Exception):  # it may already be gone
                    await connection.close()


def _static_responder(static_dir: Path | None) -> Any:
    """A websockets ``process_request`` hook: WebSocket upgrades pass through; plain GETs of a
    ``STATIC_FILES`` path get that file; anything else gets 404."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def respond(status: int, reason: str, body: bytes, content_type: str) -> Response:
        headers = Headers(
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
                ("X-Content-Type-Options", "nosniff"),
            ]
        )
        return Response(status, reason, headers, body)

    def process_request(connection: Any, request: Any) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # an app connecting: continue the handshake
        entry = STATIC_FILES.get(request.path.split("?", 1)[0])
        if static_dir is None or entry is None:
            return respond(404, "Not Found", NOT_FOUND, "text/plain; charset=utf-8")
        name, content_type = entry
        file = static_dir / name
        if not file.is_file():
            return respond(404, "Not Found", NOT_FOUND, "text/plain; charset=utf-8")
        return respond(200, "OK", file.read_bytes(), content_type)

    return process_request


async def start_server(
    hub: EventHub,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allow_remote: bool = False,
    static_dir: Path | None = DEFAULT_STATIC_DIR,
) -> Any:
    """Start serving ``hub`` (returns the websockets server; close it with ``.close()``).

    Args:
        static_dir: where the demo page lives (None: serve no page, WebSocket only).

    Raises:
        NeurOSConfigError: on a non-loopback host without ``allow_remote``, or a bad port.
    """
    from websockets.asyncio.server import serve

    check_host(host, allow_remote=allow_remote)
    if not 0 <= port <= 65535:
        raise NeurOSConfigError(f"port must be in [0, 65535], got {port}")
    folder = static_dir if static_dir is not None and static_dir.is_dir() else None
    return await serve(hub.handler, host, port, process_request=_static_responder(folder))
