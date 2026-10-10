"""Serving the engine: the local WebSocket broadcast, the demo page, replay and its score."""

from __future__ import annotations

import asyncio
import json
import urllib.request

import pytest

pytest.importorskip("websockets")  # the engine extra (`uv sync --extra engine`)

from neuros.engine.events import HOLD, IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.runner import replay, score_replay
from neuros.engine.server import EventHub, check_host, start_server
from neuros.engine.simulator import Marker, Simulator
from neuros.errors import NeurOSConfigError

from .test_engine import _config, _recording, _task_accuracy, trained  # noqa: F401


def _intent(seq: int, t: float, command: str = "LEFT", reason: str | None = None) -> IntentEvent:
    return IntentEvent(
        seq=seq,
        t=t,
        command=command,
        confidence=0.8,
        probs={"LEFT": 0.8, "RIGHT": 0.2},
        hold_reason=reason,  # type: ignore[arg-type]
        state="adapting",
        seconds_of_data=1.0,
        quality_ok=True,
    )


def test_only_loopback_unless_explicitly_allowed() -> None:
    for host in ("127.0.0.1", "localhost", "::1"):
        check_host(host, allow_remote=False)
    for host in ("0.0.0.0", "192.168.1.5", "example.com"):
        with pytest.raises(NeurOSConfigError, match="local-only"):
            check_host(host, allow_remote=False)
    check_host("0.0.0.0", allow_remote=True)


def test_apps_get_hello_then_every_event_and_the_demo_page() -> None:
    from websockets.asyncio.client import connect

    async def scenario() -> tuple[list[dict], str, str]:
        hub = EventHub(["LEFT", "RIGHT"])
        server = await start_server(hub, port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}") as app:
                hello = json.loads(await app.recv())
                await hub.broadcast(
                    StatusEvent(seq=0, t=0.0, state="settling", previous="disconnected")
                )
                await hub.broadcast(_intent(1, 2.5))
                received = [hello] + [json.loads(await app.recv()) for _ in range(2)]
            page = await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            )
            client = await asyncio.to_thread(
                lambda: (
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/neuros-client.js")
                    .read()
                    .decode()
                )
            )
            await asyncio.sleep(0.05)
            await hub.broadcast(_intent(2, 2.625))  # the app has gone: nothing breaks
            return received, page, client
        finally:
            server.close()
            await server.wait_closed()

    received, page, client = asyncio.run(scenario())
    assert received[0] == {
        "type": "hello",
        "schema": "neuros.intent.v0",
        "commands": ["LEFT", "RIGHT"],
        "state": "disconnected",
    }
    assert received[1]["type"] == "status"
    assert received[1]["state"] == "settling"
    assert received[2]["type"] == "intent"
    assert received[2]["command"] == "LEFT"
    assert "NeurOS live" in page
    assert "export class NeurOSClient" in client


def test_unknown_paths_are_not_served() -> None:
    import urllib.error

    async def scenario() -> int:
        server = await start_server(EventHub(["LEFT"]), port=0)
        port = server.sockets[0].getsockname()[1]
        try:

            def fetch() -> int:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/../pyproject.toml")
                except urllib.error.HTTPError as error:
                    return error.code
                return 200

            return await asyncio.to_thread(fetch)
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) == 404


def test_replay_streams_to_an_app_and_scores_against_markers(trained: dict[str, object]) -> None:  # noqa: F811
    from websockets.asyncio.client import connect

    recording = _recording(5, 15, n_trials=12)
    simulators = [Simulator(recording)]
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]

    async def scenario() -> tuple[list, int]:
        hub = EventHub(["LEFT", "RIGHT"])
        server = await start_server(hub, port=0, static_dir=None)
        port = server.sockets[0].getsockname()[1]
        got = 0
        try:
            async with connect(f"ws://127.0.0.1:{port}") as app:
                await app.recv()  # hello
                task = asyncio.create_task(replay(simulators, engine, hub=hub))

                async def drain() -> None:
                    nonlocal got
                    async for _ in app:
                        got += 1

                reader = asyncio.create_task(drain())
                events = await task
                await asyncio.sleep(0.2)
                reader.cancel()
            return events, got
        finally:
            server.close()
            await server.wait_closed()

    events, got = asyncio.run(scenario())
    assert got == len(events)  # every event reached the app
    commands = {"left_hand": "LEFT", "right_hand": "RIGHT"}
    score = score_replay(
        events,
        list(simulators[0].markers),
        commands,
        window_s=2.0,
        task_start_s=0.0,
        task_end_s=4.0,
    )
    accuracy, n = _task_accuracy([e for e in events if isinstance(e, IntentEvent)], simulators[0])
    assert score.task_commands == n
    assert score.task_accuracy == pytest.approx(accuracy)
    assert score.n_decisions == sum(isinstance(e, IntentEvent) for e in events)
    assert score.states[0][1] == "settling"
    assert json.dumps(score.to_dict())
    with pytest.raises(NeurOSConfigError):
        asyncio.run(replay([], engine))


def test_score_counts_only_commands_inside_task_periods() -> None:
    markers = [Marker(10.0, "left_hand"), Marker(20.0, "right_hand")]
    events = [
        _intent(0, 13.0, "LEFT"),  # window [11, 13] inside [10 + 1, 10 + 4]: correct
        _intent(1, 14.0, "RIGHT"),  # inside, wrong
        _intent(2, 12.5, "LEFT"),  # window starts at 10.5 < 11: before the cue-free start
        _intent(3, 23.5, "RIGHT"),  # inside the second trial: correct
        _intent(4, 30.0, "LEFT"),  # between trials: not scored
        _intent(5, 13.5, HOLD, "low_confidence"),
    ]
    score = score_replay(
        events,
        markers,
        {"left_hand": "LEFT", "right_hand": "RIGHT"},
        window_s=2.0,
        task_start_s=1.0,
        task_end_s=4.0,
    )
    assert (score.task_commands, score.task_correct) == (3, 2)
    assert score.per_class == {"left_hand": (1, 2), "right_hand": (1, 1)}
    assert score.balanced_task_accuracy == pytest.approx(0.75)
    assert score.holds == {"low_confidence": 1}
    assert score.n_commands == 5
    with pytest.raises(NeurOSConfigError, match="shorter"):
        score_replay(events, markers, {}, window_s=2.0, task_start_s=1.0, task_end_s=2.0)
