"""Telemetry for displays (opt-in): what the engine saw, alongside its decisions.

The demo's stage view (``clients/js/stage.html``) draws each channel's cleaned signal, its
quality and the alignment warm-up. ``Telemetry`` watches the engine (``Engine.add_observer``),
so the decision path is untouched: with each processed window it queues one
``TelemetryEvent`` holding the samples that arrived since the previous decision, decimated
for drawing, and the window's per-channel RMS. The runner sends the queued events with the
engine's own, in sequence order.

Decimating by 2 (128 → 64 Hz) keeps the 8-30 Hz band below the display's Nyquist (32 Hz) and
halves the bytes: 4 channels × 8 samples × 8 decisions per second is about 2 KB/s of JSON.
"""

from __future__ import annotations

import math

import numpy as np

from neuros.engine.events import TelemetryEvent
from neuros.engine.pipeline import Engine, EngineWindow
from neuros.errors import NeurOSConfigError


class Telemetry:
    """Queues one ``TelemetryEvent`` per processed window (see the module docs)."""

    def __init__(self, engine: Engine, *, decimate: int = 2) -> None:
        """
        Raises:
            NeurOSConfigError: if ``decimate`` isn't a positive integer dividing the hop.
        """
        hop = round(engine.hop_s * engine.config.contract.sfreq)
        if not (isinstance(decimate, int) and decimate >= 1 and hop % decimate == 0):
            raise NeurOSConfigError(f"decimate must be a positive divisor of the hop ({hop})")
        self.engine = engine
        self.decimate = decimate
        self.hop = hop
        self.sfreq = engine.config.contract.sfreq / decimate
        self.channels = tuple(engine.profile)
        self._pending: list[TelemetryEvent] = []
        engine.add_observer(self._see)

    def _see(self, window: EngineWindow) -> None:
        data = np.asarray(window.referenced, dtype=np.float64)
        newest = data[:, -self.hop :: self.decimate]
        rms = np.sqrt(np.mean(data**2, axis=1))
        if not (np.isfinite(newest).all() and np.isfinite(rms).all()):
            return  # the quality gate holds on such a window; there's nothing to draw
        self._pending.append(
            TelemetryEvent(
                seq=self.engine.next_seq(),
                t=window.t,
                channels=self.channels,
                sfreq=self.sfreq,
                samples=tuple(tuple(float(v) for v in row) for row in newest),
                rms_uv=tuple(float(v) for v in rms),
                bad_channels=tuple(c for c in window.quality.bad_channels if c in self.channels),
                in_transient=window.in_transient,
                state=self.engine.state,
                seconds_of_data=max(0.0, float(self.engine.seconds_of_data)),
            )
        )

    def drain(self) -> list[TelemetryEvent]:
        """The queued events, oldest first (the queue is emptied)."""
        taken, self._pending = self._pending, []
        return taken


def stage_info(
    engine: Engine, *, source: str, model: str, train_people: int | None
) -> dict[str, object]:
    """What a display needs to label itself, sent once in ``hello.info``."""
    policy = engine.config.policy
    contract = engine.config.contract
    return {
        "source": source,
        "channels": list(engine.profile),
        "model": model,
        "train_people": train_people,
        "threshold": policy.threshold,
        "band_hz": list(contract.band_hz),
        "decisions_per_s": round(1.0 / engine.hop_s, 3) if math.isfinite(engine.hop_s) else None,
        "settling_s": engine.config.settling_s,
        "steady_s": engine.config.steady_s,
        "telemetry": True,
    }
