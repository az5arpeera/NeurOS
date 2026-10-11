"""The engine loop (architecture §7.1-7.2): timestamped headset chunks in, events out.

Per chunk, ``Engine.push`` runs the shared preprocessing (``StreamingPreprocessor``, the exact
code that prepared the training data: principle 1, train-serve parity) and cuts a window every
hop (``WindowStream``). Per window:

1. **Quality gate** (``QualityGate``): a failing window is never adapted on or decided on; the
   event is ``HOLD`` with the gate's reason.
2. **Common average** over the headset's channels that the model knows (the *profile*), as
   training did (``neuros.eval.pooled.to_layout``).
3. **Online alignment** (``OnlineReference``): good windows update R̂; every window is aligned
   with the current R̂^(-1/2).
4. **Placement** (``place_window``): the profile's rows go to their layout rows, zeros
   elsewhere, divided by the model's input scale. Exactly ``to_layout`` then the trainer's
   scale, so a window placed live equals the same window placed offline (tested).
5. **Decoder** (``Decoder``): class probabilities.
6. **Policy** (``Policy``): smoothing, threshold, debounce, decoys; a command or ``HOLD``.

Windows that start within the filter start-up (``TRANSIENT_S`` after a (re)start) are held
with ``warmup`` and never adapted on: the causal filters haven't settled.

**States** (§7.2): ``disconnected`` until the first chunk; ``settling`` for ``settling_s``;
then ``adapting`` until the reference holds ``steady_s`` of good data, then ``steady``.
``degraded`` when the gate has failed for more than ``degraded_after_s``; the first good window
returns to the normal state. Every change emits a ``StatusEvent``.

**Gaps and bad samples.** A timestamp gap (``NeurOSSignalGapError``) or a chunk with invalid
samples restarts the preprocessing and the windowing (filter state would be wrong), emits
``HOLD("signal_gap")``, and keeps the alignment reference: it is still the same person and
headset. The engine never guesses across a gap.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from neuros.align.covariance import window_covariances
from neuros.align.ea import AlignmentReference, apply_alignment
from neuros.config.schemas import SignalContract
from neuros.engine.alignment import OnlineReference
from neuros.engine.events import (
    HOLD,
    CalibrationLevel,
    EngineState,
    Event,
    HoldReason,
    IntentEvent,
    StatusEvent,
)
from neuros.engine.policy import Policy, PolicySettings
from neuros.engine.quality import QualityGate, QualityResult, QualityThresholds
from neuros.errors import (
    DataContext,
    NeurOSConfigError,
    NeurOSDataError,
    NeurOSSignalError,
)
from neuros.signal.filters import TRANSIENT_S
from neuros.signal.preprocess import StreamingPreprocessor
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowStream

log = logging.getLogger(__name__)

# The fewest of the model's channels a headset must provide: the common average and the
# covariances need two (as ``neuros.eval.pooled.MIN_LAYOUT_CHANNELS``).
MIN_PROFILE_CHANNELS = 2


class Decoder(Protocol):
    """What the engine needs from a model: its layout, its classes, probabilities."""

    @property
    def layout(self) -> tuple[str, ...]:
        """The channel of each input row (the training layout)."""
        ...

    @property
    def classes(self) -> tuple[str, ...]:
        """The class of each output column."""
        ...

    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.floating]:
        """``[n, len(layout), samples]`` placed windows -> ``[n, len(classes)]`` probabilities."""
        ...


@dataclass(frozen=True)
class EngineConfig:
    """Everything the engine needs besides the decoder.

    Attributes:
        stream_channels: the headset stream's channel names, in stream order.
        native_sfreq: the stream's nominal sampling rate (Hz).
        prior: R_pop, ``[p, p]`` over the *profile* (the stream channels that are in the
            model's layout, in layout order), common-averaged like training windows.
        input_scale: the model's input scale (the RMS of its training windows).
        contract: the signal contract the model was trained with.
        prior_s: n₀ in seconds of windows (E2: 10 s).
        ema_s: None (running mean, E2's) or the reference EMA's memory in seconds.
        settling_s, steady_s: state boundaries (§7.2): seconds since start; seconds of good
            data in the reference.
        degraded_after_s: seconds of continuously failing windows before ``degraded``.
    """

    stream_channels: tuple[str, ...]
    native_sfreq: float
    prior: NDArray[np.float64]
    input_scale: float
    contract: SignalContract = field(default_factory=SignalContract)
    policy: PolicySettings = field(default_factory=PolicySettings)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    prior_s: float = 10.0
    ema_s: float | None = None
    settling_s: float = 30.0
    steady_s: float = 120.0
    degraded_after_s: float = 2.0

    def __post_init__(self) -> None:
        if len(set(self.stream_channels)) != len(self.stream_channels):
            raise NeurOSConfigError(f"stream channels repeat: {list(self.stream_channels)}")
        for name in ("native_sfreq", "input_scale", "prior_s", "settling_s", "steady_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise NeurOSConfigError(f"{name} must be a positive finite number, got {value!r}")
        if not (math.isfinite(self.degraded_after_s) and self.degraded_after_s >= 0):
            raise NeurOSConfigError(f"degraded_after_s must be >= 0, got {self.degraded_after_s}")
        if self.ema_s is not None and not (math.isfinite(self.ema_s) and self.ema_s > 0):
            raise NeurOSConfigError(f"ema_s must be None or > 0, got {self.ema_s}")


@dataclass(frozen=True)
class EngineWindow:
    """One window as the engine saw it (for replay evaluation and parity tests)."""

    t: float  # stream clock at the window's last sample
    start: int  # first sample, counted on the 128-Hz grid since the last (re)start
    referenced: NDArray[np.float32]  # [profile, samples], common-averaged, µV
    placed: NDArray[np.float32]  # [layout, samples], aligned and scaled: the model's input
    quality: QualityResult
    in_transient: bool


def place_window(
    referenced: NDArray[np.floating],
    reference: AlignmentReference,
    layout_rows: Sequence[int],
    n_layout: int,
    input_scale: float,
) -> NDArray[np.float32]:
    """Align a common-averaged profile window and place it in the model's layout.

    The live twin of ``to_layout`` (EA on the present channels, zeros elsewhere) followed by
    the trainer's division by the input scale.

    Raises:
        NeurOSDataError: on shapes that don't match the reference or the layout rows.
    """
    x = np.asarray(referenced)
    if x.ndim != 2 or x.shape[0] != reference.n_channels or x.shape[0] != len(layout_rows):
        raise NeurOSDataError(
            f"window is {x.shape}; the reference has {reference.n_channels} channels and "
            f"{len(layout_rows)} layout rows were given"
        )
    aligned = apply_alignment(x[None].astype(np.float32, copy=False), reference)[0]
    placed = np.zeros((n_layout, x.shape[1]), dtype=np.float32)
    placed[list(layout_rows), :] = aligned
    placed /= np.float32(input_scale)
    return placed


class Engine:
    """One headset stream's engine (not thread-safe: one per stream)."""

    def __init__(
        self,
        decoder: Decoder,
        config: EngineConfig,
        *,
        observer: Callable[[EngineWindow], None] | None = None,
        context: DataContext | None = None,
    ) -> None:
        """
        Args:
            observer: called with every window the engine processes (replay evaluation,
                parity tests); more can be added with ``add_observer``.

        Raises:
            NeurOSConfigError: if the headset shares fewer than 2 channels with the model, or
                the prior doesn't match the profile, or the policy names unknown classes.
        """
        self.decoder = decoder
        self.config = config
        self._observers: list[Callable[[EngineWindow], None]] = []
        if observer is not None:
            self._observers.append(observer)
        self.calibration_level: CalibrationLevel | None = None
        self.context = context or DataContext()
        layout = tuple(decoder.layout)
        profile = [ch for ch in layout if ch in config.stream_channels]
        if len(profile) < MIN_PROFILE_CHANNELS:
            raise NeurOSConfigError(
                f"the headset shares {profile} with the model's layout; at least "
                f"{MIN_PROFILE_CHANNELS} channels are needed"
            )
        self.profile = tuple(profile)
        self.layout = layout
        self._stream_rows = [config.stream_channels.index(ch) for ch in profile]
        self._layout_rows = [layout.index(ch) for ch in profile]
        contract = config.contract
        self.hop_s = contract.hop_s
        prior = np.asarray(config.prior, dtype=np.float64)
        if prior.shape != (len(profile), len(profile)):
            raise NeurOSConfigError(
                f"the prior is {prior.shape}; this headset's profile has {len(profile)} "
                f"channels {list(profile)}"
            )
        self.preprocessor = StreamingPreprocessor(
            len(profile), config.native_sfreq, contract, context=self.context
        )
        self.windows = WindowStream(
            len(profile), contract.window_samples, contract.hop_samples, self.context
        )
        self.gate = QualityGate(profile, config.quality)
        per_second = max(1, round(1.0 / self.hop_s))
        self.reference = OnlineReference(
            prior,
            prior_windows=config.prior_s / self.hop_s,
            update_every=per_second,
            ema_windows=None if config.ema_s is None else max(2, round(config.ema_s / self.hop_s)),
            context=self.context,
        )
        self.policy = Policy(decoder.classes, config.policy, self.hop_s)
        self._transient = math.ceil(TRANSIENT_S * contract.sfreq)
        self.state: EngineState = "disconnected"
        self.seq = 0
        self._started_t: float | None = None
        self._bad_since: float | None = None
        self._times = np.zeros(0, dtype=np.float64)  # grid times of WindowStream's buffer
        self._last_t: float | None = None

    # ------------------------------------------------------------------ public
    @property
    def seconds_of_data(self) -> float:
        """Good data in the alignment reference, in seconds (one window per hop)."""
        return self.reference.n_windows * self.hop_s

    @property
    def last_t(self) -> float | None:
        """The stream clock at the newest processed sample (None before any data)."""
        return self._last_t

    def add_observer(self, observer: Callable[[EngineWindow], None]) -> None:
        """Also call ``observer`` with every window processed (the guided calibration's
        recorder, replay evaluation, parity tests)."""
        self._observers.append(observer)

    def next_seq(self) -> int:
        """The next event number (shared by every event this engine emits, calibration
        events included, so an app can spot lost messages)."""
        seq = self.seq
        self.seq += 1
        return seq

    def set_decoder(self, decoder: Decoder, *, level: CalibrationLevel | None) -> None:
        """Switch the model, e.g. to one fine-tuned by the guided calibration (``level``
        "quick") or back to the bundle's (None). The policy's smoothing restarts, so no
        probability of the old model leaks into the new one's decisions; the alignment
        reference is kept (same person, same headset).

        Raises:
            NeurOSConfigError: if the new decoder's layout or classes differ (the engine's
                placement and policy were built for them).
        """
        if tuple(decoder.layout) != self.layout:
            raise NeurOSConfigError(
                f"the new decoder's layout {list(decoder.layout)} isn't {list(self.layout)}"
            )
        if tuple(decoder.classes) != tuple(self.decoder.classes):
            raise NeurOSConfigError(
                f"the new decoder's classes {list(decoder.classes)} aren't "
                f"{list(self.decoder.classes)}"
            )
        self.decoder = decoder
        self.calibration_level = level
        self.policy.reset()

    def push(self, chunk: NDArray[np.floating], timestamps: NDArray[np.floating]) -> list[Event]:
        """Feed one chunk ``[stream channels, n]`` with its stream-clock timestamps ``[n]``.

        Returns the events it produced, in order (often none: a decision needs a full hop).

        Raises:
            NeurOSConfigError: if the chunk's channel count doesn't match the stream's (the
                stream isn't the one the engine was configured for).
        """
        x = np.asarray(chunk)
        stamps = np.asarray(timestamps, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(self.config.stream_channels):
            raise NeurOSConfigError(
                f"chunk must be [{len(self.config.stream_channels)} channels, n], got {x.shape}"
            )
        events: list[Event] = []
        if x.shape[1] == 0:
            return events
        if self.state == "disconnected":
            first_t = float(stamps[0]) if stamps.size and math.isfinite(stamps[0]) else 0.0
            self._started_t = first_t
            self._set_state("settling", first_t, "stream connected", events)
        try:
            samples, times = self.preprocessor.process(x[self._stream_rows], stamps)
        except NeurOSSignalError as error:  # gaps included (a subclass)
            self._restart(error, events)
            try:  # the chunk itself may be fine after the restart (it starts a new segment)
                samples, times = self.preprocessor.process(x[self._stream_rows], stamps)
            except NeurOSSignalError as again:
                log.warning("chunk rejected after a restart: %s", again)
                return events
        if times.size:
            self._last_t = float(times[-1])
        before = self.windows.samples_seen
        buffered = self._times
        new_windows, starts = self.windows.push(samples)
        combined_times = np.concatenate([buffered, times])
        first_global = before - buffered.size
        width = self.windows.window
        for window, start in zip(new_windows, starts.tolist(), strict=True):
            t_end = float(combined_times[start + width - 1 - first_global])
            events.extend(self._process(window, start, t_end))
        keep = min(width - 1, combined_times.size)
        self._times = combined_times[combined_times.size - keep :].copy()
        return events

    def close(self) -> list[Event]:
        """The stream ended: state ``disconnected`` (the reference is kept for a reconnect)."""
        events: list[Event] = []
        t = self._last_t if self._last_t is not None else 0.0
        self._set_state("disconnected", t, "stream closed", events)
        self._reset_stream()
        return events

    # ------------------------------------------------------------------ internals
    def _reset_stream(self) -> None:
        self.preprocessor.reset()
        self.windows.reset()
        self.policy.reset()
        self._times = np.zeros(0, dtype=np.float64)

    def _restart(self, error: NeurOSSignalError, events: list[Event]) -> None:
        """A gap or invalid samples: restart filters and windows, tell the app."""
        log.warning("restarting the stream: %s", error)
        self._reset_stream()
        t = self._last_t if self._last_t is not None else (self._started_t or 0.0)
        events.append(self._hold_event(t, "signal_gap", quality=None, latency_ms=0.0))

    def _set_state(self, state: EngineState, t: float, reason: str, events: list[Event]) -> None:
        if state == self.state:
            return
        events.append(
            StatusEvent(
                seq=self.next_seq(),
                t=t,
                state=state,
                previous=self.state,
                reason=reason,
                details={"seconds_of_data": round(self.seconds_of_data, 3)},
            )
        )
        self.state = state

    def _normal_state(self, t: float) -> EngineState:
        started = self._started_t if self._started_t is not None else t
        if t - started < self.config.settling_s:
            return "settling"
        if self.seconds_of_data >= self.config.steady_s:
            return "steady"
        return "adapting"

    def _hold_event(
        self,
        t: float,
        reason: HoldReason,
        *,
        quality: QualityResult | None,
        latency_ms: float,
    ) -> IntentEvent:
        confidence, probs = self.policy.snapshot()
        return IntentEvent(
            seq=self.next_seq(),
            t=t,
            command=HOLD,
            confidence=confidence,
            probs=probs,
            hold_reason=reason,
            state=self.state,
            seconds_of_data=self.seconds_of_data,
            quality_ok=quality.ok if quality is not None else False,
            bad_channels=quality.bad_channels if quality is not None else (),
            latency_ms=latency_ms,
            calibration_level=self.calibration_level,
        )

    def _process(self, window: NDArray[np.float32], start: int, t: float) -> list[Event]:
        clock = time.perf_counter()
        events: list[Event] = []
        in_transient = start < self._transient
        quality = self.gate.check(window)
        if self.config.contract.reference == "car":
            referenced = np.asarray(common_average_reference(window), dtype=np.float32)
        else:
            referenced = window
        good = quality.ok and not in_transient
        if good:
            self.reference.add(window_covariances(referenced))
            self._bad_since = None
        elif not quality.ok and self._bad_since is None:
            self._bad_since = t
        placed = place_window(
            referenced,
            self.reference.current,
            self._layout_rows,
            len(self.layout),
            self.config.input_scale,
        )
        if self._observers:
            seen = EngineWindow(t, start, referenced, placed, quality, in_transient)
            for observer in self._observers:
                observer(seen)

        # State first, so the event carries the state the decision was made in.
        bad_for = 0.0 if self._bad_since is None else t - self._bad_since
        if not quality.ok and bad_for > self.config.degraded_after_s:
            self._set_state("degraded", t, f"signal quality: {quality.reason}", events)
        elif quality.ok:
            target = self._normal_state(t)
            if self.state == "degraded":
                reason = "signal recovered"
            elif target == "adapting":
                reason = f"settling period ({self.config.settling_s:g} s) over"
            else:
                reason = f"{self.config.steady_s:g} s of good data in the alignment reference"
            self._set_state(target, t, reason, events)

        if in_transient:
            self.policy.interrupt()
            event = self._hold_event(t, "warmup", quality=quality, latency_ms=0.0)
        elif quality.reason is not None:  # failed the gate (QualityResult: reason iff not ok)
            self.policy.interrupt()
            event = self._hold_event(t, quality.reason, quality=quality, latency_ms=0.0)
        else:
            probs = np.asarray(self.decoder.predict_proba(placed[None]), dtype=np.float64)
            if probs.shape != (1, len(self.decoder.classes)):
                raise NeurOSDataError(
                    f"decoder returned {probs.shape}, expected (1, {len(self.decoder.classes)})"
                )
            decision = self.policy.decide(probs[0])
            event = IntentEvent(
                seq=self.next_seq(),
                t=t,
                command=decision.command,
                confidence=decision.confidence,
                probs=decision.probs,
                hold_reason=decision.hold_reason,
                state=self.state,
                seconds_of_data=self.seconds_of_data,
                quality_ok=True,
                bad_channels=(),
                latency_ms=0.0,
                calibration_level=self.calibration_level,
            )
        latency_ms = (time.perf_counter() - clock) * 1000.0
        events.append(_with_latency(event, latency_ms))
        return events


def _with_latency(event: IntentEvent, latency_ms: float) -> IntentEvent:
    """The event with its measured compute time (events are frozen: rebuilt, not mutated)."""
    return replace(event, latency_ms=max(0.0, latency_ms))
