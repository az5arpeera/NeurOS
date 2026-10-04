"""The shared preprocessing chain: notch + band-pass → 128 Hz grid (architecture §3, §4.2).

``StreamingPreprocessor`` is the *only* implementation. The live engine feeds it chunks with
LSL timestamps; ``preprocess_recording`` feeds it a whole recording at once with evenly spaced
timestamps. Because every stage carries its state across chunks exactly, both paths produce the
same samples bit for bit (the golden parity test in ``tests/test_preprocess.py``).

Not done here, on purpose:

- **Re-referencing (CAR)** happens per window after the headset's channel subset is chosen
  (``reference``). It commutes with this chain, so doing it later loses nothing.
- **Alignment (EA)** needs a reference covariance (``neuros.align``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from neuros.config.schemas import SignalContract
from neuros.data.recording import EventTable, Recording
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal.filters import CausalFilter, FilterDesign, design_filter
from neuros.signal.resample import DEFAULT_MAX_GAP_S, TimestampResampler, default_timestamps

RESAMPLER_NAME = "linear-timestamp"


class StreamingPreprocessor:
    """Causal notch + band-pass at the native rate, then timestamp resampling to the contract.

    Not thread-safe: one instance per stream. Validation happens before any state changes, so
    a rejected chunk can be skipped, or the stream reset.
    """

    def __init__(
        self,
        n_channels: int,
        native_sfreq: float,
        contract: SignalContract | None = None,
        *,
        max_gap_s: float = DEFAULT_MAX_GAP_S,
        context: DataContext | None = None,
    ) -> None:
        """
        Raises:
            NeurOSConfigError: if the native rate is below the contract rate (no upsampling:
                every source and headset in scope records at ≥ 160 Hz) or the filter can't be
                designed for it.
        """
        self.contract = contract or SignalContract()
        if not (math.isfinite(native_sfreq) and native_sfreq >= self.contract.sfreq):
            raise NeurOSConfigError(
                f"native rate {native_sfreq!r} Hz is below the contract's "
                f"{self.contract.sfreq:g} Hz; upsampling isn't supported"
            )
        self.context = context or DataContext()
        self.native_sfreq = float(native_sfreq)
        self.design: FilterDesign = design_filter(self.native_sfreq, self.contract)
        self.filter = CausalFilter(self.design, n_channels, self.context)
        self.resampler = TimestampResampler(
            n_channels,
            self.contract.sfreq,
            max_gap_s=max_gap_s,
            nominal_sfreq=self.native_sfreq,
            context=self.context,
        )
        self.n_channels = n_channels

    def reset(self) -> None:
        """Start over (after a gap or reconnect): filter state and output grid."""
        self.filter.reset()
        self.resampler.reset()

    def process(
        self, chunk: NDArray[np.floating], timestamps: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Process one native-rate chunk. Returns ``(samples [channels, m], grid times [m])``.

        Raises:
            NeurOSSignalError, NeurOSSignalGapError: on invalid samples or timestamps (raised
                before any state changes).
        """
        # Timestamps first: a gap must not leave the filter one chunk ahead of the resampler.
        n_samples = int(chunk.shape[1]) if isinstance(chunk, np.ndarray) and chunk.ndim == 2 else -1
        self.resampler.check_timestamps(timestamps, n_samples=n_samples)
        filtered = self.filter.process(chunk)
        return self.resampler.process(filtered, timestamps)

    def describe(self) -> dict[str, str]:
        """Provenance strings for everything this chain did."""
        return {
            **self.design.describe(),
            "resampler": RESAMPLER_NAME,
            "native_sfreq": f"{self.native_sfreq:g}",
            "sfreq": f"{self.contract.sfreq:g}",
        }


def map_onsets(
    onsets: NDArray[np.int64], native_sfreq: float, out_sfreq: float, n_out: int
) -> NDArray[np.int64]:
    """Move event onsets from native samples to the output grid (nearest grid sample).

    Grid sample ``k`` sits at ``k / out_sfreq`` s and native sample ``i`` at ``i / native``,
    so ``k = round(i · out / native)`` (error ≤ half an output sample, 3.9 ms). An event within
    half a sample of the very end can round to ``n_out``; it's clamped to the last sample (its
    epoch can never be complete, so epoch extraction counts it as skipped).
    """
    if n_out < 1:
        raise NeurOSDataError("can't map events onto an empty recording")
    mapped = np.rint(onsets.astype(np.float64) * (out_sfreq / native_sfreq)).astype(np.int64)
    return np.minimum(mapped, n_out - 1)


def preprocess_recording(recording: Recording, contract: SignalContract | None = None) -> Recording:
    """Offline path: run the streaming chain over a whole recording in one chunk.

    Returns a new ``Recording`` at the contract rate (float32 µV), with events moved to the
    output grid and the chain's settings added to the provenance.

    Raises:
        NeurOSConfigError: if the recording's rate can't be processed (see the class).
        NeurOSDataError: if the output would be empty.
    """
    contract = contract or SignalContract()
    chain = StreamingPreprocessor(
        recording.n_channels, recording.sfreq, contract, context=recording.context
    )
    timestamps = default_timestamps(recording.n_samples, recording.sfreq)
    samples, _ = chain.process(recording.data, timestamps)
    n_out = samples.shape[1]
    if n_out == 0:
        raise NeurOSDataError("preprocessing produced no samples", recording.context)
    events = EventTable(
        onsets=map_onsets(recording.events.onsets, recording.sfreq, contract.sfreq, n_out),
        labels=recording.events.labels,
    )
    return Recording(
        data=samples.astype(np.float32),
        sfreq=contract.sfreq,
        ch_names=recording.ch_names,
        events=events,
        dataset=recording.dataset,
        subject=recording.subject,
        session=recording.session,
        run=recording.run,
        provenance={**recording.provenance, **chain.describe()},
    )
