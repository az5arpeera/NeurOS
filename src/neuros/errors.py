"""Typed errors for every boundary NeurOS checks.

Architecture §3, principle 3 ("typed, validated boundaries"): every public function validates its
inputs and, on failure, raises a *typed* error with an *actionable* message. Callers can catch a
whole family (``NeurOSDataError``) or one precise failure (``NeurOSChannelError``).

Principle 4 ("fail safe, never silent") applies to the online engine: there, these errors are
caught at the loop boundary and turned into ``HOLD`` events with a reason code. Offline code lets
them propagate so a bad dataset can never slip quietly into training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataContext:
    """Where a piece of data came from, attached to every data error.

    All fields are optional because errors can arise before the full context is known
    (e.g. while resolving which dataset a file belongs to).
    """

    dataset: str | None = None
    subject: int | str | None = None
    session: str | None = None
    run: str | None = None

    def describe(self) -> str:
        """Render as ``dataset=X subject=Y ...``, skipping unknown fields."""
        parts = [
            f"{name}={value}"
            for name, value in (
                ("dataset", self.dataset),
                ("subject", self.subject),
                ("session", self.session),
                ("run", self.run),
            )
            if value is not None
        ]
        return " ".join(parts)


class NeurOSError(Exception):
    """Base class for every error NeurOS raises on purpose."""


class NeurOSConfigError(NeurOSError):
    """A configuration file or value is missing, malformed or inconsistent."""


class NeurOSDataError(NeurOSError):
    """Data violates the canonical data contract (architecture §4).

    The message always carries the data context, so a failure deep inside a 380-person run
    points straight at the offending dataset, subject, session and run.
    """

    def __init__(self, message: str, context: DataContext | None = None) -> None:
        self.context = context or DataContext()
        where = self.context.describe()
        super().__init__(f"{message} [{where}]" if where else message)
        self.reason = message


class NeurOSChannelError(NeurOSDataError):
    """Unknown, duplicate or missing channel names."""


class NeurOSLabelError(NeurOSDataError):
    """An event code or label that the canonical vocabulary doesn't map."""


class NeurOSSignalError(NeurOSDataError):
    """A chunk of samples or timestamps that the streaming signal chain can't accept.

    Wrong shape or channel count, NaN/Inf samples, or timestamps that don't strictly increase.
    Offline this aborts the recording; online the engine turns it into ``HOLD("signal")``.
    """


class NeurOSSignalGapError(NeurOSSignalError):
    """The stream skipped more time than the signal chain may bridge (architecture §7.1).

    Interpolating across a long gap would invent data, and filter state from before the gap no
    longer describes the signal. The engine catches this, resets the chain and emits
    ``HOLD("signal_gap")``.
    """

    def __init__(self, message: str, gap_s: float, context: DataContext | None = None) -> None:
        super().__init__(message, context)
        self.gap_s = gap_s


class NeurOSSplitError(NeurOSError):
    """A train/test split integrity violation, e.g. a locked test person reaching training.

    Architecture §3, principle 2 (person-level isolation): this error must never be caught and
    ignored. It means a reported number would be contaminated.
    """
