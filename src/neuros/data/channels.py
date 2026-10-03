"""Channel names: normalisation to canonical 10-10/10-05 names, and spatial ordering.

Architecture §4.3: every dataset and headset spells channels differently. PhysioNet pads names
with dots (``C3..``, ``Fc5.``), clinical systems use old 10-20 names (``T3`` instead of ``T7``)
and EDF files add type prefixes or reference suffixes (``EEG Fp1``, ``C3-REF``). A model input
channel is identified *only* by its canonical name, so all of these must collapse to one
spelling before any data is combined. Getting this wrong silently feeds the wrong electrode
into the wrong input row, which is why normalisation is strict: an unrecognised name returns
``None`` and the caller must decide (declared non-EEG channel → drop; anything else → error).

The canonical spelling comes from MNE's 10-05 template montage (e.g. ``Fp1``, ``AFz``, ``FCz``,
``CPz``, ``POz``, ``FTT9h``), with the old 10-20 temporal names renamed to their 10-10
equivalents.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Iterable, Sequence
from typing import Final

from neuros.errors import DataContext, NeurOSChannelError, NeurOSConfigError

# Old 10-20 names → 10-10 names (architecture §4.3). EEGMAT, for example, still uses T3..T6.
OLD_TO_NEW: Final[dict[str, str]] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Ear/mastoid sites are valid electrode positions but are references, never model inputs.
REFERENCE_SITES: Final[frozenset[str]] = frozenset({"A1", "A2", "M1", "M2"})

# The sensorimotor region used for the canonical montage C-core: the frontal (F), fronto-central
# (FC), central (C), centro-parietal (CP) and parietal (P) rows. Prefrontal, temporal and
# occipital rows are excluded: they carry mostly eye, muscle and visual activity for motor
# imagery, and few consumer headsets cover them consistently.
C_CORE_ROWS: Final[tuple[str, ...]] = ("F", "FC", "C", "CP", "P")

# EDF and vendor conventions to strip before lookup.
_TYPE_PREFIXES: Final[tuple[str, ...]] = ("eeg ", "eeg-", "eeg_", "eeg:")
_REFERENCE_SUFFIX_RE: Final = re.compile(
    r"[-_ ](REF|LE|RE|AR|AVG|CAR|A1|A2|M1|M2|A1A2|M1M2|LINKED)$", re.IGNORECASE
)

# Template montages that list 10-05 names with canonical capitalisation. MNE 1.13 deprecates
# 'standard_1005' in favour of 'colin27_1005'; try the new name first, fall back for older MNE.
_MNE_TEMPLATE_MONTAGES: Final[tuple[str, ...]] = ("colin27_1005", "standard_1005")

# Position grammar: row letters, then 'z' (midline) or an electrode number, then an optional
# 'h' for 10-05 half positions. Examples: Fpz, FC3, CPz, FTT9h, T10.
_POSITION_RE: Final = re.compile(r"^(?P<row>[A-Za-z]+?)(?P<pos>[zZ]|\d+)(?P<half>h?)$")

# Anterior → posterior level of each row. Temporal rows sit at the same level as the central
# row they flank (FT↔FC, T↔C, TP↔CP), so T7 sorts next to C5 rather than in its own block.
_ROW_LEVEL: Final[dict[str, float]] = {
    "N": 0.0,
    "Fp": 1.0,
    "AFp": 1.5,
    "AF": 2.0,
    "AFF": 2.5,
    "F": 3.0,
    "FFC": 3.5,
    "FFT": 3.5,
    "FC": 4.0,
    "FT": 4.0,
    "FCC": 4.5,
    "FTT": 4.5,
    "C": 5.0,
    "T": 5.0,
    "A": 5.0,
    "CCP": 5.5,
    "TTP": 5.5,
    "M": 5.5,
    "CP": 6.0,
    "TP": 6.0,
    "CPP": 6.5,
    "TPP": 6.5,
    "P": 7.0,
    "PPO": 7.5,
    "PO": 8.0,
    "POO": 8.5,
    "O": 9.0,
    "OI": 9.5,
    "I": 10.0,
}


@functools.cache
def canonical_names() -> frozenset[str]:
    """Return every canonical 10-05 electrode name (old 10-20 names already renamed).

    Raises:
        NeurOSConfigError: if MNE ships none of the expected template montages.
    """
    import mne  # imported lazily: MNE is heavy and this is the only reason to load it here

    for montage_name in _MNE_TEMPLATE_MONTAGES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                montage = mne.channels.make_standard_montage(montage_name)
        except ValueError:
            continue
        return frozenset(OLD_TO_NEW.get(name, name) for name in montage.ch_names)
    raise NeurOSConfigError(
        f"None of the MNE template montages {_MNE_TEMPLATE_MONTAGES} is available; "
        "upgrade MNE (pyproject requires mne>=1.7)."
    )


@functools.cache
def _lookup_table() -> dict[str, str]:
    """Map lower-cased spellings (including old 10-20 names) to canonical names."""
    table: dict[str, str] = {}
    for name in canonical_names():
        key = name.lower()
        if key in table and table[key] != name:
            # Two canonical names differing only by case would make lookup ambiguous.
            raise NeurOSConfigError(
                f"Ambiguous channel names in template: {name!r} vs {table[key]!r}"
            )
        table[key] = name
    for old, new in OLD_TO_NEW.items():
        table[old.lower()] = new
    return table


def normalize_channel_name(raw: str) -> str | None:
    """Return the canonical spelling of a channel name, or ``None`` if it isn't an electrode.

    Steps: trim; strip an EDF type prefix (``EEG ``); strip PhysioNet padding dots; strip a
    reference suffix (``-REF``, ``-A1``, ...); drop inner spaces; case-insensitive lookup.
    Old 10-20 names map to 10-10 names (``T3`` → ``T7``).

    Bipolar derivations such as ``Fp1-F7`` are *not* electrodes and return ``None``.

    Raises:
        TypeError: if ``raw`` isn't a string.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Channel name must be str, got {type(raw).__name__}: {raw!r}")
    name = raw.strip()
    lowered = name.lower()
    for prefix in _TYPE_PREFIXES:
        if lowered.startswith(prefix):
            name = name[len(prefix) :].strip()
            break
    name = name.rstrip(".").strip()
    name = _REFERENCE_SUFFIX_RE.sub("", name)
    name = name.replace(" ", "")
    if not name:
        return None
    return _lookup_table().get(name.lower())


def is_scalp_electrode(name: str) -> bool:
    """True for a canonical scalp electrode (not an ear/mastoid reference site)."""
    return name in canonical_names() and name not in REFERENCE_SITES


def require_canonical(names: Iterable[str], context: DataContext | None = None) -> tuple[str, ...]:
    """Check that every name is already canonical and unique; return them as a tuple.

    Use this at boundaries that must receive canonical names (registry, profiles, recordings).

    Raises:
        NeurOSChannelError: on a non-canonical or duplicate name.
    """
    result = tuple(names)
    bad = [n for n in result if normalize_channel_name(n) != n]
    if bad:
        hints = {n: normalize_channel_name(n) for n in bad}
        raise NeurOSChannelError(
            f"Non-canonical channel names {bad}; canonical spellings would be {hints}",
            context,
        )
    duplicates = sorted({n for n in result if result.count(n) > 1})
    if duplicates:
        raise NeurOSChannelError(f"Duplicate channel names {duplicates}", context)
    return result


def row_of(name: str) -> str | None:
    """Return the row prefix of a canonical position (``FC3`` → ``FC``), or ``None``."""
    match = _POSITION_RE.match(name)
    return match.group("row") if match else None


def lateral_position(name: str) -> float | None:
    """Signed left-right position: negative = left, 0 = midline, positive = right.

    10-10 numbering: odd numbers are left, even numbers right, and the number grows away from
    the midline (C1 is next to Cz, C5 is far left). 10-05 'h' positions sit half a step closer
    to the midline than their number (FCC5h lies between FCC5 and FCC3).
    """
    match = _POSITION_RE.match(name)
    if match is None:
        return None
    pos = match.group("pos")
    if pos in ("z", "Z"):
        return 0.0
    n = int(pos)
    half_shift = 0.5 if match.group("half") else 0.0
    if n % 2 == 1:
        return -(n + 1) / 2 + half_shift
    return n / 2 - half_shift


def spatial_sort_key(name: str) -> tuple[float, float, str]:
    """Sort key: anterior → posterior, then left → right, then name (for determinism)."""
    row = row_of(name)
    lateral = lateral_position(name)
    if row is None or lateral is None or row not in _ROW_LEVEL:
        return (99.0, 0.0, name)
    return (_ROW_LEVEL[row], lateral, name)


def sort_spatially(names: Iterable[str]) -> list[str]:
    """Return names ordered front-to-back, then left-to-right (deterministic)."""
    return sorted(names, key=spatial_sort_key)


def in_rows(names: Iterable[str], rows: Sequence[str] = C_CORE_ROWS) -> list[str]:
    """Keep only names whose row prefix is one of ``rows`` (exact row match, not prefix)."""
    allowed = set(rows)
    return [n for n in names if row_of(n) in allowed]
