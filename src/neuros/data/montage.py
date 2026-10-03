"""The canonical montage C-core and headset profiles (architecture §4.3).

**C-core** is the model's fixed input layout: the channels that *every* core motor-imagery
dataset recorded, restricted to the sensorimotor rows (F, FC, C, CP, P). Intersecting guarantees
that, during training, every input row is real data for every core person; a channel only some
datasets have would be zero for most training windows and teach the model little.

Computed on Day 1 (2026-10-03) from the registry's verified channel lists: **26 channels**.
The architecture's earlier estimate of ~18 had also intersected BNCI2014-001's 22-channel cap,
but BNCI2014-001 is a literature dataset, not a core one (D-011), so it enters as a subset like
any headset. The committed result lives in ``montages/c_core.yaml``; a test recomputes it, so the
model's input layout can't drift silently.

**Headset profiles** map a device's channel labels to canonical names. A headset covers a
subset S of C-core; alignment runs on S only and the missing rows are zero-filled (§4.3).
Channels a headset has *outside* C-core (e.g. Unicorn's Oz) are ignored by the model.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from neuros.data.channels import C_CORE_ROWS, in_rows, normalize_channel_name, sort_spatially
from neuros.data.registry import DatasetSpec
from neuros.errors import NeurOSChannelError, NeurOSConfigError

# Channels any motor-imagery layout must keep: left/right hand imagery is read from C3 vs C4.
MOTOR_ESSENTIAL: tuple[str, ...] = ("C3", "Cz", "C4")


def compute_c_core(
    specs: Iterable[DatasetSpec], rows: Sequence[str] = C_CORE_ROWS
) -> tuple[str, ...]:
    """Intersect the datasets' EEG channels within ``rows``; return them in spatial order.

    Raises:
        NeurOSConfigError: if no datasets are given or the result lacks C3, Cz or C4.
    """
    spec_list = list(specs)
    if not spec_list:
        raise NeurOSConfigError("compute_c_core needs at least one dataset")
    common = set(spec_list[0].eeg_channels)
    for spec in spec_list[1:]:
        common &= set(spec.eeg_channels)
    core = tuple(sort_spatially(in_rows(common, rows)))
    missing = [ch for ch in MOTOR_ESSENTIAL if ch not in core]
    if missing:
        keys = [s.key for s in spec_list]
        raise NeurOSConfigError(f"C-core from {keys} lacks motor-essential channels {missing}")
    return core


class _CCoreFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    computed: str
    from_datasets: tuple[str, ...]
    rows: tuple[str, ...]
    channels: tuple[str, ...]


@functools.cache
def load_c_core() -> tuple[str, ...]:
    """The committed C-core channel list (``montages/c_core.yaml``), in model input order.

    Raises:
        NeurOSConfigError: if the file is missing or malformed.
    """
    text = _read_packaged("c_core.yaml")
    try:
        parsed = _CCoreFile.model_validate(yaml.safe_load(text))
    except (ValidationError, yaml.YAMLError) as exc:
        raise NeurOSConfigError(f"montages/c_core.yaml is invalid:\n{exc}") from exc
    if list(parsed.channels) != sort_spatially(parsed.channels):
        raise NeurOSConfigError("montages/c_core.yaml channels must be in spatial order")
    return parsed.channels


# ---------------------------------------------------------------------- headset profiles


class ChannelMap(BaseModel):
    """One device channel and the electrode position it's placed on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = Field(min_length=1)
    canonical: str

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: str) -> str:
        if normalize_channel_name(value) != value:
            raise ValueError(f"{value!r} is not a canonical 10-10 name")
        return value


class HeadsetProfile(BaseModel):
    """A headset: which device channel sits where, and how its stream looks (YAML file)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    native_sfreq: float = Field(gt=0)
    units: Literal["uV", "V"]
    lsl_stream_type: str = "EEG"
    electrodes: Literal["wet", "dry", "mixed"]
    channels: tuple[ChannelMap, ...] = Field(min_length=1)
    accelerometer: tuple[str, ...] = ()
    verified: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _unique(self) -> HeadsetProfile:
        for attr in ("device", "canonical"):
            values = [getattr(c, attr) for c in self.channels]
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"duplicate {attr} names in profile {self.name!r}: {duplicates}")
        return self

    @property
    def canonical_channels(self) -> tuple[str, ...]:
        return tuple(c.canonical for c in self.channels)

    def coverage(self, c_core: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split C-core into (present on this headset, missing) in C-core order."""
        have = set(self.canonical_channels)
        present = tuple(ch for ch in c_core if ch in have)
        missing = tuple(ch for ch in c_core if ch not in have)
        return present, missing

    def require_motor_coverage(self, c_core: Sequence[str]) -> None:
        """Raise unless the headset covers C3 and C4 within C-core (needed for left vs right).

        Raises:
            NeurOSChannelError: naming the missing positions.
        """
        present, _ = self.coverage(c_core)
        missing = [ch for ch in ("C3", "C4") if ch not in present]
        if missing:
            raise NeurOSChannelError(
                f"Headset {self.name!r} lacks {missing}; left vs right imagery can't be decoded"
            )


PACKAGED_PROFILES: tuple[str, ...] = ("ganglion", "bcic2b", "unicorn", "crown")


def load_profile(name_or_path: str | Path) -> HeadsetProfile:
    """Load a packaged profile by name (``"ganglion"``) or a YAML file by path.

    Raises:
        NeurOSConfigError: if the profile is missing or invalid.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        if not candidate.is_file():
            raise NeurOSConfigError(f"Headset profile not found: {candidate}")
        text, source = candidate.read_text(encoding="utf-8"), str(candidate)
    else:
        name = str(name_or_path)
        if name not in PACKAGED_PROFILES:
            raise NeurOSConfigError(f"Unknown headset {name!r}; packaged: {PACKAGED_PROFILES}")
        text, source = _read_packaged(f"{name}.yaml"), f"montages/{name}.yaml"
    try:
        return HeadsetProfile.model_validate(yaml.safe_load(text))
    except (ValidationError, yaml.YAMLError) as exc:
        raise NeurOSConfigError(f"{source}: invalid headset profile:\n{exc}") from exc


def _read_packaged(filename: str) -> str:
    resource = resources.files("neuros.data").joinpath("montages", filename)
    if not resource.is_file():
        raise NeurOSConfigError(f"Packaged montage file missing: montages/{filename}")
    return resource.read_text(encoding="utf-8")
