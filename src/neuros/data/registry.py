"""Dataset registry: what each public dataset is, as we rely on it (Step 3 data lock).

Each ``DatasetSpec`` declares the facts the pipeline depends on: the EEG channels, sampling rate,
cue timing, subject ids, event names and licence. The harmonisation and validation code checks
every loaded recording *against* these declarations, so a change upstream (a MOABB update, a
re-released file) fails loudly instead of silently shifting the data.

Provenance of the declarations (2026-10-03):

- Channel lists: copied from the hard-coded lists in MOABB 1.7.2's dataset modules (EEGMAT: from
  the PhysioNet page; verify on the files).
- Subject ids, event names and cue intervals: MOABB 1.7.2 dataset metadata.
- Fields still unconfirmed on the actual files are listed in ``unverified``. ``neuros validate``
  confirms them once the data is downloaded; then the entry is updated and the field removed.

Roles (D-011, D-012, D-026):

- ``core``: the five motor-imagery training datasets (~364 people).
- ``literature``: BNCI2014-001/004, for comparison with published leave-one-subject-out numbers.
  E0 uses all of their people during development, so they're excluded from the locked test set.
- ``brain_state``: EEGMAT (rest vs mental arithmetic), the brain-state channel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from neuros.data.channels import require_canonical
from neuros.errors import DataContext, NeurOSConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from neuros.config.schemas import SignalContract
    from neuros.data.vocabulary import Vocabulary

DatasetRole = Literal["core", "literature", "brain_state"]
Paradigm = Literal["motor_imagery", "mental_arithmetic"]
RestSource = Literal["explicit", "pre_cue", "none"]


@dataclass(frozen=True)
class DatasetSpec:
    """Declared facts about one dataset. Validated on construction."""

    key: str
    title: str
    role: DatasetRole
    paradigm: Paradigm
    moabb_class: str | None
    subjects: tuple[int, ...]
    eeg_channels: tuple[str, ...]
    # Channels present in the files that are *not* model inputs (EOG, EMG, stim, references).
    non_eeg_channels: tuple[str, ...]
    native_sfreq: float
    # Seconds from the event marker to the cue onset. MOABB places some markers at trial start
    # (BNCI2014-001: cue at +2 s) and others at the cue itself.
    cue_offset_s: float
    # Seconds of task (imagery, or the whole labelled segment for EEGMAT) after the cue.
    task_s: float
    # Seconds of fixation before the cue, usable as pre-cue rest; None if not established.
    pre_cue_fixation_s: float | None
    rest_source: RestSource
    licence: str
    url: str
    channels_source: str
    unverified: tuple[str, ...] = ()
    # Named subgroups with inclusive subject-id ranges, used to stratify splits.
    subgroups: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        context = DataContext(dataset=self.key)
        require_canonical(self.eeg_channels, context)
        if not self.eeg_channels:
            raise NeurOSConfigError(f"{self.key}: declares no EEG channels")
        overlap = set(self.eeg_channels) & set(self.non_eeg_channels)
        if overlap:
            raise NeurOSConfigError(f"{self.key}: channels both EEG and non-EEG: {sorted(overlap)}")
        if not self.subjects:
            raise NeurOSConfigError(f"{self.key}: declares no subjects")
        if list(self.subjects) != sorted(set(self.subjects)):
            raise NeurOSConfigError(f"{self.key}: subject ids must be unique and sorted")
        for name, value in (("native_sfreq", self.native_sfreq), ("task_s", self.task_s)):
            if not (math.isfinite(value) and value > 0):
                raise NeurOSConfigError(f"{self.key}: {name} must be positive, got {value}")
        if not math.isfinite(self.cue_offset_s) or self.cue_offset_s < 0:
            raise NeurOSConfigError(f"{self.key}: cue_offset_s must be >= 0")
        if self.pre_cue_fixation_s is not None and self.pre_cue_fixation_s <= 0:
            raise NeurOSConfigError(f"{self.key}: pre_cue_fixation_s must be positive or None")
        if self.rest_source == "pre_cue" and self.pre_cue_fixation_s is None:
            raise NeurOSConfigError(f"{self.key}: rest_source 'pre_cue' needs pre_cue_fixation_s")
        self._check_subgroups()

    def _check_subgroups(self) -> None:
        if not self.subgroups:
            return
        assigned: dict[int, str] = {}
        for group, (first, last) in self.subgroups.items():
            if first > last:
                raise NeurOSConfigError(f"{self.key}: subgroup {group!r} has first > last")
            for subject in range(first, last + 1):
                if subject in assigned:
                    raise NeurOSConfigError(
                        f"{self.key}: subject {subject} in subgroups "
                        f"{assigned[subject]!r} and {group!r}"
                    )
                assigned[subject] = group
        if set(assigned) != set(self.subjects):
            missing = sorted(set(self.subjects) - set(assigned))
            extra = sorted(set(assigned) - set(self.subjects))
            raise NeurOSConfigError(
                f"{self.key}: subgroups must cover the subjects exactly "
                f"(missing {missing}, extra {extra})"
            )

    # ------------------------------------------------------------------ queries
    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    def subgroup_of(self, subject: int) -> str:
        """Subgroup name of a subject, or ``"all"`` when the dataset has no subgroups."""
        if subject not in self.subjects:
            raise NeurOSConfigError(f"{self.key}: unknown subject {subject}")
        for group, (first, last) in self.subgroups.items():
            if first <= subject <= last:
                return group
        return "all"


# ---------------------------------------------------------------------- declarations

_PHYSIONET_MI_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz",
    "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8",
    "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3",
    "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
)  # fmt: skip

_BIOSEMI64_CHANNELS = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5",
    "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz",
    "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8",
    "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2",
    "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)  # fmt: skip

_LEE2019_CHANNELS = (
    "AF3", "AF4", "AF7", "AF8", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4",
    "CP5", "CP6", "CPz", "Cz", "F10", "F3", "F4", "F7", "F8", "F9", "FC1", "FC2", "FC3", "FC4",
    "FC5", "FC6", "FT10", "FT9", "FTT10h", "FTT9h", "Fp1", "Fp2", "Fz", "O1", "O2", "Oz", "P1",
    "P2", "P3", "P4", "P7", "P8", "PO10", "PO3", "PO4", "PO9", "POz", "Pz", "T7", "T8", "TP10",
    "TP7", "TP8", "TP9", "TPP10h", "TPP8h", "TPP9h", "TTP7h",
)  # fmt: skip

_DREYER2023_CHANNELS = (
    "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz",
    "F3", "F4", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz", "Fz", "P3", "P4", "Pz",
)  # fmt: skip

_STIEGER2021_CHANNELS = (
    "AF3", "AF4", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "CPz", "Cz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3", "FC4",
    "FC5", "FC6", "FCz", "FT7", "FT8", "Fp1", "Fp2", "Fpz", "Fz", "O1", "O2", "Oz", "P1", "P2",
    "P3", "P4", "P5", "P6", "P7", "P8", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "POz", "Pz",
    "T7", "T8", "TP7", "TP8",
)  # fmt: skip

_BNCI2014_001_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3",
    "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)  # fmt: skip

# EEGMAT uses old 10-20 names (T3, T4, T5, T6); listed here already renamed to 10-10.
_EEGMAT_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
    "P4", "P8", "O1", "O2",
)  # fmt: skip

_MOABB_SOURCE = "MOABB 1.7.2 source (hard-coded channel list), copied 2026-10-03"

_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="PhysionetMI",
        title="EEG Motor Movement/Imagery Dataset (EEGMMIDB)",
        role="core",
        paradigm="motor_imagery",
        moabb_class="PhysionetMI",
        subjects=tuple(range(1, 110)),
        eeg_channels=_PHYSIONET_MI_CHANNELS,
        non_eeg_channels=("stim", "STI 014"),
        native_sfreq=160.0,
        cue_offset_s=0.0,
        task_s=3.0,
        pre_cue_fixation_s=None,
        rest_source="explicit",
        licence="ODC Attribution 1.0 (commercial use with attribution)",
        url="https://physionet.org/content/eegmmidb/1.0.0/",
        channels_source=_MOABB_SOURCE,
        unverified=("native_sfreq",),
        notes=(
            "Imagined runs 4/8/12 (left vs right fist) and 6/10/14 (both fists vs both feet). "
            "T0 annotations between tasks are explicit rest. A few subjects are reported to "
            "have irregular sampling or run lengths: neuros validate lists them. Baseline runs "
            "1-2 (eyes open/closed) aren't loaded by MOABB; a dedicated loader comes with H1."
        ),
    ),
    DatasetSpec(
        key="Dreyer2023",
        title="Dreyer et al. 2023, large Graz-protocol MI-BCI dataset (A/B/C)",
        role="core",
        paradigm="motor_imagery",
        moabb_class="Dreyer2023",
        subjects=tuple(range(1, 88)),
        eeg_channels=_DREYER2023_CHANNELS,
        non_eeg_channels=("EMGd", "EMGg", "EOG1", "EOG2", "EOG3", "stim"),
        native_sfreq=512.0,
        cue_offset_s=0.0,
        task_s=5.0,
        pre_cue_fixation_s=3.0,
        rest_source="pre_cue",
        licence="CC BY 4.0 (verify, Q7b)",
        url="https://doi.org/10.1038/s41597-023-02445-z",
        channels_source=_MOABB_SOURCE,
        unverified=("cue_offset_s", "pre_cue_fixation_s", "licence"),
        subgroups={"A": (1, 60), "B": (61, 81), "C": (82, 87)},
        notes=(
            "Trial: green cross t=0, beep t=2 s, arrow cue t=3 s (1.25 s), feedback 4.25-8 s. "
            "Verify that MOABB's marker sits on the cue (so cue_offset_s = 0)."
        ),
    ),
    DatasetSpec(
        key="Stieger2021",
        title="Stieger et al. 2021, continuous SMR-BCI learning in 62 people",
        role="core",
        paradigm="motor_imagery",
        moabb_class="Stieger2021",
        subjects=tuple(range(1, 63)),
        eeg_channels=_STIEGER2021_CHANNELS,
        non_eeg_channels=("stim",),
        native_sfreq=1000.0,
        cue_offset_s=0.0,
        task_s=3.0,
        pre_cue_fixation_s=None,
        rest_source="explicit",
        licence="CC BY 4.0 (verify, Q7b)",
        url="https://doi.org/10.6084/m9.figshare.13123148",
        channels_source=_MOABB_SOURCE,
        unverified=("licence",),
        notes=(
            "7-11 sessions per person; 377 GB in total, so sessions 1-2 first. Targets: "
            "left/right hand, 'up' = both hands, 'down' = voluntary rest (MOABB labels it rest)."
        ),
    ),
    DatasetSpec(
        key="Lee2019_MI",
        title="Lee et al. 2019 (OpenBMI), motor-imagery paradigm",
        role="core",
        paradigm="motor_imagery",
        moabb_class="Lee2019_MI",
        subjects=tuple(range(1, 55)),
        eeg_channels=_LEE2019_CHANNELS,
        non_eeg_channels=("EMG1", "EMG2", "EMG3", "EMG4", "stim"),
        native_sfreq=1000.0,
        cue_offset_s=0.0,
        task_s=4.0,
        pre_cue_fixation_s=3.0,
        rest_source="pre_cue",
        licence="GigaDB, CC0 (verify, Q7b)",
        url="https://doi.org/10.5524/100542",
        channels_source=_MOABB_SOURCE,
        unverified=("pre_cue_fixation_s", "licence"),
        notes="2 sessions x (offline 'train' run + online 'test' run), 100 trials each.",
    ),
    DatasetSpec(
        key="Cho2017",
        title="Cho et al. 2017, GigaDB motor imagery (52 people)",
        role="core",
        paradigm="motor_imagery",
        moabb_class="Cho2017",
        subjects=tuple(range(1, 53)),
        eeg_channels=_BIOSEMI64_CHANNELS,
        non_eeg_channels=("EMG1", "EMG2", "EMG3", "EMG4", "stim"),
        native_sfreq=512.0,
        cue_offset_s=0.0,
        task_s=3.0,
        pre_cue_fixation_s=2.0,
        rest_source="pre_cue",
        licence="GigaDB, CC0 (verify, Q7b)",
        url="https://doi.org/10.5524/100295",
        channels_source=_MOABB_SOURCE,
        unverified=("pre_cue_fixation_s", "licence"),
        notes=(
            "Biosemi, DC-coupled: large offsets and drift are expected in raw data. Also has "
            "non-task recordings (blinks, eye/head movement, jaw clench) for the artifact gate."
        ),
    ),
    DatasetSpec(
        key="BNCI2014_001",
        title="BCI Competition IV 2a (BNCI2014-001)",
        role="literature",
        paradigm="motor_imagery",
        moabb_class="BNCI2014_001",
        subjects=tuple(range(1, 10)),
        eeg_channels=_BNCI2014_001_CHANNELS,
        non_eeg_channels=("EOG1", "EOG2", "EOG3", "stim"),
        native_sfreq=250.0,
        cue_offset_s=2.0,
        task_s=4.0,
        pre_cue_fixation_s=2.0,
        rest_source="pre_cue",
        licence="BNCI Horizon 2020 (verify, Q7b)",
        url="https://bnci-horizon-2020.eu/database/data-sets",
        channels_source=_MOABB_SOURCE,
        unverified=("licence",),
        notes="Marker at trial start; fixation cross from t=0, cue at t=2 s, imagery until 6 s.",
    ),
    DatasetSpec(
        key="BNCI2014_004",
        title="BCI Competition IV 2b (BNCI2014-004)",
        role="literature",
        paradigm="motor_imagery",
        moabb_class="BNCI2014_004",
        subjects=tuple(range(1, 10)),
        eeg_channels=("C3", "Cz", "C4"),
        non_eeg_channels=("EOG1", "EOG2", "EOG3", "stim"),
        native_sfreq=250.0,
        cue_offset_s=3.0,
        task_s=4.5,
        pre_cue_fixation_s=3.0,
        rest_source="pre_cue",
        licence="BNCI Horizon 2020 (verify, Q7b)",
        url="https://bnci-horizon-2020.eu/database/data-sets",
        channels_source=_MOABB_SOURCE,
        unverified=("licence",),
        notes="3 bipolar-derived channels; doubles as the 3-channel virtual headset.",
    ),
    DatasetSpec(
        key="EEGMAT",
        title="EEG During Mental Arithmetic Tasks (EEGMAT)",
        role="brain_state",
        paradigm="mental_arithmetic",
        moabb_class=None,
        subjects=tuple(range(0, 36)),
        eeg_channels=_EEGMAT_CHANNELS,
        non_eeg_channels=("A2-A1", "ECG"),
        native_sfreq=500.0,
        cue_offset_s=0.0,
        task_s=60.0,
        pre_cue_fixation_s=None,
        rest_source="none",
        licence="ODC Attribution 1.0 (commercial use with attribution)",
        url="https://physionet.org/content/eegmat/1.0.0/",
        channels_source="PhysioNet EEGMAT page (2026-10-02); verify on the files",
        unverified=("eeg_channels", "non_eeg_channels", "native_sfreq"),
        notes=(
            "Per person: _1 = background (rest), _2 = mental arithmetic, 60 s each. Not in MOABB; "
            "our own loader arrives with the other loaders (week 1)."
        ),
    ),
)

DATASETS: Mapping[str, DatasetSpec] = MappingProxyType({spec.key: spec for spec in _SPECS})


def get_spec(key: str) -> DatasetSpec:
    """Return the spec for ``key``.

    Raises:
        NeurOSConfigError: for an unknown key (the message lists the known keys).
    """
    try:
        return DATASETS[key]
    except KeyError:
        raise NeurOSConfigError(f"Unknown dataset {key!r}; known: {sorted(DATASETS)}") from None


def keys_with_role(*roles: DatasetRole) -> tuple[str, ...]:
    """Dataset keys with any of the given roles, in registry order."""
    return tuple(spec.key for spec in _SPECS if spec.role in roles)


def locked_test_dataset_keys() -> tuple[str, ...]:
    """Datasets that contribute people to the locked test set (D-026)."""
    return keys_with_role("core", "brain_state")


def check_registry(vocabulary: Vocabulary, contract: SignalContract) -> None:
    """Cross-check the registry against the vocabulary and the signal contract.

    Raises:
        NeurOSConfigError: if a dataset has no event map, or a motor-imagery task is too short
            to contain the offline epoch (``task_s`` must reach the epoch end).
    """
    problems: list[str] = []
    for spec in _SPECS:
        if spec.key not in vocabulary.event_maps:
            problems.append(f"{spec.key}: no event map in labels.yaml")
        if spec.paradigm == "motor_imagery" and spec.task_s < contract.epoch_s[1]:
            problems.append(
                f"{spec.key}: task_s={spec.task_s} s is shorter than the epoch end "
                f"{contract.epoch_s[1]} s"
            )
        if spec.native_sfreq < contract.sfreq:
            problems.append(
                f"{spec.key}: native rate {spec.native_sfreq} Hz is below the contract "
                f"{contract.sfreq} Hz (upsampling would invent data)"
            )
    if problems:
        raise NeurOSConfigError("Registry check failed:\n  " + "\n  ".join(problems))


def moabb_dataset(spec: DatasetSpec, **kwargs: Any) -> Any:
    """Instantiate the MOABB dataset class for ``spec`` (no data is downloaded).

    Raises:
        NeurOSConfigError: if the dataset isn't served by MOABB or the class is missing.
    """
    if spec.moabb_class is None:
        raise NeurOSConfigError(f"{spec.key} is not a MOABB dataset; use its dedicated loader")
    import moabb.datasets  # imported lazily: slow to import, and only needed here

    cls = getattr(moabb.datasets, spec.moabb_class, None)
    if cls is None:
        raise NeurOSConfigError(
            f"MOABB has no class {spec.moabb_class!r} for {spec.key}; check the MOABB version"
        )
    return cls(**kwargs)


def check_against_moabb(spec: DatasetSpec, vocabulary: Vocabulary) -> list[str]:
    """Compare a spec with MOABB's own metadata; return a list of discrepancies (empty = OK).

    Checks the subject ids and that every MOABB event name has an event map entry.
    """
    dataset = moabb_dataset(spec)
    problems: list[str] = []
    moabb_subjects = tuple(int(s) for s in dataset.subject_list)
    if moabb_subjects != spec.subjects:
        problems.append(
            f"{spec.key}: MOABB subjects {moabb_subjects[:3]}..{moabb_subjects[-3:]} "
            f"(n={len(moabb_subjects)}) differ from the registry (n={spec.n_subjects})"
        )
    mapping = vocabulary.event_maps.get(spec.key, {})
    unmapped = sorted(set(dataset.event_id) - set(mapping))
    if unmapped:
        problems.append(f"{spec.key}: MOABB events {unmapped} have no entry in labels.yaml")
    stale = sorted(set(mapping) - set(dataset.event_id))
    if stale:
        problems.append(f"{spec.key}: labels.yaml maps events MOABB doesn't emit: {stale}")
    interval = [float(x) for x in dataset.interval]
    if not math.isclose(interval[0], spec.cue_offset_s):
        problems.append(
            f"{spec.key}: MOABB interval starts at {interval[0]} s "
            f"but cue_offset_s={spec.cue_offset_s}"
        )
    if not math.isclose(interval[1] - interval[0], spec.task_s):
        problems.append(
            f"{spec.key}: MOABB task length {interval[1] - interval[0]} s but task_s={spec.task_s}"
        )
    return problems
