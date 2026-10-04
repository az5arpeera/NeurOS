"""Dataset registry: what each public dataset is, as we rely on it (Step 3 data lock).

Each ``DatasetSpec`` declares the facts the pipeline depends on: the EEG channels, sampling rate,
cue timing, subject ids, event names and licence. The harmonisation and validation code checks
every loaded recording *against* these declarations, so a change upstream (a MOABB update, a
re-released file) fails loudly instead of silently shifting the data.

Provenance of the declarations (2026-10-03):

- Channel lists: copied from the hard-coded lists in MOABB 1.7.2's dataset modules (EEGMAT: from
  the PhysioNet page; verify on the files).
- Subject ids, event names and cue intervals: MOABB 1.7.2 dataset metadata.
- How long the class stays on screen (``class_on_screen_s``, D-030): the paradigm descriptions in
  MOABB 1.7.2's dataset docstrings, read 2026-10-04. Where a description is unclear, the value is
  the whole task (the conservative choice: it can only flag a result, never clear one).
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
EpochRule = Literal["after_cue", "fixed"]

# Tolerance for comparing epoch boundaries in seconds (they're sums of declared decimals).
_EPOCH_TOL_S = 1e-9


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
    # Seconds from the cue's onset during which the screen shows something that depends on the
    # class: the cue itself, or feedback driven by a classifier. Equal to task_s when it stays up
    # for the whole task. None only for paradigms without class cues (EEGMAT). D-030 starts
    # offline windows after it, because cross-person models learn the cue before the imagery.
    class_on_screen_s: float | None
    # Seconds of fixation before the cue, usable as pre-cue rest; None if not established.
    pre_cue_fixation_s: float | None
    rest_source: RestSource
    licence: str
    url: str
    channels_source: str
    unverified: tuple[str, ...] = ()
    # Named subgroups with inclusive subject-id ranges, used to stratify splits.
    subgroups: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    # Subjects excluded from every use, with the reason (data-quality failures found by
    # validation). They stay declared so splits and reports remain stable and transparent.
    excluded_subjects: Mapping[int, str] = field(default_factory=dict)
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
        if self.paradigm == "motor_imagery" and self.class_on_screen_s is None:
            raise NeurOSConfigError(
                f"{self.key}: a cued paradigm must declare class_on_screen_s (D-030)"
            )
        if self.class_on_screen_s is not None and not (
            math.isfinite(self.class_on_screen_s) and 0 <= self.class_on_screen_s <= self.task_s
        ):
            raise NeurOSConfigError(
                f"{self.key}: class_on_screen_s must be within [0, task_s={self.task_s}] s, "
                f"got {self.class_on_screen_s}"
            )
        if self.pre_cue_fixation_s is not None and self.pre_cue_fixation_s <= 0:
            raise NeurOSConfigError(f"{self.key}: pre_cue_fixation_s must be positive or None")
        if self.rest_source == "pre_cue" and self.pre_cue_fixation_s is None:
            raise NeurOSConfigError(f"{self.key}: rest_source 'pre_cue' needs pre_cue_fixation_s")
        unknown_excluded = sorted(set(self.excluded_subjects) - set(self.subjects))
        if unknown_excluded:
            raise NeurOSConfigError(
                f"{self.key}: excluded subjects {unknown_excluded} are not declared"
            )
        if any(not reason.strip() for reason in self.excluded_subjects.values()):
            raise NeurOSConfigError(f"{self.key}: every excluded subject needs a reason")
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

    @property
    def usable_subjects(self) -> tuple[int, ...]:
        """Declared subjects minus documented exclusions: everyone eligible for any use."""
        return tuple(s for s in self.subjects if s not in self.excluded_subjects)

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
        # "A target appears on either the left or the right side of the screen ... until the
        # target disappears" (MOABB/PhysioNet): a lateralised target for the whole task.
        class_on_screen_s=3.0,
        pre_cue_fixation_s=None,
        rest_source="explicit",
        licence="ODC Attribution 1.0 (commercial use with attribution)",
        url="https://physionet.org/content/eegmmidb/1.0.0/",
        channels_source=_MOABB_SOURCE,
        excluded_subjects={
            88: "recorded at 128 Hz, not 160 Hz (validation: sampling_rate), 2026-10-03, D-027",
            92: "recorded at 128 Hz, not 160 Hz (validation: sampling_rate), 2026-10-03, D-027",
            100: (
                "recorded at 128 Hz, not 160 Hz, and only 6 trials per class per run "
                "(validation: sampling_rate), 2026-10-03, D-027"
            ),
        },
        notes=(
            "Imagined runs 4/8/12 (left vs right fist) and 6/10/14 (both fists vs both feet). "
            "T0 annotations between tasks are explicit rest. Raw files are unfiltered (line "
            "noise, drift). 160 Hz confirmed on all 106 usable people (3 Oct). Baseline runs "
            "1-2 (eyes open/closed) are downloaded but not loaded by MOABB; a loader comes "
            "with H1. Files come from PhysioNet's mirror, verified against SHA256SUMS.txt."
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
        # Red arrow for 1.25 s, then a blue feedback bar driven by the online classifier until
        # the task ends (MOABB, after Pillette 2021).
        class_on_screen_s=5.0,
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
        # Users steer a cursor toward one of four on-screen targets for the whole trial (MOABB).
        class_on_screen_s=3.0,
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
        # Imagery "for 4 s when the right or left arrow appeared"; in the online test runs the
        # cross also moves left or right with the classifier's output (MOABB).
        class_on_screen_s=4.0,
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
        # BCI2000 presents a left/right instruction; MOABB doesn't say for how long, so the whole
        # task is assumed until the paper is checked.
        class_on_screen_s=3.0,
        pre_cue_fixation_s=2.0,
        rest_source="pre_cue",
        licence="GigaDB, CC0 (verify, Q7b)",
        url="https://doi.org/10.5524/100295",
        channels_source=_MOABB_SOURCE,
        unverified=("pre_cue_fixation_s", "licence", "class_on_screen_s"),
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
        # The arrow "appeared and stayed on the screen for 1.25 s ... No feedback was provided"
        # (MOABB, from the competition description). The one dataset with a cue-free window.
        class_on_screen_s=1.25,
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
        # Sessions 3-5 show the cue and a smiley moved by the online classifier for the whole
        # task (competition description, Leeb et al.); sessions 1-2 only an arrow for 1.25 s.
        # MOABB's docstring repeats BNCI2014-001's text, so this is unverified.
        class_on_screen_s=4.5,
        pre_cue_fixation_s=3.0,
        rest_source="pre_cue",
        licence="BNCI Horizon 2020 (verify, Q7b)",
        url="https://bnci-horizon-2020.eu/database/data-sets",
        channels_source=_MOABB_SOURCE,
        unverified=("licence", "class_on_screen_s"),
        notes=(
            "3 bipolar-derived channels; doubles as the 3-channel virtual headset. Restricted "
            "to screening sessions 1-2, it would have a cue-free window like BNCI2014-001."
        ),
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
        class_on_screen_s=None,  # 60-s recordings, no class cue
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


@dataclass(frozen=True)
class TaskEpoch:
    """Where one dataset's offline epoch sits, in seconds from the cue's onset (D-030)."""

    start_s: float
    end_s: float
    # "after_cue": placed by the contract's rule; "fixed": the contract's own epoch_s.
    rule: EpochRule
    # True when no class-related display overlaps the window: it starts at least the
    # contract's margin after the class has left the screen, or ends before the cue appears.
    # Results on windows that aren't cue-free must be reported with the cue-only control.
    cue_free: bool

    def to_dict(self) -> dict[str, Any]:
        """The fields every result file records next to its numbers."""
        return {
            "epoch_s": [self.start_s, self.end_s],
            "epoch_rule": self.rule,
            "cue_free": self.cue_free,
        }

    def describe(self) -> str:
        """One line for logs, e.g. ``[1.5, 3.5] s after the cue (cue-free)``."""
        status = (
            "cue-free"
            if self.cue_free
            else "NOT cue-free: the class is on screen, so report the cue-only control"
        )
        return f"[{self.start_s:g}, {self.end_s:g}] s after the cue ({status})"


def task_epoch(spec: DatasetSpec, contract: SignalContract) -> TaskEpoch:
    """The offline task epoch for ``spec`` under ``contract`` (D-030).

    With the contract's rule (``epoch_s`` is None), the window starts ``cue_margin_s`` after the
    class has left the screen. When that window doesn't fit in the task, because the class
    stays on screen too long, the *latest* window inside the task is used instead: the one
    furthest from the cue's onset, whose brain response is the strongest and most alike across
    people. That window is marked ``cue_free=False``.

    A fixed ``epoch_s`` (diagnostics, the cue-only control) is used as given, and ``cue_free``
    says whether it happens to avoid the class display.

    Raises:
        NeurOSConfigError: for a paradigm without class cues, a task shorter than one window, or
            a fixed epoch that runs past the end of the task (it would cut into the next phase
            of the trial and mislabel it).
    """
    if spec.class_on_screen_s is None:
        raise NeurOSConfigError(f"{spec.key}: no class cue (paradigm {spec.paradigm}), no epochs")
    clear_from = spec.class_on_screen_s + contract.cue_margin_s
    if contract.epoch_s is None:
        rule: EpochRule = "after_cue"
        latest_start = spec.task_s - contract.window_s
        if latest_start < -_EPOCH_TOL_S:
            raise NeurOSConfigError(
                f"{spec.key}: the task ({spec.task_s} s) is shorter than one window "
                f"({contract.window_s} s)"
            )
        start = min(clear_from, max(latest_start, 0.0))
        end = start + contract.window_s
    else:
        rule = "fixed"
        start, end = contract.epoch_s
        if end > spec.task_s + _EPOCH_TOL_S:
            raise NeurOSConfigError(
                f"{spec.key}: epoch [{start:g}, {end:g}] s runs past the task, which ends "
                f"{spec.task_s:g} s after the cue"
            )
    cue_free = start >= clear_from - _EPOCH_TOL_S or end <= _EPOCH_TOL_S
    return TaskEpoch(round(start, 9), round(end, 9), rule, cue_free)


def check_registry(vocabulary: Vocabulary, contract: SignalContract) -> None:
    """Cross-check the registry against the vocabulary and the signal contract.

    Raises:
        NeurOSConfigError: if a dataset has no event map, or a motor-imagery dataset can't host
            the contract's offline epoch (see ``task_epoch``).
    """
    problems: list[str] = []
    for spec in _SPECS:
        if spec.key not in vocabulary.event_maps:
            problems.append(f"{spec.key}: no event map in labels.yaml")
        if spec.paradigm == "motor_imagery":
            try:
                task_epoch(spec, contract)
            except NeurOSConfigError as exc:
                problems.append(str(exc))
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
