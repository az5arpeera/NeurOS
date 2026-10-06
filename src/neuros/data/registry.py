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

Roles (D-011, D-012, D-026, D-033):

- ``core``: the five motor-imagery training datasets (~364 people). C-core, the model's input
  layout, is computed from these alone.
- ``literature``: BNCI2014-001/004, for comparison with published leave-one-subject-out numbers.
  E0 uses all of their people during development, so they're excluded from the locked test set.
- ``brain_state``: EEGMAT (rest vs mental arithmetic), the brain-state channel.
- ``cue_free``: motor-imagery datasets whose screen shows nothing class-related during imagery
  (D-033, 4 Oct): the cue-free evidence that D-030 asks for. Yang2025 trains and tests;
  Shin2017A only tests (GPL-3.0, see ``train_restriction``).
- ``diagnostic``: GuttmannFlury2025 (D-033). Its cue stays on screen, but it records gaze, so it
  measures how much decoding follows the eyes. Never trains a model we ship.

Which data may train a model we ship is a separate question from the role: ``train_restriction``
says why a dataset may not (licence, or a decision). Internal diagnostics on such a dataset (the
cue audit, within-dataset baselines) are allowed; their models are thrown away (D-033).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from neuros.data.channels import require_canonical
from neuros.errors import DataContext, NeurOSConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from neuros.config.schemas import SignalContract
    from neuros.data.vocabulary import Vocabulary

DatasetRole = Literal["core", "literature", "brain_state", "cue_free", "diagnostic"]
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
    # (BNCI2014-001: cue at +2 s), others at the cue itself, and a few after it: Shin2017A marks
    # the start of imagery, 2 s after its arrow appeared, so its offset is negative (-2 s).
    # "The cue onset" is always when the class is first shown; every epoch is timed from it.
    cue_offset_s: float
    # Seconds from the cue onset to the end of the task (imagery, or the whole labelled segment
    # for EEGMAT). Includes the cue period when imagery starts only after it (see task_from_s).
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
    # Seconds from the cue onset to the start of imagery. 0 when people start imagining as the
    # cue appears (most datasets); Yang2025 plays a 1.5-s video first and Shin2017A shows its
    # arrow for 2 s, then imagery starts. MOABB's interval covers [task_from_s, task_s].
    task_from_s: float = 0.0
    # Keyword arguments for the MOABB class (e.g. Yang2025's paradigm_type="2C"). Never a
    # licence acceptance: that is given per download, explicitly (licence_acceptance below).
    moabb_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # Why this dataset may NOT train a model we ship (None = it may). Internal diagnostics are
    # still allowed; nothing trained on it ships (D-033).
    train_restriction: str | None = None
    # A licence MOABB makes the downloader accept first (Shin2017A: GPL-3.0). `neuros download`
    # refuses such a dataset unless the person running it passes --accept-licence.
    licence_acceptance: str | None = None
    # EEG channels in the files that must not be used (publisher-reported faults), with reasons.
    # They stay in eeg_channels, so the file check still sees them; harmonisation drops them.
    excluded_channels: Mapping[str, str] = field(default_factory=dict)
    # Factor from what MOABB returns (after the volts -> µV conversion) to true µV. 1.0 for
    # correctly scaled files. Cho2017 (D-035): its files hold BioSemi's raw steps (31.25 nV each)
    # rather than µV, so MOABB's values read 32x too large; 1/32 corrects them. Yang2025
    # (D-036): its files hold µV but MOABB converts them as nV, so they read 1000x too small.
    # Applied in harmonize_raw and part of the cache key (every non-descriptive field is).
    unit_scale: float = 1.0
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
        # Negative is allowed: the marker may come after the cue (see the field's comment).
        if not math.isfinite(self.cue_offset_s):
            raise NeurOSConfigError(f"{self.key}: cue_offset_s must be finite")
        # A unit correction, not a gain knob: anything beyond 1000x either way is a mistake.
        if not (math.isfinite(self.unit_scale) and 1e-3 <= self.unit_scale <= 1e3):
            raise NeurOSConfigError(
                f"{self.key}: unit_scale must be within [0.001, 1000], got {self.unit_scale}"
            )
        if not (math.isfinite(self.task_from_s) and 0 <= self.task_from_s < self.task_s):
            raise NeurOSConfigError(
                f"{self.key}: task_from_s must be within [0, task_s={self.task_s}) s, "
                f"got {self.task_from_s}"
            )
        self._check_usage_fields()
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

    def _check_usage_fields(self) -> None:
        """MOABB arguments, training restrictions, licence acceptance and excluded channels."""
        if self.moabb_kwargs and self.moabb_class is None:
            raise NeurOSConfigError(f"{self.key}: moabb_kwargs given but no moabb_class")
        if any(not isinstance(name, str) or not name for name in self.moabb_kwargs):
            raise NeurOSConfigError(f"{self.key}: moabb_kwargs keys must be non-empty strings")
        if "accept" in self.moabb_kwargs:
            # A licence is accepted by a person, per download, never by a config file.
            raise NeurOSConfigError(
                f"{self.key}: 'accept' can't be a registry argument; declare "
                "licence_acceptance and pass --accept-licence to `neuros download`"
            )
        for name in ("train_restriction", "licence_acceptance"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise NeurOSConfigError(f"{self.key}: {name} must be None or a reason, not blank")
        if self.role == "diagnostic" and self.train_restriction is None:
            raise NeurOSConfigError(
                f"{self.key}: a diagnostic dataset never trains a model we ship; "
                "declare train_restriction"
            )
        unknown = sorted(set(self.excluded_channels) - set(self.eeg_channels))
        if unknown:
            raise NeurOSConfigError(
                f"{self.key}: excluded channels {unknown} are not declared EEG channels"
            )
        if any(not reason.strip() for reason in self.excluded_channels.values()):
            raise NeurOSConfigError(f"{self.key}: every excluded channel needs a reason")
        if not self.usable_eeg_channels:
            raise NeurOSConfigError(f"{self.key}: every EEG channel is excluded")

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

    @property
    def usable_eeg_channels(self) -> tuple[str, ...]:
        """Declared EEG channels minus excluded ones, in declaration order: the model inputs."""
        return tuple(ch for ch in self.eeg_channels if ch not in self.excluded_channels)

    @property
    def may_train(self) -> bool:
        """True if this dataset's people may train a model we ship (see train_restriction)."""
        return self.train_restriction is None

    @property
    def use(self) -> str:
        """One word for tables: 'train' (and test), 'test' only, or 'diagnostic'."""
        if self.role == "diagnostic":
            return "diagnostic"
        return "train" if self.may_train else "test"

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

# Neuracle NeuSen W 64-channel cap: 59 EEG channels (no AFz or CPz: probably ground and
# reference), then ECG and four EOG channels. MOABB's order.
_YANG2025_CHANNELS = (
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2",
    "C3", "C4", "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
)  # fmt: skip

# 30 electrodes on 10-5 positions. The motor strip is sampled at half positions (FCC*h, CCP*h)
# around C3 and C4, not at C3 and C4 themselves. MOABB reads the names from the files; these
# come from its metadata.
_SHIN2017A_CHANNELS = (
    "AFp1", "AFp2", "AFF1h", "AFF2h", "AFF5h", "AFF6h", "F3", "F4", "F7", "F8", "FCC3h",
    "FCC4h", "FCC5h", "FCC6h", "T7", "T8", "Cz", "CCP3h", "CCP4h", "CCP5h", "CCP6h", "Pz", "P3",
    "P4", "P7", "P8", "PPO1h", "PPO2h", "POO1", "POO2",
)  # fmt: skip

# Neuroscan Quik-Cap: 62 electrodes, of which CB1 and CB2 (over the cerebellum) have no 10-05
# position and are declared non-EEG. MOABB's order, in canonical spelling.
_GUTTMANNFLURY2025_CHANNELS = (
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7",
    "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6",
    "PO8", "O1", "Oz", "O2",
)  # fmt: skip

# Guttmann-Flury 2025, Technical Validation: "It is recommended that any channel marked as
# defective in any session be excluded from the entire analysis." Their list, verbatim.
_GUTTMANNFLURY2025_DEFECTIVE = (
    "PO3", "F1", "POz", "Oz", "F3", "O2", "P8", "PO7", "FC3", "P7", "P4",
)  # fmt: skip

_MOABB_SOURCE = "MOABB 1.7.2 source (hard-coded channel list), copied 2026-10-03"
_MOABB_SOURCE_D033 = "MOABB 1.7.2 source (hard-coded channel list), copied 2026-10-04"

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
        licence="CC BY 4.0 (Zenodo record 8089820, checked 2026-10-04)",
        url="https://doi.org/10.1038/s41597-023-02445-z",
        channels_source=_MOABB_SOURCE,
        unverified=("cue_offset_s", "pre_cue_fixation_s"),
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
        licence="CC BY 4.0 (Figshare article 13123148, checked 2026-10-04)",
        url="https://doi.org/10.6084/m9.figshare.13123148",
        channels_source=_MOABB_SOURCE,
        # Sessions 1-2 of each person (the data lock's plan): 124 files, 77.5 GB of 377 GB.
        # Set here, not per command, so downloading and loading always use the same sessions;
        # MOABB would otherwise download the rest the first time the data is loaded.
        moabb_kwargs={"sessions": (1, 2)},
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
        # D-035 (5 Oct): read as µV, every recording's typical 4-30 Hz amplitude was 160-260 µV
        # (107-143 after common average) and every window exceeded 500 µV peak-to-peak; the
        # spectrum is EEG-shaped (1/f, alpha peak) but 20-26x Dreyer2023's at every frequency.
        # The paper's own bad-trial rule (±100 µV, 8-30 Hz) only works at a ~32x smaller scale,
        # and BioSemi's raw step is 31.25 nV = 1/32 µV: the files hold raw steps. Inferred, not
        # documented by the publisher, hence listed as unverified.
        unit_scale=1 / 32,
        unverified=("pre_cue_fixation_s", "licence", "class_on_screen_s", "unit_scale"),
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
    # ------------------------------------------------------------------ added by D-033 (4 Oct)
    # Timing and licences checked at the papers and repositories on 4 Oct, not taken from
    # MOABB's metadata (which got Chang2025's licence and BNCI2014-004's paradigm wrong).
    DatasetSpec(
        key="Yang2025",
        title="Yang et al. 2025, WBCIC-SHU multi-day MI dataset (two-class part)",
        role="cue_free",
        paradigm="motor_imagery",
        moabb_class="Yang2025",
        moabb_kwargs={"paradigm_type": "2C"},
        subjects=tuple(range(1, 52)),
        eeg_channels=_YANG2025_CHANNELS,
        non_eeg_channels=("ECG", "HEOR", "HEOL", "VEOU", "VEOL", "stim"),
        native_sfreq=1000.0,
        cue_offset_s=0.0,
        # "The first 1.5 seconds of a single trial involve brief visual and auditory cues. After
        # the brief cues, participants perform the corresponding MI task ... lasting for 4
        # seconds" (paper). The marker is the video's onset (MOABB: interval [1.5, 5.5]).
        task_from_s=1.5,
        task_s=5.5,
        # The class is shown by a 1.5-s video of the hand plus a sound. The paper doesn't say
        # what the screen shows during imagery ("no unnecessary visual and auditory stimuli");
        # the white cross on black marks the break *after* it. Unverified: the cue audit's eye
        # and visual channels check it on the data.
        class_on_screen_s=1.5,
        # The 2-s break (white cross on black) before each video. It follows the previous
        # trial's imagery, so its first second may still carry the post-imagery beta rebound.
        pre_cue_fixation_s=2.0,
        rest_source="pre_cue",
        licence="CC BY 4.0 (paper and Figshare record, checked 2026-10-04)",
        url="https://doi.org/10.25452/figshare.plus.22671172",
        channels_source=_MOABB_SOURCE_D033,
        # D-036 (6 Oct): MOABB 1.7 multiplies these files by 1e-9, taking them for nanovolts,
        # but every one of the 153 two-class files declares its unit as "?V" (bytes 3F 56): a
        # micro sign lost in encoding (an ASCII "nV" would have survived), over a +-375000
        # physical range on 24 bits. The paper's Table 1 gives the amplifier's input range as
        # "+/-375 mV" (ADC 24 bits): +-375000 is that range in µV (in nV it would be +-375 µV).
        # MNE doesn't recognise "?V" and passes the values through as volts, so MOABB's factor
        # leaves them 1000x too small: every recording failed the units check (typical
        # amplitude 0.003 "µV"). The spectrum is EEG-shaped and ~1000x (690-2000x) below
        # Dreyer2023's and Lee2019_MI's at every frequency from 4 to 45 Hz; x1000 puts the
        # typical amplitude at ~3 µV, the middle of the other datasets' 2.9-4.2.
        unit_scale=1000.0,
        unverified=("class_on_screen_s", "pre_cue_fixation_s", "non_eeg_channels"),
        notes=(
            "Two-class paradigm only: 51 people x 3 sessions on different days x 200 trials. "
            "The three-class paradigm's 11 people (left/right/feet) are a separate MOABB "
            "instance, not registered. Each session starts with 60 s eyes open and 60 s eyes "
            "closed (baseline-run rest for H1; needs a loader). Raw Neuracle BDF in µV (header "
            "unit '?V'); MOABB converts it as nV, which unit_scale corrects (D-036). One 65.6 GB "
            "ZIP on Figshare+ with a published MD5."
        ),
    ),
    DatasetSpec(
        key="Shin2017A",
        title="Shin et al. 2017, hybrid EEG-NIRS dataset A (left vs right hand MI)",
        role="cue_free",
        paradigm="motor_imagery",
        moabb_class="Shin2017A",
        subjects=tuple(range(1, 30)),
        eeg_channels=_SHIN2017A_CHANNELS,
        non_eeg_channels=("HEOG", "VEOG", "Stim"),
        native_sfreq=200.0,
        # MOABB: "marker is for *task* start not cue start". The black arrow is shown for 2 s
        # before the task (MOABB's docstring, from the paper), so the cue onset is 2 s earlier.
        cue_offset_s=-2.0,
        task_from_s=2.0,
        task_s=12.0,
        # "A black arrow pointing to either the left or right side appeared at the center of the
        # screen for 2 s. The arrow disappeared with a short beep sound and then a black
        # fixation cross was displayed during the task period."
        class_on_screen_s=2.0,
        # A 15-17 s rest with the fixation cross follows each 1-s 'STOP'; at least 14 s of it
        # precedes every arrow.
        pre_cue_fixation_s=14.0,
        rest_source="pre_cue",
        licence="GPL-3.0 (TU Berlin data page; MOABB makes the downloader accept it)",
        url="https://doi.org/10.1109/TNSRE.2016.2628057",
        channels_source="MOABB 1.7.2 dataset metadata (sensors list), copied 2026-10-04",
        train_restriction=(
            "GPL-3.0: testing only until the founders decide whether GPL data may train a model "
            "we ship (D-033)"
        ),
        licence_acceptance="GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.txt)",
        unverified=("cue_offset_s", "eeg_channels", "non_eeg_channels"),
        notes=(
            "3 MI sessions x 20 trials (10 per class): 60 trials per person, 10 s of imagery "
            "each. 200 Hz release; MOABB loads the 'with occular artifact' (uncleaned) files. "
            "Only 6 of the 26 C-core channels (F3, F4, Cz, P3, Pz, P4), and no C3 or C4: the "
            "model can't read it without a montage decision (Q27). Six zips, 6.26 GB, which "
            "also hold dataset B (mental arithmetic vs rest, a candidate for testing E6). The "
            "paper is paywalled: the timing comes from MOABB's account of it."
        ),
    ),
    DatasetSpec(
        key="GuttmannFlury2025",
        title="Guttmann-Flury et al. 2025, Eye-BCI multimodal dataset (motor imagery)",
        role="diagnostic",
        paradigm="motor_imagery",
        moabb_class="GuttmannFlury2025_MI",
        subjects=tuple(range(1, 32)),
        eeg_channels=_GUTTMANNFLURY2025_CHANNELS,
        non_eeg_channels=("CB1", "CB2", "M1", "M2", "HEO", "VEO", "Trig", "stim"),
        native_sfreq=1000.0,
        cue_offset_s=0.0,
        task_s=4.0,
        # "A red rectangle cue randomly appears on either side of the cross for 4 seconds. Upon
        # cue onset, participants initiate the mental task" (paper): lateral, for the whole task.
        class_on_screen_s=4.0,
        # "Each trial begins with a white fixation cross displayed at the screen's center for 2
        # seconds" (paper).
        pre_cue_fixation_s=2.0,
        rest_source="pre_cue",
        licence="CC0 1.0 (the data: paper's Data Records and the Zenodo re-host); the article "
        "itself is CC BY-NC-ND",
        url="https://doi.org/10.7303/syn64005218",
        channels_source=_MOABB_SOURCE_D033,
        train_restriction=(
            "diagnostic only (D-033): it measures how much decoding follows gaze, so it never "
            "trains a model we ship"
        ),
        excluded_channels=dict.fromkeys(
            _GUTTMANNFLURY2025_DEFECTIVE,
            "malfunctioning or bridged in at least one session; the authors recommend excluding "
            "it from the entire analysis (paper, Technical Validation)",
        ),
        unverified=("cue_offset_s", "non_eeg_channels"),
        notes=(
            "1-3 sessions per person (63 in all) x 40 trials (20 per class). Gaze: Tobii TX300 "
            "at 300 Hz (MIxxx_tobii.csv), plus high-speed eye video and E-Prime timing. MOABB "
            "loads a Zenodo re-host (record 18970793; EEG converted from Neuroscan CSV to BDF, "
            "eye data copied byte for byte): 31 per-person zips, 26.2 GB, with published MD5s. "
            "The original trigger codes are trial numbers; the re-host encodes 1 = left, "
            "2 = right (to check: 20 + 20 per session). C-core coverage 23 of 26 (F3, FC3 and "
            "P4 are among the defective channels)."
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
    """Datasets that contribute people to the locked test set (D-026, D-033).

    Every role except ``literature``. ``neuros.eval.splits`` pins each version of the locked
    test set to an explicit list; a test checks that the current version matches this one, so
    a newly registered dataset can't be left out of the test set by accident.
    """
    return keys_with_role("core", "brain_state", "cue_free", "diagnostic")


def trainable_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """The datasets among ``keys`` whose people may train a model we ship, in the given order."""
    return tuple(key for key in keys if get_spec(key).may_train)


def assert_may_train(keys: Sequence[str], purpose: str) -> None:
    """Raise unless every dataset in ``keys`` may train a model we ship (D-033).

    Call it wherever data enters the training of a model that could ship: pooled training,
    model export. Internal diagnostics on restricted datasets don't call it.

    Raises:
        NeurOSConfigError: naming each restricted dataset and its reason.
    """
    restricted = [get_spec(key) for key in keys if not get_spec(key).may_train]
    if restricted:
        reasons = "; ".join(f"{spec.key}: {spec.train_restriction}" for spec in restricted)
        raise NeurOSConfigError(f"Datasets that may not train for {purpose}: {reasons}")


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
    class has left the screen, and never before imagery starts (``task_from_s``). When that
    window doesn't fit in the task, because the class stays on screen too long, the *latest*
    window inside the task is used instead: the one furthest from the cue's onset, whose brain
    response is the strongest and most alike across people. That window is marked
    ``cue_free=False``.

    A fixed ``epoch_s`` (diagnostics, the cue-only control) is used as given, and ``cue_free``
    says whether it happens to avoid the class display. It may start before imagery does: the
    cue-only control [0, 0.5] s is meant to cover the cue.

    Raises:
        NeurOSConfigError: for a paradigm without class cues, imagery shorter than one window,
            or a fixed epoch that runs past the end of the task (it would cut into the next
            phase of the trial and mislabel it).
    """
    if spec.class_on_screen_s is None:
        raise NeurOSConfigError(f"{spec.key}: no class cue (paradigm {spec.paradigm}), no epochs")
    clear_from = spec.class_on_screen_s + contract.cue_margin_s
    if contract.epoch_s is None:
        rule: EpochRule = "after_cue"
        latest_start = spec.task_s - contract.window_s
        if latest_start < spec.task_from_s - _EPOCH_TOL_S:
            raise NeurOSConfigError(
                f"{spec.key}: imagery ({spec.task_s - spec.task_from_s:g} s) is shorter than "
                f"one window ({contract.window_s} s)"
            )
        start = max(spec.task_from_s, min(clear_from, latest_start))
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

    The registry's ``moabb_kwargs`` come first; ``kwargs`` add to them but may not change them
    (a caller can't quietly load Yang2025's three-class paradigm under the two-class key).

    Raises:
        NeurOSConfigError: if the dataset isn't served by MOABB, the class is missing, or
            ``kwargs`` contradicts the registry.
    """
    if spec.moabb_class is None:
        raise NeurOSConfigError(f"{spec.key} is not a MOABB dataset; use its dedicated loader")
    clashes = sorted(
        k for k in kwargs if k in spec.moabb_kwargs and kwargs[k] != spec.moabb_kwargs[k]
    )
    if clashes:
        raise NeurOSConfigError(f"{spec.key}: arguments {clashes} contradict the registry")
    import moabb.datasets  # imported lazily: slow to import, and only needed here

    cls = getattr(moabb.datasets, spec.moabb_class, None)
    if cls is None:
        raise NeurOSConfigError(
            f"MOABB has no class {spec.moabb_class!r} for {spec.key}; check the MOABB version"
        )
    return cls(**{**spec.moabb_kwargs, **kwargs})


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
    # MOABB's interval is the imagery period, in seconds from its event marker. Ours is timed
    # from the cue onset, which sits cue_offset_s after the marker; imagery starts task_from_s
    # after the cue onset.
    interval = [float(x) for x in dataset.interval]
    imagery_start = spec.cue_offset_s + spec.task_from_s
    if not math.isclose(interval[0], imagery_start, abs_tol=1e-9):
        problems.append(
            f"{spec.key}: MOABB interval starts at {interval[0]} s but cue_offset_s + "
            f"task_from_s = {imagery_start} s"
        )
    imagery_s = spec.task_s - spec.task_from_s
    if not math.isclose(interval[1] - interval[0], imagery_s):
        problems.append(
            f"{spec.key}: MOABB task length {interval[1] - interval[0]} s but task_s - "
            f"task_from_s = {imagery_s} s"
        )
    return problems
