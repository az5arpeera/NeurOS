"""The canonical label vocabulary (architecture §4.4, decisions D-023 and D-024).

One vocabulary spans every dataset, paradigm and app. Datasets name the same intent differently
("left fist", "left_hand", "left"); per-dataset *event maps* collapse those names onto canonical
labels, and every label carries:

- a **channel**: ``intent`` (commands) or ``state`` (the brain-state channel);
- a **role**: ``command`` (enabled by default), ``decoy`` (trained, not a command unless an app
  opts in; winning means ``HOLD("unrecognized")``), ``heldout`` (never trained; used as
  never-seen activity), or ``state``;
- an **anchor** text for the text-anchored head.

The vocabulary is loaded from ``labels.yaml`` next to this module and validated strictly: a typo
in config must fail at load time, not halfway through a training run.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from neuros.errors import DataContext, NeurOSConfigError, NeurOSLabelError

Channel = Literal["intent", "state"]
Role = Literal["command", "decoy", "heldout", "state"]

# Label ids are lower snake case; command names are upper snake case (they appear in the
# WebSocket schema, architecture §7.3).
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COMMAND_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Commands that must exist for the MVP intent channel (D-024). HOLD is not a label: it's the
# policy's "don't act" output, so it must never be declared here.
REQUIRED_COMMANDS: tuple[str, ...] = ("LEFT", "RIGHT", "REST")
RESERVED_COMMANDS: tuple[str, ...] = ("HOLD",)


class LabelSpec(BaseModel):
    """One canonical label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: Channel
    role: Role
    anchor: str = Field(min_length=3)
    command: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> LabelSpec:
        if self.channel == "state" and self.role != "state":
            raise ValueError("labels on the 'state' channel must have role 'state'")
        if self.channel == "intent" and self.role == "state":
            raise ValueError("role 'state' is only valid on the 'state' channel")
        if self.role in ("command", "decoy") and self.command is None:
            raise ValueError(f"role {self.role!r} needs a command name (used when enabled)")
        if self.role in ("heldout", "state") and self.command is not None:
            raise ValueError(f"role {self.role!r} must not define a command")
        if self.command is not None:
            if not _COMMAND_RE.match(self.command):
                raise ValueError(f"command {self.command!r} must be UPPER_SNAKE_CASE")
            if self.command in RESERVED_COMMANDS:
                raise ValueError(f"command {self.command!r} is reserved for the policy")
        return self


class Vocabulary(BaseModel):
    """The full vocabulary plus every dataset's event map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    labels: dict[str, LabelSpec]
    event_maps: dict[str, dict[str, str]]

    @field_validator("labels")
    @classmethod
    def _check_label_ids(cls, labels: dict[str, LabelSpec]) -> dict[str, LabelSpec]:
        bad = [name for name in labels if not _LABEL_RE.match(name)]
        if bad:
            raise ValueError(f"label ids must be lower_snake_case: {bad}")
        commands = [spec.command for spec in labels.values() if spec.command is not None]
        duplicates = sorted({c for c in commands if commands.count(c) > 1})
        if duplicates:
            raise ValueError(f"command names must be unique: {duplicates}")
        default_commands = {s.command for s in labels.values() if s.role == "command"}
        missing = [c for c in REQUIRED_COMMANDS if c not in default_commands]
        if missing:
            raise ValueError(f"the default command set must include {missing} (D-024)")
        return labels

    @model_validator(mode="after")
    def _check_event_maps(self) -> Vocabulary:
        for dataset, mapping in self.event_maps.items():
            if not mapping:
                raise ValueError(f"event map for {dataset!r} is empty")
            unknown = sorted({target for target in mapping.values() if target not in self.labels})
            if unknown:
                raise ValueError(f"event map for {dataset!r} targets unknown labels {unknown}")
        return self

    # ------------------------------------------------------------------ queries
    def map_event(self, dataset: str, event: str, context: DataContext | None = None) -> str:
        """Map a dataset's event name to its canonical label.

        Raises:
            NeurOSLabelError: if the dataset has no event map or doesn't map ``event``.
        """
        mapping = self.event_maps.get(dataset)
        if mapping is None:
            raise NeurOSLabelError(
                f"No event map for dataset {dataset!r}; add one to labels.yaml",
                context or DataContext(dataset=dataset),
            )
        try:
            return mapping[event]
        except KeyError:
            raise NeurOSLabelError(
                f"Event {event!r} of dataset {dataset!r} is not mapped "
                f"(known: {sorted(mapping)}); add it to labels.yaml",
                context or DataContext(dataset=dataset),
            ) from None

    def labels_with_role(self, role: Role) -> tuple[str, ...]:
        """Label ids with the given role, in declaration order."""
        return tuple(name for name, spec in self.labels.items() if spec.role == role)

    def labels_on_channel(self, channel: Channel) -> tuple[str, ...]:
        """Label ids on the given output channel, in declaration order."""
        return tuple(name for name, spec in self.labels.items() if spec.channel == channel)

    def trainable_labels(self, channel: Channel = "intent") -> tuple[str, ...]:
        """Labels a model may be trained on: everything on the channel except ``heldout``."""
        return tuple(
            name
            for name, spec in self.labels.items()
            if spec.channel == channel and spec.role != "heldout"
        )

    def command_of(self, label: str) -> str | None:
        """The command name a label emits when enabled (``None`` for heldout/state labels)."""
        return self._spec(label).command

    def anchors(self, labels: tuple[str, ...] | None = None) -> Mapping[str, str]:
        """Anchor texts for the given labels (default: all), keyed by label id."""
        names = labels if labels is not None else tuple(self.labels)
        return {name: self._spec(name).anchor for name in names}

    def _spec(self, label: str) -> LabelSpec:
        try:
            return self.labels[label]
        except KeyError:
            raise NeurOSLabelError(f"Unknown canonical label {label!r}") from None


def _parse(text: str, source: str) -> Vocabulary:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise NeurOSConfigError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise NeurOSConfigError(f"{source}: expected a mapping at the top level")
    try:
        return Vocabulary.model_validate(raw)
    except ValidationError as exc:
        raise NeurOSConfigError(f"{source}: invalid vocabulary:\n{exc}") from exc


@functools.cache
def default_vocabulary() -> Vocabulary:
    """The packaged vocabulary (``neuros/data/labels.yaml``), parsed once and cached."""
    text = resources.files("neuros.data").joinpath("labels.yaml").read_text(encoding="utf-8")
    return _parse(text, "neuros/data/labels.yaml")


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    """Load a vocabulary from ``path``, or the packaged default when ``path`` is ``None``.

    Raises:
        NeurOSConfigError: if the file is missing, isn't valid YAML, or fails validation.
    """
    if path is None:
        return default_vocabulary()
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"Vocabulary file not found: {file}")
    return _parse(file.read_text(encoding="utf-8"), str(file))
