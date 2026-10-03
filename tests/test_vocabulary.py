"""The canonical vocabulary and event maps (architecture §4.4, D-023, D-024)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from neuros.data.vocabulary import Vocabulary, load_vocabulary
from neuros.errors import NeurOSConfigError, NeurOSLabelError


def _raw(vocab: Vocabulary) -> dict[str, Any]:
    return vocab.model_dump(mode="json")


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_command_set_matches_d024(vocab: Vocabulary) -> None:
    commands = {vocab.command_of(label) for label in vocab.labels_with_role("command")}
    assert commands == {"LEFT", "RIGHT", "REST"}
    assert vocab.labels_with_role("decoy") == ("feet", "both_hands")
    assert vocab.labels_with_role("heldout") == ("tongue",)


def test_tongue_is_never_trainable(vocab: Vocabulary) -> None:
    trainable = vocab.trainable_labels("intent")
    assert "tongue" not in trainable
    assert trainable == ("left_hand", "right_hand", "rest", "feet", "both_hands")
    assert vocab.trainable_labels("state") == ("state_rest", "state_arithmetic")


def test_every_label_has_an_anchor(vocab: Vocabulary) -> None:
    anchors = vocab.anchors()
    assert set(anchors) == set(vocab.labels)
    assert all(len(text.split()) >= 2 for text in anchors.values())


def test_map_event(vocab: Vocabulary) -> None:
    assert vocab.map_event("PhysionetMI", "hands") == "both_hands"
    assert vocab.map_event("Stieger2021", "both_hand") == "both_hands"
    assert vocab.map_event("BNCI2014_001", "tongue") == "tongue"
    with pytest.raises(NeurOSLabelError, match="not mapped"):
        vocab.map_event("PhysionetMI", "jump")
    with pytest.raises(NeurOSLabelError, match="No event map"):
        vocab.map_event("NoSuchDataset", "left_hand")


def test_load_vocabulary_from_file_roundtrip(vocab: Vocabulary, tmp_path: Path) -> None:
    assert load_vocabulary(_write(tmp_path, _raw(vocab))) == vocab


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(NeurOSConfigError, match="not found"):
        load_vocabulary(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["labels"].pop("rest"), "REST"),
        (lambda r: r["labels"]["feet"].update(command="LEFT"), "unique"),
        (lambda r: r["labels"]["tongue"].update(command="TONGUE"), "must not define a command"),
        (lambda r: r["labels"]["left_hand"].update(command="HOLD"), "reserved"),
        (lambda r: r["labels"]["left_hand"].update(command="left"), "UPPER_SNAKE_CASE"),
        (lambda r: r["labels"]["state_rest"].update(channel="intent"), "state"),
        (lambda r: r["event_maps"]["Cho2017"].update(left_hand="left_fist"), "unknown labels"),
        (lambda r: r["labels"].update({"Bad-Label": r["labels"]["feet"]}), "lower_snake_case"),
        (lambda r: r.update(extra_key=1), "extra"),
    ],
)
def test_invalid_vocabularies_fail_at_load(
    vocab: Vocabulary, tmp_path: Path, mutate: Any, message: str
) -> None:
    raw = _raw(vocab)
    mutate(raw)
    with pytest.raises(NeurOSConfigError, match=message):
        load_vocabulary(_write(tmp_path, raw))
