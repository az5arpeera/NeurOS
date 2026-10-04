"""Shared fixtures and test configuration.

Tests never touch the network. Tests that need downloaded datasets carry ``@pytest.mark.data``
and only run with ``NEUROS_DATA_TESTS=1`` (plus ``NEUROS_DATA_DIR`` if the data isn't in
``./data``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import settings

from neuros.data.registry import DatasetSpec
from neuros.data.vocabulary import Vocabulary, default_vocabulary

settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("NEUROS_DATA_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs downloaded datasets; set NEUROS_DATA_TESTS=1")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    return default_vocabulary()


def make_spec(**overrides: object) -> DatasetSpec:
    """A small, valid motor-imagery spec for tests; override any field."""
    fields: dict[str, object] = {
        "key": "Synthetic",
        "title": "synthetic test data",
        "role": "core",
        "paradigm": "motor_imagery",
        "moabb_class": None,
        "subjects": (1, 2, 3),
        "eeg_channels": (
            "FC3",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP4",
            "Pz",
        ),
        "non_eeg_channels": ("EOG1", "stim"),
        "native_sfreq": 250.0,
        "cue_offset_s": 0.0,
        "task_s": 4.0,
        "class_on_screen_s": 1.25,  # like BNCI2014-001: after-cue window [1.5, 3.5] (D-030)
        "pre_cue_fixation_s": 2.0,
        "rest_source": "pre_cue",
        "licence": "test",
        "url": "https://example.invalid",
        "channels_source": "test",
    }
    fields.update(overrides)
    return DatasetSpec(**fields)  # type: ignore[arg-type]


@pytest.fixture
def synthetic_spec() -> DatasetSpec:
    return make_spec()
