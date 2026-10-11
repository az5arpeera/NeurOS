"""The optional BrainFlow dependency fails with a clear, actionable error, on every Python version
(BrainFlow itself is only installed on 3.12+; see ``neuros.engine.sources.brainflow_api``)."""

from __future__ import annotations

import sys

import pytest

from neuros.engine.sources import BrainFlowSource, brainflow_api
from neuros.errors import NeurOSConfigError


def test_missing_brainflow_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import raise ImportError, as on a missing package
    # or on Python 3.11's missing pkg_resources.
    monkeypatch.setitem(sys.modules, "brainflow", None)
    monkeypatch.setitem(sys.modules, "brainflow.board_shim", None)
    with pytest.raises(NeurOSConfigError, match=r"engine extra on Python 3\.12"):
        brainflow_api()
    with pytest.raises(NeurOSConfigError, match="engine extra"):
        BrainFlowSource(-1)
