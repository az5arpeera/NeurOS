"""NeurOS: a zero-calibration intent layer for EEG brain-computer interfaces.

The package turns any EEG stream into intent events (LEFT / RIGHT / REST / HOLD) for a person
the model has never seen, with no calibration task. The design, math and data contract live in
``knowledge-base/architecture/mvp-architecture.md``; module docstrings cite its sections.

Layout (architecture §9):

- ``neuros.data``    datasets: registry, harmonisation, validation, montage, vocabulary
- ``neuros.signal``  causal filters, resampling, windows, quality gate          (week 1)
- ``neuros.align``   covariance + Euclidean alignment, offline and online       (week 1)
- ``neuros.models``  encoder, heads, losses, domain-specific batch norm         (week 1)
- ``neuros.engine``  live engine, policy, WebSocket server, simulator           (week 2)
- ``neuros.eval``    splits, metrics, reports
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
