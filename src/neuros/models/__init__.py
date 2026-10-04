"""Model layer (architecture §6): the EEGNet encoder and the heads.

Needs PyTorch, an optional dependency: ``uv sync --extra dev --extra ml``.
"""

from neuros.models.eegnet import EEGNetClassifier, EEGNetEncoder, LinearHead

__all__ = ["EEGNetClassifier", "EEGNetEncoder", "LinearHead"]
