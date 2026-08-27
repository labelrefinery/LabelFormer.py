"""Prediction heads for LabelFormer.

Two heads read the transformer output: a per-frame pose head predicting the
SE(2) residual ``(dx, dy, dyaw)`` of every box, and a trajectory-level size head
predicting a single ``(dl, dw)`` shared by all frames (an object's extent is
constant over time). The final layer of both heads is zero-initialized so an
untrained model predicts the identity refinement.
"""

from __future__ import annotations

from torch import Tensor, nn


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``(B, T, D)`` over frames selected by a ``(B, T)`` bool mask."""
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


def _zero_init(layer: nn.Linear) -> nn.Linear:
    """Zero a linear layer's weight and bias in place."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class PoseHead(nn.Module):
    """Per-frame MLP predicting ``(dx, dy, dyaw)`` residuals."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.out = _zero_init(nn.Linear(d_model, 3))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), self.out
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, T, d_model)`` to pose residuals ``(B, T, 3)``."""
        return self.mlp(x)


class SizeHead(nn.Module):
    """Trajectory-level MLP predicting one ``(dl, dw)`` residual per object."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.out = _zero_init(nn.Linear(d_model, 2))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), self.out
        )

    def forward(self, x: Tensor, frame_mask: Tensor) -> Tensor:
        """Pool ``(B, T, d_model)`` over valid frames and map to ``(B, 2)``."""
        return self.mlp(masked_mean(x, frame_mask))
