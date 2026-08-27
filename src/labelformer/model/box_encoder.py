"""Per-frame initial-box encoder (LabelFormer, Yang et al., CoRL 2023).

Each noisy per-frame box ``b_i = (x, y, yaw, l, w)`` is lifted to a
``d_model``-dimensional embedding by a small MLP. The heading enters as
``(sin 2*yaw, cos 2*yaw)`` rather than as a raw angle so that the encoder is
continuous across the +-pi wrap and invariant to the pi-flip ambiguity of BEV
headings (matching the heading parameterization used by the losses).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

BOX_FEAT_DIM = 6
"""Dimensionality of the encoded box vector ``(x, y, sin2yaw, cos2yaw, l, w)``."""


def encode_box_params(boxes: Tensor) -> Tensor:
    """Map BEV boxes (..., 5) to wrap-free features (..., 6).

    Args:
        boxes: ``(..., 5)`` tensor of ``(x, y, yaw, l, w)``.

    Returns:
        ``(..., 6)`` tensor ``(x, y, sin 2*yaw, cos 2*yaw, l, w)``.
    """
    two_yaw = 2.0 * boxes[..., 2]
    return torch.stack(
        [
            boxes[..., 0],
            boxes[..., 1],
            torch.sin(two_yaw),
            torch.cos(two_yaw),
            boxes[..., 3],
            boxes[..., 4],
        ],
        dim=-1,
    )


class BoxEncoder(nn.Module):
    """MLP encoder for per-frame initial boxes.

    Architecture: ``Linear(6 -> d_model) -> ReLU -> Linear(d_model -> d_model)``.
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(BOX_FEAT_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, boxes: Tensor) -> Tensor:
        """Encode boxes ``(B, T, 5)`` into embeddings ``(B, T, d_model)``."""
        return self.mlp(encode_box_params(boxes))
