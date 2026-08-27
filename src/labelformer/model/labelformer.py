"""LabelFormer: window-based offline trajectory refinement (Yang et al., CoRL 2023).

The model consumes a whole object trajectory at once. Every frame contributes
(i) its noisy initial BEV box and (ii) the object's LiDAR points expressed in
that box's frame. The two are encoded separately, fused by addition, and passed
through an ALiBi transformer that reasons over the full temporal window; heads
then predict a per-frame pose residual and a single size residual per object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .box_encoder import BoxEncoder
from .heads import PoseHead, SizeHead, masked_mean
from .pillar_encoder import PillarEncoder, PillarEncoderConfig
from .transformer import AlibiTransformerEncoder


def wrap_angle(a: Tensor) -> Tensor:
    """Wrap angles to ``(-pi, pi]`` (torch mirror of ``geometry.wrap_angle``)."""
    return -((-a + math.pi) % (2 * math.pi) - math.pi)


@dataclass
class LabelFormerConfig:
    """Hyper-parameters of the full model."""

    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    pillar: PillarEncoderConfig = field(default_factory=PillarEncoderConfig)


class LabelFormer(nn.Module):
    """Refine a noisy BEV trajectory given per-frame object points."""

    def __init__(self, config: LabelFormerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LabelFormerConfig()
        cfg = self.config
        self.box_encoder = BoxEncoder(cfg.d_model)
        self.pillar_encoder = PillarEncoder(cfg.pillar)
        self.point_fusion = nn.Linear(cfg.pillar.out_dim, cfg.d_model)
        self.transformer = AlibiTransformerEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        self.pose_head = PoseHead(cfg.d_model)
        self.size_head = SizeHead(cfg.d_model)

    def encode_frames(self, batch: dict[str, Tensor]) -> Tensor:
        """Fuse box and point embeddings into per-frame tokens ``(B, T, d_model)``."""
        p = self.pillar_encoder(batch["points"], batch["points_mask"])
        return self.box_encoder(batch["boxes_init"]) + self.point_fusion(p)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Refine a batch of trajectories.

        Args:
            batch: dict with ``boxes_init`` (B, T, 5), ``points`` (B, T, N, 4),
                ``points_mask`` (B, T, N) bool and ``frame_mask`` (B, T) bool.

        Returns:
            Dict with ``boxes_refined`` (B, T, 5), ``pose_residual`` (B, T, 3)
            and ``size_residual`` (B, 2). Values at padded frames are
            unconstrained but never influence valid frames.
        """
        boxes_init = batch["boxes_init"]
        frame_mask = batch["frame_mask"]

        tokens = self.encode_frames(batch)
        h = self.transformer(tokens, frame_mask)

        pose_residual = self.pose_head(h)
        size_residual = self.size_head(h, frame_mask)

        xy = boxes_init[..., 0:2] + pose_residual[..., 0:2]
        yaw = wrap_angle(boxes_init[..., 2] + pose_residual[..., 2])

        # One size per trajectory: mean of the valid initial sizes plus a residual.
        size = masked_mean(boxes_init[..., 3:5], frame_mask) + size_residual
        size = size.unsqueeze(1).expand(-1, boxes_init.shape[1], -1)

        boxes_refined = torch.cat([xy, yaw.unsqueeze(-1), size], dim=-1)
        return {
            "boxes_refined": boxes_refined,
            "pose_residual": pose_residual,
            "size_residual": size_residual,
        }
