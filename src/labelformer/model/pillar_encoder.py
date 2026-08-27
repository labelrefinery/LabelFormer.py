"""PointPillars-style per-frame LiDAR encoder for LabelFormer.

Each frame's object points are already expressed in that frame's *initial box*
frame (box center at the origin, +x along the box heading). The encoder
voxelizes them into a BEV pillar grid, runs a tiny PointNet per pillar, feeds
the resulting pseudo-image through a small ResNet/FPN backbone and reads out
the feature at the grid center -- i.e. at the initial box center -- giving one
observation embedding ``p_i`` per frame.

All ``(B, T)`` frames are processed as a single flattened batch of ``B * T``
pseudo-images, so the scatter-based pooling runs once for the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

POINT_FEAT_DIM = 6
"""Per-point input features: ``(x, y, z, intensity, dx, dy)``."""


@dataclass
class PillarEncoderConfig:
    """Geometry and width of the pillar encoder."""

    x_range: tuple[float, float] = (-12.0, 12.0)
    y_range: tuple[float, float] = (-4.0, 4.0)
    pillar_size: float = 0.1
    point_feat_dim: int = 64
    out_dim: int = 256

    @property
    def grid_w(self) -> int:
        """Number of pillars along x (feature-map width)."""
        return int(round((self.x_range[1] - self.x_range[0]) / self.pillar_size))

    @property
    def grid_h(self) -> int:
        """Number of pillars along y (feature-map height)."""
        return int(round((self.y_range[1] - self.y_range[0]) / self.pillar_size))


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Standard two-conv residual block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block to ``(N, C, H, W)``."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PillarBackbone(nn.Module):
    """Small ResNet + FPN over the pillar pseudo-image.

    Stem (stride 2) -> stage1 (2 blocks, stride 2, 2C) -> stage2 (2 blocks,
    stride 2, 4C); stage2 is upsampled to the stage1 resolution and summed with
    it through 1x1 lateral convolutions, yielding an ``out_dim``-channel map at
    4x the pillar-grid stride.
    """

    total_stride = 4

    def __init__(self, in_ch: int = 64, out_dim: int = 256) -> None:
        super().__init__()
        c1, c2 = 2 * in_ch, 4 * in_ch
        self.stem = nn.Sequential(
            _conv3x3(in_ch, in_ch, stride=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(in_ch, c1, stride=2), BasicBlock(c1, c1))
        self.stage2 = nn.Sequential(BasicBlock(c1, c2, stride=2), BasicBlock(c2, c2))
        self.lateral1 = nn.Conv2d(c1, out_dim, 1)
        self.lateral2 = nn.Conv2d(c2, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map a pseudo-image ``(N, in_ch, H, W)`` to ``(N, out_dim, H/4, W/4)``."""
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        up = F.interpolate(self.lateral2(f2), size=f1.shape[-2:], mode="nearest")
        return self.lateral1(f1) + up


class PillarEncoder(nn.Module):
    """Encode per-frame object points into a single feature vector per frame."""

    def __init__(self, config: PillarEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or PillarEncoderConfig()
        cfg = self.config
        self.point_net = nn.Sequential(
            nn.Linear(POINT_FEAT_DIM, cfg.point_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.backbone = PillarBackbone(cfg.point_feat_dim, cfg.out_dim)
        self.out_dim = cfg.out_dim

    @property
    def d_out(self) -> int:
        """Width of the produced per-frame embedding."""
        return self.out_dim

    def _scatter_pillars(self, points: Tensor, mask: Tensor) -> Tensor:
        """Build the ``(M, C, H, W)`` pillar pseudo-image for ``M`` frames.

        Args:
            points: ``(M, N, 4)`` ``(x, y, z, intensity)`` in the box frame.
            mask: ``(M, N)`` bool marking valid points.

        Returns:
            ``(M, point_feat_dim, grid_h, grid_w)`` max-pooled pillar features.
        """
        cfg = self.config
        m, n, _ = points.shape
        h, w = cfg.grid_h, cfg.grid_w
        x, y = points[..., 0], points[..., 1]

        col = torch.floor((x - cfg.x_range[0]) / cfg.pillar_size)
        row = torch.floor((y - cfg.y_range[0]) / cfg.pillar_size)
        in_range = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        valid = mask & in_range
        col = col.clamp(0, w - 1).long()
        row = row.clamp(0, h - 1).long()

        # Offsets from the pillar center.
        center_x = cfg.x_range[0] + (col.to(points.dtype) + 0.5) * cfg.pillar_size
        center_y = cfg.y_range[0] + (row.to(points.dtype) + 0.5) * cfg.pillar_size
        feats = torch.cat(
            [points, (x - center_x).unsqueeze(-1), (y - center_y).unsqueeze(-1)], dim=-1
        )
        feats = feats * valid.unsqueeze(-1).to(feats.dtype)
        feats = self.point_net(feats).reshape(m * n, cfg.point_feat_dim)

        # Flat pillar index per point; invalid points go to a trailing trash bin.
        frame = torch.arange(m, device=points.device).unsqueeze(1).expand(m, n)
        flat = (frame * (h * w) + row * w + col).reshape(-1)
        trash = m * h * w
        flat = torch.where(valid.reshape(-1), flat, torch.full_like(flat, trash))

        # ReLU features are >= 0, so amax against a zero grid is a masked max
        # and empty pillars stay exactly zero.
        grid = points.new_zeros((trash + 1, cfg.point_feat_dim))
        grid = grid.scatter_reduce(
            0, flat.unsqueeze(-1).expand(-1, cfg.point_feat_dim), feats, reduce="amax"
        )
        grid = grid[:trash].view(m, h, w, cfg.point_feat_dim)
        return grid.permute(0, 3, 1, 2).contiguous()

    def forward(self, points: Tensor, points_mask: Tensor) -> Tensor:
        """Encode points ``(B, T, N, 4)`` / mask ``(B, T, N)`` into ``(B, T, out_dim)``."""
        b, t = points.shape[:2]
        grid = self._scatter_pillars(
            points.reshape(b * t, *points.shape[2:]), points_mask.reshape(b * t, -1)
        )
        feat = self.backbone(grid)
        _, c, fh, fw = feat.shape

        # Read out at the grid cell containing the object-frame origin.
        cfg = self.config
        col0 = int((0.0 - cfg.x_range[0]) / cfg.pillar_size) // self.backbone.total_stride
        row0 = int((0.0 - cfg.y_range[0]) / cfg.pillar_size) // self.backbone.total_stride
        col0 = min(max(col0, 0), fw - 1)
        row0 = min(max(row0, 0), fh - 1)
        return feat[:, :, row0, col0].view(b, t, c)
