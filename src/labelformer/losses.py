"""Training losses for LabelFormer (Yang et al., CoRL 2023).

The paper's objective is ``L = L_reg + L_IoU`` where ``L_reg`` is a Smooth-L1
regression term over position, dimensions and heading (the latter in the
sin/cos-of-double-angle parameterization, weighted by lambda = 0.1) and
``L_IoU`` is a differentiable IoU loss over the per-frame axis-aligned
bounding boxes of the rotated predictions.

Boxes are ``(B, T, 5)`` float32 tensors of ``(x, y, yaw, l, w)``; ``frame_mask``
is ``(B, T)`` bool marking valid (non-padded) frames. Every term is averaged
over valid frames only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-6


def _masked_mean(per_frame: Tensor, frame_mask: Tensor) -> Tensor:
    """Mean of a (B, T) per-frame quantity over valid frames only."""
    mask = frame_mask.to(per_frame.dtype)
    return (per_frame * mask).sum() / mask.sum().clamp(min=1.0)


def regression_loss(
    pred_boxes: Tensor,
    gt_boxes: Tensor,
    frame_mask: Tensor,
    heading_weight: float = 0.1,
    beta: float = 1.0,
) -> dict[str, Tensor]:
    """Smooth-L1 regression loss on position, dimensions and heading.

    Args:
        pred_boxes: (B, T, 5) predicted ``(x, y, yaw, l, w)``.
        gt_boxes: (B, T, 5) ground-truth boxes.
        frame_mask: (B, T) bool, True for valid frames.
        heading_weight: lambda scaling the heading term (paper: 0.1).
        beta: Smooth-L1 transition point.

    Returns:
        Dict with scalar ``loss_pos``, ``loss_dim`` and ``loss_heading``.
    """
    pos = F.smooth_l1_loss(
        pred_boxes[..., 0:2], gt_boxes[..., 0:2], beta=beta, reduction="none"
    ).sum(-1)
    dim = F.smooth_l1_loss(
        pred_boxes[..., 3:5], gt_boxes[..., 3:5], beta=beta, reduction="none"
    ).sum(-1)

    # pi-periodic heading encoding: (sin 2*yaw, cos 2*yaw)
    two_pred, two_gt = 2.0 * pred_boxes[..., 2], 2.0 * gt_boxes[..., 2]
    heading = F.smooth_l1_loss(
        torch.sin(two_pred), torch.sin(two_gt), beta=beta, reduction="none"
    ) + F.smooth_l1_loss(
        torch.cos(two_pred), torch.cos(two_gt), beta=beta, reduction="none"
    )

    return {
        "loss_pos": _masked_mean(pos, frame_mask),
        "loss_dim": _masked_mean(dim, frame_mask),
        "loss_heading": heading_weight * _masked_mean(heading, frame_mask),
    }


def _aabb_half_extents(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Half-extents (hx, hy) of the axis-aligned bounds of rotated BEV boxes."""
    yaw, l, w = boxes[..., 2], boxes[..., 3], boxes[..., 4]
    c, s = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    return 0.5 * (l * c + w * s), 0.5 * (l * s + w * c)


def aabb_iou_loss(
    pred_boxes: Tensor, gt_boxes: Tensor, frame_mask: Tensor
) -> Tensor:
    """Differentiable ``1 - IoU`` between per-frame axis-aligned box bounds.

    The AABB of a rotated box has half-extents
    ``hx = (l/2)|cos yaw| + (w/2)|sin yaw|`` and
    ``hy = (l/2)|sin yaw| + (w/2)|cos yaw|``, which keeps the loss
    differentiable w.r.t. all five box parameters.
    """
    phx, phy = _aabb_half_extents(pred_boxes)
    ghx, ghy = _aabb_half_extents(gt_boxes)
    px, py = pred_boxes[..., 0], pred_boxes[..., 1]
    gx, gy = gt_boxes[..., 0], gt_boxes[..., 1]

    inter_x = (torch.min(px + phx, gx + ghx) - torch.max(px - phx, gx - ghx)).clamp(
        min=0.0
    )
    inter_y = (torch.min(py + phy, gy + ghy) - torch.max(py - phy, gy - ghy)).clamp(
        min=0.0
    )
    inter = inter_x * inter_y
    union = 4.0 * phx * phy + 4.0 * ghx * ghy - inter
    iou = inter / (union + _EPS)
    return _masked_mean(1.0 - iou, frame_mask)


def labelformer_loss(
    pred_boxes: Tensor,
    gt_boxes: Tensor,
    frame_mask: Tensor,
    heading_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Full LabelFormer objective ``L = L_reg + L_IoU``.

    Returns:
        Dict with ``loss_pos``, ``loss_dim``, ``loss_heading``, ``loss_iou``
        and their sum as ``loss_total``.
    """
    out = regression_loss(pred_boxes, gt_boxes, frame_mask, heading_weight=heading_weight)
    out["loss_iou"] = aabb_iou_loss(pred_boxes, gt_boxes, frame_mask)
    out["loss_total"] = (
        out["loss_pos"] + out["loss_dim"] + out["loss_heading"] + out["loss_iou"]
    )
    return out
