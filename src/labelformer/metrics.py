"""Evaluation metrics for LabelFormer (numpy, non-differentiable).

Exact rotated-box BEV IoU via Sutherland-Hodgman convex-polygon clipping plus
the shoelace formula, and the trajectory-level aggregates reported in the
paper: per-track mean IoU (S^k) and recall at IoU thresholds.

Boxes follow the project convention ``(x, y, yaw, l, w)``; ``frame_mask`` marks
valid (non-padded) frames.
"""

from __future__ import annotations

import numpy as np

from .geometry import box_corners_bev

_EPS = 1e-12


def _polygon_area(poly: np.ndarray) -> float:
    """Absolute shoelace area of a simple polygon (N, 2); 0 for N < 3."""
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of ``subject`` against convex CCW ``clip``."""
    output = subject
    n = clip.shape[0]
    for i in range(n):
        if output.shape[0] == 0:
            return output
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a
        # Signed distance to the (CCW) edge line: >= 0 means inside.
        dist = edge[0] * (output[:, 1] - a[1]) - edge[1] * (output[:, 0] - a[0])
        inside = dist >= 0.0
        new_pts: list[np.ndarray] = []
        m = output.shape[0]
        for j in range(m):
            k = (j + 1) % m
            if inside[j]:
                new_pts.append(output[j])
            if inside[j] != inside[k]:
                t = dist[j] / (dist[j] - dist[k])
                new_pts.append(output[j] + t * (output[k] - output[j]))
        output = np.asarray(new_pts, dtype=np.float64).reshape(-1, 2)
    return output


def _iou_single(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Rotated IoU of two convex quads given as (4, 2) CCW corner arrays."""
    area_a = _polygon_area(corners_a)
    area_b = _polygon_area(corners_b)
    union = area_a + area_b
    if union <= _EPS:
        return 0.0
    inter = _polygon_area(_clip_polygon(corners_a, corners_b))
    union -= inter
    if union <= _EPS:
        return 0.0
    return float(min(max(inter / union, 0.0), 1.0))


def rotated_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Elementwise exact rotated-box BEV IoU.

    Args:
        boxes_a, boxes_b: (..., 5) arrays of matching shape.

    Returns:
        (...,) float64 array of IoU values in [0, 1].
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if boxes_a.shape != boxes_b.shape:
        raise ValueError(f"shape mismatch: {boxes_a.shape} vs {boxes_b.shape}")
    if boxes_a.shape[-1] != 5:
        raise ValueError(f"expected trailing dim 5, got {boxes_a.shape[-1]}")

    lead = boxes_a.shape[:-1]
    ca = box_corners_bev(boxes_a).reshape(-1, 4, 2)
    cb = box_corners_bev(boxes_b).reshape(-1, 4, 2)
    out = np.array([_iou_single(ca[i], cb[i]) for i in range(ca.shape[0])])
    return out.reshape(lead)


def track_mean_iou(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, frame_mask: np.ndarray
) -> np.ndarray:
    """Per-track mean rotated IoU over valid frames (paper metric S^k), shape (B,)."""
    iou = rotated_iou(pred_boxes, gt_boxes)
    mask = np.asarray(frame_mask, dtype=bool)
    counts = mask.sum(axis=-1)
    sums = (iou * mask).sum(axis=-1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def recall_at_thresholds(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    frame_mask: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
) -> dict[float, float]:
    """Fraction of valid frames (pooled over all tracks) with IoU >= threshold."""
    iou = rotated_iou(pred_boxes, gt_boxes)
    mask = np.asarray(frame_mask, dtype=bool)
    valid = iou[mask]
    n = valid.size
    return {float(t): (float((valid >= t).mean()) if n else 0.0) for t in thresholds}


def summarize(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, frame_mask: np.ndarray
) -> dict[str, float]:
    """Frame-pooled mean IoU, mean per-track IoU and recalls in one dict."""
    iou = rotated_iou(pred_boxes, gt_boxes)
    mask = np.asarray(frame_mask, dtype=bool)
    valid = iou[mask]
    out: dict[str, float] = {
        "mean_iou": float(valid.mean()) if valid.size else 0.0,
        "track_mean_iou": float(track_mean_iou(pred_boxes, gt_boxes, mask).mean()),
    }
    for thr, rec in recall_at_thresholds(pred_boxes, gt_boxes, mask).items():
        out[f"recall@{thr}"] = rec
    return out
