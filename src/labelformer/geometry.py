"""Shared 2D (BEV) geometry utilities for LabelFormer.

Conventions used across the whole project:
- A BEV box is a 5-vector ``(x, y, yaw, l, w)``: center, heading (radians,
  CCW from +x), length along heading, width perpendicular.
- SE(2) poses are 3x3 homogeneous matrices; all functions broadcast over
  leading batch dimensions.
- All functions here are numpy; model/loss code re-derives the few pieces it
  needs in torch so gradients flow.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi]."""
    return -((-np.asarray(a) + np.pi) % (2 * np.pi) - np.pi)


def se2_from_xyt(x, y, theta) -> np.ndarray:
    """Build SE(2) matrices (..., 3, 3) from x, y, theta arrays (broadcastable)."""
    x, y, theta = np.broadcast_arrays(np.asarray(x, np.float64), y, theta)
    c, s = np.cos(theta), np.sin(theta)
    T = np.zeros((*x.shape, 3, 3))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2] = c, -s, x
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2] = s, c, y
    T[..., 2, 2] = 1.0
    return T


def se2_inv(T: np.ndarray) -> np.ndarray:
    """Invert SE(2) matrices (..., 3, 3)."""
    R = T[..., :2, :2]
    t = T[..., :2, 2:]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :2, :2] = Rt
    out[..., :2, 2:] = -Rt @ t
    out[..., 2, 2] = 1.0
    return out


def transform_points_2d(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(2) ``T`` (3, 3) to points (..., 2) -> (..., 2)."""
    return pts @ T[:2, :2].T + T[:2, 2]


def transform_boxes_bev(T: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Apply a single SE(2) ``T`` (3, 3) to BEV boxes (..., 5).

    Rotates/translates centers and offsets yaw; sizes unchanged.
    """
    out = np.array(boxes, dtype=np.float64, copy=True)
    out[..., :2] = transform_points_2d(T, boxes[..., :2])
    dtheta = np.arctan2(T[1, 0], T[0, 0])
    out[..., 2] = wrap_angle(boxes[..., 2] + dtheta)
    return out


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """Corners (..., 4, 2) of BEV boxes (..., 5), CCW order starting front-left."""
    boxes = np.asarray(boxes)
    x, y, yaw, l, w = (boxes[..., i] for i in range(5))
    # corner offsets in box frame
    dx = np.stack([l / 2, -l / 2, -l / 2, l / 2], axis=-1)
    dy = np.stack([w / 2, w / 2, -w / 2, -w / 2], axis=-1)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    cx = x[..., None] + c * dx - s * dy
    cy = y[..., None] + s * dx + c * dy
    return np.stack([cx, cy], axis=-1)


def points_in_box_mask(
    points_xy: np.ndarray, box: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Boolean mask of 2D points (N, 2) inside BEV ``box`` (5,) scaled by ``scale``."""
    x, y, yaw, l, w = box
    c, s = np.cos(yaw), np.sin(yaw)
    px = points_xy[:, 0] - x
    py = points_xy[:, 1] - y
    bx = c * px + s * py
    by = -s * px + c * py
    return (np.abs(bx) <= scale * l / 2) & (np.abs(by) <= scale * w / 2)


def canonicalize_headings(headings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote heading flip heuristic (paper sec. per-frame encoder).

    Noisy per-frame headings can be flipped by pi. Estimate the trajectory's
    dominant axis via the circular mean of doubled angles, orient it with the
    majority of the input headings, and flip the minority by pi.

    Returns (canonical_headings, flipped_mask).
    """
    headings = np.asarray(headings, dtype=np.float64)
    axis = 0.5 * np.arctan2(np.sin(2 * headings).mean(), np.cos(2 * headings).mean())
    aligned = np.abs(wrap_angle(headings - axis)) <= np.pi / 2
    if aligned.sum() < headings.size - aligned.sum():
        axis = wrap_angle(axis + np.pi)
        aligned = ~aligned
    out = wrap_angle(np.where(aligned, headings, headings + np.pi))
    return out, ~aligned
