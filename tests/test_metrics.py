"""Tests for rotated-IoU metrics with hand-computed expectations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from labelformer.geometry import se2_from_xyt, transform_boxes_bev
from labelformer.metrics import (
    recall_at_thresholds,
    rotated_iou,
    summarize,
    track_mean_iou,
)


def test_identical_boxes_iou_is_one():
    boxes = np.array(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0],
            [3.5, -1.25, 0.7, 4.6, 1.9],
            [-10.0, 8.0, -2.1, 12.0, 3.0],
        ]
    )
    np.testing.assert_allclose(rotated_iou(boxes, boxes), 1.0, atol=1e-9)


def test_disjoint_boxes_iou_is_zero():
    a = np.array([[0.0, 0.0, 0.0, 2.0, 2.0]])
    b = np.array([[100.0, 100.0, 0.3, 2.0, 2.0]])
    np.testing.assert_allclose(rotated_iou(a, b), 0.0, atol=1e-12)


def test_touching_boxes_iou_is_zero():
    # Edge-to-edge contact: zero-area intersection.
    a = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    b = np.array([2.0, 0.0, 0.0, 2.0, 2.0])
    assert rotated_iou(a, b) == pytest.approx(0.0, abs=1e-12)


def test_axis_aligned_partial_overlap_analytic():
    # a spans [-1, 1] x [-1, 1]; b spans [0, 2] x [-1, 1].
    # intersection = 1 * 2 = 2, union = 4 + 4 - 2 = 6 -> IoU = 1/3.
    a = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    b = np.array([1.0, 0.0, 0.0, 2.0, 2.0])
    assert rotated_iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-9)

    # Offset in both axes: intersection = 1 * 1, union = 4 + 4 - 1 = 7.
    c = np.array([1.0, 1.0, 0.0, 2.0, 2.0])
    assert rotated_iou(a, c) == pytest.approx(1.0 / 7.0, abs=1e-9)


def test_nested_boxes_iou_is_area_ratio():
    outer = np.array([0.0, 0.0, 0.0, 4.0, 2.0])  # area 8
    inner = np.array([0.0, 0.0, 0.0, 2.0, 1.0])  # area 2, fully inside
    assert rotated_iou(outer, inner) == pytest.approx(2.0 / 8.0, abs=1e-9)


def test_unit_square_rotated_45_degrees_octagon_case():
    # Intersection of a unit square with its 45-degree rotation is a regular
    # octagon of area 2 * (sqrt(2) - 1) ~= 0.828427; union = 2 - that, so
    # IoU = 2(sqrt(2)-1) / (4 - 2*sqrt(2)) = sqrt(2)/2.
    a = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, math.pi / 4, 1.0, 1.0])
    octagon = 2.0 * (math.sqrt(2.0) - 1.0)
    expected = octagon / (2.0 - octagon)
    assert expected == pytest.approx(math.sqrt(2.0) / 2.0, abs=1e-12)
    assert rotated_iou(a, b) == pytest.approx(expected, abs=1e-6)


def test_rotation_invariance():
    rng = np.random.default_rng(0)
    a = np.array([1.0, -2.0, 0.3, 4.0, 1.8])
    b = np.array([1.6, -1.7, 0.9, 3.6, 2.2])
    base = rotated_iou(a, b)
    assert 0.0 < base < 1.0
    for theta in rng.uniform(-np.pi, np.pi, size=8):
        T = se2_from_xyt(0.0, 0.0, theta)
        ra = transform_boxes_bev(T, a)
        rb = transform_boxes_bev(T, b)
        assert rotated_iou(ra, rb) == pytest.approx(base, abs=1e-9)


def test_translation_invariance():
    a = np.array([0.0, 0.0, 0.4, 3.0, 1.5])
    b = np.array([0.5, 0.2, 0.1, 3.0, 1.5])
    base = rotated_iou(a, b)
    shift = np.array([37.0, -12.0, 0.0, 0.0, 0.0])
    assert rotated_iou(a + shift, b + shift) == pytest.approx(base, abs=1e-9)


def test_pi_flip_of_heading_gives_same_box():
    a = np.array([1.0, 2.0, 0.35, 4.0, 2.0])
    b = a.copy()
    b[2] += np.pi
    assert rotated_iou(a, b) == pytest.approx(1.0, abs=1e-9)


def test_rotated_iou_shapes_and_range():
    rng = np.random.default_rng(7)
    a = np.stack(
        [
            rng.uniform(-3, 3, (2, 5)),
            rng.uniform(-3, 3, (2, 5)),
        ],
        axis=0,
    )
    a[..., 3:] = rng.uniform(1.0, 4.0, a[..., 3:].shape)
    b = a + rng.normal(0.0, 0.2, a.shape)
    b[..., 3:] = np.abs(b[..., 3:])
    iou = rotated_iou(a, b)
    assert iou.shape == a.shape[:-1]
    assert np.all(iou >= 0.0) and np.all(iou <= 1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rotated_iou(np.zeros((2, 5)), np.zeros((3, 5)))


def _toy_batch():
    """B=2, T=3 with the last frame of track 1 padded with garbage."""
    gt = np.array(
        [
            [
                [0.0, 0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0],
            ],
        ]
    )
    pred = gt.copy()
    pred[0, 1, 0] = 1.0  # IoU 1/3
    pred[1, 0, 0] = 1.0  # IoU 1/3
    pred[1, 2] = [500.0, 500.0, 1.0, 3.0, 3.0]  # padded frame: IoU 0
    mask = np.array([[True, True, True], [True, True, False]])
    return pred, gt, mask


def test_track_mean_iou_respects_mask():
    pred, gt, mask = _toy_batch()
    tmi = track_mean_iou(pred, gt, mask)
    assert tmi.shape == (2,)
    # track 0: (1 + 1/3 + 1) / 3 ; track 1: (1/3 + 1) / 2, padded frame excluded
    assert tmi[0] == pytest.approx((1.0 + 1.0 / 3.0 + 1.0) / 3.0, abs=1e-9)
    assert tmi[1] == pytest.approx((1.0 / 3.0 + 1.0) / 2.0, abs=1e-9)


def test_metrics_ignore_padded_frame_contents():
    pred, gt, mask = _toy_batch()
    before = summarize(pred, gt, mask)
    pred2 = pred.copy()
    pred2[1, 2] = [-99.0, 42.0, -0.7, 9.0, 5.0]
    after = summarize(pred2, gt, mask)
    assert before == after


def test_recall_at_thresholds_respects_mask():
    pred, gt, mask = _toy_batch()
    rec = recall_at_thresholds(pred, gt, mask)
    # pooled valid IoUs: [1, 1/3, 1, 1/3, 1] -> 3 of 5 above 0.5.
    assert set(rec) == {0.5, 0.6, 0.7, 0.8}
    for thr in (0.5, 0.6, 0.7, 0.8):
        assert rec[thr] == pytest.approx(3.0 / 5.0, abs=1e-9)

    custom = recall_at_thresholds(pred, gt, mask, thresholds=(0.3, 0.9))
    assert custom[0.3] == pytest.approx(1.0)
    assert custom[0.9] == pytest.approx(3.0 / 5.0)


def test_summarize_keys_and_values():
    pred, gt, mask = _toy_batch()
    out = summarize(pred, gt, mask)
    assert set(out) == {
        "mean_iou",
        "track_mean_iou",
        "recall@0.5",
        "recall@0.6",
        "recall@0.7",
        "recall@0.8",
    }
    assert out["mean_iou"] == pytest.approx((3.0 + 2.0 / 3.0) / 5.0, abs=1e-9)
    expected_track = 0.5 * (
        (1.0 + 1.0 / 3.0 + 1.0) / 3.0 + (1.0 / 3.0 + 1.0) / 2.0
    )
    assert out["track_mean_iou"] == pytest.approx(expected_track, abs=1e-9)
    assert all(isinstance(v, float) for v in out.values())


def test_summarize_all_frames_masked_out():
    pred, gt, _ = _toy_batch()
    mask = np.zeros((2, 3), dtype=bool)
    out = summarize(pred, gt, mask)
    assert out["mean_iou"] == 0.0
    assert out["track_mean_iou"] == 0.0
    assert out["recall@0.5"] == 0.0
