"""Tests for the LabelFormer training losses."""

from __future__ import annotations

import math

import pytest
import torch

from labelformer.losses import aabb_iou_loss, labelformer_loss, regression_loss

COMPONENTS = ("loss_pos", "loss_dim", "loss_heading", "loss_iou")


def _gt_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """B=2, T=4 ground truth with the last two frames of track 1 padded."""
    gt = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 4.0, 2.0],
                [1.0, 0.5, 0.1, 4.0, 2.0],
                [2.0, 1.0, 0.2, 4.0, 2.0],
                [3.0, 1.5, 0.3, 4.0, 2.0],
            ],
            [
                [-5.0, 2.0, 1.0, 3.0, 1.5],
                [-4.0, 2.5, 1.1, 3.0, 1.5],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]], dtype=torch.bool
    )
    return gt, mask


def test_perfect_prediction_is_zero():
    gt, mask = _gt_batch()
    out = labelformer_loss(gt.clone(), gt, mask)
    assert out["loss_pos"].item() == pytest.approx(0.0, abs=1e-7)
    assert out["loss_dim"].item() == pytest.approx(0.0, abs=1e-7)
    assert out["loss_heading"].item() == pytest.approx(0.0, abs=1e-7)
    # eps in the IoU denominator leaves a tiny residual.
    assert out["loss_iou"].item() == pytest.approx(0.0, abs=1e-5)
    assert out["loss_total"].item() == pytest.approx(0.0, abs=1e-5)


def test_heading_loss_is_pi_periodic():
    gt, mask = _gt_batch()
    pred = gt.clone()
    pred[..., 2] += math.pi
    out = regression_loss(pred, gt, mask)
    assert out["loss_heading"].item() == pytest.approx(0.0, abs=1e-6)
    # The pi-flipped box also has the same AABB, so L_IoU stays ~0 too.
    assert aabb_iou_loss(pred, gt, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_heading_loss_penalizes_perpendicular_heading():
    gt = torch.tensor([[[0.0, 0.0, 0.0, 4.0, 2.0]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    pred = gt.clone()
    pred[..., 2] = math.pi / 2
    # sin(2*yaw): 0 -> 0 (delta 0); cos(2*yaw): 1 -> -1 (delta 2).
    # Smooth L1 with beta=1 at |d|=2 is |d| - 0.5 = 1.5, scaled by lambda=0.1.
    out = regression_loss(pred, gt, mask)
    assert out["loss_heading"].item() == pytest.approx(0.1 * 1.5, abs=1e-6)


def test_regression_components_are_isolated():
    gt = torch.tensor([[[0.0, 0.0, 0.0, 4.0, 2.0]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)

    pred = gt.clone()
    pred[..., 0] = 3.0  # |dx| = 3 -> smooth L1 = 2.5; dy = 0
    out = regression_loss(pred, gt, mask)
    assert out["loss_pos"].item() == pytest.approx(2.5, abs=1e-6)
    assert out["loss_dim"].item() == pytest.approx(0.0, abs=1e-7)
    assert out["loss_heading"].item() == pytest.approx(0.0, abs=1e-7)

    pred = gt.clone()
    pred[..., 3] = 4.5  # |dl| = 0.5 -> quadratic branch 0.5 * 0.25 = 0.125
    out = regression_loss(pred, gt, mask)
    assert out["loss_dim"].item() == pytest.approx(0.125, abs=1e-6)
    assert out["loss_pos"].item() == pytest.approx(0.0, abs=1e-7)


def test_heading_weight_scales_only_heading():
    gt, mask = _gt_batch()
    pred = gt.clone()
    pred[..., 2] += 0.4
    pred[..., 0] += 0.3
    a = regression_loss(pred, gt, mask, heading_weight=0.1)
    b = regression_loss(pred, gt, mask, heading_weight=0.2)
    assert b["loss_heading"].item() == pytest.approx(2.0 * a["loss_heading"].item())
    assert b["loss_pos"].item() == pytest.approx(a["loss_pos"].item())
    assert b["loss_dim"].item() == pytest.approx(a["loss_dim"].item())


def test_losses_positive_and_monotone_toward_gt():
    gt, mask = _gt_batch()
    delta = torch.zeros_like(gt)
    delta[..., 0] = 0.9
    delta[..., 1] = -0.6
    delta[..., 2] = 0.35
    delta[..., 3] = 0.7
    delta[..., 4] = -0.4

    prev = None
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.1):
        out = labelformer_loss(gt + alpha * delta, gt, mask)
        for key in COMPONENTS:
            assert out[key].item() > 0.0
        assert out["loss_total"].item() > 0.0
        if prev is not None:
            for key in (*COMPONENTS, "loss_total"):
                assert out[key].item() < prev[key], key
        prev = {k: out[k].item() for k in (*COMPONENTS, "loss_total")}


def test_total_is_sum_of_components():
    gt, mask = _gt_batch()
    pred = gt + 0.3
    out = labelformer_loss(pred, gt, mask)
    assert set(out) == {*COMPONENTS, "loss_total"}
    assert all(v.ndim == 0 for v in out.values())
    total = sum(out[k] for k in COMPONENTS)
    assert out["loss_total"].item() == pytest.approx(total.item(), abs=1e-6)


def test_aabb_iou_loss_analytic_axis_aligned():
    # Both yaw=0, 2x2 boxes offset by 1 in x: inter = 2, union = 6 -> IoU = 1/3.
    gt = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0]]], dtype=torch.float32)
    pred = torch.tensor([[[1.0, 0.0, 0.0, 2.0, 2.0]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert aabb_iou_loss(pred, gt, mask).item() == pytest.approx(2.0 / 3.0, abs=1e-5)

    # Disjoint -> IoU 0 -> loss 1.
    far = torch.tensor([[[50.0, 50.0, 0.0, 2.0, 2.0]]], dtype=torch.float32)
    assert aabb_iou_loss(far, gt, mask).item() == pytest.approx(1.0, abs=1e-5)


def test_aabb_iou_loss_uses_rotated_bounds():
    # A unit square rotated 45 degrees has an AABB of half-extent sqrt(2)/2,
    # i.e. area 2, so IoU against the axis-aligned unit square is 1/2.
    gt = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0]]], dtype=torch.float32)
    pred = torch.tensor([[[0.0, 0.0, math.pi / 4, 1.0, 1.0]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert aabb_iou_loss(pred, gt, mask).item() == pytest.approx(0.5, abs=1e-5)


def test_aabb_iou_loss_gradient_flows():
    gt, mask = _gt_batch()
    pred = (gt + 0.4).detach().requires_grad_(True)
    loss = aabb_iou_loss(pred, gt, mask)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[mask].abs().sum().item() > 0.0


def test_masked_frames_do_not_affect_losses():
    gt, mask = _gt_batch()
    pred_a = gt + 0.25
    pred_b = pred_a.clone()
    pred_b[1, 2] = torch.tensor([123.0, -77.0, 2.5, 9.0, 6.0])
    pred_b[1, 3] = torch.tensor([-4.0, 31.0, -1.7, 0.5, 0.25])

    out_a = labelformer_loss(pred_a, gt, mask)
    out_b = labelformer_loss(pred_b, gt, mask)
    for key in (*COMPONENTS, "loss_total"):
        assert out_a[key].item() == pytest.approx(out_b[key].item(), abs=1e-6), key


def test_masked_frames_receive_zero_gradient():
    gt, mask = _gt_batch()
    pred = (gt + 0.25).detach().requires_grad_(True)
    labelformer_loss(pred, gt, mask)["loss_total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.all(pred.grad[~mask] == 0.0)
    assert pred.grad[mask].abs().sum().item() > 0.0


def test_all_frames_masked_out_is_finite():
    gt, mask = _gt_batch()
    out = labelformer_loss(gt + 1.0, gt, torch.zeros_like(mask))
    for key in (*COMPONENTS, "loss_total"):
        assert out[key].item() == pytest.approx(0.0, abs=1e-7), key


def test_dtype_and_variable_length_batch():
    gt, mask = _gt_batch()
    out = labelformer_loss(gt + 0.2, gt, mask)
    for v in out.values():
        assert v.dtype == torch.float32
        assert v.shape == ()
    # Track lengths differ (4 vs 2) via the mask; longer batch works too.
    long_gt = gt.repeat(1, 3, 1)
    long_mask = mask.repeat(1, 3)
    out2 = labelformer_loss(long_gt + 0.2, long_gt, long_mask)
    assert out2["loss_total"].item() == pytest.approx(
        out["loss_total"].item(), abs=1e-5
    )
