"""Tests for the LabelFormer model.

All tests use a deliberately small configuration (tiny pillar grid, narrow
transformer) so the whole file runs in a couple of seconds on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from labelformer.model import (
    LabelFormer,
    LabelFormerConfig,
    PillarEncoderConfig,
    alibi_slopes,
    build_alibi_bias,
)

D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 2


def small_config(dropout: float = 0.0) -> LabelFormerConfig:
    """A fast CPU-sized model configuration."""
    return LabelFormerConfig(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=2 * D_MODEL,
        dropout=dropout,
        pillar=PillarEncoderConfig(
            x_range=(-3.0, 3.0),
            y_range=(-1.5, 1.5),
            pillar_size=0.25,
            point_feat_dim=16,
            out_dim=D_MODEL,
        ),
    )


def make_batch(
    b: int = 2,
    t: int = 6,
    n: int = 50,
    valid_lengths: list[int] | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Random batch with ragged trajectory lengths and some invalid points."""
    g = torch.Generator().manual_seed(seed)
    lengths = valid_lengths if valid_lengths is not None else [t] * b

    boxes = torch.zeros(b, t, 5)
    boxes[..., 0] = torch.randn(b, t, generator=g) * 2.0
    boxes[..., 1] = torch.randn(b, t, generator=g) * 0.5
    boxes[..., 2] = (torch.rand(b, t, generator=g) - 0.5) * 2.0
    boxes[..., 3] = 4.0 + torch.rand(b, t, generator=g)
    boxes[..., 4] = 1.8 + torch.rand(b, t, generator=g) * 0.2

    points = torch.zeros(b, t, n, 4)
    points[..., 0] = (torch.rand(b, t, n, generator=g) - 0.5) * 5.0
    points[..., 1] = (torch.rand(b, t, n, generator=g) - 0.5) * 2.5
    points[..., 2] = torch.randn(b, t, n, generator=g) * 0.5
    points[..., 3] = torch.rand(b, t, n, generator=g)

    points_mask = torch.rand(b, t, n, generator=g) > 0.2
    frame_mask = torch.zeros(b, t, dtype=torch.bool)
    for i, length in enumerate(lengths):
        frame_mask[i, :length] = True
    points_mask &= frame_mask.unsqueeze(-1)

    return {
        "boxes_init": boxes,
        "points": points,
        "points_mask": points_mask,
        "frame_mask": frame_mask,
    }


@pytest.fixture
def model() -> LabelFormer:
    """Deterministic small model in eval mode."""
    torch.manual_seed(0)
    m = LabelFormer(small_config())
    m.eval()
    return m


def test_forward_shapes_and_finiteness(model: LabelFormer) -> None:
    b, t, n = 2, 6, 50
    batch = make_batch(b, t, n, valid_lengths=[6, 3])
    out = model(batch)

    assert out["boxes_refined"].shape == (b, t, 5)
    assert out["pose_residual"].shape == (b, t, 3)
    assert out["size_residual"].shape == (b, 2)

    valid = batch["frame_mask"]
    assert torch.isfinite(out["boxes_refined"][valid]).all()
    assert torch.isfinite(out["pose_residual"][valid]).all()
    assert torch.isfinite(out["size_residual"]).all()


def test_identity_refinement_at_init(model: LabelFormer) -> None:
    """Zero-initialized heads make the model an identity refiner."""
    batch = make_batch(2, 6, 50, valid_lengths=[6, 4], seed=1)
    out = model(batch)

    assert torch.allclose(out["pose_residual"], torch.zeros_like(out["pose_residual"]))
    assert torch.allclose(out["size_residual"], torch.zeros_like(out["size_residual"]))

    refined, init, mask = out["boxes_refined"], batch["boxes_init"], batch["frame_mask"]
    assert torch.allclose(refined[..., :3], init[..., :3], atol=1e-6)

    w = mask.unsqueeze(-1).float()
    mean_size = (init[..., 3:5] * w).sum(1) / w.sum(1)
    expected = mean_size.unsqueeze(1).expand_as(refined[..., 3:5])
    assert torch.allclose(refined[..., 3:5], expected, atol=1e-6)


def test_padding_invariance(model: LabelFormer) -> None:
    """Randomizing padded frames/points leaves valid outputs untouched."""
    # Non-zero heads, otherwise the outputs are trivially the identity.
    torch.manual_seed(3)
    for head in (model.pose_head, model.size_head):
        torch.nn.init.normal_(head.out.weight, std=0.05)
        torch.nn.init.normal_(head.out.bias, std=0.05)

    batch = make_batch(2, 7, 50, valid_lengths=[7, 3], seed=2)
    ref = model(batch)

    g = torch.Generator().manual_seed(11)
    frame_mask = batch["frame_mask"]
    pad_frames = ~frame_mask
    pad_points = ~batch["points_mask"]

    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["boxes_init"][pad_frames] = (
        torch.randn(int(pad_frames.sum()), 5, generator=g) * 10.0
    )
    noise = torch.randn(batch["points"].shape, generator=g) * 10.0
    perturbed["points"] = torch.where(
        pad_points.unsqueeze(-1), noise, perturbed["points"]
    )
    out = model(perturbed)

    for key in ("boxes_refined", "pose_residual"):
        assert torch.allclose(out[key][frame_mask], ref[key][frame_mask], atol=1e-5)
    assert torch.allclose(out["size_residual"], ref["size_residual"], atol=1e-5)


def test_points_influence_output(model: LabelFormer) -> None:
    """The pillar path is actually wired into the prediction."""
    torch.manual_seed(4)
    for head in (model.pose_head, model.size_head):
        torch.nn.init.normal_(head.out.weight, std=0.1)
        torch.nn.init.normal_(head.out.bias, std=0.1)

    batch = make_batch(2, 5, 50, seed=5)
    ref = model(batch)

    moved = {k: v.clone() for k, v in batch.items()}
    moved["points"][..., 0] += 1.0  # shift all points along the box heading
    out = model(moved)

    assert not torch.allclose(
        out["boxes_refined"], ref["boxes_refined"], atol=1e-5
    ), "moving the LiDAR points did not change the refined boxes"

    # The per-frame point embeddings themselves must differ.
    p_ref = model.pillar_encoder(batch["points"], batch["points_mask"])
    p_moved = model.pillar_encoder(moved["points"], moved["points_mask"])
    assert not torch.allclose(p_ref, p_moved, atol=1e-6)

    # ... and gradients must reach the points.
    pts = batch["points"].clone().requires_grad_(True)
    grad_batch = dict(batch, points=pts)
    model(grad_batch)["boxes_refined"].square().mean().backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum() > 0


def test_empty_frame_is_finite(model: LabelFormer) -> None:
    """A valid frame with no in-range points still yields finite features."""
    batch = make_batch(1, 4, 50, seed=6)
    batch["points_mask"][0, 1] = False  # frame 1 has zero valid points
    batch["points"][0, 2] = 100.0  # frame 2's points are all out of range
    out = model(batch)
    assert torch.isfinite(out["boxes_refined"]).all()

    p = model.pillar_encoder(batch["points"], batch["points_mask"])
    assert torch.isfinite(p).all()


def test_alibi_bias() -> None:
    """Slopes follow 2^(-8(h+1)/H) and the bias is symmetric in |i - j|."""
    t, heads = 5, 4
    slopes = alibi_slopes(heads)
    expected_slopes = torch.tensor(
        [2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)]
    )
    assert torch.allclose(slopes, expected_slopes)

    bias = build_alibi_bias(t, heads)
    assert bias.shape == (heads, t, t)
    for h in range(heads):
        for i in range(t):
            for j in range(t):
                assert bias[h, i, j] == pytest.approx(
                    -expected_slopes[h].item() * abs(i - j), abs=1e-6
                )
    assert torch.allclose(bias, bias.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(bias, dim1=-2, dim2=-1), torch.zeros(heads, t))


def test_gradient_flow() -> None:
    """A scalar loss on the refined boxes reaches both encoders."""
    torch.manual_seed(7)
    model = LabelFormer(small_config(dropout=0.1))
    model.train()
    # Break the head zero-init so gradients are non-trivial.
    for head in (model.pose_head, model.size_head):
        torch.nn.init.normal_(head.out.weight, std=0.1)

    batch = make_batch(2, 6, 50, valid_lengths=[6, 4], seed=8)
    out = model(batch)
    loss = out["boxes_refined"][batch["frame_mask"]].square().mean()
    loss.backward()

    checked = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
        checked += 1
    assert checked > 0

    def grad_norm(module: torch.nn.Module) -> float:
        return sum(
            p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None
        )

    assert grad_norm(model.box_encoder) > 0
    assert grad_norm(model.pillar_encoder.point_net) > 0
    assert grad_norm(model.pillar_encoder.backbone) > 0
    assert grad_norm(model.transformer) > 0


def test_yaw_is_wrapped() -> None:
    """Refined headings stay in (-pi, pi] even for large residuals."""
    torch.manual_seed(9)
    model = LabelFormer(small_config())
    model.eval()
    torch.nn.init.constant_(model.pose_head.out.bias, 5.0)

    batch = make_batch(2, 4, 50, seed=10)
    yaw = model(batch)["boxes_refined"][..., 2]
    assert (yaw > -math.pi - 1e-6).all() and (yaw <= math.pi + 1e-6).all()
