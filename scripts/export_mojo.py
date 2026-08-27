#!/usr/bin/env python3
"""Export LabelFormer weights and test samples for the Mojo inferencer.

Writes the simple LFT1 container consumed by labelrefinery/LabelFormer.mojo:
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)

All BatchNorms are folded into their preceding convolutions, so the Mojo side
only needs plain convs/linears with biases. Weight names use a flat scheme
(see WEIGHT_NAMES below), decoupled from the PyTorch module tree.

Usage (from repo root):
    uv run python scripts/export_mojo.py --checkpoint runs/smoke/best.pt \
        --out export --export-samples 3
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from labelformer.data.dataset import PerturbConfig, TrajectoryDataset, collate_tracks
from labelformer.train import build_model


def write_lft(path: Path, tensors: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as f:
        f.write(b"LFT1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            if arr.ndim:
                f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.tobytes())


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Fold eval-mode BN into the conv: returns (weight, bias)."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.view(-1, 1, 1, 1)
    b = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
    b = (b - bn.running_mean) * scale + bn.bias
    return w.detach().numpy(), b.detach().numpy()


def lin(layer: nn.Linear) -> tuple[np.ndarray, np.ndarray]:
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


def conv(layer: nn.Conv2d) -> tuple[np.ndarray, np.ndarray]:
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


def put(out: dict, name: str, wb: tuple[np.ndarray, np.ndarray]) -> None:
    out[f"{name}.w"], out[f"{name}.b"] = wb


def export_weights(model, cfg: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    m = cfg["model"]
    p = m["pillar"]
    # Readout cell computed here with the exact float64/int semantics of
    # PillarEncoder.forward — float32 reproduction differs (47.999... vs 48).
    col0 = int((0.0 - p["x_range"][0]) / p["pillar_size"]) // 4
    row0 = int((0.0 - p["y_range"][0]) / p["pillar_size"]) // 4
    out["__config__"] = np.array(
        [
            m["d_model"], m["nhead"], m["num_layers"], m["dim_feedforward"],
            p["x_range"][0], p["x_range"][1], p["y_range"][0], p["y_range"][1],
            p["pillar_size"], p["point_feat_dim"], p["out_dim"], row0, col0,
        ],
        dtype=np.float32,
    )

    put(out, "box.0", lin(model.box_encoder.mlp[0]))
    put(out, "box.2", lin(model.box_encoder.mlp[2]))
    put(out, "pointnet", lin(model.pillar_encoder.point_net[0]))

    bb = model.pillar_encoder.backbone
    put(out, "stem", fold_bn(bb.stem[0], bb.stem[1]))
    for si, stage in ((1, bb.stage1), (2, bb.stage2)):
        for bi, block in enumerate(stage):
            put(out, f"s{si}b{bi}.conv1", fold_bn(block.conv1, block.bn1))
            put(out, f"s{si}b{bi}.conv2", fold_bn(block.conv2, block.bn2))
            if block.downsample is not None:
                put(out, f"s{si}b{bi}.down", fold_bn(block.downsample[0], block.downsample[1]))
    put(out, "lat1", conv(bb.lateral1))
    put(out, "lat2", conv(bb.lateral2))

    put(out, "fusion", lin(model.point_fusion))
    for i, layer in enumerate(model.transformer.layers):
        put(out, f"layer{i}.norm1", (layer.norm1.weight.detach().numpy(), layer.norm1.bias.detach().numpy()))
        put(out, f"layer{i}.qkv", lin(layer.attn.qkv))
        put(out, f"layer{i}.out", lin(layer.attn.out_proj))
        put(out, f"layer{i}.norm2", (layer.norm2.weight.detach().numpy(), layer.norm2.bias.detach().numpy()))
        put(out, f"layer{i}.ffn1", lin(layer.ffn[0]))
        put(out, f"layer{i}.ffn2", lin(layer.ffn[3]))
    fn = model.transformer.norm
    put(out, "final_norm", (fn.weight.detach().numpy(), fn.bias.detach().numpy()))

    put(out, "pose.0", lin(model.pose_head.mlp[0]))
    put(out, "pose.2", lin(model.pose_head.mlp[2]))
    put(out, "size.0", lin(model.size_head.mlp[0]))
    put(out, "size.2", lin(model.size_head.mlp[2]))
    return out


@torch.no_grad()
def export_sample(model, batch: dict) -> dict[str, np.ndarray]:
    """Run one B=1 batch and capture inputs, intermediates and outputs."""
    tokens = model.encode_frames(batch)
    hidden = model.transformer(tokens, batch["frame_mask"])
    out = model(batch)

    def sq(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).detach().numpy().astype(np.float32)

    return {
        "boxes_init": sq(batch["boxes_init"]),
        "points": sq(batch["points"]),
        "points_mask": sq(batch["points_mask"].float()),
        "frame_mask": sq(batch["frame_mask"].float()),
        "boxes_gt": sq(batch["boxes_gt"]),
        "tokens": sq(tokens),
        "hidden": sq(hidden),
        "pose_residual": sq(out["pose_residual"]),
        "size_residual": out["size_residual"].squeeze(0).detach().numpy(),
        "boxes_refined": sq(out["boxes_refined"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/smoke/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--export-samples", type=int, default=3)
    ap.add_argument("--sample-indices", type=int, nargs="*", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    weights = export_weights(model, cfg)
    write_lft(args.out / "weights.lft", weights)
    n_params = sum(int(np.prod(a.shape)) for a in weights.values())
    print(f"weights.lft: {len(weights)} tensors, {n_params / 1e6:.2f}M values")

    dcfg = cfg["data"]
    ds = TrajectoryDataset(
        root=Path(dcfg["root"]),
        split="val",
        max_frames=dcfg["max_frames"],
        max_points_per_frame=dcfg["max_points_per_frame"],
        perturb=PerturbConfig(),
        subsequence=False,
        deterministic=True,
    )
    # Spread indices across the val set; sort by point density so one sparse
    # track is always included.
    indices = args.sample_indices
    if indices is None:
        cand = np.linspace(0, len(ds) - 1, num=max(8, args.export_samples * 3), dtype=int)
        by_density = sorted(cand, key=lambda i: int(ds[int(i)]["points_mask"].sum()))
        indices = [int(by_density[0]), *[int(i) for i in by_density[-(args.export_samples - 1):]]]

    for k, idx in enumerate(indices[: args.export_samples]):
        batch = collate_tracks([ds[idx]])
        sample = export_sample(model, batch)
        path = args.out / f"sample_{k}.lft"
        write_lft(path, sample)
        t, n = sample["points"].shape[:2]
        print(f"{path.name}: val[{idx}] T={t} N={n} pts={int(sample['points_mask'].sum())}")


if __name__ == "__main__":
    main()


@torch.no_grad()
def export_debug(checkpoint: Path, out: Path, sample_idx: int = 0) -> None:
    """Dump per-stage pillar/backbone tensors for one sample to localize divergence."""
    import torch.nn.functional as F

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    dcfg = cfg["data"]
    ds = TrajectoryDataset(
        root=Path(dcfg["root"]), split="val", max_frames=dcfg["max_frames"],
        max_points_per_frame=dcfg["max_points_per_frame"], perturb=PerturbConfig(),
        subsequence=False, deterministic=True,
    )
    cand = np.linspace(0, len(ds) - 1, num=max(8, 9), dtype=int)
    by_density = sorted(cand, key=lambda i: int(ds[int(i)]["points_mask"].sum()))
    indices = [int(by_density[0]), *[int(i) for i in by_density[-2:]]]
    batch = collate_tracks([ds[indices[sample_idx]]])

    pe = model.pillar_encoder
    b, t = batch["points"].shape[:2]
    grid = pe._scatter_pillars(
        batch["points"].reshape(b * t, *batch["points"].shape[2:]),
        batch["points_mask"].reshape(b * t, -1),
    )
    bb = pe.backbone
    stem = bb.stem(grid)
    f1 = bb.stage1(stem)
    f2 = bb.stage2(f1)
    l1 = bb.lateral1(f1)
    l2 = bb.lateral2(f2)
    up = F.interpolate(l2, size=f1.shape[-2:], mode="nearest")
    fpn = l1 + up
    pfeat = pe(batch["points"], batch["points_mask"]).squeeze(0)
    box_emb = model.box_encoder(batch["boxes_init"]).squeeze(0)

    fi = int(batch["points_mask"].sum(dim=-1).argmax())  # densest frame
    dump = {
        "frame_index": np.array([fi], dtype=np.float32),
        "dbg_grid": grid[fi].numpy(),
        "dbg_stem": stem[fi].numpy(),
        "dbg_f1": f1[fi].numpy(),
        "dbg_f2": f2[fi].numpy(),
        "dbg_l1": l1[fi].numpy(),
        "dbg_l2": l2[fi].numpy(),
        "dbg_fpn": fpn[fi].numpy(),
        "pfeat": pfeat.numpy(),
        "box_emb": box_emb.numpy(),
    }
    write_lft(out / "debug.lft", dump)
    print(f"debug.lft: frame {fi}, grid {tuple(grid[fi].shape)}")
