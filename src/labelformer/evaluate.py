"""Evaluate a LabelFormer checkpoint: refined vs. initial (perturbed) trajectories.

Usage (from train/):
    uv run python -m labelformer.evaluate --checkpoint runs/smoke/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from labelformer import metrics as M
from labelformer.data.dataset import PerturbConfig, TrajectoryDataset, collate_tracks
from labelformer.train import build_model, pick_device, to_device


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path (default: alongside ckpt)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = pick_device(args.device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dcfg = cfg["data"]
    ds = TrajectoryDataset(
        root=Path(dcfg["root"]),
        split=args.split,
        max_frames=dcfg["max_frames"],
        max_points_per_frame=dcfg["max_points_per_frame"],
        perturb=PerturbConfig(),
        subsequence=False,
        deterministic=True,
    )
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate_tracks)
    print(f"evaluating {len(ds)} tracks from split '{args.split}' on {device}")

    stats = {"refined": [], "initial": []}
    weights = []
    for batch in dl:
        batch = to_device(batch, device)
        out = model(batch)
        gt = batch["boxes_gt"].cpu().numpy()
        mask = batch["frame_mask"].cpu().numpy()
        weights.append(mask.sum())
        stats["refined"].append(M.summarize(out["boxes_refined"].cpu().numpy(), gt, mask))
        stats["initial"].append(M.summarize(batch["boxes_init"].cpu().numpy(), gt, mask))

    w = np.asarray(weights, dtype=np.float64)
    report = {
        kind: {
            k: float(np.average([s[k] for s in chunks], weights=w))
            for k in chunks[0]
        }
        for kind, chunks in stats.items()
    }
    report["num_tracks"] = len(ds)
    report["num_frames"] = int(w.sum())

    print(f"\n{'metric':<20} {'initial':>10} {'refined':>10} {'delta':>10}")
    for k in stats["refined"][0]:
        i, r = report["initial"][k], report["refined"][k]
        print(f"{k:<20} {i:>10.4f} {r:>10.4f} {r - i:>+10.4f}")

    out_path = args.out or args.checkpoint.parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
