"""Training entry point for LabelFormer on ArgoVerse 2 trajectories.

Usage (from train/):
    uv run python -m labelformer.train --config configs/smoke.yaml
    uv run python -m labelformer.train --config configs/labelformer_av2.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from labelformer import metrics as M
from labelformer.data.dataset import PerturbConfig, TrajectoryDataset, collate_tracks
from labelformer.losses import labelformer_loss
from labelformer.model import LabelFormer, LabelFormerConfig
from labelformer.model.pillar_encoder import PillarEncoderConfig


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(mcfg: dict) -> LabelFormer:
    pillar = PillarEncoderConfig(
        x_range=tuple(mcfg["pillar"]["x_range"]),
        y_range=tuple(mcfg["pillar"]["y_range"]),
        pillar_size=mcfg["pillar"]["pillar_size"],
        point_feat_dim=mcfg["pillar"]["point_feat_dim"],
        out_dim=mcfg["pillar"]["out_dim"],
    )
    cfg = LabelFormerConfig(
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        pillar=pillar,
    )
    return LabelFormer(cfg)


def lr_lambda_factory(total_steps: int, warmup_steps: int, final_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return fn


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, device, heading_weight: float) -> dict:
    model.eval()
    losses: list[float] = []
    refined, initial, gts, masks = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        loss = labelformer_loss(
            out["boxes_refined"], batch["boxes_gt"], batch["frame_mask"], heading_weight=heading_weight
        )
        losses.append(float(loss["loss_total"]))
        refined.append(out["boxes_refined"].cpu().numpy())
        initial.append(batch["boxes_init"].cpu().numpy())
        gts.append(batch["boxes_gt"].cpu().numpy())
        masks.append(batch["frame_mask"].cpu().numpy())

    def pooled(preds: list[np.ndarray]) -> dict:
        out: dict[str, list[float]] = {}
        for p, g, m in zip(preds, gts, masks):
            s = M.summarize(p, g, m)
            for k, v in s.items():
                out.setdefault(k, []).append(v * m.sum())
        n = sum(m.sum() for m in masks)
        return {k: float(sum(v) / n) for k, v in out.items()}

    return {
        "val_loss": float(np.mean(losses)),
        "refined": pooled(refined),
        "initial": pooled(initial),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--limit-tracks", type=int, default=None, help="debug: cap dataset size")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tcfg, dcfg = cfg["train"], cfg["data"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    device = pick_device(tcfg["device"])
    print(f"device: {device}")

    common = dict(
        root=Path(dcfg["root"]),
        max_frames=dcfg["max_frames"],
        max_points_per_frame=dcfg["max_points_per_frame"],
        perturb=PerturbConfig(),
    )
    train_ds = TrajectoryDataset(split="train", subsequence=True, deterministic=False, **common)
    val_ds = TrajectoryDataset(split="val", subsequence=False, deterministic=True, **common)
    if args.limit_tracks:
        train_ds.files = train_ds.files[: args.limit_tracks]
        val_ds.files = val_ds.files[: max(1, args.limit_tracks // 4)]
    print(f"tracks: train={len(train_ds)} val={len(val_ds)}")

    dl_kwargs = dict(
        batch_size=tcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        collate_fn=collate_tracks,
        persistent_workers=dcfg["num_workers"] > 0,
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps_per_epoch = max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda_factory(
            total_steps=tcfg["epochs"] * steps_per_epoch,
            warmup_steps=tcfg["warmup_epochs"] * steps_per_epoch,
            final_ratio=tcfg["final_lr_ratio"],
        ),
    )
    use_amp = bool(tcfg.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    best_iou, history = -1.0, []

    for epoch in range(tcfg["epochs"]):
        model.train()
        t0, running = time.time(), []
        for step, batch in enumerate(train_dl):
            batch = to_device(batch, device)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(batch)
                loss = labelformer_loss(
                    out["boxes_refined"],
                    batch["boxes_gt"],
                    batch["frame_mask"],
                    heading_weight=tcfg["heading_weight"],
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss["loss_total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(float(loss["loss_total"].detach()))
            if (step + 1) % tcfg["log_every"] == 0:
                print(
                    f"epoch {epoch} step {step + 1}/{steps_per_epoch} "
                    f"loss {np.mean(running[-tcfg['log_every']:]):.4f} lr {sched.get_last_lr()[0]:.2e}"
                )

        val = evaluate_epoch(model, val_dl, device, tcfg["heading_weight"])
        iou = val["refined"]["mean_iou"]
        print(
            f"epoch {epoch} done in {time.time() - t0:.0f}s | train loss {np.mean(running):.4f} | "
            f"val loss {val['val_loss']:.4f} | refined IoU {iou:.4f} "
            f"(initial {val['initial']['mean_iou']:.4f})"
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(running)), **val})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}
        torch.save(ckpt, run_dir / "last.pt")
        if iou > best_iou:
            best_iou = iou
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  new best (mean IoU {iou:.4f}) -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
