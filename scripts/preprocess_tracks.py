#!/usr/bin/env python3
"""Turn downloaded AV2 logs into per-track NPZ files for LabelFormer training.

Only logs carrying a ``.complete`` marker (written by ``download_av2.py``) are
processed; each finished output log directory gets a ``.done`` marker so the
script is incremental and safe to re-run while downloads are still going.

Example:
    python scripts/preprocess_tracks.py --data-root data/av2 --split train \\
        --out data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from labelformer.data.av2_extract import extract_log_tracks

DEFAULT_CATEGORIES = [
    "REGULAR_VEHICLE",
    "LARGE_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "TRUCK_CAB",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "VEHICULAR_TRAILER",
]


def process_log(log_dir: Path, out_dir: Path, categories: set[str], **kwargs) -> tuple[int, int]:
    """Extract one log into ``out_dir``; returns (n_tracks, n_frames)."""
    tracks = extract_log_tracks(log_dir, categories, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    for track in tracks:
        np.savez_compressed(out_dir / f"{track['track_uuid']}.npz", **track)
        n_frames += len(track["timestamps_ns"])
    (out_dir / ".done").touch()
    return len(tracks), n_frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/av2"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--crop-scale", type=float, default=1.5)
    ap.add_argument("--crop-margin", type=float, default=0.5)
    ap.add_argument("--min-total-points", type=int, default=20)
    args = ap.parse_args(argv)

    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        print(f"No such split directory: {split_dir}", file=sys.stderr)
        return 1

    log_dirs = sorted(d for d in split_dir.iterdir() if (d / ".complete").exists())
    print(f"Found {len(log_dirs)} complete logs in {split_dir}")

    total_tracks = total_frames = 0
    for i, log_dir in enumerate(log_dirs, 1):
        out_dir = args.out / args.split / log_dir.name
        if (out_dir / ".done").exists():
            print(f"[{i}/{len(log_dirs)}] {log_dir.name}: already processed, skipping")
            continue
        n_tracks, n_frames = process_log(
            log_dir,
            out_dir,
            set(args.categories),
            min_frames=args.min_frames,
            crop_scale=args.crop_scale,
            crop_margin=args.crop_margin,
            min_total_points=args.min_total_points,
        )
        total_tracks += n_tracks
        total_frames += n_frames
        print(f"[{i}/{len(log_dirs)}] {log_dir.name}: {n_tracks} tracks, {n_frames} frames")

    print(f"Done. {total_tracks} tracks, {total_frames} frames written to {args.out / args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
