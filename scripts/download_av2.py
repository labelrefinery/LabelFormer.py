#!/usr/bin/env python3
"""Download a subset of the ArgoVerse 2 sensor dataset via s5cmd (anonymous S3).

Only pulls what LabelFormer training needs per log: annotations, ego poses, and
lidar sweeps (no cameras, no maps). Log selection is deterministic (sorted order)
so re-runs and scale-ups are incremental. Use --num-logs -1 for the full split.

Example:
    python scripts/download_av2.py --split train --num-logs 12 --out data/av2
    python scripts/download_av2.py --split val   --num-logs 4  --out data/av2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

S3_ROOT = "s3://argoverse/datasets/av2/sensor"

PER_LOG_ITEMS = [
    "annotations.feather",
    "city_SE3_egovehicle.feather",
    "sensors/lidar/*",
]


def s5cmd(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["s5cmd", "--no-sign-request", *args]
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def list_logs(split: str) -> list[str]:
    """Return sorted log ids for a split."""
    out = s5cmd("ls", f"{S3_ROOT}/{split}/", capture=True).stdout
    logs = []
    for line in out.splitlines():
        # `s5cmd ls` on a prefix prints lines like: "                  DIR  <log_id>/"
        parts = line.split()
        if parts and parts[0] == "DIR":
            logs.append(parts[-1].rstrip("/"))
    return sorted(logs)


def download_log(split: str, log_id: str, out_root: Path) -> None:
    dest = out_root / split / log_id
    for item in PER_LOG_ITEMS:
        src = f"{S3_ROOT}/{split}/{log_id}/{item}"
        if item.endswith("*"):
            sub = dest / item[: -len("/*")]
            sub.mkdir(parents=True, exist_ok=True)
            s5cmd("cp", src, f"{sub}/")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            s5cmd("cp", src, f"{dest}/{item}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--num-logs", type=int, default=12, help="-1 for all logs")
    ap.add_argument("--out", type=Path, default=Path("data/av2"))
    args = ap.parse_args()

    logs = list_logs(args.split)
    if args.num_logs >= 0:
        logs = logs[: args.num_logs]
    print(f"Downloading {len(logs)} logs from split '{args.split}' -> {args.out}")

    for i, log_id in enumerate(logs, 1):
        marker = args.out / args.split / log_id / ".complete"
        if marker.exists():
            print(f"[{i}/{len(logs)}] {log_id} already downloaded, skipping")
            continue
        print(f"[{i}/{len(logs)}] {log_id}")
        download_log(args.split, log_id, args.out)
        marker.touch()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
