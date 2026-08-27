# LabelFormer.py

PyTorch implementation of **LabelFormer** — *Object Trajectory Refinement for Offboard Perception from LiDAR Point Clouds* (Yang et al., CoRL 2023, [arXiv:2311.01444](https://arxiv.org/abs/2311.01444)) — trained on the [ArgoVerse 2](https://www.argoverse.org/av2.html) sensor dataset.

LabelFormer refines noisy object trajectories (auto-labels) from LiDAR: each frame's box + object points are encoded independently, a transformer with ALiBi relative position biases reasons over the full trajectory, and the model decodes refined per-frame poses plus a single trajectory-level object size. Here, initial noisy trajectories are simulated by perturbing ground-truth tracks with the paper's noise model (±0.25 m translation, ±10° heading, size jitter), so no first-stage detector is required.

A Mojo port lives at [labelrefinery/LabelFormer.mojo](https://github.com/labelrefinery/LabelFormer.mojo).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for data download, [s5cmd](https://github.com/peak/s5cmd) (`brew install s5cmd`).

```sh
uv sync
```

## Data

Download AV2 sensor logs (anonymous S3; only annotations + ego poses + lidar, ~200 MB/log), then extract per-track training shards:

```sh
uv run python scripts/download_av2.py --split train --num-logs 12 --out data/av2
uv run python scripts/download_av2.py --split val   --num-logs 4  --out data/av2
uv run python scripts/preprocess_tracks.py --data-root data/av2 --split train --out data/processed
uv run python scripts/preprocess_tracks.py --data-root data/av2 --split val   --out data/processed
```

Both scripts are incremental — rerun with a larger `--num-logs` (or `-1` for the full 700/150-log splits, ~all of which you only want on a big disk) to scale up.

## Train

```sh
# small local smoke run (Apple Silicon MPS or CPU)
uv run python -m labelformer.train --config configs/smoke.yaml

# paper-scale run (CUDA GPU recommended; 40 epochs, AMP)
uv run python -m labelformer.train --config configs/labelformer_av2.yaml
```

Checkpoints, config snapshot, and per-epoch history land in `runs/<run_name>/`.

## Evaluate

```sh
uv run python -m labelformer.evaluate --checkpoint runs/smoke/best.pt
```

Prints track-level mean IoU, frame-pooled mean IoU, and recall@{0.5,0.6,0.7,0.8} for the *initial* (perturbed) and *refined* trajectories side by side — the model earns its keep when the refined column beats the initial one.

### Reference smoke result

30 epochs of `configs/smoke.yaml` (12 train / 4 val logs, 739/261 vehicle tracks, ~11 min on an Apple Silicon MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`, 1.16M-param model):

| metric | initial (perturbed) | refined | delta |
|---|---|---|---|
| mean IoU | 0.794 | 0.939 | +0.145 |
| track mean IoU | 0.793 | 0.936 | +0.143 |
| recall@0.7 | 0.966 | 0.999 | +0.034 |
| recall@0.8 | 0.450 | 0.992 | +0.542 |

## Tests

```sh
uv run pytest -q
```

## Implementation notes

- BEV formulation per the paper: boxes are `(x, y, yaw, l, w)` in a trajectory frame centered on the middle frame.
- Per-frame encoder: PointPillars-style pillar grid (10 cm, 24 m × 8 m object-centric) → PointNet → ResNet/FPN CNN, fused with an MLP-encoded box.
- Transformer: 6 pre-norm blocks, 4 heads, d=256, ALiBi relative position bias.
- Decoders: per-frame pose residuals + one mean-pooled size residual per trajectory.
- Loss: Smooth-L1 on position/size and sin/cos of doubled heading (λ=0.1) + axis-aligned IoU loss.
- AV2 feather files are read directly with pyarrow (no `av2` devkit dependency).

## Mojo export

`scripts/export_mojo.py` exports checkpoint weights (BatchNorms folded into convs) and val-track test samples in the LFT1 container consumed by [LabelFormer.mojo](https://github.com/labelrefinery/LabelFormer.mojo):

```sh
uv run python scripts/export_mojo.py --checkpoint runs/smoke/best.pt --out export
```

A trained smoke checkpoint (plus Mojo-exported weights) is published at [mseritan/LabelFormer-AV2-smoke](https://huggingface.co/mseritan/LabelFormer-AV2-smoke) on HuggingFace (CC BY-NC-SA, research use).
