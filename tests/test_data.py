"""Tests for the LabelFormer data pipeline (synthetic AV2 logs only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pandas as pd
import pytest
import torch

from labelformer.data.av2_extract import extract_log_tracks, quat_to_rotation
from labelformer.data.dataset import (
    PerturbConfig,
    TrajectoryDataset,
    collate_tracks,
)
from labelformer.geometry import (
    points_in_box_mask,
    se2_from_xyt,
    transform_boxes_bev,
    wrap_angle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_FRAMES = 8
PTS_PER_FRAME = 40
# (name, city center at frame 0, per-frame city velocity, city yaw, l, w, h)
TRACK_SPECS = [
    ("track-a", (15.0, 4.0, 0.8), (0.5, 0.1, 0.0), 0.30, 4.2, 1.9, 1.6),
    ("track-b", (10.0, -6.0, 1.0), (0.0, -0.3, 0.0), -1.20, 6.0, 2.4, 2.6),
]


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a rotation about +z."""
    return float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))


def _ego_pose(i: int) -> tuple[float, np.ndarray]:
    """Nontrivial ego motion: translation plus a growing yaw."""
    return 0.05 * i, np.array([2.0 * i, 0.5 * i, 0.1])


def _city_box(spec, i: int) -> tuple[np.ndarray, float, np.ndarray]:
    """City-frame (center xyz, yaw, sizes lwh) of a track spec at frame ``i``."""
    _, c0, vel, yaw, length, width, height = spec
    center = np.array(c0) + np.array(vel) * i
    return center, yaw, np.array([length, width, height])


def _write_synthetic_log(root: Path, log_id: str = "log0", complete: bool = True) -> Path:
    """Write a tiny but schema-faithful AV2 log to ``root/<log_id>``."""
    log_dir = root / log_id
    (log_dir / "sensors" / "lidar").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    timestamps = [1_000_000_000 + i * 100_000_000 for i in range(N_FRAMES)]

    pose_rows = []
    # Ego poses are a superset of sweep timestamps: add in-between entries.
    for i, ts in enumerate(timestamps):
        for sub, offset in ((i, 0), (i + 0.5, 50_000_000)):
            yaw, trans = _ego_pose(sub)
            qw, qx, qy, qz = _yaw_quat(yaw)
            pose_rows.append(
                dict(
                    timestamp_ns=ts + offset,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    tx_m=trans[0],
                    ty_m=trans[1],
                    tz_m=trans[2],
                )
            )
    feather.write_feather(
        pd.DataFrame(pose_rows), log_dir / "city_SE3_egovehicle.feather"
    )

    ann_rows = []
    for i, ts in enumerate(timestamps):
        ego_yaw, ego_t = _ego_pose(i)
        rot = quat_to_rotation(*_yaw_quat(ego_yaw))
        points, intensities = [], []
        for spec in TRACK_SPECS:
            center, yaw, (length, width, height) = _city_box(spec, i)
            ego_center = rot.T @ (center - ego_t)
            qw, qx, qy, qz = _yaw_quat(yaw - ego_yaw)
            ann_rows.append(
                dict(
                    timestamp_ns=ts,
                    track_uuid=spec[0],
                    category="REGULAR_VEHICLE",
                    length_m=length,
                    width_m=width,
                    height_m=height,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    tx_m=ego_center[0],
                    ty_m=ego_center[1],
                    tz_m=ego_center[2],
                    num_interior_pts=PTS_PER_FRAME,
                )
            )
            # Points well inside the box, sampled in the box frame.
            local = rng.uniform(-0.4, 0.4, (PTS_PER_FRAME, 3)) * np.array(
                [length, width, height]
            )
            c, s = np.cos(yaw), np.sin(yaw)
            city = np.stack(
                [
                    center[0] + c * local[:, 0] - s * local[:, 1],
                    center[1] + s * local[:, 0] + c * local[:, 1],
                    center[2] + local[:, 2],
                ],
                axis=1,
            )
            points.append((city - ego_t) @ rot)
            intensities.append(rng.integers(0, 256, PTS_PER_FRAME))
        # Background ring at 60 m, far outside every box.
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        points.append(
            np.stack([60 * np.cos(angles), 60 * np.sin(angles), np.zeros(200)], axis=1)
        )
        intensities.append(np.zeros(200, dtype=int))

        xyz = np.concatenate(points, axis=0)
        sweep = pd.DataFrame(
            {
                "x": xyz[:, 0].astype(np.float16),
                "y": xyz[:, 1].astype(np.float16),
                "z": xyz[:, 2].astype(np.float16),
                "intensity": np.concatenate(intensities).astype(np.uint8),
                "laser_number": np.zeros(len(xyz), dtype=np.uint8),
                "offset_ns": np.zeros(len(xyz), dtype=np.int32),
            }
        )
        feather.write_feather(sweep, log_dir / "sensors" / "lidar" / f"{ts}.feather")

    feather.write_feather(pd.DataFrame(ann_rows), log_dir / "annotations.feather")
    if complete:
        (log_dir / ".complete").touch()
    return log_dir


# --------------------------------------------------------------------------
# av2_extract
# --------------------------------------------------------------------------


def test_quat_to_rotation():
    assert np.allclose(quat_to_rotation(1, 0, 0, 0), np.eye(3))
    # 90 deg about +x maps +y -> +z.
    rot = quat_to_rotation(np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)
    assert np.allclose(rot @ np.array([0, 1, 0]), [0, 0, 1], atol=1e-12)
    rot_z = quat_to_rotation(*_yaw_quat(0.7))
    assert np.allclose(rot_z @ rot_z.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.arctan2(rot_z[1, 0], rot_z[0, 0]), 0.7)


def test_extract_log_tracks(tmp_path):
    log_dir = _write_synthetic_log(tmp_path)
    tracks = extract_log_tracks(log_dir, {"REGULAR_VEHICLE"}, min_frames=5)
    assert [t["track_uuid"] for t in tracks] == ["track-a", "track-b"]

    origin = tracks[0]["origin"]
    assert np.allclose(origin, _ego_pose(0)[1])

    for track, spec in zip(tracks, TRACK_SPECS):
        assert track["category"] == "REGULAR_VEHICLE"
        assert len(track["timestamps_ns"]) == N_FRAMES
        assert track["boxes_bev"].shape == (N_FRAMES, 5)
        assert track["point_counts"].shape == (N_FRAMES,)
        # Only the in-box points survive the crop; the 60 m ring is dropped.
        assert np.all(track["point_counts"] == PTS_PER_FRAME)
        assert len(track["points"]) == int(track["point_counts"].sum())

        offsets = np.concatenate([[0], np.cumsum(track["point_counts"])])
        for i in range(N_FRAMES):
            center, yaw, (length, width, height) = _city_box(spec, i)
            box = track["boxes_bev"][i].astype(np.float64)
            # GT box in the log frame matches the known city construction.
            assert np.allclose(box[:2] + origin[:2], center[:2], atol=1e-3)
            assert np.isclose(wrap_angle(box[2] - yaw), 0.0, atol=1e-3)
            assert np.allclose(box[3:], [length, width], atol=1e-3)
            assert np.isclose(track["z_center"][i] + origin[2], center[2], atol=1e-3)
            assert np.isclose(track["height"][i], height, atol=1e-3)

            pts = track["points"][offsets[i] : offsets[i + 1]].astype(np.float64)
            # Every stored point lies inside the generous crop of the GT box.
            half = np.maximum(1.5 * np.array([length, width]) / 2, np.array([length, width]) / 2 + 0.5)
            crop = np.array([box[0], box[1], box[2], 2 * half[0], 2 * half[1]])
            assert points_in_box_mask(pts[:, :2], crop).all()
            assert np.all(np.abs(pts[:, 2] - track["z_center"][i]) <= 0.6 * height + 0.3)
            assert np.all((pts[:, 3] >= 0) & (pts[:, 3] <= 255))


def test_extract_filters_categories_and_short_tracks(tmp_path):
    log_dir = _write_synthetic_log(tmp_path)
    assert extract_log_tracks(log_dir, {"BUS"}) == []
    assert extract_log_tracks(log_dir, {"REGULAR_VEHICLE"}, min_frames=N_FRAMES + 1) == []
    assert (
        extract_log_tracks(
            log_dir, {"REGULAR_VEHICLE"}, min_total_points=10 * N_FRAMES * PTS_PER_FRAME
        )
        == []
    )


# --------------------------------------------------------------------------
# preprocess_tracks.py
# --------------------------------------------------------------------------


def _run_preprocess(data_root: Path, out: Path) -> str:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_tracks.py"),
        "--data-root",
        str(data_root),
        "--split",
        "train",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_preprocess_script_is_incremental(tmp_path):
    data_root = tmp_path / "av2"
    _write_synthetic_log(data_root / "train", "log0")
    # An incomplete log (no .complete marker) must be ignored.
    _write_synthetic_log(data_root / "train", "log1", complete=False)
    out = tmp_path / "processed"

    stdout = _run_preprocess(data_root, out)
    assert "Found 1 complete logs" in stdout
    assert "2 tracks" in stdout
    log_out = out / "train" / "log0"
    npzs = sorted(p.name for p in log_out.glob("*.npz"))
    assert npzs == ["track-a.npz", "track-b.npz"]
    assert (log_out / ".done").exists()
    assert not (out / "train" / "log1").exists()

    with np.load(log_out / "track-a.npz") as data:
        assert set(data.files) == {
            "track_uuid",
            "category",
            "origin",
            "timestamps_ns",
            "boxes_bev",
            "z_center",
            "height",
            "points",
            "point_counts",
        }
        assert data["boxes_bev"].shape == (N_FRAMES, 5)
        assert str(data["track_uuid"]) == "track-a"

    mtimes = {p: p.stat().st_mtime_ns for p in log_out.glob("*.npz")}
    stdout = _run_preprocess(data_root, out)
    assert "already processed" in stdout
    assert {p: p.stat().st_mtime_ns for p in log_out.glob("*.npz")} == mtimes


# --------------------------------------------------------------------------
# TrajectoryDataset
# --------------------------------------------------------------------------


def _write_track_npz(
    root: Path,
    log_id: str,
    uuid: str,
    n_frames: int = 10,
    n_points: int = 30,
    empty_frames: tuple[int, ...] = (),
    seed: int = 0,
) -> Path:
    """Write one synthetic preprocessed track NPZ (log frame)."""
    rng = np.random.default_rng(seed)
    length, width, height = 4.0, 2.0, 1.5
    boxes = np.zeros((n_frames, 5), dtype=np.float32)
    z_center = np.full(n_frames, 0.9, dtype=np.float32)
    counts, points = [], []
    for i in range(n_frames):
        yaw = 0.1 + 0.01 * i
        boxes[i] = (3.0 + 1.1 * i, -2.0 + 0.2 * i, yaw, length, width)
        # Frame-dependent count so a subsequence window is identifiable.
        k = 0 if i in empty_frames else n_points + i
        counts.append(k)
        # Sampled well inside the box, so the perturbed re-crop keeps them all.
        local = rng.uniform(-1.0, 1.0, (k, 3)) * np.array(
            [0.25 * length, 0.15 * width, 0.3 * height]
        )
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.zeros((k, 4), dtype=np.float32)
        pts[:, 0] = boxes[i, 0] + c * local[:, 0] - s * local[:, 1]
        pts[:, 1] = boxes[i, 1] + s * local[:, 0] + c * local[:, 1]
        pts[:, 2] = z_center[i] + local[:, 2]
        pts[:, 3] = rng.uniform(0, 255, k)
        points.append(pts)

    out_dir = root / log_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{uuid}.npz"
    np.savez_compressed(
        path,
        track_uuid=uuid,
        category="REGULAR_VEHICLE",
        origin=np.zeros(3),
        timestamps_ns=np.arange(n_frames, dtype=np.int64),
        boxes_bev=boxes,
        z_center=z_center,
        height=np.full(n_frames, height, dtype=np.float32),
        points=np.concatenate(points, axis=0).astype(np.float32),
        point_counts=np.array(counts, dtype=np.int32),
    )
    return path


@pytest.fixture
def npz_root(tmp_path) -> Path:
    root = tmp_path / "processed"
    _write_track_npz(root / "val", "log0", "track-a", n_frames=10, n_points=30)
    _write_track_npz(root / "val", "log1", "track-b", n_frames=6, n_points=12, seed=1)
    return root


def test_dataset_basic_shapes(npz_root):
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    assert len(ds) == 2
    sample = ds[0]
    n_frames = sample["boxes_init"].shape[0]
    assert n_frames == 10
    assert sample["boxes_gt"].shape == (n_frames, 5)
    assert sample["points"].shape[0] == n_frames
    assert sample["points"].shape[2] == 4
    assert sample["points_mask"].shape == sample["points"].shape[:2]
    assert sample["frame_mask"].shape == (n_frames,)
    assert bool(sample["frame_mask"].all())
    assert sample["track_uuid"] == "track-a"
    assert sample["log_id"] == "log0"
    assert sample["boxes_init"].dtype == torch.float32


def test_perturbation_within_bounds(npz_root):
    cfg = PerturbConfig()
    ds = TrajectoryDataset(npz_root, "val", perturb=cfg)
    with np.load(ds.files[0]) as data:
        boxes_gt = data["boxes_bev"].astype(np.float64)

    max_dxy, max_dyaw, min_scale, max_scale = 0.0, 0.0, np.inf, 0.0
    rng = np.random.default_rng(7)
    for _ in range(200):
        init = ds._perturb_boxes(boxes_gt, rng)
        max_dxy = max(max_dxy, np.abs(init[:, :2] - boxes_gt[:, :2]).max())
        max_dyaw = max(max_dyaw, np.abs(wrap_angle(init[:, 2] - boxes_gt[:, 2])).max())
        scales = init[:, 3:] / boxes_gt[:, 3:]
        min_scale = min(min_scale, scales.min())
        max_scale = max(max_scale, scales.max())

    assert max_dxy <= cfg.max_translation
    assert max_dxy > 0.9 * cfg.max_translation  # noise is actually applied
    assert max_dyaw <= np.deg2rad(cfg.max_rotation_deg) + 1e-9
    assert max_dyaw > 0.9 * np.deg2rad(cfg.max_rotation_deg)
    assert min_scale >= cfg.size_scale_range[0] - 1e-9
    assert max_scale <= cfg.size_scale_range[1] + 1e-9

    # In the trajectory frame the per-axis offset is rotated, so bound the norm.
    sample = ds[0]
    delta = (sample["boxes_init"][:, :2] - sample["boxes_gt"][:, :2]).numpy()
    assert np.linalg.norm(delta, axis=1).max() <= np.sqrt(2) * cfg.max_translation + 1e-6


def test_trajectory_frame_is_anchored_at_middle_frame(npz_root):
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    sample = ds[0]
    mid = sample["boxes_init"].shape[0] // 2
    box = sample["boxes_init"][mid].numpy()
    assert np.allclose(box[:3], 0.0, atol=1e-5)
    assert 0.9 * 4.0 <= box[3] <= 1.1 * 4.0
    assert 0.9 * 2.0 <= box[4] <= 1.1 * 2.0


def test_deterministic_mode_is_reproducible(npz_root):
    a = TrajectoryDataset(npz_root, "val", deterministic=True)[0]
    b = TrajectoryDataset(npz_root, "val", deterministic=True)[0]
    for key in ("boxes_init", "boxes_gt", "points", "points_mask", "frame_mask"):
        assert torch.equal(a[key], b[key]), key
    c = TrajectoryDataset(npz_root, "val")[0]
    assert not torch.equal(a["boxes_init"], c["boxes_init"])


def test_points_are_in_the_initial_box_frame(npz_root):
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    sample = ds[0]
    boxes = sample["boxes_init"].numpy()
    points = sample["points"].numpy()
    mask = sample["points_mask"].numpy()
    assert mask.any()
    for t in range(len(boxes)):
        pts = points[t][mask[t]]
        assert len(pts) > 0
        assert np.all(np.abs(pts[:, 0]) <= 1.1 * boxes[t, 3] / 2 + 1e-4)
        assert np.all(np.abs(pts[:, 1]) <= 1.1 * boxes[t, 4] / 2 + 1e-4)
        assert np.all((pts[:, 3] >= 0.0) & (pts[:, 3] <= 1.0))
        # Padding slots stay zero.
        assert np.all(points[t][~mask[t]] == 0.0)


def test_subsequence_and_middle_window(tmp_path):
    root = tmp_path / "processed"
    _write_track_npz(root / "train", "log0", "long", n_frames=20, n_points=5)

    # Random contiguous windows of the requested length; frame i carries 5 + i
    # points, so the window start is readable off the mask.
    ds = TrajectoryDataset(root, "train", max_frames=8)
    starts = set()
    for _ in range(30):
        sample = ds[0]
        assert sample["boxes_init"].shape[0] == 8
        counts = sample["points_mask"].sum(dim=1).numpy()
        assert np.array_equal(counts, counts[0] + np.arange(8))
        starts.add(int(counts[0]) - 5)
    assert starts <= set(range(13)) and len(starts) > 1

    # subsequence=False (and deterministic) take the middle window: frames 6..13.
    for kwargs in ({"subsequence": False}, {"deterministic": True}):
        sample = TrajectoryDataset(root, "train", max_frames=8, **kwargs)[0]
        counts = sample["points_mask"].sum(dim=1).numpy()
        assert np.array_equal(counts, 5 + 6 + np.arange(8))


def test_max_points_per_frame_subsampling(npz_root):
    ds = TrajectoryDataset(npz_root, "val", max_points_per_frame=5, deterministic=True)
    sample = ds[0]
    assert sample["points"].shape[1] <= 5
    assert int(sample["points_mask"].sum(dim=1).max()) <= 5


def test_empty_points_frame(tmp_path):
    root = tmp_path / "processed"
    _write_track_npz(root / "val", "log0", "gappy", n_frames=6, empty_frames=(0, 3))
    ds = TrajectoryDataset(root, "val", deterministic=True)
    sample = ds[0]
    counts = sample["points_mask"].sum(dim=1)
    assert int(counts[0]) == 0 and int(counts[3]) == 0
    assert int(counts[1]) > 0


def test_all_points_empty_track(tmp_path):
    root = tmp_path / "processed"
    _write_track_npz(root / "val", "log0", "void", n_frames=5, empty_frames=tuple(range(5)))
    sample = TrajectoryDataset(root, "val", deterministic=True)[0]
    assert sample["points"].shape == (5, 0, 4)
    assert sample["points_mask"].shape == (5, 0)


def test_collate_ragged_batch(npz_root):
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    samples = [ds[0], ds[1]]
    batch = collate_tracks(samples)
    max_t = max(s["frame_mask"].shape[0] for s in samples)
    max_n = max(s["points"].shape[1] for s in samples)
    assert batch["boxes_init"].shape == (2, max_t, 5)
    assert batch["boxes_gt"].shape == (2, max_t, 5)
    assert batch["points"].shape == (2, max_t, max_n, 4)
    assert batch["points_mask"].shape == (2, max_t, max_n)
    assert batch["frame_mask"].shape == (2, max_t)
    assert batch["track_uuid"] == ["track-a", "track-b"]
    assert batch["log_id"] == ["log0", "log1"]

    for i, s in enumerate(samples):
        t, n = s["frame_mask"].shape[0], s["points"].shape[1]
        assert bool(batch["frame_mask"][i, :t].all())
        assert not bool(batch["frame_mask"][i, t:].any())
        assert torch.equal(batch["boxes_init"][i, :t], s["boxes_init"])
        assert torch.equal(batch["points"][i, :t, :n], s["points"])
        assert not bool(batch["points_mask"][i, :, n:].any())
        assert not bool(batch["points_mask"][i, t:].any())


def test_collate_as_dataloader_collate_fn(npz_root):
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=collate_tracks, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["boxes_init"].shape[0] == 2
    assert batch["frame_mask"].shape[0] == 2


def test_gt_boxes_agree_with_trajectory_transform(npz_root):
    """boxes_gt is the stored log-frame GT rigidly moved into the trajectory frame."""
    ds = TrajectoryDataset(npz_root, "val", deterministic=True)
    sample = ds[0]
    with np.load(ds.files[0]) as data:
        gt_log = data["boxes_bev"].astype(np.float64)
    gt_traj = sample["boxes_gt"].numpy()
    # Sizes are untouched and pairwise distances are preserved by the rigid map.
    assert np.allclose(gt_traj[:, 3:], gt_log[:, 3:], atol=1e-4)
    d_log = np.linalg.norm(gt_log[1:, :2] - gt_log[:-1, :2], axis=1)
    d_traj = np.linalg.norm(gt_traj[1:, :2] - gt_traj[:-1, :2], axis=1)
    assert np.allclose(d_log, d_traj, atol=1e-4)
    # Headings differ from the log frame by one constant rotation.
    mid = len(gt_traj) // 2
    offset = wrap_angle(gt_traj[mid, 2] - gt_log[mid, 2])
    recovered = transform_boxes_bev(se2_from_xyt(0.0, 0.0, -offset), gt_traj)
    assert np.allclose(wrap_angle(recovered[:, 2] - gt_log[:, 2]), 0.0, atol=1e-4)
