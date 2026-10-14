"""
Windows-friendly motion preview (matplotlib stick figure → mp4/gif).

Works with:
  - MLD joints npy (T,22,3)
  - MotionFix {'pose': (T,135)}  (needs --smpl for FK; else uses joints if present)
  - Bridge npy with motion_source.joint_positions or joints_mld

  python scripts/backends/visualize_motion.py \\
      --input outputs/v05/walk_joints.npy \\
      --out outputs/v05/walk_preview.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# HumanML3D / SMPL-22 kinematic chains (t2m)
SKELETON_CHAINS = [
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
    [0, 3, 6, 9, 12, 15],  # spine
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
]


def load_joints(path: Path, smpl_path: Optional[Path] = None) -> Tuple[np.ndarray, str]:
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        obj = raw.item()
        if isinstance(obj, dict):
            # Prefer fitted / MotionFix joints when present
            if "motion_source" in obj and "joint_positions" in obj["motion_source"]:
                jp = obj["motion_source"]["joint_positions"]
                if hasattr(jp, "numpy"):
                    jp = jp.numpy()
                return np.asarray(jp, dtype=np.float32), "motion_source.joint_positions"
            if "joints_mld" in obj:
                return np.asarray(obj["joints_mld"], dtype=np.float32), "joints_mld"
            if "joints" in obj:
                return np.asarray(obj["joints"], dtype=np.float32), "joints"
            if "pose" in obj:
                pose = np.asarray(obj["pose"], dtype=np.float32)
                joints = pose_to_joints(pose, smpl_path)
                return joints, "pose→FK"
        raise ValueError(f"Unrecognized object npy keys: {list(obj) if isinstance(obj, dict) else type(obj)}")
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 22:
        return arr, "joints"
    raise ValueError(f"Expected (T,22,3) joints, got {arr.shape}")


def pose_to_joints(pose: np.ndarray, smpl_path: Optional[Path]) -> np.ndarray:
    """MotionFix pose (T,135) → joints via SMPL-H FK (motionfix env)."""
    if smpl_path is None:
        raise ValueError("pose npy requires --smpl (MotionFix body_models path) for FK")
    import sys
    import torch

    mf_root = Path(__file__).resolve().parents[2] / "third_party" / "motionfix"
    if str(mf_root) not in sys.path:
        sys.path.insert(0, str(mf_root))

    import smplx
    from src.model.utils.smpl_fast import smpl_forward_fast
    from src.tools.transforms3d import transform_body_pose
    from src.utils.genutils import freeze

    pose_t = torch.from_numpy(pose[:, :135]).float()
    trans = pose_t[:, :3]
    rots_6d = pose_t[:, 3:].reshape(-1, 22, 6)
    rots_aa = transform_body_pose(rots_6d, "6d->aa")
    rots_rotm = transform_body_pose(rots_aa, "aa->rot")

    body = smplx.SMPLHLayer(
        f"{smpl_path}/smplh", model_type="smplh", gender="neutral", ext="npz"
    ).eval()
    setattr(smplx.SMPLHLayer, "smpl_forward_fast", smpl_forward_fast)
    freeze(body)
    with torch.no_grad():
        out = body.smpl_forward_fast(
            transl=trans,
            body_pose=rots_rotm[:, 1:],
            global_orient=rots_rotm[:, :1],
        )
        return out.joints[:, :22].cpu().numpy().astype(np.float32)


def render_skeleton_video(
    joints: np.ndarray,
    out_path: Path,
    fps: int = 20,
    title: str = "",
    elev: float = 15,
    azim: float = -60,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    assert joints.ndim == 3 and joints.shape[1] == 22
    T = joints.shape[0]
    # Center XY for camera; keep Y height
    centered = joints.copy()
    root_xz = joints[:, 0, [0, 2]].mean(axis=0)
    centered[:, :, 0] -= root_xz[0]
    centered[:, :, 2] -= root_xz[1]

    mins = centered.min(axis=(0, 1))
    maxs = centered.max(axis=(0, 1))
    span = float((maxs - mins).max())
    mid = (mins + maxs) / 2.0

    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    if title:
        ax.set_title(title, fontsize=10)

    lines = []
    for chain in SKELETON_CHAINS:
        (ln,) = ax.plot([], [], [], "o-", lw=2, markersize=3)
        lines.append(ln)
    (root_trail,) = ax.plot([], [], [], "-", lw=1, alpha=0.4)

    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[2] - span / 2, mid[2] + span / 2)
    ax.set_zlim(mins[1], maxs[1] + 0.05 * span)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=elev, azim=azim)

    trail_x, trail_y, trail_z = [], [], []

    def update(frame: int):
        j = centered[frame]
        for ln, chain in zip(lines, SKELETON_CHAINS):
            pts = j[chain]
            ln.set_data(pts[:, 0], pts[:, 2])
            ln.set_3d_properties(pts[:, 1])
        trail_x.append(j[0, 0])
        trail_y.append(j[0, 2])
        trail_z.append(j[0, 1])
        root_trail.set_data(trail_x, trail_y)
        root_trail.set_3d_properties(trail_z)
        return lines + [root_trail]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    try:
        if suffix == ".gif":
            anim.save(str(out_path), writer=PillowWriter(fps=fps))
        else:
            if suffix != ".mp4":
                out_path = out_path.with_suffix(".mp4")
            try:
                anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=1800))
            except Exception:
                # Fallback when ffmpeg is missing
                out_path = out_path.with_suffix(".gif")
                anim.save(str(out_path), writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Skeleton preview for MLD / MotionFix")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--title", type=str, default="")
    p.add_argument(
        "--smpl",
        type=Path,
        default=None,
        help="MotionFix data/body_models (needed for pose→joints FK)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cwd0 = Path.cwd()
    inp = args.input if args.input.is_absolute() else (cwd0 / args.input).resolve()
    smpl = None
    if args.smpl is not None:
        smpl = args.smpl if args.smpl.is_absolute() else (cwd0 / args.smpl).resolve()
    else:
        default_smpl = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "motionfix"
            / "data"
            / "body_models"
        )
        if default_smpl.is_dir():
            smpl = default_smpl

    joints, kind = load_joints(inp, smpl)
    print(f"[visualize] loaded {kind} shape={joints.shape} from {inp}")
    out = args.out
    if out is None:
        out = inp.with_name(inp.stem + "_preview.mp4")
    elif not out.is_absolute():
        out = (cwd0 / out).resolve()
    title = args.title or inp.stem
    saved = render_skeleton_video(joints, out, fps=args.fps, title=title)
    print(f"[visualize] saved {saved}")


if __name__ == "__main__":
    main()
