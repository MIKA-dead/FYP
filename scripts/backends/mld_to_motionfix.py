"""
Bridge: MLD joints (T,22,3) → MotionFix motion_source.

Env: conda `mld` (needs smplx + joints2rots / SMPLify3D)

  python scripts/backends/mld_to_motionfix.py \\
      --joints path/to/Example_60_batch0_0.npy \\
      --out outputs/v05/source_from_mld.npy

Output npy (object dict) is loadable by motionfix_edit.py --source-npy:
  {
    'motion_source': {rots (T,66), trans (T,3), joint_positions (T,22,3)},
    'joints_mld': original joints,
    'betas': (10,),
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

# chumpy (SMPL .pkl) breaks on NumPy 2.x; restore removed aliases before smplx import.
for _alias, _val in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
):
    if not hasattr(np, _alias):
        setattr(np, _alias, _val)


REPO_ROOT = Path(__file__).resolve().parents[2]
MLD_ROOT = REPO_ROOT / "third_party" / "motion-latent-diffusion"
DEFAULT_SMPL = MLD_ROOT / "deps" / "smpl"


def _setup_mld_env(root: Path) -> None:
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def joints_to_motion_source(
    joints: np.ndarray,
    smpl_dir: Path,
    num_iters: int = 50,
    device: str = "cuda",
    fix_foot: bool = False,
    max_frames: Optional[int] = None,
) -> Tuple[dict, dict]:
    """Fit SMPL to HumanML3D/AMASS-ordered joints; return motion_source + meta."""
    import smplx
    from mld.transforms.joints2rots import config
    from mld.transforms.joints2rots.smplify import SMPLify3D

    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints (T,22,3), got {joints.shape}")

    if max_frames is not None and joints.shape[0] > max_frames:
        joints = joints[:max_frames]

    # Point config at local SMPL assets
    config.SMPL_MODEL_DIR = str(smpl_dir)
    config.GMM_MODEL_DIR = str(smpl_dir)
    config.SMPL_MEAN_FILE = str(smpl_dir / "neutral_smpl_mean_params.h5")
    config.Part_Seg_DIR = str(smpl_dir / "smplx_parts_segm.pkl")

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    smplmodel = smplx.create(
        str(smpl_dir),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device_t)

    with h5py.File(config.SMPL_MEAN_FILE, "r") as f:
        init_mean_pose = torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(device_t)
        init_mean_shape = torch.from_numpy(f["shape"][:]).unsqueeze(0).float().to(device_t)

    smplify = SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=1,
        joints_category="AMASS",
        num_iters=num_iters,
        device=device_t,
    )

    T = joints.shape[0]
    rots = np.zeros((T, 66), dtype=np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    joint_positions = np.zeros((T, 22, 3), dtype=np.float32)
    betas_acc = []

    pred_pose = torch.zeros(1, 72, device=device_t)
    pred_betas = torch.zeros(1, 10, device=device_t)
    pred_cam_t = torch.zeros(1, 3, device=device_t)
    keypoints_3d = torch.zeros(1, 22, 3, device=device_t)
    confidence = torch.ones(22, device=device_t)
    if fix_foot:
        for j in (7, 8, 10, 11):
            confidence[j] = 1.5

    losses = []
    for idx in range(T):
        keypoints_3d[0] = torch.from_numpy(joints[idx]).float().to(device_t)
        if idx == 0:
            pred_betas[:] = init_mean_shape
            pred_pose[:] = init_mean_pose
            pred_cam_t.zero_()
        (
            _verts,
            opt_joints,
            opt_pose,
            opt_betas,
            opt_cam_t,
            opt_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence,
        )
        pred_pose = opt_pose.detach()
        pred_betas = opt_betas.detach()
        pred_cam_t = opt_cam_t.detach()

        # SMPL pose (72) = 24*3 aa; MotionFix uses first 22 joints (66)
        pose_np = opt_pose.detach().cpu().numpy()[0]
        rots[idx] = pose_np[:66]
        trans[idx] = opt_cam_t.detach().cpu().numpy()[0]
        # Prefer fitted SMPL joints (AMASS 22) over raw MLD joints
        j = opt_joints.detach().cpu().numpy()[0]
        joint_positions[idx] = j[:22]
        betas_acc.append(opt_betas.detach().cpu().numpy()[0])
        if opt_loss is not None:
            try:
                losses.append(float(opt_loss.detach().cpu().mean()))
            except Exception:
                pass
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[mld_to_motionfix] fitted frame {idx+1}/{T}")

    motion_source = {
        "rots": torch.from_numpy(rots),
        "trans": torch.from_numpy(trans),
        "joint_positions": torch.from_numpy(joint_positions),
    }
    meta = {
        "frames": T,
        "num_iters": num_iters,
        "mean_joint_loss": float(np.mean(losses)) if losses else None,
        "betas_mean": np.mean(betas_acc, axis=0).tolist(),
    }
    return motion_source, meta


def parse_args():
    p = argparse.ArgumentParser(description="MLD joints → MotionFix source")
    p.add_argument("--joints", type=Path, required=True, help="(T,22,3) npy from MLD")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smpl-dir", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fix-foot", action="store_true")
    p.add_argument("--mld-root", type=Path, default=MLD_ROOT)
    return p.parse_args()


def main():
    args = parse_args()
    cwd0 = Path.cwd()
    root = args.mld_root.resolve()

    def _abs(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (cwd0 / p).resolve()

    joints_path = _abs(args.joints)
    out = _abs(args.out)
    smpl_dir = _abs(args.smpl_dir)

    joints = np.load(joints_path)
    if joints.dtype == object:
        obj = joints.item()
        joints = np.asarray(obj.get("joints", obj.get("pose", obj)))
    joints = np.asarray(joints, dtype=np.float32)

    t0 = time.time()
    _setup_mld_env(root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mld_to_motionfix] device={device} joints={joints.shape}")

    motion_source, meta = joints_to_motion_source(
        joints,
        smpl_dir=smpl_dir,
        num_iters=args.iters,
        device=device,
        fix_foot=args.fix_foot,
        max_frames=args.max_frames,
    )

    if out is None:
        out = (REPO_ROOT / "outputs" / "v05" / f"{joints_path.stem}_motionfix_source.npy").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save tensors as numpy for portability across envs
    payload = {
        "motion_source": {
            "rots": motion_source["rots"].numpy(),
            "trans": motion_source["trans"].numpy(),
            "joint_positions": motion_source["joint_positions"].numpy(),
        },
        "joints_mld": joints[: meta["frames"]],
        "text": "",
        "source_from": str(joints_path),
    }
    np.save(out, payload)
    meta_path = out.with_suffix(".json")
    meta.update(
        {
            "out": str(out),
            "joints_path": str(joints_path),
            "seconds": round(time.time() - t0, 3),
            "device": device,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[mld_to_motionfix] saved {out}")
    print(f"[mld_to_motionfix] meta {meta_path}  ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
