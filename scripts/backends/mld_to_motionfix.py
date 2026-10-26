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


def _resample_clip(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resample leading axis to target_len."""
    t0 = arr.shape[0]
    if t0 == target_len:
        return arr
    old = np.linspace(0.0, 1.0, t0, dtype=np.float64)
    new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    flat = arr.reshape(t0, -1)
    out = np.stack(
        [np.interp(new, old, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape((target_len,) + arr.shape[1:]).astype(arr.dtype)


def _temporal_smooth(
    rots: np.ndarray,
    pose_full: np.ndarray,
    trans: np.ndarray,
    window: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Light moving-average on trans + rotvecs (odd window)."""
    if window < 3 or rots.shape[0] < window:
        return rots, pose_full, trans
    if window % 2 == 0:
        window += 1
    from scipy.ndimage import uniform_filter1d

    trans_s = uniform_filter1d(trans.astype(np.float64), size=window, axis=0, mode="nearest")
    rots_s = uniform_filter1d(rots.astype(np.float64), size=window, axis=0, mode="nearest")
    pose_s = uniform_filter1d(
        pose_full.astype(np.float64), size=window, axis=0, mode="nearest"
    )
    return (
        rots_s.astype(np.float32),
        pose_s.astype(np.float32),
        trans_s.astype(np.float32),
    )


def _Rx_plus_90() -> np.ndarray:
    """+90° about X maps Y-up → Z-up: (x,y,z) → (x,-z,y)."""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )


def _rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(
        rv.shape[:-1] + (3, 3)
    )


def _matrix_to_rotvec(mat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return (
        Rotation.from_matrix(mat.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(mat.shape[:-2] + (3,))
        .astype(np.float32)
    )


def y_up_to_z_up(
    rots: np.ndarray,
    pose_full: np.ndarray,
    trans: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SMPLify/MLD Y-up motion into MotionFix/AMASS Z-up.

    MotionFix canonicalize_rotations treats Z as yaw (gravity axis). Bridged
    MLD fits are Y-up; feeding them in without this convert makes face-front
    canonicalize and the editor OOD → collapsed / no-op edits.
    """
    R = _Rx_plus_90()  # (3,3)
    # translations
    trans_z = (trans.astype(np.float64) @ R.T).astype(np.float32)
    # global orient only (first 3 of pose / rots); local body pose unchanged
    glob = _rotvec_to_matrix(pose_full[:, :3])  # (T,3,3)
    glob_z = np.einsum("ij,tjk->tik", R, glob)
    glob_aa = _matrix_to_rotvec(glob_z)
    pose_z = pose_full.copy()
    pose_z[:, :3] = glob_aa
    rots_z = rots.copy()
    rots_z[:, :3] = glob_aa
    return rots_z, pose_z, trans_z


def joints_to_motion_source(
    joints: np.ndarray,
    smpl_dir: Path,
    num_iters: int = 50,
    device: str = "cuda",
    fix_foot: bool = False,
    max_frames: Optional[int] = None,
    min_frames: int = 60,
    target_len: Optional[int] = None,
) -> Tuple[dict, dict]:
    """Fit SMPL to HumanML3D/AMASS-ordered joints; return motion_source + meta."""
    import smplx
    from mld.transforms.joints2rots import config
    from mld.transforms.joints2rots.smplify import SMPLify3D

    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints (T,22,3), got {joints.shape}")

    if max_frames is not None and joints.shape[0] > max_frames:
        print(
            f"[mld_to_motionfix] truncating joints {joints.shape[0]} -> {max_frames}"
        )
        joints = joints[:max_frames]

    # MotionFix test clips are typically 60–150 frames. Sub-60 inputs push the
    # editor OOD (body collapse / no-op edits). Prefer native length when ≥60;
    # do NOT stretch longer after fit — that dilutes root speed 1:1 with T.
    if joints.shape[0] < min_frames:
        print(
            f"[mld_to_motionfix] upsampling joints {joints.shape[0]} -> {min_frames} "
            f"(MotionFix min length)"
        )
        joints = _resample_clip(joints, min_frames)

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
    pose_full = np.zeros((T, 72), dtype=np.float32)  # for SMPL FK (24*3)
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

        # SMPL pose (72) = 24*3 aa; MotionFix stores first 22 joints (66).
        # Keep absolute camera translation for now; convert after the loop.
        pose_np = opt_pose.detach().cpu().numpy()[0]
        pose_full[idx] = pose_np
        rots[idx] = pose_np[:66]
        trans[idx] = opt_cam_t.detach().cpu().numpy()[0]
        betas_acc.append(opt_betas.detach().cpu().numpy()[0])
        if opt_loss is not None:
            try:
                losses.append(float(opt_loss.detach().cpu().mean()))
            except Exception:
                pass
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[mld_to_motionfix] fitted frame {idx+1}/{T}", flush=True)

    # MotionFix convention: trans[0] == 0 and joint_positions must match
    # SMPL FK(rots, transl=trans). SMPLify's cam is an external offset, so
    # relative cam == relative root translation.
    cam_abs = trans.copy()
    trans = (cam_abs - cam_abs[0]).astype(np.float32)

    pose_t = torch.from_numpy(pose_full).float().to(device_t)
    trans_t = torch.from_numpy(trans).float().to(device_t)
    betas_t = (
        torch.from_numpy(np.mean(betas_acc, axis=0))
        .float()
        .to(device_t)
        .unsqueeze(0)
        .expand(T, -1)
    )
    def _fk(trans_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = smplmodel(
                global_orient=pose_t[:, :3],
                body_pose=pose_t[:, 3:],  # 23 * 3 for SMPL
                betas=betas_t,
                transl=torch.from_numpy(trans_np).float().to(device_t),
            )
            return out.joints[:, :22].detach().cpu().numpy().astype(np.float32)

    joint_positions = _fk(trans)

    # Soft root-step clamp (MotionFix typical p95 ≈ 0.02–0.05). Prefer logging
    # over aggressive shrink — heavy clamp was a secondary quality loss.
    step = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
    p95 = float(np.percentile(step, 95)) if len(step) else 0.0
    speed_scale = 1.0
    target_step = 0.05
    min_step = 0.015
    hard_ceil = 0.08
    if p95 > hard_ceil:
        speed_scale = target_step / p95
    elif p95 > target_step:
        # Mild pull toward target instead of hard clamp.
        speed_scale = float(np.sqrt(target_step / p95))
        print(
            f"[mld_to_motionfix] mild speed pull x{speed_scale:.3f} "
            f"(p95 {p95:.4f} > {target_step})"
        )
    elif 0.0 < p95 < min_step:
        speed_scale = min_step / p95
    if speed_scale != 1.0:
        trans = (trans * speed_scale).astype(np.float32)
        joint_positions = _fk(trans)
        new_p95 = float(
            np.percentile(np.linalg.norm(np.diff(trans, axis=0), axis=-1), 95)
        )
        print(
            f"[mld_to_motionfix] scaled root speed x{speed_scale:.3f} "
            f"(p95 step {p95:.4f} -> {new_p95:.4f})"
        )
        p95 = new_p95
    else:
        print(f"[mld_to_motionfix] root speed ok (p95 step {p95:.4f})")

    # Optional length change after fit. Upsampling here is slow-mo (dilutes
    # speed); we re-apply the step floor/ceiling afterward.
    if target_len is not None and target_len > 0 and T != target_len:
        if target_len > T:
            print(
                f"[mld_to_motionfix] WARNING: resampling {T} -> {target_len} "
                f"stretches time (slows motion). Prefer native T≥{min_frames}."
            )
        else:
            print(f"[mld_to_motionfix] resampling motion {T} -> {target_len}")
        rots = _resample_clip(rots, target_len)
        pose_full = _resample_clip(pose_full, target_len)
        trans = _resample_clip(trans, target_len)
        T = target_len
        pose_t = torch.from_numpy(pose_full).float().to(device_t)
        betas_t = (
            torch.from_numpy(np.mean(betas_acc, axis=0))
            .float()
            .to(device_t)
            .unsqueeze(0)
            .expand(T, -1)
        )
        step2 = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
        p95_2 = float(np.percentile(step2, 95)) if len(step2) else 0.0
        fix = 1.0
        if p95_2 > hard_ceil:
            fix = target_step / p95_2
        elif p95_2 > target_step:
            fix = float(np.sqrt(target_step / p95_2))
        elif 0.0 < p95_2 < min_step:
            fix = min_step / p95_2
        if fix != 1.0:
            speed_scale *= fix
            trans = (trans * fix).astype(np.float32)
            print(
                f"[mld_to_motionfix] post-resample speed x{fix:.3f} "
                f"(p95 {p95_2:.4f})"
            )
        joint_positions = _fk(trans)
        p95 = float(
            np.percentile(np.linalg.norm(np.diff(trans, axis=0), axis=-1), 95)
        )

    # Y-up (SMPL/MLD) → Z-up (MotionFix). Must update pose_t before final FK.
    rots, pose_full, trans = y_up_to_z_up(rots, pose_full, trans)
    pose_t = torch.from_numpy(pose_full).float().to(device_t)
    joint_positions = _fk(trans)
    trans = (trans - trans[0]).astype(np.float32)
    joint_positions = _fk(trans)
    hp = joint_positions[:, 15] - joint_positions[:, 0]
    print(
        f"[mld_to_motionfix] converted Y-up→Z-up "
        f"(mean head-pelvis xyz={hp.mean(0).round(3).tolist()})",
        flush=True,
    )

    # Light temporal smooth (reduces per-frame SMPLify jitter before face-front).
    rots, pose_full, trans = _temporal_smooth(rots, pose_full, trans, window=5)
    pose_t = torch.from_numpy(pose_full).float().to(device_t)
    joint_positions = _fk(trans)
    print("[mld_to_motionfix] applied light temporal smooth (window=5)", flush=True)

    p95 = float(
        np.percentile(np.linalg.norm(np.diff(trans, axis=0), axis=-1), 95)
    )

    motion_source = {
        "rots": torch.from_numpy(rots),
        "trans": torch.from_numpy(trans),
        "joint_positions": torch.from_numpy(joint_positions),
    }
    meta = {
        "frames": int(T),
        "num_iters": num_iters,
        "mean_joint_loss": float(np.mean(losses)) if losses else None,
        "betas_mean": np.mean(betas_acc, axis=0).tolist(),
        "speed_scale": speed_scale,
        "trans_p95_step": p95,
        "target_len": target_len,
        "gravity_axis": "z",
    }
    return motion_source, meta


def parse_args():
    p = argparse.ArgumentParser(description="MLD joints → MotionFix source")
    p.add_argument("--joints", type=Path, required=True, help="(T,22,3) npy from MLD")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smpl-dir", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--min-frames",
        type=int,
        default=60,
        help="Upsample shorter clips to this length before fit (MotionFix min≈60)",
    )
    p.add_argument(
        "--target-len",
        type=int,
        default=0,
        help="Resample fitted motion to this length (0=keep native; "
        "avoid stretching up — it slows the walk and weakens MotionFix edits)",
    )
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

    target_len = None if args.target_len == 0 else args.target_len
    motion_source, meta = joints_to_motion_source(
        joints,
        smpl_dir=smpl_dir,
        num_iters=args.iters,
        device=device,
        fix_foot=args.fix_foot,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        target_len=target_len,
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
        "joints_mld": np.asarray(joints, dtype=np.float32),
        "text": "",
        "source_from": str(joints_path),
        "motionfix_canonical": False,
    }
    np.save(out, payload)
    meta_path = out.with_suffix(".json")
    meta.update(
        {
            "out": str(out),
            "joints_path": str(joints_path),
            "seconds": round(time.time() - t0, 3),
            "device": device,
            "motionfix_canonical": False,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[mld_to_motionfix] saved {out}")
    print(f"[mld_to_motionfix] meta {meta_path}  ({meta['seconds']}s)")

    # MotionsFix face-front (same as dataset prep). Runs in motionfix env via
    # MOTIONFIX_PYTHON so bridged npy matches --source-id contract.
    mf_py = os.environ.get("MOTIONFIX_PYTHON", "").strip()
    if not mf_py:
        guess = Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"
        if guess.is_file():
            mf_py = str(guess)
    if mf_py:
        import subprocess

        face_script = REPO_ROOT / "scripts" / "backends" / "motionfix_facefront.py"
        print(f"[mld_to_motionfix] MotionsFix face-front via {mf_py}", flush=True)
        proc = subprocess.run(
            [mf_py, str(face_script), "--in", str(out), "--out", str(out)],
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            print(
                f"[mld_to_motionfix] WARNING: facefront failed ({proc.returncode}); "
                "run motionfix_facefront.py manually before edit"
            )
        else:
            meta["motionfix_canonical"] = True
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        print(
            "[mld_to_motionfix] WARNING: MOTIONFIX_PYTHON not set; "
            "skip face-front (edit may be OOD)"
        )


if __name__ == "__main__":
    main()
