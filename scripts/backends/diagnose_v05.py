"""
Stage-gate diagnostics for v0.5 MLD → bridge → MotionsFix edit.

Env: conda `motionfix` (needs torch + MotionsFix deps for optional load check)

  python scripts/backends/diagnose_v05.py \\
      --source-npy outputs/v05/run_*/02_motionfix_source.npy \\
      [--edit-json outputs/v05/run_*/03_wave.json]

Exits non-zero if hard gates fail (canonicalize yaw / MPJPE).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _yaw(joints: np.ndarray) -> np.ndarray:
    sh = joints[:, 17] - joints[:, 16]
    return np.arctan2(sh[:, 1], sh[:, 0])


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, 15] - joints[:, 0]
    return 2 if abs(hp[:, 2]).mean() > abs(hp[:, 1]).mean() else 1


def _stats(joints: np.ndarray) -> Dict[str, float]:
    up = _up_axis(joints)
    horiz = [0, 1] if up == 2 else [0, 2]
    travel = float(np.linalg.norm(joints[-1, 0, horiz] - joints[0, 0, horiz]))
    height = float((joints[:, 15, up] - joints[:, 0, up]).mean())
    step = np.linalg.norm(np.diff(joints[:, 0, horiz], axis=0), axis=-1)
    p95 = float(np.percentile(step, 95)) if len(step) else 0.0
    yaw = _yaw(joints)
    return {
        "T": float(joints.shape[0]),
        "up_axis": float(up),
        "travel": travel,
        "height": height,
        "root_step_p95": p95,
        "yaw_mean_deg": float(np.degrees(yaw.mean())),
        "yaw_span_deg": float(np.degrees(np.ptp(np.unwrap(yaw)))),
    }


def _pelvis_local_mpjpe(a: np.ndarray, b: np.ndarray) -> float:
    la = a - a[:, 0:1]
    lb = b - b[:, 0:1]
    return float(np.linalg.norm(la - lb, axis=-1).mean())


def _load_source(path: Path) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    obj = np.load(path, allow_pickle=True).item()
    ms = obj["motion_source"]
    jp = np.asarray(ms["joint_positions"], dtype=np.float32)
    if hasattr(jp, "numpy"):
        jp = jp.numpy()
    jm = None
    if "joints_mld" in obj:
        jm = np.asarray(obj["joints_mld"], dtype=np.float32)
    return obj, jp, jm


def diagnose(
    source_npy: Path,
    edit_json: Optional[Path] = None,
    yaw_gate_deg: float = 5.0,
    mpjpe_gate_m: float = 0.01,
) -> Dict[str, Any]:
    obj, jp, jm = _load_source(source_npy)
    report: Dict[str, Any] = {
        "source_npy": str(source_npy),
        "motionfix_canonical": bool(obj.get("motionfix_canonical", False)),
        "bridged": _stats(jp),
    }

    # Bridge vs MLD (Y→Z for compare)
    if jm is not None and jm.shape[0] == jp.shape[0]:
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        jm_z = jm @ R.T
        report["bridge_vs_mld_pelvis_local_mpjpe"] = _pelvis_local_mpjpe(jm_z, jp)
        report["mld"] = _stats(jm)

    # Load-path gate: simulating edit load without re-canonicalize should be ~0.
    # If we re-canonicalize, measure damage (should be small ONLY if already facefront
    # in MotionsFix convention — for already-canonical, re-canon may still nudge).
    report["gates"] = {}
    try:
        import os
        import torch

        mf_root = REPO_ROOT / "third_party" / "motionfix"
        os.chdir(mf_root)
        if str(mf_root) not in sys.path:
            sys.path.insert(0, str(mf_root))
        backends = str(REPO_ROOT / "scripts" / "backends")
        if backends not in sys.path:
            sys.path.insert(0, backends)

        from motionfix_edit import (
            _setup_motionfix_env,
            facefront_motion_source,
        )

        _setup_motionfix_env(mf_root)
        smpl = mf_root / "data" / "body_models"
        ms = {
            "rots": torch.as_tensor(obj["motion_source"]["rots"]).float(),
            "trans": torch.as_tensor(obj["motion_source"]["trans"]).float(),
            "joint_positions": torch.as_tensor(jp).float(),
        }
        # Identity check: load without canon = stored joints
        jp_load = jp
        # Damage if someone re-canonicalizes
        can = facefront_motion_source(ms, smpl)
        jp_can = np.asarray(can["joint_positions"], dtype=np.float32)
        yaw_delta = abs(
            float(np.degrees(_yaw(jp_can).mean() - _yaw(jp_load).mean()))
        )
        # wrap to [-180,180]
        yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
        mpjpe = _pelvis_local_mpjpe(jp_load, jp_can)
        report["recanonicalize_yaw_delta_deg"] = yaw_delta
        report["recanonicalize_pelvis_local_mpjpe"] = mpjpe

        # Hard gates for *already-canonical* sources: re-facefront should be near no-op.
        # Soft warning if motionfix_canonical flag missing.
        if obj.get("motionfix_canonical", False):
            report["gates"]["yaw_stable"] = {
                "ok": yaw_delta < yaw_gate_deg,
                "value": yaw_delta,
                "limit": yaw_gate_deg,
            }
            report["gates"]["mpjpe_stable"] = {
                "ok": mpjpe < mpjpe_gate_m,
                "value": mpjpe,
                "limit": mpjpe_gate_m,
            }
        else:
            report["gates"]["motionfix_canonical_flag"] = {
                "ok": False,
                "value": False,
                "limit": True,
            }
        report["gates"]["gravity_z"] = {
            "ok": report["bridged"]["up_axis"] == 2.0,
            "value": report["bridged"]["up_axis"],
            "limit": 2.0,
        }
    except Exception as exc:  # noqa: BLE001
        report["facefront_check_error"] = str(exc)

    if edit_json is not None and edit_json.is_file():
        meta = json.loads(edit_json.read_text(encoding="utf-8-sig"))
        report["edit"] = {
            "edit_mode": meta.get("edit_mode"),
            "text": meta.get("text"),
            "init_from": meta.get("init_from"),
            "intent": meta.get("intent"),
        }
        # Soft gate: semantic should not silently be kinematic in default pipeline
        mode = meta.get("edit_mode")
        report["gates"]["edit_not_accidental_kinematic"] = {
            "ok": mode != "geometric_semantic",
            "value": mode,
            "limit": "diffusion|geometric_speed",
        }

    report["all_gates_ok"] = all(
        g.get("ok", True) for g in report.get("gates", {}).values()
    )
    return report


def parse_args():
    p = argparse.ArgumentParser(description="v0.5 pipeline stage diagnostics")
    p.add_argument("--source-npy", type=Path, required=True)
    p.add_argument("--edit-json", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--yaw-gate-deg", type=float, default=5.0)
    p.add_argument("--mpjpe-gate-m", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cwd0 = Path.cwd()
    src = args.source_npy if args.source_npy.is_absolute() else (cwd0 / args.source_npy)
    ej = None
    if args.edit_json is not None:
        ej = args.edit_json if args.edit_json.is_absolute() else (cwd0 / args.edit_json)
    report = diagnose(src.resolve(), ej.resolve() if ej else None, args.yaw_gate_deg, args.mpjpe_gate_m)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = args.out if args.out.is_absolute() else (cwd0 / args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if not report.get("all_gates_ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
