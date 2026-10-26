"""
Geometric sit fixes (no MotionsFix required):

  - yaw_spin: damp root facing toward start
  - head_torso_misalign: rotate head/neck toward shoulder girdle facing
  - post_sit_limb_rebound: soft-damp wrist/ankle after contact (no hard freeze,
    no timeline trim by default)

  python scripts/backends/sit_hold_postprocess.py \\
      --in 03_edited.npy --out 03_edited.npy --intent-json plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SESSION = Path(__file__).resolve().parents[1] / "session"
if str(_SESSION) not in sys.path:
    sys.path.insert(0, str(_SESSION))
from motion_timeline import (  # noqa: E402
    analyze_sit_timeline,
    apply_phase_speed,
    drop_timeline_phases,
    trim_joints_by_timeline,
)

J_PELVIS, J_HEAD = 0, 15
J_NECK = 12
J_L_SHOULDER, J_R_SHOULDER = 16, 17
J_L_WRIST, J_R_WRIST = 20, 21
J_L_FOOT, J_R_FOOT = 10, 11
ARM_J = [16, 17, 18, 19, 20, 21]


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(hp[:, 2]).mean() > abs(hp[:, 1]).mean() else 1


def _yaw_from_vec(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[:, 1], v[:, 0])


def _shoulder_yaw(joints: np.ndarray) -> np.ndarray:
    sh = joints[:, J_R_SHOULDER] - joints[:, J_L_SHOULDER]
    return _yaw_from_vec(sh)


def _yaw(joints: np.ndarray) -> np.ndarray:
    sh = joints[:, J_R_SHOULDER] - joints[:, J_L_SHOULDER]
    hips = joints[:, 2] - joints[:, 1]
    sh_h = np.linalg.norm(sh[:, :2], axis=-1).mean()
    hip_h = np.linalg.norm(hips[:, :2], axis=-1).mean()
    v = sh if sh_h >= hip_h else hips
    return _yaw_from_vec(v)


def _rot_yaw(points: np.ndarray, angle: float, center: np.ndarray, up: int) -> np.ndarray:
    out = points.copy()
    hz = [0, 1] if up == 2 else [0, 2]
    c, s = np.cos(angle), np.sin(angle)
    x = points[:, hz[0]] - center[hz[0]]
    y = points[:, hz[1]] - center[hz[1]]
    out[:, hz[0]] = center[hz[0]] + c * x - s * y
    out[:, hz[1]] = center[hz[1]] + s * x + c * y
    return out


def find_contact(joints: np.ndarray, up: int) -> int:
    pz = joints[:, J_PELVIS, up]
    T = len(pz)
    drop = float(pz[0] - pz.min())
    if drop < 1e-4:
        return max(T // 2, 1)
    target = pz[0] - 0.85 * drop
    reached = np.where(pz <= target + 1e-6)[0]
    contact = int(reached[0]) if len(reached) else int(np.argmin(pz))
    return int(np.clip(contact, int(0.25 * T), T - 2))


def apply_sit_hold(
    joints: np.ndarray,
    *,
    forbid: Optional[List[str]] = None,
    allow_trim: bool = False,
    phase_speed: Optional[Dict[str, float]] = None,
    drop_phases: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    forbid = forbid or []
    phase_speed = dict(phase_speed or {})
    drop_phases = list(drop_phases or [])
    j = np.asarray(joints, dtype=np.float32).copy()
    T = j.shape[0]
    up = _up_axis(j)
    hz = [0, 1] if up == 2 else [0, 2]
    contact = find_contact(j, up)
    meta: Dict[str, Any] = {
        "contact_frame": contact,
        "up_axis": up,
        "forbid": list(forbid),
        "T_in": T,
        "allow_trim": allow_trim,
        "phase_speed": phase_speed,
        "drop_phases": drop_phases,
    }

    yaw = np.unwrap(_yaw(j))
    yaw0 = float(yaw[0])

    # --- damp whole-body yaw toward start facing (strong after early frames) ---
    if "yaw_spin" in forbid:
        for t in range(T):
            # Keep first ~15% of motion; then remove nearly all accumulated spin
            if t < max(2, int(0.12 * T)):
                continue
            w = min(1.0, (t - 0.12 * T) / max(1.0, 0.35 * T))
            w = 0.55 + 0.45 * w  # always remove majority of spin once past intro
            delta = float(yaw[t] - yaw0) * w
            if abs(delta) < 1e-4:
                continue
            j[t] = _rot_yaw(j[t], -delta, j[t, J_PELVIS], up)
        # Second pass on unwrapped yaw after first correction
        yaw_b = np.unwrap(_yaw(j))
        for t in range(T):
            if t < max(2, int(0.12 * T)):
                continue
            delta = float(yaw_b[t] - yaw0) * 0.9
            if abs(delta) < 1e-4:
                continue
            j[t] = _rot_yaw(j[t], -delta, j[t, J_PELVIS], up)
        meta["yaw_span_before_deg"] = float(np.degrees(np.ptp(yaw)))
        meta["yaw_span_after_deg"] = float(np.degrees(np.ptp(np.unwrap(_yaw(j)))))

    # --- head follows torso facing ---
    if "head_torso_misalign" in forbid:
        sh_yaw = np.unwrap(_shoulder_yaw(j))
        for t in range(T):
            # Rotate head (and neck) around pelvis vertical so head faces like shoulders
            head_vec = j[t, J_HEAD] - j[t, J_PELVIS]
            head_yaw = float(np.arctan2(head_vec[1], head_vec[0]))
            delta = float(sh_yaw[t] - head_yaw)
            # Only correct horizontal offset of head/neck relative to pelvis
            for ji in (J_NECK, J_HEAD):
                pt = j[t, ji : ji + 1]
                j[t, ji] = _rot_yaw(pt, delta * 0.85, j[t, J_PELVIS], up)[0]
        meta["head_align"] = True

    # --- soft hold after contact (no hard freeze, no trim) ---
    if any(f in forbid for f in ("post_sit_limb_rebound", "post_sit_stretch")):
        hold = j[contact].copy()
        for t in range(contact, T):
            # Clamp pelvis rebound only
            j[t, J_PELVIS, hz[0]] = 0.7 * j[t, J_PELVIS, hz[0]] + 0.3 * hold[J_PELVIS, hz[0]]
            j[t, J_PELVIS, hz[1]] = 0.7 * j[t, J_PELVIS, hz[1]] + 0.3 * hold[J_PELVIS, hz[1]]
            j[t, J_PELVIS, up] = min(float(j[t, J_PELVIS, up]), float(hold[J_PELVIS, up]) + 0.03)
            # Soft blend arms toward contact offsets (keeps small motion)
            alpha = min(0.65, 0.15 + 0.1 * (t - contact))
            for ji in ARM_J:
                rel = hold[ji] - hold[J_PELVIS]
                target = j[t, J_PELVIS] + rel
                j[t, ji] = (1 - alpha) * j[t, ji] + alpha * target
        meta["soft_hold"] = True
        # Trim ONLY if explicitly allowed (timeline_trim forbid means DO NOT trim)
        if allow_trim and "timeline_trim" not in forbid and T - contact > 40:
            cut = min(T, contact + 35)
            j = j[:cut]
            meta["trimmed_to"] = int(cut)

    # Timeline analysis + optional preamble/trailing trim + phase speed
    timeline = analyze_sit_timeline(j, forbid=list(forbid))
    meta["timeline"] = timeline
    if "timeline_trim" in forbid:
        allow_trim = False
    want_pre = "pre_sit_locomotion" in forbid
    want_post = "post_sit_extra" in forbid
    if (want_pre or want_post) and "timeline_trim" not in forbid:
        j, tmeta = trim_joints_by_timeline(j, timeline, forbid=list(forbid))
        meta["trim"] = tmeta
        timeline = analyze_sit_timeline(j, forbid=list(forbid))
        meta["timeline_after_trim"] = timeline
    if drop_phases:
        j, dmeta = drop_timeline_phases(j, timeline, drop_phases)
        meta["drop_meta"] = dmeta
        timeline = analyze_sit_timeline(j, forbid=list(forbid))
        meta["timeline_after_drop"] = timeline
    if phase_speed:
        j, pmeta = apply_phase_speed(j, timeline, phase_speed)
        meta["phase_speed_meta"] = pmeta
        timeline = analyze_sit_timeline(j, forbid=list(forbid))
    meta["timeline_after"] = timeline

    meta["T_out"] = int(j.shape[0])
    meta["contact_frame"] = int(
        (meta.get("timeline_after") or timeline).get("contact", contact)
    )
    return j, meta


def process_file(
    path_in: Path,
    path_out: Path,
    *,
    forbid: Optional[List[str]] = None,
    allow_trim: bool = False,
    phase_speed: Optional[Dict[str, float]] = None,
    drop_phases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw = np.load(path_in, allow_pickle=True)
    if raw.dtype != object:
        raise SystemExit(f"Expected object npy with joint_positions, got {raw.dtype}")
    obj = raw.item()
    if not isinstance(obj, dict) or "joint_positions" not in obj:
        raise SystemExit(f"Need joint_positions in {path_in}")
    joints = np.asarray(obj["joint_positions"], dtype=np.float32)
    if forbid and "timeline_trim" in forbid:
        allow_trim = False
    fixed, meta = apply_sit_hold(
        joints,
        forbid=forbid,
        allow_trim=allow_trim,
        phase_speed=phase_speed,
        drop_phases=drop_phases,
    )
    obj = dict(obj)
    obj["joint_positions"] = fixed
    if "pose" in obj:
        pose = np.asarray(obj["pose"], dtype=np.float32).copy()
        T = min(pose.shape[0], fixed.shape[0])
        pose[:T, :3] = fixed[:T, J_PELVIS]
        if fixed.shape[0] < pose.shape[0]:
            pose = pose[: fixed.shape[0]]
        obj["pose"] = pose
    obj["sit_hold_meta"] = meta
    # Persist timeline for UI
    tl = meta.get("timeline_after") or meta.get("timeline")
    if tl:
        obj["motion_timeline"] = tl
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_out, obj, allow_pickle=True)
    path_out.with_suffix(".sit_hold.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    tl_path = path_out.parent / "05_timeline.json"
    if tl:
        tl_path.write_text(json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[sit_hold] contact={meta.get('contact_frame')} "
        f"T={meta.get('T_in')}→{meta.get('T_out')} "
        f"trim={meta.get('trim')} → {path_out}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Sit hold geometric postprocess")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--forbid", type=str, default="")
    ap.add_argument("--intent-json", type=Path, default=None)
    ap.add_argument("--allow-trim", action="store_true")
    args = ap.parse_args()
    forbid = [x.strip() for x in (args.forbid or "").split(",") if x.strip()]
    phase_speed: Dict[str, float] = {}
    drop_phases: List[str] = []
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        refine = plan.get("refine") or (plan.get("revision") or {}).get("refine") or {}
        if refine.get("forbid"):
            forbid = list(refine["forbid"])
        if refine.get("phase_speed"):
            phase_speed = {
                str(k): float(v) for k, v in dict(refine["phase_speed"]).items()
            }
        if refine.get("drop_phases"):
            drop_phases = [str(p) for p in refine["drop_phases"]]
    process_file(
        args.inp,
        args.out,
        forbid=forbid,
        allow_trim=args.allow_trim,
        phase_speed=phase_speed or None,
        drop_phases=drop_phases or None,
    )


if __name__ == "__main__":
    main()
