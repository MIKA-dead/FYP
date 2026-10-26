"""
Geometric fix for walk + one-arm wave clips.

Fixes common MotionsFix / bad-demo failure modes:
  - lateral hop instead of forward travel
  - both arms waving
  - weak forward displacement

  python scripts/backends/wave_walk_postprocess.py \\
      --in 03_edited.npy --out 03_edited.npy --side right --intent-json plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

J_PELVIS = 0
J_L_SHOULDER, J_R_SHOULDER = 16, 17
J_L_ELBOW, J_R_ELBOW = 18, 19
J_L_WRIST, J_R_WRIST = 20, 21
J_HEAD = 15


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(hp[:, 2]).mean() > abs(hp[:, 1]).mean() else 1


def _horiz(up: int) -> List[int]:
    return [0, 1] if up == 2 else [0, 2]


def _smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    if len(x) < 3 or win < 3:
        return x
    w = min(win | 1, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if w < 3:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def project_root_forward(
    joints: np.ndarray,
    *,
    min_forward: float = 0.85,
    lateral_keep: float = 0.08,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep mostly forward root travel; damp lateral hop; boost short walks."""
    j = np.asarray(joints, dtype=np.float32).copy()
    T = j.shape[0]
    up = _up_axis(j)
    hz = _horiz(up)
    root = j[:, J_PELVIS].copy()
    delta = root[-1, hz] - root[0, hz]
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        # No net travel — invent forward along +X (or the larger horiz axis variance)
        var0 = float(np.var(root[:, hz[0]]))
        var1 = float(np.var(root[:, hz[1]]))
        fwd = np.array([1.0, 0.0], dtype=np.float32) if var0 >= var1 else np.array([0.0, 1.0], dtype=np.float32)
    else:
        fwd = (delta / dist).astype(np.float32)
    lat = np.array([-fwd[1], fwd[0]], dtype=np.float32)

    rel = root[:, hz] - root[0, hz]
    fwd_comp = rel @ fwd
    lat_comp = rel @ lat
    # Smooth + damp lateral
    lat_comp = _smooth(lat_comp, 7) * float(lateral_keep)
    # Boost forward if weak
    span = float(fwd_comp.max() - fwd_comp.min()) if T > 1 else 0.0
    end = float(fwd_comp[-1] - fwd_comp[0]) if T > 1 else 0.0
    scale = 1.0
    target = max(min_forward, abs(end) * 1.15)
    if abs(end) < min_forward * 0.55:
        # Rebuild a steady forward ramp mixed with existing signal
        ramp = np.linspace(0.0, target, T, dtype=np.float32)
        mix = 0.35
        fwd_comp = (1 - mix) * _smooth(fwd_comp, 5) + mix * ramp
        # Ensure end reaches target
        if abs(float(fwd_comp[-1] - fwd_comp[0])) < min_forward:
            fwd_comp = ramp
        scale = target / max(abs(end), 1e-3)
    elif abs(end) < min_forward:
        scale = min_forward / max(abs(end), 1e-3)
        fwd_comp = fwd_comp * scale

    new_hz = root[0, hz] + np.outer(fwd_comp, fwd) + np.outer(lat_comp, lat)
    # Apply same XY delta to all joints (rigid root shift per frame)
    shift = np.zeros_like(j)
    shift[:, :, hz[0]] = (new_hz[:, 0] - root[:, hz[0]])[:, None]
    shift[:, :, hz[1]] = (new_hz[:, 1] - root[:, hz[1]])[:, None]
    j = j + shift
    meta = {
        "forward": [float(fwd[0]), float(fwd[1])],
        "travel_in": round(dist, 4),
        "travel_out": round(float(np.linalg.norm(j[-1, J_PELVIS, hz] - j[0, J_PELVIS, hz])), 4),
        "fwd_scale": round(float(scale), 4),
        "lateral_keep": lateral_keep,
    }
    return j, meta


def damp_other_arm(
    joints: np.ndarray,
    *,
    side: str = "right",
    strength: float = 0.85,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pull the non-waving arm toward a lowered rest (early-frame pose)."""
    j = np.asarray(joints, dtype=np.float32).copy()
    T = j.shape[0]
    if side == "left":
        arm = [J_R_SHOULDER, J_R_ELBOW, J_R_WRIST]
        shoulder = J_L_SHOULDER
    else:
        arm = [J_L_SHOULDER, J_L_ELBOW, J_L_WRIST]
        shoulder = J_R_SHOULDER
    # Rest = median of first 12% frames (usually less waving)
    t0 = max(2, int(0.12 * T))
    rest = np.median(j[:t0], axis=0)
    # Also bias wrist downward relative to its shoulder
    up = _up_axis(j)
    for ji in arm:
        target = rest[ji].copy()
        # Keep horizontal offset from shoulder similar to rest, drop wrist
        if ji == arm[-1]:
            target[up] = min(float(rest[ji, up]), float(rest[shoulder, up]) - 0.15)
        alpha = float(np.clip(strength, 0.0, 1.0))
        j[:, ji] = (1 - alpha) * j[:, ji] + alpha * target[None, :]
    return j, {"side": side, "damped_joints": arm, "strength": strength}


def boost_wave_arm(
    joints: np.ndarray,
    *,
    side: str = "right",
    min_span: float = 0.12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """If waving wrist barely moves up, add a gentle sinusoidal lift."""
    j = np.asarray(joints, dtype=np.float32).copy()
    T = j.shape[0]
    up = _up_axis(j)
    wrist = J_R_WRIST if side != "left" else J_L_WRIST
    elbow = J_R_ELBOW if side != "left" else J_L_ELBOW
    shoulder = J_R_SHOULDER if side != "left" else J_L_SHOULDER
    span = float(j[:, wrist, up].max() - j[:, wrist, up].min())
    if span >= min_span:
        return j, {"boosted": False, "span": round(span, 4)}
    t = np.linspace(0, 1, T, dtype=np.float32)
    env = np.ones(T, dtype=np.float32)
    fade = max(2, T // 12)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    flap = np.sin(2 * np.pi * 3.0 * t) * env
    lift = (0.18 + 0.10 * flap) * env
    j[:, wrist, up] = j[:, wrist, up] + lift
    j[:, elbow, up] = j[:, elbow, up] + 0.55 * lift
    # Slight outward from body
    hz = _horiz(up)
    sh = j[:, shoulder]
    away = j[:, wrist, hz] - sh[:, hz]
    n = np.linalg.norm(away, axis=-1, keepdims=True) + 1e-6
    j[:, wrist, hz] = j[:, wrist, hz] + 0.04 * (away / n) * env[:, None]
    return j, {"boosted": True, "span_in": round(span, 4), "span_out": round(float(j[:, wrist, up].max() - j[:, wrist, up].min()), 4)}


def apply_wave_walk(
    joints: np.ndarray,
    *,
    side: str = "right",
    min_forward: float = 0.85,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"side": side, "T_in": int(joints.shape[0])}
    j, m1 = project_root_forward(joints, min_forward=min_forward)
    meta["root"] = m1
    j, m2 = damp_other_arm(j, side=side, strength=0.88)
    meta["other_arm"] = m2
    j, m3 = boost_wave_arm(j, side=side)
    meta["wave_arm"] = m3
    meta["T_out"] = int(j.shape[0])
    return j.astype(np.float32), meta


def process_file(
    path_in: Path,
    path_out: Path,
    *,
    side: str = "right",
    min_forward: float = 0.85,
) -> Dict[str, Any]:
    raw = np.load(path_in, allow_pickle=True)
    if raw.dtype != object:
        raise SystemExit(f"Expected object npy, got {raw.dtype}")
    obj = raw.item()
    if not isinstance(obj, dict) or "joint_positions" not in obj:
        raise SystemExit(f"Need joint_positions in {path_in}")
    joints = np.asarray(obj["joint_positions"], dtype=np.float32)
    fixed, meta = apply_wave_walk(joints, side=side, min_forward=min_forward)
    obj = dict(obj)
    obj["joint_positions"] = fixed
    if "pose" in obj:
        pose = np.asarray(obj["pose"], dtype=np.float32).copy()
        T = min(pose.shape[0], fixed.shape[0])
        pose[:T, :3] = fixed[:T, J_PELVIS]
        if fixed.shape[0] < pose.shape[0]:
            pose = pose[: fixed.shape[0]]
        obj["pose"] = pose
    obj["wave_walk_meta"] = meta
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_out, obj, allow_pickle=True)
    path_out.with_suffix(".wave_walk.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[wave_walk] side={side} travel {meta['root'].get('travel_in')}→{meta['root'].get('travel_out')} "
        f"other_arm_damped={meta['other_arm'].get('damped_joints')} → {path_out}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--min-forward", type=float, default=0.85)
    ap.add_argument("--intent-json", type=Path, default=None)
    args = ap.parse_args()
    side = args.side
    min_forward = args.min_forward
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        act = str(plan.get("primary_action") or plan.get("keep_action") or "")
        if "left" in act:
            side = "left"
        refine = plan.get("refine") or (plan.get("revision") or {}).get("refine") or {}
        hints = " ".join(str(h) for h in (refine.get("mf_hints") or []))
        raw = str(refine.get("raw_prompt") or plan.get("raw_prompt") or "")
        if any(k in raw + hints for k in ("走", "forward", "walk", "hop", "跳", "双臂", "左手")):
            min_forward = max(min_forward, 0.9)
    process_file(args.inp, args.out, side=side, min_forward=min_forward)


if __name__ == "__main__":
    main()
