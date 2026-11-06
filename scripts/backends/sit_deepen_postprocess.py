"""Lower a seated pelvis while keeping the feet planted (2-bone leg IK).

Why this exists
---------------
"sit lower" used to be handled by the geometric *sit* overlay, which subtracts
0.28 m from the root and then lifts the body until the feet clear the floor.
Those two operations cancel, so the clip came back with the same pelvis height —
a silent no-op. Running it through diffusion instead produced a clip whose feet
were 2.5 body-heights underground (see ``motion_validity``).

The correct edit is a depth change: the pelvis goes down, the feet stay where
they are, and the knees bend to absorb the difference. That is a 2-bone IK
problem per leg, solved here in joint space so it cannot alter bone lengths.

Method
------
For each frame and each leg (hip → knee → ankle):

1. translate the whole body down by ``delta`` (the torso follows the pelvis);
2. restore the ankle and the foot to their original position (feet planted);
3. place the knee on the circle intersection of the two bone spheres
   (``|hip-knee| = L1``, ``|knee-ankle| = L2``), keeping the knee on the side it
   was already on so the leg does not flip through the body.

``delta`` is clamped per frame to the reachable distance ``L1 + L2`` and blended
with a smoothstep over the descent so there is no pop at the seated hold.

Usage::

    python scripts/backends/sit_deepen_postprocess.py --input 03_edited.npy \
        --out 03_edited.npy --delta 0.10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

J_PELVIS, J_HEAD = 0, 15
J_L_FOOT, J_R_FOOT = 10, 11
LEGS = (
    # (hip, knee, ankle, foot)
    (1, 4, 7, 10),
    (2, 5, 8, 11),
)
UPPER = (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)


def up_axis(j: np.ndarray) -> int:
    hp = j[:, J_HEAD] - j[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def _solve_knee(hip: np.ndarray, ankle: np.ndarray, knee0: np.ndarray,
                l1: float, l2: float) -> Optional[np.ndarray]:
    """Place the knee at the intersection of the thigh and shin spheres.

    Returns ``None`` when the hip is out of reach (caller clamps ``delta``), so
    the leg is never stretched.
    """
    axis = ankle - hip
    d = float(np.linalg.norm(axis))
    if d < 1e-6 or d > l1 + l2 - 1e-4 or d < abs(l1 - l2) + 1e-6:
        return None
    axis = axis / d
    # Distance from the hip to the circle centre along hip→ankle.
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = float(np.sqrt(max(l1 * l1 - a * a, 0.0)))
    centre = hip + a * axis
    # Pole vector: keep the knee on its original side of the hip→ankle line.
    pole = knee0 - hip
    pole = pole - float(np.dot(pole, axis)) * axis
    if float(np.linalg.norm(pole)) < 1e-6:
        # Degenerate (straight leg): pick a stable perpendicular.
        ref = np.array([0.0, 0.0, 1.0])
        pole = np.cross(axis, ref)
        if float(np.linalg.norm(pole)) < 1e-6:
            pole = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    return centre + h * _unit(pole)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def deepen_sit(joints: np.ndarray, delta: float = 0.10,
               contact_frame: Optional[int] = None,
               delta_per_frame: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return ``(joints, meta)`` with the seated pelvis lowered by ``delta``.

    ``delta_per_frame`` overrides the scalar: a per-frame drop (metres), used by
    the seated-hold clamp to remove pelvic rebound while the legs stay planted.
    """
    j = np.asarray(joints, dtype=np.float32)
    T = int(j.shape[0])
    if T < 10 or (delta_per_frame is None and delta <= 0):
        return j, {"delta": 0.0, "applied": False, "reason": "no-op"}
    up = up_axis(j)
    pelvis = j[:, J_PELVIS, up]
    # Seated = the part of the clip at/near the lowest pelvis. Ramp into it so
    # the descent keeps its timing and the hold is uniformly deeper.
    if contact_frame is None:
        contact_frame = int(np.argmin(pelvis))
    contact_frame = max(1, min(T - 1, int(contact_frame)))
    env = np.zeros(T, dtype=np.float32)
    env[:contact_frame] = _smoothstep(np.linspace(0.0, 1.0, contact_frame, endpoint=False))
    env[contact_frame:] = 1.0
    if delta_per_frame is not None:
        per_frame = np.asarray(delta_per_frame, dtype=np.float32).reshape(-1)
        if per_frame.size != T:
            raise ValueError(f"delta_per_frame must have {T} entries, got {per_frame.size}")
        env = per_frame

    drop = (np.asarray(env, dtype=np.float32) if delta_per_frame is not None
            else (float(delta) * env))

    out = j.copy()
    applied = np.zeros(T, dtype=np.float32)
    l1 = np.zeros((T, 2), dtype=np.float32)
    l2 = np.zeros((T, 2), dtype=np.float32)
    for k, (hip, knee, ankle, _foot) in enumerate(LEGS):
        l1[:, k] = np.linalg.norm(j[:, hip] - j[:, knee], axis=-1)
        l2[:, k] = np.linalg.norm(j[:, knee] - j[:, ankle], axis=-1)

    for t in range(T):
        want = float(drop[t])
        if want <= 1e-5:
            continue
        # Feasible drop for this frame: keep |hip-ankle| <= l1 + l2.
        allowed = want
        for k, (hip, knee, ankle, _foot) in enumerate(LEGS):
            hip_new = j[t, hip].copy()
            hip_new[up] -= want
            d = float(np.linalg.norm(hip_new - j[t, ankle]))
            room = l1[t, k] + l2[t, k] - 1e-4 - d
            if room < 0:
                allowed = min(allowed, max(0.0, want + room))
        if allowed <= 1e-5:
            continue
        frame = j[t].copy()
        frame[:, up] -= allowed
        solved = True
        for k, (hip, knee, ankle, foot) in enumerate(LEGS):
            frame[ankle] = j[t, ankle]
            frame[foot] = j[t, foot]
            new_knee = _solve_knee(frame[hip], frame[ankle], j[t, knee],
                                   float(l1[t, k]), float(l2[t, k]))
            if new_knee is None:
                solved = False
                break
            frame[knee] = new_knee
        if not solved:
            continue
        out[t] = frame
        applied[t] = allowed

    meta = {
        "applied": bool(applied.max() > 0),
        "delta_requested": round(float(delta) if delta_per_frame is None
                                 else float(np.max(env)), 4),
        "delta_applied_max": round(float(applied.max()), 4),
        "delta_applied_mean_hold": round(float(applied[contact_frame:].mean()), 4),
        "contact_frame": int(contact_frame),
        "up_axis": int(up),
        "frames": T,
        "clamped": bool(applied.max() < float(np.max(drop)) * 0.99),
    }
    return out, meta


def process_file(inp: Path, out: Path, delta: float = 0.10,
                 contact_frame: Optional[int] = None) -> Dict[str, Any]:
    raw = np.load(inp, allow_pickle=True)
    payload: Any = raw
    joints = None
    if raw.dtype == object:
        obj = raw.item() if raw.shape == () else raw
        if isinstance(obj, dict):
            for key in ("joint_positions", "joints", "pose_joints"):
                if obj.get(key) is not None:
                    joints = np.asarray(obj[key], dtype=np.float32)
                    break
            payload = obj
    if joints is None:
        joints = np.asarray(raw, dtype=np.float32)
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
    fixed, meta = deepen_sit(joints, delta=delta, contact_frame=contact_frame)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["joint_positions"] = fixed
    else:
        payload = fixed
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, payload)
    meta_path = out.with_suffix(".deepen.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"out": str(out), "meta": str(meta_path), **meta}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepen a seated pose with planted feet")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.10, help="metres to lower the pelvis")
    ap.add_argument("--contact-frame", type=int, default=None)
    args = ap.parse_args()
    res = process_file(Path(args.input), Path(args.out), args.delta, args.contact_frame)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
