"""Physical validity of a joint clip, shared by the gate and the agent.

Two layers used to disagree about what "broken" means: ``quality_gate`` passed
clips whose feet were 2.5 body-heights underground (it never looked at the
floor), while the agent's own tool layer rolled them back. This module is the
single definition both import.

It also separates two situations that look identical in a global minimum but
need opposite handling:

``offset``
    The whole clip is rigidly displaced (``per-frame lowest joint`` is nearly
    constant and negative). Typical cause: a restored/diffused source whose
    root is at the origin instead of on the floor. A single vertical shift is
    the correct repair and cannot distort the motion.

``penetration``
    Only some frames go through the floor (the per-frame lowest joint varies).
    That is an articulation bug and must be rejected, never silently lifted,
    because lifting would hide it.

Usage::

    from motion_validity import physical_report, reground

    rep = physical_report(joints)          # {"valid": bool, "ground": {...}}
    fixed, off = reground(joints)          # only repairs mode == "offset"
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

J_PELVIS, J_HEAD = 0, 15
J_L_FOOT, J_R_FOOT = 10, 11

BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
)

# Thresholds are in body-heights (head→pelvis distance), so they do not depend
# on the source's metric scale.
MAX_PENETRATION = 0.10
MAX_BONE_CV = 0.08
# A rigid displacement is only "repairable" if the per-frame floor estimate is
# this flat; otherwise the clip is deforming through the floor.
OFFSET_FLATNESS = 0.03
MIN_OFFSET = 0.02


def up_axis(j: np.ndarray) -> int:
    hp = j[:, J_HEAD] - j[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def body_height(j: np.ndarray, up: Optional[int] = None) -> float:
    up = up_axis(j) if up is None else up
    h = float(np.linalg.norm(j[:, J_HEAD] - j[:, J_PELVIS], axis=-1).mean())
    return h if h > 1e-6 else 1.0


def bone_cv(j: np.ndarray) -> float:
    cvs = []
    for a, b in BONES:
        L = np.linalg.norm(j[:, a] - j[:, b], axis=-1)
        m = float(L.mean())
        if m > 1e-6:
            cvs.append(float(L.std()) / m)
    return float(np.mean(cvs)) if cvs else 0.0


def per_frame_floor(j: np.ndarray, up: Optional[int] = None) -> np.ndarray:
    """Lowest joint height at every frame (the clip's implicit floor)."""
    up = up_axis(j) if up is None else up
    return j[:, :, up].min(axis=1)


def offset_to_ground(j: np.ndarray, up: Optional[int] = None,
                     min_offset: float = 0.05) -> float:
    """Rigid lift that puts the *start* of the clip on the floor.

    MotionsFix-native and restored sources can come back with their root at the
    origin instead of on the floor (pelvis ≈ 0, feet ≈ −0.8 m). The first few
    frames are before the action starts, so their lowest joint is the floor; if
    it is clearly below 0 the whole clip is displaced and one vertical shift
    fixes it. A real articulation failure has a floor of ≈0 at the start, so it
    is not touched.
    """
    floor = per_frame_floor(j, up)
    if floor.size == 0:
        return 0.0
    head = float(np.median(floor[: max(3, len(floor) // 10)]))
    return -head if head < -min_offset else 0.0


def ground_report(j: np.ndarray, up: Optional[int] = None) -> Dict[str, Any]:
    up = up_axis(j) if up is None else up
    h = body_height(j, up)
    floor = per_frame_floor(j, up)
    lowest = float(per_frame_floor(j, up).min())
    spread = float(np.std(floor))
    penetration = float(max(0.0, -lowest) / h)
    if penetration <= 1e-3:
        mode, offset = "ok", 0.0
    elif spread <= OFFSET_FLATNESS and -float(np.mean(floor)) >= MIN_OFFSET:
        mode = "offset"
        offset = -float(np.mean(floor))
    else:
        mode, offset = "penetration", 0.0
    return {
        "mode": mode,
        "offset": round(offset, 4),
        "penetration": round(penetration, 4),
        "penetration_raw": round(float(max(0.0, -lowest)), 4),
        "floor_spread": round(spread, 4),
        "floor_start": round(float(floor[0]), 4),
        "height": round(h, 4),
        "up_axis": int(up),
        "fixable": mode == "offset",
    }


def reground(j: np.ndarray, up: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Rigidly lift a uniformly offset clip so its floor sits at y=0.

    Returns ``(joints, offset)``; the clip is returned unchanged when the
    problem is not a rigid offset (deforming penetration is not repaired here).
    """
    rep = ground_report(j, up)
    if rep["mode"] != "offset":
        return j, 0.0
    axis = int(rep["up_axis"])
    out = np.array(j, dtype=np.float32, copy=True)
    out[:, :, axis] = out[:, :, axis] + float(rep["offset"])
    return out, float(rep["offset"])


def physical_report(j: np.ndarray, up: Optional[int] = None) -> Dict[str, Any]:
    """Everything both layers need to decide whether a clip may be shown."""
    up = up_axis(j) if up is None else up
    ground = ground_report(j, up)
    finite = bool(np.isfinite(j).all())
    bcv = bone_cv(j)
    checks = {
        "finite": finite,
        # Mode only says whether the offset is repairable; the threshold decides
        # validity, so a clip that dips 0.5 cm under the floor stays usable.
        "ground": ground["penetration"] <= MAX_PENETRATION,
        "bone_rigid": bcv <= MAX_BONE_CV,
        "min_frames": int(j.shape[0]) >= 30,
    }
    invalid = [k for k, ok in checks.items() if not ok]
    return {
        "valid": not invalid,
        "invalid_checks": invalid,
        "ground": ground,
        "bone_cv": round(bcv, 5),
        "frames": int(j.shape[0]),
        "ground_fixable": ground["mode"] == "offset",
    }
