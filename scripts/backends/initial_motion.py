"""
Build the four demo bases before any revise.

Walk and wave start from a recorded walk: straighten it, then wave one arm.
Sit and turn keep that same recorded pose sequence. A spin is removed, or a
short turn is brought up to half a circle. The pose itself is not redrawn
and is not replaced by another caption hit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from constraint_postprocess import (
    J_PELVIS,
    _horiz,
    _initial_forward,
    _rot_yaw,
    _up_axis,
    _yaw,
    straighten_yaw_and_path,
)

J_L_HIP, J_R_HIP = 1, 2
J_L_KNEE, J_R_KNEE = 4, 5
J_L_ANKLE, J_R_ANKLE = 7, 8
J_L_FOOT, J_R_FOOT = 10, 11
J_R_SHOULDER, J_L_SHOULDER = 17, 16
J_R_ELBOW, J_L_ELBOW = 19, 18
J_R_WRIST, J_L_WRIST = 21, 20

LEG = {
    "left": (J_L_HIP, J_L_KNEE, J_L_ANKLE, J_L_FOOT),
    "right": (J_R_HIP, J_R_KNEE, J_R_ANKLE, J_R_FOOT),
}
ARM = {
    "left": (J_L_SHOULDER, J_L_ELBOW, J_L_WRIST),
    "right": (J_R_SHOULDER, J_R_ELBOW, J_R_WRIST),
}
# Everything that should drop with the pelvis. Legs below the hip are IK'd.
UPPER = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

REPO = Path(__file__).resolve().parents[2]
SLOW_DONOR = REPO / "outputs" / "v05" / "session_more" / "slow" / "03_edited.npy"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _load_joints(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    obj = np.load(path, allow_pickle=True).item()
    joints = np.asarray(obj["joint_positions"], dtype=np.float32)
    return obj, joints


def _pelvis_speed(joints: np.ndarray) -> np.ndarray:
    up = _up_axis(joints)
    hz = _horiz(up)
    root = joints[:, J_PELVIS].astype(np.float64)
    return np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)


def _facing(joints: np.ndarray) -> np.ndarray:
    up = _up_axis(joints)
    fwd = _initial_forward(joints, up)
    vec = np.zeros(3, dtype=np.float64)
    hz = _horiz(up)
    vec[hz[0]] = fwd[0]
    vec[hz[1]] = fwd[1]
    return vec


def _right_dir(joints: np.ndarray, forward: np.ndarray) -> np.ndarray:
    up = _up_axis(joints)
    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up] = 1.0
    right = np.cross(forward, up_vec)
    n = float(np.linalg.norm(right))
    if n < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
        n = 1.0
    right = right / n
    shoulder = joints[0, J_R_SHOULDER] - joints[0, J_PELVIS]
    if float(np.dot(shoulder, right)) < 0:
        right = -right
    return right


def _ik(root: np.ndarray, target: np.ndarray, len_a: float, len_b: float, bend: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two-bone IK. Returns the mid joint and the reached end."""
    delta = target.astype(np.float64) - root.astype(np.float64)
    dist = float(np.linalg.norm(delta))
    lo = abs(len_a - len_b) + 1e-4
    hi = len_a + len_b - 1e-4
    dist = float(np.clip(dist, lo, hi))
    along = delta / (float(np.linalg.norm(delta)) + 1e-8)
    cos_a = float(np.clip((len_a * len_a + dist * dist - len_b * len_b) / (2.0 * len_a * dist), -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
    side = bend.astype(np.float64) - along * float(np.dot(bend, along))
    sn = float(np.linalg.norm(side))
    if sn < 1e-6:
        side = np.array([1.0, 0.0, 0.0]) - along * along[0]
        sn = float(np.linalg.norm(side)) + 1e-8
    side = side / sn
    mid = root + along * (len_a * cos_a) + side * (len_a * sin_a)
    end = root + along * dist
    return mid.astype(np.float32), end.astype(np.float32)


def prepare_walk(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Straighten the path. Return the walk and a quiet standing frame."""
    straight, info = straighten_yaw_and_path(joints)
    speed = _pelvis_speed(straight)
    peak = float(speed.max()) if speed.size else 0.0
    start = 0
    if peak > 1e-4:
        thr = 0.40 * peak
        while start < len(speed) - 12 and float(speed[start]) < thr:
            start += 1
    speed_full = _pelvis_speed(straight)
    quiet = int(np.argmin(speed_full)) if speed_full.size else 0
    stand = straight[quiet].copy()
    walk = straight[start:].astype(np.float32)
    info = dict(info)
    info["trim_start"] = int(start)
    info["T"] = int(walk.shape[0])
    return walk, stand, info


def _extend_walk(walk: np.ndarray) -> np.ndarray:
    """A few alternating steps, so the wave base is visibly walking."""
    from motion_compose import compose_motion

    extended, meta = compose_motion(walk, cycle_count=4, forbid=["yaw_spin"])
    if meta.get("method") != "compose_units":
        return walk
    return extended


def overlay_wave(walk: np.ndarray, *, side: str = "right") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep the walk. Raise one arm and wave it. The other arm stays."""
    j = np.asarray(walk, dtype=np.float32).copy()
    T = j.shape[0]
    up = _up_axis(j)
    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up] = 1.0
    forward = _facing(j)
    right = _right_dir(j, forward)
    if side == "left":
        right = -right
    sh_i, el_i, wr_i = ARM[side]
    len_a = float(np.linalg.norm(j[0, el_i] - j[0, sh_i]))
    len_b = float(np.linalg.norm(j[0, wr_i] - j[0, el_i]))
    reach = 0.86 * (len_a + len_b)
    for t in range(T):
        phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * 3.0 * t / max(T - 1, 1))
        opened = _smoothstep(np.array([min(1.0, t / 8.0)]))[0]
        height = 0.25 + 0.55 * phase
        outward = 0.55
        ahead = 0.12
        direction = up_vec * height + right * outward + forward * ahead
        direction = direction / (float(np.linalg.norm(direction)) + 1e-8)
        shoulder = j[t, sh_i].astype(np.float64)
        wave_wrist = shoulder + direction * reach
        orig = j[t, wr_i].astype(np.float64)
        target = (1.0 - opened) * orig + opened * wave_wrist
        elbow, wrist = _ik(shoulder, target, len_a, len_b, right)
        j[t, el_i] = elbow
        j[t, wr_i] = wrist
    return j, {"side": side, "T": int(T), "waves": 3}


def turn_in_place(stand: np.ndarray, *, frames: int = 80) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotate one standing pose halfway around, feet orbiting the pelvis."""
    up = _up_axis(stand[None, :, :])
    T = int(frames)
    out = np.repeat(stand[None, :, :], T, axis=0).astype(np.float32)
    center = stand[J_PELVIS].astype(np.float32)
    u = np.linspace(0.0, 1.0, T)
    # Start after a short hold so the turn is visible, end held.
    ang = _smoothstep(np.clip((u - 0.12) / 0.7, 0.0, 1.0)) * np.pi
    for t in range(T):
        out[t] = _rot_yaw(stand, float(ang[t]), center, up)
    return out, {"yaw_deg": 180.0, "T": T}


def sit_from_stand(stand: np.ndarray, *, frames: int = 90, drop: float = 0.40) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Lower the standing body onto a chair height. Feet stay put. No yaw."""
    up = _up_axis(stand[None, :, :])
    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up] = 1.0
    forward = _facing(stand[None, :, :])
    T = int(frames)
    out = np.repeat(stand[None, :, :], T, axis=0).astype(np.float32)
    planted = {side: stand[idx].astype(np.float64) for side, (_, _, _, idx) in LEG.items()}
    bones = {}
    for side, (hip, knee, ankle, foot) in LEG.items():
        bones[side] = (
            float(np.linalg.norm(stand[knee] - stand[hip])),
            float(np.linalg.norm(stand[ankle] - stand[knee])),
            (stand[foot] - stand[ankle]).astype(np.float64),
        )
    u = np.linspace(0.0, 1.0, T)
    amount = _smoothstep(np.clip((u - 0.08) / 0.55, 0.0, 1.0)) * float(drop)
    for t in range(T):
        pose = stand.astype(np.float64).copy()
        pose[UPPER, up] -= amount[t]
        for side, (hip, knee, ankle, foot) in LEG.items():
            len_a, len_b, foot_off = bones[side]
            target_ankle = planted[side] - foot_off
            mid, end = _ik(pose[hip], target_ankle, len_a, len_b, forward + up_vec * 0.15)
            pose[knee] = mid
            pose[ankle] = end
            pose[foot] = end + foot_off
        out[t] = pose.astype(np.float32)
    return out, {"drop": float(drop), "T": T}


def _facts(joints: np.ndarray) -> Dict[str, float]:
    """Motion measures. Yaw is removed before counting limb travel."""
    j = np.asarray(joints, dtype=np.float64)
    up = _up_axis(j)
    hz = _horiz(up)
    shoulder = j[:, J_R_SHOULDER] - j[:, J_L_SHOULDER]
    yaw = np.unwrap(np.arctan2(shoulder[:, hz[1]], shoulder[:, hz[0]]))
    end_yaw = float(np.degrees(yaw[-1] - yaw[0]))
    pelvis = j[:, J_PELVIS, up]
    feet = j[:, [J_L_FOOT, J_R_FOOT], up]
    pf = pelvis[:, None] - feet
    drop = float(pelvis[0] - pelvis.min())
    travel = float(np.linalg.norm(j[-1, J_PELVIS, hz] - j[0, J_PELVIS, hz]))
    radii = []
    for index in (J_L_FOOT, J_R_FOOT):
        delta = j[:, index, hz] - j[0, index, hz]
        radii.append(float(np.linalg.norm(delta, axis=-1).max()))
    rel = j - j[:, J_PELVIS : J_PELVIS + 1, :]
    face = yaw - yaw[0]
    cosine, sine = np.cos(-face), np.sin(-face)
    moving = 0
    arm = 0.0
    knee = 0.0
    arm_n = 0
    knee_n = 0
    for joint in range(1, 22):
        local = rel[:, joint].copy()
        x = local[:, hz[0]].copy()
        y = local[:, hz[1]].copy()
        local[:, hz[0]] = cosine * x - sine * y
        local[:, hz[1]] = sine * x + cosine * y
        path = float(np.linalg.norm(np.diff(local, axis=0), axis=-1).sum())
        if path > 0.25:
            moving += 1
        if joint in (16, 17, 18, 19, 20, 21):
            arm += path
            arm_n += 1
        if joint in (J_L_KNEE, J_R_KNEE):
            knee += path
            knee_n += 1
    return {
        "T": float(j.shape[0]),
        "yaw": end_yaw,
        "drop": drop,
        "travel": travel,
        "drift": max(radii) if radii else 0.0,
        "pf0": float(pf[0].mean()),
        "pf1": float(pf[-1].mean()),
        "moving": float(moving),
        "arm": arm / max(arm_n, 1),
        "knee": knee / max(knee_n, 1),
    }


def _trim_sit(joints: np.ndarray) -> np.ndarray:
    """Keep the descent and a short hold. Drop the long still tail."""
    j = np.asarray(joints)
    up = _up_axis(j)
    height = j[:, J_PELVIS, up]
    fallen = float(height[0] - height.min())
    if fallen < 1e-3:
        return j
    depth = height[0] - height
    begin = int(np.argmax(depth > 0.08 * fallen))
    contact = int(np.argmax(depth > 0.85 * fallen))
    start = max(0, begin - 12)
    end = min(j.shape[0], contact + 28)
    if end - start < 16:
        return j
    return j[start:end]


def _trim_turn(joints: np.ndarray) -> np.ndarray:
    """Stop soon after the body has turned halfway. Keep the steps inside it."""
    j = np.asarray(joints)
    up = _up_axis(j)
    hz = _horiz(up)
    shoulder = j[:, J_R_SHOULDER] - j[:, J_L_SHOULDER]
    yaw = np.unwrap(np.arctan2(shoulder[:, hz[1]], shoulder[:, hz[0]]))
    yaw = np.degrees(yaw - yaw[0])
    hit = np.where(np.abs(yaw) >= 160.0)[0]
    end = int(hit[0]) + 18 if len(hit) else j.shape[0]
    end = min(j.shape[0], max(end, 16))
    return j[:end]


def _passes(kind: str, facts: Dict[str, float]) -> bool:
    if kind == "sit":
        return (
            facts["drop"] >= 0.25
            and 0.45 <= facts["pf1"] <= 0.70
            and abs(facts["yaw"]) < 25.0
            and facts["drift"] < 0.20
            and facts["travel"] < 0.35
            and facts["moving"] >= 12
            and facts["knee"] > 0.40
            and facts["arm"] > 0.50
        )
    return (
        160.0 <= abs(facts["yaw"]) <= 200.0
        and facts["drop"] < 0.15
        and facts["travel"] < 0.55
        and facts["drift"] < 0.90
        and facts["moving"] >= 8
        and facts["arm"] > 0.40
    )


def _score(kind: str, facts: Dict[str, float]) -> float:
    if kind == "sit":
        return facts["arm"] + facts["knee"] + facts["moving"] / 10.0 - facts["drift"] * 2.0
    gap = abs(abs(facts["yaw"]) - 180.0)
    return facts["arm"] + facts["moving"] / 10.0 - gap / 15.0 - facts["drift"] - facts["travel"]


def recorded_action(action: str, reference: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pick a recorded clip that already performs the action. Do not redraw it."""
    from dataset_edit import _caption_index, _compatible, align_to_source

    kind = "sit" if "sit" in (action or "") else "turn"
    hml = REPO / "HumanML3D" / "HumanML3D" / "new_joints"
    best_clip = None
    best_meta: Dict[str, Any] = {}
    best_score = -1e9
    checked = 0
    for row in _caption_index():
        caption = row["caption"].lower()
        if kind == "sit":
            if not any(phrase in caption for phrase in ("sits down", "sitting down", "sit down")):
                continue
            if any(word in caption for word in ("sit up", "stand up", "get up", "gets up", "kneel", "cross", "jump", "dance", "crawl")):
                continue
        else:
            if "turn" not in caption:
                continue
            if any(word in caption for word in ("walk", "stair", "step", "run", "jump", "dance", "sit", "crawl", "climb", "limp")):
                continue
        path = hml / f"{row['id']}.npy"
        if not path.is_file():
            continue
        raw = np.asarray(np.load(path), dtype=np.float32)
        aligned, _converted = align_to_source(raw, reference)
        ok, _why = _compatible(aligned, reference, check_speed=False)
        checked += 1
        if not ok:
            continue
        facts = _facts(aligned)
        if not _passes(kind, facts):
            continue
        score = _score(kind, facts)
        if score > best_score:
            best_score = score
            trimmed = _trim_sit(aligned) if kind == "sit" else _trim_turn(aligned)
            best_clip = trimmed.astype(np.float32)
            best_meta = {
                "kind": "recorded_sit" if kind == "sit" else "recorded_turn",
                "source_id": row["id"],
                "caption": row["caption"],
                "score": round(float(score), 3),
                **{key: round(float(val), 3) for key, val in _facts(trimmed).items()},
            }
    if best_clip is None:
        raise RuntimeError(f"no recorded {kind} passed the motion checks ({checked} captions)")
    return best_clip, best_meta


def _add_missing_yaw(joints: np.ndarray, target_deg: float = 180.0) -> np.ndarray:
    """Rotate the whole pose sequence the rest of the way. Limbs stay in the body."""
    j = np.asarray(joints, dtype=np.float32).copy()
    up = _up_axis(j)
    yaw = np.unwrap(_yaw(j, up))
    have = float(yaw[-1] - yaw[0])
    if abs(have) < 1e-3:
        extra = float(np.radians(target_deg))
    else:
        extra = float(np.sign(have) * np.radians(target_deg) - have)
    u = np.linspace(0.0, 1.0, j.shape[0])
    ramp = (u * u * (3.0 - 2.0 * u)) * extra
    for t in range(j.shape[0]):
        j[t] = _rot_yaw(j[t], float(ramp[t]), j[t, J_PELVIS], up)
    return j


def repair_performance(action: str, joints: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct one measured fault. If the correction deletes limb motion, keep the clip."""
    src = np.asarray(joints, dtype=np.float32)
    before = _facts(src)
    yaw = np.unwrap(_yaw(src, _up_axis(src)))
    span = float(np.degrees(np.ptp(yaw)))
    end = float(np.degrees(yaw[-1] - yaw[0]))
    out = src
    change = "kept"
    if "sit" in (action or "") and span > 45.0:
        out, _ymeta = straighten_yaw_and_path(src)
        change = "remove_spin"
    elif "turn" in (action or "") and abs(end) < 150.0 and before["moving"] >= 8:
        out = _add_missing_yaw(src, 180.0)
        change = "complete_yaw"
    after = _facts(out)
    limb_before = before["arm"] + before["knee"]
    limb_after = after["arm"] + after["knee"]
    if after["moving"] + 2 < before["moving"] or limb_after < 0.6 * max(limb_before, 1e-6):
        out = src
        after = before
        change = "reverted_loss_of_motion"
    meta = {
        "kind": "repair_performance",
        "change": change,
        "before_moving": before["moving"],
        "after_moving": after["moving"],
        "before_yaw": round(end, 2),
        "yaw_span_before": round(span, 2),
        "after_yaw": round(after["yaw"], 2),
        "before_arm": round(before["arm"], 3),
        "after_arm": round(after["arm"], 3),
        "before_knee": round(before["knee"], 3),
        "after_knee": round(after["knee"], 3),
        "drop": round(after["drop"], 3),
        "travel": round(after["travel"], 3),
        "T": int(out.shape[0]),
    }
    return out.astype(np.float32), meta


def build_initial(
    action: str,
    joints: np.ndarray,
    *,
    walk_donor: Optional[np.ndarray] = None,
    stand_pose: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the base clip for one of the four demo actions."""
    action = action or ""
    if action in ("turn_in_place", "turn", "sit_down", "sit"):
        clip, extra = repair_performance(action, joints)
        return clip.astype(np.float32), extra
    donor = joints if walk_donor is None else walk_donor
    walk, stand, walk_info = prepare_walk(donor)
    if stand_pose is not None:
        stand = np.asarray(stand_pose, dtype=np.float32)
    if action in ("wave_right", "wave_left", "wave"):
        body = _extend_walk(walk)
        side = "left" if action == "wave_left" else "right"
        clip, extra = overlay_wave(body, side=side)
        kind = "walk_plus_one_arm"
    else:
        clip = _extend_walk(walk)
        extra = {"T": int(clip.shape[0])}
        kind = "straight_walk"
    meta = {"kind": kind, "walk": {k: walk_info.get(k) for k in ("yaw_span_before_deg", "yaw_span_after_deg", "path_lateral_before", "path_lateral_after", "trim_start", "T")}, **extra}
    return clip.astype(np.float32), meta


def apply_initial_file(path: Path, action: str, *, donor_path: Optional[Path] = None) -> Dict[str, Any]:
    """Replace joint_positions in an existing npy with the prepared base."""
    obj, joints = _load_joints(path)
    donor = None
    donor_obj: Dict[str, Any] = {}
    donor_file = donor_path or SLOW_DONOR
    if action not in ("walk_slow", "walk_forward") and donor_file.is_file() and donor_file.resolve() != path.resolve():
        donor_obj, donor = _load_joints(donor_file)
    elif action not in ("walk_slow", "walk_forward"):
        donor = joints
    stand_pose = donor_obj.get("stand_pose") if isinstance(donor_obj.get("stand_pose"), np.ndarray) else None
    clip, meta = build_initial(
        action,
        joints,
        walk_donor=None if action in ("walk_slow", "walk_forward") else donor,
        stand_pose=None if action in ("walk_slow", "walk_forward") else stand_pose,
    )
    obj["joint_positions"] = clip
    obj["initial_quality"] = meta
    if action in ("walk_slow", "walk_forward"):
        _, stand, _ = prepare_walk(joints)
        obj["stand_pose"] = stand
    np.save(path, obj)
    side = path.with_name("03_edited.initial.json")
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
