"""
Geometric postprocess for one clip (no diffusion, no dataset swap).

Currently:
  yaw_spin  — remove accumulated body yaw vs frame 0, and drop sideways drift so
              the root path lies on the initial facing line.
  limb_match / arm_raise — move one limb on the current clip only:
              * limb_match: raise the weaker side until it matches the other.
              * arm_raise:  raise one arm to an absolute height (above the
                shoulder / overhead) — this is what "raise the right arm high"
                means when the clip has no reference arm to copy.

  python scripts/backends/constraint_postprocess.py \\
      --in 03_edited.npy --out 03_edited.npy --intent-json plan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

J_PELVIS = 0
J_HEAD = 15
J_L_SHOULDER, J_R_SHOULDER = 16, 17

# hip, knee, ankle, foot — and shoulder, elbow, wrist. Same order for every action.
_LIMB_CHAINS = {
    "leg": {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)},
    "arm": {"left": (16, 18, 20), "right": (17, 19, 21)},
}

# How high the wrist should end up, as a multiple of the shoulder→head vertical
# span (head span is ~0.19 in these clips, so 1.0 = wrist level with the head,
# 2.6 = arm straight up, which is what the wave clips already show).
_ARM_RAISE_HEIGHT = {"slightly": 0.8, "mid": 1.2, "high": 1.8, "overhead": 2.6}

# How much of the elbow flexion to remove, 0 = keep the pose, 1 = fully straight.
# A raise with the elbow still folded reads as "hand near the shoulder", so the
# default unfolds the arm as it goes up.
_ARM_RAISE_STRAIGHTEN = {"slightly": 0.35, "mid": 0.5, "high": 0.7, "overhead": 0.85}

# Forbids this module can actually change. Others stay quality-only.
HANDLED = {"yaw_spin"}


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def _horiz(up: int) -> List[int]:
    return [0, 1] if up == 2 else [0, 2]


def _yaw(joints: np.ndarray, up: int) -> np.ndarray:
    hz = _horiz(up)
    sh = joints[:, J_R_SHOULDER] - joints[:, J_L_SHOULDER]
    hips = joints[:, 2] - joints[:, 1]
    sh_h = np.linalg.norm(sh[:, hz], axis=-1).mean()
    hip_h = np.linalg.norm(hips[:, hz], axis=-1).mean()
    v = sh if sh_h >= hip_h else hips
    return np.arctan2(v[:, hz[1]], v[:, hz[0]])


def _rot_yaw(points: np.ndarray, angle: float, center: np.ndarray, up: int) -> np.ndarray:
    out = points.copy()
    hz = _horiz(up)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x = points[:, hz[0]] - center[hz[0]]
    y = points[:, hz[1]] - center[hz[1]]
    out[:, hz[0]] = center[hz[0]] + c * x - s * y
    out[:, hz[1]] = center[hz[1]] + s * x + c * y
    return out


def _path_lateral(joints: np.ndarray, up: int) -> float:
    hz = _horiz(up)
    root = joints[:, J_PELVIS]
    delta = root[-1, hz] - root[0, hz]
    travel = float(np.linalg.norm(delta))
    rel = root[:, hz] - root[0, hz]
    if travel < 1e-4:
        return float(np.linalg.norm(rel, axis=-1).max()) if len(rel) else 0.0
    direction = delta / travel
    along = rel @ direction
    lat = rel - along[:, None] * direction
    return float(np.linalg.norm(lat, axis=-1).max())


def _initial_forward(joints: np.ndarray, up: int) -> np.ndarray:
    """Unit horizontal facing at frame 0, signed to match early root motion."""
    hz = _horiz(up)
    sh = joints[0, J_R_SHOULDER] - joints[0, J_L_SHOULDER]
    axis = np.array([float(sh[hz[0]]), float(sh[hz[1]])], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        axis = np.array([1.0, 0.0])
    else:
        axis = axis / n
    cands = [np.array([-axis[1], axis[0]]), np.array([axis[1], -axis[0]])]
    T = joints.shape[0]
    k = min(T - 1, max(4, T // 5))
    vel = joints[k, J_PELVIS, hz] - joints[0, J_PELVIS, hz]
    vn = float(np.linalg.norm(vel))
    if vn < 1e-4:
        return cands[0]
    vel = vel / vn
    best = max(cands, key=lambda c: float(np.dot(c, vel)))
    return best / max(float(np.linalg.norm(best)), 1e-8)


def straighten_yaw_and_path(joints: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep the starting facing and move the root only along that line."""
    j = np.asarray(joints, dtype=np.float32).copy()
    T = j.shape[0]
    up = _up_axis(j)
    hz = _horiz(up)
    yaw = np.unwrap(_yaw(j, up))
    yaw0 = float(yaw[0])
    before_yaw = float(np.degrees(np.ptp(yaw)))
    before_lat = _path_lateral(j, up)

    for t in range(T):
        delta = float(yaw[t] - yaw0)
        if abs(delta) < 1e-4:
            continue
        j[t] = _rot_yaw(j[t], -delta, j[t, J_PELVIS], up)

    forward = _initial_forward(j, up)
    origin = j[0, J_PELVIS].astype(np.float64)
    for t in range(T):
        d = j[t, J_PELVIS].astype(np.float64) - origin
        horiz = np.array([d[hz[0]], d[hz[1]]])
        along = float(np.dot(horiz, forward))
        lateral = horiz - along * forward
        if float(np.linalg.norm(lateral)) < 1e-6:
            continue
        j[t, :, hz[0]] = j[t, :, hz[0]] - np.float32(lateral[0])
        j[t, :, hz[1]] = j[t, :, hz[1]] - np.float32(lateral[1])

    yaw_after = np.unwrap(_yaw(j, up))
    return j, {
        "yaw_span_before_deg": round(before_yaw, 2),
        "yaw_span_after_deg": round(float(np.degrees(np.ptp(yaw_after))), 2),
        "path_lateral_before": round(before_lat, 4),
        "path_lateral_after": round(_path_lateral(j, up), 4),
        "up_axis": up,
    }


def apply_constraints(
    joints: np.ndarray,
    forbid: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    forbid = [str(x) for x in (forbid or [])]
    j = np.asarray(joints, dtype=np.float32).copy()
    meta: Dict[str, Any] = {
        "forbid": forbid,
        "handled": sorted(HANDLED.intersection(forbid)),
        "T": int(j.shape[0]),
    }
    if "yaw_spin" in forbid:
        j, ymeta = straighten_yaw_and_path(j)
        meta["yaw_spin"] = ymeta
    return j, meta


def _rot_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    axis, ang = _axis_angle(src, dst)
    if ang < 1e-8:
        return np.eye(3, dtype=np.float64)
    return _rot_axis_angle(axis, ang)


def _axis_angle(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Axis and angle taking unit-ish ``src`` onto ``dst``.

    Handles the antiparallel case explicitly: a hand hanging down and an aim
    straight up are 180° apart, where the cross product vanishes. Without the
    fallback axis the caller silently gets *no* rotation on those frames, which
    showed up as a one-frame jump in the sit clip.
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    a = a / na
    b = b / nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s > 1e-6:
        return axis / s, float(np.arccos(c))
    if c > 0.0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    # Exactly opposite: any perpendicular axis is a valid 180° turn.
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, helper)
    return axis / (np.linalg.norm(axis) + 1e-12), float(np.pi)


def _rot_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    if abs(ang) < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / (np.linalg.norm(axis) + 1e-12)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(ang) * k + (1.0 - np.cos(ang)) * (k @ k)


def match_paired_lift(joints: np.ndarray, spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Raise the weaker side's swing until its lift matches the other side.

    The hip or shoulder stays put. Knee, ankle and foot rotate together,
    so the limb length does not change.
    """
    pair = str(spec.get("pair") or "leg")
    chains = _LIMB_CHAINS.get(pair) or _LIMB_CHAINS["leg"]
    up = _up_axis(joints)
    out = np.asarray(joints, dtype=np.float32).copy()

    def _amp(idx: int) -> float:
        z = out[:, idx, up]
        return float(np.percentile(z, 95) - np.percentile(z, 15))

    left_amp = _amp(chains["left"][-1])
    right_amp = _amp(chains["right"][-1])
    target = str(spec.get("target") or "weaker")
    ref = str(spec.get("reference") or "stronger")
    if target == "weaker" or target not in chains:
        target = "left" if left_amp <= right_amp else "right"
        ref = "right" if target == "left" else "left"
    if ref not in chains or ref == target:
        ref = "right" if target == "left" else "left"
    t_chain = chains[target]
    r_chain = chains[ref]
    foot = t_chain[-1]
    base = float(np.percentile(out[:, foot, up], 15))
    t_amp = _amp(foot)
    r_amp = _amp(r_chain[-1])
    scale = 1.0 if t_amp < 1e-4 else float(np.clip(r_amp / t_amp, 1.0, 2.5))
    if scale <= 1.02:
        return out, {
            "pair": pair,
            "target": target,
            "reference": ref,
            "target_lift": round(t_amp, 4),
            "reference_lift": round(r_amp, 4),
            "scale": 1.0,
            "changed": False,
            "stages": _limb_stages(pair, target, ref, t_amp, t_amp, r_amp),
        }
    for t in range(out.shape[0]):
        hip = out[t, t_chain[0]].astype(np.float64)
        foot_p = out[t, foot].astype(np.float64)
        excess = float(foot_p[up] - base)
        if excess <= 0.01:
            continue
        v = foot_p - hip
        length = float(np.linalg.norm(v))
        if length < 1e-4:
            continue
        new_up = float(v[up]) + excess * (scale - 1.0)
        remain = length * length - new_up * new_up
        if remain < 1e-8:
            new_up = float(np.sign(new_up) or 1.0) * length * 0.98
            remain = max(0.0, length * length - new_up * new_up)
        horiz = v.copy()
        horiz[up] = 0.0
        hlen = float(np.linalg.norm(horiz))
        lifted = np.zeros(3, dtype=np.float64)
        if hlen > 1e-8:
            lifted += horiz * (np.sqrt(remain) / hlen)
        lifted[up] = new_up
        rot = _rot_from_to(v, lifted)
        for j in t_chain[1:]:
            offset = out[t, j].astype(np.float64) - hip
            out[t, j] = (hip + rot @ offset).astype(np.float32)
    after = _amp(foot)
    return out, {
        "pair": pair,
        "target": target,
        "reference": ref,
        "target_lift_before": round(t_amp, 4),
        "target_lift_after": round(after, 4),
        "reference_lift": round(r_amp, 4),
        "scale": round(scale, 3),
        "changed": True,
        "stages": _limb_stages(pair, target, ref, t_amp, after, r_amp),
    }


def raise_arm(joints: np.ndarray, spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Raise one arm to an absolute height on the current clip.

    Unlike ``match_paired_lift`` this needs no reference arm: the target is a
    wrist height relative to the shoulder. For every frame:

      1. unfold the elbow by ``straighten`` (keeps upper-arm and forearm lengths),
      2. rotate elbow+wrist about the shoulder so the wrist lands at the target
         height, going up and slightly outward.

    ``straighten`` is what makes the result read as "arm raised" instead of
    "hand lifted next to the shoulder". The raise ramps in over the first frames
    and holds, so it looks like a person walking with one arm up, not a snap.
    """
    side = str(spec.get("side") or "right").lower()
    if side not in ("left", "right"):
        side = "right"
    level = str(spec.get("level") or spec.get("target") or "high").lower()
    height_ratio = float(spec.get("height_ratio") or _ARM_RAISE_HEIGHT.get(level, 1.8))
    straighten = float(
        spec.get("straighten")
        if spec.get("straighten") is not None
        else _ARM_RAISE_STRAIGHTEN.get(level, 0.7)
    )
    straighten = float(np.clip(straighten, 0.0, 1.0))
    step_up = bool(spec.get("step_up"))

    up = _up_axis(joints)
    out = np.asarray(joints, dtype=np.float32).copy()
    chain = _LIMB_CHAINS["arm"][side]
    shoulder, elbow, wrist = chain

    def _lift(idx: int) -> np.ndarray:
        return out[:, idx, up] - out[:, shoulder, up]

    before = float(np.percentile(_lift(wrist), 90))
    span = float(np.percentile(out[:, J_HEAD, up] - out[:, shoulder, up], 90))
    span = max(span, 0.05)
    # Anatomical reach (upper arm + forearm, fully unfolded) — the hard ceiling.
    reach = float(
        np.percentile(
            np.linalg.norm(out[:, elbow] - out[:, shoulder], axis=-1)
            + np.linalg.norm(out[:, wrist] - out[:, elbow], axis=-1),
            90,
        )
    )
    target_lift = height_ratio * span
    if step_up:
        # "a bit higher" on a clip that is already raised: add one visible step
        # instead of returning the same clip. The step is a fraction of the head
        # span, so it stays a nudge rather than a jump to straight up.
        target_lift = max(target_lift, before + max(0.04, 0.35 * span))
    clamped = False
    if target_lift > 0.94 * reach:
        target_lift = 0.94 * reach
        clamped = True
    if before >= target_lift * 0.92:
        return out, {
            "side": side,
            "level": level,
            "changed": False,
            "wrist_lift_before": round(before, 4),
            "wrist_lift_target": round(target_lift, 4),
            "height_ratio": height_ratio,
            "note": "arm already at the requested height; clip untouched",
        }

    T = out.shape[0]
    # ~1 s ramp at 20 fps: fast enough to read as a raise, slow enough not to snap.
    ramp = max(6, min(20, T // 4)) if T > 8 else max(1, T // 2)

    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up] = 1.0
    for t in range(T):
        sh = out[t, shoulder].astype(np.float64)
        el = out[t, elbow].astype(np.float64)
        wr = out[t, wrist].astype(np.float64)

        # Ramp the change in, then hold for the rest of the clip.
        w = 0.5 - 0.5 * np.cos(np.pi * (t / ramp)) if t < ramp else 1.0

        upper = el - sh
        fore = wr - el
        ul = float(np.linalg.norm(upper))
        fl = float(np.linalg.norm(fore))
        if ul < 1e-5 or fl < 1e-5:
            continue

        # 1) Unfold the elbow: blend the forearm direction toward the upper-arm
        #    direction. Bone lengths are untouched, only the joint angle changes.
        s = straighten * w
        d = (1.0 - s) * (fore / fl) + s * (upper / ul)
        dn = float(np.linalg.norm(d))
        d = fore / fl if dn < 1e-6 else d / dn
        wr2 = el + d * fl

        v = wr2 - sh
        length = float(np.linalg.norm(v))
        if length < 1e-5:
            continue

        # Outward = the body's left→right axis. Using the shoulder line instead of
        # shoulder-minus-pelvis matters: when the torso is upright that horizontal
        # part is tiny, so its direction is noise, and the swing plane (hence the
        # rotation axis) flipped between two frames — a visible one-frame jump.
        lat = out[t, J_R_SHOULDER].astype(np.float64) - out[t, J_L_SHOULDER].astype(np.float64)
        lat[up] = 0.0
        n = float(np.linalg.norm(lat))
        if n < 1e-4:
            lat = sh - out[t, J_PELVIS].astype(np.float64)
            lat[up] = 0.0
            n = float(np.linalg.norm(lat))
        outward = lat / n if n > 1e-6 else np.zeros(3)
        if side == "left":
            outward = -outward

        # 2) Aim the whole arm so the wrist ends at the requested height.
        cos_target = float(np.clip(target_lift / length, -1.0, 1.0))
        sin_target = float(np.sqrt(max(0.0, 1.0 - cos_target * cos_target)))
        aim = up_vec * cos_target + outward * sin_target
        if float(np.linalg.norm(aim)) < 1e-6:
            continue
        aim = aim / np.linalg.norm(aim)

        cur = v / length
        axis, ang = _axis_angle(cur, aim)
        if ang >= 1e-4:
            rot_w = _rot_axis_angle(axis, ang * float(w))
            el2 = sh + rot_w @ (el - sh)
            wr2 = sh + rot_w @ (wr2 - sh)
            out[t, elbow] = el2.astype(np.float32)
        out[t, wrist] = wr2.astype(np.float32)

    after = float(np.percentile(_lift(wrist), 90))
    after_ratio = after / max(span, 1e-6)
    return out, {
        "side": side,
        "level": level,
        "changed": True,
        "wrist_lift_before": round(before, 4),
        "wrist_lift_after": round(after, 4),
        "wrist_lift_target": round(target_lift, 4),
        "height_ratio": height_ratio,
        "ratio_after": round(after_ratio, 3),
        "straighten": round(straighten, 2),
        "reach_clamped": clamped,
        "ramp_frames": ramp,
        "stages": [
            {"step": 1, "name": "解析", "detail": f"把{'右' if side == 'right' else '左'}臂抬到目标高度，仍用当前片段。"},
            {"step": 2, "name": "定高", "detail": f"目标：手腕抬到肩以上 {target_lift:.3f}（约头高 {height_ratio:.2f} 倍，越过头顶约 2.6 倍）。"},
            {"step": 3, "name": "起点", "detail": f"当前手腕只高于肩 {before:.3f}。"},
            {"step": 4, "name": "松肘", "detail": f"先把肘部屈曲松开 {straighten * 100:.0f}%，上臂和前臂长度不变。"},
            {"step": 5, "name": "绕肩旋转", "detail": "肩不动，肘和腕一起转到目标高度，方向略向外。"},
            {"step": 6, "name": "渐变", "detail": f"前 {ramp} 帧渐入，之后保持。"},
            {"step": 7, "name": "结果", "detail": f"手腕抬到 {after:.3f}，约头高的 {after_ratio:.2f} 倍。"},
        ],
    }


def _blend_rotation(rot: np.ndarray, w: float) -> np.ndarray:
    """Identity→rot blend by weight w (matrix log/exp, small-angle safe)."""
    if w >= 0.9999:
        return rot
    if w <= 1e-4:
        return np.eye(3, dtype=np.float64)
    c = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-6:
        return np.eye(3, dtype=np.float64)
    s = float(np.sin(ang))
    axis = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    ) / (2.0 * s)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    a = ang * w
    return np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * (k @ k)


def _limb_stages(pair: str, target: str, ref: str, before: float, after: float, ref_amp: float) -> List[Dict[str, Any]]:
    side = "右" if target == "right" else "左"
    other = "左" if target == "right" else "右"
    limb = "腿" if pair == "leg" else "手臂"
    return [
        {"step": 1, "name": "解析", "detail": f"{side}{limb}的抬起向{other}{limb}对齐，仍用当前片段。"},
        {"step": 2, "name": "选参考", "detail": "参考就是画面上这一段里抬得更高的一侧。"},
        {"step": 3, "name": "数据集门", "detail": "不检索数据集，避免换成另一条动作。"},
        {"step": 4, "name": "拆分", "detail": f"{side}{limb}抬起 {before:.3f}，{other}{limb}抬起 {ref_amp:.3f}。"},
        {"step": 5, "name": "组合", "detail": f"只在离地的帧上，绕{side}侧根关节把{limb}抬到和另一侧接近。"},
        {"step": 6, "name": "测试", "detail": f"抬起 {before:.3f}→{after:.3f}，对照 {ref_amp:.3f}。"},
        {"step": 7, "name": "修改", "detail": "髋或肩不动，肢体一起转，长度保持。"},
        {"step": 8, "name": "输出", "detail": "通过。预览仍是这一条动作。"},
    ]


def process_file(
    path_in: Path,
    path_out: Path,
    *,
    forbid: Optional[List[str]] = None,
    limb_match: Optional[Dict[str, Any]] = None,
    arm_raise: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = np.load(path_in, allow_pickle=True)
    if raw.dtype != object:
        raise SystemExit(f"Expected object npy with joint_positions, got {raw.dtype}")
    obj = raw.item()
    if not isinstance(obj, dict) or "joint_positions" not in obj:
        raise SystemExit(f"Need joint_positions in {path_in}")
    joints = np.asarray(obj["joint_positions"], dtype=np.float32)
    fixed, meta = apply_constraints(joints, forbid=forbid)
    if limb_match:
        fixed, lmeta = match_paired_lift(fixed, limb_match)
        meta["limb_match"] = {k: v for k, v in lmeta.items() if k != "stages"}
        meta["stages"] = lmeta.get("stages") or []
        meta["handled"] = list(meta.get("handled") or []) + ["limb_match"]
    if arm_raise:
        if str(arm_raise.get("side") or "").lower() == "both":
            metas = []
            for side in ("left", "right"):
                fixed, one = raise_arm(fixed, {**arm_raise, "side": side})
                metas.append({k: v for k, v in one.items() if k != "stages"})
            meta["arm_raise"] = {"both": metas}
            meta["stages"] = [
                {"step": 1, "name": "解析", "detail": "双臂一起抬，仍用当前片段。"},
                {"step": 2, "name": "绕肩旋转", "detail": "两侧肩不动，肘腕随上臂转。"},
                {"step": 3, "name": "结果", "detail": "左右各抬一次，臂长不变。"},
            ]
        else:
            fixed, ameta = raise_arm(fixed, arm_raise)
            meta["arm_raise"] = {k: v for k, v in ameta.items() if k != "stages"}
            if ameta.get("stages"):
                meta["stages"] = ameta["stages"]
        meta["handled"] = list(meta.get("handled") or []) + ["arm_raise"]
    obj = dict(obj)
    obj["joint_positions"] = fixed
    if "pose" in obj:
        pose = np.asarray(obj["pose"], dtype=np.float32).copy()
        T = min(pose.shape[0], fixed.shape[0])
        pose[:T, :3] = fixed[:T, J_PELVIS]
        obj["pose"] = pose
    obj["constraint_meta"] = meta
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_out, obj, allow_pickle=True)
    side = path_out.with_suffix(".constraint.json")
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[constraint] handled={meta.get('handled')} "
        f"yaw={((meta.get('yaw_spin') or {}).get('yaw_span_before_deg'))}"
        f"→{((meta.get('yaw_spin') or {}).get('yaw_span_after_deg'))} "
        f"→ {path_out}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply refine.forbid geometric constraints")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--forbid", type=str, default="")
    ap.add_argument("--intent-json", type=Path, default=None)
    args = ap.parse_args()
    forbid = [x.strip() for x in (args.forbid or "").split(",") if x.strip()]
    limb_match = None
    arm_raise = None
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        refine = plan.get("refine") or (plan.get("revision") or {}).get("refine") or {}
        if refine.get("forbid"):
            forbid = [str(x) for x in refine["forbid"]]
        if isinstance(refine.get("limb_match"), dict):
            limb_match = refine["limb_match"]
        if isinstance(refine.get("arm_raise"), dict):
            arm_raise = refine["arm_raise"]
    if not HANDLED.intersection(forbid) and not limb_match and not arm_raise:
        # Nothing this module can change; copy through so the pipeline still has an out file.
        import shutil

        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.inp.resolve() != args.out.resolve():
            shutil.copy2(args.inp, args.out)
        print(f"[constraint] no handled forbid in {forbid}; copied")
        return
    process_file(
        args.inp, args.out, forbid=forbid, limb_match=limb_match, arm_raise=arm_raise
    )


if __name__ == "__main__":
    main()
