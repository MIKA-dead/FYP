"""
Motion clip timelines for UI / quality / trim.

- sit: preamble / descend / seated_hold / trailing
- wave (+ optional walk): walk_forward / wave_* / trailing
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

J_PELVIS, J_HEAD = 0, 15
J_L_WRIST, J_R_WRIST = 20, 21


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(hp[:, 2]).mean() > abs(hp[:, 1]).mean() else 1


def _horiz(up: int) -> List[int]:
    return [0, 1] if up == 2 else [0, 2]


def analyze_sit_timeline(
    joints: np.ndarray,
    *,
    forbid: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return phase ranges in frames [start, end) and problem flags."""
    forbid = set(forbid or [])
    j = np.asarray(joints, dtype=np.float32)
    T = int(j.shape[0])
    up = _up_axis(j)
    hz = _horiz(up)
    root = j[:, J_PELVIS]
    pz = root[:, up]

    # Contact = near min pelvis
    drop_total = float(pz[0] - pz.min())
    if drop_total < 1e-4:
        contact = max(T // 2, 1)
    else:
        target = pz[0] - 0.85 * drop_total
        hit = np.where(pz <= target + 1e-6)[0]
        contact = int(hit[0]) if len(hit) else int(np.argmin(pz))
    contact = int(np.clip(contact, int(0.2 * T), T - 2))

    # Sit start = when cumulative drop reaches ~15% OR horizontal motion slows
    # while pelvis begins falling.
    cum_drop = pz[0] - pz
    sit_start = 0
    thr = 0.15 * max(drop_total, 1e-3)
    for t in range(1, contact + 1):
        if cum_drop[t] >= thr:
            sit_start = t
            break
    # Also: last frame before contact where horiz step is still "walking"
    step = np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)
    # Walk-ish preamble: high travel in first half before sit_start
    pre_travel = float(np.linalg.norm(root[sit_start, hz] - root[0, hz])) if sit_start > 0 else 0.0
    # Refine sit_start: if early frames have lots of XY motion, keep them as preamble
    # Find when rolling horiz travel plateaus and drop accelerates
    if T > 10:
        roll_xy = np.cumsum(step)
        # Prefer later sit_start if drop is still tiny in early frames
        for t in range(sit_start, max(sit_start + 1, contact - 5)):
            if cum_drop[t] < 0.08 * max(drop_total, 1e-3) and roll_xy[min(t, len(roll_xy) - 1)] > 0.15:
                sit_start = t + 1
            else:
                break
    sit_start = int(np.clip(sit_start, 0, max(0, contact - 5)))

    # Hold end: shortly after contact; trailing = after hold
    hold_end = min(T, contact + max(12, int(0.25 * (T - contact + 1))))
    hold_end = int(np.clip(hold_end, contact + 1, T))

    pre_travel = float(np.linalg.norm(root[sit_start, hz] - root[0, hz]))
    post_wrist = 0.0
    if hold_end < T - 1:
        w = j[hold_end:, [20, 21]]
        if w.shape[0] > 1:
            post_wrist = float(np.linalg.norm(np.diff(w, axis=0), axis=-1).mean())

    problems: List[str] = []
    if pre_travel > 0.18 or sit_start > int(0.2 * T):
        problems.append("pre_sit_locomotion")
    if "pre_sit_locomotion" in forbid and pre_travel > 0.12:
        problems.append("pre_sit_locomotion")
    if T - hold_end > 15 and post_wrist > 0.02:
        problems.append("post_sit_extra")
    if "yaw_spin" in forbid:
        problems.append("yaw_spin")
    if "post_sit_limb_rebound" in forbid:
        problems.append("post_sit_limb_rebound")

    phases = [
        {
            "id": "preamble",
            "label": "坐下前（走动等）",
            "start": 0,
            "end": sit_start,
            "problem": sit_start > 3 and (
                "pre_sit_locomotion" in forbid
                or pre_travel > 0.18
                or sit_start > int(0.2 * T)
            ),
        },
        {
            "id": "descend",
            "label": "坐下下降",
            "start": sit_start,
            "end": contact,
            "problem": False,
        },
        {
            "id": "seated_hold",
            "label": "坐稳",
            "start": contact,
            "end": hold_end,
            "problem": False,
        },
        {
            "id": "trailing",
            "label": "坐下后多余",
            "start": hold_end,
            "end": T,
            "problem": hold_end < T - 5 and (
                "post_sit_extra" in forbid
                or "post_sit_limb_rebound" in forbid
                or "post_sit_extra" in problems
                or post_wrist > 0.025
            ),
        },
    ]
    # Drop empty phases for cleaner UI (keep zero-length as absent)
    phases_ui = [p for p in phases if p["end"] > p["start"]]

    return {
        "T": T,
        "up_axis": up,
        "kind": "sit",
        "sit_start": sit_start,
        "contact": contact,
        "hold_end": hold_end,
        "pre_travel": round(pre_travel, 4),
        "post_wrist_vel": round(post_wrist, 5),
        "pelvis_drop": round(drop_total, 4),
        "phases": phases_ui,
        "problems": sorted(set(problems)),
        "trim_suggestion": {
            "keep_start": sit_start if "pre_sit_locomotion" in forbid else 0,
            "keep_end": hold_end
            if ("post_sit_extra" in forbid or "post_sit_limb_rebound" in forbid)
            else T,
        },
    }


def analyze_wave_timeline(
    joints: np.ndarray,
    *,
    side: str = "right",
) -> Dict[str, Any]:
    """Rough walk + wave phases for UI (not as precise as sit)."""
    j = np.asarray(joints, dtype=np.float64)
    T = int(j.shape[0])
    up = _up_axis(j)
    hz = _horiz(up)
    wrist_i = J_R_WRIST if side != "left" else J_L_WRIST
    other_i = J_L_WRIST if wrist_i == J_R_WRIST else J_R_WRIST

    root = j[:, J_PELVIS]
    step = np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)
    step = np.concatenate([[0.0], step])
    travel = float(np.linalg.norm(root[-1, hz] - root[0, hz]))

    w = j[:, wrist_i, up]
    w_vel = np.abs(np.diff(w, prepend=w[:1]))
    o = j[:, other_i, up]
    o_vel = np.abs(np.diff(o, prepend=o[:1]))

    # Smooth-ish scores
    win = max(3, T // 20)
    def _ma(x: np.ndarray) -> np.ndarray:
        if len(x) < win:
            return x
        k = np.ones(win) / win
        return np.convolve(x, k, mode="same")

    walk_score = _ma(step)
    wave_score = _ma(w_vel)
    both_score = _ma(np.minimum(w_vel, o_vel))

    walk_thr = max(float(np.percentile(walk_score, 60)), 1e-4)
    wave_thr = max(float(np.percentile(wave_score, 55)), 1e-4)

    labels = np.array(["idle"] * T, dtype=object)
    labels[walk_score >= walk_thr] = "walk_forward"
    # Wave overrides when wrist motion dominates
    labels[(wave_score >= wave_thr) & (wave_score >= walk_score)] = "wave"

    # Compress to contiguous phases, then merge tiny fragments
    raw_phases: List[Dict[str, Any]] = []
    i = 0
    while i < T:
        lab = str(labels[i])
        j0 = i
        while i < T and str(labels[i]) == lab:
            i += 1
        pid = {
            "walk_forward": "walk_forward",
            "wave": "wave_right" if side != "left" else "wave_left",
            "idle": "idle",
        }.get(lab, lab)
        label_zh = {
            "walk_forward": "向前走",
            "wave_right": "右手挥手",
            "wave_left": "左手挥手",
            "idle": "停顿/其它",
        }.get(pid, pid)
        problem = False
        if pid.startswith("wave") and float(both_score[j0:i].mean()) > wave_thr * 0.85:
            problem = True  # both arms moving hard
            label_zh = label_zh + "（疑似双臂）"
        if pid == "walk_forward" and travel < 0.12:
            problem = True
            label_zh = "向前走（位移不足）"
        if pid == "idle" and (i - j0) > max(8, T // 5):
            problem = True
        raw_phases.append(
            {
                "id": pid,
                "label": label_zh,
                "start": int(j0),
                "end": int(i),
                "problem": problem,
            }
        )
    # Prefer fewer, readable segments on the UI bar
    phases = _merge_short_phases(raw_phases, min_len=max(6, T // 18))
    phases = _merge_short_phases(phases, min_len=max(8, T // 14))

    problems: List[str] = []
    if travel < 0.15:
        problems.append("weak_forward_travel")
    if float(both_score.mean()) > float(wave_score.mean()) * 0.75 and float(wave_score.max()) > 1e-4:
        problems.append("both_arms_waving")

    return {
        "T": T,
        "up_axis": up,
        "kind": "wave",
        "side": side,
        "root_travel": round(travel, 4),
        "phases": [p for p in phases if p["end"] > p["start"]],
        "problems": sorted(set(problems)),
        "trim_suggestion": {"keep_start": 0, "keep_end": T},
    }


def _merge_short_phases(
    phases: List[Dict[str, Any]], *, min_len: int = 5
) -> List[Dict[str, Any]]:
    """Absorb tiny fragments into neighbors to keep the UI bar readable."""
    if not phases:
        return []
    out = [dict(phases[0])]
    for ph in phases[1:]:
        cur = dict(ph)
        prev = out[-1]
        plen = int(cur["end"]) - int(cur["start"])
        if plen < min_len:
            # Merge into previous
            prev["end"] = cur["end"]
            prev["problem"] = bool(prev.get("problem") or cur.get("problem"))
            continue
        if prev["id"] == cur["id"]:
            prev["end"] = cur["end"]
            prev["problem"] = bool(prev.get("problem") or cur.get("problem"))
            continue
        out.append(cur)
    # Second pass: merge leading tiny into next
    if len(out) >= 2 and (out[0]["end"] - out[0]["start"]) < min_len:
        nxt = out[1]
        nxt["start"] = out[0]["start"]
        nxt["problem"] = bool(nxt.get("problem") or out[0].get("problem"))
        out = out[1:]
    return out


def analyze_turn_timeline(joints: np.ndarray) -> Dict[str, Any]:
    """In-place turn: wind-up / turning / settle."""
    j = np.asarray(joints, dtype=np.float64)
    T = int(j.shape[0])
    up = _up_axis(j)
    # Shoulder vector yaw rate as turn proxy
    sh = j[:, 17] - j[:, 16]
    yaw = np.arctan2(sh[:, 1], sh[:, 0])
    dyaw = np.abs(np.diff(np.unwrap(yaw), prepend=yaw[:1]))
    thr = max(float(np.percentile(dyaw, 65)), 1e-3)
    turning = dyaw >= thr
    # Find main turn span
    if turning.any():
        idx = np.where(turning)[0]
        t0, t1 = int(idx[0]), int(idx[-1]) + 1
    else:
        t0, t1 = T // 3, (2 * T) // 3
    phases = [
        {"id": "prep", "label": "准备", "start": 0, "end": t0, "problem": False},
        {
            "id": "turning",
            "label": "转身",
            "start": t0,
            "end": t1,
            "problem": float(np.rad2deg(dyaw.sum())) < 60,
        },
        {"id": "settle", "label": "收势", "start": t1, "end": T, "problem": False},
    ]
    return {
        "T": T,
        "up_axis": up,
        "kind": "turn",
        "yaw_span_deg": round(float(np.rad2deg(dyaw.sum())), 2),
        "phases": [p for p in phases if p["end"] > p["start"]],
        "problems": ["weak_turn"] if float(np.rad2deg(dyaw.sum())) < 60 else [],
        "trim_suggestion": {"keep_start": 0, "keep_end": T},
    }


def analyze_walk_timeline(joints: np.ndarray, *, slow: bool = False) -> Dict[str, Any]:
    j = np.asarray(joints, dtype=np.float64)
    T = int(j.shape[0])
    up = _up_axis(j)
    hz = _horiz(up)
    root = j[:, J_PELVIS]
    step = np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)
    step = np.concatenate([[0.0], step])
    travel = float(np.linalg.norm(root[-1, hz] - root[0, hz]))
    thr = max(float(np.percentile(step, 40)), 1e-4)
    moving = step >= thr
    if moving.any():
        idx = np.where(moving)[0]
        w0, w1 = int(idx[0]), int(idx[-1]) + 1
    else:
        w0, w1 = 0, T
    pid = "walk_slow" if slow else "walk_forward"
    label = "慢走" if slow else "向前走"
    phases = [
        {"id": "start", "label": "起步", "start": 0, "end": w0, "problem": False},
        {
            "id": pid,
            "label": label,
            "start": w0,
            "end": w1,
            "problem": travel < (0.2 if not slow else 0.12),
        },
        {"id": "end", "label": "收步", "start": w1, "end": T, "problem": False},
    ]
    return {
        "T": T,
        "up_axis": up,
        "kind": "walk",
        "root_travel": round(travel, 4),
        "phases": [p for p in phases if p["end"] > p["start"]],
        "problems": ["weak_forward_travel"] if travel < 0.15 else [],
        "trim_suggestion": {"keep_start": 0, "keep_end": T},
    }


def analyze_generic_timeline(joints: np.ndarray, *, action: str = "") -> Dict[str, Any]:
    """Fallback: split by motion energy into low / high / low."""
    j = np.asarray(joints, dtype=np.float64)
    T = int(j.shape[0])
    up = _up_axis(j)
    vel = np.linalg.norm(np.diff(j.reshape(T, -1), axis=0), axis=-1)
    vel = np.concatenate([[0.0], vel])
    thr = max(float(np.percentile(vel, 55)), 1e-4)
    active = vel >= thr
    if active.any():
        idx = np.where(active)[0]
        a0, a1 = int(idx[0]), int(idx[-1]) + 1
    else:
        a0, a1 = T // 4, (3 * T) // 4
    name = action or "motion"
    phases = [
        {"id": "intro", "label": "前段", "start": 0, "end": a0, "problem": False},
        {"id": name, "label": name, "start": a0, "end": a1, "problem": False},
        {"id": "outro", "label": "后段", "start": a1, "end": T, "problem": False},
    ]
    return {
        "T": T,
        "up_axis": up,
        "kind": "generic",
        "phases": [p for p in phases if p["end"] > p["start"]],
        "problems": [],
        "trim_suggestion": {"keep_start": 0, "keep_end": T},
    }


def analyze_motion_timeline(
    joints: np.ndarray,
    *,
    action: Optional[str] = None,
    forbid: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Dispatch timeline analyzer by primary action (always returns a timeline)."""
    act = (action or "").strip()
    if act.startswith("sit") or act in ("sit_down", "sit"):
        return analyze_sit_timeline(joints, forbid=forbid)
    if act in ("wave_right", "wave") or act.endswith("wave_right"):
        return analyze_wave_timeline(joints, side="right")
    if act in ("wave_left",) or act.endswith("wave_left"):
        return analyze_wave_timeline(joints, side="left")
    if act in ("turn_in_place", "turn") or "turn" in act:
        return analyze_turn_timeline(joints)
    if act in ("walk_slow",) or "slow" in act:
        return analyze_walk_timeline(joints, slow=True)
    if act in ("walk_forward", "walk") or act.startswith("walk"):
        return analyze_walk_timeline(joints, slow=False)
    return analyze_generic_timeline(joints, action=act or "motion")


def trim_joints_by_timeline(
    joints: np.ndarray,
    timeline: Dict[str, Any],
    *,
    forbid: Optional[List[str]] = None,
    pad_before: int = 2,
    pad_after: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    forbid = set(forbid or [])
    T = joints.shape[0]
    start = 0
    end = T
    if "pre_sit_locomotion" in forbid:
        start = max(0, int(timeline["sit_start"]) - pad_before)
    if "post_sit_extra" in forbid or (
        "post_sit_limb_rebound" in forbid and timeline.get("hold_end", T) < T - 10
    ):
        # Only trim trailing when user asked to remove post extras / rebound flail
        if "post_sit_extra" in forbid:
            end = min(T, int(timeline["hold_end"]) + pad_after)
    start = int(np.clip(start, 0, T - 5))
    end = int(np.clip(end, start + 10, T))
    meta = {"trimmed_start": start, "trimmed_end": end, "T_in": T, "T_out": end - start}
    return joints[start:end].copy(), meta


def _resample_slice(arr: np.ndarray, new_len: int) -> np.ndarray:
    """Linear resample along time axis (T, ...)."""
    t0 = int(arr.shape[0])
    new_len = int(max(2, new_len))
    if new_len == t0:
        return arr.copy()
    old = np.linspace(0.0, 1.0, t0)
    new = np.linspace(0.0, 1.0, new_len)
    flat = arr.reshape(t0, -1).astype(np.float64)
    out = np.stack([np.interp(new, old, flat[:, i]) for i in range(flat.shape[1])], axis=0).T
    return out.reshape((new_len,) + arr.shape[1:]).astype(arr.dtype)


def _timeline_spans(
    timeline: Dict[str, Any], T: int
) -> List[Tuple[str, int, int]]:
    """Ordered (phase_id, start, end) covering as much of [0,T) as phases allow."""
    raw = list(timeline.get("phases") or [])
    spans: List[Tuple[str, int, int]] = []
    if raw:
        for p in raw:
            pid = str(p.get("id") or "")
            if not pid:
                continue
            s = int(np.clip(int(p.get("start") or 0), 0, T))
            e = int(np.clip(int(p.get("end") or s), s, T))
            if e > s:
                spans.append((pid, s, e))
        if spans:
            return spans
    # Sit marker fallback when phases missing
    markers = [
        ("preamble", 0, int(timeline.get("sit_start") or 0)),
        (
            "descend",
            int(timeline.get("sit_start") or 0),
            int(timeline.get("contact") or T // 2),
        ),
        (
            "seated_hold",
            int(timeline.get("contact") or T // 2),
            int(timeline.get("hold_end") or T),
        ),
        ("trailing", int(timeline.get("hold_end") or T), T),
    ]
    for pid, s, e in markers:
        s = int(np.clip(s, 0, T))
        e = int(np.clip(e, s, T))
        if e > s:
            spans.append((pid, s, e))
    return spans


def apply_phase_speed(
    joints: np.ndarray,
    timeline: Dict[str, Any],
    phase_speed: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resample named timeline phases. speed<1 slows (more frames), >1 speeds up."""
    if not phase_speed:
        return joints, {"applied": {}}
    T = int(joints.shape[0])
    spans = _timeline_spans(timeline, T)
    parts: List[np.ndarray] = []
    applied: Dict[str, Any] = {}
    cursor = 0
    for pid, s, e in spans:
        if s > cursor:
            parts.append(joints[cursor:s])
        chunk = joints[s:e]
        scale = float(phase_speed.get(pid) or 1.0)
        if abs(scale - 1.0) > 0.02 and chunk.shape[0] >= 2:
            new_len = int(max(2, min(400, round(chunk.shape[0] / max(scale, 1e-3)))))
            chunk = _resample_slice(chunk, new_len)
            applied[pid] = {
                "speed_scale": scale,
                "T_in": e - s,
                "T_out": int(chunk.shape[0]),
            }
        parts.append(chunk)
        cursor = e
    if cursor < T:
        parts.append(joints[cursor:T])
    if not parts:
        return joints, {"applied": {}}
    out = np.concatenate(parts, axis=0).astype(np.float32)
    return out, {"applied": applied, "T_in": T, "T_out": int(out.shape[0])}


def drop_timeline_phases(
    joints: np.ndarray,
    timeline: Dict[str, Any],
    phase_ids: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove frames belonging to the named phases (keep order of the rest)."""
    want = {str(p) for p in (phase_ids or []) if p}
    if not want:
        return joints, {"dropped": {}}
    T = int(joints.shape[0])
    spans = _timeline_spans(timeline, T)
    keep_mask = np.ones(T, dtype=bool)
    dropped: Dict[str, Any] = {}
    for pid, s, e in spans:
        if pid not in want:
            continue
        keep_mask[s:e] = False
        dropped[pid] = {"start": s, "end": e, "frames": int(e - s)}
    if not dropped or keep_mask.all():
        return joints, {"dropped": {}}
    kept = int(keep_mask.sum())
    if kept < 2:
        # Never collapse the whole clip — keep at least 2 frames
        return joints, {"dropped": {}, "skipped": "would_empty"}
    out = np.asarray(joints, dtype=np.float32)[keep_mask]
    return out, {"dropped": dropped, "T_in": T, "T_out": int(out.shape[0])}
