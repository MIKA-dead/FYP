"""
Compose any clip by the same five stages:

  1. unit   — one repeating period, or the steadiest hold if it does not repeat
  2. chain  — overlap the units so the pose and the root path stay continuous
  3. test   — torso speed at the joins, and whether travel grew
  4. revise — shift the cut and widen the blend, then test again
  5. output — keep the joined clip only when the join stays within the limit

No action name is required. A walk period, a wave period, and a sit hold
all go through these stages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

J_PELVIS = 0
J_HEAD = 15
J_L_FOOT, J_R_FOOT = 10, 11
TORSO = (0, 3, 6, 9, 12, 15)

SEAM_LIMIT = 1.25


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def _horiz(up: int) -> List[int]:
    return [i for i in range(3) if i != up]


def _period(joints: np.ndarray) -> Tuple[int, float]:
    """Dominant period of root-relative joint speed. Peak is the repeat strength."""
    rel = joints - joints[:, J_PELVIS : J_PELVIS + 1, :]
    vel = np.diff(rel, axis=0).reshape(rel.shape[0] - 1, -1)
    energy = np.linalg.norm(vel, axis=1)
    energy = energy - float(energy.mean())
    n = int(energy.shape[0])
    if n < 16 or float(energy.std()) < 1e-8:
        return max(8, joints.shape[0] // 3), 0.0
    spec = np.fft.rfft(energy, n=n * 2)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac = ac / (float(ac[0]) + 1e-8)
    lo, hi = 8, max(9, min(n // 2, 48))
    lag = lo + int(np.argmax(ac[lo:hi]))
    return int(lag), float(ac[lag])


def _closure(joints: np.ndarray, start: int, period: int) -> float:
    rel = joints - joints[:, J_PELVIS : J_PELVIS + 1, :]
    end = start + period
    if end >= joints.shape[0]:
        end = joints.shape[0] - 1
        start = max(0, end - period)
    return float(np.mean(np.linalg.norm(rel[start] - rel[end], axis=-1)))


def _travel(joints: np.ndarray) -> float:
    up = _up_axis(joints)
    hz = _horiz(up)
    root = joints[:, J_PELVIS]
    return float(np.linalg.norm(root[-1, hz] - root[0, hz]))


def _contacts_in(joints: np.ndarray, start: int, period: int) -> int:
    """Foot-down onsets inside one unit. Threshold comes from the whole clip."""
    up = _up_axis(joints)
    end = min(joints.shape[0], start + period)
    if end - start < 4:
        return 1
    frames: List[int] = []
    for fi in (J_L_FOOT, J_R_FOOT):
        z_all = joints[:, fi, up]
        thr = float(np.percentile(z_all, 25)) + 0.02
        down = z_all[start:end] <= thr
        for t in range(1, len(down)):
            if down[t] and not down[t - 1]:
                frames.append(start + t)
    frames = sorted(frames)
    merged: List[int] = []
    for t in frames:
        if not merged or t - merged[-1] >= 5:
            merged.append(t)
    return max(1, len(merged))


def _active_span(joints: np.ndarray) -> Tuple[int, int]:
    """Longest stretch that is actually moving. A still lead-in is not the unit."""
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=-1)
    if speed.size < 8:
        return 0, int(joints.shape[0])
    thr = float(np.percentile(speed, 50))
    if float(speed.max()) <= thr * 1.35 + 1e-8:
        return 0, int(joints.shape[0])
    idx = np.flatnonzero(speed >= thr)
    if idx.size < 8:
        return 0, int(joints.shape[0])
    breaks = np.where(np.diff(idx) > 3)[0]
    runs = np.split(idx, breaks + 1)
    run = max(runs, key=lambda r: int(r.size))
    start = int(run[0])
    end = min(int(joints.shape[0]), int(run[-1]) + 2)
    if end - start < 8:
        return 0, int(joints.shape[0])
    return start, end


def _select_unit(joints: np.ndarray) -> Dict[str, Any]:
    """One cycle when the motion repeats, otherwise the single active phrase."""
    span_a, span_b = _active_span(joints)
    sl = joints[span_a:span_b]
    span_len = int(sl.shape[0])
    span_travel = _travel(sl)
    rel = sl - sl[:, J_PELVIS : J_PELVIS + 1, :]
    vel = np.diff(rel, axis=0).reshape(max(1, span_len - 1), -1)
    energy = np.linalg.norm(vel, axis=1)
    energy = energy - float(energy.mean())
    ac = np.zeros(span_len, dtype=np.float64)
    if energy.size >= 16 and float(energy.std()) > 1e-8:
        n = int(energy.shape[0])
        spec = np.fft.rfft(energy, n=n * 2)
        ac_full = np.fft.irfft(spec * np.conj(spec))[:n]
        ac[: ac_full.shape[0]] = ac_full / (float(ac_full[0]) + 1e-8)

    up = _up_axis(sl)
    hz = _horiz(up)
    best: Optional[Tuple[float, int, int, float, float, float]] = None
    hi = max(9, min(span_len // 2, 48))
    for lag in range(8, hi):
        ac_lag = float(ac[lag]) if lag < ac.shape[0] else 0.0
        step = max(1, lag // 8)
        for s in range(0, span_len - lag, step):
            move = float(np.linalg.norm(sl[s + lag, J_PELVIS, hz] - sl[s, J_PELVIS, hz]))
            close = _closure(sl, s, lag)
            if span_travel > 0.12:
                expect = span_travel * (lag / max(span_len, 1))
                if move < max(0.04, 0.45 * expect):
                    continue
                # A weak peak is a bob inside one phrase, not a step we can tile.
                if ac_lag < 0.35:
                    continue
                score = move / (close + 0.02)
            else:
                if ac_lag < 0.2:
                    continue
                score = ac_lag / (close + 0.02)
            if best is None or score > best[0]:
                best = (score, lag, span_a + s, close, move, ac_lag)

    if best is not None:
        _score, lag, start, close, move, ac_lag = best
        return {
            "kind": "cycle",
            "start": int(start),
            "period": int(lag),
            "peak": float(ac_lag),
            "closure": float(close),
            "move": float(move),
            "span": (span_a, span_b),
        }

    return {
        "kind": "phrase",
        "start": int(span_a),
        "period": int(span_len),
        "peak": float(ac[8:hi].max()) if hi > 8 else 0.0,
        "closure": _closure(sl, 0, span_len - 1) if span_len > 2 else 0.0,
        "move": float(span_travel),
        "span": (span_a, span_b),
    }


def _candidate_starts(joints: np.ndarray, period: int, anchor: int) -> List[int]:
    """A few cuts around the chosen unit, still inside the clip."""
    starts = [anchor]
    for shift in (period // 6, period // 3):
        for s in (anchor + shift, anchor - shift):
            if 0 <= s <= joints.shape[0] - period and s not in starts:
                starts.append(int(s))
    return starts


def _close_unit(cycle: np.ndarray, blend: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bend the end of one unit onto its own start, and keep the horizontal step."""
    cycle = np.asarray(cycle, dtype=np.float32).copy()
    length = cycle.shape[0]
    blend = int(max(2, min(blend, max(2, length // 3))))
    up = _up_axis(cycle)
    delta = np.zeros(3, dtype=np.float32)
    delta[_horiz(up)] = (cycle[-1, J_PELVIS] - cycle[0, J_PELVIS])[_horiz(up)]
    local = cycle - cycle[:, J_PELVIS : J_PELVIS + 1, :]
    root = cycle[:, J_PELVIS].copy()
    ramp = 0.5 - 0.5 * np.cos(np.linspace(np.pi / blend, np.pi, blend))
    for k in range(blend):
        t = length - blend + k
        w = float(ramp[k])
        src = min(k, length - 1)
        local[t] = (1.0 - w) * local[t] + w * local[src]
        root[t] = (1.0 - w) * root[t] + w * (root[src] + delta)
    return local + root[:, None, :], delta


def _chain(cycle: np.ndarray, delta: np.ndarray, count: int, overlap: int) -> np.ndarray:
    length = cycle.shape[0]
    overlap = int(max(0, min(overlap, max(0, length // 2))))
    pieces: List[np.ndarray] = []
    offset = np.zeros(3, dtype=np.float32)
    for i in range(count):
        chunk = cycle + offset
        if i == 0 or overlap < 2:
            pieces.append(chunk)
        else:
            prev = pieces[-1]
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap))).astype(np.float32)
            ramp = ramp[:, None, None]
            blended = (1.0 - ramp) * prev[-overlap:] + ramp * chunk[:overlap]
            pieces[-1] = prev[:-overlap]
            pieces.append(np.concatenate([blended, chunk[overlap:]], axis=0))
        offset = offset + delta
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _stitch_ratio(joints: np.ndarray, period: int, overlap: int, count: int) -> float:
    """Join speed divided by the fastest speed inside a unit.

    A step already has a fast moment. The join fails only when it is faster
    than that moment.
    """
    torso = joints[:, list(TORSO)]
    overlap = int(max(0, min(overlap, max(0, period // 2))))
    if torso.shape[0] < 3 or count < 2:
        return 0.0
    speed = np.linalg.norm(np.diff(torso, axis=0), axis=-1).mean(axis=-1)
    mask = np.zeros(speed.shape[0], dtype=bool)
    for i in range(1, count):
        cursor = i * (period - overlap)
        lo = max(0, cursor - 1)
        hi = min(speed.shape[0], cursor + max(overlap, 1))
        mask[lo:hi] = True
    if not mask.any() or not (~mask).any():
        med = float(np.median(speed)) + 1e-6
        return float(np.max(speed) / med)
    outside = float(np.max(speed[~mask])) + 1e-6
    return float(np.max(speed[mask]) / outside)


def _foot_swings(joints: np.ndarray) -> List[Dict[str, Any]]:
    """Non-overlapping steps. Each step is the swing of one foot, cut at the other foot."""
    up = _up_axis(joints)
    hz = _horiz(up)
    T = int(joints.shape[0])
    if T < 8:
        return []
    height = {}
    base = {}
    for foot, idx in (("left", J_L_FOOT), ("right", J_R_FOOT)):
        z = joints[:, idx, up].astype(np.float64)
        height[foot] = z
        base[foot] = float(np.percentile(z, 20))
    peaks: List[Dict[str, Any]] = []
    for foot, idx in (("left", J_L_FOOT), ("right", J_R_FOOT)):
        z = height[foot]
        other = "right" if foot == "left" else "left"
        lift = float(np.percentile(z, 95) - base[foot])
        if lift < 0.02:
            continue
        thr = base[foot] + 0.55 * lift
        for t in range(2, T - 2):
            if z[t] < thr:
                continue
            if not (z[t] >= z[t - 1] and z[t] >= z[t + 1] and z[t] >= z[t - 2] and z[t] >= z[t + 2]):
                continue
            if (z[t] - base[foot]) < (height[other][t] - base[other]) + 0.008:
                continue
            if any(p["foot"] == foot and abs(t - p["peak"]) < 6 for p in peaks):
                continue
            peaks.append({"foot": foot, "peak": int(t), "lift": float(z[t] - base[foot])})
    peaks.sort(key=lambda p: p["peak"])
    merged: List[Dict[str, Any]] = []
    for p in peaks:
        if merged and merged[-1]["foot"] == p["foot"]:
            merged[-1]["peaks"].append(p["peak"])
            if p["lift"] > merged[-1]["lift"]:
                merged[-1]["peak"] = p["peak"]
                merged[-1]["lift"] = p["lift"]
        else:
            merged.append({"foot": p["foot"], "peak": p["peak"], "lift": p["lift"], "peaks": [p["peak"]]})
    if not merged:
        return []
    bounds: List[int] = []
    first = merged[0]["peak"]
    z0 = height[merged[0]["foot"]]
    s = first
    while s > 1 and z0[s - 1] < z0[s]:
        s -= 1
    bounds.append(int(s))
    for i in range(len(merged) - 1):
        bounds.append(int((merged[i]["peak"] + merged[i + 1]["peak"]) // 2))
    last = merged[-1]["peak"]
    z1 = height[merged[-1]["foot"]]
    e = last
    while e < T - 2 and z1[e + 1] < z1[e]:
        e += 1
    bounds.append(int(e))
    root = joints[:, J_PELVIS]
    swings: List[Dict[str, Any]] = []
    for i, sw in enumerate(merged):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 4:
            continue
        travel = float(np.linalg.norm(root[end, hz] - root[start, hz]))
        swings.append(
            {
                "foot": sw["foot"],
                "start": int(start),
                "end": int(end),
                "peak": int(sw["peak"]),
                "lift": round(float(sw["lift"]), 4),
                "travel": travel,
            }
        )
    return swings


def _pose_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean joint distance after aligning pelvis. Large means the phases differ."""
    da = a - a[J_PELVIS]
    db = b - b[J_PELVIS]
    return float(np.mean(np.linalg.norm(da - db, axis=-1)))


def _step_templates(joints: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """The first left step and the first right step, in the order they occur.

    Later repeats of the same foot are not a second template. Replaying one
    foot twice is what reads as a limp.
    """
    swings = _foot_swings(joints)
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for sw in swings:
        if sw["foot"] in seen:
            continue
        if chosen and sw["foot"] == chosen[-1]["foot"]:
            continue
        seen.add(sw["foot"])
        chosen.append(sw)
        if len(chosen) == 2:
            break
    if len(chosen) < 2 or chosen[0]["foot"] == chosen[1]["foot"]:
        return None
    return chosen


def _pelvis_speed(joints: np.ndarray) -> np.ndarray:
    up = _up_axis(joints)
    hz = _horiz(up)
    root = joints[:, J_PELVIS].astype(np.float64)
    return np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)


def _trim_outer(joints: np.ndarray, steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Keep the ease-in and ease-out off the loop point.

    A step that slows to a stop cannot be the frame that joins the next step.
    Those frames stay available for the very start and the very end.
    """
    speed = _pelvis_speed(joints)
    if speed.size < 8:
        return steps, None
    first, last = steps[0], steps[-1]
    lo = int(first["start"])
    hi = min(int(last["end"]), int(speed.shape[0] - 1))
    if hi - lo < 8:
        return steps, None
    peak_speed = float(speed[lo:hi].max())
    if peak_speed < 1e-4:
        return steps, None
    thr = 0.50 * peak_speed
    new_lo = lo
    while new_lo < int(first["peak"]) and new_lo < hi - 6 and float(speed[new_lo]) < thr:
        new_lo += 1
    new_hi = hi
    while new_hi > int(last["peak"]) and new_hi > new_lo + 6 and float(speed[new_hi - 1]) < thr:
        new_hi -= 1
    out = [dict(sw) for sw in steps]
    for sw in out:
        sw["raw_start"] = int(sw["start"])
        sw["raw_end"] = int(sw["end"])
    out[0]["start"] = int(new_lo)
    out[-1]["end"] = int(new_hi)
    return out, {
        "trim_start": int(new_lo),
        "trim_end": int(new_hi),
        "edge_in": round(float(speed[new_lo]), 4),
        "edge_out": round(float(speed[new_hi - 1]), 4),
    }


def _speed_faults(joints: np.ndarray) -> Optional[str]:
    """A stop that recovers is a pause in the middle. A spike is a crushed blend."""
    speed = _pelvis_speed(joints)
    if speed.size < 8:
        return None
    med = float(np.median(speed))
    if med < 1e-5:
        return "拼接后几乎没有在走"
    slow = speed < (0.35 * med)
    i = 0
    while i < slow.size and slow[i]:
        i += 1
    j = int(slow.size) - 1
    while j > i and slow[j]:
        j -= 1
    if bool(slow[i : j + 1].any()):
        return "拼接处速度掉下去，出现停顿"
    cruise = float(np.percentile(speed, 90))
    if float(speed.max()) > max(2.2 * cruise, 1e-6):
        return "接缝把两段位移压进几帧，速度突然变快"
    return None


def _gait_report(joints: np.ndarray) -> Tuple[bool, str]:
    fault = _speed_faults(joints)
    if fault:
        return False, fault
    swings = _foot_swings(joints)
    if len(swings) < 2:
        return False, "拼完之后分不出左右交替的步"
    feet = [sw["foot"] for sw in swings]
    if any(feet[i] == feet[i + 1] for i in range(len(feet) - 1)):
        return False, "同一只脚连续落下"
    travels = [max(sw["travel"], 0.0) for sw in swings]
    if min(travels) < 0.02:
        return False, "有一步几乎没有前进"
    ratio = max(travels) / min(travels)
    if ratio > 2.2:
        return False, f"相邻步长相差 {ratio:.2f} 倍"
    order = "".join("左" if f == "left" else "右" for f in feet)
    return True, f"左右交替（{order}），步长比 {ratio:.2f}"


def _chain_alternating(
    joints: np.ndarray,
    steps: List[Dict[str, Any]],
    count: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Alternate steps. Loop where the speed still matches, and blend that join only."""
    empty = {"loop_pose": None, "mid_pose": None, "overlap": 0, "edge_in": None, "edge_out": None, "lead_in": 0, "lead_out": 0}
    trimmed, trim = _trim_outer(joints, steps)
    if trim is None:
        empty["reason"] = "pause"
        return None, empty
    clips: List[np.ndarray] = []
    for i, sw in enumerate(trimmed):
        stop = sw["end"] if i < len(trimmed) - 1 else sw["end"] + 1
        if stop - int(sw["start"]) < 4:
            empty["reason"] = "pause"
            empty["edge_in"] = trim["edge_in"]
            empty["edge_out"] = trim["edge_out"]
            return None, empty
        clips.append(joints[int(sw["start"]) : stop].astype(np.float32))
    loop_delta = _pose_delta(clips[-1][-1], clips[0][0])
    mid_delta = _pose_delta(clips[0][-1], clips[1][0]) if len(clips) > 1 else 0.0
    ein, eout = float(trim["edge_in"]), float(trim["edge_out"])
    info = {
        "loop_pose": round(loop_delta, 4),
        "mid_pose": round(mid_delta, 4),
        "overlap": 0,
        "edge_in": trim["edge_in"],
        "edge_out": trim["edge_out"],
        "lead_in": 0,
        "lead_out": 0,
    }
    if min(ein, eout) <= 1e-4 or max(ein, eout) / (min(ein, eout) + 1e-8) > 1.8:
        info["reason"] = "pause"
        return None, info
    if max(loop_delta, mid_delta) > 0.12:
        info["reason"] = "phase"
        return None, info
    # Pose ease on the loop only. The root keeps its speed; blending positions
    # would park two frames on the same spot and read as a pause.
    loop_overlap = 4 if loop_delta <= 0.12 else 0
    info["overlap"] = loop_overlap
    pieces: List[np.ndarray] = []
    for i in range(count):
        chunk = clips[i % len(clips)].copy()
        if not pieces:
            pieces.append(chunk)
            continue
        prev = pieces[-1]
        prev_sw = trimmed[(i - 1) % len(trimmed)]
        this_sw = trimmed[i % len(trimmed)]
        adjacent = int(this_sw["start"]) == int(prev_sw["end"])
        if adjacent and int(prev_sw["end"]) >= 1:
            prev_end = int(prev_sw["end"])
            delta = (joints[prev_end, J_PELVIS] - joints[prev_end - 1, J_PELVIS]).astype(np.float32)
        elif prev.shape[0] >= 2:
            delta = (prev[-1, J_PELVIS] - prev[-2, J_PELVIS]).astype(np.float32)
        else:
            delta = np.zeros(3, dtype=np.float32)
        chunk = chunk + (prev[-1, J_PELVIS] + delta - chunk[0, J_PELVIS])
        ov = loop_overlap if i % len(trimmed) == 0 else 0
        if ov >= 2 and chunk.shape[0] > ov:
            rel_prev = prev[-1] - prev[-1, J_PELVIS]
            for k in range(ov):
                w = float(0.5 - 0.5 * np.cos(np.pi * (k + 1) / float(ov)))
                rel = (1.0 - w) * rel_prev + w * (chunk[k] - chunk[k, J_PELVIS])
                chunk[k] = rel + chunk[k, J_PELVIS]
        pieces.append(chunk)
    chained = np.concatenate(pieces, axis=0)
    raw_start = int(trimmed[0]["raw_start"])
    raw_end = int(trimmed[-1]["raw_end"])
    trim_start = int(trimmed[0]["start"])
    trim_end = int(trimmed[-1]["end"])
    if trim_start > raw_start:
        lead = joints[raw_start:trim_start].astype(np.float32)
        lead = lead + (chained[0, J_PELVIS] - joints[trim_start, J_PELVIS]).astype(np.float32)
        chained = np.concatenate([lead, chained], axis=0)
    if raw_end > trim_end:
        tail = joints[trim_end + 1 : raw_end + 1].astype(np.float32)
        if tail.shape[0]:
            tail = tail + (chained[-1, J_PELVIS] - joints[trim_end, J_PELVIS]).astype(np.float32)
            chained = np.concatenate([chained, tail], axis=0)
    info["lead_in"] = int(max(0, trim_start - raw_start))
    info["lead_out"] = int(max(0, raw_end - trim_end))
    info["reason"] = "ok"
    return chained.astype(np.float32), info


def _compose_gait(
    joints: np.ndarray,
    *,
    cycle_count: Optional[int],
    repeat_more: bool,
    forbid: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extend a walk by alternating the left step and the right step."""
    work = joints
    applied = ""
    if "yaw_spin" in forbid:
        from constraint_postprocess import straighten_yaw_and_path

        work, yinfo = straighten_yaw_and_path(joints)
        applied = f" 走直线约束：转角 {yinfo.get('yaw_span_before_deg')}°→{yinfo.get('yaw_span_after_deg')}°。"
    source_travel = _travel(work)
    steps = _step_templates(work)
    if not steps:
        detail = "片段里凑不出左右脚交替的两步，不能拼接，保持当前片段。"
        stages = [
            {"step": 1, "name": "单元", "detail": detail},
            {"step": 2, "name": "串联", "detail": "没有下一段可接。"},
            {"step": 3, "name": "测试", "detail": "步态测试没有做，因为单元不存在。"},
            {"step": 4, "name": "自检", "detail": "保持当前片段。"},
            {"step": 5, "name": "输出", "detail": "不输出拼接结果。"},
        ]
        return joints, {
            "method": "rejected_phase",
            "kind": "gait",
            "stages": stages,
            "seam_ratio": None,
            "T_out": int(joints.shape[0]),
            "target_units": 0,
        }
    existing = 2
    if cycle_count:
        target = max(existing, int(cycle_count))
    elif repeat_more:
        target = existing + 2
    else:
        target = existing + 2
    if target % 2 == 1:
        target += 1
    names = "、".join(
        f"{'左' if sw['foot']=='left' else '右'}脚第 {sw['start']}–{sw['end']} 帧（抬起 {sw['lift']:.3f}，步长 {sw['travel']:.2f}）"
        for sw in steps
    )
    chained, join = _chain_alternating(work, steps, target)
    if chained is None and join.get("reason") == "pause":
        gait_ok, gait_text = False, "相接处一快一慢，中间会停一下"
    elif chained is None:
        gait_ok, gait_text = False, "相位差太大，没有拼接"
    else:
        gait_ok, gait_text = _gait_report(chained)
    grew = chained is not None and _travel(chained) >= source_travel * 1.1
    seam = _stitch_ratio(chained, max(8, chained.shape[0] // target), max(join.get("overlap", 0), 2), target) if chained is not None else None
    ok = chained is not None and gait_ok and grew
    stages = [
        {
            "step": 1,
            "name": "单元",
            "detail": f"拆成左右两步，开头和结尾都在脚落地附近：{names}。",
        },
        {
            "step": 2,
            "name": "串联",
            "detail": (
                f"按 {'、'.join('左' if sw['foot']=='left' else '右' for sw in steps)} 交替接到 {target} 步。"
                f"起步留在最前面 {join.get('lead_in', 0)} 帧，收步留在最后 {join.get('lead_out', 0)} 帧，"
                f"中间只用还在走的段落相接（入 {join.get('edge_in')}，出 {join.get('edge_out')}）。"
                f"循环处用 {join.get('overlap', 0)} 帧把姿势过渡过去，骨盆按原来的速度继续往前，不把两帧放在同一个位置。"
                f"两步相接的姿势差 {join.get('mid_pose')}，循环回到第一步的姿势差 {join.get('loop_pose')}。"
            ),
        },
        {
            "step": 3,
            "name": "测试",
            "detail": (
                f"{gait_text}。位移 {source_travel:.2f}→{_travel(chained) if chained is not None else source_travel:.2f}，"
                f"帧数 {joints.shape[0]}→{0 if chained is None else chained.shape[0]}。"
                + (f" 躯干接缝速度比 {seam:.2f}，只作参考。" if seam is not None else "")
            ),
        },
        {
            "step": 4,
            "name": "自检",
            "detail": (
                "左右必须交替，相邻步长不能差一倍以上，接缝姿势差要小于 0.12。"
                + applied
            ),
        },
    ]
    if not ok:
        why = gait_text if chained is not None else (
            "相接处一快一慢，接上会在中间停一下" if join.get("reason") == "pause" else "左右两步在相接处姿势不同，接上会瘸"
        )
        if chained is not None and not grew:
            why = "接上之后前进没有变多"
        stages.append({"step": 5, "name": "输出", "detail": f"步态测试没有通过（{why}），保持当前片段。"})
        return joints, {
            "method": "rejected_phase",
            "kind": "gait",
            "stages": stages,
            "seam_ratio": None if seam is None else round(float(seam), 3),
            "T_out": int(joints.shape[0]),
            "target_units": int(target),
            "travel_in": round(source_travel, 4),
        }
    stages.append(
        {
            "step": 5,
            "name": "输出",
            "detail": f"通过。输出 {chained.shape[0]} 帧，{target} 步交替。",
        }
    )
    return chained, {
        "method": "compose_units",
        "kind": "gait",
        "stages": stages,
        "seam_ratio": None if seam is None else round(float(seam), 3),
        "T_out": int(chained.shape[0]),
        "target_units": int(target),
        "travel_in": round(source_travel, 4),
        "travel_out": round(_travel(chained), 4),
        "overlap": join.get("overlap", 0),
    }


def compose_motion(
    joints: np.ndarray,
    *,
    cycle_count: Optional[int] = None,
    repeat_more: bool = False,
    forbid: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the composed clip and the five-stage record."""
    joints = np.asarray(joints, dtype=np.float32)
    forbid = list(forbid or [])
    stages: List[Dict[str, Any]] = []
    if joints.ndim != 3 or joints.shape[0] < 12:
        stages = [
            {"step": 1, "name": "单元", "detail": "片段太短，取不出一个可重复的单元。"},
            {"step": 5, "name": "输出", "detail": "保持当前片段。"},
        ]
        return joints, {"method": "rejected_short", "stages": stages, "seam_ratio": None}

    if _travel(joints) > 0.12:
        return _compose_gait(
            joints,
            cycle_count=cycle_count,
            repeat_more=repeat_more,
            forbid=forbid,
        )

    period_info = _select_unit(joints)
    period = int(period_info["period"])
    peak = float(period_info["peak"])
    start = int(period_info["start"])
    kind = str(period_info["kind"])
    starts = _candidate_starts(joints, period, start)
    per_count = _contacts_in(joints, start, period)
    if kind == "cycle":
        unit_note = (
            f"从当前片段取出一个周期：第 {start}–{start + period} 帧，"
            f"周期 {period} 帧，重复强度 {peak:.2f}，闭合误差 {period_info['closure']:.3f}，"
            f"这一步前进 {period_info['move']:.2f}。这一单元里有 {per_count} 次落脚。"
        )
        existing = max(1, int(round((period_info["span"][1] - period_info["span"][0]) / max(period, 1))))
    else:
        unit_note = (
            f"没有找到可以单独重复的短周期，单元用整段有效动作："
            f"第 {start}–{start + period} 帧，前进 {period_info['move']:.2f}，"
            f"落脚 {per_count} 次。后面按这一段接起来。"
        )
        existing = 1
    if kind != "cycle" or float(period_info.get("closure") or 1.0) > 0.08:
        stages = [
            {
                "step": 1,
                "name": "单元",
                "detail": "没有找到相位能接上的重复段，不能拼接，保持当前片段。",
            },
            {"step": 5, "name": "输出", "detail": "不输出拼接结果。"},
        ]
        return joints, {
            "method": "rejected_phase",
            "kind": kind,
            "stages": stages,
            "seam_ratio": None,
            "T_out": int(joints.shape[0]),
        }
    if cycle_count:
        target = max(existing + 1, int(np.ceil(float(cycle_count) / float(per_count))))
        count_note = f"目标次数 {cycle_count}，按每单元 {per_count} 次，接成 {target} 个单元（原来约 {existing} 个）。"
    elif repeat_more:
        target = existing + 2
        count_note = f"要求再多几个单元，从约 {existing} 个接到 {target} 个。"
    else:
        target = existing + 1
        count_note = f"接到 {target} 个单元。"
    stages.append({"step": 1, "name": "单元", "detail": unit_note})

    source_travel = _travel(joints)
    attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    seen = set()
    for start_i in starts:
        for close_b, overlap in ((4, 4), (6, 8), (8, 12)):
            key = (int(start_i), close_b, overlap)
            if key in seen:
                continue
            seen.add(key)
            sl = joints[start_i : start_i + period]
            if sl.shape[0] < period:
                continue
            # Do not bend the end pose onto the start. That blend is the limp.
            closed, delta = sl, np.zeros(3, dtype=np.float32)
            up = _up_axis(sl)
            delta[_horiz(up)] = (sl[-1, J_PELVIS] - sl[0, J_PELVIS])[_horiz(up)]
            overlap = 4 if float(period_info.get("closure") or 1) <= 0.05 else 0
            chained = _chain(closed, delta, target, overlap)
            applied = []
            if "yaw_spin" in forbid:
                from constraint_postprocess import straighten_yaw_and_path

                chained, info = straighten_yaw_and_path(chained)
                applied.append(
                    f"走直线约束：转角 {info.get('yaw_span_before_deg')}°→{info.get('yaw_span_after_deg')}°"
                )
            seam = _stitch_ratio(chained, closed.shape[0], overlap, target)
            travel = _travel(chained)
            if source_travel > 0.15:
                grew = travel >= source_travel * 1.15
            else:
                grew = chained.shape[0] > joints.shape[0] + 4
            row = {
                "start": int(start_i),
                "close": close_b,
                "overlap": overlap,
                "seam": round(seam, 3),
                "travel": round(travel, 3),
                "T": int(chained.shape[0]),
                "grew": bool(grew),
                "applied": applied,
                "joints": chained,
            }
            attempts.append({k: v for k, v in row.items() if k != "joints"})
            ok = seam <= SEAM_LIMIT and grew
            better = best is None or (ok and not best["ok"]) or (
                ok == best["ok"] and seam < best["seam"]
            )
            if better:
                best = {**row, "ok": ok}

    assert best is not None
    stages.append(
        {
            "step": 2,
            "name": "串联",
            "detail": (
                f"{count_note} 单元之间重叠 {best['overlap']} 帧，"
                f"并把单元结尾收到单元开头，避免直接拼接的跳变。"
            ),
        }
    )
    stages.append(
        {
            "step": 3,
            "name": "测试",
            "detail": (
                f"接缝速度是单元内部最快速度的 {best['seam']:.2f} 倍（上限 {SEAM_LIMIT:.2f}），"
                f"位移 {source_travel:.2f}→{best['travel']:.2f}，帧数 {joints.shape[0]}→{best['T']}。"
            ),
        }
    )
    revised = [
        a
        for a in attempts
        if (a["start"], a["close"], a["overlap"])
        != (best["start"], best["close"], best["overlap"])
    ]
    if revised:
        first = attempts[0]
        stages.append(
            {
                "step": 4,
                "name": "自检",
                "detail": (
                    f"试了 {len(attempts)} 组切点和重叠。"
                    f"第一组接缝 {first['seam']:.2f}，保留接缝 {best['seam']:.2f} 的一组"
                    f"（起点帧 {best['start']}，重叠 {best['overlap']}）。"
                    + ((" " + " ".join(best["applied"])) if best["applied"] else "")
                ),
            }
        )
    else:
        stages.append(
            {
                "step": 4,
                "name": "自检",
                "detail": "只有一组切法，测试结果直接用于输出。",
            }
        )

    if not best["ok"]:
        why = "接缝仍然过大" if best["seam"] > SEAM_LIMIT else "接长之后位移没有增加"
        stages.append(
            {
                "step": 5,
                "name": "输出",
                "detail": f"自检没有通过（{why}），保持当前片段，不把断开的结果发出去。",
            }
        )
        return joints, {
            "method": "rejected_seam",
            "kind": kind,
            "period": int(period),
            "repeat_strength": round(peak, 3),
            "target_units": int(target),
            "seam_ratio": best["seam"],
            "seam_limit": SEAM_LIMIT,
            "attempts": attempts,
            "stages": stages,
            "T_out": int(joints.shape[0]),
        }

    stages.append(
        {
            "step": 5,
            "name": "输出",
            "detail": (
                f"通过。输出 {best['T']} 帧，{target} 个{('周期' if kind == 'cycle' else '动作')}单元，"
                f"接缝 {best['seam']:.2f}。"
            ),
        }
    )
    return best["joints"], {
        "method": "compose_units",
        "kind": kind,
        "period": int(period),
        "repeat_strength": round(peak, 3),
        "start": best["start"],
        "overlap": best["overlap"],
        "target_units": int(target),
        "seam_ratio": best["seam"],
        "seam_limit": SEAM_LIMIT,
        "travel_in": round(source_travel, 4),
        "travel_out": best["travel"],
        "attempts": attempts,
        "stages": stages,
        "T_out": best["T"],
    }
