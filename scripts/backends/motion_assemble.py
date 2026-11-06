"""
One revise path for every action:

  1. parse segments
  2. current clip if the segment extends it, otherwise the dataset
  3. dataset hits must match skeleton, height, and the requested action
  4. split each accepted reference into one unit
  5. same performance is chained; a new action gets a transition
  6. test the join and the parsed request
  7. only retune the cut, the overlap, and geometric constraints
  8. output the last result that passed
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from motion_compose import (
    SEAM_LIMIT,
    _select_unit,
    compose_motion,
)

_NAMES = ("解析", "选参考", "数据集门", "拆分", "组合", "测试", "修改", "输出")


def _notes() -> Dict[int, List[str]]:
    return {i: [] for i in range(1, 9)}


def _stages(notes: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    out = []
    for i, name in enumerate(_NAMES, start=1):
        detail = "".join(notes[i]).strip() or "这一步没有新的改动。"
        out.append({"step": i, "name": name, "detail": detail})
    return out


def _transition(base: np.ndarray, extra: np.ndarray, overlap: int) -> Tuple[np.ndarray, float]:
    extra = np.asarray(extra, dtype=np.float32).copy()
    shift = base[-1, 0] - extra[0, 0]
    extra = extra + shift
    overlap = int(max(2, min(overlap, base.shape[0] // 3, extra.shape[0] // 3, 16)))
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap))).astype(np.float32)
    ramp = ramp[:, None, None]
    blended = (1.0 - ramp) * base[-overlap:] + ramp * extra[:overlap]
    out = np.concatenate([base[:-overlap], blended, extra[overlap:]], axis=0).astype(np.float32)
    torso = out[:, [0, 3, 6, 9, 12, 15]]
    speed = np.linalg.norm(np.diff(torso, axis=0), axis=-1).mean(axis=-1)
    cursor = base.shape[0] - overlap
    lo = max(0, cursor - 1)
    hi = min(speed.shape[0], cursor + overlap)
    mask = np.zeros(speed.shape[0], dtype=bool)
    mask[lo:hi] = True
    if mask.any() and (~mask).any():
        seam = float(np.max(speed[mask]) / (float(np.max(speed[~mask])) + 1e-6))
    else:
        seam = 0.0
    return out, seam


def _donor_unit(donor: np.ndarray) -> np.ndarray:
    info = _select_unit(donor)
    start = int(info["start"])
    period = int(info["period"])
    return donor[start : start + period].astype(np.float32)


def assemble_motion(
    joints: np.ndarray,
    segments: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    joints = np.asarray(joints, dtype=np.float32)
    notes = _notes()
    if not segments:
        segments = [
            {
                "action": "motion",
                "count": None,
                "repeat_more": True,
                "constraints": [],
                "relation": "extend",
                "reference": "current",
                "retrieval_text": "",
            }
        ]

    bits = []
    for seg in segments:
        count = seg.get("count")
        more = "再多几个单元" if seg.get("repeat_more") and not count else ""
        count_s = f"{count} 次" if count else more or "不改次数"
        bits.append(
            f"{seg.get('action')}，{('延长当前片段' if seg.get('relation') == 'extend' else '接一段新动作')}，{count_s}"
        )
    notes[1].append("拆成 " + "；".join(bits) + "。")

    current = joints
    last_good = joints
    method = "assemble_ok"
    seam_ratio = None
    any_change = False

    for seg in segments:
        action = str(seg.get("action") or "motion")
        constraints = [str(c) for c in (seg.get("constraints") or [])]
        forbid = constraints
        if seg.get("reference") == "current":
            notes[2].append(f"{action} 用画面上这一段。")
            notes[3].append(f"{action} 不检索数据集。")
            if not seg.get("count") and not seg.get("repeat_more"):
                if "yaw_spin" in forbid:
                    from constraint_postprocess import straighten_yaw_and_path

                    current, info = straighten_yaw_and_path(current)
                    last_good = current
                    any_change = True
                    notes[7].append(
                        f"把路径收到起始朝向，转角 {info.get('yaw_span_before')}°→{info.get('yaw_span_after')}°。"
                    )
                continue
            edited, info = compose_motion(
                current,
                cycle_count=seg.get("count"),
                repeat_more=bool(seg.get("repeat_more")),
                forbid=forbid,
            )
            unit = next((s for s in info.get("stages") or [] if s.get("name") == "单元"), None)
            chain = next((s for s in info.get("stages") or [] if s.get("name") == "串联"), None)
            test = next((s for s in info.get("stages") or [] if s.get("name") == "测试"), None)
            fix = next((s for s in info.get("stages") or [] if s.get("name") == "自检"), None)
            if unit:
                notes[4].append(unit.get("detail") or "")
            if chain:
                notes[5].append(chain.get("detail") or "")
            if test:
                notes[6].append(test.get("detail") or "")
            if fix:
                notes[7].append(fix.get("detail") or "")
            seam_ratio = info.get("seam_ratio")
            if str(info.get("method") or "").startswith("rejected"):
                method = str(info.get("method"))
                current = last_good
                notes[8].append("这一段没有通过，停在上一份结果。")
                break
            current = edited
            last_good = edited
            any_change = True
            continue

        notes[2].append(f"{action} 当前片段里没有，改去数据集找。")
        from dataset_edit import retrieve_continuous

        query = str(seg.get("retrieval_text") or f"a person {action}")
        donor, gate = retrieve_continuous(query, None, current, check_speed=False)
        if donor is None:
            method = "assemble_partial" if any_change else "rejected_gate"
            notes[3].append(
                f"{action} 没有候选同时满足骨架、身长，并且不含句中没要求的动作"
                f"（{gate.get('reason')}，跳过 {gate.get('rejected_candidates', 0)} 条）。"
            )
            notes[8].append("新动作没有进预览，输出已经通过的部分。")
            current = last_good
            break
        notes[3].append(
            f"通过：{gate.get('id')}「{gate.get('caption')}」，"
            f"朝向已对齐={bool(gate.get('axis_converted'))}。"
        )
        unit = _donor_unit(donor)
        notes[4].append(f"{action} 取参考里的一段有效动作，{unit.shape[0]} 帧。")
        joined = None
        best_seam = 99.0
        best_ov = 8
        for overlap in (8, 12, 16):
            trial, seam = _transition(current, unit, overlap)
            if seam < best_seam:
                best_seam = seam
                best_ov = overlap
                joined = trial
            if seam <= SEAM_LIMIT:
                break
        notes[5].append(f"新动作接在当前片段后面，过渡 {best_ov} 帧。")
        notes[6].append(
            f"过渡接缝是两侧动作最快速度的 {best_seam:.2f} 倍（上限 {SEAM_LIMIT:.2f}）。"
        )
        notes[7].append(f"过渡试了 8、12、16 帧，保留 {best_ov} 帧。")
        seam_ratio = round(float(best_seam), 3)
        if joined is None or best_seam > SEAM_LIMIT:
            method = "assemble_partial" if any_change else "rejected_seam"
            notes[8].append("过渡接缝没有通过，新动作不进入预览，停在上一份结果。")
            current = last_good
            break
        current = joined
        last_good = joined
        any_change = True

    if method == "assemble_ok":
        notes[8].append(f"通过。输出 {current.shape[0]} 帧。")
    elif not notes[8]:
        notes[8].append(f"输出 {current.shape[0]} 帧。")

    return current, {
        "method": method,
        "segments": segments,
        "stages": _stages(notes),
        "seam_ratio": seam_ratio,
        "T_out": int(current.shape[0]),
        "target_units": None,
    }
