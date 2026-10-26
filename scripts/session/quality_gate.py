"""
Post-edit quality gate for MotionsFix / v05 session outputs.

Checks (action-aware):
  - finite joints, length
  - foot / root slide (sit & turn should stay put)
  - sit: pelvis drops
  - wave: wrist vertical span
  - turn: facing yaw changes
  - slow: root travel slower than typical walk band

  python scripts/session/quality_gate.py \\
      --edited outputs/v05/session_sit/gen/03_edited.npy \\
      --action sit_down \\
      --out outputs/v05/session_sit/gen/04_quality.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# SMPL-22 / HumanML3D
J_PELVIS, J_HEAD = 0, 15
J_L_FOOT, J_R_FOOT = 10, 11
J_L_WRIST, J_R_WRIST = 20, 21
J_L_SHOULDER, J_R_SHOULDER = 16, 17


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(hp[:, 2]).mean() > abs(hp[:, 1]).mean() else 1


def _horiz(up: int) -> List[int]:
    return [0, 1] if up == 2 else [0, 2]


def _yaw(joints: np.ndarray) -> np.ndarray:
    """Facing angle from shoulder girdle (fallback: hips)."""
    sh = joints[:, J_R_SHOULDER] - joints[:, J_L_SHOULDER]
    # SMPL-22: 1=L_hip, 2=R_hip
    hips = joints[:, 2] - joints[:, 1]
    # Prefer the axis with larger horizontal extent (more stable).
    sh_h = np.linalg.norm(sh[:, :2], axis=-1).mean()
    hip_h = np.linalg.norm(hips[:, :2], axis=-1).mean()
    v = sh if sh_h >= hip_h else hips
    return np.arctan2(v[:, 1], v[:, 0])


def load_joints(path: Path) -> Tuple[np.ndarray, str]:
    """Load (T,22,3) from edited / bridge / joints npy."""
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        obj = raw.item()
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected object in {path}")
        for key in ("joint_positions", "joints", "joints_mld"):
            if key in obj:
                jp = obj[key]
                if hasattr(jp, "numpy"):
                    jp = jp.numpy()
                arr = np.asarray(jp, dtype=np.float32)
                if arr.ndim == 3 and arr.shape[1] == 22:
                    return arr, key
        if "motion_source" in obj and "joint_positions" in obj["motion_source"]:
            jp = obj["motion_source"]["joint_positions"]
            if hasattr(jp, "numpy"):
                jp = jp.numpy()
            return np.asarray(jp, dtype=np.float32), "motion_source.joint_positions"
        raise ValueError(
            f"{path} has no joint_positions (keys={list(obj.keys())}); "
            "re-run edit/viz so FK joints are saved"
        )
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 22:
        return arr, "joints"
    raise ValueError(f"Expected (T,22,3), got {arr.shape} from {path}")


def _check(
    name: str,
    ok: bool,
    detail: str,
    *,
    hard: bool = True,
) -> Dict[str, Any]:
    return {"id": name, "ok": bool(ok), "hard": hard, "detail": detail}


def _sit_contact_frame(pelvis_z: np.ndarray) -> int:
    T = len(pelvis_z)
    drop = float(pelvis_z[0] - pelvis_z.min())
    if drop < 1e-4:
        return max(T // 2, 1)
    target = pelvis_z[0] - 0.85 * drop
    reached = np.where(pelvis_z <= target + 1e-6)[0]
    contact = int(reached[0]) if len(reached) else int(np.argmin(pelvis_z))
    return int(np.clip(contact, int(0.15 * T), T - 2))


def evaluate_joints(
    joints: np.ndarray,
    action: Optional[str] = None,
    refine: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = (action or "").strip() or None
    refine = refine or {}
    forbid = set(refine.get("forbid") or [])
    checks: List[Dict[str, Any]] = []
    T = int(joints.shape[0])
    up = _up_axis(joints)
    hz = _horiz(up)

    timeline: Optional[Dict[str, Any]] = None
    try:
        from motion_timeline import analyze_motion_timeline

        timeline = analyze_motion_timeline(joints, action=action, forbid=list(forbid))
    except Exception:  # noqa: BLE001
        timeline = None

    finite = bool(np.isfinite(joints).all())
    checks.append(_check("finite", finite, f"isfinite={finite}"))

    len_ok = T >= 30
    checks.append(_check("min_frames", len_ok, f"T={T} (need ≥30)"))

    root = joints[:, J_PELVIS]
    travel = float(np.linalg.norm(root[-1, hz] - root[0, hz]))
    step = np.linalg.norm(np.diff(root[:, hz], axis=0), axis=-1)
    step_p95 = float(np.percentile(step, 95)) if len(step) else 0.0

    pelvis_z = root[:, up]
    pelvis_drop = float(pelvis_z[0] - pelvis_z.min())  # positive if drops
    head_rel = float((joints[:, J_HEAD, up] - joints[:, J_PELVIS, up]).mean())

    feet = 0.5 * (joints[:, J_L_FOOT] + joints[:, J_R_FOOT])
    foot_slide = float(np.linalg.norm(np.diff(feet[:, hz], axis=0), axis=-1).mean())

    lw = joints[:, J_L_WRIST, up]
    rw = joints[:, J_R_WRIST, up]
    wrist_span = float(max(np.ptp(lw), np.ptp(rw)))

    yaw = np.unwrap(_yaw(joints))
    yaw_span_deg = float(np.degrees(np.ptp(yaw)))

    contact = _sit_contact_frame(pelvis_z) if action in ("sit_down", "sit") else T // 2
    hold = joints[contact:]
    # Rebound: pelvis rises after contact
    rebound = float(pelvis_z[contact:].max() - pelvis_z[contact]) if contact < T - 1 else 0.0
    # Limb speed in hold window (wrists)
    if hold.shape[0] > 2:
        wvel = np.linalg.norm(
            np.diff(hold[:, [J_L_WRIST, J_R_WRIST]], axis=0), axis=-1
        ).mean()
    else:
        wvel = 0.0

    metrics = {
        "T": T,
        "up_axis": up,
        "root_travel": round(travel, 4),
        "root_step_p95": round(step_p95, 4),
        "pelvis_drop": round(pelvis_drop, 4),
        "head_rel_height": round(head_rel, 4),
        "foot_slide_mean": round(foot_slide, 4),
        "wrist_up_span": round(wrist_span, 4),
        "yaw_span_deg": round(yaw_span_deg, 2),
        "sit_contact_frame": int(contact),
        "sit_rebound": round(rebound, 4),
        "sit_hold_wrist_vel": round(float(wvel), 5),
        "refine_forbid": sorted(forbid),
    }

    # --- action-specific ---
    if action in ("sit_down", "sit"):
        # In-place sit: little XY travel; pelvis should drop.
        slide_ok = travel < 0.55 and step_p95 < 0.08
        checks.append(
            _check(
                "sit_anti_slide",
                slide_ok,
                f"travel={travel:.3f} step_p95={step_p95:.3f} (want travel<0.55, p95<0.08)",
            )
        )
        drop_ok = pelvis_drop > 0.08
        checks.append(
            _check(
                "sit_pelvis_drop",
                drop_ok,
                f"pelvis_drop={pelvis_drop:.3f} (want >0.08)",
            )
        )
        # Phase: descend (pelvis drops before contact)
        descend_ok = float(pelvis_z[0] - pelvis_z[contact]) > 0.06
        checks.append(
            _check(
                "sit_phase_descend",
                descend_ok,
                f"drop_to_contact={pelvis_z[0]-pelvis_z[contact]:.3f} @f{contact} (want >0.06)",
            )
        )
        # Phase: seated hold — low rebound & limb motion after contact
        rebound_lim = 0.06 if "post_sit_limb_rebound" in forbid else 0.12
        hold_ok = rebound < rebound_lim
        checks.append(
            _check(
                "sit_phase_seated_hold",
                hold_ok,
                f"rebound={rebound:.3f} (want <{rebound_lim})",
                hard="post_sit_limb_rebound" in forbid or "post_sit_stretch" in forbid,
            )
        )
        wvel_lim = 0.025 if "post_sit_limb_rebound" in forbid else 0.06
        limb_ok = float(wvel) < wvel_lim
        checks.append(
            _check(
                "sit_hold_no_stretch",
                limb_ok,
                f"hold_wrist_vel={wvel:.4f} (want <{wvel_lim})",
                hard="post_sit_stretch" in forbid or "post_sit_limb_rebound" in forbid,
            )
        )
        # Yaw: sit should not be a spin (hard only when clearly still spinning a lot)
        yaw_lim = 70.0 if "yaw_spin" in forbid else 120.0
        yaw_ok = yaw_span_deg < yaw_lim
        checks.append(
            _check(
                "sit_yaw_limited",
                yaw_ok,
                f"yaw_span={yaw_span_deg:.1f}° (want <{yaw_lim})",
                hard=("yaw_spin" in forbid and yaw_span_deg > 110.0),
            )
        )
        # Preamble walk before sit
        if timeline is not None:
            pre_t = float(timeline.get("pre_travel") or 0)
            sit_start = int(timeline.get("sit_start") or 0)
            pre_lim = 0.15 if "pre_sit_locomotion" in forbid else 0.35
            pre_ok = pre_t < pre_lim
            checks.append(
                _check(
                    "sit_no_preamble_walk",
                    pre_ok,
                    f"pre_travel={pre_t:.3f} sit_start=f{sit_start} (want <{pre_lim})",
                    hard="pre_sit_locomotion" in forbid,
                )
            )
            metrics["sit_start"] = sit_start
            metrics["pre_travel"] = round(pre_t, 4)
            metrics["hold_end"] = int(timeline.get("hold_end") or T)
    elif action in ("turn_in_place", "turn"):
        slide_ok = travel < 0.65
        checks.append(
            _check(
                "turn_anti_slide",
                slide_ok,
                f"travel={travel:.3f} (want <0.65)",
            )
        )
        # MotionsFix relative edits often under-rotate vs literal 180°;
        # hard gate = detectable turn, soft = large turn.
        yaw_ok = yaw_span_deg > 20.0
        checks.append(
            _check(
                "turn_yaw_span",
                yaw_ok,
                f"yaw_span={yaw_span_deg:.1f}° (want >20)",
            )
        )
        checks.append(
            _check(
                "turn_yaw_large",
                yaw_span_deg > 90.0,
                f"yaw_span={yaw_span_deg:.1f}° (soft want >90 for full about-face)",
                hard=False,
            )
        )
    elif action in ("wave_right", "wave_left", "wave"):
        wave_ok = wrist_span > 0.05
        checks.append(
            _check(
                "wave_wrist_span",
                wave_ok,
                f"wrist_up_span={wrist_span:.3f} (want >0.05)",
                hard=True,
            )
        )
        # Prefer clear forward travel for walk+wave (not lateral hop / in-place).
        # Soft by default so a failed MLD/rebuild still returns a clip; wave_walk_post
        # should push travel ≥0.35 when it runs.
        forward_ok = travel > 0.35
        checks.append(
            _check(
                "wave_forward_travel",
                forward_ok,
                f"travel={travel:.3f} (want >0.35 forward walk+wave)",
                hard=False,
            )
        )
        soft = travel < 3.5
        checks.append(
            _check(
                "wave_root_stable",
                soft,
                f"travel={travel:.3f} (soft <3.5)",
                hard=False,
            )
        )
        # One-arm: waving wrist should move more than the other
        w_main = J_R_WRIST if action != "wave_left" else J_L_WRIST
        w_other = J_L_WRIST if w_main == J_R_WRIST else J_R_WRIST
        main_span = float(joints[:, w_main, up].max() - joints[:, w_main, up].min())
        other_span = float(joints[:, w_other, up].max() - joints[:, w_other, up].min())
        metrics["wave_main_wrist_span"] = round(main_span, 4)
        metrics["wave_other_wrist_span"] = round(other_span, 4)
        one_arm_ok = main_span > 0.08 and other_span < main_span * 0.75
        checks.append(
            _check(
                "wave_one_arm",
                one_arm_ok,
                f"main_span={main_span:.3f} other_span={other_span:.3f} (want other < 0.75*main)",
                hard=False,
            )
        )
    elif action in ("walk_slow", "walk_forward"):
        # Slow walk: some forward travel but not sprinting.
        move_ok = travel > 0.15
        checks.append(
            _check(
                "walk_has_travel",
                move_ok,
                f"travel={travel:.3f} (want >0.15)",
            )
        )
        if action == "walk_slow":
            slow_ok = step_p95 < 0.12
            checks.append(
                _check(
                    "walk_slow_step",
                    slow_ok,
                    f"step_p95={step_p95:.3f} (want <0.12 for slow)",
                    hard=False,
                )
            )
    else:
        checks.append(
            _check(
                "generic_nonempty",
                T >= 30 and finite,
                f"action={action!r} — only generic checks",
                hard=False,
            )
        )

    hard_fail = [c for c in checks if c["hard"] and not c["ok"]]
    soft_fail = [c for c in checks if (not c["hard"]) and not c["ok"]]
    passed = len(hard_fail) == 0
    score = 100
    score -= 25 * len(hard_fail)
    score -= 8 * len(soft_fail)
    score = int(max(0, min(100, score)))

    out: Dict[str, Any] = {
        "schema_version": "quality_gate_v1",
        "action": action,
        "passed": passed,
        "score": score,
        "metrics": metrics,
        "checks": checks,
        "hard_failures": [c["id"] for c in hard_fail],
        "soft_failures": [c["id"] for c in soft_fail],
    }
    if timeline is not None:
        out["timeline"] = timeline
    return out


def evaluate_file(
    edited: Path,
    action: Optional[str] = None,
    refine: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    joints, src = load_joints(edited)
    report = evaluate_joints(joints, action=action, refine=refine)
    report["edited"] = str(edited)
    report["joints_source"] = src
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Session quality gate")
    ap.add_argument("--edited", type=Path, required=True)
    ap.add_argument("--action", type=str, default=None)
    ap.add_argument("--summary-json", type=Path, default=None, help="Read primary_action from summary")
    ap.add_argument(
        "--intent-json",
        type=Path,
        default=None,
        help="MotionEditPlan / bundle with refine.forbid for phase gates",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fail-exit", action="store_true", help="Exit 1 if hard gates fail")
    args = ap.parse_args()

    action = args.action
    refine: Optional[Dict[str, Any]] = None
    if action is None and args.summary_json and args.summary_json.is_file():
        summ = json.loads(args.summary_json.read_text(encoding="utf-8-sig"))
        action = (summ.get("motion_edit_plan") or {}).get("primary_action")
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        if "motion_edit_plan" in plan:
            plan = plan["motion_edit_plan"]
        if action is None:
            action = plan.get("primary_action")
        refine = plan.get("refine") or (plan.get("revision") or {}).get("refine")

    report = evaluate_file(args.edited, action=action, refine=refine)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[quality_gate] wrote {args.out}", file=sys.stderr)
    print(
        f"[quality_gate] passed={report['passed']} score={report['score']} "
        f"hard_fail={report['hard_failures']}",
        file=sys.stderr,
    )
    if args.fail_exit and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
