"""Measure a request against a clip's joints.

This is a *tool*, not a hidden judge. The controller is required to call
``check_requirement`` before it may ``finish``; the verdict is returned to the
model as an observation, and the model decides what to do with it. The episode
records both the model's claim and this verdict, so the benchmark can report
whether the agent was calibrated instead of silently overriding it.

The caption is deliberately never used here: it is coarse and sometimes wrong,
and a model that trusts a wrong caption would be rewarded for agreeing with it.
"""
from __future__ import annotations

from typing import Any, Dict, List


def verify_goal(goal: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    m = obs.get("metrics") or {}
    g = (goal or "").lower()
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any, need: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "value": value, "expected": need})

    side = "left" if "left" in g else "right"
    if any(w in g for w in ("arm", "hand", "wave", "raise", "overhead")):
        high = any(w in g for w in ("high", "above", "overhead", "up"))
        need = 1.8 if high else 0.5
        ratio = float(m.get(f"arm_raise_{side}") or 0)
        above = float(m.get(f"wrist_above_head_{side}") or 0)
        add(f"arm_raise_{side}", ratio >= need or (high and above >= 0.05), ratio,
            f">= {need}")
    in_place = any(w in g for w in ("in place", "in-place", "on the spot",
                                    "without moving", "stationary"))
    if any(w in g for w in ("walk", "forward", "step")):
        travel = float(m.get("root_travel") or 0)
        if in_place:
            # The request is the opposite of locomotion: the root must stay put.
            add("root_travel", travel <= 0.25, travel, "<= 0.25 m (in place)")
        else:
            add("root_travel", travel >= 0.5, travel, ">= 0.5 m")
    if any(w in g for w in ("sit", "sitting", "crouch")):
        drop = float(m.get("pelvis_drop") or 0)
        add("pelvis_drop", drop >= 0.15, drop, ">= 0.15 m")
    if any(w in g for w in ("turn", "spin", "rotate")):
        yaw = float(m.get("yaw_span_deg") or 0)
        add("yaw_span_deg", yaw >= 90, yaw, ">= 90 deg")
    if any(w in g for w in ("kick", "kicks", "kicking")):
        other = "left" if side == "right" else "right"
        lift = float(m.get(f"ankle_lift_{side}") or 0)
        lift_other = float(m.get(f"ankle_lift_{other}") or 0)
        add(f"ankle_lift_{side}", lift >= 0.30 and lift >= 2.0 * lift_other, lift,
            f">= 0.30 body heights and >= 2x the {other} foot")
    if any(w in g for w in ("jump", "leap", "hop")):
        rise = float(m.get("jump_rise") or 0)
        clear = float(m.get("jump_clearance") or 0)
        add("jump_rise", rise >= 0.15, rise, ">= 0.15 body heights above the start")
        add("jump_clearance", clear >= 0.08, clear,
            ">= 0.08 body heights of floor clearance at the apex")

    # A clip that is anatomically or physically broken is never acceptable.
    pen = float(m.get("ground_penetration") or 0)
    add("ground_penetration", pen <= 0.10, pen, "<= 0.10 body heights")
    bcv = float(m.get("bone_cv") or 0)
    add("bone_cv", bcv <= 0.08, bcv, "<= 0.08")

    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": not failed,
        "checks": checks,
        "reasons": [f"{c['check']}={c['value']} does not meet {c['expected']}" for c in failed],
    }
