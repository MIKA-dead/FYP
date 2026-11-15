"""Measure a request against a clip's joints.

This is a *tool*, not a hidden judge. The controller is required to call
``check_requirement`` before it may ``finish``; the verdict is returned to the
model as an observation, and the model decides what to do with it. The episode
records both the model's claim and this verdict, so the benchmark can report
whether the agent was calibrated instead of silently overriding it.

The caption is deliberately never used here: it is coarse and sometimes wrong,
and a model that trusts a wrong caption would be rewarded for agreeing with it.

Thresholds follow the user's wording, with a measurement tolerance: a request
for "90 degrees" is met by 77° or more, because the *measured* span of a clip
whose own caption says "turns left" is 89.5° while the net body rotation is
87.2°. Failing an 88° turn on a 90° request is a checker bug, not a finding.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# "90 degrees", "180 deg", "90°"
_TURN_DEG = re.compile(r"(\d{2,3})\s*(?:°|deg\b|degree)")
# Relative measurement tolerance: a 90° request passes from 77°.
TOLERANCE = 0.85



def verify_goal(goal: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    m = obs.get("metrics") or {}
    g = (goal or "").lower()
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any, need: str,
            target: float = None, higher: bool = True,
            gap: float = None) -> None:
        """Record one check.

        ``gap`` overrides the relative-shortfall formula for checks whose value
        is signed (a turn direction), where "how far short" is not a difference.
        """
        entry = {"check": name, "ok": bool(ok), "value": value, "expected": need,
                 "target": target, "higher_is_better": bool(higher)}
        if gap is not None:
            entry["gap"] = gap
        checks.append(entry)

    side = "left" if "left" in g else "right"
    if any(w in g for w in ("arm", "hand", "wave", "raise", "overhead")):
        high = any(w in g for w in ("high", "above", "overhead", "up"))
        need = 1.8 if high else 0.5
        ratio = float(m.get(f"arm_raise_{side}") or 0)
        above = float(m.get(f"wrist_above_head_{side}") or 0)
        add(f"arm_raise_{side}", ratio >= need or (high and above >= 0.05), ratio,
            f">= {need}", target=need)
    in_place = any(w in g for w in ("in place", "in-place", "on the spot",
                                    "without moving", "stationary"))
    if any(w in g for w in ("walk", "forward", "step")):
        travel = float(m.get("root_travel") or 0)
        if in_place:
            # The request is the opposite of locomotion: the root must stay put.
            add("root_travel", travel <= 0.25, travel, "<= 0.25 m (in place)",
                target=0.25, higher=False)
        else:
            add("root_travel", travel >= 0.5, travel, ">= 0.5 m", target=0.5)
    if any(w in g for w in ("sit", "sitting", "crouch")):
        drop = float(m.get("pelvis_drop") or 0)
        add("pelvis_drop", drop >= 0.15, drop, ">= 0.15 m", target=0.15)
    if any(w in g for w in ("turn", "spin", "rotate")):
        yaw = float(m.get("yaw_span_deg") or 0)
        asked = _TURN_DEG.search(g)
        if asked:
            target = float(min(360, max(45, int(asked.group(1)))))
            need = TOLERANCE * target
            note = f">= {need:.0f} deg (requested {target:.0f} deg)"
        elif any(w in g for w in ("around", "half", "180")):
            target = 150.0
            need = target
            note = f">= {need:.0f} deg (a turn-around)"
        else:
            target = 90.0
            need = target
            note = f">= {need:.0f} deg"
        add("yaw_span_deg", yaw >= need, yaw, note, target=need)
        net = float(m.get("yaw_net_deg") or 0)
        # Direction: span cannot tell a left turn from a right one, and the net
        # rotation is signed (positive = left). Only checked when the user named
        # a side, and the shortfall is measured against the 30° direction target.
        if re.search(r"\bleft\b", g):
            add("yaw_net_deg", net >= 30.0, net, "> +30 deg (turns left)",
                target=30.0, gap=max(0.0, (30.0 - net) / 30.0))
        elif re.search(r"\bright\b", g):
            add("yaw_net_deg", net <= -30.0, net, "< -30 deg (turns right)",
                target=30.0, higher=False,
                gap=max(0.0, (30.0 + net) / 30.0))
    if any(w in g for w in ("kick", "kicks", "kicking")):
        other = "left" if side == "right" else "right"
        lift = float(m.get(f"ankle_lift_{side}") or 0)
        lift_other = float(m.get(f"ankle_lift_{other}") or 0)
        add(f"ankle_lift_{side}", lift >= 0.30 and lift >= 2.0 * lift_other, lift,
            f">= 0.30 body heights and >= 2x the {other} foot", target=0.30)
    if any(w in g for w in ("jump", "leap", "hop")):
        rise = float(m.get("jump_rise") or 0)
        clear = float(m.get("jump_clearance") or 0)
        add("jump_rise", rise >= 0.15, rise, ">= 0.15 body heights above the start",
            target=0.15)
        add("jump_clearance", clear >= 0.08, clear,
            ">= 0.08 body heights of floor clearance at the apex", target=0.08)

    # A clip that is anatomically or physically broken is never acceptable.
    pen = float(m.get("ground_penetration") or 0)
    add("ground_penetration", pen <= 0.10, pen, "<= 0.10 body heights",
        target=0.10, higher=False)
    bcv = float(m.get("bone_cv") or 0)
    add("bone_cv", bcv <= 0.08, bcv, "<= 0.08", target=0.08, higher=False)

    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": not failed,
        "checks": checks,
        "reasons": [f"{c['check']}={c['value']} does not meet {c['expected']}" for c in failed],
    }


def requirement_gap(goal: str, obs: Dict[str, Any]) -> float:
    """How far the clip is from the requested attribute, as a relative shortfall.

    0.0 when the request is met; each failed check adds the fraction by which it
    misses its target, so ``root_travel 1.2 m`` against a 0.25 m in-place bound
    scores 3.8 while ``0.31 m`` scores 0.24. Used to rank resampled drafts: the
    checker only says pass/fail, and among failing drafts the agent still needs
    to know which is closest.
    """
    v = verify_goal(goal, obs)
    gap = 0.0
    for c in v["checks"]:
        if c["ok"] or c.get("target") in (None, 0):
            continue
        if c.get("gap") is not None:
            gap += max(0.0, float(c["gap"]))
            continue
        val = float(c["value"] or 0.0)
        tgt = float(c["target"])
        miss = (tgt - val) / tgt if c["higher_is_better"] else (val - tgt) / tgt
        gap += max(0.0, miss)
    return float(gap)
