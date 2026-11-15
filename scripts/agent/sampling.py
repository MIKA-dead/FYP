"""Generation-side rejection sampling.

The editor is *relative*: it can scale or reshape motion that is already in the
clip, but it cannot add travel, add a turn, or take locomotion away. So when a
goal names a measurable attribute (in-place walking, a 90° turn, a jump), the
only place to fix a miss is the generator itself — and a single draw cannot tell
"the model cannot do it" from "this draw missed".

This module is the shared policy for that decision. It answers three questions:

* **What must a draft satisfy?** ``acceptance(goal)`` reuses
  ``verify.verify_goal`` on an empty measurement set, so the sampling filter and
  the checker the controller consults can never disagree about what "satisfied"
  means.
* **Which draft is best?** ``rank()`` sorts by physical validity first, then by
  ``verify.requirement_gap`` (relative shortfall), then by pass/fail.
* **Keep drawing?** ``plan()`` says how many drafts to draw in one batch and how
  many batches are allowed, given the goal and the remaining budget.

Nothing here runs a model: callers pass in already-generated candidates, which
keeps the MLD process (a different conda env) out of this logic.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from verify import requirement_gap, verify_goal

# A goal with a hard, measurable constraint is worth more than one draw.
DEFAULT_BATCH = 3
MAX_BATCH = 4
# Upper bound on how many batches one generation step may draw.
MAX_BATCHES = 2


def constraints(goal: str) -> List[Dict[str, Any]]:
    """The constraint names that a draft of ``goal`` is judged on.

    Empty metrics make every check report its threshold, which is exactly the
    list of attributes the goal is graded on. Physical checks
    (``ground_penetration``, ``bone_cv``) are excluded: they are not something a
    draw is *sampled* for, they are a validity gate.
    """
    v = verify_goal(goal, {"metrics": {}})
    return [c for c in v["checks"] if c["check"] not in ("ground_penetration", "bone_cv")]


def has_constraint(goal: str) -> bool:
    return bool(constraints(goal))


def acceptance(goal: str) -> Dict[str, Any]:
    """Human/LLM-readable statement of what a draft must satisfy."""
    cons = constraints(goal)
    return {
        "goal": goal,
        "checks": [{"check": c["check"], "expected": c["expected"]} for c in cons],
        "text": "; ".join(f"{c['check']} {c['expected']}" for c in cons),
    }


def enabled(default: bool = True) -> bool:
    """Rejection sampling is on unless explicitly disabled."""
    raw = (os.environ.get("AGENT_REJECTION") or "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    return default


def batch_size(goal: str) -> int:
    """How many drafts to draw in one batch for this goal (1 = no rejection)."""
    if not enabled() or not has_constraint(goal):
        return 1
    raw = (os.environ.get("AGENT_REJECTION_BATCH") or "").strip()
    if raw.isdigit():
        return max(1, min(MAX_BATCH, int(raw)))
    return DEFAULT_BATCH


def rank(goal: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order measured candidates: usable first, then closest to the goal.

    Each candidate is a dict with ``ok``, ``valid``, and ``metrics`` (as produced
    by ``metrics.kinematic_summary``). ``gap`` and ``satisfied`` are filled in
    here so every caller ranks drafts the same way.
    """
    scored: List[Tuple[Tuple[int, float, int], Dict[str, Any]]] = []
    for c in candidates:
        if not c.get("ok", True):
            continue
        obs = {"metrics": c.get("metrics") or {}}
        v = verify_goal(goal, obs)
        gap = requirement_gap(goal, obs)
        c = {**c, "gap": round(gap, 4), "satisfied": bool(v["ok"]),
             "reasons": v["reasons"]}
        # valid first, then smallest shortfall, then already-satisfying.
        scored.append(((0 if c.get("valid", True) else 1, gap, 0 if c["satisfied"] else 1), c))
    scored.sort(key=lambda t: t[0])
    return [c for _, c in scored]


def select(goal: str, candidates: List[Dict[str, Any]]
           ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (best draft, drafts that already satisfy the goal)."""
    ranked = rank(goal, candidates)
    if not ranked:
        return None, []
    accepted = [c for c in ranked if c["satisfied"] and c.get("valid", True)]
    return ranked[0], accepted


def plan(goal: str, *, want: int = DEFAULT_BATCH, batches_left: int = MAX_BATCHES) -> Dict[str, Any]:
    """Decide the draw budget for one generation/resample step."""
    n = min(batch_size(goal), max(1, want))
    return {
        "goal": goal,
        "acceptance": acceptance(goal),
        "want": n,
        "batches": max(1, min(MAX_BATCHES, batches_left)),
        "rejection": bool(n > 1),
    }


def summary(goal: str, candidates: List[Dict[str, Any]],
            chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact record of a rejection round for the trace and the benchmark."""
    ranked = rank(goal, candidates)
    accepted = [c for c in ranked if c.get("satisfied") and c.get("valid", True)]
    return {
        "drawn": len([c for c in candidates if c.get("ok", True)]),
        "accepted": len(accepted),
        "rejected": len([c for c in ranked if not c.get("satisfied")]),
        "chosen_gap": (chosen or {}).get("gap") if chosen else None,
        "chosen_satisfied": (chosen or {}).get("satisfied") if chosen else None,
        "acceptance": acceptance(goal)["text"],
    }
