"""
Closed Refine Schema — the only vocabulary execution understands.

NL (LLM / heuristic) and UI timeline clicks both normalize into this shape.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Sit + wave/walk/turn/generic timeline segments the UI may emit.
PHASE_IDS = (
    "preamble",
    "descend",
    "seated_hold",
    "trailing",
    "walk_forward",
    "walk_slow",
    "wave_right",
    "wave_left",
    "idle",
    "prep",
    "turning",
    "settle",
    "start",
    "end",
    "intro",
    "outro",
)

OPS = (
    "geometric_only",
    "restore_sit",
    "diffusion",
    "trim_preamble",
    "trim_trailing",
    "phase_speed",
    "drop_phases",
)

FORBIDS = (
    "yaw_spin",
    "post_sit_limb_rebound",
    "post_sit_stretch",
    "head_torso_misalign",
    "timeline_trim",
    "failed_sit",
    "pre_sit_locomotion",
    "post_sit_extra",
    "jerk",
)

EDIT_SPANS = ("full", "after_contact", "descend_hold")

PHASE_LABELS = {
    "preamble": "坐下前",
    "descend": "下降",
    "seated_hold": "坐稳",
    "trailing": "坐下后",
    "walk_forward": "向前走",
    "walk_slow": "慢走",
    "wave_right": "右手挥手",
    "wave_left": "左手挥手",
    "idle": "停顿",
    "prep": "准备",
    "turning": "转身",
    "settle": "收势",
    "start": "起步",
    "end": "收步",
    "intro": "前段",
    "outro": "后段",
}

UI_ACTIONS = ("delete", "slow", "fast")

_SIT_CORE = frozenset({"preamble", "descend", "seated_hold", "trailing"})


def _uniq(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_known_phase_id(phase_id: str) -> bool:
    pid = str(phase_id or "").strip()
    if not pid or pid == "full":
        return False
    if pid in PHASE_IDS:
        return True
    # Generic analyzer may use action name as a phase id (e.g. jump).
    return bool(pid.replace("_", "").isalnum()) and len(pid) <= 48


def empty_refine(*, previous_action: Optional[str] = None) -> Dict[str, Any]:
    action = previous_action or "sit_down"
    if action == "speed_revise":
        action = "sit_down"
    return {
        "keep_action": action,
        "forbid": [],
        "phases": ["descend", "seated_hold"] if str(action).startswith("sit") else ["full"],
        "edit_span": "full",
        "mf_hints": [],
        "ops": [],
        "phase_speed": {},
        "phase_local": False,
        "source": "empty",
        "raw_prompt": "",
    }


def normalize_refine(
    raw: Optional[Dict[str, Any]],
    *,
    previous_action: Optional[str] = None,
    source: Optional[str] = None,
    raw_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Clamp unknown fields; ensure ops/forbid/phases are closed-set."""
    base = empty_refine(previous_action=previous_action)
    if not isinstance(raw, dict):
        if source:
            base["source"] = source
        return base

    action = raw.get("keep_action") or previous_action or base["keep_action"]
    if action == "speed_revise":
        action = "sit_down"
    base["keep_action"] = str(action)

    forbid = [f for f in (raw.get("forbid") or []) if f in FORBIDS]
    ops = [o for o in (raw.get("ops") or []) if o in OPS]
    phases = [
        p
        for p in (raw.get("phases") or [])
        if p == "full" or is_known_phase_id(str(p))
    ]
    if not phases:
        phases = base["phases"]

    edit_span = raw.get("edit_span") or "full"
    if edit_span not in EDIT_SPANS:
        edit_span = "full"

    phase_speed: Dict[str, float] = {}
    for k, v in dict(raw.get("phase_speed") or {}).items():
        if not is_known_phase_id(str(k)):
            continue
        try:
            phase_speed[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    drop_phases = [
        str(p)
        for p in (raw.get("drop_phases") or [])
        if is_known_phase_id(str(p))
    ]

    mf_hints = [str(h) for h in (raw.get("mf_hints") or []) if h][:8]

    # Derive ops from phase_speed / forbid if missing
    if phase_speed and "phase_speed" not in ops:
        ops.append("phase_speed")
    if drop_phases and "drop_phases" not in ops:
        ops.append("drop_phases")
    if "pre_sit_locomotion" in forbid and "trim_preamble" not in ops:
        ops.append("trim_preamble")
    if "post_sit_extra" in forbid and "trim_trailing" not in ops:
        ops.append("trim_trailing")
    if ops and "diffusion" not in ops and "geometric_only" not in ops:
        ops.insert(0, "geometric_only")
    if "restore_sit" in ops:
        ops = _uniq(["restore_sit", "geometric_only"] + [o for o in ops if o != "diffusion"])

    phase_local = bool(
        raw.get("phase_local")
        or phase_speed
        or drop_phases
        or any(
            o in ops
            for o in ("trim_preamble", "trim_trailing", "phase_speed", "drop_phases")
        )
    )

    out: Dict[str, Any] = {
        "keep_action": base["keep_action"],
        "forbid": _uniq(forbid),
        "phases": phases,
        "edit_span": edit_span,
        "mf_hints": _uniq(mf_hints),
        "ops": _uniq(ops),
        "source": source or raw.get("source") or "normalized",
        "raw_prompt": raw_prompt if raw_prompt is not None else (raw.get("raw_prompt") or ""),
        "phase_local": phase_local,
    }
    if phase_speed:
        out["phase_speed"] = phase_speed
    if drop_phases:
        out["drop_phases"] = _uniq(drop_phases)
    return out


def refine_from_phase_action(
    phase_id: str,
    action: str,
    *,
    previous_action: Optional[str] = None,
    speed_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """UI timeline click → Refine JSON (no natural language)."""
    if not is_known_phase_id(phase_id):
        raise ValueError(f"unknown phase_id: {phase_id}")
    if action not in UI_ACTIONS:
        raise ValueError(f"unknown ui action: {action}")

    label = PHASE_LABELS.get(phase_id, phase_id)
    refine = empty_refine(previous_action=previous_action)
    refine["source"] = "ui"
    refine["phases"] = [phase_id]
    refine["phase_local"] = True
    refine["ops"] = ["geometric_only"]
    refine["raw_prompt"] = f"[timeline] {action} {phase_id}"

    if action == "delete":
        if phase_id == "preamble":
            refine["forbid"] = ["pre_sit_locomotion"]
            refine["ops"] = ["geometric_only", "trim_preamble"]
            refine["edit_span"] = "descend_hold"
            refine["mf_hints"] = ["sit down in place immediately, no walking before sitting"]
        elif phase_id == "trailing":
            refine["forbid"] = ["post_sit_extra"]
            refine["ops"] = ["geometric_only", "trim_trailing"]
            refine["mf_hints"] = ["end after seated hold, no extra motion after sitting"]
        elif phase_id in _SIT_CORE:
            # Core sit phases: delete ≈ shorten
            refine["phase_speed"] = {phase_id: 1.6}
            refine["ops"] = ["geometric_only", "phase_speed"]
            refine["mf_hints"] = [f"shorten the {label} phase"]
        else:
            # Non-sit (wave/walk/turn/…): cut the segment out
            refine["drop_phases"] = [phase_id]
            refine["ops"] = ["geometric_only", "drop_phases"]
            refine["mf_hints"] = [f"remove the {label} phase"]
    elif action == "slow":
        scale = float(speed_scale) if speed_scale is not None else 0.5
        refine["phase_speed"] = {phase_id: scale}
        refine["ops"] = ["geometric_only", "phase_speed"]
        refine["mf_hints"] = [f"slow down {label}"]
    else:  # fast
        scale = float(speed_scale) if speed_scale is not None else 1.4
        refine["phase_speed"] = {phase_id: scale}
        refine["ops"] = ["geometric_only", "phase_speed"]
        refine["mf_hints"] = [f"speed up {label}"]

    return normalize_refine(refine, previous_action=previous_action, source="ui")


def schema_for_llm_prompt() -> str:
    return (
        "Return ONLY a JSON object with keys:\n"
        f"  keep_action (string),\n"
        f"  forbid (subset of {list(FORBIDS)}),\n"
        f"  phases (known timeline phase ids or ['full']),\n"
        f"  edit_span (one of {list(EDIT_SPANS)}),\n"
        f"  mf_hints (short English strings),\n"
        f"  ops (subset of {list(OPS)}),\n"
        f"  phase_speed (optional map phase_id→float; <1 slower, >1 faster),\n"
        f"  drop_phases (optional list of phase ids to remove),\n"
        f"  phase_local (bool; true if only editing named phases).\n"
        "Map user feedback about timeline segments to forbid/ops/phase_speed/drop_phases.\n"
        "Do not invent new forbid/ops ids."
    )
