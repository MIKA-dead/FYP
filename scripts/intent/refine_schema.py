"""
Closed Refine Schema — the only vocabulary execution understands.

NL (LLM / heuristic) and UI timeline clicks both normalize into this shape.
"""
from __future__ import annotations

import re
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
    "repeat_cycle",
    "retrieve",
    "arm_raise",
    "sit_deepen",
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


def _tempo_scale_from_text(text: str) -> Optional[float]:
    """Tempo multiplier from wording alone (2.0 = twice as fast, 0.8 = slower).

    Used when the refine names ``phase_speed`` but supplies no map: the timeline
    ids are only known later, so the caller stores the scale under ``*``.
    """
    raw = text or ""
    slow = re.search(r"slow|slower|slow\s+down|慢|缓", raw, flags=re.I)
    fast = re.search(r"fast|faster|speed\s+up|hurry|快|加速", raw, flags=re.I)
    if bool(slow) == bool(fast):
        return None
    strong = re.search(r"much|a\s+lot|very|太|很多|大幅|显著", raw, flags=re.I)
    if slow:
        return 0.6 if strong else 0.8
    return 1.6 if strong else 1.3


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
        # Metres to lower the seated pelvis. 0 = no change. Applied by
        # sit_deepen_postprocess with a 2-bone leg IK so the feet stay planted;
        # without the IK the geometric sit operator re-grounds the feet and
        # cancels the drop, which made "sit lower" a silent no-op.
        "sit_deepen": 0.0,
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
    # The LLM sometimes names the op but omits the map ("slow down a bit" →
    # ops=[phase_speed], phase_speed={}), which made the edit a silent no-op that
    # still scored 100 because nothing changed. Fall back to the wording.
    if "phase_speed" in ops and not phase_speed:
        text = str(raw_prompt if raw_prompt is not None else (raw.get("raw_prompt") or ""))
        scale = _tempo_scale_from_text(text)
        if scale is not None:
            # The concrete timeline ids are only known after the clip is analysed
            # (start/walk_slow/descend/...), so store a wildcard rather than the
            # default sit phases, which do not exist on a walk clip.
            phase_speed = {"*": scale}
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

    # "sit lower / deeper into the seat" — a depth change the geometric sit
    # operator cannot express (it re-grounds the feet, cancelling the drop).
    deepen = 0.0
    try:
        deepen = float(raw.get("sit_deepen") or 0.0)
    except (TypeError, ValueError):
        deepen = 0.0
    prompt_text = str(
        raw_prompt if raw_prompt is not None else (raw.get("raw_prompt") or "")
    ).lower()
    if deepen <= 0 and str(base["keep_action"]).startswith("sit"):
        depth_words = re.search(
            r"\b(sit|sitting|seat)\b.*\b(lower|deeper)\b"
            r"|\b(lower|deeper)\b.*\b(sit|seat)\b"
            r"|\b(lower|deeper)\b.*\b(hip|hips|pelvis)\b",
            prompt_text,
        )
        further_down = re.search(r"\b(further down|down more|more down|lower down)\b", prompt_text)
        it_is_slower = re.search(r"slow|slower|慢", prompt_text) and not re.search(
            r"\b(lower|deeper)\b", prompt_text
        )
        if (depth_words or further_down) and not it_is_slower:
            # Match the amount to the wording instead of always 0.10 m: the
            # acceptance check asks for a visible drop (>= 0.15 m), and a
            # second identical edit to reach it wastes the edit budget.
            strong = re.search(
                r"\b(much|a lot|lots|far|further|more|way|really|very)\b|很多|再低", prompt_text
            )
            deepen = 0.18 if strong else 0.15
    deepen = max(0.0, min(0.18, deepen))
    if deepen > 0 and str(base["keep_action"]).startswith("sit"):
        ops = _uniq(["geometric_only"] + [o for o in ops if o != "diffusion"] + ["sit_deepen"])
        # A depth-only request must not inherit a trim or a retime. The LLM
        # refine tends to add post_sit_extra/trim_trailing, which then cut an
        # 86-frame clip to 39 while claiming to lower the seat.
        asks_trim = re.search(
            r"trim|cut|remove|shorten|delete|end after|too long|删|去掉|多余|截断",
            prompt_text,
        )
        asks_tempo = re.search(r"slow|slower|fast|faster|quick|慢|快", prompt_text)
        if not asks_trim:
            ops = [
                o for o in ops
                if o not in ("trim_preamble", "trim_trailing", "drop_phases")
            ]
            forbid = [f for f in forbid if f not in ("post_sit_extra", "pre_sit_locomotion")]
        if not asks_tempo:
            phase_speed = {}
            ops = [o for o in ops if o != "phase_speed"]

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
        "raw_prompt": raw_prompt if raw_prompt is not None else (raw.get("raw_prompt") or ""),
        "phase_local": phase_local,
    }
    # A turn clip's yaw *is* the action. Inheriting "yaw_spin" from the previous
    # plan let constraint_postprocess straighten it out (45.8° → 0.0°), which then
    # failed turn_yaw_span — the edit destroyed the very action it edited.
    if str(out["keep_action"]).startswith("turn"):
        out["forbid"] = [f for f in out["forbid"] if f != "yaw_spin"]
    if phase_speed:
        out["phase_speed"] = phase_speed
    if drop_phases:
        out["drop_phases"] = _uniq(drop_phases)
    if deepen > 0 and str(out["keep_action"]).startswith("sit"):
        out["sit_deepen"] = round(deepen, 3)

    retrieval_text = str(raw.get("retrieval_text") or "").strip()
    cycle_count = _as_cycle_count(raw.get("cycle_count"))
    repeat_more = bool(raw.get("repeat_more"))
    inferred_count, inferred_more = _cycle_request_from_text(
        raw_prompt if raw_prompt is not None else (raw.get("raw_prompt") or "")
    )
    if cycle_count is None and inferred_count is not None:
        cycle_count = inferred_count
    if inferred_more:
        repeat_more = True
    if cycle_count is not None or repeat_more:
        # A step count is not a speed change and not a diffusion caption.
        out["ops"] = [
            o
            for o in out["ops"]
            if o not in ("diffusion", "phase_speed", "trim_preamble", "trim_trailing")
        ]
        if "repeat_cycle" not in out["ops"]:
            out["ops"].append("repeat_cycle")
        if "retrieve" not in out["ops"]:
            out["ops"].append("retrieve")
        out.pop("phase_speed", None)
        out["phase_local"] = False
        if cycle_count is not None:
            out["cycle_count"] = cycle_count
        if repeat_more and cycle_count is None:
            out["repeat_more"] = True
        if not retrieval_text:
            retrieval_text = _retrieval_text_for_action(str(out["keep_action"]))
    if retrieval_text and ("retrieve" in out["ops"] or cycle_count or repeat_more):
        out["retrieval_text"] = retrieval_text[:240]
    limb_match = _limb_match_from_text(str(out.get("raw_prompt") or ""))
    if limb_match:
        # Matching one side to the other is a measurement on the current clip.
        # A diffusion caption must not replace that clip.
        out["ops"] = [o for o in out["ops"] if o != "diffusion"]
        if "geometric_only" not in out["ops"]:
            out["ops"].insert(0, "geometric_only")
        out["limb_match"] = limb_match
        out.pop("phase_speed", None)
        out["phase_local"] = False
    arm_raise = _arm_raise_from_text(str(out.get("raw_prompt") or ""), str(out.get("keep_action") or ""))
    if arm_raise and not limb_match:
        # "raise the right arm high" is an absolute target on this clip; there is
        # no reference arm to copy, so limb_match cannot express it.
        #
        # An arm raise is local: it must not change *which part* of the clip is
        # shown. Trimming or dropping phases around it (e.g. the LLM asking for
        # only the seated hold while the user asked to raise an arm) removes the
        # motion the rest of the checks are about — a sit clip lost its descent
        # and scored 25 when the trim was allowed through.
        out["ops"] = ["geometric_only", "arm_raise"]
        out["arm_raise"] = arm_raise
        out["phases"] = ["full"]
        out["edit_span"] = "full"
        out.pop("phase_speed", None)
        out.pop("drop_phases", None)
        out["phase_local"] = False
        out["forbid"] = [
            f for f in (out.get("forbid") or [])
            if f not in ("pre_sit_locomotion", "post_sit_extra")
        ]
    out["segments"] = build_segments(
        str(out.get("raw_prompt") or ""),
        previous_action=str(out.get("keep_action") or previous_action or ""),
        forbid=list(out.get("forbid") or []),
        cycle_count=out.get("cycle_count"),
        repeat_more=bool(out.get("repeat_more")),
    )
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


def _as_cycle_count(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    if n < 1 or n > 32:
        return None
    return n


_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _cycle_request_from_text(text: str) -> tuple:
    """Pull an explicit repetition count out of the user sentence.

    This is a fallback for when the model maps 'N steps' onto phase_speed.
    The last count in the sentence is the requested target ("三步，走六步" → 6).
    """
    raw = text or ""
    found: List[int] = []
    for m in re.finditer(
        r"([0-9]+|[一二两三四五六七八九十])\s*(?:步|steps?|cycles?)",
        raw,
        flags=re.I,
    ):
        token = m.group(1)
        n = _as_cycle_count(token) if token.isdigit() else _CN_NUM.get(token)
        if n is not None:
            found.append(n)
    if found:
        return found[-1], False
    if re.search(r"多走|几步|再走|more steps|a few more steps|another few", raw, flags=re.I):
        return None, True
    return None, False


_ACTION_PATTERNS = (
    ("walk", r"走|walk"),
    ("wave", r"挥|wave"),
    ("sit", r"坐|sit"),
    ("run", r"跑|run"),
    ("jump", r"跳|jump"),
    ("turn", r"转身|转圈|turn"),
)

_SEGMENT_QUERY = {
    "walk": "a person walks forward",
    "wave": "a person waves",
    "sit": "a person sits down",
    "turn": "a person turns around",
    "run": "a person runs forward",
    "jump": "a person jumps",
}


def action_family(action: str) -> str:
    a = (action or "").lower()
    for name in ("walk", "wave", "sit", "turn", "run", "jump"):
        if name in a:
            return name
    return a or "motion"


def build_segments(
    text: str,
    *,
    previous_action: str = "",
    forbid: Optional[List[str]] = None,
    cycle_count: Optional[int] = None,
    repeat_more: bool = False,
) -> List[Dict[str, Any]]:
    """Ordered segments: action, count, constraints, extend-or-attach.

    An extension keeps the clip on screen. A different action is attached
    only after a dataset clip passes the same skeleton gate.
    """
    forbid = list(forbid or [])
    raw = text or ""
    prev = action_family(previous_action)
    if raw.startswith("[timeline]"):
        return [_segment(prev, None, False, forbid, "extend")]

    cleaned = re.sub(
        r"转弯|不要转|别转|走直线|straight line|go straight",
        lambda m: " " * len(m.group(0)),
        raw,
        flags=re.I,
    )
    hits: List[tuple] = []
    for name, pat in _ACTION_PATTERNS:
        for match in re.finditer(pat, cleaned, flags=re.I):
            hits.append((match.start(), name))
    hits.sort()
    ordered: List[tuple] = []
    for pos, name in hits:
        if not ordered or ordered[-1][1] != name:
            ordered.append((pos, name))
    if not ordered:
        ordered = [(0, prev or "motion")]

    yaw = "yaw_spin" in forbid or bool(
        re.search(r"直线|转弯|不要转|别转|straight", raw, flags=re.I)
    )
    pieces: List[Dict[str, Any]] = []
    for i, (pos, name) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(raw)
        clause = raw[pos:end] if len(ordered) > 1 else raw
        count, more = _cycle_request_from_text(clause)
        if len(ordered) == 1:
            if cycle_count is not None:
                count = cycle_count
            if repeat_more:
                more = True
        constraints: List[str] = []
        if yaw and name in ("walk", "run", "turn", "motion"):
            constraints.append("yaw_spin")
        relation = "extend" if name == prev else "attach"
        pieces.append(_segment(name, count, bool(more and count is None), constraints, relation))
    return pieces


def _segment(
    action: str,
    count: Optional[int],
    repeat_more: bool,
    constraints: List[str],
    relation: str,
) -> Dict[str, Any]:
    return {
        "action": action,
        "count": count,
        "repeat_more": repeat_more,
        "constraints": list(constraints),
        "relation": relation,
        "reference": "current" if relation == "extend" else "dataset",
        "retrieval_text": _SEGMENT_QUERY.get(action, "a person " + action.replace("_", " ")),
    }


def _limb_match_from_text(text: str) -> Optional[Dict[str, str]]:
    """One side should lift like the other. Decided from the sentence, applied to any clip."""
    raw = text or ""
    if not re.search(
        r"一样|对称|镜像|match|mirror|symmetr|as\s+(?:much|high)\s+as",
        raw,
        flags=re.I,
    ):
        return None
    if re.search(r"手|臂|arm|hand", raw, flags=re.I) and not re.search(r"腿|脚|leg|foot", raw, flags=re.I):
        pair = "arm"
    else:
        pair = "leg"
    ref = None
    target = None
    if re.search(r"和左|像左|跟左|与左|like the left|match(?:\s+the)?\s+left", raw, flags=re.I):
        ref = "left"
    elif re.search(r"和右|像右|跟右|与右|like the right|match(?:\s+the)?\s+right", raw, flags=re.I):
        ref = "right"
    if re.search(r"右(?:腿|脚|手|臂)|right\s+(?:leg|foot|arm|hand)", raw, flags=re.I):
        target = "right"
    elif re.search(r"左(?:腿|脚|手|臂)|left\s+(?:leg|foot|arm|hand)", raw, flags=re.I):
        target = "left"
    if ref and target == ref:
        target = "left" if ref == "right" else "right"
    if ref and not target:
        target = "left" if ref == "right" else "right"
    if target and not ref:
        ref = "left" if target == "right" else "right"
    if not target and not ref:
        target, ref = "weaker", "stronger"
    return {"pair": pair, "target": str(target), "reference": str(ref)}


def _arm_raise_from_text(
    text: str, previous_action: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Absolutes for one arm: "raise the right arm high" / "抬右手过头顶".

    Fires on a raise verb plus an arm noun. Height words pick the target; with no
    height word we default to head height. A bare "higher" ("raise it higher",
    "再抬高点") has no arm noun: it is only accepted when it is clearly a step-up
    on an arm that is already raised, and the side then comes from the previous
    action when that action names an arm.
    """
    raw = text or ""
    raise_verb = re.search(
        r"raise|lift|hold\s+up|put\s+up|举起|抬起|抬高|举高|抬|举",
        raw,
        flags=re.I,
    )
    if not raise_verb:
        return None
    step_up = bool(
        re.search(
            r"higher|even\s+more|更高|再高|再抬|抬得更高|抬高点|举高点|高一点",
            raw,
            flags=re.I,
        )
    )
    has_arm_noun = bool(re.search(r"手|臂|arm|hand", raw, flags=re.I))
    if not has_arm_noun:
        # "raise it higher" / "再抬高点" carry no side. Only treat it as an arm
        # step-up; otherwise the sentence is about something else (leg, head).
        if not step_up:
            return None
        prev = str(previous_action or "").lower()
        side = "left" if "wave_left" in prev else "right"
        return {"side": side, "level": "high", "target": "high", "step_up": True}
    # Legs are a different chain; let other ops handle those.
    if re.search(r"腿|脚|膝|leg|foot|knee", raw, flags=re.I) and not has_arm_noun:
        return None

    side = "right"
    if re.search(r"左手|左臂|left\s+(?:arm|hand)", raw, flags=re.I):
        side = "left"
    elif re.search(r"右手|右臂|right\s+(?:arm|hand)", raw, flags=re.I):
        side = "right"
    elif re.search(r"双手|双臂|both|两只", raw, flags=re.I):
        side = "both"

    level = "high"
    if re.search(r"过头顶|头顶|overhead|above\s+the\s+head|above\s+head", raw, flags=re.I):
        level = "overhead"
    elif re.search(r"稍微|一点|slightly|a\s+bit", raw, flags=re.I):
        level = "slightly"
    elif re.search(r"齐肩|shoulder\s+height|mid", raw, flags=re.I):
        level = "mid"

    return {"side": side, "level": level, "target": level, "step_up": step_up}


def _retrieval_text_for_action(action: str) -> str:
    a = (action or "").lower()
    if "walk" in a:
        return "a person walks forward"
    if "sit" in a:
        return "a person sits down"
    if "turn" in a:
        return "a person turns around"
    if "wave" in a:
        return "a person walks forward and waves"
    return "a person " + a.replace("_", " ")


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
        f"  phase_local (bool; true if only editing named phases),\n"
        "  retrieval_text (English description of the motion to find in the dataset; empty if the user only constrains the current clip),\n"
        "  cycle_count (integer 1-32, or null). Set when the user asks for a number of steps, cycles, or repetitions.\n"
        "Rules:\n"
        "- phase_speed changes tempo only. Greater than 1 makes the clip shorter. Never use it to add steps.\n"
        "- If the user asks for more steps or a step count, set cycle_count and ops ['repeat_cycle','retrieve']. Do not set diffusion or phase_speed.\n"
        "- A follow-up that extends the current action stays on that clip. Name a new action only when the user adds one.\n"
        "- A request to make one side match the other stays on the current clip. Do not set diffusion for it.\n"
        "- retrieval_text is the dataset query for an action the current clip does not contain.\n"
        "- Keep forbid for constraints that must survive retrieval, such as yaw_spin for 'walk straight'.\n"
        "Do not invent new forbid ids."
    )
