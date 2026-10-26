"""
Compile structured revise constraints (refine block).

Heuristic now; swap `compile_refine` body for DeepSeek JSON later — same output shape.

Output shape::
  {
    "keep_action": "sit_down",
    "forbid": ["yaw_spin", "post_sit_limb_rebound", ...],
    "phases": ["descend", "seated_hold"],
    "edit_span": "after_contact" | "full",
    "mf_hints": ["without turning", "hold seated pose"],
    "ops": ["geometric_only" | "restore_sit" | "diffusion"],
    "phase_speed": {"descend": 0.5, ...},
    "source": "heuristic" | "llm"
  }
"""
from __future__ import annotations

import re
from typing import Any, Optional

from parse_intent import _has_any, _negated_turn

# UI timeline phase ids ↔ user phrases
_PHASE_ALIASES: dict[str, tuple[str, ...]] = {
    "preamble": (
        "坐下前",
        "坐下之前",
        "坐之前",
        "坐下去之前",
        "走动",
        "走的",
        "前面",
        "前半",
        "前段",
        "开场",
        "之前那段",
        "前面那段",
        "preamble",
        "walking before",
        "before sit",
        "before sitting",
    ),
    "descend": (
        "下降",
        "往下坐",
        "坐下过程",
        "坐下阶段",
        "下蹲",
        "descend",
        "sitting down",
    ),
    "seated_hold": (
        "坐稳",
        "保持坐",
        "坐着不动",
        "hold",
        "seated",
        "seated hold",
    ),
    "trailing": (
        "坐下后",
        "坐完",
        "后面",
        "后半",
        "后段",
        "多余",
        "结尾",
        "trailing",
        "after sit",
        "after sitting",
    ),
    "walk_forward": (
        "向前走",
        "往前走",
        "走路",
        "行走",
        "走的那段",
        "walk",
        "walking",
        "forward",
    ),
    "walk_slow": ("慢走", "慢慢走", "walk slow", "slowly walk"),
    "wave_right": (
        "右手挥",
        "挥右手",
        "右手",
        "挥手",
        "wave right",
        "right hand",
        "waving",
    ),
    "wave_left": ("左手挥", "挥左手", "左手", "wave left", "left hand"),
    "idle": ("停顿", "发呆", "站着不动", "空档", "idle", "pause", "stand still"),
    "turning": ("转身", "转向", "转圈", "turning", "turn"),
    "prep": ("准备", "起势", "prep", "wind up"),
    "settle": ("收势", "站定", "settle"),
    "start": ("起步", "开始走"),
    "end": ("收步", "停下"),
}

_DELETE_VERBS = (
    "删去",
    "删除",
    "删掉",
    "去掉",
    "裁掉",
    "裁去",
    "移除",
    "去掉",
    "不要",
    "别要",
    "砍掉",
    "去掉这段",
    "remove",
    "delete",
    "drop",
    "cut out",
    "trim off",
)

_SLOW_VERBS = (
    "减慢",
    "放慢",
    "慢一点",
    "慢些",
    "慢一点",
    "再慢",
    "变慢",
    "慢下来",
    "slow down",
    "slower",
    "slow",
)

_FAST_VERBS = (
    "加快",
    "加速",
    "快一点",
    "快些",
    "再快",
    "变快",
    "faster",
    "speed up",
    "quicken",
)


def _match_frame_range(prompt: str) -> Optional[tuple[int, int]]:
    """Parse f0-65 / 0–65 / frames 0 to 65 → (start, end) inclusive-ish."""
    raw = prompt or ""
    m = re.search(
        r"(?:f|frame|帧)?\s*(\d+)\s*[-–—~至到]\s*(?:f|frame|帧)?\s*(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        a, b = b, a
    return a, b


def _phases_overlapping_range(
    start: int,
    end: int,
    timeline: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Map a frame range to timeline phase ids (overlap ≥45% of phase or of range)."""
    phases = (timeline or {}).get("phases") or []
    if not phases:
        # No timeline: early frames → preamble; late → trailing
        if start <= 5 and end >= 20:
            return ["preamble"]
        if start >= 80:
            return ["trailing"]
        return ["preamble"] if start < 40 else []

    hits: list[str] = []
    span = max(1, end - start)
    for ph in phases:
        pid = str(ph.get("id") or "")
        if not pid:
            continue
        ps, pe = int(ph.get("start") or 0), int(ph.get("end") or 0)
        if pe <= ps:
            continue
        ov = max(0, min(end, pe) - max(start, ps))
        if ov <= 0:
            continue
        if ov / max(1, pe - ps) >= 0.45 or ov / span >= 0.45:
            hits.append(pid)
    return hits


def _match_phases(
    prompt: str,
    *,
    timeline: Optional[dict[str, Any]] = None,
) -> list[str]:
    p = (prompt or "").lower()
    raw = prompt or ""
    hit: list[str] = []
    for pid, aliases in _PHASE_ALIASES.items():
        for a in aliases:
            if a.lower() in p or a in raw:
                hit.append(pid)
                break
    # 「删去之前 / 去掉前面」无明确阶段名时，默认 preamble
    if not hit and (
        "之前" in raw
        or "前面" in raw
        or "前段" in raw
        or "开头" in raw
    ):
        hit.append("preamble")
    fr = _match_frame_range(raw)
    if fr:
        hit.extend(_phases_overlapping_range(fr[0], fr[1], timeline))
    # dedupe keep order
    out, seen = [], set()
    for x in hit:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _has_delete_intent(prompt: str) -> bool:
    return _has_any(prompt, list(_DELETE_VERBS))


def _speed_scale_for_prompt(prompt: str) -> Optional[float]:
    pl = (prompt or "").lower()
    m = re.search(r"(?:x|×|＊|\*)\s*(\d+(?:\.\d+)?)", pl)
    if m:
        v = float(m.group(1))
        return v if v > 0 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*倍", prompt or "")
    if m:
        v = float(m.group(1))
        return v if v > 0 else None
    if _has_any(prompt, list(_SLOW_VERBS)) or "慢" in (prompt or ""):
        return 0.4
    if _has_any(prompt, list(_FAST_VERBS)) or "快" in (prompt or ""):
        return 1.6
    return None


def compile_refine(
    prompt: str,
    *,
    previous_action: Optional[str] = None,
    llm_client: Any = None,
    refine_override: Optional[dict[str, Any]] = None,
    timeline: Optional[dict[str, Any]] = None,
    previous_caption: Optional[dict[str, Any]] = None,
    prefer_llm: bool = True,
) -> dict[str, Any]:
    """Compile refine JSON.

    Priority:
      1. refine_override (UI timeline clicks / structured API) — no NL
      2. LLM (DeepSeek/OpenAI-compatible) when key present
      3. Heuristic keyword fallback (offline)
    """
    from refine_schema import normalize_refine

    if refine_override is not None:
        return normalize_refine(
            refine_override,
            previous_action=previous_action,
            source=str(refine_override.get("source") or "ui"),
            raw_prompt=prompt,
        )

    client = llm_client
    if client is None and prefer_llm:
        try:
            from llm_refine import get_default_llm_client

            client = get_default_llm_client()
        except Exception:  # noqa: BLE001
            client = None

    if client is not None:
        try:
            out = client.compile_refine(
                prompt,
                previous_action=previous_action,
                timeline=timeline,
                previous_caption=previous_caption,
            )
            if isinstance(out, dict) and out.get("keep_action"):
                return normalize_refine(
                    out,
                    previous_action=previous_action,
                    source=out.get("source") or "llm",
                    raw_prompt=prompt,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[compile_refine] LLM failed, heuristic fallback: {exc}", flush=True)

    return normalize_refine(
        _heuristic_refine(
            prompt, previous_action=previous_action, timeline=timeline
        ),
        previous_action=previous_action,
        source="heuristic",
        raw_prompt=prompt,
    )


def _heuristic_refine(
    prompt: str,
    *,
    previous_action: Optional[str] = None,
    timeline: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    p = prompt or ""
    action = previous_action or "sit_down"
    if action == "speed_revise":
        action = "sit_down"

    forbid: list[str] = []
    ops: list[str] = []
    phases = ["descend", "seated_hold"] if str(action).startswith("sit") else ["full"]
    edit_span = "full"
    mf_hints: list[str] = []
    phase_speed: dict[str, float] = {}

    # --- explicit timeline phase edits: 「删去坐下前」「下降部分减慢」「删去f0-65」---
    phase_hits = _match_phases(p, timeline=timeline)
    want_delete = _has_delete_intent(p)
    want_phase_speed = bool(phase_hits) and (
        _has_any(p, list(_SLOW_VERBS) + list(_FAST_VERBS))
        or (
            any(k in p for k in ("部分", "这段", "那一段", "阶段"))
            and ("慢" in p or "快" in p)
        )
    )

    if phase_hits and want_delete:
        for pid in phase_hits:
            if pid == "preamble":
                forbid.append("pre_sit_locomotion")
                ops.append("trim_preamble")
                mf_hints.append("sit down in place immediately, no walking before sitting")
                edit_span = "descend_hold"
            elif pid == "trailing":
                forbid.append("post_sit_extra")
                ops.append("trim_trailing")
                mf_hints.append("end after seated hold, no extra motion after sitting")
            elif pid in ("descend", "seated_hold"):
                # Core sit phases: "delete" ≈ shorten
                phase_speed[pid] = 1.6
                ops.append("phase_speed")
                mf_hints.append(f"shorten the {pid} phase")
        ops.append("geometric_only")
        phases = phase_hits

    if want_phase_speed:
        scale = _speed_scale_for_prompt(p) or 0.5
        for pid in phase_hits:
            # Prefer explicit slow/fast over delete-shorten if both present
            if want_delete and pid in ("preamble", "trailing"):
                continue
            phase_speed[pid] = scale
        if phase_speed:
            ops.append("phase_speed")
            ops.append("geometric_only")
            phases = list(dict.fromkeys(list(phases) + phase_hits))
            mf_hints.append(
                ("slow down " if scale < 1 else "speed up ") + ", ".join(phase_hits)
            )

    # --- restore sit / clip destroyed ---
    if _has_any(
        p,
        [
            "没有坐下",
            "没坐下",
            "根本没有坐",
            "无法坐下",
            "不是坐下",
            "坐不下去",
            "站着",
            "站立",
            "not sitting",
            "no sit",
            "lost the sit",
        ],
    ):
        forbid.append("failed_sit")
        ops.append("restore_sit")
        mf_hints.append("sit down fully with pelvis lowering, then hold")
        phases = ["descend", "seated_hold"]

    # --- timeline trimmed / skipped ---
    if _has_any(
        p,
        [
            "跳过",
            "被跳过",
            "砍掉",
            "截断",
            "太短",
            "少了一段",
            "后半段没了",
            "动作没了",
            "skipped",
            "trimmed",
            "cut off",
            "missing part",
        ],
    ):
        forbid.append("timeline_trim")
        ops.append("geometric_only")
        mf_hints.append("keep full sit timeline, do not skip frames after sitting")

    # --- head / torso alignment ---
    if _has_any(
        p,
        [
            "头",
            "头部",
            "回正",
            "扭头",
            "歪头",
            "头和躯",
            "头没有",
            "face forward",
            "head",
            "look forward",
        ],
    ):
        forbid.append("head_torso_misalign")
        ops.append("geometric_only")
        mf_hints.append("align head with torso facing forward")

    # --- spin ---
    if _negated_turn(p) or _has_any(p, ["转一圈", "转圈", "90度", "九十度", "转动", "转身"]):
        # "转身" alone in quality complaint still means remove spin when previous is sit
        if _negated_turn(p) or _has_any(p, ["转一圈", "转圈", "去掉转", "不要转", "别转"]):
            forbid.append("yaw_spin")
            ops.append("geometric_only")
            mf_hints.append("without turning or spinning")
            edit_span = "full"

    # --- walk / steps before sit (preamble) ---
    if _has_any(
        p,
        [
            "走几步",
            "走兩步",
            "走两步",
            "走几",
            "走动",
            "先走",
            "走再坐",
            "走了再",
            "迈步",
            "踱步",
            "挪几步",
            "多走",
            "去掉走",
            "不要走",
            "别走",
            "删掉走",
            "没删掉走",
            "还是没删",
            "walk before",
            "walks then",
            "few steps",
            "steps then sit",
            "直接坐下",
        ],
    ):
        forbid.append("pre_sit_locomotion")
        ops.append("geometric_only")
        ops.append("trim_preamble")
        mf_hints.append("sit down in place immediately, no walking steps before sitting")
        edit_span = "descend_hold"
        phases = ["descend", "seated_hold"]

    # --- extra motion after sit ---
    if _has_any(
        p,
        [
            "坐下后还",
            "坐完还",
            "后面还有",
            "多余动作",
            "额外动作",
            "after sitting",
            "extra after",
        ],
    ):
        forbid.append("post_sit_extra")
        ops.append("geometric_only")
        ops.append("trim_trailing")
        mf_hints.append("end after seated hold, no extra motion after sitting")

    # --- rebound / stretch ---
    if _has_any(
        p,
        [
            "回弹",
            "伸展",
            "伸直",
            "四肢",
            "手臂甩",
            "只要坐下",
            "只要坐",
            "rebound",
            "stretch",
            "only sit",
        ],
    ):
        forbid.append("post_sit_limb_rebound")
        forbid.append("post_sit_stretch")
        ops.append("geometric_only")
        mf_hints.append("hold the seated pose, no limb stretch or rebound after sitting")
        edit_span = "after_contact"

    if _has_any(p, ["抽搐", "不自然", "更自然", "卡顿", "smooth", "jerk", "twitch"]):
        forbid.append("jerk")
        ops.append("geometric_only")
        mf_hints.append("smooth natural motion")

    if _has_any(p, ["弯着腰", "弯腰", "跪"]):
        forbid.append("failed_sit")
        ops.append("restore_sit")
        mf_hints.append("sit fully with pelvis lowering into a chair")

    # --- wave / walk quality complaints ---
    if action in ("wave_right", "wave_left", "wave") or str(action).startswith("wave"):
        if _has_any(
            p,
            [
                "没有向前",
                "没向前",
                "没有走",
                "没走",
                "不走",
                "原地",
                "站着挥",
                "walk forward",
                "not walking",
                "no walk",
                "doesn't walk",
            ],
        ):
            ops.append("diffusion")
            mf_hints.append("walk forward while waving")
        if _has_any(
            p,
            [
                "双臂",
                "两只手",
                "双手",
                "两边",
                "左手也",
                "右手也",
                "both arms",
                "both hands",
                "two hands",
            ],
        ):
            ops.append("diffusion")
            if "left" in str(action):
                mf_hints.append("wave only with the left hand, right arm relaxed at the side")
            else:
                mf_hints.append("wave only with the right hand, left arm relaxed at the side")
        if not forbid and not ops:
            ops.append("diffusion")
            if "left" in str(action):
                mf_hints.append("walk forward and wave only with the left hand")
            else:
                mf_hints.append("walk forward and wave only with the right hand")

    # Default sit revise with empty forbid: still geometric hold, never blank recipe
    if str(action).startswith("sit") and not forbid and not ops:
        ops.append("geometric_only")
        mf_hints.append("keep a clear sit-down then seated hold")
        forbid.append("jerk")

    # Deduplicate
    def uniq(xs: list[str]) -> list[str]:
        out, s = [], set()
        for x in xs:
            if x not in s:
                s.add(x)
                out.append(x)
        return out

    # Prefer restore over stacking diffusion on a broken clip
    if "restore_sit" in ops:
        ops = uniq(["restore_sit", "geometric_only"] + [o for o in ops if o != "diffusion"])
    elif ops and "diffusion" not in ops:
        ops = uniq(["geometric_only"] + ops)

    out: dict[str, Any] = {
        "keep_action": action if action != "speed_revise" else "sit_down",
        "forbid": uniq(forbid),
        "phases": phases,
        "edit_span": edit_span,
        "mf_hints": uniq(mf_hints),
        "ops": uniq(ops),
        "source": "heuristic",
        "raw_prompt": p,
    }
    if phase_speed:
        out["phase_speed"] = phase_speed
    if phase_speed or any(
        o in ops for o in ("trim_preamble", "trim_trailing", "phase_speed")
    ):
        out["phase_local"] = True  # planner: not whole-clip speed_revise
    return out
