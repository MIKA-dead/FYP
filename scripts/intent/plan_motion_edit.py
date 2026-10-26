"""
Intent Schema → MotionsFix MotionEditPlan (agent layer before coder).

Sept architecture (1st stage report):
  prompt → Intent Schema → module plan → motion-coder → motion-tester
  then user revise → Intent(revise) → modules_to_rerun only → coder

This script is the planner step that MotionsFix / v05 must consume — not raw user text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "intent") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "intent"))

from parse_intent import (  # noqa: E402
    _negated_turn,
    _wants_new_action,
    parse_generate,
    parse_prompt,
)
from route_modules import route_modules  # noqa: E402


# MotionsFix test-set style relative texts + edit constraints.
ACTION_MF: dict[str, dict[str, Any]] = {
    "wave_right": {
        "mf_text": "wave to the side",
        "span": "small",
        "keep_lower": True,
        "freeze_xy": False,
        "sdedit_strength": 0.68,
        "init_from": "source",
        "gd_motion": 2.0,
        "preferred_source_id": "001318",
        "mld_text": "a person walks forward.",
    },
    "wave_left": {
        "mf_text": "wave to the side",
        "span": "small",
        "keep_lower": True,
        "freeze_xy": False,
        "sdedit_strength": 0.68,
        "init_from": "source",
        "gd_motion": 2.0,
        "preferred_source_id": "001318",
        "mld_text": "a person walks forward.",
    },
    "sit_down": {
        "mf_text": "sit down quicker.",
        "span": "large",
        "keep_lower": False,
        "freeze_xy": True,
        "sdedit_strength": 0.70,
        "init_from": "source",
        "gd_motion": 1.8,
        # MotionsFix-native pair whose relative text matches mf_text.
        "preferred_source_id": "000236",
        "mld_text": "a person stands still.",
    },
    "turn_in_place": {
        "mf_text": "dont bend and turn back 180 degrees",
        "span": "medium",
        "keep_lower": False,
        "freeze_xy": True,
        "sdedit_strength": 0.70,
        "init_from": "source",
        "gd_motion": 1.8,
        "preferred_source_id": "004922",
        "mld_text": "a person stands still.",
    },
    "walk_forward": {
        "mf_text": None,
        "span": "none",
        "keep_lower": False,
        "freeze_xy": False,
        "sdedit_strength": None,
        "init_from": "source",
        "gd_motion": 2.5,
        "edit_mode": "speed",
        "mld_text": "a person walks forward.",
    },
    "walk_slow": {
        "mf_text": "walk slowly",
        "span": "none",
        "keep_lower": False,
        "freeze_xy": False,
        "sdedit_strength": None,
        "init_from": "source",
        "gd_motion": 2.5,
        "edit_mode": "speed",
        "speed_scale": 0.45,
        "mld_text": "a person walks forward.",
    },
}

# Prefer these over locomotion when choosing the MotionsFix edit target.
EDIT_PRIORITY = (
    "sit_down",
    "turn_in_place",
    "wave_right",
    "wave_left",
    "bow",
    "jump_forward",
    "walk_slow",
    "walk_forward",
)


def _primary_action(intent: dict[str, Any]) -> Optional[str]:
    timeline = intent.get("timeline") or []
    actions = [t.get("action") for t in timeline if t.get("action")]
    # Walk/idle may only store wave as a gesture overlay — still prefer it.
    for t in timeline:
        g = str(t.get("gesture") or "").lower()
        if not g:
            continue
        if "left" in g and "wave" in g:
            actions.append("wave_left")
        elif "wave" in g:
            actions.append("wave_right")
    for pref in EDIT_PRIORITY:
        if pref in actions:
            return pref
    return actions[-1] if actions else None


def _action_from_prompt(prompt: str) -> Optional[str]:
    """Fallback when revise Intent has empty timeline."""
    side = parse_generate(prompt, task_id="ACTION_PROBE")
    return _primary_action(side)


def _source_class_from_hint(source_hint: str) -> str:
    h = (source_hint or "").lower()
    if "sit" in h:
        return "SIT"
    if "stand" in h or "idle" in h:
        return "STAND"
    if "run" in h:
        return "RUN"
    if "motionfix" in h or "native" in h:
        return "MOTIONFIX_NATIVE"
    return "WALK"


def _revise_speed_scale(intent: dict[str, Any]) -> Optional[float]:
    """From revision ops / prompt → geometric root-speed scale."""
    rev = intent.get("revision") or {}
    for op in rev.get("ops") or []:
        if op.get("type") != "speed":
            continue
        if op.get("scale") is not None:
            return float(op["scale"])
        to = (op.get("to") or "").lower()
        if to == "slower":
            return 0.45
        if to == "faster":
            return 1.35
    prompt = intent.get("raw_prompt") or ""
    # Fallback mirrors parse_intent._parse_speed_scale lightly
    pl = prompt.lower()
    if any(k in prompt for k in ("两倍", "2倍", "二倍")) or any(
        k in pl for k in ("twice", "2x", "double")
    ):
        return 2.0
    if any(k in pl for k in ("faster", "fast", "speed up", "quickly")) or "快" in prompt:
        return 1.35
    if any(k in pl for k in ("slower", "slowly", "slow down")) or "慢" in prompt:
        return 0.45
    return 1.35


def _compose_revise_mf_text(
    prompt: str,
    action: Optional[str],
    recipe: dict[str, Any],
    refine: Optional[dict[str, Any]] = None,
) -> str:
    """Map refine.mf_hints (+ light keyword fallback) → MotionsFix caption.

    Prefer structured hints over dumping raw ZH into the English recipe.
    """
    refine = refine or {}
    hints = [h for h in (refine.get("mf_hints") or []) if h]
    base = {
        "sit_down": "sit down",
        "turn_in_place": "turn in place",
        "wave_right": "wave with the right hand",
        "wave_left": "wave with the left hand",
        "walk_forward": "walk forward",
        "walk_slow": "walk slowly",
    }.get(action or "", "")

    # MotionsFix only understands short relative captions — never dump long hint lists.
    if action in ("wave_right", "wave_left", "wave"):
        return "wave to the side"
    if hints:
        short = hints[0]
        return f"{base} {short}.".strip() if base else f"{short}."

    p = prompt or ""
    bits: list[str] = []
    if _negated_turn(p) or any(k in p for k in ("转一圈", "转圈")):
        bits.append("without turning or spinning around")
    if str(action or "").startswith("sit") and any(
        k in p for k in ("抽搐", "抖动", "卡顿", "不自然", "回弹", "伸展")
    ):
        bits.append("smooth seated hold, no rebound or limb stretch")
    if any(k in p for k in ("无法坐下", "站立", "弯着腰", "弯腰")):
        bits.append("sit fully with pelvis lowering, then hold")
    if base and bits:
        return f"{base} {bits[0]}."
    # Action-aware fallback — never paste sit wording onto wave/walk.
    fallback = {
        "sit_down": "keep seated motion clear and natural",
        "walk_forward": "walk forward naturally",
        "walk_slow": "walk forward slowly",
        "turn_in_place": "turn in place clearly",
    }.get(action or "", "keep the motion clear and natural")
    return f"{base or 'a person'} {fallback}."


def intent_to_motion_edit_plan(
    intent: dict[str, Any],
    *,
    source_hint: str = "bridged_walk",
    module_plan: Optional[dict[str, Any]] = None,
    force_native_on_large_span: bool = True,
    previous_action: Optional[str] = None,
) -> dict[str, Any]:
    """Compile Intent → plan that motionfix_edit / v05_pipeline must follow."""
    action = _primary_action(intent)
    if not action:
        action = _action_from_prompt(intent.get("raw_prompt") or "")

    # Revise: lock previous clip action unless user clearly switches action class.
    if intent.get("task_mode") == "revise" and previous_action:
        switch = _wants_new_action(intent.get("raw_prompt") or "", previous_action)
        action = switch or previous_action

    recipe = dict(ACTION_MF.get(action or "", {}))
    src_class = _source_class_from_hint(source_hint)
    tgt_class = None
    for seg in intent.get("timeline") or []:
        if seg.get("action") == action:
            tgt_class = seg.get("humanml3d_class")
            break
    if not tgt_class and action and action in ACTION_MF:
        # coarse from taxonomy-ish
        coarse = {
            "sit_down": "SIT",
            "turn_in_place": "TURN",
            "wave_right": "REACH",
            "wave_left": "REACH",
            "walk_forward": "WALK",
            "walk_slow": "WALK",
        }
        tgt_class = coarse.get(action)

    span = recipe.get("span") or "unknown"
    if action == "sit_down" and src_class in ("WALK", "RUN", "STAND"):
        span = "large"
    if action == "sit_down" and src_class == "MOTIONFIX_NATIVE":
        span = "small"
    if action == "turn_in_place" and src_class == "WALK":
        span = "medium"
    if action == "turn_in_place" and src_class == "MOTIONFIX_NATIVE":
        span = "small"

    preferred_id = recipe.get("preferred_source_id")
    mld_text = recipe.get("mld_text") or "a person walks forward."
    edit_source = "bridged_npy"

    rev = intent.get("revision") or {}
    refine_early = (rev.get("refine") if isinstance(rev, dict) else None) or {}
    is_revise = intent.get("task_mode") == "revise"
    # Phase-local speed (「下降部分减慢」) must NOT become whole-clip speed_revise.
    phase_local = bool(
        refine_early.get("phase_local")
        or refine_early.get("phase_speed")
        or refine_early.get("drop_phases")
        or "phase_speed" in (refine_early.get("ops") or [])
        or "drop_phases" in (refine_early.get("ops") or [])
        or "trim_preamble" in (refine_early.get("ops") or [])
        or "trim_trailing" in (refine_early.get("ops") or [])
    )
    is_speed_revise = (
        is_revise
        and not phase_local
        and (
            "speed" in (rev.get("targets") or [])
            or "speed_edit" in (intent.get("modules_needed") or [])
        )
    )
    # Quality / semantic revise (not speed): keep editing the previous clip.
    is_motion_revise = is_revise and not is_speed_revise

    # MotionsFix-native preferred for *first* generate of sit/wave/turn.
    # Revise must NOT jump back to native id (that ignores user feedback & previous clip).
    prefer_native = (
        (not is_revise)
        and bool(preferred_id)
        and action
        in (
            "sit_down",
            "turn_in_place",
            "wave_right",
            "wave_left",
        )
    )

    if prefer_native and (force_native_on_large_span or span in ("large", "medium", "small")):
        source_policy = "use_motionfix_native"
        edit_source = "motionfix_id"
        span = "small"
        warn = (
            f"Edit uses MotionsFix-native source-id={preferred_id} "
            f"with mf_text={recipe.get('mf_text')!r}."
        )
    elif is_motion_revise:
        source_policy = "reuse_previous_edited"
        edit_source = "bridged_npy"
        preferred_id = None
        span = "small"
        warn = (
            "Revise reuses previous edited motion as source; "
            "mf_text comes from user feedback (not ACTION_MF recipe alone)."
        )
    elif span == "large":
        source_policy = "need_closer_source"
        warn = (
            f"Large semantic span: source≈{src_class} → action={action}. "
            "MotionsFix is a relative editor; switch to MotionsFix-native "
            f"id={preferred_id} (or regenerate nearer MLD: {mld_text!r})."
        )
    elif span == "medium":
        source_policy = "reuse_bridged_with_constraints"
        warn = (
            f"Medium span source≈{src_class} → {action}; "
            "use MotionsFix + freeze_xy."
        )
    else:
        source_policy = "reuse_bridged"
        warn = None

    mf_text = recipe.get("mf_text")
    if not mf_text and action:
        mf_text = intent.get("raw_prompt") or ""
    refine = (rev.get("refine") if isinstance(rev, dict) else None) or {}
    refine_ops = set(refine.get("ops") or [])
    if is_motion_revise:
        mf_text = _compose_revise_mf_text(
            intent.get("raw_prompt") or "", action, recipe, refine=refine
        )

    edit_mode = recipe.get("edit_mode") or "diffusion"
    speed_scale = recipe.get("speed_scale")
    if span == "none" and speed_scale is not None:
        edit_mode = "speed"

    # Recipe / timeline-driven wave fix (no MLD — Windows policy often blocks spacy/MLD).
    mem = intent.get("_edit_memory_overrides") or {}
    tl_problems = set()
    # Motion clip timeline (phases), not Intent Schema timeline segments
    for src in (
        intent.get("motion_timeline"),
        (intent.get("revision") or {}).get("motion_timeline"),
        intent.get("_motion_timeline"),
    ):
        if isinstance(src, dict):
            tl_problems.update(str(x) for x in (src.get("problems") or []))
            for ph in src.get("phases") or []:
                if ph.get("problem") and ph.get("id"):
                    tl_problems.add(str(ph["id"]))
    force_wave_walk_fix = bool(
        mem.get("prefer_kinematic_walk_wave")
        or mem.get("force_bridged_walk")
        or refine.get("prefer_kinematic_walk_wave")
        or refine.get("force_bridged_walk")
        or "wave_walk_fix" in refine_ops
        or bool(tl_problems & {"weak_forward_travel", "both_arms_waving", "walk_forward", "wave_right", "wave_left", "idle"})
    )
    # Any wave revise with phase_local / semantic recipes → geometric wave-walk fix
    if is_motion_revise and (action or "").startswith("wave") and (
        force_wave_walk_fix
        or refine.get("phase_local")
        or any(
            str(x).startswith("wave.")
            for x in (refine.get("matched_recipes") or mem.get("matched_recipes") or [])
        )
    ):
        force_wave_walk_fix = True
    force_kinematic_walk_wave = force_wave_walk_fix  # alias used below
    force_diffusion = (not force_wave_walk_fix) and (
        bool(mem.get("force_diffusion"))
        or ("diffusion" in refine_ops and "geometric_only" not in refine_ops)
    )
    if mem.get("preferred_source_id") and not force_wave_walk_fix:
        preferred_id = mem["preferred_source_id"]
    if mem.get("sdedit_strength") is not None and not force_wave_walk_fix:
        recipe["sdedit_strength"] = mem["sdedit_strength"]

    if is_motion_revise and force_wave_walk_fix:
        # Prefer MLD forward-walk rebuild; orchestrator falls back to previous clip if MLD fails.
        edit_mode = "kinematic"
        source_policy = "mld_walk_kinematic_wave"
        edit_source = "bridged_npy"
        preferred_id = None
        mld_text = "a person walks forward."
        mf_text = "wave to the side"
        span = "medium"
        recipe["keep_lower"] = False
        recipe["freeze_xy"] = False
        warn = (
            "Wave revise prefers MLD forward-walk → bridge → kinematic one-arm wave, "
            "then geometric wave_walk_post; falls back to previous clip if MLD fails. "
            f"(recipes={mem.get('matched_recipes') or refine.get('matched_recipes')}; "
            f"timeline_problems={sorted(tl_problems)})"
        )
    # Geometric-only / restore: do NOT run MotionsFix diffusion (it stacks damage),
    # unless edit-memory recipes explicitly request diffusion.
    elif is_motion_revise and not force_diffusion and (
        "geometric_only" in refine_ops
        or "restore_sit" in refine_ops
        or (
            refine.get("forbid")
            and not any(
                x in refine_ops for x in ("diffusion",)
            )
        )
    ):
        edit_mode = "geometric_only"
        source_policy = "reuse_previous_edited"
        edit_source = "bridged_npy"
        preferred_id = None
        if "restore_sit" in refine_ops:
            source_policy = "restore_last_good_or_native"
            edit_source = "restore"
            preferred_id = recipe.get("preferred_source_id") or "000236"
            warn = (
                "Sit lost/broken — restore from last good clip or MotionsFix-native, "
                "then geometric hold (skip diffusion stack)."
            )
        else:
            warn = (
                "Revise uses geometric_only postprocess (sit_hold); "
                "skip MotionsFix diffusion to avoid destroying the sit."
            )
    elif is_motion_revise and force_diffusion:
        edit_mode = "diffusion"
        source_policy = "reuse_previous_edited"
        edit_source = "bridged_npy"
        if mem.get("preferred_source_id"):
            preferred_id = mem["preferred_source_id"]
            source_policy = "use_motionfix_native"
            edit_source = "motionfix_id"
        warn = (
            "Edit-memory recipe requests MotionsFix diffusion "
            f"(recipes={mem.get('matched_recipes') or refine.get('matched_recipes')})."
        )

    if is_speed_revise:
        speed_scale = _revise_speed_scale(intent)
        edit_mode = "speed"
        edit_source = "bridged_npy"
        source_policy = "reuse_previous_edited"
        preferred_id = None
        span = "none"
        action = action or "speed_revise"
        mf_text = f"speed x{speed_scale:.2f}"
        warn = (
            f"Revise speed_edit: geometric root-speed ×{speed_scale:.2f} "
            f"on previous edited motion (Intent targets={rev.get('targets')})."
        )

    modules = [
        m
        for m in (intent.get("modules_needed") or [])
        if m
        in (
            "motionfix_semantic_edit",
            "gesture_wave",
            "action_sit",
            "action_turn",
            "speed_edit",
            "motion_edit",
            "sit_hold_postprocess",
            "timeline_postprocess",
            "wave_walk_postprocess",
            "quality_check",
        )
    ]

    base_action = action
    if action == "speed_revise" and previous_action and previous_action != "speed_revise":
        base_action = previous_action

    plan = {
        "schema_version": "motion_edit_plan_v1",
        "from_intent_task_id": intent.get("task_id"),
        "task_mode": intent.get("task_mode") or "generate",
        "raw_prompt": intent.get("raw_prompt"),
        "primary_action": action,
        "base_action": base_action,
        "source_humanml3d_class": src_class,
        "target_humanml3d_class": tgt_class,
        "span": span,
        "source_policy": source_policy,
        "edit_source": edit_source,
        "preferred_source_id": preferred_id,
        "mld_text": mld_text,
        "warning": warn,
        "mf_text": mf_text,
        "edit_mode": edit_mode,
        "constraints": {
            "keep_lower": "on" if recipe.get("keep_lower") else "off",
            "freeze_xy": bool(recipe.get("freeze_xy")),
            "sdedit_strength": recipe.get("sdedit_strength"),
            "init_from": recipe.get("init_from") or "source",
            "gd_motion": recipe.get("gd_motion"),
            "align_bridged": edit_source == "bridged_npy" and edit_mode != "speed",
            "speed_scale": speed_scale,
        },
        "modules": modules,
        "module_plan_id": (module_plan or {}).get("plan_id"),
        "revision": intent.get("revision"),
        "refine": refine or None,
        "decision": {
            "why": (
                "Agent compiled Intent first; MotionsFix receives relative mf_text "
                "and a source policy — not the raw user prompt alone."
            ),
            "intent_confidence": intent.get("confidence"),
            "intent_why": (intent.get("decision") or {}).get("why"),
            "previous_action": previous_action,
            "locked_previous_action": bool(
                is_revise and previous_action and action == previous_action
            ),
            "matched_recipes": list(
                (refine or {}).get("matched_recipes")
                or (intent.get("_edit_memory_overrides") or {}).get("matched_recipes")
                or []
            ),
        },
    }
    if "motionfix_semantic_edit" not in plan["modules"] and edit_mode == "diffusion":
        plan["modules"].insert(0, "motionfix_semantic_edit")
    # Sit hold when geometric refine / restore
    need_hold = (action or "").startswith("sit") and (
        edit_mode == "geometric_only"
        or bool(refine.get("phase_speed"))
        or bool(refine.get("drop_phases"))
        or "phase_speed" in (refine.get("ops") or [])
        or "drop_phases" in (refine.get("ops") or [])
        or any(
            f in (refine.get("forbid") or [])
            for f in (
                "post_sit_limb_rebound",
                "post_sit_stretch",
                "yaw_spin",
                "head_torso_misalign",
                "timeline_trim",
                "failed_sit",
                "pre_sit_locomotion",
                "post_sit_extra",
            )
        )
    )
    need_timeline = (not (action or "").startswith("sit")) and (
        bool(refine.get("phase_speed"))
        or bool(refine.get("drop_phases"))
        or "phase_speed" in (refine.get("ops") or [])
        or "drop_phases" in (refine.get("ops") or [])
        or edit_mode == "geometric_only"
    )
    if need_hold or need_timeline:
        plan["modules"] = [
            m
            for m in plan["modules"]
            if m != "motionfix_semantic_edit" or edit_mode not in ("geometric_only",)
        ]
        if edit_mode == "geometric_only" and "motionfix_semantic_edit" in plan["modules"]:
            plan["modules"] = [m for m in plan["modules"] if m != "motionfix_semantic_edit"]
    if need_hold and "sit_hold_postprocess" not in plan["modules"]:
        plan["modules"].append("sit_hold_postprocess")
    # Never run sit_hold on wave/walk/turn
    if not (action or "").startswith("sit"):
        plan["modules"] = [m for m in plan["modules"] if m != "sit_hold_postprocess"]
    if need_timeline and "timeline_postprocess" not in plan["modules"]:
        plan["modules"].append("timeline_postprocess")
    need_wave_walk = (action or "").startswith("wave") and (
        force_kinematic_walk_wave
        or "wave_walk_fix" in refine_ops
        or bool(refine.get("prefer_kinematic_walk_wave"))
    )
    if need_wave_walk and "wave_walk_postprocess" not in plan["modules"]:
        plan["modules"].append("wave_walk_postprocess")
    if "quality_check" not in plan["modules"]:
        plan["modules"].append("quality_check")
    plan["edit_mode"] = edit_mode
    if force_wave_walk_fix:
        plan["force_bridged_walk"] = True
        plan["source_policy"] = "mld_walk_kinematic_wave"
        plan["mld_text"] = mld_text or "a person walks forward."
        plan["mf_text"] = "wave to the side"
        plan["edit_mode"] = "kinematic"
    return plan


def build_full_agent_bundle(
    prompt: str,
    *,
    mode: str = "generate",
    task_id: str = "MF_EDIT",
    source_hint: str = "bridged_walk",
    previous_task_id: Optional[str] = None,
    previous_action: Optional[str] = None,
    previous_plan: Optional[dict[str, Any]] = None,
    refine_override: Optional[dict[str, Any]] = None,
    timeline: Optional[dict[str, Any]] = None,
    previous_caption: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if previous_plan and not previous_action:
        previous_action = previous_plan.get("primary_action")
    # After a speed revise, primary_action is speed_revise — keep semantic base.
    if previous_plan:
        previous_action = (
            previous_plan.get("base_action")
            or (
                previous_action
                if previous_action and previous_action != "speed_revise"
                else None
            )
            or previous_plan.get("primary_action")
        )
        if previous_action == "speed_revise":
            previous_action = "sit_down"  # safest demo default; orchestrator may override

    intent = parse_prompt(
        prompt,
        mode=mode,
        task_id=task_id,
        previous_task_id=previous_task_id or task_id,
        previous_action=previous_action,
    )
    if timeline:
        intent["motion_timeline"] = timeline
    # Re-compile refine with locked action (parse may have run before lock).
    if mode == "revise":
        try:
            from compile_refine import compile_refine

            refine = compile_refine(
                prompt,
                previous_action=previous_action,
                refine_override=refine_override,
                timeline=timeline,
                previous_caption=previous_caption,
            )
            # Ground to current clip timeline + strengthen via edit recipe memory
            try:
                from edit_memory import enrich_refine, plan_overrides_from_refine

                refine = enrich_refine(
                    refine,
                    action=str(previous_action or refine.get("keep_action") or ""),
                    timeline=timeline,
                    prompt=prompt,
                )
                intent["_edit_memory_overrides"] = plan_overrides_from_refine(refine)
            except Exception:  # noqa: BLE001
                pass
            # Carry forward previous forbids (转圈 then 回弹 should keep both).
            prev_forbid = []
            if previous_plan:
                prev_forbid = list(
                    (previous_plan.get("refine") or {}).get("forbid") or []
                )
            if prev_forbid:
                seen = set(refine.get("forbid") or [])
                merged = list(refine.get("forbid") or [])
                for f in prev_forbid:
                    if f not in seen and f != "failed_sit":
                        merged.append(f)
                        seen.add(f)
                refine["forbid"] = merged
                ops = list(refine.get("ops") or [])
                # Don't force geometric_only when recipe asks for diffusion
                if (
                    "diffusion" not in ops
                    and "geometric_only" not in ops
                    and "restore_sit" not in ops
                ):
                    ops.insert(0, "geometric_only")
                refine["ops"] = ops
            rev = dict(intent.get("revision") or {})
            rev["refine"] = refine
            intent["revision"] = rev
            needed = list(intent.get("modules_needed") or [])
            if (
                refine.get("phase_speed")
                or refine.get("drop_phases")
                or "phase_speed" in (refine.get("ops") or [])
                or "drop_phases" in (refine.get("ops") or [])
                or any(
                    f in (refine.get("forbid") or [])
                    for f in (
                        "post_sit_limb_rebound",
                        "post_sit_stretch",
                        "yaw_spin",
                        "head_torso_misalign",
                        "timeline_trim",
                        "failed_sit",
                        "pre_sit_locomotion",
                        "post_sit_extra",
                    )
                )
            ):
                extras = ["timeline_postprocess", "quality_check"]
                if str(previous_action or "").startswith("sit"):
                    extras.insert(0, "sit_hold_postprocess")
                if str(previous_action or "").startswith("wave"):
                    extras.append("wave_walk_postprocess")
                for m in extras:
                    if m not in needed:
                        needed.append(m)
            intent["modules_needed"] = needed
        except Exception:  # noqa: BLE001
            pass
    # Revise Intent has empty timeline. Prefer locking previous_action over
    # parse_generate(prompt) — keyword "转身" in feedback must not hijack sit→turn.
    if mode == "revise" and not (intent.get("timeline") or []):
        intent = dict(intent)
        switch = _wants_new_action(prompt, previous_action)
        lock_action = switch or previous_action
        if lock_action:
            coarse = {
                "sit_down": "SIT",
                "turn_in_place": "TURN",
                "wave_right": "REACH",
                "wave_left": "REACH",
                "walk_forward": "WALK",
                "walk_slow": "WALK",
                "speed_revise": "WALK",
            }
            intent["timeline"] = [
                {
                    "seg": 1,
                    "action": lock_action,
                    "humanml3d_class": coarse.get(lock_action),
                    "from_previous": True,
                }
            ]
            needed = list(intent.get("modules_needed") or [])
            for m in (
                "motionfix_semantic_edit",
                "quality_check",
            ):
                if m not in needed:
                    needed.append(m)
            if lock_action == "sit_down" and "action_sit" not in needed:
                needed.append("action_sit")
            # Strip turn module when refining sit / removing spin
            if lock_action != "turn_in_place" or _negated_turn(prompt):
                needed = [m for m in needed if m != "action_turn"]
            intent["modules_needed"] = needed
            if intent.get("revision"):
                rev = dict(intent["revision"])
                rerun = [m for m in (rev.get("modules_to_rerun") or needed) if m != "action_turn" or lock_action == "turn_in_place"]
                for m in needed:
                    if m not in rerun:
                        rerun.append(m)
                rev["modules_to_rerun"] = rerun
                intent["revision"] = rev
        else:
            probed = parse_generate(prompt, task_id=f"{task_id}_probe")
            intent["timeline"] = probed.get("timeline") or []
            needed = list(intent.get("modules_needed") or [])
            for m in probed.get("modules_needed") or []:
                if m not in needed and m in (
                    "motionfix_semantic_edit",
                    "gesture_wave",
                    "action_sit",
                    "action_turn",
                    "speed_edit",
                ):
                    needed.append(m)
            intent["modules_needed"] = needed
            if intent.get("revision"):
                rev = dict(intent["revision"])
                rerun = list(rev.get("modules_to_rerun") or [])
                for m in needed:
                    if m not in rerun:
                        rerun.append(m)
                rev["modules_to_rerun"] = rerun
                intent["revision"] = rev

    module_plan = route_modules(intent)
    motion_plan = intent_to_motion_edit_plan(
        intent,
        source_hint=source_hint,
        module_plan=module_plan,
        previous_action=previous_action,
    )
    return {
        "intent": intent,
        "module_plan": module_plan,
        "motion_edit_plan": motion_plan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Intent → MotionsFix MotionEditPlan (agent before coder)"
    )
    ap.add_argument("--prompt", required=True, help="User generate / revise prompt")
    ap.add_argument("--mode", default="generate", choices=["generate", "revise"])
    ap.add_argument("--task-id", default="MF_EDIT")
    ap.add_argument("--previous-task-id", default=None)
    ap.add_argument(
        "--source-hint",
        default="bridged_walk",
        help="What the current source looks like (bridged_walk / stand / sit / ...)",
    )
    ap.add_argument(
        "--previous-action",
        default=None,
        help="Lock revise to this action (e.g. sit_down) unless user switches",
    )
    ap.add_argument(
        "--previous-plan",
        type=Path,
        default=None,
        help="Prior 00_motion_edit_plan.json for revise context",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write full bundle JSON")
    ap.add_argument(
        "--out-plan",
        type=Path,
        default=None,
        help="Write motion_edit_plan only (for motionfix_edit --intent-json)",
    )
    args = ap.parse_args()

    prev_plan = None
    if args.previous_plan and args.previous_plan.is_file():
        prev_plan = json.loads(args.previous_plan.read_text(encoding="utf-8-sig"))
    bundle = build_full_agent_bundle(
        args.prompt,
        mode=args.mode,
        task_id=args.task_id,
        source_hint=args.source_hint,
        previous_task_id=args.previous_task_id,
        previous_action=args.previous_action,
        previous_plan=prev_plan,
    )
    plan = bundle["motion_edit_plan"]
    print(
        f"[plan_motion_edit] mode={args.mode} action={plan.get('primary_action')} "
        f"span={plan.get('span')} policy={plan.get('source_policy')} "
        f"edit_source={plan.get('edit_source')} "
        f"mf_id={plan.get('preferred_source_id')} "
        f"mf_text={plan.get('mf_text')!r}"
    )
    if plan.get("warning"):
        print(f"[plan_motion_edit] WARNING: {plan['warning']}")
    mp = bundle["module_plan"]
    print(
        f"[plan_motion_edit] modules={ [s['module'] for s in mp.get('steps', [])] } "
        f"frozen={mp.get('frozen_modules')}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[plan_motion_edit] wrote bundle {args.out}")
    if args.out_plan:
        args.out_plan.parent.mkdir(parents=True, exist_ok=True)
        args.out_plan.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[plan_motion_edit] wrote plan {args.out_plan}")
    if not args.out and not args.out_plan:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
