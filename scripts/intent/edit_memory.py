"""
Editable recipe memory for sit/wave/turn/walk revises.

Flow:
  prompt + timeline phases → lookup recipes → strengthen refine / plan
  quality_gate result → record_success / record_fail (updatable weights)

Recipes live in `.agents/skills/motion-patterns/edit_recipes.json` so they
can be curated like motion-patterns knowledge and updated after good runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / ".agents" / "skills" / "motion-patterns" / "edit_recipes.json"


def _load() -> Dict[str, Any]:
    if not RECIPES_PATH.is_file():
        return {"schema_version": "edit_recipes_v1", "recipes": []}
    return json.loads(RECIPES_PATH.read_text(encoding="utf-8-sig"))


def _save(data: Dict[str, Any]) -> None:
    RECIPES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECIPES_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _norm_action(action: Optional[str]) -> str:
    act = (action or "").strip()
    if act.startswith("sit"):
        return "sit_down"
    if act in ("wave",) or act.endswith("wave_right"):
        return "wave_right"
    if act.endswith("wave_left"):
        return "wave_left"
    if "turn" in act:
        return "turn_in_place"
    if act in ("walk_slow",) or "slow" in act and "walk" in act:
        return "walk_slow"
    if act.startswith("walk") or act == "walk_forward":
        return "walk_forward"
    return act


def _intent_tags(refine: Dict[str, Any], prompt: str = "") -> List[str]:
    tags: List[str] = []
    ops = set(refine.get("ops") or [])
    raw = (refine.get("raw_prompt") or prompt or "").lower()
    zh = refine.get("raw_prompt") or prompt or ""
    if "trim_preamble" in ops or "trim_trailing" in ops or "drop_phases" in ops:
        tags.append("delete")
        tags.append("trim")
    if "phase_speed" in ops or refine.get("phase_speed"):
        speeds = [float(v) for v in dict(refine.get("phase_speed") or {}).values()]
        if speeds and sum(speeds) / len(speeds) < 0.95:
            tags.append("slow")
        elif speeds and sum(speeds) / len(speeds) > 1.05:
            tags.append("fast")
        else:
            tags.append("slow")
    if any(k in zh for k in ("删", "去", "裁", "不要", "移除")) or "delete" in raw:
        tags.append("delete")
    if any(k in zh for k in ("慢", "减慢", "放慢")) or "slow" in raw:
        tags.append("slow")
    if any(k in zh for k in ("快", "加速")) or "fast" in raw or "quicker" in raw:
        tags.append("fast")
    if any(k in zh for k in ("只挥", "单手", "双臂", "往前走", "走起来", "右手", "左手")):
        tags.append("semantic")
        tags.append("fix")
    if not tags:
        tags.append("semantic")
    # dedupe
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _phase_ids(refine: Dict[str, Any], timeline: Optional[Dict[str, Any]]) -> List[str]:
    phases = [str(p) for p in (refine.get("phases") or []) if p and p != "full"]
    if refine.get("drop_phases"):
        phases.extend(str(p) for p in refine["drop_phases"])
    if refine.get("phase_speed"):
        phases.extend(str(k) for k in refine["phase_speed"].keys())
    if not phases and timeline:
        # Prefer problem phases, else all
        probs = [str(p.get("id")) for p in (timeline.get("phases") or []) if p.get("problem")]
        phases = probs or [str(p.get("id")) for p in (timeline.get("phases") or []) if p.get("id")]
    out, seen = [], set()
    for p in phases:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _score_recipe(
    recipe: Dict[str, Any],
    *,
    action: str,
    intents: Sequence[str],
    phases: Sequence[str],
    prompt: str,
) -> float:
    if _norm_action(recipe.get("action")) != _norm_action(action):
        # wave_left recipes only for wave_left; wave_right accepts wave
        ra = _norm_action(recipe.get("action"))
        aa = _norm_action(action)
        if not (ra.startswith("wave") and aa.startswith("wave") and ra == aa):
            return -1.0
    score = float(recipe.get("weight") or 1.0)
    score += 0.15 * float(recipe.get("success_count") or 0)
    score -= 0.25 * float(recipe.get("fail_count") or 0)
    r_intents = set(recipe.get("intents") or [])
    intent_hit = bool(r_intents & set(intents))
    if intent_hit:
        score += 1.5
    else:
        # No shared intent → almost never select (keeps slow≠trim)
        return -1.0
    r_phases = set(recipe.get("phases") or [])
    overlap = r_phases & set(phases)
    raw = prompt or ""
    hint_hit = False
    for hint in recipe.get("prompt_hints") or []:
        if hint and hint.lower() in raw.lower():
            score += 2.0
            hint_hit = True
            break
    if overlap:
        score += 1.2 * len(overlap)
    elif r_phases and not hint_hit:
        # Recipe targets other phases and prompt doesn't mention it → skip
        return -1.0
    return score


def lookup_recipes(
    *,
    action: str,
    refine: Dict[str, Any],
    timeline: Optional[Dict[str, Any]] = None,
    prompt: str = "",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    data = _load()
    intents = _intent_tags(refine, prompt)
    phases = _phase_ids(refine, timeline)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in data.get("recipes") or []:
        s = _score_recipe(
            r, action=action, intents=intents, phases=phases, prompt=prompt or refine.get("raw_prompt") or ""
        )
        if s >= 0:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for s, r in scored[: max(1, top_k)] if s >= 0]


def ground_phases_to_timeline(
    refine: Dict[str, Any],
    timeline: Optional[Dict[str, Any]],
    prompt: str = "",
) -> Dict[str, Any]:
    """Map refine phase ids / free text onto actual timeline segments."""
    out = dict(refine)
    if not timeline or not timeline.get("phases"):
        return out
    tl_phases = list(timeline["phases"])
    tl_ids = {str(p.get("id")) for p in tl_phases if p.get("id")}
    raw = prompt or str(out.get("raw_prompt") or "")

    # Match prompt against phase labels from THIS clip
    matched: List[str] = []
    for p in tl_phases:
        pid = str(p.get("id") or "")
        label = str(p.get("label") or "")
        if not pid:
            continue
        if pid in raw or (label and label in raw):
            matched.append(pid)
        # loose: first char tokens
        for token in re.split(r"[\s/·（）()]+", label):
            if token and len(token) >= 2 and token in raw:
                matched.append(pid)
                break

    phases = [p for p in (out.get("phases") or []) if p == "full" or p in tl_ids]
    if matched:
        for m in matched:
            if m not in phases:
                phases.append(m)
    if not phases or phases == ["full"]:
        # If user talked about problems, prefer problem segments
        probs = [str(p.get("id")) for p in tl_phases if p.get("problem") and p.get("id")]
        if probs:
            phases = probs
    out["phases"] = phases or out.get("phases") or ["full"]

    # Remap phase_speed / drop_phases keys to existing timeline ids only
    if out.get("phase_speed"):
        out["phase_speed"] = {
            str(k): float(v)
            for k, v in dict(out["phase_speed"]).items()
            if str(k) in tl_ids
        }
    if out.get("drop_phases"):
        out["drop_phases"] = [p for p in out["drop_phases"] if p in tl_ids]
    return out


def apply_recipes_to_refine(
    refine: Dict[str, Any],
    recipes: Sequence[Dict[str, Any]],
    *,
    prompt: str = "",
) -> Dict[str, Any]:
    """Merge top recipes into refine (stronger speeds, drops, hints, ops)."""
    from refine_schema import normalize_refine

    out = dict(refine)
    if not recipes:
        return normalize_refine(out, previous_action=out.get("keep_action"), source=out.get("source"))

    ops = list(out.get("ops") or [])
    forbid = list(out.get("forbid") or [])
    hints = list(out.get("mf_hints") or [])
    phase_speed = dict(out.get("phase_speed") or {})
    drop_phases = list(out.get("drop_phases") or [])
    prefer_diffusion = False
    prefer_kinematic_walk_wave = False
    force_bridged_walk = False
    preferred_source_id = None
    sdedit_strength = None
    matched_ids: List[str] = []

    for r in recipes:
        matched_ids.append(str(r.get("id")))
        for o in r.get("ops") or []:
            if o not in ops and o in (
                "geometric_only",
                "restore_sit",
                "diffusion",
                "trim_preamble",
                "trim_trailing",
                "phase_speed",
                "drop_phases",
            ):
                ops.append(o)
        for f in r.get("forbid") or []:
            if f not in forbid:
                forbid.append(f)
        for h in r.get("mf_hints") or []:
            if h and h not in hints:
                hints.append(h)
        strength = dict(r.get("strength") or {})
        # Amplify existing phase_speed with recipe strength
        for pid, scale in list(phase_speed.items()):
            if float(scale) < 1.0 and "slow" in strength:
                phase_speed[pid] = min(float(scale), float(strength["slow"]))
            elif float(scale) > 1.0 and "fast" in strength:
                phase_speed[pid] = max(float(scale), float(strength["fast"]))
        for pid, scale in dict(r.get("phase_speed") or {}).items():
            cur = phase_speed.get(pid)
            if cur is None:
                phase_speed[pid] = float(scale)
            elif float(scale) < 1.0:
                phase_speed[pid] = min(float(cur), float(scale))
            else:
                phase_speed[pid] = max(float(cur), float(scale))
        for pid in r.get("drop_phases") or []:
            if pid not in drop_phases:
                drop_phases.append(pid)
        if r.get("prefer_kinematic_walk_wave") or r.get("force_bridged_walk"):
            prefer_kinematic_walk_wave = True
            force_bridged_walk = True
            prefer_diffusion = False
        elif r.get("prefer_diffusion") and not prefer_kinematic_walk_wave:
            prefer_diffusion = True
        if r.get("preferred_source_id") and not prefer_kinematic_walk_wave:
            preferred_source_id = r["preferred_source_id"]
        if r.get("sdedit_strength") is not None and not prefer_kinematic_walk_wave:
            sdedit_strength = float(r["sdedit_strength"])

    if prefer_kinematic_walk_wave:
        ops = [o for o in ops if o not in ("diffusion",)]
        if "geometric_only" not in ops:
            ops.insert(0, "geometric_only")
        # Short MotionsFix-native caption if diffusion is ever used as fallback
        hints = ["wave to the side"]
    elif prefer_diffusion:
        ops = [o for o in ops if o not in ("geometric_only",)]
        if "diffusion" not in ops:
            ops.insert(0, "diffusion")
    elif ops and "geometric_only" not in ops and "diffusion" not in ops:
        ops.insert(0, "geometric_only")

    out["ops"] = ops
    out["forbid"] = forbid
    # Keep mf_hints short — MotionsFix collapses long captions anyway
    out["mf_hints"] = hints[:2]
    if phase_speed:
        out["phase_speed"] = phase_speed
        out["phase_local"] = True
        if "phase_speed" not in out["ops"]:
            out["ops"].append("phase_speed")
    if drop_phases:
        out["drop_phases"] = drop_phases
        out["phase_local"] = True
        if "drop_phases" not in out["ops"]:
            out["ops"].append("drop_phases")
    out["matched_recipes"] = matched_ids
    if preferred_source_id:
        out["preferred_source_id"] = preferred_source_id
    if sdedit_strength is not None:
        out["sdedit_strength"] = sdedit_strength

    normalized = normalize_refine(
        out,
        previous_action=out.get("keep_action"),
        source=out.get("source") or "edit_memory",
        raw_prompt=out.get("raw_prompt") or prompt,
    )
    # normalize strips extras — reattach learnable / plan hints
    if matched_ids:
        normalized["matched_recipes"] = matched_ids
    if preferred_source_id:
        normalized["preferred_source_id"] = preferred_source_id
    if sdedit_strength is not None:
        normalized["sdedit_strength"] = sdedit_strength
    if prefer_diffusion:
        normalized["prefer_diffusion"] = True
    if prefer_kinematic_walk_wave:
        normalized["prefer_kinematic_walk_wave"] = True
        normalized["force_bridged_walk"] = True
        normalized["prefer_diffusion"] = False
    return normalized


def enrich_refine(
    refine: Dict[str, Any],
    *,
    action: str,
    timeline: Optional[Dict[str, Any]] = None,
    prompt: str = "",
) -> Dict[str, Any]:
    """Ground to timeline → lookup recipes → strengthen refine."""
    grounded = ground_phases_to_timeline(refine, timeline, prompt=prompt)
    recipes = lookup_recipes(
        action=action or str(grounded.get("keep_action") or ""),
        refine=grounded,
        timeline=timeline,
        prompt=prompt,
        top_k=3,
    )
    enriched = apply_recipes_to_refine(grounded, recipes, prompt=prompt)
    enriched["keep_action"] = action or enriched.get("keep_action")
    return enriched


def record_outcome(
    recipe_ids: Sequence[str],
    *,
    success: bool,
    prompt: Optional[str] = None,
) -> None:
    """Update success/fail counts and weight (simple online learning)."""
    if not recipe_ids:
        return
    data = _load()
    wanted = set(recipe_ids)
    changed = False
    for r in data.get("recipes") or []:
        if r.get("id") not in wanted:
            continue
        changed = True
        if success:
            r["success_count"] = int(r.get("success_count") or 0) + 1
            r["weight"] = round(min(3.0, float(r.get("weight") or 1.0) + 0.08), 4)
            if prompt:
                hints = list(r.get("prompt_hints") or [])
                snip = str(prompt).strip()[:40]
                if snip and snip not in hints:
                    hints.append(snip)
                    r["prompt_hints"] = hints[-12:]
        else:
            r["fail_count"] = int(r.get("fail_count") or 0) + 1
            r["weight"] = round(max(0.2, float(r.get("weight") or 1.0) - 0.05), 4)
    if changed:
        _save(data)


def plan_overrides_from_refine(refine: Dict[str, Any]) -> Dict[str, Any]:
    """Extra fields plan_motion_edit / orchestrator can consume."""
    out: Dict[str, Any] = {}
    if refine.get("preferred_source_id"):
        out["preferred_source_id"] = refine["preferred_source_id"]
    if refine.get("sdedit_strength") is not None:
        out["sdedit_strength"] = float(refine["sdedit_strength"])
    if refine.get("matched_recipes"):
        out["matched_recipes"] = list(refine["matched_recipes"])
    if refine.get("prefer_kinematic_walk_wave") or refine.get("force_bridged_walk"):
        out["prefer_kinematic_walk_wave"] = True
        out["force_bridged_walk"] = True
        out["force_diffusion"] = False
        return out
    if "diffusion" in (refine.get("ops") or []) and refine.get("prefer_diffusion", True):
        out["force_diffusion"] = "diffusion" in (refine.get("ops") or []) and "geometric_only" not in (
            refine.get("ops") or []
        )
    return out
