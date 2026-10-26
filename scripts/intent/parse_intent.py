"""
Heuristic intent parser (v1 demo).

Compiles free-text prompt (+ refs) into Intent Schema v1.
LLM-backed parsing can replace `parse_prompt` later; keep the same output shape.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".agents" / "skills" / "intent-routing"


def load_taxonomy() -> dict:
    return json.loads((SKILL / "action_taxonomy.json").read_text(encoding="utf-8"))


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(k.lower() in text.lower() for k in keywords)


def _negated_turn(prompt: str) -> bool:
    """True when user wants to REMOVE spin/turn, not add a turn action."""
    if not _has_any(prompt, ["转", "turn", "spin", "圈", "around", "旋转"]):
        return False
    return _has_any(
        prompt,
        [
            "去掉",
            "去掉转",
            "别转",
            "不要转",
            "别再转",
            "不要再转",
            "消除",
            "取消",
            "删掉",
            "没有转",
            "无转",
            "without turning",
            "no turn",
            "don't turn",
            "dont turn",
            "no spin",
            "without spinning",
            "remove the turn",
            "remove spin",
        ],
    )


def _wants_new_action(prompt: str, previous_action: str | None) -> str | None:
    """If revise clearly asks to switch action class, return that action id."""
    if not previous_action:
        return None
    # Negated mentions / quality complaints are refinements of the current clip.
    if _negated_turn(prompt):
        return None
    if _has_any(
        prompt,
        [
            "抽搐",
            "抖动",
            "卡顿",
            "不自然",
            "更自然",
            "自然一点",
            "效果",
            "修改完",
            "不对",
            "有问题",
            "去掉",
            "删掉",
            "消除",
            "前半段",
            "后半段",
            "jerk",
            "twitch",
            "smooth",
        ],
    ):
        return None
    sit = _has_any(prompt, ["坐下", "坐着", "就坐", "sit", "sitting"])
    wave = _has_any(prompt, ["招手", "挥手", "wave"])
    # Explicit switch-to-turn only (mentioning 转身 while fixing sit is NOT a switch).
    turn_switch = _has_any(
        prompt,
        [
            "改成转身",
            "做成转身",
            "做一个转身",
            "改成转",
            "变成转身",
            "改为转身",
            "换成转身",
            "make it turn",
            "change to turn",
            "turn instead",
        ],
    )
    prev = previous_action or ""
    if sit and not prev.startswith("sit") and _has_any(
        prompt, ["改成坐", "做成坐", "变成坐", "改为坐", "坐下"]
    ):
        # "无法坐下" is a sit quality fix when already sitting — handled by early return
        # only when previous is not sit; "改成坐下" switches.
        if _has_any(prompt, ["改成", "做成", "变成", "改为", "换成"]) or not prev.startswith(
            "sit"
        ):
            if prev.startswith("sit"):
                return None
            return "sit_down"
    if wave and "wave" not in prev and _has_any(
        prompt, ["改成挥", "改成招", "做成挥", "wave instead"]
    ):
        return "wave_right"
    if turn_switch and "turn" not in prev:
        return "turn_in_place"
    return None


def _extract_refs_from_prompt(prompt: str) -> list[dict]:
    refs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(rid: str, role: str) -> None:
        key = (rid, role)
        if key not in seen:
            seen.add(key)
            refs.append({"id": rid, "role": role})

    for m in re.finditer(r"[Nn]2|圖c?N2|参考图\s*N2|參考圖\s*N2", prompt):
        add("N2", "pose")
        add("N2", "target_position")
        if _has_any(prompt, ["背景", "不要變化", "不要变化", "別動", "别动"]):
            add("N2", "background")
    if _has_any(prompt, ["參考front", "参考front", "形象參考front", "形象参考front", "参考 front"]):
        add("front", "appearance")
    if re.search(r"\bfront\b", prompt, re.I) and _has_any(prompt, ["形象", "全身", "參考", "参考"]):
        add("front", "appearance")
    return refs


def _merge_refs(ui_refs: list[dict] | None, prompt: str) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for src in (ui_refs or []) + _extract_refs_from_prompt(prompt):
        key = (src["id"], src["role"])
        if key not in seen:
            seen.add(key)
            merged.append({"id": src["id"], "role": src["role"], "note": src.get("note", "")})
    return merged


def _detect_revision_targets(prompt: str, taxonomy: dict) -> list[str]:
    hits = []
    for target, kws in taxonomy["revision_keywords"].items():
        if _has_any(prompt, kws):
            hits.append(target)
    # Extra EN/ZH speed phrasing not always in taxonomy
    if "speed" not in hits and _has_any(
        prompt,
        ["两倍", "2倍", "一倍半", "一半", "twice", "2x", "double", "half", "加速", "减速"],
    ):
        hits.insert(0, "speed")
    # priority if empty but revise mode: infer motion
    return hits or ["motion"]


def _parse_speed_scale(prompt: str) -> float | None:
    """Map revision text → root-speed multiplier (None if unspecified)."""
    import re

    t = (prompt or "").lower()
    if _has_any(prompt, ["两倍", "2倍", "二倍"]) or any(
        k in t for k in ("twice", "2x", "double", "two times")
    ):
        return 2.0
    if _has_any(prompt, ["三倍", "3倍"]) or "3x" in t or "triple" in t:
        return 3.0
    if _has_any(prompt, ["一倍半", "1.5倍", "一点五倍"]) or "1.5x" in t:
        return 1.5
    if _has_any(prompt, ["一半", "50%"]) or "half" in t:
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×倍]", prompt)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:x|×)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    return None


def parse_revise(
    prompt: str,
    previous_task_id: str | None = None,
    previous_action: str | None = None,
) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    targets = _detect_revision_targets(prompt, taxonomy)
    modules_to_rerun: list[str] = []
    modules_frozen = ["bg_lock", "camera_lock", "appearance_ref"]
    ops: list[dict] = []

    # Semantic motion words win over bare "姿势/pose" (avoid keypose-only misroute).
    semantic_motion = _has_any(
        prompt,
        ["坐下", "坐着", "招手", "挥手", "转身", "sit", "wave", "turn", "kneel", "跪"],
    )
    if semantic_motion and "key_pose" in targets and "motion" in targets:
        targets = [t for t in targets if t != "key_pose"]

    drop_turn = _negated_turn(prompt)

    if "speed" in targets:
        modules_to_rerun += ["speed_edit", "quality_check"]
        scale = _parse_speed_scale(prompt)
        if scale is not None and scale < 1.0:
            to = "slower"
        elif scale is not None and scale > 1.0:
            to = "faster"
        elif _has_any(prompt, ["慢", "slower", "slow down", "slowly"]):
            to = "slower"
            scale = scale or 0.45
        elif _has_any(prompt, ["快", "faster", "fast", "speed up", "quickly", "加速"]):
            to = "faster"
            scale = scale or 1.35
        else:
            to = "adjusted"
            scale = scale or 1.35  # default nudge faster when only "速度" mentioned
        ops.append(
            {
                "type": "speed",
                "seg": 2,
                "from": None,
                "to": to,
                "scale": scale,
                "ref": None,
                "scope": None,
                "note": prompt,
            }
        )
    if "key_pose" in targets:
        modules_to_rerun += ["keypose_edit", "ref_pose_align", "quality_check"]
        ref = "N2" if "N2" in prompt or "n2" in prompt.lower() else None
        ops.append({"type": "key_pose", "seg": 3, "from": None, "to": None, "ref": ref, "scope": "hands", "note": prompt})
    if "texture" in targets:
        modules_to_rerun += ["texture_edit", "quality_check"]
        modules_frozen += ["locomotion_walk", "hold_keypose", "gesture_wave"]
        ops.append({"type": "texture", "seg": None, "from": None, "to": None, "ref": None, "scope": "character", "note": prompt})
    if "motion" in targets:
        modules_to_rerun += ["motion_edit", "quality_check", "motionfix_semantic_edit"]
        if _has_any(prompt, ["招手", "wave", "左手", "右手", "挥手"]):
            modules_to_rerun += ["gesture_wave", "motionfix_semantic_edit"]
        if _has_any(
            prompt,
            ["坐下", "坐着", "就坐", "sit", "sitting", "跪", "kneel"],
        ):
            modules_to_rerun += ["action_sit", "motionfix_semantic_edit"]
        # Only route to turn module when user asks for a turn — not when removing one.
        if (
            not drop_turn
            and _has_any(
                prompt,
                ["转身", "转体", "转一圈", "turn around", "spin around", "180"],
            )
            and not _has_any(prompt, ["抽搐", "自然", "更顺", "smooth", "jerk"])
        ):
            modules_to_rerun += ["action_turn", "motionfix_semantic_edit"]
        if _has_any(prompt, ["步", "往前", "走", "walk"]):
            modules_to_rerun.append("locomotion_walk")
        if _has_any(prompt, ["背景别动", "背景別動", "背景不要"]):
            if "bg_lock" not in modules_frozen:
                modules_frozen.append("bg_lock")
        note_extra = "drop_turn" if drop_turn else None
        ops.append(
            {
                "type": "motion",
                "seg": None,
                "from": None,
                "to": None,
                "ref": None,
                "scope": None,
                "note": prompt,
                "constraints": {"no_spin": True} if drop_turn else {},
                "tag": note_extra,
            }
        )

    # dedupe keep order
    def uniq(xs: list[str]) -> list[str]:
        out, s = [], set()
        for x in xs:
            if x not in s:
                s.add(x)
                out.append(x)
        return out

    modules_to_rerun = uniq(modules_to_rerun)
    modules_frozen = uniq([m for m in modules_frozen if m not in modules_to_rerun or m in ("bg_lock", "camera_lock", "appearance_ref")])

    # freeze locomotion on pure texture
    if targets == ["texture"]:
        modules_frozen = uniq(modules_frozen + ["locomotion_walk", "timeline_compose"])

    reasons = {m: f"revision target includes related op for {targets}" for m in modules_to_rerun}

    # Structured refine (heuristic; DeepSeek can replace compile_refine later).
    try:
        from compile_refine import compile_refine

        refine = compile_refine(prompt, previous_action=previous_action)
    except Exception:  # noqa: BLE001
        refine = {
            "keep_action": previous_action or "sit_down",
            "forbid": [],
            "phases": [],
            "edit_span": "full",
            "mf_hints": [],
            "source": "fallback",
        }
    if refine.get("forbid") and "motionfix_semantic_edit" not in modules_to_rerun:
        modules_to_rerun = uniq(modules_to_rerun + ["motionfix_semantic_edit", "quality_check"])
    if any(
        f in (refine.get("forbid") or [])
        for f in ("post_sit_limb_rebound", "post_sit_stretch", "yaw_spin")
    ):
        if "sit_hold_postprocess" not in modules_to_rerun:
            modules_to_rerun.append("sit_hold_postprocess")

    return {
        "schema_version": "1.0",
        "task_id": previous_task_id or "REV",
        "task_mode": "revise",
        "raw_prompt": prompt,
        "confidence": 0.75,
        "needs_clarify": [],
        "refs": _merge_refs(None, prompt),
        "scene": {},
        "character": {},
        "timeline": [],
        "modules_needed": modules_to_rerun,
        "revision": {
            "refine": refine,
            "targets": targets,
            "ops": ops,
            "modules_to_rerun": modules_to_rerun,
            "modules_frozen": modules_frozen,
        },
        "decision": {
            "why": f"Revision prompt mapped to targets={targets}; rerun subset only.",
            "module_reasons": reasons,
            "rejected_modules": [
                {"id": m, "reason": "frozen during revision"} for m in modules_frozen if m not in modules_to_rerun
            ],
        },
    }


def parse_generate(prompt: str, refs: list[dict] | None = None, task_id: str | None = None) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    actions = taxonomy["actions"]
    merged_refs = _merge_refs(refs, prompt)
    timeline: list[dict] = []
    modules: list[str] = []
    clarify: list[str] = []
    module_reasons: dict[str, str] = {}

    def add_module(mid: str, reason: str) -> None:
        if mid not in modules:
            modules.append(mid)
            module_reasons[mid] = reason

    # Scene / camera / appearance
    lock_bg = "N2" if any(r["id"] == "N2" and r["role"] == "background" for r in merged_refs) or _has_any(
        prompt, ["背景不要", "背景别动", "背景別動", "保持圖", "保持图"]
    ) else None
    if lock_bg or _has_any(prompt, ["背景"]):
        if lock_bg is None and any(r["id"] == "N2" for r in merged_refs):
            lock_bg = "N2"
        if lock_bg:
            add_module("bg_lock", f"Lock background to {lock_bg}")

    cam_move = False if _has_any(prompt, ["镜头不要移动", "鏡頭不要移動", "镜头不动", "鏡頭不動", "不要移动"]) else None
    cam_cut = False if _has_any(prompt, ["不要切换", "不要切換", "不要切换镜头"]) else None
    if cam_move is False or cam_cut is False or _has_any(prompt, ["镜头", "鏡頭"]):
        add_module("camera_lock", "User requested fixed camera / no cuts")
        if cam_move is None:
            cam_move = False
        if cam_cut is None:
            cam_cut = False

    appearance = next((r["id"] for r in merged_refs if r["role"] == "appearance"), None)
    if appearance or _has_any(prompt, ["形象參考", "形象参考", "front"]):
        appearance = appearance or "front"
        add_module("appearance_ref", f"Appearance bound to {appearance}")

    # Conflict: fast run + professional stable
    if _has_any(prompt, ["跑"]) and _has_any(prompt, ["专业", "專業", "稳定", "穩定"]) and _has_any(prompt, ["快", "快速"]):
        clarify.append("速度冲突：快速奔跑 vs 专业稳定——请确认保留跑还是改为稳健快步走")

    # Build timeline heuristically (ordered segments for complex prompts)
    seg = 1
    # idle stand
    if _has_any(prompt, ["站在", "站立", "先站", "微笑看向"]) or (
        _has_any(prompt, ["站"]) and not _has_any(prompt, ["停在"])
    ):
        if _has_any(prompt, ["站在", "先站", "微笑", "中央偏左", "镜头中央"]):
            timeline.append(
                {
                    "seg": seg,
                    "action": "idle_stand",
                    "humanml3d_class": "STAND",
                    "speed": None,
                    "steps": None,
                    "duration_hint": "short",
                    "pos": "center_left" if "偏左" in prompt else None,
                    "target_pose_ref": None,
                    "gesture": None,
                    "hands": None,
                    "speech": None,
                    "notes": ["initial idle facing camera"],
                }
            )
            add_module("idle_stand", "Opening stand/idle segment")
            seg += 1

    # locomotion: prefer explicit 跑 over 走; "步伐" alone is not walk if running
    run_hit = _has_any(
        prompt, ["跑", "奔跑", "快跑", "run", "runs", "running", "sprint"]
    )
    walk_hit = _has_any(
        prompt,
        [
            "走",
            "往前走",
            "向前走",
            "走动",
            "慢走",
            "兩步",
            "两步",
            "walk",
            "walks",
            "walking",
            "step forward",
        ],
    ) or (_has_any(prompt, ["往前", "步伐"]) and not run_hit)
    if run_hit and not _has_any(prompt, ["不要跑", "don't run", "do not run"]):
        timeline.append(
            {
                "seg": seg,
                "action": "run_forward",
                "humanml3d_class": "RUN",
                "speed": "fast",
                "steps": None,
                "duration_hint": None,
                "pos": "center" if "中央" in prompt else None,
                "target_pose_ref": None,
                "gesture": None,
                "hands": None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("locomotion_run", "Run segment detected")
        seg += 1
    elif walk_hit and not _has_any(prompt, ["不要走动", "不要走路", "只说"]):
        steps = 2 if _has_any(prompt, ["兩步", "两步"]) else None
        speed = (
            "slow"
            if _has_any(prompt, ["缓慢", "慢慢", "慢走", "慢", "slowly", "slow walk", "slower"])
            else "normal"
        )
        gesture = None
        if _has_any(prompt, ["招手", "挥手", "wave", "waving"]):
            gesture = "occasional_left_wave" if _has_any(prompt, ["左手", "left"]) else "occasional_right_wave"
            add_module("gesture_wave", "Wave overlay during walk/idle")
            add_module("motionfix_semantic_edit", "MotionsFix relative edit for wave")
        walk_action = "walk_slow" if speed == "slow" else "walk_forward"
        timeline.append(
            {
                "seg": seg,
                "action": walk_action,
                "humanml3d_class": "WALK",
                "speed": speed,
                "steps": steps,
                "duration_hint": None,
                "pos": "center" if _has_any(prompt, ["中央", "正中央"]) else None,
                "target_pose_ref": "N2" if any(r["id"] == "N2" for r in merged_refs) else None,
                "gesture": gesture,
                "hands": None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("locomotion_walk", "Walk segment detected")
        add_module("ref_pose_align", "Align end position/pose to reference")
        if speed in ("slow", "very_slow"):
            add_module("speed_edit", "Slow walk speed constraint")
        seg += 1

    # turn / sit / wave / bow (EN + ZH; MotionsFix edit path uses these actions)
    if _has_any(
        prompt,
        [
            "转身",
            "转体",
            "转过来",
            "turn around",
            "turns around",
            "turned around",
            "turn back",
            "turns back",
            "about face",
            "180",
            "spin",
        ],
    ):
        timeline.append(
            {
                "seg": seg,
                "action": "turn_in_place",
                "humanml3d_class": "TURN",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": None,
                "gesture": None,
                "hands": None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("action_turn", "Turn segment")
        add_module("motionfix_semantic_edit", "MotionsFix relative edit for turn")
        seg += 1

    if _has_any(
        prompt,
        [
            "坐下",
            "坐下来",
            "就坐",
            "sit down",
            "sits down",
            "sat down",
            "sitting down",
            "sitting",
            "sit ",
            " sits",
        ],
    ):
        timeline.append(
            {
                "seg": seg,
                "action": "sit_down",
                "humanml3d_class": "SIT",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": None,
                "gesture": None,
                "hands": None,
                "speech": None,
                "notes": ["large semantic span if source is walk"],
            }
        )
        add_module("action_sit", "Sit segment")
        add_module("motionfix_semantic_edit", "MotionsFix relative edit for sit")
        seg += 1

    if _has_any(prompt, ["招手", "挥手", "wave", "waving"]) and not any(
        t.get("gesture") for t in timeline
    ):
        action = (
            "wave_left"
            if _has_any(prompt, ["左手", "left hand", "left wave"])
            else "wave_right"
        )
        timeline.append(
            {
                "seg": seg,
                "action": action,
                "humanml3d_class": "REACH",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": None,
                "gesture": action,
                "hands": None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("gesture_wave", "Standalone wave gesture")
        add_module("motionfix_semantic_edit", "MotionsFix relative edit for wave")
        seg += 1

    if _has_any(prompt, ["鞠躬", "bow", "bows", "bowing"]):
        timeline.append(
            {
                "seg": seg,
                "action": "bow",
                "humanml3d_class": "UNKNOWN",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": None,
                "gesture": None,
                "hands": None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("gesture_bow", "Bow segment")
        seg += 1

    # hold pose
    if _has_any(prompt, ["停在", "保持", "握住", "双手在身前", "雙手在身前", "不要走动", "只说"]):
        timeline.append(
            {
                "seg": seg,
                "action": "hold_pose",
                "humanml3d_class": "STAND",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": "center" if _has_any(prompt, ["正前", "中央"]) else None,
                "target_pose_ref": "N2" if any(r["id"] == "N2" for r in merged_refs) or "N2" in prompt else None,
                "gesture": None,
                "hands": "clasped_front" if _has_any(prompt, ["握住", "身前"]) else None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("hold_keypose", "Hold key pose segment")
        add_module("ref_pose_align", "Key pose / end pose reference alignment")
        seg += 1

    # speak
    if _has_any(prompt, ["講", "讲", "说", "粤語", "粵語", "欢迎语", "歡迎", "嘴巴"]):
        lang = "yue" if _has_any(prompt, ["粤語", "粵語", "粤语"]) else "zh"
        script_id = None
        m = re.search(r"fnaL\d+", prompt, re.I)
        if m:
            script_id = m.group(0)
        timeline.append(
            {
                "seg": seg,
                "action": "speak",
                "humanml3d_class": "STAND",
                "speed": None,
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": "N2" if any(r["id"] == "N2" for r in merged_refs) else None,
                "gesture": None,
                "hands": None,
                "speech": {"lang": lang, "script_id": script_id, "text": None},
                "notes": ["small mouth amplitude"] if _has_any(prompt, ["嘴巴开合幅度小", "开合幅度小"]) else [],
            }
        )
        add_module("speak_stub", "Speech / script attachment")
        add_module("lip_sync_stub", "Lip sync amplitude constraint")
        seg += 1

    # jump fallback for legacy prompts
    if _has_any(prompt, ["跳", "跳跃"]) and not timeline:
        timeline.append(
            {
                "seg": 1,
                "action": "jump_forward",
                "humanml3d_class": "JUMP",
                "speed": "normal",
                "steps": None,
                "duration_hint": None,
                "pos": None,
                "target_pose_ref": None,
                "gesture": None,
                "hands": "arms_up" if _has_any(prompt, ["双手", "雙臂", "上伸"]) else None,
                "speech": None,
                "notes": [],
            }
        )
        add_module("action_jump", "Jump prompt")

    if len(timeline) >= 1:
        add_module("timeline_compose", "Multi-segment or single-clip compose")
    add_module("quality_check", "Always validate against intent")

    # Reject clearly wrong modules for demos
    rejected = []
    for bad, reason in [
        ("action_jump", "No jump semantics in prompt"),
        ("locomotion_run", "No run semantics in prompt"),
        ("gesture_bow", "No bow semantics in prompt"),
        ("action_turn", "No turn semantics in prompt"),
    ]:
        if bad not in modules and not _has_any(prompt, {
            "action_jump": ["跳", "jump"],
            "locomotion_run": ["跑", "run"],
            "gesture_bow": ["鞠躬", "bow"],
            "action_turn": ["转身", "转体", "turn", "spin"],
        }[bad]):
            rejected.append({"id": bad, "reason": reason})

    style_tags = []
    for tag, kws in {
        "professional": ["专业", "專業"],
        "natural": ["自然"],
        "tech": ["科技", "蓝灰", "藍灰"],
        "friendly": ["亲切", "親切"],
    }.items():
        if _has_any(prompt, kws):
            style_tags.append(tag)

    conf = 0.9 if timeline else 0.4
    if clarify:
        conf = min(conf, 0.55)

    return {
        "schema_version": "1.0",
        "task_id": task_id or "GEN",
        "task_mode": "generate",
        "raw_prompt": prompt,
        "confidence": conf,
        "needs_clarify": clarify,
        "refs": merged_refs,
        "scene": {
            "description": prompt[:80] + ("…" if len(prompt) > 80 else ""),
            "lock_background_to": lock_bg,
            "camera": {"move": cam_move if cam_move is not None else True, "cut": cam_cut if cam_cut is not None else True, "note": ""},
            "style_tags": style_tags,
        },
        "character": {
            "appearance_ref": appearance,
            "facing": "camera" if _has_any(prompt, ["看向镜头", "对着镜头", "對着鏡頭"]) else None,
            "expression": "mild_smile" if _has_any(prompt, ["微笑", "缓和"]) else None,
            "lip": "small_amplitude" if _has_any(prompt, ["开合幅度小", "嘴巴开合"]) else None,
            "notes": [],
        },
        "timeline": timeline,
        "modules_needed": modules,
        "revision": None,
        "decision": {
            "why": "Heuristic keyword/timeline compile; HumanML3D class attached per segment only.",
            "module_reasons": module_reasons,
            "rejected_modules": rejected,
        },
    }


def parse_prompt(
    prompt: str,
    mode: str = "generate",
    refs: list[dict] | None = None,
    task_id: str | None = None,
    previous_task_id: str | None = None,
    previous_action: str | None = None,
) -> dict[str, Any]:
    if mode == "revise":
        intent = parse_revise(
            prompt,
            previous_task_id=previous_task_id or task_id,
            previous_action=previous_action,
        )
    else:
        intent = parse_generate(prompt, refs=refs, task_id=task_id)
    intent["task_id"] = task_id or intent.get("task_id") or "TASK"
    return intent


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", default="generate", choices=["generate", "revise"])
    ap.add_argument("--task-id", default="MANUAL")
    args = ap.parse_args()
    print(json.dumps(parse_prompt(args.prompt, mode=args.mode, task_id=args.task_id), ensure_ascii=False, indent=2))
