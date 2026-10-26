"""
Load module_registry.json and map semantic planner modules → backend executables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
REG_PATH = REPO_ROOT / ".agents" / "skills" / "intent-routing" / "module_registry.json"

# Fallback when registry entry has no maps_to.
SEMANTIC_TO_BACKEND: Dict[str, List[str]] = {
    "mld_generate": ["mld_generate"],
    "mld_bridge": ["mld_bridge"],
    "motionfix_semantic_edit": ["motionfix_semantic_edit"],
    "speed_edit": ["speed_edit"],
    "sit_hold_postprocess": ["sit_hold_postprocess"],
    "timeline_postprocess": ["timeline_postprocess"],
    "wave_walk_postprocess": ["wave_walk_postprocess"],
    "quality_check": ["quality_gate"],
    "quality_gate": ["quality_gate"],
    "visualize": ["visualize"],
    "action_sit": ["motionfix_semantic_edit"],
    "action_turn": ["motionfix_semantic_edit"],
    "gesture_wave": ["motionfix_semantic_edit"],
    "gesture_bow": ["motionfix_semantic_edit"],
    "action_jump": ["mld_generate", "mld_bridge", "motionfix_semantic_edit"],
    "locomotion_walk": ["mld_generate", "mld_bridge", "motionfix_semantic_edit"],
    "locomotion_run": ["mld_generate", "mld_bridge", "motionfix_semantic_edit"],
    "idle_stand": ["motionfix_semantic_edit"],
    "motion_edit": ["motionfix_semantic_edit"],
    "keypose_edit": ["motionfix_semantic_edit"],
    "hold_keypose": ["motionfix_semantic_edit"],
    "ref_pose_align": ["motionfix_semantic_edit"],
    "timeline_compose": [],
    "bg_lock": ["mock"],
    "camera_lock": ["mock"],
    "appearance_ref": ["mock"],
    "texture_edit": ["mock"],
    "lip_sync_stub": ["mock"],
    "speak_stub": ["mock"],
    "mock": ["mock"],
}


def load_registry() -> dict:
    return json.loads(REG_PATH.read_text(encoding="utf-8-sig"))


def resolve_module(module_id: str) -> Optional[Dict[str, Any]]:
    for m in load_registry().get("modules") or []:
        if m.get("id") == module_id:
            return m
    return None


def expand_semantic_modules(module_ids: List[str]) -> List[str]:
    """Expand planner module ids to unique ordered backend ids."""
    out: List[str] = []
    for mid in module_ids:
        meta = resolve_module(mid) or {}
        if "maps_to" in meta:
            backends = list(meta.get("maps_to") or [])
        elif meta.get("backend"):
            backends = [mid]
        else:
            backends = list(SEMANTIC_TO_BACKEND.get(mid, [mid] if mid else []))
        for b in backends:
            if b and b not in out:
                out.append(b)
    return out


def backend_meta(module_id: str) -> Dict[str, Any]:
    meta = resolve_module(module_id)
    if meta:
        return meta
    return {
        "id": module_id,
        "name": module_id,
        "mock": module_id == "mock",
        "owner_agent": "motion-coder",
        "backend": True,
    }
