"""
Route Intent Schema → planner steps (module call graph).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / ".agents" / "skills" / "intent-routing" / "module_registry.json"


def load_registry() -> dict:
    return json.loads(REG.read_text(encoding="utf-8"))


def route_modules(intent: dict[str, Any]) -> dict[str, Any]:
    registry = {m["id"]: m for m in load_registry()["modules"]}
    mode = intent.get("task_mode", "generate")

    if mode == "revise":
        rev = intent.get("revision") or {}
        rerun = rev.get("modules_to_rerun") or intent.get("modules_needed") or []
        frozen = rev.get("modules_frozen") or []
        steps = []
        for i, mid in enumerate(rerun, 1):
            meta = registry.get(mid, {})
            steps.append(
                {
                    "step": i,
                    "module": mid,
                    "owner_agent": meta.get("owner_agent", "motion-coder"),
                    "depends_on": [i - 1] if i > 1 and mid == "quality_check" else [],
                    "parallel_group": 0 if meta.get("parallel_safe") and mid != "quality_check" else i,
                    "mock": meta.get("mock", True),
                    "why": (intent.get("decision") or {}).get("module_reasons", {}).get(mid, "revision rerun"),
                }
            )
        return {
            "plan_id": f"{intent.get('task_id', 'TASK')}_revise_plan",
            "mode": "revise",
            "steps": steps,
            "frozen_modules": frozen,
            "summary": f"Rerun {len(rerun)} modules; freeze {len(frozen)}.",
        }

    needed = list(intent.get("modules_needed") or [])
    # Ensure compose before quality
    ordered: list[str] = []
    for mid in needed:
        if mid not in ordered and mid not in ("timeline_compose", "quality_check"):
            ordered.append(mid)
    if "timeline_compose" in needed:
        ordered.append("timeline_compose")
    if "quality_check" in needed:
        ordered.append("quality_check")

    # Parallel group: locks/appearance first (group 0), motion chain serial, QC last
    parallel_first = {"bg_lock", "camera_lock", "appearance_ref", "speak_stub"}
    steps = []
    serial_idx = 10
    for i, mid in enumerate(ordered, 1):
        meta = registry.get(mid, {})
        if mid in parallel_first:
            pg, deps = 0, []
        elif mid == "quality_check":
            pg, deps = 99, [s["step"] for s in steps]
        elif mid == "timeline_compose":
            pg, deps = 50, [s["step"] for s in steps if s["module"] not in parallel_first]
            serial_idx = 50
        else:
            serial_idx += 1
            pg = serial_idx
            prev_serial = [s["step"] for s in steps if s["parallel_group"] not in (0, 99) and s["module"] != "speak_stub"]
            deps = [prev_serial[-1]] if prev_serial else []
        steps.append(
            {
                "step": i,
                "module": mid,
                "owner_agent": meta.get("owner_agent", "motion-coder"),
                "depends_on": deps,
                "parallel_group": pg,
                "mock": meta.get("mock", True),
                "humanml3d_class": meta.get("humanml3d_class"),
                "why": (intent.get("decision") or {}).get("module_reasons", {}).get(mid, ""),
            }
        )

    return {
        "plan_id": f"{intent.get('task_id', 'TASK')}_gen_plan",
        "mode": "generate",
        "steps": steps,
        "frozen_modules": [],
        "summary": f"Generate plan with {len(steps)} module calls; timeline segs={len(intent.get('timeline') or [])}.",
    }
