"""
Mock module runners — echo params for demos until real backends exist.
"""
from __future__ import annotations

from typing import Any


def run_mock_module(module_id: str, intent: dict[str, Any], plan_step: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": module_id,
        "ok": True,
        "mock": True,
        "owner_agent": plan_step.get("owner_agent"),
        "params_echo": {
            "task_id": intent.get("task_id"),
            "task_mode": intent.get("task_mode"),
            "why": plan_step.get("why"),
            "timeline_actions": [t.get("action") for t in intent.get("timeline") or []],
            "revision_targets": (intent.get("revision") or {}).get("targets"),
        },
        "artifact": f"outputs/intent/mocks/{intent.get('task_id', 'TASK')}_{module_id}.stub.json",
    }


def run_plan(intent: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [run_mock_module(s["module"], intent, s) for s in plan.get("steps", [])]
