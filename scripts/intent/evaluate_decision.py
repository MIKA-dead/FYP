"""
Compare a decision_trace against gold labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def evaluate_trace(trace: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    intent = trace.get("intent") or {}
    plan = trace.get("plan") or {}
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if "task_mode" in gold:
        check("task_mode", intent.get("task_mode") == gold["task_mode"], f"got={intent.get('task_mode')}")

    timeline = intent.get("timeline") or []
    actions = [t.get("action") for t in timeline]

    if "seg_count" in gold:
        check("seg_count", len(timeline) == gold["seg_count"], f"got={len(timeline)}")
    if "seg_count_min" in gold:
        check("seg_count_min", len(timeline) >= gold["seg_count_min"], f"got={len(timeline)}")

    if "timeline_actions" in gold:
        check(
            "timeline_actions",
            actions == gold["timeline_actions"],
            f"got={actions}",
        )
    if "timeline_actions_any" in gold:
        ok = any(a in actions for a in gold["timeline_actions_any"])
        check("timeline_actions_any", ok, f"got={actions}")

    if "speed_by_seg" in gold:
        for seg_s, speed in gold["speed_by_seg"].items():
            seg = int(seg_s)
            hit = next((t for t in timeline if t.get("seg") == seg), None)
            check(f"speed_seg_{seg}", bool(hit) and hit.get("speed") == speed, f"got={None if not hit else hit.get('speed')}")

    if "speed_any" in gold:
        speeds = [t.get("speed") for t in timeline]
        check("speed_any", any(s in gold["speed_any"] for s in speeds), f"got={speeds}")

    if "steps_by_seg" in gold:
        for seg_s, steps in gold["steps_by_seg"].items():
            seg = int(seg_s)
            hit = next((t for t in timeline if t.get("seg") == seg), None)
            check(f"steps_seg_{seg}", bool(hit) and hit.get("steps") == steps, f"got={None if not hit else hit.get('steps')}")

    if "steps_by_seg_any" in gold:
        check("steps_any", any(t.get("steps") == gold["steps_by_seg_any"] for t in timeline), f"got={[t.get('steps') for t in timeline]}")

    if "gesture" in gold:
        gestures = [t.get("gesture") for t in timeline if t.get("gesture")]
        check("gesture", gold["gesture"] in gestures, f"got={gestures}")

    if "appearance_ref" in gold:
        check(
            "appearance_ref",
            _get(intent, "character", "appearance_ref") == gold["appearance_ref"],
            f"got={_get(intent, 'character', 'appearance_ref')}",
        )

    if "lock_background_to" in gold:
        check(
            "lock_background_to",
            _get(intent, "scene", "lock_background_to") == gold["lock_background_to"],
            f"got={_get(intent, 'scene', 'lock_background_to')}",
        )

    if "camera" in gold:
        cam = _get(intent, "scene", "camera") or {}
        for k, v in gold["camera"].items():
            check(f"camera.{k}", cam.get(k) == v, f"got={cam.get(k)}")

    if "speech" in gold:
        speeches = [t.get("speech") or {} for t in timeline if t.get("action") == "speak"]
        speech = speeches[0] if speeches else {}
        for k, v in gold["speech"].items():
            check(f"speech.{k}", speech.get(k) == v, f"got={speech.get(k)}")

    modules = set(intent.get("modules_needed") or [])
    if intent.get("task_mode") == "revise":
        modules = set((intent.get("revision") or {}).get("modules_to_rerun") or modules)

    if "modules_required" in gold:
        missing = [m for m in gold["modules_required"] if m not in modules]
        check("modules_required", not missing, f"missing={missing}")

    if "modules_forbidden" in gold:
        bad = [m for m in gold["modules_forbidden"] if m in modules]
        check("modules_forbidden", not bad, f"unexpected={bad}")

    if "humanml3d_classes" in gold:
        classes = {t.get("humanml3d_class") for t in timeline if t.get("humanml3d_class")}
        missing = [c for c in gold["humanml3d_classes"] if c not in classes]
        check("humanml3d_classes", not missing, f"got={sorted(classes)}")

    if "needs_clarify_contains_any" in gold:
        blob = " ".join(intent.get("needs_clarify") or [])
        ok = any(x.lower() in blob.lower() for x in gold["needs_clarify_contains_any"])
        check("needs_clarify", ok, f"got={intent.get('needs_clarify')}")

    if "revision_targets" in gold:
        got = (intent.get("revision") or {}).get("targets") or []
        check("revision_targets", set(got) == set(gold["revision_targets"]), f"got={got}")

    if "revision_targets_any" in gold:
        got = set((intent.get("revision") or {}).get("targets") or [])
        ok = any(t in got for t in gold["revision_targets_any"])
        check("revision_targets_any", ok, f"got={sorted(got)}")

    if "modules_to_rerun" in gold:
        got = set((intent.get("revision") or {}).get("modules_to_rerun") or [])
        missing = [m for m in gold["modules_to_rerun"] if m not in got]
        check("modules_to_rerun", not missing, f"missing={missing}; got={sorted(got)}")

    if "modules_to_rerun_any" in gold:
        got = set((intent.get("revision") or {}).get("modules_to_rerun") or [])
        ok = any(m in got for m in gold["modules_to_rerun_any"])
        check("modules_to_rerun_any", ok, f"got={sorted(got)}")

    if "modules_frozen_contains" in gold:
        got = set((intent.get("revision") or {}).get("modules_frozen") or plan.get("frozen_modules") or [])
        missing = [m for m in gold["modules_frozen_contains"] if m not in got]
        check("modules_frozen_contains", not missing, f"missing={missing}; got={sorted(got)}")

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks) or 1
    ratio = passed / total
    if ratio >= 0.95:
        verdict = "correct"
    elif ratio >= 0.6:
        verdict = "partial"
    else:
        verdict = "wrong"

    return {
        "verdict": verdict,
        "score": round(ratio * 100, 1),
        "passed": passed,
        "total": total,
        "checks": checks,
        "manual_override": None,
        "manual_note": "Set manual_override to correct|partial|wrong after human review.",
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gold-file", default="test-samples/intent/prompts_with_gold.json")
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    gold_doc = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
    case = next(c for c in gold_doc["cases"] if c["id"] == trace.get("case_id"))
    review = evaluate_trace(trace, case["gold"])
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
