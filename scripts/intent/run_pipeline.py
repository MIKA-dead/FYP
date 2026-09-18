"""
End-to-end intent routing demo:
  prompt → intent → plan → mock modules → decision_trace (+ gold review)

Usage:
  python scripts/intent/run_pipeline.py
  python scripts/intent/run_pipeline.py --case P001
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_intent import parse_prompt  # noqa: E402
from route_modules import route_modules  # noqa: E402
from mock_modules import run_plan  # noqa: E402
from evaluate_decision import evaluate_trace  # noqa: E402

GOLD_PATH = ROOT / "test-samples" / "intent" / "prompts_with_gold.json"
OUT_DIR = ROOT / "outputs" / "intent"


def run_case(case: dict) -> dict:
    intent = parse_prompt(
        prompt=case["prompt"],
        mode=case.get("mode", "generate"),
        refs=case.get("refs"),
        task_id=case["id"],
        previous_task_id=case.get("previous_task_id"),
    )
    plan = route_modules(intent)
    mock_results = run_plan(intent, plan)
    trace = {
        "case_id": case["id"],
        "title": case.get("title"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "prompt": case["prompt"],
            "mode": case.get("mode"),
            "refs": case.get("refs") or [],
            "previous_task_id": case.get("previous_task_id"),
        },
        "intent": intent,
        "plan": plan,
        "mock_results": mock_results,
    }
    review = evaluate_trace(trace, case["gold"])
    trace["review"] = review
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Run a single case id, e.g. P001")
    ap.add_argument("--gold-file", type=Path, default=GOLD_PATH)
    args = ap.parse_args()

    gold_doc = json.loads(args.gold_file.read_text(encoding="utf-8"))
    cases = gold_doc["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"Case not found: {args.case}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mocks").mkdir(parents=True, exist_ok=True)

    summary = []
    for case in cases:
        trace = run_case(case)
        out = OUT_DIR / f"{case['id']}_decision_trace.json"
        out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        r = trace["review"]
        line = f"{case['id']}: {r['verdict']} ({r['passed']}/{r['total']} = {r['score']}%)"
        print(line)
        for c in r["checks"]:
            mark = "OK" if c["ok"] else "FAIL"
            if not c["ok"]:
                print(f"  [{mark}] {c['name']}: {c['detail']}")
        summary.append(
            {
                "id": case["id"],
                "title": case.get("title"),
                "verdict": r["verdict"],
                "score": r["score"],
                "passed": r["passed"],
                "total": r["total"],
                "trace": str(out.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cases": summary,
                "correct": sum(1 for s in summary if s["verdict"] == "correct"),
                "partial": sum(1 for s in summary if s["verdict"] == "partial"),
                "wrong": sum(1 for s in summary if s["verdict"] == "wrong"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
