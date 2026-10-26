"""
Validate (or regenerate) demo coverage cases for UI.

Default: quality-gate golden runs under outputs/v05/session_*.
--rerun: call session_api for each case (slow; needs GPU envs).

  python scripts/session/run_demos.py
  python scripts/session/run_demos.py --rerun
  python scripts/session/run_demos.py --export-handoff
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = Path(__file__).resolve().parent
CASES_PATH = REPO_ROOT / "test-samples" / "session" / "demo_cases.json"
OUT_DIR = REPO_ROOT / "outputs" / "session"


def _ensure_numpy_env() -> None:
    """Re-exec under MOTIONFIX_PYTHON when host Python lacks numpy."""
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    import os
    from pathlib import Path as P

    mf = os.environ.get("MOTIONFIX_PYTHON") or str(
        P.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"
    )
    if not P(mf).is_file():
        raise SystemExit(
            "numpy missing and MOTIONFIX_PYTHON not found; "
            "run with conda env motionfix"
        )
    os.execv(mf, [mf, *sys.argv])


_ensure_numpy_env()

if str(SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(SESSION_DIR))

from quality_gate import evaluate_file  # noqa: E402
from session_api import run_session  # noqa: E402


def _load_cases() -> Dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8-sig"))


def _check_case(case: Dict[str, Any], *, rerun: bool) -> Dict[str, Any]:
    cid = case["id"]
    result: Dict[str, Any] = {
        "id": cid,
        "prompt": case["prompt"],
        "primary_action": case.get("primary_action"),
        "ok": False,
        "checks": [],
    }

    run_dir: Optional[Path] = None
    if rerun:
        resp = run_session(
            {
                "prompt": case["prompt"],
                "mode": case.get("mode") or "generate",
                "run_dir": str(OUT_DIR / "demo_runs" / cid.lower()),
                "skip_review": True,
                "skip_viz": False,
                "length": 60,
            },
            cwd=REPO_ROOT,
        )
        result["session_response"] = {
            "ok": resp.get("ok"),
            "run_dir": resp.get("run_dir"),
            "error": resp.get("error"),
            "quality_score": (resp.get("quality") or {}).get("score"),
        }
        run_dir = Path(resp["run_dir"]) if resp.get("run_dir") else None
        plan = resp.get("intent") or {}
    else:
        golden = case.get("golden_run")
        if not golden:
            result["checks"].append({"id": "golden_path", "ok": False, "detail": "missing"})
            return result
        run_dir = (REPO_ROOT / golden).resolve()
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            result["checks"].append(
                {"id": "golden_summary", "ok": False, "detail": f"missing {summary_path}"}
            )
            return result
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        plan = summary.get("motion_edit_plan") or {}
        result["golden_run"] = str(run_dir)

    # Intent expectations
    act_ok = plan.get("primary_action") == case.get("primary_action")
    result["checks"].append(
        {
            "id": "primary_action",
            "ok": act_ok,
            "detail": f"got={plan.get('primary_action')} want={case.get('primary_action')}",
        }
    )
    if case.get("expected_edit_source"):
        src_ok = plan.get("edit_source") == case["expected_edit_source"]
        result["checks"].append(
            {
                "id": "edit_source",
                "ok": src_ok,
                "detail": f"got={plan.get('edit_source')} want={case['expected_edit_source']}",
            }
        )
    if case.get("expected_source_id"):
        id_ok = str(plan.get("preferred_source_id")) == str(case["expected_source_id"])
        result["checks"].append(
            {
                "id": "preferred_source_id",
                "ok": id_ok,
                "detail": f"got={plan.get('preferred_source_id')} want={case['expected_source_id']}",
            }
        )

    assert run_dir is not None
    edited = run_dir / "03_edited.npy"
    gif = run_dir / "03_edited_preview.gif"
    result["checks"].append(
        {"id": "edited_npy", "ok": edited.is_file(), "detail": str(edited)}
    )
    result["checks"].append(
        {"id": "preview_gif", "ok": gif.is_file(), "detail": str(gif)}
    )
    result["artifacts"] = {
        "run_dir": str(run_dir),
        "edited_npy": str(edited) if edited.is_file() else None,
        "preview_gif": str(gif) if gif.is_file() else None,
    }

    q_cfg = case.get("quality") or {}
    if edited.is_file():
        q = evaluate_file(edited, action=case.get("primary_action"))
        # Persist gate next to golden
        q_path = run_dir / "04_quality.json"
        q_path.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
        passed = bool(q.get("passed"))
        score = int(q.get("score") or 0)
        min_score = int(q_cfg.get("min_score") or 0)
        req = bool(q_cfg.get("require_passed", True))
        q_ok = ((not req) or passed) and score >= min_score
        result["checks"].append(
            {
                "id": "quality_gate",
                "ok": q_ok,
                "detail": f"passed={passed} score={score} min={min_score} hard={q.get('hard_failures')}",
            }
        )
        result["quality"] = {
            "passed": passed,
            "score": score,
            "hard_failures": q.get("hard_failures"),
            "metrics": q.get("metrics"),
        }
    else:
        result["checks"].append(
            {"id": "quality_gate", "ok": False, "detail": "no edited npy"}
        )

    result["ok"] = all(c["ok"] for c in result["checks"])
    return result


def build_handoff(demo_report: Dict[str, Any]) -> Dict[str, Any]:
    """Self-contained UI handoff (no teammate wait)."""
    cases_out = []
    for c in demo_report.get("results") or []:
        cases_out.append(
            {
                "id": c["id"],
                "prompt": c["prompt"],
                "primary_action": c.get("primary_action"),
                "ok": c.get("ok"),
                "preview_gif": (c.get("artifacts") or {}).get("preview_gif"),
                "edited_npy": (c.get("artifacts") or {}).get("edited_npy"),
                "run_dir": (c.get("artifacts") or {}).get("run_dir"),
                "quality": c.get("quality"),
            }
        )
    return {
        "schema_version": "handoff_for_ui_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api": {
            "entrypoint": "scripts/session/session_api.py",
            "request": {
                "prompt": "string",
                "mode": "generate|revise",
                "previous_run": "path|null",
                "run_dir": "path|null",
                "skip_review": True,
                "skip_viz": False,
                "length": 60,
            },
            "response_schema": "session_response_v1",
            "response_file": "<run_dir>/session_response.json",
            "quality_file": "<run_dir>/04_quality.json",
            "revise_hint": {
                "mode": "revise",
                "previous_run": "<prior run_dir>",
                "prompt": "user improvement text",
            },
        },
        "demo_cases": str(CASES_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "demo_report": str(
            (OUT_DIR / "demo_report.json").relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "cases": cases_out,
        "ui_min_loop": [
            "POST/CLI generate with prompt → show intent + preview_gif",
            "If quality.passed is false, show hard_failures and offer revise",
            "User types improvement → mode=revise + previous_run",
            "Replace preview with new gif; keep Decision Trace from 00_agent_bundle.json",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run / validate session demos")
    ap.add_argument("--rerun", action="store_true", help="Regenerate via session_api (GPU)")
    ap.add_argument("--export-handoff", action="store_true", default=True)
    ap.add_argument("--no-export-handoff", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="Comma-separated case ids")
    args = ap.parse_args()

    pack = _load_cases()
    cases: List[Dict[str, Any]] = list(pack.get("cases") or [])
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]

    results = []
    for case in cases:
        print(f"\n=== {case['id']} ===", flush=True)
        r = _check_case(case, rerun=bool(args.rerun))
        results.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] {case['id']} checks={r['checks']}", flush=True)

    report = {
        "schema_version": "demo_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rerun": bool(args.rerun),
        "n_cases": len(results),
        "n_pass": sum(1 for r in results if r["ok"]),
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "demo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[run_demos] {report['n_pass']}/{report['n_cases']} passed → {report_path}")

    if args.export_handoff and not args.no_export_handoff:
        handoff = build_handoff(report)
        handoff_path = OUT_DIR / "handoff_for_ui.json"
        handoff_path.write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[run_demos] handoff → {handoff_path}")

    if report["n_pass"] < report["n_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
