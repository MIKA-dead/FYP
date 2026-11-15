"""Re-score an existing edit-battery report without re-running any backend.

Useful when only the *status mapping* or the aggregate changed: the per-case
observations are already in report.json, so the numbers can be recomputed.

  python scripts/agent/rescore_edit_bench.py outputs/edit_bench3/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from edit_benchmark import _agg

REFUSAL_ERRORS = (
    "unsupported_instruction",
    "edit_needs_existing_locomotion",
    "edit_needs_existing_turn",
    "edit_needs_existing_motion",
    "edit_needs_seated_clip",
)
NO_EFFECT_ERRORS = ("edit_had_no_effect", "edit_did_not_follow_instruction")


def status_of(r: dict) -> str:
    err = str(r.get("error") or "")
    if r.get("status") == "skipped_no_source":
        return "skipped_no_source"
    if r.get("status") == "applied":
        return "applied"
    if err in REFUSAL_ERRORS or r.get("status") == "refused":
        return "refused"
    if err in NO_EFFECT_ERRORS or r.get("status") == "no_effect":
        return "no_effect"
    return str(r.get("status") or "failed")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/edit_bench3/report.json")
    rep = json.loads(path.read_text(encoding="utf-8-sig"))
    for r in rep["rows"]:
        r["status"] = status_of(r)
    rep["aggregate"] = _agg(rep["rows"])
    fam: dict = {}
    for r in rep["rows"]:
        fam.setdefault(str(r.get("family") or "refused_or_unknown"), []).append(r)
    rep["by_family"] = {k: _agg(v) for k, v in sorted(fam.items())}
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    agg = rep["aggregate"]
    print(json.dumps({k: agg[k] for k in (
        "n_applied", "n_refused", "n_no_effect", "n_applicable",
        "instruction_coverage", "refusal_honesty",
        "instruction_following_rate", "mean_target_vs_rest",
        "mean_non_target_mpjpe_norm", "mean_frame_retention",
        "rollback_rate")}, indent=1))


if __name__ == "__main__":
    main()
