"""Cross-action battery: generate 8 actions, then raise an arm on each.

  python scripts/tests/batch_cross_actions.py                 # full 18 cases
  python scripts/tests/batch_cross_actions.py --only punch     # one carrier

Purpose is the question the arm-layer unit tests cannot answer: does the
geometric operator still hold when the body underneath is doing something other
than walking? For every carrier it runs generate once, then a revise that raises
the right arm (or both arms overhead for run / bow), and records T, score,
arm_raise_ratio and the M2T caption for both stages.

Writes <out>/results.json and <out>/report.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

# (id, generate prompt, revise prompt)
CARRIERS: List[Dict[str, str]] = [
    {"id": "kick", "prompt": "a person kicks with their right leg", "revise": "raise the right arm high"},
    {"id": "punch", "prompt": "a person punches with their right hand", "revise": "raise the right arm high"},
    {"id": "jump", "prompt": "a person jumps up and lands", "revise": "raise the right arm high"},
    {"id": "turn", "prompt": "a person turns around 180 degrees", "revise": "raise the right arm high"},
    {"id": "stand_up", "prompt": "a person stands up from a chair", "revise": "raise the right arm high"},
    {"id": "pick_up", "prompt": "a person picks up an item from the floor", "revise": "raise the right arm high"},
    {"id": "wave_both", "prompt": "a person raises both hands and waves", "revise": "raise the right arm high"},
    {"id": "walk_back", "prompt": "a person walks backward", "revise": "raise the right arm high"},
    {"id": "run_forward", "prompt": "a person runs forward", "revise": "raise both arms overhead"},
    {"id": "bow", "prompt": "a person bows", "revise": "raise both arms overhead"},
]


def _load(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _summary(run_dir: Path) -> Dict[str, Any]:
    """Everything the report needs, read back from the run's own artifacts."""
    plan = _load(run_dir / "00_motion_edit_plan.json")
    quality = _load(run_dir / "04_quality.json")
    cap = _load(run_dir / "06_motion_caption.json")
    metrics = quality.get("metrics") or {}
    return {
        "run_dir": str(run_dir),
        "exists": (run_dir / "03_edited.npy").is_file(),
        "ok": bool(quality.get("passed")) and (run_dir / "03_edited.npy").is_file(),
        "score": quality.get("score"),
        "hard_failures": quality.get("hard_failures") or [],
        "T": metrics.get("T"),
        "yaw_span_deg": metrics.get("yaw_span_deg"),
        "root_travel": metrics.get("root_travel"),
        "arm_raise_ratio": metrics.get("arm_raise_ratio"),
        "action": plan.get("primary_action"),
        "edit_mode": plan.get("edit_mode"),
        "caption": cap.get("caption"),
        "caption_source": cap.get("source"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma separated carrier ids")
    args = ap.parse_args()
    out_root = (args.out or (REPO_ROOT / "outputs" / "tests" /
                             f"cross_actions_{time.strftime('%Y%m%d-%H%M%S')}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    py = sys.executable
    session = str(REPO_ROOT / "scripts" / "orchestrator" / "run_session.py")

    rows: List[Dict[str, Any]] = []
    for carrier in CARRIERS:
        cid = carrier["id"]
        if only and cid not in only:
            continue

        # 1. generate the carrier
        gen_dir = out_root / "runs" / cid
        t0 = time.time()
        proc = _run([py, session, "--mode", "generate", "--prompt", carrier["prompt"],
                     "--run-dir", str(gen_dir), "--skip-viz"])
        gen = _summary(gen_dir)
        gen.update({"level": "generate", "id": cid, "prompt": carrier["prompt"],
                    "exit": proc.returncode, "seconds": round(time.time() - t0, 1)})
        rows.append(gen)
        print(f"[gen] {cid:<11} score={gen['score']} T={gen['T']} "
              f"src_caption={(gen['caption'] or '')[:40]!r} {gen['seconds']}s", flush=True)

        # 2. raise an arm on it; the carrier must survive the edit
        rev_dir = out_root / "runs" / f"{cid}_revise"
        t0 = time.time()
        proc = _run([py, session, "--mode", "revise", "--prompt", carrier["revise"],
                     "--previous-run", str(gen_dir), "--run-dir", str(rev_dir), "--skip-viz"])
        rev = _summary(rev_dir)
        rev.update({"level": "revise", "id": cid, "prompt": carrier["revise"],
                    "T_before": gen["T"], "exit": proc.returncode,
                    "seconds": round(time.time() - t0, 1)})
        rows.append(rev)
        print(f"[rev] {cid:<11} ok={rev['ok']} score={rev['score']} "
              f"T={rev['T_before']}→{rev['T']} ratio={rev['arm_raise_ratio']} "
              f"{rev['seconds']}s", flush=True)

    by_level = {
        lvl: {"total": sum(1 for r in rows if r["level"] == lvl),
              "passed": sum(1 for r in rows if r["level"] == lvl and r["ok"])}
        for lvl in ("generate", "revise")
    }
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {"total": len(rows), "passed": sum(1 for r in rows if r["ok"]),
                    "by_level": by_level},
        "rows": rows,
    }
    (out_root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Cross-action battery", "",
             f"- generated: {payload['generated']}",
             f"- passed: {payload['summary']['passed']}/{len(rows)}", "",
             "| level | carrier | prompt | action | edit_mode | score | T | ratio | seconds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['id']} | {r['prompt']} | {r.get('action')} | "
            f"{r.get('edit_mode')} | {r['score']} | {r['T']} | {r['arm_raise_ratio']} | {r['seconds']} |"
        )
    lines += ["", "## Captions (M2T)", "",
              "| level | carrier | caption |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['level']} | {r['id']} | {r['caption']} |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"[battery] {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
