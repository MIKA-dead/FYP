"""
Module (b) benchmark: the editor, measured on edits that must change something.

The 20-goal agent benchmark only exercises the editor when the first draft
misses, which happened twice — a pass rate over n=2 says nothing. This runs a
fixed battery of *must-edit* cases instead: a source clip is generated, one
instruction is applied through the same ``MotionTools.edit_motion`` the agent
calls, and the result is measured with the same before/after comparison.

Every case declares what a correct outcome is:

  expect="apply"   the registry accepts the instruction and the clip must change
  expect="refuse"  the editor cannot do it, and must say why instead of no-opping

Metrics (per edit and aggregated), all recomputable from the saved clips:

  instruction_coverage        applied / cases that should be applicable
  refusal_honesty             named refusals / cases that should be refused
  instruction_following_rate  did the requested attribute move the requested way
  target_change               mean displacement of the edited joints, root-relative
  non_target_mpjpe_norm       displacement outside the target region / body height
  target_vs_rest              localisation ratio (target change / rest change)
  frame_retention             frames kept / frames in the source
  no_change_rate              silent no-ops (output identical to input)
  rollback_rate               edits that introduced physical damage and were undone
  gap_before / gap_after      requirement shortfall against the instruction text

Each case gets its own copy of the source clip, so no case ever edits a clip a
previous case already changed.

  python scripts/agent/edit_benchmark.py --out outputs/edit_bench
  python scripts/agent/edit_benchmark.py --generator mld --limit 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("agent", "session", "orchestrator", "backends"):
    if str(REPO / "scripts" / _p) not in sys.path:
        sys.path.insert(0, str(REPO / "scripts" / _p))

from tools import MotionTools  # noqa: E402

# (id, source prompt, instruction, expectation, source precondition).
# Every instruction targets an attribute a measurement can confirm, so "did the
# editor do what it was told" is decidable. Instructions that say "wave" are
# deliberately absent: the wave recipe rebuilds the clip through MLD
# (force_bridged_walk), which costs minutes per edit and would measure the
# generator rather than the editor.
SCENARIOS: List[Dict[str, str]] = [
    {"id": "arm_right", "source": "a person stands still", "needs": "",
     "instruction": "raise the right arm higher", "expect": "apply"},
    {"id": "arm_left", "source": "a person stands still", "needs": "",
     "instruction": "raise the left arm higher", "expect": "apply"},
    {"id": "arms_both", "source": "a person stands still", "needs": "",
     "instruction": "raise both arms above the head", "expect": "apply"},
    {"id": "hand_right", "source": "a person walks forward", "needs": "travel",
     "instruction": "raise the right hand higher", "expect": "apply"},
    {"id": "walk_further", "source": "a person walks forward", "needs": "travel",
     "instruction": "walk forward further", "expect": "apply"},
    {"id": "walk_slower", "source": "a person walks forward", "needs": "travel",
     "instruction": "walk slower", "expect": "apply"},
    {"id": "walk_faster", "source": "a person walks forward", "needs": "travel",
     "instruction": "walk faster", "expect": "apply"},
    {"id": "walk_long_steps", "source": "a person walks forward", "needs": "travel",
     "instruction": "walk forward with longer steps", "expect": "apply"},
    {"id": "sit_lower", "source": "a person sits down on a chair", "needs": "seated",
     "instruction": "sit lower", "expect": "apply"},
    {"id": "stand_up", "source": "a person sits down on a chair", "needs": "seated",
     "instruction": "stand up", "expect": "apply"},
    {"id": "turn_stop", "source": "a person turns around", "needs": "turn",
     "instruction": "stop turning", "expect": "apply"},
    {"id": "trim_tail", "source": "a person sits down on a chair", "needs": "seated",
     "instruction": "trim the extra motion at the end", "expect": "apply"},
    {"id": "sit_no_preamble",
     "source": "a person walks and then sits down on a chair", "needs": "seated",
     "instruction": "remove the walking before sitting", "expect": "apply"},
    # Refusals: each has a named reason, and none may silently no-op.
    {"id": "walk_in_place", "source": "a person walks forward", "needs": "travel",
     "instruction": "walk in place", "expect": "refuse"},
    {"id": "sit_down", "source": "a person stands still", "needs": "",
     "instruction": "sit down", "expect": "refuse"},
    {"id": "sit_hold", "source": "a person sits down on a chair", "needs": "seated",
     "instruction": "sit down and hold", "expect": "refuse"},
    {"id": "kick_right", "source": "a person stands still", "needs": "",
     "instruction": "kick with the right leg", "expect": "refuse"},
    {"id": "jump_higher", "source": "a person jumps", "needs": "",
     "instruction": "jump higher", "expect": "refuse"},
    # Precondition refusals: the operator exists, but this clip cannot support it.
    {"id": "turn_without_turn", "source": "a person stands still", "needs": "",
     "instruction": "stop turning", "expect": "refuse"},
    {"id": "speed_without_walk", "source": "a person stands still", "needs": "",
     "instruction": "walk faster", "expect": "refuse"},
    {"id": "stand_up_while_standing", "source": "a person walks forward", "needs": "travel",
     "instruction": "stand up", "expect": "refuse"},
]

AFTER_KEYS = (
    "root_travel", "yaw_span_deg", "pelvis_drop", "arm_raise_left",
    "arm_raise_right", "ankle_lift_right", "ankle_lift_left", "jump_rise", "T",
)

# What a source clip must already contain for a relative operator to work.
NEED_THRESHOLDS = {
    "travel": ("root_travel", 0.15),
    "seated": ("pelvis_drop", 0.05),
    "turn": ("yaw_span_deg", 20.0),
}


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    applied = [r for r in rows if r.get("status") == "applied"]
    should_apply = [r for r in rows if r.get("expect") == "apply"
                    and r.get("status") != "skipped_no_source"]
    should_refuse = [r for r in rows if r.get("expect") == "refuse"]
    refused = [r for r in should_refuse if r.get("status") == "refused"]

    def mean(key: str, seq=None):
        vals = [float(r[key]) for r in (seq if seq is not None else rows)
                if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    verifiable = [r for r in applied if r.get("verifiable", True)]
    return {
        "n_scenarios": len(rows),
        "n_attempts": len(rows) - sum(1 for r in rows if r.get("status") == "skipped_no_source"),
        "n_applied": len(applied),
        "n_refused": len(refused),
        "n_applicable": len(should_apply),
        "n_expected_refusals": len(should_refuse),
        "n_unsupported": sum(1 for r in rows if r.get("status") == "refused"),
        "n_rolled_back": sum(1 for r in rows if r.get("status") == "invalid_clip"),
        "n_no_effect": sum(1 for r in rows if r.get("status") == "no_effect"),
        "n_skipped_no_source": sum(1 for r in rows
                                   if r.get("status") == "skipped_no_source"),
        "n_no_change": sum(1 for r in applied if r.get("no_change")),
        "n_unverifiable": sum(1 for r in applied if not r.get("verifiable", True)),
        "instruction_coverage": round(
            float(len(applied)) / len(should_apply), 3
        ) if should_apply else None,
        "refusal_honesty": round(
            float(len(refused)) / len(should_refuse), 3
        ) if should_refuse else None,
        "instruction_following_rate": round(
            float(np.mean([bool(r.get("followed")) for r in verifiable])), 3
        ) if verifiable else None,
        "no_change_rate": mean("no_change", applied),
        "mean_target_change": mean("target_change", applied),
        "mean_non_target_mpjpe_norm": mean("non_target_mpjpe_norm", applied),
        "mean_target_vs_rest": mean("target_vs_rest", applied),
        "mean_frame_retention": mean("frames_ratio", applied),
        "mean_seconds": mean("seconds"),
        "rollback_rate": round(
            float(sum(1 for r in rows if r.get("status") == "invalid_clip")) / max(1, len(rows)), 3
        ) if rows else None,
    }


def _stage_source(src: Path, case_dir: Path) -> Path:
    """Copy the source clip into the case directory, so no case edits another's."""
    stage = case_dir / "step00_source"
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("03_edited.npy", "04_quality.json", "summary.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, stage / name)
    for gif in src.glob("*preview*.gif"):
        shutil.copy2(gif, stage / gif.name)
    return stage


def _source_problem(sc: Dict[str, str], measured: Dict[str, Any]) -> str:
    need = sc.get("needs") or ""
    if not need:
        return ""
    key, threshold = NEED_THRESHOLDS[need]
    got = float(measured.get(key) or 0)
    if got >= threshold:
        return ""
    return f"source has {key}={got:.3f}, needs >= {threshold}"


def run(scenarios: List[Dict[str, str]], *, out_dir: Path, generator: str,
        skip_viz: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows: List[Dict[str, Any]] = []
    sources: Dict[str, Path] = {}
    for i, sc in enumerate(scenarios, 1):
        print(f"[edit_bench] {i}/{len(scenarios)}: {sc['id']} — {sc['instruction']}",
              flush=True)
        t0 = time.time()
        case_dir = out_dir / f"{i:02d}_{sc['id']}_{stamp}"
        tools = MotionTools(case_dir, generator=generator,
                            skip_viz=skip_viz, goal=sc["instruction"])
        src = sources.get(sc["source"])
        if src is None:
            gen = tools.generate_motion(sc["source"])
            if not gen.get("ok"):
                rows.append({"id": sc["id"], "status": "generate_failed",
                             "instruction": sc["instruction"], "expect": sc["expect"],
                             "error": gen.get("error"), "source": sc["source"]})
                continue
            src = Path(str(gen["run_dir"]))
            sources[sc["source"]] = src
        staged = _stage_source(src, case_dir)
        tools.run_dir = staged
        tools.described_at = tools.step
        tools._snapshot("source")
        src_metrics = dict((tools.current_snapshot() or {}).get("measured") or {})
        problem = _source_problem(sc, src_metrics)
        if problem:
            rows.append({
                "id": sc["id"], "source": sc["source"],
                "instruction": sc["instruction"], "expect": sc["expect"],
                "status": "skipped_no_source", "family": None,
                "error": problem, "followed": None, "verifiable": False,
                "checks": [], "before": {k: v for k, v in src_metrics.items() if v is not None},
                "seconds": round(time.time() - t0, 2), "run_dir": str(staged),
            })
            print(f"[edit_bench]   skipped: {problem}", flush=True)
            continue
        obs = tools.edit_motion(sc["instruction"])
        after = obs.get("after") or {}
        follow = obs.get("instruction_following") or {}
        pres = obs.get("preservation") or {}
        if obs.get("ok"):
            status = "applied"
        elif obs.get("error") in ("unsupported_instruction",
                                  "edit_needs_existing_locomotion",
                                  "edit_needs_existing_turn",
                                  "edit_needs_existing_motion",
                                  "edit_needs_seated_clip"):
            status = "refused"
        elif obs.get("error") in ("edit_had_no_effect",
                                  "edit_did_not_follow_instruction"):
            status = "no_effect"
        elif obs.get("rolled_back"):
            status = "invalid_clip"
        else:
            status = "failed"
        rows.append({
            "id": sc["id"],
            "source": sc["source"],
            "instruction": sc["instruction"],
            "expect": sc["expect"],
            "status": status,
            "family": obs.get("family"),
            "error": obs.get("error") or obs.get("reason"),
            "followed": bool(follow.get("followed")),
            "verifiable": follow.get("verifiable", True),
            "checks": follow.get("checks") or [],
            "no_change": bool(obs.get("no_change")),
            "target_change": pres.get("target_change"),
            "non_target_mpjpe_norm": pres.get("non_target_mpjpe_norm"),
            "target_vs_rest": pres.get("target_vs_rest"),
            "frames_ratio": obs.get("frames_ratio"),
            "gap_before": obs.get("gap_before"),
            "gap_after": obs.get("gap_after"),
            "before": {k: v for k, v in src_metrics.items() if v is not None},
            "after": {k: after.get(k) for k in AFTER_KEYS if after.get(k) is not None},
            "seconds": round(time.time() - t0, 2),
            "run_dir": obs.get("run_dir") or str(staged),
        })
        print(f"[edit_bench]   {status} followed={rows[-1]['followed']} "
              f"gap {rows[-1]['gap_before']} -> {rows[-1]['gap_after']}", flush=True)

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_family.setdefault(str(r.get("family") or "refused_or_unknown"), []).append(r)
    report = {
        "schema_version": "edit_bench_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"generator": generator, "skip_viz": skip_viz, "n": len(scenarios)},
        "aggregate": _agg(rows),
        "by_family": {k: _agg(v) for k, v in sorted(by_family.items())},
        "rows": rows,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Module (b) editing benchmark")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "edit_bench")
    ap.add_argument("--generator", default="dataset", choices=("dataset", "mld"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-viz", action="store_true")
    args = ap.parse_args()
    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS
    report = run(scenarios, out_dir=args.out, generator=args.generator,
                 skip_viz=args.skip_viz)
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
