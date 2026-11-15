"""Programmatic check of the two review points, against code + artifacts.

Not a unit test: it asserts that the *contract* the supervisor asked for is
present in the code and visible in a finished run, so the acceptance claim can
be re-verified instead of trusted.

  python scripts/tests/test_agentic_contract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("agent", "backends", "session", "orchestrator", "intent"):
    sys.path.insert(0, str(REPO / "scripts" / _p))

CHECKS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _newest_bench() -> Path:
    """Newest 20-goal benchmark directory that has a report.json."""
    dirs = [d for d in (REPO / "outputs" / "agent_bench").glob("_v3_final*")
            if (d / "report.json").is_file()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime)[-1]


def main() -> int:
    import tools
    from tools import TOOL_SPECS, MotionTools

    names = {t["name"] for t in TOOL_SPECS}
    # --- (ii) the loop is made of tools the controller chooses -------------
    check("tool_a_generate", "generate_motion" in names)
    check("tool_b_edit", "edit_motion" in names)
    check("tool_c_describe", "describe_motion" in names)
    check("tool_check_requirement", "check_requirement" in names)
    check("tool_restore_motion", "restore_motion" in names)
    check("tool_finish", "finish" in names)

    gen_spec = next(t for t in TOOL_SPECS if t["name"] == "generate_motion")
    check("generate_can_pick_backend",
          "generator" in gen_spec["args"],
          "retrieval path for an attribute the diffusion drafts lack")
    resample_spec = next(t for t in TOOL_SPECS if t["name"] == "resample_motion")
    check("resample_is_a_tool", "n" in resample_spec["args"])

    src = (REPO / "scripts" / "agent" / "agent.py").read_text(encoding="utf-8")
    check("mandatory_order_in_prompt", "MANDATORY ORDER" in src
          and "describe_motion before judging" in src)
    check("describe_enforced_by_code",
          "describe the new clip first" in src)
    check("check_before_finish_enforced",
          "measure the requirement before finishing" in src)
    check("worse_clip_cannot_be_final",
          "restore it before finishing" in src and "MAX_RESTORES" in src)
    check("loop_capped", "for step in range(1, max_steps + 1)" in src)

    # free-form text reaches the controller unchanged
    check("free_text_goal", "USER GOAL: {goal}" in src)

    # --- (i) metrics come from measurements, per module -------------------
    from metrics import (caption_signals, kinematic_summary, load_joints,
                         text_similarity)
    from verify import verify_goal

    bench = _newest_bench()
    clip = next((bench).glob("ep01_*/step01_generate/03_edited.npy"), None)
    check("have_a_clip_to_measure", clip is not None, str(clip))
    if clip is not None:
        m = kinematic_summary(load_joints(clip))
        needed = {"root_travel", "pelvis_drop", "yaw_span_deg", "yaw_net_deg",
                  "arm_raise_right", "jump_rise", "ground_penetration"}
        check("metrics_are_kinematic", needed <= set(m),
              f"{len(m)} measured attributes, e.g. root_travel={m['root_travel']}")

    sim = text_similarity("a person raises the right arm",
                          "a person raises their right arm high")
    check("caption_similarity_is_numeric",
          {"glove_cosine", "token_jaccard", "token_recall",
           "content_recall", "action_jaccard"} <= set(sim),
          json.dumps(sim))
    # A coarse caption that shares the boilerplate must not read as a match.
    wrong = text_similarity("a person sits down and raises the right arm",
                            "a person is stand still with their arm out to their side")
    check("content_recall_beats_bag_of_words",
          wrong["glove_cosine"] > 0.8 and wrong["content_recall"] <= 0.25,
          f"glove={wrong['glove_cosine']} content_recall={wrong['content_recall']}")
    verdict = caption_signals("a person sits down and raises the right arm",
                              "a person squat down and stand back up")
    check("caption_signal_flags_mismatch", bool(verdict["needs_fix"]),
          "; ".join(verdict["because"]))

    # --- the checker uses the user's wording, with measurement tolerance ---
    def _v(goal: str, **metrics):
        base = {"bone_cv": 0.01, "ground_penetration": 0.0}
        base.update(metrics)
        return verify_goal(goal, {"metrics": base})["ok"]

    check("turn_90_accepts_88",
          _v("a person turns left 90 degrees", yaw_span_deg=88.1, yaw_net_deg=85.3))
    check("turn_90_rejects_72",
          not _v("a person turns left 90 degrees", yaw_span_deg=71.9, yaw_net_deg=71.4))
    check("turn_direction_is_checked",
          not _v("a person turns left 90 degrees", yaw_span_deg=95.0, yaw_net_deg=-92.0))
    check("walk_and_turn_both_required",
          not _v("a person walks forward and turns around", yaw_span_deg=181.0,
                 yaw_net_deg=181.0, root_travel=0.0))

    # --- the archive ranks and restores ------------------------------------
    snaps = [
        {"tag": "a", "step": 1, "run_dir": "a", "satisfied": False, "gap": 0.4, "valid": True},
        {"tag": "b", "step": 2, "run_dir": "b", "satisfied": True, "gap": 0.0, "valid": True},
        {"tag": "c", "step": 3, "run_dir": "c", "satisfied": False, "gap": 0.1, "valid": True},
    ]
    ranked = sorted(snaps, key=MotionTools._rank_key)
    check("archive_ranks_closest_first", ranked[0]["run_dir"] == "b",
          " → ".join(s["run_dir"] for s in ranked))

    # --- an edit that does not move what it names is not kept -------------
    tsrc = (REPO / "scripts" / "agent" / "tools.py").read_text(encoding="utf-8")
    check("preconditions_are_named",
          all(k in tsrc for k in ("edit_needs_existing_turn",
                                  "edit_needs_existing_motion",
                                  "edit_needs_seated_clip")),
          "relative operators refuse clips that lack their precondition")
    check("no_effect_is_reported",
          "edit_had_no_effect" in tsrc and "edit_did_not_follow_instruction" in tsrc,
          "a silent no-op restores the previous clip")
    from metrics import instruction_following as _follow
    both = _follow("raise both arms above the head",
                   {"arm_raise_left": -2.1, "arm_raise_right": -2.0, "T": 100},
                   {"arm_raise_left": 2.2, "arm_raise_right": 2.1, "T": 100}, {})
    one = _follow("raise both arms above the head",
                  {"arm_raise_left": -2.1, "arm_raise_right": -2.0, "T": 100},
                  {"arm_raise_left": -2.1, "arm_raise_right": 2.1, "T": 100}, {})
    check("both_arms_checked_separately", both["followed"] and not one["followed"],
          str([(c["attribute"], c["ok"]) for c in both["checks"]]))

    # --- report sections and per-module metrics ---------------------------
    report = json.loads((bench / "report.json").read_text(encoding="utf-8-sig"))
    check("report_has_four_sections",
          {"module_a_generation", "module_b_editing", "module_c_caption", "loop"}
          <= set(report))
    check("no_saturating_score_field",
          "score" not in {k for k in report if k != "wall_seconds"},
          ", ".join(k for k in report))
    loop = report["loop"]
    for key in ("success_rate", "agreement_rate", "mean_edits", "mean_steps",
                "n_drafts_rejected", "n_restored_episodes"):
        check(f"loop_metric_{key}", key in loop, str(loop.get(key)))
    for key in ("r@1", "r@3", "mean_cosine"):
        check(f"gen_metric_{key}", key in report["module_a_generation"]["retrieval"],
              str(report["module_a_generation"]["retrieval"][key]))
    for key in ("instruction_following_rate", "instruction_coverage",
                "mean_non_target_mpjpe_norm"):
        check(f"edit_metric_{key}", key in report["module_b_editing"],
              str(report["module_b_editing"][key]))
    battery = report["module_b_editing"].get("battery") or {}
    agg = battery.get("aggregate") or {}
    check("edit_battery_sample_size", int(agg.get("n_scenarios") or 0) >= 15,
          f"n={agg.get('n_scenarios')} coverage={agg.get('instruction_coverage')} "
          f"following={agg.get('instruction_following_rate')}")
    check("edit_battery_has_expectations",
          battery.get("schema_version") == "edit_bench_v2"
          and "refusal_honesty" in agg,
          f"schema={battery.get('schema_version')} "
          f"honesty={agg.get('refusal_honesty')}")
    check("edit_battery_refusals_are_named",
          float(agg.get("refusal_honesty") or 0) >= 0.9,
          f"{agg.get('n_refused')} refusals ({agg.get('refusal_honesty')})")
    check("edit_battery_no_silent_noop",
          int(agg.get("n_no_change") or 0) == 0,
          f"silent no-ops={agg.get('n_no_change')}, "
          f"no-effect reported={agg.get('n_no_effect')}")
    check("edit_battery_localised",
          float(agg.get("mean_target_vs_rest") or 0) >= 2.0
          and float(agg.get("mean_non_target_mpjpe_norm") or 1) <= 0.05,
          f"target/rest={agg.get('mean_target_vs_rest')} "
          f"non-target={agg.get('mean_non_target_mpjpe_norm')}")
    rows = []
    bat_src = Path(str(battery.get("source") or ""))
    if bat_src.is_file():
        rows = json.loads(bat_src.read_text(encoding="utf-8-sig")).get("rows") or []
    check("edit_battery_covers_families",
          len({r.get("family") for r in rows if r.get("family")}) >= 8,
          str(sorted({str(r.get("family")) for r in rows if r.get("family")})))
    check("caption_metric_glove",
          "glove_cosine" in report["module_c_caption"]["aggregate"],
          str(report["module_c_caption"]["aggregate"]["glove_cosine"]))

    # --- a real episode: the loop must have gone through (b) --------------
    ui_runs = [d for d in (REPO / "outputs" / "session" / "ui_runs").glob("*UI_AGENT*")
               if (d / "agent_run.json").is_file()]
    ui_runs.sort(key=lambda d: d.stat().st_mtime)
    ep_dir = ui_runs[-1] if ui_runs else None
    check("have_a_web_replay", ep_dir is not None, str(ep_dir))
    if ep_dir is None:
        out = bench / "contract_checks.json"
        out.write_text(json.dumps({"checks": CHECKS,
                                   "passed": sum(c["ok"] for c in CHECKS),
                                   "n": len(CHECKS)},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    ep = json.loads((ep_dir / "agent_run.json").read_text(encoding="utf-8-sig"))
    actions = [s["action"] for s in ep["steps"]]
    # Order, not an exact script: the controller may loop, restore, or finish
    # without an edit when the first draft already meets the goal.
    ordered = (actions[0] == "generate_motion"
               and "describe_motion" in actions
               and actions[-1] == "finish"
               and actions.index("check_requirement") > actions.index("describe_motion")
               and all(
                   "describe_motion" in actions[a + 1:]
                   for a, act in enumerate(actions) if act == "edit_motion"
               ))
    check("episode_follows_the_mandatory_order", ordered, " → ".join(actions))
    desc = [s for s in ep["steps"] if s["action"] == "describe_motion"]
    check("caption_compared_with_goal",
          bool(desc) and all("similarity" in (s["observation"] or {}) for s in desc),
          str([round((s["observation"]["similarity"] or {}).get("glove_cosine", 0), 3)
               for s in desc]))
    check("two_captions_and_a_verdict",
          bool(desc) and all({"rule_caption", "caption_verdict"} <= set(s["observation"] or {})
                             for s in desc),
          str(desc[0]["observation"].get("caption_verdict", {}).get("level")) if desc else "")
    check("rule_caption_names_a_measurement",
          bool(desc) and any(
              any(w in str(s["observation"].get("rule_caption") or "")
                  for w in ("arm", "hand", "sitting", "turning", "walking", "foot"))
              for s in desc),
          str((desc[0]["observation"] or {}).get("rule_caption")) if desc else "")
    edits = [s for s in ep["steps"] if s["action"] == "edit_motion"]
    check("edit_text_written_by_controller",
          bool(edits) and all(s["observation"].get("instruction") for s in edits),
          str([s["observation"].get("instruction") for s in edits]))
    check("edit_followed_the_instruction",
          bool(edits) and all(
              bool((s["observation"].get("instruction_following") or {}).get("followed"))
              for s in edits if s["observation"].get("ok")),
          str([bool((s["observation"].get("instruction_following") or {}).get("followed"))
               for s in edits]))
    check("loop_stopped_when_measured",
          ep["summary"]["satisfied"] and ep["summary"]["agreement"],
          f"checker={ep['summary']['checker_satisfied']}")

    # --- no episode may end on a clip worse than the best it measured -----
    worst = [
        (r["goal"], r.get("final_gap"), r.get("best_gap_seen"))
        for r in loop["rows"]
        if r.get("best_gap_seen") is not None and r.get("final_gap") is not None
        and float(r["final_gap"]) > float(r["best_gap_seen"]) + 1e-6
    ]
    check("final_clip_is_never_worse_than_best_seen", not worst, str(worst[:3]))

    # --- screenshots delivered -------------------------------------------
    shots = sorted((Path.home() / "Downloads").glob("agentic-*.png"))
    check("screenshots_in_downloads", len(shots) >= 10, f"{len(shots)} files")

    report_out = {"checks": CHECKS, "passed": sum(c["ok"] for c in CHECKS),
                  "n": len(CHECKS)}
    out = bench / "contract_checks.json"
    out.write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{report_out['passed']}/{report_out['n']} checks passed — {out}")
    return 0 if report_out["passed"] == report_out["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
