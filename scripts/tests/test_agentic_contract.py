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


def main() -> int:
    import tools
    from tools import TOOL_SPECS, MotionTools

    names = {t["name"] for t in TOOL_SPECS}
    # --- (ii) the loop is made of tools the controller chooses -------------
    check("tool_a_generate", "generate_motion" in names)
    check("tool_b_edit", "edit_motion" in names)
    check("tool_c_describe", "describe_motion" in names)
    check("tool_check_requirement", "check_requirement" in names)
    check("tool_finish", "finish" in names)

    src = (REPO / "scripts" / "agent" / "agent.py").read_text(encoding="utf-8")
    check("mandatory_order_in_prompt", "MANDATORY ORDER" in src
          and "describe_motion before judging" in src)
    check("describe_enforced_by_code",
          "describe the new clip first" in src)
    check("check_before_finish_enforced",
          "measure the requirement before finishing" in src)
    check("loop_capped", "for step in range(1, max_steps + 1)" in src)

    # free-form text reaches the controller unchanged
    check("free_text_goal", "USER GOAL: {goal}" in src)

    # --- (i) metrics come from measurements, per module -------------------
    from metrics import kinematic_summary, load_joints, text_similarity

    clip = next((REPO / "outputs" / "agent_bench" / "_v3_final7").glob(
        "ep01_*/step01_generate/03_edited.npy"), None)
    check("have_a_clip_to_measure", clip is not None, str(clip))
    if clip is not None:
        m = kinematic_summary(load_joints(clip))
        needed = {"root_travel", "pelvis_drop", "yaw_span_deg", "arm_raise_right",
                  "jump_rise", "ground_penetration"}
        check("metrics_are_kinematic", needed <= set(m),
              f"{len(m)} measured attributes, e.g. root_travel={m['root_travel']}")
    sim = text_similarity("a person raises the right arm",
                          "a person raises their right arm high")
    check("caption_similarity_is_numeric",
          {"glove_cosine", "token_jaccard", "token_recall"} <= set(sim),
          json.dumps(sim))

    runs = sorted((REPO / "outputs" / "agent_bench").glob("_v3_final7"))
    report = json.loads((runs[0] / "report.json").read_text(encoding="utf-8-sig"))
    check("report_has_four_sections",
          {"module_a_generation", "module_b_editing", "module_c_caption", "loop"}
          <= set(report))
    check("no_saturating_score_field",
          "score" not in {k for k in report if k != "wall_seconds"},
          ", ".join(k for k in report))
    loop = report["loop"]
    for key in ("success_rate", "agreement_rate", "mean_edits", "mean_steps",
                "n_drafts_rejected"):
        check(f"loop_metric_{key}", key in loop, str(loop.get(key)))
    for key in ("r@1", "r@3", "mean_cosine"):
        check(f"gen_metric_{key}", key in report["module_a_generation"]["retrieval"],
              str(report["module_a_generation"]["retrieval"][key]))
    for key in ("instruction_following_rate", "instruction_coverage",
                "mean_non_target_mpjpe_norm"):
        check(f"edit_metric_{key}", key in report["module_b_editing"],
              str(report["module_b_editing"][key]))
    check("caption_metric_glove",
          "glove_cosine" in report["module_c_caption"]["aggregate"],
          str(report["module_c_caption"]["aggregate"]["glove_cosine"]))

    # --- a real episode: the loop must have gone through (b) --------------
    ep_dir = REPO / "outputs" / "session" / "ui_runs" / "70c66979_UI_AGENT_20261115-213540"
    ep = json.loads((ep_dir / "agent_run.json").read_text(encoding="utf-8-sig"))
    actions = [s["action"] for s in ep["steps"]]
    check("episode_order_a_c_b_c_check_finish",
          actions == ["generate_motion", "describe_motion", "edit_motion",
                      "describe_motion", "check_requirement", "finish"],
          " → ".join(actions))
    desc = [s for s in ep["steps"] if s["action"] == "describe_motion"]
    check("caption_compared_with_goal",
          all("similarity" in (s["observation"] or {}) for s in desc),
          str([round((s["observation"]["similarity"] or {}).get("glove_cosine", 0), 3)
               for s in desc]))
    edit_step = next(s for s in ep["steps"] if s["action"] == "edit_motion")
    check("edit_text_written_by_controller",
          bool(edit_step["observation"].get("instruction")),
          edit_step["observation"]["instruction"])
    check("edit_followed_the_instruction",
          bool((edit_step["observation"].get("instruction_following") or {}).get("followed")))
    check("loop_stopped_when_measured",
          ep["summary"]["satisfied"] and ep["summary"]["agreement"],
          f"checker={ep['summary']['checker_satisfied']}")

    # --- screenshots delivered -------------------------------------------
    shots = sorted((Path.home() / "Downloads").glob("agentic-0*.png"))
    check("screenshots_in_downloads", len(shots) >= 6, f"{len(shots)} files")

    report_out = {"checks": CHECKS, "passed": sum(c["ok"] for c in CHECKS),
                  "n": len(CHECKS)}
    out = runs[0] / "contract_checks.json"
    out.write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{report_out['passed']}/{report_out['n']} checks passed — {out}")
    return 0 if report_out["passed"] == report_out["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
