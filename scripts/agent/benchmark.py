"""
Benchmark: free-form goal → agent → module + loop metrics.

Runs a small battery of free-form requests through the LLM tool loop, then
reports the quantitative metrics the advisor asked for:

  module (a) generation   text–caption embedding similarity, caption-mediated
                          R@1/R@3 over the batch, kinematic plausibility,
                          batch diversity
  module (b) editing      instruction-following rate, target change,
                          non-target MPJPE, smoothness change
  module (c) captioning   BLEU-1..4, ROUGE-L, CIDEr-D on held-out HumanML3D
                          clips with ground-truth captions
  loop                    success rate, edits to success, alignment gain,
                          tool calls, wall time, tokens,
                          instruction coverage (how many edit instructions were
                          inside the tool's ability at all) and agent/checker
                          agreement (was the controller calibrated?)

  python scripts/agent/benchmark.py --n 4 --max-edits 2 --out outputs/agent_bench
  python scripts/agent/benchmark.py --goals-file test-samples/agent/goals_20.json --max-edits 2
  python scripts/agent/benchmark.py --no-llm --generator dataset   # baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts" / "agent", REPO / "scripts" / "session"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent import run_agent  # noqa: E402
from metrics import caption_benchmark, cosine, embed, load_joints, kinematic_summary  # noqa: E402

DEFAULT_GOALS = [
    "a person raises the right arm high above the head",
    "a person walks forward and waves with the right hand",
    "a person sits down on the floor",
    "a person turns around in place",
]


def _retrieval(goals: List[str], captions: List[str]) -> Dict[str, Any]:
    """Caption-mediated text→motion retrieval (the R-Precision idea)."""
    n = len(goals)
    if n < 2:
        return {"n": n, "r@1": None, "r@3": None, "mean_cosine": None}
    sim = np.zeros((n, n))
    for i, g in enumerate(goals):
        for j, c in enumerate(captions):
            sim[i, j] = cosine(embed(g), embed(c))
    ranks = (-sim).argsort(axis=1)
    r1 = float(np.mean([ranks[i, 0] == i for i in range(n)]))
    r3 = float(np.mean([i in ranks[i, :3] for i in range(n)]))
    return {
        "n": n,
        "r@1": round(r1, 3),
        "r@3": round(r3, 3),
        "mean_cosine": round(float(np.diag(sim).mean()), 4),
        "matrix": [[round(float(x), 3) for x in row] for row in sim],
    }


def _diversity(clips: List[np.ndarray], frames: int = 32) -> Dict[str, float]:
    """Mean pairwise distance of root-relative, scale-normalised pose features."""
    feats = []
    for j in clips:
        idx = np.linspace(0, len(j) - 1, frames).round().astype(int)
        rel = j[idx] - j[idx, 0:1]
        scale = float(np.linalg.norm(j[:, 15] - j[:, 0], axis=-1).mean()) or 1.0
        f = (rel / scale).reshape(-1)
        feats.append(f / (np.linalg.norm(f) or 1.0))
    if len(feats) < 2:
        return {"mean_pairwise": None}
    d = []
    for i in range(len(feats)):
        for k in range(i + 1, len(feats)):
            d.append(float(1.0 - np.dot(feats[i], feats[k])))
    return {"mean_pairwise": round(float(np.mean(d)), 4)}


def run_benchmark(goals: List[str], *, max_edits: int, generator: str, use_llm: bool,
                  out_dir: Path, skip_caption_bench: bool = False,
                  caption_n: int = 8) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    episodes: List[Dict[str, Any]] = []
    initial_captions: List[str] = []
    initial_clips: List[np.ndarray] = []
    edit_rows: List[Dict[str, Any]] = []      # edits applied to the clip
    attempt_rows: List[Dict[str, Any]] = []   # every edit_motion call
    t0 = time.time()

    for i, goal in enumerate(goals, 1):
        print(f"[benchmark] {i}/{len(goals)}: {goal}", flush=True)
        ep = run_agent(
            goal,
            out_dir=out_dir / f"ep{i:02d}_{stamp}",
            max_edits=max_edits,
            generator=generator,
            use_llm=use_llm,
            skip_viz=False,
        )
        episodes.append(ep)

        # Initial clip = first generate step; its first caption comes from the
        # first describe step after it.
        gen_step = next((s for s in ep["steps"] if s["action"] == "generate_motion"), None)
        first_desc = next(
            (s for s in ep["steps"] if s["action"] == "describe_motion"), None
        )
        initial_captions.append(str((first_desc or {}).get("observation", {}).get("caption") or ""))
        run_dir = Path(str((gen_step or {}).get("observation", {}).get("run_dir") or ""))
        if (run_dir / "03_edited.npy").is_file():
            initial_clips.append(load_joints(run_dir / "03_edited.npy"))

        for st in ep["steps"]:
            if st["action"] != "edit_motion":
                continue
            obs = st["observation"] or {}
            before = obs.get("before") or {}
            after = obs.get("after") or {}
            ok = bool(obs.get("ok"))
            err = obs.get("error")
            if ok:
                status = "applied"
            elif err == "unsupported_instruction":
                status = "unsupported"
            elif obs.get("invalid"):
                status = "invalid_clip"
            else:
                status = "session_error"
            row = {
                "goal": goal,
                "instruction": obs.get("instruction") or st.get("args", {}).get("instruction"),
                "family": obs.get("family"),
                "status": status,
                "error": err,
                "invalid": obs.get("invalid") or [],
                "rolled_back": bool(obs.get("rolled_back")),
                "followed": bool((obs.get("instruction_following") or {}).get("followed")),
                "verifiable": (obs.get("instruction_following") or {}).get("verifiable", True),
                "no_change": bool(obs.get("no_change")),
                "frames_ratio": obs.get("frames_ratio"),
                "target_vs_rest": (obs.get("preservation") or {}).get("target_vs_rest"),
                "non_target_mpjpe_norm": (obs.get("preservation") or {}).get("non_target_mpjpe_norm"),
                "jerk_before": before.get("jerk"),
                "jerk_after": after.get("jerk"),
                "penetration_after": after.get("ground_penetration"),
                "arm_raise_delta": round(
                    float(after.get("arm_raise_right") or 0) - float(before.get("arm_raise_right") or 0), 3
                ),
            }
            attempt_rows.append(row)
            if ok:
                edit_rows.append(row)

    # ---- module (a) ----
    gen_rows = []
    for ep in episodes:
        st = next((s for s in ep["steps"] if s["action"] == "describe_motion"), None)
        obs = (st or {}).get("observation") or {}
        m = obs.get("metrics") or {}
        gen_rows.append({
            "goal": ep["goal"],
            "caption": obs.get("caption"),
            "similarity": obs.get("similarity"),
            "metrics": m,
        })
    plaus = {}
    if gen_rows:
        for key in ("bone_cv", "foot_slide", "jerk", "ground_penetration"):
            vals = [float((r["metrics"] or {}).get(key) or 0) for r in gen_rows]
            plaus[key] = round(float(np.mean(vals)), 5)

    # ---- loop ----
    loop_rows = []
    for ep in episodes:
        # Only judgments made about a described clip count as alignment points.
        align = [s.get("assessment", {}).get("alignment") for s in ep["steps"]
                 if s.get("action") in ("edit_motion", "finish")
                 and isinstance(s.get("assessment"), dict)]
        align = [a for a in align if isinstance(a, (int, float))]
        loop_rows.append({
            "goal": ep["goal"],
            "satisfied": ep["summary"]["satisfied"],
            "agent_claim_satisfied": ep["summary"].get("agent_claim_satisfied"),
            "checker_satisfied": ep["summary"].get("checker_satisfied"),
            "agreement": ep["summary"].get("agreement"),
            "check_calls": ep["summary"].get("check_calls"),
            "edits": ep["summary"]["edits"],
            "edit_attempts": sum(1 for s in ep["steps"] if s["action"] == "edit_motion"),
            "steps": len(ep["steps"]),
            "tool_calls": ep["summary"]["tool_calls"],
            "seconds": ep["summary"]["seconds"],
            "tokens": ep["summary"].get("tokens", {}).get("total_tokens", 0),
            "alignment_trajectory": align,
            "final_alignment": ep["summary"].get("final_alignment"),
            "final_caption": ep.get("final_caption"),
            "episode_dir": ep["episode_dir"],
        })
    gains = []
    for row in loop_rows:
        tr = row["alignment_trajectory"]
        if len(tr) >= 2:
            gains.append(float(tr[-1]) - float(tr[0]))

    report: Dict[str, Any] = {
        "schema_version": "agent_benchmark_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "goals": goals,
            "max_edits": max_edits,
            "generator": generator,
            "policy": "llm" if use_llm else "rule",
        },
        "module_a_generation": {
            "rows": gen_rows,
            "retrieval": _retrieval(goals, initial_captions),
            "plausibility_mean": plaus,
            "diversity": _diversity(initial_clips),
        },
        "module_b_editing": {
            "edits": edit_rows,
            "attempts": attempt_rows,
            "n_attempts": len(attempt_rows),
            "n": len(edit_rows),
            "n_rolled_back": sum(1 for r in attempt_rows if r["rolled_back"]),
            "n_invalid_clip": sum(1 for r in attempt_rows if r["status"] == "invalid_clip"),
            "n_session_error": sum(1 for r in attempt_rows if r["status"] == "session_error"),
            "n_unsupported": sum(1 for r in attempt_rows if r["status"] == "unsupported"),
            "instruction_coverage": round(
                float(sum(1 for r in attempt_rows if r["status"] != "unsupported"))
                / len(attempt_rows), 3
            ) if attempt_rows else None,
            "instruction_following_rate": round(
                float(np.mean([r["followed"] for r in edit_rows if r.get("verifiable", True)])), 3
            ) if any(r.get("verifiable", True) for r in edit_rows) else None,
            "n_unverifiable_edits": sum(1 for r in edit_rows if not r.get("verifiable", True)),
            "no_change_rate": round(
                float(np.mean([r["no_change"] for r in edit_rows])), 3
            ) if edit_rows else None,
            "mean_frame_retention": round(
                float(np.mean([float(r["frames_ratio"]) for r in edit_rows
                               if r.get("frames_ratio") is not None])), 3
            ) if any(r.get("frames_ratio") is not None for r in edit_rows) else None,
            "n_truncating_edits": sum(
                1 for r in edit_rows
                if r.get("frames_ratio") is not None and float(r["frames_ratio"]) < 0.70
                and (r.get("family") or "") not in ("trim_trailing", "sit_remove_preamble")
            ),
            "mean_non_target_mpjpe_norm": round(
                float(np.mean([float(r["non_target_mpjpe_norm"]) for r in edit_rows
                               if r["non_target_mpjpe_norm"] is not None])), 4
            ) if any(r["non_target_mpjpe_norm"] is not None for r in edit_rows) else None,
            "mean_target_vs_rest": round(
                float(np.mean([float(r["target_vs_rest"]) for r in edit_rows
                               if r["target_vs_rest"] is not None])), 2
            ) if any(r["target_vs_rest"] is not None for r in edit_rows) else None,
        },
        "module_c_caption": None,
        "loop": {
            "rows": loop_rows,
            "success_rate": round(float(np.mean([r["satisfied"] for r in loop_rows])), 3),
            "agreement_rate": round(
                float(np.mean([r["agreement"] for r in loop_rows if r["agreement"] is not None])), 3
            ) if any(r["agreement"] is not None for r in loop_rows) else None,
            "check_coverage": round(
                float(np.mean([bool(r["check_calls"]) for r in loop_rows])), 3
            ) if loop_rows else None,
            "mean_edits": round(float(np.mean([r["edits"] for r in loop_rows])), 2),
            "mean_edit_attempts": round(float(np.mean([r["edit_attempts"] for r in loop_rows])), 2),
            "mean_steps": round(float(np.mean([r["steps"] for r in loop_rows])), 2),
            "mean_alignment_gain": round(float(np.mean(gains)), 1) if gains else None,
            "mean_seconds": round(float(np.mean([r["seconds"] for r in loop_rows])), 1),
            "total_tokens": int(sum(r["tokens"] for r in loop_rows)),
        },
        "wall_seconds": round(time.time() - t0, 1),
    }

    if not skip_caption_bench:
        try:
            report["module_c_caption"] = caption_benchmark(caption_n)
        except Exception as exc:  # noqa: BLE001
            report["module_c_caption"] = {"error": str(exc)}

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r: Dict[str, Any]) -> str:
    a, b, c, loop = r["module_a_generation"], r["module_b_editing"], r["module_c_caption"], r["loop"]
    lines = [
        "# Agentic pipeline benchmark",
        "",
        f"- policy: **{r['config']['policy']}** · generator: {r['config']['generator']} "
        f"· goals: {len(r['config']['goals'])} · wall: {r['wall_seconds']} s",
        "",
        "## Module (a) — text→motion generation",
        "",
        f"- caption-mediated retrieval: R@1 = {a['retrieval'].get('r@1')}, "
        f"R@3 = {a['retrieval'].get('r@3')}, mean cosine = {a['retrieval'].get('mean_cosine')}",
        f"- plausibility mean: {json.dumps(a['plausibility_mean'])}",
        f"- batch diversity (mean pairwise distance): {a['diversity'].get('mean_pairwise')}",
        "",
        "| goal | M2T caption | GloVe cos | arm_raise R | travel | jerk |",
        "|---|---|---|---|---|---|",
    ]
    for row in a["rows"]:
        m = row["metrics"] or {}
        lines.append(
            f"| {row['goal']} | {row['caption']} | "
            f"{(row['similarity'] or {}).get('glove_cosine')} | "
            f"{m.get('arm_raise_right')} | {m.get('root_travel')} | {m.get('jerk')} |"
        )

    lines += [
        "",
        "## Module (b) — motion editing",
        "",
        f"- attempts: {b['n_attempts']} · applied: {b['n']} · rolled back: {b['n_rolled_back']} "
        f"(invalid clip {b['n_invalid_clip']}, session error {b['n_session_error']})",
        f"- unsupported instructions rejected before the backend: {b.get('n_unsupported')} "
        f"· instruction coverage: **{b.get('instruction_coverage')}**",
        f"- instruction-following rate (applied edits only): **{b['instruction_following_rate']}** "
        f"· no-change rate: {b.get('no_change_rate')}",
        f"- mean frame retention: {b.get('mean_frame_retention')} "
        f"· truncating edits (kept <70% frames without a trim instruction): {b.get('n_truncating_edits')}",
        f"- mean non-target MPJPE (normalised by height): {b['mean_non_target_mpjpe_norm']}",
        f"- mean target/rest change ratio: {b['mean_target_vs_rest']}",
        "",
        "| goal | instruction | family | status | followed | no change | target/rest | non-target MPJPE | jerk Δ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in b["attempts"]:
        jd = None
        if row["jerk_before"] is not None and row["jerk_after"] is not None:
            jd = round(float(row["jerk_after"]) - float(row["jerk_before"]), 4)
        status = row["status"] + (" (rollback)" if row["rolled_back"] else "")
        lines.append(
            f"| {row['goal']} | {row['instruction']} | {row.get('family')} | {status} | "
            f"{row['followed']} | "
            f"{row.get('no_change')} | {row['target_vs_rest']} | "
            f"{row['non_target_mpjpe_norm']} | {jd} |"
        )

    lines += ["", "## Module (c) — motion→text captioning", ""]
    if isinstance(c, dict) and c.get("aggregate"):
        lines += [
            "| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr-D | GloVe cos |",
            "|---|---|---|---|---|---|---|",
            "| " + " | ".join(str(c["aggregate"].get(k)) for k in
                              ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider_d", "glove_cosine")) + " |",
            "",
            f"n = {c.get('n')} held-out HumanML3D test clips with their ground-truth captions.",
        ]
    else:
        lines.append(f"caption benchmark unavailable: {c}")

    lines += [
        "",
        "## Agent loop",
        "",
        f"- success rate: **{loop['success_rate']}** · mean applied edits: {loop['mean_edits']} "
        f"· mean edit attempts: {loop['mean_edit_attempts']} "
        f"· mean steps: {loop['mean_steps']} · mean alignment gain: {loop['mean_alignment_gain']}",
        f"- check_coverage (episodes that consulted the checker): {loop.get('check_coverage')} "
        f"· agent/checker agreement: {loop.get('agreement_rate')}",
        f"- mean wall time: {loop['mean_seconds']} s · total tokens: {loop['total_tokens']}",
        "",
        "| goal | satisfied | agent claim | checker | agreement | applied edits | attempts | steps | alignment trajectory | final caption |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in loop["rows"]:
        lines.append(
            f"| {row['goal']} | {row['satisfied']} | {row.get('agent_claim_satisfied')} | "
            f"{row.get('checker_satisfied')} | {row.get('agreement')} | {row['edits']} | {row['edit_attempts']} | {row['steps']} | "
            f"{row['alignment_trajectory']} | {row['final_caption']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic pipeline benchmark")
    ap.add_argument("--goals", nargs="*", default=None)
    ap.add_argument("--goals-file", type=Path, default=None,
                    help="JSON list of goal strings (scales the retrieval battery)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-edits", type=int, default=2)
    ap.add_argument("--generator", choices=["mld", "dataset"], default="mld")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--skip-caption-bench", action="store_true")
    ap.add_argument("--caption-n", type=int, default=8,
                    help="held-out HumanML3D clips for the caption benchmark")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.goals_file:
        goals = json.loads(args.goals_file.read_text(encoding="utf-8-sig"))
        if isinstance(goals, dict):
            goals = goals.get("goals") or []
        goals = [str(g) for g in goals]
    else:
        goals = args.goals or DEFAULT_GOALS[: args.n]
    out = args.out or REPO / "outputs" / "agent_bench" / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = run_benchmark(
        goals,
        max_edits=args.max_edits,
        generator=args.generator,
        use_llm=not args.no_llm,
        out_dir=out,
        skip_caption_bench=args.skip_caption_bench,
        caption_n=args.caption_n,
    )
    print(json.dumps({k: report[k] for k in ("module_a_generation", "module_b_editing", "loop")},
                     ensure_ascii=False, indent=2)[:4000])
    print(f"\n[benchmark] report → {out / 'report.md'}")


if __name__ == "__main__":
    main()
