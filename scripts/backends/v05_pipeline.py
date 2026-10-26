"""
v0.5 session pipeline (Sept architecture):

  generate:
    user prompt → Intent Schema → module plan → MotionEditPlan
      → MLD generate (agent mld_text) → bridge → MotionsFix edit → review

  revise:
    user feedback → Intent(revise) → modules_to_rerun only
      → (optional closer MotionsFix source) → MotionsFix edit → review
    Frozen: bg / camera / appearance; skip generate+bridge unless plan asks.

Agent understands the prompt *before* the coder runs MotionsFix.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = Path(__file__).resolve().parent
DEFAULT_MLD_PY = Path(os.environ.get(
    "MLD_PYTHON",
    str(Path.home() / "miniconda3" / "envs" / "mld" / "python.exe"),
))
DEFAULT_MF_PY = Path(os.environ.get(
    "MOTIONFIX_PYTHON",
    str(Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"),
))


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _pause(label: str, paths: List[Path], skip: bool) -> None:
    print("\n" + "=" * 60)
    print(f"[review] {label}")
    for p in paths:
        print(f"  - {p}")
    if skip:
        print("[review] --skip-review set; continuing")
        return
    try:
        input("Press Enter to continue (Ctrl+C to abort)... ")
    except EOFError:
        print("[review] no TTY; continuing")


def parse_args():
    p = argparse.ArgumentParser(
        description="v0.5 session: Intent → generate → revise MotionsFix"
    )
    p.add_argument(
        "--mode",
        choices=["generate", "revise"],
        default="generate",
        help="generate = full pipe; revise = modules_to_rerun only",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="User prompt (generate description or revise feedback). "
        "Preferred over --text/--edit-text.",
    )
    p.add_argument("--text", type=str, default=None, help="Legacy MLD generation prompt")
    p.add_argument(
        "--edit-text",
        type=str,
        default=None,
        help="Legacy edit instruction (default: --prompt or --text)",
    )
    p.add_argument(
        "--previous-run",
        type=Path,
        default=None,
        help="Prior run_dir for --mode revise (reuses summary / artifacts)",
    )
    p.add_argument("--length", type=int, default=60)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--joints", type=Path, default=None, help="Skip MLD; use existing joints npy")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-bridge", action="store_true", help="Use --source-npy directly for edit")
    p.add_argument("--source-npy", type=Path, default=None, help="Existing MotionFix source npy")
    p.add_argument("--skip-review", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--bridge-iters", type=int, default=40)
    p.add_argument("--bridge-max-frames", type=int, default=None)
    p.add_argument(
        "--bridge-target-len",
        type=int,
        default=0,
        help="Resample bridged motion to this length (0=keep native T)",
    )
    p.add_argument(
        "--edit-steps",
        type=int,
        default=0,
        help="MotionFix sample steps (0 = checkpoint train timesteps ≈300)",
    )
    p.add_argument("--gd-text", type=float, default=2.5, help="MotionFix text CFG")
    p.add_argument(
        "--gd-motion",
        type=float,
        default=None,
        help="MotionFix source-motion CFG (default: auto by edit text)",
    )
    p.add_argument(
        "--body-amp",
        type=float,
        default=1.0,
        help="Bridged-source body amp (1.0=off)",
    )
    p.add_argument(
        "--adapt-bridged",
        action="store_true",
        default=False,
        help="Enable body-amp adapt on bridged MLD sources",
    )
    p.add_argument(
        "--init-from",
        choices=["noise", "source", "auto"],
        default="auto",
        help="MotionFix diffusion init",
    )
    p.add_argument(
        "--sdedit-strength",
        type=float,
        default=None,
        help="SDEdit noise fraction (default: auto in motionfix_edit)",
    )
    p.add_argument(
        "--edit-mode",
        choices=["auto", "diffusion", "speed", "kinematic"],
        default="auto",
    )
    p.add_argument(
        "--edit-source-id",
        type=str,
        default=None,
        help="Force MotionsFix test-set id (overrides agent preferred_source_id)",
    )
    p.add_argument("--mld-python", type=Path, default=DEFAULT_MLD_PY)
    p.add_argument("--motionfix-python", type=Path, default=DEFAULT_MF_PY)
    p.add_argument(
        "--use-intent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Intent Schema → MotionEditPlan before MotionsFix (default: on)",
    )
    p.add_argument(
        "--source-hint",
        type=str,
        default="bridged_walk",
        help="What the edit source looks like (for span/policy)",
    )
    p.add_argument(
        "--intent-json",
        type=Path,
        default=None,
        help="Reuse an existing agent bundle / MotionEditPlan",
    )
    p.add_argument(
        "--force-bridged-edit",
        action="store_true",
        help="Ignore agent use_motionfix_native; always edit bridged npy",
    )
    return p.parse_args()


def _resolve_user_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.mode == "revise":
        if args.edit_text:
            return args.edit_text
        raise SystemExit("--mode revise requires --prompt (user improvement text)")
    return args.edit_text or args.text or ""


def _load_previous(prev: Path) -> Dict[str, Any]:
    summary_path = prev / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"--previous-run missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8-sig"))


def main():
    args = parse_args()
    cwd0 = Path.cwd()
    t0 = time.time()

    def _abs(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (cwd0 / p).resolve()

    user_prompt = _resolve_user_prompt(args)
    if not user_prompt and args.mode == "generate":
        raise SystemExit("Provide --prompt or --text for generate")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _abs(args.run_dir) or (REPO_ROOT / "outputs" / "v05" / f"run_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    prev_dir = _abs(args.previous_run)
    prev_summary: Dict[str, Any] = {}
    if args.mode == "revise":
        if prev_dir is None:
            raise SystemExit("--mode revise requires --previous-run")
        prev_summary = _load_previous(prev_dir)
        print(f"[v05] REVISE from {prev_dir}")

    # Legacy dual-text: MLD base vs edit. Agent may override mld_text.
    mld_text = args.text or user_prompt
    edit_text = args.edit_text or user_prompt
    mld_py = str(_abs(args.mld_python))
    mf_py = str(_abs(args.motionfix_python))
    viz_py = str(BACKENDS / "visualize_motion.py")
    intent_py = str(sys.executable)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "prompt": user_prompt,
        "text": mld_text,
        "edit_text": edit_text,
        "use_intent": bool(args.use_intent),
        "previous_run": str(prev_dir) if prev_dir else None,
        "steps": {},
    }
    print(f"[v05] mode={args.mode} run_dir={run_dir}")

    # --- 0) Agent: Intent Schema → module plan → MotionEditPlan ---
    intent_bundle_path: Optional[Path] = _abs(args.intent_json)
    intent_plan_path: Optional[Path] = None
    plan_preview: Dict[str, Any] = {}
    if args.use_intent:
        if intent_bundle_path is None:
            intent_bundle_path = run_dir / "00_agent_bundle.json"
            intent_plan_path = run_dir / "00_motion_edit_plan.json"
            cmd = [
                intent_py,
                str(REPO_ROOT / "scripts" / "intent" / "plan_motion_edit.py"),
                "--prompt",
                user_prompt,
                "--mode",
                args.mode,
                "--task-id",
                f"V05_{stamp}",
                "--source-hint",
                args.source_hint,
                "--out",
                str(intent_bundle_path),
                "--out-plan",
                str(intent_plan_path),
            ]
            if args.mode == "revise" and prev_summary.get("task_id"):
                cmd += ["--previous-task-id", str(prev_summary["task_id"])]
            elif args.mode == "revise":
                cmd += ["--previous-task-id", f"V05_PREV"]
            _run(cmd, cwd=REPO_ROOT)
        else:
            raw = json.loads(intent_bundle_path.read_text(encoding="utf-8-sig"))
            if "motion_edit_plan" in raw:
                intent_plan_path = run_dir / "00_motion_edit_plan.json"
                intent_plan_path.write_text(
                    json.dumps(raw["motion_edit_plan"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                intent_plan_path = intent_bundle_path
        plan_preview = json.loads(intent_plan_path.read_text(encoding="utf-8-sig"))
        summary["task_id"] = plan_preview.get("from_intent_task_id") or f"V05_{stamp}"
        summary["steps"]["intent"] = str(intent_bundle_path)
        summary["motion_edit_plan"] = {
            "primary_action": plan_preview.get("primary_action"),
            "span": plan_preview.get("span"),
            "source_policy": plan_preview.get("source_policy"),
            "edit_source": plan_preview.get("edit_source"),
            "preferred_source_id": plan_preview.get("preferred_source_id"),
            "mld_text": plan_preview.get("mld_text"),
            "mf_text": plan_preview.get("mf_text"),
            "modules": plan_preview.get("modules"),
            "warning": plan_preview.get("warning"),
        }
        if plan_preview.get("mld_text") and args.mode == "generate" and not args.text:
            mld_text = str(plan_preview["mld_text"])
            summary["text"] = mld_text
        print(
            f"[v05] AGENT action={plan_preview.get('primary_action')} "
            f"policy={plan_preview.get('source_policy')} "
            f"edit_source={plan_preview.get('edit_source')} "
            f"mf_id={plan_preview.get('preferred_source_id')} "
            f"mf_text={plan_preview.get('mf_text')!r}"
        )
        if plan_preview.get("warning"):
            print(f"[v05] AGENT: {plan_preview['warning']}")
        if args.mode == "revise" and plan_preview.get("revision"):
            rev = plan_preview["revision"]
            print(
                f"[v05] REVISE targets={rev.get('targets')} "
                f"rerun={rev.get('modules_to_rerun')} "
                f"frozen={rev.get('modules_frozen')}"
            )
        _pause(
            "Review Intent / MotionEditPlan",
            [intent_bundle_path, intent_plan_path],
            args.skip_review,
        )

    use_native = (
        not args.force_bridged_edit
        and (
            args.edit_source_id
            or (
                plan_preview.get("edit_source") == "motionfix_id"
                and plan_preview.get("preferred_source_id")
            )
            or plan_preview.get("source_policy") == "use_motionfix_native"
        )
    )
    native_id = args.edit_source_id or plan_preview.get("preferred_source_id")

    # --- 1) MLD generate (skip on revise unless forced) ---
    skip_gen = (
        args.skip_generate
        or args.joints is not None
        or args.mode == "revise"
        or use_native
    )
    joints_path = _abs(args.joints)
    if skip_gen:
        if args.mode == "revise" and prev_dir is not None:
            cand = prev_dir / "01_mld_joints.npy"
            if cand.is_file():
                joints_path = cand
                summary["steps"]["generate"] = f"frozen:{cand}"
                print(f"[v05] revise: freeze generate → {cand}")
        elif joints_path is None and not use_native:
            if args.mode == "generate" and not args.skip_generate:
                pass  # will generate below if not use_native
            elif not use_native:
                raise SystemExit("--skip-generate requires --joints (or revise --previous-run)")
        if joints_path is not None:
            print(f"[v05] skip generate; using {joints_path}")
    if not skip_gen:
        joints_path = run_dir / "01_mld_joints.npy"
        _run(
            [
                mld_py,
                str(BACKENDS / "mld_generate.py"),
                "--text",
                mld_text,
                "--length",
                str(args.length),
                "--out",
                str(joints_path),
            ],
            cwd=REPO_ROOT,
        )
        summary["steps"]["generate"] = str(joints_path)
        preview_mld = run_dir / "01_mld_preview.gif"
        if not args.skip_viz:
            _run(
                [
                    mld_py,
                    viz_py,
                    "--input",
                    str(joints_path),
                    "--out",
                    str(preview_mld),
                    "--title",
                    f"MLD: {mld_text}",
                ],
                cwd=REPO_ROOT,
            )
        _pause("Review MLD generation", [joints_path, preview_mld], args.skip_review)
    elif use_native and args.mode == "generate":
        print(
            f"[v05] AGENT skip MLD/bridge for edit; MotionsFix-native id={native_id} "
            "(relative edit stays in-distribution)"
        )
        summary["steps"]["generate"] = f"skipped_native_edit:{native_id}"

    # --- 2) Bridge (skip on revise / native edit) ---
    source_npy = _abs(args.source_npy)
    skip_bridge = (
        args.skip_bridge
        or args.mode == "revise"
        or use_native
        or joints_path is None
    )
    if skip_bridge:
        if args.mode == "revise" and prev_dir is not None and not use_native:
            dest = run_dir / "02_motionfix_source.npy"
            edited_prev = prev_dir / "03_edited.npy"
            src_prev = prev_dir / "02_motionfix_source.npy"
            if edited_prev.is_file():
                shutil.copy2(edited_prev, dest)
                source_npy = dest
                print(f"[v05] revise: seed source from previous edited → {dest}")
            elif src_prev.is_file():
                shutil.copy2(src_prev, dest)
                source_npy = dest
                print(f"[v05] revise: freeze bridge source → {dest}")
            summary["steps"]["bridge"] = f"frozen:{source_npy}"
        elif source_npy is None and not use_native:
            raise SystemExit("--skip-bridge requires --source-npy (or revise --previous-run)")
        elif source_npy is not None:
            print(f"[v05] skip bridge; using {source_npy}")
    else:
        source_npy = run_dir / "02_motionfix_source.npy"
        bridge_cmd = [
            mld_py,
            str(BACKENDS / "mld_to_motionfix.py"),
            "--joints",
            str(joints_path),
            "--out",
            str(source_npy),
            "--iters",
            str(args.bridge_iters),
            "--target-len",
            str(args.bridge_target_len),
        ]
        if args.bridge_max_frames:
            bridge_cmd += ["--max-frames", str(args.bridge_max_frames)]
        _run(bridge_cmd, cwd=REPO_ROOT)
        _run(
            [
                mf_py,
                str(BACKENDS / "motionfix_facefront.py"),
                "--in",
                str(source_npy),
                "--out",
                str(source_npy),
            ],
            cwd=REPO_ROOT,
        )
        summary["steps"]["bridge"] = str(source_npy)
        preview_src = run_dir / "02_source_preview.gif"
        if not args.skip_viz:
            _run(
                [
                    mld_py,
                    viz_py,
                    "--input",
                    str(source_npy),
                    "--out",
                    str(preview_src),
                    "--title",
                    "Bridge source (fitted)",
                ],
                cwd=REPO_ROOT,
            )
        _pause("Review bridged source", [source_npy, preview_src], args.skip_review)

    # --- 3) MotionsFix edit (coder) ---
    edited = run_dir / "03_edited.npy"
    edit_cmd = [
        mf_py,
        str(BACKENDS / "motionfix_edit.py"),
        "--out",
        str(edited),
        "--steps",
        str(args.edit_steps),
        "--gd-text",
        str(args.gd_text),
        "--init-from",
        args.init_from,
        "--edit-mode",
        args.edit_mode,
        "--body-amp",
        str(args.body_amp),
    ]
    if intent_plan_path is not None:
        edit_cmd += ["--intent-json", str(intent_plan_path)]
    else:
        edit_cmd += ["--text", edit_text]
    if args.gd_motion is not None:
        edit_cmd += ["--gd-motion", str(args.gd_motion)]
    if args.sdedit_strength is not None:
        edit_cmd += ["--sdedit-strength", str(args.sdedit_strength)]
    if args.adapt_bridged and args.body_amp != 1.0:
        edit_cmd.append("--adapt-bridged")
    else:
        edit_cmd.append("--no-adapt-bridged")

    if use_native and native_id:
        print(
            f"[v05] edit uses MotionsFix source-id={native_id} "
            f"(policy={plan_preview.get('source_policy')})"
        )
        edit_cmd += ["--source-id", str(native_id)]
        summary["steps"]["edit_source"] = f"motionfix_id:{native_id}"
    else:
        if source_npy is None:
            raise SystemExit("No bridged source for edit")
        edit_cmd += ["--source-npy", str(source_npy)]
        summary["steps"]["edit_source"] = f"bridged:{source_npy}"

    _run(edit_cmd, cwd=REPO_ROOT)
    summary["steps"]["edit"] = str(edited)

    preview_title = (
        (plan_preview.get("mf_text") if plan_preview else None) or edit_text or user_prompt
    )
    preview_edit = run_dir / "03_edited_preview.gif"
    if not args.skip_viz:
        _run(
            [
                mf_py,
                viz_py,
                "--input",
                str(edited),
                "--out",
                str(preview_edit),
                "--title",
                f"{args.mode}: {preview_title}",
            ],
            cwd=REPO_ROOT,
        )
    _pause("Review edited motion", [edited, preview_edit], args.skip_review)

    # --- 4) Quality gate (UI should hide hard failures) ---
    # Prefer motionfix env (has numpy); fall back to host.
    quality_path = run_dir / "04_quality.json"
    if edited.is_file():
        q_py = mf_py
        q_cmd = [
            q_py,
            str(REPO_ROOT / "scripts" / "session" / "quality_gate.py"),
            "--edited",
            str(edited),
            "--out",
            str(quality_path),
        ]
        action = (plan_preview or {}).get("primary_action")
        if action:
            q_cmd += ["--action", str(action)]
        try:
            _run(q_cmd, cwd=REPO_ROOT)
            if quality_path.is_file():
                q = json.loads(quality_path.read_text(encoding="utf-8-sig"))
                summary["quality"] = {
                    "passed": q.get("passed"),
                    "score": q.get("score"),
                    "hard_failures": q.get("hard_failures"),
                    "path": str(quality_path),
                }
                summary["steps"]["quality"] = str(quality_path)
        except SystemExit as exc:
            print(f"[v05] quality_gate non-fatal: {exc}")

    summary["total_seconds"] = round(time.time() - t0, 3)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[v05] done mode={args.mode} in {summary['total_seconds']}s")
    print(f"[v05] summary {summary_path}")
    for k, v in summary["steps"].items():
        print(f"  {k}: {v}")
    if summary.get("quality"):
        print(
            f"[v05] quality passed={summary['quality'].get('passed')} "
            f"score={summary['quality'].get('score')}"
        )
    print(f"[v05] next revise: --mode revise --previous-run {run_dir} --prompt \"...\"")


if __name__ == "__main__":
    main()
