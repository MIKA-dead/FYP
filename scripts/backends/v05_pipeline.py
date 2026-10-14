"""
v0.5 human-in-the-loop pipeline:

  prompt → MLD generate → [review] → joints→MotionFix bridge → MotionFix edit → [review]

Uses two conda envs via subprocess:
  - mld        for generate + bridge
  - motionfix  for edit (+ pose FK viz)

Example (smoke, no pause):

  python scripts/backends/v05_pipeline.py \\
      --text "a person walks forward." \\
      --edit-text "a person walks forward slowly" \\
      --length 60 \\
      --skip-review \\
      --bridge-iters 30 \\
      --edit-steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


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
    p = argparse.ArgumentParser(description="v0.5 MLD → MotionFix supervised pipeline")
    p.add_argument("--text", type=str, required=True, help="MLD generation prompt")
    p.add_argument(
        "--edit-text",
        type=str,
        default=None,
        help="MotionFix edit instruction (default: same as --text)",
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
    p.add_argument("--edit-steps", type=int, default=50)
    p.add_argument("--gd-text", type=float, default=2.0)
    p.add_argument("--gd-motion", type=float, default=2.0)
    p.add_argument("--mld-python", type=Path, default=DEFAULT_MLD_PY)
    p.add_argument("--motionfix-python", type=Path, default=DEFAULT_MF_PY)
    return p.parse_args()


def main():
    args = parse_args()
    cwd0 = Path.cwd()
    t0 = time.time()

    def _abs(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (cwd0 / p).resolve()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _abs(args.run_dir) or (REPO_ROOT / "outputs" / "v05" / f"run_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    edit_text = args.edit_text or args.text
    mld_py = str(_abs(args.mld_python))
    mf_py = str(_abs(args.motionfix_python))
    viz_py = str(BACKENDS / "visualize_motion.py")

    summary = {
        "run_dir": str(run_dir),
        "text": args.text,
        "edit_text": edit_text,
        "steps": {},
    }
    print(f"[v05] run_dir={run_dir}")

    # --- 1) MLD generate ---
    joints_path = _abs(args.joints)
    if args.skip_generate or joints_path is not None:
        if joints_path is None:
            raise SystemExit("--skip-generate requires --joints")
        print(f"[v05] skip generate; using {joints_path}")
    else:
        joints_path = run_dir / "01_mld_joints.npy"
        _run(
            [
                mld_py,
                str(BACKENDS / "mld_generate.py"),
                "--text",
                args.text,
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
                f"MLD: {args.text}",
            ],
            cwd=REPO_ROOT,
        )
    _pause("Review MLD generation", [joints_path, preview_mld], args.skip_review)

    # --- 2) Bridge ---
    source_npy = _abs(args.source_npy)
    if args.skip_bridge:
        if source_npy is None:
            raise SystemExit("--skip-bridge requires --source-npy")
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
        ]
        if args.bridge_max_frames:
            bridge_cmd += ["--max-frames", str(args.bridge_max_frames)]
        _run(bridge_cmd, cwd=REPO_ROOT)
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

    # --- 3) MotionFix edit ---
    edited = run_dir / "03_edited.npy"
    _run(
        [
            mf_py,
            str(BACKENDS / "motionfix_edit.py"),
            "--source-npy",
            str(source_npy),
            "--text",
            edit_text,
            "--out",
            str(edited),
            "--steps",
            str(args.edit_steps),
            "--gd-text",
            str(args.gd_text),
            "--gd-motion",
            str(args.gd_motion),
        ],
        cwd=REPO_ROOT,
    )
    summary["steps"]["edit"] = str(edited)

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
                f"Edit: {edit_text}",
            ],
            cwd=REPO_ROOT,
        )
    _pause("Review edited motion", [edited, preview_edit], args.skip_review)

    summary["total_seconds"] = round(time.time() - t0, 3)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[v05] done in {summary['total_seconds']}s")
    print(f"[v05] summary {summary_path}")
    for k, v in summary["steps"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
