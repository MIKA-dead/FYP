"""
Stable session API for UI (and CLI).

Contract
--------
Request JSON::

  {
    "prompt": "a person sits down",
    "mode": "generate" | "revise",
    "previous_run": null | "<run_dir>",
    "run_dir": null | "<path>",   # optional fixed output dir
    "skip_review": true,
    "skip_viz": false,
    "length": 60
  }

Response JSON (also written to <run_dir>/session_response.json)::

  {
    "schema_version": "session_response_v1",
    "ok": true,
    "task_id": "...",
    "mode": "generate",
    "prompt": "...",
    "run_dir": "...",
    "artifacts": {
      "intent_bundle": ".../00_agent_bundle.json",
      "motion_edit_plan": ".../00_motion_edit_plan.json",
      "edited_npy": ".../03_edited.npy",
      "preview_gif": ".../03_edited_preview.gif",
      "summary": ".../summary.json",
      "quality": ".../04_quality.json"
    },
    "intent": { "primary_action", "modules", "mf_text", "source_policy", ... },
    "quality": { "passed", "score", "checks", ... },
    "next_revise": { "mode": "revise", "previous_run": "<run_dir>" },
    "error": null
  }

Usage::

  python scripts/session/session_api.py \\
      --prompt "a person sits down" --mode generate \\
      --run-dir outputs/session/runs/sit_gen

  python scripts/session/session_api.py --request-json req.json
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
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = Path(__file__).resolve().parent


def _ensure_numpy_env() -> None:
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    mf = os.environ.get("MOTIONFIX_PYTHON") or str(
        Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"
    )
    if not Path(mf).is_file():
        raise SystemExit("numpy missing; set MOTIONFIX_PYTHON to motionfix env")
    os.execv(mf, [mf, *sys.argv])


_ensure_numpy_env()

if str(SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(SESSION_DIR))

ORCH_DIR = REPO_ROOT / "scripts" / "orchestrator"
if str(ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(ORCH_DIR))

from run_session import run_orchestrated_session  # noqa: E402


SCHEMA = "session_response_v1"


def _abs(p: Optional[Path], cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    return p if p.is_absolute() else (cwd / p).resolve()


def run_session(request: Dict[str, Any], *, cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run one generate/revise session via in-repo orchestrator (planner→coder→tester).

    Set request[\"backend\"]=\"v05\" to force legacy v05_pipeline (debug only).
    """
    cwd = cwd or Path.cwd()
    backend = (request.get("backend") or "orchestrator").lower()
    if backend in ("orchestrator", "opencode", "heuristic", ""):
        return run_orchestrated_session(request, cwd=cwd)
    return _run_session_v05_legacy(request, cwd=cwd)


def _run_session_v05_legacy(
    request: Dict[str, Any], *, cwd: Path
) -> Dict[str, Any]:
    """Legacy path: shell out to v05_pipeline (kept for debug / ablation)."""
    from quality_gate import evaluate_file  # local import

    t0 = time.time()
    prompt = (request.get("prompt") or "").strip()
    mode = request.get("mode") or "generate"
    if not prompt:
        return _err("prompt is required", mode=mode, prompt=prompt)

    skip_review = bool(request.get("skip_review", True))
    skip_viz = bool(request.get("skip_viz", False))
    length = int(request.get("length") or 60)
    bridge_iters = int(request.get("bridge_iters") or 40)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _abs(
        Path(request["run_dir"]) if request.get("run_dir") else None,
        cwd,
    )
    if run_dir is None:
        run_dir = (REPO_ROOT / "outputs" / "session" / "runs" / f"{mode}_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    prev = _abs(Path(request["previous_run"]) if request.get("previous_run") else None, cwd)
    if mode == "revise" and prev is None:
        return _err("revise requires previous_run", mode=mode, prompt=prompt, run_dir=run_dir)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "backends" / "v05_pipeline.py"),
        "--mode",
        mode,
        "--prompt",
        prompt,
        "--run-dir",
        str(run_dir),
        "--length",
        str(length),
        "--bridge-iters",
        str(bridge_iters),
        "--use-intent",
    ]
    if skip_review:
        cmd.append("--skip-review")
    if skip_viz:
        cmd.append("--skip-viz")
    if prev is not None:
        cmd += ["--previous-run", str(prev)]
    if request.get("force_bridged_edit"):
        cmd.append("--force-bridged-edit")
    if request.get("edit_source_id"):
        cmd += ["--edit-source-id", str(request["edit_source_id"])]

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        return _err(
            f"v05_pipeline failed ({proc.returncode})",
            mode=mode,
            prompt=prompt,
            run_dir=run_dir,
        )

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    plan = summary.get("motion_edit_plan") or {}
    action = plan.get("primary_action")
    edited = run_dir / "03_edited.npy"
    preview = run_dir / "03_edited_preview.gif"
    intent_bundle = run_dir / "00_agent_bundle.json"
    plan_path = run_dir / "00_motion_edit_plan.json"
    quality_path = run_dir / "04_quality.json"

    quality: Dict[str, Any]
    if edited.is_file():
        try:
            quality = evaluate_file(edited, action=action)
        except Exception as exc:  # noqa: BLE001
            quality = {
                "schema_version": "quality_gate_v1",
                "passed": False,
                "score": 0,
                "error": str(exc),
                "hard_failures": ["load_joints"],
                "checks": [],
            }
        quality_path.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        quality = {
            "passed": False,
            "score": 0,
            "hard_failures": ["missing_edited"],
            "checks": [],
        }

    if not skip_viz and edited.is_file() and not preview.is_file():
        mf_py = os.environ.get(
            "MOTIONFIX_PYTHON",
            str(Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"),
        )
        subprocess.run(
            [
                mf_py,
                str(REPO_ROOT / "scripts" / "backends" / "visualize_motion.py"),
                "--input",
                str(edited),
                "--out",
                str(preview),
                "--title",
                f"{mode}: {plan.get('mf_text') or prompt}",
            ],
            cwd=str(REPO_ROOT),
            env=env,
        )

    ok = bool(edited.is_file()) and bool(quality.get("passed", False))
    response: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "ok": ok,
        "task_id": summary.get("task_id") or plan.get("from_intent_task_id"),
        "mode": mode,
        "prompt": prompt,
        "run_dir": str(run_dir),
        "previous_run": str(prev) if prev else None,
        "artifacts": {
            "intent_bundle": str(intent_bundle) if intent_bundle.is_file() else None,
            "motion_edit_plan": str(plan_path) if plan_path.is_file() else None,
            "edited_npy": str(edited) if edited.is_file() else None,
            "preview_gif": str(preview) if preview.is_file() else None,
            "summary": str(summary_path) if summary_path.is_file() else None,
            "quality": str(quality_path) if quality_path.is_file() else None,
        },
        "intent": {
            "primary_action": action,
            "modules": plan.get("modules"),
            "mf_text": plan.get("mf_text"),
            "mld_text": plan.get("mld_text"),
            "source_policy": plan.get("source_policy"),
            "edit_source": plan.get("edit_source"),
            "preferred_source_id": plan.get("preferred_source_id"),
            "span": plan.get("span"),
            "warning": plan.get("warning"),
        },
        "quality": {
            "passed": quality.get("passed"),
            "score": quality.get("score"),
            "hard_failures": quality.get("hard_failures"),
            "soft_failures": quality.get("soft_failures"),
            "metrics": quality.get("metrics"),
        },
        "pipeline_seconds": summary.get("total_seconds"),
        "total_seconds": round(time.time() - t0, 3),
        "next_revise": {
            "mode": "revise",
            "previous_run": str(run_dir),
            "prompt": "<user improvement text>",
        },
        "error": None
        if ok
        else (
            "quality_gate_failed"
            if edited.is_file()
            else "missing_edited_npy"
        ),
        "backend": "v05",
    }

    out_path = run_dir / "session_response.json"
    out_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[session_api] ok={ok} action={action} quality={quality.get('score')} "
        f"→ {out_path}"
    )
    return response


def _err(
    msg: str,
    *,
    mode: str = "generate",
    prompt: str = "",
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "ok": False,
        "task_id": None,
        "mode": mode,
        "prompt": prompt,
        "run_dir": str(run_dir) if run_dir else None,
        "previous_run": None,
        "artifacts": {},
        "intent": {},
        "quality": {},
        "error": msg,
        "next_revise": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Session API (UI contract)")
    ap.add_argument("--request-json", type=Path, default=None)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--mode", choices=["generate", "revise"], default="generate")
    ap.add_argument("--previous-run", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None, help="Also write response JSON here")
    args = ap.parse_args()

    if args.request_json:
        req = json.loads(args.request_json.read_text(encoding="utf-8-sig"))
    else:
        if not args.prompt:
            raise SystemExit("Provide --prompt or --request-json")
        req = {
            "prompt": args.prompt,
            "mode": args.mode,
            "previous_run": str(args.previous_run) if args.previous_run else None,
            "run_dir": str(args.run_dir) if args.run_dir else None,
            "skip_review": True,
            "skip_viz": bool(args.skip_viz),
            "length": args.length,
        }

    resp = run_session(req)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"ok": resp["ok"], "run_dir": resp.get("run_dir"), "error": resp.get("error")}, indent=2))
    if not resp.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
