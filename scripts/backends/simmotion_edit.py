"""
Optional SimMotionEdit backend (scaffold).

SimMotionEdit (CVPR 2025) is a stronger text-conditioned motion editor than
vanilla MotionFix for many body-part edits. This wrapper:

  1. Looks for a local checkout under third_party/SimMotionEdit (or SIMMOTION_ROOT)
  2. If checkpoints + env are ready, runs their inference entrypoint
  3. Otherwise exits with a clear install checklist (no silent MotionsFix swap)

Install sketch (manual — large deps / GPU):

  git clone https://github.com/lzhyu/SimMotionEdit third_party/SimMotionEdit
  # Follow their README (MotionFix-style env + pretrained link + sim_file).
  set SIMMOTION_ROOT=...\\third_party\\SimMotionEdit
  set SIMMOTION_PYTHON=...\\envs\\motionfix\\python.exe

  python scripts/backends/simmotion_edit.py \\
      --source-npy path/to/source.npy --text "walk forward and wave right hand" \\
      --out-npy path/to/edited.npy

Until checkpoints are ready, --allow-copy-fallback copies source→out (plumbing only).
Official eval entry is motionfix_evaluate.py (folder=...); single-clip CLI may need
SIMMOTION_CMD_JSON override once you adapt their sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = "https://github.com/lzhyu/SimMotionEdit"


def _simmotion_root() -> Path:
    env = (os.environ.get("SIMMOTION_ROOT") or "").strip()
    if env:
        return Path(env)
    return REPO_ROOT / "third_party" / "SimMotionEdit"


def _simmotion_python() -> str:
    env = (os.environ.get("SIMMOTION_PYTHON") or "").strip()
    if env:
        return env
    return sys.executable


def discover() -> Dict[str, Any]:
    root = _simmotion_root()
    status = {
        "root": str(root),
        "exists": root.is_dir(),
        "readme": (root / "README.md").is_file() if root.is_dir() else False,
        "python": _simmotion_python(),
        "official_repo": OFFICIAL_REPO,
        "ready": False,
        "hint": "",
    }
    if not status["exists"]:
        status["hint"] = (
            f"Clone into third_party/SimMotionEdit ({OFFICIAL_REPO}) "
            "or set SIMMOTION_ROOT."
        )
        return status
    # Heuristic: MotionFix-style evaluate + train entrypoints
    candidates = [
        root / "motionfix_evaluate.py",
        root / "sample.py",
        root / "generate.py",
        root / "infer.py",
        root / "demo.py",
        root / "train.py",
    ]
    entry = next((p for p in candidates if p.is_file()), None)
    status["entry"] = str(entry) if entry else None
    ckpt_dirs = [root / "checkpoints", root / "ckpt", root / "pretrained", root / "logs"]
    status["has_ckpt_dir"] = any(d.is_dir() and any(d.iterdir()) for d in ckpt_dirs if d.is_dir())
    status["has_configs"] = (root / "configs").is_dir()
    # Repo alone is "installed"; ready for edit needs ckpt + CMD override or evaluate folder.
    status["ready"] = bool(entry) and (
        status["has_ckpt_dir"] or bool((os.environ.get("SIMMOTION_CMD_JSON") or "").strip())
    )
    if not status["ready"]:
        status["hint"] = (
            "Repo present. Download pretrained weights (see README link), set "
            "preproc.sim_file in configs/data/motionfix.yaml, then either set "
            "SIMMOTION_CMD_JSON for single-clip edit or use motionfix_evaluate.py."
        )
    return status


def run_edit(
    *,
    source_npy: Path,
    text: str,
    out_npy: Path,
    allow_copy_fallback: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    info = discover()
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    if not info.get("ready"):
        if allow_copy_fallback and source_npy.is_file():
            shutil.copy2(source_npy, out_npy)
            return {
                "ok": True,
                "fallback": "copy_source",
                "edited_npy": str(out_npy),
                "discover": info,
                "warning": info.get("hint") or "SimMotionEdit not ready; copied source",
            }
        return {
            "ok": False,
            "error": "simmotion_not_installed",
            "discover": info,
            "hint": info.get("hint"),
        }

    entry = Path(info["entry"])
    cmd = [
        info["python"],
        str(entry),
        "--source",
        str(source_npy),
        "--text",
        text,
        "--out",
        str(out_npy),
    ]
    if extra_args:
        cmd.extend(extra_args)
    # Many repos use different CLI flags; allow override via env JSON list.
    override = (os.environ.get("SIMMOTION_CMD_JSON") or "").strip()
    if override:
        try:
            cmd = json.loads(override)
            cmd = [
                c.format(source=str(source_npy), text=text, out=str(out_npy))
                if isinstance(c, str)
                else c
                for c in cmd
            ]
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"bad SIMMOTION_CMD_JSON: {exc}", "discover": info}

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(info["root"]),
            timeout=int(os.environ.get("SIMMOTION_TIMEOUT_S") or 600),
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "discover": info, "cmd": cmd}

    if proc.returncode != 0 or not out_npy.is_file():
        return {
            "ok": False,
            "error": "simmotion_infer_failed",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-2000:],
            "stdout": (proc.stdout or "")[-1000:],
            "cmd": cmd,
            "discover": info,
        }
    return {
        "ok": True,
        "edited_npy": str(out_npy),
        "discover": info,
        "cmd": cmd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional SimMotionEdit wrapper")
    ap.add_argument("--source-npy", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument(
        "--allow-copy-fallback",
        action="store_true",
        help="If SimMotionEdit not ready, copy source to out (plumbing only)",
    )
    ap.add_argument("--discover-only", action="store_true")
    ns = ap.parse_args()

    if ns.discover_only:
        print(json.dumps(discover(), ensure_ascii=False, indent=2))
        return

    result = run_edit(
        source_npy=ns.source_npy,
        text=ns.text,
        out_npy=ns.out_npy,
        allow_copy_fallback=ns.allow_copy_fallback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
