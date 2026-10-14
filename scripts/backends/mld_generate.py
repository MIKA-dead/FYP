"""
Thin MLD text→motion wrapper.

Env: conda `mld`

  python scripts/backends/mld_generate.py \\
      --text "a person walks forward." \\
      --length 60 \\
      --out outputs/v05/walk_joints.npy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MLD_ROOT = REPO_ROOT / "third_party" / "motion-latent-diffusion"
DEFAULT_CFG = MLD_ROOT / "configs" / "config_mld_humanml3d.yaml"
DEFAULT_ASSETS = MLD_ROOT / "configs" / "assets.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="MLD text → joints npy")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--length", type=int, default=60, help="Frames (16–196)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--cfg-assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--mld-root", type=Path, default=MLD_ROOT)
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def _newest_example_npy(results_root: Path, since: float) -> Optional[Path]:
    best = None
    best_mtime = since - 1
    if not results_root.is_dir():
        return None
    for p in results_root.rglob("Example_*.npy"):
        if "_mesh" in p.name:
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= since and m >= best_mtime:
            best = p
            best_mtime = m
    return best


def main():
    args = parse_args()
    cwd0 = Path.cwd()
    root = args.mld_root.resolve()

    def _abs(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (cwd0 / p).resolve()

    out = _abs(args.out)
    cfg = _abs(args.cfg)
    assets = _abs(args.cfg_assets)

    work = REPO_ROOT / "outputs" / "v05" / "_mld_run"
    work.mkdir(parents=True, exist_ok=True)
    prompt = work / "prompt.txt"
    # MLD demo format: "<length> <text>"
    prompt.write_text(f"{args.length} {args.text}\n", encoding="utf-8")

    t0 = time.time()
    cmd = [
        args.python,
        "demo.py",
        "--cfg",
        str(cfg),
        "--cfg_assets",
        str(assets),
        "--example",
        str(prompt),
    ]
    print(f"[mld_generate] cwd={root}")
    print(f"[mld_generate] {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"MLD demo.py failed with code {proc.returncode}")

    npy = _newest_example_npy(root / "results", since=t0 - 2)
    if npy is None:
        raise FileNotFoundError("No Example_*.npy found under results/ after demo")

    joints = np.load(npy)
    if out is None:
        out = (REPO_ROOT / "outputs" / "v05" / "mld_joints.npy").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npy, out)

    meta = {
        "out": str(out),
        "source_result": str(npy),
        "text": args.text,
        "length_requested": args.length,
        "shape": list(joints.shape),
        "seconds": round(time.time() - t0, 3),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[mld_generate] saved {out} shape={joints.shape} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
