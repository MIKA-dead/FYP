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
    p.add_argument(
        "--samples", type=int, default=1,
        help="Draw N independent samples in one process (one model load). "
             "The first goes to --out; the rest to <out>_s2.npy, <out>_s3.npy, ...",
    )
    return p.parse_args()


def _newest_example_npy(results_root: Path, since: float, *, all_in_one: bool = False) -> Optional[Path]:
    best = None
    best_mtime = since - 1
    if not results_root.is_dir():
        return None
    for p in results_root.rglob("Example_*.npy"):
        if "_mesh" in p.name:
            continue
        # With --allinone MLD also writes Example_<len>_all.npy (every
        # replication in one array); either kind can be what we want, but
        # never mix them up.
        is_all = p.stem.endswith("_all")
        if is_all != all_in_one:
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= since and m >= best_mtime:
            best = p
            best_mtime = m
    return best


def _split_samples(arr: np.ndarray) -> list:
    """Split MLD's --allinone array into one (T, J, 3) array per replication.

    MLD writes ``concat_i( concat_r( joints[r][i] ) )`` — shape
    (samples, replications, T, J, 3) for one text. With one text and N
    replications the useful axis is the replication axis, so unwrap it.
    """
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.asarray(arr.item().get("joint_positions"))
    while a.ndim > 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 5:
        a = a[0]                      # (reps, T, J, 3)
    if a.ndim == 4:
        return [np.asarray(a[i], dtype=np.float32) for i in range(a.shape[0])]
    if a.ndim == 3:
        return [np.asarray(a, dtype=np.float32)]
    raise ValueError(f"cannot split MLD output of shape {a.shape}")


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
    n_samples = max(1, int(args.samples or 1))
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
    if n_samples > 1:
        # Every MLD demo process imports mld/utils/fixseed.py, which calls
        # torch.manual_seed(10) — so re-running the same prompt reproduces the
        # same clip byte for byte. Asking for several replications inside ONE
        # process is what actually draws independent samples, and it pays the
        # ~30 s model load once instead of N times.
        cmd += ["--replication", str(n_samples), "--allinone"]
    print(f"[mld_generate] cwd={root}")
    print(f"[mld_generate] {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    # Windows App Control / spaCy DLL can flake: the same import fails with
    # "DLL load failed while importing linalg: application control policy
    # blocked this file" on one launch and succeeds on the next. Retry a few
    # times before giving up.
    attempts = 3
    proc = None
    for attempt in range(attempts):
        proc = subprocess.run(cmd, cwd=str(root), env=env)
        if proc.returncode == 0:
            break
        if attempt < attempts - 1:
            print(
                f"[mld_generate] demo.py failed (code={proc.returncode}); "
                f"retry {attempt + 1}/{attempts - 1}…",
                flush=True,
            )
            time.sleep(2.0)
    if proc is None or proc.returncode != 0:
        code = proc.returncode if proc is not None else -1
        raise SystemExit(
            f"MLD demo.py failed with code {code}. "
            "If spaCy DLL is blocked by Windows policy, allow "
            "miniconda3/envs/mld or reinstall: conda run -n mld pip install -U spacy"
        )

    npy = _newest_example_npy(root / "results", since=t0 - 2,
                              all_in_one=n_samples > 1)
    if npy is None:
        raise FileNotFoundError("No Example_*.npy found under results/ after demo")

    joints = np.load(npy, allow_pickle=True)
    if out is None:
        out = (REPO_ROOT / "outputs" / "v05" / "mld_joints.npy").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    saved = []
    if n_samples > 1:
        # Split the --allinone array, one file per replication.
        parts = _split_samples(joints)
        for i, part in enumerate(parts, start=1):
            dst = out if i == 1 else out.with_name(f"{out.stem}_s{i}{out.suffix}")
            np.save(dst, part)
            saved.append(str(dst))
        joints = parts[0]
    else:
        if isinstance(joints, np.ndarray) and joints.dtype == object:
            joints = np.asarray(joints.item().get("joint_positions"))
        shutil.copy2(npy, out)
        saved.append(str(out))

    meta = {
        "out": str(out),
        "samples": saved,
        "n_samples": len(saved),
        "source_result": str(npy),
        "text": args.text,
        "length_requested": args.length,
        "shape": list(np.shape(joints)),
        "seconds": round(time.time() - t0, 3),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[mld_generate] saved {len(saved)} sample(s) shape={np.shape(joints)} "
          f"({meta['seconds']}s)")


if __name__ == "__main__":
    main()
