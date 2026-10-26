"""
Apply MotionsFix face-front canonicalize to a bridged source npy (in-place or --out).

Env: conda `motionfix`

  python scripts/backends/motionfix_facefront.py \\
      --in outputs/v05/.../02_motionfix_source.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTIONFIX_ROOT = REPO_ROOT / "third_party" / "motionfix"
DEFAULT_SMPL = MOTIONFIX_ROOT / "data" / "body_models"


def main() -> None:
    p = argparse.ArgumentParser(description="MotionFix face-front on bridged npy")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Default: overwrite --in")
    p.add_argument("--smpl", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--motionfix-root", type=Path, default=MOTIONFIX_ROOT)
    args = p.parse_args()

    cwd0 = Path.cwd()
    root = args.motionfix_root.resolve()
    inp = args.inp if args.inp.is_absolute() else (cwd0 / args.inp).resolve()
    out = args.out
    if out is None:
        out = inp
    else:
        out = out if out.is_absolute() else (cwd0 / out).resolve()
    smpl = args.smpl if args.smpl.is_absolute() else (cwd0 / args.smpl).resolve()

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    backends = str(REPO_ROOT / "scripts" / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)

    from motionfix_edit import facefront_motion_source, _setup_motionfix_env

    _setup_motionfix_env(root)

    obj = np.load(inp, allow_pickle=True).item()
    if "motion_source" not in obj:
        raise SystemExit(f"{inp} missing motion_source")
    ms = obj["motion_source"]
    # numpy → torch for facefront
    ms_t = {
        k: (torch.as_tensor(v).float() if not torch.is_tensor(v) else v.float())
        for k, v in ms.items()
        if k in ("rots", "trans", "joint_positions")
    }
    ms_out = facefront_motion_source(ms_t, smpl)
    obj["motion_source"] = {
        "rots": ms_out["rots"].numpy()
        if torch.is_tensor(ms_out["rots"])
        else np.asarray(ms_out["rots"]),
        "trans": ms_out["trans"].numpy()
        if torch.is_tensor(ms_out["trans"])
        else np.asarray(ms_out["trans"]),
        "joint_positions": ms_out["joint_positions"].numpy()
        if torch.is_tensor(ms_out["joint_positions"])
        else np.asarray(ms_out["joint_positions"]),
    }
    obj["motionfix_canonical"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, obj)

    meta_path = out.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            meta = {}
    meta["motionfix_canonical"] = True
    meta["facefront"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[motionfix_facefront] saved {out} motionfix_canonical=True")


if __name__ == "__main__":
    main()
