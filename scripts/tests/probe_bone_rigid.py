"""Isolate which constraint stage breaks bone rigidity on a given clip."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("backends", "agent"):
    sys.path.insert(0, str(REPO / "scripts" / p))

from constraint_postprocess import raise_arm  # noqa: E402
from metrics import load_joints  # noqa: E402
from motion_validity import body_height, bone_cv  # noqa: E402

src = Path(sys.argv[1])
j = load_joints(src)
spec = {"side": "right", "level": "high", "target": "high", "step_up": True}
out, meta = raise_arm(j, spec)
print(f"source bone_cv={bone_cv(j):.4f} height={body_height(j):.4f}")
print(f"raise  bone_cv={bone_cv(out):.4f} height={body_height(out):.4f} "
      f"changed={meta.get('changed')}")
print("meta", {k: meta.get(k) for k in ("wrist_lift_before", "wrist_lift_after",
                                       "reach_clamped", "ramp_frames")})
# Per-frame pelvis displacement is the usual culprit when bones "grow".
print("pelvis move max", float(np.abs(out[:, 0] - j[:, 0]).max()))
