"""Calibrate attribute measures for goals the verifier could not check yet."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("backends", "agent"):
    sys.path.insert(0, str(REPO / "scripts" / p))

from metrics import load_joints  # noqa: E402
from motion_validity import body_height, per_frame_floor, up_axis  # noqa: E402

J_KNEE = {1: 4, 2: 5}
J_ANKLE = {1: 7, 2: 8}
J_PELVIS = 0


def measures(p: Path) -> dict:
    j = load_joints(p)
    up = int(up_axis(j))
    h = float(body_height(j, up))
    floor = per_frame_floor(j, up)
    pel = j[:, J_PELVIS, up]
    apex = int(np.argmax(pel))
    base = float(np.percentile(floor, 5))
    out = {
        "airborne": round(float(max(0.0, floor.max() - floor.min()) / h), 3),
        "pelvis_range": round(float(np.ptp(pel) / h), 3),
        "pelvis_rise": round(float((pel.max() - pel[0]) / h), 3),
        "apex_clearance": round(float((floor[apex] - base) / h), 3),
        "apex_t": round(apex / max(1, len(pel) - 1), 2),
    }
    for side, k in (("left", 1), ("right", 2)):
        ankle = j[:, J_ANKLE[k], up]
        knee = j[:, J_KNEE[k], up]
        out[f"ankle_lift_{side}"] = round(float((ankle - ankle.min()).max() / h), 3)
        out[f"knee_lift_{side}"] = round(float((knee - knee.min()).max() / h), 3)
    return out


args = sys.argv[1:]
if len(args) == 1 and args[0].startswith("--run="):
    # every final clip of a benchmark run, in goal order
    root = Path(args[0].split("=", 1)[1])
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        step = d / "step01_generate" / "03_edited.npy"
        if step.is_file():
            rows.append((d.name, measures(step)))
    for name, m in rows:
        print(f"{name:26s}", m)
    sys.exit(0)

for arg in args:
    d = Path(arg)
    f = d if d.suffix == ".npy" else d / "03_edited.npy"
    if f.is_file():
        print(f"{d.name:26s}", measures(f))
