"""Print root/horizontal motion of a clip's joints (sanity check for root_travel)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from metrics import load_joints  # noqa: E402
from motion_validity import body_height, up_axis  # noqa: E402

for p in sys.argv[1:]:
    j = load_joints(p)
    up = up_axis(j)
    plan = [a for a in (0, 1, 2) if a != up]
    pel = j[:, 0, :]
    xy = pel[:, plan]
    span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(f"{Path(p).parent.parent.name}: T={len(j)} up={up} "
          f"pelvis_span={span:.3f} pelvis_path={path:.3f} "
          f"heights={span / body_height(j, up):.2f}")
