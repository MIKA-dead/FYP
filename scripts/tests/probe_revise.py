"""Probe one revise instruction on a previous run and print the before/after numbers."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("session", "orchestrator", "backends", "agent"):
    sys.path.insert(0, str(REPO / "scripts" / p))

from metrics import kinematic_summary, load_joints  # noqa: E402
from motion_validity import physical_report  # noqa: E402
from session_api import run_session  # noqa: E402

prev = Path(sys.argv[1]).resolve()
prompt = sys.argv[2]
out = Path(sys.argv[3]).resolve()
resp = run_session(
    {
        "prompt": prompt,
        "mode": "revise",
        "previous_run": str(prev),
        "run_dir": str(out),
        "skip_review": True,
        "backend": "orchestrator",
    },
    cwd=REPO,
)
if not (out / "03_edited.npy").is_file():
    print("RESULT no clip", resp.get("ok"), resp.get("error"))
    raise SystemExit(1)
j = load_joints(out / "03_edited.npy")
a = kinematic_summary(j)
b = kinematic_summary(load_joints(prev / "03_edited.npy"))
print(
    f"RESULT ok={resp.get('ok')} drop {b['pelvis_drop']} -> {a['pelvis_drop']} | "
    f"pen {a['ground_penetration']} | bone_cv {a['bone_cv']} | "
    f"T {b['T']} -> {a['T']} | valid={physical_report(j)['valid']}"
)
