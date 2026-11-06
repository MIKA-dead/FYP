"""Score every goal of a benchmark run with verify_goal (no LLM, no backends)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("agent", "backends"):
    sys.path.insert(0, str(REPO / "scripts" / p))

from metrics import kinematic_summary, load_joints  # noqa: E402
from verify import verify_goal  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/agent_bench/_v3_final4")
fails = 0
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    run = d / "agent_run.json"
    if not run.is_file():
        continue
    a = json.loads(run.read_text(encoding="utf-8-sig"))
    goal = a.get("goal") or ""
    final = a.get("final_run") or ""
    clip = (Path(final) / "03_edited.npy") if final else (d / "step01_generate" / "03_edited.npy")
    if not clip.is_file() or not goal:
        continue
    m = kinematic_summary(load_joints(clip))
    v = verify_goal(goal, {"metrics": m})
    fails += 0 if v["ok"] else 1
    print(f"{'ok  ' if v['ok'] else 'FAIL'} {goal[:44]:46s} {v['reasons']}")
print(f"\n{fails} goal(s) fail the requirement check on the final clip")
