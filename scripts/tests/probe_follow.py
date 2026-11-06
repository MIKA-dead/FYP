"""Check instruction_following on the recorded edit pairs of a run."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("agent", "backends"):
    sys.path.insert(0, str(REPO / "scripts" / p))

from metrics import (  # noqa: E402
    infer_target_joints,
    instruction_following,
    kinematic_summary,
    load_joints,
    preservation,
)

root = Path(sys.argv[1])
for ep in sorted(p for p in root.iterdir() if p.is_dir()):
    run = ep / "agent_run.json"
    if not run.is_file():
        continue
    a = json.loads(run.read_text(encoding="utf-8-sig"))
    for step in a["steps"]:
        if step.get("action") != "edit_motion":
            continue
        obs = step.get("observation") or {}
        ins = (step.get("args") or {}).get("instruction") or ""
        before = obs.get("before") or {}
        after = obs.get("after") or {}
        if not before or not after:
            continue
        pres = {
            "target_change": obs.get("target_change"),
            "non_target_mpjpe": obs.get("non_target_mpjpe_norm"),
            "target_vs_rest": obs.get("target_vs_rest"),
        }
        f = instruction_following(ins, before, after, pres)
        print(f"{ep.name[:6]} {ins[:38]:40s} followed={f['followed']}")
        for c in f["checks"]:
            print(f"        {c['attribute']:16s} ok={c['ok']} delta={c['delta']} need {c['expected']}")
