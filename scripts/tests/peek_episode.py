"""Print one episode's tool trace as a compact table."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
a = json.loads((d / "agent_run.json").read_text(encoding="utf-8-sig"))
print("GOAL:", a.get("goal"))
print("satisfied:", a.get("satisfied"), "| claim:", a.get("agent_claim_satisfied"),
      "| checker:", a.get("checker_satisfied"), "| agreement:", a.get("agreement"))
for s in a.get("steps", []):
    obs = s.get("observation") or {}
    key = obs.get("error") or ""
    extra = ""
    if s["action"] == "check_requirement":
        extra = f"satisfied={obs.get('satisfied')} {obs.get('reasons')}"
    elif s["action"] == "edit_motion":
        extra = (f"new_fails={obs.get('new_hard_failures')} "
                 f"pre_fails={obs.get('pre_existing_hard_failures')} "
                 f"ok={obs.get('ok')} no_change={obs.get('no_change')} "
                 f"frames={obs.get('frames_ratio')}")
    elif s["action"] == "describe_motion":
        cap = obs.get("caption")
        if isinstance(cap, dict):
            cap = cap.get("text")
        m = obs.get("metrics") or {}
        extra = f"{str(cap)[:70]!r} travel={m.get('root_travel')} yaw={m.get('yaw_span_deg')}"
    elif s["action"] == "finish":
        extra = str(obs.get("reason"))[:120]
    print(f"{s['step']:>2} {s['action']:18s} {s['args']} -> {key or 'ok'} {extra}")
