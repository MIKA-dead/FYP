"""Print the headline numbers of an agent benchmark report."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/agent_bench/_v3_final/report.json")
r = json.loads(p.read_text(encoding="utf-8-sig"))
L = r["loop"]
print("wall_seconds", r.get("wall_seconds"))
print({k: L.get(k) for k in (
    "success_rate", "agreement_rate", "check_coverage", "mean_edits",
    "mean_edit_attempts", "mean_steps", "mean_seconds", "total_tokens",
    "mean_alignment_gain")})
for e in L["rows"]:
    print(f"{e['goal'][:46]:48s} ok={e['satisfied']} "
          f"edits={e['edits']}/{e['edit_attempts']} agree={e['agreement']} "
          f"align={e.get('final_alignment')} steps={e['steps']}")
