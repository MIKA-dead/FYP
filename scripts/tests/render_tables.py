"""Render the markdown tables of docs/agentic_pipeline_v1.md from a report.json."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/agent_bench/_v3_final4/report.json")
r = json.loads(p.read_text(encoding="utf-8-sig"))
L = r["loop"]
A = r["module_a_generation"]
B = r["module_b_editing"]
C = r.get("module_c_caption") or {}

print("### headline\n")
print("| Quantity | Value |")
print("|---|---|")
print(f"| Goal satisfaction | **{sum(1 for e in L['rows'] if e['satisfied'])} / "
      f"{len(L['rows'])} = {L['success_rate']:.2f}** |")
ret = A.get("retrieval") or {}
print(f"| Caption-mediated R@1 / R@3 | {ret.get('r@1')} / {ret.get('r@3')} |")
print(f"| Mean prompt↔caption GloVe cosine | {ret.get('mean_cosine')} |")
print(f"| Batch diversity (mean pairwise) | {A.get('diversity', {}).get('mean_pairwise')} |")
pl = A.get("plausibility_mean") or {}
print(f"| Foot slide / jerk / floor penetration (mean) | "
      f"{pl.get('foot_slide')} / {pl.get('jerk')} / {pl.get('ground_penetration')} body-heights |")
print(f"| Edit attempts / applied / rolled back | {B.get('n_attempts')} / "
      f"{B.get('n')} / {B.get('n_rolled_back')} ({B.get('n_session_error')} refused up front) |")
print(f"| Instruction following / frame retention | "
      f"{B.get('instruction_following_rate')} / {B.get('mean_frame_retention')} |")
print(f"| Non-target MPJPE (height-normalised) | {B.get('mean_non_target_mpjpe_norm')} |")
print(f"| Instruction coverage (supported / attempts) | {B.get('instruction_coverage')} |")
print(f"| Mean controller steps per goal | {L.get('mean_steps')} |")
print(f"| Agent ↔ checker agreement | {L.get('agreement_rate')} |")
print(f"| Mean seconds / tokens per goal | {L.get('mean_seconds')} / "
      f"{L.get('total_tokens') // max(1, len(L['rows']))} |")
print(f"| Wall time | {r.get('wall_seconds')} s |")
print()
print("### module (c)\n")
if C.get("aggregate"):
    m = C["aggregate"]
    keys = [k for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider_d", "glove_cosine") if k in m]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    print("| " + " | ".join(f"{m[k]:.4f}" if isinstance(m[k], float) else str(m[k]) for k in keys) + " |")
    print(f"\nn = {C.get('n')}, corpus-level")
print()
print("### per-goal (final clip)\n")
print("| Goal | edits | claim | checker | agreement | alignment | seconds |")
print("|---|---|---|---|---|---|---|")
for e in L["rows"]:
    print(f"| {e['goal']} | {e['edits']}/{e['edit_attempts']} | "
          f"{e['agent_claim_satisfied']} | {e['checker_satisfied']} | "
          f"{e['agreement']} | {e.get('final_alignment')} | {e['seconds']:.0f} |")
