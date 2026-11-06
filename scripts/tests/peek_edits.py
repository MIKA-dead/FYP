import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
for a in r["module_b_editing"]["attempts"]:
    print(
        f"{a['goal'][:32]:34s} | {str(a['instruction'])[:26]:28s} | {a['status']:13s}"
        f" | follow={a['followed']} no_change={a['no_change']}"
        f" | frames={a['frames_ratio']} arm_d={a['arm_raise_delta']}"
        f" | non_target={a['non_target_mpjpe_norm']}"
        f" | jerk {a['jerk_before']}->{a['jerk_after']} pen={a['penetration_after']}"
    )
