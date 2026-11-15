"""Draw N MLD drafts of one goal and print what the resample tool would keep."""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("agent", "backends", "session", "orchestrator"):
    sys.path.insert(0, str(REPO / "scripts" / _p))

from tools import MotionTools  # noqa: E402


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else "a person walks in place"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = REPO / "outputs" / "agent_probe" / f"resample_{int(time.time())}"
    tools = MotionTools(out, generator="mld", skip_viz=True, goal=goal)
    t0 = time.time()
    obs = tools.resample_motion(goal, n)
    print(json.dumps({k: v for k, v in obs.items() if k != "candidates"},
                     ensure_ascii=False, indent=2))
    for c in obs.get("candidates") or []:
        print(f"  cand{c['cand']}: valid={c.get('valid')} satisfied={c.get('satisfied')} "
              f"gap={c.get('gap')} {c.get('measured')}")
    print(f"chosen={obs.get('chosen')} in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
