"""Quick self-checks for the agent's instruction gate and physical validity."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "scripts" / "agent", REPO / "scripts" / "backends"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edits import classify  # noqa: E402
from motion_validity import body_height, offset_to_ground, physical_report, reground  # noqa: E402
from metrics import load_joints  # noqa: E402

import numpy as np  # noqa: E402

CASES = [
    ("sit lower", True, "sit_deeper"),
    ("sit deeper into the chair", True, "sit_deeper"),
    ("raise the right arm higher", True, "arm_raise"),
    ("lift the left hand above the head", True, "arm_raise"),
    ("lower the right arm", True, "arm_lower"),
    ("wave with the left hand", True, "wave"),
    ("walk forward", True, "walk"),
    ("turn around in place", True, "turn"),
    ("walk slower", True, "speed"),
    ("walk in place, keep the root planted with no forward travel", False, "in_place"),
    ("turn around in place", True, "turn"),
    ("stand up from the sit", True, "stand_up"),
    ("sit down immediately, no walking before", True, "sit_remove_preamble"),
    ("pick up the cup", False, "object"),
    ("change the camera angle", False, "camera"),
    ("make the person taller", False, "body"),
    ("jump twice", False, "jump"),
    ("do a dance", False, "dance"),
    ("", False, None),
]


def main() -> int:
    bad = 0
    for text, want_supported, want_family in CASES:
        got = classify(text)
        ok = got["supported"] == want_supported and (want_family is None or got["family"] == want_family)
        if not ok:
            bad += 1
        print(f"{'ok ' if ok else 'FAIL'} {text!r:46s} supported={got['supported']} "
              f"family={got['family']} suggestion={got.get('suggestion')}")

    broken = REPO / "outputs/agent_bench/final_llm_v2/ep03_20261106-163219/step02_edit/03_edited.npy"
    good = REPO / "outputs/agent_bench/final_dataset_llm/ep03_20261106-163034/step01_generate/03_edited.npy"
    if broken.is_file():
        rep = physical_report(load_joints(broken))
        ok = not rep["valid"] and rep["ground"]["penetration"] > 1.0
        bad += 0 if ok else 1
        print(f"{'ok ' if ok else 'FAIL'} broken clip rejected: penetration={rep['ground']['penetration']}")
    if good.is_file():
        rep = physical_report(load_joints(good))
        ok = rep["valid"]
        bad += 0 if ok else 1
        print(f"{'ok ' if ok else 'FAIL'} good clip accepted: penetration={rep['ground']['penetration']}")
        fixed, off = reground(load_joints(good))
        bad += 0 if off == 0.0 else 1
        print(f"{'ok ' if off == 0.0 else 'FAIL'} grounded clip is not shifted (offset={off})")

    # A restored MotionsFix-native source keeps its root at the origin: the
    # whole clip sits below the floor and one vertical shift fixes it.
    j = load_joints(good) if good.is_file() else np.zeros((20, 22, 3))
    h = float(body_height(j))
    shifted = j.copy()
    shifted[:, :, 2] -= 0.8 * h
    off = offset_to_ground(shifted)
    want = -float(np.median(shifted[: max(3, len(shifted) // 10), :, 2].min(axis=1)))
    ok = abs(off - want) < 1e-3 and off > 0.1
    bad += 0 if ok else 1
    print(f"{'ok ' if ok else 'FAIL'} start-offset clip is lifted by {off:.3f} (want {want:.3f})")
    ok = offset_to_ground(j) == 0.0
    bad += 0 if ok else 1
    print(f"{'ok ' if ok else 'FAIL'} grounded clip gets no lift")

    print(f"\n{bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
