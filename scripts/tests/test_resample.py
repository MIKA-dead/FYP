"""Checks for ``resample_motion`` that do not need the MLD environment.

The diffusion generator is exercised separately (``probe_resample.py``); these
tests cover the part that must be right either way — how drafts are ranked, and
what the tool says when the generator is not actually producing new clips.

  python scripts/tests/test_resample.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("agent", "backends", "session", "orchestrator"):
    sys.path.insert(0, str(REPO / "scripts" / _p))

from constraint_postprocess import _horiz, _up_axis  # noqa: E402
from tools import MotionTools  # noqa: E402

GOAL = "a person walks in place"
SOURCES = sorted((REPO / "outputs").glob("agent_bench/*/ep*/step01_*/01_draft.npy"))
CHECKS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _source_joints() -> np.ndarray:
    for path in SOURCES:
        try:
            raw = np.load(path, allow_pickle=True)
        except Exception:  # noqa: BLE001
            continue
        if raw.dtype == object:
            raw = np.asarray(raw.item().get("joint_positions"))
        j = np.asarray(raw, dtype=np.float32)
        if j.ndim == 3 and j.shape[0] >= 20 and j.shape[1] >= 22:
            return j
    raise RuntimeError("no draft clip found under outputs/agent_bench")


def _with_travel(j: np.ndarray, metres: float) -> np.ndarray:
    """Scale the root's horizontal displacement so total travel is ``metres``."""
    out = j.copy()
    up = _up_axis(out)
    hz = _horiz(up)
    disp = out[:, 0, hz] - out[0, 0, hz]
    total = float(np.linalg.norm(disp[-1] - disp[0]))
    if total > 1e-6:
        out[:, :, hz] += (disp * (metres / total))[:, None, :] - disp[:, None, :]
    return out


def _write_samples(dirpath: Path, travels: list) -> list:
    dirpath.mkdir(parents=True, exist_ok=True)
    base = _source_joints()
    paths = []
    for i, t in enumerate(travels, start=1):
        p = dirpath / f"mld_sample{i}.npy"
        np.save(p, _with_travel(base, t).astype(np.float32))
        paths.append(p)
    return paths


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="resample_test_"))
    print(f"workdir: {tmp}")
    base = _source_joints()

    # 1) ranking: of three drafts, the in-place one must win.
    tools = MotionTools(tmp / "rank", generator="mld", skip_viz=True, goal=GOAL)
    samples = _write_samples(tmp / "draw", [1.40, 0.60, 0.05])
    tools._draw_samples = lambda prompt, n, out_dir: samples  # noqa: ARG005
    obs = tools.resample_motion(GOAL, 3)
    chosen = obs.get("chosen")
    cands = {c["cand"]: c for c in obs.get("candidates") or []}
    travel = (cands.get(chosen) or {}).get("measured", {}).get("root_travel")
    check("resample_ok", bool(obs.get("ok")), str(obs.get("error") or ""))
    check("distinct_drafts", obs.get("distinct_candidates") == 3,
          f"distinct={obs.get('distinct_candidates')}")
    check("picks_lowest_gap", obs.get("chosen_satisfied") is True and float(travel or 9) <= 0.25,
          f"chosen=cand{chosen} travel={travel} gap={obs.get('chosen_gap')}")
    check("reports_every_candidate", len(cands) == 3, f"n={len(cands)}")
    check("budget_charged", tools.resamples == 1, f"resamples={tools.resamples}")

    # 2) a generator that produced no fresh draw must not look like success.
    def fake_fallback(self, prompt, out_dir, draft_joints=None):  # noqa: ARG001
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "03_edited.npy",
                {"joint_positions": base, "stage": {"generator": "dataset"}})
        return True, "dataset_fallback", {}

    tools2 = MotionTools(tmp / "fallback", generator="mld", skip_viz=True, goal=GOAL)
    tools2._draw_samples = lambda prompt, n, out_dir: []  # noqa: ARG005
    tools2._generate_into = fake_fallback.__get__(tools2, MotionTools)
    obs2 = tools2.resample_motion(GOAL, 2)
    check("fallback_is_reported", obs2.get("ok") is False
          and obs2.get("error") == "resample_unavailable", str(obs2.get("error")))
    check("fallback_keeps_budget", tools2.resamples == 0, f"resamples={tools2.resamples}")

    # 3) the budget is finite.
    tools3 = MotionTools(tmp / "budget", generator="mld", skip_viz=True, goal=GOAL)
    tools3._draw_samples = lambda prompt, n, out_dir: samples  # noqa: ARG005
    tools3.resample_motion(GOAL, 2)
    tools3.resample_motion(GOAL, 2)
    third = tools3.resample_motion(GOAL, 2)
    check("budget_is_capped", third.get("error") == "resample_budget_reached",
          str(third.get("error")))

    # 4) MLD's --allinone array must split into one clip per replication.
    from mld_generate import _split_samples  # noqa: E402

    five_d = np.stack([np.stack([np.zeros((20, 22, 3), np.float32)] * 3)] * 1)
    check("split_allinone_5d", len(_split_samples(five_d)) == 3
          and _split_samples(five_d)[0].shape == (20, 22, 3),
          f"shape_in={five_d.shape}")
    four_d = np.zeros((3, 20, 22, 3), np.float32)
    check("split_allinone_4d", len(_split_samples(four_d)) == 3,
          f"shape_in={four_d.shape}")

    # 5) the generation-side policy: what a draft is filtered on, and how a
    #    batch is sized (this is what stage_generate._mld_draft consults).
    import os

    import sampling  # noqa: E402

    names = [c["check"] for c in sampling.constraints("a person walks in place")]
    check("constraint_in_place", names == ["root_travel"], str(names))
    check("constraint_turn", "yaw_span_deg" in
          [c["check"] for c in sampling.constraints("a person turns left 90 degrees")],
          str(sampling.acceptance("a person turns left 90 degrees")["text"]))
    check("no_constraint_is_single_draw",
          sampling.batch_size("a person moves gracefully") == 1,
          f"n={sampling.batch_size('a person moves gracefully')}")
    check("constrained_goal_batches",
          sampling.batch_size("a person walks in place") >= 2,
          f"n={sampling.batch_size('a person walks in place')}")
    os.environ["AGENT_REJECTION"] = "off"
    check("rejection_can_be_disabled",
          sampling.batch_size("a person walks in place") == 1,
          "AGENT_REJECTION=off")
    os.environ.pop("AGENT_REJECTION", None)
    best, accepted = sampling.select(GOAL, [
        {"cand": 1, "ok": True, "valid": True, "metrics": {"root_travel": 1.4}},
        {"cand": 2, "ok": True, "valid": True, "metrics": {"root_travel": 0.05}},
        {"cand": 3, "ok": True, "valid": True, "metrics": {"root_travel": 0.4}},
    ])
    check("select_prefers_accepted", (best or {}).get("cand") == 2
          and len(accepted) == 1, f"best={best and best['cand']} n={len(accepted)}")

    report = {"checks": CHECKS, "passed": sum(c["ok"] for c in CHECKS), "n": len(CHECKS)}
    (tmp / "resample_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{report['passed']}/{report['n']} checks passed — {tmp / 'resample_checks.json'}")
    return 0 if report["passed"] == report["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
