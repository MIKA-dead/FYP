"""
Arm-raise test battery (Pipeline v1 geometric arm layer).

Three levels, because the failure that motivated the layer could hide in any of
them:

  unit      -- does a sentence become the right refine.arm_raise spec?
  geometry  -- does raise_arm() move only the requested arm, keep bone lengths,
               keep the rest of the skeleton and the root path identical?
  e2e       -- does run_session.py (planner + constraint_postprocess + quality
               gate + M2T caption) accept the revise and score it?

  python scripts/tests/test_arm_raise.py --mode all
  python scripts/tests/test_arm_raise.py --mode geometry --out outputs/tests/x

Writes <out>/results.json, <out>/report.md and <out>/contact_sheet.png.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "backends"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "intent"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session"))

import constraint_postprocess as cp  # noqa: E402
from refine_schema import normalize_refine  # noqa: E402
from visualize_motion import SKELETON_CHAINS, load_joints  # noqa: E402

UI_RUNS = REPO_ROOT / "outputs" / "session" / "ui_runs"

# (run dir, action) of the clips used as "the clip already on screen"
BASE_CLIPS = {
    "walk_slow": ("9ddc04a5_UI_20261102-221627", "walk_slow"),
    "walk_forward": ("bd6e6404_UI_20261028-113149", "walk_forward"),
    "run_forward": ("bd6e6404_UI_20261028-112445", "run_forward"),
    "bow": ("bd6e6404_UI_20261028-112710", "bow"),
    "sit_down": ("9e1f0296_UI_20261026-231857", "sit_down"),
    "wave_right": ("9e1f0296_UI_20261026-231556", "wave_right"),
}

# Joint indices that raise_arm() is allowed to touch: elbow + wrist of one arm.
_TOUCH_OK = {
    ("right",): {19, 21},
    ("left",): {18, 20},
    ("both",): {18, 19, 20, 21},
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clip(action: str) -> Tuple[np.ndarray, Path]:
    name, _ = BASE_CLIPS[action]
    run = UI_RUNS / name
    path = run / "03_edited.npy"
    joints, _kind = load_joints(path)
    return np.asarray(joints, dtype=np.float32), run


def _lift(joints: np.ndarray, side: str) -> Tuple[float, float]:
    """(wrist lift above shoulder, head lift above shoulder), p90."""
    up = cp._up_axis(joints)
    if side in ("left", "right"):
        sh, wr = cp._LIMB_CHAINS["arm"][side][0], cp._LIMB_CHAINS["arm"][side][-1]
    else:  # both -> the higher of the two, used only for change checks
        a = np.percentile(
            joints[:, cp._LIMB_CHAINS["arm"]["left"][-1], up]
            - joints[:, cp._LIMB_CHAINS["arm"]["left"][0], up],
            90,
        )
        b = np.percentile(
            joints[:, cp._LIMB_CHAINS["arm"]["right"][-1], up]
            - joints[:, cp._LIMB_CHAINS["arm"]["right"][0], up],
            90,
        )
        return float(a), float(b)
    lift = float(np.percentile(joints[:, wr, up] - joints[:, sh, up], 90))
    span = float(np.percentile(joints[:, cp.J_HEAD, up] - joints[:, sh, up], 90))
    return lift, span


def _ratio(joints: np.ndarray, side: str) -> float:
    lift, span = _lift(joints, side)
    return float(lift / max(span, 0.05))


def _bone_lens(joints: np.ndarray, side: str) -> np.ndarray:
    chain = cp._LIMB_CHAINS["arm"][side]
    sh, el, wr = chain
    upper = np.linalg.norm(joints[:, el] - joints[:, sh], axis=-1)
    fore = np.linalg.norm(joints[:, wr] - joints[:, el], axis=-1)
    return np.concatenate([upper, fore])


def _diff_joints(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Max abs coordinate change per joint, shape (22,)."""
    return np.abs(a - b).max(axis=(0, 2))


def _ok(cond: bool, detail: str) -> Dict[str, Any]:
    return {"ok": bool(cond), "detail": detail}


def _raise(joints: np.ndarray, spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = joints
    metas = []
    sides = ["left", "right"] if str(spec.get("side")) == "both" else [str(spec.get("side") or "right")]
    for side in sides:
        out, m = cp.raise_arm(out, {**spec, "side": side})
        metas.append(m)
    return out, (metas[0] if len(metas) == 1 else {"both": metas})


# --------------------------------------------------------------------------- #
# unit: prompt -> refine spec
# --------------------------------------------------------------------------- #
UNIT_CASES = [
    # prompt, expected arm_raise (None = must not fire), expected extra key
    ("raise the right arm high", {"side": "right", "level": "high"}, None),
    ("raise the left arm high overhead", {"side": "left", "level": "overhead"}, None),
    ("raise both arms overhead", {"side": "both", "level": "overhead"}, None),
    ("抬右手过头顶", {"side": "right", "level": "overhead"}, None),
    ("把左手抬高一点", {"side": "left", "level": "slightly"}, None),
    ("举双手过头顶", {"side": "both", "level": "overhead"}, None),
    ("raise the arm higher", {"step_up": True}, None),
    ("再抬高点", {"step_up": True}, None),
    # negative controls: must NOT become an arm raise
    ("slow down a bit", None, None),
    ("walk two more steps", None, None),
    ("no spinning", None, None),
    ("lift the right leg as much as the left", None, "limb_match"),
    ("raise the right arm as much as the left", None, "limb_match"),
]


def run_unit() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prompt, want, extra in UNIT_CASES:
        t0 = time.time()
        got = normalize_refine({}, raw_prompt=prompt)
        arm = got.get("arm_raise")
        checks = []
        if want is None:
            checks.append(_ok(arm is None, f"arm_raise={arm!r} (want none)"))
        else:
            checks.append(_ok(isinstance(arm, dict), f"arm_raise={arm!r}"))
            for k, v in want.items():
                if isinstance(v, bool):
                    checks.append(_ok(bool((arm or {}).get(k)) is v, f"{k}={v}"))
                else:
                    checks.append(_ok((arm or {}).get(k) == v, f"{k}={v}"))
            checks.append(
                _ok(
                    "geometric_only" in (got.get("ops") or []),
                    f"ops={got.get('ops')} (need geometric_only)",
                )
            )
        if extra == "limb_match":
            checks.append(_ok(isinstance(got.get("limb_match"), dict), "limb_match set"))
        rows.append(
            {
                "level": "unit",
                "id": prompt,
                "ok": all(c["ok"] for c in checks),
                "details": checks,
                "raw": {"arm_raise": arm, "ops": got.get("ops"), "limb_match": got.get("limb_match")},
                "seconds": round(time.time() - t0, 3),
            }
        )
    return rows


TEMPO_CASES = [
    # prompt, op list the LLM returned, expected wildcard scale
    ("slow down a bit", ["phase_speed"], 0.8),
    ("再慢一点", ["phase_speed"], 0.8),
    ("walk much faster", ["phase_speed"], 1.6),
]


def run_unit_tempo() -> List[Dict[str, Any]]:
    """The LLM sometimes names phase_speed but omits the map; wording decides."""
    rows: List[Dict[str, Any]] = []
    for prompt, ops, want in TEMPO_CASES:
        got = normalize_refine({"ops": list(ops)}, raw_prompt=prompt)
        ps = got.get("phase_speed") or {}
        scale = ps.get("*")
        checks = [
            _ok(scale is not None, f"phase_speed={ps} (want a wildcard entry)"),
            _ok(
                scale is not None and abs(float(scale) - want) < 1e-6,
                f"scale={scale} (want {want})",
            ),
            _ok("geometric_only" in (got.get("ops") or []), f"ops={got.get('ops')}"),
        ]
        rows.append(
            {
                "level": "unit",
                "kind": "tempo",
                "id": prompt,
                "ok": all(c["ok"] for c in checks),
                "details": checks,
                "raw": {"phase_speed": ps, "ops": got.get("ops")},
                "want": f"phase_speed['*']={want}",
                "seconds": 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# geometry: raise_arm on real clips
# --------------------------------------------------------------------------- #
def geom_case(
    cid: str,
    action: str,
    spec: Dict[str, Any],
    *,
    expect_ratio: Optional[float] = None,
    expect_changed: Optional[bool] = None,
) -> Dict[str, Any]:
    joints, run = _clip(action)
    side = str(spec.get("side") or "right")
    t0 = time.time()
    out, meta = _raise(joints, spec)
    secs = round(time.time() - t0, 3)
    checks: List[Dict[str, Any]] = []

    checks.append(_ok(out.shape == joints.shape, f"shape {joints.shape} -> {out.shape}"))
    checks.append(_ok(bool(np.isfinite(out).all()), "no NaN/Inf"))
    changed = bool((meta.get("changed") if side != "both" else any(m["changed"] for m in meta["both"])))
    if expect_changed is not None:
        checks.append(_ok(changed is expect_changed, f"changed={changed}"))

    # only the requested arm may move
    delta = _diff_joints(joints, out)
    allowed = set()
    for s in (["left", "right"] if side == "both" else [side]):
        allowed |= _TOUCH_OK[(s,)]
    moved = {int(i) for i in np.where(delta > 1e-6)[0]}
    checks.append(_ok(moved <= allowed, f"moved joints={sorted(moved)} allowed⊆{sorted(allowed)}"))

    # bone lengths preserved on the edited arm(s)
    for s in (["left", "right"] if side == "both" else [side]):
        before = _bone_lens(joints, s)
        after = _bone_lens(out, s)
        dev = float(np.abs(before - after).max()) if before.size else 0.0
        checks.append(_ok(dev < 2e-3, f"{s} bone length dev={dev:.5f}"))

    # root path must be identical (lower body untouched)
    root_dev = float(np.abs(joints[:, 0] - out[:, 0]).max())
    checks.append(_ok(root_dev < 1e-6, f"root dev={root_dev:.7f}"))

    # no snap: measure the *edit-induced* offset on frames where the source clip is
    # itself continuous. Several source clips contain a cut (a 0.3 m one-frame joint
    # jump), and no operator can be smoother than its input; those frames are
    # reported separately instead of counted as our failure.
    src_step = np.linalg.norm(np.diff(joints, axis=0), axis=-1).max(axis=1)
    glitch = src_step > 0.15
    offset = out - joints
    d_off = np.linalg.norm(np.diff(offset, axis=0), axis=-1).max(axis=1)
    clean = d_off[~glitch]
    peak_off = float(clean.max()) if clean.size else 0.0
    checks.append(
        _ok(
            peak_off < 0.12,
            f"peak edit-offset step={peak_off:.3f} on {int((~glitch).sum())} continuous frames "
            f"(want <0.12; {int(glitch.sum())} source-cut frames skipped)",
        )
    )

    # requested height reached
    for s in (["left", "right"] if side == "both" else [side]):
        ratio = _ratio(out, s)
        if expect_ratio is not None:
            checks.append(
                _ok(
                    ratio >= expect_ratio - 0.06,
                    f"{s} ratio={ratio:.2f} (target ≥{expect_ratio:.2f})",
                )
            )
    # untouched arm must not drift
    if side != "both":
        other = "left" if side == "right" else "right"
        idx = sorted(_TOUCH_OK[(other,)])
        other_dev = float(np.abs(joints[:, idx] - out[:, idx]).max())
        checks.append(_ok(other_dev < 1e-6, f"{other} arm dev={other_dev:.7f}"))

    return {
        "level": "geometry",
        "id": cid,
        "action": action,
        "spec": spec,
        "ok": all(c["ok"] for c in checks),
        "details": checks,
        "metrics": {
            "changed": changed,
            "ratio_before": round(_ratio(joints, side if side != "both" else "right"), 3),
            "ratio_after": round(_ratio(out, side if side != "both" else "right"), 3),
            "lift_before": round(_lift(joints, side if side != "both" else "right")[0], 4),
            "lift_after": round(_lift(out, side if side != "both" else "right")[0], 4),
            "moved_joints": sorted(moved),
        },
        "meta": {k: v for k, v in (meta.items() if side != "both" else [("both", meta)]) if k != "stages"},
        "seconds": secs,
    }


def chain_case(cid: str, action: str, levels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply several raises in a row; height must be monotone and bounded."""
    joints, _run = _clip(action)
    cur = joints
    side = str(levels[0].get("side") or "right")
    ratios: List[float] = [_ratio(cur, side)]
    bone_dev = 0.0
    for spec in levels:
        cur, _m = _raise(cur, spec)
        ratios.append(_ratio(cur, side))
        bone_dev = max(bone_dev, float(np.abs(_bone_lens(joints, side) - _bone_lens(cur, side)).max()))
    steps = np.diff(ratios[1:])
    checks = [
        _ok(bool(np.all(steps > 0.01)), f"ratios {['%.2f' % r for r in ratios]} strictly increase"),
        _ok(ratios[-1] <= 3.0, f"final ratio {ratios[-1]:.2f} ≤ 3.0 (no runaway)"),
        _ok(bool(np.isfinite(cur).all()), "no NaN/Inf"),
        _ok(bone_dev < 2e-3, f"bone length dev={bone_dev:.5f}"),
    ]
    return {
        "level": "geometry",
        "id": cid,
        "action": action,
        "ok": all(c["ok"] for c in checks),
        "details": checks,
        "metrics": {"ratios": [round(r, 3) for r in ratios], "bone_dev": round(bone_dev, 5)},
        "seconds": 0.0,
    }


def idempotence_case(cid: str, action: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    joints, _run = _clip(action)
    once, m1 = _raise(joints, spec)
    twice, m2 = _raise(once, spec)
    dev = float(np.abs(once - twice).max())
    changed2 = bool(m2.get("changed")) if "changed" in m2 else bool(m2["both"][0]["changed"])
    checks = [
        _ok(changed2 is False, f"second pass changed={changed2} (want False)"),
        _ok(dev < 1e-6, f"second pass max drift={dev:.7f}"),
        _ok(bool(np.isfinite(twice).all()), "no NaN/Inf"),
    ]
    return {
        "level": "geometry",
        "id": cid,
        "action": action,
        "spec": spec,
        "ok": all(c["ok"] for c in checks),
        "details": checks,
        "metrics": {"drift": dev},
        "seconds": 0.0,
    }


def run_geometry() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        geom_case("G1 walk + right arm high", "walk_slow", {"side": "right", "level": "high"}, expect_ratio=1.75)
    )
    rows.append(
        geom_case(
            "G2 walk + right arm overhead",
            "walk_slow",
            {"side": "right", "level": "overhead"},
            expect_ratio=2.45,
        )
    )
    rows.append(
        geom_case(
            "G3 walk + left arm slightly",
            "walk_slow",
            {"side": "left", "level": "slightly"},
            expect_ratio=0.75,
        )
    )
    rows.append(
        geom_case(
            "G4 walk + both arms overhead",
            "walk_slow",
            {"side": "both", "level": "overhead"},
            expect_ratio=2.40,
        )
    )
    rows.append(
        geom_case("G5 run + right arm high", "run_forward", {"side": "right", "level": "high"}, expect_ratio=1.0)
    )
    rows.append(
        geom_case("G6 bow + right arm high", "bow", {"side": "right", "level": "high"}, expect_ratio=1.0)
    )
    rows.append(
        geom_case(
            "G7 sit + both arms overhead",
            "sit_down",
            {"side": "both", "level": "overhead"},
            expect_ratio=2.40,
        )
    )
    rows.append(
        geom_case(
            "G8 wave clip + right arm overhead",
            "wave_right",
            {"side": "right", "level": "overhead"},
        )
    )
    rows.append(
        idempotence_case("G9 second identical raise is a no-op", "walk_slow", {"side": "right", "level": "high"})
    )
    rows.append(
        chain_case(
            "G10 high → higher → higher is monotone",
            "walk_slow",
            [
                {"side": "right", "level": "high"},
                {"side": "right", "level": "high", "step_up": True},
                {"side": "right", "level": "high", "step_up": True},
            ],
        )
    )
    return rows


# --------------------------------------------------------------------------- #
# e2e: planner -> backend -> quality -> caption
# --------------------------------------------------------------------------- #
E2E_CASES = [
    {
        "id": "E1 walk revise: raise the right arm high",
        "base": "walk_slow",
        "prompts": ["raise the right arm high"],
        "action": "walk_slow",
    },
    {
        "id": "E2 walk revise: raise both arms overhead",
        "base": "walk_slow",
        "prompts": ["raise both arms overhead"],
        "action": "walk_slow",
    },
    {
        "id": "E3 sit revise: raise both arms high",
        "base": "sit_down",
        "prompts": ["raise both arms high"],
        "action": "sit_down",
    },
    {
        "id": "E4 REGRESSION speed revise: slow down a bit",
        "base": "walk_slow",
        "prompts": ["slow down a bit"],
        "action": "walk_slow",
        "expect_slower": True,
    },
    {
        "id": "E5 walk revise: raise the left arm high",
        "base": "walk_slow",
        "prompts": ["raise the left arm high"],
        "action": "walk_slow",
    },
    {
        # multi-turn: the same clip takes two raises in a row
        "id": "E6 walk chain: high, then higher",
        "base": "walk_slow",
        "prompts": ["raise the right arm high", "raise the arm higher"],
        "action": "walk_slow",
        "monotone": True,
    },
    {
        "id": "E7 wave clip: arm already overhead",
        "base": "wave_right",
        "prompts": ["raise the right arm high overhead"],
        "action": "wave_right",
    },
]


def _run_one(
    py: str, prompt: str, prev_run: Path, run_dir: Path
) -> Tuple[subprocess.CompletedProcess, float]:
    t0 = time.time()
    proc = subprocess.run(
        [
            py,
            str(REPO_ROOT / "scripts" / "orchestrator" / "run_session.py"),
            "--mode",
            "revise",
            "--prompt",
            prompt,
            "--previous-run",
            str(prev_run),
            "--run-dir",
            str(run_dir),
            "--skip-viz",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, round(time.time() - t0, 1)


def _load(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


def _inherited_only(src: Path, run_dir: Path) -> bool:
    """True when the step only inherits the physical defects its source had.

    The e2e bases are UI sessions, and one of them ends with the legs below the
    floor: a hard `ground_clearance` failure that no arm edit can remove (its
    stored 04_quality.json predates that check, so the source is re-measured
    here rather than trusted). The check is meant to catch damage the step
    introduced, so an unchanged defect set is accepted.
    """
    from motion_validity import physical_report

    agent_dir = str(REPO_ROOT / "scripts" / "agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    from metrics import load_joints as _load_j

    def _invalid(d: Path) -> set:
        f = Path(d) / "03_edited.npy"
        return set(physical_report(_load_j(f))["invalid_checks"]) if f.is_file() else set()

    after = _invalid(run_dir)
    if not after:
        return False
    return after <= _invalid(src)


def run_e2e(out_root: Path, only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    py = sys.executable
    for case in E2E_CASES:
        cid = case["id"]
        if only and cid.split()[0] not in only:
            continue
        prompts: List[str] = case["prompts"]
        base_run = UI_RUNS / BASE_CLIPS[case["base"]][0]
        stem = cid.split()[0]
        prev = base_run
        step_rows: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        last: Dict[str, Any] = {}
        ratios: List[float] = []
        total_s = 0.0
        for i, prompt in enumerate(prompts, start=1):
            run_dir = out_root / "e2e" / stem / (f"step{i}" if len(prompts) > 1 else "")
            run_dir.mkdir(parents=True, exist_ok=True)
            proc, secs = _run_one(py, prompt, prev, run_dir)
            total_s += secs
            q = _load(run_dir / "04_quality.json")
            cap = _load(run_dir / "06_motion_caption.json")
            cmeta = _load(run_dir / "03_edited.constraint.json")
            plan = _load(run_dir / "00_motion_edit_plan.json")

            step_checks = [
                _ok(
                    proc.returncode == 0
                    or _inherited_only(prev if i > 1 else base_run, run_dir),
                    f"step{i} exit={proc.returncode}",
                ),
                _ok(
                    bool(q.get("passed")) or _inherited_only(prev if i > 1 else base_run, run_dir),
                    f"step{i} quality passed={q.get('passed')} score={q.get('score')}",
                ),
                _ok(
                    str(plan.get("primary_action") or "") == case["action"],
                    f"step{i} action={plan.get('primary_action')!r} (want {case['action']!r})",
                ),
                _ok(bool(cap.get("caption")), f"step{i} caption={cap.get('caption')!r}"),
            ]
            if "raise" in prompt:
                arm_meta = cmeta.get("arm_raise") or {}
                step_checks.append(
                    _ok(plan.get("edit_mode") == "geometric_only", f"step{i} edit_mode={plan.get('edit_mode')}")
                )
                lift = (q.get("metrics") or {}).get("arm_raise_lift")
                step_checks.append(_ok(float(lift or 0) > 0.02, f"step{i} arm_raise_lift={lift}"))
                # a further raise must actually raise (step_up), not no-op
                if "higher" in prompt:
                    step_checks.append(
                        _ok(
                            bool((arm_meta.get("changed"))),
                            f"step{i} step_up applied (changed={arm_meta.get('changed')})",
                        )
                    )
            checks += step_checks
            ratio = (q.get("metrics") or {}).get("arm_raise_ratio")
            if ratio is not None:
                ratios.append(float(ratio))
            step_rows.append(
                {
                    "step": i,
                    "prompt": prompt,
                    "run_dir": str(run_dir),
                    "score": q.get("score"),
                    "caption": cap.get("caption"),
                    "arm_raise_ratio": ratio,
                    "arm_raise_lift": (q.get("metrics") or {}).get("arm_raise_lift"),
                    "edit_mode": plan.get("edit_mode"),
                    "changed": (cmeta.get("arm_raise") or {}).get("changed"),
                    "seconds": secs,
                    "ok": all(c["ok"] for c in step_checks),
                }
            )
            last = {"q": q, "cap": cap, "plan": plan, "run_dir": run_dir, "proc": proc}
            prev = run_dir

        if case.get("monotone") and len(ratios) >= 2:
            checks.append(
                _ok(
                    ratios[-1] > ratios[0] + 0.1,
                    f"ratio grows across turns: {['%.2f' % r for r in ratios]}",
                )
            )
        if case.get("expect_slower") and last.get("run_dir"):
            tp = _load(Path(last["run_dir"]) / "03_edited.timeline_post.json")
            t_in, t_out = tp.get("T_in"), tp.get("T_out")
            checks.append(
                _ok(
                    bool(t_in and t_out and int(t_out) > int(t_in)),
                    f"tempo applied: T {t_in} → {t_out} (slower = more frames)",
                )
            )
            checks.append(
                _ok(
                    "slow" in str((last.get("cap") or {}).get("caption") or "").lower(),
                    f"caption mentions slow: {(last.get('cap') or {}).get('caption')!r}",
                )
            )
        rows.append(
            {
                "level": "e2e",
                "id": cid,
                "prompts": prompts,
                "base": BASE_CLIPS[case["base"]][0],
                "run_dir": str(last.get("run_dir")),
                "steps": step_rows,
                "ok": all(c["ok"] for c in checks),
                "details": checks,
                "metrics": {
                    "score": (last.get("q") or {}).get("score"),
                    "passed": (last.get("q") or {}).get("passed"),
                    "hard_failures": (last.get("q") or {}).get("hard_failures"),
                    "edit_mode": (last.get("plan") or {}).get("edit_mode"),
                    "caption": (last.get("cap") or {}).get("caption"),
                    "caption_source": (last.get("cap") or {}).get("source"),
                    "arm_raise_lift": ((last.get("q") or {}).get("metrics") or {}).get("arm_raise_lift"),
                    "arm_raise_ratio": ((last.get("q") or {}).get("metrics") or {}).get("arm_raise_ratio"),
                    "arm_raise_ratios": [round(r, 2) for r in ratios],
                    "root_travel": ((last.get("q") or {}).get("metrics") or {}).get("root_travel"),
                },
                "seconds": round(total_s, 1),
                "stderr_tail": ((last.get("proc").stderr if last.get("proc") else "") or "")[-600:],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# contact sheet
# --------------------------------------------------------------------------- #
def contact_sheet(
    action: str, spec: Dict[str, Any], out_png: Path, frames: Optional[List[int]] = None
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joints, _run = _clip(action)
    raised, _m = _raise(joints, spec)
    up = cp._up_axis(joints)
    hz = [0, 1, 2] if up == 2 else [0, 2, 1]
    va, vb = hz[0], hz[2]

    # center on the first frame root; same window for both rows
    def centred(j: np.ndarray) -> np.ndarray:
        c = j.copy()
        c[:, :, va] -= j[0, 0, va]
        c[:, :, vb] -= j[0, 0, vb]
        c[:, :, up] -= j[0, 0, up]
        return c

    a, b = centred(joints), centred(raised)
    allp = np.concatenate([a, b], axis=0)
    lo = allp.min(axis=(0, 1))
    hi = allp.max(axis=(0, 1))
    span = float((hi - lo).max())
    mid = (lo + hi) / 2.0
    if frames is None:
        T = joints.shape[0]
        frames = [int(T * f) for f in (0.35, 0.55, 0.75, 0.92)]

    fig = plt.figure(figsize=(13, 6.2), dpi=130)
    titles = [f"before — {action}", f"after — raise {spec.get('side')} arm ({spec.get('level')})"]
    for row, (data, title) in enumerate(((a, titles[0]), (b, titles[1]))):
        for col, f in enumerate(frames):
            ax = fig.add_subplot(2, len(frames), row * len(frames) + col + 1, projection="3d")
            j = data[f]
            for chain in SKELETON_CHAINS:
                pts = j[chain]
                ax.plot(pts[:, va], pts[:, vb], pts[:, up], "o-", lw=2, markersize=3)
            ax.set_xlim(mid[va] - span / 2, mid[va] + span / 2)
            ax.set_ylim(mid[vb] - span / 2, mid[vb] + span / 2)
            ax.set_zlim(lo[up], hi[up])
            ax.view_init(elev=8, azim=-88)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1.5))
            if row == 0:
                ax.set_title(f"frame {f}", fontsize=9)
            if col == 0:
                ax.text2D(-0.12, 0.5, title, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=10)
    fig.suptitle(f"{action}: arm raise is applied on the same clip (root path and legs unchanged)", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(rows: List[Dict[str, Any]], out_root: Path, extra: Dict[str, Any]) -> Path:
    lines = ["# Arm-raise test results", ""]
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- python: {sys.version.split()[0]}")
    lines.append("")
    for level in ("unit", "geometry", "e2e"):
        sub = [r for r in rows if r["level"] == level]
        if not sub:
            continue
        passed = sum(1 for r in sub if r["ok"])
        lines.append(f"## {level}: {passed}/{len(sub)} passed")
        lines.append("")
        if level == "unit":
            lines.append("| prompt | expect | got | ok |")
            lines.append("|---|---|---|---|")
            for r in sub:
                raw = r["raw"]
                if r.get("kind") == "tempo":
                    got = raw.get("phase_speed")
                    want = r.get("want")
                else:
                    got = raw.get("arm_raise") or (raw.get("limb_match") and "limb_match") or "—"
                    want = next(
                        (c["detail"] for c in r["details"] if "want none" not in c["detail"]),
                        "none",
                    )
                lines.append(f"| `{r['id']}` | {want} | `{got}` | {'PASS' if r['ok'] else 'FAIL'} |")
        elif level == "geometry":
            lines.append("| case | clip | ratio before → after | moved joints | ok | checks |")
            lines.append("|---|---|---|---|---|---|")
            for r in sub:
                m = r.get("metrics") or {}
                rb = m.get("ratio_before", "—")
                ra = m.get("ratio_after", "—")
                moved = m.get("moved_joints")
                moved_s = "—" if moved is None else ",".join(str(x) for x in moved)
                failed = [c["detail"] for c in r["details"] if not c["ok"]]
                lines.append(
                    f"| {r['id']} | {r.get('action', '')} | {rb} → {ra} | {moved_s} | "
                    f"{'PASS' if r['ok'] else 'FAIL'} | {'; '.join(failed) if failed else 'all checks'} |"
                )
        else:
            lines.append("| case | steps | score | edit_mode | caption (M2T) | arm ratio | seconds | ok |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in sub:
                m = r["metrics"]
                lines.append(
                    f"| {r['id']} | {len(r.get('prompts') or [])} | {m.get('score')} | {m.get('edit_mode')} | "
                    f"{m.get('caption')} | {m.get('arm_raise_ratios') or m.get('arm_raise_ratio')} | "
                    f"{r['seconds']} | {'PASS' if r['ok'] else 'FAIL'} |"
                )
        lines.append("")
        for r in sub:
            if not r["ok"]:
                lines.append(f"**FAIL {r['id']}**")
                for c in r["details"]:
                    if not c["ok"]:
                        lines.append(f"- {c['detail']}")
                if r.get("stderr_tail"):
                    lines.append("```")
                    lines.append(r["stderr_tail"].strip())
                    lines.append("```")
                lines.append("")
    if extra.get("sheet"):
        lines.append(f"![contact sheet]({extra['sheet']})")
        lines.append("")
    out = out_root / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Arm-raise test battery")
    ap.add_argument("--mode", choices=["unit", "geometry", "e2e", "all"], default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--e2e-only", type=str, default="", help="comma separated ids, e.g. E1,E4")
    args = ap.parse_args()

    out_root = args.out or (REPO_ROOT / "outputs" / "tests" / f"arm_raise_{time.strftime('%Y%m%d-%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if args.mode in ("unit", "all"):
        rows += run_unit()
        rows += run_unit_tempo()
        print(f"[tests] unit done ({sum(1 for r in rows if r['ok'])}/{len(rows)})", flush=True)
    if args.mode in ("geometry", "all"):
        geo = run_geometry()
        rows += geo
        print(f"[tests] geometry done ({sum(1 for r in geo if r['ok'])}/{len(geo)})", flush=True)
    if args.mode in ("e2e", "all"):
        only = [x.strip() for x in args.e2e_only.split(",") if x.strip()] or None
        e2e = run_e2e(out_root, only=only)
        rows += e2e
        print(f"[tests] e2e done ({sum(1 for r in e2e if r['ok'])}/{len(e2e)})", flush=True)

    extra: Dict[str, Any] = {}
    try:
        sheet = contact_sheet(
            "walk_slow", {"side": "right", "level": "overhead"}, out_root / "contact_sheet.png"
        )
        extra["sheet"] = str(sheet)
        print(f"[tests] sheet {sheet}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tests] sheet failed: {exc}", flush=True)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total": len(rows),
            "passed": sum(1 for r in rows if r["ok"]),
            "failed": sum(1 for r in rows if not r["ok"]),
            "by_level": {
                lv: {
                    "total": sum(1 for r in rows if r["level"] == lv),
                    "passed": sum(1 for r in rows if r["level"] == lv and r["ok"]),
                }
                for lv in ("unit", "geometry", "e2e")
            },
        },
        "rows": rows,
    }
    (out_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(rows, out_root, extra)
    print(f"[tests] results {out_root / 'results.json'}")
    print(f"[tests] report  {report}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
