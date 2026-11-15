"""
The tools the LLM controller may call.

    generate_motion(prompt)      (a) MLD text→motion (falls back to the fast
                                     dataset draft if MLD is unavailable)
    describe_motion()            (c) TM2T motion→text + measured kinematics
    check_requirement()          measure the request against the joints
    edit_motion(instruction)     (b) one relative edit via the session orchestrator
    finish(reason)               stop

The tools own the run directory, so the controller only has to write the
request/instruction text. Every call returns a compact JSON observation that is
appended to the conversation and written to ``agent_trace.json``.

Two postconditions guard the state the controller is allowed to see:

* an instruction outside the editor's ability is rejected *before* the backend
  runs (``edits.classify``), instead of costing a 15 s no-op;
* a clip is grounded when it is rigidly offset (a restored source with its root
  at the origin) and rolled back when it is genuinely deforming through the
  floor. Both use ``motion_validity``, the same module the quality gate reads.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "scripts" / "session",
    REPO / "scripts" / "orchestrator",
    REPO / "scripts" / "backends",
    REPO / "scripts" / "agent",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edits import capability_text, classify  # noqa: E402
from metrics import (  # noqa: E402
    infer_target_joints,
    instruction_following,
    kinematic_summary,
    load_joints,
    preservation,
    text_similarity,
)
from motion_validity import physical_report, reground  # noqa: E402
from verify import requirement_gap, verify_goal  # noqa: E402

MAX_RESAMPLES = 2


def _candidate_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """The handful of measurements a draft is ranked on, kept short for the LLM."""
    keys = (
        "root_travel", "yaw_span_deg", "pelvis_drop",
        "arm_raise_left", "arm_raise_right",
        "ankle_lift_left", "ankle_lift_right",
        "jump_rise", "jump_clearance", "ground_penetration",
    )
    return {k: m.get(k) for k in keys if m.get(k) is not None}

TOOL_SPECS = [
    {
        "name": "generate_motion",
        "description": (
            "Text→motion. Call once at the start with the user's request, lightly "
            "rewritten as a plain motion sentence (keep every requested detail)."
        ),
        "args": {"prompt": "str — the text to generate"},
    },
    {
        "name": "describe_motion",
        "description": (
            "Motion→text (learned M2T model) plus measured joint kinematics and a "
            "similarity score against the goal. Call this before judging any clip, "
            "and again after every edit."
        ),
        "args": {},
    },
    {
        "name": "check_requirement",
        "description": (
            "Measure the user's request against the current clip and return "
            "pass/fail per requirement (arm height, travel, sit depth, turn, "
            "physical validity). Call once before finishing; its verdict is "
            "recorded next to your own judgment."
        ),
        "args": {},
    },
    {
        "name": "resample_motion",
        "description": (
            "Draw several independent drafts of the same prompt and keep the one "
            "closest to the goal. Use this when the checker says a measurable "
            "attribute is missing (too little travel, too little turn) and no edit "
            "can add it — the generator is stochastic, so one draft cannot tell a "
            "capability limit from bad luck. Costs one step, not an edit. At most "
            "2 per episode."
        ),
        "args": {
            "prompt": "str — the text to draw (defaults to the user goal)",
            "n": "int — 2..4 drafts, default 3",
        },
    },
    {
        "name": "edit_motion",
        "description": (
            "Apply exactly one relative edit to the current clip. The instruction is "
            "free text but must be one change the backend supports (see capabilities)."
        ),
        "args": {"instruction": "str — e.g. 'raise the right arm higher'"},
    },
    {
        "name": "finish",
        "description": "Stop when the measured motion satisfies the user, or the budget is exhausted.",
        "args": {"reason": "str — one sentence on why you stop"},
    },
]

SUPPORTED_EDITS = capability_text()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


def _read_quality(run_dir: Optional[Path], key: Optional[str] = None):
    """Read the session gate report next to a clip (None if not written)."""
    if run_dir is None:
        return None
    p = Path(run_dir) / "04_quality.json"
    if not p.is_file():
        return None
    try:
        q = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None
    return q.get(key) if key else q


class MotionTools:
    """Stateful tool environment for one agent episode."""

    def __init__(self, episode_dir: Path, *, generator: str = "mld",
                 skip_viz: bool = False, goal: str = "") -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.skip_viz = skip_viz
        self.goal = goal
        self.run_dir: Optional[Path] = None
        self.step = 0
        self.described_at: Optional[int] = None
        self.checked_at: Optional[int] = None
        self.edits = 0
        self.resamples = 0
        self.calls: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------
    def _next_dir(self, tag: str) -> Path:
        self.step += 1
        d = self.episode_dir / f"step{self.step:02d}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _caption(self, run_dir: Path, goal: str) -> Dict[str, Any]:
        """Refresh the caption with the learned M2T model and score the text."""
        from motion_caption import caption_edited

        quality = _read_json(run_dir / "04_quality.json")
        plan = _read_json(run_dir / "00_motion_edit_plan.json")
        summary = _read_json(run_dir / "summary.json")
        edited = run_dir / "03_edited.npy"
        cap = caption_edited(
            edited,
            quality=quality,
            action=str(plan.get("primary_action") or quality.get("action") or ""),
            intent_text=goal,
            run_dir=run_dir,
        )
        (run_dir / "06_motion_caption.json").write_text(
            json.dumps(cap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cap["similarity"] = text_similarity(goal, str(cap.get("caption") or ""))
        return cap

    def _observe_clip(self, run_dir: Path, goal: str) -> Dict[str, Any]:
        edited = run_dir / "03_edited.npy"
        if not edited.is_file():
            return {"ok": False, "error": f"no clip in {run_dir}"}
        j = load_joints(edited)
        quality = _read_json(run_dir / "04_quality.json")
        plan = _read_json(run_dir / "00_motion_edit_plan.json")
        stages = _read_json(run_dir / "03_edited.stages.json")
        draft = stages.get("draft") or {}
        m = kinematic_summary(j)
        rep = physical_report(j)
        return {
            "run_dir": str(run_dir),
            "frames": int(j.shape[0]),
            "preview": str(next(iter(run_dir.glob("*preview*.gif")), "")),
            "generator": draft.get("generator") or "dataset",
            "source_policy": plan.get("source_policy"),
            "primary_action": plan.get("primary_action"),
            "quality_score": quality.get("score"),
            "quality_passed": quality.get("passed"),
            "quality_hard_failures": quality.get("hard_failures") or [],
            "valid": bool(rep["valid"]),
            "invalid_checks": rep["invalid_checks"],
            "ground": rep["ground"],
            "metrics": m,
        }
    # -- tools -------------------------------------------------------------
    def _generate_into(self, prompt: str, out_dir: Path,
                       draft_joints: Optional[Path] = None) -> tuple:
        """Run the generator into ``out_dir``; returns (ok, generator_used, resp)."""
        from session_api import run_session

        req = {
            "prompt": str(prompt),
            "mode": "generate",
            "run_dir": str(out_dir),
            "skip_review": True,
            "skip_viz": self.skip_viz,
            "backend": "orchestrator",
            "generator": self.generator,
        }
        if draft_joints is not None:
            # Already drawn (one of several MLD samples): skip diffusion and let
            # the pipeline observe/edit/visualise this exact clip.
            req["draft_joints"] = str(draft_joints)
        resp = run_session(req, cwd=REPO)
        used = self.generator
        if not (out_dir / "03_edited.npy").is_file() and self.generator == "mld":
            # MLD is slow/heavy; keep the episode alive with the dataset draft.
            resp = run_session({**req, "generator": "dataset"}, cwd=REPO)
            used = "dataset_fallback"
        return (out_dir / "03_edited.npy").is_file(), used, resp

    def _draw_samples(self, prompt: str, n: int, out_dir: Path) -> List[Path]:
        """Draw ``n`` samples of one prompt with a single model load.

        MLD's demo already loops ``cfg.DEMO.REPLICATION`` times over the
        denoiser, so asking for ``--replication n --allinone`` costs one ~30 s
        model load instead of ``n``. The samples share a text and a length, so
        any difference between them is sampling noise — exactly the variance a
        resample is meant to explore.
        """
        import os
        import subprocess

        py = os.environ.get("MLD_PYTHON") or str(
            Path.home() / "miniconda3" / "envs" / "mld" / "python.exe"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        first = out_dir / "mld_sample1.npy"
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        try:
            proc = subprocess.run(
                [py, str(REPO / "scripts" / "backends" / "mld_generate.py"),
                 "--text", str(prompt), "--length", "60",
                 "--out", str(first), "--samples", str(n)],
                cwd=str(REPO), env=env, timeout=1800,
            )
        except Exception:  # noqa: BLE001
            return []
        if proc.returncode != 0:
            return []
        meta_path = first.with_suffix(".json")
        if not meta_path.is_file():
            return []
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            return []
        return [Path(p) for p in (meta.get("samples") or []) if Path(p).is_file()]

    def generate_motion(self, prompt: str) -> Dict[str, Any]:
        t0 = time.time()
        out_dir = self._next_dir("generate")
        ok, used, resp = self._generate_into(str(prompt), out_dir)
        if not ok:
            obs = {"ok": False, "error": resp.get("error") or "generation produced no clip"}
        else:
            self.run_dir = out_dir
            self.described_at = None
            self.checked_at = None
            obs = self._observe_clip(out_dir, str(prompt))
            obs.update({
                "ok": True,
                "requested_prompt": prompt,
                "generator_used": used,
                "seconds": round(time.time() - t0, 2),
                "note": "No caption yet — call describe_motion before judging.",
            })
        self._record("generate_motion", {"prompt": prompt}, obs)
        return obs

    def resample_motion(self, prompt: str = "", n: int = 3) -> Dict[str, Any]:
        """Draw ``n`` drafts of one prompt and keep the one closest to the goal.

        One draft cannot separate "this generator cannot do it" from "this draw
        missed". Every candidate is measured with the same requirement the
        checker uses, the best is adopted (valid first, then lowest relative
        shortfall), and all of them are reported so the controller sees what it
        rejected instead of guessing from a single text.
        """
        goal = str(prompt or self.goal or "")
        if self.resamples >= MAX_RESAMPLES:
            return {
                "ok": False,
                "error": "resample_budget_reached",
                "note": f"{MAX_RESAMPLES} resamples already used. Edit, accept what "
                        "you have, or finish naming the gap.",
            }
        n = max(2, min(4, int(n or 3)))
        self.step += 1
        base = self.episode_dir / f"step{self.step:02d}_resample"
        base.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        drawn = self._draw_samples(goal, n, base / "draw") if self.generator == "mld" else []
        cands: List[Dict[str, Any]] = []
        for i in range(1, n + 1):
            d = base / f"cand{i}"
            d.mkdir(parents=True, exist_ok=True)
            sample = drawn[i - 1] if i - 1 < len(drawn) else None
            ok, used, resp = self._generate_into(goal, d, draft_joints=sample)
            if not ok:
                cands.append({"cand": i, "run_dir": str(d), "ok": False,
                              "error": resp.get("error") or "no clip"})
                continue
            j = load_joints(d / "03_edited.npy")
            m = kinematic_summary(j)
            rep = physical_report(j)
            v = verify_goal(goal, {"metrics": m})
            cands.append({
                "cand": i,
                "run_dir": str(d),
                "ok": True,
                "generator_used": used,
                "from_generator": sample is not None,
                "valid": bool(rep["valid"]),
                "satisfied": bool(v["ok"]),
                "reasons": v["reasons"],
                "gap": round(requirement_gap(goal, {"metrics": m}), 4),
                "measured": _candidate_metrics(m),
            })
        usable = [c for c in cands if c.get("ok")]
        if not usable:
            obs = {"ok": False, "error": "resample produced no clip",
                   "candidates": cands, "seconds": round(time.time() - t0, 2)}
            self._record("resample_motion", {"prompt": goal, "n": n}, obs)
            return obs
        if not any(c.get("from_generator") for c in usable):
            # No candidate came from a fresh draw: either the diffusion
            # generator is down (each run fell back to the retrieved clip) or
            # this path is deterministic. Extra draws would repeat the same clip,
            # so say that instead of charging the budget for a no-op.
            obs = {
                "ok": False,
                "error": "resample_unavailable",
                "candidates": cands,
                "seconds": round(time.time() - t0, 2),
                "note": (
                    "No draft came from the diffusion generator (all candidates "
                    "reused the deterministic retrieved clip), so another resample "
                    "would return the same motion. Edit, change the prompt, or "
                    "finish naming the gap. If the generator should be running, "
                    "check its environment (MLD_PYTHON / the `mld` conda env)."
                ),
            }
            self._record("resample_motion", {"prompt": goal, "n": n}, obs)
            return obs
        usable.sort(key=lambda c: (not c["valid"], c["gap"], not c["satisfied"]))
        best = usable[0]
        self.resamples += 1
        self.run_dir = Path(best["run_dir"])
        self.described_at = None
        self.checked_at = None
        distinct = len({json.dumps(c.get("measured"), sort_keys=True) for c in usable})
        obs = self._observe_clip(self.run_dir, goal)
        obs.update({
            "ok": True,
            "candidates": cands,
            "chosen": best["cand"],
            "chosen_gap": best["gap"],
            "chosen_satisfied": best["satisfied"],
            "distinct_candidates": distinct,
            "resamples_left": max(0, MAX_RESAMPLES - self.resamples),
            "seconds": round(time.time() - t0, 2),
            "note": (
                "Kept the closest draft (lowest requirement gap, valid first). "
                "No caption for it yet — call describe_motion, then compare it with "
                "the candidates list before deciding."
                if distinct > 1
                else "Every draft measured identically, so another draw of this "
                     "prompt will only repeat it. Edit, change the prompt, or "
                     "finish naming the gap."
            ),
        })
        self._record("resample_motion", {"prompt": goal, "n": n}, obs)
        return obs

    def describe_motion(self, goal: str) -> Dict[str, Any]:
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        t0 = time.time()
        cap = self._caption(self.run_dir, goal)
        obs = self._observe_clip(self.run_dir, goal)
        obs.update({
            "ok": True,
            "caption": cap.get("caption"),
            "caption_source": cap.get("source"),
            "caption_problems": cap.get("problems") or [],
            "similarity": cap.get("similarity"),
            "seconds": round(time.time() - t0, 2),
        })
        self.described_at = self.step
        self._record("describe_motion", {}, obs)
        return obs

    def check_requirement(self, goal: str) -> Dict[str, Any]:
        """Measure the request against the current clip (the verifier as a tool)."""
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        j = load_joints(self.run_dir / "03_edited.npy")
        m = kinematic_summary(j)
        rep = physical_report(j)
        verdict = verify_goal(goal, {"metrics": m})
        obs = {
            "ok": True,
            "satisfied": bool(verdict["ok"]),
            "checks": verdict["checks"],
            "reasons": verdict["reasons"],
            "measured": {
                k: m.get(k) for k in (
                    "arm_raise_right", "arm_raise_left",
                    "wrist_above_head_right", "wrist_above_head_left",
                    "root_travel", "pelvis_drop", "yaw_span_deg",
                    "ankle_lift_right", "ankle_lift_left",
                    "jump_rise", "jump_clearance",
                    "ground_penetration", "bone_cv",
                )
            },
            "physically_valid": bool(rep["valid"]),
            "quality_hard_failures": _read_quality(self.run_dir, key="hard_failures") or [],
            "note": (
                "Measurements meet the request. You may finish."
                if verdict["ok"]
                else "Measurements do NOT meet the request. You may still finish, "
                     "but say which check fails and why you stop."
            ),
        }
        self.checked_at = self.step
        self._record("check_requirement", {}, obs)
        return obs

    def edit_motion(self, instruction: str) -> Dict[str, Any]:
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        cls = classify(instruction)
        if not cls["supported"]:
            # Reject before spending a backend pass: the editor cannot apply this.
            obs = {
                "ok": False,
                "error": "unsupported_instruction",
                "instruction": str(instruction),
                "reason": cls["reason"],
                "supported_family": cls["family"],
                "suggestion": cls.get("suggestion"),
                "note": "Do not repeat this instruction. Use a supported one, or "
                        "finish and report that the request is out of scope.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs

        from session_api import run_session

        t0 = time.time()
        before_dir = self.run_dir
        before = load_joints(before_dir / "03_edited.npy")
        # A relative locomotion operator only extends travel that already exists.
        # On a clip that walks in place it would rebuild the clip and drop other
        # motion (a 186° turn, in ep13), so refuse it here and point to the
        # generator instead of spending a pass that damages the clip.
        if cls["family"] == "walk":
            m0 = kinematic_summary(before)
            travel0 = float(m0.get("root_travel") or 0)
            if travel0 < 0.15:
                obs = {
                    "ok": False,
                    "error": "edit_needs_existing_locomotion",
                    "instruction": str(instruction),
                    "root_travel": round(travel0, 3),
                    "reason": "this clip has almost no forward travel, and the "
                              "walk operator only scales travel that is already "
                              "there",
                    "suggestion": "call generate_motion with the full goal "
                                  "(mention walking forward in the prompt)",
                    "note": "Do not repeat this instruction on this clip.",
                    "seconds": round(time.time() - t0, 2),
                }
                self._record("edit_motion", {"instruction": instruction}, obs)
                return obs
        out_dir = self._next_dir("edit")
        resp = run_session(
            {
                "prompt": str(instruction),
                "mode": "revise",
                "previous_run": str(before_dir),
                "run_dir": str(out_dir),
                "skip_review": True,
                "skip_viz": self.skip_viz,
                "backend": "orchestrator",
            },
            cwd=REPO,
        )
        if not (out_dir / "03_edited.npy").is_file():
            obs = {"ok": False, "error": resp.get("error") or "edit produced no clip"}
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs

        self.run_dir = out_dir
        self.described_at = None
        self.checked_at = None
        self.edits += 1
        after_f = out_dir / "03_edited.npy"
        after = load_joints(after_f)
        # A restored/diffused source can come back with its root at the origin
        # instead of on the floor. That is a rigid offset: repair it. Deforming
        # penetration is not repaired — see physical_report().
        grounded_offset = 0.0
        rep = physical_report(after)
        if rep["ground"]["mode"] == "offset":
            fixed, grounded_offset = reground(after)
            if grounded_offset:
                np.save(after_f, fixed)
                after = fixed
                rep = physical_report(after)
        kb = kinematic_summary(before)
        ka = kinematic_summary(after)
        rep_before = physical_report(before)
        # Only *damage* justifies a rollback. If the source was already invalid
        # (a sit draft whose legs sink through the floor, say), refusing every
        # edit would lock the agent out of improving it — report the inherited
        # defect instead, and still roll back the moment the edit adds one.
        introduced = [c for c in rep["invalid_checks"] if c not in rep_before["invalid_checks"]]
        if not rep["valid"] and (rep_before["valid"] or introduced):
            # A broken clip must not become the state the agent judges.
            self.run_dir = before_dir
            self.described_at = None
            self.checked_at = None
            obs = {
                "ok": False,
                "error": "edit_produced_invalid_clip",
                "invalid": rep["invalid_checks"],
                "ground": rep["ground"],
                "rolled_back": True,
                "instruction": str(instruction),
                "family": cls["family"],
                "before": kb,
                "after": ka,
                "seconds": round(time.time() - t0, 2),
                "note": "The edit damaged the clip; the previous clip is restored. "
                        "Try a different instruction.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
        target, region = infer_target_joints(instruction)
        pres = preservation(before, after, target)
        follow = instruction_following(instruction, kb, ka, pres)
        # The session gate may already fail on the source clip (a static sit
        # draft, for instance). An edit must not be blamed for a defect it did
        # not introduce, so report only the delta and let the controller decide.
        q_before = _read_quality(before_dir)
        q_after = _read_quality(out_dir)
        fails_before = set((q_before or {}).get("hard_failures") or [])
        fails_after = set((q_after or {}).get("hard_failures") or [])
        new_fails = sorted(fails_after - fails_before)
        inherited_fails = sorted(fails_after & fails_before)
        no_change = (
            pres.get("target_change", 0.0) <= 1e-4
            and pres.get("non_target_mpjpe", 0.0) <= 1e-4
            and after.shape == before.shape
        )
        frames_ratio = round(float(after.shape[0]) / max(1, int(before.shape[0])), 3)
        # A "lower the seat" edit that silently drops half the clip is not a
        # depth edit. Trimming families are allowed to shorten.
        shrunk = (
            frames_ratio < 0.70 and cls["family"] not in ("trim_trailing", "sit_remove_preamble")
        )
        obs = {
            "ok": True,
            "instruction": str(instruction),
            "family": cls["family"],
            "run_dir": str(out_dir),
            "frames": int(after.shape[0]),
            "frames_ratio": frames_ratio,
            "preview": str(next(iter(out_dir.glob("*preview*.gif")), "")),
            "region": region,
            "grounded_offset": round(grounded_offset, 3),
            "ground": rep["ground"],
            "before": kb,
            "after": ka,
            "preservation": pres,
            "instruction_following": follow,
            "no_change": bool(no_change),
            "physically_valid": bool(rep["valid"]),
            "inherited_invalidity": [c for c in rep["invalid_checks"]],
            "quality_passed": (q_after or {}).get("passed"),
            "new_hard_failures": new_fails,
            "pre_existing_hard_failures": inherited_fails,
            "seconds": round(time.time() - t0, 2),
            "note": (
                "The clip is byte-for-byte the same: the backend cannot apply this "
                "instruction. Do not repeat it; try another supported edit or finish."
                if no_change
                else (
                    f"The edit kept only {frames_ratio:.0%} of the frames — that is a "
                    "truncation, not the change you asked for. Check whether the action "
                    "is still complete before you accept it."
                    if shrunk
                    else (
                        "The edit introduced new quality-gate failures: "
                        f"{', '.join(new_fails)}. Consider a different instruction."
                        if new_fails
                        else (
                            "The clip was already invalid before this edit "
                            f"({', '.join(rep['invalid_checks'])}); the edit did not make it "
                            "worse. The defect is inherited from the source, not caused here."
                            if rep["invalid_checks"]
                            else "No caption of the new clip yet — call describe_motion."
                        )
                    )
                )
            ),
        }
        self._record("edit_motion", {"instruction": instruction}, obs)
        return obs

    def finish(self, reason: str = "") -> Dict[str, Any]:
        obs = {"ok": True, "finished": True, "reason": reason, "run_dir": str(self.run_dir or "")}
        self._record("finish", {}, obs)
        return obs

    # -- trace -------------------------------------------------------------
    def _record(self, action: str, args: Dict[str, Any], obs: Dict[str, Any]) -> None:
        self.calls.append({
            "call": len(self.calls) + 1,
            "action": action,
            "args": args,
            "observation": obs,
            "at": datetime.now().isoformat(timespec="seconds"),
        })

    def observation_text(self, obs: Dict[str, Any]) -> str:
        """Compact JSON for the LLM context (drop long paths/duplicates)."""
        return json.dumps(obs, ensure_ascii=False)
