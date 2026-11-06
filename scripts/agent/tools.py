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
from verify import verify_goal  # noqa: E402

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

    def __init__(self, episode_dir: Path, *, generator: str = "mld", skip_viz: bool = False) -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.skip_viz = skip_viz
        self.run_dir: Optional[Path] = None
        self.step = 0
        self.described_at: Optional[int] = None
        self.checked_at: Optional[int] = None
        self.edits = 0
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
    def generate_motion(self, prompt: str) -> Dict[str, Any]:
        from session_api import run_session

        t0 = time.time()
        out_dir = self._next_dir("generate")
        req = {
            "prompt": str(prompt),
            "mode": "generate",
            "run_dir": str(out_dir),
            "skip_review": True,
            "skip_viz": self.skip_viz,
            "backend": "orchestrator",
            "generator": self.generator,
        }
        resp = run_session(req, cwd=REPO)
        used = self.generator
        if not (out_dir / "03_edited.npy").is_file() and self.generator == "mld":
            # MLD is slow/heavy; keep the episode alive with the dataset draft.
            resp = run_session({**req, "generator": "dataset"}, cwd=REPO)
            used = "dataset_fallback"
        if not (out_dir / "03_edited.npy").is_file():
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
