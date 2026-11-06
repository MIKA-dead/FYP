"""
The four tools the LLM controller may call.

    generate_motion(prompt)      (a) MLD text→motion (falls back to the fast
                                     dataset draft if MLD is unavailable)
    describe_motion()            (c) TM2T motion→text + measured kinematics
    edit_motion(instruction)     (b) one relative edit via the session orchestrator
    finish()                     stop

The tools own the run directory, so the controller only has to write the
request/instruction text. Every call returns a compact JSON observation that is
appended to the conversation and written to ``agent_trace.json``.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "scripts" / "session",
    REPO / "scripts" / "orchestrator",
    REPO / "scripts" / "backends",
    REPO / "scripts" / "agent",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics import (  # noqa: E402
    infer_target_joints,
    instruction_following,
    kinematic_summary,
    load_joints,
    preservation,
    text_similarity,
)

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
        "args": {},
    },
]

SUPPORTED_EDITS = """The editing backend can apply (one per call):
- raise / lower one arm: "raise the right arm higher", "lower the left arm"
- one-arm wave; walk forward / more steps; slower / faster
- stop turning / keep the path straight (a much larger new turn cannot be invented)
- sit lower / stand up / remove walking before a sit
It cannot add objects, change the camera, change body shape, or replace the
whole action while keeping the performance."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


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
        invalid = []
        if float(m.get("ground_penetration") or 0) > 0.10:
            invalid.append("ground_penetration")
        if float(m.get("bone_cv") or 0) > 0.08:
            invalid.append("bone_cv")
        return {
            "run_dir": str(run_dir),
            "frames": int(j.shape[0]),
            "preview": str(next(iter(run_dir.glob("*preview*.gif")), "")),
            "generator": draft.get("generator") or "dataset",
            "source_policy": plan.get("source_policy"),
            "primary_action": plan.get("primary_action"),
            "quality_score": quality.get("score"),
            "valid": not invalid,
            "invalid_checks": invalid,
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

    def edit_motion(self, instruction: str) -> Dict[str, Any]:
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        from session_api import run_session

        t0 = time.time()
        before_dir = self.run_dir
        before = load_joints(before_dir / "03_edited.npy")
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
        self.edits += 1
        after = load_joints(out_dir / "03_edited.npy")
        kb = kinematic_summary(before)
        ka = kinematic_summary(after)
        invalid = []
        if float(ka.get("ground_penetration") or 0) > 0.15:
            invalid.append(f"ground_penetration {ka['ground_penetration']} > 0.15")
        if float(ka.get("bone_cv") or 0) > 0.08:
            invalid.append(f"bone_cv {ka['bone_cv']} > 0.08")
        if invalid:
            # A broken clip must not become the state the agent judges.
            self.run_dir = before_dir
            self.described_at = None
            obs = {
                "ok": False,
                "error": "edit_produced_invalid_clip",
                "invalid": invalid,
                "rolled_back": True,
                "instruction": str(instruction),
                "before": kb,
                "after": ka,
                "seconds": round(time.time() - t0, 2),
                "note": "The edit damaged the clip; the previous clip is restored. Try a different instruction.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
        target, region = infer_target_joints(instruction)
        pres = preservation(before, after, target)
        follow = instruction_following(instruction, kb, ka, pres)
        no_change = (
            pres.get("target_change", 0.0) <= 1e-4
            and pres.get("non_target_mpjpe", 0.0) <= 1e-4
            and after.shape == before.shape
        )
        obs = {
            "ok": True,
            "instruction": str(instruction),
            "run_dir": str(out_dir),
            "frames": int(after.shape[0]),
            "preview": str(next(iter(out_dir.glob("*preview*.gif")), "")),
            "region": region,
            "before": kb,
            "after": ka,
            "preservation": pres,
            "instruction_following": follow,
            "no_change": bool(no_change),
            "seconds": round(time.time() - t0, 2),
            "note": (
                "The clip is byte-for-byte the same: the backend cannot apply this "
                "instruction. Do not repeat it; try another supported edit or finish."
                if no_change
                else "No caption of the new clip yet — call describe_motion."
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
