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
import re
import shutil
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
    caption_signals,
    infer_target_joints,
    instruction_following,
    kinematic_summary,
    load_joints,
    preservation,
    structured_caption,
    text_similarity,
)
from motion_validity import physical_report, reground  # noqa: E402
import sampling  # noqa: E402
from verify import requirement_gap, verify_goal  # noqa: E402

MAX_RESAMPLES = 2
MAX_RESTORES = 2


def _candidate_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """The handful of measurements a draft is ranked on, kept short for the LLM."""
    keys = (
        "root_travel", "yaw_span_deg", "yaw_net_deg", "pelvis_drop",
        "arm_raise_left", "arm_raise_right",
        "ankle_lift_left", "ankle_lift_right",
        "jump_rise", "jump_clearance", "ground_penetration", "bone_cv",
    )
    return {k: m.get(k) for k in keys if m.get(k) is not None}


TOOL_SPECS = [
    {
        "name": "generate_motion",
        "description": (
            "Text→motion. Call once at the start with the user's request, lightly "
            "rewritten as a plain motion sentence (keep every requested detail). "
            "generator='mld' diffuses a new clip; generator='dataset' retrieves a "
            "real HumanML3D clip whose caption matches the prompt. Use the "
            "retrieval path when several diffusion drafts systematically lack a "
            "measured attribute (zero travel, a turn that never happens)."
        ),
        "args": {
            "prompt": "str — the text to generate",
            "generator": "str — optional: 'mld' (diffusion, default) or 'dataset' (retrieval)",
        },
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
        "name": "restore_motion",
        "description": (
            "Go back to the closest clip you already produced in this episode. Use "
            "it after an edit or resample made the measurement worse: a clip you "
            "already had is never lost, and you may not finish on a clip that is "
            "further from the goal than one you already saw. At most 2 per episode."
        ),
        "args": {"reason": "str — why you go back (one sentence)"},
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
        self.resamples = 0          # resamples that replaced the clip
        self.resample_attempts = 0  # every call that actually drew drafts
        self.restores = 0
        self.calls: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        # Every clip state this episode has measured, so a worse edit or a bad
        # resample can never destroy a closer draft (ep12 lost an 88° turn that
        # way) and so ``finish`` can be refused on a clip known to be worse.
        self.archive: List[Dict[str, Any]] = []

    # -- clip archive ------------------------------------------------------
    def _snapshot(self, tag: str = "") -> Dict[str, Any]:
        """Measure the current clip against the goal and remember it."""
        if self.run_dir is None:
            return {}
        edited = Path(self.run_dir) / "03_edited.npy"
        if not edited.is_file():
            return {}
        j = load_joints(edited)
        m = kinematic_summary(j)
        obs = {"metrics": m}
        v = verify_goal(self.goal, obs)
        rep = physical_report(j)
        snap = {
            "tag": tag,
            "step": self.step,
            "run_dir": str(self.run_dir),
            "satisfied": bool(v["ok"]),
            "gap": round(requirement_gap(self.goal, obs), 4),
            "valid": bool(rep["valid"]),
            "failed": [c["check"] for c in v["checks"] if not c["ok"]],
            "measured": _candidate_metrics(m),
        }
        self.archive = [a for a in self.archive if a.get("run_dir") != snap["run_dir"]]
        self.archive.append(snap)
        return snap

    @staticmethod
    def _rank_key(a: Dict[str, Any]) -> tuple:
        """Closest first: valid, then satisfied, then smallest requirement gap."""
        return (0 if a.get("valid") else 1, 0 if a.get("satisfied") else 1,
                float(a.get("gap") or 0.0), -int(a.get("step") or 0))

    def best_snapshot(self) -> Dict[str, Any]:
        if not self.archive:
            return {}
        return sorted(self.archive, key=self._rank_key)[0]

    def current_snapshot(self) -> Dict[str, Any]:
        if self.run_dir is None:
            return {}
        return next((a for a in self.archive if a.get("run_dir") == str(self.run_dir)), {})

    def archive_summary(self, limit: int = 4) -> List[Dict[str, Any]]:
        """Compact best-first list for the controller (no paths it cannot use)."""
        return [
            {k: a.get(k) for k in ("tag", "step", "satisfied", "gap", "valid", "failed")}
            | {"current": a.get("run_dir") == str(self.run_dir)}
            for a in sorted(self.archive, key=self._rank_key)[:limit]
        ]

    # -- helpers -----------------------------------------------------------
    def _next_dir(self, tag: str) -> Path:
        self.step += 1
        d = self.episode_dir / f"step{self.step:02d}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _caption(self, run_dir: Path, goal: str) -> Dict[str, Any]:
        """Refresh the caption with the learned M2T model and score the text.

        Two captions are returned, because they disagree in an informative way:
        the learned M2T sentence is what module (c) is scored on, while the
        rule sentence is built from the same measurements the checker reads. The
        controller is asked to compare *both* with the goal before deciding that
        a fix is needed, and the measurements stay the tie-breaker.
        """
        from motion_caption import caption_edited

        quality = _read_json(run_dir / "04_quality.json")
        plan = _read_json(run_dir / "00_motion_edit_plan.json")
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
        learned = str(cap.get("caption") or "")
        rule_text = structured_caption(kinematic_summary(load_joints(edited)))
        cap["similarity"] = text_similarity(goal, learned)
        cap["rule_caption"] = rule_text
        cap["rule_similarity"] = text_similarity(goal, rule_text)
        # The caption half of the fix decision. The rule caption is built from
        # the same measurements the checker reads, so a disagreement between the
        # two captions is a reason to verify, not to edit.
        learned_check = caption_signals(goal, learned)
        rule_check = caption_signals(goal, rule_text)
        if rule_check["needs_fix"]:
            level, needs_fix = "mismatch", True
        elif learned_check["needs_fix"]:
            level, needs_fix = "uncertain", False
        else:
            level, needs_fix = "match", False
        cap["caption_check"] = learned_check
        cap["rule_caption_check"] = rule_check
        cap["caption_verdict"] = {
            "level": level,
            "needs_fix": needs_fix,
            "learned_says": learned_check["needs_fix"],
            "rule_says": rule_check["needs_fix"],
            "because": rule_check["because"] or learned_check["because"],
            "note": (
                "Caption-level signal only. 'uncertain' means the learned caption "
                "reads wrong while the measurement-grounded caption reads right — "
                "confirm with check_requirement before editing; 'mismatch' means "
                "both captions disagree with the goal."
            ),
        }
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
        rejection = draft.get("rejection") or None
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
            **({"draft_rejection": rejection} if rejection else {}),
        }
    # -- tools -------------------------------------------------------------
    def _generate_into(self, prompt: str, out_dir: Path,
                       draft_joints: Optional[Path] = None,
                       generator: Optional[str] = None) -> tuple:
        """Run the generator into ``out_dir``; returns (ok, generator_used, resp)."""
        from session_api import run_session

        gen = str(generator or self.generator)
        req = {
            "prompt": str(prompt),
            "mode": "generate",
            "run_dir": str(out_dir),
            "skip_review": True,
            "skip_viz": self.skip_viz,
            "backend": "orchestrator",
            "generator": gen,
        }
        if draft_joints is not None:
            # Already drawn (one of several MLD samples): skip diffusion and let
            # the pipeline observe/edit/visualise this exact clip.
            req["draft_joints"] = str(draft_joints)
        resp = run_session(req, cwd=REPO)
        used = gen
        if not (out_dir / "03_edited.npy").is_file() and gen == "mld":
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

    def generate_motion(self, prompt: str, generator: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        gen = str(generator or self.generator or "mld").strip().lower()
        if gen not in ("mld", "dataset"):
            gen = self.generator
        out_dir = self._next_dir("generate")
        ok, used, resp = self._generate_into(str(prompt), out_dir, generator=gen)
        if not ok:
            obs = {"ok": False, "error": resp.get("error") or "generation produced no clip"}
        else:
            self.run_dir = out_dir
            self.described_at = None
            self.checked_at = None
            obs = self._observe_clip(out_dir, str(prompt))
            snap = self._snapshot("generate")
            obs.update({
                "ok": True,
                "requested_prompt": prompt,
                "generator_used": used,
                "requirement": {"satisfied": snap.get("satisfied"),
                                "gap": snap.get("gap"),
                                "failed": snap.get("failed")},
                "seconds": round(time.time() - t0, 2),
                "note": "No caption yet — call describe_motion before judging.",
            })
        self._record("generate_motion", {"prompt": prompt, "generator": gen}, obs)
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
        # The budget caps *attempts*, not replacements: a resample that keeps the
        # clip it already had still cost a generator load, and charging only
        # adopted clips let a controller draw forever without ever improving.
        if self.resample_attempts >= MAX_RESAMPLES:
            return {
                "ok": False,
                "error": "resample_budget_reached",
                "note": f"{MAX_RESAMPLES} resamples already used. Edit, accept what "
                        "you have, or finish naming the gap.",
            }
        n = max(2, min(4, int(n or 3)))
        self.step += 1
        self.resample_attempts += 1
        base = self.episode_dir / f"step{self.step:02d}_resample"
        base.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        plan = sampling.plan(goal, want=n, batches_left=1)
        cands: List[Dict[str, Any]] = []
        # The clip already in hand competes with the new drafts. Without this a
        # resample could silently replace a closer clip with a worse one (ep12
        # went 88° -> 72° and lost the draft that nearly met the goal).
        if self.run_dir is not None and (self.run_dir / "03_edited.npy").is_file():
            j0 = load_joints(self.run_dir / "03_edited.npy")
            cands.append({
                "cand": 0,
                "run_dir": str(self.run_dir),
                "ok": True,
                "batch": 0,
                "generator_used": "current_clip",
                "from_generator": False,
                "baseline": True,
                "valid": bool(physical_report(j0)["valid"]),
                "metrics": kinematic_summary(j0),
            })
        batches_used = 0
        accepted: List[Dict[str, Any]] = []
        while batches_used < plan["batches"]:
            batches_used += 1
            batch_dir = base if batches_used == 1 else base / f"batch{batches_used}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            drawn = (
                self._draw_samples(goal, n, batch_dir / "draw")
                if self.generator == "mld" else []
            )
            for k in range(1, n + 1):
                i = (batches_used - 1) * n + k
                d = batch_dir / f"cand{k}"
                d.mkdir(parents=True, exist_ok=True)
                sample = drawn[k - 1] if k - 1 < len(drawn) else None
                ok, used, resp = self._generate_into(goal, d, draft_joints=sample)
                if not ok:
                    cands.append({"cand": i, "run_dir": str(d), "ok": False,
                                  "error": resp.get("error") or "no clip"})
                    continue
                j = load_joints(d / "03_edited.npy")
                m = kinematic_summary(j)
                rep = physical_report(j)
                cands.append({
                    "cand": i,
                    "run_dir": str(d),
                    "ok": True,
                    "batch": batches_used,
                    "generator_used": used,
                    "from_generator": sample is not None,
                    "valid": bool(rep["valid"]),
                    "metrics": m,
                })
            ranked = sampling.rank(goal, cands)
            accepted = [c for c in ranked if c.get("satisfied") and c.get("valid")]
            if accepted or batches_used >= plan["batches"]:
                break
        usable = [c for c in cands if c.get("ok")]
        if not usable:
            obs = {"ok": False, "error": "resample produced no clip",
                   "candidates": cands, "seconds": round(time.time() - t0, 2)}
            self._record("resample_motion", {"prompt": goal, "n": n}, obs)
            return obs
        fresh = [c for c in usable if c.get("from_generator")]
        if not fresh:
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
                    "would return the same motion. The current clip is unchanged. "
                    "Edit, change the prompt, or finish naming the gap. If the "
                    "generator should be running, check its environment "
                    "(MLD_PYTHON / the `mld` conda env)."
                ),
            }
            self._record("resample_motion", {"prompt": goal, "n": n}, obs)
            return obs
        best = (accepted or sampling.rank(goal, cands))[0]
        kept_current = bool(best.get("baseline"))
        if not kept_current:
            self.resamples += 1
            self.run_dir = Path(best["run_dir"])
            self.described_at = None
            self.checked_at = None
        distinct = len({json.dumps(_candidate_metrics(c["metrics"]), sort_keys=True)
                        for c in usable})
        round_summary = sampling.summary(goal, cands, best)
        obs = self._observe_clip(self.run_dir, goal)
        snap = self._snapshot("resample_keep" if kept_current else "resample")
        obs.update({
            "ok": True,
            "candidates": [
                {k: v for k, v in c.items() if k != "metrics"}
                | {"measured": _candidate_metrics(c.get("metrics") or {})}
                for c in cands
            ],
            "chosen": best["cand"],
            "chosen_gap": best["gap"],
            "chosen_satisfied": best["satisfied"],
            "kept_current": kept_current,
            "accepted": len(accepted),
            "distinct_candidates": distinct,
            "batches": batches_used,
            "acceptance": round_summary["acceptance"],
            "resamples_left": max(0, MAX_RESAMPLES - self.resample_attempts),
            "requirement": {"satisfied": snap.get("satisfied"),
                            "gap": snap.get("gap"), "failed": snap.get("failed")},
            "seconds": round(time.time() - t0, 2),
            "note": (
                "Every new draft was further from the goal than the clip you "
                "already had, so the current clip was kept (cand0). No describe is "
                "needed for it. Edit, accept it, or finish naming the gap."
                if kept_current
                else (
                    f"Kept cand{best['cand']} (valid first, then lowest requirement gap). "
                    f"{len(accepted)}/{len(usable)} drafts already satisfied: "
                    f"{round_summary['acceptance'] or 'no measurable constraint'}. "
                    "No caption for it yet — call describe_motion, then compare it with "
                    "the candidates list before deciding."
                    if distinct > 1
                    else "Every draft measured identically, so another draw of this "
                         "prompt will only repeat it. Edit, change the prompt, or "
                         "finish naming the gap."
                )
            ),
        })
        self._record("resample_motion", {"prompt": goal, "n": n}, obs)
        return obs

    def restore_motion(self, reason: str = "") -> Dict[str, Any]:
        """Go back to the closest clip this episode has already measured."""
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        if self.restores >= MAX_RESTORES:
            return {
                "ok": False,
                "error": "restore_budget_reached",
                "note": f"{MAX_RESTORES} restores already used. Work with the "
                        "current clip or finish naming the gap.",
            }
        cur = self.current_snapshot() or self._snapshot("current")
        best = self.best_snapshot()
        if not best or best.get("run_dir") == str(self.run_dir):
            return {
                "ok": False,
                "error": "already_best",
                "note": "The current clip is already the closest one measured. "
                        "Nothing to restore.",
                "archive": self.archive_summary(),
            }
        better = (best.get("satisfied") and not cur.get("satisfied")) or (
            float(best.get("gap") or 0) + 1e-6 < float(cur.get("gap") or 0)
        )
        if not better:
            return {
                "ok": False,
                "error": "already_best",
                "note": "No earlier clip is closer to the goal than the current one.",
                "archive": self.archive_summary(),
            }
        self.restores += 1
        self.run_dir = Path(str(best["run_dir"]))
        self.described_at = None
        self.checked_at = None
        obs = self._observe_clip(self.run_dir, self.goal)
        obs.update({
            "ok": True,
            "restored": True,
            "reason": reason,
            "restored_from_step": best.get("step"),
            "gap_before": cur.get("gap"),
            "gap_after": best.get("gap"),
            "restore_budget_left": max(0, MAX_RESTORES - self.restores),
            "archive": self.archive_summary(),
            "note": "The earlier clip is back. Call describe_motion, then "
                    "check_requirement before deciding.",
        })
        self._record("restore_motion", {"reason": reason}, obs)
        return obs

    def describe_motion(self, goal: str) -> Dict[str, Any]:
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        t0 = time.time()
        cap = self._caption(self.run_dir, goal)
        obs = self._observe_clip(self.run_dir, goal)
        snap = self._snapshot("describe")
        obs.update({
            "ok": True,
            "caption": cap.get("caption"),
            "caption_source": cap.get("source"),
            "caption_problems": cap.get("problems") or [],
            "similarity": cap.get("similarity"),
            "rule_caption": cap.get("rule_caption"),
            "rule_similarity": cap.get("rule_similarity"),
            "caption_verdict": cap.get("caption_verdict"),
            "requirement": {"satisfied": snap.get("satisfied"),
                            "gap": snap.get("gap"), "failed": snap.get("failed")},
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
        gap = requirement_gap(goal, {"metrics": m})
        self._snapshot("check")
        obs = {
            "ok": True,
            "satisfied": bool(verdict["ok"]),
            "requirement_gap": round(gap, 4),
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

    def _edit_precondition(self, family: str, instruction: str,
                           kb: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Refuse an edit the clip cannot support, before any backend runs.

        These operators are *relative*: they scale or extend motion that is
        already in the clip. Running one on a clip that lacks that motion
        returns an identical file, which used to be reported as a successful
        edit. Naming the missing precondition is more useful than a no-op.
        """
        txt = (instruction or "").lower()
        yaw = abs(float(kb.get("yaw_span_deg") or 0))
        travel = float(kb.get("root_travel") or 0)
        drop = float(kb.get("pelvis_drop") or 0)
        feet = max(float(kb.get("ankle_lift_right") or 0),
                   float(kb.get("ankle_lift_left") or 0))
        if family == "turn" and yaw < 20:
            wants_more = re.search(r"further|more|bigger|larger|again|再", txt) is not None
            return {
                "ok": False,
                "error": "edit_needs_existing_turn",
                "instruction": instruction,
                "yaw_span_deg": round(yaw, 1),
                "reason": "this clip does not turn, and the turn operator can only "
                          "stop or limit a turn that is already there",
                "suggestion": ("ask generate_motion for a turning clip, then edit it"
                               if wants_more else
                               "skip this edit; the clip has no turn to stop"),
                "note": "Do not repeat this instruction on this clip.",
            }
        if family == "speed" and travel < 0.1 and feet < 0.1:
            return {
                "ok": False,
                "error": "edit_needs_existing_motion",
                "instruction": instruction,
                "root_travel": round(travel, 3),
                "reason": "a tempo change retimes motion; this clip barely moves",
                "suggestion": "generate a moving clip first, then retime it",
                "note": "Do not repeat this instruction on this clip.",
            }
        if family in ("stand_up", "sit_deeper", "sit_remove_preamble") and drop < 0.05:
            return {
                "ok": False,
                "error": "edit_needs_seated_clip",
                "instruction": instruction,
                "pelvis_drop": round(drop, 3),
                "reason": "this clip is not seated, so there is no sit to deepen "
                          "or stand up from",
                "suggestion": "generate a sitting clip first, then edit it",
                "note": "Do not repeat this instruction on this clip.",
            }
        return None

    def edit_motion(self, instruction: str) -> Dict[str, Any]:
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        cls = classify(instruction)
        if not cls["supported"]:
            if cls.get("open_vocab"):
                # The closed set cannot apply it, but a generative editor takes
                # the sentence as written. Its result is judged by the same
                # measurement and rollback rule as any other edit.
                return self._edit_open_vocab(str(instruction), cls)
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
        pre = self._edit_precondition(str(cls["family"]), str(instruction),
                                      kinematic_summary(before))
        if pre is not None:
            pre["seconds"] = round(time.time() - t0, 2)
            self._record("edit_motion", {"instruction": instruction}, pre)
            return pre
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
        kb = kinematic_summary(before)
        return self._assess_after(
            before=before, kb=kb, before_dir=before_dir, out_dir=out_dir,
            instruction=str(instruction), cls=cls, t0=t0,
            grounded_offset=0.0, backend="session",
        )

    def _edit_open_vocab(self, instruction: str, cls: Dict[str, Any]) -> Dict[str, Any]:
        """Send an out-of-registry instruction to the generative editor.

        SimMotionEdit consumes the sentence directly, so this is the one path
        where the instruction is neither matched nor rejected. Everything after
        the backend call is deliberately identical to the closed-set path: the
        same grounding repair, the same before/after measurement, and the same
        rollback if the edit *introduces* physical damage.
        """
        if self.run_dir is None:
            return {"ok": False, "error": "no clip yet; call generate_motion first"}
        t0 = time.time()
        before_dir = self.run_dir
        before = load_joints(before_dir / "03_edited.npy")
        out_dir = self._next_dir("edit_oov")
        out_npy = out_dir / "03_edited.npy"
        backends = str(REPO / "scripts" / "backends")
        if backends not in sys.path:
            sys.path.insert(0, backends)
        try:
            import simmotion_edit  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            obs = {"ok": False, "error": "open_vocab_backend_missing",
                   "detail": str(exc), "instruction": instruction}
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
        res = simmotion_edit.run_edit(
            source_npy=before_dir / "03_edited.npy",
            text=instruction,
            out_npy=out_npy,
        )
        if not res.get("ok") or not out_npy.is_file():
            obs = {
                "ok": False,
                "error": res.get("error") or "open_vocab_edit_failed",
                "backend": "simmotion",
                "instruction": instruction,
                "family": cls.get("family"),
                "hint": res.get("hint"),
                "seconds": round(time.time() - t0, 2),
                "note": "The open-vocabulary editor did not produce a clip. Do not "
                        "repeat this instruction; edit within the supported set or "
                        "finish naming the gap.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
        self.run_dir = out_dir
        self.described_at = None
        self.checked_at = None
        self.edits += 1
        kb = kinematic_summary(before)
        return self._assess_after(
            before=before, kb=kb, before_dir=before_dir, out_dir=out_dir,
            instruction=instruction,
            cls={"family": cls.get("family") or "open_vocab", "supported": True},
            t0=t0, grounded_offset=0.0, backend="simmotion",
        )

    def _assess_after(self, *, before: np.ndarray, kb: Dict[str, Any],
                      before_dir: Path, out_dir: Path, instruction: str,
                      cls: Dict[str, Any], t0: float,
                      grounded_offset: float = 0.0,
                      backend: str = "session") -> Dict[str, Any]:
        """Measure an edit's result and roll it back if it *introduced* damage.

        Shared by the closed-set session editor and the open-vocabulary backend,
        so both are judged by exactly the same rule: inherited defects are
        reported, new ones restore the previous clip.
        """
        after_f = out_dir / "03_edited.npy"
        after = load_joints(after_f)
        rep = physical_report(after)
        if rep["ground"]["mode"] == "offset":
            fixed, shift = reground(after)
            if shift:
                np.save(after_f, fixed)
                after = fixed
                rep = physical_report(after)
                grounded_offset = shift
        rep_before = physical_report(before)
        ka = kinematic_summary(after)
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
                "backend": backend,
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
        gap_before = round(requirement_gap(self.goal, {"metrics": kb}), 4)
        gap_after = round(requirement_gap(self.goal, {"metrics": ka}), 4)
        # A clip can come back *different* without doing what it was told: the
        # walk assembler rejects its own 4-cycle repeat and returns the source,
        # a stand-up only trims frames, a "stop turning" leaves the yaw alone.
        # Keeping that result would silently trade correct motion for nothing,
        # so restore the source and say which attribute did not move.
        named_fails = [
            c for c in (follow.get("checks") or [])
            if c.get("ok") is False
            and c.get("attribute") not in ("target_vs_rest", "non_target_mpjpe_norm")
        ]
        if follow.get("verifiable") and named_fails and gap_after >= gap_before - 1e-6:
            self.run_dir = before_dir
            self.described_at = None
            self.checked_at = None
            obs = {
                "ok": False,
                "error": "edit_did_not_follow_instruction",
                "instruction": str(instruction),
                "family": cls["family"],
                "backend": backend,
                "failed_checks": named_fails,
                "no_change": True,
                "rolled_back": True,
                "run_dir": str(before_dir),
                "gap_before": gap_before,
                "gap_after": gap_before,
                "instruction_following": follow,
                "seconds": round(time.time() - t0, 2),
                "note": "The clip changed but the attribute you named did not "
                        "move the requested way; the previous clip is restored. "
                        "Do not repeat this instruction — regenerate instead.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
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
        if no_change:
            # A byte-identical output is not an edit. Keep the source as the
            # state and say so, instead of letting the controller score a
            # silent no-op as success.
            self.run_dir = before_dir
            self.described_at = None
            self.checked_at = None
            obs = {
                "ok": False,
                "error": "edit_had_no_effect",
                "instruction": str(instruction),
                "family": cls["family"],
                "backend": backend,
                "no_change": True,
                "rolled_back": True,
                "run_dir": str(before_dir),
                "gap_before": gap_before,
                "gap_after": gap_before,
                "instruction_following": follow,
                "seconds": round(time.time() - t0, 2),
                "note": "The output was identical to the input: this operator "
                        "cannot make the requested change on this clip. Do not "
                        "repeat it; try a different edit or regenerate.",
            }
            self._record("edit_motion", {"instruction": instruction}, obs)
            return obs
        snap = self._snapshot("edit")
        obs = {
            "ok": True,
            "instruction": str(instruction),
            "family": cls["family"],
            "backend": backend,
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
            "gap_before": gap_before,
            "gap_after": gap_after,
            "worse_than_before": bool(gap_after > gap_before + 1e-6),
            "requirement": {"satisfied": snap.get("satisfied"),
                            "gap": snap.get("gap"), "failed": snap.get("failed")},
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
                            else (
                                f"This edit moved the clip further from the goal "
                                f"(gap {gap_before:.2f} -> {gap_after:.2f}); an earlier "
                                "clip is closer. Call restore_motion to go back."
                                if gap_after > gap_before + 1e-6
                                else "No caption of the new clip yet — call describe_motion."
                            )
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
