"""
Orchestrator: motion-planner → motion-coder (run_module) → motion-tester.

Mirrors the successful v05 generate/revise path, but every backend call goes
through scripts/modules/run_module.py so OpenCode agents can schedule the same tools.

  python scripts/orchestrator/run_session.py \\
      --prompt "a person sits down" --mode generate \\
      --run-dir outputs/session/orch_sit

UI / session_api may call run_orchestrated_session(request).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
INTENT_DIR = REPO_ROOT / "scripts" / "intent"
MODULES_DIR = REPO_ROOT / "scripts" / "modules"
SESSION_DIR = REPO_ROOT / "scripts" / "session"

sys.path.insert(0, str(INTENT_DIR))
sys.path.insert(0, str(MODULES_DIR))
sys.path.insert(0, str(SESSION_DIR))

from plan_motion_edit import build_full_agent_bundle  # noqa: E402
from run_module import run_module  # noqa: E402

SCHEMA = "session_response_v1"
AGENT_ROLES = {
    "planner": ".opencode/agents/motion-planner.md",
    "coder": ".opencode/agents/motion-coder.md",
    "tester": ".opencode/agents/motion-tester.md",
    "documenter": ".opencode/agents/motion-documenter.md",
}


def _abs(p: Optional[Path], cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    return p if p.is_absolute() else (cwd / p).resolve()


def _trace_step(
    agent: str,
    phase: str,
    title: str,
    *,
    module: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    ok: bool = True,
    detail: Any = None,
    seconds: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "agent": agent,
        "phase": phase,
        "title": title,
        "module": module,
        "args": args or {},
        "artifacts": artifacts or {},
        "ok": ok,
        "detail": detail,
        "seconds": seconds,
    }


def _use_native(plan: Dict[str, Any], force_bridged: bool, edit_source_id: Optional[str]) -> bool:
    if force_bridged:
        return False
    if edit_source_id:
        return True
    return bool(
        (
            plan.get("edit_source") == "motionfix_id"
            and plan.get("preferred_source_id")
        )
        or plan.get("source_policy") == "use_motionfix_native"
    )


def _find_last_good_edited(start: Optional[Path], *, min_pelvis_drop: float = 0.12) -> Optional[Path]:
    """Walk previous_run chain; return last 03_edited.npy that still looks like a sit."""
    seen = set()
    cur = start
    best: Optional[Path] = None
    while cur is not None and str(cur) not in seen:
        seen.add(str(cur))
        edited = cur / "03_edited.npy"
        qpath = cur / "04_quality.json"
        ok = False
        if qpath.is_file():
            try:
                q = json.loads(qpath.read_text(encoding="utf-8-sig"))
                drop = float((q.get("metrics") or {}).get("pelvis_drop") or 0)
                ok = bool(q.get("passed")) and drop >= min_pelvis_drop
            except Exception:  # noqa: BLE001
                ok = False
        if ok and edited.is_file():
            best = edited
            break
        # Prefer any ancestor with enough pelvis drop even if soft-failed
        if edited.is_file() and qpath.is_file():
            try:
                q = json.loads(qpath.read_text(encoding="utf-8-sig"))
                drop = float((q.get("metrics") or {}).get("pelvis_drop") or 0)
                if drop >= min_pelvis_drop and best is None:
                    best = edited
            except Exception:  # noqa: BLE001
                pass
        summ = cur / "summary.json"
        nxt = None
        if summ.is_file():
            try:
                prev = json.loads(summ.read_text(encoding="utf-8-sig")).get("previous_run")
                if prev:
                    nxt = Path(prev)
            except Exception:  # noqa: BLE001
                nxt = None
        cur = nxt
    return best


def _build_coder_steps(
    *,
    mode: str,
    plan: Dict[str, Any],
    run_dir: Path,
    prev_dir: Optional[Path],
    length: int,
    bridge_iters: int,
    skip_viz: bool,
    force_bridged: bool,
    edit_source_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Expand MotionEditPlan into ordered run_module calls (v05-equivalent)."""
    steps: List[Dict[str, Any]] = []
    edit_mode = plan.get("edit_mode") or "diffusion"
    is_speed = edit_mode == "speed"
    is_geo = edit_mode == "geometric_only" or plan.get("source_policy") in (
        "restore_last_good_or_native",
    )
    force_walk_wave = bool(
        plan.get("force_bridged_walk")
        or plan.get("source_policy")
        in ("mld_walk_kinematic_wave", "reuse_previous_wave_fix")
        or (edit_mode == "kinematic" and str(plan.get("primary_action") or "").startswith("wave"))
    )
    native = (
        _use_native(plan, force_bridged, edit_source_id)
        and not is_speed
        and not is_geo
        and not force_walk_wave
    )
    native_id = edit_source_id or plan.get("preferred_source_id")
    intent_plan = run_dir / "00_motion_edit_plan.json"
    edited = run_dir / "03_edited.npy"
    preview = run_dir / "03_edited_preview.gif"
    mld_text = plan.get("mld_text") or plan.get("raw_prompt") or ""
    refine = plan.get("refine") or {}
    refine_ops = set(refine.get("ops") or [])

    if force_walk_wave:
        # Prefer MLD walk → bridge → kinematic wave; soft-fail each step.
        # ensure_copy falls back to previous edited if MLD/bridge/edit failed.
        joints = run_dir / "01_mld_joints.npy"
        source = run_dir / "02_motionfix_source.npy"
        walk_text = (
            mld_text
            if "walk" in str(mld_text).lower()
            else "a person walks forward."
        )
        steps.append(
            {
                "kind": "module",
                "id": "mld_generate",
                "title": "MLD forward walk (wave rebuild)",
                "args": {
                    "text": walk_text,
                    "length": max(int(length), 80),
                    "out": str(joints),
                },
                "soft": True,
            }
        )
        steps.append(
            {
                "kind": "module",
                "id": "mld_bridge",
                "title": "Bridge walk → MotionsFix source",
                "args": {
                    "joints": str(joints),
                    "out": str(source),
                    "iters": bridge_iters,
                },
                "soft": True,
            }
        )
        steps.append(
            {
                "kind": "module",
                "id": "motionfix_semantic_edit",
                "title": "Kinematic one-arm wave on MLD walk",
                "args": {
                    "out": str(edited),
                    "source_npy": str(source),
                    "intent_json": str(intent_plan),
                },
                "soft": True,
            }
        )
        # Fallback seed if MLD path did not produce edited.npy
        fallback_src = None
        if prev_dir:
            for cand in (
                prev_dir / "03_edited.npy",
                prev_dir / "02_motionfix_source.npy",
            ):
                if cand.is_file():
                    fallback_src = cand
                    break
        if fallback_src is not None:
            steps.append(
                {
                    "kind": "ensure_copy",
                    "title": "Fallback: seed from previous clip if MLD path failed",
                    "src": str(fallback_src),
                    "dst": str(edited),
                }
            )
    elif is_speed:
        # Seed source from previous edited
        source = run_dir / "02_motionfix_source.npy"
        if not prev_dir:
            raise ValueError("speed revise requires previous_run")
        prev_edited = prev_dir / "03_edited.npy"
        steps.append(
            {
                "kind": "copy",
                "title": "Seed revise source from previous edited",
                "src": str(prev_edited),
                "dst": str(source),
            }
        )
        cons = plan.get("constraints") or {}
        steps.append(
            {
                "kind": "module",
                "id": "speed_edit",
                "title": "Geometric speed edit",
                "args": {
                    "out": str(edited),
                    "source_npy": str(source),
                    "intent_json": str(intent_plan),
                    "edit_mode": "speed",
                    "speed_scale": cons.get("speed_scale"),
                },
            }
        )
    elif is_geo and mode == "revise":
        # Copy last-good (or previous) edited → sit_hold only. No MotionsFix diffusion.
        seed: Optional[Path] = None
        if "restore_sit" in refine_ops or plan.get("source_policy") == "restore_last_good_or_native":
            seed = _find_last_good_edited(prev_dir)
            if seed is None and native_id:
                # Fall back: MotionsFix native once, then geometric hold
                steps.append(
                    {
                        "kind": "module",
                        "id": "motionfix_semantic_edit",
                        "title": f"Restore sit via MotionsFix native id={native_id}",
                        "args": {
                            "out": str(edited),
                            "source_id": str(native_id),
                            "intent_json": str(intent_plan),
                        },
                    }
                )
            elif seed is not None:
                steps.append(
                    {
                        "kind": "copy",
                        "title": "Restore from last good sit clip",
                        "src": str(seed),
                        "dst": str(edited),
                    }
                )
            elif prev_dir and (prev_dir / "03_edited.npy").is_file():
                steps.append(
                    {
                        "kind": "copy",
                        "title": "Seed from previous edited (fallback)",
                        "src": str(prev_dir / "03_edited.npy"),
                        "dst": str(edited),
                    }
                )
        else:
            if not prev_dir or not (prev_dir / "03_edited.npy").is_file():
                raise ValueError("geometric_only revise requires previous_run/03_edited.npy")
            # If previous already failed sit, hop to last good
            seed = prev_dir / "03_edited.npy"
            qprev = prev_dir / "04_quality.json"
            # Only hop when sit itself is gone — not when yaw soft/hard alone fails.
            bad = False
            if qprev.is_file():
                try:
                    q = json.loads(qprev.read_text(encoding="utf-8-sig"))
                    drop = float((q.get("metrics") or {}).get("pelvis_drop") or 0)
                    hard = set(q.get("hard_failures") or [])
                    sit_broken = drop < 0.08 or bool(
                        hard & {"sit_pelvis_drop", "sit_phase_descend", "missing_edited"}
                    )
                    bad = sit_broken
                except Exception:  # noqa: BLE001
                    bad = False
            if bad:
                good = _find_last_good_edited(prev_dir)
                if good is not None:
                    seed = good
            steps.append(
                {
                    "kind": "copy",
                    "title": "Geometric revise seed (no diffusion)",
                    "src": str(seed),
                    "dst": str(edited),
                }
            )
    elif mode == "revise" and not native:
        source = run_dir / "02_motionfix_source.npy"
        if prev_dir:
            prev_edited = prev_dir / "03_edited.npy"
            src_prev = prev_dir / "02_motionfix_source.npy"
            seed = prev_edited if prev_edited.is_file() else src_prev
            if seed and Path(seed).is_file():
                steps.append(
                    {
                        "kind": "copy",
                        "title": "Seed revise source",
                        "src": str(seed),
                        "dst": str(source),
                    }
                )
        steps.append(
            {
                "kind": "module",
                "id": "motionfix_semantic_edit",
                "title": "MotionsFix revise edit",
                "args": {
                    "out": str(edited),
                    "source_npy": str(source),
                    "intent_json": str(intent_plan),
                },
            }
        )
    elif native and native_id:
        steps.append(
            {
                "kind": "module",
                "id": "motionfix_semantic_edit",
                "title": f"MotionsFix native edit id={native_id}",
                "args": {
                    "out": str(edited),
                    "source_id": str(native_id),
                    "intent_json": str(intent_plan),
                },
            }
        )
    else:
        # Full MLD → bridge → edit
        joints = run_dir / "01_mld_joints.npy"
        source = run_dir / "02_motionfix_source.npy"
        steps.append(
            {
                "kind": "module",
                "id": "mld_generate",
                "title": "MLD generate",
                "args": {
                    "text": mld_text,
                    "length": length,
                    "out": str(joints),
                },
            }
        )
        if not skip_viz:
            steps.append(
                {
                    "kind": "module",
                    "id": "visualize",
                    "title": "Preview MLD joints",
                    "args": {
                        "input": str(joints),
                        "out": str(run_dir / "01_mld_preview.gif"),
                        "title": f"MLD: {mld_text}",
                        "env": "mld",
                    },
                    "soft": True,
                }
            )
        steps.append(
            {
                "kind": "module",
                "id": "mld_bridge",
                "title": "Bridge to MotionsFix source",
                "args": {
                    "joints": str(joints),
                    "out": str(source),
                    "iters": bridge_iters,
                },
            }
        )
        steps.append(
            {
                "kind": "module",
                "id": "motionfix_semantic_edit",
                "title": "MotionsFix bridged edit",
                "args": {
                    "out": str(edited),
                    "source_npy": str(source),
                    "intent_json": str(intent_plan),
                },
            }
        )

    # Sit hold geometric postprocess when refine / geometric_only asks for it
    primary = str(plan.get("primary_action") or refine.get("keep_action") or "")
    need_hold = (
        is_geo
        or "sit_hold_postprocess" in (plan.get("modules") or [])
        or bool(refine.get("phase_speed"))
        or "phase_speed" in refine_ops
        or any(
            f in (refine.get("forbid") or [])
            for f in (
                "post_sit_limb_rebound",
                "post_sit_stretch",
                "yaw_spin",
                "head_torso_misalign",
                "timeline_trim",
                "failed_sit",
                "pre_sit_locomotion",
                "post_sit_extra",
            )
        )
    )
    need_timeline_edit = bool(
        refine.get("phase_speed")
        or refine.get("drop_phases")
        or "phase_speed" in refine_ops
        or "drop_phases" in refine_ops
    )
    if need_hold and primary.startswith("sit"):
        steps.append(
            {
                "kind": "module",
                "id": "sit_hold_postprocess",
                "title": "Sit hold postprocess (geometric)",
                "args": {
                    "input": str(edited),
                    "out": str(edited),
                    "intent_json": str(intent_plan),
                    "forbid": refine.get("forbid")
                    or ["yaw_spin", "post_sit_limb_rebound", "post_sit_stretch"],
                },
            }
        )
    elif need_timeline_edit:
        # wave / walk / turn / generic: apply phase speed or drop without sit-hold
        tl_json = None
        if prev_dir and (prev_dir / "05_timeline.json").is_file():
            tl_json = str(prev_dir / "05_timeline.json")
        steps.append(
            {
                "kind": "module",
                "id": "timeline_postprocess",
                "title": "Timeline phase edit (speed/drop)",
                "args": {
                    "input": str(edited),
                    "out": str(edited),
                    "intent_json": str(intent_plan),
                    "action": primary,
                    **({"timeline_json": tl_json} if tl_json else {}),
                },
            }
        )

    need_wave_walk = (
        primary.startswith("wave")
        and (
            force_walk_wave
            or "wave_walk_postprocess" in (plan.get("modules") or [])
            or "wave_walk_fix" in refine_ops
            or bool(refine.get("prefer_kinematic_walk_wave"))
        )
    )
    if need_wave_walk:
        side = "left" if "left" in primary else "right"
        steps.append(
            {
                "kind": "module",
                "id": "wave_walk_postprocess",
                "title": "Wave walk fix (forward + one arm)",
                "args": {
                    "input": str(edited),
                    "out": str(edited),
                    "intent_json": str(intent_plan),
                    "side": side,
                    "min_forward": 0.9,
                },
            }
        )

    if not skip_viz:
        steps.append(
            {
                "kind": "module",
                "id": "visualize",
                "title": "Preview edited motion",
                "args": {
                    "input": str(edited),
                    "out": str(preview),
                    "title": f"{mode}: {plan.get('mf_text') or plan.get('raw_prompt')}",
                    "env": "motionfix",
                },
                "soft": True,
            }
        )
    return steps


def run_orchestrated_session(
    request: Dict[str, Any],
    *,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    cwd = cwd or Path.cwd()
    t0 = time.time()
    prompt = (request.get("prompt") or "").strip()
    mode = request.get("mode") or "generate"
    if not prompt:
        return _err("prompt is required", mode=mode, prompt=prompt)

    skip_viz = bool(request.get("skip_viz", False))
    length = int(request.get("length") or 60)
    bridge_iters = int(request.get("bridge_iters") or 40)
    force_bridged = bool(request.get("force_bridged_edit"))
    edit_source_id = request.get("edit_source_id")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _abs(Path(request["run_dir"]) if request.get("run_dir") else None, cwd)
    if run_dir is None:
        run_dir = (REPO_ROOT / "outputs" / "session" / "orch" / f"{mode}_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    prev = _abs(Path(request["previous_run"]) if request.get("previous_run") else None, cwd)
    if mode == "revise" and prev is None:
        return _err("revise requires previous_run", mode=mode, prompt=prompt, run_dir=run_dir)

    agent_trace: List[Dict[str, Any]] = []
    module_results: List[Dict[str, Any]] = []

    # --- Planner ---
    task_id = request.get("task_id") or f"ORCH_{stamp}"
    t_plan = time.time()
    prev_summary: Dict[str, Any] = {}
    prev_plan: Dict[str, Any] = {}
    if prev is not None:
        sp = prev / "summary.json"
        pp = prev / "00_motion_edit_plan.json"
        if sp.is_file():
            prev_summary = json.loads(sp.read_text(encoding="utf-8-sig"))
        if pp.is_file():
            prev_plan = json.loads(pp.read_text(encoding="utf-8-sig"))
        elif prev_summary.get("motion_edit_plan"):
            prev_plan = dict(prev_summary["motion_edit_plan"])

    prev_action = prev_plan.get("base_action") or prev_plan.get("primary_action")
    if prev_action == "speed_revise" and prev is not None:
        # Walk one hop: speed revise's previous_run often holds the semantic clip.
        hop = prev_summary.get("previous_run")
        if hop:
            hop_plan = Path(hop) / "00_motion_edit_plan.json"
            if hop_plan.is_file():
                try:
                    hp = json.loads(hop_plan.read_text(encoding="utf-8-sig"))
                    prev_action = hp.get("base_action") or hp.get("primary_action") or prev_action
                except Exception:  # noqa: BLE001
                    pass

    # Optional closed Refine JSON from UI timeline clicks / structured API.
    refine_override = request.get("refine") or request.get("refine_override")
    if refine_override is not None and not isinstance(refine_override, dict):
        refine_override = None
    timeline_ctx = request.get("timeline")
    if timeline_ctx is not None and not isinstance(timeline_ctx, dict):
        timeline_ctx = None
    if timeline_ctx is None and prev is not None:
        tl_path = prev / "05_timeline.json"
        if tl_path.is_file():
            try:
                timeline_ctx = json.loads(tl_path.read_text(encoding="utf-8-sig"))
            except Exception:  # noqa: BLE001
                timeline_ctx = None

    prev_caption: Optional[Dict[str, Any]] = None
    if isinstance(request.get("previous_caption"), dict):
        prev_caption = request["previous_caption"]
    elif prev is not None:
        cap_path = prev / "06_motion_caption.json"
        if cap_path.is_file():
            try:
                prev_caption = json.loads(cap_path.read_text(encoding="utf-8-sig"))
            except Exception:  # noqa: BLE001
                prev_caption = None

    bundle = build_full_agent_bundle(
        prompt,
        mode=mode,
        task_id=task_id,
        source_hint=str(request.get("source_hint") or "bridged_walk"),
        previous_task_id=prev_summary.get("task_id"),
        previous_action=prev_action if prev_action != "speed_revise" else None,
        previous_plan=prev_plan or None,
        refine_override=refine_override,
        timeline=timeline_ctx,
        previous_caption=prev_caption,
    )
    intent = bundle["intent"]
    module_plan = bundle["module_plan"]
    plan = bundle["motion_edit_plan"]

    bundle_path = run_dir / "00_agent_bundle.json"
    plan_path = run_dir / "00_motion_edit_plan.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    agent_trace.append(
        _trace_step(
            "motion-planner",
            "plan",
            "Intent Schema → module plan → MotionEditPlan",
            args={"prompt": prompt, "mode": mode},
            artifacts={
                "intent_bundle": str(bundle_path),
                "motion_edit_plan": str(plan_path),
            },
            detail={
                "primary_action": plan.get("primary_action"),
                "mf_text": plan.get("mf_text"),
                "source_policy": plan.get("source_policy"),
                "edit_source": plan.get("edit_source"),
                "preferred_source_id": plan.get("preferred_source_id"),
                "edit_mode": plan.get("edit_mode"),
                "modules": plan.get("modules"),
                "module_plan_steps": [
                    {"module": s.get("module"), "why": s.get("why")}
                    for s in (module_plan.get("steps") or [])
                ],
                "agent_role": AGENT_ROLES["planner"],
            },
            seconds=round(time.time() - t_plan, 3),
        )
    )

    # --- Coder ---
    coder_steps = _build_coder_steps(
        mode=mode,
        plan=plan,
        run_dir=run_dir,
        prev_dir=prev,
        length=length,
        bridge_iters=bridge_iters,
        skip_viz=skip_viz,
        force_bridged=force_bridged,
        edit_source_id=str(edit_source_id) if edit_source_id else None,
    )

    edited = run_dir / "03_edited.npy"
    preview = run_dir / "03_edited_preview.gif"
    coder_ok = True
    for step in coder_steps:
        if step["kind"] in ("copy", "ensure_copy"):
            src, dst = Path(step["src"]), Path(step["dst"])
            if step["kind"] == "ensure_copy" and dst.is_file():
                agent_trace.append(
                    _trace_step(
                        "motion-coder",
                        "coder",
                        step["title"],
                        ok=True,
                        detail="edited already present; skip fallback",
                    )
                )
                continue
            if not src.is_file():
                if step["kind"] == "ensure_copy":
                    agent_trace.append(
                        _trace_step(
                            "motion-coder",
                            "coder",
                            step["title"],
                            ok=False,
                            detail=f"fallback missing {src}",
                        )
                    )
                    continue
                coder_ok = False
                agent_trace.append(
                    _trace_step(
                        "motion-coder",
                        "coder",
                        step["title"],
                        ok=False,
                        detail=f"missing {src}",
                    )
                )
                break
            shutil.copy2(src, dst)
            agent_trace.append(
                _trace_step(
                    "motion-coder",
                    "coder",
                    step["title"],
                    artifacts={"copied": str(dst)},
                    detail={"src": str(src), "ensure": step["kind"] == "ensure_copy"},
                )
            )
            continue

        mid = step["id"]
        args = step["args"]
        soft = bool(step.get("soft"))
        t_m = time.time()
        result = run_module(mid, args)
        module_results.append(result)
        agent_trace.append(
            _trace_step(
                "motion-coder",
                "run_module",
                step.get("title") or mid,
                module=mid,
                args=_safe_args(args),
                artifacts=result.get("artifacts") or {},
                ok=bool(result.get("ok")),
                detail=result.get("error"),
                seconds=result.get("seconds") or round(time.time() - t_m, 3),
            )
        )
        if not result.get("ok") and not soft:
            coder_ok = False
            break

    # --- Tester ---
    quality_path = run_dir / "04_quality.json"
    quality: Dict[str, Any] = {
        "passed": False,
        "score": 0,
        "hard_failures": ["missing_edited"],
        "checks": [],
    }
    if edited.is_file():
        q_res = run_module(
            "quality_gate",
            {
                "edited": str(edited),
                "out": str(quality_path),
                "action": plan.get("primary_action"),
                "intent_json": str(plan_path),
            },
        )
        module_results.append(q_res)
        if q_res.get("ok") and quality_path.is_file():
            quality = json.loads(quality_path.read_text(encoding="utf-8-sig"))
        elif q_res.get("report"):
            quality = q_res["report"]
        agent_trace.append(
            _trace_step(
                "motion-tester",
                "quality",
                "quality_gate",
                module="quality_gate",
                args={"edited": str(edited), "action": plan.get("primary_action")},
                artifacts={"quality_json": str(quality_path) if quality_path.is_file() else None},
                ok=bool(q_res.get("ok")),
                detail={
                    "passed": quality.get("passed"),
                    "score": quality.get("score"),
                    "hard_failures": quality.get("hard_failures"),
                    "agent_role": AGENT_ROLES["tester"],
                },
                seconds=q_res.get("seconds"),
            )
        )
        # Online update edit recipe weights (learn from success / failure)
        try:
            from edit_memory import record_outcome

            refine = plan.get("refine") or {}
            recipe_ids = list(refine.get("matched_recipes") or [])
            if recipe_ids:
                ok_q = bool(quality.get("passed")) or float(quality.get("score") or 0) >= 70
                record_outcome(recipe_ids, success=ok_q, prompt=prompt)
                agent_trace.append(
                    _trace_step(
                        "motion-documenter",
                        "edit_memory",
                        f"Update recipes {recipe_ids} success={ok_q}",
                        ok=True,
                        detail={"recipe_ids": recipe_ids, "success": ok_q},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[orchestrator] edit_memory update skipped: {exc}", flush=True)
    else:
        agent_trace.append(
            _trace_step(
                "motion-tester",
                "quality",
                "quality_gate skipped (no edited npy)",
                ok=False,
                detail="missing_edited",
            )
        )

    # --- Motion→text feedback (for next revise / LLM refine) ---
    caption: Dict[str, Any] = {}
    caption_path = run_dir / "06_motion_caption.json"
    if edited.is_file():
        try:
            from motion_caption import caption_edited

            caption = caption_edited(
                edited,
                quality=quality,
                action=str(plan.get("primary_action") or ""),
                intent_text=str(prompt or ""),
                run_dir=run_dir,
            )
            caption_path.write_text(
                json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            agent_trace.append(
                _trace_step(
                    "motion-tester",
                    "motion_caption",
                    f"M2T feedback source={caption.get('source')}",
                    artifacts={"motion_caption": str(caption_path)},
                    ok=True,
                    detail={
                        "caption": (caption.get("caption") or "")[:240],
                        "problems": caption.get("problems"),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[orchestrator] motion_caption skipped: {exc}", flush=True)
            caption = {}

    # --- Documenter (light) ---
    summary = {
        "run_dir": str(run_dir),
        "mode": mode,
        "prompt": prompt,
        "task_id": task_id,
        "use_intent": True,
        "orchestrator": True,
        "previous_run": str(prev) if prev else None,
        "motion_edit_plan": {
            "primary_action": plan.get("primary_action"),
            "span": plan.get("span"),
            "source_policy": plan.get("source_policy"),
            "edit_source": plan.get("edit_source"),
            "preferred_source_id": plan.get("preferred_source_id"),
            "mld_text": plan.get("mld_text"),
            "mf_text": plan.get("mf_text"),
            "modules": plan.get("modules"),
            "edit_mode": plan.get("edit_mode"),
            "warning": plan.get("warning"),
        },
        "steps": {
            "intent": str(bundle_path),
            "edit": str(edited) if edited.is_file() else None,
            "quality": str(quality_path) if quality_path.is_file() else None,
            "motion_caption": str(caption_path) if caption_path.is_file() else None,
        },
        "quality": {
            "passed": quality.get("passed"),
            "score": quality.get("score"),
            "hard_failures": quality.get("hard_failures"),
            "path": str(quality_path) if quality_path.is_file() else None,
        },
        "motion_caption": {
            "caption": caption.get("caption"),
            "critique": caption.get("critique"),
            "source": caption.get("source"),
            "path": str(caption_path) if caption_path.is_file() else None,
        },
        "agent_trace": agent_trace,
        "total_seconds": round(time.time() - t0, 3),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    agent_trace.append(
        _trace_step(
            "motion-documenter",
            "document",
            "Write session summary",
            artifacts={"summary": str(summary_path)},
            detail={"agent_role": AGENT_ROLES["documenter"]},
        )
    )
    # refresh summary with final documenter step
    summary["agent_trace"] = agent_trace
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ok = bool(edited.is_file()) and bool(quality.get("passed", False)) and coder_ok
    response: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "ok": ok,
        "task_id": task_id,
        "mode": mode,
        "prompt": prompt,
        "run_dir": str(run_dir),
        "previous_run": str(prev) if prev else None,
        "artifacts": {
            "intent_bundle": str(bundle_path) if bundle_path.is_file() else None,
            "motion_edit_plan": str(plan_path) if plan_path.is_file() else None,
            "edited_npy": str(edited) if edited.is_file() else None,
            "preview_gif": str(preview) if preview.is_file() else None,
            "summary": str(summary_path),
            "quality": str(quality_path) if quality_path.is_file() else None,
            "motion_caption": str(caption_path) if caption_path.is_file() else None,
        },
        "motion_caption": caption or None,
        "intent": {
            "primary_action": plan.get("primary_action"),
            "modules": plan.get("modules"),
            "mf_text": plan.get("mf_text"),
            "mld_text": plan.get("mld_text"),
            "source_policy": plan.get("source_policy"),
            "edit_source": plan.get("edit_source"),
            "preferred_source_id": plan.get("preferred_source_id"),
            "span": plan.get("span"),
            "warning": plan.get("warning"),
            "edit_mode": plan.get("edit_mode"),
        },
        "quality": {
            "passed": quality.get("passed"),
            "score": quality.get("score"),
            "hard_failures": quality.get("hard_failures"),
            "soft_failures": quality.get("soft_failures"),
            "metrics": quality.get("metrics"),
            "timeline": quality.get("timeline"),
        },
        "timeline": quality.get("timeline")
        or (
            json.loads((run_dir / "05_timeline.json").read_text(encoding="utf-8-sig"))
            if (run_dir / "05_timeline.json").is_file()
            else None
        ),
        "agent_trace": agent_trace,
        "module_results": [
            {
                "module": r.get("module"),
                "ok": r.get("ok"),
                "seconds": r.get("seconds"),
                "artifacts": r.get("artifacts"),
                "error": r.get("error"),
            }
            for r in module_results
        ],
        "pipeline_seconds": summary["total_seconds"],
        "total_seconds": round(time.time() - t0, 3),
        "next_revise": {
            "mode": "revise",
            "previous_run": str(run_dir),
            "prompt": "<user improvement text>",
        },
        "error": None
        if ok
        else (
            "coder_failed"
            if not coder_ok
            else (
                "quality_gate_failed"
                if edited.is_file()
                else "missing_edited_npy"
            )
        ),
        "orchestrator": True,
    }

    out_path = run_dir / "session_response.json"
    out_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[orchestrator] ok={ok} action={plan.get('primary_action')} "
        f"quality={quality.get('score')} → {out_path}"
    )
    return response


def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate long paths for UI; keep keys intact."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 120:
            out[k] = "…" + v[-100:]
        else:
            out[k] = v
    return out


def _err(
    msg: str,
    *,
    mode: str = "generate",
    prompt: str = "",
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "ok": False,
        "task_id": None,
        "mode": mode,
        "prompt": prompt,
        "run_dir": str(run_dir) if run_dir else None,
        "previous_run": None,
        "artifacts": {},
        "intent": {},
        "quality": {},
        "agent_trace": [],
        "error": msg,
        "next_revise": None,
        "orchestrator": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrated session (planner→coder→tester)")
    ap.add_argument("--request-json", type=Path, default=None)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--mode", choices=["generate", "revise"], default="generate")
    ap.add_argument("--previous-run", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--edit-source-id", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.request_json:
        req = json.loads(args.request_json.read_text(encoding="utf-8-sig"))
    else:
        if not args.prompt:
            raise SystemExit("Provide --prompt or --request-json")
        req = {
            "prompt": args.prompt,
            "mode": args.mode,
            "previous_run": str(args.previous_run) if args.previous_run else None,
            "run_dir": str(args.run_dir) if args.run_dir else None,
            "skip_viz": bool(args.skip_viz),
            "length": args.length,
            "edit_source_id": args.edit_source_id,
        }

    resp = run_orchestrated_session(req, cwd=REPO_ROOT)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "ok": resp["ok"],
                "run_dir": resp.get("run_dir"),
                "error": resp.get("error"),
                "quality": (resp.get("quality") or {}).get("score"),
            },
            indent=2,
        )
    )
    if not resp.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
