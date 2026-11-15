"""
Generate in four stages:

1. Fast draft from the first text hit. It may be the wrong motion.
2. Observe that draft (metrics, timeline, caption).
3. Change only a fault the observation named.
4. Draw the skeleton preview.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dataset_edit import _load_hml, _rank_captions, align_to_source
from initial_motion import repair_performance
from constraint_postprocess import straighten_yaw_and_path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "outputs" / "v05" / "session_more" / "slow" / "03_edited.before_initial.npy"
MAX_FRAMES = 100


def _reference() -> np.ndarray:
    obj = np.load(REFERENCE, allow_pickle=True).item()
    return np.asarray(obj["joint_positions"], dtype=np.float32)


def _save(path: Path, joints: np.ndarray, meta: Dict[str, Any]) -> None:
    np.save(path, {"joint_positions": np.asarray(joints, dtype=np.float32), "stage": meta})


def draft_clip(
    prompt: str,
    *,
    run_dir: Optional[Path] = None,
    generator: str = "dataset",
    draft_joints: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """First caption match, aligned to the preview skeleton. No diffusion.

    ``generator="mld"`` runs the MLD latent-diffusion text→motion model first
    and uses that clip as the draft (slow, ~30 s, needs the ``mld`` env).

    ``draft_joints`` takes an already-generated joint array instead (one of the
    samples from ``mld_generate --samples N``). MLD seeds the RNG at import
    (``mld/utils/fixseed.py``), so re-running the same text reproduces the same
    clip; drawing several samples inside one process is the only way to get
    independent drafts, and this is how they re-enter the pipeline.
    """
    if draft_joints is not None:
        path = Path(draft_joints)
        if not path.is_file():
            raise FileNotFoundError(f"draft_joints not found: {path}")
        raw = np.load(path, allow_pickle=True)
        if raw.dtype == object:
            raw = np.asarray(raw.item().get("joint_positions"))
        joints, converted = align_to_source(np.asarray(raw, dtype=np.float32), _reference())
        if joints.shape[0] > MAX_FRAMES:
            joints = joints[:MAX_FRAMES]
        return joints.astype(np.float32), {
            "source_id": "mld_sample",
            "caption": prompt,
            "text_score": 1.0,
            "converted_up_axis": bool(converted),
            "T": int(joints.shape[0]),
            "generator": "mld",
            "mld_out": str(path),
        }
    if generator == "mld":
        return _mld_draft(prompt, run_dir=run_dir)
    ranked = _rank_captions(prompt, limit=5)
    if not ranked:
        raise RuntimeError(f"no text hit for {prompt!r}")
    row = ranked[0]
    raw = _load_hml(row["id"])
    joints, converted = align_to_source(raw, _reference())
    if joints.shape[0] > MAX_FRAMES:
        joints = joints[:MAX_FRAMES]
    meta = {
        "source_id": row["id"],
        "caption": row["caption"],
        "text_score": round(float(row["score"]), 3),
        "converted_up_axis": bool(converted),
        "T": int(joints.shape[0]),
    }
    return joints.astype(np.float32), meta


def _mld_draft(prompt: str, *, run_dir: Optional[Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Text → MLD diffusion → joints, then map into the preview skeleton.

    When the prompt names a measurable attribute the editor cannot create
    ("in place", "turns left", "jumps"), more than one draft is drawn in a single
    MLD process and the closest one is kept — generation-side rejection sampling
    (``scripts/agent/sampling.py``). One draw cannot distinguish "the model
    cannot do it" from "this draw missed": the same "walks in place" prompt gave
    1.35 m, 0.12 m and 0.02 m of travel on three draws.
    """
    import os
    import subprocess

    _agent_dir = REPO / "scripts" / "agent"
    if str(_agent_dir) not in sys.path:
        sys.path.insert(0, str(_agent_dir))
    import sampling

    py = os.environ.get("MLD_PYTHON") or str(
        Path.home() / "miniconda3" / "envs" / "mld" / "python.exe"
    )
    out = (run_dir or REPO / "outputs" / "v05" / "_mld_run") / "01_mld_joints.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    want = sampling.batch_size(prompt)
    cmd = [
        py,
        str(REPO / "scripts" / "backends" / "mld_generate.py"),
        "--text",
        prompt,
        "--length",
        "60",
        "--out",
        str(out),
    ]
    if want > 1:
        cmd += ["--samples", str(want)]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, timeout=900)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"MLD generation failed (exit {proc.returncode})")

    # One draw, or every draw of the batch, mapped into the preview skeleton.
    paths = [out]
    meta_path = out.with_suffix(".json")
    if meta_path.is_file():
        try:
            meta_json = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            paths = [Path(p) for p in (meta_json.get("samples") or [])] or [out]
        except Exception:  # noqa: BLE001
            paths = [out]

    from metrics import kinematic_summary
    from motion_validity import physical_report, reground

    cands: List[Dict[str, Any]] = []
    drafts: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    for i, path in enumerate(paths, start=1):
        if not Path(path).is_file():
            continue
        raw = np.load(path, allow_pickle=True)
        if raw.dtype == object:
            raw = np.asarray(raw.item().get("joint_positions"))
        joints, converted = align_to_source(np.asarray(raw, dtype=np.float32), _reference())
        if joints.shape[0] > MAX_FRAMES:
            joints = joints[:MAX_FRAMES]
        grounded, _shift = reground(joints)
        m = kinematic_summary(grounded)
        cands.append({
            "cand": i,
            "ok": True,
            "valid": bool(physical_report(grounded)["valid"]),
            "metrics": m,
            "path": str(path),
        })
        drafts.append((joints, {"converted_up_axis": bool(converted),
                               "T": int(joints.shape[0]), "sample": i, "path": str(path)}))
    if not drafts:
        raise RuntimeError("MLD produced no readable sample")

    best, accepted = sampling.select(prompt, cands)
    pick = int((best or cands[0])["cand"]) - 1
    joints, info = drafts[pick]
    rejection = None
    if len(cands) > 1:
        rejection = sampling.summary(prompt, cands, best)
        rejection["chosen"] = pick + 1
        rejection["paths"] = [c["path"] for c in cands]
        rejection["per_candidate"] = [
            {"cand": c["cand"], "gap": c["gap"], "satisfied": c["satisfied"],
             "valid": c["valid"]}
            for c in sampling.rank(prompt, cands)
        ]
    meta = {
        "source_id": "mld",
        "caption": prompt,
        "text_score": 1.0,
        "converted_up_axis": bool(info["converted_up_axis"]),
        "T": int(info["T"]),
        "generator": "mld",
        "mld_out": info["path"],
        "rejection": rejection,
    }
    return joints.astype(np.float32), meta


def _observe(joints: np.ndarray, action: str, prompt: str) -> Dict[str, Any]:
    session = str(REPO / "scripts" / "session")
    if session not in sys.path:
        sys.path.insert(0, session)
    from motion_caption import caption_from_quality_timeline
    from quality_gate import evaluate_joints

    quality = evaluate_joints(joints, action=action)
    timeline = quality.get("timeline") or {}
    caption = caption_from_quality_timeline(
        action=action,
        quality=quality,
        timeline=timeline,
        intent_text=prompt,
    )
    return {"quality": quality, "timeline": timeline, "caption": caption}


def _plant_root(joints: np.ndarray) -> np.ndarray:
    """Drop horizontal root travel. The pose relative to the pelvis stays."""
    from constraint_postprocess import _horiz, _up_axis

    j = np.asarray(joints, dtype=np.float32).copy()
    up = _up_axis(j)
    hz = _horiz(up)
    delta = j[:, 0, hz] - j[0, 0, hz]
    j[:, :, hz[0]] -= delta[:, 0][:, None]
    j[:, :, hz[1]] -= delta[:, 1][:, None]
    return j


def _edit(joints: np.ndarray, action: str, observation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Change only a fault named by the observation."""
    metrics = (observation.get("quality") or {}).get("metrics") or {}
    problems = set((observation.get("caption") or {}).get("problems") or [])
    yaw = float(metrics.get("yaw_span_deg") or 0)
    lateral = float(metrics.get("path_lateral_max") or 0)
    travel = float(metrics.get("root_travel") or 0)
    moving = int(metrics.get("body_moving") or 0)
    action = action or ""
    notes: List[str] = []
    clip = joints
    if "sit" in action and yaw > 45.0:
        clip, _meta = repair_performance(action, clip)
        notes.append(f"去掉观测到的转身 {yaw:.0f}°")
    if "sit" in action and ("pre_sit_locomotion" in problems or "preamble" in problems):
        start = int((observation.get("timeline") or {}).get("sit_start") or 0)
        if start >= 8 and clip.shape[0] - start >= 30:
            clip = clip[max(0, start - 4) :]
            notes.append(f"去掉坐下前的走动，从第 {start} 帧留下")
    if "turn" in action and yaw < 150.0 and moving >= 8:
        clip, _meta = repair_performance(action, clip)
        notes.append(f"把观测到的 {yaw:.0f}° 补到半圈，肢体保留")
    if "turn" in action and (travel > 0.65 or "turn_anti_slide" in problems):
        clip = _plant_root(clip)
        notes.append(f"去掉观测到的位移 {travel:.2f}")
    if (action.startswith("walk") or action.startswith("wave")) and (yaw > 40.0 or lateral > 0.12):
        clip, _info = straighten_yaw_and_path(clip)
        notes.append(f"拉直观测到的转角 {yaw:.0f}°、侧向 {lateral:.2f}")
    if not notes:
        return joints, {"change": "kept", "because": "观测没有指出这一步能改的一项，草稿保持原样"}
    return clip, {"change": "edit", "because": "；".join(notes)}


def generate_stages(
    run_dir: Path,
    prompt: str,
    action: str,
    *,
    skip_viz: bool = False,
    generator: str = "dataset",
    draft_joints: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write draft, observation, edited clip, and preview. Return the four notes."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    draft, draft_meta = draft_clip(prompt, run_dir=run_dir, generator=generator,
                                   draft_joints=draft_joints)
    _save(run_dir / "01_draft.npy", draft, draft_meta)
    observation = _observe(draft, action, prompt)
    (run_dir / "01_observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    edited, edit_meta = _edit(draft, action, observation)
    _save(run_dir / "03_edited.npy", edited, {"draft": draft_meta, "edit": edit_meta})
    preview = run_dir / "03_edited_preview.gif"
    if not skip_viz:
        from visualize_motion import render_skeleton_video

        render_skeleton_video(edited, preview, title=action or "draft")
    cap = (observation.get("caption") or {}).get("caption") or ""
    problems = (observation.get("caption") or {}).get("problems") or []
    stages: List[Dict[str, Any]] = [
        {
            "step": 1,
            "name": "快速生成",
            "detail": (
                f"MLD 扩散生成 {draft_meta['T']} 帧（text→motion）。"
                if draft_meta.get("generator") == "mld"
                else (
                    f"文本命中 {draft_meta['source_id']}「{draft_meta['caption']}」，"
                    f"{draft_meta['T']} 帧。不跑扩散，允许不准。"
                )
            ),
        },
        {
            "step": 2,
            "name": "观测",
            "detail": cap + (f" 问题：{', '.join(problems)}。" if problems else " 没有列出问题。"),
        },
        {
            "step": 3,
            "name": "修改",
            "detail": f"{edit_meta.get('change')}。{edit_meta.get('because') or ''}",
        },
        {
            "step": 4,
            "name": "可视化",
            "detail": str(preview.name) if preview.is_file() else "未出图",
        },
    ]
    (run_dir / "03_edited.stages.json").write_text(
        json.dumps({"stages": stages, "draft": draft_meta, "edit": edit_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"stages": stages, "draft": draft_meta, "edit": edit_meta, "observation": observation}
