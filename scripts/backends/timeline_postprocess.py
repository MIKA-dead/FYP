"""
Action-agnostic timeline edits: phase_speed + drop_phases.

Used for wave/walk/turn (and sit when only speed/drop is needed without sit-hold geometry).

  python scripts/backends/timeline_postprocess.py \\
      --in 03_edited.npy --out 03_edited.npy --intent-json plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_SESSION = Path(__file__).resolve().parents[1] / "session"
if str(_SESSION) not in sys.path:
    sys.path.insert(0, str(_SESSION))
from motion_timeline import (  # noqa: E402
    analyze_motion_timeline,
    apply_phase_speed,
    drop_timeline_phases,
)


def _load_joints(path: Path):
    raw = np.load(path, allow_pickle=True)
    if raw.dtype != object:
        raise SystemExit(f"Expected object npy with joint_positions, got {raw.dtype}")
    obj = raw.item()
    if not isinstance(obj, dict) or "joint_positions" not in obj:
        raise SystemExit(f"Need joint_positions in {path}")
    return obj, np.asarray(obj["joint_positions"], dtype=np.float32)


def process_file(
    path_in: Path,
    path_out: Path,
    *,
    action: str = "",
    phase_speed: Optional[Dict[str, float]] = None,
    drop_phases: Optional[List[str]] = None,
    timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj, joints = _load_joints(path_in)
    phase_speed = dict(phase_speed or {})
    drop_phases = list(drop_phases or [])
    meta: Dict[str, Any] = {
        "action": action,
        "T_in": int(joints.shape[0]),
        "phase_speed": phase_speed,
        "drop_phases": drop_phases,
    }
    tl = timeline if isinstance(timeline, dict) and timeline.get("phases") else None
    if tl is None:
        tl = analyze_motion_timeline(joints, action=action or "", forbid=[])
    meta["timeline"] = tl

    j = joints
    if drop_phases:
        j, dmeta = drop_timeline_phases(j, tl, drop_phases)
        meta["drop_meta"] = dmeta
        tl = analyze_motion_timeline(j, action=action or "", forbid=[])
        meta["timeline_after_drop"] = tl
    if phase_speed:
        j, pmeta = apply_phase_speed(j, tl, phase_speed)
        meta["phase_speed_meta"] = pmeta
        tl = analyze_motion_timeline(j, action=action or "", forbid=[])
    meta["timeline_after"] = tl
    meta["T_out"] = int(j.shape[0])

    obj = dict(obj)
    obj["joint_positions"] = j
    if "pose" in obj:
        pose = np.asarray(obj["pose"], dtype=np.float32).copy()
        T = min(pose.shape[0], j.shape[0])
        pose[:T, :3] = j[:T, 0]
        if j.shape[0] < pose.shape[0]:
            pose = pose[: j.shape[0]]
        obj["pose"] = pose
    obj["timeline_post_meta"] = meta
    if tl:
        obj["motion_timeline"] = tl
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_out, obj, allow_pickle=True)
    path_out.with_suffix(".timeline_post.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if tl:
        (path_out.parent / "05_timeline.json").write_text(
            json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[timeline_post] action={action} T={meta['T_in']}→{meta['T_out']} "
        f"speed={list(phase_speed)} drop={drop_phases} → {path_out}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Timeline phase speed / drop postprocess")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--action", type=str, default="")
    ap.add_argument("--intent-json", type=Path, default=None)
    ap.add_argument("--timeline-json", type=Path, default=None)
    args = ap.parse_args()

    action = args.action or ""
    phase_speed: Dict[str, float] = {}
    drop_phases: List[str] = []
    timeline = None
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        refine = plan.get("refine") or (plan.get("revision") or {}).get("refine") or {}
        action = action or str(
            refine.get("keep_action") or plan.get("primary_action") or ""
        )
        if refine.get("phase_speed"):
            phase_speed = {
                str(k): float(v) for k, v in dict(refine["phase_speed"]).items()
            }
        if refine.get("drop_phases"):
            drop_phases = [str(p) for p in refine["drop_phases"]]
    if args.timeline_json and args.timeline_json.is_file():
        timeline = json.loads(args.timeline_json.read_text(encoding="utf-8-sig"))

    process_file(
        args.inp,
        args.out,
        action=action,
        phase_speed=phase_speed or None,
        drop_phases=drop_phases or None,
        timeline=timeline,
    )


if __name__ == "__main__":
    main()
