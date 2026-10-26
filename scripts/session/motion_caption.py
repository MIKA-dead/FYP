"""
Motion → textual feedback for the agent (pipeline v1 style).

Primary path (always available): synthesize an English caption + issue list from
quality metrics + motion timeline (no VLM, no MotionsFix).

Optional path: external TM2T / MotionGPT via env MOTION_CAPTION_CMD, e.g.
  MOTION_CAPTION_CMD="python path/to/tm2t_infer.py --in {npy} --out {txt}"

  python scripts/session/motion_caption.py \\
      --edited outputs/.../03_edited.npy --quality 04_quality.json --out 06_caption.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _action_phrase(action: str) -> str:
    a = (action or "").strip()
    return {
        "sit_down": "sitting down",
        "wave_right": "waving the right hand",
        "wave_left": "waving the left hand",
        "turn_in_place": "turning in place",
        "walk_forward": "walking forward",
        "walk_slow": "walking slowly",
    }.get(a, a.replace("_", " ") or "moving")


def caption_from_quality_timeline(
    *,
    action: str = "",
    quality: Optional[Dict[str, Any]] = None,
    timeline: Optional[Dict[str, Any]] = None,
    intent_text: str = "",
) -> Dict[str, Any]:
    """Rule-based motion-to-text feedback (deterministic, no heavy model)."""
    quality = quality or {}
    timeline = timeline or quality.get("timeline") or {}
    metrics = dict(quality.get("metrics") or {})
    problems: List[str] = list(timeline.get("problems") or [])
    for ph in timeline.get("phases") or []:
        if ph.get("problem") and ph.get("id"):
            pid = str(ph["id"])
            if pid not in problems:
                problems.append(pid)
    for c in quality.get("checks") or []:
        if not c.get("ok") and c.get("id"):
            problems.append(str(c["id"]))
    # dedupe
    seen = set()
    problems = [p for p in problems if not (p in seen or seen.add(p))]

    bits: List[str] = []
    act = _action_phrase(action or str(quality.get("action") or ""))
    travel = float(metrics.get("root_travel") or 0)
    wrist = float(metrics.get("wrist_up_span") or metrics.get("wave_main_wrist_span") or 0)
    other = float(metrics.get("wave_other_wrist_span") or 0)
    drop = float(metrics.get("pelvis_drop") or 0)

    if str(action).startswith("wave") or "wave" in act:
        if travel < 0.25:
            bits.append("little forward walking (mostly in place or lateral hop)")
        elif travel < 0.55:
            bits.append("weak forward travel")
        else:
            bits.append("clear forward walking")
        if other > 0 and wrist > 0 and other >= wrist * 0.75:
            bits.append("both arms move similarly (not one-arm wave)")
        elif wrist > 0.08:
            bits.append("clear arm waving")
        if "both_arms_waving" in problems or any("双臂" in str(p) for p in problems):
            bits.append("suspected both-arm waving")
    elif str(action).startswith("sit"):
        if drop < 0.08:
            bits.append("pelvis barely lowers (failed or weak sit)")
        else:
            bits.append("pelvis lowers into a sit")
        if travel > 0.4:
            bits.append("notable horizontal travel before/during sit")
    elif "turn" in str(action):
        yaw = float(metrics.get("yaw_span_deg") or 0)
        bits.append(f"yaw change about {yaw:.0f} degrees")
    else:
        bits.append(f"root travel {travel:.2f}")

    phase_labels = [
        str(p.get("label") or p.get("id"))
        for p in (timeline.get("phases") or [])
        if p.get("problem")
    ]
    if phase_labels:
        bits.append("problem phases: " + ", ".join(phase_labels[:6]))

    caption = f"A person is {act}"
    if bits:
        caption += ", with " + "; ".join(bits)
    caption = caption.rstrip(".") + "."

    # Agent-oriented critique (English, for LLM refine)
    critique_parts: List[str] = []
    if intent_text:
        critique_parts.append(f"User wanted: {intent_text.strip()[:160]}")
    critique_parts.append(f"Observed: {caption}")
    if problems:
        critique_parts.append("Issues: " + ", ".join(problems[:12]))
    if str(action).startswith("wave"):
        critique_parts.append(
            "Prefer: walk forward continuously and wave only with one arm; "
            "keep the other arm down; avoid lateral hopping."
        )
    critique = " ".join(critique_parts)

    return {
        "schema_version": "motion_caption_v1",
        "source": "quality_timeline_rules",
        "action": action or quality.get("action"),
        "caption": caption,
        "critique": critique,
        "problems": problems,
        "metrics_used": {
            "root_travel": metrics.get("root_travel"),
            "wrist_up_span": metrics.get("wrist_up_span"),
            "wave_main_wrist_span": metrics.get("wave_main_wrist_span"),
            "wave_other_wrist_span": metrics.get("wave_other_wrist_span"),
            "pelvis_drop": metrics.get("pelvis_drop"),
            "yaw_span_deg": metrics.get("yaw_span_deg"),
        },
        "note": (
            "Rule-based M2T for agent feedback. "
            "Set MOTION_CAPTION_CMD for external TM2T/MotionGPT when available."
        ),
    }


def try_external_caption(edited_npy: Path, out_txt: Path) -> Optional[str]:
    """Optional external motion-to-text command."""
    tmpl = (os.environ.get("MOTION_CAPTION_CMD") or "").strip()
    if not tmpl:
        return None
    cmd = tmpl.format(npy=str(edited_npy), txt=str(out_txt), out=str(out_txt))
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), timeout=180)
        if proc.returncode != 0:
            return None
        if out_txt.is_file():
            return out_txt.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def caption_edited(
    edited_npy: Path,
    *,
    quality: Optional[Dict[str, Any]] = None,
    action: str = "",
    intent_text: str = "",
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    quality = quality or {}
    timeline = quality.get("timeline") or {}
    if run_dir and (Path(run_dir) / "05_timeline.json").is_file():
        try:
            timeline = json.loads(
                (Path(run_dir) / "05_timeline.json").read_text(encoding="utf-8-sig")
            )
        except Exception:  # noqa: BLE001
            pass

    base = caption_from_quality_timeline(
        action=action or str(quality.get("action") or ""),
        quality=quality,
        timeline=timeline,
        intent_text=intent_text,
    )

    if edited_npy.is_file():
        tmp = (run_dir or edited_npy.parent) / "_ext_caption.txt"
        ext = try_external_caption(edited_npy, tmp)
        if ext:
            base["external_caption"] = ext
            base["caption"] = ext
            base["source"] = "external_MOTION_CAPTION_CMD"
            base["critique"] = (
                f"User wanted: {intent_text.strip()[:160]}. "
                f"Observed (TM2T/external): {ext}. "
                f"Issues: {', '.join(base.get('problems') or []) or 'n/a'}."
            )
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Motion → text feedback for agent")
    ap.add_argument("--edited", type=Path, required=True)
    ap.add_argument("--quality", type=Path, default=None)
    ap.add_argument("--timeline", type=Path, default=None)
    ap.add_argument("--action", type=str, default="")
    ap.add_argument("--intent-text", type=str, default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    quality: Dict[str, Any] = {}
    if args.quality and args.quality.is_file():
        quality = json.loads(args.quality.read_text(encoding="utf-8-sig"))
    if args.timeline and args.timeline.is_file():
        quality = dict(quality)
        quality["timeline"] = json.loads(args.timeline.read_text(encoding="utf-8-sig"))

    out = caption_edited(
        args.edited,
        quality=quality,
        action=args.action or str(quality.get("action") or ""),
        intent_text=args.intent_text,
        run_dir=args.out.parent,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "caption": out.get("caption"), "source": out.get("source")}))


if __name__ == "__main__":
    main()
