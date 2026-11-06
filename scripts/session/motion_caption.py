"""
Motion → textual feedback for the agent (pipeline v1).

Backends, in the order ``caption_edited`` tries them:

1. ``tm2t`` — the learned motion-to-text model in ``scripts/backends/tm2t_infer.py``
   (TM2T M2T transformer + VQ tokenizer). Default when its checkpoints exist and
   ``M2T_BACKEND`` is not set to something else.
2. ``external`` — any command via ``MOTION_CAPTION_CMD``, e.g.
   ``MOTION_CAPTION_CMD="python path/to/m2t.py --in {npy} --out {txt}"``.
3. ``rules`` — deterministic sentence from quality metrics + timeline. Always
   available; keeps the loop running when no model is installed.

``M2T_BACKEND=rules`` forces the rule sentence, ``M2T_BACKEND=external`` forces
the command hook, ``M2T_BACKEND=tm2t`` refuses to fall back silently.

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

    yaw = float(metrics.get("yaw_span_deg") or 0)
    lateral = float(metrics.get("path_lateral_max") or 0)
    if "turn" not in str(action) and yaw >= 40:
        bits.append(f"body yaw changes about {yaw:.0f} degrees")
    if travel > 0.2 and lateral >= 0.15:
        bits.append(f"the path bends sideways (lateral {lateral:.2f})")

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


def try_tm2t_caption(edited_npy: Path, *, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Learned M2T: TM2T M2T transformer on the edited joints.

    Cached per clip under ``cache_dir/06_motion_caption.tm2t.json`` so a loop that
    re-reads the same clip does not reload the checkpoints.
    """
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / "06_motion_caption.tm2t.json"
        if cache.is_file():
            try:
                data = json.loads(cache.read_text(encoding="utf-8-sig"))
                if data.get("caption"):
                    return data
            except Exception:  # noqa: BLE001
                pass
    backends = REPO_ROOT / "scripts" / "backends"
    if str(backends) not in sys.path:
        sys.path.insert(0, str(backends))
    data: Optional[Dict[str, Any]] = None
    try:
        import torch  # noqa: F401  (TM2T needs torch; the base interpreter may not have it)
        from tm2t_infer import describe, m2t_ready

        ready, _why = m2t_ready()
        if not ready:
            return None
        data = describe(Path(edited_npy))
    except ImportError:
        # Run the M2T model in the conda env that owns torch, then read its JSON.
        py = os.environ.get("MOTIONFIX_PYTHON") or str(
            Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"
        )
        if not Path(py).is_file():
            return None
        tmp = Path(cache_dir or edited_npy.parent) / "_tm2t_caption.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        try:
            proc = subprocess.run(
                [py, str(backends / "tm2t_infer.py"), "--npy", str(edited_npy), "--out", str(tmp)],
                cwd=str(REPO_ROOT), env=env, timeout=300,
            )
            if proc.returncode != 0 or not tmp.is_file():
                return None
            data = json.loads(tmp.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            print(f"[motion_caption] tm2t subprocess failed: {exc}", flush=True)
            return None
    except SystemExit as exc:
        print(f"[motion_caption] tm2t unavailable: {exc}", flush=True)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[motion_caption] tm2t failed: {exc}", flush=True)
        return None
    if cache is not None and data and data.get("caption"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data



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
        mode = (os.environ.get("M2T_BACKEND") or "auto").strip().lower()
        if mode in ("rules", "off", "none"):
            return base
        if mode in ("auto", "tm2t"):
            learned = try_tm2t_caption(edited_npy, cache_dir=run_dir or edited_npy.parent)
            if learned and learned.get("caption"):
                sentence = str(learned["caption"]).strip()
                base["m2t"] = learned
                base["caption"] = sentence
                base["source"] = "tm2t_m2t_transformer"
                base["critique"] = (
                    f"User wanted: {intent_text.strip()[:160]}. "
                    f"Observed (TM2T): {sentence}. "
                    f"Issues: {', '.join(base.get('problems') or []) or 'n/a'}."
                )
                return base
            if mode == "tm2t":
                base["m2t_error"] = "TM2T checkpoints missing or inference failed"
                return base
        if mode in ("auto", "external"):
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
