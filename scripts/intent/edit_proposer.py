"""
Iteration controller for pipeline v1 (item b).

Reads two M2T captions — the clip before the last edit and the clip after it —
plus the quality metrics, and asks the LLM to either declare the goal met or
write the next relative edit instruction.

  python scripts/intent/edit_proposer.py --goal "a person jumps" \\
      --prev-caption 06_motion_caption.json --cur-caption 06_motion_caption.json

The reply is a closed JSON object:

  {"done": bool, "verdict": "...", "reason": "...",
   "next_edit_prompt": "...", "expected_change": "..."}

``rule_proposal`` is the deterministic fallback used when no LLM key is set or
the call fails, so the loop still terminates instead of stalling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "intent") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "intent"))

from llm_refine import _cfg, _extract_json, llm_available  # noqa: E402

CAPABILITIES = """What the editing backend can actually apply to the current clip:
- speed change (faster / slower), including "walk slower"
- more steps of the same gait ("walk a few more steps")
- raise one arm to an absolute height ("raise the right arm high / overhead"),
  measured as arm_raise_lift = wrist height above the shoulder
- retime or rescale one limb ("raise the right arm higher", "lift the right leg as much as the left")
- direction on the ground ("walk forward", "stop turning", "keep the path straight")
- remove a fault: no spin, no sideways drift, no extra walk before a sit
- replace the whole action with another dataset clip (last resort, breaks continuity)
It cannot: add an object, change the camera, change body shape, or invent a
brand-new action while keeping the performance.

The caption comes from a learned M2T model with a coarse vocabulary: it says
"wave" for any raised arm and does not name heights. Treat the measured metrics
as ground truth. arm_raise_ratio = wrist height above the shoulder divided by
head height above the shoulder: ≈1.0 = wrist level with the head, ≈1.8 = clearly
above the head, ≈2.6 = arm straight up. If arm_raise_ratio reaches the requested
level, the arm IS raised — do not ask again just because the caption says "wave"."""

SYSTEM = (
    "You are the iteration controller of a text-driven 3D motion editing agent.\n"
    "You see the user goal, the caption of the clip before the last edit, the "
    "caption after it, and the measured problems. Decide whether the goal is met.\n"
    "The captions are coarse; the metrics are measured on the joints and win "
    "when the two disagree.\n"
    "If not, write exactly ONE next edit instruction in the style of a short "
    "relative edit ('raise the right arm higher', 'walk two more steps', 'stop "
    "spinning', 'bend forward more'). It must be a change to the current clip, "
    "not a request for a different clip.\n"
    "Never repeat an instruction that was already tried and did not change the "
    "caption or the metrics. Prefer the single change that closes the largest "
    "gap between the current caption and the goal.\n"
    + CAPABILITIES
    + "\nReply with JSON only: "
    '{"done": true|false, "verdict": "goal_met|improved|no_change|worse", '
    '"reason": "one sentence citing both captions", '
    '"next_edit_prompt": "one short instruction, empty when done", '
    '"expected_change": "what the next caption should say"}'
)


def _caption_text(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return "(none)"
    return str(cap.get("caption") or "").strip() or "(empty caption)"


def _post(payload: Dict[str, Any], timeout_s: float = 120.0) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg.get("api_key"):
        raise RuntimeError("no LLM API key")
    body: Dict[str, Any] = {
        "model": cfg["model"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    body["response_format"] = {"type": "json_object"}
    if "opencode.ai" in cfg["base_url"]:
        body.pop("response_format", None)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
        "User-Agent": "BuildanAgent-pipeline-v1/1.0",
        "Accept": "application/json",
    }
    if "opencode.ai" in cfg["base_url"]:
        headers["x-opencode-session"] = os.environ.get("REFINE_LLM_SESSION") or "pipeline-v1"
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    content = (
        (((raw.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    )
    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError(f"LLM returned non-JSON: {content[:200]}")
    return parsed


def _normalise(obj: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    done = bool(obj.get("done"))
    prompt = str(obj.get("next_edit_prompt") or "").strip()
    if done:
        prompt = ""
    return {
        "schema_version": "edit_proposal_v1",
        "source": source,
        "done": done,
        "verdict": str(obj.get("verdict") or ("goal_met" if done else "improved")),
        "reason": str(obj.get("reason") or "").strip(),
        "next_edit_prompt": prompt,
        "expected_change": str(obj.get("expected_change") or "").strip(),
    }


def rule_proposal(
    *,
    goal: str,
    prev_caption: Optional[Dict[str, Any]],
    cur_caption: Optional[Dict[str, Any]],
    quality: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Deterministic fallback: turn the largest measured problem into one edit."""
    quality = quality or {}
    metrics = quality.get("metrics") or {}
    problems = set((cur_caption or {}).get("problems") or [])
    tried = {
        str(h.get("next_edit_prompt") or "").strip().lower()
        for h in (history or [])
        if h.get("next_edit_prompt")
    }

    candidates: List[str] = []
    yaw = float(metrics.get("yaw_span_deg") or 0)
    lateral = float(metrics.get("path_lateral_max") or 0)
    travel = float(metrics.get("root_travel") or 0)
    drop = float(metrics.get("pelvis_drop") or 0)
    if "pre_sit_locomotion" in problems or "preamble" in problems:
        candidates.append("remove the walking before the sit")
    if "turn_anti_slide" in problems or (travel > 0.65 and "turn" in goal.lower()):
        candidates.append("stop sliding across the floor while turning")
    if yaw > 40:
        candidates.append("keep facing the same direction, do not spin")
    if lateral > 0.12:
        candidates.append("keep the path straight, do not drift sideways")
    if "sit" in goal.lower() and drop and drop < 0.15:
        candidates.append("lower the hips further into the sit")
    if any(w in goal.lower() for w in ("walk", "run")) and travel < 0.6:
        candidates.append("walk forward further")
    if any(w in goal.lower() for w in ("raise", "wave", "arm")):
        candidates.append("raise the arm higher")
    if any(w in goal.lower() for w in ("kick", "leg")):
        candidates.append("lift the kicking leg higher")

    for cand in candidates:
        if cand.lower() not in tried:
            return _normalise(
                {
                    "done": False,
                    "verdict": "improved",
                    "reason": "Measured problems remain; no LLM key, using the metric rule.",
                    "next_edit_prompt": cand,
                    "expected_change": f"the caption stops reporting: {cand}",
                },
                source="rules",
            )
    return _normalise(
        {
            "done": True,
            "verdict": "no_change",
            "reason": "Every metric-level fix has been tried; stopping the loop.",
            "next_edit_prompt": "",
            "expected_change": "",
        },
        source="rules",
    )


def propose_next_edit(
    *,
    goal: str,
    prev_caption: Optional[Dict[str, Any]],
    cur_caption: Optional[Dict[str, Any]],
    quality: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    iteration: int = 1,
    max_iters: int = 3,
    require_llm: bool = False,
) -> Dict[str, Any]:
    payload = {
        "iteration": iteration,
        "max_iterations": max_iters,
        "goal": goal,
        "caption_before_last_edit": _caption_text(prev_caption),
        "caption_after_last_edit": _caption_text(cur_caption),
        "caption_model": (cur_caption or {}).get("source"),
        "measured_problems": (cur_caption or {}).get("problems") or [],
        "quality_score": (quality or {}).get("score"),
        "quality_passed": (quality or {}).get("passed"),
        "metrics": {
            k: (quality or {}).get("metrics", {}).get(k)
            for k in (
                "root_travel",
                "yaw_span_deg",
                "pelvis_drop",
                "path_lateral_max",
                "body_moving",
                "arm_raise_lift",
                "arm_raise_side",
                "arm_raise_ratio",
                "arm_raise_head_span",
                "head_rel_height",
                "wrist_up_span",
            )
        },
        "history": [
            {
                "prompt": h.get("next_edit_prompt"),
                "caption_after": h.get("caption_after"),
                "verdict": h.get("verdict"),
            }
            for h in (history or [])
        ],
    }
    if not llm_available():
        if require_llm:
            raise RuntimeError(
                "LLM not configured (no REFINE_LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY "
                "and no opencode auth.json)"
            )
        return rule_proposal(
            goal=goal, prev_caption=prev_caption, cur_caption=cur_caption,
            quality=quality, history=history,
        )
    try:
        return _normalise(_post(payload), source="llm")
    except Exception as exc:  # noqa: BLE001
        if require_llm:
            raise RuntimeError(f"LLM proposal failed: {exc}") from exc
        print(f"[edit_proposer] LLM failed ({exc}); using rules", flush=True)
        fallback = rule_proposal(
            goal=goal, prev_caption=prev_caption, cur_caption=cur_caption,
            quality=quality, history=history,
        )
        fallback["llm_error"] = str(exc)[:300]
        return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose the next edit prompt from two M2T captions")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--prev-caption", type=Path, default=None)
    ap.add_argument("--cur-caption", type=Path, default=None)
    ap.add_argument("--quality", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    def _load(p: Optional[Path]) -> Optional[Dict[str, Any]]:
        if p and p.is_file():
            return json.loads(p.read_text(encoding="utf-8-sig"))
        return None

    out = propose_next_edit(
        goal=args.goal,
        prev_caption=_load(args.prev_caption),
        cur_caption=_load(args.cur_caption),
        quality=_load(args.quality),
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
