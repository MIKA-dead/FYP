"""
LLM-driven tool loop for the agentic text→motion pipeline.

    goal (free text)
      → the LLM reasons, calls generate_motion
      → calls describe_motion (learned M2T caption + measurements)
      → compares caption/measurements with the goal, writes the gap in text
      → calls edit_motion with one instruction
      → describe again …
      → calls finish when the measured clip satisfies the goal

The controller never edits a clip itself; it only chooses a tool and writes the
text that the tool consumes. Every decision, observation and assessment is
written to ``agent_run.json`` / ``agent_run.md``.

  python scripts/agent/agent.py --goal "a person raises the right arm high"
  python scripts/agent/agent.py --goal "..." --no-llm      # scripted baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "scripts" / "agent",
    REPO / "scripts" / "intent",
    REPO / "scripts" / "session",
    REPO / "scripts" / "orchestrator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics import kinematic_summary, load_joints, text_similarity  # noqa: E402
from tools import SUPPORTED_EDITS, TOOL_SPECS, MotionTools  # noqa: E402

SYSTEM = (
    "You are the controller of a text-driven 3D motion agent. You never write "
    "motion yourself: you call one tool at a time and keep looping until the "
    "measured clip satisfies the user.\n\n"
    "TOOLS\n"
    + "\n".join(f"- {t['name']}({', '.join(t['args'].keys())}): {t['description']}" for t in TOOL_SPECS)
    + "\n\nMANDATORY ORDER\n"
    "1. With no clip: call generate_motion.\n"
    "2. After any clip is produced or edited: call describe_motion before judging it.\n"
    "3. Before finish: call check_requirement and read its verdict.\n"
    "4. Then either edit_motion (one change) or finish.\n\n"
    "HOW TO JUDGE\n"
    "- The learned caption is coarse; joint measurements are ground truth when "
    "they disagree. arm_raise_ratio >= 1.8 and wrist_above_head (metres) > 0 both "
    "mean the wrist is above the head. root_travel is metres travelled; "
    "pelvis_drop is how far the hips descend; yaw_span_deg is how much the body turns.\n"
    "- alignment (0-100) must cite both the caption and at least one measurement.\n"
    "- Judge the user's actual wording, not an idealized version: 'turns around' is "
    "met by a visible ~180° turn, and 'raises the arm' is met by one arm above the "
    "head; do not demand a 360° spin or a specific style the user never asked for.\n"
    "- `missing` lists concrete unmet requirements, not general quality comments.\n"
    "- `next_edit` is ONE relative instruction for the current clip.\n"
    "- The decision is yours: check_requirement reports measurements, it does not "
    "command you. If you disagree with it, say why in `thought`, but never claim a "
    "measurement that the observation contradicts.\n"
    "- If an observation says no_change=true, that instruction cannot be applied; "
    "never repeat it — try a different supported edit or finish with the gap.\n"
    "- An edit that returns unsupported_instruction never reached the backend. Do "
    "not rephrase the same impossible request; finish and name it as out of scope.\n"
    "- A clip with valid=false is physically broken: never accept it. You may call "
    "generate_motion again with a simpler prompt, or finish and report the failure.\n"
    "- quality_passed=false lists the gate's failed checks; weigh them but they are "
    "not the user's request — the requirement checks are.\n"
    "- edit_motion reports new_hard_failures and pre_existing_hard_failures: only "
    "the new ones are damage from your instruction; the pre-existing ones were "
    "already true of the clip you started from, so do not 'fix' them with an "
    "unrelated edit.\n"
    "- If two consecutive edits failed to raise alignment, finish and say so; "
    "do not loop forever.\n\n"
    + SUPPORTED_EDITS
    + "\n\nReply with JSON only, no markdown:\n"
    '{"thought": "one or two sentences", '
    '"assessment": {"alignment": 0, "satisfied": false, "matches": [], '
    '"missing": [], "next_edit": ""}, '
    '"action": "generate_motion|describe_motion|check_requirement|edit_motion|finish", '
    '"args": {}}'
)


def _cfg() -> Dict[str, str]:
    from llm_refine import _cfg as cfg  # noqa: WPS433

    return cfg()


def llm_available() -> bool:
    from llm_refine import llm_available as available

    return available()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:  # noqa: BLE001
                    start, depth = None, 0
    return None


def chat(messages: List[Dict[str, str]], *, timeout_s: float = 180.0) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """One chat-completions call. Returns (parsed_json, usage, raw_text)."""
    cfg = _cfg()
    if not cfg.get("api_key"):
        raise RuntimeError("no LLM API key")
    body: Dict[str, Any] = {
        "model": cfg["model"],
        "temperature": 0.2,
        "messages": messages,
    }
    if "opencode.ai" not in cfg["base_url"]:
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
        "User-Agent": "BuildanAgent-agent/1.0",
        "Accept": "application/json",
    }
    if "opencode.ai" in cfg["base_url"]:
        headers["x-opencode-session"] = os.environ.get("REFINE_LLM_SESSION") or "agent-loop"
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = (((raw.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    parsed = _extract_json(content)
    if parsed is None:
        raise RuntimeError(f"LLM returned non-JSON: {content[:300]}")
    return parsed, raw.get("usage") or {}, content


# --------------------------------------------------------------------------
# scripted baseline (no LLM): same tools, deterministic gap→edit rules
# --------------------------------------------------------------------------
def _rule_assess(goal: str, obs: Dict[str, Any], edits: int, max_edits: int,
                 failed: Optional[set] = None) -> Dict[str, Any]:
    failed = failed or set()
    m = obs.get("metrics") or {}
    cap = str(obs.get("caption") or "")
    g = (goal or "").lower()
    missing: List[str] = []
    instruction = ""
    side = "left" if "left" in g else "right"
    if any(w in g for w in ("arm", "hand", "wave", "raise")):
        ratio = float(m.get(f"arm_raise_{side}") or 0)
        above = float(m.get(f"wrist_above_head_{side}") or 0)
        if ratio < 1.8 and above < 0.05:
            missing.append(
                f"{side} arm not above the head (ratio {ratio:.2f} < 1.8, "
                f"wrist {above:.2f} m above head)"
            )
            instruction = f"raise the {side} arm higher"
    if any(w in g for w in ("walk", "step", "forward")):
        travel = float(m.get("root_travel") or 0)
        if travel < 0.5:
            missing.append(f"too little forward travel ({travel:.2f} m)")
            instruction = "walk forward further"
    if "sit" in g:
        drop = float(m.get("pelvis_drop") or 0)
        if drop < 0.10:
            missing.append(f"hips do not lower enough (drop {drop:.2f} m)")
            instruction = "sit down lower"
    if "turn" in g:
        yaw = float(m.get("yaw_span_deg") or 0)
        if yaw < 90:
            missing.append(f"turn is too small ({yaw:.0f} deg)")
            instruction = "turn further"
    if float(m.get("yaw_span_deg") or 0) > 60 and "turn" not in g and "spin" not in g:
        missing.append("unwanted spinning")
        instruction = "keep facing the same direction, do not spin"

    alignment = max(0, 100 - 25 * len(missing))
    if not missing:
        return {"alignment": 92, "satisfied": True, "matches": [cap] if cap else [],
                "missing": [], "next_edit": ""}
    if edits >= max_edits or instruction in failed:
        return {"alignment": alignment, "satisfied": False, "matches": [],
                "missing": missing, "next_edit": ""}
    return {"alignment": alignment, "satisfied": False, "matches": [],
            "missing": missing, "next_edit": instruction}


# --------------------------------------------------------------------------
# episode
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (s[:40] or "run")


def run_agent(
    goal: str,
    *,
    out_dir: Optional[Path] = None,
    max_steps: int = 10,
    max_edits: int = 3,
    generator: str = "mld",
    skip_viz: bool = False,
    use_llm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    episode = Path(out_dir) if out_dir else REPO / "outputs" / "agent_runs" / f"{stamp}_{_slug(goal)}"
    episode.mkdir(parents=True, exist_ok=True)

    tools = MotionTools(episode, generator=generator, skip_viz=skip_viz)
    trace: List[Dict[str, Any]] = []
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"USER GOAL: {goal}\n\nStart by calling generate_motion."},
    ]
    last_obs: Dict[str, Any] = {}
    last_assessment: Dict[str, Any] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    llm_errors = 0
    finish_rejections = 0
    verified_success = False
    accepted_finish = False
    last_check: Dict[str, Any] = {}
    applied_ok = 0
    attempted_unsupported = 0
    rule_mode = not (use_llm and llm_available())
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # ---- decide ----
        decision: Dict[str, Any]
        usage: Dict[str, Any] = {}
        if not rule_mode:
            state = {
                "step": step,
                "edits_used": tools.edits,
                "edits_left": max(0, max_edits - tools.edits),
                "has_clip": tools.run_dir is not None,
                "current_clip_described": tools.described_at is not None,
                "tried_edits": [
                    {"instruction": h.get("instruction"), "followed": (h.get("following") or {}).get("followed")}
                    for h in tools.history
                ],
                "last_observation": last_obs,
            }
            messages.append({"role": "user", "content": "STATE: " + json.dumps(state, ensure_ascii=False)})
            try:
                decision, usage, _raw = chat(messages)
                llm_errors = 0
            except Exception as exc:  # noqa: BLE001
                llm_errors += 1
                if verbose:
                    print(f"[agent] LLM error ({exc}); fallback" + (" to rules" if llm_errors >= 2 else ""), flush=True)
                messages.pop()
                if llm_errors >= 2:
                    rule_mode = True
                decision = {}
        if rule_mode or not decision:
            if tools.run_dir is None:
                decision = {"thought": "No clip yet: generate from the goal.",
                            "assessment": {"alignment": 0, "satisfied": False, "matches": [], "missing": [], "next_edit": ""},
                            "action": "generate_motion", "args": {"prompt": goal}}
            elif tools.described_at is None:
                decision = {"thought": "A clip exists but has not been described; call describe_motion.",
                            "assessment": last_assessment or {"alignment": 0, "satisfied": False, "matches": [], "missing": [], "next_edit": ""},
                            "action": "describe_motion", "args": {}}
            else:
                assess = _rule_assess(
                    goal, last_obs, tools.edits, max_edits,
                    failed={h.get("instruction") for h in tools.history if h.get("no_change")},
                )
                if assess["satisfied"] or not assess["next_edit"]:
                    decision = {"thought": "Rule assessment: " + "; ".join(assess["missing"]) or "goal met",
                                "assessment": assess, "action": "finish", "args": {"reason": "rule_policy"}}
                else:
                    decision = {"thought": "Rule gap: " + "; ".join(assess["missing"]),
                                "assessment": assess, "action": "edit_motion",
                                "args": {"instruction": assess["next_edit"]}}

        action = str(decision.get("action") or "").strip()
        args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
        thought = str(decision.get("thought") or "")
        assessment = decision.get("assessment") if isinstance(decision.get("assessment"), dict) else {}

        # ---- enforce the mandatory order ----
        if tools.run_dir is None and action != "generate_motion":
            action, args = "generate_motion", {"prompt": goal}
            thought += " [corrected: no clip yet]"
        elif tools.run_dir is not None and tools.described_at is None and action not in ("describe_motion", "generate_motion"):
            action, args = "describe_motion", {}
            thought += " [corrected: describe the new clip first]"
        elif action == "finish" and tools.checked_at is None:
            action, args = "check_requirement", {}
            thought += " [corrected: measure the requirement before finishing]"
        if action == "edit_motion" and tools.edits >= max_edits:
            action = "finish"
            args = {"reason": f"edit budget reached ({max_edits})"}
            thought += " [corrected: edit budget reached]"

        # ---- execute ----
        if action == "generate_motion":
            obs = tools.generate_motion(str(args.get("prompt") or goal))
        elif action == "describe_motion":
            obs = tools.describe_motion(goal)
        elif action == "check_requirement":
            obs = tools.check_requirement(goal)
            last_check = obs
        elif action == "edit_motion":
            instruction = str(args.get("instruction") or "").strip()
            if not instruction:
                obs = {"ok": False, "error": "edit_motion needs a non-empty instruction"}
            else:
                obs = tools.edit_motion(instruction)
                tools.history.append({"instruction": instruction, **obs})
                if obs.get("ok"):
                    applied_ok += 1
                elif obs.get("error") == "unsupported_instruction":
                    attempted_unsupported += 1
        elif action == "finish":
            claim = bool(assessment.get("satisfied"))
            checker = last_check or {}
            checker_ok = checker.get("satisfied")
            broken = not bool((last_obs or {}).get("valid", True))
            limitation = bool(args.get("limitation"))
            disagree = claim and checker_ok is False and not limitation
            if disagree and finish_rejections < 1:
                # Not a veto: the agent is told what the measurement says and may
                # still stop, but must say it is reporting a limitation.
                finish_rejections += 1
                obs = {
                    "ok": False,
                    "error": "checker_disagrees",
                    "checker_satisfied": False,
                    "agent_claim_satisfied": True,
                    "reasons": checker.get("reasons") or [],
                    "note": (
                        "You claimed the clip satisfies the request; the measurement "
                        "does not. Fix the first failing check with one edit, or call "
                        "finish again with args.{\"limitation\": true} to stop and "
                        "report this as a limitation."
                    ),
                }
                accepted_finish = False
            else:
                obs = tools.finish(str(args.get("reason") or ""))
                obs.update({
                    "agent_claim_satisfied": claim,
                    "checker_satisfied": checker_ok,
                    "checker_reasons": checker.get("reasons") or [],
                    "limitation": limitation,
                    "agreement": (checker_ok is not None and claim == bool(checker_ok)),
                })
                if broken:
                    obs["physical_veto"] = True
                verified_success = bool(claim and checker_ok) and not broken
                accepted_finish = True
        else:
            obs = {"ok": False, "error": f"unknown action {action!r}"}

        last_obs = obs
        if assessment:
            last_assessment = assessment
        for key in usage_totals:
            usage_totals[key] += int(usage.get(key) or 0) if isinstance(usage, dict) else 0

        record = {
            "step": step,
            "thought": thought,
            "assessment": assessment,
            "action": action,
            "args": args,
            "observation": obs,
            "usage": usage,
            "policy": "rule" if rule_mode else "llm",
            "seconds": obs.get("seconds"),
        }
        trace.append(record)
        messages.append({"role": "assistant", "content": json.dumps(
            {"thought": thought, "assessment": assessment, "action": action, "args": args},
            ensure_ascii=False)})
        messages.append({"role": "user", "content": "OBSERVATION: " + json.dumps(obs, ensure_ascii=False)})

        if verbose:
            print(f"[agent] step {step}: {action} {json.dumps(args, ensure_ascii=False)[:120]} "
                  f"→ ok={obs.get('ok')} {str(obs.get('error') or '')}", flush=True)

        if action == "finish" and accepted_finish:
            break
        if len(messages) > 24:
            # Keep the system prompt and the last observations only.
            messages = [messages[0]] + messages[-12:]

    final_run = tools.run_dir
    final_metrics = {}
    final_caption = ""
    final_preview = ""
    if final_run is not None:
        cap = tools._caption(final_run, goal) if (final_run / "03_edited.npy").is_file() else {}
        final_caption = str(cap.get("caption") or "")
        final_metrics = kinematic_summary(load_joints(final_run / "03_edited.npy"))
        final_preview = str(next(iter(final_run.glob("*preview*.gif")), ""))

    final_assessment = trace[-1].get("assessment") if trace else {}
    result = {
        "schema_version": "agent_run_v1",
        "goal": goal,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "rule" if rule_mode else "llm",
        "max_steps": max_steps,
        "max_edits": max_edits,
        "generator": generator,
        "episode_dir": str(episode),
        "final_run": str(final_run) if final_run else None,
        "final_preview": final_preview,
        "final_caption": final_caption,
        "final_metrics": final_metrics,
        "steps": trace,
        "summary": {
            "stopped": trace[-1].get("action") if trace else None,
            "satisfied": bool(verified_success),
            "verified": bool(verified_success),
            "agent_claim_satisfied": bool((final_assessment or {}).get("satisfied")),
            "checker_satisfied": bool(last_check.get("satisfied")) if last_check else None,
            "agreement": (
                bool((final_assessment or {}).get("satisfied")) == bool(last_check.get("satisfied"))
                if last_check else None
            ),
            "check_calls": sum(1 for t in trace if t["action"] == "check_requirement"),
            "finish_rejections": finish_rejections,
            "final_alignment": (final_assessment or {}).get("alignment"),
            "edits": tools.edits,
            "edits_applied": applied_ok,
            "edits_unsupported": attempted_unsupported,
            "tool_calls": len(tools.calls),
            "llm_rounds": sum(1 for t in trace if t["policy"] == "llm"),
            "seconds": round(time.time() - t0, 2),
            "tokens": usage_totals,
        },
    }
    (episode / "agent_run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (episode / "agent_run.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        f"# Agent run: {result['goal']}",
        "",
        f"- policy: **{result['policy']}** · generator: {result['generator']}",
        f"- steps: {len(result['steps'])} · edits: {result['summary']['edits']} "
        f"· seconds: {result['summary']['seconds']}",
        f"- final caption: _{result.get('final_caption') or 'n/a'}_",
        f"- final preview: `{result.get('final_preview') or 'n/a'}`",
        "",
        "| step | thought | action | observation | alignment |",
        "|---|---|---|---|---|",
    ]
    for st in result["steps"]:
        obs = st.get("observation") or {}
        if st["action"] == "describe_motion":
            summary = f"caption: {obs.get('caption')}"
        elif st["action"] == "edit_motion":
            summary = f"following={ (obs.get('instruction_following') or {}).get('followed') }, " \
                      f"non-target={ (obs.get('preservation') or {}).get('non_target_mpjpe') }"
        elif st["action"] == "generate_motion":
            summary = f"generator={obs.get('generator_used')}, frames={obs.get('frames')}"
        elif st["action"] == "check_requirement":
            summary = f"satisfied={obs.get('satisfied')} " + "; ".join(obs.get("reasons") or [])[:120]
        else:
            summary = str(obs.get("reason") or "")
        lines.append(
            f"| {st['step']} | {st['thought'][:160]} | `{st['action']}` | {str(summary)[:160]} | "
            f"{(st.get('assessment') or {}).get('alignment', '')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic text→motion loop (LLM tool calling)")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--generator", choices=["mld", "dataset"], default="mld")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="Run the scripted baseline")
    args = ap.parse_args()
    result = run_agent(
        args.goal,
        out_dir=args.out,
        max_steps=args.max_steps,
        max_edits=args.max_edits,
        generator=args.generator,
        skip_viz=args.skip_viz,
        use_llm=not args.no_llm,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
