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
from verify import verify_goal  # noqa: E402
from tools import MAX_RESTORES, SUPPORTED_EDITS, TOOL_SPECS, MotionTools  # noqa: E402

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
    "4. Then either edit_motion (one change), resample_motion, or finish.\n\n"
    "HOW TO JUDGE\n"
    "- The learned caption is coarse; joint measurements are ground truth when "
    "they disagree. arm_raise_ratio >= 1.8 and wrist_above_head (metres) > 0 both "
    "mean the wrist is above the head. root_travel is metres travelled; "
    "pelvis_drop is how far the hips descend; yaw_span_deg is how much the body "
    "turns and yaw_net_deg is the signed net rotation (positive = left).\n"
    "- describe_motion returns TWO captions and a caption_verdict: `caption` from "
    "the learned M2T model, `rule_caption` built from the same measurements the "
    "checker reads, and their similarity to the goal. The verdict is three-way: "
    "`match` (both agree), `uncertain` (only the learned caption looks wrong — it is "
    "usually too coarse; verify with check_requirement and do not edit on that "
    "basis) and `mismatch` (both captions disagree with the goal — that is a reason "
    "to look at the measurements, still not a verdict on its own).\n"
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
    "- The editing tools are RELATIVE: they scale or reshape motion that is already "
    "in the clip. They can never add travel, add a turn, or remove locomotion. When "
    "the measurement says an attribute is missing entirely (root_travel 0.0, "
    "yaw_span 10 deg, no knee lift) and no edit can create it, first call "
    "resample_motion to draw several drafts of the same prompt and keep the closest; "
    "if every draw still lacks it, that is a generator limit — call generate_motion "
    "again with generator=\"dataset\", which retrieves a real HumanML3D clip whose "
    "caption matches the prompt and can contain the missing attribute. Do not spend "
    "edits on something an edit cannot do.\n"
    "- Generation already rejection-samples: when the prompt names a measurable "
    "attribute, several drafts are drawn and the closest is kept, reported as "
    "draft_rejection (drawn / accepted / rejected). If that report says the goal was "
    "still not met after N drafts, the diffusion model is the limit — a resample of "
    "the same prompt will not help; try generator=\"dataset\", or say so and finish.\n"
    "- Every clip you have produced stays available. If an edit moved the "
    "measurement away from the goal (worse_than_before=true, or gap_after > "
    "gap_before), call restore_motion to go back to the closest one instead of "
    "stacking edits on a clip you already know is worse. You may not finish on a "
    "clip that is further from the goal than one you already had.\n"
    "- An edit that returns unsupported_instruction never reached the backend. Do "
    "not rephrase the same impossible request; finish and name it as out of scope.\n"
    "- Three edit errors mean 'this clip cannot take this operator': "
    "edit_needs_existing_locomotion, edit_needs_existing_turn, "
    "edit_needs_existing_motion, edit_needs_seated_clip. The previous clip is kept. "
    "Use generate_motion with the full goal instead of retrying the edit.\n"
    "- edit_had_no_effect / edit_did_not_follow_instruction mean the backend ran but "
    "the attribute you named did not move; the previous clip is restored. Do not "
    "repeat the instruction — regenerate with a prompt that asks for it directly.\n"
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
    '"action": "generate_motion|resample_motion|restore_motion|describe_motion|'
    'check_requirement|edit_motion|finish", '
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
                 failed: Optional[set] = None,
                 metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Scripted baseline policy, answered by the same measurements as the tools.

    ``metrics`` is the current clip's kinematic summary. It is passed in
    explicitly because observations differ by tool (``check_requirement``
    reports ``measured``, not ``metrics``) and reading the wrong key silently
    scored every clip as "no turn at all".
    """
    failed = failed or set()
    m = metrics or obs.get("metrics") or obs.get("measured") or {}
    cap = str(obs.get("caption") or "")
    g = (goal or "").lower()
    missing: List[str] = []
    instruction = ""
    side = "left" if "left" in g else "right"
    verdict = verify_goal(goal, {"metrics": m})
    reasons = {c["check"]: c for c in verdict["checks"] if not c["ok"]}

    def need(name: str, text: str, edit: str) -> None:
        if name in reasons:
            c = reasons[name]
            missing.append(f"{text} (measured {c['value']}, needs {c['expected']})")
            return edit
        return ""

    instruction = need(f"arm_raise_{side}", f"{side} arm not high enough",
                       f"raise the {side} arm higher") or instruction
    instruction = need("root_travel", "forward travel is too small",
                       "walk forward further") or instruction
    instruction = need("pelvis_drop", "hips do not lower enough",
                       "sit down lower") or instruction
    instruction = need("yaw_span_deg", "turn is too small", "turn further") or instruction
    instruction = need("ankle_lift_right", "right leg does not kick",
                       "kick with the right leg") or instruction
    instruction = need("ankle_lift_left", "left leg does not kick",
                       "kick with the left leg") or instruction
    instruction = need("jump_rise", "no jump", "jump higher") or instruction
    if not missing and float(m.get("yaw_span_deg") or 0) > 60 and "turn" not in g and "spin" not in g:
        missing.append("unwanted spinning")
        instruction = "keep facing the same direction, do not spin"

    missing = [r for r in verdict["reasons"] if r not in missing] + missing
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

    tools = MotionTools(episode, generator=generator, skip_viz=skip_viz, goal=goal)
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
                "restores_left": max(0, MAX_RESTORES - tools.restores),
                "has_clip": tools.run_dir is not None,
                "current_clip_described": tools.described_at is not None,
                "closest_clip_so_far": tools.archive_summary(limit=3),
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
                    metrics=(tools.current_snapshot() or {}).get("measured")
                    or (last_obs or {}).get("metrics"),
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
        elif tools.run_dir is not None and tools.described_at is None and action not in ("describe_motion", "generate_motion", "resample_motion"):
            action, args = "describe_motion", {}
            thought += " [corrected: describe the new clip first]"
        elif action == "finish" and tools.checked_at is None:
            action, args = "check_requirement", {}
            thought += " [corrected: measure the requirement before finishing]"
        if action == "edit_motion" and tools.edits >= max_edits:
            action = "finish"
            args = {"reason": f"edit budget reached ({max_edits})"}
            thought += " [corrected: edit budget reached]"
        # Never stop on a clip that is further from the goal than one already
        # measured in this episode: the archive still holds it, so going back is
        # one cheap step (this is what ep12 needed after an edit erased an 88° turn).
        if action == "finish" and tools.restores < MAX_RESTORES:
            cur = tools.current_snapshot()
            best = tools.best_snapshot()
            if cur and best and best.get("run_dir") != cur.get("run_dir"):
                better = (
                    (best.get("satisfied") and not cur.get("satisfied"))
                    or float(best.get("gap") or 0) + 1e-6 < float(cur.get("gap") or 0)
                )
                if better:
                    action = "restore_motion"
                    args = {"reason": "an earlier clip is closer to the goal"}
                    thought += (" [corrected: an earlier clip is closer to the goal "
                                "(gap %.2f < %.2f); restore it before finishing]"
                                % (float(best.get("gap") or 0), float(cur.get("gap") or 0)))

        # ---- execute ----
        if action == "generate_motion":
            obs = tools.generate_motion(
                str(args.get("prompt") or goal),
                generator=args.get("generator"),
            )
        elif action == "resample_motion":
            obs = tools.resample_motion(
                str(args.get("prompt") or ""), int(args.get("n") or 3)
            )
        elif action == "describe_motion":
            obs = tools.describe_motion(goal)
        elif action == "check_requirement":
            obs = tools.check_requirement(goal)
            last_check = obs
        elif action == "restore_motion":
            obs = tools.restore_motion(str(args.get("reason") or ""))
            if obs.get("ok"):
                last_check = {}
        elif action == "edit_motion":
            instruction = str(args.get("instruction") or "").strip()
            if not instruction:
                obs = {"ok": False, "error": "edit_motion needs a non-empty instruction"}
            else:
                obs = tools.edit_motion(instruction)
                tools.history.append({
                    **obs,
                    "instruction": instruction,
                    "following": obs.get("instruction_following") or {},
                    "no_change": bool(obs.get("no_change")),
                })
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

    # If the budget ran out on a clip that is worse than one already measured,
    # deliver the closest clip instead of the last one. This is the same rule the
    # finish guard enforces; it only fires when the loop ended without finishing.
    if tools.run_dir is not None and not verified_success:
        cur = tools.current_snapshot()
        best = tools.best_snapshot()
        if (
            cur and best and best.get("run_dir") != cur.get("run_dir")
            and (
                (best.get("satisfied") and not cur.get("satisfied"))
                or float(best.get("gap") or 0) + 1e-6 < float(cur.get("gap") or 0)
            )
        ):
            obs = tools.restore_motion("loop budget exhausted; deliver the closest clip")
            if obs.get("ok"):
                trace.append({
                    "step": len(trace) + 1,
                    "thought": "Budget exhausted. The closest clip measured in this "
                               "episode is restored as the result.",
                    "assessment": final_assessment if trace else {},
                    "action": "restore_motion",
                    "args": {"reason": "loop budget exhausted"},
                    "observation": obs,
                    "usage": {},
                    "policy": "auto",
                    "auto": True,
                    "seconds": obs.get("seconds"),
                })

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
            "resamples": tools.resamples,
            "restores": tools.restores,
            "best_gap_seen": (tools.best_snapshot() or {}).get("gap"),
            "final_gap": (tools.current_snapshot() or {}).get("gap"),
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
        elif st["action"] == "resample_motion":
            summary = (
                f"chosen cand{obs.get('chosen')} gap={obs.get('chosen_gap')} "
                f"satisfied={obs.get('chosen_satisfied')}"
            )
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
