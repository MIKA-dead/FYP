"""
Pipeline v1 loop: edit → describe (M2T) → compare with the previous caption →
LLM proposes the next edit → repeat until the goal is met or the budget runs out.

  # start from an existing run
  python scripts/orchestrator/run_loop.py \\
      --run-dir outputs/session/ui_runs/xxx --goal "a person jumps" --max-iters 3

  # or generate the first clip in the loop
  python scripts/orchestrator/run_loop.py \\
      --prompt "a person jumps" --goal "a person jumps once and lands still" --max-iters 3

Every step is written next to the starting run: ``07_loop/iterNN/`` holds the
session artifacts, ``07_loop.json`` the captions and the prompt that produced
them. The loop stops on: goal met, no caption change twice in a row, a failed
session, or the iteration budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (
    REPO_ROOT / "scripts" / "session",
    REPO_ROOT / "scripts" / "orchestrator",
    REPO_ROOT / "scripts" / "intent",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from session_api import run_session  # noqa: E402
from edit_proposer import propose_next_edit  # noqa: E402


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None


def _ensure_caption(run_dir: Path) -> Dict[str, Any]:
    """Pipeline v1 always describes the clip with the M2T model.

    Older runs only have the rule sentence, so re-describe the clip once and
    overwrite ``06_motion_caption.json``. The rules stay as ``problems``.
    """
    cap = _load_json(run_dir / "06_motion_caption.json") or {}
    if str(cap.get("source") or "").startswith("tm2t"):
        return cap
    edited = run_dir / "03_edited.npy"
    if not edited.is_file():
        return cap
    try:
        from motion_caption import caption_edited

        summary = _load_json(run_dir / "summary.json") or {}
        plan = _load_json(run_dir / "00_motion_edit_plan.json") or {}
        fresh = caption_edited(
            edited,
            quality=_load_json(run_dir / "04_quality.json") or {},
            action=str(plan.get("primary_action") or ""),
            intent_text=str(summary.get("prompt") or ""),
            run_dir=run_dir,
        )
        (run_dir / "06_motion_caption.json").write_text(
            json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return fresh
    except Exception as exc:  # noqa: BLE001
        print(f"[run_loop] caption refresh failed: {exc}", flush=True)
        return cap


def _caption_of(run_dir: Path) -> Dict[str, Any]:
    return _load_json(run_dir / "06_motion_caption.json") or {}


def _quality_of(run_dir: Path) -> Dict[str, Any]:
    return _load_json(run_dir / "04_quality.json") or {}


def _caption_text(cap: Dict[str, Any]) -> str:
    return str(cap.get("caption") or "").strip()


def iterate(
    *,
    goal: str,
    start_run: Optional[Path] = None,
    prompt: Optional[str] = None,
    max_iters: int = 3,
    skip_viz: bool = False,
    stop_on_no_change: int = 2,
    require_llm: bool = False,
) -> Dict[str, Any]:
    log: Dict[str, Any] = {
        "schema_version": "pipeline_v1_loop_v1",
        "goal": goal,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "max_iters": max_iters,
        "iterations": [],
    }

    if start_run is None:
        if not prompt:
            raise SystemExit("give --run-dir or --prompt")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        gen_dir = REPO_ROOT / "outputs" / "session" / "loops" / f"{stamp}_generate"
        resp = run_session(
            {
                "prompt": prompt,
                "mode": "generate",
                "run_dir": str(gen_dir),
                "skip_review": True,
                "skip_viz": skip_viz,
                "backend": "orchestrator",
            },
            cwd=REPO_ROOT,
        )
        log["generate"] = {
            "prompt": prompt,
            "ok": resp.get("ok"),
            "run_dir": resp.get("run_dir"),
            "error": resp.get("error"),
        }
        if not resp.get("run_dir") or not (Path(resp["run_dir"]) / "03_edited.npy").is_file():
            log["stopped"] = "generate produced no clip"
            return log
        start_run = Path(resp["run_dir"])

    start_run = start_run.resolve()
    loop_dir = start_run / "07_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    log["start_run"] = str(start_run)

    cur_run = start_run
    before_caption = _ensure_caption(cur_run)
    no_change_streak = 0
    history: List[Dict[str, Any]] = []

    for iteration in range(1, max_iters + 1):
        quality = _quality_of(cur_run)
        try:
            proposal = propose_next_edit(
                goal=goal,
                prev_caption=before_caption,
                cur_caption=_caption_of(cur_run),
                quality=quality,
                history=history,
                iteration=iteration,
                max_iters=max_iters,
                require_llm=require_llm,
            )
        except RuntimeError as exc:
            log["stopped"] = f"llm_required: {exc}"
            log["final_run"] = str(cur_run)
            log["final_caption"] = _caption_text(_caption_of(cur_run))
            # Do not clobber a previous loop record with an aborted run.
            (loop_dir / "07_loop.llm_error.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "run_dir": str(cur_run),
            "caption_before": _caption_text(before_caption),
            "caption": _caption_text(_caption_of(cur_run)),
            "caption_source": _caption_of(cur_run).get("source"),
            "quality_score": quality.get("score"),
            "quality_passed": quality.get("passed"),
            "proposal": proposal,
        }
        if proposal.get("done") or not proposal.get("next_edit_prompt"):
            entry["result"] = "proposer_stop"
            log["iterations"].append(entry)
            log["stopped"] = (
                f"proposer stopped ({proposal.get('verdict')}): {proposal.get('reason') or 'no further edit'}"
            )
            break

        iter_dir = loop_dir / f"iter{iteration:02d}"
        if iter_dir.exists():
            iter_dir = loop_dir / f"iter{iteration:02d}_{datetime.now().strftime('%H%M%S')}"
        resp = run_session(
            {
                "prompt": proposal["next_edit_prompt"],
                "mode": "revise",
                "previous_run": str(cur_run),
                "run_dir": str(iter_dir),
                "skip_review": True,
                "skip_viz": skip_viz,
                "backend": "orchestrator",
            },
            cwd=REPO_ROOT,
        )
        new_run = Path(resp["run_dir"]) if resp.get("run_dir") else None
        edited = (new_run / "03_edited.npy") if new_run else None
        entry["session"] = {
            "ok": resp.get("ok"),
            "run_dir": resp.get("run_dir"),
            "error": resp.get("error"),
            "quality_score": (resp.get("quality") or {}).get("score"),
        }
        if not resp.get("ok") and resp.get("error") != "quality_gate_failed":
            # The edit never produced a fresh clip (coder/planner error); a file
            # left over from an older attempt must not be adopted.
            entry["result"] = "session_failed"
            log["iterations"].append(entry)
            log["stopped"] = f"edit failed: {resp.get('error')}"
            break
        if not edited or not edited.is_file():
            # Iteration dirs are unique, so anything here belongs to this attempt.
            entry["result"] = "session_failed"
            log["iterations"].append(entry)
            log["stopped"] = "edit produced no clip"
            break

        prev_caption_obj = _caption_of(cur_run)
        prev_score = entry.get("quality_score")
        new_run_score = _quality_of(new_run).get("score")
        new_caption = _caption_of(new_run)
        entry["caption_after"] = _caption_text(new_caption)
        entry["caption_after_source"] = new_caption.get("source")
        entry["quality_score_after"] = new_run_score

        same = _caption_text(new_caption) == entry["caption"]
        regressed = (
            prev_score is not None
            and new_run_score is not None
            and float(new_run_score) < float(prev_score) - 5.0
        )
        entry["caption_changed"] = not same
        if regressed:
            # Keep the clip that scored better; do not carry a worse edit forward.
            entry["result"] = "regressed"
            log["iterations"].append(entry)
            log["stopped"] = (
                f"edit lowered the quality score {prev_score} → {new_run_score}; kept the previous clip"
            )
            break

        before_caption = prev_caption_obj
        cur_run = new_run
        no_change_streak = no_change_streak + 1 if same else 0
        entry["result"] = "unchanged" if same else "edited"
        history.append(
            {
                "next_edit_prompt": proposal.get("next_edit_prompt"),
                "caption_after": entry.get("caption_after"),
                "verdict": proposal.get("verdict"),
            }
        )
        log["iterations"].append(entry)
        (loop_dir / "07_loop.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if no_change_streak >= stop_on_no_change:
            log["stopped"] = "caption did not change twice"
            break

    log["final_run"] = str(cur_run)
    log["final_caption"] = _caption_text(_caption_of(cur_run))
    log["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (loop_dir / "07_loop.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline v1 iteration loop (M2T + LLM next edit)")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--prompt", type=str, default=None, help="Generate this prompt first")
    ap.add_argument("--goal", type=str, required=True)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail instead of falling back to the rule proposer (use for LLM verification runs)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    try:
        log = iterate(
            goal=args.goal,
            start_run=args.run_dir,
            prompt=args.prompt,
            max_iters=args.max_iters,
            skip_viz=args.skip_viz,
            require_llm=args.require_llm,
        )
    except RuntimeError as exc:
        print(f"[run_loop] stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "goal": log.get("goal"),
            "final_run": log.get("final_run"),
            "final_caption": log.get("final_caption"),
            "stopped": log.get("stopped"),
            "iterations": len(log.get("iterations") or []),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
