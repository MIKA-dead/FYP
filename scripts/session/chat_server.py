"""
Conversational motion studio UI (Cursor-like).

  python scripts/session/chat_server.py
  → http://127.0.0.1:8787

Left: chat + reasoning (Intent / skills / modules)
Right: motion preview + quality
Composer: model picker + prompt
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = Path(__file__).resolve().parent
UI_DIR = SESSION_DIR / "ui"
INTENT_DIR = REPO_ROOT / "scripts" / "intent"
OUT_SESSION = REPO_ROOT / "outputs" / "session"
PORT = int(os.environ.get("SESSION_UI_PORT", "8787"))

sys.path.insert(0, str(INTENT_DIR))
sys.path.insert(0, str(SESSION_DIR))

from plan_motion_edit import build_full_agent_bundle  # noqa: E402
from session_api import run_session  # noqa: E402

HANDOFF_PATH = OUT_SESSION / "handoff_for_ui.json"
DEMO_CASES = REPO_ROOT / "test-samples" / "session" / "demo_cases.json"

# In-memory chat threads: thread_id → {messages, previous_run, model}
_THREADS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

MODELS = [
    {
        "id": "agent-heuristic",
        "label": "Agent Heuristic",
        "desc": "编排器；revise→闭集 Refine（有 API key 则 LLM，否则启发式）；时间轴点选直发算子",
        "backend": "orchestrator",
    },
    {
        "id": "motionfix-native",
        "label": "MotionsFix Native",
        "desc": "首轮优先 MotionsFix test-set id；revise 仍复用上一 clip",
        "backend": "orchestrator",
    },
    {
        "id": "opencode-deepseek",
        "label": "OpenCode 角色",
        "desc": ".opencode/agents 约束 + 编排器；Refine 可用 DEEPSEEK_API_KEY（同 OpenAI 兼容）",
        "backend": "orchestrator",
        "badge": "refine-LLM",
    },
]

SKILL_RAG = [
    {
        "id": "intent-routing",
        "path": ".agents/skills/intent-routing/SKILL.md",
        "why": "Compile prompt → Intent Schema + taxonomy",
    },
    {
        "id": "motion-patterns",
        "path": ".agents/skills/motion-patterns/SKILL.md",
        "why": "Action speed baselines / success patterns",
    },
    {
        "id": "style-library",
        "path": ".agents/skills/style-library/SKILL.md",
        "why": "Style tags if present in prompt",
    },
]


def _load_demos() -> List[Dict[str, Any]]:
    if HANDOFF_PATH.is_file():
        data = json.loads(HANDOFF_PATH.read_text(encoding="utf-8-sig"))
        return list(data.get("cases") or [])
    if DEMO_CASES.is_file():
        data = json.loads(DEMO_CASES.read_text(encoding="utf-8-sig"))
        return list(data.get("cases") or [])
    return []


def _norm_prompt(p: str) -> str:
    """Normalize for demo matching (sit/sits, punctuation, whitespace)."""
    import re

    s = (p or "").strip().lower()
    s = re.sub(r"[.!?]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    # verb agreement noise: sit/sits/sitting → sit
    s = re.sub(r"\bsits\b", "sit", s)
    s = re.sub(r"\bsitting\b", "sit", s)
    s = re.sub(r"\bwalks\b", "walk", s)
    s = re.sub(r"\bwalking\b", "walk", s)
    s = re.sub(r"\bturns\b", "turn", s)
    s = re.sub(r"\bwaves\b", "wave", s)
    return s


def _match_demo(prompt: str) -> Optional[Dict[str, Any]]:
    p = _norm_prompt(prompt)
    for c in _load_demos():
        if _norm_prompt(c.get("prompt") or "") == p:
            return c
    return None


def _safe_under_repo(path: Path) -> Path:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if not str(resolved).startswith(str(root)):
        raise PermissionError("path escapes repo")
    return resolved


def _rel_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        rel = _safe_under_repo(p).relative_to(REPO_ROOT.resolve())
    except Exception:
        return None
    return "/api/file?path=" + str(rel).replace("\\", "/")


def _build_reasoning(
    bundle: Dict[str, Any],
    model_id: str,
    *,
    agent_trace: Optional[List[Dict[str, Any]]] = None,
    module_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    intent = bundle.get("intent") or {}
    plan = bundle.get("motion_edit_plan") or {}
    modules = bundle.get("module_plan") or {}
    steps = [
        {
            "phase": "skill_warmup",
            "title": "Knowledge warmup (RAG-style)",
            "items": SKILL_RAG,
        },
        {
            "phase": "intent",
            "title": "Intent Schema (motion-planner)",
            "items": [
                {"k": "task_mode", "v": intent.get("task_mode")},
                {"k": "confidence", "v": intent.get("confidence")},
                {"k": "why", "v": (intent.get("decision") or {}).get("why")},
                {
                    "k": "timeline",
                    "v": [
                        {"seg": t.get("seg"), "action": t.get("action"), "class": t.get("humanml3d_class")}
                        for t in (intent.get("timeline") or [])
                    ],
                },
            ],
        },
        {
            "phase": "modules",
            "title": "Module routing",
            "items": [
                {
                    "step": s.get("step"),
                    "module": s.get("module"),
                    "owner": s.get("owner_agent"),
                    "why": s.get("why"),
                    "mock": s.get("mock"),
                }
                for s in (modules.get("steps") or [])
            ],
            "frozen": modules.get("frozen_modules") or [],
        },
        {
            "phase": "motion_plan",
            "title": "MotionEditPlan → coder",
            "items": [
                {"k": "primary_action", "v": plan.get("primary_action")},
                {"k": "mf_text", "v": plan.get("mf_text")},
                {"k": "source_policy", "v": plan.get("source_policy")},
                {"k": "edit_source", "v": plan.get("edit_source")},
                {"k": "preferred_source_id", "v": plan.get("preferred_source_id")},
                {"k": "edit_mode", "v": plan.get("edit_mode")},
                {"k": "span", "v": plan.get("span")},
                {"k": "warning", "v": plan.get("warning")},
                {"k": "model", "v": model_id},
            ],
        },
    ]
    if agent_trace:
        steps.append(
            {
                "phase": "agent_trace",
                "title": "Orchestrator agent trace",
                "items": [
                    {
                        "agent": t.get("agent"),
                        "phase": t.get("phase"),
                        "title": t.get("title"),
                        "module": t.get("module"),
                        "ok": t.get("ok"),
                        "seconds": t.get("seconds"),
                        "args": t.get("args"),
                        "artifacts": t.get("artifacts"),
                        "detail": t.get("detail")
                        if not isinstance(t.get("detail"), dict)
                        else {
                            k: t["detail"][k]
                            for k in (
                                "primary_action",
                                "mf_text",
                                "source_policy",
                                "passed",
                                "score",
                                "hard_failures",
                            )
                            if k in (t.get("detail") or {})
                        },
                    }
                    for t in agent_trace
                ],
            }
        )
    if module_results:
        steps.append(
            {
                "phase": "run_module",
                "title": "run_module results (motion-coder / tester)",
                "items": [
                    {
                        "module": r.get("module"),
                        "ok": r.get("ok"),
                        "seconds": r.get("seconds"),
                        "artifacts": r.get("artifacts"),
                        "error": r.get("error"),
                    }
                    for r in module_results
                ],
            }
        )
    return {
        "model": model_id,
        "steps": steps,
        "prompt_compiled": plan.get("mf_text") or intent.get("raw_prompt"),
        "orchestrator": True,
    }


def _expected_timeline_kind(action: str) -> str:
    act = (action or "").strip()
    if act.startswith("sit") or act in ("sit_down", "sit"):
        return "sit"
    if "wave" in act:
        return "wave"
    if "turn" in act:
        return "turn"
    if act.startswith("walk") or "walk" in act:
        return "walk"
    return "generic"


def _timeline_matches_action(tl: Optional[Dict[str, Any]], action: str) -> bool:
    if not tl or not tl.get("phases"):
        return False
    kind = str(tl.get("kind") or "")
    expect = _expected_timeline_kind(action)
    if not kind:
        # Legacy sit-only timelines lack kind; accept only for sit actions.
        return expect == "sit" and ("sit_start" in tl or "descend" in {
            str(p.get("id")) for p in (tl.get("phases") or [])
        })
    return kind == expect or (expect == "generic")


def _ensure_timeline(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attach action-aware timeline for UI (recompute when kind mismatches)."""
    action = (resp.get("intent") or {}).get("primary_action") or ""
    if not action:
        action = (resp.get("quality") or {}).get("action") or ""

    existing = resp.get("timeline") or (resp.get("quality") or {}).get("timeline")
    if existing and _timeline_matches_action(existing, action):
        resp["timeline"] = existing
        q = dict(resp.get("quality") or {})
        q.setdefault("timeline", existing)
        resp["quality"] = q
        return existing

    run_dir = resp.get("run_dir")
    edited = None
    arts = resp.get("artifacts") or {}
    if arts.get("edited_npy"):
        edited = Path(str(arts["edited_npy"]))
    if run_dir:
        rd = Path(str(run_dir))
        tl_path = rd / "05_timeline.json"
        if tl_path.is_file():
            try:
                tl = json.loads(tl_path.read_text(encoding="utf-8-sig"))
                if _timeline_matches_action(tl, action):
                    resp["timeline"] = tl
                    q = dict(resp.get("quality") or {})
                    q["timeline"] = tl
                    resp["quality"] = q
                    return tl
            except Exception:  # noqa: BLE001
                pass
        if edited is None or not edited.is_file():
            cand = rd / "03_edited.npy"
            if cand.is_file():
                edited = cand
        qpath = rd / "04_quality.json"
        if qpath.is_file():
            try:
                qfile = json.loads(qpath.read_text(encoding="utf-8-sig"))
                tl_q = qfile.get("timeline")
                if _timeline_matches_action(tl_q, action):
                    resp["timeline"] = tl_q
                    q = dict(resp.get("quality") or {})
                    for k in ("passed", "score", "metrics", "timeline", "checks"):
                        if qfile.get(k) is not None:
                            q[k] = qfile[k]
                    resp["quality"] = q
                    return tl_q
            except Exception:  # noqa: BLE001
                pass

    if edited is not None and Path(edited).is_file():
        try:
            from quality_gate import load_joints
            from motion_timeline import analyze_motion_timeline

            joints, _ = load_joints(Path(edited))
            tl = analyze_motion_timeline(joints, action=action or "", forbid=[])
            if tl and tl.get("phases"):
                resp["timeline"] = tl
                q = dict(resp.get("quality") or {})
                q["timeline"] = tl
                resp["quality"] = q
                # Persist so next load matches action
                if run_dir:
                    try:
                        Path(str(run_dir)).joinpath("05_timeline.json").write_text(
                            json.dumps(tl, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                return tl
        except Exception as exc:  # noqa: BLE001
            print(f"[chat_ui] timeline compute skipped: {exc}", flush=True)
    return None


def _response_from_golden(demo: Dict[str, Any], prompt: str, mode: str) -> Dict[str, Any]:
    run_dir = Path(demo["run_dir"])
    resp_path = run_dir / "session_response.json"
    if resp_path.is_file():
        resp = json.loads(resp_path.read_text(encoding="utf-8-sig"))
    else:
        resp = {
            "schema_version": "session_response_v1",
            "ok": bool(demo.get("ok")),
            "mode": mode,
            "prompt": prompt,
            "run_dir": str(run_dir),
            "artifacts": {
                "edited_npy": demo.get("edited_npy"),
                "preview_gif": demo.get("preview_gif"),
                "summary": str(run_dir / "summary.json"),
                "quality": str(run_dir / "04_quality.json"),
            },
            "intent": {"primary_action": demo.get("primary_action")},
            "quality": demo.get("quality") or {},
        }
    resp["from_demo_cache"] = True
    resp["demo_id"] = demo.get("id")
    # Ensure intent.primary_action present for timeline dispatch (wave vs sit).
    intent = dict(resp.get("intent") or {})
    if not intent.get("primary_action") and demo.get("primary_action"):
        intent["primary_action"] = demo.get("primary_action")
        resp["intent"] = intent
    _ensure_timeline(resp)
    return resp


def handle_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (body.get("prompt") or "").strip()
    refine_override = body.get("refine") or body.get("refine_override")
    if refine_override is not None and not isinstance(refine_override, dict):
        refine_override = None
    # Timeline UI can send refine without free-text prompt.
    if not prompt and refine_override:
        prompt = str(refine_override.get("raw_prompt") or "[timeline] refine")
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    mode = body.get("mode") or "generate"
    if refine_override is not None:
        mode = "revise"
    model_id = body.get("model") or "agent-heuristic"
    thread_id = body.get("thread_id") or str(uuid.uuid4())
    use_demo_cache = bool(body.get("use_demo_cache", True))
    force_full = bool(body.get("force_full_run", False))
    if refine_override is not None:
        use_demo_cache = False
        force_full = True
    previous_run = body.get("previous_run")

    with _LOCK:
        thread = _THREADS.setdefault(
            thread_id,
            {"messages": [], "previous_run": None, "model": model_id},
        )
        if mode == "revise" and not previous_run:
            previous_run = thread.get("previous_run")

    task_id = f"UI_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    prev_action = None
    prev_plan = None
    prev_caption = None
    timeline_ctx = body.get("timeline") if isinstance(body.get("timeline"), dict) else None
    if mode == "revise" and previous_run:
        pp = Path(previous_run) / "00_motion_edit_plan.json"
        if pp.is_file():
            try:
                prev_plan = json.loads(pp.read_text(encoding="utf-8-sig"))
                prev_action = prev_plan.get("base_action") or prev_plan.get(
                    "primary_action"
                )
                if prev_action == "speed_revise":
                    prev_action = None
            except Exception:  # noqa: BLE001
                pass
        if timeline_ctx is None:
            tl_path = Path(previous_run) / "05_timeline.json"
            if tl_path.is_file():
                try:
                    timeline_ctx = json.loads(tl_path.read_text(encoding="utf-8-sig"))
                except Exception:  # noqa: BLE001
                    pass
        cap_path = Path(previous_run) / "06_motion_caption.json"
        if cap_path.is_file():
            try:
                prev_caption = json.loads(cap_path.read_text(encoding="utf-8-sig"))
            except Exception:  # noqa: BLE001
                prev_caption = None
    bundle = build_full_agent_bundle(
        prompt,
        mode=mode,
        task_id=task_id,
        source_hint="bridged_walk",
        previous_action=prev_action,
        previous_plan=prev_plan,
        refine_override=refine_override,
        timeline=timeline_ctx,
        previous_caption=prev_caption,
    )

    session_resp: Optional[Dict[str, Any]] = None
    demo = _match_demo(prompt) if (use_demo_cache and mode == "generate" and not force_full) else None
    if demo and demo.get("run_dir") and Path(demo["run_dir"]).is_dir():
        session_resp = _response_from_golden(demo, prompt, mode)
    else:
        run_dir = OUT_SESSION / "ui_runs" / f"{thread_id[:8]}_{task_id}"
        req = {
            "prompt": prompt,
            "mode": mode,
            "previous_run": previous_run,
            "run_dir": str(run_dir),
            "skip_review": True,
            "skip_viz": False,
            "length": int(body.get("length") or 60),
            "backend": "orchestrator",
            "task_id": task_id,
        }
        if refine_override is not None:
            req["refine"] = refine_override
        if timeline_ctx is not None:
            req["timeline"] = timeline_ctx
        if model_id == "motionfix-native":
            pref = (bundle.get("motion_edit_plan") or {}).get("preferred_source_id")
            if pref:
                req["edit_source_id"] = pref
        try:
            session_resp = run_session(req, cwd=REPO_ROOT)
        except Exception as exc:  # noqa: BLE001
            session_resp = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc()[-2000:],
            }

    assert session_resp is not None
    # Prefer bundle written by orchestrator when present
    arts = session_resp.get("artifacts") or {}
    bundle_path = arts.get("intent_bundle")
    if bundle_path and Path(bundle_path).is_file():
        try:
            disk_bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8-sig"))
            if disk_bundle.get("motion_edit_plan"):
                bundle = disk_bundle
        except Exception:  # noqa: BLE001
            pass

    _ensure_timeline(session_resp)

    reasoning = _build_reasoning(
        bundle,
        model_id,
        agent_trace=session_resp.get("agent_trace"),
        module_results=session_resp.get("module_results"),
    )

    preview_url = _rel_url(arts.get("preview_gif"))
    if not preview_url and demo:
        preview_url = _rel_url(demo.get("preview_gif"))

    assistant_text = _compose_assistant_text(
        session_resp,
        reasoning,
        plan=bundle.get("motion_edit_plan") or {},
    )

    msg_user = {
        "role": "user",
        "content": prompt,
        "mode": mode,
        "model": model_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    msg_asst = {
        "role": "assistant",
        "content": assistant_text,
        "reasoning": reasoning,
        "session": {
            "ok": session_resp.get("ok"),
            "run_dir": session_resp.get("run_dir"),
            "from_demo_cache": session_resp.get("from_demo_cache", False),
            "intent": session_resp.get("intent"),
            "quality": session_resp.get("quality"),
            "timeline": session_resp.get("timeline")
            or (session_resp.get("quality") or {}).get("timeline"),
            "preview_url": preview_url,
            "artifacts": {
                k: _rel_url(v)
                if isinstance(v, str) and v.endswith((".gif", ".npy", ".json"))
                else v
                for k, v in arts.items()
            },
            "error": session_resp.get("error"),
        },
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if not msg_asst["session"].get("intent"):
        plan = bundle.get("motion_edit_plan") or {}
        msg_asst["session"]["intent"] = {
            "primary_action": plan.get("primary_action"),
            "mf_text": plan.get("mf_text"),
            "source_policy": plan.get("source_policy"),
            "edit_source": plan.get("edit_source"),
            "preferred_source_id": plan.get("preferred_source_id"),
            "modules": plan.get("modules"),
        }
    else:
        # Merge plan fields useful for the right rail
        plan = bundle.get("motion_edit_plan") or {}
        intent = dict(msg_asst["session"]["intent"] or {})
        for k in (
            "primary_action",
            "mf_text",
            "source_policy",
            "edit_source",
            "preferred_source_id",
            "modules",
        ):
            if intent.get(k) is None and plan.get(k) is not None:
                intent[k] = plan[k]
        msg_asst["session"]["intent"] = intent

    with _LOCK:
        thread = _THREADS[thread_id]
        thread["messages"].extend([msg_user, msg_asst])
        thread["model"] = model_id
        # Only advance previous_run on success (failed dirs lack summary.json).
        if session_resp.get("ok") and session_resp.get("run_dir"):
            thread["previous_run"] = session_resp["run_dir"]
        elif previous_run and not thread.get("previous_run"):
            thread["previous_run"] = previous_run

    return {
        "ok": bool(session_resp.get("ok")),
        "thread_id": thread_id,
        "previous_run": _THREADS[thread_id].get("previous_run"),
        "user_message": msg_user,
        "assistant_message": msg_asst,
        "bundle": {
            "intent": bundle.get("intent"),
            "module_plan": bundle.get("module_plan"),
            "motion_edit_plan": bundle.get("motion_edit_plan"),
        },
    }


def _compose_assistant_text(
    session: Dict[str, Any],
    reasoning: Dict[str, Any],
    *,
    plan: Optional[Dict[str, Any]] = None,
) -> str:
    intent = session.get("intent") or {}
    plan = plan or {}
    q = session.get("quality") or {}
    action = (
        intent.get("primary_action")
        or plan.get("primary_action")
        or "—"
    )
    mf = (
        intent.get("mf_text")
        or plan.get("mf_text")
        or reasoning.get("prompt_compiled")
        or ""
    )
    lines = [
        f"已编译动作 **{action}** → MotionsFix 文本 `{mf}`。",
    ]
    if session.get("from_demo_cache"):
        lines.append("使用 demo 缓存结果（即时预览）。取消「快速演示」可强制全量管线。")
    refine = (plan.get("refine") if isinstance(plan, dict) else None) or {}
    if not refine and isinstance(session.get("intent"), dict):
        refine = ((session.get("intent") or {}).get("revision") or {}).get("refine") or {}
    if refine.get("source"):
        src = refine.get("source")
        src_note = {
            "llm": "DeepSeek/OpenCode Go LLM",
            "heuristic": "启发式兜底（LLM 未成功）",
            "ui": "时间轴点选算子",
            "edit_memory": "编辑配方记忆加强",
        }.get(str(src), str(src))
        lines.append(f"Refine 来源：{src_note}（source={src}）。")
    if refine.get("matched_recipes"):
        lines.append(f"匹配配方：{', '.join(str(x) for x in refine['matched_recipes'])}。")
    if q:
        lines.append(
            f"质量门：passed={q.get('passed')} score={q.get('score')}"
            + (f" hard={q.get('hard_failures')}" if q.get("hard_failures") else "")
        )
    cap = session.get("motion_caption") or {}
    if cap.get("caption"):
        lines.append(f"自动动作描述（M2T）：{cap.get('caption')}")
    if cap.get("critique") and cap.get("critique") != cap.get("caption"):
        lines.append(f"反馈摘要：{str(cap.get('critique'))[:220]}")
    tl = session.get("timeline") or q.get("timeline") or {}
    if tl.get("phases"):
        bad = [p for p in tl["phases"] if p.get("problem")]
        if bad:
            bits = [
                f"{p.get('label') or p.get('id')} f{p.get('start')}–{p.get('end')}"
                for p in bad
            ]
            lines.append("时间轴问题段：" + "；".join(bits) + "（见预览下方进度条）。")
        else:
            lines.append(
                f"时间轴：sit_start=f{tl.get('sit_start')} contact=f{tl.get('contact')} "
                f"T={tl.get('T')}（预览下方可查看阶段条）。"
            )
    if session.get("error") and not session.get("ok"):
        err = str(session.get("error"))
        lines.append(f"错误：{err}")
        if "v05_pipeline failed" in err:
            lines.append(
                "提示：若只是用了接近 demo 的提示词，勾选「快速演示」可走缓存；"
                "全量失败常见于 MotionsFix 环境依赖（aitviewer/USD）被系统拦截。"
            )
    else:
        lines.append("右侧可查看动作预览与时间轴；继续输入改进意见将进入 revise。")
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[chat_ui] " + (fmt % args) + "\n")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, code: int, data: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            html = (UI_DIR / "index.html").read_bytes()
            self._bytes(200, html, "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            fp = UI_DIR / rel
            if not fp.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            self._bytes(200, fp.read_bytes(), ctype)
            return
        if path == "/api/models":
            self._json(200, {"models": MODELS})
            return
        if path == "/api/demos":
            demos = []
            for c in _load_demos():
                demos.append(
                    {
                        "id": c.get("id"),
                        "prompt": c.get("prompt"),
                        "primary_action": c.get("primary_action"),
                        "ok": c.get("ok"),
                        "preview_url": _rel_url(c.get("preview_gif")),
                        "quality": c.get("quality"),
                    }
                )
            self._json(200, {"cases": demos})
            return
        if path == "/api/handoff":
            if HANDOFF_PATH.is_file():
                self._json(200, json.loads(HANDOFF_PATH.read_text(encoding="utf-8-sig")))
            else:
                self._json(404, {"error": "handoff missing; run scripts/session/run_demos.py"})
            return
        if path == "/api/thread":
            qs = parse_qs(parsed.query)
            tid = (qs.get("id") or [""])[0]
            with _LOCK:
                th = _THREADS.get(tid)
            if not th:
                self._json(404, {"error": "unknown thread"})
                return
            self._json(200, {"thread_id": tid, **th})
            return
        if path == "/api/file":
            qs = parse_qs(parsed.query)
            rel = unquote((qs.get("path") or [""])[0])
            if not rel or ".." in rel:
                self._json(400, {"error": "bad path"})
                return
            fp = _safe_under_repo(REPO_ROOT / rel)
            if not fp.is_file():
                self._json(404, {"error": "file not found"})
                return
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            self._bytes(200, fp.read_bytes(), ctype)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if parsed.path == "/api/chat":
            try:
                out = handle_chat(body)
                self._json(200, out)
            except Exception as exc:  # noqa: BLE001
                self._json(
                    500,
                    {"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-2000:]},
                )
            return
        if parsed.path == "/api/refine_op":
            # Closed-set UI op → Refine JSON (no NL keywords).
            try:
                sys.path.insert(0, str(INTENT_DIR))
                from refine_schema import (
                    PHASE_IDS,
                    UI_ACTIONS,
                    is_known_phase_id,
                    refine_from_phase_action,
                )

                phase_id = str(body.get("phase_id") or "")
                action = str(body.get("action") or "")
                if not is_known_phase_id(phase_id) or action not in UI_ACTIONS:
                    self._json(
                        400,
                        {
                            "ok": False,
                            "error": (
                                f"need known phase_id (e.g. {PHASE_IDS[:8]}…) "
                                f"and action in {UI_ACTIONS}; got phase_id={phase_id!r} action={action!r}"
                            ),
                        },
                    )
                    return
                refine = refine_from_phase_action(
                    phase_id,
                    action,
                    previous_action=body.get("previous_action") or "sit_down",
                    speed_scale=body.get("speed_scale"),
                )
                self._json(200, {"ok": True, "refine": refine})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if parsed.path == "/api/plan":
            # Fast path: reasoning only
            prompt = (body.get("prompt") or "").strip()
            mode = body.get("mode") or "generate"
            model_id = body.get("model") or "agent-heuristic"
            bundle = build_full_agent_bundle(prompt, mode=mode, task_id="UI_PLAN")
            self._json(
                200,
                {
                    "ok": True,
                    "reasoning": _build_reasoning(bundle, model_id),
                    "bundle": bundle,
                },
            )
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    if not (UI_DIR / "index.html").is_file():
        raise SystemExit(f"Missing UI at {UI_DIR / 'index.html'}")
    OUT_SESSION.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"[chat_ui] {url}")
    print("[chat_ui] demos: GET /api/demos  |  chat: POST /api/chat")
    print("[chat_ui] refine ops: POST /api/refine_op  |  timeline UI clicks → closed Refine JSON")
    try:
        sys.path.insert(0, str(INTENT_DIR))
        from llm_refine import _cfg, llm_available

        if llm_available():
            cfg = _cfg()
            print(
                f"[chat_ui] refine LLM: ON  source={cfg.get('key_source')}  "
                f"base={cfg.get('base_url')}  model={cfg.get('model')}"
            )
        else:
            print("[chat_ui] refine LLM: OFF → heuristic fallback")
    except Exception:  # noqa: BLE001
        print("[chat_ui] refine LLM: OFF → heuristic fallback")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chat_ui] bye")


if __name__ == "__main__":
    main()
