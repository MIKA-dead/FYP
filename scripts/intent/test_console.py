"""
Local Intent Routing test console (dev harness for you — not the teammate product UI).

  python scripts/intent/test_console.py
  → open http://127.0.0.1:8765

Features:
  - Pick gold prompts or type freeform
  - Live view: modules called, timeline, plan deps, auto score
  - Manual verdict override + save
  - Export handoff JSON for UI teammates
"""
from __future__ import annotations

import json
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_intent import parse_prompt  # noqa: E402
from route_modules import route_modules  # noqa: E402
from mock_modules import run_plan  # noqa: E402
from evaluate_decision import evaluate_trace  # noqa: E402

GOLD_PATH = ROOT / "test-samples" / "intent" / "prompts_with_gold.json"
OUT_DIR = ROOT / "outputs" / "intent"
SESSION_PATH = OUT_DIR / "console_session.json"
HANDOFF_PATH = OUT_DIR / "handoff_for_ui.json"
PORT = 8765

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Intent Routing 测试台</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a222c;
    --line: #2c3846;
    --text: #e7eef6;
    --muted: #8fa3b8;
    --ok: #3dd68c;
    --partial: #f5c542;
    --bad: #ff6b6b;
    --accent: #4da3ff;
    --chip: #243041;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a3050, transparent), var(--bg);
    color: var(--text); min-height: 100vh;
  }
  header {
    padding: 18px 24px; border-bottom: 1px solid var(--line);
    display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap;
  }
  header h1 { margin: 0; font-size: 1.25rem; font-weight: 650; }
  header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .layout {
    display: grid; grid-template-columns: 340px 1fr; gap: 0; min-height: calc(100vh - 64px);
  }
  @media (max-width: 960px) { .layout { grid-template-columns: 1fr; } }
  aside {
    border-right: 1px solid var(--line); background: rgba(26,34,44,.9);
    padding: 16px; overflow: auto; max-height: calc(100vh - 64px);
  }
  main { padding: 16px 20px; overflow: auto; max-height: calc(100vh - 64px); }
  .case {
    border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; margin-bottom: 8px;
    cursor: pointer; background: var(--panel);
  }
  .case:hover, .case.active { border-color: var(--accent); }
  .case .id { font-weight: 650; color: var(--accent); }
  .case .meta { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
  label { display:block; font-size: 0.8rem; color: var(--muted); margin: 10px 0 4px; }
  textarea, select, input[type=text] {
    width: 100%; background: #101820; color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 10px; font: inherit;
  }
  textarea { min-height: 120px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  button {
    border: 0; border-radius: 8px; padding: 10px 14px; font-weight: 600; cursor: pointer;
    background: var(--accent); color: #041018;
  }
  button.secondary { background: var(--chip); color: var(--text); }
  button:disabled { opacity: .5; cursor: wait; }
  .panel {
    border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
    padding: 14px 16px; margin-bottom: 14px;
  }
  .panel h2 { margin: 0 0 10px; font-size: 1rem; }
  .verdict {
    display: inline-flex; align-items: center; gap: 8px; font-size: 1.1rem; font-weight: 700;
  }
  .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
  .badge.correct { background: rgba(61,214,140,.2); color: var(--ok); }
  .badge.partial { background: rgba(245,197,66,.2); color: var(--partial); }
  .badge.wrong { background: rgba(255,107,107,.2); color: var(--bad); }
  .badge.freeform { background: rgba(77,163,255,.2); color: var(--accent); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
    padding: 4px 10px; font-size: 0.8rem;
  }
  .chip.frozen { opacity: .55; text-decoration: line-through; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  .ok { color: var(--ok); } .fail { color: var(--bad); }
  pre {
    margin: 0; padding: 12px; background: #101820; border-radius: 8px;
    overflow: auto; max-height: 320px; font-size: 0.75rem; line-height: 1.45;
  }
  .muted { color: var(--muted); font-size: 0.85rem; }
  #banner {
    display: none; margin: 0; padding: 10px 24px; background: rgba(255,107,107,.15);
    color: var(--bad); border-bottom: 1px solid var(--line); font-size: 0.9rem;
  }
  #banner.show { display: block; }
</style>
</head>
<body>
<header>
  <h1>Intent Routing 测试台</h1>
  <p>开发自测用 · 输入提示词 → 看模块调用与评分 · 导出给 UI 组员</p>
</header>
<div id="banner"></div>
<div class="layout">
  <aside>
    <div class="row">
      <button type="button" class="secondary" onclick="loadCases()">刷新用例</button>
      <button type="button" class="secondary" onclick="runAll()">跑全部 gold</button>
    </div>
    <div id="cases" style="margin-top:12px"></div>
    <hr style="border-color:var(--line);margin:16px 0" />
    <label>模式</label>
    <select id="mode">
      <option value="generate">generate</option>
      <option value="revise">revise</option>
    </select>
    <label>任务 ID（可空）</label>
    <input id="taskId" type="text" placeholder="FREE / P001" />
    <label>previous_task_id（修订时）</label>
    <input id="prevId" type="text" placeholder="P001" />
    <label>refs JSON（可选）</label>
    <textarea id="refs" placeholder='[{"id":"N2","role":"pose"}]'></textarea>
    <label>提示词</label>
    <textarea id="prompt" placeholder="在此粘贴或输入提示词…"></textarea>
    <div class="row">
      <button type="button" id="runBtn" onclick="runOnce()">运行解析</button>
      <button type="button" class="secondary" onclick="exportHandoff()">导出给组员</button>
    </div>
    <p class="muted" style="margin-top:10px">结果会写入 outputs/intent/</p>
  </aside>
  <main>
    <div class="panel" id="summaryPanel">
      <h2>实时结果</h2>
      <p class="muted">选择左侧用例或输入提示词后点击「运行解析」。</p>
    </div>
    <div class="panel"><h2>调用模块</h2><div id="modules" class="chips muted">—</div></div>
    <div class="panel"><h2>Timeline</h2><div id="timeline" class="muted">—</div></div>
    <div class="panel"><h2>Plan 步骤</h2><div id="plan" class="muted">—</div></div>
    <div class="panel"><h2>Gold 检查项</h2><div id="checks" class="muted">—</div></div>
    <div class="panel">
      <h2>人工判定（交给组员前可覆盖）</h2>
      <div class="row">
        <select id="manualVerdict">
          <option value="">保持自动</option>
          <option value="correct">correct</option>
          <option value="partial">partial</option>
          <option value="wrong">wrong</option>
        </select>
        <input id="manualNote" type="text" placeholder="备注（可选）" style="flex:1;min-width:180px" />
        <button type="button" class="secondary" onclick="saveOverride()">保存判定</button>
      </div>
    </div>
    <div class="panel"><h2>Decision Trace JSON</h2><pre id="json">{}</pre></div>
  </main>
</div>
<script>
let currentTrace = null;
let cases = [];

function showBanner(msg) {
  const b = document.getElementById('banner');
  b.textContent = msg;
  b.classList.add('show');
}
function hideBanner() {
  document.getElementById('banner').classList.remove('show');
}
function serverDownMsg(err) {
  return '后端未连接（ERR_CONNECTION_REFUSED）。请在项目根目录终端运行：python scripts/intent/test_console.py ，保持窗口不关，然后刷新本页。详情：' + (err && err.message ? err.message : err);
}

async function loadCases() {
  try {
    const res = await fetch('/api/cases');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    hideBanner();
    cases = data.cases || [];
    const el = document.getElementById('cases');
    el.innerHTML = cases.map(c => `
      <div class="case" data-id="${c.id}" onclick="pickCase('${c.id}')">
        <div class="id">${c.id} · ${c.mode}</div>
        <div>${c.title || ''}</div>
        <div class="meta">${c.difficulty || ''} · ${((c.prompt||'').slice(0,48))}…</div>
      </div>`).join('');
  } catch (e) {
    showBanner(serverDownMsg(e));
    document.getElementById('cases').innerHTML = '<p class="fail">无法加载用例：服务未启动</p>';
  }
}

function pickCase(id) {
  const c = cases.find(x => x.id === id);
  if (!c) return;
  document.querySelectorAll('.case').forEach(n => n.classList.toggle('active', n.dataset.id === id));
  document.getElementById('mode').value = c.mode || 'generate';
  document.getElementById('taskId').value = c.id;
  document.getElementById('prevId').value = c.previous_task_id || '';
  document.getElementById('prompt').value = c.prompt || '';
  document.getElementById('refs').value = JSON.stringify(c.refs || [], null, 2);
  runOnce();
}

function renderTrace(trace) {
  currentTrace = trace;
  const review = trace.review || {};
  const intent = trace.intent || {};
  const plan = trace.plan || {};
  const verdict = (review.manual_override || review.verdict || 'freeform');
  const score = review.score != null ? `${review.score}% (${review.passed}/${review.total})` : '无 gold（自由输入）';
  document.getElementById('summaryPanel').innerHTML = `
    <h2>实时结果 · ${trace.case_id || ''}</h2>
    <div class="verdict"><span class="badge ${verdict}">${verdict}</span> ${score}</div>
    <p class="muted" style="margin:8px 0 0">${(intent.decision&&intent.decision.why)||''}</p>
    ${(intent.needs_clarify||[]).length ? `<p class="fail">需澄清：${intent.needs_clarify.join('；')}</p>` : ''}
  `;

  const modules = intent.task_mode === 'revise'
    ? ((intent.revision||{}).modules_to_rerun || intent.modules_needed || [])
    : (intent.modules_needed || []);
  const frozen = (intent.revision||{}).modules_frozen || plan.frozen_modules || [];
  document.getElementById('modules').innerHTML = [
    ...modules.map(m => `<span class="chip">${m}</span>`),
    ...frozen.map(m => `<span class="chip frozen">${m} (frozen)</span>`)
  ].join('') || '—';

  const tl = intent.timeline || [];
  document.getElementById('timeline').innerHTML = tl.length ? `<table>
    <tr><th>seg</th><th>action</th><th>HML3D</th><th>speed</th><th>steps</th><th>gesture</th><th>speech</th></tr>
    ${tl.map(t => `<tr>
      <td>${t.seg}</td><td>${t.action}</td><td>${t.humanml3d_class||''}</td>
      <td>${t.speed||''}</td><td>${t.steps??''}</td><td>${t.gesture||''}</td>
      <td>${t.speech ? (t.speech.lang||'')+' '+(t.speech.script_id||'') : ''}</td>
    </tr>`).join('')}
  </table>` : '<span class="muted">（修订模式通常无新 timeline）</span>';

  const steps = plan.steps || [];
  document.getElementById('plan').innerHTML = steps.length ? `<table>
    <tr><th>#</th><th>module</th><th>agent</th><th>deps</th><th>parallel</th><th>why</th></tr>
    ${steps.map(s => `<tr>
      <td>${s.step}</td><td>${s.module}</td><td>${s.owner_agent||''}</td>
      <td>${(s.depends_on||[]).join(',')}</td><td>${s.parallel_group}</td><td>${s.why||''}</td>
    </tr>`).join('')}
  </table>` : '—';

  const checks = (review.checks || []);
  document.getElementById('checks').innerHTML = checks.length ? `<table>
    <tr><th>结果</th><th>检查项</th><th>细节</th></tr>
    ${checks.map(c => `<tr>
      <td class="${c.ok?'ok':'fail'}">${c.ok?'OK':'FAIL'}</td>
      <td>${c.name}</td><td>${c.detail||''}</td>
    </tr>`).join('')}
  </table>` : '<span class="muted">自由输入无 gold 对照</span>';

  document.getElementById('json').textContent = JSON.stringify(trace, null, 2);
  document.getElementById('manualVerdict').value = review.manual_override || '';
  document.getElementById('manualNote').value = review.manual_note || '';
}

async function runOnce() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  let refs = [];
  try {
    const raw = document.getElementById('refs').value.trim();
    if (raw) refs = JSON.parse(raw);
  } catch (e) {
    alert('refs JSON 无效');
    btn.disabled = false;
    return;
  }
  const body = {
    prompt: document.getElementById('prompt').value,
    mode: document.getElementById('mode').value,
    task_id: document.getElementById('taskId').value || null,
    previous_task_id: document.getElementById('prevId').value || null,
    refs
  };
  try {
    const res = await fetch('/api/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const trace = await res.json();
    hideBanner();
    if (trace.error) alert(trace.error);
    else renderTrace(trace);
  } catch (e) {
    showBanner(serverDownMsg(e));
    alert(serverDownMsg(e));
  } finally {
    btn.disabled = false;
  }
}

async function runAll() {
  try {
    const res = await fetch('/api/run_all', { method:'POST' });
    const data = await res.json();
    hideBanner();
    alert(`全部跑完：correct=${data.correct} partial=${data.partial} wrong=${data.wrong}`);
    if (data.last_trace) renderTrace(data.last_trace);
    loadCases();
  } catch (e) {
    showBanner(serverDownMsg(e));
    alert(serverDownMsg(e));
  }
}

async function saveOverride() {
  if (!currentTrace) return alert('先运行一次');
  try {
    const res = await fetch('/api/override', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        case_id: currentTrace.case_id,
        manual_override: document.getElementById('manualVerdict').value || null,
        manual_note: document.getElementById('manualNote').value || ''
      })
    });
    const trace = await res.json();
    hideBanner();
    renderTrace(trace);
  } catch (e) {
    showBanner(serverDownMsg(e));
    alert(serverDownMsg(e));
  }
}

async function exportHandoff() {
  try {
    const res = await fetch('/api/export', { method:'POST' });
    const data = await res.json();
    hideBanner();
    alert(`已导出：${data.path}\n含 ui_contract + 用例 + 最近 traces 摘要`);
  } catch (e) {
    showBanner(serverDownMsg(e));
    alert(serverDownMsg(e));
  }
}

loadCases();
</script>
</body>
</html>
"""


def load_gold() -> dict:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def find_case(case_id: str | None) -> dict | None:
    if not case_id:
        return None
    for c in load_gold()["cases"]:
        if c["id"] == case_id:
            return c
    return None


def build_trace(
    prompt: str,
    mode: str = "generate",
    refs: list | None = None,
    task_id: str | None = None,
    previous_task_id: str | None = None,
    case: dict | None = None,
) -> dict:
    tid = task_id or (case or {}).get("id") or f"FREE_{datetime.now().strftime('%H%M%S')}"
    intent = parse_prompt(
        prompt=prompt,
        mode=mode,
        refs=refs if refs is not None else (case or {}).get("refs"),
        task_id=tid,
        previous_task_id=previous_task_id or (case or {}).get("previous_task_id"),
    )
    plan = route_modules(intent)
    mock_results = run_plan(intent, plan)
    trace = {
        "case_id": tid,
        "title": (case or {}).get("title") or "freeform",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "prompt": prompt,
            "mode": mode,
            "refs": refs if refs is not None else (case or {}).get("refs") or [],
            "previous_task_id": previous_task_id or (case or {}).get("previous_task_id"),
        },
        "intent": intent,
        "plan": plan,
        "mock_results": mock_results,
    }
    if case and case.get("gold"):
        trace["review"] = evaluate_trace(trace, case["gold"])
    else:
        # try match by id
        matched = find_case(tid)
        if matched and matched.get("gold"):
            trace["review"] = evaluate_trace(trace, matched["gold"])
        else:
            trace["review"] = {
                "verdict": "freeform",
                "score": None,
                "passed": None,
                "total": None,
                "checks": [],
                "manual_override": None,
                "manual_note": "No gold label — inspect modules manually.",
            }
    return trace


def save_trace(trace: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{trace['case_id']}_decision_trace.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    # append session log
    session = []
    if SESSION_PATH.exists():
        try:
            session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session = []
    session.append(
        {
            "case_id": trace["case_id"],
            "verdict": (trace.get("review") or {}).get("manual_override")
            or (trace.get("review") or {}).get("verdict"),
            "score": (trace.get("review") or {}).get("score"),
            "modules": (trace.get("intent") or {}).get("modules_needed"),
            "at": trace.get("created_at"),
        }
    )
    SESSION_PATH.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_handoff() -> Path:
    gold = load_gold()
    traces = []
    for p in sorted(OUT_DIR.glob("*_decision_trace.json")):
        try:
            traces.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    # keep latest per case_id
    latest: dict[str, dict] = {}
    for t in traces:
        latest[t.get("case_id") or p.name] = t
    handoff = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "note": "Handoff for UI teammates. Wire UI to ui_contract; routing logic stays on agent side.",
        "ui_contract": gold.get("ui_contract"),
        "module_registry_path": ".agents/skills/intent-routing/module_registry.json",
        "intent_schema_path": ".agents/skills/intent-routing/intent_schema.json",
        "cases": [
            {
                "id": c["id"],
                "title": c.get("title"),
                "mode": c.get("mode"),
                "prompt": c["prompt"],
                "refs": c.get("refs") or [],
                "previous_task_id": c.get("previous_task_id"),
            }
            for c in gold["cases"]
        ],
        "latest_results": [
            {
                "case_id": t.get("case_id"),
                "title": t.get("title"),
                "verdict": (t.get("review") or {}).get("manual_override")
                or (t.get("review") or {}).get("verdict"),
                "score": (t.get("review") or {}).get("score"),
                "modules_needed": (t.get("intent") or {}).get("modules_needed"),
                "revision": (t.get("intent") or {}).get("revision"),
                "plan_summary": (t.get("plan") or {}).get("summary"),
            }
            for t in latest.values()
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HANDOFF_PATH.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = OUT_DIR / "HANDOFF_README.md"
    readme.write_text(
        """# UI 组员对接包

本目录由 Intent 测试台导出，**不是最终产品 UI**。

## 你需要接的输入
见 `handoff_for_ui.json` → `ui_contract`：

- generate: `{ prompt, refs[], mode: "generate" }`
- revise: `{ prompt, mode: "revise", previous_task_id }`

## 你需要展示的输出（建议）
1. timeline 分段（动作 / 速度 / 手势 / 台词）
2. modules_needed（将要调用的模块芯片）
3. revise 时的 modules_to_rerun vs modules_frozen
4. decision.why（一句话理由）

## 后端接口（当前演示）
本地测试：`python scripts/intent/test_console.py` → `POST /api/run`

真实接入时由 agent/服务返回同等结构的 `intent` + `plan`。
""",
        encoding="utf-8",
    )
    return HANDOFF_PATH


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict | list) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/cases":
            gold = load_gold()
            self._json(200, {"cases": gold["cases"], "ui_contract": gold.get("ui_contract")})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/run":
                body = self._read_json()
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    self._json(400, {"error": "prompt required"})
                    return
                case = find_case(body.get("task_id"))
                # If freeform matches a gold prompt text, still ok
                trace = build_trace(
                    prompt=prompt,
                    mode=body.get("mode") or (case or {}).get("mode") or "generate",
                    refs=body.get("refs"),
                    task_id=body.get("task_id"),
                    previous_task_id=body.get("previous_task_id"),
                    case=case,
                )
                save_trace(trace)
                self._json(200, trace)
                return

            if path == "/api/run_all":
                gold = load_gold()
                summary = {"correct": 0, "partial": 0, "wrong": 0, "freeform": 0}
                last = None
                for case in gold["cases"]:
                    trace = build_trace(
                        prompt=case["prompt"],
                        mode=case.get("mode", "generate"),
                        refs=case.get("refs"),
                        task_id=case["id"],
                        previous_task_id=case.get("previous_task_id"),
                        case=case,
                    )
                    save_trace(trace)
                    v = (trace.get("review") or {}).get("verdict") or "freeform"
                    summary[v] = summary.get(v, 0) + 1
                    last = trace
                # summary file
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUT_DIR / "summary.json").write_text(
                    json.dumps({**summary, "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._json(200, {**summary, "last_trace": last})
                return

            if path == "/api/override":
                body = self._read_json()
                case_id = body.get("case_id")
                path_t = OUT_DIR / f"{case_id}_decision_trace.json"
                if not path_t.exists():
                    self._json(404, {"error": "trace not found"})
                    return
                trace = json.loads(path_t.read_text(encoding="utf-8"))
                review = trace.setdefault("review", {})
                review["manual_override"] = body.get("manual_override") or None
                review["manual_note"] = body.get("manual_note") or ""
                save_trace(trace)
                self._json(200, trace)
                return

            if path == "/api/export":
                p = export_handoff()
                self._json(200, {"path": str(p.relative_to(ROOT)).replace("\\", "/")})
                return

            self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": str(e)})

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[console] " + (fmt % args) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Intent test console → {url}")
    print("Ctrl+C to stop")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
