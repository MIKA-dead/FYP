"""
Render the acceptance page for the two review points.

Reads one live episode and one 20-goal report, writes a single static HTML file
so the result can be screenshotted (and read) without a slide deck.

  python scripts/agent/build_acceptance.py \
      --report outputs/agent_bench/_v3_final8/report.json \
      --episode outputs/session/ui_runs/70c66979_UI_AGENT_20261115-213540 \
      --out outputs/agent_bench/_v3_final8/acceptance.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]


def esc(s) -> str:
    return html.escape(str(s))


def table(headers: List[str], rows: List[List[str]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def obs_summary(action: str, o: Dict[str, Any]) -> str:
    m = o.get("metrics") or {}
    if action == "generate_motion":
        rej = o.get("draft_rejection") or {}
        rej_txt = (
            f"抽 {rej.get('drawn')} 稿 / 接受 {rej.get('accepted')} / 弃 {rej.get('rejected')}"
            if rej.get("drawn") else "单稿"
        )
        req = o.get("requirement") or {}
        return (f"generator=<b>{esc(o.get('generator'))}</b>, {o.get('frames')} 帧，{rej_txt}<br>"
                f"pelvis_drop {m.get('pelvis_drop', 0):.3f} m · "
                f"arm_raise_right {m.get('arm_raise_right', 0):.2f} · "
                f"requirement gap {esc(req.get('gap'))}")
    if action == "describe_motion":
        sim = o.get("similarity") or {}
        ver = o.get("caption_verdict") or {}
        return (f"caption「<b>{esc(o.get('caption'))}</b>」<br>"
                f"goal↔caption: GloVe {sim.get('glove_cosine', 0):.3f}, "
                f"content recall {sim.get('content_recall', 0):.2f} → verdict "
                f"<b>{esc(ver.get('level'))}</b> (needs_fix={esc(ver.get('needs_fix'))})")
    if action == "restore_motion":
        return (f"回到更近的一稿：gap {esc(o.get('gap_before'))} → "
                f"<b>{esc(o.get('gap_after'))}</b>，原因：{esc(o.get('reason'))}")
    if action == "edit_motion":
        fol = o.get("instruction_following") or {}
        chk = (fol.get("checks") or [{}])[0]
        extra = ""
        if not o.get("ok"):
            extra = (f"<br><span class=\"bad\">{esc(o.get('error'))}</span> · "
                     f"已回退={esc(o.get('rolled_back'))} — 原片段保留，失败被记录")
        return (f"family=<b>{esc(o.get('family'))}</b>, 已应用="
                f"{esc(o.get('ok'))}, 跟随={esc(fol.get('followed'))}<br>"
                f"{esc(chk.get('attribute'))} Δ {chk.get('delta')} · "
                f"gap {esc(o.get('gap_before'))} → {esc(o.get('gap_after'))} · "
                f"非目标 MPJPE {o.get('preservation', {}).get('non_target_mpjpe_norm')}"
                + extra)
    if action == "check_requirement":
        checks = o.get("checks") or []
        ok = [c for c in checks if c.get("ok")]
        return (f"<b>{'满足' if o.get('satisfied') else '未满足'}</b> "
                f"({len(ok)}/{len(checks)} 项通过, gap {esc(o.get('requirement_gap'))})<br>"
                + " · ".join(f"{esc(c['check'])}={c['value']}" for c in checks[:3]))
    if action == "finish":
        return (f"checker={esc(o.get('checker_satisfied'))}, "
                f"agent={esc(o.get('agent_claim_satisfied'))}, "
                f"一致={esc(o.get('agreement'))}")
    return ""


def build(report_path: Path, episode_dir: Path, out: Path) -> Path:
    report = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
    ep = json.loads((Path(episode_dir) / "agent_run.json").read_text(encoding="utf-8-sig"))
    gen, edit, cap, loop = (
        report["module_a_generation"],
        report["module_b_editing"],
        report["module_c_caption"],
        report["loop"],
    )
    battery = (edit.get("battery") or {}).get("aggregate") or {}

    rows = []
    for s in ep["steps"]:
        thought = (s.get("thought") or "").strip()
        nxt = (s.get("assessment") or {}).get("next_edit") or ""
        if nxt:
            thought += f'<br><span class="nxt">→ 下一句修改文本：<b>“{esc(nxt)}”</b></span>'
        rows.append([
            f"<b>{s['step']}</b>",
            f"<code>{esc(s['action'])}</code>",
            thought,
            obs_summary(s["action"], s.get("observation") or {}),
            str((s.get("assessment") or {}).get("alignment") or ""),
        ])

    n_goals = len(loop["rows"])
    n_ok = sum(1 for r in loop["rows"] if r.get("satisfied"))
    check_rows = [
        ["自由文本输入", "UI 输入框 → <code>goal</code> 字符串",
         f"本页目标「{esc(ep['goal'])}」，无人写死步骤"],
        ["LLM 自己选工具 (a)(b)(c)",
         "<code>agent.py</code> SYSTEM + TOOL_SPECS，一次一个动作",
         f"本次 {ep['summary']['tool_calls']} 次工具调用全部由控制器选出"],
        ["(a) 生成动作", "<code>generate_motion</code> → MLD / 检索",
         "生成端按测量属性拒绝采样，先把最接近的一稿交给循环"],
        ["(c) 描述 + 相似度判断", "<code>describe_motion</code> → 学习 caption + 测量 grounded caption",
         "两段 caption 各自与目标做 GloVe / content recall / action family 比较，给出 match / uncertain / mismatch"],
        ["写成文本再 (b) 修改", "<code>edit_motion(instruction)</code>",
         "判断缺什么就写哪一句；编辑后 gap 变大时归档里仍留着更近的一稿"],
        ["循环直到满足", "MANDATORY ORDER: describe → check → edit / restore / finish",
         "finish 前必须 check；若存在更近的旧稿，finish 被纠正为 restore_motion"],
        ["(i) 量化指标代替 SCORE", "report.json 的 module_a/b/c + loop 四段",
         "下方四表：无 SCORE 字段，全部为可复算指标"],
    ]

    battery_table = ""
    case_table = ""
    if battery:
        bat_src = Path(str((edit.get("battery") or {}).get("source") or ""))
        cases = []
        if bat_src.is_file():
            cases = json.loads(bat_src.read_text(encoding="utf-8-sig")).get("rows") or []
        if cases:
            case_rows = []
            for c in cases:
                chk = (c.get("checks") or [{}])[0]
                case_rows.append([
                    esc(c.get("id")),
                    esc(c.get("family") or "—"),
                    esc(c.get("instruction")),
                    esc(c.get("expect")),
                    f"<b>{esc(c.get('status'))}</b>",
                    esc(chk.get("attribute") or "—"),
                    f"{esc(chk.get('delta'))} {esc(chk.get('expected'))}".strip(),
                ])
            case_table = f"""
<div class="card" id="cases" style="margin-top:14px">
  <h3>(b) 专项基准的逐条结果 <span class="muted">{len(cases)} 例，每例从各自的源片段副本出发</span></h3>
  {table(["#", "family", "instruction", "期望", "实际", "测量属性", "Δ / 期望值"], case_rows)}
  <div class="muted" style="margin-top:8px">
    期望 = apply 的用例要求指令被应用且目标属性按方向变化；期望 = refuse 的用例要求给出具名理由。
    没有一条是"看起来改了但其实没动"：无效果会被回退（见 no-effect 计数）。
  </div>
</div>"""
        fam = (edit.get("battery") or {}).get("by_family") or {}
        fam_rows = [
            [fam_name,
             f"{v.get('n_applied')}/{v.get('n_attempts')}",
             f"{v.get('instruction_following_rate')}",
             f"{v.get('mean_target_vs_rest')}"]
            for fam_name, v in fam.items()
        ]
        battery_table = f"""
<div class="card" id="battery" style="margin-bottom:14px">
  <h3>(b) 专项编辑基准 <span class="muted">{battery.get('n_scenarios')} 个必改用例 ·
    {battery.get('n_applied')} 已应用 / {battery.get('n_applicable')} 可应用 +
    {battery.get('n_expected_refusals')} 应拒绝</span></h3>
  {table(["指标", "值", "含义"], [
      ["instruction coverage", f"{battery.get('instruction_coverage')}",
       f"可应用用例中真正应用的比例（{battery.get('n_applied')}/{battery.get('n_applicable')}）"],
      ["refusal honesty", f"{battery.get('refusal_honesty')}",
       "应拒绝的用例是否都给出具名理由，而不是静默空操作"],
      ["instruction following", f"{battery.get('instruction_following_rate')}",
       "目标属性是否按指令方向变化（只统计已应用的编辑）"],
      ["no-effect edits", f"{battery.get('n_no_effect')}",
       "改了别的属性却没动指令点的属性 → 已回退到原片段"],
      ["silent no-ops", f"{battery.get('n_no_change')}",
       "输出与输入完全相同（应为 0）"],
      ["rollback rate", f"{battery.get('rollback_rate')}", "引入新物理损伤被回滚"],
      ["non-target MPJPE", f"{battery.get('mean_non_target_mpjpe_norm')}", "非目标区域位移 / 身高"],
      ["target vs rest", f"{battery.get('mean_target_vs_rest')}", "目标区变化 / 其余区变化"],
      ["frame retention", f"{battery.get('mean_frame_retention')}", "保留帧比例"],
  ])}
  <div class="muted" style="margin:10px 0 6px">
    按编辑族拆开（8 族，每族都从各自的源片段副本出发）：
  </div>
  {table(["family", "applied / attempts", "following", "target/rest"], fam_rows)}
</div>"""

    metric_tables = f"""<div class="grid2">
  <div class="card">
    <h3>(a) 生成模块 <span class="muted">MLD · {n_goals} goals</span></h3>
    {table(["指标", "值", "含义"], [
      ["plausibility · bone_cv", f"{gen['plausibility_mean']['bone_cv']:.4f}", "骨长变异系数，越小越刚体"],
      ["plausibility · foot_slide", f"{gen['plausibility_mean']['foot_slide']:.4f}", "脚底滑动 / 身高"],
      ["plausibility · jerk", f"{gen['plausibility_mean']['jerk']:.4f}", "加速度抖动"],
      ["floor penetration", f"{gen['plausibility_mean']['ground_penetration']:.4f}", "穿地深度 / 身高"],
      ["retrieval R@1 / R@3", f"{(gen['retrieval']['r@1'] or 0):.2f} / {(gen['retrieval']['r@3'] or 0):.2f}", "caption 能否找回目标句"],
      ["prompt↔caption cosine", f"{(gen['retrieval']['mean_cosine'] or 0):.3f}", "GloVe 平均余弦"],
      ["batch diversity", f"{(gen['diversity']['mean_pairwise'] or 0):.3f}", "批内两两差异"],
      ["drafts drawn / rejected", f"{loop['n_rejection_episodes']} 轮 · 弃 {loop['n_drafts_rejected']}", "生成端拒绝采样"],
    ])}
  </div>
  <div class="card">
    <h3>(b) 循环内编辑 <span class="muted">session revise · {edit['n']} applied</span></h3>
    {table(["指标", "值", "含义"], [
      ["edits applied / attempts", f"{edit['n']} / {edit['n_attempts']}", "应用数与尝试数"],
      ["rolled back", f"{edit['n_rolled_back']}", "引入新损伤被回滚"],
      ["instruction coverage", f"{edit['instruction_coverage']}", "指令在闭集能力内"],
      ["instruction following", f"{edit['instruction_following_rate']}", "目标属性是否真动了"],
      ["frame retention", f"{edit['mean_frame_retention']}", "保留帧比例"],
      ["non-target MPJPE", f"{edit['mean_non_target_mpjpe_norm']}", "非目标区域位移(归一)"],
      ["target vs rest", f"{edit['mean_target_vs_rest']}×", "目标区变化/其余区"],
      ["unsupported / no-change", f"{edit['n_unsupported']} / {edit['no_change_rate']}", "前端拒绝与空操作"],
    ])}
  </div>
  <div class="card">
    <h3>(c) 描述模块 <span class="muted">TM2T · {cap.get('n')} clips</span></h3>
    {table(["指标", "值", "含义"], [
      ["BLEU-1 / 2 / 3", f"{cap['aggregate']['bleu1']:.2f} / {cap['aggregate']['bleu2']:.2f} / {cap['aggregate']['bleu3']:.2f}", "n-gram 匹配"],
      ["BLEU-4", f"{cap['aggregate']['bleu4']:.2f}", "短参考下几乎不可用"],
      ["ROUGE-L", f"{cap['aggregate']['rouge_l']:.3f}", "最长公共子序列"],
      ["CIDEr-D", f"{cap['aggregate']['cider_d']:.3f}", "共识加权"],
      ["GloVe cosine", f"{cap['aggregate']['glove_cosine']:.3f}", "语义相似度(主指标)"],
    ])}
  </div>
  <div class="card">
    <h3>循环 <span class="muted">{n_goals} goals · 同一套工具</span></h3>
    {table(["指标", "值", "含义"], [
      ["goal satisfaction", f"{loop['success_rate']:.2f} ({n_ok}/{n_goals})", "checker 判定达成"],
      ["agent↔checker agreement", f"{loop['agreement_rate']}", "控制器是否校准"],
      ["check coverage", f"{loop['check_coverage']}", "finish 前是否查过"],
      ["mean edits / steps", f"{loop['mean_edits']} / {loop['mean_steps']}", "每目标编辑与步数"],
      ["resampled / restored", f"{loop.get('n_resampled_episodes')} / {loop.get('n_restored_episodes')}", "生成端补救与回退"],
      ["ended worse than best", f"{loop.get('n_worse_than_best')}", "最终稿劣于已测最接近稿的 episode 数"],
      ["mean alignment gain", f"{loop['mean_alignment_gain']}", "首末对齐提升"],
      ["mean seconds", f"{loop['mean_seconds']} s", "每目标耗时"],
      ["total tokens", f"{loop['total_tokens']:,}", "全部目标合计"],
    ])}
  </div>
</div>"""

    goal_rows = []
    for i, r in enumerate(loop["rows"], 1):
        verdict = ('<span class="ok">✓</span>' if r.get("satisfied")
                   else '<span class="bad">✗</span>')
        goal_rows.append([
            str(i),
            esc(r["goal"]),
            str(r.get("edits") or 0),
            str(r.get("resamples") or 0),
            str(r.get("restores") or 0),
            f"{float(r.get('best_gap_seen') or 0):.2f}",
            f"{float(r.get('final_gap') or 0):.2f}",
            verdict,
            str(r.get("final_alignment") if r.get("final_alignment") is not None else "—"),
            f"{float(r.get('seconds') or 0):.0f}",
        ])

    failed = [r for r in loop["rows"] if not r.get("satisfied")]
    failed_txt = (
        "全部目标都由关节测量确认达成。" if not failed else
        "未达成的目标如实保留测量值：" + "；".join(
            f"#{loop['rows'].index(r) + 1} {esc(r['goal'])} "
            f"(final gap {float(r.get('final_gap') or 0):.2f}, "
            f"caption「{esc(r.get('final_caption'))}」)"
            for r in failed
        )
    )

    HTML = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Agentic pipeline · 验收</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 28px 34px 40px; background: #f6f7f9; color: #1b1f24;
         font: 14px/1.55 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; }}
  h2 {{ font-size: 17px; margin: 26px 0 10px; padding-left: 9px; border-left: 4px solid #1f4e79; }}
  h3 {{ font-size: 14px; margin: 0 0 8px; }}
  .sub {{ color: #5a5f66; margin-bottom: 4px; }}
  .muted {{ color: #7a8087; font-weight: 400; font-size: 12px; }}
  .card {{ background: #fff; border: 1px solid #e2e5e9; border-radius: 8px;
           padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th {{ text-align: left; background: #eef2f7; color: #1f4e79; font-weight: 600;
        padding: 6px 8px; border-bottom: 1px solid #dfe4ea; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  code {{ background: #f1f4f8; border-radius: 4px; padding: 1px 5px; font-size: 12px;
          color: #14406b; font-family: Consolas, monospace; }}
  .req td:first-child {{ white-space: nowrap; font-weight: 600; }}
  .nxt {{ color: #0b6b3a; }}
  .ok {{ color: #0b6b3a; font-weight: 600; }}
  .bad {{ color: #a3261b; font-weight: 600; }}
  .note {{ background: #fff8e6; border: 1px solid #f0d9a6; border-radius: 8px;
           padding: 10px 14px; margin-top: 12px; font-size: 13px; }}
</style></head>
<body>
<h1>学长的两条要求：实现位置与实测结果</h1>
<div class="sub">现场 episode <code>{esc(Path(episode_dir).name)}</code> ·
  20-goal 报告 <code>{esc(Path(report_path).parent.name)}</code> ·
  policy <code>{esc(report['config']['policy'])}</code></div>

<h2>1. 要求 → 代码位置 → 本次证据</h2>
<div class="card">
{table(["要求", "实现位置", "本次实测"], check_rows, "req")}
</div>

<h2>2. 一次完整循环的轨迹（自由文本进，工具调用出）</h2>
<div class="card">
{table(["步", "控制器调用", "LLM 的理由 / 它写的修改文本", "观测（工具返回）", "对齐"], rows)}
<div class="note" style="margin-top:10px">
控制器自己判定不足、自己写出修改文本；编辑后重新描述、再让 checker 测量，通过才 finish。
<span class="ok">本次结果：{esc(ep['summary']['satisfied'])} ·
一致={esc(ep['summary']['agreement'])} · {ep['summary']['edits']} 次编辑 ·
{ep['summary']['seconds']} s · {ep['summary'].get('tokens', {}).get('total_tokens', 0):,} tokens。</span>
</div>
</div>

<h2>3. (i) 每个模块的量化指标（替代恒为 100 的 SCORE）</h2>
{battery_table}
{case_table}
{metric_tables}
<div class="note">
SCORE 已从报告指标里移除：<code>report.json</code> 只有
<code>module_a_generation</code> / <code>module_b_editing</code> /
<code>module_c_caption</code> / <code>loop</code> 四段，每个数字都能由 joints、
caption、工具轨迹重算。失败也如实计入：{failed_txt}
</div>

<h2>4. {n_goals} 个自由文本目标的结果（同一控制器与工具）</h2>
<div class="card">
{table(["#", "goal（自由文本）", "编辑", "重采样", "回退", "最接近 gap", "最终 gap", "达成", "对齐", "秒"], goal_rows)}
</div>

<div class="muted" style="margin-top:18px">
数据来源：{esc(str(report_path))} ·
{esc(str(Path(episode_dir) / 'agent_run.json'))}
</div>
</body></html>
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Acceptance page for the agentic review")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(build(args.report, args.episode, args.out))


if __name__ == "__main__":
    main()
