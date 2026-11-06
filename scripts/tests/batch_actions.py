"""Batch action tests: run many actions, then emit a GIF + an analysis card each.

  python scripts/tests/batch_actions.py --out outputs/tests/actions_20261105

Per case it writes <case>/run (session artifacts), <case>/before_after.gif,
<case>/flat.gif and <case>/ai_analysis.png, and a top-level report.md /
results.json. The analysis card is rendered from that run's own artifacts
(planner decision, geometry stages, M2T caption, critique, quality metrics), so
every number in it is the number the pipeline actually produced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "backends"))

from visualize_motion import SKELETON_CHAINS, load_joints, render_skeleton_video  # noqa: E402

UI_RUNS = REPO_ROOT / "outputs" / "session" / "ui_runs"

# last known-good clip per action, used as the base of a revise
BASES = {
    "walk_slow": "7a8c8ece_UI_20261105-151814",
    "walk_forward": "bd6e6404_UI_20261028-113149",
    "run_forward": "bd6e6404_UI_20261028-112445",
    "bow": "bd6e6404_UI_20261028-112710",
    "sit_down": "9e1f0296_UI_20261026-231857",
    "wave_right": "9e1f0296_UI_20261026-231556",
    "turn_in_place": "ae58d22c_UI_20261022-230022",
}

CASES: List[Dict[str, Any]] = [
    # --- new actions, from the dataset ------------------------------------
    {"id": "A1 jump", "mode": "generate", "prompt": "a person jumps up and lands"},
    {"id": "A2 kick", "mode": "generate", "prompt": "a person kicks with their right leg"},
    {"id": "A3 pick up", "mode": "generate", "prompt": "a person picks up an item from the floor"},
    # --- arm raise on new carriers ----------------------------------------
    {"id": "A4 turn + right arm high", "mode": "revise", "base": "turn_in_place",
     "prompt": "raise the right arm high"},
    {"id": "A5 bow + both arms overhead", "mode": "revise", "base": "bow",
     "prompt": "raise both arms high overhead"},
    {"id": "A6 sit + right arm high", "mode": "revise", "base": "sit_down",
     "prompt": "raise the right arm high"},
    {"id": "A7 wave + both arms overhead", "mode": "revise", "base": "wave_right",
     "prompt": "raise both arms overhead"},
    {"id": "A8 walk_forward + left arm high", "mode": "revise", "base": "walk_forward",
     "prompt": "raise the left arm high"},
    # --- tempo on new carriers --------------------------------------------
    {"id": "A9 run + slow down", "mode": "revise", "base": "run_forward",
     "prompt": "slow down a bit"},
    {"id": "A10 turn + slow down", "mode": "revise", "base": "turn_in_place",
     "prompt": "slow down a bit"},
    # --- real action change ------------------------------------------------
    {"id": "A11 walk -> turn around", "mode": "revise", "base": "walk_slow",
     "prompt": "turn around and walk back"},
]


def _load(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


def _setup_fonts() -> str:
    """Pick a CJK-capable font so the analysis card's Chinese labels render."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun"):
        if name in have:
            rcParams["font.family"] = [name]
            rcParams["axes.unicode_minus"] = False
            return name
    rcParams["axes.unicode_minus"] = False
    return ""


def _wrap(text: str, width: int) -> List[str]:
    out: List[str] = []
    for para in str(text or "").split("\n"):
        out += textwrap.wrap(para, width) or [""]
    return out


def render_pair(
    before: Optional[np.ndarray],
    after: np.ndarray,
    out_gif: Path,
    titles: List[str],
    fps: int = 20,
) -> Path:
    """One or two panels side by side; the window is shared so motion is comparable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    data = [d for d in (before, after) if d is not None]
    up_detect = data[0]
    hp = up_detect[:, 15] - up_detect[:, 0]
    up = 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1
    va, vb = (0, 1) if up == 2 else (0, 2)
    lo = np.minimum.reduce([d.min(axis=(0, 1)) for d in data])
    hi = np.maximum.reduce([d.max(axis=(0, 1)) for d in data])
    span = float((hi - lo).max()) * 1.05
    mid = (lo + hi) / 2.0
    n = len(titles)

    fig = plt.figure(figsize=(5.4 * n, 5.0), dpi=110)
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xlim(mid[va] - span / 2, mid[va] + span / 2)
        ax.set_ylim(mid[vb] - span / 2, mid[vb] + span / 2)
        ax.set_zlim(lo[up], hi[up])
        ax.view_init(elev=8, azim=-88)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1.6))

    lines = [[] for _ in range(n)]
    for k in range(n):
        for chain in SKELETON_CHAINS:
            (ln,) = axes[k].plot([], [], [], "o-", lw=2, markersize=3)
            lines[k].append(ln)

    frames = min(d.shape[0] for d in data)

    def update(t: int):
        for k, d in enumerate(data):
            j = d[min(t, d.shape[0] - 1)]
            for ln, chain in zip(lines[k], SKELETON_CHAINS):
                pts = j[chain]
                ln.set_data(pts[:, va], pts[:, vb])
                ln.set_3d_properties(pts[:, up])
        return [ln for row in lines for ln in row]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_gif


def analysis_card(
    out_png: Path,
    *,
    case_id: str,
    prompt: str,
    mode: str,
    plan: Dict[str, Any],
    cmeta: Dict[str, Any],
    quality: Dict[str, Any],
    cap: Dict[str, Any],
    timeline: Dict[str, Any],
    seconds: float,
) -> Path:
    """Dark card summarising what the agent decided, did, and measured."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    refine = plan.get("refine") or {}
    metrics = quality.get("metrics") or {}
    score = quality.get("score")
    passed = quality.get("passed")
    hard = quality.get("hard_failures") or []

    left: List[str] = []
    left += _wrap("提示词： " + prompt, 44)
    left.append("")
    left.append("— 计划 —")
    left += _wrap(f"action: {plan.get('primary_action')}", 44)
    left += _wrap(f"edit_mode: {plan.get('edit_mode')}  span: {plan.get('span')}", 44)
    left += _wrap(f"policy: {plan.get('source_policy')}", 44)
    left += _wrap("modules: " + ", ".join(plan.get("modules") or []), 44)
    left.append("")
    left.append("— 执行 —")
    ar = cmeta.get("arm_raise") if isinstance(cmeta.get("arm_raise"), dict) else None
    stages = cmeta.get("stages") if isinstance(cmeta.get("stages"), list) else None
    sides: List[Dict[str, Any]] = []
    if ar:
        if stages is None and "stages" in ar:
            stages = ar.get("stages")
            sides = [ar]
        elif isinstance(ar.get("both"), list) and ar["both"]:
            sides = [s for s in ar["both"] if isinstance(s, dict)]
    if stages:
        for s in stages:
            detail = str(s.get("detail") or "")
            if len(detail) > 58:
                detail = detail[:57] + "…"
            left += _wrap(f"{s.get('step')}. {s.get('name')}： {detail}", 46)
        for s in sides:
            left += _wrap(
                f"[{s.get('side')}] changed={s.get('changed')}  lift {s.get('wrist_lift_before')} → "
                f"{s.get('wrist_lift_after')}  ratio {s.get('ratio_after')}",
                46,
            )
    else:
        for key, label in (
            ("phase_speed_meta", "phase_speed"),
            ("phases", "phases"),
            ("dropped", "drop_phases"),
        ):
            if key in cmeta:
                left += _wrap(f"{label}: {json.dumps(cmeta[key], ensure_ascii=False)}", 46)
        if (cmeta.get("timeline") or {}).get("T_in") is not None:
            left += _wrap(
                f"T {cmeta.get('T_in')} → {cmeta.get('T_out')} (resampled)", 46
            )
        if len(left) and left[-1].startswith("— 执行 —"):
            left.append("(no geometric stage trace for this edit mode)")

    right: List[str] = []
    right.append("— 观测（M2T 描述）—")
    right += _wrap(str(cap.get("caption") or "—"), 46)
    right.append("")
    right.append("— 反馈 —")
    right += _wrap(str(cap.get("critique") or "—"), 46)
    right.append("")
    right.append("— 质量 —")
    right += _wrap(f"score={score}  passed={passed}  T={metrics.get('T')}", 46)
    if hard:
        right += _wrap("hard_failures: " + ", ".join(map(str, hard)), 46)
    for key in (
        "root_travel",
        "wrist_up_span",
        "arm_raise_lift",
        "arm_raise_ratio",
        "yaw_span_deg",
        "foot_slide_mean",
        "sit_pelvis_drop",
    ):
        if metrics.get(key) is not None:
            right += _wrap(f"{key}: {metrics[key]}", 46)
    right.append("")
    right.append("— 时间轴 —")
    phases = timeline.get("phases") or []
    if phases:
        right += _wrap(" → ".join(f"{p.get('id')}[{p.get('start')}-{p.get('end')}]" for p in phases), 46)
    else:
        right.append("—")
    right.append("")
    right += _wrap(f"耗时 {seconds:.1f}s   判定 {'通过' if passed else '未通过'}", 46)

    fig = plt.figure(figsize=(14.0, 8.8), dpi=130, facecolor="#101418")
    fig.text(
        0.035, 0.955, f"AI 分析 · {case_id}", color="#e8eef5", fontsize=19, weight="bold", va="top"
    )
    fig.text(
        0.035,
        0.912,
        f"mode={mode}    prompt=\"{prompt}\"    total={seconds:.1f}s",
        color="#7f8c99",
        fontsize=10.5,
        va="top",
    )
    badge = "#2ec27e" if passed else "#e5484d"
    fig.patches.append(
        plt.Rectangle((0.86, 0.925), 0.11, 0.045, transform=fig.transFigure,
                      facecolor=badge, alpha=0.85, zorder=1)
    )
    fig.text(
        0.915, 0.9475, "APPROVED" if passed else "REJECTED", color="#0b1014",
        fontsize=10, weight="bold", ha="center", va="center", zorder=2,
    )
    for x, lines in ((0.035, left), (0.515, right)):
        fig.text(x, 0.862, "\n".join(lines), color="#cfd8e3", fontsize=10.2,
                 va="top", family=list(rcParams.get("font.family") or []) + ["DejaVu Sans Mono"],
                 linespacing=1.62)
    fig.text(
        0.035, 0.035,
        "source: 00_motion_edit_plan.json / 03_edited.constraint.json / 04_quality.json / "
        "06_motion_caption.json / 05_timeline.json",
        color="#5a6673", fontsize=8.2,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def run_case(case: Dict[str, Any], out_root: Path, rerender: bool = False) -> Dict[str, Any]:
    py = sys.executable
    run_dir = out_root / case["id"].split()[0] / "run"
    prep_dir = out_root / case["id"].split()[0]
    cmd = [
        py,
        str(REPO_ROOT / "scripts" / "orchestrator" / "run_session.py"),
        "--mode",
        case["mode"],
        "--prompt",
        case["prompt"],
        "--run-dir",
        str(run_dir),
        "--skip-viz",
    ]
    base_run: Optional[Path] = None
    if case.get("base"):
        base_run = UI_RUNS / BASES[case["base"]]
        cmd += ["--previous-run", str(base_run)]
    t0 = time.time()
    proc = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    if not rerender:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    secs = time.time() - t0

    if rerender:
        prev = {}
        res_path = out_root / "results.json"
        if res_path.is_file():
            for row in _load(res_path).get("rows") or []:
                if row.get("id") == case["id"]:
                    prev = row
        secs = float(prev.get("seconds") or 0.0)

    plan = _load(run_dir / "00_motion_edit_plan.json")
    quality = _load(run_dir / "04_quality.json")
    cap = _load(run_dir / "06_motion_caption.json")
    timeline = _load(run_dir / "05_timeline.json")
    cmeta: Dict[str, Any] = _load(run_dir / "03_edited.constraint.json")
    if not cmeta:
        st = _load(run_dir / "03_edited.stages.json")
        if st:
            cmeta = st
    tp = _load(run_dir / "03_edited.timeline_post.json")
    if tp:
        cmeta = {**tp, **cmeta}

    edited = run_dir / "03_edited.npy"
    gifs: List[str] = []
    if edited.is_file():
        after, _k = load_joints(edited)
        titles = [f"after: {case['prompt']}"]
        before = None
        if base_run is not None and (base_run / "03_edited.npy").is_file():
            before, _k2 = load_joints(base_run / "03_edited.npy")
            titles = [f"before: {case['base']}", f"after: {case['prompt']}"]
        g = render_pair(before, after, prep_dir / "before_after.gif", titles)
        gifs.append(str(g.relative_to(REPO_ROOT)))
        g2 = render_skeleton_video(after, prep_dir / "result.gif", fps=20, title=case["prompt"])
        gifs.append(str(g2.relative_to(REPO_ROOT)))
    card = analysis_card(
        prep_dir / "ai_analysis.png",
        case_id=case["id"],
        prompt=case["prompt"],
        mode=case["mode"],
        plan=plan,
        cmeta=cmeta,
        quality=quality,
        cap=cap,
        timeline=timeline,
        seconds=secs,
    )

    metrics = quality.get("metrics") or {}
    row = {
        "id": case["id"],
        "mode": case["mode"],
        "base": case.get("base"),
        "prompt": case["prompt"],
        "ok": bool(quality.get("passed")) and proc.returncode == 0,
        "action": plan.get("primary_action"),
        "edit_mode": plan.get("edit_mode"),
        "score": quality.get("score"),
        "hard_failures": quality.get("hard_failures"),
        "T": metrics.get("T"),
        "root_travel": metrics.get("root_travel"),
        "arm_raise_ratio": metrics.get("arm_raise_ratio"),
        "wrist_up_span": metrics.get("wrist_up_span"),
        "caption": cap.get("caption"),
        "critique": cap.get("critique"),
        "seconds": round(secs, 1),
        "run_dir": str(run_dir),
        "gifs": gifs,
        "analysis": str(card.relative_to(REPO_ROOT)),
        "stderr_tail": (proc.stderr or "")[-500:],
    }
    return row


def write_report(rows: List[Dict[str, Any]], out_root: Path) -> Path:
    ok = [r for r in rows if r["ok"]]
    lines = ["# Batch action tests", ""]
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- passed: {len(ok)}/{len(rows)}")
    lines.append("")
    lines.append("| case | mode | action | edit_mode | score | T | ratio | seconds |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['mode']} | {r['action']} | {r['edit_mode']} | {r['score']} | "
            f"{r['T']} | {r['arm_raise_ratio']} | {r['seconds']} |"
        )
    lines.append("")
    for r in rows:
        lines.append(f"## {r['id']}")
        lines.append("")
        lines.append(f"- prompt: `{r['prompt']}`")
        lines.append(f"- M2T: {r['caption']}")
        lines.append(f"- critique: {r['critique']}")
        if r.get("hard_failures"):
            lines.append(f"- hard_failures: {r['hard_failures']}")
        lines.append(f"- run: `{r['run_dir']}`")
        for g in r.get("gifs") or []:
            gp = Path(g)
            rel = gp.relative_to(out_root) if gp.is_absolute() and out_root in gp.parents else gp
            lines.append(f"\n![{gp.stem}]({rel.as_posix()})")
        card = Path(r["analysis"])
        rel_card = card.relative_to(out_root) if card.is_absolute() and out_root in card.parents else card
        lines.append(f"\n![ai analysis]({rel_card.as_posix()})")
        lines.append("")
    out = out_root / "report.md"
    if (out_root / "ui").is_dir():
        lines.extend(UI_SECTION.splitlines())
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


UI_SECTION = """
---

## 网页回放截图（AI 分析，http://127.0.0.1:8787）

同一段会话内逐条发送，中途不刷新：先点 `turn` demo，再两条 revise；随后刷新开新段，generate `jump`，再一条 revise。
每一步的 run_dir / 分数 / 过程记录在 `outputs/session/prompt_logs/20261105-multi-action/log.json`。

| 步骤 | 提示词 | 模式 | 分数 | 关键过程 |
|---|---|---|---|---|
| 1 | a person turns around 180 degrees | generate | 100 | demo 片段 T=100，yaw 180° |
| 2 | raise the right arm high | revise | 100 | constraint 只处理 arm_raise，T 不变，yaw 仍 180° |
| 3 | slow down a bit | revise | 100 | T 100→232，speed=['turning']，yaw 仍 180° |
| 4 | a person jumps up and lands | generate | 100 | diffusion/dataset，T=52 |
| 5 | raise both arms overhead | revise | 100 | 双臂各抬一次（1.466 / 1.532），T 仍 52 |

### 1 → 2：转身片段 + 右臂抬高

![turn base](ui/01_turn_base_result.png)

![turn arm reasoning](ui/02_turn_arm_reasoning.png)

![turn arm result](ui/03_turn_arm_result.png)

### 2 → 3：同一片段再放慢

![turn slow reasoning](ui/04_turn_slow_reasoning.png)

![turn slow result](ui/05_turn_slow_result.png)

### 4：跳跃生成

![jump reasoning](ui/06_jump_reasoning.png)

![jump result](ui/07_jump_result.png)

### 4 → 5：跳跃片段抬双臂

![jump arms reasoning](ui/08_jump_arms_reasoning.png)

![jump arms result](ui/09_jump_arms_result.png)

## 本轮发现

1. **turn 片段抬臂会把转身清零（已修）**：revise 沿用了上一层的 `forbid=["yaw_spin"]`，`constraint_postprocess` 直接把 45.8° 的 yaw 抹成 0，随后 `turn_yaw_span` 硬失败（A4 首测 67 分）。`scripts/intent/refine_schema.py` 现在对 `keep_action` 以 `turn` 开头的 revise 丢掉 `yaw_spin`，A4/A10 回到 92 分且 yaw 保留。
2. **抬臂计划可能缺几何模块（已修）**：当片段本身没有 yaw 问题时，`constraint_postprocess` 不会进 modules，抬臂指令等于没生效。`scripts/intent/plan_motion_edit.py` 现在只要 `refine.arm_raise` 存在就强制加入该模块（第 5 步正是这条路径）。
3. **TM2T 描述与 critique 脱节（待定）**：A11 要「转身走回去」，实际 T=127、yaw 196°、travel 0.76 都对，但 TM2T 写成 "walk forward and wave"，critique 仍报 `Issues: n/a`（因为它只汇总质量检查项）。建议下一轮把规则版 caption 一并写进 critique，避免 pipeline v1 的 LLM 拿到"没问题"的错误输入。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma separated ids, e.g. A1,A4")
    ap.add_argument("--rerender", action="store_true", help="rebuild GIFs/cards from existing runs")
    args = ap.parse_args()
    out_root = (args.out or (REPO_ROOT / "outputs" / "tests" / f"actions_{time.strftime('%Y%m%d-%H%M%S')}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    font = _setup_fonts()
    print(f"[batch] cjk font: {font or 'none (labels may show boxes)'}", flush=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows: List[Dict[str, Any]] = []
    for case in CASES:
        if only and case["id"].split()[0] not in only:
            continue
        row = run_case(case, out_root, rerender=args.rerender)
        rows.append(row)
        print(f"[batch] {row['id']:<30} score={row['score']} ok={row['ok']} {row['seconds']}s", flush=True)
    if args.rerender:
        # A partial re-render must not drop the cases it did not touch.
        prev_rows = _load(out_root / "results.json").get("rows") or []
        by_id = {r.get("id"): r for r in prev_rows}
        for r in rows:
            by_id[r["id"]] = r
        order = [c["id"] for c in CASES]
        rows = sorted(by_id.values(), key=lambda r: order.index(r["id"]) if r["id"] in order else 99)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {"total": len(rows), "passed": sum(1 for r in rows if r["ok"])},
        "rows": rows,
    }
    (out_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rep = write_report(rows, out_root)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"[batch] report {rep}")


if __name__ == "__main__":
    main()
