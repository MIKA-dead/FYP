"""Build a clickable gallery: goal -> draft -> each edit -> final.

  python scripts/agent/build_motion_gallery.py --report outputs/agent_bench/_v3_final10/report.json --out outputs/agent_gallery

Every clip is the per-step preview the benchmark already wrote, copied next to
the page, so the page opens from the file system with no server. Next to each
clip the page prints the measurement that step was judged on, which is what
makes a failure visible: watch the draft, then read what the checker said.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEY = [
    ("root_travel", "travel", 1),
    ("pelvis_drop", "sit depth", 3),
    ("yaw_net_deg", "turn", 0),
    ("arm_raise_right", "arm R", 2),
    ("arm_raise_left", "arm L", 2),
    ("ankle_lift_right", "ankle R", 2),
    ("jump_rise", "jump", 2),
    ("ground_penetration", "floor", 4),
]


def step_metrics(obs):
    m = obs.get("metrics") or {}
    out = []
    for key, label, nd in KEY:
        if key in m:
            out.append([label, "{:.{}f}".format(float(m[key]), nd)])
    return out[:5]


def build(report_path, out, template):
    rep = json.loads(report_path.read_text(encoding="utf-8-sig"))
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    goals = []
    for i, row in enumerate(rep["loop"]["rows"], 1):
        ep_dir = Path(str(row.get("episode_dir") or ""))
        run_f = ep_dir / "agent_run.json"
        if not run_f.is_file():
            continue
        ep = json.loads(run_f.read_text(encoding="utf-8-sig"))
        steps = []
        for s in ep["steps"]:
            obs = s.get("observation") or {}
            pat = "step{:02d}_*/03_edited_preview.gif".format(s["step"])
            cands = sorted(ep_dir.glob(pat))
            rel = ""
            if cands:
                name = "{:02d}_{:02d}_{}.gif".format(i, s["step"], s["action"])
                shutil.copyfile(cands[0], clips / name)
                rel = "clips/" + name
            req = obs.get("requirement") or {}
            steps.append({
                "step": s["step"],
                "action": s["action"],
                "thought": (s.get("thought") or "").strip(),
                "instruction": obs.get("instruction") or "",
                "gif": rel,
                "metrics": step_metrics(obs),
                "gap": req.get("gap"),
                "caption": obs.get("caption") or "",
                "verdict": (obs.get("caption_verdict") or {}).get("level") or "",
                "similarity": obs.get("similarity") or {},
            })
        goals.append({
            "n": i,
            "goal": row.get("goal") or ep.get("goal"),
            "satisfied": bool(row.get("satisfied")),
            "edits": int(row.get("edits") or 0),
            "resamples": int(row.get("resamples") or 0),
            "restores": int(row.get("restores") or 0),
            "steps": steps,
            "caption": row.get("final_caption") or "",
            "gap": row.get("final_gap"),
            "seconds": row.get("seconds"),
        })
    n_ok = sum(g["satisfied"] for g in goals)
    page = template.read_text(encoding="utf-8")
    page = page.replace("__PAYLOAD__", json.dumps(goals, ensure_ascii=False))
    page = page.replace("__NAME__", html.escape(report_path.parent.name))
    page = page.replace("__OK__", str(n_ok)).replace("__TOTAL__", str(len(goals)))
    p = out / "index.html"
    p.write_text(page, encoding="utf-8")
    print("[gallery] {}  ({} goals, {} clips)".format(
        p, len(goals), len(list(clips.glob("*.gif")))))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=REPO / "outputs/agent_bench/_v3_final10/report.json")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/agent_gallery")
    ap.add_argument("--template", type=Path, default=REPO / "docs/gallery_template.html")
    args = ap.parse_args()
    build(args.report.resolve(), args.out.resolve(), args.template.resolve())


if __name__ == "__main__":
    main()
