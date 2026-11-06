"""Build a folder that is safe to hand to someone else.

  python scripts/tests/package_share.py --src outputs/tests/actions_20261105

The battery output is a working directory: it keeps `.npy` clips, every run's
JSON, and two GIFs per case (a single view and a before/after pair), so it is
~27 MB and the filenames (`A1`, `A4`, ...) mean nothing without `results.json`.
This copies out one labelled GIF per case, the analysis cards, the two summary
images and a handful of page screenshots, plus a README and a gallery page, then
zips it.

Chooses the before/after GIF when the case really changed an existing clip and
the single view when the case only generated a new one, because a one-panel
"before/after" is just a duplicate of the single view.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

# case id -> (filename slug, one-line label, what to show)
CASES: Dict[str, Dict[str, str]] = {
    "A1": {"slug": "01_jump_generate", "label": "jump (new action, generated)", "gif": "result"},
    "A2": {"slug": "02_kick_generate", "label": "kick (new action, generated)", "gif": "result"},
    "A3": {"slug": "03_pickup_generate", "label": "pick up (new action, generated)", "gif": "result"},
    "A4": {"slug": "04_turn_plus_right_arm_raised", "label": "turn + right arm raised", "gif": "before_after"},
    "A5": {"slug": "05_bow_plus_both_arms_overhead", "label": "bow + both arms overhead", "gif": "before_after"},
    "A6": {"slug": "06_sit_plus_right_arm_raised", "label": "sit + right arm raised", "gif": "before_after"},
    "A7": {"slug": "07_wave_plus_both_arms_overhead", "label": "wave + both arms overhead", "gif": "before_after"},
    "A8": {"slug": "08_walk_plus_left_arm_raised", "label": "walk + left arm raised", "gif": "before_after"},
    "A9": {"slug": "09_run_slowed_down", "label": "run + slower", "gif": "before_after"},
    "A10": {"slug": "10_turn_slowed_down", "label": "turn + slower", "gif": "before_after"},
    "A11": {"slug": "11_walk_then_turn_and_walk_back", "label": "walk → turn around and walk back", "gif": "before_after"},
}

UI_SHOTS = [
    ("ui/03_turn_arm_result.png", "page_turn_arm_result.png", "page: turn clip, right arm raised (quality card + timeline)"),
    ("ui/02_turn_arm_reasoning.png", "page_turn_arm_reasoning.png", "page: reasoning panel for the turn + arm raise"),
    ("ui/08_jump_arms_reasoning.png", "page_jump_arms_reasoning.png", "page: both arms overhead on the jump clip"),
]


def _load(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}


def build(src: Path, out: Path) -> Path:
    rows = {r["id"].split()[0]: r for r in _load(src / "results.json").get("rows") or []}
    if not rows:
        raise SystemExit(f"no results.json in {src}")

    if out.exists():
        shutil.rmtree(out)
    (out / "gifs").mkdir(parents=True)
    (out / "analysis").mkdir()
    (out / "page_screenshots").mkdir()

    entries: List[Dict[str, Any]] = []
    for cid, spec in CASES.items():
        row = rows.get(cid)
        if not row:
            continue
        case_dir = src / cid
        name = spec["gif"]

        if name == "before_after" and (case_dir / "before_after.gif").is_file():
            gif_src = case_dir / "before_after.gif"
            kind = "before | after"
        else:
            gif_src = case_dir / "result.gif"
            kind = "after"
            name = "result"

        gif_out = out / "gifs" / f"{spec['slug']}.gif"
        if gif_src.is_file():
            shutil.copy2(gif_src, gif_out)

        ana_src = case_dir / "ai_analysis.png"
        ana_out = out / "analysis" / f"{spec['slug']}.png"
        if ana_src.is_file():
            shutil.copy2(ana_src, ana_out)

        entries.append({
            "id": cid,
            "slug": spec["slug"],
            "label": spec["label"],
            "prompt": row.get("prompt"),
            "mode": row.get("mode"),
            "score": row.get("score"),
            "T": row.get("T"),
            "ratio": row.get("arm_raise_ratio"),
            "seconds": row.get("seconds"),
            "kind": kind,
            "gif": f"gifs/{spec['slug']}.gif" if gif_src.is_file() else None,
            "analysis": f"analysis/{spec['slug']}.png" if ana_src.is_file() else None,
            "caption": row.get("caption"),
        })

    for rel_src, rel_dst, caption in UI_SHOTS:
        s = src / rel_src
        if s.is_file():
            shutil.copy2(s, out / "page_screenshots" / rel_dst)

    extras = []
    for rel, label in (
        ("contact_sheet.png", "one pose from each of the eleven cases"),
        ("arm_peak_check.png", "peak wrist frame: bow, wave, walk, turn"),
    ):
        s = src / rel
        if s.is_file():
            shutil.copy2(s, out / rel)
            extras.append({"file": rel, "label": label})

    (out / "README.md").write_text(_readme(entries, extras), encoding="utf-8")
    (out / "index.html").write_text(_gallery(entries, extras), encoding="utf-8")
    return out


def _readme(entries: List[Dict[str, Any]], extras: List[Dict[str, str]]) -> str:
    lines = [
        "# Motion Studio — action test clips, 5 November 2026",
        "",
        "Every clip below came out of the same pipeline the page runs: generate a",
        "draft from the dataset, then edit only the fault the observation names.",
        "The right-hand panel of a before/after GIF is the edited clip.",
        "",
        "`score` is the quality gate's 0-100. A 92 is not a failure: it means one",
        "*soft* check missed while every hard check passed (see each row's note).",
        "",
        "| # | case | mode | score | frames | arm ratio | file |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        ratio = e["ratio"] if e["ratio"] is not None else "—"
        lines.append(
            f"| {e['id']} | {e['label']} | {e['mode']} | {e['score']} | {e['T']} | "
            f"{ratio} | `{e['gif']}` |"
        )
    lines += [
        "",
        "## What to look at first",
        "",
        "1. `04_turn_plus_right_arm_raised.gif` — the arm is added to a clip that is",
        "   already rotating. The 45.8° turn survives, which it did not before the fix.",
        "2. `06_sit_plus_right_arm_raised.gif` — the sit descent is untouched; only the",
        "   elbow and wrist move.",
        "3. `11_walk_then_turn_and_walk_back.gif` — the one case where the body plan",
        "   itself changes: an existing walk, then a turn joined onto it.",
        "",
        "## Notes on the two 92s",
        "",
        "- `04` / `10` (`turn_yaw_large`): the source clip only turns 45.8°, so the",
        "  \"want > 90° for a full about-face\" soft check is honestly missed. The point",
        "  of the test is that the 45.8° is preserved through the edit.",
        "- `07` (`wave_one_arm`): the check wants the non-waving arm to stay down. We",
        "  asked for both arms overhead, so it is correct to miss it.",
        "",
        "## Also in here",
        "",
    ]
    for ex in extras:
        lines.append(f"- `{ex['file']}` — {ex['label']}")
    lines += [
        "- `analysis/<case>.png` — the per-case card: intent, module plan, the",
        "  geometric stages, the M2T description, the critique, the metrics.",
        "- `page_screenshots/` — the live page, same clips through the chat UI.",
        "",
        "`index.html` shows the whole set in a browser; open it if the GIFs are hard",
        "to find in a file list.",
        "",
    ]
    return "\n".join(lines)


def _gallery(entries: List[Dict[str, Any]], extras: List[Dict[str, str]]) -> str:
    cards = []
    for e in entries:
        badge = "#2ec27e" if (e["score"] or 0) >= 100 else "#c9a227"
        ana = (
            f'<a class="ana" href="{e["analysis"]}">analysis card</a>' if e["analysis"] else ""
        )
        media = (
            f'<img src="{e["gif"]}" alt="{e["label"]}" loading="lazy">'
            if e["gif"] else '<div class="missing">no gif</div>'
        )
        ratio = f'<span>arm {e["ratio"]}</span>' if e["ratio"] is not None else ""
        cards.append(f"""
    <article class="card">
      <header>
        <span class="id">{e['id']}</span>
        <span class="score" style="background:{badge}">{e['score']}</span>
      </header>
      <h2>{e['label']}</h2>
      <p class="prompt">{e['prompt']}</p>
      {media}
      <footer>
        <span>{e['mode']}</span><span>frames {e['T']}</span>{ratio}
        <span>{e['seconds']}s</span>{ana}
      </footer>
    </article>""")

    extra_html = "".join(
        f'<figure><img src="{ex["file"]}" alt="{ex["label"]}" loading="lazy">'
        f'<figcaption>{ex["label"]}</figcaption></figure>'
        for ex in extras
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Motion Studio — action test clips</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 40px 32px 64px; background: #0e1216; color: #dbe4ee;
         font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -.01em; }}
  .sub {{ color: #7d8b99; margin: 0 0 32px; max-width: 70ch; }}
  .grid {{ display: grid; gap: 22px; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }}
  .card {{ background: #161c22; border: 1px solid #232c35; border-radius: 10px;
           padding: 14px 14px 12px; display: flex; flex-direction: column; }}
  .card header {{ display: flex; justify-content: space-between; align-items: center; }}
  .id {{ font-variant: tabular-nums; color: #5f6f7d; font-size: 12px; letter-spacing: .08em; }}
  .score {{ color: #0b1014; font-weight: 700; font-size: 12px; border-radius: 999px;
            padding: 2px 9px; }}
  .card h2 {{ font-size: 15px; margin: 8px 0 2px; font-weight: 600; }}
  .prompt {{ margin: 0 0 10px; color: #8b99a7; font-size: 12.5px; font-family: ui-monospace, Consolas, monospace; }}
  img {{ width: 100%; height: auto; display: block; border-radius: 6px; background: #0b0f13; }}
  footer {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 12px; color: #7d8b99; }}
  .ana {{ margin-left: auto; color: #57b3ff; text-decoration: none; }}
  .ana:hover {{ text-decoration: underline; }}
  .missing {{ padding: 60px 0; text-align: center; color: #5f6f7d; font-size: 13px; }}
  h2.sec {{ font-size: 18px; margin: 48px 0 16px; }}
  .row {{ display: grid; gap: 22px; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }}
  figure {{ margin: 0; }}
  figcaption {{ color: #7d8b99; font-size: 12px; margin-top: 6px; }}
</style>
<h1>Motion Studio — action test clips</h1>
<p class="sub">5 November 2026 · 11 cases from the same pipeline the page runs. In a
before/after GIF the right panel is the edited clip. A score of 92 means one soft
check missed while every hard check passed.</p>
<div class="grid">{''.join(cards)}
</div>
<h2 class="sec">Summary images</h2>
<div class="row">{extra_html}</div>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--zip", action="store_true", help="also write a .zip next to the folder")
    args = ap.parse_args()
    src = args.src.resolve()
    out = (args.out or (REPO_ROOT / "outputs" / "share" / f"{src.name}_share")).resolve()
    build(src, out)

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n_gif = len(list((out / "gifs").glob("*.gif")))
    print(f"{out}\n  {n_gif} gifs, {total / 1024 / 1024:.1f} MB")
    if args.zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out.parent))
        print(f"  {zp} ({zp.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
