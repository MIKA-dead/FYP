"""Static contact sheet: one mid-clip pose per tested action, for eyeballing.

  python scripts/tests/action_contact_sheet.py --root outputs/tests/actions_20261105
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends"))
from visualize_motion import load_joints  # noqa: E402

SKELETON_CHAINS = [
    [0, 1, 2, 3, 4],
    [1, 5, 6, 7, 8],
    [1, 9, 10, 11, 12],
    [9, 13, 14, 15],
    [9, 16, 17, 18],
]

LABELS = {
    "A1": "jump",
    "A2": "kick",
    "A3": "pick up",
    "A4": "turn + right arm",
    "A5": "bow + both arms",
    "A6": "sit + right arm",
    "A7": "wave + both arms",
    "A8": "walk + left arm",
    "A9": "run + slower",
    "A10": "turn + slower",
    "A11": "walk → turn back",
}


def _font() -> list:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in have:
            rcParams["font.family"] = [name]
            return [name]
    return ["DejaVu Sans"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    fam = _font()
    rows = json.loads((root / "results.json").read_text(encoding="utf-8"))["rows"]

    n = len(rows)
    cols = 4
    rws = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.0 * cols, 4.3 * rws), facecolor="#101418")
    for i, r in enumerate(rows):
        key = r["id"].split()[0]
        npy = Path(r["run_dir"]) / "03_edited.npy"
        if not npy.is_file():
            continue
        j, _fps = load_joints(npy)
        ax = fig.add_subplot(rws, cols, i + 1, projection="3d")
        ax.set_facecolor("#101418")
        t = int(len(j) * 0.62)
        pose = j[min(t, len(j) - 1)]
        for chain in SKELETON_CHAINS:
            p = pose[chain]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], "o-", lw=2.0, markersize=2.6, color="#57b3ff")
        lo = j.min(axis=(0, 1))
        hi = j.max(axis=(0, 1))
        span = float((hi - lo).max()) * 0.62
        mid = (lo + hi) / 2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(lo[2] - 0.05, hi[2] + 0.05)
        ax.view_init(elev=6, azim=-88)
        ax.set_box_aspect((1, 1, 1.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ok = r.get("ok")
        ax.set_title(
            f"{key} {LABELS.get(key, '')}\nscore {r.get('score')} · T={r.get('T')} · {r.get('seconds')}s",
            color="#e8eef5" if ok else "#ff8a8a",
            fontsize=10,
            family=fam,
        )
    fig.suptitle(
        "Batch actions · mid-clip pose (frame ≈62%)",
        color="#e8eef5",
        fontsize=17,
        family=fam,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = args.out or (root / "contact_sheet.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
