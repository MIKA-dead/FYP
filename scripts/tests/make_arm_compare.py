"""Side-by-side before/after GIF for one arm-raise case (test deliverable)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "backends"))

import constraint_postprocess as cp  # noqa: E402
from visualize_motion import SKELETON_CHAINS, load_joints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", type=Path, required=True)
    ap.add_argument("--side", default="right", choices=["left", "right", "both"])
    ap.add_argument("--level", default="overhead", choices=["slightly", "mid", "high", "overhead"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    joints, _ = load_joints(args.npy)
    spec = {"side": args.side, "level": args.level}
    if args.side == "both":
        raised = joints
        for s in ("left", "right"):
            raised, _ = cp.raise_arm(raised, {**spec, "side": s})
        meta = {}
    else:
        raised, meta = cp.raise_arm(joints, spec)

    up = cp._up_axis(joints)
    va, vb = (0, 1) if up == 2 else (0, 2)
    lo = np.minimum(joints.min(axis=(0, 1)), raised.min(axis=(0, 1)))
    hi = np.maximum(joints.max(axis=(0, 1)), raised.max(axis=(0, 1)))
    span = float((hi - lo).max()) * 1.05
    mid = (lo + hi) / 2.0

    fig = plt.figure(figsize=(11, 5.2), dpi=110)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    titles = ["before", f"after: raise {args.side} arm ({args.level})"]

    def setup(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(mid[va] - span / 2, mid[va] + span / 2)
        ax.set_ylim(mid[vb] - span / 2, mid[vb] + span / 2)
        ax.set_zlim(lo[up], hi[up])
        ax.view_init(elev=8, azim=-88)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1.6))

    for ax, t in zip(axes, titles):
        setup(ax, t)

    lines = [[], []]
    for k in range(2):
        for chain in SKELETON_CHAINS:
            (ln,) = axes[k].plot([], [], [], "o-", lw=2, markersize=3)
            lines[k].append(ln)

    def update(t: int):
        for k, data in enumerate((joints, raised)):
            j = data[t]
            for ln, chain in zip(lines[k], SKELETON_CHAINS):
                pts = j[chain]
                ln.set_data(pts[:, va], pts[:, vb])
                ln.set_3d_properties(pts[:, up])
        return lines[0] + lines[1]

    fig.suptitle(
        "arm raise is applied to the same clip (root path, legs and the other arm unchanged)",
        fontsize=10,
    )
    anim = FuncAnimation(fig, update, frames=joints.shape[0], interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"[compare] {args.out}  meta={ {k: v for k, v in (meta or {}).items() if k != 'stages'} }")


if __name__ == "__main__":
    main()
