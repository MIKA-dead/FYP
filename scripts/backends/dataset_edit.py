"""
Revise by composing the current clip: one unit, then a continuous chain,
then a seam test, then a retry, then output.

The same stages apply to every action. A dataset clip in another skeleton
is not substituted.

  python scripts/backends/dataset_edit.py \\
      --in prev.npy --out 03_edited.npy --intent-json plan.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HML_ROOT = REPO_ROOT / "HumanML3D" / "HumanML3D"
INDEX_PATH = REPO_ROOT / "outputs" / "session" / "hml_text_index.json"

J_PELVIS = 0
J_HEAD = 15
J_L_FOOT, J_R_FOOT = 10, 11

_CONFLICT = {
    "stumble",
    "stumbles",
    "jump",
    "jumps",
    "run",
    "runs",
    "kick",
    "kicks",
    "sit",
    "sits",
    "turn",
    "turns",
    "dance",
    "dances",
    "crawl",
    "hop",
    "spin",
    "spins",
}

_STOP = {
    "a",
    "an",
    "the",
    "person",
    "man",
    "woman",
    "is",
    "and",
    "with",
    "his",
    "her",
    "their",
    "to",
    "of",
    "in",
    "on",
    "then",
    "while",
}


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def _load_joints(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        obj = raw.item()
        if isinstance(obj, dict) and "joint_positions" in obj:
            return dict(obj), np.asarray(obj["joint_positions"], dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 22:
        return {"joint_positions": arr}, arr
    raise SystemExit(f"No (T,22,3) joints in {path}")


def _save_joints(path: Path, obj: Dict[str, Any], joints: np.ndarray, meta: Dict[str, Any]) -> None:
    out = dict(obj)
    out["joint_positions"] = np.asarray(joints, dtype=np.float32)
    if "pose" in out:
        pose = np.asarray(out["pose"], dtype=np.float32).copy()
        t = min(pose.shape[0], joints.shape[0])
        pose = pose[: joints.shape[0]]
        pose[:t, :3] = joints[:t, J_PELVIS]
        out["pose"] = pose
    out["dataset_edit_meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out, allow_pickle=True)
    path.with_suffix(".dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _foot_strikes(joints: np.ndarray) -> List[int]:
    """Frames where either foot starts a ground contact."""
    up = _up_axis(joints)
    feet = joints[:, [J_L_FOOT, J_R_FOOT], up]
    strikes: List[int] = []
    for fi in range(2):
        z = feet[:, fi]
        thr = float(np.percentile(z, 25)) + 0.02
        down = z <= thr
        for t in range(1, len(down)):
            if down[t] and not down[t - 1]:
                strikes.append(t)
    strikes = sorted(strikes)
    merged: List[int] = []
    for t in strikes:
        if not merged or t - merged[-1] >= 5:
            merged.append(t)
    return merged


def _body_height(joints: np.ndarray) -> float:
    up = _up_axis(joints)
    return float(np.mean(joints[:, J_HEAD, up] - joints[:, J_PELVIS, up]))


def _root_speed(joints: np.ndarray) -> float:
    up = _up_axis(joints)
    hz = [0, 1] if up == 2 else [0, 2]
    root = joints[:, J_PELVIS]
    travel = float(np.linalg.norm(root[-1, hz] - root[0, hz]))
    return travel / max(1, joints.shape[0])


def align_to_source(joints: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Put a dataset clip into the same up-axis as the clip being edited.

    HumanML3D joints are Y-up. Edited MotionsFix clips are Z-up.
    (x, y, z) → (x, -z, y) matches the bridge used by mld_to_motionfix.
    """
    src_up = _up_axis(source)
    up = _up_axis(joints)
    if up == src_up:
        return joints, False
    if up == 1 and src_up == 2:
        out = np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)
        return out.astype(np.float32), True
    if up == 2 and src_up == 1:
        out = np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)
        return out.astype(np.float32), True
    return joints, False


def _compatible(
    candidate: np.ndarray,
    source: np.ndarray,
    *,
    check_speed: bool = True,
) -> Tuple[bool, str]:
    if _up_axis(candidate) != _up_axis(source):
        return False, "up_axis"
    h_c = _body_height(candidate)
    h_s = max(_body_height(source), 1e-4)
    if not (0.7 <= h_c / h_s <= 1.4):
        return False, f"height {h_c / h_s:.2f}"
    if check_speed:
        v_c = _root_speed(candidate)
        v_s = max(_root_speed(source), 1e-4)
        ratio = v_c / v_s
        if ratio > 1.6 or ratio < 0.55:
            return False, f"speed x{ratio:.2f}"
    return True, "ok"


def _step_count(joints: np.ndarray) -> int:
    return max(0, len(_foot_strikes(joints)) - 1)


def trim_to_steps(joints: np.ndarray, target_steps: int) -> np.ndarray:
    """Keep the opening of one continuous clip, ending on a foot strike."""
    strikes = _foot_strikes(joints)
    if len(strikes) - 1 >= target_steps:
        end = min(joints.shape[0], strikes[target_steps] + 1)
        return joints[:end].copy()
    return joints.copy()


def _tokens(text: str) -> set:
    words = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {w for w in words if w not in _STOP and len(w) > 1}


def _caption_index() -> List[Dict[str, str]]:
    if INDEX_PATH.is_file():
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:  # noqa: BLE001
            pass
    texts = HML_ROOT / "texts"
    if not texts.is_dir():
        raise SystemExit(f"HumanML3D texts not found: {texts}")
    rows: List[Dict[str, str]] = []
    for path in sorted(texts.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not raw:
            continue
        caption = raw[0].split("#", 1)[0].strip()
        if caption:
            rows.append({"id": path.stem, "caption": caption})
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _rank_captions(query: str, limit: int = 40) -> List[Dict[str, Any]]:
    q = _tokens(query)
    if not q:
        q = {"walk", "forward"}
    ranked: List[Dict[str, Any]] = []
    for row in _caption_index():
        cap = row["caption"]
        toks = _tokens(cap)
        if not toks:
            continue
        overlap = len(q & toks)
        if overlap <= 0:
            continue
        score = overlap / len(q)
        score -= 0.02 * max(0, len(toks) - len(q))
        ranked.append({"id": row["id"], "caption": cap, "score": score})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:limit]


def _load_hml(motion_id: str) -> np.ndarray:
    npy = HML_ROOT / "new_joints" / f"{motion_id}.npy"
    joints = np.asarray(np.load(npy), dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] != 22:
        raise SystemExit(f"Unexpected joints shape {joints.shape} in {npy}")
    return joints


def retrieve_continuous(
    query: str,
    target_steps: Optional[int],
    source: np.ndarray,
    *,
    check_speed: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Pick one recorded clip in the same frame and speed band as source.

    A text hit that is Y-up, a different body scale, or several times faster
    is not a substitute for the clip on screen.
    """
    ranked = _rank_captions(query)
    if not ranked:
        return None, {"reason": "no_caption", "query": query}
    best_joints = None
    best_info: Dict[str, Any] = {}
    best_key = None
    rejected = 0
    for row in ranked:
        if best_key is not None and row["score"] < 0.45:
            break
        raw = _load_hml(row["id"])
        joints, converted = align_to_source(raw, source)
        extra = (_tokens(row["caption"]) & _CONFLICT) - _tokens(query)
        if extra:
            rejected += 1
            continue
        ok, why = _compatible(joints, source, check_speed=check_speed)
        if not ok:
            rejected += 1
            continue
        steps = _step_count(joints)
        if target_steps:
            short = max(0, target_steps - steps)
            key = (0 if steps >= target_steps else 1, short, abs(steps - target_steps), -row["score"])
        else:
            key = (0, 0, 0, -row["score"])
        if best_key is None or key < best_key:
            best_key = key
            best_joints = joints
            best_info = {
                "id": row["id"],
                "caption": row["caption"],
                "score": round(float(row["score"]), 4),
                "query": query,
                "steps_in_clip": steps,
                "axis_converted": converted,
            }
    if best_joints is None:
        return None, {
            "reason": "incompatible",
            "query": query,
            "rejected_candidates": rejected,
        }
    edited = best_joints
    if target_steps and best_info.get("steps_in_clip", 0) >= target_steps:
        edited = trim_to_steps(best_joints, target_steps)
        best_info["trimmed_to_steps"] = target_steps
    best_info["steps_out"] = _step_count(edited)
    return edited, best_info


def edit_from_dataset(
    source: Path,
    dest: Path,
    *,
    cycle_count: Optional[int] = None,
    repeat_more: bool = False,
    retrieval_text: str = "",
    forbid: Optional[List[str]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from motion_assemble import assemble_motion

    _, joints = _load_joints(source)
    segments = list(segments or [])
    if not segments:
        segments = [
            {
                "action": "walk",
                "count": cycle_count,
                "repeat_more": bool(repeat_more and cycle_count is None),
                "constraints": list(forbid or []),
                "relation": "extend",
                "reference": "current",
                "retrieval_text": retrieval_text,
            }
        ]
    edited, info = assemble_motion(joints, segments)
    meta: Dict[str, Any] = {
        "source": str(source),
        "cycle_count": cycle_count,
        "repeat_more": repeat_more,
        "retrieval_text": retrieval_text,
        **info,
    }
    meta["T_out"] = int(edited.shape[0])
    _save_joints(dest, {"joint_positions": edited}, edited, meta)
    print(
        f"[dataset_edit] method={meta.get('method')} "
        f"T={meta['T_out']} seam={meta.get('seam_ratio')}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose motion units from the current clip")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--intent-json", type=Path, default=None)
    ap.add_argument("--cycle-count", type=int, default=None)
    ap.add_argument("--retrieval-text", type=str, default="")
    ap.add_argument("--repeat-more", action="store_true")
    args = ap.parse_args()
    cycle = args.cycle_count
    text = args.retrieval_text
    more = bool(args.repeat_more)
    forbid: List[str] = []
    segments = None
    if args.intent_json and args.intent_json.is_file():
        plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        refine = plan.get("refine") or {}
        if refine.get("cycle_count") is not None:
            cycle = int(refine["cycle_count"])
        if refine.get("retrieval_text"):
            text = str(refine["retrieval_text"])
        more = more or bool(refine.get("repeat_more"))
        forbid = [str(x) for x in (refine.get("forbid") or [])]
        if isinstance(refine.get("segments"), list):
            segments = refine["segments"]
    edit_from_dataset(
        args.inp,
        args.out,
        cycle_count=cycle,
        repeat_more=more,
        retrieval_text=text,
        forbid=forbid,
        segments=segments,
    )


if __name__ == "__main__":
    main()
