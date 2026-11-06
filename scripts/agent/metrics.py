"""
Quantitative metrics for the agentic text→motion pipeline.

Four groups, all computed from artifacts the loop already writes:

  (a) generation  text–motion alignment, caption-mediated retrieval,
                  kinematic plausibility, batch diversity
  (b) editing     instruction following, target change, non-target
                  preservation (DTW-aligned MPJPE), bone-length drift,
                  smoothness ratio, semantic gain
  (c) captioning  BLEU-1..4, ROUGE-L, CIDEr-D against HumanML3D references,
                  plus GloVe embedding similarity
  (loop)          success rate, iterations to success, tool calls, wall time

The joint-space metrics need only ``(T, 22, 3)`` positions. Text metrics use
the GloVe table TM2T already ships (``third_party/TM2T/glove``), so no extra
download is required.

  python scripts/agent/metrics.py --joints outputs/.../03_edited.npy
  python scripts/agent/metrics.py --caption-bench --n 8
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HML_ROOT = REPO / "HumanML3D" / "HumanML3D"
GLOVE_DIR = REPO / "third_party" / "TM2T" / "glove"

# SMPL-22 / HumanML3D joint layout.
J_PELVIS, J_HEAD = 0, 15
J_L_FOOT, J_R_FOOT = 10, 11
J_L_WRIST, J_R_WRIST = 20, 21
J_L_SHOULDER, J_R_SHOULDER = 16, 17

BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
)

RIGHT_ARM = (17, 19, 21)
LEFT_ARM = (16, 18, 20)
LEGS = (1, 2, 4, 5, 7, 8, 10, 11)
LOWER_BODY = (0, 1, 2, 4, 5, 7, 8, 10, 11)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_joints(path: Path | str) -> np.ndarray:
    raw = np.load(Path(path), allow_pickle=True)
    if raw.dtype == object:
        obj = raw.item() if raw.shape == () else raw
        if isinstance(obj, dict):
            for key in ("joint_positions", "joints", "pose_joints"):
                if obj.get(key) is not None:
                    raw = np.asarray(obj[key])
                    break
            else:
                raise ValueError(f"no joint_positions in {path}")
    j = np.asarray(raw, dtype=np.float32)
    if j.ndim == 2:
        j = j.reshape(j.shape[0], -1, 3)
    if j.ndim != 3 or j.shape[-1] != 3:
        raise ValueError(f"expected (T,J,3), got {j.shape}")
    return j


def up_axis(j: np.ndarray) -> int:
    hp = j[:, J_HEAD] - j[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def horiz(up: int) -> List[int]:
    return [0, 1] if up == 2 else [0, 2]


def _height(j: np.ndarray, up: int) -> float:
    return float(np.linalg.norm(j[:, J_HEAD] - j[:, J_PELVIS], axis=-1).mean()) or 1.0


# --------------------------------------------------------------------------
# (a) kinematic plausibility
# --------------------------------------------------------------------------
def bone_cv(j: np.ndarray) -> float:
    """Mean coefficient of variation of bone lengths over time (0 = rigid)."""
    cvs = []
    for a, b in BONES:
        L = np.linalg.norm(j[:, a] - j[:, b], axis=-1)
        m = float(L.mean())
        if m > 1e-6:
            cvs.append(float(L.std()) / m)
    return float(np.mean(cvs)) if cvs else 0.0


def foot_slide(j: np.ndarray, up: int) -> float:
    """Mean horizontal foot displacement (m/frame) while the foot is on the floor."""
    hz = horiz(up)
    h = _height(j, up)
    out = []
    for idx in (J_L_FOOT, J_R_FOOT):
        p = j[:, idx]
        y = p[:, up]
        contact = y <= float(y.min()) + 0.02 * h
        if contact.sum() < 2:
            continue
        d = np.linalg.norm(np.diff(p[:, hz], axis=0), axis=-1)
        out.append(float(d[contact[1:]].mean()) if contact[1:].any() else 0.0)
    return float(np.mean(out)) if out else 0.0


def jerk(j: np.ndarray, up: int) -> float:
    """Mean |third difference| of joint positions, normalised by body height."""
    if j.shape[0] < 4:
        return 0.0
    acc = j[2:] - 2 * j[1:-1] + j[:-2]
    jrk = acc[1:] - acc[:-1]
    return float(np.linalg.norm(jrk, axis=-1).mean() / _height(j, up))


def ground_penetration(j: np.ndarray, up: int) -> float:
    h = _height(j, up)
    return float(max(0.0, -(j[:, :, up].min())) / h)


def arm_raise_ratio(j: np.ndarray, up: int, side: str = "right") -> float:
    """Wrist height above the shoulder in head-above-shoulder units.

    1.0 = wrist level with the head, 1.8 = clearly above the head,
    2.6 = arm straight up. The denominator is floored at a quarter of the
    pelvis→head height so a raised shoulder cannot inflate the ratio.
    """
    sh = J_R_SHOULDER if side == "right" else J_L_SHOULDER
    wr = J_R_WRIST if side == "right" else J_L_WRIST
    denom = float((j[:, J_HEAD, up] - j[:, sh, up]).mean())
    body = float((j[:, J_HEAD, up] - j[:, J_PELVIS, up]).mean())
    denom = max(denom, 0.25 * abs(body), 1e-6)
    return float((j[:, wr, up] - j[:, sh, up]).max() / denom)


def wrist_above_head(j: np.ndarray, up: int, side: str = "right") -> float:
    """Metres by which the highest wrist point clears the head (can be negative)."""
    wr = J_R_WRIST if side == "right" else J_L_WRIST
    return float((j[:, wr, up] - j[:, J_HEAD, up]).max())


def yaw_span(j: np.ndarray) -> float:
    v = j[:, J_R_SHOULDER] - j[:, J_L_SHOULDER]
    yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return float(np.degrees(yaw.max() - yaw.min()))


def ankle_lift(j: np.ndarray, up: int, side: str = "right") -> float:
    """How far that ankle rises off its own lowest point, in body heights.

    A kick lifts the striking foot well clear of the ground while the supporting
    foot stays down, so this separates it from walking (both feet alternate).
    """
    idx = J_R_FOOT if side == "right" else J_L_FOOT
    ankle = j[:, idx, up]
    return float((ankle - ankle.min()).max() / _height(j, up))


def jump_rise(j: np.ndarray, up: int) -> Tuple[float, float]:
    """``(pelvis_rise, apex_clearance)`` in body heights.

    A jump raises the pelvis above its starting height *and* leaves the floor at
    that apex with both feet — standing up also raises the pelvis, but keeps the
    feet planted, so the pair separates the two.
    """
    h = _height(j, up)
    pel = j[:, J_PELVIS, up]
    floor = j[:, :, up].min(axis=1)
    apex = int(np.argmax(pel))
    rise = (float(pel.max()) - float(pel[0])) / h
    clearance = (float(floor[apex]) - float(np.percentile(floor, 5))) / h
    return float(rise), float(clearance)


def kinematic_summary(j: np.ndarray) -> Dict[str, float]:
    up = up_axis(j)
    hz = horiz(up)
    root = j[:, J_PELVIS]
    travel = float(np.linalg.norm(root[-1, hz] - root[0, hz]))
    h = _height(j, up)
    return {
        "T": int(j.shape[0]),
        "up_axis": int(up),
        "height": round(h, 4),
        "root_travel": round(travel, 4),
        "pelvis_drop": round(float(root[0, up] - root[:, up].min()), 4),
        "yaw_span_deg": round(yaw_span(j), 2),
        "foot_slide": round(foot_slide(j, up), 5),
        "ground_penetration": round(ground_penetration(j, up), 5),
        "bone_cv": round(bone_cv(j), 5),
        "jerk": round(jerk(j, up), 4),
        "arm_raise_right": round(arm_raise_ratio(j, up, "right"), 3),
        "arm_raise_left": round(arm_raise_ratio(j, up, "left"), 3),
        "wrist_above_head_right": round(wrist_above_head(j, up, "right"), 3),
        "wrist_above_head_left": round(wrist_above_head(j, up, "left"), 3),
        "ankle_lift_right": round(ankle_lift(j, up, "right"), 3),
        "ankle_lift_left": round(ankle_lift(j, up, "left"), 3),
        "jump_rise": round(jump_rise(j, up)[0], 3),
        "jump_clearance": round(jump_rise(j, up)[1], 3),
    }


# --------------------------------------------------------------------------
# (b) editing: target change / non-target preservation
# --------------------------------------------------------------------------
def _root_relative(j: np.ndarray) -> np.ndarray:
    return j - j[:, J_PELVIS:J_PELVIS + 1]


def _dtw_path(a: np.ndarray, b: np.ndarray) -> List[Tuple[int, int]]:
    """DTW over frames with mean squared joint distance as the local cost."""
    n, m = a.shape[0], b.shape[0]
    cost = np.full((n, m), np.inf, dtype=np.float64)
    acc = np.full((n, m), np.inf, dtype=np.float64)
    for i in range(n):
        for k in range(m):
            cost[i, k] = float(np.mean(np.sum((a[i] - b[k]) ** 2, axis=-1)))
    acc[0, 0] = cost[0, 0]
    for i in range(1, n):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for k in range(1, m):
        acc[0, k] = acc[0, k - 1] + cost[0, k]
    for i in range(1, n):
        for k in range(1, m):
            acc[i, k] = cost[i, k] + min(acc[i - 1, k], acc[i, k - 1], acc[i - 1, k - 1])
    path = []
    i, k = n - 1, m - 1
    while i > 0 or k > 0:
        path.append((i, k))
        if i == 0:
            k -= 1
        elif k == 0:
            i -= 1
        else:
            prev = min((acc[i - 1, k - 1], (i - 1, k - 1)),
                       (acc[i - 1, k], (i - 1, k)),
                       (acc[i, k - 1], (i, k - 1)))
            i, k = prev[1]
    path.append((0, 0))
    return path[::-1]


def infer_target_joints(instruction: str) -> Tuple[Tuple[int, ...], str]:
    t = (instruction or "").lower()
    if any(w in t for w in ("right arm", "right hand", "right wrist")):
        return RIGHT_ARM, "right_arm"
    if any(w in t for w in ("left arm", "left hand", "left wrist")):
        return LEFT_ARM, "left_arm"
    if any(w in t for w in ("both arm", "arms", "waving", "wave")):
        return LEFT_ARM + RIGHT_ARM, "arms"
    if any(w in t for w in ("leg", "kick", "step", "knee", "foot")):
        return LEGS, "legs"
    if any(w in t for w in ("sit", "hip", "crouch", "squat")):
        return LOWER_BODY, "lower_body"
    return tuple(range(1, 22)), "whole_body"


def preservation(before: np.ndarray, after: np.ndarray, target: Sequence[int]) -> Dict[str, Any]:
    """DTW-aligned pose error, split into edited and untouched joints.

    Both clips are root-relative, so a global translation never counts as a
    pose change. The DTW path absorbs a speed change.
    """
    up = up_axis(before)
    h = _height(before, up)
    a = _root_relative(before)
    b = _root_relative(after)
    path = _dtw_path(a, b)
    all_j = set(range(a.shape[1]))
    tgt = sorted(set(int(x) for x in target))
    rest = sorted(all_j - set(tgt))

    def _err(ids: Sequence[int]) -> float:
        if not ids:
            return 0.0
        vals = [float(np.linalg.norm(a[i, ids] - b[k, ids], axis=-1).mean()) for i, k in path]
        return float(np.mean(vals))

    target_err = _err(tgt)
    rest_err = _err(rest)
    return {
        "target_joints": tgt,
        "target_change": round(target_err, 4),
        "non_target_mpjpe": round(rest_err, 4),
        "non_target_mpjpe_norm": round(rest_err / h, 4),
        "target_vs_rest": round(target_err / rest_err, 3) if rest_err > 1e-6 else None,
        "dtw_frames": len(path),
    }


def instruction_following(
    instruction: str,
    before: Dict[str, float],
    after: Dict[str, float],
    pres: Dict[str, Any],
) -> Dict[str, Any]:
    """Did the requested attribute move in the requested direction?

    Matching is on whole words: "slower" must not be read as an instruction to
    go *lower*, and "sitting" must not be read as *sit down*.
    """
    t = (instruction or "").lower()
    checks: List[Dict[str, Any]] = []

    def word(*cands: str) -> bool:
        return any(re.search(rf"\b{re.escape(c)}\b", t) for c in cands)

    def add(name: str, ok: bool, delta: float, need: str) -> None:
        checks.append({"attribute": name, "ok": bool(ok), "delta": round(float(delta), 4), "expected": need})

    if word("arm", "wave", "hand"):
        side = "left" if word("left") else "right"
        key = f"arm_raise_{side}"
        d = float(after.get(key, 0)) - float(before.get(key, 0))
        higher = word("higher", "raise", "up", "overhead")
        add(key, d > 0.05 if higher else abs(d) > 0.03, d, "> +0.05" if higher else "|Δ| > 0.03")
    if word("walk", "forward", "further", "step"):
        tb = max(1.0, float(before.get("T") or 1))
        ta = max(1.0, float(after.get("T") or 1))
        sb = float(before.get("root_travel", 0)) / tb
        sa = float(after.get("root_travel", 0)) / ta
        if word("slower", "slow"):
            # A slowed walk covers less ground per frame, but the clip gets
            # longer, so total travel can be unchanged. Compare speed, not
            # distance, or a correct slow-down reads as a failure.
            add("speed_per_frame", sa < sb - 1e-3, sa - sb, "< 0 m/frame")
        elif word("faster", "quicker", "speed", "up"):
            add("speed_per_frame", sa > sb + 1e-3, sa - sb, "> 0 m/frame")
        else:
            d = float(after.get("root_travel", 0)) - float(before.get("root_travel", 0))
            add("root_travel", d > 0.04, d, "> +0.04 m")
    if any(w in t for w in ("stop", "straight", "no spin", "do not spin")):
        d = float(after.get("yaw_span_deg", 0)) - float(before.get("yaw_span_deg", 0))
        add("yaw_span_deg", d < -3.0, d, "< -3 deg")
    if word("sit") and word("down", "lower", "deeper", "seat") or word("crouch", "squat", "lower"):
        d = float(after.get("pelvis_drop", 0)) - float(before.get("pelvis_drop", 0))
        add("pelvis_drop", d > 0.02, d, "> +0.02 m")
    if word("faster", "slower", "speed"):
        d = float(after.get("T", 0)) - float(before.get("T", 0))
        add("duration_frames", abs(d) > 1.0, d, "changed duration")
    if word("trim", "shorten", "cut", "remove") and word("motion", "extra", "tail", "end"):
        d = float(after.get("T", 0)) - float(before.get("T", 0))
        add("duration_frames", d < -1.0, d, "shorter clip")
    if any(p in t for p in ("walk in place", "step in place", "stay in place",
                            "without moving", "keep the root", "do not move",
                            "don't move", "in place without")):
        d = float(after.get("root_travel", 0))
        add("root_travel", d <= 0.25, d, "<= 0.25 m (in place)")

    verifiable = True
    if not checks:
        ratio = pres.get("target_vs_rest")
        checks.append({
            "attribute": "target_vs_rest",
            "ok": bool(ratio is not None and ratio > 1.2),
            "delta": ratio,
            "expected": "> 1.2",
        })
        # No attribute test applies and the localisation ratio is unavailable
        # (an edit that changed nothing, for instance). Say so instead of
        # counting an unmeasurable instruction as a failure.
        verifiable = ratio is not None
    return {"checks": checks, "followed": all(c["ok"] for c in checks), "verifiable": verifiable}


# --------------------------------------------------------------------------
# (c) captioning metrics (no extra dependencies)
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9']+")
_GLOVE: Optional[Tuple[np.ndarray, Dict[str, int]]] = None


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def _load_glove() -> Tuple[np.ndarray, Dict[str, int]]:
    global _GLOVE
    if _GLOVE is not None:
        return _GLOVE
    words_path = GLOVE_DIR / "our_vab_words.pkl"
    data_path = GLOVE_DIR / "our_vab_data.npy"
    if not (words_path.is_file() and data_path.is_file()):
        _GLOVE = (np.zeros((0, 300), dtype=np.float32), {})
        return _GLOVE
    words = pickle.loads(words_path.read_bytes())
    data = np.load(data_path).astype(np.float32)
    index = {str(w): i for i, w in enumerate(words)}
    _GLOVE = (data, index)
    return _GLOVE


def embed(text: str) -> np.ndarray:
    """Mean GloVe vector, L2-normalised (zero vector when no word is known)."""
    data, index = _load_glove()
    ids = [index[w] for w in tokenize(text) if w in index]
    if not ids or data.size == 0:
        return np.zeros(data.shape[1] if data.size else 300, dtype=np.float32)
    v = data[ids].mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def text_similarity(a: str, b: str) -> Dict[str, float]:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    inter = len(ta & tb)
    union = len(ta | tb) or 1
    return {
        "glove_cosine": round(cosine(embed(a), embed(b)), 4),
        "token_jaccard": round(inter / union, 4),
        "token_recall": round(inter / max(1, len(tb)), 4),
    }


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(refs: Sequence[str], hyp: str, max_n: int = 4) -> List[float]:
    h = tokenize(hyp)
    refs_t = [tokenize(r) for r in refs]
    if not h:
        return [0.0] * max_n
    closest = min((abs(len(r) - len(h)) for r in refs_t), default=0)
    ref_len = min((len(r) for r in refs_t if abs(len(r) - len(h)) == closest), default=0)
    bp = 1.0 if len(h) > ref_len else math.exp(1.0 - ref_len / max(1, len(h)))
    precisions = []
    for n in range(1, max_n + 1):
        hc = _ngrams(h, n)
        max_ref = Counter()
        for r in refs_t:
            rc = _ngrams(r, n)
            for g, c in rc.items():
                max_ref[g] = max(max_ref[g], c)
        clipped = sum(min(c, max_ref[g]) for g, c in hc.items())
        precisions.append(clipped / max(1, len(h) - n + 1))
    if min(precisions) <= 0:
        # BLEU-4 is zero when any order has no match, but report the lower
        # orders that are still defined instead of zeroing the whole vector.
        out = []
        for n in range(1, max_n + 1):
            p = precisions[:n]
            out.append(0.0 if min(p) <= 0 else bp * math.exp(sum(math.log(x) for x in p) / n))
        return out
    return [bp * math.exp(sum(math.log(p) for p in precisions[:n]) / n) for n in range(1, max_n + 1)]


def corpus_bleu(references: Sequence[Sequence[str]], hypotheses: Sequence[str],
                max_n: int = 4) -> List[float]:
    """Corpus-level BLEU (clipped n-gram counts pooled over the test set)."""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0
    for refs, hyp in zip(references, hypotheses):
        h = tokenize(hyp)
        refs_t = [tokenize(r) for r in refs]
        if not h or not refs_t:
            continue
        hyp_len += len(h)
        closest = min(abs(len(r) - len(h)) for r in refs_t)
        ref_len += min(len(r) for r in refs_t if abs(len(r) - len(h)) == closest)
        for n in range(1, max_n + 1):
            hc = _ngrams(h, n)
            max_ref: Counter = Counter()
            for r in refs_t:
                for g, c in _ngrams(r, n).items():
                    max_ref[g] = max(max_ref[g], c)
            matches[n - 1] += sum(min(c, max_ref[g]) for g, c in hc.items())
            totals[n - 1] += max(1, len(h) - n + 1)
    if hyp_len == 0:
        return [0.0] * max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    precisions = [m / max(1, t) for m, t in zip(matches, totals)]
    out = []
    for n in range(1, max_n + 1):
        p = precisions[:n]
        out.append(0.0 if min(p) <= 0 else bp * math.exp(sum(math.log(x) for x in p) / n))
    return out


def rouge_l(refs: Sequence[str], hyp: str) -> float:
    h = tokenize(hyp)
    best = 0.0
    for r in refs:
        rr = tokenize(r)
        if not h or not rr:
            continue
        dp = np.zeros((len(h) + 1, len(rr) + 1), dtype=np.int32)
        for i in range(1, len(h) + 1):
            for k in range(1, len(rr) + 1):
                dp[i, k] = dp[i - 1, k - 1] + 1 if h[i - 1] == rr[k - 1] else max(dp[i - 1, k], dp[i, k - 1])
        lcs = int(dp[-1, -1])
        if lcs == 0:
            continue
        p = lcs / len(h)
        rec = lcs / len(rr)
        best = max(best, 2 * p * rec / (p + rec))
    return float(best)


def _corpus_df(references: Sequence[Sequence[str]], max_n: int = 4) -> Tuple[List[Dict[Any, int]], int]:
    dfs: List[Dict[Any, int]] = [{} for _ in range(max_n)]
    docs = [tokenize(r) for refs in references for r in refs]
    for doc in docs:
        for n in range(1, max_n + 1):
            for g in _ngrams(doc, n):
                dfs[n - 1][g] = dfs[n - 1].get(g, 0) + 1
    return dfs, max(1, len(docs))


def cider_d(refs: Sequence[str], hyp: str, dfs: List[Dict[Any, int]], n_docs: int,
            max_n: int = 4, sigma: float = 6.0) -> float:
    h = tokenize(hyp)
    if not h:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        hv: Dict[Any, float] = {}
        for g, c in _ngrams(h, n).items():
            idf = math.log(n_docs / (1.0 + dfs[n - 1].get(g, 0)))
            hv[g] = c * idf
        norm = math.sqrt(sum(v * v for v in hv.values())) or 1.0
        sims = []
        for r in refs:
            rt = tokenize(r)
            rv: Dict[Any, float] = {}
            for g, c in _ngrams(rt, n).items():
                idf = math.log(n_docs / (1.0 + dfs[n - 1].get(g, 0)))
                rv[g] = c * idf
            rn = math.sqrt(sum(v * v for v in rv.values())) or 1.0
            dot = sum(v * rv.get(g, 0.0) for g, v in hv.items())
            pen = math.exp(-((len(h) - len(rt)) ** 2) / (2 * sigma ** 2))
            sims.append(dot / (norm * rn) * pen)
        scores.append(float(np.mean(sims)) if sims else 0.0)
    return 10.0 * float(np.mean(scores))


def caption_scores(references: Sequence[Sequence[str]], hypothesis: str,
                   dfs: Optional[List[Dict[Any, int]]] = None,
                   n_docs: Optional[int] = None) -> Dict[str, float]:
    refs = list(references)
    bleu = sentence_bleu(refs, hypothesis)
    if dfs is None or n_docs is None:
        dfs, n_docs = _corpus_df([refs])
    return {
        "bleu1": round(bleu[0], 4),
        "bleu2": round(bleu[1], 4),
        "bleu3": round(bleu[2], 4),
        "bleu4": round(bleu[3], 4),
        "rouge_l": round(rouge_l(refs, hypothesis), 4),
        "cider_d": round(cider_d(refs, hypothesis, dfs, n_docs), 4),
        "glove_cosine": round(
            float(np.mean([cosine(embed(hypothesis), embed(r)) for r in refs])), 4
        ),
    }


# --------------------------------------------------------------------------
# HumanML3D caption references
# --------------------------------------------------------------------------
def hml_gt_captions(motion_id: str) -> List[str]:
    """HumanML3D text rows are ``caption#tokens#start#end``; keep the caption."""
    path = HML_ROOT / "texts" / f"{motion_id}.txt"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        caption = line.split("#")[0].strip()
        if caption:
            out.append(caption)
    return out


def hml_test_ids(limit: int = 8, seed: int = 7) -> List[str]:
    split = HML_ROOT / "test.txt"
    ids = [x.strip() for x in split.read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    out = []
    for mid in ids:
        if (HML_ROOT / "new_joints" / f"{mid}.npy").is_file() and hml_gt_captions(mid):
            out.append(mid)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# CLI: clip report / caption benchmark
# --------------------------------------------------------------------------
def clip_report(path: Path) -> Dict[str, Any]:
    j = load_joints(path)
    out = {"joints": str(path), **kinematic_summary(j)}
    return out


def caption_benchmark(n: int = 8, out: Optional[Path] = None) -> Dict[str, Any]:
    """Run the M2T model on held-out HumanML3D clips and score the captions."""
    backends = REPO / "scripts" / "backends"
    if str(backends) not in sys.path:
        sys.path.insert(0, str(backends))
    from tm2t_infer import describe  # noqa: E402

    ids = hml_test_ids(n)
    refs_all = [hml_gt_captions(mid) for mid in ids]
    dfs, n_docs = _corpus_df(refs_all)
    rows = []
    for mid, refs in zip(ids, refs_all):
        pred = describe(HML_ROOT / "new_joints" / f"{mid}.npy")
        hyp = str(pred.get("caption") or "")
        rows.append({
            "id": mid,
            "reference": refs,
            "hypothesis": hyp,
            **caption_scores(refs, hyp, dfs, n_docs),
        })
    agg = {}
    corpus = {}
    if rows:
        for key in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider_d", "glove_cosine"):
            agg[key] = round(float(np.mean([r[key] for r in rows])), 4)
        cb = corpus_bleu([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
        corpus = {f"bleu{i + 1}": round(float(x), 4) for i, x in enumerate(cb)}
        corpus["rouge_l"] = agg["rouge_l"]
        corpus["cider_d"] = agg["cider_d"]
        corpus["glove_cosine"] = agg["glove_cosine"]
    report = {
        "schema_version": "caption_bench_v1",
        "n": len(rows),
        "aggregate": corpus or agg,
        "aggregate_sentence_mean": agg,
        "rows": rows,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantitative metrics for motion artifacts")
    ap.add_argument("--joints", type=Path, default=None)
    ap.add_argument("--caption-bench", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.caption_bench:
        report = caption_benchmark(args.n, args.out)
        print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
        return
    if not args.joints:
        raise SystemExit("give --joints or --caption-bench")
    report = clip_report(args.joints)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
