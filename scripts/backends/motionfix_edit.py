"""
MotionFix single-sample text edit.

Env: conda `motionfix`
Usage (from repo root or anywhere):

  python scripts/backends/motionfix_edit.py \\
      --source-id 005454 \\
      --text "a person walks forward slowly" \\
      --out outputs/motionfix/edit_005454.npy

Source options:
  --source-id     sample id from MotionFix test split (data/motionfix_test.pth.tar)
  --source-npy    prior MotionFix output {'pose': (T, 135)}  [trans + 6d rots]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTIONFIX_ROOT = REPO_ROOT / "third_party" / "motionfix"
DEFAULT_CKPT = MOTIONFIX_ROOT / "experiments" / "tmed" / "checkpoints" / "last.ckpt"
DEFAULT_STATS = MOTIONFIX_ROOT / "deps" / "stats" / "statistics_motionfix.npy"
DEFAULT_SMPL = MOTIONFIX_ROOT / "data" / "body_models"
DEFAULT_CLIP = MOTIONFIX_ROOT / "deps" / "clip-vit-large-patch14"
DEFAULT_TEST_DB = MOTIONFIX_ROOT / "data" / "motionfix-dataset" / "motionfix_test.pth.tar"
LOAD_FEATS = [
    "body_transl_delta_pelv",
    "body_orient_xy",
    "z_orient_delta",
    "body_pose",
    "body_joints_local_wo_z_rot",
]


def _patch_torch_load() -> None:
    """Lightning ckpt embeds OmegaConf; torch>=2.6 defaults weights_only=True."""
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]


def _patch_paths(root: Path) -> None:
    """Rewrite Linux absolute deps paths stored in checkpoints / hparams."""
    import src.utils.file_io as fio
    import hydra

    def hack_path(path, keyword="deps"):
        s = str(path).replace("\\\\", "/").replace("\\", "/")
        parts = s.split("/")
        if keyword in parts:
            idx = parts.index(keyword)
            return str(root.joinpath(*parts[idx:]))
        p = Path(path)
        if p.exists():
            return str(p)
        return str(root / s)

    fio.hack_path = hack_path
    hydra.utils.get_original_cwd = lambda: str(root)


def _setup_motionfix_env(root: Path) -> None:
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _patch_torch_load()
    _patch_paths(root)
    _stub_render_imports()


def _stub_render_imports() -> None:
    """Skip aitviewer/pxr import chain (often blocked by Windows WDAC / AppLocker).

    MotionsFix ``BaseModel`` imports ``src.render.mesh_viz`` / ``aitviewer`` at load;
    edit-only path does not need headless rendering.
    """
    import types

    def _pkg(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
        return mod

    if "src.render.mesh_viz" not in sys.modules:
        mesh = types.ModuleType("src.render.mesh_viz")
        mesh.render_motion = lambda *a, **k: None  # type: ignore[attr-defined]
        render = _pkg("src.render")
        render.mesh_viz = mesh  # type: ignore[attr-defined]
        render.render_motion = mesh.render_motion  # type: ignore[attr-defined]
        sys.modules["src.render.mesh_viz"] = mesh

    ait = _pkg("aitviewer")
    _pkg("aitviewer.headless")
    _pkg("aitviewer.viewer")
    _pkg("aitviewer.renderables")
    _pkg("aitviewer.renderables.billboard")
    models = _pkg("aitviewer.models")
    smpl_mod = types.ModuleType("aitviewer.models.smpl")

    class SMPLLayer:  # noqa: N801 — match aitviewer API name
        def __init__(self, *a, **k):
            pass

    smpl_mod.SMPLLayer = SMPLLayer  # type: ignore[attr-defined]
    sys.modules["aitviewer.models.smpl"] = smpl_mod
    models.smpl = smpl_mod  # type: ignore[attr-defined]
    ait.models = models  # type: ignore[attr-defined]


def load_model(
    root: Path,
    ckpt: Path,
    stats: Path,
    smpl: Path,
    clip: Path,
    device: str,
):
    from omegaconf import OmegaConf
    from src.model.base_diffusion import MD

    text_encoder = OmegaConf.create(
        {
            "_target_": "src.model.textencoder.clip_encoder.ClipTextEncoder",
            "finetune": False,
            "last_hidden_state": True,
            "modelpath": str(clip),
        }
    )
    model = MD.load_from_checkpoint(
        str(ckpt),
        renderer=None,
        statistics_path=str(stats),
        smpl_path=str(smpl),
        text_encoder=text_encoder,
        strict=False,
    )
    model.eval()
    model.freeze()
    model = model.to(device)
    transl_feats = [x for x in model.input_feats if "transl" in x]
    if set(transl_feats).issubset(
        ["body_transl_delta", "body_transl_delta_pelv", "body_transl_delta_pelv_xy"]
    ):
        model.using_deltas_transl = True
    return model


def _pose_npy_to_motion_source(pose: np.ndarray, smpl_path: Path) -> Dict[str, Any]:
    """Convert MotionFix {'pose'} array (T, 135) → motion_source dict with joints."""
    import smplx
    from src.model.utils.smpl_fast import smpl_forward_fast
    from src.tools.transforms3d import transform_body_pose
    from src.utils.genutils import freeze

    if pose.ndim != 2 or pose.shape[-1] < 135:
        raise ValueError(f"Expected pose (T, >=135), got {pose.shape}")

    pose_t = torch.from_numpy(pose[:, :135]).float()
    trans = pose_t[:, :3]
    rots_6d = pose_t[:, 3:]  # (T, 132) = 22 * 6
    rots_aa = transform_body_pose(rots_6d.reshape(-1, 22, 6), "6d->aa")  # (T, 22, 3)
    rots_flat = rots_aa.reshape(pose_t.shape[0], -1)  # (T, 66)
    rots_rotm = transform_body_pose(rots_aa, "aa->rot")

    body_model = smplx.SMPLHLayer(
        f"{smpl_path}/smplh", model_type="smplh", gender="neutral", ext="npz"
    ).eval()
    setattr(smplx.SMPLHLayer, "smpl_forward_fast", smpl_forward_fast)
    freeze(body_model)

    with torch.no_grad():
        out = body_model.smpl_forward_fast(
            transl=trans,
            body_pose=rots_rotm[:, 1:],
            global_orient=rots_rotm[:, :1],
        )
        joints = out.joints[:, :22].cpu()

    return {
        "rots": rots_flat.cpu(),
        "trans": trans.cpu(),
        "joint_positions": joints,
    }


def facefront_motion_source(ms: Dict[str, Any], smpl_path: Path) -> Dict[str, Any]:
    """Apply MotionsFix dataset face-front canonicalize + re-FK (Z-up yaw)."""
    from src.data.motionfix import MotionFixDataset

    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    else:
        rots = rots.float()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        trans = trans.float()

    rots_can, trans_can = MotionFixDataset._canonica_facefront(rots, trans)
    if not torch.is_tensor(rots_can):
        rots_can = torch.from_numpy(np.asarray(rots_can)).float()
    if not torch.is_tensor(trans_can):
        trans_can = torch.from_numpy(np.asarray(trans_can)).float()
    joints = _fk_joints_smplh(rots_can, trans_can, smpl_path)
    return {
        "rots": rots_can.cpu(),
        "trans": trans_can.cpu(),
        "joint_positions": joints,
    }


def load_source_datum(
    source_id: Optional[str],
    source_npy: Optional[Path],
    test_db: Path,
    smpl_path: Path,
    adapt_bridged: bool = False,
    body_amp: float = 1.45,
    canonicalize: bool = False,
    align_bridged: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Return (datum with motion_source/target/text/id/split, resolved_id).

    Bridged ``--source-npy`` should already be MotionsFix face-front canonical
    (produced by the bridge). Re-canonicalizing here twists facing (~150 deg)
    and makes MotionsFix diffusion OOD — so default is ``canonicalize=False``,
    matching ``--source-id`` (dataset already prepared).
    """
    from src.utils.genutils import cast_dict_to_tensors

    if source_id is not None:
        db = joblib.load(test_db)
        if source_id not in db:
            # allow numeric without zero-pad
            padded = str(source_id).zfill(6)
            if padded in db:
                source_id = padded
            else:
                raise KeyError(
                    f"source-id {source_id!r} not in {test_db} "
                    f"(keys e.g. {list(db.keys())[:5]})"
                )
        raw = db[source_id]
        datum = cast_dict_to_tensors(
            {
                "motion_source": raw["motion_source"],
                "motion_target": raw["motion_target"],
                "text": raw.get("text", ""),
                "id": source_id,
                "split": 2,
            }
        )
        return datum, source_id

    if source_npy is None:
        raise ValueError("Provide --source-id or --source-npy")

    loaded = np.load(source_npy, allow_pickle=True)
    obj = loaded.item() if loaded.dtype == object else {"pose": loaded}
    if "motion_source" in obj:
        ms = cast_dict_to_tensors(obj["motion_source"])
    elif "pose" in obj:
        ms = _pose_npy_to_motion_source(np.asarray(obj["pose"]), smpl_path)
    else:
        raise ValueError(
            f"{source_npy} must contain 'pose' or 'motion_source', got keys={list(obj)}"
        )
    # Debug override or explicit --canonicalize only.
    do_canon = bool(canonicalize) or bool(
        getattr(load_source_datum, "_force_canonicalize", False)
    )
    if do_canon:
        print("[motionfix_edit] WARNING: re-canonicalizing source-npy (debug)")
        ms = _canonicalize_motion_source(ms, smpl_path)
    elif not obj.get("motionfix_canonical", False):
        print(
            "[motionfix_edit] NOTE: source-npy has motionfix_canonical=false; "
            "prefer re-bridge with face-front (see mld_to_motionfix / "
            "motionfix_facefront.py)"
        )
    # Stats align first (root speed + body |aa| → MotionsFix medians).
    if align_bridged:
        ms = align_bridged_to_motionfix_stats(ms, smpl_path)
    if adapt_bridged:
        ms = adapt_bridged_source(ms, smpl_path, body_amp=body_amp)
    sid = source_npy.stem
    datum = {
        "motion_source": ms,
        "motion_target": {k: v.clone() if torch.is_tensor(v) else v for k, v in ms.items()},
        "text": str(obj.get("text", "")),
        "id": sid,
        "split": 2,
    }
    return datum, sid


def _fk_joints_smplh(
    rots_aa_flat: torch.Tensor,
    trans: torch.Tensor,
    smpl_path: Path,
) -> torch.Tensor:
    """(T,66) aa + (T,3) transl → (T,22,3) joints via SMPL-H."""
    import smplx
    from einops import rearrange
    from src.model.utils.smpl_fast import smpl_forward_fast
    from src.tools.transforms3d import transform_body_pose
    from src.utils.genutils import freeze

    rots_aa = rearrange(rots_aa_flat, "f (j d) -> f j d", j=22, d=3)
    rots_rotm = transform_body_pose(rots_aa, "aa->rot")
    body = smplx.SMPLHLayer(
        f"{smpl_path}/smplh", model_type="smplh", gender="neutral", ext="npz"
    ).eval()
    setattr(smplx.SMPLHLayer, "smpl_forward_fast", smpl_forward_fast)
    freeze(body)
    with torch.no_grad():
        out = body.smpl_forward_fast(
            transl=trans,
            body_pose=rots_rotm[:, 1:],
            global_orient=rots_rotm[:, :1],
        )
        return out.joints[:, :22].cpu()


def _canonicalize_motion_source(ms: Dict[str, Any], smpl_path: Path) -> Dict[str, Any]:
    """Face-front canonicalize + joints consistent with rots/trans (MotionFix prep)."""
    return facefront_motion_source(ms, smpl_path)


# MotionsFix test-split medians (≈200 samples): body |aa| and root step p95.
MF_BODY_ABS_MEAN = 0.195
MF_ROOT_STEP_P95 = 0.025


def adapt_bridged_source(
    ms: Dict[str, Any],
    smpl_path: Path,
    body_amp: float = 1.45,
    max_aa: float = 1.25,
) -> Dict[str, Any]:
    """Amplify under-articulated MLD/SMPL body pose toward MotionFix range.

    Bridged walks often have weak limb swing vs AMASS/MotionFix; MotionsFix then
    under-edits semantic gestures. Scale non-pelvis axis-angles, clamp, re-FK.
    """
    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    else:
        rots = rots.float().clone()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        trans = trans.float().clone()

    T = rots.shape[0]
    body = rots[:, 3:].reshape(T, 21, 3)
    body = body * float(body_amp)
    mag = body.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    body = body * (mag.clamp(max=float(max_aa)) / mag)
    rots = rots.clone()
    rots[:, 3:] = body.reshape(T, -1)
    joints = _fk_joints_smplh(rots, trans, smpl_path)
    print(
        f"[motionfix_edit] adapted bridged source body_amp={body_amp:.2f} "
        f"max_aa={max_aa:.2f}"
    )
    return {"rots": rots.cpu(), "trans": trans.cpu(), "joint_positions": joints}


def align_bridged_to_motionfix_stats(
    ms: Dict[str, Any],
    smpl_path: Path,
    target_body_abs: float = MF_BODY_ABS_MEAN,
    target_step_p95: float = MF_ROOT_STEP_P95,
) -> Dict[str, Any]:
    """Affine-align bridged root speed + body |aa| toward MotionsFix medians.

    Soft pull (sqrt) avoids over-shrinking travel; body scale clamped to [0.85, 1.6].
    """
    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    else:
        rots = rots.float().clone()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        trans = trans.float().clone()

    trans_np = trans.detach().cpu().numpy()
    step = np.linalg.norm(np.diff(trans_np, axis=0), axis=-1)
    p95 = float(np.percentile(step, 95)) if len(step) else 0.0
    speed_scale = 1.0
    if p95 > 1e-6:
        ratio = float(target_step_p95) / p95
        # Mild pull toward target (full clamp was too aggressive historically).
        speed_scale = float(np.clip(np.sqrt(ratio), 0.35, 1.8))
        if abs(speed_scale - 1.0) > 0.02:
            trans = (trans * speed_scale).float()
            # Re-zero start after scale.
            trans = trans - trans[:1]

    body = rots[:, 3:]
    cur = float(body.abs().mean().item())
    body_scale = 1.0
    if cur > 1e-6:
        body_scale = float(np.clip(float(target_body_abs) / cur, 0.85, 1.6))
        if abs(body_scale - 1.0) > 0.02:
            body = body * body_scale
            mag = body.reshape(-1, 21, 3).norm(dim=-1, keepdim=True).clamp(min=1e-8)
            body = (body.reshape(-1, 21, 3) * (mag.clamp(max=1.35) / mag)).reshape(
                body.shape[0], -1
            )
            rots = rots.clone()
            rots[:, 3:] = body

    joints = _fk_joints_smplh(rots, trans, smpl_path)
    new_step = np.linalg.norm(
        np.diff(trans.detach().cpu().numpy(), axis=0), axis=-1
    )
    new_p95 = float(np.percentile(new_step, 95)) if len(new_step) else 0.0
    new_body = float(rots[:, 3:].abs().mean().item())
    print(
        f"[motionfix_edit] stats-align speed_x={speed_scale:.3f} "
        f"(step_p95 {p95:.4f}->{new_p95:.4f}) body_x={body_scale:.3f} "
        f"(|aa| {cur:.3f}->{new_body:.3f})"
    )
    return {"rots": rots.cpu(), "trans": trans.cpu(), "joint_positions": joints}


def build_batch(datum: Dict[str, Any], edit_text: str, stats_file: Path, smpl_path: Path):
    from src.data.motionfix import MotionFixDataset
    from src.data.tools.collate import collate_batch_last_padding

    # For free edit, target length follows source (duplicate source as target feats).
    datum = dict(datum)
    datum["motion_target"] = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in datum["motion_source"].items()
    }
    datum["text"] = edit_text

    ds = MotionFixDataset(
        data=[datum],
        n_body_joints=22,
        stats_file=str(stats_file),
        norm_type="std",
        smplh_path=str(smpl_path),
        rot_repr="6d",
        load_feats=LOAD_FEATS,
    )
    item = ds[0]
    item["text"] = edit_text
    batch = collate_batch_last_padding([item], LOAD_FEATS)
    return batch


def prepare_test_batch(model, batch):
    batch = {
        k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()
    }
    input_batch = model.norm_and_cat(batch, model.input_feats)
    for k, v in input_batch.items():
        batch[f"{k}_motion"] = v
    return batch


def infer_speed_scale(edit_text: str) -> Optional[float]:
    """Return root-speed scale for pure speed edits, else None."""
    import re

    t = (edit_text or "").lower()
    # Explicit multipliers first (ZH/EN)
    if any(k in edit_text for k in ("两倍", "2倍", "二倍")) or any(
        k in t for k in ("twice", "2x", "double", "two times")
    ):
        return 2.0
    if any(k in edit_text for k in ("三倍", "3倍")) or "3x" in t or "triple" in t:
        return 3.0
    if any(k in edit_text for k in ("一半",)) or "half" in t:
        return 0.5
    m = re.search(r"speed\s*x\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×倍]", edit_text or "")
    if m:
        return float(m.group(1))
    # order matters: more specific phrases first
    rules = [
        (("very slowly", "much slower"), 0.35),
        (("slowly", "slower", "slow down", "slow walk"), 0.45),
        (("very fast", "much faster", "quickly"), 1.6),
        (("faster", "fast", "speed up"), 1.35),
    ]
    for keys, scale in rules:
        if any(k in t for k in keys):
            return scale
    if "快" in (edit_text or "") or "加速" in (edit_text or ""):
        return 1.35
    if "慢" in (edit_text or "") or "减速" in (edit_text or ""):
        return 0.45
    return None


def infer_kinematic_intent(edit_text: str) -> Optional[str]:
    """Map text → kinematic overlay intent for OOD bridged sources.

    MotionsFix diffusion is reliable on MotionsFix-native motions; bridged MLD
    sources are OOD and often collapse under diffusion. For common intents we
    apply geometry on top of the source (same idea as geometric speed edit).
    """
    t = edit_text.lower()
    if any(k in t for k in ("wave", "waving", "hand wave")):
        return "wave"
    if any(k in t for k in ("sit down", "sits down", "sitting down", "sit on")) or (
        "sit" in t and "side" not in t
    ):
        return "sit"
    if any(
        k in t
        for k in (
            "turn around",
            "turns around",
            "turn back",
            "walk back",
            "about face",
            "180",
        )
    ):
        return "turn"
    if any(k in t for k in ("turn left", "turn to the left")):
        return "turn_left"
    if any(k in t for k in ("turn right", "turn to the right", "turn 90")):
        return "turn_right"
    return None


def _aa_add(rots_flat: torch.Tensor, joint: int, delta_xyz: torch.Tensor) -> None:
    """In-place add axis-angle delta to one joint (T,66) layout."""
    s = joint * 3
    rots_flat[:, s : s + 3] = rots_flat[:, s : s + 3] + delta_xyz


def geometric_semantic_edit(
    datum: Dict[str, Any],
    intent: str,
    smpl_path: Path,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Kinematic overlays for wave / sit / turn on bridged (OOD) sources."""
    ms = datum["motion_source"]
    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    else:
        rots = rots.float().clone()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        trans = trans.float().clone()

    T = int(rots.shape[0])
    t = torch.linspace(0, 1, T)
    meta: Dict[str, Any] = {"edit_mode": "geometric_semantic", "intent": intent}

    if intent == "wave":
        # Clear right-arm wave while keeping source locomotion (no MotionsFix OOD).
        n_waves = 3.0
        phase = 2 * np.pi * n_waves * t
        env = torch.ones(T)
        fade = max(2, T // 10)
        env[:fade] = torch.linspace(0, 1, fade)
        env[-fade:] = torch.linspace(1, 0, fade)
        flap = torch.sin(phase) * env
        # Strong raise + visible flap (SMPL local aa on R_shoulder / R_elbow / R_collar).
        d_sh = torch.stack(
            [
                0.25 * env + 0.15 * flap,
                1.15 * env + 0.25 * flap,
                -1.35 * env + 0.75 * flap,
            ],
            dim=-1,
        )
        d_el = torch.stack(
            [
                1.15 * env + 0.45 * torch.sin(phase + 0.4) * env,
                0.1 * flap,
                0.15 * flap,
            ],
            dim=-1,
        )
        d_col = torch.stack(
            [0.08 * env, 0.45 * env, -0.35 * env + 0.2 * flap], dim=-1
        )
        _aa_add(rots, 17, d_sh)
        _aa_add(rots, 19, d_el)
        _aa_add(rots, 14, d_col)
        meta["n_waves"] = n_waves
        meta["keep_locomotion"] = True

    elif intent == "sit":
        # Sit in place: freeze XY from frame 0, hip+knee flex, pelvis drop (chair-like).
        sit_frac = torch.clamp((t - 0.08) / 0.55, 0, 1)
        sit_frac = sit_frac * sit_frac * (3 - 2 * sit_frac)  # smoothstep
        trans = trans.clone()
        freeze_xy = trans[0:1, :2].clone()
        trans[:, :2] = freeze_xy.expand(T, 2)
        # Chair-height drop (~seat); foot re-ground later — avoid collapsing underground.
        pelvis_drop = 0.28
        trans[:, 2] = trans[:, 2] - pelvis_drop * sit_frac
        # Hip flex + knee so thighs near horizontal (sit), not deep kneel.
        for j in (1, 2):
            _aa_add(
                rots,
                j,
                torch.stack([0.85 * sit_frac, torch.zeros(T), torch.zeros(T)], -1),
            )
        for j in (4, 5):
            _aa_add(
                rots,
                j,
                torch.stack([1.15 * sit_frac, torch.zeros(T), torch.zeros(T)], -1),
            )
        for j, amt in ((3, 0.15), (6, 0.18)):
            _aa_add(
                rots,
                j,
                torch.stack([amt * sit_frac, torch.zeros(T), torch.zeros(T)], -1),
            )
        meta["pelvis_drop"] = pelvis_drop
        meta["freeze_xy"] = True

    elif intent in ("turn", "turn_left", "turn_right"):
        # In-place yaw only — no path arc (arc reads as sliding/平移).
        if intent == "turn_left":
            yaw_target = 0.5 * np.pi
        elif intent == "turn_right":
            yaw_target = -0.5 * np.pi
        else:
            yaw_target = np.pi
        turn_frac = torch.clamp((t - 0.08) / 0.75, 0, 1)
        turn_frac = turn_frac * turn_frac * (3 - 2 * turn_frac)
        yaw = yaw_target * turn_frac
        from src.tools.transforms3d import transform_body_pose

        go = rots[:, 0:3].reshape(T, 1, 3)
        go_mat = transform_body_pose(go, "aa->rot").reshape(T, 3, 3)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        zeros = torch.zeros(T)
        ones = torch.ones(T)
        Rz = torch.stack(
            [
                torch.stack([cos_y, -sin_y, zeros], -1),
                torch.stack([sin_y, cos_y, zeros], -1),
                torch.stack([zeros, zeros, ones], -1),
            ],
            dim=1,
        )
        go_new = torch.bmm(Rz, go_mat)
        go_aa = transform_body_pose(go_new.reshape(T, 1, 3, 3), "rot->aa").reshape(T, 3)
        rots = rots.clone()
        rots[:, 0:3] = go_aa
        # Plant root XY at start; keep Z as-is (standing turn).
        trans = trans.clone()
        freeze_xy = trans[0:1, :2].clone()
        trans[:, :2] = freeze_xy.expand(T, 2)
        meta["yaw_target"] = float(yaw_target)
        meta["freeze_xy"] = True

    else:
        raise ValueError(f"Unknown kinematic intent: {intent}")

    joints = _fk_joints_smplh(rots, trans, smpl_path)
    # For sit: only lift if feet punched through the floor. Do NOT pull the
    # body down when bent knees raise the feet (that collapses into a kneel).
    if intent == "sit":
        src_joints = ms.get("joint_positions")
        if src_joints is not None:
            if torch.is_tensor(src_joints):
                src_jp = src_joints.float().cpu().numpy()
            else:
                src_jp = np.asarray(src_joints, dtype=np.float32)
            if src_jp.shape[0] != joints.shape[0]:
                idx = np.linspace(0, src_jp.shape[0] - 1, joints.shape[0])
                foot_src = 0.5 * (
                    np.interp(idx, np.arange(src_jp.shape[0]), src_jp[:, 10, 2])
                    + np.interp(idx, np.arange(src_jp.shape[0]), src_jp[:, 11, 2])
                )
            else:
                foot_src = src_jp[:, [10, 11], 2].mean(axis=-1)
            foot_new = joints[:, [10, 11], 2].mean(axis=-1).numpy()
            # floor = source mean foot height; only correct penetration
            floor_z = float(np.median(foot_src[: max(1, len(foot_src) // 5)]))
            pen = np.minimum(0.0, foot_new - floor_z)  # negative if below floor
            if float(pen.min()) < -1e-4:
                trans = trans.clone()
                trans[:, 2] = trans[:, 2] - torch.from_numpy(pen.astype(np.float32))
                joints = _fk_joints_smplh(rots, trans, smpl_path)
                meta["foot_grounded"] = True
            else:
                meta["foot_grounded"] = False

    ms_out = {"rots": rots.cpu(), "trans": trans.cpu(), "joint_positions": joints}
    pose = motion_source_to_pose135(ms_out)
    meta["frames"] = int(pose.shape[0])
    return pose, int(pose.shape[0]), {**meta, "joint_positions": joints.numpy()}


def normalize_edit_text(edit_text: str) -> str:
    """Map absolute captions → MotionFix-style *relative* edit instructions.

    MotionFix was trained on short relative edits (e.g. \"wave to the side\",
    \"sit down quicker\"), not full HumanML captions like \"a person walks
    forward and waves\". Absolute captions often collapse to T-pose + root slide.
    """
    raw = (edit_text or "").strip()
    t = " ".join(raw.lower().split())
    if not t:
        return raw

    has_abs_frame = t.startswith(
        ("a person", "someone", "the person", "a man", "a woman", "a human")
    )

    # MotionsFix *training* relative phrasing (sampled from motionfix_test).
    # Applied even to short prompts so "sit down" / "wave" map to in-distribution text.
    rules = [
        (
            ("wave", "waving", "hand wave"),
            "wave to the side",
        ),
        (
            ("sit down", "sits down", "sitting down", "sit on", "sit "),
            "sit down quicker.",
        ),
        (
            ("turn around", "turns around", "turn back", "walk back", "180"),
            "dont bend and turn back 180 degrees",
        ),
        (
            ("turn left", "turn to the left"),
            "turn to the left",
        ),
        (
            ("turn right", "turn to the right", "turn 90"),
            "turn 90 degrees right",
        ),
        (
            ("raise both arms", "arms out", "t-pose", "t pose"),
            "raise both arms to the sides",
        ),
        (
            ("jump", "hop"),
            "jump once",
        ),
        (
            ("crouch", "squat"),
            "crouch down",
        ),
        (
            ("kick",),
            "kick with the right leg",
        ),
        (
            ("clap",),
            "clap hands",
        ),
    ]
    # Already MotionsFix-native — leave untouched.
    native = {r[1].lower().rstrip(".") for r in rules}
    native |= {
        "wave to the side",
        "sit down quicker",
        "sit down quicker.",
        "turn back 180 degrees",
        "turn 90 degrees right",
        "spin around faster and raise the right arm",
    }
    if t.rstrip(".") in {n.rstrip(".") for n in native} or t in native:
        return raw

    for keys, rewrite in rules:
        if any(k in t for k in keys):
            print(
                f"[motionfix_edit] rewrite edit text {raw!r} → {rewrite!r} "
                "(MotionFix expects relative edits)"
            )
            return rewrite

    # Short non-matched prompts: keep as-is (likely already relative).
    if not has_abs_frame and len(t.split()) <= 10:
        return raw

    # Strip common absolute framing if present.
    for prefix in (
        "a person who is ",
        "a person is ",
        "a person ",
        "someone is ",
        "someone ",
        "the person ",
        "a man ",
        "a woman ",
    ):
        if t.startswith(prefix):
            stripped = t[len(prefix) :].strip(" .")
            if stripped and stripped != t:
                print(
                    f"[motionfix_edit] rewrite edit text {raw!r} → {stripped!r} "
                    "(strip absolute framing)"
                )
                return stripped
    return raw


def motion_source_to_pose135(ms: Dict[str, Any]) -> np.ndarray:
    """Pack MotionFix pose (T,135) = trans(3) + 22*6d from aa rots(T,66)."""
    from einops import rearrange
    from src.tools.transforms3d import transform_body_pose

    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    rots_aa = rearrange(rots, "t (j d) -> t j d", j=22, d=3)
    rots_6d = transform_body_pose(rots_aa, "aa->6d").reshape(rots.shape[0], -1)
    return torch.cat([trans, rots_6d], dim=-1).detach().cpu().numpy().astype(np.float32)


def geometric_speed_edit(
    datum: Dict[str, Any],
    speed_scale: float,
    smpl_path: Path,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Slow/fast while keeping body articulations.

    - Locomotion (meaningful XY travel): scale root translation.
    - In-place (sit/turn/wave): resample timeline (faster → fewer frames).

    MotionFix diffusion on OOD bridged MLD sources often collapses; geometric
    edit is the correct underlying fix for speed-only prompts.
    """
    # Caller (load_source_datum) already face-front canonicalized; do not
    # re-canonicalize or facing will drift vs the step-2 preview.
    ms = datum["motion_source"]
    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        rots = rots.float()
        trans = trans.float()

    scale = float(speed_scale)
    T0 = int(rots.shape[0])
    xy = trans[:, :2]
    travel = float(torch.linalg.norm(xy[-1] - xy[0]).item()) if T0 > 1 else 0.0
    temporal = travel < 0.08  # sit / turn / wave ≈ in place
    method = "temporal_resample" if temporal else "root_translate"

    if temporal and abs(scale - 1.0) > 0.02:
        new_T = int(max(30, min(300, round(T0 / max(scale, 1e-3)))))
        if new_T != T0:
            idx = torch.linspace(0, T0 - 1, new_T)
            i0 = idx.long().clamp(0, T0 - 1)
            i1 = (i0 + 1).clamp(0, T0 - 1)
            w = (idx - i0.float()).unsqueeze(-1)
            rots = rots[i0] * (1 - w) + rots[i1] * w
            trans = trans[i0] * (1 - w) + trans[i1] * w
            print(
                f"[motionfix_edit] in-place speed: temporal {T0}→{new_T} "
                f"(scale={scale:.2f})"
            )
    else:
        trans = trans * scale
        print(f"[motionfix_edit] locomotion speed: root_xy ×{scale:.2f}")

    trans = trans - trans[:1]
    ms_out = {"rots": rots, "trans": trans, "joint_positions": None}

    from einops import rearrange
    import smplx
    from src.model.utils.smpl_fast import smpl_forward_fast
    from src.tools.transforms3d import transform_body_pose
    from src.utils.genutils import freeze

    rots_aa = rearrange(rots, "t (j d) -> t j d", j=22, d=3)
    rots_rotm = transform_body_pose(rots_aa, "aa->rot")
    body = smplx.SMPLHLayer(
        f"{smpl_path}/smplh", model_type="smplh", gender="neutral", ext="npz"
    ).eval()
    setattr(smplx.SMPLHLayer, "smpl_forward_fast", smpl_forward_fast)
    freeze(body)
    with torch.no_grad():
        out = body.smpl_forward_fast(
            transl=trans,
            body_pose=rots_rotm[:, 1:],
            global_orient=rots_rotm[:, :1],
        )
        joints = out.joints[:, :22].cpu()
    ms_out["joint_positions"] = joints
    pose = motion_source_to_pose135(ms_out)
    meta = {
        "edit_mode": "geometric_speed",
        "speed_method": method,
        "speed_scale": float(speed_scale),
        "frames": int(pose.shape[0]),
        "root_travel_xy": travel,
    }
    return pose, int(pose.shape[0]), {**meta, "joint_positions": joints.numpy()}


def _resample_motion_source(ms: Dict[str, Any], target_len: int, smpl_path: Path) -> Dict[str, Any]:
    """Linearly resample MotionsFix motion_source to target_len frames."""
    rots = ms["rots"]
    trans = ms["trans"]
    if not torch.is_tensor(rots):
        rots = torch.from_numpy(np.asarray(rots)).float()
    else:
        rots = rots.float()
    if not torch.is_tensor(trans):
        trans = torch.from_numpy(np.asarray(trans)).float()
    else:
        trans = trans.float()
    t0 = int(rots.shape[0])
    if target_len <= 0 or target_len == t0:
        return ms
    # Interp in aa + trans (good enough for length padding into MotionsFix range).
    x_old = torch.linspace(0, 1, t0)
    x_new = torch.linspace(0, 1, target_len)

    def _interp(x: torch.Tensor) -> torch.Tensor:
        # x: (T, C)
        x = x.cpu()
        out = []
        for c in range(x.shape[1]):
            out.append(
                torch.from_numpy(
                    np.interp(x_new.numpy(), x_old.numpy(), x[:, c].numpy())
                ).float()
            )
        return torch.stack(out, dim=1)

    rots_i = _interp(rots)
    trans_i = _interp(trans)
    joints = _fk_joints_smplh(rots_i, trans_i, smpl_path)
    print(f"[motionfix_edit] resampled source {t0} → {target_len} frames")
    return {"rots": rots_i, "trans": trans_i, "joint_positions": joints}


def _blend_pose_keep_lower(src_pose: np.ndarray, edit_pose: np.ndarray) -> np.ndarray:
    """Keep pelvis+legs+global root from source; take torso/arms from edit.

    MotionsFix pose layout: trans(3) + 22*6d. Joint order matches SMPL:
      0 pelvis, 1-2 hips, 3 spine1, 4-5 knees, ... legs: 1,2,4,5,7,8,10,11
    """
    assert src_pose.shape == edit_pose.shape
    out = edit_pose.copy()
    # Always keep root translation from source (preserves walk path).
    out[:, :3] = src_pose[:, :3]
    # Keep global orient (joint 0) from source — prevents 90° tumbles.
    out[:, 3:9] = src_pose[:, 3:9]
    # Leg joints (aa→6d slots): indices 1,2,4,5,7,8,10,11
    leg_j = [1, 2, 4, 5, 7, 8, 10, 11]
    for j in leg_j:
        a = 3 + j * 6
        out[:, a : a + 6] = src_pose[:, a : a + 6]
    return out.astype(np.float32)


def _freeze_root_xy(
    pose: np.ndarray,
    src_pose: Optional[np.ndarray] = None,
    keep_z_from_edit: bool = True,
) -> np.ndarray:
    """Pin root XY to source start (stop sliding). Keep edit Z for sit drop."""
    out = pose.copy()
    ref = src_pose if src_pose is not None else pose
    xy0 = ref[0, :2].copy()
    out[:, 0] = xy0[0]
    out[:, 1] = xy0[1]
    if not keep_z_from_edit:
        out[:, 2] = ref[:, 2]
    return out.astype(np.float32)


def _infer_keep_lower(edit_text: str) -> bool:
    """Keep legs/root from source for arm gestures (MotionsFix often under-edits arms).

    Legs stay from source; arms come from diffusion. If diffusion arms are still
    weak, ``_maybe_boost_wave_arms`` adds a light MotionsFix-compatible overlay.
    """
    t = (edit_text or "").lower()
    if any(k in t for k in ("wave", "clap", "raise", "hand", "arm")):
        if not any(k in t for k in ("sit", "turn", "jump", "kick", "crouch", "spin")):
            return True
    return False


def _maybe_boost_wave_arms(
    pose: np.ndarray,
    edit_text: str,
    smpl_path: Path,
    min_wrist_z_span: float = 0.08,
) -> Tuple[np.ndarray, bool]:
    """If MotionsFix wave barely moves the wrist, add a light right-arm overlay."""
    t = (edit_text or "").lower()
    if not any(k in t for k in ("wave", "waving", "hand")):
        return pose, False
    if any(k in t for k in ("sit", "turn", "spin")):
        return pose, False
    try:
        ms = _pose_npy_to_motion_source(pose, smpl_path)
        joints = np.asarray(ms["joint_positions"])
        if torch.is_tensor(joints):
            joints = joints.detach().cpu().numpy()
        span = float(joints[:, 19, 2].max() - joints[:, 19, 2].min())
        if span >= min_wrist_z_span:
            return pose, False
        # Light geometric wave on MotionsFix output (still diffusion base).
        datum = {
            "motion_source": {
                "rots": ms["rots"],
                "trans": ms["trans"],
                "joint_positions": ms["joint_positions"],
            }
        }
        pose2, _, _ = geometric_semantic_edit(datum, "wave", smpl_path)
        print(
            f"[motionfix_edit] wave-rescue: MotionsFix wristZspan={span:.3f} "
            f"< {min_wrist_z_span}; blended geometric arm overlay"
        )
        # Keep MotionsFix root/legs; take rescued arms/torso from geometric.
        out = _blend_pose_keep_lower(pose, pose2)
        return out.astype(np.float32), True
    except Exception as exc:  # noqa: BLE001
        print(f"[motionfix_edit] wave-rescue skipped: {exc}")
        return pose, False


def _infer_freeze_xy(edit_text: str) -> bool:
    """Sit/turn should not slide — pin root XY after MotionsFix diffusion."""
    t = (edit_text or "").lower()
    return any(
        k in t
        for k in ("sit", "crouch", "squat", "turn", "spin", "180", "around")
    )


def infer_sdedit_strength(edit_text: str) -> float:
    """Moderate SDEdit: high noise + OOD bridge → garbled; too low → no-op."""
    t = (edit_text or "").lower()
    if any(k in t for k in ("sit", "crouch", "squat")):
        return 0.70
    if any(k in t for k in ("turn", "spin", "180", "around")):
        return 0.70
    if any(k in t for k in ("wave", "clap", "raise", "hand", "arm")):
        return 0.68
    return 0.55


def infer_gd_motion(edit_text: str) -> float:
    """Moderate source CFG so text can edit without discarding the bridge."""
    t = (edit_text or "").lower()
    if any(k in t for k in ("sit", "crouch", "squat", "turn", "spin", "180", "jump")):
        return 1.8
    if any(k in t for k in ("wave", "clap", "raise", "hand", "arm")):
        return 2.0
    return 2.5


def infer_init_from(edit_text: str) -> str:
    """Always SDEdit from source for bridged edits — full noise was garbled."""
    return "source"


_ACTION_TO_KIN = {
    "wave_right": "wave",
    "wave_left": "wave",
    "sit_down": "sit",
    "turn_in_place": "turn",
}


def run_edit(
    model,
    batch,
    edit_text: str,
    steps: int,
    gd_text: float,
    gd_motion: float,
    init_from: str,
    seed: int,
    sdedit_strength: float = 0.75,
    keep_lower: Optional[bool] = None,
    freeze_xy: Optional[bool] = None,
    source_pose135: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    from src.diffusion import create_diffusion

    pl = __import__("pytorch_lightning")
    pl.seed_everything(seed)

    text = [edit_text.lower()]
    source_lens = batch["length_source"]
    target_lens = batch["length_target"]
    # free edit: target len == source len
    if isinstance(source_lens, list):
        target_lens = list(source_lens)
        batch["length_target"] = target_lens

    input_batch = prepare_test_batch(model, batch)
    source_mot_pad = input_batch["source_motion"].clone()

    if getattr(model, "pad_inputs", False):
        mask_source, mask_target = model.prepare_mot_masks(
            source_lens, target_lens, max_len=300
        )
    else:
        mask_source, mask_target = model.prepare_mot_masks(
            source_lens, target_lens, max_len=None
        )

    # Official MotionFix eval: noise init + full train schedule (≈300).
    # init_from=source with SDEdit: q_sample(source, t*) then denoise t*→0.
    # Feeding clean x0 as x_T (old MotionsFix 'source' path) collapses to T-pose.
    train_steps = int(getattr(model.diff_params, "num_train_timesteps", 300))
    if steps <= 0 or steps > train_steps:
        steps = train_steps

    diffusion_process = create_diffusion(
        timestep_respacing=None,
        learn_sigma=False,
        sigma_small=True,
        diffusion_steps=train_steps,
        noise_schedule=model.diff_params.noise_schedule,
        predict_xstart=False
        if model.diff_params.predict_type == "noise"
        else True,
    )

    use_sdedit = init_from == "source" and float(sdedit_strength) < 0.999
    if use_sdedit:
        # MotionsFix features are (T, B, C); q_sample expects (B, T, C) like p_sample.
        x0 = source_mot_pad.permute(1, 0, 2).contiguous()  # (B, T, C)
        t_start = max(1, min(train_steps - 1, int(float(sdedit_strength) * (train_steps - 1))))
        t_tensor = torch.full((x0.shape[0],), t_start, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = diffusion_process.q_sample(x0, t_tensor, noise=noise)
        init_vec = x_t  # (B, T, C) — noise_prev expects this layout after generate_motion permute
        # generate_motion noise_prev: init_diff_rev = init_vec (used as initial_latents BxTxC)
        init_method = "noise_prev"
        print(
            f"[motionfix_edit] SDEdit strength={sdedit_strength:.2f} "
            f"t_start={t_start}/{train_steps}"
        )
        # Patch p_sample_loop to start from t_start instead of T-1.
        _orig_prog = diffusion_process.p_sample_loop_progressive

        def _prog_from_t(model_fn, shape, noise=None, clip_denoised=False,
                         denoised_fn=None, cond_fn=None, model_kwargs=None,
                         device=None, progress=False):
            if device is None:
                device = next(model_fn.parameters()).device if hasattr(model_fn, "parameters") else x_t.device
            img = noise if noise is not None else torch.randn(*shape, device=device)
            indices = list(range(t_start + 1))[::-1]
            if progress:
                from tqdm.auto import tqdm
                indices = tqdm(indices)
            for i in indices:
                t = torch.tensor([i] * shape[0], device=device)
                with torch.no_grad():
                    out = diffusion_process.p_sample(
                        model_fn,
                        img,
                        t,
                        clip_denoised=False,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                    yield out
                    img = out["sample"]

        diffusion_process.p_sample_loop_progressive = _prog_from_t  # type: ignore
    elif init_from == "source":
        init_vec = source_mot_pad
        init_method = "source"
        print(
            "[motionfix_edit] WARNING: init_from=source without SDEdit feeds "
            "clean features as x_T; prefer --sdedit-strength 0.55 or --init-from noise"
        )
    else:
        init_vec = None
        init_method = "noise"

    print(
        f"[motionfix_edit] diffusion train_steps={train_steps} "
        f"sample_steps={steps} init={init_method}"
    )

    with torch.no_grad():
        diffout = model.generate_motion(
            text,
            source_mot_pad,
            mask_source,
            mask_target,
            diffusion_process,
            init_vec=init_vec,
            init_vec_method=init_method,
            condition_mode="full_cond",
            gd_motion=gd_motion,
            gd_text=gd_text,
            num_diff_steps=steps,
            inpaint_dict=None,
            use_linear=False,
            prob_way="3way",
            show_progress=True,
        )
        gen_mo = model.diffout2motion(diffout)

    tlen = int(target_lens[0])
    pose = gen_mo[0, :tlen].detach().cpu().numpy()

    do_keep = _infer_keep_lower(edit_text) if keep_lower is None else keep_lower
    if do_keep and source_pose135 is not None and source_pose135.shape[0] == pose.shape[0]:
        pose = _blend_pose_keep_lower(source_pose135[:tlen], pose)
        print("[motionfix_edit] blended: kept source root/legs, took edit upper body")

    do_freeze = _infer_freeze_xy(edit_text) if freeze_xy is None else bool(freeze_xy)
    if do_freeze and source_pose135 is not None:
        src = source_pose135[:tlen]
        if src.shape[0] == pose.shape[0]:
            pose = _freeze_root_xy(pose, src_pose=src, keep_z_from_edit=True)
            print("[motionfix_edit] post: froze root XY (sit/turn anti-slide)")

    return pose, tlen


def parse_args():
    p = argparse.ArgumentParser(description="MotionFix single-sample edit")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-id", type=str, help="Test-set sample id (e.g. 005454)")
    src.add_argument(
        "--source-npy",
        type=Path,
        help="MotionFix npy with {'pose': (T,135)} or {'motion_source': ...}",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Edit instruction text (optional if --intent-json provides mf_text)",
    )
    p.add_argument(
        "--intent-json",
        type=Path,
        default=None,
        help="MotionEditPlan / agent bundle from scripts/intent/plan_motion_edit.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npy path (default: outputs/motionfix/edit_<id>.npy)",
    )
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    p.add_argument("--smpl", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    p.add_argument("--test-db", type=Path, default=DEFAULT_TEST_DB)
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Diffusion sample steps (0 = model num_train_timesteps, usually 300)",
    )
    p.add_argument(
        "--gd-text",
        type=float,
        default=2.5,
        help="Text CFG scale (MotionFix yaml default 2.5; val often 2.0)",
    )
    p.add_argument(
        "--gd-motion",
        type=float,
        default=None,
        help="Source-motion CFG (default: auto by text; sit/turn~1.5, wave~2.0)",
    )
    p.add_argument(
        "--init-from",
        choices=["noise", "source", "auto"],
        default="auto",
        help="Diffusion init: auto=noise for sit/turn else SDEdit; source=SDEdit; noise=full",
    )
    p.add_argument(
        "--sdedit-strength",
        type=float,
        default=None,
        help="SDEdit noise fraction (default: auto by text; sit~0.85, turn~0.80, wave~0.75)",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=0,
        help="Resample bridged sources shorter than this (0=disable; do not stretch ≥60 clips)",
    )
    p.add_argument(
        "--keep-lower",
        choices=["auto", "on", "off"],
        default="auto",
        help="Keep source root/legs; take upper body from edit (auto: on for wave/clap)",
    )
    p.add_argument(
        "--align-bridged",
        dest="align_bridged",
        action="store_true",
        default=True,
        help="Align bridged --source-npy root/body stats to MotionsFix medians (default on)",
    )
    p.add_argument(
        "--no-align-bridged",
        dest="align_bridged",
        action="store_false",
        help="Disable MotionsFix stats alignment on bridged sources",
    )
    p.add_argument(
        "--edit-mode",
        choices=["auto", "diffusion", "speed", "kinematic"],
        default="auto",
        help="auto: speed→geometric; else MotionsFix diffusion+SDEdit "
        "(kinematic only via --edit-mode kinematic)",
    )
    p.add_argument(
        "--canonicalize",
        action="store_true",
        default=False,
        help="Re-run MotionsFix face-front on --source-npy (debug only; "
        "bridge should already be canonical)",
    )
    p.add_argument(
        "--speed-scale",
        type=float,
        default=None,
        help="Override geometric speed scale (implies --edit-mode speed)",
    )
    p.add_argument(
        "--adapt-bridged",
        dest="adapt_bridged",
        action="store_true",
        default=False,
        help="Amplify body articulation on --source-npy (MLD bridge) before diffusion",
    )
    p.add_argument(
        "--no-adapt-bridged",
        dest="adapt_bridged",
        action="store_false",
        help="Disable bridged-source articulation boost (default)",
    )
    p.add_argument(
        "--raw-text",
        action="store_true",
        help="Do not rewrite absolute captions into MotionFix relative edits",
    )
    p.add_argument(
        "--body-amp",
        type=float,
        default=1.45,
        help="Body axis-angle amplify factor when adapting bridged sources",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--motionfix-root",
        type=Path,
        default=MOTIONFIX_ROOT,
        help="Path to third_party/motionfix",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve user paths before chdir into MotionFix root.
    cwd0 = Path.cwd()
    root = args.motionfix_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MotionFix root not found: {root}")

    def _abs(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (cwd0 / p).resolve()

    args.ckpt = _abs(args.ckpt)
    args.stats = _abs(args.stats)
    args.smpl = _abs(args.smpl)
    args.clip = _abs(args.clip)
    args.test_db = _abs(args.test_db)
    args.source_npy = _abs(args.source_npy)
    args.out = _abs(args.out)

    t0 = time.time()
    _setup_motionfix_env(root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[motionfix_edit] device={device} root={root}")

    # --- Agent plan (Sept architecture: Intent → plan → coder) ---
    intent_plan: Optional[Dict[str, Any]] = None
    if args.intent_json is not None:
        args.intent_json = _abs(args.intent_json)
        raw_plan = json.loads(args.intent_json.read_text(encoding="utf-8-sig"))
        if "motion_edit_plan" in raw_plan:
            intent_plan = raw_plan["motion_edit_plan"]
        elif "mf_text" in raw_plan or "primary_action" in raw_plan:
            intent_plan = raw_plan
        else:
            raise SystemExit(
                f"--intent-json {args.intent_json} missing motion_edit_plan / mf_text"
            )
        cons = intent_plan.get("constraints") or {}
        if intent_plan.get("mf_text"):
            args.text = str(intent_plan["mf_text"])
            args.raw_text = True  # already MotionsFix-native from agent
        if intent_plan.get("edit_mode"):
            args.edit_mode = str(intent_plan["edit_mode"])
        if cons.get("sdedit_strength") is not None and args.sdedit_strength is None:
            args.sdedit_strength = float(cons["sdedit_strength"])
        if cons.get("gd_motion") is not None and args.gd_motion is None:
            args.gd_motion = float(cons["gd_motion"])
        if cons.get("init_from") and args.init_from == "auto":
            args.init_from = str(cons["init_from"])
        if cons.get("keep_lower") in ("on", "off", "auto"):
            args.keep_lower = str(cons["keep_lower"])
        if cons.get("speed_scale") is not None and args.speed_scale is None:
            args.speed_scale = float(cons["speed_scale"])
        if cons.get("align_bridged") is False:
            args.align_bridged = False
        if "freeze_xy" in cons:
            args._agent_freeze_xy = bool(cons["freeze_xy"])  # type: ignore[attr-defined]
        else:
            args._agent_freeze_xy = None  # type: ignore[attr-defined]
        print(
            f"[motionfix_edit] AGENT plan action={intent_plan.get('primary_action')} "
            f"span={intent_plan.get('span')} policy={intent_plan.get('source_policy')} "
            f"mf_text={args.text!r}"
        )
        if intent_plan.get("warning"):
            print(f"[motionfix_edit] AGENT WARNING: {intent_plan['warning']}")
    else:
        args._agent_freeze_xy = None  # type: ignore[attr-defined]

    if not args.text:
        # Speed revise may only set constraints.speed_scale (no MotionsFix caption).
        if args.edit_mode == "speed" or args.speed_scale is not None:
            scale = args.speed_scale if args.speed_scale is not None else 1.0
            args.text = f"speed x{scale:.2f}"
            args.raw_text = True
        elif intent_plan and intent_plan.get("raw_prompt"):
            args.text = str(intent_plan["raw_prompt"])
        else:
            raise SystemExit("Provide --text or --intent-json with mf_text / speed_scale")

    # Prefer agent primary_action over string matching on mf_text.
    raw_prompt = (intent_plan or {}).get("raw_prompt") or args.text
    primary_action = (intent_plan or {}).get("primary_action")
    kin_from_agent = _ACTION_TO_KIN.get(primary_action or "")
    kin_from_user = kin_from_agent or infer_kinematic_intent(raw_prompt)
    user_text = args.text
    edit_text = user_text if args.raw_text else normalize_edit_text(user_text)
    args.text = edit_text

    speed_scale = args.speed_scale
    edit_mode = args.edit_mode
    kinematic_intent: Optional[str] = None
    if speed_scale is not None:
        edit_mode = "speed"
    elif edit_mode == "auto":
        speed_scale = (
            infer_speed_scale(raw_prompt)
            or infer_speed_scale(user_text)
            or infer_speed_scale(args.text)
        )
        edit_mode = "speed" if speed_scale is not None else "diffusion"
    elif edit_mode == "kinematic":
        kinematic_intent = kin_from_user or infer_kinematic_intent(args.text) or "wave"

    if edit_mode == "diffusion":
        if args.sdedit_strength is None:
            args.sdedit_strength = infer_sdedit_strength(args.text)
            print(
                f"[motionfix_edit] auto sdedit-strength={args.sdedit_strength:.2f} "
                f"for text={args.text!r}"
            )
        if args.gd_motion is None:
            args.gd_motion = infer_gd_motion(args.text)
            print(
                f"[motionfix_edit] auto gd-motion={args.gd_motion:.2f} "
                f"for text={args.text!r}"
            )
        if args.init_from == "auto":
            args.init_from = infer_init_from(args.text)
            print(
                f"[motionfix_edit] auto init-from={args.init_from} "
                f"for text={args.text!r}"
            )
    elif args.init_from == "auto":
        args.init_from = "source"
    if args.sdedit_strength is None:
        args.sdedit_strength = 0.55
    if args.gd_motion is None:
        args.gd_motion = 2.5

    do_adapt = bool(args.source_npy) and bool(args.adapt_bridged)
    do_align = bool(args.source_npy) and bool(args.align_bridged)
    # Geometric edits keep original articulation; skip boost/align.
    if edit_mode in ("speed", "kinematic"):
        do_adapt = False
        do_align = False
    if do_align:
        print("[motionfix_edit] bridged MotionsFix path: stats-align + constrained SDEdit")
    datum, sid = load_source_datum(
        args.source_id,
        args.source_npy,
        args.test_db,
        args.smpl,
        adapt_bridged=do_adapt,
        body_amp=args.body_amp,
        canonicalize=bool(args.canonicalize),
        align_bridged=do_align,
    )
    src_len = int(datum["motion_source"]["rots"].shape[0])
    if (
        edit_mode not in ("speed", "kinematic")
        and args.min_frames
        and src_len < int(args.min_frames)
        and args.source_npy is not None
    ):
        datum = dict(datum)
        datum["motion_source"] = _resample_motion_source(
            datum["motion_source"], int(args.min_frames), args.smpl
        )
        datum["motion_target"] = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in datum["motion_source"].items()
        }
        src_len = int(args.min_frames)
    print(
        f"[motionfix_edit] source={sid} frames={src_len} mode={edit_mode} "
        f"adapt={do_adapt}"
    )
    if src_len < 60:
        print(
            f"[motionfix_edit] WARNING: source has only {src_len} frames; "
            "MotionFix is trained on ≥60 (median≈120). Short clips often "
            "yield collapsed / no-op edits. Re-bridge without --max-frames."
        )
    print(f"[motionfix_edit] edit_text={args.text!r}")

    t_load = 0.0
    t_inf0 = time.time()
    joints = None
    edit_meta: Dict[str, Any] = {"edit_mode": edit_mode}
    if edit_mode == "speed":
        if speed_scale is None:
            speed_scale = 0.45
        print(
            f"[motionfix_edit] geometric speed edit x{speed_scale:.3f} "
            "(preserves body pose; avoids T-pose collapse on bridged sources)"
        )
        pose, tlen, geo = geometric_speed_edit(datum, speed_scale, args.smpl)
        joints = geo.get("joint_positions")
        edit_meta.update({k: v for k, v in geo.items() if k != "joint_positions"})
    elif edit_mode == "kinematic":
        assert kinematic_intent is not None
        print(
            f"[motionfix_edit] kinematic semantic edit intent={kinematic_intent!r} "
            "(bridged/OOD sources; MotionsFix diffusion reserved for native ids)"
        )
        pose, tlen, geo = geometric_semantic_edit(
            datum, kinematic_intent, args.smpl
        )
        joints = geo.get("joint_positions")
        edit_meta.update({k: v for k, v in geo.items() if k != "joint_positions"})
    else:
        model = load_model(root, args.ckpt, args.stats, args.smpl, args.clip, device)
        t_load = time.time() - t0
        print(f"[motionfix_edit] model loaded in {t_load:.1f}s  nfeats={model.nfeats}")
        batch = build_batch(datum, args.text, args.stats, args.smpl)
        src_pose135 = motion_source_to_pose135(datum["motion_source"])
        keep_lower = {"auto": None, "on": True, "off": False}[args.keep_lower]
        freeze_xy = getattr(args, "_agent_freeze_xy", None)
        pose, tlen = run_edit(
            model,
            batch,
            args.text,
            steps=args.steps,
            gd_text=args.gd_text,
            gd_motion=args.gd_motion,
            init_from=args.init_from,
            seed=args.seed,
            sdedit_strength=args.sdedit_strength,
            keep_lower=keep_lower,
            freeze_xy=freeze_xy,
            source_pose135=src_pose135,
        )
        pose, wave_rescued = _maybe_boost_wave_arms(pose, args.text, args.smpl)
        did_freeze = (
            bool(freeze_xy)
            if freeze_xy is not None
            else _infer_freeze_xy(args.text)
        )
        edit_meta.update(
            {
                "sdedit_strength": args.sdedit_strength,
                "keep_lower": args.keep_lower,
                "min_frames": args.min_frames,
                "align_bridged": bool(do_align),
                "freeze_xy": did_freeze,
                "wave_rescue": bool(wave_rescued),
                "primary_action": primary_action,
                "agent_span": (intent_plan or {}).get("span"),
            }
        )
        try:
            joints = _pose_npy_to_motion_source(pose, args.smpl)["joint_positions"]
            if torch.is_tensor(joints):
                joints = joints.detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            print(f"[motionfix_edit] joint FK skipped: {exc}")
            joints = None
    t_inf = time.time() - t_inf0

    out = args.out
    if out is None:
        out = (REPO_ROOT / "outputs" / "motionfix" / f"edit_{sid}.npy").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {"pose": pose, "text": args.text, "source_id": sid}
    if joints is not None:
        payload["joint_positions"] = np.asarray(joints, dtype=np.float32)
    np.save(out, payload)
    meta = {
        "out": str(out),
        "source_id": sid,
        "text": args.text,
        "pose_shape": list(pose.shape),
        "frames": tlen,
        "steps": args.steps,
        "gd_text": args.gd_text,
        "gd_motion": args.gd_motion,
        "init_from": args.init_from,
        **edit_meta,
        "load_seconds": round(t_load, 3),
        "infer_seconds": round(t_inf, 3),
        "total_seconds": round(time.time() - t0, 3),
        "device": device,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[motionfix_edit] saved {out} shape={pose.shape}")
    print(f"[motionfix_edit] meta {meta_path}")
    print(
        f"[motionfix_edit] done infer={t_inf:.1f}s total={time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
