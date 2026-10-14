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


def load_source_datum(
    source_id: Optional[str],
    source_npy: Optional[Path],
    test_db: Path,
    smpl_path: Path,
) -> Tuple[Dict[str, Any], str]:
    """Return (datum with motion_source/target/text/id/split, resolved_id)."""
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
    sid = source_npy.stem
    datum = {
        "motion_source": ms,
        "motion_target": {k: v.clone() if torch.is_tensor(v) else v for k, v in ms.items()},
        "text": str(obj.get("text", "")),
        "id": sid,
        "split": 2,
    }
    return datum, sid


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


def run_edit(
    model,
    batch,
    edit_text: str,
    steps: int,
    gd_text: float,
    gd_motion: float,
    init_from: str,
    seed: int,
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

    source_init = source_mot_pad if init_from == "source" else None
    diffusion_process = create_diffusion(
        timestep_respacing=None,
        learn_sigma=False,
        sigma_small=True,
        diffusion_steps=steps,
        noise_schedule=model.diff_params.noise_schedule,
        predict_xstart=False
        if model.diff_params.predict_type == "noise"
        else True,
    )

    with torch.no_grad():
        diffout = model.generate_motion(
            text,
            source_mot_pad,
            mask_source,
            mask_target,
            diffusion_process,
            init_vec=source_init,
            init_vec_method=init_from,
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
    p.add_argument("--text", type=str, required=True, help="Edit instruction text")
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
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--gd-text", type=float, default=2.0)
    p.add_argument("--gd-motion", type=float, default=2.0)
    p.add_argument(
        "--init-from",
        choices=["noise", "source"],
        default="noise",
        help="Diffusion init (source = SDEdit-style)",
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

    model = load_model(root, args.ckpt, args.stats, args.smpl, args.clip, device)
    t_load = time.time() - t0
    print(f"[motionfix_edit] model loaded in {t_load:.1f}s  nfeats={model.nfeats}")

    datum, sid = load_source_datum(
        args.source_id, args.source_npy, args.test_db, args.smpl
    )
    src_len = int(datum["motion_source"]["rots"].shape[0])
    print(f"[motionfix_edit] source={sid} frames={src_len}")
    print(f"[motionfix_edit] edit_text={args.text!r}")

    batch = build_batch(datum, args.text, args.stats, args.smpl)
    t_inf0 = time.time()
    pose, tlen = run_edit(
        model,
        batch,
        args.text,
        steps=args.steps,
        gd_text=args.gd_text,
        gd_motion=args.gd_motion,
        init_from=args.init_from,
        seed=args.seed,
    )
    t_inf = time.time() - t_inf0

    out = args.out
    if out is None:
        out = (REPO_ROOT / "outputs" / "motionfix" / f"edit_{sid}.npy").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {"pose": pose, "text": args.text, "source_id": sid}
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
