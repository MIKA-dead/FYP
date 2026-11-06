"""
TM2T motion-to-text inference for one edited clip (pipeline v1, item a).

Takes the 03_edited.npy the session already produced, converts its joints to the
HumanML3D 263-d feature the TM2T checkpoints expect, tokenises it with the VQ-VAE
tokenizer, and decodes a sentence with the M2T transformer (beam search).

  python scripts/backends/tm2t_infer.py \\
      --npy outputs/session/ui_runs/xxx/03_edited.npy \\
      --out outputs/session/ui_runs/xxx/06_motion_caption.tm2t.json

The checkpoint is expected under third_party/TM2T/checkpoints (see README there).
TM2T_ROOT overrides the repo location.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# TM2T was written for numpy 1.x / torch 1.x. Keep the upstream tree untouched
# and patch the two removals on import instead.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

J_PELVIS = 0
J_HEAD = 15

M2T_NAME = "M2T_EL4_DL4_NH8_PS"
TOKENIZER_NAME = "VQVAEV3_CB1024_CMT_H1024_NRES3"
CODEBOOK = 1024
MAX_MOTION_TOKEN = 55
MAX_MOTION_FRAME = 196
FEET_THRE = 0.002


def _tm2t_root() -> Path:
    return Path(os.environ.get("TM2T_ROOT") or (REPO / "third_party" / "TM2T")).resolve()


def checkpoints_dir() -> Path:
    return Path(os.environ.get("TM2T_CHECKPOINTS") or (_tm2t_root() / "checkpoints")).resolve()


def m2t_ready() -> Tuple[bool, str]:
    ck = checkpoints_dir() / "t2m"
    need = [
        ck / TOKENIZER_NAME / "model" / "finest.tar",
        ck / TOKENIZER_NAME / "meta" / "mean.npy",
        ck / TOKENIZER_NAME / "meta" / "std.npy",
        ck / M2T_NAME / "model" / "finest.tar",
        _tm2t_root() / "glove" / "our_vab_data.npy",
    ]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return False, "missing " + ", ".join(missing[:3])
    return True, "ok"


def _load_joints(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        obj = raw.item()
        joints = obj.get("joint_positions") if isinstance(obj, dict) else None
        if joints is None:
            raise SystemExit(f"no joint_positions in {path}")
    else:
        joints = raw
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] != 22:
        raise SystemExit(f"expected (T,22,3) joints, got {joints.shape}")
    return joints


def _up_axis(joints: np.ndarray) -> int:
    hp = joints[:, J_HEAD] - joints[:, J_PELVIS]
    return 2 if abs(float(hp[:, 2].mean())) > abs(float(hp[:, 1].mean())) else 1


def to_y_up(joints: np.ndarray) -> Tuple[np.ndarray, bool]:
    """HumanML3D/TM2T expect Y-up; the session's edited clips are Z-up."""
    if _up_axis(joints) == 1:
        return joints, False
    out = np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)
    return out.astype(np.float32), True


def _import_tm2t():
    root = _tm2t_root()
    if not (root / "networks").is_dir():
        raise SystemExit(f"TM2T repo not found at {root}")
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from common.skeleton import Skeleton
    from networks.modules import VQEncoderV3
    from networks.quantizer import Quantizer
    from networks.transformer import TransformerV2
    from networks.translator import Translator
    from scripts import motion_process
    from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
    from utils.word_vectorizer import WordVectorizerV2

    # TM2T's motion_process keeps the HumanML3D constants inside its
    # ``if __name__ == "__main__"`` block, so importing it leaves them undefined.
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    motion_process.l_idx1, motion_process.l_idx2 = 5, 8
    motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.r_hip, motion_process.l_hip = 2, 1
    motion_process.joints_num = 22
    if not hasattr(motion_process, "tgt_offsets"):
        sample = _hml_sample_joints()
        sample_t = torch.from_numpy(sample[0]).float()
        tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
        motion_process.tgt_offsets = tgt_skel.get_offsets_joints(sample_t)

    return {
        "torch": torch,
        "VQEncoderV3": VQEncoderV3,
        "Quantizer": Quantizer,
        "TransformerV2": TransformerV2,
        "Translator": Translator,
        "process_file": motion_process.process_file,
        "WordVectorizerV2": WordVectorizerV2,
    }


def _hml_sample_joints() -> np.ndarray:
    root = REPO / "HumanML3D" / "HumanML3D" / "new_joints"
    preferred = root / "000000.npy"
    path = preferred if preferred.is_file() else next(iter(sorted(root.glob("*.npy"))), None)
    if path is None:
        raise SystemExit(f"no HumanML3D joints under {root}")
    joints = np.load(path)
    return np.asarray(joints, dtype=np.float32)


def _features(joints: np.ndarray, mean: np.ndarray, std: np.ndarray, torch, process_file) -> np.ndarray:
    pose = np.asarray(joints, dtype=np.float64).copy()
    feats = process_file(pose, FEET_THRE)[0]
    feats = np.asarray(feats, dtype=np.float32)
    feats = (feats - mean) / std
    if feats.shape[0] > MAX_MOTION_FRAME:
        feats = feats[:MAX_MOTION_FRAME]
    # The VQ tokenizer downsamples by 4 per stage (n_down=2 → 4 frames per token).
    unit = 4
    pad = (-feats.shape[0]) % unit
    if pad:
        feats = np.concatenate([feats, np.zeros((pad, feats.shape[1]), dtype=np.float32)], axis=0)
    if feats.shape[0] < 40:
        feats = np.concatenate(
            [feats, np.zeros((40 - feats.shape[0], feats.shape[1]), dtype=np.float32)], axis=0
        )
    return feats


_MODELS: Optional[Dict[str, Any]] = None


def _load_models() -> Dict[str, Any]:
    """Load the VQ tokenizer + M2T transformer once per process."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    ready, why = m2t_ready()
    if not ready:
        raise SystemExit(f"TM2T not ready: {why}")

    mods = _import_tm2t()
    torch = mods["torch"]
    ck = checkpoints_dir() / "t2m"

    mean = np.load(ck / TOKENIZER_NAME / "meta" / "mean.npy").astype(np.float32)
    std = np.load(ck / TOKENIZER_NAME / "meta" / "std.npy").astype(np.float32)

    device = torch.device(os.environ.get("TM2T_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))

    enc_channels = [1024, 1024]
    vq_encoder = mods["VQEncoderV3"](263 - 4, enc_channels, 2)
    quantizer = mods["Quantizer"](CODEBOOK, 1024, 1.0)
    tok_ck = torch.load(
        ck / TOKENIZER_NAME / "model" / "finest.tar", map_location="cpu", weights_only=False
    )
    vq_encoder.load_state_dict(tok_ck["vq_encoder"])
    quantizer.load_state_dict(tok_ck["quantizer"])
    vq_encoder.to(device).eval()
    quantizer.to(device).eval()

    w_vectorizer = mods["WordVectorizerV2"](str(_tm2t_root() / "glove"), "our_vab")
    n_txt_vocab = len(w_vectorizer) + 1
    _, _, txt_start_idx = w_vectorizer["sos/OTHER"]
    _, _, txt_end_idx = w_vectorizer["eos/OTHER"]
    txt_pad_idx = len(w_vectorizer)

    mot_start = CODEBOOK
    mot_end = CODEBOOK + 1
    mot_pad = CODEBOOK + 2

    m2t = mods["TransformerV2"](
        CODEBOOK + 3,
        mot_pad,
        n_txt_vocab,
        txt_pad_idx,
        d_src_word_vec=512,
        d_trg_word_vec=512,
        d_model=512,
        d_inner=2048,
        n_enc_layers=4,
        n_dec_layers=4,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_src_position=100,
        n_trg_position=50,
        trg_emb_prj_weight_sharing=True,
    )
    m2t_ck = torch.load(ck / M2T_NAME / "model" / "finest.tar", map_location="cpu", weights_only=False)
    m2t.load_state_dict(m2t_ck["m2t_transformer"])
    m2t.to(device).eval()

    _MODELS = {
        "torch": torch,
        "mean": mean,
        "std": std,
        "device": device,
        "process_file": mods["process_file"],
        "vq_encoder": vq_encoder,
        "quantizer": quantizer,
        "m2t": m2t,
        "w_vectorizer": w_vectorizer,
        "txt_start_idx": txt_start_idx,
        "txt_end_idx": txt_end_idx,
        "txt_pad_idx": txt_pad_idx,
        "mot_start": mot_start,
        "mot_end": mot_end,
        "mot_pad": mot_pad,
        "Translator": mods["Translator"],
    }
    return _MODELS


def describe(npy_path: Path, *, beam_size: int = 2) -> Dict[str, Any]:
    M = _load_models()
    torch = M["torch"]
    joints = _load_joints(npy_path)
    joints_y, converted = to_y_up(joints)
    feats = _features(joints_y, M["mean"], M["std"], torch, M["process_file"])

    with torch.no_grad():
        motion = torch.from_numpy(feats).float().unsqueeze(0).to(M["device"])
        pre_latents = M["vq_encoder"](motion[..., :-4])
        raw_idx = M["quantizer"].map2index(pre_latents)
        arr = raw_idx.detach().cpu().numpy() if hasattr(raw_idx, "detach") else np.asarray(raw_idx)
        if arr.ndim > 1:
            arr = arr[0]
        indices = [int(x) for x in arr.reshape(-1).tolist()]

    tokens = [M["mot_start"]] + indices[: MAX_MOTION_TOKEN - 2] + [M["mot_end"]]
    tokens += [M["mot_pad"]] * (MAX_MOTION_TOKEN - len(tokens))
    src = torch.tensor([tokens], dtype=torch.long, device=M["device"])

    translator = M["Translator"](
        M["m2t"],
        beam_size=beam_size,
        max_seq_len=30,
        src_pad_idx=M["mot_pad"],
        trg_pad_idx=M["txt_pad_idx"],
        trg_sos_idx=M["txt_start_idx"],
        trg_eos_idx=M["txt_end_idx"],
    ).to(M["device"])
    pred = translator.translate_sentence(src)
    words = []
    for idx in pred:
        if idx in (M["txt_start_idx"], M["txt_end_idx"], M["txt_pad_idx"]):
            continue
        words.append(M["w_vectorizer"].itos(int(idx)))

    return {
        "schema_version": "m2t_tm2t_v1",
        "model": M2T_NAME,
        "tokenizer": TOKENIZER_NAME,
        "device": str(M["device"]),
        "source_npy": str(npy_path),
        "frames": int(joints.shape[0]),
        "converted_to_y_up": bool(converted),
        "motion_tokens": len(indices),
        "caption": " ".join(words).strip(),
        "raw_tokens": pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TM2T motion → text for one clip")
    ap.add_argument("--npy", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--beam-size", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="Only report readiness")
    args = ap.parse_args()

    if args.check:
        ready, why = m2t_ready()
        print(json.dumps({"ready": ready, "detail": why}, indent=2))
        return
    if args.npy is None:
        raise SystemExit("--npy is required unless --check is used")

    out = describe(args.npy, beam_size=args.beam_size)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"caption": out["caption"], "source": out["model"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
