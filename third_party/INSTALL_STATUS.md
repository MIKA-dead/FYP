# MLD + MotionFix 本机安装状态

**Date:** 2026-10-11  
**Machine:** Windows + RTX 5060 Laptop (8GB) + CUDA driver 13.2  
**PyTorch:** 2.11.0+cu128（官方旧版 torch 不支持 Blackwell，已改用 cu128）

## 已完成

| 组件 | 路径 / 环境 | 状态 |
|------|-------------|------|
| Git / Git-LFS / Miniconda | 系统已装 | OK |
| MotionFix repo | `third_party/motionfix` | OK |
| MLD repo | `third_party/motion-latent-diffusion` | OK |
| conda env `motionfix` | `conda activate motionfix` | GPU OK |
| conda env `mld` | `conda activate mld` | GPU OK |
| MotionFix CLIP | `motionfix/deps/clip-vit-large-patch14` | OK |
| MotionFix dataset | `motionfix/data/motionfix` | OK |
| MotionFix eval-deps | `motionfix/eval-deps` | OK |
| MotionFix SMPL-H | `motionfix/data/body_models/smplh` | OK |
| MotionFix TMED ckpt | `motionfix/experiments/tmed/checkpoints/last.ckpt` (~1.9GB) | OK |
| MLD CLIP / SMPL / t2m / glove | `motion-latent-diffusion/deps/*` | OK |
| MLD pretrained ckpt | `motion-latent-diffusion/models/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt` | OK |

## 激活方式

新开 PowerShell：

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
# 重启终端后：
conda activate motionfix   # 或 mld
```

不 init 时可直接用解释器：

```powershell
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe
C:\Users\ALIENWARE\miniconda3\envs\mld\python.exe
```

## 冒烟验证（已通过）

```powershell
# 两端 GPU
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
C:\Users\ALIENWARE\miniconda3\envs\mld\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期：`2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU`

## MLD walk 已打通（2026-10-11）

- 从 HuggingFace `QuangNguyen22/humanml3d` 替换了正确数字 ID 的 `new_joint_vecs` / `new_joints`（29228 条）
- 兼容 patch：`mld/models/architectures/mld_clip.py`（transformers 5.x 的 CLIP 输出）
- 成功命令：

```powershell
conda activate mld
cd D:\课件\FYP\BuildanAgent\third_party\motion-latent-diffusion
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/walk_example.txt
```

输出示例：`results/mld/.../Example_60_batch0_0.npy`，shape `(60, 22, 3)`。

可删以省空间：`HumanML3D/_hf_download/`、`HumanML3D/_bad_joints_backup/`。

## 后续

### 2) MotionFix 单条编辑推理 — 已打通（2026-10-11）

接口：scripts/backends/motionfix_edit.py

`powershell
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe 
  scripts/backends/motionfix_edit.py 
  --source-id 005454 
  --text "a person walks forward slowly" 
  --out outputs/motionfix/edit_005454.npy
`

冒烟结果：outputs/motionfix/edit_005454.npy shape (120, 135)，推理 ~1s（50 steps）。

### 3) 表示桥接 + v0.5 端到端 — 已打通（2026-10-14）

- 桥接：scripts/backends/mld_to_motionfix.py（SMPLify joints→MotionFix source）
- 可视化：scripts/backends/visualize_motion.py（matplotlib GIF，无需 Xvfb）
- 流水线：scripts/backends/v05_pipeline.py（MLD → 人工审核 → bridge → MotionFix → 审核）
- 冒烟输出：outputs/v05/smoke_e2e/（含 preview GIF）

`powershell
python scripts/backends/v05_pipeline.py 
  --text "a person walks forward." 
  --edit-text "a person walks forward slowly" 
  --skip-review --bridge-iters 40 --edit-steps 50
`

### 4) v1（未做）
VLM/critic 替代人工审核。

## 备注

- 两仓用**独立 conda env**，不要混装。
- MotionFix license：非商业科研用途。
- Windows 上官方 bash 脚本未直接跑，依赖已用 `gdown` / `pip` 等价安装。
