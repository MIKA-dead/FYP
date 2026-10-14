# Third-party patches for local Windows v0.5

Apply after cloning upstream into `third_party/`:

```powershell
Copy-Item -Recurse -Force patches\motion-latent-diffusion\* third_party\motion-latent-diffusion\
Copy-Item -Force patches\motionfix\requirements.windows.txt third_party\motionfix\
```

Contents:
- MLD: CLIP `pooler_output` fix, local SMPL paths, NumPy `float32` prior dtype, walk demo prompt
- MotionFix: relaxed Windows pip requirements

Do not commit HumanML3D/MotionFix datasets or checkpoints here (download separately; see `third_party/INSTALL_STATUS.md`).
