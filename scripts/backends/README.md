# Model backends (v0.5)

Thin wrappers around third-party models. Envs:

| Step | Conda env | Script |
|------|-----------|--------|
| Generate | `mld` | `mld_generate.py` |
| Bridge | `mld` | `mld_to_motionfix.py` |
| Edit | `motionfix` | `motionfix_edit.py` |
| Preview | either | `visualize_motion.py` |
| Full pipe | host Python 3 | `v05_pipeline.py` (spawns both envs) |

See `third_party/INSTALL_STATUS.md` for install status.

## End-to-end (recommended)

```powershell
cd D:\课件\FYP\BuildanAgent
python scripts/backends/v05_pipeline.py `
  --text "a person walks forward." `
  --edit-text "a person walks forward slowly" `
  --length 60 `
  --skip-review `
  --bridge-iters 40 `
  --edit-steps 50
```

Omit `--skip-review` to pause after generate / bridge / edit (human check). Previews are GIFs under `outputs/v05/run_*/`.

Reuse an existing MLD npy:

```powershell
python scripts/backends/v05_pipeline.py `
  --joints third_party/motion-latent-diffusion/results/mld/.../Example_60_batch0_0.npy `
  --text "a person walks forward." `
  --edit-text "a person walks forward slowly" `
  --skip-generate --skip-review
```

## MotionFix single edit

```powershell
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe `
  scripts/backends/motionfix_edit.py `
  --source-id 005454 `
  --text "a person walks forward slowly" `
  --out outputs/motionfix/edit_005454.npy
```

- `--source-id` from test set, or `--source-npy` with `{'pose': (T,135)}` / `{'motion_source': ...}`
- Defaults: `--steps 50`, `--gd-text 2.0`, `--gd-motion 2.0`

## Bridge only

```powershell
C:\Users\ALIENWARE\miniconda3\envs\mld\python.exe `
  scripts/backends/mld_to_motionfix.py `
  --joints path\to\joints.npy `
  --out outputs/v05/source.npy `
  --iters 40
```

## Visualize

```powershell
python scripts/backends/visualize_motion.py --input outputs/v05\...\01_mld_joints.npy --out preview.gif
```

Env vars (optional): `MLD_PYTHON`, `MOTIONFIX_PYTHON`.
