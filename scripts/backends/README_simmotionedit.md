# SimMotionEdit (optional editor)

Paper/code: [lzhyu/SimMotionEdit](https://github.com/lzhyu/SimMotionEdit) (CVPR 2025) — text-conditioned motion editing with motion-similarity prediction; MotionFix-compatible stack.

## Why optional

Our default path remains **MLD → MotionsFix** (+ geometric postprocess). SimMotionEdit needs its own conda env, MotionCritic / diffusion checkpoints, and a HumanML3D-style feature layout. Wire it when those are ready; until then MotionsFix + `wave_walk_postprocess` stay the production path.

## Setup

```powershell
cd D:\课件\FYP\BuildanAgent
git clone https://github.com/lzhyu/SimMotionEdit third_party/SimMotionEdit
# Follow their README: MotionFix env, pretrained link, set preproc.sim_file

$env:SIMMOTION_ROOT = "$PWD\third_party\SimMotionEdit"
$env:SIMMOTION_PYTHON = "C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe"
# Official path is batch evaluate (folder=...). For single-clip, set:
# $env:SIMMOTION_CMD_JSON = '["python","motionfix_evaluate.py","folder=..."]'
```

Probe:

```powershell
python scripts/backends/simmotion_edit.py --discover-only `
  --source-npy dummy.npy --text x --out-npy dummy_out.npy
```

Edit (when ready + SIMMOTION_CMD_JSON or ckpts):

```powershell
python scripts/backends/simmotion_edit.py `
  --source-npy outputs/.../02_motionfix_source.npy `
  --text "a person walks forward and waves the right hand" `
  --out-npy outputs/.../03_edited_simmotion.npy
```

Module id (registry): `simmotion_edit` — optional; default session still uses MotionsFix.

## Related: Motion→Text feedback

Automatic textual feedback for revise/LLM refine is **always on** (no SimMotionEdit required):

- Rule-based caption from quality + timeline → `06_motion_caption.json`
- Optional external TM2T: set `MOTION_CAPTION_CMD` (e.g. Motius / MotionGPT m2t when installed)

```powershell
python scripts/session/motion_caption.py `
  --edited outputs/.../03_edited.npy `
  --quality outputs/.../04_quality.json `
  --out outputs/.../06_motion_caption.json
```
