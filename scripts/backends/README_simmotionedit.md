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

## Agent integration (open-vocabulary routing)

`scripts/agent/edits.py` holds the closed-set registry the controller is gated
on. An instruction outside that registry is normally rejected in milliseconds
(`unsupported_instruction`) instead of burning a backend pass. SimMotionEdit is
the escape hatch for exactly those instructions, because it consumes the
sentence directly:

```powershell
# 1) make the backend discoverable
$env:SIMMOTION_ROOT = "$PWD\third_party\SimMotionEdit"
$env:SIMMOTION_PYTHON = "C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe"
$env:SIMMOTION_CMD_JSON = '["python","their_single_clip_entry.py","--source","{source}","--text","{text}","--out","{out}"]'

# 2) let the agent route to it (off by default: the benchmark measures the
#    closed set, and a half-installed backend would turn rejections into noise)
$env:AGENT_OPEN_VOCAB_EDIT = "1"
```

With the flag on, `classify()` marks unknown phrasing as `open_vocab: true` and
`tools.edit_motion` sends the raw instruction to `simmotion_edit.run_edit()`.
Everything after the backend call is unchanged: the clip is regrounded, measured
before/after (`preservation`, `instruction_following`, physical validity), and
rolled back if the edit *introduces* damage. The observation carries
`"backend": "simmotion"` so traces and the benchmark can tell the two editors
apart.

Props, camera, appearance and body-shape requests are never routed: no motion
editor can satisfy them.

Checklist:

- `python scripts/backends/simmotion_edit.py --discover-only --source-npy x --text x --out-npy x`
  must report `"ready": true` (repo + `checkpoints/` or `SIMMOTION_CMD_JSON`).
- `python scripts/tests/test_open_vocab.py` exercises the routing with a fake
  backend (10 checks, no GPU): default rejection, routing, prop blocking,
  capability text, and a failing backend reported as failure.
- Without checkpoints the wrapper either errors (`simmotion_infer_failed`) or,
  if you pass `--allow-copy-fallback`, copies the source — the tool reports that
  as `no_change`, never as a successful edit.

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
