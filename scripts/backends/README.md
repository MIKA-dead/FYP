# Model backends (v0.5)

Thin wrappers around third-party models. Envs:

| Step | Conda env | Script |
|------|-----------|--------|
| Generate | `mld` | `mld_generate.py` |
| Bridge | `mld` | `mld_to_motionfix.py` |
| Edit | `motionfix` | `motionfix_edit.py` |
| Preview | either | `visualize_motion.py` |
| Diagnose | `motionfix` | `diagnose_v05.py` |
| Full pipe | host Python 3 | `v05_pipeline.py` (spawns both envs) |

See `third_party/INSTALL_STATUS.md` for install status.

## End-to-end (recommended)

Sept session flow: **prompt → Intent → modules → generate → user revise → improve**.

```powershell
cd D:\课件\FYP\BuildanAgent

# 1) Generate (agent picks mld_text / MotionsFix-native when span is large)
python scripts/backends/v05_pipeline.py `
  --mode generate `
  --prompt "a person sits down" `
  --length 60 `
  --skip-review `
  --run-dir outputs/v05/session_sit/gen

# 2) User improve (only modules_to_rerun; bg/camera/appearance frozen)
python scripts/backends/v05_pipeline.py `
  --mode revise `
  --previous-run outputs/v05/session_sit/gen `
  --prompt "坐下姿势不对，改成真正坐下，不要跪着" `
  --skip-review `
  --run-dir outputs/v05/session_sit/rev1
```

Agent writes `00_agent_bundle.json` + `00_motion_edit_plan.json`. Large-span actions (sit) auto-switch to MotionsFix-native `preferred_source_id`. Pass `--force-bridged-edit` only for ablation.

**UI-facing session API** (preferred over raw v05 flags):

```powershell
python scripts/session/session_api.py --prompt "a person sits down" --mode generate
python scripts/session/run_demos.py          # validate sit/wave/turn/slow + handoff_for_ui.json
python scripts/session/chat_server.py        # Cursor-like UI → http://127.0.0.1:8787
```

Contract: `outputs/session/handoff_for_ui.json`. Each run writes `session_response.json` + `04_quality.json` + `06_motion_caption.json` (automatic M2T feedback for revise/LLM).

**Motion→text feedback:** rule-based caption from quality/timeline (always on). Optional external TM2T via `MOTION_CAPTION_CMD`. See `scripts/session/motion_caption.py`.

**Optional stronger editor:** SimMotionEdit scaffold — `scripts/backends/simmotion_edit.py` + `README_simmotionedit.md` (not required for default MotionsFix path).

**MotionFix length:** keep bridged clips at **≥60 frames** (MLD `--length 60` is enough). Do **not** truncate with `--bridge-max-frames 24`.

**Gravity axis:** MLD/SMPL is Y-up; MotionFix is Z-up. `mld_to_motionfix.py` converts Y→Z after the fit, then MotionsFix face-front. Previews auto-detect Z-up.

**Semantic quality contract:** MotionsFix diffusion only works well when the bridged npy is MotionsFix-canonical (`motionfix_canonical=true` via face-front). Do **not** re-canonicalize on `--source-npy` load (that twists yaw ~150° and makes edits OOD). Use `diagnose_v05.py` gates before judging GIFs.

**Step-3 edits (`--edit-mode auto`):**
- **Speed** (`slowly` / `faster`): geometric root-speed scaling (pose preserved).
- **Semantic (`auto`)**: MotionsFix diffusion for both bridged npy and native ids.
  - Bridged: **stats-align** (root step + body |aa| → MotionsFix medians) → relative text → moderate SDEdit.
  - Wave: `keep_lower` (source legs/root, edit arms). Sit/turn: **freeze root XY** after diffusion (anti-slide).
  - Fallback ablation: `--edit-mode kinematic` (geometric overlays only).
- Latest MotionsFix constrained demos: `outputs/v05/run_semantic_v9/`.

**Diagnose:** `python scripts/backends/diagnose_v05.py --source-npy outputs/v05/.../02_motionfix_source.npy`

Latest semantic demos: `outputs/v05/run_semantic_v7/` (contract fix + stronger MotionsFix edit params).

## MotionFix single edit

```powershell
# Agent plan first, then MotionsFix coder
python scripts/intent/plan_motion_edit.py `
  --prompt "a person walks forward and waves with the right hand" `
  --source-hint bridged_walk `
  --out-plan outputs/v05/run_semantic_v10/00_wave_plan.json

C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe `
  scripts/backends/motionfix_edit.py `
  --source-npy outputs/v05/run_semantic_v7/02_motionfix_source.npy `
  --intent-json outputs/v05/run_semantic_v10/00_wave_plan.json `
  --out outputs/v05/run_semantic_v10/03_wave.npy `
  --edit-mode auto

# Native MotionsFix id control
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe `
  scripts/backends/motionfix_edit.py `
  --source-id 001318 `
  --text "wave to the side" `
  --out outputs/v05/run_semantic_v7/val_mf001318.npy `
  --edit-mode diffusion

# Ablation only: kinematic overlay
C:\Users\ALIENWARE\miniconda3\envs\motionfix\python.exe `
  scripts/backends/motionfix_edit.py `
  --source-npy outputs/v05/run_semantic_v7/02_motionfix_source.npy `
  --text "sit down" `
  --out outputs/v05/run_semantic_v7/ablate_sit_kin.npy `
  --edit-mode kinematic
```

- `--source-id` from test set, or `--source-npy` with `{'pose': (T,135)}` / `{'motion_source': ...}`
- Defaults: `--edit-mode auto`, `--init-from source` (SDEdit auto-strength), `--gd-text 2.5`, `--gd-motion` auto, `--keep-lower off`

## Bridge only

```powershell
C:\Users\ALIENWARE\miniconda3\envs\mld\python.exe `
  scripts/backends/mld_to_motionfix.py `
  --joints path\to\joints.npy `
  --out outputs/v05/source.npy `
  --iters 40
```

Bridge applies mild root-speed pull, light temporal smooth, Y→Z, then MotionsFix face-front (`MOTIONFIX_PYTHON`).

## Visualize

```powershell
python scripts/backends/visualize_motion.py --input path\to.npy --out preview.gif
```
