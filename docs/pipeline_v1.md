# Pipeline v1 — closed loop with motion-to-text and an LLM controller

Pipeline 0.5 was "generate one clip, then edit it once". Pipeline v1 closes the
loop: after every edit the clip is described in text again, and an LLM compares
that description with the one before the edit and writes the next edit prompt.

```
generate (T2M / dataset draft)
      │
      ▼
edit 03_edited.npy  ──►  06_motion_caption.json   (M2T model, TM2T)
      ▲                        │
      │                        ▼
      └──── next edit prompt ◄─ LLM controller
             (07_loop.json)     compares caption_before / caption_after + metrics
```

## What was added

| File | Role |
|---|---|
| `scripts/backends/tm2t_infer.py` | Real M2T: joints `(T,22,3)` → HumanML3D 263-d features → VQ tokens → TM2T M2T transformer (beam search) → sentence |
| `scripts/session/motion_caption.py` | Backend selection: `tm2t` → `external` (`MOTION_CAPTION_CMD`) → metric rules; writes `06_motion_caption.json` |
| `scripts/intent/edit_proposer.py` | LLM controller: goal + caption before/after + measured problems → `{done, verdict, reason, next_edit_prompt}`; deterministic `rule_proposal` fallback |
| `scripts/orchestrator/run_loop.py` | The loop: describe → propose → revise → describe, with rollback and stop rules |
| `scripts/session/chat_server.py` | `POST /api/iterate` |
| `scripts/session/ui/index.html` | `自动迭代` button (uses the current run + the prompt box as the goal) |

The M2T checkpoints are the TM2T HumanML3D release (M2T_EL4_DL4_NH8_PS +
VQVAEV3 tokenizer), unpacked under `third_party/TM2T/checkpoints`. The upstream
tree is used unpatched; `tm2t_infer.py` restores the HumanML3D skeleton constants
that TM2T keeps inside its `__main__` block and maps `np.float` for numpy 2.

## Running it

```powershell
# loop over an existing run
python scripts/orchestrator/run_loop.py ^
  --run-dir outputs/session/ui_runs/<run> ^
  --goal "a person walks forward and waves with one arm" ^
  --max-iters 3

# generate the first clip inside the loop
python scripts/orchestrator/run_loop.py ^
  --prompt "a person sits down" ^
  --goal "a person sits down from standing without walking first" ^
  --max-iters 2
```

In the page (`http://127.0.0.1:8787`): send one prompt, then put the target in
the input box and press `自动迭代`. The chat prints, per round, the caption
before the edit, the proposed next prompt, the caption after it, and the score.

## Artifacts

`06_motion_caption.json`

```json
{
  "source": "tm2t_m2t_transformer",
  "caption": "a person jump and spin in the air",
  "m2t": {"model": "M2T_EL4_DL4_NH8_PS", "frames": 83, "caption": "..."},
  "problems": ["..."],
  "critique": "User wanted: ... Observed (TM2T): ... Issues: ..."
}
```

`<start_run>/07_loop/07_loop.json` records every round: `caption_before`,
`proposal`, `session`, `caption_after`, scores, `result`, `stopped`.

## Stop and safety rules

- proposer says `done` → stop, keep the current clip.
- caption unchanged twice in a row → stop.
- an edit that lowers the quality score by more than 5 → **roll back**: the loop
  keeps the previous clip and records `result: "regressed"`.
- a failed session (no `03_edited.npy`, or `edit_not_applied`) → stop and keep the
  previous clip.
- otherwise stop at `--max-iters`.
- `--require-llm` turns any fallback or LLM error into a hard failure, so an
  LLM-verification run can never be silently recorded as a rules run.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `TM2T_ROOT` | `third_party/TM2T` | TM2T checkout |
| `TM2T_CHECKPOINTS` | `third_party/TM2T/checkpoints` | checkpoint folder |
| `TM2T_DEVICE` | auto | `cpu` / `cuda` |
| `M2T_BACKEND` | `auto` | `tm2t`, `external`, `rules`, `auto` |
| `MOTION_CAPTION_CMD` | — | fallback command template `{npy}` / `{out}` |
| `REFINE_LLM_*` | opencode auth | same client as `llm_refine` |

## Verified on 2026-11-02

M2T (TM2T, beam 2) on existing clips:

| clip | caption |
|---|---|
| jump run | `a person jump and spin in the air` |
| wave run | `a person wave with their right hand` |
| kick run | `a person step forward with their right foot` |
| sit run | `a person sit down on a stool` |
| walk run | `a person slowly walk forward` |

Loop runs (log: `outputs/session/prompt_logs/20261102-pipeline-v1/log.json`):

- **wave clip, real LLM, 3 iterations allowed** — the round that shows the loop
  working end to end. Caption before: `a person wave with their right hand`.
  The LLM noticed "near-zero root travel" and proposed `walk forward while
  waving`. The edit ran, and M2T described the result as `a person step forward
  with their right foot and reach out with their right hand to grab something` —
  a *different* motion, but not the goal. The gate scored it 75, so the loop
  rolled back to the 100-point wave clip.
- walk clip, real LLM, goal "walks slowly and raises the right arm high": the LLM
  proposed `raise the right arm high overhead`, the `walk_slow` editor answered
  `edit_not_applied`, loop stopped and kept the original clip.
- walk clip, page `自动迭代` on the rules fallback: one round, proposal
  `raise the arm higher`, edit scored 70 → rolled back.
- sit clip, goal "sits down without walking first": the edit scored 75 →
  rolled back, final caption `a person sit down on a stool`.
- a re-used iteration directory was found to hand a stale `03_edited.npy` to the
  loop; iteration dirs are now unique per attempt and a failed session is not
  adopted.

## Geometric arm layer (fix for the editor bottleneck, 2026-11-02 22:50)

`walk_slow` could not raise an arm: the planner sent "raise the right arm high"
down the MLD-rebuild → MotionsFix diffusion path (12 min, then
`edit_not_applied`), or mis-planned it as a speed edit. The arm raise is now a
geometric operator on the clip already on screen, so it costs ~20 s and cannot
destroy the rest of the motion.

| Piece | What changed |
|---|---|
| `scripts/backends/constraint_postprocess.py` | new `raise_arm()`: unfolds the elbow by `straighten` (upper-arm / forearm lengths preserved), then rotates elbow+wrist about the shoulder until the wrist reaches the target height. Targets are 0.8× / 1.2× / 1.8× / 2.6× head-above-shoulder for `slightly` / `mid` / `high` / `overhead` (2.6× = arm straight up, what the wave clips already show), clamped to 94 % of anatomical reach and ramped in over ~1 s. Supports `side: left\|right\|both` and `step_up` for "a bit higher". The swing plane uses the shoulder line, not shoulder-minus-pelvis: with an upright torso the latter's horizontal part is noise and the rotation axis flipped between two frames |
| `scripts/intent/refine_schema.py` | `arm_raise` in `OPS`; `_arm_raise_from_text()` parses EN/中文 ("raise the right arm high", "抬右手过头顶", "举双手过头顶", "higher/再高") → `{side, level, step_up}` and forces `geometric_only`. An arm raise is local, so trims / phase drops are stripped from the refine (`ops=[geometric_only, arm_raise]`, `phases=[full]`) |
| `scripts/intent/plan_motion_edit.py` | `refine.arm_raise` wins over the wave-rebuild, dataset and sit-hold branches: `edit_mode=geometric_only`, `source_policy=reuse_previous_edited`, no MLD, no diffusion, no trim |
| `scripts/orchestrator/run_session.py` | skips the wave-walk rebuild and the "hop to last good sit" seed when an arm raise is requested, so the seed is exactly the previous clip |
| `scripts/session/quality_gate.py` | new hard check `arm_raise_applied` + metrics `arm_raise_lift`, `arm_raise_ratio` (wrist above shoulder ÷ head above shoulder; ≈1.0 = at head, ≈1.8 = clearly above, ≈2.6 = straight up) |
| `scripts/intent/edit_proposer.py` | the metrics above are passed to the LLM, and the prompt says the caption's vocabulary is coarse ("wave" for any raised arm) so the measured ratio wins |

Verified (walk clip, `walk_slow`, real LLM, ~15 s total):

| round | prompt | caption after | score | arm_raise_ratio |
|---|---|---|---|---|
| 1 | `raise the right arm high overhead` (from M2T walk caption) | `a person walk forward and wave with their right hand` | 100 | 2.59 |
| 2 | proposer: `goal_met` — *"the measured arm_raise_ratio of 2.59 confirms the wrist is clearly overhead"* | — | — | — |

Test battery (2026-11-05, `scripts/tests/test_arm_raise.py`, 33/33):
16 parser cases (EN/中文, negatives such as "slow down a bit" and "lift the right
leg as much as the left", plus the `phase_speed` wildcard fallback when the LLM
names the op but omits the map), 10 geometry cases (walk / run / bow / sit / wave,
left / right / both, idempotence, `high → higher → higher` monotone 1.8 → 2.15 →
2.5, bone length, root path and the untouched arm bit-identical, no snap beyond
the source clip's own cuts), and 7 end-to-end revises (E1–E7, all score 100,
8–10 s each; the two-turn chain costs 17.5 s; the speed regression goes 100 → 125
frames and the caption becomes "a person slowly walk forward"). Artifacts:
`outputs/tests/arm_raise_20261105_final/report.md`, `results.json`,
`contact_sheet.png`, `compare_walk_right_overhead.gif`; loop run in
`outputs/tests/loop_arm/`.

Before the fix, the same goal ended in `edit_not_applied` (score 75, rollback)
after a 12-minute MLD rebuild. Artifacts:
`outputs/session/ui_runs/9ddc04a5_UI_20261102-221627/07_loop/iter01_225040/`.

## Known limits

- ~~**Editor, not the LLM, is the current bottleneck.**~~ Fixed for the arm case
  on 2026-11-02: `raise_arm` in `constraint_postprocess.py` applies absolute limb
  raises geometrically (see the section above). Still open: arbitrary upper-body
  gestures (waving with a specific rhythm, grabbing, pointing) and edits that
  need the whole body to reorganise (sit ↔ walk). Those still need SimMotionEdit
  or a retarget stage, and the planner will still fall back to a dataset swap.
- The caption comes from HumanML3D-scale joints; clips that are far from the
  training distribution (e.g. the geometric bow) are described approximately.
  TM2T also has no height words — a raised arm is always "wave", which is why the
  proposer now reads `arm_raise_ratio` instead of trusting the caption alone.

## LLM gateway note (resolved 2026-11-02 22:25)

The `Model is unavailable` errors were **not** an account or permission problem.
`deepseek-v4-flash` — the model `llm_refine` defaulted to for `opencode.ai`
bases — was retired from the OpenCode Go catalog and is absent from
`opencode models opencode-go`, so the Go relay rejected it for every request.

Fix: the default is now `glm-5.3-flash` (`scripts/intent/llm_refine.py`), which
the account's catalog lists and which answered a real request. Override with
`REFINE_LLM_MODEL` when the catalog rotates again; `opencode models opencode-go`
is the authoritative list. `deepseek-v4-pro` and `deepseek-v4.1-flash` also
remain available. DeepSeek models on Go may additionally require the
"Enable models hosted in China" opt-in on the workspace page.
