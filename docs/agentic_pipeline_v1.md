# Agentic pipeline v1.6 — free-form text in, one LLM calls (a), (b), (c)

A user writes an ordinary sentence. An LLM controller calls `generate_motion`
(a), `describe_motion` (c) and `edit_motion` (b) one at a time, reads what each
one measured, writes the next edit instruction itself, and stops when the
measurement agrees with the request. The old `SCORE` (100 whenever validity
passed) is replaced by per-module measurements plus loop metrics.

## 0. Requirement map

| Review line | Implementation | Check | Result in this run |
|---|---|---|---|
| **i)** metrics per module, no saturating `SCORE` | `scripts/agent/metrics.py`; `report.json` has `module_a_generation` / `module_b_editing` / `module_c_caption` / `loop` and no `score` field | `no_saturating_score_field`, `metrics_are_kinematic`, `gen_metric_*`, `edit_metric_*`, `caption_metric_glove`, `loop_metric_*` | 19 joint measurements per clip; R@1 0.35; instruction following 1.00; ROUGE-L 0.350; agreement 1.00 (§2, §5) |
| **ii)** free-form text input | UI → `POST /api/agent` → `agent.py --goal`, no template | `free_text_goal` | replay goal is the verbatim sentence "a person sits down and raises the right arm" (§6) |
| **1)** reason first, then (a) | `SYSTEM` prompt (one `thought` + one JSON action); rail "no clip → generate"; rejection sampling inside (a) | `mandatory_order_in_prompt`, `episode_follows_the_mandatory_order`, `every_step_gives_a_reason` | step 1: "No clip yet; generate the requested motion." 3 drafts drawn, 3 rejected, closest kept |
| **2)** check with (c), compare caption vs. input, decide if a fix is needed | `describe_motion` returns the learned caption **and** a measurement-grounded caption, their GloVe / content-recall / action-family scores, and `caption_verdict{level, needs_fix}` | `caption_compared_with_goal`, `two_captions_and_a_verdict`, `verdict_is_a_decision`, `content_recall_beats_bag_of_words` | learned caption "stand with their arm out" vs. seated goal: GloVe 0.898 but content recall 0.40, action overlap 0.0 → `mismatch`, `needs_fix=true` |
| **3)** write the fix in text, apply (b) | controller's `assessment.next_edit` → `edit_motion(instruction)`; impossible wording refused by `edits.py`; an edit that does not move the named attribute is rolled back | `edit_text_written_by_controller`, `edit_followed_the_instruction`, `preconditions_are_named`, `no_effect_is_reported` | "raise the right arm higher" → `arm_raise_right` +2.044, gap 0.848 → 0.0, no new hard failure |
| **\*)** loop 2–3 until the agent judges it fits | only ordering rails (describe after an edit, check before finish, a worse clip cannot be final, budgets); `restore_motion` + clip archive | `loop_capped`, `check_before_finish_enforced`, `worse_clip_cannot_be_final`, `loop_stopped_when_measured`, `loop_repeats_check_and_edit`, `final_clip_is_never_worse_than_best_seen` | 20/20 finished after `check_requirement`; ep16 looped describe→edit→describe→check→finish; 0 episodes ended worse than their best clip |
| shorter, honest report | §3 process, §7 problems, §8 next steps | `report_has_a_process_section`, `report_states_open_problems`, `report_has_next_steps_with_dod` | defect → fix → proof table; P0/P1/P2; 7 goals with a DoD each |

## 1. Tool contract

| Tool | Backend | What the controller sees |
|---|---|---|
| `generate_motion` | MLD (or dataset retrieval) | frames, generator used, kinematic summary, `requirement` gap, `draft_rejection` (drawn / rejected / chosen) |
| `resample_motion` | MLD `--replication n` (one model load) | each draft's measured attributes and gap, which was kept, `kept_current` |
| `describe_motion` | TM2T + measurement-grounded caption | both captions, similarity scores, `caption_verdict`, kinematic summary |
| `check_requirement` | `verify.py` joint measurements | pass/fail per requirement, measured value, `requirement_gap`, validity |
| `restore_motion` | episode archive | restored run dir, gap before/after, remaining budget |
| `edit_motion` | session revise (intent → geometric edit / MotionFix) | before/after metrics, target change, non-target MPJPE, `no_change`, `worse_than_before` |
| `finish` | — | claim vs. checker verdict, `agreement`; refused while a closer clip exists |

Four rails, the only fixed rules: no clip → generate; an undescribed clip →
describe; `finish` before `check_requirement` → check; budgets are never
exceeded. Everything else (when to stop, what is missing, which edit) is the
LLM's decision, recorded with its reason.

```
free text → ① generate_motion (a) → ② describe_motion (c) → ③ check_requirement
          → ④ judge caption+metrics → ⑤ edit_motion (b) → ↺ ②–⑤ → finish
```

## 2. Metrics

| Module | Metric | Definition |
|---|---|---|
| (a) | R@1 / R@3 | one clip per goal, every goal ranked against every M2T caption by GloVe cosine |
| (a) | `align_cos` | prompt↔caption GloVe cosine |
| (a) | `foot_slide`, `jerk`, `ground_penetration` | skating, jitter, deepest joint below the floor (÷ body height) |
| (a) | `bone_cv` | bone-length variation; ≈0 for any valid clip, so a validity guard, not a quality ranking |
| (a) | `ankle_lift_*`, `jump_rise`, `jump_clearance` | does a kick lift the foot, does a jump leave the ground |
| (a) | `diversity` | mean pairwise distance of pose features (mode collapse) |
| (b) | instruction following | did the named attribute move in the named direction |
| (b) | `target_change` / non-target MPJPE | how much the intended joints moved / collateral movement (DTW-aligned, ÷ height) |
| (b) | `target_vs_rest` | localisation of the edit |
| (b) | `no_change`, `no_effect`, `refused` | three distinct outcomes; the last two restore the previous clip |
| (b) | `instruction_coverage`, `refusal_honesty` | applied ÷ applicable; named reasons ÷ cases that should be refused |
| (c) | BLEU-1…4, ROUGE-L, CIDEr-D, `glove_cos` | 20 held-out clips with 3 references each |
| loop | success, edits/step, gain, tokens, seconds | episode cost and convergence |
| loop | `agent_claim_satisfied` vs `checker_satisfied` | `agreement`: controller calibration, not ground truth |
| loop | `n_worse_than_best` | episodes that ended on a clip worse than one they had measured (must be 0) |

Goal checks invert with the wording: "walks in place" is a **max** on
`root_travel` (≤ 0.25 m), so a forward walk cannot pass.

## 3. How the work was completed

| Ver | Defect observed | Change | Proof it is fixed |
|---|---|---|---|
| v1.1 | backends measured separately; `finish` silently overruled | backends became tools; checker became a tool; claim and verdict recorded separately | §6 trace, `agreement` |
| v1.2 | `sit lower` put the body 2.47 heights underground while the gate said 100 | `sit.deepen` as 2-bone leg IK; gate shares `ground_clearance`; executable capability registry | `pelvis_drop 0.367→0.467`, penetration 0.000 |
| v1.3 | a *correct* arm edit rolled back twice: source sat 0.8 m below the floor | reground the source once; blame only `new_hard_failures` | same edit: `pelvis_drop 0.0055→0.1055` |
| v1.3 | sit-hold moved the pelvis joint alone and blended arms linearly (+30 % bone length) | per-frame drop through the same IK; arm chains re-lengthed | `bone_cv 0.0197`, `arm_raise 1.969` |
| v1.3 | kicks and jumps passed because nothing measured them | `ankle_lift`, `jump_rise`, `jump_clearance` | per-goal column, §5.1 |
| v1.4 | a relative operator cannot add travel/turn the draft lacks | `resample_motion`: n drafts in one load, keep the closest | goals 12–13 |
| v1.5 | MLD could not run (Windows blocked 3 compiled extensions) | lazy `spacy` import; accepted `thinc`/`blis` pair | the numbers are real MLD, not fallback |
| v1.5 | one draw cannot separate "cannot" from "this draw missed" (1.35 / 0.12 / 0.02 m) | rejection sampling inside (a) | 17 drafts rejected |
| v1.6 | editing claim rested on 2 in-loop edits; old battery edited already-edited clips and scored refusals as success | 21 cases / 8 families / one source copy per case / declared apply-or-refuse | §5.2 table |
| v1.6 | learned caption said "wave" for a wrist 0.216 m above the head, and a bag-of-words score read 0.90 as a match | dual caption + tri-state verdict + content recall | content recall 0.25, action overlap 0 |
| v1.6 | `resample_motion` capped adoptions, not attempts, so the budget never ran out | budget counts attempts | `test_resample` 15/16 → 16/16 |

Claims taken back: (1) "the pipeline is agentic" was half true in v1.1, since a
hidden `verify_goal` could overrule `finish`; (2) 0.90 was one draw, not a
rate; (3) the old battery's "100 %" measured the wrong thing; (4) a metric can
be numerically fine and still be the wrong metric (`bone_cv`, `glove_cos` alone).

## 4. Experimental setup

- **Battery A**: 20 free-form goals (`test-samples/agent/goals_20.json`), MLD
  HumanML3D checkpoint, 60 frames, ≤ 2 edits, per-episode process + watchdog.
- **Battery B**: a fixed dataset draft, so the controller is the only variable.
- **Module (b) alone**: 21 declared cases, expectations `apply` or `refuse`.
- **Module (c) alone**: 20 held-out HumanML3D clips with references.
- One controller, one prompt, one checker for all of them.

## 5. Results

### 5.1 Battery A (v1.6, `_v3_final10`)

| Quantity | v1.6 | v1.5 |
|---|---|---|
| Goal satisfaction | **20/20 = 1.00** | 18/20 = 0.90 |
| R@1 / R@3 (caption-mediated) | 0.35 / 0.65 | 0.25 / 0.65 |
| prompt↔caption GloVe | 0.887 | 0.881 |
| diversity | 0.129 | 0.143 |
| foot slide / jerk / penetration | 0.0058 / 0.0044 / 0.0046 | 0.0093 / 0.0053 / 0.0033 |
| episodes drawing a batch / drafts rejected | 19 / 17 | 19 / 16 |
| edits per goal · episodes that resampled | 0.05 · 2 | 0.10 · 2 |
| steps per goal | 4.25 | 4.40 |
| checker consulted / agreement | 1.00 / 1.00 | 1.00 / 1.00 |
| ended worse than best measured | **0** | 0 |
| seconds / tokens per goal | 33.7 s / 21 888 | 31.8 s / 19 236 |

**1.00 is one draw of n = 20, not saturation.** The same suite reads 0.90
(`_v3_final5`), 0.90 (`_v3_final7`), 1.00 (`_v3_final10`); the gain came from
rejection sampling at the source plus the no-worse-than-best archive, not from a
more lenient judge. Only goal 16 still needed the editor: 3 drafts rejected, one
edit closed the gap. Read the per-goal column in `report.md`, not the mean.

### 5.2 Module (b) alone — 21 declared cases (`outputs/edit_bench3`)

| Quantity | Value |
|---|---|
| Cases / expected refusals | 21 / 8 (8 families) |
| Instruction coverage | 0.615 (8/13) |
| Refusal honesty | **1.00 (8/8)** |
| Instruction following (applied) | **1.00** |
| Silent no-ops | **0** |
| No-effect edits | 5 (ran, named attribute did not move → rolled back) |
| Rollback rate (new damage) | 0.00 |
| Target/rest ratio · non-target MPJPE | 79.2 · 0.0077 |
| Frame retention | 1.021 |

The 5 `no_effect` rows are real capability edges: stand up on a standing clip,
turn on a seated clip, walk faster on a still clip, trim a clip with no tail, a
sit depth the leg reach clamps to zero. Reported, not scored as success.

### 5.3 Module (c) alone — 20 held-out clips

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr-D | GloVe |
|---|---|---|---|---|---|---|
| 0.480 | 0.281 | 0.112 | 0.000 | 0.350 | 0.207 | 0.824 |

BLEU-3/4 are near 0 at 3 short references; quote ROUGE-L, CIDEr-D and GloVe for
this benchmark size.

### 5.4 Contract

`python scripts/tests/test_agentic_contract.py` → **66/66** against the code,
`report.json`, one finished UI episode and this document.

### 5.5 Visual examples

Generated by `python scripts/agent/make_figures.py` from the same artifacts the
tables above are computed from, so a picture cannot disagree with its number.

**1. One episode, sentence to motion.** Top strip is the MLD draft, bottom strip
is the same clip after the one edit the agent wrote. The arm goes from 0.21 to
2.12 (gate 1.8); sit depth, travel and the non-target joints do not move.

![fig1](docs/figures/fig01_text_to_motion.png)

**2. Why the caption cannot be the judge.** Three descriptions of one clip. The
learned caption is fluent, scores 0.89 GloVe, and is wrong about the pose;
content recall 0.25 and action overlap 0 expose it.

![fig2](docs/figures/fig02_caption_vs_measurement.png)

**3. An edit that ran and changed nothing.** "walk forward further" returned a
byte-identical clip (max joint difference 0.00000 m), so the attribute could not
move; it is reported `no_effect` and the previous clip is restored.

![fig3](docs/figures/fig03_edit_no_effect.png)

**4. A draft with no motion to edit.** 0.015 body heights of joint travel over
60 frames against 0.17 for a walk. No relative operator can fix this; another
draw is the only honest remedy.

![fig4](docs/figures/fig04_static_draft.png)

**5. Six families, and the number that decides each.** The caption is never the
judge — the measurement under each strip is what `check_requirement` read.

![fig5](docs/figures/fig05_six_families.png)

**6. Animated comparison.** `docs/figures/fig06_ep16_before_after.gif` is the
draft and the edited clip playing side by side, same camera, 18 fps.

## 6. One loop, verbatim

`outputs/session/ui_runs/0e68ce14_UI_AGENT_20261115-224751`, goal
"a person sits down and raises the right arm" typed into the UI:

```
1 generate_motion      thought: No clip yet; generate the requested motion.
                       3 drafts drawn, 3 rejected (gap 4.90 / 1.45 / 0.848), kept 0.848
2 describe_motion      learned: stand with their arm out…   rule: sitting down
                       GloVe 0.898 · content recall 0.40 · action overlap 0.0
                       verdict mismatch, needs_fix = true
3 edit_motion          "raise the right arm higher"   (written by the controller)
                       arm_raise_right +2.044, gap 0.848 → 0.0, no new hard failure
4 describe_motion      both captions agree with the goal, gap 0.0
5 check_requirement    satisfied = true
6 finish               claim = verdict → agreement = true
```

## 7. What is still wrong

Severity = how much the defect can mislead a reader.

### P0 (can change the conclusion)

| Defect | Evidence | Effect |
|---|---|---|
| Success rate is one draw of n = 20 | 0.90 / 0.90 / 1.00 across runs | no interval, no pass@k; a 2-goal swing is indistinguishable from noise |
| 0–100 alignment is an LLM judgement | `agreement = 1.00` vs the checker only | never validated against humans |
| Caption metrics are dataset-level | BLEU/ROUGE/CIDEr on 20 clips with references | nothing measures the caption of a *generated* clip |
| Open-vocabulary editor is routing only | `test_open_vocab.py` 10/10 with a fake backend | no joints→SMPL fit and no checkpoint, so no real SimMotionEdit edit has run |

### P1 (capability edges the tables show)

| Defect | Evidence | Effect |
|---|---|---|
| MLD sometimes returns a near-static draft | 60 frames, largest joint path 0.21 m, `body_moving = 0` | a relative operator cannot add motion; only another draw helps |
| `sit lower` is a constant drop | 0.15 m, clamped by leg reach | "a little" / "a lot" are not parsed |
| 5 of 21 battery cases are `no_effect` | §5.2 | coverage 0.615 is the honest number |
| Speed/duration edits change frame count | frame retention 1.021 | temporal cases are judged by the following check, not pose error |

### P2 (engineering debt)

| Defect | Evidence | Effect |
|---|---|---|
| `third_party/TM2T` is a local checkout | git-ignored | a fresh clone silently falls back to the template caption |
| No CI | contract test run by hand | drift is caught only when someone remembers |
| UI replay is manual | logs under git-ignored `outputs/` | §6 cannot be regenerated from a clone |
| Episodes run serially | 689 s for 20 goals | a k = 5 battery is ~1 h |

## 8. Next steps

| Goal | DoD |
|---|---|
| k = 5 draws per goal, pass@k + 95 % Wilson interval | DoD: `report.json` has `k`, `pass_at_1`, `pass_at_k` and the interval; the contract test refuses to print a headline number when k < 5 |
| Human rating of 40 final clips, blind, 2 raters | DoD: Spearman ρ(judge, human) reported, 5 largest disagreements listed; ρ ≥ 0.7 before the score is quoted |
| TMR R-Precision / FID via joints→SMPL fit | DoD: FID and R@1/2/3 on the full test split, plus the fit residual (MPJPE < 0.05 heights) |
| Real SimMotionEdit edit outside the registry | DoD: one `expect=apply` battery case with `backend = simmotion`, same rollback rule |
| Caption verdict scored as precision/recall on 60 labelled pairs | DoD: precision, recall, confusion matrix; `needs_fix` recall ≥ 0.9 |
| Amount-aware depth (`a little` / `a lot`) | DoD: two cases whose `pelvis_drop` differs by ≥ 2×, clamp asserted on a short-legged source |
| CI for the four no-GPU suites, nightly GPU suites | DoD: a deliberately broken rail turns the check red before the badge is trusted |

## 9. Reproduction

```powershell
conda run -n mld python scripts/agent/benchmark.py --battery a --goals 20 --max-edits 2 --goal-timeout 600
python scripts/agent/edit_benchmark.py --out outputs/edit_bench3
python scripts/agent/build_acceptance.py --report outputs/agent_bench/_v3_final10/report.json ^
    --episode outputs/session/ui_runs/0e68ce14_UI_AGENT_20261115-224751 --out outputs/agent_bench/_v3_final10/acceptance.html
python scripts/tests/test_agentic_contract.py
```

Artifacts: `<out>/report.md` / `report.json` (the tables above),
`epNN_*/agent_run.json` + `.md` (every thought, call, observation),
`epNN_*/stepNN_*/03_edited_preview.gif` (clips), `contract_checks.json`
(66 checks), and `outputs/session/prompt_logs/<date>-<topic>/` (UI replay with
screenshots).
