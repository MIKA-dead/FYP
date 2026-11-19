# Agentic pipeline v1.6 — free-form text in, one LLM calls (a), (b), (c)

A user writes an ordinary sentence. An LLM controller decides which backend to
call next, reads what that backend measured, writes the next edit instruction
itself, and stops only when the measurement it was shown agrees with the
request. The three modules are tools, not stages: `generate_motion` (a),
`describe_motion` (c) and `edit_motion` (b) are chosen one call at a time, and
each choice and its reason are in `agent_run.json`.

This version also replaces the old `SCORE` — 100 whenever the hard validity
checks passed — with measurements per module and for the loop.

## 0. Requirement map: every line of the review → implementation → evidence

| Review line | Where it lives | Re-runnable check | Evidence in this report |
|---|---|---|---|
| **i)** quantitative metrics per module, instead of a `SCORE` that is 100 most of the time | `scripts/agent/metrics.py`; `report.json` has four blocks (`module_a_generation`, `module_b_editing`, `module_c_caption`, `loop`) and no field named `score` | `no_saturating_score_field`, `metrics_are_kinematic`, `gen_metric_*`, `edit_metric_*`, `caption_metric_glove`, `loop_metric_*` | Definitions §2, values §5.1–§5.3 and §5.9. 19 measured attributes per clip (`root_travel`, `pelvis_drop`, `yaw_net_deg`, `arm_raise_right`, …), plus R@1/R@3, BLEU/ROUGE/CIDEr/GloVe, instruction following, non-target MPJPE, success and agreement |
| **ii)** the user input is free-form text | UI input → `POST /api/agent` (`prompt`) → `agent.py --goal`; no template, no slot filling | `free_text_goal` | The replay's goal is verbatim "a person sits down and raises the right arm", and the controller chose all six tool calls (§6) |
| **1)** the agent first reasons about the requirement, then uses (a) to generate | `SYSTEM` in `agent.py` (a `thought` plus one JSON action) and the rail "with no clip: `generate_motion`"; generation-side rejection sampling runs inside (a) | `mandatory_order_in_prompt`, `episode_follows_the_mandatory_order`, `every_step_gives_a_reason` | Step 1 thought: "No clip yet; generate the requested motion." MLD drew 3 drafts, rejected 3, kept the closest (§5.9, §6) |
| **2)** it checks the result with (c), compares the caption with the user input, and decides whether a fix is needed | `describe_motion` returns the learned M2T caption **and** a caption written from the same joints the checker reads, their GloVe / content-recall / action-family similarity to the goal, and `caption_verdict {level, needs_fix}`; `check_requirement` adds the measured verdict | `caption_compared_with_goal`, `two_captions_and_a_verdict`, `content_recall_beats_bag_of_words`, `verdict_is_a_decision` | Step 2: learned caption "a person stand with their arm out…" for a seated clip with a low arm → GloVe 0.898 but content recall 0.40 and action overlap 0.0 → verdict `mismatch`, `needs_fix = true` (§6) |
| **3)** it writes in text what must be fixed, then applies (b) | the controller's `assessment.next_edit` and the `instruction` argument of `edit_motion`; the registry refuses impossible wording, and `_assess_after` rolls back an edit that did not move the named attribute | `edit_text_written_by_controller`, `edit_followed_the_instruction`, `preconditions_are_named`, `no_effect_is_reported` | Step 3: instruction "raise the right arm higher", `arm_raise_right` Δ 2.044, gap 0.848 → 0.0, no new hard failure (§6) |
| **\*)** loop steps 2–3 until the agent judges that the clip fits | rails only for ordering (describe after an edit, check before finish, a worse clip cannot be final, budgets); everything else is the controller's call; `restore_motion` plus the clip archive mean an episode cannot end on a clip worse than the best it measured | `loop_capped`, `check_before_finish_enforced`, `worse_clip_cannot_be_final`, `loop_stopped_when_measured`, `loop_repeats_check_and_edit`, `final_clip_is_never_worse_than_best_seen` | 20/20 episodes ended on `finish` after `check_requirement`; ep16 looped describe → edit → describe → check → finish; 0 episodes ended worse than their best clip (§5.1, §5.9) |
| restructure the report: how the work was done, then current problems and next goals | §3, §7 and §8 of this file | `report_has_a_process_section`, `report_states_open_problems`, `report_has_next_steps_with_dod` | §3 defect → change → proof table; §7 severity-ranked defects with evidence; §8 ordered goals, each with a definition of done |

How the rest of this note is organised:

- **§1** the tool contract — what the controller can call, and the four rails
  that are the only fixed rules;
- **§2** the metrics — why the old score could not compare methods, and the
  definition of every number reported later;
- **§3** how the work was completed — each observed defect, what was changed,
  and the check that proves the change, including judgement calls that turned
  out wrong;
- **§4–§6** setup, results, and one verbatim loop trace from the UI;
- **§7** what is still wrong, by severity; **§8** what comes next, each item
  with a definition of done; **§9** reproduction commands and artifacts.

```
free-form goal
     │
     ▼
┌───────────────────────────── LLM controller ─────────────────────────────┐
│  thought → tool choice → arguments          (JSON action, one per turn)  │
│                                                                          │
│  ① generate_motion(prompt)    ── MLD text→motion ──► 03_edited.npy       │
│  ①' resample_motion(prompt,n) ── n drafts, keep closest ──► draft         │
│  ② describe_motion()          ── TM2T M2T ────────► caption + metrics    │
│  ③ check_requirement()        ── measure the request against the joints  │
│  ④ judge: caption + metrics vs. goal ────────────► alignment, gap        │
│  ⑤ edit_motion(instruction)   ── session revise ──► new 03_edited.npy    │
│  ⑥ restore_motion()           ── archive ─────────► closest clip so far  │
│  ↺ ②–⑥ until finish or budget                                            │
└──────────────────────────────────────────────────────────────────────────┘
     │
     ▼
agent_run.json / agent_run.md  (every thought, call, observation, assessment)
```

## 1. Tool contract

`scripts/agent/tools.py` exposes the three backends as stateful tools. The
controller never handles a path; it only writes the text the tool consumes.

| Tool | Backend | Arguments | Observation given to the LLM |
|---|---|---|---|
| `generate_motion` | MLD (`mld_generate.py`, conda env `mld`); `generator="dataset"` retrieves a HumanML3D clip by caption match | `prompt`, `generator` (optional) | frames, generator used, kinematic summary, `requirement` (satisfied / gap / failed checks), `draft_rejection` (drawn / accepted / rejected / chosen) when a batch was drawn, preview path |
| `resample_motion` | MLD `--replication n --allinone` (one model load); the current clip competes as candidate 0 | `prompt`, `n` (2–4) | every draft's measured attributes, requirement gap, which draft was kept, `kept_current` when every new draft was worse |
| `describe_motion` | TM2T `M2T_EL4_DL4_NH8_PS` + VQ tokenizer, plus a measurement-grounded rule caption | — | both captions, GloVe / content-recall / action-family similarity to the goal, `caption_verdict` (match / uncertain / mismatch), kinematic summary |
| `check_requirement` | `scripts/agent/verify.py` (joint measurements) | — | pass/fail per requirement, the measured value, the relative `requirement_gap`, the physical validity report |
| `restore_motion` | the episode's own archive of measured clips | `reason` | restored clip's run dir, gap before/after, remaining restore budget |
| `edit_motion` | session revise (intent compiler → geometric edit / MotionFix) | `instruction` | before/after metrics, target change, non-target MPJPE, instruction-following checks, `no_change`, `gap_before`/`gap_after`, `worse_than_before`, instruction family, grounding offset |
| `finish` | — | `reason`, `limitation` | agent claim vs. checker verdict, agreement; refused while an earlier clip is closer |

If MLD fails, `generate_motion` retries with the in-repo dataset draft and
records `generator_used = dataset_fallback`; the trace never hides which backend
produced the clip.

**Rejection sampling runs inside the generator, not after it.** When the prompt
names a measurable attribute, `stage_generate._mld_draft` asks MLD for a batch
of 3 drafts in one process, maps each into the preview skeleton, grounds it, and
keeps the one `sampling.select` ranks first (physical validity, then relative
`requirement_gap`). The same prompt produced 1.35 m, 0.12 m and 0.02 m of travel
on three consecutive draws, so a single draw cannot separate "the model cannot"
from "this draw missed". The observation carries `draft_rejection` with the
number drawn, rejected and chosen; `AGENT_REJECTION=off` restores one draw and
`AGENT_REJECTION_BATCH=N` changes the batch size, which is the ablation switch
for §5.1.

`resample_motion` exists because the editor is *relative*: it can scale or
reshape motion that is already present, but it cannot add travel, add a turn, or
remove locomotion. When the measurement says an attribute is missing outright
(`root_travel 0.0`, `yaw_span 10°`), the only honest remedy is another draw from
the generator. The tool asks MLD for `n` samples in **one** process
(`--replication n --allinone`, one ~30 s model load), runs each through the
pipeline, measures each with `verify.verify_goal` plus a relative
`requirement_gap`, and adopts the closest valid draft while reporting every
candidate. If no candidate came from a fresh draw — the generator is down and
everything fell back to the retrieved clip — it returns
`resample_unavailable` and does **not** charge the budget, so the controller
cannot mistake a deterministic repeat for a successful resample. The budget is
two resamples per episode.

`edit_motion` is gated by `scripts/agent/edits.py`, a registry that maps wording
to a supported operator family (`arm_raise`, `wave`, `walk`, `speed`, `turn`,
`sit_deeper`, `stand_up`, `trim_trailing`, …) and to an explicit out-of-scope
list (objects, camera, appearance, body shape, jump, kick, dance). An
instruction outside it returns `unsupported_instruction` with a reason and a
suggestion *before* any backend work, so the controller cannot burn a step on
an impossible edit and the benchmark can report coverage. The same registry
renders the capability text in the prompt, so the prose the model reads and the
gate that rejects it can never disagree.

The controller's JSON contract is

```json
{"thought": "...",
 "assessment": {"alignment": 0, "satisfied": false, "matches": [],
                "missing": [], "next_edit": ""},
 "action": "generate_motion | resample_motion | describe_motion | check_requirement | edit_motion | finish",
 "args": {}}
```

`scripts/agent/agent.py` enforces four rails, which are the *only* fixed rules
in the loop:

- no clip → `generate_motion`;
- a clip that has not been described since the last edit → `describe_motion`;
- `finish` before `check_requirement` → `check_requirement`;
- the edit budget is never exceeded.

Everything else — when to stop, what is still missing, which single edit to
try next — is the LLM's decision, recorded with its justification. The checker
does not overrule the controller. If the controller claims satisfaction and the
measurement disagrees, it is told once what failed and may either fix it or stop
with `{"limitation": true}`; the episode records `agent_claim_satisfied`,
`checker_satisfied` and `agreement` so the report can show how often the
controller was right on its own.

## 2. Why the old SCORE was not a metric

`quality_gate` answers "is this a valid clip?" (finite joints, minimum length,
foot sliding, sit depth, turn anti-slide). It returns 100 as soon as no rule
fires. Two measurements from the runs in §5 show why that number cannot be the
experiment:

- Every generated clip scored **100** while the semantic alignment the
  controller assigned ranged from **70 to 95**, and one of the five clips did
  not satisfy its goal at all.
- The `sit lower` edit that the controller requested produced a clip whose
  lowest joint was **2.47 body-heights below the floor** — and `quality_gate`
  still returned `passed = true, score = 100`, because it never checks ground
  penetration. The agent's own validity check caught it and rolled the clip
  back.

A number that is 100 for a correct clip and for a clip with its feet 1.5 m
underground cannot compare methods.

The replacement metrics are defined next; the implementation is
`scripts/agent/metrics.py`.

### (a) Generation

| Metric | Definition | What it catches |
|---|---|---|
| `R@1`, `R@3` (caption-mediated) | Generate one clip per goal; describe each with M2T; rank every goal against every caption by GloVe cosine; report top-1 / top-3 retrieval accuracy | Whether a clip is *specifically* the requested motion, not just plausible. This is the R-Precision idea with the M2T caption as the text–motion bridge |
| `align_cos` | GloVe cosine between the user prompt and the M2T caption | Text–motion semantic alignment |
| `bone_cv` | Mean coefficient of variation of the 21 bone lengths over time | Numeric corruption / non-rigid retargeting. Any valid skeletal clip gives ≈0, so this is a *validity* guard, not a quality signal — reported for completeness, not used to rank methods |
| `foot_slide` | Mean horizontal foot displacement while the foot is in contact | Skating |
| `jerk` | Mean ‖third difference of joint positions‖ ÷ body height | Jitter / implausible acceleration |
| `ground_penetration` | Deepest joint below the floor ÷ body height | Feet through the floor. Checked by `quality_gate` (hard, ≤ 0.10 heights) and by the agent's edit postcondition since v1.2 |
| `ankle_lift_left/right` | Rise of that ankle above its own lowest point ÷ body height | Whether a kick actually lifts the striking foot. A kick gives ≥ 0.3 on one side and a large left/right asymmetry; walking alternates both feet |
| `jump_rise`, `jump_clearance` | Pelvis rise above frame 0, and floor clearance at the pelvis apex, both ÷ body height | Whether a jump leaves the ground. Standing up also raises the pelvis, but with the feet planted (`jump_clearance ≈ 0`), so the pair separates a jump from a stand |
| `diversity` | Mean pairwise distance of root-relative, height-normalised pose features | Mode collapse across prompts |

Goal checks invert where the wording inverts them: "walks in place" is a *max*
on `root_travel` (≤ 0.25 m) rather than a min, because the request is the
opposite of locomotion. Before this, a forward walk scored a success on a
"walks in place" goal unless the checker happened to phrase it as a walk.

The retrieval battery is 20 free-form goals (`test-samples/agent/goals_20.json`),
all in-distribution for HumanML3D so the score measures caption–goal alignment
rather than generator coverage. At n = 20 the R@1 resolution is 0.05, which is
still coarse; HumanML3D's own protocol uses the full test split, so treat these
numbers as a relative signal between batteries, not as a leaderboard entry. A
TMR-based R-Precision / FID is possible with the checkpoints under
`third_party/motionfix/eval-deps`, but they consume SMPL 6-D poses while the
pipeline outputs joints; we did not add a joints→SMPL fit just for the report.

### (b) Editing

| Metric | Definition | What it catches |
|---|---|---|
| Instruction-following rate | Does the requested attribute move in the requested direction? arm_raise / travel / yaw / sit depth / duration, each with a threshold | Did the editor do what the agent asked? |
| `target_change` | Mean displacement of the edited joints, root-relative, DTW-aligned | How much the edit moved the intended body part |
| Non-target MPJPE | Same measure restricted to joints the instruction should not touch, normalised by body height | Collateral damage / identity preservation |
| `target_vs_rest` | target change ÷ non-target change | Is the edit localised? |
| Bone-length drift | `bone_cv` after − before | Anatomical damage |
| Smoothness Δ | `jerk` after − before | New jitter introduced by the edit |
| `no_change` rate | Fraction of edits whose output is byte-identical to the input | Silent editor failures, which the agent must see |
| `instruction_coverage` | Applied ÷ cases that should be applicable | Whether the editor can do what was asked, measured on a declared battery |
| `refusal_honesty` | Named refusals ÷ cases that should be refused | Does the editor say *why* instead of no-opping? |
| No-effect count | Edits that returned a different clip without moving the named attribute | Silent "looks edited" results; they are rolled back |
| Unsupported count | Instructions rejected by the registry before the backend ran | Wasted steps avoided, and a map of what users ask for but the editor cannot do |
| `n_skipped_no_source` | Cases whose source clip lacked the precondition (no travel / turn / sit) | Distinguishes "operator failed" from "source could not support it" |

The DTW alignment makes the preservation metric fair when the edit changes
speed: frames are matched by pose, not by index.

Instruction matching is on **whole words**, because substring matching produced
two wrong verdicts on real instructions: "walk **slower**" contains "lower" and
was scored as a request to sit deeper, and "remove the extra motion after
**sitting**" contains "sit" and was scored as a request to deepen a sit. Speed
words ("slower", "faster") are judged on *speed* (travel ÷ frames), not total
travel: a correct slow-down lengthens the clip and can leave total travel
unchanged, so a distance test marks the right edit as a failure. An edit with no
applicable attribute and no localisation ratio is reported as **unverifiable**
and left out of the following rate instead of being counted as a miss.

v1.6 adds the part the metric set was missing: an edit is only counted as
applied if it **moved the attribute the instruction names**. Three outcomes are
now distinct and all are visible to the controller — `refused` (the operator
cannot do it on this clip: no travel to extend, no turn to stop, no sit to
deepen, or outside the registry), `no_effect` (the backend ran but the named
attribute did not move, or the output was byte-identical) and `applied`. The
first two restore the previous clip, so a failed edit cannot quietly replace
correct motion; `scripts/agent/edit_benchmark.py` reports them separately
instead of folding them into a coverage number.

The `sit lower` case is why this layer exists. The wording used to be mapped to
the recipe `sit.slow_descend` (which listed `"sit lower"` among its hints), so
the edit slowed the descent and re-grounded the feet, leaving the pelvis at the
same height — and when it did run the geometric sit overlay, the feet went
underground. v1.2 removes that hint, adds `sit.deepen`, and implements the depth
change as a joint-space 2-bone leg IK (`sit_deepen_postprocess.py`): the pelvis
lowers by δ, the ankles and feet are restored to their original position, and
each knee is placed at the intersection of the thigh and shin spheres, so bone
lengths are preserved exactly and no foot can go through the floor. Measured on
the Battery-A sit clip: `pelvis_drop` 0.367 → 0.467 m, `ground_penetration`
0.000, `bone_cv` 0.000, feet displaced 0.0 m, and the frame count unchanged
(the earlier mis-parse also stretched 54 → 81 frames).

### (c) Captioning

| Metric | Definition |
|---|---|
| BLEU-1…4 | Corpus-level clipped n-gram precision with brevity penalty |
| ROUGE-L | Longest-common-subsequence F1 (best of 3 references) |
| CIDEr-D | TF-IDF n-gram cosine with length penalty, corpus document frequencies |
| `glove_cos` | Mean GloVe cosine to the 3 references |

Measured on 20 held-out HumanML3D test clips with their ground-truth captions.
This is the honest way to evaluate an M2T model: the pipeline clips have no
reference sentence, so caption quality must be measured on the dataset the model
was trained for.

### Agent loop

| Metric | Definition |
|---|---|
| Success rate | Episodes where the controller finishes with `satisfied = true` and the last measured clip is the one it judged |
| Edits to success | Number of `edit_motion` calls before `finish` |
| Alignment trajectory | The controller's 0–100 judgment after each `describe_motion` |
| Alignment gain | Last − first judgment in an episode |
| Tool calls / steps / wall time / tokens | Cost of the episode |
| `agent_claim_satisfied` | What the controller said about the clip in its final assessment |
| `checker_satisfied` | What `check_requirement` measured |
| `agreement` | Claim == verdict. A calibration number: how often the controller's own judgment matched the measurement |
| `check_coverage` | Episodes where the controller consulted the checker before finishing (should be 1.0 — the rail enforces it) |
| Failure taxonomy | `judged_unsatisfied`, `no_change`, `unsupported`, `budget`, `generate_failed` |

The alignment score is an LLM judgment, not a ground-truth label; that is why
the report also lists the measured quantities it cites (arm ratio, travel, yaw,
caption). The judge is constrained by a rubric and told that measurements win
over the coarse caption.

## 3. How the work was completed

This section is the audit trail: what was observed, what was changed, and the
check that makes the change re-verifiable. None of the numbers in §5 came from a
first attempt; each one followed a defect found by running the previous
version's claim against its own artifacts.

### 3.1 Defect → change → proof

| Version | Defect observed (evidence) | Change | Proof it is fixed |
|---|---|---|---|
| v1.1 | Each backend was measured alone; the loop's "satisfaction" was the controller's own claim, and `finish` was silently overruled by `verify_goal` | The three backends became tools; `check_requirement` became a tool the controller must call; claim and verdict are recorded separately (`agreement`) | §5.5 trace; `check_before_finish_enforced` |
| v1.2 | `sit lower` put the body 2.47 body-heights underground while `quality_gate` still returned 100 | `sit.deepen` implemented as 2-bone leg IK; the gate shares `ground_clearance`; an executable capability registry rejects impossible wording before the backend runs | §5.4; `pelvis_drop 0.367 → 0.467 m` with penetration 0.000 |
| v1.3 | A *correct* arm edit was rolled back twice (`ep16`, `ep17`) because the restored source sat 0.8 m below the floor | Sources are regrounded once (`offset_to_ground`); an edit is blamed only for failures it introduced (`new_hard_failures` vs `pre_existing_hard_failures`) | §5.6; the same edit now gives `pelvis_drop 0.0055 → 0.1055 m` |
| v1.3 | The sit rebound stage moved the pelvis joint alone and blended arms linearly, stretching arm bones by up to 30 % (`bone_cv 0.0851`) | The clamp is a per-frame drop through the same leg IK; arm chains are re-lengthed to the contact frame | §5.7; `bone_cv 0.0197`, the edit applies, `arm_raise_right = 1.969` |
| v1.3 | "kicks" and "jumps" passed because nothing measured them | `ankle_lift_{left,right}`, `jump_rise`, `jump_clearance` added to the checker | §5.1 per-goal column |
| v1.3 | "walk in place" scored a success on a 1.97 m forward walk | Goal checks invert where the wording inverts (`root_travel ≤ 0.25 m`) | §5.1 goal 10 |
| v1.4 | The editor is relative, so it cannot add travel or rotation the draft lacks, and the controller kept trying | `resample_motion` draws n drafts in one model load, measures each, keeps the closest, and reports the rejected ones | §5.1 goals 12–13 |
| v1.5 | MLD could not run at all: Windows Application Control blocked three compiled extensions in the `mld` environment | Lazy `spacy` import inside the vendored MLD, plus the accepted `thinc` / `blis` pair | §5.1 is real MLD output, not `dataset_fallback` |
| v1.5 | One draw cannot separate "the model cannot" from "this draw missed" (1.35 / 0.12 / 0.02 m on the same prompt) | Rejection sampling inside (a) (`sampling.py`), 3 drafts for a measurable prompt | 17 drafts rejected in `_v3_final10` (§5.1) |
| v1.5 | The loop existed only in scripts; the UI demo was a different path | `agent` mode in the UI → `POST /api/agent`, the same tools | §6 replay |
| v1.6 | The editing claim rested on n = 2 in-loop edits; the old battery edited a clip a previous case had already changed, and scored refusals as successes | 21 declared cases over 8 families, one source copy per case, `expect=apply` or `refuse`; three outcomes (`applied`, `no_effect`, `refused`) | §5.9 battery table; `edit_battery_*` checks |
| v1.6 | The learned caption said "wave" for a clip whose wrist was 0.22 m above the head, and a bag-of-words score read that as a match (GloVe 0.87) | Dual caption plus a tri-state `caption_verdict`; content recall and action families alongside GloVe | §6 step 2; `content_recall_beats_bag_of_words` |
| v1.6 | An edit that returned a different clip without moving the named attribute was counted as applied | `edit_did_not_follow_instruction` and `edit_had_no_effect` roll the clip back; the 5 such cases stay visible in the battery | §5.9; `no_effect_is_reported` |
| v1.6 | A hung backend froze the whole 20-goal suite mid-run | Per-episode subprocess with a watchdog (`--goal-timeout`) | `_v3_final10` finished with 0 episode errors |
| v1.6 | `resample_motion` capped *replacements* rather than attempts, so a resample that kept the current clip was free and the budget could never be reached | The budget counts attempts; `resamples` still counts adopted clips | `test_resample.py` 15/16 → 16/16 |

### 3.2 Four diagnoses worth keeping

**The rollback that blamed the wrong layer (v1.3).** Two sit edits were rolled
back as "produced an invalid clip". The instruction was not the cause: the
restored MotionsFix-native source kept its root at the origin, so SMPL FK put
the feet 0.77 m below the floor before any edit ran, and the depth operator then
had a floor 0.8 m too low to plant against. The fix was to ground the source
once, at the door, and to compare the edit's physical report with the source's
instead of with an absolute rule. The general lesson is in the metric design:
`new_hard_failures` is a *delta*, and a delta is the only thing an edit can be
blamed for.

**The stage that changed the skeleton (v1.3).** The arm-raise edit produced the
requested pose and was still rejected for `bone_rigid`. Applied alone to the
same input, it changed `bone_cv` by 0.000; the damage came from the sit-hold
stage that ran after it, which translated the pelvis joint (owner of three
bones) and blended arms linearly instead of rotating them. Both stages now
preserve the skeleton they correct, and the regression suite reports 33/33.
The lesson kept in the code: a post-process that "only fixes jitter" still has
to be measured against bone lengths.

**The caption that was not evidence (v1.6).** On a clip with the wrist 0.22 m
above the head, the learned caption read "a person wave with their right hand",
and the previous bag-of-words comparison scored that 0.87 — a match, so no edit.
Content recall on the same pair is 0.17, and the action-family overlap is 0. The
caption is now only one of two signals: the measurement-grounded caption and the
checker decide, and the three-way verdict says which of the two disagreed. This
is also why `caption_verdict` has a `needs_fix` boolean rather than only a
label: a decision that the controller must act on has to be a field, not prose.

**The budget that never ran out (v1.6, found while re-checking this report).**
`resample_motion` returned `kept_current` when every fresh draft was worse, and
that path did not increment `resamples` — so the controller could draw batches
forever while the counter stayed at 0. `test_resample.py` caught it as the one
failing check of 16. The budget now counts attempts (the work) separately from
adoptions (the effect), which is also the honest split for the report.

### 3.3 Claims this work had to take back

- **"The pipeline is agentic" (v1.1) was only half true.** The controller chose
  the steps, but `finish` was overruled by a hidden `verify_goal` call, so the
  loop could not tell "my judgement is wrong" from "the tool refused". v1.2
  made the checker a tool, and `agreement` exists because the old claim hid the
  disagreement.
- **"0.90 success" was one draw.** The same suite has read 0.90, 0.90 and 1.00
  (`_v3_final5/7/10`). The mechanism that moved it is rejection sampling at the
  source plus the no-worse-than-best archive, not a better judge; §8.1 is how to
  replace the single number with an interval.
- **The old editing battery's "100 %" measured the wrong thing.** Cases edited a
  clip an earlier case had already modified, and refusals counted as successes.
  The current battery reports 0.615 coverage and 8/8 honest refusals instead.
- **A metric can be numerically fine and still be the wrong metric.** `bone_cv`
  is ≈0 for every valid clip, so it belongs in a validity guard, not in a
  ranking; `glove_cos` is high for boilerplate, so content recall had to be
  added next to it. Both are stated in §2 as such.

## 4. Experimental setup

Two batteries are reported. Neither one runs a module in isolation: in both,
the same free-form sentence enters the same controller and every backend is
reached only through a tool call.

**Battery A — MLD is the generator.** Twenty free-form goals from
`test-samples/agent/goals_20.json`, MLD HumanML3D checkpoint, 60 frames. This is
the intended system: the motion the agent edits is generated from the prompt,
not retrieved.

**Battery B — fixed draft, controller is the only variable.** Four goals, the
draft is retrieved once from the in-repo HumanML3D index (`generator=dataset`).
The same draft is returned on every run, so any change between episodes is
attributable to the controller's decisions rather than to generation variance.
This battery is the controlled one; Battery A is the realistic one.

Common settings:

- Captioner: TM2T M2T transformer (`M2T_EL4_DL4_NH8_PS`), beam 2, cached model.
  `third_party/TM2T` is an upstream checkout, not a submodule: clone it there and
  put the released `M2T_EL4_DL4_NH8_PS` checkpoint plus its VQ tokenizer under
  `third_party/TM2T/checkpoints/` (git ignores the directory, the checkpoints and
  the zips; the GloVe table it ships is what §2 (c) scores with). Without the
  checkout `caption_edited` falls back to the quality/timeline template and
  records that in `source`, while the measurement-grounded rule caption and the
  `caption_verdict` are still produced — the loop keeps running and the missing
  model is visible in the artifact rather than silently implied.
- Editor: the session revise path (intent compiler → geometric operator or
  MotionFix). `geometric_only` in a plan now means *exactly* that: the editor
  routes it to the kinematic branch and raises on an unknown mode instead of
  silently falling through to diffusion.
- Generator: MLD `demo.py` in the `mld` conda env. Goals that name a measurable
  attribute (19 of 20 in Battery A) draw a batch of 3 drafts in one process and
  keep the closest one (`AGENT_REJECTION`, batch size 3); goals with no
  measurable attribute draw once. `AGENT_REJECTION=off` reproduces the
  single-draw behaviour of v1.4 for ablation.
- Controller: `glm-5.3-flash` through the OpenCode gateway, temperature 0.2.
- Budget: ≤ 8 controller steps, ≤ 2 edits and ≤ 2 resamples per goal.
- Machine: one Windows workstation, RTX GPU for MLD / TM2T.

## 5. Results

### 5.1 Battery A — MLD generator (20 goals, LLM controller)

Run: `outputs/agent_bench/_v3_final7` (MLD working, generation-side rejection
sampling on: 19 of 20 goals drew 3 drafts, 16 drafts rejected in total).

| Quantity | Value (v1.5 run) | v1.4 (`_v3_final6`) |
|---|---|---|
| Goal satisfaction | **18 / 20 = 0.90** | 17 / 20 = 0.85 |
| Caption-mediated R@1 / R@3 | 0.25 / 0.65 | 0.30 / 0.50 |
| Mean prompt↔caption GloVe cosine | 0.881 | 0.889 |
| Batch diversity (mean pairwise) | 0.143 | 0.134 |
| Foot slide / jerk / floor penetration (mean) | 0.0093 / 0.0053 / 0.0033 body-heights | 0.0056 / 0.0050 / 0.0030 |
| Edit attempts · applied · rolled back · refused up front | 2 · 2 · 0 · 0 | 8 · 7 · 0 · 1 |
| Instruction following (verifiable edits) / frame retention | 1.00 / 1.00 | 0.833 / 0.896 |
| Non-target MPJPE (height-normalised) | 0.0023 | 0.0099 |
| Mean controller steps per goal | 4.40 | 5.35 |
| Checker consulted / agent↔checker agreement | 1.00 / 1.00 | 1.00 / 1.00 |
| Drafts drawn at generation / rejected / episodes that resampled | 57 / 16 / 2 | — |
| Mean seconds and tokens per goal | 31.8 s / 19 236 | 38.8 s / 21 190 |
| Wall time (20 goals, incl. module (c) bench) | 635 s | 776 s |

| # | Goal | edits | checker | align | s |
|---|---|---|---|---|---|
| 1 | raises the right arm high above the head | 0 | ✓ | 90 | 29 |
| 2 | raises the left arm above the head | 0 | ✓ | 95 | 24 |
| 3 | raises both arms above the head | 0 | ✓ | 95 | 25 |
| 4 | waves with the right hand | 0 (1 draft rejected) | ✓ | 95 | 23 |
| 5 | waves with the left hand | 0 (1 draft rejected) | ✓ | 95 | 24 |
| 6 | walks forward and waves with the right hand | 0 (2 drafts rejected) | ✓ | 90 | 23 |
| 7 | walks forward slowly | 0 | ✓ | 92 | 26 |
| 8 | walks forward quickly | 0 | ✓ | 90 | 24 |
| 9 | walks forward taking long steps | 0 | ✓ | 92 | 25 |
| 10 | walks in place | 0 (1 draft rejected) | ✓ | 85 | 25 |
| 11 | turns around in place | 0 (1 draft rejected) | ✓ | 88 | 27 |
| 12 | turns left 90 degrees | 1 (+2 resamples, 3/3 drafts rejected) | ✗ yaw 82.5° | 60 | 110 |
| 13 | walks forward and turns around | 0 (+1 resample) | ✗ travel 0.0 m | 45 | 55 |
| 14 | sits down on a chair | 0 | ✓ | 90 | 25 |
| 15 | sits down and stays seated | 0 | ✓ | 92 | 22 |
| 16 | sits down and raises the right arm | 1 (3/3 drafts rejected) | ✓ | 90 | 43 |
| 17 | stands up from a chair | 0 | ✓ | 80 | 23 |
| 18 | crouches down slowly | 0 (2 drafts rejected) | ✓ | 88 | 27 |
| 19 | kicks with the right leg | 0 (1 draft rejected) | ✓ | 95 | 27 |
| 20 | jumps forward | 0 (1 draft rejected) | ✓ | 95 | 25 |

**What changed between the two runs is the generator, not the judge.** The same
controller, same registry, same checker: the only new mechanism is that a
measurable prompt now draws three drafts and keeps the closest one. That flipped
**#10 "walks in place"** (previously 1.97 m of travel, now a draft at ≤ 0.25 m,
alignment 85) and collapsed the edit count from 8 attempts to 2 — the loop no
longer spends edits repairing a draw it could have rejected. Steps fell 5.35 →
4.40, tokens 21 190 → 19 236, and wall time 776 s → 635 s, even though each
constrained generation now runs three samples in one model load.

R@1 dropped 0.30 → 0.25 while R@3 rose 0.50 → 0.65. That is a ranking property
of the same four near-duplicate goal pairs (slow/quick/long steps, left/right
arm), not a caption regression: the mean cosine is unchanged at 0.881. Foot
slide rose from 0.0056 to 0.0093 because the selected drafts include more
locomotion than before (mean travel 0.54 m); all 20 clips still pass the 0.03
threshold.

This run is also the first with a working `mld` environment (§3.1): every
generation in the table is a real diffusion sample, not `dataset_fallback`.

**Run-to-run spread.** Three runs of this suite now read 17, 18 and 18 of 20
(`_v3_final5`, `_v3_final6`, `_v3_final7`). Rejection sampling removed the most
volatile goal (#10) but did not remove variance: #12 and #13 flip depending on
what the three drafts contain. Quote 0.85–0.90 (n = 20) rather than a single
number, and read the per-goal column as the result.

Three things changed relative to the 5-goal battery of v1.1, and they point in
different directions, which is why the table is now the primary result rather
than the headline number:

- **The loop now covers goals it used to fake.** "kicks with the right leg" and
  "jumps forward" previously passed *because nothing measured them*: the
  requirement checker had no kick or jump test, so any physically valid clip
  scored a success. `ankle_lift_{left,right}` and `jump_rise` / `jump_clearance`
  are now measured (§2), and both clips clear their thresholds on their own
  merits (right ankle lifts 1.65 body heights vs 0.07 for the left foot; the
  pelvis rises 0.28 and the floor clears 0.24 at the apex). A goal the checker
  cannot see is no longer a success by default.
- **Edits are traceable.** 2 attempts, 2 applied, 0 rolled back, 0 refused
  before the backend ran (`instruction_coverage` 1.00). One edit has a
  measurable target (the arm raise: +2.30 body heights, non-target MPJPE 0.0046)
  and it followed; the turn edit is counted as applied because the clip changed,
  but `verifiable = false` — it leaves the 82.5° yaw short of 90°, which is
  exactly the "changed, not fixed" case the flag exists to expose.
- **Retrieval becomes harder, not worse.** R@1 is 0.25 because the suite
  contains near-duplicate goals that a single sentence cannot distinguish
  (walks slowly / quickly / with long steps; left vs right arm). Cosines stay at
  0.88, i.e. the captions are on-topic; the ranking is what breaks down. At
  n = 20 with 4 near-duplicates, quote R@3 (0.65) and the cosine, not R@1.

Frame retention is 1.00 because both applied edits were in-place reshaping (no
trims): the controller asked for an arm raise and a turn, not a removal.

The two failures are reported honestly, and both are now *clearly* the model's
limit rather than a bookkeeping artefact:

- **#12 turns left 90°.** All three drafts of this prompt were rejected by the
  rejection filter (`rej 3`), the two resamples also came back under-rotated,
  and the turn operator can limit a turn but not extend it. Final yaw 82.5°
  against a 90° target; the clip stayed valid, the requirement did not. The 110 s
  for this goal is the cost of exhausting the resample budget and one edit before
  saying so.
- **#13 walks forward and turns around.** The controller resampled although the
  first draft already met both checks, and the adopted draw has the turn
  (181°) but **0.0 m of travel** — `root_travel ≥ 0.5` fails. This is a
  controller decision error (it replaced a clip that passed) rather than a
  backend limitation, and the trace shows it: alignment 45, one resample, no
  edits.

`mean_target_vs_rest` is not quoted: on a near-static non-target region the
denominator approaches zero, so the ratio is unstable by construction.
`non_target_mpjpe_norm` is the preservation number to use.

### 5.2 Battery B — fixed draft (4 goals, LLM controller)

| Quantity | Value |
|---|---|
| Goal satisfaction | **4 / 4 = 1.00** |
| Caption-mediated R@1 / R@3 | 0.75 / 1.00 |
| Mean prompt↔caption GloVe cosine | 0.897 |
| Batch diversity (mean pairwise) | 0.138 |
| Foot slide / jerk (mean) | 0.0086 / 0.0093 body-heights |
| Edit attempts / applied / rolled back | 1 / 1 / 0 |
| Instruction-following rate | 1.00 |
| Non-target MPJPE (height-normalised) | 0.0051 |
| Target-vs-rest change ratio | 76.2 |
| Mean controller steps per goal | 3.5 |
| Tokens (4 goals) / wall time | 28 643 / 88 s |

The one applied edit is the intended loop behaviour. The draft for "raises the
right arm high" was captioned "raise their right hand up to their head" with
`arm_raise_right = 1.37`; the controller judged alignment 70, wrote
`raise the right arm higher`, the editor applied it, and the re-description
raised alignment to 90 while the non-target joints moved only 0.5 % of body
height. The trace is in §6.

### 5.3 Module (c) — captioning against held-out references

Module (c) is evaluated on 20 held-out HumanML3D test clips with their
ground-truth sentences, so the number does not depend on the generator:

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr-D | GloVe cos |
|---|---|---|---|---|---|---|
| 0.480 | 0.281 | 0.112 | 0.000 | 0.350 | 0.207 | 0.824 |

Corpus-level (n = 20, three references per clip); sentence-mean BLEU-1 is 0.435.

BLEU-3/4 are near 0 because the references are three short sentences and the
model's sentences are longer; this is a property of the benchmark, not evidence
that the captioner is broken. ROUGE-L, CIDEr-D and GloVe cosine are the metrics
to quote at this reference size; BLEU-4 becomes usable with the full
~30-clip HumanML3D test split.

### 5.4 The failure case, and the fix it produced

The most informative v1.1 episode was the sit edit (`ep03` in
`outputs/agent_bench/final_llm_v2`). The trace, verbatim:

```
step 3  edit_motion  "sit lower"
        -> invalid_clip: ground_penetration 2.47 > 0.15   (rolled back)
        quality_gate of the same file: passed=True score=100
step 4  describe_motion  (restored clip)      alignment 85
step 5  finish
        -> rejected by the verifier: pelvis_drop 0.049 < 0.15 m
step 6  edit_motion  "sit lower"  (only supported fix)
        -> invalid_clip: ground_penetration 2.70 > 0.15   (rolled back)
step 8  finish
        -> verification_failed, satisfied=false
        reason: both sit-lower attempts broke the clip; reporting the limitation
```

Following that episode produced four fixes, each verified separately:

1. **The recipe was wrong.** `edit_recipes.json` listed `"sit lower"` as a hint
   for `sit.slow_descend`, so the intent compiler slowed the descent instead of
   lowering the seat (54 → 81 frames, `pelvis_drop` unchanged). The hint is
   removed and a new `sit.deepen` recipe is added.
2. **`geometric_only` silently ran diffusion.** The planner emitted
   `edit_mode = geometric_only`, but `motionfix_edit` only branched on
   `speed | kinematic | diffusion`, so the plan's "no diffusion" instruction
   fell into the diffusion branch. That branch resets the root, and the clip
   came back 0.82 m below the floor *before* the sit even started. The mode is
   now mapped to `kinematic`, and an unknown mode raises instead of guessing.
3. **The depth edit had no operator.** `sit_deepen_postprocess.py` implements
   the actual request: pelvis down by δ, feet planted, knees solved by 2-bone
   IK. Re-running the same prompt on the same clip now gives `pelvis_drop`
   0.367 → 0.467 m with `ground_penetration` 0.000, `bone_cv` 0.000, feet
   displaced 0.0 m and 54 frames in / 54 out (no spurious retime).
4. **The gate could not see the floor.** `quality_gate` scored the underground
   clip 100. It now runs the shared `ground_clearance` check and fails it hard;
   re-scoring the old artifact gives `passed=False, score=67, hard_failures=["ground_clearance"]`,
   while the healthy clips still score 100.

The v1.1 lesson — the bottleneck was in the sit backend, not the controller —
was right, but it was only half the story: the planner, the editor's mode
router and the recipe file each contributed to the same failure. That is what
the per-layer metrics are for.

### 5.5 The checker is a tool, not a veto

In v1.1 the controller's `finish` was silently overruled by `verify_goal()`, and
a rejected finish forced another iteration. The controller could not tell the
difference between "your judgment is wrong" and "the tool refused".

v1.2 makes the measurement a tool the controller must call. The trace of a
successful episode now reads:

```
step 3  check_requirement
        satisfied=True · arm_raise_right=2.177 (>=1.8) · penetration=0.004
step 4  finish  reason: "ratio 2.177 and wrist +0.216 m above head; checker agrees"
        -> agent_claim_satisfied=True · checker_satisfied=True · agreement=True
```

If the claim and the measurement disagree, the controller is told once what
failed and may fix it or explicitly stop with `{"limitation": true}`. The
episode records both, so `agreement` is a property of the controller that the
benchmark can track instead of a rule that hides it. On the fixed-draft battery
the rule-based baseline (no LLM) shows why this matters: its caption-based
judgment said "not satisfied" on a clip whose measurements passed every
requirement, i.e. `agreement = False` — a real calibration error that the old
aggregate hid.

### 5.6 The two remaining rollbacks: a source that sits at the origin

The 20-goal run left two failures, both on sit edits (`ep16`, `ep17`):

```
ep16  "a person sits down and raises the right arm"
step 3  edit_motion "sit down lower (deeper into the seat)"
        -> edit_produced_invalid_clip · penetration 2.16 m · rolled back
        -> pelvis_drop stays 0.0055 m (goal asks 0.15 m)
step 5  edit_motion "raise the right arm higher" -> ok
step 8  finish  "sit depth unmet; edit budget exhausted"

ep17  "a person stands up from a chair"
step 3  edit_motion "stand up from the sit"
        -> edit_produced_invalid_clip · rolled back
step 4  generate_motion "a person stands up"   (the controller retries)
step 7  finish  satisfied=true
```

Both rollbacks were **not** caused by the instruction. The edit restored a
MotionsFix-native clip, and that path keeps the root at the origin: SMPL FK puts
the pelvis at 0.04 m and the feet at −0.77 m, so the whole clip is 0.8 m below
the floor before anything is edited. The pose (135-D) and the joints were
similarly displaced, so `sit_hold` had a floor to plant against 0.8 m too low.

Two changes, one for the cause and one for the diagnosis:

1. **Rigid regrounding at the source.** `motion_validity.offset_to_ground()`
   reads the floor of the first 10 % of the clip — before the action starts —
   and returns one vertical shift when that floor is below −0.05 m. A genuine
   articulation failure has a floor of ≈0 at the start, so it is left alone and
   still rejected. `motionfix_edit` applies the shift to both the joints and the
   pose transl, so the saved clip and its joints agree (`ground_shift` in the
   edit metadata), and the agent keeps the same rule as a last line of defence.
   Re-running `ep16`'s edit after the fix: `pelvis_drop 0.0055 → 0.1055 m`,
   penetration unchanged at 0.006, 60 frames in / 60 out.
2. **Blame the delta, not the clip.** `edit_motion` now reports
   `new_hard_failures` and `pre_existing_hard_failures` relative to the clip it
   started from. The sit drafts of `ep16`/`ep17` are near-static (largest
   pelvis-relative joint path 0.21 m over 60 frames, `body_moving = 0`), so the
   session gate fails them for `body_performance` *before* any edit. That is a
   property of the MLD draft, not of the depth operator, and the controller now
   sees it as inherited instead of as damage it caused.

With both in place the two goals pass end to end (`_v3_fix2`, same generator,
same controller, `success_rate = 1.0`, `agreement = 1.0`): `ep16` satisfies the
goal after one arm edit, and `ep17`'s draft already contains the sit-to-stand
cycle so it needs no edit at all.

What is left is upstream and belongs in the generator, not in the controller:
MLD occasionally returns a nearly static sit draft, and the 2-bone depth
operator can only deepen a movement that is there.

### 5.7 The sit hold stretched arm bones

Re-running the 20 goals after §5.6 (`_v3_final3`) exposed one more sit defect.
On "a person sits down and raises the right arm", the arm-raise edit produced a
correct pose — wrist 0.40 m above the head — and the agent rolled it back:

```
step 3  edit_motion "raise the right arm higher"
        -> edit_produced_invalid_clip · invalid=['bone_rigid'] · rolled back
step 5  edit_motion "raise the right arm above the head"
        -> edit_produced_invalid_clip · invalid=['bone_rigid'] · rolled back
step 7  check_requirement · arm_raise_right = -2.286  (arm was never raised)
```

`constraint_postprocess.raise_arm()` was innocent: applied to the same input
alone it gives `bone_cv 0.0197 → 0.0197`. The damage came from
`sit_hold_postprocess`, which runs after it to damp the post-contact rebound.
Two problems in that stage:

1. It clamped the pelvis by writing `j[t, J_PELVIS, up]` and the horizontal
   axes directly. The pelvis owns three bones (to both hips and to the spine),
   so moving that joint alone lengthens all three.
2. It blended the arm joints toward the contact pose with a linear
   `(1-α)·p + α·target`, which is not a rotation: upper arms and forearms grew
   up to 30 % (`0.110 → 0.146`, `0.135 → 0.167`), pushing `bone_cv` from 0.0197
   to 0.0851 — over the 0.08 gate.

The fix is to make both stages preserve the skeleton they are correcting:

- the rebound clamp is now a per-frame vertical drop fed into the same 2-bone
  leg IK as the depth operator (`deepen_sit(..., delta_per_frame=...)`), so the
  pelvis descends with the ankles planted instead of being translated;
- after the arm blend, each root-anchored arm chain
  (`13→16→18→20`, `14→17→19→21`) has its bone lengths reset to the contact
  frame while keeping its direction (`_restore_chain_lengths`).

Re-running the same instruction on the same clip now gives `valid=True`,
`bone_cv 0.0197`, penetration 0.002, 40 frames in / 40 out, and
`arm_raise_right = 1.969` with the wrist 0.163 m above the head — i.e. the pose
the controller asked for, on a clip the gate accepts. A `sit lower` edit on the
same clip still works (`pelvis_drop 0.432 → 0.582 m`, `valid=True`).

### 5.8 Grounding belongs at the door, not in the edit

§5.6 fixed the offset where the agent noticed it (the edit). The arm-raise
regression suite showed the same defect arriving through the other door: two of
the stored UI bases are displaced 0.85–0.88 m below the floor before any edit
runs, so a *correct* geometric arm edit inherited a hard `ground_clearance`
failure and the session reported `ok=false`.

Two changes, both about who owns which defect:

- **The session grounds the clip once, before the gate judges it.** After the
  backends finish and before `04_quality.json` is written, `run_session` reads
  `offset_to_ground`; if the source is rigidly displaced it shifts the joints
  and the pose transl together (recording `ground_shift`) and logs a trace step.
  A clip that starts on the floor is untouched, so genuine penetration still
  reaches the gate unchanged.
- **A rollback needs *new* invalidity.** `edit_motion` now compares the physical
  report of the source and the result: it rolls back when a valid clip became
  invalid, or when the edit added an invalid check. If the source was already
  invalid with the same checks, the edit is applied and the inherited defect is
  returned as `inherited_invalidity` — refusing every edit on a broken draft
  would lock the agent out of improving it, which is the opposite of useful.

Measured on the regression suite (`scripts/tests/test_arm_raise.py`, 33 checks):
the two displaced bases now report `ground_shift = 0.879 / 0.846 m`, the arm
edits apply (`arm_raise_lift 0.24 / 0.42`), and the suite goes from 31 passed /
2 failed to **33 passed / 0 failed**. One of those bases is also anatomically
stretched (its ankles sit 1.37 m below the pelvis on a 0.47 m torso), and that
one still *reports* a floor failure — correctly, because a rigid lift cannot fix
a stretched skeleton.

### 5.9 The v1.6 run: the numbers that go with the four fixes

Run: `outputs/agent_bench/_v3_final10` (same 20 goals, same controller, MLD with
generation-side rejection sampling, the v1.6 editor and dual caption). Each
episode runs in its own process with a 600 s watchdog, so a hung backend
reports `episode error` instead of freezing the suite.

| Quantity | v1.6 (`_v3_final10`) | v1.5 (`_v3_final7`) |
|---|---|---|
| Goal satisfaction | **20 / 20 = 1.00** | 18 / 20 = 0.90 |
| Caption-mediated R@1 / R@3 | 0.35 / 0.65 | 0.25 / 0.65 |
| Mean prompt↔caption GloVe cosine | 0.887 | 0.881 |
| Batch diversity (mean pairwise) | 0.129 | 0.143 |
| Foot slide / jerk / floor penetration (mean) | 0.0058 / 0.0044 / 0.0046 body-heights | 0.0093 / 0.0053 / 0.0033 |
| Episodes that drew a batch / drafts rejected | 19 / 17 | 19 / 16 |
| Edits per goal · episodes that resampled | 0.05 · 2 | 0.10 · 2 |
| Mean controller steps per goal | 4.25 | 4.40 |
| Checker consulted / agent↔checker agreement | 1.00 / 1.00 | 1.00 / 1.00 |
| Episodes that ended worse than their best measured clip | **0** | 0 |
| Mean seconds and tokens per goal | 33.7 s / 21 888 | 31.8 s / 19 236 |
| Wall time (20 goals, incl. module (c) bench) | 689 s | 635 s |

**1.00 here is one draw of n = 20, not a saturation claim.** The same suite
reads 0.90 (`_v3_final5`), 0.90 (`_v3_final7`) and 1.00 (`_v3_final10`); the
mechanism that moved it is rejection sampling at the source plus the
no-worse-than-best archive, and the per-goal column in `report.md` is the
result to read. Mean edits fell to 0.05 because only one goal still needed the
editor (goal 16, "sits down and raises the right arm"): three drafts were
rejected and one edit closed the gap, which is exactly the split the two
modules are supposed to have.

**Module (b) on its own, 21 declared cases** (`outputs/edit_bench3/report.json`,
schema `edit_bench_v2`, every case staged from its own copy of the source):

| Quantity | Value | Meaning |
|---|---|---|
| Cases / expected refusals | 21 / 8 | fixed battery over 8 operator families |
| Instruction coverage | 0.615 (8 / 13) | applied / cases that should be applicable |
| Refusal honesty | **1.00 (8 / 8)** | every rejection carries a named reason |
| Instruction following (applied edits) | **1.00** | the named attribute moved in the named direction |
| Silent no-ops | **0** | output identical to input |
| No-effect edits | 5 | ran, but the named attribute did not move → rolled back |
| Rollback rate (new physical damage) | 0.00 | no applied edit introduced a hard failure |
| Target / rest motion ratio | 79.2 | localisation of the change |
| Non-target MPJPE (height-normalised) | 0.0077 | collateral displacement |
| Frame retention | 1.021 | truncating edits would show < 1 |
| Mean seconds per case | 10.4 | one backend call + measurement |

Per family: `arm_raise` 4 / 4 applied (ratio 111.8), `sit_deeper` 1 / 1 (25.9),
`sit_remove_preamble` 1 / 1 (1.9), `speed` 2 / 2, and the eight `refuse` cases
0 / 8 applied with 8 named refusals. The five `no_effect` rows are the honest
middle: `stand_up` on a clip that is already standing, `turn` on a seated clip,
`walk faster` on a still clip, `trim_trailing` on a clip with no tail, and one
`sit` depth that the leg reach clamps to zero. Each is reported with the
attribute it expected and the previous clip is restored, so a zero here is
visible in the table rather than scored as a success.

**The contract behind both review points** is executable:
`python scripts/tests/test_agentic_contract.py` runs 66 checks against the code,
`report.json`, one finished UI episode and this document, and reads
**66 / 66 passed** (`contract_checks.json`). It fails the build if the battery
drops below 15 cases, if a refusal is unnamed, if a finish lands on a clip worse
than one the episode already measured, if the UI episode never calls the editor,
if a step carries no reason, if a reported number cannot be recomputed from its
clip, or if this note loses its requirement map (§0), its process table (§3), its
severity-ranked problems (§7) or its definitions of done (§8).

## 6. What the trace looks like

`outputs/agent_bench/<run>/epNN_*/agent_run.md` is the human-readable version of
the JSON trace. One real episode (`final_dataset_llm/ep01`), verbatim:

```
step 1  generate_motion  "a person raises the right arm high above the head"
        thought: No clip yet, so generate the requested motion.
step 2  describe_motion
        caption: a person raise their right hand up to their head
        arm_raise_right = 1.367 · wrist_above_head_right = +0.105 m
        alignment 0 (nothing judged yet)
step 3  edit_motion  "raise the right arm higher"
        thought: wrist is above the head but arm_raise_ratio 1.367 is below
                 1.8 and the caption says "up to their head" rather than
                 high above; one edit to raise it.
        alignment 70
        -> instruction following = True, non-target MPJPE = 0.0031
step 4  describe_motion
        caption: a person wave with their right hand
        arm_raise_right = 2.177 · wrist_above_head_right = +0.216 m
        alignment 90
step 5  finish
        thought: ratio 2.177 and wrist 0.216 m above head confirm it; the coarse
                 caption says "wave" but the ground-truth kinematic measure is
                 what the rubric says to trust.
```

The caption is not trusted blindly, and neither is the first judge: the same
episode shows a low-confidence caption being overruled by the measured wrist
height, and in the failure case of §5.4 the verifier overrules the controller
itself.

## 7. What is still wrong (by severity)

Severity here is how much a defect can mislead a reader, not how hard it is to
fix. Each row says where it is visible.

### P0 — would change the conclusion if ignored

| Defect | Evidence | Why it matters |
|---|---|---|
| The success rate is a single draw of n = 20 | 0.90 (`_v3_final5`), 0.90 (`_v3_final7`), 1.00 (`_v3_final10`) | 1.00 is not a saturation claim. Without k draws per goal there is no interval and no pass@k, so a 2-goal swing is indistinguishable from noise. §8.1 |
| The 0–100 alignment is an LLM judgement | `agreement = 1.00` says the judge matched the checker on these 20 clips, not that either is right | It can only be validated against human ratings. §8.2 |
| Caption metrics are dataset-level | BLEU/ROUGE/CIDEr/GloVe are computed on 20 held-out HumanML3D clips with references | Pipeline clips have no reference sentence, so nothing in §5.3 measures the caption of a *generated* clip. §8.5 |
| The open-vocabulary editor is routing, not a result | `test_open_vocab.py` 10/10 with a fake backend | SimMotionEdit consumes MotionFix-style SMPL features; the joints→SMPL fit and the checkpoint are still missing, so no open-vocabulary edit has ever run on a real backend. §8.4 |
| BLEU-3/4 are ≈0 at this reference size | §5.3 | Quoting them as a caption regression would be wrong; they only become usable on the full test split. |

### P1 — capability edges the tables already show

| Defect | Evidence | Consequence |
|---|---|---|
| MLD sometimes returns a near-static draft | goal 17 in `_v3_final3`: 60 frames, largest pelvis-relative joint path 0.21 m, `body_moving = 0` | No relative operator can add motion the draft does not contain; the honest remedy is another draw, not an edit. |
| `sit lower` is a constant drop | `edits.py`: 0.15 m, 0.18 m for "much lower", clamped by leg reach | "a little" and "a lot" are not parsed. §8.6 |
| 5 of 21 battery cases are `no_effect` | §5.9 per-case table | These are real capability edges (stand up on a standing clip, turn on a seated clip, walk faster on a still clip, trim a clip with no tail, a depth clamped to zero). They are reported, not hidden, but coverage 0.615 is the honest number. |
| An applied edit can be unverifiable | `n_unverifiable_edits = 0` today; the category exists | If a future family has no localisation ratio, it would be excluded from the following rate rather than counted as a miss — visible, but easy to misread. |
| Speed and duration edits change frame count | frame retention 1.02 in the battery | Sub-clip extraction (trim, remove a preamble) is measured by retention, not by pose error; a purely temporal case can only be judged by the following check. |

### P2 — engineering debt

| Defect | Evidence | Consequence |
|---|---|---|
| `third_party/TM2T` is a local checkout, not a submodule | `.gitignore` ignores the tree and the checkpoints | A fresh clone has no learned caption until it is cloned there; `caption_edited` falls back and records `source`, but the report's learned-caption column would quietly become a template. |
| There is no CI | `test_agentic_contract.py` is run by hand and writes `contract_checks.json` | Drift between the claim and the code is caught only when someone remembers to run it. §8.7 |
| The UI replay is manual | `outputs/session/prompt_logs/…` lives under the git-ignored `outputs/` | The screenshots in §6 cannot be regenerated from a clone without redoing the clicks. |
| `benchmark.py` runs episodes serially | 689 s for 20 goals | A k = 5 battery is ~1 h; parallel episodes over one GPU would need model-server work. |
| The acceptance page is generated from one episode | `build_acceptance.py --episode <dir>` | It is only as current as the run it was pointed at; the numbers come from `report.json`, but the trace does not. |

## 8. Next steps, each with a definition of done

Ordered by how much each one changes what the experiment can claim. "Done" means
the check lives in the repo and fails when the property is broken — the same
standard as §3, not "we plan to".

### 8.1 k draws per goal, reported as pass@k with an interval

Why: turns §7's P0 into a measured distribution, and separates "the model
cannot" from "this draw missed" at the episode level rather than the draft
level. Method: freeze the controller, run `benchmark.py` with k = 5 seeds
(100 episodes), report per-goal pass@1/5, mean ± 95 % Wilson interval, and the
per-goal flip matrix. DoD: `report.json` carries `k`, `pass_at_1`, `pass_at_k`
and the interval; `test_agentic_contract.py` requires `k ≥ 5` and n ≥ 100
before the headline number is printed. Cost: ~1 h of GPU per battery; the
watchdog already exists.

### 8.2 Human rating of the final clips

Why: the alignment judge has never been validated (§7 P0). Method: 40 final
clips (2 per goal), two raters, 5-point "does this motion do what the sentence
says", blind to the controller's score. DoD: Spearman ρ between the judge and
the mean human score is reported, the 5 largest disagreements are listed with
their measurements, and ρ ≥ 0.7 is the acceptance bar for quoting the judge at
all. Risk: rater availability; a single rater still gives a ceiling on the
claim.

### 8.3 TMR R-Precision and FID through a joints→SMPL fit

Why: R@1 at n = 20 has a 0.05 resolution and all goals are in-distribution
(§2 (a)). Method: fit SMPL 6-D poses to the pipeline's joints, then run the TMR
evaluator already vendored under `third_party/motionfix/eval-deps`. DoD: FID and
R-Precision (top-1/2/3) on the full test split, plus the fit residual, so a bad
fit cannot be mistaken for a bad generator. Risk: the fit is a new source of
error and needs its own check (re-projected MPJPE < 0.05 body heights).

### 8.4 Real SimMotionEdit edits

Why: the open-vocabulary path is plumbing (§7 P0). Method: complete the
joints→SMPL fit from §8.3, supply `preproc.sim_file` and the checkpoint, and
route one instruction outside the registry through it. DoD: an `expect=apply`
case in the battery whose `backend` is `simmotion`, judged by the same
measurement and rollback rule; if it fails, the failure is a named refusal in
the same table. Risk: their preprocessing expects MotionFix features; a
conversion mismatch would show up as a constant offset, which the measurement
would catch but not fix.

### 8.5 Caption consistency as an accuracy number

Why: `caption_verdict` is a decision signal, never scored (§7 P0). Method:
label 60 (clip, goal) pairs as fit / not fit from the measurements, then score
the verdict's `needs_fix` as precision/recall against those labels, and report
learned-vs-rule agreement per class. DoD: a table with precision, recall and the
confusion matrix; `caption_verdict` is quoted in the report only when recall on
"needs_fix" ≥ 0.9. Risk: the labels come from the same measurements the checker
uses, so this measures consistency, not truth — §8.2 is the truth anchor.

### 8.6 Amount-aware depth, and the rest of the P1 list

Why: `sit lower` is a constant (§7 P1), and "a little" / "a lot" is the most
common phrasing a user will try. Method: parse the amount into δ, clamp against
the leg reach the IK already computes, and add two declared battery cases
(`sit a little lower`, `sit much lower`) whose expected deltas differ by ≥ 2×.
DoD: both cases pass with distinct `pelvis_drop` values, and the clamp is
asserted on a short-legged source. Also in this tranche: a `speed ×0.5` case
whose check is speed, not distance, so the temporal family has a positive
example.

### 8.7 Continuous integration

Why: the contract is only as good as the last manual run (§7 P2). Method: run
`test_agentic_contract.py`, `test_agent_edits.py`, `test_resample.py` and
`test_open_vocab.py` (no GPU) on every push, with the GPU suites nightly on a
self-hosted runner. DoD: a red check on a deliberately broken contract (e.g.
remove a rail) before the workflow is trusted; the badge is linked from this
note. Risk: the GPU suites need the `mld` / `motionfix` environments, so the
nightly job must pin them.

## 9. Reproduction

```powershell
# Battery A: MLD generator, 20 goals, <=2 edits, module (c) on 20 clips.
# --goal-timeout puts every episode in its own process with a watchdog, so a
# hung backend is reported as an episode error instead of freezing the suite.
conda run -n motionfix python scripts/agent/benchmark.py ^
  --goals-file test-samples/agent/goals_20.json ^
  --generator mld --max-edits 2 --caption-n 20 --goal-timeout 600 ^
  --out outputs/agent_bench/_v3_final10

# module (b) alone: 21 declared must-edit cases over 8 families
conda run -n motionfix python scripts/agent/edit_benchmark.py ^
  --out outputs/edit_bench3

# the same benchmark, with that battery folded into report.json / report.md
conda run -n motionfix python scripts/agent/benchmark.py ^
  --goals-file test-samples/agent/goals_20.json --generator mld ^
  --edit-bench outputs/edit_bench3/report.json ^
  --out outputs/agent_bench/_v3_final10

# one acceptance page from a finished run (self-contained HTML)
python scripts/agent/build_acceptance.py ^
  --report outputs/agent_bench/_v3_final10/report.json ^
  --episode outputs/session/ui_runs/<episode> ^
  --out outputs/agent_bench/_v3_final10/acceptance.html

# Battery B: fixed dataset draft (controller is the only variable)
conda run -n motionfix python scripts/agent/benchmark.py ^
  --generator dataset --max-edits 2 ^
  --out outputs/agent_bench/final_dataset_llm

# one free-form goal, watch every decision
conda run -n motionfix python scripts/agent/agent.py ^
  --goal "a person walks forward and waves with the right hand" ^
  --generator mld --max-edits 2

# resample tool: rank N drafts of one prompt (real MLD, ~60 s)
conda run -n motionfix python scripts/tests/probe_resample.py "a person walks in place" 3

# resample tool: ranking / availability / budget, no MLD needed (16 checks)
python scripts/tests/test_resample.py

# open-vocabulary routing (SimMotionEdit), fake backend, no GPU (10 checks)
python scripts/tests/test_open_vocab.py

# the UI's agent mode without a browser (same handler the page calls)
python -c "import sys; sys.path[:0]=['scripts/session']; from chat_server import handle_agent; print(handle_agent({'prompt':'a person waves with the right hand','max_edits':1,'generator':'mld'})['assistant_message']['content'])"

# the contract behind both review points, checked against code + a finished run
python scripts/tests/test_agentic_contract.py

# scripted baseline with the same tools (no LLM)
conda run -n motionfix python scripts/agent/agent.py ^
  --goal "..." --no-llm --generator mld

# module (c) alone, against HumanML3D references
conda run -n motionfix python scripts/agent/metrics.py --caption-bench --n 20

# instruction gate + physical validity (no GPU needed)
python scripts/tests/test_agent_edits.py

# one revise instruction on an existing run, before/after numbers
python scripts/tests/probe_revise.py ^
  outputs/agent_bench/<run>/ep16_*/step01_generate ^
  "raise the right arm higher" outputs/agent_probe/arm
```

Artifacts per run:

- `<out>/report.md`, `report.json` — the tables in §5;
- `<out>/epNN_*/agent_run.json` — full controller trace (thought, action, args,
  observation, assessment, token usage);
- `<out>/epNN_*/agent_run.md` — same trace as a table;
- `<out>/epNN_*/stepNN_*/03_edited_preview.gif` — per-step clips;
- `<out>/contract_checks.json` — the 66 contract checks, one line of evidence
  each, written by `scripts/tests/test_agentic_contract.py`;
- `outputs/session/prompt_logs/<date>-<topic>/log.json` + `*.png` — the manual
  web replay, one message per step, with the run dir, trace, score, previews and
  screenshots of the page at that point (v1.6: four messages, two of which the
  controller finished without an edit because the measurement already passed).
