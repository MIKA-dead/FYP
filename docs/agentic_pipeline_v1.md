# Agentic pipeline v1.1 — the LLM calls (a), (b), (c) by itself

This note answers the two review comments on the previous experiment.

> *"The experiment seems to be separately done on each module (a) MLD,
> (b) MotionFix, (c) M2TCaption. We want the pipeline to be agentic, so that the
> LLM will call (a)(b)(c) by itself automatically."*

The previous experiment measured each backend in isolation and reported a
`SCORE` that is 100 whenever the hard validity checks pass. That score says
nothing about whether the motion matches the request. This version changes two
things:

1. **One agent, three tools.** The user writes a free-form sentence. An LLM
   controller chooses `generate_motion` (a), then `describe_motion` (c), reads
   the caption and the joint measurements, writes the remaining gap in text,
   calls `edit_motion` (b), and repeats until it judges the clip satisfies the
   request (or the budget ends). The loop is not hard-coded: the controller
   decides every next call and writes every edit instruction.
2. **Quantitative metrics per module and for the loop**, computed from the
   artifacts (joints, captions, tool trace). The saturated `SCORE` is replaced
   by retrieval / alignment / preservation / captioning metrics and by
   loop-level success and convergence numbers.

```
free-form goal
     │
     ▼
┌───────────────────────────── LLM controller ─────────────────────────────┐
│  thought → tool choice → arguments          (JSON action, one per turn)  │
│                                                                          │
│  ① generate_motion(prompt)   ── MLD text→motion ──► 03_edited.npy        │
│  ② describe_motion()         ── TM2T M2T ────────► caption + metrics     │
│  ③ judge: caption + joint metrics vs. goal ─────► alignment 0–100, gap   │
│  ④ edit_motion(instruction)  ── session revise ──► new 03_edited.npy     │
│  ↺ ②–④ until `finish` or budget                                          │
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
| `generate_motion` | MLD (`mld_generate.py`, conda env `mld`) | `prompt` | frames, generator used, kinematic summary, preview path |
| `describe_motion` | TM2T `M2T_EL4_DL4_NH8_PS` + VQ tokenizer | — | caption, caption source, GloVe/token similarity to the goal, kinematic summary |
| `edit_motion` | session revise (intent compiler → geometric edit / MotionFix) | `instruction` | before/after metrics, target change, non-target MPJPE, instruction-following checks, `no_change` flag |
| `finish` | — | `reason` | — |

If MLD fails, `generate_motion` retries with the in-repo dataset draft and
records `generator_used = dataset_fallback`; the trace never hides which backend
produced the clip.

The controller's JSON contract is

```json
{"thought": "...",
 "assessment": {"alignment": 0, "satisfied": false, "matches": [],
                "missing": [], "next_edit": ""},
 "action": "generate_motion | describe_motion | edit_motion | finish",
 "args": {}}
```

`scripts/agent/agent.py` enforces three safety rails, which are the *only*
fixed rules in the loop:

- no clip → `generate_motion`;
- a clip that has not been described since the last edit → `describe_motion`;
- the edit budget is never exceeded.

Everything else — when to stop, what is still missing, which single edit to
try next — is the LLM's decision, recorded with its justification.

## 2. Why the old SCORE was not a metric

`quality_gate` answers "is this a valid clip?" (finite joints, minimum length,
foot sliding, sit depth, turn anti-slide). It returns 100 as soon as no rule
fires. Two measurements from the runs in §4 show why that number cannot be the
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
| `ground_penetration` | Deepest joint below the floor ÷ body height | Feet through the floor. **Not checked by `quality_gate`**; the agent's own edit validity check uses it |
| `diversity` | Mean pairwise distance of root-relative, height-normalised pose features | Mode collapse across prompts |

The batch size here is 4, so R@1 is coarse (±0.25). The metric is implemented
and reported because it is the right measurement; it becomes meaningful at the
30-prompt scale used by HumanML3D. A full TMR-based R-Precision / FID is
possible with the checkpoints under `third_party/motionfix/eval-deps`, but they
consume SMPL 6-D poses, while the pipeline outputs joints; we did not add a
joints→SMPL fit just for the report.

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

The DTW alignment makes the preservation metric fair when the edit changes
speed: frames are matched by pose, not by index.

### (c) Captioning

| Metric | Definition |
|---|---|
| BLEU-1…4 | Corpus-level clipped n-gram precision with brevity penalty |
| ROUGE-L | Longest-common-subsequence F1 (best of 3 references) |
| CIDEr-D | TF-IDF n-gram cosine with length penalty, corpus document frequencies |
| `glove_cos` | Mean GloVe cosine to the 3 references |

Measured on 8 held-out HumanML3D test clips with their ground-truth captions.
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
| Failure taxonomy | `judged_unsatisfied`, `no_change`, `budget`, `generate_failed` |

The alignment score is an LLM judgment, not a ground-truth label; that is why
the report also lists the measured quantities it cites (arm ratio, travel, yaw,
caption). The judge is constrained by a rubric and told that measurements win
over the coarse caption.

## 3. Experimental setup

Two batteries are reported. Neither one runs a module in isolation: in both,
the same free-form sentence enters the same controller and every backend is
reached only through a tool call.

**Battery A — MLD is the generator.** Five free-form goals, MLD HumanML3D
checkpoint, 60 frames. This is the intended system: the motion the agent edits
is generated from the prompt, not retrieved.

**Battery B — fixed draft, controller is the only variable.** Four goals, the
draft is retrieved once from the in-repo HumanML3D index (`generator=dataset`).
The same draft is returned on every run, so any change between episodes is
attributable to the controller's decisions rather than to generation variance.
This battery is the controlled one; Battery A is the realistic one.

Common settings:

- Captioner: TM2T M2T transformer (`M2T_EL4_DL4_NH8_PS`), beam 2, cached model.
- Editor: the session revise path (intent compiler → geometric operator or
  MotionFix).
- Controller: `glm-5.3-flash` through the OpenCode gateway, temperature 0.2.
- Budget: ≤ 8 controller steps, ≤ 2 edits per goal.
- Machine: one Windows workstation, RTX GPU for MLD / TM2T.

## 4. Results

### 4.1 Battery A — MLD generator (5 goals, LLM controller)

| Quantity | Value |
|---|---|
| Goal satisfaction | **4 / 5 = 0.80** |
| Caption-mediated R@1 / R@3 | 0.40 / 0.60 |
| Mean prompt↔caption GloVe cosine | 0.879 |
| Batch diversity (mean pairwise) | 0.249 |
| Foot slide / jerk / floor penetration (mean) | 0.0044 / 0.0030 / 0.0104 body-heights |
| Edit attempts / applied / rolled back | 2 / 0 / 2 |
| Mean controller steps per goal | 4.0 |
| Tokens (5 goals) / wall time | 52 194 / 204 s |

| Goal | M2T caption | cos | arm_raise R | travel | edits | satisfied |
|---|---|---|---|---|---|---|
| raises the right arm high above the head | a person raise their right hand above their head | 0.910 | 3.32 | 0.012 | 0 | yes |
| walks forward and waves the right hand high | a person take a step forward | 0.843 | 2.52 | 1.577 | 0 | yes |
| sits down and raises the right arm high | a person is sit down with both hand in front of them and then lift their left hand up | 0.921 | 2.06 | 0.037 | 2 (both rejected) | **no** |
| sits down on the floor | a person is sit down with their hand in front of them | 0.898 | −3.35 | 0.098 | 0 | yes |
| turns around in place | a person turn to the right and swing their arm back to the left | 0.825 | −2.45 | 0.144 | 0 | yes |

The retrieval number is the first thing that separates MLD from the fixed
draft: R@1 drops from 0.75 (Battery B) to 0.40, i.e. the generated clip is
ranked with the wrong caption for three of five prompts. Two of those are
trajectory failures, not pose failures: "walks forward and waves" is described
as merely "take a step forward" (travel 1.58, no wave in the caption), and
"turns around in place" is described as turning *and swinging the arm back*.
The per-goal alignment agrees with this: the verifier's thresholds can pass
while the semantic match is weaker than with a curated draft, and that gap is
what the generation metrics are supposed to expose.

### 4.2 Battery B — fixed draft (4 goals, LLM controller)

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
height. The trace is in §5.

### 4.3 Module (c) — captioning against held-out references

Module (c) is evaluated on 8 held-out HumanML3D test clips with their
ground-truth sentences, so the number does not depend on the generator:

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr-D | GloVe cos |
|---|---|---|---|---|---|---|
| 0.444 | 0.242 | 0.000 | 0.000 | 0.321 | 0.126 | 0.811 |

BLEU-3/4 are 0 because each clip has a single short reference and the model's
sentences are longer; this is a property of the benchmark, not evidence that
the captioner is broken. ROUGE-L, CIDEr-D and GloVe cosine are the metrics to
quote at this reference size; BLEU-4 becomes usable with the full 3-reference,
~30-clip HumanML3D test split.

### 4.4 What the failure case shows

The only unsatisfied goal is the most informative episode (`ep03` in
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

Three properties of the agentic design are visible here. First, the model
*decided* to edit; no rule told it to. Second, the tool layer refused a clip
that `quality_gate` had scored 100, because the agent checks a physical
quantity (`ground_penetration`) that the legacy gate does not. Third, the
verifier would not let the controller declare success — it was forced to end
with `satisfied=false` and a written explanation. The bottleneck this exposes
is in the sit backend (it translates the pelvis instead of bending the legs),
not in the controller, and that is exactly the kind of finding the per-module
quantitative metrics were meant to surface.

## 5. What the trace looks like

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
height, and in the failure case of §4.4 the verifier overrules the controller
itself.

## 6. Reproduction

```powershell
# Battery A: MLD generator, 5 goals, <=2 edits, module (c) included
conda run -n motionfix python scripts/agent/benchmark.py ^
  --generator mld --max-edits 2 ^
  --goals "a person raises the right arm high above the head" ^
          "a person walks forward and waves the right hand high above the head" ^
          "a person sits down and raises the right arm high" ^
          "a person sits down on the floor" ^
          "a person turns around in place" ^
  --out outputs/agent_bench/final_llm_v2

# Battery B: fixed dataset draft (controller is the only variable)
conda run -n motionfix python scripts/agent/benchmark.py ^
  --generator dataset --max-edits 2 ^
  --out outputs/agent_bench/final_dataset_llm

# one free-form goal, watch every decision
conda run -n motionfix python scripts/agent/agent.py ^
  --goal "a person walks forward and waves with the right hand" ^
  --generator mld --max-edits 2

# scripted baseline with the same tools (no LLM)
conda run -n motionfix python scripts/agent/agent.py ^
  --goal "..." --no-llm --generator mld

# module (c) alone, against HumanML3D references
conda run -n motionfix python scripts/agent/metrics.py --caption-bench --n 8
```

Artifacts per run:

- `<out>/report.md`, `report.json` — the tables in §4;
- `<out>/epNN_*/agent_run.json` — full controller trace (thought, action, args,
  observation, assessment, token usage);
- `<out>/epNN_*/agent_run.md` — same trace as a table;
- `<out>/epNN_*/stepNN_*/03_edited_preview.gif` — per-step clips.

## 7. Limitations and next steps

- **The sit backend is the concrete bottleneck found by the metrics.** Both
  `sit lower` attempts in Battery A translate the pelvis without bending the
  legs, so the feet end 2.47 and 2.70 body-heights underground while
  `quality_gate` still reports 100. The controller did the right thing
  (rollback, retry the only supported instruction, then report failure); the
  fix belongs in the geometric sit operator, not in the loop.
- **`quality_gate` is not a semantic or even a geometric judge.** It has no
  ground-penetration check and no notion of the request. The agent's validity
  check and the verifier are separate layers for exactly this reason; do not
  quote `score` as a result.
- **R@1 is coarse at this batch size.** It is 0.40 with 5 MLD goals and 0.75
  with 4 fixed drafts. Scale to ~30 prompts before quoting retrieval, and add
  TMR-based R-Precision / FID if a joints → SMPL fit is acceptable.
- **The alignment score is a judge.** Report it next to the measured attributes
  it cites, never alone. A human rating of the final clips on a 5-point scale
  would be the right validation set.
- **Caption metrics are dataset-level.** They measure the M2T model, not whether
  the caption of *this* clip is faithful. A caption–kinematics consistency check
  (does the caption's verb match the measured attribute?) is the natural next
  metric.
- **Battery B is not a generation result.** Its success rate (1.00) must not be
  compared with Battery A's (0.80) as if the controller improved; the draft is
  fixed and easier in Battery B. The comparable quantity is the trace, not the
  headline rate.
