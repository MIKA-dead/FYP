# Agentic pipeline v1.5 — the LLM calls (a), (b), (c) by itself

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

**v1.2 changes the four things v1.1 listed as known gaps** (details in §7):

- the geometric sit operator now has a real *depth* edit (2-bone leg IK with
  planted feet), so `sit lower` no longer sinks the body through the floor;
- the editor's ability is an executable registry (`scripts/agent/edits.py`),
  so an impossible instruction is rejected in milliseconds instead of after a
  wasted backend pass, and `instruction_coverage` is measured;
- the acceptance check is a **tool** (`check_requirement`), not a hidden veto:
  the controller must consult it, sees the measurements, and its own claim is
  recorded next to the verdict (`agreement`);
- `quality_gate` now checks the floor, and both it and the agent share one
  definition of physical validity (`scripts/backends/motion_validity.py`).

**v1.3 closes the rollbacks and the blind checks that the 20-goal run exposed**
(details in §4.6, §4.7 and §7):

- a restored MotionsFix-native source keeps its root at the origin, so SMPL FK
  returned a body 0.8 m below the floor and the agent rolled back a *correct*
  edit. Sources are now rigidly regrounded once (`offset_to_ground`), and an
  edit is only blamed for gate failures it introduced (`new_hard_failures` vs
  `pre_existing_hard_failures`);
- the post-contact sit hold was moving the pelvis joint alone (stretching its
  child bones) and blending arms linearly (up to +30 % bone length), so the
  arm-raise edit on a sit clip failed `bone_rigid`. Both stages now preserve the
  skeleton, and the 20-goal run applies that edit cleanly;
- "kicks" and "jumps" used to pass because **nothing measured them**. They are
  now checked (`ankle_lift`, `jump_rise` / `jump_clearance`), so a goal the
  checker cannot see is no longer a default success;
- a relative operator cannot undo what the clip already does, so contradictions
  ("walk in place") and edits with no material to work on (`walk` on a clip with
  0.0 m travel) are refused before the backend runs, instead of damaging the
  clip.

**v1.4 gives the controller a way to act on a generation-side gap** (details in
§1 and §7):

- the editor is *relative*, so no edit can create travel or rotation that the
  draft does not contain. `resample_motion(prompt, n)` draws `n` drafts of one
  prompt in a single MLD process (one model load), measures each with the same
  requirement the checker uses plus a relative `requirement_gap`, adopts the
  closest valid one, and reports every candidate it rejected. Two per episode;
- if no draft came from a fresh draw — the generator is down and everything fell
  back to the retrieved clip — the tool returns `resample_unavailable` and does
  not charge the budget, so a deterministic repeat can never masquerade as a
  successful resample.

**v1.5 moves the filtering to the generator itself, and opens the editor beyond
the closed set** (details in §1 and §7):

- the `mld` environment works again. The blocker was Windows Application Control
  refusing to load three compiled extensions (`thinc/backends/linalg`,
  `blis/py`, `spacy/…/span_group`) — reinstalling the same wheels reproduces the
  same block, because the policy decision follows the file. Nothing on the
  text→motion path needs spaCy, so the vendored MLD keeps its only `spacy` import
  lazy (inside `RawTextDataset`), and `thinc 8.3.14` + `blis 1.2.0` were the pair
  whose binaries the policy accepts. `demo.py` now runs end to end again, which
  is what makes the numbers below real MLD numbers rather than `dataset_fallback`;
- **generation-side rejection sampling**: when the prompt names a measurable
  attribute, one MLD process draws a batch of drafts and `scripts/agent/sampling.py`
  keeps the closest one *before the clip enters the loop*. This is the same
  "walks in place" prompt that produced 1.35 m, 0.12 m and 0.02 m of travel on
  three consecutive draws — a single draw cannot tell "the model cannot do it"
  from "this draw missed";
- `resample_motion` now uses that same policy (shared `sampling.rank` /
  `select`), draws a second batch when the first contains nothing acceptable, and
  reports `accepted` / `batches` alongside the candidates it rejected;
- the Web UI can drive the loop (`agent` in the mode picker → `POST /api/agent`):
  the trace, the final caption and the preview come back in one response, so the
  demo is the benchmark's loop, not a separate script;
- `edit_motion` can route instructions *outside* the closed registry to the
  open-vocabulary editor (SimMotionEdit) when `AGENT_OPEN_VOCAB_EDIT=1` and the
  backend reports itself ready. The result is judged by the same measurement and
  rollback rule as a closed-set edit; props/camera/appearance stay impossible.

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
│  ↺ ②–⑤ until `finish` or budget                                          │
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
| `generate_motion` | MLD (`mld_generate.py`, conda env `mld`) | `prompt` | frames, generator used, kinematic summary, `draft_rejection` (drawn / accepted / rejected / chosen) when a batch was drawn, preview path |
| `resample_motion` | MLD `--replication n --allinone` (one model load) | `prompt`, `n` (2–4) | every draft's measured attributes, requirement gap, which draft was kept |
| `describe_motion` | TM2T `M2T_EL4_DL4_NH8_PS` + VQ tokenizer | — | caption, caption source, GloVe/token similarity to the goal, kinematic summary |
| `check_requirement` | `scripts/agent/verify.py` (joint measurements) | — | pass/fail per requirement, the measured value, the physical validity report |
| `edit_motion` | session revise (intent compiler → geometric edit / MotionFix) | `instruction` | before/after metrics, target change, non-target MPJPE, instruction-following checks, `no_change` flag, instruction family, grounding offset |
| `finish` | — | `reason`, `limitation` | agent claim vs. checker verdict, agreement |

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
for §4.1.

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
| `instruction_coverage` | Supported instructions ÷ all instructions the controller attempted | How much of what the controller wants is inside the tool's ability at all |
| Unsupported count | Instructions rejected by the registry before the backend ran | Wasted steps avoided, and a map of what users ask for but the editor cannot do |

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

## 3. Experimental setup

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

## 4. Results

### 4.1 Battery A — MLD generator (20 goals, LLM controller)

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
| Drafts drawn / rejected / episodes resampled | 60 / 16 / 2 | — |
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

This run is also the first with a working `mld` environment (§7): every
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

### 4.4 The failure case, and the fix it produced

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
   re-scoring the old artifact gives `passed=False, score=67,
   hard_failures=["ground_clearance"]`, while the healthy clips still score 100.

The v1.1 lesson — the bottleneck was in the sit backend, not the controller —
was right, but it was only half the story: the planner, the editor's mode
router and the recipe file each contributed to the same failure. That is what
the per-layer metrics are for.

### 4.5 The checker is a tool, not a veto

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

### 4.6 The two remaining rollbacks: a source that sits at the origin

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

### 4.7 The sit hold stretched arm bones

Re-running the 20 goals after §4.6 (`_v3_final3`) exposed one more sit defect.
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

### 4.8 Grounding belongs at the door, not in the edit

§4.6 fixed the offset where the agent noticed it (the edit). The arm-raise
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
# Battery A: MLD generator, 20 goals, <=2 edits, module (c) on 20 clips
conda run -n motionfix python scripts/agent/benchmark.py ^
  --goals-file test-samples/agent/goals_20.json ^
  --generator mld --max-edits 2 --caption-n 20 ^
  --out outputs/agent_bench/_v3_final7

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

- `<out>/report.md`, `report.json` — the tables in §4;
- `<out>/epNN_*/agent_run.json` — full controller trace (thought, action, args,
  observation, assessment, token usage);
- `<out>/epNN_*/agent_run.md` — same trace as a table;
- `<out>/epNN_*/stepNN_*/03_edited_preview.gif` — per-step clips.

## 7. Limitations and next steps

Fixed in v1.2 (each verified, not just described):

- **Sit depth.** `sit lower` now runs a real depth operator with planted feet
  and preserved bone lengths; the underground clips of v1.1 are gone, and
  `quality_gate` fails any clip that still goes through the floor.
- **Mode routing.** `geometric_only` no longer means "run diffusion anyway";
  unknown modes raise instead of silently choosing a backend.
- **Impossible instructions.** Rejected by the executable capability registry
  before any backend work, with a reason and a suggestion, and counted in
  `instruction_coverage`.
- **The acceptance check.** It is a tool the controller must consult, its
  verdict is returned to the model, and the controller's own claim is recorded
  next to it (`agreement`), instead of being silently overridden.

Fixed in v1.3:

- **Restored sources at the origin.** A MotionsFix-native clip keeps its root at
  0, so FK lands the body ~0.8 m below the floor; that was the real cause of the
  last two rollbacks. `offset_to_ground` lifts such a source once (joints and
  pose transl together), and the agent still rejects genuine articulation
  failures because those start on the floor.
- **Attribution of gate failures.** `edit_motion` splits gate failures into
  `new_hard_failures` and `pre_existing_hard_failures`. A near-static MLD draft
  that already fails `body_performance` no longer makes a correct depth edit
  look destructive.
- **The sit hold stretched bones.** The post-contact rebound clamp moved the
  pelvis joint alone (lengthening its three child bones) and blended the arms
  linearly (up to +30 % bone length). It now lowers the pelvis through the leg
  IK and restores each arm chain's bone lengths, so the arm-raise edit on a sit
  clip passes `bone_rigid` instead of being rolled back.
- **Instruction gate: contradictions and empty locomotion.** "Walk in place,
  keep the root planted" is now rejected as `in_place` (a relative operator
  cannot remove travel), `verify_goal` reads "in place" as a *max* on travel,
  and a `walk` edit on a clip with < 0.15 m of travel returns
  `edit_needs_existing_locomotion` instead of rebuilding the clip and erasing
  a 186° turn.
- **Sit depth matched to the wording.** `sit lower` now drops 0.15 m (0.18 m
  for "much lower") instead of 0.10 m, so one edit clears the 0.15 m acceptance
  threshold instead of needing a second identical pass.
- **Goals the checker could not see no longer pass by default.** `ankle_lift`
  and `jump_rise` / `jump_clearance` are measured, so "kicks" and "jumps" have
  to be visible in the joints. Two of the 20 goals used to succeed purely
  because nothing tested them.

Fixed in v1.4:

- **A generation-side gap had no action.** `resample_motion(prompt, n)` draws
  `n` drafts of the same prompt from a single MLD process, measures each with
  the checker's own requirement plus a relative gap, adopts the closest valid
  one and reports the rest; `resample_unavailable` when every candidate fell
  back to the retrieved clip, without charging the budget.

Fixed in v1.5:

- **The MLD environment.** The blocked file was `thinc/backends/linalg`
  (`WinError 4551`), and reinstalling the same wheels reproduces the block
  because the verdict follows the file. Text→motion never needs spaCy, so the
  vendored MLD imports it lazily, and `thinc 8.3.14` + `blis 1.2.0` are the
  binaries this policy loads; `demo.py` runs again, so Battery A is real MLD
  rather than `dataset_fallback`.
- **Rejection sampling happens where the constraint can still be satisfied.**
  `scripts/agent/sampling.py` is the single policy: `constraints(goal)` comes
  from the same `verify_goal` the checker uses, `rank` orders drafts by physical
  validity then relative gap, and `_mld_draft` draws a batch when the goal names
  a measurable attribute, keeping the closest draft *before* the pipeline runs.
  The same prompt that gave 1.35 m / 0.12 m / 0.02 m of travel on three draws is
  now filtered at the source.
- **The loop is reachable from the UI.** `agent` in the mode picker posts to
  `/api/agent`; the response carries the trace (`step / action / thought /
  line`), the final caption and a session block, so the preview and quality
  panels are the real artifacts of the same loop the benchmark measures.
- **The editor can leave the closed set.** With `AGENT_OPEN_VOCAB_EDIT=1` and a
  ready SimMotionEdit checkout, wording outside the registry is routed to it as
  written and judged by the same measurement + rollback rule
  (`backend: "simmotion"` in the observation). It stays off by default because
  the benchmark's contract is the closed set, and a half-installed backend must
  not turn honest rejections into `simmotion_infer_failed` noise.

Still open, in order of expected value:

- **Success is still one draw per run in Battery A.** Rejection sampling cuts
  the variance of a single draft, but the headline rate is still measured on 20
  goals × 1 run. A battery with k draws per goal (k · 20 episodes) is what would
  separate "the model cannot" from "this run drew badly"; `distinct_candidates`
  and `drafts_rejected` in the report are the inputs for that analysis.
- **The generator sometimes returns a static clip.** On "a person stands up from
  a chair" MLD produced 60 frames whose largest pelvis-relative joint path was
  0.21 m (`body_moving = 0`) — a seated pose, not a movement. The gate now says
  so and the agent no longer blames the edit, but no operator can add motion the
  draft does not contain, and the in-place filter has nothing to select.
- **The open-vocabulary path is plumbing, not a result.** `simmotion_edit.py`
  speaks joints-in/joints-out, while SimMotionEdit consumes MotionFix-style
  SMPL features; a real run needs a joints→SMPL fit plus their checkpoint and
  `preproc.sim_file`. `test_open_vocab.py` proves routing, capability text and
  failure reporting with a fake backend, and the docs state the requirement —
  it does not yet claim better edits than the geometric operators.
- **The sit depth is a constant.** `sit lower` drops a fixed 0.15 m (0.18 m for
  "much lower") rather than parsing "a little" or "a lot"; the IK already
  computes the reachable range, so clamping δ against it is the natural next
  step. It now clamps against the leg reach, so a short-legged draft gets less.
- **R@1 is still coarse at n = 20.** Resolution 0.05, and all goals are
  in-distribution by design. Quote it as a between-battery signal. TMR-based
  R-Precision / FID needs a joints→SMPL fit and is the correct next measurement.
- **The alignment score is a judge.** `agreement` reports how often it matched
  the measurement, which is a calibration number, not an accuracy number. A
  human rating of the final clips on a 5-point scale is the right validation.
- **Caption metrics are dataset-level.** BLEU-3/4 are near 0 with three short
  references per clip: the references exist, but the model paraphrases and its
  sentences are longer, so few 3-gram matches survive and the brevity penalty
  compounds it. ROUGE-L, CIDEr-D and GloVe cosine are the metrics to quote at
  this size; a caption–kinematics consistency check ("does the caption's verb
  match the measured attribute?") is the metric that would actually validate
  the loop's use of the caption — and `agreement` only measures the controller,
  not the captioner.
- **Battery B is not a generation result.** Its success rate must not be
  compared with Battery A's as if the controller improved; the draft is fixed
  and easier. The comparable quantity is the trace, not the headline rate.
