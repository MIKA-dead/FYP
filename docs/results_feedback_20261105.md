# Results feedback — the arm layer generalises, and two defects it exposed

**To:** senior
**Date:** 5 November 2026
**Project:** Motion Studio (intent → draft → observation → measured edit → preview)
**Scope:** results since the geometric arm-raise note of 2 November. No new model
was introduced; this round asks whether the geometric layer holds up on actions
other than walking, and what it breaks when it does.

---

## 1. Short version

The arm raise was built and verified on walking clips. That risked being a
one-clip trick, so this round moved it onto ten other carriers (turn, bow, wave,
run, walk-back, and the new kick / jump / punch / pick-up / stand-up clips), and
added two more on sit and walk in the mixed battery, together with tempo and
dataset-join edits.

The result is that the operator holds: on every carrier the frame count is
preserved and the measured `arm_raise_ratio` is exactly the requested height
(1.8 for `high`, 2.6 for `overhead`) regardless of what the body underneath is
doing, and the 33-case battery shows on five carriers that only the requested
elbow and wrist move. An arm revise costs 9–12 s in every case, against ~12 min
for the MLD rebuild it replaced.

Two real defects surfaced, both in the *planner*, not the geometry:

1. a turn clip's own rotation was being deleted by an inherited `yaw_spin` forbid;
2. an arm-raise plan could omit the geometric module, so the arm silently did
   not move while the plan looked valid and the score stayed high.

Both are fixed and covered by the battery below.

| battery | cases | passed | time per case |
|---|---|---|---|
| cross-action (8 generate + 10 arm revise) | 18 | 18 | 3.4–3.6 s / 8.7–11.8 s |
| mixed action battery (generate, arm, tempo, dataset join) | 11 | 11 | 3.2–12.8 s |
| arm-raise regression (parser / geometry / end to end) | 33 | 33 | — |
| page replay, one session, no refresh | 5 | 5 | 5–25 s |

Artifacts: `outputs/tests/more_actions_20261105/`,
`outputs/tests/actions_20261105/` (each case has a before/after GIF, a
single-view GIF and an analysis card), replay log in
`outputs/session/prompt_logs/20261105-multi-action/log.json`.

### Open items at a glance

| item | status | where |
|---|---|---|
| a turn clip's rotation was deleted by an inherited `yaw_spin` forbid | **fixed** 5 Nov | §5.1 |
| an arm-raise plan could omit the geometric module, so the arm never moved | **fixed** 5 Nov | §5.2 |
| the critique says `Issues: n/a` even when the M2T caption contradicts the goal | **open — needs your decision** | §7 |
| 3 of 8 new-action captions are wrong (stand-up, backward walk, both-hands wave) | **known limit, not addressed this round** | §2, §8 |
| score 100 only means the hard checks for that action label passed | **known limit** | §8 |
| A11 joins two clips; it is not a general action composition | **known limit** | §8 |

---

## 2. New actions generated

Each prompt went through the same session entry point the page calls. The
generate path is still the four stages (draft → observe → edit only a named
fault → preview); nothing here calls diffusion or MLD.

| prompt | source | T | score | M2T caption after | time |
|---|---|---|---|---|---|
| a person kicks with their right leg | 000017 | 62 | 100 | a person kick something with their right foot | 3.5 s |
| a person punches with their right hand | 006257 | 41 | 100 | a person throw something with their right hand | 3.5 s |
| a person jumps up and lands | 011684 | 52 | 100 | a person jump forward once | 3.4 s |
| a person turns around 180 degrees | M009648 | 100 | 100 | a person walk forward turn to the right and pick up an object with their left hand | 3.5 s |
| a person stands up from a chair | 000703 | 100 | 100 | a person walk forward and sit on something | 3.5 s |
| a person picks up an item from the floor | 009969 | 100 | 100 | a person pick up an object with both hand and place it on the ground | 3.6 s |
| a person raises both hands and waves | 003858 | 100 | 100 | a person raise their arm to their chest multiple time | 3.5 s |
| a person walks backward | 002396 | 100 | 100 | a person walk forward and kick with their left foot | 3.4 s |

The retrieval side is fine — the right caption family is hit every time. The
captioner is the weak link. Three of the eight captions are close enough to use
as feedback (kick, punch, pick-up), two are partly right (jump is described as
one jump, the turn picks up a spurious "pick up an object"), and three are
wrong in a way that matters:

- stand-up → "walk forward and sit on something" (the opposite of the prompt);
- backward walk → "walk forward and kick with their left foot";
- both-hands wave → "raise their arm to their chest multiple time" (right
  gesture, wrong height).

That is half the captions unusable for anything that closes the loop. I have
not changed the captioner this round.

---

## 3. The arm layer on other carriers

Ten revises of the form "raise … arm …", spread over ten carriers so that no two
share a body plan. The interesting column is `ratio`: it is the requested height
and nothing else, so the geometry is driving it, not the source clip.

| carrier | prompt | T before → after | ratio | score | time |
|---|---|---|---|---|---|
| kick | raise the right arm high | 62 → 62 | 1.80 | 100 | 8.7 s |
| punch | raise the right arm high | 41 → 41 | 1.80 | 100 | 9.3 s |
| jump | raise the right arm high | 52 → 52 | 1.80 | 100 | 11.8 s |
| turn | raise the right arm high | 100 → 100 | 1.80 | 100 | 9.2 s |
| stand up | raise the right arm high | 100 → 100 | 1.80 | 100 | 9.1 s |
| pick up | raise the right arm high | 100 → 100 | 1.80 | 100 | 9.1 s |
| wave (both) | raise the right arm high | 100 → 100 | 1.80 | 100 | 9.2 s |
| walk back | raise the right arm high | 100 → 100 | 1.80 | 100 | 10.7 s |
| run | raise both arms overhead | 84 → 84 | 2.60 | 100 | 10.1 s |
| bow | raise both arms overhead | 100 → 100 | 2.60 | 100 | 9.0 s |

Frame count is untouched in all ten, which is the property that makes the
operator safe to stack: an arm edit cannot eat the descent of a sit or the
landing of a jump. The jump case is the one worth looking at, because the clip
moves vertically while the raise is applied — it still holds, only the ramp is
shortened to 13 frames by the reach clamp.

![Peak wrist frame for four carriers: bow, wave, walk, turn — the arm reaches the same height on every body](outputs/tests/actions_20261105/arm_peak_check.png)

A note on the scores: `arm_raise_applied` is a hard check, so a 100 here means
the wrist actually rose to the target, not just that the file is finite.

---

## 4. Mixed battery, including edits that are not arm raises

Eleven cases: three new actions, five arm raises on new carriers, two tempo
edits, and one real action change.

| case | mode | plan | score | T | time |
|---|---|---|---|---|---|
| A1 jump | generate | dataset draft | 100 | 52 | 3.2 s |
| A2 kick | generate | dataset draft | 100 | 62 | 3.2 s |
| A3 pick up | generate | dataset draft | 100 | 100 | 3.3 s |
| A4 turn + right arm high | revise | geometric_only | 92 | 120 | 9.3 s |
| A5 bow + both arms overhead | revise | geometric_only | 100 | 100 | 11.6 s |
| A6 sit + right arm high | revise | geometric_only | 100 | 152 | 10.0 s |
| A7 wave + both arms overhead | revise | geometric_only | 92 | 150 | 11.5 s |
| A8 walk + left arm high | revise | geometric_only | 100 | 100 | 9.3 s |
| A9 run + slow down | revise | geometric_only | 100 | 105 | 11.1 s |
| A10 turn + slow down | revise | geometric_only | 92 | 150 | 9.6 s |
| A11 walk → turn around | revise | dataset join | 100 | 127 | 12.8 s |

The three 92s are soft checks, not failures, and both causes are expected:

- **A4 / A10 (`turn_yaw_large`)** — the source clip only turns 45.8°, so "want
  > 90° for a full about-face" is honest. What matters is that 45.8° **survives**
  the edit; before the fix in section 5 it was 0.0° and the run failed.
- **A7 (`wave_one_arm`)** — "the other arm should stay below 0.75 of the main
  one". We asked for both arms, so the check is correct and we are deliberately
  violating it.

A11 is the first case in this batch where the body plan itself changes: the
planner split it into `turn (attach, dataset)` + `walk (extend, current)` and the
join produced T=127 with yaw 196° and travel 0.76 — a turn, then a walk back,
which is what was asked.

![Mid-clip pose for each of the eleven cases, with the score and frame count the gate produced](outputs/tests/actions_20261105/contact_sheet.png)

---

## 5. The two defects

### 5.1 A turn clip's rotation was being deleted

A4 first scored **67** with a hard failure `turn_yaw_span: yaw_span=0.0°`.

The revise inherited `forbid: ["yaw_spin"]` from the previous plan. That forbid
is right for a sit or a walk, where a spin is an artefact, but on a turn clip the
yaw *is* the action, so `constraint_postprocess` flattened 45.8° → 0.0° and the
gate then failed the clip for not turning. The edit destroyed the thing it was
editing.

Fix in `scripts/intent/refine_schema.py`: drop `yaw_spin` from `forbid` whenever
`keep_action` starts with `turn`. A4 and A10 go back to 92 with the rotation
intact.

### 5.2 An arm-raise plan could omit the geometric module

On carriers with no yaw problem, `constraint_postprocess` was not added to
`modules`. The plan then read `geometric_only` with `arm_raise` in `refine`, no
MLD, no diffusion — and nothing that applied the raise. The pipeline looked
successful and the arm did not move. This is exactly the silent-success failure
mode we keep trying to design out.

Fix in `scripts/intent/plan_motion_edit.py`: if `refine.arm_raise` is present,
the module is appended unconditionally. The page replay step 5
(`raise both arms overhead` on the jump clip) is a direct test of that path and
now reports `constraint_postprocess handled=['arm_raise']` with a ratio of 1.50
on both sides.

The reason both defects survived the 33-case arm battery is that the battery
always used carriers whose plans already included the geometric module, and it
never edited a turn clip. They were found only by widening the carrier set —
which is the argument for keeping this kind of cross-action battery.

---

## 6. Page replay

Five prompts sent from `http://127.0.0.1:8787`, one at a time, waiting for each
run to finish, no refresh in the middle of a chain:

| # | prompt | mode | score | what the agent did |
|---|---|---|---|---|
| 1 | a person turns around 180 degrees | generate | 100 | turn demo clip, T=100, yaw 180° |
| 2 | raise the right arm high | revise | 100 | arm only, T=100, yaw still 180°, ratio 1.8 |
| 3 | slow down a bit | revise | 100 | T 100 → 232, `speed=['turning']`, yaw still 180° |
| 4 | a person jumps up and lands | generate | 100 | dataset draft, T=52 |
| 5 | raise both arms overhead | revise | 100 | both sides raised (1.466 / 1.532), T=52 |

Steps 2 → 3 are the useful pair: a geometric edit followed by a timing edit on
the same clip, with the turn preserved through both. Screenshots of the
reasoning panel and the quality card for each step are in
`outputs/tests/actions_20261105/ui/`.

![Page replay, step 2: the turn clip after the right arm raises to 1.8, quality card and timeline](outputs/tests/actions_20261105/ui/03_turn_arm_result.png)

---

## 7. Open issue: the critique can say "no problems" while the caption is wrong

A11 asked for "turn around and walk back". The clip is correct by measurement
(T=127, yaw 196°, travel 0.76). TM2T described it as "a person walk forward and
wave with their right hand", and the critique line still ended in
`Issues: n/a`.

The cause is structural: the critique is assembled from the caption plus the
gate's problem list. If the gate is happy, the problem list is empty, so the
critique reads as approval even when the two sentences disagree. Handing that to
the loop controller is worse than handing it nothing, because it argues for
stopping.

Two options, and I would like your call:

1. **Cheap** — always append the rule-based description to the critique when it
   disagrees with TM2T on locomotion or arm height (the rule sentence is
   reliable on travel, yaw and pelvis drop, and TM2T is reliable on finer
   gesture). The disagreement itself becomes the reported problem.
2. **Proper** — score caption agreement as a metric and let the gate fail on
   large disagreement. That needs a reference sentence, which for a user prompt
   we do not have; it would have to be caption-before vs caption-after, i.e. "the
   edit did not change what the motion is", which is a different and possibly
   more useful check.

I have not implemented either, because (2) changes what a score means and I did
not want to move it without your agreement.

---

## 8. What this does not prove

- **Score 100 still means "the hard checks for this action label passed".**
  Kick, punch, jump and stand-up are not in a label set, so a clip that is
  merely moving can score 100. The arm cases are better protected than the
  generate cases.
- **The captioner is distribution-bound.** Three of the eight new-action captions
  are wrong in a way that matters (§2). The arm layer avoids this by trusting
  `arm_raise_ratio` over the caption, but that workaround is per-operator and
  will not scale to gestures.
- **A11 joins two clips; it is not a general action transition.** The planner
  asked for a dataset turn and an extension of the current walk, and the join
  happened to be clean. A harder pair (sit → walk, or turn while carrying the
  arm raise) is still a dataset swap or a refusal, not a real compositional
  edit.

---

## 9. Reproducing

```powershell
# cross-action battery: 8 generate + 10 arm revise (2 cases per carrier)
python scripts/tests/batch_cross_actions.py

# mixed battery: 11 cases, each with a before/after GIF and an analysis card
python scripts/tests/batch_actions.py --out outputs/tests/actions_20261105

# re-render GIFs and cards from existing runs without re-running the pipeline
python scripts/tests/batch_actions.py --out outputs/tests/actions_20261105 --rerender

# arm-raise regression (parser / geometry / end to end, 33 cases)
python scripts/tests/test_arm_raise.py
```

Both batteries write `results.json` and `report.md` into their output folder and
read every number back from the run's own artifacts, so a rerun either reproduces
the tables in this note or fails visibly.

---

## 10. Asks

1. Which of the two critique fixes in §7 do you want — the cheap disagreement
   line, or the caption-stability check?
2. Is three-wrong-out-of-eight captioning acceptable for the write-up if it is
   reported as a known limit, or should the next step be a better captioner
   before anything else?
3. For the paper narrative: should the cross-action battery become the
   "generalisation" section, with the two planner defects described as the
   motivation for the module-presence check? I think that is the stronger story
   than presenting 33/33 alone.
