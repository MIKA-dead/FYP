# Progress since the MotionFix limitation note

**To:** senior  
**Date:** 28 October 2026  
**Project:** Motion Studio (intent → draft → observation → measured edit → preview)  
**Scope:** everything done after the note that MotionFix is the bottleneck, including the choice on the two searches and the motions currently on the page.

Your note asked for two searches:

1. An automatic evaluator that can give the agent **textual** feedback. Motion transcription / motion-to-text is enough. It does not have to be a vision-language model.
2. Motion editing that is stronger than MotionFix, which is already a few years old (SIGGRAPH Asia 2024).

This note answers both, then records what was actually changed in the system and what the latest page tests show.

---

## 1. MotionFix, as we ran into it

MotionFix (Athanasiou et al., SIGGRAPH Asia 2024; code [atnikos/motionfix](https://github.com/atnikos/motionfix)) is the first public benchmark of source–instruction–target triplets. Its editor, TMED, is a conditional diffusion model. The dataset and the checkpoint are installed locally and were the revise backend.

The failure we kept seeing is the one you named. TMED is trained on short relative instructions (“wave to the side”). A long user sentence is shortened before it reaches the model. The source clip often does not contain the action the sentence asks for, and the diffusion step does not reliably add it. A quality score could still pass. On the page this showed up as a lateral hop where a forward walk was requested, a revise that produced no new clip, and a high score on a motion that did not match the sentence.

---

## 2. Textual feedback — what we use

The agent reads one English sentence about the clip it just made, then the edit step changes only a fault that sentence names. That sentence is written from the joint metrics and the timeline, saved as `06_motion_caption.json`, and printed in the chat (“自动动作描述” / “反馈摘要”). It is not a vision-language model. It names travel, yaw, pelvis drop, one arm versus the other, and a walk before a sit.

A learned captioner is not installed. The slot for one is `MOTION_CAPTION_CMD` in `scripts/session/motion_caption.py`. The model that slot is for is **TM2T** (Guo et al., ECCV 2022; [EricGuo5513/TM2T](https://github.com/EricGuo5513/TM2T)): motion-to-text on HumanML3D, the same skeleton family as the drafts. The environment variable has not been pointed at a checkpoint.

The HumanML3D caption stored beside a clip describes that dataset clip. After we trim, plant, or straighten, it is no longer a description of the motion on the page, so it is not the feedback sentence.

The page tests below show the limit of the rule sentence. Jump, kick, and raise-both-arms all come back as “a person is moving”. The edit step then has no named fault, so it keeps the draft.

---

## 3. Editor — what we use

Generate does not call MotionFix. The clip on the page is the first HumanML3D caption hit, then a small geometric change when the observation names a spin on a sit, a walk before a sit, a short turn, root travel on a turn, or a turned walk or wave.

The editor prepared to replace MotionFix is **SimMotionEdit** (Li et al., CVPR 2025; [lzhyu/SimMotionEdit](https://github.com/lzhyu/SimMotionEdit)). It trains on the same source–instruction–target triplets, and adds a motion-similarity head so the edit stays tied to the instruction. The slot is `scripts/backends/simmotion_edit.py` (module id `simmotion_edit`). It still needs their environment, pretrained weights, and a similarity file, and it is not the path that produced the previews. Revise can still call the installed TMED checkpoint.

---

## 4. What changed in the system

### 4.1 Feedback, as far as question 1 allows today

- After quality runs, the session writes a caption and a short “user wanted / observed” line. The UI prints the four generate stages and that caption.
- Refine still prefers a closed JSON (keep, forbid, ops, counts). The caption is context for that step. It is not yet a learned description of the joints.
- `MOTION_CAPTION_CMD` is the place a TM2T-class model would go.

### 4.2 Generate no longer shows a MotionFix sample

The page’s generate path is four stages (`scripts/backends/stage_generate.py`). Diffusion, MLD, and TMED are not called.

1. **Draft.** Rank HumanML3D captions by token overlap, take the first hit, convert it onto the preview skeleton, cut at 100 frames. The draft is allowed to be the wrong action.
2. **Observe.** Metrics, timeline, and the rule caption. Saved as `01_observation.json`.
3. **Edit only a named fault.** Sit with a large yaw: remove the spin. Sit whose observation says there is a walk before the sit: drop that preamble. Turn short of 150° with the body still moving: add the rest of the half-turn on the existing poses. Turn that travels: plant the root, keep the pose relative to the pelvis. Walk or wave with yaw above 40° or a large sideways path: straighten. Anything else, including run, jump, bow, kick, and a raised arm: keep the draft.
4. **Preview.** Skeleton gif.

A walk prompt used to die before stage 1. The walk recipe is marked `edit_mode = speed`, and the old coder builder treated that as a speed revise and demanded a previous clip. Generate now returns no coder steps and goes straight to the four stages. That is why “a person walks backward” failed once on 28 Oct and passed on the retry. The failure is archived.

### 4.3 Revise, for when the current clip is kept

These paths are for a follow-up sentence, not for the ten generate tests in section 5.

- A dataset swap whose skeleton or up-axis does not match the preview is refused. The page keeps the current clip.
- “More steps” is a gait join: one step whose start and end are the same kind of footfall, then the other foot, no position blend across a mismatched pose, no zero-speed frame at the join. If the checks fail, the previous clip stays.
- A different action is retrieved and joined with a short transition. A failed join does not fall through into diffusion.
- “Make the right leg match the left” stays on the current clip and scales that lift. It does not ask TMED to redraw the walk.
- If a coder writes no new `.npy`, the last good clip is copied forward and the quality gate fails the step as “edit not applied”, so the next revise still has a clip.

### 4.4 Quality

The gate used to pass a rigid spin and a knee-only squat, because it only checked yaw, drop, and travel. Sit and turn now also require the body to keep moving after facing is removed (`body_moving` ≥ 8). A score of 100 still only means the hard checks for that **action label** passed. Jump, kick, and raise-arms are not in that label set, so “moving a little” can score 100.

---

## 5. Current results on the page

All of these were sent from http://127.0.0.1:8787, one prompt at a time, mode = generate. Logs:

- four original prompts: `outputs/session/prompt_logs/20261027-stages/`
- six new prompts: `outputs/session/prompt_logs/20261028-more/`

### 5.1 Original four (27 Oct)

These are the clips we were willing to keep after the four-stage change.

| Prompt | Text hit | What the observation did | Score |
|---|---|---|---|
| a person sits down | 001090, “sits down on something”, then trimmed to 41 frames | Draft walks, then sits. Preamble cut from frame 23. Pelvis drop 0.31, travel 0.47. About 8 frames at the start are still labelled as a walk. | 100 |
| a person walks forward and waves with the right hand | M013032, walks, bows slightly, waves, 100 frames | Yaw about 50° straightened. Travel 0.68, right-wrist span 0.91, yaw 0. | 100 |
| a person turns around 180 degrees | M009648, walks, turns, then sprints, 100 frames | Draft yaw about 62°, travel 2.50. Completed to 180° and the root was planted. Body still moving. Travel 0. | 100 |
| a person walks forward slowly | 000257, “walks forward slowly”, 100 frames | No named fault. Travel 1.83, yaw about 18°. Kept. | 100 |

### 5.2 Six further actions (28 Oct)

| Prompt | Text hit | Observation and edit | Score |
|---|---|---|---|
| a person runs forward | 000034, “runs and jumps forward”, 84 frames | Called a forward run. Travel 9.35, yaw about 76°. No listed problem, so kept. The jump in the caption is still in the clip. | 100 |
| a person jumps | 002534, “jumps while spinning”, 83 frames | Called “moving”. Travel 0.17, yaw about 388°. Spin was not listed, so kept. | 100 |
| a person bows | 004175, “bows then walks forward”, 100 frames | Called a bow. Travel 3.40. The following walk was not listed, so kept. | 100 |
| a person kicks with the right foot | 001247, “kicks with right foot”, 85 frames | Text is close. Observation says “moving”, travel 0.08. Kept. | 100 |
| a person raises both arms | 000124, “raises both arms and waves”, 100 frames | Observation says “moving”, travel about 0. The wave in the caption stays. Kept. | 100 |
| a person walks backward | first send: error, no preview. Retry: 002396, “walks forward, backward, and forward again”, 100 frames | First send hit the speed-revise bug above. Retry: yaw 94° and lateral 1.14 were straightened. The caption after that is “walking forward”, travel 0.77. The backward part was not kept. | 100 |

---

## 6. Where the two questions stand

**Textual feedback.** The rule sentence is still what most runs show, but the M2T slot is now filled: `scripts/backends/tm2t_infer.py` runs the TM2T M2T transformer on the edited clip, and `motion_caption.py` prefers it whenever its checkpoints exist (`M2T_BACKEND=rules` forces the old behaviour). On the existing clips it says “jump and spin in the air”, “wave with their right hand”, “sit down on a stool” — detail the metric rule cannot produce. The sentence is now stable enough to drive an edit loop, which is what section 7 adds.

**Editor.** The ten clips are the first caption hit, plus a geometric change when the rule names yaw, a sit preamble, or root travel. SimMotionEdit is the replacement prepared for the MotionFix slot, and it is not on this path yet. That is why a run can still include a jump, a jump can still be a spin, and a backward walk can be straightened into a forward walk. The score stays 100 because those faults are outside the current hard checks. Turning SimMotionEdit on needs its checkpoints first.

---

## 7. Pipeline v1 (2 Nov)

The loop the note above was missing is now in the repo:

1. **Edit** the current clip (`03_edited.npy`), as in 0.5.
2. **Describe** the result with the M2T model (`06_motion_caption.json`, `source: tm2t_m2t_transformer`).
3. **Compare and propose**: the LLM receives the goal, the caption **before** the last edit, the caption **after** it, and the measured problems, and returns `{done, verdict, reason, next_edit_prompt}` (`scripts/intent/edit_proposer.py`).
4. **Repeat** with `next_edit_prompt` until done, no caption change twice, a worse score, or the budget (`scripts/orchestrator/run_loop.py`).

The loop also rolls back: if an edit lowers the quality score by more than 5, the previous clip is kept and the round is logged as `regressed`. Every round is written to `<run>/07_loop/07_loop.json`; the page has an `自动迭代` button that calls the same path (`POST /api/iterate`).

Both halves are now live: M2T is verified on the existing clips, and the LLM controller returns real proposals. On 2 Nov it was asked for a slow forward walk with the right arm raised and answered `raise the right arm high overhead`, reasoning that the caption still only said “a person slowly walk forward”. The round then stopped at `edit_not_applied`: the `walk_slow` edit path cannot raise an arm, so the loop kept the original clip. The controller is therefore ahead of the editor — the next gain comes from the editor, not from the loop.

One gateway note for the record: the `Model is unavailable` errors were a stale model id, not a permissions problem. `deepseek-v4-flash` had been removed from the OpenCode Go catalog; `glm-5.3-flash` is the new default and works. `opencode models opencode-go` is the authoritative list.

### 7.1 Editor fix: geometric arm raise (2 Nov, evening)

The `edit_not_applied` above was a planner/editor gap, and it is now closed for
arm raises. "Raise the right arm high / overhead" is applied **on the clip
already on screen** instead of rebuilding it:

- `constraint_postprocess.py::raise_arm()` unfolds the elbow (upper-arm and
  forearm lengths unchanged) and then rotates elbow and wrist about the shoulder
  to a target height, ramped over ~1 s. Targets are 0.8× / 1.2× / 1.8× / 2.6× of
  head-above-shoulder for `slightly` / `mid` / `high` / `overhead` — 2.6× is the
  arm straight up, which is the height the existing wave clips already have, so
  "overhead" now matches what it says. The swing plane is built from the shoulder
  line; using shoulder-minus-pelvis made the axis flip between frames when the
  torso was upright, which showed up as a one-frame jump. `side: left|right|both`,
  and a `step_up` flag for "a bit higher" on an already-raised clip.
- `refine_schema.py` parses EN/中文 raise phrasing into `refine.arm_raise` and
  forces `geometric_only`; an arm raise is local, so trims and phase drops are
  stripped (`ops=[geometric_only, arm_raise]`, `phases=[full]`) — without that,
  "raise both arms" on the sit clip lost its descent and scored 25.
  `plan_motion_edit.py` lets the raise win over the MLD-rebuild, dataset, sit-hold
  and wave-rebuild branches; `run_session.py` seeds from the previous clip only.
- `quality_gate.py` adds a hard check `arm_raise_applied` plus
  `arm_raise_ratio`, so a raise that did not happen is caught instead of
  silently scoring 100.

Verified end to end on the walk clip with the real LLM (~15 s, was 12 min then
`edit_not_applied`): round 1 applied the raise (`arm_raise_ratio` 2.585, score
100, M2T caption `a person walk forward and wave with their right hand`), round
2 the controller answered `goal_met`. The proposer now also receives
`arm_raise_ratio`, because TM2T has no height words and would otherwise call an
overhead arm a "wave" forever.

Test battery (5 Nov, `scripts/tests/test_arm_raise.py`, 33/33 pass):

| level | cases | what it covers | result |
|---|---|---|---|
| parser | 16 | EN/中文 side+height, "higher/再抬高点", negatives (`slow down a bit`, `walk two more steps`, `no spinning`, two `as much as the left` cases that must stay `limb_match`), plus the `phase_speed` wildcard fallback | 16/16 |
| geometry | 10 | walk / run / bow / sit / wave × left/right/both; only the requested elbow+wrist move; bone lengths, root path and the other arm bit-identical; idempotent second pass; `high → higher → higher` = 1.8 → 2.15 → 2.5; no edit jump beyond the source clip's own cuts | 10/10 |
| end to end | 7 | right / left / both, sit clip, two-turn "high then higher", wave clip already overhead, plus a speed regression (`slow down a bit` → 100 → 125 frames) | 7/7, score 100, 8–10 s each |

Artifacts: `outputs/tests/arm_raise_20261105_final/report.md` (+ `results.json`,
`contact_sheet.png`, `compare_walk_right_overhead.gif`), loop run in
`outputs/tests/loop_arm/`. A one-turn page replay is logged in
`outputs/session/prompt_logs/2026-11-05-arm-layer/`.

---

## References

- Athanasiou et al., MotionFix: Text-Driven 3D Human Motion Editing, SIGGRAPH Asia 2024. https://github.com/atnikos/motionfix
- Guo et al., TM2T, ECCV 2022. https://github.com/EricGuo5513/TM2T
- Li et al., SimMotionEdit, CVPR 2025. https://github.com/lzhyu/SimMotionEdit
