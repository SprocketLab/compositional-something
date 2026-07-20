# Two-Stage Candidate Rollouts for Faster Adaptive Self-Improvement

Date: 2026-07-16

Status: implemented; one-attempt smoke submitted and main run dependency-gated

Experiment ID: `two-stage-candidate-rollout-speed`

## Purpose

This document is the source of truth for the experiment that replaces expensive,
full-fidelity training and evaluation of every proposed composition config with a
two-stage procedure:

1. cheaply screen every proposed action;
2. select one action provisionally;
3. retrain that action properly from the pre-attempt checkpoint; and
4. advance the self-improvement state only if the full update improves held-out
   average accuracy.

The initial experiment is restricted to the Qwen3.5-2B addition config condition
initialized with 8,192 examples of synthetic proposal SFT. This is the condition
where synthetic proposal training yielded valid actions and substantial downstream
improvement, so it is the cleanest place to study runtime without conflating the
result with proposal-format collapse.

## Research Question

Can low-fidelity candidate rollouts preserve enough information to choose a useful
composition action while making it affordable to perform a stronger task update
after selection?

The hypothesis is that the current candidate loop spends most of its time obtaining
more evaluation precision and task-training fidelity than action ranking requires.
If the cheap rollout reward remains a useful ranking proxy, the loop can spend that
saved compute on one full selected-action update and obtain similar or better final
accuracy in substantially less wall time.

## Reference Condition

### Model and checkpoint

- Base model family: `Qwen/Qwen3.5-2B`
- Task: exact-digit addition
- Proposal condition: config generation
- Initial source sizes: 3-7 digits
- Evaluation sizes: 3-31 digits
- Synthetic proposal SFT: 8,192 examples
- Reusable post-synthetic checkpoint:

```text
artifacts/runs/adaptive_qwen35_2b_full_resubmit_20260701_212609/
  100steps/addition-config-numeric-n8-reward-outcome-grpo-skip-lr-5em6-syn8192/
  round_00/synthetic_proposal_sft/model
```

The speed experiment must start directly from this checkpoint with
`TREAT_SEED_AS_ROUND_ZERO=1` and synthetic proposal SFT disabled. Rerunning the
8,192-example synthetic stage would add unrelated initialization time and make
iteration slower.

### Existing results

| Job | Candidate update | Attempts | Selected | GRPO updates | Initial avg. | Final avg. | Delta | Full job runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `10540023` | 100 steps | 25 | 8 | 7 | 0.6093 | 0.9455 | +0.3362 | 19:08:21 |
| `10540027` | one epoch | 25 | 9 | 9 | 0.6083 | 0.9614 | +0.3531 | 18:01:07 |

For job `10540023`, the synthetic stage finished around 46 minutes after job start.
The adaptive portion therefore took approximately 18h22m. Runtime comparisons for
the new prepared-start experiment must use that adaptive-only duration, not the
full 19h08m job duration.

### Runtime profile

The 100-step syn8192 addition run recorded:

| Measurement | Value |
|---|---:|
| Candidate task SFT, 100 steps | about 44-50 seconds per candidate at attempt 1 |
| Candidate evaluation | 2,900 examples, about 372 seconds at attempt 1 |
| Candidate evaluation across retained logs | 376 seconds mean |
| Total attempt | 44.1 minutes mean, 43.7 minutes median |
| Round-model phase | 31.0 minutes mean |
| Candidate-dispatch phase | 11.5 minutes mean |

Evaluation is roughly eight times as expensive as the initial candidate training.
Later round-model phases also generate component predictions for as many as eight
5,000-example candidate datasets. Reducing only `MAX_STEPS` would therefore leave
most of the wall time untouched.

## Two-Stage Algorithm

Let `M_t` be the current checkpoint at the beginning of attempt `t`. Let
`D_full` be the fixed full evaluation set and `D_screen` its fixed stratified
subset.

### 1. Reuse the current state

Keep the full and screening metrics associated with `M_t` in the adaptive state.
Do not reevaluate an unchanged checkpoint at the start of every attempt. The prompt
continues to receive full per-size accuracy, not screening accuracy.

### 2. Generate actions

Sample eight unique config actions using the established proposal policy. Each
action contains `left`, `right`, and `guard`; its target is implied by composition.

### 3. Build cheap rollouts

For each valid unique action:

1. deterministically construct the full 5,000-example composed dataset;
2. use only a deterministic 512-example prefix for the screening rollout;
3. generate those 512 pseudolabels with `M_t`;
4. initialize a temporary candidate from `M_t`;
5. train for 25 optimizer steps at task learning rate `5e-6`; and
6. evaluate on `D_screen`, which contains 20 examples for every size from 3 to
   31, or 580 examples total.

The screening reward for action `i` is

\[
r_i^{\mathrm{screen}}
= A_{\mathrm{screen}}(M_{t,i}^{25})
- A_{\mathrm{screen}}(M_t).
\]

All candidates use the same checkpoint, training budget, evaluation examples, and
baseline. The highest-reward action is provisionally selected only when
`r_i^screen > 0`. Zero is not an improvement.

All temporary screening checkpoints are deleted after their metrics are collected.

### 4. Train the provisional winner properly

For the provisional winning action:

1. finish pseudolabel generation for all 5,000 composed examples using the same
   pre-attempt model `M_t`;
2. initialize a fresh model from `M_t`, not from the 25-step screening model;
3. train for one full epoch at learning rate `5e-6` and batch size 16; and
4. evaluate on `D_full`, containing 100 examples per size, or 2,900 examples.

The confirmation reward is

\[
r^{\mathrm{full}}
= A_{\mathrm{full}}(M_{t,*}^{1\text{ epoch}})
- A_{\mathrm{full}}(M_t).
\]

The update is accepted only when `r_full > 0`. If the confirmation reward is zero
or negative:

- delete the fully trained candidate checkpoint;
- keep `M_t` as the current task checkpoint;
- leave the source pool and selected-round count unchanged;
- record a no-selection attempt and a screen/full disagreement; and
- do not train the runner-up.

### 5. Update the proposer and state

If the full update is accepted:

- adopt the one-epoch candidate as the task checkpoint;
- apply the existing target-accuracy source-admission threshold of 0.80 using the
  confirmed full evaluation;
- train proposal GRPO with all comparable screening rewards from the attempt;
- add the winning action's confirmed result as a separate high-fidelity
  state-action-outcome trace; and
- evaluate the post-GRPO checkpoint once on `D_full`, deriving both full and screen
  state metrics for the next attempt.

The selected full reward must not replace its screening reward inside the GRPO
group. Doing so would compare one one-epoch/full-evaluation outcome against seven
25-step/screen-evaluation outcomes. The full result supervises the world model and
checkpoint transition; the screening group supervises relative proposal policy.

## Data and Label Invariants

- Candidate task SFT uses composed pseudolabels only.
- Pseudolabels are generated by the pre-attempt current model `M_t`.
- True target labels and source examples are never mixed into candidate task SFT.
- The selected full dataset contains the 512 screening examples as a deterministic
  subset of the 5,000 examples.
- `D_screen` is a deterministic, stratified subset of `D_full` and is shared by all
  candidates and baselines in the run.
- The prompt's accuracy table always comes from `D_full`.
- Source admission occurs only after full confirmation.

## Frozen Pilot Configuration

| Setting | Value |
|---|---|
| Task | `addition` |
| Model | cached Qwen3.5-2B post-syn8192 checkpoint |
| Candidate training mode | `two_stage` |
| Maximum attempts | 25 |
| No-selection patience | 10 |
| Proposals per attempt | 8 unique actions |
| Proposal temperature / top-p | `0.9 / 0.95` |
| Action history | enabled, at most 5 items |
| Novelty bonus | `0.05` |
| Rollout pseudolabels | 512 per action |
| Rollout task training | 25 steps, batch 16, LR `5e-6` |
| Rollout evaluation | 20 examples per size, 580 total |
| Selected pseudolabels | 5,000 |
| Selected task training | one epoch, batch 16, LR `5e-6` |
| Full evaluation | 100 examples per size, 2,900 total |
| Proposal GRPO LR / KL | `5e-6 / 0.01` |
| Proposal microbatch / accumulation | `4 / 2` |
| GRPO zero-variance policy | `skip` |
| GRPO reward mode | `outcome` |
| Source-admission threshold | `0.80` |
| Candidate local parallelism / pack size | `2 / 2` |
| CPU memory | 48 GB |
| Final checkpoint | retained |

### Planned interface

The launcher and adaptive CLI should expose the two stages explicitly:

```text
CANDIDATE_TRAINING_MODE=two_stage
MAX_STEPS=25
ROLLOUT_TRAIN_PER_SIZE=512
ROLLOUT_EVAL_PER_SIZE=20
CANDIDATE_TRAIN_PER_SIZE=5000
EVAL_PER_SIZE=100
SELECTED_MAX_STEPS=0
NUM_EPOCHS=1
```

`SELECTED_MAX_STEPS=0` means epoch-based selected training. The existing
`single_stage` mode remains available as a scientific baseline; it must not share
the two-stage confirmation semantics accidentally.

## Implementation Requirements

### Adaptive state and evaluation

- Persist full and screening accuracy, per-size accuracy, and checkpoint identity.
- Extend evaluation to emit per-example correctness records so one full generation
  pass can aggregate both `D_full` and `D_screen`.
- Skip start-of-attempt evaluation whenever the cached state matches the current
  checkpoint.
- Invalidate cached metrics whenever task SFT or proposal GRPO changes the
  checkpoint.

### Candidate lifecycle

- Keep screening metrics distinct from confirmed metrics in types and artifacts.
- Rank only screening metrics during provisional selection.
- Recreate the selected model from the pre-attempt checkpoint for full training.
- Use full confirmed metrics for checkpoint advancement and source admission.
- Delete screening models immediately and delete rejected full models.

### Learning traces

- Tag outcome observations with fidelity: `screen` or `confirmed`.
- Build the GRPO group only from screen rewards.
- Record the confirmed observation even when it rejects the task update. Accepted
  confirmations additionally enter the same attempt's observation CE targets.
- Preserve the current behavior of skipping the proposal update when an attempt has
  no accepted candidate.

### Timing and artifacts

Record at least these phases in `attempt_timing.json`:

- proposal generation;
- rollout composed-data construction;
- rollout pseudolabel generation;
- rollout task training;
- rollout evaluation;
- selected full pseudolabel generation;
- selected full task training;
- selected full evaluation;
- proposal update; and
- post-GRPO evaluation.

Per attempt, retain compact artifacts for:

- the screening evaluation set identity;
- all screening metrics and rewards;
- provisional selection;
- full confirmation or rejection;
- screen/full reward disagreement;
- checkpoint transitions and deletions; and
- screen and confirmed outcome traces.

Do not retain temporary candidate model weights unless an explicit debugging flag
requests them.

## Test Plan

### Unit and integration tests

- The screening set contains exactly 20 examples for each size and is nested in the
  full set.
- Every screening candidate receives at most 512 pseudolabels and exactly 25 task
  steps when enough examples exist.
- Full selected training starts from the pre-attempt checkpoint and receives all
  5,000 pseudolabels for one epoch.
- A positive screen and positive full reward advances the checkpoint.
- A positive screen and non-positive full reward leaves checkpoint, source pool,
  selected count, and source sizes unchanged.
- GRPO receives only screening rewards while confirmed outcomes enter the replay
  buffer with `confirmed` fidelity.
- Cached current-state evaluation prevents repeated evaluation of an unchanged
  checkpoint.
- Temporary and rejected model directories are removed under the default cleanup
  policy.
- `single_stage` retains its existing behavior.

### Cluster smoke test

Run one attempt from the cached syn8192 checkpoint with a two-hour walltime. Verify:

- all eight screening work items are attempted;
- candidate counts are 512 / 25 steps / 580 eval examples;
- a provisional winner, if any, is freshly retrained on 5,000 examples;
- the full confirmation transition is correct;
- timing fields are populated;
- peak GPU and CPU memory are within the allocation; and
- the run leaves only the intended retained checkpoint.

### Main pilot

After the smoke test passes, submit one 25-attempt addition job with a 12-hour
walltime. Keep the final checkpoint so the run can be continued.

## Success Criteria

The pilot is successful when:

- it completes without OOM or invalid checkpoint transitions;
- adaptive-loop wall time is below 9 hours, at least twice as fast as the reference
  adaptive portion;
- median non-selected attempt time is below 15 minutes;
- final average accuracy is at least 0.94; and
- the result reports how often a positive screening winner fails full confirmation.

The primary scientific comparison is not selected-count equality. A faster method
may reach the same accuracy with fewer confirmed actions. Report final accuracy,
area under the accuracy-versus-wall-time curve, confirmed gain per GPU-hour, and
screen/full agreement together.

## Analysis Outputs

The result analysis should include:

- accuracy heatmap over attempts and digit sizes;
- target, left, and right action markers;
- screening and confirmed selections marked separately;
- screen reward versus full reward scatter plot;
- screen/full sign agreement and rank diagnostics;
- valid proposal ratio by attempt;
- confirmed improvement and cumulative improvement by attempt;
- wall-time decomposition by phase; and
- comparison against jobs `10540023` and `10540027`.

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-07-16 | Test one screening setting: 25 steps and 20 eval examples per size | Establish whether the basic two-stage idea works before running a grid. |
| 2026-07-16 | Retrain the winner fresh for one epoch | Match the strongest prior task-update condition and avoid screening-schedule bias. |
| 2026-07-16 | Reject a non-positive full update | The cheap rollout selects an action provisionally; full held-out improvement controls state advancement. |
| 2026-07-16 | Do not try the runner-up | Keep the first implementation and runtime interpretation simple. |
| 2026-07-16 | Use screen rewards for GRPO and full reward for a confirmed trace | Avoid mixing different training/evaluation fidelities inside one GRPO group. |
| 2026-07-16 | Run 25 attempts after a one-attempt smoke test | Measure full online dynamics rather than only startup behavior. |

## Implementation Log

### 2026-07-16: Experiment specification

- Created this standalone experiment document.
- Recorded the existing runtime profile and reference results.
- Froze the first pilot's algorithm, hyperparameters, rejection policy, learning
  signals, launch sequence, and success criteria.
- Code changes and cluster submission were pending at this point.

### 2026-07-16: Two-stage implementation

- Added `single_stage` / `two_stage` candidate-training modes and explicit rollout
  train size, rollout eval size, and selected-update step-cap arguments.
- Added a deterministic per-size screening subset and limited screening
  pseudolabel generation to 512 examples per candidate.
- Made two-stage screening task SFT pseudo-only, with no source-example,
  proposal-trace, or outcome-trace replay mixed into candidate task training.
- Added strict provisional selection (`screen_reward > 0`) and deletion of all
  temporary screening model directories under the default cleanup policy.
- Added fresh selected-action training from the pre-attempt checkpoint using all
  5,000 pseudolabels and one epoch, followed by full 2,900-example evaluation.
- Added strict full confirmation (`full_reward > 0`), no runner-up fallback, and
  deletion of rejected full checkpoints.
- Kept all screen metrics as the GRPO policy group. An accepted confirmed result
  is passed separately to observation CE and cannot affect policy advantages.
- Added `selected_confirmation.json`, `confirmed_outcome_trace.jsonl`, rollout
  evaluation identity, and confirmation timing artifacts.
- Added the frozen launcher config
  `launchers/self/config/two_stage_candidate_rollout_addition.env`.

### 2026-07-16: Verification and submission

- Python compilation and `git diff --check` passed.
- Focused and broad adaptive tests passed: 112 tests in the broad run, followed by
  69 tests after the confirmed-observation dispatcher wiring was added.
- Launcher dry run resolved to exactly one addition job with the cached post-syn8192
  checkpoint, 48 GB CPU RAM, one H200, and the frozen two-stage settings.
- Submitted one-attempt smoke job `11263650` with a two-hour walltime. It was
  priority-pending when this entry was written.
- Submitted 25-attempt job `11263752` with a 12-hour walltime and Slurm dependency
  `afterok:11263650`. It cannot consume a GPU unless the smoke exits successfully.

Future entries must record the code commit, tests, deviations from the frozen
configuration, and any compatibility decisions.

## Run Registry

| Date | Commit | Job ID | Run root | Rollout steps | Eval/size | Full train | Status |
|---|---|---:|---|---:|---:|---|---|
| 2026-07-16 | working tree | 11263650 | `artifacts/runs/two_stage_rollout_smoke_20260716_083333` | 25 | 20 | one epoch | completed in 00:13:26 |
| 2026-07-16 | working tree | 11263752 | `artifacts/runs/two_stage_rollout_25a_20260716_083751` | 25 | 20 | one epoch | completed in 03:28:41 |

## Results

Both jobs completed with exit code zero and no OOM or traceback.

| Metric | Smoke `11263650` | Main `11263752` |
|---|---:|---:|
| Wall time | 00:13:26 | 03:28:41 |
| Attempts | 1 | 25 |
| Provisional full confirmations | 1 | 13 |
| Accepted confirmations | 1 | 6 |
| Rejected confirmations | 0 | 7 |
| GRPO updates | 0 | 6 |
| Initial average accuracy | 0.6093 | 0.6093 |
| Final average accuracy | 0.6241 | 0.9703 |
| Net accuracy gain | +0.0148 | +0.3610 |
| Peak CPU RSS | 27.4 GiB | 31.0 GiB |
| Peak GPU memory | not reported | 119,108 MiB |

The 25-attempt adaptive loop was approximately 5.3 times faster than the 18h22m
adaptive portion of reference job `10540023`. Its final 0.9703 accuracy exceeded
the prior 100-step result (0.9455) and prior one-epoch result (0.9614), although
this is one stochastic run under a changed selection procedure rather than a
replicated controlled comparison.

The six accepted full updates occurred at attempts 1, 2, 3, 5, 14, and 18. Their
full local gains summed to +0.3641. The six GRPO updates changed task accuracy by
approximately -0.0031 in aggregate, producing the final net +0.3610 gain. Final
per-size accuracy was at least 0.90 over sizes 3-31.

Screening precision fell near saturation. Thirteen positive screen winners reached
full confirmation, but only six remained positive on the 2,900-example evaluation;
seven were rejected. The first three large screen gains all confirmed, while most
later gains in the 0.0017-0.0069 range were evaluation-noisy or short-horizon
mismatches.

Runtime decomposition over 25 attempts:

- mean / median attempt: 8.29 / 8.53 minutes;
- mean round-model phase: 1.07 minutes;
- mean candidate dispatch: 4.98 minutes; and
- mean full confirmation when triggered: 3.70 minutes.

Checkpoint cleanup worked: the main run occupies 4.0 GB and retains exactly one
`model.safetensors`, the final attempt-18 proposal-GRPO checkpoint.

## Failures and Deviations

- The implementation caches full state metrics and avoids repeated full evaluation,
  but recomputes the 580-example screening baseline each attempt. It does not yet
  persist per-example correctness records or derive screen metrics from a full
  generation pass.
- Full 5,000-example composed datasets are constructed before screening and then
  viewed through a deterministic 512-example prefix. This preserves exact selected
  data identity but does not remove the CPU construction or raw-data serialization
  cost for non-selected actions.
- Rejected confirmations are logged and appended to the in-memory outcome trace
  buffer, but do not receive gradient updates because no-selection attempts retain
  the established proposal-update skip behavior.
- Proposal efficiency still degraded. Across 1,186 proposal draws, 128 (10.8%)
  became unique trainable actions, 676 (57.0%) hit failed-action cooldown, 160
  (13.5%) had parse errors, 156 (13.2%) were out of range, and 58 (4.9%) repeated
  a valid action. Later attempts therefore trained fewer than eight candidates,
  including zero candidates at attempts 11 and 25.

## Conclusion and Next Decision

The two-stage method passed its primary speed and final-accuracy criteria. Full
confirmation is necessary: the 580-example, 25-step screen had a 46% positive
predictive value once the model approached saturation.

The next controlled ablation should increase only screening evaluation fidelity,
for example from 20 to 40 or 50 examples per size, while keeping 512 pseudolabels
and 25 rollout steps fixed. This directly tests whether late false positives come
from evaluation noise. Proposal collapse and cooldown exhaustion remain a separate
problem and should not be changed in the same ablation.

### Implementation Log: 2026-07-16 13:29:01 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/two_stage_rollout_smoke_20260716_083333/addition-config-numeric-n8-reward-outcome-grpo-skip-lr-5em6-syn0`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `True`; max items: `5`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 14]`.
- Selected proposal traces retained for replay: `1`.
- Outcome trace target mode: `numeric`; retained outcome traces: `32`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- No-selection proposal update policy: `skip_grpo_and_cooldown_failed_actions`.
- Failed-action cooldown key: canonical `(min(left,right), max(left,right), guard, target)` until next selected candidate.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Synthetic proposal SFT: `False`; examples: `0`.
- Synthetic proposal seed mix: `False`.
- Prepared start run dir: `None`.
- Keep initial model checkpoints: `True`.
- Keep final model checkpoint: `True`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-07-16 17:43:05 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/two_stage_rollout_25a_20260716_083751/addition-config-numeric-n8-reward-outcome-grpo-skip-lr-5em6-syn0`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `True`; max items: `5`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 9, 12, 14, 21, 28]`.
- Selected proposal traces retained for replay: `6`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1199`.
- Proposal GRPO updates: `6`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- No-selection proposal update policy: `skip_grpo_and_cooldown_failed_actions`.
- Failed-action cooldown key: canonical `(min(left,right), max(left,right), guard, target)` until next selected candidate.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Synthetic proposal SFT: `False`; examples: `0`.
- Synthetic proposal seed mix: `False`.
- Prepared start run dir: `None`.
- Keep initial model checkpoints: `True`.
- Keep final model checkpoint: `True`.
- Keep all proposal-GRPO checkpoints: `False`.
