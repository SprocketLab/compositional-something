# Initial Plan: SEAL-Style Self-Improvement for Composition

Date: 2026-06-03

## Purpose

The workshop version used manually specified composition rules and schedules.
For the ICLR main-conference extension, the target contribution is a
self-improvement pipeline where the current model proposes how to compose its
own predictions for the next round.

This plan starts from the public SEAL idea:

- SEAL repository: <https://github.com/Continual-Intelligence/SEAL>
- Inspected SEAL commit: `6d9c9f9ee392c6cc618e771f399d436d190f6ca4`
- Default model: `Qwen/Qwen3-1.7B`

The first implementation should support two independent adaptive arms:

1. **Config generation:** the model emits JSON composition configurations.
2. **Program generation:** the model emits restricted Python composition code.

The config arm is the more reliable path. The program arm is the more ideal
path, but it should be treated as risky and validation-heavy.

## My Understanding of SEAL

SEAL trains a model to emit **self-edits**: generated artifacts that, when
applied by an external pipeline, improve the model on a future task or input.

The few-shot ARC path in the public SEAL repo is approximately:

1. Prompt the model to generate multiple JSON self-edit configs for a task.
2. Parse, validate, and deduplicate those configs.
3. For each config, construct temporary train data.
4. Train a temporary LoRA adapter using that config.
5. Evaluate each adapter on the task.
6. Keep successful config-output traces.
7. Behavior-clone or ReST-EM train the model on successful traces so the next
   model emits better self-edits.

The general-knowledge path has the same shape with different artifacts:

1. Generate candidate synthetic completions.
2. For each completion, train a temporary test-time-training adapter.
3. Evaluate whether the adapter improves downstream QA.
4. Build an SFT dataset from top completions.
5. Fine-tune the model so future completions are more useful.

The key distinction from ordinary self-training is that the model is not only
generating labels. It is generating an **update artifact** whose value is judged
by an offline training/evaluation loop.

Our adaptation:

- SEAL self-edit artifact -> composition config or composition program.
- SEAL temporary adapter evaluation -> temporary LoRA self-improvement
  candidate.
- SEAL outer ReST-EM/SFT update -> proposal-trace SFT on high-reward generated
  configs/programs.
- The model never calls tools. It emits artifacts; the driver executes and
  evaluates them.

## Core Research Setup

### Tasks

Start with:

- addition
- run length

Run length should use an exact compositional state, because the existing
`(max_run, prefix_run, suffix_run)` target does not contain enough information
to compose exactly across a boundary. The new exact target should be:

```text
max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run
```

This lets the composition program or config-backed operator determine whether
the left suffix and right prefix continue the same run.

### Required Baseline

The required baseline is the paper-exact manual composition baseline for each
task.

For debugging, use a one-round truncated pilot that preserves the same task
format and schedule topology. For the paper-facing comparison, rerun the exact
manual baseline schedule/settings.

### Initial Pilot Scale

The first runnable milestone should be:

- one expansion round
- addition and run length
- config and program arms
- small beam:
  - 4 config proposals per task/round
  - 2 program proposals per task/round

## Full Training Pipeline

For each task, condition, and round:

1. Let `M_r` be the current checkpoint.
2. Prompt `M_r` to generate proposal candidates.
3. Validate proposals.
4. For every valid proposal:
   - generate composed candidate examples,
   - obtain component predictions from `M_r`,
   - compose pseudo-labels using the proposed config/program,
   - train a temporary LoRA candidate from `M_r`,
   - evaluate the temporary candidate.
5. Score each proposal by:

```text
reward = frontier_delta + lambda_final * final_accuracy_delta
final_accuracy_delta = candidate_final_accuracy - init_time_final_accuracy
lambda_final = 0.1
```

`frontier_delta` is the temporary candidate's improvement on newly expanded
held-out sizes/digits relative to the starting checkpoint.
`final_accuracy_delta` measures whether the candidate's global/final held-out
accuracy is above or below the init-time model, rather than rewarding inherited
absolute accuracy. Keep `lambda_final` tunable.

6. Select the best positive-reward proposal.
7. Merge the selected proposal's temporary LoRA into `M_r` to create the
   committed task checkpoint.
8. Build a ReST-EM trace SFT dataset from top positive proposals, capped at top
   2 per task/condition/round.
9. SFT the committed checkpoint on proposal traces.
10. Eval-gate the trace-updated checkpoint. If reward drops beyond tolerance,
    keep the pre-trace committed task checkpoint and log the trace failure.

Important interpretation:

- The selected self-training result is the main task-learning update.
- The proposal-trace SFT is the outer SEAL-style update that teaches the same
  model to generate better future configs/programs.
- Reward is measured before trace SFT, using the selected self-training
  candidate.
- Trace SFT is output-only: the completion is the selected JSON or Python code,
  not a rationale or full transcript.
- For the config arm, do not hand-gate source-pool growth by a target-slice
  accuracy threshold in the first adaptive implementation. The intended pressure
  is that iterative reward teaches the model which composed slices become useful
  sources later.

## Frontier Definition

For the first implementation, use the original compositional
self-improvement frontier definition. Do not define the frontier by an
additional weak-regime score.

Let the seed/init regime be:

```text
initial_min_size ... initial_max_size
```

When `frontier_min_size` is not explicitly provided, the frontier is everything
strictly beyond the init regime:

```text
frontier_min = initial_max_size + 1
frontier_max = initial_max_size + expand_num_size * num_expand_rounds
```

Each expansion round gets one fixed-width frontier slice:

```text
round 1 slice = initial_max_size + 1
                ... initial_max_size + expand_num_size
round r slice = initial_max_size + (r - 1) * expand_num_size + 1
                ... initial_max_size + r * expand_num_size
```

If `frontier_min_size` is explicitly provided, then:

```text
frontier_min = frontier_min_size
frontier_max = frontier_min_size + expand_num_size * num_expand_rounds - 1
round r slice = frontier_min_size + (r - 1) * expand_num_size
                ... frontier_min_size + r * expand_num_size - 1
```

Concrete schedules used by the current launchers:

- Addition recipe fullpack:
  - init digits: `3 ... 7`
  - `expand_num_digits = 3`
  - `num_expand_rounds = 8`
  - frontier digits: `8 ... 31`
  - round slices: `8...10`, `11...13`, `14...16`, `17...19`,
    `20...22`, `23...25`, `26...28`, `29...31`
- Run-length recipe fullpack:
  - init bits: `8 ... 16`
  - `expand_num_bits = 4`
  - `num_expand_rounds = 8`
  - frontier bits: `17 ... 48`
  - round slices: `17...20`, `21...24`, `25...28`, `29...32`,
    `33...36`, `37...40`, `41...44`, `45...48`

In the adaptive proposal setup, generated configs do not choose frontier ranges.
The driver fixes the source/init range and the post-init frontier. Config
generation only chooses which two source slices to compose and which guard to
apply. The first config pilot therefore uses addition source `3...7` with
target frontier `8...31`, and run-length source `8...16` with target frontier
`17...48`.

## Model and Candidate Training Defaults

Default model:

```text
Qwen/Qwen3-1.7B
```

Candidate evaluator:

- one temporary LoRA per valid proposal
- rank: 32
- alpha: 64
- dropout: 0
- target modules:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Implementation note:

- PEFT/LoRA support is optional in the current repo, so the new entrypoint
  should preflight `peft` availability and fail with a clear installation
  message if it is missing.

## Oracle and Leakage Policy

Allowed:

- The offline driver may use oracle labels for proposal reward and evaluation.
- Future proposal prompts may include aggregate oracle-derived diagnostics by
  task, size, slice, condition, and prior proposal.

Not allowed:

- Pseudo-label construction must never use oracle labels.
- Proposal prompts must not include per-example oracle answers.
- The LLM must never call tools directly.
- Program proposals must not receive enough full-input access to directly solve
  the task without component predictions.

## Proposal Arm 1: Config Generation

The config arm lets the model choose a simple vetted composition route. It must
not allow arbitrary executable predicates, and it must not choose experiment
knobs such as frontier range, data budget, or sampling schedule.

### Config Search Space

Generated config fields:

- `left`: integer source slice size
- `right`: integer source slice size
- `guard`: task-specific vetted guard enum
- optional `notes`: short string ignored by execution

The driver derives:

```text
target = left + right
```

Validation rules:

- `left` and `right` must lie inside the current source/init range.
- `target` must lie inside the fixed post-init frontier.
- `guard` must be one of the task-specific allowed guard names.
- Driver-owned fields such as `frontier_min`, `frontier_max`,
  `examples_per_size`, `composition_path`, `compose_arity`, and
  `composition_error_percent` are rejected.

Initial guard enums:

- Addition: `none`, `reject_boundary_carry`
- Run length: `none`, `reject_boundary_continue`,
  `require_boundary_continue`

### Config Proposal Prompt

System:

```text
You are generating a composition configuration for a compositional self-improvement pipeline.
You must output only valid JSON matching the requested schema.
Do not include explanations, markdown, or comments.
You cannot call tools. You only choose a composition configuration.
```

User:

```text
Task: {task_name}

Goal:
Choose two source slices and a guard rule for composing model component
predictions into pseudo-labels.

Current round:
- round_index: {round_index}
- current_source_slices: {current_source_slices}
- allowed_target_frontier: {allowed_target_frontier}
- model: Qwen/Qwen3-1.7B

Aggregate diagnostics from prior evaluation:
{aggregate_metrics_json}

Allowed configuration fields:
{
  "left": integer,
  "right": integer,
  "guard": one of {guard_choices},
  "notes": optional string
}

Constraints:
- left and right must be source slice sizes currently available to the model.
- left + right must land inside allowed_target_frontier.
- Choose only a listed guard value.
- Do not choose frontier ranges, data budgets, sampling schedules, or
  composition paths; the driver owns those.
- The driver composes pseudo-labels from model component predictions, not
  oracle labels.

Output only JSON.
```

Example config trace target:

```json
{
  "left": 4,
  "right": 4,
  "guard": "reject_boundary_carry",
  "notes": "Compose two reliable 4-digit source slices into the 8-digit frontier and reject boundary-carry cases."
}
```

If notes make parsing inconvenient, the implementation may keep `notes` in the
schema but ignore it during execution.

## Proposal Arm 2: Restricted Program Generation

The program arm lets the model emit Python code for the composition rule. This
is more ideal scientifically but more dangerous, so validation must be strict.

### Program API

The generated code must define:

```python
def compose(components, metadata):
    ...
```

Inputs:

- `components`: ordered list of dictionaries.
- Each component contains:
  - `size`: integer
  - `input_id`: string
  - `prediction`: string generated by the current model
  - `metadata`: safe component metadata
- `metadata`: safe composed-example metadata.

Output:

```python
{"accept": True, "target": "<target string>"}
```

or:

```python
{"accept": False, "reason": "<short reason>"}
```

The generated function must not receive oracle labels.

### Program Sandbox Requirements

Reject a program if it fails any of:

- parseable Python AST
- exactly one top-level `compose` function
- no imports
- no `eval`, `exec`, `compile`, `open`, `globals`, `locals`, `vars`, `dir`,
  `getattr`, `setattr`, `delattr`, `__import__`
- no filesystem/network/subprocess access
- no global mutation
- deterministic output on repeated calls
- timeout bounded execution
- randomized oracle/property tests
- output format checks

For the first implementation, allow exactly one automatic repair attempt after
the first validation failure. The repair prompt may include the original
proposal, the validation failure category, and a short sanitized error summary,
but it must not include per-example oracle labels. Apply the same strict
validation gate to the repaired program. If repair fails, reject the proposal
and log both the original and repaired failure reasons.

### Program Proposal Prompt

System:

```text
You are generating a restricted Python composition program for a self-improvement pipeline.
Output only Python code. Do not include markdown or explanations.
The code will be statically checked and sandboxed.
Imports, file access, network access, subprocesses, eval, exec, and global mutation are forbidden.
```

User:

```text
Task: {task_name}

Goal:
Write a composition function that combines component predictions into one pseudo-label.

Allowed function signature:
def compose(components, metadata):
    ...

Inputs:
- components: ordered list of dictionaries.
- Each component has:
  - "size": integer
  - "input_id": string
  - "prediction": string produced by the current model
  - "metadata": safe component metadata
- metadata: safe composed-example metadata.
- You do not receive oracle labels.

Output:
- Return {"accept": True, "target": "<target string>"} if composition succeeds.
- Return {"accept": False, "reason": "<short reason>"} if it should be skipped.

Task target format:
{target_format}

Examples of valid component predictions:
{component_prediction_examples}

Constraints:
- Use only component predictions and metadata.
- Do not solve directly from the full input.
- Reject malformed component predictions.
- Be deterministic.
- Keep code short.

Output only Python code.
```

### Program Repair Prompt

System:

```text
You are repairing a restricted Python composition program for a self-improvement pipeline.
Output only the corrected Python code. Do not include markdown or explanations.
The same sandbox rules still apply: no imports, file access, network access, subprocesses, eval, exec, or global mutation.
```

User:

```text
The previous program failed validation.

Task: {task_name}

Target format:
{target_format}

Validation failure category:
{failure_category}

Sanitized validation summary:
{failure_summary}

Previous program:
{previous_program}

Repair requirements:
- Keep the same function signature: def compose(components, metadata):
- Use only component predictions and safe metadata.
- Do not inspect oracle labels.
- Do not solve directly from full inputs.
- Return {"accept": True, "target": "<target string>"} or {"accept": False, "reason": "<short reason>"}.
- Make the smallest correction needed to pass validation.

Output only Python code.
```

### Example Run-Length Program Trace Target

Target format:

```text
max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run
```

Completion:

```python
def compose(components, metadata):
    if not components:
        return {"accept": False, "reason": "no_components"}

    parsed = []
    for c in components:
        parts = str(c["prediction"]).split("|")
        if len(parts) != 5:
            return {"accept": False, "reason": "bad_component_format"}
        max_run = int(parts[0])
        prefix_symbol = parts[1]
        prefix_run = int(parts[2])
        suffix_symbol = parts[3]
        suffix_run = int(parts[4])
        size = int(c["size"])
        if max_run < 0 or prefix_run < 0 or suffix_run < 0:
            return {"accept": False, "reason": "negative_run"}
        if max_run > size or prefix_run > size or suffix_run > size:
            return {"accept": False, "reason": "run_exceeds_size"}
        parsed.append((size, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run))

    size, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = parsed[0]
    for right in parsed[1:]:
        r_size, r_max, r_prefix_symbol, r_prefix_run, r_suffix_symbol, r_suffix_run = right
        boundary = suffix_run + r_prefix_run if suffix_symbol == r_prefix_symbol else 0
        new_size = size + r_size
        new_max = max(max_run, r_max, boundary)

        new_prefix_run = prefix_run
        if prefix_run == size and prefix_symbol == r_prefix_symbol:
            new_prefix_run = size + r_prefix_run

        new_suffix_run = r_suffix_run
        if r_suffix_run == r_size and suffix_symbol == r_suffix_symbol:
            new_suffix_run = r_size + suffix_run

        size = new_size
        max_run = new_max
        prefix_run = new_prefix_run
        suffix_symbol = r_suffix_symbol
        suffix_run = new_suffix_run

    target = f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"
    return {"accept": True, "target": target}
```

Note: this example is intentionally simple. Property tests should catch edge
cases such as all-one-symbol strings, alternating strings, malformed component
predictions, and rejected empty component lists.

## Proposal Trace SFT Dataset

The ReST-EM trace SFT dataset should train the model to emit exactly the final
parseable artifact.

Do include:

- exact proposal prompt
- exact selected JSON or Python completion
- metadata outside the text fields for audit/filtering

Do not include:

- hidden rationale
- failed attempts
- validation errors
- per-example oracle errors
- reward transcripts

### JSONL Row Shape

```json
{
  "round": 1,
  "task": "addition",
  "condition": "config",
  "reward": 0.183,
  "frontier_delta": 0.142,
  "final_accuracy": 0.410,
  "prompt": "<exact proposal prompt>",
  "completion": "<exact selected JSON or Python output>"
}
```

Trace inclusion rule:

- Include all valid positive-reward proposals.
- Cap at top 2 per task/condition/round.
- Keep completion output-only.
- Put aggregate metrics in future prompts, not in completions.

### Example Config Trace Row

```json
{
  "round": 1,
  "task": "addition",
  "condition": "config",
  "reward": 0.183,
  "frontier_delta": 0.142,
  "final_accuracy": 0.410,
  "prompt": "Task: addition\n\nGoal:\nChoose two source slices...",
  "completion": "{\"guard\":\"reject_boundary_carry\",\"left\":4,\"notes\":\"Compose two reliable 4-digit source slices into the 8-digit frontier and reject boundary-carry cases.\",\"right\":4}"
}
```

### Example Program Trace Row

```json
{
  "round": 1,
  "task": "run_length",
  "condition": "program",
  "reward": 0.096,
  "frontier_delta": 0.071,
  "final_accuracy": 0.250,
  "prompt": "Task: run_length\n\nGoal:\nWrite a composition function...",
  "completion": "def compose(components, metadata):\n    parsed = []\n    for c in components:\n        parts = str(c[\"prediction\"]).split(\"|\")\n        if len(parts) != 5:\n            return {\"accept\": False, \"reason\": \"bad_component_format\"}\n    ..."
}
```

## Artifacts to Save

For each proposal:

- raw prompt
- raw model output
- parsed proposal
- validation result
- validation failure reason, if rejected
- repair prompt and repaired output, if one automatic repair was attempted
- repaired validation result and failure reason, if repair failed
- pseudo-label candidate count
- retained pseudo-label count
- missing/rejected pseudo-label count
- temporary LoRA output path
- candidate evaluation metrics
- reward components
- final reward
- selection status
- trace-SFT inclusion status

For each round:

- starting checkpoint path
- selected proposal id
- selected merged checkpoint path
- trace SFT dataset path
- trace-updated checkpoint path, if accepted
- trace eval-gate result
- aggregate diagnostics for next round prompt

## Implementation Changes

Add a controller layer around the existing `SelfImprovementTask` abstractions
rather than replacing the current task code.

Expected modules:

- proposal generation and prompt rendering
- proposal schema validation
- restricted program validation/execution
- temporary LoRA candidate training/evaluation
- proposal reward/selection
- trace SFT dataset construction
- trace SFT update and eval gate

Run-length task changes:

- add exact-state target mode for:

```text
max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run
```

- implement exact composition and parser support for that mode
- add tests for boundary continuation

## Tests

Unit tests:

- Config schema accepts valid configs and rejects invalid enums/ranges.
- Program sandbox rejects imports, file access, subprocess calls, `eval`, `exec`,
  forbidden builtins, nondeterminism, timeout violations, and malformed outputs.
- Program repair is attempted exactly once after an initial validation failure,
  receives only sanitized failure information, and is accepted only if it passes
  the same strict gate.
- Run-length exact-state composition handles:
  - same-symbol boundary continuation
  - different-symbol boundary
  - all-one-symbol components
  - malformed component predictions
- Addition config proposals map to existing composed-data generation behavior.

Integration tests:

- Mocked proposal-generation dry run validates parsing, deduplication, reward
  ranking, no-update fallback, and artifact writes.
- Tiny real task run validates temporary LoRA creation, selection, merge, and
  trace dataset creation.
- Trace SFT eval gate rejects a degraded post-trace checkpoint.

Acceptance criteria for first pilot:

- One-round addition config and program arms complete.
- One-round run-length config and program arms complete.
- Proposal-level logs are complete.
- Manual paper-exact baseline can be compared against adaptive selected arms.
- No pseudo-label path uses oracle labels.
- Proposal prompts contain aggregate diagnostics only, not per-example oracle
  answers.

### Implementation Log: 2026-06-03 16:20:27 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `config`.
- Output directory: `artifacts/runs/adaptive_self_improvement_init/run_length_config`.
- Valid proposals: 2 / 3.
- Trace rows written: 2.
- Selected proposal: `config-fixed-binary-run-state` with reward 0.268000.

### Implementation Log: 2026-06-03 16:20:27 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `program`.
- Output directory: `artifacts/runs/adaptive_self_improvement_init/run_length_program`.
- Valid proposals: 1 / 2.
- Trace rows written: 1.
- Selected proposal: `program-repaired-run-state` with reward 0.200000.

### Implementation Log: 2026-06-03 Code Landing Notes

- Implemented run-length exact-state target mode `run_state` with parser,
  target formatting, exact state merge, and pseudo-label composition from model
  component predictions.
- Added proposal prompt/schema/trace helpers in `self/adaptive_proposals.py`.
- Added restricted program validation in `self/program_sandbox.py`, including
  AST checks, forbidden imports/builtins, subprocess timeout execution,
  determinism checks, output checks, randomized run-length property tests, and
  one automatic repair pass.
- Added the first adaptive controller entrypoint,
  `self/adaptive_self_improvement.py`, for fixture/model-output dry-run
  proposal validation, reward ranking, selection, trace JSONL construction, and
  plan-doc logging.
- Added reproducible run-length config/program pilot fixtures under
  `tests/fixtures/` and wrote pilot artifacts under
  `artifacts/runs/adaptive_self_improvement_init/`.
- Verified with focused tests:
  `python -m pytest tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py tests/test_self_improvement_tasks.py -k 'adaptive or run_state or symbol_run_pair_target_parser or run_length_symbol_pair_guarded_compose_uses_left_tie' -q`
  passed with 11 selected tests.
- Remaining implementation gap: this first landed controller is still a
  proposal-level dry-run pilot. Temporary LoRA candidate training/evaluation,
  adapter merge, trace SFT, and trace eval-gating are still not wired into the
  real training loop.

### Implementation Log: 2026-06-03 AILab Slurm Pilot

- Added `launchers/self/run_adaptive_self_improvement_ailab.sbatch`.
- Submitted AILab pilot job `9155594` on partition `ailab`, requesting one GPU,
  2 CPUs, 32G memory, and 1 hour.
- This submitted job is a runnable proposal-pilot/preflight job: compile check,
  focused tests, run-length config dry-run, and run-length program dry-run with
  one repair. It is not the full temporary-LoRA adaptive training experiment.
- Slurm logs:
  `artifacts/logs/adaptive-selfimp-pilot-9155594.out` and
  `artifacts/logs/adaptive-selfimp-pilot-9155594.err`.

### Implementation Log: 2026-06-03 AILab Pilot Result

- Job `9155594` completed successfully on `della-i23g2` with one NVIDIA H200.
- Slurm state: `COMPLETED`, exit code `0:0`, elapsed time `00:00:33`.
- Focused tests passed inside the job: 11 selected tests, 35 deselected.
- Config pilot selected `config-fixed-binary-run-state` with reward `0.268`
  and wrote 2 trace rows.
- Program pilot selected repaired `program-repaired-run-state` with reward
  `0.200` and wrote 1 trace row.
- Stderr log was empty.

### Implementation Log: 2026-06-03 Main AILab Submissions

- Added `launchers/self/submit_main_experiments_ailab.sh`.
- Updated `launchers/self/run_figure2_recipe_aggressive.sh` to forward
  task-specific `target_mode`, so run-length exact-state jobs can train and
  evaluate with `run_state` consistently.
- Submitted runnable AILab baseline packs under
  `artifacts/runs/main_experiments_ailab_20260603_132416/`.
- Run-length exact-state job: `9157352`, job name `main-rl-runstate`,
  partition `ailab`, one H200 GPU, 4 CPUs, 96G memory, 72h limit. It trains a
  fresh `run_state` seed, then runs pilot/fullpack baselines with fixed-binary
  composition.
- Addition fixed-binary fullpack job: `9157353`, job name
  `main-add-fullpack`, partition `ailab`, one H200 GPU, 4 CPUs, 96G memory,
  72h limit.
- These are still baseline/fullpack training jobs. They are not the future
  adaptive temporary-LoRA candidate-selection loop.

### Implementation Log: 2026-06-03 16:40:01 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `config`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_self_improvement_ailab_pilot_20260603_123935/run_length_config`.
- Valid proposals: 2 / 3.
- Trace rows written: 2.
- Selected proposal: `config-fixed-binary-run-state` with reward 0.268000.

### Implementation Log: 2026-06-03 16:40:06 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `program`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_self_improvement_ailab_pilot_20260603_123935/run_length_program`.
- Valid proposals: 1 / 2.
- Trace rows written: 1.
- Selected proposal: `program-repaired-run-state` with reward 0.200000.

### Implementation Log: 2026-06-03 Split Adaptive Condition Jobs

- Cancelled the two broader pending AILab baseline submissions, `9157352` and
  `9157353`, before they started.
- Added condition-level AILab launchers:
  `launchers/self/run_adaptive_condition_ailab.sbatch` and
  `launchers/self/submit_adaptive_condition_pilots_ailab.sh`.
- Added addition config/program fixtures so the split covers four cells:
  addition/config, addition/program, run-length/config, and
  run-length/program.
- Local validation before submission:
  `bash -n` passed for both new launchers; all four fixture JSONL files parsed;
  addition config selected `config-addition-fixed-binary-filtered` with reward
  `0.246`; addition program selected `program-addition-concat` with reward
  `0.182`; focused adaptive tests passed with 6 tests.
- Submitted four AILab split adaptive proposal/preflight jobs under
  `artifacts/runs/adaptive_condition_pilots_ailab_20260603_134730/`, each with
  partition `ailab`, `gpu:h200:1`, 2 CPUs, 32G memory, and a 1h limit:
  `9158161` addition/config, `9158162` addition/program, `9158163`
  run-length/config, and `9158164` run-length/program.
- Slurm queue state immediately after submission: all four jobs were `PENDING`
  with reason `(Priority)`.
- Scope caveat: these are split adaptive proposal/preflight condition jobs, not
  the full temporary-LoRA candidate training/evaluation, adapter merge, trace
  SFT, and eval-gated self-improvement loop.

### Implementation Log: 2026-06-03 Split Resource Resubmission

- Updated the split adaptive condition launchers back to the previous CPU and
  memory shape: 4 CPUs and 96G memory per job.
- Cancelled the pending 2-CPU/32G split jobs `9158161`, `9158162`, `9158163`,
  and `9158164`.
- Resubmitted the same four AILab split proposal/preflight jobs with 4 CPUs,
  96G memory, `gpu:h200:1`, and a 1h limit:
  `9158291` addition/config, `9158292` addition/program, `9158293`
  run-length/config, and `9158294` run-length/program.
- These split jobs run one adaptive proposal round per condition
  (`round_index=1`). The earlier cancelled baseline/fullpack jobs were the ones
  configured for 8 expansion rounds.

### Implementation Log: 2026-06-03 18:15:08 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `addition` / `config`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_condition_pilots_ailab_20260603_135234/addition-config`.
- Valid proposals: 2 / 3.
- Trace rows written: 2.
- Selected proposal: `config-addition-fixed-binary-filtered` with reward 0.246000.

### Implementation Log: 2026-06-03 18:17:23 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `config`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_condition_pilots_ailab_20260603_135234/run-length-config`.
- Valid proposals: 2 / 3.
- Trace rows written: 2.
- Selected proposal: `config-fixed-binary-run-state` with reward 0.268000.

### Implementation Log: 2026-06-03 18:17:23 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `addition` / `program`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_condition_pilots_ailab_20260603_135234/addition-program`.
- Valid proposals: 1 / 2.
- Trace rows written: 1.
- Selected proposal: `program-addition-concat` with reward 0.182000.

### Implementation Log: 2026-06-03 18:17:23 UTC

- Added adaptive proposal pilot artifacts for this run.
- Task/condition: `run_length` / `program`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_condition_pilots_ailab_20260603_135234/run-length-program`.
- Valid proposals: 1 / 2.
- Trace rows written: 1.
- Selected proposal: `program-repaired-run-state` with reward 0.200000.

### Implementation Log: 2026-06-03 Split Adaptive Condition Results

- All four resubmitted AILab split proposal/preflight jobs completed with Slurm
  state `COMPLETED` and exit code `0:0`.
- Elapsed times: `9158291` addition/config `00:00:31`, `9158292`
  addition/program `00:00:33`, `9158293` run-length/config `00:00:33`, and
  `9158294` run-length/program `00:00:33`.
- Stderr logs were empty for all four jobs.
- Artifact root:
  `artifacts/runs/adaptive_condition_pilots_ailab_20260603_135234/`.
- Trace rows written across the four cells: 6 total.

### Implementation Log: 2026-06-03 Adaptive Frontier and Proposal Metrics

- Added `self/adaptive_frontier.py`.
- Added weak-regime frontier selection from aggregate diagnostics:
  per-size/per-digit/per-bit accuracy, composed-eval slice metrics, and generic
  `regime_metrics` entries.
- Added proposal-quality metrics written to
  `proposal_quality_metrics.json`: valid rate, duplicate rate, positive/eligible
  rate, repair attempt/success rate, invalid categories, and best/mean reward.
- Updated `self/adaptive_self_improvement.py` to write
  `frontier_selection.json`, include the selected frontier in the proposal
  prompt, and include proposal-quality metrics in `summary.json`.
- Threaded frontier-policy controls through the AILab condition runner:
  `FRONTIER_POLICY`, `FRONTIER_DIAGNOSTICS_PATH`, threshold knobs, and
  `ENFORCE_SELECTED_FRONTIER`.
- Verified with:
  `python -m pytest tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q`
  passed with 10 tests.
- Remaining gap: this still measures proposal generation and frontier selection
  around fixture/model-output proposals. The full temporary-LoRA
  candidate-training/evaluation loop is still the next implementation step for
  measuring actual task-learning improvement.

### Implementation Log: 2026-06-03 Fixed Frontier Clarification

- Decided to use the original compositional self-improvement frontier
  definition for the first adaptive experiments: init is the seed range, and
  frontier is `initial_max_size + 1` through the final expanded size.
- Rewrote the plan's frontier section to remove the weak-regime score as the
  canonical first-version definition.
- Updated the run-length adaptive config fixture and submitter bounds to match
  the original bit-task schedule: init `8...16`, first frontier `17...20`.
- Verified the updated run-length config fixture with
  `--current-frontier-min 8 --current-frontier-max 16 --frontier-min-allowed 17 --frontier-max-allowed 20`.
- Focused adaptive tests still pass with 10 tests.

### Implementation Log: 2026-06-03 Slice-Pair Config Schema

- Refactored config generation to the minimal schema:
  `left`, `right`, and `guard`, with optional `notes`.
- Removed `proposal_type`, task name, frontier range, data budget,
  composition-path, arity, and composition-error fields from generated config
  completions. The validator now rejects those driver-owned fields.
- The driver validates `left/right` against the current source range and checks
  that `left + right` lands inside the fixed target frontier.
- Updated config prompts, traces, controller validation, tests, and config
  fixtures.
- Updated split submitter bounds for config/program preflights:
  addition source `3...7`, target frontier `8...31`; run-length source
  `8...16`, target frontier `17...48`.
- Added explicit `--source-min-allowed` and `--source-max-allowed` controller
  arguments; the old `--current-frontier-*` names remain only as deprecated
  aliases.
- Verification:
  `python -m pytest tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_frontier.py -q`
  passed with 10 tests.

### Implementation Log: 2026-06-03 Reward Baseline Delta

- Changed adaptive proposal reward bookkeeping from the absolute final accuracy
  term to an init-baseline delta:
  `frontier_delta + lambda_final * (final_accuracy - init_final_accuracy)`.
- Added `--init-final-accuracy` and support for aggregate metric keys
  `init_final_accuracy`, `initial_final_accuracy`,
  `init_time_final_accuracy`, or `baseline_final_accuracy`.
- Proposal results now log `init_final_accuracy` and
  `final_accuracy_delta`.
- Clarified that config-source-pool growth should not be manually eval-gated in
  the first adaptive version.

### Implementation Log: 2026-06-03 Candidate-Training Loop

- Added `self/adaptive_candidate_training.py`, a config-condition candidate loop
  that trains temporary candidates rather than only ranking fixture rewards.
- Per round, the driver:
  1. evaluates the current checkpoint on the fixed held-out final eval set,
  2. asks for or loads `left/right/guard` config proposals,
  3. validates `left/right` against the actual current source-size pool,
  4. builds exact-pair composed examples for each valid candidate,
  5. generates pseudo-labels from current-model component predictions,
  6. fine-tunes one temporary candidate per proposal,
  7. evaluates each candidate on the same final eval set,
  8. selects by
     `frontier_delta + lambda_final * (candidate_final_accuracy - init_final_accuracy)`,
  9. adds the selected target slice to the source pool without a manual eval
     gate.
- Implemented exact-pair data construction for:
  - addition: `none` and `reject_boundary_carry`,
  - run length: `none`, `reject_boundary_continue`, and
    `require_boundary_continue`.
- Run-length pseudo-label composition supports `target_mode=run_state` as the
  intended first path, plus the older default `max|prefix|suffix` format.
- The loop writes per-round artifacts:
  `proposal_prompt.json`, `raw_proposals.json`, `proposal_results.json`,
  per-candidate `composed_raw.jsonl`, `component_map.json`,
  `pseudo_examples.jsonl`, `candidate_metrics.json`,
  `selected_candidate.json`, `selected_pseudo_examples.jsonl`,
  `trace_examples.jsonl`, and the top-level
  `adaptive_candidate_training_results.json`.
- If a syntactically valid config is infeasible under its guard/source pool, the
  candidate is rejected and logged to `data_build_failure.json`; the round does
  not abort.
- Added AILab launchers:
  `launchers/self/run_adaptive_candidate_training_ailab.sbatch` and
  `launchers/self/submit_adaptive_candidate_training_ailab.sh`.
  Defaults are Qwen3 1.7B, AILab H200, 24h, 96G, 100 rounds, and 4 candidates,
  with data/step/model knobs overridable through environment variables.
- Verification:
  `python -m pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q`
  passed with 12 tests.
- Additional smoke checks:
  - addition and run-length `--dry-run-data-only` CLI runs built exact-pair
    candidate artifacts successfully,
  - a one-step tiny scratch addition run completed seed training, candidate
    pseudo-labeling, temporary candidate fine-tuning, candidate selection, and
    source-pool update from `3...7` to include `8`.
- Current scope note: program-generation proposals are still sandboxed and
  repairable in the proposal pilot, but program-generated composition functions
  are not yet wired into this candidate-training loop. The full training loop
  implemented here is for the config-generation condition.

### Implementation Log: 2026-06-03 Config Candidate Submission

- Submitted the config-generation candidate-training experiments to AILab.
- Artifact root:
  `artifacts/runs/adaptive_candidate_config_ailab_20260603_202616/`.
- Jobs:
  - addition: Slurm job `9174040`, output directory
    `artifacts/runs/adaptive_candidate_config_ailab_20260603_202616/addition`
  - run length: Slurm job `9174041`, output directory
    `artifacts/runs/adaptive_candidate_config_ailab_20260603_202616/run_length`
- Submission manifest:
  `artifacts/runs/adaptive_candidate_config_ailab_20260603_202616/submission_manifest.json`.
- Queue check immediately after submission showed both jobs pending on
  partition `ailab` with time limit `1-00:00:00`.
- Submission used the launcher defaults: Qwen3 1.7B, config condition,
  proposal model `current`, 100 rounds, 4 candidates per round, one H200 GPU,
  96G memory, and 24h walltime.

### Implementation Log: 2026-06-03 Qwen3 1.7B Cache Fix

- The first config candidate jobs (`9174040`, `9174041`) failed before
  training because compute nodes could not load `Qwen/Qwen3-1.7B` from the Hub.
- A partial home-cache download hit user disk quota, so the incomplete
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B` entry was removed.
- Downloaded the exact non-base model `Qwen/Qwen3-1.7B` to scratch cache:
  `/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3-1.7B/`.
- Verified offline loading with `local_files_only=True` for config and
  tokenizer from that scratch cache.
- Snapshot includes both weight shards:
  `model-00001-of-00002.safetensors` and
  `model-00002-of-00002.safetensors`.
- Updated `launchers/self/run_adaptive_candidate_training_ailab.sbatch` to set
  `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `HF_XET_CACHE`
  under `/scratch/gpfs/BRENDEN/changho/hf_cache` by default.
- Verified launcher syntax with `bash -n` and dry-ran the submitter.

### Implementation Log: 2026-06-03 Offline Resubmission

- Resubmitted after the scratch-cache fix under artifact root
  `artifacts/runs/adaptive_candidate_config_ailab_20260603_204944_resubmit/`.
  Jobs `9174675` (addition) and `9174676` (run length) still failed during
  tokenizer loading because the compute node attempted an online Hub metadata
  request for optional tokenizer files.
- Updated the AILab launcher to force offline Hugging Face/Transformers mode:
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Resubmitted again under artifact root
  `artifacts/runs/adaptive_candidate_config_ailab_20260603_205200_offline_resubmit/`.
  Jobs:
  - addition: `9174747`
  - run length: `9174748`
- Short monitor result:
  - addition `9174747` started on `della-i21g1` with one H200, loaded
    `Qwen/Qwen3-1.7B` from the scratch cache with `HF_HUB_OFFLINE=1`, completed
    the 100-step seed training, wrote the seed checkpoint, and was actively
    using the GPU during the initial post-seed evaluation/proposal phase.
  - run length `9174748` was still pending with Slurm reason `Priority` during
    the monitor window.
- Relevant logs:
  - `artifacts/logs/adaptive-cand-addition-9174747.out`
  - `artifacts/logs/adaptive-cand-addition-9174747.err`
  - `artifacts/logs/adaptive-cand-run-length-9174748.out`
  - `artifacts/logs/adaptive-cand-run-length-9174748.err`

### Implementation Log: 2026-06-03 Program/Policy/Meta Candidate Arms

- Extended `self/adaptive_candidate_training.py` beyond the config arm with
  three executable self-labeling conditions:
  - `program`: the driver chooses the next frontier source pair; the model
    writes only `compose(components, metadata)`.
  - `policy`: the model chooses `left`, `right`, optional `guard`, and a
    sandboxed `compose` program.
  - `meta`: the model chooses `left`, `right`, optional `guard`, declares an
    intermediate representation, and writes a sandboxed `compose` program.
- The generated program never receives oracle labels or full target examples.
  Candidate execution passes only component sizes, opaque component IDs,
  current-model component predictions, and safe metadata. The driver still
  builds target inputs and held-out evaluation sets.
- Added `ExecutableProposal` normalization so config and executable candidates
  share the same training/evaluation/reward path:
  `frontier_delta + lambda_final * (candidate_final_accuracy - init_final_accuracy)`.
- Added bounded sandbox batch execution in `self/program_sandbox.py`; candidate
  programs are statically validated, property-tested, optionally repaired once,
  then executed in a subprocess over candidate examples with output-format
  checks.
- Added proposal prompts for the three executable arms:
  - `program` outputs raw Python code only,
  - `policy` outputs JSON with `left/right/guard/code`,
  - `meta` outputs JSON with `left/right/guard/representation/target_format/code`.
- Added `tests/fixtures/adaptive_addition_policy_fixture.jsonl` and
  `tests/fixtures/adaptive_addition_meta_fixture.jsonl`.
- Updated AILab launchers:
  - `launchers/self/run_adaptive_candidate_training_ailab.sbatch` now accepts
    `CONDITION=config|program|policy|meta`.
  - `launchers/self/submit_adaptive_candidate_training_ailab.sh` now accepts
    `CONDITIONS`, and writes one manifest entry per task-condition cell.
- Verification:
  `python -m pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py -q`
  passed with 16 tests.
- Dry-run CLI smoke checks completed for addition `program`, addition `policy`,
  addition `meta`, and run-length `program`; each produced one valid proposal
  and one candidate data build.

### Implementation Log: 2026-06-03 Program/Policy/Meta Full Submission

- Submitted the six non-config full candidate-training cells to AILab.
- Artifact root:
  `artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/`.
- Submission manifest:
  `artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/submission_manifest.json`.
- Jobs:
  - addition/program: `9176298`
  - addition/policy: `9176299`
  - addition/meta: `9176300`
  - run_length/program: `9176301`
  - run_length/policy: `9176302`
  - run_length/meta: `9176303`
- Immediate Slurm check showed all six jobs accepted and pending on `ailab`
  with reason `Priority`, 24h walltime, one H200 each.

### Implementation Log: 2026-06-04 Candidate Experiment Outcomes

- All eight submitted candidate-training Slurm jobs completed with exit code
  `0:0`.
- Config-generation outcomes:
  - addition/config (`9174747`) completed in `00:50:07`, selected candidates in
    rounds 1 and 2, then stopped in round 3 after no valid candidate. Init final
    accuracy was `0.377931`; selected round-2 final accuracy was `0.503103`.
    Source sizes expanded from `3..7` to include `10` and `17`.
  - run_length/config (`9174748`) completed in `00:15:40`, selected one round-1
    candidate, then stopped in round 2 after no valid proposal. Init final
    accuracy was `0.427273`; selected round-1 final accuracy was `0.871515`.
    Source sizes expanded from `8..16` to include `24`.
- Executable proposal outcomes:
  - addition/program (`9176298`), addition/policy (`9176299`),
    addition/meta (`9176300`), run_length/program (`9176301`),
    run_length/policy (`9176302`), and run_length/meta (`9176303`) all completed
    successfully at the Slurm level but stopped after round 1 with zero valid
    candidates.
  - Main invalid categories were malformed JSON/schema output, markdown or
    explanatory text around the requested object/code, Python syntax errors,
    sandbox runtime errors, output-format failures, and property-test failures.
- Interpretation: the training/evaluation driver and sandbox completed without
  crashing, but Qwen3-1.7B proposal quality is currently the bottleneck for the
  executable arms. The config arm gives positive evidence that candidate
  selection can improve the frontier, but proposal-format adherence is not yet
  robust enough for 100-round autonomous runs.

### Implementation Log: 2026-06-04 Selected-Round Semantics

- Updated `self/adaptive_candidate_training.py` so `--num-rounds` now means the
  requested number of rounds with a selected candidate, not raw proposal
  attempts.
- Failed attempts still write artifacts under `attempt_0001`, `attempt_0002`,
  etc., including proposal results, candidate metrics, and attempt summaries,
  but they do not update the checkpoint/source pool or consume a selected
  round.
- Added safety caps:
  - `--max-attempt-rounds`, defaulting to `10 * num_rounds`,
  - `--no-selection-patience`, defaulting to `max_attempt_rounds`.
- Updated AILab launcher controls:
  - `MAX_ATTEMPT_ROUNDS`
  - `NO_SELECTION_PATIENCE`
- Verification:
  `python -m pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py -q`
  passed with 17 tests.
- Smoke checked the new artifact layout and summary fields:
  `selected_rounds_completed`, `attempts_completed`, and
  `target_selected_rounds`.

### Implementation Log: 2026-06-04 Full Selected-Round Resubmission

- Resubmitted the full 8-cell candidate-training grid after changing
  `--num-rounds` to count selected-candidate rounds.
- Artifact root:
  `artifacts/runs/adaptive_candidate_full_selected_rounds_20260604_140650/`.
- Submission manifest:
  `artifacts/runs/adaptive_candidate_full_selected_rounds_20260604_140650/submission_manifest.json`.
- Jobs:
  - addition/config: `9214987`
  - addition/program: `9214988`
  - addition/policy: `9214989`
  - addition/meta: `9214990`
  - run_length/config: `9214991`
  - run_length/program: `9214992`
  - run_length/policy: `9214993`
  - run_length/meta: `9214994`
- Immediate Slurm check showed all eight jobs accepted and pending on `ailab`,
  24h walltime, one H200 each. Launcher defaults use 100 selected rounds,
  `MAX_ATTEMPT_ROUNDS=1000`, and `NO_SELECTION_PATIENCE=1000`.

### Implementation Log: 2026-06-07 Full Selected-Round Outcomes

- All eight selected-round resubmission jobs hit the 24h walltime and were
  marked `TIMEOUT` by Slurm. They did not complete the requested 100 selected
  rounds.
- Config arms made selected-candidate progress before timing out:
  - addition/config (`9214987`): 228 recorded attempts, 7 selected rounds.
    Source sizes expanded from `3..7` to include `10, 13, 16, 17, 23, 26, 27`.
    Init final accuracy was `0.377931`; the last selected candidate final
    accuracy was `0.631379`.
  - run_length/config (`9214991`): 538 recorded attempts, 6 selected rounds.
    Source sizes expanded from `8..16` to include `17, 18, 20, 21, 23, 24`.
    Init final accuracy was `0.427273`; the best selected candidate reached
    final accuracy `0.951212`, and the last selected candidate final accuracy
    was `0.902727`.
- Executable arms did not select candidates:
  - addition/program (`9214988`): 205 recorded attempts, 0 selected rounds.
    Three proposals passed static/property validation, but candidate execution
    retained no pseudo-labels because sandbox batch execution timed out.
  - addition/policy (`9214989`): 231 recorded attempts, 0 selected rounds.
  - addition/meta (`9214990`): 230 recorded attempts, 0 selected rounds.
  - run_length/program (`9214992`): 433 recorded attempts, 0 selected rounds.
  - run_length/policy (`9214993`): 474 recorded attempts, 0 selected rounds.
  - run_length/meta (`9214994`): 462 recorded attempts, 0 selected rounds.
- Main executable-arm failure modes remained malformed schema/JSON, syntax
  errors, property-test failures, output-format failures, runtime errors, and
  range/enum errors. This supports the interpretation that the bottleneck is
  proposal-format/program quality under Qwen3-1.7B, not Slurm or model-cache
  startup.

### Implementation Log: 2026-06-04 01:27:16 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 1; candidates per round: 1.
- Output directory: `artifacts/runs/smoke_addition_program_candidate`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 01:27:16 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 1; candidates per round: 1.
- Output directory: `artifacts/runs/smoke_addition_policy_candidate`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 01:27:16 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 1; candidates per round: 1.
- Output directory: `artifacts/runs/smoke_addition_meta_candidate`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 01:27:16 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; rounds requested: 1; candidates per round: 1.
- Output directory: `artifacts/runs/smoke_run_length_program_candidate`.
- Final source sizes tracked by driver: `[2, 3, 4]`.

### Implementation Log: 2026-06-04 01:42:26 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_config_ailab_20260603_205200_offline_resubmit/addition`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 17]`.

### Implementation Log: 2026-06-04 04:45:13 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/addition-program`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 04:46:04 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_config_ailab_20260603_205200_offline_resubmit/run_length`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.

### Implementation Log: 2026-06-04 04:59:36 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/addition-policy`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 05:00:21 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/addition-meta`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-04 05:18:56 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/run_length-program`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.

### Implementation Log: 2026-06-04 05:19:06 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/run_length-policy`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.

### Implementation Log: 2026-06-04 05:26:31 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; rounds requested: 100; candidates per round: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_program_policy_meta_20260603_213326/run_length-meta`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.

### Implementation Log: 2026-06-04 18:04:16 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 1; attempts used: 1; candidates per attempt: 1.
- Output directory: `/tmp/pytest-of-cs1095/pytest-18/test_dry_run_attempts_continue0/run`.
- Final source sizes tracked by driver: `[2, 3]`.

### Implementation Log: 2026-06-04 18:04:43 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 1; attempts used: 2; candidates per attempt: 1.
- Output directory: `/tmp/pytest-of-cs1095/pytest-19/test_dry_run_attempts_continue0/run`.
- Final source sizes tracked by driver: `[2, 3]`.

### Implementation Log: 2026-06-04 18:05:00 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 1; attempts used: 1; candidates per attempt: 1.
- Output directory: `artifacts/runs/smoke_selected_round_semantics`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.

### Implementation Log: 2026-06-07 15:33:26 UTC

- Updated adaptive candidate Slurm defaults after proposal-format collapse diagnosis.
- Default learner/proposer coupling is preserved with `PROPOSAL_MODEL_NAME=current`.
- Training now defaults to epoch-based updates: `NUM_EPOCHS=1`, `MAX_STEPS=0`.
- Lowered default learning rate from `5e-5` to `5e-6` in the Python CLI and ailab launcher.
- Increased default train/eval batch sizes from `4/4` to `32/32` in the Python CLI and ailab launcher.
- Added launcher logging for epochs/max steps, learning rate, batch sizes, and gradient accumulation.

### Implementation Log: 2026-06-07 15:52:05 UTC

- Integrated selected proposal trace rehearsal into adaptive candidate training.
- Each selected candidate now appends its proposal prompt/completion pair to `selected_proposal_trace_buffer.jsonl`.
- Each valid candidate update now also includes that candidate's own proposal prompt/completion; only the selected checkpoint persists, so the selected model has immediate gradient on its selected proposal trace.
- Future candidate updates additionally mix replayed selected proposal traces into the same Trainer dataset as task/pseudo-label examples.
- Added bounded replay controls: `--proposal-trace-replay-ratio` and `--proposal-trace-replay-max-examples`.
- Ailab launcher defaults now use `PROPOSAL_TRACE_REPLAY_RATIO=0.02` and `PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES=256`.
- Candidate directories now write `candidate_proposal_trace_example.jsonl`, write `proposal_trace_replay_examples.jsonl` when replay is used, and always write `train_mix_summary.json`.
- This is a supervised positive-trace approximation to SEAL's outer self-edit learning, not yet policy-gradient training with explicit negative rewards.

### Implementation Log: 2026-06-07 16:23:44 UTC

- Submitted the full adaptive trace-replay Slurm grid on ailab GPUs for `addition` and `run_length` across `config`, `program`, `policy`, and `meta`.
- Initial batch-size-32 submission used output root `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_trace_replay_full_20260607_120016` with jobs `9345176`, `9345177`, `9345178`, `9345180`, `9345181`, `9345182`, `9345183`, and `9345184`.
- Monitoring showed batch size 32 was not stable: `addition-config` job `9345176` completed seed training and selected the first candidate, then failed with CUDA OOM during the second candidate-training update on an H200.
- Updated default train/eval batch size from `32/32` to `16/16` in the Python CLI and ailab launcher, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launcher.
- Canceled the remaining batch-size-32 jobs and resubmitted the full grid under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205`.
- Replacement jobs: `addition-config=9346764`, `addition-program=9346765`, `addition-policy=9346766`, `addition-meta=9346767`, `run_length-config=9346768`, `run_length-program=9346769`, `run_length-policy=9346770`, `run_length-meta=9346771`.
- Short monitoring through roughly 10-11 minutes showed all eight replacement jobs running, no OOM/traceback/error keyword matches, confirmed `Train/eval batch size: 16/16`, `Learning rate: 5e-6`, `MAX_STEPS=0`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in startup logs.
- Addition seed training completed cleanly in about 109-118 seconds; run-length seed training completed cleanly in about 509-515 seconds for the conditions that had reached the seed-training summary by the final poll.

### Implementation Log: 2026-06-08 05:09:02 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 100; attempts used: 1000; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205/run_length-config`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40, 48]`.
- Selected proposal traces retained for replay: `3`.

### Implementation Log: 2026-06-08 14:04:09 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 100; attempts used: 1000; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205/run_length-policy`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.

### Implementation Log: 2026-06-08 15:38:35 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 100; attempts used: 1000; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205/run_length-meta`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.

### Implementation Log: 2026-06-09 01:11:28 UTC

- Checked final status for the batch-size-16 full adaptive trace-replay grid.
- Slurm outcomes: `run_length-config=COMPLETED`, `run_length-policy=COMPLETED`, `run_length-meta=COMPLETED`; `addition-config`, `addition-program`, `addition-policy`, `addition-meta`, and `run_length-program` reached the 24h walltime.
- No CUDA OOM/traceback/runtime-error failures were found in the replacement batch-size-16 logs; timeout was the remaining resource failure mode.
- Accepted selected rounds: `addition-config=4`, `run_length-config=3`, all other task/condition pairs `0`.
- Accepted config trajectory for addition: targets `10`, `13`, `16`, `19`; final accuracy moved from initial `0.3897` to `0.6600` at the fourth selected round.
- Accepted config trajectory for run_length: targets `24`, `40`, `48`; final accuracy moved from initial `0.4082` to `0.9788` at the third selected round.
- Valid proposal counts were sparse: `addition-config` produced `5` valid proposals and `run_length-config` produced `3`; program/policy/meta produced `0` valid proposals in both tasks.
- Non-config proposal failures were mostly malformed or sandbox/schema-invalid outputs: repeated markdown/code blocks, forbidden imports, wrong JSON/schema fields, empty targets, and property-test failures.

### Implementation Log: 2026-06-09 01:26:21 UTC

- Added executed analysis notebook `notebooks/adaptive_candidate_trace_replay_analysis.ipynb`.
- The notebook loads adaptive candidate run roots, summarizes attempts/proposals/selected rounds, plots length-by-round and regime-by-round accuracy heatmaps, and analyzes valid/invalid proposal categories.
- Added LLM output/data-handling diagnostics for markdown/code-fence leakage, repeated `compose` blocks, forbidden imports, placeholders, prompt/path echoing, empty targets, lowercase JSON booleans inside Python code, and related malformed-output patterns.
- Notebook execution generated PNG figures under `artifacts/figures/adaptive_candidate_trace_replay_analysis/`.

### Implementation Log: 2026-06-10 04:28:19 UTC

- Submitted config-only selected-proposal-trace replay sweep on ailab H200 GPUs.
- Output root: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808`.
- Manifest: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/submission_manifest.json`.
- Matrix: tasks `addition` and `run_length`, condition `config`, replay settings `0.05/2048`, `0.10/4096`, and `0.20/8192`.
- Attempt budget uses `NUM_ROUNDS=25`, `MAX_ATTEMPT_ROUNDS=50`, and `NO_SELECTION_PATIENCE=50`; this caps each job at 50 proposal attempts while allowing up to 25 selected candidates.
- Training defaults kept fixed at `Qwen/Qwen3-1.7B`, `PROPOSAL_MODEL_NAME=current`, `NUM_EPOCHS=1`, `MAX_STEPS=0`, `LEARNING_RATE=5e-6`, train/eval batch size `16/16`, `NUM_CANDIDATES=4`, and `BF16=1`.
- Slurm jobs: `addition-r005-m2048=9493052`, `addition-r010-m4096=9493053`, `addition-r020-m8192=9493054`, `run_length-r005-m2048=9493055`, `run_length-r010-m4096=9493056`, `run_length-r020-m8192=9493057`.
- Initial scheduler check showed all six jobs pending with the requested `06:00:00` walltime.

### Implementation Log: 2026-06-10 11:12:25 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/addition-config-r010-m4096`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 17, 20, 24, 27]`.
- Selected proposal traces retained for replay: `5`.

### Implementation Log: 2026-06-10 11:56:38 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/addition-config-r005-m2048`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 16, 19, 22, 25, 28, 31]`.
- Selected proposal traces retained for replay: `8`.

### Implementation Log: 2026-06-10 12:37:19 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/addition-config-r020-m8192`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 16, 23, 26, 29]`.
- Selected proposal traces retained for replay: `6`.

### Implementation Log: 2026-06-10 13:29:21 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/run_length-config-r005-m2048`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 26, 40]`.
- Selected proposal traces retained for replay: `3`.

### Implementation Log: 2026-06-10 15:37:13 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/run_length-config-r020-m8192`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 42]`.
- Selected proposal traces retained for replay: `2`.

### Implementation Log: 2026-06-10 15:39:42 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 50; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808/run_length-config-r010-m4096`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 42]`.
- Selected proposal traces retained for replay: `2`.

### Implementation Log: 2026-06-11 07:20:00 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/addition-config-none`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.

### Implementation Log: 2026-06-11 07:24:10 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/addition-config-numeric-textual`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 8, 10, 13, 23]`.
- Selected proposal traces retained for replay: `4`.
- Outcome trace target mode: `numeric_textual`; retained outcome traces: `100`.

### Implementation Log: 2026-06-11 08:00:58 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/addition-config-textual`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 23]`.
- Selected proposal traces retained for replay: `5`.
- Outcome trace target mode: `textual`; retained outcome traces: `100`.

### Implementation Log: 2026-06-11 08:44:30 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/run_length-config-none`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 48]`.
- Selected proposal traces retained for replay: `11`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.

### Implementation Log: 2026-06-11 10:12:59 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/addition-config-numeric`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 8, 10, 13, 17, 18, 20, 23, 27, 30]`.
- Selected proposal traces retained for replay: `9`.
- Outcome trace target mode: `numeric`; retained outcome traces: `100`.

### Implementation Log: 2026-06-11 11:38:11 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/run_length-config-numeric-textual`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40, 48]`.
- Selected proposal traces retained for replay: `9`.
- Outcome trace target mode: `numeric_textual`; retained outcome traces: `100`.

### Implementation Log: 2026-06-11 12:12:47 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824/run_length-config-textual`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48]`.
- Selected proposal traces retained for replay: `10`.
- Outcome trace target mode: `textual`; retained outcome traces: `100`.

### Implementation Log: 2026-06-11 19:20:35 UTC

- Added config-only GRPO-style proposal-validity training after each nonterminal attempt.
- Reward shaping: valid `1.0`, range error `0.6`, enum error `0.5`, schema error `0.25`, parse error `0.0`; exact same-batch duplicate completions remain valid and unpenalized.
- Advantages are group-normalized when possible; zero-variance groups use `fixed_baseline` by default or can be skipped.
- Proposal GRPO logs `proposal_grpo/proposal_grpo_traces.jsonl`, `proposal_grpo/proposal_grpo_metrics.json`, and the updated next-attempt checkpoint under `proposal_grpo/model`.
- Changed default outcome-trace target mode to `numeric` and exposed GRPO settings in the AILAB Slurm launchers.

### Implementation Log: 2026-06-11 19:27:55 UTC

- Removed the duplicate-completion penalty from config proposal validity training.
- Exact repeated completions in the same proposal batch are now still `valid: true`, get proposal-validity reward `1.0`, and are only logged with `duplicate: true` for diagnostics.

### Implementation Log: 2026-06-12 04:32:14 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654/addition-config-numeric-grpo-skip`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 12, 22]`.
- Selected proposal traces retained for replay: `3`.
- Outcome trace target mode: `numeric`; retained outcome traces: `100`.
- Proposal GRPO validity updates: `20`; steps/update: `1`; zero-variance mode: `skip`.

### Implementation Log: 2026-06-12 05:13:13 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654/addition-config-numeric-grpo-fixed-baseline`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 20]`.
- Selected proposal traces retained for replay: `3`.
- Outcome trace target mode: `numeric`; retained outcome traces: `100`.
- Proposal GRPO validity updates: `24`; steps/update: `1`; zero-variance mode: `fixed_baseline`.

### Implementation Log: 2026-06-12 05:26:43 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654/run_length-config-numeric-grpo-fixed-baseline`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40]`.
- Selected proposal traces retained for replay: `11`.
- Outcome trace target mode: `numeric`; retained outcome traces: `100`.
- Proposal GRPO validity updates: `24`; steps/update: `1`; zero-variance mode: `fixed_baseline`.

### Implementation Log: 2026-06-12 15:56:44 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; selected rounds requested: 25; attempts used: 25; candidates per attempt: 4.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654/run_length-config-numeric-grpo-skip`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 33, 40, 41]`.
- Selected proposal traces retained for replay: `14`.
- Outcome trace target mode: `numeric`; retained outcome traces: `100`.
- Proposal GRPO validity updates: `19`; steps/update: `1`; zero-variance mode: `skip`.

### Implementation Log: 2026-06-12 16:45:44 UTC

- Added GRPO learning-dynamics analysis notebook: `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_grpo_learning_dynamics.ipynb`.
- Executed notebook: `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_grpo_learning_dynamics.executed.ipynb`.
- Figures written under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/figures/adaptive_grpo_learning_dynamics/`.
- Main diagnosis: the GRPO validity update was too weak and too misaligned with candidate training. Each GRPO step trained on 4 raw completions, while candidate SFT trained on roughly 40k examples/update for addition and 130k examples/update for run length; proposal trace replay was `0.0` in these GRPO jobs.
- Additional diagnosis: positive GRPO reward trains raw sampled completions, but accepted completions were often verbose text with embedded JSON, not clean JSON. Clean raw JSON rate was `0.0` across the analyzed runs.
- Addition-specific funnel: both GRPO arms had 14/25 attempts with at least one valid proposal but only 3/25 selected attempts, so validity alone did not learn a useful acquisition rule.

### Implementation Log: 2026-06-12 17:33:16 UTC

- Fixed config proposal GRPO target bug: valid proposals now train on the validator-normalized `completion` string, while invalid proposals still train on raw sampled output for the negative signal.
- Increased default proposal count from `4` to `8` in the Python CLI and AILAB launcher; the submit script now exports `NUM_CANDIDATES` explicitly and records it in `submission_manifest.json`.
- Memory estimate for Qwen3-1.7B on AILAB H200: prior GRPO traces had max prompt+completion length about `1198` tokens; moving GRPO batch from 4 to 16 proposals raises rough hidden-state activation scale from about `2 GiB` to `8 GiB` under a conservative backward multiplier. Full-model optimizer state is the larger fixed cost and exists regardless of proposal count.
- Recommendation: use `NUM_CANDIDATES=8` as the default full-run setting; try `NUM_CANDIDATES=16` as a short monitored pilot because runtime can increase when more valid proposals trigger more candidate trainings.

### Implementation Log: 2026-06-12 17:48:22 UTC

- Implemented post-task proposal rehearsal: candidate updates now run task/pseudolabel SFT first, then optionally run a bounded proposal-trace SFT phase on the same model checkpoint.
- Default post-task proposal rehearsal settings: enabled, repeat count `64`, max examples `256`.
- The main candidate SFT mix no longer includes proposal traces when post-task proposal rehearsal is enabled; proposal traces are written to `post_task_proposal_rehearsal_examples.jsonl` and trained under each candidate's `proposal_rehearsal/` directory.
- After successful proposal rehearsal, intermediate task-SFT candidate checkpoints under `training/` are removed unless `KEEP_ALL_CANDIDATE_MODELS=1`.
- Expanded the AILAB submit matrix to vary proposal count. Submitted 8 config jobs under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_post_rehearsal_grpo_n8_n16_20260612_134657`.
- Job IDs: `9592085` addition n8 fixed-baseline, `9592086` addition n16 fixed-baseline, `9592087` addition n8 skip, `9592088` addition n16 skip, `9592089` run_length n8 fixed-baseline, `9592090` run_length n16 fixed-baseline, `9592091` run_length n8 skip, `9592092` run_length n16 skip.
- Short monitoring showed all 8 jobs pending on `ailab`; no startup logs were available yet.

### Implementation Log: 2026-06-13 13:34:00 UTC

- Checked the 8 post-task proposal-rehearsal GRPO jobs. All failed with `safetensors_rust.SafetensorError: Error while serializing: I/O error: Disk quota exceeded (os error 122)`.
- No run wrote a final `summary.json`; the jobs failed while saving full `model.safetensors` checkpoints.
- Main disk source was repeated full proposal-GRPO checkpoints, not the Hugging Face model cache. The run root `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_post_rehearsal_grpo_n8_n16_20260612_134657` reached about `1.6T`; `addition-config-numeric-n8-grpo-fixed-baseline` alone had `361` proposal-GRPO model checkpoints and about `1.17TiB` of model weights.
- Added default latest-only retention for proposal-GRPO checkpoints: when a newer proposal-GRPO checkpoint supersedes an older one inside the same run, the driver removes only the older `proposal_grpo/model/` directory and keeps JSON/JSONL logs.
- Added `--keep-all-proposal-grpo-checkpoints` and `KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS=1` as an opt-out when full checkpoint history is needed.

### Implementation Log: 2026-06-13 13:49:47 UTC

- Deleted failed run root `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_post_rehearsal_grpo_n8_n16_20260612_134657` after confirming it had no final summaries and failed from disk quota exhaustion.
- Pruned all existing `artifacts/runs/**/proposal_grpo/model/` weight directories while preserving `proposal_grpo_metrics.json` and `proposal_grpo_traces.jsonl` logs. Verification after cleanup found `0` remaining proposal-GRPO model directories and `96` retained proposal-GRPO metric/trace file pairs.
- Generalized the slim pipeline: by default, when a new in-run model checkpoint supersedes an older source, candidate, seed, or proposal-GRPO checkpoint, the driver deletes the old `model/` directory and keeps the logs. This preserves the active continuation checkpoint rather than the full checkpoint chain.
- Existing opt-outs remain explicit: `--keep-all-candidate-models` keeps candidate model directories, and `--keep-all-proposal-grpo-checkpoints` keeps proposal-GRPO model directories.

### Implementation Log: 2026-06-13 14:04:50 UTC

- Resubmitted the 8 failed config post-task-rehearsal GRPO jobs with the slim checkpoint policy under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_post_rehearsal_grpo_slim_n8_n16_20260613_100113`.
- Job IDs: `9622769` addition n8 fixed-baseline, `9622770` addition n16 fixed-baseline, `9622771` addition n8 skip, `9622772` addition n16 skip, `9622773` run_length n8 fixed-baseline, `9622774` run_length n16 fixed-baseline, `9622775` run_length n8 skip, `9622776` run_length n16 skip.
- Startup monitoring showed four addition jobs running and four run-length jobs pending for priority. Running logs reported `Keep all proposal-GRPO checkpoints: 0`, post-task proposal rehearsal enabled, and no disk/CUDA/traceback errors.
- Early disk use for the new run root was about `13G` after four seed checkpoints, consistent with slim checkpoint retention rather than the previous runaway proposal-GRPO checkpoint accumulation.

### Implementation Log: 2026-06-15 15:18:12 UTC

- Checked the slim post-task-rehearsal GRPO jobs. All 8 jobs reached the 24h walltime (`TIMEOUT`), not disk quota or CUDA OOM. No run wrote final `summary.json`, but all wrote partial `adaptive_candidate_training_results.json`.
- Partial selected rounds before timeout: addition n8 fixed-baseline `13/14` attempts, addition n8 skip `13/50`, addition n16 fixed-baseline `4/597`, addition n16 skip `10/93`, run_length n8 fixed-baseline `10/10`, run_length n8 skip `9/9`, run_length n16 fixed-baseline `6/6`, run_length n16 skip `5/5`.
- The slim run root was about `94G`, much smaller than the previous failed `1.6T` run. Remaining model footprint came mostly from active/interrupted attempt checkpoints after walltime cancellation.
- Added notebook `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_config_post_rehearsal_slim_analysis.ipynb` and executed copy `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_config_post_rehearsal_slim_analysis.executed.ipynb`.

### Implementation Log: 2026-06-15 20:54:00 UTC

- Implemented SLURM-array candidate training for the adaptive loop. The controller still generates proposals, pseudo-labels, and selection decisions, but candidate model training/evaluation can now run as one array task per candidate.
- Added `--candidate-execution-mode {slurm_array,serial}` with `slurm_array` as the default and `serial` as the backward-compatible single-job path.
- Added worker spec serialization under each attempt's `candidate_jobs/` directory: shared source/eval examples, selected proposal/outcome trace buffers, the proposal prompt, and one JSON spec per candidate.
- Added hidden worker mode `--run-candidate-worker --candidate-worker-spec ...`; workers call the existing `train_and_score_candidate` implementation and write the same `candidate_metrics.json` artifacts as serial execution.
- Added `launchers/self/run_adaptive_candidate_worker_ailab.sbatch` and updated the main AILAB launcher to pass `CANDIDATE_EXECUTION_MODE`, `CANDIDATE_ARRAY_MAX_PARALLEL`, poll interval, timeout, worker walltime, and worker script path.
- Default worker array throttle is `4` concurrent candidates to reduce queue pressure and GPU contention; set `CANDIDATE_ARRAY_MAX_PARALLEL=0` for unthrottled arrays or `CANDIDATE_EXECUTION_MODE=serial` to restore the old behavior.
- Added tests for parser defaults, metric JSON round-trip, and worker-spec reconstruction with the expensive scorer monkeypatched out.

### Implementation Log: 2026-06-15 21:45:00 UTC

- Submitted the 8-cell config candidate-array matrix under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_candidate_array_n8_n16_20260615_123034`.
- Controller job IDs: `9727043` addition n8 fixed-baseline, `9727044` addition n16 fixed-baseline, `9727045` addition n8 skip, `9727046` addition n16 skip, `9727047` run_length n8 fixed-baseline, `9727048` run_length n16 fixed-baseline, `9727049` run_length n8 skip, `9727050` run_length n16 skip.
- During the first smoke submission, monitoring exposed that `slurm_array` still fell back to serial candidate training when only one candidate was valid. Patched this so `slurm_array` always dispatches worker jobs, even for a single candidate; `serial` remains the explicit compatibility mode.
- Added regression test `test_slurm_array_mode_dispatches_even_single_candidate`; focused test suite passed with `29 passed`.
- Monitoring of corrected run confirmed nested worker submission: controller `9727043` wrote `attempt_0001/candidate_jobs/slurm_dispatch.json` and submitted worker array `9727602`; controller `9727044` submitted worker array `9727679`.
- Worker arrays were pending with SLURM reason `Priority` at monitoring time, not failing from missing specs or launcher errors.

### Implementation Log: 2026-06-15 17:09:02 UTC

- Added config-setting trial summary notebook: `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_config_trial_summary.ipynb`.
- Executed copy: `/scratch/gpfs/BRENDEN/changho/compositional-something/notebooks/adaptive_config_trial_summary.executed.ipynb`.
- The notebook organizes config trials from initial config generation through trace replay, outcome traces, proposal-validity GRPO, post-task proposal rehearsal, and the in-progress candidate-array run.
- Generated result figures under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/figures/adaptive_config_trial_summary/`, including selected-round summaries, rolling valid-rate curves, invalid-category bars, candidate funnel plots, selected-config distributions, and representative accuracy heatmaps with target stars.
### Implementation Log: 2026-06-15 17:21:28 UTC

- Implemented option 2 scheduler split for adaptive candidate training: the parent adaptive launcher is now CPU-only on `ailab`, and controller-owned GPU phases dispatch to short single-GPU worker jobs.
- Added `--controller-execution-mode {local,slurm}` with local as the Python default and `slurm` as the ailab launcher default.
- Added `launchers/self/run_adaptive_controller_worker_ailab.sbatch` for seed training/eval, current-model proposal+pseudolabel preparation, and proposal-GRPO updates.
- Candidate training remains backward compatible: `--candidate-execution-mode serial` still works, while the default ailab path uses the existing candidate worker array.
- Verified `bash -n` for adaptive launchers, `py_compile` for `self/adaptive_candidate_training.py`, and `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`29 passed`).

### Implementation Log: 2026-06-15 17:40:29 UTC

- Refactored adaptive candidate scheduler code before resubmission.
- Extracted SLURM submission/poll/cancel helpers to `self/slurm_utils.py`.
- Extracted JSON spec/path/key helpers to `self/adaptive_worker_io.py`.
- Added explicit controller phase contracts/constants in `self/adaptive_controller_phases.py`.
- Consolidated local and SLURM controller behavior through shared seed and round-model phase helpers, reducing drift between `local` and `slurm` modes.
- Centralized checkpoint deletion policy in `CheckpointManager`; old cleanup helper names remain as wrappers.
- Re-verified launcher shell syntax, Python compilation, dry-run submission, and `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`29 passed`).

### Implementation Log: 2026-06-15 17:59:40 UTC

- Updated the adaptive parent launcher to use the `cpu` partition because `ailab` rejects non-GPU jobs.
- Ran a one-cell smoke: `addition/config/numeric/n8/fixed_baseline`, `NUM_ROUNDS=1`, `MAX_ATTEMPT_ROUNDS=1`, small data sizes.
- Smoke passed: parent job `9730204` ran CPU-only; seed worker `9730231`, round-model worker `9730271`, and candidate worker array `9730331` ran on `ailab` with one H200 each; one selected round completed.
- Submitted the full 8-cell config matrix under `artifacts/runs/adaptive_config_cpu_controller_full_20260615_135743`.
- Full controller job IDs: `9730400`-`9730407`; all parent jobs are CPU-only and have submitted seed controller workers on `ailab`.

### Implementation Log: 2026-06-15 18:33:23 UTC

- Reduced CPU parent-controller resources from `4 CPU / 96G` to `1 CPU / 16G`; parent time limit remains `24h`.
- Reduced default controller-worker and candidate-worker walltime from `8h` to `4h`.
- Cancelled the first full matrix (`9730400`-`9730407` and pending seed workers) before any worker outputs were produced.
- Resubmitted the 8-cell config matrix under `artifacts/runs/adaptive_config_cpu_controller_slim_20260615_143150`.
- New controller job IDs: `9731483`-`9731490`; Slurm shows parent jobs request `cpu=1,mem=16G`; first seed workers `9731520` and `9731521` show `4:00:00` walltime.

### Implementation Log: 2026-06-15 17:57:03 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; selected rounds requested: 1; attempts used: 1; candidates per attempt: 8.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_cpu_controller_smoke_20260615_135139/addition-config-numeric-n8-grpo-fixed-baseline`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `True`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `8`.
- Proposal GRPO validity updates: `0`; steps/update: `1`; zero-variance mode: `fixed_baseline`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-15 18:41:11 UTC

- Cancelled the first slim CPU-parent run (`9731483`-`9731490`) and its pending/running seed workers before any seed `worker_output.json`, summaries, or candidate metrics were produced.
- Increased the CPU parent-controller walltime from `24h` to `5-00:00:00`; parent resources remain `1 CPU / 16G` on the `cpu` partition. `sbatch --test-only` accepted 5 days but rejected 7 days for this account/partition/QOS.
- Replaced the single blanket GPU worker walltime with phase/task-aware defaults: seed workers use `2h` for addition and `4h` for run_length, round-model workers use `1h`, proposal-GRPO workers use `1h`, candidate arrays use `2h` for addition and `4h` for run_length.
- Added CLI plumbing for `--controller-seed-worker-time-limit`, `--controller-round-worker-time-limit`, and `--controller-grpo-worker-time-limit`; direct Python fallback controller-worker time is now `4h`.
- Re-verified launcher shell syntax, Python compilation, and `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`29 passed`).

### Implementation Log: 2026-06-15 18:46:35 UTC

- Submitted the 8-cell config matrix under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_cpu_controller_phase_time_20260615_144249`.
- CPU parent controller job IDs: `9732174` addition n8 fixed-baseline, `9732175` addition n16 fixed-baseline, `9732176` addition n8 skip, `9732177` addition n16 skip, `9732178` run_length n8 fixed-baseline, `9732179` run_length n16 fixed-baseline, `9732180` run_length n8 skip, `9732181` run_length n16 skip.
- Startup monitoring confirmed all 8 CPU parents are running with `TimeLimit=5-00:00:00`, `NumCPUs=1`, and `MinMemoryNode=16G`.
- All 8 seed controller workers were dispatched on `ailab`; addition seed workers show `2:00:00` walltime and run_length seed workers show `4:00:00` walltime.
- Seed workers were pending for `Priority` at monitoring time; no missing-spec, launcher syntax, or Python import failures were observed.

### Implementation Log: 2026-06-15 19:08:37 UTC

- Cancelled the `adaptive_config_cpu_controller_phase_time_20260615_144249` matrix before any seed GPU worker started, so that run_length workers would not keep the old `4h` seed/candidate setting.
- Updated run_length GPU worker defaults to `3h` for both seed controller workers and candidate arrays. Addition remains `2h`; round-model and proposal-GRPO controller workers remain `1h`.
- Re-verified launcher shell syntax and Python compilation before resubmission.

### Implementation Log: 2026-06-15 19:12:09 UTC

- Resubmitted the 8-cell config matrix under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_cpu_controller_phase_time_20260615_150906`.
- CPU parent controller job IDs: `9733671` addition n8 fixed-baseline, `9733672` addition n16 fixed-baseline, `9733673` addition n8 skip, `9733674` addition n16 skip, `9733675` run_length n8 fixed-baseline, `9733676` run_length n16 fixed-baseline, `9733677` run_length n8 skip, `9733678` run_length n16 skip.
- Startup monitoring confirmed all 8 seed controller workers were dispatched. Addition seed workers show `2:00:00`; run_length seed workers now show `3:00:00`.
- Parent logs confirm run_length candidate arrays will use `3:00:00` (`Candidate array max parallel/poll/time limit: 4/30/03:00:00`).

### Implementation Log: 2026-06-16 02:06:39 UTC

- Implemented packed local candidate training to avoid CPU parents waiting on nested GPU workers.
- Added `candidate_execution_mode=local_parallel` as the default and `--candidate-local-parallelism` with default `4`; `serial` and `slurm_array` remain available.
- Local candidate training reuses the existing worker-spec format but launches candidate workers as local subprocesses on the allocated GPU job, writing logs under each attempt's `candidate_jobs/logs/`.
- Changed the main AILAB adaptive launcher back to a top-level GPU job: `ailab`, `gpu:h200:1`, `8 CPU`, `192G`, `24h`, `CONTROLLER_EXECUTION_MODE=local`, `CANDIDATE_EXECUTION_MODE=local_parallel`, `CANDIDATE_LOCAL_PARALLELISM=4`.
- Cancelled the nested CPU-parent run (`9733671`-`9733678`, pending seed workers, and candidate array `9750263`) before resubmission.
- Submitted the packed 8-cell config matrix under `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_config_packed_gpu_local_parallel_20260615_220549`.
- Top-level GPU job IDs: `9752553` addition n8 fixed-baseline, `9752554` addition n16 fixed-baseline, `9752555` addition n8 skip, `9752556` addition n16 skip, `9752557` run_length n8 fixed-baseline, `9752558` run_length n16 fixed-baseline, `9752559` run_length n8 skip, `9752560` run_length n16 skip.
- SLURM monitoring confirmed the new jobs request `Partition=ailab`, `gres/gpu=1`, `NumCPUs=8`, `MinMemoryNode=192G`, and no nested adaptive jobs were submitted.
- Verified `bash -n`, Python compilation, AILAB `sbatch --test-only`, and `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`31 passed`).

### Implementation Log: 2026-06-17

- Research Computing memory-efficiency email for packed jobs `9752553`-`9752560` was legitimate: the jobs requested `192G` CPU RAM but `seff` showed only about `13`-`25G` used.
- Reduced the default CPU RAM request for the packed AILAB adaptive candidate launcher from `192G` to `48G`.
- Added `SBATCH_MEM="${SBATCH_MEM:-48G}"` to the adaptive candidate submit wrapper and pass `--mem "${SBATCH_MEM}"`, so future sweeps can override memory without editing the sbatch script.
- This change only affects CPU RAM scheduling; it does not address GPU VRAM OOMs from local candidate parallelism.

### Implementation Log: 2026-06-17 17:12:53 UTC

- Changed config-candidate reward from target-slice accuracy to static-frontier improvement. The static frontier is `frontier_min_size..frontier_max_size`; missing per-size accuracies count as `0.0`. Candidate reward is now `frontier_delta + lambda_final * (final_accuracy - init_final_accuracy)`.
- Kept target-slice accuracy and `target_delta` in `candidate_metrics.json` for inspection, but candidate selection now tie-breaks by `frontier_delta` before `target_delta`.
- Added `frontier_accuracy`, `current_frontier_accuracy`, and `target_delta` to candidate metric serialization, selected proposal metadata, and outcome traces.
- Changed proposal-GRPO default reward mode to `outcome`: valid trained candidates get clipped `candidate_reward / proposal_grpo_outcome_scale` in `[-1, 1]`; malformed proposals get negative verifier rewards by validation category; valid proposals that produce no pseudo labels get `0`; system-side candidate failures are skipped rather than punished.
- Preserved the previous validity-only proposal-GRPO objective as `--proposal-grpo-reward-mode validity` for ablations. The AILAB launcher now exposes `PROPOSAL_GRPO_REWARD_MODE` and `PROPOSAL_GRPO_OUTCOME_SCALE`.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/adaptive_candidate_training.py`; `bash -n launchers/self/run_adaptive_candidate_training_ailab.sbatch`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`34 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 18:28:54 UTC

- Implemented the agent-like config proposal schema as `--proposal-output-schema action_prediction`, now the Python driver default for config runs. The legacy flat `{"left","right","guard"}` schema remains available as `--proposal-output-schema plain`.
- The action-prediction completion schema is:
  `{"proposal":{"left":int,"right":int,"guard":str},"prediction":{"target":int,"expected_frontier_delta":float,"expected_final_delta_from_init":float,"rationale":str}}`.
- Validation executes only `proposal`, but rejects the full completion if `prediction` is missing, has non-finite expected deltas, or has a target inconsistent with `left + right`. Valid completions are normalized as the full action+prediction JSON, so proposal SFT rehearsal and proposal-GRPO both train the same agent-style string rather than a stripped action only.
- Candidate work items, worker specs, candidate metrics, selected proposal traces, outcome traces, and proposal-GRPO traces now retain `proposal_prediction` / `parsed_prediction` metadata for analysis.
- Outcome traces now include the candidate's prediction in the state when available and add calibration targets: realized-minus-predicted `frontier_delta_error` and `final_delta_from_init_error`.
- Updated the AILAB launcher to default `PROPOSAL_OUTPUT_SCHEMA=action_prediction` and increased config `PROPOSAL_MAX_NEW_TOKENS` from `160` to `256` to fit the nested JSON.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/adaptive_candidate_training.py self/adaptive_proposals.py`; `bash -n launchers/self/run_adaptive_candidate_training_ailab.sbatch`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`36 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 18:44:48 UTC

- Started cleanup/refactoring of the adaptive candidate code after the action-prediction additions made the main driver too large.
- Extracted proposal-GRPO reward shaping, trace construction, token collation, and lightweight policy update into `self/adaptive_proposal_grpo.py`. The main driver still re-exports the public helpers used by tests/notebooks: `proposal_grpo_reward`, `proposal_grpo_advantages`, `build_proposal_grpo_traces`, and `apply_proposal_grpo_update`.
- Moved action-prediction schema helpers into `self/adaptive_proposals.py`, alongside the config prompt and base config parser: `PROPOSAL_OUTPUT_SCHEMAS`, `proposal_output_schema`, `proposal_payload_for_schema`, `validate_config_prediction`, and `normalized_config_completion`.
- Removed the unused `_optional_float` helper from `self/adaptive_candidate_training.py`.
- The monolithic driver dropped from roughly `5860` lines to `5335` lines. It is still large, but GRPO and config-schema behavior now have clearer ownership boundaries.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/adaptive_candidate_training.py self/adaptive_proposal_grpo.py self/adaptive_proposals.py`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py` (`36 passed`); `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_proposals_and_sandbox.py` (`5 passed`). Only existing multiprocessing fork warnings appeared.

### Implementation Log: 2026-06-17 19:01:52 UTC

- Continued the cleanup/refactoring pass to reduce the size and coupling of `self/adaptive_candidate_training.py`.
- Extracted proposal/outcome trace dataclasses, trace serialization, replay sampling, selected-proposal trace construction, post-task proposal rehearsal construction, and outcome trace builders into `self/adaptive_experience_traces.py`.
- Extracted candidate worker spec generation plus `local_parallel` and `slurm_array` candidate dispatch into `self/adaptive_candidate_workers.py`. The main driver keeps thin compatibility wrappers for `train_candidates_local_parallel`, `train_candidates_slurm_array`, and `_prepare_candidate_worker_specs`, so existing tests/notebooks can still use the old entry points.
- Kept candidate metric aggregation and candidate selection in the main driver because they depend on `CandidateMetrics`, reward fields, and selection policy.
- The main driver is now `4308` lines. The extracted modules are `self/adaptive_experience_traces.py` (`741` lines), `self/adaptive_candidate_workers.py` (`439` lines), `self/adaptive_proposal_grpo.py` (`467` lines), and `self/adaptive_proposals.py` (`489` lines).
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/adaptive_candidate_training.py self/adaptive_candidate_workers.py self/adaptive_experience_traces.py self/adaptive_proposal_grpo.py self/adaptive_proposals.py`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py` (`41 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:13:07 UTC

- Reorganized the active adaptive self-improvement implementation into canonical package directories while preserving old import/CLI paths.
- Moved core implementation modules to `self/core/`: `driver.py`, `candidate_workers.py`, `controller_phases.py`, `experience_traces.py`, `frontier.py`, `program_sandbox.py`, `proposal_grpo.py`, `proposals.py`, `slurm.py`, and `worker_io.py`.
- Moved the dry-run/pilot adaptive controller to `self/experiments/adaptive_self_improvement.py`.
- Added compatibility wrappers at the old top-level paths, including `self/adaptive_candidate_training.py`, `self/adaptive_proposals.py`, `self/program_sandbox.py`, and `self/slurm_utils.py`. The driver wrapper forwards attribute writes to `self.core.driver`, preserving existing monkeypatch-based tests and older notebooks.
- Updated active adaptive launchers to call canonical modules (`python -m self.core.driver` and `python -m self.experiments.adaptive_self_improvement`) and to compile the new core file paths.
- Added `self/README.md` documenting the canonical layout, wrapper policy, and remaining cleanup queue.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/core/driver.py self/core/candidate_workers.py self/core/controller_phases.py self/core/experience_traces.py self/core/frontier.py self/core/program_sandbox.py self/core/proposal_grpo.py self/core/proposals.py self/core/slurm.py self/core/worker_io.py self/experiments/adaptive_self_improvement.py self/adaptive_candidate_training.py self/adaptive_candidate_workers.py self/adaptive_controller_phases.py self/adaptive_experience_traces.py self/adaptive_frontier.py self/adaptive_proposal_grpo.py self/adaptive_proposals.py self/adaptive_worker_io.py self/adaptive_self_improvement.py self/program_sandbox.py self/slurm_utils.py`; `bash -n launchers/self/run_adaptive_candidate_training_ailab.sbatch launchers/self/run_adaptive_candidate_worker_ailab.sbatch launchers/self/run_adaptive_controller_worker_ailab.sbatch launchers/self/run_adaptive_self_improvement_ailab.sbatch launchers/self/run_adaptive_condition_ailab.sbatch`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:21:09 UTC

- Continued the human-accessibility cleanup of `self/core/driver.py`.
- Extracted CLI parser construction, argument validation, and task-specific default normalization into `self/core/args.py`. The driver still re-exports `build_parser`, `normalize_args`, and the choice constants for compatibility.
- Extracted shared data containers and JSON conversion into `self/core/models.py`: `ExecutableProposal`, `ExactPairDataset`, `CandidateWorkItem`, `CandidateMetrics`, `proposal_from_payload`, and `candidate_metrics_from_json`.
- Updated `self/README.md` with the new `args.py` and `models.py` ownership boundaries and clarified that the next useful splits are composition/pseudolabel generation and controller-worker dispatch.
- The active driver is now `3672` lines, down from `4308` after the package reorg and `5860` before the refactor sequence.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/core/driver.py self/core/args.py self/core/models.py self/adaptive_candidate_training.py`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:26:04 UTC

- Extracted exact-pair composition and pseudolabel generation from `self/core/driver.py` into `self/core/composition.py`.
- The new module owns addition/run-length exact-pair dataset construction, composition guards, run-length example merging, config pseudolabel composition, program-sandbox pseudolabel composition, and `target_pattern_for_task`.
- The driver imports these helpers back into its namespace, preserving old access patterns such as `self.adaptive_candidate_training.build_exact_pair_addition_dataset` and `merge_run_length_examples`.
- Updated `self/README.md` to document `self/core/composition.py` and to mark composition/pseudolabel extraction complete.
- The active driver is now `3086` lines, down from `3672` after the args/models split, `4308` after the package reorg, and about `5860` before the refactor sequence.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/core/driver.py self/core/composition.py self/adaptive_candidate_training.py`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:32:54 UTC

- Extracted generic controller-worker dispatch from `self/core/driver.py` into `self/core/controller_workers.py`.
- The new module owns controller worker output/failure paths, phase-specific Slurm walltime selection, `sbatch` submission, polling, spec writing, cached worker-output loading, and worker failure recording.
- Kept phase-specific seed/round/proposal-GRPO worker bodies in `self/core/driver.py` for now because they still call driver-local training, proposal, and candidate-scoring functions.
- The driver imports the controller-worker helpers back under the old private names (`_run_controller_worker_slurm`, `_controller_worker_output_path`, etc.) so existing internal callers and notebooks remain compatible.
- Updated `self/README.md` to document `self/core/controller_workers.py` and to mark generic controller-worker dispatch extraction complete.
- The active driver is now `2966` lines, down from `3086` after composition extraction, `4308` after the package reorg, and about `5860` before the refactor sequence.
- Verification: `~/.conda/envs/torch-env/bin/python -m py_compile self/core/driver.py self/core/controller_workers.py self/adaptive_candidate_training.py`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:39:21 UTC

- Extracted candidate train/eval scoring from `self/core/driver.py` into `self/core/candidate_scoring.py`.
- The new module owns adaptive `TrainingConfig` construction, checkpoint training, model evaluation, static-frontier averaging, candidate train-mix construction, post-task proposal rehearsal, candidate reward computation, and `candidate_metrics.json` writing.
- Kept candidate execution-mode dispatch and failure aggregation in `self/core/driver.py` for now so existing driver-level monkeypatch tests and compatibility wrappers continue to work.
- Updated `self/README.md` to document `self/core/candidate_scoring.py` and to mark candidate scoring extraction complete.
- The active driver is now `2621` lines, down from `2966` after generic controller-worker extraction, `4308` after the package reorg, and about `5860` before the refactor sequence.
- Verification: `python -m py_compile self/core/driver.py self/core/candidate_scoring.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings). A first pytest attempt under the default Python failed because that environment did not have `torch`; `torch-env` has PyTorch `2.10.0+cu130`.

### Implementation Log: 2026-06-17 19:43:53 UTC

- Extracted candidate execution and worker-result aggregation from `self/core/driver.py` into `self/core/candidate_execution.py`.
- The new module owns candidate worker payload conversion, worker-payload deserialization, failure metric construction, serial candidate execution, candidate array/local worker metric gathering, and the thin local/Slurm dispatch glue around `self/core/candidate_workers.py`.
- Kept driver-level wrappers for `_candidate_failure_metrics`, `train_candidates_serial`, `_collect_candidate_array_metrics`, `train_candidates_slurm_array`, and `train_candidates_local_parallel`. This preserves compatibility with tests/notebooks that monkeypatch the driver globals, including patched `train_and_score_candidate` and patched `loop.subprocess.Popen`.
- Left `train_candidate_metrics` in `self/core/driver.py` for now because it is the branch policy entry point and deliberately calls the monkeypatchable driver-level execution wrappers.
- Updated `self/README.md` to document `self/core/candidate_execution.py` and to mark candidate execution/aggregation extraction complete.
- The active driver is now `2541` lines, down from `2621` after candidate-scoring extraction, `4308` after the package reorg, and about `5860` before the refactor sequence.
- Verification: `python -m py_compile self/core/driver.py self/core/candidate_execution.py self/core/candidate_scoring.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`46 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 19:50:06 UTC

- Started splitting `self/self_improvement_tasks.py` into task-specific modules under `self/tasks/`.
- Added `self/tasks/addition.py` and moved addition dataset construction, composed train/eval generation, boundary-carry slicing, `AdditionTask`, addition pseudolabel derivation, metadata validation, and summary aliases there.
- Added `self/tasks/__init__.py` as the task package entry point and re-exported addition symbols from `self/self_improvement_tasks.py`, preserving the old import path.
- Kept a compatibility resolver inside `self/tasks/addition.py` for old monkeypatch patterns such as patching `self.self_improvement_tasks.generate_prediction_map` or `self.self_improvement_tasks.build_composed_pseudo_map` before calling `AdditionTask.derive_round_targets`.
- `self/self_improvement_tasks.py` is now `3253` lines, down from `3835`; the remaining bit-task adapter, run-length, and multiplication are still in the compatibility module and remain the next task-split targets.
- Verification: `python -m py_compile self/self_improvement_tasks.py self/tasks/__init__.py self/tasks/addition.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`86 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:00:05 UTC

- Continued the task split by extracting shared bit-task utilities into `self/tasks/bit_common.py`.
- `bit_common.py` now owns bit-task constants, parser helpers, target-format normalization, component-size selection, direct pseudolabel construction, guarded pseudolabel refill, guard-slice partitioning, reachable-size helpers, and unique bitstring sampling.
- Moved run-length examples, dataset builders, composition helpers, run-state formatting/merging, and `RunLengthTask` into `self/tasks/run_length.py`.
- Re-exported run-length and shared bit symbols from `self/self_improvement_tasks.py`, preserving old imports used by launchers, tests, and adaptive modules.
- Added the same monkeypatch compatibility pattern used for addition: `self/tasks/run_length.py` resolves `generate_prediction_map` through `self.self_improvement_tasks` when available, so old tests/notebooks that patch the facade still affect `RunLengthTask.derive_round_targets`.
- During verification, the first focused test run caught an accidental deletion of the remaining bit-task helper/example block from the compatibility module; restored that block from git history before final verification.
- `self/self_improvement_tasks.py` is now `1720` lines, down from `3253` after the addition split and `3835` before task splitting began. The remaining bit-task adapter and multiplication remain in the compatibility module.
- Verification: `python -m py_compile self/self_improvement_tasks.py self/tasks/__init__.py self/tasks/addition.py self/tasks/bit_common.py self/tasks/run_length.py self/core/driver.py self/core/composition.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py tests/test_run_length_balanced_eval.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`93 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:04:11 UTC

- Moved the remaining auxiliary bit-task examples, dataset builders, composition helpers, guarded plain-output pseudolabel logic, metadata validation, and task adapter into a task-specific module.
- Re-exported the moved symbols from `self/self_improvement_tasks.py` and `self/tasks/__init__.py`, preserving old imports for that temporary compatibility pass.
- Preserved old monkeypatch behavior by resolving generation and composed-dataset helpers through the `self.self_improvement_tasks` facade when the moved task adapter derives round targets. This keeps existing guarded-refill tests and notebook-style patches working.
- `self/self_improvement_tasks.py` is now `879` lines, down from `1720` after the run-length split and `3835` before task splitting began. Multiplication is now the only task family still implemented in the compatibility module.
- Verification: `python -m py_compile self/self_improvement_tasks.py self/tasks/__init__.py self/tasks/addition.py self/tasks/bit_common.py self/tasks/run_length.py self/core/driver.py self/core/composition.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py tests/test_run_length_balanced_eval.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`93 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:09:51 UTC

- Finished the task-family split by moving multiplication examples, blocked component payload construction, seed/long dataset builders, blocked-component pseudolabel derivation, metadata validation, and `MultiplicationTask` into `self/tasks/multiplication.py`.
- Re-exported multiplication symbols from `self/self_improvement_tasks.py` and `self/tasks/__init__.py`, preserving old imports such as `MultiplicationExample`, `build_multiplication_component_payload`, `multiplication_key`, and `MultiplicationTask`.
- Preserved old monkeypatch behavior by resolving `generate_prediction_map` through the `self.self_improvement_tasks` facade when `MultiplicationTask.derive_round_targets` runs. This keeps existing multiplication pseudolabel tests and notebook-style patches working after the move.
- `self/self_improvement_tasks.py` is now `149` lines, down from `879` after the auxiliary bit-task split and `3835` before task splitting began. It is now a compatibility import surface rather than a task implementation module.
- Updated `self/README.md` to document `self/tasks/multiplication.py` and mark task-family splitting complete.
- Verification: `python -m py_compile self/self_improvement_tasks.py self/tasks/__init__.py self/tasks/addition.py self/tasks/bit_common.py self/tasks/multiplication.py self/tasks/run_length.py self/core/driver.py self/core/composition.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py tests/test_run_length_balanced_eval.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py` (`93 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:14:22 UTC

- Started splitting `self/self_improvement_core.py` into focused core modules.
- Added `self/core/data_io.py` for directory creation, example JSONL IO, round-checkpoint cleanup, save-model policy resolution, RNG-state serialization, JSON sanitization, and summary-record IO.
- Added `self/core/model_io.py` for tokenizer construction, added-token embedding initialization, special-token synchronization, model loading from checkpoints/configs, and recipe/fixed-char model instantiation.
- Re-exported the moved functions from `self/self_improvement_core.py`, preserving old imports such as `ensure_dir`, `save_examples`, `load_examples`, `sanitize_json_value`, `load_model_for_tokenizer`, and `instantiate_model_and_tokenizer`.
- Kept `run_self_improvement` in `self/self_improvement_core.py` and left calls routed through facade-level globals, preserving existing monkeypatch behavior in tests and older scripts.
- `self/self_improvement_core.py` is now `1545` lines, down from `1781` before this split. Training construction, evaluation/generation helpers, and the non-adaptive round loop remain there.
- Updated `self/README.md` with the new `data_io.py` and `model_io.py` ownership boundaries.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/data_io.py self/core/model_io.py self/self_improvement_tasks.py self/tasks/__init__.py self/tasks/addition.py self/tasks/bit_common.py self/tasks/multiplication.py self/tasks/run_length.py self/core/driver.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py` (`99 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:18:29 UTC

- Continued splitting `self/self_improvement_core.py` by extracting generation and evaluation helpers into `self/core/evaluation.py`.
- The new module owns prediction parsing, numeric answer extraction, decode-token resolution, generation input encoding, accuracy breakdown evaluation, prediction-map generation, and prediction-debug sample writing.
- Re-exported the moved functions from `self/self_improvement_core.py`, preserving old imports such as `extract_numeric_answer`, `build_generation_encodings`, `evaluate_accuracy_with_breakdown`, `generate_prediction_map`, and `resolve_max_new_tokens`.
- Preserved existing monkeypatch behavior because `run_self_improvement` still resolves these helpers through the `self.self_improvement_core` module globals.
- `self/self_improvement_core.py` is now `1318` lines, down from `1545` after the model/data split and `1781` before core splitting began. Training construction and the non-adaptive round loop remain there.
- Updated `self/README.md` with the new `evaluation.py` ownership boundary.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/evaluation.py self/core/data_io.py self/core/model_io.py`; `conda run -n torch-env python -c "import self.self_improvement_core as core; print(core.generate_prediction_map.__module__); print(core.extract_numeric_answer('answer 123'))"` confirmed facade exports resolve through `self.core.evaluation`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py` (`99 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:22:25 UTC

- Finished the planned helper split of `self/self_improvement_core.py` by extracting training construction into `self/core/training.py`.
- The new module owns `TrainingConfig`, prompt/target tokenized datasets, exact-size batch sampling, the Trainer subclass for explicit batch samplers, causal-LM collation, training-argument creation, and Trainer construction.
- Re-exported the moved symbols from `self/self_improvement_core.py`, preserving old imports such as `TrainingConfig`, `TokenizedPromptTargetDataset`, `CausalLMDataCollator`, `SizeBucketBatchSampler`, `make_training_args`, and `build_trainer`.
- Preserved existing monkeypatch behavior because `run_self_improvement` still resolves `make_training_args` and `build_trainer` through the `self.self_improvement_core` module globals.
- `self/self_improvement_core.py` is now `987` lines, down from `1318` after the evaluation split and `1781` before core splitting began. Summary helpers and the non-adaptive round loop remain there.
- Updated `self/README.md` with the new `training.py` ownership boundary.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/training.py self/core/evaluation.py self/core/data_io.py self/core/model_io.py`; `conda run -n torch-env python -c "import self.self_improvement_core as core; print(core.TrainingConfig.__module__); print(core.make_training_args.__module__); print(core.SizeBucketBatchSampler.__module__)"` confirmed facade exports resolve through `self.core.training`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py` (`99 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:26:56 UTC

- Extracted the remaining non-loop summary helpers from `self/self_improvement_core.py` into `self/core/summaries.py`.
- The new module owns `SliceMetric`, `RoundSummary`, summary-to-metrics payload conversion, accuracy formatting, and console round-summary printing.
- Re-exported the moved symbols from `self/self_improvement_core.py`, preserving old imports such as `RoundSummary`, `SliceMetric`, `summary_to_payload`, `format_accuracy`, and `summarize_round`.
- `self/self_improvement_core.py` is now `888` lines, down from `987` after the training split and `1781` before core splitting began. It now primarily contains the task protocol and non-adaptive round loop plus compatibility re-exports.
- Updated `self/README.md` with the new `summaries.py` ownership boundary.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/summaries.py self/core/training.py self/core/evaluation.py self/core/data_io.py self/core/model_io.py`; `conda run -n torch-env python -c "import self.self_improvement_core as core; print(core.RoundSummary.__module__); print(core.summary_to_payload.__module__); print(core.format_accuracy.__module__)"` confirmed facade exports resolve through `self.core.summaries`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py` (`99 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:31:32 UTC

- Extracted shared task/example protocols and type aliases from `self/self_improvement_core.py` into `self/core/task_protocols.py`.
- The new module owns `JsonDict`, `SplitName`, prediction/size/key getter aliases, `PromptTargetExample`, and `SelfImprovementTask`.
- Re-exported the moved symbols from `self/self_improvement_core.py`, preserving old imports used by task modules and scripts.
- `self/self_improvement_core.py` is now `760` lines, down from `888` after the summary split and `1781` before core splitting began. It is now mostly compatibility imports plus the non-adaptive round loop.
- Updated `self/README.md` with the new `task_protocols.py` ownership boundary.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/task_protocols.py self/core/summaries.py self/core/training.py self/core/evaluation.py self/core/data_io.py self/core/model_io.py`; `conda run -n torch-env python -c "import self.self_improvement_core as core; print(core.SelfImprovementTask.__module__); print(core.JsonDict); print(core.PromptTargetExample.__module__)"` confirmed facade exports resolve through `self.core.task_protocols`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_frontier.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py` (`99 passed`, only existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:36:20 UTC

- Added `self/analysis/artifacts.py` as the first stable artifact-loading layer for notebooks.
- The new module owns JSON/JSONL helpers, adaptive run discovery, attempt loading, proposal/candidate flattening, trace JSONL loading, non-adaptive result loading, and per-size accuracy row construction.
- Added `tests/test_analysis_artifacts.py` with synthetic adaptive and non-adaptive artifacts so notebook-facing loaders are covered without depending on large experiment outputs.
- Updated `self/README.md` with a `Current Analysis` section and changed the cleanup queue to route future notebook work through `self/analysis/artifacts.py`.
- Verification: `python -m py_compile self/analysis/__init__.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. pytest tests/test_analysis_artifacts.py` (`2 passed`).

### Implementation Log: 2026-06-17 20:38:15 UTC

- Extracted adaptive checkpoint-retention policy from `self/core/driver.py` into `self/core/checkpoints.py`.
- The new module owns `CheckpointManager`, `cleanup_unselected_models`, and `cleanup_replaced_model_checkpoint`; `self/core/driver.py` imports those names back so old imports through `self.adaptive_candidate_training` continue to work.
- Removed the now-unneeded `shutil` and `dataclass` imports from the driver.
- Updated `self/README.md` with the new checkpoint module and marked checkpoint cleanup as extracted from the driver.
- Verification: `python -m py_compile self/core/checkpoints.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'cleanup_replaced_model_checkpoint or cleanup_unselected'` (`3 passed`, `33 deselected`).
- Broader sanity check: `PYTHONPATH=. conda run -n torch-env pytest tests/test_analysis_artifacts.py tests/test_adaptive_candidate_training.py` (`38 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:44:31 UTC

- Extracted adaptive proposal runtime code from `self/core/driver.py` into `self/core/proposal_runtime.py`.
- The new module owns task target-format strings, component-prediction examples for program prompts, default program-pair selection, program/policy/meta prompt rendering, fixture/current-model/external-model proposal loading, config proposal validation, executable proposal validation, and program repair dispatch.
- `self/core/driver.py` imports the moved helper names back so old access through `self.adaptive_candidate_training` remains available.
- Driver size is now `1784` lines, down from `2463` before this pass and `2541` before the recent checkpoint/artifact cleanup continuation.
- Updated `self/README.md` with the new proposal-runtime module and marked proposal generation/validation as extracted from the driver.
- Verification: `python -m py_compile self/core/proposal_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'validate_config_rows or validate_proposal_rows or choose_default_program_pair or policy_validation or meta_validation or program'` (`4 passed`, `32 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:47:49 UTC

- Extracted candidate composed-data construction and pseudo-label attachment from `self/core/driver.py` into `self/core/candidate_data.py`.
- The new module owns `examples_by_key`, `build_candidate_work_items`, and `attach_pseudo_labels`, including composed raw-example artifacts, component maps, composed diagnostics, component prediction summaries, pseudo-example JSONL files, and pseudo diagnostics.
- `self/core/driver.py` imports those helper names back so old access through `self.adaptive_candidate_training` remains available.
- Driver size is now `1649` lines, down from `1784` before this pass.
- Updated `self/README.md` with the new candidate-data module and marked candidate data construction as extracted from the driver.
- Verification: `python -m py_compile self/core/candidate_data.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'build_candidate_work_items or candidate_worker_spec_roundtrip_loads_inputs or program_pseudo_labels'` (`2 passed`, `34 deselected`, `1` existing multiprocessing fork warning); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:50:17 UTC

- Extracted adaptive run setup helpers from `self/core/driver.py` into `self/core/run_setup.py`.
- The new module owns `source_sizes_from_examples`, `append_plan_log`, `prepare_datasets`, and `load_trace_jsonl`.
- `self/core/driver.py` imports those helper names back so old access through `self.adaptive_candidate_training` remains available.
- Driver size is now `1613` lines, down from `1649` before this pass.
- Updated `self/README.md` with the new run-setup module and marked run setup/trace loading as extracted from the driver.
- Verification: `python -m py_compile self/core/run_setup.py self/core/candidate_data.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'candidate_worker_spec_roundtrip_loads_inputs or build_candidate_work_items or slurm_array_mode_dispatches_even_single_candidate'` (`2 passed`, `34 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:54:54 UTC

- Extracted in-process controller phase implementations from `self/core/driver.py` into `self/core/controller_phase_runtime.py`.
- The new module owns `run_seed_phase` and `run_round_model_phase`, including seed training/evaluation, proposal prompt generation, proposal loading/validation, candidate composed-data construction, and pseudo-label attachment.
- Controller-worker spec wrappers remain in `self/core/driver.py` for now and call the imported phase functions, preserving old access through `self.adaptive_candidate_training`.
- Driver size is now `1379` lines, down from `1613` before this pass.
- Updated `self/README.md` with the new controller-phase runtime module and marked seed/round-model phases as extracted from the driver.
- Verification: `python -m py_compile self/core/controller_phase_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'candidate_worker_spec_roundtrip_loads_inputs or parser_defaults or program_pseudo_labels or policy_validation or meta_validation'` (`5 passed`, `31 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 20:59:57 UTC

- Extracted the candidate-worker spec entry point from `self/core/driver.py` into `self/core/candidate_worker_runtime.py`.
- The new module owns spec loading, argument reconstruction, trace/pseudo-example loading, `CandidateWorkItem` reconstruction, candidate scoring invocation, and worker-failure artifact writing.
- `self/core/driver.py` now exposes thin wrappers that build a `CandidateWorkerRuntimeDeps` object from driver-level globals. This preserves old monkeypatch behavior, especially tests and notebooks that patch `self.adaptive_candidate_training.train_and_score_candidate`.
- Driver size is now `1317` lines, down from `1379` before this pass.
- Updated `self/README.md` with the new candidate-worker runtime module and marked candidate-worker runtime as extracted from the driver.
- Verification: `python -m py_compile self/core/candidate_worker_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'candidate_worker_spec_roundtrip_loads_inputs or local_parallel_candidate_worker_failure_becomes_metric or local_parallel_candidate_workers_respect_concurrency_cap'` (`3 passed`, `33 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:03:19 UTC

- Extracted candidate training dispatch wrappers from `self/core/driver.py` into `self/core/candidate_dispatch_runtime.py`.
- The new module owns dispatch for serial, local-parallel, and Slurm-array candidate execution, plus candidate-array metric collection and failure metric construction.
- `self/core/driver.py` keeps thin wrappers that inject driver-level functions into the runtime module. This preserves old monkeypatch behavior for tests/notebooks that patch `train_candidates_serial`, `train_candidates_slurm_array`, or `train_and_score_candidate` through `self.adaptive_candidate_training`.
- Driver size is now `1278` lines, down from `1317` before this pass.
- Updated `self/README.md` with the new candidate-dispatch runtime module and marked candidate dispatch as extracted from the driver.
- Verification: `python -m py_compile self/core/candidate_dispatch_runtime.py self/core/candidate_worker_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'slurm_array_mode_dispatches_even_single_candidate or local_parallel_candidate_workers_respect_concurrency_cap or candidate_worker_spec_roundtrip_loads_inputs'` (`3 passed`, `33 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:06:35 UTC

- Extracted controller-worker spec entry points from `self/core/driver.py` into `self/core/controller_worker_runtime.py`.
- The new module owns seed, round-model, and proposal-GRPO controller-worker spec loading, argument reconstruction, phase dispatch, phase-output payload construction, and generic controller-worker output/failure wrapping.
- `self/core/driver.py` keeps thin wrappers that inject driver-level functions into the runtime module. This preserves old monkeypatch behavior for tests/notebooks that patch `run_seed_phase`, `run_round_model_phase`, or `apply_proposal_grpo_update` through `self.adaptive_candidate_training`.
- Driver size is now `1202` lines, down from `1278` before this pass.
- Updated `self/README.md` with the new controller-worker runtime module and marked controller-worker spec runtime as extracted from the driver.
- Verification: `python -m py_compile self/core/controller_worker_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_candidate_training.py -k 'controller or candidate_worker_spec_roundtrip_loads_inputs or parser_defaults or program_pseudo_labels or policy_validation or meta_validation'` (`7 passed`, `31 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:10:27 UTC

- Extracted dry-run adaptive attempt handling from `self/core/driver.py` into `self/core/dry_run_runtime.py`.
- The new module owns fixture proposal selection for an attempt, dry-run proposal validation, candidate work-item construction, `dry_run_summary.json`, and dry-run result-record updates.
- `self/core/driver.py` injects driver-level helpers into `run_dry_attempt`, preserving old patch/import behavior while reducing the inline round-loop branch.
- Driver size is now `1173` lines, down from `1202` before this pass.
- Updated `self/README.md` with the new dry-run runtime module and marked dry-run attempt handling as extracted from the driver.
- Verification: `python -m py_compile self/core/dry_run_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'dry_run_attempts_continue_until_valid_candidate or candidate_work_item_logs_infeasible_guard'` (`2 passed`, `34 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:12:42 UTC

- Extracted proposal-GRPO local-vs-Slurm dispatch from `self/core/driver.py` into `self/core/proposal_grpo_dispatch.py`.
- The new module owns proposal-GRPO worker input artifact writing and dispatches either direct `apply_proposal_grpo_update` execution or a Slurm controller-worker call.
- `self/core/driver.py` keeps a thin wrapper that injects driver-level dependencies, preserving old patch/import behavior.
- Driver size is now `1161` lines, down from `1173` before this pass.
- Updated `self/README.md` with the new proposal-GRPO dispatch module and marked proposal-GRPO dispatch as extracted from the driver.
- Verification: `python -m py_compile self/core/proposal_grpo_dispatch.py self/core/dry_run_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'proposal_grpo or parser_defaults'` (`4 passed`, `32 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:16:21 UTC

- Extracted selected/no-selection adaptive attempt outcome handling from `self/core/driver.py` into `self/core/attempt_outcome_runtime.py`.
- The new module owns candidate metrics/selected-candidate artifact writing, outcome trace writing, no-selection attempt summaries, selected proposal trace buffering, selected pseudo-example artifacts, source-pool updates, checkpoint replacement cleanup, proposal-GRPO follow-up updates, and selected-round summaries.
- `self/core/driver.py` now delegates post-candidate outcome handling through `handle_attempt_outcome` with explicit dependencies, preserving driver-level patch/import behavior.
- Driver size is now `1015` lines, down from `1161` before this pass.
- Updated `self/README.md` with the new attempt-outcome runtime module and marked selected/no-selection attempt outcome handling as extracted from the driver.
- Verification: `python -m py_compile self/core/attempt_outcome_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'selected_proposal_trace_replay_examples_preserve_completion or build_proposal_grpo_traces_uses_outcome_rewards_and_skips_system_failures or select_candidate_tiebreaks_by_frontier_delta_before_target_delta'` (`3 passed`, `33 deselected`); `PYTHONPATH=. pytest tests/test_analysis_artifacts.py` (`2 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:20:40 UTC

- Extracted final adaptive artifact writing from `self/core/driver.py` into `self/core/run_finalization.py`.
- The new module owns final `adaptive_candidate_training_results.json` writing, `summary.json` construction, and plan-log finalization for adaptive candidate-training runs.
- `self/core/driver.py` now calls `finalize_adaptive_run(...)` with explicit writer/logger/sanitizer dependencies, preserving the existing output schema and plan-log formatting.
- Driver size is now `965` lines, down from `1015` before this pass.
- Updated `self/README.md` with the new run-finalization module and marked final result/log finalization as extracted from the driver.
- Verification: `python -m py_compile self/core/run_finalization.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'dry_run_attempts_continue_until_valid_candidate or selected_proposal_trace_replay_examples_preserve_completion or build_proposal_grpo_traces_uses_outcome_rewards_and_skips_system_failures or select_candidate_tiebreaks_by_frontier_delta_before_target_delta'` (`4 passed`, `32 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:22:46 UTC

- Collapsed `self/core/driver.py`'s private JSON/spec/key serialization wrapper functions into compatibility aliases backed by `self/core/worker_io.py`.
- The old driver names (`_load_json`, `_namespace_from_json_args`, `write_key_set`, etc.) still resolve for tests, notebooks, and legacy imports, but the implementation now has a single owner in `worker_io`.
- Driver size is now `944` lines, down from `965` before this pass.
- Updated `self/README.md` to clarify that `worker_io` owns these helpers and the driver only preserves aliases.
- Verification: `python -m py_compile self/core/driver.py self/core/worker_io.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:27:15 UTC

- Extracted attempt-level proposal prompt construction into `self/core/attempt_prompt_runtime.py`.
- The new module owns current-source/frontier prompt payload construction, aggregate metric payload construction, default program-pair selection for program proposals, and config/program prompt rendering.
- `self/core/driver.py` and `self/core/controller_phase_runtime.py` now share this helper, removing duplicated prompt-construction code while preserving the existing controller-only `proposal_output_schema` aggregate metric.
- Driver size is now `911` lines, down from `944` before this pass.
- Updated `self/README.md` with the new prompt-runtime module and marked attempt-level prompt construction as extracted.
- Verification: `python -m py_compile self/core/attempt_prompt_runtime.py self/core/controller_phase_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -k 'choose_default_program_pair or controller or dry_run_attempts_continue_until_valid_candidate or parser_defaults or program_pseudo_labels or policy_validation or meta_validation'` (`7 passed`, `31 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:30:24 UTC

- Extracted per-attempt round-model local-vs-Slurm dispatch into `self/core/round_model_dispatch_runtime.py`.
- The new module owns Slurm controller input artifact writing, round-model controller-worker dispatch/parsing, local `run_round_model_phase` fallback, and `RoundModelDispatchResult` construction.
- `self/core/driver.py` now injects its existing serializer, worker, and phase functions into `run_round_model_dispatch(...)`, preserving compatibility with driver-level imports/patching while removing worker IO plumbing from the loop.
- Driver size is now `875` lines, down from `911` before this pass.
- Updated `self/README.md` with the new dispatch module and marked round-model local/Slurm dispatch as extracted.
- Verification: `python -m py_compile self/core/round_model_dispatch_runtime.py self/core/driver.py self/core/controller_phase_runtime.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -k 'controller or slurm_array_mode_dispatches_even_single_candidate or candidate_worker_spec_roundtrip_loads_inputs or dry_run_attempts_continue_until_valid_candidate or parser_defaults'` (`6 passed`, `32 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:35:00 UTC

- Extracted seed/dry-run initialization and initial adaptive summary construction into `self/core/seed_dispatch_runtime.py`.
- The new module owns dry-run seed state, seed Slurm controller-worker dispatch/parsing, in-process `run_seed_phase` fallback, and the round-0 `summary_records` payload.
- `self/core/driver.py` now calls `run_seed_dispatch(...)` with explicit dependencies for Slurm dispatch, float parsing, and in-process seed training, preserving driver-level patch/import behavior.
- Driver size is now `838` lines, down from `875` before this pass.
- Updated `self/README.md` with the new seed-dispatch module and marked seed initialization/initial summary construction as extracted.
- Verification: `python -m py_compile self/core/seed_dispatch_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -k 'dry_run_attempts_continue_until_valid_candidate or controller or parser_defaults or candidate_worker_spec_roundtrip_loads_inputs'` (`5 passed`, `33 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:38:37 UTC

- Extracted adaptive output/data initialization into `self/core/run_initialization_runtime.py`.
- The new module owns output/data directory setup, checkpoint manager construction, initial train/validation/test/eval artifact writing, metadata writing, source pool initialization, and exclude-key initialization.
- `self/core/driver.py` now calls `initialize_adaptive_run(...)` with explicit dependencies for config construction, dataset preparation, example writing, and JSON writing, preserving the driver as the compatibility surface.
- Driver size is now `824` lines, down from `838` before this pass.
- Updated `self/README.md` with the new run-initialization module and marked output/data initialization as extracted.
- Verification: `python -m py_compile self/core/run_initialization_runtime.py self/core/seed_dispatch_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -k 'dry_run_attempts_continue_until_valid_candidate or controller or parser_defaults or candidate_worker_spec_roundtrip_loads_inputs'` (`5 passed`, `33 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:43:26 UTC

- Extracted selected-round adaptive attempt-loop orchestration into `self/core/attempt_loop_runtime.py`.
- The new module owns the attempt `while` loop, per-attempt prompt persistence, dry-run attempt branch, round-model dispatch branch, candidate training/selection, round trace writing, unselected checkpoint cleanup, and selected/no-selection outcome application.
- `self/core/driver.py` now calls `run_adaptive_attempt_loop(...)` with explicit dependencies for prompt building, dry-run handling, round-model dispatch, candidate metrics, selection, trace writing, and outcome handling, preserving driver-level compatibility and patch/import behavior.
- Driver size is now `707` lines, down from `824` before this pass.
- Updated `self/README.md` with the new attempt-loop runtime module and marked selected-round attempt-loop orchestration as extracted.
- Verification: `python -m py_compile self/core/attempt_loop_runtime.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -k 'dry_run_attempts_continue_until_valid_candidate or controller or parser_defaults or selected_proposal_trace_replay_examples_preserve_completion or select_candidate_tiebreaks_by_frontier_delta_before_target_delta or slurm_array_mode_dispatches_even_single_candidate'` (`7 passed`, `31 deselected`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py` (`36 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 21:46:05 UTC

- Started Slurm launcher consolidation by adding `launchers/self/lib/adaptive_common.sh`.
- The shared helper now owns adaptive launcher repo-root resolution, `artifacts/logs` creation, HF cache/offline environment setup, `PYTORCH_CUDA_ALLOC_CONF`, Python resolution through `TORCH_ENV_PATH`/`PYTHON_BIN`, and worker context logging.
- Updated `launchers/self/run_adaptive_candidate_training_ailab.sbatch`, `launchers/self/run_adaptive_candidate_worker_ailab.sbatch`, and `launchers/self/run_adaptive_controller_worker_ailab.sbatch` to source the shared helper while preserving their existing resource requests, required environment variables, and Python entry points.
- Worker launchers are now substantially smaller: candidate worker is `37` lines and controller worker is `34` lines; the parent adaptive candidate launcher is `346` lines after removing duplicated setup boilerplate.
- Updated `self/README.md` with a `Current Launchers` section and noted that adaptive AILAB scripts share the helper.
- Verification: `bash -n launchers/self/lib/adaptive_common.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch launchers/self/run_adaptive_candidate_worker_ailab.sbatch launchers/self/run_adaptive_controller_worker_ailab.sbatch`; broader adaptive syntax check also passed for `launchers/self/run_adaptive_condition_ailab.sbatch`, `launchers/self/run_adaptive_self_improvement_ailab.sbatch`, `launchers/self/submit_adaptive_candidate_training_ailab.sh`, and `launchers/self/submit_adaptive_condition_pilots_ailab.sh`.

### Implementation Log: 2026-06-17 21:48:21 UTC

- Added explicit adaptive candidate-training config files under `launchers/self/config/`: `adaptive_candidate_base.env`, `adaptive_candidate_addition.env`, and `adaptive_candidate_run_length.env`.
- Added `adaptive_source_config_file` and `adaptive_source_config_files` to `launchers/self/lib/adaptive_common.sh`. Launchers can now load `ADAPTIVE_CONFIG_FILE` or colon-separated `ADAPTIVE_CONFIG_FILES`; relative paths resolve against `ROOT_DIR`.
- Config files use default-only shell assignments, so caller-provided environment variables still override config defaults.
- Updated `launchers/self/run_adaptive_candidate_training_ailab.sbatch` to source optional config files before applying its built-in defaults, preserving the existing default behavior when no config is provided.
- Updated `launchers/self/submit_adaptive_candidate_training_ailab.sh` to forward `ADAPTIVE_CONFIG_FILES` through `sbatch --export` and record it in `submission_manifest.json`.
- Updated `self/README.md` to document adaptive candidate config files and override semantics.
- Verification: `bash -n` passed for the helper, touched launchers, submit script, and config files; source-only smoke checks confirmed config defaults load and caller environment variables win; `DRY_RUN=1` submit smoke test produced a valid `submission_manifest.json` with `adaptive_config_files`.

### Implementation Log: 2026-06-17 21:50:09 UTC

- Migrated `launchers/self/run_adaptive_condition_ailab.sbatch` and `launchers/self/run_adaptive_self_improvement_ailab.sbatch` onto `launchers/self/lib/adaptive_common.sh`.
- The two older adaptive proposal launchers now share repo-root resolution, log-directory creation, runtime environment setup, Python resolution, and worker context logging with the adaptive candidate-training launchers.
- Preserved their fixture-specific defaults, compile checks, focused-test behavior, and Python entry points; this pass only consolidated duplicated bootstrap code.
- `run_adaptive_condition_ailab.sbatch` is now `97` lines and `run_adaptive_self_improvement_ailab.sbatch` is now `82` lines.
- Updated `self/README.md` to list both scripts among the launchers using the shared helper.
- Verification: `bash -n` passed for `launchers/self/lib/adaptive_common.sh`, all five adaptive AILAB run launchers, `submit_adaptive_candidate_training_ailab.sh`, `submit_adaptive_condition_pilots_ailab.sh`, and the adaptive candidate config files.

### Implementation Log: 2026-06-17 21:51:57 UTC

- Migrated `launchers/self/submit_adaptive_candidate_training_ailab.sh` and `launchers/self/submit_adaptive_condition_pilots_ailab.sh` onto `launchers/self/lib/adaptive_common.sh` for repo-root and Python resolution.
- Preserved both submitters' existing Slurm command matrices, `DRY_RUN` behavior, output roots, log directories, and manifest schemas.
- Candidate-training submit dry-run still records `adaptive_config_files` in `submission_manifest.json`.
- Updated `self/README.md` to document that the adaptive submit scripts also source the shared helper.
- Verification: `bash -n` passed for the helper, adaptive submitters, and representative adaptive run launchers; `DRY_RUN=1` smoke tests for both submitters generated valid JSON manifests.

### Implementation Log: 2026-06-17 21:58:57 UTC

- Added packed local candidate-worker dispatch with `--candidate-local-pack-size`.
- Default behavior remains one candidate per local subprocess (`candidate_local_pack_size=1`). When set above one, local dispatch writes `candidate_jobs/pack_specs/pack_*.json` and launches one worker process per pack through the hidden `--run-candidate-pack-worker` entry point.
- Packed workers execute existing candidate specs sequentially and preserve candidate semantics: every candidate still trains from the same current checkpoint, so the model is not reused after mutation across candidates.
- The optimization reduces local subprocess launch overhead and repeated worker bootstrap/spec parsing. It does not yet eliminate per-candidate model weight loads, because safely sharing a loaded mutable model across candidates would change the independent-candidate experiment unless a larger immutable/checkpoint-cloning design is added.
- Updated `self/README.md` with packed-local worker runtime notes and narrowed the remaining optimization caveat.
- Verification: `python -m py_compile self/core/args.py self/core/worker_io.py self/core/candidate_worker_runtime.py self/core/candidate_workers.py self/core/driver.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -k 'parser_defaults or candidate_worker_spec_roundtrip_loads_inputs or local_parallel_candidate_workers'` (`4 passed`, `33 deselected`).

### Implementation Log: 2026-06-17 22:17:30 UTC

- Extracted the non-adaptive iterative self-improvement loop from `self/self_improvement_core.py` into `self/core/nonadaptive_loop.py`.
- `self/core/nonadaptive_loop.py` now owns dataset/resume handling, round training/evaluation, composed-eval metrics, dynamic composed-pool refresh, pseudo-label generation, checkpoint cleanup, and non-adaptive summary writing.
- `self/self_improvement_core.py` is now a compatibility facade for legacy imports. Its `run_self_improvement(...)` wrapper syncs selected facade globals into the extracted runtime before dispatch, preserving existing tests/scripts that monkeypatch helpers through `self.self_improvement_core`.
- `self/self_improvement_core.py` is now `132` lines, down from `760` before this extraction and `1781` before the core split began.
- Updated `self/README.md` to list `self/core/nonadaptive_loop.py` and to mark `self/self_improvement_core.py` as a facade rather than the loop owner.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/nonadaptive_loop.py tests/test_bit_task_seed_round_zero.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_self_improvement_launchers.py tests/test_bit_task_recipe.py -q` (`53 passed`); `PYTHONPATH=. conda run -n torch-env python -c "import self.self_improvement_core as compat; import self.core.nonadaptive_loop as loop; print(compat.run_self_improvement.__module__); print(loop.run_self_improvement.__module__); print(hasattr(compat, 'build_trainer')); print(hasattr(compat, 'TrainingConfig'))"` confirmed the facade and runtime import under `torch-env`.

### Implementation Log: 2026-06-17 22:23:05 UTC

- Removed internal imports through the legacy `self.self_improvement_core` facade from implementation packages: `self/core`, `self/tasks`, `self/experiments`, and `self/analysis`.
- Core modules now import directly from their canonical owners, e.g. `self.core.data_io`, `self.core.evaluation`, `self.core.model_io`, `self.core.training`, and `self.core.task_protocols`.
- Task modules keep their existing `self.self_improvement_tasks` compatibility hooks for monkeypatching proposal/prediction helpers, but their default helper imports now come from canonical modules.
- Remaining `self.self_improvement_core` imports are old top-level scripts/wrappers, which should be migrated gradually when those scripts are moved into `self/experiments`, `self/diagnostics`, `self/analysis`, or `self/legacy`.
- Updated `self/README.md` to document that implementation modules no longer depend on the facade.
- Verification: `python -m py_compile self/core/*.py self/tasks/*.py self/experiments/adaptive_self_improvement.py self/self_improvement_core.py self/self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_seed_round_zero.py tests/test_bit_task_recipe.py tests/test_adaptive_candidate_training.py -q` (`85 passed`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_analysis_artifacts.py -q` (`12 passed`).

### Implementation Log: 2026-06-17 22:27:40 UTC

- Created `self/diagnostics/` and moved the first batch of diagnostic implementations there:
  - `check_self_improvement_overfit.py`
  - `analyze_symbolic_training_dynamics.py`
  - `evaluate_fixed_composition_slices.py`
  - `addition_recipe_diagnostic.py`
  - `rectangular_multiplication_compose_diagnostic.py`
- Left top-level compatibility wrappers for each moved diagnostic so existing launchers and commands such as `python -m self.check_self_improvement_overfit` and `python -m self.addition_recipe_diagnostic` continue to work.
- Updated moved diagnostics to import from canonical `self.core.*` modules and, where needed, from `self.diagnostics.*` instead of from legacy top-level wrappers.
- Updated `self/README.md` with a `Current Diagnostics` section and wrapper mappings.
- Verification: `python -m py_compile self/diagnostics/*.py self/analyze_symbolic_training_dynamics.py self/check_self_improvement_overfit.py self/evaluate_fixed_composition_slices.py self/addition_recipe_diagnostic.py self/rectangular_multiplication_compose_diagnostic.py self/rectangular_multiplication_self_improvement.py`; `PYTHONPATH=. conda run -n torch-env python -c "import self.check_self_improvement_overfit as old; import self.diagnostics.check_self_improvement_overfit as new; print(old.build_task_bundle is new.build_task_bundle); import self.addition_recipe_diagnostic as old_add; import self.diagnostics.addition_recipe_diagnostic as new_add; print(old_add.build_parser is new_add.build_parser)"` (`True`, `True`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_recipe_recovery_launchers.py tests/test_addition_tiny_seed_mig_launcher.py tests/test_multiplication_rectangular_square_launchers.py -q` (`15 passed`); CLI help smoke checks passed for old and new diagnostic module paths.

### Implementation Log: 2026-06-17 22:31:59 UTC

- Moved paper/figure experiment implementations from top-level `self/` into `self/experiments/`:
  - `figure2_condition_sweep.py`
  - `figure2_paper_retune.py`
  - `figure3_seed_quality_sweep.py`
  - `figure3_real_seed_data_ablation.py`
  - `paper_schedule_selection.py`
- Left top-level wrappers for each moved module so old imports and launchers such as `python -m self.figure2_condition_sweep submit` and `python -m self.figure3_seed_quality_sweep submit` continue to work.
- Updated moved modules to import sibling helpers through `self.experiments.*` and adjusted `ROOT_DIR` resolution after the one-level-deeper move.
- Updated `self/README.md` with the new experiment modules and compatibility wrapper mappings.
- Verification: `python -m py_compile self/experiments/figure2_condition_sweep.py self/experiments/figure2_paper_retune.py self/experiments/figure3_real_seed_data_ablation.py self/experiments/figure3_seed_quality_sweep.py self/experiments/paper_schedule_selection.py self/figure2_condition_sweep.py self/figure2_paper_retune.py self/figure3_real_seed_data_ablation.py self/figure3_seed_quality_sweep.py self/paper_schedule_selection.py`; `PYTHONPATH=. conda run -n torch-env python -c "import self.figure2_condition_sweep as old; import self.experiments.figure2_condition_sweep as new; print(old.build_parser is new.build_parser); print(new.ROOT_DIR); import self.paper_schedule_selection as old_sel; import self.experiments.paper_schedule_selection as new_sel; print(old_sel.selection_schedule is new_sel.selection_schedule)"` confirmed wrapper identity and repo-root resolution; `PYTHONPATH=. conda run -n torch-env pytest tests/test_figure2_condition_sweep.py tests/test_figure2_recipe_aggressive_launchers.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py tests/test_paper_schedule_selection.py -q` (`27 passed`); old/new CLI help smoke checks passed for Figure 2 and Figure 3 module paths.

### Implementation Log: 2026-06-17 22:35:24 UTC

- Moved notebook/plot/result-analysis utilities from top-level `self/` into `self/analysis/`:
  - `training_curve_notebook_utils.py`
  - `seed_fit_curve_notebook_utils.py`
  - `plot_appendix_baseline_heatmaps.py`
  - `plot_self_improvement_figure.py`
  - `summarize_seed_fit_grid.py`
- Left top-level compatibility wrappers so existing notebooks and commands importing `self.training_curve_notebook_utils`, `self.seed_fit_curve_notebook_utils`, or running `python -m self.plot_self_improvement_figure` continue to work.
- Updated `self/experiments/figure2_paper_retune.py` to import the heatmap helper from `self.analysis.training_curve_notebook_utils`.
- Updated `self/README.md` with analysis utility modules and wrapper mappings.
- Verification: `python -m py_compile self/analysis/training_curve_notebook_utils.py self/analysis/seed_fit_curve_notebook_utils.py self/analysis/plot_appendix_baseline_heatmaps.py self/analysis/plot_self_improvement_figure.py self/analysis/summarize_seed_fit_grid.py self/training_curve_notebook_utils.py self/seed_fit_curve_notebook_utils.py self/plot_appendix_baseline_heatmaps.py self/plot_self_improvement_figure.py self/summarize_seed_fit_grid.py self/experiments/figure2_paper_retune.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_training_curve_notebook_utils.py -q` (`4 passed`); wrapper identity checks passed for training-curve and seed-fit notebook utility imports; old/new CLI help smoke checks passed for self-improvement plotting, appendix heatmaps, and seed-fit grid summary module paths.

### Implementation Log: 2026-06-17 22:41:07 UTC

- Moved seed-fit and rectangular multiplication experiment implementations under `self/experiments/` while preserving top-level compatibility wrappers:
  - `seed_fit_experiment.py`
  - `rectangular_multiplication_seed_fit.py`
  - `rectangular_multiplication_recipe_seed_fit.py`
  - `rectangular_multiplication_self_improvement.py`
  - `multiplication_rectangular_tune.py`
- Updated the moved seed-fit experiment to import task adapters directly from `self.tasks.*` instead of through the old `self.self_improvement_tasks` facade.
- At this point, `self/multiplication_rectangular.py` was temporarily kept at top level because it was still a shared rectangular-task helper used by tests, launchers, and experiment modules rather than a single CLI script. It was moved under `self/tasks/rectangular_multiplication.py` in the later `2026-06-17 23:04:07 UTC` cleanup pass.
- Updated `self/README.md` with the moved experiment modules, wrapper mappings, and remaining cleanup queue status.
- Verification: stale-import search over the moved modules found no `self.self_improvement_core`, `self.self_improvement_tasks`, or old rectangular diagnostic imports; `python -m py_compile self/experiments/seed_fit_experiment.py self/experiments/rectangular_multiplication_seed_fit.py self/experiments/rectangular_multiplication_recipe_seed_fit.py self/experiments/rectangular_multiplication_self_improvement.py self/experiments/multiplication_rectangular_tune.py self/seed_fit_experiment.py self/rectangular_multiplication_seed_fit.py self/rectangular_multiplication_recipe_seed_fit.py self/rectangular_multiplication_self_improvement.py self/multiplication_rectangular_tune.py self/multiplication_rectangular.py`; wrapper identity checks passed for seed-fit, rectangular recipe seed-fit, rectangular self-improvement, and rectangular tune imports under `torch-env`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py tests/test_multiplication_rectangular_seed_launchers.py tests/test_multiplication_rectangular_self_improvement_launchers.py tests/test_multiplication_rectangular_square_launchers.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_addition_tiny_seed_mig_launcher.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`63 passed`); old/new CLI help smoke checks passed for seed-fit, rectangular recipe seed-fit, rectangular self-improvement, and rectangular tune module paths.

### Implementation Log: 2026-06-17 22:48:31 UTC

- Removed remaining `self.self_improvement_tasks` facade imports from current implementation packages touched by adaptive/config work.
- `self/core/args.py`, `self/core/composition.py`, `self/core/proposal_runtime.py`, and `self/core/driver.py` now import task classes, run-length constants, parsers, and run-state helpers from canonical `self.tasks.bit_common`, `self.tasks.run_length`, and `self.tasks.addition` modules.
- `self/diagnostics/check_self_improvement_overfit.py`, `self/diagnostics/evaluate_fixed_composition_slices.py`, and `self/diagnostics/addition_recipe_diagnostic.py` now import task classes and run-length helpers from canonical task/core modules rather than through the old task facade.
- Remaining `self.self_improvement_core` / `self.self_improvement_tasks` imports are now confined to the facades themselves and old top-level compatibility CLIs/scripts such as `self/run_length_self_improvement.py`, the legacy auxiliary bit-task wrapper, `self/multiplication_self_improvement.py`, `self/run_length_balanced_eval.py`, and the large historical `self/self_improvement.py`.
- Updated `self/README.md` to document the cleaned implementation import boundary and the remaining top-level compatibility boundary.
- Verification: `python -m py_compile self/core/args.py self/core/composition.py self/core/proposal_runtime.py self/core/driver.py self/diagnostics/check_self_improvement_overfit.py self/diagnostics/evaluate_fixed_composition_slices.py self/diagnostics/addition_recipe_diagnostic.py`; stale-import search over `self/core`, `self/tasks`, `self/experiments`, `self/diagnostics`, and `self/analysis` shows no implementation-package imports through `self.self_improvement_core` or `self.self_improvement_tasks`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py tests/test_self_improvement_tasks.py tests/test_bit_task_seed_round_zero.py tests/test_bit_task_recipe.py tests/test_addition_recipe_diagnostic.py tests/test_addition_recipe_recovery_launchers.py tests/test_multiplication_rectangular_square_launchers.py -q` (`111 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 22:52:36 UTC

- Created `self/legacy/` and moved the large historical addition self-improvement implementation from `self/self_improvement.py` to `self/legacy/addition_self_improvement.py`.
- Replaced `self/self_improvement.py` with a compatibility wrapper that reexports the legacy implementation and still calls `main()` for `python -m self.self_improvement`.
- Updated the moved implementation to import `build_generation_encodings` from canonical `self.core.evaluation` instead of through the `self.self_improvement_core` facade.
- Kept `self/self_improvement_experiment.py` and `self/self_improvement_composition_error_experiment.py` compatible because they import `main` from the old `self.self_improvement` module path.
- Updated `self/README.md` with a `Current Legacy` section and the new compatibility-wrapper mapping.
- Verification: `python -m py_compile self/legacy/__init__.py self/legacy/addition_self_improvement.py self/self_improvement.py self/self_improvement_experiment.py self/self_improvement_composition_error_experiment.py`; import identity under `torch-env` confirmed `self.self_improvement.main`, `parse_args`, and `normalize_args` are the same function objects as `self.legacy.addition_self_improvement`; stale-import search shows no `self.self_improvement_core` imports inside `self/legacy`; old/new CLI help smoke checks passed for `self.self_improvement` and `self.legacy.addition_self_improvement`, with only the expected argparse program-name wrapping difference; wrapper help checks passed for `self.self_improvement_composition_error_experiment` and `self.self_improvement_experiment`; `bash -n` passed for the addition launchers that call `self.self_improvement`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_launchers.py tests/test_addition_fullpack_filtered_launcher.py tests/test_addition_recipe_recovery_launchers.py tests/test_addition_exact_digits_fixed_binary_launcher.py tests/test_addition_fixedwidth_moredata_launcher.py tests/test_figure2_condition_sweep.py tests/test_figure2_recipe_aggressive_launchers.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`42 passed`).

### Implementation Log: 2026-06-17 23:00:13 UTC

- Moved the remaining historical task-specific non-adaptive CLI implementations under `self/legacy/`:
  - `bit_task_self_improvement.py`
  - the auxiliary bit-task self-improvement wrapper
  - `run_length_self_improvement.py`
  - `multiplication_self_improvement.py`
- Left top-level compatibility wrappers for all four module paths so active launchers and commands such as `python -m self.run_length_self_improvement` keep working.
- Updated moved bit-task CLIs to import the shared parser from `self.legacy.bit_task_self_improvement` instead of through the top-level wrapper.
- Moved `run_length_balanced_eval.py` into `self/diagnostics/run_length_balanced_eval.py`, left a top-level wrapper, and updated the diagnostic to import IO/model/evaluation helpers from `self.core.*` plus run-length symbols from `self.tasks.*` instead of through the old facades.
- Updated `self/diagnostics/check_self_improvement_overfit.py` to import parser/normalization helpers directly from the legacy implementation modules rather than from top-level wrappers.
- Updated `self/README.md` with the new legacy and diagnostic modules plus wrapper mappings.
- Verification: `python -m py_compile self/legacy/bit_task_self_improvement.py self/legacy/run_length_self_improvement.py self/legacy/multiplication_self_improvement.py self/run_length_self_improvement.py self/multiplication_self_improvement.py self/diagnostics/run_length_balanced_eval.py self/run_length_balanced_eval.py self/diagnostics/check_self_improvement_overfit.py`; wrapper identity checks under `torch-env` passed for the legacy run-length bit-string helper, run-length, multiplication, and balanced-eval module paths; stale-import search shows remaining `self.self_improvement_core` / `self.self_improvement_tasks` imports only inside `self/legacy/*` and the task facade itself, preserving old non-adaptive monkeypatch behavior; old/new CLI help smoke checks passed for legacy run-length bit-string helper, run-length, multiplication, and balanced-eval module paths; `bash -n` passed for launchers that reference the old task CLI paths; `PYTHONPATH=. conda run -n torch-env pytest tests/test_run_length_balanced_eval.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_self_improvement_launchers.py tests/test_run_length_fixed_binary_launchers.py tests/test_figure2_condition_sweep.py tests/test_figure2_recipe_aggressive_launchers.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py tests/test_bit_task_seed_round_zero.py tests/test_bit_task_recipe.py -q` (`45 passed`).

### Implementation Log: 2026-06-17 23:04:07 UTC

- Moved the shared rectangular multiplication helper from `self/multiplication_rectangular.py` to `self/tasks/rectangular_multiplication.py`.
- Replaced the old top-level module with a compatibility proxy wrapper that forwards attribute reads and writes to the canonical task module. This preserves old imports and monkeypatch patterns such as patching `self.multiplication_rectangular.sample_int_with_exact_digits` before calling `build_sampled_rectangular_dataset`.
- Updated rectangular experiment and diagnostic implementation modules to import from `self.tasks.rectangular_multiplication` directly:
  - `self/experiments/rectangular_multiplication_recipe_seed_fit.py`
  - `self/experiments/rectangular_multiplication_self_improvement.py`
  - `self/diagnostics/rectangular_multiplication_compose_diagnostic.py`
- Updated `self/README.md` with the new task module and wrapper mapping.
- Verification: `python -m py_compile self/tasks/rectangular_multiplication.py self/multiplication_rectangular.py self/experiments/rectangular_multiplication_recipe_seed_fit.py self/experiments/rectangular_multiplication_self_improvement.py self/diagnostics/rectangular_multiplication_compose_diagnostic.py`; wrapper identity and attribute-forwarding checks passed under `torch-env`; stale-import search over implementation packages found no imports through `self.multiplication_rectangular`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py tests/test_multiplication_rectangular_seed_launchers.py tests/test_multiplication_rectangular_self_improvement_launchers.py tests/test_multiplication_rectangular_square_launchers.py -q` (`44 passed`); `bash -n` passed for rectangular multiplication seed, self-improvement, square diagnostic, tune, and submit launchers.

### Implementation Log: 2026-06-17 23:07:27 UTC

- Moved the historical multiplication CoT pseudo-addition curriculum implementation from `self/self_improvement_multiplication_cot_pseudo_addition.py` to `self/legacy/multiplication_cot_pseudo_addition.py`.
- Replaced the old top-level module with a compatibility wrapper, preserving `python -m self.self_improvement_multiplication_cot_pseudo_addition` for `run_scripts/run_self_improvement.multiplication_cot.curriculum_pseudo_addition.sh`.
- Updated `self/README.md` with the new legacy module and wrapper mapping.
- Verification: `python -m py_compile self/legacy/multiplication_cot_pseudo_addition.py self/self_improvement_multiplication_cot_pseudo_addition.py`; import identity checks under `torch-env` confirmed the old module reexports `main`, `parse_args`, and `summary_to_payload` from the moved implementation; old/new CLI help smoke checks passed, with only expected argparse usage wrapping/program-name differences; `bash -n run_scripts/run_self_improvement.multiplication_cot.curriculum_pseudo_addition.sh` passed.

### Implementation Log: 2026-06-17 23:12:35 UTC

- Extracted the high-level adaptive candidate-training `run(args)` orchestration from `self/core/driver.py` into `self/core/run_orchestration.py`.
- Added `AdaptiveRunDeps` so `self/core/driver.py` still injects driver-level functions such as `train_candidate_metrics`, `select_candidate`, worker dispatch, proposal validation, JSON writing, and trace builders. This preserves existing monkeypatch behavior through `self.adaptive_candidate_training` while moving the actual run sequence into a focused module.
- `self/core/driver.py` now delegates to `run_adaptive_candidate_training(...)` and dropped from `721` lines to `637` lines in this pass.
- Updated `self/README.md` with the new orchestration module.
- Verification: `python -m py_compile self/core/run_orchestration.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py -q` (`47 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 23:16:09 UTC

- Extracted adaptive CLI/worker-mode dispatch from `self/core/driver.py` into `self/core/entrypoint.py`.
- Added `DriverEntrypointDeps` so the driver still injects driver-level handlers for normal runs, controller workers, candidate workers, and packed candidate workers. This keeps `self.adaptive_candidate_training.main(...)` behavior and old monkeypatch paths intact while isolating CLI routing.
- `self/core/driver.py` now delegates `main(argv)` to `run_driver_entrypoint(...)` and dropped from `637` lines to `624` lines in this pass.
- Updated `self/README.md` with the new entrypoint module.
- Verification: `python -m py_compile self/core/entrypoint.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py -q` (`47 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 23:22:13 UTC

- Extracted candidate/controller worker spec-entrypoint wiring from `self/core/driver.py` into `self/core/worker_entrypoints.py`.
- Added `WorkerEntrypointDeps` plus candidate/controller runtime dependency builders in the new module. The driver still builds this dependency object from driver-level globals at call time, preserving old monkeypatch behavior for `train_and_score_candidate`, `run_seed_phase`, `run_round_model_phase`, `apply_proposal_grpo_update`, `run_candidate_worker_from_spec`, and `run_controller_worker_from_spec`.
- Kept public driver-level worker functions (`run_candidate_worker_from_spec`, `run_candidate_worker`, `run_candidate_worker_pack_from_spec`, `run_candidate_pack_worker`, `run_seed_controller_worker_from_spec`, `run_round_model_controller_worker_from_spec`, `run_proposal_grpo_controller_worker_from_spec`, `run_controller_worker_from_spec`, and `run_controller_worker`) as thin wrappers.
- Removed unused private worker runtime factory wrappers from the driver; `self/core/driver.py` dropped from `624` lines to `616` lines in this pass.
- Updated `self/README.md` with the new worker-entrypoint module.
- Verification: `python -m py_compile self/core/worker_entrypoints.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py -q` (`47 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 23:26:02 UTC

- Extracted adaptive candidate selection from `self/core/driver.py` into `self/core/candidate_selection.py`.
- Extracted adaptive task-name lookup from `self/core/driver.py` into `self/core/task_registry.py`.
- Moved generic JSON artifact writing into `self/core/data_io.py` as `write_json`.
- Kept `select_candidate`, `task_for_name`, and `write_json` imported and reexported by the driver so `self.adaptive_candidate_training` compatibility imports and monkeypatch paths continue to work.
- `self/core/driver.py` dropped from `616` lines to `587` lines in this pass.
- Updated `self/README.md` with the new module ownership and remaining-cleanup status.
- Verification: `python -m py_compile self/core/candidate_selection.py self/core/task_registry.py self/core/data_io.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py -q` (`47 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 23:32:18 UTC

- Extracted candidate-dispatch compatibility wiring from `self/core/driver.py` into `self/core/candidate_dispatch_entrypoints.py`.
- The new `CandidateDispatchEntrypointDeps` object threads driver-level functions into the focused dispatch wrappers, preserving monkeypatch behavior for `train_and_score_candidate`, `_candidate_failure_metrics`, `_collect_candidate_array_metrics`, and the serial/local-parallel/Slurm candidate-training entry points.
- Kept the old driver-level dispatch functions as compact keyword-forwarding wrappers so existing imports through `self.adaptive_candidate_training` continue to work.
- `self/core/driver.py` dropped from `587` lines to `437` lines in this pass.
- Updated `self/README.md` with the new candidate-dispatch entrypoint module and remaining-cleanup status.
- Verification: `python -m py_compile self/core/candidate_dispatch_entrypoints.py self/core/driver.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_frontier.py -q` (`47 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-17 23:35:39 UTC

- Continued Slurm launcher consolidation for the adaptive AILAB scripts.
- Added shared Slurm helper functions to `launchers/self/lib/adaptive_common.sh`:
  - `adaptive_set_sbatch_defaults` for caller-overridable partition/GRES/CPU/memory/time defaults.
  - `adaptive_add_sbatch_resources` for appending the common Slurm resource arguments to a command array.
  - `adaptive_print_command` for consistent dry-run/submit command logging.
- Updated `launchers/self/submit_adaptive_condition_pilots_ailab.sh` to use the shared Slurm resource helpers when building condition-pilot jobs.
- Updated `launchers/self/submit_main_experiments_ailab.sh` to source `adaptive_common.sh` for repo-root/Python resolution and shared Slurm resource helpers instead of maintaining its own setup block.
- Updated `self/README.md` with the launcher helper ownership and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/adaptive_common.sh launchers/self/submit_adaptive_condition_pilots_ailab.sh launchers/self/submit_main_experiments_ailab.sh launchers/self/submit_adaptive_candidate_training_ailab.sh`; dry-run smoke checks passed for `submit_adaptive_condition_pilots_ailab.sh` and `submit_main_experiments_ailab.sh` with temporary output/log directories and `DRY_RUN=1`.

### Implementation Log: 2026-06-17 23:39:08 UTC

- Added conservative source/artifact hygiene rules to the root `.gitignore`.
- New ignored generated artifacts include executed notebooks (`*.executed.ipynb`), heavyweight model/run files (`*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.tar.gz`, `*.zip`), unpacked `self_improvement_*_addmodel/` bundles, downloaded `meta/models/` caches, local editor/agent state, OS/tool caches, and Slurm/LaTeX intermediate logs.
- Kept source notebooks (`*.ipynb` without `.executed`) and report sources visible; tracked model-card assets such as `models/addition_self_improvement_round8/*` remain tracked despite matching new artifact suffixes.
- Updated `self/README.md` with the source/artifact policy and remaining-cleanup status.
- Verification: `git check-ignore -v` confirms `.codex`, `.vscode/settings.json`, `.DS_Store`, executed notebooks, generated archives, downloaded `meta/models`, unpacked `self_improvement_*_addmodel/` bundles, and LaTeX logs are ignored; normal `git status --short --untracked-files=normal` no longer reports those generated patterns while still reporting source notebooks and code.

### Implementation Log: 2026-06-17 23:43:07 UTC

- Continued launcher consolidation by splitting generic setup/Slurm helpers into `launchers/self/lib/self_common.sh`.
- `launchers/self/lib/adaptive_common.sh` now sources `self_common.sh` and keeps the existing `adaptive_*` function names as compatibility wrappers, while retaining adaptive-only HF cache/offline setup, config-file sourcing, and worker-context logging.
- Updated `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` to use `self_common.sh` for repo-root setup and common MIG Slurm resource argument construction. Experiment commands, defaults, manifest format, and dry-run output remain unchanged.
- Updated `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` with the same generic helper pattern. Experiment commands, defaults, manifest format, and dry-run output remain unchanged.
- Fixed generic repo-root detection to resolve the outermost launcher script in the Bash source stack, so calls routed through `adaptive_common.sh` still resolve the repository root rather than `launchers/`.
- Updated `self/README.md` with the generic launcher helper and migrated submitters.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh launchers/self/submit_adaptive_condition_pilots_ailab.sh launchers/self/submit_main_experiments_ailab.sh`; manual `DRY_RUN=1` smoke checks passed for the two migrated addition submitters and for `submit_main_experiments_ailab.sh` root detection; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_exact_digits_fixed_binary_launcher.py tests/test_addition_fixedwidth_moredata_launcher.py -q` (`5 passed`).

### Implementation Log: 2026-06-17 23:45:40 UTC

- Continued launcher consolidation by adding `self_add_sbatch_explicit_resources` to `launchers/self/lib/self_common.sh` for submitters that mix GPU and CPU jobs or otherwise need per-job resource blocks.
- Updated `launchers/self/submit_run_length_fixed_binary_mig.sh` to source `self_common.sh` for repo-root setup, Python resolution, command printing, and explicit Slurm resource argument construction.
- Preserved the run-length fixed-binary experiment commands, job dependencies, manifest schema, and dry-run behavior.
- Updated `self/README.md` with the explicit-resource helper and migrated run-length fixed-binary submitter.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/submit_run_length_fixed_binary_mig.sh`; manual `DRY_RUN=1` smoke check passed for `submit_run_length_fixed_binary_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_run_length_fixed_binary_launchers.py -q` (`3 passed`).

### Implementation Log: 2026-06-17 23:48:44 UTC

- Continued launcher consolidation for the guarded plain-output bit diagnostics.
- Added `self_print_command_stdout` to `launchers/self/lib/self_common.sh` so launchers whose dry-run tests inspect stdout can still share command-printing logic without moving command output to stderr.
- Updated `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` to source `self_common.sh` for repo-root setup and shared stdout command printing. Its task matrix, exported environment, and dry-run messages are preserved.
- Updated `launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch` to source `self_common.sh` for repo-root setup, Python resolution, and command printing. Its task-specific command construction and dry-run stdout remain unchanged.
- Updated `self/README.md` with the migrated guarded diagnostic runner/submitter.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch`; manual `DRY_RUN=1` smoke checks passed for the guarded diagnostic submitter and runner; `PYTHONPATH=. conda run -n torch-env pytest tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`5 passed`).

### Implementation Log: 2026-06-17 23:51:35 UTC

- Continued launcher consolidation for the rectangular multiplication square probe/resweep scripts and seed/diagnostic runners.
- Added `self_parse_bool` and `self_print_prefixed_command_stdout` to `launchers/self/lib/self_common.sh`, then implemented `self_print_command_stdout` through the prefix-aware helper. This preserves existing dry-run labels such as `Seed submit`, `Diagnostic submit`, and `Submit` while sharing the command-printing logic.
- Updated `launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch` to source `self_common.sh` for repo-root setup, Python resolution, boolean parsing, and command printing. Its square defaults, command construction, and dry-run stdout are preserved.
- Updated `launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch` with the same shared helper pattern while preserving seed/frontier partition defaults and diagnostic command construction.
- Updated `launchers/self/submit_multiplication_rectangular_square_probe_mig.sh` and `launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh` to source `self_common.sh` for repo-root setup and prefixed dry-run command printing.
- Updated `self/README.md` with the migrated rectangular square launcher family.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_multiplication_rectangular_square_probe_mig.sh launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch`; manual `DRY_RUN=1` smoke checks passed for the probe wrapper, seed runner, and diagnostic runner; `PYTHONPATH=. conda run -n torch-env pytest tests/test_multiplication_rectangular_square_launchers.py -q` (`6 passed`).

### Implementation Log: 2026-06-17 23:56:24 UTC

- Continued launcher consolidation for the rectangular multiplication non-square seed sweep and full-pack self-improvement launcher family.
- Added `self_print_prefixed_command` to `launchers/self/lib/self_common.sh` so submitters that intentionally log commands to stderr can share the same command-formatting path as stdout dry-run scripts.
- Updated `launchers/self/run_multiplication_rectangular_seed_mig.sbatch` and `launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch` to source `self_common.sh` for repo-root setup, Python resolution, boolean parsing, and dry-run command printing. Their task-specific command construction, defaults, and dry-run stdout remain unchanged.
- Updated `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` and `launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` to source `self_common.sh` for repo-root setup, Python resolution, and prefixed submit-command printing while preserving their existing stdout/stderr behavior.
- Updated `self/README.md` with the migrated rectangular non-square launcher family and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_seed_mig.sbatch launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh launchers/self/submit_multiplication_rectangular_fullpack_mig.sh`; manual `DRY_RUN=1` smoke check passed for `submit_multiplication_rectangular_seed_sweep_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_multiplication_rectangular_seed_launchers.py tests/test_multiplication_rectangular_self_improvement_launchers.py -q` (`10 passed`).

### Implementation Log: 2026-06-17 23:59:04 UTC

- Continued launcher consolidation for `launchers/self/submit_multiplication_rectangular_tune_mig.sh`.
- Replaced its duplicated repo-root and Python-resolution shell setup with `launchers/self/lib/self_common.sh`, and switched dry-run flag parsing and command printing to the shared boolean/command helpers.
- Preserved the rectangular tune CLI command, default output/log/model paths, stage selection paths, and stdout dry-run behavior.
- Updated `self/README.md` with the migrated tune submitter and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_multiplication_rectangular_tune_mig.sh`; manual `DRY_RUN=1` smoke check passed for `submit_multiplication_rectangular_tune_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_multiplication_rectangular_tune.py -q` (`11 passed`).

### Implementation Log: 2026-06-18 00:01:55 UTC

- Continued launcher consolidation for the addition fullpack-filtered runner and submitter.
- Updated `launchers/self/run_addition_fullpack_filtered.sbatch` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and dry-run command printing. Its seed-model defaults, tokenizer mode, baseline commands, and dry-run stdout remain unchanged.
- Updated `launchers/self/submit_addition_fullpack_filtered_mig.sh` to use the same helper for repo-root setup and boolean parsing while preserving its five-baseline dry-run listing.
- Updated `self/README.md` with the migrated addition fullpack-filtered launcher pair and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fullpack_filtered.sbatch launchers/self/submit_addition_fullpack_filtered_mig.sh`; manual `DRY_RUN=1` smoke checks passed for `run_addition_fullpack_filtered.sbatch` with `BASELINE=with_carry_filtered` and for `submit_addition_fullpack_filtered_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_fullpack_filtered_launcher.py -q` (`5 passed`).

### Implementation Log: 2026-06-18 00:03:27 UTC

- Continued launcher consolidation for `launchers/self/run_addition_their_recipe_diagnostic.sh`.
- Replaced its duplicated repo-root/Python setup with `launchers/self/lib/self_common.sh`, and switched dry-run parsing plus command printing to the shared helper functions.
- Preserved the addition recipe diagnostic command, recipe/device defaults, optional train/eval/max-step flags, and dry-run stdout behavior.
- Updated `self/README.md` with the migrated addition recipe diagnostic launcher and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_their_recipe_diagnostic.sh`; manual `DRY_RUN=1` smoke check passed with explicit device/batch/max-step overrides; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_their_recipe_diagnostic_launcher.py -q` (`2 passed`).

### Implementation Log: 2026-06-18 00:06:18 UTC

- Continued launcher consolidation for the fixed-width mixed-prompt addition launcher family.
- Updated `launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and dry-run command printing. Its seed-fit command, scratch recipe defaults, fixed-width mixed-prompt flags, and seed-model symlink behavior are preserved.
- Updated `launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh` to use the same helper for repo-root/Python setup, boolean parsing, and command printing. Its baseline set, fullpack defaults, original composition path controls, and optional extra-argument forwarding are preserved.
- Updated `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` to use the generic helper for repo-root setup and boolean parsing while preserving its seed/fullpack/original-composition dry-run flow and submission manifest.
- Updated `self/README.md` with the migrated fixed-width mixed-prompt addition launcher family and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_mixed_mig.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; manual `DRY_RUN=1` smoke checks passed for the seed runner, fullpack runner, and mixed submitter; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_fixedwidth_moredata_launcher.py -q` (`3 passed`).

### Implementation Log: 2026-06-18 00:08:26 UTC

- Continued launcher consolidation for the addition recipe focused/fullpack/recovery workflow.
- Updated `launchers/self/run_addition_recipe_focused.sh` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and dry-run command printing. Its recipe baseline selector, focused schedule defaults, and baseline argument mapping are preserved.
- Updated `launchers/self/run_addition_recipe_fullpack.sh` to use the same helper for repo-root/Python setup, boolean parsing, and command printing while preserving paper-schedule environment sourcing and fullpack baseline defaults.
- Updated `launchers/self/run_addition_recipe_recovery.sh` to source the generic helper for repo-root setup and dry-run parsing while preserving the diagnostic/focused/fullpack stage chain and gate logic.
- Updated `self/README.md` with the migrated addition recipe workflow and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_recipe_focused.sh launchers/self/run_addition_recipe_fullpack.sh launchers/self/run_addition_recipe_recovery.sh`; manual `DRY_RUN=1` smoke checks passed for focused, fullpack, and recovery scripts; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_recipe_recovery_launchers.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 00:11:28 UTC

- Continued launcher consolidation for the generic seed-fit grid runner and submitter.
- Updated `launchers/self/run_seed_fit_experiment.sbatch` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, and boolean parsing.
- Added a `DRY_RUN=1` verification path to the seed-fit runner that prints the resolved `self.seed_fit_experiment` command and skips CUDA/model/tokenizer preflight plus training. Normal non-dry-run execution remains unchanged.
- Updated `launchers/self/submit_seed_fit_grid.sh` to use the generic helper for repo-root setup and dry-run parsing while preserving its grid construction and TSV metadata outputs.
- Added `tests/test_seed_fit_launchers.py` covering shell syntax, runner dry-run command emission, and submitter dry-run manifest generation.
- Updated `self/README.md` with the migrated seed-fit launcher pair and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_seed_fit_experiment.sbatch launchers/self/submit_seed_fit_grid.sh`; manual `DRY_RUN=1` smoke checks passed for `run_seed_fit_experiment.sbatch` and `submit_seed_fit_grid.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_seed_fit_launchers.py -q` (`3 passed`).

### Implementation Log: 2026-06-18 00:13:53 UTC

- Continued launcher consolidation for the generic self-improvement budget-grid runner and submitter.
- Updated `launchers/self/run_task_self_improvement.sbatch` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, and boolean parsing.
- Added a `DRY_RUN=1` verification path to the generic task runner that prints the resolved task module command and skips CUDA/model/tokenizer preflight plus training. Normal non-dry-run execution remains unchanged.
- Updated `launchers/self/submit_budget_grid_self_improvement.sh` to use the generic helper for repo-root setup and dry-run parsing while preserving its task/mode/budget grid construction and TSV metadata outputs.
- Added `tests/test_task_self_improvement_launchers.py` covering shell syntax, runner dry-run command emission, and budget-grid manifest generation.
- Updated `self/README.md` with the migrated task self-improvement budget-grid launcher pair and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_task_self_improvement.sbatch launchers/self/submit_budget_grid_self_improvement.sh launchers/self/run_seed_fit_experiment.sbatch launchers/self/submit_seed_fit_grid.sh`; manual `DRY_RUN=1` smoke checks passed for `run_task_self_improvement.sbatch` and `submit_budget_grid_self_improvement.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_task_self_improvement_launchers.py tests/test_seed_fit_launchers.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 00:16:34 UTC

- Continued launcher consolidation for the small Figure 2/3 experiment submitter wrappers.
- Updated `launchers/self/submit_figure2_condition_sweep_mig.sh`, `launchers/self/submit_figure3_seed_quality_sweep_mig.sh`, and `launchers/self/submit_figure3_real_seed_data_ablation_mig.sh` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and dry-run command printing.
- Preserved the Python experiment CLI invocations, output/log defaults, and dry-run submission matrix behavior.
- Updated `self/README.md` with the migrated Figure 2/3 submitter wrappers and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_figure2_condition_sweep_mig.sh launchers/self/submit_figure3_seed_quality_sweep_mig.sh launchers/self/submit_figure3_real_seed_data_ablation_mig.sh`; manual `DRY_RUN=1` smoke check passed for `submit_figure2_condition_sweep_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_figure2_condition_sweep.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`16 passed`).

### Implementation Log: 2026-06-18 00:19:20 UTC

- Continued launcher consolidation for the Figure 2 recipe workflow.
- Updated `launchers/self/run_figure2_recipe_aggressive.sh` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and command printing. Its stage orchestration, paper-schedule environment sourcing, task schedules, gates, and dry-run behavior are preserved.
- Updated `launchers/self/submit_figure2_recipe_aggressive.sh` to use the generic helper for repo-root setup, Python resolution, dry-run parsing, and submission-command printing while preserving its Slurm resource resolution.
- Updated `launchers/self/run_figure2_paper_retune.sh` to use the same helper path for repo-root/Python setup, boolean flag parsing, and command printing.
- Updated `self/README.md` with the migrated Figure 2 recipe launcher family and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_figure2_recipe_aggressive.sh launchers/self/submit_figure2_recipe_aggressive.sh launchers/self/run_figure2_paper_retune.sh`; manual `DRY_RUN=1` smoke check passed for `run_figure2_recipe_aggressive.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_figure2_recipe_aggressive_launchers.py -q` (`5 passed`).

### Implementation Log: 2026-06-18 00:22:33 UTC

- Continued launcher consolidation for the addition seed runner scripts.
- Updated `launchers/self/run_addition_tiny_seed_mig.sbatch` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, boolean parsing, and staged dry-run command printing. The stage-0 seed-fit command, stage-1 sweep commands, staged status output, and existing seed defaults are preserved.
- Updated `launchers/self/run_addition_seed_shared.sbatch` to use the same generic helper for repo-root/Python setup, boolean parsing, and dry-run command printing while preserving its shared seed-training command and CUDA preflight behavior.
- Updated `self/README.md` with the migrated addition seed runner pair and remaining launcher-cleanup status.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_tiny_seed_mig.sbatch launchers/self/run_addition_seed_shared.sbatch`; manual `DRY_RUN=1` smoke checks passed for both addition seed runners; `PYTHONPATH=. conda run -n torch-env pytest tests/test_addition_tiny_seed_mig_launcher.py -q` (`3 passed`).

### Implementation Log: 2026-06-18 00:26:49 UTC

- Completed the current top-level self launcher helper-consolidation pass.
- Updated `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh` to source `launchers/self/lib/self_common.sh` for repo-root setup, Python resolution, default Slurm resource construction, dry-run parsing, and submit-command printing. The three baseline commands and default MIG resources are preserved.
- Updated `launchers/self/run_self_improvement_mig_boundary_eval.sbatch` and `launchers/self/run_self_improvement_qwen_no_growth.sbatch` to use the shared repo-root/Python helper while preserving their current worktree experiment commands and Slurm-safe `SLURM_SUBMIT_DIR` helper lookup.
- Updated `launchers/self/run_local_workshop_batch.sh` and `launchers/self/run_refocused_self_improvement_local.sh` to use the generic helper for repo-root setup and print-only boolean parsing while preserving their generated command matrices.
- Updated `launchers/self/run_composition_error_sweep_self_improvement.sh` to use the generic helper for repo-root/Python setup and dry-run command printing. This also fixes its stale launcher-local repo-root calculation, which pointed one directory above the repository from the current file location.
- Updated `self/README.md` with the migrated launcher batch and noted that the current `launchers/self/*.{sh,sbatch}` inventory no longer has repo-root/Python setup bypasses.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_composition_error_sweep_self_improvement.sh launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh launchers/self/run_self_improvement_mig_boundary_eval.sbatch launchers/self/run_self_improvement_qwen_no_growth.sbatch launchers/self/run_local_workshop_batch.sh launchers/self/run_refocused_self_improvement_local.sh`; `DRY_RUN=1 bash launchers/self/run_composition_error_sweep_self_improvement.sh`; `OUT_ROOT=/tmp/rl-a10-dryrun DRY_RUN=1 bash launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`; `BASE_OUT=/tmp/workshop-local-print bash launchers/self/run_local_workshop_batch.sh --print-only --gpus 0`; `BASE_OUT=/tmp/refocused-print bash launchers/self/run_refocused_self_improvement_local.sh --print-only --only-addition`; helper-bypass inventory command over `launchers/self/*.{sh,sbatch}` returned no files.

### Implementation Log: 2026-06-18 00:37:32 UTC

- Continued shrinking `self/core/driver.py` by separating its broad legacy import/export surface from its actual CLI and worker dependency wiring.
- Added `self/core/driver_compat_exports.py` for old helper/container names that legacy tests, notebooks, and `self.adaptive_candidate_training` users still import through the driver.
- Added `self/core/driver_compat_manifest.py` so `driver.py` can expose lazy `__getattr__`, `__dir__`, and `__all__` compatibility without eagerly importing the full legacy surface.
- Kept patch-sensitive globals such as `train_and_score_candidate`, `train_candidates_serial`, `train_candidates_slurm_array`, `train_candidates_local_parallel`, and `subprocess` directly on `driver.py`, preserving the existing monkeypatch behavior tested through `self.adaptive_candidate_training`.
- `self/core/driver.py` is now `336` lines, down from `437`, and primarily contains dependency factories plus the public CLI/worker wrappers.
- Updated `self/README.md` with the new driver compatibility modules and revised driver ownership.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_compat_exports.py self/core/driver_compat_manifest.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -q` (`37 passed`, `3` multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env python -c "from self import adaptive_candidate_training as loop; names=['build_exact_pair_addition_dataset','ConfigProposal','CandidateWorkItem','ExactPairDataset','proposal_grpo_reward','train_candidate_metrics','train_and_score_candidate','subprocess']; print({name: type(getattr(loop, name)).__name__ for name in names})"` returned the expected function/type/module mapping.

### Implementation Log: 2026-06-18 00:45:40 UTC

- Continued shrinking `self/core/driver.py` by moving dependency-factory construction out of the driver wrapper.
- Added `self/core/driver_default_bindings.py` for concrete runtime defaults that the driver exposes lazily, including old patch points such as `train_and_score_candidate`, `subprocess`, `_load_json`, and `_prepare_candidate_worker_specs`.
- Added `self/core/driver_wiring.py` for candidate-dispatch deps, worker-entrypoint deps, proposal-GRPO dispatch deps, adaptive run deps, and CLI entrypoint deps. The wiring reads through the live driver module, so monkeypatches on `self.adaptive_candidate_training` still override the bound functions.
- Kept `self/core/driver.py` as the public wrapper surface for the old module path. It is now `145` lines, down from `336` after the prior pass and `437` before the compatibility/export split.
- The first focused test run caught a missing `CandidateMetrics` compatibility export; restored it in `self/core/driver_compat_manifest.py` and `self/core/driver_compat_exports.py` before final verification.
- Updated `self/README.md` with the new driver wiring/default-binding modules and the revised remaining-cleanup status.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_wiring.py self/core/driver_default_bindings.py self/core/driver_compat_exports.py self/core/driver_compat_manifest.py self/adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -q` (`37 passed`, `3` multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env python -c "from self import adaptive_candidate_training as loop; names=['build_parser','CandidateMetrics','ConfigProposal','_load_json','_prepare_candidate_worker_specs','train_candidate_metrics','train_and_score_candidate','subprocess']; print({name: type(getattr(loop, name)).__name__ for name in names})"` returned the expected function/type/module mapping.

### Implementation Log: 2026-06-18 00:52:00 UTC

- Implemented a safe packed-local candidate-worker optimization in `self/core/candidate_worker_runtime.py`.
- Added a per-pack shared-input cache for candidate args normalization, task/config construction, source/eval example loading, proposal/outcome trace loading, and proposal prompt loading. Candidate-specific pseudo examples, seeds, `CandidateWorkItem` reconstruction, training, evaluation, and metric writing remain per candidate.
- Threaded the optional `shared_cache` keyword through `self/core/worker_entrypoints.py`, `self/core/driver_wiring.py`, and `self/core/driver.py` so packed workers can use the cache while one-candidate workers and old entry points remain compatible.
- The optimization deliberately does not reuse mutable trained model objects or cached model weights yet. Each candidate still calls `train_checkpoint(...)` from the same current checkpoint so candidate isolation and selection semantics remain unchanged.
- Added `test_candidate_pack_worker_reuses_shared_inputs` to verify that a two-candidate pack loads the shared trace buffers once while still scoring both candidates.
- Updated `self/README.md` runtime notes and remaining optimization notes to document the shared-input cache and the remaining model-reload limitation.
- Verification: `python -m py_compile self/core/candidate_worker_runtime.py self/core/worker_entrypoints.py self/core/driver_wiring.py self/core/driver.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -q` (`38 passed`, `3` multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py -q` (`2 passed`).

### Implementation Log: 2026-06-18 00:58:21 UTC

- Added an opt-in packed-local checkpoint bootstrap cache to reduce repeated source-checkpoint weight reads while preserving candidate isolation.
- Added `ModelBootstrapCache` in `self/core/model_io.py`. It caches tokenizers by checkpoint/tokenizer mode and, when `cache_base_state=True`, stores a CPU copy of the loaded source checkpoint state after the first load. Later candidates instantiate a fresh model object from config and `load_state_dict(...)` from the cached CPU tensors rather than calling `from_pretrained(...)` on the same checkpoint again.
- Added `--candidate-local-cache-base-state` in `self/core/args.py`. The flag is off by default because it trades disk IO for extra per-worker CPU memory, roughly one checkpoint copy per packed local worker.
- Threaded `model_bootstrap_cache` through `train_checkpoint(...)`, `train_and_score_candidate(...)`, and packed candidate shared inputs. The proposal-rehearsal phase is intentionally not cached because its source checkpoint is candidate-specific.
- Extended `test_candidate_pack_worker_reuses_shared_inputs` to verify that candidates in one pack receive the same bootstrap cache object when the flag is enabled.
- Added `tests/test_model_io_bootstrap_cache.py` to verify tokenizer reuse, cached-state reuse, and isolation from mutations to a previously returned model object.
- Updated `self/README.md` runtime notes and remaining optimization notes to document `--candidate-local-cache-base-state` and the CPU-memory tradeoff.
- Verification: `python -m py_compile self/core/model_io.py self/core/args.py self/core/candidate_scoring.py self/core/candidate_worker_runtime.py tests/test_model_io_bootstrap_cache.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_model_io_bootstrap_cache.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py -q` (`38 passed`, `3` multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_self_improvement_controller.py -q` (`2 passed`).

### Implementation Log: 2026-06-18 01:03:29 UTC

- Moved the shared recipe implementation from the top-level `self/self_improvement_recipe.py` module into `self/core/recipes.py`.
- Replaced `self/self_improvement_recipe.py` with a compatibility wrapper that reexports `self.core.recipes`, preserving old imports used by tests, notebooks, and legacy scripts.
- Updated current implementation imports in `self/core`, `self/experiments`, `self/diagnostics`, `self/self_improvement_core.py`, and `self/addition_recipe.py` to use `self.core.recipes` directly. Legacy scripts and tests still use the wrapper path where appropriate.
- Updated `self/README.md` with the new core recipe module, compatibility-wrapper mapping, and current import-boundary note.
- Verification: `python -m compileall -q self/core self/experiments self/diagnostics self/addition_recipe.py self/self_improvement_core.py self/self_improvement_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_bit_task_recipe.py tests/test_addition_recipe_diagnostic.py tests/test_multiplication_rectangular.py -q` (`27 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_model_io_bootstrap_cache.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`42 passed`, `3` multiprocessing fork warnings).

### Implementation Log: 2026-06-18 01:09:48 UTC

- Moved tokenizer implementation from top-level `self/task_tokenizer.py` into `self/core/tokenizers.py`.
- Replaced `self/task_tokenizer.py` with a compatibility wrapper that reexports `self.core.tokenizers`, preserving old imports used by legacy scripts and tests.
- Updated current core imports in `self/core/model_io.py` and `self/core/recipes.py` to use the canonical `self.core.tokenizers` path. The historical addition CLI intentionally keeps importing through the wrapper path as a compatibility check.
- While running the focused legacy checks, restored monkeypatch-compatible attribute forwarding in `self/self_improvement.py`, which wraps `self.legacy.addition_self_improvement`. This preserves old tests/notebooks that patch `self.self_improvement.instantiate_model_and_tokenizer`, `build_fixed_char_tokenizer`, or `load_model_for_tokenizer`.
- Updated `self/README.md` with the new core tokenizer module, compatibility-wrapper mapping, and current import-boundary note.
- Verification: `python -m compileall -q self/core self/task_tokenizer.py self/self_improvement.py self/legacy/addition_self_improvement.py tests/test_model_io_bootstrap_cache.py tests/test_bit_task_recipe.py tests/test_legacy_addition_self_improvement.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_model_io_bootstrap_cache.py tests/test_bit_task_recipe.py tests/test_addition_recipe_diagnostic.py tests/test_legacy_addition_self_improvement.py -q` (`36 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings).

### Implementation Log: 2026-06-18 01:19:16 UTC

- Converted `self/self_improvement_tasks.py` into an explicit compatibility facade over canonical `self/tasks/*` and `self/core/*` modules.
- Removed the facade's remaining import dependency on `self.self_improvement_core`; it now imports `JsonDict` and `SelfImprovementTask` from `self.core.task_protocols`, and `extract_numeric_answer` / `generate_prediction_map` from `self.core.evaluation`.
- Moved the public `corrupt_numeric_target` helper into `self/tasks/addition.py` and exported it through `self/tasks/__init__.py` and the old facade path.
- Verified by search that current implementation packages (`self/core`, `self/tasks`, `self/experiments`, `self/diagnostics`, and `self/analysis`) do not import through `self.self_improvement_tasks` or `self.self_improvement_core`.
- Updated `self/README.md` with the explicit task-facade status and the moved addition helper.
- Verification: `python -m compileall -q self/self_improvement_tasks.py self/tasks self/core tests/test_self_improvement_tasks.py tests/test_adaptive_candidate_training.py tests/test_run_length_balanced_eval.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_bit_task_recipe.py -q` (`47 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings).

### Implementation Log: 2026-06-18 01:26:52 UTC

- Started splitting the large run-length task adapter by extracting pure run-length state/target helpers into `self/tasks/run_length_logic.py`.
- Moved `compute_run_stats`, `compute_run_state`, `format_run_length_run_state`, `merge_run_state`, `leftmost_max_run_pair`, and `format_run_length_target` into the new logic module.
- Kept the old `self.tasks.run_length` and `self.self_improvement_tasks` import paths working by reexporting those helpers through the existing modules.
- Updated core adaptive composition and driver compatibility exports to import pure run-length helpers from `self.tasks.run_length_logic` directly. Updated run-length diagnostics similarly for `compute_run_stats`.
- Updated `self/README.md` with the new run-length logic module and remaining task-splitting status.
- Verification: `python -m compileall -q self/tasks/run_length.py self/tasks/run_length_logic.py self/tasks/__init__.py self/self_improvement_tasks.py self/core/composition.py self/core/driver_compat_exports.py self/diagnostics/run_length_balanced_eval.py self/diagnostics/evaluate_fixed_composition_slices.py tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_bit_task_recipe.py -q` (`47 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings); import sanity confirmed `self.tasks.run_length.compute_run_stats` is the canonical `self.tasks.run_length_logic.compute_run_stats` and facade reexports still work.

### Implementation Log: 2026-06-18 01:34:06 UTC

- Continued splitting the run-length adapter by extracting example/data helpers into `self/tasks/run_length_data.py`.
- Moved `RunLengthExample`, key encode/decode helpers, example generation, example composition, composed-dataset construction, bucketing, and override cloning out of `self/tasks/run_length.py`.
- Kept `self.tasks.run_length` and `self.self_improvement_tasks` compatibility paths working by reexporting the moved symbols.
- Updated current implementation imports in `self/core`, `self/diagnostics`, `self/tasks/__init__.py`, and `self/self_improvement_tasks.py` so data/example helpers come from `self.tasks.run_length_data`, while `RunLengthTask` remains owned by `self.tasks.run_length`.
- Updated `self/README.md` with the new run-length data module and revised run-length adapter ownership.
- Verification: `python -m compileall -q self/tasks/run_length.py self/tasks/run_length_data.py self/tasks/run_length_logic.py self/tasks/__init__.py self/self_improvement_tasks.py self/core/composition.py self/core/driver_compat_exports.py self/diagnostics/run_length_balanced_eval.py self/diagnostics/evaluate_fixed_composition_slices.py tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_bit_task_recipe.py -q` (`47 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings); import sanity confirmed `self.tasks.run_length` and `self.self_improvement_tasks` reexports point to the canonical `self.tasks.run_length_data` symbols.

### Implementation Log: 2026-06-18 01:44:21 UTC

- Extracted run-length round-target pseudolabel derivation into `self/tasks/run_length_pseudolabels.py`.
- Split the extracted logic into direct, guarded pair-output, run-state, and default tuple pseudolabel helpers. `RunLengthTask.derive_round_targets` now delegates to `derive_run_length_round_targets`.
- Preserved existing monkeypatch behavior by passing the `self.tasks.run_length.generate_prediction_map` wrapper into the extracted module; tests that patch `self.self_improvement_tasks.generate_prediction_map` still affect run-length derivation.
- Kept `RunLengthTask` in `self/tasks/run_length.py` as the orchestration layer. After the run-length logic/data/pseudolabel splits, the adapter is now `477` lines, down from `1172` before the sequence of run-length cleanups.
- Updated `self/README.md` with the new run-length pseudolabel module and revised run-length adapter ownership.
- Verification: `python -m compileall -q self/tasks/run_length.py self/tasks/run_length_pseudolabels.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_run_length_balanced_eval.py tests/test_bit_task_recipe.py -q` (`47 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings); import sanity confirmed `RunLengthTask` and the facade `generate_prediction_map` remain callable through old paths.

### Implementation Log: 2026-06-18 02:00:21 UTC

- Removed an out-of-scope auxiliary bit task from the runnable self-improvement repository because it is not part of the workshop/main-track experiment scope.
- Deleted the task module, the interrupted pseudolabel extraction, and the task-specific top-level/legacy CLI wrappers. Unlike earlier compatibility moves, the removed task's module entry point is intentionally no longer preserved so it does not stay discoverable.
- Removed the task exports from `self/tasks/__init__.py` and `self/self_improvement_tasks.py`, and removed task-specific parser/guard helpers from `self/tasks/bit_common.py`.
- Updated seed-fit, overfit diagnostics, Figure 2 schedule selection/retune helpers, local batch launchers, guarded bit diagnostics, and budget-grid launchers so supported/default tasks are now addition, run-length, and multiplication as appropriate.
- Updated launcher/task/schedule tests to use run-length where they were only exercising the shared bit-task path, and removed task-only test cases.
- Updated root `README.md` and `self/README.md` to remove obsolete commands/current module listings and to document the removal decision.
- Removed stale figure artifacts and schedule entries from the workshop and ICLR paper bundles under `icmlw26_comp-self-improvement` and `iclr27_comp-self-improvement`, and revised stale root workshop handoff/plan prose to use run-length as the clean transfer task.
- Verification: `python -m compileall -q self/tasks self/self_improvement_tasks.py self/experiments/seed_fit_experiment.py self/experiments/paper_schedule_selection.py self/experiments/figure2_condition_sweep.py self/experiments/figure2_paper_retune.py self/diagnostics/check_self_improvement_overfit.py self/diagnostics/analyze_symbolic_training_dynamics.py self/analysis/training_curve_notebook_utils.py self/analysis/seed_fit_curve_notebook_utils.py`; `bash -n launchers/self/run_seed_fit_experiment.sbatch launchers/self/submit_seed_fit_grid.sh launchers/self/run_task_self_improvement.sbatch launchers/self/submit_budget_grid_self_improvement.sh launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh launchers/self/run_refocused_self_improvement_local.sh launchers/self/run_local_workshop_batch.sh launchers/self/run_figure2_recipe_aggressive.sh launchers/self/submit_figure2_recipe_aggressive.sh`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py tests/test_self_improvement_launchers.py tests/test_seed_fit_launchers.py tests/test_task_self_improvement_launchers.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_paper_schedule_selection.py tests/test_figure2_condition_sweep.py tests/test_figure2_recipe_aggressive_launchers.py tests/test_training_curve_notebook_utils.py -q` (`69 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` multiprocessing fork warnings); `python -m json.tool` passed for both cleaned paper schedule JSON files; repository-local search found no current `self`/`tests`/`launchers` task imports or entry points for the removed task, and no task-named files remain under the workshop/ICLR paper bundles.

### Implementation Log: 2026-06-18 02:13:53 UTC

- Continued task-module cleanup by extracting multiplication example/data helpers from `self/tasks/multiplication.py` into `self/tasks/multiplication_data.py`.
- Moved `MultiplicationExample`, key encode/decode helpers, override cloning, random operand generation, seed/long dataset construction, partial-product analysis, component payload construction, and multiplication slice naming into the new data module.
- Kept old imports through `self.tasks.multiplication` working by importing and reexporting the moved names from the task adapter. Updated `self/tasks/__init__.py` and `self/self_improvement_tasks.py` so public helper exports point at the canonical data module, while `MultiplicationTask` remains owned by `self/tasks/multiplication.py`.
- `self/tasks/multiplication.py` is now `458` lines, down from `788`; `self/tasks/multiplication_data.py` owns `360` lines of example/data logic.
- Updated `self/README.md` with the new multiplication data module and revised task-splitting status.
- Verification: `python -m compileall -q self/tasks/multiplication.py self/tasks/multiplication_data.py self/tasks/__init__.py self/self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`36 passed`).

### Implementation Log: 2026-06-18 02:21:12 UTC

- Finished the removed-task cleanup pass by deleting stale ignored run/log/model artifacts under `artifacts/` and the remaining ignored figure PDFs in the workshop and ICLR paper bundles.
- Confirmed no removed-task-named files remain outside ignored/generated artifacts and historical plan-log prose. A targeted symbol search found no current task class, example, self-improvement wrapper, key helper, parser, builder, or compose entry points under `self`, `tests`, `launchers`, or current docs.
- Filesystem free-space check after cleanup showed `/scratch/gpfs` at `3.0T` available; the full repository-size walk was intentionally stopped because it was traversing too many experiment outputs for this pass.

### Implementation Log: 2026-06-18 02:30:25 UTC

- Continued task-module cleanup by extracting pure addition example/data helpers from `self/tasks/addition.py` into `self/tasks/addition_data.py`.
- The new module owns the canonical addition pipeline reexports, initial/composed/eval dataset preparation, boundary-carry status/slice helpers, and numeric-target corruption helper. `AdditionTask` remains in `self/tasks/addition.py` as the adapter and pseudolabel orchestration layer.
- Kept old imports through `self.tasks.addition` working by importing the moved symbols back into the adapter module. Updated `self/tasks/__init__.py` and `self/self_improvement_tasks.py` so public helper exports point at the canonical data module.
- `self/tasks/addition.py` is now `512` lines, down from `687`; `self/tasks/addition_data.py` owns `210` lines of addition data/preparation logic.

- Verification: `python -m py_compile self/tasks/addition.py self/tasks/addition_data.py self/tasks/__init__.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `python -m compileall -q self/tasks self/self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`36 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` existing multiprocessing fork warnings); import sanity confirmed old `self.tasks.addition` and facade exports point to the canonical `self.tasks.addition_data` symbols.

### Implementation Log: 2026-06-18 02:38:14 UTC

- Continued the addition task split by extracting direct/compose/compose-corrupt pseudolabel derivation into `self/tasks/addition_pseudolabels.py`.
- `AdditionTask.derive_round_targets` now delegates to `derive_addition_round_targets(...)` while still passing the adapter-level compatibility functions for `generate_prediction_map` and `build_composed_pseudo_map`; old monkeypatch paths through `self.self_improvement_tasks` remain active.
- `self/tasks/addition.py` is now `370` lines, down from `512` after the data-helper split and `687` before addition splitting began. `self/tasks/addition_pseudolabels.py` owns `192` lines of pseudolabel/diagnostic logic.
- Verification: `python -m py_compile self/tasks/addition.py self/tasks/addition_data.py self/tasks/addition_pseudolabels.py self/tasks/__init__.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `python -m compileall -q self/tasks self/self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`36 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 02:45:09 UTC

- Continued proposal-runtime cleanup by extracting prompt/task-metadata helpers from `self/core/proposal_runtime.py` into `self/core/proposal_prompts.py`.
- The new module owns task target-format descriptions, example component-prediction strings, sandbox validation-case selection, default executable source-pair selection, and executable program/policy/meta prompt rendering.
- Kept old imports through `self.core.proposal_runtime` working by reimporting and reexporting the moved helper names. Compatibility identity checks confirmed `target_format_for_task`, `component_prediction_examples_for_task`, `program_validation_cases`, `choose_default_program_pair`, and `render_program_candidate_prompt` resolve to the canonical `self.core.proposal_prompts` functions through the old module path.
- `self/core/proposal_runtime.py` is now `567` lines, down from `733`; `self/core/proposal_prompts.py` owns `178` lines of prompt/task-metadata logic.
- Verification: `python -m py_compile self/core/proposal_runtime.py self/core/proposal_prompts.py self/core/driver_compat_exports.py self/core/driver_default_bindings.py self/core/attempt_prompt_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py`; `python -m compileall -q self/core`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py -q` (`45 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 02:50:19 UTC

- Continued proposal-runtime cleanup by extracting config proposal row validation from `self/core/proposal_runtime.py` into `self/core/proposal_config_validation.py`.
- The new module owns raw config row output extraction, plain/action-prediction proposal payload normalization, schema validation, duplicate marking, repeat-target annotation, and sanitized validation result construction.
- Kept old imports through `self.core.proposal_runtime` working by reimporting `_raw_output` and `validate_config_rows`; compatibility identity checks confirmed both names resolve to the canonical `self.core.proposal_config_validation` functions through the old module path.
- `self/core/proposal_runtime.py` is now `445` lines, down from `567` after the prompt split and `733` before the proposal-runtime cleanup sequence. `self/core/proposal_config_validation.py` owns `136` lines of config-validation logic.
- Verification: `python -m py_compile self/core/proposal_runtime.py self/core/proposal_config_validation.py self/core/driver_compat_exports.py tests/test_adaptive_candidate_training.py`; `python -m compileall -q self/core`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_self_improvement_controller.py -q` (`45 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 02:56:21 UTC

- Started trimming `self/core/nonadaptive_loop.py` by extracting deterministic size/frontier arithmetic into `self/core/nonadaptive_schedule.py`.
- The new module owns `NonAdaptiveSizeSchedule`, `normalize_frontier_min_size(...)`, and `build_nonadaptive_size_schedule(...)`, covering both the legacy contiguous expansion schedule and explicit `frontier_min_size` schedules.
- `run_self_improvement(...)` now delegates final/composed size computation plus per-round current/target max-size computation to the schedule object, preserving the existing behavior that round 0 remains at `initial_max_size` when an explicit frontier is used.
- Added `tests/test_nonadaptive_schedule.py` for legacy schedule arithmetic, explicit frontier arithmetic, zero-expand frontier behavior, and frontier validation.
- `self/core/nonadaptive_loop.py` is now `739` lines, down from `759`; `self/core/nonadaptive_schedule.py` owns `57` lines.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_schedule.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py -q` (`8 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`44 passed`).

### Implementation Log: 2026-06-18 03:06:38 UTC

- Continued trimming `self/core/nonadaptive_loop.py` by extracting argument preflight, save-model policy resolution, recipe default normalization, dynamic/static composed-mode setup, and reset-each-round setup into `self/core/nonadaptive_setup.py`.
- `run_self_improvement(...)` now receives a `NonAdaptiveRunSetup` object and then uses the same local variable names as before for the dataset/resume/training loop. The call passes loop-level `resolve_save_model_policy`, `recipe_enabled`, and `resolve_self_improvement_recipe` bindings into the setup helper so old `self.self_improvement_core` monkeypatch behavior stays compatible.
- Added `tests/test_nonadaptive_setup.py` for resume/stop validation, save-model policy side effects, CUDA bf16 defaulting, recipe tokenizer warning/default batch sizes, and explicit recipe batch-size preservation.
- `self/core/nonadaptive_loop.py` is now `710` lines, down from `739`; `self/core/nonadaptive_setup.py` owns `106` lines of preflight/setup logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_setup.py self/core/nonadaptive_schedule.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`9 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`49 passed`).

### Implementation Log: 2026-06-18 03:12:47 UTC

- Continued trimming `self/core/nonadaptive_loop.py` by extracting non-adaptive output/data artifact path construction, reset-each-round directory setup, metadata/result loading, metadata persistence, stored-metadata lookup, and `config_args.json` writing into `self/core/nonadaptive_state.py`.
- Kept the old compatibility surface intact by passing the loop-level `json`, `ensure_dir`, `load_summary_records`, `encode_rng_state`, and `sanitize_json_value` bindings into the extracted state helpers. This preserves old `self.self_improvement_core` monkeypatch behavior while moving ownership out of the loop.
- Added `tests/test_nonadaptive_state.py` for reset path creation, resume metadata/result loading, stored metadata fallback lookup, RNG-state metadata persistence, and config-args sanitization.
- `self/core/nonadaptive_loop.py` is now `709` lines; `self/core/nonadaptive_state.py` owns `122` lines of artifact/run-state logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_state.py self/core/nonadaptive_setup.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`13 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`53 passed`).

### Implementation Log: 2026-06-18 03:17:21 UTC

- Continued non-adaptive cleanup by moving loaded-dataset metadata compatibility validation from the resume branch in `self/core/nonadaptive_loop.py` into `validate_loaded_nonadaptive_metadata(...)` in `self/core/nonadaptive_state.py`.
- The extracted validator owns task-name checks, initial/frontier/composed size compatibility, reset-each-round compatibility, dynamic/static composed refresh compatibility, composed-eval density checks, and the final task-specific `validate_loaded_metadata(...)` call.
- Extended `tests/test_nonadaptive_state.py` with direct validator coverage for a matching metadata payload, task mismatch, and frontier mismatch.
- `self/core/nonadaptive_loop.py` is now `657` lines, down from `709`; `self/core/nonadaptive_state.py` owns `196` lines of artifact/run-state and loaded-metadata validation logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_state.py tests/test_nonadaptive_state.py tests/test_bit_task_seed_round_zero.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`16 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`56 passed`).

### Implementation Log: 2026-06-18 03:24:29 UTC

- Continued non-adaptive cleanup by extracting dataset generation/loading from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_datasets.py`.
- The new helper owns reserved shared-eval seeding, initial split generation/saving, initial composed-pool generation/saving, composed-eval generation/saving, final eval construction, generation metadata creation, persisted dataset loading, loaded composed-eval warnings, and base-record rebuilding.
- The loop still passes patchable `save_examples` and `load_examples` bindings into the helper, plus callback hooks for metadata persistence and `config_args.json` writing, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_datasets.py` for a new-run generation path with reserved eval behavior and an existing-run loading path with validation callback coverage.
- `self/core/nonadaptive_loop.py` is now `555` lines, down from `657`; `self/core/nonadaptive_datasets.py` owns `217` lines of dataset generation/loading logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_datasets.py self/core/nonadaptive_state.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`18 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`58 passed`).

### Implementation Log: 2026-06-18 03:30:28 UTC

- Continued non-adaptive cleanup by extracting resume/model/bootstrap setup from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_bootstrap.py`.
- The new helper owns resume-round resolution, completed-summary pruning, checkpoint path selection, missing-checkpoint errors, model/tokenizer instantiation, `TrainingConfig` construction, decode-budget derivation, collator selection, summary-record initialization, and resumed pseudo-seed loading.
- The loop still passes patchable `instantiate_model_and_tokenizer`, `TrainingConfig`, collator classes, `resolve_max_new_tokens`, and `load_examples` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_bootstrap.py` for initial model/config/collator setup and resume-from-round checkpoint/pseudo-seed loading.
- `self/core/nonadaptive_loop.py` is now `508` lines, down from `555`; `self/core/nonadaptive_bootstrap.py` owns `151` lines of resume/bootstrap logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_bootstrap.py tests/test_nonadaptive_bootstrap.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`20 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`60 passed`).

### Implementation Log: 2026-06-18 03:35:47 UTC

- Continued non-adaptive cleanup by extracting per-round training setup/execution from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_training.py`.
- The new helper owns recipe phase-name resolution, self-improvement recipe phase overrides, seed-round skip handling, `TokenizedPromptTargetDataset` construction, `make_training_args(...)`, `build_trainer(...)`, trainer execution, and model/tokenizer save handling.
- The loop still passes patchable `TokenizedPromptTargetDataset`, `make_training_args`, and `build_trainer` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_training.py` for the seed-round skip branch and recipe self-improvement override/trainer construction branch.
- `self/core/nonadaptive_loop.py` is now `457` lines, down from `508`; `self/core/nonadaptive_training.py` owns `136` lines of per-round training logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_training.py tests/test_nonadaptive_training.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`22 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`62 passed`).

### Implementation Log: 2026-06-18 03:40:30 UTC

- Continued non-adaptive cleanup by extracting base/composed round evaluation from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_evaluation.py`.
- The new helper owns base held-out evaluation, composed eval slice evaluation, guarded-slice debug sample writing, per-slice `SliceMetric` construction, and aggregate composed-eval accuracy calculation.
- The loop still passes patchable `evaluate_accuracy_with_breakdown`, `write_prediction_debug_samples`, and `SliceMetric` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_evaluation.py` for composed-slice aggregation, empty/nan slice handling, and guarded-slice debug sample dispatch.
- `self/core/nonadaptive_loop.py` is now `425` lines, down from `457`; `self/core/nonadaptive_evaluation.py` owns `93` lines of round evaluation logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_evaluation.py tests/test_nonadaptive_evaluation.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`24 passed`); `python -m compileall -q self/core self/self_improvement_core.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py tests/test_bit_task_seed_round_zero.py tests/test_self_improvement_tasks.py tests/test_bit_task_recipe.py -q` (`64 passed`).

### Implementation Log: 2026-06-18 03:50:49 UTC

- Continued non-adaptive cleanup by extracting dynamic composed-pool refresh and next-round pseudo-label generation from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_pseudo.py`.
- The new helper owns final-round pseudo-generation skipping, dynamic composed-pool rebuilding, persistent and per-round composed-pool snapshots, metadata refresh labels, metadata persistence before pseudo-labeling, pseudo decode-budget resolution, `pseudo_label_mode=none` diagnostics, task-specific pseudo-target derivation, `bit_composition_path_mode` diagnostics, next-round pseudo JSONL writing, and pseudo-label warning messages.
- The loop still passes patchable `save_examples`, `resolve_max_new_tokens`, and `random.Random` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior while keeping the loop focused on orchestration and summary writing.
- Added `tests/test_nonadaptive_pseudo.py` for final-round side-effect skipping, dynamic refresh plus target derivation, composed-pool/component-map snapshot paths, metadata persistence, decode-budget handoff, warning output, and `pseudo_label_mode=none` empty-stat behavior.
- `self/core/nonadaptive_loop.py` is now `388` lines, down from `425`; `self/core/nonadaptive_pseudo.py` owns `135` lines of pseudo-label refresh/generation logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_pseudo.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`27 passed`).

### Implementation Log: 2026-06-18 03:56:06 UTC

- Continued non-adaptive cleanup by extracting round-summary construction and persistence from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_results.py`.
- The new helper owns `RoundSummary` construction, console summary dispatch, payload conversion, `save_model_policy` / `model_dir` annotation, per-round `metrics.json` writing, in-memory summary-record updates, and run-level `self_improvement_results.json` rewriting.
- The loop still passes patchable `RoundSummary`, `summarize_round`, `summary_to_payload`, `write_summary_records`, and `json` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_results.py` for real metrics/results artifact writing and injected summary/payload/write bindings.
- `self/core/nonadaptive_loop.py` is now `382` lines, down from `388`; `self/core/nonadaptive_results.py` owns `62` lines of summary persistence logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_results.py tests/test_nonadaptive_results.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`29 passed`).

### Implementation Log: 2026-06-18 04:02:15 UTC

- Continued non-adaptive cleanup by extracting post-round stop/final/reset lifecycle handling from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_lifecycle.py`.
- The new helper owns stop-after-round break decisions, final-round continue decisions, trainer release, CUDA cache clearing, reset-each-round model release, recipe scratch reset, recipe checkpoint reload, non-recipe base-model reload, and missing reset-checkpoint errors.
- To preserve the old memory behavior, the loop now passes a mutable `NonAdaptiveRoundResources` slot and clears its direct `model` / `trainer` references before invoking the lifecycle helper. This lets the helper release the old model before loading a replacement, rather than increasing peak memory during reset-each-round runs.
- The loop still passes patchable `Path`, `torch.cuda.is_available`, `torch.cuda.empty_cache`, `instantiate_recipe_model`, `load_recipe_model`, and `load_model_for_tokenizer` bindings into the helper, preserving old `self.self_improvement_core` monkeypatch behavior.
- Added `tests/test_nonadaptive_lifecycle.py` for stop-after-round cleanup, final-round cleanup, recipe scratch reset ordering, non-recipe reset reloads, and missing recipe checkpoint failure after resource release.
- `self/core/nonadaptive_loop.py` is now `371` lines, down from `382`; `self/core/nonadaptive_lifecycle.py` owns `112` lines of post-round lifecycle logic.
- Verification: `python -m py_compile self/core/nonadaptive_loop.py self/core/nonadaptive_lifecycle.py tests/test_nonadaptive_lifecycle.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_state.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_schedule.py -q` (`34 passed`).

### Implementation Log: 2026-06-18 04:09:08 UTC

- Continued adaptive cleanup by extracting local and packed-local candidate worker process scheduling from `self/core/candidate_workers.py` into `self/core/candidate_local_workers.py`.
- The new helper owns local dispatch log directories, process packing, `self.core.driver` worker commands, concurrency throttling, `local_dispatch.json`, local worker progress reporting, timeout termination/failure recording, final active-process cleanup, local worker failure artifacts, and the shared sanitized JSON writer used by candidate-worker dispatch.
- `self/core/candidate_workers.py` still owns candidate spec/pack-spec construction and Slurm-array dispatch, and reexports `candidate_metric_path`, `candidate_worker_failure_path`, `write_json`, and `write_local_candidate_failure` through the old module path. Its `train_candidates_local_parallel(...)` wrapper now delegates to `train_candidates_local_parallel_from_specs(...)` while passing the module-level `subprocess` binding, preserving old monkeypatch behavior used by tests and driver compatibility wrappers.
- `self/core/candidate_workers.py` is now `341` lines, down from `526`; `self/core/candidate_local_workers.py` owns `240` lines of local/packed-local scheduling logic.
- Verification: `python -m py_compile self/core/candidate_workers.py self/core/candidate_local_workers.py tests/test_adaptive_candidate_training.py`; `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_respect_concurrency_cap tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_can_pack_processes tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_worker_failure_becomes_metric tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs -q` (`4 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` (`45 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 04:16:40 UTC

- Continued adaptive cleanup by extracting Slurm-array candidate worker submission/polling from `self/core/candidate_workers.py` into `self/core/candidate_slurm_workers.py`.
- The new helper owns candidate-array submission, Slurm array-spec construction and throttling, `slurm_dispatch.json`, candidate-array progress polling, timeout cancellation, `slurm_timeout.json`, and post-array metric collection handoff.
- `self/core/candidate_workers.py` now focuses on candidate spec/pack-spec construction plus compatibility wrappers/reexports. Its old `submit_candidate_array(...)`, `wait_for_candidate_array(...)`, and `train_candidates_slurm_array(...)` names remain available and pass module-level `submit_sbatch`, `cancel_job`, `slurm_job_active`, `sys`, and `time` bindings into the extracted implementation, preserving old monkeypatch behavior.
- Added `tests/test_candidate_slurm_workers.py` for the old submit wrapper's module-level Slurm binding, timeout cancellation/timeout artifact writing, and submit/wait/collect sequencing through the extracted Slurm dispatch helper.
- `self/core/candidate_workers.py` is now `301` lines, down from `341`; `self/core/candidate_slurm_workers.py` owns `140` lines of Slurm-array dispatch logic.
- Verification: `python -m py_compile self/core/candidate_workers.py self/core/candidate_slurm_workers.py tests/test_candidate_slurm_workers.py`; `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py::test_slurm_array_mode_dispatches_even_single_candidate tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_respect_concurrency_cap tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_can_pack_processes tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_worker_failure_becomes_metric -q` (`7 passed`); `PYTHONPATH=. conda run -n torch-env pytest tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` (`48 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 04:23:53 UTC

- Continued adaptive cleanup by extracting candidate worker shared-input spec generation and pack-spec generation from `self/core/candidate_workers.py` into `self/core/candidate_worker_specs.py`.
- The new helper owns shared candidate input JSONL/trace/prompt artifacts, per-candidate worker spec JSON payloads, spec manifests, pack-spec chunking, pack manifests, and candidate metrics/failure path helpers.
- `self/core/candidate_workers.py` is now a thin compatibility wrapper/reexport surface over the canonical spec, local-worker, and Slurm-worker modules. Old imports such as `prepare_candidate_worker_specs`, `prepare_candidate_worker_pack_specs`, `candidate_metric_path`, `candidate_worker_failure_path`, `write_json`, and `write_local_candidate_failure` still resolve through `self.core.candidate_workers`.
- Added `tests/test_candidate_worker_specs.py` for worker spec input/spec/manifest payloads, pack-spec chunking and manifests, invalid pack sizes, and canonical helper identity through the old `candidate_workers` module path.
- `self/core/candidate_workers.py` is now `172` lines, down from `301`; `self/core/candidate_worker_specs.py` owns `153` lines of worker-spec artifact logic.
- Verification: `python -m py_compile self/core/candidate_workers.py self/core/candidate_worker_specs.py tests/test_candidate_worker_specs.py`; `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_can_pack_processes -q` (`10 passed`); a first broad adaptive run hit pytest `/tmp` temp-directory exhaustion before test execution, then `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_specs tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` passed (`52 passed`, `7` existing multiprocessing fork warnings). The temporary `.pytest_tmp_adaptive_specs` directory was removed after verification.

### Implementation Log: 2026-06-18 04:33:26 UTC

- Redacted stale historical references to the removed auxiliary bit task from this plan log so repository search does not keep surfacing an out-of-scope task label.
- Confirmed the tracked repository has no remaining removed-task source files, entry points, exports, or tracked filenames. The tracked `git grep` and filename checks return no occurrences for the removed task name.
- Left untracked notebooks, paper bundles, and external writing/reference directories alone because they are not part of `main` and may contain user-local or generated material.

### Implementation Log: 2026-06-18 04:38:01 UTC

- Continued adaptive candidate-worker cleanup by extracting candidate work-item payload serialization/deserialization from `self/core/candidate_execution.py` and `self/core/candidate_worker_runtime.py` into `self/core/candidate_worker_payloads.py`.
- The new helper owns controller handoff payloads, worker-spec candidate blocks, pseudo-example JSONL loading for handoff payloads, composed-key JSON conversion, and reconstruction of `CandidateWorkItem` shells for worker execution.
- `self/core/candidate_execution.py` now reexports `work_item_to_worker_payload` and `work_item_from_worker_payload` from the new module, preserving old imports used by driver bindings and notebooks. `self/core/candidate_worker_specs.py` writes candidate spec blocks through the same helper, and `self/core/candidate_worker_runtime.py` reconstructs candidate items through the shared path.
- Added `tests/test_candidate_worker_payloads.py` for controller payload fields, pseudo-example/composed-key round-trip loading, worker-spec candidate payload round-tripping, and old `candidate_execution` reexport identity.
- Verification: `python -m py_compile self/core/candidate_execution.py self/core/candidate_worker_payloads.py self/core/candidate_worker_runtime.py self/core/candidate_worker_specs.py tests/test_candidate_worker_payloads.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_payloads tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs -q` (`10 passed`); `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_payloads tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` (`56 passed`, `7` existing multiprocessing fork warnings). The temporary `.pytest_tmp_payloads` directory was removed after verification.

### Implementation Log: 2026-06-18 04:46:55 UTC

- Continued adaptive candidate-worker cleanup by extracting worker metric loading and missing-metric recovery from `self/core/candidate_execution.py` into `self/core/candidate_metric_collection.py`.
- The new helper owns failed-candidate metric construction, candidate metric JSON loading, worker-failure reason handling, fallback metrics when workers finish without `candidate_metrics.json`, and `candidate_jobs/gather_failures.json` writing.
- `self/core/candidate_execution.py` now reexports `candidate_failure_metrics` and `collect_candidate_array_metrics` from the new module, preserving driver-level wrapper and notebook imports. The dispatch entrypoint still injects the old monkeypatchable failure-metric function into collection, so behavior stays compatible.
- Added `tests/test_candidate_metric_collection.py` for failure metric fields, existing metric loading, failure-metric/gather-manifest writing, and old `candidate_execution` reexport identity.
- Verification: `python -m py_compile self/core/candidate_execution.py self/core/candidate_metric_collection.py tests/test_candidate_metric_collection.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_metrics tests/test_candidate_metric_collection.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_worker_failure_becomes_metric tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs -q` (`9 passed`); `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_candidate_metric_collection.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_metrics tests/test_candidate_metric_collection.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` (`60 passed`, `7` existing multiprocessing fork warnings). After final formatting, the broad compile and pytest command was repeated with `--basetemp=.pytest_tmp_metrics_final` and passed again (`60 passed`, `7` existing multiprocessing fork warnings). The temporary `.pytest_tmp_metrics` and `.pytest_tmp_metrics_final` directories were removed after verification.

### Implementation Log: 2026-06-18 04:56:47 UTC

- Finished the `self/core/candidate_execution.py` cleanup by moving serial candidate execution and local/Slurm dispatch glue into `self/core/candidate_dispatch_runtime.py`.
- `self/core/candidate_execution.py` is now a compatibility reexport surface for payload conversion, metric collection, and dispatch helpers. Old imports such as `self.core.candidate_execution.train_candidates_serial` and `collect_candidate_array_metrics` still resolve, but implementation ownership is split across focused modules.
- Added `tests/test_candidate_dispatch_runtime.py` for serial seed routing, local subprocess binding/delegation, Slurm delegation, and old `candidate_execution` reexport identity.
- Verification: `python -m py_compile self/core/candidate_execution.py self/core/candidate_dispatch_runtime.py tests/test_candidate_dispatch_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_dispatch tests/test_candidate_dispatch_runtime.py tests/test_candidate_metric_collection.py tests/test_candidate_worker_payloads.py -q` (`12 passed`); `python -m compileall -q self/core self/adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_metric_collection.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_dispatch tests/test_candidate_dispatch_runtime.py tests/test_candidate_metric_collection.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_slurm_workers.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py -q` (`64 passed`, `7` existing multiprocessing fork warnings). After final test formatting, the broad compile and pytest command was repeated and passed again (`64 passed`, `7` existing multiprocessing fork warnings). The temporary `.pytest_tmp_dispatch` directory was removed after verification.

### Implementation Log: 2026-06-18 05:04:11 UTC

- Extended `self/analysis/artifacts.py` with `adaptive_proposal_grpo_records(...)`, a stable notebook-facing flattener for `attempt_*/proposal_grpo/proposal_grpo_metrics.json`.
- The new helper adds the same run/attempt context as the existing adaptive attempt/proposal/candidate record helpers, avoiding notebook-side direct path parsing for proposal-GRPO diagnostics.
- Updated `tests/test_analysis_artifacts.py` to cover proposal-GRPO metric record loading from the synthetic adaptive-run fixture.
- Verification: `python -m py_compile self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest tests/test_analysis_artifacts.py -q` (`2 passed`); `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; an initial broader pytest run hit pytest `/tmp` numbered-directory exhaustion during fixture setup, then `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` passed (`6 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:12:35 UTC

- Extended `self/analysis/artifacts.py` with `adaptive_candidate_per_size_records(...)`, a notebook-facing flattener for adaptive candidate `per_size_accuracy` maps.
- The helper carries run, attempt, candidate, selected-candidate, reward, and proposal fields so adaptive heatmaps can use stable records instead of direct `candidate_metrics.json` parsing.
- Updated `tests/test_analysis_artifacts.py` to cover candidate per-size record loading from the synthetic adaptive-run fixture.
- Updated `self/README.md` to document adaptive candidate per-size records in the current analysis module map.
- Verification: `python -m py_compile self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`6 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:17:14 UTC

- Extended `self/analysis/artifacts.py` with `adaptive_selected_per_size_timeline_records(...)`, a stable notebook-facing helper for selected-checkpoint heatmaps.
- The helper starts from `round_00/metrics.json`, updates the checkpoint state on attempts with a selected candidate carrying `per_size_accuracy`, and carries that state forward through no-selection attempts.
- Added selected-payload fallback handling across `selected_candidate.json`, `round_summary.selected`, and `attempt_summary.selected`, preserving compatibility with older adaptive artifacts.
- Updated `tests/test_analysis_artifacts.py` with a two-attempt fixture covering selected-candidate update and no-selection carry-forward behavior.
- Updated `self/README.md` to document selected-checkpoint per-size timelines in the current analysis module map.
- Verification: `python -m py_compile self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`7 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:23:03 UTC

- Split the analysis artifact layer so adaptive-run loading and flattening now live in `self/analysis/adaptive_artifacts.py`.
- Added `self/analysis/artifact_io.py` for shared JSON/JSONL readers, artifact filename constants, and natural path sorting.
- Kept `self/analysis/artifacts.py` as the old notebook-facing compatibility surface while retaining non-adaptive `load_self_improvement_rounds(...)`, `per_size_accuracy_records(...)`, and `records_to_dataframe(...)` there.
- Added compatibility assertions in `tests/test_analysis_artifacts.py` so adaptive helpers imported from `self.analysis.artifacts` remain identical to the canonical `self.analysis.adaptive_artifacts` functions.
- Updated `self/README.md` with the new analysis module ownership.
- Verification: `python -m py_compile self/analysis/artifact_io.py self/analysis/adaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`7 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:27:58 UTC

- Split non-adaptive analysis result helpers from `self/analysis/artifacts.py` into `self/analysis/nonadaptive_artifacts.py`.
- The new module owns `load_self_improvement_rounds(...)`, `per_size_accuracy_records(...)`, and `records_to_dataframe(...)`.
- Kept the old `self.analysis.artifacts` imports as compatibility reexports, reducing that file to a thin notebook-facing import surface.
- Added compatibility assertions in `tests/test_analysis_artifacts.py` so non-adaptive helpers imported from `self.analysis.artifacts` remain identical to the canonical `self.analysis.nonadaptive_artifacts` functions.
- Updated `self/README.md` with the new non-adaptive analysis module ownership.
- Verification: `python -m py_compile self/analysis/artifact_io.py self/analysis/adaptive_artifacts.py self/analysis/nonadaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`7 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:32:16 UTC

- Split direct `self_improvement_results.json` frame construction from `self/analysis/training_curve_notebook_utils.py` into `self/analysis/training_curve_results.py`.
- The new module owns `resolve_results_path(...)`, `load_round_payload(...)`, `round_summary_frame(...)`, and `per_size_accuracy_frame_from_results(...)`.
- Kept the old `self.analysis.training_curve_notebook_utils` and top-level `self.training_curve_notebook_utils` imports working by importing the canonical helpers back into the notebook utility module.
- Added a compatibility assertion in `tests/test_training_curve_notebook_utils.py` so the top-level wrapper exposes the canonical `per_size_accuracy_frame_from_results(...)` function.
- Updated `self/README.md` with the new training-curve result-frame module ownership.
- Verification: `python -m py_compile self/analysis/training_curve_results.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`7 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:36:06 UTC

- Split training-curve plot styling and labels from `self/analysis/training_curve_notebook_utils.py` into `self/analysis/training_curve_style.py`.
- The new module owns `BASELINE_COLORS`, `BUDGET_ORDER`, `MODE_ORDER`, `configure_plot_style(...)`, and `mode_label(...)`.
- Kept the old `self.analysis.training_curve_notebook_utils` and top-level `self.training_curve_notebook_utils` imports working by importing the canonical style helpers back into the notebook utility module.
- Added a compatibility assertion in `tests/test_training_curve_notebook_utils.py` so the top-level wrapper exposes the canonical `configure_plot_style(...)` function.
- Updated `self/README.md` with the new training-curve style module ownership.
- Verification: `python -m py_compile self/analysis/training_curve_style.py self/analysis/training_curve_results.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`7 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:41:56 UTC

- Split Slurm training-log parsing and round-level metric loading from `self/analysis/training_curve_notebook_utils.py` into `self/analysis/training_curve_logs.py`.
- The new module owns `ROUND_PATTERN`, `_to_float(...)`, `parse_training_log(...)`, and `load_round_metrics(...)`.
- Kept the old `self.analysis.training_curve_notebook_utils` and top-level `self.training_curve_notebook_utils` imports working by importing the canonical log helpers back into the notebook utility module.
- Added a focused parser/metric fixture in `tests/test_training_curve_notebook_utils.py`, plus compatibility assertions for `parse_training_log(...)` and `load_round_metrics(...)`.
- Updated `self/README.md` with the new training-curve log parser module ownership.
- Verification: `python -m py_compile self/analysis/training_curve_logs.py self/analysis/training_curve_style.py self/analysis/training_curve_results.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`8 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:54:20 UTC

- Split submission-table loading, curve bundle assembly, bundle summaries, and bundle per-size frame construction from `self/analysis/training_curve_notebook_utils.py` into `self/analysis/training_curve_bundle.py`.
- Kept the old `self.analysis.training_curve_notebook_utils` and top-level `self.training_curve_notebook_utils` imports working by importing `CurveBundle`, `load_submission_jobs(...)`, `load_curve_bundle(...)`, `get_job_record(...)`, `per_size_accuracy_frame(...)`, and `build_run_summary(...)` back into the notebook utility module.
- Added a realistic TSV/log/results fixture in `tests/test_training_curve_notebook_utils.py` to cover bundle loading, train/validation log attachment, round metrics, run summaries, per-size accuracy expansion, and compatibility identities.
- Updated `self/README.md` with the new training-curve bundle module ownership and narrowed `training_curve_notebook_utils.py` to plotting plus compatibility imports.
- Verification: `python -m py_compile self/analysis/training_curve_bundle.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`9 passed`). The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 05:58:14 UTC

- Split task curve plots, per-size heatmaps, comparison curves, sparse annotation helpers, and figure export from `self/analysis/training_curve_notebook_utils.py` into `self/analysis/training_curve_plots.py`.
- Replaced `self/analysis/training_curve_notebook_utils.py` with a compatibility import surface over the canonical bundle, log, result, style, and plotting modules, preserving old notebook imports and the top-level `self/training_curve_notebook_utils.py` wrapper.
- Added a compatibility assertion in `tests/test_training_curve_notebook_utils.py` so `plot_per_size_accuracy_heatmap_from_results(...)` imported through the old top-level path is the canonical plotting helper.
- Updated `self/README.md` with the new training-curve plotting module ownership.
- Verification: `python -m py_compile self/analysis/training_curve_plots.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`9 passed`). Compatibility sanity confirmed `self.training_curve_notebook_utils.plot_per_size_accuracy_heatmap_from_results` is the canonical `self.analysis.training_curve_plots` helper. The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 06:02:50 UTC

- Split seed-fit result loading, flattened train/validation log construction, task summaries, and threshold-budget selection from `self/analysis/seed_fit_curve_notebook_utils.py` into `self/analysis/seed_fit_bundle.py`.
- Split seed-fit plotting defaults, loss curves, and budget-sweep curves into `self/analysis/seed_fit_plots.py`.
- Replaced `self/analysis/seed_fit_curve_notebook_utils.py` with a compatibility import surface over the canonical seed-fit modules, preserving the top-level `self/seed_fit_curve_notebook_utils.py` wrapper.
- Added `tests/test_seed_fit_curve_notebook_utils.py` with synthetic `seed_fit_results.json` fixtures covering bundle loading, log flattening, task summaries, threshold selection, plot rendering, and compatibility identities.
- Updated `self/README.md` with the new seed-fit analysis module ownership.
- Verification: `python -m py_compile self/analysis/seed_fit_bundle.py self/analysis/seed_fit_plots.py self/analysis/seed_fit_curve_notebook_utils.py self/seed_fit_curve_notebook_utils.py tests/test_seed_fit_curve_notebook_utils.py`; `python -m compileall -q self/analysis self/seed_fit_curve_notebook_utils.py tests/test_seed_fit_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis tests/test_seed_fit_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`11 passed`). Compatibility sanity confirmed top-level seed-fit bundle and plot helpers are the canonical analysis helpers. The temporary `.pytest_tmp_analysis` directory was removed after verification.

### Implementation Log: 2026-06-18 06:07:58 UTC

- Extracted per-round directory planning, save-model policy resolution, resume-skip detection, and round training/pseudo-example artifact persistence from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_round_setup.py`.
- `self/core/nonadaptive_loop.py` still orchestrates the non-adaptive round sequence, but now delegates round-local path/save/input setup through `prepare_nonadaptive_round_plan(...)` and `prepare_nonadaptive_round_training_data(...)`.
- Added `tests/test_nonadaptive_round_setup.py` for directory creation, final-round save-policy behavior, resume skip marking, train-example ordering, pseudo-example counting, and persisted artifact payloads.
- Updated `self/README.md` with the new non-adaptive round setup module ownership and revised the remaining cleanup queue.
- Verification: `python -m py_compile self/core/nonadaptive_round_setup.py self/core/nonadaptive_loop.py tests/test_nonadaptive_round_setup.py`; `python -m compileall -q self/core tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_schedule.py tests/test_nonadaptive_state.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py -q` (`32 passed`). The temporary `.pytest_tmp_nonadaptive` directory was removed after verification.

### Implementation Log: 2026-06-18 06:12:16 UTC

- Extracted loaded-dataset validation/reporting, composed-eval slice reporting, and eval-key construction from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_dataset_context.py`.
- `self/core/nonadaptive_loop.py` now receives a `NonAdaptiveDatasetContext` from `prepare_nonadaptive_dataset_context(...)`, keeping the loop focused on bootstrap and round orchestration.
- Added `tests/test_nonadaptive_dataset_context.py` for dataset count messages, composed-eval slice count messages, eval-key derivation, skipped slice reporting when there are no composed-eval examples, and empty-train rejection.
- Updated `self/README.md` with the new dataset-context module ownership and revised the remaining cleanup queue.
- Verification: `python -m py_compile self/core/nonadaptive_dataset_context.py self/core/nonadaptive_loop.py tests/test_nonadaptive_dataset_context.py`; `python -m compileall -q self/core tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_schedule.py tests/test_nonadaptive_state.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py -q` (`35 passed`). The temporary `.pytest_tmp_nonadaptive` directory was removed after verification.

### Implementation Log: 2026-06-18 06:16:36 UTC

- Extracted non-adaptive RNG seeding, resumed RNG-state restoration, and metadata persistence from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_metadata_runtime.py`.
- The loop now creates a `NonAdaptiveMetadataRuntime` with the same monkeypatchable `set_seed`, `random.Random`, `decode_rng_state`, `persist_nonadaptive_metadata`, `encode_rng_state`, and sanitization functions, preserving old `self.self_improvement_core` patch behavior while removing the inline metadata closure.
- Added `tests/test_nonadaptive_metadata_runtime.py` for seed calls, restored RNG state, default metadata persistence, explicit target metadata persistence, and updating the default metadata reference after dataset generation.
- Updated `self/README.md` with the new metadata runtime module ownership and revised the remaining cleanup queue.
- Verification: `python -m py_compile self/core/nonadaptive_metadata_runtime.py self/core/nonadaptive_loop.py tests/test_nonadaptive_metadata_runtime.py`; `python -m compileall -q self/core tests/test_nonadaptive_metadata_runtime.py tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_metadata_runtime.py tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_schedule.py tests/test_nonadaptive_state.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py -q` (`37 passed`). The temporary `.pytest_tmp_nonadaptive` directory was removed after verification.

### Implementation Log: 2026-06-18 06:20:31 UTC

- Extracted non-adaptive final checkpoint cleanup and final result-path reporting from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_finalization.py`.
- The loop now delegates to `finalize_nonadaptive_run(...)` and passes the monkeypatchable `cleanup_round_checkpoints` function through explicitly, preserving old `self.self_improvement_core` patch behavior.
- Added `tests/test_nonadaptive_finalization.py` for cleanup enabled, cleanup skipped by `keep_checkpoints`, cleanup skipped by `save_model_policy=none`, and final result message reporting.
- Updated `self/README.md` with the new non-adaptive finalization module ownership and revised the remaining cleanup queue.
- Verification: `python -m py_compile self/core/nonadaptive_finalization.py self/core/nonadaptive_loop.py tests/test_nonadaptive_finalization.py`; `python -m compileall -q self/core tests/test_nonadaptive_finalization.py tests/test_nonadaptive_metadata_runtime.py tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_finalization.py tests/test_nonadaptive_metadata_runtime.py tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_schedule.py tests/test_nonadaptive_state.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py -q` (`40 passed`). The temporary `.pytest_tmp_nonadaptive` directory was removed after verification.

### Implementation Log: 2026-06-18 06:31:17 UTC

- Extracted single-round non-adaptive orchestration from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_round_runtime.py`.
- The outer loop now builds a `NonAdaptiveRoundRuntimeContext` and mutable `NonAdaptiveRoundRuntimeState`, then delegates each round through `run_nonadaptive_round(...)`. The helper preserves the old execution order: round plan, train/pseudo artifact write, train, evaluate, next pseudo-label generation, summary write, and lifecycle/reset handling.
- Kept monkeypatch compatibility by passing the same low-level bindings from `self/core/nonadaptive_loop.py` into the round runtime, including `save_examples`, `make_training_args`, `build_trainer`, evaluation/debug helpers, summary helpers, `resolve_max_new_tokens`, model reload helpers, and CUDA cache functions.
- Added `tests/test_nonadaptive_round_runtime.py` for skipped resumed rounds and full round state transitions through fake train/eval/pseudo/summary/lifecycle hooks.
- Updated `self/README.md` with the new round-runtime module ownership and revised the remaining cleanup queue. `self/core/nonadaptive_loop.py` is now `302` lines, down from `380` after the previous pass.
- Verification: `python -m py_compile self/core/nonadaptive_round_runtime.py self/core/nonadaptive_loop.py tests/test_nonadaptive_round_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_round_runtime.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive tests/test_nonadaptive_round_runtime.py tests/test_nonadaptive_finalization.py tests/test_nonadaptive_metadata_runtime.py tests/test_nonadaptive_dataset_context.py tests/test_nonadaptive_round_setup.py tests/test_nonadaptive_lifecycle.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py tests/test_nonadaptive_datasets.py tests/test_nonadaptive_evaluation.py tests/test_nonadaptive_schedule.py tests/test_nonadaptive_state.py tests/test_nonadaptive_training.py tests/test_nonadaptive_bootstrap.py tests/test_bit_task_seed_round_zero.py -q` (`46 passed`). The temporary `.pytest_tmp_nonadaptive` directory was removed after verification.

### Implementation Log: 2026-06-18 06:41:05 UTC

- Extracted proposal/outcome trace example data models, JSON parsing, proposal replay sampling, post-task proposal rehearsal sampling, and outcome replay sampling from `self/core/experience_traces.py` into `self/core/experience_trace_models.py`.
- Kept compatibility by reexporting `ProposalTraceExample`, `OutcomeTraceExample`, `proposal_trace_from_json(...)`, `outcome_trace_from_json(...)`, `sample_proposal_trace_replay(...)`, `build_post_task_proposal_rehearsal_examples(...)`, and `sample_outcome_trace_replay(...)` from `self/core/experience_traces.py`.
- Updated internal candidate scoring, candidate worker runtime, candidate dispatch runtime/entrypoints, and driver compatibility exports to import trace model/replay names from the new canonical module where possible. Trace construction helpers remain in `self/core/experience_traces.py`.
- Added `tests/test_experience_trace_models.py` for trace JSON round-trips, old import identity, replay ratio/cap behavior, post-task rehearsal sampling, and outcome replay disabling when the target mode is `none`.
- Updated `self/README.md` with the new trace-model module ownership. `self/core/experience_traces.py` is now `567` lines, down from `741`.
- Verification: `python -m py_compile self/core/experience_trace_models.py self/core/experience_traces.py self/core/candidate_scoring.py self/core/candidate_worker_runtime.py self/core/candidate_dispatch_runtime.py self/core/candidate_dispatch_entrypoints.py self/core/driver_compat_exports.py tests/test_experience_trace_models.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_traces tests/test_experience_trace_models.py tests/test_adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`48 passed`, `3` existing multiprocessing fork warnings). The temporary `.pytest_tmp_traces` directory was removed after verification.

### Implementation Log: 2026-06-18 06:50:44 UTC

- Extracted outcome trace construction and rendering from `self/core/experience_traces.py` into `self/core/experience_outcome_traces.py`.
- The new module owns `build_outcome_trace_example(...)`, `build_round_outcome_trace_examples(...)`, compact state/candidate JSON rendering, failure-code derivation, numeric/textual completion payloads, prediction-error fields, and feedback text. Proposal trace construction remains in `self/core/experience_traces.py`.
- Kept compatibility by reexporting the outcome builder names from `self/core/experience_traces.py`, while updating driver default bindings and driver compatibility exports to import the outcome builders from the new canonical module.
- Added `tests/test_experience_outcome_traces.py` for canonical outcome trace construction, old import identity, prediction-error fields, invalid-proposal failure coding, and disabled outcome-trace mode.
- Updated `self/README.md` with the new outcome-trace module ownership. `self/core/experience_traces.py` is now `130` lines, down from `567` after the previous trace-model pass.
- Verification: `python -m py_compile self/core/experience_outcome_traces.py self/core/experience_traces.py self/core/driver_default_bindings.py self/core/driver_compat_exports.py tests/test_experience_outcome_traces.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_outcomes tests/test_experience_outcome_traces.py tests/test_experience_trace_models.py -q` (`4 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_outcomes tests/test_experience_outcome_traces.py tests/test_experience_trace_models.py tests/test_adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`50 passed`, `3` existing multiprocessing fork warnings). The temporary `.pytest_tmp_outcomes` directory was removed after verification.

### Implementation Log: 2026-06-18 06:58:32 UTC

- Wired the existing packed-local candidate worker and base-checkpoint cache optimization into the main adaptive AILAB candidate-training config path.
- `launchers/self/config/adaptive_candidate_base.env` now defaults to `CANDIDATE_LOCAL_PACK_SIZE=2` and `CANDIDATE_LOCAL_CACHE_BASE_STATE=1`, while the parser defaults remain unchanged for arbitrary callers.
- `launchers/self/run_adaptive_candidate_training_ailab.sbatch` now reads those env vars, logs `parallelism/pack/cache-base-state`, passes `--candidate-local-pack-size`, and conditionally adds `--candidate-local-cache-base-state`.
- Added `tests/test_adaptive_candidate_launcher.py` with bash syntax coverage and a stub-Python execution check that verifies the config emits `--candidate-local-pack-size 2` and `--candidate-local-cache-base-state` without starting real training.
- Updated `self/README.md` runtime notes to distinguish parser defaults from the shared AILAB adaptive candidate config defaults.
- Verification: `bash -n launchers/self/run_adaptive_candidate_training_ailab.sbatch launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/config/adaptive_candidate_base.env`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_launcher tests/test_adaptive_candidate_launcher.py -q` (`2 passed`). The temporary `.pytest_tmp_launcher` directory was removed after verification.

### Implementation Log: 2026-06-18 07:08:03 UTC

- Extracted candidate task/pseudo/replay trace train-mix construction and artifact writing from `self/core/candidate_scoring.py` into `self/core/candidate_training_mix.py`.
- The new module owns `CandidateTrainingMix`, `build_candidate_training_mix(...)`, and `write_candidate_training_mix_artifacts(...)`, preserving existing artifact paths and `train_mix_summary.json` fields.
- `candidate_scoring.py` now focuses on no-pseudo rejection, checkpoint training, optional proposal rehearsal phase, evaluation, reward/metric construction, and model cleanup; it consumes the returned training mix for counts and train examples.
- Added `tests/test_candidate_training_mix.py` for mixed replay with post-task rehearsal disabled, separated post-task rehearsal examples, and train-mix artifact writing.
- Updated `self/README.md` with the candidate training-mix module ownership. `self/core/candidate_scoring.py` is now `318` lines, down from `387`.
- Verification: `python -m py_compile self/core/candidate_training_mix.py self/core/candidate_scoring.py tests/test_candidate_training_mix.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_mix tests/test_candidate_training_mix.py -q` (`3 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_mix tests/test_candidate_training_mix.py tests/test_adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`49 passed`, `3` existing multiprocessing fork warnings). The temporary `.pytest_tmp_mix` directory was removed after verification.

### Implementation Log: 2026-06-18 07:15:39 UTC

- Extracted static-frontier aggregation, no-pseudo failure metrics, trained-candidate reward calculation, and candidate metric construction from `self/core/candidate_scoring.py` into `self/core/candidate_rewards.py`.
- `candidate_scoring.py` now delegates reward/metric construction through `build_no_pseudo_candidate_metrics(...)` and `build_trained_candidate_metrics(...)`, leaving the module focused on training, optional proposal rehearsal, evaluation, metric artifact writing, and cleanup.
- Kept old compatibility names working: `static_frontier_sizes(...)` and `mean_accuracy_for_sizes(...)` remain importable through the driver facade and through `self/core/candidate_scoring.py`, while `self/core/candidate_rewards.py` is the canonical owner.
- Added `tests/test_candidate_rewards.py` for missing/bad frontier accuracy handling, no-pseudo failure metrics, and the unchanged reward formula `frontier_delta + lambda_final * (candidate_final_accuracy - init_final_accuracy)`.
- Updated `self/README.md` with the candidate reward/metric module ownership. `self/core/candidate_scoring.py` is now `260` lines, down from `318`.
- Verification: `python -m py_compile self/core/candidate_rewards.py self/core/candidate_scoring.py self/core/driver_compat_exports.py tests/test_candidate_rewards.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rewards tests/test_candidate_rewards.py tests/test_candidate_training_mix.py tests/test_adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`52 passed`, `3` existing multiprocessing fork warnings). The temporary `.pytest_tmp_rewards` directory was removed after verification.

### Implementation Log: 2026-06-18 07:21:39 UTC

- Extracted candidate `TrainingConfig` construction, checkpoint fine-tuning, held-out evaluation, CUDA cache clearing, and post-task proposal rehearsal training/summary writing from `self/core/candidate_scoring.py` into `self/core/candidate_training_runtime.py`.
- Updated `candidate_scoring.py` to orchestrate candidate scoring through the training runtime, train-mix helper, and reward helper. It still explicitly releases the task model before calling post-task proposal rehearsal training, preserving the previous GPU-memory behavior.
- Updated internal controller phases and driver default/compat exports to import train/eval helpers from `self/core/candidate_training_runtime.py`; old facade names such as `make_config`, `train_checkpoint`, and `evaluate_model` remain reachable through compatibility paths.
- Added `tests/test_candidate_training_runtime.py` with fake trainer/model/evaluation hooks covering config construction, checkpoint-training wiring, decode-budget evaluation wiring, and proposal rehearsal summary/cleanup behavior.
- Updated `self/README.md` with the candidate training runtime module ownership. `self/core/candidate_scoring.py` is now `140` lines, down from `260`.
- Verification: `python -m py_compile self/core/candidate_training_runtime.py self/core/candidate_scoring.py self/core/controller_phase_runtime.py self/core/driver_default_bindings.py self/core/driver_compat_exports.py tests/test_candidate_training_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_runtime tests/test_candidate_training_runtime.py -q` (`4 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_runtime tests/test_candidate_training_runtime.py tests/test_candidate_rewards.py tests/test_candidate_training_mix.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`58 passed`, `3` existing multiprocessing fork warnings). The temporary `.pytest_tmp_runtime` directory was removed after verification.

### Implementation Log: 2026-06-18 07:29:33 UTC

- Moved the composition-error experiment entry point from `self/self_improvement_composition_error_experiment.py` into canonical `self/experiments/composition_error_sweep.py`.
- Replaced the old top-level module with a compatibility wrapper that forwards attribute reads and writes to the canonical experiment module, preserving old imports and monkeypatch-style tests.
- Updated `launchers/self/run_composition_error_sweep_self_improvement.sh` and `launchers/self/run_self_improvement_mig_boundary_eval.sbatch` to call `python -m self.experiments.composition_error_sweep` instead of the old wrapper path.
- Added `tests/test_composition_error_sweep.py` for argument forwarding, explicit strategy preservation, invalid percent rejection, old-wrapper monkeypatch forwarding, launcher bash syntax, and dry-run command output.
- Updated `self/README.md` with the new experiment owner and compatibility wrapper mapping.
- Verification: `python -m py_compile self/experiments/composition_error_sweep.py self/self_improvement_composition_error_experiment.py tests/test_composition_error_sweep.py`; `bash -n launchers/self/run_composition_error_sweep_self_improvement.sh launchers/self/run_self_improvement_mig_boundary_eval.sbatch`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_composition tests/test_composition_error_sweep.py -q` (`5 passed`). The temporary `.pytest_tmp_composition` directory was removed after verification.

### Implementation Log: 2026-06-18 07:33:22 UTC

- Moved the addition-specific recipe alias and resolver into canonical `self/core/recipes.py` as `AdditionRecipePreset` and `resolve_addition_recipe(...)`.
- Converted `self/addition_recipe.py` into a compatibility proxy over `self.core.recipes`, preserving old imports and attribute-write forwarding for legacy monkeypatch patterns.
- Updated current `self/diagnostics/addition_recipe_diagnostic.py` to import recipe helpers from `self.core.recipes`; legacy scripts continue to use the old wrapper path.
- Added a compatibility assertion in `tests/test_addition_recipe_diagnostic.py` confirming old `self.addition_recipe` exports resolve to the canonical recipe helpers.
- Updated `self/README.md` with the addition recipe wrapper mapping and recipe-helper ownership.
- Verification: `python -m py_compile self/core/recipes.py self/addition_recipe.py self/diagnostics/addition_recipe_diagnostic.py tests/test_addition_recipe_diagnostic.py tests/test_legacy_addition_self_improvement.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_recipe tests/test_addition_recipe_diagnostic.py tests/test_legacy_addition_self_improvement.py -q` (`31 passed`); import sanity confirmed `self.addition_recipe.resolve_addition_recipe` and `self.addition_recipe.AdditionRecipePreset` are the canonical `self.core.recipes` objects. The temporary `.pytest_tmp_recipe` directory was removed after verification.

### Implementation Log: 2026-06-18 07:37:16 UTC

- Added `resolve_self_improvement_results_path(...)` to `self/analysis/nonadaptive_artifacts.py` and reexported it through `self/analysis/artifacts.py` so notebooks and plotting helpers can resolve run directories without hard-coding `self_improvement_results.json`.
- Updated `self/analysis/plot_self_improvement_figure.py` to use `resolve_self_improvement_results_path(...)` and `load_self_improvement_rounds(...)` instead of direct JSON loading, while preserving existing missing-file errors and sorted-round plotting behavior.
- Added `tests/test_analysis_artifacts.py` coverage for the resolver, artifacts reexport identity, plot helper path resolution, sorted record loading, and non-list payload rejection.
- Updated `self/README.md` to document non-adaptive result-path resolution and the plot CLI's use of the stable artifact loader.
- Verification: `python -m py_compile self/analysis/nonadaptive_artifacts.py self/analysis/artifacts.py self/analysis/plot_self_improvement_figure.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_loader tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`10 passed`). The temporary `.pytest_tmp_analysis_loader` directory was removed after verification.

### Implementation Log: 2026-06-18 07:43:16 UTC

- Removed the stale multiplication aggregation flag/default and its diagnostics/metadata fields. Multiplication now records only the supported blocked-component composition path.
- Renamed the multiplication corruption test away from aggregation-condition wording and removed the unused argument plumbing from the seed-fit and legacy multiplication wrappers.
- Updated `self/README.md` and the old workshop planning doc so tracked docs no longer imply an extra aggregation condition beyond exact blocked composition.
- Verification: `python -m py_compile self/tasks/multiplication.py self/experiments/seed_fit_experiment.py self/legacy/multiplication_self_improvement.py tests/test_self_improvement_tasks.py`; tracked terminology grep returned no matches; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_aggregation_cleanup tests/test_self_improvement_tasks.py tests/test_nonadaptive_setup.py tests/test_candidate_rewards.py -q` (`40 passed`).

### Implementation Log: 2026-06-18 07:48:01 UTC

- Added `self_add_dry_run_arg(...)` and `self_print_context(...)` to `launchers/self/lib/self_common.sh` so small submitter wrappers can share dry-run flag insertion and `[INFO]` context printing in addition to repo-root/Python setup and command formatting.
- Updated `launchers/self/submit_figure2_condition_sweep_mig.sh`, `launchers/self/submit_figure3_seed_quality_sweep_mig.sh`, and `launchers/self/submit_figure3_real_seed_data_ablation_mig.sh` to use the new helpers while preserving their Python CLI delegation and dry-run command output.
- Updated `self/README.md` to note that the small Figure 2/3 wrappers now share dry-run and context-printing helpers as well as the generic setup path.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_figure2_condition_sweep_mig.sh launchers/self/submit_figure3_seed_quality_sweep_mig.sh launchers/self/submit_figure3_real_seed_data_ablation_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure_launchers tests/test_figure2_condition_sweep.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`16 passed`).

### Implementation Log: 2026-06-18 07:52:13 UTC

- Extended the shared launcher helper cleanup to the Figure 2 recipe family.
- Updated `launchers/self/run_figure2_recipe_aggressive.sh` and `launchers/self/submit_figure2_recipe_aggressive.sh` to use `self_print_context(...)` for repeated `[INFO]` context output while preserving schedule resolution, stage orchestration, and Slurm submission logic.
- Updated `launchers/self/run_figure2_paper_retune.sh` to use `self_add_dry_run_arg(...)` and `self_print_context(...)` while preserving its Python CLI command construction and optional override forwarding.
- Updated `self/README.md` to note that the Figure 2 recipe runner/submitter/retune scripts now share context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_figure2_recipe_aggressive.sh launchers/self/submit_figure2_recipe_aggressive.sh launchers/self/run_figure2_paper_retune.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_fig2_recipe_helpers tests/test_figure2_recipe_aggressive_launchers.py -q` (`5 passed`).

### Implementation Log: 2026-06-18 07:55:15 UTC

- Extended the shared launcher context helper cleanup to the addition recipe focused/fullpack/recovery workflow.
- Updated `launchers/self/run_addition_recipe_focused.sh`, `launchers/self/run_addition_recipe_fullpack.sh`, and `launchers/self/run_addition_recipe_recovery.sh` to route their top-level `[INFO]` context banners through `self_print_context(...)` while preserving per-baseline status lines, schedule resolution, gate logic, and command execution behavior.
- Updated `self/README.md` to note that the addition recipe focused/fullpack/recovery context banners now use the shared helper path.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_recipe_focused.sh launchers/self/run_addition_recipe_fullpack.sh launchers/self/run_addition_recipe_recovery.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_recipe_helpers tests/test_addition_recipe_recovery_launchers.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 07:58:59 UTC

- Extended shared context-banner printing to the rectangular non-square multiplication seed/search and self-improvement launchers.
- Updated `launchers/self/run_multiplication_rectangular_seed_mig.sbatch`, `launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch`, and `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` to route repeated top-level `[INFO]` context output through `self_print_context(...)` while preserving CUDA dry-run status lines, Slurm export construction, and command execution behavior.
- Updated `self/README.md` to note that rectangular non-square launchers now share context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_seed_mig.sbatch launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh launchers/self/submit_multiplication_rectangular_fullpack_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rect_helpers tests/test_multiplication_rectangular_seed_launchers.py tests/test_multiplication_rectangular_self_improvement_launchers.py -q` (`10 passed`).

### Implementation Log: 2026-06-18 08:01:50 UTC

- Extended shared context-banner printing to the rectangular square multiplication seed and composition-diagnostic runners.
- Updated `launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch` and `launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch` to route repeated top-level `[INFO]` context output through `self_print_context(...)` while preserving CUDA dry-run status lines, command construction, and wrapper submission behavior.
- Updated `self/README.md` to note that rectangular square runners now share context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch launchers/self/submit_multiplication_rectangular_square_probe_mig.sh launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_square_helpers tests/test_multiplication_rectangular_square_launchers.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 08:04:51 UTC

- Extended shared context-banner printing to the tested fixed-width mixed-prompt addition fullpack and more-data sweep launchers.
- Updated `launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh` and `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` to route repeated top-level `[INFO]` context output through `self_print_context(...)` while preserving baseline command construction, Slurm manifest writing, and dry-run grid output.
- Updated `self/README.md` to note that the fixed-width fullpack and more-data sweep context banners now use the shared helper path.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_fixedwidth_helpers tests/test_addition_fixedwidth_moredata_launcher.py -q` (`3 passed`).

### Implementation Log: 2026-06-18 08:08:19 UTC

- Extended shared context-banner printing to the guarded plain-output bit diagnostic runner and submitter.
- Updated `launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch` and `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` to route repeated top-level `[INFO]` context output through `self_print_context(...)` while preserving diagnostic command construction, `nvidia-smi` probe behavior, and dry-run submission output.
- Updated `self/README.md` to note that the guarded diagnostic launchers now share context-printing helpers in addition to repo-root/Python and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_guarded_helpers tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`5 passed`).

### Implementation Log: 2026-06-18 08:11:02 UTC

- Extended shared context-banner printing to `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`.
- Replaced the submitter's repeated top-level `[INFO]` context `echo` block with `self_print_context(...)` while preserving baseline command construction, Slurm resource helper use, manifest writing, and dry-run submission output.
- Updated `self/README.md` to note that the run-length fixed-binary baseline submitter now shares context-printing helpers in addition to explicit Slurm resource helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rl_baseline_helpers tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`5 passed`).

### Implementation Log: 2026-06-18 08:13:47 UTC

- Switched `launchers/self/submit_multiplication_rectangular_tune_mig.sh` from manual dry-run flag insertion to the shared `self_add_dry_run_arg(...)` helper.
- Preserved the rectangular tune Python CLI command construction, stage manifest paths, dry-run output, and command-printing behavior.
- Updated `self/README.md` to note that the rectangular tune submitter now shares the dry-run flag helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_multiplication_rectangular_tune_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rect_tune_helper tests/test_multiplication_rectangular_tune.py -q` (`11 passed`).

### Implementation Log: 2026-06-18 08:18:45 UTC

- Confirmed the workshop/main-track repo surface should not include the old auxiliary classification task path.
- Removed the stale top-level auxiliary bit-string compatibility wrapper so the public non-adaptive task CLIs remain addition, run-length, and multiplication.
- Updated the adaptive pilot compile check to point at the explicit legacy run-length bit-string helper and updated the guarded diagnostic launcher test import accordingly.
- Updated `self/README.md` and redacted the stale wrapper mention from this log so tracked docs no longer direct readers to a public auxiliary bit-task entry point.
- Verification: `python -m py_compile self/legacy/bit_task_self_improvement.py self/legacy/run_length_self_improvement.py self/run_length_self_improvement.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py`; `bash -n launchers/self/run_adaptive_self_improvement_ailab.sbatch launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh`; tracked grep confirms no stale auxiliary task references and no stale public auxiliary-wrapper path references; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_aux_cleanup tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_self_improvement_launchers.py -q` (`8 passed`).

### Implementation Log: 2026-06-18 08:22:51 UTC

- Extended shared context-banner printing to the addition fullpack-filtered runner and submitter.
- Updated `launchers/self/run_addition_fullpack_filtered.sbatch` and `launchers/self/submit_addition_fullpack_filtered_mig.sh` to route their repeated top-level `[INFO]` context output through `self_print_context(...)` while preserving seed-link setup, preflight behavior, per-baseline status output, Slurm submission, and dry-run behavior.
- Updated `self/README.md` to note that the addition fullpack-filtered launcher pair now shares context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fullpack_filtered.sbatch launchers/self/submit_addition_fullpack_filtered_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_fullpack_context tests/test_addition_fullpack_filtered_launcher.py -q` (`5 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:25:10 UTC

- Extended shared context-banner printing to the generic task self-improvement runner.
- Updated `launchers/self/run_task_self_improvement.sbatch` to route its repeated task/module/budget/model context output through `self_print_context(...)` while preserving output directory creation, preflight behavior, launch command printing, and dry-run behavior.
- Updated `self/README.md` to note that the generic task self-improvement budget-grid runner/submitter now shares context-printing helpers in addition to repo-root/Python/boolean and dry-run helper behavior.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_task_self_improvement.sbatch launchers/self/submit_budget_grid_self_improvement.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_task_context tests/test_task_self_improvement_launchers.py -q` (`3 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:27:56 UTC

- Extended shared context-banner printing to the generic seed-fit runner.
- Updated `launchers/self/run_seed_fit_experiment.sbatch` to route its repeated task/data/model context output through `self_print_context(...)` while preserving output directory creation, preflight behavior, launch command printing, and dry-run behavior.
- Updated `self/README.md` to note that the seed-fit grid runner/submitter now shares context-printing helpers in addition to repo-root/Python/boolean and dry-run helper behavior.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_seed_fit_experiment.sbatch launchers/self/submit_seed_fit_grid.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_seedfit_context tests/test_seed_fit_launchers.py -q` (`3 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:30:22 UTC

- Extended shared context-banner printing to the shared addition seed runner.
- Updated `launchers/self/run_addition_seed_shared.sbatch` to route its repeated output/model/batch context output through `self_print_context(...)` while preserving CUDA probe behavior, seed-training command construction, dry-run behavior, and final seed-model symlink creation.
- Added focused dry-run coverage for the shared seed launcher in `tests/test_addition_tiny_seed_mig_launcher.py` alongside the existing tiny-seed launcher checks.
- Updated `self/README.md` to note that addition seed runners now share context-printing helpers in addition to repo-root/Python/boolean and staged dry-run command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_seed_shared.sbatch launchers/self/run_addition_tiny_seed_mig.sbatch`; `python -m py_compile tests/test_addition_tiny_seed_mig_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_seed_shared_context tests/test_addition_tiny_seed_mig_launcher.py -q` (`4 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:34:02 UTC

- Extended shared context-banner printing to the fixed-width mixed-prompt addition seed runner.
- Updated `launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch` to route its repeated repo/Python/output/sampling context output through `self_print_context(...)` while preserving the seed-fit command, dry-run behavior, threshold gate, and stable seed symlink update.
- Added focused dry-run coverage for the fixed-width mixed seed runner in `tests/test_addition_fixedwidth_moredata_launcher.py` alongside the existing fullpack and more-data submitter checks.
- Updated `self/README.md` to note that the fixed-width mixed-prompt addition launcher family now shares context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_mixed_mig.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; `python -m py_compile tests/test_addition_fixedwidth_moredata_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_fixedwidth_seed_context tests/test_addition_fixedwidth_moredata_launcher.py -q` (`4 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:37:03 UTC

- Extended shared context-banner printing to the fixed-width mixed-prompt addition submitter.
- Updated `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` to route its repeated run-root/output/model/sampling context output through `self_print_context(...)` while preserving seed/fullpack/original-composition dry-run branches, Slurm submission dependencies, and manifest writing.
- Added focused dry-run coverage for the mixed submitter's three dry-run branches in `tests/test_addition_fixedwidth_moredata_launcher.py`.
- Updated `self/README.md` to note that the fixed-width mixed-prompt addition family has submitter branch dry-run coverage in addition to shared repo-root/Python/boolean/context/command helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_mixed_mig.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; `python -m py_compile tests/test_addition_fixedwidth_moredata_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_fixedwidth_mixed_submitter_context tests/test_addition_fixedwidth_moredata_launcher.py -q` (`5 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:40:39 UTC

- Extended shared context-banner printing to the addition recipe diagnostic launcher.
- Updated `launchers/self/run_addition_their_recipe_diagnostic.sh` to route its repo/Python/output/device/dry-run context output through `self_print_context(...)` while preserving diagnostic command construction, optional train/eval/max-step forwarding, and dry-run behavior.
- Updated `self/README.md` to note that the addition recipe diagnostic launcher now shares context-printing helpers in addition to repo-root/Python/boolean and command-printing helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_their_recipe_diagnostic.sh`; `python -m py_compile tests/test_addition_their_recipe_diagnostic_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_recipe_diag_context tests/test_addition_their_recipe_diagnostic_launcher.py -q` (`2 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:43:16 UTC

- Extended shared context-banner printing to the exact-digits fixed-binary addition submitter.
- Updated `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` to route its root/model/log/Slurm/schedule context output through `self_print_context(...)` while preserving the fullpack command construction, dry-run output, Slurm resource helper use, and manifest format.
- Updated `self/README.md` to note that the exact-digits addition submitter now routes its schedule banner through the shared context printer in addition to common MIG resource helpers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_exact_digits_context tests/test_addition_exact_digits_fixed_binary_launcher.py -q` (`2 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:50:14 UTC

- Cleaned the current workshop/main-track code surface so it no longer exposes the removed auxiliary classification task family through generic `bit_task` names.
- Renamed the legacy run-length bit-string CLI helper from `self/legacy/bit_task_self_improvement.py` to `self/legacy/run_length_bit_cli.py`, renamed its parser/normalization functions, and updated the run-length legacy entry point plus adaptive pilot compile checks to use the new path.
- Renamed the current run-length recipe and non-adaptive seed-round tests away from `bit_task` filenames/function names, and changed the Figure 2 manifest kind from `bit_task` to `run_length_bit`.
- Updated user-facing wording in `self/README.md`, `self/core/nonadaptive_setup.py`, `self/diagnostics/check_self_improvement_overfit.py`, `self/experiments/seed_fit_experiment.py`, and `launchers/self/run_refocused_self_improvement_local.sh` so the remaining bit-size controls are described as run-length bit-string controls rather than a generic auxiliary bit task.
- Tracked search over current `self`, `core`, `launchers`, `tests`, `docs`, and `self/README.md` now has no removed auxiliary task or old generic `bit_task` helper references. Historical plan-log verification lines are intentionally left as history.

### Implementation Log: 2026-06-18 08:54:57 UTC

- Extended shared context-banner printing to the adaptive condition runner.
- Updated `launchers/self/run_adaptive_condition_ailab.sbatch` to route its task/condition, output directory, and fixture context output through `self_print_context(...)` while preserving required environment validation, runtime setup, optional compile check, worker context logging, and the adaptive proposal-condition command.
- Added `tests/test_adaptive_condition_launcher.py` with bash syntax coverage and a fake-`PYTHON_BIN` execution test that verifies the shared context banner plus the wired adaptive proposal command without importing Torch or running model code.
- Updated `self/README.md` to note that the adaptive condition runner now uses the shared context printer for its task/output/fixture banner.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/run_adaptive_condition_ailab.sbatch`; `python -m py_compile tests/test_adaptive_condition_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_condition_context tests/test_adaptive_condition_launcher.py -q` (`2 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 08:57:24 UTC

- Centralized the repeated adaptive launcher Torch/CUDA probe in `launchers/self/lib/adaptive_common.sh`.
- Added `adaptive_print_torch_probe(...)` and updated `run_adaptive_self_improvement_ailab.sbatch`, `run_adaptive_condition_ailab.sbatch`, and `run_adaptive_candidate_training_ailab.sbatch` to call it instead of duplicating the inline Python probe.
- Updated `self/README.md` to record that `adaptive_common.sh` now owns Torch/CUDA probe printing in addition to cache/runtime setup, config sourcing, and worker context logging.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/run_adaptive_self_improvement_ailab.sbatch launchers/self/run_adaptive_condition_ailab.sbatch launchers/self/run_adaptive_candidate_training_ailab.sbatch`; `python -m py_compile tests/test_adaptive_condition_launcher.py tests/test_adaptive_candidate_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_torch_probe tests/test_adaptive_condition_launcher.py tests/test_adaptive_candidate_launcher.py -q` (`4 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 09:00:49 UTC

- Continued shrinking the adaptive driver wiring layer by extracting driver-to-worker entrypoint binding construction into `self/core/driver_worker_wiring.py`.
- Moved the `WorkerEntrypointDeps` construction and candidate/controller worker wrapper functions out of `self/core/driver_wiring.py`; `driver_wiring.py` now imports those bridge functions while continuing to expose the same names for `self.core.driver` compatibility wrappers.
- Updated `self/README.md` to document `self/core/driver_worker_wiring.py` as the focused owner for candidate worker, packed candidate worker, and controller worker driver-binding bridges.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_wiring.py self/core/driver_worker_wiring.py self/core/worker_entrypoints.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_worker_specs.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_worker_wiring tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_worker_specs.py -q` (`44 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 09:04:42 UTC

- Continued shrinking the adaptive driver wiring layer by extracting driver-to-candidate-dispatch binding construction into `self/core/driver_candidate_dispatch_wiring.py`.
- Moved the `CandidateDispatchEntrypointDeps` construction and serial/local-parallel/Slurm-array candidate dispatch wrapper functions out of `self/core/driver_wiring.py`; `driver_wiring.py` now imports those bridge functions while continuing to expose the same names for `self.core.driver` compatibility wrappers.
- Updated `self/README.md` to document `self/core/driver_candidate_dispatch_wiring.py` as the focused owner for candidate training dispatch bridges.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_wiring.py self/core/driver_candidate_dispatch_wiring.py self/core/candidate_dispatch_entrypoints.py self/core/candidate_dispatch_runtime.py tests/test_candidate_dispatch_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_worker_specs.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_candidate_dispatch tests/test_candidate_dispatch_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_worker_specs.py -q` (`48 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 09:09:41 UTC

- Removed the final tracked plan-log wording that named the old auxiliary classification task which is no longer part of the workshop/main-track repo surface.
- Verification: `git grep -n -i <removed-task-name> -- .` returned no tracked matches; `git diff --check`.

### Implementation Log: 2026-06-18 09:12:18 UTC

- Continued shrinking the adaptive driver wiring layer by extracting driver-to-proposal-GRPO dispatch binding construction into `self/core/driver_proposal_grpo_wiring.py`.
- Moved the `ProposalGrpoDispatchDeps` construction and proposal-GRPO local-vs-Slurm dispatch wrapper out of `self/core/driver_wiring.py`; `driver_wiring.py` now imports the bridge function while continuing to expose the same name for `self.core.driver` compatibility wrappers.
- Updated `self/README.md` to document `self/core/driver_proposal_grpo_wiring.py` as the focused owner for proposal-GRPO update dispatch bridges.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_wiring.py self/core/driver_proposal_grpo_wiring.py self/core/proposal_grpo_dispatch.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_proposal_grpo tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 09:17:17 UTC

- Continued shrinking the adaptive driver wiring layer by extracting driver-to-run-orchestration binding construction into `self/core/driver_run_wiring.py`.
- Moved `AdaptiveRunDeps` construction and the full adaptive run wrapper out of `self/core/driver_wiring.py`; `driver_wiring.py` now reexports the run bridge while staying focused on compatibility aggregation and CLI entrypoint wiring.
- Updated `self/README.md` to document `self/core/driver_run_wiring.py` as the focused owner for adaptive run orchestration bridges.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_wiring.py self/core/driver_run_wiring.py self/core/run_orchestration.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_run_wiring tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`40 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 09:23:35 UTC

- Extended the stable adaptive artifact loader in `self/analysis/adaptive_artifacts.py` so notebooks no longer need to hard-code raw paths for proposal prompts, per-candidate `train_mix_summary.json`, or common attempt trace JSONL files.
- Added `adaptive_prompt_records(...)`, `adaptive_candidate_train_mix_records(...)`, and `adaptive_trace_records(...)`, and reexported them through `self/analysis/artifacts.py` for the existing notebook compatibility import surface.
- Updated `self/README.md` to document prompt/train-mix/trace flattening as part of the adaptive analysis utilities.
- Verification: `python -m py_compile self/analysis/adaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_artifacts tests/test_analysis_artifacts.py -q` (`4 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 09:26:41 UTC

- Consolidated repeated wrapped Slurm-job submission boilerplate into `self_submit_wrapped_job(...)` in `launchers/self/lib/self_common.sh`.
- Updated `launchers/self/submit_run_length_fixed_binary_mig.sh` to use the shared helper for its paper-default, alpha10 template, and alpha10 beam jobs while preserving dry-run job IDs, dependency forwarding, explicit resource requests, and manifest output.
- Updated `self/README.md` to document wrapped-job submission/dry-run handling as a shared launcher helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_run_length_fixed_binary_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rl_fixed_binary_launcher tests/test_run_length_fixed_binary_launchers.py -q` (`3 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 09:30:12 UTC

- Continued launcher consolidation by migrating `launchers/self/submit_main_experiments_ailab.sh` to the shared `self_submit_wrapped_job(...)` helper while preserving its AILAB resource defaults, wrapped commands, dry-run job IDs, and manifest schema.
- Added `tests/test_main_experiments_launcher.py` with bash syntax coverage and a dry-run manifest check for the main experiment submitter.
- Updated `self/README.md` to note that the main-experiment submitter now shares wrapped-job submission in addition to adaptive repo-root/Python/resource helper setup.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/submit_main_experiments_ailab.sh`; `python -m py_compile tests/test_main_experiments_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_main_experiments_launcher tests/test_main_experiments_launcher.py -q` (`2 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 09:33:38 UTC

- Added `self_submit_sbatch_command(...)` to `launchers/self/lib/self_common.sh` for submitters that call an sbatch script directly instead of wrapping an inline shell command.
- Updated `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` to use the shared helper while preserving staged dry-run output, sbatch export construction, job-state polling, and model-selection flow.
- Updated `self/README.md` to document script-job submission/dry-run handling in the shared launcher helper and the rectangular seed-sweep submitter.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh launchers/self/run_multiplication_rectangular_seed_mig.sbatch`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rect_seed_submit tests/test_multiplication_rectangular_seed_launchers.py -q` (`5 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 09:38:28 UTC

- Removed the remaining tracked historical plan wording that described the removed auxiliary classification task as part of the old workshop cleanup trail.
- Updated `self/README.md` to state the supported workshop/main-track task surface directly: addition, run-length, and multiplication.
- Verification: the tracked grep checks for the removed task name and related removed-selection wording returned no tracked matches; `git diff --check`.

### Implementation Log: 2026-06-18 09:44:35 UTC

- Reduced repeated packed-local candidate worker bootstrap work by creating a per-pack `ModelBootstrapCache` whenever shared packed-worker inputs are reused.
- Kept CPU checkpoint-state caching gated behind `--candidate-local-cache-base-state`, but now packed workers get tokenizer bootstrap reuse by default without changing candidate training/evaluation semantics.
- Added detailed bootstrap cache hit/miss counters to `ModelBootstrapCache` and included them in packed-worker summaries under `model_bootstrap_cache_details`, while preserving the existing compact `model_bootstrap_cache` summary shape.
- Updated `self/README.md` runtime notes to separate tokenizer bootstrap reuse from optional checkpoint-state reuse.
- Verification: `python -m py_compile self/core/model_io.py self/core/candidate_worker_runtime.py tests/test_model_io_bootstrap_cache.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_model_cache tests/test_model_io_bootstrap_cache.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_candidate_cache tests/test_adaptive_candidate_training.py -q` (`39 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 09:50:00 UTC

- Continued shrinking `self/core/driver.py` by moving the repetitive public compatibility delegate registration into `self/core/driver_public_api.py`.
- Kept the live driver module as the binding surface for all generated delegates so existing monkeypatches through `self.adaptive_candidate_training` and `self.core.driver` still flow into candidate dispatch, worker entry points, proposal-GRPO dispatch, and run orchestration.
- Left `driver.py` responsible for lazy default/compat exports, `_default_bf16_on_cuda`, `main(...)`, and module execution only.
- Updated `self/README.md` to document the new public-delegate installer.
- Verification: `python -m py_compile self/core/driver.py self/core/driver_public_api.py self/core/driver_wiring.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py`; `PYTHONPATH=. conda run -n torch-env python - <<'PY' ...` public-delegate probe; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_public tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`41 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 09:54:13 UTC

- Continued shrinking the old non-adaptive compatibility facade by moving its monkeypatch sync policy into `self/core/nonadaptive_compat.py`.
- Preserved the private `_NONADAPTIVE_PATCHABLE_NAMES` alias on `self/self_improvement_core.py`, but the canonical patchable-name list and sync helper now live in the focused core compatibility module.
- Added `tests/test_nonadaptive_compat.py` to pin the helper's direct behavior, while existing non-adaptive tests continue to verify old facade monkeypatch paths.
- Updated `self/README.md` to document `self/core/nonadaptive_compat.py` and clarify that non-adaptive monkeypatch sync policy has moved out of the facade.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/nonadaptive_compat.py tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_self_improvement_launchers.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive_compat tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_self_improvement_launchers.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py -q` (`44 passed`).

### Implementation Log: 2026-06-18 09:59:38 UTC

- Split bit-string composition-path helpers out of `self/tasks/bit_common.py` into `self/tasks/bit_composition.py`.
- Moved component-size selection and reachable target-size helpers into the new module, while preserving old `self.tasks.bit_common` and `self.self_improvement_tasks` imports as compatibility reexports.
- Updated run-length implementation modules to import composition-path helpers from the new canonical module.
- Added `tests/test_bit_composition.py` to pin direct behavior and old reexport compatibility.
- Updated `self/README.md` to document the new bit-composition task helper.
- Verification: `python -m py_compile self/tasks/bit_composition.py self/tasks/bit_common.py self/tasks/run_length.py self/tasks/run_length_data.py self/self_improvement_tasks.py tests/test_bit_composition.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_bit_composition tests/test_bit_composition.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py -q` (`41 passed`).

### Implementation Log: 2026-06-18 10:04:01 UTC

- Split rectangular multiplication partition aliases/defaults/parsing/grid helpers out of `self/tasks/rectangular_multiplication.py` into `self/tasks/rectangular_partitions.py`.
- Preserved old `self.tasks.rectangular_multiplication` and top-level `self.multiplication_rectangular` imports by reexporting the partition helpers through the existing module.
- Added `tests/test_rectangular_partitions.py` to pin direct helper behavior and old-path reexport compatibility.
- Updated `self/README.md` to document the rectangular partition helper module.
- Verification: `python -m py_compile self/tasks/rectangular_partitions.py self/tasks/rectangular_multiplication.py self/multiplication_rectangular.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rectangular_partitions tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py -q` (`30 passed`).

### Implementation Log: 2026-06-18 10:11:13 UTC

- Split rectangular multiplication digit-order and reverse-CoT helpers out of `self/tasks/rectangular_multiplication.py` into `self/tasks/rectangular_digits.py`.
- The new module owns least-significant-digit ordering, reverse digit text, reverse-CoT prompt/target formatting, trace normalization, final digit extraction, and final value parsing for rectangular diagnostics.
- Preserved old `self.tasks.rectangular_multiplication` and top-level `self.multiplication_rectangular` imports by importing the helper names back into the existing module.
- Added `tests/test_rectangular_digits.py` to pin direct helper behavior and old-path reexport compatibility.
- Updated `self/README.md` with the new rectangular digit helper ownership boundary.
- Verification: `python -m py_compile self/tasks/rectangular_digits.py self/tasks/rectangular_multiplication.py self/multiplication_rectangular.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rectangular_digits tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py -q` (`33 passed`).

### Implementation Log: 2026-06-18 10:17:20 UTC

- Split rectangular multiplication example/data helpers out of `self/tasks/rectangular_multiplication.py` into `self/tasks/rectangular_data.py`.
- The new module owns the rectangular example container, exact-digit integer sampling, sampled partition dataset construction, prediction parsing/normalization, key construction, and prediction-match checks.
- Kept `self.tasks.rectangular_multiplication.build_sampled_rectangular_dataset(...)` as a thin compatibility wrapper that delegates to the new data helper while passing the old module's current `sample_int_with_exact_digits` binding. This preserves existing monkeypatch behavior through `self.multiplication_rectangular.sample_int_with_exact_digits`.
- Added `tests/test_rectangular_data.py` to pin direct data-helper behavior, injected sampler support, and old-path reexport compatibility.
- Updated `self/README.md` with the new rectangular data helper ownership boundary.
- Verification: `python -m py_compile self/tasks/rectangular_data.py self/tasks/rectangular_multiplication.py self/tasks/rectangular_digits.py self/multiplication_rectangular.py tests/test_rectangular_data.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rectangular_data tests/test_rectangular_data.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py -q` (`37 passed`).

### Implementation Log: 2026-06-18 10:24:44 UTC

- Split rectangular multiplication component construction and target-composition helpers out of `self/tasks/rectangular_multiplication.py` into `self/tasks/rectangular_composition.py`.
- The new module owns `RectangularCompositionLeaf`, multiplier-digit components, least-significant block splitting, supported-partition component construction, multiplier-digit target composition, and weighted-component target composition.
- Preserved old `self.tasks.rectangular_multiplication` and top-level `self.multiplication_rectangular` imports by reexporting the composition helpers through the existing module. The old rectangular module is now `78` lines and primarily acts as compatibility import surface plus the sampled-data monkeypatch wrapper.
- Added `tests/test_rectangular_composition.py` to pin direct composition behavior and old-path reexport compatibility.
- Updated `self/README.md` with the new rectangular composition helper ownership boundary.
- Verification: `python -m py_compile self/tasks/rectangular_composition.py self/tasks/rectangular_multiplication.py self/tasks/rectangular_data.py self/tasks/rectangular_digits.py self/multiplication_rectangular.py tests/test_rectangular_composition.py tests/test_rectangular_data.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rectangular_composition tests/test_rectangular_composition.py tests/test_rectangular_data.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py -q` (`41 passed`).

### Implementation Log: 2026-06-18 10:31:19 UTC

- Added `self/core/module_proxy.py`, a reusable compatibility proxy installer for old module paths that should forward missing reads and monkeypatch-style attribute writes to canonical modules.
- Migrated `self/multiplication_rectangular.py` onto the shared module-proxy helper, replacing its local proxy class while preserving old imports and sampler monkeypatch behavior.
- Added `tests/test_module_proxy.py` to pin rectangular old-path exports, `__all__`/`dir()` exposure, and write-forwarding to `self.tasks.rectangular_multiplication`.
- Updated `self/README.md` with the shared proxy helper ownership boundary and wrapper policy.
- Verification: `python -m py_compile self/core/module_proxy.py self/multiplication_rectangular.py tests/test_module_proxy.py tests/test_rectangular_composition.py tests/test_rectangular_data.py tests/test_multiplication_rectangular.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_module_proxy tests/test_module_proxy.py tests/test_rectangular_composition.py tests/test_rectangular_data.py tests/test_rectangular_digits.py tests/test_rectangular_partitions.py tests/test_multiplication_rectangular.py tests/test_multiplication_rectangular_tune.py -q` (`43 passed`).

### Implementation Log: 2026-06-18 10:36:29 UTC

- Migrated the remaining local top-level module-proxy wrappers onto `self/core/module_proxy.py`: `self/addition_recipe.py`, `self/adaptive_candidate_training.py`, `self/adaptive_self_improvement.py`, `self/self_improvement.py`, and `self/self_improvement_composition_error_experiment.py`.
- Preserved each wrapper's canonical target, explicit export list where present, and `__main__` dispatch behavior.
- Kept the legacy addition wrapper's star import while installing the shared proxy, preserving its existing eager globals as well as old monkeypatch-style attribute forwarding.
- Extended `tests/test_module_proxy.py` to cover explicit export-list wrappers in addition to default-export wrappers.
- Updated `self/README.md` so future wrappers use the shared proxy helper instead of reintroducing local proxy classes.
- Verification: `python -m py_compile self/core/module_proxy.py self/addition_recipe.py self/adaptive_candidate_training.py self/adaptive_self_improvement.py self/self_improvement.py self/self_improvement_composition_error_experiment.py tests/test_module_proxy.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proxy_wrappers tests/test_module_proxy.py tests/test_addition_recipe_diagnostic.py::test_addition_recipe_wrapper_points_to_canonical_recipe_helpers tests/test_adaptive_self_improvement_controller.py tests/test_composition_error_sweep.py::test_old_composition_error_module_forwards_attribute_patches tests/test_adaptive_candidate_training.py::test_slurm_array_mode_dispatches_even_single_candidate tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_workers_respect_concurrency_cap tests/test_legacy_addition_self_improvement.py::test_instantiate_model_and_tokenizer_uses_fixed_char_builder tests/test_legacy_addition_self_improvement.py::test_direct_pseudo_can_use_larger_expansion_pool_than_seed_replay -q` (`11 passed`).

### Implementation Log: 2026-06-18 10:40:59 UTC

- Converted `self/self_improvement_experiment.py` from a redirect through `self.self_improvement` into a direct compatibility proxy over `self/legacy/addition_self_improvement.py`.
- Added module-proxy coverage for the old `self.self_improvement_experiment` path, including `main` identity and monkeypatch-style `parse_args` forwarding to the legacy module.
- Updated `self/README.md` so the wrapper map lists `self/self_improvement_experiment.py` explicitly instead of leaving it as an undocumented wrapper-to-wrapper redirect.
- Verification: `python -m py_compile self/self_improvement_experiment.py tests/test_module_proxy.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_self_improvement_experiment tests/test_module_proxy.py tests/test_self_improvement_launchers.py tests/test_legacy_addition_self_improvement.py::test_instantiate_model_and_tokenizer_uses_fixed_char_builder -q` (`8 passed`).

### Implementation Log: 2026-06-18 10:44:55 UTC

- Added `module_star_export_names(...)` to `self/core/module_proxy.py` so wrappers that used `from canonical import *` can preserve Python star-import export policy when migrated to the shared proxy helper.
- Migrated a focused batch of analysis/diagnostic CLI wrappers onto the shared module proxy: `self/addition_recipe_diagnostic.py`, `self/analyze_symbolic_training_dynamics.py`, `self/check_self_improvement_overfit.py`, `self/evaluate_fixed_composition_slices.py`, `self/plot_appendix_baseline_heatmaps.py`, `self/plot_self_improvement_figure.py`, `self/summarize_seed_fit_grid.py`, and `self/run_length_balanced_eval.py`.
- Preserved each wrapper's canonical target and `python -m` dispatch while removing repeated star-import plus `_main` boilerplate.
- Added proxy tests that check the migrated wrappers expose canonical `main` objects and keep `__all__` aligned with star-import semantics.
- Updated `self/README.md` with the star-import wrapper policy.
- Verification: `python -m py_compile self/core/module_proxy.py self/addition_recipe_diagnostic.py self/analyze_symbolic_training_dynamics.py self/check_self_improvement_overfit.py self/evaluate_fixed_composition_slices.py self/plot_appendix_baseline_heatmaps.py self/plot_self_improvement_figure.py self/summarize_seed_fit_grid.py self/run_length_balanced_eval.py tests/test_module_proxy.py`; `bash -n launchers/self/run_addition_tiny_seed_mig.sbatch launchers/self/run_addition_their_recipe_diagnostic.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_star_wrappers tests/test_module_proxy.py tests/test_run_length_balanced_eval.py tests/test_addition_recipe_diagnostic.py tests/test_analysis_artifacts.py tests/test_addition_tiny_seed_mig_launcher.py tests/test_addition_recipe_recovery_launchers.py -q` (`30 passed`).

### Implementation Log: 2026-06-18 10:48:24 UTC

- Migrated the remaining experiment and legacy top-level CLI wrappers onto `self/core/module_proxy.py` with `module_star_export_names(...)`.
- The migrated wrappers are `self/figure2_condition_sweep.py`, `self/figure2_paper_retune.py`, `self/figure3_real_seed_data_ablation.py`, `self/figure3_seed_quality_sweep.py`, `self/multiplication_rectangular_tune.py`, `self/rectangular_multiplication_recipe_seed_fit.py`, `self/rectangular_multiplication_seed_fit.py`, `self/rectangular_multiplication_self_improvement.py`, `self/seed_fit_experiment.py`, `self/multiplication_self_improvement.py`, `self/run_length_self_improvement.py`, and `self/self_improvement_multiplication_cot_pseudo_addition.py`.
- Preserved old import paths, star-import behavior, and `python -m` dispatch for each wrapper while removing repeated `from canonical import *` / `_main` boilerplate.
- Extended `tests/test_module_proxy.py` so the generic star-CLI proxy test covers the experiment and legacy wrapper paths too.
- Updated `self/README.md` to state that current analysis, diagnostic, experiment, and legacy CLI wrappers follow the shared proxy pattern.
- Verification: `python -m py_compile self/figure2_condition_sweep.py self/figure2_paper_retune.py self/figure3_real_seed_data_ablation.py self/figure3_seed_quality_sweep.py self/multiplication_rectangular_tune.py self/rectangular_multiplication_recipe_seed_fit.py self/rectangular_multiplication_seed_fit.py self/rectangular_multiplication_self_improvement.py self/seed_fit_experiment.py self/multiplication_self_improvement.py self/run_length_self_improvement.py self/self_improvement_multiplication_cot_pseudo_addition.py tests/test_module_proxy.py`; `bash -n launchers/self/run_seed_fit_experiment.sbatch launchers/self/submit_seed_fit_grid.sh launchers/self/run_figure2_paper_retune.sh launchers/self/submit_figure2_condition_sweep_mig.sh launchers/self/submit_figure3_seed_quality_sweep_mig.sh launchers/self/submit_figure3_real_seed_data_ablation_mig.sh launchers/self/submit_multiplication_rectangular_tune_mig.sh launchers/self/run_multiplication_rectangular_seed_mig.sbatch launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch launchers/self/run_task_self_improvement.sbatch launchers/self/submit_budget_grid_self_improvement.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_experiment_wrappers tests/test_module_proxy.py tests/test_figure2_condition_sweep.py tests/test_figure2_recipe_aggressive_launchers.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py tests/test_multiplication_rectangular_tune.py tests/test_multiplication_rectangular_seed_launchers.py tests/test_multiplication_rectangular_self_improvement_launchers.py tests/test_multiplication_rectangular.py tests/test_seed_fit_launchers.py tests/test_self_improvement_launchers.py tests/test_task_self_improvement_launchers.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`79 passed`).

### Implementation Log: 2026-06-18 10:51:29 UTC

- Migrated the remaining non-special star wrappers onto `self/core/module_proxy.py`: `self/rectangular_multiplication_compose_diagnostic.py`, `self/seed_fit_curve_notebook_utils.py`, `self/training_curve_notebook_utils.py`, and `self/paper_schedule_selection.py`.
- Extended the generic star-wrapper proxy test to include non-CLI compatibility facades as well as CLI wrappers; the test now checks `__all__` compatibility for all pairs and `main` identity only when the canonical module exposes `main`.
- Verified that the only remaining top-level moved-wrapper star import is the intentionally special `self/self_improvement.py` legacy addition wrapper, which keeps eager globals while forwarding writes through the proxy.
- Updated `self/README.md` to document that exception.
- Verification: `python -m py_compile self/rectangular_multiplication_compose_diagnostic.py self/seed_fit_curve_notebook_utils.py self/training_curve_notebook_utils.py self/paper_schedule_selection.py tests/test_module_proxy.py`; `bash -n launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch launchers/self/run_figure2_paper_retune.sh launchers/self/run_figure2_recipe_aggressive.sh launchers/self/run_addition_recipe_fullpack.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_remaining_wrappers tests/test_module_proxy.py tests/test_seed_fit_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py tests/test_paper_schedule_selection.py tests/test_multiplication_rectangular_square_launchers.py -q` (`25 passed`).

### Implementation Log: 2026-06-18 10:55:35 UTC

- Inspected the candidate-worker packed-local model bootstrap cache path and left runtime semantics unchanged: packed workers already share tokenizer cache by default, optional `--candidate-local-cache-base-state` controls CPU checkpoint-state caching, and existing tests cover both modes.
- Refactored `self/self_improvement_tasks.py` from one flat `__all__` list into grouped compatibility export tuples for protocol/evaluation, addition, bit-common, bit-composition, multiplication, and run-length symbols.
- Added `test_task_facade_all_exports_are_available_and_unique` so the old task facade fails fast if a future edit advertises duplicate or missing legacy exports, and to pin `AdditionTask`, `MultiplicationTask`, and `RunLengthTask` to their canonical task modules.
- Updated `self/README.md` to document the grouped task-facade export surface.
- Verification: `python -m py_compile self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_task_facade tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_passes_tokenizer_cache_without_base_state -q` (`42 passed`).

### Implementation Log: 2026-06-18 10:59:33 UTC

- Refactored `self/self_improvement_core.py` to expose an explicit grouped `__all__` for its legacy compatibility surface, covering module/typing aliases, data IO, evaluation, model IO, recipe helpers, summaries, task protocols, training helpers, and non-adaptive compatibility functions.
- Preserved the old monkeypatch behavior: `run_self_improvement(...)` still syncs facade globals listed in `NONADAPTIVE_PATCHABLE_NAMES` into `self.core.nonadaptive_loop` before dispatch.
- Added `test_nonadaptive_facade_exports_are_available_and_patchable_names_are_public`, which guards duplicate/missing facade exports, requires all non-adaptive patchable names to be public, and pins representative exports to canonical `self/core/*` modules.
- Updated `self/README.md` to document the grouped core-facade export surface and patchable-name guard.
- Verification: `python -m py_compile self/self_improvement_core.py self/core/nonadaptive_compat.py tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_self_improvement_launchers.py tests/test_run_length_recipe.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_core_facade tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_self_improvement_launchers.py tests/test_run_length_recipe.py tests/test_self_improvement_tasks.py -q` (`46 passed`).

### Implementation Log: 2026-06-18 11:07:21 UTC

- Extended `self/analysis/adaptive_artifacts.py` with stable per-candidate artifact loaders: `AdaptiveCandidateArtifacts`, `iter_candidate_dirs(...)`, `load_adaptive_candidate(...)`, `load_adaptive_candidates(...)`, and `adaptive_candidate_artifact_records(...)`.
- Centralized adaptive candidate artifact filenames for metrics, train-mix summaries, and worker-failure files so notebooks do not need to hard-code `attempt_*/candidates/candidate_*` layouts when inspecting candidate artifacts.
- Refactored `adaptive_candidate_train_mix_records(...)` to use the candidate artifact loader while preserving the existing row schema and artifact paths.
- Reexported the new candidate artifact helpers through `self/analysis/artifacts.py` for the notebook compatibility surface.
- Extended `tests/test_analysis_artifacts.py` with a failure-only candidate directory to cover incomplete candidate artifacts, worker-failure loading, candidate sorting, train-mix loading, and compatibility identity for the new loader.
- Updated `self/README.md` to document candidate artifact loading in the current analysis layer and remaining cleanup guidance.
- Verification: `python -m py_compile self/analysis/adaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_candidate_artifacts tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`10 passed`); `git diff --check`. The temporary `.pytest_tmp_analysis_candidate_artifacts` directory was removed after verification.

### Implementation Log: 2026-06-18 11:11:00 UTC

- Continued Slurm launcher consolidation by adding two generic helpers to `launchers/self/lib/self_common.sh`: `self_print_python_launcher_context(...)` for the common root/Python/output-root banner and `self_print_and_run_command_stdout(...)` for the repeated print-then-execute command pattern.
- Migrated the paper-facing Figure 2/3 wrappers to these helpers: `submit_figure2_condition_sweep_mig.sh`, `submit_figure3_seed_quality_sweep_mig.sh`, `submit_figure3_real_seed_data_ablation_mig.sh`, and `run_figure2_paper_retune.sh`.
- Preserved the existing command arrays, dry-run flag insertion, output paths, figure paths, selection-env arguments, and Python module entry points; this pass only removes duplicated shell boilerplate.
- Updated `self/README.md` to document that `self_common.sh` now owns generic context-banner and print-then-execute command helpers for launcher wrappers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_figure2_condition_sweep_mig.sh launchers/self/submit_figure3_seed_quality_sweep_mig.sh launchers/self/submit_figure3_real_seed_data_ablation_mig.sh launchers/self/run_figure2_paper_retune.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_launcher_common tests/test_figure2_condition_sweep.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py tests/test_figure2_recipe_aggressive_launchers.py -q` (`21 passed`).

### Implementation Log: 2026-06-18 11:14:47 UTC

- Extracted non-adaptive round iteration from `self/core/nonadaptive_loop.py` into `self/core/nonadaptive_round_loop.py`.
- The new helper owns ordered round execution, injected dependency forwarding into `run_nonadaptive_round(...)`, round-directory collection for checkpoint finalization, and early-stop handling.
- Kept the legacy patchable dependency surface in `self/core/nonadaptive_loop.py`: `self.self_improvement_core` still syncs patched globals into that module, and the loop passes those current globals through to the extracted round-loop helper.
- Added `tests/test_nonadaptive_round_loop.py` for dependency forwarding, early-stop behavior, and empty-loop behavior.
- Updated `self/README.md` to document the new non-adaptive round-loop owner and to narrow `nonadaptive_loop.py` to setup/bootstrap/finalization wiring.
- Verification: `python -m py_compile self/core/nonadaptive_round_loop.py self/core/nonadaptive_loop.py tests/test_nonadaptive_round_loop.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_nonadaptive_round_loop tests/test_nonadaptive_round_loop.py tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_nonadaptive_results.py tests/test_nonadaptive_pseudo.py -q` (`13 passed`).

### Implementation Log: 2026-06-18 11:19:47 UTC

- Extracted candidate-worker shared input loading and packed-worker cache setup from `self/core/candidate_worker_runtime.py` into `self/core/candidate_worker_inputs.py`.
- The new module owns `CandidateWorkerRuntimeDeps`, `CandidateWorkerSharedInputs`, shared-input cache keys, source/eval example loading, proposal/outcome trace-buffer loading, proposal prompt reconstruction, model bootstrap cache construction, and candidate payload-to-work-item reconstruction.
- Kept old imports stable by reexporting the dependency/shared-input classes through `candidate_worker_runtime.py`; `worker_entrypoints.py` can continue importing `CandidateWorkerRuntimeDeps` from the runtime module.
- Added `tests/test_candidate_worker_inputs.py` to pin shared artifact reuse, model-bootstrap cache flag propagation, prompt/example/trace loading, and validation-call behavior without requiring model construction.
- Updated `self/README.md` to document `candidate_worker_inputs.py` ownership and the packed-worker optimization path.
- Verification: `python -m py_compile self/core/candidate_worker_inputs.py self/core/candidate_worker_runtime.py tests/test_candidate_worker_inputs.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_candidate_worker_inputs tests/test_candidate_worker_inputs.py tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_passes_tokenizer_cache_without_base_state tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py -q` (`12 passed`).

### Implementation Log: 2026-06-18 11:23:37 UTC

- Extracted candidate-worker failure artifact construction from `self/core/candidate_worker_runtime.py` into `self/core/candidate_worker_failures.py`.
- The new helper owns failure payload construction, candidate worker-failure path resolution via `worker_io.candidate_worker_failure_path(...)`, and injected IO writing from a candidate spec payload.
- Replaced the duplicated single-worker and packed-worker failure-writing blocks in `candidate_worker_runtime.py` with calls through the shared helper.
- Added `tests/test_candidate_worker_failures.py` for payload construction, zero-padded candidate failure paths, and injected load/write behavior.
- Updated `self/README.md` to document `candidate_worker_failures.py` ownership and narrow the candidate-worker runtime description.
- Verification: `python -m py_compile self/core/candidate_worker_failures.py self/core/candidate_worker_runtime.py tests/test_candidate_worker_failures.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_candidate_worker_failures tests/test_candidate_worker_failures.py tests/test_candidate_metric_collection.py tests/test_adaptive_candidate_training.py::test_local_parallel_candidate_worker_failure_becomes_metric tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_passes_tokenizer_cache_without_base_state -q` (`10 passed`).

### Implementation Log: 2026-06-18 11:29:48 UTC

- Continued Slurm launcher consolidation by moving colon-separated config-file sourcing into the generic `launchers/self/lib/self_common.sh` helper layer.
- `self_common.sh` now owns `self_source_config_file(...)` and `self_source_config_files(...)`, including relative path resolution against `ROOT_DIR`, missing-file diagnostics, labeled source announcements, and colon-separated config lists.
- Reduced `launchers/self/lib/adaptive_common.sh` to compatibility wrappers around the generic helpers while preserving the old adaptive output text, including `[INFO] Loaded adaptive config: ...` and `[ERROR] Missing adaptive config file: ...`.
- Added `tests/test_self_common_launcher_helpers.py` for generic config sourcing and missing-file diagnostics, and extended the adaptive candidate launcher test to pin the adaptive config load announcement.
- Updated `self/README.md` to document that generic launcher helpers own config-file sourcing and adaptive launchers only add adaptive-labeled wrappers.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch launchers/self/submit_adaptive_candidate_training_ailab.sh`; `python -m py_compile tests/test_self_common_launcher_helpers.py tests/test_adaptive_candidate_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_launcher_config tests/test_self_common_launcher_helpers.py tests/test_adaptive_candidate_launcher.py tests/test_adaptive_condition_launcher.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 11:34:16 UTC

- Extracted packed candidate-worker execution from `self/core/candidate_worker_runtime.py` into `self/core/candidate_worker_pack_runtime.py`.
- The new module owns per-pack shared input-cache dispatch, backward-compatible runners that do not accept `shared_cache`, per-candidate failure artifact recording inside a pack, packed worker success/failure summaries, and model-bootstrap cache stats/details reporting.
- Kept old import compatibility by reexporting `run_candidate_worker_pack_from_spec(...)` through `self/core/candidate_worker_runtime.py`, while `self/core/worker_entrypoints.py` now imports packed execution from the new owner module.
- Added `tests/test_candidate_worker_pack_runtime.py` for cache-aware packed runners, legacy no-cache runners, and the old runtime reexport.
- Updated `self/README.md` to document the packed-worker runtime owner and narrow `candidate_worker_runtime.py` to single-spec execution plus compatibility.
- Verification: `python -m py_compile self/core/candidate_worker_pack_runtime.py self/core/candidate_worker_runtime.py self/core/worker_entrypoints.py tests/test_candidate_worker_pack_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_candidate_worker_pack_runtime tests/test_candidate_worker_pack_runtime.py tests/test_candidate_worker_inputs.py tests/test_candidate_worker_failures.py tests/test_adaptive_candidate_training.py::test_candidate_worker_spec_roundtrip_loads_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_reuses_shared_inputs tests/test_adaptive_candidate_training.py::test_candidate_pack_worker_passes_tokenizer_cache_without_base_state tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py -q` (`17 passed`).

### Implementation Log: 2026-06-18 11:39:45 UTC

- Split candidate-level adaptive artifact helpers out of `self/analysis/adaptive_artifacts.py` into `self/analysis/adaptive_candidate_artifacts.py`.
- The new module owns `AdaptiveCandidateArtifacts`, candidate directory discovery, candidate metrics/train-mix/worker-failure loading, and candidate/candidate-artifact/train-mix/per-size row flatteners.
- Preserved old notebook import paths by reexporting the moved names from `self/analysis/adaptive_artifacts.py`; the umbrella `self/analysis/artifacts.py` now imports candidate helpers from the new owner module.
- Extended `tests/test_analysis_artifacts.py` to pin the new owner module and the old compatibility aliases.
- Updated `self/README.md` to document the split between run/attempt adaptive loaders and candidate-level adaptive loaders.
- Verification: `python -m py_compile self/analysis/adaptive_artifacts.py self/analysis/adaptive_candidate_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_candidate_split tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`10 passed`).

### Implementation Log: 2026-06-18 11:44:29 UTC

- Extracted shared adaptive artifact constants, dataclasses, and row helpers into `self/analysis/adaptive_artifact_common.py`.
- The common module now owns `DEFAULT_ADAPTIVE_TRACE_FILES`, candidate artifact filename constants, `AdaptiveAttemptArtifacts`, `AdaptiveRunArtifacts`, attempt/candidate index parsing, run-context row fields, selected-id handling, proposal-field flattening, selected-payload selection, and candidate metric fallback lookup.
- Updated `self/analysis/adaptive_artifacts.py` and `self/analysis/adaptive_candidate_artifacts.py` to depend on the neutral common layer instead of having candidate helpers import private symbols from the run-level loader.
- Preserved compatibility aliases from `self/analysis/adaptive_artifacts.py` and `self/analysis/artifacts.py`; tests now pin that the old public dataclass aliases point at the common owner.
- Updated `self/README.md` to document the common adaptive artifact module and its ownership.
- Verification: `python -m py_compile self/analysis/adaptive_artifact_common.py self/analysis/adaptive_artifacts.py self/analysis/adaptive_candidate_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_artifact_common tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`10 passed`); import sanity check confirmed `self.analysis.artifacts`, `self.analysis.adaptive_artifacts`, and `self.analysis.adaptive_candidate_artifacts` expose the common dataclasses/helpers consistently.

### Implementation Log: 2026-06-18 11:48:59 UTC

- Split proposal-row generation from `self/core/proposal_runtime.py` into `self/core/proposal_generation.py`.
- The new module owns `_rows_for_round(...)`, fixture proposal slicing by selected round/attempt, current-model proposal sampling, alternate proposal-model loading, and proposal-model CUDA cleanup after generation.
- Kept `proposal_runtime.py` focused on executable proposal validation and program repair while reexporting the moved generation helpers for old imports and monkeypatches.
- Updated current implementation imports so `controller_phase_runtime.py` and `driver_default_bindings.py` use `self/core/proposal_generation.py` directly where appropriate.
- Added `tests/test_proposal_generation.py` for fixture row selection, candidate limiting/copying, and old `proposal_runtime` compatibility aliases.
- Updated `self/README.md` to document the proposal-generation owner and narrow `proposal_runtime.py` to validation/repair plus compatibility reexports.
- Verification: `python -m py_compile self/core/proposal_generation.py self/core/proposal_runtime.py self/core/controller_phase_runtime.py self/core/driver_default_bindings.py tests/test_proposal_generation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_generation tests/test_proposal_generation.py tests/test_adaptive_candidate_training.py -k 'parser_defaults or program_pseudo_labels or policy_validation or meta_validation or candidate_worker_spec_roundtrip_loads_inputs' tests/test_adaptive_self_improvement_controller.py -q` (`5 passed`, `39 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_controller tests/test_adaptive_self_improvement_controller.py -q` (`2 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_full tests/test_proposal_generation.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`49 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 11:56:20 UTC

- Split executable proposal validation and repair from `self/core/proposal_runtime.py` into `self/core/proposal_executable_validation.py`.
- The new module owns executable-code extraction, raw row payload parsing, fixture repair-output parsing, repair generation through the current model, program/policy/meta range checks, sandbox validation with repair, duplicate marking, and executable proposal row construction.
- Kept `proposal_runtime.py` as the config-vs-executable validation dispatcher and compatibility reexport surface for old imports and monkeypatches.
- Added `tests/test_proposal_executable_validation.py` for code extraction, repair-output fixture parsing, rejection of malformed repair fixtures, and old `proposal_runtime` compatibility aliases.
- Updated `self/README.md` to document the executable-validation owner and narrow `proposal_runtime.py` to dispatch plus compatibility reexports.
- Verification: `python -m py_compile self/core/proposal_executable_validation.py self/core/proposal_runtime.py tests/test_proposal_executable_validation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_exec_validation tests/test_proposal_executable_validation.py tests/test_proposal_generation.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`52 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:01:55 UTC

- Canonicalized remaining proposal imports in implementation code after the generation/executable-validation split.
- `self/core/attempt_prompt_runtime.py` and `self/core/driver_default_bindings.py` now import prompt helpers directly from `self/core/proposal_prompts.py`; `driver_default_bindings.py` keeps only the config-vs-executable validation dispatcher import from `proposal_runtime.py`.
- `self/core/driver_compat_exports.py` now reexports proposal symbols from their owner modules (`proposal_generation`, `proposal_config_validation`, `proposal_executable_validation`, `proposal_prompts`, and `proposals`) rather than through `proposal_runtime.py`.
- Extended `tests/test_proposal_generation.py` to pin canonical ownership for prompt helpers, proposal generation, executable validation, and compatibility exports.
- Updated `self/README.md` to state that new implementation code should import proposal helpers from owner modules rather than the `proposal_runtime.py` compatibility wrapper.
- Verification: `python -m py_compile self/core/attempt_prompt_runtime.py self/core/driver_default_bindings.py self/core/driver_compat_exports.py tests/test_proposal_generation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_imports tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py -k 'parser_defaults or program_pseudo_labels or policy_validation or meta_validation or candidate_worker_spec_roundtrip_loads_inputs' tests/test_adaptive_self_improvement_controller.py -q` (`5 passed`, `43 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_imports_full tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`53 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:10:31 UTC

- Audited tracked files for the old auxiliary classification task using exact-name grep across source and notebooks. The audit returned no tracked hits before this cleanup entry, so `main` already excludes that task's code and notebooks. Old untracked notebooks/artifacts may still mention it locally, but they are outside the git repository state and were left untouched.
- Split config/program pseudolabel composition out of `self/core/composition.py` into `self/core/composition_pseudolabels.py`.
- `self/core/composition.py` now owns exact-pair addition/run-length dataset construction, composition guards, and run-length example merging while preserving old compatibility reexports for pseudolabel helper imports.
- `self/core/driver_compat_exports.py` now imports pseudolabel helper compatibility exports from the new canonical owner module.
- Added `tests/test_composition_pseudolabels.py` to pin old `self.core.composition` and driver compatibility aliases.
- Verification: `python -m py_compile self/core/composition.py self/core/composition_pseudolabels.py self/core/driver_compat_exports.py tests/test_composition_pseudolabels.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_composition_pseudolabels tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py -k 'exact_pair or program_pseudo_labels or policy_validation or meta_validation or pseudolabel' -q` (`7 passed`, `34 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_composition_adaptive tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`43 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:16:40 UTC

- Split sandbox data models out of `self/core/program_sandbox.py` into `self/core/program_sandbox_models.py`.
- Split task-specific sandbox oracle/property cases out of `self/core/program_sandbox.py` into `self/core/program_sandbox_cases.py`.
- Kept `self/core/program_sandbox.py` as the bounded validation/execution engine and compatibility surface, reexporting `SandboxCase`, program result dataclasses, and `build_*_program_cases(...)` for old imports.
- Updated implementation imports in proposal prompts, executable validation, composition pseudolabels, and the adaptive experiment module to use the new owner modules where appropriate.
- Added `tests/test_program_sandbox_modules.py` to pin direct owner modules and old sandbox reexports.
- Verification: `python -m py_compile self/core/program_sandbox.py self/core/program_sandbox_models.py self/core/program_sandbox_cases.py self/core/proposal_prompts.py self/core/proposal_executable_validation.py self/core/composition_pseudolabels.py self/experiments/adaptive_self_improvement.py tests/test_program_sandbox_modules.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_program_sandbox_split tests/test_program_sandbox_modules.py tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py -k 'program_sandbox or program_repair or program_pseudo_labels or policy_validation or meta_validation' -q` (`9 passed`, `41 deselected`, `7` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_program_sandbox_full tests/test_program_sandbox_modules.py tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`52 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:22:47 UTC

- Split proposal fixture/trace JSONL IO out of `self/core/proposals.py` into `self/core/proposal_io.py`.
- The new module owns fixture proposal loading, trace row construction, and trace JSONL writing. `self/core/proposals.py` keeps compatibility reexports for old imports and the top-level `self/adaptive_proposals.py` wrapper.
- Updated implementation imports in proposal generation, driver default bindings, trace construction, candidate training mix, worker specs, proposal-GRPO, and the adaptive experiment module to use `self/core/proposal_io.py` directly.
- Extended `tests/test_adaptive_proposals_and_sandbox.py` to pin the old proposal facade aliases against the new owner module.
- Verification: `python -m py_compile self/core/proposal_io.py self/core/proposals.py self/core/proposal_generation.py self/core/driver_default_bindings.py self/core/candidate_training_mix.py self/core/candidate_worker_specs.py self/core/proposal_grpo.py self/core/experience_traces.py self/experiments/adaptive_self_improvement.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_io_split tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_candidate_training_mix.py tests/test_candidate_worker_specs.py tests/test_experience_trace_models.py -q` (`19 passed`, `4` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_io_full tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_candidate_training_mix.py tests/test_candidate_worker_specs.py tests/test_experience_trace_models.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`60 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:31:00 UTC

- Split config proposal schema/parsing helpers out of `self/core/proposals.py` into `self/core/proposal_config_schema.py`.
- The new module owns `ConfigProposal`, `ProposalValidation`, config search spaces, output-schema choices, JSON extraction, action-prediction payload normalization, prediction validation, normalized completions, and config proposal parsing.
- Kept `self/core/proposals.py` and the top-level `self/adaptive_proposals.py` wrapper as compatibility reexport surfaces for old imports.
- Updated implementation imports in args, proposal validation/runtime, prompt construction, models, composition, outcome traces, driver compatibility exports, and the adaptive experiment module to use the new owner module directly.
- Extended `tests/test_adaptive_proposals_and_sandbox.py` to pin old proposal and driver compatibility aliases against the new owner module.
- Verification: `python -m py_compile self/core/proposal_config_schema.py self/core/proposals.py self/core/proposal_config_validation.py self/core/proposal_executable_validation.py self/core/attempt_prompt_runtime.py self/core/args.py self/core/dry_run_runtime.py self/core/proposal_prompts.py self/core/models.py self/core/composition.py self/core/composition_pseudolabels.py self/core/experience_outcome_traces.py self/core/proposal_runtime.py self/core/driver_compat_exports.py self/experiments/adaptive_self_improvement.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_config_schema_split tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py -k 'config_proposal or config_schema or parser_defaults or action_prediction or policy_validation or meta_validation or program_pseudo_labels' -q` (`9 passed`, `44 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_config_schema_full tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_rewards.py tests/test_candidate_training_mix.py tests/test_candidate_worker_specs.py -q` (`65 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:38:29 UTC

- Consolidated proposal prompt rendering ownership in `self/core/proposal_prompts.py`.
- Moved `PromptBundle`, `render_config_prompt(...)`, `render_program_prompt(...)`, and `render_program_repair_prompt(...)` out of `self/core/proposals.py`. The prompt module now owns config prompts, generic program prompts, repair prompts, and executable program/policy/meta candidate prompts.
- Kept `self/core/proposals.py` and the top-level `self/adaptive_proposals.py` wrapper as compatibility reexport surfaces for old prompt imports. `self/core/proposals.py` is now `43` lines and only owns `ProgramProposal` directly.
- Updated core implementation imports and the adaptive experiment module to use `self/core/proposal_prompts.py` directly for prompt types/renderers.
- Extended `tests/test_adaptive_proposals_and_sandbox.py` to pin old prompt aliases and driver compatibility aliases against the new owner module.
- Verification: `python -m py_compile self/core/proposals.py self/core/proposal_prompts.py self/core/proposal_config_schema.py self/core/proposal_executable_validation.py self/core/proposal_runtime.py self/core/driver_compat_exports.py self/core/driver_default_bindings.py self/core/attempt_prompt_runtime.py self/experiments/adaptive_self_improvement.py tests/test_adaptive_proposals_and_sandbox.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_prompt_owner_split tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py -k 'config_proposal or config_schema or prompt_owner or proposal_io or parser_defaults or policy_validation or meta_validation or program_pseudo_labels or program_repair' -q` (`10 passed`, `44 deselected`, `5` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_prompt_owner_full tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_training_mix.py tests/test_candidate_worker_specs.py tests/test_experience_trace_models.py -q` (`69 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:47:31 UTC

- Split outcome-trace rendering and payload extraction from `self/core/experience_outcome_traces.py` into `self/core/experience_outcome_rendering.py`.
- The new rendering module owns compact state/action prompt rendering, candidate payload extraction from proposal rows/metrics, target and prediction parsing, numeric/textual completion formatting, failure-code normalization, and reward-feedback text rendering.
- Kept `self/core/experience_outcome_traces.py` as the trace-construction orchestration layer while reexporting the moved private helpers for older notebooks/tests that imported them through the old module.
- Updated `self/README.md` with the new outcome-trace ownership boundary.
- Extended `tests/test_experience_outcome_traces.py` to pin the compatibility aliases against the new rendering owner module.
- Verification: `python -m py_compile self/core/experience_outcome_rendering.py self/core/experience_outcome_traces.py self/core/experience_traces.py tests/test_experience_outcome_traces.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_outcome_rendering_split tests/test_experience_outcome_traces.py tests/test_experience_trace_models.py -q` (`5 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_outcome_rendering_adaptive tests/test_experience_outcome_traces.py tests/test_experience_trace_models.py tests/test_candidate_training_mix.py tests/test_adaptive_candidate_training.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_worker_specs.py -q` (`55 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 12:54:31 UTC

- Split proposal-GRPO reward shaping and trace construction from `self/core/proposal_grpo.py` into `self/core/proposal_grpo_traces.py`.
- The new trace module owns reward-mode constants, validation-category rewards, outcome-reward shaping, zero-variance advantage handling, `ProposalGRPOTrace`, and `build_proposal_grpo_traces(...)`.
- Kept `self/core/proposal_grpo.py` as the lightweight policy-update/checkpoint-writing owner while reexporting the moved reward and trace helpers for old imports.
- Updated implementation imports in `self/core/args.py` and `self/core/driver_compat_exports.py` to use `self/core/proposal_grpo_traces.py` directly for moved names.
- Added `tests/test_proposal_grpo_traces.py` to pin the new owner module and compatibility reexports.
- Verification: `python -m py_compile self/core/proposal_grpo.py self/core/proposal_grpo_traces.py self/core/args.py self/core/driver_compat_exports.py tests/test_proposal_grpo_traces.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_grpo_traces tests/test_proposal_grpo_traces.py tests/test_adaptive_candidate_training.py -k 'proposal_grpo or parser_defaults' -q` (`5 passed`, `35 deselected`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_proposal_grpo_full tests/test_proposal_grpo_traces.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_training_mix.py tests/test_candidate_worker_specs.py -q` (`49 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 13:01:08 UTC

- Split adaptive argument validation and task-specific default filling from `self/core/args.py` into `self/core/args_normalization.py`.
- Kept `self/core/args.py` as the parser-construction owner and compatibility reexport surface for `normalize_args(...)`, so old driver/notebook imports continue to work.
- The new normalization module owns cross-field validation, proposal-GRPO default handling, task-specific addition/run-length size defaults, derived digit/bit aliases, and frontier minimum derivation.
- Added `tests/test_adaptive_args_normalization.py` to pin the new owner module, old reexport, addition defaults, run-length defaults, and config-vs-program proposal-GRPO step defaults.
- Verification: `python -m py_compile self/core/args.py self/core/args_normalization.py self/core/driver_default_bindings.py self/core/driver_compat_exports.py tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_args_normalization tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_training.py -k 'parser_defaults or args_normalization' -q` (`3 passed`, `38 deselected`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_args_normalization_full tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_candidate_worker_inputs.py tests/test_candidate_worker_pack_runtime.py -q` (`47 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 13:08:48 UTC

- Split self-improvement recipe presets from `self/core/recipes.py` into `self/core/recipe_presets.py`.
- The new preset module owns recipe constants, supported recipe names, `RecipeTrainingPhaseConfig`, `SelfImprovementRecipePreset`, addition/self-improvement recipe resolution, phase resolution, `recipe_enabled(...)`, and max-step schedule compression.
- Kept `self/core/recipes.py` as the recipe runtime owner for tokenizer/model construction, trainer variants, training-argument construction, and compatibility reexports for old preset imports.
- Updated clear core implementation imports (`model_io.py`, `training.py`, `nonadaptive_loop.py`, `nonadaptive_setup.py`, `candidate_training_runtime.py`) plus the constant-only run-length diagnostic import to use `recipe_presets.py` directly.
- Added `tests/test_recipe_presets.py` to pin the new owner module, old `recipes.py` reexports, legacy `self.self_improvement_recipe` wrapper, and multiplication schedule compression behavior.
- Verification: `python -m py_compile self/core/recipe_presets.py self/core/recipes.py self/core/model_io.py self/core/training.py self/core/nonadaptive_loop.py self/core/nonadaptive_setup.py self/core/candidate_training_runtime.py self/diagnostics/run_length_balanced_eval.py self/addition_recipe.py self/self_improvement_recipe.py tests/test_recipe_presets.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_recipe_presets tests/test_recipe_presets.py tests/test_addition_recipe_diagnostic.py tests/test_run_length_recipe.py tests/test_multiplication_rectangular.py -k 'recipe or warmup_stable_decay or multiplication_recipe' -q` (`16 passed`, `14 deselected`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_recipe_presets_full tests/test_recipe_presets.py tests/test_addition_recipe_diagnostic.py tests/test_run_length_recipe.py tests/test_multiplication_rectangular.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_training.py tests/test_candidate_training_runtime.py tests/test_model_io_bootstrap_cache.py -q` (`45 passed`).

### Implementation Log: 2026-06-18 13:13:48 UTC

- Split executable program/policy/meta pseudolabel construction from `self/core/composition_pseudolabels.py` into `self/core/composition_program_pseudolabels.py`.
- The new executable module owns component payload construction, sandbox case construction, batch sandbox execution, target-pattern selection, accepted-target validation, and executable pseudolabel diagnostics.
- Kept `self/core/composition_pseudolabels.py` as the config-style addition/run-length pseudolabel owner and config-vs-executable dispatch surface, with compatibility reexports for `compose_program_pseudo_examples(...)` and `target_pattern_for_task(...)`.
- Updated `self/core/composition.py` and `self/core/driver_compat_exports.py` to import executable pseudolabel helpers from the new owner module directly.
- Extended `tests/test_composition_pseudolabels.py` to pin old composition/driver aliases against the new executable pseudolabel owner.
- Verification: `python -m py_compile self/core/composition_program_pseudolabels.py self/core/composition_pseudolabels.py self/core/composition.py self/core/driver_compat_exports.py tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_program_pseudolabel_split tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py -k 'composition_pseudolabel or program_pseudo_labels or policy_validation or meta_validation or exact_pair or pseudolabel' -q` (`7 passed`, `34 deselected`, `3` existing multiprocessing fork warnings); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_program_pseudolabel_full tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py tests/test_proposal_executable_validation.py tests/test_adaptive_proposals_and_sandbox.py -q` (`54 passed`, `7` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 13:24:08 UTC

- Split the remaining explicit legacy export manifests out of the visible compatibility facades.
- Added `self/tasks/compat_exports.py` to own the grouped `self.self_improvement_tasks` compatibility export surface for protocol/evaluation, addition, bit-common, bit-composition, multiplication, and run-length names.
- Added `self/core/nonadaptive_facade_exports.py` to own the grouped `self.self_improvement_core` compatibility export surface for module aliases, typing aliases, data IO, evaluation, model IO, recipe helpers, summaries, task protocols, training helpers, and non-adaptive compatibility helpers.
- Kept `self/self_improvement_tasks.py` and `self/self_improvement_core.py` as thin old-path facades with the same public `__all__` names and the same non-adaptive monkeypatch sync behavior.
- Updated `self/README.md` to document the manifest modules and the thinner facade ownership boundary.
- Verification: `python -m py_compile self/tasks/compat_exports.py self/self_improvement_tasks.py self/core/nonadaptive_facade_exports.py self/self_improvement_core.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_facade_exports tests/test_self_improvement_tasks.py tests/test_nonadaptive_compat.py tests/test_nonadaptive_seed_round_zero.py tests/test_run_length_recipe.py -q` (`43 passed`).

### Implementation Log: 2026-06-18 13:30:26 UTC

- Continued Slurm launcher consolidation by adding `self_submit_sbatch_script(...)` to `launchers/self/lib/self_common.sh` for submitters that launch an sbatch script with shared job/log/export metadata plus extra sbatch arguments.
- Updated `launchers/self/submit_adaptive_candidate_training_ailab.sh` to use the shared sbatch-script helper while preserving its matrix loops, dry-run job IDs, environment export payload, and submission manifest schema.
- Updated `launchers/self/submit_adaptive_condition_pilots_ailab.sh` to use the same helper together with the existing shared resource-default helper while preserving its four condition jobs and manifest schema.
- Added dry-run manifest tests for both adaptive submitters so the shared helper is covered through real launcher entry points.
- Updated `self/README.md` to document that the adaptive candidate and condition-pilot submitters now share sbatch-script submission plumbing.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/lib/adaptive_common.sh launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/submit_adaptive_condition_pilots_ailab.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch launchers/self/run_adaptive_condition_ailab.sbatch`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_launcher_helper tests/test_adaptive_candidate_launcher.py tests/test_adaptive_condition_launcher.py tests/test_main_experiments_launcher.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 13:34:26 UTC

- Added `self/analysis/adaptive_manifest_artifacts.py` as a stable notebook-facing loader for adaptive `submission_manifest.json` files.
- The new loader owns manifest path resolution, recursive manifest discovery, manifest JSON loading, and job-row flattening across candidate-training, condition-pilot, and main-experiment manifest schemas (`output_dir` and `output_root` are normalized to `output_dir`).
- Reexported the manifest helpers from `self/analysis/artifacts.py` so notebooks can use the existing stable analysis import surface instead of hard-coding raw manifest paths.
- Extended `tests/test_analysis_artifacts.py` with manifest fixtures covering Slurm metadata, adaptive candidate metadata, and main-experiment-style output-root metadata.
- Updated `self/README.md` to document the new analysis owner module.
- Verification: `python -m py_compile self/analysis/adaptive_manifest_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_manifest_artifacts tests/test_analysis_artifacts.py -q` (`5 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 13:40:37 UTC

- Split packed-local model bootstrap cache containers/helpers from `self/core/model_io.py` into `self/core/model_bootstrap_cache.py`.
- The new module owns `TokenizerBootstrap`, `CachedModelState`, `ModelBootstrapCache`, tokenizer/cache-key helpers, and CPU state-dict cloning.
- Kept `self/core/model_io.py` as the model/tokenizer loading owner and compatibility reexport surface for old cache imports.
- Updated candidate scoring/training/worker-input modules to import `ModelBootstrapCache` from the new owner module.
- Extended cache tests to pin old `model_io.py` aliases against the new owner.
- Verification: `python -m py_compile self/core/model_bootstrap_cache.py self/core/model_io.py self/core/candidate_worker_inputs.py self/core/candidate_training_runtime.py self/core/candidate_scoring.py tests/test_model_io_bootstrap_cache.py tests/test_candidate_worker_inputs.py tests/test_candidate_worker_pack_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_model_bootstrap_cache tests/test_model_io_bootstrap_cache.py tests/test_candidate_worker_inputs.py tests/test_candidate_worker_pack_runtime.py tests/test_candidate_training_runtime.py -q` (`10 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 13:48:09 UTC

- Split run-length split preparation and composed-eval guard slicing from `self/tasks/run_length.py` into `self/tasks/run_length_splits.py`.
- The new module owns initial split construction, composed train/eval construction, eval-example construction, fixed-binary target-size handling, and guard-slice partitioning for run-length composed evals.
- Kept `RunLengthTask` as the task adapter and metadata/pseudolabel orchestration owner; its public methods now delegate to the split helper module without changing call signatures.
- Reduced `self/tasks/run_length.py` from `479` to `345` lines in this pass, after earlier logic/data/pseudolabel extractions.
- Updated `self/README.md` to document `self/tasks/run_length_splits.py`.
- Verification: `python -m py_compile self/tasks/run_length.py self/tasks/run_length_splits.py self/tasks/run_length_data.py self/tasks/run_length_pseudolabels.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_run_length_splits tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`82 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 13:56:53 UTC

- Split multiplication split preparation and composed-eval overlap/carry slicing from `self/tasks/multiplication.py` into `self/tasks/multiplication_splits.py`.
- The new module owns seed/long initial split construction, composed train/eval construction, eval-example construction, composed-size rounding to block boundaries, and overlap/carry slice partitioning.
- Kept `MultiplicationTask` as the task adapter and blocked-component pseudolabel orchestration owner; its public methods now delegate to the split helper module without changing call signatures.
- Reduced `self/tasks/multiplication.py` from `449` to `367` lines in this pass, while preserving old helper imports through `self.tasks.multiplication`.
- Added a direct task-adapter test covering multiplication initial/composed/eval split delegation and composed-eval slice partitioning.
- Updated `self/README.md` to document `self/tasks/multiplication_splits.py`.
- Verification: `python -m py_compile self/tasks/multiplication.py self/tasks/multiplication_splits.py self/tasks/multiplication_data.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_multiplication_rectangular.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_multiplication_splits tests/test_self_improvement_tasks.py tests/test_multiplication_rectangular.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`93 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 14:02:15 UTC

- Split run-length guarded plain-output and symbol-pair pseudolabel derivation from `self/tasks/run_length_pseudolabels.py` into `self/tasks/run_length_guarded_pseudolabels.py`.
- The new module owns guarded component prediction evaluation, no-boundary filtering, unfiltered pair composition, guarded refill construction, and retained/missing diagnostics for the guarded pair modes.
- Kept `self/tasks/run_length_pseudolabels.py` as the run-length pseudolabel dispatcher plus direct, run-state, and default tuple pseudolabel owner; it reimports the moved guarded helper as `_derive_guarded_pair_pseudo` for old private imports.
- Reduced `self/tasks/run_length_pseudolabels.py` from `445` to `324` lines in this pass.
- Added a compatibility test pinning the old private helper alias to the new owner.
- Updated `self/README.md` to document `self/tasks/run_length_guarded_pseudolabels.py`.
- Verification: `python -m py_compile self/tasks/run_length_pseudolabels.py self/tasks/run_length_guarded_pseudolabels.py self/tasks/run_length.py self/tasks/run_length_splits.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_run_length_guarded_pseudolabels tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`84 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 14:07:07 UTC

- Split training-curve per-size heatmap plotting from `self/analysis/training_curve_plots.py` into `self/analysis/training_curve_heatmaps.py`.
- The new module owns sparse tick/annotation helpers, bundle-backed per-size heatmaps, direct results-file heatmaps, paper-layout controls, and heatmap colorbar labeling.
- Kept `self/analysis/training_curve_plots.py` as the task-curve/comparison/export owner and compatibility reexport surface for old heatmap imports.
- Updated `self/analysis/training_curve_notebook_utils.py` to import heatmap helpers from the canonical heatmap module while preserving object identity through the old plotting module.
- Reduced `self/analysis/training_curve_plots.py` from `455` to `220` lines in this pass.
- Added a compatibility test pinning the top-level notebook utility heatmap helper to both the old plotting path and the new heatmap owner.
- Updated `self/README.md` to document `self/analysis/training_curve_heatmaps.py`.
- Verification: `python -m py_compile self/analysis/training_curve_heatmaps.py self/analysis/training_curve_plots.py self/analysis/training_curve_notebook_utils.py self/training_curve_notebook_utils.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_training_curve_heatmaps tests/test_training_curve_notebook_utils.py tests/test_analysis_artifacts.py -q` (`11 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:13:42 UTC

- Split shared attempt-outcome containers from `self/core/attempt_outcome_runtime.py` into `self/core/attempt_outcome_models.py`.
- Split the no-selection attempt branch into `self/core/attempt_no_selection_runtime.py`; it owns no-selection patience handling, proposal-GRPO retry updates, checkpoint replacement cleanup, failure summaries, and no-selection stop decisions.
- Kept `self/core/attempt_outcome_runtime.py` as the public outcome orchestrator and selected-candidate branch owner while reexporting `AttemptOutcomeDeps`, `AttemptOutcomeResult`, and `_handle_no_selection_attempt` for old imports/tests.
- Added `tests/test_attempt_outcome_runtime.py` to pin compatibility aliases and exercise the no-selection public handler path, including summary artifacts and proposal-GRPO retry metrics.
- Updated `self/README.md` with the narrower ownership boundary for attempt-outcome modules.
- Verification: `python -m py_compile self/core/attempt_outcome_models.py self/core/attempt_no_selection_runtime.py self/core/attempt_outcome_runtime.py tests/test_attempt_outcome_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_attempt_outcome tests/test_attempt_outcome_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`43 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 14:19:23 UTC

- Split the selected-candidate attempt branch from `self/core/attempt_outcome_runtime.py` into `self/core/attempt_selected_runtime.py`.
- The new module owns selected proposal-trace buffering, source-pool mutation, selected pseudo-example artifact writing, selected-round summary writing, checkpoint replacement cleanup, and optional post-selection proposal-GRPO updates.
- Kept `self/core/attempt_outcome_runtime.py` as the branch router plus shared candidate/outcome-trace artifact writer while reexporting `_handle_selected_attempt` for old imports/tests.
- Extended `tests/test_attempt_outcome_runtime.py` with a selected-candidate public-handler smoke test covering source sizes, exclusion keys, selected trace buffering, selected pseudo-example JSONL, and selected-round summary artifacts.
- Updated `self/README.md` with the selected-attempt runtime ownership boundary.
- Verification: `python -m py_compile self/core/attempt_selected_runtime.py self/core/attempt_outcome_runtime.py tests/test_attempt_outcome_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_attempt_selected tests/test_attempt_outcome_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`44 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 14:25:11 UTC

- Split the model-backed adaptive attempt body from `self/core/attempt_loop_runtime.py` into `self/core/attempt_candidate_runtime.py`.
- The new module owns round-model dispatch, candidate metric training, candidate selection, trace writing, unselected checkpoint cleanup, and handoff to attempt-outcome handling for non-dry-run attempts.
- Kept `self/core/attempt_loop_runtime.py` as the selected-round iteration owner across attempt directory setup, prompt construction, dry-run routing, candidate-attempt dispatch, and loop-state updates.
- Added `tests/test_attempt_candidate_runtime.py` to pin the new helper's call ordering and argument forwarding, including updated round-model accuracies into candidate training and outcome handling.
- Updated `self/README.md` with the candidate-attempt runtime ownership boundary.
- Verification: `python -m py_compile self/core/attempt_candidate_runtime.py self/core/attempt_loop_runtime.py tests/test_attempt_candidate_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_attempt_candidate tests/test_attempt_candidate_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`42 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 14:29:53 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_addition_fullpack_filtered_mig.sh` onto `self_submit_sbatch_script(...)`.
- The submitter now uses explicit per-baseline job names, stdout/stderr log paths, and `--export ALL,BASELINE=...,OUT_ROOT=...` payloads while preserving the five-baseline matrix and output-root layout.
- Kept the existing dry-run baseline listing and added dry-run assertions for the shared helper's generated Slurm metadata/export command.
- Updated `self/README.md` to document that the addition fullpack filtered submitter now uses the shared sbatch-script submission helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_addition_fullpack_filtered.sbatch launchers/self/submit_addition_fullpack_filtered_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/addition_fullpack_test LOG_DIR=/tmp/logs bash launchers/self/submit_addition_fullpack_filtered_mig.sh` smoke check showed all five baselines and shared-helper sbatch commands; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_fullpack_submit_helper tests/test_addition_fullpack_filtered_launcher.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:33:33 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` onto `self_submit_sbatch_script(...)`.
- The submitter now uses explicit per-task job names, stdout/stderr log paths, and `--export ALL,TASK=...,OUT_ROOT=...,TRAIN_BATCH_SIZE=...,EVAL_BATCH_SIZE=...,SEED=...,SYMBOL_ALPHABET_SIZE=...,DRY_RUN=...` payloads while preserving the task matrix and output-root layout.
- Kept the existing dry-run `sbatch not executed` message and added dry-run assertions for the shared helper's generated Slurm metadata/export command.
- Updated `self/README.md` to document that the guarded diagnostic submitter now uses the shared sbatch-script submission helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/guarded_diag_submit_test LOG_DIR=/tmp/logs TASKS='run_length' bash launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` smoke check showed the run-length task and shared-helper sbatch command; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_guarded_submit_helper tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:37:10 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` onto `self_submit_sbatch_script(...)`.
- The submitter now uses explicit per-baseline job names, stdout/stderr log paths, and the same `--export ALL,OUT_ROOT=...,BASELINE=...,SEED_MODEL=...` payload fields as before while preserving the baseline matrix and output-root layout.
- Preserved the dry-run `job_id=dryrun` status shape and added dry-run assertions for the shared helper's generated Slurm metadata/export command.
- Updated `self/README.md` to document that the multiplication rectangular fullpack submitter now uses the shared sbatch-script submission helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch launchers/self/submit_multiplication_rectangular_fullpack_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/mult_rect_pack_test LOG_DIR=/tmp/logs BASELINES='short_only direct compose compose_corrupt' SEED_MODEL=/tmp/missing_seed_ok bash launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` smoke check showed all four baselines and shared-helper sbatch commands; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_mult_rect_fullpack_submit_helper tests/test_multiplication_rectangular_self_improvement_launchers.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:40:13 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_multiplication_rectangular_square_probe_mig.sh` onto `self_submit_sbatch_script(...)`.
- The submitter now uses explicit seed/diagnostic job names, stdout/stderr log paths, and the same export payload fields as before while preserving the seed output, diagnostic output, summary paths, and dry-run job IDs.
- Preserved the diagnostic dependency by passing `--dependency afterok:${seed_job_id}` through the shared helper.
- Updated `self/README.md` to document that the square probe submitter now uses the shared sbatch-script submission helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch launchers/self/submit_multiplication_rectangular_square_probe_mig.sh launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/square_probe_test LOG_DIR=/tmp/logs bash launchers/self/submit_multiplication_rectangular_square_probe_mig.sh` smoke check showed the seed job, dependent diagnostic job, and shared-helper sbatch commands; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_square_probe_submit_helper tests/test_multiplication_rectangular_square_launchers.py tests/test_self_common_launcher_helpers.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:43:04 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh` onto `self_submit_sbatch_script(...)`.
- The submitter now uses explicit per-cell job names, stdout/stderr log paths, and the same export payload fields as before while preserving the train-count/learning-rate matrix, output directories, summary paths, and `job_id=dryrun` dry-run status shape.
- Updated `self/README.md` to document that the square seed-resweep submitter now uses the shared sbatch-script submission helper.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch launchers/self/submit_multiplication_rectangular_square_probe_mig.sh launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/square_resweep_test LOG_DIR=/tmp/logs bash launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh` smoke check showed all six train-count/learning-rate cells and shared-helper sbatch commands; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_square_resweep_submit_helper tests/test_multiplication_rectangular_square_launchers.py tests/test_self_common_launcher_helpers.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:49:25 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` onto `self_submit_sbatch_script(...)`.
- The seed-sweep submitter now uses explicit per-stage/per-cell job names, stdout/stderr log paths, and the same `--export ALL,OUT_ROOT=...,LR=...,TRAIN_PER_PARTITION=...,MAX_STEPS=...,TRAIN_BATCH_SIZE=...,EVAL_BATCH_SIZE=...,SEED=...,SAVE_MODEL=...,HELDOUT_PER_PARTITION=...` payload fields as before while preserving the staged search flow, polling, model-selection logic, status file, and dry-run stdout shape.
- Updated dry-run regression coverage to assert the shared helper's generated Slurm metadata/export command in addition to the existing six stage-1 grid lines.
- Updated `self/README.md` to document that the rectangular seed-sweep submitter now uses the shared sbatch-script helper for per-job names, logs, and export payloads.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/run_multiplication_rectangular_seed_mig.sbatch launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/mult_rect_seed_sweep_test LOG_DIR=/tmp/logs bash launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` smoke check showed the stage-0 shared-helper sbatch command and all six stage-1 grid lines; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rect_seed_sweep_submit_helper tests/test_multiplication_rectangular_seed_launchers.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 14:53:53 UTC

- Continued Slurm launcher consolidation by moving `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh` onto the shared `self_submit_wrapped_job(...)` helper.
- The submitter keeps the three-baseline matrix, command arguments, manifest schema, and dry-run manifest IDs while centralizing per-baseline job names, stdout/stderr log paths, Slurm resource arguments, and wrapped command submission.
- Updated dry-run regression coverage to inspect the helper-generated command stream and assert the shared Slurm metadata for the direct baseline.
- Updated `self/README.md` to document that the run-length alpha10 baseline pack submitter now shares wrapped-job submission plumbing.
- Verification: `bash -n launchers/self/lib/self_common.sh launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`; manual `DRY_RUN=1 OUT_ROOT=/tmp/rl-a10-baseline-pack-test LOG_DIR=/tmp/logs bash launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh` smoke check showed all three baselines, stable dry-run manifest IDs, and shared-helper wrapped `sbatch` commands; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rl_baseline_submit_helper tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 15:01:09 UTC

- Split run-length and multiplication prediction parsing out of `self/tasks/bit_common.py` into `self/tasks/bit_parsing.py`.
- The new module owns numeric extraction wrappers, run-length alphabet/target constants, run-length default/symbol-pair/run-state parsers, and multiplication symbolic-target formatting/parsing.
- Kept old `self.tasks.bit_common` and `self.self_improvement_tasks` parser/constant imports compatible by reexporting the moved names, while updating task/core/diagnostic implementation imports to use the canonical parsing module directly.
- Added focused tests for parser behavior and old-path object identity.
- Updated `self/README.md` with the new bit parsing ownership boundary.
- Verification: `python -m py_compile self/tasks/bit_parsing.py self/tasks/bit_common.py self/tasks/compat_exports.py self/tasks/run_length_pseudolabels.py self/tasks/run_length_guarded_pseudolabels.py self/tasks/multiplication.py self/tasks/multiplication_data.py self/tasks/run_length.py self/tasks/run_length_splits.py self/tasks/run_length_logic.py self/core/composition_pseudolabels.py self/core/composition_program_pseudolabels.py self/core/proposal_prompts.py self/core/args.py self/core/driver_compat_exports.py self/diagnostics/evaluate_fixed_composition_slices.py self/diagnostics/run_length_balanced_eval.py self/experiments/rectangular_multiplication_seed_fit.py tests/test_bit_parsing.py tests/test_self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_bit_parsing tests/test_bit_parsing.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_composition_pseudolabels.py tests/test_adaptive_candidate_training.py -q` (`86 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 15:08:18 UTC

- Split frontier value objects out of `self/core/frontier.py` into `self/core/frontier_models.py`.
- Split weak-regime diagnostics parsing, candidate clipping/filtering, and frontier candidate scoring into `self/core/frontier_candidates.py`.
- Kept `self/core/frontier.py` as the public compatibility facade for diagnostics loading, `select_frontier(...)`, `build_frontier_candidates(...)`, frontier dataclasses, and proposal-quality metrics; `self/adaptive_frontier.py` continues to reexport the same public surface.
- Added compatibility tests pinning the old frontier import paths to the new canonical modules.
- Updated `self/README.md` with the new frontier ownership boundaries.
- Verification: `python -m py_compile self/core/frontier_models.py self/core/frontier_candidates.py self/core/frontier.py self/adaptive_frontier.py self/experiments/adaptive_self_improvement.py tests/test_adaptive_frontier.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_frontier_split tests/test_adaptive_frontier.py tests/test_adaptive_self_improvement_controller.py tests/test_adaptive_proposals_and_sandbox.py tests/test_adaptive_candidate_training.py -q` (`53 passed`, `7` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 15:16:54 UTC

- Split training dataset, causal-LM collator, exact-size batch sampler, and batch-sampler Trainer variant out of `self/core/training.py` into `self/core/training_data.py`.
- Kept `self/core/training.py` as the public compatibility facade for those data/batching helpers plus the owner of `TrainingConfig`, training-argument construction, and recipe-aware Trainer construction.
- Preserved old imports through `self.core.training` and `self.self_improvement_core` by reexporting the moved classes.
- Added focused tests for old-path object identity plus tokenized prompt-target masking and collator padding behavior.
- Updated `self/README.md` with the new training-data ownership boundary.
- Verification: `python -m py_compile self/core/training_data.py self/core/training.py self/core/nonadaptive_facade_exports.py self/self_improvement_core.py tests/test_training_data.py tests/test_run_length_recipe.py tests/test_nonadaptive_compat.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_training_data tests/test_training_data.py tests/test_run_length_recipe.py tests/test_nonadaptive_compat.py tests/test_nonadaptive_training.py tests/test_candidate_training_runtime.py tests/test_model_io_bootstrap_cache.py -q` (`17 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 15:28:10 UTC

- Split shared bit pseudolabel and guard-refill helpers out of `self/tasks/bit_common.py` into `self/tasks/bit_pseudolabels.py`.
- The new module owns direct model-prediction pseudolabel construction, run-length boundary guard checks, guard-slice partitioning, guarded/refill pseudolabel construction, and retained-count diagnostics.
- Kept old imports through `self.tasks.bit_common` and `self.self_improvement_tasks` compatible by reexporting moved names, while updating run-length and multiplication implementation imports to use the canonical pseudolabel module directly.
- Added focused tests for old-path object identity, facade monkeypatch behavior, guard partitioning, and guarded pseudolabel diagnostics.
- Updated `self/README.md` with the new bit-pseudolabel ownership boundary.
- Verification: `python -m py_compile self/tasks/bit_pseudolabels.py self/tasks/bit_common.py self/tasks/compat_exports.py self/tasks/run_length_pseudolabels.py self/tasks/run_length_guarded_pseudolabels.py self/tasks/run_length_splits.py self/tasks/multiplication.py tests/test_bit_pseudolabels.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_bit_pseudolabels tests/test_bit_pseudolabels.py tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py tests/test_nonadaptive_pseudo.py tests/test_adaptive_candidate_training.py -q` (`86 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 15:37:55 UTC

- Split recipe runtime/model helpers out of `self/core/recipes.py` into `self/core/recipe_models.py`.
- Split recipe collator, warmup-stable-decay scheduler, recipe Trainer variants, and recipe training-argument construction into `self/core/recipe_training.py`.
- Kept `self/core/recipes.py` as the compatibility facade for old recipe imports and made current core/diagnostic/experiment implementation imports use the canonical owner modules directly.
- Added focused tests pinning `recipes.py` reexports and `self.self_improvement_recipe` compatibility names to the new owner modules.
- Updated `self/README.md` with the new recipe ownership boundaries.
- Verification: `python -m py_compile self/core/recipe_models.py self/core/recipe_training.py self/core/recipes.py self/core/model_io.py self/core/training.py self/core/nonadaptive_bootstrap.py self/core/nonadaptive_lifecycle.py self/core/nonadaptive_loop.py self/core/nonadaptive_round_runtime.py self/core/nonadaptive_facade_exports.py self/core/candidate_training_runtime.py self/diagnostics/addition_recipe_diagnostic.py self/diagnostics/evaluate_fixed_composition_slices.py self/diagnostics/rectangular_multiplication_compose_diagnostic.py self/experiments/seed_fit_experiment.py self/experiments/rectangular_multiplication_self_improvement.py self/experiments/rectangular_multiplication_recipe_seed_fit.py self/addition_recipe.py self/self_improvement_recipe.py tests/test_recipe_presets.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_recipe_runtime_split tests/test_recipe_presets.py tests/test_addition_recipe_diagnostic.py tests/test_run_length_recipe.py tests/test_multiplication_rectangular.py tests/test_nonadaptive_setup.py tests/test_nonadaptive_bootstrap.py tests/test_nonadaptive_training.py tests/test_candidate_training_runtime.py tests/test_model_io_bootstrap_cache.py -q` (`47 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 15:43:30 UTC

- Split adaptive prompt, trace, selected-checkpoint timeline, and proposal-GRPO metric row helpers out of `self/analysis/adaptive_artifacts.py` into `self/analysis/adaptive_trace_artifacts.py`.
- Kept old notebook imports through `self.analysis.adaptive_artifacts` and `self.analysis.artifacts` compatible by reexporting the moved functions, while making the aggregate artifacts facade import from the canonical trace module directly.
- Added focused compatibility assertions so old and notebook-facing imports remain identical to the new owner module.
- Updated `self/README.md` with the new adaptive trace artifact ownership boundary.
- Verification: `python -m py_compile self/analysis/adaptive_trace_artifacts.py self/analysis/adaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `python -m compileall -q self/analysis tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_trace_artifacts tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`11 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 15:48:36 UTC

- Split multiplication round-target pseudolabel derivation out of `self/tasks/multiplication.py` into `self/tasks/multiplication_pseudolabels.py`.
- The new module owns multiplication direct, compose, and corrupt blocked-component pseudolabel derivation, component prediction collection, corruption accounting, and pseudolabel diagnostics.
- Kept `MultiplicationTask.derive_round_targets(...)` as the task protocol entry point and continued injecting the compatibility-aware `generate_prediction_map` wrapper so old `self.self_improvement_tasks` monkeypatch paths still affect pseudolabel generation.
- Added a focused ownership assertion in `tests/test_self_improvement_tasks.py`.
- Updated `self/README.md` with the new multiplication pseudolabel ownership boundary.
- Verification: `python -m py_compile self/tasks/multiplication_pseudolabels.py self/tasks/multiplication.py self/self_improvement_tasks.py tests/test_self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_multiplication_pseudolabels tests/test_self_improvement_tasks.py tests/test_nonadaptive_pseudo.py tests/test_multiplication_rectangular.py tests/test_adaptive_candidate_training.py -q` (`96 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 16:01:22 UTC

- Split multiplication sampling, seed/long dataset construction, blocked-component payload construction, exact-digit sampling, and overlap/carry slice naming out of `self/tasks/multiplication_data.py` into `self/tasks/multiplication_sampling.py`.
- Kept `self/tasks/multiplication_data.py` as the multiplication example/key/override owner and as a compatibility reexport surface for older imports.
- Updated `MultiplicationTask`, multiplication split preparation, task package exports, and `self.self_improvement_tasks` compatibility exports to import sampling helpers from the new canonical owner.
- Added a focused compatibility assertion that old `self.tasks.multiplication_data` sampling imports are identical to the new owner functions.
- Updated `self/README.md` with the new multiplication sampling ownership boundary.
- Verification: `python -m py_compile self/tasks/multiplication_sampling.py self/tasks/multiplication_data.py self/tasks/multiplication.py self/tasks/multiplication_splits.py self/tasks/compat_exports.py self/tasks/__init__.py tests/test_self_improvement_tasks.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_multiplication_sampling tests/test_self_improvement_tasks.py tests/test_nonadaptive_pseudo.py tests/test_multiplication_rectangular.py tests/test_adaptive_candidate_training.py -q` (`97 passed`, `3` existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 16:10:01 UTC

- Added shared launcher helpers in `launchers/self/lib/self_common.sh` for shell-quoted env-wrapped command construction and resource-backed `sbatch --wrap` submission.
- Migrated `submit_addition_exact_digits_fixed_binary_mig.sh` and `submit_addition_fixedwidth_moredata_mig.sh` off local `q()` helpers and hand-built wrapped `sbatch` command arrays while preserving their matrices, manifests, dry-run output, and resource defaults.
- Updated `self/README.md` to document the new generic launcher helper responsibilities and the addition sweep submitter ownership boundary.
- Verification: per-file `bash -n` on `launchers/self/lib/self_common.sh`, `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh`, and `launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_launcher_helpers tests/test_addition_exact_digits_fixed_binary_launcher.py tests/test_addition_fixedwidth_moredata_launcher.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:14:40 UTC

- Migrated the non-dry Slurm path in `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` to the shared sbatch-script helper for the seed job and the shared env-wrapped command builder plus wrapped-job submitter for fixed-binary and original-composition fullpack jobs.
- Preserved the existing dry-run path, branch matrix, `submission_manifest.txt` fields, job names, resource requests, dependency structure, and log paths.
- Added a fake-`sbatch` launcher test so the non-dry helper path is exercised without submitting jobs.
- Updated `self/README.md` with the mixed-submit ownership boundary.
- Verification: per-file `bash -n` on the touched launcher/helper files; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_mixed_launcher_helpers tests/test_addition_fixedwidth_moredata_launcher.py tests/test_addition_exact_digits_fixed_binary_launcher.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:20:41 UTC

- Added `self_wrap_repo_command` to `launchers/self/lib/self_common.sh` for shell-quoted `cd ${ROOT_DIR} && PYTHONPATH=.` command-array wrapping.
- Migrated `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh` off its local `printf -v quoted_root/quoted_run_cmd` block and onto the shared repo-command wrapper before wrapped-job submission.
- Added focused helper coverage for paths/arguments requiring shell quoting, and kept the existing alpha10 baseline dry-run launcher coverage.
- Updated `self/README.md` with the new generic launcher helper responsibility and alpha10 baseline submitter boundary.
- Verification: per-file `bash -n` on `launchers/self/lib/self_common.sh`, `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`, `launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch`, and `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_repo_wrap_launcher tests/test_self_common_launcher_helpers.py tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:24:40 UTC

- Migrated `launchers/self/submit_figure2_recipe_aggressive.sh` from local `printf '%q '` wrap construction to the shared `self_wrap_repo_command` helper.
- Preserved the Figure 2 recipe submitter's environment matrix, device-target resource resolution, dry-run path, job metadata, and direct `sbatch` submission behavior.
- Added a fake-`sbatch` launcher test to cover the non-dry submission path and verify the wrapped command includes `PYTHONPATH=.`, forwarded environment values, and the canonical runner.
- Updated `self/README.md` with the Figure 2 recipe submitter boundary.
- Verification: per-file `bash -n` on `launchers/self/lib/self_common.sh`, `launchers/self/submit_figure2_recipe_aggressive.sh`, `launchers/self/run_figure2_recipe_aggressive.sh`, and `launchers/self/run_figure2_paper_retune.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure2_wrap tests/test_figure2_recipe_aggressive_launchers.py tests/test_self_common_launcher_helpers.py -q` (`9 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:31:46 UTC

- Verified that tracked `main` has no residue from the unsupported pseudolabel path removed earlier, and that the prior removal commits are ancestors of current `HEAD`.
- Migrated `launchers/self/submit_figure2_recipe_aggressive.sh` from direct `sbatch_args` construction and direct `sbatch` invocation to `self_submit_wrapped_job`.
- Preserved device-target resource resolution, constraint propagation through the shared helper, dry-run early exit, job metadata, and wrapped command construction through `self_wrap_repo_command`.
- Updated the fake-`sbatch` launcher test to assert the shared helper's separated Slurm argument form.
- Updated `self/README.md` with the Figure 2 wrapped-job submission boundary.
- Verification: per-file `bash -n` on `launchers/self/lib/self_common.sh`, `launchers/self/submit_figure2_recipe_aggressive.sh`, `launchers/self/run_figure2_recipe_aggressive.sh`, and `launchers/self/run_figure2_paper_retune.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure2_submit_helper tests/test_figure2_recipe_aggressive_launchers.py tests/test_self_common_launcher_helpers.py -q` (`9 passed`); tracked unsupported-pseudolabel terminology grep returned no matches.

### Implementation Log: 2026-06-18 16:37:31 UTC

- Added `self/experiments/figure3_common.py` for Figure 3 shared JSON writing, Slurm wrapped-job submission, seed-result metric parsing, final-row loading, and max-at-90 summary helpers.
- Migrated `self/experiments/figure3_seed_quality_sweep.py` to import those shared helpers while preserving the old underscored helper names as module-level compatibility aliases.
- Updated `self/experiments/figure3_real_seed_data_ablation.py` to import shared helpers from the new common owner instead of reaching into private helpers in the seed-quality script, and removed its unused duplicate command-runner helper.
- Added a compatibility test pinning the old seed-quality helper aliases to the new common owner.
- Updated `self/README.md` with the Figure 3 common helper ownership boundary.
- Verification: `python -m py_compile self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py self/figure3_seed_quality_sweep.py self/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_common tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py tests/test_module_proxy.py -q` (`18 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:42:35 UTC

- Extended `self/experiments/figure3_common.py` with shared CSV artifact writing for Figure 3 summaries.
- Updated both Figure 3 scripts to use the common `write_csv` helper, removing their direct `csv.DictWriter` ownership while preserving `_write_csv` as a compatibility alias where it existed.
- Added alias tests for the seed-quality and real-seed ablation helper surfaces.
- Updated `self/README.md` so the Figure 3 common module documents JSON/CSV artifact ownership.
- Verification: `python -m py_compile self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_csv tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`13 passed`).

### Implementation Log: 2026-06-18 16:46:37 UTC

- Extended `self/experiments/figure3_common.py` with shared Figure 3 seed-candidate loading, seed-band constants, generic band filtering, high-seed candidate selection, and missing-band helpers.
- Migrated `self/experiments/figure3_seed_quality_sweep.py` to delegate candidate loading, high-source preference, and missing-band aggregation through the common helpers while preserving `load_seed_candidates`, `SEED_BANDS`, `select_seed_bands`, and `missing_seed_bands` at the old module path.
- Migrated `self/experiments/figure3_real_seed_data_ablation.py` to delegate candidate loading, generic band candidate extraction, high-source preference, and missing-band checks through the common helpers while keeping its monotone addition low/medium policy local.
- Added alias tests for the shared loader and seed-band constant on both Figure 3 module surfaces.
- Updated `self/README.md` so `figure3_common.py` documents seed-band selection ownership.
- Verification: `python -m py_compile self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_seed_helpers tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`13 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:51:43 UTC

- Extended `self/experiments/figure3_common.py` with shared seed-fit and run-length self-improvement command builders for Figure 3 experiment submission.
- Migrated `self/experiments/figure3_seed_quality_sweep.py` to delegate `_seed_job_command` and `_run_length_si_command` through the common builders while preserving the old private function names and the seed-quality command ordering.
- Migrated `self/experiments/figure3_real_seed_data_ablation.py` to delegate the same command builders through the common module while preserving its explicit `max_steps` placement and 8-round run-length matrix setting.
- Added command-builder tests for both Figure 3 scripts, including checks for the experiment-specific max-step placement and run-length round counts.
- Updated `self/README.md` so `figure3_common.py` documents shared command-builder ownership.
- Verification: `python -m py_compile self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_commands tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`15 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:55:40 UTC

- Split Figure 3 command-builder ownership out of `self/experiments/figure3_common.py` into `self/experiments/figure3_commands.py`.
- Kept `figure3_common.py` as a compatibility reexport surface for `seed_fit_command` and `run_length_self_improvement_command`, so existing imports and tests using the common module continue to work.
- Added a reexport identity test to keep the common helper surface pinned to the new command owner.
- Updated `self/README.md` to document `figure3_commands.py` separately from the shared artifact/selection/submission helpers.
- Verification: `python -m py_compile self/experiments/figure3_commands.py self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_commands_split tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`15 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 16:59:56 UTC

- Added `self/experiments/figure3_cli.py` for shared Figure 3 manifest/selection/summary default paths and common argparse option registration.
- Migrated both Figure 3 scripts to use the shared CLI helper while preserving `_parse_common`, `_default_manifest`, `_default_selection`, and `_default_summary` at the old module paths as direct compatibility aliases.
- Added alias tests for the seed-quality and real-seed ablation CLI helper surfaces.
- Updated `self/README.md` to document the new Figure 3 CLI helper boundary.
- Verification: `python -m py_compile self/experiments/figure3_cli.py self/experiments/figure3_commands.py self/experiments/figure3_common.py self/experiments/figure3_seed_quality_sweep.py self/experiments/figure3_real_seed_data_ablation.py tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure3_cli tests/test_figure3_seed_quality_sweep.py tests/test_figure3_real_seed_data_ablation.py -q` (`15 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:06:48 UTC

- Added `launchers/self/config/figure2_run_length.env` as the explicit Figure 2 recipe task-default file for the run-length module, default seed model, archived comparison results, and default expansion size.
- Added `launchers/self/lib/figure2_recipe_common.sh` to own Figure 2 recipe task lookup and environment-aware run-length schedule resolution.
- Migrated `launchers/self/run_figure2_recipe_aggressive.sh` off its local task-default case blocks and onto the shared helper/config while preserving existing environment overrides, dry-run command output, gates, and stage/baseline behavior.
- Added launcher coverage for custom `FIGURE2_TASK_CONFIG` sourcing and updated `self/README.md` with the new Figure 2 task-config/helper boundary.
- Verification: `bash -n launchers/self/run_figure2_recipe_aggressive.sh launchers/self/lib/figure2_recipe_common.sh launchers/self/config/figure2_run_length.env launchers/self/submit_figure2_recipe_aggressive.sh launchers/self/run_figure2_paper_retune.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure2_recipe_config tests/test_figure2_recipe_aggressive_launchers.py tests/test_self_common_launcher_helpers.py -q` (`10 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:10:12 UTC

- Added `launchers/self/config/multiplication_rectangular_seed_sweep.env` as the explicit rectangular multiplication seed-sweep matrix/default file for stage-0 sanity training, stage-1 train-count/LR grid, and stage-3 escalation size.
- Updated `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh` to source the config, parse the Stage 1 matrix from string variables, and print the resolved matrix in its context banner.
- Preserved the default six Stage 1 dry-run jobs, stage-0 export payload, polling/selection flow, and model-link behavior while allowing alternate matrices through `SEED_SWEEP_CONFIG`.
- Added launcher coverage for custom seed-sweep config sourcing and updated `self/README.md` with the new config boundary.
- Verification: `bash -n launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh launchers/self/config/multiplication_rectangular_seed_sweep.env launchers/self/run_multiplication_rectangular_seed_mig.sbatch`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_rect_seed_sweep_config tests/test_multiplication_rectangular_seed_launchers.py tests/test_self_common_launcher_helpers.py -q` (`9 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:13:13 UTC

- Added `launchers/self/config/addition_fixedwidth_mixed.env` as the explicit baseline-list config for the fixed-width mixed-prompt addition submitter.
- Updated `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` to source the config, parse fixed-binary and original-composition baseline lists from string variables, validate that both lists are nonempty, and print the resolved lists in the context banner.
- Preserved the default four fixed-binary baseline submissions, three original-composition submissions, seed submission, dependency wiring, dry-run smoke path, and manifest fields while allowing alternate lists through `ADDITION_MIXED_CONFIG`.
- Added fake-`sbatch` coverage for custom baseline-list config sourcing and updated `self/README.md` with the new config boundary.
- Verification: `bash -n launchers/self/submit_addition_fixedwidth_mixed_mig.sh launchers/self/config/addition_fixedwidth_mixed.env launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_moredata_mig.sh`; `python -m py_compile tests/test_addition_fixedwidth_moredata_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_mixed_config tests/test_addition_fixedwidth_moredata_launcher.py tests/test_self_common_launcher_helpers.py -q` (`10 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:16:18 UTC

- Added `launchers/self/config/addition_fixedwidth_moredata.env` as the explicit Stage 1 grid config for the fixed-width mixed-prompt addition more-data submitter.
- Updated `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` to source the config, parse colon-delimited schedule rows, validate row shape, and print the resolved schedule rows in the context banner.
- Preserved the default four schedule rows, manifest columns, fixed knobs, dry-run output shape, and wrapped-job submission path while allowing alternate grids through `ADDITION_MOREDATA_CONFIG`.
- Added dry-run coverage for custom more-data grid config sourcing and updated `self/README.md` with the new config boundary.
- Verification: `bash -n launchers/self/submit_addition_fixedwidth_moredata_mig.sh launchers/self/config/addition_fixedwidth_moredata.env launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh launchers/self/submit_addition_fixedwidth_mixed_mig.sh`; `python -m py_compile tests/test_addition_fixedwidth_moredata_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_moredata_config tests/test_addition_fixedwidth_moredata_launcher.py tests/test_self_common_launcher_helpers.py -q` (`11 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:18:40 UTC

- Added `launchers/self/config/addition_exact_digits_fixed_binary.env` as the explicit baseline-list config for the exact-digits fixed-binary addition submitter.
- Updated `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` to source the config, parse the baseline list from a string variable, validate that it is nonempty, and print the resolved list in the context banner.
- Preserved the default five baseline submissions, exact-digits schedule knobs, manifest columns, dry-run output shape, and wrapped-job submission path while allowing alternate lists through `ADDITION_EXACT_DIGITS_CONFIG`.
- Added dry-run coverage for custom exact-digits config sourcing and updated `self/README.md` with the new config boundary.
- Verification: `bash -n launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh launchers/self/config/addition_exact_digits_fixed_binary.env launchers/self/run_addition_recipe_fullpack.sh`; `python -m py_compile tests/test_addition_exact_digits_fixed_binary_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_exact_digits_config tests/test_addition_exact_digits_fixed_binary_launcher.py tests/test_self_common_launcher_helpers.py -q` (`6 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:22:06 UTC

- Added `launchers/self/config/run_length_alpha10_baseline_pack.env` as the explicit row config for the run-length alpha10 baseline pack.
- Updated `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh` to source the config, parse `baseline:pseudo_label_mode:guarded_compose_rule` rows, validate row shape, and print the resolved rows in the context banner.
- Preserved the default direct, unfiltered-compose, and guarded-compose jobs, command flags, manifest format, dry-run output shape, and wrapped-job submission path while allowing alternate rows through `RUN_LENGTH_ALPHA10_BASELINE_CONFIG`.
- Added dry-run coverage for custom alpha10 baseline row config sourcing and updated `self/README.md` with the new config boundary.
- Verification: `bash -n launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh launchers/self/config/run_length_alpha10_baseline_pack.env`; `python -m py_compile tests/test_guarded_plain_output_bit_diagnostic_launchers.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_run_length_alpha10_config tests/test_guarded_plain_output_bit_diagnostic_launchers.py tests/test_self_common_launcher_helpers.py -q` (`9 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:28:55 UTC

- Extended `self/analysis/adaptive_artifacts.py` with `adaptive_validity_summary_records(...)`, a stable per-attempt valid-rate/category-count loader for adaptive proposal analysis notebooks.
- Extended `adaptive_selected_per_size_timeline_records(...)` in `self/analysis/adaptive_trace_artifacts.py` with `target_size` and `is_target_size` fields so heatmaps can mark selected target digits/lengths without re-parsing proposal JSON.
- Reexported the new validity summary helper through `self/analysis/artifacts.py` and pinned both additions in `tests/test_analysis_artifacts.py`.
- Updated `self/README.md` so new adaptive notebooks use the analysis loaders for valid-rate evolution, selected-target heatmap markers, candidate metrics, train-mix summaries, and worker-failure files instead of direct raw JSON globbing.
- Verification: `python -m py_compile self/analysis/adaptive_artifacts.py self/analysis/adaptive_trace_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_artifacts tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`11 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:33:59 UTC

- Added `self/analysis/seed_fit_artifacts.py` as the canonical owner for `seed_fit_results.json` constants, run-directory detection, result-path resolution, discovery, and validated payload loading.
- Updated `self/analysis/seed_fit_bundle.py` and `self/analysis/summarize_seed_fit_grid.py` to use the shared seed-fit artifact resolver instead of each hard-coding recursive `seed_fit_results.json` glob/load behavior.
- Preserved the old private `_load_json(...)` seed-fit bundle helper for notebook compatibility and reexported the new artifact helpers through `self/analysis/seed_fit_curve_notebook_utils.py` and `self/analysis/artifacts.py`.
- Extended `tests/test_seed_fit_curve_notebook_utils.py` to pin resolver behavior, notebook-helper reexports, bundle loading, and seed-fit summary-grid row/selection behavior.
- Updated `self/README.md` so new seed-fit notebooks and scripts use the shared artifact resolver rather than raw `**/seed_fit_results.json` path conventions.
- Verification: `python -m py_compile self/analysis/seed_fit_artifacts.py self/analysis/seed_fit_bundle.py self/analysis/seed_fit_curve_notebook_utils.py self/analysis/summarize_seed_fit_grid.py self/analysis/artifacts.py tests/test_seed_fit_curve_notebook_utils.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_seed_fit_artifacts tests/test_seed_fit_curve_notebook_utils.py tests/test_module_proxy.py -q` (`8 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:39:30 UTC

- Moved the alpha-10 symbol-run run-length trainer-seed beam implementation from `launchers/self/run_run_length_alpha10_seed_beam_mig.py` into `self/experiments/run_length_alpha10_seed_beam.py`.
- Replaced the old launcher path with a thin compatibility wrapper that inserts the repo root into `sys.path` and calls the canonical experiment `main()`.
- Updated `launchers/self/submit_run_length_fixed_binary_mig.sh` to launch new beam jobs through `python -m self.experiments.run_length_alpha10_seed_beam` while preserving the old wrapper for direct invocations.
- Extended `tests/test_run_length_fixed_binary_launchers.py` to pin the canonical module import, fixed-binary submitter command, and legacy wrapper dry-run behavior.
- Updated `self/README.md` with the new experiment owner and compatibility-wrapper boundary.
- Verification: `python -m py_compile self/experiments/run_length_alpha10_seed_beam.py launchers/self/run_run_length_alpha10_seed_beam_mig.py tests/test_run_length_fixed_binary_launchers.py`; `bash -n launchers/self/submit_run_length_fixed_binary_mig.sh`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_run_length_beam_move tests/test_run_length_fixed_binary_launchers.py -q` (`3 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:44:22 UTC

- Added `launchers/self/config/run_length_fixed_binary.env` as the explicit default config for the fixed-binary run-length submitter's paper-default, alpha-10 template, and beam settings.
- Updated `launchers/self/submit_run_length_fixed_binary_mig.sh` to source the tracked default config first and then optionally source a partial `RUN_LENGTH_FIXED_BINARY_CONFIG` override file, preserving environment overrides and default behavior.
- Replaced hard-coded paper/template/beam command knobs and Slurm resource values in the submitter with named config variables while keeping the existing manifest shape and dry-run flow.
- Extended `tests/test_run_length_fixed_binary_launchers.py` with dry-run coverage for partial config overrides reaching paper, alpha-10 template, and beam commands.
- Updated `self/README.md` with the new fixed-binary run-length config boundary.
- Verification: `bash -n launchers/self/submit_run_length_fixed_binary_mig.sh launchers/self/config/run_length_fixed_binary.env`; `python -m py_compile tests/test_run_length_fixed_binary_launchers.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_run_length_fixed_binary_config tests/test_run_length_fixed_binary_launchers.py tests/test_self_common_launcher_helpers.py -q` (`7 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:48:13 UTC

- Added `launchers/self/config/multiplication_rectangular_fullpack.env` as the explicit default config for rectangular multiplication fullpack baselines, seed model path, train/eval batch sizes, replay/expand budgets, frontier profile, round schedule, held-out budget, LR/max-step overrides, save policy, and seed.
- Updated `launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` to source the tracked default config first and then optionally source a partial `MULT_RECT_FULLPACK_CONFIG` override file before parsing the baseline list.
- Preserved existing runner-facing environment variable names in the sbatch export payload while moving the defaults out of the submitter body.
- Extended `tests/test_multiplication_rectangular_self_improvement_launchers.py` with syntax coverage for the new config and dry-run coverage for partial fullpack overrides.
- Updated `self/README.md` with the new rectangular fullpack config boundary.
- Verification: `bash -n launchers/self/submit_multiplication_rectangular_fullpack_mig.sh launchers/self/config/multiplication_rectangular_fullpack.env launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch`; `python -m py_compile tests/test_multiplication_rectangular_self_improvement_launchers.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_mult_rect_fullpack_config tests/test_multiplication_rectangular_self_improvement_launchers.py tests/test_self_common_launcher_helpers.py -q` (`9 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 17:58:35 UTC

- Added `launchers/self/config/addition_fullpack_filtered.env` as the explicit default config for the addition fullpack-filtered submitter's job script path, log directory, dry-run default, and baseline list.
- Updated `launchers/self/submit_addition_fullpack_filtered_mig.sh` to source the tracked default config first and then optionally source a partial `ADDITION_FULLPACK_FILTERED_CONFIG` override file before parsing the baseline list.
- Preserved the default five baseline submissions and existing `BASELINES` environment shortcut while moving the baseline matrix out of the submitter body.
- Extended `tests/test_addition_fullpack_filtered_launcher.py` with config-file coverage plus dry-run checks for `BASELINES` and override-config behavior.
- Updated `self/README.md` with the new addition fullpack-filtered config boundary.
- Verification: `bash -n launchers/self/submit_addition_fullpack_filtered_mig.sh launchers/self/config/addition_fullpack_filtered.env launchers/self/run_addition_fullpack_filtered.sbatch`; `python -m py_compile tests/test_addition_fullpack_filtered_launcher.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_addition_fullpack_filtered_config tests/test_addition_fullpack_filtered_launcher.py tests/test_self_common_launcher_helpers.py -q` (`11 passed`); `git diff --check`.

### Implementation Log: 2026-06-18 18:02:33 UTC

- Added `launchers/self/config/figure2_recipe_aggressive.env` as the explicit default config for Figure 2 aggressive recipe top-level stage/task/device/dry-run settings and the fullpack baseline list.
- Updated `launchers/self/run_figure2_recipe_aggressive.sh` and `launchers/self/submit_figure2_recipe_aggressive.sh` to source the tracked default config first and then optionally source one or more `FIGURE2_RECIPE_CONFIG` override files.
- Preserved existing `BASELINES` environment overrides by using them to initialize `FIGURE2_RECIPE_BASELINES_RAW`, while adding a stable override variable for config-file use.
- Extended `tests/test_figure2_recipe_aggressive_launchers.py` with config-file coverage plus runner and submitter dry-run checks for `FIGURE2_RECIPE_CONFIG`.
- Updated `self/README.md` with the new Figure 2 aggressive recipe config boundary.
- Verification: `bash -n launchers/self/run_figure2_recipe_aggressive.sh launchers/self/submit_figure2_recipe_aggressive.sh launchers/self/config/figure2_recipe_aggressive.env launchers/self/lib/figure2_recipe_common.sh launchers/self/config/figure2_run_length.env launchers/self/run_figure2_paper_retune.sh`; `python -m py_compile tests/test_figure2_recipe_aggressive_launchers.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_figure2_recipe_aggressive_config tests/test_figure2_recipe_aggressive_launchers.py tests/test_self_common_launcher_helpers.py -q` (`13 passed`).

### Implementation Log: 2026-06-18 18:06:06 UTC

- Added serial candidate-dispatch bootstrap-cache reuse in `self/core/candidate_dispatch_runtime.py`.
- Serial candidate mode now creates one `ModelBootstrapCache` per dispatch call and passes it to candidate scoring functions that accept `model_bootstrap_cache`, matching the existing train/checkpoint API and preserving per-candidate fresh model instantiation.
- Without `--candidate-local-cache-base-state`, serial mode shares tokenizer bootstrap work only; with the flag, it can also reuse an unmodified CPU copy of the source checkpoint state across candidates.
- Extended `tests/test_candidate_dispatch_runtime.py` to verify that serial candidates receive the same cache object when the scorer supports it, that `cache_base_state` follows the existing flag, and that legacy injected scorers without the cache keyword remain compatible.
- Updated `self/README.md` runtime notes with the serial-cache behavior.
- Verification: `python -m py_compile self/core/candidate_dispatch_runtime.py tests/test_candidate_dispatch_runtime.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_serial_candidate_cache tests/test_candidate_dispatch_runtime.py tests/test_candidate_training_runtime.py tests/test_model_io_bootstrap_cache.py -q` (`11 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_serial_candidate_adaptive tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`41 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 18:12:42 UTC

- Added stable local-candidate dispatch plan metadata in `self/core/candidate_local_workers.py`.
- `attempt_*/candidate_jobs/local_dispatch.json` now records `candidate_count`, `planned_processes`, `packed_workers`, `cache_plan`, and per-process planned candidate indices in addition to the live launched/active/pending state.
- The new `cache_plan` exposes whether shared input caching, tokenizer bootstrap caching, and base-state caching are active for the resolved local-worker plan, making pack/cache tuning auditable from run artifacts without changing candidate execution semantics.
- Extended `tests/test_adaptive_candidate_training.py` to pin both non-packed and packed local dispatch metadata.
- Updated `self/README.md` runtime notes with the new dispatch-artifact fields.
- Verification: `python -m py_compile self/core/candidate_local_workers.py tests/test_adaptive_candidate_training.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_local_dispatch_plan tests/test_adaptive_candidate_training.py -k 'local_parallel_candidate_workers_respect_concurrency_cap or local_parallel_candidate_workers_can_pack_processes' -q` (`2 passed`, `37 deselected`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_local_dispatch_adaptive tests/test_adaptive_candidate_training.py tests/test_adaptive_self_improvement_controller.py -q` (`41 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 18:17:30 UTC

- Added stable adaptive local-dispatch artifact loading in `self/analysis/adaptive_candidate_artifacts.py`.
- New helpers `load_adaptive_local_dispatch(...)` and `adaptive_local_dispatch_records(...)` expose `attempt_*/candidate_jobs/local_dispatch.json` through the analysis layer, including flattened cache-plan flags and planned candidate groups.
- Reexported the helpers through `self/analysis/adaptive_artifacts.py` and `self/analysis/artifacts.py` so notebooks can import them from the existing compatibility surface.
- Extended `tests/test_analysis_artifacts.py` to cover local-dispatch loading and row flattening using the existing adaptive-run fixture.
- Updated `self/README.md` analysis guidance so new notebooks use the loader rather than hard-coding `candidate_jobs/local_dispatch.json`.
- Verification: `python -m py_compile self/analysis/adaptive_artifact_common.py self/analysis/adaptive_candidate_artifacts.py self/analysis/adaptive_artifacts.py self/analysis/artifacts.py tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_analysis_local_dispatch tests/test_analysis_artifacts.py tests/test_training_curve_notebook_utils.py -q` (`11 passed`).

### Implementation Log: 2026-06-18 18:25:52 UTC

- Added `read_round_summaries(...)` to `self/analysis/artifact_io.py` as the
  shared parser for non-adaptive round-result JSON. It supports both legacy
  top-level lists and newer `{"rounds": [...]}` payloads while rejecting
  unrelated dict-shaped files.
- Migrated non-adaptive artifact loading, appendix heatmap plotting,
  training-curve result frames, and training-curve round metrics to use the
  shared parser instead of each module opening/parsing raw JSON directly.
- Reexported the helper through `self/analysis/artifacts.py` so notebooks can
  keep importing from the stable compatibility surface.
- Updated `self/README.md` to document the round-summary loader boundary.

### Implementation Log: 2026-06-18 18:30:15 UTC

- Extracted fixture dry-run proposal processing from
  `self/experiments/adaptive_self_improvement.py` into
  `self/core/proposal_pilot_runtime.py`.
- The new core runtime owns pilot config/program validation, reward
  normalization, raw-output normalization, duplicate filtering, selected
  proposal ranking, prompt rendering, and proposal-trace row construction.
- Kept the old private helper names available from the experiment module as
  compatibility aliases so existing tests/notebooks can keep importing them
  while new implementation code has a focused owner module.
- Updated `self/README.md` with the new pilot-runtime boundary.

### Implementation Log: 2026-06-18 18:36:44 UTC

- Added shared launcher helpers `self_print_torch_cuda_info(...)` and
  `self_preflight_model_snapshot(...)` to `launchers/self/lib/self_common.sh`.
- Migrated the standard Hugging Face/local model-snapshot preflight in
  `run_task_self_improvement.sbatch`, `run_seed_fit_experiment.sbatch`,
  `run_self_improvement_mig_boundary_eval.sbatch`, and
  `run_self_improvement_qwen_no_growth.sbatch` from duplicated embedded
  Python heredocs to the shared helper.
- Left the fixed-char tokenizer preflight in
  `run_addition_fullpack_filtered.sbatch` unchanged because it has
  tokenizer-mode-specific behavior beyond the standard AutoTokenizer check.
- Updated `self/README.md` with the new launcher helper responsibility.

### Implementation Log: 2026-06-18 18:41:07 UTC

- Extended `self_preflight_model_snapshot(...)` with an optional tokenizer-mode
  argument. With no tokenizer-mode argument it preserves the previous standard
  AutoTokenizer preflight output; with `fixed_char` it builds and reports the
  fixed-char tokenizer used by addition recipe launchers.
- Migrated `run_addition_fullpack_filtered.sbatch` onto the shared preflight
  helper via `self_preflight_model_snapshot "${MODEL_NAME}" "${TOKENIZER_MODE}"`.
- Kept the fixed-char behavior and output shape documented explicitly because
  it differs from the standard AutoTokenizer-only launchers.
- Updated launcher helper tests and `self/README.md` for the optional
  tokenizer-mode preflight path.

### Implementation Log: 2026-06-18 18:49:37 UTC

- Added shared stable symlink helpers to `launchers/self/lib/self_common.sh`:
  `self_update_symlink(...)` and `self_update_symlink_or_dry_run(...)`.
- Migrated addition seed/fullpack/recovery launchers and the rectangular
  multiplication seed-sweep winner-link updates from hand-written
  `mkdir`/`ln -sfn`/`echo` blocks to the shared helper without changing their
  acceptance, status-file, or dry-run semantics.
- Added direct helper tests covering parent-directory creation, real symlink
  updates, and dry-run-only reporting; updated the tiny addition seed launcher
  test to assert the shared helper boundary.
- Updated `self/README.md` with the new launcher helper responsibility.
- Verification: `bash -n launchers/self/lib/self_common.sh
  launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch
  launchers/self/run_addition_fullpack_filtered.sbatch
  launchers/self/run_addition_recipe_recovery.sh
  launchers/self/run_addition_seed_shared.sbatch
  launchers/self/run_addition_tiny_seed_mig.sbatch
  launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh`;
  `python -m py_compile tests/test_self_common_launcher_helpers.py
  tests/test_addition_tiny_seed_mig_launcher.py`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_symlink_helper
  tests/test_self_common_launcher_helpers.py
  tests/test_addition_tiny_seed_mig_launcher.py
  tests/test_addition_fullpack_filtered_launcher.py
  tests/test_addition_fixedwidth_moredata_launcher.py
  tests/test_addition_recipe_recovery_launchers.py
  tests/test_multiplication_rectangular_seed_launchers.py -q` (`40 passed`).

### Implementation Log: 2026-06-18 18:57:40 UTC

- Split task-specific proposal prompt metadata out of
  `self/core/proposal_prompts.py` into
  `self/core/proposal_prompt_metadata.py`.
- The new module owns target-format descriptions, component-prediction example
  strings, generated-program sandbox validation-case selection, and the
  driver-selected default executable source-pair policy.
- Kept old imports working by reexporting those helpers through
  `self/core/proposal_prompts.py`; implementation code that only needs
  metadata now imports the new owner directly.
- Updated proposal ownership tests to pin both canonical metadata ownership and
  old-path compatibility through `proposal_prompts` and driver compatibility
  exports.
- Updated `self/README.md` with the new prompt-metadata boundary.
- Verification: `python -m py_compile self/core/proposal_prompt_metadata.py
  self/core/proposal_prompts.py self/core/proposal_runtime.py
  self/core/proposal_executable_validation.py self/core/attempt_prompt_runtime.py
  self/core/driver_default_bindings.py self/core/driver_compat_exports.py
  tests/test_proposal_generation.py tests/test_adaptive_proposals_and_sandbox.py`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_prompt_metadata
  tests/test_proposal_generation.py tests/test_adaptive_proposals_and_sandbox.py
  tests/test_proposal_executable_validation.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`57 passed`, `7`
  existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:03:38 UTC

- Made the adaptive driver facade lazy for concrete default bindings and driver
  wiring imports.
- Added `self/core/driver_default_binding_manifest.py` as a name-only manifest
  for default binding names. `self/core/driver.py` now imports this lightweight
  manifest instead of importing `self/core/driver_default_bindings.py` at module
  import time.
- Installed driver public delegates through a lazy wiring proxy, so
  `dir(self.core.driver)`, `self.core.driver.__all__`, and the top-level
  `self.adaptive_candidate_training` module proxy can enumerate compatibility
  names without importing `driver_wiring`, `run_orchestration`, Torch, or
  Transformers.
- Preserved old behavior when a default binding or delegate is actually used:
  default bindings still resolve through `self/core/driver_default_bindings.py`,
  and delegates still call the same `driver_wiring` functions with the live
  driver module as the binding surface.
- Added `tests/test_driver_lazy_imports.py` to pin the import-light facade
  behavior in a fresh subprocess.
- Verification: base-Python import smoke confirmed `import self.core.driver`
  and `import self.adaptive_candidate_training` succeed without loading
  `self.core.driver_default_bindings` or `self.core.driver_wiring`;
  `python -m py_compile self/core/driver.py
  self/core/driver_default_bindings.py
  self/core/driver_default_binding_manifest.py tests/test_driver_lazy_imports.py`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_lazy
  tests/test_driver_lazy_imports.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py tests/test_proposal_generation.py -q`
  (`47 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:09:39 UTC

- Kept adaptive prompt and attempt orchestration modules import-light for
  inspection and static tooling.
- Removed the `self.tasks` package import from
  `self/core/proposal_prompt_metadata.py` by using local target-mode
  normalization and the stable `run_state` target-mode string needed only for
  prompt metadata. This avoids loading task adapters and model-evaluation
  dependencies while rendering proposal prompts.
- Converted `TrainingConfig` imports in
  `self/core/attempt_loop_runtime.py`,
  `self/core/attempt_candidate_runtime.py`,
  `self/core/round_model_dispatch_runtime.py`,
  `self/core/run_initialization_runtime.py`, and
  `self/core/seed_dispatch_runtime.py` to `TYPE_CHECKING` imports because those
  modules use the class only in annotations.
- Verified in base Python that `self.core.proposal_prompt_metadata`,
  `self.core.proposal_prompts`, `self.core.round_model_dispatch_runtime`,
  `self.core.seed_dispatch_runtime`, `self.core.run_initialization_runtime`,
  `self.core.attempt_candidate_runtime`, and `self.core.attempt_loop_runtime`
  import successfully without Torch installed.
- Extended `tests/test_driver_lazy_imports.py` to pin those runtime contract
  imports in a fresh subprocess without loading `self.tasks`,
  `self.core.training`, Torch, or Transformers.
- Verification: `python -m py_compile self/core/proposal_prompt_metadata.py
  self/core/proposal_prompts.py self/core/round_model_dispatch_runtime.py
  self/core/seed_dispatch_runtime.py self/core/run_initialization_runtime.py
  self/core/attempt_candidate_runtime.py self/core/attempt_loop_runtime.py
  tests/test_driver_lazy_imports.py`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_import_light_runtime
  tests/test_driver_lazy_imports.py tests/test_proposal_generation.py
  tests/test_adaptive_proposals_and_sandbox.py tests/test_attempt_candidate_runtime.py
  tests/test_attempt_outcome_runtime.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`60 passed`, `7`
  existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:19:03 UTC

- Converted `self/core/driver_compat_exports.py` from an eager import barrel to
  a manifest-backed lazy resolver keyed by
  `self/core/driver_compat_manifest.py`.
- Preserved the legacy export names and old attribute/import behavior while
  avoiding import-time loading of task adapters, training code, Torch, and
  Transformers when code only imports, lists, or inspects the compatibility
  module.
- Added a missing-target guard so new names added to the compatibility
  manifest must be wired to a canonical owner explicitly.
- Extended `tests/test_driver_lazy_imports.py` to pin import-light listing of
  `self.core.driver_compat_exports` in a fresh subprocess.
- Verification: `python -m py_compile self/core/driver_compat_exports.py
  tests/test_driver_lazy_imports.py`; base-Python smoke confirmed importing
  and listing `self.core.driver_compat_exports` leaves task/training/Torch/
  Transformers modules unloaded and that `ConfigProposal` still resolves to
  `self.core.proposal_config_schema.ConfigProposal`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_driver_compat_lazy
  tests/test_driver_lazy_imports.py tests/test_proposal_generation.py
  tests/test_adaptive_proposals_and_sandbox.py tests/test_composition_pseudolabels.py
  tests/test_proposal_grpo_traces.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`60 passed`, `7`
  existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:25:19 UTC

- Converted `self/core/nonadaptive_facade_exports.py` from an eager
  compatibility barrel to a lazy resolver for the old
  `self.self_improvement_core` public surface.
- Updated `self/self_improvement_core.py` so importing, listing, or inspecting
  the legacy facade no longer star-imports model IO, evaluation, training,
  Torch, Transformers, or the canonical non-adaptive loop.
- Preserved legacy attribute imports and monkeypatch behavior: requested names
  resolve to the same canonical owners, and `run_self_improvement(...)` loads
  default values for all `NONADAPTIVE_PATCHABLE_NAMES` before syncing facade
  globals into `self.core.nonadaptive_loop`.
- Added `tests/test_driver_lazy_imports.py` coverage showing that
  `self.self_improvement_core` and `self.core.nonadaptive_facade_exports` can
  be imported/listed in a fresh subprocess without loading the non-adaptive
  loop, training stack, Torch, or Transformers.
- Verification: `python -m py_compile
  self/core/nonadaptive_facade_exports.py self/self_improvement_core.py
  tests/test_driver_lazy_imports.py`; base-Python smoke confirmed facade
  import/listing leaves heavy modules unloaded; `PYTHONPATH=. conda run -n
  torch-env pytest --basetemp=.pytest_tmp_nonadaptive_lazy
  tests/test_driver_lazy_imports.py tests/test_nonadaptive_compat.py
  tests/test_training_data.py tests/test_self_improvement_tasks.py
  tests/test_run_length_recipe.py tests/test_nonadaptive_seed_round_zero.py -q`
  (`55 passed`); final focused check with adaptive smoke coverage:
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_nonadaptive_lazy_final
  tests/test_driver_lazy_imports.py tests/test_nonadaptive_compat.py
  tests/test_training_data.py tests/test_self_improvement_tasks.py
  tests/test_run_length_recipe.py tests/test_nonadaptive_seed_round_zero.py
  tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`97 passed`, `3`
  existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:31:17 UTC

- Factored common lazy-compatibility export mechanics into
  `self/core/lazy_exports.py`.
- Moved the driver compatibility owner map into
  `self/core/driver_compat_targets.py`, leaving
  `self/core/driver_compat_exports.py` as a thin public lazy facade over
  `self/core/driver_compat_manifest.py` plus the target map.
- Moved the legacy `self.self_improvement_core` owner map into
  `self/core/nonadaptive_facade_targets.py`, leaving
  `self/core/nonadaptive_facade_exports.py` focused on grouped export names and
  lazy facade behavior.
- Preserved import-light behavior for both public compatibility surfaces: the
  new target-map modules contain only strings and do not import task adapters,
  the non-adaptive loop, training, Torch, or Transformers.
- Verification: `python -m py_compile self/core/lazy_exports.py
  self/core/driver_compat_targets.py self/core/driver_compat_exports.py
  self/core/nonadaptive_facade_targets.py self/core/nonadaptive_facade_exports.py
  self/self_improvement_core.py tests/test_driver_lazy_imports.py`;
  base-Python smoke confirmed importing all lazy facade/target modules leaves
  task, non-adaptive-loop, training, Torch, and Transformers modules unloaded;
  target-map completeness checks reported no missing driver or non-adaptive
  compatibility targets; `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_lazy_export_targets tests/test_driver_lazy_imports.py
  tests/test_nonadaptive_compat.py tests/test_training_data.py
  tests/test_self_improvement_tasks.py tests/test_run_length_recipe.py
  tests/test_nonadaptive_seed_round_zero.py tests/test_proposal_generation.py
  tests/test_adaptive_proposals_and_sandbox.py tests/test_composition_pseudolabels.py
  tests/test_proposal_grpo_traces.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`112 passed`, `7`
  existing multiprocessing fork warnings).

### Implementation Log: 2026-06-18 19:37:23 UTC

- Added `self_sbatch_export_all(...)` to `launchers/self/lib/self_common.sh`
  as a shared helper for building `sbatch --export ALL,NAME=value,...` payloads
  from explicit `NAME=value` arguments.
- Migrated representative submitters away from hand-built comma strings:
  `submit_addition_fullpack_filtered_mig.sh`,
  `submit_guarded_plain_output_bit_diagnostic_mig.sh`,
  `submit_multiplication_rectangular_square_probe_mig.sh`, and
  `submit_multiplication_rectangular_square_seed_resweep_mig.sh`.
- Kept submission semantics unchanged: the helper produces the same
  `ALL,...` export string passed into `self_submit_sbatch_script(...)`, while
  making launcher matrices easier to inspect and edit.
- Added direct helper coverage to `tests/test_self_common_launcher_helpers.py`.
- Verification: `bash -n launchers/self/lib/self_common.sh
  launchers/self/submit_addition_fullpack_filtered_mig.sh
  launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh
  launchers/self/submit_multiplication_rectangular_square_probe_mig.sh
  launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_export_helper
  tests/test_self_common_launcher_helpers.py
  tests/test_addition_fullpack_filtered_launcher.py
  tests/test_multiplication_rectangular_square_launchers.py
  tests/test_guarded_plain_output_bit_diagnostic_launchers.py -q` (`29 passed`).

### Implementation Log: 2026-06-18 19:41:04 UTC

- Completed the first `self_submit_sbatch_script(...)` export-string cleanup:
  no remaining `launchers/self` submitter uses hand-written `"ALL,..."`
  strings or `export_vars="ALL,..."` construction.
- Migrated the remaining representative callers to `self_sbatch_export_all(...)`:
  `submit_addition_fixedwidth_mixed_mig.sh`,
  `submit_multiplication_rectangular_seed_sweep_mig.sh`,
  `submit_multiplication_rectangular_fullpack_mig.sh`,
  `submit_adaptive_candidate_training_ailab.sh`, and
  `submit_adaptive_condition_pilots_ailab.sh`.
- Preserved the existing exported variable names and values; this is a
  readability/consolidation-only launcher change.
- Verification: `rg -n "\"ALL,|export_vars=\"ALL," launchers/self` returned no
  matches; `bash -n launchers/self/submit_addition_fixedwidth_mixed_mig.sh
  launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh
  launchers/self/submit_multiplication_rectangular_fullpack_mig.sh
  launchers/self/submit_adaptive_candidate_training_ailab.sh
  launchers/self/submit_adaptive_condition_pilots_ailab.sh`;
  `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_export_helper_round2
  tests/test_adaptive_candidate_launcher.py tests/test_adaptive_condition_launcher.py
  tests/test_addition_fixedwidth_moredata_launcher.py
  tests/test_multiplication_rectangular_seed_launchers.py
  tests/test_multiplication_rectangular_self_improvement_launchers.py
  tests/test_self_common_launcher_helpers.py -q` (`35 passed`).

### Implementation Log: 2026-06-18 19:49:46 UTC

- Added `self/launcher_manifests.py` as the importable owner for adaptive
  candidate-training and adaptive condition-pilot submission-manifest JSON
  construction.
- Replaced the embedded Python here-doc manifest writers in
  `submit_adaptive_candidate_training_ailab.sh` and
  `submit_adaptive_condition_pilots_ailab.sh` with
  `python -m self.launcher_manifests ...`.
- Preserved the existing manifest schemas, job keys, integer conversions,
  `frontier_diagnostics_path` empty-string handling, and dry-run behavior.
- Added focused manifest-builder/CLI tests while keeping existing launcher
  dry-run tests as end-to-end coverage for the submitter wiring.
- Verification: `python -m py_compile self/launcher_manifests.py
  tests/test_launcher_manifests.py`; `bash -n
  launchers/self/submit_adaptive_candidate_training_ailab.sh
  launchers/self/submit_adaptive_condition_pilots_ailab.sh`;
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_adaptive_manifest_helpers
  tests/test_launcher_manifests.py tests/test_adaptive_candidate_launcher.py
  tests/test_adaptive_condition_launcher.py -q` (`10 passed`).

### Implementation Log: 2026-06-18 19:56:32 UTC

- Continued non-adaptive cleanup by extracting round-loop context/state and
  single-round dependency-map assembly from `self/core/nonadaptive_loop.py`
  into `self/core/nonadaptive_round_context.py`.
- Kept the legacy monkeypatch boundary intact: `self/core/nonadaptive_loop.py`
  still owns the patchable globals synced from `self.self_improvement_core`,
  and passes those live globals into the new assembly helper before dispatching
  the round loop.
- Reduced `self/core/nonadaptive_loop.py` from `303` to `223` lines; the new
  context helper owns `130` lines with focused tests.
- Verification: `python -m py_compile self/core/nonadaptive_round_context.py
  self/core/nonadaptive_loop.py tests/test_nonadaptive_round_context.py`;
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_nonadaptive_round_context
  tests/test_nonadaptive_round_context.py tests/test_nonadaptive_round_runtime.py
  tests/test_nonadaptive_round_loop.py tests/test_nonadaptive_compat.py
  tests/test_nonadaptive_seed_round_zero.py -q` (`12 passed`).

### Implementation Log: 2026-06-18 20:03:32 UTC

- Added `self/analysis/adaptive_summary_artifacts.py` with
  `adaptive_run_overview_records(...)` and
  `adaptive_validity_summary_records_for_runs(...)`.
- Reexported the new notebook-facing helpers through
  `self.analysis.artifacts`.
- The helpers give notebooks stable per-run overview rows and
  validity-by-attempt rows without hard-coding `attempt_*`,
  `proposal_results.json`, `candidate_metrics.json`, or local-dispatch paths.
- Verification: `python -m py_compile
  self/analysis/adaptive_summary_artifacts.py self/analysis/artifacts.py
  tests/test_analysis_artifacts.py`; `PYTHONPATH=. conda run -n torch-env
  pytest --basetemp=.pytest_tmp_adaptive_summary_artifacts
  tests/test_analysis_artifacts.py -q` (`6 passed`).

### Implementation Log: 2026-06-18 20:09:27 UTC

- Split serial candidate dispatch from `self/core/candidate_dispatch_runtime.py`
  into `self/core/candidate_serial_runtime.py`.
- The new serial module owns attempt-index seed routing, scorer signature
  inspection for the optional `model_bootstrap_cache` keyword, and the
  per-dispatch `ModelBootstrapCache` used to avoid repeated tokenizer/source
  checkpoint bootstrap work when semantics allow it.
- Kept `self.core.candidate_dispatch_runtime.train_candidates_serial` and
  `self.core.candidate_execution.train_candidates_serial` as compatibility
  reexports.
- Reduced `self/core/candidate_dispatch_runtime.py` from `281` to `215` lines,
  leaving it focused on candidate execution-mode selection plus local/Slurm
  dispatch delegation.
- Verification: `python -m py_compile self/core/candidate_serial_runtime.py
  self/core/candidate_dispatch_runtime.py self/core/candidate_execution.py
  tests/test_candidate_dispatch_runtime.py`; `PYTHONPATH=. conda run -n
  torch-env pytest --basetemp=.pytest_tmp_candidate_serial_runtime
  tests/test_candidate_dispatch_runtime.py tests/test_candidate_training_runtime.py
  tests/test_model_io_bootstrap_cache.py -q` (`11 passed`);
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_candidate_serial_adaptive
  tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`42 passed`, `3`
  existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 20:15:42 UTC

- Split local-parallel and Slurm-array candidate dispatch wrappers from
  `self/core/candidate_dispatch_runtime.py` into
  `self/core/candidate_parallel_runtime.py`.
- The new module owns the concrete local/Slurm wrapper calls into
  `self.core.candidate_workers`, including the compatibility subprocess binding
  used by local-parallel monkeypatch tests.
- Kept `self.core.candidate_dispatch_runtime.train_candidates_local_parallel`,
  `self.core.candidate_dispatch_runtime.train_candidates_slurm_array`, and the
  corresponding `self.core.candidate_execution` names as compatibility
  reexports.
- Reduced `self/core/candidate_dispatch_runtime.py` from `215` to `137` lines;
  it now focuses on candidate execution-mode selection plus candidate
  metric/failure wrapper reexports.
- Verification: `python -m py_compile self/core/candidate_parallel_runtime.py
  self/core/candidate_serial_runtime.py self/core/candidate_dispatch_runtime.py
  self/core/candidate_execution.py tests/test_candidate_dispatch_runtime.py`;
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_candidate_parallel_runtime
  tests/test_candidate_dispatch_runtime.py tests/test_candidate_training_runtime.py
  tests/test_model_io_bootstrap_cache.py -q` (`11 passed`);
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_candidate_parallel_adaptive
  tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`42 passed`, `3`
  existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 20:20:35 UTC

- Split candidate-dispatch dependency ownership from
  `self/core/candidate_dispatch_entrypoints.py` and
  `self/core/driver_candidate_dispatch_wiring.py` into
  `self/core/candidate_dispatch_deps.py`.
- The new module owns `CandidateDispatchEntrypointDeps` and
  `build_candidate_dispatch_deps(...)`, while the old entrypoint and driver
  wiring paths keep compatibility reexports/wrappers.
- This keeps driver monkeypatch binding construction in one inspectable place
  and leaves `candidate_dispatch_entrypoints.py` focused on public wrapper
  calls into the runtime dispatch modules.
- Verification: `python -m py_compile self/core/candidate_dispatch_deps.py
  self/core/candidate_dispatch_entrypoints.py
  self/core/driver_candidate_dispatch_wiring.py
  tests/test_candidate_dispatch_runtime.py`; `PYTHONPATH=. conda run -n
  torch-env pytest --basetemp=.pytest_tmp_candidate_dispatch_deps
  tests/test_candidate_dispatch_runtime.py tests/test_candidate_training_runtime.py
  tests/test_model_io_bootstrap_cache.py -q` (`12 passed`);
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_candidate_dispatch_deps_adaptive
  tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`42 passed`, `3`
  existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 20:25:40 UTC

- Split non-adaptive round data contracts from
  `self/core/nonadaptive_round_runtime.py` into
  `self/core/nonadaptive_round_models.py`.
- The new module owns `NonAdaptiveRoundRuntimeContext`,
  `NonAdaptiveRoundRuntimeState`, and `NonAdaptiveRoundRuntimeResult`; the old
  runtime module still reexports those names for compatibility.
- Updated `self/core/nonadaptive_round_context.py` and
  `self/core/nonadaptive_round_loop.py` to import the round contracts from the
  lightweight models module instead of loading the single-round runtime.
- Verification: `python -m py_compile self/core/nonadaptive_round_models.py
  self/core/nonadaptive_round_runtime.py self/core/nonadaptive_round_context.py
  self/core/nonadaptive_round_loop.py tests/test_nonadaptive_round_runtime.py
  tests/test_nonadaptive_round_context.py tests/test_nonadaptive_round_loop.py`;
  first pytest attempt included a stale filename and failed before collection;
  corrected command `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_nonadaptive_round_models
  tests/test_nonadaptive_round_runtime.py tests/test_nonadaptive_round_context.py
  tests/test_nonadaptive_round_loop.py tests/test_nonadaptive_round_setup.py
  tests/test_nonadaptive_seed_round_zero.py -q` passed (`14 passed`);
  `git diff --check`.

### Implementation Log: 2026-06-18 20:30:17 UTC

- Audited the tracked repository for unsupported sampled label-family
  terminology and implementation hooks; no tracked source, tests, launchers, or
  maintained docs exposed such a mode.
- Clarified `self/README.md` with the supported pseudolabel mode surface:
  `none`, `direct`, `compose`, and `compose_corrupt`.
- No runtime behavior changed. Unsupported label-family baselines stay out of
  the maintained workshop/main-track surface unless they are later introduced
  as explicit experiments with tests and launcher/docs coverage.

### Implementation Log: 2026-06-18 20:33:24 UTC

- Split adaptive attempt-loop data contracts from runtime code into
  `self/core/attempt_loop_models.py`.
- The new module owns `CandidateAttemptDeps`, `AttemptLoopDeps`, and
  `AttemptLoopResult`; `self/core/attempt_candidate_runtime.py` and
  `self/core/attempt_loop_runtime.py` still reexport the old import-path names
  for compatibility.
- Added `tests/test_attempt_loop_models.py` to pin those compatibility aliases.
- Verification: `python -m py_compile self/core/attempt_loop_models.py
  self/core/attempt_loop_runtime.py self/core/attempt_candidate_runtime.py
  tests/test_attempt_loop_models.py tests/test_attempt_candidate_runtime.py`;
  `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_attempt_loop_models
  tests/test_attempt_loop_models.py tests/test_attempt_candidate_runtime.py
  tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py -q` (`44 passed`, `3`
  existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 20:38:58 UTC

- Split `AdaptiveRunDeps` from `self/core/run_orchestration.py` into
  `self/core/run_models.py`.
- Kept `self.core.run_orchestration.AdaptiveRunDeps` available by importing
  the container from the new owner module.
- Updated `self/core/driver_run_wiring.py` to import only `run_models` at
  module load time and lazy-load `run_adaptive_candidate_training(...)` inside
  `run(...)`. This keeps driver wiring imports from eagerly loading the
  Torch/Transformers run orchestration stack.
- Added `tests/test_run_models.py` for old-path compatibility and the lazy
  import behavior.
- Verification: `python -m py_compile self/core/run_models.py
  self/core/run_orchestration.py self/core/driver_run_wiring.py
  tests/test_run_models.py`; manual import check confirmed
  `self.core.driver_run_wiring` does not load `self.core.run_orchestration`,
  `torch`, or `transformers`; `PYTHONPATH=. conda run -n torch-env pytest
  --basetemp=.pytest_tmp_run_models tests/test_run_models.py
  tests/test_adaptive_self_improvement_controller.py
  tests/test_adaptive_candidate_training.py -q` (`44 passed`, `3` existing
  multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 20:46:47 UTC

- Created and pushed checkpoint tag
  `pre-outdated-cleanup-20260618-204614` at commit `c4ab771` before deleting
  any tracked legacy code.
- Removed the unreferenced old weak-to-strong standalone addition monolith
  `legacy/weak_to_strong/weak_to_strong_addition_experiment_old.py`.
- Kept the active archived weak-to-strong entry point
  `legacy/weak_to_strong/weak_to_strong_addition_experiment_v2.py`, which
  delegates to the canonical `core.addition_pipeline` implementation used by
  the archived weak-to-strong shell scripts.
- Added `legacy/weak_to_strong/README.md` and updated the root `README.md` so
  future readers know where the removed monolith can be recovered.
- Removed ignored generated `__pycache__` directories under `self/core`,
  `self/tasks`, `self/analysis`, and `launchers/self` from the working tree.
- Verification: `python -m py_compile
  legacy/weak_to_strong/weak_to_strong_addition_experiment_v2.py
  legacy/weak_to_strong/weak_to_strong_composition_error_experiment.py`;
  `bash -n` over the archived weak-to-strong shell launchers; tracked grep for
  the deleted filename outside this plan log returned no matches; generated
  `__pycache__` directories were removed after verification.

### Implementation Log: 2026-06-18 20:59:21 UTC

- Created and pushed checkpoint tag
  `pre-wrapper-cleanup-20260618-165850` at commit `2f0800d` before deleting
  top-level compatibility wrappers.
- Removed pure star-import wrappers that had no tracked references outside
  `self/README.md`: `self/adaptive_candidate_workers.py`,
  `self/adaptive_controller_phases.py`, `self/adaptive_experience_traces.py`,
  `self/adaptive_proposal_grpo.py`, `self/adaptive_worker_io.py`, and
  `self/slurm_utils.py`.
- Kept test-covered compatibility wrappers such as `self/adaptive_frontier.py`
  and `self/self_improvement_experiment.py` for this pass.
- Verification: live tracked grep outside this plan log found no references to
  the removed wrapper paths/imports; `~/.conda/envs/torch-env/bin/python -m
  compileall -q self/core self/legacy self/experiments self/diagnostics
  self/analysis self/tasks`; `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python
  -m pytest --basetemp=.pytest_tmp_wrapper_cleanup -q
  tests/test_module_proxy.py tests/test_adaptive_candidate_training.py
  tests/test_adaptive_self_improvement_controller.py` (`48 passed`, `3`
  existing multiprocessing fork warnings); `git diff --check`.

### Implementation Log: 2026-06-18 21:19:32 UTC

- Created and pushed checkpoint tag
  `pre-self-package-refactor-20260618-171519` at commit `22fe3e5` before the
  larger package reorganization.
- Split the flat `self/core` namespace into human-facing subsystem packages:
  `self/adaptive/run`, `self/adaptive/attempts`,
  `self/adaptive/candidates`, `self/adaptive/proposals`,
  `self/adaptive/traces`, `self/adaptive/frontier`,
  `self/adaptive/controller`, `self/adaptive/sandbox`, and
  `self/nonadaptive`.
- Reduced `self/core` from 127 Python files to shared utilities only. It now
  holds composition, data/model IO, evaluation, shared dataclasses, recipe
  helpers, Slurm helpers, tokenizers, training utilities, task protocols,
  summaries, lazy export helpers, and module proxy helpers.
- Updated tracked implementation and tests to import from the new subsystem
  owners. Adaptive tests no longer import implementation behavior through
  top-level CLI wrappers such as `self.adaptive_candidate_training`,
  `self.adaptive_proposals`, `self.adaptive_frontier`, or
  `self.program_sandbox`.
- Replaced the stale file-by-file core inventory in `self/README.md` with a
  compact package map and documented the tests-vs-CLI boundary.
- Verification: import-sensitive adaptive tests (`38 passed`, `4` existing
  multiprocessing fork warnings); adaptive candidate/worker tests (`79
  passed`, `3` existing multiprocessing fork warnings); nonadaptive tests (`58
  passed`); launcher/CLI-facing tests (`53 passed`; touched launcher subset
  rerun after compile-list refresh with `30 passed`); live tracked grep found no
  stale `self.core.*` references to moved adaptive/nonadaptive modules;
  `~/.conda/envs/torch-env/bin/python -m compileall -q self tests`; `bash -n`
  for the touched adaptive Slurm launchers; `python -m
  self.adaptive_candidate_training --help`; `python -m self.adaptive.driver
  --help`; `git diff --check`.

### Implementation Log: 2026-06-18 21:41:02 UTC

- Created and pushed checkpoint tag
  `pre-self-consolidation-20260618-173223` at commit `8d81bdf` before deleting
  additional internal modules.
- Consolidated narrow helper/compatibility files into their owning modules:
  attempt-loop/outcome containers now live in
  `self/adaptive/attempts/attempt_models.py`; candidate dispatch dependencies
  live in `candidate_dispatch_entrypoints.py`; candidate worker failure helpers
  live in `candidate_worker_payloads.py`; proposal compatibility exports live
  in `self/adaptive/proposals/__init__.py`; nonadaptive facade targets live in
  `nonadaptive_facade_exports.py`; adaptive driver compat names/targets live in
  `driver_compat_exports.py`; and driver delegate installation now lives in
  `driver.py`.
- Removed test-only compatibility facades including
  `candidate_execution.py`, `candidate_dispatch_deps.py`,
  `candidate_worker_failures.py`, `attempt_loop_models.py`,
  `attempt_outcome_models.py`, `proposal/proposals.py`,
  `nonadaptive_facade_targets.py`, `driver_compat_manifest.py`,
  `driver_compat_targets.py`, `driver_default_binding_manifest.py`, and
  `driver_public_api.py`.
- Merged the remaining thin adaptive driver-binding files
  (`driver_candidate_dispatch_wiring.py`, `driver_worker_wiring.py`,
  `driver_proposal_grpo_wiring.py`, and `driver_run_wiring.py`) into
  `self/adaptive/run/driver_wiring.py`; the adaptive/nonadaptive/shared runtime
  surface is now 123 Python files, down from 136 immediately after the package
  reorganization.
- Verification: focused consolidation suite
  (`tests/test_driver_lazy_imports.py`, attempt/candidate/proposal/nonadaptive
  touched tests, `tests/test_run_models.py`, and `tests/test_module_proxy.py`)
  passed; stale deleted-module grep found no live references; `compileall`;
  `bash -n` for touched adaptive launchers; `python -m
  self.adaptive_candidate_training --help`; `python -m self.adaptive.driver
  --help`; `git diff --check`.

### Implementation Log: 2026-06-19 02:38 UTC

- Created and pushed checkpoint tag `pre-self-slimdown-20260618-220652` before
  the aggressive cleanup.
- Removed the remaining top-level `self/*.py` implementation wrappers; only
  `self/__init__.py` and `self/launcher_manifests.py` remain at package top
  level.
- Deleted old compatibility modules after migrating tracked launchers/tests/docs
  to canonical modules.
- Consolidated shared helpers: recipes into `self/core/recipes.py`, composition
  pseudolabel helpers into `self/core/composition.py`, task lookup into
  `self/core/task_protocols.py`, training-curve analysis into
  `self/analysis/training_curves.py`, seed-fit analysis into
  `self/analysis/seed_fit.py`, and adaptive artifact loaders into
  `self/analysis/adaptive_artifacts.py`.
- Consolidated task code into task-owned modules: `addition.py`, `bit.py`,
  `multiplication.py`, `rectangular.py`, and `run_length.py`.
- Consolidated adaptive code that had grown into prefix-split files: attempts
  into `self/adaptive/attempts/attempts.py`, traces into
  `self/adaptive/traces/traces.py`, controller phases/workers into
  `self/adaptive/controller/controller.py`, candidates into
  `self/adaptive/candidates/{training,dispatch,workers}.py`, frontier helpers
  into `self/adaptive/frontier/frontier.py`, sandbox models/cases into
  `self/adaptive/sandbox/program_sandbox.py`, and prompt metadata into
  `self/adaptive/proposals/proposal_prompts.py`.
- Preserved the lightweight import contract by keeping phase constants in
  `self/adaptive/controller/__init__.py` and keeping `CheckpointManager` local
  to run initialization rather than importing the full candidate training stack.
- Current `self/` Python file count is 122, with tasks down to 6 files and
  adaptive candidates down to 4 files.
- Verification so far: `compileall -q self tests`; stale deleted-module grep
  over tracked source/docs; focused refactor suite (`157 passed`, with existing
  multiprocessing fork warnings).

### Implementation Log: 2026-06-18 23:22 UTC

- Created and pushed checkpoint tag `pre-adaptive-flatten-20260618-191056` at
  commit `c913c79` before flattening `self/adaptive`.
- Removed adaptive subpackages and moved their contents into flat
  `self/adaptive/*.py` modules: driver/run orchestration, attempts,
  candidates, controller, frontier, proposals, sandbox, and traces now share a
  single package level.
- Updated tracked imports and adaptive Slurm/worker command paths from
  `python -m self.adaptive.run.driver` to `python -m self.adaptive.driver`.
- Added lightweight `self/adaptive/phases.py` so seed/round/proposal dispatch
  can import phase names without loading the full controller or training stack.
- Verification so far: deleted adaptive subpackage grep is clean; focused
  adaptive suite passed (`108 passed`, with existing multiprocessing fork
  warnings).

### Implementation Log: 2026-06-19 00:07 UTC

- Created and pushed checkpoint tag `pre-adaptive-prefix-merge-20260618-195044`
  at commit `312ec63` before merging the remaining same-prefix adaptive files.
- Merged adaptive prefix groups into owning modules with no shim files:
  `candidate_*.py` into `self/adaptive/candidate.py`, `proposal_*.py` plus
  `proposals.py` into `self/adaptive/proposal.py`, `run_*.py` plus seed/round
  dispatch into `self/adaptive/run.py`, and `driver_wiring.py`,
  `driver_default_bindings.py`, and `entrypoint.py` into
  `self/adaptive/driver.py`.
- Preserved the lightweight import contract after the merge: importing
  `self.adaptive.driver`, `self.adaptive.proposal`, or `self.adaptive.run`
  does not load `torch` or `transformers`; training/generation imports now
  happen inside the runtime functions that need them.
- Current adaptive layout is 12 Python files:
  `args.py`, `attempts.py`, `candidate.py`, `controller.py`, `driver.py`,
  `frontier.py`, `phases.py`, `program_sandbox.py`, `proposal.py`, `run.py`,
  `traces.py`, and `__init__.py`. Current `self/` Python file count is 95.
- Verification so far: `compileall -q self/adaptive tests`, import smoke, CLI
  help smoke, and focused adaptive suite (`108 passed`, with existing
  multiprocessing fork warnings).

### Implementation Log: 2026-06-19 03:45 UTC

- Implemented the cleaner config proposal update as a merged proposal optimizer:
  `policy_GRPO + 0.01 * KL + 0.2 * realized_outcome_CE + 0.1 * format_CE`.
- Kept legacy full-completion GRPO available via
  `--proposal-update-loss-mode legacy_grpo`; config runs now default to
  `merged_agent`.
- In merged mode, policy GRPO masks only the executable `proposal` JSON value
  inside the full `action_prediction` completion, world CE masks the
  `prediction` value using realized frontier/final deltas, and format CE trains
  full valid `action_prediction` JSON from current valid proposals plus bounded
  selected proposal-trace replay.
- Disabled post-task proposal rehearsal by default for `merged_agent`, so
  selected proposal traces are learned in the merged proposal update instead of
  a separate second SFT pass after candidate task training.
- Wired selected proposal trace buffers through local and SLURM controller
  proposal-update paths and wrote `proposal_world_targets.jsonl` /
  `proposal_format_targets.jsonl` when those terms have trainable examples.
- Reduced config proposal generation default from 256 to 160 new tokens and
  exported merged-loss settings from the ailab submitter.
- Broke two import-time cycles in the flattened proposal module by removing
  runtime-only `core.models` imports.
- Verification so far: focused py_compile and bash syntax checks; focused
  launcher/arg/proposal tests (`5 passed`); adaptive candidate training tests
  (`41 passed`, with existing multiprocessing fork warnings).

### Submission Log: 2026-06-19 03:45 UTC

- Submitted the cleaner merged config matrix to ailab:
  addition/run_length x `num_candidates={8,16}` x
  `proposal_grpo_zero_variance={fixed_baseline,skip}`.
- Submission manifest:
  `artifacts/runs/adaptive_candidate_training_ailab_20260618_214438/submission_manifest.json`.
- Job IDs:
  `9934720` addition n8 fixed_baseline,
  `9934721` addition n16 fixed_baseline,
  `9934722` addition n8 skip,
  `9934723` addition n16 skip,
  `9934724` run_length n8 fixed_baseline,
  `9934725` run_length n16 fixed_baseline,
  `9934726` run_length n8 skip,
  `9934727` run_length n16 skip.
- Initial monitoring after roughly one minute: all jobs were still pending on
  ailab and no stdout/stderr logs had been created yet.

### Submission Log: 2026-06-19 03:50 UTC

- Dropped the `proposal_grpo_zero_variance=skip` condition before any skip job
  started.
- Canceled skip jobs: `9934722`, `9934723`, `9934726`, `9934727`.
- Remaining cleaner merged config jobs are the fixed-baseline runs:
  `9934720`, `9934721`, `9934724`, `9934725`.

### Submission Log: 2026-06-19 19:40 UTC

- The fixed-baseline jobs `9934720`, `9934721`, `9934724`, and `9934725`
  failed before Python startup with exit code `1:0` after roughly 2 seconds.
- Root cause: SLURM executes a copied script from `/var/spool/slurmd/...`;
  the adaptive SBATCH runners used `${BASH_SOURCE[0]}` to locate
  `lib/adaptive_common.sh`, so they looked for the helper under the SLURM
  spool directory instead of the repo.
- Patched the adaptive candidate main, candidate-worker, and controller-worker
  SBATCH scripts to prefer `${SLURM_SUBMIT_DIR}/launchers/self` when the helper
  exists, with the old `${BASH_SOURCE[0]}` behavior as fallback.
- Verification: `bash -n` for the patched SBATCH scripts and
  `tests/test_adaptive_candidate_launcher.py` passed.
- Resubmitted fixed-baseline-only matrix to
  `artifacts/runs/adaptive_candidate_training_ailab_20260619_153743`:
  `9978744` addition n8,
  `9978745` addition n16,
  `9978746` run_length n8,
  `9978747` run_length n16.
- Initial monitoring: all four resubmitted jobs were pending on ailab; no
  stdout/stderr logs had been created yet.

### Runtime Issue Log: 2026-06-19 21:50 UTC

- Job `9978745` (`addition`, `num_candidates=16`) failed in attempt 1 during
  the merged proposal update, after seed training and candidate training had
  completed.
- Failure was CUDA OOM in `apply_proposal_grpo_update` while computing
  `format_loss`; this is not a different experimental pipeline from the
  surviving `9978744` (`addition`, `num_candidates=8`), except that n16 doubled
  proposal traces and valid format/world targets.
- Diagnosis: the merged update currently does policy, world, and format
  forwards before one combined backward, so all three activation graphs are
  resident at once. n16 attempt 1 had 16 policy traces, 8 world targets, and 9
  format targets; n8 had 8/4/5 and survived.
- Planned fix: keep the exact same objective and one optimizer step, but
  compute proposal-update losses with gradient-accumulated microbatches:
  backward policy microbatches, backward world microbatches, backward format
  microbatches, then clip and step once. This should preserve the pipeline and
  only change memory scheduling, aside from minor floating-point ordering
  differences.
- Do not patch while currently running jobs may still import repo files; apply
  after the active runs finish or after intentionally canceling/restarting them.

### Implementation Log: 2026-06-20 OOM Fix

- Implemented the planned merged proposal-update memory fix.
- Added `--proposal-update-microbatch-size` with default `8`; launchers and
  `adaptive_candidate_base.env` now pass and log it explicitly.
- The update still performs one optimizer step per proposal update. It now
  backprops policy microbatches, world-model CE microbatches, and format CE
  microbatches before gradient clipping and stepping, instead of retaining all
  graphs at once.
- Also changed proposal log-prob computation from explicit full-vocab
  `log_softmax(...).gather(...)` to unreduced cross entropy, reducing the
  transient fp32 memory spike.
- Expected runtime impact: modest, because this only affects the small proposal
  update after candidate evaluation, not seed/candidate task SFT. Use
  microbatch `8` by default; if OOM persists, try `4` or `2`.
- Verification: focused parser/launcher/proposal-microbatch tests passed, and
  `tests/test_adaptive_candidate_training.py` passed (`42 passed`) under the
  torch environment.

### Implementation Log: 2026-06-20 Repeated-Candidate Mitigations

- Added proposal-action deduplication before proposal-policy GRPO advantage
  computation. Duplicate actions remain valid candidates, but only the best
  measured instance contributes policy-gradient credit.
- Changed the default zero-variance proposal-GRPO behavior from
  `fixed_baseline` to `skip`, so batches with no reward contrast do not
  reinforce every sampled action.
- Added compact prompt history under aggregate diagnostics as `recent_actions`.
  Defaults: last `4` attempts, top `3` actions per attempt, with rounded reward
  and frontier-delta fields.
- Added rank-based proposal rewards via `--proposal-grpo-reward-mode rank`.
  Invalid proposals keep verifier penalties, valid-untrained proposals get
  `0`, and trained valid candidates are ranked into `[0, 1]`.
- Updated the adaptive candidate submitter/manifest schema to support reward
  ablations through `PROPOSAL_GRPO_REWARD_MODES`, e.g. `outcome rank`.
- Verification: bash syntax, Python compile, focused proposal/launcher tests
  (`13 passed`), and broader adaptive candidate/launcher/manifest suite
  (`53 passed`, only existing multiprocessing fork warnings).

### Submission Log: 2026-06-20 History/Rank Quick Ablation

- Added a history axis to the adaptive candidate submitter/manifest so history
  on/off runs get distinct output directories and manifest metadata.
- Submitted an 8-job quick matrix to ailab:
  `task={addition,run_length}` x `proposal_grpo_reward_mode={outcome,rank}` x
  `proposal_history_max_attempts={0,4}`.
- Common settings: config condition, `num_candidates=8`,
  `proposal_grpo_zero_variance=skip`, `proposal_grpo_deduplicate_actions=1`,
  `proposal_history_max_actions=3`, `max_attempt_rounds=10`,
  `no_selection_patience=10`.
- Manifest:
  `artifacts/runs/adaptive_candidate_history_rank_ablation_20260620_111035/submission_manifest.json`.
- Job IDs:
  `10009446` addition outcome history-off,
  `10009447` addition outcome history-4,
  `10009448` addition rank history-off,
  `10009449` addition rank history-4,
  `10009450` run_length outcome history-off,
  `10009451` run_length outcome history-4,
  `10009452` run_length rank history-off,
  `10009453` run_length rank history-4.
- Initial monitoring: all eight jobs were accepted by SLURM and pending on
  ailab with reason `Priority`; no stdout/stderr logs had been created yet.
- Follow-up: all eight jobs failed before entering the adaptive loop. Root
  cause was a stale Python validation requiring
  `max_attempt_rounds >= num_rounds`; the quick ablation intentionally used
  `num_rounds=100` and `max_attempt_rounds=10`.
- Fixed the validation to allow attempt-capped smoke/diagnostic runs while
  still rejecting negative `max_attempt_rounds`. No resubmission was performed
  after this fix.

### Implementation Log: 2026-06-20 Attempt-Based Adaptive Control

- Removed the workshop-era adaptive default that treated `--num-rounds` as the
  primary run budget. New adaptive runs are bounded by `--max-attempt-rounds`
  by default, currently `100`.
- Added `--max-selected-rounds` as an explicit optional cap on selected
  candidates. The default is `0`, meaning unlimited selected candidates within
  the attempt/no-selection budget.
- Kept `--num-rounds` as a deprecated compatibility alias: when it is provided
  and `--max-selected-rounds` is not, it sets the selected-candidate cap; when
  it is provided and `--max-attempt-rounds` is not, attempts default to
  `10 * num_rounds`.
- Removed `target_selected_rounds` from proposal prompts so the proposer sees
  current attempt/selection state but is not encouraged to imitate a fixed
  selected-round schedule.
- Updated AILab adaptive launchers/configs to stop exporting `NUM_ROUNDS`; they
  now export `MAX_ATTEMPT_ROUNDS` and `MAX_SELECTED_ROUNDS`.
- No failed history/rank ablation jobs were resubmitted as part of this cleanup.
- Verification: bash syntax checks passed for the adaptive runner/submitter,
  Python compile checks passed for `self/adaptive/{args,attempts,run,proposal}.py`, and
  the focused adaptive suite passed (`60 passed`, with existing multiprocessing
  fork warnings).

### Implementation Log: 2026-06-20 100-Step Candidate SFT Default

- Changed the adaptive candidate AILab runner/config default from epoch-based
  task SFT (`MAX_STEPS=0`) to `MAX_STEPS=100`.
- Left Python parser defaults unchanged for direct CLI compatibility; the
  100-step behavior is a launcher/config default for new AILab adaptive runs.
- Kept generated task example counts unchanged so the first run isolates the
  compute-budget change from data-distribution changes.
- Kept proposal-side training settings unchanged: no selected proposal-trace
  rehearsal under `merged_agent`, and the lightweight proposal update remains
  separate from task SFT.

### Submission Log: 2026-06-20 25-Attempt 100-Step Config Jobs

- Added an adaptive submitter `SBATCH_TIME` hook so shorter walltime requests
  override the runner script's legacy `24:00:00` directive at submission time.
- Estimated runtime from prior 25-attempt/full-epoch and merged-agent n8 runs:
  historical jobs used roughly 5-10.5 hours; with `MAX_STEPS=100`, the new
  runs should be substantially faster, so requested `08:00:00`.
- Submitted config condition jobs with:
  `task={addition,run_length}`, `num_candidates=8`,
  `max_attempt_rounds=25`, `no_selection_patience=25`,
  `max_selected_rounds=0`, `max_steps=100`,
  `proposal_update_loss_mode=merged_agent`,
  `proposal_grpo_reward_mode=outcome`,
  `proposal_grpo_zero_variance=skip`,
  `proposal_history_max_attempts=4`, `post_task_proposal_rehearsal=0`,
  `mem=48G`, and `time=08:00:00`.
- Manifest:
  `artifacts/runs/adaptive_candidate_25a_100steps_20260620_203943/submission_manifest.json`.
- Job IDs:
  `10020270` addition config,
  `10020271` run_length config.
- Initial monitoring: both jobs were pending on ailab with reason `Priority`;
  no stdout/stderr logs had been created yet.
- Correction: this two-job submission was too narrow. It only used the cleaned
  default condition instead of rerunning the full failed history/rank ablation
  matrix. Jobs `10020270` and `10020271` were still pending with no logs and
  were canceled.
- Resubmitted the intended 8-job matrix:
  `task={addition,run_length}` x `proposal_grpo_reward_mode={outcome,rank}` x
  `proposal_history_max_attempts={0,4}`, with the same 25-attempt, 100-step,
  merged-agent, skip-zero-variance settings.
- Corrected manifest:
  `artifacts/runs/adaptive_candidate_25a_100steps_8job_20260620_204158/submission_manifest.json`.
- Corrected job IDs:
  `10020277` addition outcome history-off,
  `10020278` addition outcome history-4,
  `10020279` addition rank history-off,
  `10020280` addition rank history-4,
  `10020281` run_length outcome history-off,
  `10020282` run_length outcome history-4,
  `10020283` run_length rank history-off,
  `10020284` run_length rank history-4.
- Initial monitoring of the corrected matrix: all eight jobs were pending on
  ailab with `08:00:00` time limits; no startup logs yet.

### Implementation Log: 2026-06-20 Seed/Candidate Step Split

- Found a budget-wiring issue before the 8-job matrix started: launcher-level
  `MAX_STEPS=100` was passed through the shared `TrainingConfig`, so it capped
  both round-0 seed training and candidate-attempt training.
- Canceled pending jobs `10020277`-`10020284` before startup; no stdout/stderr
  logs had been created.
- Added `--seed-max-steps` to the adaptive driver. Default `None` preserves old
  direct-CLI behavior by using `--max-steps` for seed training too; the AILab
  adaptive launcher now sets `SEED_MAX_STEPS=0`, meaning seed training remains
  epoch-based/full while candidate attempts use `MAX_STEPS=100`.
- Updated launcher logs to print
  `Train epochs/candidate max steps/seed max steps`.
- Verification: `bash -n` for runner/submitter and focused adaptive launcher,
  argument, and candidate-training tests passed (`53 passed`, with existing
  multiprocessing fork warnings).
- Resubmitted the intended 8-job matrix with seed full and candidate attempts
  capped at 100 steps:
  `task={addition,run_length}` x `proposal_grpo_reward_mode={outcome,rank}` x
  `proposal_history_max_attempts={0,4}`.
- Manifest:
  `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/submission_manifest.json`.
- Job IDs:
  `10020309` addition outcome history-off,
  `10020310` addition outcome history-4,
  `10020311` addition rank history-off,
  `10020312` addition rank history-4,
  `10020313` run_length outcome history-off,
  `10020314` run_length outcome history-4,
  `10020315` run_length rank history-off,
  `10020316` run_length rank history-4.
- Initial monitoring: all eight jobs were pending on ailab with `08:00:00`
  time limits; no startup logs yet.

### Implementation Log: 2026-06-21 Local Candidate Parallelism Default

- Live monitoring of the 25-attempt matrix showed parent jobs continuing, but
  many local candidate workers hit GPU OOMs because up to four candidate
  training subprocesses shared one H200.
- Reduced the default local candidate concurrency from `4` to `2` in the Python
  argument default, the AILab adaptive runner fallback, and
  `adaptive_candidate_base.env`.
- Kept `CANDIDATE_LOCAL_PACK_SIZE=2` and
  `CANDIDATE_LOCAL_CACHE_BASE_STATE=1` in the shared base config. This should
  keep candidate dispatch reasonably efficient now that candidate task training
  is capped at 100 steps, while avoiding the worst local GPU contention.

### Implementation Log: 2026-06-21 Prompt History and Clean Rerun Setup

- Made config proposal history explicit in prompts: when
  `proposal_history_max_attempts > 0`, compact prior attempt actions now render
  in a separate `Recent evaluated actions` block rather than being buried in
  aggregate diagnostics.
- Hardened the AILab adaptive submitter so it defaults to
  `adaptive_candidate_base.env`, explicitly exports
  `CANDIDATE_LOCAL_PARALLELISM=2`, `CANDIDATE_LOCAL_PACK_SIZE=2`, and
  `CANDIDATE_LOCAL_CACHE_BASE_STATE=1`, and accepts `SBATCH_DEPENDENCY`.
- Verification: runner/submitter `bash -n` passed; focused adaptive launcher,
  argument, and candidate-training tests passed (`53 passed`).
- Submitted the clean dependent 8-job rerun under
  `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938`
  with dependency
  `afterany:10020309:10020310:10020311:10020312:10020313:10020314:10020315:10020316`.
- Job IDs:
  `10023747` addition outcome history-off,
  `10023748` addition outcome history-4,
  `10023749` addition rank history-off,
  `10023750` addition rank history-4,
  `10023751` run_length outcome history-off,
  `10023752` run_length outcome history-4,
  `10023753` run_length rank history-off,
  `10023754` run_length rank history-4.
- Slurm monitoring confirmed all eight new jobs are pending with reason
  `(Dependency)`.

### Implementation Log: 2026-06-21 03:17:18 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/run_length-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48]`.
- Selected proposal traces retained for replay: `23`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `23`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 12:39:32 UTC

- Checked the clean 25-attempt 8-job rerun
  `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938`.
- All eight jobs completed:
  `10023747`, `10023748`, `10023749`, `10023750`, `10023751`,
  `10023752`, `10023753`, `10023754`.
- Runtime was about `3.6`-`4.4` hours/job. Each job requested `48G` CPU RAM;
  `sacct` batch MaxRSS was about `25`-`26G`, so the CPU-memory request is no
  longer over-allocating badly.
- No top-level CUDA OOMs/tracebacks were found in this rerun, and logs report
  `Candidate local parallelism/pack/cache-base-state: 2/2/1`.
- Important caveat: this rerun is OOM-clean, but it is not a valid explicit
  prompt-history ablation. Saved `proposal_prompt.json` files in the history-on
  runs still have `history_prompts=0`.
- Root cause: the adaptive loop built a prompt with recent-action history, but
  the round-model dispatch path rebuilt the actual generation prompt and dropped
  `extra_aggregate_metrics`.
- Fix implemented after detecting the issue:
  - `run_candidate_attempt` now passes compact recent-action history to
    `run_round_model_dispatch`.
  - `run_round_model_dispatch` forwards prompt extras to both local and SLURM
    controller paths.
  - `run_round_model_controller_worker_from_spec` forwards
    `extra_aggregate_metrics` into `run_round_model_phase`.
  - `run_round_model_phase` merges those extras with
    `proposal_output_schema` before calling `build_attempt_prompt`.
- Added regression coverage in `tests/test_attempt_candidate_runtime.py` for:
  local dispatch prompt extras, SLURM worker payload prompt extras, and
  candidate-attempt recent-action forwarding.
- Verification:
  `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_attempt_candidate_runtime.py tests/test_adaptive_candidate_training.py tests/test_adaptive_candidate_launcher.py -q`
  passed with `53 passed, 3 warnings`.

### Implementation Log: 2026-06-21 03:18:31 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/run_length-config-numeric-n8-reward-rank-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48]`.
- Selected proposal traces retained for replay: `22`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `22`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 03:21:47 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/run_length-config-numeric-n8-reward-outcome-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48]`.
- Selected proposal traces retained for replay: `23`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `21`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 03:25:47 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/run_length-config-numeric-n8-reward-rank-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40]`.
- Selected proposal traces retained for replay: `23`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `21`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 04:06:48 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/addition-config-numeric-n8-reward-rank-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 15, 23]`.
- Selected proposal traces retained for replay: `20`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `20`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 04:08:39 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/addition-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 15, 17, 23]`.
- Selected proposal traces retained for replay: `20`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `21`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 04:16:00 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/addition-config-numeric-n8-reward-outcome-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 15, 25]`.
- Selected proposal traces retained for replay: `21`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `22`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 04:20:58 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_8job_20260620_204706/addition-config-numeric-n8-reward-rank-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 15, 17, 23]`.
- Selected proposal traces retained for replay: `17`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `21`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 08:16:02 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/addition-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 9, 11, 12, 14, 16, 17, 19, 20, 22, 29]`.
- Selected proposal traces retained for replay: `24`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 08:43:39 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/addition-config-numeric-n8-reward-outcome-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 13, 15, 17, 23]`.
- Selected proposal traces retained for replay: `25`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `23`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 09:39:59 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/addition-config-numeric-n8-reward-rank-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 12, 13, 15, 17, 22]`.
- Selected proposal traces retained for replay: `22`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 10:42:13 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/addition-config-numeric-n8-reward-rank-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 14, 15, 17, 24, 25]`.
- Selected proposal traces retained for replay: `24`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 11:53:59 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/run_length-config-numeric-n8-reward-rank-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48]`.
- Selected proposal traces retained for replay: `24`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 12:10:06 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/run_length-config-numeric-n8-reward-outcome-history-off-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40]`.
- Selected proposal traces retained for replay: `25`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `19`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 12:17:06 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/run_length-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 48]`.
- Selected proposal traces retained for replay: `25`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `23`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 12:37:08 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938/run_length-config-numeric-n8-reward-rank-history-4-grpo-skip`.
- Proposal output schema: `action_prediction`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 40]`.
- Selected proposal traces retained for replay: `25`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `200`.
- Proposal GRPO updates: `23`; steps/update: `1`; reward mode: `rank`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; world/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 13:39:17 UTC

- Cleaned proposer-facing config prompt/trace terminology after manual inspection.
- Removed `already_in_source` from `allowed_target_frontier`; repeated targets are allowed and should be judged by measured reward, so this field was unnecessary prompt clutter.
- Renamed prompt diagnostics from `current_final_accuracy`/`init_final_accuracy` to `current_avg_accuracy`/`init_avg_accuracy`.
- Renamed action-prediction schema field from `expected_final_delta_from_init` to `expected_avg_delta_from_init`; old completions with `expected_final_delta_from_init` remain valid as a backward-compatible alias, but normalized new training completions use the `avg` key.
- Renamed compact outcome-trace prompt state from `current_final`/`init_final` to `current_avg`/`init_avg`, and target keys from `final_delta_*` to `avg_delta_*`.
- Verified with `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py tests/test_experience_outcome_traces.py -q` (`50 passed`, `3 warnings`).

### Implementation Log: 2026-06-21 14:23:08 UTC

- Replaced the default config proposal schema with ECHO-style `action_observation`.
- New model proposal output is action-only plus rationale, e.g. `{"left":5,"right":3,"guard":"reject_boundary_carry","rationale":"..."}`; the driver executes only `left/right/guard`.
- Kept `action_prediction` as a backward-compatible schema for old runs and tests.
- Replaced separate realized-prediction CE with driver-appended observation CE: the model is conditioned on the original prompt plus `Assistant action: <action>` and trained to predict `Environment observation: <observation_json>`.
- Observation JSON includes validity/training status, measured reward, target, target/frontier/average deltas, and `delta_per_size`; `advantage` is intentionally excluded because it is batch-normalized optimizer bookkeeping rather than an environment fact.
- Added `per_size_delta` to `CandidateMetrics` serialization so observations can expose per-size changes without reparsing prompts.
- Updated launcher defaults from `action_prediction` to `action_observation`.
- Verified with `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py tests/test_experience_outcome_traces.py tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_candidate_metric_collection.py tests/test_candidate_rewards.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_training_mix.py -q` (`89 passed`, `7 warnings`).

### Implementation Log: 2026-06-21 14:56:44 UTC

- Tightened the config proposal format to the explicit tagged trace: `<reasoning>...</reasoning><action>{...}</action>`.
- The validator now parses only the JSON object inside `<action>` and rejects missing tags, extra text outside the two blocks, malformed action JSON, and action fields other than `left/right/guard`.
- Removed the active `action_prediction` schema path and renamed the auxiliary supervised target from world prediction to driver-appended observation prediction.
- Observation CE targets no longer include the scalar policy reward; they keep validity/training status, measured target/frontier/average deltas, per-size deltas, and failure metadata when present.
- Added `--proposal-grpo-span` with default `reasoning_action`; `action_only` remains available as a later ablation that masks GRPO to the JSON inside `<action>`.
- Updated launcher defaults and summaries to use `PROPOSAL_OBSERVATION_LOSS_WEIGHT` and the new GRPO span flag.
- Verified with `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py tests/test_experience_outcome_traces.py tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py tests/test_adaptive_proposals_and_sandbox.py tests/test_proposal_generation.py tests/test_candidate_metric_collection.py tests/test_candidate_rewards.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py tests/test_candidate_training_mix.py -q` (`88 passed`, `7 warnings`) and a focused final-state recheck over proposal/parser/launcher tests (`61 passed`, `3 warnings`).

### Implementation Log: 2026-06-21 15:06:18 UTC

- Submitted the no-example tagged-format pilot with `MAX_ATTEMPT_ROUNDS=3`, `NO_SELECTION_PATIENCE=3`, `MAX_SELECTED_ROUNDS=0`, `NUM_CANDIDATES=8`, `MAX_STEPS=100`, `SEED_MAX_STEPS=0`, `SBATCH_MEM=48G`, and `SBATCH_TIME=01:00:00`.
- Output root: `artifacts/runs/adaptive_candidate_format_pilot_3a_20260621_110338`.
- Jobs: addition `10040839`; run_length `10040840`.
- Both jobs started on `della-i22g2`; launcher logs confirmed `PROPOSAL_OUTPUT_SCHEMA=action_observation`, `PROPOSAL_UPDATE_LOSS_MODE=merged_agent`, and `PROPOSAL_GRPO_SPAN=reasoning_action`.

### Implementation Log: 2026-06-21 15:19:20 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_format_pilot_3a_20260621_110338/run_length-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 18:27:39 UTC

- Added pre-training candidate action deduplication in `build_candidate_work_items`: config actions are keyed by ordered `(left, right, guard, target)` and executable/program actions additionally include condition/code.
- Duplicate valid actions are now skipped before composed data construction, pseudolabeling, candidate worker dispatch, task SFT, and candidate evaluation.
- Wrote per-attempt `candidate_action_dedup.json` plus `candidate_dedup_*` annotations in `proposal_results.json` so notebooks can distinguish raw proposal validity from distinct trained/scored candidates.
- Updated proposal-GRPO trace construction to skip candidate-deduped rows instead of assigning them fake `valid_untrained=0.0` rewards, which could otherwise override a trained negative-reward duplicate.
- Updated outcome trace construction to skip candidate-deduped rows so observation CE does not label them as data-build failures.
- Verification: `python -m py_compile self/adaptive/candidate.py self/adaptive/proposal.py self/adaptive/traces.py tests/test_adaptive_candidate_training.py`; focused pytest for candidate/GRPO dedup (`3 passed`); broader `tests/test_adaptive_candidate_training.py tests/test_candidate_worker_payloads.py tests/test_candidate_dispatch_runtime.py -q` (`59 passed`, `3` existing multiprocessing fork warnings).

### Implementation Log: 2026-06-21 15:19:59 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_format_pilot_3a_20260621_110338/addition-config-numeric-n8-reward-outcome-history-4-grpo-skip`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 15:46:35 UTC

- Relaxed the config proposal trace format from strict `<reasoning>...</reasoning><action>...</action>` to free-form reasoning text plus exactly one `<action>...</action>` block.
- The validator now parses and executes only the JSON inside `<action>`; surrounding text is retained as proposal notes rather than requiring a separate reasoning tag.
- Normalized selected proposal completions now serialize as brief reasoning followed by the canonical `<action>` block.
- Removed proposal history mode entirely from prompts, CLI args, launcher exports, submission matrix/manifests, and tests; current prompts no longer include recent evaluated actions.
- Verified launcher syntax with `bash -n` for the adaptive candidate sbatch and submitter.
- Verified focused behavior with `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py tests/test_attempt_candidate_runtime.py tests/test_adaptive_args_normalization.py tests/test_proposal_generation.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py -q` (`71 passed`, `3 warnings`).

### Implementation Log: 2026-06-21 16:12:42 UTC

- Submitted the free-reasoning action-format 3-attempt pilot with `MAX_ATTEMPT_ROUNDS=3`, `NO_SELECTION_PATIENCE=3`, `MAX_SELECTED_ROUNDS=0`, `NUM_CANDIDATES=8`, `MAX_STEPS=100`, `SEED_MAX_STEPS=0`, `SBATCH_MEM=48G`, and `SBATCH_TIME=01:00:00`.
- Output root: `artifacts/runs/adaptive_candidate_format_pilot_3a_free_reasoning_20260621_115533`.
- Jobs: addition `10042967`; run_length `10042968`.
- Launcher logs confirmed `PROPOSAL_OUTPUT_SCHEMA=action_observation`, `PROPOSAL_UPDATE_LOSS_MODE=merged_agent`, `PROPOSAL_GRPO_SPAN=reasoning_action`, and no proposal-history export/path suffix.
- Short monitoring: both jobs started on `della-i23g1`, passed model load and seed training startup, and were still running at roughly 16.5 minutes.
- Current artifacts during monitoring: addition attempt 1 had `1/8` valid proposals and one candidate metric; addition attempt 2 had `0/8` valid proposals. run_length attempt 1 had `0/8` valid proposals, attempt 2 had `1/8`, and attempt 3 had `2/8`.

### Implementation Log: 2026-06-21 16:14:09 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_format_pilot_3a_free_reasoning_20260621_115533/addition-config-numeric-n8-reward-outcome-grpo-skip`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `1`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 16:16:56 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_format_pilot_3a_free_reasoning_20260621_115533/run_length-config-numeric-n8-reward-outcome-grpo-skip`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 32]`.
- Selected proposal traces retained for replay: `2`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 18:46:04 UTC

- Added an optional candidate evaluation backend: `--candidate-eval-backend {transformers,vllm}`. The default remains `transformers`.
- Added vLLM knobs: `--vllm-python-bin`, `--vllm-gpu-memory-utilization`, and `--vllm-dtype`; the AILAB launcher wires them from `CANDIDATE_EVAL_BACKEND`, `VLLM_PYTHON_BIN`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_DTYPE`.
- Implemented `self/core/vllm_evaluation.py` as a subprocess-only evaluator. Candidate training saves the model, releases the in-process HF model/tokenizer, clears CUDA cache, and then runs `python -m self.core.vllm_evaluation --spec ...` in the vLLM environment.
- The subprocess writes `vllm_eval_spec.json`, `vllm_eval_result.json`, `vllm_eval_summary.json`, `vllm_eval_stdout.txt`, and `vllm_eval_stderr.txt`; the scorer writes a backend-agnostic `candidate_eval_summary.json`.
- Preserved the existing model cleanup protocol: unselected candidate model directories are still deleted by `CheckpointManager.cleanup_unselected_candidates`; selected model handling is unchanged.
- Created a project-local vLLM environment at `/scratch/gpfs/BRENDEN/changho/envs/adaptive-vllm` with `vllm==0.23.0` and `torch==2.11.0+cu130`. The temporary uv cache was removed after installation.
- Verification: `python -m py_compile self/core/vllm_evaluation.py self/adaptive/candidate.py self/adaptive/args.py self/launcher_manifests.py`; `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_vllm_eval tests/test_vllm_evaluation.py tests/test_candidate_training_runtime.py tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py -q` (`17 passed`); `PYTHONPATH=. conda run -n torch-env pytest --basetemp=.pytest_tmp_adaptive_vllm tests/test_adaptive_candidate_training.py tests/test_candidate_worker_inputs.py tests/test_candidate_worker_pack_runtime.py tests/test_candidate_dispatch_runtime.py tests/test_candidate_metric_collection.py tests/test_vllm_evaluation.py -q` (`65 passed`, `3` existing multiprocessing fork warnings).
- vLLM env probe: `PYTHONPATH=. /scratch/gpfs/BRENDEN/changho/envs/adaptive-vllm/bin/python -c "import vllm; import torch; from self.core.task_protocols import task_for_name; ..."` reported `vllm 0.23.0`, `torch 2.11.0+cu130`, CUDA visible, and `RunLengthTask`.
- Submitted one-attempt run-length eval-backend benchmark under `artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604`.
- Benchmark job `10050706`: `transformers` eval, `CANDIDATE_LOCAL_PARALLELISM=2`, `CANDIDATE_LOCAL_PACK_SIZE=2`, `CANDIDATE_LOCAL_CACHE_BASE_STATE=1`.
- Benchmark job `10050707`: `vllm` eval, `CANDIDATE_LOCAL_PARALLELISM=1`, `CANDIDATE_LOCAL_PACK_SIZE=1`, `VLLM_PYTHON_BIN=/scratch/gpfs/BRENDEN/changho/envs/adaptive-vllm/bin/python`.
- Short monitoring immediately after submission: both jobs were pending in the AILAB queue with reason `(None)` and had not emitted launcher logs yet.

### Analysis Log: 2026-06-21 16:23:45 UTC

- Slurm results: addition job `10042967` completed in `00:18:30`; run_length job `10042968` completed in `00:21:16`; both exited `0:0` with requested `48G` CPU memory and roughly `14GB` MaxRSS.
- Addition validity by attempt: `1/8`, `0/8`, `0/8`; selected one candidate, composing `3 + 7 -> 10` with `reject_boundary_carry`.
- Addition selected candidate improved average eval accuracy from `0.3897` to `0.4117`; frontier accuracy delta `+0.0271`; selected reward `0.0293`.
- run_length validity by attempt: `0/8`, `1/8`, `2/8`; selected two candidates, both based on `16 + 16 -> 32` with `require_boundary_continue`.
- run_length improved average eval accuracy from `0.4082` to `0.4503` after attempt 2 and to `0.4700` after attempt 3; selected rewards `0.0470` and `0.0262`.
- Format diagnosis: the relaxed free-reasoning parser fixed the missing `<reasoning>` tag failure mode, but proposal validity remains low because many generations omit `<action>`, emit markdown/code fences, or copy instruction/meta text into the reasoning.

### Implementation Log: 2026-06-21 16:47:01 UTC

- Changed the active `action_observation` proposal surface from free reasoning plus `<action>` tags to one flat JSON object with `reasoning`, `left`, `right`, and `guard`.
- The driver still executes only `left/right/guard`, stores `reasoning` as proposal notes, and trains the same merged-agent GRPO + observation CE + format CE objective.
- Updated normalized valid traces to serialize as compact JSON with `reasoning` first.
- During the first JSON-only pilot (`10044816`, `10044817`), monitoring showed repeated fenced JSON completions that contained a valid first object but failed because the extractor parsed from the first `{` to the last `}` across repeated snippets.
- Cancelled those flawed pilot jobs and fixed `extract_json_object` to return the first complete JSON object found in the completion; added a regression test for repeated fenced JSON with a truncated trailing object.
- Verified with `PYTHONPATH=. /home/cs1095/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py tests/test_attempt_candidate_runtime.py tests/test_adaptive_args_normalization.py tests/test_proposal_generation.py tests/test_candidate_worker_payloads.py tests/test_candidate_worker_specs.py -q` (`72 passed`, `3 warnings`).
- Resubmitted fixed-parser 3-attempt pilots at output root `artifacts/runs/adaptive_candidate_format_pilot_3a_json_reasoning_firstjson_20260621_124408`; jobs are addition `10045315` and run_length `10045316`.
- Short monitoring: both fixed-parser jobs were still pending on priority at submission time; no launcher logs had been created yet.

### Implementation Log: 2026-06-21 17:21:24 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_candidate_format_pilot_3a_json_reasoning_firstjson_20260621_124408/run_length-config-numeric-n8-reward-outcome-grpo-skip`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 19:29:54 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-transformers-lp2`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `8`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 19:57:25 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-vllm-lp1`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `8`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 20:54:06 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-vllm-lp1-fixed`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `8`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 21:03:24 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `1`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-vllm-smoke-cudafix`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `0`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 21:08:12 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `1`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-vllm-smoke-seedckpt-cudafix`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `0`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Manual Note: vLLM Candidate-Eval Smoke, 2026-06-21

- Initial full vLLM benchmark failed inside flashinfer JIT: first because `ninja` was not on the vLLM subprocess `PATH`, then because flashinfer wrote to `~/.cache/flashinfer` and could not find CUDA sampling headers during JIT compilation.
- Updated the vLLM subprocess wrapper to prepend the vLLM env `bin` directory, redirect flashinfer workspace under `artifacts/cache/vllm/flashinfer`, and expose CUDA include dirs through `CUDA_HOME`/`CUDA_PATH`/`CPATH`/`CPLUS_INCLUDE_PATH`.
- A smoke run with `treat_seed_as_round_zero` and the raw base model did not reach vLLM because no pseudo-labels were retained.
- A second smoke run used the existing trained seed checkpoint as `MODEL_NAME`, still with `treat_seed_as_round_zero`, plus a one-candidate fixture. It reached vLLM successfully and wrote `vllm_eval_summary.json`/`vllm_eval_result.json`.
- Successful smoke job: `10054314`; output: `artifacts/runs/adaptive_vllm_eval_benchmark_20260621_144604/run_length-vllm-smoke-seedckpt-cudafix`; Slurm elapsed: `00:03:09`; vLLM candidate eval runtime: `123.6s` for `205` examples, dominated by first-time flashinfer compile/cache warmup.

### Manual Note: vLLM Eval Optimization Pass, 2026-06-21

- Added adaptive CLI/launcher controls for vLLM candidate eval: `--vllm-flashinfer-sampler`, `--vllm-enforce-eager`, `--vllm-max-model-len`, `--vllm-max-num-seqs`, and `--vllm-max-num-batched-tokens`.
- Defaulted adaptive vLLM candidate eval to `--vllm-flashinfer-sampler off` because our deterministic short-output scoring does not need FlashInfer sampling and the previous smoke run spent most of its time in FlashInfer JIT/cache setup.
- The vLLM subprocess now sets repo-local vLLM/FlashInfer cache paths, disables standalone compile/pregrad passes by default when unset, and records worker/runtime/settings in `vllm_eval_summary.json`.
- Focused verification passed: `tests/test_vllm_evaluation.py`, `tests/test_candidate_training_runtime.py`, `tests/test_adaptive_args_normalization.py`, and `tests/test_adaptive_candidate_launcher.py`.
- One-candidate smoke benchmark output: `artifacts/runs/adaptive_vllm_eval_optimized_20260621_181952`.
- Jobs completed: sampler-off vLLM `10056669` (`00:02:37` Slurm elapsed), eager-small vLLM `10056670` (`00:02:04`), matched Transformers comparison `10056807` (`00:01:04`).
- Candidate eval runtime on the same 205 run-length examples: sampler-off vLLM `78.9s`, eager-small vLLM `48.8s`, Transformers `3.8s`.
- Conclusion for current small per-candidate eval: optimized vLLM is much better than the first FlashInfer-JIT smoke (`123.6s`) but still not competitive with in-process Transformers because vLLM process/model startup dominates.

### Implementation Log: 2026-06-21 22:22:42 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `1`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_optimized_20260621_181952/eager-small`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `0`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 22:23:12 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `1`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_optimized_20260621_181952/sampler-off`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `0`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-21 22:25:51 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `1`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_vllm_eval_optimized_20260621_181952/transformers-small`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `none`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `0`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 00:22:31 UTC

- Reduced proposal format-CE pressure on concrete actions by adding `--proposal-format-mask-config-values` / `--no-proposal-format-mask-config-values`, defaulting to masked values.
- In merged-agent proposal updates, format replay still trains JSON braces, keys, colons, commas, and string delimiters, but masks the contents/values for `reasoning`, `notes`, `rationale`, `left`, `right`, `guard`, and legacy `target`.
- GRPO policy samples and driver-appended observation CE samples are unchanged; the masking applies only to `proposal_format_targets.jsonl` replay samples.
- Proposal-GRPO metrics now log `format_mask_config_values`, `format_masked_value_token_counts`, `format_completion_token_counts`, and `format_total_completion_token_counts`.
- Verification: `python -m py_compile self/adaptive/args.py self/adaptive/proposal.py tests/test_adaptive_candidate_training.py tests/test_adaptive_args_normalization.py`; focused pytest for parser defaults, format masking, action-span masking, observation completion, and argument normalization passed (`8` tests total).

### Implementation Log: 2026-06-22 00:31:44 UTC

- Changed config candidate reward to local average-accuracy improvement: `candidate_avg_accuracy - current_avg_accuracy`.
- Kept `frontier_delta`, `target_delta`, and init-relative average delta in candidate metrics and observations for analysis, but they no longer enter the scalar candidate reward used for selection or proposal-GRPO outcome rewards.
- Updated the proposal prompt reward formula and the old dry-run experiment summary text to avoid advertising the previous static-frontier/init-relative reward.
- Verification: `python -m py_compile self/adaptive/candidate.py self/adaptive/attempts.py self/adaptive/proposal.py self/experiments/adaptive_self_improvement.py tests/test_candidate_rewards.py tests/test_adaptive_candidate_training.py`; focused pytest for candidate reward metrics and config prompt formula passed (`4` tests).

### Implementation Log: 2026-06-22 01:02:32 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_avg_delta_pilot_3a_20260621_203811/run_length-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `21`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 01:05:11 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_candidate_avg_delta_pilot_3a_20260621_203811/addition-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `3`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 01:28:00 UTC

- Changed default proposal sampling temperature from `0.8` to `0.9`.
- Added `--force-unique-proposals` for config generation. When enabled, the driver samples until it returns `num_candidates` unique normalized config actions `(left, right, guard, target)` or exhausts `--proposal-unique-max-draws`.
- Launcher defaults now set `FORCE_UNIQUE_PROPOSALS=1`, `PROPOSAL_TEMPERATURE=0.9`, `PROPOSAL_TOP_P=0.95`, and `PROPOSAL_UNIQUE_MAX_DRAWS=0` (`0` auto-expands to `max(8 * num_candidates, num_candidates + 16)`).
- Each attempt with forced uniqueness writes `proposal_unique_sampling.json`, including total raw draws, unique valid action count, whether the requested unique count was reached, and per-draw validation reasons.
- Candidate-action dedup remains in place after validation/training dispatch; the new sampling stage is earlier and is meant to reduce duplicate candidate training attempts.
- Verification: `PYTHONPATH=. ~/.conda/envs/torch-env/bin/python -m pytest -q tests/test_proposal_generation.py tests/test_adaptive_candidate_training.py::test_parser_defaults_enable_numeric_outcome_and_config_grpo tests/test_adaptive_candidate_launcher.py` passed (`9` tests).

### Implementation Log: 2026-06-22 07:22:28 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_unique_temp09_25a_20260621_213009/addition-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `30`.
- Proposal GRPO updates: `4`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 08:33:34 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_unique_temp09_25a_20260621_213009/run_length-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 31, 32, 40, 48]`.
- Selected proposal traces retained for replay: `21`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `195`.
- Proposal GRPO updates: `23`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 09:20:00 UTC

- Added `self/diagnostics/proposal_state_sensitivity.py` to test whether proposal generation is conditioned on the state prompt.
- The diagnostic loads a saved `proposal_prompt.json`, keeps the same prompt template, and creates swapped-state variants by changing `current_source_slices`, `allowed_target_frontier`, and the aggregate per-size accuracy table.
- It reports prompt token count/context fit, greedy proposal behavior, optional sampled proposal behavior, and normalized action log-likelihood rankings for valid hypothetical actions.
- Added diagnostic safeguards: `--num-samples 0` skips sampling for faster greedy/logprob-only probes; `--logprob-batch-size` chunks scoring to avoid diagnostic-only GPU OOM; `--max-logprob-candidate-actions` caps broad-frontier action ranking while preserving observed/preferred actions when valid.
- Addition probe output: `artifacts/analysis/proposal_state_sensitivity/addition_attempt1_final.json`. Prompt length was about `900` tokens with a `40960` token effective context. Greedy/sampled generation kept emitting `(5, 3, 8, none)` even when the frontier was changed to exactly `9` or `14`; logprob rankings did move to valid frontier actions under strict frontier swaps.
- Run-length probe output: `artifacts/analysis/proposal_state_sensitivity/run_length_attempt8_final.json`. Prompt length was about `805-1113` tokens with a `40960` token effective context. Greedy generation produced invalid stale text/actions under all swapped states; logprob rankings changed with frontier constraints but accuracy-only swaps still favored the old observed action `(16, 24, 40, none)`.
- Interpretation: context length is not the limiting issue. The current proposal policy has a state-conditioning failure at the decoded-action level, with some frontier sensitivity visible in normalized action logprobs but weak use of per-size accuracy/state details.

### Implementation Log: 2026-06-22 09:55:00 UTC

- Prepared the Qwen3-4B adaptive pilot.
- Cached non-base `Qwen/Qwen3-4B` into the scratch Hugging Face cache used by AILab offline jobs: `/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3-4B`.
- Offline preflight passed for config/tokenizer from scratch cache: `qwen3`, `36` layers, hidden size `2560`, max positions `40960`, tokenizer size `151669`.
- Updated adaptive candidate submitter/manifests to explicitly export and record `MODEL_NAME` and `PROPOSAL_MODEL_NAME`; this makes model-size experiments auditable instead of relying on ambient `--export ALL`.
- Verification: `bash -n launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch`; `py_compile` for `self/launcher_manifests.py` and the proposal-state diagnostic; focused pytest for launcher/manifests passed (`7` tests).
- Dry-run submission with `MODEL_NAME=Qwen/Qwen3-4B` confirmed Slurm exports and manifest fields.
- Submitted 3-attempt Qwen3-4B AILab pilot with conservative one-candidate-worker settings: `MAX_ATTEMPT_ROUNDS=3`, `NUM_CANDIDATES=8`, `MAX_STEPS=100`, `CANDIDATE_LOCAL_PARALLELISM=1`, `CANDIDATE_LOCAL_PACK_SIZE=1`, `CANDIDATE_LOCAL_CACHE_BASE_STATE=0`, `SBATCH_MEM=64G`, `SBATCH_TIME=06:00:00`.
- Output root: `artifacts/runs/adaptive_qwen3_4b_pilot_3a_20260622_095411`.
- Submitted jobs: addition `10094651`, run_length `10094652`. Initial monitoring showed both jobs pending in AILab with reason `Priority`, so no runtime logs were available yet.

### Implementation Log: 2026-06-22 11:10:00 UTC

- Qwen3-4B pilot jobs completed successfully.
- Slurm: addition `10094651` completed in `00:53:57`; run_length `10094652` completed in `00:56:52`; both exit code `0:0`; CPU RSS about `19-20GB` under `64G` allocation.
- Addition result: `3` attempts, `0` selected candidates, `0` proposal-GRPO updates. Initial/final held-out average stayed at `0.6007`; all trained valid candidate actions were variants of `(5, 3)->8` or `(3, 5)->8` and had negative local average-accuracy rewards.
- Addition proposals: per attempt, 3-5 of 8 returned valid parsed actions, but duplicates dominated and parse errors remained common.
- Run-length result: `3` attempts, `3` selected candidates, `2` proposal-GRPO updates. Initial held-out average was `0.4452`; selected candidate averages moved to `0.6973`, `0.7464`, and `0.8015`.
- Run-length selected actions: attempt 1 `(15, 16)->31`, reward `+0.2521`; attempt 2 `(16, 8)->24`, reward `+0.0503`; attempt 3 `(16, 24)->40`, reward `+0.0506`.
- Qwen3-4B state-sensitivity diagnostics were written to `artifacts/analysis/proposal_state_sensitivity/qwen3_4b_addition_attempt1_final.json` and `artifacts/analysis/proposal_state_sensitivity/qwen3_4b_run_length_attempt3_final.json`.
- Diagnostic interpretation: model context is still fine (`~900-1100` prompt tokens vs `40960` effective context). Qwen3-4B logits react to strict frontier swaps, but decoded greedy behavior still fails state-conditioning. Addition greedy outputs prose and no parseable JSON; run_length original state emits valid `(16, 24)->40`, but swapped-frontier states still decode stale out-of-frontier actions even when logprob ranking favors the valid frontier action.

### Implementation Log: 2026-06-22 14:54:20 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_qwen3_4b_pilot_3a_20260622_095411/addition-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `17`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 14:57:14 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_qwen3_4b_pilot_3a_20260622_095411/run_length-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 31, 40]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `18`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.1`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 15:20:00 UTC

- Fixed a proposal-prompt leakage bug in the config `action_observation` schema.
- The previous prompt included a concrete pseudo-example with `left=5`, `right=3`, and target `8`; addition pilots then repeatedly proposed `(5, 3)->8` / `(3, 5)->8`.
- The prompt is now schema-only: it lists required keys (`reasoning`, `left`, `right`, `guard`) and tells the model to choose all numeric values from `current_source_slices` and `allowed_target_frontier`.
- Added a lightweight regression test to ensure the concrete `(5, 3)->8` exemplar does not return.
- Verification: `PYTHONPATH=. pytest -q tests/test_adaptive_proposals_and_sandbox.py` passed (`9` tests); `python -m py_compile self/adaptive/proposal.py` passed.
- Iteration note: before full candidate-training loops, use prompt-only and validation-only smoke runs to catch this class of issue. The cheap sequence should be: render prompt, sample proposals, parse/validate/deduplicate, inspect action histogram/state-sensitivity, then train at most one candidate if the proposal distribution is sane.

### Implementation Log: 2026-06-22 16:00:00 UTC

- Added `self/diagnostics/proposal_smoke.py`, a prompt-only diagnostic that renders the current fixed prompt, samples proposal completions, validates/deduplicates actions, writes action histograms, and emits a small synthetic run directory for state-sensitivity probes.
- Ran Qwen3-4B fixed-prompt smoke diagnostics without candidate training.
- Artifact root: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/analysis/proposal_smoke/fixed_prompt_qwen3_4b_20260622_153630`.
- Addition smoke (`32` samples, temp `0.9`): `13/32` valid, `8` unique valid actions. The old hard collapse to `(5, 3)->8` disappeared; top actions were `(3,5)->8` (`3`), `(5,7)->12` with carry guard (`2`), `(3,5)->8` with carry guard (`2`), and `(5,6)->11` (`2`). Remaining failures were mostly parse errors (`11`), range errors (`6`), and schema errors (`2`).
- Run-length smoke (`32` samples, temp `0.9`): `25/32` valid, but only `5` unique valid actions. Distribution remained concentrated on `(16,24)->40` with `require_boundary_continue` (`11`) and `(40,8)->48` with `require_boundary_continue` (`8`).
- Addition state-sensitivity: prompt length `~928` tokens. Greedy moved correctly to `(3,5)->8` when the frontier was exactly `8`. For frontier exactly `14`, normalized action logprobs ranked `(7,7)->14` highest, but greedy still emitted an invalid lower-sum action. Accuracy-only swaps did not reliably move greedy actions toward the preferred `(7,7)->14` target.
- Run-length state-sensitivity: prompt length `~1051` tokens. Original greedy stayed at `(16,24)->40`; frontier-only swaps changed logprob rankings to the valid exact-frontier actions but greedy often produced parse/range errors. When both frontier and accuracy preferred target `47`, greedy emitted valid `(16,31)->47`.
- Interpretation: removing the concrete example helped addition diversity, but the remaining bottlenecks are strict output formatting, weak decoding-level state conditioning, and an overstrong run-length prior toward previously successful actions/guards. Prompt-only diagnostics took about `3.6-3.7` minutes per task with `32` samples on the local CUDA environment; future smoke tests should default to `8-16` samples or use a batched/vLLM proposal backend.

### Implementation Log: 2026-06-22 16:45:00 UTC

- Implemented strict valid-unique config proposal sampling for candidate training.
- Candidate training now receives only kept valid unique `left/right/guard/target` actions. Invalid parses and duplicate valid actions are rejected instead of being backfilled into candidate slots.
- Added batched proposal draws through `--proposal-sampling-batch-size` (default `8` in launch configs) so rejection sampling does not call `model.generate` one sample at a time.
- Added `proposal_draw_results.json` for all validated proposal draws, including invalid and duplicate samples, while keeping `proposal_results.json` as the candidate-facing kept set.
- Wired proposal-GRPO to use all draw results when available. Kept draw rows carry `candidate_proposal_index`, so downstream candidate metrics still attach to the trained candidate even when the draw index differs from the candidate index.
- If every trainable GRPO draw is invalid and rewards have zero variance, the default `skip` mode is overridden to a fixed-baseline update so malformed-only batches still create a negative training signal.
- Disabled merged-agent format CE by default (`PROPOSAL_FORMAT_LOSS_WEIGHT=0.0`) and skip building format replay targets when the weight is zero. Formatting pressure now primarily comes from rejection sampling plus invalid-draw GRPO.
- Updated launchers/config defaults to export `PROPOSAL_SAMPLING_BATCH_SIZE=8` and the new zero format-CE default.
- Verification: `py_compile` passed for adaptive proposal/controller/run/traces/args; focused torch-env pytest passed for proposal generation, launcher wiring, args normalization, and proposal-GRPO trace tests (`21` tests total).

### Submission Log: 2026-06-22 13:25 EDT

- Submitted a 3-attempt model-size pilot comparing `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`.
- Common settings: `task={addition,run_length}`, `condition=config`, `num_candidates=8`, `MAX_ATTEMPT_ROUNDS=3`, `NO_SELECTION_PATIENCE=3`, `MAX_STEPS=100`, strict unique proposal sampling, `PROPOSAL_SAMPLING_BATCH_SIZE=8`, `PROPOSAL_TEMPERATURE=0.9`, `PROPOSAL_TOP_P=0.95`, `PROPOSAL_GRPO_REWARD_MODE=outcome`, `PROPOSAL_GRPO_ZERO_VARIANCE=skip`, `PROPOSAL_FORMAT_LOSS_WEIGHT=0.0`.
- Output root: `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528`.
- 1.7B jobs: addition `10102327`, run_length `10102328`; resources `48G`, `03:00:00`, local candidate parallelism `2`, pack size `2`, cache base state on.
- 4B jobs: addition `10102339`, run_length `10102340`; resources `64G`, `04:00:00`, local candidate parallelism `1`, pack size `1`, cache base state off.
- Initial monitoring after submission: all four jobs were pending on `ailab` with reason `Priority`; no stdout/stderr logs had started yet.

### Implementation Log: 2026-06-22 17:10 EDT

- Re-enabled weak proposal format CE by default after the 3-attempt model-size pilot still showed substantial malformed/prose-heavy proposal draws under strict rejection sampling.
- New default: `PROPOSAL_FORMAT_LOSS_WEIGHT=0.02` in Python defaults, the AILab runner, submitter, and shared adaptive config.
- Kept `PROPOSAL_FORMAT_MASK_CONFIG_VALUES=1`, so format CE masks concrete config values and mainly trains structural JSON/key/delimiter tokens. This is intended to improve parseability without strongly copying the selected `left/right/guard` action.
- Existing runs from `adaptive_qwen3_model_compare_3a_20260622_132528` used the old `0.0` default; reruns are needed to test this change.

### Implementation Log: 2026-06-22 19:11:28 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528/qwen3-1.7b/run_length-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 31, 47]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `173`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.0`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 19:24:39 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528/qwen3-4b/run_length-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 31, 47]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `192`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.0`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 19:48:03 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528/qwen3-1.7b/addition-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10, 17, 27]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `114`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.0`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 20:46:40 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `3`; max selected candidates: `unlimited`; attempts used: `3`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528/qwen3-4b/addition-config-numeric-n8-reward-outcome-grpo-skip-eval-transformers`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 14, 21, 28]`.
- Selected proposal traces retained for replay: `3`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `97`.
- Proposal GRPO updates: `2`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.0`.
- Keep all proposal-GRPO checkpoints: `False`.

### Analysis Log: 2026-06-22 17:45 EDT

- Updated `notebooks/adaptive_qwen3_model_compare_candidate_analysis.ipynb` with intended-behavior diagnostics in addition to aggregate accuracy, candidate filtering, and selected-action tables.
- New diagnostics track whether selected candidates use newly learned source sizes, whether the selected target is novel relative to the current source pool, and whether a round repeats an already learned target.
- In the 3-attempt model-size pilot, both addition runs show clean bootstrapping chains: 1.7B selects `10 -> 17 -> 27`, and 4B selects `14 -> 21 -> 28`, using learned source sizes after attempt 1.
- Both run-length runs improve accuracy but repeat target `47` on attempt 3 with a different guard, so their accuracy gains should be separated from acquisition-policy success.
- The pilot used `PROPOSAL_FORMAT_LOSS_WEIGHT=0.0`; the weak masked format CE default (`0.02`) was implemented after this run and still needs a rerun to evaluate format robustness.

### Runtime Analysis Log: 2026-06-22 18:15 EDT

- Added `notebooks/adaptive_runtime_disaggregation.ipynb` to parse adaptive run artifacts and summarize wall time, seed training time, candidate-dispatch wall estimates, summed candidate SFT time, and summed candidate evaluation time.
- For the 3-attempt Qwen3 model-size pilot, candidate evaluation dominated candidate SFT. Addition was the slowest case: 1.7B addition spent about `84.5` summed candidate-eval minutes versus `12.7` summed candidate-SFT minutes; 4B addition spent about `62.6` versus `14.6`.
- Run-length candidate evaluation was cheaper but still larger than candidate SFT: 1.7B run-length `15.2` eval minutes versus `5.1` SFT minutes; 4B run-length `9.8` versus `3.8`.
- Added future-run instrumentation in `self/adaptive/attempts.py` to write `attempt_timing.json` per attempt with exact phase timings for round/proposal dispatch, candidate dispatch, selection/trace/cleanup, outcome/GRPO, and total attempt time.

### Seed-Cache Semantics: 2026-06-22 18:30 EDT

- The adaptive loop trains/evaluates the seed model before attempt 1; the first proposal is generated from the seed-trained checkpoint, not from the raw base model.
- Cached seed reuse is therefore valid across proposal-prompt, output-schema, proposal-GRPO, observation-loss, format-CE, reward-scaling, and candidate-selection experiments, as long as the seed-stage model/data/training recipe is unchanged.
- Seed checkpoints should be invalidated when any seed-stage ingredient changes: task, base model, tokenizer mode, task output/target format, initial source size range, initial train/eval counts, random seed, seed max steps/epochs, learning rate, train batch size, precision, or recipe/init-from-scratch settings.
- Suggested naming convention: `artifacts/seed_cache/adaptive/{task}/{model_slug}/init{min}-{max}_frontier{frontier}_train{train_per_size}_eval{eval_per_size}_target{target_mode}_tok{tokenizer}_seed{seed}_steps{seed_steps}_lr{lr}_bs{batch}/model`.
- To reuse a cached seed checkpoint in the adaptive launcher, set `TREAT_SEED_AS_ROUND_ZERO=1` and set `MODEL_NAME` to the cached `.../model` path. In that mode the driver evaluates `MODEL_NAME` as round 0 and attempt 1 proposes from it directly.
- Made `TREAT_SEED_AS_ROUND_ZERO` explicit in `adaptive_candidate_base.env`, Slurm logs, and the submitter export so cached-seed runs are auditable.

### Seed-Only Cache Jobs: 2026-06-22 19:25 EDT

- Submitted seed-checkpoint-only jobs for `{addition, run_length} x {Qwen3-1.7B, Qwen3-4B}` using the normal adaptive runner with `MAX_ATTEMPT_ROUNDS=0`, `MAX_SELECTED_ROUNDS=0`, and `NO_SELECTION_PATIENCE=1`.
- These jobs train and evaluate only `round_00/seed_training/model`; no proposals, candidate SFT, selection, or proposal updates should run.
- Common seed recipe: `SEED_MAX_STEPS=0` epoch-based seed training, `LEARNING_RATE=5e-6`, `TRAIN_BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `BF16=1`, `TREAT_SEED_AS_ROUND_ZERO=0`.
- Submission manifest: `artifacts/seed_cache/adaptive/seed_cache_submission_20260622.tsv`.
- Job IDs: `10113852` addition/Qwen3-1.7B, `10113853` run_length/Qwen3-1.7B, `10113854` addition/Qwen3-4B, `10113855` run_length/Qwen3-4B.
- Cache output roots:
  - `artifacts/seed_cache/adaptive/addition/qwen3-1.7b/init3-7_frontier31_train5000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16`
  - `artifacts/seed_cache/adaptive/run_length/qwen3-1.7b/init8-16_frontier48_train50000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16_targetrun_state_alpha2`
  - `artifacts/seed_cache/adaptive/addition/qwen3-4b/init3-7_frontier31_train5000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16`
  - `artifacts/seed_cache/adaptive/run_length/qwen3-4b/init8-16_frontier48_train50000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16_targetrun_state_alpha2`

### Eval Throughput Change: 2026-06-22 19:50 EDT

- Candidate evaluation was slower than 100-step candidate SFT because `transformers` eval runs autoregressive `model.generate` over the full held-out size grid.
- Increased adaptive default `EVAL_BATCH_SIZE` from `16` to `128` in Python and AILab launcher defaults.
- Updated `evaluate_accuracy_with_breakdown` to size-bucket examples before generation batching, preserving accuracy metrics while reducing padding waste for larger eval batches.
- Verification: focused generation/eval and candidate runtime tests passed (`7 passed`), and adaptive launcher shell syntax checks passed.

### Cached-Seed Eval Batch Smoke: 2026-06-22 19:14 EDT

- Submitted one-attempt cached-seed smoke jobs to exercise the real adaptive pipeline with `EVAL_BATCH_SIZE=128` and size-bucketed generation eval.
- Output root: `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406`.
- Jobs: `10114604` addition/Qwen3-1.7B, `10114605` run_length/Qwen3-1.7B, `10114606` addition/Qwen3-4B, `10114607` run_length/Qwen3-4B.
- The run_length/Qwen3-4B smoke job was submitted with dependency `afterok:10113855` on the seed-only cache job.
- Early monitoring: the running jobs loaded cached seed checkpoints on compute nodes, skipped seed training via `TREAT_SEED_AS_ROUND_ZERO=1`, reached adaptive attempt 1, and logged `Train/eval batch size: 16/128`.
- First candidate eval summaries confirm real candidate eval used `batch_size=128`: addition candidates over `2900` examples finished in about `27-29s`; run_length candidates over `3300` examples finished in about `12-43s`.

### Submission Log: 2026-06-22 19:40 EDT

- Submitted cached-seed 25-attempt adaptive config jobs with `EVAL_BATCH_SIZE=128` and hard `02:30:00` walltime.
- Output root: `artifacts/runs/adaptive_cached_seed_25a_evalbs128_20260622_193946`.
- Jobs: `10115214` addition/Qwen3-1.7B, `10115215` run_length/Qwen3-1.7B, `10115216` addition/Qwen3-4B, `10115217` run_length/Qwen3-4B.
- Common settings: `TREAT_SEED_AS_ROUND_ZERO=1`, `MAX_ATTEMPT_ROUNDS=25`, `MAX_SELECTED_ROUNDS=0`, `NO_SELECTION_PATIENCE=25`, `NUM_CANDIDATES=8`, `MAX_STEPS=100`, `TRAIN_BATCH_SIZE=16`, `CANDIDATE_EVAL_BACKEND=transformers`.
- Initial monitoring: all four jobs were pending on `ailab` with reason `Priority`; no job logs had started yet.

### Implementation Log: 2026-06-22 22:57:26 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `0`; max selected candidates: `unlimited`; attempts used: `0`; candidates per attempt: `1`.
- Output directory: `artifacts/seed_cache/adaptive/addition/qwen3-1.7b/init3-7_frontier31_train5000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-23 03:33:21 UTC

- Added explicit config-proposal decision rules to the prompt:
  reliable source sizes, weak reachable targets, local average-accuracy improvement,
  source-pool growth, conditional exact-repeat avoidance, and task-specific guard failure modes.
- Added `--proposal-prompt-action-history` / `PROPOSAL_PROMPT_ACTION_HISTORY` as an ablation knob.
  Default is off; when enabled, prompts include only a compact recent selected-action summary
  (`left`, `right`, `target`, `guard`, reward, and selected outcome deltas), not full trace text.
- No candidate reward/selection formula changed in this edit.

### Implementation Log: 2026-06-23 03:50:36 UTC

- Fixed the stale adaptive candidate-training prompt assertion after replacing the confusing numeric-copy instruction.
- Verification: `PYTHONPATH=. "$HOME/.conda/envs/torch-env/bin/python" -m pytest -q tests/test_adaptive_candidate_training.py`
  passed (`58 passed`, `3` existing multiprocessing fork warnings).
- Submitted 8 cached-seed Qwen3-1.7B adaptive config jobs:
  `{addition, run_length} x history {0, 1} x novelty beta {0, 0.05}`.
- Common settings: `MAX_ATTEMPT_ROUNDS=25`, `NO_SELECTION_PATIENCE=25`,
  `NUM_CANDIDATES=8`, `MAX_STEPS=100`, `TREAT_SEED_AS_ROUND_ZERO=1`,
  `SBATCH_TIME=03:00:00`, `PROPOSAL_GRPO_REWARD_MODE=outcome`,
  `PROPOSAL_GRPO_ZERO_VARIANCE=skip`, `PROPOSAL_UPDATE_LOSS_MODE=merged_agent`.
- Output root: `artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947`.
- Submission manifest: `artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947/submission_manifest.tsv`.
- Job IDs: `10127771`, `10127772`, `10127773`, `10127774`,
  `10127775`, `10127776`, `10127777`, `10127778`.
- Initial monitoring: all 8 jobs were still pending on `ailab`; no Slurm logs had been created yet.

### Implementation Log: 2026-06-23 03:17:49 UTC

- Confirmed current config jobs already use `FORCE_UNIQUE_PROPOSALS=1` in the AILAB launchers and `--proposal-grpo-deduplicate-actions` by default, so exact duplicate actions are deduplicated before candidate training and before proposal-policy GRPO credit.
- Added proposal-GRPO action entropy diagnostics: pre-dedup/trainable action entropy, effective action count, duplicate action rate, and top action counts are now written in `proposal_grpo_metrics.json`.
- Added novelty reward shaping with `--proposal-grpo-novelty-bonus-beta` defaulting to `0.05`. Valid config actions receive `beta / sqrt(count + 1)`, where `count` comes from selected-action history plus current raw duplicates; set beta to `0` for ablation.
- Added source-pool admission gating with `--source-admission-target-accuracy-threshold` defaulting to `0.80`. A selected model checkpoint still advances the learner, but the selected target and its pseudo examples enter future composition source data only when held-out target accuracy clears the threshold.
- `round_summary.json` now logs the source-admission decision, and run summaries/logs record both novelty beta and the source-admission threshold.

### Implementation Log: 2026-06-23 01:59:42 UTC

- Tightened adaptive checkpoint retention for the config self-improvement pipeline.
- Candidate workers now delete their own unselectable candidate `model/` directory immediately after scoring when `reward < selection_min_reward` or the candidate is invalid, unless `--keep-all-candidate-models` is set.
- Attempt-level cleanup now deletes only unselected candidate `model/` directories and preserves surrounding JSON/JSONL logs such as `candidate_metrics.json`, eval summaries, trace files, and proposal-rehearsal summaries.
- Added `--keep-final-model-checkpoint` / `KEEP_FINAL_MODEL_CHECKPOINT=1`. The default is logs-only finalization: if the final current checkpoint lives under the run output directory, its `model/` directory is deleted after summary/log writing.
- Existing protected caches and external seed checkpoints are not pruned by finalization because deletion is bounded to the run output directory.
- Deleted old non-live adaptive smoke-run candidate model directories under `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406`, freeing about `21.4 GiB`; current live 25-attempt jobs were left untouched.

### Implementation Log: 2026-06-22 22:58:45 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `0`; max selected candidates: `unlimited`; attempts used: `0`; candidates per attempt: `1`.
- Output directory: `artifacts/seed_cache/adaptive/addition/qwen3-4b/init3-7_frontier31_train5000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:02:20 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `0`; max selected candidates: `unlimited`; attempts used: `0`; candidates per attempt: `1`.
- Output directory: `artifacts/seed_cache/adaptive/run_length/qwen3-1.7b/init8-16_frontier48_train50000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16_targetrun_state_alpha2`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:13:56 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `0`; max selected candidates: `unlimited`; attempts used: `0`; candidates per attempt: `1`.
- Output directory: `artifacts/seed_cache/adaptive/run_length/qwen3-4b/init8-16_frontier48_train50000_initeval50_eval100_seed42_steps0_lr5e-6_bs16_bf16_targetrun_state_alpha2`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `0`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `0`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:21:09 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406/qwen3-1.7b/addition-one-attempt-evalbs128`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 10]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `64`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:22:30 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406/qwen3-1.7b/run_length-one-attempt-evalbs128`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `24`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:31:56 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406/qwen3-4b/run_length-one-attempt-evalbs128`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 31]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `64`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-22 23:32:12 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `addition`; max attempts: `1`; max selected candidates: `unlimited`; attempts used: `1`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_eval_batch128_cached_seed_smoke_20260622_191406/qwen3-4b/addition-one-attempt-evalbs128`.
- Proposal output schema: `action_observation`.
- Final source sizes tracked by driver: `[3, 4, 5, 6, 7, 13]`.
- Selected proposal traces retained for replay: `1`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `37`.
- Proposal GRPO updates: `0`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-23 08:15:19 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947/run_length-history0-novelty0`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `10`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.0`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 16:08:30 UTC

- Fixed the anchor-KL failure in the adaptive config-generation loop.
- Root cause: the seed checkpoint used as the `adaptive_init` anchor was pruned after the first proposal-GRPO update. On the next attempt, anchor-KL tried to load `round_00/seed_training/model`; because the local directory had been deleted, HuggingFace interpreted the relative path as a repo id and raised `HFValidationError`.
- Implementation: `CheckpointManager.cleanup_replaced_checkpoint` now accepts protected checkpoints, and the attempt loop protects `proposal_kl_reference_checkpoint` while the run is active. Finalization still prunes the anchor checkpoint when `keep_final_model_checkpoint=False`, so this does not keep seed checkpoints indefinitely.
- Added local-path normalization before loading a distinct anchor reference checkpoint.
- Added regression tests for protected anchor cleanup and reran targeted proposal/checkpoint/launcher tests.

### Implementation Log: 2026-06-23 Anchored Proposal KL

- Added a GRPO-standard proposal anchor KL path: `--proposal-grpo-anchor-kl-reference adaptive_init` with `--proposal-grpo-anchor-kl-coef`.
- The adaptive loop now captures the seed/adaptive-loop initial checkpoint immediately after seed training and passes it to every proposal-GRPO update as the frozen reference.
- Kept the old sampled pre-update logprob proxy as `--proposal-grpo-kl-coef`; it remains a local trust-region proxy, while the new anchor KL is the cross-attempt regularizer.
- `proposal_grpo_metrics.json` now logs `anchor_kl_enabled`, `anchor_kl_reference_checkpoint`, `anchor_kl_proxy`, and the reference mean logprob so notebooks can distinguish local KL from anchor KL.
- Launcher defaults set `PROPOSAL_GRPO_ANCHOR_KL_COEF=0.01` and `PROPOSAL_GRPO_ANCHOR_KL_REFERENCE=adaptive_init`.

### Implementation Log: 2026-06-23 08:52:18 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947/run_length-history0-novelty0p05`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 23]`.
- Selected proposal traces retained for replay: `18`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-23 08:58:37 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947/run_length-history1-novelty0`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `True`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `17`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `21`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.0`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-23 Option A Outcome Prediction

- Changed the active config `action_observation` completion target to put outcome predictions inside the same generated JSON object after `reasoning` and before the executable config fields:
  `{"reasoning": str, "expected_avg_delta_from_current": float, "expected_target_delta": float, "expected_frontier_delta": float, "left": int, "right": int, "guard": str}`.
- The driver still executes only `left`, `right`, and `guard`; predictions are stored as `parsed_prediction` / `proposal_prediction` metadata.
- Merged-agent observation CE now appends realized feedback with the stored prediction and error terms:
  `frontier_delta_error`, `target_delta_error`, `avg_delta_from_current_error`, and `avg_delta_from_init_error`.
- Format CE masks the actual reasoning, prediction numeric values, and config values while keeping JSON syntax and field names trainable. This keeps format pressure without directly imitating specific acquisition choices.
- The prompt now explicitly asks for reasoning, then expected deltas, then the config action, and removes the older “do not predict reward” language.
- Verification in the current login environment: `python -m py_compile self/adaptive/proposal.py self/adaptive/traces.py tests/test_adaptive_candidate_training.py tests/test_adaptive_proposals_and_sandbox.py`.
- Full focused pytest was not run in this shell because `torch` is unavailable in the active Python environment.

### Implementation Log: 2026-06-23 Proposal-GRPO LR / Anchor-KL Sweep Support

- Added adaptive candidate submitter sweep axes for `PROPOSAL_GRPO_LEARNING_RATES` and `PROPOSAL_GRPO_ANCHOR_KL_COEFS`, while keeping `PROPOSAL_GRPO_LEARNING_RATE` and `PROPOSAL_GRPO_ANCHOR_KL_COEF` as single-value defaults.
- Output directories, SLURM job names, stdout/stderr log names, and submission manifest keys now include `lr-*` and `akl-*` labels.
- The submission manifest now records per-job `proposal_grpo_learning_rate`, `proposal_grpo_kl_coef`, and `proposal_grpo_anchor_kl_coef`, plus top-level sweep lists.
- The planned sweep disables the old sampled-token local KL proxy by launching with `PROPOSAL_GRPO_KL_COEF=0`; anchor KL remains the only KL-style regularizer in that sweep.
- Verification: `bash -n launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch`, `python -m py_compile self/launcher_manifests.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py`, and `python -m pytest tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py -q` (`8 passed`).

### Implementation Log: 2026-06-23 Dr.GRPO Proposal Objective

- Added `--proposal-grpo-objective {grpo,dr_grpo}` / `PROPOSAL_GRPO_OBJECTIVE`, defaulting to `grpo`.
- `grpo` preserves the previous proposal policy update: std-normalized group advantages and mean logprob over trainable policy tokens.
- `dr_grpo` uses reward-minus-group-mean advantages and summed logprob over trainable policy tokens, removing reward-std and per-completion length normalization from the proposal policy term.
- Observation CE, format CE, reward shaping, zero-variance handling, deduplication, novelty bonus, and KL/anchor-KL knobs are unchanged.
- Runner, submitter, job/log/output labels, and submission manifests now record the objective with `obj-grpo` / `obj-dr-grpo` labels.
- Verification:
  - `bash -n launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch`
  - `python -m py_compile self/adaptive/proposal.py self/adaptive/args.py self/launcher_manifests.py tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_training.py tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py`
  - `python -m pytest tests/test_adaptive_candidate_launcher.py tests/test_launcher_manifests.py -q` (`9 passed`)
  - `conda run -n torch-env python -m pytest tests/test_adaptive_args_normalization.py tests/test_adaptive_candidate_training.py -k 'proposal_grpo_reward_mapping_and_advantages or proposal_policy_microbatches_match_full_batch_gradient or parser_defaults_enable_numeric_outcome_and_config_grpo or preserves_task_defaults' -q` (`4 passed`, `57 deselected`)

### Implementation Log: 2026-06-24 01:23:22 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-3em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `4`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.0`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-3em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 01:40:15 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-1em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `10`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.0`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-1em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 09:04:55 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-5em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.
- Selected proposal traces retained for replay: `16`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1583`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.0`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700/run_length-config-numeric-n8-reward-outcome-grpo-skip-lr-5em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 22:31:50 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.
- Selected proposal traces retained for replay: `12`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 22:57:39 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0p03-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `5`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.03`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0p03-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 23:01:35 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 24]`.
- Selected proposal traces retained for replay: `17`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 23:26:31 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0p01-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `8`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.01`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0p01-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 23:48:54 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0p01-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 27]`.
- Selected proposal traces retained for replay: `18`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.01`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-1em6-akl-0p01-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-24 23:56:30 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0p03-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `16`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.03`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-3em6-akl-0p03-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-25 00:33:52 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5em6-akl-0p03-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `12`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.03`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5em6-akl-0p03-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-25 00:50:15 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_dr_grpo_default_cell_anchorfix_20260624_120907/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-dr-grpo-lr-1em6-akl-0p01-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16]`.
- Selected proposal traces retained for replay: `13`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.01`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_dr_grpo_default_cell_anchorfix_20260624_120907/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-dr-grpo-lr-1em6-akl-0p01-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-25 00:55:30 UTC

- Implemented/running adaptive candidate-training loop.
- Task: `run_length`; max attempts: `25`; max selected candidates: `unlimited`; attempts used: `25`; candidates per attempt: `8`.
- Output directory: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5em6-akl-0-eval-transformers`.
- Proposal output schema: `action_observation`.
- Proposal prompt action history: `False`; max items: `5`.
- Final source sizes tracked by driver: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 27]`.
- Selected proposal traces retained for replay: `14`.
- Post-task proposal rehearsal: `False`; repeat/max examples: `64`/`256`.
- Outcome trace target mode: `numeric`; retained outcome traces: `1600`.
- Proposal GRPO updates: `24`; steps/update: `1`; reward mode: `outcome`; zero-variance mode: `skip`.
- Proposal GRPO KL: old-policy coef `0.01`; anchor `adaptive_init` coef `0.0`.
- Proposal GRPO anchor checkpoint: `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855/run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5em6-akl-0-eval-transformers/round_00/seed_training/model`.
- Proposal GRPO action dedup: `True`.
- Proposal GRPO novelty beta: `0.05`.
- Source admission target-accuracy threshold: `0.8`.
- Proposal update loss: `merged_agent`; observation/format weights: `0.2`/`0.02`.
- Keep final model checkpoint: `False`.
- Keep all proposal-GRPO checkpoints: `False`.

### Implementation Log: 2026-06-25 Anchor-KL Fix Sweep Outcome

- Updated analysis notebook: `notebooks/adaptive_recent_grpo_sweep_results.ipynb`.
- Fixed-run roots analyzed:
  - `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855`
  - `artifacts/runs/adaptive_dr_grpo_default_cell_anchorfix_20260624_120907`
- Scheduler outcome:
  - All addition jobs reached the `3h` walltime, but wrote partial results with roughly `12`-`17` attempts.
  - Most run-length jobs completed the full `25` attempts. The `run_length grpo lr=5e-6 akl=0.01` cell reached `23` attempts before walltime.
  - Dr.GRPO run_length completed; Dr.GRPO addition timed out after `12` attempts.
- Anchor-KL fix status:
  - The previous `HFValidationError` / deleted relative seed-checkpoint failure did not recur.
  - All `anchor_kl > 0` proposal-GRPO updates reported `anchor_kl_enabled=True` with no skip reason.
  - The anchor checkpoint is now protected during the adaptive loop and pruned only at final cleanup when checkpoint retention is disabled.
- Quick outcome summary from logged attempt/candidate records:
  - Addition best partial cell by selected local reward: `grpo lr=5e-6 akl=0.01`, `15` attempts, `8` selected candidates, selected reward sum `+0.1445`, selected frontier-delta sum `+0.1755`, `72` evaluated candidates, `17` unique actions.
  - Run-length best completed cell by selected local reward: `grpo lr=1e-6 akl=0.01`, `25` attempts, `18` selected candidates, selected reward sum `+0.3104`, selected frontier-delta sum `+0.3205`, `73` evaluated candidates, `20` unique actions.
  - Dr.GRPO did not clearly improve this batch: addition selected `4` candidates before timeout with reward sum `+0.0430`; run_length selected `13` candidates with reward sum `+0.1875`.
- Caveats:
  - The result files reliably record selected candidate rewards/actions, validity, diversity, and proposal-GRPO metrics, but they do not preserve a clean post-attempt global held-out average accuracy trajectory after final checkpoint cleanup.
  - Because final model checkpoints are deleted by default, exact final heatmaps/final-delta comparisons require logging post-update eval summaries before cleanup in future runs.
  - Repetition is still visible in the best addition partial run, especially repeated `13+7->20 reject_boundary_carry` after that action becomes selected.

### Implementation Log: 2026-06-25 Adaptive GRPO Simplification

- Checkpointed the pre-cleanup adaptive sweep code in git: `43ceb83 Checkpoint adaptive GRPO sweep code`.
- Adopted the current default config cell:
  - `PROPOSAL_PROMPT_ACTION_HISTORY=1`
  - `PROPOSAL_GRPO_NOVELTY_BONUS_BETA=0.05`
  - `PROPOSAL_GRPO_LEARNING_RATE=1e-6`
  - old-policy sampled-token KL remains controlled by `PROPOSAL_GRPO_KL_COEF=0.01`.
- Removed Dr.GRPO objective plumbing from the adaptive proposal update path, CLI parser, submitter matrix, manifest builder, and launcher tests.
- Removed anchor-KL plumbing from the adaptive proposal update path, controller payloads, final cleanup, CLI parser, submitter matrix, manifest builder, and launchers.
- Simplified launcher job keys/output dirs to omit `obj-*` and `akl-*` suffixes; LR remains the only proposal-GRPO sweep dimension in the current submitter.
- Updated tests to cover the simplified launcher/manifest contract and the current GRPO microbatch loss.
- Verification:
  - `python -m py_compile self/adaptive/args.py self/adaptive/proposal.py self/adaptive/attempts.py self/adaptive/driver.py self/adaptive/run.py self/adaptive/controller.py self/launcher_manifests.py`
  - `bash -n launchers/self/submit_adaptive_candidate_training_ailab.sh launchers/self/run_adaptive_candidate_training_ailab.sbatch`
  - `~/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_args_normalization.py tests/test_launcher_manifests.py tests/test_adaptive_candidate_launcher.py -q`
  - `~/.conda/envs/torch-env/bin/python -m pytest tests/test_adaptive_candidate_training.py -k 'parser_defaults_enable_numeric_outcome_and_config_grpo or proposal_grpo_reward_mapping_and_advantages or proposal_policy_microbatches_match_full_batch_gradient' -q`
