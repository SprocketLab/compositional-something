# Della Handoff

This document is the working-context handoff for moving `compositional-something` to Della.

Last updated: `2026-04-18`

## Current Goal

The active project is the workshop paper and experiment stack for **compositional self-improvement**.

The current empirical story is:

- `run_length` is the strong bit-string positive case.
- `addition` is the hard anchor.
- For `addition`, the only variant that showed a real gain was `with_carry_filtered`.
- The next hypothesis is that the `135M` addition run was still **seed-limited**, so the next step is to increase seed data and reuse a shared seed checkpoint.

## Transfer Command

Full copy:

```bash
cd /n/fs/cogai/cs1095
ssh cs1095@della-gpu.princeton.edu 'mkdir -p /scratch/gpfs/BRENDEN/changho'
rsync -aH --info=progress2 --partial --append-verify \
  compositional-something \
  cs1095@della-gpu.princeton.edu:/scratch/gpfs/BRENDEN/changho/
```

Lighter copy without large artifacts:

```bash
cd /n/fs/cogai/cs1095
ssh cs1095@della-gpu.princeton.edu 'mkdir -p /scratch/gpfs/BRENDEN/changho'
rsync -aH --info=progress2 --partial --append-verify \
  --exclude artifacts \
  --exclude .pytest_cache \
  compositional-something \
  cs1095@della-gpu.princeton.edu:/scratch/gpfs/BRENDEN/changho/
```

Current repo size is about `37G`.

## Slurm State

As of this handoff:

- there are **no active or pending addition jobs** in the current cluster queue
- the queued Qwen addition job was canceled
- the queued seed-only addition job was also canceled before migration

So Della can be treated as a clean restart point.

## Important Local Code Changes

These are the most important experiment-side changes made locally and not yet turned into a clean top-level commit.

### 1. Addition composed-data control is now bucketed at construction time

Previously, `with_carry_filtered` built a broad composed pool and then dropped many examples during pseudo-label construction.

Now, the composed pool itself can be constrained by a boundary-carry policy:

- file: [core/addition_pipeline.py](/n/fs/cogai/cs1095/compositional-something/core/addition_pipeline.py:308)
- key additions:
  - `matches_boundary_carry_policy(...)`
  - `boundary_carry_policy` argument in `compose_to_length(...)`
  - `boundary_carry_policy` argument in `build_composed_datasets(...)`

Current behavior:

- `boundary_carry_policy="any"`: no restriction
- `boundary_carry_policy="no_boundary_carry"`: only keep composed examples whose component boundary does not induce carry
- `boundary_carry_policy="boundary_carry"`: only keep composed examples with boundary carry

### 2. `with_carry_filtered` now requests the boundary-free composed bucket

- file: [self/self_improvement_tasks.py](/n/fs/cogai/cs1095/compositional-something/self/self_improvement_tasks.py:1317)

The addition task now maps:

- `with_carry_filtered -> no_boundary_carry`
- everything else -> `any`

This means we now control composed-data size at dataset generation time instead of losing examples unpredictably later.

There is still a backward-compatible metadata fallback for old datasets that predate this explicit bucketing.

### 3. Regression test for the new addition bucket logic

- file: [tests/test_self_improvement_tasks.py](/n/fs/cogai/cs1095/compositional-something/tests/test_self_improvement_tasks.py:129)

This test verifies that the composed dataset builder can produce a fixed-size `no_boundary_carry` bucket.

Test status before handoff:

```bash
pytest -q tests/test_self_improvement_tasks.py
```

Result: `14 passed`

### 4. New seed-only reusable launcher

- file: [launchers/self/run_addition_seed_shared.sbatch](/n/fs/cogai/cs1095/compositional-something/launchers/self/run_addition_seed_shared.sbatch:1)

This launcher trains a **seed-only** addition model with:

- model: `HuggingFaceTB/SmolLM2-135M`
- digits: `3..7`
- seed train per digit: `50000`
- seed eval per digit: `100`
- rounds: `0`
- batch sizes: train `64`, eval `128`
- precision: `bf16`
- checkpoints kept

At the end it creates:

```text
<OUT_ROOT>/seed_model -> <OUT_ROOT>/round_00
```

So later experiments can initialize from the same trained seed model.

## Important Experiment Results

### Run-length

This is the strongest bit-string positive result and is suitable for the paper story.

Corrected trusted run root:

- [artifacts/runs/self_improvement_refocus_20260416_bitpatched2](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_refocus_20260416_bitpatched2)

Key finding:

- `run_length`: `compose` extends cleanly while direct/self-training controls are weaker.

Useful result files:

- [run_length/compose](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_refocus_20260416_bitpatched2/run_length/compose/self_improvement_results.json)
- [run_length/direct](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_refocus_20260416_bitpatched2/run_length/direct/self_improvement_results.json)

### Addition on `SmolLM2-135M`

Finished run roots:

- filtered:
  - [artifacts/runs/self_improvement_addition_filtered_135m_bs64_20260417](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_addition_filtered_135m_bs64_20260417)
- other baselines:
  - [artifacts/runs/self_improvement_addition_135m_bs64_20260417](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_addition_135m_bs64_20260417)

Main result:

- `with_carry_filtered` is the only addition variant that gives a meaningful gain.
- `short_only`, `direct`, `with_carry`, and `compose_corrupt` all stay weak.

Final round-8 comparison:

- `with_carry_filtered`: `eval_accuracy = 0.556`, `stitched_eval_accuracy = 0.420`
- `short_only`: `0.270`, stitched `0.0588`
- `direct`: `0.261`, stitched `0.0525`
- `with_carry`: `0.262`, stitched `0.0625`
- `compose_corrupt`: `0.257`, stitched `0.0550`

Key result file:

- [with_carry_filtered/self_improvement_results.json](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_addition_filtered_135m_bs64_20260417/addition/with_carry_filtered/self_improvement_results.json)

### Addition seed accuracy issue

The likely bottleneck is that the addition seed regime was still not strong enough.

Round-0 seed-bucket held-out accuracy for the `135M` addition runs:

- `3-digit`: `0.95`
- `4-digit`: `0.95`
- `5-digit`: `0.90`
- `6-digit`: `0.85`
- `7-digit`: `0.80`

So the seed model was already weakening by `6-7` digits.

Source:

- [round_00/metrics.json](/n/fs/cogai/cs1095/compositional-something/artifacts/runs/self_improvement_addition_filtered_135m_bs64_20260417/addition/with_carry_filtered/round_00/metrics.json)

Interpretation:

- filtered composition helped a lot relative to the baselines
- but the seed model was not strong enough to reproduce the stronger result remembered from earlier runs

## Suggested Next Experiments On Della

### Priority 1: shared seed-only training

Run the new seed-only launcher first:

```bash
sbatch launchers/self/run_addition_seed_shared.sbatch
```

This should produce a reusable seed checkpoint under:

```text
artifacts/runs/addition_seed_shared_135m_50kpd_<timestamp>/seed_model
```

### Priority 2: rerun `with_carry_filtered` from the stronger seed

The intended workflow is:

1. train the seed-only model with `50000` per digit
2. reuse that `round_00` model as the initialization for later addition self-improvement runs

This is not yet wrapped in a dedicated launcher. The current seed launcher creates the reusable checkpoint, but a follow-up launcher still needs to be added or invoked manually.

### Priority 3: if needed, try Qwen on Della

The earlier Qwen run here was canceled before execution. If Della has better availability, `Qwen3-0.6B` can be retried there.

## GPU / Memory Guidance

Practical estimate from current repo conventions:

- tiny custom bit-task model (`~14M`): fine for `MIG 10GB`
- `SmolLM2-135M`: probably workable on `MIG 10GB` with smaller batch sizes
- `SmolLM2-360M`: likely only workable on `MIG 10GB` in a very conservative setup
- `Qwen3-0.6B` full fine-tuning: do **not** assume `10GB` is enough

Relevant evidence:

- existing MIG launcher:
  - [launchers/self/run_self_improvement_mig_boundary_eval.sbatch](/n/fs/cogai/cs1095/compositional-something/launchers/self/run_self_improvement_mig_boundary_eval.sbatch:1)
  - uses `SmolLM2-360M` with batch sizes `1/2` and grad accumulation `8`

## Paper Status

The workshop paper lives in the submodule:

- [icmlw26_comp-self-improvement](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement)

Current paper repo HEAD:

- `f4dd9b5`

Current local paper changes not yet pushed inside the submodule:

- [appendix/experiment_details.tex](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/appendix/experiment_details.tex)
- [appendix/theory_proofs.tex](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/appendix/theory_proofs.tex)
- [ref.bib](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/ref.bib)
- [related_work.bib](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/related_work.bib)
- [sections/2_setup_and_method.tex](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/sections/2_setup_and_method.tex)
- [sections/3_theoretical_analysis.tex](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/sections/3_theoretical_analysis.tex)
- untracked:
  - [main_merged.tex](/n/fs/cogai/cs1095/compositional-something/icmlw26_comp-self-improvement/main_merged.tex)

The paper already has:

- theory section condensed in main body
- fuller details moved to appendix
- first-page main figure
- merged single-file export `main_merged.tex`

## Parent Repo Status

The top-level repo is still dirty and contains many local additions beyond the paper repo. Notable local changes include:

- [core/addition_pipeline.py](/n/fs/cogai/cs1095/compositional-something/core/addition_pipeline.py)
- [self/self_improvement_tasks.py](/n/fs/cogai/cs1095/compositional-something/self/self_improvement_tasks.py)
- [tests/](/n/fs/cogai/cs1095/compositional-something/tests)
- [launchers/self/run_addition_seed_shared.sbatch](/n/fs/cogai/cs1095/compositional-something/launchers/self/run_addition_seed_shared.sbatch)

There are also many untracked launcher / notebook / experiment files. So the move to Della should be treated as copying a **working directory**, not a clean git snapshot.

## Old Migration Notes

This repository had two major layout changes:

- `78b17ec` checkpoint before refactoring
- `e2662ba` split into `w2s/core`, `w2s/self`, `w2s/meta`, `w2s/launchers`, `w2s/legacy`
- current layout removed the `w2s/` namespace and moved those folders to repository root

### Path Mapping

- `w2s/core/` -> `core/`
- `w2s/self/` -> `self/`
- `w2s/meta/` -> `meta/`
- `w2s/launchers/` -> `launchers/`
- `w2s/legacy/` -> `legacy/`

### Module Invocation Mapping

- `python -m w2s.self.legacy.addition_self_improvement` -> `python -m self.legacy.addition_self_improvement`
- `python -m w2s.self.experiments.composition_error_sweep` -> `python -m self.experiments.composition_error_sweep`
- `python -m w2s.meta.train_meta_self_improvement_rope` -> `python -m meta.train_meta_self_improvement_rope`

### Artifact Roots

- `artifacts/logs/`
- `artifacts/models/`
- `artifacts/runs/self_improvement/`
- `artifacts/runs/meta_self_improvement/`
