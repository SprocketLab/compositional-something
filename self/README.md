# Self-Improvement Code Layout

Canonical implementation modules now live under package directories. Only
top-level module names that are still referenced by tracked launchers, tests,
notebooks, or current docs remain as compatibility wrappers.

The current workshop/main-track task surface is addition, run-length, and
multiplication. Older auxiliary classification task experiments are not part of
the supported public surface.

## Supported Pseudolabel Modes

The maintained workshop/main-track pipeline exposes only these pseudolabel
families:

- `none`: no generated labels are added.
- `direct`: the current model labels the target example directly.
- `compose`: the current model labels source components, then task-specific
  composition builds the target label.
- `compose_corrupt`: the `compose` control with injected label noise.

Any new label-family baseline should be added as an explicit experimental mode
with tests, launcher coverage, and paper-facing documentation before it becomes
part of the supported surface.

## Current Implementation Layout

The implementation surface is organized by subsystem instead of one large flat
`self/core` namespace:

- `self/adaptive/run/`: adaptive CLI driver, argument parsing, run setup,
  seed/round dispatch, high-level orchestration, and remaining driver
  compatibility bindings used by the top-level CLI wrapper.
- `self/adaptive/attempts/`: per-attempt prompt construction, dry-run handling,
  candidate-attempt execution, and selected/no-selection outcome handling.
- `self/adaptive/candidates/`: candidate dataset construction, pseudo-label
  attachment, training mix construction, checkpoint training/evaluation,
  scoring, selection, local/Slurm dispatch, and candidate-worker IO.
- `self/adaptive/proposals/`: config/program proposal schemas, prompting,
  generation, validation, trace writing, pilot processing, and proposal-GRPO
  updates.
- `self/adaptive/traces/`: proposal/outcome trace data models, replay sampling,
  and trace rendering.
- `self/adaptive/frontier/`: frontier candidate extraction, scoring, and public
  frontier selection helpers.
- `self/adaptive/controller/`: controller-worker phases, Slurm controller
  dispatch, and worker entrypoints.
- `self/adaptive/sandbox/`: sandboxed generated-program validation, repair,
  and property-test cases.
- `self/nonadaptive/`: nonadaptive self-improvement setup, lifecycle,
  scheduling, per-round training/evaluation, pseudo-label generation, results,
  and legacy facade exports used by `self.self_improvement_core`.
- `self/core/`: shared utilities only: composition helpers, data/model IO,
  evaluation, shared dataclasses, recipe training/model helpers, Slurm helpers,
  tokenizers, training utilities, task protocols/registry, summaries, lazy export
  helpers, and module proxy helpers.

Top-level `self/*.py` files are now treated as CLI or legacy compatibility
wrappers. Tests should import implementation modules from `self/adaptive/`,
`self/nonadaptive/`, `self/tasks/`, or the reduced `self/core/` directly rather
than importing through CLI wrappers.

## Current Tasks

- `self/tasks/addition.py`: addition task orchestration, compose/direct
  pseudolabel dispatch, metadata validation, and `AdditionTask`.
- `self/tasks/addition_data.py`: addition example/data helper surface,
  canonical addition pipeline reexports, initial/composed/eval dataset
  preparation, boundary-carry slicing, and numeric-target corruption helper.
- `self/tasks/addition_pseudolabels.py`: addition direct/compose/corrupt
  pseudolabel derivation and pseudolabel diagnostics.
- `self/tasks/bit_composition.py`: bit-string composition-path constants,
  component-size selection, and reachable target-size helpers.
- `self/tasks/bit_parsing.py`: run-length and multiplication prediction
  parsing, run-length alphabet/target constants, and multiplication target
  formatting.
- `self/tasks/bit_common.py`: shared bit-task constants, target-format
  normalization helpers, unique bitstring sampling, and compatibility reexports
  for older bit-composition, bit-parsing, and bit-pseudolabel imports.
- `self/tasks/bit_pseudolabels.py`: shared bit-task direct pseudolabel
  construction, guarded/refill pseudolabel construction, guard-slice
  partitioning, retained-count diagnostics, and the run-length boundary guard.
- `self/tasks/multiplication.py`: multiplication task orchestration,
  metadata validation, and `MultiplicationTask`.
- `self/tasks/multiplication_data.py`: multiplication example container, key
  encoding, override cloning, and compatibility reexports for older data-helper
  imports.
- `self/tasks/multiplication_pseudolabels.py`: multiplication direct,
  compose, and corrupt blocked-component pseudolabel derivation plus
  diagnostics.
- `self/tasks/multiplication_sampling.py`: multiplication seed/long dataset
  construction, exact-digit sampling, blocked component payloads, and
  overlap/carry slice naming.
- `self/tasks/multiplication_splits.py`: multiplication initial/eval/composed
  split preparation and composed-eval overlap/carry slice partitioning.
- `self/tasks/rectangular_partitions.py`: rectangular multiplication partition
  aliases, edge partition defaults, partition parsing/labels, partition-grid
  construction, and partition bucket IDs.
- `self/tasks/rectangular_digits.py`: rectangular multiplication digit-order
  helpers plus reverse-CoT prompt, target, normalization, and final-answer
  parsing utilities for diagnostics.
- `self/tasks/rectangular_data.py`: rectangular multiplication example
  container, exact-digit sampling, sampled partition datasets, final-answer
  parsing, prediction normalization, and key construction.
- `self/tasks/rectangular_composition.py`: rectangular multiplication
  multiplier-digit component construction, supported-partition component
  construction, least-significant block splitting, and target composition.
- `self/tasks/rectangular_multiplication.py`: rectangular multiplication
  compatibility import surface plus the sampled-dataset wrapper that preserves
  old sampler monkeypatch behavior.
- `self/tasks/run_length.py`: run-length task orchestration, metadata
  validation, and `RunLengthTask`.
- `self/tasks/run_length_data.py`: run-length example container, key
  encoding, example generation, composition, bucketing, dataset construction,
  and override cloning.
- `self/tasks/run_length_logic.py`: pure run-length statistics, state
  formatting, state merging, leftmost max-run tie-breaking, and target
  formatting.
- `self/tasks/run_length_guarded_pseudolabels.py`: run-length guarded
  plain-output and symbol-pair pseudolabel derivation.
- `self/tasks/run_length_pseudolabels.py`: run-length pseudolabel dispatch
  plus direct, run-state, and default tuple pseudolabel derivation.
- `self/tasks/run_length_splits.py`: run-length initial/eval/composed split
  preparation and composed-eval guard-slice partitioning.
- `self/tasks/compat_exports.py`: explicit legacy export manifest for
  `self.self_improvement_tasks`, grouping old-path task exports by protocol,
  evaluation, addition, bit-common, bit-composition, bit-pseudolabel,
  multiplication, and run-length owner.
- `self/self_improvement_tasks.py`: explicit compatibility import surface for
  the old task-module path, backed by canonical `self/tasks/*` and
  `self/core/*` modules. Its public compatibility exports now come from
  `self/tasks/compat_exports.py` so old-path compatibility stays auditable
  without keeping the export manifest in the facade itself.

## Current Experiments

- `self/experiments/adaptive_self_improvement.py`: dry-run/pilot proposal
  controller used for fixture-based adaptive proposal analysis; validation,
  ranking, and trace-row construction live in `self/adaptive/proposals/proposal_pilot_runtime.py`.
- `self/experiments/composition_error_sweep.py`: addition self-improvement
  launcher wrapper for controlled boundary-carry composition-error retention.
- `self/experiments/figure2_condition_sweep.py`: Figure 2 MIG condition
  sweep submission, stage-2 selection, and final heatmap refresh.
- `self/experiments/figure2_paper_retune.py`: paper-facing Figure 2 schedule
  retuning and figure refresh.
- `self/experiments/figure3_common.py`: shared Figure 3 JSON/CSV artifact
  writing, Slurm submission, seed-metric parsing, seed-band selection, and
  final-summary helpers used by the seed-quality and real-seed ablation
  scripts.
- `self/experiments/figure3_cli.py`: shared Figure 3 CLI path defaults and
  common argparse option registration.
- `self/experiments/figure3_commands.py`: shared Figure 3 seed-fit and
  run-length self-improvement command builders.
- `self/experiments/figure3_seed_quality_sweep.py`: Figure 3 seed-quality and
  sample-size sweep submission/collection.
- `self/experiments/figure3_real_seed_data_ablation.py`: Figure 3 real-task
  seed/data ablation submission/collection.
- `self/experiments/run_length_alpha10_seed_beam.py`: alpha-10
  symbol-run run-length trainer-seed beam orchestration, branch copying,
  Slurm polling, candidate scoring, and best-branch symlink creation.
- `self/experiments/seed_fit_experiment.py`: seed-only fit experiment for
  addition, run-length, and multiplication task adapters.
- `self/experiments/rectangular_multiplication_seed_fit.py`: rectangular
  multiplication seed-fit experiment over digit partitions.
- `self/experiments/rectangular_multiplication_recipe_seed_fit.py`: sampled
  rectangular multiplication seed-fit experiment with recipe support.
- `self/experiments/rectangular_multiplication_self_improvement.py`:
  iterative rectangular multiplication self-improvement from an edge-only seed.
- `self/experiments/multiplication_rectangular_tune.py`: MIG submission and
  collection helper for rectangular multiplication tuning sweeps.
- `self/experiments/paper_schedule_selection.py`: paper-facing schedule
  scoring and selection helpers.

## Current Diagnostics

- `self/diagnostics/check_self_improvement_overfit.py`: small overfit sanity
  checks for base/composed supervision across self-improvement tasks.
- `self/diagnostics/analyze_symbolic_training_dynamics.py`: training-loss and
  exact-match curve diagnostics for symbolic task settings.
- `self/diagnostics/evaluate_fixed_composition_slices.py`: offline evaluation
  of saved fixed-binary checkpoints on boundary-event slices.
- `self/diagnostics/addition_recipe_diagnostic.py`: arithmetic recipe
  diagnostic runner for addition recovery/frontier checks.
- `self/diagnostics/rectangular_multiplication_compose_diagnostic.py`:
  rectangular multiplication frontier diagnostic with direct or composed
  pseudo labels.
- `self/diagnostics/run_length_balanced_eval.py`: balanced-answer
  run-length evaluation diagnostic for plain-output regimes.

## Current Analysis

- `self/analysis/artifact_io.py`: shared JSON/JSONL readers, round-summary
  result loading for list-shaped and `{"rounds": ...}` payloads, and artifact
  filename constants for analysis loaders.
- `self/analysis/adaptive_artifact_common.py`: shared adaptive artifact
  constants, run/attempt dataclasses, index parsing, row-context construction,
  selected-id handling, and proposal-field flattening helpers used by both
  run-level and candidate-level adaptive loaders.
- `self/analysis/adaptive_artifacts.py`: adaptive-run discovery, attempt
  loading, attempt-row flattening, proposal-row flattening, per-attempt
  proposal-validity summaries, and compatibility reexports for candidate and
  trace artifact helpers.
- `self/analysis/adaptive_candidate_artifacts.py`: adaptive candidate
  artifact loading and candidate/train-mix/per-size row flattening for
  per-candidate metrics, train-mix summaries, worker-failure files, and
  local-dispatch plan/cache records.
- `self/analysis/adaptive_trace_artifacts.py`: adaptive prompt records,
  selected-checkpoint per-size timelines with selected-target markers for
  heatmaps, proposal-GRPO metric records, and trace JSONL flattening for
  notebooks.
- `self/analysis/adaptive_manifest_artifacts.py`: adaptive
  `submission_manifest.json` discovery, loading, and job-row flattening so
  notebooks can recover submitted condition metadata without hard-coding raw
  manifest paths.
- `self/analysis/adaptive_summary_artifacts.py`: adaptive run-overview rows and
  cross-run proposal-validity time series for notebooks, built from the stable
  run/attempt/candidate artifact loaders instead of raw `attempt_*` path
  assumptions.
- `self/analysis/nonadaptive_artifacts.py`: non-adaptive result-path
  resolution, round loading, per-size accuracy row construction, and
  records-to-DataFrame conversion.
- `self/analysis/artifacts.py`: compatibility surface for notebook artifact
  loaders.
- `self/analysis/training_curve_results.py`: result-path resolution, raw
  round-summary loading, round summary frames, and per-size accuracy frames
  for training-curve notebooks.
- `self/analysis/training_curve_style.py`: shared baseline colors, budget/mode
  ordering, plot rcParams, and display labels for training-curve plots.
- `self/analysis/training_curve_logs.py`: Slurm training-log parsing and
  round-level metric loading for training-curve bundles.
- `self/analysis/training_curve_bundle.py`: submission-table loading, curve
  bundle assembly, bundle summaries, and bundle per-size accuracy frames.
- `self/analysis/training_curve_heatmaps.py`: per-size heatmap plotting,
  sparse annotation selection, paper-layout controls, and direct results-file
  heatmaps.
- `self/analysis/training_curve_plots.py`: task training curves, comparison
  curves, figure bundle export helpers, and compatibility reexports for older
  heatmap imports.
- `self/analysis/training_curve_notebook_utils.py`: notebook helpers for
  compatibility imports over the canonical training-curve analysis modules.
- `self/analysis/seed_fit_artifacts.py`: seed-fit result filename constants,
  run-directory detection, result-path resolution, discovery, and validated
  `seed_fit_results.json` loading shared by notebooks and summary scripts.
- `self/analysis/seed_fit_bundle.py`: seed-fit result loading, flattened
  training/validation log tables, compact task summaries, and threshold
  budget selection.
- `self/analysis/seed_fit_plots.py`: seed-fit loss curves and budget-sweep
  plotting helpers.
- `self/analysis/seed_fit_curve_notebook_utils.py`: compatibility imports over
  the canonical seed-fit analysis modules.
- `self/analysis/plot_appendix_baseline_heatmaps.py`: appendix baseline
  heatmap export CLI.
- `self/analysis/plot_self_improvement_figure.py`: comparison-curve plotting
  CLI for one or more self-improvement runs, routed through the stable
  non-adaptive artifact loader.
- `self/analysis/summarize_seed_fit_grid.py`: seed-fit grid summary and
  threshold-based config selection CLI.

## Current Legacy

- `self/legacy/addition_self_improvement.py`: historical addition
  self-improvement CLI and resumable round implementation. Active launchers
  still call `python -m self.self_improvement`, which is kept as a
  compatibility wrapper.
- `self/legacy/run_length_bit_cli.py`: parser/normalization helper for
  historical run-length bit-string self-improvement CLIs.
- `self/legacy/run_length_self_improvement.py`: historical run-length
  self-improvement CLI backed by the shared non-adaptive loop.
- `self/legacy/multiplication_self_improvement.py`: historical multiplication
  self-improvement CLI backed by the shared non-adaptive loop.
- `self/legacy/multiplication_cot_pseudo_addition.py`: historical
  multiplication curriculum self-improvement CLI whose composition step uses
  an addition model for pseudo-label construction.

## Current Launchers

- `launchers/self/lib/self_common.sh`: generic launcher helpers for repo-root
  resolution, Python resolution, Slurm resource defaults, Slurm resource
  argument construction for default and per-job explicit resource blocks, and
  wrapped-job or script-job submission/dry-run handling. It also owns command
  printing to stdout or stderr, shell-quoted env-wrapped and repo-root
  command construction, colon-separated config-file sourcing, shared
  Torch/CUDA probe printing, Hugging Face/local model-snapshot preflight with
  optional tokenizer-mode checks, stable model/seed symlink updates with
  dry-run reporting, and small shell utilities such as boolean parsing for
  launcher flags.
- `launchers/self/lib/adaptive_common.sh`: shared setup for adaptive AILAB
  launchers. It sources the generic launcher helper and adds HF cache/offline
  environment setup, adaptive-labeled config-file sourcing wrappers, worker
  context logging, and Torch/CUDA probe printing.
- `launchers/self/run_adaptive_candidate_training_ailab.sbatch`,
  `launchers/self/run_adaptive_candidate_worker_ailab.sbatch`,
  `launchers/self/run_adaptive_controller_worker_ailab.sbatch`,
  `launchers/self/run_adaptive_condition_ailab.sbatch`, and
  `launchers/self/run_adaptive_self_improvement_ailab.sbatch` source the
  shared adaptive launcher helper. The condition runner also routes its
  task/output/fixture banner through the shared context printer.
- `launchers/self/submit_adaptive_candidate_training_ailab.sh`,
  `launchers/self/submit_adaptive_condition_pilots_ailab.sh`, and
  `launchers/self/submit_main_experiments_ailab.sh` source the shared helper
  before building Slurm command matrices and manifests. The condition-pilot
  and main-experiment submitters use the shared Slurm resource/default helpers
  rather than hand-assembling the common resource arguments. The adaptive
  candidate and condition-pilot submitters use the shared sbatch-script
  submission helper, while the main-experiment submitter uses the shared
  wrapped-job submission helper.
- `self/launcher_manifests.py`: importable submission-manifest builders and
  CLI used by adaptive submitters, keeping `submission_manifest.json` schemas
  out of shell here-docs while preserving the old JSON shape.
- `launchers/self/config/adaptive_candidate_*.env`: default-only config files
  for adaptive candidate-training runs. Source them through
  `ADAPTIVE_CONFIG_FILE` or colon-separated `ADAPTIVE_CONFIG_FILES`; explicit
  caller environment variables still take precedence.
- `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` and
  `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` now use the
  generic launcher helper for repo-root setup and common MIG Slurm resource
  argument construction. Both submitters use the shared env-wrapped command
  builder and resource-backed wrapped-job submitter; the exact-digits submitter
  also routes its schedule banner through the shared context printer. The
  exact-digits baseline list lives in
  `launchers/self/config/addition_exact_digits_fixed_binary.env`, and the
  more-data submitter's default Stage 1 grid lives in
  `launchers/self/config/addition_fixedwidth_moredata.env`.
- `launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch`,
  `launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh`, and
  `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing. The mixed submitter also uses the shared sbatch
  script helper for seed submission and shared env-wrapped command construction
  plus wrapped-job submission for fixed-binary and original-composition
  fullpack branches, with test coverage for dry-run and fake-Slurm submission
  paths. Its default baseline lists live in
  `launchers/self/config/addition_fixedwidth_mixed.env`.
- `launchers/self/run_addition_fullpack_filtered.sbatch` and
  `launchers/self/submit_addition_fullpack_filtered_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing. The runner also uses the shared model-snapshot
  preflight with `TOKENIZER_MODE`, including the fixed-char tokenizer branch,
  and the shared dry-run-aware seed-model symlink updater. The submitter also
  uses the shared sbatch-script submission helper for
  per-baseline Slurm metadata and explicit environment export payloads. Its
  default baseline list lives in
  `launchers/self/config/addition_fullpack_filtered.env`.
- `launchers/self/run_addition_their_recipe_diagnostic.sh` uses the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing before delegating to the addition recipe diagnostic
  CLI.
- `launchers/self/run_addition_recipe_focused.sh`,
  `launchers/self/run_addition_recipe_fullpack.sh`, and
  `launchers/self/run_addition_recipe_recovery.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, and dry-run command
  printing across the recipe recovery workflow. The recovery launcher uses the
  shared stable seed-model symlink updater after selecting the recovered seed.
- `launchers/self/run_addition_tiny_seed_mig.sbatch` and
  `launchers/self/run_addition_seed_shared.sbatch` use the generic helper for
  repo-root/Python setup, boolean parsing, context printing, and staged
  dry-run command printing across the addition seed workflows, with final seed
  checkpoint symlink updates routed through the shared helper.
- `launchers/self/submit_run_length_fixed_binary_mig.sh` uses the generic
  helper for repo-root/Python setup and per-job explicit Slurm resource
  argument construction across GPU and CPU jobs. It now launches the alpha-10
  seed-beam controller through `self.experiments.run_length_alpha10_seed_beam`;
  `launchers/self/run_run_length_alpha10_seed_beam_mig.py` remains a thin
  compatibility wrapper for older direct invocations. Paper-default, alpha-10
  template, and beam defaults live in
  `launchers/self/config/run_length_fixed_binary.env`, with
  `RUN_LENGTH_FIXED_BINARY_CONFIG` available for partial overrides.
- `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` and
  `launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch` use the
  generic helper for repo-root/Python setup and dry-run command printing. The
  submitter uses the shared sbatch-script submission helper for per-task Slurm
  metadata and explicit environment export payloads.
- `launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch`,
  `launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch`,
  `launchers/self/submit_multiplication_rectangular_square_probe_mig.sh`, and
  `launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`
  use the generic helper for repo-root/Python setup, boolean parsing, and
  dry-run command printing. The square probe and seed-resweep submitters use
  the shared sbatch-script submission helper for their Slurm submissions.
- `launchers/self/run_multiplication_rectangular_seed_mig.sbatch`,
  `launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch`,
  `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh`, and
  `launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` use the
  generic helper for repo-root/Python setup, boolean parsing, and stdout/stderr
  command-printing helpers while preserving their existing dry-run streams.
  The seed-sweep and fullpack submitters also use the shared sbatch-script
  submission helper for per-job names, logs, and explicit export payloads; the
  seed-sweep winner model link is updated through the shared symlink helper.
  The seed-sweep matrix defaults live in
  `launchers/self/config/multiplication_rectangular_seed_sweep.env`, and the
  fullpack baseline/training defaults live in
  `launchers/self/config/multiplication_rectangular_fullpack.env`.
- `launchers/self/submit_multiplication_rectangular_tune_mig.sh` uses the
  generic helper for repo-root/Python setup, boolean parsing, and stdout
  command printing before delegating to the rectangular tune CLI.
- `launchers/self/run_seed_fit_experiment.sbatch` and
  `launchers/self/submit_seed_fit_grid.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, context printing, and
  seed-fit dry-run verification. The seed-fit runner uses the shared standard
  model-snapshot preflight.
- `launchers/self/run_task_self_improvement.sbatch` and
  `launchers/self/submit_budget_grid_self_improvement.sh` use the generic
  helper for repo-root/Python setup where needed, boolean parsing, context
  printing, shared standard model-snapshot preflight, and self-improvement
  budget-grid dry-run verification.
- `launchers/self/submit_figure2_condition_sweep_mig.sh`,
  `launchers/self/submit_figure3_seed_quality_sweep_mig.sh`, and
  `launchers/self/submit_figure3_real_seed_data_ablation_mig.sh` use the
  generic helper for repo-root/Python setup, boolean parsing, and dry-run
  command printing before delegating to their experiment CLIs.
- `launchers/self/run_figure2_recipe_aggressive.sh`,
  `launchers/self/submit_figure2_recipe_aggressive.sh`, and
  `launchers/self/run_figure2_paper_retune.sh` use the generic helper for
  repo-root/Python setup, boolean parsing, command printing, shared
  repo-command wrapping, and wrapped-job submission across the Figure 2 recipe
  workflow. The aggressive workflow's top-level stage/task/baseline defaults
  live in `launchers/self/config/figure2_recipe_aggressive.env`.
- `launchers/self/lib/figure2_recipe_common.sh` and
  `launchers/self/config/figure2_run_length.env` own Figure 2 recipe task
  metadata and run-length defaults that were previously embedded in
  `run_figure2_recipe_aggressive.sh`.
- `launchers/self/run_composition_error_sweep_self_improvement.sh`,
  `launchers/self/run_local_workshop_batch.sh`,
  `launchers/self/run_refocused_self_improvement_local.sh`,
  `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`,
  `launchers/self/run_self_improvement_mig_boundary_eval.sbatch`, and
  `launchers/self/run_self_improvement_qwen_no_growth.sbatch` now source the
  generic helper for repo-root/Python setup, boolean parsing, Slurm resource
  construction where applicable, shell-quoted repo-command wrapping,
  wrapped-job submission, structured `sbatch --export=ALL,...` construction,
  and dry-run/print-only command verification.
  Current `self_submit_sbatch_script(...)` callers build export payloads
  through `self_sbatch_export_all(...)` instead of hand-written comma strings.
  The boundary-eval and qwen-no-growth runners use the shared
  model-snapshot preflight. The
  alpha10 baseline-pack matrix lives in
  `launchers/self/config/run_length_alpha10_baseline_pack.env`.

## Runtime Notes

- Local candidate execution parser defaults are `--candidate-execution-mode
  local_parallel`, `--candidate-local-parallelism 4`, and
  `--candidate-local-pack-size 1`. The shared AILAB adaptive candidate config
  `launchers/self/config/adaptive_candidate_base.env` now opts main
  experiments into `CANDIDATE_LOCAL_PACK_SIZE=2` and
  `CANDIDATE_LOCAL_CACHE_BASE_STATE=1`.
- Increase `--candidate-local-pack-size` to run multiple candidates
  sequentially inside one local worker subprocess. This preserves candidate
  semantics because each candidate still starts from the current checkpoint, but
  it reduces process launch overhead and repeated worker-spec/bootstrap work.
  Packed workers also reuse shared args/source/eval/trace/prompt inputs across
  candidates in the same pack; candidate pseudo examples, seeds, training, and
  metrics remain candidate-specific. Packed workers now also pass a per-pack
  tokenizer bootstrap cache by default, without caching model weights unless
  explicitly requested. Worker summaries include
  `model_bootstrap_cache_details` with tokenizer/model-state hit and miss
  counters for debugging cache effectiveness. Cache containers and pure
  key/state-copy helpers live in `self/core/model_bootstrap_cache.py`;
  `model_io.py` keeps compatibility reexports.
- Add `--candidate-local-cache-base-state` when using packed local workers to
  extend the per-pack bootstrap cache with a CPU copy of the shared source
  checkpoint state after the first load. Later candidates instantiate fresh
  model objects from that cached state instead of rereading checkpoint weights
  from disk. This preserves candidate isolation but increases per-worker CPU
  memory use by roughly one checkpoint.

## Source And Artifacts

- Source notebooks should use the normal `.ipynb` suffix. Executed notebook
  snapshots with `.executed.ipynb` are treated as generated artifacts and are
  ignored by Git.
- Large run products such as `*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`,
  `*.bin`, `*.tar.gz`, and `*.zip` are ignored for new files. Existing tracked
  model-card assets stay tracked.
- Local editor/agent state, downloaded `meta/models/` caches, Slurm/LaTeX
  intermediate logs, and unpacked `self_improvement_*_addmodel/` bundles are
  ignored so normal `git status` remains focused on source changes.

## Compatibility Wrappers

Top-level wrappers are kept only for runnable commands, old notebooks, and
legacy import surfaces that still matter. They should stay thin and should not
own implementation logic.

- Adaptive wrappers:
  - `self/adaptive_candidate_training.py` -> `self/adaptive/run/driver.py`
  - `self/adaptive_frontier.py` -> `self/adaptive/frontier/frontier.py`
  - `self/adaptive_proposals.py` -> `self/adaptive/proposals/proposals.py`
  - `self/program_sandbox.py` -> `self/adaptive/sandbox/program_sandbox.py`
  - `self/adaptive_self_improvement.py` -> `self/experiments/adaptive_self_improvement.py`
- Nonadaptive/legacy wrappers:
  - `self/self_improvement_core.py` -> `self/nonadaptive/*` plus shared `self/core/*` utilities
  - `self/self_improvement.py` and `self/self_improvement_experiment.py` -> `self/legacy/addition_self_improvement.py`
  - `self/run_length_self_improvement.py` -> `self/legacy/run_length_self_improvement.py`
  - `self/multiplication_self_improvement.py` -> `self/legacy/multiplication_self_improvement.py`
- Task/experiment/analysis wrappers still forward to `self/tasks`,
  `self/experiments`, `self/diagnostics`, or `self/analysis`. Keep them
  facade-only and prefer canonical package imports in new tests/code.

## Remaining Cleanup Queue

- Continue reducing compatibility machinery inside `self/adaptive/run/` once old
  notebook/import needs are clearer. The current refactor moved it out of
  shared core but did not remove every lazy export or monkeypatch bridge.
- Consider a second, smaller consolidation pass inside each subsystem package,
  especially `self/adaptive/candidates`, `self/adaptive/run`, and
  `self/nonadaptive`, where many files are still narrow extraction modules.
- Keep CLI wrappers and implementation tests separate: tests should target
  canonical subsystem packages, while top-level wrappers should remain command
  surfaces and compatibility facades only.
- Update untracked/local notebooks manually if they import old internal paths
  such as `self.core.driver` or `self.core.proposals`; tracked source and tests
  have already moved to the new package owners.
