# Self-Improvement Code Layout

Canonical implementation modules now live under package directories. The old
top-level adaptive module names remain as compatibility wrappers for launchers,
tests, notebooks, and older artifacts.

## Current Core

- `self/core/driver.py`: thin adaptive CLI/worker entry point and
  monkeypatch-compatible wrapper surface for the extracted adaptive runtime.
- `self/core/driver_wiring.py`: dependency-factory construction for candidate
  dispatch, worker entry points, proposal-GRPO dispatch, and full adaptive run
  orchestration. It reads through the live driver module so old monkeypatches
  still affect execution.
- `self/core/driver_default_bindings.py`: concrete default runtime bindings
  exposed lazily by the driver for old imports and patch points such as
  `train_and_score_candidate`, `subprocess`, `_load_json`, and
  `_prepare_candidate_worker_specs`.
- `self/core/driver_compat_exports.py`: lazy compatibility export surface for
  helpers and containers that older notebooks/tests imported through
  `self.adaptive_candidate_training`.
- `self/core/driver_compat_manifest.py`: lightweight manifest of the driver
  compatibility names, kept separate so the driver can expose `dir()`/`__all__`
  without importing the full legacy surface eagerly.
- `self/core/args.py`: CLI parser construction plus argument validation and
  task-specific default normalization.
- `self/core/attempt_loop_runtime.py`: selected-round adaptive attempt loop
  orchestration across prompt construction, dry-run handling, round-model
  dispatch, candidate scoring, trace writing, and outcome application.
- `self/core/attempt_prompt_runtime.py`: attempt-level proposal prompt
  construction shared by the driver loop and controller-worker round phase.
- `self/core/attempt_outcome_runtime.py`: selected/no-selection adaptive
  attempt outcome handling, including outcome traces, selected proposal traces,
  source-pool updates, proposal-GRPO follow-up updates, and attempt summaries.
- `self/core/candidate_scoring.py`: candidate checkpoint training,
  post-task proposal rehearsal, evaluation, and reward/metric construction.
- `self/core/candidate_selection.py`: candidate eligibility filtering and
  selection tie-break policy.
- `self/core/candidate_data.py`: candidate composed-data construction,
  component-prediction collection, pseudo-label attachment, and per-candidate
  data artifacts.
- `self/core/candidate_dispatch_runtime.py`: candidate training mode
  selection plus serial, local-parallel, and Slurm-array execution helpers.
- `self/core/candidate_dispatch_entrypoints.py`: compatibility-aware wiring
  between driver-level monkeypatchable names and candidate dispatch runtimes.
- `self/core/checkpoints.py`: checkpoint retention policy and cleanup helpers
  for unselected candidates and superseded model directories.
- `self/core/candidate_execution.py`: compatibility reexport surface for old
  candidate execution imports.
- `self/core/candidate_metric_collection.py`: candidate worker metric loading,
  failure-metric construction, missing-metric recovery, and gather-failure
  manifests.
- `self/core/candidate_worker_payloads.py`: candidate work-item payload
  serialization/deserialization for controller handoff artifacts and
  candidate-worker spec blocks.
- `self/core/candidate_worker_runtime.py`: candidate-worker spec entry point
  runtime, including spec loading, trace/pseudo-example reconstruction, metric
  generation, packed local-worker execution, and worker-failure artifact
  writing.
- `self/core/candidate_workers.py`: compatibility wrapper/reexport surface for
  candidate worker specs, local candidate-worker dispatch, and Slurm-array
  candidate-worker dispatch.
- `self/core/candidate_worker_specs.py`: candidate worker shared input/spec
  generation, pack-spec generation, candidate manifest writing, and candidate
  metrics/failure path helpers.
- `self/core/candidate_local_workers.py`: local and packed-local
  candidate-worker process scheduling, local dispatch manifests, timeout
  handling, and local worker failure artifact writing.
- `self/core/candidate_slurm_workers.py`: Slurm-array candidate-worker
  submission, dispatch manifests, polling, timeout cancellation, and
  post-array metric collection handoff.
- `self/core/composition.py`: exact-pair target construction, composition
  guards, and self-labeled pseudolabel generation for config/program proposals.
- `self/core/controller_phases.py`: controller worker phase constants and
  payload containers.
- `self/core/controller_phase_runtime.py`: in-process seed and round-model
  controller phases, including seed training/evaluation, proposal generation,
  proposal validation, and candidate pseudo-label preparation.
- `self/core/controller_worker_runtime.py`: controller-worker spec entry point
  runtime for seed, round-model, and proposal-GRPO worker phases.
- `self/core/controller_workers.py`: generic controller-worker spec IO,
  Slurm dispatch, polling, and worker output/failure recording.
- `self/core/data_io.py`: directory creation, generic JSON writing, example
  JSONL IO, checkpoint cleanup, RNG-state serialization, JSON sanitization,
  and summary-record IO.
- `self/core/dry_run_runtime.py`: dry-run adaptive attempt handling for
  fixture proposals, proposal validation, candidate data construction, and
  dry-run attempt summaries.
- `self/core/entrypoint.py`: CLI entrypoint dispatch for normal adaptive
  runs, controller workers, candidate workers, and packed candidate workers.
  The driver injects concrete handlers to preserve old module-level patching.
- `self/core/evaluation.py`: prediction parsing, decode-length resolution,
  generation encodings, accuracy evaluation, prediction maps, and debug
  prediction sample writing.
- `self/core/experience_traces.py`: proposal/outcome trace construction,
  serialization, and replay sampling.
- `self/core/frontier.py`: frontier selection and proposal-quality summaries.
- `self/core/model_io.py`: tokenizer construction, special-token syncing,
  added-token embedding initialization, and model loading/instantiation.
- `self/core/models.py`: shared proposal, candidate work-item, candidate
  metrics, and JSON conversion containers.
- `self/core/nonadaptive_bootstrap.py`: non-adaptive resume checkpoint
  selection, model/tokenizer instantiation, training config/collator setup,
  decode-budget derivation, and resumed pseudo-seed loading.
- `self/core/nonadaptive_datasets.py`: non-adaptive initial/composed/eval
  dataset generation and persisted dataset loading.
- `self/core/nonadaptive_evaluation.py`: non-adaptive base/composed round
  evaluation, composed-slice aggregation, and guarded-slice debug sample
  writing.
- `self/core/nonadaptive_lifecycle.py`: non-adaptive post-round lifecycle
  handling, including stop/final-round decisions, trainer release, CUDA cache
  clearing, reset-each-round model release, and reset checkpoint/model reloads.
- `self/core/nonadaptive_loop.py`: non-adaptive iterative
  self-improvement loop runtime, including round-by-round orchestration,
  training/evaluation dispatch, pseudo-label handoff, and summary writing.
- `self/core/nonadaptive_pseudo.py`: non-adaptive dynamic composed-pool
  refresh, next-round pseudo-label generation, pseudo-generation diagnostics,
  and per-round pseudo/composed artifact snapshots.
- `self/core/nonadaptive_results.py`: non-adaptive round summary creation,
  console summary dispatch, `metrics.json` writing, run-level summary-record
  updates, and compatibility-injected summary/payload bindings.
- `self/core/nonadaptive_schedule.py`: non-adaptive size-schedule arithmetic
  for legacy contiguous expansion and explicit frontier-based expansion.
- `self/core/nonadaptive_state.py`: non-adaptive output/data artifact path
  construction, metadata/result loading, loaded-metadata compatibility
  validation, metadata persistence, and config-arg snapshot writing.
- `self/core/nonadaptive_training.py`: non-adaptive per-round recipe phase
  resolution, training argument/trainer construction, round training, and
  model/tokenizer save handling.
- `self/core/program_sandbox.py`: sandboxed composition-program validation.
- `self/core/proposals.py`: proposal schemas, prompts, parsing, validation,
  and trace row helpers.
- `self/core/proposal_prompts.py`: task-specific proposal target formats,
  default executable source-pair selection, sandbox validation-case selection,
  and executable program/policy/meta prompt rendering.
- `self/core/proposal_config_validation.py`: config proposal row output
  normalization, schema/action-prediction validation, duplicate marking, and
  repeat-target annotation.
- `self/core/proposal_runtime.py`: runtime proposal loading/generation,
  executable program/policy validation, and program-repair dispatch.
- `self/core/proposal_grpo_dispatch.py`: local-vs-Slurm proposal-GRPO update
  dispatch and proposal-GRPO worker input artifact writing.
- `self/core/proposal_grpo.py`: proposal-GRPO rewards, trace construction,
  advantages, and lightweight policy update.
- `self/core/recipes.py`: shared algorithmic self-improvement recipe presets,
  recipe tokenizer/model construction, recipe training arguments, and
  recipe-aware Trainer variants.
- `self/core/run_finalization.py`: final adaptive result writing, summary
  artifact construction, and plan-log finalization.
- `self/core/run_initialization_runtime.py`: adaptive output/data directory
  setup, initial split/eval artifact writing, checkpoint manager construction,
  and source/exclusion pool initialization.
- `self/core/run_orchestration.py`: high-level adaptive run sequence across
  argument normalization, seed initialization, selected-round attempt loop, and
  finalization. The driver injects concrete dependencies to preserve
  monkeypatch-compatible old entry points.
- `self/core/run_setup.py`: adaptive run setup helpers, including initial
  split/eval construction, trace JSONL loading, source-size extraction, and
  plan-log appends.
- `self/core/round_model_dispatch_runtime.py`: per-attempt round-model
  dispatch for local execution vs Slurm controller-worker execution.
- `self/core/seed_dispatch_runtime.py`: seed/dry-run model initialization,
  seed Slurm dispatch parsing, and initial adaptive summary construction.
- `self/core/slurm.py`: small Slurm submission/polling helpers.
- `self/core/summaries.py`: non-adaptive round summary containers, metrics
  payload conversion, accuracy formatting, and console summary printing.
- `self/core/task_protocols.py`: shared task/example protocols and type
  aliases used by task-agnostic self-improvement code.
- `self/core/task_registry.py`: adaptive task-name lookup for concrete task
  adapters.
- `self/core/tokenizers.py`: fixed-character and arithmetic recipe
  tokenizers used by scratch symbolic models.
- `self/core/training.py`: training configuration, prompt/target dataset,
  causal-LM collator, exact-size batch sampler, training-argument creation,
  and Trainer construction.
- `self/core/worker_io.py`: JSON/path helpers shared by controller and
  candidate workers. The adaptive driver keeps compatibility aliases for the
  old private helper names.
- `self/core/worker_entrypoints.py`: compatibility-aware wiring for candidate
  and controller worker spec entry points. The driver injects current
  module-level functions so old monkeypatch paths keep affecting worker
  execution.
- `self/self_improvement_core.py`: compatibility import surface for legacy
  scripts, launchers, notebooks, and tests. Model IO, example/data IO,
  evaluation/generation helpers, training construction, summary helpers, task
  protocols, and the non-adaptive loop have moved into focused `self/core/`
  modules.

## Current Tasks

- `self/tasks/addition.py`: addition task orchestration, compose/direct
  pseudolabel dispatch, metadata validation, and `AdditionTask`.
- `self/tasks/addition_data.py`: addition example/data helper surface,
  canonical addition pipeline reexports, initial/composed/eval dataset
  preparation, boundary-carry slicing, and numeric-target corruption helper.
- `self/tasks/addition_pseudolabels.py`: addition direct/compose/corrupt
  pseudolabel derivation and pseudolabel diagnostics.
- `self/tasks/bit_common.py`: shared bit-task constants, parsers,
  composition-size helpers, guarded pseudolabel refill, and direct
  pseudolabel utilities.
- `self/tasks/multiplication.py`: multiplication task orchestration,
  oracle-aggregation pseudolabel derivation, metadata validation, and
  `MultiplicationTask`.
- `self/tasks/multiplication_data.py`: multiplication example container, key
  encoding, seed/long dataset construction, blocked component payloads, slice
  naming, and override cloning.
- `self/tasks/rectangular_multiplication.py`: rectangular multiplication
  example formats, partition parsing, sampled partition datasets, supported
  component construction, and composition helpers for rectangular experiments.
- `self/tasks/run_length.py`: run-length task orchestration, metadata
  validation, and `RunLengthTask`.
- `self/tasks/run_length_data.py`: run-length example container, key
  encoding, example generation, composition, bucketing, dataset construction,
  and override cloning.
- `self/tasks/run_length_logic.py`: pure run-length statistics, state
  formatting, state merging, leftmost max-run tie-breaking, and target
  formatting.
- `self/tasks/run_length_pseudolabels.py`: run-length direct, guarded-pair,
  run-state, and default tuple pseudolabel derivation.
- `self/self_improvement_tasks.py`: explicit compatibility import surface for
  the old task-module path, backed by canonical `self/tasks/*` and
  `self/core/*` modules.

## Current Experiments

- `self/experiments/adaptive_self_improvement.py`: dry-run/pilot proposal
  controller used for fixture-based adaptive proposal analysis.
- `self/experiments/figure2_condition_sweep.py`: Figure 2 MIG condition
  sweep submission, stage-2 selection, and final heatmap refresh.
- `self/experiments/figure2_paper_retune.py`: paper-facing Figure 2 schedule
  retuning and figure refresh.
- `self/experiments/figure3_seed_quality_sweep.py`: Figure 3 seed-quality and
  sample-size sweep submission/collection.
- `self/experiments/figure3_real_seed_data_ablation.py`: Figure 3 real-task
  seed/data ablation submission/collection.
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

- `self/analysis/artifacts.py`: stable JSON/JSONL loaders for notebooks,
  adaptive-run discovery, attempt/proposal/candidate record flattening, trace
  loading, proposal-GRPO metric records, and non-adaptive per-size accuracy
  records.
- `self/analysis/training_curve_notebook_utils.py`: notebook helpers for
  training-curve parsing, per-size heatmaps, and paper-style curve plots.
- `self/analysis/seed_fit_curve_notebook_utils.py`: notebook helpers for
  seed-fit training curves and budget sweeps.
- `self/analysis/plot_appendix_baseline_heatmaps.py`: appendix baseline
  heatmap export CLI.
- `self/analysis/plot_self_improvement_figure.py`: comparison-curve plotting
  CLI for one or more self-improvement result files.
- `self/analysis/summarize_seed_fit_grid.py`: seed-fit grid summary and
  threshold-based config selection CLI.

## Current Legacy

- `self/legacy/addition_self_improvement.py`: historical addition
  self-improvement CLI and resumable round implementation. Active launchers
  still call `python -m self.self_improvement`, which is kept as a
  compatibility wrapper.
- `self/legacy/bit_task_self_improvement.py`: shared parser/normalization
  helper for historical bit-string self-improvement CLIs.
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
  command printing to stdout or stderr. It also owns small shell utilities such
  as boolean parsing for launcher flags.
- `launchers/self/lib/adaptive_common.sh`: shared setup for adaptive AILAB
  launchers. It sources the generic launcher helper and adds HF cache/offline
  environment setup, adaptive config-file sourcing, and worker context logging.
- `launchers/self/run_adaptive_candidate_training_ailab.sbatch`,
  `launchers/self/run_adaptive_candidate_worker_ailab.sbatch`,
  `launchers/self/run_adaptive_controller_worker_ailab.sbatch`,
  `launchers/self/run_adaptive_condition_ailab.sbatch`, and
  `launchers/self/run_adaptive_self_improvement_ailab.sbatch` source the
  shared adaptive launcher helper.
- `launchers/self/submit_adaptive_candidate_training_ailab.sh`,
  `launchers/self/submit_adaptive_condition_pilots_ailab.sh`, and
  `launchers/self/submit_main_experiments_ailab.sh` source the shared helper
  before building Slurm command matrices and manifests. The condition-pilot
  and main-experiment submitters use the shared Slurm resource/default helpers
  rather than hand-assembling the common resource arguments.
- `launchers/self/config/adaptive_candidate_*.env`: default-only config files
  for adaptive candidate-training runs. Source them through
  `ADAPTIVE_CONFIG_FILE` or colon-separated `ADAPTIVE_CONFIG_FILES`; explicit
  caller environment variables still take precedence.
- `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` and
  `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` now use the
  generic launcher helper for repo-root setup and common MIG Slurm resource
  argument construction.
- `launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch`,
  `launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh`, and
  `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, and dry-run command
  printing.
- `launchers/self/run_addition_fullpack_filtered.sbatch` and
  `launchers/self/submit_addition_fullpack_filtered_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, and dry-run command
  printing.
- `launchers/self/run_addition_their_recipe_diagnostic.sh` uses the generic
  helper for repo-root/Python setup, boolean parsing, and dry-run command
  printing before delegating to the addition recipe diagnostic CLI.
- `launchers/self/run_addition_recipe_focused.sh`,
  `launchers/self/run_addition_recipe_fullpack.sh`, and
  `launchers/self/run_addition_recipe_recovery.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, and dry-run command
  printing across the recipe recovery workflow.
- `launchers/self/run_addition_tiny_seed_mig.sbatch` and
  `launchers/self/run_addition_seed_shared.sbatch` use the generic helper for
  repo-root/Python setup, boolean parsing, and staged dry-run command printing
  across the addition seed workflows.
- `launchers/self/submit_run_length_fixed_binary_mig.sh` uses the generic
  helper for repo-root/Python setup and per-job explicit Slurm resource
  argument construction across GPU and CPU jobs.
- `launchers/self/submit_guarded_plain_output_bit_diagnostic_mig.sh` and
  `launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch` use the
  generic helper for repo-root/Python setup and dry-run command printing.
- `launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch`,
  `launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch`,
  `launchers/self/submit_multiplication_rectangular_square_probe_mig.sh`, and
  `launchers/self/submit_multiplication_rectangular_square_seed_resweep_mig.sh`
  use the generic helper for repo-root/Python setup, boolean parsing, and
  dry-run command printing.
- `launchers/self/run_multiplication_rectangular_seed_mig.sbatch`,
  `launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch`,
  `launchers/self/submit_multiplication_rectangular_seed_sweep_mig.sh`, and
  `launchers/self/submit_multiplication_rectangular_fullpack_mig.sh` use the
  generic helper for repo-root/Python setup, boolean parsing, and stdout/stderr
  command-printing helpers while preserving their existing dry-run streams.
- `launchers/self/submit_multiplication_rectangular_tune_mig.sh` uses the
  generic helper for repo-root/Python setup, boolean parsing, and stdout
  command printing before delegating to the rectangular tune CLI.
- `launchers/self/run_seed_fit_experiment.sbatch` and
  `launchers/self/submit_seed_fit_grid.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, and seed-fit dry-run
  verification.
- `launchers/self/run_task_self_improvement.sbatch` and
  `launchers/self/submit_budget_grid_self_improvement.sh` use the generic
  helper for repo-root/Python setup where needed, boolean parsing, and
  self-improvement budget-grid dry-run verification.
- `launchers/self/submit_figure2_condition_sweep_mig.sh`,
  `launchers/self/submit_figure3_seed_quality_sweep_mig.sh`, and
  `launchers/self/submit_figure3_real_seed_data_ablation_mig.sh` use the
  generic helper for repo-root/Python setup, boolean parsing, and dry-run
  command printing before delegating to their experiment CLIs.
- `launchers/self/run_figure2_recipe_aggressive.sh`,
  `launchers/self/submit_figure2_recipe_aggressive.sh`, and
  `launchers/self/run_figure2_paper_retune.sh` use the generic helper for
  repo-root/Python setup, boolean parsing, and command printing across the
  Figure 2 recipe workflow.
- `launchers/self/run_composition_error_sweep_self_improvement.sh`,
  `launchers/self/run_local_workshop_batch.sh`,
  `launchers/self/run_refocused_self_improvement_local.sh`,
  `launchers/self/submit_run_length_alpha10_baseline_pack_mig.sh`,
  `launchers/self/run_self_improvement_mig_boundary_eval.sbatch`, and
  `launchers/self/run_self_improvement_qwen_no_growth.sbatch` now source the
  generic helper for repo-root/Python setup, boolean parsing, Slurm resource
  construction where applicable, and dry-run/print-only command verification.

## Runtime Notes

- Local candidate execution defaults to `--candidate-execution-mode
  local_parallel`, `--candidate-local-parallelism 4`, and
  `--candidate-local-pack-size 1`.
- Increase `--candidate-local-pack-size` to run multiple candidates
  sequentially inside one local worker subprocess. This preserves candidate
  semantics because each candidate still starts from the current checkpoint, but
  it reduces process launch overhead and repeated worker-spec/bootstrap work.
  Packed workers also reuse shared args/source/eval/trace/prompt inputs across
  candidates in the same pack; candidate pseudo examples, seeds, training, and
  metrics remain candidate-specific.
- Add `--candidate-local-cache-base-state` when using packed local workers to
  cache a CPU copy of the shared source checkpoint state after the first load.
  Later candidates instantiate fresh model objects from that cached state
  instead of rereading checkpoint weights from disk. This preserves candidate
  isolation but increases per-worker CPU memory use by roughly one checkpoint.

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

These top-level modules forward to the canonical locations and should not gain
new implementation code:

- `self/adaptive_candidate_training.py` -> `self/core/driver.py`
- `self/adaptive_candidate_workers.py` -> `self/core/candidate_workers.py`
- `self/adaptive_controller_phases.py` -> `self/core/controller_phases.py`
- `self/adaptive_experience_traces.py` -> `self/core/experience_traces.py`
- `self/adaptive_frontier.py` -> `self/core/frontier.py`
- `self/adaptive_proposal_grpo.py` -> `self/core/proposal_grpo.py`
- `self/adaptive_proposals.py` -> `self/core/proposals.py`
- `self/adaptive_worker_io.py` -> `self/core/worker_io.py`
- `self/adaptive_self_improvement.py` -> `self/experiments/adaptive_self_improvement.py`
- `self/addition_recipe_diagnostic.py` -> `self/diagnostics/addition_recipe_diagnostic.py`
- `self/analyze_symbolic_training_dynamics.py` -> `self/diagnostics/analyze_symbolic_training_dynamics.py`
- `self/check_self_improvement_overfit.py` -> `self/diagnostics/check_self_improvement_overfit.py`
- `self/evaluate_fixed_composition_slices.py` -> `self/diagnostics/evaluate_fixed_composition_slices.py`
- `self/figure2_condition_sweep.py` -> `self/experiments/figure2_condition_sweep.py`
- `self/figure2_paper_retune.py` -> `self/experiments/figure2_paper_retune.py`
- `self/figure3_real_seed_data_ablation.py` -> `self/experiments/figure3_real_seed_data_ablation.py`
- `self/figure3_seed_quality_sweep.py` -> `self/experiments/figure3_seed_quality_sweep.py`
- `self/multiplication_rectangular_tune.py` -> `self/experiments/multiplication_rectangular_tune.py`
- `self/multiplication_rectangular.py` -> `self/tasks/rectangular_multiplication.py`
- `self/paper_schedule_selection.py` -> `self/experiments/paper_schedule_selection.py`
- `self/plot_appendix_baseline_heatmaps.py` -> `self/analysis/plot_appendix_baseline_heatmaps.py`
- `self/plot_self_improvement_figure.py` -> `self/analysis/plot_self_improvement_figure.py`
- `self/program_sandbox.py` -> `self/core/program_sandbox.py`
- `self/self_improvement_recipe.py` -> `self/core/recipes.py`
- `self/task_tokenizer.py` -> `self/core/tokenizers.py`
- `self/rectangular_multiplication_compose_diagnostic.py` -> `self/diagnostics/rectangular_multiplication_compose_diagnostic.py`
- `self/rectangular_multiplication_recipe_seed_fit.py` -> `self/experiments/rectangular_multiplication_recipe_seed_fit.py`
- `self/rectangular_multiplication_seed_fit.py` -> `self/experiments/rectangular_multiplication_seed_fit.py`
- `self/rectangular_multiplication_self_improvement.py` -> `self/experiments/rectangular_multiplication_self_improvement.py`
- `self/run_length_balanced_eval.py` -> `self/diagnostics/run_length_balanced_eval.py`
- `self/seed_fit_curve_notebook_utils.py` -> `self/analysis/seed_fit_curve_notebook_utils.py`
- `self/seed_fit_experiment.py` -> `self/experiments/seed_fit_experiment.py`
- `self/self_improvement.py` -> `self/legacy/addition_self_improvement.py`
  with wrapper attribute writes forwarded to the legacy module for old
  monkeypatch-based tests.
- `self/self_improvement_multiplication_cot_pseudo_addition.py` -> `self/legacy/multiplication_cot_pseudo_addition.py`
- `self/bit_task_self_improvement.py` -> `self/legacy/bit_task_self_improvement.py`
- `self/multiplication_self_improvement.py` -> `self/legacy/multiplication_self_improvement.py`
- `self/run_length_self_improvement.py` -> `self/legacy/run_length_self_improvement.py`
- `self/slurm_utils.py` -> `self/core/slurm.py`
- `self/summarize_seed_fit_grid.py` -> `self/analysis/summarize_seed_fit_grid.py`
- `self/training_curve_notebook_utils.py` -> `self/analysis/training_curve_notebook_utils.py`

## Remaining Cleanup Queue

- Continue splitting `self/core/driver.py` into smaller modules for round-loop
  orchestration. The CLI/args, shared data model, composition/pseudolabel,
  generic controller-worker dispatch, controller-worker spec runtime,
  seed/round-model controller phases, candidate train/eval scoring, checkpoint
  cleanup, proposal runtime generation/validation, proposal prompt rendering,
  candidate data construction, dry-run attempt handling, selected/no-selection
  attempt outcome handling, proposal-GRPO dispatch, run setup/trace loading,
  final result/log finalization, output/data initialization, worker-spec
  JSON/key serialization, attempt-level proposal prompt construction, seed
  initialization/initial summary construction, round-model local/Slurm
  dispatch, selected-round attempt loop orchestration, candidate-worker
  runtime, candidate dispatch runtime/entrypoint wiring, candidate
  execution/aggregation, candidate selection, adaptive task-name lookup, driver
  compatibility exports, driver default bindings, and driver dependency-wiring
  pieces have already been extracted. `self/core/driver.py` is now a thin
  wrapper module rather than the owner of adaptive run logic.
- Task-family splitting is complete for addition, multiplication, and
  run-length. `self/self_improvement_tasks.py` is now an explicit
  compatibility facade over canonical `self/tasks/*` and `self/core/*`
  imports; keep it facade-only. Addition data and slice helpers live in
  `self/tasks/addition_data.py`, pseudolabel derivation lives in
  `self/tasks/addition_pseudolabels.py`, and `AdditionTask` remains the
  orchestration layer. The large run-length adapter has split pure
  state/target logic into `self/tasks/run_length_logic.py`, data/example
  construction into `self/tasks/run_length_data.py`, and round-target
  pseudolabel derivation into `self/tasks/run_length_pseudolabels.py`.
  Multiplication example/data construction now lives in
  `self/tasks/multiplication_data.py`, while `MultiplicationTask` remains the
  orchestration layer.
- `self/self_improvement_core.py` is now a compatibility facade. Continue
  migrating internal imports to canonical `self/core/*` modules when touching
  old top-level scripts, but preserve the facade for old imports and
  monkeypatch-based tests. Current `self/core`, `self/tasks`,
  `self/experiments`, `self/diagnostics`, and `self/analysis`
  implementation modules no longer import through this facade.
- `self/core/nonadaptive_loop.py` still owns the main non-adaptive training,
  dynamic refresh, pseudo-label, and summary loop. Base/composed round
  evaluation now lives in `self/core/nonadaptive_evaluation.py`; per-round
  training setup and execution lives in `self/core/nonadaptive_training.py`;
  dataset generation/loading lives in `self/core/nonadaptive_datasets.py`;
  resume/model/bootstrap setup lives in `self/core/nonadaptive_bootstrap.py`;
  preflight/default normalization and derived setup values live in
  `self/core/nonadaptive_setup.py`; output/data artifact paths plus metadata
  IO and loaded-metadata compatibility checks live in
  `self/core/nonadaptive_state.py`; and deterministic size/frontier
  arithmetic lives in `self/core/nonadaptive_schedule.py`. These helpers have
  direct unit coverage.
- Recipe helpers now live in `self/core/recipes.py`; `self/self_improvement_recipe.py`
  is a compatibility wrapper. Current `self/core`, `self/experiments`, and
  `self/diagnostics` modules import recipe helpers from the canonical core
  module, while legacy scripts and old tests can keep using the wrapper.
- Tokenizer helpers now live in `self/core/tokenizers.py`; `self/task_tokenizer.py`
  is a compatibility wrapper. Current core modules import tokenizer helpers
  from the canonical core module, while legacy scripts and old tests can keep
  using the wrapper path.
- Current implementation packages also avoid importing task symbols through
  `self/self_improvement_tasks.py`; `self/core`, `self/experiments`,
  `self/diagnostics`, and `self/analysis` import task classes, run-length
  parsers, and constants from their canonical `self/tasks/*` owners. The old
  task facade remains for top-level compatibility scripts and legacy imports.
- Continue moving historical baselines and paper-specific scripts into
  `self/legacy/`, `self/experiments`, `self/diagnostics/`, or `self/analysis/`
  gradually with wrapper modules where old launchers still need them. The first
  diagnostics, paper/figure, analysis-utility, seed-fit, and rectangular
  multiplication experiment batches have moved under canonical package
  directories, the rectangular multiplication shared helper now lives under
  `self/tasks/`, and the historical addition self-improvement CLI now lives
  in `self/legacy/addition_self_improvement.py`. The historical run-length,
  multiplication, and bit-task CLI helpers have moved under `self/legacy/`,
  the multiplication CoT
  pseudo-addition curriculum script has
  moved under `self/legacy/`, and the balanced run-length evaluation
  diagnostic has moved under `self/diagnostics/`.
- Continue consolidating Slurm launchers into fewer templates plus explicit
  matrix configs. The adaptive AILAB parent/candidate-worker/controller-worker
  scripts now share `launchers/self/lib/adaptive_common.sh`; the generic
  helper lives in `launchers/self/lib/self_common.sh`; the adaptive
  condition-pilot, main-experiment, exact-digit addition, and fixed-width
  more-data addition submitters use common Slurm resource/default helpers; the
  fixed-width mixed-prompt addition seed/fullpack runners and mixed submitter
  share generic repo-root/Python/boolean/command-printing helpers; the addition
  fullpack-filtered runner/submitter share generic
  repo-root/Python/boolean/command-printing helpers; the addition recipe
  diagnostic launcher and recipe focused/fullpack/recovery workflow share the
  same helper path; the run-length fixed-binary submitter uses the generic
  explicit-resource helper for its mixed GPU/CPU jobs; the guarded plain-output
  bit diagnostic runner and submitter share generic
  repo-root/Python/command-printing helpers; the rectangular square
  seed/diagnostic runners and probe/resweep submitters share generic
  repo-root/Python/boolean/command-printing helpers; the rectangular non-square
  seed/self-improvement runners and seed/fullpack submitters share generic
  repo-root/Python/boolean/stdout-or-stderr command-printing helpers; the
  rectangular tune submitter shares the same repo-root/Python/boolean setup;
  the addition tiny/shared seed runners share generic repo-root/Python/boolean
  and staged dry-run command-printing helpers;
  the seed-fit grid runner/submitter share generic repo-root/Python/boolean
  helpers and now have a dry-run verification path; the generic task
  self-improvement budget-grid runner/submitter share the same helper pattern;
  the small Figure 2/3 submitter wrappers share the helper path before
  delegating to their Python experiment CLIs; the Figure 2 recipe
  runner/submitter/retune scripts share generic repo-root/Python/boolean and
  command-printing helpers; the composition-error sweep, local/refocused
  workshop batchers, run-length alpha10 baseline submitter, and self-improvement
  MIG runners also share the generic helper path, leaving the current
  `launchers/self/*.{sh,sbatch}` inventory without duplicated repo-root/Python
  setup;
  and adaptive candidate-training defaults can be loaded from
  `launchers/self/config/adaptive_candidate_*.env`.
- Use `self/analysis/artifacts.py` for new notebooks and migrate older
  notebooks away from direct raw JSON path parsing when they are edited.
- Keep source/artifact hygiene conservative: source notebooks and report
  sources remain visible, while executed notebooks, heavyweight model/run
  bundles, local cache/editor state, and command/LaTeX logs are ignored.
- Optimize candidate training to avoid repeated full model reloads for each
  local candidate worker where semantics allow it. Current packed-local workers
  reduce subprocess and shared-input IO/parsing overhead by default. With
  `--candidate-local-cache-base-state`, packed workers also avoid repeated disk
  reads of the shared source checkpoint while still instantiating a fresh model
  object per candidate from an unmodified cached CPU state.
