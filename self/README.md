# Self-Improvement Code Layout

Canonical implementation modules now live under package directories. The old
top-level adaptive module names remain as compatibility wrappers for launchers,
tests, notebooks, and older artifacts.

The current workshop/main-track task surface is addition, run-length, and
multiplication. Older auxiliary classification task experiments are not part of
the supported public surface.

## Current Core

- `self/core/driver.py`: thin adaptive CLI/worker entry point and
  monkeypatch-compatible wrapper surface for the extracted adaptive runtime.
- `self/core/driver_wiring.py`: compatibility aggregation for driver bridge
  functions plus CLI entry point wiring. It reads through the live driver
  module so old monkeypatches still affect execution.
- `self/core/driver_candidate_dispatch_wiring.py`: driver-binding bridge for
  serial, local-parallel, and Slurm-array candidate dispatch. It keeps
  candidate training dispatch monkeypatch-compatible while `driver_wiring.py`
  stays focused on top-level orchestration wiring.
- `self/core/driver_proposal_grpo_wiring.py`: driver-binding bridge for
  local-vs-Slurm proposal-GRPO update dispatch. It keeps proposal-GRPO update
  dispatch monkeypatch-compatible while `driver_wiring.py` stays focused on
  top-level orchestration wiring.
- `self/core/driver_run_wiring.py`: driver-binding bridge for full adaptive
  run orchestration. It builds `AdaptiveRunDeps` from live driver bindings so
  run-level monkeypatches still flow into `run_orchestration.py`.
- `self/core/driver_worker_wiring.py`: driver-binding bridge for candidate
  workers, packed candidate workers, and controller worker entry points. It
  keeps worker dispatch monkeypatch-compatible while `driver_wiring.py` stays
  focused on top-level orchestration wiring.
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
- `self/core/driver_public_api.py`: mechanical installer for public driver
  delegates that forward through `driver_wiring.py` with the live driver module
  as the binding surface, keeping `driver.py` small without breaking old
  monkeypatch points.
- `self/core/module_proxy.py`: shared compatibility helper for old module
  paths that should proxy canonical modules while forwarding monkeypatch-style
  attribute writes.
- `self/core/args.py`: adaptive CLI parser construction, public choice
  constants, and compatibility reexport for `normalize_args(...)`.
- `self/core/args_normalization.py`: adaptive argument validation,
  cross-field checks, task-specific defaults, and derived size/count aliases.
- `self/core/attempt_loop_runtime.py`: selected-round adaptive attempt loop
  orchestration across prompt construction, dry-run handling, round-model
  dispatch, candidate scoring, trace writing, and outcome application.
- `self/core/attempt_prompt_runtime.py`: attempt-level proposal prompt
  construction shared by the driver loop and controller-worker round phase.
- `self/core/attempt_outcome_runtime.py`: selected/no-selection adaptive
  attempt outcome handling, including outcome traces, selected proposal traces,
  source-pool updates, proposal-GRPO follow-up updates, and attempt summaries.
- `self/core/candidate_scoring.py`: candidate scoring orchestration across
  train-mix setup, checkpoint training, optional proposal rehearsal,
  evaluation, reward construction, metric artifact writing, and cleanup.
- `self/core/candidate_training_runtime.py`: candidate `TrainingConfig`
  construction, checkpoint fine-tuning, held-out evaluation, CUDA cache
  clearing, and post-task proposal rehearsal training/summary writing.
- `self/core/candidate_selection.py`: candidate eligibility filtering and
  selection tie-break policy.
- `self/core/candidate_data.py`: candidate composed-data construction,
  component-prediction collection, pseudo-label attachment, and per-candidate
  data artifacts.
- `self/core/candidate_training_mix.py`: candidate task/pseudo/replay trace
  training mix construction plus train-mix JSONL/summary artifact writing.
- `self/core/candidate_rewards.py`: static-frontier aggregation, no-pseudo
  failure metrics, trained-candidate reward, and candidate metric construction.
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
- `self/core/candidate_worker_inputs.py`: candidate-worker shared input
  loading, packed-worker shared-input cache keys, proposal/outcome trace
  buffer loading, proposal prompt reconstruction, and per-pack model bootstrap
  cache setup.
- `self/core/candidate_worker_failures.py`: candidate-worker failure payload
  construction plus metrics-compatible worker-failure path resolution/writing.
- `self/core/candidate_worker_pack_runtime.py`: packed candidate-worker
  execution, including shared input-cache dispatch, backward-compatible
  no-cache runner support, per-candidate failure recording, and pack cache
  summary reporting.
- `self/core/candidate_worker_runtime.py`: candidate-worker spec entry point
  runtime, including spec loading, pseudo-example reconstruction, metric
  generation, and failure handling. It reexports packed-worker execution for
  old import compatibility.
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
- `self/core/composition.py`: exact-pair addition/run-length target
  construction, composition guards, run-length example merging, and
  compatibility reexports for pseudolabel helpers.
- `self/core/composition_pseudolabels.py`: config-style addition/run-length
  self-labeled pseudolabel generation from component predictions plus
  config-vs-executable pseudolabel dispatch.
- `self/core/composition_program_pseudolabels.py`: executable
  program/policy/meta pseudolabel case construction, sandbox execution,
  target-format validation, and target regex selection.
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
- `self/core/experience_trace_models.py`: proposal/outcome trace example
  data models, JSON parsing, and replay sampling.
- `self/core/experience_outcome_traces.py`: outcome trace construction and
  compatibility reexports for older outcome-rendering helper imports.
- `self/core/experience_outcome_rendering.py`: compact outcome-trace
  state/action rendering, numeric/textual target formatting, candidate payload
  extraction, and failure/reward feedback rendering.
- `self/core/experience_traces.py`: proposal trace construction and
  compatibility exports for older trace imports.
- `self/core/frontier.py`: frontier selection and proposal-quality summaries.
- `self/core/model_bootstrap_cache.py`: process-local model/tokenizer
  bootstrap cache containers, cache-key helpers, CPU state-dict cloning, and
  hit/miss summaries used by packed local candidate workers.
- `self/core/model_io.py`: tokenizer construction, special-token syncing,
  added-token embedding initialization, and model loading/instantiation. It
  reexports bootstrap cache containers for old imports; new code should import
  cache types from `model_bootstrap_cache.py`.
- `self/core/models.py`: shared proposal, candidate work-item, candidate
  metrics, and JSON conversion containers.
- `self/core/nonadaptive_bootstrap.py`: non-adaptive resume checkpoint
  selection, model/tokenizer instantiation, training config/collator setup,
  decode-budget derivation, and resumed pseudo-seed loading.
- `self/core/nonadaptive_dataset_context.py`: non-adaptive loaded-dataset
  validation/reporting, composed-eval slice reporting, and eval-key context.
- `self/core/nonadaptive_datasets.py`: non-adaptive initial/composed/eval
  dataset generation and persisted dataset loading.
- `self/core/nonadaptive_evaluation.py`: non-adaptive base/composed round
  evaluation, composed-slice aggregation, and guarded-slice debug sample
  writing.
- `self/core/nonadaptive_finalization.py`: non-adaptive final checkpoint
  cleanup and final result-path reporting.
- `self/core/nonadaptive_lifecycle.py`: non-adaptive post-round lifecycle
  handling, including stop/final-round decisions, trainer release, CUDA cache
  clearing, reset-each-round model release, and reset checkpoint/model reloads.
- `self/core/nonadaptive_loop.py`: top-level non-adaptive
  self-improvement runtime, wiring run setup, dataset/bootstrap preparation,
  round-loop dispatch, and finalization.
- `self/core/nonadaptive_metadata_runtime.py`: non-adaptive RNG seeding,
  resumed RNG-state restoration, and metadata persistence runtime.
- `self/core/nonadaptive_pseudo.py`: non-adaptive dynamic composed-pool
  refresh, next-round pseudo-label generation, pseudo-generation diagnostics,
  and per-round pseudo/composed artifact snapshots.
- `self/core/nonadaptive_results.py`: non-adaptive round summary creation,
  console summary dispatch, `metrics.json` writing, run-level summary-record
  updates, and compatibility-injected summary/payload bindings.
- `self/core/nonadaptive_round_runtime.py`: single-round non-adaptive
  orchestration, wiring round setup, training, evaluation, next pseudo-label
  generation, summary recording, and lifecycle handling.
- `self/core/nonadaptive_round_loop.py`: non-adaptive round iteration,
  dependency forwarding into the single-round runtime, round-directory
  collection, and early-stop handling.
- `self/core/nonadaptive_round_setup.py`: per-round directory/save-policy
  planning plus round training/pseudo-example artifact persistence.
- `self/core/nonadaptive_schedule.py`: non-adaptive size-schedule arithmetic
  for legacy contiguous expansion and explicit frontier-based expansion.
- `self/core/nonadaptive_state.py`: non-adaptive output/data artifact path
  construction, metadata/result loading, loaded-metadata compatibility
  validation, metadata persistence, and config-arg snapshot writing.
- `self/core/nonadaptive_training.py`: non-adaptive per-round recipe phase
  resolution, training argument/trainer construction, round training, and
  model/tokenizer save handling.
- `self/core/program_sandbox.py`: sandboxed composition-program validation
  and bounded execution, with compatibility reexports for sandbox models and
  property-case builders.
- `self/core/program_sandbox_models.py`: sandbox case/result dataclasses and
  repair callback typing.
- `self/core/program_sandbox_cases.py`: addition and run-length
  property-test cases for generated composition programs.
- `self/core/proposals.py`: program proposal shape plus compatibility
  reexports for prompt, config-schema, and proposal-IO helpers.
- `self/core/proposal_config_schema.py`: config proposal dataclasses, search
  spaces, output-schema handling, JSON extraction, config parsing, and
  action-prediction normalization.
- `self/core/proposal_io.py`: proposal fixture loading plus proposal/trace
  JSONL row construction and writing.
- `self/core/proposal_prompts.py`: prompt bundle dataclass, config/program
  prompt rendering, task-specific target formats, default executable
  source-pair selection, sandbox validation-case selection, and executable
  program/policy/meta prompt rendering.
- `self/core/proposal_config_validation.py`: config proposal row output
  normalization, schema/action-prediction validation, duplicate marking, and
  repeat-target annotation.
- `self/core/proposal_generation.py`: proposal fixture row selection,
  current/alternate model sampling, and proposal-model cleanup after
  generation.
- `self/core/proposal_executable_validation.py`: executable
  program/policy/meta proposal parsing, sandbox validation, repair prompt/model
  dispatch, and executable proposal row construction.
- `self/core/proposal_runtime.py`: config-vs-executable proposal validation
  dispatch plus compatibility reexports for proposal generation and executable
  validation helpers. New implementation code should import prompt,
  generation, config-validation, and executable-validation helpers from their
  owner modules rather than through this compatibility wrapper.
- `self/core/proposal_grpo_dispatch.py`: local-vs-Slurm proposal-GRPO update
  dispatch and proposal-GRPO worker input artifact writing.
- `self/core/proposal_grpo_traces.py`: proposal-GRPO reward shaping,
  advantage construction, and raw/normalized proposal trace construction.
- `self/core/proposal_grpo.py`: proposal-GRPO sample encoding, lightweight
  policy update, checkpoint writing, metrics writing, and compatibility
  reexports for old reward/trace imports.
- `self/core/recipe_presets.py`: shared algorithmic self-improvement recipe
  constants, preset dataclasses, phase resolution, and max-step schedule
  compression.
- `self/core/recipes.py`: recipe tokenizer/model construction, recipe training
  arguments, recipe-aware Trainer variants, and compatibility reexports for
  old preset imports.
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
- `self/core/nonadaptive_compat.py`: compatibility helper for syncing patched
  `self.self_improvement_core` facade globals into the canonical
  non-adaptive loop before execution, preserving old monkeypatch-based tests
  and scripts without making implementation modules import through the facade.
- `self/core/nonadaptive_facade_exports.py`: explicit legacy export manifest
  for `self.self_improvement_core`, grouping compatibility names by canonical
  owner while keeping the old facade small and auditable.
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
  protocols, non-adaptive loop execution, and non-adaptive monkeypatch sync
  policy have moved into focused `self/core/` modules. Its public compatibility
  exports now come from `self/core/nonadaptive_facade_exports.py`, and the
  patchable names synced into the non-adaptive loop are tested against that
  export surface.

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
- `self/tasks/bit_common.py`: shared run-length/multiplication parsers,
  normalization helpers, guarded pseudolabel refill, direct pseudolabel
  utilities, and compatibility reexports for older bit-composition imports.
- `self/tasks/multiplication.py`: multiplication task orchestration,
  blocked-component pseudolabel derivation, metadata validation, and
  `MultiplicationTask`.
- `self/tasks/multiplication_data.py`: multiplication example container, key
  encoding, seed/long dataset construction, blocked component payloads, slice
  naming, and override cloning.
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
- `self/tasks/run_length_pseudolabels.py`: run-length direct, guarded-pair,
  run-state, and default tuple pseudolabel derivation.
- `self/tasks/run_length_splits.py`: run-length initial/eval/composed split
  preparation and composed-eval guard-slice partitioning.
- `self/tasks/compat_exports.py`: explicit legacy export manifest for
  `self.self_improvement_tasks`, grouping old-path task exports by protocol,
  evaluation, addition, bit-common, bit-composition, multiplication, and
  run-length owner.
- `self/self_improvement_tasks.py`: explicit compatibility import surface for
  the old task-module path, backed by canonical `self/tasks/*` and
  `self/core/*` modules. Its public compatibility exports now come from
  `self/tasks/compat_exports.py` so old-path compatibility stays auditable
  without keeping the export manifest in the facade itself.

## Current Experiments

- `self/experiments/adaptive_self_improvement.py`: dry-run/pilot proposal
  controller used for fixture-based adaptive proposal analysis.
- `self/experiments/composition_error_sweep.py`: addition self-improvement
  launcher wrapper for controlled boundary-carry composition-error retention.
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

- `self/analysis/artifact_io.py`: shared JSON/JSONL readers and artifact
  filename constants for analysis loaders.
- `self/analysis/adaptive_artifact_common.py`: shared adaptive artifact
  constants, run/attempt dataclasses, index parsing, row-context construction,
  selected-id handling, and proposal-field flattening helpers used by both
  run-level and candidate-level adaptive loaders.
- `self/analysis/adaptive_artifacts.py`: adaptive-run discovery, attempt
  loading, prompt/proposal/trace record flattening, selected-checkpoint
  per-size timelines, and proposal-GRPO metric records.
- `self/analysis/adaptive_candidate_artifacts.py`: adaptive candidate
  artifact loading and candidate/train-mix/per-size row flattening for
  per-candidate metrics, train-mix summaries, and worker-failure files.
- `self/analysis/adaptive_manifest_artifacts.py`: adaptive
  `submission_manifest.json` discovery, loading, and job-row flattening so
  notebooks can recover submitted condition metadata without hard-coding raw
  manifest paths.
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
- `self/analysis/training_curve_plots.py`: task training curves, per-size
  heatmaps, comparison curves, and figure bundle export helpers.
- `self/analysis/training_curve_notebook_utils.py`: notebook helpers for
  compatibility imports over the canonical training-curve analysis modules.
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
  printing to stdout or stderr, colon-separated config-file sourcing, and
  small shell utilities such as boolean parsing for launcher flags.
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
- `launchers/self/config/adaptive_candidate_*.env`: default-only config files
  for adaptive candidate-training runs. Source them through
  `ADAPTIVE_CONFIG_FILE` or colon-separated `ADAPTIVE_CONFIG_FILES`; explicit
  caller environment variables still take precedence.
- `launchers/self/submit_addition_exact_digits_fixed_binary_mig.sh` and
  `launchers/self/submit_addition_fixedwidth_moredata_mig.sh` now use the
  generic launcher helper for repo-root setup and common MIG Slurm resource
  argument construction; the exact-digits submitter also routes its schedule
  banner through the shared context printer.
- `launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch`,
  `launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh`, and
  `launchers/self/submit_addition_fixedwidth_mixed_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing, with dry-run coverage for the submitter's
  seed/fullpack/original-composition branches.
- `launchers/self/run_addition_fullpack_filtered.sbatch` and
  `launchers/self/submit_addition_fullpack_filtered_mig.sh` use the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing.
- `launchers/self/run_addition_their_recipe_diagnostic.sh` uses the generic
  helper for repo-root/Python setup, boolean parsing, context printing, and
  dry-run command printing before delegating to the addition recipe diagnostic
  CLI.
- `launchers/self/run_addition_recipe_focused.sh`,
  `launchers/self/run_addition_recipe_fullpack.sh`, and
  `launchers/self/run_addition_recipe_recovery.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, and dry-run command
  printing across the recipe recovery workflow.
- `launchers/self/run_addition_tiny_seed_mig.sbatch` and
  `launchers/self/run_addition_seed_shared.sbatch` use the generic helper for
  repo-root/Python setup, boolean parsing, context printing, and staged
  dry-run command printing across the addition seed workflows.
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
  The seed-sweep submitter also uses the shared sbatch-script submission helper.
- `launchers/self/submit_multiplication_rectangular_tune_mig.sh` uses the
  generic helper for repo-root/Python setup, boolean parsing, and stdout
  command printing before delegating to the rectangular tune CLI.
- `launchers/self/run_seed_fit_experiment.sbatch` and
  `launchers/self/submit_seed_fit_grid.sh` use the generic helper for
  repo-root/Python setup where needed, boolean parsing, context printing, and
  seed-fit dry-run verification.
- `launchers/self/run_task_self_improvement.sbatch` and
  `launchers/self/submit_budget_grid_self_improvement.sh` use the generic
  helper for repo-root/Python setup where needed, boolean parsing, context
  printing, and self-improvement budget-grid dry-run verification.
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
- `self/addition_recipe.py` -> `self/core/recipes.py`
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
- `self/self_improvement_composition_error_experiment.py` -> `self/experiments/composition_error_sweep.py`
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
- `self/self_improvement_experiment.py` -> `self/legacy/addition_self_improvement.py`
- `self/self_improvement_multiplication_cot_pseudo_addition.py` -> `self/legacy/multiplication_cot_pseudo_addition.py`
- `self/multiplication_self_improvement.py` -> `self/legacy/multiplication_self_improvement.py`
- `self/run_length_self_improvement.py` -> `self/legacy/run_length_self_improvement.py`
- `self/slurm_utils.py` -> `self/core/slurm.py`
- `self/summarize_seed_fit_grid.py` -> `self/analysis/summarize_seed_fit_grid.py`
- `self/training_curve_notebook_utils.py` -> `self/analysis/training_curve_notebook_utils.py`

## Remaining Cleanup Queue

- Continue splitting `self/core/driver.py` into smaller modules for round-loop
  orchestration. The CLI/args, shared data model, composition/pseudolabel,
  generic controller-worker dispatch, controller-worker spec runtime,
  seed/round-model controller phases, candidate train/eval runtime, candidate
  scoring orchestration, candidate training-mix construction/artifact writing,
  candidate reward/metric construction, checkpoint cleanup, proposal runtime
  generation/validation, proposal prompt rendering, candidate data
  construction, dry-run attempt handling, selected/no-selection attempt outcome
  handling, proposal-GRPO dispatch, run setup/trace loading,
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
  construction into `self/tasks/run_length_data.py`, bit-string composition
  path helpers into `self/tasks/bit_composition.py`, and round-target
  pseudolabel derivation into `self/tasks/run_length_pseudolabels.py`.
  Initial/eval/composed split preparation and composed-eval guard slicing live
  in `self/tasks/run_length_splits.py`.
  Multiplication example/data construction now lives in
  `self/tasks/multiplication_data.py`, rectangular partition helpers live in
  `self/tasks/rectangular_partitions.py`, rectangular digit-order/reverse-CoT
  helpers live in `self/tasks/rectangular_digits.py`, and rectangular example,
  parsing, prediction-normalization, and sampled-data helpers live in
  `self/tasks/rectangular_data.py`. Rectangular component construction and
  target composition now live in `self/tasks/rectangular_composition.py`, while
  `MultiplicationTask` remains the orchestration layer.
- `self/self_improvement_core.py` is now a compatibility facade. Continue
  migrating internal imports to canonical `self/core/*` modules when touching
  old top-level scripts, but preserve the facade for old imports and
  monkeypatch-based tests. Current `self/core`, `self/tasks`,
  `self/experiments`, `self/diagnostics`, and `self/analysis`
  implementation modules no longer import through this facade. The facade
  keeps grouped `__all__` exports so legacy imports and patch points remain
  auditable.
- Experience trace example models, JSON parsing, and replay sampling now live
  in `self/core/experience_trace_models.py`; outcome trace construction lives
  in `self/core/experience_outcome_traces.py`; outcome rendering/payload
  helpers live in `self/core/experience_outcome_rendering.py`; proposal trace
  construction remains in `self/core/experience_traces.py`, which still
  reexports old trace imports for compatibility.
- `self/core/nonadaptive_loop.py` still owns the main non-adaptive run setup,
  bootstrap, and finalization. Round iteration now lives in
  `self/core/nonadaptive_round_loop.py`; single-round orchestration now lives
  in `self/core/nonadaptive_round_runtime.py`; dataset-context validation
  and reporting now live in `self/core/nonadaptive_dataset_context.py`;
  base/composed round evaluation now lives in `self/core/nonadaptive_evaluation.py`; per-round
  training setup and execution lives in `self/core/nonadaptive_training.py`;
  dataset generation/loading lives in `self/core/nonadaptive_datasets.py`;
  resume/model/bootstrap setup lives in `self/core/nonadaptive_bootstrap.py`;
  RNG-state and metadata persistence lives in
  `self/core/nonadaptive_metadata_runtime.py`;
  final checkpoint cleanup and result-path reporting live in
  `self/core/nonadaptive_finalization.py`;
  preflight/default normalization and derived setup values live in
  `self/core/nonadaptive_setup.py`; output/data artifact paths plus metadata
  IO and loaded-metadata compatibility checks live in
  `self/core/nonadaptive_state.py`; deterministic size/frontier arithmetic
  lives in `self/core/nonadaptive_schedule.py`; and per-round directory,
  save-policy, and round input artifact setup lives in
  `self/core/nonadaptive_round_setup.py`. These helpers have direct unit
  coverage.
- Recipe presets now live in `self/core/recipe_presets.py`; runtime recipe
  helpers live in `self/core/recipes.py`; `self/self_improvement_recipe.py`
  and `self/addition_recipe.py` are compatibility wrappers. Current
  implementation modules should import preset-only helpers from
  `recipe_presets.py` and model/trainer helpers from `recipes.py`, while legacy
  scripts and old tests can keep using the wrappers.
- Compatibility wrappers that need monkeypatch-style write forwarding should use
  `self/core/module_proxy.py`. The top-level rectangular, adaptive,
  addition-recipe, legacy addition, and composition-error proxy wrappers now use
  the shared helper instead of carrying local proxy classes. Star-import CLI
  wrappers can use `module_star_export_names(...)` with the same helper to
  preserve old `from self.wrapper import *` behavior while avoiding repeated
  local wrapper boilerplate; current analysis, diagnostic, experiment, and
  legacy CLI wrappers follow this pattern. `self/self_improvement.py` remains
  the one special legacy wrapper that keeps eager star-import globals while also
  forwarding writes through the proxy.
- Tokenizer helpers now live in `self/core/tokenizers.py`; `self/task_tokenizer.py`
  is a compatibility wrapper. Current core modules import tokenizer helpers
  from the canonical core module, while legacy scripts and old tests can keep
  using the wrapper path.
- Current implementation packages also avoid importing task symbols through
  `self/self_improvement_tasks.py`; `self/core`, `self/experiments`,
  `self/diagnostics`, and `self/analysis` import task classes, run-length
  parsers, and constants from their canonical `self/tasks/*` owners. The old
  task facade remains for top-level compatibility scripts and legacy imports,
  with grouped exports and tests guarding duplicate or missing legacy names.
- Continue moving historical baselines and paper-specific scripts into
  `self/legacy/`, `self/experiments`, `self/diagnostics/`, or `self/analysis/`
  gradually with wrapper modules where old launchers still need them. The first
  diagnostics, paper/figure, analysis-utility, seed-fit, and rectangular
  multiplication experiment batches have moved under canonical package
  directories, the rectangular multiplication shared helper now lives under
  `self/tasks/`, and the historical addition self-improvement CLI now lives
  in `self/legacy/addition_self_improvement.py`. The historical run-length
  and multiplication CLIs, plus the run-length bit-string helper, have moved
  under `self/legacy/`,
  the multiplication CoT
  pseudo-addition curriculum script has
  moved under `self/legacy/`, and the balanced run-length evaluation
  diagnostic has moved under `self/diagnostics/`.
- Continue consolidating Slurm launchers into fewer templates plus explicit
  matrix configs. The adaptive AILAB parent/candidate-worker/controller-worker
  scripts now share `launchers/self/lib/adaptive_common.sh`; the generic
  helper lives in `launchers/self/lib/self_common.sh` and owns common repo-root
  setup, Python resolution, context banners, and stdout/stderr command
  printing/execution helpers; the adaptive
  condition-pilot, main-experiment, exact-digit addition, and fixed-width
  more-data addition submitters use common Slurm resource/default helpers, with
  the exact-digit schedule banner routed through `self_print_context`; the
  fixed-width mixed-prompt addition seed/fullpack runners and mixed submitter
  share generic repo-root/Python/boolean/context-printing/command-printing
  helpers; the addition
  fullpack-filtered runner/submitter share generic
  repo-root/Python/boolean/context-printing/command-printing helpers; the
  addition recipe
  diagnostic launcher and recipe focused/fullpack/recovery workflow share the
  same helper path, with focused/fullpack/recovery context banners routed
  through `self_print_context`; the run-length fixed-binary submitter uses the
  generic explicit-resource and context-printing helpers for its mixed GPU/CPU
  jobs; the guarded
  plain-output bit diagnostic runner and submitter share generic
  repo-root/Python/context-printing/command-printing helpers; the rectangular
  square
  seed/diagnostic runners and probe/resweep submitters share generic
  repo-root/Python/boolean/context-printing/command-printing helpers; the
  rectangular non-square seed/self-improvement runners and seed/fullpack
  submitters share generic
  repo-root/Python/boolean/context-printing/stdout-or-stderr command-printing
  helpers; the rectangular tune submitter shares the same repo-root/Python setup
  and dry-run flag helper;
  the addition tiny/shared seed runners share generic
  repo-root/Python/boolean/context-printing and staged dry-run
  command-printing helpers;
  the seed-fit grid runner/submitter share generic
  repo-root/Python/boolean/context-printing helpers and now have a dry-run
  verification path; the generic task
  self-improvement budget-grid runner/submitter share the same helper pattern,
  including context printing;
  the small Figure 2/3 submitter wrappers share the helper path, including
  shared dry-run flag insertion, context printing, and print-then-execute
  command dispatch, before delegating to their Python experiment CLIs; the
  Figure 2 recipe
  runner/submitter/retune scripts share generic repo-root/Python/boolean,
  context-printing, and command-printing helpers; the composition-error sweep
  now invokes the
  canonical `self.experiments.composition_error_sweep` module while preserving
  the old top-level wrapper, and the local/refocused
  workshop batchers, run-length alpha10 baseline submitter, and self-improvement
  MIG runners also share the generic helper path, leaving the current
  `launchers/self/*.{sh,sbatch}` inventory without duplicated repo-root/Python
  setup;
  and adaptive candidate-training defaults can be loaded from
  `launchers/self/config/adaptive_candidate_*.env`.
- Use `self/analysis/adaptive_artifacts.py` and
  `self/analysis/adaptive_candidate_artifacts.py` for new adaptive notebooks,
  and keep `self/analysis/artifacts.py` as the compatibility surface for older
  notebooks while migrating direct raw JSON parsing when files are edited. The
  candidate loader exposes helpers for per-candidate metrics, train-mix
  summaries, and worker-failure files, so notebooks should not glob
  `attempt_*/candidates/candidate_*` directly.
- Keep source/artifact hygiene conservative: source notebooks and report
  sources remain visible, while executed notebooks, heavyweight model/run
  bundles, local cache/editor state, and command/LaTeX logs are ignored.
- Optimize candidate training to avoid repeated full model reloads for each
  local candidate worker where semantics allow it. Current packed-local workers
  reduce subprocess and shared-input IO/parsing overhead by default through
  `self/core/candidate_worker_inputs.py`, reuse tokenizer bootstrap work inside
  each pack, and report bootstrap cache hit/miss counters. With
  `--candidate-local-cache-base-state`, packed workers also avoid repeated disk
  reads of the shared source checkpoint while still instantiating a fresh model
  object per candidate from an unmodified cached CPU state.
