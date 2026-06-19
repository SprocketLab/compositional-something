# Self-Improvement Code Layout

This package now keeps implementation code under subsystem directories. The
top-level `self/` package contains only `__init__.py` and
`launcher_manifests.py`; CLI entrypoints live under `self/legacy/`,
`self/experiments/`, `self/diagnostics/`, or `self/analysis/`.

## Supported Pseudolabel Modes

- `none`: no generated labels are added.
- `direct`: the current model labels the target example directly.
- `compose`: the current model labels source components, then task-specific
  composition builds the target label.
- `compose_corrupt`: the `compose` control with injected label noise.

Any new label-family baseline should be added as an explicit experimental mode
with tests, launcher coverage, and paper-facing documentation before it becomes
part of the supported surface.

## Implementation Map

- `self/adaptive/`: flat adaptive implementation modules. Use `driver.py` for
  the CLI facade and dependency wiring, `args.py` for argument parsing,
  `run.py` for run setup/dispatch/orchestration/finalization, `attempts.py`
  for attempt-loop behavior, `candidate.py` for candidate data/training/worker
  execution, `proposal.py` for proposal schemas/prompts/validation/runtime/GRPO,
  `traces.py` for proposal and outcome traces, `frontier.py` for frontier
  helpers, `controller.py` for controller workers, `program_sandbox.py` for
  generated-program checks, and `phases.py` for lightweight phase constants.
- `self/nonadaptive/`: nonadaptive setup, lifecycle, per-round runtime,
  pseudolabel generation, results, and the public `nonadaptive_loop` facade.
- `self/core/`: shared utilities only: composition, data/model IO, evaluation,
  dataclasses, recipes, Slurm helpers, tokenizers, training utilities, task
  protocols, summaries, and worker IO.
- `self/tasks/`: task-owned modules: `addition.py`, `bit.py`,
  `multiplication.py`, `rectangular.py`, and `run_length.py`, plus package
  exports in `__init__.py`.
- `self/analysis/`: compact notebook/plotting loaders:
  `adaptive_artifacts.py`, `training_curves.py`, `seed_fit.py`,
  `nonadaptive_artifacts.py`, `artifact_io.py`, `artifacts.py`, and plotting
  CLIs.

## Canonical Commands

Use module paths rather than deleted top-level wrapper paths:

```bash
python -m self.legacy.addition_self_improvement --help
python -m self.legacy.run_length_self_improvement --help
python -m self.legacy.multiplication_self_improvement --help
python -m self.experiments.seed_fit_experiment --help
python -m self.experiments.figure2_condition_sweep --help
python -m self.experiments.figure3_seed_quality_sweep --help
python -m self.diagnostics.check_self_improvement_overfit --help
python -m self.analysis.plot_self_improvement_figure --help
python -m self.adaptive.driver --help
```

## Cleanup Policy

- Do not add new top-level `self/*.py` wrappers for implementation modules.
- Do not add adaptive subpackages; keep `self/adaptive` flat unless there is a
  strong reason to reintroduce a directory.
- Do not reintroduce adaptive prefix-split files such as `candidate_*`,
  `proposal_*`, `run_*`, or `driver_*`; add small helpers to the owning module
  unless there is a clear new subsystem boundary.
- Prefer task-owned or subsystem-owned modules over prefix-split helper files.
- Keep compatibility only when a tracked launcher or current notebook needs it;
  otherwise migrate the caller to the canonical module.
- If a module grows again because multiple concepts are mixed together, split by
  concept, not by one-function files.
