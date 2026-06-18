# Experiment Layout

Repository modules are organized into:

- `core`: shared addition/composition pipeline utilities
- `self`: self-improvement experiment entrypoints
- `meta`: meta self-improvement experiment entrypoints
- `launchers`: Slurm/shell launch scripts
- `legacy/weak_to_strong`: archived weak-to-strong launch scripts and wrappers
  around the canonical addition pipeline. The pre-refactor standalone addition
  monolith was removed after checkpoint tag
  `pre-outdated-cleanup-20260618-204614`.

## Canonical Commands

Run self-improvement:

```bash
python -m self.legacy.addition_self_improvement --help
```

Run self-improvement composition wrapper:

```bash
python -m self.experiments.composition_error_sweep --help
```

Run multiplication self-improvement:

```bash
python -m self.legacy.multiplication_self_improvement --help
```

Run run-length self-improvement:

```bash
python -m self.legacy.run_length_self_improvement --help
```

Plot a self-improvement figure from one or more completed runs:

```bash
python -m self.analysis.plot_self_improvement_figure RUN_DIR [RUN_DIR ...]
```

Run meta self-improvement:

```bash
python -m meta.train_meta_self_improvement_rope --help
```

Artifact outputs default to `artifacts/`.

The workshop defaults now target `HuggingFaceTB/SmolLM2-360M` and use `bf16`
automatically on CUDA when neither `--bf16` nor `--fp16` is specified.

The Slurm launchers prefer a local `artifacts/models/SmolLM2-360M` snapshot
when it exists and otherwise fall back to the Hugging Face model id
`HuggingFaceTB/SmolLM2-360M`.
