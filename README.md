# Experiment Layout

Repository modules are organized into:

- `core`: shared addition/composition pipeline utilities
- `self`: self-improvement experiment entrypoints
- `meta`: meta self-improvement experiment entrypoints
- `launchers`: Slurm/shell launch scripts
- `legacy/weak_to_strong`: archived weak-to-strong scripts

## Canonical Commands

Run self-improvement:

```bash
python -m self.self_improvement --help
```

Run self-improvement composition wrapper:

```bash
python -m self.self_improvement_composition_error_experiment --help
```

Run multiplication self-improvement:

```bash
python -m self.multiplication_self_improvement --help
```

Run run-length self-improvement:

```bash
python -m self.run_length_self_improvement --help
```

Plot a self-improvement figure from one or more completed runs:

```bash
python -m self.plot_self_improvement_figure RUN_DIR [RUN_DIR ...]
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
