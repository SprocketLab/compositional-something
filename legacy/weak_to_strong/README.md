# Weak-To-Strong Archive

This folder keeps archived weak-to-strong launch scripts for reproducing older
addition experiments. The maintained entry point is
`weak_to_strong_addition_experiment_v2.py`, a thin wrapper around the canonical
`core.addition_pipeline` implementation.

The older standalone addition experiment monolith was removed after checkpoint
tag `pre-outdated-cleanup-20260618-204614`. Recover it from git history if a
historical rerun requires the exact pre-refactor script.
