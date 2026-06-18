"""Compatibility helpers for the legacy non-adaptive facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


NONADAPTIVE_PATCHABLE_NAMES: tuple[str, ...] = (
    "json",
    "math",
    "random",
    "Path",
    "torch",
    "set_seed",
    "cleanup_round_checkpoints",
    "decode_rng_state",
    "encode_rng_state",
    "ensure_dir",
    "load_examples",
    "load_summary_records",
    "resolve_save_model_policy",
    "sanitize_json_value",
    "save_examples",
    "write_summary_records",
    "evaluate_accuracy_with_breakdown",
    "resolve_max_new_tokens",
    "write_prediction_debug_samples",
    "instantiate_model_and_tokenizer",
    "instantiate_recipe_model",
    "load_recipe_model",
    "load_model_for_tokenizer",
    "recipe_enabled",
    "resolve_self_improvement_recipe",
    "PaddingAwareCausalLMDataCollator",
    "CausalLMDataCollator",
    "TokenizedPromptTargetDataset",
    "TrainingConfig",
    "SliceMetric",
    "RoundSummary",
    "summarize_round",
    "summary_to_payload",
    "make_training_args",
    "build_trainer",
)


def sync_nonadaptive_loop_globals(
    *,
    source_globals: Mapping[str, Any],
    target_module: Any,
    names: Sequence[str] = NONADAPTIVE_PATCHABLE_NAMES,
) -> None:
    """Copy facade globals into the non-adaptive loop before execution.

    Old tests and scripts patch symbols on ``self.self_improvement_core`` before
    calling ``run_self_improvement``. The canonical implementation now lives in
    ``self.nonadaptive.nonadaptive_loop``, so this sync preserves those old patch
    points without making the canonical loop import through the facade.
    """

    for name in names:
        setattr(target_module, name, source_globals[name])
