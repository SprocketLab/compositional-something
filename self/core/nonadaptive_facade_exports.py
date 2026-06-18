#!/usr/bin/env python3
"""Lazy compatibility exports for :mod:`self.self_improvement_core`."""

from __future__ import annotations

from typing import Any

from self.core.lazy_exports import (
    lazy_export_dir,
    resolve_lazy_export,
    validate_lazy_export_targets,
)
from self.core.nonadaptive_facade_targets import NONADAPTIVE_FACADE_EXPORT_TARGETS

MODULE_EXPORTS = (
    "annotations",
    "json",
    "math",
    "random",
    "Path",
    "torch",
    "set_seed",
)

TYPING_EXPORTS = (
    "Any",
    "Dict",
    "List",
    "Optional",
)

DATA_IO_EXPORTS = (
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
)

EVALUATION_EXPORTS = (
    "build_generation_encodings",
    "evaluate_accuracy_with_breakdown",
    "extract_numeric_answer",
    "generate_prediction_map",
    "parse_prediction",
    "resolve_max_new_tokens",
    "write_prediction_debug_samples",
)

MODEL_IO_EXPORTS = (
    "add_token_initializers",
    "initialize_copied_embeddings",
    "instantiate_model_and_tokenizer",
    "load_model_for_tokenizer",
    "load_model_from_config",
    "lookup_single_token_id",
    "sync_model_special_token_ids",
)

RECIPE_EXPORTS = (
    "PaddingAwareCausalLMDataCollator",
    "instantiate_recipe_model",
    "load_recipe_model",
    "recipe_enabled",
    "resolve_self_improvement_recipe",
)

SUMMARY_EXPORTS = (
    "RoundSummary",
    "SliceMetric",
    "format_accuracy",
    "summarize_round",
    "summary_to_payload",
)

TASK_PROTOCOL_EXPORTS = (
    "JsonDict",
    "KeyGetter",
    "PredictionParser",
    "PromptTargetExample",
    "SelfImprovementTask",
    "SizeGetter",
    "SplitName",
)

TRAINING_EXPORTS = (
    "BatchSamplerTrainer",
    "CausalLMDataCollator",
    "SizeBucketBatchSampler",
    "TRAINING_ARGUMENT_FIELDS",
    "TokenizedPromptTargetDataset",
    "TrainingConfig",
    "build_trainer",
    "make_training_args",
    "training_arg_supported",
)

NONADAPTIVE_COMPAT_EXPORTS = (
    "NONADAPTIVE_PATCHABLE_NAMES",
    "sync_nonadaptive_loop_globals",
)

NONADAPTIVE_FACADE_BASE_EXPORT_GROUPS = (
    MODULE_EXPORTS,
    TYPING_EXPORTS,
    DATA_IO_EXPORTS,
    EVALUATION_EXPORTS,
    MODEL_IO_EXPORTS,
    RECIPE_EXPORTS,
    SUMMARY_EXPORTS,
    TASK_PROTOCOL_EXPORTS,
    TRAINING_EXPORTS,
    NONADAPTIVE_COMPAT_EXPORTS,
)

NONADAPTIVE_FACADE_BASE_EXPORT_NAMES = tuple(
    name for group in NONADAPTIVE_FACADE_BASE_EXPORT_GROUPS for name in group
)
NONADAPTIVE_FACADE_EXPORT_NAMES = (*NONADAPTIVE_FACADE_BASE_EXPORT_NAMES, "run_self_improvement")

validate_lazy_export_targets(
    export_names=NONADAPTIVE_FACADE_BASE_EXPORT_NAMES,
    targets=NONADAPTIVE_FACADE_EXPORT_TARGETS,
    label="non-adaptive facade",
)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        targets=NONADAPTIVE_FACADE_EXPORT_TARGETS,
        module_globals=globals(),
    )


def __dir__() -> list[str]:
    return lazy_export_dir(globals(), NONADAPTIVE_FACADE_BASE_EXPORT_NAMES)


__all__ = list(NONADAPTIVE_FACADE_BASE_EXPORT_NAMES)
