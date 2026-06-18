#!/usr/bin/env python3
"""Lazy compatibility exports for :mod:`self.self_improvement_core`."""

from __future__ import annotations

from typing import Any

from self.core.lazy_exports import (
    LazyExportTarget,
    lazy_export_dir,
    resolve_lazy_export,
    validate_lazy_export_targets,
)

NONADAPTIVE_FACADE_EXPORT_TARGETS: dict[str, LazyExportTarget] = {
    "annotations": ("__future__", "annotations"),
    "json": ("json", None),
    "math": ("math", None),
    "random": ("random", None),
    "Path": ("pathlib", "Path"),
    "torch": ("torch", None),
    "set_seed": ("transformers", "set_seed"),
    "Any": ("typing", "Any"),
    "Dict": ("typing", "Dict"),
    "List": ("typing", "List"),
    "Optional": ("typing", "Optional"),
    "cleanup_round_checkpoints": ("self.core.data_io", "cleanup_round_checkpoints"),
    "decode_rng_state": ("self.core.data_io", "decode_rng_state"),
    "encode_rng_state": ("self.core.data_io", "encode_rng_state"),
    "ensure_dir": ("self.core.data_io", "ensure_dir"),
    "load_examples": ("self.core.data_io", "load_examples"),
    "load_summary_records": ("self.core.data_io", "load_summary_records"),
    "resolve_save_model_policy": ("self.core.data_io", "resolve_save_model_policy"),
    "sanitize_json_value": ("self.core.data_io", "sanitize_json_value"),
    "save_examples": ("self.core.data_io", "save_examples"),
    "write_summary_records": ("self.core.data_io", "write_summary_records"),
    "build_generation_encodings": ("self.core.evaluation", "build_generation_encodings"),
    "evaluate_accuracy_with_breakdown": ("self.core.evaluation", "evaluate_accuracy_with_breakdown"),
    "extract_numeric_answer": ("self.core.evaluation", "extract_numeric_answer"),
    "generate_prediction_map": ("self.core.evaluation", "generate_prediction_map"),
    "parse_prediction": ("self.core.evaluation", "parse_prediction"),
    "resolve_max_new_tokens": ("self.core.evaluation", "resolve_max_new_tokens"),
    "write_prediction_debug_samples": ("self.core.evaluation", "write_prediction_debug_samples"),
    "add_token_initializers": ("self.core.model_io", "add_token_initializers"),
    "initialize_copied_embeddings": ("self.core.model_io", "initialize_copied_embeddings"),
    "instantiate_model_and_tokenizer": ("self.core.model_io", "instantiate_model_and_tokenizer"),
    "load_model_for_tokenizer": ("self.core.model_io", "load_model_for_tokenizer"),
    "load_model_from_config": ("self.core.model_io", "load_model_from_config"),
    "lookup_single_token_id": ("self.core.model_io", "lookup_single_token_id"),
    "sync_model_special_token_ids": ("self.core.model_io", "sync_model_special_token_ids"),
    "NONADAPTIVE_PATCHABLE_NAMES": ("self.nonadaptive.nonadaptive_compat", "NONADAPTIVE_PATCHABLE_NAMES"),
    "sync_nonadaptive_loop_globals": ("self.nonadaptive.nonadaptive_compat", "sync_nonadaptive_loop_globals"),
    "instantiate_recipe_model": ("self.core.recipe_models", "instantiate_recipe_model"),
    "load_recipe_model": ("self.core.recipe_models", "load_recipe_model"),
    "recipe_enabled": ("self.core.recipe_presets", "recipe_enabled"),
    "resolve_self_improvement_recipe": ("self.core.recipe_presets", "resolve_self_improvement_recipe"),
    "PaddingAwareCausalLMDataCollator": ("self.core.recipe_training", "PaddingAwareCausalLMDataCollator"),
    "RoundSummary": ("self.core.summaries", "RoundSummary"),
    "SliceMetric": ("self.core.summaries", "SliceMetric"),
    "format_accuracy": ("self.core.summaries", "format_accuracy"),
    "summarize_round": ("self.core.summaries", "summarize_round"),
    "summary_to_payload": ("self.core.summaries", "summary_to_payload"),
    "JsonDict": ("self.core.task_protocols", "JsonDict"),
    "KeyGetter": ("self.core.task_protocols", "KeyGetter"),
    "PredictionParser": ("self.core.task_protocols", "PredictionParser"),
    "PromptTargetExample": ("self.core.task_protocols", "PromptTargetExample"),
    "SelfImprovementTask": ("self.core.task_protocols", "SelfImprovementTask"),
    "SizeGetter": ("self.core.task_protocols", "SizeGetter"),
    "SplitName": ("self.core.task_protocols", "SplitName"),
    "BatchSamplerTrainer": ("self.core.training", "BatchSamplerTrainer"),
    "CausalLMDataCollator": ("self.core.training", "CausalLMDataCollator"),
    "SizeBucketBatchSampler": ("self.core.training", "SizeBucketBatchSampler"),
    "TRAINING_ARGUMENT_FIELDS": ("self.core.training", "TRAINING_ARGUMENT_FIELDS"),
    "TokenizedPromptTargetDataset": ("self.core.training", "TokenizedPromptTargetDataset"),
    "TrainingConfig": ("self.core.training", "TrainingConfig"),
    "build_trainer": ("self.core.training", "build_trainer"),
    "make_training_args": ("self.core.training", "make_training_args"),
    "training_arg_supported": ("self.core.training", "training_arg_supported"),
}


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
