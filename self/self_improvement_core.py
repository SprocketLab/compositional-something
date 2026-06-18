#!/usr/bin/env python3
"""Task-agnostic scaffold for iterative compositional self-improvement."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import set_seed

from self.core import nonadaptive_loop as _nonadaptive_loop
from self.core.data_io import (
    cleanup_round_checkpoints,
    decode_rng_state,
    encode_rng_state,
    ensure_dir,
    load_examples,
    load_summary_records,
    resolve_save_model_policy,
    sanitize_json_value,
    save_examples,
    write_summary_records,
)
from self.core.evaluation import (
    build_generation_encodings,
    evaluate_accuracy_with_breakdown,
    extract_numeric_answer,
    generate_prediction_map,
    parse_prediction,
    resolve_max_new_tokens,
    write_prediction_debug_samples,
)
from self.core.model_io import (
    add_token_initializers,
    initialize_copied_embeddings,
    instantiate_model_and_tokenizer,
    load_model_for_tokenizer,
    load_model_from_config,
    lookup_single_token_id,
    sync_model_special_token_ids,
)
from self.core.nonadaptive_compat import (
    NONADAPTIVE_PATCHABLE_NAMES,
    sync_nonadaptive_loop_globals,
)
from self.core.recipes import (
    PaddingAwareCausalLMDataCollator,
    instantiate_recipe_model,
    load_recipe_model,
    recipe_enabled,
    resolve_self_improvement_recipe,
)
from self.core.summaries import (
    RoundSummary,
    SliceMetric,
    format_accuracy,
    summarize_round,
    summary_to_payload,
)
from self.core.task_protocols import (
    JsonDict,
    KeyGetter,
    PredictionParser,
    PromptTargetExample,
    SelfImprovementTask,
    SizeGetter,
    SplitName,
)
from self.core.training import (
    BatchSamplerTrainer,
    CausalLMDataCollator,
    SizeBucketBatchSampler,
    TRAINING_ARGUMENT_FIELDS,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    build_trainer,
    make_training_args,
    training_arg_supported,
)

_MODULE_EXPORTS = (
    "annotations",
    "json",
    "math",
    "random",
    "Path",
    "torch",
    "set_seed",
)

_TYPING_EXPORTS = (
    "Any",
    "Dict",
    "List",
    "Optional",
)

_DATA_IO_EXPORTS = (
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

_EVALUATION_EXPORTS = (
    "build_generation_encodings",
    "evaluate_accuracy_with_breakdown",
    "extract_numeric_answer",
    "generate_prediction_map",
    "parse_prediction",
    "resolve_max_new_tokens",
    "write_prediction_debug_samples",
)

_MODEL_IO_EXPORTS = (
    "add_token_initializers",
    "initialize_copied_embeddings",
    "instantiate_model_and_tokenizer",
    "load_model_for_tokenizer",
    "load_model_from_config",
    "lookup_single_token_id",
    "sync_model_special_token_ids",
)

_RECIPE_EXPORTS = (
    "PaddingAwareCausalLMDataCollator",
    "instantiate_recipe_model",
    "load_recipe_model",
    "recipe_enabled",
    "resolve_self_improvement_recipe",
)

_SUMMARY_EXPORTS = (
    "RoundSummary",
    "SliceMetric",
    "format_accuracy",
    "summarize_round",
    "summary_to_payload",
)

_TASK_PROTOCOL_EXPORTS = (
    "JsonDict",
    "KeyGetter",
    "PredictionParser",
    "PromptTargetExample",
    "SelfImprovementTask",
    "SizeGetter",
    "SplitName",
)

_TRAINING_EXPORTS = (
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

_NONADAPTIVE_COMPAT_EXPORTS = (
    "NONADAPTIVE_PATCHABLE_NAMES",
    "sync_nonadaptive_loop_globals",
    "run_self_improvement",
)

_COMPAT_EXPORT_GROUPS = (
    _MODULE_EXPORTS,
    _TYPING_EXPORTS,
    _DATA_IO_EXPORTS,
    _EVALUATION_EXPORTS,
    _MODEL_IO_EXPORTS,
    _RECIPE_EXPORTS,
    _SUMMARY_EXPORTS,
    _TASK_PROTOCOL_EXPORTS,
    _TRAINING_EXPORTS,
    _NONADAPTIVE_COMPAT_EXPORTS,
)

__all__ = [name for group in _COMPAT_EXPORT_GROUPS for name in group]

_NONADAPTIVE_PATCHABLE_NAMES = NONADAPTIVE_PATCHABLE_NAMES


def run_self_improvement(args: Any, task: SelfImprovementTask) -> None:
    sync_nonadaptive_loop_globals(
        source_globals=globals(),
        target_module=_nonadaptive_loop,
        names=_NONADAPTIVE_PATCHABLE_NAMES,
    )
    return _nonadaptive_loop.run_self_improvement(args, task)
