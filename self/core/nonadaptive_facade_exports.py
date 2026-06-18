#!/usr/bin/env python3
"""Compatibility export manifest for :mod:`self.self_improvement_core`."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import set_seed

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
    TRAINING_ARGUMENT_FIELDS,
    BatchSamplerTrainer,
    CausalLMDataCollator,
    SizeBucketBatchSampler,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    build_trainer,
    make_training_args,
    training_arg_supported,
)

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

__all__ = list(NONADAPTIVE_FACADE_BASE_EXPORT_NAMES)
