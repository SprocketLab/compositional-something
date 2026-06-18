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

from self.core.recipes import (
    PaddingAwareCausalLMDataCollator,
    instantiate_recipe_model,
    load_recipe_model,
    recipe_enabled,
    resolve_self_improvement_recipe,
)



_NONADAPTIVE_PATCHABLE_NAMES = (
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


def _sync_nonadaptive_loop_globals() -> None:
    """Preserve old monkeypatch/import behavior for run_self_improvement."""

    for name in _NONADAPTIVE_PATCHABLE_NAMES:
        setattr(_nonadaptive_loop, name, globals()[name])


def run_self_improvement(args: Any, task: SelfImprovementTask) -> None:
    _sync_nonadaptive_loop_globals()
    return _nonadaptive_loop.run_self_improvement(args, task)
