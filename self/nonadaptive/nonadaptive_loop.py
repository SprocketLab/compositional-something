#!/usr/bin/env python3
"""Non-adaptive iterative self-improvement loop runtime."""

from __future__ import annotations

import json
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
from self.nonadaptive.nonadaptive_bootstrap import prepare_nonadaptive_bootstrap
from self.nonadaptive.nonadaptive_dataset_context import prepare_nonadaptive_dataset_context
from self.nonadaptive.nonadaptive_datasets import prepare_nonadaptive_datasets
from self.nonadaptive.nonadaptive_finalization import finalize_nonadaptive_run
from self.nonadaptive.nonadaptive_metadata_runtime import prepare_nonadaptive_metadata_runtime
from self.nonadaptive.nonadaptive_round_context import (
    build_nonadaptive_round_runtime_bundle,
    build_nonadaptive_round_runtime_kwargs,
)
from self.nonadaptive.nonadaptive_round_loop import run_nonadaptive_round_loop
from self.nonadaptive.nonadaptive_round_runtime import run_nonadaptive_round
from self.nonadaptive.nonadaptive_setup import prepare_nonadaptive_run_setup
from self.nonadaptive.nonadaptive_state import (
    persist_nonadaptive_metadata,
    prepare_nonadaptive_run_state,
    write_nonadaptive_config_args,
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

from self.core.recipe_models import instantiate_recipe_model, load_recipe_model
from self.core.recipe_presets import (
    recipe_enabled,
    resolve_self_improvement_recipe,
)
from self.core.recipe_training import PaddingAwareCausalLMDataCollator


def run_self_improvement(args: Any, task: SelfImprovementTask) -> None:
    setup = prepare_nonadaptive_run_setup(
        args,
        task,
        cuda_available_fn=torch.cuda.is_available,
        resolve_save_model_policy_fn=resolve_save_model_policy,
        recipe_enabled_fn=recipe_enabled,
        resolve_recipe_fn=resolve_self_improvement_recipe,
    )
    run_state = prepare_nonadaptive_run_state(
        args,
        reset_each_round=setup.reset_each_round,
        json_module=json,
        ensure_dir_fn=ensure_dir,
        load_summary_records_fn=load_summary_records,
    )
    paths = run_state.paths

    metadata_runtime = prepare_nonadaptive_metadata_runtime(
        seed=args.seed,
        metadata=run_state.metadata,
        metadata_path=paths.metadata_path,
        set_seed_fn=set_seed,
        random_cls=random.Random,
        decode_rng_state_fn=decode_rng_state,
        persist_metadata_fn=persist_nonadaptive_metadata,
        json_module=json,
        encode_rng_state_fn=encode_rng_state,
        sanitize_json_value_fn=sanitize_json_value,
    )

    datasets = prepare_nonadaptive_datasets(
        args,
        task,
        metadata_runtime.rng,
        run_state,
        size_schedule=setup.size_schedule,
        final_max_size=setup.final_max_size,
        composed_min_size=setup.composed_min_size,
        frontier_min_size=setup.frontier_min_size,
        reset_each_round=setup.reset_each_round,
        dynamic_composed=setup.dynamic_composed,
        persist_metadata_fn=metadata_runtime.persist_metadata,
        write_config_args_fn=lambda: write_nonadaptive_config_args(
            args,
            paths.base_output_dir,
            json_module=json,
            sanitize_json_value_fn=sanitize_json_value,
        ),
        save_examples_fn=save_examples,
        load_examples_fn=load_examples,
    )
    metadata_runtime.set_metadata(datasets.metadata)
    dataset_context = prepare_nonadaptive_dataset_context(
        task=task,
        base_splits=datasets.base_splits,
        composed_examples=datasets.composed_examples,
        eval_examples=datasets.eval_examples,
        composed_eval_examples=datasets.composed_eval_examples,
        composed_eval_component_map=datasets.composed_eval_component_map,
    )
    bootstrap = prepare_nonadaptive_bootstrap(
        args,
        task,
        base_output_dir=paths.base_output_dir,
        base_train_examples=datasets.base_splits["train"],
        eval_examples=datasets.eval_examples,
        composed_eval_examples=datasets.composed_eval_examples,
        existing_summaries=run_state.existing_summaries,
        resume_requested=run_state.resume_requested,
        reset_each_round=setup.reset_each_round,
        use_recipe=setup.use_recipe,
        recipe_name=setup.recipe_name,
        load_examples_fn=load_examples,
        instantiate_model_and_tokenizer_fn=instantiate_model_and_tokenizer,
        training_config_cls=TrainingConfig,
        resolve_max_new_tokens_fn=resolve_max_new_tokens,
        recipe_collator_cls=PaddingAwareCausalLMDataCollator,
        default_collator_cls=CausalLMDataCollator,
    )
    round_runtime = build_nonadaptive_round_runtime_bundle(
        args=args,
        task=task,
        setup=setup,
        run_state=run_state,
        metadata_runtime=metadata_runtime,
        datasets=datasets,
        dataset_context=dataset_context,
        bootstrap=bootstrap,
    )
    round_loop_result = run_nonadaptive_round_loop(
        context=round_runtime.context,
        state=round_runtime.state,
        num_rounds=args.num_expand_rounds + 1,
        run_round_fn=run_nonadaptive_round,
        round_runtime_kwargs=build_nonadaptive_round_runtime_kwargs(
            ensure_dir_fn=ensure_dir,
            save_examples_fn=save_examples,
            dataset_cls=TokenizedPromptTargetDataset,
            make_training_args_fn=make_training_args,
            build_trainer_fn=build_trainer,
            evaluate_accuracy_fn=evaluate_accuracy_with_breakdown,
            write_debug_samples_fn=write_prediction_debug_samples,
            slice_metric_cls=SliceMetric,
            round_summary_cls=RoundSummary,
            summarize_round_fn=summarize_round,
            summary_to_payload_fn=summary_to_payload,
            write_summary_records_fn=write_summary_records,
            json_module=json,
            resolve_max_new_tokens_fn=resolve_max_new_tokens,
            random_cls=random.Random,
            path_cls=Path,
            cuda_is_available_fn=torch.cuda.is_available,
            empty_cache_fn=torch.cuda.empty_cache,
            instantiate_recipe_model_fn=instantiate_recipe_model,
            load_recipe_model_fn=load_recipe_model,
            load_model_for_tokenizer_fn=load_model_for_tokenizer,
        ),
    )

    finalize_nonadaptive_run(
        keep_checkpoints=bool(args.keep_checkpoints),
        save_model_policy=setup.save_model_policy,
        round_dirs=round_loop_result.round_dirs,
        results_path=paths.results_path,
        cleanup_round_checkpoints_fn=cleanup_round_checkpoints,
    )
