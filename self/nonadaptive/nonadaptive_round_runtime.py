"""Single-round orchestration for the non-adaptive self-improvement loop."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional

from self.core.data_io import JsonDict, ensure_dir, save_examples, write_summary_records
from self.core.evaluation import (
    evaluate_accuracy_with_breakdown,
    resolve_max_new_tokens,
    write_prediction_debug_samples,
)
from self.core.model_io import load_model_for_tokenizer
from self.nonadaptive.nonadaptive_evaluation import evaluate_nonadaptive_round
from self.nonadaptive.nonadaptive_lifecycle import NonAdaptiveRoundResources, finish_nonadaptive_round
from self.nonadaptive.nonadaptive_pseudo import prepare_nonadaptive_next_pseudo_round
from self.nonadaptive.nonadaptive_results import record_nonadaptive_round_summary
from self.nonadaptive.nonadaptive_round_models import (
    NonAdaptiveRoundRuntimeContext,
    NonAdaptiveRoundRuntimeResult,
    NonAdaptiveRoundRuntimeState,
)
from self.nonadaptive.nonadaptive_round_setup import (
    prepare_nonadaptive_round_plan,
    prepare_nonadaptive_round_training_data,
)
from self.nonadaptive.nonadaptive_training import train_nonadaptive_round_model
from self.core.recipes import instantiate_recipe_model, load_recipe_model
from self.core.summaries import RoundSummary, SliceMetric, summarize_round, summary_to_payload
from self.core.training import TokenizedPromptTargetDataset, build_trainer, make_training_args


def run_nonadaptive_round(
    *,
    context: NonAdaptiveRoundRuntimeContext,
    state: NonAdaptiveRoundRuntimeState,
    round_idx: int,
    ensure_dir_fn: Callable[[Path], Path | None] = ensure_dir,
    save_examples_fn: Callable[..., None] = save_examples,
    train_round_model_fn: Callable[..., Any] = train_nonadaptive_round_model,
    evaluate_round_fn: Callable[..., Any] = evaluate_nonadaptive_round,
    prepare_next_pseudo_round_fn: Callable[..., Any] = prepare_nonadaptive_next_pseudo_round,
    record_round_summary_fn: Callable[..., Any] = record_nonadaptive_round_summary,
    finish_round_fn: Callable[..., Any] = finish_nonadaptive_round,
    resources_cls: Callable[..., Any] = NonAdaptiveRoundResources,
    dataset_cls: Callable[..., Any] = TokenizedPromptTargetDataset,
    make_training_args_fn: Callable[..., Any] = make_training_args,
    build_trainer_fn: Callable[..., Any] = build_trainer,
    evaluate_accuracy_fn: Callable[..., Any] = evaluate_accuracy_with_breakdown,
    write_debug_samples_fn: Callable[..., None] = write_prediction_debug_samples,
    slice_metric_cls: Callable[..., Any] = SliceMetric,
    round_summary_cls: Callable[..., Any] = RoundSummary,
    summarize_round_fn: Callable[..., None] = summarize_round,
    summary_to_payload_fn: Callable[..., JsonDict] = summary_to_payload,
    write_summary_records_fn: Callable[..., None] = write_summary_records,
    json_module: Any = None,
    resolve_max_new_tokens_fn: Callable[..., int] = resolve_max_new_tokens,
    random_cls: Callable[[float], Any] = random.Random,
    path_cls: Callable[[Any], Path] = Path,
    cuda_is_available_fn: Callable[[], bool],
    empty_cache_fn: Callable[[], None],
    instantiate_recipe_model_fn: Callable[..., Any] = instantiate_recipe_model,
    load_recipe_model_fn: Callable[..., Any] = load_recipe_model,
    load_model_for_tokenizer_fn: Callable[..., Any] = load_model_for_tokenizer,
    print_fn: Callable[..., None] = print,
) -> NonAdaptiveRoundRuntimeResult:
    """Run one non-adaptive training/evaluation/pseudo-labeling round."""
    round_plan = prepare_nonadaptive_round_plan(
        base_output_dir=context.base_output_dir,
        round_idx=round_idx,
        size_schedule=context.size_schedule,
        save_model_policy=context.save_model_policy,
        num_expand_rounds=context.args.num_expand_rounds,
        resume_requested=context.resume_requested,
        resume_round=context.resume_round,
        ensure_dir_fn=ensure_dir_fn,
    )
    round_dir = round_plan.round_dir
    if round_plan.should_skip_completed_round:
        print_fn(f"[INFO] Skipping already completed round {round_idx}.", flush=True)
        return NonAdaptiveRoundRuntimeResult(round_dir=round_dir, skipped=True, should_break=False)

    round_training_data = prepare_nonadaptive_round_training_data(
        round_dir=round_dir,
        base_train_examples=context.base_splits["train"],
        pseudo_examples=state.pseudo_examples,
        task=context.task,
        save_examples_fn=save_examples_fn,
    )

    round_training = train_round_model_fn(
        args=context.args,
        task=context.task,
        model=state.model,
        tokenizer=context.tokenizer,
        train_examples=round_training_data.train_examples,
        round_dir=round_dir,
        config=context.config,
        data_collator=context.data_collator,
        round_idx=round_idx,
        new_run=context.new_run,
        save_model_this_round=round_plan.save_model_this_round,
        use_recipe=context.use_recipe,
        recipe_name=context.recipe_name,
        dataset_cls=dataset_cls,
        make_training_args_fn=make_training_args_fn,
        build_trainer_fn=build_trainer_fn,
    )
    state.model = round_training.model
    trainer: Optional[Any] = round_training.trainer

    evaluation = evaluate_round_fn(
        model=state.model,
        tokenizer=context.tokenizer,
        task=context.task,
        eval_examples=context.eval_examples,
        composed_eval_slices=context.composed_eval_slices,
        composed_eval_component_map=context.composed_eval_component_map,
        round_dir=round_dir,
        batch_size=context.config.per_device_eval_batch_size,
        eval_decode_tokens=context.eval_decode_tokens,
        composed_eval_decode_tokens=context.composed_eval_decode_tokens,
        evaluate_accuracy_fn=evaluate_accuracy_fn,
        write_debug_samples_fn=write_debug_samples_fn,
        slice_metric_cls=slice_metric_cls,
    )

    next_pseudo_round = prepare_next_pseudo_round_fn(
        args=context.args,
        task=context.task,
        model=state.model,
        tokenizer=context.tokenizer,
        rng=context.rng,
        round_idx=round_idx,
        round_dir=round_dir,
        train_examples=round_training_data.train_examples,
        base_splits=context.base_splits,
        base_records=context.base_records,
        composed_examples=state.composed_examples,
        component_map=state.component_map,
        composed_pool_path=context.composed_pool_path,
        component_map_path=context.component_map_path,
        metadata=context.metadata,
        eval_keys=context.eval_keys,
        size_schedule=context.size_schedule,
        composed_min_size=context.composed_min_size,
        final_max_size=context.final_max_size,
        train_base_decode_tokens=context.train_base_decode_tokens,
        config_decode_max_new_tokens=context.config.decode_max_new_tokens,
        eval_batch_size=context.config.per_device_eval_batch_size,
        dynamic_composed=context.dynamic_composed,
        persist_metadata_fn=context.persist_metadata_fn,
        save_examples_fn=save_examples_fn,
        resolve_max_new_tokens_fn=resolve_max_new_tokens_fn,
        random_cls=random_cls,
    )
    state.composed_examples = next_pseudo_round.composed_examples
    state.component_map = next_pseudo_round.component_map
    state.pseudo_examples = next_pseudo_round.pseudo_examples

    record_round_summary_fn(
        round_idx=round_idx,
        max_size=round_plan.max_size,
        train_example_count=len(round_training_data.train_examples),
        pseudo_used_count=round_training_data.pseudo_used_count,
        evaluation=evaluation,
        pseudo_generation_stats=next_pseudo_round.pseudo_generation_stats,
        round_dir=round_dir,
        save_model_policy=context.save_model_policy,
        save_model_this_round=round_plan.save_model_this_round,
        summary_records=context.summary_records,
        results_path=context.results_path,
        task=context.task,
        round_summary_cls=round_summary_cls,
        summarize_round_fn=summarize_round_fn,
        summary_to_payload_fn=summary_to_payload_fn,
        write_summary_records_fn=write_summary_records_fn,
        json_module=json_module,
    )

    round_resources = resources_cls(model=state.model, trainer=trainer)
    state.model = None
    trainer = None
    del trainer
    post_round_action = finish_round_fn(
        args=context.args,
        tokenizer=context.tokenizer,
        resources=round_resources,
        round_idx=round_idx,
        stop_after_round=context.stop_after_round,
        reset_each_round=context.reset_each_round,
        use_recipe=context.use_recipe,
        recipe_preset=context.recipe_preset,
        path_cls=path_cls,
        cuda_is_available_fn=cuda_is_available_fn,
        empty_cache_fn=empty_cache_fn,
        instantiate_recipe_model_fn=instantiate_recipe_model_fn,
        load_recipe_model_fn=load_recipe_model_fn,
        load_model_for_tokenizer_fn=load_model_for_tokenizer_fn,
    )
    state.model = round_resources.model

    return NonAdaptiveRoundRuntimeResult(
        round_dir=round_dir,
        skipped=False,
        should_break=bool(post_round_action.should_break),
    )
