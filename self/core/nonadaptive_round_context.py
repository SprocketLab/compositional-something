"""Round-loop context assembly for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from self.core.nonadaptive_bootstrap import NonAdaptiveBootstrap
from self.core.nonadaptive_dataset_context import NonAdaptiveDatasetContext
from self.core.nonadaptive_datasets import NonAdaptiveDatasets
from self.core.nonadaptive_metadata_runtime import NonAdaptiveMetadataRuntime
from self.core.nonadaptive_round_runtime import (
    NonAdaptiveRoundRuntimeContext,
    NonAdaptiveRoundRuntimeState,
)
from self.core.nonadaptive_setup import NonAdaptiveRunSetup
from self.core.nonadaptive_state import NonAdaptiveRunState


@dataclass(frozen=True)
class NonAdaptiveRoundRuntimeBundle:
    context: NonAdaptiveRoundRuntimeContext
    state: NonAdaptiveRoundRuntimeState


def build_nonadaptive_round_runtime_bundle(
    *,
    args: Any,
    task: Any,
    setup: NonAdaptiveRunSetup,
    run_state: NonAdaptiveRunState,
    metadata_runtime: NonAdaptiveMetadataRuntime,
    datasets: NonAdaptiveDatasets,
    dataset_context: NonAdaptiveDatasetContext,
    bootstrap: NonAdaptiveBootstrap,
) -> NonAdaptiveRoundRuntimeBundle:
    paths = run_state.paths
    context = NonAdaptiveRoundRuntimeContext(
        args=args,
        task=task,
        base_output_dir=paths.base_output_dir,
        base_splits=datasets.base_splits,
        base_records=datasets.base_records,
        eval_examples=datasets.eval_examples,
        composed_eval_slices=dataset_context.composed_eval_slices,
        composed_eval_component_map=datasets.composed_eval_component_map,
        composed_pool_path=paths.composed_pool_path,
        component_map_path=paths.component_map_path,
        metadata=datasets.metadata,
        eval_keys=dataset_context.eval_keys,
        size_schedule=setup.size_schedule,
        composed_min_size=setup.composed_min_size,
        final_max_size=setup.final_max_size,
        train_base_decode_tokens=bootstrap.train_base_decode_tokens,
        eval_decode_tokens=bootstrap.eval_decode_tokens,
        composed_eval_decode_tokens=bootstrap.composed_eval_decode_tokens,
        config=bootstrap.config,
        data_collator=bootstrap.data_collator,
        tokenizer=bootstrap.tokenizer,
        rng=metadata_runtime.rng,
        new_run=run_state.new_run,
        dynamic_composed=setup.dynamic_composed,
        save_model_policy=setup.save_model_policy,
        resume_requested=run_state.resume_requested,
        resume_round=bootstrap.resume_round,
        stop_after_round=setup.stop_after_round,
        reset_each_round=setup.reset_each_round,
        use_recipe=setup.use_recipe,
        recipe_name=setup.recipe_name,
        recipe_preset=setup.recipe_preset,
        summary_records=bootstrap.summary_records,
        results_path=paths.results_path,
        persist_metadata_fn=metadata_runtime.persist_metadata,
    )
    state = NonAdaptiveRoundRuntimeState(
        model=bootstrap.model,
        composed_examples=datasets.composed_examples,
        component_map=datasets.component_map,
        pseudo_examples=bootstrap.pseudo_examples,
    )
    return NonAdaptiveRoundRuntimeBundle(context=context, state=state)


def build_nonadaptive_round_runtime_kwargs(
    *,
    ensure_dir_fn: Any,
    save_examples_fn: Any,
    dataset_cls: Any,
    make_training_args_fn: Any,
    build_trainer_fn: Any,
    evaluate_accuracy_fn: Any,
    write_debug_samples_fn: Any,
    slice_metric_cls: Any,
    round_summary_cls: Any,
    summarize_round_fn: Any,
    summary_to_payload_fn: Any,
    write_summary_records_fn: Any,
    json_module: Any,
    resolve_max_new_tokens_fn: Any,
    random_cls: Any,
    path_cls: Any,
    cuda_is_available_fn: Any,
    empty_cache_fn: Any,
    instantiate_recipe_model_fn: Any,
    load_recipe_model_fn: Any,
    load_model_for_tokenizer_fn: Any,
) -> Mapping[str, Any]:
    return {
        "ensure_dir_fn": ensure_dir_fn,
        "save_examples_fn": save_examples_fn,
        "dataset_cls": dataset_cls,
        "make_training_args_fn": make_training_args_fn,
        "build_trainer_fn": build_trainer_fn,
        "evaluate_accuracy_fn": evaluate_accuracy_fn,
        "write_debug_samples_fn": write_debug_samples_fn,
        "slice_metric_cls": slice_metric_cls,
        "round_summary_cls": round_summary_cls,
        "summarize_round_fn": summarize_round_fn,
        "summary_to_payload_fn": summary_to_payload_fn,
        "write_summary_records_fn": write_summary_records_fn,
        "json_module": json_module,
        "resolve_max_new_tokens_fn": resolve_max_new_tokens_fn,
        "random_cls": random_cls,
        "path_cls": path_cls,
        "cuda_is_available_fn": cuda_is_available_fn,
        "empty_cache_fn": empty_cache_fn,
        "instantiate_recipe_model_fn": instantiate_recipe_model_fn,
        "load_recipe_model_fn": load_recipe_model_fn,
        "load_model_for_tokenizer_fn": load_model_for_tokenizer_fn,
    }
