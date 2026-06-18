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
from self.core.nonadaptive_bootstrap import prepare_nonadaptive_bootstrap
from self.core.nonadaptive_datasets import prepare_nonadaptive_datasets
from self.core.nonadaptive_evaluation import evaluate_nonadaptive_round
from self.core.nonadaptive_setup import prepare_nonadaptive_run_setup
from self.core.nonadaptive_state import (
    persist_nonadaptive_metadata,
    prepare_nonadaptive_run_state,
    write_nonadaptive_config_args,
)
from self.core.nonadaptive_training import train_nonadaptive_round_model
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

def run_self_improvement(args: Any, task: SelfImprovementTask) -> None:
    setup = prepare_nonadaptive_run_setup(
        args,
        task,
        cuda_available_fn=torch.cuda.is_available,
        resolve_save_model_policy_fn=resolve_save_model_policy,
        recipe_enabled_fn=recipe_enabled,
        resolve_recipe_fn=resolve_self_improvement_recipe,
    )
    stop_after_round = setup.stop_after_round
    save_model_policy = setup.save_model_policy
    frontier_min_size = setup.frontier_min_size
    recipe_name = setup.recipe_name
    use_recipe = setup.use_recipe
    recipe_preset = setup.recipe_preset
    dynamic_composed = setup.dynamic_composed
    size_schedule = setup.size_schedule
    final_max_size = setup.final_max_size
    composed_min_size = setup.composed_min_size
    reset_each_round = setup.reset_each_round

    run_state = prepare_nonadaptive_run_state(
        args,
        reset_each_round=reset_each_round,
        json_module=json,
        ensure_dir_fn=ensure_dir,
        load_summary_records_fn=load_summary_records,
    )
    paths = run_state.paths
    base_output_dir = paths.base_output_dir
    metadata_path = paths.metadata_path
    results_path = paths.results_path
    resume_requested = run_state.resume_requested
    metadata = run_state.metadata
    existing_summaries = run_state.existing_summaries

    set_seed(args.seed)
    rng = random.Random(args.seed)

    def persist_metadata(target_metadata: JsonDict | None = None) -> None:
        metadata_to_persist = metadata if target_metadata is None else target_metadata
        persist_nonadaptive_metadata(
            metadata_to_persist,
            metadata_path,
            rng.getstate(),
            json_module=json,
            encode_rng_state_fn=encode_rng_state,
            sanitize_json_value_fn=sanitize_json_value,
        )

    if metadata and "rng_state" in metadata:
        rng.setstate(decode_rng_state(metadata["rng_state"]))

    composed_pool_path = paths.composed_pool_path
    component_map_path = paths.component_map_path
    new_run = run_state.new_run

    datasets = prepare_nonadaptive_datasets(
        args,
        task,
        rng,
        run_state,
        size_schedule=size_schedule,
        final_max_size=final_max_size,
        composed_min_size=composed_min_size,
        frontier_min_size=frontier_min_size,
        reset_each_round=reset_each_round,
        dynamic_composed=dynamic_composed,
        persist_metadata_fn=persist_metadata,
        write_config_args_fn=lambda: write_nonadaptive_config_args(
            args,
            base_output_dir,
            json_module=json,
            sanitize_json_value_fn=sanitize_json_value,
        ),
        save_examples_fn=save_examples,
        load_examples_fn=load_examples,
    )
    metadata = datasets.metadata
    base_splits = datasets.base_splits
    base_records = datasets.base_records
    composed_examples = datasets.composed_examples
    component_map = datasets.component_map
    eval_examples = datasets.eval_examples
    composed_eval_examples = datasets.composed_eval_examples
    composed_eval_component_map = datasets.composed_eval_component_map

    if not base_splits["train"]:
        raise ValueError("Base training split is empty; cannot proceed.")

    print(
        "[INFO] Dataset sizes -- base train: {} | composed pool: {} | eval: {} | composed eval: {}".format(
            len(base_splits["train"]),
            len(composed_examples),
            len(eval_examples),
            len(composed_eval_examples),
        ),
        flush=True,
    )

    composed_eval_slices = task.split_composed_eval_slices(composed_eval_examples, composed_eval_component_map)
    if composed_eval_examples and composed_eval_slices:
        counts_text = " | ".join(f"{name}: {len(examples)}" for name, examples in composed_eval_slices.items())
        print(f"[INFO] Composed eval slices -- {counts_text}", flush=True)

    eval_keys = task.keys_for_examples(eval_examples)

    bootstrap = prepare_nonadaptive_bootstrap(
        args,
        task,
        base_output_dir=base_output_dir,
        base_train_examples=base_splits["train"],
        eval_examples=eval_examples,
        composed_eval_examples=composed_eval_examples,
        existing_summaries=existing_summaries,
        resume_requested=resume_requested,
        reset_each_round=reset_each_round,
        use_recipe=use_recipe,
        recipe_name=recipe_name,
        load_examples_fn=load_examples,
        instantiate_model_and_tokenizer_fn=instantiate_model_and_tokenizer,
        training_config_cls=TrainingConfig,
        resolve_max_new_tokens_fn=resolve_max_new_tokens,
        recipe_collator_cls=PaddingAwareCausalLMDataCollator,
        default_collator_cls=CausalLMDataCollator,
    )
    resume_round = bootstrap.resume_round
    model = bootstrap.model
    tokenizer = bootstrap.tokenizer
    config = bootstrap.config
    train_base_decode_tokens = bootstrap.train_base_decode_tokens
    eval_decode_tokens = bootstrap.eval_decode_tokens
    composed_eval_decode_tokens = bootstrap.composed_eval_decode_tokens
    data_collator = bootstrap.data_collator
    summary_records = bootstrap.summary_records
    pseudo_examples = bootstrap.pseudo_examples
    round_dirs: List[Path] = []

    for round_idx in range(args.num_expand_rounds + 1):
        max_size = size_schedule.round_max_size_for_index(round_idx)
        round_dir = base_output_dir / f"round_{round_idx:02d}"
        ensure_dir(round_dir)
        round_dirs.append(round_dir)
        save_model_this_round = save_model_policy == "all_rounds" or (
            save_model_policy == "final_only" and round_idx == args.num_expand_rounds
        )

        if resume_requested and round_idx < resume_round:
            print(f"[INFO] Skipping already completed round {round_idx}.", flush=True)
            continue

        train_examples = list(base_splits["train"])
        train_examples.extend(pseudo_examples)
        pseudo_used_count = len(pseudo_examples)

        save_examples(round_dir / "train_examples.jsonl", train_examples, task.serialize_example)
        save_examples(round_dir / "pseudo_examples_used.jsonl", pseudo_examples, task.serialize_example)

        round_training = train_nonadaptive_round_model(
            args=args,
            task=task,
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            round_dir=round_dir,
            config=config,
            data_collator=data_collator,
            round_idx=round_idx,
            new_run=new_run,
            save_model_this_round=save_model_this_round,
            use_recipe=use_recipe,
            recipe_name=recipe_name,
            dataset_cls=TokenizedPromptTargetDataset,
            make_training_args_fn=make_training_args,
            build_trainer_fn=build_trainer,
        )
        model = round_training.model
        trainer: Optional[Trainer] = round_training.trainer

        evaluation = evaluate_nonadaptive_round(
            model=model,
            tokenizer=tokenizer,
            task=task,
            eval_examples=eval_examples,
            composed_eval_slices=composed_eval_slices,
            composed_eval_component_map=composed_eval_component_map,
            round_dir=round_dir,
            batch_size=config.per_device_eval_batch_size,
            eval_decode_tokens=eval_decode_tokens,
            composed_eval_decode_tokens=composed_eval_decode_tokens,
            evaluate_accuracy_fn=evaluate_accuracy_with_breakdown,
            write_debug_samples_fn=write_prediction_debug_samples,
            slice_metric_cls=SliceMetric,
        )
        eval_accuracy = evaluation.eval_accuracy
        per_size_accuracy = evaluation.per_size_accuracy
        composed_eval_accuracy = evaluation.composed_eval_accuracy
        composed_slice_metrics = evaluation.composed_slice_metrics

        pseudo_generation_stats: JsonDict = {}
        if round_idx >= args.num_expand_rounds:
            pseudo_examples = []
        else:
            if dynamic_composed:
                additional_exclude = eval_keys if eval_keys else None
                if composed_min_size <= final_max_size and args.expand_train_per_size > 0:
                    refresh_label = f"round_{round_idx:02d}_next"
                    composed_build_exclude = set(eval_keys)
                    composed_build_exclude.update(task.keys_for_examples(train_examples))
                    composed_examples, component_map, _ = task.prepare_composed_train(
                        rng,
                        args,
                        base_splits={**base_splits, "train": train_examples},
                        base_records=base_records,
                        min_size=composed_min_size,
                        max_size=size_schedule.target_max_size_for_round(round_idx),
                        additional_exclude=composed_build_exclude if composed_build_exclude else None,
                    )
                    save_examples(composed_pool_path, composed_examples, task.serialize_example)
                    task.save_component_map(component_map_path, component_map)
                    metadata["last_composed_refresh"] = refresh_label
                    save_examples(round_dir / "composed_pool_for_next_round.jsonl", composed_examples, task.serialize_example)
                    task.save_component_map(round_dir / "composed_component_map_next_round.json", component_map)
                else:
                    metadata["last_composed_refresh"] = f"skipped_round_{round_idx:02d}"
            persist_metadata()

            target_max_size = size_schedule.target_max_size_for_round(round_idx)
            pseudo_rng = random.Random(rng.random())
            pseudo_decode_tokens = max(
                train_base_decode_tokens,
                resolve_max_new_tokens(composed_examples, config.decode_max_new_tokens),
            )
            if args.pseudo_label_mode == "none":
                next_pseudo_examples = []
                missing_labels = 0
                pseudo_generation_stats = {
                    "mode": "none",
                    "target_max_size": int(target_max_size),
                    "candidate_total": 0,
                    "retained_total": 0,
                    "missing_total": 0,
                }
            else:
                next_pseudo_examples, missing_labels, pseudo_generation_stats = task.derive_round_targets(
                    model,
                    tokenizer,
                    composed_examples,
                    component_map,
                    target_max_size=target_max_size,
                    base_examples=train_examples,
                    batch_size=config.per_device_eval_batch_size,
                    decode_max_new_tokens=pseudo_decode_tokens,
                    args=args,
                    rng=pseudo_rng,
                )
            if hasattr(args, "bit_composition_path_mode") and isinstance(pseudo_generation_stats, dict):
                pseudo_generation_stats.setdefault("bit_composition_path_mode", str(args.bit_composition_path_mode))
            save_examples(round_dir / "pseudo_for_next_round.jsonl", next_pseudo_examples, task.serialize_example)
            pseudo_examples = next_pseudo_examples
            if missing_labels > 0:
                print(
                    f"[WARN] Round {round_idx}: skipped {missing_labels} composed examples without pseudo labels.",
                    flush=True,
                )
            if not pseudo_examples:
                print(
                    "[WARN] No pseudo-labeled examples generated; subsequent rounds will have no additional data.",
                    flush=True,
                )

        summary = RoundSummary(
            index=round_idx,
            max_size=max_size,
            train_example_count=len(train_examples),
            pseudo_example_count=pseudo_used_count,
            eval_accuracy=eval_accuracy,
            per_size_accuracy=per_size_accuracy,
            output_dir=round_dir,
            composed_eval_accuracy=composed_eval_accuracy,
            composed_eval_slices=composed_slice_metrics,
            pseudo_generation_stats=pseudo_generation_stats,
        )
        summarize_round(summary, task)

        metrics_payload = summary_to_payload(summary, task)
        metrics_payload["save_model_policy"] = save_model_policy
        metrics_payload["model_dir"] = str(round_dir) if save_model_this_round else None
        with (round_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)

        summary_records[round_idx] = metrics_payload
        write_summary_records(summary_records, results_path)

        if stop_after_round is not None and round_idx >= stop_after_round:
            print(f"[INFO] Stop-after-round reached at round {round_idx}; exiting.", flush=True)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        if round_idx >= args.num_expand_rounds:
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if reset_each_round:
            del trainer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if use_recipe:
                if getattr(args, "init_from_scratch", False):
                    model = instantiate_recipe_model(tokenizer, recipe_preset, bf16=args.bf16, fp16=args.fp16)
                else:
                    model_dir = Path(args.model_name)
                    if not model_dir.exists():
                        raise FileNotFoundError(
                            f"Recipe-backed reset-in-each-round expects a local checkpoint directory, got {args.model_name!r}."
                        )
                    model = load_recipe_model(model_dir, tokenizer, bf16=args.bf16, fp16=args.fp16)
            else:
                model = load_model_for_tokenizer(
                    args.model_name,
                    tokenizer,
                    bf16=args.bf16,
                    fp16=args.fp16,
                )
        else:
            del trainer

    if not args.keep_checkpoints and save_model_policy != "none":
        cleanup_round_checkpoints(round_dirs)

    print(f"[INFO] Saved round summaries to {results_path}", flush=True)
