#!/usr/bin/env python3
"""Non-adaptive iterative self-improvement loop runtime."""

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
from self.core.nonadaptive_datasets import prepare_nonadaptive_datasets
from self.core.nonadaptive_setup import prepare_nonadaptive_run_setup
from self.core.nonadaptive_state import (
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

    resume_round = 0
    if resume_requested:
        if args.resume_from_round is not None:
            resume_round = args.resume_from_round
        elif existing_summaries:
            resume_round = max(existing_summaries) + 1
        if resume_round > args.num_expand_rounds:
            print(
                f"[INFO] Requested resume round {resume_round} exceeds configured num_expand_rounds={args.num_expand_rounds}; "
                "no additional training will be performed.",
                flush=True,
            )
        for round_idx in list(existing_summaries.keys()):
            if round_idx >= resume_round:
                existing_summaries.pop(round_idx, None)
        if resume_round > 0 and not reset_each_round:
            checkpoint_dir = base_output_dir / f"round_{resume_round-1:02d}"
            if not checkpoint_dir.exists():
                raise ValueError(
                    f"Cannot resume from round {resume_round}; checkpoint directory {checkpoint_dir} is missing."
                )
            model_name_or_path = str(checkpoint_dir)
        else:
            model_name_or_path = args.model_name
        print(f"[INFO] Resuming training from round {resume_round}.", flush=True)
    else:
        model_name_or_path = args.model_name

    token_initializers = task.token_initializers(args) if hasattr(task, "token_initializers") else {}
    model, tokenizer = instantiate_model_and_tokenizer(
        model_name_or_path,
        bf16=args.bf16,
        fp16=args.fp16,
        token_initializers=token_initializers,
        init_from_scratch=getattr(args, "init_from_scratch", False),
        tokenizer_mode=str(getattr(args, "tokenizer_mode", "auto")),
        recipe=recipe_name,
    )

    config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )

    train_base_decode_tokens = resolve_max_new_tokens(base_splits["train"], config.decode_max_new_tokens)
    eval_decode_tokens = resolve_max_new_tokens(eval_examples, config.decode_max_new_tokens)
    composed_eval_decode_tokens = resolve_max_new_tokens(composed_eval_examples, config.decode_max_new_tokens)

    if use_recipe:
        data_collator = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")
    else:
        data_collator = CausalLMDataCollator(tokenizer)
    summary_records = dict(existing_summaries)
    pseudo_examples: List[Any] = []
    round_dirs: List[Path] = []

    if resume_round > 0:
        prev_round_dir = base_output_dir / f"round_{resume_round-1:02d}"
        pseudo_seed_path = prev_round_dir / "pseudo_for_next_round.jsonl"
        if not pseudo_seed_path.exists():
            raise RuntimeError(
                f"Pseudo dataset for round {resume_round} is missing (expected {pseudo_seed_path}). "
                "Please rerun the previous round to regenerate the pseudo labels before resuming."
            )
        pseudo_examples = load_examples(pseudo_seed_path, task.deserialize_example)
        print(
            f"[INFO] Loaded {len(pseudo_examples)} pseudo examples for upcoming round {resume_round} "
            f"from {pseudo_seed_path}.",
            flush=True,
        )

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

        recipe_phase_name = "seed" if use_recipe and round_idx == 0 else "self_improve"
        skip_round_training = bool(getattr(args, "treat_seed_as_round_zero", False) and new_run and round_idx == 0)
        trainer: Optional[Trainer] = None
        recipe_phase_name = (
            "seed"
            if use_recipe and round_idx == 0 and not getattr(args, "treat_seed_as_round_zero", False)
            else "self_improve"
        )
        recipe_phase_overrides: Optional[Dict[str, object]] = None
        if use_recipe and recipe_phase_name == "self_improve":
            overrides: Dict[str, object] = {}
            lr_override = getattr(args, "self_improve_learning_rate", None)
            lr_switch_round = getattr(args, "self_improve_lr_switch_round", None)
            lr_after_switch = getattr(args, "self_improve_learning_rate_after_switch", None)
            if (
                lr_switch_round is not None
                and lr_after_switch is not None
                and round_idx >= int(lr_switch_round)
            ):
                lr_override = lr_after_switch
            if lr_override is not None:
                overrides["learning_rate"] = float(lr_override)
            warmup_override = getattr(args, "self_improve_warmup_steps", None)
            if warmup_override is not None:
                overrides["warmup_steps"] = int(warmup_override)
            if overrides:
                recipe_phase_overrides = overrides

        if skip_round_training:
            print(
                "[INFO] Treating seed checkpoint as completed round_00; skipping round-0 training.",
                flush=True,
            )
            if save_model_this_round:
                model.save_pretrained(round_dir)
                tokenizer.save_pretrained(round_dir)
        else:
            train_dataset = TokenizedPromptTargetDataset(train_examples, tokenizer)
            training_args = make_training_args(
                round_dir,
                config,
                bf16=args.bf16,
                fp16=args.fp16,
                skip_save=not bool(getattr(args, "keep_checkpoints", False)),
                keep_checkpoints=bool(getattr(args, "keep_checkpoints", False)),
                seed=args.seed,
                recipe=recipe_name,
                recipe_phase_name=recipe_phase_name,
                recipe_phase_overrides=recipe_phase_overrides,
            )
            trainer = build_trainer(
                model=model,
                training_args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                seed=args.seed + round_idx * 9973,
                size_getter=task.size_of,
                bucket_train_batches_by_size=bool(
                    getattr(args, "bucket_train_batches_by_size", getattr(args, "bucket_train_batches_by_bits", False))
                ),
                recipe=recipe_name,
                recipe_phase_name=recipe_phase_name,
                recipe_phase_overrides=recipe_phase_overrides,
            )
            trainer.train()
            model = trainer.model
            if save_model_this_round:
                if use_recipe:
                    trainer.save_model(str(round_dir))
                else:
                    trainer.save_model()
                tokenizer.save_pretrained(round_dir)

        eval_accuracy, per_size_accuracy = evaluate_accuracy_with_breakdown(
            model=model,
            tokenizer=tokenizer,
            examples=eval_examples,
            batch_size=config.per_device_eval_batch_size,
            max_new_tokens=eval_decode_tokens,
            size_getter=task.size_of,
            prediction_parser=task.prediction_parser,
        )

        composed_slice_metrics: Dict[str, SliceMetric] = {}
        composed_correct_total = 0.0
        composed_count_total = 0
        for slice_name, slice_examples in composed_eval_slices.items():
            if slice_examples:
                slice_accuracy, slice_per_size_accuracy = evaluate_accuracy_with_breakdown(
                    model=model,
                    tokenizer=tokenizer,
                    examples=slice_examples,
                    batch_size=config.per_device_eval_batch_size,
                    max_new_tokens=composed_eval_decode_tokens,
                    size_getter=task.size_of,
                    prediction_parser=task.prediction_parser,
                )
            else:
                slice_accuracy = math.nan
                slice_per_size_accuracy = {}
            composed_slice_metrics[slice_name] = SliceMetric(
                accuracy=slice_accuracy,
                count=len(slice_examples),
                per_size_accuracy=slice_per_size_accuracy,
            )
            if slice_name in {"accepted_by_guard", "rejected_by_guard"} and slice_examples:
                write_prediction_debug_samples(
                    round_dir / f"composed_eval_{slice_name}_debug.jsonl",
                    model=model,
                    tokenizer=tokenizer,
                    examples=slice_examples,
                    batch_size=config.per_device_eval_batch_size,
                    max_new_tokens=composed_eval_decode_tokens,
                    size_getter=task.size_of,
                    key_getter=task.key_for_example,
                    component_map=composed_eval_component_map,
                    prediction_parser=task.prediction_parser,
                )
            if slice_examples and not math.isnan(slice_accuracy):
                composed_correct_total += slice_accuracy * len(slice_examples)
                composed_count_total += len(slice_examples)
        composed_eval_accuracy = (
            composed_correct_total / composed_count_total if composed_count_total > 0 else math.nan
        )

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
