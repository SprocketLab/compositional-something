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
from self.core.nonadaptive_schedule import build_nonadaptive_size_schedule, normalize_frontier_min_size
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
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] No precision flag provided; defaulting to bf16 on CUDA.", flush=True)
    if args.initial_min_size < 1:
        raise ValueError("initial_min_size must be at least 1.")
    if args.initial_max_size < args.initial_min_size:
        raise ValueError("initial_max_size must be >= initial_min_size.")
    if args.eval_per_size < 0:
        raise ValueError("eval_per_size must be non-negative.")
    if args.composed_eval_per_size < 0:
        raise ValueError("composed_eval_per_size must be non-negative.")
    if args.expand_num_size < 1 and args.num_expand_rounds > 0:
        raise ValueError("expand_num_size must be positive when num_expand_rounds > 0.")
    if args.num_expand_rounds < 0:
        raise ValueError("num_expand_rounds cannot be negative.")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")
    if args.resume_from_round is not None and args.resume_from_round < 0:
        raise ValueError("resume_from_round must be non-negative if provided.")
    stop_after_round = getattr(args, "stop_after_round", None)
    if stop_after_round is not None:
        if stop_after_round < 0:
            raise ValueError("stop_after_round must be non-negative if provided.")
        if args.resume_from_round is not None and stop_after_round < args.resume_from_round:
            raise ValueError("stop_after_round must be greater than or equal to resume_from_round.")
    save_model_policy = resolve_save_model_policy(args)
    args.skip_save_model = save_model_policy == "none"
    frontier_min_size = normalize_frontier_min_size(args)
    task.validate_args(args)

    recipe_name = str(getattr(args, "recipe", "none"))
    use_recipe = recipe_enabled(recipe_name)
    recipe_preset = resolve_self_improvement_recipe(recipe_name) if use_recipe else None
    if use_recipe and getattr(args, "tokenizer_mode", "auto") != "auto":
        print("[INFO] Recipe-backed bit-task path ignores --tokenizer-mode and uses the recipe tokenizer.", flush=True)
    if use_recipe and not args.bf16 and not args.fp16 and recipe_preset is not None:
        args.bf16 = bool(recipe_preset.bf16)
    if use_recipe and recipe_preset is not None:
        if args.per_device_train_batch_size == 4:
            args.per_device_train_batch_size = recipe_preset.per_device_train_batch_size
        if args.per_device_eval_batch_size == 4:
            args.per_device_eval_batch_size = recipe_preset.per_device_eval_batch_size

    dynamic_composed = args.composed_refresh_mode == "dynamic"
    size_schedule = build_nonadaptive_size_schedule(args, frontier_min_size)
    final_max_size = size_schedule.final_max_size
    composed_min_size = size_schedule.composed_min_size
    reset_each_round = args.reset_in_each_round

    original_output_dir = Path(args.output_dir)
    if reset_each_round:
        base_output_dir = original_output_dir / "reset_each_round"
        ensure_dir(original_output_dir)
        print(
            f"[INFO] reset_in_each_round enabled; writing artifacts to {base_output_dir}",
            flush=True,
        )
    else:
        base_output_dir = original_output_dir
    ensure_dir(base_output_dir)

    data_dir = base_output_dir / "data"
    ensure_dir(data_dir)
    metadata_path = data_dir / "metadata.json"
    results_path = base_output_dir / "self_improvement_results.json"
    resume_requested = args.resume or args.resume_from_round is not None

    metadata: JsonDict = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    existing_summaries = load_summary_records(results_path) if resume_requested else {}

    set_seed(args.seed)
    rng = random.Random(args.seed)

    def persist_metadata() -> None:
        metadata["rng_state"] = encode_rng_state(rng.getstate())
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(sanitize_json_value(metadata), handle, indent=2)

    if metadata and "rng_state" in metadata:
        rng.setstate(decode_rng_state(metadata["rng_state"]))

    base_train_path = data_dir / "initial_train.jsonl"
    base_val_path = data_dir / "initial_validation.jsonl"
    base_test_path = data_dir / "initial_test.jsonl"
    composed_pool_path = data_dir / "composed_pool.jsonl"
    component_map_path = data_dir / "composed_component_map.json"
    eval_path = data_dir / "evaluation.jsonl"
    composed_eval_path = data_dir / "composed_evaluation.jsonl"
    composed_eval_component_map_path = data_dir / "composed_evaluation_component_map.json"

    def stored_value(*keys: str) -> Any:
        for key in keys:
            if key in metadata:
                return metadata[key]
        return None

    new_run = not resume_requested or not base_train_path.exists()

    if new_run:
        print(f"[INFO] Generating {task.name} datasets from scratch.", flush=True)
        reserved_eval_examples: List[Any] = []
        reserved_eval_keys: set[Any] = set()
        if getattr(args, "reserve_shared_eval_first", False) and args.eval_per_size > 0:
            reserved_eval_examples = task.prepare_eval_examples(
                rng,
                args,
                min_size=args.initial_min_size,
                max_size=final_max_size,
                exclude=set(),
            )
            reserved_eval_keys = task.keys_for_examples(reserved_eval_examples)
            setattr(args, "_initial_exclude_keys", reserved_eval_keys)
            print(
                f"[INFO] Reserved {len(reserved_eval_examples)} shared evaluation examples before dataset construction.",
                flush=True,
            )
        else:
            setattr(args, "_initial_exclude_keys", None)

        base_splits, base_records = task.prepare_initial_splits(rng, args)
        save_examples(base_train_path, base_splits["train"], task.serialize_example)
        save_examples(base_val_path, base_splits["validation"], task.serialize_example)
        save_examples(base_test_path, base_splits["test"], task.serialize_example)

        initial_train_examples = list(base_splits["train"])
        initial_dynamic_exclude = set(reserved_eval_keys)
        initial_dynamic_exclude.update(task.keys_for_examples(initial_train_examples))

        initial_composed_max_size = size_schedule.target_max_size_for_round(0)
        composed_examples, component_map, composed_keys = task.prepare_composed_train(
            rng,
            args,
            base_splits={**base_splits, "train": initial_train_examples},
            base_records=base_records,
            min_size=composed_min_size,
            max_size=initial_composed_max_size,
            additional_exclude=initial_dynamic_exclude if initial_dynamic_exclude else None,
        )
        save_examples(composed_pool_path, composed_examples, task.serialize_example)
        task.save_component_map(component_map_path, component_map)

        composed_eval_exclude = set(reserved_eval_keys)
        composed_eval_exclude.update(composed_keys)
        composed_eval_examples, composed_eval_component_map, composed_eval_keys = task.prepare_composed_eval(
            rng,
            args,
            base_splits=base_splits,
            base_records=base_records,
            min_size=composed_min_size,
            max_size=final_max_size,
            additional_exclude=composed_eval_exclude if composed_eval_exclude else None,
        )
        save_examples(composed_eval_path, composed_eval_examples, task.serialize_example)
        task.save_component_map(composed_eval_component_map_path, composed_eval_component_map)

        if reserved_eval_examples:
            eval_examples = reserved_eval_examples
        else:
            training_union = set().union(*base_records.values())
            training_union.update(composed_keys)
            training_union.update(composed_eval_keys)
            eval_examples = task.prepare_eval_examples(
                rng,
                args,
                min_size=args.initial_min_size,
                max_size=final_max_size,
                exclude=training_union,
            )
        save_examples(eval_path, eval_examples, task.serialize_example)

        metadata = {
            "task": task.name,
            "size_label": task.size_label,
            "initial_min_size": args.initial_min_size,
            "initial_max_size": args.initial_max_size,
            "frontier_min_size": frontier_min_size,
            "expand_num_size": args.expand_num_size,
            "expand_train_per_size": args.expand_train_per_size,
            "eval_per_size": args.eval_per_size,
            "composed_eval_per_size": args.composed_eval_per_size,
            "composed_max_size": final_max_size,
            "reset_each_round": reset_each_round,
            "composed_refresh_mode": args.composed_refresh_mode,
            "task_config": task.build_task_metadata(args, final_max_size),
        }
        metadata.update(task.metadata_aliases(args, final_max_size))
        metadata["last_composed_refresh"] = "initial_dynamic" if dynamic_composed else "static_initial"
        persist_metadata()

        with (base_output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
            json.dump(sanitize_json_value(vars(args)), handle, indent=2)
    else:
        print(f"[INFO] Loading {task.name} datasets from disk.", flush=True)
        if not metadata:
            raise ValueError("metadata.json missing; cannot resume without dataset metadata.")

        stored_task = metadata.get("task")
        if stored_task is not None and stored_task != task.name:
            raise ValueError(f"Output directory contains task={stored_task!r}, but current run requests {task.name!r}.")

        stored_initial_min = stored_value("initial_min_size", f"initial_min_{task.size_alias_plural}")
        stored_initial_max = stored_value("initial_max_size", f"initial_max_{task.size_alias_plural}")
        stored_frontier_min = stored_value("frontier_min_size")
        stored_composed_max = stored_value("composed_max_size", f"composed_max_{task.size_alias_plural}")
        if stored_initial_min is None or stored_initial_max is None or stored_composed_max is None:
            raise ValueError("metadata.json is missing required size-range keys; please regenerate datasets.")
        if int(stored_initial_min) != args.initial_min_size:
            raise ValueError(
                f"initial_min_size mismatch (stored={stored_initial_min} requested={args.initial_min_size})."
            )
        if int(stored_initial_max) != args.initial_max_size:
            raise ValueError(
                f"initial_max_size mismatch (stored={stored_initial_max} requested={args.initial_max_size})."
            )
        normalized_stored_frontier_min = None if stored_frontier_min is None else int(stored_frontier_min)
        if normalized_stored_frontier_min != frontier_min_size:
            raise ValueError(
                f"frontier_min_size mismatch (stored={normalized_stored_frontier_min} requested={frontier_min_size})."
            )
        if final_max_size > int(stored_composed_max):
            raise ValueError(
                "Requested num_expand_rounds requires more sizes than available in stored composed data. "
                "Regenerate datasets with a larger range."
            )

        stored_reset_flag = bool(metadata.get("reset_each_round", False))
        if stored_reset_flag != reset_each_round:
            raise ValueError(
                "Output directory was created with a different reset_each_round setting. "
                "Please choose a different --output-dir to avoid mixing trajectories."
            )

        stored_refresh_mode = metadata.get("composed_refresh_mode", "static")
        if stored_refresh_mode not in ("dynamic", "static"):
            stored_refresh_mode = "dynamic" if dynamic_composed else "static"
        if stored_refresh_mode == "static" and dynamic_composed:
            raise ValueError(
                "Existing output directory was created with static composed pools but current run requests dynamic refresh."
            )
        if stored_refresh_mode == "dynamic" and not dynamic_composed:
            raise ValueError(
                "Existing output directory was created with dynamic composed pools but current run requests static refresh."
            )

        stored_composed_eval_per = stored_value(
            "composed_eval_per_size",
            f"composed_eval_per_{task.size_alias_singular}",
        )
        if stored_composed_eval_per is not None and int(stored_composed_eval_per) != args.composed_eval_per_size:
            raise ValueError(
                "composed_eval_per_size does not match stored datasets. Please regenerate datasets or use a matching value."
            )

        task.validate_loaded_metadata(args, metadata, final_max_size, dynamic_composed)

        base_splits = {
            "train": load_examples(base_train_path, task.deserialize_example),
            "validation": load_examples(base_val_path, task.deserialize_example),
            "test": load_examples(base_test_path, task.deserialize_example),
        }
        composed_examples = load_examples(composed_pool_path, task.deserialize_example)
        component_map = task.load_component_map(component_map_path)
        eval_examples = load_examples(eval_path, task.deserialize_example)
        composed_eval_examples = load_examples(composed_eval_path, task.deserialize_example)
        composed_eval_component_map = task.load_component_map(composed_eval_component_map_path)
        if not composed_eval_examples and args.composed_eval_per_size > 0:
            print(
                "[WARN] Held-out composed evaluation set is missing; composed slice metrics will be unavailable "
                "for this run. Regenerate datasets to enable them.",
                flush=True,
            )
        base_records = task.rebuild_records(base_splits)

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
