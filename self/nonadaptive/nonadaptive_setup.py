"""Preflight and derived setup values for the non-adaptive loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from self.core.data_io import resolve_save_model_policy
from self.nonadaptive.nonadaptive_schedule import (
    NonAdaptiveSizeSchedule,
    build_nonadaptive_size_schedule,
    normalize_frontier_min_size,
)
from self.core.recipes import (
    SelfImprovementRecipePreset,
    recipe_enabled,
    resolve_self_improvement_recipe,
)


@dataclass(frozen=True)
class NonAdaptiveRunSetup:
    stop_after_round: Optional[int]
    save_model_policy: str
    frontier_min_size: Optional[int]
    recipe_name: str
    use_recipe: bool
    recipe_preset: Optional[SelfImprovementRecipePreset]
    dynamic_composed: bool
    size_schedule: NonAdaptiveSizeSchedule
    final_max_size: int
    composed_min_size: int
    reset_each_round: bool


def prepare_nonadaptive_run_setup(
    args: Any,
    task: Any,
    *,
    cuda_available_fn: Callable[[], bool],
    resolve_save_model_policy_fn: Callable[[Any], str] = resolve_save_model_policy,
    recipe_enabled_fn: Callable[[str], bool] = recipe_enabled,
    resolve_recipe_fn: Callable[[str], SelfImprovementRecipePreset] = resolve_self_improvement_recipe,
) -> NonAdaptiveRunSetup:
    """Validate args, apply legacy defaults, and derive loop setup values."""

    if not args.bf16 and not args.fp16 and cuda_available_fn():
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

    save_model_policy = resolve_save_model_policy_fn(args)
    args.skip_save_model = save_model_policy == "none"
    frontier_min_size = normalize_frontier_min_size(args)
    task.validate_args(args)

    recipe_name = str(getattr(args, "recipe", "none"))
    use_recipe = recipe_enabled_fn(recipe_name)
    recipe_preset = resolve_recipe_fn(recipe_name) if use_recipe else None
    if use_recipe and getattr(args, "tokenizer_mode", "auto") != "auto":
        print(
            "[INFO] Recipe-backed run-length bit-string path ignores --tokenizer-mode "
            "and uses the recipe tokenizer.",
            flush=True,
        )
    if use_recipe and not args.bf16 and not args.fp16 and recipe_preset is not None:
        args.bf16 = bool(recipe_preset.bf16)
    if use_recipe and recipe_preset is not None:
        if args.per_device_train_batch_size == 4:
            args.per_device_train_batch_size = recipe_preset.per_device_train_batch_size
        if args.per_device_eval_batch_size == 4:
            args.per_device_eval_batch_size = recipe_preset.per_device_eval_batch_size

    dynamic_composed = args.composed_refresh_mode == "dynamic"
    size_schedule = build_nonadaptive_size_schedule(args, frontier_min_size)
    return NonAdaptiveRunSetup(
        stop_after_round=stop_after_round,
        save_model_policy=save_model_policy,
        frontier_min_size=frontier_min_size,
        recipe_name=recipe_name,
        use_recipe=use_recipe,
        recipe_preset=recipe_preset,
        dynamic_composed=dynamic_composed,
        size_schedule=size_schedule,
        final_max_size=size_schedule.final_max_size,
        composed_min_size=size_schedule.composed_min_size,
        reset_each_round=args.reset_in_each_round,
    )
