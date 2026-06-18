"""Post-round lifecycle handling for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from self.core.model_io import load_model_for_tokenizer
from self.core.recipes import instantiate_recipe_model, load_recipe_model


@dataclass
class NonAdaptiveRoundResources:
    model: Any
    trainer: Any


@dataclass(frozen=True)
class NonAdaptivePostRoundAction:
    should_break: bool = False
    should_continue: bool = False


def _release_trainer(resources: NonAdaptiveRoundResources) -> None:
    trainer = resources.trainer
    resources.trainer = None
    del trainer


def _release_model(resources: NonAdaptiveRoundResources) -> None:
    model = resources.model
    resources.model = None
    del model


def _clear_cuda_cache_if_available(
    *,
    cuda_is_available_fn: Callable[[], bool],
    empty_cache_fn: Callable[[], None],
) -> None:
    if cuda_is_available_fn():
        empty_cache_fn()


def finish_nonadaptive_round(
    *,
    args: Any,
    tokenizer: Any,
    resources: NonAdaptiveRoundResources,
    round_idx: int,
    stop_after_round: int | None,
    reset_each_round: bool,
    use_recipe: bool,
    recipe_preset: Any,
    path_cls: Callable[[Any], Path] = Path,
    cuda_is_available_fn: Callable[[], bool],
    empty_cache_fn: Callable[[], None],
    instantiate_recipe_model_fn: Callable[..., Any] = instantiate_recipe_model,
    load_recipe_model_fn: Callable[..., Any] = load_recipe_model,
    load_model_for_tokenizer_fn: Callable[..., Any] = load_model_for_tokenizer,
) -> NonAdaptivePostRoundAction:
    if stop_after_round is not None and round_idx >= stop_after_round:
        print(f"[INFO] Stop-after-round reached at round {round_idx}; exiting.", flush=True)
        _release_trainer(resources)
        _clear_cuda_cache_if_available(
            cuda_is_available_fn=cuda_is_available_fn,
            empty_cache_fn=empty_cache_fn,
        )
        return NonAdaptivePostRoundAction(should_break=True)

    if round_idx >= args.num_expand_rounds:
        _release_trainer(resources)
        _clear_cuda_cache_if_available(
            cuda_is_available_fn=cuda_is_available_fn,
            empty_cache_fn=empty_cache_fn,
        )
        return NonAdaptivePostRoundAction(should_continue=True)

    if reset_each_round:
        _release_trainer(resources)
        _release_model(resources)
        _clear_cuda_cache_if_available(
            cuda_is_available_fn=cuda_is_available_fn,
            empty_cache_fn=empty_cache_fn,
        )
        if use_recipe:
            if getattr(args, "init_from_scratch", False):
                resources.model = instantiate_recipe_model_fn(
                    tokenizer,
                    recipe_preset,
                    bf16=args.bf16,
                    fp16=args.fp16,
                )
            else:
                model_dir = path_cls(args.model_name)
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Recipe-backed reset-in-each-round expects a local checkpoint directory, got {args.model_name!r}."
                    )
                resources.model = load_recipe_model_fn(model_dir, tokenizer, bf16=args.bf16, fp16=args.fp16)
        else:
            resources.model = load_model_for_tokenizer_fn(
                args.model_name,
                tokenizer,
                bf16=args.bf16,
                fp16=args.fp16,
            )
    else:
        _release_trainer(resources)

    return NonAdaptivePostRoundAction()
