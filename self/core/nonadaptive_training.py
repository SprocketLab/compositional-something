"""Per-round training setup/execution for the non-adaptive loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from self.core.training import TokenizedPromptTargetDataset, build_trainer, make_training_args


@dataclass
class NonAdaptiveRoundTrainingResult:
    model: Any
    trainer: Optional[Any]
    skipped: bool
    recipe_phase_name: str
    recipe_phase_overrides: Optional[Dict[str, object]]


def recipe_phase_name_for_round(args: Any, *, use_recipe: bool, round_idx: int) -> str:
    if use_recipe and round_idx == 0 and not getattr(args, "treat_seed_as_round_zero", False):
        return "seed"
    return "self_improve"


def recipe_phase_overrides_for_round(
    args: Any,
    *,
    use_recipe: bool,
    recipe_phase_name: str,
    round_idx: int,
) -> Optional[Dict[str, object]]:
    if not use_recipe or recipe_phase_name != "self_improve":
        return None

    overrides: Dict[str, object] = {}
    lr_override = getattr(args, "self_improve_learning_rate", None)
    lr_switch_round = getattr(args, "self_improve_lr_switch_round", None)
    lr_after_switch = getattr(args, "self_improve_learning_rate_after_switch", None)
    if lr_switch_round is not None and lr_after_switch is not None and round_idx >= int(lr_switch_round):
        lr_override = lr_after_switch
    if lr_override is not None:
        overrides["learning_rate"] = float(lr_override)
    warmup_override = getattr(args, "self_improve_warmup_steps", None)
    if warmup_override is not None:
        overrides["warmup_steps"] = int(warmup_override)
    return overrides or None


def train_nonadaptive_round_model(
    *,
    args: Any,
    task: Any,
    model: Any,
    tokenizer: Any,
    train_examples: List[Any],
    round_dir: Path,
    config: Any,
    data_collator: Any,
    round_idx: int,
    new_run: bool,
    save_model_this_round: bool,
    use_recipe: bool,
    recipe_name: str,
    dataset_cls: Callable[[List[Any], Any], Any] = TokenizedPromptTargetDataset,
    make_training_args_fn: Callable[..., Any] = make_training_args,
    build_trainer_fn: Callable[..., Any] = build_trainer,
) -> NonAdaptiveRoundTrainingResult:
    skip_round_training = bool(getattr(args, "treat_seed_as_round_zero", False) and new_run and round_idx == 0)
    recipe_phase_name = recipe_phase_name_for_round(args, use_recipe=use_recipe, round_idx=round_idx)
    recipe_phase_overrides = recipe_phase_overrides_for_round(
        args,
        use_recipe=use_recipe,
        recipe_phase_name=recipe_phase_name,
        round_idx=round_idx,
    )

    if skip_round_training:
        print(
            "[INFO] Treating seed checkpoint as completed round_00; skipping round-0 training.",
            flush=True,
        )
        if save_model_this_round:
            model.save_pretrained(round_dir)
            tokenizer.save_pretrained(round_dir)
        return NonAdaptiveRoundTrainingResult(
            model=model,
            trainer=None,
            skipped=True,
            recipe_phase_name=recipe_phase_name,
            recipe_phase_overrides=recipe_phase_overrides,
        )

    train_dataset = dataset_cls(train_examples, tokenizer)
    training_args = make_training_args_fn(
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
    trainer = build_trainer_fn(
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
    trained_model = trainer.model
    if save_model_this_round:
        if use_recipe:
            trainer.save_model(str(round_dir))
        else:
            trainer.save_model()
        tokenizer.save_pretrained(round_dir)
    return NonAdaptiveRoundTrainingResult(
        model=trained_model,
        trainer=trainer,
        skipped=False,
        recipe_phase_name=recipe_phase_name,
        recipe_phase_overrides=recipe_phase_overrides,
    )
