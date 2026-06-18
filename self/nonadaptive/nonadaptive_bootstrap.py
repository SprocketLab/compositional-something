"""Resume and model/bootstrap setup for the non-adaptive loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from self.core.data_io import JsonDict, load_examples
from self.core.evaluation import resolve_max_new_tokens
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.recipe_training import PaddingAwareCausalLMDataCollator
from self.core.training import CausalLMDataCollator, TrainingConfig


@dataclass
class NonAdaptiveBootstrap:
    resume_round: int
    model: Any
    tokenizer: Any
    config: TrainingConfig
    train_base_decode_tokens: int
    eval_decode_tokens: int
    composed_eval_decode_tokens: int
    data_collator: Any
    summary_records: Dict[int, JsonDict]
    pseudo_examples: List[Any]


def prepare_nonadaptive_bootstrap(
    args: Any,
    task: Any,
    *,
    base_output_dir: Path,
    base_train_examples: List[Any],
    eval_examples: List[Any],
    composed_eval_examples: List[Any],
    existing_summaries: Dict[int, JsonDict],
    resume_requested: bool,
    reset_each_round: bool,
    use_recipe: bool,
    recipe_name: str,
    load_examples_fn: Callable[[Path, Callable[[JsonDict], Any]], List[Any]] = load_examples,
    instantiate_model_and_tokenizer_fn: Callable[..., tuple[Any, Any]] = instantiate_model_and_tokenizer,
    training_config_cls: Any = TrainingConfig,
    resolve_max_new_tokens_fn: Callable[[List[Any], int], int] = resolve_max_new_tokens,
    recipe_collator_cls: Any = PaddingAwareCausalLMDataCollator,
    default_collator_cls: Any = CausalLMDataCollator,
) -> NonAdaptiveBootstrap:
    resume_round, model_name_or_path = _resolve_resume_model_path(
        args,
        base_output_dir=base_output_dir,
        existing_summaries=existing_summaries,
        resume_requested=resume_requested,
        reset_each_round=reset_each_round,
    )

    token_initializers = task.token_initializers(args) if hasattr(task, "token_initializers") else {}
    model, tokenizer = instantiate_model_and_tokenizer_fn(
        model_name_or_path,
        bf16=args.bf16,
        fp16=args.fp16,
        token_initializers=token_initializers,
        init_from_scratch=getattr(args, "init_from_scratch", False),
        tokenizer_mode=str(getattr(args, "tokenizer_mode", "auto")),
        recipe=recipe_name,
    )
    config = training_config_cls(
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
    if use_recipe:
        data_collator = recipe_collator_cls(tokenizer=tokenizer, padding_side="right")
    else:
        data_collator = default_collator_cls(tokenizer)

    pseudo_examples: List[Any] = []
    if resume_round > 0:
        prev_round_dir = base_output_dir / f"round_{resume_round-1:02d}"
        pseudo_seed_path = prev_round_dir / "pseudo_for_next_round.jsonl"
        if not pseudo_seed_path.exists():
            raise RuntimeError(
                f"Pseudo dataset for round {resume_round} is missing (expected {pseudo_seed_path}). "
                "Please rerun the previous round to regenerate the pseudo labels before resuming."
            )
        pseudo_examples = load_examples_fn(pseudo_seed_path, task.deserialize_example)
        print(
            f"[INFO] Loaded {len(pseudo_examples)} pseudo examples for upcoming round {resume_round} "
            f"from {pseudo_seed_path}.",
            flush=True,
        )

    return NonAdaptiveBootstrap(
        resume_round=resume_round,
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_base_decode_tokens=resolve_max_new_tokens_fn(base_train_examples, config.decode_max_new_tokens),
        eval_decode_tokens=resolve_max_new_tokens_fn(eval_examples, config.decode_max_new_tokens),
        composed_eval_decode_tokens=resolve_max_new_tokens_fn(
            composed_eval_examples,
            config.decode_max_new_tokens,
        ),
        data_collator=data_collator,
        summary_records=dict(existing_summaries),
        pseudo_examples=pseudo_examples,
    )


def _resolve_resume_model_path(
    args: Any,
    *,
    base_output_dir: Path,
    existing_summaries: Dict[int, JsonDict],
    resume_requested: bool,
    reset_each_round: bool,
) -> tuple[int, str]:
    resume_round = 0
    if not resume_requested:
        return resume_round, args.model_name

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
            raise ValueError(f"Cannot resume from round {resume_round}; checkpoint directory {checkpoint_dir} is missing.")
        model_name_or_path = str(checkpoint_dir)
    else:
        model_name_or_path = args.model_name
    print(f"[INFO] Resuming training from round {resume_round}.", flush=True)
    return resume_round, model_name_or_path
