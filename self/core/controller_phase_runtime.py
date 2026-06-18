"""In-process controller phases for adaptive candidate training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import set_seed

from self.core.attempt_prompt_runtime import build_attempt_prompt
from self.core.candidate_data import attach_pseudo_labels, build_candidate_work_items
from self.core.candidate_scoring import evaluate_model, train_checkpoint
from self.core.controller_phases import RoundModelPhaseResult, SeedPhaseResult
from self.core.proposal_runtime import load_or_generate_proposal_rows, validate_proposal_rows
from self.core import worker_io
from self.core.data_io import ensure_dir
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import TrainingConfig

JsonDict = Dict[str, Any]


def run_seed_phase(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    output_dir: Path,
    seed: int,
) -> SeedPhaseResult:
    if args.treat_seed_as_round_zero:
        current_checkpoint = args.model_name
        model, tokenizer = instantiate_model_and_tokenizer(
            current_checkpoint,
            bf16=args.bf16,
            fp16=args.fp16,
            init_from_scratch=args.init_from_scratch,
            tokenizer_mode=args.tokenizer_mode,
            recipe=args.recipe,
        )
        model_dir: Optional[Path] = None
    else:
        seed_dir = output_dir / "round_00" / "seed_training"
        model, tokenizer, model_dir = train_checkpoint(
            source_checkpoint=args.model_name,
            train_examples=source_examples,
            output_dir=seed_dir,
            task=task,
            args=args,
            config=config,
            seed=seed,
            recipe_phase_name="seed",
        )
        current_checkpoint = str(model_dir)

    try:
        current_final_accuracy, current_per_size_accuracy = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            task=task,
            examples=eval_examples,
            batch_size=config.per_device_eval_batch_size,
            decode_max_new_tokens=config.decode_max_new_tokens,
        )
        init_final_accuracy = (
            float(args.init_final_accuracy) if args.init_final_accuracy is not None else current_final_accuracy
        )
        round_zero_dir = output_dir / "round_00"
        ensure_dir(round_zero_dir)
        metrics_payload: JsonDict = {
            "round": 0,
            "eval_accuracy": current_final_accuracy,
            "per_size_accuracy": current_per_size_accuracy,
            "init_final_accuracy": init_final_accuracy,
        }
        if args.treat_seed_as_round_zero:
            metrics_payload.update(
                {
                    "seed_checkpoint": current_checkpoint,
                    "treat_seed_as_round_zero": True,
                }
            )
        else:
            metrics_payload.update(
                {
                    "model_dir": current_checkpoint,
                    "train_examples": len(source_examples),
                }
            )
        worker_io.write_json(round_zero_dir / "metrics.json", metrics_payload)
        return SeedPhaseResult(
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy={int(size): float(score) for size, score in current_per_size_accuracy.items()},
            init_final_accuracy=init_final_accuracy,
            model_dir=model_dir,
        )
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_round_model_phase(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    current_checkpoint: str,
    round_dir: Path,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    exclude_keys: set[Any],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    init_final_accuracy: float,
    seed: int,
) -> RoundModelPhaseResult:
    set_seed(seed)
    rng = random.Random(seed)
    frontier_min = args.frontier_min_size
    frontier_max = args.frontier_max_size
    current_model, current_tokenizer = instantiate_model_and_tokenizer(
        current_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        current_final_accuracy, current_per_size_accuracy = evaluate_model(
            model=current_model,
            tokenizer=current_tokenizer,
            task=task,
            examples=eval_examples,
            batch_size=config.per_device_eval_batch_size,
            decode_max_new_tokens=config.decode_max_new_tokens,
        )
        attempt_prompt = build_attempt_prompt(
            args=args,
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            init_final_accuracy=init_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            source_sizes=source_sizes,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            extra_aggregate_metrics={"proposal_output_schema": args.proposal_output_schema},
        )
        prompt = attempt_prompt.prompt
        default_program_pair = attempt_prompt.default_program_pair
        worker_io.write_json(round_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})

        rows = load_or_generate_proposal_rows(
            args=args,
            prompt=prompt,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
            round_index=selected_round_for_prompt,
            attempt_index=attempt_index,
        )
        proposal_results = validate_proposal_rows(
            rows=rows,
            args=args,
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
            default_pair=default_program_pair,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
        )
        worker_io.write_json(round_dir / "raw_proposals.json", rows)
        worker_io.write_json(round_dir / "proposal_results.json", proposal_results)

        work_items = build_candidate_work_items(
            args=args,
            task=task,
            round_dir=round_dir,
            proposal_results=proposal_results,
            source_examples=source_examples,
            exclude_keys=exclude_keys,
            rng=rng,
        )
        work_items = attach_pseudo_labels(
            args=args,
            task=task,
            round_dir=round_dir,
            work_items=work_items,
            source_examples=source_examples,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
            config=config,
        )
        return RoundModelPhaseResult(
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy={
                int(size): float(score) for size, score in current_per_size_accuracy.items()
            },
            prompt=prompt,
            proposal_results=list(proposal_results),
            work_items=list(work_items),
        )
    finally:
        del current_model
        del current_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
