#!/usr/bin/env python3
"""Candidate train/eval scoring helpers for adaptive self-improvement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from self.core import worker_io
from self.core.candidate_training_runtime import (
    clear_cuda_cache,
    evaluate_model,
    make_config,
    train_checkpoint,
    train_post_task_proposal_rehearsal,
)
from self.core.candidate_training_mix import (
    build_candidate_training_mix,
    write_candidate_training_mix_artifacts,
)
from self.core.candidate_rewards import (
    build_no_pseudo_candidate_metrics,
    build_trained_candidate_metrics,
    mean_accuracy_for_sizes,
    static_frontier_sizes,
)
from self.core.experience_trace_models import OutcomeTraceExample, ProposalTraceExample
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposal_prompts import PromptBundle
from self.core.training import TrainingConfig


def train_and_score_candidate(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    item: CandidateWorkItem,
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    seed: int,
    model_bootstrap_cache: ModelBootstrapCache | None = None,
) -> CandidateMetrics:
    candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
    if not item.pseudo_examples:
        metrics = build_no_pseudo_candidate_metrics(
            args=args,
            item=item,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
        )
        worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
        return metrics

    training_mix = build_candidate_training_mix(
        args=args,
        source_examples=source_examples,
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        seed=seed,
    )
    write_candidate_training_mix_artifacts(
        candidate_dir=candidate_dir,
        task=task,
        args=args,
        source_examples=source_examples,
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        mix=training_mix,
    )
    model, tokenizer, task_model_dir = train_checkpoint(
        source_checkpoint=current_checkpoint,
        train_examples=training_mix.train_examples,
        output_dir=candidate_dir / "training",
        task=task,
        args=args,
        config=config,
        seed=seed,
        recipe_phase_name="self_improve",
        model_bootstrap_cache=model_bootstrap_cache,
    )
    model_dir = task_model_dir
    if training_mix.post_task_rehearsal_examples:
        del model
        del tokenizer
        clear_cuda_cache()
        model, tokenizer, model_dir = train_post_task_proposal_rehearsal(
            task_model_dir=task_model_dir,
            candidate_dir=candidate_dir,
            task=task,
            args=args,
            config=config,
            seed=seed,
            proposal_trace_buffer=proposal_trace_buffer,
            candidate_trace_examples=training_mix.candidate_trace_examples,
            post_task_rehearsal_examples=training_mix.post_task_rehearsal_examples,
        )
    final_accuracy, per_size_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=eval_examples,
        batch_size=config.per_device_eval_batch_size,
        decode_max_new_tokens=config.decode_max_new_tokens,
    )
    metrics = build_trained_candidate_metrics(
        args=args,
        item=item,
        final_accuracy=final_accuracy,
        per_size_accuracy=per_size_accuracy,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        model_dir=model_dir,
        proposal_trace_replay_count=len(training_mix.mixed_proposal_replay_examples),
        candidate_proposal_trace_count=len(training_mix.candidate_trace_examples),
        post_task_proposal_rehearsal_count=len(training_mix.post_task_rehearsal_examples),
        outcome_trace_replay_count=len(training_mix.outcome_replay_examples),
    )
    worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
    del model
    del tokenizer
    clear_cuda_cache()
    return metrics
