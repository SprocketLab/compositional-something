"""Candidate training-example mix construction and artifact writing."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence

from self.core import worker_io
from self.core.data_io import save_examples
from self.core.experience_trace_models import (
    OutcomeTraceExample,
    ProposalTraceExample,
    build_post_task_proposal_rehearsal_examples,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.core.experience_traces import build_candidate_proposal_trace_example
from self.core.models import CandidateWorkItem
from self.core.proposals import PromptBundle, write_trace_jsonl


@dataclass(frozen=True)
class CandidateTrainingMix:
    task_train_examples: List[Any]
    outcome_replay_examples: List[OutcomeTraceExample]
    candidate_trace_examples: List[ProposalTraceExample]
    mixed_proposal_replay_examples: List[ProposalTraceExample]
    mixed_candidate_trace_examples: List[ProposalTraceExample]
    post_task_rehearsal_examples: List[ProposalTraceExample]
    train_examples: List[Any]

    @property
    def summary_counts(self) -> dict[str, int]:
        return {
            "task_train_examples": len(self.task_train_examples),
            "outcome_trace_replay_examples": len(self.outcome_replay_examples),
            "proposal_trace_replay_examples": len(self.mixed_proposal_replay_examples),
            "candidate_proposal_trace_examples": len(self.candidate_trace_examples),
            "mixed_candidate_proposal_trace_examples": len(self.mixed_candidate_trace_examples),
            "post_task_proposal_rehearsal_examples": len(self.post_task_rehearsal_examples),
            "total_train_examples": len(self.train_examples),
        }


def build_candidate_training_mix(
    *,
    args: argparse.Namespace,
    source_examples: Sequence[Any],
    item: CandidateWorkItem,
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    seed: int,
    random_cls: Callable[[int], random.Random] = random.Random,
) -> CandidateTrainingMix:
    task_train_examples = list(source_examples) + list(item.pseudo_examples)
    outcome_replay_examples = sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcome_trace_buffer,
        task_train_count=len(task_train_examples),
        rng=random_cls(seed + 6151),
    )
    candidate_trace_examples: List[ProposalTraceExample] = []
    if item.completion and (args.post_task_proposal_rehearsal or args.proposal_trace_replay_ratio > 0.0):
        candidate_trace_examples.append(
            build_candidate_proposal_trace_example(
                task_name=args.task,
                condition=args.condition,
                round_index=round_index,
                prompt=proposal_prompt,
                item=item,
            )
        )
    mixed_proposal_replay_examples: List[ProposalTraceExample] = []
    if not args.post_task_proposal_rehearsal:
        mixed_proposal_replay_examples = sample_proposal_trace_replay(
            args=args,
            trace_buffer=proposal_trace_buffer,
            task_train_count=len(task_train_examples),
            rng=random_cls(seed + 7919),
        )
    mixed_candidate_trace_examples = [] if args.post_task_proposal_rehearsal else list(candidate_trace_examples)
    post_task_rehearsal_examples = build_post_task_proposal_rehearsal_examples(
        args=args,
        proposal_trace_buffer=proposal_trace_buffer,
        candidate_trace_examples=candidate_trace_examples,
        rng=random_cls(seed + 8863),
    )
    train_examples = (
        task_train_examples
        + list(outcome_replay_examples)
        + mixed_proposal_replay_examples
        + mixed_candidate_trace_examples
    )
    return CandidateTrainingMix(
        task_train_examples=task_train_examples,
        outcome_replay_examples=list(outcome_replay_examples),
        candidate_trace_examples=candidate_trace_examples,
        mixed_proposal_replay_examples=mixed_proposal_replay_examples,
        mixed_candidate_trace_examples=mixed_candidate_trace_examples,
        post_task_rehearsal_examples=list(post_task_rehearsal_examples),
        train_examples=train_examples,
    )


def write_candidate_training_mix_artifacts(
    *,
    candidate_dir: Path,
    task: Any,
    args: argparse.Namespace,
    source_examples: Sequence[Any],
    item: CandidateWorkItem,
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    mix: CandidateTrainingMix,
    save_examples_fn: Callable[..., None] = save_examples,
    write_trace_jsonl_fn: Callable[..., None] = write_trace_jsonl,
    write_json_fn: Callable[[Path, Any], None] = worker_io.write_json,
) -> None:
    save_examples_fn(candidate_dir / "train_examples.jsonl", mix.task_train_examples, task.serialize_example)
    if mix.outcome_replay_examples:
        write_trace_jsonl_fn(
            candidate_dir / "outcome_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in mix.outcome_replay_examples],
        )
    if mix.mixed_proposal_replay_examples:
        write_trace_jsonl_fn(
            candidate_dir / "proposal_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in mix.mixed_proposal_replay_examples],
        )
    if mix.candidate_trace_examples:
        write_trace_jsonl_fn(
            candidate_dir / "candidate_proposal_trace_example.jsonl",
            [example.to_json_dict() for example in mix.candidate_trace_examples],
        )
    if mix.post_task_rehearsal_examples:
        write_trace_jsonl_fn(
            candidate_dir / "post_task_proposal_rehearsal_examples.jsonl",
            [example.to_json_dict() for example in mix.post_task_rehearsal_examples],
        )
    write_json_fn(
        candidate_dir / "train_mix_summary.json",
        {
            **mix.summary_counts,
            "source_examples": len(source_examples),
            "pseudo_examples": len(item.pseudo_examples),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "outcome_trace_target_mode": args.outcome_trace_target_mode,
            "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
            "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
            "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
            "post_task_proposal_rehearsal": bool(args.post_task_proposal_rehearsal),
            "post_task_proposal_rehearsal_repeat_count": args.post_task_proposal_rehearsal_repeat_count,
            "post_task_proposal_rehearsal_max_examples": args.post_task_proposal_rehearsal_max_examples,
        },
    )
