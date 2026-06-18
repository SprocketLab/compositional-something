"""Trace example data models and replay samplers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ProposalTraceExample:
    """Prompt/target example for rehearsing selected proposal generation."""

    prompt_text: str
    completion: str
    task: str
    condition: str
    round_index: int
    reward: float
    metadata: JsonDict

    def prompt(self) -> str:
        return self.prompt_text

    def target(self) -> str:
        return self.completion

    def target_prefix(self) -> str:
        return ""

    def size_for_batching(self) -> int:
        return 0

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "prompt": self.prompt_text,
                "completion": self.completion,
                "task": self.task,
                "condition": self.condition,
                "round": self.round_index,
                "reward": self.reward,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class OutcomeTraceExample:
    """Compact state/action/outcome example for learning config consequences."""

    prompt_text: str
    completion: str
    task: str
    condition: str
    round_index: int
    mode: str
    reward: float
    metadata: JsonDict

    def prompt(self) -> str:
        return self.prompt_text

    def target(self) -> str:
        return self.completion

    def target_prefix(self) -> str:
        return ""

    def size_for_batching(self) -> int:
        target = self.metadata.get("target")
        try:
            return int(target)
        except (TypeError, ValueError):
            return 0

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "prompt": self.prompt_text,
                "completion": self.completion,
                "task": self.task,
                "condition": self.condition,
                "round": self.round_index,
                "mode": self.mode,
                "reward": self.reward,
                "metadata": self.metadata,
            }
        )


def proposal_trace_from_json(payload: Mapping[str, Any]) -> ProposalTraceExample:
    return ProposalTraceExample(
        prompt_text=str(payload.get("prompt", "")),
        completion=str(payload.get("completion", "")),
        task=str(payload.get("task", "")),
        condition=str(payload.get("condition", "")),
        round_index=int(payload.get("round") or 0),
        reward=_float_or_nan(payload.get("reward")),
        metadata=dict(payload.get("metadata") or {}),
    )


def outcome_trace_from_json(payload: Mapping[str, Any]) -> OutcomeTraceExample:
    return OutcomeTraceExample(
        prompt_text=str(payload.get("prompt", "")),
        completion=str(payload.get("completion", "")),
        task=str(payload.get("task", "")),
        condition=str(payload.get("condition", "")),
        round_index=int(payload.get("round") or 0),
        mode=str(payload.get("mode", "numeric")),
        reward=_float_or_nan(payload.get("reward")),
        metadata=dict(payload.get("metadata") or {}),
    )


def sample_proposal_trace_replay(
    *,
    args: Any,
    trace_buffer: Sequence[ProposalTraceExample],
    task_train_count: int,
    rng: random.Random,
) -> List[ProposalTraceExample]:
    if not trace_buffer:
        return []
    if args.proposal_trace_replay_ratio <= 0.0 or args.proposal_trace_replay_max_examples <= 0:
        return []
    requested = int(math.ceil(float(task_train_count) * float(args.proposal_trace_replay_ratio)))
    if requested <= 0:
        return []
    replay_count = min(int(args.proposal_trace_replay_max_examples), requested)
    return [rng.choice(trace_buffer) for _ in range(replay_count)]


def build_post_task_proposal_rehearsal_examples(
    *,
    args: Any,
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    candidate_trace_examples: Sequence[ProposalTraceExample],
    rng: random.Random,
) -> List[ProposalTraceExample]:
    if not args.post_task_proposal_rehearsal:
        return []
    if args.post_task_proposal_rehearsal_repeat_count <= 0:
        return []
    if args.post_task_proposal_rehearsal_max_examples <= 0:
        return []
    base_examples = list(proposal_trace_buffer) + list(candidate_trace_examples)
    if not base_examples:
        return []
    requested = len(base_examples) * int(args.post_task_proposal_rehearsal_repeat_count)
    replay_count = min(int(args.post_task_proposal_rehearsal_max_examples), requested)
    if replay_count <= 0:
        return []
    if replay_count <= len(base_examples):
        return rng.sample(base_examples, replay_count)
    examples = list(base_examples)
    while len(examples) < replay_count:
        examples.append(rng.choice(base_examples))
    rng.shuffle(examples)
    return examples


def sample_outcome_trace_replay(
    *,
    args: Any,
    trace_buffer: Sequence[OutcomeTraceExample],
    task_train_count: int,
    rng: random.Random,
) -> List[OutcomeTraceExample]:
    if args.outcome_trace_target_mode == "none" or not trace_buffer:
        return []
    if args.outcome_trace_replay_ratio <= 0.0 or args.outcome_trace_replay_max_examples <= 0:
        return []
    requested = int(math.ceil(float(task_train_count) * float(args.outcome_trace_replay_ratio)))
    if requested <= 0:
        return []
    replay_count = min(int(args.outcome_trace_replay_max_examples), requested)
    return [rng.choice(trace_buffer) for _ in range(replay_count)]


def _float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
