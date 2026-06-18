"""Proposal-GRPO reward shaping and trace construction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core.data_io import sanitize_json_value
from self.core.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]

PROPOSAL_GRPO_ZERO_VARIANCE_MODES = ("fixed_baseline", "skip")
PROPOSAL_GRPO_REWARD_MODES = ("outcome", "validity")
PROPOSAL_GRPO_REWARD_BY_CATEGORY: Dict[str, float] = {
    "valid": 1.0,
    "range_error": 0.6,
    "enum_error": 0.5,
    "schema_error": 0.25,
    "parse_error": 0.0,
}
PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY: Dict[str, float] = {
    "range_error": -0.4,
    "enum_error": -0.5,
    "schema_error": -0.7,
    "parse_error": -1.0,
}


@dataclass(frozen=True)
class ProposalGRPOTrace:
    """Raw proposal completion and shaped reward used for one GRPO update."""

    proposal_index: int
    proposal_id: Optional[str]
    prompt_text: str
    completion: str
    reward: float
    advantage: float
    validation_category: str
    validation_message: str
    valid: bool
    metadata: JsonDict

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "proposal_index": self.proposal_index,
                "proposal_id": self.proposal_id,
                "prompt": self.prompt_text,
                "completion": self.completion,
                "reward": self.reward,
                "advantage": self.advantage,
                "validation_category": self.validation_category,
                "validation_message": self.validation_message,
                "valid": self.valid,
                "metadata": self.metadata,
            }
        )


def proposal_grpo_reward(result: Mapping[str, Any]) -> float:
    if bool(result.get("valid")):
        return PROPOSAL_GRPO_REWARD_BY_CATEGORY["valid"]
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_REWARD_BY_CATEGORY.get(category, 0.0))


def proposal_grpo_outcome_invalid_reward(result: Mapping[str, Any]) -> float:
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY.get(category, -1.0))


def _proposal_index(result: Mapping[str, Any], default: int) -> int:
    try:
        return int(result.get("proposal_index", default))
    except (TypeError, ValueError):
        return default


def _is_system_candidate_failure(metric: Any) -> bool:
    if metric.valid:
        return False
    reason = str(metric.failure_reason or "").lower()
    if reason == "no pseudo labels retained":
        return False
    return bool(reason)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def proposal_grpo_reward_for_result(
    result: Mapping[str, Any],
    *,
    metric: Optional[Any],
    reward_mode: str,
    outcome_scale: float,
) -> Tuple[Optional[float], str]:
    if reward_mode == "validity":
        return proposal_grpo_reward(result), "validity"
    if reward_mode != "outcome":
        raise ValueError(f"Unsupported proposal_grpo_reward_mode={reward_mode!r}.")
    if not bool(result.get("valid")):
        return proposal_grpo_outcome_invalid_reward(result), "invalid"
    if metric is None:
        return 0.0, "valid_untrained"
    if not metric.valid:
        if _is_system_candidate_failure(metric):
            return None, "skipped_system_failure"
        return 0.0, "valid_untrained"
    return _clamp(float(metric.reward) / float(outcome_scale), -1.0, 1.0), "outcome"


def proposal_grpo_advantages(
    rewards: Sequence[float],
    *,
    zero_variance: str,
    fixed_baseline: float,
    eps: float = 1e-6,
) -> Tuple[List[float], bool, str]:
    if not rewards:
        return [], True, "no_rewards"
    reward_values = [float(reward) for reward in rewards]
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((reward - mean_reward) ** 2 for reward in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)
    if std_reward > eps:
        return [(reward - mean_reward) / (std_reward + eps) for reward in reward_values], False, "normalized"
    if zero_variance == "skip":
        return [0.0 for _ in reward_values], True, "zero_variance"
    if zero_variance != "fixed_baseline":
        raise ValueError(f"Unsupported proposal_grpo_zero_variance={zero_variance!r}.")
    return [reward - fixed_baseline for reward in reward_values], False, "fixed_baseline"


def build_proposal_grpo_traces(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any] = (),
) -> Tuple[List[ProposalGRPOTrace], JsonDict]:
    prompt_text = prompt.text()
    metrics_by_index = {int(metric.index): metric for metric in candidate_metrics}
    rewards: List[float] = []
    included_results: List[Tuple[int, Mapping[str, Any], Optional[Any], float, str]] = []
    reward_source_counts: Dict[str, int] = {}
    skipped_system_failure_count = 0
    for index, result in enumerate(proposal_results):
        proposal_index = _proposal_index(result, index)
        metric = metrics_by_index.get(proposal_index)
        reward, reward_source = proposal_grpo_reward_for_result(
            result,
            metric=metric,
            reward_mode=args.proposal_grpo_reward_mode,
            outcome_scale=args.proposal_grpo_outcome_scale,
        )
        reward_source_counts[reward_source] = reward_source_counts.get(reward_source, 0) + 1
        if reward is None:
            skipped_system_failure_count += 1
            continue
        rewards.append(float(reward))
        included_results.append((proposal_index, result, metric, float(reward), reward_source))
    advantages, skip_update, advantage_mode = proposal_grpo_advantages(
        rewards,
        zero_variance=args.proposal_grpo_zero_variance,
        fixed_baseline=args.proposal_grpo_fixed_baseline,
    )
    traces: List[ProposalGRPOTrace] = []
    for proposal_index, result, metric, reward, reward_source, advantage in (
        (*payload, advantage)
        for payload, advantage in zip(included_results, advantages)
    ):
        raw = result.get("raw_output", "")
        normalized_completion = result.get("completion")
        if result.get("valid") and isinstance(normalized_completion, str) and normalized_completion:
            completion = normalized_completion
            completion_source = "normalized"
        else:
            completion = raw if isinstance(raw, str) else json.dumps(sanitize_json_value(raw), sort_keys=True)
            completion_source = "raw"
        traces.append(
            ProposalGRPOTrace(
                proposal_index=proposal_index,
                proposal_id=str(result["id"]) if result.get("id") is not None else None,
                prompt_text=prompt_text,
                completion=completion,
                reward=float(reward),
                advantage=float(advantage),
                validation_category=str(
                    result.get("validation_category") or ("valid" if result.get("valid") else "unknown")
                ),
                validation_message=str(result.get("validation_message") or ""),
                valid=bool(result.get("valid")),
                metadata={
                    "duplicate": bool(result.get("duplicate")),
                    "repeat_target": bool(result.get("repeat_target")),
                    "parsed_proposal": result.get("parsed_proposal"),
                    "parsed_prediction": result.get("parsed_prediction"),
                    "proposal_output_schema": result.get("proposal_output_schema"),
                    "completion_source": completion_source,
                    "reward_source": reward_source,
                    "candidate_reward": metric.reward if metric is not None else None,
                    "frontier_delta": metric.frontier_delta if metric is not None else None,
                    "target_delta": metric.target_delta if metric is not None else None,
                    "proposal_prediction": metric.proposal_prediction if metric is not None else None,
                },
            )
        )
    mean_reward = sum(rewards) / len(rewards) if rewards else math.nan
    reward_std = (
        math.sqrt(sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards))
        if rewards
        else math.nan
    )
    return traces, {
        "reward_mean": mean_reward,
        "reward_std": reward_std,
        "advantage_mode": advantage_mode,
        "zero_variance_skip": bool(skip_update),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "reward_source_counts": reward_source_counts,
        "skipped_system_failure_count": skipped_system_failure_count,
        "input_proposal_count": len(proposal_results),
        "trace_candidate_metric_count": len(candidate_metrics),
    }
