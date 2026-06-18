"""Outcome-trace construction helpers."""

from __future__ import annotations

import argparse
from typing import Any, List, Mapping, Optional, Sequence

from self.core.experience_outcome_rendering import (
    _candidate_payload_from_proposal,
    _candidate_payload_from_result,
    _candidate_prediction,
    _outcome_completion,
    _render_outcome_trace_prompt,
    _short_failure_code,
    _target_from_candidate,
)
from self.core.experience_trace_models import OutcomeTraceExample


def build_outcome_trace_example(
    *,
    args: argparse.Namespace,
    task_name: str,
    condition: str,
    round_index: int,
    result: Mapping[str, Any],
    metric: Optional[Any],
    selected: bool,
    source_sizes: Sequence[int],
    frontier_min: int,
    frontier_max: int,
    current_final_accuracy: float,
    init_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
) -> OutcomeTraceExample:
    if metric is not None:
        candidate = _candidate_payload_from_proposal(metric.proposal, metric.proposal_prediction)
    else:
        candidate = _candidate_payload_from_result(result)
    prediction = _candidate_prediction(candidate)
    target = _target_from_candidate(candidate)
    repeat_target = bool(target is not None and target in set(int(size) for size in source_sizes))
    valid = bool(result.get("valid"))
    trained = bool(metric is not None and metric.valid)
    if metric is None:
        if valid:
            failure = "data_build_failure"
        else:
            failure = _short_failure_code(result.get("validation_category"), result.get("validation_message"))
        reward = float(args.invalid_outcome_reward)
        target_delta = None
        frontier_delta = None
        final_delta_init = None
        final_delta_current = None
    elif metric.valid:
        failure = None
        reward = float(metric.reward)
        target_delta = float(metric.target_delta)
        frontier_delta = float(metric.frontier_delta)
        final_delta_init = float(metric.final_accuracy_delta)
        final_delta_current = float(metric.final_accuracy_delta_from_current)
    else:
        failure = _short_failure_code("candidate_failure", metric.failure_reason)
        reward = float(args.invalid_outcome_reward)
        target_delta = None
        frontier_delta = None
        final_delta_init = None
        final_delta_current = None

    prompt_text = _render_outcome_trace_prompt(
        mode=args.outcome_trace_target_mode,
        task_name=task_name,
        source_sizes=source_sizes,
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        current_final_accuracy=current_final_accuracy,
        init_final_accuracy=init_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        candidate=candidate,
    )
    completion = _outcome_completion(
        mode=args.outcome_trace_target_mode,
        valid=valid,
        trained=trained,
        selected=selected,
        repeat_target=repeat_target,
        target=target,
        reward=reward,
        target_delta=target_delta,
        frontier_delta=frontier_delta,
        final_delta_init=final_delta_init,
        final_delta_current=final_delta_current,
        prediction=prediction,
        failure=failure,
    )
    return OutcomeTraceExample(
        prompt_text=prompt_text,
        completion=completion,
        task=task_name,
        condition=condition,
        round_index=round_index,
        mode=args.outcome_trace_target_mode,
        reward=reward,
        metadata={
            "proposal_index": int(result.get("proposal_index", metric.index if metric else -1)),
            "proposal_id": result.get("id") if metric is None else metric.row_id,
            "target": target,
            "repeat_target": repeat_target,
            "valid": valid,
            "trained": trained,
            "selected": selected,
            "failure": failure,
            "prediction": prediction,
        },
    )


def build_round_outcome_trace_examples(
    *,
    args: argparse.Namespace,
    task_name: str,
    condition: str,
    round_index: int,
    proposal_results: Sequence[Mapping[str, Any]],
    metrics: Sequence[Any],
    selected: Optional[Any],
    source_sizes: Sequence[int],
    frontier_min: int,
    frontier_max: int,
    current_final_accuracy: float,
    init_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
) -> List[OutcomeTraceExample]:
    if args.outcome_trace_target_mode == "none" or condition != "config":
        return []
    metrics_by_index = {metric.index: metric for metric in metrics}
    selected_index = selected.index if selected is not None else None
    traces: List[OutcomeTraceExample] = []
    for result in proposal_results:
        try:
            index = int(result["proposal_index"])
        except (KeyError, TypeError, ValueError):
            continue
        metric = metrics_by_index.get(index)
        traces.append(
            build_outcome_trace_example(
                args=args,
                task_name=task_name,
                condition=condition,
                round_index=round_index,
                result=result,
                metric=metric,
                selected=bool(selected_index is not None and index == selected_index),
                source_sizes=source_sizes,
                frontier_min=frontier_min,
                frontier_max=frontier_max,
                current_final_accuracy=current_final_accuracy,
                init_final_accuracy=init_final_accuracy,
                current_per_size_accuracy=current_per_size_accuracy,
            )
        )
    return traces
