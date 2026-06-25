#!/usr/bin/env python3
"""Adaptive proposal and outcome trace builders."""

from __future__ import annotations

# --- from traces.py ---
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


# --- from traces.py ---
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import extract_json_object


JsonDict = Dict[str, Any]


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    numeric = _finite_or_none(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(sanitize_json_value(dict(payload)), sort_keys=True, separators=(",", ":"))


def _compact_accuracy_map(per_size_accuracy: Mapping[int, float]) -> JsonDict:
    return {
        str(int(size)): _round_or_none(value)
        for size, value in sorted(per_size_accuracy.items(), key=lambda item: int(item[0]))
    }


def _truncate_text(value: Any, max_chars: int = 160) -> str:
    text = str(value).strip().replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _limit_words(text: str, max_words: int = 35) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def _short_failure_code(category: Any, message: Any) -> Optional[str]:
    category_text = str(category or "").strip()
    message_text = str(message or "").lower()
    if not category_text and not message_text:
        return None
    if category_text == "duplicate" or "duplicate" in message_text:
        return "duplicate"
    if "already in the current source pool" in message_text:
        return "target_already_in_source"
    if "outside the allowed frontier" in message_text:
        return "target_outside_frontier"
    if "source pool" in message_text or "outside allowed source" in message_text:
        return "source_not_available"
    if "guard" in message_text or category_text == "enum_error":
        return "invalid_guard"
    if "pseudo" in message_text:
        return "no_pseudo_labels"
    if "data build" in message_text:
        return "data_build_failure"
    if category_text:
        return category_text
    return "invalid"


def _candidate_payload_from_result(result: Mapping[str, Any]) -> JsonDict:
    parsed = result.get("parsed_proposal")
    if isinstance(parsed, Mapping):
        payload: JsonDict = {}
        for field in ("left", "right", "guard", "target"):
            if field in parsed:
                payload[field] = parsed[field]
        parsed_prediction = result.get("parsed_prediction")
        if isinstance(parsed_prediction, Mapping):
            payload["prediction"] = dict(parsed_prediction)
        return sanitize_json_value(payload)

    raw = result.get("raw_output")
    raw_payload = raw if isinstance(raw, Mapping) else extract_json_object(str(raw))
    if isinstance(raw_payload, Mapping):
        if isinstance(raw_payload.get("proposal"), Mapping):
            proposal_payload = dict(raw_payload["proposal"])
            prediction_payload = raw_payload.get("prediction")
        else:
            proposal_payload = raw_payload
            prediction_payload = raw_payload.get("prediction")
        payload = {}
        for field in ("left", "right", "guard"):
            if field in proposal_payload:
                payload[field] = proposal_payload[field]
        try:
            payload["target"] = int(proposal_payload["left"]) + int(proposal_payload["right"])
        except (KeyError, TypeError, ValueError):
            pass
        if isinstance(prediction_payload, Mapping):
            payload["prediction"] = dict(prediction_payload)
        return sanitize_json_value(payload)

    return {"raw": _truncate_text(raw)}


def _candidate_payload_from_proposal(proposal: Any, prediction: Optional[Mapping[str, Any]] = None) -> JsonDict:
    payload: JsonDict = {
        "left": int(proposal.left),
        "right": int(proposal.right),
        "guard": str(proposal.guard),
        "target": int(proposal.target),
    }
    if prediction:
        payload["prediction"] = dict(prediction)
    return payload


def _target_from_candidate(candidate: Mapping[str, Any]) -> Optional[int]:
    if "target" in candidate:
        try:
            return int(candidate["target"])
        except (TypeError, ValueError):
            return None
    try:
        return int(candidate["left"]) + int(candidate["right"])
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_prediction(candidate: Mapping[str, Any]) -> Optional[JsonDict]:
    prediction = candidate.get("prediction")
    if isinstance(prediction, Mapping):
        return dict(prediction)
    return None


def _prediction_delta_error(
    *,
    prediction: Optional[Mapping[str, Any]],
    key: str,
    realized: Optional[float],
) -> Optional[float]:
    if prediction is None or realized is None:
        return None
    expected = _finite_or_none(prediction.get(key))
    if expected is None and key == "expected_avg_delta_from_init":
        expected = _finite_or_none(prediction.get("expected_final_delta_from_init"))
    if expected is None:
        return None
    return float(realized) - expected


def _render_outcome_trace_prompt(
    *,
    mode: str,
    task_name: str,
    source_sizes: Sequence[int],
    frontier_min: int,
    frontier_max: int,
    current_final_accuracy: float,
    init_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    candidate: Mapping[str, Any],
) -> str:
    state = {
        "source": [int(size) for size in sorted(source_sizes)],
        "frontier": [int(frontier_min), int(frontier_max)],
        "current_avg": _round_or_none(current_final_accuracy),
        "init_avg": _round_or_none(init_final_accuracy),
        "acc": _compact_accuracy_map(current_per_size_accuracy),
    }
    return "\n".join(
        [
            "TASK: predict_config_outcome",
            f"mode={mode}",
            f"task={task_name}",
            f"state={_compact_json(state)}",
            f"candidate={_compact_json(candidate)}",
        ]
    )


def _outcome_feedback(
    *,
    valid: bool,
    trained: bool,
    selected: bool,
    repeat_target: bool,
    target: Optional[int],
    reward: float,
    target_delta: Optional[float],
    frontier_delta: Optional[float],
    failure: Optional[str],
) -> str:
    target_text = "unknown target" if target is None else f"target {target}"
    if not valid:
        if failure == "duplicate":
            text = "Invalid candidate: it exactly duplicates another proposal in the same batch, so it spends evaluation budget without adding a distinct option."
        elif failure == "target_outside_frontier":
            text = f"Invalid candidate: {target_text} is outside the allowed range, so choose available source slices whose sum is inside the frontier."
        elif failure == "source_not_available":
            text = "Invalid candidate: at least one source slice is unavailable, so choose left and right from the current source pool."
        elif failure == "invalid_guard":
            text = "Invalid candidate: the guard is not one of the allowed guard rules for this task."
        else:
            text = f"Invalid candidate: {failure or 'the proposal could not be evaluated'}."
        return _limit_words(text)

    if not trained:
        return _limit_words(
            f"Valid but untrained candidate: {target_text} produced no retained pseudo-labels, so it gave no useful update signal."
        )

    if selected:
        repeat = "repeat " if repeat_target else ""
        return _limit_words(
            f"Selected {repeat}candidate: {target_text} achieved reward {reward:.4f} with frontier_delta {frontier_delta or 0.0:.4f}, so it was the best measured option."
        )

    if reward <= 0.0:
        return _limit_words(
            f"Valid but weak candidate: {target_text} was trainable but measured reward {reward:.4f}, so this state-action choice did not improve enough."
        )

    return _limit_words(
        f"Useful but not selected: {target_text} improved with reward {reward:.4f}, but another candidate had a stronger measured outcome."
    )


def _outcome_completion(
    *,
    mode: str,
    valid: bool,
    trained: bool,
    selected: bool,
    repeat_target: bool,
    target: Optional[int],
    reward: float,
    target_delta: Optional[float],
    frontier_delta: Optional[float],
    final_delta_init: Optional[float],
    final_delta_current: Optional[float],
    prediction: Optional[Mapping[str, Any]],
    failure: Optional[str],
) -> str:
    feedback = _outcome_feedback(
        valid=valid,
        trained=trained,
        selected=selected,
        repeat_target=repeat_target,
        target=target,
        reward=reward,
        target_delta=target_delta,
        frontier_delta=frontier_delta,
        failure=failure,
    )
    numeric_payload: JsonDict = {
        "valid": bool(valid),
        "trained": bool(trained),
        "selected": bool(selected),
        "repeat_target": bool(repeat_target),
        "target": target,
        "reward": _round_or_none(reward),
        "target_delta": _round_or_none(target_delta),
        "frontier_delta": _round_or_none(frontier_delta),
        "avg_delta_init": _round_or_none(final_delta_init),
        "avg_delta_current": _round_or_none(final_delta_current),
        "failure": failure,
    }
    if prediction is not None:
        numeric_payload["prediction"] = sanitize_json_value(dict(prediction))
        numeric_payload["frontier_delta_error"] = _round_or_none(
            _prediction_delta_error(
                prediction=prediction,
                key="expected_frontier_delta",
                realized=frontier_delta,
            )
        )
        numeric_payload["target_delta_error"] = _round_or_none(
            _prediction_delta_error(
                prediction=prediction,
                key="expected_target_delta",
                realized=target_delta,
            )
        )
        numeric_payload["avg_delta_from_current_error"] = _round_or_none(
            _prediction_delta_error(
                prediction=prediction,
                key="expected_avg_delta_from_current",
                realized=final_delta_current,
            )
        )
        numeric_payload["avg_delta_from_init_error"] = _round_or_none(
            _prediction_delta_error(
                prediction=prediction,
                key="expected_avg_delta_from_init",
                realized=final_delta_init,
            )
        )
    if mode == "numeric":
        return _compact_json(numeric_payload)
    if mode == "textual":
        return _compact_json({"feedback": feedback})
    if mode == "numeric_textual":
        payload = dict(numeric_payload)
        payload["feedback"] = feedback
        return _compact_json(payload)
    raise ValueError(f"Unsupported outcome trace target mode: {mode}")


# --- from traces.py ---
import argparse
from typing import Any, List, Mapping, Optional, Sequence



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
        if bool(result.get("candidate_dedup_skipped")):
            continue
        try:
            proposal_index = int(result["proposal_index"])
        except (KeyError, TypeError, ValueError):
            continue
        metric_index = result.get("candidate_proposal_index", proposal_index)
        try:
            metric_index_int = int(metric_index)
        except (TypeError, ValueError):
            metric_index_int = proposal_index
        metric = metrics_by_index.get(metric_index_int)
        traces.append(
            build_outcome_trace_example(
                args=args,
                task_name=task_name,
                condition=condition,
                round_index=round_index,
                result=result,
                metric=metric,
                selected=bool(selected_index is not None and metric_index_int == selected_index),
                source_sizes=source_sizes,
                frontier_min=frontier_min,
                frontier_max=frontier_max,
                current_final_accuracy=current_final_accuracy,
                init_final_accuracy=init_final_accuracy,
                current_per_size_accuracy=current_per_size_accuracy,
            )
        )
    return traces


# --- from traces.py ---
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from self.adaptive.proposal import build_trace_row, write_trace_jsonl
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


def write_round_trace(
    *,
    args: argparse.Namespace,
    task_name: str,
    round_index: int,
    prompt: PromptBundle,
    work_items: Sequence[Any],
    metrics: Sequence[Any],
    path: Path,
) -> List[JsonDict]:
    by_index = {item.index: item for item in work_items}
    positive = [metric for metric in metrics if metric.valid and metric.reward > 0.0]
    positive.sort(key=lambda metric: metric.reward, reverse=True)
    rows: List[JsonDict] = []
    for metric in positive[: max(0, args.max_traces_per_round)]:
        item = by_index.get(metric.index)
        if item is None:
            continue
        rows.append(
            build_trace_row(
                round_index=round_index,
                task=task_name,
                condition=args.condition,
                reward=metric.reward,
                frontier_delta=metric.frontier_delta,
                final_accuracy=metric.final_accuracy,
                prompt=prompt.text(),
                completion=item.completion,
                extra_metadata=proposal_trace_metadata(metric),
            )
        )
    write_trace_jsonl(path, rows)
    return rows


def proposal_trace_metadata(metric: Any) -> JsonDict:
    return {
        "proposal_index": metric.index,
        "proposal_id": metric.row_id,
        "target": metric.proposal.target,
        "left": metric.proposal.left,
        "right": metric.proposal.right,
        "guard": metric.proposal.guard,
        "proposal_prediction": metric.proposal_prediction,
        "target_accuracy": metric.target_accuracy,
        "current_target_accuracy": metric.current_target_accuracy,
        "target_delta": metric.target_delta,
        "frontier_accuracy": metric.frontier_accuracy,
        "current_frontier_accuracy": metric.current_frontier_accuracy,
        "frontier_delta": metric.frontier_delta,
        "init_final_accuracy": metric.init_final_accuracy,
        "final_accuracy_delta": metric.final_accuracy_delta,
        "final_accuracy_delta_from_current": metric.final_accuracy_delta_from_current,
        "pseudo_count": metric.pseudo_count,
    }


def build_selected_proposal_trace_example(
    *,
    task_name: str,
    condition: str,
    round_index: int,
    prompt: PromptBundle,
    selected_item: Any,
    selected: Any,
) -> ProposalTraceExample:
    return ProposalTraceExample(
        prompt_text=prompt.text(),
        completion=selected_item.completion,
        task=task_name,
        condition=condition,
        round_index=round_index,
        reward=selected.reward,
        metadata=proposal_trace_metadata(selected),
    )


def build_candidate_proposal_trace_example(
    *,
    task_name: str,
    condition: str,
    round_index: int,
    prompt: PromptBundle,
    item: Any,
) -> ProposalTraceExample:
    proposal = item.proposal
    return ProposalTraceExample(
        prompt_text=prompt.text(),
        completion=item.completion,
        task=task_name,
        condition=condition,
        round_index=round_index,
        reward=math.nan,
        metadata={
            "proposal_index": item.index,
            "proposal_id": item.row_id,
            "target": proposal.target,
            "left": proposal.left,
            "right": proposal.right,
            "guard": proposal.guard,
            "proposal_prediction": item.proposal_prediction,
            "candidate_local_trace": True,
        },
    )
