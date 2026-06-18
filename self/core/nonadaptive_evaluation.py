"""Round evaluation helpers for the non-adaptive self-improvement loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from self.core.evaluation import evaluate_accuracy_with_breakdown, write_prediction_debug_samples
from self.core.summaries import SliceMetric


@dataclass
class NonAdaptiveEvaluationResult:
    eval_accuracy: float
    per_size_accuracy: Dict[int, float]
    composed_eval_accuracy: float
    composed_slice_metrics: Dict[str, SliceMetric]


def evaluate_nonadaptive_round(
    *,
    model: Any,
    tokenizer: Any,
    task: Any,
    eval_examples: List[Any],
    composed_eval_slices: Dict[str, List[Any]],
    composed_eval_component_map: Any,
    round_dir: Path,
    batch_size: int,
    eval_decode_tokens: int,
    composed_eval_decode_tokens: int,
    evaluate_accuracy_fn: Callable[..., tuple[float, Dict[int, float]]] = evaluate_accuracy_with_breakdown,
    write_debug_samples_fn: Callable[..., None] = write_prediction_debug_samples,
    slice_metric_cls: Callable[..., SliceMetric] = SliceMetric,
) -> NonAdaptiveEvaluationResult:
    eval_accuracy, per_size_accuracy = evaluate_accuracy_fn(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        batch_size=batch_size,
        max_new_tokens=eval_decode_tokens,
        size_getter=task.size_of,
        prediction_parser=task.prediction_parser,
    )

    composed_slice_metrics: Dict[str, SliceMetric] = {}
    composed_correct_total = 0.0
    composed_count_total = 0
    for slice_name, slice_examples in composed_eval_slices.items():
        if slice_examples:
            slice_accuracy, slice_per_size_accuracy = evaluate_accuracy_fn(
                model=model,
                tokenizer=tokenizer,
                examples=slice_examples,
                batch_size=batch_size,
                max_new_tokens=composed_eval_decode_tokens,
                size_getter=task.size_of,
                prediction_parser=task.prediction_parser,
            )
        else:
            slice_accuracy = math.nan
            slice_per_size_accuracy = {}
        composed_slice_metrics[slice_name] = slice_metric_cls(
            accuracy=slice_accuracy,
            count=len(slice_examples),
            per_size_accuracy=slice_per_size_accuracy,
        )
        if slice_name in {"accepted_by_guard", "rejected_by_guard"} and slice_examples:
            write_debug_samples_fn(
                round_dir / f"composed_eval_{slice_name}_debug.jsonl",
                model=model,
                tokenizer=tokenizer,
                examples=slice_examples,
                batch_size=batch_size,
                max_new_tokens=composed_eval_decode_tokens,
                size_getter=task.size_of,
                key_getter=task.key_for_example,
                component_map=composed_eval_component_map,
                prediction_parser=task.prediction_parser,
            )
        if slice_examples and not math.isnan(slice_accuracy):
            composed_correct_total += slice_accuracy * len(slice_examples)
            composed_count_total += len(slice_examples)

    composed_eval_accuracy = composed_correct_total / composed_count_total if composed_count_total > 0 else math.nan
    return NonAdaptiveEvaluationResult(
        eval_accuracy=eval_accuracy,
        per_size_accuracy=per_size_accuracy,
        composed_eval_accuracy=composed_eval_accuracy,
        composed_slice_metrics=composed_slice_metrics,
    )
