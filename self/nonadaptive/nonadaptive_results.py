"""Round-summary persistence for non-adaptive self-improvement runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from self.core.data_io import JsonDict, write_summary_records
from self.core.summaries import RoundSummary, summarize_round, summary_to_payload


@dataclass
class NonAdaptiveRoundSummaryRecord:
    summary: Any
    metrics_payload: JsonDict


def record_nonadaptive_round_summary(
    *,
    round_idx: int,
    max_size: int,
    train_example_count: int,
    pseudo_used_count: int,
    evaluation: Any,
    pseudo_generation_stats: JsonDict,
    round_dir: Path,
    save_model_policy: str,
    save_model_this_round: bool,
    summary_records: Dict[int, JsonDict],
    results_path: Path,
    task: Any,
    round_summary_cls: Callable[..., Any] = RoundSummary,
    summarize_round_fn: Callable[[Any, Any], None] = summarize_round,
    summary_to_payload_fn: Callable[[Any, Any], JsonDict] = summary_to_payload,
    write_summary_records_fn: Callable[[Dict[int, JsonDict], Path], None] = write_summary_records,
    json_module: Any = json,
) -> NonAdaptiveRoundSummaryRecord:
    summary = round_summary_cls(
        index=round_idx,
        max_size=max_size,
        train_example_count=train_example_count,
        pseudo_example_count=pseudo_used_count,
        eval_accuracy=evaluation.eval_accuracy,
        per_size_accuracy=evaluation.per_size_accuracy,
        output_dir=round_dir,
        composed_eval_accuracy=evaluation.composed_eval_accuracy,
        composed_eval_slices=evaluation.composed_slice_metrics,
        pseudo_generation_stats=pseudo_generation_stats,
    )
    summarize_round_fn(summary, task)

    metrics_payload = summary_to_payload_fn(summary, task)
    metrics_payload["save_model_policy"] = save_model_policy
    metrics_payload["model_dir"] = str(round_dir) if save_model_this_round else None
    with (round_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json_module.dump(metrics_payload, handle, indent=2)

    summary_records[round_idx] = metrics_payload
    write_summary_records_fn(summary_records, results_path)
    return NonAdaptiveRoundSummaryRecord(summary=summary, metrics_payload=metrics_payload)
