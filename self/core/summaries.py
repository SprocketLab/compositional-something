"""Round summary containers, payload conversion, and console reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from self.core.data_io import sanitize_float, sanitize_json_value


JsonDict = Dict[str, Any]


class SummaryTask(Protocol):
    size_label: str
    size_alias_singular: str

    def summary_payload_aliases(self, summary: "RoundSummary") -> JsonDict:
        ...


@dataclass
class SliceMetric:
    accuracy: Optional[float]
    count: int
    per_size_accuracy: Dict[int, float] = field(default_factory=dict)


@dataclass
class RoundSummary:
    index: int
    max_size: int
    train_example_count: int
    pseudo_example_count: int
    eval_accuracy: float
    per_size_accuracy: Dict[int, float]
    output_dir: Path
    composed_eval_accuracy: Optional[float] = None
    composed_eval_slices: Dict[str, SliceMetric] = field(default_factory=dict)
    pseudo_generation_stats: JsonDict = field(default_factory=dict)


def summary_to_payload(summary: RoundSummary, task: SummaryTask) -> JsonDict:
    composed_eval_slices = {
        slice_name: {
            "accuracy": sanitize_float(metric.accuracy),
            "count": int(metric.count),
            "per_size_accuracy": {
                str(size): sanitize_float(score)
                for size, score in metric.per_size_accuracy.items()
            },
        }
        for slice_name, metric in summary.composed_eval_slices.items()
    }
    payload: JsonDict = {
        "round": summary.index,
        "max_size": summary.max_size,
        "train_examples": summary.train_example_count,
        "pseudo_examples": summary.pseudo_example_count,
        "eval_accuracy": sanitize_float(summary.eval_accuracy),
        "per_size_accuracy": {str(size): sanitize_float(score) for size, score in summary.per_size_accuracy.items()},
        "composed_eval_accuracy": sanitize_float(summary.composed_eval_accuracy),
        "composed_eval_slices": composed_eval_slices,
        "pseudo_generation_stats": sanitize_json_value(summary.pseudo_generation_stats),
        "output_dir": str(summary.output_dir),
    }
    threshold = 0.90
    solved_sizes = [
        int(size)
        for size, score in summary.per_size_accuracy.items()
        if score is not None and not math.isnan(score) and score >= threshold
    ]
    payload["max_solved_size_at_90_accuracy"] = max(solved_sizes) if solved_sizes else None
    if isinstance(summary.pseudo_generation_stats, dict):
        candidate_total = summary.pseudo_generation_stats.get("candidate_total")
        retained_total = summary.pseudo_generation_stats.get("retained_total")
        if isinstance(candidate_total, (int, float)) and candidate_total:
            payload["pseudo_retention_rate"] = sanitize_float(float(retained_total) / float(candidate_total))
        else:
            payload["pseudo_retention_rate"] = None
    else:
        payload["pseudo_retention_rate"] = None
    payload["stitched_eval_accuracy"] = payload["composed_eval_accuracy"]
    payload.update(task.summary_payload_aliases(summary))
    return sanitize_json_value(payload)


def format_accuracy(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def summarize_round(summary: RoundSummary, task: SummaryTask) -> None:
    print(
        f"[ROUND {summary.index}] {task.size_label}<= {summary.max_size}: "
        f"train={summary.train_example_count} pseudo={summary.pseudo_example_count} "
        f"eval_acc={format_accuracy(summary.eval_accuracy)}",
        flush=True,
    )
    if summary.per_size_accuracy:
        breakdown = " ".join(
            f"{size}:{summary.per_size_accuracy[size]:.4f}" for size in sorted(summary.per_size_accuracy)
        )
        print(f"  per-{task.size_alias_singular} {breakdown}", flush=True)
    total_slice_count = sum(metric.count for metric in summary.composed_eval_slices.values())
    if total_slice_count > 0:
        only_all_slice = set(summary.composed_eval_slices) == {"all"}
        parts: list[str] = []
        if not only_all_slice:
            parts.append(f"all={format_accuracy(summary.composed_eval_accuracy)}")
        for slice_name, metric in summary.composed_eval_slices.items():
            parts.append(f"{slice_name}={format_accuracy(metric.accuracy)} (n={metric.count})")
        print(f"  composed-eval {' '.join(parts)}", flush=True)
    stats = summary.pseudo_generation_stats
    if isinstance(stats, dict) and "candidate_total" in stats:
        print(
            "  next-pseudo "
            f"retained={stats.get('retained_total', 0)}/{stats.get('candidate_total', 0)} "
            f"missing={stats.get('missing_total', 0)}",
            flush=True,
        )
