#!/usr/bin/env python3
"""Plot self-improvement summary curves from one or more run directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from self.analysis.nonadaptive_artifacts import (
    load_self_improvement_rounds,
    resolve_self_improvement_results_path,
)


MetricSpec = Tuple[str, str]

METRIC_LABELS: Dict[str, str] = {
    "eval_accuracy": "Eval Accuracy",
    "stitched_no_boundary_carry_accuracy": "Stitched No-Boundary",
    "stitched_boundary_carry_accuracy": "Stitched Boundary",
    "stitched_unknown_accuracy": "Stitched Unknown",
    "composed_eval_accuracy": "Composed Eval Accuracy",
}

BASELINE_COLORS: Dict[str, str] = {
    "short-only": "#394867",
    "short_only": "#394867",
    "none": "#394867",
    "direct": "#b65e16",
    "compose": "#18794e",
    "with_carry": "#1f77b4",
    "with_carry_filtered": "#127c91",
    "compose_corrupt": "#c1121f",
    "corrupt": "#c1121f",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot self-improvement results from one or more run directories. "
            "Each input can be a run directory or a self_improvement_results.json file."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories or self_improvement_results.json files to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional legend labels matching the provided runs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric keys to plot. Defaults are inferred from the input results.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="self_improvement_figure.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Self-Improvement Accuracy by Expansion Round",
        help="Figure title.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("auto", "max_size", "max_digits", "max_bits", "max_ops", "round"),
        default="auto",
        help="Which round attribute to use for the x-axis.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include stitched_unknown_accuracy when it is available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI.",
    )
    return parser.parse_args(argv)


def resolve_results_path(raw_path: str) -> Path:
    path = resolve_self_improvement_results_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file at {path}.")
    return path


def load_records(path: Path) -> List[Dict[str, object]]:
    return sorted(load_self_improvement_rounds(path), key=lambda item: int(item["round"]))


def infer_label(path: Path) -> str:
    if path.name == "self_improvement_results.json":
        return path.parent.name
    return path.name


def value_or_none(entry: Dict[str, object], key: str) -> Optional[float]:
    value = entry.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_x_key(records: Sequence[Dict[str, object]], requested_key: str) -> str:
    if requested_key != "auto":
        return requested_key
    if not records:
        return "round"
    sample = records[0]
    for candidate in ("max_size", "max_digits", "max_bits", "max_ops", "round"):
        if candidate in sample:
            return candidate
    return "round"


def infer_metric_specs(
    records_by_run: Sequence[Sequence[Dict[str, object]]],
    requested_metrics: Optional[Sequence[str]],
    include_unknown: bool,
) -> List[MetricSpec]:
    if requested_metrics:
        return [(metric, METRIC_LABELS.get(metric, metric.replace("_", " ").title())) for metric in requested_metrics]

    available = set()
    for records in records_by_run:
        for entry in records:
            for key, value in entry.items():
                if key.endswith("accuracy") and value is not None:
                    available.add(key)

    if {
        "stitched_no_boundary_carry_accuracy",
        "stitched_boundary_carry_accuracy",
    }.issubset(available):
        metric_keys = [
            "eval_accuracy",
            "stitched_no_boundary_carry_accuracy",
            "stitched_boundary_carry_accuracy",
        ]
        if include_unknown and "stitched_unknown_accuracy" in available:
            metric_keys.append("stitched_unknown_accuracy")
        return [(metric, METRIC_LABELS.get(metric, metric.replace("_", " ").title())) for metric in metric_keys]

    metric_keys = ["eval_accuracy"]
    if "composed_eval_accuracy" in available:
        metric_keys.append("composed_eval_accuracy")
    return [(metric, METRIC_LABELS.get(metric, metric.replace("_", " ").title())) for metric in metric_keys]


def infer_color(label: str, index: int) -> Optional[str]:
    normalized = label.lower()
    for hint, color in BASELINE_COLORS.items():
        if hint in normalized:
            return color
    fallback = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    return fallback[index % len(fallback)]


def infer_x_label(x_key: str) -> str:
    return {
        "round": "Round",
        "max_bits": "Max Bits",
        "max_ops": "Max Ops",
        "max_digits": "Max Digits",
        "max_size": "Max Size",
    }.get(x_key, "Max Size")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    result_paths = [resolve_results_path(run) for run in args.runs]
    if args.labels is not None and len(args.labels) != len(result_paths):
        raise ValueError("--labels must have the same length as the provided runs.")

    labels = list(args.labels) if args.labels is not None else [infer_label(path) for path in result_paths]
    records_by_run = [load_records(path) for path in result_paths]
    metric_specs = infer_metric_specs(records_by_run, args.metrics, args.include_unknown)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
        }
    )

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.8 * len(metric_specs), 4.8), sharey=True)
    if len(metric_specs) == 1:
        axes = [axes]

    for axis, (metric_key, metric_title) in zip(axes, metric_specs):
        for index, (label, records) in enumerate(zip(labels, records_by_run)):
            x_key = resolve_x_key(records, args.x_axis)
            x_values: List[int] = []
            y_values: List[float] = []
            for entry in records:
                y_value = value_or_none(entry, metric_key)
                if y_value is None:
                    continue
                x_values.append(int(entry[x_key]))
                y_values.append(y_value)
            if not x_values:
                continue
            axis.plot(
                x_values,
                y_values,
                marker="o",
                markersize=6.0,
                linewidth=2.4,
                color=infer_color(label, index),
                label=label,
            )

        axis.set_title(metric_title)
        axis.set_xlabel(infer_x_label(resolve_x_key(records_by_run[0] if records_by_run else [], args.x_axis)))
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel("Accuracy")

    handles: List[object] = []
    legend_labels: List[str] = []
    for axis in axes:
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(len(legend_labels), 4), frameon=False)

    fig.suptitle(args.title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if handles else 0.96))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved figure to {output_path}")


if __name__ == "__main__":
    main()
