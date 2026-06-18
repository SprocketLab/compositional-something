from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self.analysis import (
    training_curve_bundle,
    training_curve_heatmaps,
    training_curve_logs,
    training_curve_plots,
    training_curve_results,
    training_curve_style,
)
from self.training_curve_notebook_utils import (
    build_run_summary,
    configure_plot_style,
    load_curve_bundle,
    load_round_metrics,
    load_submission_jobs,
    per_size_accuracy_frame,
    per_size_accuracy_frame_from_results,
    parse_training_log,
    plot_per_size_accuracy_heatmap_from_results,
)


def test_per_size_accuracy_frame_supports_addition_schema(tmp_path: Path):
    assert (
        per_size_accuracy_frame_from_results
        is training_curve_results.per_size_accuracy_frame_from_results
    )
    assert configure_plot_style is training_curve_style.configure_plot_style
    assert (
        plot_per_size_accuracy_heatmap_from_results
        is training_curve_plots.plot_per_size_accuracy_heatmap_from_results
    )
    assert (
        plot_per_size_accuracy_heatmap_from_results
        is training_curve_heatmaps.plot_per_size_accuracy_heatmap_from_results
    )

    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.98, "7": 0.91},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.99, "12": 0.73},
                },
            ]
        ),
        encoding="utf-8",
    )

    frame = per_size_accuracy_frame_from_results(results_path, run_label="filtered")

    observed = [
        (int(row.size), int(row.max_size), float(row.accuracy))
        for row in frame.itertuples(index=False)
    ]
    assert observed == [
        (3, 7, 1.0),
        (3, 12, 1.0),
        (4, 7, 0.98),
        (4, 12, 0.99),
        (7, 7, 0.91),
        (12, 12, 0.73),
    ]


def test_curve_bundle_helpers_keep_compatibility_and_load_fixture(tmp_path: Path):
    assert load_curve_bundle is training_curve_bundle.load_curve_bundle
    assert load_submission_jobs is training_curve_bundle.load_submission_jobs
    assert build_run_summary is training_curve_bundle.build_run_summary
    assert per_size_accuracy_frame is training_curve_bundle.per_size_accuracy_frame

    run_root = tmp_path / "runs" / "grid" / "addition"
    out_dir = tmp_path / "out" / "addition_compose_small"
    logs_dir = tmp_path / "logs"
    run_root.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (run_root / "submission_jobs.tsv").write_text(
        "\t".join(["job_id", "task", "mode", "budget", "out_dir"])
        + "\n"
        + "\t".join(["123", "addition", "compose", "small", str(out_dir)])
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "selfimp-grid-123.out").write_text(
        "\n".join(
            [
                "{'loss': 0.4, 'epoch': 0.25}",
                "{'eval_loss': 0.3, 'epoch': 0.5}",
                "{'train_loss': 0.35}",
                "[ROUND 2] eval_acc=0.72",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "self_improvement_results.json").write_text(
        json.dumps(
            [
                {
                    "round": 2,
                    "max_size": 12,
                    "train_examples": 100,
                    "pseudo_examples": 40,
                    "eval_accuracy": 0.72,
                    "composed_eval_accuracy": 0.68,
                    "pseudo_retention_rate": 0.5,
                    "max_solved_size_at_90_accuracy": 8,
                    "per_size_accuracy": {"12": 0.72},
                }
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_curve_bundle(run_root, logs_dir=logs_dir)
    assert bundle.jobs["job_id"].tolist() == ["123"]
    assert bundle.train_logs["loss"].tolist() == [0.4]
    assert bundle.validation_logs["validation_loss"].tolist() == [0.3]
    assert bundle.round_metrics["eval_accuracy"].tolist() == [0.72]

    summary = build_run_summary(bundle)
    assert summary[["job_id", "rounds", "final_train_examples", "final_eval_accuracy"]].to_dict(
        orient="records"
    ) == [
        {
            "job_id": "123",
            "rounds": 1,
            "final_train_examples": 100,
            "final_eval_accuracy": 0.72,
        }
    ]

    per_size = per_size_accuracy_frame(bundle, "addition", "compose", "small")
    assert [
        (int(row.size), int(row.max_size), float(row.accuracy))
        for row in per_size.itertuples(index=False)
    ] == [(12, 12, 0.72)]


def test_training_log_and_round_metric_helpers_keep_compatibility(tmp_path: Path):
    assert parse_training_log is training_curve_logs.parse_training_log
    assert load_round_metrics is training_curve_logs.load_round_metrics

    log_path = tmp_path / "slurm.out"
    log_path.write_text(
        "\n".join(
            [
                "{'loss': 0.4, 'epoch': 0.25, 'grad_norm': 1.5, 'learning_rate': 1e-5}",
                "{'eval_loss': 0.3, 'epoch': 0.5}",
                "{'train_loss': 0.35}",
                "[ROUND 2] eval_acc=0.72",
            ]
        ),
        encoding="utf-8",
    )
    train_frame, validation_frame = parse_training_log(log_path)
    assert train_frame.to_dict(orient="records") == [
        {
            "epoch": 0.25,
            "loss": 0.4,
            "grad_norm": 1.5,
            "learning_rate": 1e-5,
            "line_number": 1,
            "round": 2,
            "step_in_round": 1,
            "round_progress": 2.25,
            "train_loss_summary": 0.35,
        }
    ]
    assert validation_frame.to_dict(orient="records") == [
        {
            "epoch": 0.5,
            "validation_loss": 0.3,
            "line_number": 2,
            "round": 2,
            "step_in_round": 1,
            "round_progress": 2.5,
        }
    ]

    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 2,
                    "max_size": 12,
                    "train_examples": 100,
                    "pseudo_examples": 40,
                    "eval_accuracy": "0.72",
                    "composed_eval_accuracy": 0.68,
                    "pseudo_retention_rate": 0.5,
                    "max_solved_size_at_90_accuracy": 8,
                }
            ]
        ),
        encoding="utf-8",
    )
    metric_rows = load_round_metrics(results_path).to_dict(orient="records")
    assert metric_rows == [
        {
            "round": 2,
            "max_size": 12,
            "train_examples": 100,
            "pseudo_examples": 40,
            "eval_accuracy": 0.72,
            "composed_eval_accuracy": 0.68,
            "pseudo_retention_rate": 0.5,
            "max_solved_size_at_90_accuracy": 8,
        }
    ]


def test_plot_per_size_accuracy_heatmap_supports_addition_schema(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.98, "7": 0.91},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.99, "12": 0.73},
                },
            ]
        ),
        encoding="utf-8",
    )

    figure = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="addition",
        mode="with_carry_filtered",
        title="Addition Test",
    )

    assert figure.axes[0].get_ylabel() == "Digits in addends"


def test_plot_per_size_accuracy_heatmap_sparse_annotations_reduce_text_count(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_bits": 16,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 17)},
                },
                {
                    "round": 1,
                    "max_bits": 24,
                    "per_bit_accuracy": {str(bits): 0.80 for bits in range(8, 25)},
                },
                {
                    "round": 2,
                    "max_bits": 32,
                    "per_bit_accuracy": {str(bits): 0.60 for bits in range(8, 33)},
                },
            ]
        ),
        encoding="utf-8",
    )

    dense = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="run_length",
        mode="compose",
        annotate_mode="all",
    )
    sparse = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="run_length",
        mode="compose",
        annotate_mode="sparse",
        y_tick_stride=4,
        dense_y_ticks_through=16,
    )

    assert len(sparse.axes[0].texts) < len(dense.axes[0].texts)


def test_plot_per_size_accuracy_heatmap_supports_paper_layout_controls(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {str(digits): 0.99 for digits in range(3, 8)},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {str(digits): 0.75 for digits in range(3, 13)},
                },
            ]
        ),
        encoding="utf-8",
    )

    figure = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="addition",
        mode="with_carry_filtered",
        title="Should Be Hidden",
        annotate_mode="sparse",
        show_title=False,
        y_tick_stride=2,
        dense_y_ticks_through=12,
        font_scale=1.4,
        fixed_canvas_size=(7.4, 9.0),
    )

    assert tuple(round(value, 1) for value in figure.get_size_inches()) == (7.4, 9.0)
    assert figure.axes[0].get_title() == ""
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == [str(value) for value in range(3, 13)]
