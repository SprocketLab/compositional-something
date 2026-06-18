from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self.analysis import seed_fit_artifacts, seed_fit_bundle, seed_fit_plots, summarize_seed_fit_grid
from self.analysis.artifacts import (
    SEED_FIT_RESULTS_FILE,
    discover_seed_fit_results,
    is_seed_fit_run_dir,
    load_seed_fit_result,
    load_seed_fit_results,
    resolve_seed_fit_results_path,
)
from self.seed_fit_curve_notebook_utils import (
    configure_plot_style,
    discover_seed_fit_results as notebook_discover_seed_fit_results,
    find_threshold_budget,
    load_seed_fit_bundle,
    plot_task_budget_curve,
    plot_task_loss_curves,
    summarize_task,
)


def _write_seed_fit_result(
    output_dir: Path,
    *,
    train_per_size: int,
    train_examples: int,
    test_min_accuracy: float,
) -> None:
    output_dir.mkdir(parents=True)
    payload = {
        "task": "addition",
        "output_dir": str(output_dir),
        "model_name": "tiny",
        "seed": 7,
        "initial_min_size": 3,
        "initial_max_size": 5,
        "initial_train_per_size": train_per_size,
        "initial_eval_per_size": 4,
        "train_examples": train_examples,
        "validation_examples": 12,
        "test_examples": 12,
        "training": {
            "learning_rate": 1e-4,
            "effective_batch_size": 8,
            "num_epochs": 2,
            "max_steps": 100,
            "final_epoch": 2.0,
            "approx_effective_epochs_from_steps": 2.0,
            "log_history": [
                {"step": 1, "epoch": 0.5, "loss": "0.8"},
                {"step": 2, "epoch": 1.0, "eval_loss": 0.4},
                {"step": 3, "epoch": 1.5, "loss": 0.3},
                {"step": 4, "epoch": 2.0, "eval_loss": "0.2"},
            ],
        },
        "results": {
            "train": {"accuracy": 0.99, "min_per_size_accuracy": 0.97},
            "validation": {"accuracy": 0.96, "min_per_size_accuracy": 0.94},
            "test": {"accuracy": 0.95, "min_per_size_accuracy": test_min_accuracy},
        },
        "meets_threshold": test_min_accuracy >= 0.95,
    }
    (output_dir / "seed_fit_results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_seed_fit_bundle_helpers_keep_compatibility_and_load_fixture(tmp_path: Path):
    assert load_seed_fit_bundle is seed_fit_bundle.load_seed_fit_bundle
    assert summarize_task is seed_fit_bundle.summarize_task
    assert find_threshold_budget is seed_fit_bundle.find_threshold_budget
    assert SEED_FIT_RESULTS_FILE == seed_fit_artifacts.SEED_FIT_RESULTS_FILE
    assert discover_seed_fit_results is seed_fit_artifacts.discover_seed_fit_results
    assert notebook_discover_seed_fit_results is seed_fit_artifacts.discover_seed_fit_results

    root = tmp_path / "seed_grid"
    _write_seed_fit_result(root / "low_budget", train_per_size=10, train_examples=30, test_min_accuracy=0.91)
    _write_seed_fit_result(root / "high_budget", train_per_size=20, train_examples=60, test_min_accuracy=0.96)

    low_results = root / "low_budget" / "seed_fit_results.json"
    high_results = root / "high_budget" / "seed_fit_results.json"
    assert is_seed_fit_run_dir(root / "low_budget") is True
    assert resolve_seed_fit_results_path(root / "low_budget") == low_results
    assert resolve_seed_fit_results_path(low_results) == low_results
    assert discover_seed_fit_results(root) == [high_results, low_results]
    assert load_seed_fit_result(root / "high_budget")["train_examples"] == 60
    assert [path for path, _ in load_seed_fit_results(root)] == [high_results, low_results]

    bundle = load_seed_fit_bundle([root])

    assert bundle.runs["run_name"].tolist() == ["low_budget", "high_budget"]
    assert bundle.runs["budget_label"].tolist() == ["10/size", "20/size"]
    assert bundle.runs["num_sizes"].tolist() == [3, 3]
    assert bundle.runs["final_train_loss"].tolist() == [0.3, 0.3]
    assert bundle.runs["final_validation_loss"].tolist() == [0.2, 0.2]
    assert bundle.train_logs["loss"].tolist() == [0.8, 0.3, 0.8, 0.3]
    assert bundle.validation_logs["eval_loss"].tolist() == [0.4, 0.2, 0.4, 0.2]

    summary = summarize_task(bundle, "addition")
    assert summary[["run_name", "budget_label", "test_min_per_size_accuracy"]].to_dict(orient="records") == [
        {"run_name": "low_budget", "budget_label": "10/size", "test_min_per_size_accuracy": 0.91},
        {"run_name": "high_budget", "budget_label": "20/size", "test_min_per_size_accuracy": 0.96},
    ]

    selected = find_threshold_budget(bundle, "addition", threshold=0.95)
    assert selected is not None
    assert selected["run_name"] == "high_budget"

    rows = summarize_seed_fit_grid.build_rows(root)
    assert [
        (row["initial_train_per_size"], row["test_min_per_size_accuracy"])
        for row in sorted(rows, key=lambda item: item["initial_train_per_size"])
    ] == [(10, 0.91), (20, 0.96)]
    assert summarize_seed_fit_grid.choose_best(rows, threshold=0.94)["addition"][
        "results_path"
    ] == str(high_results)


def test_seed_fit_plot_helpers_keep_compatibility_and_render_fixture(tmp_path: Path):
    assert configure_plot_style is seed_fit_plots.configure_plot_style
    assert plot_task_loss_curves is seed_fit_plots.plot_task_loss_curves
    assert plot_task_budget_curve is seed_fit_plots.plot_task_budget_curve

    root = tmp_path / "seed_grid"
    _write_seed_fit_result(root / "small_budget", train_per_size=10, train_examples=30, test_min_accuracy=0.91)
    bundle = load_seed_fit_bundle([root])

    configure_plot_style()
    loss_fig = plot_task_loss_curves(bundle, "addition")
    budget_fig = plot_task_budget_curve(bundle, "addition")

    assert [axis.get_ylabel() for axis in loss_fig.axes] == ["Train loss", "Validation loss"]
    assert budget_fig.axes[0].get_ylabel() == "Exact-match accuracy"

    plt.close(loss_fig)
    plt.close(budget_fig)
