"""Compatibility imports for training-curve notebook helpers."""

from __future__ import annotations

from self.analysis.training_curve_bundle import (
    CurveBundle,
    build_run_summary,
    get_job_record,
    load_curve_bundle,
    load_submission_jobs,
    per_size_accuracy_frame,
)
from self.analysis.training_curve_logs import (
    ROUND_PATTERN,
    _to_float,
    load_round_metrics,
    parse_training_log,
)
from self.analysis.training_curve_heatmaps import (
    _should_annotate_sparse_cell,
    _visible_tick_indices,
    plot_per_size_accuracy_heatmap,
    plot_per_size_accuracy_heatmap_from_results,
)
from self.analysis.training_curve_plots import (
    plot_self_improvement_comparison_curve,
    plot_task_curves,
    save_figure_bundle,
)
from self.analysis.training_curve_results import (
    load_round_payload,
    per_size_accuracy_frame_from_results,
    resolve_results_path,
    round_summary_frame,
)
from self.analysis.training_curve_style import (
    BASELINE_COLORS,
    BUDGET_ORDER,
    MODE_ORDER,
    configure_plot_style,
    mode_label,
)
