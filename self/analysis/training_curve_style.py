"""Shared style and label helpers for training-curve analysis plots."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt


BASELINE_COLORS: Dict[str, str] = {
    "short_only": "#394867",
    "short-only": "#394867",
    "none": "#394867",
    "direct": "#b65e16",
    "compose": "#18794e",
    "with_carry": "#1f77b4",
    "with_carry_filtered": "#127c91",
    "compose_corrupt": "#c1121f",
}

BUDGET_ORDER = ["small", "medium", "large"]
MODE_ORDER = ["none", "direct", "compose", "compose_corrupt"]


def configure_plot_style() -> None:
    """Set legible, paper-friendly plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.3,
            "lines.markersize": 4.5,
        }
    )


def mode_label(mode: str) -> str:
    labels = {
        "short_only": "Short-only",
        "none": "Short-only",
        "direct": "Direct",
        "compose": "Compose",
        "with_carry": "With Carry",
        "with_carry_filtered": "Filtered Compose",
        "compose_corrupt": "Compose-Corrupt",
    }
    return labels.get(mode, mode)
