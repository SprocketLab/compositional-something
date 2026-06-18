#!/usr/bin/env python3
"""Export appendix baseline heatmaps for addition and run-length."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from self.analysis.artifact_io import read_round_summaries


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "self").exists() and (candidate / "artifacts").exists():
            return candidate
    raise RuntimeError("Could not locate repository root.")


def load_rows(path: Path) -> List[dict]:
    return read_round_summaries(path)


def per_size_key(row: dict) -> str:
    if "per_digit_accuracy" in row:
        return "per_digit_accuracy"
    if "per_bit_accuracy" in row:
        return "per_bit_accuracy"
    return "per_size_accuracy"


def heatmap_matrix(rows: List[dict]) -> tuple[np.ndarray, List[int], List[int]]:
    values: Dict[tuple[int, int], float] = {}
    rounds = sorted({int(row["round"]) for row in rows})
    sizes = set()
    for row in rows:
        round_idx = int(row["round"])
        for size_text, value in (row.get(per_size_key(row)) or {}).items():
            if value is None:
                continue
            size = int(size_text)
            sizes.add(size)
            values[(size, round_idx)] = float(value)
    sorted_sizes = sorted(sizes)
    matrix = np.full((len(sorted_sizes), len(rounds)), np.nan)
    size_index = {size: idx for idx, size in enumerate(sorted_sizes)}
    round_index = {round_idx: idx for idx, round_idx in enumerate(rounds)}
    for (size, round_idx), value in values.items():
        matrix[size_index[size], round_index[round_idx]] = value
    return matrix, sorted_sizes, rounds


def plot_panels(
    panels: Dict[str, List[dict]],
    *,
    y_label: str,
    out_base: Path,
    y_tick_stride: int,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.35 * len(panels), 4.3), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    image = None
    for ax, (title, rows) in zip(axes, panels.items()):
        matrix, sizes, rounds = heatmap_matrix(rows)
        image = ax.imshow(matrix, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Self-improvement round", fontsize=13)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds, fontsize=11)
        tick_positions = list(range(0, len(sizes), y_tick_stride))
        if tick_positions and tick_positions[-1] != len(sizes) - 1:
            tick_positions.append(len(sizes) - 1)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([sizes[idx] for idx in tick_positions], fontsize=11)
        ax.grid(color="white", alpha=0.15, linewidth=0.4)
    axes[0].set_ylabel(y_label, fontsize=13)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.92)
        cbar.set_label("Held-out accuracy", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {out_base.with_suffix('.pdf')}")


def existing_results(root: Path, names: Iterable[str]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name in names:
        path = root / name / "self_improvement_results.json"
        if path.exists():
            paths[name] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addition-root",
        type=Path,
        default=Path("artifacts/runs/figure2_condition_sweep_20260419_212011/stage1/addition/expand2_replay5000_train10000"),
    )
    parser.add_argument("--run-length-root", type=Path, default=None)
    parser.add_argument("--run-length-direct-results", type=Path, default=None)
    parser.add_argument("--run-length-unfiltered-results", type=Path, default=None)
    parser.add_argument("--run-length-guarded-results", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=Path("icmlw26_comp-self-improvement/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = find_repo_root()
    figures_dir = (repo / args.figures_dir).resolve()
    addition_root = (repo / args.addition_root).resolve()

    addition_paths = existing_results(addition_root, ["direct", "with_carry", "with_carry_filtered"])
    missing_addition = {"direct", "with_carry", "with_carry_filtered"} - set(addition_paths)
    if missing_addition:
        raise FileNotFoundError(f"Missing addition results under {addition_root}: {sorted(missing_addition)}")
    plot_panels(
        {
            "Direct": load_rows(addition_paths["direct"]),
            "Unfiltered carry": load_rows(addition_paths["with_carry"]),
            "Filtered carry": load_rows(addition_paths["with_carry_filtered"]),
        },
        y_label="Digits in addends",
        out_base=figures_dir / "appendix_addition_baseline_heatmaps",
        y_tick_stride=2,
    )

    explicit_run_length = {
        "direct": args.run_length_direct_results,
        "unfiltered_compose": args.run_length_unfiltered_results,
        "guarded_compose": args.run_length_guarded_results,
    }
    if any(path is not None for path in explicit_run_length.values()):
        missing_args = [name for name, path in explicit_run_length.items() if path is None]
        if missing_args:
            raise ValueError(f"Explicit run-length paths require all three baselines; missing {missing_args}")
        run_length_paths = {
            name: ((repo / path).resolve() if not path.is_absolute() else path)
            for name, path in explicit_run_length.items()
            if path is not None
        }
        missing_files = [str(path) for path in run_length_paths.values() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing explicit run-length result files: {missing_files}")
        plot_panels(
            {
                "Direct": load_rows(run_length_paths["direct"]),
                "Unfiltered": load_rows(run_length_paths["unfiltered_compose"]),
                "Guarded": load_rows(run_length_paths["guarded_compose"]),
            },
            y_label="String length",
            out_base=figures_dir / "appendix_run_length_baseline_heatmaps",
            y_tick_stride=8,
        )
    elif args.run_length_root is not None:
        run_length_root = (repo / args.run_length_root).resolve()
        run_length_paths = existing_results(run_length_root, ["direct", "unfiltered_compose", "guarded_compose"])
        missing_run_length = {"direct", "unfiltered_compose", "guarded_compose"} - set(run_length_paths)
        if missing_run_length:
            raise FileNotFoundError(f"Missing run-length results under {run_length_root}: {sorted(missing_run_length)}")
        plot_panels(
            {
                "Direct": load_rows(run_length_paths["direct"]),
                "Unfiltered": load_rows(run_length_paths["unfiltered_compose"]),
                "Guarded": load_rows(run_length_paths["guarded_compose"]),
            },
            y_label="String length",
            out_base=figures_dir / "appendix_run_length_baseline_heatmaps",
            y_tick_stride=8,
        )


if __name__ == "__main__":
    main()
