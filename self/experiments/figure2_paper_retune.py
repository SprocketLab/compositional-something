#!/usr/bin/env python3
"""Run paper-facing Figure 2 schedule retuning and figure refresh."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt

from self.experiments.paper_schedule_selection import (
    DEFAULT_PAPER_SCHEDULE_SELECTION_PATH,
    choose_addition_candidate,
    score_addition_candidate,
)
from self.analysis.training_curve_notebook_utils import plot_per_size_accuracy_heatmap_from_results


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_LENGTH_SEED_SOURCE = (
    ROOT_DIR / "artifacts/runs/figure2_recipe_aggressive_safe_sched_mig_20260419_152939/run_length/seed/model"
)
DEFAULT_RUN_LENGTH_RESULTS = (
    ROOT_DIR / "artifacts/runs/figure2_recipe_aggressive_safe_sched_mig_20260419_152939/run_length/pilot/compose/self_improvement_results.json"
)
DEFAULT_ADDITION_SEED_MODEL = ROOT_DIR / "artifacts/models/addition_recipe_seed_best"
DEFAULT_RUN_LENGTH_SEED_LINK = ROOT_DIR / "artifacts/models/run_length_recipe_seed_best"
DEFAULT_PAPER_SCHEDULE_ENV = ROOT_DIR / "artifacts/paper/paper_schedule_selection.env"

ADDITION_CANDIDATES = (
    (4, 5000),
    (4, 8000),
    (3, 5000),
    (3, 8000),
    (2, 5000),
    (2, 8000),
)
ADDITION_FALLBACK = (1, 8000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retune paper-facing Figure 2 schedules and refresh figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "artifacts/runs" / f"figure2_paper_retune_{Path.cwd().name}",
        help="Root directory for pilot artifacts.",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Where to write the selected paper schedule JSON. Defaults to <output-dir>/paper_schedule_selection.json.",
    )
    parser.add_argument(
        "--paper-schedule-env",
        type=Path,
        default=None,
        help="Where to write the launcher-friendly environment file. Defaults to <output-dir>/paper_schedule_selection.env.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT_DIR / "icmlw26_comp-self-improvement/figures",
        help="Figure bundle destination.",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-length-force-rerun", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--run-length-seed-source", type=Path, default=DEFAULT_RUN_LENGTH_SEED_SOURCE)
    parser.add_argument("--addition-seed-model", type=Path, default=DEFAULT_ADDITION_SEED_MODEL)
    parser.add_argument("--run-length-seed-link", type=Path, default=DEFAULT_RUN_LENGTH_SEED_LINK)
    return parser


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[INFO] Command: {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=ROOT_DIR, check=True)


def ensure_symlink(link_path: Path, target_path: Path, *, dry_run: bool) -> None:
    if not target_path.exists():
        raise FileNotFoundError(f"Seed source does not exist: {target_path}")
    print(f"[INFO] Seed link: {link_path} -> {target_path}", flush=True)
    if dry_run:
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path.resolve())


def optional_batch_args(train_batch_size: Optional[int], eval_batch_size: Optional[int]) -> List[str]:
    args: List[str] = []
    if train_batch_size is not None:
        args += ["--per-device-train-batch-size", str(train_batch_size)]
    if eval_batch_size is not None:
        args += ["--per-device-eval-batch-size", str(eval_batch_size)]
    return args


def addition_candidate_dir(root: Path, expand_num_digits: int, train_per_digit: int) -> Path:
    return root / "addition" / "pilots" / f"expand{expand_num_digits}_train{train_per_digit}"


def run_run_length_safe(
    *,
    output_dir: Path,
    seed_model: Path,
    python_bin: str,
    seed: int,
    train_batch_size: Optional[int],
    eval_batch_size: Optional[int],
    dry_run: bool,
) -> Path:
    cmd = [
        python_bin,
        "-m",
        "self.run_length_self_improvement",
        "--model-name",
        str(seed_model),
        "--output-dir",
        str(output_dir),
        "--format-version",
        "legacy",
        "--initial-min-bits",
        "8",
        "--initial-max-bits",
        "16",
        "--initial-train-per-bit",
        "100000",
        "--initial-eval-per-bit",
        "50",
        "--num-expand-rounds",
        "8",
        "--expand-num-bits",
        "4",
        "--expand-train-per-bit",
        "1200",
        "--eval-per-bit",
        "50",
        "--composed-eval-per-bit",
        "50",
        "--pseudo-label-mode",
        "compose",
        "--recipe",
        "algorithmic_self_improve_v1",
        "--bucket-train-batches-by-bits",
        "--resume",
        "--seed",
        str(seed),
    ]
    cmd += optional_batch_args(train_batch_size, eval_batch_size)
    run_command(cmd, dry_run=dry_run)
    return output_dir / "self_improvement_results.json"


def run_addition_candidate(
    *,
    output_dir: Path,
    seed_model: Path,
    expand_num_digits: int,
    train_per_digit: int,
    python_bin: str,
    seed: int,
    train_batch_size: Optional[int],
    eval_batch_size: Optional[int],
    dry_run: bool,
) -> Path:
    cmd = [
        python_bin,
        "-m",
        "self.self_improvement",
        "--model-name",
        str(seed_model),
        "--output-dir",
        str(output_dir),
        "--recipe",
        "arithmetic_self_improve_v1",
        "--treat-seed-as-round-zero",
        "--seed-range-train-mode",
        "direct_pseudo",
        "--initial-min-digits",
        "3",
        "--initial-max-digits",
        "7",
        "--initial-train-per-digit",
        "0",
        "--initial-eval-per-digit",
        "200",
        "--num-expand-rounds",
        "8",
        "--expand-num-digits",
        str(expand_num_digits),
        "--seed-replay-train-per-digit",
        str(train_per_digit),
        "--expand-train-per-digit",
        str(train_per_digit),
        "--eval-per-digit",
        "100",
        "--composed-eval-per-digit",
        "50",
        "--gradient-accumulation-steps",
        "1",
        "--decode-max-new-tokens",
        "48",
        "--composed-refresh-mode",
        "dynamic",
        "--bucket-train-batches-by-digits",
        "--resume",
        "--seed",
        str(seed),
        "--pseudo-label-mode",
        "compose",
        "--composed-strategy",
        "with_carry_filtered",
        "--composition-error-percent",
        "0",
    ]
    cmd += optional_batch_args(train_batch_size, eval_batch_size)
    run_command(cmd, dry_run=dry_run)
    return output_dir / "self_improvement_results.json"


def render_paper_heatmap(results_path: Path, *, task: str, mode: str, basename: str, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = {
        "annotate_mode": "none",
        "show_title": False,
        "show_round_max_labels": False,
        "font_scale": 1.35,
        "fixed_canvas_size": (6.9, 6.2),
    }
    if task == "addition":
        common_kwargs["annotate_mode"] = "none"
        common_kwargs["y_tick_stride"] = 2
        common_kwargs["dense_y_ticks_through"] = 12
    else:
        common_kwargs["y_tick_stride"] = 16
        common_kwargs["dense_y_ticks_through"] = None

    fig = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task=task,
        mode=mode,
        title=None,
        **common_kwargs,
    )
    base = figure_dir / basename
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote figure bundle to {base}", flush=True)


def write_schedule_env(path: Path, selection: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_length = selection["run_length"]
    addition = selection["addition"]
    lines = [
        f"RUN_LENGTH_SEED_MODEL={shlex.quote(str(run_length['seed_model']))}",
        f"RUN_LENGTH_NUM_EXPAND_ROUNDS={shlex.quote(str(run_length['num_expand_rounds']))}",
        f"RUN_LENGTH_EXPAND_NUM_BITS={shlex.quote(str(run_length['expand_num_bits']))}",
        f"RUN_LENGTH_EXPAND_TRAIN_PER_BIT={shlex.quote(str(run_length['expand_train_per_bit']))}",
        f"ADDITION_NUM_EXPAND_ROUNDS={shlex.quote(str(addition['num_expand_rounds']))}",
        f"ADDITION_EXPAND_NUM_DIGITS={shlex.quote(str(addition['expand_num_digits']))}",
        f"ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT={shlex.quote(str(addition['seed_replay_train_per_digit']))}",
        f"ADDITION_EXPAND_TRAIN_PER_DIGIT={shlex.quote(str(addition['expand_train_per_digit']))}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.selection_json is None:
        args.selection_json = args.output_dir / DEFAULT_PAPER_SCHEDULE_SELECTION_PATH.name
    if args.paper_schedule_env is None:
        args.paper_schedule_env = args.output_dir / DEFAULT_PAPER_SCHEDULE_ENV.name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.paper_schedule_env.parent.mkdir(parents=True, exist_ok=True)

    ensure_symlink(args.run_length_seed_link, args.run_length_seed_source, dry_run=args.dry_run)
    if not args.addition_seed_model.exists() and not args.dry_run:
        raise FileNotFoundError(f"Addition recipe seed model is missing: {args.addition_seed_model}")

    if args.run_length_force_rerun or not DEFAULT_RUN_LENGTH_RESULTS.exists():
        run_length_results_path = run_run_length_safe(
            output_dir=args.output_dir / "run_length" / "pilot" / "compose",
            seed_model=args.run_length_seed_link,
            python_bin=args.python_bin,
            seed=args.seed,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            dry_run=args.dry_run,
        )
    else:
        run_length_results_path = DEFAULT_RUN_LENGTH_RESULTS
        print(f"[INFO] Reusing run-length safe compose results: {run_length_results_path}", flush=True)

    addition_result_paths: List[tuple[int, int, Path]] = []
    for expand_num_digits, train_per_digit in ADDITION_CANDIDATES:
        result_path = run_addition_candidate(
            output_dir=addition_candidate_dir(args.output_dir, expand_num_digits, train_per_digit),
            seed_model=args.addition_seed_model,
            expand_num_digits=expand_num_digits,
            train_per_digit=train_per_digit,
            python_bin=args.python_bin,
            seed=args.seed,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            dry_run=args.dry_run,
        )
        addition_result_paths.append((expand_num_digits, train_per_digit, result_path))

    if args.dry_run:
        print(f"[INFO] DRY_RUN=1; selection files not written: {args.selection_json}", flush=True)
        return

    addition_scores = [
        {
            **score_addition_candidate(result_path, expand_num_digits=expand_num_digits),
            "seed_replay_train_per_digit": train_per_digit,
            "expand_train_per_digit": train_per_digit,
        }
        for expand_num_digits, train_per_digit, result_path in addition_result_paths
    ]
    selected_addition = choose_addition_candidate(addition_scores)
    if selected_addition is None:
        fallback_digits, fallback_train = ADDITION_FALLBACK
        fallback_result_path = run_addition_candidate(
            output_dir=addition_candidate_dir(args.output_dir, fallback_digits, fallback_train),
            seed_model=args.addition_seed_model,
            expand_num_digits=fallback_digits,
            train_per_digit=fallback_train,
            python_bin=args.python_bin,
            seed=args.seed,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            dry_run=False,
        )
        fallback_score = {
            **score_addition_candidate(fallback_result_path, expand_num_digits=fallback_digits),
            "seed_replay_train_per_digit": fallback_train,
            "expand_train_per_digit": fallback_train,
        }
        addition_scores.append(fallback_score)
        selected_addition = choose_addition_candidate(addition_scores)
        if selected_addition is None:
            raise RuntimeError("Addition schedule retune did not produce any eligible paper-facing schedule.")

    selection_payload = {
        "selected_schedules": {
            "run_length": {
                "seed_model": str(args.run_length_seed_link),
                "results_path": str(run_length_results_path),
                "num_expand_rounds": 8,
                "expand_num_bits": 4,
                "expand_train_per_bit": 1200,
            },
            "addition": {
                "seed_model": str(args.addition_seed_model),
                "results_path": selected_addition["results_path"],
                "num_expand_rounds": 8,
                "expand_num_digits": int(selected_addition["expand_num_digits"]),
                "seed_replay_train_per_digit": int(selected_addition["seed_replay_train_per_digit"]),
                "expand_train_per_digit": int(selected_addition["expand_train_per_digit"]),
            },
        },
        "addition_candidates": addition_scores,
    }
    args.selection_json.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    write_schedule_env(args.paper_schedule_env, selection_payload["selected_schedules"])
    print(f"[INFO] Wrote selection JSON to {args.selection_json}", flush=True)
    print(f"[INFO] Wrote schedule env to {args.paper_schedule_env}", flush=True)

    if args.skip_render:
        return

    render_paper_heatmap(
        Path(selection_payload["selected_schedules"]["run_length"]["results_path"]),
        task="run_length",
        mode="compose",
        basename="run_length_self_improvement_heatmap",
        figure_dir=args.figure_dir,
    )
    render_paper_heatmap(
        Path(selection_payload["selected_schedules"]["addition"]["results_path"]),
        task="addition",
        mode="with_carry_filtered",
        basename="addition_filtered_heatmap",
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
