"""Seed-model initialization and initial adaptive summary construction."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.core.controller_phases import PHASE_SEED
from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class SeedDispatchDeps:
    run_controller_worker_slurm: Callable[..., Mapping[str, Any]]
    float_or_nan: Callable[[Any], float]
    run_seed_phase: Callable[..., Any]


@dataclass(frozen=True)
class SeedDispatchResult:
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    init_final_accuracy: float
    summary_records: list[JsonDict]


def run_seed_dispatch(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    output_dir: Path,
    data_dir: Path,
    source_sizes: set[int],
    deps: SeedDispatchDeps,
) -> SeedDispatchResult:
    if args.dry_run_data_only:
        print("[INFO] dry_run_data_only enabled; skipping seed/candidate model work.", flush=True)
        current_checkpoint = args.model_name
        current_final_accuracy = math.nan
        current_per_size_accuracy: Mapping[int, float] = {}
        init_final_accuracy = args.init_final_accuracy if args.init_final_accuracy is not None else 0.0
    elif args.controller_execution_mode == "slurm":
        seed_output = deps.run_controller_worker_slurm(
            args=args,
            worker_dir=output_dir / "round_00" / "controller_worker",
            phase=PHASE_SEED,
            payload={
                "output_dir": str(output_dir),
                "source_examples_path": str(data_dir / "initial_train.jsonl"),
                "eval_examples_path": str(data_dir / "evaluation.jsonl"),
                "seed": args.seed,
            },
        )
        current_checkpoint = str(seed_output["current_checkpoint"])
        current_final_accuracy = deps.float_or_nan(seed_output.get("current_final_accuracy"))
        current_per_size_accuracy = {
            int(size): float(score)
            for size, score in dict(seed_output.get("current_per_size_accuracy", {})).items()
            if score is not None
        }
        init_final_accuracy = deps.float_or_nan(seed_output.get("init_final_accuracy"))
    else:
        seed_result = deps.run_seed_phase(
            args=args,
            task=task,
            config=config,
            source_examples=source_examples,
            eval_examples=eval_examples,
            output_dir=output_dir,
            seed=args.seed,
        )
        current_checkpoint = seed_result.current_checkpoint
        current_final_accuracy = seed_result.current_final_accuracy
        current_per_size_accuracy = seed_result.current_per_size_accuracy
        init_final_accuracy = seed_result.init_final_accuracy

    return SeedDispatchResult(
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        summary_records=build_initial_summary_records(
            current_checkpoint=current_checkpoint,
            source_sizes=source_sizes,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
        ),
    )


def build_initial_summary_records(
    *,
    current_checkpoint: str,
    source_sizes: set[int],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> list[JsonDict]:
    return [
        {
            "round": 0,
            "selected": None,
            "current_checkpoint": current_checkpoint,
            "source_sizes": sorted(source_sizes),
            "eval_accuracy": current_final_accuracy,
            "per_size_accuracy": current_per_size_accuracy,
            "init_final_accuracy": init_final_accuracy,
        }
    ]
