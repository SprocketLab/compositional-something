#!/usr/bin/env python3
"""Adaptive run setup, dispatch, initialization, orchestration, and finalization."""

from __future__ import annotations


# --- from run_setup.py ---
"""Run setup and lightweight trace-loading helpers."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def source_sizes_from_examples(task: Any, examples: Sequence[Any]) -> set[int]:
    return {int(task.size_of(example)) for example in examples}


def append_plan_log(plan_path: Path, lines: Iterable[str]) -> None:
    if not plan_path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with plan_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### Implementation Log: {timestamp}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")


def prepare_datasets(
    args: argparse.Namespace,
    task: Any,
    rng: random.Random,
) -> Tuple[Dict[str, List[Any]], Dict[str, set[Any]], List[Any], set[Any]]:
    base_splits, base_records = task.prepare_initial_splits(rng, args)
    training_union = set().union(*base_records.values())
    eval_examples = task.prepare_eval_examples(
        rng,
        args,
        min_size=args.initial_min_size,
        max_size=args.frontier_max_size,
        exclude=training_union,
    )
    eval_keys = task.keys_for_examples(eval_examples)
    return base_splits, base_records, eval_examples, eval_keys


def load_trace_jsonl(path: Path, builder: Any) -> List[Any]:
    if not path.exists():
        return []
    traces: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            traces.append(builder(json.loads(line)))
    return traces


# --- from run_initialization.py ---
"""Adaptive run output, dataset, and source-pool initialization."""

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from self.core.data_io import load_examples

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CheckpointManager:
    output_dir: Path
    keep_candidate_models: bool = False
    keep_proposal_grpo_checkpoints: bool = False

    def _is_protected_checkpoint(self, model_dir: Path, protected_checkpoints: Sequence[str] = ()) -> bool:
        try:
            resolved_model_dir = model_dir.resolve()
        except OSError:
            return False
        for checkpoint in protected_checkpoints:
            if not checkpoint:
                continue
            try:
                if Path(checkpoint).resolve() == resolved_model_dir:
                    return True
            except OSError:
                continue
        return False

    def _cleanup_model_dir(self, model_dir: Path) -> List[str]:
        if model_dir.name != "model":
            return []
        if not model_dir.exists():
            return []
        try:
            model_dir.resolve().relative_to(self.output_dir.resolve())
        except (ValueError, OSError):
            return []
        shutil.rmtree(model_dir, ignore_errors=True)
        return [str(model_dir)]

    def cleanup_unselected_candidates(
        self,
        *,
        metrics: Sequence[Any],
        selected: Optional[Any],
    ) -> List[str]:
        if self.keep_candidate_models:
            return []
        deleted: List[str] = []
        selected_dir = selected.model_dir if selected is not None else None
        for metric in metrics:
            model_dir = metric.model_dir
            if model_dir is None or model_dir == selected_dir:
                continue
            deleted.extend(self._cleanup_model_dir(model_dir))
        return deleted

    def cleanup_final_checkpoint(
        self,
        *,
        checkpoint: str,
        keep_final: bool,
    ) -> List[str]:
        if keep_final:
            return []
        model_dir = Path(checkpoint)
        if model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in model_dir.parts and self.keep_candidate_models:
            return []
        return self._cleanup_model_dir(model_dir)

    def cleanup_replaced_checkpoint(
        self,
        *,
        old_checkpoint: str,
        new_checkpoint: str,
        protected_checkpoints: Sequence[str] = (),
    ) -> List[str]:
        if old_checkpoint == new_checkpoint:
            return []
        old_model_dir = Path(old_checkpoint)
        new_model_dir = Path(new_checkpoint)
        if old_model_dir.name != "model":
            return []
        if not old_model_dir.exists() or not new_model_dir.exists():
            return []
        try:
            old_model_dir.resolve().relative_to(self.output_dir.resolve())
        except (ValueError, OSError):
            return []
        if old_model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in old_model_dir.parts and self.keep_candidate_models:
            return []
        if self._is_protected_checkpoint(old_model_dir, protected_checkpoints):
            return []
        return self._cleanup_model_dir(old_model_dir)


@dataclass(frozen=True)
class RunInitializationDeps:
    ensure_dir: Callable[[Path], None]
    make_config: Callable[[argparse.Namespace], TrainingConfig]
    prepare_datasets: Callable[..., tuple[Dict[str, list[Any]], Dict[str, set[Any]], list[Any], set[Any]]]
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class RunInitializationResult:
    output_dir: Path
    data_dir: Path
    checkpoint_manager: CheckpointManager
    config: TrainingConfig
    source_examples: list[Any]
    source_sizes: set[int]
    exclude_keys: set[Any]
    eval_examples: list[Any]
    prepared_start: Optional[JsonDict] = None


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _extract_prepared_start_state(run_dir: Path) -> JsonDict:
    summary = _load_json_file(run_dir / "summary.json")
    results = _load_json_file(run_dir / "adaptive_candidate_training_results.json")
    if not isinstance(results, list) or not results:
        raise ValueError(f"prepared_start_run_dir has no round records: {run_dir}")
    round0 = dict(results[0])
    current_checkpoint = str(summary.get("current_checkpoint") or round0.get("current_checkpoint") or "")
    if not current_checkpoint:
        raise ValueError(f"prepared_start_run_dir does not record current_checkpoint: {run_dir}")
    if not Path(current_checkpoint).exists():
        raise ValueError(f"prepared_start checkpoint does not exist: {current_checkpoint}")
    current_final_accuracy = float(round0.get("eval_accuracy"))
    current_per_size_accuracy = {
        int(size): float(score)
        for size, score in dict(round0.get("per_size_accuracy") or {}).items()
        if score is not None
    }
    return {
        "prepared_start_run_dir": str(run_dir),
        "current_checkpoint": current_checkpoint,
        "current_final_accuracy": current_final_accuracy,
        "current_per_size_accuracy": current_per_size_accuracy,
        "init_final_accuracy": current_final_accuracy,
        "prior_init_final_accuracy": summary.get("init_final_accuracy"),
        "prior_summary_path": str(run_dir / "summary.json"),
        "prior_results_path": str(run_dir / "adaptive_candidate_training_results.json"),
        "prior_round0": round0,
        "prior_summary": summary,
    }


def _initialize_prepared_start_run(
    *,
    args: argparse.Namespace,
    task: Any,
    output_dir: Path,
    data_dir: Path,
    config: TrainingConfig,
    deps: RunInitializationDeps,
) -> RunInitializationResult:
    run_dir = args.prepared_start_run_dir
    if run_dir is None:
        raise ValueError("prepared_start_run_dir is required.")
    prior_data_dir = run_dir / "data"
    for filename in (
        "initial_train.jsonl",
        "initial_validation.jsonl",
        "initial_test.jsonl",
        "evaluation.jsonl",
        "metadata.json",
    ):
        _copy_if_exists(prior_data_dir / filename, data_dir / filename)

    source_examples = load_examples(prior_data_dir / "initial_train.jsonl", task.deserialize_example)
    validation_examples = load_examples(prior_data_dir / "initial_validation.jsonl", task.deserialize_example)
    test_examples = load_examples(prior_data_dir / "initial_test.jsonl", task.deserialize_example)
    eval_examples = load_examples(prior_data_dir / "evaluation.jsonl", task.deserialize_example)
    prepared_start = _extract_prepared_start_state(run_dir)
    source_sizes = {
        int(size)
        for size in prepared_start.get("prior_summary", {}).get("source_sizes", [])
    } or source_sizes_from_examples(task, source_examples)
    exclude_keys = task.keys_for_examples(
        list(source_examples) + list(validation_examples) + list(test_examples) + list(eval_examples)
    )
    deps.write_json(output_dir / "prepared_start.json", prepared_start)

    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=args.keep_all_candidate_models,
        keep_proposal_grpo_checkpoints=args.keep_all_proposal_grpo_checkpoints,
    )
    return RunInitializationResult(
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_manager=checkpoint_manager,
        config=config,
        source_examples=source_examples,
        source_sizes=source_sizes,
        exclude_keys=exclude_keys,
        eval_examples=eval_examples,
        prepared_start=prepared_start,
    )


def initialize_adaptive_run(
    *,
    args: argparse.Namespace,
    task: Any,
    rng: random.Random,
    deps: RunInitializationDeps,
) -> RunInitializationResult:
    output_dir = args.output_dir
    deps.ensure_dir(output_dir)
    data_dir = output_dir / "data"
    deps.ensure_dir(data_dir)
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=args.keep_all_candidate_models,
        keep_proposal_grpo_checkpoints=args.keep_all_proposal_grpo_checkpoints,
    )

    config = deps.make_config(args)
    if args.prepared_start_run_dir is not None:
        return _initialize_prepared_start_run(
            args=args,
            task=task,
            output_dir=output_dir,
            data_dir=data_dir,
            config=config,
            deps=deps,
        )

    base_splits, base_records, eval_examples, eval_keys = deps.prepare_datasets(args, task, rng)
    deps.save_examples(data_dir / "initial_train.jsonl", base_splits["train"], task.serialize_example)
    deps.save_examples(data_dir / "initial_validation.jsonl", base_splits["validation"], task.serialize_example)
    deps.save_examples(data_dir / "initial_test.jsonl", base_splits["test"], task.serialize_example)
    deps.save_examples(data_dir / "evaluation.jsonl", eval_examples, task.serialize_example)
    deps.write_json(
        data_dir / "metadata.json",
        {
            "task": args.task,
            "initial_min_size": args.initial_min_size,
            "initial_max_size": args.initial_max_size,
            "frontier_min_size": args.frontier_min_size,
            "frontier_max_size": args.frontier_max_size,
            "initial_train_per_size": args.initial_train_per_size,
            "candidate_train_per_size": args.candidate_train_per_size,
            "eval_per_size": args.eval_per_size,
            "seed": args.seed,
        },
    )

    source_examples = list(base_splits["train"])
    source_sizes = set(range(args.initial_min_size, args.initial_max_size + 1))
    exclude_keys = set().union(*base_records.values())
    exclude_keys.update(eval_keys)

    return RunInitializationResult(
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_manager=checkpoint_manager,
        config=config,
        source_examples=source_examples,
        source_sizes=source_sizes,
        exclude_keys=exclude_keys,
        eval_examples=eval_examples,
    )


# --- from run_finalization.py ---
"""Adaptive run summary and plan-log finalization."""

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


WriteJsonFn = Callable[[Path, Any], None]
AppendPlanLogFn = Callable[[Path, Iterable[str]], None]
SanitizeJsonFn = Callable[[Any], Any]


def finalize_adaptive_run(
    *,
    args: Any,
    output_dir: Path,
    checkpoint_manager: Any,
    summary_records: Sequence[Mapping[str, Any]],
    selected_rounds: int,
    attempt_index: int,
    current_checkpoint: str,
    source_sizes: set[int],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_grpo_update_count: int,
    init_final_accuracy: float,
    write_json: WriteJsonFn,
    append_plan_log: AppendPlanLogFn,
    sanitize_json_value: SanitizeJsonFn,
) -> Any:
    """Write final adaptive artifacts and return the sanitized summary."""

    results_path = output_dir / "adaptive_candidate_training_results.json"
    max_selected_rounds = int(getattr(args, "max_selected_rounds", 0) or 0)
    max_selected_label = str(max_selected_rounds) if max_selected_rounds > 0 else "unlimited"
    write_json(results_path, summary_records)
    deleted_final_model_dirs = checkpoint_manager.cleanup_final_checkpoint(
        checkpoint=current_checkpoint,
        keep_final=bool(getattr(args, "keep_final_model_checkpoint", False)),
    )
    if deleted_final_model_dirs:
        write_json(
            output_dir / "deleted_final_model_dirs.json",
            {
                "current_checkpoint": current_checkpoint,
                "deleted_model_dirs": deleted_final_model_dirs,
                "keep_final_model_checkpoint": bool(getattr(args, "keep_final_model_checkpoint", False)),
            },
        )
    append_plan_log(
        args.plan_log_path,
        [
            "Implemented/running adaptive candidate-training loop.",
            (
                f"Task: `{args.task}`; max attempts: `{args.max_attempt_rounds}`; "
                f"max selected candidates: `{max_selected_label}`; attempts used: `{attempt_index}`; "
                f"candidates per attempt: `{args.num_candidates}`."
            ),
            f"Output directory: `{output_dir}`.",
            f"Proposal output schema: `{args.proposal_output_schema}`.",
            (
                "Proposal prompt action history: "
                f"`{args.proposal_prompt_action_history}`; max items: "
                f"`{args.proposal_prompt_action_history_max_items}`."
            ),
            f"Final source sizes tracked by driver: `{sorted(source_sizes)}`.",
            f"Selected proposal traces retained for replay: `{len(proposal_trace_buffer)}`.",
            f"Outcome trace target mode: `{args.outcome_trace_target_mode}`; retained outcome traces: `{len(outcome_trace_buffer)}`.",
            (
                f"Proposal GRPO updates: `{proposal_grpo_update_count}`; "
                f"steps/update: `{args.proposal_grpo_steps}`; reward mode: `{args.proposal_grpo_reward_mode}`; "
                f"zero-variance mode: `{args.proposal_grpo_zero_variance}`."
            ),
            (
                "Proposal GRPO KL: "
                f"old-policy coef `{args.proposal_grpo_kl_coef}`."
            ),
            f"Proposal GRPO action dedup: `{args.proposal_grpo_deduplicate_actions}`.",
            f"Proposal GRPO novelty beta: `{args.proposal_grpo_novelty_bonus_beta}`.",
            f"Source admission target-accuracy threshold: `{args.source_admission_target_accuracy_threshold}`.",
            (
                "Proposal update loss: `merged_agent`; "
                f"observation/format weights: `{args.proposal_observation_loss_weight}`/`{args.proposal_format_loss_weight}`."
            ),
            (
                "Synthetic proposal SFT: "
                f"`{args.synthetic_proposal_sft}`; examples: `{args.synthetic_proposal_sft_examples}`."
            ),
            f"Synthetic proposal seed mix: `{args.synthetic_proposal_sft_seed_mix}`.",
            f"Prepared start run dir: `{args.prepared_start_run_dir}`.",
            f"Keep final model checkpoint: `{args.keep_final_model_checkpoint}`.",
            f"Keep all proposal-GRPO checkpoints: `{args.keep_all_proposal_grpo_checkpoints}`.",
        ],
    )
    final_summary = {
        "task": args.task,
        "condition": args.condition,
        "output_dir": str(output_dir),
        "rounds_recorded": len(summary_records),
        "selected_rounds_completed": selected_rounds,
        "attempts_completed": attempt_index,
        "max_selected_rounds": max_selected_rounds,
        "max_attempt_rounds": args.max_attempt_rounds,
        "no_selection_patience": args.no_selection_patience,
        "num_candidates": args.num_candidates,
        "current_checkpoint": current_checkpoint,
        "source_sizes": sorted(source_sizes),
        "proposal_trace_buffer_size": len(proposal_trace_buffer),
        "proposal_output_schema": args.proposal_output_schema,
        "proposal_prompt_action_history": args.proposal_prompt_action_history,
        "proposal_prompt_action_history_max_items": args.proposal_prompt_action_history_max_items,
        "proposal_trace_buffer_path": str(output_dir / "selected_proposal_trace_buffer.jsonl"),
        "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
        "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
        "outcome_trace_target_mode": args.outcome_trace_target_mode,
        "outcome_trace_buffer_size": len(outcome_trace_buffer),
        "outcome_trace_buffer_path": str(output_dir / "outcome_trace_buffer.jsonl"),
        "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
        "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
        "proposal_grpo_update_count": proposal_grpo_update_count,
        "proposal_grpo_steps": args.proposal_grpo_steps,
        "proposal_grpo_learning_rate": args.proposal_grpo_learning_rate,
        "proposal_grpo_kl_coef": args.proposal_grpo_kl_coef,
        "proposal_grpo_zero_variance": args.proposal_grpo_zero_variance,
        "proposal_grpo_reward_mode": args.proposal_grpo_reward_mode,
        "proposal_grpo_span": args.proposal_grpo_span,
        "proposal_grpo_outcome_scale": args.proposal_grpo_outcome_scale,
        "proposal_grpo_fixed_baseline": args.proposal_grpo_fixed_baseline,
        "proposal_grpo_deduplicate_actions": args.proposal_grpo_deduplicate_actions,
        "proposal_grpo_novelty_bonus_beta": args.proposal_grpo_novelty_bonus_beta,
        "proposal_update": "merged_agent",
        "proposal_observation_loss_weight": args.proposal_observation_loss_weight,
        "proposal_format_loss_weight": args.proposal_format_loss_weight,
        "proposal_format_replay_max_examples": args.proposal_format_replay_max_examples,
        "prepared_start_run_dir": None if args.prepared_start_run_dir is None else str(args.prepared_start_run_dir),
        "synthetic_proposal_sft": args.synthetic_proposal_sft,
        "synthetic_proposal_sft_seed_mix": args.synthetic_proposal_sft_seed_mix,
        "synthetic_proposal_sft_examples": args.synthetic_proposal_sft_examples,
        "synthetic_proposal_sft_num_epochs": args.synthetic_proposal_sft_num_epochs,
        "synthetic_proposal_sft_learning_rate": args.synthetic_proposal_sft_learning_rate,
        "synthetic_proposal_sft_top_k": args.synthetic_proposal_sft_top_k,
        "synthetic_proposal_sft_temperature": args.synthetic_proposal_sft_temperature,
        "source_admission_target_accuracy_threshold": args.source_admission_target_accuracy_threshold,
        "keep_final_model_checkpoint": args.keep_final_model_checkpoint,
        "deleted_final_model_dirs": deleted_final_model_dirs,
        "keep_all_proposal_grpo_checkpoints": args.keep_all_proposal_grpo_checkpoints,
        "init_final_accuracy": init_final_accuracy,
        "results_path": str(results_path),
    }
    write_json(output_dir / "summary.json", final_summary)
    return sanitize_json_value(final_summary)


# --- from seed_dispatch.py ---
"""Seed-model initialization and initial adaptive summary construction."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence

from self.adaptive.phases import PHASE_SEED

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class SeedDispatchDeps:
    run_controller_worker_slurm: Callable[..., Mapping[str, Any]]
    float_or_nan: Callable[[Any], float]
    run_seed_phase: Callable[..., Any]
    build_synthetic_proposal_seed_mix: Callable[..., tuple[list[Any], JsonDict]]
    apply_synthetic_proposal_sft: Callable[..., tuple[str, JsonDict]]


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
    prepared_start: Mapping[str, Any] | None = None,
) -> SeedDispatchResult:
    if prepared_start is not None:
        current_checkpoint = str(prepared_start["current_checkpoint"])
        current_final_accuracy = float(prepared_start["current_final_accuracy"])
        current_per_size_accuracy = {
            int(size): float(score)
            for size, score in dict(prepared_start.get("current_per_size_accuracy") or {}).items()
        }
        init_final_accuracy = float(prepared_start.get("init_final_accuracy", current_final_accuracy))
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
                prepared_start=prepared_start,
            ),
        )

    synthetic_seed_mix_metrics = None
    seed_source_examples = list(source_examples)
    if (
        not args.dry_run_data_only
        and args.controller_execution_mode == "local"
        and bool(getattr(args, "synthetic_proposal_sft_seed_mix", False))
        and int(getattr(args, "synthetic_proposal_sft_examples", 0)) > 0
    ):
        seed_source_examples, synthetic_seed_mix_metrics = deps.build_synthetic_proposal_seed_mix(
            args=args,
            output_dir=output_dir / "round_00" / "synthetic_seed_mix",
            source_examples=source_examples,
            source_sizes=source_sizes,
            seed=args.seed + 3571,
        )

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
            source_examples=seed_source_examples,
            eval_examples=eval_examples,
            output_dir=output_dir,
            seed=args.seed,
        )
        current_checkpoint = seed_result.current_checkpoint
        current_final_accuracy = seed_result.current_final_accuracy
        current_per_size_accuracy = seed_result.current_per_size_accuracy
        init_final_accuracy = seed_result.init_final_accuracy

    synthetic_metrics = None
    if (
        not args.dry_run_data_only
        and bool(getattr(args, "synthetic_proposal_sft", False))
        and not bool(getattr(args, "synthetic_proposal_sft_seed_mix", False))
        and int(getattr(args, "synthetic_proposal_sft_examples", 0)) > 0
    ):
        next_checkpoint, synthetic_metrics = deps.apply_synthetic_proposal_sft(
            args=args,
            task=task,
            source_checkpoint=current_checkpoint,
            output_dir=output_dir / "round_00" / "synthetic_proposal_sft",
            eval_examples=eval_examples,
            source_sizes=source_sizes,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            config=config,
            seed=args.seed + 7919,
        )
        if not synthetic_metrics.get("skipped", True):
            current_checkpoint = next_checkpoint
            current_final_accuracy = float(synthetic_metrics["post_sft_eval_accuracy"])
            current_per_size_accuracy = {
                int(size): float(score)
                for size, score in dict(synthetic_metrics["post_sft_per_size_accuracy"]).items()
            }

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
            synthetic_proposal_sft=synthetic_metrics,
            synthetic_proposal_sft_seed_mix=synthetic_seed_mix_metrics,
        ),
    )


def build_initial_summary_records(
    *,
    current_checkpoint: str,
    source_sizes: set[int],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    synthetic_proposal_sft: Mapping[str, Any] | None = None,
    synthetic_proposal_sft_seed_mix: Mapping[str, Any] | None = None,
    prepared_start: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    record: JsonDict = {
        "round": 0,
        "selected": None,
        "current_checkpoint": current_checkpoint,
        "source_sizes": sorted(source_sizes),
        "eval_accuracy": current_final_accuracy,
        "per_size_accuracy": current_per_size_accuracy,
        "init_final_accuracy": init_final_accuracy,
    }
    if synthetic_proposal_sft is not None:
        record["synthetic_proposal_sft"] = dict(synthetic_proposal_sft)
    if synthetic_proposal_sft_seed_mix is not None:
        record["synthetic_proposal_sft_seed_mix"] = dict(synthetic_proposal_sft_seed_mix)
    if prepared_start is not None:
        record["prepared_start"] = {
            "prepared_start_run_dir": prepared_start.get("prepared_start_run_dir"),
            "prior_init_final_accuracy": prepared_start.get("prior_init_final_accuracy"),
            "prior_summary_path": prepared_start.get("prior_summary_path"),
            "prior_results_path": prepared_start.get("prior_results_path"),
        }
    return [record]


# --- from round_model_dispatch.py ---
"""Round-model local/Slurm dispatch for adaptive attempts."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence

from self.adaptive.phases import PHASE_ROUND_MODEL
from self.core.models import CandidateWorkItem
from self.adaptive.proposal import PromptBundle

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RoundModelDispatchDeps:
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]
    write_key_set: Callable[[Path, set[Any]], None]
    run_controller_worker_slurm: Callable[..., Mapping[str, Any]]
    float_or_nan: Callable[[Any], float]
    load_json: Callable[[Path], Any]
    work_item_from_worker_payload: Callable[..., CandidateWorkItem]
    run_round_model_phase: Callable[..., Any]


@dataclass(frozen=True)
class RoundModelDispatchResult:
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    prompt: PromptBundle
    proposal_results: Sequence[Mapping[str, Any]]
    work_items: Sequence[CandidateWorkItem]


def run_round_model_dispatch(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    current_checkpoint: str,
    round_dir: Path,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    exclude_keys: set[Any],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    init_final_accuracy: float,
    extra_aggregate_metrics: Mapping[str, Any] | None = None,
    deps: RoundModelDispatchDeps,
) -> RoundModelDispatchResult:
    if args.controller_execution_mode == "slurm":
        return _run_round_model_slurm(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            round_dir=round_dir,
            source_examples=source_examples,
            eval_examples=eval_examples,
            exclude_keys=exclude_keys,
            source_sizes=source_sizes,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            init_final_accuracy=init_final_accuracy,
            extra_aggregate_metrics=extra_aggregate_metrics,
            deps=deps,
        )

    round_result = deps.run_round_model_phase(
        args=args,
        task=task,
        config=config,
        current_checkpoint=current_checkpoint,
        round_dir=round_dir,
        source_examples=source_examples,
        eval_examples=eval_examples,
        exclude_keys=exclude_keys,
        source_sizes=source_sizes,
        selected_round_for_prompt=selected_round_for_prompt,
        attempt_index=attempt_index,
        selected_rounds=selected_rounds,
        consecutive_no_selection=consecutive_no_selection,
        init_final_accuracy=init_final_accuracy,
        extra_aggregate_metrics=extra_aggregate_metrics,
        seed=args.seed + attempt_index * 7919,
    )
    return RoundModelDispatchResult(
        current_final_accuracy=round_result.current_final_accuracy,
        current_per_size_accuracy=round_result.current_per_size_accuracy,
        prompt=round_result.prompt,
        proposal_results=round_result.proposal_results,
        work_items=round_result.work_items,
    )


def _run_round_model_slurm(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    round_dir: Path,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    exclude_keys: set[Any],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    init_final_accuracy: float,
    extra_aggregate_metrics: Mapping[str, Any] | None = None,
    deps: RoundModelDispatchDeps,
) -> RoundModelDispatchResult:
    controller_input_dir = round_dir / "controller_worker" / "inputs"
    controller_input_dir.mkdir(parents=True, exist_ok=True)
    source_examples_path = controller_input_dir / "source_examples.jsonl"
    eval_examples_path = controller_input_dir / "eval_examples.jsonl"
    exclude_keys_path = controller_input_dir / "exclude_keys.json"
    deps.save_examples(source_examples_path, source_examples, task.serialize_example)
    deps.save_examples(eval_examples_path, eval_examples, task.serialize_example)
    deps.write_key_set(exclude_keys_path, exclude_keys)
    round_output = deps.run_controller_worker_slurm(
        args=args,
        worker_dir=round_dir / "controller_worker",
        phase=PHASE_ROUND_MODEL,
        payload={
            "current_checkpoint": current_checkpoint,
            "round_dir": str(round_dir),
            "source_examples_path": str(source_examples_path),
            "eval_examples_path": str(eval_examples_path),
            "exclude_keys_path": str(exclude_keys_path),
            "source_sizes": sorted(source_sizes),
            "selected_round_for_prompt": selected_round_for_prompt,
            "attempt_index": attempt_index,
            "selected_rounds": selected_rounds,
            "consecutive_no_selection": consecutive_no_selection,
            "init_final_accuracy": init_final_accuracy,
            "extra_aggregate_metrics": dict(extra_aggregate_metrics or {}),
            "seed": args.seed + attempt_index * 7919,
        },
    )
    prompt_payload = deps.load_json(Path(round_output["prompt_path"]))
    prompt = PromptBundle(
        system=str(prompt_payload.get("system", "")),
        user=str(prompt_payload.get("user", "")),
    )
    proposal_results = deps.load_json(
        Path(round_output.get("proposal_grpo_results_path") or round_output["proposal_results_path"])
    )
    work_items = [
        deps.work_item_from_worker_payload(payload=item_payload, task=task)
        for item_payload in round_output.get("work_items", [])
    ]
    return RoundModelDispatchResult(
        current_final_accuracy=deps.float_or_nan(round_output.get("current_final_accuracy")),
        current_per_size_accuracy={
            int(size): float(score)
            for size, score in dict(round_output.get("current_per_size_accuracy", {})).items()
            if score is not None
        },
        prompt=prompt,
        proposal_results=proposal_results,
        work_items=work_items,
    )


# --- from run_orchestration.py ---
"""High-level adaptive run orchestration."""

import argparse
import random
from typing import Any, Dict

from self.adaptive.attempts import AttemptLoopDeps, run_adaptive_attempt_loop
from self.adaptive.attempts import AttemptPromptDeps
from self.adaptive.attempts import AttemptOutcomeDeps
from self.adaptive.attempts import DryRunAttemptDeps


JsonDict = Dict[str, Any]


def run_adaptive_candidate_training(args: argparse.Namespace, deps: AdaptiveRunDeps) -> JsonDict:
    import torch
    from transformers import set_seed

    args = deps.normalize_args(args)
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] No precision flag provided; defaulting to bf16 on CUDA.", flush=True)
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    set_seed(args.seed)
    rng = random.Random(args.seed)
    run_inputs = initialize_adaptive_run(
        args=args,
        task=task,
        rng=rng,
        deps=RunInitializationDeps(
            ensure_dir=deps.ensure_dir,
            make_config=deps.make_config,
            prepare_datasets=deps.prepare_datasets,
            save_examples=deps.save_examples,
            write_json=deps.write_json,
        ),
    )
    output_dir = run_inputs.output_dir
    data_dir = run_inputs.data_dir
    checkpoint_manager = run_inputs.checkpoint_manager
    config = run_inputs.config
    source_examples = run_inputs.source_examples
    source_sizes = run_inputs.source_sizes
    exclude_keys = run_inputs.exclude_keys
    eval_examples = run_inputs.eval_examples

    seed_result = run_seed_dispatch(
        args=args,
        task=task,
        config=config,
        source_examples=source_examples,
        eval_examples=eval_examples,
        output_dir=output_dir,
        data_dir=data_dir,
        source_sizes=source_sizes,
        prepared_start=run_inputs.prepared_start,
        deps=SeedDispatchDeps(
            run_controller_worker_slurm=deps.run_controller_worker_slurm,
            float_or_nan=deps.float_or_nan,
            run_seed_phase=deps.run_seed_phase,
            build_synthetic_proposal_seed_mix=deps.build_synthetic_proposal_seed_mix,
            apply_synthetic_proposal_sft=deps.apply_synthetic_proposal_sft,
        ),
    )
    current_checkpoint = seed_result.current_checkpoint
    current_final_accuracy = seed_result.current_final_accuracy
    current_per_size_accuracy = seed_result.current_per_size_accuracy
    init_final_accuracy = seed_result.init_final_accuracy
    summary_records = seed_result.summary_records

    loop_result = run_adaptive_attempt_loop(
        args=args,
        task=task,
        config=config,
        rng=rng,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        source_examples=source_examples,
        source_sizes=source_sizes,
        exclude_keys=exclude_keys,
        eval_examples=eval_examples,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        summary_records=summary_records,
        deps=AttemptLoopDeps(
            ensure_dir=deps.ensure_dir,
            build_attempt_prompt=deps.build_attempt_prompt,
            write_json=deps.write_json,
            run_dry_attempt=deps.run_dry_attempt,
            run_round_model_dispatch=deps.run_round_model_dispatch,
            train_candidate_metrics=deps.train_candidate_metrics,
            select_candidate=deps.select_candidate,
            write_round_trace=deps.write_round_trace,
            handle_attempt_outcome=deps.handle_attempt_outcome,
            attempt_prompt_deps=AttemptPromptDeps(
                render_config_prompt=deps.render_config_prompt,
            ),
            dry_run_attempt_deps=DryRunAttemptDeps(
                load_fixture_proposals=deps.load_fixture_proposals,
                rows_for_round=deps.rows_for_round,
                validate_proposal_rows=deps.validate_proposal_rows,
                build_candidate_work_items=deps.build_candidate_work_items,
                write_json=deps.write_json,
            ),
            round_model_dispatch_deps=RoundModelDispatchDeps(
                save_examples=deps.save_examples,
                write_key_set=deps.write_key_set,
                run_controller_worker_slurm=deps.run_controller_worker_slurm,
                float_or_nan=deps.float_or_nan,
                load_json=deps.load_json,
                work_item_from_worker_payload=deps.work_item_from_worker_payload,
                run_round_model_phase=deps.run_round_model_phase,
            ),
            attempt_outcome_deps=AttemptOutcomeDeps(
                build_round_outcome_trace_examples=deps.build_round_outcome_trace_examples,
                build_selected_proposal_trace_example=deps.build_selected_proposal_trace_example,
                apply_or_dispatch_proposal_grpo_update=deps.apply_or_dispatch_proposal_grpo_update,
                write_json=deps.write_json,
                write_trace_jsonl=deps.write_trace_jsonl,
                save_examples=deps.save_examples,
            ),
        ),
    )

    return finalize_adaptive_run(
        args=args,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        summary_records=summary_records,
        selected_rounds=loop_result.selected_rounds,
        attempt_index=loop_result.attempt_index,
        current_checkpoint=loop_result.current_checkpoint,
        source_sizes=loop_result.source_sizes,
        proposal_trace_buffer=loop_result.proposal_trace_buffer,
        outcome_trace_buffer=loop_result.outcome_trace_buffer,
        proposal_grpo_update_count=loop_result.proposal_grpo_update_count,
        init_final_accuracy=init_final_accuracy,
        write_json=deps.write_json,
        append_plan_log=deps.append_plan_log,
        sanitize_json_value=deps.sanitize_json_value,
    )
