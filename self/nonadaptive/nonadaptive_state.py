"""Filesystem and metadata state for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from self.core.data_io import JsonDict, ensure_dir, load_summary_records, sanitize_json_value


@dataclass(frozen=True)
class NonAdaptiveArtifactPaths:
    original_output_dir: Path
    base_output_dir: Path
    data_dir: Path
    metadata_path: Path
    results_path: Path
    base_train_path: Path
    base_val_path: Path
    base_test_path: Path
    composed_pool_path: Path
    component_map_path: Path
    eval_path: Path
    composed_eval_path: Path
    composed_eval_component_map_path: Path


@dataclass
class NonAdaptiveRunState:
    paths: NonAdaptiveArtifactPaths
    metadata: JsonDict
    existing_summaries: Dict[int, JsonDict]
    resume_requested: bool

    @property
    def new_run(self) -> bool:
        return not self.resume_requested or not self.paths.base_train_path.exists()

    def stored_value(self, *keys: str) -> Any:
        for key in keys:
            if key in self.metadata:
                return self.metadata[key]
        return None


def build_nonadaptive_artifact_paths(output_dir: str | Path, reset_each_round: bool) -> NonAdaptiveArtifactPaths:
    original_output_dir = Path(output_dir)
    base_output_dir = original_output_dir / "reset_each_round" if reset_each_round else original_output_dir
    data_dir = base_output_dir / "data"
    return NonAdaptiveArtifactPaths(
        original_output_dir=original_output_dir,
        base_output_dir=base_output_dir,
        data_dir=data_dir,
        metadata_path=data_dir / "metadata.json",
        results_path=base_output_dir / "self_improvement_results.json",
        base_train_path=data_dir / "initial_train.jsonl",
        base_val_path=data_dir / "initial_validation.jsonl",
        base_test_path=data_dir / "initial_test.jsonl",
        composed_pool_path=data_dir / "composed_pool.jsonl",
        component_map_path=data_dir / "composed_component_map.json",
        eval_path=data_dir / "evaluation.jsonl",
        composed_eval_path=data_dir / "composed_evaluation.jsonl",
        composed_eval_component_map_path=data_dir / "composed_evaluation_component_map.json",
    )


def prepare_nonadaptive_run_state(
    args: Any,
    *,
    reset_each_round: bool,
    json_module: Any,
    ensure_dir_fn: Callable[[Path], None] = ensure_dir,
    load_summary_records_fn: Callable[[Path], Dict[int, JsonDict]] = load_summary_records,
) -> NonAdaptiveRunState:
    paths = build_nonadaptive_artifact_paths(args.output_dir, reset_each_round)
    if reset_each_round:
        ensure_dir_fn(paths.original_output_dir)
        print(
            f"[INFO] reset_in_each_round enabled; writing artifacts to {paths.base_output_dir}",
            flush=True,
        )
    ensure_dir_fn(paths.base_output_dir)
    ensure_dir_fn(paths.data_dir)

    metadata: JsonDict = {}
    if paths.metadata_path.exists():
        with paths.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json_module.load(handle)
    resume_requested = bool(args.resume or args.resume_from_round is not None)
    existing_summaries = load_summary_records_fn(paths.results_path) if resume_requested else {}
    return NonAdaptiveRunState(
        paths=paths,
        metadata=metadata,
        existing_summaries=existing_summaries,
        resume_requested=resume_requested,
    )


def validate_loaded_nonadaptive_metadata(
    args: Any,
    task: Any,
    run_state: NonAdaptiveRunState,
    *,
    final_max_size: int,
    frontier_min_size: int | None,
    reset_each_round: bool,
    dynamic_composed: bool,
) -> None:
    metadata = run_state.metadata
    if not metadata:
        raise ValueError("metadata.json missing; cannot resume without dataset metadata.")

    stored_task = metadata.get("task")
    if stored_task is not None and stored_task != task.name:
        raise ValueError(f"Output directory contains task={stored_task!r}, but current run requests {task.name!r}.")

    stored_initial_min = run_state.stored_value("initial_min_size", f"initial_min_{task.size_alias_plural}")
    stored_initial_max = run_state.stored_value("initial_max_size", f"initial_max_{task.size_alias_plural}")
    stored_frontier_min = run_state.stored_value("frontier_min_size")
    stored_composed_max = run_state.stored_value("composed_max_size", f"composed_max_{task.size_alias_plural}")
    if stored_initial_min is None or stored_initial_max is None or stored_composed_max is None:
        raise ValueError("metadata.json is missing required size-range keys; please regenerate datasets.")
    if int(stored_initial_min) != args.initial_min_size:
        raise ValueError(
            f"initial_min_size mismatch (stored={stored_initial_min} requested={args.initial_min_size})."
        )
    if int(stored_initial_max) != args.initial_max_size:
        raise ValueError(
            f"initial_max_size mismatch (stored={stored_initial_max} requested={args.initial_max_size})."
        )
    normalized_stored_frontier_min = None if stored_frontier_min is None else int(stored_frontier_min)
    if normalized_stored_frontier_min != frontier_min_size:
        raise ValueError(
            f"frontier_min_size mismatch (stored={normalized_stored_frontier_min} requested={frontier_min_size})."
        )
    if final_max_size > int(stored_composed_max):
        raise ValueError(
            "Requested num_expand_rounds requires more sizes than available in stored composed data. "
            "Regenerate datasets with a larger range."
        )

    stored_reset_flag = bool(metadata.get("reset_each_round", False))
    if stored_reset_flag != reset_each_round:
        raise ValueError(
            "Output directory was created with a different reset_each_round setting. "
            "Please choose a different --output-dir to avoid mixing trajectories."
        )

    stored_refresh_mode = metadata.get("composed_refresh_mode", "static")
    if stored_refresh_mode not in ("dynamic", "static"):
        stored_refresh_mode = "dynamic" if dynamic_composed else "static"
    if stored_refresh_mode == "static" and dynamic_composed:
        raise ValueError(
            "Existing output directory was created with static composed pools but current run requests dynamic refresh."
        )
    if stored_refresh_mode == "dynamic" and not dynamic_composed:
        raise ValueError(
            "Existing output directory was created with dynamic composed pools but current run requests static refresh."
        )

    stored_composed_eval_per = run_state.stored_value(
        "composed_eval_per_size",
        f"composed_eval_per_{task.size_alias_singular}",
    )
    if stored_composed_eval_per is not None and int(stored_composed_eval_per) != args.composed_eval_per_size:
        raise ValueError(
            "composed_eval_per_size does not match stored datasets. Please regenerate datasets or use a matching value."
        )

    task.validate_loaded_metadata(args, metadata, final_max_size, dynamic_composed)


def persist_nonadaptive_metadata(
    metadata: JsonDict,
    metadata_path: Path,
    rng_state: tuple[Any, ...],
    *,
    json_module: Any,
    encode_rng_state_fn: Callable[[tuple[Any, ...]], JsonDict],
    sanitize_json_value_fn: Callable[[Any], Any] = sanitize_json_value,
) -> None:
    metadata["rng_state"] = encode_rng_state_fn(rng_state)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json_module.dump(sanitize_json_value_fn(metadata), handle, indent=2)


def write_nonadaptive_config_args(
    args: Any,
    base_output_dir: Path,
    *,
    json_module: Any,
    sanitize_json_value_fn: Callable[[Any], Any] = sanitize_json_value,
) -> None:
    with (base_output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
        json_module.dump(sanitize_json_value_fn(vars(args)), handle, indent=2)
