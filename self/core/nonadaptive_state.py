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
