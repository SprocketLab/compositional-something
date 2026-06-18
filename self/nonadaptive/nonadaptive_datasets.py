"""Dataset generation/loading for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from self.core.data_io import JsonDict, load_examples, save_examples
from self.nonadaptive.nonadaptive_schedule import NonAdaptiveSizeSchedule
from self.nonadaptive.nonadaptive_state import NonAdaptiveRunState, validate_loaded_nonadaptive_metadata


@dataclass
class NonAdaptiveDatasets:
    base_splits: Dict[str, List[Any]]
    base_records: Dict[str, Any]
    composed_examples: List[Any]
    component_map: Any
    eval_examples: List[Any]
    composed_eval_examples: List[Any]
    composed_eval_component_map: Any
    metadata: JsonDict


def prepare_nonadaptive_datasets(
    args: Any,
    task: Any,
    rng: Any,
    run_state: NonAdaptiveRunState,
    *,
    size_schedule: NonAdaptiveSizeSchedule,
    final_max_size: int,
    composed_min_size: int,
    frontier_min_size: int | None,
    reset_each_round: bool,
    dynamic_composed: bool,
    persist_metadata_fn: Callable[[JsonDict], None],
    write_config_args_fn: Callable[[], None],
    save_examples_fn: Callable[[Any, List[Any], Callable[[Any], JsonDict]], None] = save_examples,
    load_examples_fn: Callable[[Any, Callable[[JsonDict], Any]], List[Any]] = load_examples,
    validate_loaded_metadata_fn: Callable[..., None] = validate_loaded_nonadaptive_metadata,
) -> NonAdaptiveDatasets:
    paths = run_state.paths
    if run_state.new_run:
        return _generate_nonadaptive_datasets(
            args,
            task,
            rng,
            run_state,
            size_schedule=size_schedule,
            final_max_size=final_max_size,
            composed_min_size=composed_min_size,
            frontier_min_size=frontier_min_size,
            reset_each_round=reset_each_round,
            dynamic_composed=dynamic_composed,
            save_examples_fn=save_examples_fn,
            persist_metadata_fn=persist_metadata_fn,
            write_config_args_fn=write_config_args_fn,
        )

    print(f"[INFO] Loading {task.name} datasets from disk.", flush=True)
    validate_loaded_metadata_fn(
        args,
        task,
        run_state,
        final_max_size=final_max_size,
        frontier_min_size=frontier_min_size,
        reset_each_round=reset_each_round,
        dynamic_composed=dynamic_composed,
    )

    base_splits = {
        "train": load_examples_fn(paths.base_train_path, task.deserialize_example),
        "validation": load_examples_fn(paths.base_val_path, task.deserialize_example),
        "test": load_examples_fn(paths.base_test_path, task.deserialize_example),
    }
    composed_examples = load_examples_fn(paths.composed_pool_path, task.deserialize_example)
    component_map = task.load_component_map(paths.component_map_path)
    eval_examples = load_examples_fn(paths.eval_path, task.deserialize_example)
    composed_eval_examples = load_examples_fn(paths.composed_eval_path, task.deserialize_example)
    composed_eval_component_map = task.load_component_map(paths.composed_eval_component_map_path)
    if not composed_eval_examples and args.composed_eval_per_size > 0:
        print(
            "[WARN] Held-out composed evaluation set is missing; composed slice metrics will be unavailable "
            "for this run. Regenerate datasets to enable them.",
            flush=True,
        )
    return NonAdaptiveDatasets(
        base_splits=base_splits,
        base_records=task.rebuild_records(base_splits),
        composed_examples=composed_examples,
        component_map=component_map,
        eval_examples=eval_examples,
        composed_eval_examples=composed_eval_examples,
        composed_eval_component_map=composed_eval_component_map,
        metadata=run_state.metadata,
    )


def _generate_nonadaptive_datasets(
    args: Any,
    task: Any,
    rng: Any,
    run_state: NonAdaptiveRunState,
    *,
    size_schedule: NonAdaptiveSizeSchedule,
    final_max_size: int,
    composed_min_size: int,
    frontier_min_size: int | None,
    reset_each_round: bool,
    dynamic_composed: bool,
    save_examples_fn: Callable[[Any, List[Any], Callable[[Any], JsonDict]], None],
    persist_metadata_fn: Callable[[JsonDict], None],
    write_config_args_fn: Callable[[], None],
) -> NonAdaptiveDatasets:
    paths = run_state.paths
    print(f"[INFO] Generating {task.name} datasets from scratch.", flush=True)
    reserved_eval_examples: List[Any] = []
    reserved_eval_keys: set[Any] = set()
    if getattr(args, "reserve_shared_eval_first", False) and args.eval_per_size > 0:
        reserved_eval_examples = task.prepare_eval_examples(
            rng,
            args,
            min_size=args.initial_min_size,
            max_size=final_max_size,
            exclude=set(),
        )
        reserved_eval_keys = task.keys_for_examples(reserved_eval_examples)
        setattr(args, "_initial_exclude_keys", reserved_eval_keys)
        print(
            f"[INFO] Reserved {len(reserved_eval_examples)} shared evaluation examples before dataset construction.",
            flush=True,
        )
    else:
        setattr(args, "_initial_exclude_keys", None)

    base_splits, base_records = task.prepare_initial_splits(rng, args)
    save_examples_fn(paths.base_train_path, base_splits["train"], task.serialize_example)
    save_examples_fn(paths.base_val_path, base_splits["validation"], task.serialize_example)
    save_examples_fn(paths.base_test_path, base_splits["test"], task.serialize_example)

    initial_train_examples = list(base_splits["train"])
    initial_dynamic_exclude = set(reserved_eval_keys)
    initial_dynamic_exclude.update(task.keys_for_examples(initial_train_examples))

    initial_composed_max_size = size_schedule.target_max_size_for_round(0)
    composed_examples, component_map, composed_keys = task.prepare_composed_train(
        rng,
        args,
        base_splits={**base_splits, "train": initial_train_examples},
        base_records=base_records,
        min_size=composed_min_size,
        max_size=initial_composed_max_size,
        additional_exclude=initial_dynamic_exclude if initial_dynamic_exclude else None,
    )
    save_examples_fn(paths.composed_pool_path, composed_examples, task.serialize_example)
    task.save_component_map(paths.component_map_path, component_map)

    composed_eval_exclude = set(reserved_eval_keys)
    composed_eval_exclude.update(composed_keys)
    composed_eval_examples, composed_eval_component_map, composed_eval_keys = task.prepare_composed_eval(
        rng,
        args,
        base_splits=base_splits,
        base_records=base_records,
        min_size=composed_min_size,
        max_size=final_max_size,
        additional_exclude=composed_eval_exclude if composed_eval_exclude else None,
    )
    save_examples_fn(paths.composed_eval_path, composed_eval_examples, task.serialize_example)
    task.save_component_map(paths.composed_eval_component_map_path, composed_eval_component_map)

    if reserved_eval_examples:
        eval_examples = reserved_eval_examples
    else:
        training_union = set().union(*base_records.values())
        training_union.update(composed_keys)
        training_union.update(composed_eval_keys)
        eval_examples = task.prepare_eval_examples(
            rng,
            args,
            min_size=args.initial_min_size,
            max_size=final_max_size,
            exclude=training_union,
        )
    save_examples_fn(paths.eval_path, eval_examples, task.serialize_example)

    metadata = {
        "task": task.name,
        "size_label": task.size_label,
        "initial_min_size": args.initial_min_size,
        "initial_max_size": args.initial_max_size,
        "frontier_min_size": frontier_min_size,
        "expand_num_size": args.expand_num_size,
        "expand_train_per_size": args.expand_train_per_size,
        "eval_per_size": args.eval_per_size,
        "composed_eval_per_size": args.composed_eval_per_size,
        "composed_max_size": final_max_size,
        "reset_each_round": reset_each_round,
        "composed_refresh_mode": args.composed_refresh_mode,
        "task_config": task.build_task_metadata(args, final_max_size),
    }
    metadata.update(task.metadata_aliases(args, final_max_size))
    metadata["last_composed_refresh"] = "initial_dynamic" if dynamic_composed else "static_initial"
    run_state.metadata = metadata
    persist_metadata_fn(metadata)
    write_config_args_fn()
    return NonAdaptiveDatasets(
        base_splits=base_splits,
        base_records=base_records,
        composed_examples=composed_examples,
        component_map=component_map,
        eval_examples=eval_examples,
        composed_eval_examples=composed_eval_examples,
        composed_eval_component_map=composed_eval_component_map,
        metadata=metadata,
    )
