"""Dataset context preparation for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set


@dataclass(frozen=True)
class NonAdaptiveDatasetContext:
    composed_eval_slices: Dict[str, List[Any]]
    eval_keys: Set[Any]


def prepare_nonadaptive_dataset_context(
    *,
    task: Any,
    base_splits: Mapping[str, Sequence[Any]],
    composed_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    composed_eval_examples: Sequence[Any],
    composed_eval_component_map: Any,
    print_fn: Callable[..., None] = print,
) -> NonAdaptiveDatasetContext:
    """Validate/report loaded datasets and derive evaluation lookup context."""
    if not base_splits["train"]:
        raise ValueError("Base training split is empty; cannot proceed.")

    print_fn(
        "[INFO] Dataset sizes -- base train: {} | composed pool: {} | eval: {} | composed eval: {}".format(
            len(base_splits["train"]),
            len(composed_examples),
            len(eval_examples),
            len(composed_eval_examples),
        ),
        flush=True,
    )

    composed_eval_slices = task.split_composed_eval_slices(composed_eval_examples, composed_eval_component_map)
    if composed_eval_examples and composed_eval_slices:
        counts_text = " | ".join(f"{name}: {len(examples)}" for name, examples in composed_eval_slices.items())
        print_fn(f"[INFO] Composed eval slices -- {counts_text}", flush=True)

    return NonAdaptiveDatasetContext(
        composed_eval_slices=composed_eval_slices,
        eval_keys=task.keys_for_examples(eval_examples),
    )
