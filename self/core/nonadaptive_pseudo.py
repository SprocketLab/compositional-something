"""Pseudo-label refresh/generation for the non-adaptive self-improvement loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from self.core.data_io import JsonDict, save_examples
from self.core.evaluation import resolve_max_new_tokens
from self.core.nonadaptive_schedule import NonAdaptiveSizeSchedule


@dataclass
class NonAdaptiveNextPseudoRound:
    composed_examples: List[Any]
    component_map: Any
    pseudo_examples: List[Any]
    pseudo_generation_stats: JsonDict
    missing_labels: int


def prepare_nonadaptive_next_pseudo_round(
    *,
    args: Any,
    task: Any,
    model: Any,
    tokenizer: Any,
    rng: Any,
    round_idx: int,
    round_dir: Path,
    train_examples: List[Any],
    base_splits: Dict[str, List[Any]],
    base_records: Dict[str, Any],
    composed_examples: List[Any],
    component_map: Any,
    composed_pool_path: Path,
    component_map_path: Path,
    metadata: JsonDict,
    eval_keys: set[Any],
    size_schedule: NonAdaptiveSizeSchedule,
    composed_min_size: int,
    final_max_size: int,
    train_base_decode_tokens: int,
    config_decode_max_new_tokens: int,
    eval_batch_size: int,
    dynamic_composed: bool,
    persist_metadata_fn: Callable[[], None],
    save_examples_fn: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None] = save_examples,
    resolve_max_new_tokens_fn: Callable[[Sequence[Any], int], int] = resolve_max_new_tokens,
    random_cls: Callable[[float], Any] = random.Random,
) -> NonAdaptiveNextPseudoRound:
    if round_idx >= args.num_expand_rounds:
        return NonAdaptiveNextPseudoRound(
            composed_examples=composed_examples,
            component_map=component_map,
            pseudo_examples=[],
            pseudo_generation_stats={},
            missing_labels=0,
        )

    if dynamic_composed:
        if composed_min_size <= final_max_size and args.expand_train_per_size > 0:
            refresh_label = f"round_{round_idx:02d}_next"
            composed_build_exclude = set(eval_keys)
            composed_build_exclude.update(task.keys_for_examples(train_examples))
            composed_examples, component_map, _ = task.prepare_composed_train(
                rng,
                args,
                base_splits={**base_splits, "train": train_examples},
                base_records=base_records,
                min_size=composed_min_size,
                max_size=size_schedule.target_max_size_for_round(round_idx),
                additional_exclude=composed_build_exclude if composed_build_exclude else None,
            )
            save_examples_fn(composed_pool_path, composed_examples, task.serialize_example)
            task.save_component_map(component_map_path, component_map)
            metadata["last_composed_refresh"] = refresh_label
            save_examples_fn(round_dir / "composed_pool_for_next_round.jsonl", composed_examples, task.serialize_example)
            task.save_component_map(round_dir / "composed_component_map_next_round.json", component_map)
        else:
            metadata["last_composed_refresh"] = f"skipped_round_{round_idx:02d}"
    persist_metadata_fn()

    target_max_size = size_schedule.target_max_size_for_round(round_idx)
    pseudo_rng = random_cls(rng.random())
    pseudo_decode_tokens = max(
        train_base_decode_tokens,
        resolve_max_new_tokens_fn(composed_examples, config_decode_max_new_tokens),
    )
    if args.pseudo_label_mode == "none":
        next_pseudo_examples: List[Any] = []
        missing_labels = 0
        pseudo_generation_stats: JsonDict = {
            "mode": "none",
            "target_max_size": int(target_max_size),
            "candidate_total": 0,
            "retained_total": 0,
            "missing_total": 0,
        }
    else:
        next_pseudo_examples, missing_labels, pseudo_generation_stats = task.derive_round_targets(
            model,
            tokenizer,
            composed_examples,
            component_map,
            target_max_size=target_max_size,
            base_examples=train_examples,
            batch_size=eval_batch_size,
            decode_max_new_tokens=pseudo_decode_tokens,
            args=args,
            rng=pseudo_rng,
        )
    if hasattr(args, "bit_composition_path_mode") and isinstance(pseudo_generation_stats, dict):
        pseudo_generation_stats.setdefault("bit_composition_path_mode", str(args.bit_composition_path_mode))
    save_examples_fn(round_dir / "pseudo_for_next_round.jsonl", next_pseudo_examples, task.serialize_example)
    if missing_labels > 0:
        print(
            f"[WARN] Round {round_idx}: skipped {missing_labels} composed examples without pseudo labels.",
            flush=True,
        )
    if not next_pseudo_examples:
        print(
            "[WARN] No pseudo-labeled examples generated; subsequent rounds will have no additional data.",
            flush=True,
        )

    return NonAdaptiveNextPseudoRound(
        composed_examples=composed_examples,
        component_map=component_map,
        pseudo_examples=next_pseudo_examples,
        pseudo_generation_stats=pseudo_generation_stats,
        missing_labels=missing_labels,
    )
