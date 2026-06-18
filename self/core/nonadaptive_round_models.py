"""Data contracts for non-adaptive round orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from self.core.data_io import JsonDict


@dataclass(frozen=True)
class NonAdaptiveRoundRuntimeContext:
    args: Any
    task: Any
    base_output_dir: Path
    base_splits: Dict[str, List[Any]]
    base_records: Dict[str, Any]
    eval_examples: List[Any]
    composed_eval_slices: Dict[str, List[Any]]
    composed_eval_component_map: Any
    composed_pool_path: Path
    component_map_path: Path
    metadata: JsonDict
    eval_keys: set[Any]
    size_schedule: Any
    composed_min_size: int
    final_max_size: int
    train_base_decode_tokens: int
    eval_decode_tokens: int
    composed_eval_decode_tokens: int
    config: Any
    data_collator: Any
    tokenizer: Any
    rng: Any
    new_run: bool
    dynamic_composed: bool
    save_model_policy: str
    resume_requested: bool
    resume_round: int
    stop_after_round: Optional[int]
    reset_each_round: bool
    use_recipe: bool
    recipe_name: str
    recipe_preset: Any
    summary_records: Dict[int, JsonDict]
    results_path: Path
    persist_metadata_fn: Callable[[], None]


@dataclass
class NonAdaptiveRoundRuntimeState:
    model: Any
    composed_examples: List[Any]
    component_map: Any
    pseudo_examples: List[Any]


@dataclass(frozen=True)
class NonAdaptiveRoundRuntimeResult:
    round_dir: Path
    skipped: bool
    should_break: bool
