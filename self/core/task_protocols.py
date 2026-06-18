"""Shared protocols and type aliases for task-agnostic self-improvement code."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from self.core.summaries import RoundSummary


JsonDict = Dict[str, Any]
SplitName = str
PredictionParser = Callable[..., Optional[str]]
SizeGetter = Callable[[Any], int]
KeyGetter = Callable[[Any], Any]


class PromptTargetExample(Protocol):
    def prompt(self) -> str:
        ...

    def target(self) -> str:
        ...


class SelfImprovementTask(Protocol):
    name: str
    size_label: str
    size_alias_singular: str
    size_alias_plural: str

    def validate_args(self, args: Any) -> None:
        ...

    def serialize_example(self, example: Any) -> JsonDict:
        ...

    def deserialize_example(self, payload: JsonDict) -> Any:
        ...

    def save_component_map(self, path: Path, component_map: Dict[Any, List[Any]]) -> None:
        ...

    def load_component_map(self, path: Path) -> Dict[Any, List[Any]]:
        ...

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[Any]], Dict[SplitName, set[Any]]]:
        ...

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[Any]],
        base_records: Dict[SplitName, set[Any]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Any]] = None,
    ) -> Tuple[List[Any], Dict[Any, List[Any]], set[Any]]:
        ...

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[Any]],
        base_records: Dict[SplitName, set[Any]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Any]] = None,
    ) -> Tuple[List[Any], Dict[Any, List[Any]], set[Any]]:
        ...

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Any],
    ) -> List[Any]:
        ...

    def split_composed_eval_slices(
        self,
        examples: Sequence[Any],
        component_map: Dict[Any, List[Any]],
    ) -> Dict[str, List[Any]]:
        ...

    def keys_for_examples(self, examples: Sequence[Any]) -> set[Any]:
        ...

    def rebuild_records(self, splits: Dict[SplitName, List[Any]]) -> Dict[SplitName, set[Any]]:
        ...

    def key_for_example(self, example: Any) -> Any:
        ...

    def clone_with_override(self, example: Any, override: Optional[str]) -> Any:
        ...

    def size_of(self, example: Any) -> int:
        ...

    def prediction_parser(self, text: str, example: Optional[Any] = None) -> Optional[str]:
        ...

    def derive_round_targets(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        composed_examples: Sequence[Any],
        component_map: Dict[Any, Any],
        target_max_size: int,
        base_examples: Sequence[Any],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[Any], int, JsonDict]:
        ...

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        ...

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        ...

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        ...

    def summary_payload_aliases(self, summary: RoundSummary) -> JsonDict:
        ...


def task_for_name(task_name: str) -> Any:
    from self.tasks.addition import AdditionTask
    from self.tasks.run_length import RunLengthTask

    if task_name == "addition":
        return AdditionTask()
    if task_name == "run_length":
        return RunLengthTask()
    raise ValueError(f"Unsupported task={task_name!r}.")
