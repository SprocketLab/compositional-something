"""Task-specific adapters for compositional self-improvement."""

from self.tasks.addition import (
    AdditionTask,
    corrupt_numeric_target,
    get_boundary_carry_status,
    prepare_addition_composed_eval,
    prepare_addition_composed_train,
    prepare_addition_eval_examples,
    prepare_addition_initial_splits,
    split_addition_examples_by_boundary_status,
)
from self.tasks.multiplication import (
    MultiplicationTask,
)
from self.tasks.multiplication_data import (
    MultiplicationExample,
    build_multiplication_component_payload,
    build_multiplication_long_dataset,
    build_multiplication_seed_dataset,
    clone_multiplication_with_override,
    multiplication_key,
)
from self.tasks.run_length import RunLengthTask
from self.tasks.run_length_data import (
    RunLengthExample,
    bucket_run_length_by_bits,
    build_run_length_composed_dataset,
    build_run_length_length_bucket_dataset,
    clone_run_length_with_override,
    compose_run_length_examples,
    run_length_key,
)
from self.tasks.run_length_logic import compute_run_stats, format_run_length_run_state

__all__ = [
    "AdditionTask",
    "MultiplicationExample",
    "MultiplicationTask",
    "RunLengthExample",
    "RunLengthTask",
    "bucket_run_length_by_bits",
    "build_multiplication_component_payload",
    "build_multiplication_long_dataset",
    "build_multiplication_seed_dataset",
    "build_run_length_composed_dataset",
    "build_run_length_length_bucket_dataset",
    "clone_multiplication_with_override",
    "clone_run_length_with_override",
    "compose_run_length_examples",
    "compute_run_stats",
    "corrupt_numeric_target",
    "format_run_length_run_state",
    "get_boundary_carry_status",
    "prepare_addition_composed_eval",
    "prepare_addition_composed_train",
    "prepare_addition_eval_examples",
    "prepare_addition_initial_splits",
    "multiplication_key",
    "run_length_key",
    "split_addition_examples_by_boundary_status",
]
