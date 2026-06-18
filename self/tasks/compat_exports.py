#!/usr/bin/env python3
"""Compatibility export manifest for :mod:`self.self_improvement_tasks`."""

from __future__ import annotations

from self.core.evaluation import extract_numeric_answer, generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask
from self.tasks.addition import AdditionTask
from self.tasks.addition_data import (
    ADDITION_SAMPLING_MODES,
    ADDITION_SAMPLING_NATURAL,
    ADDITION_WIDTH_EXACT_DIGITS,
    ADDITION_WIDTH_FIXED_MIXED_PROMPT,
    ADDITION_WIDTH_MODES,
    COMPOSITION_PATH_MODES,
    COMPOSITION_PATH_RANDOM,
    AdditionExample,
    build_composed_datasets,
    build_composed_pseudo_map,
    build_length_bucket_dataset,
    clone_with_override,
    corrupt_numeric_target,
    decode_key,
    encode_key,
    example_key,
    get_boundary_carry_status,
    has_component_boundary_carry,
    prepare_addition_composed_eval,
    prepare_addition_composed_train,
    prepare_addition_eval_examples,
    prepare_addition_initial_splits,
    split_addition_examples_by_boundary_status,
)
from self.tasks.bit_common import (
    BIT_COMPOSE_ARITIES,
    BIT_GUARDED_COMPOSE_RULES,
    BIT_TARGET_MODES,
    build_direct_pseudo_examples,
    build_guarded_bit_pseudo_examples,
    count_examples_by_size,
    format_size_count_map,
    guard_slice_partition,
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
    normalize_symbol_alphabet_size,
    normalize_task_format_version,
    run_length_guard_accepts_true_components,
    sample_unique_bitstrings,
)
from self.tasks.bit_composition import (
    BIT_COMPOSITION_PATH_FIXED_BINARY,
    BIT_COMPOSITION_PATH_MODES,
    BIT_COMPOSITION_PATH_RANDOM,
    bit_composed_target_sizes_from_examples,
    choose_component_sizes,
    exact2_reachable_sizes_from_examples,
    fixed_binary_reachable_sizes_from_examples,
)
from self.tasks.bit_parsing import (
    INTEGER_PATTERN,
    MULTIPLICATION_FORMATS,
    RUN_LENGTH_ALPHABET_SYMBOLS,
    RUN_LENGTH_FORMATS,
    RUN_LENGTH_TARGET_RUN_STATE,
    format_multiplication_target,
    parse_multiplication_prediction,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
    parse_run_length_symbol_pair_prediction,
)
from self.tasks.multiplication import MultiplicationTask
from self.tasks.multiplication_data import (
    MultiplicationExample,
    analyze_partial_products,
    build_multiplication_component_payload,
    build_multiplication_long_dataset,
    build_multiplication_seed_dataset,
    clone_multiplication_with_override,
    decode_multiplication_key,
    encode_multiplication_key,
    generate_long_multiplication_example,
    generate_multiplication_seed_example,
    get_multiplication_slice_name,
    iter_multiplication_sizes,
    multiplication_key,
    random_int_with_exact_digits,
    split_value_into_blocks,
)
from self.tasks.run_length import RunLengthTask
from self.tasks.run_length_data import (
    RunLengthExample,
    bucket_run_length_by_bits,
    build_run_length_composed_dataset,
    build_run_length_length_bucket_dataset,
    clone_run_length_with_override,
    compose_run_length_examples,
    compose_run_length_to_length,
    decode_run_length_key,
    encode_run_length_key,
    generate_run_length_example,
    merge_run_length,
    run_length_key,
)
from self.tasks.run_length_logic import (
    compute_run_state,
    compute_run_stats,
    format_run_length_run_state,
    format_run_length_target,
    leftmost_max_run_pair,
    merge_run_state,
)

SplitName = str

CORE_PROTOCOL_EXPORTS = (
    "JsonDict",
    "SelfImprovementTask",
    "SplitName",
)

CORE_EVALUATION_EXPORTS = (
    "extract_numeric_answer",
    "generate_prediction_map",
)

ADDITION_EXPORTS = (
    "ADDITION_SAMPLING_MODES",
    "ADDITION_SAMPLING_NATURAL",
    "ADDITION_WIDTH_EXACT_DIGITS",
    "ADDITION_WIDTH_FIXED_MIXED_PROMPT",
    "ADDITION_WIDTH_MODES",
    "COMPOSITION_PATH_MODES",
    "COMPOSITION_PATH_RANDOM",
    "AdditionExample",
    "AdditionTask",
    "build_composed_datasets",
    "build_composed_pseudo_map",
    "build_length_bucket_dataset",
    "clone_with_override",
    "corrupt_numeric_target",
    "decode_key",
    "encode_key",
    "example_key",
    "get_boundary_carry_status",
    "has_component_boundary_carry",
    "prepare_addition_composed_eval",
    "prepare_addition_composed_train",
    "prepare_addition_eval_examples",
    "prepare_addition_initial_splits",
    "split_addition_examples_by_boundary_status",
)

BIT_COMMON_EXPORTS = (
    "BIT_COMPOSE_ARITIES",
    "BIT_GUARDED_COMPOSE_RULES",
    "BIT_TARGET_MODES",
    "INTEGER_PATTERN",
    "MULTIPLICATION_FORMATS",
    "RUN_LENGTH_ALPHABET_SYMBOLS",
    "RUN_LENGTH_FORMATS",
    "RUN_LENGTH_TARGET_RUN_STATE",
    "build_direct_pseudo_examples",
    "build_guarded_bit_pseudo_examples",
    "count_examples_by_size",
    "format_multiplication_target",
    "format_size_count_map",
    "guard_slice_partition",
    "normalize_bit_composition_path_mode",
    "normalize_bit_target_mode",
    "normalize_compose_arity",
    "normalize_guarded_compose_rule",
    "normalize_symbol_alphabet_size",
    "normalize_task_format_version",
    "parse_multiplication_prediction",
    "parse_run_length_prediction",
    "parse_run_length_run_state_prediction",
    "parse_run_length_symbol_pair_prediction",
    "run_length_guard_accepts_true_components",
    "sample_unique_bitstrings",
)

BIT_COMPOSITION_EXPORTS = (
    "BIT_COMPOSITION_PATH_FIXED_BINARY",
    "BIT_COMPOSITION_PATH_MODES",
    "BIT_COMPOSITION_PATH_RANDOM",
    "bit_composed_target_sizes_from_examples",
    "choose_component_sizes",
    "exact2_reachable_sizes_from_examples",
    "fixed_binary_reachable_sizes_from_examples",
)

MULTIPLICATION_EXPORTS = (
    "MultiplicationExample",
    "MultiplicationTask",
    "analyze_partial_products",
    "build_multiplication_component_payload",
    "build_multiplication_long_dataset",
    "build_multiplication_seed_dataset",
    "clone_multiplication_with_override",
    "decode_multiplication_key",
    "encode_multiplication_key",
    "generate_long_multiplication_example",
    "generate_multiplication_seed_example",
    "get_multiplication_slice_name",
    "iter_multiplication_sizes",
    "multiplication_key",
    "random_int_with_exact_digits",
    "split_value_into_blocks",
)

RUN_LENGTH_EXPORTS = (
    "RunLengthExample",
    "RunLengthTask",
    "bucket_run_length_by_bits",
    "build_run_length_composed_dataset",
    "build_run_length_length_bucket_dataset",
    "clone_run_length_with_override",
    "compose_run_length_examples",
    "compose_run_length_to_length",
    "compute_run_state",
    "compute_run_stats",
    "decode_run_length_key",
    "encode_run_length_key",
    "format_run_length_run_state",
    "format_run_length_target",
    "generate_run_length_example",
    "leftmost_max_run_pair",
    "merge_run_length",
    "merge_run_state",
    "run_length_key",
)

TASK_COMPAT_EXPORT_GROUPS = (
    CORE_PROTOCOL_EXPORTS,
    CORE_EVALUATION_EXPORTS,
    ADDITION_EXPORTS,
    BIT_COMMON_EXPORTS,
    BIT_COMPOSITION_EXPORTS,
    MULTIPLICATION_EXPORTS,
    RUN_LENGTH_EXPORTS,
)

TASK_COMPAT_EXPORT_NAMES = tuple(name for group in TASK_COMPAT_EXPORT_GROUPS for name in group)

__all__ = list(TASK_COMPAT_EXPORT_NAMES)
