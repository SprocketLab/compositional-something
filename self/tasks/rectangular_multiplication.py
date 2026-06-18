#!/usr/bin/env python3
"""Shared helpers for rectangular multiplication experiments.

The primary path in this repo uses final-answer targets. We keep the optional
`cot_reverse_v1` format around for diagnostics, but it is not the default.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from self.tasks.rectangular_composition import (
    RectangularCompositionLeaf,
    _lsd_block_digit_sizes,
    _split_value_into_lsd_blocks,
    build_multiplier_digit_components,
    build_partition_supported_components,
    compose_target_from_multiplier_digit_values,
    compose_target_from_weighted_component_values,
)
from self.tasks.rectangular_data import (
    RECTANGULAR_MULTIPLICATION_FORMATS,
    RectangularMultiplicationExample,
    RectangularMultiplicationKey,
    build_sampled_rectangular_dataset as _build_sampled_rectangular_dataset,
    extract_numeric_answer,
    normalize_rectangular_prediction_for_training,
    parse_rectangular_multiplication_final_value,
    prediction_matches_example,
    rectangular_multiplication_key,
    sample_int_with_exact_digits,
    values_for_digits,
)
from self.tasks.rectangular_digits import (
    ALLOWED_COT_TRACE_CHARS,
    extract_cot_reverse_final_digits,
    format_cot_reverse_prompt,
    format_cot_reverse_target,
    normalize_cot_reverse_prediction_for_training,
    parse_cot_reverse_final_value,
    reverse_digit_text,
    split_digits_lsd_first,
)
from self.tasks.rectangular_partitions import (
    EDGE_ONLY_MULTIPLICATION_PARTITIONS,
    PartitionKey,
    iter_partition_grid,
    parse_partition_spec,
    partition_bucket_id,
    partition_label,
)


def build_sampled_rectangular_dataset(
    *,
    partitions: Sequence[PartitionKey],
    per_partition_counts: Dict[str, int],
    rng: random.Random,
    format_version: str,
    exclude_keys: Optional[set[RectangularMultiplicationKey]] = None,
    record_keys: Optional[Dict[str, set[RectangularMultiplicationKey]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 50_000,
    include_zero_single_digit: bool = False,
) -> Dict[str, List[RectangularMultiplicationExample]]:
    return _build_sampled_rectangular_dataset(
        partitions=partitions,
        per_partition_counts=per_partition_counts,
        rng=rng,
        format_version=format_version,
        exclude_keys=exclude_keys,
        record_keys=record_keys,
        progress_name=progress_name,
        max_attempts=max_attempts,
        include_zero_single_digit=include_zero_single_digit,
        sample_int_fn=sample_int_with_exact_digits,
    )
