"""Component construction and target composition for rectangular multiplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from self.tasks.rectangular_data import RectangularMultiplicationExample
from self.tasks.rectangular_digits import reverse_digit_text, split_digits_lsd_first
from self.tasks.rectangular_partitions import PartitionKey, partition_label


@dataclass(frozen=True, slots=True)
class RectangularCompositionLeaf:
    shift_digits: int
    example: RectangularMultiplicationExample


def build_multiplier_digit_components(
    example: RectangularMultiplicationExample,
    *,
    component_format_version: Optional[str] = None,
) -> List[Tuple[int, RectangularMultiplicationExample]]:
    format_version = component_format_version or example.format_version
    components: List[Tuple[int, RectangularMultiplicationExample]] = []
    for digit_index, digit_value in enumerate(split_digits_lsd_first(example.b)):
        components.append(
            (
                digit_index,
                RectangularMultiplicationExample(
                    a=example.a,
                    b=digit_value,
                    a_digits=example.a_digits,
                    b_digits=1,
                    format_version=format_version,
                ),
            )
        )
    return components


def _lsd_block_digit_sizes(total_digits: int, max_chunk_digits: int) -> List[int]:
    if total_digits <= 0:
        raise ValueError("total_digits must be positive.")
    if max_chunk_digits <= 0:
        raise ValueError("max_chunk_digits must be positive.")
    sizes: List[int] = []
    digits_consumed = 0
    while digits_consumed < total_digits:
        block_digits = min(max_chunk_digits, total_digits - digits_consumed)
        sizes.append(block_digits)
        digits_consumed += block_digits
    return sizes


def _split_value_into_lsd_blocks(
    value: int,
    *,
    total_digits: int,
    max_chunk_digits: int,
) -> List[Tuple[int, int, int]]:
    if total_digits <= 0:
        raise ValueError("total_digits must be positive.")
    blocks: List[Tuple[int, int, int]] = []
    remaining = value
    offset_digits = 0
    for block_digits in _lsd_block_digit_sizes(total_digits, max_chunk_digits):
        modulus = 10**block_digits
        block_value = remaining % modulus
        blocks.append((offset_digits, block_value, block_digits))
        remaining //= modulus
        offset_digits += block_digits
    return blocks


def build_partition_supported_components(
    example: RectangularMultiplicationExample,
    *,
    supported_partitions: Sequence[PartitionKey],
    component_format_version: Optional[str] = None,
) -> List[RectangularCompositionLeaf]:
    format_version = component_format_version or example.format_version
    supported = tuple(dict.fromkeys(tuple(partition) for partition in supported_partitions))
    supported_set = set(supported)
    b_only_memo: Dict[PartitionKey, Optional[Tuple[int, int, int, str, int]]] = {}
    memo: Dict[PartitionKey, Optional[Tuple[int, int, int, str, int]]] = {}

    def best_b_only_strategy(
        a_digits: int,
        b_digits: int,
    ) -> Optional[Tuple[int, int, int, str, int]]:
        partition = (a_digits, b_digits)
        if partition in b_only_memo:
            return b_only_memo[partition]
        if partition in supported_set:
            result = (1, 0, 0, "leaf", 0)
            b_only_memo[partition] = result
            return result

        best: Optional[Tuple[int, int, int, str, int]] = None
        for split_digits in range(1, b_digits):
            child_sizes = _lsd_block_digit_sizes(b_digits, split_digits)
            child_strategies = [
                best_b_only_strategy(a_digits, child_b_digits)
                for child_b_digits in child_sizes
            ]
            if any(strategy is None for strategy in child_strategies):
                continue
            leaf_count = sum(
                strategy[0] for strategy in child_strategies if strategy is not None
            )
            depth = 1 + max(
                strategy[1] for strategy in child_strategies if strategy is not None
            )
            candidate = (leaf_count, depth, 0, "split_b", split_digits)
            if best is None or candidate < best:
                best = candidate
        b_only_memo[partition] = best
        return best

    def best_strategy(
        a_digits: int,
        b_digits: int,
    ) -> Optional[Tuple[int, int, int, str, int]]:
        partition = (a_digits, b_digits)
        if partition in memo:
            return memo[partition]
        if partition in supported_set:
            result = (1, 0, 0, "leaf", 0)
            memo[partition] = result
            return result

        b_only = best_b_only_strategy(a_digits, b_digits)
        if b_only is not None:
            memo[partition] = b_only
            return b_only

        best: Optional[Tuple[int, int, int, str, int]] = None

        for split_digits in range(1, b_digits):
            child_sizes = _lsd_block_digit_sizes(b_digits, split_digits)
            child_strategies = [
                best_strategy(a_digits, child_b_digits)
                for child_b_digits in child_sizes
            ]
            if any(strategy is None for strategy in child_strategies):
                continue
            leaf_count = sum(
                strategy[0] for strategy in child_strategies if strategy is not None
            )
            depth = 1 + max(
                strategy[1] for strategy in child_strategies if strategy is not None
            )
            candidate = (leaf_count, depth, 0, "split_b", split_digits)
            if best is None or candidate < best:
                best = candidate

        for split_digits in range(1, a_digits):
            child_sizes = _lsd_block_digit_sizes(a_digits, split_digits)
            child_strategies = [
                best_strategy(child_a_digits, b_digits)
                for child_a_digits in child_sizes
            ]
            if any(strategy is None for strategy in child_strategies):
                continue
            leaf_count = sum(
                strategy[0] for strategy in child_strategies if strategy is not None
            )
            depth = 1 + max(
                strategy[1] for strategy in child_strategies if strategy is not None
            )
            candidate = (leaf_count, depth, 1, "split_a", split_digits)
            if best is None or candidate < best:
                best = candidate

        memo[partition] = best
        return best

    def build_leaves(
        current_example: RectangularMultiplicationExample,
    ) -> List[RectangularCompositionLeaf]:
        strategy = best_strategy(current_example.a_digits, current_example.b_digits)
        if strategy is None:
            supported_labels = ", ".join(partition_label(partition) for partition in supported)
            current_partition = partition_label(
                (current_example.a_digits, current_example.b_digits)
            )
            raise ValueError(
                "Could not compose partition "
                f"{current_partition} "
                f"from supported partitions: {supported_labels}"
            )

        _, _, _, mode, split_digits = strategy
        if mode == "leaf":
            return [RectangularCompositionLeaf(shift_digits=0, example=current_example)]

        leaves: List[RectangularCompositionLeaf] = []
        if mode == "split_b":
            for offset_digits, block_value, block_digits in _split_value_into_lsd_blocks(
                current_example.b,
                total_digits=current_example.b_digits,
                max_chunk_digits=split_digits,
            ):
                child_example = RectangularMultiplicationExample(
                    a=current_example.a,
                    b=block_value,
                    a_digits=current_example.a_digits,
                    b_digits=block_digits,
                    format_version=format_version,
                )
                for child_leaf in build_leaves(child_example):
                    leaves.append(
                        RectangularCompositionLeaf(
                            shift_digits=offset_digits + child_leaf.shift_digits,
                            example=child_leaf.example,
                        )
                    )
            return leaves

        if mode == "split_a":
            for offset_digits, block_value, block_digits in _split_value_into_lsd_blocks(
                current_example.a,
                total_digits=current_example.a_digits,
                max_chunk_digits=split_digits,
            ):
                child_example = RectangularMultiplicationExample(
                    a=block_value,
                    b=current_example.b,
                    a_digits=block_digits,
                    b_digits=current_example.b_digits,
                    format_version=format_version,
                )
                for child_leaf in build_leaves(child_example):
                    leaves.append(
                        RectangularCompositionLeaf(
                            shift_digits=offset_digits + child_leaf.shift_digits,
                            example=child_leaf.example,
                        )
                    )
            return leaves

        raise ValueError(f"Unsupported composition strategy {mode!r}.")

    return build_leaves(example)


def compose_target_from_multiplier_digit_values(
    example: RectangularMultiplicationExample,
    component_values: Sequence[int],
) -> str:
    total_value = 0
    for digit_index, partial_value in enumerate(component_values):
        total_value += int(partial_value) * (10**digit_index)

    if example.format_version == "legacy":
        return str(total_value)
    if example.format_version == "symbolic_v1":
        return f"{total_value:0{example.total_digits}d}"
    if example.format_version != "cot_reverse_v1":
        raise ValueError(
            f"Unsupported rectangular multiplication format {example.format_version!r}."
        )

    chunks: List[str] = []
    cumulative = 0
    for digit_index, partial_value in enumerate(component_values):
        shifted_partial = int(partial_value) * (10**digit_index)
        cumulative += shifted_partial
        partial_text = reverse_digit_text(shifted_partial).ljust(
            digit_index + example.a_digits + 1,
            "0",
        )
        if digit_index == 0:
            chunks.append(partial_text)
            continue
        chunks.append("+" + partial_text)
        if digit_index < len(component_values) - 1:
            cumulative_text = reverse_digit_text(cumulative).ljust(
                digit_index + example.a_digits + 1,
                "0",
            )
            chunks.append("(" + cumulative_text + ")")

    final_answer = reverse_digit_text(total_value).ljust(example.total_digits, "0")
    chunks.append("=" + final_answer)
    return "".join(chunks)


def compose_target_from_weighted_component_values(
    example: RectangularMultiplicationExample,
    weighted_component_values: Sequence[Tuple[int, int]],
) -> str:
    total_value = 0
    for shift_digits, partial_value in weighted_component_values:
        total_value += int(partial_value) * (10**shift_digits)

    if example.format_version == "legacy":
        return str(total_value)
    if example.format_version == "symbolic_v1":
        return f"{total_value:0{example.total_digits}d}"
    raise ValueError(
        "Weighted rectangular multiplication composition is only supported for "
        "format_version='legacy' and 'symbolic_v1'."
    )
