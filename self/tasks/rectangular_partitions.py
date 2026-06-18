"""Partition helpers for rectangular multiplication tasks."""

from __future__ import annotations

from typing import Any, List, Tuple


PartitionKey = Tuple[int, int]

EDGE_ONLY_MULTIPLICATION_PARTITIONS: Tuple[PartitionKey, ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
)


def partition_label(partition: PartitionKey) -> str:
    return f"{partition[0]}x{partition[1]}"


def parse_partition_spec(spec: str) -> List[PartitionKey]:
    partitions: List[PartitionKey] = []
    seen: set[PartitionKey] = set()
    for chunk in spec.split(","):
        piece = chunk.strip().lower()
        if not piece:
            continue
        if "x" not in piece:
            raise ValueError(f"Invalid partition spec {chunk!r}; expected AxB.")
        left, right = piece.split("x", 1)
        partition = (int(left), int(right))
        if partition in seen:
            continue
        seen.add(partition)
        partitions.append(partition)
    if not partitions:
        raise ValueError("Expected at least one partition in the spec.")
    return partitions


def iter_partition_grid(
    min_a_digits: int,
    max_a_digits: int,
    min_b_digits: int,
    max_b_digits: int,
) -> List[PartitionKey]:
    partitions: List[PartitionKey] = []
    for a_digits in range(min_a_digits, max_a_digits + 1):
        for b_digits in range(min_b_digits, max_b_digits + 1):
            partitions.append((a_digits, b_digits))
    return partitions


def partition_bucket_id(example: Any) -> int:
    return int(example.a_digits) * 1000 + int(example.b_digits)
