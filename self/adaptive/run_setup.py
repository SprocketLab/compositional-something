"""Run setup and lightweight trace-loading helpers."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def source_sizes_from_examples(task: Any, examples: Sequence[Any]) -> set[int]:
    return {int(task.size_of(example)) for example in examples}


def append_plan_log(plan_path: Path, lines: Iterable[str]) -> None:
    if not plan_path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with plan_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### Implementation Log: {timestamp}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")


def prepare_datasets(
    args: argparse.Namespace,
    task: Any,
    rng: random.Random,
) -> Tuple[Dict[str, List[Any]], Dict[str, set[Any]], List[Any], set[Any]]:
    base_splits, base_records = task.prepare_initial_splits(rng, args)
    training_union = set().union(*base_records.values())
    eval_examples = task.prepare_eval_examples(
        rng,
        args,
        min_size=args.initial_min_size,
        max_size=args.frontier_max_size,
        exclude=training_union,
    )
    eval_keys = task.keys_for_examples(eval_examples)
    return base_splits, base_records, eval_examples, eval_keys


def load_trace_jsonl(path: Path, builder: Any) -> List[Any]:
    if not path.exists():
        return []
    traces: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            traces.append(builder(json.loads(line)))
    return traces
