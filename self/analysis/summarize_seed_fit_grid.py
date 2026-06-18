#!/usr/bin/env python3
"""Summarize a seed-fit Slurm grid and pick threshold-satisfying configs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from self.analysis.seed_fit import discover_seed_fit_results, load_seed_fit_result


def load_result(path: Path) -> Dict[str, Any]:
    return load_seed_fit_result(path)


def _split_metric(payload: Dict[str, Any], split: str, key: str) -> Any:
    split_payload = payload.get("results", {}).get(split, {})
    if isinstance(split_payload, dict):
        return split_payload.get(key)
    return None


def build_rows(base_out: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in discover_seed_fit_results(base_out):
        payload = load_result(path)
        training = payload.get("training", {})
        rows.append(
            {
                "task": payload["task"],
                "train_examples": payload["train_examples"],
                "initial_train_per_size": payload["initial_train_per_size"],
                "max_steps": training.get("max_steps"),
                "effective_batch_size": training.get("effective_batch_size"),
                "approx_effective_epochs_from_steps": training.get("approx_effective_epochs_from_steps"),
                "validation_accuracy": payload["results"]["validation"]["accuracy"],
                "validation_min_per_size_accuracy": payload.get(
                    "validation_min_per_size_accuracy",
                    _split_metric(payload, "validation", "min_per_size_accuracy"),
                ),
                "test_accuracy": payload["results"]["test"]["accuracy"],
                "test_min_per_size_accuracy": payload.get(
                    "test_min_per_size_accuracy",
                    _split_metric(payload, "test", "min_per_size_accuracy"),
                ),
                "meets_threshold": payload.get("meets_threshold", False),
                "results_path": str(path),
            }
        )
    return rows


def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task = str(row["task"])
        validation_min = row.get("validation_min_per_size_accuracy")
        test_min = row.get("test_min_per_size_accuracy")
        if validation_min is None or test_min is None:
            continue
        if validation_min < threshold or test_min < threshold:
            continue
        score = (
            int(row["initial_train_per_size"]),
            int(row["max_steps"]),
            -float(test_min),
            -float(validation_min),
        )
        current = best.get(task)
        if current is None:
            best[task] = dict(row)
            best[task]["_score"] = score
            continue
        if score < current["_score"]:
            best[task] = dict(row)
            best[task]["_score"] = score

    for task in list(best):
        best[task].pop("_score", None)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a seed-fit grid sweep.")
    parser.add_argument("--base-out", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    base_out = Path(args.base_out)
    rows = build_rows(base_out)
    rows.sort(key=lambda row: (row["task"], row["initial_train_per_size"], row["max_steps"]))

    summary_tsv = base_out / "seed_fit_summary.tsv"
    write_tsv(summary_tsv, rows)

    selected = choose_best(rows, args.threshold)
    selected_path = base_out / "seed_fit_selected_configs.json"
    with selected_path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    print(f"[INFO] Found {len(rows)} completed seed-fit runs.", flush=True)
    print(f"[INFO] Wrote summary to {summary_tsv}", flush=True)
    print(f"[INFO] Wrote selected configs to {selected_path}", flush=True)
    for task in sorted(selected):
        row = selected[task]
        print(
            "[SELECT] {task}: train_per_size={train_per_size} max_steps={max_steps} "
            "val_min={val_min:.4f} test_min={test_min:.4f}".format(
                task=task,
                train_per_size=row["initial_train_per_size"],
                max_steps=row["max_steps"],
                val_min=row["validation_min_per_size_accuracy"],
                test_min=row["test_min_per_size_accuracy"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
