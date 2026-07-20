#!/usr/bin/env python3
"""Prepare, train, submit, and collect Qwen3.5 coding atomic sweeps."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from self.coding.atomic_data import (
    AtomicExample,
    build_bfcl_atomic_splits,
    build_bfcl_controlled_frontier,
    build_bfcl_natural_examples,
    build_commitpack_atomic_splits,
    iter_jsonl_paths,
    read_examples,
    stable_hash,
    write_examples,
    write_json,
)
from self.coding.evaluation import evaluate_predictions
from self.coding.training import (
    ChatTargetDataset,
    adapter_parameter_summary,
    generate_predictions,
    load_adapter_for_evaluation,
    load_qwen_lora_model,
    load_qwen_tokenizer,
    train_lora,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
COMMITPACK_REVISION = "fc56fe33c030c6daa414c2b112c932b8eed085e6"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_MODEL = Path(
    "/scratch/gpfs/BRENDEN/changho/hf_cache/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
DEFAULT_BFCL_ROOT = (
    ROOT_DIR
    / "artifacts/data/coding_task_data_exploration"
    / BFCL_REVISION
)
DEFAULT_PYTHON = Path("/home/cs1095/.conda/envs/torch-env/bin/python")
TASKS = ("bfcl", "commitpack")
LEARNING_RATES = (1e-5, 5e-5, 2e-4)
TASK_SETTINGS: Dict[str, Dict[str, Any]] = {
    "bfcl": {
        "data_sizes": (30, 60, 120, 240),
        "full_data_size": 240,
        "steps": (10, 30, 100),
        "max_length": 512,
        "micro_batch_size": 16,
        "eval_batch_size": 16,
    },
    "commitpack": {
        "data_sizes": (250, 500, 1000, 2000),
        "full_data_size": 2000,
        "steps": (50, 150, 450),
        "max_length": 1024,
        "micro_batch_size": 4,
        "eval_batch_size": 4,
    },
}
INITIAL_SEED = 7
REPLICATION_SEEDS = (23, 42)
EFFECTIVE_BATCH_SIZE = 16


@dataclass(frozen=True)
class SweepCell:
    task: str
    data_size: int
    max_steps: int
    learning_rate: float
    seed: int
    stage: int

    @property
    def schedule_key(self) -> Tuple[int, float]:
        return self.max_steps, self.learning_rate

    @property
    def config_key(self) -> Tuple[str, int, int, float]:
        return self.task, self.data_size, self.max_steps, self.learning_rate

    @property
    def slug(self) -> str:
        learning_rate = f"{self.learning_rate:.0e}".replace("-", "m").replace("+", "p")
        return f"n{self.data_size}-s{self.max_steps}-lr{learning_rate}-seed{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "data_size": self.data_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "stage": self.stage,
            "slug": self.slug,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepCell":
        return cls(
            task=str(payload["task"]),
            data_size=int(payload["data_size"]),
            max_steps=int(payload["max_steps"]),
            learning_rate=float(payload["learning_rate"]),
            seed=int(payload["seed"]),
            stage=int(payload["stage"]),
        )


def stage1_cells(task: str) -> List[SweepCell]:
    settings = TASK_SETTINGS[task]
    return [
        SweepCell(
            task=task,
            data_size=int(settings["full_data_size"]),
            max_steps=int(steps),
            learning_rate=float(learning_rate),
            seed=INITIAL_SEED,
            stage=1,
        )
        for learning_rate in LEARNING_RATES
        for steps in settings["steps"]
    ]


def _metric_sort_key(cell: SweepCell, metrics: Mapping[str, Any]) -> Tuple[Any, ...]:
    validation = metrics["validation"]
    return (
        -float(validation["exact_accuracy"]),
        -float(validation["format_accuracy"]),
        cell.max_steps,
        cell.data_size,
        cell.learning_rate,
    )


def select_stage1_schedules(results: Sequence[Tuple[SweepCell, Mapping[str, Any]]]) -> List[Tuple[int, float]]:
    if len(results) != 9:
        raise ValueError(f"Stage 1 requires nine completed cells, found {len(results)}")
    ranked = sorted(results, key=lambda item: _metric_sort_key(item[0], item[1]))
    return [item[0].schedule_key for item in ranked[:2]]


def stage2_cells(task: str, schedules: Sequence[Tuple[int, float]]) -> List[SweepCell]:
    smaller_sizes = TASK_SETTINGS[task]["data_sizes"][:-1]
    cells = [
        SweepCell(task, int(size), int(steps), float(lr), INITIAL_SEED, 2)
        for steps, lr in schedules
        for size in smaller_sizes
    ]
    if len(cells) != 6:
        raise AssertionError("Stage 2 must contain six cells per task")
    return cells


def select_stage3_configs(results: Sequence[Tuple[SweepCell, Mapping[str, Any]]]) -> List[SweepCell]:
    if len(results) != 15:
        raise ValueError(f"Stage 3 selection requires 15 completed seed-7 cells, found {len(results)}")
    ranked = sorted(results, key=lambda item: _metric_sort_key(item[0], item[1]))
    return [item[0] for item in ranked[:3]]


def stage3_cells(selected: Sequence[SweepCell]) -> List[SweepCell]:
    cells = [
        SweepCell(
            task=cell.task,
            data_size=cell.data_size,
            max_steps=cell.max_steps,
            learning_rate=cell.learning_rate,
            seed=seed,
            stage=3,
        )
        for cell in selected
        for seed in REPLICATION_SEEDS
    ]
    if len(cells) != 6:
        raise AssertionError("Stage 3 must contain six cells per task")
    return cells


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_length_summary(examples: Sequence[AtomicExample], tokenizer: Any, max_length: int) -> Dict[str, Any]:
    dataset = ChatTargetDataset(examples, tokenizer, max_length=max_length)
    lengths = sorted(len(dataset[index]["input_ids"]) for index in range(len(dataset)))
    if not lengths:
        return {"count": 0}
    percentile = lambda fraction: lengths[round((len(lengths) - 1) * fraction)]
    return {
        "count": len(lengths),
        "min": lengths[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "mean": sum(lengths) / len(lengths),
    }


def _download_commitpack_file(filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="bigcode/commitpackft",
        repo_type="dataset",
        filename=filename,
        revision=COMMITPACK_REVISION,
        cache_dir=str(cache_dir),
    )
    return Path(downloaded).resolve()


def prepare_data(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    tokenizer = load_qwen_tokenizer(args.model_name)

    bfcl_question = args.bfcl_root / "BFCL_v3_simple.jsonl"
    bfcl_answer = args.bfcl_root / "possible_answer/BFCL_v3_simple.jsonl"
    bfcl_splits, bfcl_audit = build_bfcl_atomic_splits(bfcl_question, bfcl_answer)
    for split, examples in bfcl_splits.items():
        write_examples(data_dir / "bfcl" / f"{split}.jsonl", examples)
    controlled = build_bfcl_controlled_frontier(bfcl_splits["test"])
    bfcl_frontier_dir = data_dir / "bfcl" / "frontier"
    for component_count, examples in controlled.items():
        write_examples(bfcl_frontier_dir / f"controlled_{component_count}.jsonl", examples)
    for stem, question_name, answer_name in (
        ("natural_parallel", "BFCL_v3_parallel.jsonl", "possible_answer/BFCL_v3_parallel.jsonl"),
        (
            "natural_parallel_multiple",
            "BFCL_v3_parallel_multiple.jsonl",
            "possible_answer/BFCL_v3_parallel_multiple.jsonl",
        ),
    ):
        natural = build_bfcl_natural_examples(
            args.bfcl_root / question_name,
            args.bfcl_root / answer_name,
            track_name="natural",
        )
        write_examples(bfcl_frontier_dir / f"{stem}.jsonl", natural)
    bfcl_audit["frontier_counts"] = {
        **{f"controlled_{count}": len(examples) for count, examples in controlled.items()},
        "natural_parallel": 200,
        "natural_parallel_multiple": 200,
    }
    bfcl_audit["token_lengths"] = {
        split: _token_length_summary(examples, tokenizer, 512)
        for split, examples in bfcl_splits.items()
    }
    write_json(data_dir / "bfcl" / "audit.json", bfcl_audit)

    if args.commitpack_json_path is None:
        commitpack_json = _download_commitpack_file(
            "data/json/data.jsonl", output_dir / "source_cache"
        )
    else:
        commitpack_json = args.commitpack_json_path.resolve()
    if args.commitpack_yaml_path is None:
        commitpack_yaml = _download_commitpack_file(
            "data/yaml/data.jsonl", output_dir / "source_cache"
        )
    else:
        commitpack_yaml = args.commitpack_yaml_path.resolve()
    commitpack_splits, commitpack_audit = build_commitpack_atomic_splits(
        iter_jsonl_paths([commitpack_json, commitpack_yaml]),
        tokenizer=tokenizer,
    )
    commitpack_audit["token_lengths"] = {
        split: _token_length_summary(examples, tokenizer, 1024)
        for split, examples in commitpack_splits.items()
        if split in {"train", "validation", "test"}
    }
    for split, examples in commitpack_splits.items():
        if split.startswith("frontier_"):
            write_examples(data_dir / "commitpack" / "frontier" / f"natural_{split.rsplit('_', 1)[-1]}.jsonl", examples)
        else:
            write_examples(data_dir / "commitpack" / f"{split}.jsonl", examples)
    commitpack_audit["sources"] = {
        "json": {"path": str(commitpack_json), "sha256": sha256_file(commitpack_json)},
        "yaml": {"path": str(commitpack_yaml), "sha256": sha256_file(commitpack_yaml)},
    }
    write_json(data_dir / "commitpack" / "audit.json", commitpack_audit)
    manifest = {
        "bfcl_revision": BFCL_REVISION,
        "commitpack_revision": COMMITPACK_REVISION,
        "qwen_revision": QWEN_REVISION,
        "model_name": args.model_name,
        "bfcl_sources": {
            "question": {"path": str(bfcl_question), "sha256": sha256_file(bfcl_question)},
            "answer": {"path": str(bfcl_answer), "sha256": sha256_file(bfcl_answer)},
        },
        "created_at_unix": time.time(),
    }
    write_json(data_dir / "manifest.json", manifest)
    print(f"[INFO] Prepared coding atomic data under {data_dir}", flush=True)


def cell_dir(run_root: Path, cell: SweepCell) -> Path:
    return run_root / "cells" / cell.task / cell.slug


def load_cell_metrics(run_root: Path, cell: SweepCell) -> Dict[str, Any]:
    path = cell_dir(run_root, cell) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cell metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_prediction_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[AtomicExample],
    batch_size: int,
    output_path: Path,
) -> Dict[str, Any]:
    largest_component_count = max((example.component_count for example in examples), default=1)
    max_new_tokens = max(128, 64 * largest_component_count + 32)
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    summary, rows = evaluate_predictions(examples, predictions)
    _write_prediction_rows(output_path, rows)
    return summary


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_cell(args: argparse.Namespace) -> None:
    cell = SweepCell(
        task=args.task,
        data_size=args.data_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        stage=args.stage,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = TASK_SETTINGS[cell.task]
    train_examples = read_examples(args.data_dir / cell.task / "train.jsonl")[: cell.data_size]
    validation_examples = read_examples(args.data_dir / cell.task / "validation.jsonl")
    if len(train_examples) != cell.data_size:
        raise ValueError(f"Requested {cell.data_size} training examples, found {len(train_examples)}")
    config = {
        **cell.to_dict(),
        "model_name": args.model_name,
        "qwen_revision": QWEN_REVISION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "max_length": settings["max_length"],
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"},
    }
    write_json(output_dir / "config.json", config)
    micro_batch = int(args.micro_batch_size or settings["micro_batch_size"])
    attempts: List[Dict[str, Any]] = []
    while micro_batch >= 1:
        model = None
        try:
            torch.manual_seed(cell.seed)
            torch.cuda.manual_seed_all(cell.seed)
            tokenizer = load_qwen_tokenizer(args.model_name)
            model = load_qwen_lora_model(args.model_name)
            parameter_summary = adapter_parameter_summary(model)
            train_metrics = train_lora(
                model=model,
                tokenizer=tokenizer,
                examples=train_examples,
                output_dir=output_dir,
                max_length=int(settings["max_length"]),
                max_steps=cell.max_steps,
                learning_rate=cell.learning_rate,
                micro_batch_size=micro_batch,
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
                seed=cell.seed,
            )
            validation = _evaluate_split(
                model=model,
                tokenizer=tokenizer,
                examples=validation_examples,
                batch_size=int(settings["eval_batch_size"]),
                output_path=output_dir / "predictions" / "validation.jsonl",
            )
            train_evaluation = None
            if args.evaluate_train:
                train_evaluation = _evaluate_split(
                    model=model,
                    tokenizer=tokenizer,
                    examples=train_examples,
                    batch_size=int(settings["eval_batch_size"]),
                    output_path=output_dir / "predictions" / "train.jsonl",
                )
            metrics = {
                "cell": cell.to_dict(),
                "training": train_metrics,
                "validation": validation,
                "train_evaluation": train_evaluation,
                "parameters": parameter_summary,
                "oom_attempts": attempts,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            write_json(output_dir / "metrics.json", metrics)
            print(json.dumps(metrics["validation"], sort_keys=True), flush=True)
            return
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or micro_batch == 1:
                raise
            attempts.append({"micro_batch_size": micro_batch, "error": str(exc)})
            micro_batch //= 2
            print(f"[WARN] CUDA OOM; retrying with micro_batch_size={micro_batch}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _load_base_model(model_name: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM

    tokenizer = load_qwen_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
        dtype=torch.bfloat16,
    )
    model.to(torch.device("cuda"))
    model.eval()
    return model, tokenizer


def evaluate_base(args: argparse.Namespace) -> None:
    settings = TASK_SETTINGS[args.task]
    model, tokenizer = _load_base_model(args.model_name)
    payload: Dict[str, Any] = {"task": args.task, "model_name": args.model_name}
    for split in ("validation", "test"):
        examples = read_examples(args.data_dir / args.task / f"{split}.jsonl")
        payload[split] = _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            batch_size=int(settings["eval_batch_size"]),
            output_path=args.output_dir / "predictions" / f"{split}.jsonl",
        )
    frontier_dir = args.data_dir / args.task / "frontier"
    if frontier_dir.exists():
        for path in sorted(frontier_dir.glob("*.jsonl")):
            examples = read_examples(path)
            payload[path.stem] = _evaluate_split(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                batch_size=int(settings["eval_batch_size"]),
                output_path=args.output_dir / "predictions" / f"{path.stem}.jsonl",
            )
    write_json(args.output_dir / "metrics.json", payload)


def evaluate_final(args: argparse.Namespace) -> None:
    cell = SweepCell.from_dict(json.loads(args.cell_json))
    settings = TASK_SETTINGS[cell.task]
    adapter_dir = cell_dir(args.run_root, cell) / "adapter"
    model, tokenizer = load_adapter_for_evaluation(args.model_name, adapter_dir)
    payload: Dict[str, Any] = {"cell": cell.to_dict(), "adapter_dir": str(adapter_dir)}
    train_examples = read_examples(args.data_dir / cell.task / "train.jsonl")[: cell.data_size]
    payload["train"] = _evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=train_examples,
        batch_size=int(settings["eval_batch_size"]),
        output_path=args.output_dir / "predictions" / "train.jsonl",
    )
    split_paths = [("test", args.data_dir / cell.task / "test.jsonl")]
    frontier_dir = args.data_dir / cell.task / "frontier"
    if frontier_dir.exists():
        split_paths.extend((path.stem, path) for path in sorted(frontier_dir.glob("*.jsonl")))
    for split, path in split_paths:
        examples = read_examples(path)
        payload[split] = _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            batch_size=int(settings["eval_batch_size"]),
            output_path=args.output_dir / "predictions" / f"{split}.jsonl",
        )
    write_json(args.output_dir / "metrics.json", payload)


def verify_adapter(args: argparse.Namespace) -> None:
    """Reload a saved adapter in a fresh process and evaluate a small slice."""
    settings = TASK_SETTINGS[args.task]
    examples = read_examples(args.data_dir / args.task / "validation.jsonl")[: args.limit]
    if not examples:
        raise ValueError(f"No validation examples found for {args.task}")
    model, tokenizer = load_adapter_for_evaluation(args.model_name, args.adapter_dir)
    payload = {
        "task": args.task,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "limit": len(examples),
        "validation": _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            batch_size=min(int(settings["eval_batch_size"]), len(examples)),
            output_path=args.output_dir / "predictions" / "validation.jsonl",
        ),
    }
    write_json(args.output_dir / "metrics.json", payload)
    print(json.dumps(payload["validation"], sort_keys=True), flush=True)


def _read_manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / "sweep_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(run_root: Path, payload: Mapping[str, Any]) -> None:
    write_json(run_root / "sweep_manifest.json", payload)


def _run_command(command: Sequence[str], *, dry_run: bool) -> Optional[str]:
    print(f"[INFO] Command: {shlex.join(list(command))}", flush=True)
    if dry_run:
        return None
    completed = subprocess.run(
        list(command),
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr, flush=True)
    return completed.stdout.strip()


def _submit_job(
    *,
    command: Sequence[str],
    job_name: str,
    log_dir: Path,
    dependencies: Sequence[str],
    gpu: bool,
    dry_run: bool,
) -> Optional[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition=ailab" if gpu else "--partition=cpu",
        "--cpus-per-task=8" if gpu else "--cpus-per-task=1",
        "--mem=48G" if gpu else "--mem=8G",
        "--time=06:00:00" if gpu else "--time=00:30:00",
        f"--job-name={job_name}",
        f"--output={log_dir / (job_name + '-%j.out')}",
        f"--error={log_dir / (job_name + '-%j.err')}",
    ]
    if gpu:
        sbatch.append("--gres=gpu:h200:1")
    if dependencies:
        sbatch.append(f"--dependency=afterany:{':'.join(dependencies)}")
    wrapped = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        *command,
    ]
    sbatch.append(f"--wrap={shlex.join(wrapped)}")
    output = _run_command(sbatch, dry_run=dry_run)
    return None if output is None else output.split(";")[0].strip()


def _cell_command(
    *,
    python_bin: Path,
    run_root: Path,
    data_dir: Path,
    model_name: str,
    cell: SweepCell,
    evaluate_train: bool = False,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "self.experiments.coding_atomic_sweep",
        "run-cell",
        "--task",
        cell.task,
        "--data-dir",
        str(data_dir),
        "--model-name",
        model_name,
        "--output-dir",
        str(cell_dir(run_root, cell)),
        "--data-size",
        str(cell.data_size),
        "--max-steps",
        str(cell.max_steps),
        "--learning-rate",
        str(cell.learning_rate),
        "--seed",
        str(cell.seed),
        "--stage",
        str(cell.stage),
    ]
    if evaluate_train:
        command.append("--evaluate-train")
    return command


def _submit_cells_in_waves(
    *,
    cells: Sequence[SweepCell],
    python_bin: Path,
    run_root: Path,
    data_dir: Path,
    model_name: str,
    log_dir: Path,
    initial_dependencies: Sequence[str],
    dry_run: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    submitted: List[Dict[str, Any]] = []
    previous_wave = list(initial_dependencies)
    for wave_start in range(0, len(cells), 4):
        wave_ids: List[str] = []
        for cell in cells[wave_start : wave_start + 4]:
            job_id = _submit_job(
                command=_cell_command(
                    python_bin=python_bin,
                    run_root=run_root,
                    data_dir=data_dir,
                    model_name=model_name,
                    cell=cell,
                    evaluate_train=cell.stage == 3,
                ),
                job_name=f"code-{cell.task}-st{cell.stage}-{cell.slug}"[:120],
                log_dir=log_dir,
                dependencies=previous_wave,
                gpu=True,
                dry_run=dry_run,
            )
            submitted.append({**cell.to_dict(), "job_id": job_id})
            if job_id:
                wave_ids.append(job_id)
        previous_wave = wave_ids
    return submitted, previous_wave


def _stage_command(
    command: str,
    *,
    python_bin: Path,
    run_root: Path,
    data_dir: Path,
    model_name: str,
    log_dir: Path,
) -> List[str]:
    return [
        str(python_bin),
        "-m",
        "self.experiments.coding_atomic_sweep",
        command,
        "--run-root",
        str(run_root),
        "--data-dir",
        str(data_dir),
        "--model-name",
        model_name,
        "--python-bin",
        str(python_bin),
        "--log-dir",
        str(log_dir),
    ]


def submit_sweep(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    data_dir = args.data_dir.resolve()
    log_dir = args.log_dir.resolve()
    stage1 = [cell for task in TASKS for cell in stage1_cells(task)]
    manifest: Dict[str, Any] = {
        "model_name": args.model_name,
        "qwen_revision": QWEN_REVISION,
        "data_dir": str(data_dir),
        "stage1": [cell.to_dict() for cell in stage1],
        "stage2": [],
        "stage3": [],
        "base_evaluations": [],
        "created_at_unix": time.time(),
    }
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
        _write_manifest(run_root, manifest)
    base_job_ids: List[str] = []
    for task in TASKS:
        command = [
            str(args.python_bin),
            "-m",
            "self.experiments.coding_atomic_sweep",
            "evaluate-base",
            "--task",
            task,
            "--data-dir",
            str(data_dir),
            "--model-name",
            args.model_name,
            "--output-dir",
            str(run_root / "base" / task),
        ]
        job_id = _submit_job(
            command=command,
            job_name=f"code-{task}-base",
            log_dir=log_dir,
            dependencies=(),
            gpu=True,
            dry_run=args.dry_run,
        )
        manifest["base_evaluations"].append({"task": task, "job_id": job_id})
        if job_id:
            base_job_ids.append(job_id)
    submitted, last_wave = _submit_cells_in_waves(
        cells=stage1,
        python_bin=args.python_bin,
        run_root=run_root,
        data_dir=data_dir,
        model_name=args.model_name,
        log_dir=log_dir,
        initial_dependencies=base_job_ids,
        dry_run=args.dry_run,
    )
    manifest["stage1_jobs"] = submitted
    selector_id = _submit_job(
        command=_stage_command(
            "stage2",
            python_bin=args.python_bin,
            run_root=run_root,
            data_dir=data_dir,
            model_name=args.model_name,
            log_dir=log_dir,
        ),
        job_name="code-atomic-stage2-select",
        log_dir=log_dir,
        dependencies=last_wave,
        gpu=False,
        dry_run=args.dry_run,
    )
    manifest["stage2_selector_job_id"] = selector_id
    if not args.dry_run:
        _write_manifest(run_root, manifest)
    else:
        print("[INFO] Dry-run matrix: 18 stage-1, 12 dynamic stage-2, 12 dynamic stage-3 training cells.")


def _completed_results(run_root: Path, cells: Sequence[SweepCell]) -> List[Tuple[SweepCell, Dict[str, Any]]]:
    return [(cell, load_cell_metrics(run_root, cell)) for cell in cells]


def command_stage2(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.run_root)
    all_cells: List[SweepCell] = []
    selections: Dict[str, Any] = {}
    for task in TASKS:
        stage1 = [SweepCell.from_dict(row) for row in manifest["stage1"] if row["task"] == task]
        schedules = select_stage1_schedules(_completed_results(args.run_root, stage1))
        cells = stage2_cells(task, schedules)
        all_cells.extend(cells)
        selections[task] = [{"max_steps": steps, "learning_rate": lr} for steps, lr in schedules]
    manifest["stage1_selection"] = selections
    manifest["stage2"] = [cell.to_dict() for cell in all_cells]
    submitted, last_wave = _submit_cells_in_waves(
        cells=all_cells,
        python_bin=args.python_bin,
        run_root=args.run_root,
        data_dir=args.data_dir,
        model_name=args.model_name,
        log_dir=args.log_dir,
        initial_dependencies=(),
        dry_run=False,
    )
    manifest["stage2_jobs"] = submitted
    selector_id = _submit_job(
        command=_stage_command(
            "stage3",
            python_bin=args.python_bin,
            run_root=args.run_root,
            data_dir=args.data_dir,
            model_name=args.model_name,
            log_dir=args.log_dir,
        ),
        job_name="code-atomic-stage3-select",
        log_dir=args.log_dir,
        dependencies=last_wave,
        gpu=False,
        dry_run=False,
    )
    manifest["stage3_selector_job_id"] = selector_id
    _write_manifest(args.run_root, manifest)


def command_stage3(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.run_root)
    selected_by_task: Dict[str, Any] = {}
    replication_cells: List[SweepCell] = []
    for task in TASKS:
        seed7 = [
            SweepCell.from_dict(row)
            for key in ("stage1", "stage2")
            for row in manifest[key]
            if row["task"] == task
        ]
        selected = select_stage3_configs(_completed_results(args.run_root, seed7))
        selected_by_task[task] = [cell.to_dict() for cell in selected]
        replication_cells.extend(stage3_cells(selected))
    manifest["stage3_selection"] = selected_by_task
    manifest["stage3"] = [cell.to_dict() for cell in replication_cells]
    submitted, last_wave = _submit_cells_in_waves(
        cells=replication_cells,
        python_bin=args.python_bin,
        run_root=args.run_root,
        data_dir=args.data_dir,
        model_name=args.model_name,
        log_dir=args.log_dir,
        initial_dependencies=(),
        dry_run=False,
    )
    manifest["stage3_jobs"] = submitted
    selector_id = _submit_job(
        command=_stage_command(
            "finalize",
            python_bin=args.python_bin,
            run_root=args.run_root,
            data_dir=args.data_dir,
            model_name=args.model_name,
            log_dir=args.log_dir,
        ),
        job_name="code-atomic-final-select",
        log_dir=args.log_dir,
        dependencies=last_wave,
        gpu=False,
        dry_run=False,
    )
    manifest["final_selector_job_id"] = selector_id
    _write_manifest(args.run_root, manifest)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _final_winner(
    run_root: Path,
    selected_seed7: Sequence[SweepCell],
    replication_cells: Sequence[SweepCell],
) -> Tuple[SweepCell, List[Dict[str, Any]]]:
    groups: List[Tuple[SweepCell, List[Dict[str, Any]]]] = []
    for seed7 in selected_seed7:
        matching = [
            seed7,
            *[cell for cell in replication_cells if cell.config_key == seed7.config_key],
        ]
        metrics = [load_cell_metrics(run_root, cell) for cell in matching]
        if len(metrics) != 3:
            raise ValueError(f"Configuration {seed7.config_key} does not have three seeds")
        groups.append((seed7, metrics))
    best_mean = max(_mean([item["validation"]["exact_accuracy"] for item in metrics]) for _, metrics in groups)
    eligible = [
        (cell, metrics)
        for cell, metrics in groups
        if _mean([item["validation"]["exact_accuracy"] for item in metrics]) >= best_mean - 0.01
        and _mean([item["validation"]["format_accuracy"] for item in metrics]) >= 0.98
    ]
    if not eligible:
        eligible = groups
    eligible.sort(key=lambda item: (item[0].data_size, item[0].max_steps, item[0].learning_rate))
    return eligible[0]


def command_finalize(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.run_root)
    final_selections: Dict[str, Any] = {}
    evaluation_jobs: List[Dict[str, Any]] = []
    job_ids: List[str] = []
    previous_wave: List[str] = []
    pending_evaluations: List[Tuple[str, SweepCell, Path, List[str]]] = []
    for task in TASKS:
        selected_seed7 = [SweepCell.from_dict(row) for row in manifest["stage3_selection"][task]]
        replications = [SweepCell.from_dict(row) for row in manifest["stage3"] if row["task"] == task]
        winner, metrics = _final_winner(args.run_root, selected_seed7, replications)
        winning_cells = [
            winner,
            *[cell for cell in replications if cell.config_key == winner.config_key],
        ]
        final_selections[task] = {
            "configuration": winner.to_dict(),
            "validation_means": {
                "exact_accuracy": _mean([item["validation"]["exact_accuracy"] for item in metrics]),
                "format_accuracy": _mean([item["validation"]["format_accuracy"] for item in metrics]),
            },
            "cells": [cell.to_dict() for cell in winning_cells],
        }
        for cell in winning_cells:
            output_dir = args.run_root / "final_evaluation" / task / f"seed{cell.seed}"
            command = [
                str(args.python_bin),
                "-m",
                "self.experiments.coding_atomic_sweep",
                "evaluate-final",
                "--run-root",
                str(args.run_root),
                "--data-dir",
                str(args.data_dir),
                "--model-name",
                args.model_name,
                "--output-dir",
                str(output_dir),
                "--cell-json",
                json.dumps(cell.to_dict(), separators=(",", ":")),
            ]
            pending_evaluations.append((task, cell, output_dir, command))
    for wave_start in range(0, len(pending_evaluations), 4):
        wave_ids: List[str] = []
        for task, cell, _output_dir, command in pending_evaluations[wave_start : wave_start + 4]:
            job_id = _submit_job(
                command=command,
                job_name=f"code-{task}-final-seed{cell.seed}",
                log_dir=args.log_dir,
                dependencies=previous_wave,
                gpu=True,
                dry_run=False,
            )
            evaluation_jobs.append({"task": task, "cell": cell.to_dict(), "job_id": job_id})
            if job_id:
                wave_ids.append(job_id)
                job_ids.append(job_id)
        previous_wave = wave_ids
    manifest["final_selection"] = final_selections
    manifest["final_evaluation_jobs"] = evaluation_jobs
    collect_id = _submit_job(
        command=_stage_command(
            "collect",
            python_bin=args.python_bin,
            run_root=args.run_root,
            data_dir=args.data_dir,
            model_name=args.model_name,
            log_dir=args.log_dir,
        ),
        job_name="code-atomic-collect",
        log_dir=args.log_dir,
        dependencies=previous_wave,
        gpu=False,
        dry_run=False,
    )
    manifest["collector_job_id"] = collect_id
    _write_manifest(args.run_root, manifest)


def command_collect(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.run_root)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"tasks": {}, "acceptance_thresholds": {"mean_test": 0.85, "minimum_seed": 0.80, "format": 0.98, "max_train_test_gap": 0.10}}
    for task in TASKS:
        selection = manifest["final_selection"][task]
        test_metrics: List[Dict[str, Any]] = []
        train_metrics: List[Dict[str, Any]] = []
        final_payloads: List[Dict[str, Any]] = []
        for cell_payload in selection["cells"]:
            seed = int(cell_payload["seed"])
            path = args.run_root / "final_evaluation" / task / f"seed{seed}" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            final_payloads.append(payload)
            test_metrics.append(payload["test"])
            train_metrics.append(payload["train"])
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "data_size": cell_payload["data_size"],
                    "max_steps": cell_payload["max_steps"],
                    "learning_rate": cell_payload["learning_rate"],
                    "test_exact_accuracy": payload["test"]["exact_accuracy"],
                    "test_format_accuracy": payload["test"]["format_accuracy"],
                    "test_behavior_valid_accuracy": payload["test"]["behavior_valid_accuracy"],
                    "train_exact_accuracy": payload["train"]["exact_accuracy"],
                }
            )
        mean_test = _mean([metric["exact_accuracy"] for metric in test_metrics])
        minimum_test = min(metric["exact_accuracy"] for metric in test_metrics)
        mean_format = _mean([metric["format_accuracy"] for metric in test_metrics])
        mean_train = _mean([metric["exact_accuracy"] for metric in train_metrics])
        train_test_gap = mean_train - mean_test
        reliable = (
            mean_test >= 0.85
            and minimum_test >= 0.80
            and mean_format >= 0.98
            and train_test_gap <= 0.10
        )
        frontier_keys = sorted(
            set.intersection(
                *[
                    {
                        key
                        for key, value in payload.items()
                        if isinstance(value, dict) and "exact_accuracy" in value
                    }
                    for payload in final_payloads
                ]
            )
            - {"train", "test"}
        )
        frontier_means = {
            key: {
                "exact_accuracy": _mean([payload[key]["exact_accuracy"] for payload in final_payloads]),
                "format_accuracy": _mean([payload[key]["format_accuracy"] for payload in final_payloads]),
                "behavior_valid_accuracy": _mean(
                    [payload[key]["behavior_valid_accuracy"] for payload in final_payloads]
                ),
                "count": final_payloads[0][key]["count"],
            }
            for key in frontier_keys
        }
        base_path = args.run_root / "base" / task / "metrics.json"
        base_metrics = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
        summary["tasks"][task] = {
            **selection,
            "mean_test_exact_accuracy": mean_test,
            "minimum_seed_test_exact_accuracy": minimum_test,
            "mean_test_format_accuracy": mean_format,
            "mean_train_exact_accuracy": mean_train,
            "train_test_gap": train_test_gap,
            "frontier_means": frontier_means,
            "zero_shot": base_metrics,
            "atomic_supervision_reliable": reliable,
        }
    write_json(args.run_root / "summary.json", summary)
    csv_path = args.run_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _add_common_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--log-dir", type=Path, default=ROOT_DIR / "artifacts/logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model-name", default=str(DEFAULT_MODEL))
    prepare.add_argument("--bfcl-root", type=Path, default=DEFAULT_BFCL_ROOT)
    prepare.add_argument("--commitpack-json-path", type=Path)
    prepare.add_argument("--commitpack-yaml-path", type=Path)

    cell = subparsers.add_parser("run-cell")
    cell.add_argument("--task", choices=TASKS, required=True)
    cell.add_argument("--data-dir", type=Path, required=True)
    cell.add_argument("--model-name", default=str(DEFAULT_MODEL))
    cell.add_argument("--output-dir", type=Path, required=True)
    cell.add_argument("--data-size", type=int, required=True)
    cell.add_argument("--max-steps", type=int, required=True)
    cell.add_argument("--learning-rate", type=float, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--stage", type=int, required=True)
    cell.add_argument("--micro-batch-size", type=int)
    cell.add_argument("--evaluate-train", action="store_true")

    base = subparsers.add_parser("evaluate-base")
    base.add_argument("--task", choices=TASKS, required=True)
    base.add_argument("--data-dir", type=Path, required=True)
    base.add_argument("--model-name", default=str(DEFAULT_MODEL))
    base.add_argument("--output-dir", type=Path, required=True)

    final = subparsers.add_parser("evaluate-final")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--data-dir", type=Path, required=True)
    final.add_argument("--model-name", default=str(DEFAULT_MODEL))
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--cell-json", required=True)

    verify = subparsers.add_parser("verify-adapter")
    verify.add_argument("--task", choices=TASKS, required=True)
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--model-name", default=str(DEFAULT_MODEL))
    verify.add_argument("--adapter-dir", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--limit", type=int, default=2)

    submit = subparsers.add_parser("submit")
    _add_common_stage_arguments(submit)
    submit.add_argument("--dry-run", action="store_true")
    for command in ("stage2", "stage3", "finalize", "collect"):
        stage = subparsers.add_parser(command)
        _add_common_stage_arguments(stage)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "run-cell":
        run_cell(args)
    elif args.command == "evaluate-base":
        evaluate_base(args)
    elif args.command == "evaluate-final":
        evaluate_final(args)
    elif args.command == "verify-adapter":
        verify_adapter(args)
    elif args.command == "submit":
        submit_sweep(args)
    elif args.command == "stage2":
        command_stage2(args)
    elif args.command == "stage3":
        command_stage3(args)
    elif args.command == "finalize":
        command_finalize(args)
    elif args.command == "collect":
        command_collect(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
