"""vLLM-backed evaluation for saved candidate checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from self.core.data_io import ensure_dir, load_examples, sanitize_json_value, save_examples, write_json
from self.core.evaluation import parse_prediction, resolve_max_new_tokens
from self.core.task_protocols import task_for_name


JsonDict = Dict[str, Any]


class VllmEvaluationError(RuntimeError):
    """Raised when the vLLM evaluation worker fails."""


def _existing_dirs(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _prepend_env_paths(env: dict[str, str], key: str, paths: Sequence[Path]) -> None:
    existing = env.get(key)
    values = _existing_dirs(paths)
    if existing:
        values.append(existing)
    if values:
        env[key] = ":".join(values)


def _preferred_cuda_home() -> Path | None:
    candidates = [
        Path("/usr/local/cuda-13.0"),
        Path("/usr/local/cuda-13.1"),
        Path("/usr/local/cuda-13.2"),
        Path("/usr/local/cuda"),
    ]
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists() and (
            candidate / "targets" / "x86_64-linux" / "include" / "curand.h"
        ).exists():
            return candidate
    return None


def _python_env_cuda_include_dirs(python_bin_path: Path) -> list[Path]:
    env_root = python_bin_path.parent.parent
    site_packages = sorted((env_root / "lib").glob("python*/site-packages"))
    include_dirs: list[Path] = []
    for site_package in site_packages:
        include_dirs.extend(sorted(site_package.glob("nvidia/cu*/include")))
    return include_dirs


def evaluate_model_with_vllm_subprocess(
    *,
    model_dir: Path,
    task_name: str,
    examples: Sequence[Any],
    task: Any,
    output_dir: Path,
    batch_size: int,
    decode_max_new_tokens: int,
    python_bin: str | None,
    gpu_memory_utilization: float,
    dtype: str,
    flashinfer_sampler: str = "off",
    enforce_eager: bool = False,
    max_model_len: int = 0,
    max_num_seqs: int = 0,
    max_num_batched_tokens: int = 0,
) -> Tuple[float, Dict[int, float]]:
    """Evaluate a saved model in a separate Python process that imports vLLM."""

    if flashinfer_sampler not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported vLLM FlashInfer sampler mode: {flashinfer_sampler}")
    ensure_dir(output_dir)
    examples_path = output_dir / "eval_examples.jsonl"
    spec_path = output_dir / "vllm_eval_spec.json"
    result_path = output_dir / "vllm_eval_result.json"
    stdout_path = output_dir / "vllm_eval_stdout.txt"
    stderr_path = output_dir / "vllm_eval_stderr.txt"

    save_examples(examples_path, examples, task.serialize_example)
    write_json(
        spec_path,
        {
            "model_dir": str(model_dir),
            "task": task_name,
            "examples_path": str(examples_path),
            "result_path": str(result_path),
            "batch_size": int(batch_size),
            "decode_max_new_tokens": int(decode_max_new_tokens),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": str(dtype),
            "flashinfer_sampler": str(flashinfer_sampler),
            "enforce_eager": bool(enforce_eager),
            "max_model_len": int(max_model_len),
            "max_num_seqs": int(max_num_seqs),
            "max_num_batched_tokens": int(max_num_batched_tokens),
        },
    )

    command = [
        python_bin or sys.executable,
        "-m",
        "self.core.vllm_evaluation",
        "--spec",
        str(spec_path),
    ]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    python_bin_path = Path(command[0]).resolve()
    if python_bin_path.parent.exists():
        env["PATH"] = f"{python_bin_path.parent}:{env.get('PATH', '')}"
    cache_root = Path(repo_root) / "artifacts" / "cache" / "vllm"
    ensure_dir(cache_root)
    ensure_dir(cache_root / "flashinfer")
    env.setdefault("VLLM_CACHE_ROOT", str(cache_root))
    env.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(cache_root / "flashinfer"))
    env.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache_root / "flashinfer"))
    env.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    env.setdefault("VLLM_ENABLE_PREGRAD_PASSES", "0")
    if flashinfer_sampler == "off":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    elif flashinfer_sampler == "on":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    cuda_home = _preferred_cuda_home()
    if cuda_home is not None:
        env.setdefault("CUDA_HOME", str(cuda_home))
        env.setdefault("CUDA_PATH", str(cuda_home))
        _prepend_env_paths(env, "PATH", [cuda_home / "bin"])
    cuda_include_dirs: list[Path] = []
    if cuda_home is not None:
        cuda_include_dirs.extend(
            [
                cuda_home / "targets" / "x86_64-linux" / "include",
                cuda_home / "include",
            ]
        )
    cuda_include_dirs.extend(_python_env_cuda_include_dirs(python_bin_path))
    _prepend_env_paths(env, "CPATH", cuda_include_dirs)
    _prepend_env_paths(env, "CPLUS_INCLUDE_PATH", cuda_include_dirs)
    env.pop("VLLM_PYTHON_BIN", None)
    env.pop("VLLM_GPU_MEMORY_UTILIZATION", None)
    env.pop("VLLM_DTYPE", None)
    env.pop("VLLM_FLASHINFER_SAMPLER", None)
    env.pop("VLLM_ENFORCE_EAGER", None)
    env.pop("VLLM_MAX_MODEL_LEN", None)
    env.pop("VLLM_MAX_NUM_SEQS", None)
    env.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise VllmEvaluationError(
            "vLLM evaluation failed "
            f"(returncode={completed.returncode}, spec={spec_path}, stderr={stderr_path})"
        )
    if not result_path.exists():
        raise VllmEvaluationError(f"vLLM evaluation did not write result file: {result_path}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    per_size = {int(size): float(value) for size, value in dict(payload.get("per_size_accuracy", {})).items()}
    write_json(
        output_dir / "vllm_eval_summary.json",
        {
            "backend": "vllm",
            "model_dir": str(model_dir),
            "task": task_name,
            "examples": len(examples),
            "batch_size": int(batch_size),
            "max_new_tokens": int(payload.get("max_new_tokens", decode_max_new_tokens)),
            "python_bin": command[0],
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": str(dtype),
            "flashinfer_sampler": str(flashinfer_sampler),
            "enforce_eager": bool(enforce_eager),
            "max_model_len": int(max_model_len),
            "max_num_seqs": int(max_num_seqs),
            "max_num_batched_tokens": int(max_num_batched_tokens),
            "worker_runtime_seconds": payload.get("runtime_seconds"),
            "runtime_seconds": time.monotonic() - started,
            "result_path": str(result_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    return float(payload.get("accuracy", math.nan)), per_size


def evaluate_from_spec(spec_path: Path) -> JsonDict:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    os.environ.setdefault("VLLM_ENABLE_PREGRAD_PASSES", "0")
    flashinfer_sampler = str(payload.get("flashinfer_sampler", "off"))
    if flashinfer_sampler not in {"auto", "on", "off"}:
        raise VllmEvaluationError(f"Unsupported vLLM FlashInfer sampler mode: {flashinfer_sampler}")
    if flashinfer_sampler == "off":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    elif flashinfer_sampler == "on":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    task_name = str(payload["task"])
    task = task_for_name(task_name)
    examples = load_examples(Path(str(payload["examples_path"])), task.deserialize_example)
    max_new_tokens = resolve_max_new_tokens(examples, int(payload["decode_max_new_tokens"]))
    result = _evaluate_examples_with_vllm(
        model_dir=Path(str(payload["model_dir"])),
        examples=examples,
        task=task,
        batch_size=max(1, int(payload.get("batch_size", 1))),
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.80)),
        dtype=str(payload.get("dtype", "auto")),
        enforce_eager=bool(payload.get("enforce_eager", False)),
        max_model_len=max(0, int(payload.get("max_model_len", 0))),
        max_num_seqs=max(0, int(payload.get("max_num_seqs", 0))),
        max_num_batched_tokens=max(0, int(payload.get("max_num_batched_tokens", 0))),
    )
    result.update(
        {
            "backend": "vllm",
            "task": task_name,
            "model_dir": str(payload["model_dir"]),
            "examples": len(examples),
            "max_new_tokens": max_new_tokens,
        }
    )
    write_json(Path(str(payload["result_path"])), result)
    return result


def _evaluate_examples_with_vllm(
    *,
    model_dir: Path,
    examples: Sequence[Any],
    task: Any,
    batch_size: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    dtype: str,
    enforce_eager: bool = False,
    max_model_len: int = 0,
    max_num_seqs: int = 0,
    max_num_batched_tokens: int = 0,
) -> JsonDict:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - exercised only in the vLLM env
        raise VllmEvaluationError(f"Could not import vLLM: {exc}") from exc

    started = time.monotonic()
    llm_kwargs: JsonDict = {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": bool(enforce_eager),
    }
    if max_model_len > 0:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = int(max_num_seqs)
    if max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    total = len(examples)
    correct = 0
    size_totals: Dict[int, int] = defaultdict(int)
    size_correct: Dict[int, int] = defaultdict(int)
    samples: list[JsonDict] = []

    for start in range(0, total, batch_size):
        batch = examples[start : start + batch_size]
        prompts = [example.prompt() for example in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for example, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            prediction = parse_prediction(task.prediction_parser, text, example)
            target = example.target()
            size_value = int(task.size_of(example))
            size_totals[size_value] += 1
            if prediction == target:
                correct += 1
                size_correct[size_value] += 1
            if len(samples) < 16:
                samples.append(
                    {
                        "size": size_value,
                        "prompt": example.prompt(),
                        "target": target,
                        "text": text,
                        "prediction": prediction,
                        "correct": prediction == target,
                    }
                )

    accuracy = correct / total if total else math.nan
    per_size_accuracy = {
        int(size): size_correct[size] / count if count else math.nan
        for size, count in size_totals.items()
    }
    return sanitize_json_value(
        {
            "accuracy": accuracy,
            "per_size_accuracy": per_size_accuracy,
            "correct": correct,
            "total": total,
            "runtime_seconds": time.monotonic() - started,
            "llm_kwargs": {
                key: value
                for key, value in llm_kwargs.items()
                if key not in {"model", "tokenizer"}
            },
            "sample_predictions": samples,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vLLM evaluation from a JSON spec.")
    parser.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    evaluate_from_spec(args.spec)


if __name__ == "__main__":
    main()
