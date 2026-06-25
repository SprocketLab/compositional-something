from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from self.core import vllm_evaluation


class _Example:
    def __init__(self, size: int, target: str):
        self.size = size
        self._target = target

    def prompt(self) -> str:
        return f"prompt-{self.size}"

    def target(self) -> str:
        return self._target


class _Task:
    @staticmethod
    def serialize_example(example: _Example) -> dict[str, object]:
        return {"size": example.size, "target": example.target()}


def test_vllm_subprocess_evaluator_writes_spec_and_reads_result(monkeypatch, tmp_path: Path) -> None:
    calls = {}
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    conda_cuda_include = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "include"
    conda_cuda_include.mkdir(parents=True)
    (conda_cuda_include / "curand.h").write_text("", encoding="utf-8")
    cuda_home = tmp_path / "cuda-13.0"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
    cuda_include = cuda_home / "targets" / "x86_64-linux" / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "curand.h").write_text("", encoding="utf-8")
    monkeypatch.setattr(vllm_evaluation, "_preferred_cuda_home", lambda: cuda_home)

    def fake_run(command, cwd, env, capture_output, text):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env"] = env
        spec_path = Path(command[-1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        Path(spec["result_path"]).write_text(
            json.dumps({"accuracy": 0.5, "per_size_accuracy": {"4": 0.25}, "max_new_tokens": 9}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(vllm_evaluation.subprocess, "run", fake_run)

    accuracy, per_size = vllm_evaluation.evaluate_model_with_vllm_subprocess(
        model_dir=tmp_path / "model",
        task_name="run_length",
        examples=[_Example(4, "answer")],
        task=_Task(),
        output_dir=tmp_path / "eval",
        batch_size=8,
        decode_max_new_tokens=7,
        python_bin=str(python_bin),
        gpu_memory_utilization=0.7,
        dtype="float16",
        flashinfer_sampler="off",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    )

    spec = json.loads((tmp_path / "eval" / "vllm_eval_spec.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "eval" / "vllm_eval_summary.json").read_text(encoding="utf-8"))
    assert calls["command"][:3] == [str(python_bin), "-m", "self.core.vllm_evaluation"]
    assert str(Path(calls["cwd"])) in calls["env"]["PYTHONPATH"]
    assert str(cuda_home / "bin") in calls["env"]["PATH"].split(":")
    assert calls["env"]["PATH"].split(":")[1] == str(python_bin.parent)
    assert calls["env"]["VLLM_CACHE_ROOT"].endswith("artifacts/cache/vllm")
    assert calls["env"]["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"].endswith("artifacts/cache/vllm/flashinfer")
    assert calls["env"]["FLASHINFER_WORKSPACE_BASE"].endswith("artifacts/cache/vllm/flashinfer")
    assert calls["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert calls["env"]["VLLM_USE_STANDALONE_COMPILE"] == "0"
    assert calls["env"]["VLLM_ENABLE_PREGRAD_PASSES"] == "0"
    assert str(cuda_include) in calls["env"]["CPATH"].split(":")
    assert str(conda_cuda_include) in calls["env"]["CPATH"].split(":")
    assert str(cuda_home) == calls["env"]["CUDA_HOME"]
    assert spec["model_dir"] == str(tmp_path / "model")
    assert spec["task"] == "run_length"
    assert spec["decode_max_new_tokens"] == 7
    assert spec["flashinfer_sampler"] == "off"
    assert spec["enforce_eager"] is True
    assert spec["max_model_len"] == 256
    assert spec["max_num_seqs"] == 8
    assert spec["max_num_batched_tokens"] == 1024
    assert summary["backend"] == "vllm"
    assert summary["python_bin"] == str(python_bin)
    assert summary["flashinfer_sampler"] == "off"
    assert summary["enforce_eager"] is True
    assert summary["max_model_len"] == 256
    assert summary["max_num_seqs"] == 8
    assert summary["max_num_batched_tokens"] == 1024
    assert (tmp_path / "eval" / "vllm_eval_stdout.txt").read_text(encoding="utf-8") == "ok"
    assert accuracy == 0.5
    assert per_size == {4: 0.25}


def test_vllm_subprocess_evaluator_fails_fast(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, cwd, env, capture_output, text):
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="missing vllm")

    monkeypatch.setattr(vllm_evaluation.subprocess, "run", fake_run)

    with pytest.raises(vllm_evaluation.VllmEvaluationError, match="returncode=3"):
        vllm_evaluation.evaluate_model_with_vllm_subprocess(
            model_dir=tmp_path / "model",
            task_name="run_length",
            examples=[_Example(4, "answer")],
            task=_Task(),
            output_dir=tmp_path / "eval",
            batch_size=8,
            decode_max_new_tokens=7,
            python_bin="/tmp/vllm-python",
            gpu_memory_utilization=0.7,
            dtype="float16",
        )

    assert (tmp_path / "eval" / "vllm_eval_stderr.txt").read_text(encoding="utf-8") == "missing vllm"
