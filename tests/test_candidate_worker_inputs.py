from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from self.core.candidate_worker_inputs import (
    CandidateWorkerRuntimeDeps,
    load_candidate_worker_shared_inputs,
)


class _Task:
    def __init__(self) -> None:
        self.validated_args = []

    def validate_args(self, args) -> None:
        self.validated_args.append(args)

    @staticmethod
    def deserialize_example(payload):
        return dict(payload)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_candidate_worker_shared_inputs_cache_reuses_loaded_artifacts(tmp_path: Path):
    source_path = tmp_path / "inputs" / "source.jsonl"
    eval_path = tmp_path / "inputs" / "eval.jsonl"
    proposal_trace_path = tmp_path / "inputs" / "proposal_trace.jsonl"
    outcome_trace_path = tmp_path / "inputs" / "outcome_trace.jsonl"
    prompt_path = tmp_path / "inputs" / "proposal_prompt.json"
    _write_jsonl(source_path, [{"name": "source"}])
    _write_jsonl(eval_path, [{"name": "eval"}])
    _write_jsonl(proposal_trace_path, [{"trace": "proposal"}])
    _write_jsonl(outcome_trace_path, [{"trace": "outcome"}])
    prompt_path.write_text(json.dumps({"system": "system", "user": "user"}), encoding="utf-8")

    task = _Task()
    trace_loads = []
    prompt_loads = []

    def namespace_from_json_args(payload):
        return Namespace(
            task=payload["task"],
            bf16=False,
            fp16=False,
            candidate_local_cache_base_state=True,
        )

    def load_trace_jsonl(path, parser_fn):
        trace_loads.append((Path(path), parser_fn))
        return [{"path": Path(path).name}]

    def load_json(path):
        prompt_loads.append(Path(path))
        return json.loads(Path(path).read_text(encoding="utf-8"))

    deps = CandidateWorkerRuntimeDeps(
        load_json=load_json,
        namespace_from_json_args=namespace_from_json_args,
        normalize_args=lambda args: args,
        task_for_name=lambda task_name: task,
        make_config=lambda args: {"task": args.task},
        load_trace_jsonl=load_trace_jsonl,
        train_and_score_candidate=lambda **kwargs: None,
        write_json=lambda path, payload: None,
    )
    payload = {
        "args": {"task": "addition"},
        "source_examples_path": str(source_path),
        "eval_examples_path": str(eval_path),
        "proposal_trace_buffer_path": str(proposal_trace_path),
        "outcome_trace_buffer_path": str(outcome_trace_path),
        "proposal_prompt_path": str(prompt_path),
    }
    shared_cache = {}

    first = load_candidate_worker_shared_inputs(
        payload,
        tmp_path / "spec_0.json",
        deps=deps,
        shared_cache=shared_cache,
    )
    second = load_candidate_worker_shared_inputs(
        payload,
        tmp_path / "spec_1.json",
        deps=deps,
        shared_cache=shared_cache,
    )

    assert first is second
    assert first.source_examples == [{"name": "source"}]
    assert first.eval_examples == [{"name": "eval"}]
    assert first.proposal_trace_buffer == [{"path": "proposal_trace.jsonl"}]
    assert first.outcome_trace_buffer == [{"path": "outcome_trace.jsonl"}]
    assert first.proposal_prompt.system == "system"
    assert first.proposal_prompt.user == "user"
    assert first.model_bootstrap_cache is not None
    assert first.model_bootstrap_cache.cache_base_state is True
    assert len(shared_cache) == 1
    assert len(trace_loads) == 2
    assert prompt_loads == [prompt_path]
    assert task.validated_args == [first.args]
