from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.adaptive import candidate_workers as workers
from self.adaptive.candidate_workers import CandidateWorkerRuntimeDeps


def _deps() -> CandidateWorkerRuntimeDeps:
    return CandidateWorkerRuntimeDeps(
        load_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        namespace_from_json_args=lambda payload: SimpleNamespace(**payload),
        normalize_args=lambda args: args,
        task_for_name=lambda name: None,
        make_config=lambda args: None,
        load_trace_jsonl=lambda path, loader: [],
        train_and_score_candidate=lambda **kwargs: None,
        write_json=lambda path, payload: Path(path).write_text(json.dumps(payload), encoding="utf-8"),
    )


def test_pack_runtime_passes_shared_cache_to_cache_aware_runner(tmp_path: Path):
    spec_paths = [tmp_path / "candidate_0.json", tmp_path / "candidate_1.json"]
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps({"spec_paths": [str(path) for path in spec_paths]}),
        encoding="utf-8",
    )
    cache_ids: list[int] = []

    def run_from_spec(path: Path, *, shared_cache):
        cache_ids.append(id(shared_cache))
        shared_cache.setdefault("inputs", SimpleNamespace(model_bootstrap_cache=None))
        return SimpleNamespace(index=len(cache_ids) - 1)

    summary = workers.run_candidate_worker_pack_from_spec(
        pack_path,
        deps=_deps(),
        run_from_spec_fn=run_from_spec,
    )

    assert summary["total"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["shared_input_cache_entries"] == 1
    assert cache_ids[0] == cache_ids[1]
    assert [row["candidate_index"] for row in summary["results"]] == [0, 1]


def test_pack_runtime_supports_legacy_runner_without_shared_cache(tmp_path: Path):
    spec_path = tmp_path / "candidate_0.json"
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({"spec_paths": [str(spec_path)]}), encoding="utf-8")
    calls: list[Path] = []

    def run_from_spec(path: Path):
        calls.append(path)
        return SimpleNamespace(index=5)

    summary = workers.run_candidate_worker_pack_from_spec(
        pack_path,
        deps=_deps(),
        run_from_spec_fn=run_from_spec,
    )

    assert calls == [spec_path]
    assert summary["shared_input_cache_entries"] == 0
    assert summary["results"] == [
        {
            "spec_path": str(spec_path),
            "status": "ok",
            "candidate_index": 5,
        }
    ]


def test_candidate_worker_runtime_reexports_pack_runtime_for_compatibility():
    assert (
        workers.run_candidate_worker_pack_from_spec
        is workers.run_candidate_worker_pack_from_spec
    )
