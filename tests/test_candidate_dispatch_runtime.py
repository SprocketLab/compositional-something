from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.adaptive import candidate_dispatch as dispatch, candidate_workers as workers
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.adaptive.proposals import ConfigProposal, PromptBundle


def _work_item(index: int) -> CandidateWorkItem:
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    return CandidateWorkItem(
        index=index,
        row_id=f"row-{index}",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output=proposal.to_json_dict(),
        composed=ExactPairDataset(
            examples=[],
            component_map={},
            keys=set(),
            diagnostics={},
        ),
        pseudo_examples=[],
        pseudo_diagnostics={},
    )


def _metric(item: CandidateWorkItem, seed: int) -> CandidateMetrics:
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=True,
        reward=float(seed),
        frontier_delta=0.0,
        target_accuracy=0.0,
        current_target_accuracy=0.0,
        final_accuracy=0.0,
        current_final_accuracy=0.0,
        init_final_accuracy=0.0,
        final_accuracy_delta=0.0,
        final_accuracy_delta_from_current=0.0,
        per_size_accuracy={},
        pseudo_count=0,
        model_dir=None,
    )


def test_train_candidates_serial_scores_items_with_attempt_seed_offset(tmp_path: Path):
    args = SimpleNamespace(seed=11, candidate_local_cache_base_state=True)
    items = [_work_item(0), _work_item(3)]
    seen = []
    cache_ids = []
    cache_base_state = []

    def score_candidate_fn(**kwargs):
        seen.append((kwargs["item"].index, kwargs["seed"]))
        cache = kwargs["model_bootstrap_cache"]
        cache_ids.append(id(cache))
        cache_base_state.append(cache.cache_base_state)
        return _metric(kwargs["item"], kwargs["seed"])

    metrics = dispatch.train_candidates_serial(
        args=args,
        task=object(),
        current_checkpoint="checkpoint",
        source_examples=["source"],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="system", user="user"),
        round_index=2,
        work_items=items,
        round_dir=tmp_path,
        eval_examples=["eval"],
        current_final_accuracy=0.4,
        current_per_size_accuracy={},
        init_final_accuracy=0.1,
        config=object(),
        attempt_index=5,
        score_candidate_fn=score_candidate_fn,
    )

    assert seen == [(0, 11 + 5 * 1009), (3, 11 + 5 * 1009 + 3)]
    assert [metric.reward for metric in metrics] == [float(seed) for _, seed in seen]
    assert len(set(cache_ids)) == 1
    assert cache_base_state == [True, True]


def test_train_candidates_serial_supports_legacy_scorer_without_cache_kwarg(tmp_path: Path):
    args = SimpleNamespace(seed=11, candidate_local_cache_base_state=True)
    item = _work_item(0)
    seen = []

    def legacy_score_candidate_fn(
        *,
        args,
        task,
        current_checkpoint,
        source_examples,
        proposal_trace_buffer,
        outcome_trace_buffer,
        proposal_prompt,
        round_index,
        item,
        round_dir,
        eval_examples,
        current_final_accuracy,
        current_per_size_accuracy,
        init_final_accuracy,
        config,
        seed,
    ):
        del (
            args,
            task,
            current_checkpoint,
            source_examples,
            proposal_trace_buffer,
            outcome_trace_buffer,
            proposal_prompt,
            round_index,
            round_dir,
            eval_examples,
            current_final_accuracy,
            current_per_size_accuracy,
            init_final_accuracy,
            config,
        )
        seen.append(seed)
        return _metric(item, seed)

    metrics = dispatch.train_candidates_serial(
        args=args,
        task=object(),
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=2,
        work_items=[item],
        round_dir=tmp_path,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={},
        init_final_accuracy=0.1,
        config=object(),
        attempt_index=5,
        score_candidate_fn=legacy_score_candidate_fn,
    )

    assert seen == [11 + 5 * 1009]
    assert [metric.reward for metric in metrics] == [float(seen[0])]


def test_local_parallel_dispatch_sets_subprocess_binding_and_delegates(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    fake_subprocess = object()
    monkeypatch.setattr(workers, "subprocess", object())

    def fake_train_candidates_local_parallel(**kwargs):
        calls.append(kwargs)
        assert workers.subprocess is fake_subprocess
        return ["metric"]

    monkeypatch.setattr(
        workers,
        "train_candidates_local_parallel",
        fake_train_candidates_local_parallel,
    )

    result = dispatch.train_candidates_local_parallel(
        args=SimpleNamespace(seed=1),
        task=object(),
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=[_work_item(0)],
        round_dir=tmp_path,
        eval_examples=[],
        current_final_accuracy=0.0,
        current_per_size_accuracy={},
        init_final_accuracy=0.0,
        attempt_index=0,
        collect_metrics_fn=lambda **kwargs: [],
        subprocess_module=fake_subprocess,
    )

    assert result == ["metric"]
    assert calls[0]["round_dir"] == tmp_path
    assert calls[0]["work_items"][0].index == 0


def test_slurm_array_dispatch_delegates_to_candidate_workers(tmp_path: Path, monkeypatch):
    calls = []

    def fake_train_candidates_slurm_array(**kwargs):
        calls.append(kwargs)
        return ["metric"]

    monkeypatch.setattr(
        workers,
        "train_candidates_slurm_array",
        fake_train_candidates_slurm_array,
    )

    result = dispatch.train_candidates_slurm_array(
        args=SimpleNamespace(seed=1),
        task=object(),
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=[_work_item(2)],
        round_dir=tmp_path,
        eval_examples=[],
        current_final_accuracy=0.0,
        current_per_size_accuracy={},
        init_final_accuracy=0.0,
        attempt_index=0,
        collect_metrics_fn=lambda **kwargs: [],
    )

    assert result == ["metric"]
    assert calls[0]["round_dir"] == tmp_path
    assert calls[0]["work_items"][0].index == 2


def test_candidate_dispatch_helpers_live_in_merged_module():
    assert dispatch.CandidateDispatchEntrypointDeps.__module__ == "self.adaptive.candidate_dispatch"
    assert dispatch.train_candidates_serial.__module__ == "self.adaptive.candidate_dispatch"
    assert dispatch.train_candidates_local_parallel.__module__ == "self.adaptive.candidate_dispatch"
    assert dispatch.train_candidates_slurm_array.__module__ == "self.adaptive.candidate_dispatch"


def test_build_candidate_dispatch_deps_reads_driver_bindings():
    bindings = SimpleNamespace(
        train_and_score_candidate=object(),
        _candidate_failure_metrics=object(),
        _collect_candidate_array_metrics=object(),
        train_candidates_serial=object(),
        train_candidates_local_parallel=object(),
        train_candidates_slurm_array=object(),
        subprocess=object(),
    )

    deps = dispatch.build_candidate_dispatch_deps(bindings)

    assert deps.train_and_score_candidate is bindings.train_and_score_candidate
    assert deps.candidate_failure_metrics is bindings._candidate_failure_metrics
    assert deps.collect_candidate_array_metrics is bindings._collect_candidate_array_metrics
    assert deps.train_candidates_serial is bindings.train_candidates_serial
    assert deps.train_candidates_local_parallel is bindings.train_candidates_local_parallel
    assert deps.train_candidates_slurm_array is bindings.train_candidates_slurm_array
    assert deps.subprocess_module is bindings.subprocess
