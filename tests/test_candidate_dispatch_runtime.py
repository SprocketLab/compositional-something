from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.core import candidate_dispatch_runtime, candidate_execution, candidate_workers
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.core.proposals import ConfigProposal, PromptBundle


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
    args = SimpleNamespace(seed=11)
    items = [_work_item(0), _work_item(3)]
    seen = []

    def score_candidate_fn(**kwargs):
        seen.append((kwargs["item"].index, kwargs["seed"]))
        return _metric(kwargs["item"], kwargs["seed"])

    metrics = candidate_dispatch_runtime.train_candidates_serial(
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


def test_local_parallel_dispatch_sets_subprocess_binding_and_delegates(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    fake_subprocess = object()
    monkeypatch.setattr(candidate_workers, "subprocess", object())

    def fake_train_candidates_local_parallel(**kwargs):
        calls.append(kwargs)
        assert candidate_workers.subprocess is fake_subprocess
        return ["metric"]

    monkeypatch.setattr(
        candidate_workers,
        "train_candidates_local_parallel",
        fake_train_candidates_local_parallel,
    )

    result = candidate_dispatch_runtime.train_candidates_local_parallel(
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
        candidate_workers,
        "train_candidates_slurm_array",
        fake_train_candidates_slurm_array,
    )

    result = candidate_dispatch_runtime.train_candidates_slurm_array(
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


def test_candidate_execution_reexports_dispatch_runtime_helpers():
    assert candidate_execution.train_candidates_serial is candidate_dispatch_runtime.train_candidates_serial
    assert (
        candidate_execution.train_candidates_local_parallel
        is candidate_dispatch_runtime.train_candidates_local_parallel
    )
    assert (
        candidate_execution.train_candidates_slurm_array
        is candidate_dispatch_runtime.train_candidates_slurm_array
    )
