import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from self.adaptive import attempts
from self.adaptive.attempts import AttemptOutcomeDeps, handle_attempt_outcome
from self.core.data_io import save_examples, write_json
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.adaptive.proposal import ConfigProposal
from self.adaptive.proposal import write_trace_jsonl
from self.adaptive.proposal import PromptBundle


@dataclass(frozen=True)
class _Trace:
    label: str

    def to_json_dict(self) -> dict[str, str]:
        return {"label": self.label}


class _CheckpointManager:
    def cleanup_replaced_checkpoint(
        self,
        *,
        old_checkpoint: str,
        new_checkpoint: str,
        protected_checkpoints: Sequence[str] = (),
    ) -> list[str]:
        if old_checkpoint in protected_checkpoints:
            return []
        return [f"deleted:{old_checkpoint}->{new_checkpoint}"]


class _Task:
    @staticmethod
    def serialize_example(example: Any) -> dict[str, Any]:
        return dict(example)


def _candidate_metric() -> CandidateMetrics:
    return CandidateMetrics(
        index=0,
        row_id="candidate-0",
        proposal=ConfigProposal(left=1, right=2, guard="none", target=3),
        valid=True,
        reward=-0.1,
        frontier_delta=-0.1,
        target_accuracy=0.25,
        current_target_accuracy=0.35,
        final_accuracy=0.40,
        init_final_accuracy=0.30,
        final_accuracy_delta=0.10,
        current_final_accuracy=0.42,
        final_accuracy_delta_from_current=-0.02,
        per_size_accuracy={1: 1.0, 3: 0.25},
        pseudo_count=4,
        model_dir=None,
        failure_reason="below selection threshold",
    )


def test_attempt_outcome_helpers_live_in_merged_module() -> None:
    assert attempts.AttemptOutcomeDeps is attempts.AttemptOutcomeDeps
    assert attempts.AttemptOutcomeResult is attempts.AttemptOutcomeResult
    assert attempts.handle_no_selection_attempt.__module__ == "self.adaptive.attempts"
    assert attempts.handle_selected_attempt.__module__ == "self.adaptive.attempts"


def test_no_selection_attempt_writes_summary_and_updates_proposal_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    round_dir = output_dir / "attempt_0001"
    apply_calls: list[dict[str, Any]] = []

    def _build_outcome_traces(**_: Any) -> list[_Trace]:
        return [_Trace("outcome")]

    def _apply_grpo_update(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        apply_calls.append(dict(kwargs))
        return str(round_dir / "proposal_grpo" / "model"), {"skipped": False, "loss": 0.5}

    deps = AttemptOutcomeDeps(
        build_round_outcome_trace_examples=_build_outcome_traces,
        build_selected_proposal_trace_example=lambda **_: _Trace("selected"),
        apply_or_dispatch_proposal_grpo_update=_apply_grpo_update,
        write_json=write_json,
        write_trace_jsonl=write_trace_jsonl,
        save_examples=save_examples,
    )
    args = argparse.Namespace(
        task="addition",
        condition="config",
        frontier_min_size=3,
        frontier_max_size=5,
        selection_min_reward=0.0,
        no_selection_patience=3,
        max_attempt_rounds=5,
        max_selected_rounds=2,
        seed=11,
    )
    metrics: Sequence[CandidateMetrics] = [_candidate_metric()]

    result = handle_attempt_outcome(
        args=args,
        task=None,
        output_dir=output_dir,
        round_dir=round_dir,
        attempt_index=1,
        selected_round_for_prompt=0,
        selected_rounds=0,
        consecutive_no_selection=0,
        current_checkpoint="checkpoint-current",
        current_final_accuracy=0.42,
        current_per_size_accuracy={1: 1.0, 3: 0.35},
        init_final_accuracy=0.30,
        source_sizes={1, 2},
        source_examples=[],
        exclude_keys=set(),
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_grpo_update_count=0,
        deleted_replaced_model_dirs=[],
        summary_records=[],
        prompt=PromptBundle(system="system", user="user"),
        proposal_results=[{"raw_completion": "{}"}],
        metrics=metrics,
        work_items=[],
        selected=None,
        trace_rows=[{"completion": "{}"}],
        checkpoint_manager=_CheckpointManager(),
        deps=deps,
    )

    assert result.selected_rounds == 0
    assert result.consecutive_no_selection == 1
    assert result.proposal_grpo_update_count == 1
    assert result.current_checkpoint == str(round_dir / "proposal_grpo" / "model")
    assert result.should_break is False
    assert apply_calls[0]["source_checkpoint"] == "checkpoint-current"
    assert apply_calls[0]["seed"] == 1554

    with (round_dir / "attempt_summary.json").open("r", encoding="utf-8") as handle:
        attempt_summary = json.load(handle)
    assert attempt_summary["no_selection"] is True
    assert attempt_summary["proposal_grpo"]["skipped"] is False
    assert attempt_summary["proposal_grpo"]["deleted_replaced_model_dirs"] == [
        f"deleted:checkpoint-current->{round_dir / 'proposal_grpo' / 'model'}"
    ]
    assert attempt_summary["candidate_metrics_path"] == str(round_dir / "candidate_metrics.json")

    with (output_dir / "adaptive_candidate_training_results.json").open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    assert run_summary[0]["no_selection"] is True
    assert run_summary[0]["current_checkpoint"] == str(round_dir / "proposal_grpo" / "model")

    with (round_dir / "candidate_metrics.json").open("r", encoding="utf-8") as handle:
        candidate_metrics = json.load(handle)
    assert candidate_metrics[0]["parsed_proposal"] == {
        "guard": "none",
        "left": 1,
        "notes": "",
        "right": 2,
        "target": 3,
    }

    with (round_dir / "outcome_trace_examples.jsonl").open("r", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"label": "outcome"}


def test_selected_attempt_updates_source_pool_and_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    round_dir = output_dir / "attempt_0001"
    selected_model_dir = output_dir / "selected_model"
    apply_calls: list[dict[str, Any]] = []

    def _build_outcome_traces(**_: Any) -> list[_Trace]:
        return [_Trace("outcome")]

    def _build_selected_trace(**_: Any) -> _Trace:
        return _Trace("selected")

    def _apply_grpo_update(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        apply_calls.append(dict(kwargs))
        return str(round_dir / "proposal_grpo" / "model"), {"skipped": False}

    deps = AttemptOutcomeDeps(
        build_round_outcome_trace_examples=_build_outcome_traces,
        build_selected_proposal_trace_example=_build_selected_trace,
        apply_or_dispatch_proposal_grpo_update=_apply_grpo_update,
        write_json=write_json,
        write_trace_jsonl=write_trace_jsonl,
        save_examples=save_examples,
    )
    args = argparse.Namespace(
        task="addition",
        condition="config",
        frontier_min_size=3,
        frontier_max_size=5,
        selection_min_reward=0.0,
        no_selection_patience=3,
        max_attempt_rounds=5,
        max_selected_rounds=1,
        source_admission_target_accuracy_threshold=0.80,
        seed=11,
    )
    selected = replace(
        _candidate_metric(),
        reward=0.25,
        frontier_delta=0.25,
        target_accuracy=0.85,
        final_accuracy=0.60,
        per_size_accuracy={1: 1.0, 3: 0.60},
        model_dir=selected_model_dir,
        failure_reason=None,
    )
    pseudo_examples = [{"input": "1+2", "target": "3"}]
    work_items = [
        CandidateWorkItem(
            index=selected.index,
            row_id=selected.row_id,
            proposal=selected.proposal,
            completion=selected.proposal.to_completion(),
            raw_output=selected.proposal.to_completion(),
            composed=ExactPairDataset(
                examples=[],
                component_map={},
                keys={("addition", selected.proposal.target)},
                diagnostics={},
            ),
            pseudo_examples=pseudo_examples,
            pseudo_diagnostics={},
        )
    ]
    source_examples: list[Any] = []
    exclude_keys: set[Any] = set()
    source_sizes = {1, 2}
    proposal_trace_buffer: list[Any] = []

    result = handle_attempt_outcome(
        args=args,
        task=_Task(),
        output_dir=output_dir,
        round_dir=round_dir,
        attempt_index=1,
        selected_round_for_prompt=0,
        selected_rounds=0,
        consecutive_no_selection=2,
        current_checkpoint="checkpoint-current",
        current_final_accuracy=0.42,
        current_per_size_accuracy={1: 1.0, 3: 0.35},
        init_final_accuracy=0.30,
        source_sizes=source_sizes,
        source_examples=source_examples,
        exclude_keys=exclude_keys,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=[],
        proposal_grpo_update_count=0,
        deleted_replaced_model_dirs=[],
        summary_records=[],
        prompt=PromptBundle(system="system", user="user"),
        proposal_results=[{"raw_completion": "{}"}],
        metrics=[selected],
        work_items=work_items,
        selected=selected,
        trace_rows=[{"completion": "{}"}],
        checkpoint_manager=_CheckpointManager(),
        deps=deps,
    )

    assert result.selected_rounds == 1
    assert result.consecutive_no_selection == 0
    assert result.current_checkpoint == str(selected_model_dir)
    assert result.current_final_accuracy == 0.60
    assert result.current_per_size_accuracy == {1: 1.0, 3: 0.60}
    assert result.proposal_grpo_update_count == 0
    assert apply_calls == []
    assert source_examples == pseudo_examples
    assert source_sizes == {1, 2, 3}
    assert exclude_keys == {("addition", 3)}
    assert [trace.to_json_dict() for trace in proposal_trace_buffer] == [{"label": "selected"}]

    with (round_dir / "round_summary.json").open("r", encoding="utf-8") as handle:
        round_summary = json.load(handle)
    assert round_summary["selected_round"] == 1
    assert round_summary["source_sizes_after"] == [1, 2, 3]
    assert round_summary["source_example_count_after"] == 1
    assert round_summary["source_admission"] == {
        "admitted": True,
        "reason": "target_accuracy_clears_threshold",
        "target": 3,
        "target_accuracy": 0.85,
        "threshold": 0.8,
    }
    assert round_summary["current_checkpoint"] == str(selected_model_dir)
    assert round_summary["proposal_grpo"] is None
    assert round_summary["deleted_replaced_model_dirs"] == [
        f"deleted:checkpoint-current->{selected_model_dir}"
    ]

    with (round_dir / "selected_pseudo_examples.jsonl").open("r", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"input": "1+2", "target": "3"}

    with (output_dir / "selected_proposal_trace_buffer.jsonl").open("r", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"label": "selected"}


def test_selected_attempt_does_not_admit_low_accuracy_target_to_source_pool(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    round_dir = output_dir / "attempt_0001"
    selected_model_dir = output_dir / "selected_model"

    deps = AttemptOutcomeDeps(
        build_round_outcome_trace_examples=lambda **_: [],
        build_selected_proposal_trace_example=lambda **_: _Trace("selected"),
        apply_or_dispatch_proposal_grpo_update=lambda **_: ("unused", {"skipped": True}),
        write_json=write_json,
        write_trace_jsonl=write_trace_jsonl,
        save_examples=save_examples,
    )
    args = argparse.Namespace(
        task="addition",
        condition="config",
        frontier_min_size=3,
        frontier_max_size=5,
        selection_min_reward=0.0,
        no_selection_patience=3,
        max_attempt_rounds=5,
        max_selected_rounds=1,
        source_admission_target_accuracy_threshold=0.80,
        seed=11,
    )
    selected = replace(
        _candidate_metric(),
        reward=0.25,
        frontier_delta=0.25,
        target_accuracy=0.60,
        final_accuracy=0.60,
        per_size_accuracy={1: 1.0, 3: 0.60},
        model_dir=selected_model_dir,
        failure_reason=None,
    )
    work_items = [
        CandidateWorkItem(
            index=selected.index,
            row_id=selected.row_id,
            proposal=selected.proposal,
            completion=selected.proposal.to_completion(),
            raw_output=selected.proposal.to_completion(),
            composed=ExactPairDataset(
                examples=[],
                component_map={},
                keys={("addition", selected.proposal.target)},
                diagnostics={},
            ),
            pseudo_examples=[{"input": "1+2", "target": "3"}],
            pseudo_diagnostics={},
        )
    ]
    source_examples: list[Any] = []
    exclude_keys: set[Any] = set()
    source_sizes = {1, 2}

    result = handle_attempt_outcome(
        args=args,
        task=_Task(),
        output_dir=output_dir,
        round_dir=round_dir,
        attempt_index=1,
        selected_round_for_prompt=0,
        selected_rounds=0,
        consecutive_no_selection=0,
        current_checkpoint="checkpoint-current",
        current_final_accuracy=0.42,
        current_per_size_accuracy={1: 1.0, 3: 0.35},
        init_final_accuracy=0.30,
        source_sizes=source_sizes,
        source_examples=source_examples,
        exclude_keys=exclude_keys,
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_grpo_update_count=0,
        deleted_replaced_model_dirs=[],
        summary_records=[],
        prompt=PromptBundle(system="system", user="user"),
        proposal_results=[{"raw_completion": "{}"}],
        metrics=[selected],
        work_items=work_items,
        selected=selected,
        trace_rows=[],
        checkpoint_manager=_CheckpointManager(),
        deps=deps,
    )

    assert result.current_checkpoint == str(selected_model_dir)
    assert source_examples == []
    assert source_sizes == {1, 2}
    assert exclude_keys == set()
    with (round_dir / "round_summary.json").open("r", encoding="utf-8") as handle:
        round_summary = json.load(handle)
    assert round_summary["source_sizes_after"] == [1, 2]
    assert round_summary["source_example_count_after"] == 0
    assert round_summary["source_admission"]["admitted"] is False
    assert round_summary["source_admission"]["reason"] == "target_accuracy_below_threshold"
