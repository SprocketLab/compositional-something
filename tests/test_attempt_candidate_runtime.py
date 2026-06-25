import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from self.adaptive.attempts import CandidateAttemptDeps, run_candidate_attempt
from self.adaptive.attempts import AttemptOutcomeResult
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.adaptive.proposal import ConfigProposal
from self.adaptive.proposal import PromptBundle
from self.adaptive.run import RoundModelDispatchDeps, run_round_model_dispatch
from self.adaptive.run import RoundModelDispatchResult


class _CheckpointManager:
    def __init__(self) -> None:
        self.cleanup_calls: list[dict[str, Any]] = []

    def cleanup_unselected_candidates(self, **kwargs: Any) -> None:
        self.cleanup_calls.append(dict(kwargs))


def _metric(*, index: int, reward: float) -> CandidateMetrics:
    return CandidateMetrics(
        index=index,
        row_id=f"candidate-{index}",
        proposal=ConfigProposal(left=1, right=2, guard="none", target=3),
        valid=True,
        reward=reward,
        frontier_delta=reward,
        target_accuracy=0.50,
        current_target_accuracy=0.25,
        final_accuracy=0.55,
        init_final_accuracy=0.20,
        final_accuracy_delta=0.35,
        current_final_accuracy=0.40,
        final_accuracy_delta_from_current=0.15,
        per_size_accuracy={3: 0.50},
        pseudo_count=4,
        model_dir=Path(f"candidate_{index}") / "model",
    )


def _work_item(index: int) -> CandidateWorkItem:
    proposal = ConfigProposal(left=1, right=2, guard="none", target=3)
    return CandidateWorkItem(
        index=index,
        row_id=f"candidate-{index}",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output=proposal.to_completion(),
        composed=ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=[],
        pseudo_diagnostics={},
    )


def test_candidate_attempt_dispatches_training_selection_trace_and_outcome(tmp_path: Path) -> None:
    prompt = PromptBundle(system="round-system", user="round-user")
    work_items = [_work_item(0), _work_item(1)]
    metrics: Sequence[CandidateMetrics] = [_metric(index=0, reward=0.1), _metric(index=1, reward=0.4)]
    selected = metrics[1]
    checkpoint_manager = _CheckpointManager()
    round_model_dispatch_deps = object()
    attempt_outcome_deps = object()
    calls: list[str] = []
    captured: dict[str, Any] = {}

    def _run_round_model_dispatch(**kwargs: Any) -> RoundModelDispatchResult:
        calls.append("round_model")
        assert kwargs["deps"] is round_model_dispatch_deps
        assert kwargs["current_checkpoint"] == "checkpoint-current"
        assert kwargs["selected_round_for_prompt"] == 1
        assert kwargs["extra_aggregate_metrics"] == {}
        return RoundModelDispatchResult(
            current_final_accuracy=0.56,
            current_per_size_accuracy={3: 0.56},
            prompt=prompt,
            proposal_results=[{"id": "proposal-0"}],
            work_items=work_items,
        )

    def _train_candidate_metrics(**kwargs: Any) -> Sequence[CandidateMetrics]:
        calls.append("train")
        assert kwargs["proposal_prompt"] is prompt
        assert kwargs["work_items"] is work_items
        assert kwargs["current_final_accuracy"] == 0.56
        assert kwargs["current_per_size_accuracy"] == {3: 0.56}
        return metrics

    def _select_candidate(candidate_metrics: Sequence[CandidateMetrics], min_reward: float) -> CandidateMetrics:
        calls.append("select")
        assert candidate_metrics is metrics
        assert min_reward == 0.25
        return selected

    def _write_round_trace(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append("trace")
        assert kwargs["prompt"] is prompt
        assert kwargs["metrics"] is metrics
        assert kwargs["path"] == tmp_path / "run" / "attempt_0001" / "trace_examples.jsonl"
        return [{"completion": "{}"}]

    def _handle_attempt_outcome(**kwargs: Any) -> AttemptOutcomeResult:
        calls.append("outcome")
        captured.update(kwargs)
        return AttemptOutcomeResult(
            selected_rounds=1,
            consecutive_no_selection=0,
            current_checkpoint="checkpoint-next",
            current_final_accuracy=0.60,
            current_per_size_accuracy={3: 0.60},
            proposal_grpo_update_count=2,
        )

    result = run_candidate_attempt(
        args=argparse.Namespace(
            task="addition",
            selection_min_reward=0.25,
        ),
        task=object(),
        config=object(),
        output_dir=tmp_path / "run",
        round_dir=tmp_path / "run" / "attempt_0001",
        checkpoint_manager=checkpoint_manager,
        source_examples=[],
        source_sizes={1, 2},
        exclude_keys=set(),
        eval_examples=[],
        current_checkpoint="checkpoint-current",
        current_final_accuracy=0.40,
        current_per_size_accuracy={3: 0.40},
        init_final_accuracy=0.20,
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_grpo_update_count=1,
        summary_records=[
            {
                "attempt": 0,
                "attempt_actions": [
                    {"attempt": 0, "left": 1, "right": 1, "target": 2, "reward": 0.2},
                    {"attempt": 0, "left": 1, "right": 2, "target": 3, "reward": 0.1},
                ],
            }
        ],
        selected_round_for_prompt=1,
        attempt_index=1,
        selected_rounds=0,
        consecutive_no_selection=0,
        deps=CandidateAttemptDeps(
            run_round_model_dispatch=_run_round_model_dispatch,
            train_candidate_metrics=_train_candidate_metrics,
            select_candidate=_select_candidate,
            write_round_trace=_write_round_trace,
            handle_attempt_outcome=_handle_attempt_outcome,
            round_model_dispatch_deps=round_model_dispatch_deps,
            attempt_outcome_deps=attempt_outcome_deps,
        ),
    )

    assert calls == ["round_model", "train", "select", "trace", "outcome"]
    assert result.current_checkpoint == "checkpoint-next"
    assert checkpoint_manager.cleanup_calls == [{"metrics": metrics, "selected": selected}]
    assert captured["current_final_accuracy"] == 0.56
    assert captured["current_per_size_accuracy"] == {3: 0.56}
    assert captured["proposal_results"] == [{"id": "proposal-0"}]
    assert captured["metrics"] is metrics
    assert captured["work_items"] is work_items
    assert captured["selected"] is selected
    assert captured["trace_rows"] == [{"completion": "{}"}]
    assert captured["deleted_replaced_model_dirs"] == []
    assert captured["deps"] is attempt_outcome_deps


def test_round_model_dispatch_forwards_prompt_extras_to_serial_phase(tmp_path: Path) -> None:
    prompt = PromptBundle(system="system", user="user")
    captured: dict[str, Any] = {}

    def _run_round_model_phase(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            current_final_accuracy=0.5,
            current_per_size_accuracy={2: 0.5},
            prompt=prompt,
            proposal_results=[],
            work_items=[],
        )

    result = run_round_model_dispatch(
        args=argparse.Namespace(controller_execution_mode="local", seed=11),
        task=object(),
        config=object(),
        current_checkpoint="checkpoint",
        round_dir=tmp_path,
        source_examples=[],
        eval_examples=[],
        exclude_keys=set(),
        source_sizes={1},
        selected_round_for_prompt=1,
        attempt_index=2,
        selected_rounds=0,
        consecutive_no_selection=0,
        init_final_accuracy=0.1,
        extra_aggregate_metrics={"custom_metric": [{"target": 2}]},
        deps=RoundModelDispatchDeps(
            save_examples=lambda *args, **kwargs: None,
            write_key_set=lambda *args, **kwargs: None,
            run_controller_worker_slurm=lambda *args, **kwargs: {},
            float_or_nan=float,
            load_json=lambda path: [],
            work_item_from_worker_payload=lambda **kwargs: None,
            run_round_model_phase=_run_round_model_phase,
        ),
    )

    assert result.prompt is prompt
    assert captured["extra_aggregate_metrics"] == {"custom_metric": [{"target": 2}]}


def test_round_model_dispatch_forwards_prompt_extras_to_slurm_payload(tmp_path: Path) -> None:
    prompt_path = tmp_path / "proposal_prompt.json"
    proposal_results_path = tmp_path / "proposal_results.json"
    captured: dict[str, Any] = {}

    class _Task:
        @staticmethod
        def serialize_example(example: Any) -> dict[str, Any]:
            return dict(example)

    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _run_controller_worker_slurm(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        _write_json(prompt_path, {"system": "system", "user": "user"})
        _write_json(proposal_results_path, [])
        return {
            "current_final_accuracy": 0.5,
            "current_per_size_accuracy": {"2": 0.5},
            "prompt_path": str(prompt_path),
            "proposal_results_path": str(proposal_results_path),
            "work_items": [],
        }

    result = run_round_model_dispatch(
        args=argparse.Namespace(controller_execution_mode="slurm", seed=11),
        task=_Task(),
        config=object(),
        current_checkpoint="checkpoint",
        round_dir=tmp_path,
        source_examples=[],
        eval_examples=[],
        exclude_keys=set(),
        source_sizes={1},
        selected_round_for_prompt=1,
        attempt_index=2,
        selected_rounds=0,
        consecutive_no_selection=0,
        init_final_accuracy=0.1,
        extra_aggregate_metrics={"custom_metric": [{"target": 2}]},
        deps=RoundModelDispatchDeps(
            save_examples=lambda path, examples, serializer: _write_json(path, [serializer(ex) for ex in examples]),
            write_key_set=lambda path, values: _write_json(path, sorted(values)),
            run_controller_worker_slurm=_run_controller_worker_slurm,
            float_or_nan=float,
            load_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
            work_item_from_worker_payload=lambda **kwargs: None,
            run_round_model_phase=lambda **kwargs: None,
        ),
    )

    assert result.prompt.user == "user"
    assert captured["payload"]["extra_aggregate_metrics"] == {"custom_metric": [{"target": 2}]}
