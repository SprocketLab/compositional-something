import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from self.core import attempt_no_selection_runtime, attempt_outcome_models, attempt_outcome_runtime
from self.core.attempt_outcome_runtime import AttemptOutcomeDeps, handle_attempt_outcome
from self.core.data_io import save_examples, write_json
from self.core.models import CandidateMetrics
from self.core.proposal_config_schema import ConfigProposal
from self.core.proposal_io import write_trace_jsonl
from self.core.proposal_prompts import PromptBundle


@dataclass(frozen=True)
class _Trace:
    label: str

    def to_json_dict(self) -> dict[str, str]:
        return {"label": self.label}


class _CheckpointManager:
    def cleanup_replaced_checkpoint(self, *, old_checkpoint: str, new_checkpoint: str) -> list[str]:
        return [f"deleted:{old_checkpoint}->{new_checkpoint}"]


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


def test_attempt_outcome_model_and_no_selection_compat_aliases() -> None:
    assert attempt_outcome_runtime.AttemptOutcomeDeps is attempt_outcome_models.AttemptOutcomeDeps
    assert attempt_outcome_runtime.AttemptOutcomeResult is attempt_outcome_models.AttemptOutcomeResult
    assert (
        attempt_outcome_runtime._handle_no_selection_attempt
        is attempt_no_selection_runtime.handle_no_selection_attempt
    )


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
        num_rounds=2,
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
