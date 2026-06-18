from __future__ import annotations

import json
from types import SimpleNamespace

from self.adaptive.traces import experience_outcome_rendering, experience_outcome_traces, experience_traces
from self.adaptive.traces.experience_outcome_traces import build_outcome_trace_example, build_round_outcome_trace_examples


def test_outcome_trace_builder_uses_compact_state_prediction_and_legacy_identity():
    args = SimpleNamespace(outcome_trace_target_mode="numeric_textual", invalid_outcome_reward=-0.25)
    proposal = SimpleNamespace(left=2, right=3, guard="no_carry", target=5)
    metric = SimpleNamespace(
        index=7,
        row_id="row-7",
        proposal=proposal,
        proposal_prediction={
            "expected_frontier_delta": 0.10,
            "expected_final_delta_from_init": 0.20,
        },
        valid=True,
        reward=0.75,
        target_delta=0.12,
        frontier_delta=0.15,
        final_accuracy_delta=0.25,
        final_accuracy_delta_from_current=0.05,
    )

    trace = build_outcome_trace_example(
        args=args,
        task_name="addition",
        condition="config",
        round_index=2,
        result={"proposal_index": 7, "valid": True, "id": "row-7"},
        metric=metric,
        selected=True,
        source_sizes=[2, 3],
        frontier_min=4,
        frontier_max=8,
        current_final_accuracy=0.3333333,
        init_final_accuracy=0.1111111,
        current_per_size_accuracy={2: 1.0, 3: 0.5, 5: 0.0},
    )

    assert experience_traces.build_outcome_trace_example is build_outcome_trace_example
    assert experience_traces.build_round_outcome_trace_examples is build_round_outcome_trace_examples
    assert "TASK: predict_config_outcome" in trace.prompt()
    assert '"source":[2,3]' in trace.prompt()
    assert '"target":5' in trace.prompt()

    completion = json.loads(trace.target())
    assert completion["selected"] is True
    assert completion["repeat_target"] is False
    assert completion["frontier_delta_error"] == 0.05
    assert completion["final_delta_from_init_error"] == 0.05
    assert "feedback" in completion


def test_outcome_rendering_owner_is_reexported_for_compatibility():
    assert experience_outcome_traces._outcome_completion is experience_outcome_rendering._outcome_completion
    assert (
        experience_outcome_traces._render_outcome_trace_prompt
        is experience_outcome_rendering._render_outcome_trace_prompt
    )
    assert (
        experience_outcome_traces._candidate_payload_from_result
        is experience_outcome_rendering._candidate_payload_from_result
    )


def test_round_outcome_trace_builder_skips_disabled_modes_and_bad_indices():
    args = SimpleNamespace(outcome_trace_target_mode="numeric", invalid_outcome_reward=-0.25)
    traces = build_round_outcome_trace_examples(
        args=args,
        task_name="addition",
        condition="config",
        round_index=0,
        proposal_results=[
            {"proposal_index": "bad"},
            {
                "proposal_index": 1,
                "valid": False,
                "validation_category": "enum_error",
                "validation_message": "bad guard",
                "parsed_proposal": {"left": 1, "right": 2, "guard": "bad", "target": 3},
            },
        ],
        metrics=[],
        selected=None,
        source_sizes=[1, 2],
        frontier_min=3,
        frontier_max=4,
        current_final_accuracy=0.4,
        init_final_accuracy=0.2,
        current_per_size_accuracy={1: 1.0},
    )

    assert len(traces) == 1
    assert traces[0].metadata["failure"] == "invalid_guard"
    assert json.loads(traces[0].target())["reward"] == -0.25

    args.outcome_trace_target_mode = "none"
    assert (
        build_round_outcome_trace_examples(
            args=args,
            task_name="addition",
            condition="config",
            round_index=0,
            proposal_results=[{"proposal_index": 1}],
            metrics=[],
            selected=None,
            source_sizes=[],
            frontier_min=0,
            frontier_max=0,
            current_final_accuracy=0.0,
            init_final_accuracy=0.0,
            current_per_size_accuracy={},
        )
        == []
    )
