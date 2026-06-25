from __future__ import annotations

import json
from pathlib import Path

from self.adaptive import proposal as proposals
from self.adaptive import proposal as proposal_config_schema, proposal as proposal_prompts
from self.adaptive import proposal as proposal_io


def test_config_proposal_schema_accepts_valid_json_and_trace_write(tmp_path: Path):
    raw = json.dumps(
        {
            "left": 8,
            "right": 9,
            "guard": "none",
            "notes": "test",
        }
    )

    validation = proposals.parse_config_proposal(
        raw,
        task_name="run_length",
        source_min_allowed=8,
        source_max_allowed=16,
        frontier_min_allowed=17,
        frontier_max_allowed=48,
        guards=["none"],
    )

    assert validation.valid
    assert validation.proposal.to_json_dict()["target"] == 17
    assert validation.proposal.to_completion().startswith('{"guard":"none"')

    trace_path = tmp_path / "trace.jsonl"
    proposals.write_trace_jsonl(
        trace_path,
        [
            proposals.build_trace_row(
                round_index=1,
                task="run_length",
                condition="config",
                reward=0.5,
                frontier_delta=0.4,
                final_accuracy=1.0,
                prompt="prompt",
                completion=validation.proposal.to_completion(),
            )
        ],
    )
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["completion"] == validation.proposal.to_completion()


def test_proposal_io_owner_reexports() -> None:
    assert proposals.load_fixture_proposals is proposal_io.load_fixture_proposals
    assert proposals.build_trace_row is proposal_io.build_trace_row
    assert proposals.write_trace_jsonl is proposal_io.write_trace_jsonl


def test_config_schema_owner_reexports() -> None:
    assert proposals.ConfigProposal is proposal_config_schema.ConfigProposal
    assert proposals.DEFAULT_CONFIG_SEARCH_SPACES is proposal_config_schema.DEFAULT_CONFIG_SEARCH_SPACES
    assert proposals.parse_config_proposal is proposal_config_schema.parse_config_proposal
    assert proposals.normalized_config_completion is proposal_config_schema.normalized_config_completion


def test_prompt_owner_reexports() -> None:
    assert proposals.PromptBundle is proposal_prompts.PromptBundle
    assert proposals.render_config_prompt is proposal_prompts.render_config_prompt


def test_action_observation_config_prompt_uses_schema_without_concrete_action_example() -> None:
    prompt = proposals.render_config_prompt(
        task_name="addition",
        round_index=1,
        current_source={"sizes": [3, 4, 5, 6, 7]},
        allowed_target_frontier={"min_size": 8, "max_size": 14},
        aggregate_metrics={"current_avg_accuracy": 0.4, "accuracy_by_size": {"8": 0.0}},
        guard_choices=["none", "reject_boundary_carry"],
        proposal_output_schema="action_observation",
    )

    assert "Return exactly one JSON object with these keys" in prompt.user
    assert "expected_avg_delta_from_current" in prompt.user
    assert "expected_target_delta" in prompt.user
    assert "expected_frontier_delta" in prompt.user
    assert "Put prediction fields after reasoning and before left/right/guard." in prompt.user
    assert "Predict only expected deltas" in prompt.user
    assert "Do not copy numeric values from these instructions" not in prompt.user
    assert "Source sizes 5 and 3" not in prompt.user
    assert '"left": 5' not in prompt.user
    assert '"right": 3' not in prompt.user
    assert "target 8 is in the frontier" not in prompt.user
    assert "Output exactly one JSON object like this" not in prompt.user


def test_config_proposal_schema_rejects_ranges_and_enums():
    invalid_target = proposals.parse_config_proposal(
        {
            "left": 3,
            "right": 3,
            "guard": "none",
        },
        task_name="addition",
        source_min_allowed=3,
        source_max_allowed=7,
        frontier_min_allowed=8,
        frontier_max_allowed=12,
        guards=["none"],
    )
    invalid_enum = proposals.parse_config_proposal(
        {
            "left": 4,
            "right": 4,
            "guard": "bad",
        },
        task_name="addition",
        source_min_allowed=3,
        source_max_allowed=7,
        frontier_min_allowed=8,
        frontier_max_allowed=12,
        guards=["none"],
    )
    old_driver_schema = proposals.parse_config_proposal(
        {
            "proposal_type": "config",
            "task": "addition",
            "frontier_min": 8,
            "frontier_max": 10,
            "composition_path": "fixed_binary",
            "compose_arity": "exact2",
            "filter_rule": "with_carry_filtered",
            "examples_per_size": 128,
        },
        task_name="addition",
        source_min_allowed=3,
        source_max_allowed=7,
        frontier_min_allowed=8,
        frontier_max_allowed=12,
        guards=["none"],
    )

    assert not invalid_target.valid
    assert invalid_target.category == "range_error"
    assert not invalid_enum.valid
    assert invalid_enum.category == "enum_error"
    assert not old_driver_schema.valid
    assert old_driver_schema.category == "schema_error"
