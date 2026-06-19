from __future__ import annotations

import json
from pathlib import Path

from self.adaptive import proposal as proposals
from self.adaptive import program_sandbox
from self.adaptive import proposal as proposal_config_schema, proposal as proposal_prompts
from self.adaptive import proposal as proposal_io


VALID_RUN_LENGTH_PROGRAM = '''def compose(components, metadata):
    if not components:
        return {"accept": False, "reason": "no_components"}
    parsed = []
    for c in components:
        parts = str(c["prediction"]).split("|")
        if len(parts) != 5:
            return {"accept": False, "reason": "bad_component_format"}
        try:
            max_run = int(parts[0])
            prefix_symbol = parts[1]
            prefix_run = int(parts[2])
            suffix_symbol = parts[3]
            suffix_run = int(parts[4])
            size = int(c["size"])
        except ValueError:
            return {"accept": False, "reason": "bad_component_format"}
        if max_run < 0 or prefix_run < 0 or suffix_run < 0:
            return {"accept": False, "reason": "negative_run"}
        if max_run > size or prefix_run > size or suffix_run > size:
            return {"accept": False, "reason": "run_exceeds_size"}
        parsed.append((size, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run))
    size, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = parsed[0]
    for right in parsed[1:]:
        r_size, r_max, r_prefix_symbol, r_prefix_run, r_suffix_symbol, r_suffix_run = right
        boundary = suffix_run + r_prefix_run if suffix_symbol == r_prefix_symbol else 0
        new_size = size + r_size
        new_max = max(max_run, r_max, boundary)
        new_prefix_run = prefix_run
        if prefix_run == size and prefix_symbol == r_prefix_symbol:
            new_prefix_run = size + r_prefix_run
        new_suffix_run = r_suffix_run
        if r_suffix_run == r_size and suffix_symbol == r_suffix_symbol:
            new_suffix_run = r_size + suffix_run
        size = new_size
        max_run = new_max
        prefix_run = new_prefix_run
        suffix_symbol = r_suffix_symbol
        suffix_run = new_suffix_run
    return {"accept": True, "target": f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"}
'''


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
    assert proposals.render_program_prompt is proposal_prompts.render_program_prompt
    assert proposals.render_program_repair_prompt is proposal_prompts.render_program_repair_prompt


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


def test_program_sandbox_accepts_run_length_property_cases():
    result = program_sandbox.validate_program(
        VALID_RUN_LENGTH_PROGRAM,
        cases=program_sandbox.build_run_length_program_cases(random_seed=7, random_count=4),
        timeout_seconds=1.0,
    )
    assert result.valid


def test_program_sandbox_rejects_forbidden_import_eval_and_timeout():
    forbidden_import = program_sandbox.validate_program(
        "import os\n\ndef compose(components, metadata):\n    return {\"accept\": False, \"reason\": \"x\"}\n"
    )
    forbidden_eval = program_sandbox.validate_program(
        "def compose(components, metadata):\n    return eval(\"1\")\n"
    )
    timeout = program_sandbox.validate_program(
        "def compose(components, metadata):\n    while True:\n        pass\n",
        cases=[program_sandbox.SandboxCase(name="one", components=[], expected_accept=False)],
        timeout_seconds=0.1,
    )

    assert forbidden_import.category in {"schema_error", "forbidden_import"}
    assert forbidden_eval.category in {"forbidden_name", "forbidden_call"}
    assert timeout.category == "timeout"


def test_program_repair_is_attempted_once_and_revalidated():
    calls = []

    def repair_callback(category: str, message: str, previous_program: str) -> str:
        calls.append((category, message, previous_program))
        return VALID_RUN_LENGTH_PROGRAM

    result = program_sandbox.validate_program_with_repair(
        "def compose(components, metadata):\n    return 1\n",
        repair_callback=repair_callback,
        cases=program_sandbox.build_run_length_program_cases(random_seed=0, random_count=1),
        timeout_seconds=1.0,
    )

    assert result.valid
    assert result.repaired
    assert result.repair_attempted
    assert result.original_category == "output_format"
    assert len(calls) == 1
