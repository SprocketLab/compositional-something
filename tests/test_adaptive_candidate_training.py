from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

from core.addition_pipeline import AdditionExample, compose_examples, example_key, has_component_boundary_carry
from self.adaptive import attempts as adaptive_attempts
from self.adaptive import candidate as candidate_dispatch
from self.adaptive import candidate as candidate_training
from self.adaptive import proposal as adaptive_proposal
from self.adaptive.proposal import validate_config_rows
from self.adaptive.proposal import (
    build_proposal_grpo_traces,
    proposal_grpo_advantages,
    proposal_grpo_reward,
)
from self.adaptive.proposal import validate_proposal_rows
from self.adaptive import driver as adaptive_driver
from self.adaptive import traces as adaptive_traces
from self.core import composition
from self.core.data_io import save_examples
from self.core.models import (
    CandidateMetrics,
    CandidateWorkItem,
    ExactPairDataset,
    candidate_metrics_from_json,
)
from self.core.task_protocols import task_for_name
from self.adaptive.proposal import ConfigProposal, PromptBundle
from self.tasks import (
    RUN_LENGTH_TARGET_RUN_STATE,
    RunLengthExample,
    format_run_length_run_state,
    run_length_key,
)


def _make_loop_facade() -> SimpleNamespace:
    """Small test binding object for adaptive driver wiring.

    These tests used to import the adaptive driver as a giant module-level facade.
    The production code now keeps the driver slim, so the tests bind only the
    names they exercise while still routing worker/dispatch calls through the
    same dependency wiring used by the CLI.
    """

    loop = SimpleNamespace()
    driver_bindings = adaptive_driver._default_bindings()
    driver_wiring = adaptive_driver._driver_wiring()
    for name in adaptive_driver.DEFAULT_BINDING_NAMES:
        setattr(loop, name, getattr(driver_bindings, name))

    loop.build_exact_pair_addition_dataset = composition.build_exact_pair_addition_dataset
    loop.build_exact_pair_run_length_dataset = composition.build_exact_pair_run_length_dataset
    loop.merge_run_length_examples = composition.merge_run_length_examples
    loop.compose_run_length_pseudo_examples = composition.compose_run_length_pseudo_examples
    loop.compose_examples = compose_examples
    loop.validate_config_rows = validate_config_rows
    loop.proposal_grpo_reward = proposal_grpo_reward
    loop.proposal_grpo_advantages = proposal_grpo_advantages
    loop.build_proposal_grpo_traces = build_proposal_grpo_traces
    loop.static_frontier_sizes = candidate_training.static_frontier_sizes
    loop.mean_accuracy_for_sizes = candidate_training.mean_accuracy_for_sizes
    loop.cleanup_replaced_model_checkpoint = candidate_training.cleanup_replaced_model_checkpoint
    loop.select_candidate = candidate_training.select_candidate
    loop.task_for_name = task_for_name
    loop.CandidateWorkItem = CandidateWorkItem
    loop.ExactPairDataset = ExactPairDataset
    loop.CandidateMetrics = CandidateMetrics
    loop.candidate_metrics_from_json = candidate_metrics_from_json
    loop.ProposalTraceExample = adaptive_traces.ProposalTraceExample
    loop.OutcomeTraceExample = adaptive_traces.OutcomeTraceExample
    loop.sample_proposal_trace_replay = adaptive_traces.sample_proposal_trace_replay
    loop.sample_outcome_trace_replay = adaptive_traces.sample_outcome_trace_replay
    loop.build_selected_proposal_trace_example = adaptive_traces.build_selected_proposal_trace_example
    loop.build_candidate_proposal_trace_example = adaptive_traces.build_candidate_proposal_trace_example
    loop.build_outcome_trace_example = adaptive_traces.build_outcome_trace_example
    loop.validate_proposal_rows = validate_proposal_rows
    loop.save_examples = save_examples

    loop._candidate_failure_metrics = candidate_dispatch.candidate_failure_metrics
    loop._collect_candidate_worker_metrics = candidate_dispatch.collect_candidate_worker_metrics
    loop.run = lambda args: driver_wiring.run(loop, args)
    loop.apply_or_dispatch_proposal_grpo_update = (
        lambda **kwargs: driver_wiring.apply_or_dispatch_proposal_grpo_update(loop, **kwargs)
    )
    loop.train_candidates_serial = lambda **kwargs: driver_wiring.train_candidates_serial(loop, **kwargs)
    loop.train_candidates_local_parallel = lambda **kwargs: driver_wiring.train_candidates_local_parallel(loop, **kwargs)
    loop.train_candidate_metrics = lambda **kwargs: driver_wiring.train_candidate_metrics(loop, **kwargs)
    loop.run_candidate_worker_from_spec = lambda spec_path, **kwargs: driver_wiring.run_candidate_worker_from_spec(
        loop, spec_path, **kwargs
    )
    loop.run_candidate_worker_pack_from_spec = lambda pack_spec_path: driver_wiring.run_candidate_worker_pack_from_spec(
        loop, pack_spec_path
    )
    return loop


loop = _make_loop_facade()


def test_exact_pair_addition_guard_rejects_boundary_carry():
    rng = random.Random(0)
    source = [
        AdditionExample(a=12, b=13, result=25, digits=2, has_carry=False, operand_width=2),
        AdditionExample(a=34, b=11, result=45, digits=2, has_carry=False, operand_width=2),
        AdditionExample(a=123, b=111, result=234, digits=3, has_carry=False, operand_width=3),
        AdditionExample(a=234, b=222, result=456, digits=3, has_carry=False, operand_width=3),
    ]
    proposal = ConfigProposal(left=2, right=3, guard="reject_boundary_carry", target=5)

    dataset = loop.build_exact_pair_addition_dataset(
        source_examples=source,
        proposal=proposal,
        per_size_count=8,
        rng=rng,
    )

    assert len(dataset.examples) == 8
    assert dataset.diagnostics["rejected_by_guard"] == 0
    for example in dataset.examples:
        component_keys = dataset.component_map[example_key(example)]
        component_digits = [key[0] for key in component_keys]
        assert component_digits == [2, 3]
        assert not has_component_boundary_carry(example, component_digits)


def test_exact_pair_run_length_guard_rejects_boundary_continue():
    rng = random.Random(1)
    source = [
        RunLengthExample("01", 2, 1, 1, 1, target_mode=RUN_LENGTH_TARGET_RUN_STATE),
        RunLengthExample("10", 2, 1, 1, 1, target_mode=RUN_LENGTH_TARGET_RUN_STATE),
        RunLengthExample("010", 3, 1, 1, 1, target_mode=RUN_LENGTH_TARGET_RUN_STATE),
        RunLengthExample("101", 3, 1, 1, 1, target_mode=RUN_LENGTH_TARGET_RUN_STATE),
    ]
    proposal = ConfigProposal(left=2, right=3, guard="reject_boundary_continue", target=5)

    dataset = loop.build_exact_pair_run_length_dataset(
        source_examples=source,
        proposal=proposal,
        per_size_count=8,
        rng=rng,
    )

    assert len(dataset.examples) == 8
    for example in dataset.examples:
        left_key, right_key = dataset.component_map[run_length_key(example)]
        assert left_key[1][-1] != right_key[1][0]


def test_run_length_run_state_pseudo_composes_component_predictions():
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    left = RunLengthExample("11", 2, 2, 2, 2, target_mode=RUN_LENGTH_TARGET_RUN_STATE)
    right = RunLengthExample("100", 3, 2, 1, 2, target_mode=RUN_LENGTH_TARGET_RUN_STATE)
    composed = loop.merge_run_length_examples(left, right)
    component_map = {run_length_key(composed): [run_length_key(left), run_length_key(right)]}
    predictions = {
        run_length_key(left): format_run_length_run_state((2, "1", 2, "1", 2)),
        run_length_key(right): format_run_length_run_state((2, "1", 1, "0", 2)),
    }

    pseudo, diagnostics = loop.compose_run_length_pseudo_examples(
        proposal=proposal,
        composed_examples=[composed],
        component_map=component_map,
        component_predictions=predictions,
        target_mode=RUN_LENGTH_TARGET_RUN_STATE,
    )

    assert diagnostics["retained_total"] == 1
    assert pseudo[0].target_override == "3|1|3|0|2"


def test_config_validation_rejects_source_size_not_in_pool():
    args = argparse.Namespace(
        task="run_length",
        initial_min_size=8,
        initial_max_size=16,
        allow_repeat_targets=False,
    )
    rows = [
        {
            "id": "unavailable-gap",
            "raw_output": {"left": 12, "right": 17, "guard": "none"},
        },
        {
            "id": "available",
            "raw_output": {"left": 8, "right": 17, "guard": "none"},
        },
    ]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={8, 9, 17},
        frontier_min=18,
        frontier_max=40,
    )

    assert not results[0]["valid"]
    assert "source pool" in results[0]["validation_message"]
    assert results[1]["valid"]


def test_config_validation_allows_repeat_target_in_source():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
    )
    rows = [{"id": "repeat", "raw_output": {"left": 2, "right": 3, "guard": "none"}}]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={2, 3, 5},
        frontier_min=4,
        frontier_max=10,
    )

    assert results[0]["valid"]
    assert results[0]["repeat_target"] is True
    assert results[0]["parsed_proposal"]["target"] == 5


def test_config_validation_allows_same_batch_duplicate_completion():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
    )
    rows = [
        {"id": "first", "raw_output": {"left": 2, "right": 4, "guard": "none"}},
        {"id": "second", "raw_output": {"left": 2, "right": 4, "guard": "none"}},
    ]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={2, 3, 4},
        frontier_min=5,
        frontier_max=10,
    )

    assert results[0]["valid"]
    assert results[0]["duplicate"] is False
    assert results[1]["valid"]
    assert results[1]["duplicate"] is True
    assert loop.proposal_grpo_reward(results[1]) == 1.0


def test_config_validation_rejects_failed_action_cooldown_with_canonical_key():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
    )
    rows = [{"id": "swapped-repeat", "raw_output": {"left": 3, "right": 2, "guard": "none"}}]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={2, 3},
        frontier_min=4,
        frontier_max=10,
        failed_action_cooldown=[("proposal", 2, 3, "none", 5)],
    )

    assert not results[0]["valid"]
    assert results[0]["validation_category"] == "failed_action_cooldown"
    assert results[0]["parsed_proposal"] == {
        "left": 3,
        "right": 2,
        "guard": "none",
        "target": 5,
        "notes": "",
    }


def test_action_observation_config_validation_normalizes_flat_json_completion():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
    )
    reasoning = "We are solid at sizes 2 and 3, and target 5 is the next useful frontier."
    rows = [
        {
            "id": "agent",
            "raw_output": json.dumps(
                {
                    "reasoning": reasoning,
                    "expected_avg_delta_from_current": 0.03,
                    "expected_target_delta": 0.11,
                    "expected_frontier_delta": 0.02,
                    "left": 2,
                    "right": 3,
                    "guard": "none",
                }
            ),
        }
    ]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={2, 3},
        frontier_min=4,
        frontier_max=10,
    )

    assert results[0]["valid"]
    assert results[0]["parsed_proposal"] == {
        "left": 2,
        "right": 3,
        "guard": "none",
        "target": 5,
        "notes": reasoning,
    }
    assert results[0]["parsed_prediction"] == {
        "expected_avg_delta_from_current": 0.03,
        "expected_target_delta": 0.11,
        "expected_frontier_delta": 0.02,
    }
    assert results[0]["proposal_output_schema"] == "action_observation"
    assert results[0]["completion"] == (
        json.dumps(
            {
                "reasoning": reasoning,
                "expected_avg_delta_from_current": 0.03,
                "expected_target_delta": 0.11,
                "expected_frontier_delta": 0.02,
                "left": 2,
                "right": 3,
                "guard": "none",
            },
            separators=(",", ":"),
        )
    )


def test_action_observation_rejects_non_json_output():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
    )
    rows = [
        {
            "id": "agent",
            "raw_output": (
                "We are solid at sizes 3-7 and target 8 is the next frontier.\n"
            ),
        }
    ]

    results = loop.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={3, 4, 5, 6, 7},
        frontier_min=8,
        frontier_max=10,
    )

    assert not results[0]["valid"]
    assert results[0]["validation_category"] == "parse_error"
    assert "JSON object" in results[0]["validation_message"]


def test_action_observation_uses_first_complete_json_object():
    args = argparse.Namespace(
        task="addition",
        initial_min_size=2,
        initial_max_size=5,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
    )
    raw = (
        '```json\n{"reasoning":"first valid object","left":2,"right":3,"guard":"none"}\n```\n'
        '```json\n{"reasoning":"second object","left":3,"right":3,"guard":"none"}\n```\n'
        '```json\n{"reasoning":"truncated","left":'
    )

    results = loop.validate_config_rows(
        rows=[{"id": "agent", "raw_output": raw}],
        args=args,
        source_sizes={2, 3},
        frontier_min=4,
        frontier_max=10,
    )

    assert results[0]["valid"]
    assert results[0]["parsed_proposal"] == {
        "left": 2,
        "right": 3,
        "guard": "none",
        "target": 5,
        "notes": "first valid object",
    }


def test_static_frontier_accuracy_counts_missing_sizes_as_zero():
    args = argparse.Namespace(frontier_min_size=4, frontier_max_size=6)

    assert loop.static_frontier_sizes(args) == [4, 5, 6]
    assert loop.mean_accuracy_for_sizes({4: 1.0, 6: 0.5}, loop.static_frontier_sizes(args)) == 0.5


def test_candidate_work_item_logs_infeasible_guard(tmp_path):
    args = argparse.Namespace(
        task="run_length",
        candidate_train_per_size=2,
    )
    task = loop.task_for_name("run_length")
    source = [
        RunLengthExample("01", 2, 1, 1, 1, target_mode=RUN_LENGTH_TARGET_RUN_STATE),
    ]
    proposal = ConfigProposal(left=2, right=2, guard="require_boundary_continue", target=4)
    results = [
        {
            "proposal_index": 0,
            "id": "bad-guard",
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        }
    ]

    work_items = loop.build_candidate_work_items(
        args=args,
        task=task,
        round_dir=tmp_path,
        proposal_results=results,
        source_examples=source,
        exclude_keys=set(),
        rng=random.Random(0),
    )

    assert work_items == []
    failure_path = tmp_path / "candidates" / "candidate_00" / "data_build_failure.json"
    assert failure_path.exists()


def test_candidate_work_items_deduplicate_equivalent_config_actions(tmp_path):
    args = argparse.Namespace(
        task="addition",
        candidate_train_per_size=2,
    )
    task = loop.task_for_name("addition")
    source = [
        AdditionExample(a=12, b=13, result=25, digits=2, has_carry=False, operand_width=2),
        AdditionExample(a=34, b=11, result=45, digits=2, has_carry=False, operand_width=2),
        AdditionExample(a=123, b=111, result=234, digits=3, has_carry=False, operand_width=3),
        AdditionExample(a=234, b=222, result=456, digits=3, has_carry=False, operand_width=3),
    ]
    first = ConfigProposal(left=2, right=3, guard="none", target=5, notes="first")
    duplicate = ConfigProposal(left=2, right=3, guard="none", target=5, notes="different reasoning")
    reversed_order = ConfigProposal(left=3, right=2, guard="none", target=5, notes="ordered action differs")
    results = [
        {
            "proposal_index": 0,
            "id": "first",
            "valid": True,
            "parsed_proposal": first.to_json_dict(),
            "completion": first.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "duplicate",
            "valid": True,
            "parsed_proposal": duplicate.to_json_dict(),
            "completion": duplicate.to_completion(),
        },
        {
            "proposal_index": 2,
            "id": "reversed",
            "valid": True,
            "parsed_proposal": reversed_order.to_json_dict(),
            "completion": reversed_order.to_completion(),
        },
    ]

    work_items = loop.build_candidate_work_items(
        args=args,
        task=task,
        round_dir=tmp_path,
        proposal_results=results,
        source_examples=source,
        exclude_keys=set(),
        rng=random.Random(0),
    )

    assert [item.index for item in work_items] == [0, 2]
    assert results[0]["candidate_dedup_skipped"] is False
    assert results[1]["candidate_dedup_skipped"] is True
    assert results[1]["candidate_dedup_kept_proposal_index"] == 0
    assert results[2]["candidate_dedup_skipped"] is False

    dedup_summary = json.loads((tmp_path / "candidate_action_dedup.json").read_text())
    assert dedup_summary["unique_action_count"] == 2
    assert dedup_summary["skipped_duplicate_count"] == 1
    assert dedup_summary["skipped_duplicates"][0]["proposal_index"] == 1

    annotated_results = json.loads((tmp_path / "proposal_results.json").read_text())
    assert annotated_results[1]["candidate_dedup_skipped"] is True

def test_dry_run_attempts_continue_until_valid_candidate(tmp_path):
    fixture = tmp_path / "fixtures.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps(
                        {
                            "attempt": 1,
                            "id": "invalid",
                            "raw_output": {"left": 2, "right": 4, "guard": "none"},
                        }
                ),
                json.dumps(
                    {
                        "attempt": 2,
                        "id": "valid",
                        "raw_output": {"left": 2, "right": 3, "guard": "none"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parser = loop.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "addition",
            "--condition",
            "config",
            "--proposal-output-schema",
            "plain",
            "--proposal-fixture-jsonl",
            str(fixture),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-selected-rounds",
            "1",
            "--max-attempt-rounds",
            "2",
            "--num-candidates",
            "1",
            "--initial-min-size",
            "2",
            "--initial-max-size",
            "3",
            "--frontier-max-size",
            "5",
            "--initial-train-per-size",
            "2",
            "--initial-eval-per-size",
            "0",
            "--candidate-train-per-size",
            "1",
            "--eval-per-size",
            "0",
            "--composed-eval-per-size",
            "0",
            "--plan-log-path",
            str(tmp_path / "plan.md"),
            "--dry-run-data-only",
        ]
    )

    summary = loop.run(args)

    assert summary["attempts_completed"] == 2
    assert (tmp_path / "run" / "attempt_0001" / "dry_run_summary.json").exists()
    assert (tmp_path / "run" / "attempt_0002" / "dry_run_summary.json").exists()
    first = json.loads((tmp_path / "run" / "attempt_0001" / "dry_run_summary.json").read_text())
    second = json.loads((tmp_path / "run" / "attempt_0002" / "dry_run_summary.json").read_text())
    assert first["work_items"] == 0
    assert second["work_items"] == 1


def test_dry_run_attempt_loop_defaults_to_attempt_budget_without_selected_cap(tmp_path):
    fixture = tmp_path / "fixtures.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps({"attempt": 1, "id": "valid-1", "raw_output": {"left": 2, "right": 3, "guard": "none"}}),
                json.dumps({"attempt": 2, "id": "valid-2", "raw_output": {"left": 2, "right": 3, "guard": "none"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parser = loop.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "addition",
            "--condition",
            "config",
            "--proposal-output-schema",
            "plain",
            "--proposal-fixture-jsonl",
            str(fixture),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-attempt-rounds",
            "2",
            "--num-candidates",
            "1",
            "--initial-min-size",
            "2",
            "--initial-max-size",
            "3",
            "--frontier-max-size",
            "5",
            "--initial-train-per-size",
            "2",
            "--initial-eval-per-size",
            "0",
            "--candidate-train-per-size",
            "1",
            "--eval-per-size",
            "0",
            "--composed-eval-per-size",
            "0",
            "--plan-log-path",
            str(tmp_path / "plan.md"),
            "--dry-run-data-only",
        ]
    )

    summary = loop.run(args)

    assert summary["attempts_completed"] == 2
    assert summary["selected_rounds_completed"] == 2
    assert summary["max_selected_rounds"] == 0


def test_dry_run_attempt_loop_honors_explicit_selected_cap(tmp_path):
    fixture = tmp_path / "fixtures.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps({"attempt": 1, "id": "valid-1", "raw_output": {"left": 2, "right": 3, "guard": "none"}}),
                json.dumps({"attempt": 2, "id": "valid-2", "raw_output": {"left": 2, "right": 3, "guard": "none"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parser = loop.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "addition",
            "--condition",
            "config",
            "--proposal-output-schema",
            "plain",
            "--proposal-fixture-jsonl",
            str(fixture),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-attempt-rounds",
            "5",
            "--max-selected-rounds",
            "1",
            "--num-candidates",
            "1",
            "--initial-min-size",
            "2",
            "--initial-max-size",
            "3",
            "--frontier-max-size",
            "5",
            "--initial-train-per-size",
            "2",
            "--initial-eval-per-size",
            "0",
            "--candidate-train-per-size",
            "1",
            "--eval-per-size",
            "0",
            "--composed-eval-per-size",
            "0",
            "--plan-log-path",
            str(tmp_path / "plan.md"),
            "--dry-run-data-only",
        ]
    )

    summary = loop.run(args)

    assert summary["attempts_completed"] == 1
    assert summary["selected_rounds_completed"] == 1
    assert summary["max_selected_rounds"] == 1
    assert not (tmp_path / "run" / "attempt_0002").exists()


def test_selected_proposal_trace_replay_examples_preserve_completion():
    prompt = PromptBundle(system="System prompt", user="User prompt")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    item = loop.CandidateWorkItem(
        index=0,
        row_id="candidate-0",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output={"left": 2, "right": 3, "guard": "none"},
        composed=loop.ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=[],
        pseudo_diagnostics={},
    )
    metric = loop.CandidateMetrics(
        index=0,
        row_id="candidate-0",
        proposal=proposal,
        valid=True,
        reward=0.5,
        frontier_delta=0.4,
        target_accuracy=0.7,
        current_target_accuracy=0.3,
        final_accuracy=0.6,
        init_final_accuracy=0.2,
        final_accuracy_delta=0.4,
        per_size_accuracy={5: 0.7},
        pseudo_count=10,
        model_dir=None,
    )

    trace = loop.build_selected_proposal_trace_example(
        task_name="addition",
        condition="config",
        round_index=1,
        prompt=prompt,
        selected_item=item,
        selected=metric,
    )

    assert trace.prompt() == prompt.text()
    assert trace.target() == '{"guard":"none","left":2,"right":3}'
    assert trace.target_prefix() == ""
    assert trace.to_json_dict()["metadata"]["target"] == 5


def test_candidate_proposal_trace_marks_candidate_local_trace():
    prompt = PromptBundle(system="System prompt", user="User prompt")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    item = loop.CandidateWorkItem(
        index=1,
        row_id="candidate-1",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output={"left": 2, "right": 3, "guard": "none"},
        composed=loop.ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=[],
        pseudo_diagnostics={},
    )

    trace = loop.build_candidate_proposal_trace_example(
        task_name="addition",
        condition="config",
        round_index=2,
        prompt=prompt,
        item=item,
    )
    payload = trace.to_json_dict()

    assert trace.prompt() == prompt.text()
    assert trace.target() == proposal.to_completion()
    assert trace.target_prefix() == ""
    assert payload["reward"] is None
    assert payload["metadata"]["candidate_local_trace"] is True


def test_proposal_trace_replay_sampling_uses_ratio_and_cap():
    traces = [
        loop.ProposalTraceExample(
            prompt_text=f"prompt {idx}",
            completion=f"completion {idx}",
            task="addition",
            condition="config",
            round_index=idx,
            reward=1.0,
            metadata={},
        )
        for idx in range(2)
    ]
    args = argparse.Namespace(
        proposal_trace_replay_ratio=0.5,
        proposal_trace_replay_max_examples=3,
    )

    replay = loop.sample_proposal_trace_replay(
        args=args,
        trace_buffer=traces,
        task_train_count=10,
        rng=random.Random(0),
    )

    assert len(replay) == 3
    assert {example.completion for example in replay} <= {"completion 0", "completion 1"}


def test_proposal_trace_replay_sampling_can_be_disabled():
    trace = loop.ProposalTraceExample(
        prompt_text="prompt",
        completion="completion",
        task="addition",
        condition="config",
        round_index=1,
        reward=1.0,
        metadata={},
    )
    args = argparse.Namespace(
        proposal_trace_replay_ratio=0.0,
        proposal_trace_replay_max_examples=3,
    )

    replay = loop.sample_proposal_trace_replay(
        args=args,
        trace_buffer=[trace],
        task_train_count=10,
        rng=random.Random(0),
    )

    assert replay == []


def test_numeric_textual_outcome_trace_uses_compact_state_and_target():
    args = argparse.Namespace(
        outcome_trace_target_mode="numeric_textual",
        invalid_outcome_reward=-0.1,
    )
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    metric = loop.CandidateMetrics(
        index=0,
        row_id="candidate-0",
        proposal=proposal,
        valid=True,
        reward=0.123456,
        frontier_delta=0.222222,
        target_accuracy=0.5,
        current_target_accuracy=0.388889,
        target_delta=0.111111,
        frontier_accuracy=0.6,
        current_frontier_accuracy=0.377778,
        final_accuracy=0.42,
        init_final_accuracy=0.3,
        final_accuracy_delta=0.12,
        per_size_accuracy={5: 0.5},
        pseudo_count=8,
        model_dir=None,
        current_final_accuracy=0.4,
        final_accuracy_delta_from_current=0.02,
        proposal_prediction={
            "expected_frontier_delta": 0.2,
            "expected_final_delta_from_init": 0.1,
            "expected_avg_delta_from_current": 0.015,
            "expected_target_delta": 0.1,
        },
    )
    result = {
        "proposal_index": 0,
        "id": "candidate-0",
        "valid": True,
        "parsed_proposal": proposal.to_json_dict(),
    }

    trace = loop.build_outcome_trace_example(
        args=args,
        task_name="addition",
        condition="config",
        round_index=1,
        result=result,
        metric=metric,
        selected=True,
        source_sizes=[2, 3],
        frontier_min=4,
        frontier_max=10,
        current_final_accuracy=0.4,
        init_final_accuracy=0.3,
        current_per_size_accuracy={2: 0.98765, 3: 0.87654, 5: 0.388889},
    )

    target = json.loads(trace.target())
    assert "TASK: predict_config_outcome" in trace.prompt()
    assert '"acc":{"2":0.9877,"3":0.8765,"5":0.3889}' in trace.prompt()
    assert target["valid"] is True
    assert target["trained"] is True
    assert target["selected"] is True
    assert target["repeat_target"] is False
    assert target["reward"] == 0.1235
    assert target["target_delta"] == 0.1111
    assert target["frontier_delta"] == 0.2222
    assert target["frontier_delta_error"] == 0.0222
    assert target["target_delta_error"] == 0.0111
    assert target["avg_delta_from_current_error"] == 0.005
    assert target["avg_delta_from_init_error"] == 0.02
    assert target["avg_delta_current"] == 0.02
    assert "feedback" in target


def test_invalid_outcome_trace_uses_penalty_and_failure_code():
    args = argparse.Namespace(
        outcome_trace_target_mode="numeric",
        invalid_outcome_reward=-0.1,
    )
    result = {
        "proposal_index": 1,
        "id": "bad",
        "raw_output": {"left": 2, "right": 9, "guard": "none"},
        "valid": False,
        "validation_category": "range_error",
        "validation_message": "right source slice is not in the current source pool",
    }

    trace = loop.build_outcome_trace_example(
        args=args,
        task_name="addition",
        condition="config",
        round_index=1,
        result=result,
        metric=None,
        selected=False,
        source_sizes=[2, 3],
        frontier_min=4,
        frontier_max=10,
        current_final_accuracy=0.4,
        init_final_accuracy=0.3,
        current_per_size_accuracy={2: 1.0, 3: 0.9},
    )

    target = json.loads(trace.target())
    assert target == {
        "failure": "source_not_available",
        "avg_delta_init": None,
        "avg_delta_current": None,
        "repeat_target": False,
        "reward": -0.1,
        "selected": False,
        "target": 11,
        "target_delta": None,
        "frontier_delta": None,
        "trained": False,
        "valid": False,
    }


def test_outcome_trace_replay_sampling_uses_ratio_cap_and_mode():
    traces = [
        loop.OutcomeTraceExample(
            prompt_text=f"prompt {idx}",
            completion=f"completion {idx}",
            task="addition",
            condition="config",
            round_index=idx,
            mode="numeric",
            reward=0.0,
            metadata={"target": idx},
        )
        for idx in range(2)
    ]
    args = argparse.Namespace(
        outcome_trace_target_mode="numeric",
        outcome_trace_replay_ratio=0.5,
        outcome_trace_replay_max_examples=3,
    )

    replay = loop.sample_outcome_trace_replay(
        args=args,
        trace_buffer=traces,
        task_train_count=10,
        rng=random.Random(0),
    )

    assert len(replay) == 3
    args.outcome_trace_target_mode = "none"
    assert (
        loop.sample_outcome_trace_replay(
            args=args,
            trace_buffer=traces,
            task_train_count=10,
            rng=random.Random(0),
        )
        == []
    )


def test_parser_defaults_enable_numeric_outcome_and_config_grpo():
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
            ]
        )
    )

    assert args.outcome_trace_target_mode == "numeric"
    assert args.proposal_output_schema == "action_observation"
    assert args.num_candidates == 8
    assert args.proposal_temperature == 0.9
    assert args.proposal_top_p == 0.95
    assert args.proposal_sampling_batch_size == 8
    assert args.force_unique_proposals is False
    assert args.proposal_unique_max_draws == 0
    assert args.max_attempt_rounds == 100
    assert args.max_selected_rounds == 0
    assert args.proposal_grpo_steps == 1
    assert args.proposal_grpo_span == "reasoning_action"
    assert args.proposal_observation_loss_weight == 0.2
    assert args.proposal_format_loss_weight == 0.02
    assert args.proposal_format_replay_max_examples == 256
    assert args.proposal_format_mask_config_values is True
    assert args.proposal_grpo_deduplicate_actions is True
    assert args.proposal_update_microbatch_size == 8
    assert args.proposal_update_accumulation_steps == 1
    assert args.proposal_grpo_learning_rate == 1e-6
    assert args.proposal_grpo_zero_variance == "skip"
    assert args.proposal_grpo_reward_mode == "outcome"
    assert args.proposal_grpo_outcome_scale == 0.05
    assert args.proposal_grpo_novelty_bonus_beta == 0.05
    assert args.source_admission_target_accuracy_threshold == 0.80
    assert args.keep_final_model_checkpoint is False
    assert args.keep_all_proposal_grpo_checkpoints is False
    assert args.candidate_execution_mode == "local_parallel"
    assert args.candidate_local_parallelism == 2
    assert args.candidate_local_pack_size == 1


def test_config_prompt_omits_action_history():
    args = argparse.Namespace(
        task="addition",
        condition="config",
        frontier_min_size=4,
        frontier_max_size=10,
        max_selected_rounds=0,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
        proposal_prompt_action_history=False,
        proposal_prompt_action_history_max_items=5,
    )

    prompt = adaptive_attempts.build_attempt_prompt(
        args=args,
        current_checkpoint="checkpoint",
        current_final_accuracy=0.4,
        init_final_accuracy=0.3,
        current_per_size_accuracy={4: 0.2, 5: 0.3},
        source_sizes={2, 3, 4},
        selected_round_for_prompt=3,
        attempt_index=3,
        selected_rounds=2,
        consecutive_no_selection=0,
        extra_aggregate_metrics={},
    ).prompt

    assert "Recent evaluated actions:" not in prompt.user
    assert '"recent_actions"' not in prompt.user
    assert "target_selected_rounds" not in prompt.user
    assert "Current round" not in prompt.user
    assert "round_index" not in prompt.user
    assert "- model:" not in prompt.user
    assert "already_in_source" not in prompt.user
    assert "current_final_accuracy" not in prompt.user
    assert "init_final_accuracy" not in prompt.user
    assert "current_avg_accuracy" in prompt.user
    assert "init_avg_accuracy" in prompt.user
    assert "candidate_avg_accuracy - current_avg_accuracy" in prompt.user
    assert "frontier_delta + lambda_final" not in prompt.user
    assert "Decision rules:" in prompt.user
    assert "Identify reliable source sizes" in prompt.user
    assert "Identify weak reachable targets" in prompt.user
    assert "avoid exact repeats" in prompt.user
    assert "reject_boundary_carry" in prompt.user
    assert "next_selected_trace_index_if_chosen" in prompt.user
    assert '"target": 5' not in prompt.user
    assert "Source sizes 5 and 3" not in prompt.user
    assert '"left": 5' not in prompt.user
    assert '"right": 3' not in prompt.user
    assert "target 8" not in prompt.user


def test_config_prompt_can_include_compact_selected_action_history():
    args = argparse.Namespace(
        task="addition",
        condition="config",
        frontier_min_size=4,
        frontier_max_size=10,
        max_selected_rounds=0,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
        proposal_prompt_action_history=True,
        proposal_prompt_action_history_max_items=1,
        source_admission_target_accuracy_threshold=0.8,
    )
    selected_trace = adaptive_traces.ProposalTraceExample(
        prompt_text="old prompt",
        completion='{"reasoning":"old","left":5,"right":3,"guard":"reject_boundary_carry"}',
        task="addition",
        condition="config",
        round_index=2,
        reward=0.07,
        metadata={
            "left": 5,
            "right": 3,
            "target": 8,
            "guard": "reject_boundary_carry",
            "target_accuracy": 0.82,
            "target_delta": 0.12,
            "final_accuracy_delta_from_current": 0.04,
        },
    )

    prompt = adaptive_attempts.build_attempt_prompt(
        args=args,
        current_checkpoint="checkpoint",
        current_final_accuracy=0.4,
        init_final_accuracy=0.3,
        current_per_size_accuracy={4: 0.2, 5: 0.3},
        source_sizes={2, 3, 4, 5},
        selected_round_for_prompt=3,
        attempt_index=3,
        selected_rounds=2,
        consecutive_no_selection=0,
        proposal_trace_buffer=[selected_trace],
        extra_aggregate_metrics={},
    ).prompt

    assert '"recent_selected_actions"' in prompt.user
    assert '"selected_round": 2' in prompt.user
    assert '"left": 5' in prompt.user
    assert '"right": 3' in prompt.user
    assert '"target": 8' in prompt.user
    assert '"guard": "reject_boundary_carry"' in prompt.user
    assert '"source_admission_target_accuracy_threshold": 0.8' in prompt.user
    assert "Output exactly one JSON object like this" not in prompt.user
    assert "Choose numeric left and right values only from current_source_slices" in prompt.user
    assert "Do not copy numeric values from these instructions" not in prompt.user
    assert "<action>" not in prompt.user
    assert "Output only JSON." in prompt.user


def test_candidate_metrics_json_roundtrip(tmp_path):
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5, notes="test")
    metrics = loop.CandidateMetrics(
        index=1,
        row_id="row-1",
        proposal=proposal,
        valid=True,
        reward=0.25,
        frontier_delta=0.2,
        frontier_accuracy=0.55,
        current_frontier_accuracy=0.35,
        target_accuracy=0.8,
        current_target_accuracy=0.6,
        target_delta=0.2,
        final_accuracy=0.7,
        current_final_accuracy=0.65,
        init_final_accuracy=0.5,
        final_accuracy_delta=0.2,
        final_accuracy_delta_from_current=0.05,
        per_size_accuracy={5: 0.8, 6: 0.4},
        pseudo_count=10,
        model_dir=tmp_path / "model",
        proposal_trace_replay_count=2,
        candidate_proposal_trace_count=1,
        outcome_trace_replay_count=4,
        proposal_prediction={
            "target": 5,
            "expected_frontier_delta": 0.1,
            "expected_final_delta_from_init": 0.2,
        },
    )

    restored = loop.candidate_metrics_from_json(metrics.to_json_dict())

    assert restored.index == 1
    assert restored.row_id == "row-1"
    assert restored.proposal == proposal
    assert restored.per_size_accuracy == {5: 0.8, 6: 0.4}
    assert restored.model_dir == tmp_path / "model"
    assert restored.frontier_accuracy == 0.55
    assert restored.current_frontier_accuracy == 0.35
    assert restored.target_delta == 0.2
    assert restored.proposal_prediction["expected_frontier_delta"] == 0.1


def test_candidate_worker_spec_roundtrip_loads_inputs(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "serial",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    candidate_dir = round_dir / "candidates" / "candidate_00"
    candidate_dir.mkdir(parents=True)
    source_examples = [
        AdditionExample(a=12, b=34, result=46, digits=2, has_carry=False, operand_width=2),
    ]
    eval_examples = [
        AdditionExample(a=111, b=222, result=333, digits=3, has_carry=False, operand_width=3),
    ]
    pseudo_examples = [
        AdditionExample(a=12345, b=11111, result=23456, digits=5, has_carry=False, operand_width=5),
    ]
    loop.save_examples(candidate_dir / "pseudo_examples.jsonl", pseudo_examples, task.serialize_example)
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    work_item = loop.CandidateWorkItem(
        index=0,
        row_id="candidate-row",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output=proposal.to_json_dict(),
        composed=loop.ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=pseudo_examples,
        pseudo_diagnostics={"retained_total": 1},
    )
    prompt = PromptBundle(system="system", user="user")
    proposal_traces = [
        loop.ProposalTraceExample(
            prompt_text=prompt.text(),
            completion=proposal.to_completion(),
            task="addition",
            condition="config",
            round_index=1,
            reward=0.1,
            metadata={"proposal_index": 0},
        )
    ]
    outcome_traces = [
        loop.OutcomeTraceExample(
            prompt_text="outcome prompt",
            completion='{"reward":0.1}',
            task="addition",
            condition="config",
            round_index=1,
            mode="numeric",
            reward=0.1,
            metadata={"target": 5},
        )
    ]
    spec_paths = loop._prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=source_examples,
        proposal_trace_buffer=proposal_traces,
        outcome_trace_buffer=outcome_traces,
        proposal_prompt=prompt,
        round_index=1,
        work_items=[work_item],
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=0.4,
        current_per_size_accuracy={3: 0.4, 5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )
    seen = {}

    def fake_train_and_score_candidate(**kwargs):
        seen.update(kwargs)
        return loop.CandidateMetrics(
            index=kwargs["item"].index,
            row_id=kwargs["item"].row_id,
            proposal=kwargs["item"].proposal,
            valid=True,
            reward=0.1,
            frontier_delta=0.1,
            target_accuracy=0.1,
            current_target_accuracy=0.0,
            final_accuracy=0.5,
            current_final_accuracy=0.4,
            init_final_accuracy=0.3,
            final_accuracy_delta=0.2,
            final_accuracy_delta_from_current=0.1,
            per_size_accuracy={5: 0.1},
            pseudo_count=len(kwargs["item"].pseudo_examples),
            model_dir=tmp_path / "model",
        )

    monkeypatch.setattr(loop, "train_and_score_candidate", fake_train_and_score_candidate)

    metrics = loop.run_candidate_worker_from_spec(spec_paths[0])

    assert metrics.valid
    assert seen["source_examples"] == source_examples
    assert seen["eval_examples"] == eval_examples
    assert seen["proposal_trace_buffer"] == proposal_traces
    assert seen["outcome_trace_buffer"] == outcome_traces
    assert seen["proposal_prompt"] == prompt
    assert seen["item"].pseudo_examples == pseudo_examples
    assert seen["current_per_size_accuracy"] == {3: 0.4, 5: 0.0}


def _write_candidate_pseudo_examples(tmp_path, round_dir, task, count):
    pseudo_examples = [
        AdditionExample(a=12345, b=11111, result=23456, digits=5, has_carry=False, operand_width=5),
    ]
    work_items = []
    for index in range(count):
        candidate_dir = round_dir / "candidates" / f"candidate_{index:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        loop.save_examples(candidate_dir / "pseudo_examples.jsonl", pseudo_examples, task.serialize_example)
        proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
        work_items.append(
            loop.CandidateWorkItem(
                index=index,
                row_id=f"candidate-{index}",
                proposal=proposal,
                completion=proposal.to_completion(),
                raw_output=proposal.to_json_dict(),
                composed=loop.ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
                pseudo_examples=pseudo_examples,
                pseudo_diagnostics={"retained_total": 1},
            )
        )
    return work_items


def test_local_parallel_candidate_workers_respect_concurrency_cap(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "local_parallel",
                "--candidate-local-parallelism",
                "4",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    work_items = _write_candidate_pseudo_examples(tmp_path, round_dir, task, count=5)
    active = {"count": 0, "max": 0, "next_pid": 1000}

    class FakePopen:
        def __init__(self, command, stdout=None, stderr=None):
            self.command = command
            self.pid = active["next_pid"]
            active["next_pid"] += 1
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            self.returncode = None
            payload = loop._load_json(Path(command[-1]))
            candidate_index = int(payload["candidate_index"])
            item = work_items[candidate_index]
            metric = loop.CandidateMetrics(
                index=item.index,
                row_id=item.row_id,
                proposal=item.proposal,
                valid=True,
                reward=0.1,
                frontier_delta=0.1,
                target_accuracy=0.2,
                current_target_accuracy=0.0,
                final_accuracy=0.5,
                current_final_accuracy=0.4,
                init_final_accuracy=0.3,
                final_accuracy_delta=0.2,
                final_accuracy_delta_from_current=0.1,
                per_size_accuracy={5: 0.2},
                pseudo_count=1,
                model_dir=tmp_path / f"model_{candidate_index}",
            )
            loop.write_json(round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "candidate_metrics.json", metric.to_json_dict())

        def poll(self):
            if self.returncode is None:
                active["count"] -= 1
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(loop.subprocess, "Popen", FakePopen)

    metrics = loop.train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )

    assert len(metrics) == 5
    assert all(metric.valid for metric in metrics)
    assert active["max"] == 4
    dispatch = loop._load_json(round_dir / "candidate_jobs" / "local_dispatch.json")
    assert dispatch["candidate_count"] == 5
    assert dispatch["planned_processes"] == 5
    assert dispatch["packed_workers"] is False
    assert dispatch["cache_plan"] == {
        "shared_input_cache": False,
        "tokenizer_bootstrap_cache": False,
        "base_state_cache": False,
    }


def test_local_parallel_candidate_workers_can_pack_processes(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "local_parallel",
                "--candidate-local-parallelism",
                "2",
                "--candidate-local-pack-size",
                "2",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    work_items = _write_candidate_pseudo_examples(tmp_path, round_dir, task, count=5)
    active = {"count": 0, "max": 0, "next_pid": 2000, "processes": 0}

    class FakePackedPopen:
        def __init__(self, command, stdout=None, stderr=None):
            self.command = command
            self.pid = active["next_pid"]
            active["next_pid"] += 1
            active["count"] += 1
            active["processes"] += 1
            active["max"] = max(active["max"], active["count"])
            self.returncode = None
            assert "--run-candidate-pack-worker" in command
            pack_payload = loop._load_json(Path(command[-1]))
            assert len(pack_payload["spec_paths"]) <= 2
            for spec_path in pack_payload["spec_paths"]:
                payload = loop._load_json(Path(spec_path))
                candidate_index = int(payload["candidate_index"])
                item = work_items[candidate_index]
                metric = loop.CandidateMetrics(
                    index=item.index,
                    row_id=item.row_id,
                    proposal=item.proposal,
                    valid=True,
                    reward=0.1,
                    frontier_delta=0.1,
                    target_accuracy=0.2,
                    current_target_accuracy=0.0,
                    final_accuracy=0.5,
                    current_final_accuracy=0.4,
                    init_final_accuracy=0.3,
                    final_accuracy_delta=0.2,
                    final_accuracy_delta_from_current=0.1,
                    per_size_accuracy={5: 0.2},
                    pseudo_count=1,
                    model_dir=tmp_path / f"model_{candidate_index}",
                )
                loop.write_json(
                    round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "candidate_metrics.json",
                    metric.to_json_dict(),
                )

        def poll(self):
            if self.returncode is None:
                active["count"] -= 1
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(loop.subprocess, "Popen", FakePackedPopen)

    metrics = loop.train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )

    assert len(metrics) == 5
    assert all(metric.valid for metric in metrics)
    assert active["processes"] == 3
    assert active["max"] == 2
    dispatch = loop._load_json(round_dir / "candidate_jobs" / "local_dispatch.json")
    assert dispatch["candidate_count"] == 5
    assert dispatch["planned_processes"] == 3
    assert dispatch["pack_size"] == 2
    assert dispatch["packed_workers"] is True
    assert dispatch["cache_plan"] == {
        "shared_input_cache": True,
        "tokenizer_bootstrap_cache": True,
        "base_state_cache": False,
    }
    assert [unit["candidate_indices"] for unit in dispatch["planned_units"]] == [[0, 1], [2, 3], [4]]
    assert len(dispatch["launched"]) == 3


def test_candidate_pack_worker_reuses_shared_inputs(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "local_parallel",
                "--candidate-local-cache-base-state",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    work_items = _write_candidate_pseudo_examples(tmp_path, round_dir, task, count=2)
    spec_paths = loop._prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )
    pack_path = round_dir / "candidate_jobs" / "pack_specs" / "manual_pack.json"
    loop.write_json(pack_path, {"spec_paths": [str(path) for path in spec_paths]})

    original_load_trace_jsonl = loop.load_trace_jsonl
    trace_load_counts = {}
    scored_indices = []
    bootstrap_cache_ids = []

    def counting_load_trace_jsonl(path, parser_fn):
        trace_load_counts[str(path)] = trace_load_counts.get(str(path), 0) + 1
        return original_load_trace_jsonl(path, parser_fn)

    def fake_train_and_score_candidate(**kwargs):
        scored_indices.append(kwargs["item"].index)
        bootstrap_cache_ids.append(id(kwargs["model_bootstrap_cache"]))
        return loop.CandidateMetrics(
            index=kwargs["item"].index,
            row_id=kwargs["item"].row_id,
            proposal=kwargs["item"].proposal,
            valid=True,
            reward=0.1,
            frontier_delta=0.1,
            target_accuracy=0.2,
            current_target_accuracy=0.0,
            final_accuracy=0.5,
            current_final_accuracy=0.4,
            init_final_accuracy=0.3,
            final_accuracy_delta=0.2,
            final_accuracy_delta_from_current=0.1,
            per_size_accuracy={5: 0.2},
            pseudo_count=len(kwargs["item"].pseudo_examples),
            model_dir=tmp_path / f"model_{kwargs['item'].index}",
        )

    monkeypatch.setattr(loop, "load_trace_jsonl", counting_load_trace_jsonl)
    monkeypatch.setattr(loop, "train_and_score_candidate", fake_train_and_score_candidate)

    summary = loop.run_candidate_worker_pack_from_spec(pack_path)

    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["shared_input_cache_entries"] == 1
    assert summary["model_bootstrap_cache"] == [
        {"model_state_cache_entries": 0, "tokenizer_cache_entries": 0}
    ]
    assert summary["model_bootstrap_cache_details"] == [
        {
            "cache_base_state": 1,
            "model_state_cache_entries": 0,
            "model_state_cache_hits": 0,
            "model_state_cache_misses": 0,
            "tokenizer_cache_entries": 0,
            "tokenizer_cache_hits": 0,
            "tokenizer_cache_misses": 0,
        }
    ]
    assert scored_indices == [0, 1]
    assert len(set(bootstrap_cache_ids)) == 1
    assert sorted(trace_load_counts.values()) == [1, 1]


def test_candidate_pack_worker_passes_tokenizer_cache_without_base_state(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "local_parallel",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    work_items = _write_candidate_pseudo_examples(tmp_path, round_dir, task, count=2)
    spec_paths = loop._prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )
    pack_path = round_dir / "candidate_jobs" / "pack_specs" / "manual_pack.json"
    loop.write_json(pack_path, {"spec_paths": [str(path) for path in spec_paths]})

    bootstrap_cache_ids = []
    cache_base_state_values = []

    def fake_train_and_score_candidate(**kwargs):
        cache = kwargs["model_bootstrap_cache"]
        bootstrap_cache_ids.append(id(cache))
        cache_base_state_values.append(cache.cache_base_state)
        return loop.CandidateMetrics(
            index=kwargs["item"].index,
            row_id=kwargs["item"].row_id,
            proposal=kwargs["item"].proposal,
            valid=True,
            reward=0.1,
            frontier_delta=0.1,
            target_accuracy=0.2,
            current_target_accuracy=0.0,
            final_accuracy=0.5,
            current_final_accuracy=0.4,
            init_final_accuracy=0.3,
            final_accuracy_delta=0.2,
            final_accuracy_delta_from_current=0.1,
            per_size_accuracy={5: 0.2},
            pseudo_count=len(kwargs["item"].pseudo_examples),
            model_dir=tmp_path / f"model_{kwargs['item'].index}",
        )

    monkeypatch.setattr(loop, "train_and_score_candidate", fake_train_and_score_candidate)

    summary = loop.run_candidate_worker_pack_from_spec(pack_path)

    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["shared_input_cache_entries"] == 1
    assert summary["model_bootstrap_cache"] == [
        {"model_state_cache_entries": 0, "tokenizer_cache_entries": 0}
    ]
    assert summary["model_bootstrap_cache_details"] == [
        {
            "cache_base_state": 0,
            "model_state_cache_entries": 0,
            "model_state_cache_hits": 0,
            "model_state_cache_misses": 0,
            "tokenizer_cache_entries": 0,
            "tokenizer_cache_hits": 0,
            "tokenizer_cache_misses": 0,
        }
    ]
    assert len(set(bootstrap_cache_ids)) == 1
    assert cache_base_state_values == [False, False]


def test_local_parallel_candidate_worker_failure_becomes_metric(tmp_path, monkeypatch):
    parser = loop.build_parser()
    args = loop.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--output-dir",
                str(tmp_path / "run"),
                "--candidate-execution-mode",
                "local_parallel",
                "--candidate-local-parallelism",
                "1",
            ]
        )
    )
    task = loop.task_for_name("addition")
    round_dir = tmp_path / "run" / "attempt_0001"
    work_items = _write_candidate_pseudo_examples(tmp_path, round_dir, task, count=1)

    class FakeFailedPopen:
        pid = 1234

        def __init__(self, command, stdout=None, stderr=None):
            self.command = command
            self.returncode = None

        def poll(self):
            if self.returncode is None:
                self.returncode = 9
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(loop.subprocess, "Popen", FakeFailedPopen)

    metrics = loop.train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint="checkpoint",
        source_examples=[],
        proposal_trace_buffer=[],
        outcome_trace_buffer=[],
        proposal_prompt=PromptBundle(system="", user=""),
        round_index=1,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=[],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.0},
        init_final_accuracy=0.3,
        attempt_index=1,
    )

    assert len(metrics) == 1
    assert not metrics[0].valid
    assert "exited with code 9" in str(metrics[0].failure_reason)


def test_cleanup_replaced_model_checkpoint_prunes_only_model_dir(tmp_path):
    output_dir = tmp_path / "run"
    old_model_dir = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    new_model_dir = output_dir / "attempt_0002" / "proposal_grpo" / "model"
    old_model_dir.mkdir(parents=True)
    new_model_dir.mkdir(parents=True)
    (old_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    old_log = old_model_dir.parent / "proposal_grpo_metrics.json"
    old_log.write_text("{}", encoding="utf-8")

    deleted = loop.cleanup_replaced_model_checkpoint(
        old_checkpoint=str(old_model_dir),
        new_checkpoint=str(new_model_dir),
        output_dir=output_dir,
        keep_candidate_models=False,
        keep_proposal_grpo_checkpoints=False,
    )

    assert deleted == [str(old_model_dir)]
    assert not old_model_dir.exists()
    assert old_log.exists()
    assert new_model_dir.exists()


def test_cleanup_replaced_model_checkpoint_respects_keep_flags_and_run_boundary(tmp_path):
    output_dir = tmp_path / "run"
    inside_old = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    inside_new = output_dir / "attempt_0002" / "proposal_grpo" / "model"
    outside_old = tmp_path / "other" / "attempt_0001" / "proposal_grpo" / "model"
    for path in (inside_old, inside_new, outside_old):
        path.mkdir(parents=True)
        (path / "model.safetensors").write_text("weights", encoding="utf-8")

    assert (
        loop.cleanup_replaced_model_checkpoint(
            old_checkpoint=str(inside_old),
            new_checkpoint=str(inside_new),
            output_dir=output_dir,
            keep_candidate_models=False,
            keep_proposal_grpo_checkpoints=True,
        )
        == []
    )
    assert inside_old.exists()

    assert (
        loop.cleanup_replaced_model_checkpoint(
            old_checkpoint=str(outside_old),
            new_checkpoint=str(inside_new),
            output_dir=output_dir,
            keep_candidate_models=False,
            keep_proposal_grpo_checkpoints=False,
        )
        == []
    )
    assert outside_old.exists()


def test_cleanup_replaced_model_checkpoint_can_prune_superseded_candidate(tmp_path):
    output_dir = tmp_path / "run"
    old_model_dir = output_dir / "attempt_0001" / "candidates" / "candidate_00" / "proposal_rehearsal" / "model"
    new_model_dir = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    old_model_dir.mkdir(parents=True)
    new_model_dir.mkdir(parents=True)

    deleted = loop.cleanup_replaced_model_checkpoint(
        old_checkpoint=str(old_model_dir),
        new_checkpoint=str(new_model_dir),
        output_dir=output_dir,
        keep_candidate_models=False,
        keep_proposal_grpo_checkpoints=False,
    )

    assert deleted == [str(old_model_dir)]
    assert not old_model_dir.exists()


def test_cleanup_replaced_model_checkpoint_preserves_protected_anchor(tmp_path):
    output_dir = tmp_path / "run"
    old_model_dir = output_dir / "round_00" / "seed_training" / "model"
    new_model_dir = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    old_model_dir.mkdir(parents=True)
    new_model_dir.mkdir(parents=True)

    deleted = loop.cleanup_replaced_model_checkpoint(
        old_checkpoint=str(old_model_dir),
        new_checkpoint=str(new_model_dir),
        output_dir=output_dir,
        keep_candidate_models=False,
        keep_proposal_grpo_checkpoints=False,
        protected_checkpoints=[str(old_model_dir)],
    )

    assert deleted == []
    assert old_model_dir.exists()


def test_checkpoint_manager_prunes_unselectable_candidate_model_only(tmp_path):
    output_dir = tmp_path / "run"
    model_dir = output_dir / "attempt_0001" / "candidates" / "candidate_00" / "training" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    summary_path = model_dir.parent / "train_log.json"
    summary_path.write_text("{}", encoding="utf-8")
    metric = CandidateMetrics(
        index=0,
        row_id="candidate",
        proposal=ConfigProposal(left=2, right=2, guard="none", target=4),
        valid=True,
        reward=-0.1,
        frontier_delta=0.0,
        target_accuracy=0.0,
        current_target_accuracy=0.0,
        final_accuracy=0.0,
        init_final_accuracy=0.0,
        final_accuracy_delta=0.0,
        per_size_accuracy={},
        pseudo_count=1,
        model_dir=model_dir,
    )

    deleted = candidate_training.CheckpointManager(output_dir=output_dir).cleanup_unselectable_candidate(
        metric=metric,
        min_reward=0.0,
    )

    assert deleted == [str(model_dir)]
    assert not model_dir.exists()
    assert summary_path.exists()


def test_checkpoint_manager_prunes_final_checkpoint_by_default(tmp_path):
    output_dir = tmp_path / "run"
    model_dir = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    metrics_path = model_dir.parent / "proposal_grpo_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    deleted = candidate_training.CheckpointManager(output_dir=output_dir).cleanup_final_checkpoint(
        checkpoint=str(model_dir),
        keep_final=False,
    )

    assert deleted == [str(model_dir)]
    assert not model_dir.exists()
    assert metrics_path.exists()


def test_checkpoint_manager_preserves_protected_anchor_checkpoint(tmp_path):
    output_dir = tmp_path / "run"
    anchor_model_dir = output_dir / "round_00" / "seed_training" / "model"
    new_model_dir = output_dir / "attempt_0001" / "proposal_grpo" / "model"
    anchor_model_dir.mkdir(parents=True)
    new_model_dir.mkdir(parents=True)
    (anchor_model_dir / "model.safetensors").write_text("anchor", encoding="utf-8")
    (new_model_dir / "model.safetensors").write_text("new", encoding="utf-8")

    deleted = candidate_training.CheckpointManager(output_dir=output_dir).cleanup_replaced_checkpoint(
        old_checkpoint=str(anchor_model_dir),
        new_checkpoint=str(new_model_dir),
        protected_checkpoints=[str(anchor_model_dir)],
    )

    assert deleted == []
    assert anchor_model_dir.exists()


def test_proposal_grpo_reward_mapping_and_advantages():
    results = [
        {"valid": True},
        {"valid": False, "validation_category": "range_error"},
        {"valid": False, "validation_category": "enum_error"},
        {"valid": False, "validation_category": "schema_error"},
        {"valid": False, "validation_category": "parse_error"},
    ]

    rewards = [loop.proposal_grpo_reward(result) for result in results]
    assert rewards == [1.0, 0.6, 0.5, 0.25, 0.0]

    advantages, skipped, mode = loop.proposal_grpo_advantages(
        rewards,
        zero_variance="fixed_baseline",
        fixed_baseline=0.5,
    )
    assert not skipped
    assert mode == "normalized"
    assert abs(sum(advantages)) < 1e-6
    assert advantages[0] > advantages[-1]

    advantages, skipped, mode = loop.proposal_grpo_advantages(
        [1.0, 1.0],
        zero_variance="fixed_baseline",
        fixed_baseline=0.5,
    )
    assert not skipped
    assert mode == "fixed_baseline"
    assert advantages == [0.5, 0.5]

    advantages, skipped, mode = loop.proposal_grpo_advantages(
        [0.0, 0.0],
        zero_variance="skip",
        fixed_baseline=0.5,
    )
    assert skipped
    assert mode == "zero_variance"
    assert advantages == [0.0, 0.0]


def test_build_proposal_grpo_traces_uses_normalized_valid_completions():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="fixed_baseline",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="validity",
        proposal_grpo_outcome_scale=0.05,
    )
    prompt = PromptBundle(system="system", user="user")
    results = [
        {
            "proposal_index": 0,
            "id": "valid",
            "raw_output": 'Sure, here is the config:\n{"left":2,"right":3,"guard":"none"}',
            "valid": True,
            "parsed_proposal": {"left": 2, "right": 3, "guard": "none", "target": 5},
            "completion": '{"guard":"none","left":2,"right":3}',
        },
        {
            "proposal_index": 1,
            "id": "bad-json",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
            "validation_message": "raw output is not a JSON object",
        },
    ]

    traces, summary = loop.build_proposal_grpo_traces(args=args, prompt=prompt, proposal_results=results)

    assert summary["advantage_mode"] == "normalized"
    assert traces[0].completion == '{"guard":"none","left":2,"right":3}'
    assert traces[0].reward == 1.0
    assert traces[0].metadata["completion_source"] == "normalized"
    assert traces[1].completion == "not json"
    assert traces[1].reward == 0.0
    assert traces[1].metadata["completion_source"] == "raw"
    assert traces[0].advantage > traces[1].advantage


def test_build_proposal_grpo_traces_keeps_empty_eos_outputs_trainable():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="fixed_baseline",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="validity",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=False,
    )
    prompt = PromptBundle(system="system", user="user")
    results = [
        {
            "proposal_index": 0,
            "id": "empty-eos",
            "raw_output": "",
            "raw_output_with_special_tokens": "<|im_end|>",
            "first_generated_token_text": "<|im_end|>",
            "first_generated_token_id": 248046,
            "valid": False,
            "validation_category": "parse_error",
            "validation_message": "action_observation output must contain a JSON object",
        }
    ]

    traces, summary = loop.build_proposal_grpo_traces(args=args, prompt=prompt, proposal_results=results)

    assert len(traces) == 1
    assert traces[0].completion == "<|im_end|>"
    assert traces[0].reward == 0.0
    assert traces[0].advantage < 0.0
    assert traces[0].metadata["completion_source"] == "first_generated_token_for_empty_raw"
    assert summary["input_proposal_count"] == 1


def test_build_proposal_grpo_traces_clamps_positive_invalid_advantages():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=False,
    )
    prompt = PromptBundle(system="system", user="user")
    results = [
        {
            "proposal_index": 0,
            "id": "parse-error",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
        },
        {
            "proposal_index": 1,
            "id": "schema-error",
            "raw_output": '{"reasoning":"ok","unexpected":1}',
            "valid": False,
            "validation_category": "schema_error",
        },
    ]

    traces, summary = loop.build_proposal_grpo_traces(args=args, prompt=prompt, proposal_results=results)
    by_category = {trace.validation_category: trace for trace in traces}

    assert summary["advantage_mode"] == "normalized"
    assert summary["invalid_positive_advantage_clamped_count"] == 1
    assert by_category["parse_error"].advantage < 0.0
    assert by_category["schema_error"].metadata["raw_advantage"] > 0.0
    assert by_category["schema_error"].advantage == 0.0
    assert by_category["schema_error"].metadata["advantage_clamped"] is True
    assert all(trace.advantage <= 0.0 for trace in traces)


def test_build_proposal_grpo_traces_uses_outcome_rewards_and_skips_system_failures():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="fixed_baseline",
        proposal_grpo_fixed_baseline=0.0,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=False,
    )
    prompt = PromptBundle(system="system", user="user")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    results = [
        {
            "proposal_index": 0,
            "id": "trained-good",
            "raw_output": "verbose json",
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "bad-json",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
        },
        {
            "proposal_index": 2,
            "id": "system-failure",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 3,
            "id": "no-pseudo",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
    ]
    metrics = [
        loop.CandidateMetrics(
            index=0,
            row_id="trained-good",
            proposal=proposal,
            valid=True,
            reward=0.025,
            frontier_delta=0.02,
            frontier_accuracy=0.42,
            current_frontier_accuracy=0.4,
            target_accuracy=0.5,
            current_target_accuracy=0.45,
            target_delta=0.05,
            final_accuracy=0.6,
            current_final_accuracy=0.55,
            init_final_accuracy=0.5,
            final_accuracy_delta=0.1,
            final_accuracy_delta_from_current=0.05,
            per_size_accuracy={5: 0.5},
            pseudo_count=10,
            model_dir=None,
        ),
        loop.CandidateMetrics(
            index=2,
            row_id="system-failure",
            proposal=proposal,
            valid=False,
            reward=float("-inf"),
            frontier_delta=float("-inf"),
            target_accuracy=math.nan,
            current_target_accuracy=0.45,
            final_accuracy=math.nan,
            current_final_accuracy=0.55,
            init_final_accuracy=0.5,
            final_accuracy_delta=math.nan,
            final_accuracy_delta_from_current=math.nan,
            per_size_accuracy={},
            pseudo_count=10,
            model_dir=None,
            failure_reason="CUDA out of memory",
        ),
        loop.CandidateMetrics(
            index=3,
            row_id="no-pseudo",
            proposal=proposal,
            valid=False,
            reward=float("-inf"),
            frontier_delta=float("-inf"),
            target_accuracy=math.nan,
            current_target_accuracy=0.45,
            final_accuracy=math.nan,
            current_final_accuracy=0.55,
            init_final_accuracy=0.5,
            final_accuracy_delta=math.nan,
            final_accuracy_delta_from_current=math.nan,
            per_size_accuracy={},
            pseudo_count=0,
            model_dir=None,
            failure_reason="no pseudo labels retained",
        ),
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.proposal_index for trace in traces] == [0, 1, 3]
    assert [trace.reward for trace in traces] == [0.6, -1.0, 0.0]
    assert traces[0].completion == proposal.to_completion()
    assert traces[0].metadata["reward_source"] == "outcome"
    assert traces[0].metadata["candidate_reward"] == 0.025
    assert traces[0].metadata["final_accuracy_delta"] == 0.1
    assert traces[1].metadata["reward_source"] == "invalid"
    assert traces[2].metadata["reward_source"] == "valid_untrained"
    assert summary["skipped_system_failure_count"] == 1
    assert summary["reward_source_counts"] == {
        "outcome": 1,
        "invalid": 1,
        "skipped_system_failure": 1,
        "valid_untrained": 1,
    }


def _metric_stub(*, index, proposal, valid=True, reward=0.0, failure_reason=None):
    return SimpleNamespace(
        index=index,
        valid=valid,
        failure_reason=failure_reason,
        reward=reward,
        frontier_delta=reward,
        final_accuracy=0.5 + reward,
        final_accuracy_delta=reward,
        final_accuracy_delta_from_current=reward,
        target_delta=reward,
        proposal_prediction=None,
    )


def test_build_proposal_grpo_traces_deduplicates_actions_before_advantages():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=True,
    )
    prompt = PromptBundle(system="system", user="user")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    results = [
        {
            "proposal_index": 0,
            "id": "duplicate-weaker",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "duplicate-stronger",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 2,
            "id": "bad-json",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
        },
    ]
    metrics = [
        _metric_stub(index=0, proposal=proposal, reward=0.01),
        _metric_stub(index=1, proposal=proposal, reward=0.03),
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.proposal_index for trace in traces] == [1, 2]
    assert [trace.reward for trace in traces] == [0.53, -1.0]
    assert summary["pre_dedup_trace_count"] == 3
    assert summary["deduplicated_action_count"] == 1
    assert summary["unique_action_count"] == 2
    assert summary["zero_variance_skip"] is False


def test_build_proposal_grpo_traces_does_not_deduplicate_invalid_outputs():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=True,
    )
    prompt = PromptBundle(system="system", user="user")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    results = [
        {
            "proposal_index": 0,
            "id": "duplicate-valid-a",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "duplicate-valid-b",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        },
        {
            "proposal_index": 2,
            "id": "empty-eos-a",
            "raw_output": "",
            "first_generated_token_text": "<|im_end|>",
            "valid": False,
            "validation_category": "parse_error",
        },
        {
            "proposal_index": 3,
            "id": "empty-eos-b",
            "raw_output": "",
            "first_generated_token_text": "<|im_end|>",
            "valid": False,
            "validation_category": "parse_error",
        },
    ]
    metrics = [
        _metric_stub(index=0, proposal=proposal, reward=0.01),
        _metric_stub(index=1, proposal=proposal, reward=0.02),
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.proposal_index for trace in traces] == [1, 2, 3]
    assert [trace.completion for trace in traces[1:]] == ["<|im_end|>", "<|im_end|>"]
    assert summary["deduplicated_action_count"] == 1
    assert summary["pre_dedup_trace_count"] == 4
    assert summary["trace_candidate_metric_count"] == 2


def test_build_proposal_grpo_traces_adds_novelty_bonus_and_entropy_metrics():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=True,
        proposal_grpo_novelty_bonus_beta=0.05,
    )
    prompt = PromptBundle(system="system", user="user")
    repeated = ConfigProposal(left=2, right=3, guard="none", target=5)
    novel = ConfigProposal(left=4, right=3, guard="none", target=7)
    history = [
        SimpleNamespace(metadata={"left": 2, "right": 3, "guard": "none", "target": 5}),
        SimpleNamespace(metadata={"left": 2, "right": 3, "guard": "none", "target": 5}),
    ]
    results = [
        {
            "proposal_index": 0,
            "id": "repeated-a",
            "raw_output": repeated.to_completion(),
            "valid": True,
            "parsed_proposal": repeated.to_json_dict(),
            "completion": repeated.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "repeated-b",
            "raw_output": repeated.to_completion(),
            "valid": True,
            "parsed_proposal": repeated.to_json_dict(),
            "completion": repeated.to_completion(),
        },
        {
            "proposal_index": 2,
            "id": "novel",
            "raw_output": novel.to_completion(),
            "valid": True,
            "parsed_proposal": novel.to_json_dict(),
            "completion": novel.to_completion(),
        },
    ]
    metrics = [
        _metric_stub(index=0, proposal=repeated, reward=0.0),
        _metric_stub(index=1, proposal=repeated, reward=0.0),
        _metric_stub(index=2, proposal=novel, reward=0.0),
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
        action_history=history,
    )

    by_index = {trace.proposal_index: trace for trace in traces}
    assert set(by_index) == {0, 2}
    assert by_index[0].metadata["action_history_count"] == 2
    assert by_index[0].metadata["current_action_multiplicity"] == 2
    assert by_index[2].metadata["action_history_count"] == 0
    assert by_index[2].metadata["novelty_bonus"] > by_index[0].metadata["novelty_bonus"]
    assert by_index[2].reward > by_index[0].reward
    assert summary["pre_dedup_action_count"] == 3
    assert summary["pre_dedup_unique_action_count"] == 2
    assert summary["pre_dedup_duplicate_action_rate"] > 0.0
    assert summary["trainable_unique_action_count"] == 2
    assert summary["action_history_count"] == 2
    assert summary["novelty_bonus_max"] == by_index[2].metadata["novelty_bonus"]


def test_build_proposal_grpo_traces_maps_draw_results_to_candidate_metrics():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="fixed_baseline",
        proposal_grpo_fixed_baseline=0.0,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=False,
    )
    prompt = PromptBundle(system="system", user="user")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    results = [
        {
            "proposal_index": 7,
            "candidate_proposal_index": 0,
            "kept_for_candidate": True,
            "id": "draw-7",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
        }
    ]
    metrics = [_metric_stub(index=0, proposal=proposal, reward=0.025)]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.proposal_index for trace in traces] == [7]
    assert traces[0].reward == 0.525
    assert traces[0].metadata["candidate_reward"] == 0.025
    assert summary["trace_candidate_metric_count"] == 1


def test_build_proposal_grpo_traces_invalid_only_group_still_trains_with_default_skip():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=False,
    )
    prompt = PromptBundle(system="system", user="user")
    results = [
        {
            "proposal_index": 0,
            "id": "bad-json-a",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
        },
        {
            "proposal_index": 1,
            "id": "bad-json-b",
            "raw_output": "still not json",
            "valid": False,
            "validation_category": "parse_error",
        },
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=[],
    )

    assert len(traces) == 2
    assert summary["requested_zero_variance_mode"] == "skip"
    assert summary["advantage_mode"] == "fixed_baseline"
    assert summary["zero_variance_skip"] is False
    assert summary["invalid_only_fixed_baseline"] is True
    assert all(trace.advantage < 0.0 for trace in traces)


def test_build_proposal_grpo_traces_skips_candidate_deduped_actions():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="outcome",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=True,
    )
    prompt = PromptBundle(system="system", user="user")
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    results = [
        {
            "proposal_index": 0,
            "id": "trained",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
            "candidate_dedup_skipped": False,
        },
        {
            "proposal_index": 1,
            "id": "duplicate-skipped",
            "raw_output": proposal.to_completion(),
            "valid": True,
            "parsed_proposal": proposal.to_json_dict(),
            "completion": proposal.to_completion(),
            "candidate_dedup_skipped": True,
            "candidate_dedup_kept_proposal_index": 0,
        },
    ]
    metrics = [_metric_stub(index=0, proposal=proposal, reward=-0.03)]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.proposal_index for trace in traces] == [0]
    assert traces[0].reward == 0.47
    assert summary["skipped_candidate_duplicate_count"] == 1
    assert summary["reward_source_counts"]["candidate_dedup_skipped"] == 1
    assert summary["deduplicated_action_count"] == 0


def test_build_proposal_grpo_traces_supports_rank_rewards():
    args = argparse.Namespace(
        proposal_grpo_zero_variance="skip",
        proposal_grpo_fixed_baseline=0.5,
        proposal_grpo_reward_mode="rank",
        proposal_grpo_outcome_scale=0.05,
        proposal_grpo_deduplicate_actions=True,
    )
    prompt = PromptBundle(system="system", user="user")
    weak = ConfigProposal(left=2, right=3, guard="none", target=5)
    strong = ConfigProposal(left=3, right=4, guard="none", target=7)
    results = [
        {
            "proposal_index": 0,
            "id": "weak",
            "raw_output": weak.to_completion(),
            "valid": True,
            "parsed_proposal": weak.to_json_dict(),
            "completion": weak.to_completion(),
        },
        {
            "proposal_index": 1,
            "id": "strong",
            "raw_output": strong.to_completion(),
            "valid": True,
            "parsed_proposal": strong.to_json_dict(),
            "completion": strong.to_completion(),
        },
        {
            "proposal_index": 2,
            "id": "bad-json",
            "raw_output": "not json",
            "valid": False,
            "validation_category": "parse_error",
        },
    ]
    metrics = [
        _metric_stub(index=0, proposal=weak, reward=0.10),
        _metric_stub(index=1, proposal=strong, reward=0.30),
    ]

    traces, summary = loop.build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=results,
        candidate_metrics=metrics,
    )

    assert [trace.reward for trace in traces] == [0.0, 1.0, -1.0]
    assert [trace.metadata["reward_source"] for trace in traces] == ["rank_outcome", "rank_outcome", "invalid"]
    assert summary["reward_mode"] == "rank"
    assert summary["reward_source_counts"] == {"rank_outcome": 2, "invalid": 1}


class _CharTokenizer:
    bos_token_id = None
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_merged_proposal_policy_mask_supports_reasoning_action_and_action_only():
    tokenizer = _CharTokenizer()
    completion = '{"reasoning":"We are solid at 2 and 3, so I will target 5.","left":2,"right":3,"guard":"none"}'
    trace = adaptive_proposal.ProposalGRPOTrace(
        proposal_index=0,
        proposal_id="candidate",
        prompt_text="prompt",
        completion=completion,
        reward=1.0,
        advantage=1.0,
        validation_category="valid",
        validation_message="",
        valid=True,
        metadata={"proposal_output_schema": "action_observation"},
    )
    full_sample = adaptive_proposal._encode_proposal_grpo_sample(
        tokenizer=tokenizer,
        prompt_text="prompt",
        completion=completion,
        completion_char_span=adaptive_proposal._proposal_policy_span(trace, span_mode="reasoning_action"),
    )
    action_sample = adaptive_proposal._encode_proposal_grpo_sample(
        tokenizer=tokenizer,
        prompt_text="prompt",
        completion=completion,
        completion_char_span=adaptive_proposal._proposal_policy_span(trace, span_mode="action_only"),
    )

    assert full_sample is not None
    assert action_sample is not None
    assert full_sample["completion_tokens"] == full_sample["total_completion_tokens"]
    assert action_sample["completion_tokens"] == action_sample["total_completion_tokens"]


def test_format_replay_mask_removes_config_values_but_keeps_json_syntax():
    tokenizer = _CharTokenizer()
    completion = (
        '{"reasoning":"We are solid at 2 and 3.",'
        '"expected_avg_delta_from_current":0.03,'
        '"expected_target_delta":0.11,"left":2,'
        '"right":3,"guard":"none"}'
    )

    exclude_spans = adaptive_proposal._proposal_format_value_char_spans(completion)
    sample = adaptive_proposal._encode_proposal_grpo_sample(
        tokenizer=tokenizer,
        prompt_text="P",
        completion=completion,
        completion_char_exclude_spans=exclude_spans,
    )

    assert sample is not None
    assert sample["completion_tokens"] < sample["total_completion_tokens"]
    completion_mask = sample["completion_mask"]
    assert len(completion_mask) == len(completion)
    assert completion_mask[0] is True
    assert completion_mask[completion.index('"reasoning"')] is True

    reasoning_span = adaptive_proposal._json_value_content_char_span(completion, "reasoning")
    assert reasoning_span is not None
    assert not any(completion_mask[index] for index in range(*reasoning_span))

    left_value_index = completion.index('"left":') + len('"left":')
    right_value_index = completion.index('"right":') + len('"right":')
    assert completion_mask[left_value_index] is False
    assert completion_mask[right_value_index] is False

    target_prediction_span = adaptive_proposal._json_value_content_char_span(
        completion,
        "expected_target_delta",
    )
    assert target_prediction_span is not None
    assert not any(completion_mask[index] for index in range(*target_prediction_span))

    guard_value_span = adaptive_proposal._json_value_char_span(completion, "guard")
    guard_content_span = adaptive_proposal._json_value_content_char_span(completion, "guard")
    assert guard_value_span is not None
    assert guard_content_span is not None
    assert completion_mask[guard_value_span[0]] is True
    assert completion_mask[guard_content_span[0]] is False
    assert completion_mask[guard_value_span[1] - 1] is True


def test_format_replay_mask_can_be_disabled_by_passing_no_exclude_spans():
    tokenizer = _CharTokenizer()
    completion = '{"left":2,"right":3,"guard":"none"}'

    sample = adaptive_proposal._encode_proposal_grpo_sample(
        tokenizer=tokenizer,
        prompt_text="P",
        completion=completion,
        completion_char_exclude_spans=(),
    )

    assert sample is not None
    assert sample["completion_tokens"] == sample["total_completion_tokens"]
    assert all(sample["completion_mask"])


def test_merged_observation_completion_uses_realized_observation():
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    trace = adaptive_proposal.ProposalGRPOTrace(
        proposal_index=0,
        proposal_id="candidate",
        prompt_text="prompt",
        completion="{}",
        reward=0.5,
        advantage=1.0,
        validation_category="valid",
        validation_message="",
        valid=True,
        metadata={
            "reward_source": "outcome",
            "parsed_proposal": proposal.to_json_dict(),
            "frontier_delta": 0.123,
            "final_accuracy_delta": 0.456,
            "final_accuracy_delta_from_current": 0.05,
            "target_delta": 0.111,
            "per_size_delta": {4: -0.01, 5: 0.222},
            "proposal_prediction": {
                "expected_frontier_delta": 0.1,
                "expected_target_delta": 0.1,
                "expected_avg_delta_from_current": 0.02,
                "expected_avg_delta_from_init": 0.4,
            },
        },
    )

    completion = adaptive_proposal._realized_observation_completion(trace)

    assert completion is not None
    payload = json.loads(completion)
    assert payload["valid"] is True
    assert payload["trained"] is True
    assert "reward" not in payload
    assert payload["target"] == 5
    assert payload["frontier_delta"] == 0.123
    assert payload["avg_delta_from_init"] == 0.456
    assert payload["avg_delta_from_current"] == 0.05
    assert payload["target_delta"] == 0.111
    assert payload["prediction"] == {
        "expected_frontier_delta": 0.1,
        "expected_target_delta": 0.1,
        "expected_avg_delta_from_current": 0.02,
        "expected_avg_delta_from_init": 0.4,
    }
    assert payload["frontier_delta_error"] == 0.023
    assert payload["target_delta_error"] == 0.011
    assert payload["avg_delta_from_current_error"] == 0.03
    assert payload["avg_delta_from_init_error"] == 0.056
    assert payload["delta_per_size"] == {"4": -0.01, "5": 0.222}


def test_proposal_policy_microbatches_match_full_batch_gradient():
    import torch

    class _TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class _TinyCausalModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 8):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.linspace(-0.3, 0.4, vocab_size))

        def forward(self, input_ids, attention_mask=None):
            logits = self.bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
            return SimpleNamespace(logits=logits)

    tokenizer = _TinyTokenizer()
    samples = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "completion_mask": [False, True],
        },
        {
            "input_ids": [1, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1],
            "completion_mask": [False, True, True],
        },
        {
            "input_ids": [1, 2, 6],
            "attention_mask": [1, 1, 1],
            "completion_mask": [False, True],
        },
    ]
    advantages = torch.tensor([1.0, -0.25, 0.5], dtype=torch.float32)
    old_logprobs = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)
    kl_coef = 0.07

    full_model = _TinyCausalModel()
    micro_model = _TinyCausalModel()
    micro_model.load_state_dict(full_model.state_dict())

    full_batch = adaptive_proposal._collate_proposal_grpo_samples(
        tokenizer=tokenizer,
        samples=samples,
        device=torch.device("cpu"),
    )
    full_logprobs = adaptive_proposal._proposal_completion_mean_logprobs(full_model, full_batch)
    summed_logprobs = adaptive_proposal._proposal_completion_logprobs(
        full_model,
        full_batch,
        normalize_by_length=False,
    )
    token_counts = full_batch["completion_mask"].sum(dim=1).clamp_min(1).float()
    assert torch.allclose(summed_logprobs, full_logprobs * token_counts, atol=1e-6)
    full_loss = (
        (-(advantages * full_logprobs)).sum()
        + kl_coef * ((full_logprobs - old_logprobs) ** 2).sum()
    ) / len(samples)
    full_loss.backward()

    micro_metrics = adaptive_proposal._backward_policy_microbatches(
        model=micro_model,
        tokenizer=tokenizer,
        samples=samples,
        advantages=advantages,
        old_logprobs=old_logprobs,
        device=torch.device("cpu"),
        microbatch_size=1,
        kl_coef=kl_coef,
        normalize_by_length=True,
    )

    assert torch.allclose(micro_model.bias.grad, full_model.bias.grad, atol=1e-6)
    assert math.isclose(
        micro_metrics["policy_loss"]
        + kl_coef * micro_metrics["kl_proxy"],
        float(full_loss.detach()),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def test_proposal_completion_loss_requests_tail_logits_and_disables_cache():
    import torch

    class _TailLogitModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 8):
            super().__init__()
            self.vocab_size = vocab_size
            self.calls = []

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0, use_cache=True):
            self.calls.append(
                {
                    "sequence_length": int(input_ids.shape[1]),
                    "logits_to_keep": int(logits_to_keep),
                    "use_cache": bool(use_cache),
                }
            )
            batch_size, sequence_length = input_ids.shape
            positions = torch.arange(sequence_length, dtype=torch.float32).view(1, sequence_length, 1)
            vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, self.vocab_size)
            logits = (positions * 0.17 + vocab * 0.03).expand(batch_size, sequence_length, self.vocab_size)
            if logits_to_keep:
                logits = logits[:, -int(logits_to_keep) :, :]
            return SimpleNamespace(logits=logits)

    model = _TailLogitModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "completion_mask": torch.tensor([[False, False, True, True]], dtype=torch.bool),
    }

    logprobs = adaptive_proposal._proposal_completion_logprobs(
        model,
        batch,
        normalize_by_length=False,
    )

    assert logprobs.shape == (1,)
    assert model.calls == [
        {
            "sequence_length": 5,
            "logits_to_keep": 3,
            "use_cache": False,
        }
    ]


def test_select_candidate_tiebreaks_by_frontier_delta_before_target_delta():
    proposal_a = ConfigProposal(left=2, right=3, guard="none", target=5)
    proposal_b = ConfigProposal(left=3, right=3, guard="none", target=6)
    common = dict(
        valid=True,
        reward=0.1,
        target_accuracy=0.5,
        current_target_accuracy=0.4,
        final_accuracy=0.6,
        current_final_accuracy=0.5,
        init_final_accuracy=0.45,
        final_accuracy_delta=0.15,
        final_accuracy_delta_from_current=0.1,
        per_size_accuracy={},
        pseudo_count=10,
        model_dir=None,
    )
    weaker_frontier = loop.CandidateMetrics(
        index=0,
        row_id="target-good",
        proposal=proposal_a,
        frontier_delta=0.01,
        target_delta=0.09,
        **common,
    )
    stronger_frontier = loop.CandidateMetrics(
        index=1,
        row_id="frontier-good",
        proposal=proposal_b,
        frontier_delta=0.02,
        target_delta=0.01,
        **common,
    )

    assert loop.select_candidate([weaker_frontier, stronger_frontier], min_reward=0.0) == stronger_frontier
