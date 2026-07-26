from __future__ import annotations

import json
from pathlib import Path

from self.coding.atomic_data import AtomicExample, canonical_json
from self.coding.evaluation import evaluate_bfcl
from self.experiments.bfcl_schema_generalization_audit import (
    VARIANTS,
    audit_cells,
    generation_budget,
    permute_schema_order,
    rename_schema_identifiers,
    structured_bfcl_metrics,
)


def example() -> AtomicExample:
    functions = [
        {
            "name": "weather_lookup",
            "description": "Find weather using city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "currency_lookup",
            "description": "Find a currency using country.",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string"}},
                "required": ["country"],
            },
        },
    ]
    target = [
        {"name": "weather_lookup", "arguments": {"city": "Paris"}},
        {"name": "currency_lookup", "arguments": {"country": "France"}},
    ]
    accepted = [
        {"name": "weather_lookup", "arguments": {"city": ["Paris"]}},
        {"name": "currency_lookup", "arguments": {"country": ["France"]}},
    ]
    user = (
        "User request:\nCheck Paris weather and the currency of France."
        "\n\nAvailable functions:\n"
        + canonical_json(functions)
    )
    return AtomicExample(
        task="bfcl",
        source_id="heldout-example",
        source_group_id="heldout-example",
        split="test",
        messages=(
            {"role": "system", "content": "Return calls."},
            {"role": "user", "content": user},
        ),
        target=canonical_json(target),
        evaluator={"functions": functions, "accepted_calls": accepted},
        component_count=2,
        source_component_ids=("a", "b"),
        evaluation_track="controlled",
    )


def test_audit_grid_crosses_two_models_and_seven_variants():
    cells = audit_cells()
    assert len(VARIANTS) == 7
    assert len(cells) == 14
    assert [cell.index for cell in cells] == list(range(14))
    assert cells[0].cell_id == "seed--original"
    assert cells[-1].cell_id == "g1_round3--identifier_renamed"


def test_identifier_renaming_doubles_generation_budget():
    assert generation_budget(1, "original") == 128
    assert generation_budget(8, "original") == 544
    assert generation_budget(1, "identifier_renamed") == 256
    assert generation_budget(8, "identifier_renamed") == 1088


def test_schema_permutation_changes_only_schema_order():
    original = example()
    transformed = permute_schema_order(original, permutation_index=3)
    assert transformed.target == original.target
    assert transformed.evaluator["accepted_calls"] == original.evaluator["accepted_calls"]
    assert {
        function["name"] for function in transformed.evaluator["functions"]
    } == {
        function["name"] for function in original.evaluator["functions"]
    }
    prompt = transformed.messages[-1]["content"]
    rendered = json.loads(prompt.rsplit("\n\nAvailable functions:\n", 1)[1])
    assert rendered == transformed.evaluator["functions"]


def test_identifier_renaming_is_consistent_and_preserves_gold_exactness():
    original = example()
    renamed = rename_schema_identifiers(original)
    assert evaluate_bfcl(renamed, renamed.target).exact
    assert {
        function["name"] for function in renamed.evaluator["functions"]
    }.isdisjoint(
        function["name"] for function in original.evaluator["functions"]
    )
    assert "weather_lookup" not in renamed.target
    assert '"city"' not in renamed.target
    prompt = renamed.messages[-1]["content"]
    rendered = json.loads(prompt.rsplit("\n\nAvailable functions:\n", 1)[1])
    assert rendered == renamed.evaluator["functions"]


def test_structured_metrics_separate_selection_keys_and_values():
    item = example()
    prediction = canonical_json(
        [
            {"name": "weather_lookup", "arguments": {"city": "London"}},
            {"name": "currency_lookup", "arguments": {"country": "France"}},
        ]
    )
    summary, rows = structured_bfcl_metrics([item], [prediction])
    assert summary["function_selection_accuracy"] == 1.0
    assert summary["argument_key_call_accuracy"] == 1.0
    assert summary["argument_value_call_accuracy"] == 0.5
    assert summary["argument_value_exact_accuracy"] == 0.0
    assert rows[0]["argument_value_match_count"] == 1
