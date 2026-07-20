from __future__ import annotations

import json

from self.coding.atomic_data import AtomicExample, canonical_json
from self.coding.bfcl_composition import (
    HELDOUT_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    audit_decision,
    build_mutated_atomic_variants,
    build_round1_cross_candidates,
    build_round1_repeat_candidates,
    compose_component_predictions,
    guard_prediction,
    oracle_example,
    render_joined_request,
)


def example(
    source_id: str,
    function_name: str,
    question: str,
    argument_name: str,
    value,
    value_type: str,
) -> AtomicExample:
    function = {
        "name": function_name,
        "description": f"Call {function_name}",
        "parameters": {
            "type": "object",
            "properties": {argument_name: {"type": value_type}},
            "required": [argument_name],
        },
    }
    calls = [{"name": function_name, "arguments": {argument_name: value}}]
    return AtomicExample(
        task="bfcl",
        source_id=source_id,
        source_group_id=source_id,
        split="hidden_composition",
        messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": question},
        ),
        target=canonical_json(calls),
        evaluator={
            "functions": [function],
            "accepted_calls": [
                {"name": function_name, "arguments": {argument_name: [value]}}
            ],
        },
        component_count=1,
        source_component_ids=(source_id,),
        metadata={"question": question},
    )


def test_template_registry_is_disjoint_and_supports_arbitrary_arity():
    assert set(TRAIN_TEMPLATE_IDS).isdisjoint(HELDOUT_TEMPLATE_IDS)
    for template_id in (*TRAIN_TEMPLATE_IDS, *HELDOUT_TEMPLATE_IDS):
        rendered = render_joined_request(["Do alpha.", "Do beta?", "Do gamma!"], template_id)
        assert "alpha" in rendered
        assert "beta" in rendered
        assert "gamma" in rendered


def test_cross_candidates_are_deterministic_and_public_records_do_not_leak_labels():
    left = example("left", "weather", "Weather in Paris", "city", "Paris", "string")
    right = example("right", "stock", "Stock price for ACME", "ticker", "ACME", "string")
    public, oracle = build_round1_cross_candidates([left, right], seed=7)
    public_again, oracle_again = build_round1_cross_candidates([left, right], seed=7)
    assert public == public_again
    assert oracle == oracle_again
    assert len(public) == 1
    assert not ({"target", "canonical_calls", "accepted_calls", "evaluator"} & set(public[0]))
    assert len(oracle[0]["canonical_calls"]) == 2


def test_g4_rejects_structural_errors_but_allows_semantically_wrong_string():
    item = example("x", "weather", "Weather in Paris", "city", "Paris", "string")
    spec = {
        "expected_call_count": 1,
        "functions": item.evaluator["functions"],
        "allow_exact_duplicates": False,
    }
    wrong_value = '[{"name":"weather","arguments":{"city":"Rome"}}]'
    assert guard_prediction(wrong_value, spec, level="g4")["accepted"]
    wrong_name = '[{"name":"currency","arguments":{"city":"Paris"}}]'
    assert guard_prediction(wrong_name, spec, level="g4")["reasons"] == ["function_not_available"]
    missing = '[{"name":"weather","arguments":{}}]'
    assert "missing_required_argument" in guard_prediction(missing, spec, level="g4")["reasons"]
    fenced = "```json\n[]\n```"
    assert guard_prediction(fenced, spec, level="g1")["reasons"] == ["invalid_json_array"]


def test_component_composition_and_oracle_audit_expose_false_accept():
    left = example("left", "weather", "Weather in Paris", "city", "Paris", "string")
    right = example("right", "stock", "Stock price for ACME", "ticker", "ACME", "string")
    public, oracle = build_round1_cross_candidates([left, right], seed=7)
    raw = {
        "left": '[{"name":"weather","arguments":{"city":"Rome"}}]',
        "right": '[{"name":"stock","arguments":{"ticker":"ACME"}}]',
    }
    decision = compose_component_predictions(public[0], raw, level="g4")
    audit = audit_decision(public[0], oracle[0], decision)
    assert decision["accepted"]
    assert audit["false_accept"]
    assert not audit["oracle_exact"]


def test_numeric_mutation_updates_question_and_hidden_call():
    item = example("fall", "fall_speed", "Compute speed after 5 seconds", "time", 5, "integer")
    variants = build_mutated_atomic_variants(
        item,
        split="hidden_composition",
        max_variants=2,
        seed=7,
    )
    assert len(variants) == 2
    for variant in variants:
        target = json.loads(variant.target)[0]
        assert target["arguments"]["time"] != 5
        assert str(target["arguments"]["time"]) in variant.metadata["question"]
        assert variant.metadata["synthetic_mutation"]["source_id"] == "fall"


def test_repeat_candidates_keep_semantic_pair_provenance_across_renders():
    item = example("weather", "weather", "Weather in London", "city", "London", "string")
    public, oracle, audit = build_round1_repeat_candidates(
        [item],
        split="hidden_composition",
        seed=7,
        max_variants_per_source=1,
        template_partition="train",
        renders_per_pair=2,
    )
    assert audit["qualifying_source_count"] == 1
    assert audit["semantic_pair_count"] == 1
    assert len(public) == len(oracle) == 2
    assert set(public[0]["source_component_ids"]) == set(public[1]["source_component_ids"])
    assert public[0]["candidate_id"] != public[1]["candidate_id"]


def test_oracle_example_uses_true_calls_only_in_explicit_oracle_adapter():
    left = example("left", "weather", "Weather in Paris", "city", "Paris", "string")
    right = example("right", "stock", "Stock price for ACME", "ticker", "ACME", "string")
    public, oracle = build_round1_cross_candidates([left, right], seed=7)
    adapted = oracle_example(public[0], oracle[0])
    assert json.loads(adapted.target) == oracle[0]["canonical_calls"]
    assert adapted.evaluator["accepted_calls"] == oracle[0]["accepted_calls"]
