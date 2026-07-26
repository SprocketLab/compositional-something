from __future__ import annotations

import json

from self.coding.atomic_data import AtomicExample, canonical_json
from self.coding.bfcl_composition import (
    HELDOUT_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    audit_decision,
    build_mutated_atomic_variants,
    build_hierarchical_cross_candidates,
    build_next_repeat_candidates,
    build_round1_cross_candidates,
    build_round1_repeat_candidates,
    build_round2_cross_candidates,
    build_controlled_evaluation,
    compose_component_predictions,
    guard_prediction,
    oracle_example,
    render_joined_request,
)


def _clause_positions(candidate, clauses) -> list:
    """Locate the clauses in the rendered parent request, scanning forward.

    Scanning forward means the returned offsets are strictly increasing exactly
    when the clause list is in rendered-request order, even if two clauses share
    the same surface text.
    """

    positions = []
    cursor = 0
    for clause in clauses:
        stripped = clause.strip().rstrip(".!?; ")
        index = candidate["question"].find(stripped, cursor)
        assert index >= 0, f"clause {stripped!r} is out of order in the joined request"
        positions.append(index)
        cursor = index + len(stripped)
    return positions


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


def test_round2_component_ids_identify_the_rendered_prompt():
    items = [
        example(
            f"source-{index}",
            f"function-{index}",
            f"Question {index}",
            f"argument-{index}",
            index,
            "integer",
        )
        for index in range(6)
    ]
    public, _oracle = build_round2_cross_candidates(items, seed=7, count=15)
    prompt_by_component_id = {}
    for candidate in public:
        for spec in candidate["component_specs"]:
            prompt = (spec["question"], spec["messages"])
            component_id = spec["component_id"]
            assert prompt_by_component_id.setdefault(component_id, prompt) == prompt


def test_hierarchical_candidates_cover_two_four_and_eight_calls():
    items = [
        example(
            f"source-{index}",
            f"function-{index}",
            f"Question {index}",
            f"argument-{index}",
            index,
            "integer",
        )
        for index in range(12)
    ]
    for calls in (2, 4, 8):
        public, oracle = build_hierarchical_cross_candidates(
            items, component_count=calls, count=5, seed=7
        )
        assert len(public) == len(oracle) == 5
        for candidate, hidden in zip(public, oracle):
            assert candidate["component_count"] == calls
            assert len(candidate["source_component_ids"]) == calls
            assert len(set(candidate["source_component_ids"])) == calls
            assert [spec["expected_call_count"] for spec in candidate["component_specs"]] == [
                calls // 2,
                calls // 2,
            ]
            assert len(hidden["canonical_calls"]) == calls


def test_oracle_example_uses_true_calls_only_in_explicit_oracle_adapter():
    left = example("left", "weather", "Weather in Paris", "city", "Paris", "string")
    right = example("right", "stock", "Stock price for ACME", "ticker", "ACME", "string")
    public, oracle = build_round1_cross_candidates([left, right], seed=7)
    adapted = oracle_example(public[0], oracle[0])
    assert json.loads(adapted.target) == oracle[0]["canonical_calls"]
    assert adapted.evaluator["accepted_calls"] == oracle[0]["accepted_calls"]


def _sources(count: int, *, start: int = 0):
    return [
        example(
            f"source-{index}",
            f"function-{index}",
            f"Question {index}",
            f"argument-{index}",
            index,
            "integer",
        )
        for index in range(start, start + count)
    ]


def _assert_clause_aligned_target(candidate, hidden, call_by_source) -> None:
    """Clause k of the joined request must be answered by call k of the target."""

    clauses = candidate["clause_questions"]
    sources = candidate["source_component_ids"]
    calls = hidden["canonical_calls"]
    assert len(clauses) == len(sources) == len(calls) == candidate["component_count"]
    assert calls == [call_by_source[source] for source in sources]
    positions = _clause_positions(candidate, clauses)
    assert positions == sorted(positions), "clause list is not in rendered request order"


def test_composed_targets_follow_prompt_clause_order_at_every_frontier():
    items = _sources(12)
    call_by_source = {item.source_id: json.loads(item.target)[0] for item in items}
    for calls in (2, 4, 8):
        public, oracle = build_hierarchical_cross_candidates(
            items, component_count=calls, count=8, seed=7
        )
        for candidate, hidden in zip(public, oracle):
            _assert_clause_aligned_target(candidate, hidden, call_by_source)
            # Each component answers a contiguous block of the parent clauses.
            offset = 0
            for spec in candidate["component_specs"]:
                width = len(spec["source_component_ids"])
                assert (
                    list(spec["source_component_ids"])
                    == candidate["source_component_ids"][offset : offset + width]
                )
                offset += width


def test_round2_and_controlled_evaluation_targets_follow_clause_order():
    items = _sources(8)
    call_by_source = {item.source_id: json.loads(item.target)[0] for item in items}
    public, oracle = build_round2_cross_candidates(items, seed=7, count=10)
    for candidate, hidden in zip(public, oracle):
        _assert_clause_aligned_target(candidate, hidden, call_by_source)
    for name, examples in build_controlled_evaluation(
        items, component_counts=(2, 4), examples_per_cell=6, seed=7
    ).items():
        arity = int(name.rsplit("_", 1)[1])
        for evaluation in examples:
            calls = json.loads(evaluation.target)
            assert len(calls) == arity
            assert calls == [
                call_by_source[source] for source in evaluation.source_component_ids
            ]


def test_repeat_frontier_targets_follow_clause_order():
    items = [
        example(f"repeat-{index}", f"function-{index}", f"Wait {index + 3} seconds", "time", index + 3, "integer")
        for index in range(6)
    ]
    public, oracle, _audit = build_round1_repeat_candidates(
        items,
        split="hidden_composition",
        seed=7,
        max_variants_per_source=2,
        template_partition="train",
        renders_per_pair=1,
    )
    call_by_source = {
        str(spec["component_id"]): component["canonical_calls"][0]
        for candidate, hidden in zip(public, oracle)
        for spec, component in zip(candidate["component_specs"], hidden["component_oracles"])
    }
    for candidate, hidden in zip(public, oracle):
        _assert_clause_aligned_target(candidate, hidden, call_by_source)
    paired_public, paired_oracle = build_next_repeat_candidates(
        public,
        oracle,
        round_index=2,
        component_call_count=2,
        seed=7,
        count=4,
    )
    for candidate, hidden in zip(paired_public, paired_oracle):
        _assert_clause_aligned_target(candidate, hidden, call_by_source)


def test_schema_presentation_order_is_independent_of_clause_order():
    items = _sources(16)
    name_by_source = {item.source_id: f"function-{item.source_id.rsplit('-', 1)[1]}" for item in items}
    public, _oracle = build_hierarchical_cross_candidates(
        items, component_count=4, count=120, seed=7
    )
    matched = 0
    for candidate in public:
        clause_order = [name_by_source[source] for source in candidate["source_component_ids"]]
        assert sorted(candidate["schema_order"]) == sorted(clause_order)
        matched += candidate["schema_order"] == clause_order
    # Clause-aligned schema listings would let a model answer positionally.
    # With four independent clauses, chance agreement is 1/24.
    assert matched / len(public) < 0.20
    reordered, _ = build_hierarchical_cross_candidates(
        items, component_count=4, count=120, seed=7
    )
    assert [row["schema_order"] for row in reordered] == [
        row["schema_order"] for row in public
    ]
