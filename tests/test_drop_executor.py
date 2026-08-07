"""Unit checks for the DROP-QDMR symbolic executor (plan §10, §11).

The executor is the exactly-executable half of the composition operator, so a
silent change in its semantics would corrupt every pseudo-label without
producing a visible failure.  These tests pin the semantics that the plan's
claims depend on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports" / "composition_screen"))

from drop_executor import (  # noqa: E402
    NodeRejection,
    UnsupportedOperator,
    execute_dag,
    execute_node,
    executor_depth,
    model_owned_count,
    parse_list,
    parse_number,
    parse_program,
    render_list,
    score,
    sink_type,
    verbalize,
)


def build(program, decomposition):
    return parse_program(program, decomposition)


# --------------------------------------------------------------------------
# the worked example of plan §1
# --------------------------------------------------------------------------

WORKED_PROGRAM = [
    "SELECT['the Raiders field goal']",
    "PROJECT['yards of #REF', '#1']",
    "SELECT['the Broncos field goal']",
    "PROJECT['yards of #REF', '#3']",
    "ARITHMETIC['difference', '#2', '#4']",
]
WORKED_DECOMP = ("return the Raiders field goal ;return yards  of #1 ;"
                 "return the Broncos field goal ;return yards of #3 ;"
                 "return the  difference of #2 and  #4")


def test_worked_example_executes_to_22() -> None:
    nodes = build(WORKED_PROGRAM, WORKED_DECOMP)
    assert model_owned_count(nodes) == 4
    assert sink_type(nodes) == "number"
    assert executor_depth(nodes) == 1

    answers = {1: ["a 45-yard field goal"], 2: ["45"],
               3: ["a 23-yard field goal"], 4: ["23"]}
    value, trace = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value == ["22"]
    assert trace.accepted
    assert score(value, "22") == (1.0, 1.0)


def test_difference_is_unsigned() -> None:
    """DROP phrases differences as "how many more", so operand order is free."""
    nodes = build(WORKED_PROGRAM, WORKED_DECOMP)
    answers = {1: ["fg"], 2: ["23"], 3: ["fg"], 4: ["45"]}
    value, _ = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value == ["22"]


def test_verbalization_fills_refs_and_normalizes_whitespace() -> None:
    nodes = build(WORKED_PROGRAM, WORKED_DECOMP)
    # node 2 is `return yards  of #1` with the doubled space BREAK emits
    question = verbalize(nodes[1], {1: ["a 45-yard field goal"]})
    assert question == "What is yards of a 45-yard field goal?"


def test_verbalization_prefers_longest_reference_index() -> None:
    """`#10` must not be clobbered by the substitution for `#1`."""
    program = [f"SELECT['s{i}']" for i in range(1, 11)] + [
        "PROJECT['x of #REF', '#10']"]
    decomp = " ;".join([f"return s{i}" for i in range(1, 11)] +
                       ["return x of #10 and #1"])
    nodes = build(program, decomp)
    parents = {1: ["one"], 10: ["ten"]}
    assert verbalize(nodes[10], parents) == "What is x of ten and one?"


def test_select_uses_the_agreement_free_template() -> None:
    """SELECT carries bare noun phrases and proper nouns; "What is people?"
    and "What is Adam Vinatieri?" both read badly, so SELECT gets its own
    wording (plan §9.1)."""
    nodes = build(["SELECT['people']"], "return people")
    assert verbalize(nodes[0], {}) == "Find in the passage: people"
    nodes = build(["SELECT['Adam Vinatieri']"], "return Adam Vinatieri")
    assert verbalize(nodes[0], {}) == "Find in the passage: Adam Vinatieri"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

COUNT_PROGRAM = ["SELECT['field goals']",
                 "PROJECT['yards of #REF', '#1']",
                 "AGGREGATE['count', '#2']"]
COUNT_DECOMP = "return field goals ;return yards of #1 ;return number of  #2"


def test_count_over_a_three_element_list() -> None:
    nodes = build(COUNT_PROGRAM, COUNT_DECOMP)
    answers = {1: ["fg1", "fg2", "fg3"], 2: ["45", "23", "31"]}
    value, trace = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value == ["3"]
    assert trace.executor_io[0]["inputs"] == {"2": ["45", "23", "31"]}


def test_sum_and_max_read_the_parent_list_as_numbers() -> None:
    for fn, expected in (("sum", "99"), ("max", "45"), ("min", "23"), ("avg", "33")):
        program = ["SELECT['field goals']", "PROJECT['yards of #REF', '#1']",
                   f"AGGREGATE['{fn}', '#2']"]
        nodes = build(program, COUNT_DECOMP)
        answers = {1: ["fg1", "fg2", "fg3"], 2: ["45", "23", "31"]}
        value, _ = execute_dag(nodes, lambda node, q: answers[node.node_id])
        assert value == [expected], fn


def test_ambiguous_count_is_rejected_with_the_node_id() -> None:
    """Plan Risk 6: `count` over a single numeric span is overloaded.

    "How many men in the army?" decomposes to AGGREGATE['count', ...] over a
    list holding `50,000`.  Returning 1 would be a confidently wrong label.
    """
    nodes = build(COUNT_PROGRAM, COUNT_DECOMP)
    answers = {1: ["the army"], 2: ["50,000"]}
    value, trace = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value is None
    assert trace.rejection == {"node_id": 3, "reason": "ambiguous_count",
                               "detail": "single numeric element '50,000'"}


def test_count_of_a_single_non_numeric_span_is_one() -> None:
    nodes = build(COUNT_PROGRAM, COUNT_DECOMP)
    answers = {1: ["the army"], 2: ["a field goal"]}
    value, _ = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value == ["1"]


# --------------------------------------------------------------------------
# typed rejections
# --------------------------------------------------------------------------

def test_non_numeric_operand_rejects_at_the_failing_node() -> None:
    nodes = build(WORKED_PROGRAM, WORKED_DECOMP)
    answers = {1: ["fg"], 2: ["forty-five"], 3: ["fg"], 4: ["23"]}
    value, trace = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value is None
    assert trace.rejection["node_id"] == 5
    assert trace.rejection["reason"] == "operand_parse"


def test_arithmetic_rejects_a_list_operand() -> None:
    nodes = build(WORKED_PROGRAM, WORKED_DECOMP)
    answers = {1: ["fg"], 2: ["45", "31"], 3: ["fg"], 4: ["23"]}
    value, trace = execute_dag(nodes, lambda node, q: answers[node.node_id])
    assert value is None
    assert trace.rejection == {"node_id": 5, "reason": "operand_not_scalar",
                               "detail": "#2 has 2 elements"}


def test_empty_model_answer_rejects() -> None:
    nodes = build(COUNT_PROGRAM, COUNT_DECOMP)
    value, trace = execute_dag(nodes, lambda node, q: [])
    assert value is None
    assert trace.rejection["reason"] == "empty_answer"
    assert trace.rejection["node_id"] == 1


def test_range_check_rejects_an_implausible_count() -> None:
    nodes = build(COUNT_PROGRAM, COUNT_DECOMP)
    long_list = [f"item {i}" for i in range(200)]
    with pytest.raises(NodeRejection) as excinfo:
        execute_node(nodes[2], {2: long_list})
    assert excinfo.value.reason == "range"
    assert excinfo.value.node_id == 3


def test_division_by_zero_is_a_typed_rejection() -> None:
    program = ["SELECT['a']", "SELECT['b']", "ARITHMETIC['division', '#1', '#2']"]
    decomp = "return a ;return b ;return division of #1 and #2"
    nodes = build(program, decomp)
    with pytest.raises(NodeRejection) as excinfo:
        execute_node(nodes[2], {1: ["10"], 2: ["0"]})
    assert excinfo.value.reason == "division_by_zero"


# --------------------------------------------------------------------------
# program parsing
# --------------------------------------------------------------------------

def test_unsupported_operator_is_reported_by_name() -> None:
    program = ["SELECT['quarters']", "PROJECT['goals of #REF', '#1']",
               "GROUP['count', '#2', '#1']"]
    decomp = "return quarters ;return goals of #1 ;return number of #2 for each #1"
    with pytest.raises(UnsupportedOperator) as excinfo:
        build(program, decomp)
    assert excinfo.value.op == "GROUP"


def test_comparison_is_out_of_scope() -> None:
    """COMPARISON is always the sink and always non-numeric (plan §6)."""
    program = ["SELECT['a']", "SELECT['b']", "COMPARISON['max', '#1', '#2']"]
    decomp = "return a ;return b ;return which is higher of #1 , #2"
    with pytest.raises(UnsupportedOperator):
        build(program, decomp)


def test_n_ary_arithmetic_sum() -> None:
    program = ["SELECT['army']", "PROJECT['men of #REF', '#1']",
               "PROJECT['horses of #REF', '#1']", "PROJECT['elephants of #REF', '#1']",
               "ARITHMETIC['sum', '#2', '#3', '#4']"]
    decomp = ("return army ;return men of #1 ;return horses of #1 ;"
              "return elephants of #1 ;return sum of #2 , #3 , #4")
    nodes = build(program, decomp)
    assert nodes[4].parents == [2, 3, 4]
    value = execute_node(nodes[4], {2: ["100"], 3: ["50"], 4: ["7"]})
    assert value == ["157"]


def test_forward_reference_is_rejected() -> None:
    program = ["SELECT['#2']", "SELECT['b']"]
    decomp = "return #2 ;return b"
    with pytest.raises(ValueError, match="later node"):
        build(program, decomp)


def test_step_count_must_match_program_length() -> None:
    with pytest.raises(ValueError, match="decomposition has"):
        build(["SELECT['a']"], "return a ;return b")


# --------------------------------------------------------------------------
# value parsing and scoring
# --------------------------------------------------------------------------

def test_list_round_trip() -> None:
    assert parse_list("Rhodes | Gonzalez") == ["Rhodes", "Gonzalez"]
    assert parse_list("Rhodes") == ["Rhodes"]
    assert parse_list("45 | 23\nbecause the passage says so") == ["45", "23"]
    assert render_list(["a", "b"]) == "a | b"


def test_operand_parser_handles_drop_number_surface_forms() -> None:
    assert parse_number("45-yard") == 45.0
    assert parse_number("50,000 men") == 50000.0
    assert parse_number("  23 ") == 23.0
    with pytest.raises(ValueError):
        parse_number("forty-five")
    with pytest.raises(ValueError):
        parse_number("between 30 and 40")


def test_official_metric_including_multi_span() -> None:
    assert score("22", "22") == (1.0, 1.0)
    assert score(["Rhodes", "Gonzalez"], ["Gonzalez", "Rhodes"]) == (1.0, 1.0)
    em, f1 = score(["Rhodes"], ["Gonzalez", "Rhodes"])
    assert em == 0.0 and 0.0 < f1 < 1.0
    # the official normalizer does not convert written-out numbers; this is why
    # `parse_number` exists as a separate operand parser (plan Risk 5)
    assert score("twenty-two", "22")[0] == 0.0
