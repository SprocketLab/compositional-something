#!/usr/bin/env python3
"""QDMR node model, symbolic executor, and verbalization for DROP-QDMR-CSI.

Plan §8-§11.  This module is the exactly-executable half of the composition
operator: BREAK's logical forms mark some nodes as arithmetic or aggregation
over their parents, and those nodes are evaluated in Python instead of being
asked of the model.  Every node the executor owns removes label noise from the
pseudo-label, which is the property that distinguishes this benchmark from
MuSiQue (plan §1).

Value model
-----------
Every value is a `list[str]`.  A scalar is the length-one case.  This is forced
by the operator distribution: `AGGREGATE['count', '#2']` accounts for 4,349 of
6,317 aggregate nodes across DROP train and dev, and it consumes the list a
PROJECT or FILTER node produced.  Making the atom list-valued everywhere keeps
one output format at every model-owned node (plan §9.1).

Operator table, measured over DROP rows in break_dataset/logical-forms
(train + dev, 2026-08-06):

    model-owned   SELECT 11420, PROJECT 13363, FILTER 3995
    executor      AGGREGATE  count 4349, sum 813, min 549, max 534, avg 72
                  ARITHMETIC difference 2176, sum 285, division 6,
                             multiplication 1
    excluded      COMPARATIVE, GROUP, DISCARD, BOOLEAN, SUPERLATIVE,
                  INTERSECTION, UNION

COMPARISON is deliberately absent.  It occurs 1,614 times and is the sink every
single time (0 mid-DAG occurrences), and COMPARISON sinks are non-numeric, so
no numeric-sink DAG contains one.  Its semantics also differ from the rest:
`COMPARISON['max', '#5', '#6']` returns *which reference* was larger, not the
larger value.

ARITHMETIC is n-ary.  34 rows use three operands and a few use more, e.g.
`ARITHMETIC['sum', '#5', '#6', '#7']`.

Scoring uses the official DROP metric, vendored verbatim in `drop_eval.py`.
`parse_number` below is an operand parser, not a second answer normalizer: it
turns a span such as "45-yard" into 45.0 so an arithmetic node can run.
Scoring never goes through it (plan Risk 5).
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).parent))
from drop_eval import get_metrics  # noqa: E402  official DROP EM/F1

LIST_DELIMITER = " | "

MODEL_OWNED_OPS = frozenset({"SELECT", "PROJECT", "FILTER"})
EXECUTOR_OPS = frozenset({"AGGREGATE", "ARITHMETIC"})
SUPPORTED_OPS = MODEL_OWNED_OPS | EXECUTOR_OPS

AGGREGATE_FNS = frozenset({"count", "sum", "min", "max", "avg"})
ARITHMETIC_FNS = frozenset({"difference", "sum", "division", "multiplication"})

# AGGREGATE fns other than `count` read their parent list as numbers.
NUMERIC_PARENT_AGGREGATES = frozenset({"sum", "min", "max", "avg"})

_STEP_RE = re.compile(r"^([A-Z]+)\[(.*)\]$", re.S)
_REF_RE = re.compile(r"#(\d+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


class NodeRejection(Exception):
    """Execution stopped at a node.  Carries the node id and a typed reason.

    Reasons are the rejection-cause vocabulary of plan §16 and must stay stable,
    because the rejection-cause distribution is a reported metric.
    """

    def __init__(self, node_id: int, reason: str, detail: str = ""):
        super().__init__(f"node {node_id}: {reason}" + (f" ({detail})" if detail else ""))
        self.node_id = node_id
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# value parsing
# --------------------------------------------------------------------------

def parse_list(text: str) -> list[str]:
    """Parse a model answer into the list value of plan §9.1.

    Only the first line is read; the generation limit is short but the model can
    still append an explanation.
    """
    first = text.split("\n")[0]
    parts = [p.strip() for p in first.split(LIST_DELIMITER.strip())]
    return [p for p in parts if p]


def render_list(values: Sequence[str]) -> str:
    return LIST_DELIMITER.join(str(v).strip() for v in values)


def parse_number(span: str) -> float:
    """Operand parser: pull a single number out of a span.

    Handles the surface forms DROP passages use around numbers -- "45-yard",
    "50,000 men", "23 yards" -- by taking the first numeric token.  A span with
    no numeric token, or with more than one, is not a usable operand.
    """
    matches = _NUMBER_RE.findall(span.replace("−", "-"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one number in {span!r}, found {len(matches)}")
    return float(matches[0].replace(",", ""))


def parse_number_list(values: Sequence[str], node_id: int) -> list[float]:
    out = []
    for v in values:
        try:
            out.append(parse_number(v))
        except ValueError as exc:
            raise NodeRejection(node_id, "operand_parse", str(exc)) from exc
    return out


def is_numeric_span(span: str) -> bool:
    try:
        parse_number(span)
        return True
    except ValueError:
        return False


def normalize_span(span: str) -> str:
    """Normalization used for passage-membership and set-equality guards.

    This is the official DROP answer normalizer, reached through `get_metrics`'s
    own helper so that guard decisions and scoring agree.
    """
    from drop_eval import _normalize_answer

    return _normalize_answer(span)


# --------------------------------------------------------------------------
# node model
# --------------------------------------------------------------------------

@dataclass
class Node:
    node_id: int              # 1-based, matching BREAK's #N references
    op: str
    args: list[str]
    parents: list[int]
    owner: str                # "model" | "executor"
    fn: str | None
    value_type: str           # "list" | "number"
    text: str                 # the QDMR step text for this node

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "op": self.op, "text": self.text,
            "args": self.args, "parents": self.parents, "owner": self.owner,
            "fn": self.fn, "value_type": self.value_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(node_id=d["node_id"], op=d["op"], args=list(d["args"]),
                   parents=list(d["parents"]), owner=d["owner"], fn=d["fn"],
                   value_type=d["value_type"], text=d["text"])


class UnsupportedOperator(ValueError):
    """The DAG contains an operator outside the scope of plan §6."""

    def __init__(self, op: str):
        super().__init__(op)
        self.op = op


def parse_program(program: Sequence[str], decomposition: str) -> list[Node]:
    """Turn BREAK's `program` column into typed nodes.

    `decomposition` is the `;`-separated step text, used for the node `text`
    field that the verbalizer renders.  Its step count must match the program's.
    """
    steps = [s.strip() for s in decomposition.split(";")]
    if len(steps) != len(program):
        raise ValueError(f"decomposition has {len(steps)} steps, program has {len(program)}")

    nodes: list[Node] = []
    for i, (raw, text) in enumerate(zip(program, steps), start=1):
        m = _STEP_RE.match(raw.strip())
        if not m:
            raise ValueError(f"unparsable program step: {raw!r}")
        op, body = m.group(1), m.group(2)
        try:
            args = [str(a) for a in ast.literal_eval("[" + body + "]")]
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"unparsable arguments in {raw!r}") from exc
        if op not in SUPPORTED_OPS:
            raise UnsupportedOperator(op)

        parents = sorted({int(r) for a in args for r in _REF_RE.findall(a)})
        if any(p >= i for p in parents):
            raise ValueError(f"node {i} references a later node: {parents}")

        if op in MODEL_OWNED_OPS:
            owner, fn, value_type = "model", None, "list"
        else:
            owner, fn = "executor", args[0]
            table = AGGREGATE_FNS if op == "AGGREGATE" else ARITHMETIC_FNS
            if fn not in table:
                raise UnsupportedOperator(f"{op}:{fn}")
            value_type = "number"
        nodes.append(Node(node_id=i, op=op, args=args, parents=parents,
                          owner=owner, fn=fn, value_type=value_type, text=text))
    return nodes


def model_owned_count(nodes: Sequence[Node]) -> int:
    """The frontier key of plan §3.2.  Executor nodes are free (plan §10)."""
    return sum(1 for n in nodes if n.owner == "model")


def sink_type(nodes: Sequence[Node]) -> str:
    return nodes[-1].value_type


def executor_depth(nodes: Sequence[Node]) -> int:
    """Executor-owned nodes on the path from the sink back to a model node.

    The binning variable for the symbolic-tail analysis (plan §16.5).
    """
    depth = 0
    idx = {n.node_id: n for n in nodes}
    cur = nodes[-1]
    while cur.owner == "executor":
        depth += 1
        if not cur.parents:
            break
        cur = idx[cur.parents[0]]
    return depth


# --------------------------------------------------------------------------
# verbalization
# --------------------------------------------------------------------------

_LEADING_RETURN = re.compile(r"^\s*return\s+", re.I)


def normalize_step_text(text: str) -> str:
    """BREAK step text carries whitespace noise: `return yards  of   #2`."""
    return " ".join(text.split())


def verbalize(node: Node, parent_values: dict[int, list[str]]) -> str:
    """Deterministic verbalization of a model-owned step (plan §9.1).

    Two templates, keyed on operator, both chosen to avoid subject-verb
    agreement.  SELECT steps carry a bare noun phrase that may be plural
    ("people", "field goals") or a proper noun ("Adam Vinatieri"), and no
    single "What is/are X?" wording covers both:

        SELECT          "Find in the passage: {X}"
        PROJECT/FILTER  "What is {X}?"

    PROJECT and FILTER read as attribute lookups over an already-named thing,
    so the question form fits them.  `#k` slots are filled with the rendered
    parent lists, longest index first so that `#10` is not clobbered by `#1`.

    This template set is pinned: changing it invalidates every pseudo-label
    generated under the old one.
    """
    body = _LEADING_RETURN.sub("", normalize_step_text(node.text)).strip()
    for ref in sorted(parent_values, reverse=True):
        body = body.replace(f"#{ref}", render_list(parent_values[ref]))
    body = " ".join(body.split())
    if node.op == "SELECT":
        return f"Find in the passage: {body}"
    return f"What is {body}?"


def has_unfilled_refs(text: str) -> bool:
    return bool(_REF_RE.search(text))


# --------------------------------------------------------------------------
# executor
# --------------------------------------------------------------------------

COUNT_POLICIES = ("reject", "length", "value")


def _agg_count(values: list[str], node_id: int, policy: str = "reject") -> float:
    """`count` is overloaded in BREAK (plan Risk 6).

    "How many field goals did Hartley make?" wants the length of the parent
    list.  "How many men, horses, elephants combined?" annotates the same
    operator over a parent list holding one span such as `50,000`, where the
    intended answer is 50000 rather than 1.  The two cases are
    indistinguishable from DAG structure alone.

    The policy is a parameter rather than a fixed choice because the rejection
    rate it produces is large enough to decide pool sizes, and which reading is
    right is a Week 1 guard question:

        reject  refuse the node and count it (default; conservative)
        length  always len(list) -- the literal COUNT semantics
        value   read the single numeric element as the answer
    """
    if len(values) == 1 and is_numeric_span(values[0]):
        if policy == "reject":
            raise NodeRejection(node_id, "ambiguous_count",
                                f"single numeric element {values[0]!r}")
        if policy == "value":
            return parse_number(values[0])
    return float(len(values))


AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "sum": lambda xs: float(sum(xs)),
    "min": min,
    "max": max,
    "avg": lambda xs: sum(xs) / len(xs),
}


def _apply_aggregate(node: Node, values: list[str],
                     count_policy: str = "reject") -> float:
    if node.fn == "count":
        return _agg_count(values, node.node_id, count_policy)
    if not values:
        raise NodeRejection(node.node_id, "empty_operand", node.fn or "")
    numbers = parse_number_list(values, node.node_id)
    return AGGREGATORS[node.fn](numbers)


def _apply_arithmetic(node: Node, operands: list[float]) -> float:
    fn = node.fn
    if fn == "sum":
        return float(sum(operands))
    if len(operands) != 2:
        raise NodeRejection(node.node_id, "arity",
                            f"{fn} needs 2 operands, got {len(operands)}")
    a, b = operands
    if fn == "difference":
        return abs(a - b)          # DROP difference questions are unsigned
    if fn == "multiplication":
        return a * b
    if b == 0:
        raise NodeRejection(node.node_id, "division_by_zero", "")
    return a / b


def format_number(x: float) -> str:
    """Render an executor result the way DROP writes numbers."""
    if x == int(x):
        return str(int(x))
    return f"{x:.4f}".rstrip("0").rstrip(".")


RANGE_CHECKS: dict[str, Callable[[float], bool]] = {
    "count": lambda x: x >= 0 and x == int(x) and x <= 100,
    "sum": lambda x: abs(x) < 1e9,
    "min": lambda x: abs(x) < 1e9,
    "max": lambda x: abs(x) < 1e9,
    "avg": lambda x: abs(x) < 1e9,
    "difference": lambda x: abs(x) < 1e9,
    "division": lambda x: abs(x) < 1e9,
    "multiplication": lambda x: abs(x) < 1e9,
}


def execute_node(node: Node, parent_values: dict[int, list[str]],
                 range_check: bool = True,
                 count_policy: str = "reject") -> list[str]:
    """Evaluate one executor-owned node.  Raises NodeRejection on failure."""
    if node.owner != "executor":
        raise ValueError(f"node {node.node_id} is model-owned")

    operand_lists = [parent_values[p] for p in node.parents]
    if len(operand_lists) != len(node.parents):
        raise NodeRejection(node.node_id, "missing_parent", "")

    if node.op == "AGGREGATE":
        if len(operand_lists) != 1:
            raise NodeRejection(node.node_id, "arity",
                                f"AGGREGATE over {len(operand_lists)} parents")
        result = _apply_aggregate(node, operand_lists[0], count_policy)
    else:
        operands: list[float] = []
        for ref, values in zip(node.parents, operand_lists):
            if len(values) != 1:
                raise NodeRejection(node.node_id, "operand_not_scalar",
                                    f"#{ref} has {len(values)} elements")
            operands.extend(parse_number_list(values, node.node_id))
        result = _apply_arithmetic(node, operands)

    if range_check and not RANGE_CHECKS[node.fn](result):
        raise NodeRejection(node.node_id, "range", f"{node.fn} produced {result}")
    return [format_number(result)]


@dataclass
class Trace:
    """Full record of one DAG execution, kept for accepted and rejected alike."""

    values: dict[int, list[str]] = field(default_factory=dict)
    questions: dict[int, str] = field(default_factory=dict)
    executor_io: list[dict] = field(default_factory=list)
    rejection: dict | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection is None

    def to_dict(self) -> dict:
        return {"values": {str(k): v for k, v in self.values.items()},
                "questions": {str(k): v for k, v in self.questions.items()},
                "executor_io": self.executor_io,
                "rejection": self.rejection}


def execute_dag(nodes: Sequence[Node],
                ask: Callable[[Node, str], list[str]],
                range_check: bool = True,
                count_policy: str = "reject") -> tuple[list[str] | None, Trace]:
    """Run the composition algorithm of plan §10 over one DAG.

    `ask(node, question)` supplies the model answer for a model-owned node and
    returns a parsed list.  Nodes arrive in BREAK order, which is already
    topological: `parse_program` rejects any forward reference.

    Returns (sink value, trace).  The sink value is None when a node rejected.
    """
    trace = Trace()
    for node in nodes:
        parent_values = {p: trace.values[p] for p in node.parents
                         if p in trace.values}
        if len(parent_values) != len(node.parents):
            trace.rejection = {"node_id": node.node_id, "reason": "missing_parent",
                               "detail": ""}
            return None, trace
        try:
            if node.owner == "model":
                question = verbalize(node, parent_values)
                trace.questions[node.node_id] = question
                if has_unfilled_refs(question):
                    raise NodeRejection(node.node_id, "unfilled_reference", question)
                value = ask(node, question)
                if not value:
                    raise NodeRejection(node.node_id, "empty_answer", "")
            else:
                value = execute_node(node, parent_values,
                                     range_check=range_check,
                                     count_policy=count_policy)
                trace.executor_io.append({
                    "node_id": node.node_id, "op": node.op, "fn": node.fn,
                    "inputs": {str(p): parent_values[p] for p in node.parents},
                    "output": value,
                })
        except NodeRejection as rej:
            trace.rejection = {"node_id": rej.node_id, "reason": rej.reason,
                               "detail": rej.detail}
            return None, trace
        trace.values[node.node_id] = value
    return trace.values[nodes[-1].node_id], trace


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(predicted: Sequence[str] | str, gold: Sequence[str] | str) -> tuple[float, float]:
    """Official DROP (EM, F1).  Lists are passed through as multi-span answers."""
    pred = list(predicted) if not isinstance(predicted, str) else predicted
    ref = list(gold) if not isinstance(gold, str) else gold
    em, f1 = get_metrics(pred, ref)
    return float(em), float(f1)


def score_em(predicted: Sequence[str] | str, gold_variants: Iterable[Any]) -> bool:
    """EM against any accepted gold rendering of the answer."""
    return any(score(predicted, g)[0] == 1.0 for g in gold_variants)
