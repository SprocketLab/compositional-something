"""Deterministic BFCL composition, guard, and audit helpers.

The public candidate records produced here deliberately omit reference calls.
References are written to a separate oracle stream that is consumed only by
the explicit oracle baseline and post-hoc auditing.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from self.coding.atomic_data import (
    BFCL_SYSTEM_PROMPT,
    AtomicExample,
    canonical_json,
    stable_hash,
)
from self.coding.evaluation import evaluate_bfcl, parse_strict_json_array


TRAIN_TEMPLATE_IDS = ("also", "in_addition", "then", "independent_semicolon")
HELDOUT_TEMPLATE_IDS = ("numbered", "bullets", "ordinal")

# How much of the parent prompt a subproblem predictor sees.
#   component_schemas: only the schemas its own clauses need.  Decomposition
#     then removes the schema-selection problem as well as the call count, so a
#     component is easier than the corresponding slice of the parent task.
#   candidate_union: the same shuffled schema union the parent prompt shows, so
#     the component must still select the right function.  Clause-to-schema
#     provenance is retained for auditing but never shown to the model.
COMPONENT_CONTEXT_MODES = ("component_schemas", "candidate_union")
DEFAULT_COMPONENT_CONTEXT = "component_schemas"

TRAIN_STRING_VALUES: Dict[str, Tuple[str, ...]] = {
    "place": ("Rome", "Berlin", "Tokyo", "Toronto"),
    "country": ("Italy", "Germany", "Japan", "Canada"),
    "state": ("Texas", "Florida", "Washington", "Arizona"),
    "color": ("blue", "green", "black", "white"),
    "currency": ("EUR", "JPY", "GBP", "CAD"),
}
TEST_STRING_VALUES: Dict[str, Tuple[str, ...]] = {
    "place": ("Madrid", "Sydney", "Dublin", "Chicago"),
    "country": ("Spain", "Australia", "Ireland", "Brazil"),
    "state": ("Ohio", "Georgia", "Colorado", "Oregon"),
    "color": ("orange", "purple", "yellow", "brown"),
    "currency": ("AUD", "CHF", "SEK", "NZD"),
}

DEPENDENCY_PATTERNS = (
    re.compile(r"\bprevious result\b", re.IGNORECASE),
    re.compile(r"\bresult of (?:the )?(?:first|second|prior|above)\b", re.IGNORECASE),
    re.compile(r"\bthat output\b", re.IGNORECASE),
    re.compile(r"\busing (?:its|that) result\b", re.IGNORECASE),
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_terminal(text: str) -> str:
    return text.strip().rstrip(".!?; ")


def render_joined_request(questions: Sequence[str], template_id: str) -> str:
    """Render an arity-independent, deterministic request-joining template."""

    clauses = [_strip_terminal(question) for question in questions]
    if len(clauses) < 2:
        raise ValueError("A composed request requires at least two clauses")
    if template_id == "also":
        return ". Also, ".join(clauses) + "."
    if template_id == "in_addition":
        return ". In addition, ".join(clauses) + "."
    if template_id == "then":
        return ". Then, ".join(clauses) + "."
    if template_id == "independent_semicolon":
        return "Please handle each request independently: " + "; ".join(clauses) + "."
    if template_id == "numbered":
        requests = "\n".join(f"{index}. {clause}." for index, clause in enumerate(clauses, 1))
        return "Complete every numbered request. Calls may be returned in any order.\n\nRequests:\n" + requests
    if template_id == "bullets":
        requests = "\n".join(f"- {clause}." for clause in clauses)
        return "Complete every request below. Calls may be returned in any order.\n\n" + requests
    if template_id == "ordinal":
        labels: List[str] = []
        for index in range(len(clauses)):
            if index == 0:
                labels.append("First")
            elif index == len(clauses) - 1:
                labels.append("Finally")
            else:
                labels.append("Next")
        return " ".join(f"{label}, {clause}." for label, clause in zip(labels, clauses))
    raise ValueError(f"Unknown BFCL join template: {template_id!r}")


def _schema_union(examples: Sequence[AtomicExample]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for example in examples:
        for function in example.evaluator.get("functions", []):
            name = str(function["name"])
            if name in by_name and by_name[name] != function:
                raise ValueError(f"Incompatible schemas share function name {name!r}")
            by_name[name] = copy.deepcopy(function)
    return list(by_name.values())


def _ordered_schema_union(
    examples: Sequence[AtomicExample],
    *,
    seed: int,
    key: str,
) -> List[Dict[str, Any]]:
    """Union component schemas and order them independently of clause order.

    Listing schemas in clause order leaks a positional shortcut: a model can
    emit the k-th schema for the k-th clause without reading either.  The
    presentation order is therefore a deterministic function of the candidate
    identity and the function name only.
    """

    return sorted(
        _schema_union(examples),
        key=lambda function: stable_hash(seed, "schema-order", key, str(function["name"])),
    )


def _messages(question: str, functions: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    user = f"User request:\n{question}\n\nAvailable functions:\n" + canonical_json(list(functions))
    return [
        {"role": "system", "content": BFCL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _atomic_spec(example: AtomicExample, *, component_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "component_id": component_id or example.source_id,
        "source_component_ids": list(example.source_component_ids or (example.source_id,)),
        "question": str(example.metadata["question"]),
        "clause_questions": [str(example.metadata["question"])],
        "functions": copy.deepcopy(example.evaluator["functions"]),
        "messages": [dict(message) for message in example.messages],
        "expected_call_count": int(example.component_count),
        "allow_exact_duplicates": False,
        "metadata": {
            key: copy.deepcopy(example.metadata[key])
            for key in ("complexity", "synthetic_mutation")
            if key in example.metadata
        },
    }


def _oracle_for_atomic(example: AtomicExample, *, component_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "component_id": component_id or example.source_id,
        "canonical_calls": json.loads(example.target),
        "accepted_calls": copy.deepcopy(example.evaluator["accepted_calls"]),
    }


def _candidate_id(round_index: int, family: str, component_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(component_ids).encode("utf-8")).hexdigest()[:16]
    return f"r{round_index}-{family}-{digest}"


def _ordered_examples(
    examples: Sequence[AtomicExample],
    *,
    seed: int,
    key: str,
) -> List[AtomicExample]:
    return sorted(examples, key=lambda item: stable_hash(seed, key, item.source_id))


def _spec_clauses(spec: Mapping[str, Any]) -> List[str]:
    """Return a component's leaf clauses in the order its calls are emitted."""

    clauses = spec.get("clause_questions")
    if clauses:
        return [str(clause) for clause in clauses]
    return [str(spec["question"])]


def _apply_candidate_union_context(
    ordered_specs: Sequence[Mapping[str, Any]],
    ordered_oracles: Sequence[Mapping[str, Any]],
    *,
    functions: Sequence[Mapping[str, Any]],
    candidate_id: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Show every subproblem the parent's full schema union.

    The component prompt then depends on its parent, so component IDs are
    re-keyed by parent: two parents sharing the same sources are genuinely
    different prompts and must be generated separately.
    """

    specs: List[Dict[str, Any]] = []
    oracles: List[Dict[str, Any]] = []
    for spec, oracle in zip(ordered_specs, ordered_oracles):
        base_id = str(spec["component_id"])
        component_id = "ctx-" + hashlib.sha256(
            "\x1f".join([candidate_id, base_id]).encode("utf-8")
        ).hexdigest()[:16]
        context_functions = sorted(
            copy.deepcopy(list(functions)),
            key=lambda function: stable_hash(
                seed, "component-schema-order", component_id, str(function["name"])
            ),
        )
        question = str(spec["question"])
        specs.append(
            {
                **copy.deepcopy(dict(spec)),
                "component_id": component_id,
                "base_component_id": base_id,
                # Audit-only provenance; never part of the rendered prompt.
                "relevant_function_names": [
                    str(function["name"]) for function in spec["functions"]
                ],
                "functions": context_functions,
                "messages": _messages(question, context_functions),
                "component_context": "candidate_union",
            }
        )
        oracles.append({**copy.deepcopy(dict(oracle)), "component_id": component_id})
    return specs, oracles


def _make_candidate(
    *,
    round_index: int,
    split: str,
    family: str,
    atomic_examples: Sequence[AtomicExample],
    component_specs: Sequence[Mapping[str, Any]],
    component_oracles: Sequence[Mapping[str, Any]],
    template_id: str,
    template_partition: str,
    seed: int,
    candidate_id: Optional[str] = None,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if component_context not in COMPONENT_CONTEXT_MODES:
        raise ValueError(f"Unknown component context mode: {component_context!r}")
    order_key = candidate_id or family
    ordered_specs = [
        copy.deepcopy(dict(spec))
        for spec in sorted(
            component_specs,
            key=lambda spec: stable_hash(seed, order_key, str(spec["component_id"])),
        )
    ]
    oracle_by_component = {
        str(oracle["component_id"]): copy.deepcopy(dict(oracle))
        for oracle in component_oracles
    }
    if len(oracle_by_component) != len(component_oracles):
        raise ValueError("Component oracles must carry unique component IDs")
    ordered_oracles = [
        oracle_by_component[str(spec["component_id"])] for spec in ordered_specs
    ]
    # Input composition must use the same permutation as output composition:
    # the parent clause list is the concatenation of the component clause
    # lists, so clause k is always answered by call k.
    questions = [
        clause for spec in ordered_specs for clause in _spec_clauses(spec)
    ]
    ids = [
        str(source_id)
        for spec in ordered_specs
        for source_id in spec["source_component_ids"]
    ]
    canonical_calls = [
        call for oracle in ordered_oracles for call in oracle["canonical_calls"]
    ]
    accepted_calls = [
        call for oracle in ordered_oracles for call in oracle["accepted_calls"]
    ]
    if not len(questions) == len(ids) == len(canonical_calls) == len(accepted_calls):
        raise ValueError(
            "Clause, source, and call counts must align for a composed candidate: "
            f"{len(questions)} clauses, {len(ids)} sources, "
            f"{len(canonical_calls)} canonical calls, {len(accepted_calls)} accepted calls"
        )
    joined = render_joined_request(questions, template_id)
    resolved_id = candidate_id or _candidate_id(round_index, family, ids)
    functions = _ordered_schema_union(atomic_examples, seed=seed, key=resolved_id)
    if component_context == "candidate_union":
        ordered_specs, ordered_oracles = _apply_candidate_union_context(
            ordered_specs,
            ordered_oracles,
            functions=functions,
            candidate_id=resolved_id,
            seed=seed,
        )
    public = {
        "candidate_id": resolved_id,
        "round": int(round_index),
        "split": split,
        "composition_family": family,
        "component_count": len(canonical_calls),
        "source_component_ids": ids,
        "source_group_id": resolved_id,
        "question": joined,
        "clause_questions": questions,
        "functions": functions,
        "schema_order": [str(function["name"]) for function in functions],
        "messages": _messages(joined, functions),
        "template_id": template_id,
        "template_partition": template_partition,
        "component_specs": ordered_specs,
        "independent": not any(
            pattern.search(question) for pattern in DEPENDENCY_PATTERNS for question in questions
        ),
    }
    oracle = {
        "candidate_id": resolved_id,
        "canonical_calls": canonical_calls,
        "accepted_calls": accepted_calls,
        "component_oracles": ordered_oracles,
    }
    return public, oracle


def build_round1_cross_candidates(
    hidden_examples: Sequence[AtomicExample],
    *,
    seed: int,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Construct every compatible pair from the hidden atomic pool."""

    ranked_pairs = sorted(
        itertools.combinations(hidden_examples, 2),
        key=lambda pair: stable_hash(seed, "r1-pair", *(item.source_id for item in pair)),
    )
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for index, pair in enumerate(ranked_pairs):
        template_id = TRAIN_TEMPLATE_IDS[index % len(TRAIN_TEMPLATE_IDS)]
        specs = [_atomic_spec(item) for item in pair]
        oracles = [_oracle_for_atomic(item) for item in pair]
        public, oracle = _make_candidate(
            round_index=1,
            split="hidden_composition",
            family="cross_function",
            atomic_examples=pair,
            component_specs=specs,
            component_oracles=oracles,
            template_id=template_id,
            template_partition="train",
            seed=seed,
            component_context=component_context,
        )
        public_rows.append(public)
        oracle_rows.append(oracle)
    return public_rows, oracle_rows


def _smallest_combinations(
    examples: Sequence[AtomicExample],
    *,
    count: int,
    arity: int,
    seed: int,
    key: str,
) -> List[Tuple[AtomicExample, ...]]:
    if arity > len(examples):
        raise ValueError("Not enough examples for requested composition arity")
    total = math.comb(len(examples), arity)
    if count > total:
        raise ValueError(f"Requested {count} groups but only {total} combinations exist")
    if total > 1_000_000:
        source_pool = sorted(examples, key=lambda item: item.source_id)
        output: List[Tuple[AtomicExample, ...]] = []
        seen: set[Tuple[str, ...]] = set()
        attempts = 0
        while len(output) < count and attempts < count * 100:
            selected = sorted(
                source_pool,
                key=lambda item: stable_hash(seed, key, attempts, item.source_id),
            )[:arity]
            attempts += 1
            signature = tuple(sorted(item.source_id for item in selected))
            if signature in seen:
                continue
            seen.add(signature)
            output.append(tuple(selected))
        if len(output) != count:
            raise ValueError(f"Could construct only {len(output)} of {count} requested groups")
        return output
    return heapq.nsmallest(
        count,
        itertools.combinations(examples, arity),
        key=lambda group: stable_hash(seed, key, *(item.source_id for item in group)),
    )


def _pair_subproblem(
    pair: Sequence[AtomicExample],
    *,
    seed: int,
    key: str,
    template_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ordered = _ordered_examples(pair, seed=seed, key=key)
    clauses = [str(item.metadata["question"]) for item in ordered]
    question = render_joined_request(clauses, template_id)
    component_id = "sub-" + hashlib.sha256(
        "\x1f".join(
            [template_id, *(item.source_id for item in ordered)]
        ).encode("utf-8")
    ).hexdigest()[:16]
    functions = _ordered_schema_union(ordered, seed=seed, key=component_id)
    spec = {
        "component_id": component_id,
        "source_component_ids": [item.source_id for item in ordered],
        "question": question,
        "clause_questions": clauses,
        "functions": functions,
        "messages": _messages(question, functions),
        "expected_call_count": 2,
        "allow_exact_duplicates": False,
    }
    oracle = {
        "component_id": component_id,
        "canonical_calls": [call for item in ordered for call in json.loads(item.target)],
        "accepted_calls": [
            call for item in ordered for call in item.evaluator["accepted_calls"]
        ],
    }
    return spec, oracle


def build_round2_cross_candidates(
    hidden_examples: Sequence[AtomicExample],
    *,
    seed: int,
    count: int = 2000,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups = _smallest_combinations(
        hidden_examples,
        count=count,
        arity=4,
        seed=seed,
        key="r2-quad",
    )
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for index, group in enumerate(groups):
        ordered = _ordered_examples(group, seed=seed, key=f"r2-order-{index}")
        specs: List[Dict[str, Any]] = []
        oracles: List[Dict[str, Any]] = []
        for pair_index, pair in enumerate((ordered[:2], ordered[2:])):
            template = TRAIN_TEMPLATE_IDS[(index + pair_index) % len(TRAIN_TEMPLATE_IDS)]
            spec, oracle = _pair_subproblem(
                pair,
                seed=seed,
                key=f"r2-{index}-sub-{pair_index}",
                template_id=template,
            )
            specs.append(spec)
            oracles.append(oracle)
        template_id = TRAIN_TEMPLATE_IDS[index % len(TRAIN_TEMPLATE_IDS)]
        candidate_id = _candidate_id(2, "cross_function", [item.source_id for item in ordered])
        public, oracle = _make_candidate(
            round_index=2,
            split="hidden_composition",
            family="cross_function",
            atomic_examples=ordered,
            component_specs=specs,
            component_oracles=oracles,
            template_id=template_id,
            template_partition="train",
            seed=seed,
            candidate_id=candidate_id,
            component_context=component_context,
        )
        public_rows.append(public)
        oracle_rows.append(oracle)
    return public_rows, oracle_rows


def _hierarchical_subproblem(
    examples: Sequence[AtomicExample],
    *,
    seed: int,
    key: str,
    template_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Render a public multi-call component and keep its oracle separately."""

    ordered = _ordered_examples(examples, seed=seed, key=key)
    clauses = [str(item.metadata["question"]) for item in ordered]
    question = render_joined_request(clauses, template_id)
    component_id = "sub-" + hashlib.sha256(
        "\x1f".join(
            [str(len(ordered)), template_id, *(item.source_id for item in ordered)]
        ).encode("utf-8")
    ).hexdigest()[:16]
    functions = _ordered_schema_union(ordered, seed=seed, key=component_id)
    return (
        {
            "component_id": component_id,
            "source_component_ids": [item.source_id for item in ordered],
            "question": question,
            "clause_questions": clauses,
            "functions": functions,
            "messages": _messages(question, functions),
            "expected_call_count": len(ordered),
            "allow_exact_duplicates": False,
        },
        {
            "component_id": component_id,
            "canonical_calls": [
                call for item in ordered for call in json.loads(item.target)
            ],
            "accepted_calls": [
                call for item in ordered for call in item.evaluator["accepted_calls"]
            ],
        },
    )


def build_hierarchical_cross_candidates(
    examples: Sequence[AtomicExample],
    *,
    component_count: int,
    count: int,
    seed: int,
    round_index: Optional[int] = None,
    split: str = "hidden_composition",
    template_partition: str = "train",
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build 2/4/8-call candidates from disjoint atomic source groups.

    Each candidate is presented as two equal-size public subproblems.  This is
    the binary curriculum interface used by compositional pseudo-labeling; the
    hidden component calls are returned only in the parallel oracle records.
    """

    if component_count not in {2, 4, 8}:
        raise ValueError("component_count must be one of 2, 4, or 8")
    resolved_round = round_index or int(math.log2(component_count))
    total_groups = math.comb(len(examples), component_count)
    groups = _smallest_combinations(
        examples,
        count=min(total_groups, count * 3),
        arity=component_count,
        seed=seed,
        key=f"hierarchical-{component_count}",
    )
    template_ids = (
        TRAIN_TEMPLATE_IDS if template_partition == "train" else HELDOUT_TEMPLATE_IDS
    )
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    half = component_count // 2
    rejected_incompatible = 0
    for group_index, group in enumerate(groups):
        index = len(public_rows)
        ordered = _ordered_examples(
            group,
            seed=seed,
            key=f"hierarchical-order-{component_count}-{group_index}",
        )
        try:
            _schema_union(ordered)
        except ValueError as exc:
            if "Incompatible schemas share function name" not in str(exc):
                raise
            rejected_incompatible += 1
            continue
        specs: List[Dict[str, Any]] = []
        oracles: List[Dict[str, Any]] = []
        for side, subgroup in enumerate((ordered[:half], ordered[half:])):
            if half == 1:
                specs.append(_atomic_spec(subgroup[0]))
                oracles.append(_oracle_for_atomic(subgroup[0]))
            else:
                template_id = template_ids[(index + side) % len(template_ids)]
                spec, oracle = _hierarchical_subproblem(
                    subgroup,
                    seed=seed,
                    key=f"hierarchical-{component_count}-{index}-side-{side}",
                    template_id=template_id,
                )
                specs.append(spec)
                oracles.append(oracle)
        template_id = template_ids[index % len(template_ids)]
        candidate_id = _candidate_id(
            resolved_round,
            "cross_function",
            [item.source_id for item in ordered],
        )
        try:
            public, oracle = _make_candidate(
                round_index=resolved_round,
                split=split,
                family="cross_function",
                atomic_examples=ordered,
                component_specs=specs,
                component_oracles=oracles,
                template_id=template_id,
                template_partition=template_partition,
                seed=seed,
                candidate_id=candidate_id,
                component_context=component_context,
            )
        except ValueError as exc:
            if "Incompatible schemas share function name" not in str(exc):
                raise
            rejected_incompatible += 1
            continue
        public_rows.append(public)
        oracle_rows.append(oracle)
        if len(public_rows) == count:
            break
    if len(public_rows) != count:
        raise ValueError(
            f"Could construct only {len(public_rows)} of {count} compatible "
            f"{component_count}-call groups; rejected {rejected_incompatible}"
        )
    return public_rows, oracle_rows


def _literal_matches(question: str, value: Any) -> List[re.Match[str]]:
    rendered = str(value)
    if isinstance(value, bool) or value is None or not rendered:
        return []
    if isinstance(value, (int, float)):
        pattern = re.compile(rf"(?<![\w.]){re.escape(rendered)}(?![\w.])", re.IGNORECASE)
    elif isinstance(value, str) and len(value) >= 2:
        pattern = re.compile(re.escape(value), re.IGNORECASE)
    else:
        return []
    return list(pattern.finditer(question))


def _string_class(argument_name: str) -> Optional[str]:
    name = argument_name.lower()
    if "country" in name:
        return "country"
    if "state" in name:
        return "state"
    if "color" in name or "colour" in name:
        return "color"
    if "currency" in name:
        return "currency"
    if any(token in name for token in ("city", "location", "origin", "destination")):
        return "place"
    return None


def _numeric_replacements(value: Any, schema: Mapping[str, Any], *, split: str) -> List[Any]:
    offsets = (1, 2, -1, -2) if split == "hidden_composition" else (3, -3, 4, -4)
    minimum = schema.get("minimum", -math.inf)
    maximum = schema.get("maximum", math.inf)
    exclusive_minimum = schema.get("exclusiveMinimum")
    exclusive_maximum = schema.get("exclusiveMaximum")
    multiple = schema.get("multipleOf")
    output: List[Any] = []
    for offset in offsets:
        candidate: Any = value + offset
        if schema.get("type") == "integer":
            candidate = int(candidate)
        if candidate < minimum or candidate > maximum:
            continue
        if exclusive_minimum is not None and candidate <= exclusive_minimum:
            continue
        if exclusive_maximum is not None and candidate >= exclusive_maximum:
            continue
        if multiple and not math.isclose(float(candidate) / float(multiple), round(float(candidate) / float(multiple))):
            continue
        if candidate != value and candidate not in output:
            output.append(candidate)
    return output


def _replacement_values(
    argument_name: str,
    value: Any,
    schema: Mapping[str, Any],
    *,
    split: str,
) -> List[Any]:
    enum = schema.get("enum")
    if isinstance(enum, list) and len(enum) > 1:
        return [item for item in enum if item != value]
    expected = schema.get("type")
    if expected in ("integer", "number", "float") and isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_replacements(value, schema, split=split)
    if expected == "string" and isinstance(value, str):
        value_class = _string_class(argument_name)
        registry = TRAIN_STRING_VALUES if split == "hidden_composition" else TEST_STRING_VALUES
        if value_class:
            return [candidate for candidate in registry[value_class] if candidate.lower() != value.lower()]
    return []


def build_mutated_atomic_variants(
    example: AtomicExample,
    *,
    split: str,
    max_variants: int,
    seed: int,
) -> List[AtomicExample]:
    """Create conservative scalar substitutions with exact hidden references."""

    if split not in {"hidden_composition", "test"}:
        raise ValueError("Synthetic mutations are restricted to hidden_composition or test")
    canonical_calls = json.loads(example.target)
    if len(canonical_calls) != 1:
        return []
    call = canonical_calls[0]
    functions = example.evaluator.get("functions", [])
    if len(functions) != 1:
        return []
    properties = functions[0].get("parameters", {}).get("properties", {})
    question = str(example.metadata["question"])
    candidates: List[Tuple[str, Any, Any, re.Match[str]]] = []
    for argument_name, value in call.get("arguments", {}).items():
        schema = properties.get(argument_name, {})
        matches = _literal_matches(question, value)
        if len(matches) != 1:
            continue
        for replacement in _replacement_values(
            argument_name,
            value,
            schema,
            split=split,
        ):
            candidates.append((argument_name, value, replacement, matches[0]))
    candidates.sort(
        key=lambda item: stable_hash(seed, split, example.source_id, item[0], item[2])
    )
    variants: List[AtomicExample] = []
    for variant_index, (argument_name, old_value, replacement, match) in enumerate(candidates[:max_variants]):
        mutated_question = question[: match.start()] + str(replacement) + question[match.end() :]
        mutated_calls = copy.deepcopy(canonical_calls)
        mutated_calls[0]["arguments"][argument_name] = replacement
        accepted_calls = copy.deepcopy(example.evaluator["accepted_calls"])
        accepted_calls[0]["arguments"][argument_name] = [replacement]
        variant_id = f"{example.source_id}-mut-{variant_index}-{stable_hash(replacement) & 0xffff:04x}"
        variants.append(
            AtomicExample(
                task="bfcl",
                source_id=variant_id,
                source_group_id=example.source_id,
                split=split,
                messages=tuple(_messages(mutated_question, functions)),
                target=canonical_json(mutated_calls),
                evaluator={"accepted_calls": accepted_calls, "functions": copy.deepcopy(functions)},
                component_count=1,
                source_component_ids=(variant_id,),
                evaluation_track="synthetic_repeat_component",
                metadata={
                    **copy.deepcopy(example.metadata),
                    "question": mutated_question,
                    "synthetic_mutation": {
                        "source_id": example.source_id,
                        "argument": argument_name,
                        "old_value": old_value,
                        "new_value": replacement,
                        "split_registry": split,
                    },
                },
            )
        )
    return variants


def build_round1_repeat_candidates(
    examples: Sequence[AtomicExample],
    *,
    split: str,
    seed: int,
    max_variants_per_source: int,
    template_partition: str,
    renders_per_pair: int = 1,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    template_ids = TRAIN_TEMPLATE_IDS if template_partition == "train" else HELDOUT_TEMPLATE_IDS
    rows: List[Tuple[AtomicExample, AtomicExample]] = []
    source_counter: Counter[str] = Counter()
    for example in examples:
        variants = build_mutated_atomic_variants(
            example,
            split=split,
            max_variants=max_variants_per_source,
            seed=seed,
        )
        for variant in variants:
            rows.append((example, variant))
            source_counter[example.source_id] += 1
    rows.sort(key=lambda pair: stable_hash(seed, "repeat", pair[0].source_id, pair[1].source_id))
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    if renders_per_pair < 1:
        raise ValueError("renders_per_pair must be positive")
    for index, pair in enumerate(rows):
        base_id = _candidate_id(1, "synthetic_repeat", [item.source_id for item in pair])
        for render_index in range(renders_per_pair):
            template_id = template_ids[(index + render_index) % len(template_ids)]
            candidate_id = f"{base_id}-{template_id}-v{render_index}"
            public, oracle = _make_candidate(
                round_index=1,
                split=split,
                family="synthetic_repeat",
                atomic_examples=pair,
                component_specs=[_atomic_spec(item) for item in pair],
                component_oracles=[_oracle_for_atomic(item) for item in pair],
                template_id=template_id,
                template_partition=template_partition,
                seed=seed,
                candidate_id=candidate_id,
                component_context=component_context,
            )
            public_rows.append(public)
            oracle_rows.append(oracle)
    audit = {
        "candidate_count": len(public_rows),
        "semantic_pair_count": len(rows),
        "qualifying_source_count": len(source_counter),
        "variants_per_source": dict(sorted(source_counter.items())),
    }
    return public_rows, oracle_rows, audit


def build_round2_repeat_candidates(
    round1_public: Sequence[Mapping[str, Any]],
    round1_oracle: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    count: int = 500,
    split: str = "hidden_composition",
    template_partition: str = "train",
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    oracle_by_id = {str(row["candidate_id"]): row for row in round1_oracle}
    compatible_pairs = (
        pair
        for pair in itertools.combinations(round1_public, 2)
        if set(pair[0]["source_component_ids"]).isdisjoint(pair[1]["source_component_ids"])
        and pair[0]["functions"][0]["name"] != pair[1]["functions"][0]["name"]
    )
    selected = heapq.nsmallest(
        count,
        compatible_pairs,
        key=lambda pair: stable_hash(seed, "repeat-r2", pair[0]["candidate_id"], pair[1]["candidate_id"]),
    )
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for index, pair in enumerate(selected):
        atomic_specs = [spec for row in pair for spec in row["component_specs"]]
        atomic_examples: List[AtomicExample] = []
        for spec in atomic_specs:
            atomic_examples.append(public_spec_to_example(spec, split="hidden_composition"))
        component_specs: List[Dict[str, Any]] = []
        component_oracles: List[Dict[str, Any]] = []
        for row in pair:
            component_specs.append(
                {
                    "component_id": row["candidate_id"],
                    "source_component_ids": list(row["source_component_ids"]),
                    "question": row["question"],
                    "clause_questions": _spec_clauses(row),
                    "functions": copy.deepcopy(row["functions"]),
                    "messages": copy.deepcopy(row["messages"]),
                    "expected_call_count": 2,
                    "allow_exact_duplicates": False,
                }
            )
            oracle_row = oracle_by_id[str(row["candidate_id"])]
            component_oracles.append(
                {
                    "component_id": row["candidate_id"],
                    "canonical_calls": copy.deepcopy(oracle_row["canonical_calls"]),
                    "accepted_calls": copy.deepcopy(oracle_row["accepted_calls"]),
                }
            )
        template_ids = TRAIN_TEMPLATE_IDS if template_partition == "train" else HELDOUT_TEMPLATE_IDS
        template_id = template_ids[index % len(template_ids)]
        candidate_id = _candidate_id(
            2, "paired_repeat", [str(row["candidate_id"]) for row in pair]
        )
        public, oracle = _make_candidate(
            round_index=2,
            split=split,
            family="paired_repeat",
            atomic_examples=atomic_examples,
            component_specs=component_specs,
            component_oracles=component_oracles,
            template_id=template_id,
            template_partition=template_partition,
            seed=seed,
            candidate_id=candidate_id,
            component_context=component_context,
        )
        public_rows.append(public)
        oracle_rows.append(oracle)
    return public_rows, oracle_rows


def build_next_repeat_candidates(
    previous_public: Sequence[Mapping[str, Any]],
    previous_oracle: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
    component_call_count: int,
    seed: int,
    count: int,
    split: str = "hidden_composition",
    template_partition: str = "train",
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pair disjoint rendered components to form the next repeat frontier."""

    if component_call_count < 1:
        raise ValueError("component_call_count must be positive")
    oracle_by_id = {str(row["candidate_id"]): row for row in previous_oracle}
    compatible_pairs = (
        pair
        for pair in itertools.combinations(previous_public, 2)
        if set(pair[0]["source_component_ids"]).isdisjoint(
            pair[1]["source_component_ids"]
        )
    )
    selected = heapq.nsmallest(
        count * 3,
        compatible_pairs,
        key=lambda pair: stable_hash(
            seed,
            f"repeat-r{round_index}",
            pair[0]["candidate_id"],
            pair[1]["candidate_id"],
        ),
    )
    if len(selected) < count:
        raise ValueError(
            f"Could identify only {len(selected)} candidate pairs for {count} repeats"
        )
    template_ids = (
        TRAIN_TEMPLATE_IDS if template_partition == "train" else HELDOUT_TEMPLATE_IDS
    )
    public_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    rejected_incompatible = 0
    for pair_index, pair in enumerate(selected):
        index = len(public_rows)
        atomic_specs = [spec for row in pair for spec in row["component_specs"]]
        atomic_examples = [
            public_spec_to_example(spec, split=split) for spec in atomic_specs
        ]
        try:
            _schema_union(atomic_examples)
        except ValueError as exc:
            if "Incompatible schemas share function name" not in str(exc):
                raise
            rejected_incompatible += 1
            continue
        component_specs: List[Dict[str, Any]] = []
        component_oracles: List[Dict[str, Any]] = []
        for row in pair:
            candidate_id = str(row["candidate_id"])
            component_specs.append(
                {
                    "component_id": candidate_id,
                    "source_component_ids": list(row["source_component_ids"]),
                    "question": row["question"],
                    "clause_questions": _spec_clauses(row),
                    "functions": copy.deepcopy(row["functions"]),
                    "messages": copy.deepcopy(row["messages"]),
                    "expected_call_count": component_call_count,
                    "allow_exact_duplicates": False,
                }
            )
            oracle_row = oracle_by_id[candidate_id]
            component_oracles.append(
                {
                    "component_id": candidate_id,
                    "canonical_calls": copy.deepcopy(oracle_row["canonical_calls"]),
                    "accepted_calls": copy.deepcopy(oracle_row["accepted_calls"]),
                }
            )
        leaf_ids = [str(value) for row in pair for value in row["source_component_ids"]]
        # Two renders of one semantic pair share their leaf sources, so the ID
        # must come from the rendered components or it collides.
        candidate_id = _candidate_id(
            round_index, "paired_repeat", [str(row["candidate_id"]) for row in pair]
        )
        public, oracle = _make_candidate(
            round_index=round_index,
            split=split,
            family="paired_repeat",
            atomic_examples=atomic_examples,
            component_specs=component_specs,
            component_oracles=component_oracles,
            template_id=template_ids[index % len(template_ids)],
            template_partition=template_partition,
            seed=seed,
            candidate_id=candidate_id,
            component_context=component_context,
        )
        # ``_make_candidate`` already records leaf provenance in clause order;
        # the two orderings must agree or clause k would not answer call k.
        if sorted(public["source_component_ids"]) != sorted(leaf_ids):
            raise AssertionError(
                f"Leaf provenance diverged for repeat candidate {candidate_id}"
            )
        public_rows.append(public)
        oracle_rows.append(oracle)
        if len(public_rows) == count:
            break
    if len(public_rows) != count:
        raise ValueError(
            f"Could construct only {len(public_rows)} of {count} compatible repeat "
            f"candidates; rejected {rejected_incompatible}"
        )
    return public_rows, oracle_rows


def public_spec_to_example(spec: Mapping[str, Any], *, split: str = "inference") -> AtomicExample:
    return AtomicExample(
        task="bfcl",
        source_id=str(spec["component_id"]),
        source_group_id=str(spec["component_id"]),
        split=split,
        messages=tuple(copy.deepcopy(spec["messages"])),
        target="[]",
        evaluator={"functions": copy.deepcopy(spec["functions"]), "accepted_calls": []},
        component_count=int(spec["expected_call_count"]),
        source_component_ids=tuple(spec["source_component_ids"]),
        evaluation_track="public_component",
        metadata={"question": spec["question"], **copy.deepcopy(spec.get("metadata", {}))},
    )


def public_candidate_to_example(
    candidate: Mapping[str, Any],
    *,
    target: str = "[]",
    evaluator: Optional[Mapping[str, Any]] = None,
) -> AtomicExample:
    return AtomicExample(
        task="bfcl",
        source_id=str(candidate["candidate_id"]),
        source_group_id=str(candidate.get("source_group_id", candidate["candidate_id"])),
        split=str(candidate.get("split", "inference")),
        messages=tuple(copy.deepcopy(candidate["messages"])),
        target=target,
        evaluator=copy.deepcopy(dict(evaluator or {"functions": candidate["functions"], "accepted_calls": []})),
        component_count=int(candidate["component_count"]),
        source_component_ids=tuple(candidate["source_component_ids"]),
        evaluation_track=str(candidate.get("template_partition", "composed")),
        metadata={
            "question": candidate["question"],
            "join_template": candidate["template_id"],
            "composition_family": candidate["composition_family"],
        },
    )


def oracle_example(candidate: Mapping[str, Any], oracle: Mapping[str, Any]) -> AtomicExample:
    return public_candidate_to_example(
        candidate,
        target=canonical_json(oracle["canonical_calls"]),
        evaluator={
            "functions": copy.deepcopy(candidate["functions"]),
            "accepted_calls": copy.deepcopy(oracle["accepted_calls"]),
        },
    )


def _value_matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_value_matches_schema(value, {**schema, "type": member}) for member in expected)
    aliases = {"dict": "object", "list": "array", "tuple": "array", "float": "number"}
    expected = aliases.get(str(expected), expected)
    if expected in (None, "any"):
        return True
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list) and (
            "items" not in schema or all(_value_matches_schema(item, schema["items"]) for item in value)
        )
    if expected == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if any(key not in value for key in required):
            return False
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties and not _value_matches_schema(item, properties[key]):
                return False
            if key not in properties and additional is False:
                return False
            if key not in properties and isinstance(additional, dict) and not _value_matches_schema(item, additional):
                return False
        return True
    return True


def _call_shape_valid(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call.get("name"), str)
        and isinstance(call.get("arguments"), dict)
    )


def _call_schema_errors(call: Mapping[str, Any], functions: Mapping[str, Mapping[str, Any]]) -> List[str]:
    name = str(call["name"])
    if name not in functions:
        return ["function_not_available"]
    parameters = functions[name].get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    arguments = call["arguments"]
    errors: List[str] = []
    if any(key not in arguments for key in required):
        errors.append("missing_required_argument")
    additional = parameters.get("additionalProperties", False)
    if any(key not in properties for key in arguments) and additional is not True and not isinstance(additional, dict):
        errors.append("unknown_argument")
    for key, value in arguments.items():
        if key in properties and not _value_matches_schema(value, properties[key]):
            errors.append("argument_type_mismatch")
        elif key not in properties and isinstance(additional, dict) and not _value_matches_schema(value, additional):
            errors.append("argument_type_mismatch")
    return errors


def guard_prediction(
    raw_prediction: str,
    spec: Mapping[str, Any],
    *,
    level: str,
) -> Dict[str, Any]:
    if level not in {"g1", "g4"}:
        raise ValueError(f"Unsupported guard level: {level}")
    parsed, parse_error = parse_strict_json_array(raw_prediction)
    reasons: List[str] = []
    if parsed is None:
        reasons.append("invalid_json_array")
        return {
            "accepted": False,
            "guard_level": level,
            "reasons": reasons,
            "parse_error": parse_error,
            "parsed_calls": None,
        }
    if not all(_call_shape_valid(call) for call in parsed):
        reasons.append("invalid_call_shape")
    if level == "g4" and not reasons:
        if len(parsed) != int(spec["expected_call_count"]):
            reasons.append("wrong_call_count")
        functions = {str(function["name"]): function for function in spec["functions"]}
        for call in parsed:
            reasons.extend(_call_schema_errors(call, functions))
        canonical_calls = [canonical_json(call) for call in parsed]
        if len(canonical_calls) != len(set(canonical_calls)) and not spec.get("allow_exact_duplicates", False):
            reasons.append("duplicate_call")
    return {
        "accepted": not reasons,
        "guard_level": level,
        "reasons": sorted(set(reasons)),
        "parse_error": None,
        "parsed_calls": parsed,
    }


def compose_component_predictions(
    candidate: Mapping[str, Any],
    raw_by_component: Mapping[str, str],
    *,
    level: str,
) -> Dict[str, Any]:
    component_decisions: List[Dict[str, Any]] = []
    composed_calls: List[Dict[str, Any]] = []
    reasons: List[str] = []
    for spec in candidate["component_specs"]:
        component_id = str(spec["component_id"])
        if component_id not in raw_by_component:
            decision = {
                "accepted": False,
                "guard_level": level,
                "reasons": ["missing_component_prediction"],
                "parse_error": None,
                "parsed_calls": None,
            }
        else:
            decision = guard_prediction(raw_by_component[component_id], spec, level=level)
        component_decisions.append({"component_id": component_id, **decision})
        if decision["accepted"]:
            composed_calls.extend(decision["parsed_calls"])
        else:
            reasons.extend(f"{component_id}:{reason}" for reason in decision["reasons"])
    if level == "g4" and not reasons:
        if len(composed_calls) != int(candidate["component_count"]):
            reasons.append("composed_wrong_call_count")
        canonical_calls = [canonical_json(call) for call in composed_calls]
        if len(canonical_calls) != len(set(canonical_calls)):
            reasons.append("composed_duplicate_call")
        if not candidate.get("independent", False):
            reasons.append("dependency_detected")
        try:
            _schema_union(
                [
                    public_spec_to_example(spec)
                    for spec in candidate["component_specs"]
                ]
            )
        except ValueError:
            reasons.append("schema_collision")
    accepted = not reasons
    return {
        "candidate_id": candidate["candidate_id"],
        "accepted": accepted,
        "guard_level": level,
        "reasons": sorted(set(reasons)),
        "component_decisions": component_decisions,
        "composed_calls": composed_calls if accepted else None,
        "composed_target": canonical_json(composed_calls) if accepted else None,
    }


def guard_direct_prediction(
    candidate: Mapping[str, Any],
    raw_prediction: str,
    *,
    level: str = "g4",
) -> Dict[str, Any]:
    """Guard a whole-request prediction at the same level as a composition arm.

    Comparing ``compose_g1`` against a G4-filtered direct arm confounds label
    source with guard strength, so the level is a parameter.
    """

    spec = {
        "expected_call_count": candidate["component_count"],
        "functions": candidate["functions"],
        "allow_exact_duplicates": False,
    }
    decision = guard_prediction(raw_prediction, spec, level=level)
    return {
        "candidate_id": candidate["candidate_id"],
        **decision,
        "composed_calls": decision["parsed_calls"] if decision["accepted"] else None,
        "composed_target": canonical_json(decision["parsed_calls"]) if decision["accepted"] else None,
    }


def audit_decision(
    candidate: Mapping[str, Any],
    oracle: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    exact = False
    evaluation: Optional[Dict[str, Any]] = None
    calls = decision.get("composed_calls")
    if calls is not None:
        result = evaluate_bfcl(oracle_example(candidate, oracle), canonical_json(calls))
        exact = bool(result.exact)
        evaluation = result.to_dict()
    return {
        "candidate_id": candidate["candidate_id"],
        "accepted": bool(decision["accepted"]),
        "oracle_exact": exact,
        "false_accept": bool(decision["accepted"] and not exact),
        "false_reject": bool(not decision["accepted"] and exact),
        "evaluation": evaluation,
    }


def summarize_guard_audit(
    decisions: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    audit_by_id = {str(row["candidate_id"]): row for row in audits}
    accepted = [row for row in decisions if row["accepted"]]
    accepted_exact = sum(bool(audit_by_id[str(row["candidate_id"])]["oracle_exact"]) for row in accepted)
    reason_counts: Counter[str] = Counter(
        reason for row in decisions for reason in row.get("reasons", [])
    )
    return {
        "candidate_count": len(decisions),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / max(len(decisions), 1),
        "accepted_oracle_exact_count": accepted_exact,
        "accepted_precision": accepted_exact / max(len(accepted), 1),
        "false_accept_count": sum(bool(row["false_accept"]) for row in audits),
        "false_reject_count": sum(bool(row["false_reject"]) for row in audits),
        "rejection_reasons": dict(sorted(reason_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
    }


def build_controlled_evaluation(
    test_examples: Sequence[AtomicExample],
    *,
    component_counts: Sequence[int] = (2, 4, 8),
    examples_per_cell: int = 200,
    seed: int,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Dict[str, List[AtomicExample]]:
    return {
        name: [oracle_example(public, oracle) for public, oracle in zip(*rows)]
        for name, rows in build_controlled_evaluation_candidates(
            test_examples,
            component_counts=component_counts,
            examples_per_cell=examples_per_cell,
            seed=seed,
            component_context=component_context,
        ).items()
    }


def build_controlled_evaluation_candidates(
    test_examples: Sequence[AtomicExample],
    *,
    component_counts: Sequence[int] = (2, 4, 8),
    examples_per_cell: int = 200,
    seed: int,
    component_context: str = DEFAULT_COMPONENT_CONTEXT,
) -> Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Controlled cells as candidate records, keeping the component specs.

    The frozen-composition baseline decomposes evaluation items the same way
    the learned conditions decompose training items, so it needs the specs
    rather than only the flattened evaluation examples.
    """

    output: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for partition, template_ids in (
        ("seen", TRAIN_TEMPLATE_IDS),
        ("heldout", HELDOUT_TEMPLATE_IDS),
    ):
        for arity in component_counts:
            total = math.comb(len(test_examples), arity)
            groups = _smallest_combinations(
                test_examples,
                count=min(total, examples_per_cell * 3),
                arity=arity,
                seed=seed,
                key=f"eval-{partition}-{arity}",
            )
            public_rows: List[Dict[str, Any]] = []
            oracle_rows: List[Dict[str, Any]] = []
            rejected_incompatible = 0
            for group in groups:
                index = len(public_rows)
                template_id = template_ids[index % len(template_ids)]
                try:
                    public, oracle = _make_candidate(
                        round_index=0,
                        split="test",
                        family="cross_function",
                        atomic_examples=group,
                        component_specs=[_atomic_spec(item) for item in group],
                        component_oracles=[_oracle_for_atomic(item) for item in group],
                        template_id=template_id,
                        template_partition=partition,
                        seed=seed,
                        candidate_id=f"eval-{partition}-{arity}-{index:04d}",
                        component_context=component_context,
                    )
                except ValueError as exc:
                    # BFCL reuses some function names with different schemas;
                    # such groups cannot form a well-defined union.
                    if "Incompatible schemas share function name" not in str(exc):
                        raise
                    rejected_incompatible += 1
                    continue
                public_rows.append(public)
                oracle_rows.append(oracle)
                if len(public_rows) == examples_per_cell:
                    break
            if len(public_rows) != examples_per_cell:
                raise ValueError(
                    f"Built only {len(public_rows)} of {examples_per_cell} compatible "
                    f"{arity}-call {partition} evaluation examples; "
                    f"rejected {rejected_incompatible}"
                )
            output[f"controlled_{partition}_{arity}"] = (public_rows, oracle_rows)
    return output


def candidate_counts(public_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(public_rows),
        "distinct_source_ids": len(
            {source for row in public_rows for source in row["source_component_ids"]}
        ),
        "templates": dict(Counter(str(row["template_id"]) for row in public_rows)),
        "families": dict(Counter(str(row["composition_family"]) for row in public_rows)),
    }
