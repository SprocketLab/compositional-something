"""Atomic BFCL and CommitPack data preparation.

The builders in this module intentionally avoid model-generated data.  They
turn pinned public records into deterministic, source-disjoint atomic SFT and
evaluation examples for the manual coding curriculum.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml


BFCL_SYSTEM_PROMPT = (
    "You are a function-calling model. Given a user request and available function schemas, "
    "return only a JSON array of calls. Each call must have exactly two keys: \"name\" and "
    "\"arguments\". Do not include prose or Markdown."
)
COMMITPACK_SYSTEM_PROMPT = (
    "You edit configuration documents. Apply exactly the requested change and return only an "
    "RFC 6902 JSON Patch array. Do not include prose or Markdown."
)


@dataclass(frozen=True)
class AtomicExample:
    """Portable JSONL record consumed by coding SFT and evaluators."""

    task: str
    source_id: str
    source_group_id: str
    split: str
    messages: Tuple[Dict[str, str], ...]
    target: str
    evaluator: Dict[str, Any]
    component_count: int = 1
    source_component_ids: Tuple[str, ...] = ()
    evaluation_track: str = "atomic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = list(self.messages)
        payload["source_component_ids"] = list(self.source_component_ids or (self.source_id,))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomicExample":
        return cls(
            task=str(payload["task"]),
            source_id=str(payload["source_id"]),
            source_group_id=str(payload["source_group_id"]),
            split=str(payload["split"]),
            messages=tuple(dict(message) for message in payload["messages"]),
            target=str(payload["target"]),
            evaluator=dict(payload.get("evaluator", {})),
            component_count=int(payload.get("component_count", 1)),
            source_component_ids=tuple(payload.get("source_component_ids", (payload["source_id"],))),
            evaluation_track=str(payload.get("evaluation_track", "atomic")),
            metadata=dict(payload.get("metadata", {})),
        )


def stable_hash(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    kwargs: Dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_examples(path: Path, examples: Sequence[AtomicExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_examples(path: Path) -> List[AtomicExample]:
    with path.open("r", encoding="utf-8") as handle:
        return [AtomicExample.from_dict(json.loads(line)) for line in handle if line.strip()]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _interleave_buckets(buckets: Mapping[str, Sequence[Any]]) -> List[Any]:
    queues = {key: deque(values) for key, values in sorted(buckets.items())}
    output: List[Any] = []
    while queues:
        for key in list(queues):
            queue = queues[key]
            if queue:
                output.append(queue.popleft())
            if not queue:
                del queues[key]
    return output


def stratified_exact_split(
    items: Sequence[Any],
    *,
    split_counts: Mapping[str, int],
    stratum_getter,
    id_getter,
    seed: int,
) -> Dict[str, List[Any]]:
    """Create exact-size deterministic partitions while interleaving strata."""

    if sum(split_counts.values()) != len(items):
        raise ValueError("split_counts must sum to the number of items")
    buckets: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        buckets[str(stratum_getter(item))].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: stable_hash(seed, id_getter(item)))
    ordered = _interleave_buckets(buckets)
    remaining = dict(split_counts)
    original = dict(split_counts)
    result = {name: [] for name in split_counts}
    names = list(split_counts)
    for item in ordered:
        eligible = [name for name in names if remaining[name] > 0]
        if not eligible:
            raise RuntimeError("partition allocation exhausted early")
        chosen = max(
            eligible,
            key=lambda name: (remaining[name] / max(original[name], 1), -names.index(name)),
        )
        result[chosen].append(item)
        remaining[chosen] -= 1
    if any(remaining.values()):
        raise RuntimeError(f"partition allocation did not fill requested counts: {remaining}")
    return result


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _normalize_schema(item) for key, item in value.items()}
    if result.get("type") == "dict":
        result["type"] = "object"
    elif result.get("type") == "list":
        result["type"] = "array"
    return result


def _first_accepted(values: Sequence[Any]) -> Tuple[bool, Any]:
    for value in values:
        if value != "":
            return True, value
    return False, None


def _bfcl_question_text(row: Mapping[str, Any]) -> str:
    conversations = row.get("question", [])
    if not conversations:
        raise ValueError(f"BFCL row {row.get('id')} has no question")
    turns = conversations[0]
    text = "\n".join(str(turn.get("content", "")) for turn in turns if turn.get("role") == "user").strip()
    if not text:
        raise ValueError(f"BFCL row {row.get('id')} has no user content")
    return text


def _resolve_bfcl_name(reference_name: str, schema_names: Sequence[str]) -> Tuple[str, bool]:
    if reference_name in schema_names:
        return reference_name, False
    suffix = reference_name.rsplit(".", 1)[-1]
    candidates = [name for name in schema_names if name.rsplit(".", 1)[-1] == suffix]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot uniquely map BFCL reference function {reference_name!r} to schemas {schema_names!r}"
        )
    return candidates[0], True


def canonicalize_bfcl_row(
    question_row: Mapping[str, Any],
    answer_row: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    functions = [_normalize_schema(function) for function in question_row.get("function", [])]
    if not functions:
        raise ValueError(f"BFCL row {question_row.get('id')} has no functions")
    schema_by_name = {str(function["name"]): function for function in functions}
    canonical_calls: List[Dict[str, Any]] = []
    accepted_calls: List[Dict[str, Any]] = []
    repairs: List[Dict[str, str]] = []
    for raw_call in answer_row.get("ground_truth", []):
        if not isinstance(raw_call, dict) or len(raw_call) != 1:
            raise ValueError(f"Malformed BFCL ground truth in {question_row.get('id')}")
        raw_name, raw_arguments = next(iter(raw_call.items()))
        name, repaired = _resolve_bfcl_name(str(raw_name), list(schema_by_name))
        if repaired:
            repairs.append({"reference": str(raw_name), "schema": name})
        if not isinstance(raw_arguments, dict):
            raise ValueError(f"Malformed arguments in {question_row.get('id')}")
        canonical_arguments: Dict[str, Any] = {}
        accepted_arguments: Dict[str, List[Any]] = {}
        for argument_name, raw_values in raw_arguments.items():
            values = list(raw_values) if isinstance(raw_values, list) else [raw_values]
            accepted_arguments[str(argument_name)] = values
            present, selected = _first_accepted(values)
            if present:
                canonical_arguments[str(argument_name)] = selected
        required = schema_by_name[name].get("parameters", {}).get("required", [])
        missing = [argument for argument in required if argument not in canonical_arguments]
        if missing:
            raise ValueError(f"BFCL row {question_row.get('id')} misses required arguments: {missing}")
        canonical_calls.append({"name": name, "arguments": canonical_arguments})
        accepted_calls.append({"name": name, "arguments": accepted_arguments})
    if not canonical_calls:
        raise ValueError(f"BFCL row {question_row.get('id')} has no reference calls")
    return canonical_calls, accepted_calls, {"functions": functions, "namespace_repairs": repairs}


def _bfcl_complexity(row: Mapping[str, Any]) -> str:
    functions = row.get("function", [])
    required = sum(len(function.get("parameters", {}).get("required", [])) for function in functions)
    properties = sum(len(function.get("parameters", {}).get("properties", {})) for function in functions)
    return f"r{min(required, 4)}-p{min(properties, 6)}"


def make_bfcl_example(
    question_row: Mapping[str, Any],
    answer_row: Mapping[str, Any],
    *,
    split: str,
    evaluation_track: str = "atomic",
) -> AtomicExample:
    canonical_calls, accepted_calls, details = canonicalize_bfcl_row(question_row, answer_row)
    question = _bfcl_question_text(question_row)
    functions = details["functions"]
    user = (
        f"User request:\n{question}\n\nAvailable functions:\n"
        + canonical_json(functions)
    )
    source_id = str(question_row["id"])
    return AtomicExample(
        task="bfcl",
        source_id=source_id,
        source_group_id=source_id,
        split=split,
        messages=(
            {"role": "system", "content": BFCL_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ),
        target=canonical_json(canonical_calls),
        evaluator={"accepted_calls": accepted_calls, "functions": functions},
        component_count=len(canonical_calls),
        source_component_ids=(source_id,),
        evaluation_track=evaluation_track,
        metadata={
            "question": question,
            "complexity": _bfcl_complexity(question_row),
            "namespace_repairs": details["namespace_repairs"],
        },
    )


def build_bfcl_atomic_splits(
    question_path: Path,
    answer_path: Path,
    *,
    seed: int = 20260718,
) -> Tuple[Dict[str, List[AtomicExample]], Dict[str, Any]]:
    questions = read_jsonl(question_path)
    answers = {str(row["id"]): row for row in read_jsonl(answer_path)}
    if len(questions) != 400:
        raise ValueError(f"Expected 400 pinned BFCL Simple rows, found {len(questions)}")
    if {str(row["id"]) for row in questions} != set(answers):
        raise ValueError("BFCL question and answer IDs differ")
    partitioned = stratified_exact_split(
        questions,
        split_counts={"train": 240, "hidden_composition": 60, "validation": 40, "test": 60},
        stratum_getter=_bfcl_complexity,
        id_getter=lambda row: row["id"],
        seed=seed,
    )
    built: Dict[str, List[AtomicExample]] = {}
    repair_count = 0
    for split, rows in partitioned.items():
        examples = [make_bfcl_example(row, answers[str(row["id"])], split=split) for row in rows]
        buckets: Dict[str, List[AtomicExample]] = defaultdict(list)
        for example in examples:
            buckets[str(example.metadata["complexity"])].append(example)
            repair_count += len(example.metadata.get("namespace_repairs", []))
        for values in buckets.values():
            values.sort(key=lambda example: stable_hash(seed, split, example.source_id))
        built[split] = _interleave_buckets(buckets)
    audit = {
        "source_rows": len(questions),
        "split_counts": {split: len(rows) for split, rows in built.items()},
        "namespace_repairs": repair_count,
        "split_seed": seed,
    }
    return built, audit


def build_bfcl_natural_examples(
    question_path: Path,
    answer_path: Path,
    *,
    track_name: str,
) -> List[AtomicExample]:
    questions = read_jsonl(question_path)
    answers = {str(row["id"]): row for row in read_jsonl(answer_path)}
    return [
        make_bfcl_example(
            row,
            answers[str(row["id"])],
            split="test",
            evaluation_track=track_name,
        )
        for row in questions
    ]


def build_bfcl_controlled_frontier(
    atomic_test: Sequence[AtomicExample],
    *,
    component_counts: Sequence[int] = (2, 4, 8),
    examples_per_count: int = 200,
    seed: int = 20260718,
) -> Dict[int, List[AtomicExample]]:
    """Join held-out atomic BFCL requests without exposing their labels in prompts."""

    if len(atomic_test) < max(component_counts):
        raise ValueError("Not enough atomic test sources for the requested frontier")
    frontier: Dict[int, List[AtomicExample]] = {}
    source_pool = sorted(atomic_test, key=lambda item: stable_hash(seed, item.source_id))
    for component_count in component_counts:
        examples: List[AtomicExample] = []
        seen_groups = set()
        attempts = 0
        while len(examples) < examples_per_count and attempts < examples_per_count * 50:
            selected = sorted(
                source_pool,
                key=lambda item: stable_hash(seed, component_count, attempts, item.source_id),
            )[:component_count]
            attempts += 1
            functions_by_name: Dict[str, Dict[str, Any]] = {}
            collision = False
            for component in selected:
                for function in component.evaluator["functions"]:
                    name = str(function["name"])
                    if name in functions_by_name and functions_by_name[name] != function:
                        collision = True
                        break
                    functions_by_name[name] = function
                if collision:
                    break
            if collision:
                continue
            component_ids = tuple(component.source_id for component in selected)
            group_key = hashlib.sha256("\x1f".join(component_ids).encode("utf-8")).hexdigest()[:16]
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            requests = "\n".join(
                f"{index}. {component.metadata['question']}"
                for index, component in enumerate(selected, start=1)
            )
            user = (
                "Complete every numbered request. Calls may be returned in any order.\n\n"
                f"Requests:\n{requests}\n\nAvailable functions:\n"
                + canonical_json(list(functions_by_name.values()))
            )
            targets = [call for component in selected for call in json.loads(component.target)]
            accepted = [
                call
                for component in selected
                for call in component.evaluator["accepted_calls"]
            ]
            examples.append(
                AtomicExample(
                    task="bfcl",
                    source_id=f"controlled-{component_count}-{group_key}",
                    source_group_id=f"controlled-{component_count}-{group_key}",
                    split="test",
                    messages=(
                        {"role": "system", "content": BFCL_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ),
                    target=canonical_json(targets),
                    evaluator={
                        "accepted_calls": accepted,
                        "functions": list(functions_by_name.values()),
                    },
                    component_count=component_count,
                    source_component_ids=component_ids,
                    evaluation_track="controlled",
                    metadata={"join_template": "numbered_all_requests"},
                )
            )
        if len(examples) < examples_per_count:
            raise ValueError(
                f"Could only construct {len(examples)} controlled BFCL examples at count={component_count}"
            )
        frontier[component_count] = examples
    return frontier


class StrictYAMLLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-string mapping keys."""


def _construct_unique_mapping(loader: StrictYAMLLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    if any(getattr(key_node, "value", None) == "<<" for key_node, _value_node in node.value):
        raise ValueError("YAML merge keys are not supported")
    loader.flatten_mapping(node)
    mapping: Dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")
        if key == "<<":
            raise ValueError("YAML merge keys are not supported")
        if key in mapping:
            raise ValueError(f"Duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictYAMLLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_yaml_events(text: str) -> None:
    allowed_tags = {
        "tag:yaml.org,2002:map",
        "tag:yaml.org,2002:seq",
        "tag:yaml.org,2002:str",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:null",
    }
    for event in yaml.parse(text):
        if isinstance(event, yaml.events.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ValueError("YAML aliases and anchors are not supported")
        tag = getattr(event, "tag", None)
        if tag is not None and tag not in allowed_tags:
            raise ValueError(f"Unsupported YAML tag: {tag}")


def validate_json_tree(value: Any, *, path: str = "") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float at {path or '/'}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_tree(item, path=f"{path}/{index}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Non-string key at {path or '/'}")
            validate_json_tree(item, path=f"{path}/{pointer_escape(key)}")
        return
    raise ValueError(f"Non-JSON value of type {type(value).__name__} at {path or '/'}")


def parse_config_document(text: str, language: str) -> Any:
    text = text.lstrip("\ufeff")
    if language.lower() == "json":
        value = json.loads(text)
    elif language.lower() == "yaml":
        _validate_yaml_events(text)
        documents = list(yaml.load_all(text, Loader=StrictYAMLLoader))
        if len(documents) != 1:
            raise ValueError("YAML input must contain exactly one document")
        value = documents[0]
    else:
        raise ValueError(f"Unsupported configuration language: {language}")
    validate_json_tree(value)
    return value


def pointer_escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def pointer_tokens(path: str) -> List[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON Pointer: {path!r}")
    return [pointer_unescape(token) for token in path[1:].split("/")]


def structural_scalar_diff(old: Any, new: Any, *, path: str = "") -> List[Dict[str, Any]]:
    """Return a deterministic scalar-only JSON Patch or reject the diff."""

    if type(old) is not type(new):
        if path and not isinstance(old, (dict, list)) and not isinstance(new, (dict, list)):
            return [{"op": "replace", "path": path, "value": new, "old_value": old}]
        raise ValueError("Container-type changes are not supported")
    if isinstance(old, dict):
        operations: List[Dict[str, Any]] = []
        for key in sorted(old.keys() - new.keys()):
            value = old[key]
            if isinstance(value, (dict, list)):
                raise ValueError("Removing containers is not supported")
            operations.append({"op": "remove", "path": f"{path}/{pointer_escape(key)}", "old_value": value})
        for key in sorted(new.keys() - old.keys()):
            value = new[key]
            if isinstance(value, (dict, list)):
                raise ValueError("Adding containers is not supported")
            operations.append({"op": "add", "path": f"{path}/{pointer_escape(key)}", "value": value})
        for key in sorted(old.keys() & new.keys()):
            operations.extend(
                structural_scalar_diff(old[key], new[key], path=f"{path}/{pointer_escape(key)}")
            )
        return operations
    if isinstance(old, list):
        if old != new:
            raise ValueError("Array changes are not supported")
        return []
    if old != new:
        if not path:
            raise ValueError("Replacing the document root is not supported")
        return [{"op": "replace", "path": path, "value": new, "old_value": old}]
    return []


def _resolve_parent(document: Any, path: str) -> Tuple[MutableMapping[str, Any], str]:
    tokens = pointer_tokens(path)
    if not tokens:
        raise ValueError("Root patch operations are not supported")
    parent = document
    for token in tokens[:-1]:
        if not isinstance(parent, dict) or token not in parent:
            raise ValueError(f"Patch path does not exist: {path}")
        parent = parent[token]
    if not isinstance(parent, dict):
        raise ValueError("Only mapping-key patch operations are supported")
    return parent, tokens[-1]


def apply_scalar_patch(document: Any, operations: Sequence[Mapping[str, Any]]) -> Any:
    result = copy.deepcopy(document)
    for operation in operations:
        parent, key = _resolve_parent(result, str(operation["path"]))
        op = operation["op"]
        if op == "add":
            if key in parent:
                raise ValueError(f"Add target already exists: {operation['path']}")
            parent[key] = copy.deepcopy(operation["value"])
        elif op == "remove":
            if key not in parent:
                raise ValueError(f"Remove target is missing: {operation['path']}")
            del parent[key]
        elif op == "replace":
            if key not in parent:
                raise ValueError(f"Replace target is missing: {operation['path']}")
            parent[key] = copy.deepcopy(operation["value"])
        else:
            raise ValueError(f"Unsupported patch operation: {op!r}")
    return result


def normalize_repository(raw_repositories: Any) -> str:
    repositories = {
        item.strip().lower()
        for item in str(raw_repositories or "").split(",")
        if item.strip()
    }
    if not repositories:
        raise ValueError("CommitPack row has no repository")
    return sorted(repositories)[0]


def repository_split(repository: str) -> str:
    bucket = stable_hash("commitpack-repository-split", repository) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _path_words(path: str) -> Tuple[str, str]:
    tokens = pointer_tokens(path)
    leaf = tokens[-1]
    parent = ".".join(tokens[:-1]) or "the document root"
    dotted = ".".join(tokens)
    return dotted, parent


def instruction_for_operation(operation: Mapping[str, Any]) -> str:
    dotted, parent = _path_words(str(operation["path"]))
    op = operation["op"]
    if op == "replace":
        return f"Set `{dotted}` to {canonical_json(operation['value'])}."
    if op == "add":
        leaf = pointer_tokens(str(operation["path"]))[-1]
        return f"Add `{leaf}` under `{parent}` with value {canonical_json(operation['value'])}."
    if op == "remove":
        return f"Remove `{dotted}`."
    raise ValueError(f"Unsupported operation: {op!r}")


def public_patch(operation: Mapping[str, Any]) -> Dict[str, Any]:
    result = {"op": operation["op"], "path": operation["path"]}
    if operation["op"] in {"add", "replace"}:
        result["value"] = operation["value"]
    return result


def _value_at(document: Any, tokens: Sequence[str]) -> Any:
    value = document
    for token in tokens:
        if not isinstance(value, dict) or token not in value:
            raise KeyError(token)
        value = value[token]
    return value


def _cropped_parent(document: Any, operation: Mapping[str, Any], *, sibling_limit: int = 6) -> Tuple[str, Any]:
    tokens = pointer_tokens(str(operation["path"]))
    parent_tokens = tokens[:-1]
    parent = _value_at(document, parent_tokens) if parent_tokens else document
    if not isinstance(parent, dict):
        raise ValueError("Only mapping-key operations are supported")
    leaf = tokens[-1]
    keys = sorted(parent)
    selected: List[str] = []
    if leaf in parent:
        selected.append(leaf)
    for key in keys:
        if key != leaf and len(selected) < sibling_limit:
            selected.append(key)
    cropped = {key: parent[key] for key in selected}
    prefix = "/" + "/".join(pointer_escape(token) for token in parent_tokens) if parent_tokens else "/"
    return prefix, cropped


def render_config_context(document: Any, language: str) -> str:
    if language.lower() == "yaml":
        return yaml.safe_dump(document, sort_keys=False, allow_unicode=True).rstrip()
    return canonical_json(document, pretty=True)


def _chat_token_count(tokenizer: Any, messages: Sequence[Mapping[str, str]], target: str) -> int:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    return len(prefix_ids) + len(tokenizer.encode(target, add_special_tokens=False)) + 1


def build_commitpack_prompt(
    *,
    old_document: Any,
    language: str,
    operation: Mapping[str, Any],
    tokenizer: Any,
    max_tokens: int,
) -> Tuple[Tuple[Dict[str, str], ...], Dict[str, Any]]:
    target = canonical_json([public_patch(operation)])
    instruction = instruction_for_operation(operation)

    def messages_for(prefix: str, context: Any) -> Tuple[Dict[str, str], ...]:
        rendered_context = render_config_context(context, language)
        user = (
            f"Configuration language: {language.upper()}\n"
            f"Context rooted at {prefix}:\n{rendered_context}\n\n"
            f"Requested change:\n{instruction}\n\nReturn only a JSON Patch array."
        )
        return (
            {"role": "system", "content": COMMITPACK_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        )

    full = messages_for("/", old_document)
    if _chat_token_count(tokenizer, full, target) <= max_tokens:
        return full, {"context_root": "/", "context_mode": "full"}
    tokens = pointer_tokens(str(operation["path"]))
    for depth in range(1, len(tokens)):
        try:
            subtree = _value_at(old_document, tokens[:depth])
        except KeyError:
            continue
        prefix = "/" + "/".join(pointer_escape(token) for token in tokens[:depth])
        candidate = messages_for(prefix, subtree)
        if _chat_token_count(tokenizer, candidate, target) <= max_tokens:
            return candidate, {"context_root": prefix, "context_mode": "subtree"}
    prefix, cropped = _cropped_parent(old_document, operation)
    candidate = messages_for(prefix, cropped)
    if _chat_token_count(tokenizer, candidate, target) > max_tokens:
        raise ValueError("Context does not fit the configured token limit")
    return candidate, {"context_root": prefix, "context_mode": "cropped_parent"}


def commitpack_source_id(row: Mapping[str, Any]) -> str:
    raw = "\x1f".join(
        str(row.get(key, "")) for key in ("repos", "commit", "old_file", "new_file")
    )
    return "commitpack-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def candidate_from_commitpack_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_tokens: int = 1024,
) -> Tuple[AtomicExample, Dict[str, Any]]:
    language = str(row.get("lang", "")).lower()
    if language not in {"json", "yaml"}:
        raise ValueError("Only JSON and YAML rows are supported")
    if str(row.get("license", "")).strip().lower() in {"", "unknown"}:
        raise ValueError("Unknown licenses are excluded")
    if row.get("old_file") != row.get("new_file"):
        raise ValueError("Renames are excluded")
    old_document = parse_config_document(str(row.get("old_contents", "")), language)
    new_document = parse_config_document(str(row.get("new_contents", "")), language)
    operations = structural_scalar_diff(old_document, new_document)
    if not operations:
        raise ValueError("No supported structural changes")
    source_id = commitpack_source_id(row)
    operation = sorted(
        operations,
        key=lambda item: stable_hash(source_id, item["op"], item["path"]),
    )[0]
    public_operation = public_patch(operation)
    intended_document = apply_scalar_patch(old_document, [public_operation])
    repository = normalize_repository(row.get("repos"))
    split = repository_split(repository)
    messages, context_metadata = build_commitpack_prompt(
        old_document=old_document,
        language=language,
        operation=operation,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    target = canonical_json([public_operation])
    source_group_id = f"{repository}@{row.get('commit')}:{row.get('old_file')}"
    example = AtomicExample(
        task="commitpack",
        source_id=source_id,
        source_group_id=source_group_id,
        split=split,
        messages=messages,
        target=target,
        evaluator={
            "old_document": old_document,
            "intended_document": intended_document,
            "operation": public_operation,
        },
        component_count=1,
        source_component_ids=(f"{source_id}#{operation['path']}",),
        evaluation_track="atomic",
        metadata={
            "language": language,
            "operation": operation["op"],
            "repository": repository,
            "commit": row.get("commit"),
            "file": row.get("old_file"),
            "license": str(row.get("license", "")).lower(),
            "source_operation_count": len(operations),
            **context_metadata,
        },
    )
    return example, {"operation_count": len(operations), "all_operations": operations}


def make_commitpack_natural_example(
    atomic_example: AtomicExample,
    operations: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_tokens: int = 2048,
) -> AtomicExample:
    public_operations = [public_patch(operation) for operation in operations]
    old_document = atomic_example.evaluator["old_document"]
    target = canonical_json(public_operations)
    instructions = "\n".join(
        f"{index}. {instruction_for_operation(operation)}"
        for index, operation in enumerate(operations, start=1)
    )
    language = str(atomic_example.metadata["language"])
    user = (
        f"Configuration language: {language.upper()}\n"
        f"Context rooted at /:\n{render_config_context(old_document, language)}\n\n"
        f"Requested changes:\n{instructions}\n\nReturn only a JSON Patch array."
    )
    messages = (
        {"role": "system", "content": COMMITPACK_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    if _chat_token_count(tokenizer, messages, target) > max_tokens:
        raise ValueError("Natural multi-edit context exceeds the evaluation token limit")
    intended = apply_scalar_patch(old_document, public_operations)
    return AtomicExample(
        task="commitpack",
        source_id=f"{atomic_example.source_id}-natural-{len(operations)}",
        source_group_id=atomic_example.source_group_id,
        split="test",
        messages=messages,
        target=target,
        evaluator={
            "old_document": old_document,
            "intended_document": intended,
            "operations": public_operations,
        },
        component_count=len(operations),
        source_component_ids=tuple(
            f"{atomic_example.source_id}#{operation['path']}" for operation in operations
        ),
        evaluation_track="natural",
        metadata={**atomic_example.metadata, "context_root": "/", "context_mode": "full"},
    )


def _deduplicate_commitpack(examples: Sequence[AtomicExample]) -> List[AtomicExample]:
    chosen: Dict[str, AtomicExample] = {}
    for example in examples:
        key = hashlib.sha256(
            (canonical_json(example.evaluator["old_document"]) + "\x1f" + example.target).encode("utf-8")
        ).hexdigest()
        current = chosen.get(key)
        if current is None or example.source_id < current.source_id:
            chosen[key] = example
    return list(chosen.values())


def _balanced_commitpack_order(examples: Sequence[AtomicExample], *, seed: int) -> List[AtomicExample]:
    buckets: Dict[str, List[AtomicExample]] = defaultdict(list)
    for example in examples:
        stratum = f"{example.metadata['language']}:{example.metadata['operation']}"
        buckets[stratum].append(example)
    for values in buckets.values():
        values.sort(key=lambda item: stable_hash(seed, item.source_id))
    return _interleave_buckets(buckets)


def build_commitpack_atomic_splits(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_tokens: int = 1024,
    seed: int = 20260718,
    train_count: int = 2000,
    validation_count: int = 600,
    test_count: int = 1200,
    min_total_candidates: int = 10_000,
    min_per_stratum: int = 700,
    min_eval_per_stratum: int = 50,
) -> Tuple[Dict[str, List[AtomicExample]], Dict[str, Any]]:
    candidates: List[AtomicExample] = []
    frontier_candidates: Dict[int, List[AtomicExample]] = defaultdict(list)
    frontier_rejections: Dict[str, int] = defaultdict(int)
    rejection_counts: Dict[str, int] = defaultdict(int)
    source_operation_counts: Dict[str, int] = defaultdict(int)
    scanned = 0
    for row in rows:
        scanned += 1
        if scanned % 10_000 == 0:
            print(
                f"[INFO] CommitPack preprocessing: scanned={scanned} accepted={len(candidates)}",
                flush=True,
            )
        try:
            candidate, details = candidate_from_commitpack_row(
                row,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            reason = commitpack_rejection_category(exc)
            rejection_counts[reason] += 1
            continue
        source_operation_counts[str(details["operation_count"])] += 1
        candidates.append(candidate)
        operation_count = int(details["operation_count"])
        if candidate.split == "test" and operation_count in {2, 4, 8}:
            try:
                frontier_candidates[operation_count].append(
                    make_commitpack_natural_example(
                        candidate,
                        details["all_operations"],
                        tokenizer=tokenizer,
                    )
                )
            except Exception as exc:
                frontier_rejections[str(exc)] += 1
    candidates = _deduplicate_commitpack(candidates)
    if len(candidates) < min_total_candidates:
        raise ValueError(
            f"CommitPack yielded {len(candidates)} deduplicated atomic candidates; "
            f"at least {min_total_candidates} are required"
        )
    global_strata: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        global_strata[f"{candidate.metadata['language']}:{candidate.metadata['operation']}"] += 1
    required_strata = [
        f"{language}:{operation}"
        for language in ("json", "yaml")
        for operation in ("add", "remove", "replace")
    ]
    if any(global_strata.get(stratum, 0) < min_per_stratum for stratum in required_strata):
        raise ValueError(
            f"CommitPack does not meet the {min_per_stratum}-per-stratum global gate: "
            f"{dict(sorted(global_strata.items()))}"
        )
    by_split: Dict[str, List[AtomicExample]] = defaultdict(list)
    for candidate in candidates:
        by_split[candidate.split].append(candidate)
    requested = {"train": train_count, "validation": validation_count, "test": test_count}
    selected: Dict[str, List[AtomicExample]] = {}
    stratum_counts: Dict[str, Dict[str, int]] = {}
    for split, count in requested.items():
        ordered = _balanced_commitpack_order(by_split[split], seed=seed)
        available: Dict[str, int] = defaultdict(int)
        for example in by_split[split]:
            available[f"{example.metadata['language']}:{example.metadata['operation']}"] += 1
        stratum_counts[split] = dict(sorted(available.items()))
        split_quota = (
            math.ceil(count / len(required_strata))
            if split == "train"
            else min(min_eval_per_stratum, math.ceil(count / len(required_strata)))
        )
        if any(available.get(stratum, 0) < split_quota for stratum in required_strata):
            raise ValueError(
                f"CommitPack {split} split cannot fill a balanced {count}-example set: "
                f"{dict(sorted(available.items()))}"
            )
        if len(ordered) < count:
            raise ValueError(f"CommitPack {split} split has {len(ordered)} candidates, needs {count}")
        selected[split] = ordered[:count]
    frontier_limits = {2: 200, 4: 200, 8: 50}
    frontier_selected: Dict[str, int] = {}
    for component_count, limit in frontier_limits.items():
        values = sorted(
            frontier_candidates.get(component_count, []),
            key=lambda item: stable_hash(seed, "frontier", item.source_id),
        )[:limit]
        selected[f"frontier_{component_count}"] = values
        frontier_selected[str(component_count)] = len(values)
    selected_strata: Dict[str, Dict[str, int]] = {}
    for split in requested:
        counts: Dict[str, int] = defaultdict(int)
        for example in selected[split]:
            counts[f"{example.metadata['language']}:{example.metadata['operation']}"] += 1
        selected_strata[split] = dict(sorted(counts.items()))
    audit = {
        "source_rows_scanned": scanned,
        "deduplicated_candidates": len(candidates),
        "available_split_counts": {split: len(values) for split, values in by_split.items()},
        "selected_split_counts": {split: len(selected[split]) for split in requested},
        "stratum_counts": stratum_counts,
        "global_stratum_counts": dict(sorted(global_strata.items())),
        "selected_stratum_counts": selected_strata,
        "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))),
        "source_operation_counts": dict(sorted(source_operation_counts.items(), key=lambda item: int(item[0]))),
        "split_seed": seed,
        "max_tokens": max_tokens,
        "minimum_global_per_stratum": min_per_stratum,
        "minimum_eval_split_per_stratum": min_eval_per_stratum,
        "frontier_selected_counts": frontier_selected,
        "frontier_rejection_counts": dict(sorted(frontier_rejections.items())),
    }
    return selected, audit


def commitpack_rejection_category(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    if isinstance(exc, yaml.YAMLError):
        return "yaml_parse_error"
    message = re.sub(r"\s+", " ", str(exc)).strip()
    rules = (
        ("Unknown licenses", "unknown_license"),
        ("Renames", "rename"),
        ("Array changes", "array_change"),
        ("Adding containers", "container_add"),
        ("Removing containers", "container_remove"),
        ("Container-type changes", "container_type_change"),
        ("Duplicate YAML", "yaml_duplicate_key"),
        ("YAML aliases", "yaml_anchor_or_alias"),
        ("YAML merge", "yaml_merge_key"),
        ("Unsupported YAML tag", "yaml_custom_tag"),
        ("YAML mapping keys", "yaml_non_string_key"),
        ("Non-JSON value", "yaml_non_json_value"),
        ("Non-finite float", "non_finite_float"),
        ("exactly one document", "yaml_multiple_documents"),
        ("Context does not fit", "context_too_long"),
        ("No supported structural changes", "no_supported_change"),
        ("Replacing the document root", "root_replacement"),
    )
    for prefix, category in rules:
        if prefix in message:
            return category
    return type(exc).__name__


def iter_jsonl_paths(paths: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
