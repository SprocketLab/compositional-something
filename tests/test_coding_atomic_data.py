from __future__ import annotations

import json

import pytest

from self.coding.atomic_data import (
    AtomicExample,
    apply_scalar_patch,
    candidate_from_commitpack_row,
    canonicalize_bfcl_row,
    parse_config_document,
    structural_scalar_diff,
)
from self.coding.evaluation import evaluate_bfcl, evaluate_commitpack


class FakeTokenizer:
    unk_token_id = -1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt
        assert enable_thinking is False
        text = " ".join(message["content"] for message in messages) + " assistant"
        if tokenize:
            return {"input_ids": list(range(len(text.split())))}
        return text

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_bfcl_canonicalization_repairs_unique_namespace_and_preserves_options():
    question = {
        "id": "x",
        "question": [[{"role": "user", "content": "Find sushi."}]],
        "function": [
            {
                "name": "restaurant_search.find_closest",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}, "patio": {"type": "boolean"}},
                    "required": ["city"],
                },
            }
        ],
    }
    answer = {
        "id": "x",
        "ground_truth": [{"find_closest": {"city": ["Boston"], "patio": [True, ""]}}],
    }
    calls, accepted, details = canonicalize_bfcl_row(question, answer)
    assert calls == [
        {"name": "restaurant_search.find_closest", "arguments": {"city": "Boston", "patio": True}}
    ]
    assert accepted[0]["arguments"]["patio"] == [True, ""]
    assert details["namespace_repairs"] == [
        {"reference": "find_closest", "schema": "restaurant_search.find_closest"}
    ]
    assert details["functions"][0]["parameters"]["type"] == "object"


@pytest.mark.parametrize(
    "source",
    [
        "a: 1\na: 2\n",
        "a: &value 1\nb: *value\n",
        "base: {a: 1}\nderived: {<<: {a: 1}}\n",
        "when: 2026-07-18\n",
    ],
)
def test_strict_yaml_rejects_unsafe_or_non_json_features(source):
    with pytest.raises((ValueError, yaml_error_types())):
        parse_config_document(source, "yaml")


def yaml_error_types():
    import yaml

    return yaml.YAMLError


def test_scalar_diff_and_patch_round_trip_with_pointer_escaping():
    old = {"service": {"a/b": 1, "legacy": True}, "unchanged": [1, 2]}
    new = {"service": {"a/b": 2, "timeout": 30}, "unchanged": [1, 2]}
    operations = structural_scalar_diff(old, new)
    public = [{key: value for key, value in operation.items() if key != "old_value"} for operation in operations]
    assert [operation["op"] for operation in public] == ["remove", "add", "replace"]
    assert public[-1]["path"] == "/service/a~1b"
    assert apply_scalar_patch(old, public) == new


def test_commitpack_candidate_is_atomic_and_behaviorally_evaluable():
    row = {
        "commit": "abc",
        "old_file": "config.json",
        "new_file": "config.json",
        "old_contents": json.dumps({"service": {"port": 80, "debug": True}}),
        "new_contents": json.dumps({"service": {"port": 81, "debug": False}}),
        "lang": "JSON",
        "license": "mit",
        "repos": "Owner/Repo",
    }
    example, details = candidate_from_commitpack_row(row, tokenizer=FakeTokenizer())
    assert example.component_count == 1
    assert example.metadata["repository"] == "owner/repo"
    assert details["operation_count"] == 2
    assert evaluate_commitpack(example, example.target).exact
    assert not evaluate_commitpack(example, "```json\n[]\n```").format_valid


def test_bfcl_evaluator_accepts_optional_omission_and_rejects_extra_arguments():
    example = AtomicExample(
        task="bfcl",
        source_id="x",
        source_group_id="x",
        split="test",
        messages=(),
        target='[{"name":"f","arguments":{"x":1}}]',
        evaluator={
            "functions": [
                {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "unit": {"type": "string"}},
                        "required": ["x"],
                    },
                }
            ],
            "accepted_calls": [{"name": "f", "arguments": {"x": [1], "unit": ["meters", ""]}}],
        },
    )
    assert evaluate_bfcl(example, '[{"name":"f","arguments":{"x":1}}]').exact
    assert not evaluate_bfcl(example, '[{"name":"f","arguments":{"x":1,"bad":2}}]').exact
