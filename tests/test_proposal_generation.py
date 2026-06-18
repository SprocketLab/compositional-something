from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.adaptive.attempts import attempt_prompt_runtime
from self.adaptive.proposals import (
    proposal_executable_validation,
    proposal_generation,
    proposal_prompt_metadata,
    proposal_prompts,
    proposal_runtime,
)
from self.adaptive.run import driver_compat_exports, driver_default_bindings
from self.adaptive.proposals import PromptBundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_rows_for_round_prefers_attempt_specific_fixture_rows():
    rows = [
        {"round": 1, "id": "round-1"},
        {"round": 2, "attempt": 3, "id": "attempt-3-a"},
        {"round": 2, "attempt": 3, "id": "attempt-3-b"},
        {"round": 2, "attempt": 4, "id": "attempt-4"},
    ]

    assert [row["id"] for row in proposal_generation._rows_for_round(rows, 2, attempt_index=3)] == [
        "attempt-3-a",
        "attempt-3-b",
    ]
    assert [row["id"] for row in proposal_generation._rows_for_round(rows, 1, attempt_index=9)] == [
        "round-1"
    ]
    assert proposal_generation._rows_for_round(rows, 99) == rows


def test_load_or_generate_proposal_rows_uses_fixture_round_slice_and_candidate_limit(tmp_path: Path):
    fixture = tmp_path / "proposals.jsonl"
    _write_jsonl(
        fixture,
        [
            {"round": 1, "id": "unused"},
            {"round": 2, "id": "first"},
            {"round": 2, "id": "second"},
            {"round": 2, "id": "third"},
        ],
    )
    args = SimpleNamespace(
        proposal_fixture_jsonl=fixture,
        num_candidates=2,
        proposal_model_name="current",
        proposal_max_new_tokens=8,
        proposal_temperature=0.0,
        proposal_top_p=1.0,
    )

    rows = proposal_generation.load_or_generate_proposal_rows(
        args=args,
        prompt=PromptBundle(system="system", user="user"),
        current_model=None,
        current_tokenizer=None,
        round_index=2,
    )

    assert [row["id"] for row in rows] == ["first", "second"]
    rows[0]["id"] = "mutated"
    assert json.loads(fixture.read_text(encoding="utf-8").splitlines()[1])["id"] == "first"


def test_proposal_runtime_reexports_generation_helpers_for_compatibility():
    assert proposal_runtime._rows_for_round is proposal_generation._rows_for_round
    assert proposal_runtime.generate_proposals_from_model is proposal_generation.generate_proposals_from_model
    assert proposal_runtime.load_or_generate_proposal_rows is proposal_generation.load_or_generate_proposal_rows


def test_proposal_implementation_imports_use_canonical_owners():
    assert proposal_prompts.choose_default_program_pair is proposal_prompt_metadata.choose_default_program_pair
    assert proposal_runtime.choose_default_program_pair is proposal_prompt_metadata.choose_default_program_pair
    assert attempt_prompt_runtime.choose_default_program_pair is proposal_prompt_metadata.choose_default_program_pair
    assert attempt_prompt_runtime.render_program_candidate_prompt is proposal_prompts.render_program_candidate_prompt
    assert driver_default_bindings.choose_default_program_pair is proposal_prompt_metadata.choose_default_program_pair
    assert driver_default_bindings.render_program_candidate_prompt is proposal_prompts.render_program_candidate_prompt
    assert driver_compat_exports.load_or_generate_proposal_rows is proposal_generation.load_or_generate_proposal_rows
    assert driver_compat_exports._extract_python_code is proposal_executable_validation._extract_python_code
    assert driver_compat_exports.validate_executable_rows is proposal_executable_validation.validate_executable_rows
