from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.adaptive import proposal as proposal_generation
from self.adaptive import proposal as proposal_runtime
from self.adaptive.proposal import PromptBundle


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


def test_model_generation_logs_special_token_only_completions():
    import torch

    class DummyTokenizer:
        bos_token_id = None
        pad_token_id = 0
        eos_token_id = 2
        all_special_ids = [0, 2]
        padding_side = "right"

        def encode(self, text, add_special_tokens=False):
            return [10 + (ord(char) % 100) for char in text]

        def decode(self, token_ids, skip_special_tokens=True):
            pieces = []
            for token_id in token_ids:
                token_id = int(token_id)
                if token_id == self.eos_token_id:
                    if not skip_special_tokens:
                        pieces.append("<eos>")
                    continue
                if token_id == self.pad_token_id:
                    if not skip_special_tokens:
                        pieces.append("<pad>")
                    continue
                pieces.append(chr((token_id - 10) % 100))
            return "".join(pieces)

    class DummyModel:
        training = True

        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.generate_kwargs = None

        def parameters(self):
            yield self.param

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, **kwargs):
            self.generate_kwargs = dict(kwargs)
            input_ids = kwargs["input_ids"]
            eos_column = torch.full(
                (input_ids.shape[0], 1),
                2,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            return torch.cat([input_ids, eos_column], dim=1)

    model = DummyModel()
    tokenizer = DummyTokenizer()

    rows = proposal_generation.generate_proposals_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt=PromptBundle(system="system", user="user"),
        num_candidates=2,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        batch_size=2,
    )

    assert model.training is True
    assert model.generate_kwargs["pad_token_id"] == 0
    assert model.generate_kwargs["eos_token_id"] == 2
    assert "bos_token_id" not in model.generate_kwargs
    assert [row["raw_output"] for row in rows] == ["", ""]
    assert [row["raw_output_with_special_tokens"] for row in rows] == ["<eos>", "<eos>"]
    assert [row["generated_token_count"] for row in rows] == [1, 1]
    assert [row["first_generated_token_id"] for row in rows] == [2, 2]
    assert [row["first_generated_token_text"] for row in rows] == ["<eos>", "<eos>"]
    assert rows[0]["generation_pad_token_id"] == 0
    assert rows[0]["generation_eos_token_id"] == 2
    assert rows[0]["generation_bos_token_id"] is None


def test_load_or_generate_proposal_rows_can_force_unique_config_actions(tmp_path: Path, monkeypatch):
    generated = [
        {"reasoning": "compose reliable sizes", "left": 5, "right": 3, "guard": "none"},
        {"reasoning": "same action again", "left": 5, "right": 3, "guard": "none"},
        {"reasoning": "try a different target", "left": 4, "right": 3, "guard": "none"},
    ]
    calls: list[dict] = []

    def fake_generate_proposals_from_model(**kwargs):
        calls.append(kwargs)
        return [{"id": "generated", "raw_output": json.dumps(generated.pop(0))}]

    monkeypatch.setattr(
        proposal_generation,
        "generate_proposals_from_model",
        fake_generate_proposals_from_model,
    )
    args = SimpleNamespace(
        proposal_fixture_jsonl=None,
        num_candidates=2,
        proposal_model_name="current",
        proposal_max_new_tokens=8,
        proposal_temperature=0.9,
        proposal_top_p=0.95,
        proposal_sampling_batch_size=1,
        force_unique_proposals=True,
        proposal_unique_max_draws=5,
        condition="config",
        task="addition",
        initial_min_size=1,
        initial_max_size=5,
        proposal_output_schema="action_observation",
    )
    unique_log_path = tmp_path / "proposal_unique_sampling.json"
    draw_results_path = tmp_path / "proposal_draw_results.json"

    rows = proposal_generation.load_or_generate_proposal_rows(
        args=args,
        prompt=PromptBundle(system="system", user="user"),
        current_model=None,
        current_tokenizer=None,
        round_index=2,
        source_sizes={3, 4, 5},
        frontier_min=6,
        frontier_max=9,
        unique_log_path=unique_log_path,
        draw_results_log_path=draw_results_path,
    )

    payloads = [json.loads(row["raw_output"]) for row in rows]
    assert [(payload["left"], payload["right"]) for payload in payloads] == [(5, 3), (4, 3)]
    assert [row["id"] for row in rows] == ["model_candidate_0", "model_candidate_1"]
    assert len(calls) == 3
    assert all(call["temperature"] == 0.9 for call in calls)
    summary = json.loads(unique_log_path.read_text(encoding="utf-8"))
    assert summary["requested_unique_proposals"] == 2
    assert summary["total_draws"] == 3
    assert summary["unique_valid_actions"] == 2
    assert summary["reached_requested_unique_count"] is True
    assert summary["strict_valid_unique"] is True
    assert summary["fallback_rows_returned"] == 0
    assert summary["draws"][1]["reason"] == "duplicate_valid_action"
    draw_results = json.loads(draw_results_path.read_text(encoding="utf-8"))
    assert len(draw_results) == 3
    assert draw_results[0]["kept_for_candidate"] is True
    assert draw_results[0]["candidate_proposal_index"] == 0
    assert draw_results[1]["kept_for_candidate"] is False
    assert draw_results[1]["unique_generation_reason"] == "duplicate_valid_action"


def test_force_unique_config_generation_does_not_backfill_invalid_rows(tmp_path: Path, monkeypatch):
    generated = [
        {"left": 5, "right": 3, "guard": "none"},
        {"left": 5, "right": 3, "guard": "none"},
        "not json",
    ]

    def fake_generate_proposals_from_model(**kwargs):
        raw = generated.pop(0)
        return [{"id": "generated", "raw_output": raw if isinstance(raw, str) else json.dumps(raw)}]

    monkeypatch.setattr(
        proposal_generation,
        "generate_proposals_from_model",
        fake_generate_proposals_from_model,
    )
    args = SimpleNamespace(
        proposal_fixture_jsonl=None,
        num_candidates=3,
        proposal_model_name="current",
        proposal_max_new_tokens=8,
        proposal_temperature=0.9,
        proposal_top_p=0.95,
        proposal_sampling_batch_size=1,
        force_unique_proposals=True,
        proposal_unique_max_draws=3,
        condition="config",
        task="addition",
        initial_min_size=1,
        initial_max_size=5,
        proposal_output_schema="plain",
    )
    unique_log_path = tmp_path / "proposal_unique_sampling.json"
    draw_results_path = tmp_path / "proposal_draw_results.json"

    rows = proposal_generation.load_or_generate_proposal_rows(
        args=args,
        prompt=PromptBundle(system="system", user="user"),
        current_model=None,
        current_tokenizer=None,
        round_index=2,
        source_sizes={3, 4, 5},
        frontier_min=6,
        frontier_max=9,
        unique_log_path=unique_log_path,
        draw_results_log_path=draw_results_path,
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "model_candidate_0"
    summary = json.loads(unique_log_path.read_text(encoding="utf-8"))
    assert summary["requested_unique_proposals"] == 3
    assert summary["returned_rows"] == 1
    assert summary["fallback_rows_returned"] == 0
    assert summary["reached_requested_unique_count"] is False
    assert [draw["reason"] for draw in summary["draws"]] == [
        "unique_valid_action",
        "duplicate_valid_action",
        "invalid",
    ]
    draw_results = json.loads(draw_results_path.read_text(encoding="utf-8"))
    assert [row["kept_for_candidate"] for row in draw_results] == [True, False, False]
    assert draw_results[2]["validation_category"] == "parse_error"


def test_proposal_runtime_reexports_generation_helpers_for_compatibility():
    assert proposal_runtime._rows_for_round is proposal_generation._rows_for_round
    assert proposal_runtime.generate_proposals_from_model is proposal_generation.generate_proposals_from_model
    assert proposal_runtime.load_or_generate_proposal_rows is proposal_generation.load_or_generate_proposal_rows
