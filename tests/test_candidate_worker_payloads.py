from __future__ import annotations

from pathlib import Path

from self.core import candidate_execution, candidate_worker_payloads, worker_io
from self.core.candidate_worker_payloads import (
    candidate_payload_from_work_item,
    candidate_payload_to_work_item,
    work_item_from_worker_payload,
    work_item_to_worker_payload,
)
from self.core.data_io import save_examples
from self.core.models import CandidateWorkItem, ExactPairDataset
from self.core.proposals import ConfigProposal


class _Task:
    @staticmethod
    def deserialize_example(payload):
        return ("loaded", payload["value"])


def _work_item() -> CandidateWorkItem:
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5, notes="candidate")
    return CandidateWorkItem(
        index=7,
        row_id="row-7",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output={"raw": "text"},
        composed=ExactPairDataset(
            examples=["composed-a", "composed-b"],
            component_map={},
            keys={(2, "left"), 5},
            diagnostics={"retained": 2},
        ),
        pseudo_examples=[{"value": "pseudo-a"}, {"value": "pseudo-b"}],
        pseudo_diagnostics={"retained_total": 2},
        proposal_prediction={"target": 5, "expected_frontier_delta": 0.2},
    )


def test_work_item_to_worker_payload_includes_paths_counts_and_composed_keys(tmp_path: Path):
    item = _work_item()
    round_dir = tmp_path / "attempt_0001"

    payload = work_item_to_worker_payload(item=item, round_dir=round_dir)

    assert payload["index"] == 7
    assert payload["row_id"] == "row-7"
    assert payload["proposal"] == {
        "left": 2,
        "right": 3,
        "guard": "none",
        "target": 5,
        "notes": "candidate",
    }
    assert payload["completion"] == item.completion
    assert payload["raw_output"] == {"raw": "text"}
    assert payload["proposal_prediction"] == {"target": 5, "expected_frontier_delta": 0.2}
    assert payload["pseudo_diagnostics"] == {"retained_total": 2}
    assert payload["pseudo_examples_path"] == str(
        round_dir / "candidates" / "candidate_07" / "pseudo_examples.jsonl"
    )
    assert payload["pseudo_count"] == 2
    assert payload["composed_count"] == 2
    assert payload["composed_keys"] == [
        worker_io.json_ready_key(key) for key in sorted(item.composed.keys, key=repr)
    ]


def test_work_item_from_worker_payload_loads_pseudo_examples_and_composed_keys(tmp_path: Path):
    item = _work_item()
    payload = work_item_to_worker_payload(item=item, round_dir=tmp_path / "attempt_0001")
    save_examples(
        Path(payload["pseudo_examples_path"]),
        [{"value": "pseudo-a"}, {"value": "pseudo-b"}],
        lambda example: example,
    )

    restored = work_item_from_worker_payload(payload=payload, task=_Task())

    assert restored.index == item.index
    assert restored.row_id == item.row_id
    assert restored.proposal == item.proposal
    assert restored.completion == item.completion
    assert restored.raw_output == item.raw_output
    assert restored.proposal_prediction == item.proposal_prediction
    assert restored.pseudo_diagnostics == item.pseudo_diagnostics
    assert restored.pseudo_examples == [("loaded", "pseudo-a"), ("loaded", "pseudo-b")]
    assert restored.composed.keys == item.composed.keys
    assert restored.composed.examples == []
    assert restored.composed.component_map == {}


def test_candidate_payload_roundtrip_for_worker_specs():
    item = _work_item()

    payload = candidate_payload_from_work_item(item)
    restored = candidate_payload_to_work_item(payload=payload, pseudo_examples=["pseudo"])

    assert payload == {
        "index": 7,
        "row_id": "row-7",
        "proposal": {
            "left": 2,
            "right": 3,
            "guard": "none",
            "target": 5,
            "notes": "candidate",
        },
        "completion": item.completion,
        "raw_output": {"raw": "text"},
        "proposal_prediction": {"target": 5, "expected_frontier_delta": 0.2},
        "pseudo_diagnostics": {"retained_total": 2},
    }
    assert restored.index == item.index
    assert restored.row_id == item.row_id
    assert restored.proposal == item.proposal
    assert restored.pseudo_examples == ["pseudo"]
    assert restored.composed.keys == set()


def test_candidate_execution_reexports_payload_helpers():
    assert candidate_execution.work_item_to_worker_payload is candidate_worker_payloads.work_item_to_worker_payload
    assert candidate_execution.work_item_from_worker_payload is candidate_worker_payloads.work_item_from_worker_payload
