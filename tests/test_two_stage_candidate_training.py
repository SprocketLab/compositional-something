from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from self.adaptive import candidate
from self.adaptive.proposal import ConfigProposal
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.core.training import TrainingConfig


def _work_item(*, pseudo_examples=None) -> CandidateWorkItem:
    proposal = ConfigProposal(left=3, right=5, guard="none", target=8)
    return CandidateWorkItem(
        index=2,
        row_id="candidate-2",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output=proposal.to_completion(),
        composed=ExactPairDataset(examples=list(range(5000)), component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=list(pseudo_examples or []),
        pseudo_diagnostics={},
        proposal_prediction={},
    )


def _metric(reward: float) -> CandidateMetrics:
    item = _work_item()
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=True,
        reward=reward,
        frontier_delta=reward,
        target_accuracy=0.7,
        current_target_accuracy=0.5,
        final_accuracy=0.6,
        init_final_accuracy=0.4,
        final_accuracy_delta=0.2,
        current_final_accuracy=0.5,
        final_accuracy_delta_from_current=reward,
        per_size_accuracy={8: 0.7},
        pseudo_count=512,
        model_dir=Path("screen") / "model",
    )


@pytest.mark.parametrize(("confirmed_accuracy", "accepted"), [(0.6, True), (0.5, False)])
def test_confirmation_retrains_from_pre_attempt_checkpoint_and_uses_strict_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirmed_accuracy: float,
    accepted: bool,
) -> None:
    item = _work_item()
    full_item = _work_item(pseudo_examples=list(range(5000)))
    observed = {}

    monkeypatch.setattr(candidate, "instantiate_model_and_tokenizer", lambda *args, **kwargs: (object(), object()))

    def _attach(**kwargs):
        observed["example_limit"] = kwargs["example_limit"]
        observed["fidelity"] = kwargs["fidelity"]
        return [full_item]

    def _train(**kwargs):
        observed["source_checkpoint"] = kwargs["source_checkpoint"]
        observed["max_steps"] = kwargs["config"].max_steps
        observed["num_epochs"] = kwargs["config"].num_epochs
        model_dir = kwargs["output_dir"] / "model"
        model_dir.mkdir(parents=True)
        return object(), object(), model_dir

    monkeypatch.setattr(candidate, "attach_pseudo_labels", _attach)
    monkeypatch.setattr(candidate, "train_checkpoint", _train)
    monkeypatch.setattr(candidate, "evaluate_model", lambda **kwargs: (confirmed_accuracy, {8: 0.7}))
    monkeypatch.setattr(candidate, "clear_cuda_cache", lambda: None)

    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        tokenizer_mode="auto",
        recipe="none",
        candidate_train_per_size=5000,
        selected_max_steps=0,
        num_epochs=1,
        selection_min_reward=0.0,
        frontier_min_size=3,
        frontier_max_size=31,
    )
    config = TrainingConfig(
        num_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=1,
        max_steps=25,
    )
    result = candidate.confirm_two_stage_candidate(
        args=args,
        task=object(),
        current_checkpoint="pre-attempt-model",
        source_examples=[],
        provisional=_metric(0.1),
        work_item=item,
        round_dir=tmp_path,
        eval_examples=[],
        current_final_accuracy=0.5,
        current_per_size_accuracy={8: 0.5},
        init_final_accuracy=0.4,
        config=config,
        seed=7,
    )

    assert result.accepted is accepted
    assert observed == {
        "example_limit": 5000,
        "fidelity": "confirmed_full",
        "source_checkpoint": "pre-attempt-model",
        "max_steps": None,
        "num_epochs": 1,
    }
    assert result.metrics.reward == pytest.approx(confirmed_accuracy - 0.5)
    assert result.work_item is full_item
    assert result.metrics.model_dir.exists() is accepted
