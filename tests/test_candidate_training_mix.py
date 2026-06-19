from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.adaptive.candidate import (
    build_candidate_training_mix,
    write_candidate_training_mix_artifacts,
)
from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateWorkItem, ExactPairDataset
from self.adaptive.proposal import ConfigProposal, PromptBundle


class _Task:
    @staticmethod
    def serialize_example(example):
        return {"value": example}


def _args(**overrides):
    values = dict(
        task="addition",
        condition="config",
        proposal_trace_replay_ratio=0.5,
        proposal_trace_replay_max_examples=2,
        post_task_proposal_rehearsal=False,
        post_task_proposal_rehearsal_repeat_count=3,
        post_task_proposal_rehearsal_max_examples=4,
        outcome_trace_target_mode="none",
        outcome_trace_replay_ratio=0.5,
        outcome_trace_replay_max_examples=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _item(pseudo_examples=None):
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    return CandidateWorkItem(
        index=0,
        row_id="candidate-0",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output={},
        composed=ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=list(pseudo_examples or ["p0", "p1", "p2"]),
        pseudo_diagnostics={},
        proposal_prediction={},
    )


def _proposal_trace(label: str) -> ProposalTraceExample:
    return ProposalTraceExample(
        prompt_text=f"prompt-{label}",
        completion=f"completion-{label}",
        task="addition",
        condition="config",
        round_index=0,
        reward=1.0,
        metadata={"label": label},
    )


def _outcome_trace(label: str) -> OutcomeTraceExample:
    return OutcomeTraceExample(
        prompt_text=f"state-{label}",
        completion=f"outcome-{label}",
        task="addition",
        condition="config",
        round_index=0,
        mode="numeric",
        reward=0.1,
        metadata={"label": label},
    )


def test_candidate_training_mix_includes_mixed_replay_when_post_task_rehearsal_disabled():
    item = _item()
    mix = build_candidate_training_mix(
        args=_args(),
        source_examples=["s0"],
        item=item,
        proposal_trace_buffer=[_proposal_trace("a"), _proposal_trace("b")],
        outcome_trace_buffer=[_outcome_trace("a")],
        proposal_prompt=PromptBundle(system="system", user="user"),
        round_index=1,
        seed=11,
    )

    assert mix.task_train_examples == ["s0", "p0", "p1", "p2"]
    assert len(mix.outcome_replay_examples) == 0
    assert len(mix.mixed_proposal_replay_examples) == 2
    assert len(mix.candidate_trace_examples) == 1
    assert mix.mixed_candidate_trace_examples == mix.candidate_trace_examples
    assert len(mix.post_task_rehearsal_examples) == 0
    assert len(mix.train_examples) == 7
    assert mix.summary_counts == {
        "task_train_examples": 4,
        "outcome_trace_replay_examples": 0,
        "proposal_trace_replay_examples": 2,
        "candidate_proposal_trace_examples": 1,
        "mixed_candidate_proposal_trace_examples": 1,
        "post_task_proposal_rehearsal_examples": 0,
        "total_train_examples": 7,
    }


def test_candidate_training_mix_separates_post_task_rehearsal_examples():
    item = _item(["p0"])
    mix = build_candidate_training_mix(
        args=_args(post_task_proposal_rehearsal=True, outcome_trace_target_mode="numeric"),
        source_examples=["s0"],
        item=item,
        proposal_trace_buffer=[_proposal_trace("selected")],
        outcome_trace_buffer=[_outcome_trace("a"), _outcome_trace("b")],
        proposal_prompt=PromptBundle(system="system", user="user"),
        round_index=1,
        seed=13,
    )

    assert mix.task_train_examples == ["s0", "p0"]
    assert len(mix.outcome_replay_examples) == 1
    assert mix.mixed_proposal_replay_examples == []
    assert mix.mixed_candidate_trace_examples == []
    assert len(mix.candidate_trace_examples) == 1
    assert len(mix.post_task_rehearsal_examples) == 4
    assert len(mix.train_examples) == 3


def test_candidate_training_mix_artifacts_match_existing_paths(tmp_path: Path):
    item = _item(["p0"])
    args = _args(post_task_proposal_rehearsal=True, outcome_trace_target_mode="numeric")
    proposal_trace_buffer = [_proposal_trace("selected")]
    outcome_trace_buffer = [_outcome_trace("a"), _outcome_trace("b")]
    mix = build_candidate_training_mix(
        args=args,
        source_examples=["s0"],
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=PromptBundle(system="system", user="user"),
        round_index=1,
        seed=13,
    )

    write_candidate_training_mix_artifacts(
        candidate_dir=tmp_path,
        task=_Task(),
        args=args,
        source_examples=["s0"],
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        mix=mix,
    )

    assert (tmp_path / "train_examples.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"value": "s0"}',
        '{"value": "p0"}',
    ]
    assert (tmp_path / "outcome_trace_replay_examples.jsonl").exists()
    assert (tmp_path / "candidate_proposal_trace_example.jsonl").exists()
    assert (tmp_path / "post_task_proposal_rehearsal_examples.jsonl").exists()
    assert not (tmp_path / "proposal_trace_replay_examples.jsonl").exists()

    summary = json.loads((tmp_path / "train_mix_summary.json").read_text(encoding="utf-8"))
    assert summary["task_train_examples"] == 2
    assert summary["source_examples"] == 1
    assert summary["pseudo_examples"] == 1
    assert summary["outcome_trace_buffer_size"] == 2
    assert summary["proposal_trace_buffer_size"] == 1
    assert summary["post_task_proposal_rehearsal"] is True
    assert summary["total_train_examples"] == len(mix.train_examples)
