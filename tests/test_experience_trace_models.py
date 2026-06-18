from __future__ import annotations

import math
import random
from types import SimpleNamespace

from self.adaptive.traces import traces
from self.adaptive.traces.traces import (
    OutcomeTraceExample,
    ProposalTraceExample,
    build_post_task_proposal_rehearsal_examples,
    outcome_trace_from_json,
    proposal_trace_from_json,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)


def test_trace_example_json_roundtrip_and_legacy_identity():
    proposal = ProposalTraceExample(
        prompt_text="prompt",
        completion='{"left":1}',
        task="addition",
        condition="config",
        round_index=3,
        reward=1.25,
        metadata={"target": 4},
    )
    outcome = OutcomeTraceExample(
        prompt_text="state",
        completion='{"reward":0.5}',
        task="addition",
        condition="config",
        round_index=4,
        mode="numeric",
        reward=0.5,
        metadata={"target": "7"},
    )

    assert traces.ProposalTraceExample is ProposalTraceExample
    assert traces.OutcomeTraceExample is OutcomeTraceExample
    assert traces.proposal_trace_from_json is proposal_trace_from_json
    assert traces.outcome_trace_from_json is outcome_trace_from_json

    proposal_roundtrip = proposal_trace_from_json(proposal.to_json_dict())
    outcome_roundtrip = outcome_trace_from_json(outcome.to_json_dict())

    assert proposal_roundtrip == proposal
    assert outcome_roundtrip == outcome
    assert outcome_roundtrip.size_for_batching() == 7
    assert math.isnan(proposal_trace_from_json({"reward": "not-a-number"}).reward)


def test_trace_replay_sampling_respects_ratio_caps_and_modes():
    proposals = [
        ProposalTraceExample("p0", "c0", "task", "config", 0, 0.1, {}),
        ProposalTraceExample("p1", "c1", "task", "config", 0, 0.2, {}),
    ]
    outcomes = [
        OutcomeTraceExample("s0", "o0", "task", "config", 0, "numeric", 0.1, {}),
        OutcomeTraceExample("s1", "o1", "task", "config", 0, "numeric", 0.2, {}),
    ]
    args = SimpleNamespace(
        proposal_trace_replay_ratio=0.5,
        proposal_trace_replay_max_examples=3,
        post_task_proposal_rehearsal=True,
        post_task_proposal_rehearsal_repeat_count=3,
        post_task_proposal_rehearsal_max_examples=5,
        outcome_trace_target_mode="numeric",
        outcome_trace_replay_ratio=0.4,
        outcome_trace_replay_max_examples=2,
    )

    proposal_replay = sample_proposal_trace_replay(
        args=args,
        trace_buffer=proposals,
        task_train_count=5,
        rng=random.Random(0),
    )
    outcome_replay = sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcomes,
        task_train_count=5,
        rng=random.Random(1),
    )
    rehearsal = build_post_task_proposal_rehearsal_examples(
        args=args,
        proposal_trace_buffer=proposals[:1],
        candidate_trace_examples=proposals[1:],
        rng=random.Random(2),
    )

    assert len(proposal_replay) == 3
    assert all(trace in proposals for trace in proposal_replay)
    assert len(outcome_replay) == 2
    assert all(trace in outcomes for trace in outcome_replay)
    assert len(rehearsal) == 5
    assert all(trace in proposals for trace in rehearsal)

    args.outcome_trace_target_mode = "none"
    assert sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcomes,
        task_train_count=5,
        rng=random.Random(3),
    ) == []
