from __future__ import annotations

from self.adaptive import proposal as proposal_grpo, proposal as proposal_grpo_traces


def test_proposal_grpo_trace_owner_reexports() -> None:
    assert proposal_grpo.ProposalGRPOTrace is proposal_grpo_traces.ProposalGRPOTrace
    assert proposal_grpo.build_proposal_grpo_traces is proposal_grpo_traces.build_proposal_grpo_traces
    assert proposal_grpo.proposal_grpo_reward is proposal_grpo_traces.proposal_grpo_reward
    assert proposal_grpo.proposal_grpo_advantages is proposal_grpo_traces.proposal_grpo_advantages
    assert (
        proposal_grpo.proposal_grpo_reward_for_result
        is proposal_grpo_traces.proposal_grpo_reward_for_result
    )
