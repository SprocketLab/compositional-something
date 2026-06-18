#!/usr/bin/env python3
"""Compatibility exports for adaptive proposal helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from self.adaptive.proposals.proposal_config_schema import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    PROPOSAL_OUTPUT_SCHEMAS,
    ConfigProposal,
    ProposalValidation,
    extract_json_object,
    normalized_config_completion,
    parse_config_proposal,
    proposal_output_schema,
    proposal_payload_for_schema,
    validate_config_prediction,
)
from self.adaptive.proposals.proposal_io import build_trace_row, load_fixture_proposals, write_trace_jsonl
from self.adaptive.proposals.proposal_prompts import (
    PromptBundle,
    render_config_prompt,
    render_program_prompt,
    render_program_repair_prompt,
)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ProgramProposal:
    proposal_type: str
    task: str
    code: str

    def to_json_dict(self) -> JsonDict:
        return asdict(self)

    def to_completion(self) -> str:
        return self.code
