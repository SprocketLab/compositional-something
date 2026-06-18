"""Proposal generation, validation, prompting, and GRPO modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from self.adaptive.proposal_config_schema import (
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
from self.adaptive.proposal_io import build_trace_row, load_fixture_proposals, write_trace_jsonl
from self.adaptive.proposal_prompts import (
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


__all__ = [
    "DEFAULT_CONFIG_SEARCH_SPACES",
    "PROPOSAL_OUTPUT_SCHEMAS",
    "ConfigProposal",
    "JsonDict",
    "ProgramProposal",
    "PromptBundle",
    "ProposalValidation",
    "build_trace_row",
    "extract_json_object",
    "load_fixture_proposals",
    "normalized_config_completion",
    "parse_config_proposal",
    "proposal_output_schema",
    "proposal_payload_for_schema",
    "render_config_prompt",
    "render_program_prompt",
    "render_program_repair_prompt",
    "validate_config_prediction",
    "write_trace_jsonl",
]
