"""Runtime proposal generation and validation for adaptive self-improvement."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.core.proposal_executable_validation import (
    _extract_python_code,
    _repair_program_with_model,
    _row_payload,
    _row_repair_output,
    validate_executable_rows,
)
from self.core.proposal_generation import (
    _rows_for_round,
    generate_proposals_from_model,
    load_or_generate_proposal_rows,
)
from self.core.proposal_config_validation import _raw_output, validate_config_rows
from self.core.proposal_prompts import (
    choose_default_program_pair,
    component_prediction_examples_for_task,
    program_validation_cases,
    render_program_candidate_prompt,
    target_format_for_task,
)
from self.core.proposals import ConfigProposal, render_program_repair_prompt

JsonDict = Dict[str, Any]


def validate_proposal_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    default_pair: Optional[ConfigProposal],
    current_model: Optional[Any] = None,
    current_tokenizer: Optional[Any] = None,
) -> List[JsonDict]:
    if args.condition == "config":
        return validate_config_rows(
            rows=rows,
            args=args,
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
        )
    return validate_executable_rows(
        rows=rows,
        args=args,
        source_sizes=source_sizes,
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        default_pair=default_pair,
        current_model=current_model,
        current_tokenizer=current_tokenizer,
    )
