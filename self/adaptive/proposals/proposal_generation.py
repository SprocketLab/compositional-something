"""Proposal-row loading and model generation for adaptive self-improvement."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self.core.evaluation import build_generation_encodings
from self.core.model_io import instantiate_model_and_tokenizer
from self.adaptive.proposals.proposal_io import load_fixture_proposals
from self.adaptive.proposals.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]


def _rows_for_round(
    rows: Sequence[Mapping[str, Any]],
    round_index: int,
    *,
    attempt_index: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    if attempt_index is not None:
        attempt_matching = [
            row for row in rows if "attempt" in row and int(row.get("attempt", -1)) == attempt_index
        ]
        if attempt_matching:
            return attempt_matching
    matching = [row for row in rows if int(row.get("round", round_index)) == round_index]
    return matching if matching else list(rows)


def generate_proposals_from_model(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: PromptBundle,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[JsonDict]:
    device = next(model.parameters()).device
    encodings = build_generation_encodings(tokenizer, [prompt.text()], device)
    rows: List[JsonDict] = []
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for idx in range(num_candidates):
            generation_kwargs: Dict[str, Any] = {
                **encodings,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
            prompt_width = encodings["input_ids"].shape[1]
            decoded = tokenizer.decode(output_ids[0, prompt_width:].tolist(), skip_special_tokens=True)
            rows.append({"id": f"model_candidate_{idx}", "raw_output": decoded})
    if model_was_training:
        model.train()
    return rows


def load_or_generate_proposal_rows(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    current_model: AutoModelForCausalLM,
    current_tokenizer: AutoTokenizer,
    round_index: int,
    attempt_index: Optional[int] = None,
) -> List[JsonDict]:
    if args.proposal_fixture_jsonl is not None:
        rows = _rows_for_round(
            load_fixture_proposals(args.proposal_fixture_jsonl),
            round_index,
            attempt_index=attempt_index,
        )
        return [dict(row) for row in rows[: args.num_candidates]]

    if args.proposal_model_name == "current":
        return generate_proposals_from_model(
            model=current_model,
            tokenizer=current_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )

    proposal_model, proposal_tokenizer = instantiate_model_and_tokenizer(
        args.proposal_model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe="none",
    )
    try:
        return generate_proposals_from_model(
            model=proposal_model,
            tokenizer=proposal_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )
    finally:
        del proposal_model
        del proposal_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
