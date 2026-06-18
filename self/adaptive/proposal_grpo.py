#!/usr/bin/env python3
"""Proposal-GRPO lightweight policy updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from self.core.data_io import sanitize_json_value
from self.core.model_io import instantiate_model_and_tokenizer
from self.adaptive.proposal_grpo_traces import (
    PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY,
    PROPOSAL_GRPO_REWARD_BY_CATEGORY,
    PROPOSAL_GRPO_REWARD_MODES,
    PROPOSAL_GRPO_ZERO_VARIANCE_MODES,
    ProposalGRPOTrace,
    build_proposal_grpo_traces,
    proposal_grpo_advantages,
    proposal_grpo_outcome_invalid_reward,
    proposal_grpo_reward,
    proposal_grpo_reward_for_result,
)
from self.adaptive.proposal_io import write_trace_jsonl
from self.adaptive.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]


def _encode_proposal_grpo_sample(
    *,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    completion: str,
) -> Optional[JsonDict]:
    if completion == "":
        return None
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        return None
    input_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(int(tokenizer.bos_token_id))
    input_ids.extend(int(token_id) for token_id in prompt_ids)
    completion_start = len(input_ids)
    input_ids.extend(int(token_id) for token_id in completion_ids)
    if len(input_ids) < 2:
        return None
    completion_mask = [bool(position + 1 >= completion_start) for position in range(len(input_ids) - 1)]
    if not any(completion_mask):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "completion_mask": completion_mask,
        "completion_tokens": len(completion_ids),
    }


def _collate_proposal_grpo_samples(
    *,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    device: torch.device,
) -> JsonDict:
    if not samples:
        raise ValueError("Expected at least one proposal GRPO sample.")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for proposal GRPO padding.")
    max_length = max(len(sample["input_ids"]) for sample in samples)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    completion_mask: List[List[bool]] = []
    for sample in samples:
        sample_ids = list(sample["input_ids"])
        pad_count = max_length - len(sample_ids)
        input_ids.append(sample_ids + [int(pad_token_id)] * pad_count)
        attention_mask.append(list(sample["attention_mask"]) + [0] * pad_count)
        sample_completion_mask = list(sample["completion_mask"])
        completion_mask.append(sample_completion_mask + [False] * pad_count)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.bool, device=device),
    }


def _proposal_completion_mean_logprobs(model: AutoModelForCausalLM, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["input_ids"][:, 1:]
    mask = batch["completion_mask"][:, : labels.shape[1]]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked_logprobs = token_logprobs * mask.float()
    token_counts = mask.sum(dim=1).clamp_min(1).float()
    return masked_logprobs.sum(dim=1) / token_counts


def apply_proposal_grpo_update(
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any],
    seed: int,
) -> Tuple[str, JsonDict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: JsonDict = {
        "enabled": bool(args.proposal_grpo_steps > 0),
        "skipped": True,
        "skip_reason": None,
        "source_checkpoint": source_checkpoint,
        "model_dir": None,
        "proposal_count": len(proposal_results),
        "steps": int(args.proposal_grpo_steps),
        "learning_rate": float(args.proposal_grpo_learning_rate),
        "kl_coef": float(args.proposal_grpo_kl_coef),
        "grad_clip": float(args.proposal_grpo_grad_clip),
        "zero_variance": args.proposal_grpo_zero_variance,
        "fixed_baseline": float(args.proposal_grpo_fixed_baseline),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "candidate_metric_count": len(candidate_metrics),
    }
    if args.proposal_grpo_steps <= 0:
        metrics["skip_reason"] = "disabled"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.condition != "config":
        metrics["skip_reason"] = "non_config_condition"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.proposal_model_name != "current":
        metrics["skip_reason"] = "off_policy_proposal_model"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.proposal_fixture_jsonl is not None:
        metrics["skip_reason"] = "off_policy_fixture"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if not proposal_results:
        metrics["skip_reason"] = "no_proposals"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics

    traces, trace_summary = build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
    )
    write_trace_jsonl(output_dir / "proposal_grpo_traces.jsonl", [trace.to_json_dict() for trace in traces])
    metrics.update(trace_summary)
    if trace_summary["zero_variance_skip"]:
        metrics["skip_reason"] = "zero_variance"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics

    set_seed(seed)
    model, tokenizer = instantiate_model_and_tokenizer(
        source_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        device = next(model.parameters()).device
        encoded_samples: List[JsonDict] = []
        encoded_traces: List[ProposalGRPOTrace] = []
        for trace in traces:
            sample = _encode_proposal_grpo_sample(
                tokenizer=tokenizer,
                prompt_text=trace.prompt_text,
                completion=trace.completion,
            )
            if sample is None:
                continue
            encoded_samples.append(sample)
            encoded_traces.append(trace)
        metrics["trace_count"] = len(traces)
        metrics["trainable_trace_count"] = len(encoded_samples)
        metrics["completion_token_counts"] = [int(sample["completion_tokens"]) for sample in encoded_samples]
        if not encoded_samples:
            metrics["skip_reason"] = "no_tokenizable_completions"
            write_json(output_dir / "proposal_grpo_metrics.json", metrics)
            return source_checkpoint, metrics

        batch = _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=encoded_samples, device=device)
        advantages = torch.tensor(
            [trace.advantage for trace in encoded_traces],
            dtype=torch.float32,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            old_logprobs = _proposal_completion_mean_logprobs(model, batch).detach()

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.proposal_grpo_learning_rate))
        loss_history: List[JsonDict] = []
        for step_index in range(int(args.proposal_grpo_steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            new_logprobs = _proposal_completion_mean_logprobs(model, batch)
            policy_loss = -(advantages * new_logprobs).mean()
            kl_proxy = ((new_logprobs - old_logprobs) ** 2).mean()
            loss = policy_loss + float(args.proposal_grpo_kl_coef) * kl_proxy
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.proposal_grpo_grad_clip))
            optimizer.step()
            loss_history.append(
                sanitize_json_value(
                    {
                        "step": step_index + 1,
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "kl_proxy": float(kl_proxy.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                        "mean_logprob_before": float(old_logprobs.mean().detach().cpu()),
                        "mean_logprob_after": float(new_logprobs.mean().detach().cpu()),
                    }
                )
            )

        model_dir = output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        metrics.update(
            {
                "skipped": False,
                "skip_reason": None,
                "model_dir": str(model_dir),
                "loss_history": loss_history,
                "reward_values": [trace.reward for trace in traces],
                "advantages": [trace.advantage for trace in traces],
            }
        )
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return str(model_dir), metrics
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
