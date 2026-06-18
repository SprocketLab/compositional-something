#!/usr/bin/env python3
"""Proposal-GRPO reward shaping and lightweight policy updates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from self.core.data_io import sanitize_json_value
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.proposal_io import write_trace_jsonl
from self.core.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]

PROPOSAL_GRPO_ZERO_VARIANCE_MODES = ("fixed_baseline", "skip")
PROPOSAL_GRPO_REWARD_MODES = ("outcome", "validity")
PROPOSAL_GRPO_REWARD_BY_CATEGORY: Dict[str, float] = {
    "valid": 1.0,
    "range_error": 0.6,
    "enum_error": 0.5,
    "schema_error": 0.25,
    "parse_error": 0.0,
}
PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY: Dict[str, float] = {
    "range_error": -0.4,
    "enum_error": -0.5,
    "schema_error": -0.7,
    "parse_error": -1.0,
}


@dataclass(frozen=True)
class ProposalGRPOTrace:
    """Raw proposal completion and shaped reward used for one GRPO update."""

    proposal_index: int
    proposal_id: Optional[str]
    prompt_text: str
    completion: str
    reward: float
    advantage: float
    validation_category: str
    validation_message: str
    valid: bool
    metadata: JsonDict

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "proposal_index": self.proposal_index,
                "proposal_id": self.proposal_id,
                "prompt": self.prompt_text,
                "completion": self.completion,
                "reward": self.reward,
                "advantage": self.advantage,
                "validation_category": self.validation_category,
                "validation_message": self.validation_message,
                "valid": self.valid,
                "metadata": self.metadata,
            }
        )


def proposal_grpo_reward(result: Mapping[str, Any]) -> float:
    if bool(result.get("valid")):
        return PROPOSAL_GRPO_REWARD_BY_CATEGORY["valid"]
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_REWARD_BY_CATEGORY.get(category, 0.0))


def proposal_grpo_outcome_invalid_reward(result: Mapping[str, Any]) -> float:
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY.get(category, -1.0))


def _proposal_index(result: Mapping[str, Any], default: int) -> int:
    try:
        return int(result.get("proposal_index", default))
    except (TypeError, ValueError):
        return default


def _is_system_candidate_failure(metric: Any) -> bool:
    if metric.valid:
        return False
    reason = str(metric.failure_reason or "").lower()
    if reason == "no pseudo labels retained":
        return False
    return bool(reason)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def proposal_grpo_reward_for_result(
    result: Mapping[str, Any],
    *,
    metric: Optional[Any],
    reward_mode: str,
    outcome_scale: float,
) -> Tuple[Optional[float], str]:
    if reward_mode == "validity":
        return proposal_grpo_reward(result), "validity"
    if reward_mode != "outcome":
        raise ValueError(f"Unsupported proposal_grpo_reward_mode={reward_mode!r}.")
    if not bool(result.get("valid")):
        return proposal_grpo_outcome_invalid_reward(result), "invalid"
    if metric is None:
        return 0.0, "valid_untrained"
    if not metric.valid:
        if _is_system_candidate_failure(metric):
            return None, "skipped_system_failure"
        return 0.0, "valid_untrained"
    return _clamp(float(metric.reward) / float(outcome_scale), -1.0, 1.0), "outcome"


def proposal_grpo_advantages(
    rewards: Sequence[float],
    *,
    zero_variance: str,
    fixed_baseline: float,
    eps: float = 1e-6,
) -> Tuple[List[float], bool, str]:
    if not rewards:
        return [], True, "no_rewards"
    reward_values = [float(reward) for reward in rewards]
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((reward - mean_reward) ** 2 for reward in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)
    if std_reward > eps:
        return [(reward - mean_reward) / (std_reward + eps) for reward in reward_values], False, "normalized"
    if zero_variance == "skip":
        return [0.0 for _ in reward_values], True, "zero_variance"
    if zero_variance != "fixed_baseline":
        raise ValueError(f"Unsupported proposal_grpo_zero_variance={zero_variance!r}.")
    return [reward - fixed_baseline for reward in reward_values], False, "fixed_baseline"


def build_proposal_grpo_traces(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any] = (),
) -> Tuple[List[ProposalGRPOTrace], JsonDict]:
    prompt_text = prompt.text()
    metrics_by_index = {int(metric.index): metric for metric in candidate_metrics}
    rewards: List[float] = []
    included_results: List[Tuple[int, Mapping[str, Any], Optional[Any], float, str]] = []
    reward_source_counts: Dict[str, int] = {}
    skipped_system_failure_count = 0
    for index, result in enumerate(proposal_results):
        proposal_index = _proposal_index(result, index)
        metric = metrics_by_index.get(proposal_index)
        reward, reward_source = proposal_grpo_reward_for_result(
            result,
            metric=metric,
            reward_mode=args.proposal_grpo_reward_mode,
            outcome_scale=args.proposal_grpo_outcome_scale,
        )
        reward_source_counts[reward_source] = reward_source_counts.get(reward_source, 0) + 1
        if reward is None:
            skipped_system_failure_count += 1
            continue
        rewards.append(float(reward))
        included_results.append((proposal_index, result, metric, float(reward), reward_source))
    advantages, skip_update, advantage_mode = proposal_grpo_advantages(
        rewards,
        zero_variance=args.proposal_grpo_zero_variance,
        fixed_baseline=args.proposal_grpo_fixed_baseline,
    )
    traces: List[ProposalGRPOTrace] = []
    for proposal_index, result, metric, reward, reward_source, advantage in (
        (*payload, advantage)
        for payload, advantage in zip(included_results, advantages)
    ):
        raw = result.get("raw_output", "")
        normalized_completion = result.get("completion")
        if result.get("valid") and isinstance(normalized_completion, str) and normalized_completion:
            completion = normalized_completion
            completion_source = "normalized"
        else:
            completion = raw if isinstance(raw, str) else json.dumps(sanitize_json_value(raw), sort_keys=True)
            completion_source = "raw"
        traces.append(
            ProposalGRPOTrace(
                proposal_index=proposal_index,
                proposal_id=str(result["id"]) if result.get("id") is not None else None,
                prompt_text=prompt_text,
                completion=completion,
                reward=float(reward),
                advantage=float(advantage),
                validation_category=str(
                    result.get("validation_category") or ("valid" if result.get("valid") else "unknown")
                ),
                validation_message=str(result.get("validation_message") or ""),
                valid=bool(result.get("valid")),
                metadata={
                    "duplicate": bool(result.get("duplicate")),
                    "repeat_target": bool(result.get("repeat_target")),
                    "parsed_proposal": result.get("parsed_proposal"),
                    "parsed_prediction": result.get("parsed_prediction"),
                    "proposal_output_schema": result.get("proposal_output_schema"),
                    "completion_source": completion_source,
                    "reward_source": reward_source,
                    "candidate_reward": metric.reward if metric is not None else None,
                    "frontier_delta": metric.frontier_delta if metric is not None else None,
                    "target_delta": metric.target_delta if metric is not None else None,
                    "proposal_prediction": metric.proposal_prediction if metric is not None else None,
                },
            )
        )
    mean_reward = sum(rewards) / len(rewards) if rewards else math.nan
    reward_std = (
        math.sqrt(sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards))
        if rewards
        else math.nan
    )
    return traces, {
        "reward_mean": mean_reward,
        "reward_std": reward_std,
        "advantage_mode": advantage_mode,
        "zero_variance_skip": bool(skip_update),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "reward_source_counts": reward_source_counts,
        "skipped_system_failure_count": skipped_system_failure_count,
        "input_proposal_count": len(proposal_results),
        "trace_candidate_metric_count": len(candidate_metrics),
    }


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
