#!/usr/bin/env python3
"""Probe whether proposal generation responds to changed state fields.

The diagnostic keeps the prompt template fixed, perturbs the state JSON blocks
inside a saved proposal prompt, and measures both sampled actions and normalized
action log-likelihood rankings under one proposal checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from self.adaptive.proposal import (
    ConfigProposal,
    PromptBundle,
    generate_proposals_from_model,
    normalized_config_completion,
    validate_config_rows,
)
from self.core.data_io import sanitize_json_value
from self.core.model_io import instantiate_model_and_tokenizer


JsonDict = dict[str, Any]
AGGREGATE_RE = re.compile(
    r"Aggregate diagnostics from prior evaluation:\n(?P<payload>\{.*?\})\n\nObjective:",
    re.DOTALL,
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _extract_line_json(user: str, field: str) -> JsonDict:
    pattern = re.compile(rf"^- {re.escape(field)}: (?P<payload>\{{.*\}})$", re.MULTILINE)
    match = pattern.search(user)
    if match is None:
        raise ValueError(f"Could not find prompt field {field!r}.")
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt field {field!r} is not a JSON object.")
    return dict(payload)


def _replace_line_json(user: str, field: str, payload: Mapping[str, Any]) -> str:
    pattern = re.compile(rf"^- {re.escape(field)}: \{{.*\}}$", re.MULTILINE)
    replacement = f"- {field}: {json.dumps(sanitize_json_value(dict(payload)), sort_keys=True)}"
    updated, count = pattern.subn(replacement, user)
    if count != 1:
        raise ValueError(f"Expected to replace exactly one prompt field {field!r}, replaced {count}.")
    return updated


def _extract_aggregate(user: str) -> JsonDict:
    match = AGGREGATE_RE.search(user)
    if match is None:
        raise ValueError("Could not find aggregate diagnostics JSON block.")
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict):
        raise ValueError("Aggregate diagnostics payload is not a JSON object.")
    return dict(payload)


def _replace_aggregate(user: str, payload: Mapping[str, Any]) -> str:
    replacement = (
        "Aggregate diagnostics from prior evaluation:\n"
        f"{json.dumps(sanitize_json_value(dict(payload)), sort_keys=True, indent=2)}\n\n"
        "Objective:"
    )
    updated, count = AGGREGATE_RE.subn(replacement, user)
    if count != 1:
        raise ValueError(f"Expected to replace one aggregate diagnostics block, replaced {count}.")
    return updated


def _prompt_bundle(prompt_json: Mapping[str, Any], *, user: str | None = None) -> PromptBundle:
    return PromptBundle(system=str(prompt_json["system"]), user=str(prompt_json["user"] if user is None else user))


def _valid_pairs(source_sizes: Sequence[int], frontier: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    sizes = sorted({int(size) for size in source_sizes})
    frontier_min = int(frontier["min"])
    frontier_max = int(frontier["max"])
    pairs: list[tuple[int, int, int]] = []
    for left in sizes:
        for right in sizes:
            target = left + right
            if frontier_min <= target <= frontier_max:
                pairs.append((left, right, target))
    return pairs


def _observed_action(run_dir: Path, attempt: int) -> tuple[int, int, int, str] | None:
    proposal_path = run_dir / f"attempt_{attempt:04d}" / "proposal_results.json"
    if not proposal_path.exists():
        return None
    for row in _read_json(proposal_path):
        parsed = row.get("parsed_proposal") if isinstance(row, Mapping) else None
        if bool(row.get("valid")) and isinstance(parsed, Mapping):
            return (
                int(parsed["left"]),
                int(parsed["right"]),
                int(parsed.get("target", int(parsed["left"]) + int(parsed["right"]))),
                str(parsed.get("guard", "none")),
            )
    return None


def _choose_alt_pairs(
    pairs: Sequence[tuple[int, int, int]],
    observed: tuple[int, int, int, str] | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not pairs:
        raise ValueError("No valid source pairs are available for the prompt frontier.")
    observed_target = observed[2] if observed is not None else None
    non_observed = [pair for pair in pairs if pair[2] != observed_target] or list(pairs)
    return min(non_observed, key=lambda item: (item[2], item[0], item[1])), max(
        non_observed,
        key=lambda item: (item[2], item[0], item[1]),
    )


def _accuracy_pref_payload(
    *,
    aggregate: Mapping[str, Any],
    source_sizes: Sequence[int],
    frontier: Mapping[str, Any],
    preferred_pair: tuple[int, int, int],
    observed: tuple[int, int, int, str] | None,
) -> JsonDict:
    payload: JsonDict = dict(aggregate)
    current = payload.get("per_size_accuracy")
    if not isinstance(current, Mapping):
        current = {}
    all_sizes = sorted(
        {
            int(size)
            for size in list(source_sizes)
            + list(range(int(frontier["min"]), int(frontier["max"]) + 1))
            + [preferred_pair[0], preferred_pair[1], preferred_pair[2]]
        }
    )
    acc = {str(size): 0.15 for size in all_sizes}
    left, right, target = preferred_pair
    acc[str(left)] = 0.99
    acc[str(right)] = 0.99
    acc[str(target)] = 0.97
    if observed is not None:
        for size in observed[:3]:
            acc[str(int(size))] = min(acc.get(str(int(size)), 0.15), 0.02)
    payload["per_size_accuracy"] = acc
    payload["current_avg_accuracy"] = 0.25
    return payload


def build_variants(prompt_json: Mapping[str, Any], run_dir: Path, attempt: int) -> list[JsonDict]:
    user = str(prompt_json["user"])
    source = _extract_line_json(user, "current_source_slices")
    frontier = _extract_line_json(user, "allowed_target_frontier")
    aggregate = _extract_aggregate(user)
    source_sizes = [int(size) for size in source.get("sizes", aggregate.get("source_sizes", []))]
    pairs = _valid_pairs(source_sizes, frontier)
    observed = _observed_action(run_dir, attempt)
    alt_low, alt_high = _choose_alt_pairs(pairs, observed)

    def make_variant(
        name: str,
        *,
        source_payload: Mapping[str, Any] = source,
        frontier_payload: Mapping[str, Any] = frontier,
        aggregate_payload: Mapping[str, Any] = aggregate,
        preferred_pair: tuple[int, int, int] | None = None,
    ) -> JsonDict:
        variant_user = user
        variant_user = _replace_line_json(variant_user, "current_source_slices", source_payload)
        variant_user = _replace_line_json(variant_user, "allowed_target_frontier", frontier_payload)
        variant_user = _replace_aggregate(variant_user, aggregate_payload)
        return {
            "name": name,
            "prompt": _prompt_bundle(prompt_json, user=variant_user),
            "source": dict(source_payload),
            "frontier": dict(frontier_payload),
            "aggregate": dict(aggregate_payload),
            "preferred_pair": preferred_pair,
        }

    variants = [
        make_variant("original"),
        make_variant(
            f"frontier_low_target_{alt_low[2]}",
            frontier_payload={"min": alt_low[2], "max": alt_low[2]},
            preferred_pair=alt_low,
        ),
        make_variant(
            f"frontier_high_target_{alt_high[2]}",
            frontier_payload={"min": alt_high[2], "max": alt_high[2]},
            preferred_pair=alt_high,
        ),
        make_variant(
            f"accuracy_prefers_{alt_high[0]}_{alt_high[1]}_{alt_high[2]}",
            aggregate_payload=_accuracy_pref_payload(
                aggregate=aggregate,
                source_sizes=source_sizes,
                frontier=frontier,
                preferred_pair=alt_high,
                observed=observed,
            ),
            preferred_pair=alt_high,
        ),
        make_variant(
            f"frontier_and_accuracy_prefers_{alt_high[0]}_{alt_high[1]}_{alt_high[2]}",
            frontier_payload={"min": alt_high[2], "max": alt_high[2]},
            aggregate_payload=_accuracy_pref_payload(
                aggregate=aggregate,
                source_sizes=source_sizes,
                frontier={"min": alt_high[2], "max": alt_high[2]},
                preferred_pair=alt_high,
                observed=observed,
            ),
            preferred_pair=alt_high,
        ),
    ]
    for variant in variants:
        variant["observed_action"] = observed
    return variants


def _action_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    parsed = row.get("parsed_proposal")
    if not isinstance(parsed, Mapping):
        return None
    return (
        int(parsed["left"]),
        int(parsed["right"]),
        int(parsed.get("target", int(parsed["left"]) + int(parsed["right"]))),
        str(parsed.get("guard", "none")),
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    source_sizes: Sequence[int],
    frontier: Mapping[str, Any],
) -> list[JsonDict]:
    return validate_config_rows(
        rows=rows,
        args=args,
        source_sizes={int(size) for size in source_sizes},
        frontier_min=int(frontier["min"]),
        frontier_max=int(frontier["max"]),
    )


def _sample_actions(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptBundle,
    args: argparse.Namespace,
    source_sizes: Sequence[int],
    frontier: Mapping[str, Any],
    num_samples: int,
    temperature: float,
) -> JsonDict:
    if num_samples <= 0:
        return {
            "temperature": float(temperature),
            "num_samples": int(num_samples),
            "valid_count": 0,
            "action_counts": [],
            "category_counts": {},
            "raw_results": [],
        }
    rows = generate_proposals_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_candidates=num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
    )
    results = _validate_rows(rows, args=args, source_sizes=source_sizes, frontier=frontier)
    action_counts: Counter[tuple[Any, ...]] = Counter()
    category_counts: Counter[str] = Counter()
    for result in results:
        category = str(result.get("validation_category") or ("valid" if result.get("valid") else "unknown"))
        category_counts[category] += 1
        if result.get("valid"):
            key = _action_key(result)
            if key is not None:
                action_counts[key] += 1
    return {
        "num_samples": num_samples,
        "temperature": temperature,
        "valid_count": sum(action_counts.values()),
        "category_counts": dict(category_counts),
        "action_counts": [
            {"action": list(key), "count": count}
            for key, count in action_counts.most_common()
        ],
        "raw_results": results,
    }


def _encode_logprob_sample(tokenizer: Any, prompt_text: str, completion: str) -> JsonDict | None:
    from self.adaptive.proposal import _encode_proposal_grpo_sample

    return _encode_proposal_grpo_sample(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        completion=completion,
        completion_char_span=None,
    )


def _mean_logprobs_for_actions(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptBundle,
    actions: Sequence[tuple[int, int, int, str]],
    top_k: int,
    batch_size: int,
) -> list[JsonDict]:
    from self.adaptive.proposal import (
        _collate_proposal_grpo_samples,
        _proposal_completion_mean_logprobs,
    )

    device = next(model.parameters()).device
    samples: list[JsonDict] = []
    sample_actions: list[tuple[int, int, int, str]] = []
    prompt_text = prompt.text()
    for left, right, target, guard in actions:
        proposal = ConfigProposal(
            left=int(left),
            right=int(right),
            target=int(target),
            guard=str(guard),
            notes=(
                f"Source sizes {left} and {right} look reliable, target {target} is in the frontier, "
                f"and this guard matches the expected boundary behavior."
            ),
        )
        completion = normalized_config_completion(
            proposal=proposal,
            prediction=None,
            schema="action_observation",
        )
        sample = _encode_logprob_sample(tokenizer, prompt_text, completion)
        if sample is None:
            continue
        samples.append(sample)
        sample_actions.append((left, right, target, guard))
    if not samples:
        return []
    model.eval()
    all_logprobs: list[float] = []
    with torch.no_grad():
        safe_batch_size = max(1, int(batch_size))
        for start in range(0, len(samples), safe_batch_size):
            batch = _collate_proposal_grpo_samples(
                tokenizer=tokenizer,
                samples=samples[start : start + safe_batch_size],
                device=device,
            )
            logprobs = _proposal_completion_mean_logprobs(model, batch).detach().float().cpu().tolist()
            all_logprobs.extend(float(score) for score in logprobs)
    rows = [
        {"action": list(action), "mean_logprob": float(score)}
        for action, score in zip(sample_actions, all_logprobs)
    ]
    rows.sort(key=lambda row: row["mean_logprob"], reverse=True)
    return rows[:top_k]


def _candidate_actions(source_sizes: Sequence[int], frontier: Mapping[str, Any], guard: str = "none") -> list[tuple[int, int, int, str]]:
    return [(left, right, target, guard) for left, right, target in _valid_pairs(source_sizes, frontier)]


def _bounded_actions(
    actions: Sequence[tuple[int, int, int, str]],
    *,
    observed_action: Sequence[Any] | None,
    preferred_pair: Sequence[Any] | None,
    max_count: int,
) -> list[tuple[int, int, int, str]]:
    if max_count <= 0 or len(actions) <= max_count:
        return list(actions)
    action_set = set(actions)
    selected: list[tuple[int, int, int, str]] = []

    def add(action: tuple[int, int, int, str]) -> None:
        if action in action_set and action not in selected:
            selected.append(action)

    if observed_action is not None and len(observed_action) >= 4:
        add((int(observed_action[0]), int(observed_action[1]), int(observed_action[2]), str(observed_action[3])))
    if preferred_pair is not None and len(preferred_pair) >= 3:
        add((int(preferred_pair[0]), int(preferred_pair[1]), int(preferred_pair[2]), "none"))

    slots = max_count - len(selected)
    if slots <= 0:
        return selected[:max_count]
    if slots == 1:
        add(actions[len(actions) // 2])
        return selected[:max_count]
    for offset in range(slots):
        index = round(offset * (len(actions) - 1) / (slots - 1))
        add(actions[index])
        if len(selected) >= max_count:
            break
    return selected[:max_count]


def run_diagnostic(args: argparse.Namespace) -> JsonDict:
    run_dir = args.run_dir.resolve()
    summary = _read_json(run_dir / "summary.json")
    args.task = str(summary["task"])
    args.condition = "config"
    args.proposal_output_schema = "action_observation"
    args.initial_min_size = 0
    args.initial_max_size = 10**9
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else Path(summary["current_checkpoint"]).resolve()
    prompt_json = _read_json(run_dir / f"attempt_{args.attempt:04d}" / "proposal_prompt.json")
    variants = build_variants(prompt_json, run_dir, args.attempt)

    model, tokenizer = instantiate_model_and_tokenizer(
        str(checkpoint),
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        model_max_length = int(getattr(tokenizer, "model_max_length", 0) or 0)
        config_max_positions = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
        effective_max_length = config_max_positions or model_max_length
        variant_rows: list[JsonDict] = []
        for variant in variants:
            prompt = variant["prompt"]
            source_sizes = [int(size) for size in variant["source"]["sizes"]]
            frontier = variant["frontier"]
            prompt_tokens = len(tokenizer.encode(prompt.text(), add_special_tokens=False))
            candidate_actions = _candidate_actions(source_sizes, frontier)
            bounded_actions = _bounded_actions(
                candidate_actions,
                observed_action=variant.get("observed_action"),
                preferred_pair=variant.get("preferred_pair"),
                max_count=args.max_logprob_candidate_actions,
            )
            greedy = _sample_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                args=args,
                source_sizes=source_sizes,
                frontier=frontier,
                num_samples=1,
                temperature=0.0,
            )
            sampled = _sample_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                args=args,
                source_sizes=source_sizes,
                frontier=frontier,
                num_samples=args.num_samples,
                temperature=args.temperature,
            )
            logprob_rank = _mean_logprobs_for_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                actions=bounded_actions,
                top_k=args.top_logprob_actions,
                batch_size=args.logprob_batch_size,
            )
            preferred = variant.get("preferred_pair")
            preferred_action = list(preferred) + ["none"] if preferred is not None else None
            variant_rows.append(
                {
                    "name": variant["name"],
                    "preferred_action": preferred_action,
                    "observed_action": list(variant["observed_action"]) if variant.get("observed_action") else None,
                    "source_sizes": source_sizes,
                    "frontier": dict(frontier),
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "effective_max_length": effective_max_length,
                    "fits_context": (
                        True if effective_max_length <= 0 else prompt_tokens + args.max_new_tokens <= effective_max_length
                    ),
                    "logprob_candidate_count": len(bounded_actions),
                    "logprob_candidate_count_total": len(candidate_actions),
                    "greedy": greedy,
                    "sampled": sampled,
                    "logprob_top_actions": logprob_rank,
                }
            )
        return {
            "run_dir": str(run_dir),
            "task": summary.get("task"),
            "attempt": args.attempt,
            "checkpoint": str(checkpoint),
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "variants": variant_rows,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--top-logprob-actions", type=int, default=10)
    parser.add_argument("--logprob-batch-size", type=int, default=1)
    parser.add_argument("--max-logprob-candidate-actions", type=int, default=16)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--recipe", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_diagnostic(args)
    _write_json(args.output, payload)
    print(json.dumps(sanitize_json_value(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
