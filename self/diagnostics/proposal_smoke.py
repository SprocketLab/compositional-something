#!/usr/bin/env python3
"""Fast prompt-only proposal smoke test for adaptive config runs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from self.adaptive.attempts import build_attempt_prompt
from self.adaptive.proposal import (
    PromptBundle,
    generate_proposals_from_model,
    validate_config_rows,
)
from self.core.data_io import ensure_dir, sanitize_json_value, write_json
from self.core.model_io import instantiate_model_and_tokenizer


JsonDict = dict[str, Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_selected_record(results_path: Path) -> Mapping[str, Any] | None:
    if not results_path.exists():
        return None
    rows = _read_json(results_path)
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if isinstance(row, Mapping) and isinstance(row.get("selected"), Mapping):
            return row
    return None


def _state_from_run_dir(run_dir: Path, *, state: str) -> JsonDict:
    summary = _read_json(run_dir / "summary.json")
    metadata = _read_json(run_dir / "data" / "metadata.json")
    seed_metrics = _read_json(run_dir / "round_00" / "metrics.json")

    task = str(summary["task"])
    if state == "latest":
        selected_row = _latest_selected_record(run_dir / "adaptive_candidate_training_results.json")
    elif state == "seed":
        selected_row = None
    else:
        raise ValueError(f"Unsupported state={state!r}.")

    if selected_row is not None:
        selected = dict(selected_row["selected"])
        current_checkpoint = str(selected.get("model_dir") or summary["current_checkpoint"])
        current_avg = float(selected["final_accuracy"])
        current_per_size = {int(k): float(v) for k, v in dict(selected["per_size_accuracy"]).items()}
        source_sizes = [int(size) for size in selected_row.get("source_sizes_after") or summary["source_sizes"]]
        attempt_index = int(selected_row.get("attempt", 0)) + 1
        selected_rounds = int(selected_row.get("selected_round", summary.get("selected_rounds_completed", 0)) or 0)
        consecutive_no_selection = 0
    else:
        current_checkpoint = str(seed_metrics.get("model_dir") or summary["current_checkpoint"])
        current_avg = float(seed_metrics["eval_accuracy"])
        current_per_size = {int(k): float(v) for k, v in dict(seed_metrics["per_size_accuracy"]).items()}
        source_sizes = [int(size) for size in summary["source_sizes"]]
        attempt_index = int(summary.get("attempts_completed", 0) or 0) + 1
        selected_rounds = int(summary.get("selected_rounds_completed", 0) or 0)
        consecutive_no_selection = 0

    return {
        "task": task,
        "condition": "config",
        "current_checkpoint": current_checkpoint,
        "current_avg_accuracy": current_avg,
        "init_avg_accuracy": float(seed_metrics["init_final_accuracy"]),
        "current_per_size_accuracy": current_per_size,
        "source_sizes": sorted(set(source_sizes)),
        "attempt_index": attempt_index,
        "selected_rounds": selected_rounds,
        "selected_round_for_prompt": selected_rounds + 1,
        "consecutive_no_selection": consecutive_no_selection,
        "initial_min_size": int(metadata["initial_min_size"]),
        "initial_max_size": int(metadata["initial_max_size"]),
        "frontier_min_size": int(metadata["frontier_min_size"]),
        "frontier_max_size": int(metadata["frontier_max_size"]),
        "summary": summary,
    }


def _prompt_args(state_payload: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        task=state_payload["task"],
        condition="config",
        initial_min_size=state_payload["initial_min_size"],
        initial_max_size=state_payload["initial_max_size"],
        frontier_min_size=state_payload["frontier_min_size"],
        frontier_max_size=state_payload["frontier_max_size"],
        max_selected_rounds=0,
        allow_repeat_targets=False,
        proposal_output_schema="action_observation",
    )


def _action_key(result: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if not bool(result.get("valid")):
        return None
    parsed = result.get("parsed_proposal")
    if not isinstance(parsed, Mapping):
        return None
    return (
        int(parsed["left"]),
        int(parsed["right"]),
        int(parsed.get("target", int(parsed["left"]) + int(parsed["right"]))),
        str(parsed.get("guard", "none")),
    )


def _summarize_results(results: list[JsonDict]) -> JsonDict:
    category_counts: Counter[str] = Counter()
    action_counts: Counter[tuple[Any, ...]] = Counter()
    for result in results:
        category = str(result.get("validation_category") or ("valid" if result.get("valid") else "unknown"))
        category_counts[category] += 1
        action = _action_key(result)
        if action is not None:
            action_counts[action] += 1
    return {
        "total": len(results),
        "valid_count": sum(action_counts.values()),
        "unique_valid_actions": len(action_counts),
        "category_counts": dict(category_counts),
        "action_counts": [
            {"action": list(action), "count": count}
            for action, count in action_counts.most_common()
        ],
    }


def _write_sensitivity_run_dir(
    *,
    output_dir: Path,
    run_dir: Path,
    prompt: PromptBundle,
    state_payload: Mapping[str, Any],
    results: list[JsonDict],
) -> Path:
    diagnostic_run_dir = output_dir / "state_sensitivity_input"
    attempt_dir = diagnostic_run_dir / "attempt_0001"
    ensure_dir(attempt_dir)
    write_json(
        diagnostic_run_dir / "summary.json",
        {
            "task": state_payload["task"],
            "condition": "config",
            "current_checkpoint": state_payload["current_checkpoint"],
            "source_run_dir": str(run_dir),
            "source_state": {
                "source_sizes": state_payload["source_sizes"],
                "current_avg_accuracy": state_payload["current_avg_accuracy"],
                "attempt_index": state_payload["attempt_index"],
                "selected_rounds": state_payload["selected_rounds"],
            },
        },
    )
    write_json(attempt_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})
    write_json(attempt_dir / "proposal_results.json", results)
    return diagnostic_run_dir


def run_smoke(args: argparse.Namespace) -> JsonDict:
    started = time.time()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    state_payload = _state_from_run_dir(run_dir, state=args.state)
    prompt_result = build_attempt_prompt(
        args=_prompt_args(state_payload),
        current_checkpoint=str(state_payload["current_checkpoint"]),
        current_final_accuracy=float(state_payload["current_avg_accuracy"]),
        init_final_accuracy=float(state_payload["init_avg_accuracy"]),
        current_per_size_accuracy=dict(state_payload["current_per_size_accuracy"]),
        source_sizes=set(int(size) for size in state_payload["source_sizes"]),
        selected_round_for_prompt=int(state_payload["selected_round_for_prompt"]),
        attempt_index=int(state_payload["attempt_index"]),
        selected_rounds=int(state_payload["selected_rounds"]),
        consecutive_no_selection=int(state_payload["consecutive_no_selection"]),
    )
    prompt = prompt_result.prompt
    write_json(output_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})

    model, tokenizer = instantiate_model_and_tokenizer(
        str(args.checkpoint or state_payload["current_checkpoint"]),
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        prompt_tokens = len(tokenizer.encode(prompt.text(), add_special_tokens=False))
        rows = generate_proposals_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_candidates=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validation_args = argparse.Namespace(
        task=state_payload["task"],
        condition="config",
        proposal_output_schema="action_observation",
        initial_min_size=state_payload["initial_min_size"],
        initial_max_size=state_payload["initial_max_size"],
    )
    results = validate_config_rows(
        rows=rows,
        args=validation_args,
        source_sizes=set(int(size) for size in state_payload["source_sizes"]),
        frontier_min=int(state_payload["frontier_min_size"]),
        frontier_max=int(state_payload["frontier_max_size"]),
    )
    diagnostic_run_dir = _write_sensitivity_run_dir(
        output_dir=output_dir,
        run_dir=run_dir,
        prompt=prompt,
        state_payload=state_payload,
        results=results,
    )
    payload = {
        "run_dir": str(run_dir),
        "state": args.state,
        "task": state_payload["task"],
        "checkpoint": str(args.checkpoint or state_payload["current_checkpoint"]),
        "output_dir": str(output_dir),
        "diagnostic_run_dir": str(diagnostic_run_dir),
        "prompt_tokens": prompt_tokens,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "source_sizes": state_payload["source_sizes"],
        "frontier": {
            "min": state_payload["frontier_min_size"],
            "max": state_payload["frontier_max_size"],
        },
        "summary": _summarize_results(results),
        "raw_results": results,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "proposal_smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", choices=["seed", "latest"], default="latest")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--recipe", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> None:
    payload = run_smoke(build_parser().parse_args())
    print(json.dumps(sanitize_json_value(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
