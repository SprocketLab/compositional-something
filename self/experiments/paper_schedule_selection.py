"""Helpers for selecting paper-facing self-improvement schedules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_PAPER_SCHEDULE_SELECTION_PATH = Path("artifacts/paper/paper_schedule_selection.json")


def load_round_summaries(results_path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of round summaries in {results_path}.")
    return [entry for entry in payload if isinstance(entry, dict)]


def row_max_size(row: Dict[str, Any]) -> int:
    value = row.get("max_digits", row.get("max_bits", row.get("max_size")))
    if value is None:
        raise ValueError("Round summary does not include a max size field.")
    return int(value)


def per_size_accuracy_map(row: Dict[str, Any]) -> Dict[int, float]:
    raw_map = row.get("per_digit_accuracy", row.get("per_bit_accuracy", row.get("per_size_accuracy", {})))
    if not isinstance(raw_map, dict):
        return {}
    result: Dict[int, float] = {}
    for raw_key, raw_value in raw_map.items():
        try:
            result[int(raw_key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return result


def max_size_at_threshold(row: Dict[str, Any], threshold: float) -> Optional[int]:
    per_size = per_size_accuracy_map(row)
    sizes = [size for size, value in per_size.items() if value >= threshold]
    if not sizes:
        return None
    return max(sizes)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_gap(gap: Optional[int]) -> float:
    if gap is None or gap < 0:
        return math.inf
    return float(gap)


def _score_bit_candidate(results_path: str | Path, *, expand_num_bits: int) -> Dict[str, Any]:
    rounds = load_round_summaries(results_path)
    if not rounds:
        raise ValueError(f"No round summaries found in {results_path}.")

    final_row = rounds[-1]
    prev_row = rounds[-2] if len(rounds) >= 2 else final_row
    final_max_bits = row_max_size(final_row)
    prev_max_bits = row_max_size(prev_row)
    final_max_bits_at_90 = max_size_at_threshold(final_row, 0.90)
    prev_max_bits_at_90 = max_size_at_threshold(prev_row, 0.90)
    gap90_final = None if final_max_bits_at_90 is None else final_max_bits - final_max_bits_at_90
    gap90_prev = None if prev_max_bits_at_90 is None else prev_max_bits - prev_max_bits_at_90
    accepted = (
        gap90_final is not None
        and 0 <= gap90_final <= expand_num_bits
        and gap90_prev is not None
        and gap90_prev > 0
    )
    return {
        "results_path": str(Path(results_path)),
        "expand_num_bits": int(expand_num_bits),
        "final_max_bits": final_max_bits,
        "final_eval_accuracy": _safe_float(final_row.get("eval_accuracy")),
        "final_composed_eval_accuracy": _safe_float(final_row.get("composed_eval_accuracy")),
        "final_max_bits_at_90_accuracy": final_max_bits_at_90,
        "prev_max_bits_at_90_accuracy": prev_max_bits_at_90,
        "gap90_final": gap90_final,
        "gap90_prev": gap90_prev,
        "accepted": accepted,
    }


def score_run_length_candidate(results_path: str | Path, *, expand_num_bits: int) -> Dict[str, Any]:
    return _score_bit_candidate(results_path, expand_num_bits=expand_num_bits)


def choose_run_length_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Need at least one run-length candidate to choose from.")

    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    pool = accepted if accepted else list(candidates)
    if accepted:
        return max(
            pool,
            key=lambda candidate: (
                int(candidate["final_max_bits"]),
                candidate.get("final_composed_eval_accuracy", float("-inf")) or float("-inf"),
                candidate.get("final_eval_accuracy", float("-inf")) or float("-inf"),
            ),
        )

    return min(
        pool,
        key=lambda candidate: (
            _nonnegative_gap(candidate.get("gap90_final")),
            -int(candidate["final_max_bits"]),
            -(candidate.get("final_composed_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("final_eval_accuracy", float("-inf")) or float("-inf")),
        ),
    )


def score_addition_candidate(results_path: str | Path, *, expand_num_digits: int) -> Dict[str, Any]:
    rounds = load_round_summaries(results_path)
    if not rounds:
        raise ValueError(f"No round summaries found in {results_path}.")

    final_row = rounds[-1]
    prev_row = rounds[-2] if len(rounds) >= 2 else final_row
    final_max_digits = row_max_size(final_row)
    prev_max_digits = row_max_size(prev_row)

    final_max_digits_at_80 = max_size_at_threshold(final_row, 0.80)
    prev_max_digits_at_80 = max_size_at_threshold(prev_row, 0.80)
    final_max_digits_at_50 = max_size_at_threshold(final_row, 0.50)
    gap80_final = None if final_max_digits_at_80 is None else final_max_digits - final_max_digits_at_80
    gap80_prev = None if prev_max_digits_at_80 is None else prev_max_digits - prev_max_digits_at_80
    gap50_final = None if final_max_digits_at_50 is None else final_max_digits - final_max_digits_at_50

    seed_eval_accuracy = _safe_float(final_row.get("seed_eval_accuracy"))
    frontier_train_accuracy = _safe_float(final_row.get("frontier_train_accuracy"))
    expanded_eval_accuracy = _safe_float(final_row.get("expanded_eval_accuracy"))
    accepted = (
        seed_eval_accuracy is not None
        and seed_eval_accuracy >= 0.95
        and frontier_train_accuracy is not None
        and frontier_train_accuracy >= 0.50
        and expanded_eval_accuracy is not None
        and expanded_eval_accuracy >= 0.35
        and gap80_final is not None
        and 0 <= gap80_final <= expand_num_digits
        and gap80_prev is not None
        and gap80_prev > 0
    )
    fallback_eligible = seed_eval_accuracy is not None and seed_eval_accuracy >= 0.95
    return {
        "results_path": str(Path(results_path)),
        "expand_num_digits": int(expand_num_digits),
        "final_max_digits": final_max_digits,
        "expanded_eval_accuracy": expanded_eval_accuracy,
        "frontier_train_accuracy": frontier_train_accuracy,
        "seed_eval_accuracy": seed_eval_accuracy,
        "gap80_final": gap80_final,
        "gap80_prev": gap80_prev,
        "gap50_final": gap50_final,
        "accepted": accepted,
        "fallback_eligible": fallback_eligible,
        "final_max_digits_at_80_accuracy": final_max_digits_at_80,
        "prev_max_digits_at_80_accuracy": prev_max_digits_at_80,
        "final_max_digits_at_50_accuracy": final_max_digits_at_50,
    }


def choose_addition_candidate(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        raise ValueError("Need at least one addition candidate to choose from.")

    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    if accepted:
        return max(
            accepted,
            key=lambda candidate: (
                int(candidate["final_max_digits"]),
                candidate.get("expanded_eval_accuracy", float("-inf")) or float("-inf"),
                candidate.get("frontier_train_accuracy", float("-inf")) or float("-inf"),
            ),
        )

    fallback_candidates = [candidate for candidate in candidates if candidate.get("fallback_eligible")]
    if not fallback_candidates:
        return None
    return min(
        fallback_candidates,
        key=lambda candidate: (
            _nonnegative_gap(candidate.get("gap50_final")),
            -int(candidate["final_max_digits"]),
            -(candidate.get("expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("frontier_train_accuracy", float("-inf")) or float("-inf")),
        ),
    )


def final_round_summary(results_path: str | Path) -> Dict[str, Any]:
    rounds = load_round_summaries(results_path)
    if not rounds:
        raise ValueError(f"No round summaries found in {results_path}.")
    return rounds[-1]


def _required_result_paths(
    baseline_results: Mapping[str, str | Path],
    baselines: Sequence[str],
) -> Dict[str, str]:
    missing = [baseline for baseline in baselines if baseline not in baseline_results]
    if missing:
        raise ValueError(f"Missing required baselines: {', '.join(sorted(missing))}")
    return {baseline: str(Path(baseline_results[baseline])) for baseline in baselines}


def _baseline_metric(row: Dict[str, Any], key: str) -> float:
    value = _safe_float(row.get(key))
    return float("-inf") if value is None else value


def score_addition_stage1_schedule(
    baseline_results: Mapping[str, str | Path],
    *,
    expand_num_digits: int,
    seed_replay_train_per_digit: int,
    expand_train_per_digit: int,
) -> Dict[str, Any]:
    result_paths = _required_result_paths(
        baseline_results,
        ("direct", "with_carry", "with_carry_filtered"),
    )
    direct_row = final_round_summary(result_paths["direct"])
    with_carry_row = final_round_summary(result_paths["with_carry"])
    filtered_row = final_round_summary(result_paths["with_carry_filtered"])

    direct_seed_eval = _safe_float(direct_row.get("seed_eval_accuracy"))
    direct_expanded_eval = _safe_float(direct_row.get("expanded_eval_accuracy"))
    with_carry_expanded_eval = _safe_float(with_carry_row.get("expanded_eval_accuracy"))
    filtered_expanded_eval = _safe_float(filtered_row.get("expanded_eval_accuracy"))
    with_carry_frontier = _safe_float(with_carry_row.get("frontier_train_accuracy"))
    filtered_frontier = _safe_float(filtered_row.get("frontier_train_accuracy"))

    compose_best_expanded_eval = max(
        value
        for value in (
            with_carry_expanded_eval,
            filtered_expanded_eval,
        )
        if value is not None
    ) if any(value is not None for value in (with_carry_expanded_eval, filtered_expanded_eval)) else None
    mean_compose_frontier = (
        (with_carry_frontier + filtered_frontier) / 2.0
        if with_carry_frontier is not None and filtered_frontier is not None
        else None
    )
    compose_beats_direct = (
        compose_best_expanded_eval is not None
        and direct_expanded_eval is not None
        and compose_best_expanded_eval > direct_expanded_eval
    )
    has_frontier_fit = any(
        value is not None and value >= 0.25
        for value in (with_carry_frontier, filtered_frontier)
    )
    direct_seed_ok = direct_seed_eval is not None and direct_seed_eval >= 0.95

    accepted = compose_beats_direct and has_frontier_fit and direct_seed_ok
    fallback_eligible = direct_seed_ok
    return {
        "baseline_results": result_paths,
        "expand_num_digits": int(expand_num_digits),
        "seed_replay_train_per_digit": int(seed_replay_train_per_digit),
        "expand_train_per_digit": int(expand_train_per_digit),
        "max_digits": max(
            row_max_size(direct_row),
            row_max_size(with_carry_row),
            row_max_size(filtered_row),
        ),
        "compose_best_expanded_eval_accuracy": compose_best_expanded_eval,
        "with_carry_expanded_eval_accuracy": with_carry_expanded_eval,
        "with_carry_filtered_expanded_eval_accuracy": filtered_expanded_eval,
        "direct_expanded_eval_accuracy": direct_expanded_eval,
        "mean_compose_frontier_train_accuracy": mean_compose_frontier,
        "with_carry_frontier_train_accuracy": with_carry_frontier,
        "with_carry_filtered_frontier_train_accuracy": filtered_frontier,
        "direct_seed_eval_accuracy": direct_seed_eval,
        "compose_beats_direct": compose_beats_direct,
        "has_frontier_fit": has_frontier_fit,
        "accepted": accepted,
        "fallback_eligible": fallback_eligible,
    }


def choose_addition_stage1_topk(
    candidates: Sequence[Dict[str, Any]],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        raise ValueError("Need at least one addition stage-1 candidate to choose from.")
    if k < 1:
        raise ValueError("k must be positive.")

    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    if accepted:
        pool = accepted
        sort_key = lambda candidate: (
            -(candidate.get("compose_best_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("with_carry_filtered_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("direct_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("mean_compose_frontier_train_accuracy", float("-inf")) or float("-inf")),
        )
    else:
        pool = [candidate for candidate in candidates if candidate.get("fallback_eligible")]
        if len(pool) < k:
            raise ValueError(
                "Need at least two addition stage-1 fallback-eligible schedules "
                "with direct.seed_eval_accuracy >= 0.95."
            )
        sort_key = lambda candidate: (
            -(candidate.get("compose_best_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("with_carry_filtered_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("direct_expanded_eval_accuracy", float("-inf")) or float("-inf")),
            -(candidate.get("mean_compose_frontier_train_accuracy", float("-inf")) or float("-inf")),
        )

    ranked = sorted(pool, key=sort_key)
    return ranked[:k]


def score_addition_fullpack_candidate(
    baseline_results: Mapping[str, str | Path],
    *,
    expand_num_digits: int,
    seed_replay_train_per_digit: int,
    expand_train_per_digit: int,
) -> Dict[str, Any]:
    result_paths = _required_result_paths(
        baseline_results,
        ("short_only", "direct", "with_carry", "with_carry_filtered", "compose_corrupt"),
    )
    with_carry_row = final_round_summary(result_paths["with_carry"])
    filtered_row = final_round_summary(result_paths["with_carry_filtered"])
    return {
        "baseline_results": result_paths,
        "expand_num_digits": int(expand_num_digits),
        "seed_replay_train_per_digit": int(seed_replay_train_per_digit),
        "expand_train_per_digit": int(expand_train_per_digit),
        "with_carry_filtered_expanded_eval_accuracy": _safe_float(filtered_row.get("expanded_eval_accuracy")),
        "with_carry_filtered_frontier_train_accuracy": _safe_float(filtered_row.get("frontier_train_accuracy")),
        "with_carry_expanded_eval_accuracy": _safe_float(with_carry_row.get("expanded_eval_accuracy")),
        "max_digits": row_max_size(filtered_row),
    }


def choose_addition_fullpack_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Need at least one addition full-pack candidate to choose from.")

    return max(
        candidates,
        key=lambda candidate: (
            candidate.get("with_carry_filtered_expanded_eval_accuracy", float("-inf")) or float("-inf"),
            candidate.get("with_carry_filtered_frontier_train_accuracy", float("-inf")) or float("-inf"),
            candidate.get("with_carry_expanded_eval_accuracy", float("-inf")) or float("-inf"),
        ),
    )


def load_schedule_selection(selection_path: str | Path = DEFAULT_PAPER_SCHEDULE_SELECTION_PATH) -> Dict[str, Any]:
    path = Path(selection_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {selection_path}.")
    return payload


def selection_schedule(
    task: str,
    *,
    selection_path: str | Path = DEFAULT_PAPER_SCHEDULE_SELECTION_PATH,
) -> Dict[str, Any]:
    payload = load_schedule_selection(selection_path)
    schedules = payload.get("selected_schedules", {})
    schedule = schedules.get(task, {})
    return schedule if isinstance(schedule, dict) else {}
