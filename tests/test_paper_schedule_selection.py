from __future__ import annotations

import json
from pathlib import Path

from self.experiments.paper_schedule_selection import (
    choose_addition_fullpack_candidate,
    choose_addition_stage1_topk,
    choose_addition_candidate,
    choose_run_length_candidate,
    score_addition_fullpack_candidate,
    score_addition_stage1_schedule,
    score_addition_candidate,
    score_run_length_candidate,
)


def _write_results(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_choose_run_length_candidate_uses_same_end_saturation_rule(tmp_path: Path):
    candidate_a = score_run_length_candidate(
        _write_results(
            tmp_path / "run_length_a.json",
            [
                {
                    "round": 0,
                    "max_bits": 32,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 30)},
                    "eval_accuracy": 0.90,
                    "composed_eval_accuracy": 0.88,
                },
                {
                    "round": 1,
                    "max_bits": 40,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 39)},
                    "eval_accuracy": 0.91,
                    "composed_eval_accuracy": 0.89,
                },
            ],
        ),
        expand_num_bits=4,
    )
    candidate_a["expand_train_per_bit"] = 1200

    candidate_b = score_run_length_candidate(
        _write_results(
            tmp_path / "run_length_b.json",
            [
                {
                    "round": 0,
                    "max_bits": 32,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 28)},
                    "eval_accuracy": 0.89,
                    "composed_eval_accuracy": 0.87,
                },
                {
                    "round": 1,
                    "max_bits": 40,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 37)},
                    "eval_accuracy": 0.90,
                    "composed_eval_accuracy": 0.88,
                },
            ],
        ),
        expand_num_bits=4,
    )
    candidate_b["expand_train_per_bit"] = 2400

    selected = choose_run_length_candidate([candidate_a, candidate_b])

    assert candidate_a["accepted"] is True
    assert candidate_b["accepted"] is True
    assert selected["results_path"].endswith("run_length_a.json")


def test_choose_addition_candidate_prefers_primary_gate_then_frontier(tmp_path: Path):
    candidate_a = score_addition_candidate(
        _write_results(
            tmp_path / "addition_a.json",
            [
                {
                    "round": 0,
                    "max_digits": 15,
                    "per_digit_accuracy": {str(d): 0.82 for d in range(3, 13)},
                    "seed_eval_accuracy": 0.96,
                    "frontier_train_accuracy": 0.55,
                    "expanded_eval_accuracy": 0.36,
                },
                {
                    "round": 1,
                    "max_digits": 19,
                    "per_digit_accuracy": {str(d): 0.82 for d in range(3, 18)},
                    "seed_eval_accuracy": 0.96,
                    "frontier_train_accuracy": 0.56,
                    "expanded_eval_accuracy": 0.38,
                },
            ],
        ),
        expand_num_digits=4,
    )
    candidate_a["seed_replay_train_per_digit"] = 5000
    candidate_a["expand_train_per_digit"] = 5000

    candidate_b = score_addition_candidate(
        _write_results(
            tmp_path / "addition_b.json",
            [
                {
                    "round": 0,
                    "max_digits": 17,
                    "per_digit_accuracy": {str(d): 0.82 for d in range(3, 15)},
                    "seed_eval_accuracy": 0.97,
                    "frontier_train_accuracy": 0.58,
                    "expanded_eval_accuracy": 0.41,
                },
                {
                    "round": 1,
                    "max_digits": 21,
                    "per_digit_accuracy": {str(d): 0.82 for d in range(3, 20)},
                    "seed_eval_accuracy": 0.97,
                    "frontier_train_accuracy": 0.61,
                    "expanded_eval_accuracy": 0.44,
                },
            ],
        ),
        expand_num_digits=4,
    )
    candidate_b["seed_replay_train_per_digit"] = 8000
    candidate_b["expand_train_per_digit"] = 8000

    selected = choose_addition_candidate([candidate_a, candidate_b])

    assert candidate_a["accepted"] is True
    assert candidate_b["accepted"] is True
    assert selected["results_path"].endswith("addition_b.json")


def test_choose_addition_candidate_returns_none_without_fallback_eligible_runs(tmp_path: Path):
    candidate = score_addition_candidate(
        _write_results(
            tmp_path / "addition_bad.json",
            [
                {
                    "round": 0,
                    "max_digits": 15,
                    "per_digit_accuracy": {"3": 0.4, "4": 0.4},
                    "seed_eval_accuracy": 0.90,
                    "frontier_train_accuracy": 0.40,
                    "expanded_eval_accuracy": 0.20,
                },
                {
                    "round": 1,
                    "max_digits": 19,
                    "per_digit_accuracy": {"3": 0.4, "4": 0.4},
                    "seed_eval_accuracy": 0.90,
                    "frontier_train_accuracy": 0.40,
                    "expanded_eval_accuracy": 0.20,
                },
            ],
        ),
        expand_num_digits=4,
    )
    candidate["seed_replay_train_per_digit"] = 5000
    candidate["expand_train_per_digit"] = 5000

    assert choose_addition_candidate([candidate]) is None


def test_choose_addition_stage1_topk_prefers_compose_strength_then_filtered_then_direct(tmp_path: Path):
    def write_triplet(label: str, direct: dict, with_carry: dict, filtered: dict) -> dict:
        return score_addition_stage1_schedule(
            {
                "direct": _write_results(tmp_path / f"{label}_direct.json", [direct]),
                "with_carry": _write_results(tmp_path / f"{label}_with_carry.json", [with_carry]),
                "with_carry_filtered": _write_results(tmp_path / f"{label}_filtered.json", [filtered]),
            },
            expand_num_digits=3,
            seed_replay_train_per_digit=5000,
            expand_train_per_digit=10000,
        )

    candidate_a = write_triplet(
        "sched_a",
        {
            "round": 8,
            "max_digits": 31,
            "expanded_eval_accuracy": 0.12,
            "seed_eval_accuracy": 0.97,
            "frontier_train_accuracy": 0.20,
        },
        {
            "round": 8,
            "max_digits": 31,
            "expanded_eval_accuracy": 0.41,
            "seed_eval_accuracy": 0.96,
            "frontier_train_accuracy": 0.32,
        },
        {
            "round": 8,
            "max_digits": 31,
            "expanded_eval_accuracy": 0.38,
            "seed_eval_accuracy": 0.96,
            "frontier_train_accuracy": 0.28,
        },
    )
    candidate_a["schedule_label"] = "sched_a"
    candidate_a["output_root"] = str(tmp_path / "sched_a")

    candidate_b = write_triplet(
        "sched_b",
        {
            "round": 8,
            "max_digits": 39,
            "expanded_eval_accuracy": 0.18,
            "seed_eval_accuracy": 0.98,
            "frontier_train_accuracy": 0.22,
        },
        {
            "round": 8,
            "max_digits": 39,
            "expanded_eval_accuracy": 0.37,
            "seed_eval_accuracy": 0.98,
            "frontier_train_accuracy": 0.29,
        },
        {
            "round": 8,
            "max_digits": 39,
            "expanded_eval_accuracy": 0.45,
            "seed_eval_accuracy": 0.98,
            "frontier_train_accuracy": 0.31,
        },
    )
    candidate_b["schedule_label"] = "sched_b"
    candidate_b["output_root"] = str(tmp_path / "sched_b")

    candidate_c = write_triplet(
        "sched_c",
        {
            "round": 8,
            "max_digits": 47,
            "expanded_eval_accuracy": 0.11,
            "seed_eval_accuracy": 0.96,
            "frontier_train_accuracy": 0.21,
        },
        {
            "round": 8,
            "max_digits": 47,
            "expanded_eval_accuracy": 0.28,
            "seed_eval_accuracy": 0.96,
            "frontier_train_accuracy": 0.24,
        },
        {
            "round": 8,
            "max_digits": 47,
            "expanded_eval_accuracy": 0.27,
            "seed_eval_accuracy": 0.96,
            "frontier_train_accuracy": 0.21,
        },
    )
    candidate_c["schedule_label"] = "sched_c"
    candidate_c["output_root"] = str(tmp_path / "sched_c")

    selected = choose_addition_stage1_topk([candidate_a, candidate_b, candidate_c], k=2)

    assert candidate_a["accepted"] is True
    assert candidate_b["accepted"] is True
    assert candidate_c["accepted"] is False
    assert [candidate["schedule_label"] for candidate in selected] == ["sched_b", "sched_a"]


def test_choose_addition_fullpack_candidate_prefers_filtered_schedule(tmp_path: Path):
    def write_fullpack(label: str, filtered_expanded: float, filtered_frontier: float, with_carry_expanded: float) -> dict:
        return score_addition_fullpack_candidate(
            {
                "short_only": _write_results(tmp_path / f"{label}_short.json", [{"round": 8, "max_digits": 31}]),
                "direct": _write_results(tmp_path / f"{label}_direct.json", [{"round": 8, "max_digits": 31}]),
                "with_carry": _write_results(
                    tmp_path / f"{label}_with_carry.json",
                    [{"round": 8, "max_digits": 31, "expanded_eval_accuracy": with_carry_expanded}],
                ),
                "with_carry_filtered": _write_results(
                    tmp_path / f"{label}_filtered.json",
                    [
                        {
                            "round": 8,
                            "max_digits": 31,
                            "expanded_eval_accuracy": filtered_expanded,
                            "frontier_train_accuracy": filtered_frontier,
                        }
                    ],
                ),
                "compose_corrupt": _write_results(tmp_path / f"{label}_corrupt.json", [{"round": 8, "max_digits": 31}]),
            },
            expand_num_digits=3,
            seed_replay_train_per_digit=5000,
            expand_train_per_digit=10000,
        )

    candidate_a = write_fullpack("sched_a", filtered_expanded=0.42, filtered_frontier=0.31, with_carry_expanded=0.35)
    candidate_b = write_fullpack("sched_b", filtered_expanded=0.39, filtered_frontier=0.37, with_carry_expanded=0.44)

    selected = choose_addition_fullpack_candidate([candidate_a, candidate_b])

    assert selected["with_carry_filtered_expanded_eval_accuracy"] == 0.42
