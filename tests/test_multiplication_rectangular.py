from __future__ import annotations

import random

import pytest

from self.multiplication_rectangular import (
    EDGE_ONLY_MULTIPLICATION_PARTITIONS,
    RectangularCompositionLeaf,
    RectangularMultiplicationExample,
    build_partition_supported_components,
    build_multiplier_digit_components,
    build_sampled_rectangular_dataset,
    compose_target_from_multiplier_digit_values,
    compose_target_from_weighted_component_values,
    format_cot_reverse_prompt,
    format_cot_reverse_target,
    iter_partition_grid,
    normalize_cot_reverse_prediction_for_training,
    parse_partition_spec,
    parse_cot_reverse_final_value,
    partition_label,
    prediction_matches_example,
)
from self.rectangular_multiplication_recipe_seed_fit import main as rectangular_seed_fit_main
from self.self_improvement_recipe import fit_recipe_phase_to_max_steps, resolve_self_improvement_recipe


def test_cot_reverse_prompt_and_target_match_expected_trace():
    assert format_cot_reverse_prompt(12, 345) == "21*543="
    assert format_cot_reverse_target(12, 345) == "060+0840(0450)+00630=04140"


def test_parse_cot_reverse_final_value_accepts_shorter_tail():
    example = RectangularMultiplicationExample(
        a=12,
        b=34,
        a_digits=2,
        b_digits=2,
        format_version="cot_reverse_v1",
    )
    assert parse_cot_reverse_final_value("840+0630=804", example.total_digits) == 408
    assert prediction_matches_example("840+0630=804", example) is True


def test_compose_target_from_multiplier_digit_values_matches_gold_trace():
    example = RectangularMultiplicationExample(
        a=12,
        b=345,
        a_digits=2,
        b_digits=3,
        format_version="cot_reverse_v1",
    )
    components = build_multiplier_digit_components(example)
    component_values = [component.a * component.b for _, component in components]

    assert compose_target_from_multiplier_digit_values(example, component_values) == example.target()


def test_compose_target_from_multiplier_digit_values_matches_gold_symbolic():
    example = RectangularMultiplicationExample(
        a=12,
        b=345,
        a_digits=2,
        b_digits=3,
        format_version="symbolic_v1",
    )
    components = build_multiplier_digit_components(example)
    component_values = [component.a * component.b for _, component in components]

    assert compose_target_from_multiplier_digit_values(example, component_values) == "04140"
    assert compose_target_from_multiplier_digit_values(example, component_values) == example.target()


def test_build_partition_supported_components_reduces_to_current_edge_rule():
    example = RectangularMultiplicationExample(
        a=12,
        b=345,
        a_digits=2,
        b_digits=3,
        format_version="symbolic_v1",
    )

    leaves = build_partition_supported_components(
        example,
        supported_partitions=EDGE_ONLY_MULTIPLICATION_PARTITIONS,
    )

    assert leaves == [
        RectangularCompositionLeaf(
            shift_digits=0,
            example=RectangularMultiplicationExample(a=12, b=5, a_digits=2, b_digits=1, format_version="symbolic_v1"),
        ),
        RectangularCompositionLeaf(
            shift_digits=1,
            example=RectangularMultiplicationExample(a=12, b=4, a_digits=2, b_digits=1, format_version="symbolic_v1"),
        ),
        RectangularCompositionLeaf(
            shift_digits=2,
            example=RectangularMultiplicationExample(a=12, b=3, a_digits=2, b_digits=1, format_version="symbolic_v1"),
        ),
    ]


def test_build_partition_supported_components_supports_square_seed():
    example = RectangularMultiplicationExample(
        a=12345678,
        b=87654321,
        a_digits=8,
        b_digits=8,
        format_version="symbolic_v1",
    )
    square_seed = iter_partition_grid(1, 3, 1, 3)

    leaves = build_partition_supported_components(
        example,
        supported_partitions=square_seed,
    )

    assert len(leaves) == 9
    assert all(leaf.example.a_digits <= 3 for leaf in leaves)
    assert all(leaf.example.b_digits <= 3 for leaf in leaves)
    weighted_values = [(leaf.shift_digits, leaf.example.a * leaf.example.b) for leaf in leaves]
    assert compose_target_from_weighted_component_values(example, weighted_values) == example.target()


def test_compose_target_from_weighted_component_values_matches_square_seed_gold():
    example = RectangularMultiplicationExample(
        a=12345678,
        b=87654321,
        a_digits=8,
        b_digits=8,
        format_version="symbolic_v1",
    )

    weighted_values = [
        (0, 678 * 321),
        (3, 678 * 654),
        (6, 678 * 87),
        (3, 345 * 321),
        (6, 345 * 654),
        (9, 345 * 87),
        (6, 12 * 321),
        (9, 12 * 654),
        (12, 12 * 87),
    ]

    assert compose_target_from_weighted_component_values(example, weighted_values) == example.target()


def test_build_sampled_rectangular_dataset_respects_partition_counts():
    partitions = [(1, 6), (6, 1)]
    generated = build_sampled_rectangular_dataset(
        partitions=partitions,
        per_partition_counts={"train": 3, "validation": 2, "test": 2},
        rng=random.Random(0),
        format_version="cot_reverse_v1",
    )

    assert len(generated["train"]) == 6
    assert len(generated["validation"]) == 4
    assert len(generated["test"]) == 4
    assert {partition_label((example.a_digits, example.b_digits)) for example in generated["train"]} == {"1x6", "6x1"}


def test_build_sampled_rectangular_dataset_switches_to_fast_duplicates(monkeypatch):
    call_count = {"value": 0}

    def always_one_digit(*args, **kwargs):
        call_count["value"] += 1
        return 1

    monkeypatch.setattr(
        "self.multiplication_rectangular.sample_int_with_exact_digits",
        always_one_digit,
    )

    generated = build_sampled_rectangular_dataset(
        partitions=[(1, 1)],
        per_partition_counts={"train": 4, "validation": 0, "test": 0},
        rng=random.Random(0),
        format_version="symbolic_v1",
        max_attempts=3,
    )

    assert len(generated["train"]) == 4
    # 1 unique example, then 1 duplicate after exhausting attempts, then 2 more
    # duplicates immediately on the next draws. Each example samples `a` and `b`.
    assert call_count["value"] == 12


def test_normalize_cot_reverse_prediction_for_training_filters_whitespace():
    assert normalize_cot_reverse_prediction_for_training(" 060 +0840(0450)\n+00630=04140 ") == "060+0840(0450)+00630=04140"


def test_edge_only_partition_list_is_exact_and_ordered():
    assert EDGE_ONLY_MULTIPLICATION_PARTITIONS == (
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
    )


def test_parse_partition_spec_deduplicates_while_preserving_order():
    assert parse_partition_spec("1x1,1x2,1x2,2x1,1x1") == [(1, 1), (1, 2), (2, 1)]


def test_multiplication_recipe_preset_matches_seed_regime():
    preset = resolve_self_improvement_recipe("multiplication_self_improve_v1")

    assert preset.hidden_size == 384
    assert preset.intermediate_size == 1536
    assert preset.num_attention_heads == 6
    assert preset.num_hidden_layers == 6
    assert preset.per_device_train_batch_size == 256
    assert preset.per_device_eval_batch_size == 256
    assert preset.seed_phase.learning_rate == 5e-5
    assert preset.seed_phase.weight_decay == 0.01
    assert preset.seed_phase.adam_beta2 == 0.98
    assert preset.seed_phase.warmup_steps == 1000
    assert preset.seed_phase.max_steps == 10000
    assert preset.seed_phase.num_stable_steps == 7000
    assert preset.seed_phase.num_decay_steps == 2000


def test_multiplication_recipe_short_overfit_budget_compresses_schedule():
    preset = resolve_self_improvement_recipe("multiplication_self_improve_v1")
    compressed = fit_recipe_phase_to_max_steps(preset.seed_phase, max_steps=1000)

    assert compressed.warmup_steps == 100
    assert compressed.num_stable_steps == 700
    assert compressed.num_decay_steps == 200


def test_rectangular_seed_fit_dry_run_accepts_explicit_edge_partitions(tmp_path, capsys):
    out_dir = tmp_path / "seed_dry_run"

    rectangular_seed_fit_main(
        [
            "--output-dir",
            str(out_dir),
            "--recipe",
            "multiplication_self_improve_v1",
            "--partitions",
            "1x1,1x2,2x1,1x2",
            "--dry-run",
        ]
    )

    stdout = capsys.readouterr().out
    assert '"recipe": "multiplication_self_improve_v1"' in stdout
    assert '"partitions": [' in stdout
    assert '"1x1"' in stdout
    assert '"1x2"' in stdout
    assert '"2x1"' in stdout
    payload = (out_dir / "dry_run_plan.json").read_text(encoding="utf-8")
    assert '"partitions": [' in payload
    assert payload.count('"1x2"') == 1


def test_rectangular_seed_fit_rejects_non_symbolic_format_for_multiplication_recipe(tmp_path):
    with pytest.raises(ValueError, match="only supports format_version='symbolic_v1'"):
        rectangular_seed_fit_main(
            [
                "--output-dir",
                str(tmp_path / "bad_format"),
                "--recipe",
                "multiplication_self_improve_v1",
                "--format-version",
                "legacy",
                "--dry-run",
            ]
        )


def test_rectangular_seed_fit_skip_train_eval_dry_run_records_flag(tmp_path, capsys):
    out_dir = tmp_path / "seed_skip_train_eval"

    rectangular_seed_fit_main(
        [
            "--output-dir",
            str(out_dir),
            "--recipe",
            "multiplication_self_improve_v1",
            "--partitions",
            "1x1,1x2,2x1",
            "--skip-train-eval",
            "--dry-run",
        ]
    )

    stdout = capsys.readouterr().out
    assert '"skip_train_eval": true' in stdout
    payload = (out_dir / "dry_run_plan.json").read_text(encoding="utf-8")
    assert '"skip_train_eval": true' in payload
