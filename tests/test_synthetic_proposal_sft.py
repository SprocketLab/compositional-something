from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.adaptive import proposal as adaptive_proposal
from self.adaptive import run as adaptive_run


def _args(**overrides):
    values = dict(
        task="addition",
        condition="config",
        dry_run_data_only=False,
        controller_execution_mode="local",
        proposal_output_schema="action_observation",
        initial_min_size=3,
        initial_max_size=7,
        frontier_min_size=8,
        frontier_max_size=14,
        source_admission_target_accuracy_threshold=0.8,
        synthetic_proposal_sft=False,
        synthetic_proposal_sft_seed_mix=False,
        synthetic_proposal_sft_examples=0,
        synthetic_proposal_sft_num_epochs=1,
        synthetic_proposal_sft_learning_rate=1e-6,
        synthetic_proposal_sft_top_k=4,
        synthetic_proposal_sft_temperature=0.7,
        seed=11,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_synthetic_proposal_rows_are_valid_config_completions() -> None:
    args = _args(synthetic_proposal_sft_examples=24)

    rows = adaptive_proposal.generate_synthetic_proposal_sft_rows(
        args=args,
        task_name="addition",
        source_sizes={3, 4, 5, 6, 7},
        current_per_size_accuracy={3: 0.95, 4: 0.93, 5: 0.82, 8: 0.20, 9: 0.15},
        current_avg_accuracy=0.55,
        init_avg_accuracy=0.55,
        count=24,
        seed=123,
    )

    assert len(rows) == 24
    assert all("reliable" in row["completion"] for row in rows)
    validated = adaptive_proposal.validate_config_rows(
        rows=rows,
        args=args,
        source_sizes=set(range(3, 15)),
        frontier_min=8,
        frontier_max=14,
    )
    assert all(row["valid"] for row in validated)
    for row in validated:
        proposal = row["parsed_proposal"]
        assert proposal["left"] in set(range(3, 15))
        assert proposal["right"] in set(range(3, 15))
        assert 8 <= proposal["target"] <= 14
        assert row["parsed_prediction"]["expected_avg_delta_from_current"] >= 0.0


def test_synthetic_action_score_prefers_reliable_sources_and_weak_target() -> None:
    candidates = adaptive_proposal._synthetic_action_candidates(
        task_name="addition",
        source_sizes=[3, 4, 5],
        frontier_min=8,
        frontier_max=10,
        per_size_accuracy={3: 0.95, 4: 0.90, 5: 0.20, 8: 0.10, 9: 0.80},
        guard_choices=["none"],
    )
    by_action = {(row["left"], row["right"], row["target"]): row for row in candidates}

    reliable_weak = by_action[(4, 4, 8)]
    weak_source = by_action[(3, 5, 8)]
    saturated_target = by_action[(4, 5, 9)]

    assert reliable_weak["score"] > weak_source["score"]
    assert reliable_weak["score"] > saturated_target["score"]


def test_synthetic_seed_mix_builds_prompt_target_examples(tmp_path: Path) -> None:
    args = _args(synthetic_proposal_sft_seed_mix=True, synthetic_proposal_sft_examples=12)

    mixed, metrics = adaptive_proposal.build_synthetic_proposal_seed_mix(
        args=args,
        output_dir=tmp_path / "mix",
        source_examples=["task-a", "task-b"],
        source_sizes={3, 4, 5, 6, 7},
        seed=99,
    )

    assert metrics["skipped"] is False
    assert metrics["mode"] == "seed_mix_joint_from_base"
    assert metrics["task_examples"] == 2
    assert metrics["proposal_examples"] == 12
    assert len(mixed) == 14
    proposal_example = mixed[-1]
    assert proposal_example.prompt()
    assert "\"left\":" in proposal_example.target()
    assert proposal_example.size_for_batching() == 0
    assert (tmp_path / "mix" / "synthetic_proposal_sft_examples.jsonl").exists()
    assert (tmp_path / "mix" / "synthetic_seed_mix_metrics.json").exists()


def test_seed_dispatch_uses_synthetic_checkpoint_and_post_sft_eval(tmp_path: Path) -> None:
    calls = []

    def run_seed_phase(**kwargs):
        calls.append(("seed", kwargs))
        return SimpleNamespace(
            current_checkpoint="seed-model",
            current_final_accuracy=0.50,
            current_per_size_accuracy={3: 0.9, 4: 0.8, 8: 0.2},
            init_final_accuracy=0.50,
        )

    def apply_synthetic_proposal_sft(**kwargs):
        calls.append(("synthetic", kwargs))
        assert kwargs["source_checkpoint"] == "seed-model"
        assert kwargs["output_dir"] == tmp_path / "run" / "round_00" / "synthetic_proposal_sft"
        return "synthetic-model", {
            "skipped": False,
            "model_dir": "synthetic-model",
            "post_sft_eval_accuracy": 0.47,
            "post_sft_per_size_accuracy": {"3": 0.88, "4": 0.79, "8": 0.25},
        }

    result = adaptive_run.run_seed_dispatch(
        args=_args(synthetic_proposal_sft=True, synthetic_proposal_sft_examples=32),
        task=object(),
        config=object(),
        source_examples=[],
        eval_examples=[],
        output_dir=tmp_path / "run",
        data_dir=tmp_path / "run" / "data",
        source_sizes={3, 4, 5, 6, 7},
        deps=adaptive_run.SeedDispatchDeps(
            run_controller_worker_slurm=lambda **_: {},
            float_or_nan=float,
            run_seed_phase=run_seed_phase,
            build_synthetic_proposal_seed_mix=lambda **_: (_ for _ in ()).throw(
                AssertionError("seed mix should not run in post-seed SFT mode")
            ),
            apply_synthetic_proposal_sft=apply_synthetic_proposal_sft,
        ),
    )

    assert [call[0] for call in calls] == ["seed", "synthetic"]
    assert result.current_checkpoint == "synthetic-model"
    assert result.current_final_accuracy == 0.47
    assert result.current_per_size_accuracy == {3: 0.88, 4: 0.79, 8: 0.25}
    assert result.summary_records[0]["current_checkpoint"] == "synthetic-model"
    assert result.summary_records[0]["synthetic_proposal_sft"]["model_dir"] == "synthetic-model"


def test_seed_dispatch_mixes_synthetic_examples_into_seed_training(tmp_path: Path) -> None:
    calls = []
    task_examples = ["task-0", "task-1"]
    mixed_examples = task_examples + ["proposal-0", "proposal-1", "proposal-2"]

    def build_synthetic_proposal_seed_mix(**kwargs):
        calls.append(("seed_mix", kwargs))
        assert kwargs["output_dir"] == tmp_path / "run" / "round_00" / "synthetic_seed_mix"
        assert kwargs["source_examples"] == task_examples
        return mixed_examples, {
            "skipped": False,
            "mode": "seed_mix_joint_from_base",
            "task_examples": 2,
            "proposal_examples": 3,
            "mixed_examples": 5,
        }

    def run_seed_phase(**kwargs):
        calls.append(("seed", kwargs))
        assert kwargs["source_examples"] == mixed_examples
        return SimpleNamespace(
            current_checkpoint="seed-mix-model",
            current_final_accuracy=0.51,
            current_per_size_accuracy={3: 0.9},
            init_final_accuracy=0.51,
        )

    def apply_synthetic_proposal_sft(**kwargs):
        raise AssertionError("post-seed synthetic SFT should not run in seed-mix mode")

    result = adaptive_run.run_seed_dispatch(
        args=_args(synthetic_proposal_sft_seed_mix=True, synthetic_proposal_sft_examples=3),
        task=object(),
        config=object(),
        source_examples=task_examples,
        eval_examples=[],
        output_dir=tmp_path / "run",
        data_dir=tmp_path / "run" / "data",
        source_sizes={3, 4},
        deps=adaptive_run.SeedDispatchDeps(
            run_controller_worker_slurm=lambda **_: {},
            float_or_nan=float,
            run_seed_phase=run_seed_phase,
            build_synthetic_proposal_seed_mix=build_synthetic_proposal_seed_mix,
            apply_synthetic_proposal_sft=apply_synthetic_proposal_sft,
        ),
    )

    assert [call[0] for call in calls] == ["seed_mix", "seed"]
    assert result.current_checkpoint == "seed-mix-model"
    assert result.summary_records[0]["synthetic_proposal_sft_seed_mix"]["proposal_examples"] == 3
    assert "synthetic_proposal_sft" not in result.summary_records[0]


def test_seed_dispatch_skips_synthetic_when_amount_is_zero(tmp_path: Path) -> None:
    def run_seed_phase(**kwargs):
        return SimpleNamespace(
            current_checkpoint="seed-model",
            current_final_accuracy=0.50,
            current_per_size_accuracy={3: 0.9},
            init_final_accuracy=0.50,
        )

    def apply_synthetic_proposal_sft(**kwargs):
        raise AssertionError("synthetic SFT should not run when examples=0")

    result = adaptive_run.run_seed_dispatch(
        args=_args(synthetic_proposal_sft=True, synthetic_proposal_sft_examples=0),
        task=object(),
        config=object(),
        source_examples=[],
        eval_examples=[],
        output_dir=tmp_path / "run",
        data_dir=tmp_path / "run" / "data",
        source_sizes={3, 4},
        deps=adaptive_run.SeedDispatchDeps(
            run_controller_worker_slurm=lambda **_: {},
            float_or_nan=float,
            run_seed_phase=run_seed_phase,
            build_synthetic_proposal_seed_mix=lambda **_: (_ for _ in ()).throw(
                AssertionError("seed mix should not run when disabled")
            ),
            apply_synthetic_proposal_sft=apply_synthetic_proposal_sft,
        ),
    )

    assert result.current_checkpoint == "seed-model"
    assert "synthetic_proposal_sft" not in result.summary_records[0]
