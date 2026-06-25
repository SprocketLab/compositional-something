from __future__ import annotations

import json

import pytest

from self.launcher_manifests import (
    build_adaptive_candidate_submission_manifest,
    build_adaptive_condition_submission_manifest,
    main,
)


def test_adaptive_candidate_submission_manifest_preserves_schema() -> None:
    payload = build_adaptive_candidate_submission_manifest(
        out_root="/tmp/run",
        tasks="addition run_length",
        conditions="config",
        model_name="Qwen/Qwen3-4B",
        proposal_model_name="current",
        outcome_trace_target_modes="numeric textual",
        proposal_grpo_reward_modes="outcome rank",
        proposal_grpo_zero_variance_modes="fixed_baseline skip",
        candidate_eval_backends="transformers vllm",
        num_candidates_list="8 16",
        proposal_grpo_learning_rates="1e-6 3e-6",
        proposal_grpo_kl_coef="0.01",
        adaptive_config_files="/tmp/base.env",
        job_fields=[
            "addition",
            "config",
            "numeric",
            "outcome",
            "fixed_baseline",
            "transformers",
            "8",
            "1e-6",
            "0.01",
            "123",
            "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-eval-transformers",
            "run_length",
            "config",
            "textual",
            "rank",
            "skip",
            "vllm",
            "16",
            "3e-6",
            "0.01",
            "124",
            "/tmp/run/run_length-config-textual-n16-reward-rank-grpo-skip-lr-3em6-eval-vllm",
        ],
    )

    assert payload["out_root"] == "/tmp/run"
    assert payload["tasks"] == ["addition", "run_length"]
    assert payload["conditions"] == ["config"]
    assert payload["model_name"] == "Qwen/Qwen3-4B"
    assert payload["proposal_model_name"] == "current"
    assert payload["outcome_trace_target_modes"] == ["numeric", "textual"]
    assert payload["proposal_grpo_reward_modes"] == ["outcome", "rank"]
    assert payload["proposal_grpo_zero_variance_modes"] == ["fixed_baseline", "skip"]
    assert payload["candidate_eval_backends"] == ["transformers", "vllm"]
    assert payload["num_candidates_list"] == [8, 16]
    assert payload["proposal_grpo_learning_rates"] == ["1e-6", "3e-6"]
    assert payload["proposal_grpo_kl_coef"] == "0.01"
    assert payload["adaptive_config_files"] == "/tmp/base.env"
    assert payload["jobs"]["addition-config-numeric-n8-reward-outcome-grpo-fixed_baseline-lr-1e-6-eval-transformers"] == {
        "task": "addition",
        "condition": "config",
        "outcome_trace_target_mode": "numeric",
        "proposal_grpo_reward_mode": "outcome",
        "proposal_grpo_zero_variance": "fixed_baseline",
        "candidate_eval_backend": "transformers",
        "num_candidates": 8,
        "proposal_grpo_learning_rate": "1e-6",
        "proposal_grpo_kl_coef": "0.01",
        "job_id": "123",
        "output_dir": "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-eval-transformers",
        "status": "submitted",
    }
    assert (
        payload["jobs"]["run_length-config-textual-n16-reward-rank-grpo-skip-lr-3e-6-eval-vllm"][
            "proposal_grpo_learning_rate"
        ]
        == "3e-6"
    )


def test_adaptive_candidate_submission_manifest_rejects_partial_job_fields() -> None:
    with pytest.raises(ValueError, match="groups of 11"):
        build_adaptive_candidate_submission_manifest(
            out_root="/tmp/run",
            tasks="addition",
            conditions="config",
            model_name="Qwen/Qwen3-1.7B",
            proposal_model_name="current",
            outcome_trace_target_modes="numeric",
            proposal_grpo_reward_modes="outcome",
            proposal_grpo_zero_variance_modes="fixed_baseline",
            candidate_eval_backends="transformers",
            num_candidates_list="8",
            proposal_grpo_learning_rates="1e-6",
            proposal_grpo_kl_coef="0.01",
            adaptive_config_files="",
            job_fields=["addition", "config"],
        )


def test_adaptive_condition_submission_manifest_preserves_schema() -> None:
    payload = build_adaptive_condition_submission_manifest(
        out_root="/tmp/condition",
        partition="ailab",
        gres="gpu:h200:1",
        time_limit="01:00:00",
        cpus_per_task="4",
        mem="96G",
        frontier_policy="fixed",
        frontier_min_count="1",
        frontier_max_accuracy="0.85",
        frontier_max_width="1",
        frontier_prefer_larger_weight="0.01",
        enforce_selected_frontier="0",
        frontier_diagnostics_path="",
        addition_config_job_id="11",
        addition_program_job_id="12",
        run_length_config_job_id="13",
        run_length_program_job_id="14",
    )

    assert payload["out_root"] == "/tmp/condition"
    assert payload["slurm"]["partition"] == "ailab"
    assert payload["slurm"]["frontier_diagnostics_path"] is None
    assert payload["jobs"]["addition_config"] == {
        "job_id": "11",
        "task": "addition",
        "condition": "config",
        "output_dir": "/tmp/condition/addition-config",
    }
    assert payload["jobs"]["run_length_program"]["output_dir"] == "/tmp/condition/run-length-program"
    assert "split adaptive proposal/preflight" in payload["scope_note"]


def test_launcher_manifest_cli_writes_json(tmp_path) -> None:
    manifest = tmp_path / "submission_manifest.json"

    main(
        [
            "adaptive-candidate",
            "--manifest",
            str(manifest),
            "--out-root",
            "/tmp/run",
            "--tasks",
            "addition",
            "--conditions",
            "config",
            "--model-name",
            "Qwen/Qwen3-4B",
            "--proposal-model-name",
            "current",
            "--outcome-trace-target-modes",
            "numeric",
            "--proposal-grpo-reward-modes",
            "outcome",
            "--proposal-grpo-zero-variance-modes",
            "fixed_baseline",
            "--candidate-eval-backends",
            "transformers",
            "--num-candidates-list",
            "8",
            "--proposal-grpo-learning-rates",
            "1e-6",
            "--proposal-grpo-kl-coef",
            "0.01",
            "--adaptive-config-files",
            "/tmp/base.env",
            "--job-fields",
            "addition",
            "config",
            "numeric",
            "outcome",
            "fixed_baseline",
            "transformers",
            "8",
            "1e-6",
            "0.01",
            "123",
            "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-eval-transformers",
        ]
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["adaptive_config_files"] == "/tmp/base.env"
    assert payload["model_name"] == "Qwen/Qwen3-4B"
    assert payload["proposal_model_name"] == "current"
    assert (
        payload["jobs"]["addition-config-numeric-n8-reward-outcome-grpo-fixed_baseline-lr-1e-6-eval-transformers"][
            "job_id"
        ]
        == "123"
    )
