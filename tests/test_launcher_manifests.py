from __future__ import annotations

import json

import pytest

from self.launcher_manifests import (
    build_adaptive_candidate_submission_manifest,
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
        num_candidates_list="8 16",
        proposal_grpo_learning_rates="1e-6 3e-6",
        proposal_grpo_kl_coef="0.01",
        synthetic_proposal_sft_examples_list="0 2048",
        synthetic_proposal_sft_seed_mix="0",
        synthetic_proposal_sft_num_epochs="1",
        synthetic_proposal_sft_learning_rate="1e-6",
        synthetic_proposal_sft_top_k="4",
        synthetic_proposal_sft_temperature="0.7",
        proposal_update_microbatch_size="8",
        proposal_update_accumulation_steps="1",
        adaptive_config_files="/tmp/base.env",
        job_fields=[
            "addition",
            "config",
            "numeric",
            "outcome",
            "fixed_baseline",
            "8",
            "1e-6",
            "0.01",
            "0",
            "0",
            "123",
            "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-syn0",
            "run_length",
            "config",
            "textual",
            "rank",
            "skip",
            "16",
            "3e-6",
            "0.01",
            "2048",
            "0",
            "124",
            "/tmp/run/run_length-config-textual-n16-reward-rank-grpo-skip-lr-3em6-syn2048",
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
    assert payload["num_candidates_list"] == [8, 16]
    assert payload["proposal_grpo_learning_rates"] == ["1e-6", "3e-6"]
    assert payload["proposal_grpo_kl_coef"] == "0.01"
    assert payload["synthetic_proposal_sft_examples_list"] == [0, 2048]
    assert payload["synthetic_proposal_sft_seed_mix"] is False
    assert payload["synthetic_proposal_sft_num_epochs"] == 1
    assert payload["synthetic_proposal_sft_learning_rate"] == "1e-6"
    assert payload["synthetic_proposal_sft_top_k"] == 4
    assert payload["synthetic_proposal_sft_temperature"] == 0.7
    assert payload["proposal_update_microbatch_size"] == 8
    assert payload["proposal_update_accumulation_steps"] == 1
    assert payload["adaptive_config_files"] == "/tmp/base.env"
    assert payload["jobs"]["addition-config-numeric-n8-reward-outcome-grpo-fixed_baseline-lr-1e-6-syn0"] == {
        "task": "addition",
        "condition": "config",
        "outcome_trace_target_mode": "numeric",
        "proposal_grpo_reward_mode": "outcome",
        "proposal_grpo_zero_variance": "fixed_baseline",
        "num_candidates": 8,
        "proposal_grpo_learning_rate": "1e-6",
        "proposal_grpo_kl_coef": "0.01",
        "synthetic_proposal_sft_examples": 0,
        "synthetic_proposal_sft_seed_mix": False,
        "job_id": "123",
        "output_dir": "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-syn0",
        "status": "submitted",
    }
    assert (
        payload["jobs"]["run_length-config-textual-n16-reward-rank-grpo-skip-lr-3e-6-syn2048"][
            "proposal_grpo_learning_rate"
        ]
        == "3e-6"
    )
    assert (
        payload["jobs"]["run_length-config-textual-n16-reward-rank-grpo-skip-lr-3e-6-syn2048"][
            "synthetic_proposal_sft_examples"
        ]
        == 2048
    )


def test_adaptive_candidate_submission_manifest_rejects_partial_job_fields() -> None:
    with pytest.raises(ValueError, match="groups of 12"):
        build_adaptive_candidate_submission_manifest(
            out_root="/tmp/run",
            tasks="addition",
            conditions="config",
            model_name="Qwen/Qwen3-1.7B",
            proposal_model_name="current",
            outcome_trace_target_modes="numeric",
            proposal_grpo_reward_modes="outcome",
            proposal_grpo_zero_variance_modes="fixed_baseline",
            num_candidates_list="8",
            proposal_grpo_learning_rates="1e-6",
            proposal_grpo_kl_coef="0.01",
            synthetic_proposal_sft_examples_list="0",
            synthetic_proposal_sft_seed_mix="0",
            synthetic_proposal_sft_num_epochs="1",
            synthetic_proposal_sft_learning_rate="1e-6",
            synthetic_proposal_sft_top_k="4",
            synthetic_proposal_sft_temperature="0.7",
            proposal_update_microbatch_size="8",
            proposal_update_accumulation_steps="1",
            adaptive_config_files="",
            job_fields=["addition", "config"],
        )

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
            "--num-candidates-list",
            "8",
            "--proposal-grpo-learning-rates",
            "1e-6",
            "--proposal-grpo-kl-coef",
            "0.01",
            "--synthetic-proposal-sft-examples-list",
            "0",
            "--synthetic-proposal-sft-seed-mix",
            "0",
            "--synthetic-proposal-sft-num-epochs",
            "1",
            "--synthetic-proposal-sft-learning-rate",
            "1e-6",
            "--synthetic-proposal-sft-top-k",
            "4",
            "--synthetic-proposal-sft-temperature",
            "0.7",
            "--proposal-update-microbatch-size",
            "8",
            "--proposal-update-accumulation-steps",
            "1",
            "--adaptive-config-files",
            "/tmp/base.env",
            "--job-fields",
            "addition",
            "config",
            "numeric",
            "outcome",
            "fixed_baseline",
            "8",
            "1e-6",
            "0.01",
            "0",
            "0",
            "123",
            "/tmp/run/addition-config-numeric-n8-reward-outcome-grpo-fixed-baseline-lr-1em6-syn0",
        ]
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["adaptive_config_files"] == "/tmp/base.env"
    assert payload["model_name"] == "Qwen/Qwen3-4B"
    assert payload["proposal_model_name"] == "current"
    assert payload["proposal_update_microbatch_size"] == 8
    assert payload["proposal_update_accumulation_steps"] == 1
    assert (
        payload["jobs"]["addition-config-numeric-n8-reward-outcome-grpo-fixed_baseline-lr-1e-6-syn0"][
            "job_id"
        ]
        == "123"
    )
