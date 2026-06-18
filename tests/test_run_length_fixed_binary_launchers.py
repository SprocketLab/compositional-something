from __future__ import annotations

import os
import subprocess
from pathlib import Path

from self.experiments import run_length_alpha10_seed_beam


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "launchers" / "self" / "submit_run_length_fixed_binary_mig.sh"
BEAM = ROOT / "launchers" / "self" / "run_run_length_alpha10_seed_beam_mig.py"


def test_run_length_fixed_binary_submitter_has_valid_bash_syntax():
    assert SUBMITTER.exists()
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_run_length_fixed_binary_submitter_dry_run_forwards_fixed_binary(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rl_fixed_binary")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    assert "--bit-composition-path-mode fixed_binary" in combined
    assert "BIT_COMPOSITION_PATH_MODE=fixed_binary" in combined
    assert "-m self.experiments.run_length_alpha10_seed_beam" in combined
    assert "run_length_fixed_binary" in combined
    assert "run_manifest.json" in combined


def test_run_length_fixed_binary_submitter_accepts_partial_config_override(tmp_path):
    config_path = tmp_path / "fixed_binary_override.env"
    config_path.write_text(
        "\n".join(
            [
                "RUN_LENGTH_FIXED_BINARY_PAPER_EXPAND_ROUNDS=3",
                "RUN_LENGTH_FIXED_BINARY_PAPER_TRAIN_BATCH_SIZE=17",
                "RUN_LENGTH_FIXED_BINARY_PAPER_COMPOSITION_PATH_MODE=fixed_binary",
                "RUN_LENGTH_FIXED_BINARY_ALPHA_NUM_EXPAND_ROUNDS=2",
                "RUN_LENGTH_FIXED_BINARY_ALPHA_EXPAND_TRAIN_PER_BIT=456",
                "RUN_LENGTH_FIXED_BINARY_ALPHA_TRAIN_BATCH_SIZE=77",
                "RUN_LENGTH_FIXED_BINARY_ALPHA_EVAL_BATCH_SIZE=78",
                "RUN_LENGTH_FIXED_BINARY_ALPHA_WARMUP_STEPS=111",
                "RUN_LENGTH_FIXED_BINARY_BEAM_JOB_NAME=custom-beam",
                "RUN_LENGTH_FIXED_BINARY_BEAM_MAX_ROUND=2",
                "RUN_LENGTH_FIXED_BINARY_BEAM_WARMUP_STEPS=222",
                "RUN_LENGTH_FIXED_BINARY_BEAM_TRAIN_BATCH_SIZE=79",
                "RUN_LENGTH_FIXED_BINARY_BEAM_EVAL_BATCH_SIZE=80",
                "RUN_LENGTH_FIXED_BINARY_BEAM_EXPAND_TRAIN_PER_BIT=333",
                "RUN_LENGTH_FIXED_BINARY_BEAM_MEM=4G",
                "RUN_LENGTH_FIXED_BINARY_BEAM_TIME=01:00:00",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rl_fixed_binary_custom")
    env["RUN_LENGTH_FIXED_BINARY_CONFIG"] = str(config_path)

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    assert "Loaded run-length fixed-binary override config" in combined
    assert "RUN_LENGTH_NUM_EXPAND_ROUNDS=3" in combined
    assert "TRAIN_BATCH_SIZE=17" in combined
    assert "--num-expand-rounds 2" in combined
    assert "--expand-train-per-bit 456" in combined
    assert "--per-device-train-batch-size 77" in combined
    assert "--per-device-eval-batch-size 78" in combined
    assert "--self-improve-warmup-steps 111" in combined
    assert "--job-name custom-beam" in combined
    assert "--max-round 2" in combined
    assert "--round-warmup-steps 222" in combined
    assert "--train-batch-size 79" in combined
    assert "--eval-batch-size 80" in combined
    assert "--expand-train-per-bit 333" in combined
    assert "--mem 4G" in combined
    assert "--time 01:00:00" in combined


def test_alpha10_seed_beam_dry_run_forwards_fixed_binary(tmp_path):
    assert run_length_alpha10_seed_beam.metric_score(
        {"max_bits_at_90_accuracy": 12, "eval_accuracy": 0.5, "composed_eval_accuracy": 0.25}
    ) == (12.0, 0.5, 0.25)

    template = tmp_path / "template"
    (template / "round_00").mkdir(parents=True)
    (template / "self_improvement_results.json").write_text("[]\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            str(BEAM),
            "--dry-run",
            "--out-root",
            str(tmp_path / "beam"),
            "--template-run",
            str(template),
            "--bit-composition-path-mode",
            "fixed_binary",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--bit-composition-path-mode fixed_binary" in result.stdout
