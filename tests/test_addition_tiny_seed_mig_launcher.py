from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchers" / "self" / "run_addition_tiny_seed_mig.sbatch"
SHARED_SEED_SCRIPT = ROOT / "launchers" / "self" / "run_addition_seed_shared.sbatch"


def test_addition_tiny_seed_mig_launcher_has_valid_bash_syntax():
    for script in (SCRIPT, SHARED_SEED_SCRIPT):
        assert script.exists()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_addition_tiny_seed_mig_launcher_uses_expected_seed_fit_pipeline():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'MODEL_NAME="meta/models/tiny_gpt2_8l_384d"' in text
    assert "-m self.diagnostics.check_self_improvement_overfit" in text
    assert "--tasks addition" in text
    assert "--settings base" in text
    assert "-m self.experiments.seed_fit_experiment" in text
    assert "--task addition" in text
    assert "--init-from-scratch" in text
    assert "--tokenizer-mode fixed_char" in text
    assert 'STAGE1_LRS=("5e-5" "1e-4" "2e-4")' in text
    assert 'STAGE1_TRAIN_PER_DIGIT=20000' in text
    assert 'STAGE1_MAX_STEPS=16000' in text
    assert 'TRAIN_PER_DIGIT="${TRAIN_PER_DIGIT:-50000}"' in text
    assert 'MAX_STEPS="${MAX_STEPS:-48000}"' in text
    assert 'STAGE3_TRAIN_PER_DIGIT=$(( TRAIN_PER_DIGIT * 2 ))' in text
    assert 'STAGE3_MAX_STEPS=$(( MAX_STEPS * 2 ))' in text
    assert 'write_status "non_viable_stage1"' in text
    assert 'self_update_symlink "${WINNER_MODEL_DIR}" "${OUT_ROOT}/seed_model"' in text


def test_addition_tiny_seed_mig_launcher_dry_run_prints_expected_commands(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "addition_tiny_seed_mig")

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "[INFO] Dry run: skipping CUDA and scratch-model config probes." in stdout
    assert "-m self.diagnostics.check_self_improvement_overfit" in stdout
    assert "--tasks addition" in stdout
    assert "--settings base" in stdout
    assert "-m self.experiments.seed_fit_experiment" in stdout
    assert "--task addition" in stdout
    assert "--model-name meta/models/tiny_gpt2_8l_384d" in stdout
    assert "--init-from-scratch" in stdout
    assert "--tokenizer-mode fixed_char" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_addition_shared_seed_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "addition_shared_seed")
    env["MODEL_NAME"] = "stub-model"
    env["TRAIN_BATCH_SIZE"] = "16"

    result = subprocess.run(
        ["bash", str(SHARED_SEED_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "Shared checkpoint path (after completion)" in stdout
    assert "--model-name stub-model" in stdout
    assert "--initial-train-per-digit 50000" in stdout
    assert "--pseudo-label-mode none" in stdout
    assert "--per-device-train-batch-size 16" in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout
