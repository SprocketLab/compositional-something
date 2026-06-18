from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchers" / "self" / "run_addition_tiny_seed_mig.sbatch"


def test_addition_tiny_seed_mig_launcher_has_valid_bash_syntax():
    assert SCRIPT.exists()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_addition_tiny_seed_mig_launcher_uses_expected_seed_fit_pipeline():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'MODEL_NAME="meta/models/tiny_gpt2_8l_384d"' in text
    assert "-m self.check_self_improvement_overfit" in text
    assert "--tasks addition" in text
    assert "--settings base" in text
    assert "-m self.seed_fit_experiment" in text
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
    assert 'ln -sfn "${WINNER_MODEL_DIR}" "${OUT_ROOT}/seed_model"' in text


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
    assert "-m self.check_self_improvement_overfit" in stdout
    assert "--tasks addition" in stdout
    assert "--settings base" in stdout
    assert "-m self.seed_fit_experiment" in stdout
    assert "--task addition" in stdout
    assert "--model-name meta/models/tiny_gpt2_8l_384d" in stdout
    assert "--init-from-scratch" in stdout
    assert "--tokenizer-mode fixed_char" in stdout
    assert "[INFO] Status: dry_run" in stdout
