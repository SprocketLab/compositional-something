from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers" / "self" / "run_addition_fullpack_filtered.sbatch"
SUBMIT_WRAPPER = ROOT / "launchers" / "self" / "submit_addition_fullpack_filtered_mig.sh"
CONFIG = ROOT / "launchers" / "self" / "config" / "addition_fullpack_filtered.env"


def test_addition_fullpack_launcher_has_valid_bash_syntax():
    assert LAUNCHER.exists()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_addition_fullpack_submit_wrapper_has_valid_bash_syntax():
    assert SUBMIT_WRAPPER.exists()
    subprocess.run(["bash", "-n", str(SUBMIT_WRAPPER)], check=True)


def test_addition_fullpack_submit_wrapper_config_exists():
    assert CONFIG.exists()
    text = CONFIG.read_text(encoding="utf-8")
    assert "ADDITION_FULLPACK_FILTERED_BASELINES_RAW" in text
    assert "short_only direct with_carry with_carry_filtered compose_corrupt" in text
    assert "${BASELINES:-" in text


def test_addition_fullpack_launcher_uses_tiny_seed_mig_defaults():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '#SBATCH --gres=gpu:1g.10gb:1' in text
    assert '#SBATCH --partition=mig' in text
    assert 'SEED_MODEL_LINK="${ROOT_DIR}/artifacts/models/addition_tiny_seed_best"' in text
    assert 'TOKENIZER_MODE="${TOKENIZER_MODE:-fixed_char}"' in text
    assert '--tokenizer-mode "${TOKENIZER_MODE}"' in text
    assert '--treat-seed-as-round-zero' in text
    assert '--seed-range-train-mode direct_pseudo' in text
    assert '--initial-train-per-digit 0' in text
    assert '--expand-num-digits 5' in text
    assert 'SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-2500}"' in text
    assert 'EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-5000}"' in text
    assert '--seed-replay-train-per-digit "${SEED_REPLAY_TRAIN_PER_DIGIT}"' in text
    assert '--expand-train-per-digit "${EXPAND_TRAIN_PER_DIGIT}"' in text
    assert 'LEARNING_RATE="${LEARNING_RATE:-2e-4}"' in text
    assert '--per-device-train-batch-size "${TRAIN_BATCH_SIZE}"' in text
    assert '--per-device-eval-batch-size "${EVAL_BATCH_SIZE}"' in text
    assert '--skip-save-model' not in text
    assert 'build_fixed_char_tokenizer' in text


def test_addition_fullpack_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["BASELINE"] = "with_carry_filtered"
    env["OUT_ROOT"] = str(tmp_path / "addition_fullpack")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "[INFO] DRY_RUN=1; skipping CUDA and model/tokenizer preflight." in stdout
    assert "--tokenizer-mode fixed_char" in stdout
    assert "--treat-seed-as-round-zero" in stdout
    assert "--seed-range-train-mode direct_pseudo" in stdout
    assert "--initial-train-per-digit 0" in stdout
    assert "--expand-num-digits 5" in stdout
    assert "--seed-replay-train-per-digit 2500" in stdout
    assert "--expand-train-per-digit 5000" in stdout
    assert "--learning-rate 2e-4" in stdout
    assert "--per-device-train-batch-size 32" in stdout
    assert "--per-device-eval-batch-size 64" in stdout
    assert "--composed-strategy with_carry_filtered" in stdout


def test_addition_fullpack_submit_wrapper_dry_run_lists_five_jobs(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "addition_fullpack")

    result = subprocess.run(
        ["bash", str(SUBMIT_WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    combined = result.stdout + result.stderr
    assert stdout.count("[DRY_RUN] baseline=") == 5
    assert "baseline=short_only" in stdout
    assert "baseline=direct" in stdout
    assert "baseline=with_carry" in stdout
    assert "baseline=with_carry_filtered" in stdout
    assert "baseline=compose_corrupt" in stdout
    assert str(tmp_path / "addition_fullpack") in stdout
    assert "--job-name addition-fullpack-short-only" in combined
    assert "--output" in combined
    assert "--error" in combined
    assert r"--export ALL\,BASELINE=with_carry_filtered\,OUT_ROOT=" in combined
    assert "dryrun-addition-fullpack-compose-corrupt" in stdout


def test_addition_fullpack_submit_wrapper_honors_baselines_env(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["BASELINES"] = "direct compose_corrupt"
    env["OUT_ROOT"] = str(tmp_path / "addition_fullpack")

    result = subprocess.run(
        ["bash", str(SUBMIT_WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert stdout.count("[DRY_RUN] baseline=") == 2
    assert "baseline=direct" in stdout
    assert "baseline=compose_corrupt" in stdout
    assert "baseline=short_only" not in stdout


def test_addition_fullpack_submit_wrapper_honors_override_config(tmp_path):
    config = tmp_path / "addition_override.env"
    config.write_text(
        'ADDITION_FULLPACK_FILTERED_BASELINES_RAW="with_carry_filtered"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ADDITION_FULLPACK_FILTERED_CONFIG"] = str(config)
    env["OUT_ROOT"] = str(tmp_path / "addition_fullpack")

    result = subprocess.run(
        ["bash", str(SUBMIT_WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert stdout.count("[DRY_RUN] baseline=") == 1
    assert "baseline=with_carry_filtered" in stdout
    assert f"Loaded addition fullpack filtered override config: {config}" in stdout
