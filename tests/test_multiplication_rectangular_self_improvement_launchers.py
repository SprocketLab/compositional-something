from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers" / "self" / "run_multiplication_rectangular_self_improvement_mig.sbatch"
WRAPPER = ROOT / "launchers" / "self" / "submit_multiplication_rectangular_fullpack_mig.sh"


def test_multiplication_self_improvement_launchers_have_valid_bash_syntax():
    for script in (LAUNCHER, WRAPPER):
        assert script.exists()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_multiplication_self_improvement_launcher_uses_recipe_seed_defaults():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'RECIPE_NAME="multiplication_self_improve_v1"' in text
    assert 'FORMAT_VERSION="symbolic_v1"' in text
    assert 'SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/multiplication_rectangular_seed_best}"' in text
    assert "-m self.rectangular_multiplication_self_improvement" in text
    assert 'BASELINE="${BASELINE:-compose}"' in text
    assert 'INITIAL_MAX_B_DIGITS="${INITIAL_MAX_B_DIGITS:-8}"' in text
    assert 'EXPAND_B_DIGITS="${EXPAND_B_DIGITS:-2}"' in text
    assert 'NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-4}"' in text


def test_multiplication_self_improvement_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rect_si")
    env["SEED_MODEL"] = str(ROOT / "artifacts" / "models" / "addition_recipe_seed_best")
    env["BASELINE"] = "compose_corrupt"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--seed-model" in stdout
    assert "--pseudo-label-mode compose_corrupt" in stdout
    assert "--corruption-rate 0.10" in stdout
    assert "--initial-max-b-digits 8" in stdout
    assert "--expand-b-digits 2" in stdout
    assert "--num-expand-rounds 4" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_multiplication_self_improvement_launcher_forwards_tune_overrides(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rect_si_tune")
    env["SEED_MODEL"] = str(ROOT / "artifacts" / "models" / "multiplication_rectangular_seed_best")
    env["BASELINE"] = "direct"
    env["FRONTIER_ROW_PROFILE"] = "hard_rows_v1"
    env["LEARNING_RATE"] = "1e-4"
    env["MAX_STEPS"] = "6000"
    env["INITIAL_MAX_B_DIGITS"] = "8"
    env["EXPAND_B_DIGITS"] = "1"
    env["NUM_EXPAND_ROUNDS"] = "8"
    env["SEED_REPLAY_TRAIN_PER_PARTITION"] = "3000"
    env["EXPAND_TRAIN_PER_PARTITION"] = "2000"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--pseudo-label-mode direct" in stdout
    assert "--frontier-row-profile hard_rows_v1" in stdout
    assert "--learning-rate 1e-4" in stdout
    assert "--max-steps 6000" in stdout
    assert "--expand-b-digits 1" in stdout
    assert "--num-expand-rounds 8" in stdout
    assert "--seed-replay-train-per-partition 3000" in stdout
    assert "--expand-train-per-partition 2000" in stdout


def test_multiplication_self_improvement_wrapper_dry_run_emits_four_baselines(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rect_si_pack")
    env["SEED_MODEL"] = str(ROOT / "artifacts" / "models" / "addition_recipe_seed_best")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    combined = result.stdout + result.stderr
    assert stdout.count("[INFO] baseline=") == 4
    assert "baseline=short_only" in stdout
    assert "baseline=direct" in stdout
    assert "baseline=compose" in stdout
    assert "baseline=compose_corrupt" in stdout
    assert "--job-name mult-rect-si-short-only" in combined
    assert "--output" in combined
    assert "--error" in combined
    assert r"--export ALL\,OUT_ROOT=" in combined
    assert r"\,BASELINE=compose_corrupt\," in combined
    assert "job_id=dryrun" in stdout
