from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers" / "self" / "run_multiplication_rectangular_seed_mig.sbatch"
WRAPPER = ROOT / "launchers" / "self" / "submit_multiplication_rectangular_seed_sweep_mig.sh"


def test_multiplication_seed_launchers_have_valid_bash_syntax():
    for script in (LAUNCHER, WRAPPER):
        assert script.exists()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_multiplication_seed_launcher_uses_edge_only_recipe_defaults():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'RECIPE_NAME="multiplication_self_improve_v1"' in text
    assert 'FORMAT_VERSION="symbolic_v1"' in text
    assert 'PARTITIONS_SPEC="${PARTITIONS_SPEC:-1x1,1x2,1x3,1x4,1x5,1x6,2x1,3x1,4x1,5x1,6x1}"' in text
    assert "-m self.rectangular_multiplication_recipe_seed_fit" in text
    assert "--recipe" in text
    assert "--format-version" in text
    assert "--partitions" in text
    assert 'TRAIN_PER_PARTITION="${TRAIN_PER_PARTITION:-50000}"' in text
    assert 'MAX_STEPS="${MAX_STEPS:-10000}"' in text
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"' in text
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"' in text


def test_multiplication_seed_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "seed_launcher")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe multiplication_self_improve_v1" in stdout
    assert "--format-version symbolic_v1" in stdout
    assert "[INFO] Partitions: 1x1,1x2,1x3,1x4,1x5,1x6,2x1,3x1,4x1,5x1,6x1" in stdout
    assert "--train-per-partition 50000" in stdout
    assert "--max-steps 10000" in stdout
    assert "--per-device-train-batch-size 256" in stdout
    assert "--per-device-eval-batch-size 256" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_multiplication_seed_launcher_can_forward_skip_train_eval(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["SKIP_TRAIN_EVAL"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "seed_launcher_skip_eval")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "[INFO] Skip train eval: 1" in stdout
    assert "--skip-train-eval" in stdout


def test_multiplication_seed_wrapper_dry_run_emits_six_stage1_jobs(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "seed_wrapper")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert stdout.count("[INFO] Stage 1 dry-run job=") == 6
    assert "train_25000_lr_2em5" in stdout
    assert "train_25000_lr_5em5" in stdout
    assert "train_25000_lr_1em4" in stdout
    assert "train_50000_lr_2em5" in stdout
    assert "train_50000_lr_5em5" in stdout
    assert "train_50000_lr_1em4" in stdout
    assert "[INFO] Status: dry_run" in stdout
