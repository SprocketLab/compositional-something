from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED_LAUNCHER = ROOT / "launchers" / "self" / "run_multiplication_rectangular_square_seed_mig.sbatch"
DIAG_LAUNCHER = ROOT / "launchers" / "self" / "run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch"
WRAPPER = ROOT / "launchers" / "self" / "submit_multiplication_rectangular_square_probe_mig.sh"
RESWEEP_WRAPPER = ROOT / "launchers" / "self" / "submit_multiplication_rectangular_square_seed_resweep_mig.sh"


def test_multiplication_square_launchers_have_valid_bash_syntax():
    for script in (SEED_LAUNCHER, DIAG_LAUNCHER, WRAPPER, RESWEEP_WRAPPER):
        assert script.exists()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_multiplication_square_seed_launcher_uses_square_defaults():
    text = SEED_LAUNCHER.read_text(encoding="utf-8")

    assert 'RECIPE_NAME="multiplication_self_improve_v1"' in text
    assert 'FORMAT_VERSION="symbolic_v1"' in text
    assert 'PARTITIONS_SPEC="${PARTITIONS_SPEC:-1x1,1x2,1x3,2x1,2x2,2x3,3x1,3x2,3x3}"' in text
    assert "-m self.experiments.rectangular_multiplication_recipe_seed_fit" in text


def test_multiplication_square_seed_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "square_seed")

    result = subprocess.run(
        ["bash", str(SEED_LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe multiplication_self_improve_v1" in stdout
    assert "--format-version symbolic_v1" in stdout
    assert "[INFO] Partitions: 1x1,1x2,1x3,2x1,2x2,2x3,3x1,3x2,3x3" in stdout
    assert "--train-per-partition 50000" in stdout
    assert "--max-steps 10000" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_multiplication_square_compose_diagnostic_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "square_diag")
    env["SEED_MODEL"] = str(tmp_path / "seed_model")

    result = subprocess.run(
        ["bash", str(DIAG_LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe multiplication_self_improve_v1" in stdout
    assert "--format-version symbolic_v1" in stdout
    assert "[INFO] Seed partitions: 1x1,1x2,1x3,2x1,2x2,2x3,3x1,3x2,3x3" in stdout
    assert "[INFO] Frontier partitions: 4x4,4x5,4x6,4x7,4x8,5x4,5x5,5x6,5x7,5x8,6x4,6x5,6x6,6x7,6x8,7x4,7x5,7x6,7x7,7x8,8x4,8x5,8x6,8x7,8x8" in stdout
    assert "--mode compose" in stdout
    assert "--train-per-partition 2000" in stdout
    assert "--max-steps 3000" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_multiplication_square_probe_wrapper_dry_run_emits_seed_and_diagnostic(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "square_probe")

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
    assert "[INFO] seed_job_id=dryrun_seed" in stdout
    assert "[INFO] diagnostic_job_id=dryrun_diag" in stdout
    assert "--job-name mult-rect-square-seed" in combined
    assert "--job-name mult-rect-square-diag" in combined
    assert "--dependency afterok:dryrun_seed" in combined
    assert r"--export ALL\,OUT_ROOT=" in combined
    assert r"\,SEED_MODEL=" in combined
    assert "multiplication_rectangular_square_probe_" not in stdout
    assert "[INFO] Seed summary:" in stdout
    assert "[INFO] Diagnostic summary:" in stdout
    assert "[INFO] Status: dry_run" in stdout


def test_multiplication_square_seed_resweep_wrapper_dry_run_emits_six_jobs(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "square_resweep")

    result = subprocess.run(
        ["bash", str(RESWEEP_WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    combined = result.stdout + result.stderr
    assert stdout.count("[INFO] square_seed_job=") == 6
    assert "train_50000_lr_2em5" in stdout
    assert "train_50000_lr_5em5" in stdout
    assert "train_50000_lr_1em4" in stdout
    assert "train_100000_lr_2em5" in stdout
    assert "train_100000_lr_5em5" in stdout
    assert "train_100000_lr_1em4" in stdout
    assert "--job-name mult-rect-square-seed-train-50000-lr-2em5" in combined
    assert "--output" in combined
    assert "--error" in combined
    assert r"--export ALL\,OUT_ROOT=" in combined
    assert r"\,LR=1e-4\," in combined
    assert "[INFO] Status: dry_run" in stdout
