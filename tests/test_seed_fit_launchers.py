from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_seed_fit_experiment.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_seed_fit_grid.sh"


def test_seed_fit_launchers_have_valid_bash_syntax():
    for launcher in (RUNNER, SUBMITTER):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_seed_fit_runner_dry_run_prints_command_and_skips_preflight(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["TASK_NAME"] = "run_length"
    env["OUT_ROOT"] = str(tmp_path / "seedfit")
    env["MODEL_NAME"] = "stub-model"
    env["MAX_STEPS"] = "12"
    env["EXTRA_ARGS"] = (
        "--format-version legacy "
        "--initial-min-size 4 "
        "--initial-max-size 8 "
        "--initial-train-per-size 5 "
        "--initial-eval-per-size 6 "
        "--expand-num-size 4"
    )

    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "[INFO] DRY_RUN=1; skipping CUDA and model/tokenizer preflight." in stdout
    assert "self.experiments.seed_fit_experiment" in stdout
    assert "--task run_length" in stdout
    assert "--model-name stub-model" in stdout
    assert "--max-steps 12" in stdout
    assert "--initial-train-per-size 5" in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout


def test_seed_fit_submitter_dry_run_writes_grid_manifest(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["BASE_OUT"] = str(tmp_path / "grid")
    env["TASKS"] = "run_length"
    env["RUN_LENGTH_TRAIN_SIZES"] = "10 20"
    env["RUN_LENGTH_STEP_BUDGETS"] = "960 3840"
    env["MODEL_NAME"] = "stub-model"

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    manifest = tmp_path / "grid" / "submission_jobs.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")

    assert stdout.count("[DRY RUN] sfit-run_length") == 4
    assert "DRYRUN-run_length-10-960" in manifest_text
    assert "DRYRUN-run_length-20-3840" in manifest_text
    assert "stub-model" in manifest_text
