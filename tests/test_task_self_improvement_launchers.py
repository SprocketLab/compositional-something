from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_task_self_improvement.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_budget_grid_self_improvement.sh"


def test_task_self_improvement_launchers_have_valid_bash_syntax():
    for launcher in (RUNNER, SUBMITTER):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_task_self_improvement_runner_dry_run_prints_command_and_skips_preflight(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["TASK_MODULE"] = "self.run_length_self_improvement"
    env["OUT_ROOT"] = str(tmp_path / "task")
    env["MODEL_NAME"] = "stub-model"
    env["MAX_STEPS"] = "12"
    env["EXTRA_ARGS"] = (
        "--initial-min-bits 4 "
        "--initial-max-bits 8 "
        "--initial-train-per-bit 5 "
        "--initial-eval-per-bit 6 "
        "--expand-num-bits 4 "
        "--pseudo-label-mode compose"
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
    assert "self.run_length_self_improvement" in stdout
    assert "--model-name stub-model" in stdout
    assert "--max-steps 12" in stdout
    assert "--initial-train-per-bit 5" in stdout
    assert "--pseudo-label-mode compose" in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout


def test_budget_grid_submitter_dry_run_writes_manifest(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["BASE_OUT"] = str(tmp_path / "grid")
    env["TASKS"] = "run_length"
    env["MODES"] = "none compose"
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

    assert stdout.count("[DRY RUN] sibg-run_length") == 6
    assert "DRYRUN-run_length-none-small" in manifest_text
    assert "DRYRUN-run_length-compose-large" in manifest_text
    assert "stub-model" in manifest_text
