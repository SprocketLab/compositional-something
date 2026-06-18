from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers" / "self" / "run_addition_their_recipe_diagnostic.sh"


def test_addition_their_recipe_diagnostic_launcher_has_valid_bash_syntax():
    assert LAUNCHER.exists()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_addition_their_recipe_diagnostic_launcher_dry_run_prints_expected_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "their_recipe_diag")
    env["DEVICE_TARGET"] = "local_a100_40gb"
    env["TRAIN_BATCH_SIZE"] = "256"
    env["EVAL_BATCH_SIZE"] = "512"
    env["MAX_STEPS"] = "123"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe arithmetic_self_improve_v1" in stdout
    assert "--device-target local_a100_40gb" in stdout
    assert "--per-device-train-batch-size 256" in stdout
    assert "--per-device-eval-batch-size 512" in stdout
    assert "--max-steps 123" in stdout
    assert "--dry-run" in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout
