from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "launchers" / "self" / "submit_main_experiments_ailab.sh"


def test_main_experiments_submitter_has_valid_bash_syntax():
    assert SUBMITTER.exists()
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_main_experiments_submitter_dry_run_writes_manifest(tmp_path: Path):
    env = os.environ.copy()
    out_root = tmp_path / "main_experiments"
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(out_root)
    env["LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest_path = out_root / "submission_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["slurm"] == {
        "partition": "ailab",
        "gres": "gpu:h200:1",
        "time": "72:00:00",
    }
    assert payload["jobs"]["run_length_run_state"]["job_id"] == "dryrun-main-rl-runstate"
    assert payload["jobs"]["addition_recipe_fullpack"]["job_id"] == "dryrun-main-add-fullpack"

    combined = result.stdout + result.stderr
    assert "--job-name main-rl-runstate" in combined
    assert "--job-name main-add-fullpack" in combined
    assert "RUN_LENGTH_TARGET_MODE=run_state" in combined
    assert "ADDITION_COMPOSITION_PATH_MODE=fixed_binary" in combined
