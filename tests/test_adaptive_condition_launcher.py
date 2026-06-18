from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_adaptive_condition_ailab.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_adaptive_condition_pilots_ailab.sh"


def test_adaptive_condition_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_adaptive_condition_launcher_prints_context_and_wires_command(tmp_path: Path):
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[PYTHON_STUB]'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "condition"

    env = os.environ.copy()
    env.update(
        {
            "TASK": "run_length",
            "CONDITION": "config",
            "FIXTURE": str(fixture),
            "OUT_DIR": str(out_dir),
            "RUN_COMPILE_CHECK": "0",
            "PYTHON_BIN": str(python_stub),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
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
    assert "[INFO] Task/condition: run_length/config" in stdout
    assert f"[INFO] Output dir: {out_dir}" in stdout
    assert f"[INFO] Fixture: {fixture}" in stdout
    assert "[INFO] Adaptive proposal condition" in stdout
    assert "self.experiments.adaptive_self_improvement" in stdout
    assert "--task run_length" in stdout
    assert "--condition config" in stdout
    assert f"--proposal-fixture-jsonl {fixture}" in stdout


def test_adaptive_condition_submitter_dry_run_writes_manifest(tmp_path: Path):
    out_root = tmp_path / "adaptive_condition_submit"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "LOG_DIR": str(tmp_path / "logs"),
            "PYTHON_BIN": sys.executable,
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((out_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["slurm"]["partition"] == "ailab"
    assert manifest["slurm"]["gres"] == "gpu:h200:1"
    assert manifest["jobs"]["addition_config"]["job_id"] == "dryrun-adapt-addition-config"
    assert manifest["jobs"]["run_length_program"]["job_id"] == "dryrun-adapt-run-length-program"

    combined = result.stdout + result.stderr
    assert "--job-name adapt-addition-config" in combined
    assert "--partition ailab" in combined
    assert "--gres gpu:h200:1" in combined
