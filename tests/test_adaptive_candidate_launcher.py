from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_adaptive_candidate_training_ailab.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_adaptive_candidate_training_ailab.sh"
BASE_CONFIG = ROOT / "launchers" / "self" / "config" / "adaptive_candidate_base.env"


def test_adaptive_candidate_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_adaptive_candidate_launcher_wires_packed_cached_local_workers(tmp_path: Path):
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[PYTHON_STUB]'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "TASK": "addition",
            "OUT_DIR": str(tmp_path / "run"),
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
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
    assert f"[INFO] Loaded adaptive config: {BASE_CONFIG}" in stdout
    assert "Candidate local parallelism/pack/cache-base-state: 4/2/1" in stdout
    assert "--candidate-local-pack-size 2" in stdout
    assert "--candidate-local-cache-base-state" in stdout


def test_adaptive_candidate_submitter_dry_run_writes_matrix_manifest(tmp_path: Path):
    out_root = tmp_path / "adaptive_candidate_submit"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "LOG_DIR": str(tmp_path / "logs"),
            "PYTHON_BIN": sys.executable,
            "TASKS": "addition",
            "CONDITIONS": "config",
            "OUTCOME_TRACE_TARGET_MODES": "numeric",
            "PROPOSAL_GRPO_ZERO_VARIANCE_MODES": "fixed_baseline skip",
            "NUM_CANDIDATES_LIST": "8 16",
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
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
    assert manifest["tasks"] == ["addition"]
    assert manifest["conditions"] == ["config"]
    assert manifest["proposal_grpo_zero_variance_modes"] == ["fixed_baseline", "skip"]
    assert manifest["num_candidates_list"] == [8, 16]
    assert len(manifest["jobs"]) == 4
    assert (
        manifest["jobs"]["addition-config-numeric-n8-grpo-fixed_baseline"]["job_id"]
        == "dryrun-adaptive-cand-addition-config-numeric-n8-fixed-baseline"
    )
    assert manifest["jobs"]["addition-config-numeric-n16-grpo-skip"]["output_dir"] == str(
        out_root / "addition-config-numeric-n16-grpo-skip"
    )

    combined = result.stdout + result.stderr
    assert "--job-name adaptive-cand-addition-config-numeric-n8-fixed-baseline" in combined
    assert "--export" in combined
    assert f"ADAPTIVE_CONFIG_FILES={BASE_CONFIG}" in combined
