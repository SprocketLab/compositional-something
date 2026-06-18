from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_adaptive_candidate_training_ailab.sbatch"
BASE_CONFIG = ROOT / "launchers" / "self" / "config" / "adaptive_candidate_base.env"


def test_adaptive_candidate_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


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
    assert "Candidate local parallelism/pack/cache-base-state: 4/2/1" in stdout
    assert "--candidate-local-pack-size 2" in stdout
    assert "--candidate-local-cache-base-state" in stdout
