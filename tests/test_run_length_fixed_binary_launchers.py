from __future__ import annotations

import os
import subprocess
from pathlib import Path

from self.experiments import run_length_alpha10_seed_beam


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "launchers" / "self" / "submit_run_length_fixed_binary_mig.sh"
BEAM = ROOT / "launchers" / "self" / "run_run_length_alpha10_seed_beam_mig.py"


def test_run_length_fixed_binary_submitter_has_valid_bash_syntax():
    assert SUBMITTER.exists()
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_run_length_fixed_binary_submitter_dry_run_forwards_fixed_binary(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rl_fixed_binary")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    assert "--bit-composition-path-mode fixed_binary" in combined
    assert "BIT_COMPOSITION_PATH_MODE=fixed_binary" in combined
    assert "-m self.experiments.run_length_alpha10_seed_beam" in combined
    assert "run_length_fixed_binary" in combined
    assert "run_manifest.json" in combined


def test_alpha10_seed_beam_dry_run_forwards_fixed_binary(tmp_path):
    assert run_length_alpha10_seed_beam.metric_score(
        {"max_bits_at_90_accuracy": 12, "eval_accuracy": 0.5, "composed_eval_accuracy": 0.25}
    ) == (12.0, 0.5, 0.25)

    template = tmp_path / "template"
    (template / "round_00").mkdir(parents=True)
    (template / "self_improvement_results.json").write_text("[]\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            str(BEAM),
            "--dry-run",
            "--out-root",
            str(tmp_path / "beam"),
            "--template-run",
            str(template),
            "--bit-composition-path-mode",
            "fixed_binary",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--bit-composition-path-mode fixed_binary" in result.stdout
