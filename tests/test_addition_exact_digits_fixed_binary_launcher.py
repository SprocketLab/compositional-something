from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "launchers" / "self" / "submit_addition_exact_digits_fixed_binary_mig.sh"


def test_addition_exact_digits_fixed_binary_submitter_has_valid_bash_syntax():
    assert SUBMITTER.exists()
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_addition_exact_digits_fixed_binary_submitter_dry_run_emits_full_pack(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["RUN_ROOT"] = str(tmp_path / "exact_digits_fixed_binary")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    manifest = tmp_path / "exact_digits_fixed_binary" / "manifest.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")

    assert stdout.count("[DRYRUN] baseline=") == 5
    for baseline in ["short_only", "direct", "with_carry", "with_carry_filtered", "compose_corrupt"]:
        assert f"BASELINE={baseline}" in stdout
        assert f"\t{baseline}\tfixed_binary\t" in manifest_text
    assert "ADDITION_COMPOSITION_PATH_MODE=fixed_binary" in stdout
    assert "NUM_EXPAND_ROUNDS=8" in stdout
    assert "EXPAND_NUM_DIGITS=2" in stdout
    assert "SEED_REPLAY_TRAIN_PER_DIGIT=5000" in stdout
    assert "EXPAND_TRAIN_PER_DIGIT=10000" in stdout
