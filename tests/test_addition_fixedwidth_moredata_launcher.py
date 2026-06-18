from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULLPACK = ROOT / "launchers" / "self" / "run_addition_fixedwidth_mixed_recipe_fullpack.sh"
SUBMITTER = ROOT / "launchers" / "self" / "submit_addition_fixedwidth_moredata_mig.sh"


def test_addition_fixedwidth_moredata_launchers_have_valid_bash_syntax():
    for launcher in (FULLPACK, SUBMITTER):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_addition_fixedwidth_fullpack_dry_run_forwards_moredata_overrides(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "fullpack")
    env["BASELINE"] = "with_carry_filtered"
    env["EXPAND_TRAIN_PER_DIGIT"] = "20000"
    env["ADDITION_COMPOSITION_PATH_MODE"] = "random"
    env["EXTRA_ARGS"] = (
        "--max-steps 4500 "
        "--self-improve-warmup-steps 0 "
        "--self-improve-stable-steps 3500 "
        "--self-improve-decay-steps 1000"
    )

    result = subprocess.run(
        ["bash", str(FULLPACK)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--addition-width-mode fixed_width_mixed_prompt" in stdout
    assert "--addition-sampling-mode balanced_visible_lengths" in stdout
    assert "--addition-composition-path-mode random" in stdout
    assert "--expand-train-per-digit 20000" in stdout
    assert "--max-steps 4500" in stdout
    assert "--self-improve-warmup-steps 0" in stdout
    assert "--self-improve-stable-steps 3500" in stdout
    assert "--self-improve-decay-steps 1000" in stdout
    assert "--composed-strategy with_carry_filtered" in stdout


def test_addition_fixedwidth_moredata_submitter_dry_run_emits_stage1_grid(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["RUN_ROOT"] = str(tmp_path / "moredata")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    manifest = tmp_path / "moredata" / "stage1_manifest.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")

    assert stdout.count("[DRYRUN] fixed_binary_expand") == 2
    assert stdout.count("[DRYRUN] random_expand") == 2
    assert "BASELINE=with_carry_filtered" in stdout
    assert "ADDITION_WIDTH_MODE=fixed_width_mixed_prompt" in stdout
    assert "ADDITION_COMPOSITION_PATH_MODE=fixed_binary" in stdout
    assert "ADDITION_COMPOSITION_PATH_MODE=random" in stdout
    assert "--max-steps 4500" in stdout
    assert "--max-steps 6000" in stdout
    assert "--self-improve-stable-steps 3500" in stdout
    assert "--self-improve-stable-steps 5000" in stdout
    assert "fixed_binary_expand20000_steps4500" in manifest_text
    assert "fixed_binary_expand40000_steps6000" in manifest_text
    assert "random_expand20000_steps4500" in manifest_text
    assert "random_expand40000_steps6000" in manifest_text
