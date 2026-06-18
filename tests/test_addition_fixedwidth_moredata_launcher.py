from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "launchers" / "self" / "run_addition_fixedwidth_mixed_seed_mig.sbatch"
FULLPACK = ROOT / "launchers" / "self" / "run_addition_fixedwidth_mixed_recipe_fullpack.sh"
MIXED_SUBMITTER = ROOT / "launchers" / "self" / "submit_addition_fixedwidth_mixed_mig.sh"
MIXED_CONFIG = ROOT / "launchers" / "self" / "config" / "addition_fixedwidth_mixed.env"
SUBMITTER = ROOT / "launchers" / "self" / "submit_addition_fixedwidth_moredata_mig.sh"


def test_addition_fixedwidth_moredata_launchers_have_valid_bash_syntax():
    for launcher in (SEED, FULLPACK, MIXED_SUBMITTER, MIXED_CONFIG, SUBMITTER):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_addition_fixedwidth_seed_dry_run_prints_expected_command(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "seed")
    env["TRAIN_PER_DIGIT"] = "123"
    env["LR"] = "1e-4"
    env["SAVE_MODEL"] = "0"

    result = subprocess.run(
        ["bash", str(SEED)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "Stable seed symlink" in stdout
    assert "--addition-width-mode fixed_width_mixed_prompt" in stdout
    assert "--addition-composition-path-mode fixed_binary" in stdout
    assert "--initial-train-per-size 123" in stdout
    assert "--learning-rate 1e-4" in stdout
    assert "--save-model" not in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout


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


def test_addition_fixedwidth_mixed_submitter_dry_run_runs_three_branches(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["RUN_ROOT"] = str(tmp_path / "mixed")
    env["TRAIN_PER_DIGIT"] = "321"
    env["EXPAND_TRAIN_PER_DIGIT"] = "654"

    result = subprocess.run(
        ["bash", str(MIXED_SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "[INFO] Seed dry-run:" in stdout
    assert "[INFO] Fullpack dry-run:" in stdout
    assert "[INFO] Original-composition dry-run:" in stdout
    assert "--initial-train-per-size 321" in stdout
    assert "--expand-train-per-digit 654" in stdout
    assert "--addition-composition-path-mode random" in stdout


def test_addition_fixedwidth_mixed_submitter_uses_shared_sbatch_wrapping(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_stub = fake_bin / "sbatch"
    sbatch_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%q ' \"$@\" >> \"$SBATCH_LOG\"\n"
        "printf '\\n' >> \"$SBATCH_LOG\"\n"
        "echo '12345;cluster'\n",
        encoding="utf-8",
    )
    sbatch_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SBATCH_LOG"] = str(sbatch_log)
    env["RUN_ROOT"] = str(tmp_path / "mixed_submit")

    result = subprocess.run(
        ["bash", str(MIXED_SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    manifest_text = (tmp_path / "mixed_submit" / "submission_manifest.txt").read_text(encoding="utf-8")
    sbatch_text = sbatch_log.read_text(encoding="utf-8")

    assert "seed_job=12345" in manifest_text
    assert "fullpack_output=" in manifest_text
    assert "original_composition_output=" in manifest_text
    assert stdout.count("Submitted baseline job 12345") == 4
    assert stdout.count("Submitted original-composition baseline job 12345") == 3
    assert "--dependency afterok:12345" in sbatch_text
    assert "--wrap" in sbatch_text
    assert "ADDITION_COMPOSITION_PATH_MODE=fixed_binary" in sbatch_text
    assert "ADDITION_COMPOSITION_PATH_MODE=random" in sbatch_text


def test_addition_fixedwidth_mixed_submitter_can_source_custom_baseline_config(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_stub = fake_bin / "sbatch"
    sbatch_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%q ' \"$@\" >> \"$SBATCH_LOG\"\n"
        "printf '\\n' >> \"$SBATCH_LOG\"\n"
        "echo '54321;cluster'\n",
        encoding="utf-8",
    )
    sbatch_stub.chmod(0o755)

    config_path = tmp_path / "addition_mixed.env"
    config_path.write_text(
        "\n".join(
            [
                "ADDITION_MIXED_BASELINES_RAW='direct compose'",
                "ADDITION_MIXED_ORIGINAL_COMPOSITION_BASELINES_RAW='with_carry'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SBATCH_LOG"] = str(sbatch_log)
    env["RUN_ROOT"] = str(tmp_path / "mixed_submit_custom")
    env["ADDITION_MIXED_CONFIG"] = str(config_path)

    result = subprocess.run(
        ["bash", str(MIXED_SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    manifest_text = (tmp_path / "mixed_submit_custom" / "submission_manifest.txt").read_text(encoding="utf-8")
    sbatch_text = sbatch_log.read_text(encoding="utf-8")

    assert "Loaded addition fixed-width mixed config" in stdout
    assert "Baselines: direct compose" in stdout
    assert "Original-composition baselines: with_carry" in stdout
    assert stdout.count("Submitted baseline job 54321") == 2
    assert stdout.count("Submitted original-composition baseline job 54321") == 1
    assert "baselines=direct compose" in manifest_text
    assert "original_composition_baselines=with_carry" in manifest_text
    assert "BASELINE=direct" in sbatch_text
    assert "BASELINE=compose" in sbatch_text
    assert "BASELINE=with_carry" in sbatch_text


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
