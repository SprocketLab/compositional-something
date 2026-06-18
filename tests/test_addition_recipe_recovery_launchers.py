from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOCUSED = ROOT / "launchers" / "self" / "run_addition_recipe_focused.sh"
FULLPACK = ROOT / "launchers" / "self" / "run_addition_recipe_fullpack.sh"
RECOVERY = ROOT / "launchers" / "self" / "run_addition_recipe_recovery.sh"


def test_addition_recipe_launchers_have_valid_bash_syntax():
    for launcher in (FOCUSED, FULLPACK, RECOVERY):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_addition_recipe_focused_launcher_dry_run_prints_recipe_command(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "focused")
    env["BASELINE"] = "with_carry_filtered"

    result = subprocess.run(
        ["bash", str(FOCUSED)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe arithmetic_self_improve_v1" in stdout
    assert "--treat-seed-as-round-zero" in stdout
    assert "--seed-range-train-mode direct_pseudo" in stdout
    assert "--expand-train-per-digit 5000" in stdout
    assert "--bucket-train-batches-by-digits" in stdout
    assert "--composed-strategy with_carry_filtered" in stdout
    assert "[INFO] DRY_RUN=1; command not executed." in stdout


def test_addition_recipe_fullpack_launcher_dry_run_prints_fullpack_defaults(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "fullpack")
    env["BASELINE"] = "with_carry_filtered"
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "missing_paper_schedule.env")

    result = subprocess.run(
        ["bash", str(FULLPACK)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe arithmetic_self_improve_v1" in stdout
    assert "--num-expand-rounds 8" in stdout
    assert "--expand-num-digits 3" in stdout
    assert "--expand-train-per-digit 5000" in stdout
    assert "--addition-composition-path-mode random" in stdout
    assert "Addition composition path mode: random" in stdout
    assert "--early-stop-patience 2" in stdout
    assert "--early-stop-expanded-eval-threshold 0.01" in stdout
    assert "--early-stop-frontier-train-threshold 0.50" in stdout
    assert "--composed-strategy with_carry_filtered" in stdout


def test_addition_recipe_fullpack_launcher_forwards_fixed_binary_composition_path(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "fullpack")
    env["BASELINE"] = "with_carry_filtered"
    env["ADDITION_COMPOSITION_PATH_MODE"] = "fixed_binary"
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "missing_paper_schedule.env")

    result = subprocess.run(
        ["bash", str(FULLPACK)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--addition-composition-path-mode fixed_binary" in stdout
    assert "Addition composition path mode: fixed_binary" in stdout


def test_addition_recipe_fullpack_launcher_can_source_paper_schedule_env(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "fullpack")
    env["BASELINE"] = "with_carry_filtered"
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "paper_schedule_selection.env")
    Path(env["PAPER_SCHEDULE_ENV"]).write_text(
        "\n".join(
            [
                "ADDITION_EXPAND_NUM_DIGITS=2",
                "ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT=8000",
                "ADDITION_EXPAND_TRAIN_PER_DIGIT=8000",
                "",
            ]
        ),
        encoding="utf-8",
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
    assert "Paper schedule env:" in stdout
    assert "Resolved schedule: num_expand_rounds=8 expand_num_digits=2 seed_replay_train_per_digit=8000 expand_train_per_digit=8000" in stdout
    assert "--expand-num-digits 2" in stdout
    assert "--seed-replay-train-per-digit 8000" in stdout
    assert "--expand-train-per-digit 8000" in stdout


def test_addition_recipe_recovery_launcher_dry_run_chains_all_stages(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "recovery")

    result = subprocess.run(
        ["bash", str(RECOVERY)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "self.diagnostics.addition_recipe_diagnostic" in stdout
    assert "--recipe arithmetic_self_improve_v1" in stdout
    assert "--num-expand-rounds 1" in stdout
    assert "--num-expand-rounds 8" in stdout
