from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_figure2_recipe_aggressive.sh"
SUBMITTER = ROOT / "launchers" / "self" / "submit_figure2_recipe_aggressive.sh"
RETUNE = ROOT / "launchers" / "self" / "run_figure2_paper_retune.sh"


def test_figure2_recipe_aggressive_launchers_have_valid_bash_syntax():
    for launcher in (RUNNER, SUBMITTER, RETUNE):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_figure2_recipe_aggressive_runner_dry_run_prints_recipe_schedule(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure2_recipe")
    env["INITIAL_TRAIN_PER_BIT"] = "10000"
    env["RUN_LENGTH_BIT_COMPOSITION_PATH_MODE"] = "fixed_binary"
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "missing_paper_schedule.env")

    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "--recipe algorithmic_self_improve_v1" in stdout
    assert "--initial-min-size 8" in stdout
    assert "--initial-max-size 16" in stdout
    assert "--initial-train-per-size 10000" in stdout
    assert "--num-expand-rounds 8" in stdout
    assert "--expand-num-bits 4" in stdout
    assert "Task schedule run_length: seed_model=" in stdout
    assert "bit_composition_path_mode=fixed_binary" in stdout
    assert "--bit-composition-path-mode fixed_binary" in stdout
    assert "num_expand_rounds=8 expand_num_bits=4 expand_train_per_bit=1200" in stdout
    assert "--bucket-train-batches-by-bits" in stdout
    assert "--bucket-train-batches-by-size" in stdout
    assert "--pseudo-label-mode compose" in stdout


def test_figure2_recipe_aggressive_runner_can_source_paper_schedule_env(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure2_recipe")
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "paper_schedule_selection.env")
    Path(env["PAPER_SCHEDULE_ENV"]).write_text(
        "\n".join(
            [
                "RUN_LENGTH_EXPAND_NUM_BITS=4",
                "RUN_LENGTH_EXPAND_TRAIN_PER_BIT=1200",
                "",
            ]
        ),
        encoding="utf-8",
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
    assert "expand_num_bits=4" in stdout
    assert "expand_train_per_bit=1200" in stdout
    assert "Figure style: paper" in stdout


def test_figure2_recipe_aggressive_submitter_dry_run_prints_submission(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure2_recipe_submit")
    env["TRAIN_BATCH_SIZE"] = "256"
    env["EVAL_BATCH_SIZE"] = "256"
    env["INITIAL_TRAIN_PER_BIT"] = "10000"

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "run_figure2_recipe_aggressive.sh" in stdout
    assert "STAGE=all" in stdout
    assert "TASKS=run_length" in stdout
    assert "TRAIN_BATCH_SIZE=256" in stdout
    assert "EVAL_BATCH_SIZE=256" in stdout
    assert "INITIAL_TRAIN_PER_BIT=10000" in stdout
    assert "Resolved partition: all" in stdout
    assert "Resolved gres: gpu:a100:1" in stdout
    assert "Resolved constraint: a100&gpu40&nomig" in stdout
    assert "[INFO] DRY_RUN=1; sbatch not executed." in stdout


def test_figure2_recipe_aggressive_submitter_uses_shared_wrap_helper(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_stub = fake_bin / "sbatch"
    sbatch_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%q ' \"$@\" >> \"$SBATCH_LOG\"\n"
        "printf '\\n' >> \"$SBATCH_LOG\"\n"
        "echo '24680'\n",
        encoding="utf-8",
    )
    sbatch_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SBATCH_LOG"] = str(sbatch_log)
    env["OUT_ROOT"] = str(tmp_path / "figure2_recipe_submit")
    env["TRAIN_BATCH_SIZE"] = "128"
    env["DEVICE_TARGET"] = "mig_10gb"

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    sbatch_text = sbatch_log.read_text(encoding="utf-8")

    assert "[INFO] Submitted fig2-rec-aggr -> 24680" in stdout
    assert "--partition=mig" in sbatch_text
    assert "--gres=gpu:1g.10gb:1" in sbatch_text
    assert "--wrap=" in sbatch_text
    assert "PYTHONPATH=." in sbatch_text
    assert "TRAIN_BATCH_SIZE=128" in sbatch_text
    assert "run_figure2_recipe_aggressive.sh" in sbatch_text


def test_figure2_paper_retune_launcher_dry_run_prints_candidate_grid(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "paper_retune")
    env["SELECTION_JSON"] = str(tmp_path / "paper_schedule_selection.json")
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "paper_schedule_selection.env")

    result = subprocess.run(
        ["bash", str(RETUNE)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "self.figure2_paper_retune" in stdout
    assert "addition/pilots/expand4_train8000" in stdout
    assert "addition/pilots/expand2_train5000" in stdout
    assert "selection files not written" in stdout
