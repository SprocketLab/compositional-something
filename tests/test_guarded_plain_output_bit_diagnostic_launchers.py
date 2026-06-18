from __future__ import annotations

import os
import subprocess
from pathlib import Path

from self.legacy.run_length_bit_cli import build_run_length_bit_parser


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_guarded_plain_output_bit_diagnostic_mig.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_guarded_plain_output_bit_diagnostic_mig.sh"
RUN_LENGTH_ALPHA10_BASELINE_PACK = ROOT / "launchers" / "self" / "submit_run_length_alpha10_baseline_pack_mig.sh"
RUN_LENGTH_ALPHA10_BASELINE_CONFIG = ROOT / "launchers" / "self" / "config" / "run_length_alpha10_baseline_pack.env"


def test_run_length_bit_cli_defaults_to_all_round_model_saving():
    parser = build_run_length_bit_parser(description="test", default_output_dir="out")
    args = parser.parse_args([])
    assert args.save_model_policy == "all_rounds"
    assert args.self_improve_warmup_steps is None
    assert args.bit_composition_path_mode == "random"


def test_guarded_plain_output_bit_launchers_have_valid_bash_syntax():
    for launcher in (RUNNER, SUBMITTER, RUN_LENGTH_ALPHA10_BASELINE_PACK, RUN_LENGTH_ALPHA10_BASELINE_CONFIG):
        assert launcher.exists()
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_guarded_plain_output_bit_runner_dry_run_prints_expected_commands(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["TASK"] = "run_length"
    env["SYMBOL_ALPHABET_SIZE"] = "3"
    env["ROUND_WARMUP_STEPS"] = "300"
    env["BIT_COMPOSITION_PATH_MODE"] = "fixed_binary"
    env["OUT_ROOT"] = str(tmp_path / "guarded_diag")

    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.replace("\\ ", " ")
    assert "-m self.experiments.seed_fit_experiment" in stdout
    assert "--task run_length" in stdout
    assert "--target-mode symbol_run_pair" in stdout
    assert "--symbol-alphabet-size 3" in stdout
    assert "--initial-min-size 8" in stdout
    assert "--initial-max-size 12" in stdout
    assert "--bucket-train-batches-by-size" in stdout
    assert "-m self.legacy.run_length_self_improvement" in stdout
    assert "--compose-arity exact2" in stdout
    assert "--bit-composition-path-mode fixed_binary" in stdout
    assert "--frontier-min-bits 18" in stdout
    assert "--treat-seed-as-round-zero" in stdout
    assert "--save-model-policy all_rounds" in stdout
    assert "--skip-save-model" not in stdout
    assert "--pseudo-label-mode direct" in stdout
    assert "--pseudo-label-mode compose" in stdout
    assert "--guarded-compose-rule run_length_no_boundary_continue" in stdout
    assert "--expand-num-bits 8" in stdout
    assert "--num-expand-rounds 1" in stdout
    assert "--self-improve-warmup-steps 300" in stdout
    assert "Target mode: symbol_run_pair" in stdout
    assert "Bit composition path mode: fixed_binary" in stdout
    assert "Round warmup steps: 300" in stdout
    assert "Save model policy: all_rounds" in stdout


def test_guarded_plain_output_bit_submitter_dry_run_prints_run_length(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "guarded_diag_submit")

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.replace("\\ ", " ")
    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    assert "TASK=run_length" in combined
    assert "--job-name guarded-bit-run-length" in combined
    assert "--output" in combined
    assert "--error" in combined
    assert "run_guarded_plain_output_bit_diagnostic_mig.sbatch" in combined
    assert "DRY_RUN=1; sbatch not executed for run_length." in stdout
    assert "job_id=dryrun-guarded-bit-run-length" in stdout


def test_run_length_alpha10_baseline_pack_dry_run_prints_three_baselines(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rl_alpha10_baselines")
    env["LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        ["bash", str(RUN_LENGTH_ALPHA10_BASELINE_PACK)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.replace("\\ ", " ")
    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    assert "run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model" in combined
    assert "--target-mode symbol_run_pair" in combined
    assert "--compose-arity exact2" in combined
    assert "--bit-composition-path-mode random" in combined
    assert "--num-expand-rounds 7" in combined
    assert "--expand-num-bits 9" in combined
    assert "--expand-train-per-bit 2000" in combined
    assert "--self-improve-warmup-steps 500" in combined
    assert "--pseudo-label-mode direct" in combined
    assert "--guarded-compose-rule run_length_unfiltered_pair" in combined
    assert "--guarded-compose-rule run_length_no_boundary_continue" in combined
    assert "--job-name rl-a10-direct" in combined
    assert "--output" in combined
    assert "--error" in combined
    assert "--wrap" in combined
    assert "DRY_RUN=1; sbatch not executed for direct." in stdout
    assert "DRY_RUN=1; sbatch not executed for unfiltered_compose." in stdout
    assert "DRY_RUN=1; sbatch not executed for guarded_compose." in stdout


def test_run_length_alpha10_baseline_pack_can_source_custom_config(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "rl_alpha10_custom")
    env["LOG_DIR"] = str(tmp_path / "logs")
    env["RUN_LENGTH_ALPHA10_BASELINE_CONFIG"] = str(tmp_path / "alpha10.env")
    Path(env["RUN_LENGTH_ALPHA10_BASELINE_CONFIG"]).write_text(
        "\n".join(
            [
                "RUN_LENGTH_ALPHA10_BASELINE_ROWS_RAW='direct:direct:none guarded_only:compose:run_length_no_boundary_continue'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(RUN_LENGTH_ALPHA10_BASELINE_PACK)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout.replace("\\ ", " ")
    combined = (result.stdout + result.stderr).replace("\\ ", " ")
    manifest_text = (tmp_path / "rl_alpha10_custom" / "manifest.tsv").read_text(encoding="utf-8")

    assert "Loaded run-length alpha10 baseline config" in stdout
    assert "Baseline rows: direct:direct:none guarded_only:compose:run_length_no_boundary_continue" in stdout
    assert "DRY_RUN=1; sbatch not executed for direct." in stdout
    assert "DRY_RUN=1; sbatch not executed for guarded_only." in stdout
    assert "DRY_RUN=1; sbatch not executed for unfiltered_compose." not in stdout
    assert "--output-dir " + str(tmp_path / "rl_alpha10_custom" / "guarded_only") in combined
    assert "--pseudo-label-mode direct" in combined
    assert "--pseudo-label-mode compose" in combined
    assert "--guarded-compose-rule run_length_no_boundary_continue" in combined
    assert "direct\tdryrun-direct\t" in manifest_text
    assert "guarded_only\tdryrun-guarded_only\t" in manifest_text
    assert "unfiltered_compose" not in manifest_text
