from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self.experiments.figure3_seed_quality_sweep import (
    build_seed_jobs,
    build_self_improvement_jobs,
    load_seed_candidates,
    missing_seed_bands,
    select_seed_bands,
)
from self.experiments import figure3_cli
from self.experiments import figure3_common
from self.experiments import figure3_commands
from self.experiments import figure3_seed_quality_sweep


WRAPPER = ROOT / "launchers" / "self" / "submit_figure3_seed_quality_sweep_mig.sh"


def _write_seed_result(root: Path, *, validation: float, test: float) -> dict:
    root.mkdir(parents=True)
    (root / "model").mkdir()
    results_path = root / "seed_fit_results.json"
    results_path.write_text(
        json.dumps(
            {
                "validation_min_per_size_accuracy": validation,
                "test_min_per_size_accuracy": test,
                "results": {
                    "validation": {"min_per_size_accuracy": validation},
                    "test": {"min_per_size_accuracy": test},
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "kind": "seed",
        "task": root.parent.name,
        "train_count": int(root.name.replace("train", "")),
        "output_root": str(root),
        "results_path": str(results_path),
        "model_dir": str(root / "model"),
    }


def test_submit_figure3_seed_quality_wrapper_has_valid_bash_syntax():
    assert WRAPPER.exists()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_build_seed_jobs_emits_16_initial_jobs(tmp_path: Path):
    jobs = build_seed_jobs(out_root=tmp_path / "sweep", python_bin="python")
    assert len(jobs) == 16
    assert sum(1 for entry in jobs if entry["task"] == "run_length") == 8
    assert sum(1 for entry in jobs if entry["task"] == "addition") == 8
    assert {entry["train_count"] for entry in jobs if entry["task"] == "addition"} == {
        250,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
    }


def test_select_seed_bands_picks_low_medium_and_existing_high(tmp_path: Path):
    entries = []
    for task in ("run_length", "addition"):
        entries.append(_write_seed_result(tmp_path / task / "train250", validation=0.74, test=0.76))
        entries.append(_write_seed_result(tmp_path / task / "train500", validation=0.83, test=0.86))
        high = _write_seed_result(tmp_path / task / "train50000", validation=0.99, test=0.98)
        high["source"] = "existing_paper_seed"
        entries.append(high)

    candidates = load_seed_candidates(entries)
    selection = select_seed_bands(candidates)

    assert missing_seed_bands(selection) == {}
    assert selection["run_length"]["low"]["train_count"] == 250
    assert selection["run_length"]["medium"]["train_count"] == 500
    assert selection["run_length"]["high"]["source"] == "existing_paper_seed"
    assert selection["addition"]["low"]["train_count"] == 250
    assert selection["addition"]["medium"]["train_count"] == 500


def test_build_self_improvement_jobs_uses_exact_figure3_settings(tmp_path: Path):
    selection = {}
    for task in ("run_length", "addition"):
        selection[task] = {}
        for level, score in (("low", 0.75), ("medium", 0.85), ("high", 0.99)):
            model_dir = tmp_path / task / level / "model"
            model_dir.mkdir(parents=True)
            selection[task][level] = {
                "model_dir": str(model_dir),
                "worst_case_accuracy": score,
                "train_count": 1000,
                "validation_min_accuracy": score,
                "test_min_accuracy": score,
            }

    jobs = build_self_improvement_jobs(out_root=tmp_path / "sweep", selection=selection)

    assert len(jobs) == 14
    assert sum(1 for entry in jobs if entry["kind"] == "seed_quality_si") == 6
    assert sum(1 for entry in jobs if entry["kind"] == "sample_size_si") == 8
    assert {entry["sample_size"] for entry in jobs if entry["task"] == "run_length"} >= {500, 1000, 2000, 4000}
    assert {entry["sample_size"] for entry in jobs if entry["task"] == "addition"} >= {2500, 5000, 10000, 20000}


def test_seed_quality_private_helpers_reexport_common_owner():
    assert figure3_seed_quality_sweep._json_dump is figure3_common.json_dump
    assert figure3_seed_quality_sweep._submit_sbatch_job is figure3_common.submit_sbatch_job
    assert figure3_seed_quality_sweep._metric_from_seed_payload is figure3_common.metric_from_seed_payload
    assert figure3_seed_quality_sweep._final_row is figure3_common.final_row
    assert figure3_seed_quality_sweep._max_at_90 is figure3_common.max_at_90
    assert figure3_seed_quality_sweep._write_csv is figure3_common.write_csv
    assert figure3_seed_quality_sweep.load_seed_candidates is figure3_common.load_seed_candidates
    assert figure3_seed_quality_sweep.SEED_BANDS is figure3_common.DEFAULT_SEED_BANDS
    assert figure3_common.seed_fit_command is figure3_commands.seed_fit_command
    assert figure3_common.run_length_self_improvement_command is (
        figure3_commands.run_length_self_improvement_command
    )
    assert figure3_seed_quality_sweep._parse_common is figure3_cli.add_common_args
    assert figure3_seed_quality_sweep._default_manifest is figure3_cli.default_manifest
    assert figure3_seed_quality_sweep._default_selection is figure3_cli.default_selection
    assert figure3_seed_quality_sweep._default_summary is figure3_cli.default_summary


def test_seed_quality_command_builders_delegate_to_common_helpers(tmp_path: Path):
    seed_entry = {
        "task": "run_length",
        "train_count": 250,
        "output_root": str(tmp_path / "seed"),
    }
    seed_cmd = figure3_seed_quality_sweep._seed_job_command(seed_entry, python_bin="python")
    assert seed_cmd == figure3_common.seed_fit_command(
        seed_entry,
        python_bin="python",
        max_steps_position="task_specific",
    )
    assert seed_cmd.index("--max-steps") > seed_cmd.index("--initial-eval-per-size")

    si_entry = {
        "seed_model": str(tmp_path / "model"),
        "output_root": str(tmp_path / "si"),
        "sample_size": 2000,
    }
    si_cmd = figure3_seed_quality_sweep._run_length_si_command(si_entry, python_bin="python")
    assert si_cmd == figure3_common.run_length_self_improvement_command(
        si_entry,
        python_bin="python",
        num_expand_rounds=7,
    )
    assert si_cmd[si_cmd.index("--num-expand-rounds") + 1] == "7"


def test_submit_figure3_seed_quality_wrapper_dry_run_prints_expected_counts(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure3_seed_quality")
    env["LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "self.experiments.figure3_seed_quality_sweep submit" in stdout
    assert "Initial seed jobs: run_length=8 addition=8 total=16" in stdout
    assert stdout.count("--job-name=fig3-seed-run_length") == 8
    assert stdout.count("--job-name=fig3-seed-addition") == 8
    assert "fig3-seed-quality-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout
