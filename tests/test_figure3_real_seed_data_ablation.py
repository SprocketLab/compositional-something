from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self.experiments.figure3_real_seed_data_ablation import (
    ADDITION_MAX_STEPS,
    ADDITION_TRAIN_COUNTS,
    RUN_LENGTH_MEDIUM_TRAIN_COUNTS,
    RUN_LENGTH_SAMPLE_SIZES,
    build_run_length_si_jobs,
    build_seed_jobs,
    load_seed_candidates,
    missing_bands,
    select_seed_bands,
)
from self.experiments import figure3_cli
from self.experiments import figure3_common
from self.experiments import figure3_real_seed_data_ablation


WRAPPER = ROOT / "launchers" / "self" / "submit_figure3_real_seed_data_ablation_mig.sh"


def _write_seed_result(root: Path, *, task: str, validation: float, test: float, source: str | None = None) -> dict:
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
    entry = {
        "kind": "seed",
        "task": task,
        "train_count": int(root.name.replace("train", "").split("_")[0]),
        "max_steps": 10000,
        "output_root": str(root),
        "results_path": str(results_path),
        "model_dir": str(root / "model"),
    }
    if source:
        entry["source"] = source
    return entry


def test_real_ablation_wrapper_has_valid_bash_syntax():
    assert WRAPPER.exists()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_build_seed_jobs_emits_expected_refined_grids(tmp_path: Path):
    jobs = build_seed_jobs(out_root=tmp_path / "ablation")
    assert len(jobs) == 33
    assert sum(1 for job in jobs if job["task"] == "run_length") == 9
    assert sum(1 for job in jobs if job["task"] == "addition") == 24
    assert {job["train_count"] for job in jobs if job["task"] == "run_length"} == set(
        RUN_LENGTH_MEDIUM_TRAIN_COUNTS
    )
    assert {job["train_count"] for job in jobs if job["task"] == "addition"} == set(ADDITION_TRAIN_COUNTS)
    assert {job["max_steps"] for job in jobs if job["task"] == "addition"} == set(ADDITION_MAX_STEPS)


def test_seed_band_selection_allows_run_length_matrix_without_addition_medium(tmp_path: Path):
    entries = [
        _write_seed_result(tmp_path / "run_length" / "train1000", task="run_length", validation=0.74, test=0.76),
        _write_seed_result(tmp_path / "run_length" / "train1200", task="run_length", validation=0.84, test=0.86),
        _write_seed_result(
            tmp_path / "run_length" / "train50000",
            task="run_length",
            validation=0.99,
            test=0.98,
            source="existing_high_seed",
        ),
        _write_seed_result(
            tmp_path / "addition" / "train50000",
            task="addition",
            validation=0.99,
            test=0.98,
            source="existing_high_seed",
        ),
    ]

    selected = select_seed_bands(load_seed_candidates(entries))
    assert missing_bands(selected, "run_length") == []
    assert missing_bands(selected, "addition") == ["low", "medium"]
    jobs = build_run_length_si_jobs(out_root=tmp_path / "ablation", selection=selected)
    assert len(jobs) == 9
    assert {job["sample_size"] for job in jobs} == set(RUN_LENGTH_SAMPLE_SIZES.values())


def test_addition_seed_selection_rejects_crossed_low_medium_data_counts(tmp_path: Path):
    entries = [
        _write_seed_result(tmp_path / "addition" / "train1000", task="addition", validation=0.73, test=0.74),
        _write_seed_result(tmp_path / "addition" / "train850", task="addition", validation=0.84, test=0.86),
        _write_seed_result(
            tmp_path / "addition" / "train50000",
            task="addition",
            validation=0.99,
            test=0.98,
            source="existing_high_seed",
        ),
    ]

    selected = select_seed_bands(load_seed_candidates(entries))

    assert missing_bands(selected, "addition") == ["low", "medium"]
    assert selected["addition"]["high"]["source"] == "existing_high_seed"


def test_addition_seed_selection_prefers_monotone_low_medium_pair(tmp_path: Path):
    entries = [
        _write_seed_result(tmp_path / "addition" / "train800", task="addition", validation=0.73, test=0.76),
        _write_seed_result(tmp_path / "addition" / "train850", task="addition", validation=0.84, test=0.86),
        _write_seed_result(tmp_path / "addition" / "train1000", task="addition", validation=0.72, test=0.75),
        _write_seed_result(
            tmp_path / "addition" / "train50000",
            task="addition",
            validation=0.99,
            test=0.98,
            source="existing_high_seed",
        ),
    ]

    selected = select_seed_bands(load_seed_candidates(entries))

    assert selected["addition"]["low"]["train_count"] == 800
    assert selected["addition"]["medium"]["train_count"] == 850
    assert selected["addition"]["low"]["train_count"] < selected["addition"]["medium"]["train_count"]


def test_real_ablation_private_csv_helper_reexports_common_owner():
    assert figure3_real_seed_data_ablation._write_csv is figure3_common.write_csv
    assert figure3_real_seed_data_ablation.load_seed_candidates is figure3_common.load_seed_candidates
    assert figure3_real_seed_data_ablation.SEED_BANDS is figure3_common.DEFAULT_SEED_BANDS
    assert figure3_real_seed_data_ablation._parse_common is figure3_cli.add_common_args
    assert figure3_real_seed_data_ablation._default_manifest is figure3_cli.default_manifest
    assert figure3_real_seed_data_ablation._default_selection is figure3_cli.default_selection
    assert figure3_real_seed_data_ablation._default_summary is figure3_cli.default_summary


def test_real_ablation_command_builders_delegate_to_common_helpers(tmp_path: Path):
    seed_entry = {
        "task": "addition",
        "train_count": 900,
        "max_steps": 2500,
        "output_root": str(tmp_path / "seed"),
    }
    seed_cmd = figure3_real_seed_data_ablation._seed_job_command(seed_entry, python_bin="python")
    assert seed_cmd == figure3_common.seed_fit_command(
        seed_entry,
        python_bin="python",
        max_steps_position="before_batch_args",
    )
    assert seed_cmd.index("--max-steps") < seed_cmd.index("--per-device-train-batch-size")

    si_entry = {
        "seed_model": str(tmp_path / "model"),
        "output_root": str(tmp_path / "si"),
        "sample_size": 1000,
    }
    si_cmd = figure3_real_seed_data_ablation._run_length_si_command(si_entry, python_bin="python")
    assert si_cmd == figure3_common.run_length_self_improvement_command(
        si_entry,
        python_bin="python",
        num_expand_rounds=8,
    )
    assert si_cmd[si_cmd.index("--num-expand-rounds") + 1] == "8"


def test_submit_wrapper_dry_run_prints_expected_counts(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure3_real_ablation")
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
    assert "self.experiments.figure3_real_seed_data_ablation submit" in stdout
    assert "Initial seed jobs: run_length_medium=9 addition_refined=24 total=33" in stdout
    assert stdout.count("--job-name=fig3-real-seed-run_length") == 9
    assert stdout.count("--job-name=fig3-real-seed-addition") == 24
    assert "fig3-real-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout
