from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self.figure3_seed_quality_sweep import (
    build_seed_jobs,
    build_self_improvement_jobs,
    load_seed_candidates,
    missing_seed_bands,
    select_seed_bands,
)
from self.experiments import figure3_common
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
    assert "self.figure3_seed_quality_sweep submit" in stdout
    assert "Initial seed jobs: run_length=8 addition=8 total=16" in stdout
    assert stdout.count("--job-name=fig3-seed-run_length") == 8
    assert stdout.count("--job-name=fig3-seed-addition") == 8
    assert "fig3-seed-quality-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout
