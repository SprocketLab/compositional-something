from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self.figure3_real_seed_data_ablation import (
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
    assert "self.figure3_real_seed_data_ablation submit" in stdout
    assert "Initial seed jobs: run_length_medium=9 addition_refined=24 total=33" in stdout
    assert stdout.count("--job-name=fig3-real-seed-run_length") == 9
    assert stdout.count("--job-name=fig3-real-seed-addition") == 24
    assert "fig3-real-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout
