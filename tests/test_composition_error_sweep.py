from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from self.experiments import composition_error_sweep as experiment
from self import self_improvement_composition_error_experiment as wrapper


ROOT = Path(__file__).resolve().parents[1]
RUN_SWEEP = ROOT / "launchers" / "self" / "run_composition_error_sweep_self_improvement.sh"
BOUNDARY_EVAL = ROOT / "launchers" / "self" / "run_self_improvement_mig_boundary_eval.sbatch"


def test_composition_error_sweep_builds_forward_args_with_default_strategy():
    args = experiment.parse_args(
        [
            "--composition-error-percent",
            "25",
            "--",
            "--output-dir",
            "artifacts/tmp",
        ]
    )

    assert experiment.build_forward_args(args) == [
        "--output-dir",
        "artifacts/tmp",
        "--composition-error-percent=25.0",
        "--composed-strategy=with_carry_filtered",
    ]


def test_composition_error_sweep_preserves_explicit_strategy():
    args = experiment.parse_args(
        [
            "--composition-error-percent",
            "0",
            "--",
            "--composed-strategy",
            "with_carry",
        ]
    )

    assert experiment.build_forward_args(args) == [
        "--composed-strategy",
        "with_carry",
        "--composition-error-percent=0.0",
    ]


def test_composition_error_sweep_rejects_out_of_range_percent():
    args = experiment.parse_args(["--composition-error-percent", "101"])

    with pytest.raises(ValueError, match="between 0 and 100"):
        experiment.build_forward_args(args)


def test_old_composition_error_module_forwards_attribute_patches(monkeypatch):
    captured = {}

    def fake_self_improvement_main(argv):
        captured["argv"] = argv

    monkeypatch.setattr(wrapper, "self_improvement_main", fake_self_improvement_main)

    wrapper.main(["--composition-error-percent", "10", "--", "--output-dir", "out"])

    assert captured["argv"] == [
        "--output-dir",
        "out",
        "--composition-error-percent=10.0",
        "--composed-strategy=with_carry_filtered",
    ]
    assert experiment.self_improvement_main is fake_self_improvement_main


def test_composition_error_launchers_use_canonical_module_and_valid_bash_syntax(tmp_path: Path):
    subprocess.run(["bash", "-n", str(RUN_SWEEP)], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", str(BOUNDARY_EVAL)], cwd=ROOT, check=True)

    result = subprocess.run(
        ["bash", str(RUN_SWEEP)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python",
        },
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "python -m self.experiments.composition_error_sweep" in result.stdout
    assert "self.self_improvement_composition_error_experiment" not in result.stdout
