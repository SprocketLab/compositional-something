from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF_COMMON = ROOT / "launchers" / "self" / "lib" / "self_common.sh"


def test_self_common_sources_colon_separated_configs_relative_to_root(tmp_path: Path):
    (tmp_path / "first.env").write_text("FIRST_VALUE=alpha\n", encoding="utf-8")
    (tmp_path / "second.env").write_text("SECOND_VALUE=beta\n", encoding="utf-8")

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"ROOT_DIR={tmp_path}\n"
        "self_source_config_files 'first.env:second.env' 'test config'\n"
        "printf 'values=%s/%s\\n' \"${FIRST_VALUE}\" \"${SECOND_VALUE}\"\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"[INFO] Loaded test config: {tmp_path / 'first.env'}" in result.stdout
    assert f"[INFO] Loaded test config: {tmp_path / 'second.env'}" in result.stdout
    assert "values=alpha/beta" in result.stdout


def test_self_common_reports_missing_config_with_label(tmp_path: Path):
    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"ROOT_DIR={tmp_path}\n"
        "self_source_config_file missing.env 'adaptive config'\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"[ERROR] Missing adaptive config file: {tmp_path / 'missing.env'}" in result.stderr


def test_self_common_wraps_repo_command_with_pythonpath_and_quotes(tmp_path: Path):
    root_with_space = tmp_path / "repo root"
    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"ROOT_DIR={shlex.quote(str(root_with_space))}\n"
        "self_wrap_repo_command python -m self.run_length_self_improvement --output-dir 'path with spaces'\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.startswith("cd ")
    assert "&& PYTHONPATH=. python -m self.run_length_self_improvement" in result.stdout
    assert "path\\ with\\ spaces" in result.stdout
