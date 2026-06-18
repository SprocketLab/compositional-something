from __future__ import annotations

import os
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


def test_self_common_updates_symlink_and_creates_parent(tmp_path: Path):
    target = tmp_path / "model target"
    link = tmp_path / "artifacts" / "models" / "seed_best"
    target.mkdir()

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"self_update_symlink {shlex.quote(str(target))} {shlex.quote(str(link))}\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"[INFO] Updated {link} -> {target}" in result.stdout
    assert link.is_symlink()
    assert os.readlink(link) == str(target)


def test_self_common_symlink_dry_run_only_reports_update(tmp_path: Path):
    target = tmp_path / "model target"
    link = tmp_path / "artifacts" / "models" / "seed_best"
    target.mkdir()

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"self_update_symlink_or_dry_run {shlex.quote(str(target))} {shlex.quote(str(link))} 1\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"[INFO] DRY_RUN=1; would update {link} -> {target}" in result.stdout
    assert link.parent.is_dir()
    assert not link.exists()


def test_self_common_model_preflight_uses_configured_python_and_model_name(tmp_path: Path):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'fake_args=%s\\n' \"$*\"\n"
        "printf 'model=%s\\n' \"${MODEL_NAME:-}\"\n"
        "cat >/dev/null\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"PYTHON_BIN={shlex.quote(str(fake_python))}\n"
        "self_preflight_model_snapshot '/tmp/local model'\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "fake_args=-" in result.stdout
    assert "model=/tmp/local model" in result.stdout


def test_self_common_model_preflight_passes_optional_tokenizer_mode(tmp_path: Path):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'fake_args=%s\\n' \"$*\"\n"
        "printf 'model=%s\\n' \"${MODEL_NAME:-}\"\n"
        "printf 'tokenizer=%s\\n' \"${TOKENIZER_MODE:-}\"\n"
        "cat >/dev/null\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"PYTHON_BIN={shlex.quote(str(fake_python))}\n"
        "self_preflight_model_snapshot '/tmp/local model' fixed_char\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "fake_args=-" in result.stdout
    assert "model=/tmp/local model" in result.stdout
    assert "tokenizer=fixed_char" in result.stdout


def test_self_common_torch_info_uses_configured_python(tmp_path: Path):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'fake_args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    script = (
        "set -euo pipefail\n"
        f"source {SELF_COMMON}\n"
        f"PYTHON_BIN={shlex.quote(str(fake_python))}\n"
        "self_print_torch_cuda_info\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "fake_args=-c import torch;" in result.stdout
