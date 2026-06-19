import subprocess
import sys

from self.adaptive.driver import AdaptiveRunDeps


def test_adaptive_run_deps_owned_by_driver_facade() -> None:
    assert AdaptiveRunDeps.__module__ == "self.adaptive.driver"


def test_driver_import_does_not_load_orchestration_stack() -> None:
    code = (
        "import sys\n"
        "import self.adaptive.driver\n"
        "print('self.adaptive.run' in sys.modules)\n"
        "print('torch' in sys.modules)\n"
        "print('transformers' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False", "False"]
