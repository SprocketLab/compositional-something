import subprocess
import sys

from self.adaptive.run.driver_wiring import AdaptiveRunDeps
from self.adaptive.run.run_orchestration import AdaptiveRunDeps as OrchestrationAdaptiveRunDeps


def test_adaptive_run_deps_shared_between_driver_and_orchestration() -> None:
    assert OrchestrationAdaptiveRunDeps is AdaptiveRunDeps


def test_driver_wiring_import_does_not_load_orchestration_stack() -> None:
    code = (
        "import sys\n"
        "import self.adaptive.run.driver_wiring\n"
        "print('self.adaptive.run.run_orchestration' in sys.modules)\n"
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
