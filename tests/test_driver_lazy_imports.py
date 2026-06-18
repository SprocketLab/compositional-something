from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_driver_facade_does_not_import_heavy_bindings_for_public_api_listing() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        driver = importlib.import_module("self.core.driver")
        wrapper = importlib.import_module("self.adaptive_candidate_training")
        print(json.dumps({
            "driver_default_bindings_loaded": "self.core.driver_default_bindings" in sys.modules,
            "driver_wiring_loaded": "self.core.driver_wiring" in sys.modules,
            "run_orchestration_loaded": "self.core.run_orchestration" in sys.modules,
            "driver_has_build_parser": "build_parser" in dir(driver),
            "wrapper_exports_build_parser": "build_parser" in wrapper.__all__,
            "wrapper_exports_run": "run" in wrapper.__all__,
        }, sort_keys=True))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "driver_default_bindings_loaded": False,
        "driver_has_build_parser": True,
        "driver_wiring_loaded": False,
        "run_orchestration_loaded": False,
        "wrapper_exports_build_parser": True,
        "wrapper_exports_run": True,
    }
