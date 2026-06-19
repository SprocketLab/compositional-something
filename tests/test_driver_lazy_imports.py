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

        driver = importlib.import_module("self.adaptive.driver")
        wrapper = importlib.import_module("self.adaptive.driver")
        old_default_module = "self.adaptive." + "driver_default_bindings"
        old_wiring_module = "self.adaptive." + "driver_wiring"
        print(json.dumps({
            "candidate_loaded": "self.adaptive.candidate" in sys.modules,
            "old_driver_default_bindings_loaded": old_default_module in sys.modules,
            "old_driver_wiring_loaded": old_wiring_module in sys.modules,
            "run_orchestration_loaded": "self.adaptive.run" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "transformers_loaded": "transformers" in sys.modules,
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
        "candidate_loaded": False,
        "driver_has_build_parser": True,
        "old_driver_default_bindings_loaded": False,
        "old_driver_wiring_loaded": False,
        "run_orchestration_loaded": False,
        "torch_loaded": False,
        "transformers_loaded": False,
        "wrapper_exports_build_parser": True,
        "wrapper_exports_run": True,
    }


def test_adaptive_runtime_contract_modules_do_not_import_training_stack() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        modules = [
            "self.adaptive.proposal",
            "self.adaptive.proposal",
            "self.adaptive.run",
            "self.adaptive.run",
            "self.adaptive.run",
            "self.adaptive.attempts",
            "self.adaptive.attempts",
        ]
        for module_name in modules:
            importlib.import_module(module_name)
        print(json.dumps({
            "tasks_loaded": "self.tasks" in sys.modules,
            "training_loaded": "self.core.training" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "transformers_loaded": "transformers" in sys.modules,
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

    assert json.loads(result.stdout) == {
        "tasks_loaded": False,
        "torch_loaded": False,
        "training_loaded": False,
        "transformers_loaded": False,
    }
