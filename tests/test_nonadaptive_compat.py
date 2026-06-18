from __future__ import annotations

from types import SimpleNamespace

from self.core import nonadaptive_facade_exports
from self.core.nonadaptive_compat import NONADAPTIVE_PATCHABLE_NAMES, sync_nonadaptive_loop_globals
from self import self_improvement_core as core_facade


def test_sync_nonadaptive_loop_globals_copies_named_facade_values():
    target = SimpleNamespace(old_value="unchanged")
    source_globals = {
        "patched_model_loader": object(),
        "patched_evaluator": object(),
    }

    sync_nonadaptive_loop_globals(
        source_globals=source_globals,
        target_module=target,
        names=("patched_model_loader", "patched_evaluator"),
    )

    assert target.patched_model_loader is source_globals["patched_model_loader"]
    assert target.patched_evaluator is source_globals["patched_evaluator"]
    assert target.old_value == "unchanged"


def test_nonadaptive_facade_exports_are_available_and_patchable_names_are_public():
    assert core_facade.__all__ == list(nonadaptive_facade_exports.NONADAPTIVE_FACADE_EXPORT_NAMES)
    assert len(core_facade.__all__) == len(set(core_facade.__all__))
    assert all(hasattr(core_facade, name) for name in core_facade.__all__)
    assert set(NONADAPTIVE_PATCHABLE_NAMES).issubset(core_facade.__all__)
    assert "run_self_improvement" in core_facade.__all__
    assert core_facade.TrainingConfig.__module__ == "self.core.training"
    assert core_facade.SelfImprovementTask.__module__ == "self.core.task_protocols"
