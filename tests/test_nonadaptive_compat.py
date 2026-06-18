from __future__ import annotations

from types import SimpleNamespace

from self.core.nonadaptive_compat import sync_nonadaptive_loop_globals


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
