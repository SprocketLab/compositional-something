from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.core.nonadaptive_round_loop import run_nonadaptive_round_loop


def test_run_nonadaptive_round_loop_forwards_deps_and_stops_early(tmp_path: Path):
    calls = []
    context = object()
    state = object()
    sentinel_dep = object()

    def run_round_fn(**kwargs):
        calls.append(kwargs)
        round_idx = kwargs["round_idx"]
        return SimpleNamespace(
            round_dir=tmp_path / f"round_{round_idx:02d}",
            should_break=(round_idx == 1),
        )

    result = run_nonadaptive_round_loop(
        context=context,
        state=state,
        num_rounds=4,
        run_round_fn=run_round_fn,
        round_runtime_kwargs={"sentinel_dep": sentinel_dep},
    )

    assert result.round_dirs == [tmp_path / "round_00", tmp_path / "round_01"]
    assert result.completed_rounds == 2
    assert result.stopped_early is True
    assert [call["round_idx"] for call in calls] == [0, 1]
    assert all(call["context"] is context for call in calls)
    assert all(call["state"] is state for call in calls)
    assert all(call["sentinel_dep"] is sentinel_dep for call in calls)


def test_run_nonadaptive_round_loop_handles_empty_range():
    result = run_nonadaptive_round_loop(
        context=object(),
        state=object(),
        num_rounds=0,
        run_round_fn=lambda **kwargs: None,
        round_runtime_kwargs={},
    )

    assert result.round_dirs == []
    assert result.completed_rounds == 0
    assert result.stopped_early is False
