from __future__ import annotations

from pathlib import Path

from self.nonadaptive.nonadaptive_finalization import finalize_nonadaptive_run


def test_finalize_nonadaptive_run_cleans_checkpoints_and_reports_results(tmp_path: Path):
    round_dirs = [tmp_path / "round_00", tmp_path / "round_01"]
    cleaned = []
    messages = []

    result = finalize_nonadaptive_run(
        keep_checkpoints=False,
        save_model_policy="final_only",
        round_dirs=round_dirs,
        results_path=tmp_path / "self_improvement_results.json",
        cleanup_round_checkpoints_fn=lambda dirs: cleaned.append(list(dirs)),
        print_fn=lambda message, **kwargs: messages.append((message, kwargs)),
    )

    assert result.checkpoints_cleaned is True
    assert cleaned == [round_dirs]
    assert messages == [
        (
            f"[INFO] Saved round summaries to {tmp_path / 'self_improvement_results.json'}",
            {"flush": True},
        )
    ]


def test_finalize_nonadaptive_run_keeps_checkpoints_when_requested(tmp_path: Path):
    cleaned = []

    result = finalize_nonadaptive_run(
        keep_checkpoints=True,
        save_model_policy="all_rounds",
        round_dirs=[tmp_path / "round_00"],
        results_path=tmp_path / "results.json",
        cleanup_round_checkpoints_fn=lambda dirs: cleaned.append(list(dirs)),
        print_fn=lambda *args, **kwargs: None,
    )

    assert result.checkpoints_cleaned is False
    assert cleaned == []


def test_finalize_nonadaptive_run_skips_cleanup_when_no_models_are_saved(tmp_path: Path):
    cleaned = []

    result = finalize_nonadaptive_run(
        keep_checkpoints=False,
        save_model_policy="none",
        round_dirs=[tmp_path / "round_00"],
        results_path=tmp_path / "results.json",
        cleanup_round_checkpoints_fn=lambda dirs: cleaned.append(list(dirs)),
        print_fn=lambda *args, **kwargs: None,
    )

    assert result.checkpoints_cleaned is False
    assert cleaned == []
