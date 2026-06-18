from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.nonadaptive.nonadaptive_results import record_nonadaptive_round_summary
from self.core.summaries import SliceMetric


class _Task:
    size_label = "size"
    size_alias_singular = "unit"

    @staticmethod
    def summary_payload_aliases(summary):
        return {"task_alias": f"round-{summary.index}"}


def _evaluation(**overrides):
    values = dict(
        eval_accuracy=0.75,
        per_size_accuracy={4: 1.0, 8: 0.5},
        composed_eval_accuracy=0.25,
        composed_slice_metrics={"all": SliceMetric(accuracy=0.25, count=2, per_size_accuracy={9: 0.25})},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_record_nonadaptive_round_summary_writes_metrics_and_results(tmp_path: Path):
    round_dir = tmp_path / "round_00"
    round_dir.mkdir()
    summary_records = {}

    result = record_nonadaptive_round_summary(
        round_idx=0,
        max_size=8,
        train_example_count=5,
        pseudo_used_count=2,
        evaluation=_evaluation(),
        pseudo_generation_stats={"candidate_total": 4, "retained_total": 1, "missing_total": 3},
        round_dir=round_dir,
        save_model_policy="final_only",
        save_model_this_round=False,
        summary_records=summary_records,
        results_path=tmp_path / "results.json",
        task=_Task(),
    )

    metrics = json.loads((round_dir / "metrics.json").read_text(encoding="utf-8"))
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert metrics["round"] == 0
    assert metrics["max_size"] == 8
    assert metrics["train_examples"] == 5
    assert metrics["pseudo_examples"] == 2
    assert metrics["eval_accuracy"] == 0.75
    assert metrics["per_size_accuracy"] == {"4": 1.0, "8": 0.5}
    assert metrics["composed_eval_accuracy"] == 0.25
    assert metrics["pseudo_retention_rate"] == 0.25
    assert metrics["save_model_policy"] == "final_only"
    assert metrics["model_dir"] is None
    assert metrics["task_alias"] == "round-0"
    assert results == [metrics]
    assert summary_records == {0: metrics}
    assert result.metrics_payload == metrics
    assert result.summary.index == 0


def test_record_nonadaptive_round_summary_uses_injected_bindings(tmp_path: Path):
    round_dir = tmp_path / "round_03"
    round_dir.mkdir()
    calls = {}
    summary_records = {}

    def round_summary_cls(**kwargs):
        calls["summary_kwargs"] = kwargs
        return {"summary": kwargs["index"]}

    def summarize_round_fn(summary, task):
        calls["summarize"] = (summary, task)

    def summary_to_payload_fn(summary, task):
        calls["payload"] = (summary, task)
        return {"round": summary["summary"]}

    def write_summary_records_fn(records, path):
        calls["write_records"] = (dict(records), path)

    result = record_nonadaptive_round_summary(
        round_idx=3,
        max_size=14,
        train_example_count=7,
        pseudo_used_count=1,
        evaluation=_evaluation(eval_accuracy=0.1),
        pseudo_generation_stats={"candidate_total": 0},
        round_dir=round_dir,
        save_model_policy="all_rounds",
        save_model_this_round=True,
        summary_records=summary_records,
        results_path=tmp_path / "results.json",
        task="task",
        round_summary_cls=round_summary_cls,
        summarize_round_fn=summarize_round_fn,
        summary_to_payload_fn=summary_to_payload_fn,
        write_summary_records_fn=write_summary_records_fn,
    )

    assert calls["summary_kwargs"]["index"] == 3
    assert calls["summary_kwargs"]["eval_accuracy"] == 0.1
    assert calls["summary_kwargs"]["output_dir"] == round_dir
    assert calls["summarize"] == ({"summary": 3}, "task")
    assert calls["payload"] == ({"summary": 3}, "task")
    expected_payload = {"round": 3, "save_model_policy": "all_rounds", "model_dir": str(round_dir)}
    assert json.loads((round_dir / "metrics.json").read_text(encoding="utf-8")) == expected_payload
    assert summary_records == {3: expected_payload}
    assert calls["write_records"] == ({3: expected_payload}, tmp_path / "results.json")
    assert result.summary == {"summary": 3}
    assert result.metrics_payload == expected_payload
