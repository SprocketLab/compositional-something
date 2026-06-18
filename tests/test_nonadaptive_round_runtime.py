from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.core import nonadaptive_round_models
from self.core.nonadaptive_round_runtime import (
    NonAdaptiveRoundRuntimeContext,
    NonAdaptiveRoundRuntimeResult,
    NonAdaptiveRoundRuntimeState,
    run_nonadaptive_round,
)


class _Schedule:
    def round_max_size_for_index(self, round_idx: int) -> int:
        return 10 + round_idx


class _Task:
    @staticmethod
    def serialize_example(example):
        return {"value": example}


def test_nonadaptive_round_runtime_reexports_round_models():
    assert NonAdaptiveRoundRuntimeContext is nonadaptive_round_models.NonAdaptiveRoundRuntimeContext
    assert NonAdaptiveRoundRuntimeState is nonadaptive_round_models.NonAdaptiveRoundRuntimeState
    assert NonAdaptiveRoundRuntimeResult is nonadaptive_round_models.NonAdaptiveRoundRuntimeResult


def _context(tmp_path: Path, **overrides):
    defaults = dict(
        args=SimpleNamespace(num_expand_rounds=1),
        task=_Task(),
        base_output_dir=tmp_path,
        base_splits={"train": ["base"]},
        base_records={},
        eval_examples=["eval"],
        composed_eval_slices={},
        composed_eval_component_map={},
        composed_pool_path=tmp_path / "composed.jsonl",
        component_map_path=tmp_path / "components.json",
        metadata={},
        eval_keys=set(),
        size_schedule=_Schedule(),
        composed_min_size=2,
        final_max_size=4,
        train_base_decode_tokens=8,
        eval_decode_tokens=8,
        composed_eval_decode_tokens=8,
        config=SimpleNamespace(per_device_eval_batch_size=2, decode_max_new_tokens=8),
        data_collator="collator",
        tokenizer="tokenizer",
        rng=SimpleNamespace(random=lambda: 0.25),
        new_run=True,
        dynamic_composed=False,
        save_model_policy="all_rounds",
        resume_requested=False,
        resume_round=0,
        stop_after_round=None,
        reset_each_round=False,
        use_recipe=False,
        recipe_name="none",
        recipe_preset=None,
        summary_records={},
        results_path=tmp_path / "results.json",
        persist_metadata_fn=lambda: None,
    )
    defaults.update(overrides)
    return NonAdaptiveRoundRuntimeContext(**defaults)


def test_run_nonadaptive_round_skips_completed_resume_round(tmp_path: Path, capsys):
    context = _context(tmp_path, resume_requested=True, resume_round=1)
    state = NonAdaptiveRoundRuntimeState(
        model="model",
        composed_examples=["composed"],
        component_map="component-map",
        pseudo_examples=["pseudo"],
    )

    result = run_nonadaptive_round(
        context=context,
        state=state,
        round_idx=0,
        cuda_is_available_fn=lambda: False,
        empty_cache_fn=lambda: None,
        train_round_model_fn=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should skip training")),
    )

    assert result.round_dir == tmp_path / "round_00"
    assert result.skipped is True
    assert result.should_break is False
    assert state.model == "model"
    assert "Skipping already completed round 0" in capsys.readouterr().out


def test_run_nonadaptive_round_updates_state_and_records_summary(tmp_path: Path):
    context = _context(tmp_path)
    state = NonAdaptiveRoundRuntimeState(
        model="initial-model",
        composed_examples=["old-composed"],
        component_map="old-map",
        pseudo_examples=["old-pseudo"],
    )
    events = []

    def save_examples_fn(path, examples, serializer):
        events.append(("save", path.name, list(examples), serializer("x")))

    def train_round_model_fn(**kwargs):
        events.append(("train", kwargs["train_examples"], kwargs["save_model_this_round"]))
        assert kwargs["model"] == "initial-model"
        assert kwargs["round_dir"] == tmp_path / "round_00"
        return SimpleNamespace(model="trained-model", trainer="trainer")

    def evaluate_round_fn(**kwargs):
        events.append(("evaluate", kwargs["model"], kwargs["eval_examples"]))
        return "evaluation"

    def prepare_next_pseudo_round_fn(**kwargs):
        events.append(("pseudo", kwargs["model"], kwargs["train_examples"], kwargs["composed_examples"]))
        return SimpleNamespace(
            composed_examples=["new-composed"],
            component_map="new-map",
            pseudo_examples=["new-pseudo"],
            pseudo_generation_stats={"retained_total": 1},
        )

    def record_round_summary_fn(**kwargs):
        events.append(
            (
                "summary",
                kwargs["max_size"],
                kwargs["train_example_count"],
                kwargs["pseudo_used_count"],
                kwargs["pseudo_generation_stats"],
            )
        )

    def finish_round_fn(**kwargs):
        events.append(("finish", kwargs["resources"].model, kwargs["resources"].trainer))
        kwargs["resources"].model = "post-round-model"
        kwargs["resources"].trainer = None
        return SimpleNamespace(should_break=True)

    result = run_nonadaptive_round(
        context=context,
        state=state,
        round_idx=0,
        save_examples_fn=save_examples_fn,
        train_round_model_fn=train_round_model_fn,
        evaluate_round_fn=evaluate_round_fn,
        prepare_next_pseudo_round_fn=prepare_next_pseudo_round_fn,
        record_round_summary_fn=record_round_summary_fn,
        finish_round_fn=finish_round_fn,
        cuda_is_available_fn=lambda: False,
        empty_cache_fn=lambda: None,
    )

    assert result.round_dir == tmp_path / "round_00"
    assert result.skipped is False
    assert result.should_break is True
    assert state.model == "post-round-model"
    assert state.composed_examples == ["new-composed"]
    assert state.component_map == "new-map"
    assert state.pseudo_examples == ["new-pseudo"]
    assert events == [
        ("save", "train_examples.jsonl", ["base", "old-pseudo"], {"value": "x"}),
        ("save", "pseudo_examples_used.jsonl", ["old-pseudo"], {"value": "x"}),
        ("train", ["base", "old-pseudo"], True),
        ("evaluate", "trained-model", ["eval"]),
        ("pseudo", "trained-model", ["base", "old-pseudo"], ["old-composed"]),
        ("summary", 10, 2, 1, {"retained_total": 1}),
        ("finish", "trained-model", "trainer"),
    ]
