from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.nonadaptive.nonadaptive_round_context import (
    build_nonadaptive_round_runtime_bundle,
    build_nonadaptive_round_runtime_kwargs,
)


def test_build_nonadaptive_round_runtime_bundle_maps_run_state(tmp_path: Path) -> None:
    paths = SimpleNamespace(
        base_output_dir=tmp_path,
        composed_pool_path=tmp_path / "composed.jsonl",
        component_map_path=tmp_path / "components.json",
        results_path=tmp_path / "results.json",
    )
    setup = SimpleNamespace(
        size_schedule="schedule",
        composed_min_size=8,
        final_max_size=16,
        dynamic_composed=True,
        save_model_policy="final",
        stop_after_round=2,
        reset_each_round=False,
        use_recipe=False,
        recipe_name="none",
        recipe_preset=None,
    )
    run_state = SimpleNamespace(
        paths=paths,
        new_run=True,
        resume_requested=False,
    )
    metadata_runtime = SimpleNamespace(
        rng="rng",
        persist_metadata="persist-metadata",
    )
    datasets = SimpleNamespace(
        base_splits={"train": ["base"]},
        base_records={"train": {"key"}},
        eval_examples=["eval"],
        composed_eval_component_map={"eval": ["component"]},
        metadata={"task": "dummy"},
        composed_examples=["composed"],
        component_map={"composed": ["left", "right"]},
    )
    dataset_context = SimpleNamespace(
        composed_eval_slices={"frontier": ["eval"]},
        eval_keys={"eval-key"},
    )
    bootstrap = SimpleNamespace(
        train_base_decode_tokens=8,
        eval_decode_tokens=9,
        composed_eval_decode_tokens=10,
        config="config",
        data_collator="collator",
        tokenizer="tokenizer",
        resume_round=1,
        summary_records={0: {"accuracy": 1.0}},
        model="model",
        pseudo_examples=["pseudo"],
    )

    bundle = build_nonadaptive_round_runtime_bundle(
        args="args",
        task="task",
        setup=setup,
        run_state=run_state,
        metadata_runtime=metadata_runtime,
        datasets=datasets,
        dataset_context=dataset_context,
        bootstrap=bootstrap,
    )

    assert bundle.context.args == "args"
    assert bundle.context.base_output_dir == tmp_path
    assert bundle.context.composed_pool_path == tmp_path / "composed.jsonl"
    assert bundle.context.base_splits == {"train": ["base"]}
    assert bundle.context.composed_eval_slices == {"frontier": ["eval"]}
    assert bundle.context.eval_keys == {"eval-key"}
    assert bundle.context.train_base_decode_tokens == 8
    assert bundle.context.resume_round == 1
    assert bundle.context.persist_metadata_fn == "persist-metadata"
    assert bundle.state.model == "model"
    assert bundle.state.composed_examples == ["composed"]
    assert bundle.state.component_map == {"composed": ["left", "right"]}
    assert bundle.state.pseudo_examples == ["pseudo"]


def test_build_nonadaptive_round_runtime_kwargs_preserves_dependency_identities() -> None:
    deps = {name: object() for name in _ROUND_RUNTIME_DEP_NAMES}

    kwargs = build_nonadaptive_round_runtime_kwargs(**deps)

    assert set(kwargs) == set(_ROUND_RUNTIME_DEP_NAMES)
    for name, value in deps.items():
        assert kwargs[name] is value


_ROUND_RUNTIME_DEP_NAMES = (
    "ensure_dir_fn",
    "save_examples_fn",
    "dataset_cls",
    "make_training_args_fn",
    "build_trainer_fn",
    "evaluate_accuracy_fn",
    "write_debug_samples_fn",
    "slice_metric_cls",
    "round_summary_cls",
    "summarize_round_fn",
    "summary_to_payload_fn",
    "write_summary_records_fn",
    "json_module",
    "resolve_max_new_tokens_fn",
    "random_cls",
    "path_cls",
    "cuda_is_available_fn",
    "empty_cache_fn",
    "instantiate_recipe_model_fn",
    "load_recipe_model_fn",
    "load_model_for_tokenizer_fn",
)
