from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from self.self_improvement_core import cleanup_round_checkpoints, run_self_improvement


@dataclass(frozen=True)
class _DummyExample:
    bits: int
    value: str

    def prompt(self) -> str:
        return f"Q: dummy({self.bits}) = ?\nA:"

    def target(self) -> str:
        return self.value


class _DummyModel:
    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy-model.txt").write_text("ok", encoding="utf-8")


class _DummyTokenizer:
    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy-tokenizer.txt").write_text("ok", encoding="utf-8")


class _DummyTrainer:
    def __init__(self, model: _DummyModel, output_dir: str | Path):
        self.model = model
        self.output_dir = Path(output_dir)

    def train(self) -> None:
        return None

    def save_model(self, output_dir: str | Path | None = None) -> None:
        if output_dir is None:
            output_dir = self.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.safetensors").write_text("weights", encoding="utf-8")


class _DummyTask:
    name = "dummy"
    size_label = "bits"
    size_alias_singular = "bit"
    size_alias_plural = "bits"
    prediction_parser = staticmethod(lambda text, example=None: text)

    def validate_args(self, args) -> None:
        return None

    def serialize_example(self, example: _DummyExample) -> dict[str, object]:
        return {"bits": example.bits, "value": example.value}

    def deserialize_example(self, payload: dict[str, object]) -> _DummyExample:
        return _DummyExample(bits=int(payload["bits"]), value=str(payload["value"]))

    def save_component_map(self, path: Path, component_map) -> None:
        path.write_text("{}", encoding="utf-8")

    def load_component_map(self, path: Path):
        return {}

    def prepare_initial_splits(self, rng, args):
        train = [_DummyExample(bits=4, value="0")]
        val = [_DummyExample(bits=4, value="0")]
        test = [_DummyExample(bits=4, value="0")]
        records = {
            "train": {("train", 4, 0)},
            "validation": {("validation", 4, 0)},
            "test": {("test", 4, 0)},
        }
        return {"train": train, "validation": val, "test": test}, records

    def prepare_composed_train(self, rng, args, base_splits, base_records, min_size, max_size, additional_exclude=None):
        return [], {}, set()

    def prepare_composed_eval(self, rng, args, base_splits, base_records, min_size, max_size, additional_exclude=None):
        return [], {}, set()

    def prepare_eval_examples(self, rng, args, min_size, max_size, exclude):
        return [_DummyExample(bits=4, value="0")]

    def split_composed_eval_slices(self, examples, component_map):
        return {}

    def keys_for_examples(self, examples):
        return {(example.bits, example.value) for example in examples}

    def rebuild_records(self, base_splits):
        return {
            split: {(split, example.bits, idx) for idx, example in enumerate(examples)}
            for split, examples in base_splits.items()
        }

    def build_task_metadata(self, args, final_max_size):
        return {}

    def metadata_aliases(self, args, final_max_size):
        return {}

    def validate_loaded_metadata(self, args, metadata, final_max_size, dynamic_composed):
        return None

    def size_of(self, example: _DummyExample) -> int:
        return example.bits

    def summarize(self, summary) -> list[str]:
        return [f"[ROUND {summary.index}] dummy"]

    def summary_payload_aliases(self, summary) -> dict[str, object]:
        return {}


def test_treat_seed_as_round_zero_skips_round0_training(monkeypatch, tmp_path: Path):
    import self.self_improvement_core as core

    trainer_called = {"value": False}

    def fake_instantiate_model_and_tokenizer(*args, **kwargs):
        return _DummyModel(), _DummyTokenizer()

    def fake_build_trainer(*args, **kwargs):
        trainer_called["value"] = True
        raise AssertionError("build_trainer should not be called when round_00 is treated as a completed seed.")

    def fake_evaluate_accuracy_with_breakdown(*, examples, size_getter, **kwargs):
        sizes = {int(size_getter(example)): 1.0 for example in examples}
        return 1.0, sizes

    monkeypatch.setattr(core, "instantiate_model_and_tokenizer", fake_instantiate_model_and_tokenizer)
    monkeypatch.setattr(core, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(core, "evaluate_accuracy_with_breakdown", fake_evaluate_accuracy_with_breakdown)

    args = SimpleNamespace(
        model_name="dummy-seed",
        output_dir=str(tmp_path / "guarded_seed_round_zero"),
        bf16=False,
        fp16=False,
        initial_min_size=4,
        initial_max_size=8,
        initial_train_per_size=10,
        initial_eval_per_size=2,
        expand_num_size=4,
        expand_train_per_size=5,
        eval_per_size=2,
        composed_eval_per_size=0,
        num_expand_rounds=0,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=10,
        eval_steps=0,
        max_steps=-1,
        num_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        decode_max_new_tokens=8,
        seed=7,
        reset_in_each_round=False,
        resume=False,
        resume_from_round=None,
        tokenizer_mode="auto",
        recipe="none",
        bucket_train_batches_by_size=False,
        bucket_train_batches_by_bits=False,
        skip_save_model=True,
        keep_checkpoints=False,
        composed_refresh_mode="dynamic",
        init_from_scratch=False,
        reserve_shared_eval_first=False,
        treat_seed_as_round_zero=True,
        pseudo_label_mode="none",
    )

    run_self_improvement(args, _DummyTask())

    assert trainer_called["value"] is False
    metrics = json.loads((tmp_path / "guarded_seed_round_zero" / "round_00" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["round"] == 0
    assert metrics["pseudo_examples"] == 0


def test_final_only_save_policy_keeps_only_final_round_model(monkeypatch, tmp_path: Path):
    import self.self_improvement_core as core

    dummy_model = _DummyModel()

    def fake_instantiate_model_and_tokenizer(*args, **kwargs):
        return dummy_model, _DummyTokenizer()

    def fake_make_training_args(*args, **kwargs):
        return SimpleNamespace(per_device_train_batch_size=4, output_dir=str(args[0]))

    def fake_build_trainer(*args, **kwargs):
        return _DummyTrainer(kwargs["model"], kwargs["training_args"].output_dir)

    def fake_evaluate_accuracy_with_breakdown(*, examples, size_getter, **kwargs):
        sizes = {int(size_getter(example)): 1.0 for example in examples}
        return 1.0, sizes

    monkeypatch.setattr(core, "instantiate_model_and_tokenizer", fake_instantiate_model_and_tokenizer)
    monkeypatch.setattr(core, "make_training_args", fake_make_training_args)
    monkeypatch.setattr(core, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(core, "evaluate_accuracy_with_breakdown", fake_evaluate_accuracy_with_breakdown)

    out_dir = tmp_path / "final_only"
    args = SimpleNamespace(
        model_name="dummy-seed",
        output_dir=str(out_dir),
        bf16=False,
        fp16=False,
        initial_min_size=4,
        initial_max_size=8,
        initial_train_per_size=10,
        initial_eval_per_size=2,
        expand_num_size=4,
        expand_train_per_size=0,
        eval_per_size=2,
        composed_eval_per_size=0,
        num_expand_rounds=1,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=10,
        eval_steps=0,
        max_steps=-1,
        num_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        decode_max_new_tokens=8,
        seed=7,
        reset_in_each_round=False,
        resume=False,
        resume_from_round=None,
        tokenizer_mode="auto",
        recipe="none",
        bucket_train_batches_by_size=False,
        bucket_train_batches_by_bits=False,
        skip_save_model=False,
        save_model_policy="final_only",
        keep_checkpoints=False,
        composed_refresh_mode="dynamic",
        init_from_scratch=False,
        reserve_shared_eval_first=False,
        treat_seed_as_round_zero=False,
        pseudo_label_mode="none",
    )

    run_self_improvement(args, _DummyTask())

    assert not (out_dir / "round_00" / "model.safetensors").exists()
    assert not (out_dir / "round_00" / "dummy-tokenizer.txt").exists()
    assert (out_dir / "round_01" / "model.safetensors").exists()
    assert (out_dir / "round_01" / "dummy-tokenizer.txt").exists()

    round0_metrics = json.loads((out_dir / "round_00" / "metrics.json").read_text(encoding="utf-8"))
    round1_metrics = json.loads((out_dir / "round_01" / "metrics.json").read_text(encoding="utf-8"))
    assert round0_metrics["save_model_policy"] == "final_only"
    assert round0_metrics["model_dir"] is None
    assert round1_metrics["model_dir"] == str(out_dir / "round_01")


def test_all_rounds_save_policy_keeps_every_round_model(monkeypatch, tmp_path: Path):
    import self.self_improvement_core as core

    dummy_model = _DummyModel()

    def fake_instantiate_model_and_tokenizer(*args, **kwargs):
        return dummy_model, _DummyTokenizer()

    def fake_make_training_args(*args, **kwargs):
        return SimpleNamespace(per_device_train_batch_size=4, output_dir=str(args[0]))

    def fake_build_trainer(*args, **kwargs):
        return _DummyTrainer(kwargs["model"], kwargs["training_args"].output_dir)

    def fake_evaluate_accuracy_with_breakdown(*, examples, size_getter, **kwargs):
        sizes = {int(size_getter(example)): 1.0 for example in examples}
        return 1.0, sizes

    monkeypatch.setattr(core, "instantiate_model_and_tokenizer", fake_instantiate_model_and_tokenizer)
    monkeypatch.setattr(core, "make_training_args", fake_make_training_args)
    monkeypatch.setattr(core, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(core, "evaluate_accuracy_with_breakdown", fake_evaluate_accuracy_with_breakdown)

    out_dir = tmp_path / "all_rounds"
    args = SimpleNamespace(
        model_name="dummy-seed",
        output_dir=str(out_dir),
        bf16=False,
        fp16=False,
        initial_min_size=4,
        initial_max_size=8,
        initial_train_per_size=10,
        initial_eval_per_size=2,
        expand_num_size=4,
        expand_train_per_size=0,
        eval_per_size=2,
        composed_eval_per_size=0,
        num_expand_rounds=1,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=10,
        eval_steps=0,
        max_steps=-1,
        num_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        decode_max_new_tokens=8,
        seed=7,
        reset_in_each_round=False,
        resume=False,
        resume_from_round=None,
        tokenizer_mode="auto",
        recipe="none",
        bucket_train_batches_by_size=False,
        bucket_train_batches_by_bits=False,
        skip_save_model=False,
        save_model_policy="all_rounds",
        keep_checkpoints=False,
        composed_refresh_mode="dynamic",
        init_from_scratch=False,
        reserve_shared_eval_first=False,
        treat_seed_as_round_zero=False,
        pseudo_label_mode="none",
    )

    run_self_improvement(args, _DummyTask())

    assert (out_dir / "round_00" / "model.safetensors").exists()
    assert (out_dir / "round_00" / "dummy-tokenizer.txt").exists()
    assert (out_dir / "round_01" / "model.safetensors").exists()
    assert (out_dir / "round_01" / "dummy-tokenizer.txt").exists()


def test_cleanup_round_checkpoints_does_not_delete_final_model(tmp_path: Path):
    round_dir = tmp_path / "round_04"
    checkpoint_dir = round_dir / "checkpoint-123"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}", encoding="utf-8")
    (round_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    (round_dir / "dummy-tokenizer.txt").write_text("tokenizer", encoding="utf-8")

    cleanup_round_checkpoints([round_dir])

    assert not checkpoint_dir.exists()
    assert (round_dir / "model.safetensors").exists()
    assert (round_dir / "dummy-tokenizer.txt").exists()
