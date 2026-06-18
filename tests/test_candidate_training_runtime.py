from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from self.core import candidate_training_runtime as runtime


class _Task:
    @staticmethod
    def size_of(example):
        return getattr(example, "size", 1)

    @staticmethod
    def prediction_parser(text):
        return text


class _SizedExample:
    def __init__(self, size: int):
        self.size = size

    def size_for_batching(self) -> int:
        return self.size + 10


def _args(**overrides):
    values = dict(
        num_epochs=1.5,
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        logging_steps=3,
        max_steps=-1,
        eval_steps=0,
        decode_max_new_tokens=12,
        bf16=False,
        fp16=False,
        init_from_scratch=True,
        model_name="base",
        tokenizer_mode="auto",
        recipe="none",
        bucket_train_batches_by_size=True,
        post_task_proposal_rehearsal_repeat_count=3,
        post_task_proposal_rehearsal_max_examples=4,
        keep_all_candidate_models=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_make_config_maps_nonpositive_step_limits_to_none():
    config = runtime.make_config(_args(max_steps=0, eval_steps=-1))

    assert config.num_epochs == 1.5
    assert config.learning_rate == 5e-6
    assert config.max_steps is None
    assert config.eval_steps is None
    assert config.decode_max_new_tokens == 12


def test_train_checkpoint_wires_trainer_and_saves_model(monkeypatch, tmp_path: Path):
    calls = {}

    class FakeTokenizer:
        def save_pretrained(self, path):
            calls["tokenizer_path"] = Path(path)

    class FakeDataset:
        def __init__(self, examples, tokenizer):
            calls["dataset_examples"] = list(examples)
            calls["dataset_tokenizer"] = tokenizer

    class FakeTrainer:
        def __init__(self, model):
            self.model = model

        def train(self):
            calls["trained"] = True

        def save_model(self, path):
            calls["model_path"] = Path(path)

    def fake_instantiate(source_checkpoint, **kwargs):
        calls["source_checkpoint"] = source_checkpoint
        calls["instantiate_kwargs"] = kwargs
        return "model", FakeTokenizer()

    def fake_make_training_args(output_dir, config, **kwargs):
        calls["training_args"] = (Path(output_dir), config, kwargs)
        return {"output_dir": Path(output_dir)}

    def fake_build_trainer(**kwargs):
        calls["trainer_kwargs"] = kwargs
        calls["size_for_batch"] = kwargs["size_getter"](_SizedExample(7))
        return FakeTrainer(kwargs["model"])

    monkeypatch.setattr(runtime, "instantiate_model_and_tokenizer", fake_instantiate)
    monkeypatch.setattr(runtime, "recipe_enabled", lambda recipe: False)
    monkeypatch.setattr(runtime, "CausalLMDataCollator", lambda tokenizer: ("collator", tokenizer))
    monkeypatch.setattr(runtime, "TokenizedPromptTargetDataset", FakeDataset)
    monkeypatch.setattr(runtime, "make_training_args", fake_make_training_args)
    monkeypatch.setattr(runtime, "build_trainer", fake_build_trainer)

    examples = [_SizedExample(3)]
    config = runtime.make_config(_args())
    model, tokenizer, model_dir = runtime.train_checkpoint(
        source_checkpoint="base",
        train_examples=examples,
        output_dir=tmp_path / "training",
        task=_Task(),
        args=_args(),
        config=config,
        seed=17,
        recipe_phase_name="self_improve",
        model_bootstrap_cache="cache",
    )

    assert model == "model"
    assert isinstance(tokenizer, FakeTokenizer)
    assert model_dir == tmp_path / "training" / "model"
    assert calls["source_checkpoint"] == "base"
    assert calls["instantiate_kwargs"]["init_from_scratch"] is True
    assert calls["instantiate_kwargs"]["bootstrap_cache"] == "cache"
    assert calls["training_args"][2]["skip_save"] is True
    assert calls["training_args"][2]["keep_checkpoints"] is False
    assert calls["trainer_kwargs"]["bucket_train_batches_by_size"] is True
    assert calls["size_for_batch"] == 17
    assert calls["trained"] is True
    assert calls["model_path"] == model_dir
    assert calls["tokenizer_path"] == model_dir


def test_evaluate_model_resolves_decode_budget(monkeypatch):
    calls = {}

    def fake_resolve(examples, decode_max_new_tokens):
        calls["resolve"] = (list(examples), decode_max_new_tokens)
        return 9

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        assert kwargs["size_getter"](_SizedExample(4)) == 4
        assert kwargs["prediction_parser"]("answer") == "answer"
        return 0.75, {4: 0.5}

    monkeypatch.setattr(runtime, "resolve_max_new_tokens", fake_resolve)
    monkeypatch.setattr(runtime, "evaluate_accuracy_with_breakdown", fake_evaluate)

    accuracy, per_size = runtime.evaluate_model(
        model="model",
        tokenizer="tokenizer",
        task=_Task(),
        examples=["e0", "e1"],
        batch_size=8,
        decode_max_new_tokens=12,
    )

    assert calls["resolve"] == (["e0", "e1"], 12)
    assert calls["evaluate"]["max_new_tokens"] == 9
    assert calls["evaluate"]["batch_size"] == 8
    assert accuracy == 0.75
    assert per_size == {4: 0.5}


def test_train_post_task_proposal_rehearsal_writes_summary_and_cleans_task_model(tmp_path: Path):
    calls = {}
    task_model_dir = tmp_path / "candidate" / "training" / "model"
    task_model_dir.mkdir(parents=True)
    candidate_dir = tmp_path / "candidate"

    def fake_train_checkpoint(**kwargs):
        calls["train"] = kwargs
        return "rehearsal-model", "rehearsal-tokenizer", candidate_dir / "proposal_rehearsal" / "model"

    def fake_write_json(path, payload):
        calls["summary"] = (Path(path), payload)

    model, tokenizer, model_dir = runtime.train_post_task_proposal_rehearsal(
        task_model_dir=task_model_dir,
        candidate_dir=candidate_dir,
        task=_Task(),
        args=_args(),
        config=runtime.make_config(_args()),
        seed=23,
        proposal_trace_buffer=["selected"],
        candidate_trace_examples=["candidate"],
        post_task_rehearsal_examples=["r0", "r1"],
        train_checkpoint_fn=fake_train_checkpoint,
        write_json_fn=fake_write_json,
    )

    assert model == "rehearsal-model"
    assert tokenizer == "rehearsal-tokenizer"
    assert model_dir == candidate_dir / "proposal_rehearsal" / "model"
    assert calls["train"]["source_checkpoint"] == str(task_model_dir)
    assert calls["train"]["output_dir"] == candidate_dir / "proposal_rehearsal"
    assert calls["train"]["train_examples"] == ["r0", "r1"]
    assert calls["train"]["seed"] == 60
    assert calls["train"]["recipe_phase_name"] == "proposal_rehearsal"
    summary_path, summary = calls["summary"]
    assert summary_path == candidate_dir / "proposal_rehearsal_summary.json"
    assert summary["examples"] == 2
    assert summary["base_candidate_trace_examples"] == 1
    assert summary["base_selected_trace_buffer_size"] == 1
    assert summary["repeat_count"] == 3
    assert summary["max_examples"] == 4
    assert not task_model_dir.parent.exists()
