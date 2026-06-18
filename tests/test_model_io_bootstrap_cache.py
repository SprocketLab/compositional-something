from __future__ import annotations

from types import SimpleNamespace

import torch

from self.core import model_bootstrap_cache, model_io


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.padding_side = "right"

    def __len__(self) -> int:
        return 2

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1}


class _FakeModel(torch.nn.Module):
    def __init__(self, value: float, config=None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.config = config or SimpleNamespace()
        self.generation_config = SimpleNamespace()
        self._input_embeddings = SimpleNamespace(weight=torch.zeros((2, 1), dtype=torch.float32))

    def get_input_embeddings(self):
        return self._input_embeddings

    def get_output_embeddings(self):
        return None

    def resize_token_embeddings(self, size: int) -> None:
        self._input_embeddings = SimpleNamespace(weight=torch.zeros((size, 1), dtype=torch.float32))
        self.config.vocab_size = size


def test_model_bootstrap_cache_reuses_tokenizer_and_cached_base_state(monkeypatch):
    assert model_io.ModelBootstrapCache is model_bootstrap_cache.ModelBootstrapCache
    assert model_io.TokenizerBootstrap is model_bootstrap_cache.TokenizerBootstrap
    assert model_io.CachedModelState is model_bootstrap_cache.CachedModelState

    calls = {"tokenizer": 0, "from_pretrained": 0, "from_config": 0}

    def fake_tokenizer_from_pretrained(model_path, trust_remote_code=True):
        del model_path, trust_remote_code
        calls["tokenizer"] += 1
        return _FakeTokenizer()

    def fake_model_from_pretrained(model_path, trust_remote_code=True, torch_dtype=None):
        del model_path, trust_remote_code, torch_dtype
        calls["from_pretrained"] += 1
        return _FakeModel(float(calls["from_pretrained"]), config=SimpleNamespace(vocab_size=2))

    def fake_model_from_config(config, trust_remote_code=True):
        del trust_remote_code
        calls["from_config"] += 1
        return _FakeModel(0.0, config=config)

    monkeypatch.setattr(model_io.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(model_io.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(model_io.AutoModelForCausalLM, "from_config", fake_model_from_config)

    cache = model_io.ModelBootstrapCache(cache_base_state=True)
    first_model, first_tokenizer = model_io.instantiate_model_and_tokenizer(
        "checkpoint",
        bf16=False,
        fp16=False,
        tokenizer_mode="auto",
        bootstrap_cache=cache,
    )
    first_model.weight.data.fill_(99.0)
    second_model, second_tokenizer = model_io.instantiate_model_and_tokenizer(
        "checkpoint",
        bf16=False,
        fp16=False,
        tokenizer_mode="auto",
        bootstrap_cache=cache,
    )

    assert calls == {"tokenizer": 1, "from_pretrained": 1, "from_config": 1}
    assert first_tokenizer is second_tokenizer
    assert float(second_model.weight.item()) == 1.0
    assert cache.stats() == {"model_state_cache_entries": 1, "tokenizer_cache_entries": 1}
    assert cache.detailed_stats() == {
        "cache_base_state": 1,
        "model_state_cache_entries": 1,
        "model_state_cache_hits": 1,
        "model_state_cache_misses": 1,
        "tokenizer_cache_entries": 1,
        "tokenizer_cache_hits": 1,
        "tokenizer_cache_misses": 1,
    }


def test_model_bootstrap_cache_can_reuse_tokenizer_without_state_cache(monkeypatch):
    calls = {"tokenizer": 0, "from_pretrained": 0}

    def fake_tokenizer_from_pretrained(model_path, trust_remote_code=True):
        del model_path, trust_remote_code
        calls["tokenizer"] += 1
        return _FakeTokenizer()

    def fake_model_from_pretrained(model_path, trust_remote_code=True, torch_dtype=None):
        del model_path, trust_remote_code, torch_dtype
        calls["from_pretrained"] += 1
        return _FakeModel(float(calls["from_pretrained"]), config=SimpleNamespace(vocab_size=2))

    monkeypatch.setattr(model_io.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(model_io.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    cache = model_io.ModelBootstrapCache(cache_base_state=False)
    first_model, first_tokenizer = model_io.instantiate_model_and_tokenizer(
        "checkpoint",
        bf16=False,
        fp16=False,
        tokenizer_mode="auto",
        bootstrap_cache=cache,
    )
    second_model, second_tokenizer = model_io.instantiate_model_and_tokenizer(
        "checkpoint",
        bf16=False,
        fp16=False,
        tokenizer_mode="auto",
        bootstrap_cache=cache,
    )

    assert calls == {"tokenizer": 1, "from_pretrained": 2}
    assert first_tokenizer is second_tokenizer
    assert float(first_model.weight.item()) == 1.0
    assert float(second_model.weight.item()) == 2.0
    assert cache.stats() == {"model_state_cache_entries": 0, "tokenizer_cache_entries": 1}
    assert cache.detailed_stats() == {
        "cache_base_state": 0,
        "model_state_cache_entries": 0,
        "model_state_cache_hits": 0,
        "model_state_cache_misses": 0,
        "tokenizer_cache_entries": 1,
        "tokenizer_cache_hits": 1,
        "tokenizer_cache_misses": 1,
    }
