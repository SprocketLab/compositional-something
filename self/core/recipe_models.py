#!/usr/bin/env python3
"""Recipe-backed tokenizer and no-position LLaMA model helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Tuple

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from self.core.recipe_presets import SelfImprovementRecipePreset
from self.core.tokenizers import ArithmeticCharacterTokenizer, build_arithmetic_self_improve_tokenizer


def sync_model_special_token_ids(model: Any, tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = model.config.pad_token_id
        generation_config.bos_token_id = model.config.bos_token_id
        generation_config.eos_token_id = model.config.eos_token_id


class NoPositionRotaryEmbedding(nn.Module):
    """Return an identity rotary embedding: cos=1, sin=0 for every position."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = position_ids.shape
        shape = (batch_size, seq_len, self.head_dim)
        cos = torch.ones(shape, dtype=x.dtype, device=x.device)
        sin = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return cos, sin


class NoPELlamaForCausalLM(LlamaForCausalLM):
    """LLaMA causal LM with rotary embeddings replaced by an identity transform."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model.rotary_emb = NoPositionRotaryEmbedding(config)
        self.config.use_no_position_embeddings = True
        self.config.architectures = [self.__class__.__name__]


def apply_recipe_runtime_settings(preset: SelfImprovementRecipePreset) -> None:
    if not torch.cuda.is_available():
        return
    if preset.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_recipe_tokenizer(preset: SelfImprovementRecipePreset) -> ArithmeticCharacterTokenizer:
    tokenizer = build_arithmetic_self_improve_tokenizer(model_max_length=preset.max_position_embeddings)
    tokenizer.padding_side = "left"
    return tokenizer


def build_recipe_model_config(
    tokenizer: ArithmeticCharacterTokenizer,
    preset: SelfImprovementRecipePreset,
) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_attention_heads,
        num_hidden_layers=preset.num_hidden_layers,
        max_position_embeddings=preset.max_position_embeddings,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    config.use_no_position_embeddings = True
    config.recipe_name = preset.name
    return config


def instantiate_recipe_model(
    tokenizer: ArithmeticCharacterTokenizer,
    preset: SelfImprovementRecipePreset,
    *,
    bf16: bool,
    fp16: bool,
) -> NoPELlamaForCausalLM:
    config = build_recipe_model_config(tokenizer, preset)
    model = NoPELlamaForCausalLM(config)
    sync_model_special_token_ids(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=dtype)
    return model


def load_recipe_model(
    model_path: Path,
    tokenizer: ArithmeticCharacterTokenizer,
    *,
    bf16: bool,
    fp16: bool,
) -> NoPELlamaForCausalLM:
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    config = LlamaConfig.from_pretrained(str(model_path))
    config.use_no_position_embeddings = bool(getattr(config, "use_no_position_embeddings", True))
    model = NoPELlamaForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=dtype,
    )
    sync_model_special_token_ids(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    return model


@contextmanager
def tokenizer_padding_side(tokenizer: ArithmeticCharacterTokenizer, side: str) -> Iterator[None]:
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = original_side
