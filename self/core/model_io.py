"""Tokenizer and model loading helpers for self-improvement runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from self.core.model_bootstrap_cache import (
    CachedModelState,
    ModelBootstrapCache,
    TokenizerBootstrap,
    clone_state_dict_to_cpu,
    model_state_cache_key,
    token_initializers_key,
)
from self.core.recipes import (
    apply_recipe_runtime_settings,
    build_recipe_tokenizer,
    instantiate_recipe_model,
    load_recipe_model,
)
from self.core.recipes import (
    recipe_enabled,
    resolve_self_improvement_recipe,
)
from self.core.tokenizers import build_fixed_char_tokenizer


def _dtype_for_precision(*, bf16: bool, fp16: bool) -> torch.dtype:
    return torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)


def lookup_single_token_id(tokenizer: AutoTokenizer, token_text: str) -> int:
    encoded = tokenizer.encode(token_text, add_special_tokens=False)
    if len(encoded) == 1:
        return int(encoded[0])
    fallback_id = tokenizer.convert_tokens_to_ids(token_text)
    if fallback_id is None or fallback_id == tokenizer.unk_token_id:
        raise ValueError(f"Unable to map {token_text!r} to a single tokenizer id.")
    return int(fallback_id)


def add_token_initializers(
    tokenizer: AutoTokenizer,
    token_initializers: Optional[Dict[str, str]],
) -> Dict[int, int]:
    if not token_initializers:
        return {}
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [token for token in token_initializers if token not in existing_vocab]
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
    initializer_map: Dict[int, int] = {}
    for token, source_token in token_initializers.items():
        if token in existing_vocab:
            continue
        initializer_map[lookup_single_token_id(tokenizer, token)] = lookup_single_token_id(tokenizer, source_token)
    return initializer_map


def initialize_copied_embeddings(model: AutoModelForCausalLM, initializer_map: Dict[int, int]) -> None:
    if not initializer_map:
        return
    input_weights = model.get_input_embeddings().weight.data
    output_head = model.get_output_embeddings()
    output_weights = output_head.weight.data if output_head is not None and hasattr(output_head, "weight") else None
    for new_id, source_id in initializer_map.items():
        input_weights[new_id].copy_(input_weights[source_id])
        if output_weights is not None:
            output_weights[new_id].copy_(output_weights[source_id])


def _sync_config_special_token_ids(config: object, tokenizer: AutoTokenizer) -> None:
    if config is None:
        return
    if tokenizer.pad_token_id is not None:
        setattr(config, "pad_token_id", tokenizer.pad_token_id)
    if tokenizer.bos_token_id is not None:
        setattr(config, "bos_token_id", tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        setattr(config, "eos_token_id", tokenizer.eos_token_id)


def sync_model_special_token_ids(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    _sync_config_special_token_ids(model.config, tokenizer)
    for nested_name in ("text_config", "decoder_config", "language_config"):
        _sync_config_special_token_ids(getattr(model.config, nested_name, None), tokenizer)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = getattr(model.config, "pad_token_id", None)
        generation_config.bos_token_id = getattr(model.config, "bos_token_id", None)
        generation_config.eos_token_id = getattr(model.config, "eos_token_id", None)


def load_model_for_tokenizer(
    model_path: str,
    tokenizer: AutoTokenizer,
    *,
    bf16: bool,
    fp16: bool,
    added_token_initializers: Optional[Dict[int, int]] = None,
) -> AutoModelForCausalLM:
    dtype = _dtype_for_precision(bf16=bf16, fp16=fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    sync_model_special_token_ids(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    initialize_copied_embeddings(model, added_token_initializers or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def load_model_from_config(
    model_path: str,
    tokenizer: AutoTokenizer,
    *,
    bf16: bool,
    fp16: bool,
    added_token_initializers: Optional[Dict[int, int]] = None,
) -> AutoModelForCausalLM:
    dtype = _dtype_for_precision(bf16=bf16, fp16=fp16)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    sync_model_special_token_ids(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    initialize_copied_embeddings(model, added_token_initializers or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=dtype)
    return model


def _build_tokenizer_bootstrap(
    model_path: str,
    *,
    token_initializers: Optional[Dict[str, str]],
    tokenizer_mode: str,
) -> TokenizerBootstrap:
    if tokenizer_mode == "fixed_char":
        tokenizer = build_fixed_char_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"
    return TokenizerBootstrap(
        tokenizer=tokenizer,
        added_token_initializers=add_token_initializers(tokenizer, token_initializers),
    )


def _tokenizer_bootstrap(
    model_path: str,
    *,
    token_initializers: Optional[Dict[str, str]],
    tokenizer_mode: str,
    bootstrap_cache: Optional[ModelBootstrapCache],
) -> TokenizerBootstrap:
    cache_key = (str(model_path), tokenizer_mode, token_initializers_key(token_initializers))
    if bootstrap_cache is not None and cache_key in bootstrap_cache.tokenizer_cache:
        bootstrap_cache.tokenizer_cache_hits += 1
        return bootstrap_cache.tokenizer_cache[cache_key]
    if bootstrap_cache is not None:
        bootstrap_cache.tokenizer_cache_misses += 1
    bootstrap = _build_tokenizer_bootstrap(
        model_path,
        token_initializers=token_initializers,
        tokenizer_mode=tokenizer_mode,
    )
    if bootstrap_cache is not None:
        bootstrap_cache.tokenizer_cache[cache_key] = bootstrap
    return bootstrap


def _load_model_from_cached_state(
    cached: CachedModelState,
    tokenizer: AutoTokenizer,
    *,
    bf16: bool,
    fp16: bool,
) -> AutoModelForCausalLM:
    dtype = _dtype_for_precision(bf16=bf16, fp16=fp16)
    model = AutoModelForCausalLM.from_config(copy.deepcopy(cached.config), trust_remote_code=True)
    sync_model_special_token_ids(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device=torch.device("cpu"), dtype=dtype)
    model.load_state_dict(cached.state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def instantiate_model_and_tokenizer(
    model_path: str,
    *,
    bf16: bool,
    fp16: bool,
    token_initializers: Optional[Dict[str, str]] = None,
    init_from_scratch: bool = False,
    tokenizer_mode: str = "auto",
    recipe: str = "none",
    bootstrap_cache: Optional[ModelBootstrapCache] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if recipe_enabled(recipe):
        preset = resolve_self_improvement_recipe(recipe)
        apply_recipe_runtime_settings(preset)
        recipe_tokenizer_key = ("recipe", preset.name)
        if bootstrap_cache is not None and recipe_tokenizer_key in bootstrap_cache.tokenizer_cache:
            bootstrap_cache.tokenizer_cache_hits += 1
            tokenizer = bootstrap_cache.tokenizer_cache[recipe_tokenizer_key].tokenizer
        else:
            if bootstrap_cache is not None:
                bootstrap_cache.tokenizer_cache_misses += 1
            tokenizer = build_recipe_tokenizer(preset)
            if bootstrap_cache is not None:
                bootstrap_cache.tokenizer_cache[recipe_tokenizer_key] = TokenizerBootstrap(
                    tokenizer=tokenizer,
                    added_token_initializers={},
                )
        if init_from_scratch:
            model = instantiate_recipe_model(tokenizer, preset, bf16=bf16, fp16=fp16)
        else:
            model_dir = Path(model_path)
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Recipe-backed self-improvement expects a local checkpoint directory, got {model_path!r}."
                )
            model = load_recipe_model(model_dir, tokenizer, bf16=bf16, fp16=fp16)
        return model, tokenizer

    tokenizer_bootstrap = _tokenizer_bootstrap(
        model_path,
        token_initializers=token_initializers,
        tokenizer_mode=tokenizer_mode,
        bootstrap_cache=bootstrap_cache,
    )
    tokenizer = tokenizer_bootstrap.tokenizer
    added_token_initializers = tokenizer_bootstrap.added_token_initializers
    if init_from_scratch:
        model = load_model_from_config(
            model_path,
            tokenizer,
            bf16=bf16,
            fp16=fp16,
            added_token_initializers=added_token_initializers,
        )
    else:
        state_key = model_state_cache_key(
            model_path,
            bf16=bf16,
            fp16=fp16,
            tokenizer_mode=tokenizer_mode,
            token_initializers=token_initializers,
        )
        if bootstrap_cache is not None and bootstrap_cache.cache_base_state and state_key in bootstrap_cache.model_state_cache:
            bootstrap_cache.model_state_cache_hits += 1
            model = _load_model_from_cached_state(
                bootstrap_cache.model_state_cache[state_key],
                tokenizer,
                bf16=bf16,
                fp16=fp16,
            )
        else:
            if bootstrap_cache is not None and bootstrap_cache.cache_base_state:
                bootstrap_cache.model_state_cache_misses += 1
            model = load_model_for_tokenizer(
                model_path,
                tokenizer,
                bf16=bf16,
                fp16=fp16,
                added_token_initializers=added_token_initializers,
            )
            if bootstrap_cache is not None and bootstrap_cache.cache_base_state:
                bootstrap_cache.model_state_cache[state_key] = CachedModelState(
                    config=copy.deepcopy(model.config),
                    state_dict=clone_state_dict_to_cpu(model),
                )
    return model, tokenizer
