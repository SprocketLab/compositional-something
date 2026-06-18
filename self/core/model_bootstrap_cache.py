"""Process-local bootstrap cache helpers for repeated model initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class TokenizerBootstrap:
    tokenizer: Any
    added_token_initializers: Dict[int, int]


@dataclass
class CachedModelState:
    config: Any
    state_dict: Dict[str, torch.Tensor]


@dataclass
class ModelBootstrapCache:
    """Per-process cache for repeated candidate checkpoint bootstrap.

    The cache is intentionally process-local and only stores immutable bootstrap
    inputs. A fresh model object is still instantiated for each candidate.
    """

    cache_base_state: bool = False
    tokenizer_cache: Dict[Tuple[Any, ...], TokenizerBootstrap] = field(default_factory=dict)
    model_state_cache: Dict[Tuple[Any, ...], CachedModelState] = field(default_factory=dict)
    tokenizer_cache_hits: int = 0
    tokenizer_cache_misses: int = 0
    model_state_cache_hits: int = 0
    model_state_cache_misses: int = 0

    def stats(self) -> Dict[str, int]:
        return {
            "tokenizer_cache_entries": len(self.tokenizer_cache),
            "model_state_cache_entries": len(self.model_state_cache),
        }

    def detailed_stats(self) -> Dict[str, int]:
        return {
            **self.stats(),
            "cache_base_state": int(self.cache_base_state),
            "tokenizer_cache_hits": self.tokenizer_cache_hits,
            "tokenizer_cache_misses": self.tokenizer_cache_misses,
            "model_state_cache_hits": self.model_state_cache_hits,
            "model_state_cache_misses": self.model_state_cache_misses,
        }


def token_initializers_key(token_initializers: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (token_initializers or {}).items()))


def model_state_cache_key(
    model_path: str,
    *,
    bf16: bool,
    fp16: bool,
    tokenizer_mode: str,
    token_initializers: Optional[Dict[str, str]],
) -> Tuple[Any, ...]:
    return (
        str(model_path),
        bool(bf16),
        bool(fp16),
        tokenizer_mode,
        token_initializers_key(token_initializers),
    )


def clone_state_dict_to_cpu(model: Any) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


__all__ = [
    "CachedModelState",
    "ModelBootstrapCache",
    "TokenizerBootstrap",
    "clone_state_dict_to_cpu",
    "model_state_cache_key",
    "token_initializers_key",
]
