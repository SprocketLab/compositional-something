"""RNG and metadata persistence runtime for non-adaptive runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from self.core.data_io import JsonDict, sanitize_json_value


@dataclass
class NonAdaptiveMetadataRuntime:
    rng: Any
    metadata: JsonDict
    metadata_path: Path
    json_module: Any
    persist_metadata_fn: Callable[..., None]
    encode_rng_state_fn: Callable[[tuple[Any, ...]], JsonDict]
    sanitize_json_value_fn: Callable[[Any], Any]

    def set_metadata(self, metadata: JsonDict) -> None:
        self.metadata = metadata

    def persist_metadata(self, target_metadata: JsonDict | None = None) -> None:
        metadata_to_persist = self.metadata if target_metadata is None else target_metadata
        self.persist_metadata_fn(
            metadata_to_persist,
            self.metadata_path,
            self.rng.getstate(),
            json_module=self.json_module,
            encode_rng_state_fn=self.encode_rng_state_fn,
            sanitize_json_value_fn=self.sanitize_json_value_fn,
        )


def prepare_nonadaptive_metadata_runtime(
    *,
    seed: int,
    metadata: JsonDict,
    metadata_path: Path,
    set_seed_fn: Callable[[int], None],
    random_cls: Callable[[int], Any],
    decode_rng_state_fn: Callable[[JsonDict], tuple[Any, ...]],
    persist_metadata_fn: Callable[..., None],
    json_module: Any,
    encode_rng_state_fn: Callable[[tuple[Any, ...]], JsonDict],
    sanitize_json_value_fn: Callable[[Any], Any] = sanitize_json_value,
) -> NonAdaptiveMetadataRuntime:
    """Seed RNGs, restore resumed RNG state, and return a metadata persister."""
    set_seed_fn(seed)
    rng = random_cls(seed)
    if metadata and "rng_state" in metadata:
        rng.setstate(decode_rng_state_fn(metadata["rng_state"]))

    return NonAdaptiveMetadataRuntime(
        rng=rng,
        metadata=metadata,
        metadata_path=metadata_path,
        json_module=json_module,
        persist_metadata_fn=persist_metadata_fn,
        encode_rng_state_fn=encode_rng_state_fn,
        sanitize_json_value_fn=sanitize_json_value_fn,
    )
