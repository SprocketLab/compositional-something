from __future__ import annotations

from pathlib import Path

from self.nonadaptive.nonadaptive_metadata_runtime import prepare_nonadaptive_metadata_runtime


class _FakeRng:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.state = ("seed", seed)

    def getstate(self):
        return self.state

    def setstate(self, state) -> None:
        self.state = state


def test_prepare_nonadaptive_metadata_runtime_seeds_and_restores_rng(tmp_path: Path):
    seed_calls = []

    runtime = prepare_nonadaptive_metadata_runtime(
        seed=17,
        metadata={"rng_state": {"encoded": "resume"}},
        metadata_path=tmp_path / "metadata.json",
        set_seed_fn=seed_calls.append,
        random_cls=_FakeRng,
        decode_rng_state_fn=lambda payload: ("decoded", payload["encoded"]),
        persist_metadata_fn=lambda *args, **kwargs: None,
        json_module=None,
        encode_rng_state_fn=lambda state: {"state": state},
    )

    assert seed_calls == [17]
    assert runtime.rng.seed == 17
    assert runtime.rng.getstate() == ("decoded", "resume")


def test_metadata_runtime_persists_default_and_target_metadata(tmp_path: Path):
    calls = []

    def persist_metadata(metadata, metadata_path, rng_state, **kwargs):
        calls.append((metadata, metadata_path, rng_state, kwargs))

    runtime = prepare_nonadaptive_metadata_runtime(
        seed=3,
        metadata={"run": "initial"},
        metadata_path=tmp_path / "metadata.json",
        set_seed_fn=lambda seed: None,
        random_cls=_FakeRng,
        decode_rng_state_fn=lambda payload: ("decoded", payload),
        persist_metadata_fn=persist_metadata,
        json_module="json-module",
        encode_rng_state_fn=lambda state: {"state": state},
        sanitize_json_value_fn=lambda value: {"sanitized": value},
    )

    runtime.persist_metadata()
    explicit = {"run": "explicit"}
    runtime.persist_metadata(explicit)
    updated = {"run": "updated"}
    runtime.set_metadata(updated)
    runtime.persist_metadata()

    assert [call[0] for call in calls] == [{"run": "initial"}, explicit, updated]
    assert [call[1] for call in calls] == [tmp_path / "metadata.json"] * 3
    assert [call[2] for call in calls] == [("seed", 3)] * 3
    assert calls[0][3]["json_module"] == "json-module"
    assert calls[0][3]["encode_rng_state_fn"](("x",)) == {"state": ("x",)}
    assert calls[0][3]["sanitize_json_value_fn"]("x") == {"sanitized": "x"}
