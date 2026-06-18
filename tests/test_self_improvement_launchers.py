from __future__ import annotations

from pathlib import Path

from self.legacy import run_length_self_improvement as run_length_launcher


def test_run_length_launcher_normalizes_bit_schedule():
    args = run_length_launcher.normalize_args(
        run_length_launcher.parse_args(
            [
                "--output-dir",
                "artifacts/tmp_run_length_launcher",
                "--initial-min-bits",
                "6",
                "--initial-max-bits",
                "10",
                "--initial-train-per-bit",
                "222",
                "--initial-eval-per-bit",
                "9",
                "--expand-num-bits",
                "4",
                "--expand-train-per-bit",
                "55",
                "--eval-per-bit",
                "77",
                "--composed-eval-per-bit",
                "11",
            ]
        )
    )

    assert args.initial_min_size == 6
    assert args.initial_max_size == 10
    assert args.initial_train_per_size == 222
    assert args.initial_eval_per_size == 9
    assert args.expand_num_size == 4
    assert args.expand_train_per_size == 55
    assert args.eval_per_size == 77
    assert args.composed_eval_per_size == 11


def test_run_length_launcher_preserves_recipe_and_bucket_flags():
    args = run_length_launcher.normalize_args(
        run_length_launcher.parse_args(
            [
                "--output-dir",
                "artifacts/tmp_run_length_launcher_recipe",
                "--recipe",
                "algorithmic_self_improve_v1",
                "--bucket-train-batches-by-bits",
                "--treat-seed-as-round-zero",
            ]
        )
    )

    assert args.recipe == "algorithmic_self_improve_v1"
    assert args.bucket_train_batches_by_bits is True
    assert args.bucket_train_batches_by_size is True
    assert args.treat_seed_as_round_zero is True


def test_run_length_launcher_main_wires_run_length_task(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_run_self_improvement(args, task) -> None:
        captured["args"] = args
        captured["task"] = task

    import self.nonadaptive.nonadaptive_loop as core

    monkeypatch.setattr(core, "run_self_improvement", fake_run_self_improvement)

    run_length_launcher.main(["--output-dir", str(tmp_path / "run_length")])

    assert captured["task"].name == "run_length"
    assert Path(captured["args"].output_dir) == tmp_path / "run_length"
