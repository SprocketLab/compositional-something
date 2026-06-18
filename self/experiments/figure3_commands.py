"""Command builders shared by Figure 3 experiment submitters."""

from __future__ import annotations

from typing import Any, List, Mapping


def seed_fit_command(
    entry: Mapping[str, Any],
    *,
    python_bin: str,
    max_steps_position: str,
    run_length_max_steps: int = 15000,
    addition_max_steps: int = 10000,
) -> List[str]:
    task = str(entry["task"])
    train_count = int(entry["train_count"])
    max_steps = int(
        entry.get(
            "max_steps",
            run_length_max_steps if task == "run_length" else addition_max_steps,
        )
    )
    command = [
        python_bin,
        "-m",
        "self.experiments.seed_fit_experiment",
        "--task",
        task,
        "--model-name",
        "recipe_scratch",
        "--output-dir",
        str(entry["output_root"]),
        "--format-version",
        "legacy",
        "--recipe",
        "algorithmic_self_improve_v1" if task == "run_length" else "arithmetic_self_improve_v1",
        "--init-from-scratch",
        "--initial-train-per-size",
        str(train_count),
        "--expand-num-size",
        "1",
        "--expand-train-per-size",
        "0",
        "--eval-per-size",
        "0",
        "--composed-eval-per-size",
        "0",
        "--num-expand-rounds",
        "0",
        "--pseudo-label-mode",
        "none",
        "--num-epochs",
        "1",
    ]
    if max_steps_position == "before_batch_args":
        command += ["--max-steps", str(max_steps)]
    elif max_steps_position != "task_specific":
        raise ValueError(f"Unsupported max_steps_position: {max_steps_position}")
    command += [
        "--per-device-train-batch-size",
        "256",
        "--per-device-eval-batch-size",
        "256",
        "--gradient-accumulation-steps",
        "1",
        "--bucket-train-batches-by-size",
        "--save-model",
        "--bf16",
        "--seed",
        "0",
    ]
    if task == "run_length":
        command += [
            "--target-mode",
            "symbol_run_pair",
            "--symbol-alphabet-size",
            "10",
            "--initial-min-size",
            "6",
            "--initial-max-size",
            "10",
            "--initial-eval-per-size",
            "100",
        ]
        if max_steps_position == "task_specific":
            command += ["--max-steps", str(max_steps)]
        command += ["--decode-max-new-tokens", "16"]
    else:
        command += [
            "--addition-width-mode",
            "exact_digits",
            "--addition-composition-path-mode",
            "random",
            "--initial-min-size",
            "3",
            "--initial-max-size",
            "7",
            "--initial-eval-per-size",
            "200",
        ]
        if max_steps_position == "task_specific":
            command += ["--max-steps", str(max_steps)]
        command += ["--decode-max-new-tokens", "48"]
    return command


def run_length_self_improvement_command(
    entry: Mapping[str, Any],
    *,
    python_bin: str,
    num_expand_rounds: int,
) -> List[str]:
    return [
        python_bin,
        "-m",
        "self.legacy.run_length_self_improvement",
        "--model-name",
        str(entry["seed_model"]),
        "--output-dir",
        str(entry["output_root"]),
        "--format-version",
        "legacy",
        "--target-mode",
        "symbol_run_pair",
        "--compose-arity",
        "exact2",
        "--bit-composition-path-mode",
        "random",
        "--recipe",
        "algorithmic_self_improve_v1",
        "--treat-seed-as-round-zero",
        "--symbol-alphabet-size",
        "10",
        "--initial-min-bits",
        "6",
        "--initial-max-bits",
        "10",
        "--initial-train-per-bit",
        "50000",
        "--initial-eval-per-bit",
        "100",
        "--frontier-min-bits",
        "12",
        "--num-expand-rounds",
        str(num_expand_rounds),
        "--expand-num-bits",
        "9",
        "--expand-train-per-bit",
        str(entry["sample_size"]),
        "--eval-per-bit",
        "100",
        "--composed-eval-per-bit",
        "100",
        "--pseudo-label-mode",
        "compose",
        "--guarded-compose-rule",
        "run_length_no_boundary_continue",
        "--bucket-train-batches-by-bits",
        "--save-model-policy",
        "all_rounds",
        "--self-improve-warmup-steps",
        "500",
        "--per-device-train-batch-size",
        "256",
        "--per-device-eval-batch-size",
        "256",
        "--bf16",
        "--seed",
        "7",
    ]
