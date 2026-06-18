#!/usr/bin/env python3
"""Round-wise trainer-seed beam search for alpha-10 symbol-run run-length."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-run",
        type=Path,
        default=Path(
            "artifacts/runs/run_length_multisymbol_pair_alpha10_strongseed_warmup500_schedulerfix_20260423_202440/"
            "run_length/pilot/guarded_compose"
        ),
        help="Existing run directory that contains data/ and round_00 pseudo artifacts.",
    )
    parser.add_argument(
        "--seed-model",
        type=Path,
        default=Path("artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model"),
    )
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--candidate-seeds", type=str, default="7,123,999")
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--max-round", type=int, default=8)
    parser.add_argument("--round-warmup-steps", type=int, default=500)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--python-bin", type=Path, default=Path.home() / ".conda/envs/torch-env/bin/python")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--expand-train-per-bit", type=int, default=2000)
    parser.add_argument(
        "--round-learning-rate",
        type=float,
        default=None,
        help="Optional recipe learning-rate override for the continued self-improvement rounds.",
    )
    parser.add_argument(
        "--round-lr-switch-round",
        type=int,
        default=None,
        help="Optional round index at which to switch to --round-learning-rate-after-switch.",
    )
    parser.add_argument(
        "--round-learning-rate-after-switch",
        type=float,
        default=None,
        help="Learning rate used from --round-lr-switch-round onward.",
    )
    parser.add_argument("--bit-composition-path-mode", choices=("random", "fixed_binary"), default="random")
    parser.add_argument(
        "--baseline",
        choices=("direct", "unfiltered_compose", "guarded_compose"),
        default="guarded_compose",
        help="Which alpha-10 branch to run with the round-wise trainer-seed beam.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: List[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def load_round_metrics(run_dir: Path, round_idx: int) -> Dict[str, Any]:
    results_path = run_dir / "self_improvement_results.json"
    payload = json.loads(results_path.read_text())
    for row in payload:
        if int(row["round"]) == round_idx:
            return row
    raise RuntimeError(f"Could not find round {round_idx} in {results_path}")


def metric_score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    max90 = row.get("max_bits_at_90_accuracy")
    eval_accuracy = row.get("eval_accuracy")
    composed_eval_accuracy = row.get("composed_eval_accuracy")
    return (
        float(max90) if max90 is not None else -math.inf,
        float(eval_accuracy) if eval_accuracy is not None else -math.inf,
        float(composed_eval_accuracy) if composed_eval_accuracy is not None else -math.inf,
    )


def trim_results(run_dir: Path, *, resume_round: int) -> None:
    results_path = run_dir / "self_improvement_results.json"
    if not results_path.exists():
        return
    payload = json.loads(results_path.read_text())
    trimmed = [row for row in payload if int(row.get("round", -1)) < resume_round]
    results_path.write_text(json.dumps(trimmed, indent=2) + "\n")


def copy_branch(parent_dir: Path, candidate_dir: Path, *, resume_round: int) -> None:
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_dir, candidate_dir, symlinks=True)
    for child in candidate_dir.glob("round_*"):
        if not child.is_dir():
            continue
        try:
            round_idx = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if round_idx >= resume_round:
            shutil.rmtree(child)
    trim_results(candidate_dir, resume_round=resume_round)


def prune_model_weights(run_dir: Path) -> None:
    """Keep metrics/data for failed branches, but remove reloadable checkpoint weights."""
    for round_dir in run_dir.glob("round_*"):
        if not round_dir.is_dir():
            continue
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ):
            path = round_dir / name
            if path.exists():
                path.unlink()
        trainer_dir = round_dir / "trainer"
        if trainer_dir.exists():
            shutil.rmtree(trainer_dir)


def job_state(job_id: str, *, repo_root: Path) -> str | None:
    try:
        output = run_command(
            ["sacct", "-j", job_id, "--format=JobID,State", "-n", "-P"],
            cwd=repo_root,
        )
    except subprocess.CalledProcessError:
        output = ""
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] == job_id:
            return parts[1].split()[0]

    try:
        output = run_command(["squeue", "-h", "-j", job_id, "-o", "%T"], cwd=repo_root)
    except subprocess.CalledProcessError:
        return None
    return output.strip() or None


def wait_for_jobs(job_ids: List[str], *, repo_root: Path, poll_seconds: int) -> Dict[str, str]:
    states: Dict[str, str] = {}
    pending = set(job_ids)
    while pending:
        for job_id in list(pending):
            state = job_state(job_id, repo_root=repo_root)
            if state in TERMINAL_STATES:
                states[job_id] = state
                pending.remove(job_id)
        if pending:
            print(f"[INFO] Waiting on jobs: {', '.join(sorted(pending))}", flush=True)
            time.sleep(poll_seconds)
    return states


def submit_candidate(
    *,
    repo_root: Path,
    python_bin: Path,
    seed_model: Path,
    candidate_dir: Path,
    candidate_seed: int,
    round_idx: int,
    max_round: int,
    round_warmup_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    expand_train_per_bit: int,
    round_learning_rate: float | None,
    round_lr_switch_round: int | None,
    round_learning_rate_after_switch: float | None,
    bit_composition_path_mode: str,
    baseline: str,
    logs_dir: Path,
    dry_run: bool,
) -> str:
    if baseline == "direct":
        pseudo_label_mode = "direct"
        guarded_rule = "none"
    elif baseline == "unfiltered_compose":
        pseudo_label_mode = "compose"
        guarded_rule = "run_length_unfiltered_pair"
    elif baseline == "guarded_compose":
        pseudo_label_mode = "compose"
        guarded_rule = "run_length_no_boundary_continue"
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    command = [
        f"cd {repo_root}",
        "&&",
        "PYTHONPATH=.",
        str(python_bin),
        "-m",
        "self.legacy.run_length_self_improvement",
        "--output-dir",
        str(candidate_dir),
        "--model-name",
        str(seed_model),
        "--format-version",
        "legacy",
        "--target-mode",
        "symbol_run_pair",
        "--compose-arity",
        "exact2",
        "--bit-composition-path-mode",
        bit_composition_path_mode,
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
        str(max_round),
        "--expand-num-bits",
        "9",
        "--expand-train-per-bit",
        str(expand_train_per_bit),
        "--eval-per-bit",
        "100",
        "--composed-eval-per-bit",
        "100",
        "--pseudo-label-mode",
        pseudo_label_mode,
        "--guarded-compose-rule",
        guarded_rule,
        "--bucket-train-batches-by-bits",
        "--bf16",
        "--per-device-train-batch-size",
        str(train_batch_size),
        "--per-device-eval-batch-size",
        str(eval_batch_size),
        "--seed",
        str(candidate_seed),
        "--save-model-policy",
        "all_rounds",
        "--self-improve-warmup-steps",
        str(round_warmup_steps),
        "--resume",
        "--resume-from-round",
        str(round_idx),
        "--stop-after-round",
        str(round_idx),
    ]
    if round_learning_rate is not None:
        command.extend(["--self-improve-learning-rate", str(round_learning_rate)])
    if round_lr_switch_round is not None:
        command.extend(["--self-improve-lr-switch-round", str(round_lr_switch_round)])
    if round_learning_rate_after_switch is not None:
        command.extend(["--self-improve-learning-rate-after-switch", str(round_learning_rate_after_switch)])
    wrapped = " ".join(command)
    output_stem = logs_dir / f"rl-a10-beam-r{round_idx:02d}-s{candidate_seed}-%j"
    sbatch_command = [
        "sbatch",
        "--job-name",
        f"rl-a10-b{round_idx:02d}-s{candidate_seed}",
        "--output",
        str(output_stem) + ".out",
        "--error",
        str(output_stem) + ".err",
        "--partition",
        "mig",
        "--gres",
        "gpu:1g.10gb:1",
        "--cpus-per-task",
        "1",
        "--mem",
        "64G",
        "--time",
        "12:00:00",
        "--wrap",
        wrapped,
    ]
    if dry_run:
        print("[DRY-RUN]", " ".join(sbatch_command), flush=True)
        return f"dryrun-r{round_idx}-s{candidate_seed}"
    output = run_command(sbatch_command, cwd=repo_root)
    return output.split()[-1]


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    template_run = (repo_root / args.template_run).resolve() if not args.template_run.is_absolute() else args.template_run
    seed_model = (repo_root / args.seed_model).resolve() if not args.seed_model.is_absolute() else args.seed_model
    candidate_seeds = [int(item) for item in args.candidate_seeds.split(",") if item.strip()]
    if not candidate_seeds:
        raise ValueError("candidate-seeds must contain at least one integer.")
    if args.out_root is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_root = repo_root / f"artifacts/runs/run_length_symbol_pair_alpha10_seed_beam8_{timestamp}"
    else:
        out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    logs_dir = repo_root / "artifacts/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    parent_dir = out_root / "round_00_seed_branch" / args.baseline
    # When resuming from a later beam round, keep the already-selected
    # checkpoint/data history in the template branch.
    copy_branch(template_run, parent_dir, resume_round=args.start_round)

    summary: Dict[str, Any] = {
        "template_run": str(template_run),
        "seed_model": str(seed_model),
        "candidate_seeds": candidate_seeds,
        "bit_composition_path_mode": args.bit_composition_path_mode,
        "baseline": args.baseline,
        "rounds": [],
    }
    (out_root / "beam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for round_idx in range(args.start_round, args.max_round + 1):
        print(f"[INFO] Starting beam round {round_idx}", flush=True)
        candidate_records: List[Dict[str, Any]] = []
        job_ids: List[str] = []
        for candidate_seed in candidate_seeds:
            candidate_dir = out_root / f"round_{round_idx:02d}" / f"seed_{candidate_seed}" / args.baseline
            copy_branch(parent_dir, candidate_dir, resume_round=round_idx)
            job_id = submit_candidate(
                repo_root=repo_root,
                python_bin=args.python_bin,
                seed_model=seed_model,
                candidate_dir=candidate_dir,
                candidate_seed=candidate_seed,
                round_idx=round_idx,
                max_round=args.max_round,
                round_warmup_steps=args.round_warmup_steps,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                expand_train_per_bit=args.expand_train_per_bit,
                round_learning_rate=args.round_learning_rate,
                round_lr_switch_round=args.round_lr_switch_round,
                round_learning_rate_after_switch=args.round_learning_rate_after_switch,
                bit_composition_path_mode=args.bit_composition_path_mode,
                baseline=args.baseline,
                logs_dir=logs_dir,
                dry_run=args.dry_run,
            )
            record = {
                "job_id": job_id,
                "seed": candidate_seed,
                "run_dir": str(candidate_dir),
            }
            candidate_records.append(record)
            job_ids.append(job_id)
            print(f"[INFO] Submitted round {round_idx} seed {candidate_seed}: job {job_id}", flush=True)

        if args.dry_run:
            round_summary = {
                "round": round_idx,
                "candidates": candidate_records,
                "selected": None,
                "dry_run": True,
            }
            summary["rounds"].append(round_summary)
            (out_root / f"round_{round_idx:02d}_selection.json").write_text(json.dumps(round_summary, indent=2) + "\n")
            (out_root / "beam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("[INFO] Dry run complete; not waiting for or scoring jobs.", flush=True)
            break

        states = wait_for_jobs(
            job_ids,
            repo_root=repo_root,
            poll_seconds=args.poll_seconds,
        )
        for record in candidate_records:
            record["state"] = states.get(record["job_id"], "UNKNOWN")
            if record["state"] != "COMPLETED":
                record["score"] = [-math.inf, -math.inf, -math.inf]
                continue
            metrics = load_round_metrics(Path(record["run_dir"]), round_idx)
            score = metric_score(metrics)
            record["score"] = list(score)
            record["metrics"] = {
                "round": round_idx,
                "eval_accuracy": metrics.get("eval_accuracy"),
                "composed_eval_accuracy": metrics.get("composed_eval_accuracy"),
                "max_bits_at_90_accuracy": metrics.get("max_bits_at_90_accuracy"),
            }

        completed = [record for record in candidate_records if record.get("state") == "COMPLETED"]
        if not completed:
            raise RuntimeError(f"No completed candidates for round {round_idx}: {candidate_records}")
        best_record = max(completed, key=lambda record: tuple(record["score"]))
        parent_dir = Path(best_record["run_dir"])
        for record in candidate_records:
            if record is not best_record:
                prune_model_weights(Path(record["run_dir"]))

        round_summary = {
            "round": round_idx,
            "candidates": candidate_records,
            "selected": best_record,
        }
        summary["rounds"].append(round_summary)
        (out_root / f"round_{round_idx:02d}_selection.json").write_text(json.dumps(round_summary, indent=2) + "\n")
        (out_root / "beam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "[INFO] Selected round {round_idx} seed {seed}: score={score} metrics={metrics}".format(
                round_idx=round_idx,
                seed=best_record["seed"],
                score=best_record["score"],
                metrics=best_record.get("metrics"),
            ),
            flush=True,
        )

    best_link = out_root / f"best_{args.baseline}"
    if best_link.exists() or best_link.is_symlink():
        best_link.unlink()
    os.symlink(os.path.relpath(parent_dir, out_root), best_link)
    print(f"[INFO] Beam search complete. Best branch: {parent_dir}", flush=True)
    print(f"[INFO] Stable symlink: {best_link}", flush=True)


if __name__ == "__main__":
    main()
