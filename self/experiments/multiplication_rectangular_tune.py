#!/usr/bin/env python3
"""Submit and collect rectangular multiplication tuning sweeps on MIG."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "artifacts/logs"
DEFAULT_SEED_MODEL = ROOT_DIR / "artifacts/models/multiplication_rectangular_seed_best"
SELF_IMPROVEMENT_LAUNCHER = ROOT_DIR / "launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch"

STAGE1_CANDIDATES = (
    {
        "frontier_row_profile": "uniform",
        "expand_train_per_partition": 4000,
        "seed_replay_train_per_partition": 2000,
        "learning_rate": 5e-5,
        "max_steps": 3000,
    },
    {
        "frontier_row_profile": "uniform",
        "expand_train_per_partition": 4000,
        "seed_replay_train_per_partition": 2000,
        "learning_rate": 1e-4,
        "max_steps": 6000,
    },
    {
        "frontier_row_profile": "hard_rows_v1",
        "expand_train_per_partition": 1000,
        "seed_replay_train_per_partition": 3000,
        "learning_rate": 5e-5,
        "max_steps": 3000,
    },
    {
        "frontier_row_profile": "hard_rows_v1",
        "expand_train_per_partition": 1000,
        "seed_replay_train_per_partition": 3000,
        "learning_rate": 1e-4,
        "max_steps": 6000,
    },
    {
        "frontier_row_profile": "hard_rows_v1",
        "expand_train_per_partition": 2000,
        "seed_replay_train_per_partition": 3000,
        "learning_rate": 5e-5,
        "max_steps": 3000,
    },
    {
        "frontier_row_profile": "hard_rows_v1",
        "expand_train_per_partition": 2000,
        "seed_replay_train_per_partition": 3000,
        "learning_rate": 1e-4,
        "max_steps": 6000,
    },
)
STAGE2_BASELINE = "direct"
STAGE3_BASELINES = ("short_only", "compose_corrupt")

FIXED_SCHEDULE = {
    "initial_max_b_digits": 8,
    "expand_b_digits": 1,
    "num_expand_rounds": 8,
    "frontier_min_a_digits": 1,
    "frontier_max_a_digits": 6,
    "frontier_min_b_digits": 2,
    "train_batch_size": 256,
    "eval_batch_size": 256,
}

SELECTOR_JOB_CONFIG = {
    "cpus_per_task": "1",
    "mem": "8G",
    "time": "01:00:00",
}


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> Optional[str]:
    printable = shlex.join(list(cmd))
    print(f"[INFO] Command: {printable}", flush=True)
    if dry_run:
        return None
    completed = subprocess.run(
        list(cmd),
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout, flush=True)
    stderr = completed.stderr.strip()
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    return stdout


def _lr_tag(value: float) -> str:
    return f"{value:.0e}".replace("-", "m")


def _schedule_label(entry: Mapping[str, Any]) -> str:
    return (
        f"profile_{entry['frontier_row_profile']}"
        f"_expand{entry['expand_train_per_partition']}"
        f"_replay{entry['seed_replay_train_per_partition']}"
        f"_lr{_lr_tag(float(entry['learning_rate']))}"
        f"_steps{entry['max_steps']}"
    )


def _selector_log_paths(log_dir: Path, job_name: str) -> tuple[Path, Path]:
    safe_name = job_name.replace("/", "-")
    return (
        log_dir / f"{safe_name}-%j.out",
        log_dir / f"{safe_name}-%j.err",
    )


def _default_stage1_manifest(out_root: Path) -> Path:
    return out_root / "stage1_manifest.json"


def _default_stage1_selection(out_root: Path) -> Path:
    return out_root / "stage1_selection.json"


def _default_stage2_selection(out_root: Path) -> Path:
    return out_root / "stage2_selection.json"


def _default_stage3_selection(out_root: Path) -> Path:
    return out_root / "stage3_selection.json"


def build_stage1_manifest(
    *,
    out_root: Path,
    log_dir: Path,
    python_bin: str,
    seed_model: Path,
) -> Dict[str, Any]:
    stage1_jobs: List[Dict[str, Any]] = []
    for candidate in STAGE1_CANDIDATES:
        schedule_label = _schedule_label(candidate)
        output_root = out_root / "stage1" / schedule_label
        stage1_jobs.append(
            {
                "task": "multiplication_rectangular",
                "stage": 1,
                "baseline": "compose",
                "schedule_label": schedule_label,
                "output_root": str(output_root),
                "results_path": str(output_root / "compose" / "self_improvement_results.json"),
                **FIXED_SCHEDULE,
                **candidate,
            }
        )
    return {
        "meta": {
            "out_root": str(out_root),
            "log_dir": str(log_dir),
            "python_bin": python_bin,
            "seed_model": str(seed_model),
            "launcher": str(SELF_IMPROVEMENT_LAUNCHER),
        },
        "stage1_jobs": stage1_jobs,
    }


def _submit_rectangular_job(
    *,
    entry: Mapping[str, Any],
    seed_model: Path,
    dry_run: bool,
) -> Optional[str]:
    cmd = [
        "sbatch",
        "--parsable",
        "--export",
        (
            "ALL,"
            f"OUT_ROOT={entry['output_root']},"
            f"BASELINE={entry['baseline']},"
            f"SEED_MODEL={seed_model},"
            f"TRAIN_BATCH_SIZE={entry['train_batch_size']},"
            f"EVAL_BATCH_SIZE={entry['eval_batch_size']},"
            f"SEED_REPLAY_TRAIN_PER_PARTITION={entry['seed_replay_train_per_partition']},"
            f"EXPAND_TRAIN_PER_PARTITION={entry['expand_train_per_partition']},"
            f"FRONTIER_ROW_PROFILE={entry['frontier_row_profile']},"
            f"INITIAL_MAX_B_DIGITS={entry['initial_max_b_digits']},"
            f"EXPAND_B_DIGITS={entry['expand_b_digits']},"
            f"NUM_EXPAND_ROUNDS={entry['num_expand_rounds']},"
            f"FRONTIER_MIN_A_DIGITS={entry['frontier_min_a_digits']},"
            f"FRONTIER_MAX_A_DIGITS={entry['frontier_max_a_digits']},"
            f"FRONTIER_MIN_B_DIGITS={entry['frontier_min_b_digits']},"
            f"LEARNING_RATE={entry['learning_rate']},"
            f"MAX_STEPS={entry['max_steps']},"
            "HELDOUT_PER_PARTITION=200,"
            "SEED=0,"
            f"DRY_RUN={1 if dry_run else 0}"
        ),
        str(SELF_IMPROVEMENT_LAUNCHER),
    ]
    result = _run_command(cmd, dry_run=dry_run)
    return None if result is None else result.split(";")[0].strip()


def submit_stage1_jobs(
    manifest: Mapping[str, Any],
    *,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    seed_model = Path(str(manifest["meta"]["seed_model"]))
    submitted: List[Dict[str, Any]] = []
    for entry in manifest["stage1_jobs"]:
        payload = dict(entry)
        payload["job_id"] = _submit_rectangular_job(entry=payload, seed_model=seed_model, dry_run=dry_run)
        submitted.append(payload)
    return submitted


def _submit_selector_job(
    *,
    job_name: str,
    wrap_cmd: Sequence[str],
    log_dir: Path,
    dependency_job_ids: Sequence[str],
    dry_run: bool,
) -> Optional[str]:
    stdout_log, stderr_log = _selector_log_paths(log_dir, job_name)
    cmd = [
        "sbatch",
        "--parsable",
        f"--cpus-per-task={SELECTOR_JOB_CONFIG['cpus_per_task']}",
        f"--mem={SELECTOR_JOB_CONFIG['mem']}",
        f"--time={SELECTOR_JOB_CONFIG['time']}",
        f"--job-name={job_name}",
        f"--output={stdout_log}",
        f"--error={stderr_log}",
    ]
    if dependency_job_ids:
        cmd.append(f"--dependency=afterany:{':'.join(dependency_job_ids)}")
    cmd.append(f"--wrap={shlex.join(list(wrap_cmd))}")
    result = _run_command(cmd, dry_run=dry_run)
    return None if result is None else result.split(";")[0].strip()


def _required_file(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Missing required result file: {candidate}")
    return candidate


def _load_results(path: str | Path) -> Dict[str, Any]:
    return json.loads(_required_file(path).read_text(encoding="utf-8"))


def _final_round_payload(results_path: str | Path) -> Dict[str, Any]:
    payload = _load_results(results_path)
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError(f"No rounds found in results payload: {results_path}")
    return rounds[-1]


def _mean_accuracy_for_rows(summary: Mapping[str, Any], rows: Sequence[int]) -> float:
    grouped = summary.get("mean_accuracy_by_a_digits") or {}
    values = [
        float(grouped[str(row)])
        for row in rows
        if str(row) in grouped
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_stage1_compose_candidate(entry: Mapping[str, Any]) -> Dict[str, Any]:
    final_round = _final_round_payload(entry["results_path"])
    results = final_round["results"]
    seed_min = float(results["seed_test"]["min_partition_accuracy"])
    frontier_train_accuracy = float(results["frontier_train"]["accuracy"])
    frontier_test_accuracy = float(results["frontier_test"]["accuracy"])
    row_234_mean = _mean_accuracy_for_rows(results["frontier_test"], rows=(2, 3, 4))
    paper_viable = (
        seed_min >= 0.95
        and frontier_train_accuracy >= 0.12
        and frontier_test_accuracy >= 0.12
        and row_234_mean >= 0.05
    )
    return {
        **dict(entry),
        "seed_test_min_partition_accuracy": seed_min,
        "frontier_train_accuracy": frontier_train_accuracy,
        "frontier_test_accuracy": frontier_test_accuracy,
        "frontier_test_mean_accuracy_rows_2_4": row_234_mean,
        "paper_viable": paper_viable,
        "final_round": final_round["round"],
        "final_max_b_digits": final_round["max_b_digits"],
    }


def choose_stage1_top2(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    viable = [dict(candidate) for candidate in candidates if candidate["paper_viable"]]
    if viable:
        pool = viable
        selection_mode = "paper_viable"
    else:
        pool = [
            dict(candidate)
            for candidate in candidates
            if float(candidate["seed_test_min_partition_accuracy"]) >= 0.90
        ]
        selection_mode = "seed_min_fallback"
    if not pool:
        raise ValueError("No multiplication stage-1 schedules met the selection floor.")
    ranked = sorted(
        pool,
        key=lambda candidate: (
            float(candidate["frontier_test_mean_accuracy_rows_2_4"]),
            float(candidate["frontier_test_accuracy"]),
            float(candidate["seed_test_min_partition_accuracy"]),
        ),
        reverse=True,
    )
    selected = ranked[:2]
    for candidate in selected:
        candidate["selection_mode"] = selection_mode
    return selected


def build_stage2_followups(selected_stage1: Sequence[Mapping[str, Any]], *, out_root: Path) -> List[Dict[str, Any]]:
    followups: List[Dict[str, Any]] = []
    for entry in selected_stage1:
        output_root = out_root / "stage2" / entry["schedule_label"]
        followups.append(
            {
                **dict(entry),
                "stage": 2,
                "baseline": STAGE2_BASELINE,
                "output_root": str(output_root),
                "results_path": str(output_root / STAGE2_BASELINE / "self_improvement_results.json"),
            }
        )
    return followups


def score_stage2_candidate(
    compose_candidate: Mapping[str, Any],
    *,
    direct_results_path: str | Path,
) -> Dict[str, Any]:
    final_round = _final_round_payload(direct_results_path)
    results = final_round["results"]
    direct_frontier_test_accuracy = float(results["frontier_test"]["accuracy"])
    compose_frontier_test_accuracy = float(compose_candidate["frontier_test_accuracy"])
    return {
        **dict(compose_candidate),
        "direct_results_path": str(direct_results_path),
        "direct_frontier_test_accuracy": direct_frontier_test_accuracy,
        "compose_minus_direct_frontier_test_accuracy": (
            compose_frontier_test_accuracy - direct_frontier_test_accuracy
        ),
    }


def choose_final_stage2_candidate(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            float(candidate["compose_minus_direct_frontier_test_accuracy"]),
            float(candidate["frontier_test_mean_accuracy_rows_2_4"]),
            float(candidate["seed_test_min_partition_accuracy"]),
        ),
        reverse=True,
    )
    if not ranked:
        raise ValueError("No stage-2 multiplication candidates were available for final selection.")
    return ranked[0]


def build_stage3_followups(selected_final: Mapping[str, Any], *, out_root: Path) -> List[Dict[str, Any]]:
    followups: List[Dict[str, Any]] = []
    for baseline in STAGE3_BASELINES:
        output_root = out_root / "stage3" / selected_final["schedule_label"]
        followups.append(
            {
                **dict(selected_final),
                "stage": 3,
                "baseline": baseline,
                "output_root": str(output_root),
                "results_path": str(output_root / baseline / "self_improvement_results.json"),
            }
        )
    return followups


def finalize_stage3_selection(
    *,
    stage2_selection: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_final = dict(stage2_selection["selected_final_schedule"])
    compose_results_path = selected_final["results_path"]
    direct_results_path = selected_final["direct_results_path"]
    baseline_results = {
        "compose": compose_results_path,
        "direct": direct_results_path,
    }
    for followup in stage2_selection["stage3_followups"]:
        baseline_results[followup["baseline"]] = followup["results_path"]
        _required_file(followup["results_path"])

    compose_final_round = _final_round_payload(compose_results_path)
    compose_results = compose_final_round["results"]
    paper_ready = (
        float(compose_results["frontier_test"]["accuracy"]) >= 0.12
        and float(compose_results["seed_test"]["min_partition_accuracy"]) >= 0.95
    )
    return {
        "selected_final_schedule": {
            **selected_final,
            "baseline_results": baseline_results,
            "paper_ready": paper_ready,
            "stop_condition_passed": paper_ready,
        },
        "stage2_candidates": stage2_selection["stage2_candidates"],
        "stage3_followups": stage2_selection["stage3_followups"],
    }


def _parse_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed-model", type=Path, default=DEFAULT_SEED_MODEL)
    parser.add_argument("--stage1-manifest", type=Path, default=None)
    parser.add_argument("--stage1-selection", type=Path, default=None)
    parser.add_argument("--stage2-selection", type=Path, default=None)
    parser.add_argument("--stage3-selection", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rectangular multiplication tuning sweep orchestration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Write manifest, submit stage-1 jobs, and queue stage-2 selector.")
    _parse_common_args(submit_parser)

    stage2_parser = subparsers.add_parser("stage2", help="Select stage-1 winners and submit stage-2 direct followups.")
    _parse_common_args(stage2_parser)

    stage3_parser = subparsers.add_parser("stage3", help="Select final schedule and submit stage-3 backfill baselines.")
    _parse_common_args(stage3_parser)

    finalize_parser = subparsers.add_parser("finalize", help="Verify backfills and write the final stage-3 summary.")
    _parse_common_args(finalize_parser)

    return parser


def command_submit(args: argparse.Namespace) -> None:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.stage1_manifest or _default_stage1_manifest(out_root)
    stage1_selection_path = args.stage1_selection or _default_stage1_selection(out_root)
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(out_root)
    stage3_selection_path = args.stage3_selection or _default_stage3_selection(out_root)

    manifest = build_stage1_manifest(
        out_root=out_root,
        log_dir=log_dir,
        python_bin=args.python_bin,
        seed_model=args.seed_model,
    )
    stage1_jobs = submit_stage1_jobs(manifest, dry_run=args.dry_run)
    manifest["stage1_jobs"] = stage1_jobs
    _json_dump(manifest_path, manifest)
    print(f"[INFO] Wrote stage-1 manifest to {manifest_path}", flush=True)
    print(f"[INFO] Stage-1 counts: compose={len(stage1_jobs)}", flush=True)

    dependency_job_ids = [job["job_id"] for job in stage1_jobs if job.get("job_id")]
    stage2_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.multiplication_rectangular_tune",
        "stage2",
        "--out-root",
        str(out_root),
        "--log-dir",
        str(log_dir),
        "--seed-model",
        str(args.seed_model),
        "--python-bin",
        args.python_bin,
        "--stage1-manifest",
        str(manifest_path),
        "--stage1-selection",
        str(stage1_selection_path),
        "--stage2-selection",
        str(stage2_selection_path),
        "--stage3-selection",
        str(stage3_selection_path),
    ]
    if args.dry_run:
        print("[INFO] Stage-2 selector job name: mult-rect-tune-stage2", flush=True)
        print("[INFO] DRY_RUN=1; stage-2 selector job not submitted.", flush=True)
        return
    stage2_job_id = _submit_selector_job(
        job_name="mult-rect-tune-stage2",
        wrap_cmd=stage2_cmd,
        log_dir=log_dir,
        dependency_job_ids=dependency_job_ids,
        dry_run=False,
    )
    print(f"[INFO] Submitted stage-2 selector job_id={stage2_job_id}", flush=True)


def command_stage2(args: argparse.Namespace) -> None:
    out_root = args.out_root
    log_dir = args.log_dir
    manifest_path = args.stage1_manifest or _default_stage1_manifest(out_root)
    stage1_selection_path = args.stage1_selection or _default_stage1_selection(out_root)
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(out_root)
    stage3_selection_path = args.stage3_selection or _default_stage3_selection(out_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stage1_candidates = [
        score_stage1_compose_candidate(entry)
        for entry in manifest["stage1_jobs"]
    ]
    selected_stage1 = choose_stage1_top2(stage1_candidates)
    stage2_followups = build_stage2_followups(selected_stage1, out_root=out_root)
    stage1_selection = {
        "stage1_candidates": stage1_candidates,
        "selected_stage1_top2": selected_stage1,
        "stage2_followups": stage2_followups,
    }
    _json_dump(stage1_selection_path, stage1_selection)
    print(f"[INFO] Wrote stage-1 selection to {stage1_selection_path}", flush=True)

    seed_model = args.seed_model
    submitted_followups: List[Dict[str, Any]] = []
    for entry in stage2_followups:
        payload = dict(entry)
        payload["job_id"] = _submit_rectangular_job(entry=payload, seed_model=seed_model, dry_run=args.dry_run)
        submitted_followups.append(payload)
    stage1_selection["stage2_followups"] = submitted_followups
    _json_dump(stage1_selection_path, stage1_selection)
    print(f"[INFO] Stage-2 counts: direct={len(submitted_followups)}", flush=True)

    dependency_job_ids = [entry["job_id"] for entry in submitted_followups if entry.get("job_id")]
    stage3_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.multiplication_rectangular_tune",
        "stage3",
        "--out-root",
        str(out_root),
        "--log-dir",
        str(log_dir),
        "--seed-model",
        str(args.seed_model),
        "--python-bin",
        args.python_bin,
        "--stage1-manifest",
        str(manifest_path),
        "--stage1-selection",
        str(stage1_selection_path),
        "--stage2-selection",
        str(stage2_selection_path),
        "--stage3-selection",
        str(stage3_selection_path),
    ]
    if args.dry_run:
        print("[INFO] Stage-3 selector job name: mult-rect-tune-stage3", flush=True)
        print("[INFO] DRY_RUN=1; stage-3 selector job not submitted.", flush=True)
        return
    stage3_job_id = _submit_selector_job(
        job_name="mult-rect-tune-stage3",
        wrap_cmd=stage3_cmd,
        log_dir=log_dir,
        dependency_job_ids=dependency_job_ids,
        dry_run=False,
    )
    print(f"[INFO] Submitted stage-3 selector job_id={stage3_job_id}", flush=True)


def command_stage3(args: argparse.Namespace) -> None:
    out_root = args.out_root
    log_dir = args.log_dir
    stage1_selection_path = args.stage1_selection or _default_stage1_selection(out_root)
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(out_root)
    stage3_selection_path = args.stage3_selection or _default_stage3_selection(out_root)
    stage1_selection = json.loads(stage1_selection_path.read_text(encoding="utf-8"))

    stage2_candidates = [
        score_stage2_candidate(candidate, direct_results_path=followup["results_path"])
        for candidate, followup in zip(stage1_selection["selected_stage1_top2"], stage1_selection["stage2_followups"], strict=True)
    ]
    selected_final = choose_final_stage2_candidate(stage2_candidates)
    stage3_followups = build_stage3_followups(selected_final, out_root=out_root)
    stage2_selection = {
        "stage2_candidates": stage2_candidates,
        "selected_final_schedule": selected_final,
        "stage3_followups": stage3_followups,
    }
    _json_dump(stage2_selection_path, stage2_selection)
    print(f"[INFO] Wrote stage-2 selection to {stage2_selection_path}", flush=True)

    seed_model = args.seed_model
    submitted_followups: List[Dict[str, Any]] = []
    for entry in stage3_followups:
        payload = dict(entry)
        payload["job_id"] = _submit_rectangular_job(entry=payload, seed_model=seed_model, dry_run=args.dry_run)
        submitted_followups.append(payload)
    stage2_selection["stage3_followups"] = submitted_followups
    _json_dump(stage2_selection_path, stage2_selection)
    print(f"[INFO] Stage-3 counts: backfill={len(submitted_followups)}", flush=True)

    dependency_job_ids = [entry["job_id"] for entry in submitted_followups if entry.get("job_id")]
    finalize_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.multiplication_rectangular_tune",
        "finalize",
        "--out-root",
        str(out_root),
        "--log-dir",
        str(log_dir),
        "--seed-model",
        str(args.seed_model),
        "--python-bin",
        args.python_bin,
        "--stage1-selection",
        str(stage1_selection_path),
        "--stage2-selection",
        str(stage2_selection_path),
        "--stage3-selection",
        str(stage3_selection_path),
    ]
    if args.dry_run:
        print("[INFO] Finalize job name: mult-rect-tune-finalize", flush=True)
        print("[INFO] DRY_RUN=1; finalize job not submitted.", flush=True)
        return
    finalize_job_id = _submit_selector_job(
        job_name="mult-rect-tune-finalize",
        wrap_cmd=finalize_cmd,
        log_dir=log_dir,
        dependency_job_ids=dependency_job_ids,
        dry_run=False,
    )
    print(f"[INFO] Submitted finalize job_id={finalize_job_id}", flush=True)


def command_finalize(args: argparse.Namespace) -> None:
    out_root = args.out_root
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(out_root)
    stage3_selection_path = args.stage3_selection or _default_stage3_selection(out_root)
    stage2_selection = json.loads(stage2_selection_path.read_text(encoding="utf-8"))
    payload = finalize_stage3_selection(stage2_selection=stage2_selection)
    _json_dump(stage3_selection_path, payload)
    print(f"[INFO] Wrote stage-3 selection to {stage3_selection_path}", flush=True)
    print(
        json.dumps(
            {
                "selected_schedule": payload["selected_final_schedule"]["schedule_label"],
                "paper_ready": payload["selected_final_schedule"]["paper_ready"],
            }
        ),
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        command_submit(args)
    elif args.command == "stage2":
        command_stage2(args)
    elif args.command == "stage3":
        command_stage3(args)
    elif args.command == "finalize":
        command_finalize(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
