#!/usr/bin/env python3
"""Submit and collect MIG-first Figure 2 condition sweeps."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.experiments.figure2_paper_retune import render_paper_heatmap, write_schedule_env
from self.experiments.paper_schedule_selection import (
    choose_addition_fullpack_candidate,
    choose_addition_stage1_topk,
    choose_run_length_candidate,
    score_addition_fullpack_candidate,
    score_addition_stage1_schedule,
    score_run_length_candidate,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION_JSON = ROOT_DIR / "artifacts/paper/paper_schedule_selection.json"
DEFAULT_PAPER_SCHEDULE_ENV = ROOT_DIR / "artifacts/paper/paper_schedule_selection.env"
DEFAULT_FIGURE_DIR = ROOT_DIR / "icmlw26_comp-self-improvement/figures"
DEFAULT_LOG_DIR = ROOT_DIR / "artifacts/logs"
DEFAULT_RUN_LENGTH_SEED_MODEL = ROOT_DIR / "artifacts/models/run_length_recipe_seed_best"
DEFAULT_ADDITION_SEED_MODEL = ROOT_DIR / "artifacts/models/addition_recipe_seed_best"

MIG_PARTITION = "mig"
MIG_GRES = "gpu:1g.10gb:1"
MIG_MEM = "64G"
MIG_CPUS = "1"
MIG_TIME = "48:00:00"

BIT_STAGE1_CONFIGS = {
    "run_length": {
        "expand_num_bits": (4, 3),
        "expand_train_per_bit": (1200, 2400),
        "train_batch_size": 128,
        "eval_batch_size": 128,
    },
}
BIT_DEFAULTS = {
    "num_expand_rounds": 8,
    "initial_train_per_bit": 100000,
    "initial_eval_per_bit": 50,
    "eval_per_bit": 50,
    "composed_eval_per_bit": 50,
}
ADDITION_STAGE1_BASELINES = ("direct", "with_carry", "with_carry_filtered")
ADDITION_STAGE2_BASELINES = ("short_only", "compose_corrupt")
ADDITION_STAGE1_DIGITS = (2, 3, 4)
ADDITION_NUM_EXPAND_ROUNDS = 8
ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT = 5000
ADDITION_EXPAND_TRAIN_PER_DIGIT = 10000
ADDITION_TRAIN_BATCH_SIZE = 256
ADDITION_EVAL_BATCH_SIZE = 256


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _required_file(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Missing required result file: {candidate}")
    return candidate


def _slurm_config() -> Dict[str, str]:
    return {
        "partition": MIG_PARTITION,
        "gres": MIG_GRES,
        "mem": MIG_MEM,
        "cpus_per_task": MIG_CPUS,
        "time": MIG_TIME,
    }


def _log_paths(log_dir: Path, job_name: str) -> tuple[Path, Path]:
    safe_name = job_name.replace("/", "-")
    return (
        log_dir / f"{safe_name}-%j.out",
        log_dir / f"{safe_name}-%j.err",
    )


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


def _submit_sbatch_job(
    *,
    job_name: str,
    wrap_cmd: Sequence[str],
    log_dir: Path,
    dependency_job_ids: Sequence[str] = (),
    dry_run: bool,
) -> Optional[str]:
    stdout_log, stderr_log = _log_paths(log_dir, job_name)
    slurm = _slurm_config()
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={slurm['partition']}",
        f"--gres={slurm['gres']}",
        f"--mem={slurm['mem']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--time={slurm['time']}",
        f"--job-name={job_name}",
        f"--output={stdout_log}",
        f"--error={stderr_log}",
    ]
    if dependency_job_ids:
        sbatch_cmd.append(f"--dependency=afterany:{':'.join(dependency_job_ids)}")
    sbatch_cmd.append(f"--wrap={shlex.join(list(wrap_cmd))}")
    result = _run_command(sbatch_cmd, dry_run=dry_run)
    return None if result is None else result.split(";")[0].strip()


def _bit_candidate_output_root(out_root: Path, task: str, expand_num_bits: int, expand_train_per_bit: int) -> Path:
    return out_root / "stage1" / task / f"expand{expand_num_bits}_train{expand_train_per_bit}"


def _addition_schedule_root(out_root: Path, expand_num_digits: int) -> Path:
    return (
        out_root
        / "stage1"
        / "addition"
        / (
            f"expand{expand_num_digits}_replay{ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT}"
            f"_train{ADDITION_EXPAND_TRAIN_PER_DIGIT}"
        )
    )


def build_stage1_manifest(
    *,
    out_root: Path,
    figure_dir: Path,
    selection_json: Path,
    paper_schedule_env: Path,
    python_bin: str,
    run_length_seed_model: Path,
    addition_seed_model: Path,
) -> Dict[str, Any]:
    stage1_jobs: List[Dict[str, Any]] = []

    for task, config in BIT_STAGE1_CONFIGS.items():
        for expand_num_bits in config["expand_num_bits"]:
            for expand_train_per_bit in config["expand_train_per_bit"]:
                candidate_root = _bit_candidate_output_root(
                    out_root,
                    task,
                    expand_num_bits,
                    expand_train_per_bit,
                )
                stage1_jobs.append(
                    {
                        "kind": "run_length_bit",
                        "task": task,
                        "baseline": "compose",
                        "schedule_label": f"expand{expand_num_bits}_train{expand_train_per_bit}",
                        "output_root": str(candidate_root),
                        "results_path": str(candidate_root / task / "pilot" / "compose" / "self_improvement_results.json"),
                        "expand_num_bits": expand_num_bits,
                        "expand_train_per_bit": expand_train_per_bit,
                        "num_expand_rounds": BIT_DEFAULTS["num_expand_rounds"],
                        "initial_train_per_bit": BIT_DEFAULTS["initial_train_per_bit"],
                        "initial_eval_per_bit": BIT_DEFAULTS["initial_eval_per_bit"],
                        "eval_per_bit": BIT_DEFAULTS["eval_per_bit"],
                        "composed_eval_per_bit": BIT_DEFAULTS["composed_eval_per_bit"],
                        "train_batch_size": config["train_batch_size"],
                        "eval_batch_size": config["eval_batch_size"],
                    }
                )

    for expand_num_digits in ADDITION_STAGE1_DIGITS:
        schedule_root = _addition_schedule_root(out_root, expand_num_digits)
        for baseline in ADDITION_STAGE1_BASELINES:
            stage1_jobs.append(
                {
                    "kind": "addition_stage1",
                    "task": "addition",
                    "baseline": baseline,
                    "schedule_label": (
                        f"expand{expand_num_digits}_replay{ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT}"
                        f"_train{ADDITION_EXPAND_TRAIN_PER_DIGIT}"
                    ),
                    "output_root": str(schedule_root),
                    "results_path": str(schedule_root / baseline / "self_improvement_results.json"),
                    "expand_num_digits": expand_num_digits,
                    "seed_replay_train_per_digit": ADDITION_SEED_REPLAY_TRAIN_PER_DIGIT,
                    "expand_train_per_digit": ADDITION_EXPAND_TRAIN_PER_DIGIT,
                    "num_expand_rounds": ADDITION_NUM_EXPAND_ROUNDS,
                    "train_batch_size": ADDITION_TRAIN_BATCH_SIZE,
                    "eval_batch_size": ADDITION_EVAL_BATCH_SIZE,
                }
            )

    return {
        "meta": {
            "out_root": str(out_root),
            "figure_dir": str(figure_dir),
            "selection_json": str(selection_json),
            "paper_schedule_env": str(paper_schedule_env),
            "python_bin": python_bin,
            "run_length_seed_model": str(run_length_seed_model),
            "addition_seed_model": str(addition_seed_model),
            "slurm": _slurm_config(),
        },
        "stage1_jobs": stage1_jobs,
    }


def _bit_stage1_wrap_cmd(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    task = entry["task"]
    env_items = [
        "env",
        f"PYTHON_BIN={python_bin}",
        f"OUT_ROOT={entry['output_root']}",
        "STAGE=pilot",
        f"TASKS={task}",
        f"TRAIN_BATCH_SIZE={entry['train_batch_size']}",
        f"EVAL_BATCH_SIZE={entry['eval_batch_size']}",
    ]
    env_items += [
        f"RUN_LENGTH_NUM_EXPAND_ROUNDS={entry['num_expand_rounds']}",
        f"RUN_LENGTH_EXPAND_NUM_BITS={entry['expand_num_bits']}",
        f"RUN_LENGTH_EXPAND_TRAIN_PER_BIT={entry['expand_train_per_bit']}",
    ]
    env_items += ["bash", str(ROOT_DIR / "launchers/self/run_figure2_recipe_aggressive.sh")]
    return env_items


def _addition_wrap_cmd(
    entry: Mapping[str, Any],
    *,
    python_bin: str,
    addition_seed_model: str,
) -> List[str]:
    return [
        "env",
        f"PYTHON_BIN={python_bin}",
        f"OUT_ROOT={entry['output_root']}",
        f"SEED_MODEL={addition_seed_model}",
        f"BASELINE={entry['baseline']}",
        f"TRAIN_BATCH_SIZE={entry['train_batch_size']}",
        f"EVAL_BATCH_SIZE={entry['eval_batch_size']}",
        f"NUM_EXPAND_ROUNDS={entry['num_expand_rounds']}",
        f"EXPAND_NUM_DIGITS={entry['expand_num_digits']}",
        f"SEED_REPLAY_TRAIN_PER_DIGIT={entry['seed_replay_train_per_digit']}",
        f"EXPAND_TRAIN_PER_DIGIT={entry['expand_train_per_digit']}",
        "bash",
        str(ROOT_DIR / "launchers/self/run_addition_recipe_focused.sh"),
    ]


def submit_stage1_jobs(
    manifest: Mapping[str, Any],
    *,
    log_dir: Path,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(manifest["meta"]["python_bin"])
    addition_seed_model = str(manifest["meta"]["addition_seed_model"])
    submitted: List[Dict[str, Any]] = []
    for entry in manifest["stage1_jobs"]:
        if entry["kind"] == "run_length_bit":
            wrap_cmd = _bit_stage1_wrap_cmd(entry, python_bin=python_bin)
            job_name = f"fig2-{entry['task']}-{entry['schedule_label']}"
        else:
            wrap_cmd = _addition_wrap_cmd(
                entry,
                python_bin=python_bin,
                addition_seed_model=addition_seed_model,
            )
            job_name = f"fig2-add-{entry['baseline']}-{entry['schedule_label']}"
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=wrap_cmd,
            log_dir=log_dir,
            dry_run=dry_run,
        )
        submitted.append(
            {
                **dict(entry),
                "job_name": job_name,
                "job_id": job_id,
            }
        )
        if job_id is not None:
            print(
                f"[INFO] Submitted job_name={job_name} job_id={job_id} "
                f"results_path={entry['results_path']}",
                flush=True,
            )
    return submitted


def _group_stage1_jobs(entries: Sequence[Mapping[str, Any]], *, task: str) -> List[Mapping[str, Any]]:
    return [entry for entry in entries if entry["task"] == task]


def _load_stage1_job_results(entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        _required_file(entry["results_path"])


def select_stage1_candidates(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    stage1_entries = manifest["stage1_jobs"]
    _load_stage1_job_results(stage1_entries)

    run_length_scores = [
        {
            **score_run_length_candidate(
                entry["results_path"],
                expand_num_bits=int(entry["expand_num_bits"]),
            ),
            "expand_train_per_bit": int(entry["expand_train_per_bit"]),
            "output_root": entry["output_root"],
        }
        for entry in _group_stage1_jobs(stage1_entries, task="run_length")
    ]
    selected_run_length = choose_run_length_candidate(run_length_scores)

    addition_entries = _group_stage1_jobs(stage1_entries, task="addition")
    addition_by_schedule: Dict[str, Dict[str, Any]] = {}
    for entry in addition_entries:
        schedule = addition_by_schedule.setdefault(
            entry["schedule_label"],
            {
                "output_root": entry["output_root"],
                "expand_num_digits": int(entry["expand_num_digits"]),
                "seed_replay_train_per_digit": int(entry["seed_replay_train_per_digit"]),
                "expand_train_per_digit": int(entry["expand_train_per_digit"]),
                "baseline_results": {},
            },
        )
        schedule["baseline_results"][entry["baseline"]] = entry["results_path"]

    addition_stage1_scores = [
        {
            **score_addition_stage1_schedule(
                schedule["baseline_results"],
                expand_num_digits=int(schedule["expand_num_digits"]),
                seed_replay_train_per_digit=int(schedule["seed_replay_train_per_digit"]),
                expand_train_per_digit=int(schedule["expand_train_per_digit"]),
            ),
            "schedule_label": schedule_label,
            "output_root": schedule["output_root"],
        }
        for schedule_label, schedule in sorted(addition_by_schedule.items())
    ]
    selected_addition_topk = choose_addition_stage1_topk(addition_stage1_scores, k=2)

    return {
        "run_length_candidates": run_length_scores,
        "addition_stage1_candidates": addition_stage1_scores,
        "selected_run_length": selected_run_length,
        "selected_addition_topk": selected_addition_topk,
    }


def build_addition_stage2_followups(
    selected_addition_topk: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    followups: List[Dict[str, Any]] = []
    for candidate in selected_addition_topk:
        for baseline in ADDITION_STAGE2_BASELINES:
            followups.append(
                {
                    "task": "addition",
                    "baseline": baseline,
                    "schedule_label": candidate["schedule_label"],
                    "output_root": candidate["output_root"],
                    "results_path": str(Path(candidate["output_root"]) / baseline / "self_improvement_results.json"),
                    "expand_num_digits": int(candidate["expand_num_digits"]),
                    "seed_replay_train_per_digit": int(candidate["seed_replay_train_per_digit"]),
                    "expand_train_per_digit": int(candidate["expand_train_per_digit"]),
                    "num_expand_rounds": ADDITION_NUM_EXPAND_ROUNDS,
                    "train_batch_size": ADDITION_TRAIN_BATCH_SIZE,
                    "eval_batch_size": ADDITION_EVAL_BATCH_SIZE,
                }
            )
    return followups


def finalize_selection_payload(
    *,
    stage2_selection: Mapping[str, Any],
    run_length_seed_model: str,
    addition_seed_model: str,
) -> Dict[str, Any]:
    addition_fullpack_candidates = []
    for selected in stage2_selection["selected_addition_topk"]:
        baseline_results = dict(selected["baseline_results"])
        for followup in stage2_selection["addition_stage2_followups"]:
            if followup["schedule_label"] == selected["schedule_label"]:
                baseline_results[followup["baseline"]] = followup["results_path"]
        for path in baseline_results.values():
            _required_file(path)
        addition_fullpack_candidates.append(
            {
                **score_addition_fullpack_candidate(
                    baseline_results,
                    expand_num_digits=int(selected["expand_num_digits"]),
                    seed_replay_train_per_digit=int(selected["seed_replay_train_per_digit"]),
                    expand_train_per_digit=int(selected["expand_train_per_digit"]),
                ),
                "schedule_label": selected["schedule_label"],
                "output_root": selected["output_root"],
            }
        )

    selected_addition = choose_addition_fullpack_candidate(addition_fullpack_candidates)
    selected_run_length = dict(stage2_selection["selected_run_length"])

    return {
        "selected_schedules": {
            "run_length": {
                "seed_model": run_length_seed_model,
                "results_path": selected_run_length["results_path"],
                "num_expand_rounds": BIT_DEFAULTS["num_expand_rounds"],
                "expand_num_bits": int(selected_run_length["expand_num_bits"]),
                "expand_train_per_bit": int(selected_run_length["expand_train_per_bit"]),
            },
            "addition": {
                "seed_model": addition_seed_model,
                "results_path": selected_addition["baseline_results"]["with_carry_filtered"],
                "num_expand_rounds": ADDITION_NUM_EXPAND_ROUNDS,
                "expand_num_digits": int(selected_addition["expand_num_digits"]),
                "seed_replay_train_per_digit": int(selected_addition["seed_replay_train_per_digit"]),
                "expand_train_per_digit": int(selected_addition["expand_train_per_digit"]),
            },
        },
        "run_length_candidates": stage2_selection["run_length_candidates"],
        "addition_stage1_candidates": stage2_selection["addition_stage1_candidates"],
        "addition_stage2_fullpack_candidates": addition_fullpack_candidates,
    }


def _parse_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, default=None)
    parser.add_argument("--stage2-selection", type=Path, default=None)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    parser.add_argument("--paper-schedule-env", type=Path, default=DEFAULT_PAPER_SCHEDULE_ENV)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--run-length-seed-model", type=Path, default=DEFAULT_RUN_LENGTH_SEED_MODEL)
    parser.add_argument("--addition-seed-model", type=Path, default=DEFAULT_ADDITION_SEED_MODEL)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit and collect Figure 2 MIG condition sweeps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Write manifest, submit stage-1 jobs, and queue stage-2 selector.")
    _parse_common_args(submit_parser)

    stage2_parser = subparsers.add_parser("stage2", help="Select stage-1 winners and submit addition follow-up baselines.")
    _parse_common_args(stage2_parser)

    finalize_parser = subparsers.add_parser("finalize", help="Build official paper selection files and refresh figure bundles.")
    _parse_common_args(finalize_parser)

    return parser


def _default_stage1_manifest(out_root: Path) -> Path:
    return out_root / "stage1_manifest.json"


def _default_stage2_selection(out_root: Path) -> Path:
    return out_root / "stage2_selection.json"


def command_submit(args: argparse.Namespace) -> None:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.stage1_manifest or _default_stage1_manifest(out_root)
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(out_root)

    manifest = build_stage1_manifest(
        out_root=out_root,
        figure_dir=args.figure_dir,
        selection_json=args.selection_json,
        paper_schedule_env=args.paper_schedule_env,
        python_bin=args.python_bin,
        run_length_seed_model=args.run_length_seed_model,
        addition_seed_model=args.addition_seed_model,
    )
    _json_dump(manifest_path, manifest)
    print(f"[INFO] Wrote stage-1 manifest to {manifest_path}", flush=True)

    stage1_jobs = submit_stage1_jobs(manifest, log_dir=args.log_dir, dry_run=args.dry_run)
    run_length_count = sum(1 for job in stage1_jobs if job["task"] == "run_length")
    addition_count = sum(1 for job in stage1_jobs if job["task"] == "addition")
    print(
        f"[INFO] Stage-1 counts: run_length={run_length_count} addition={addition_count}",
        flush=True,
    )

    dependency_job_ids = [job["job_id"] for job in stage1_jobs if job.get("job_id")]
    stage2_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.figure2_condition_sweep",
        "stage2",
        "--out-root",
        str(out_root),
        "--stage1-manifest",
        str(manifest_path),
        "--stage2-selection",
        str(stage2_selection_path),
        "--selection-json",
        str(args.selection_json),
        "--paper-schedule-env",
        str(args.paper_schedule_env),
        "--figure-dir",
        str(args.figure_dir),
        "--log-dir",
        str(args.log_dir),
        "--python-bin",
        args.python_bin,
        "--run-length-seed-model",
        str(args.run_length_seed_model),
        "--addition-seed-model",
        str(args.addition_seed_model),
    ]
    stage2_job_id = _submit_sbatch_job(
        job_name="fig2-sweep-stage2",
        wrap_cmd=stage2_cmd,
        log_dir=args.log_dir,
        dependency_job_ids=dependency_job_ids,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=1; stage-2 selector job not submitted.", flush=True)
        return
    print(f"[INFO] Submitted stage-2 selector job_id={stage2_job_id}", flush=True)


def command_stage2(args: argparse.Namespace) -> None:
    manifest_path = args.stage1_manifest or _default_stage1_manifest(args.out_root)
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(args.out_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = select_stage1_candidates(manifest)
    followups = build_addition_stage2_followups(selection["selected_addition_topk"])

    stage2_selection = {
        **selection,
        "addition_stage2_followups": followups,
    }
    _json_dump(stage2_selection_path, stage2_selection)
    print(f"[INFO] Wrote stage-2 selection to {stage2_selection_path}", flush=True)

    followup_job_ids: List[str] = []
    addition_seed_model = str(args.addition_seed_model)
    for entry in followups:
        wrap_cmd = _addition_wrap_cmd(
            entry,
            python_bin=args.python_bin,
            addition_seed_model=addition_seed_model,
        )
        job_name = f"fig2-add-{entry['baseline']}-{entry['schedule_label']}"
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=wrap_cmd,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
        )
        if job_id is not None:
            followup_job_ids.append(job_id)
            print(
                f"[INFO] Submitted addition follow-up job_name={job_name} job_id={job_id} "
                f"results_path={entry['results_path']}",
                flush=True,
            )

    finalize_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.figure2_condition_sweep",
        "finalize",
        "--out-root",
        str(args.out_root),
        "--stage1-manifest",
        str(manifest_path),
        "--stage2-selection",
        str(stage2_selection_path),
        "--selection-json",
        str(args.selection_json),
        "--paper-schedule-env",
        str(args.paper_schedule_env),
        "--figure-dir",
        str(args.figure_dir),
        "--log-dir",
        str(args.log_dir),
        "--python-bin",
        args.python_bin,
        "--run-length-seed-model",
        str(args.run_length_seed_model),
        "--addition-seed-model",
        str(args.addition_seed_model),
    ]
    finalize_job_id = _submit_sbatch_job(
        job_name="fig2-sweep-finalize",
        wrap_cmd=finalize_cmd,
        log_dir=args.log_dir,
        dependency_job_ids=followup_job_ids,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=1; follow-up jobs and final collector not submitted.", flush=True)
        return
    print(f"[INFO] Submitted final collector job_id={finalize_job_id}", flush=True)


def command_finalize(args: argparse.Namespace) -> None:
    stage2_selection_path = args.stage2_selection or _default_stage2_selection(args.out_root)
    stage2_selection = json.loads(stage2_selection_path.read_text(encoding="utf-8"))
    final_payload = finalize_selection_payload(
        stage2_selection=stage2_selection,
        run_length_seed_model=str(args.run_length_seed_model),
        addition_seed_model=str(args.addition_seed_model),
    )
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.paper_schedule_env.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    write_schedule_env(args.paper_schedule_env, final_payload["selected_schedules"])
    print(f"[INFO] Wrote selection JSON to {args.selection_json}", flush=True)
    print(f"[INFO] Wrote schedule env to {args.paper_schedule_env}", flush=True)

    render_paper_heatmap(
        Path(final_payload["selected_schedules"]["run_length"]["results_path"]),
        task="run_length",
        mode="compose",
        basename="run_length_self_improvement_heatmap",
        figure_dir=args.figure_dir,
    )
    render_paper_heatmap(
        Path(final_payload["selected_schedules"]["addition"]["results_path"]),
        task="addition",
        mode="with_carry_filtered",
        basename="addition_filtered_heatmap",
        figure_dir=args.figure_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        command_submit(args)
    elif args.command == "stage2":
        command_stage2(args)
    elif args.command == "finalize":
        command_finalize(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
