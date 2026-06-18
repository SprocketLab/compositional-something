#!/usr/bin/env python3
"""Submit and collect Figure 3 seed-quality/sample-size MIG sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.experiments.figure3_common import (
    DEFAULT_SEED_BANDS,
    DEFAULT_LOG_DIR,
    MIG_CPUS,
    MIG_GRES,
    MIG_MEM,
    MIG_PARTITION,
    MIG_TIME,
    ROOT_DIR,
    SEED_BAND_NAMES,
    final_row as _final_row,
    json_dump as _json_dump,
    log_paths as _log_paths,
    load_seed_candidates as _load_seed_candidates,
    max_at_90 as _max_at_90,
    metric_from_seed_payload as _metric_from_seed_payload,
    missing_seed_bands_by_task as _missing_seed_bands_by_task,
    run_command as _run_command,
    run_length_self_improvement_command as _run_length_self_improvement_command,
    select_seed_band as _select_seed_band,
    seed_fit_command as _seed_fit_command,
    slurm_config as _slurm_config,
    submit_sbatch_job as _submit_sbatch_job,
    write_csv as _write_csv,
)

DEFAULT_RUN_LENGTH_HIGH_SEED_MODEL = (
    ROOT_DIR / "artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model"
)
DEFAULT_RUN_LENGTH_HIGH_SEED_RESULTS = (
    ROOT_DIR / "artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/seed_fit_results.json"
)
DEFAULT_ADDITION_HIGH_SEED_MODEL = ROOT_DIR / "artifacts/models/addition_recipe_seed_best"
DEFAULT_ADDITION_HIGH_SEED_RESULTS = (
    ROOT_DIR / "artifacts/runs/addition_recipe_recovery_mig_20260419_072835/diagnostic/summary.json"
)

SEED_TRAIN_COUNTS = (250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
REFINEMENT_TRAIN_COUNTS = (50, 100, 150, 300, 750, 1_500, 3_000)
SEED_BANDS = DEFAULT_SEED_BANDS
TASKS = ("run_length", "addition")
RUN_LENGTH_SAMPLE_SIZES = (500, 1_000, 2_000, 4_000)
ADDITION_SAMPLE_SIZES = (2_500, 5_000, 10_000, 20_000)
RUN_LENGTH_DEFAULT_SAMPLE_SIZE = 2_000
ADDITION_DEFAULT_SAMPLE_SIZE = 10_000


def _seed_output_root(out_root: Path, task: str, train_count: int, *, refinement: bool = False) -> Path:
    prefix = "seed_refinement" if refinement else "seed_calibration"
    return out_root / prefix / task / f"train{train_count}"


def build_seed_jobs(
    *,
    out_root: Path,
    python_bin: str,
    train_counts: Sequence[int] = SEED_TRAIN_COUNTS,
    refinement: bool = False,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for task in TASKS:
        for train_count in train_counts:
            root = _seed_output_root(out_root, task, int(train_count), refinement=refinement)
            jobs.append(
                {
                    "kind": "seed",
                    "task": task,
                    "train_count": int(train_count),
                    "output_root": str(root),
                    "results_path": str(root / "seed_fit_results.json"),
                    "model_dir": str(root / "model"),
                    "python_bin": python_bin,
                    "refinement": refinement,
                }
            )
    return jobs


def existing_high_seed_candidates(
    *,
    run_length_model: Path = DEFAULT_RUN_LENGTH_HIGH_SEED_MODEL,
    run_length_results: Path = DEFAULT_RUN_LENGTH_HIGH_SEED_RESULTS,
    addition_model: Path = DEFAULT_ADDITION_HIGH_SEED_MODEL,
    addition_results: Path = DEFAULT_ADDITION_HIGH_SEED_RESULTS,
) -> List[Dict[str, Any]]:
    return [
        {
            "kind": "existing_high_seed",
            "task": "run_length",
            "train_count": 50_000,
            "output_root": str(run_length_results.parent),
            "results_path": str(run_length_results),
            "model_dir": str(run_length_model),
            "source": "existing_paper_seed",
        },
        {
            "kind": "existing_high_seed",
            "task": "addition",
            "train_count": 50_000,
            "output_root": str(addition_results.parent),
            "results_path": str(addition_results),
            "model_dir": str(addition_model),
            "source": "existing_paper_seed",
        },
    ]


load_seed_candidates = _load_seed_candidates


def select_seed_bands(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        task_candidates = [dict(candidate) for candidate in candidates if candidate["task"] == task]
        selected[task] = {}
        for band in SEED_BAND_NAMES:
            winner = _select_seed_band(
                task_candidates,
                band=band,
                high_source="existing_paper_seed",
                seed_bands=SEED_BANDS,
            )
            if winner is not None:
                selected[task][band] = winner
    return selected


def missing_seed_bands(selection: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    return _missing_seed_bands_by_task(selection, TASKS, bands=SEED_BAND_NAMES)


def _seed_job_command(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    return _seed_fit_command(entry, python_bin=python_bin, max_steps_position="task_specific")


def submit_seed_jobs(
    entries: Sequence[Mapping[str, Any]],
    *,
    log_dir: Path,
    python_bin: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        job_name = f"fig3-seed-{entry['task']}-train{entry['train_count']}"
        if entry.get("refinement"):
            job_name = f"fig3-refine-{entry['task']}-train{entry['train_count']}"
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=_seed_job_command(entry, python_bin=python_bin),
            log_dir=log_dir,
            dry_run=dry_run,
        )
        submitted.append({**dict(entry), "job_name": job_name, "job_id": job_id})
        if job_id is not None:
            print(
                f"[INFO] Submitted {job_name} job_id={job_id} results_path={entry['results_path']}",
                flush=True,
            )
    return submitted


def _run_length_si_command(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    return _run_length_self_improvement_command(entry, python_bin=python_bin, num_expand_rounds=7)


def _addition_si_command(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    return [
        python_bin,
        "-m",
        "self.self_improvement",
        "--model-name",
        str(entry["seed_model"]),
        "--output-dir",
        str(entry["output_root"]),
        "--recipe",
        "arithmetic_self_improve_v1",
        "--treat-seed-as-round-zero",
        "--seed-range-train-mode",
        "direct_pseudo",
        "--addition-width-mode",
        "exact_digits",
        "--addition-composition-path-mode",
        "random",
        "--initial-min-digits",
        "3",
        "--initial-max-digits",
        "7",
        "--initial-train-per-digit",
        "0",
        "--initial-eval-per-digit",
        "200",
        "--num-expand-rounds",
        "8",
        "--expand-num-digits",
        "2",
        "--seed-replay-train-per-digit",
        "5000",
        "--expand-train-per-digit",
        str(entry["sample_size"]),
        "--eval-per-digit",
        "100",
        "--composed-eval-per-digit",
        "50",
        "--pseudo-label-mode",
        "compose",
        "--composed-strategy",
        "with_carry_filtered",
        "--composition-error-percent",
        "0",
        "--composed-refresh-mode",
        "dynamic",
        "--bucket-train-batches-by-digits",
        "--decode-max-new-tokens",
        "48",
        "--per-device-train-batch-size",
        "256",
        "--per-device-eval-batch-size",
        "256",
        "--gradient-accumulation-steps",
        "1",
        "--bf16",
        "--resume",
        "--seed",
        "0",
    ]


def build_self_improvement_jobs(
    *,
    out_root: Path,
    selection: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    defaults = {
        "run_length": RUN_LENGTH_DEFAULT_SAMPLE_SIZE,
        "addition": ADDITION_DEFAULT_SAMPLE_SIZE,
    }
    samples = {
        "run_length": RUN_LENGTH_SAMPLE_SIZES,
        "addition": ADDITION_SAMPLE_SIZES,
    }
    for task in TASKS:
        high_seed = selection[task]["high"]
        for level in ("low", "medium", "high"):
            seed = selection[task][level]
            output_root = out_root / "self_improvement" / task / "seed_quality" / level
            jobs.append(
                {
                    "kind": "seed_quality_si",
                    "task": task,
                    "seed_level": level,
                    "sample_size": defaults[task],
                    "seed_model": seed["model_dir"],
                    "seed_score": seed["worst_case_accuracy"],
                    "output_root": str(output_root),
                    "results_path": str(output_root / "self_improvement_results.json"),
                }
            )
        for sample_size in samples[task]:
            output_root = out_root / "self_improvement" / task / "sample_size" / f"train{sample_size}"
            jobs.append(
                {
                    "kind": "sample_size_si",
                    "task": task,
                    "seed_level": "high",
                    "sample_size": int(sample_size),
                    "seed_model": high_seed["model_dir"],
                    "seed_score": high_seed["worst_case_accuracy"],
                    "output_root": str(output_root),
                    "results_path": str(output_root / "self_improvement_results.json"),
                }
            )
    return jobs


def submit_self_improvement_jobs(
    entries: Sequence[Mapping[str, Any]],
    *,
    log_dir: Path,
    python_bin: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    for entry in entries:
        job_name = f"fig3-si-{entry['task']}-{entry['kind']}-{entry['seed_level']}-n{entry['sample_size']}"
        command = (
            _run_length_si_command(entry, python_bin=python_bin)
            if entry["task"] == "run_length"
            else _addition_si_command(entry, python_bin=python_bin)
        )
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=command,
            log_dir=log_dir,
            dry_run=dry_run,
        )
        submitted.append({**dict(entry), "job_name": job_name, "job_id": job_id})
        if job_id is not None:
            print(
                f"[INFO] Submitted {job_name} job_id={job_id} results_path={entry['results_path']}",
                flush=True,
            )
    return submitted


def collect_summary(*, selection_path: Path, output_path: Path) -> Dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seed_rows: List[Dict[str, Any]] = []
    for task, band_map in selection["selected_seeds"].items():
        for level, candidate in band_map.items():
            seed_rows.append(
                {
                    "task": task,
                    "seed_level": level,
                    "train_count": candidate["train_count"],
                    "validation_min_accuracy": candidate["validation_min_accuracy"],
                    "test_min_accuracy": candidate["test_min_accuracy"],
                    "worst_case_accuracy": candidate["worst_case_accuracy"],
                    "model_dir": candidate["model_dir"],
                }
            )

    si_rows: List[Dict[str, Any]] = []
    for entry in selection["self_improvement_jobs"]:
        results_path = Path(entry["results_path"])
        if not results_path.exists():
            raise FileNotFoundError(f"Missing self-improvement result: {results_path}")
        final = _final_row(results_path)
        si_rows.append(
            {
                "task": entry["task"],
                "kind": entry["kind"],
                "seed_level": entry["seed_level"],
                "sample_size": entry["sample_size"],
                "seed_score": entry["seed_score"],
                "eval_accuracy": final.get("expanded_eval_accuracy", final.get("eval_accuracy")),
                "overall_eval_accuracy": final.get("eval_accuracy"),
                "composed_eval_accuracy": final.get("stitched_eval_accuracy", final.get("composed_eval_accuracy")),
                "max_size_at_90_accuracy": _max_at_90(final),
                "results_path": str(results_path),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seed_summary.csv", seed_rows), ("self_improvement_summary.csv", si_rows)):
        _write_csv(output_path.parent / name, rows)

    payload = {
        "seed_summary": seed_rows,
        "self_improvement_summary": si_rows,
    }
    _json_dump(output_path, payload)
    return payload


def _parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit", help="Submit initial seed calibration and dependent selector.")
    _parse_common(submit)
    stage2 = subparsers.add_parser("stage2", help="Select seed bands, optionally refine, and submit SI jobs.")
    _parse_common(stage2)
    stage2.add_argument("--refinement-attempt", type=int, default=0)
    collect = subparsers.add_parser("collect", help="Validate SI results and write compact summaries.")
    _parse_common(collect)
    return parser


def _default_manifest(out_root: Path) -> Path:
    return out_root / "manifest.json"


def _default_selection(out_root: Path) -> Path:
    return out_root / "selection.json"


def _default_summary(out_root: Path) -> Path:
    return out_root / "summary.json"


def command_submit(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or _default_manifest(args.out_root)
    selection_path = args.selection or _default_selection(args.out_root)
    seed_jobs = build_seed_jobs(out_root=args.out_root, python_bin=args.python_bin)
    manifest = {
        "meta": {
            "out_root": str(args.out_root),
            "python_bin": args.python_bin,
            "log_dir": str(args.log_dir),
            "slurm": _slurm_config(),
        },
        "seed_jobs": seed_jobs,
        "refinement_jobs": [],
    }
    _json_dump(manifest_path, manifest)
    submitted = submit_seed_jobs(seed_jobs, log_dir=args.log_dir, python_bin=args.python_bin, dry_run=args.dry_run)
    dep_ids = [entry["job_id"] for entry in submitted if entry.get("job_id")]
    print(
        "[INFO] Initial seed jobs: run_length=8 addition=8 total=16",
        flush=True,
    )
    stage2_cmd = [
        args.python_bin,
        "-m",
        "self.figure3_seed_quality_sweep",
        "stage2",
        "--out-root",
        str(args.out_root),
        "--manifest",
        str(manifest_path),
        "--selection",
        str(selection_path),
        "--log-dir",
        str(args.log_dir),
        "--python-bin",
        args.python_bin,
    ]
    if args.dry_run:
        stage2_cmd.append("--dry-run")
    stage2_job_id = _submit_sbatch_job(
        job_name="fig3-seed-quality-stage2",
        wrap_cmd=stage2_cmd,
        log_dir=args.log_dir,
        dependency_job_ids=dep_ids,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=1; stage-2 selector job not submitted.", flush=True)
    else:
        print(f"[INFO] Submitted stage-2 selector job_id={stage2_job_id}", flush=True)


def command_stage2(args: argparse.Namespace) -> None:
    manifest_path = args.manifest or _default_manifest(args.out_root)
    selection_path = args.selection or _default_selection(args.out_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_entries = list(manifest.get("seed_jobs", [])) + list(manifest.get("refinement_jobs", []))
    seed_entries += existing_high_seed_candidates()
    candidates = load_seed_candidates(seed_entries)
    selected = select_seed_bands(candidates)
    missing = missing_seed_bands(selected)
    if missing and args.refinement_attempt < 1:
        refine_tasks = sorted(missing)
        refinement_jobs = [
            job
            for job in build_seed_jobs(
                out_root=args.out_root,
                python_bin=args.python_bin,
                train_counts=REFINEMENT_TRAIN_COUNTS,
                refinement=True,
            )
            if job["task"] in refine_tasks
        ]
        manifest["refinement_jobs"] = refinement_jobs
        _json_dump(manifest_path, manifest)
        submitted = submit_seed_jobs(
            refinement_jobs,
            log_dir=args.log_dir,
            python_bin=args.python_bin,
            dry_run=args.dry_run,
        )
        dep_ids = [entry["job_id"] for entry in submitted if entry.get("job_id")]
        retry_cmd = [
            args.python_bin,
            "-m",
            "self.figure3_seed_quality_sweep",
            "stage2",
            "--out-root",
            str(args.out_root),
            "--manifest",
            str(manifest_path),
            "--selection",
            str(selection_path),
            "--log-dir",
            str(args.log_dir),
            "--python-bin",
            args.python_bin,
            "--refinement-attempt",
            "1",
        ]
        retry_job_id = _submit_sbatch_job(
            job_name="fig3-seed-quality-stage2-refined",
            wrap_cmd=retry_cmd,
            log_dir=args.log_dir,
            dependency_job_ids=dep_ids,
            dry_run=args.dry_run,
        )
        print(f"[INFO] Missing seed bands after initial pass: {missing}", flush=True)
        if args.dry_run:
            print("[INFO] DRY_RUN=1; refinement retry selector not submitted.", flush=True)
        else:
            print(f"[INFO] Submitted refinement retry selector job_id={retry_job_id}", flush=True)
        return
    if missing:
        raise RuntimeError(f"Missing seed bands after refinement: {missing}")

    si_jobs = build_self_improvement_jobs(out_root=args.out_root, selection=selected)
    selection_payload = {
        "selected_seeds": selected,
        "self_improvement_jobs": si_jobs,
        "all_seed_candidates": candidates,
    }
    _json_dump(selection_path, selection_payload)
    submitted = submit_self_improvement_jobs(
        si_jobs,
        log_dir=args.log_dir,
        python_bin=args.python_bin,
        dry_run=args.dry_run,
    )
    dep_ids = [entry["job_id"] for entry in submitted if entry.get("job_id")]
    collect_cmd = [
        args.python_bin,
        "-m",
        "self.figure3_seed_quality_sweep",
        "collect",
        "--out-root",
        str(args.out_root),
        "--selection",
        str(selection_path),
        "--summary",
        str(args.summary or _default_summary(args.out_root)),
        "--log-dir",
        str(args.log_dir),
        "--python-bin",
        args.python_bin,
    ]
    collect_job_id = _submit_sbatch_job(
        job_name="fig3-seed-quality-collect",
        wrap_cmd=collect_cmd,
        log_dir=args.log_dir,
        dependency_job_ids=dep_ids,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=1; collector not submitted.", flush=True)
    else:
        print(f"[INFO] Submitted collector job_id={collect_job_id}", flush=True)


def command_collect(args: argparse.Namespace) -> None:
    selection_path = args.selection or _default_selection(args.out_root)
    summary_path = args.summary or _default_summary(args.out_root)
    payload = collect_summary(selection_path=selection_path, output_path=summary_path)
    print(
        f"[INFO] Wrote summary to {summary_path} "
        f"(seed_rows={len(payload['seed_summary'])}, si_rows={len(payload['self_improvement_summary'])})",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        command_submit(args)
    elif args.command == "stage2":
        command_stage2(args)
    elif args.command == "collect":
        command_collect(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
