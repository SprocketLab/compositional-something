#!/usr/bin/env python3
"""Submit and collect Figure 3 real-task seed/data ablations on MIG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.experiments.figure3_cli import (
    add_common_args as _parse_common,
    default_manifest as _default_manifest,
    default_selection as _default_selection,
    default_summary as _default_summary,
)
from self.experiments.figure3_common import (
    DEFAULT_SEED_BANDS,
    SEED_BAND_NAMES,
    final_row as _final_row,
    json_dump as _json_dump,
    load_seed_candidates as _load_seed_candidates,
    max_at_90 as _max_at_90,
    metric_from_seed_payload as _metric_from_seed_payload,
    missing_seed_bands_for_task as _missing_seed_bands_for_task,
    run_length_self_improvement_command as _run_length_self_improvement_command,
    seed_band_candidates as _seed_band_candidates,
    seed_fit_command as _seed_fit_command,
    select_seed_band as _select_seed_band_common,
    submit_sbatch_job as _submit_sbatch_job,
    write_csv as _write_csv,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "artifacts/logs"

RUN_LENGTH_LOW_SEED_MODEL = (
    ROOT_DIR
    / "artifacts/runs/figure3_seed_quality_sweep_20260425_225711/seed_calibration/run_length/train1000/model"
)
RUN_LENGTH_LOW_SEED_RESULTS = (
    ROOT_DIR
    / "artifacts/runs/figure3_seed_quality_sweep_20260425_225711/seed_calibration/run_length/train1000/seed_fit_results.json"
)
RUN_LENGTH_HIGH_SEED_MODEL = (
    ROOT_DIR / "artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model"
)
RUN_LENGTH_HIGH_SEED_RESULTS = (
    ROOT_DIR
    / "artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/seed_fit_results.json"
)
ADDITION_HIGH_SEED_MODEL = ROOT_DIR / "artifacts/models/addition_recipe_seed_best"
ADDITION_HIGH_SEED_RESULTS = (
    ROOT_DIR / "artifacts/runs/addition_recipe_recovery_mig_20260419_072835/diagnostic/summary.json"
)

RUN_LENGTH_MEDIUM_TRAIN_COUNTS = (1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450)
ADDITION_TRAIN_COUNTS = (850, 900, 950, 1000)
ADDITION_MAX_STEPS = (1500, 2500, 3500, 5000, 7500, 10000)
RUN_LENGTH_SAMPLE_SIZES = {"low": 500, "medium": 1000, "high": 2000}

SEED_BANDS = DEFAULT_SEED_BANDS


def _seed_entry(
    *,
    task: str,
    train_count: int,
    output_root: Path,
    max_steps: int,
    kind: str,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "task": task,
        "train_count": int(train_count),
        "max_steps": int(max_steps),
        "output_root": str(output_root),
        "results_path": str(output_root / "seed_fit_results.json"),
        "model_dir": str(output_root / "model"),
    }


def build_seed_jobs(*, out_root: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for train_count in RUN_LENGTH_MEDIUM_TRAIN_COUNTS:
        jobs.append(
            _seed_entry(
                task="run_length",
                train_count=train_count,
                max_steps=15000,
                kind="run_length_medium_seed",
                output_root=out_root / "seed_calibration" / "run_length_medium" / f"train{train_count}",
            )
        )
    for train_count in ADDITION_TRAIN_COUNTS:
        for max_steps in ADDITION_MAX_STEPS:
            jobs.append(
                _seed_entry(
                    task="addition",
                    train_count=train_count,
                    max_steps=max_steps,
                    kind="addition_refined_seed",
                    output_root=(
                        out_root
                        / "seed_calibration"
                        / "addition_refined"
                        / f"train{train_count}_steps{max_steps}"
                    ),
                )
            )
    return jobs


def _seed_job_command(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    return _seed_fit_command(entry, python_bin=python_bin, max_steps_position="before_batch_args")


def submit_seed_jobs(
    entries: Sequence[Mapping[str, Any]],
    *,
    log_dir: Path,
    python_bin: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    for entry in entries:
        suffix = f"{entry['task']}-train{entry['train_count']}-steps{entry['max_steps']}"
        job_name = f"fig3-real-seed-{suffix}"
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=_seed_job_command(entry, python_bin=python_bin),
            log_dir=log_dir,
            dry_run=dry_run,
        )
        submitted.append({**dict(entry), "job_name": job_name, "job_id": job_id})
        if job_id:
            print(f"[INFO] Submitted {job_name} job_id={job_id}", flush=True)
    return submitted


def existing_seed_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "kind": "existing_run_length_low_seed",
            "task": "run_length",
            "train_count": 1000,
            "max_steps": 15000,
            "output_root": str(RUN_LENGTH_LOW_SEED_RESULTS.parent),
            "results_path": str(RUN_LENGTH_LOW_SEED_RESULTS),
            "model_dir": str(RUN_LENGTH_LOW_SEED_MODEL),
            "source": "existing_low_seed",
        },
        {
            "kind": "existing_run_length_high_seed",
            "task": "run_length",
            "train_count": 50000,
            "max_steps": 15000,
            "output_root": str(RUN_LENGTH_HIGH_SEED_RESULTS.parent),
            "results_path": str(RUN_LENGTH_HIGH_SEED_RESULTS),
            "model_dir": str(RUN_LENGTH_HIGH_SEED_MODEL),
            "source": "existing_high_seed",
        },
        {
            "kind": "existing_addition_high_seed",
            "task": "addition",
            "train_count": 50000,
            "max_steps": 10000,
            "output_root": str(ADDITION_HIGH_SEED_RESULTS.parent),
            "results_path": str(ADDITION_HIGH_SEED_RESULTS),
            "model_dir": str(ADDITION_HIGH_SEED_MODEL),
            "source": "existing_high_seed",
        },
    ]


load_seed_candidates = _load_seed_candidates


def _select_band(candidates: Sequence[Mapping[str, Any]], band: str) -> Optional[Dict[str, Any]]:
    return _select_seed_band_common(
        candidates,
        band=band,
        high_source="existing_high_seed",
        seed_bands=SEED_BANDS,
        extra_sort_key=lambda candidate: (int(candidate.get("max_steps", 0)),),
    )


def select_seed_bands(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {"run_length": {}, "addition": {}}
    for task in selected:
        task_candidates = [candidate for candidate in candidates if candidate["task"] == task]
        for band in SEED_BAND_NAMES:
            winner = _select_band(task_candidates, band)
            if winner is not None:
                selected[task][band] = winner
    _enforce_addition_monotone_low_medium(selected, candidates)
    return selected


def _band_candidates(candidates: Sequence[Mapping[str, Any]], *, task: str, band: str) -> List[Dict[str, Any]]:
    return _seed_band_candidates(candidates, task=task, band=band, seed_bands=SEED_BANDS)


def _enforce_addition_monotone_low_medium(
    selected: Dict[str, Dict[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    """Avoid reporting a low seed that used more supervised data than medium."""
    addition = selected.get("addition", {})
    low = addition.get("low")
    medium = addition.get("medium")
    if not low or not medium or int(low["train_count"]) < int(medium["train_count"]):
        return

    lows = _band_candidates(candidates, task="addition", band="low")
    mediums = _band_candidates(candidates, task="addition", band="medium")
    pairs = [
        (low_candidate, medium_candidate)
        for low_candidate in lows
        for medium_candidate in mediums
        if int(low_candidate["train_count"]) < int(medium_candidate["train_count"])
    ]
    if not pairs:
        addition.pop("low", None)
        addition.pop("medium", None)
        return

    pairs.sort(
        key=lambda pair: (
            abs(float(pair[0]["worst_case_accuracy"]) - SEED_BANDS["low"][2])
            + abs(float(pair[1]["worst_case_accuracy"]) - SEED_BANDS["medium"][2]),
            int(pair[1]["train_count"]) - int(pair[0]["train_count"]),
            int(pair[0]["train_count"]),
            int(pair[1]["train_count"]),
        )
    )
    addition["low"], addition["medium"] = pairs[0]


def missing_bands(selection: Mapping[str, Mapping[str, Any]], task: str) -> List[str]:
    return _missing_seed_bands_for_task(selection, task, bands=SEED_BAND_NAMES)


def build_run_length_si_jobs(
    *,
    out_root: Path,
    selection: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for seed_level in ("low", "medium", "high"):
        seed = selection["run_length"][seed_level]
        for sample_level, sample_size in RUN_LENGTH_SAMPLE_SIZES.items():
            output_root = out_root / "self_improvement" / "run_length" / f"seed_{seed_level}_sample_{sample_level}"
            jobs.append(
                {
                    "kind": "run_length_seed_data_si",
                    "task": "run_length",
                    "seed_level": seed_level,
                    "sample_level": sample_level,
                    "sample_size": int(sample_size),
                    "seed_model": seed["model_dir"],
                    "seed_score": seed["worst_case_accuracy"],
                    "seed_train_count": seed["train_count"],
                    "seed_max_steps": seed.get("max_steps"),
                    "output_root": str(output_root),
                    "results_path": str(output_root / "self_improvement_results.json"),
                }
            )
    return jobs


def _run_length_si_command(entry: Mapping[str, Any], *, python_bin: str) -> List[str]:
    return _run_length_self_improvement_command(entry, python_bin=python_bin, num_expand_rounds=8)


def submit_run_length_si_jobs(
    entries: Sequence[Mapping[str, Any]],
    *,
    log_dir: Path,
    python_bin: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    for entry in entries:
        job_name = f"fig3-real-rl-si-{entry['seed_level']}-{entry['sample_level']}"
        job_id = _submit_sbatch_job(
            job_name=job_name,
            wrap_cmd=_run_length_si_command(entry, python_bin=python_bin),
            log_dir=log_dir,
            dry_run=dry_run,
        )
        submitted.append({**dict(entry), "job_name": job_name, "job_id": job_id})
        if job_id:
            print(f"[INFO] Submitted {job_name} job_id={job_id}", flush=True)
    return submitted


def collect_summary(*, selection_path: Path, output_path: Path) -> Dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seed_rows: List[Dict[str, Any]] = []
    for candidate in selection.get("all_seed_candidates", []):
        seed_rows.append(
            {
                "task": candidate["task"],
                "kind": candidate["kind"],
                "train_count": candidate["train_count"],
                "max_steps": candidate.get("max_steps"),
                "validation_min_accuracy": candidate["validation_min_accuracy"],
                "test_min_accuracy": candidate["test_min_accuracy"],
                "worst_case_accuracy": candidate["worst_case_accuracy"],
                "model_dir": candidate["model_dir"],
            }
        )

    selected_rows: List[Dict[str, Any]] = []
    for task, band_map in selection.get("selected_seeds", {}).items():
        for band, candidate in band_map.items():
            selected_rows.append(
                {
                    "task": task,
                    "seed_level": band,
                    "train_count": candidate["train_count"],
                    "max_steps": candidate.get("max_steps"),
                    "worst_case_accuracy": candidate["worst_case_accuracy"],
                    "model_dir": candidate["model_dir"],
                }
            )

    si_rows: List[Dict[str, Any]] = []
    for entry in selection.get("run_length_self_improvement_jobs", []):
        results_path = Path(entry["results_path"])
        if not results_path.exists():
            raise FileNotFoundError(f"Missing self-improvement result: {results_path}")
        final = _final_row(results_path)
        si_rows.append(
            {
                "task": "run_length",
                "seed_level": entry["seed_level"],
                "sample_level": entry["sample_level"],
                "sample_size": entry["sample_size"],
                "seed_score": entry["seed_score"],
                "eval_accuracy": final.get("eval_accuracy"),
                "composed_eval_accuracy": final.get("composed_eval_accuracy", final.get("stitched_eval_accuracy")),
                "max_size_at_90_accuracy": _max_at_90(final),
                "results_path": str(results_path),
            }
        )

    payload = {
        "seed_candidates": seed_rows,
        "selected_seeds": selected_rows,
        "run_length_self_improvement_summary": si_rows,
        "missing_addition_seed_bands": selection.get("missing_addition_seed_bands", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(output_path, payload)
    _write_csv(output_path.parent / "seed_candidates.csv", seed_rows)
    _write_csv(output_path.parent / "selected_seeds.csv", selected_rows)
    _write_csv(output_path.parent / "run_length_matrix_summary.csv", si_rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit", help="Submit seed jobs and the dependent stage-2 selector.")
    _parse_common(submit)
    stage2 = subparsers.add_parser("stage2", help="Select seeds and submit run-length SI matrix.")
    _parse_common(stage2)
    collect = subparsers.add_parser("collect", help="Validate SI results and write summaries.")
    _parse_common(collect)
    return parser


def command_submit(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or _default_manifest(args.out_root)
    selection_path = args.selection or _default_selection(args.out_root)
    seed_jobs = build_seed_jobs(out_root=args.out_root)
    manifest = {
        "meta": {
            "out_root": str(args.out_root),
            "python_bin": args.python_bin,
            "log_dir": str(args.log_dir),
        },
        "seed_jobs": seed_jobs,
    }
    _json_dump(manifest_path, manifest)
    submitted = submit_seed_jobs(seed_jobs, log_dir=args.log_dir, python_bin=args.python_bin, dry_run=args.dry_run)
    dep_ids = [entry["job_id"] for entry in submitted if entry.get("job_id")]
    print(
        "[INFO] Initial seed jobs: run_length_medium=9 addition_refined=24 total=33",
        flush=True,
    )
    stage2_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.figure3_real_seed_data_ablation",
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
        job_name="fig3-real-stage2",
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
    candidates = load_seed_candidates(list(manifest.get("seed_jobs", [])) + existing_seed_candidates())
    selected = select_seed_bands(candidates)
    run_length_missing = missing_bands(selected, "run_length")
    if run_length_missing:
        raise RuntimeError(f"Missing required run-length seed bands: {run_length_missing}")
    addition_missing = missing_bands(selected, "addition")
    if addition_missing:
        print(f"[WARN] Addition seed bands missing after refined grid: {addition_missing}", flush=True)

    si_jobs = build_run_length_si_jobs(out_root=args.out_root, selection=selected)
    selection_payload = {
        "selected_seeds": selected,
        "missing_addition_seed_bands": addition_missing,
        "run_length_self_improvement_jobs": si_jobs,
        "all_seed_candidates": candidates,
    }
    _json_dump(selection_path, selection_payload)
    submitted = submit_run_length_si_jobs(
        si_jobs,
        log_dir=args.log_dir,
        python_bin=args.python_bin,
        dry_run=args.dry_run,
    )
    dep_ids = [entry["job_id"] for entry in submitted if entry.get("job_id")]
    print("[INFO] Run-length self-improvement jobs: 9", flush=True)
    collect_cmd = [
        args.python_bin,
        "-m",
        "self.experiments.figure3_real_seed_data_ablation",
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
        job_name="fig3-real-collect",
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
        f"(seed_rows={len(payload['seed_candidates'])}, si_rows={len(payload['run_length_self_improvement_summary'])})",
        flush=True,
    )
    if payload.get("missing_addition_seed_bands"):
        print(f"[WARN] Missing addition seed bands: {payload['missing_addition_seed_bands']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        command_submit(args)
    elif args.command == "stage2":
        command_stage2(args)
    elif args.command == "collect":
        command_collect(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
