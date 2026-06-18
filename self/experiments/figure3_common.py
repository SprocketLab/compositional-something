"""Shared helpers for Figure 3 experiment submission and collection."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "artifacts/logs"

MIG_PARTITION = "mig"
MIG_GRES = "gpu:1g.10gb:1"
MIG_MEM = "64G"
MIG_CPUS = "1"
MIG_TIME = "48:00:00"
DEFAULT_SEED_BANDS = {
    "low": (0.70, 0.80, 0.75),
    "medium": (0.80, 0.90, 0.85),
    "high": (0.95, 1.01, 1.00),
}
SEED_BAND_NAMES = ("low", "medium", "high")


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slurm_config() -> Dict[str, str]:
    return {
        "partition": MIG_PARTITION,
        "gres": MIG_GRES,
        "mem": MIG_MEM,
        "cpus_per_task": MIG_CPUS,
        "time": MIG_TIME,
    }


def log_paths(log_dir: Path, job_name: str) -> tuple[Path, Path]:
    safe_name = job_name.replace("/", "-")
    return log_dir / f"{safe_name}-%j.out", log_dir / f"{safe_name}-%j.err"


def run_command(cmd: Sequence[str], *, dry_run: bool) -> Optional[str]:
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
    stderr = completed.stderr.strip()
    if stdout:
        print(stdout, flush=True)
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    return stdout


def submit_sbatch_job(
    *,
    job_name: str,
    wrap_cmd: Sequence[str],
    log_dir: Path,
    dependency_job_ids: Sequence[str] = (),
    dry_run: bool,
) -> Optional[str]:
    stdout_log, stderr_log = log_paths(log_dir, job_name)
    slurm = slurm_config()
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
    result = run_command(sbatch_cmd, dry_run=dry_run)
    return None if result is None else result.split(";")[0].strip()


def metric_from_seed_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "seed" in payload and isinstance(payload["seed"], dict):
        seed_payload = payload["seed"]
        validation_min = seed_payload.get("validation_min_per_digit_accuracy")
        test_min = seed_payload.get("test_min_per_digit_accuracy")
        results = seed_payload.get("results", {})
    else:
        validation_min = payload.get("validation_min_per_size_accuracy")
        test_min = payload.get("test_min_per_size_accuracy")
        results = payload.get("results", {})
    if validation_min is None or test_min is None:
        raise KeyError(f"Could not find validation/test seed minima in {path}")
    return {
        "validation_min_accuracy": float(validation_min),
        "test_min_accuracy": float(test_min),
        "worst_case_accuracy": min(float(validation_min), float(test_min)),
        "results": results,
    }


def load_seed_candidates(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for entry in entries:
        results_path = Path(str(entry["results_path"]))
        model_dir = Path(str(entry["model_dir"]))
        if not results_path.exists():
            raise FileNotFoundError(f"Missing seed result file: {results_path}")
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing seed model directory: {model_dir}")
        metrics = metric_from_seed_payload(results_path)
        candidates.append({**dict(entry), **metrics})
    return candidates


def seed_band_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    band: str,
    task: Optional[str] = None,
    seed_bands: Mapping[str, tuple[float, float, float]] = DEFAULT_SEED_BANDS,
) -> List[Dict[str, Any]]:
    lower, upper, _target = seed_bands[band]
    return [
        dict(candidate)
        for candidate in candidates
        if (task is None or candidate["task"] == task)
        and lower <= float(candidate["worst_case_accuracy"]) < upper
    ]


def select_seed_band(
    candidates: Sequence[Mapping[str, Any]],
    *,
    band: str,
    high_source: Optional[str] = None,
    seed_bands: Mapping[str, tuple[float, float, float]] = DEFAULT_SEED_BANDS,
    extra_sort_key: Optional[Callable[[Mapping[str, Any]], tuple[Any, ...]]] = None,
) -> Optional[Dict[str, Any]]:
    lower, upper, target = seed_bands[band]
    del upper
    candidate_dicts = [dict(candidate) for candidate in candidates]
    if band == "high":
        eligible: List[Dict[str, Any]] = []
        if high_source is not None:
            eligible = [
                candidate
                for candidate in candidate_dicts
                if candidate.get("source") == high_source and float(candidate["worst_case_accuracy"]) >= lower
            ]
        if not eligible:
            eligible = [candidate for candidate in candidate_dicts if float(candidate["worst_case_accuracy"]) >= lower]
    else:
        eligible = seed_band_candidates(candidate_dicts, band=band, seed_bands=seed_bands)
    if not eligible:
        return None

    def sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        extra = extra_sort_key(candidate) if extra_sort_key is not None else ()
        return (
            abs(float(candidate["worst_case_accuracy"]) - target),
            -float(candidate["worst_case_accuracy"]),
            int(candidate["train_count"]),
            *extra,
        )

    eligible.sort(key=sort_key)
    return eligible[0]


def missing_seed_bands_for_task(
    selection: Mapping[str, Mapping[str, Any]],
    task: str,
    *,
    bands: Sequence[str] = SEED_BAND_NAMES,
) -> List[str]:
    return [band for band in bands if band not in selection.get(task, {})]


def missing_seed_bands_by_task(
    selection: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[str],
    *,
    bands: Sequence[str] = SEED_BAND_NAMES,
) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for task in tasks:
        absent = missing_seed_bands_for_task(selection, task, bands=bands)
        if absent:
            missing[task] = absent
    return missing


def final_row(results_path: Path) -> Dict[str, Any]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"Empty self-improvement results: {results_path}")
    return dict(rows[-1])


def max_at_90(final: Mapping[str, Any]) -> Optional[int]:
    for key in ("max_bits_at_90_accuracy", "max_digits_at_90_accuracy", "max_solved_size_at_90_accuracy"):
        value = final.get(key)
        if value is not None:
            return int(value)
    per_size = final.get("per_bit_accuracy") or final.get("per_digit_accuracy") or final.get("per_size_accuracy")
    if isinstance(per_size, dict):
        solved = [int(size) for size, accuracy in per_size.items() if accuracy is not None and float(accuracy) >= 0.90]
        return max(solved) if solved else None
    return None
