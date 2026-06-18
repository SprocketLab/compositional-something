"""Submission-manifest loaders for adaptive self-improvement runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, natural_sort_key, read_json


SUBMISSION_MANIFEST_FILE = "submission_manifest.json"


def resolve_submission_manifest_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    return resolved / SUBMISSION_MANIFEST_FILE


def load_submission_manifest(path: Path | str, default: Any = None) -> JsonDict:
    payload = read_json(resolve_submission_manifest_path(path), default if default is not None else {})
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object payload in submission manifest: {path}")
    return dict(payload)


def discover_submission_manifests(root: Path | str) -> list[Path]:
    resolved = Path(root)
    if resolved.is_file():
        return [resolved] if resolved.name == SUBMISSION_MANIFEST_FILE else []
    manifest = resolved / SUBMISSION_MANIFEST_FILE
    if manifest.exists():
        return [manifest]
    if not resolved.exists():
        return []
    return sorted(resolved.rglob(SUBMISSION_MANIFEST_FILE), key=natural_sort_key)


def adaptive_submission_job_records(root: Path | str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for manifest_path in discover_submission_manifests(root):
        payload = load_submission_manifest(manifest_path)
        jobs = payload.get("jobs") or {}
        if not isinstance(jobs, Mapping):
            continue
        slurm = payload.get("slurm") or {}
        if not isinstance(slurm, Mapping):
            slurm = {}
        for job_key, job in jobs.items():
            if not isinstance(job, Mapping):
                job = {"value": job}
            output_dir = job.get("output_dir", job.get("output_root"))
            row: JsonDict = {
                "manifest_path": str(manifest_path),
                "manifest_dir": str(manifest_path.parent),
                "manifest_name": manifest_path.parent.name,
                "out_root": payload.get("out_root"),
                "job_key": job_key,
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "output_dir": output_dir,
                "task": job.get("task"),
                "condition": job.get("condition"),
                "outcome_trace_target_mode": job.get("outcome_trace_target_mode"),
                "proposal_grpo_zero_variance": job.get("proposal_grpo_zero_variance"),
                "num_candidates": job.get("num_candidates"),
                "target_mode": job.get("target_mode"),
                "composition_path": job.get("composition_path"),
            }
            for key, value in slurm.items():
                row[f"slurm_{key}"] = value
            for key, value in job.items():
                row.setdefault(key, value)
            rows.append(row)
    return rows


__all__ = [
    "SUBMISSION_MANIFEST_FILE",
    "adaptive_submission_job_records",
    "discover_submission_manifests",
    "load_submission_manifest",
    "resolve_submission_manifest_path",
]
