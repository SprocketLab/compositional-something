"""Stable loaders for self-improvement artifacts.

Notebook code should use these helpers instead of hard-coding raw JSON paths.
The helpers intentionally return plain Python records so pandas remains
optional. Adaptive-run helpers are implemented in
``self.analysis.adaptive_artifacts`` and reexported here for compatibility.
"""

from __future__ import annotations

from self.analysis.adaptive_artifacts import (
    ADAPTIVE_CANDIDATE_FAILURE_FILE,
    ADAPTIVE_CANDIDATE_METRICS_FILE,
    ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE,
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    adaptive_attempt_records,
    adaptive_proposal_records,
    discover_adaptive_runs,
    is_adaptive_run_dir,
    iter_attempt_dirs,
    load_adaptive_attempt,
    load_adaptive_run,
    load_adaptive_runs,
)
from self.analysis.adaptive_candidate_artifacts import (
    AdaptiveCandidateArtifacts,
    adaptive_candidate_artifact_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_candidate_train_mix_records,
    iter_candidate_dirs,
    load_adaptive_candidate,
    load_adaptive_candidates,
)
from self.analysis.adaptive_manifest_artifacts import (
    SUBMISSION_MANIFEST_FILE,
    adaptive_submission_job_records,
    discover_submission_manifests,
    load_submission_manifest,
    resolve_submission_manifest_path,
)
from self.analysis.adaptive_artifact_common import DEFAULT_ADAPTIVE_TRACE_FILES
from self.analysis.artifact_io import (
    ADAPTIVE_RESULTS_FILE,
    SELF_IMPROVEMENT_RESULTS_FILE,
    JsonDict,
    read_json,
    read_jsonl,
)
from self.analysis.nonadaptive_artifacts import (
    load_self_improvement_rounds,
    per_size_accuracy_records,
    records_to_dataframe,
    resolve_self_improvement_results_path,
)
from self.analysis.adaptive_trace_artifacts import (
    adaptive_prompt_records,
    adaptive_proposal_grpo_records,
    adaptive_selected_per_size_timeline_records,
    adaptive_trace_records,
    adaptive_trace_rows,
)
