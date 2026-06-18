"""Stable loaders for self-improvement artifacts.

Notebook code should use these helpers instead of hard-coding raw JSON paths.
The helpers intentionally return plain Python records so pandas remains
optional. Adaptive-run helpers are implemented in
``self.analysis.adaptive_artifacts`` and reexported here for compatibility.
"""

from __future__ import annotations

from self.analysis.adaptive_artifacts import (
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    DEFAULT_ADAPTIVE_TRACE_FILES,
    adaptive_attempt_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_candidate_train_mix_records,
    adaptive_prompt_records,
    adaptive_proposal_grpo_records,
    adaptive_proposal_records,
    adaptive_selected_per_size_timeline_records,
    adaptive_trace_records,
    adaptive_trace_rows,
    discover_adaptive_runs,
    is_adaptive_run_dir,
    iter_attempt_dirs,
    load_adaptive_attempt,
    load_adaptive_run,
    load_adaptive_runs,
)
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
