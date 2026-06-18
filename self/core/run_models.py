"""Adaptive run dependency containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveRunDeps:
    normalize_args: Any
    task_for_name: Any
    ensure_dir: Any
    make_config: Any
    prepare_datasets: Any
    save_examples: Any
    write_json: Any
    run_controller_worker_slurm: Any
    float_or_nan: Any
    run_seed_phase: Any
    build_attempt_prompt: Any
    run_dry_attempt: Any
    run_round_model_dispatch: Any
    train_candidate_metrics: Any
    select_candidate: Any
    write_round_trace: Any
    handle_attempt_outcome: Any
    choose_default_program_pair: Any
    render_config_prompt: Any
    render_program_candidate_prompt: Any
    load_fixture_proposals: Any
    rows_for_round: Any
    validate_proposal_rows: Any
    build_candidate_work_items: Any
    write_key_set: Any
    load_json: Any
    work_item_from_worker_payload: Any
    run_round_model_phase: Any
    build_round_outcome_trace_examples: Any
    build_selected_proposal_trace_example: Any
    apply_or_dispatch_proposal_grpo_update: Any
    write_trace_jsonl: Any
    append_plan_log: Any
    sanitize_json_value: Any
