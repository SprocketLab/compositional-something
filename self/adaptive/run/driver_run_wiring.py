"""Driver-binding bridge for full adaptive run orchestration."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from self.adaptive.run.run_models import AdaptiveRunDeps


JsonDict = Dict[str, Any]


def adaptive_run_deps(bindings: Any) -> AdaptiveRunDeps:
    return AdaptiveRunDeps(
        normalize_args=bindings.normalize_args,
        task_for_name=bindings.task_for_name,
        ensure_dir=bindings.ensure_dir,
        make_config=bindings.make_config,
        prepare_datasets=bindings.prepare_datasets,
        save_examples=bindings.save_examples,
        write_json=bindings.write_json,
        run_controller_worker_slurm=bindings._run_controller_worker_slurm,
        float_or_nan=bindings._float_or_nan,
        run_seed_phase=bindings.run_seed_phase,
        build_attempt_prompt=bindings.build_attempt_prompt,
        run_dry_attempt=bindings.run_dry_attempt,
        run_round_model_dispatch=bindings.run_round_model_dispatch,
        train_candidate_metrics=bindings.train_candidate_metrics,
        select_candidate=bindings.select_candidate,
        write_round_trace=bindings.write_round_trace,
        handle_attempt_outcome=bindings.handle_attempt_outcome,
        choose_default_program_pair=bindings.choose_default_program_pair,
        render_config_prompt=bindings.render_config_prompt,
        render_program_candidate_prompt=bindings.render_program_candidate_prompt,
        load_fixture_proposals=bindings.load_fixture_proposals,
        rows_for_round=bindings._rows_for_round,
        validate_proposal_rows=bindings.validate_proposal_rows,
        build_candidate_work_items=bindings.build_candidate_work_items,
        write_key_set=bindings.write_key_set,
        load_json=bindings._load_json,
        work_item_from_worker_payload=bindings.work_item_from_worker_payload,
        run_round_model_phase=bindings.run_round_model_phase,
        build_round_outcome_trace_examples=bindings.build_round_outcome_trace_examples,
        build_selected_proposal_trace_example=bindings.build_selected_proposal_trace_example,
        apply_or_dispatch_proposal_grpo_update=bindings.apply_or_dispatch_proposal_grpo_update,
        write_trace_jsonl=bindings.write_trace_jsonl,
        append_plan_log=bindings.append_plan_log,
        sanitize_json_value=bindings.sanitize_json_value,
    )


def run(bindings: Any, args: argparse.Namespace) -> JsonDict:
    from self.adaptive.run.run_orchestration import run_adaptive_candidate_training

    return run_adaptive_candidate_training(args=args, deps=adaptive_run_deps(bindings))
