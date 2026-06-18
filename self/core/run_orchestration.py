"""High-level adaptive run orchestration."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import set_seed

from self.core.attempt_loop_runtime import AttemptLoopDeps, run_adaptive_attempt_loop
from self.core.attempt_prompt_runtime import AttemptPromptDeps
from self.core.attempt_outcome_runtime import AttemptOutcomeDeps
from self.core.dry_run_runtime import DryRunAttemptDeps
from self.core.round_model_dispatch_runtime import RoundModelDispatchDeps
from self.core.run_finalization import finalize_adaptive_run
from self.core.run_initialization_runtime import RunInitializationDeps, initialize_adaptive_run
from self.core.seed_dispatch_runtime import SeedDispatchDeps, run_seed_dispatch


JsonDict = Dict[str, Any]


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


def run_adaptive_candidate_training(args: argparse.Namespace, deps: AdaptiveRunDeps) -> JsonDict:
    args = deps.normalize_args(args)
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] No precision flag provided; defaulting to bf16 on CUDA.", flush=True)
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    set_seed(args.seed)
    rng = random.Random(args.seed)
    run_inputs = initialize_adaptive_run(
        args=args,
        task=task,
        rng=rng,
        deps=RunInitializationDeps(
            ensure_dir=deps.ensure_dir,
            make_config=deps.make_config,
            prepare_datasets=deps.prepare_datasets,
            save_examples=deps.save_examples,
            write_json=deps.write_json,
        ),
    )
    output_dir = run_inputs.output_dir
    data_dir = run_inputs.data_dir
    checkpoint_manager = run_inputs.checkpoint_manager
    config = run_inputs.config
    source_examples = run_inputs.source_examples
    source_sizes = run_inputs.source_sizes
    exclude_keys = run_inputs.exclude_keys
    eval_examples = run_inputs.eval_examples

    seed_result = run_seed_dispatch(
        args=args,
        task=task,
        config=config,
        source_examples=source_examples,
        eval_examples=eval_examples,
        output_dir=output_dir,
        data_dir=data_dir,
        source_sizes=source_sizes,
        deps=SeedDispatchDeps(
            run_controller_worker_slurm=deps.run_controller_worker_slurm,
            float_or_nan=deps.float_or_nan,
            run_seed_phase=deps.run_seed_phase,
        ),
    )
    current_checkpoint = seed_result.current_checkpoint
    current_final_accuracy = seed_result.current_final_accuracy
    current_per_size_accuracy = seed_result.current_per_size_accuracy
    init_final_accuracy = seed_result.init_final_accuracy
    summary_records = seed_result.summary_records

    loop_result = run_adaptive_attempt_loop(
        args=args,
        task=task,
        config=config,
        rng=rng,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        source_examples=source_examples,
        source_sizes=source_sizes,
        exclude_keys=exclude_keys,
        eval_examples=eval_examples,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        summary_records=summary_records,
        deps=AttemptLoopDeps(
            ensure_dir=deps.ensure_dir,
            build_attempt_prompt=deps.build_attempt_prompt,
            write_json=deps.write_json,
            run_dry_attempt=deps.run_dry_attempt,
            run_round_model_dispatch=deps.run_round_model_dispatch,
            train_candidate_metrics=deps.train_candidate_metrics,
            select_candidate=deps.select_candidate,
            write_round_trace=deps.write_round_trace,
            handle_attempt_outcome=deps.handle_attempt_outcome,
            attempt_prompt_deps=AttemptPromptDeps(
                choose_default_program_pair=deps.choose_default_program_pair,
                render_config_prompt=deps.render_config_prompt,
                render_program_candidate_prompt=deps.render_program_candidate_prompt,
            ),
            dry_run_attempt_deps=DryRunAttemptDeps(
                load_fixture_proposals=deps.load_fixture_proposals,
                rows_for_round=deps.rows_for_round,
                validate_proposal_rows=deps.validate_proposal_rows,
                build_candidate_work_items=deps.build_candidate_work_items,
                write_json=deps.write_json,
            ),
            round_model_dispatch_deps=RoundModelDispatchDeps(
                save_examples=deps.save_examples,
                write_key_set=deps.write_key_set,
                run_controller_worker_slurm=deps.run_controller_worker_slurm,
                float_or_nan=deps.float_or_nan,
                load_json=deps.load_json,
                work_item_from_worker_payload=deps.work_item_from_worker_payload,
                run_round_model_phase=deps.run_round_model_phase,
            ),
            attempt_outcome_deps=AttemptOutcomeDeps(
                build_round_outcome_trace_examples=deps.build_round_outcome_trace_examples,
                build_selected_proposal_trace_example=deps.build_selected_proposal_trace_example,
                apply_or_dispatch_proposal_grpo_update=deps.apply_or_dispatch_proposal_grpo_update,
                write_json=deps.write_json,
                write_trace_jsonl=deps.write_trace_jsonl,
                save_examples=deps.save_examples,
            ),
        ),
    )

    return finalize_adaptive_run(
        args=args,
        output_dir=output_dir,
        summary_records=summary_records,
        selected_rounds=loop_result.selected_rounds,
        attempt_index=loop_result.attempt_index,
        current_checkpoint=loop_result.current_checkpoint,
        source_sizes=loop_result.source_sizes,
        proposal_trace_buffer=loop_result.proposal_trace_buffer,
        outcome_trace_buffer=loop_result.outcome_trace_buffer,
        proposal_grpo_update_count=loop_result.proposal_grpo_update_count,
        init_final_accuracy=init_final_accuracy,
        write_json=deps.write_json,
        append_plan_log=deps.append_plan_log,
        sanitize_json_value=deps.sanitize_json_value,
    )
