"""Default concrete bindings used by the adaptive driver.

`self.core.driver` exposes these names lazily so old imports and monkeypatches
keep working without forcing the driver module itself to import every runtime
helper at top level.
"""

from __future__ import annotations

import argparse
import subprocess

import torch

from self.core import worker_io
from self.core.args import build_parser, normalize_args
from self.core.attempt_outcome_runtime import handle_attempt_outcome
from self.core.attempt_prompt_runtime import build_attempt_prompt
from self.core.candidate_data import build_candidate_work_items
from self.core.candidate_execution import work_item_from_worker_payload, work_item_to_worker_payload
from self.core.candidate_scoring import make_config, train_and_score_candidate
from self.core.candidate_selection import select_candidate
from self.core.candidate_workers import prepare_candidate_worker_specs as _prepare_candidate_worker_specs
from self.core.controller_phase_runtime import run_round_model_phase, run_seed_phase
from self.core.controller_workers import (
    run_controller_worker as _run_controller_worker_generic,
    run_controller_worker_slurm as _run_controller_worker_slurm,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples, write_json
from self.core.dry_run_runtime import run_dry_attempt
from self.core.experience_traces import (
    build_round_outcome_trace_examples,
    build_selected_proposal_trace_example,
    write_round_trace,
)
from self.core.models import candidate_metrics_from_json, float_or_nan as _float_or_nan
from self.core.proposal_grpo import apply_proposal_grpo_update
from self.core.proposal_runtime import (
    _rows_for_round,
    choose_default_program_pair,
    render_program_candidate_prompt,
    validate_proposal_rows,
)
from self.core.proposals import PromptBundle, load_fixture_proposals, render_config_prompt, write_trace_jsonl
from self.core.round_model_dispatch_runtime import run_round_model_dispatch
from self.core.run_setup import append_plan_log, load_trace_jsonl, prepare_datasets
from self.core.seed_dispatch_runtime import run_seed_dispatch
from self.core.task_registry import task_for_name


def _default_bf16_on_cuda(args: argparse.Namespace, label: str) -> None:
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print(f"[INFO] {label} defaulting to bf16 on CUDA.", flush=True)


_json_ready_args = worker_io.json_ready_args
_PATH_ARG_NAMES = worker_io.PATH_ARG_NAMES
_namespace_from_json_args = worker_io.namespace_from_json_args
_load_json = worker_io.load_json
_json_ready_key = worker_io.json_ready_key
_key_from_json = worker_io.key_from_json
write_key_set = worker_io.write_key_set
load_key_set = worker_io.load_key_set


DEFAULT_BINDING_NAMES = (
    "PromptBundle",
    "_PATH_ARG_NAMES",
    "_default_bf16_on_cuda",
    "_float_or_nan",
    "_json_ready_args",
    "_json_ready_key",
    "_key_from_json",
    "_load_json",
    "_namespace_from_json_args",
    "_prepare_candidate_worker_specs",
    "_rows_for_round",
    "_run_controller_worker_generic",
    "_run_controller_worker_slurm",
    "append_plan_log",
    "apply_proposal_grpo_update",
    "build_attempt_prompt",
    "build_candidate_work_items",
    "build_parser",
    "build_round_outcome_trace_examples",
    "build_selected_proposal_trace_example",
    "candidate_metrics_from_json",
    "choose_default_program_pair",
    "ensure_dir",
    "handle_attempt_outcome",
    "load_fixture_proposals",
    "load_key_set",
    "load_trace_jsonl",
    "make_config",
    "normalize_args",
    "prepare_datasets",
    "render_config_prompt",
    "render_program_candidate_prompt",
    "run_dry_attempt",
    "run_round_model_dispatch",
    "run_round_model_phase",
    "run_seed_dispatch",
    "run_seed_phase",
    "sanitize_json_value",
    "save_examples",
    "select_candidate",
    "subprocess",
    "task_for_name",
    "train_and_score_candidate",
    "validate_proposal_rows",
    "work_item_from_worker_payload",
    "work_item_to_worker_payload",
    "write_json",
    "write_key_set",
    "write_round_trace",
    "write_trace_jsonl",
)


__all__ = list(DEFAULT_BINDING_NAMES)
