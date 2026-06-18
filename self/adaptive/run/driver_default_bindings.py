"""Default concrete bindings used by the adaptive driver.

`self.adaptive.run.driver` exposes these names lazily so old imports and monkeypatches
keep working without forcing the driver module itself to import every runtime
helper at top level.
"""

from __future__ import annotations

import argparse
import subprocess

import torch

from self.core import worker_io
from self.adaptive.run.args import build_parser, normalize_args
from self.adaptive.attempts.attempts import handle_attempt_outcome
from self.adaptive.attempts.attempts import build_attempt_prompt
from self.adaptive.candidates.training import build_candidate_work_items
from self.adaptive.candidates.training import train_and_score_candidate
from self.adaptive.candidates.training import make_config
from self.adaptive.candidates.training import select_candidate
from self.adaptive.candidates.workers import (
    work_item_from_worker_payload,
    work_item_to_worker_payload,
)
from self.adaptive.candidates.workers import prepare_candidate_worker_specs as _prepare_candidate_worker_specs
from self.adaptive.controller.controller import run_round_model_phase, run_seed_phase
from self.adaptive.controller.controller import (
    run_controller_worker as _run_controller_worker_generic,
    run_controller_worker_slurm as _run_controller_worker_slurm,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples, write_json
from self.adaptive.attempts.attempts import run_dry_attempt
from self.adaptive.traces.traces import build_round_outcome_trace_examples
from self.adaptive.traces.traces import (
    build_selected_proposal_trace_example,
    write_round_trace,
)
from self.core.models import candidate_metrics_from_json, float_or_nan as _float_or_nan
from self.adaptive.proposals.proposal_grpo import apply_proposal_grpo_update
from self.adaptive.proposals.proposal_generation import _rows_for_round
from self.adaptive.proposals.proposal_prompts import choose_default_program_pair
from self.adaptive.proposals.proposal_prompts import (
    PromptBundle,
    render_config_prompt,
    render_program_candidate_prompt,
)
from self.adaptive.proposals.proposal_runtime import (
    validate_proposal_rows,
)
from self.adaptive.proposals.proposal_io import load_fixture_proposals, write_trace_jsonl
from self.adaptive.run.round_model_dispatch_runtime import run_round_model_dispatch
from self.adaptive.run.run_setup import append_plan_log, load_trace_jsonl, prepare_datasets
from self.adaptive.run.seed_dispatch_runtime import run_seed_dispatch
from self.core.task_protocols import task_for_name
from self.adaptive.run.driver import DEFAULT_BINDING_NAMES


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


__all__ = list(DEFAULT_BINDING_NAMES)
