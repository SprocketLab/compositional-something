"""Compatibility exports for :mod:`self.core.driver`.

The adaptive driver used to import a broad surface of task helpers, proposal
helpers, and data classes directly. Keep those old attribute/import paths
available here so the driver itself can stay focused on entrypoint wiring.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from core.addition_pipeline import (
    AdditionExample,
    bucket_by_digits,
    build_composed_pseudo_map,
    compose_examples,
    example_key,
    has_component_boundary_carry,
)
from self.core.driver_compat_manifest import COMPAT_EXPORT_NAMES
from self.core.args import (
    CANDIDATE_EXECUTION_MODES,
    CONDITION_CHOICES,
    CONTROLLER_EXECUTION_MODES,
    OUTCOME_TRACE_TARGET_MODES,
    TASK_CHOICES,
)
from self.core.candidate_data import attach_pseudo_labels, examples_by_key
from self.core.candidate_rewards import mean_accuracy_for_sizes, static_frontier_sizes
from self.core.candidate_training_runtime import evaluate_model, train_checkpoint
from self.core.checkpoints import CheckpointManager, cleanup_replaced_model_checkpoint, cleanup_unselected_models
from self.core.composition import (
    build_exact_pair_addition_dataset,
    build_exact_pair_dataset,
    build_exact_pair_run_length_dataset,
    merge_run_length_examples,
)
from self.core.composition_pseudolabels import (
    compose_addition_pseudo_examples,
    compose_program_pseudo_examples,
    compose_pseudo_examples,
    compose_run_length_pseudo_examples,
    target_pattern_for_task,
)
from self.core.controller_workers import (
    controller_worker_failure_path,
    controller_worker_output_path,
    controller_worker_time_limit_for_phase,
    submit_controller_worker,
    wait_for_controller_worker,
)
from self.core.data_io import load_examples
from self.core.evaluation import build_generation_encodings, generate_prediction_map, resolve_max_new_tokens
from self.core.experience_outcome_traces import build_outcome_trace_example
from self.core.experience_traces import (
    build_candidate_proposal_trace_example,
    proposal_trace_metadata,
)
from self.core.experience_trace_models import (
    OutcomeTraceExample,
    ProposalTraceExample,
    build_post_task_proposal_rehearsal_examples,
    outcome_trace_from_json,
    proposal_trace_from_json,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset, ExecutableProposal, proposal_from_payload
from self.core.program_sandbox import execute_program_cases
from self.core.proposal_grpo_traces import build_proposal_grpo_traces, proposal_grpo_advantages, proposal_grpo_reward
from self.core.proposal_config_validation import _raw_output, validate_config_rows
from self.core.proposal_executable_validation import (
    _extract_python_code,
    _repair_program_with_model,
    _row_payload,
    _row_repair_output,
    validate_executable_rows,
)
from self.core.proposal_generation import load_or_generate_proposal_rows
from self.core.proposal_config_schema import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    ConfigProposal,
    extract_json_object,
    normalized_config_completion,
    parse_config_proposal,
    proposal_output_schema,
    proposal_payload_for_schema,
    validate_config_prediction,
)
from self.core.proposal_prompts import (
    component_prediction_examples_for_task,
    program_validation_cases,
    render_program_repair_prompt,
    target_format_for_task,
)
from self.core.run_setup import source_sizes_from_examples
from self.core.training import TrainingConfig
from self.tasks.addition import AdditionTask
from self.tasks.bit_common import (
    RUN_LENGTH_TARGET_RUN_STATE,
    normalize_bit_target_mode,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
)
from self.tasks.run_length import RunLengthTask
from self.tasks.run_length_data import (
    RunLengthExample,
    bucket_run_length_by_bits,
    clone_run_length_with_override,
    run_length_key,
)
from self.tasks.run_length_logic import compute_run_stats, format_run_length_run_state, merge_run_state


__all__ = list(COMPAT_EXPORT_NAMES)
