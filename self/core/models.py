"""Shared data containers for adaptive candidate training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import ConfigProposal


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ExecutableProposal:
    left: int
    right: int
    guard: str
    target: int
    code: str
    condition: str
    notes: str = ""
    representation: str = ""
    target_format: str = ""
    repaired: bool = False
    original_validation_category: Optional[str] = None
    original_validation_message: Optional[str] = None

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "left": self.left,
                "right": self.right,
                "guard": self.guard,
                "target": self.target,
                "code": self.code,
                "condition": self.condition,
                "notes": self.notes,
                "representation": self.representation,
                "target_format": self.target_format,
                "repaired": self.repaired,
                "original_validation_category": self.original_validation_category,
                "original_validation_message": self.original_validation_message,
            }
        )

    def to_completion(self) -> str:
        return self.code


@dataclass(frozen=True)
class ExactPairDataset:
    examples: List[Any]
    component_map: Dict[Any, List[Any]]
    keys: set[Any]
    diagnostics: JsonDict


@dataclass(frozen=True)
class CandidateWorkItem:
    index: int
    row_id: Optional[str]
    proposal: Any
    completion: str
    raw_output: Any
    composed: ExactPairDataset
    pseudo_examples: List[Any]
    pseudo_diagnostics: JsonDict
    proposal_prediction: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMetrics:
    index: int
    row_id: Optional[str]
    proposal: Any
    valid: bool
    reward: float
    frontier_delta: float
    target_accuracy: float
    current_target_accuracy: float
    final_accuracy: float
    init_final_accuracy: float
    final_accuracy_delta: float
    per_size_accuracy: Dict[int, float]
    pseudo_count: int
    model_dir: Optional[Path]
    failure_reason: Optional[str] = None
    proposal_trace_replay_count: int = 0
    candidate_proposal_trace_count: int = 0
    post_task_proposal_rehearsal_count: int = 0
    outcome_trace_replay_count: int = 0
    current_final_accuracy: float = math.nan
    final_accuracy_delta_from_current: float = math.nan
    target_delta: float = math.nan
    frontier_accuracy: float = math.nan
    current_frontier_accuracy: float = math.nan
    proposal_prediction: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "index": self.index,
                "id": self.row_id,
                "parsed_proposal": self.proposal.to_json_dict(),
                "valid": self.valid,
                "reward": self.reward,
                "frontier_delta": self.frontier_delta,
                "frontier_accuracy": self.frontier_accuracy,
                "current_frontier_accuracy": self.current_frontier_accuracy,
                "proposal_prediction": self.proposal_prediction,
                "target_accuracy": self.target_accuracy,
                "current_target_accuracy": self.current_target_accuracy,
                "target_delta": self.target_delta,
                "final_accuracy": self.final_accuracy,
                "current_final_accuracy": self.current_final_accuracy,
                "init_final_accuracy": self.init_final_accuracy,
                "final_accuracy_delta": self.final_accuracy_delta,
                "final_accuracy_delta_from_current": self.final_accuracy_delta_from_current,
                "per_size_accuracy": self.per_size_accuracy,
                "pseudo_count": self.pseudo_count,
                "proposal_trace_replay_count": self.proposal_trace_replay_count,
                "candidate_proposal_trace_count": self.candidate_proposal_trace_count,
                "post_task_proposal_rehearsal_count": self.post_task_proposal_rehearsal_count,
                "outcome_trace_replay_count": self.outcome_trace_replay_count,
                "model_dir": str(self.model_dir) if self.model_dir is not None else None,
                "failure_reason": self.failure_reason,
            }
        )


def proposal_from_payload(payload: Mapping[str, Any]) -> Any:
    if "code" in payload:
        return ExecutableProposal(
            left=int(payload["left"]),
            right=int(payload["right"]),
            guard=str(payload["guard"]),
            target=int(payload["target"]),
            code=str(payload["code"]),
            condition=str(payload.get("condition") or "program"),
            notes=str(payload.get("notes") or ""),
            representation=str(payload.get("representation") or ""),
            target_format=str(payload.get("target_format") or ""),
            repaired=bool(payload.get("repaired", False)),
            original_validation_category=(
                str(payload["original_validation_category"])
                if payload.get("original_validation_category") is not None
                else None
            ),
            original_validation_message=(
                str(payload["original_validation_message"])
                if payload.get("original_validation_message") is not None
                else None
            ),
        )
    return ConfigProposal(
        left=int(payload["left"]),
        right=int(payload["right"]),
        guard=str(payload["guard"]),
        target=int(payload.get("target", int(payload["left"]) + int(payload["right"]))),
        notes=str(payload.get("notes") or ""),
    )


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def candidate_metrics_from_json(payload: Mapping[str, Any]) -> CandidateMetrics:
    per_size_accuracy = {
        int(size): float(score)
        for size, score in dict(payload.get("per_size_accuracy", {})).items()
    }
    return CandidateMetrics(
        index=int(payload["index"]),
        row_id=str(payload["id"]) if payload.get("id") is not None else None,
        proposal=proposal_from_payload(dict(payload["parsed_proposal"])),
        valid=bool(payload.get("valid", False)),
        reward=float_or_nan(payload.get("reward")),
        frontier_delta=float_or_nan(payload.get("frontier_delta")),
        frontier_accuracy=float_or_nan(payload.get("frontier_accuracy")),
        current_frontier_accuracy=float_or_nan(payload.get("current_frontier_accuracy")),
        target_accuracy=float_or_nan(payload.get("target_accuracy")),
        current_target_accuracy=float_or_nan(payload.get("current_target_accuracy")),
        target_delta=float_or_nan(payload.get("target_delta")),
        final_accuracy=float_or_nan(payload.get("final_accuracy")),
        init_final_accuracy=float_or_nan(payload.get("init_final_accuracy")),
        final_accuracy_delta=float_or_nan(payload.get("final_accuracy_delta")),
        current_final_accuracy=float_or_nan(payload.get("current_final_accuracy")),
        final_accuracy_delta_from_current=float_or_nan(payload.get("final_accuracy_delta_from_current")),
        per_size_accuracy=per_size_accuracy,
        pseudo_count=int(payload.get("pseudo_count", 0)),
        proposal_trace_replay_count=int(payload.get("proposal_trace_replay_count", 0)),
        candidate_proposal_trace_count=int(payload.get("candidate_proposal_trace_count", 0)),
        post_task_proposal_rehearsal_count=int(payload.get("post_task_proposal_rehearsal_count", 0)),
        outcome_trace_replay_count=int(payload.get("outcome_trace_replay_count", 0)),
        model_dir=Path(str(payload["model_dir"])) if payload.get("model_dir") else None,
        failure_reason=str(payload["failure_reason"]) if payload.get("failure_reason") else None,
        proposal_prediction=dict(payload.get("proposal_prediction") or {}),
    )
