"""Data models for sandboxed composition-program validation and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]
RepairCallback = Callable[[str, str, str], Optional[str]]


@dataclass(frozen=True)
class SandboxCase:
    name: str
    components: Sequence[Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    expected_accept: Optional[bool] = None
    expected_target: Optional[str] = None
    target_pattern: Optional[str] = None


@dataclass(frozen=True)
class ProgramValidationResult:
    valid: bool
    category: str = ""
    message: str = ""
    repaired: bool = False
    code: str = ""
    repaired_code: Optional[str] = None
    original_category: Optional[str] = None
    original_message: Optional[str] = None
    repair_attempted: bool = False


@dataclass(frozen=True)
class ProgramExecutionResult:
    valid: bool
    category: str = ""
    message: str = ""
    outputs: Tuple[JsonDict, ...] = ()
