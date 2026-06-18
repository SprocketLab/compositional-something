#!/usr/bin/env python3
"""Sandboxed validation and execution for generated composition programs."""

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


import random
from typing import List, Sequence, Tuple



def _run_length_state(bitstring: str) -> Tuple[int, str, int, str, int]:
    if not bitstring:
        return 0, "", 0, "", 0
    max_run = 1
    current = 1
    for prev, ch in zip(bitstring, bitstring[1:]):
        if prev == ch:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    prefix_symbol = bitstring[0]
    prefix_run = 0
    for ch in bitstring:
        if ch != prefix_symbol:
            break
        prefix_run += 1
    suffix_symbol = bitstring[-1]
    suffix_run = 0
    for ch in reversed(bitstring):
        if ch != suffix_symbol:
            break
        suffix_run += 1
    return max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run


def _format_run_length_state(bitstring: str) -> str:
    max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = _run_length_state(bitstring)
    return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"


def build_run_length_program_cases(*, random_seed: int = 0, random_count: int = 8) -> List[SandboxCase]:
    cases: List[SandboxCase] = []

    def add_case(name: str, parts: Sequence[str]) -> None:
        components = [
            {
                "size": len(part),
                "input_id": f"{len(part)}:{part}",
                "prediction": _format_run_length_state(part),
                "metadata": {"part_index": index},
            }
            for index, part in enumerate(parts)
        ]
        joined = "".join(parts)
        cases.append(
            SandboxCase(
                name=name,
                components=components,
                metadata={"component_count": len(parts)},
                expected_accept=True,
                expected_target=_format_run_length_state(joined),
                target_pattern=r"\d+\|[0-9A-Z]\|\d+\|[0-9A-Z]\|\d+",
            )
        )

    add_case("same_boundary", ["0011", "11100"])
    add_case("different_boundary", ["0011", "22200"])
    add_case("all_same", ["111", "11"])
    add_case("multi_part", ["0100", "000", "2222"])
    cases.append(
        SandboxCase(
            name="empty_components",
            components=[],
            metadata={},
            expected_accept=False,
        )
    )
    cases.append(
        SandboxCase(
            name="malformed_prediction",
            components=[{"size": 3, "input_id": "bad", "prediction": "not|state", "metadata": {}}],
            metadata={},
            expected_accept=False,
        )
    )

    rng = random.Random(random_seed)
    alphabet = "012"
    for index in range(random_count):
        part_count = rng.randint(2, 4)
        parts = []
        for _ in range(part_count):
            size = rng.randint(1, 6)
            parts.append("".join(rng.choice(alphabet) for _ in range(size)))
        add_case(f"random_{index}", parts)
    return cases


def build_addition_program_cases() -> List[SandboxCase]:
    return [
        SandboxCase(
            name="concat_no_carry",
            components=[
                {"size": 2, "input_id": "2:12+34", "prediction": "46", "metadata": {}},
                {"size": 2, "input_id": "2:21+43", "prediction": "64", "metadata": {}},
            ],
            metadata={"component_count": 2},
            expected_accept=True,
            expected_target="4664",
            target_pattern=r"-?\d+",
        ),
        SandboxCase(
            name="malformed_prediction",
            components=[{"size": 2, "input_id": "bad", "prediction": "x", "metadata": {}}],
            metadata={},
            expected_accept=False,
        ),
    ]


import ast
import multiprocessing as mp
import queue
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


FORBIDDEN_NAMES = {
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
    "breakpoint",
    "input",
    "help",
    "memoryview",
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "random",
    "time",
}
FORBIDDEN_ATTRIBUTES = {
    "read",
    "write",
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "unlink",
    "rename",
    "replace",
    "rmdir",
    "mkdir",
    "chdir",
    "chmod",
    "connect",
    "send",
    "recv",
    "request",
    "getattr",
    "setattr",
}
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
    "TypeError": TypeError,
}


def _static_validate(code: str) -> Tuple[bool, str, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, "syntax_error", f"line {exc.lineno}: {exc.msg}"

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    compose_functions = [node for node in functions if node.name == "compose"]
    if len(compose_functions) != 1 or len(tree.body) != 1:
        return False, "schema_error", "program must contain exactly one top-level compose function"

    compose = compose_functions[0]
    arg_names = [arg.arg for arg in compose.args.args]
    if arg_names != ["components", "metadata"] or compose.args.vararg or compose.args.kwarg:
        return False, "schema_error", "compose signature must be def compose(components, metadata)"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "forbidden_import", "imports are not allowed"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global_mutation", "global and nonlocal declarations are forbidden"
        if isinstance(node, ast.Delete):
            return False, "global_mutation", "delete statements are forbidden"
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False, "schema_error", "only a single compose function is allowed"
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                return False, "forbidden_name", f"name {node.id!r} is forbidden"
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("__"):
                return False, "forbidden_attribute", f"attribute {node.attr!r} is forbidden"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                return False, "forbidden_call", f"call to {func.id!r} is forbidden"
            if isinstance(func, ast.Attribute) and (
                func.attr in FORBIDDEN_ATTRIBUTES or func.attr.startswith("__")
            ):
                return False, "forbidden_call", f"call to attribute {func.attr!r} is forbidden"
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets: List[ast.AST]
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
            for target in targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)):
                    return False, "global_mutation", "assignment to object attributes/items is forbidden"

    return True, "", ""


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_jsonish(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_jsonish(item) for item in value)
    return value


def _validate_output(value: Any) -> Tuple[bool, str, str]:
    if not isinstance(value, dict):
        return False, "output_format", "compose must return a dictionary"
    if "accept" not in value or not isinstance(value["accept"], bool):
        return False, "output_format", "return dictionary must contain boolean accept"
    if value["accept"]:
        if "target" not in value or not isinstance(value["target"], str) or value["target"] == "":
            return False, "output_format", "accepted output must contain a non-empty string target"
    else:
        if "reason" not in value or not isinstance(value["reason"], str) or value["reason"] == "":
            return False, "output_format", "rejected output must contain a non-empty string reason"
    return True, "", ""


def _execute_worker(code: str, cases: Sequence[SandboxCase], result_queue: mp.Queue) -> None:
    env: JsonDict = {"__builtins__": ALLOWED_BUILTINS}
    try:
        exec(code, env, env)
        compose = env.get("compose")
        if not callable(compose):
            result_queue.put((False, "schema_error", "compose is not callable"))
            return
        for case in cases:
            first = compose(_copy_jsonish(case.components), _copy_jsonish(case.metadata))
            ok, category, message = _validate_output(first)
            if not ok:
                result_queue.put((False, category, f"{case.name}: {message}"))
                return
            second = compose(_copy_jsonish(case.components), _copy_jsonish(case.metadata))
            if first != second:
                result_queue.put((False, "nondeterministic", f"{case.name}: output changed on repeated calls"))
                return
            if case.expected_accept is not None and first["accept"] != case.expected_accept:
                result_queue.put((False, "property_test_failed", f"{case.name}: accept flag mismatch"))
                return
            if case.expected_target is not None:
                if not first.get("accept") or first.get("target") != case.expected_target:
                    result_queue.put((False, "property_test_failed", f"{case.name}: target mismatch"))
                    return
            if case.target_pattern is not None and first.get("accept"):
                if re.fullmatch(case.target_pattern, str(first.get("target", ""))) is None:
                    result_queue.put((False, "output_format", f"{case.name}: target does not match pattern"))
                    return
        result_queue.put((True, "", ""))
    except BaseException as exc:  # noqa: BLE001 - sandbox reports sanitized runtime failures.
        result_queue.put((False, "runtime_error", f"{type(exc).__name__}: {str(exc)[:200]}"))


def _execute_outputs_worker(code: str, cases: Sequence[SandboxCase], result_queue: mp.Queue) -> None:
    env: JsonDict = {"__builtins__": ALLOWED_BUILTINS}
    try:
        exec(code, env, env)
        compose = env.get("compose")
        if not callable(compose):
            result_queue.put((False, "schema_error", "compose is not callable", ()))
            return
        outputs: List[JsonDict] = []
        for case in cases:
            first = compose(_copy_jsonish(case.components), _copy_jsonish(case.metadata))
            ok, category, message = _validate_output(first)
            if not ok:
                result_queue.put((False, category, f"{case.name}: {message}", ()))
                return
            second = compose(_copy_jsonish(case.components), _copy_jsonish(case.metadata))
            if first != second:
                result_queue.put((False, "nondeterministic", f"{case.name}: output changed on repeated calls", ()))
                return
            if case.target_pattern is not None and first.get("accept"):
                if re.fullmatch(case.target_pattern, str(first.get("target", ""))) is None:
                    result_queue.put((False, "output_format", f"{case.name}: target does not match pattern", ()))
                    return
            outputs.append(dict(first))
        result_queue.put((True, "", "", tuple(outputs)))
    except BaseException as exc:  # noqa: BLE001 - sandbox reports sanitized runtime failures.
        result_queue.put((False, "runtime_error", f"{type(exc).__name__}: {str(exc)[:200]}", ()))


def validate_program(
    code: str,
    *,
    cases: Sequence[SandboxCase] = (),
    timeout_seconds: float = 1.0,
) -> ProgramValidationResult:
    ok, category, message = _static_validate(code)
    if not ok:
        return ProgramValidationResult(False, category=category, message=message, code=code)

    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_execute_worker, args=(code, list(cases), result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return ProgramValidationResult(False, category="timeout", message="sandbox execution timed out", code=code)
    try:
        valid, runtime_category, runtime_message = result_queue.get_nowait()
    except queue.Empty:
        return ProgramValidationResult(False, category="runtime_error", message="sandbox produced no result", code=code)
    if not valid:
        return ProgramValidationResult(False, category=runtime_category, message=runtime_message, code=code)
    return ProgramValidationResult(True, code=code)


def execute_program_cases(
    code: str,
    *,
    cases: Sequence[SandboxCase],
    timeout_seconds: float = 5.0,
) -> ProgramExecutionResult:
    ok, category, message = _static_validate(code)
    if not ok:
        return ProgramExecutionResult(False, category=category, message=message)

    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_execute_outputs_worker, args=(code, list(cases), result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return ProgramExecutionResult(False, category="timeout", message="sandbox execution timed out")
    try:
        valid, runtime_category, runtime_message, outputs = result_queue.get_nowait()
    except queue.Empty:
        return ProgramExecutionResult(False, category="runtime_error", message="sandbox produced no result")
    if not valid:
        return ProgramExecutionResult(False, category=runtime_category, message=runtime_message)
    return ProgramExecutionResult(True, outputs=tuple(outputs))


def validate_program_with_repair(
    code: str,
    *,
    repair_callback: Optional[RepairCallback],
    cases: Sequence[SandboxCase] = (),
    timeout_seconds: float = 1.0,
    repair_attempts: int = 1,
) -> ProgramValidationResult:
    first = validate_program(code, cases=cases, timeout_seconds=timeout_seconds)
    if first.valid or repair_callback is None or repair_attempts <= 0:
        return first
    repaired_code = repair_callback(first.category, first.message, code)
    if not repaired_code:
        return ProgramValidationResult(
            False,
            category=first.category,
            message=first.message,
            code=code,
            original_category=first.category,
            original_message=first.message,
            repair_attempted=True,
        )
    repaired = validate_program(repaired_code, cases=cases, timeout_seconds=timeout_seconds)
    return ProgramValidationResult(
        repaired.valid,
        category=repaired.category,
        message=repaired.message,
        repaired=repaired.valid,
        code=code,
        repaired_code=repaired_code,
        original_category=first.category,
        original_message=first.message,
        repair_attempted=True,
    )
