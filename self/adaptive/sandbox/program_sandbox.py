#!/usr/bin/env python3
"""Restricted validation and execution for model-generated composition code."""

from __future__ import annotations

import ast
import multiprocessing as mp
import queue
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.adaptive.sandbox.program_sandbox_cases import build_addition_program_cases, build_run_length_program_cases
from self.adaptive.sandbox.program_sandbox_models import (
    JsonDict,
    ProgramExecutionResult,
    ProgramValidationResult,
    RepairCallback,
    SandboxCase,
)

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
