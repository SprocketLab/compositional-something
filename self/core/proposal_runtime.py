"""Runtime proposal generation and validation for adaptive self-improvement."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self.core.models import ExecutableProposal
from self.core.program_sandbox import (
    ProgramValidationResult,
    SandboxCase,
    build_addition_program_cases,
    build_run_length_program_cases,
    validate_program_with_repair,
)
from self.core.proposals import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    ConfigProposal,
    PromptBundle,
    extract_json_object,
    load_fixture_proposals,
    normalized_config_completion,
    parse_config_proposal,
    proposal_output_schema,
    proposal_payload_for_schema,
    render_program_repair_prompt,
    validate_config_prediction,
)
from self.core.data_io import sanitize_json_value
from self.core.evaluation import build_generation_encodings
from self.core.model_io import instantiate_model_and_tokenizer
from self.tasks.bit_common import (
    RUN_LENGTH_TARGET_RUN_STATE,
    normalize_bit_target_mode,
)

JsonDict = Dict[str, Any]


def target_format_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return "a non-negative integer string formed only from component prediction strings"
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
        return "max_run|prefix_run|suffix_run"
    raise ValueError(f"Unsupported task={task_name!r}.")


def component_prediction_examples_for_task(task_name: str, args: argparse.Namespace) -> List[str]:
    if task_name == "addition":
        return ["46", "064", "1002"]
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return ["3|0|2|1|1", "5|1|5|1|5"]
        return ["3|2|1", "5|5|5"]
    raise ValueError(f"Unsupported task={task_name!r}.")


def program_validation_cases(task_name: str, args: argparse.Namespace) -> List[SandboxCase]:
    if task_name == "addition":
        return build_addition_program_cases()
    if task_name == "run_length":
        if normalize_bit_target_mode(args) != RUN_LENGTH_TARGET_RUN_STATE:
            return []
        return build_run_length_program_cases(random_seed=args.seed, random_count=8)
    raise ValueError(f"Unsupported task={task_name!r}.")


def choose_default_program_pair(
    *,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    allow_repeat_targets: bool,
) -> ConfigProposal:
    for target in range(frontier_min, frontier_max + 1):
        if not allow_repeat_targets and target in source_sizes:
            continue
        pairs = [(left, target - left) for left in sorted(source_sizes) if target - left in source_sizes]
        if not pairs:
            continue
        left, right = sorted(pairs, key=lambda pair: (abs(pair[0] - pair[1]), pair[0]))[0]
        return ConfigProposal(left=left, right=right, guard="none", target=target)
    raise ValueError(
        "No driver-selected program pair is available from the current source pool "
        f"for frontier={frontier_min}..{frontier_max}."
    )


def render_program_candidate_prompt(
    *,
    args: argparse.Namespace,
    round_index: int,
    current_checkpoint: str,
    current_source: Mapping[str, Any],
    allowed_target_frontier: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
    default_pair: Optional[ConfigProposal],
) -> PromptBundle:
    target_format = target_format_for_task(args.task, args)
    examples = component_prediction_examples_for_task(args.task, args)
    common = (
        f"Task: {args.task}\n\n"
        "Current round context:\n"
        f"- round_index: {round_index}\n"
        f"- current_source_slices: {json.dumps(sanitize_json_value(dict(current_source)), sort_keys=True)}\n"
        f"- allowed_target_frontier: {json.dumps(sanitize_json_value(dict(allowed_target_frontier)), sort_keys=True)}\n"
        f"- model: {current_checkpoint}\n"
        "Aggregate diagnostics:\n"
        f"{json.dumps(sanitize_json_value(dict(aggregate_metrics)), sort_keys=True, indent=2)}\n\n"
        "Self-labeling rule:\n"
        "- The target label must be derived from current-model component predictions.\n"
        "- Do not use oracle labels or write a direct task solver from raw target inputs.\n"
        "- The driver constructs target inputs and held-out evaluation; your code only composes pseudo-labels.\n\n"
        "Function contract:\n"
        "def compose(components, metadata):\n"
        "    ...\n\n"
        "Each component is a dictionary with size, input_id, prediction, and metadata.\n"
        "The input_id is opaque. Use component predictions and sizes.\n"
        f"Target output format: {target_format}\n"
        f"Example component predictions: {json.dumps(examples)}\n"
        '- Return {"accept": True, "target": "<target string>"} or '
        '{"accept": False, "reason": "<short reason>"}.\n\n'
        "Sandbox rules:\n"
        "- exactly one top-level compose function\n"
        "- no imports, filesystem, network, subprocess, eval, exec, compile, open, globals, locals, vars, dir, getattr, setattr, delattr, or __import__\n"
        "- deterministic output on repeated calls\n"
    )
    if args.condition == "program":
        if default_pair is None:
            raise ValueError("program condition requires a driver-selected default pair")
        system = (
            "You are generating a restricted Python composition program.\n"
            "Output only Python code. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "Driver-selected source policy:\n"
            f"- left: {default_pair.left}\n"
            f"- right: {default_pair.right}\n"
            f"- target: {default_pair.target}\n"
            "- You only write the compose function for this source pair.\n\n"
            "Output only Python code."
        )
        return PromptBundle(system=system, user=user)

    if args.condition == "policy":
        system = (
            "You are generating a source-selection policy plus a restricted Python composer.\n"
            "Output only valid JSON. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "JSON schema:\n"
            "{\n"
            '  "left": integer source slice size,\n'
            '  "right": integer source slice size,\n'
            '  "guard": "none" or a listed task guard,\n'
            '  "code": "Python code defining compose(components, metadata)",\n'
            '  "notes": "optional short rationale"\n'
            "}\n\n"
            "Policy constraints:\n"
            "- left and right must be current source slices.\n"
            "- left + right must be inside the allowed target frontier.\n"
            "- Choose slices based on current slice accuracy and expected self-labeling reliability.\n\n"
            "Output only JSON."
        )
        return PromptBundle(system=system, user=user)

    if args.condition == "meta":
        system = (
            "You are generating an intermediate representation policy plus a restricted Python composer.\n"
            "Output only valid JSON. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "JSON schema:\n"
            "{\n"
            '  "left": integer source slice size,\n'
            '  "right": integer source slice size,\n'
            '  "guard": "none" or a listed task guard,\n'
            '  "representation": "short name for the intermediate state you are composing through",\n'
            '  "target_format": "final target format your compose function returns",\n'
            '  "code": "Python code defining compose(components, metadata)",\n'
            '  "notes": "optional short rationale"\n'
            "}\n\n"
            "Meta-composition constraints:\n"
            "- You may invent the intermediate representation used inside compose.\n"
            f"- The final returned target must still match the task target format: {target_format}.\n"
            "- left and right must be current source slices, and left + right must be inside the allowed target frontier.\n\n"
            "Output only JSON."
        )
        return PromptBundle(system=system, user=user)

    raise ValueError(f"Unsupported condition={args.condition!r}.")


def _raw_output(row: Mapping[str, Any]) -> Any:
    if "code_lines" in row:
        code_lines = row["code_lines"]
        if isinstance(code_lines, list):
            return "\n".join(str(line) for line in code_lines)
    for key in ("raw_output", "output", "completion", "proposal", "code"):
        if key in row:
            return row[key]
    return row


def _extract_python_code(raw: Any, payload: Optional[Mapping[str, Any]] = None) -> str:
    if payload is not None:
        if "code_lines" in payload and isinstance(payload["code_lines"], list):
            return "\n".join(str(line) for line in payload["code_lines"]).strip()
        for key in ("code", "completion", "program"):
            if key in payload:
                return str(payload[key]).strip()
    text = str(raw).strip()
    if "```" in text:
        pieces = text.split("```")
        for piece in pieces:
            stripped = piece.strip()
            if stripped.startswith("python"):
                stripped = stripped[len("python") :].strip()
            if "def compose" in stripped:
                text = stripped
                break
    start = text.find("def compose")
    if start > 0:
        text = text[start:]
    return text.strip()


def _row_payload(raw: Any) -> Optional[JsonDict]:
    if isinstance(raw, dict):
        return dict(raw)
    return extract_json_object(str(raw))


def _row_repair_output(row: Mapping[str, Any]) -> Optional[str]:
    if "repair_output_lines" in row:
        repair_lines = row["repair_output_lines"]
        if not isinstance(repair_lines, list):
            raise ValueError("repair_output_lines must be a list of strings")
        return "\n".join(str(line) for line in repair_lines)
    for key in ("repair_output", "repaired_output", "repair_code"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _repair_program_with_model(
    *,
    args: argparse.Namespace,
    current_model: Optional[Any],
    current_tokenizer: Optional[Any],
    category: str,
    message: str,
    code: str,
) -> Optional[str]:
    if current_model is None or current_tokenizer is None:
        return None
    repair_prompt = render_program_repair_prompt(
        task_name=args.task,
        target_format=target_format_for_task(args.task, args),
        failure_category=category,
        failure_summary=message,
        previous_program=code,
    )
    rows = generate_proposals_from_model(
        model=current_model,
        tokenizer=current_tokenizer,
        prompt=repair_prompt,
        num_candidates=1,
        max_new_tokens=args.proposal_max_new_tokens,
        temperature=args.proposal_temperature,
        top_p=args.proposal_top_p,
    )
    if not rows:
        return None
    return _extract_python_code(rows[0].get("raw_output", ""))


def _rows_for_round(
    rows: Sequence[Mapping[str, Any]],
    round_index: int,
    *,
    attempt_index: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    if attempt_index is not None:
        attempt_matching = [
            row for row in rows if "attempt" in row and int(row.get("attempt", -1)) == attempt_index
        ]
        if attempt_matching:
            return attempt_matching
    matching = [row for row in rows if int(row.get("round", round_index)) == round_index]
    return matching if matching else list(rows)


def generate_proposals_from_model(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: PromptBundle,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[JsonDict]:
    device = next(model.parameters()).device
    encodings = build_generation_encodings(tokenizer, [prompt.text()], device)
    rows: List[JsonDict] = []
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for idx in range(num_candidates):
            generation_kwargs: Dict[str, Any] = {
                **encodings,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
            prompt_width = encodings["input_ids"].shape[1]
            decoded = tokenizer.decode(output_ids[0, prompt_width:].tolist(), skip_special_tokens=True)
            rows.append({"id": f"model_candidate_{idx}", "raw_output": decoded})
    if model_was_training:
        model.train()
    return rows


def validate_config_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
) -> List[JsonDict]:
    source_min = min(source_sizes) if source_sizes else args.initial_min_size
    source_max = max(source_sizes) if source_sizes else args.initial_max_size
    guards = DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"]
    schema = proposal_output_schema(args)
    results: List[JsonDict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        raw = _raw_output(row)
        proposal_raw, prediction_payload, pre_category, pre_message = proposal_payload_for_schema(
            raw,
            schema,
        )
        if pre_category is not None:
            completion = ""
            validation_valid = False
            category = pre_category
            message = str(pre_message)
            proposal_payload = None
            parsed_prediction = None
            duplicate = False
            repeat_target = False
            results.append(
                sanitize_json_value(
                    {
                        "proposal_index": index,
                        "id": row.get("id"),
                        "raw_output": raw,
                        "valid": validation_valid,
                        "validation_category": category,
                        "validation_message": message,
                        "parsed_proposal": proposal_payload,
                        "parsed_prediction": parsed_prediction,
                        "proposal_output_schema": schema,
                        "completion": completion,
                        "duplicate": duplicate,
                        "repeat_target": repeat_target,
                    }
                )
            )
            continue
        validation = parse_config_proposal(
            proposal_raw,
            task_name=args.task,
            source_min_allowed=source_min,
            source_max_allowed=source_max,
            source_sizes_allowed=sorted(source_sizes),
            frontier_min_allowed=frontier_min,
            frontier_max_allowed=frontier_max,
            guards=guards,
        )
        parsed_prediction = None
        prediction_error = None
        if validation.valid:
            parsed_prediction, prediction_error = validate_config_prediction(
                prediction_payload=prediction_payload,
                proposal=validation.proposal,
                schema=schema,
            )
        validation_valid = bool(validation.valid and prediction_error is None)
        category = validation.category
        message = validation.message
        if validation.valid and prediction_error is not None:
            category = "schema_error"
            message = prediction_error
        completion = (
            normalized_config_completion(
                proposal=validation.proposal,
                prediction=parsed_prediction,
                schema=schema,
            )
            if validation_valid
            else ""
        )
        duplicate = bool(completion and completion in seen)
        if completion:
            seen.add(completion)
        repeat_target = bool(validation_valid and validation.proposal.target in source_sizes)
        proposal_payload = validation.proposal.to_json_dict() if validation.valid else None
        results.append(
            sanitize_json_value(
                {
                    "proposal_index": index,
                    "id": row.get("id"),
                    "raw_output": raw,
                    "valid": validation_valid,
                    "validation_category": category,
                    "validation_message": message,
                    "parsed_proposal": proposal_payload,
                    "parsed_prediction": parsed_prediction,
                    "proposal_output_schema": schema,
                    "completion": completion,
                    "duplicate": duplicate,
                    "repeat_target": repeat_target,
                }
            )
        )
    return results


def validate_executable_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    default_pair: Optional[ConfigProposal],
    current_model: Optional[Any] = None,
    current_tokenizer: Optional[Any] = None,
) -> List[JsonDict]:
    source_min = min(source_sizes) if source_sizes else args.initial_min_size
    source_max = max(source_sizes) if source_sizes else args.initial_max_size
    guards = set(DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"])
    results: List[JsonDict] = []
    seen: set[str] = set()
    cases = program_validation_cases(args.task, args)

    for index, row in enumerate(rows):
        raw = _raw_output(row)
        payload = _row_payload(raw)
        condition = str(row.get("condition", args.condition))
        row_raw_for_code: Any = raw
        if payload is None and isinstance(row, Mapping):
            payload = dict(row)
        if condition != args.condition:
            continue

        if args.condition == "program":
            if default_pair is None:
                validation_valid = False
                category = "range_error"
                message = "program condition has no driver-selected source pair"
                proposal_payload = None
                completion = ""
                duplicate = False
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": validation_valid,
                            "validation_category": category,
                            "validation_message": message,
                            "parsed_proposal": proposal_payload,
                            "completion": completion,
                            "duplicate": duplicate,
                        }
                    )
                )
                continue
            left = default_pair.left
            right = default_pair.right
            target = default_pair.target
            guard = "none"
        else:
            if payload is None:
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": False,
                            "validation_category": "parse_error",
                            "validation_message": f"{args.condition} proposal must be a JSON object",
                            "parsed_proposal": None,
                            "completion": "",
                            "duplicate": False,
                        }
                    )
                )
                continue
            try:
                left = int(payload["left"])
                right = int(payload["right"])
            except (KeyError, TypeError, ValueError):
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": False,
                            "validation_category": "schema_error",
                            "validation_message": f"{args.condition} proposal requires integer left and right",
                            "parsed_proposal": None,
                            "completion": "",
                            "duplicate": False,
                        }
                    )
                )
                continue
            target = left + right
            guard = str(payload.get("guard", "none"))
            row_raw_for_code = payload

        range_error = ""
        if left < source_min or left > source_max or right < source_min or right > source_max:
            range_error = "source slice is outside allowed source bounds"
        elif left not in source_sizes or right not in source_sizes:
            range_error = "source slice is not in the current source pool"
        elif target < frontier_min or target > frontier_max:
            range_error = "left + right target is outside the allowed frontier"
        elif not args.allow_repeat_targets and target in source_sizes:
            range_error = "target slice is already in the current source pool"
        elif guard not in guards:
            range_error = f"invalid guard={guard!r}"

        code = _extract_python_code(row_raw_for_code, payload if isinstance(row_raw_for_code, Mapping) else None)
        if range_error:
            results.append(
                sanitize_json_value(
                    {
                        "proposal_index": index,
                        "id": row.get("id"),
                        "raw_output": raw,
                        "valid": False,
                        "validation_category": "range_error" if "guard" not in range_error else "enum_error",
                        "validation_message": range_error,
                        "parsed_proposal": None,
                        "completion": "",
                        "duplicate": False,
                    }
                )
            )
            continue

        repair_prompt_text: Optional[str] = None

        def repair_callback(category: str, message: str, previous_program: str) -> Optional[str]:
            nonlocal repair_prompt_text
            repair_prompt = render_program_repair_prompt(
                task_name=args.task,
                target_format=target_format_for_task(args.task, args),
                failure_category=category,
                failure_summary=message,
                previous_program=previous_program,
            )
            repair_prompt_text = repair_prompt.text()
            fixture_repair = _row_repair_output(row)
            if fixture_repair is not None:
                return _extract_python_code(fixture_repair)
            return _repair_program_with_model(
                args=args,
                current_model=current_model,
                current_tokenizer=current_tokenizer,
                category=category,
                message=message,
                code=previous_program,
            )

        validation: ProgramValidationResult = validate_program_with_repair(
            code,
            repair_callback=repair_callback,
            cases=cases,
            timeout_seconds=args.program_timeout_seconds,
            repair_attempts=args.repair_attempts,
        )
        valid_code = validation.repaired_code if validation.valid and validation.repaired_code else code
        completion = valid_code if validation.valid else ""
        duplicate = bool(completion and completion in seen)
        if completion:
            seen.add(completion)
        proposal = None
        if validation.valid:
            proposal = ExecutableProposal(
                left=left,
                right=right,
                guard=guard,
                target=target,
                code=valid_code,
                condition=args.condition,
                notes=str((payload or {}).get("notes", "")),
                representation=str((payload or {}).get("representation", "")),
                target_format=str((payload or {}).get("target_format", target_format_for_task(args.task, args))),
                repaired=validation.repaired,
                original_validation_category=validation.original_category,
                original_validation_message=validation.original_message,
            )
        results.append(
            sanitize_json_value(
                {
                    "proposal_index": index,
                    "id": row.get("id"),
                    "raw_output": raw,
                    "valid": validation.valid,
                    "validation_category": validation.category,
                    "validation_message": validation.message,
                    "original_validation_category": validation.original_category,
                    "original_validation_message": validation.original_message,
                    "repair_attempted": validation.repair_attempted,
                    "repair_prompt": repair_prompt_text,
                    "repaired": validation.repaired,
                    "repaired_output": validation.repaired_code,
                    "parsed_proposal": proposal.to_json_dict() if proposal is not None else None,
                    "completion": completion,
                    "duplicate": duplicate,
                }
            )
        )
    return results


def validate_proposal_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    default_pair: Optional[ConfigProposal],
    current_model: Optional[Any] = None,
    current_tokenizer: Optional[Any] = None,
) -> List[JsonDict]:
    if args.condition == "config":
        return validate_config_rows(
            rows=rows,
            args=args,
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
        )
    return validate_executable_rows(
        rows=rows,
        args=args,
        source_sizes=source_sizes,
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        default_pair=default_pair,
        current_model=current_model,
        current_tokenizer=current_tokenizer,
    )


def load_or_generate_proposal_rows(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    current_model: AutoModelForCausalLM,
    current_tokenizer: AutoTokenizer,
    round_index: int,
    attempt_index: Optional[int] = None,
) -> List[JsonDict]:
    if args.proposal_fixture_jsonl is not None:
        rows = _rows_for_round(
            load_fixture_proposals(args.proposal_fixture_jsonl),
            round_index,
            attempt_index=attempt_index,
        )
        return [dict(row) for row in rows[: args.num_candidates]]

    if args.proposal_model_name == "current":
        return generate_proposals_from_model(
            model=current_model,
            tokenizer=current_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )

    proposal_model, proposal_tokenizer = instantiate_model_and_tokenizer(
        args.proposal_model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe="none",
    )
    try:
        return generate_proposals_from_model(
            model=proposal_model,
            tokenizer=proposal_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )
    finally:
        del proposal_model
        del proposal_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
