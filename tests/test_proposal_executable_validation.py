from __future__ import annotations

from self.adaptive import proposal_executable_validation, proposal_runtime


def test_extract_python_code_handles_markdown_fences_and_prefix_text():
    raw = "Here is code:\n```python\ndef compose(left, right):\n    return left + right\n```"

    assert proposal_executable_validation._extract_python_code(raw) == (
        "def compose(left, right):\n    return left + right"
    )


def test_row_repair_output_accepts_line_fixture_and_rejects_bad_type():
    assert proposal_executable_validation._row_repair_output(
        {"repair_output_lines": ["def compose(left, right):", "    return left"]}
    ) == "def compose(left, right):\n    return left"

    try:
        proposal_executable_validation._row_repair_output({"repair_output_lines": "bad"})
    except ValueError as exc:
        assert "repair_output_lines must be a list" in str(exc)
    else:
        raise AssertionError("bad repair_output_lines should be rejected")


def test_proposal_runtime_reexports_executable_validation_helpers_for_compatibility():
    assert proposal_runtime._extract_python_code is proposal_executable_validation._extract_python_code
    assert proposal_runtime._row_payload is proposal_executable_validation._row_payload
    assert proposal_runtime._row_repair_output is proposal_executable_validation._row_repair_output
    assert proposal_runtime._repair_program_with_model is proposal_executable_validation._repair_program_with_model
    assert proposal_runtime.validate_executable_rows is proposal_executable_validation.validate_executable_rows
