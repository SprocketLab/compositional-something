from self.core import program_sandbox, program_sandbox_cases, program_sandbox_models


def test_program_sandbox_model_reexports() -> None:
    assert program_sandbox.SandboxCase is program_sandbox_models.SandboxCase
    assert program_sandbox.ProgramValidationResult is program_sandbox_models.ProgramValidationResult
    assert program_sandbox.ProgramExecutionResult is program_sandbox_models.ProgramExecutionResult


def test_program_sandbox_case_builder_reexports() -> None:
    assert program_sandbox.build_run_length_program_cases is program_sandbox_cases.build_run_length_program_cases
    assert program_sandbox.build_addition_program_cases is program_sandbox_cases.build_addition_program_cases


def test_program_sandbox_case_builders_produce_cases() -> None:
    run_length_cases = program_sandbox_cases.build_run_length_program_cases(random_seed=3, random_count=2)
    addition_cases = program_sandbox_cases.build_addition_program_cases()

    assert len(run_length_cases) == 8
    assert all(isinstance(case, program_sandbox_models.SandboxCase) for case in run_length_cases)
    assert [case.name for case in addition_cases] == ["concat_no_carry", "malformed_prediction"]
