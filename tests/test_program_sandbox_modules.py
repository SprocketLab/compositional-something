from self.adaptive.sandbox import program_sandbox


def test_program_sandbox_case_builders_produce_cases() -> None:
    run_length_cases = program_sandbox.build_run_length_program_cases(random_seed=3, random_count=2)
    addition_cases = program_sandbox.build_addition_program_cases()

    assert len(run_length_cases) == 8
    assert all(isinstance(case, program_sandbox.SandboxCase) for case in run_length_cases)
    assert [case.name for case in addition_cases] == ["concat_no_carry", "malformed_prediction"]
