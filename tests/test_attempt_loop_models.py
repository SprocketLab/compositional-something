from self.core import attempt_loop_models
from self.core.attempt_candidate_runtime import CandidateAttemptDeps
from self.core.attempt_loop_runtime import AttemptLoopDeps, AttemptLoopResult


def test_attempt_runtime_paths_reexport_shared_models() -> None:
    assert CandidateAttemptDeps is attempt_loop_models.CandidateAttemptDeps
    assert AttemptLoopDeps is attempt_loop_models.AttemptLoopDeps
    assert AttemptLoopResult is attempt_loop_models.AttemptLoopResult
