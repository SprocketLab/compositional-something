from self.adaptive.attempts import attempt_models
from self.adaptive.attempts.attempt_candidate_runtime import CandidateAttemptDeps
from self.adaptive.attempts.attempt_loop_runtime import AttemptLoopDeps, AttemptLoopResult


def test_attempt_runtime_paths_reexport_shared_models() -> None:
    assert CandidateAttemptDeps is attempt_models.CandidateAttemptDeps
    assert AttemptLoopDeps is attempt_models.AttemptLoopDeps
    assert AttemptLoopResult is attempt_models.AttemptLoopResult
