from self.adaptive import attempts
from self.adaptive.attempts import CandidateAttemptDeps
from self.adaptive.attempts import AttemptLoopDeps, AttemptLoopResult


def test_attempt_runtime_paths_reexport_shared_models() -> None:
    assert CandidateAttemptDeps is attempts.CandidateAttemptDeps
    assert AttemptLoopDeps is attempts.AttemptLoopDeps
    assert AttemptLoopResult is attempts.AttemptLoopResult
