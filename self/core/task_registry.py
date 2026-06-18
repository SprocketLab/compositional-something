"""Task lookup for adaptive self-improvement runs."""

from __future__ import annotations

from typing import Any

from self.tasks.addition import AdditionTask
from self.tasks.run_length import RunLengthTask


def task_for_name(task_name: str) -> Any:
    if task_name == "addition":
        return AdditionTask()
    if task_name == "run_length":
        return RunLengthTask()
    raise ValueError(f"Unsupported task={task_name!r}.")
