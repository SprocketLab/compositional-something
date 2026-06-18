"""CLI entrypoint dispatch for adaptive candidate-training workers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence


JsonDict = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class DriverEntrypointDeps:
    build_parser: Callable[[], argparse.ArgumentParser]
    run_controller_worker: Callable[[Path], JsonDict]
    run_candidate_worker: Callable[[Path], JsonDict]
    run_candidate_pack_worker: Callable[[Path], JsonDict]
    run: Callable[[argparse.Namespace], JsonDict]


def run_driver_entrypoint(argv: Optional[Sequence[str]], deps: DriverEntrypointDeps) -> None:
    parser = deps.build_parser()
    args = parser.parse_args(argv)
    if args.run_controller_worker:
        if args.controller_worker_spec is None:
            parser.error("--controller-worker-spec is required with --run-controller-worker")
        summary = deps.run_controller_worker(args.controller_worker_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.run_candidate_worker:
        if args.candidate_worker_spec is None:
            parser.error("--candidate-worker-spec is required with --run-candidate-worker")
        summary = deps.run_candidate_worker(args.candidate_worker_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.run_candidate_pack_worker:
        if args.candidate_worker_pack_spec is None:
            parser.error("--candidate-worker-pack-spec is required with --run-candidate-pack-worker")
        summary = deps.run_candidate_pack_worker(args.candidate_worker_pack_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.task is None:
        parser.error("--task is required unless --run-candidate-worker is set")
    summary = deps.run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
