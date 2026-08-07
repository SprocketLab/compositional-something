#!/usr/bin/env python3
"""Passage-level partition of the DROP-QDMR train split (plan §7.2).

DROP asks many questions about the same passage, so seed questions and
composition questions drawn from one passage share extractable content.  Every
partition therefore splits on `section_id`, never on the question.

The seed-source partition is not drawn here: it is the set of DROP train
passages BREAK does not cover, which `drop_qdmr_build.py` already separated
into `drop_seed_pool_train.jsonl`.  Taking the seed from outside BREAK's
coverage satisfies the anti-memorization rule at no cost to the composition
pool, and it is the only source with gold-labeled extraction supervision --
BREAK releases no gold intermediate values (plan §7.1).

What remains is splitting the BREAK-covered passages into composition-source
and internal-audit, stratified by domain so the `nfl` / `history` mix is the
same on both sides.

Writes:
  data/drop_partition.json   passage ids and draw parameters (tracked)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

AUDIT_FRACTION = 0.15
RNG_SEED = 13   # distinct from the eval-sampling seed (7) on purpose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdmr-train", type=Path, required=True)
    ap.add_argument("--seed-pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit-fraction", type=float, default=AUDIT_FRACTION)
    args = ap.parse_args()

    by_domain: dict[str, set[str]] = defaultdict(set)
    rows_per_passage: dict[str, int] = defaultdict(int)
    for row in map(json.loads, open(args.qdmr_train)):
        by_domain[row["domain"]].add(row["passage_id"])
        rows_per_passage[row["passage_id"]] += 1

    seed_passages = {r["passage_id"] for r in map(json.loads, open(args.seed_pool))}
    covered = {p for ps in by_domain.values() for p in ps}
    overlap = seed_passages & covered
    if overlap:
        raise SystemExit(
            f"seed-source and composition-source share {len(overlap)} passages; "
            "the seed would have seen the passages it later pseudo-labels")

    rng = random.Random(RNG_SEED)
    audit: set[str] = set()
    for domain in sorted(by_domain):
        passages = sorted(by_domain[domain])
        n_audit = int(round(len(passages) * args.audit_fraction))
        audit.update(rng.sample(passages, n_audit))
    composition = sorted(covered - audit)

    part = {
        "audit_fraction": args.audit_fraction,
        "rng_seed": RNG_SEED,
        "n_covered_passages": len(covered),
        "n_composition_passages": len(composition),
        "n_audit_passages": len(audit),
        "n_seed_passages": len(seed_passages),
        "composition_rows": sum(rows_per_passage[p] for p in composition),
        "audit_rows": sum(rows_per_passage[p] for p in audit),
        "by_domain": {d: len(ps) for d, ps in sorted(by_domain.items())},
        "composition_passages": composition,
        "audit_passages": sorted(audit),
    }
    args.out.write_text(json.dumps(part, indent=1))
    print(f"covered {len(covered)} passages -> composition {len(composition)} "
          f"({part['composition_rows']} DAGs) / audit {len(audit)} "
          f"({part['audit_rows']} DAGs); seed-source {len(seed_passages)} "
          f"passages, disjoint", flush=True)


if __name__ == "__main__":
    main()
