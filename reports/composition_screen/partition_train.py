#!/usr/bin/env python3
"""Partition the MuSiQue train split for Round 1.

Plan §6 / handoff §5b: seed-source and composition-source must be disjoint at
FULL-EXAMPLE level, or the seed has seen the sub-questions it will later
pseudo-label and the pseudo-labels are partly memorized.  The current headline
seed used all of train, so Round 1 needs a fresh seed trained only on the
30% cut this script draws.

Writes:
  data/musique_round1_partition.json  -- ids + draw parameters (tracked)
  data/musique_train_seed30.jsonl     -- the seed-source rows (untracked, big)

The composition-source is the complement; Round 1 derives it from the ids file
rather than a second jsonl so there is one source of truth.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FRACTION = 0.30
RNG_SEED = 13   # distinct from the eval-sampling seed (7) on purpose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train_data)]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate train ids"

    rng = random.Random(RNG_SEED)
    seed_ids = set(rng.sample(ids, int(len(ids) * FRACTION)))

    part = {"fraction": FRACTION, "rng_seed": RNG_SEED,
            "n_train": len(ids), "n_seed": len(seed_ids),
            "n_composition": len(ids) - len(seed_ids),
            "seed_ids": sorted(seed_ids)}
    (args.out_dir / "musique_round1_partition.json").write_text(
        json.dumps(part, indent=1))

    with open(args.out_dir / "musique_train_seed30.jsonl", "w") as f:
        for r in rows:
            if r["id"] in seed_ids:
                f.write(json.dumps(r) + "\n")
    print(f"partitioned {len(ids)} -> seed {len(seed_ids)} / "
          f"composition {len(ids) - len(seed_ids)}", flush=True)


if __name__ == "__main__":
    main()
