#!/usr/bin/env python3
"""SParC/Spider preprocessing: canonical records, monotone coverage, partition.

Everything downstream reads this script's outputs (all under data/):

  sparc_sequences.jsonl     gold-free canonical sequences (plan §8)
  sparc_gold.jsonl          audit-only gold turn/final SQL (separate file by design)
  sparc_spider_singles.jsonl  Spider questions for seed + transfer eval
  sparc_gold_spider.jsonl   audit-only gold for the singles
  sparc_schemas.json        serialized schema + token length + exclusion flag per db
  sparc_partition.json      database-level 30/60/10 split over TRAIN dbs (plan §7)
  sparc_data_report.json    monotone coverage, dedup, exclusions, tripwires

Tripwires printed loudly (plan §19):
  Risk 2 -- monotone 2-turn composition-source sequences < 1500 means the
            Spider-SS fallback decision must be made before any GPU job;
  Risk 4 -- schema exclusion rate over the context budget.

Gold SQL is written ONLY to the sparc_gold* files; generation scripts receive
them through an explicit --gold flag used inside measurement blocks (plan §8).

Run on the login node:
  PYTHONPATH=. $PY reports/composition_screen/sparc_data.py \
      --raw-root reports/composition_screen/data/sparc_raw \
      --out-dir reports/composition_screen/data \
      --model "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/<hash> \
      --audit-scorer
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "spider_eval"))

from evaluation import Evaluator                     # vendored, see its README
from process_sql import Schema as PSchema, get_sql

from self.coding.sparc_composition import (
    load_schemas,
    monotone_sequence,
    serialize_schema,
    suite_correct,
    suite_paths,
)

FRACTIONS = (0.30, 0.60, 0.10)      # seed / composition / audit over train dbs
RNG_SEED = 13                       # partition seed, distinct from eval seed 7
SCHEMA_MARGIN = 512                 # prompt tokens reserved beyond the schema

_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+(\.\d+)?\b")


def dedup_key(turns: list[str]) -> str:
    return "||".join(_LITERAL_RE.sub("?", t.lower()).strip() for t in turns)


def hardness_of(query: str, pschema: PSchema, evaluator: Evaluator) -> str:
    try:
        return evaluator.eval_hardness(get_sql(pschema, query))
    except Exception:
        return "unknown"


def mutate_for_audit(sql: str, rng: random.Random) -> str | None:
    """A denotation-changing edit for the scorer discrimination check."""
    if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
        return re.sub(r"\bLIMIT\s+\d+", "LIMIT 9999", sql, flags=re.IGNORECASE)
    if re.search(r"\bDESC\b", sql, re.IGNORECASE):
        return re.sub(r"\bDESC\b", "ASC", sql, count=1, flags=re.IGNORECASE)
    m = re.search(r"= *'([^']+)'", sql)
    if m:
        return sql.replace(m.group(0), "= 'zzz_no_such_value'", 1)
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql + " LIMIT 1"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", required=True,
                    help="tokenizer path (CPU-only load) for schema lengths")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--value-samples", type=int, default=2)
    ap.add_argument("--audit-scorer", action="store_true")
    ap.add_argument("--audit-sample", type=int, default=300)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.raw_root

    tables = json.load(open(raw / "spider" / "tables.json"))
    schemas = load_schemas(tables)
    db_root = raw / "spider" / "database"
    ts_root = raw / "testsuite"
    evaluator = Evaluator()
    pschemas = {
        e["db_id"]: PSchema({t.lower(): [c.lower() for c in schemas[e["db_id"]].tables[t]]
                             for t in schemas[e["db_id"]].tables})
        for e in tables}

    spider_train = (json.load(open(raw / "spider" / "train_spider.json"))
                    + json.load(open(raw / "spider" / "train_others.json")))
    spider_dev = json.load(open(raw / "spider" / "dev.json"))
    train_dbs = sorted({r["db_id"] for r in spider_train})
    dev_dbs = sorted({r["db_id"] for r in spider_dev})
    assert not set(train_dbs) & set(dev_dbs), "train/dev db overlap"

    # --- canonical sequences + gold (plan §8) -------------------------------
    seq_rows, gold_rows = [], []
    stats = {"skipped_short": 0, "duplicates": 0}
    mono_counts: dict = defaultdict(lambda: [0, 0])   # (split,k) -> [monotone, total]
    reason_counts: Counter = Counter()
    seen_keys: set = set()
    for split in ("train", "dev"):
        records = json.load(open(raw / "sparc" / f"{split}.json"))
        for i, rec in enumerate(records):
            turns = rec["interaction"]
            if len(turns) < 2:
                stats["skipped_short"] += 1
                continue
            db_id = rec["database_id"]
            sid = f"sparc-{split}-{i:05d}"
            final_q = (rec.get("final") or {}).get("utterance", "").strip()
            final_sql = (rec.get("final") or {}).get("query", "").strip()
            assert final_q and final_sql, f"{sid}: missing final intent"
            gold_sqls = [t["query"].strip() for t in turns]
            verdicts = monotone_sequence(gold_sqls, schemas[db_id])
            turn_mono = [v.monotone for v in verdicts]
            reasons = [v.reason for v in verdicts]
            for r in reasons:
                if r:
                    reason_counts[r.split(":")[0]] += 1
            usable = not any(
                r and ("parse" in r or "schema" in r) for r in reasons)
            key = f"{db_id}::{dedup_key([t['utterance'] for t in turns])}"
            duplicate = key in seen_keys
            seen_keys.add(key)
            stats["duplicates"] += duplicate
            k = len(turns)
            k_bucket = "4+" if k >= 4 else str(k)
            mono_counts[(split, k_bucket)][1] += 1
            mono_counts[(split, k_bucket)][0] += all(turn_mono)
            seq_rows.append({
                "sequence_id": sid, "db_id": db_id, "split": split,
                "turns": [{"turn_id": j + 1, "question": t["utterance"].strip()}
                          for j, t in enumerate(turns)],
                "final_intent_question": final_q,
                "turn_count": k, "monotone": all(turn_mono),
                "turn_monotone": turn_mono, "monotone_reasons": reasons,
                "hardness": hardness_of(final_sql, pschemas[db_id], evaluator),
                "usable": usable and not duplicate,
                "duplicate": duplicate,
            })
            gold_rows.append({"sequence_id": sid, "turn_sqls": gold_sqls,
                              "final_sql": final_sql})

    with open(args.out_dir / "sparc_sequences.jsonl", "w") as f:
        for r in seq_rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out_dir / "sparc_gold.jsonl", "w") as f:
        for r in gold_rows:
            f.write(json.dumps(r) + "\n")

    # --- Spider singles ------------------------------------------------------
    singles, singles_gold = [], []
    for split, records in (("train", spider_train), ("dev", spider_dev)):
        for i, rec in enumerate(records):
            sid = f"spider-{split}-{i:05d}"
            singles.append({
                "id": sid, "db_id": rec["db_id"], "split": split,
                "question": rec["question"].strip(),
                "hardness": evaluator.eval_hardness(rec["sql"])})
            singles_gold.append({"id": sid, "query": rec["query"].strip()})
    with open(args.out_dir / "sparc_spider_singles.jsonl", "w") as f:
        for r in singles:
            f.write(json.dumps(r) + "\n")
    with open(args.out_dir / "sparc_gold_spider.jsonl", "w") as f:
        for r in singles_gold:
            f.write(json.dumps(r) + "\n")

    # --- schema serializations + token lengths (plan §19 Risk 4) -------------
    from self.coding.training import load_qwen_tokenizer
    tok = load_qwen_tokenizer(args.model)
    schema_out, excluded = {}, []
    for db_id, schema in schemas.items():
        text = serialize_schema(schema, db_root / db_id / f"{db_id}.sqlite",
                                value_samples=args.value_samples)
        n_tokens = len(tok(text)["input_ids"])
        is_excluded = n_tokens + SCHEMA_MARGIN > args.max_length
        excluded += [db_id] if is_excluded else []
        schema_out[db_id] = {
            "text": text, "n_tokens": n_tokens, "excluded": is_excluded,
            "n_suite_instances": len(suite_paths(ts_root, db_root, db_id)),
            "has_distilled_suite": (ts_root / db_id).is_dir(),
        }
    (args.out_dir / "sparc_schemas.json").write_text(
        json.dumps(schema_out, indent=1))

    # --- database-level partition (plan §7) ----------------------------------
    rng = random.Random(RNG_SEED)
    shuffled = train_dbs[:]
    rng.shuffle(shuffled)
    n_seed = round(FRACTIONS[0] * len(shuffled))
    n_comp = round(FRACTIONS[1] * len(shuffled))
    seed_dbs = sorted(shuffled[:n_seed])
    comp_dbs = sorted(shuffled[n_seed:n_seed + n_comp])
    audit_dbs = sorted(shuffled[n_seed + n_comp:])
    assert not (set(seed_dbs) & set(comp_dbs) & set(audit_dbs))
    bucket_of = {**{d: "seed" for d in seed_dbs},
                 **{d: "composition" for d in comp_dbs},
                 **{d: "audit" for d in audit_dbs}}
    n_seq = Counter(bucket_of.get(r["db_id"], "dev") for r in seq_rows)
    partition = {"fractions": FRACTIONS, "rng_seed": RNG_SEED,
                 "seed_dbs": seed_dbs, "composition_dbs": comp_dbs,
                 "audit_dbs": audit_dbs, "dev_dbs": dev_dbs,
                 "n_sequences_per_bucket": dict(n_seq)}
    (args.out_dir / "sparc_partition.json").write_text(
        json.dumps(partition, indent=1))

    # --- coverage report and tripwires ---------------------------------------
    comp_mono_by_k: Counter = Counter()
    for r in seq_rows:
        if (r["split"] == "train" and r["usable"] and r["monotone"]
                and bucket_of.get(r["db_id"]) == "composition"
                and not schema_out[r["db_id"]]["excluded"]):
            comp_mono_by_k["4+" if r["turn_count"] >= 4 else str(r["turn_count"])] += 1
    dev_mono_by_k: Counter = Counter()
    for r in seq_rows:
        if (r["split"] == "dev" and r["usable"] and r["monotone"]
                and not schema_out[r["db_id"]]["excluded"]):
            dev_mono_by_k["4+" if r["turn_count"] >= 4 else str(r["turn_count"])] += 1

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_sequences": len(seq_rows), **stats,
        "monotone_fraction": {
            f"{split}/k{k}": {"monotone": m, "total": t, "fraction": m / t}
            for (split, k), (m, t) in sorted(mono_counts.items())},
        "nonmonotone_reasons": dict(reason_counts.most_common()),
        "schema_lengths": {
            "min": min(v["n_tokens"] for v in schema_out.values()),
            "median": sorted(v["n_tokens"] for v in schema_out.values())[len(schema_out) // 2],
            "max": max(v["n_tokens"] for v in schema_out.values()),
            "excluded_dbs": excluded,
            "exclusion_rate": len(excluded) / len(schema_out)},
        "suite_coverage": {
            "train_dbs_with_distilled_suite":
                sum(schema_out[d]["has_distilled_suite"] for d in train_dbs),
            "dev_dbs_with_distilled_suite":
                sum(schema_out[d]["has_distilled_suite"] for d in dev_dbs)},
        "composition_source_monotone_by_k": dict(comp_mono_by_k),
        "dev_monotone_by_k": dict(dev_mono_by_k),
        "partition_sequences": dict(n_seq),
    }

    tripwires = []
    if comp_mono_by_k.get("2", 0) < 1500:
        tripwires.append(
            f"RISK 2 TRIPWIRE: only {comp_mono_by_k.get('2', 0)} monotone 2-turn "
            "composition-source sequences (< 1500). The Spider-SS fallback "
            "decision (plan §5.4) must be made before GPU jobs are submitted.")
    if report["schema_lengths"]["exclusion_rate"] > 0.15:
        tripwires.append(
            f"RISK 4 TRIPWIRE: schema exclusion rate "
            f"{report['schema_lengths']['exclusion_rate']:.2%} exceeds 15%.")
    report["tripwires"] = tripwires

    # --- scorer self-audit (module docstring contract) ------------------------
    if args.audit_scorer:
        rng_a = random.Random(RNG_SEED)
        by_seq = {g["sequence_id"]: g for g in gold_rows}
        pool = [r for r in seq_rows if r["usable"]]
        sample = rng_a.sample(pool, min(args.audit_sample, len(pool)))
        self_fail, mut_pass, mut_total, exec_fail = [], 0, 0, []
        for r in sample:
            gold = by_seq[r["sequence_id"]]["final_sql"]
            paths = suite_paths(ts_root, db_root, r["db_id"])
            if not suite_correct(paths, gold, gold):
                self_fail.append(r["sequence_id"])
                continue
            mut = mutate_for_audit(gold, rng_a)
            if mut is None:
                continue
            mut_total += 1
            mut_pass += suite_correct(paths, mut, gold)
        report["scorer_audit"] = {
            "n": len(sample), "gold_vs_gold_failures": self_fail,
            "mutated_accepted": mut_pass, "mutated_total": mut_total,
            "mutation_discrimination":
                1 - mut_pass / mut_total if mut_total else None}
        if self_fail:
            print(f"SCORER AUDIT: {len(self_fail)} gold-vs-gold failures: "
                  f"{self_fail[:10]}", flush=True)

    (args.out_dir / "sparc_data_report.json").write_text(
        json.dumps(report, indent=2))

    print(f"sequences: {len(seq_rows)} (skipped {stats['skipped_short']} short, "
          f"{stats['duplicates']} duplicates)", flush=True)
    for key, v in report["monotone_fraction"].items():
        print(f"  {key}: {v['monotone']}/{v['total']} = {v['fraction']:.3f}", flush=True)
    print(f"nonmonotone reasons: {report['nonmonotone_reasons']}", flush=True)
    print(f"schema tokens min/med/max: {report['schema_lengths']['min']}/"
          f"{report['schema_lengths']['median']}/{report['schema_lengths']['max']} "
          f"excluded: {excluded}", flush=True)
    print(f"suite coverage: {report['suite_coverage']}", flush=True)
    print(f"composition-source monotone by k: {dict(comp_mono_by_k)}", flush=True)
    print(f"dev monotone by k: {dict(dev_mono_by_k)}", flush=True)
    if args.audit_scorer:
        print(f"scorer audit: {report['scorer_audit']}", flush=True)
    for t in tripwires:
        print(f"\n*** {t}", flush=True)
    if not tripwires:
        print("\nno tripwires fired", flush=True)


if __name__ == "__main__":
    main()
