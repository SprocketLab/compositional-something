#!/usr/bin/env python3
"""Round 1, stage 1: pseudo-label k-turn sequences with the seed's own chain.

Runs the seed adapter's self-fed turn chain (plan §10) over monotone
composition-source sequences, applying the full guard stack (plan §11) with
TWO-VIEW generation: every turn is generated under both fixed prompt templates
(greedy each); template A's SQL advances the state, and the denotational
agreement between A and B on every test-suite instance is the L4 guard.

The chain always runs to the sink so that L1-L4 acceptance sets are all
measured in one job; `rejected_at_turn` records where the accept-level guard
would have stopped it (plan §10.3).  Guard levels (plan §11.4):

  L1  every turn produced extractable SQL
  L2  L1 + static: parses, schema-valid, typed literals, monotone edit
  L3  L2 + executes on every suite instance, shape sane, nonempty where the
      question presupposes existence
  L4  L3 + two-view denotational agreement

Also generates the seed's DIRECT answer per sequence (the B1 self-distillation
label), guarded at the final-query analog of L3 (plan §15 B1).  Gold SQL is
read only inside the measurement block: guard precision/recall, per-turn
correctness, first-error turn (plan §11.4: never for acceptance).

Output keys follow the musique_round1_train.py contract (composed / direct /
step_questions / step_answers / levels / accept / *_correct) so the trainer
ports over; gold is NOT stored in the label file (plan §8).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sparc_seed import k_bucket, load_inputs

from self.coding.sparc_composition import (
    execute_sql,
    extract_sql,
    guard_levels,
    load_schemas,
    presupposes_existence,
    result_shape_ok,
    single_prompt,
    sparc_ex,
    static_check,
    suite_agree,
    suite_correct,
    suite_paths,
    turn_prompt,
)
from self.coding.training import generate_predictions, load_adapter_for_evaluation

LEVELS = ("L1", "L2", "L3", "L4")


def chunked_generate(model, tok, items, bs, chunk, max_new, label):
    preds = []
    for start in range(0, len(items), chunk):
        preds.extend(generate_predictions(
            model=model, tokenizer=tok, examples=items[start:start + chunk],
            batch_size=bs, max_new_tokens=max_new))
        print(f"  [{label}] {min(start + chunk, len(items))}/{len(items)}",
              flush=True)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    for a in ("--sequences", "--gold", "--schemas", "--partition"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True,
                    help="spider tables.json (structured schemas for the "
                         "static guard)")
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--testsuite-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--accept-level", choices=LEVELS, default="L4")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke flag: cap the pool at N sequences")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = args.turns

    class _A:                      # load_inputs reads attribute names
        sequences, gold, schemas, partition = (args.sequences, args.gold,
                                               args.schemas, args.partition)
        spider_singles = None
    data = load_inputs(_A)
    schemas_meta, gold = data["schemas"], data["gold"]
    db_schemas = load_schemas(json.load(open(args.tables)))
    comp_dbs = set(data["partition"]["composition_dbs"])

    pool = [r for r in data["sequences"]
            if r["split"] == "train" and r["db_id"] in comp_dbs
            and r["usable"] and r["monotone"] and r["turn_count"] == K
            and not schemas_meta[r["db_id"]]["excluded"]]
    print(f"composition-source monotone {K}-turn pool: {len(pool)}", flush=True)
    rng = random.Random(args.seed)
    cap = min(args.pool_size, args.limit) if args.limit else args.pool_size
    if len(pool) > cap:
        pool = rng.sample(pool, cap)
    print(f"labeling {len(pool)} sequences at accept level "
          f"{args.accept_level}", flush=True)

    model, tok = load_adapter_for_evaluation(args.model, args.adapter)
    suite_cache = {}

    def paths_for(db_id):
        if db_id not in suite_cache:
            suite_cache[db_id] = suite_paths(args.testsuite_root, args.db_root,
                                             db_id)
        return suite_cache[db_id]

    # --- two-view chain: A advances the state, B is the agreement witness ---
    state = [None] * len(pool)
    step_answers = [[None] * K for _ in pool]
    step_alt = [[None] * K for _ in pool]
    step_checks = [[None] * K for _ in pool]
    for i in range(1, K + 1):
        items_a, items_b = [], []
        for r, s in zip(pool, state):
            schema_text = schemas_meta[r["db_id"]]["text"]
            q = r["turns"][i - 1]["question"]
            items_a.append(sparc_ex(turn_prompt(schema_text, s, q, "A"),
                                    "", f"{r['sequence_id']}|{i}a",
                                    r["db_id"], K))
            items_b.append(sparc_ex(turn_prompt(schema_text, s, q, "B"),
                                    "", f"{r['sequence_id']}|{i}b",
                                    r["db_id"], K))
        preds_a = chunked_generate(model, tok, items_a, args.batch_size,
                                   args.chunk, args.max_new_tokens,
                                   f"turn{i}A")
        preds_b = chunked_generate(model, tok, items_b, args.batch_size,
                                   args.chunk, args.max_new_tokens,
                                   f"turn{i}B")
        for j, (r, pa, pb) in enumerate(zip(pool, preds_a, preds_b)):
            sql_a, sql_b = extract_sql(pa), extract_sql(pb)
            q = r["turns"][i - 1]["question"]
            paths = paths_for(r["db_id"])
            check = {"produced": bool(sql_a), "parses": False,
                     "schema_valid": False, "literals_typed": False,
                     "monotone": False, "exec_ok": False, "shape_ok": False,
                     "nonempty_ok": False, "twoview_agree": False,
                     "n_suite_instances": len(paths), "reason": None}
            if sql_a:
                st = static_check(sql_a, db_schemas[r["db_id"]],
                                  prev_sql=state[j])
                check.update({k: st[k] for k in
                              ("parses", "schema_valid", "literals_typed",
                               "monotone")})
                check["reason"] = st["reason"]
                if st["parses"] and st["schema_valid"]:
                    results = [execute_sql(p, sql_a, timeout_s=args.timeout_s)
                               for p in paths]
                    check["exec_ok"] = all(res.ok for res in results)
                    if not check["exec_ok"]:
                        check["reason"] = check["reason"] or "exec_error"
                    shape = result_shape_ok(q, results[0])
                    check["shape_ok"] = shape["ok"]
                    if not shape["ok"]:
                        check["reason"] = check["reason"] or shape["reason"]
                    nonempty = any(res.ok and res.rows for res in results)
                    check["nonempty_ok"] = (nonempty
                                            or not presupposes_existence(q))
                    if not check["nonempty_ok"]:
                        check["reason"] = check["reason"] or "empty_result"
                    check["twoview_agree"] = bool(sql_b) and suite_agree(
                        paths, sql_a, sql_b, timeout_s=args.timeout_s)
                    if check["exec_ok"] and not check["twoview_agree"]:
                        check["reason"] = check["reason"] or "twoview_disagree"
            else:
                check["reason"] = "no_sql"
            step_answers[j][i - 1] = sql_a or None
            step_alt[j][i - 1] = sql_b or None
            step_checks[j][i - 1] = check
            if sql_a:
                state[j] = sql_a
        print(f"  turn {i} done", flush=True)

    # --- direct labels (B1 arm), guarded at the final-query analog of L3 ---
    d_items = [sparc_ex(single_prompt(schemas_meta[r["db_id"]]["text"],
                                      r["final_intent_question"]),
                        "", r["sequence_id"], r["db_id"], K) for r in pool]
    d_pred = chunked_generate(model, tok, d_items, args.batch_size, args.chunk,
                              args.max_new_tokens, "direct")
    directs = [extract_sql(p) for p in d_pred]

    def final_l3(sql, r):
        if not sql:
            return False
        st = static_check(sql, db_schemas[r["db_id"]])
        if not (st["parses"] and st["schema_valid"] and st["literals_typed"]):
            return False
        paths = paths_for(r["db_id"])
        results = [execute_sql(p, sql, timeout_s=args.timeout_s) for p in paths]
        if not all(res.ok for res in results):
            return False
        q = r["final_intent_question"]
        if not result_shape_ok(q, results[0])["ok"]:
            return False
        return (any(res.rows for res in results)
                or not presupposes_existence(q))

    # --- assemble rows; gold only for measurement ---
    rows = []
    counts = {lv: [0, 0] for lv in LEVELS}
    n_composed_ok = n_direct_ok = 0
    reject_causes = Counter()
    first_err_hist = Counter()
    per_db_accept = Counter()
    LEVEL_KEYS = {"L1": ("produced",),
                  "L2": ("produced", "parses", "schema_valid",
                         "literals_typed", "monotone"),
                  "L3": ("produced", "parses", "schema_valid",
                         "literals_typed", "monotone", "exec_ok", "shape_ok",
                         "nonempty_ok"),
                  "L4": ("produced", "parses", "schema_valid",
                         "literals_typed", "monotone", "exec_ok", "shape_ok",
                         "nonempty_ok", "twoview_agree")}
    for j, r in enumerate(pool):
        checks = step_checks[j]
        levels = guard_levels(checks)
        accept = levels[args.accept_level]
        rejected_at, cause = None, None
        for i, c in enumerate(checks, 1):
            if not all(c.get(k) for k in LEVEL_KEYS[args.accept_level]):
                rejected_at, cause = i, c["reason"] or "unknown"
                break
        if accept:
            per_db_accept[r["db_id"]] += 1
        else:
            reject_causes[cause] += 1

        g = gold[r["sequence_id"]]
        composed = state[j] if all(c["produced"] for c in checks) else None
        paths = paths_for(r["db_id"])
        composed_ok = composed is not None and suite_correct(
            paths, composed, g["final_sql"], timeout_s=args.timeout_s)
        direct_ok = bool(directs[j]) and suite_correct(
            paths, directs[j], g["final_sql"], timeout_s=args.timeout_s)
        turn_correct = [
            a is not None and suite_correct(paths, a, gt,
                                            timeout_s=args.timeout_s)
            for a, gt in zip(step_answers[j], g["turn_sqls"])]
        first_error = next((i for i, okc in enumerate(turn_correct, 1)
                            if not okc), None)
        if first_error:
            first_err_hist[first_error] += 1
        n_composed_ok += composed_ok
        n_direct_ok += direct_ok
        for lv, ok in levels.items():
            if ok:
                counts[lv][0] += 1
                counts[lv][1] += composed_ok

        rows.append({
            "id": r["sequence_id"], "db_id": r["db_id"], "k": K,
            "question": r["final_intent_question"],
            "composed": composed, "composed_alt": step_alt[j][K - 1],
            "direct": directs[j] or None,
            "step_questions": [t["question"] for t in r["turns"]],
            "step_answers": step_answers[j], "step_checks": checks,
            "levels": levels, "accept": accept,
            "direct_accept": final_l3(directs[j], r),
            "composed_correct": bool(composed_ok),
            "direct_correct": bool(direct_ok),
            "turn_correct": turn_correct, "first_error_turn": first_error,
            "rejected_at_turn": rejected_at, "reject_cause": cause})

    n = len(pool)
    guard_table = {}
    for lv, (acc, correct) in counts.items():
        guard_table[lv] = {
            "accepted": acc,
            "precision": correct / acc if acc else None,
            "recall": correct / n_composed_ok if n_composed_ok else None}
    empties = sum(1 for row in rows
                  if row["reject_cause"] == "empty_result")
    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "pool": n,
        "mechanism": {"composed_acc": n_composed_ok / n,
                      "direct_acc": n_direct_ok / n,
                      "headroom": (n_composed_ok - n_direct_ok) / n},
        "guards": guard_table,
        "reject_causes": dict(reject_causes.most_common()),
        "empty_result_rejections": empties,
        "first_error_turn_hist": dict(sorted(first_err_hist.items())),
        "direct_accepted": sum(row["direct_accept"] for row in rows),
        "suite_instances": dict(Counter(
            schemas_meta[r["db_id"]]["n_suite_instances"] for r in pool)),
        "accepted_per_db_top": dict(per_db_accept.most_common(10)),
    }
    with open(args.out_dir / "round1_labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(args.out_dir / "round1_label_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"A mechanism: composed={report['mechanism']['composed_acc']:.3f} "
          f"direct={report['mechanism']['direct_acc']:.3f} "
          f"({report['mechanism']['headroom']:+.3f})", flush=True)
    for lv, gt in guard_table.items():
        p = f"{gt['precision']:.3f}" if gt["precision"] is not None else "-"
        rc = f"{gt['recall']:.3f}" if gt["recall"] is not None else "-"
        print(f"B {lv}: accepted={gt['accepted']}/{n} precision={p} "
              f"recall={rc}", flush=True)
    print(f"reject causes: {report['reject_causes']}", flush=True)
    print(f"wrote {args.out_dir}/round1_labels.jsonl", flush=True)


if __name__ == "__main__":
    main()
