#!/usr/bin/env python3
"""Join DROP with BREAK QDMR and emit the canonical DAG dataset of plan §8.

Week 0, step 1.  Downloads nothing: the two sources are expected under
`data/` already (see `--help` for the URLs), so the job is reproducible offline
and the pinned hashes are checked rather than re-fetched.

Sources
-------
  data/drop_dataset/drop_dataset_{train,dev}.json
      Official DROP release.  Keyed by passage id (`nfl_1087`), each entry
      holding `passage` and `qa_pairs` with `query_id`, `question`, `answer`.
  data/break_logical_forms_{train,dev}.csv
      BREAK `break_dataset/logical-forms/{train,dev}.csv`.  The `program`
      column carries operator arguments explicitly, which the plain
      `break_dataset/QDMR/` files do not; the executor's node table cannot be
      built without it (plan §24).

The join key is BREAK's question id, `DROP_{split}_{section_id}_{query_id}`
(plan §3.4).  `section_id` is the DROP passage id and `query_id` the question
UUID, so the join is exact and passage-level partitioning reads straight off
the id.

Outputs
-------
  data/drop_qdmr_{train,dev}.jsonl        canonical DAGs, gold answer removed
  data/drop_gold_{train,dev}.jsonl        gold answers, separate audit file
  data/drop_seed_pool_train.jsonl         span-answer atoms on uncovered passages
  drop_qdmr_build_report.json             counts, join rate, exclusion rates

Gold final answers live only in the audit file.  Generation code must not
import it (plan §8).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drop_executor import (
    UnsupportedOperator,
    executor_depth,
    model_owned_count,
    parse_program,
    sink_type,
)

csv.field_size_limit(10**7)

# Sink is numeric when the last node is an executor node; the executor table
# only contains number-producing operators (plan §6).
NUMERIC_AGGREGATE_FNS = frozenset({"count", "sum", "min", "max", "avg"})

# Expected counts, measured 2026-08-06 (plan §3.1).  A mismatch means the
# upstream release changed and blocks the run.
EXPECTED = {
    "train": {"break_rows": 7672, "break_passages": 3624,
              "drop_passages": 5565, "drop_questions": 77409},
    "dev": {"break_rows": 1265, "break_passages": 247,
            "drop_passages": 582, "drop_questions": 9536},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def answer_type(answer: dict) -> str:
    if answer.get("number"):
        return "number"
    if any(answer.get("date", {}).values()):
        return "date"
    spans = answer.get("spans") or []
    if len(spans) > 1:
        return "multispan"
    if len(spans) == 1:
        return "span"
    return "empty"


def answer_strings(answer: dict) -> list[str]:
    """The gold answer as a list of strings, in DROP's own rendering."""
    if answer.get("number"):
        return [str(answer["number"])]
    date = answer.get("date", {})
    if any(date.values()):
        return [" ".join(p for p in (date.get("day"), date.get("month"),
                                     date.get("year")) if p)]
    return list(answer.get("spans") or [])


def load_drop(path: Path) -> dict:
    return json.load(open(path))


def parse_break_id(question_id: str) -> tuple[str, str, str] | None:
    """`DROP_train_nfl_1087_<uuid>` -> ("train", "nfl_1087", "<uuid>")."""
    if not question_id.startswith("DROP_"):
        return None
    rest = question_id[len("DROP_"):]
    split, _, body = rest.partition("_")
    section_id, _, query_id = body.rpartition("_")
    if not section_id or not query_id:
        return None
    return split, section_id, query_id


def build_split(split: str, drop_path: Path, break_path: Path, out_dir: Path,
                report: dict) -> None:
    drop = load_drop(drop_path)
    n_questions = sum(len(v["qa_pairs"]) for v in drop.values())
    qa_index = {(pid, qa["query_id"]): (pid, qa)
                for pid, entry in drop.items() for qa in entry["qa_pairs"]}

    stats = {
        "drop_passages": len(drop), "drop_questions": n_questions,
        "break_rows": 0, "break_passages": 0,
        "joined": 0, "join_failures": 0,
        "excluded_operator": Counter(), "excluded_parse": 0,
        "sink_type": Counter(), "aggregate_sink_fn": Counter(),
        "numeric_sink": 0, "k_all": Counter(), "k_numeric": Counter(),
        "executor_depth": Counter(), "questions_per_passage": Counter(),
    }

    rows, gold_rows = [], []
    covered_passages: set[str] = set()
    per_passage: Counter = Counter()

    with open(break_path) as f:
        for raw in csv.DictReader(f):
            parsed = parse_break_id(raw["question_id"])
            if parsed is None:
                continue
            break_split, section_id, query_id = parsed
            stats["break_rows"] += 1
            covered_passages.add(section_id)

            key = (section_id, query_id)
            if key not in qa_index:
                stats["join_failures"] += 1
                continue
            stats["joined"] += 1
            pid, qa = qa_index[key]
            per_passage[pid] += 1

            try:
                program = [str(s) for s in ast.literal_eval(raw["program"])]
                nodes = parse_program(program, raw["decomposition"])
            except UnsupportedOperator as exc:
                stats["excluded_operator"][exc.op] += 1
                continue
            except (ValueError, SyntaxError):
                stats["excluded_parse"] += 1
                continue

            sink = nodes[-1]
            s_type = sink_type(nodes)
            stats["sink_type"][sink.op] += 1
            if sink.op == "AGGREGATE":
                stats["aggregate_sink_fn"][sink.fn] += 1
            k = model_owned_count(nodes)
            stats["k_all"][k] += 1

            numeric = s_type == "number"
            if not numeric:
                continue
            stats["numeric_sink"] += 1
            stats["k_numeric"][k] += 1
            stats["executor_depth"][executor_depth(nodes)] += 1

            gold = qa["answer"]
            rows.append({
                "example_id": raw["question_id"],
                "passage_id": pid,
                "query_id": query_id,
                "split": break_split,
                "domain": pid.split("_")[0],
                "passage": drop[pid]["passage"],
                "original_question": qa["question"],
                "qdmr": [n.to_dict() for n in nodes],
                "sink_id": sink.node_id,
                "sink_type": s_type,
                "step_count": len(nodes),
                "model_owned_count": k,
                "executor_depth": executor_depth(nodes),
                "op_family": sink.fn or sink.op.lower(),
            })
            gold_rows.append({
                "example_id": raw["question_id"],
                "answer": answer_strings(gold),
                "answer_type": answer_type(gold),
                "validated": [answer_strings(a)
                              for a in qa.get("validated_answers", [])],
            })

    stats["break_passages"] = len(covered_passages)
    stats["questions_per_passage"] = Counter(per_passage.values())

    exp = EXPECTED[split]
    mismatches = {k: (stats[k], v) for k, v in exp.items() if stats[k] != v}
    stats["expected_counts_match"] = not mismatches
    stats["count_mismatches"] = mismatches

    write_jsonl(out_dir / f"drop_qdmr_{split}.jsonl", rows)
    write_jsonl(out_dir / f"drop_gold_{split}.jsonl", gold_rows)

    stats = {k: (dict(sorted(v.items(), key=lambda x: str(x[0])))
                 if isinstance(v, Counter) else v)
             for k, v in stats.items()}
    report[split] = stats
    print(f"[{split}] break_rows={stats['break_rows']} joined={stats['joined']} "
          f"numeric_sink={stats['numeric_sink']} "
          f"k_numeric={stats['k_numeric']}", flush=True)
    if mismatches:
        print(f"[{split}] COUNT MISMATCH vs plan §3.1: {mismatches}", flush=True)


def build_seed_pool(drop_path: Path, break_path: Path, out_dir: Path,
                    report: dict) -> None:
    """Span-answer atoms on passages BREAK does not cover (plan §7.1).

    The seed cannot be built from QDMR steps: BREAK releases no gold
    intermediate values and holds 23 single-step DROP decompositions in train.
    Passages outside BREAK's coverage supply gold-labeled extraction with no
    cost to the composition pool.
    """
    drop = load_drop(drop_path)
    covered: set[str] = set()
    with open(break_path) as f:
        for raw in csv.DictReader(f):
            parsed = parse_break_id(raw["question_id"])
            if parsed:
                covered.add(parsed[1])

    uncovered = sorted(set(drop) - covered)
    rows, types = [], Counter()
    for pid in uncovered:
        for qa in drop[pid]["qa_pairs"]:
            t = answer_type(qa["answer"])
            types[t] += 1
            if t not in ("span", "multispan"):
                continue
            rows.append({
                "example_id": f"seed_{pid}_{qa['query_id']}",
                "passage_id": pid,
                "query_id": qa["query_id"],
                "domain": pid.split("_")[0],
                "passage": drop[pid]["passage"],
                "question": qa["question"],
                "answer": answer_strings(qa["answer"]),
                "answer_type": t,
            })
    write_jsonl(out_dir / "drop_seed_pool_train.jsonl", rows)
    report["seed_pool"] = {
        "covered_passages": len(covered),
        "uncovered_passages": len(uncovered),
        "questions_on_uncovered": sum(types.values()),
        "answer_types": dict(sorted(types.items())),
        "atoms": len(rows),
    }
    print(f"[seed] uncovered passages={len(uncovered)} atoms={len(rows)}", flush=True)


def build_atom_pool_dev(drop_path: Path, qdmr_dev_path: Path, out_dir: Path,
                        report: dict) -> None:
    """Span-answer questions on the dev passages that carry usable numeric DAGs.

    The screen's `atom_proxy` arm (plan §4).  BREAK releases no gold
    intermediate values, so per-node accuracy cannot be scored; these questions
    measure extraction difficulty on the same passages under the same
    interface.  They are a proxy, not a per-node measurement.
    """
    drop = load_drop(drop_path)
    passage_ids = {json.loads(l)["passage_id"] for l in open(qdmr_dev_path)}
    rows, types = [], Counter()
    for pid in sorted(passage_ids):
        for qa in drop[pid]["qa_pairs"]:
            t = answer_type(qa["answer"])
            types[t] += 1
            if t not in ("span", "multispan"):
                continue
            rows.append({
                "example_id": f"atom_{pid}_{qa['query_id']}",
                "passage_id": pid,
                "query_id": qa["query_id"],
                "domain": pid.split("_")[0],
                "passage": drop[pid]["passage"],
                "question": qa["question"],
                "answer": answer_strings(qa["answer"]),
                "answer_type": t,
            })
    write_jsonl(out_dir / "drop_atom_pool_dev.jsonl", rows)
    lengths = sorted(len(drop[p]["passage"].split()) for p in passage_ids)
    report["atom_pool_dev"] = {
        "passages": len(passage_ids),
        "questions_on_those_passages": sum(types.values()),
        "answer_types": dict(sorted(types.items())),
        "atoms": len(rows),
        "passage_words": {
            "median": lengths[len(lengths) // 2],
            "p99": lengths[int(0.99 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
    }
    print(f"[atoms] dev passages={len(passage_ids)} atoms={len(rows)}", flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sources:\n"
               "  https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip\n"
               "  https://raw.githubusercontent.com/allenai/Break/master/"
               "break_dataset/logical-forms/{train,dev}.csv")
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).parent / "data")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "drop_qdmr_build_report.json")
    args = ap.parse_args()

    d = args.data_dir
    paths = {
        "drop_train": d / "drop_dataset" / "drop_dataset_train.json",
        "drop_dev": d / "drop_dataset" / "drop_dataset_dev.json",
        "break_train": d / "break_logical_forms_train.csv",
        "break_dev": d / "break_logical_forms_dev.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(missing))

    report: dict = {"hashes": {k: sha256(p) for k, p in paths.items()}}
    build_split("train", paths["drop_train"], paths["break_train"], d, report)
    build_split("dev", paths["drop_dev"], paths["break_dev"], d, report)
    build_seed_pool(paths["drop_train"], paths["break_train"], d, report)
    build_atom_pool_dev(paths["drop_dev"], d / "drop_qdmr_dev.jsonl", d, report)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)

    ok = all(report[s]["expected_counts_match"] for s in ("train", "dev"))
    print("VERDICT:", "counts reproduce plan §3.1" if ok
          else "COUNTS DIFFER FROM plan §3.1 -- treat as a version change",
          flush=True)


if __name__ == "__main__":
    main()
