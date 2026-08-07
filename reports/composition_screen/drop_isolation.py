#!/usr/bin/env python3
"""DROP-QDMR isolation screen: does composing through the executor beat asking?

Week 0 gate, plan §4.  Four arms, paired on the same dev instances, binned by
model-owned node count k in {2, 3, 4}:

  direct       original DROP question over the passage, one call
  atom_proxy   DROP span-answer questions on the SAME passages
  composed     full DAG execution -- model at extraction nodes, Python at
               ARITHMETIC/AGGREGATE nodes, the model's own upstream values
  corrupt      composed, with one non-sink model-owned value replaced by a
               value drawn from a different instance in the same domain

The MuSiQue screen's `part` arm has no analogue here.  MuSiQue releases a gold
answer per step so per-node accuracy is measurable; BREAK releases structure
only, so no intermediate node carries a label.  `atom_proxy` measures the same
quantity the `part` arm was there to measure -- is the atom easier than the
composite -- using DROP's own span questions over the same passages.  It is a
proxy, not a per-node measurement, and every report must say so.

`corrupt` is the load-bearing check: if corrupting an upstream value leaves the
sink unchanged, the composition contributes nothing and the sink extraction
determines the answer alone.

Per-instance outcomes are written for every arm.  The MuSiQue retrieval control
stored only aggregates and its odd pattern could never be tested afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drop_executor import COUNT_POLICIES, Node, execute_dag, parse_list, score
from musique_seed import boot_ci, mcnemar

from self.coding.atomic_data import AtomicExample
from self.coding.training import generate_predictions

EXTRACT_INSTRUCTION = (
    "Answer the question using the passage. Reply with the answer only -- "
    "a number or short phrase, no sentence. If the answer has several parts, "
    "separate them with \" | \"."
)


def prompt(passage: str, question: str) -> str:
    return f"{EXTRACT_INSTRUCTION}\n\nPassage: {passage}\n\nQuestion: {question}\nAnswer:"


def ex(passage: str, question: str, sid: str, k: int) -> AtomicExample:
    return AtomicExample(
        task="drop_qdmr", source_id=sid, source_group_id=sid, split="dev",
        messages=({"role": "user", "content": prompt(passage, question)},),
        target="", evaluator={}, metadata={"k": k},
    )


def gold_variants(gold_row: dict) -> list:
    """Accepted renderings of the gold answer, including validated answers."""
    out = [gold_row["answer"]]
    out += [v for v in gold_row.get("validated", []) if v]
    return out


def em_against(prediction, gold_row: dict) -> bool:
    return any(score(prediction, g)[0] == 1.0 for g in gold_variants(gold_row))


class BatchedRunner:
    """Runs the composition chain node-index by node-index across instances.

    Nodes arrive in BREAK order, which `parse_program` has already verified is
    topological, so every instance can be advanced one node position at a time
    and the model calls for that position batched together.
    """

    def __init__(self, model, tok, batch_size: int, max_new_tokens: int = 48):
        self.model, self.tok = model, tok
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens

    def generate(self, items: list[AtomicExample]) -> list[str]:
        if not items:
            return []
        return generate_predictions(model=self.model, tokenizer=self.tok,
                                    examples=items, batch_size=self.batch_size,
                                    max_new_tokens=self.max_new_tokens)


def run_chain(runner: BatchedRunner, rows: list[dict],
              overrides: dict[int, dict[int, list[str]]] | None = None,
              count_policy: str = "reject"
              ) -> tuple[list[list[str] | None], list[dict]]:
    """Execute every row's DAG, batching model calls across rows per node index.

    `overrides[row_index][node_id]` replaces the model's answer at that node,
    which is how the corrupt arm is built without a second generation pass.

    `execute_dag` is driven as a coroutine would be: each row is stepped until
    it needs a model answer, the pending questions are generated as one batch,
    and the answers are fed back.  Rather than restructure the executor, the
    same deterministic pass is replayed per node index with a cache of answers
    already known for that row.
    """
    overrides = overrides or {}
    cache: list[dict[int, list[str]]] = [dict() for _ in rows]
    max_nodes = max(len(r["qdmr"]) for r in rows)

    for _ in range(max_nodes):
        pending: list[tuple[int, int, str]] = []   # (row index, node id, question)

        def ask_factory(j: int):
            def ask(node: Node, question: str) -> list[str]:
                if node.node_id in cache[j]:
                    return cache[j][node.node_id]
                pending.append((j, node.node_id, question))
                raise _NeedAnswer()
            return ask

        for j, row in enumerate(rows):
            nodes = [Node.from_dict(d) for d in row["qdmr"]]
            try:
                execute_dag(nodes, ask_factory(j), count_policy=count_policy)
            except _NeedAnswer:
                pass
        if not pending:
            break
        items = [ex(rows[j]["passage"], q, f"{rows[j]['example_id']}|{nid}",
                    rows[j]["model_owned_count"])
                 for j, nid, q in pending]
        preds = runner.generate(items)
        for (j, nid, _), pred in zip(pending, preds):
            value = overrides.get(j, {}).get(nid) or parse_list(pred)
            cache[j][nid] = value

    values, traces = [], []
    for j, row in enumerate(rows):
        nodes = [Node.from_dict(d) for d in row["qdmr"]]
        def ask(node: Node, question: str, j=j) -> list[str]:
            return cache[j].get(node.node_id, [])
        value, trace = execute_dag(nodes, ask, count_policy=count_policy)
        values.append(value)
        traces.append(trace)
    return values, traces


class _NeedAnswer(Exception):
    """Internal control signal: this row needs a model call before continuing."""


def build_corruption(rows: list[dict], traces: list, rng: random.Random
                     ) -> dict[int, dict[int, list[str]]]:
    """One non-sink model-owned value per row, replaced from a different row.

    Donors are drawn from the same domain (`nfl` or `history`) so the swap stays
    plausible in surface form; if the sink is unchanged by a plausible swap, the
    composition is not load-bearing.
    """
    by_domain: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for j, (row, trace) in enumerate(zip(rows, traces)):
        for node in row["qdmr"][:-1]:
            if node["owner"] == "model" and node["node_id"] in trace.values:
                by_domain[row["domain"]].append((j, trace.values[node["node_id"]]))

    overrides: dict[int, dict[int, list[str]]] = {}
    for j, (row, trace) in enumerate(zip(rows, traces)):
        candidates = [n for n in row["qdmr"][:-1]
                      if n["owner"] == "model" and n["node_id"] in trace.values]
        donors = [v for i, v in by_domain[row["domain"]] if i != j]
        if not candidates or not donors:
            continue
        target = rng.choice(candidates)
        replacement = rng.choice(donors)
        if replacement == trace.values[target["node_id"]]:
            continue
        overrides[j] = {target["node_id"]: replacement}
    return overrides


def load_atom_proxy(atom_pool_path: Path, passage_ids: set[str], size: int,
                    rng: random.Random) -> list[dict]:
    """DROP span questions restricted to exactly the passages the screen sampled."""
    rows = [r for r in map(json.loads, open(atom_pool_path))
            if r["passage_id"] in passage_ids]
    if len(rows) > size:
        rows = rng.sample(rows, size)
    return rows


def screen(runner: BatchedRunner, sample_by_k: dict[int, list[dict]],
           gold: dict[str, dict], atoms: list[dict], tag: str,
           records: list[dict], seed: int,
           count_policy: str = "reject") -> dict:
    """Run all four arms and return the report for one model state.

    `drop_seed.py` calls this before and after seed training, which is why the
    arms live here rather than inside `main`.  `records` accumulates
    per-instance outcomes across calls; each row carries `phase` so the two
    states stay separable (plan §18).
    """
    report: dict = {"by_k": {}}

    for k, sample in sorted(sample_by_k.items()):
        print(f"== [{tag}] k={k}: {len(sample)} instances ==", flush=True)
        n = len(sample)

        d_items = [ex(r["passage"], r["original_question"], r["example_id"], k)
                   for r in sample]
        d_pred = runner.generate(d_items)
        d_ok = [em_against(parse_list(p), gold[r["example_id"]])
                for p, r in zip(d_pred, sample)]

        c_values, c_traces = run_chain(runner, sample,
                                      count_policy=count_policy)
        c_ok = [v is not None and em_against(v, gold[r["example_id"]])
                for v, r in zip(c_values, sample)]

        overrides = build_corruption(sample, c_traces, random.Random(seed + k))
        x_values, _ = run_chain(runner, sample, overrides=overrides,
                                count_policy=count_policy)
        x_ok = [v is not None and em_against(v, gold[r["example_id"]])
                for v, r in zip(x_values, sample)]
        x_changed = sum(1 for a, b in zip(c_values, x_values) if a != b)

        rejections: dict[str, int] = defaultdict(int)
        for t in c_traces:
            if t.rejection:
                rejections[t.rejection["reason"]] += 1

        for r, do, co, xo, tr in zip(sample, d_ok, c_ok, x_ok, c_traces):
            records.append({
                "phase": tag, "arm_set": "screen", "k": k, "id": r["example_id"],
                "passage_id": r["passage_id"], "op_family": r["op_family"],
                "executor_depth": r["executor_depth"],
                "direct_ok": bool(do), "composed_ok": bool(co),
                "corrupt_ok": bool(xo),
                "rejection": tr.rejection["reason"] if tr.rejection else None,
            })

        pairs = list(zip(c_ok, d_ok))
        cell = {
            "n": n,
            "direct": sum(d_ok) / n,
            "composed": sum(c_ok) / n,
            "corrupt": sum(x_ok) / n,
            "headroom": (sum(c_ok) - sum(d_ok)) / n,
            "corruption_drop": (sum(c_ok) - sum(x_ok)) / n,
            "corrupt_sinks_changed": x_changed,
            "chain_rejected": sum(1 for t in c_traces if t.rejection),
            "rejection_causes": dict(sorted(rejections.items())),
            "mcnemar": mcnemar(pairs),
            "ci95": list(boot_ci(pairs, random.Random(seed))),
        }
        report["by_k"][k] = cell
        print(f"  direct={cell['direct']:.3f} composed={cell['composed']:.3f} "
              f"({cell['headroom']:+.3f}) corrupt={cell['corrupt']:.3f} "
              f"rejected={cell['chain_rejected']}", flush=True)

    # --- atom proxy: the substitute for MuSiQue's part arm ---
    a_items = [ex(a["passage"], a["question"], a["example_id"], 1) for a in atoms]
    a_pred = runner.generate(a_items)
    a_ok = [score(parse_list(p), a["answer"])[0] == 1.0
            for p, a in zip(a_pred, atoms)]
    for a, ok in zip(atoms, a_ok):
        records.append({"phase": tag, "arm_set": "atom_proxy", "k": 1,
                        "id": a["example_id"], "passage_id": a["passage_id"],
                        "answer_type": a["answer_type"], "atom_ok": bool(ok)})
    report["atom_proxy"] = {"n": len(atoms),
                            "accuracy": sum(a_ok) / max(len(a_ok), 1)}
    print(f"[{tag}] atom proxy: n={len(atoms)} "
          f"accuracy={report['atom_proxy']['accuracy']:.3f}", flush=True)

    # --- pooled statistics ---
    phase_rows = [r for r in records
                  if r["phase"] == tag and r["arm_set"] == "screen"]
    pairs = [(r["composed_ok"], r["direct_ok"]) for r in phase_rows]
    lo, hi = boot_ci(pairs, random.Random(seed))
    pooled = {
        "n": len(pairs),
        "direct": sum(d for _, d in pairs) / len(pairs),
        "composed": sum(c for c, _ in pairs) / len(pairs),
        "corrupt": sum(r["corrupt_ok"] for r in phase_rows) / len(phase_rows),
        "headroom": (sum(c for c, _ in pairs) - sum(d for _, d in pairs)) / len(pairs),
        "ci95": [lo, hi],
        "mcnemar": mcnemar(pairs),
    }
    pooled["atom_minus_direct"] = report["atom_proxy"]["accuracy"] - pooled["direct"]
    pooled["corruption_drop"] = pooled["composed"] - pooled["corrupt"]
    report["pooled"] = pooled
    print_pooled(tag, pooled)
    return report


def print_pooled(tag: str, pooled: dict) -> None:
    print(f"\n[{tag}] POOLED: composed={pooled['composed']:.3f} "
          f"direct={pooled['direct']:.3f} headroom={pooled['headroom']:+.3f} "
          f"95%CI[{pooled['ci95'][0]:+.3f},{pooled['ci95'][1]:+.3f}] "
          f"McNemar p={pooled['mcnemar']['p_value']:.4f}", flush=True)
    print(f"[{tag}] ATOM PROXY - DIRECT: {pooled['atom_minus_direct']:+.3f}  "
          f"(proxy for the part-vs-whole gap; not a per-node measurement)",
          flush=True)
    print(f"[{tag}] CORRUPTION DROP: {pooled['corruption_drop']:+.3f}", flush=True)


def verdict(pooled: dict) -> bool:
    """The pre-registered rule of plan §4."""
    return (pooled["headroom"] > 0 and pooled["ci95"][0] > 0
            and pooled["mcnemar"]["p_value"] < 0.05)


def draw_cells(rows_all: list[dict], cells: list[int], per_k: int,
               rng: random.Random) -> dict[int, list[dict]]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for r in rows_all:
        if r["model_owned_count"] in cells:
            by_k[r["model_owned_count"]].append(r)
    out = {}
    for k, v in sorted(by_k.items()):
        out[k] = v if len(v) <= per_k else rng.sample(v, per_k)
        print(f"cell k={k}: {len(out[k])} instances (available {len(v)})", flush=True)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA adapter; omit to screen the base model")
    ap.add_argument("--dev-data", type=Path, required=True)
    ap.add_argument("--dev-gold", type=Path, required=True)
    ap.add_argument("--atom-pool", type=Path, required=True,
                    help="span-answer questions over dev passages")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count-policy", choices=COUNT_POLICIES, default="reject",
                    help="how to read AGGREGATE['count'] over a single numeric "
                         "span (plan Risk 6); the rejection rate this produces "
                         "is large enough to decide pool sizes")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_all = [json.loads(l) for l in open(args.dev_data)]
    gold = {r["example_id"]: r for r in map(json.loads, open(args.dev_gold))}

    rng = random.Random(args.seed)
    sample_by_k = draw_cells(rows_all, args.cells, args.per_k, rng)
    passage_ids = {r["passage_id"] for s in sample_by_k.values() for r in s}
    atoms = load_atom_proxy(args.atom_pool, passage_ids,
                            args.per_k * len(sample_by_k), rng)

    if args.adapter:
        from self.coding.training import load_adapter_for_evaluation
        model, tok = load_adapter_for_evaluation(args.model, args.adapter)
    else:
        from self.coding.training import load_qwen_lora_model, load_qwen_tokenizer
        tok = load_qwen_tokenizer(args.model)
        model = load_qwen_lora_model(args.model)
    runner = BatchedRunner(model, tok, args.batch_size)

    records: list[dict] = []
    tag = "after_seed" if args.adapter else "base"
    report = screen(runner, sample_by_k, gold, atoms, tag, records, args.seed,
                    count_policy=args.count_policy)
    report["config"] = {k: str(v) for k, v in vars(args).items()}

    (args.out_dir / "per_instance.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    (args.out_dir / "drop_isolation.json").write_text(json.dumps(report, indent=2))

    print("VERDICT:", "composition wins" if verdict(report["pooled"])
          else "no significant headroom -- rerun after seeding (plan §4)",
          flush=True)


if __name__ == "__main__":
    main()
