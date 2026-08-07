# DROP-QDMR — continuation guide

Practical handoff, in the form of `MUSIQUE_HANDOFF.md`. The scientific record is
`DROP_QDMR_CSI_Research_Plan.md`; this file is what you need to *run* things.
Written 2026-08-06.

---

## 1. Where we are in one paragraph

**DROP-QDMR did not pass its Week 0 gate and is discontinued**, at the same
gate that discontinued BFCL and CLUTRR. Composed − direct is −.377 at base
(job 3648794) and −.290 after seeding (job 3648795), both with 95% CIs well
below zero and McNemar p < 1e-4 on 300 paired dev instances. The seed worked as
intended on its own terms — atom accuracy .473 → .683 — and moved composed
accuracy by +.017. Nothing was pseudo-labeled and nothing was retrained on
pseudo-labels. §1.3 records why, because the reason is a property of the
benchmark rather than of the implementation.

---

## 1.1 Base screen results (2026-08-06, job 3648794)

| cell | direct | composed | headroom | corrupt | chains rejected |
|---|---:|---:|---:|---:|---:|
| k=2 | .640 | .220 | −.420 | .080 | 37/100 |
| k=3 | .560 | .200 | −.360 | .130 | 36/100 |
| k=4 | .630 | .280 | −.350 | .040 | 50/100 |
| **pooled** | **.610** | **.233** | **−.377** | .083 | 123/300 |

Atom proxy .473 (n=300). Corruption drop +.150.

The seed job's own before-phase reproduced these numbers exactly (composed .233,
direct .610, headroom −.377) on a different node, so evaluation is deterministic
across nodes here and the before block can be reused as a paired baseline
without re-measuring, as `musique_round1_train.py` does.

**What the breakdown says.** Three separate things are wrong, and they need
separate responses.

*Rejection is large but is not the whole story.* 41% of chains reject:
`ambiguous_count` 84, `operand_parse` 32, `operand_not_scalar` 7. Among the
177 chains that do complete, composed is .395 against direct .638 **on the same
rows**. So composition loses even where it runs.

*The atom is harder than the composite.* The proxy (.473) sits below direct
(.610), and this is not only the answer-type confound — DROP's own span
questions are simply harder for this model than its numeric questions
(span .492, multi-span .370). This is the screening criterion failing in its
most direct form. DROP is single-passage, so the model can often read a numeric
answer off the text without decomposing anything, which is exactly the
structural property MuSiQue lacked and DROP has.

*More executor nodes made labels worse, not better.* Plan §16.5 predicts
pseudo-label precision rises with executor-owned depth. Measured: depth 1
composed .438 (n=144), depth 2 composed .212 (n=33). Depth-2 rows are harder
overall (direct .515 against .667), so this is not a clean refutation, but the
predicted direction is absent at base.

By op family among completed chains: `difference` .532, `sum` .378, `count`
.274. The scalar-arithmetic slice — the one the plan's §1 example is drawn
from — is the only family where composition is close to competitive.

---

## 1.2 Seed rescreen (2026-08-06, job 3648795, 2h33m)

4,000 seed examples, 0 dropped over the 2,048-token limit, 300 steps, LoRA
r16/a32 with gradient checkpointing, train loss .364.

| | before | after | change |
|---|---:|---:|---:|
| atom proxy | .473 | .683 | +.210 |
| composed | .233 | .250 | +.017 |
| direct | .610 | .540 | **−.070** |
| headroom | −.377 | −.290 | +.087 |
| chains completing | 177/300 | 181/300 | +4 |
| composed among completed | .395 | .414 | +.019 |

95% CI after seeding [−.363, −.217], McNemar p < 1e-4. **Verdict: gate not met.**

The +.087 headroom gain is mostly direct getting worse. The paired
before/after test on direct accuracy gives −.070, McNemar p = .028,
CI [−.127, −.013] — a statistically significant **regression** in the thing the
method is supposed to improve.

## 1.3 Why it failed — the part that generalizes

**The only available atomic supervision teaches the wrong output
distribution.** BREAK releases no gold intermediate values, so the seed cannot
be built from QDMR steps and has to come from DROP's own span-answer questions
(§2, row 3). But the DAGs in scope have *numeric* sinks. Training on span
extraction therefore moved the model toward emitting spans, and the effect is
visible in both directions at once:

* atom span accuracy .492 → .732, but multi-span only .370 → .413 — the
  list-valued atom that AGGREGATE actually consumes barely moved;
* rejection causes shifted rather than shrank: `ambiguous_count` 84 → 55, while
  `operand_parse` went 32 → 60, i.e. the seeded model emits spans where the
  executor needs numbers;
* `sum`-family direct accuracy collapsed .622 → .200 among completed chains.

This is not an implementation defect that a better seed pool fixes. The
supervision that exists and the supervision the task needs are different kinds,
and BREAK's missing intermediates are what forces the substitution.

**The composite is not much harder than its parts.** DROP is single-passage, so
a numeric question can often be answered by reading the text directly.
Direct accuracy at base is .610 against MuSiQue's .563, on questions whose
decomposition has 2–4 extraction nodes. Decomposition has less to win here, and
each extra node is another chance to lose.

**The arithmetic never closes.** With executor nodes exact, composed accuracy is
roughly the product of atom accuracy over model-owned nodes. Parity needs atom
accuracy above .80 / .82 / .89 at k = 2 / 3 / 4 against the base direct
baselines. MuSiQue's seed topped out at part accuracy .815 / .770 / .703 on
single-span atoms; DROP's atom is list-valued and scored on exact set match.
The measured seed reached .683 on the easier proxy and delivered +.017 on
composed accuracy.

**What did work,** and is worth carrying to another benchmark: composition is
load-bearing (corruption drop +.150 at base, +.140 after seeding), the executor
is exact and its rejections are typed and informative, and the seed lifted its
own target metric by .210. The framework machinery is fine; the benchmark does
not have the property the framework needs.

---

## 2. What was measured, and what it changed in the plan

The plan was written before the data was inspected. Five of its assumptions did
not survive contact with it. All five are now corrected in the plan; they are
listed here because each is a trap worth not re-entering.

| Plan assumption | Measurement | Consequence |
|---|---|---|
| QDMR covers "a large fraction" of DROP | 7,672 of 77,409 train questions (10%), over 3,624 of 5,565 passages | pools are an order of magnitude smaller than MuSiQue's |
| curriculum runs 1 → 2 → 3 QDMR steps | 23 one-step and 616 two-step rows exist; the mode is 3–6 steps | the frontier axis is **model-owned node count k**, not step count. k = 2/3/4 gives 1,234 / 1,051 / 1,078 usable rows |
| seed from single-step extraction questions with gold labels | BREAK releases no gold intermediate values, and holds 23 single-step rows | seed comes from **DROP span-answer questions on the 1,941 passages BREAK does not cover** — 5,900 atoms, disjoint from the composition pool by construction |
| count family excluded, atoms are scalars | AGGREGATE is the largest sink family (2,243), and consumes a **list** | the atom is a list of spans everywhere; the list-completeness guard is in the MVE, not deferred |
| COMPARISON is an executor op to implement | 1,614 occurrences, sink every time, 0 mid-DAG, always non-numeric | not implemented; no numeric-sink DAG contains one |

Two further findings have no counterpart in the plan as written:

**`AGGREGATE['count', X]` is overloaded.** "How many field goals did Hartley
make?" wants `len(list)`. "How many men, horses, elephants combined in Nuri Khan
army?" annotates the same operator over a list holding one span, `50,000`, where
the intended answer is 50000. The two are indistinguishable from DAG structure.
The executor rejects the ambiguous case by default and `--count-policy` exposes
`reject` / `length` / `value` so the rate can be measured. **In the 15-instance
smoke run this was the dominant rejection cause, 7 of 15 chains.** If it holds at
n=300 it decides pool sizes: at that rate the k=2 pool falls from 1,063 to ~560,
below the 800-example floor of plan §12. This is the first thing to look at in
the full screen output.

**The MuSiQue `part` arm has no analogue here.** With no gold intermediates,
per-node accuracy cannot be scored at scale. The screen substitutes an `atom
proxy` arm — DROP span questions on the same passages — plus the 100-trace hand
audit of §16.2. The proxy has a known weakness: it scores *span* questions while
`direct` scores *numeric* ones, so a gap between them mixes atom difficulty with
answer-type difficulty. In the smoke run the proxy came in *below* direct
(.400 vs .733, n=15), which is the wrong sign for the screening criterion.
Before reading anything into that number at n=300, restrict the proxy to
matched answer types or treat the criterion as informative only in the
composed-versus-direct contrast.

---

## 3. Assets

Everything below is produced by `drop_qdmr_build.py` in about 4 seconds from the
two pinned sources, so none of it needs to survive a migration — but the sources
do, and they are large.

| path | rows | note |
|---|---:|---|
| `data/drop_dataset/drop_dataset_{train,dev}.json` | — | official DROP, 68 MB, untracked |
| `data/break_logical_forms_{train,dev}.csv` | — | BREAK logical forms, 23 MB, untracked |
| `data/drop_qdmr_{train,dev}.jsonl` | 3,864 / 599 | canonical numeric-sink DAGs, gold removed |
| `data/drop_gold_{train,dev}.jsonl` | 3,864 / 599 | gold answers, separate audit file |
| `data/drop_seed_pool_train.jsonl` | 5,900 | seed atoms, BREAK-uncovered passages |
| `data/drop_atom_pool_dev.jsonl` | 1,619 | atom-proxy arm, dev passages |
| `data/drop_partition.json` | — | 2,125 composition / 375 audit passages |

Source hashes are recorded in `drop_qdmr_build_report.json`.

## 4. Reproducing the data

```bash
cd reports/composition_screen/data
curl -sSL -o drop_dataset.zip \
  "https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip"
unzip -q drop_dataset.zip && rm drop_dataset.zip
for S in train dev; do
  curl -sSL -o break_logical_forms_$S.csv \
    "https://raw.githubusercontent.com/allenai/Break/master/break_dataset/logical-forms/$S.csv"
done
cd ../../..
$PY reports/composition_screen/drop_qdmr_build.py
$PY reports/composition_screen/drop_partition.py \
  --qdmr-train reports/composition_screen/data/drop_qdmr_train.jsonl \
  --seed-pool  reports/composition_screen/data/drop_seed_pool_train.jsonl \
  --out        reports/composition_screen/data/drop_partition.json
```

The build prints `VERDICT: counts reproduce plan §3.1`. Anything else means the
upstream release changed; treat it as a version change, not a bug to patch.

Read `break_dataset/logical-forms/`, not `break_dataset/QDMR/`. Only the former
carries the `program` column with explicit operator arguments, and the node
table cannot be built without it.

## 5. Running the gate

```bash
source reports/composition_screen/cluster_env.sh
sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
       reports/composition_screen/drop_isolation_portable.slurm   # base screen
sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
       reports/composition_screen/drop_seed_portable.slurm        # seed + rescreen
```

`drop_seed` carries the verdict. The base screen is expected to be
uninformative — MuSiQue's was +.023 with a CI spanning zero, and only the seed
moved it to +.080 (p=.0009). Both scripts print a `VERDICT:` line and write
`per_instance.jsonl` with one record per instance per arm.

Environment overrides both scripts accept: `PER_K`, `BATCH_SIZE`, and for the
seed `TRAIN_SIZE`, `MAX_STEPS`, `MICRO_BATCH`. Pass `ADAPTER_DIR` to
`drop_isolation_portable.slurm` to screen a trained checkpoint.

## 6. Code map

| file | role |
|---|---|
| `drop_eval.py` | official DROP EM/F1, vendored verbatim from `allenai/allennlp-models`. Needs scipy, installed into the project venv 2026-08-06 |
| `drop_executor.py` | node model, verbalization, symbolic executor, typed rejections |
| `drop_qdmr_build.py` | join, node parsing, all four data files, build report |
| `drop_partition.py` | passage-level composition / audit split |
| `drop_isolation.py` | four-arm screen; `screen()` is reused by `drop_seed.py` |
| `drop_seed.py` | seed training and rescreen, paired before/after statistics |
| `tests/test_drop_executor.py` | 22 unit checks on executor semantics |

## 7. Rules that carry over from MuSiQue

**Any arm consuming released gold structure needs a matched no-oracle arm.**
QDMR at inference is the same kind of oracle `question_decomposition` was. The
end-to-end arm (§5.2) is the matched counterpart. MuSiQue's self-proposed
decomposition arm failed twice, so the standing claim there is "composition
helps when handed the decomposition"; the same limit applies here and §21 says so.

**Baselines use the seeded direct model, not the base model.** On MuSiQue the
seed lifted direct prediction by +.120 at 3 hops on its own.

**Log per-instance outcomes in every arm.** Both scripts do.

**`train_lora` rejects over-length examples rather than truncating.** Both the
seed pool filter and any later round must filter on true tokenized length.
DROP passages are short enough that this should be rare: median 181 words, 99th
percentile 377, maximum 918 on the screened dev passages, against a 2,048-token
limit.

**Gradient checkpointing is required on the 48 GB L40 cards.** `GRAD_CKPT=1` in
`cluster_env.sh` feeds the `--gradient-checkpointing` flag.

## 8. Status: discontinued

The gate was applied and not met (§1.2). No further experiments are planned.
This section records what would have to be true to reopen it, so the decision
does not have to be re-derived.

**Reopening conditions.** Any one of these would change the arithmetic of §1.3:

1. **A source of gold intermediate values for DROP.** This removes the
   substitution that caused the seed to teach spans for numeric sinks. Nothing
   in BREAK provides it.
2. **A model whose list-valued extraction reaches ~.85 exact-match.** The 4B
   model reached .683 on single spans and .413 on multi-span after seeding.
   A larger model is the obvious untested lever, and it is cheap to test —
   `ADAPTER_DIR=... sbatch drop_isolation_portable.slurm` with a different base.
3. **A narrower scope that was explicitly ruled out.** The `difference` family
   is the only competitive one (composed .534 against direct .638 among
   completed chains after seeding, essentially unchanged by the seed). It holds
   ~1,001 train rows, ~600 after partitioning, below the 800-example floor of
   plan §12. A positive result there would be small and fragile.

**What to reuse.** The build, executor, partition, and screen code is not
DROP-specific in any deep way: it consumes a DAG with typed nodes and an
operator table. Any benchmark that publishes an executable decomposition can
reuse `drop_executor.py` and `drop_isolation.py` by writing a new build script.
The four-arm screen with per-instance logging and a printed verdict is the part
worth keeping.

**What the negative is good for.** "Composition does not help when the
composite is not much harder than its parts, and single-passage discrete
reasoning is such a case" is a quantified boundary condition for the framework,
measured on 300 paired instances at two model states. It belongs in the
conference version's task-selection discussion alongside BFCL and CLUTRR.
