# Research Plan: SParC-Spider-CSI

## Compositional self-improvement over text-to-SQL with turn-level decompositions

**Primary dataset:** SParC (question sequences over Spider databases), with Spider 1.0 as the end-to-end surface and test-suite execution for evaluation and guarding.

**Main setting:** monotone question sequences, database schema in context, turn structure visible to the pseudo-label generator. The atomic operation is a single-turn SQL edit.

**Core curriculum:** turn-1 single-clause queries → 2-turn sequences → 3-turn sequences → 4+-turn sequences; hardness tiers within each length.

**Primary training target:** SQL generation for the final turn (equivalently, for the matched Spider question); scored by execution.

**Compute profile:** low to moderate. Each pseudo-label requires one short generation call per turn plus SQLite execution. Target model: Qwen3.5-4B. Qwen3-1.7B is exploratory only (§14).

---

## 1. Executive summary

SParC decomposes complex Spider-style questions into interactive sequences where each turn refines the previous query:

```text
DB: flights(origin, dest, airline, ...)

Turn 1: "Show all flights."
        -> SELECT * FROM flights
Turn 2: "Only those departing from LAX."
        -> SELECT * FROM flights WHERE origin = 'LAX'
Turn 3: "Which airline runs the most of them?"
        -> SELECT airline FROM flights WHERE origin = 'LAX'
           GROUP BY airline ORDER BY COUNT(*) DESC LIMIT 1
```

The CSI mapping treats the running SQL query as the composition state (the analog of MuSiQue's bridge entity):

- **atom:** one turn — (current SQL, turn question) → next SQL;
- **composition:** apply turns in order; the final turn's SQL is the composed program;
- **pseudo-label:** the final SQL and its execution result for the sequence's overall intent;
- **guard:** parses, executes on the database, references only valid schema elements, and two independently generated candidates agree denotationally on test-suite database instances.

Gold SQL per turn is hidden and used only for audit — SParC releases it, so per-turn error localization is exact, as in KQA Pro.

The seed model is trained on turn-1 questions and Spider-easy single-clause queries from disjoint databases. Round 1 composes 2-turn sequences, round 2 uses the round-1 model on 2-turn prefixes to label 3-turn sequences, and round 3 extends to 4+ turns.

This benchmark connects the reasoning track (MuSiQue, DROP, KQA Pro) to the coding track: the output is executable code, the guard is the execution environment, and the composition state is a program rather than an entity or value.

---

## 2. Research question

> Given a visible turn decomposition, can guarded composition of single-turn SQL edits create sufficiently clean full-query supervision to improve a small model's direct text-to-SQL on complex questions?

A positive result should show that:

- guarded turn-by-turn composition produces more accurate final SQL than direct whole-query generation at matched hardness;
- retraining on composed queries improves direct generation on same-hardness questions;
- the round-1 model can serve as a prefix solver for longer sequences;
- improvements transfer from the sequence interface to single-shot Spider questions;
- denotational agreement between two views is a sufficient guard without any gold access.

---

## 3. Why SParC fits the framework

Spider contains 10,181 questions over 200 databases (train 8,659 / dev 1,034; test originally hidden). SParC contains 4,298 sequences (~12k questions) built by decomposing Spider-derived intents over the same databases; exact counts pinned at download. Spider-SS additionally aligns single-turn Spider questions to clause-level sub-sentences and is an optional second decomposition source (§5.4).

| CSI object | SParC instantiation |
|---|---|
| atomic input | (schema, current SQL, one turn question) |
| atomic output | the next SQL query |
| composed input | the full turn sequence |
| composition | sequential query editing; state is the running SQL |
| parent label | final-turn SQL + execution result |
| guard | parse, execution, schema-validity, denotational two-view agreement |
| frontier | sequence length and hardness tier |

### The monotonicity constraint

SParC turns are not uniformly refinements: some turns shift topic or drop constraints. The pipeline uses only **monotone sequences**, defined by an AST check — each turn's gold SQL must be derivable from the previous turn's by additions and single-clause modifications. Measure and report the monotone fraction during preprocessing; the framework's composition story only covers that subset, and the paper must state the coverage number. Non-monotone sequences go to a held-out transfer slice, never to training.

---

## 4. Pre-registered isolation screen (gate)

Run before any other implementation, per MuSiQue plan §25 (BFCL and CLUTRR were discontinued at this gate; MuSiQue passed only after seeding, §26 there).

**Setup:** base Qwen3.5-4B, no seed training, 100 paired dev instances per length cell (2, 3, 4+ turns), monotone sequences only, per-instance logging for McNemar. Scoring: test-suite execution accuracy of the final query.

**Arms:**

| arm | description |
|---|---|
| direct | final-turn intent as a standalone question (the matched Spider-style question) → whole SQL, one call |
| part | each turn alone, gold previous SQL provided |
| composed (self-fed) | turns applied in order, each conditioned on the model's own previous SQL |
| corrupt | composed, with the previous SQL at one turn replaced by SQL from a different sequence on the same database |

**Screening criterion:** `p(part) − p(composite)` clearly positive, and composed beats direct with CI excluding zero, at base or after a turn-1 seed.

**Corruption check:** corrupt must fall substantially below composed; otherwise the final turn alone determines the query and composition is not load-bearing.

**Oracle accounting (standing rule from MuSiQue §25):** the turn decomposition is released gold structure, analogous to `question_decomposition`; arms consuming it are structure-provided and need matched no-structure counterparts before headline claims. Database execution is the task environment and legitimately available to every arm. Gold SQL per turn is audit-only.

**Verdict rule (pre-registered):** proceed only if composed headroom over direct is positive with 95% CI lower bound above zero pooled, McNemar p < .05. Additional feasibility bar specific to this benchmark: base or seeded part accuracy must reach ≥ .60 — below that, the 4B model cannot support the pipeline and the benchmark is deferred rather than redesigned.

---

## 5. Experimental variants

### 5.1 Primary: SParC-CSI-Turns-Visible

Input: schema serialization, turn questions in order, running SQL state. Model emits one query per turn.

### 5.2 Transfer: Spider-Single-Shot

Input: schema plus a single complex question, no turns. Pseudo-labels are generated through the hidden turn pipeline; the student answers in one pass. Evaluation on Spider dev by hardness tier.

### 5.3 Phase 2: cross-database transfer

Train on one subset of databases, evaluate on unseen databases (Spider's defining split). Schema generalization is the realistic difficulty axis for text-to-SQL and must be reported separately from length generalization.

### 5.4 Optional: Spider-SS clause decomposition

Spider-SS sub-sentence alignments give a second, finer decomposition source for single-turn Spider questions. Use only if SParC monotone coverage (§3) is too small.

---

## 6. Scope

### Included in the minimum viable experiment

- SParC monotone sequences and matched Spider questions;
- SELECT/WHERE/GROUP BY/ORDER BY/LIMIT clause vocabulary;
- 2-turn, then 3-turn, then 4+-turn sequences;
- SQLite execution with distilled test suites;
- Qwen3.5-4B with LoRA r16/a32;
- two or three CSI rounds.

### Excluded initially

- non-monotone sequences (held-out slice only);
- nested subqueries and set operations (INTERSECT/UNION/EXCEPT) — these define the extra-hard tier and enter only in round 3 if at all;
- multi-database joins beyond two tables;
- CoSQL (dialogue acts add an orthogonal difficulty);
- schema linking as a separate model;
- 1.7B as a primary model.

---

## 7. Data acquisition, versioning, and leakage control

1. Download Spider 1.0, SParC, and the test-suite databases (distilled multi-instance DBs for denotational checking); pin versions and hashes.
2. Verify the SParC-to-Spider database mapping; both use the same 200 databases and the train/dev database split must be respected jointly.
3. **Partition at the database level.** Queries over the same database share schema and content; the seed/composition split must not leak schemas:

| Partition | Suggested fraction | Use |
|---|---:|---|
| Seed-source | 30% of train databases | turn-1 and Spider-easy queries with gold SQL |
| Composition-source | 60% of train databases | hide gold; construct multi-turn pseudo-labels |
| Internal audit/test | 10% of train databases | hidden evaluation and threshold tuning |

4. Spider/SParC dev databases remain untouched by training and threshold tuning.
5. Deduplicate templated question variants within a database.

### Anti-memorization rule

Seed queries and composition sequences must come from disjoint databases. Database-level disjointness subsumes question-level disjointness and also prevents value memorization (cell values appear in queries as literals).

---

## 8. Canonical data representation

```json
{
  "sequence_id": "...",
  "db_id": "flight_2",
  "turns": [
    {"turn_id": 1, "question": "Show all flights.", "gold_sql_hidden": "SELECT * FROM flights"},
    {"turn_id": 2, "question": "Only those departing from LAX.", "gold_sql_hidden": "..."},
    {"turn_id": 3, "question": "Which airline runs the most of them?", "gold_sql_hidden": "..."}
  ],
  "final_intent_question": "Which airline runs the most flights departing from LAX?",
  "monotone": true,
  "turn_count": 3,
  "hardness": "medium"
}
```

Gold SQL lives in a separate audit file that the generation process cannot import. Schema serializations are precomputed and versioned (table names, column names with types, foreign keys; no cell values except a bounded sample per column for literal grounding, with the sampling seed pinned).

---

## 9. Model interfaces

### 9.1 Turn mode

```text
[MODE: TURN]
Schema: flights(origin TEXT, dest TEXT, airline TEXT, ...)
Current SQL: SELECT * FROM flights WHERE origin = 'LAX'
Request: Which airline runs the most of them?
Return only the new SQL.
```

### 9.2 Visible-sequence mode

```text
[MODE: SEQ]
Schema: ...
Turn 1: Show all flights.
Turn 2: Only those departing from LAX.
Turn 3: Which airline runs the most of them?
Return only the SQL for Turn 3.
```

### 9.3 Single-shot mode

```text
[MODE: SINGLE]
Schema: ...
Question: Which airline runs the most flights departing from LAX?
Return only the SQL.
```

---

## 10. Composition algorithm

1. Initialize the state as empty.
2. For each turn, prompt in turn mode with the accepted state and the turn question; the emitted SQL becomes the new state after passing local guards.
3. Rejection at any turn rejects the sequence at that node.
4. The final state is the pseudo-label SQL; its execution result on the test-suite instances is stored with it.

Roundwise sub-DAG composition: at frontier `h_t`, the current model may solve a prefix of up to `h_t` turns in one visible-sequence call; the remaining turns proceed turn-by-turn. Prefix cuts are always available because sequences are linear — the articulation-point concern from MuSiQue §19 Risk 5 does not arise.

Store the full trace: per-turn SQL, execution results (row counts and a bounded result sample), rejection causes.

---

## 11. Guarded aggregation

### 11.1 Static checks

Accept a turn's SQL only when:

- it parses under the SQLite grammar;
- every table and column reference exists in the schema;
- literals type-match their comparison columns;
- the edit is monotone with respect to the previous state (AST diff is additions/single-clause modifications), matching the sequence-construction constraint.

### 11.2 Execution checks

- executes without error on all test-suite instances of the database;
- result-shape sanity: the sink query's column count and types are consistent with the turn question's expected answer form ("which airline" → one text column; "how many" → one numeric value);
- an empty result on every test-suite instance is a rejection for turns whose question presupposes existence; report how often this fires, since some questions legitimately return empty.

### 11.3 Denotational agreement guard

Generate the turn twice (two fixed prompt templates, greedy each). Accept when both candidates return identical results on every test-suite database instance. This is the strongest gold-free guard available in this domain: multi-instance execution makes coincidental agreement rare. No large sampling.

### 11.4 Guard levels

1. no guard;
2. static only;
3. static + execution;
4. full guard with denotational agreement.

Gold SQL is used to measure guard precision and recall after generation, never for acceptance.

---

## 12. Curriculum and pseudo-label pools

| Stage | Content | Target accepted examples |
|---|---|---:|
| Seed | turn-1 questions + Spider-easy, gold SQL, seed-source DBs | 3,000–6,000 |
| Round 1 | 2-turn monotone sequences | 2,000–3,000 |
| Round 2 | 3-turn sequences, round-1 model on 2-turn prefixes | 1,500–2,500 |
| Round 3 | 4+-turn sequences; optionally the hard tier | 1,000–2,000 |

SParC is an order of magnitude smaller than MuSiQue after monotone filtering; the targets above must be revalidated against the measured monotone coverage, and Spider-SS (§5.4) is the fallback volume source.

Training mixture per round: 30–40% seed replay, 20–30% prior-round tasks, 20–30% current-round turn/sequence tasks, 20–30% current-round single-shot tasks.

---

## 13. Training targets

1. **SQL-only** (primary): the final query. Intermediate turns are trained as separate turn-mode examples, each with its accepted state as input.
2. **Edit-trace** (ablation): the sequence of per-turn queries concatenated. Increases exposure to intermediate noise; flagged ablation, as with trace targets in the sibling plans.

---

## 14. Model and compute configuration

- Qwen3.5-4B, LoRA r16/a32;
- context limit 2,048–4,096 tokens (schema serializations dominate; measure the distribution before fixing the limit);
- output limit 128 tokens per query;
- greedy decoding, no chain-of-thought;
- SQLite and test-suite execution co-located with generation.

**1.7B status:** exploratory. Prior text-to-SQL results indicate 1.7B-class models are unreliable at combined SQL syntax + schema linking. Run the screen once at 1.7B for the record; commit only to 4B unless the 1.7B screen passes its feasibility bar (§4).

---

## 15. Baselines

- **B0** — seeded model, direct single-shot generation (mandatory seeded baseline, MuSiQue §26).
- **B1** — direct pseudo-labeling: single-shot generation on composition-source questions, guarded at guard level 3 (matched final-query guard).
- **B2** — frozen seed executing every turn at all lengths, no retraining.
- **B3** — frozen frontier prefix execution without training a direct student.
- **B4** — unfiltered turn composition.
- **B5** — guarded turn composition with retraining (proposed method).
- **B6** — gold-prefix execution: each turn conditioned on gold previous SQL (exact cascade diagnostic).
- **B7** — gold-SQL supervised upper bound, matched counts and curriculum.
- **B8** — direct model with turns visible at inference, no pseudo-labeling.

---

## 16. Evaluation and required analyses

### Primary metrics

- test-suite execution accuracy (primary; plain single-instance execution accuracy reported alongside for comparability with the literature);
- exact-set-match accuracy (secondary);
- accuracy by turn count and Spider hardness tier;
- cross-database generalization gap (§5.3).

### Pseudo-label metrics

- final-query test-suite accuracy against hidden gold;
- per-turn accuracy against hidden gold turn SQL (exact, released);
- guard acceptance rate, precision among accepted, rejection-cause distribution;
- first-error turn index.

### 16.1 Length-frontier heatmap

Rows: turn counts; columns: seed and CSI rounds; panels for sequence-visible and single-shot inference.

### 16.2 Error propagation

`P(final correct | k turns)` versus the product of per-turn accuracies, using B6 to separate cascade cost from per-turn difficulty. Compare the observed regime with BFCL (product-matched) and MuSiQue (above product).

### 16.3 Guard tradeoff

Precision versus acceptance across guard levels, with attention to 11.3: how much precision the denotational agreement adds over execution alone, and at what acceptance cost.

### 16.4 Retraining versus execution

Direct CSI versus B2/B3 at matched generation budgets.

### 16.5 False-positive audit of execution guards

Sample accepted-but-wrong queries (accepted by guard level 3, wrong against gold) and classify: coincidental denotation match, under-constrained question, gold ambiguity. This calibrates how far execution guarding can be trusted on databases without test suites.

---

## 17. Ablations

1. direct versus composed pseudo-labels;
2. frozen execution versus retraining;
3. single-turn atoms versus frontier-sized prefixes;
4. guard levels 2/3/4;
5. SQL-only versus edit-trace training;
6. single-instance execution versus test-suite execution in the guard;
7. monotone-only training versus adding non-monotone sequences;
8. schema serialization with versus without cell-value samples.

---

## 18. Statistical protocol

- Thresholds tuned only on internal audit partitions; Spider/SParC dev after pipeline freeze.
- Per-instance outcome logging in every arm (MuSiQue §25 rule).
- Matched accepted-example counts across B1/B4/B5.
- Three final seeds; paired bootstrap CIs over sequences, clustered by database.

---

## 19. Risks and mitigations

### Risk 1 — Part accuracy below the feasibility bar

Text-to-SQL atoms are harder than QA atoms; a 4B model may sit below .60 per turn. **Mitigation:** the screen's feasibility bar (§4) defers the benchmark before pipeline investment. Do not redesign the task downward (e.g., to slot filling) to force a pass; that changes the claim.

### Risk 2 — Monotone filtering shrinks the pool below training volume

**Mitigation:** measure coverage in week 0; fall back to Spider-SS decompositions (§5.4); revalidate §12 targets. If both sources together yield under ~1,500 round-1 sequences, discontinue.

### Risk 3 — Execution-guard false positives

Wrong SQL can return correct denotations. **Mitigation:** test suites (multi-instance) as primary; §16.5 quantifies the residual rate; exact-set-match reported alongside.

### Risk 4 — Schema-length pressure on context

200 databases vary widely in schema size. **Mitigation:** measure the serialized-schema length distribution; exclude databases exceeding the context budget rather than truncating schemas silently; report the exclusion rate.

### Risk 5 — Cross-database transfer masks composition gains

Length generalization and schema generalization are different axes; a method can win one and lose the other. **Mitigation:** §5.3 reports them separately; the headline claim is scoped to the axis that improves.

### Risk 6 — Value-literal grounding

WHERE-clause literals require knowing cell-value formats. **Mitigation:** bounded cell-value samples in the schema serialization (§8), ablated in §17.8.

---

## 20. Go/no-go criteria

Proceed past round 1 only if:

- screen verdict rule and feasibility bar met (§4);
- seed turn-1/easy accuracy ≥ .70 test-suite execution on held-out seed-source databases;
- guarded 2-turn pseudo-label accuracy exceeds direct (B1) by ≥ 5 points;
- accepted pseudo-label precision ≥ .85 test-suite execution;
- ≥ 1,000 2-turn sequences pass the guard (lower than sibling plans; SParC is smaller);
- round-1 single-shot accuracy improves ≥ 3 points over B0 on matched hardness;
- turn-1 regression after replay ≤ 2 points.

---

## 21. Expected paper claims

> When complex text-to-SQL questions expose a turn-level decomposition, single-turn edits validated by multi-instance execution compose into higher-precision supervision than direct whole-query generation, improving a small model's single-shot parsing within the covered clause vocabulary.

The experiment does not establish automatic decomposition, does not cover non-monotone interactions, and the execution guard's strength depends on test-suite availability.

---

## 22. Implementation schedule

### Week 0 — Isolation screen (gate)

- download and pin Spider, SParC, test suites;
- AST monotonicity checker; measure monotone coverage;
- run the four screen arms at 4B (and once at 1.7B for the record); apply the verdict rule.

### Week 1 — Partitions, seed, guards

- database-level partitions; schema serializations;
- seed training; guard levels; audit 2-turn pseudo-label precision.

### Week 2 — Round 1 (2-turn)

- B1/B2/B4/B5 labelers; round-1 training; internal evaluation.

### Week 3 — Rounds 2–3 and analyses

- prefix cuts; error propagation, guard tradeoff, false-positive audit.

### Week 4 — Transfer and final runs

- single-shot Spider evaluation by hardness; cross-database slice; three-seed finals.

---

## 23. Reproducibility checklist

- [ ] Spider/SParC/test-suite versions and hashes pinned.
- [ ] Database-level partition disjointness asserted at run time.
- [ ] Gold SQL stored outside generation code.
- [ ] Monotone coverage measured and reported.
- [ ] Schema serialization versioned; cell-sample seed pinned.
- [ ] Per-instance outcomes logged in every arm.
- [ ] Guard thresholds fixed before dev evaluation.
- [ ] Matched budgets across B1/B4/B5.
- [ ] Single-shot evaluation hides the turn decomposition.
- [ ] Results sliced by turn count, hardness, and seen/unseen database.

---

## 24. Primary sources

- Yu, T., et al. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL](https://aclanthology.org/D18-1425/).
- Yu, T., et al. (2019). [SParC: Cross-Domain Semantic Parsing in Context](https://aclanthology.org/P19-1443/).
- Zhong, R., Yu, T., Klein, D. (2020). [Semantic Evaluation for Text-to-SQL with Distilled Test Suites](https://aclanthology.org/2020.emnlp-main.29/).
- Gan, Y., et al. (2022). [Spider-SS: Decomposing a Sentence into Sub-sentences for Text-to-SQL](https://aclanthology.org/2022.findings-acl.99/) (optional decomposition source).
- Official repositories: [taoyds/spider](https://github.com/taoyds/spider), [taoyds/sparc](https://github.com/taoyds/sparc), [ruiqi-zhong/TestSuiteEval](https://github.com/ruiqi-zhong/TestSuiteEval).

---

## 25. Data, monotone coverage, and pool volume, 2026-08-06

Week 0 preprocessing (§22). Artifacts: `reports/composition_screen/sparc_fetch.py`,
`sparc_data.py`, `data/sparc_pins.json`, `data/sparc_manifest.json`,
`data/sparc_data_report.json`. Composition primitives and their tests:
`self/coding/sparc_composition.py`, `tests/test_sparc_composition.py` (31 tests).

Versions pinned by sha256: Spider 1.0 (7,000 / 1,659 / 1,034 questions, 166
databases), SParC (3,034 train / 422 dev sequences), test-suite databases
(3,194 sqlite files over 28 database directories).

### Monotone coverage

Measured by the AST checker of §3, implemented over sqlglot with clause-atom
diffing: a turn is monotone when at most one clause category has removals;
set operations, subqueries, and self-joins are classified as structure changes.

| split | 2 turns | 3 turns | 4+ turns |
|---|---:|---:|---:|
| train | .841 (667/793) | .668 (1036/1551) | .576 (393/682) |
| dev | .849 (129/152) | .676 (123/182) | .568 (50/88) |

Non-monotone causes across all sequences: structure change 1,001, multi-clause
change 498, parse error 15, previous turn unparseable 3, schema resolution
error 1.

### Parser finding

Spider and SParC gold SQL quotes string values with double quotes
(`WHERE name = "John Dorian"`). Under the SQL standard those are identifiers,
so schema qualification attempted to resolve them as columns. Before the
rewrite in `normalize_quotes`, 1,754 sequences were marked unusable with
schema resolution errors; after it, 1. The official Spider evaluation applies
the same rewrite.

### Pool volume

| pool | count | plan target |
|---|---:|---:|
| Seed: turn-1 questions from seed-source databases | 913 | — |
| Seed: Spider-easy from seed-source databases | 428 | — |
| Seed total | 1,341 | 3,000–6,000 (§12) |
| Composition-source monotone 2-turn | 431 | ≥ 1,500 (§19 Risk 2) |
| Composition-source monotone 3-turn | 625 | — |
| Composition-source monotone 4+-turn | 191 | — |
| Internal audit evaluation cells (2 / 3 / 4+) | 31 / 92 / 51 | — |
| Dev screen cells (2 / 3 / 4+) | 129 / 123 / 50 | 100 per cell (§4) |

**Risk 2 tripwire fired.** The composition-source 2-turn pool is 431 against a
bar of 1,500. Under §19 Risk 2 this alone requires either the Spider-SS
fallback (§5.4) or discontinuation before round 1.

### Schema length and test-suite coverage

Serialized schemas measured with the Qwen3.5-4B tokenizer: minimum 58,
median 222, maximum 2,433 tokens. At a 4,096-token context with a 512-token
prompt reserve, the exclusion rate is 0 (§19 Risk 4 does not bind).

Distilled multi-instance test suites cover 20 of 20 dev databases and 5 of 140
train databases. The denotational agreement guard (§11.3) therefore runs on
single-instance execution over composition-source databases unless suites are
distilled for them, which weakens that guard relative to the design. The
labeler records `n_suite_instances` per turn so the difference stays visible
(§17.6).

### Scorer audit

The test-suite comparator is reimplemented rather than imported from the
official repository (module docstring records the reason). Audit over 300
sampled gold queries: gold-versus-gold agreement 300/300; denotation-changing
mutations rejected in 79.7% of cases. The residual 20.3% consists of mutations
that do not change the denotation on the available instances, which is the
same false-positive mechanism §16.5 was written to quantify.

---

## 26. Isolation screen, 2026-08-06

Pre-registered gate of §4. Base Qwen3.5-4B, no seed training, monotone dev
sequences, 100 paired instances at 2 and 3 turns and 50 at 4+ turns, all four
arms on the same instances, scored by test-suite execution accuracy of the
final query. Artifacts: `sparc_isolation.py`, `sparc_isolation.json`,
`sparc_isolation_per_instance.jsonl`.

| turns | n | direct | part | composed | corrupt |
|---|---:|---:|---:|---:|---:|
| 2 | 100 | .590 | .615 | .480 | .190 |
| 3 | 100 | .540 | .597 | .400 | .220 |
| 4+ | 50 | .480 | .635 | .320 | .120 |
| **all** | **250** | **.548** | **.613** | **.416** | **.188** |

Pooled composed − direct = **−.132**, 95% CI [−.196, −.068], McNemar p = .0001.
The chain produced extractable SQL at every turn for 100% of instances, so the
gap is not a protocol-execution failure.

### Verdict against the pre-registered rule

| criterion | value | met |
|---|---|---|
| headroom CI lower bound above zero | [−.196, −.068] | no |
| McNemar p < .05 | .0001 | yes (in the direction of direct) |
| part ≥ .60 feasibility bar | .613 | yes |
| corrupt substantially below composed | drop .228 | yes |

Composition is load-bearing: replacing the running SQL at one turn with SQL
from a different sequence on the same database costs .228. Parts are also
easier than composites, +.197 pooled, which is the §4 screening criterion
`p(part) − p(composite)`. The criterion that fails is composed versus direct.

### 1.7B record run

Run once for the record per §14. Pooled over the same cells: direct .488,
part .543, composed .304, headroom −.184, corrupt drop .164. Part accuracy
.543 is below the .60 feasibility bar, so 1.7B is not viable for this
benchmark and no further 1.7B work is warranted.

---

## 27. Seed and seeded re-screen, 2026-08-06 -- verdict and discontinuation

§4 permits the screen to pass "at base or after a turn-1 seed", and the base
screen met the feasibility bar, so the seeded re-screen was run as the
remaining branch of the gate. Artifacts: `sparc_seed.py`,
`sparc_seed/sparc_seed.json`, `sparc_isolation_seeded.json`.

Seed configuration: LoRA r16/a32 on Qwen3.5-4B, 1,341 examples (913 turn-1
edits in turn mode with empty state, 428 Spider-easy questions in single mode,
interleaved), 300 steps, lr 2e-4, `max_length` 4096, gradient checkpointing,
final train loss .162, runtime 2h19m. Training databases are the 30%
seed-source partition; database-level disjointness from audit and dev
databases is asserted at run time.

### Seed effect, internal-audit databases

| phase | turn-1 | direct | part | composed | headroom | 95% CI | McNemar |
|---|---:|---:|---:|---:|---:|---|---|
| before | .747 | .730 | — | .523 | −.207 | [−.282, −.132] | p < .0001 |
| after | .793 | .684 | — | .534 | −.149 | [−.236, −.063] | p = .0013 |

The §20 seed bar (turn-1 ≥ .70 test-suite execution) is met at .793.

### Seeded re-screen, dev databases

| turns | n | direct | part | composed | corrupt |
|---|---:|---:|---:|---:|---:|
| 2 | 100 | .730 | .705 | .600 | .190 |
| 3 | 100 | .620 | .667 | .480 | .300 |
| 4+ | 50 | .480 | .630 | .240 | .120 |
| **all** | **250** | **.636** | **.667** | **.480** | **.220** |

Pooled composed − direct = **−.156**, 95% CI [−.224, −.092],
McNemar p = 9.8e-6. Corrupt drop .260.

Seeding raised both arms on the dev slice: direct .548 → .636 (+.088),
composed .416 → .480 (+.064). Pooled headroom therefore changed from −.132 to
−.156. On the internal-audit slice headroom narrowed from −.207 to −.149. The
two slices are different database sets and the pre-registered screen surface
is dev.

### Error propagation

Observed composed accuracy exceeds the product of per-turn part accuracy in
every cell (seeded: .600 versus .497 at 2 turns, .480 versus .296 at 3 turns,
.240 versus .158 at 4 turns). The cascade is therefore milder than independent
per-turn errors would predict, which places SParC in the same regime as
MuSiQue rather than the product-matched regime of BFCL (§16.2). This does not
change the verdict: direct whole-query generation is stronger than the chain
at every turn count.

### Verdict

The pre-registered verdict rule of §4 requires pooled composed headroom over
direct positive with 95% CI lower bound above zero. That condition fails at
base (−.132) and after seeding (−.156), with McNemar significant in the
direction of direct in both cases. The feasibility bar and the corruption
check are both met, so this is a failure of the composition-advantage
criterion rather than a model-capability deferral.

**SParC-Spider-CSI is discontinued at the gate**, alongside BFCL and CLUTRR.
Per §19 Risk 1 the task is not redesigned downward to force a pass. The
independent Risk 2 tripwire (§25: 431 composition-source 2-turn sequences
against a bar of 1,500) would have required the Spider-SS fallback or
discontinuation regardless of the screen outcome.

### Observations for sibling plans

**Direct whole-query generation is a strong baseline in this domain.** Seeded
direct accuracy is .636 pooled and .730 at 2 turns. The turn decomposition
supplies information the model does not need for these questions: SParC turn
sequences decompose Spider-derived intents whose direct form the model already
parses at that rate.

**Per-turn accuracy near .67 is insufficient for chains of length 2 or more
even without independent errors.** MuSiQue cleared its screen with part
accuracy .72–.77 and 2-hop composed .720; the operative difference here is
that the composed arm must carry an exact SQL string forward, where a
recoverable paraphrase in the MuSiQue setting corresponds to a query that
executes to a different denotation.

**The scaffold is complete and unrun past the gate.** Round-1 labeling and
training scripts, guard levels, and the baseline harness exist and pass smoke
runs (§25 artifacts plus `sparc_round1_label.py`, `sparc_round1_train.py`,
`sparc_eval.py`). If Spider-SS decomposition is later adopted as the
decomposition source (§5.4), the screen is the only stage that must be rerun
before the pipeline can proceed.
