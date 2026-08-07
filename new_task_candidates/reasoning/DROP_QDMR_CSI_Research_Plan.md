# Research Plan: DROP-QDMR-CSI

> **Status: discontinued 2026-08-06 at the §4 gate.** Composed − direct is
> −.377 at base and −.290 after seeding, both with 95% CIs excluding zero and
> McNemar p < 1e-4 on 300 paired dev instances. §25 records the outcome and the
> findings; the sections before it are the plan as it stood when the gate was
> run, corrected against measured data.

## Compositional self-improvement over discrete-reasoning questions with visible QDMR decompositions

**Primary dataset:** DROP (single-passage discrete reasoning) with QDMR decompositions from BREAK.

**Main setting:** DROP questions whose BREAK QDMR is available, single gold passage in context, QDMR structure visible to the pseudo-label generator.

**Core curriculum:** 2 → 3 → 4 model-owned extraction nodes per QDMR DAG. Executor-owned nodes are free and do not count against the frontier (§10). QDMR *step* count is not the frontier axis; see §3 for the measured distributions.

**Primary training target:** direct answer prediction for the original DROP question; a structured trace target is an optional secondary condition.

**Compute profile:** low. Each pseudo-label requires two to four short extraction calls plus deterministic symbolic execution. Target models: Qwen3.5-4B primary, Qwen3-1.7B secondary.

---

## 1. Executive summary

DROP questions require discrete operations (subtraction, comparison, counting, superlatives) over spans extracted from a single passage. BREAK provides QDMR decompositions that expose the operation DAG.

A schematic example with 2 model-owned nodes and 1 executor-owned node:

```text
Passage: "...the Broncos scored on a 23-yard field goal ... the Raiders
answered with a 45-yard field goal..."
Question: How many yards longer was the Raiders' field goal than the Broncos'?

QDMR:
1. return the yards of the Raiders' field goal   -> 45     [model]
2. return the yards of the Broncos' field goal   -> 23     [model]
3. return the difference of #1 and #2            -> 22     [executor]
```

The released decomposition for a question of this shape is usually longer, because BREAK splits entity selection from attribute projection. `DROP_train_nfl_1092` decomposes the same pattern into 7 steps — SELECT, PROJECT, PROJECT, SELECT, PROJECT, PROJECT, ARITHMETIC — of which 6 are model-owned. §3.2 gives the measured distributions.

The framework split is:

- **extraction nodes** (SELECT, PROJECT, FILTER): answered by the model, one short call per node, passage in context, returning a list of spans;
- **symbolic nodes** (ARITHMETIC, AGGREGATE, COMPARISON): executed deterministically in Python over the lists their parents produced.

The composed sink value becomes the pseudo-label for the original question. This differs from MuSiQue in three ways that matter for the framework:

1. part of the composition operator is exactly executable, so a subset of nodes contributes zero label noise;
2. BREAK provides no gold intermediate values, so intermediate-error audits require executing hand-checks rather than reading released annotations (§16.2), and seed supervision cannot come from QDMR steps (§7.1);
3. the atomic output is a list rather than a single span, because the largest executor family (count and sum) consumes lists (§3.3).

The seed model is trained on DROP span-answer questions from a disjoint passage partition. BREAK releases no gold intermediate values and contains 23 single-step DROP decompositions in total, so extraction supervision comes from DROP's own span-type questions rather than from QDMR steps (§7). Round 1 composes DAGs with 2 model-owned nodes, round 2 composes 3-node DAGs using the round-1 model on 2-node sub-DAGs, and round 3 extends to 4 nodes.

The main scientific comparison mirrors the MuSiQue plan:

- direct full-question pseudo-labeling;
- frozen step-by-step execution with the seed model;
- guarded QDMR composition plus retraining;
- gold-supervised training.

The added scientific value over MuSiQue: the composition operators are arithmetic and set operations rather than entity substitution, so a positive result shows the framework covers a second operator family; and the numeric guard is stronger than MuSiQue's span guard.

---

## 2. Research question

> Given a visible QDMR DAG whose non-extraction steps are exactly executable, can guarded composition of small-model extraction answers create sufficiently clean discrete-reasoning supervision to improve a small model's direct DROP accuracy?

A positive result should show that:

- guarded QDMR execution produces more accurate pseudo-labels than direct prediction at matched frontier cells;
- retraining on k=2 pseudo-labels improves direct k=2 accuracy;
- the round-1 model can serve as a sub-DAG solver for k=3 composition;
- improvements transfer from the QDMR-visible interface to the original end-to-end question interface;
- the gain is larger on op families whose symbolic tail is longer (more of the label is computed exactly).

---

## 3. Why DROP + QDMR fits the framework

DROP contains 77,409 train questions over 5,565 passages and 9,536 dev questions over 582 passages. BREAK provides QDMR for about 10% of them.

### 3.1 Measured coverage

Counts below were obtained by streaming `break_dataset/logical-forms/{train,dev}.csv` from `allenai/Break` and filtering on the `DROP_` question-id prefix, 2026-08-06. They must be reproduced by the build script (§7) and any deviation treated as a version change.

| Quantity | BREAK train | BREAK dev |
|---|---:|---:|
| DROP questions with QDMR | 7,672 | 1,265 |
| Distinct passages | 3,624 | 247 |
| Mean questions per passage | 2.12 | 5.12 |
| Numeric-sink rows | 4,228 | 654 |
| Rows whose model-owned nodes all return scalars and whose sink is ARITHMETIC | 1,001 | 150 |
| Rows with a 1-step QDMR | 23 | 3 |
| Rows with a 2-step QDMR | 616 | — |

Domain split in train: `history` 4,459 questions, `nfl` 3,213.

### 3.2 Step counts and the frontier axis

QDMR step counts for DROP concentrate at 3–6 steps (train: 1,859 at 3 steps, 1,441 at 4, 1,921 at 5, 708 at 6). One- and two-step decompositions are rare, so a curriculum indexed on step count has no low-frontier cell to start from.

Counting only model-owned nodes (SELECT, PROJECT, FILTER, BOOLEAN) gives a usable axis, because executor-owned nodes contribute no model error and are free under the frontier rule of §10:

| model-owned nodes k | train, all sinks | train, numeric sinks | dev, numeric sinks |
|---:|---:|---:|---:|
| 1 | 82 | 56 | 12 |
| 2 | 2,329 | 1,394 | 214 |
| 3 | 2,290 | 1,198 | 162 |
| 4 | 2,225 | 1,115 | 181 |
| 5 | 344 | 183 | 35 |
| 6+ | 402 | 282 | 50 |

Cells k = 2 / 3 / 4 carry 1,394 / 1,198 / 1,115 numeric-sink train rows, which is the structure the MuSiQue experiment used at 2 / 3 / 4 hops.

### 3.2.1 Pools after operator exclusion

A numeric-sink DAG is unusable if any node anywhere in it carries an operator outside the executor table (§6). Measured by `drop_qdmr_build.py`:

| | train | dev |
|---|---:|---:|
| numeric-sink candidates | 4,228 | 654 |
| blocked by a mid-DAG excluded operator | 364 (8.6%) | 55 (8.4%) |
| **usable numeric-sink DAGs** | **3,864** | **599** |
| usable at k = 2 / 3 / 4 | 1,234 / 1,051 / 1,078 | 181 / 149 / 176 |

Blocking operators in train: COMPARATIVE 237, DISCARD 96, INTERSECTION 27, GROUP 26, UNION 10, SUPERLATIVE 5.

The overall exclusion rate against all BREAK-covered DROP rows is 29.5% train and 33.5% dev, but that figure is dominated by COMPARISON (1,267 train rows), whose sink is non-numeric and therefore out of scope regardless. The rate that Risk 1 tests is the 8.6% above.

Every dev cell holds at least 149 usable rows, so the 100-instances-per-cell screen of §4 is drawable.

Executor-owned nodes on the sink path (the binning variable of §16.5): depth 1 for 3,081 train rows, depth 2 for 775, depth 3 for 8. The symbolic-tail analysis therefore has two populated bins rather than a range.

### 3.3 Operator distribution

Sink operators, train: AGGREGATE 2,243 (count 1,647, sum 423, max 80, min 65, avg 28), ARITHMETIC 1,985, COMPARISON 1,360, PROJECT 1,327, FILTER 304, COMPARATIVE 284, SUPERLATIVE 76, SELECT 25, DISCARD 21, INTERSECTION 21, UNION 15, GROUP 8, BOOLEAN 3.

Node operators, train: project 11,481, select 9,816, aggregate 5,353, filter 3,323, arithmetic 2,099, comparison 1,360, comparative 715, group 245, discard 171, boolean 151, superlative 111, intersection 48, union 31.

The count and sum family is the largest numeric slice. Its executor node consumes a **list** produced by an upstream PROJECT or FILTER node, so the atomic output type is a list of spans rather than a scalar (§6, §9.1, §11.1).

### 3.4 Join key

BREAK question IDs have the form `DROP_{split}_{section_id}_{query_id}`, where `section_id` is the DROP passage ID (`nfl_1087`, `history_442`) and `query_id` is the DROP question UUID. The join against DROP on `(section_id, query_id)` is exact and requires no fuzzy matching. Passage-level partitioning reads `section_id` directly off the BREAK ID.

| CSI object | DROP-QDMR instantiation |
|---|---|
| atomic input | one QDMR extraction step, verbalized, plus the passage |
| atomic output | a list of spans, possibly of length one |
| composed input | the QDMR DAG for the original question |
| composition | substitute step outputs into `#k` references; execute symbolic steps in Python |
| parent label | value at the QDMR sink |
| guard | type, range, list-completeness, execution-validity, and agreement checks |
| frontier | model-owned node count k, and op family |

The composition rule extends MuSiQue's: a predicted output changes a downstream input, and additionally some nodes are deterministic functions rather than model calls:

- extraction node: `a_v = M(q_v[a_parents], passage)`
- symbolic node: `a_v = f_op(a_parents)` with `f_op` exact.

Every symbolic node in the sink path reduces the surface over which model error can enter the label.

### Differences from MuSiQue that must be handled

1. **No gold intermediate values.** MuSiQue releases gold answers per node; BREAK releases only structure. This has two effects: intermediate audits use a hand-audited sample (§16.2), and seed supervision cannot be drawn from QDMR steps (§7).
2. **Answer types are heterogeneous.** Numbers, dates, single spans, and multi-spans require the official DROP metric. The initial scope restricts to number-typed sinks (§6).
3. **Atoms are list-valued.** AGGREGATE with fn ∈ {count, sum, max, min, avg} accounts for 2,243 of the sinks and consumes a list. The extraction interface therefore returns a list at every model-owned node, and recall failures on that list are silent (§11.1, Risk 4).
4. **QDMR annotation noise.** BREAK decompositions are crowdsourced; a fraction do not correctly reflect the question. The pipeline must measure and report the executable-and-consistent fraction rather than assume it is 1 (§19 Risk 1).

---

## 4. Pre-registered isolation screen (gate)

Run this before any other implementation, per the corrected criterion in MuSiQue plan §25. BFCL and CLUTRR were discontinued at this gate; MuSiQue passed it only after a one-hop seed (§26 there).

**Setup:** base Qwen3.5-4B, no seed training, 100 paired dev instances per frontier cell k ∈ {2, 3, 4} model-owned nodes, number-typed sinks only, giving 300 paired instances and matching the MuSiQue screen size. Usable dev availability per cell is 181 / 149 / 176 (§3.2.1), so 100 per cell is drawable. Log per-instance outcomes for McNemar; the earlier MuSiQue screen stored only aggregates and this blocked paired testing.

**Arms:**

| arm | description |
|---|---|
| direct | original question + passage, one call |
| atom proxy | DROP span-answer questions on the same passages, scored against gold |
| composed (self-fed) | full QDMR execution: model extraction steps, Python symbolic steps, own upstream values |
| corrupt | composed, with one non-sink model-owned value replaced by a value sampled from a different instance in the same passage domain |

**The `part` arm of the MuSiQue screen has no direct analogue here.** MuSiQue releases a gold answer per step, so part accuracy is measurable at every node. BREAK releases only structure (§3), so no intermediate node has a label and per-node accuracy cannot be scored at scale. Two substitutes are used:

- **atom proxy**, at scale: DROP's own span-answer questions on the same dev passages, which measure extraction difficulty on the same text under the same interface. The 214 dev passages carrying usable numeric DAGs hold 1,388 single-span and 231 multi-span questions;
- **hand audit**, on a sample: 100 traces per round, labeling the first incorrect node (§16.2).

**Known confound in the atom proxy.** The proxy scores span-answer questions while `direct` scores number-answer questions, so a difference between them mixes atom difficulty with answer-type difficulty. A 15-instance smoke run put the proxy below direct (.400 against .733), which is the wrong sign for the criterion below and is not interpretable at that sample size. Before drawing any conclusion from the proxy gap at full size, either restrict the proxy to matched answer types or report the criterion as informative only through the composed-versus-direct contrast.

**Screening criterion:** `p(atom proxy) − p(direct)` must be clearly positive (the atom must be easier than the composite), and the composed arm must beat direct with a CI excluding zero either at base or, failing that, after a seed (repeat of the MuSiQue §26 resolution path). Report the atom-proxy gap alongside the composed-minus-direct headroom, and state in every write-up that the first number is a proxy rather than a per-node measurement.

**Seed-then-rescreen is part of Week 0.** On MuSiQue the base-model screen returned +.023 with a CI spanning zero, and the same contrast reached +.080 (p=.0009) only after the one-hop seed was trained. Budget the seed run inside the gate rather than treating a null base screen as a discontinuation.

**Corruption check:** the corrupt arm must be substantially below composed; otherwise the sink extraction determines the answer alone and the composition is not load-bearing.

**Oracle accounting (standing rule from MuSiQue §25):** QDMR is released gold structure, exactly analogous to `question_decomposition`. Every arm that consumes it is structure-provided and needs a matched no-structure counterpart before any headline claim. There is no analog of the `paragraph_support_idx` oracle here because DROP is single-passage; this removes the retrieval confound that cost MuSiQue .117.

**Verdict rule (pre-registered):** proceed only if seed-or-base composed headroom over direct is positive with the 95% CI lower bound above zero on the pooled cells, McNemar p < .05.

---

## 5. Experimental variants

### 5.1 Primary: DROP-CSI-QDMR-Visible

Input includes the verbalized QDMR steps with `#k` references, the passage, and an instruction to produce the sink value. Symbolic steps are marked as executor-owned; the model is only queried on extraction steps.

### 5.2 Transfer: DROP-CSI-End2End

Input is the original DROP question and passage, no decomposition. Pseudo-labels are still generated through the QDMR pipeline; the student answers directly.

### 5.3 Phase 2: op-family transfer

Train on ARITHMETIC-sink questions only; evaluate on AGGREGATE-sink (count and sum) and COMPARISON-sink questions. This measures whether learned extraction and composition transfer across operator families. Pool sizes are 1,985 / 2,243 / 1,360 in train before partitioning (§3.3).

### 5.4 Optional: span- and date-typed sinks

Add after number-typed sinks work. PROJECT-sink rows (1,327 in train) are the largest span-typed family.

---

## 6. Scope

### Included in the minimum viable experiment

- DROP train/dev with BREAK QDMR available;
- number-typed sinks: ARITHMETIC sinks, and AGGREGATE sinks with fn ∈ {count, sum, max, min, avg}. The count and sum family is included rather than deferred, because it is the largest numeric slice (2,070 of 4,228 numeric-sink train rows) and excluding it leaves only the 1,001 scalar-arithmetic rows, which do not survive the passage partition at usable size;
- executor-owned ops: ARITHMETIC and AGGREGATE. COMPARISON is not needed: it occurs 1,614 times across DROP train and dev and is the sink every time (0 mid-DAG occurrences), and COMPARISON sinks are non-numeric, so no numeric-sink DAG contains one;
- model-owned ops: SELECT, PROJECT, FILTER;
- atomic output type is a **list of spans**, with scalar-valued steps as the length-one case;
- frontier cells k = 2, then k = 3, then k = 4 model-owned nodes;
- single passage in context (no retrieval anywhere in this benchmark);
- one small model (Qwen3.5-4B, LoRA r16/a32, matching the MuSiQue seed configuration);
- two or three CSI rounds.

### Excluded initially

- non-numeric sinks (PROJECT, FILTER, COMPARATIVE, SUPERLATIVE, SELECT sinks);
- GROUP, COMPARATIVE, SUPERLATIVE, DISCARD, UNION, INTERSECTION, SORT, BOOLEAN ops. Any DAG containing one of these sends the example to the excluded pool;
- questions without BREAK coverage, other than as seed-source material (§7);
- free-form rationales;
- large sampling budgets.

The exclusion rate is reported by operator at build time (§7). Excluding both the non-numeric sinks and the unimplemented operators leaves the 4,228 train / 654 dev numeric-sink rows of §3.1 as the upper bound on the usable pool, before the passage partition.

---

## 7. Data acquisition, versioning, and leakage control

1. Download DROP from the official release and BREAK `break_dataset/logical-forms/{train,dev}.csv` from `allenai/Break`; pin commits and record file hashes.
2. Join on `(section_id, query_id)` parsed from the BREAK question ID (§3.4); assert the join rate and record it.
3. BREAK train derives from DROP train and BREAK dev from DROP dev; verify this holds in the downloaded version and keep DROP dev untouched by training.
4. Reproduce the §3.1 counts. Any deviation is a version change and blocks the run.

### 7.1 Seed supervision source

The seed cannot be built from QDMR steps. BREAK releases no gold intermediate values, and it contains 23 single-step DROP decompositions in train, so there is neither a step-level label nor a single-step question pool. Seed supervision comes instead from **DROP span-answer questions**, whose gold answer is exactly one extraction result:

- single-span answers supply the length-one atom;
- multi-span answers supply the list atom that AGGREGATE nodes consume, which is the supervision the count and sum family requires.

Targets are rendered in the same list format the extraction interface uses (§9.1), so the seed teaches the output format alongside extraction.

Measured availability on the 1,941 DROP train passages that BREAK does not cover: 15,669 questions, of which 5,065 have a single-span answer and 835 have a multi-span answer, giving 5,900 extraction atoms. This is inside the 4,000–8,000 seed target of §12 and requires no BREAK-covered passage. All 3,624 BREAK-covered train passages are present in DROP train, so the passage-level join rate is 1.00.

### 7.2 Passage-level partition

DROP has many questions per passage; seed questions and composition questions over the same passage share extractable content. All partitions split on `section_id`.

| Partition | Source | Use |
|---|---|---|
| Seed-source | the 1,941 DROP train passages BREAK does not cover | span-type questions with gold answers, 5,900 available |
| Composition-source | 85% of the covered passages that carry a usable numeric DAG, 2,125 passages / 3,309 DAGs | hide answers; construct pseudo-labels |
| Internal audit | the remaining 15%, 375 passages / 555 DAGs | threshold tuning and hand audits |
| Evaluation | DROP dev, 247 BREAK-covered passages / 1,265 questions | model selection only, after the pipeline is fixed |

Drawing seed data from BREAK-uncovered passages satisfies the anti-memorization rule at no cost to the composition pool, and it is the only source with enough gold-labeled extraction supervision. The number of uncovered train passages is asserted at build time; the seed target of §12 is contingent on it.

5. Stratify passage partitions by domain (`nfl` 3,213 questions, `history` 4,459 in the covered set), passage length, and per-passage question count.
6. Deduplicate near-identical questions within a passage (DROP contains templated variants).

### Anti-memorization rule

Do not draw seed questions from passages later used for composition. The extraction atoms are passage-specific, so passage-level disjointness is the binding constraint (the MuSiQue analog was full-example disjointness). Assert disjointness on `section_id`, `query_id`, and normalized question text at run time, and hard-exit on any overlap, following `musique_seed.py`.

---

## 8. Canonical data representation

```json
{
  "example_id": "...",
  "passage_id": "...",
  "original_question": "...",
  "final_answer_hidden": {"number": "22"},
  "qdmr": [
    {"node_id": 1, "op": "SELECT",     "text": "yards of the Raiders' field goal", "args": ["yards of the Raiders' field goal"], "parents": [], "owner": "model",    "fn": null,         "value_type": "list"},
    {"node_id": 2, "op": "SELECT",     "text": "yards of the Broncos' field goal", "args": ["yards of the Broncos' field goal"], "parents": [], "owner": "model",    "fn": null,         "value_type": "list"},
    {"node_id": 3, "op": "ARITHMETIC", "text": "difference of #1 and #2",          "args": ["difference", "#1", "#2"],           "parents": [1, 2], "owner": "executor", "fn": "difference", "value_type": "number"}
  ],
  "sink_id": 3,
  "sink_type": "number",
  "step_count": 3,
  "model_owned_count": 2,
  "op_family": "arithmetic"
}
```

`args` and `fn` are parsed from BREAK's `program` column, which carries the operator arguments explicitly (`ARITHMETIC['difference', '#4', '#5']`, `AGGREGATE['count', '#2']`, `PROJECT['yards of #REF', '#3']`). The `operators` column carries only the operator name and is used as a cross-check.

`model_owned_count` is the frontier key (§3.2). The loader must map BREAK's operator strings to the executor's function table and mark each node's owner. Any node whose op has no executor implementation and is not an extraction op sends the example to the excluded pool; report the exclusion rate by operator.

Store gold final answers in a separate audit file that the generation process cannot import. There are no gold intermediates to store.

---

## 9. Model interfaces

### 9.1 Extraction-step mode

```text
[MODE: EXTRACT]
Passage: ...
Question: What is the yards of the Raiders' field goal?
Return only the values, separated by " | ".
```

Extraction-step verbalization is a deterministic template over the QDMR step text, keyed on operator. BREAK step text carries whitespace noise (`return yards  of   #2`), so the template normalizes whitespace first, strips the leading `return`, substitutes `#k` slots, and then applies:

| node op | template |
|---|---|
| SELECT | `Find in the passage: {X}` |
| PROJECT, FILTER | `What is {X}?` |

SELECT steps carry a bare noun phrase that can be plural (`people`, `field goals`) or a proper noun (`Adam Vinatieri`), and no single "What is/are X?" wording reads correctly for both, so SELECT gets a form that needs no subject-verb agreement. PROJECT and FILTER read as attribute lookups over something already named, so the question form fits them. This template set is pinned; changing it invalidates every pseudo-label generated under the old one.

**Output format.** Every model-owned node returns a list, rendered as values joined by `" | "`. A single value is the length-one case and is written without a separator. The format is uniform across nodes so that a downstream AGGREGATE node always receives a list, and so that the seed can teach the format from DROP's multi-span answers (§7.1). Pin the delimiter before the screen; changing it invalidates every generated pseudo-label.

### 9.2 Visible sub-DAG mode

```text
[MODE: QDMR]
Passage: ...
Step 1: yards of the Raiders' field goal
Step 2: yards of the Broncos' field goal
Step 3: difference of #1 and #2
Return the value of Step 3.
```

### 9.3 End-to-end mode

```text
[MODE: END2END]
Passage: ...
Question: How many yards longer was the Raiders' field goal than the Broncos'?
Return only the value.
```

Use explicit mode tokens so one checkpoint supports all three.

---

## 10. Composition algorithm

1. Topologically order the QDMR DAG.
2. For each model-owned node, verbalize with parent values substituted into `#k` slots and query the model.
3. For each executor-owned node, parse parent values into typed operands and apply the exact function. A parse failure (non-numeric operand for an arithmetic op) rejects the example at this node.
4. The sink value is the pseudo-label.

Roundwise sub-DAG composition follows MuSiQue §9.3: at frontier `k_t`, the current model may directly solve any connected model-owned sub-DAG of at most `k_t` extraction nodes; executor-owned nodes are free and do not count against the frontier. This rule is what makes `model_owned_count` the frontier key of §3.2 rather than QDMR step count.

Store the full trace (node values, instantiated step texts, executor inputs/outputs) for every example, accepted or rejected.

---

## 11. Guarded aggregation

DROP's numeric sinks admit stronger guards than MuSiQue's spans.

### 11.1 Local extraction checks

Every model-owned node returns a list (§9.1). Accept an extraction-step answer only when:

- the parsed list is non-empty, and every element is non-empty, below a short token limit, and free of explanation text;
- the element type matches the step's expected operand type (number for arithmetic and sum/max/min/avg parents, span otherwise);
- a number normalizes cleanly under the official DROP number normalizer;
- every element occurs in the passage after normalization;
- the list contains no duplicate elements after normalization, which is the failure mode that inflates COUNT.

### 11.2 List-completeness guard

Recall failures on an extracted list are silent: a COUNT over a list missing one member returns a well-formed but incorrect number, and no local check detects it. This guard is part of the minimum viable experiment rather than a later addition, because the count and sum family is included in scope (§6).

- two-view agreement on the **extracted list itself** — set equality after normalization, not agreement on its length and not agreement on the sink alone;
- for AGGREGATE parents, list length is range-checked against a per-operator bound.

### 11.3 Executor checks

- all operands parse into the declared types;
- ARITHMETIC operands are length-one lists; a multi-element list at an arithmetic node is a rejection, not a silent choice of element;
- a `count` node whose parent list holds a single numeric element is rejected as ambiguous (Risk 6);
- results are range-checked per op: counts are non-negative integers; year differences fall in a plausible range; percentages fall in [0, 100] when the question implies one.

Rejection causes are a fixed vocabulary, since their distribution is a reported metric (§16): `operand_parse`, `operand_not_scalar`, `ambiguous_count`, `empty_operand`, `empty_answer`, `unfilled_reference`, `arity`, `division_by_zero`, `range`, `missing_parent`.

### 11.4 Sink checks

- sink type matches the question's expected answer type inferred from surface cues ("how many" → number, "which" → span);
- the sink value is finite and within the range check for its op.

### 11.5 Agreement guard

Two harmless deterministic views, per MuSiQue §10.4: plain prompt versus a second fixed template, or current versus EMA checkpoint. No large self-consistency sampling.

### 11.6 Guard levels

| Level | Content |
|---|---|
| L1 | no guard; the chain produced a sink value |
| L2 | type and parse checks (§11.1) at every node |
| L3 | L2 plus range checks (§11.3, §11.4) |
| L4 | L3 plus two-view agreement on extracted lists and on the sink (§11.2, §11.5) |

Gold final answers may be used to measure guard precision and recall after generation, never to accept individual examples. Report precision against acceptance rate at every level, separately per op family (§16.3).

---

## 12. Curriculum and pseudo-label pools

Availability below is the realized partition, not an estimate: `drop_partition.py` splits the 2,500 passages carrying usable numeric DAGs into 2,125 composition-source (3,309 DAGs) and 375 internal-audit (555 DAGs), stratified by domain.

| Stage | Content | Available | Target accepted |
|---|---|---:|---:|
| Seed | DROP span-answer questions, gold labels, seed-source passages (§7.1) | 5,900 | 4,000–5,900 |
| Round 1 | k=2 numeric-sink DAGs | 1,063 | 800–1,063 |
| Round 2 | k=3 numeric-sink DAGs, round-1 model solves k=2 sub-DAGs | 906 | 600–906 |
| Round 3 | k=4 numeric-sink DAGs, additional op families | 932 | 600–932 |

These pools are roughly a fifth of MuSiQue's Round 1 set (4,792 accepted). The round-1 effect size is therefore estimated with correspondingly wider confidence intervals, and the go/no-go thresholds of §20 are set against that. If the guard acceptance rate at L4 falls below about 0.76, the k=2 pool drops under the 800-example floor and the guard level for training acceptance must be lowered to L3, with the precision cost reported.

Training mixture per round follows MuSiQue §12.3: 30–40% seed replay, 20–30% prior-round DAG tasks, 20–30% current-round DAG tasks, 20–30% current-round end-to-end tasks. Seed replay is not optional here: MuSiQue Round 1 showed part accuracy eroding under multi-hop training (.885 → .825 at 2 hops) without it.

---

## 13. Training targets

1. **Answer-only** (primary): the sink value string.
2. **Compact trace** (ablation): `#1 = 45; #2 = 23; FINAL = 22`. The symbolic steps make this trace partially exact, which may make trace training more useful here than on MuSiQue; treat as a flagged ablation.

---

## 14. Model and compute configuration

- Qwen3.5-4B, LoRA r16/a32 (the configuration validated in MuSiQue plan §26); Qwen3-1.7B as the smaller replication;
- gradient checkpointing enabled, required to fit training on the 48 GB L40 cards;
- context limit 2,048 tokens. Measured on the 214 dev passages carrying usable numeric DAGs: median 181 words, 99th percentile 423, maximum 918, so a single passage plus a short question fits with margin. `train_lora` rejects over-length examples rather than truncating, so the pool is still filtered on true tokenized length;
- answer limit 16 tokens;
- greedy decoding, no chain-of-thought;
- one or two epochs per round.

---

## 15. Baselines

- **B0** — seed model, direct multi-step inference. Per MuSiQue §26: the seeded direct model is the mandatory baseline; the base model understates it.
- **B1** — direct pseudo-labeling of full questions, matched sink-type guard.
- **B2** — frozen seed executing every QDMR node at all frontier cells, no retraining.
- **B3** — frozen frontier execution without training a direct student.
- **B4** — unfiltered QDMR composition.
- **B5** — guarded QDMR composition with retraining (proposed method).
- **B6** — oracle-extraction execution: gold-consistent extraction values obtained by hand-audit sample only (no released gold intermediates exist; this baseline is a small-sample diagnostic, unlike MuSiQue's B6).
- **B7** — gold-supervised upper bound on final answers, matched counts and curriculum.
- **B8** — direct model with gold QDMR visible at inference, no pseudo-labeling.

---

## 16. Evaluation and required analyses

### Primary metrics

- official DROP EM and F1 (number/date normalization, multi-span handling);
- accuracy by model-owned node count k and by op family;
- largest k above a pre-registered F1 threshold.

### Pseudo-label metrics

- sink exact match against hidden gold;
- guard acceptance rate and precision among accepted;
- rejection-cause distribution (parse failure, type, range, list disagreement, sink disagreement).

### 16.1 Frontier heatmap

Rows: k = 2, 3, 4 model-owned nodes; columns: seed and CSI rounds; separate panels for QDMR-visible and end-to-end inference.

### 16.2 Error localization without gold intermediates

BREAK provides no gold intermediate values. Substitute: (a) a 100-trace hand audit per round, labeling the first incorrect node; (b) the corruption test from §4 rerun per round; (c) exact-executor nodes as known-correct anchors — any error in a sink whose parents are executor-owned localizes to extraction. Report the hand-audit sample size next to every localization claim.

### 16.3 Guard tradeoff

Precision versus acceptance rate across guard levels L1–L4, separately per op family. The L3-to-L4 step is the one that carries the list-completeness guard (§11.2); report its precision gain and acceptance cost separately, since §12 makes the choice between L3 and L4 a pool-size decision.

### 16.4 Retraining versus execution

Direct CSI versus B2/B3 at matched generation budgets, as in MuSiQue §16.4.

### 16.5 Symbolic-tail analysis

Bin examples by the number of executor-owned nodes on the sink path. The framework predicts pseudo-label precision increases with this count; this analysis is unique to DROP and is a required figure.

Measured bin sizes in train (§3.2.1): depth 1 holds 3,081 rows, depth 2 holds 775, depth 3 holds 8. The comparison is therefore depth 1 against depth 2, with depth 3 folded into depth 2 or reported as a count only. State the bin sizes in the figure caption; a two-point comparison does not support a claimed trend beyond its endpoints.

---

## 17. Ablations

1. direct versus composed pseudo-labels;
2. frozen execution versus retraining;
3. atomic extraction versus frontier sub-DAG components;
4. guard levels L2/L3/L4;
5. answer-only versus compact-trace training;
6. ARITHMETIC-sink training versus mixed op families;
7. executor-owned symbolic steps versus model-owned symbolic steps (does letting the model do arithmetic degrade labels as expected?);
8. seed-source passage fraction.

Ablation 7 is the DROP-specific one: it isolates how much of the method's advantage comes from the exact executor.

---

## 18. Statistical protocol

- Thresholds tuned only on internal audit partitions.
- DROP dev used for model selection only after the pipeline is fixed; no DROP dev passage enters any training pool.
- Log per-instance outcomes in every arm (required for McNemar; see MuSiQue §25 for the cost of not doing this).
- Three final seeds; paired bootstrap CIs over questions, clustered by passage.

---

## 19. Risks and mitigations

### Risk 1 — QDMR noise contaminates the pipeline

A fraction of BREAK decompositions misrepresent the question. **Mitigation:** measure the executable-and-type-consistent fraction during preprocessing; exclude non-executable DAGs and report the rate. If the rate exceeds ~20% on number-typed sinks, re-scope to the cleanest op families. Cause of individual annotation errors: not investigated here; treat as exclusion, not correction.

**Measured:** 8.6% of numeric-sink train candidates and 8.4% of dev are blocked by a mid-DAG excluded operator (§3.2.1), below the 20% threshold. This measures structural executability only; it does not measure whether an executable decomposition answers the question it is attached to, which Risk 6 covers for the `count` case and the hand audit of §16.2 covers in general.

### Risk 2 — Structure-provided setting is too assisted

QDMR at inference is a large oracle. **Mitigation:** identical to MuSiQue Risk 2 — name the setting `structure-provided`, evaluate end-to-end without QDMR, and do not claim automatic decomposition.

### Risk 3 — Extraction atoms are near ceiling for the 4B model

If atom accuracy is already ≥ .95 at base, seeding adds little and the screen may pass while round-1 training has no room. **Mitigation:** the screen (§4) reports part accuracy; if at ceiling, shift the frontier definition toward op families rather than extraction difficulty, and consider the 1.7B model as primary.

### Risk 4 — List recall failures are silent

COUNT and SUM require extracting all members of a set, and a list missing one member produces a well-formed but incorrect number. **Mitigation:** the count and sum family is inside the MVE (§6), so the list-completeness guard of §11.2 is implemented from the start: two-view set equality on the extracted list, plus a per-operator length bound. The L3-to-L4 precision gain is the measurement that tells whether this guard works (§16.3).

### Risk 5 — Number normalization disagreements

"22" versus "twenty-two" versus "22 yards". **Mitigation:** use the official DROP normalizer everywhere, including inside guards; never write a second normalizer.

### Risk 6 — `AGGREGATE['count', X]` is overloaded

BREAK uses `count` for two different intents. In "How many field goals did Hartley make?" the parent list holds the field goals and the answer is its length. In "How many men, horses, elephants combined in Nuri Khan army?" the parent list holds one span such as `50,000` and the intended answer is that number, not 1. Executing `len()` on the second case returns 1 and produces a confidently wrong label.

**Mitigation:** the executor flags a `count` node whose parent list has length 1 and whose single element parses as a number. Three policies are implemented and selectable (`--count-policy`), because which reading is right is a Week 1 guard question and the rate is large enough to decide pool sizes:

| policy | behavior |
|---|---|
| `reject` | refuse the node and count it (default; conservative) |
| `length` | always `len(list)`, the literal COUNT semantics |
| `value` | read the single numeric element as the answer |

Measure precision under all three against hidden gold on the audit partition before fixing the policy. A 15-instance smoke run at base made `ambiguous_count` the dominant rejection cause, 7 of 15 chains; if that rate holds at full size it takes the k=2 pool from 1,063 to roughly 560, below the 800 floor of §12. Cause of the annotation inconsistency: not investigated here; treat as exclusion, not correction.

### Risk 7 — Pools are small

The usable numeric-sink pools are roughly 1,185 / 1,018 / 948 at k = 2 / 3 / 4 after partitioning (§12), against MuSiQue's 4,792. **Mitigation:** confidence intervals are reported on every contrast; the go/no-go thresholds of §20 are stated against these pool sizes; if the L4 acceptance rate puts the k=2 pool below 800, the training acceptance level drops to L3 with the precision cost reported. Widening scope to non-numeric sinks (§5.4) is the reserve, not the first response.

---

## 20. Go/no-go criteria

Proceed past round 1 only if:

- screen verdict rule met (§4);
- seed extraction EM ≥ .80 on held-out seed-source passages;
- guarded k=2 pseudo-label EM exceeds direct pseudo-label EM by ≥ 5 points;
- accepted k=2 pseudo-label precision ≥ .85 EM;
- ≥ 800 k=2 examples pass the guard at the acceptance level chosen in §12;
- round-1 direct k=2 EM improves ≥ 3 points over the seeded direct baseline (B0, per MuSiQue §26);
- extraction-atom regression after replay ≤ 2 points.

---

## 21. Expected paper claims

> When a discrete-reasoning benchmark exposes its operation DAG, small-model extraction answers composed through an exact symbolic executor yield cleaner supervision than direct frontier prediction, and the gain grows with the executor-owned fraction of the DAG.

The experiment does not establish automatic decomposition, and the symbolic-executor advantage does not transfer to benchmarks without executable ops.

---

## 22. Implementation schedule

### Week 0 — Isolation screen and seed rescreen (gate)

- join DROP and BREAK, pin versions, reproduce the §3.1 counts;
- implement verbalization templates and the symbolic executor;
- run the four screen arms at base with per-instance logging;
- passage-level partitions (§7.2);
- train the seed on DROP span questions and rerun the screen;
- apply the verdict rule to the after-seed statistics before any further work.

The seed rescreen sits inside the gate rather than after it, because on MuSiQue the base screen was uninformative and the seed decided the benchmark (§4).

### Week 1 — Guards and label audit

- implement guard levels L1–L4 including the list-completeness guard;
- audit k=2 pseudo-label precision against hidden gold;
- choose the training acceptance level against the §12 pool floor.

### Week 2 — Round 1 (k=2)

- run B1/B2/B4/B5 labelers; train round-1 models with seed replay; internal evaluation.

### Week 3 — Rounds 2–3 and analyses

- k=3 and k=4 rounds with sub-DAG cuts;
- symbolic-tail analysis, guard tradeoff, hand-audit error localization.

### Week 4 — Transfer and final runs

- op-family transfer; three-seed final runs; figures.

---

## 23. Reproducibility checklist

- [ ] DROP and BREAK versions and hashes pinned; join rate recorded.
- [ ] §3.1 counts reproduced at build time (7,672 / 1,265 rows, 3,624 / 247 passages, 4,228 / 654 numeric sinks).
- [ ] Passage-level partition disjointness asserted at run time on `section_id`, `query_id`, and normalized question text.
- [ ] Gold final answers stored outside generation code.
- [ ] Executor function table versioned; excluded-op rate reported by operator.
- [ ] List delimiter and verbalization templates pinned before the screen.
- [ ] Per-instance outcomes logged in every arm.
- [ ] Guard thresholds fixed before dev evaluation.
- [ ] Matched budgets across B1/B4/B5.
- [ ] End-to-end evaluation hides QDMR.
- [ ] Results sliced by model-owned node count k and op family.
- [ ] 99th-percentile prompt length measured against the context limit before training.

---

## 24. Primary sources

- Dua, D., et al. (2019). [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246/).
- Wolfson, T., et al. (2020). [Break It Down: A Question Understanding Benchmark](https://aclanthology.org/2020.tacl-1.13/) — QDMR annotations.
- Official repositories: [allenai/drop](https://allenai.org/data/drop), [allenai/Break](https://github.com/allenai/Break).

QDMR is read from `break_dataset/logical-forms/{train,dev}.csv`, which carries the `program` column with explicit operator arguments. `break_dataset/QDMR/{train,dev}.csv` holds the same decompositions without the parsed program and is not sufficient for building the executor's node table.

---

## 25. Gate outcome — discontinued 2026-08-06

The §4 screen was run at base (job 3648794) and after seed training (job 3648795), 300 paired dev instances per state, per-instance outcomes logged. **The pre-registered verdict rule of §4 was not met and this benchmark is discontinued**, at the same gate that discontinued BFCL and CLUTRR.

| | base | after seed |
|---|---:|---:|
| direct | .610 | .540 |
| composed | .233 | .250 |
| headroom | −.377 | −.290 |
| 95% CI | [−.443, −.310] | [−.363, −.217] |
| McNemar p | <1e-4 | <1e-4 |
| atom proxy | .473 | .683 |
| corruption drop | +.150 | +.140 |
| chains completing | 177/300 | 181/300 |

Seed: 4,000 DROP span-answer examples from BREAK-uncovered passages, 0 dropped over the 2,048-token limit, 300 steps, LoRA r16/a32, train loss .364.

### 25.1 Findings

**The seed regressed direct accuracy.** Paired before/after on direct: −.070, McNemar p = .028, CI [−.127, −.013]. Most of the +.087 headroom gain is the baseline falling rather than composition rising; composed moved +.017.

**The available atomic supervision has the wrong output type.** §7.1 established that no gold intermediate values exist, so the seed must come from DROP span-answer questions while the DAGs in scope have numeric sinks. The consequence is measurable in both directions: atom span accuracy .492 → .732 but multi-span only .370 → .413, and rejection causes shifted rather than shrank — `ambiguous_count` 84 → 55 while `operand_parse` went 32 → 60, the seeded model emitting spans where the executor needs numbers. This is a property of what BREAK releases, not of the seed pool construction.

**The composite is not much harder than its parts.** Base direct accuracy is .610 on questions whose decomposition has 2–4 extraction nodes, against MuSiQue's .563 on 2–4 hop questions. DROP is single-passage, so a numeric answer can often be read off the text without decomposing. The base atom proxy sat *below* direct (−.137), inverting the premise of the screening criterion; after seeding it rose above (+.143) and composition still lost by .290.

**Parity was never reachable.** With executor nodes exact, composed accuracy is approximately the product of atom accuracy over model-owned nodes, so parity required atom accuracy above .80 / .82 / .89 at k = 2 / 3 / 4 against the base direct baselines. MuSiQue's seed reached part accuracy .815 / .770 / .703 on single-span atoms; DROP's atom is list-valued and scored on exact set match, and its seed reached .683 on the easier proxy.

**§16.5's prediction did not appear.** Pseudo-label accuracy did not rise with executor-owned depth: depth 1 composed .438 (n=144), depth 2 .212 (n=33) at base. Depth-2 rows are harder overall (direct .515 against .667), so this is not a clean refutation, but the predicted direction is absent.

### 25.2 What held up

Composition is load-bearing: corrupting one upstream value costs .150 at base and .140 after seeding. The executor is exact and its rejections are typed. The seed lifted its own target metric by .210. The machinery is sound; the benchmark does not have the property the framework needs.

### 25.3 Reopening conditions

Recorded in `DROP_HANDOFF.md` §8. In short: a source of gold intermediate values, a model reaching roughly .85 list-valued extraction, or a retreat to the `difference` family alone — which holds ~600 examples after partitioning, below the §12 floor.
