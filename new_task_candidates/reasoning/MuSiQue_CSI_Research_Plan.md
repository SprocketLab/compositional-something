# Research Plan: MuSiQue-CSI

## Low-compute compositional self-improvement over visible multihop question DAGs

**Primary dataset:** MuSiQue-Ans v1.0.

**Main setting:** answerable examples, gold supporting paragraphs only, and the released question-decomposition DAG visible to the pseudo-label generator.

**Core curriculum:** one-hop component QA → two-hop DAGs → three-hop DAGs → four-hop DAGs.

**Primary training target:** direct answer prediction for the original multihop question; a structured trace target is an optional secondary condition.

**Compute profile:** low to moderate. Each pseudo-label requires two to four short QA calls rather than a long agent trajectory.

---

## 1. Executive summary

MuSiQue is the strongest natural-language candidate for testing compositional self-improvement because its multihop questions were built from connected single-hop questions and the release preserves the underlying decomposition.

A two-hop example can be represented as:

```text
Node 1: Who wrote Armageddon in Retrospect?
        -> Kurt Vonnegut

Node 2: What 1969 satire novel was #1 best known for?
        -> Slaughterhouse-Five
```

The first prediction is substituted into the second question. The final node answer becomes the pseudo-label for the original multihop question.

The proposed experiment has two linked model interfaces:

1. **DAG execution mode:** the model receives one or more decomposed questions with their designated support paragraphs and returns the answer to the sink node.
2. **End-to-end mode:** the model receives the original MuSiQue question and its support paragraphs and returns the final answer directly.

The seed model is trained on one-hop nodes extracted from a disjoint subset of MuSiQue training examples. In round 1, its one-hop predictions are composed to label two-hop questions. The model is then trained on accepted two-hop DAG-execution tasks and on the corresponding original questions. In round 2, a three-hop graph can be partitioned into a two-hop subgraph plus a one-hop remainder; the current model solves the former as an in-frontier component. Round 3 analogously composes four-hop graphs.

The main scientific comparison is between:

- direct full-question pseudo-labeling;
- frozen step-by-step execution with the seed model;
- guarded DAG composition plus retraining;
- gold-supervised multihop training.

The experiment deliberately starts with gold supporting paragraphs and linear chains. This keeps context length and retrieval out of the loop so that the result measures compositional reasoning rather than document selection. Branching graphs and distractors are phase-2 transfer tests.

---

## 2. Research question

> Given an explicit natural-language question DAG, can guarded composition of reliable sub-DAG answers create sufficiently clean multihop supervision to improve a small model's direct two-to-four-hop reasoning?

A positive result should show that:

- guarded DAG execution produces more accurate multihop pseudo-labels than direct prediction;
- retraining on two-hop pseudo-labels improves direct two-hop QA;
- the round-1 model can serve as a larger component solver for three-hop construction;
- iterative retraining improves over permanently executing the one-hop seed model;
- improvements transfer from the DAG-visible training interface to the original end-to-end question interface;
- gains persist when a small number of distractor paragraphs are added.

---

## 3. Why MuSiQue fits the framework

MuSiQue-Ans contains roughly 25,000 answerable questions requiring two to four connected hops. The official release includes train, development, and test splits; supporting-paragraph annotations; final answers and aliases; and the constituent question decomposition used to construct each multihop item.

The framework mapping is:

| CSI object | MuSiQue instantiation |
|---|---|
| atomic input | one constituent question plus its support paragraph |
| atomic output | a short answer or entity span |
| composed input | a connected question DAG |
| composition | substitute parent answers into dependent questions |
| parent label | answer at the sink node |
| guard | span, type, bridge, support, and consistency checks |
| frontier | number of DAG nodes, graph shape, and context noise |

Unlike SCAN, the composition rule is not an algebra over two output labels. A predicted output changes a downstream input:

\[
\hat a_v = M\left(q_v[\hat a_{pa(v)}], p_v\right),
\]

where `p_v` is the supporting paragraph for node `v`. The composed label is the sink answer.

---

## 4. Experimental variants

### 4.1 Primary: MuSiQue-CSI-DAG-Support

Input to the model includes:

- the visible decomposition nodes;
- dependency references such as `#1`;
- only the gold supporting paragraph for each node;
- an instruction to answer the sink node.

This is the cleanest test of compositional self-improvement.

### 4.2 Transfer: MuSiQue-CSI-End2End-Support

Input includes:

- the original multihop question;
- the union of gold supporting paragraphs;
- no decomposition at inference.

Pseudo-labels are still generated through the hidden DAG pipeline, but the student learns to answer the original question directly.

### 4.3 Phase 2: support plus distractors

Add one, three, or five non-supporting paragraphs sampled from the same example. This tests whether learned reasoning survives modest context noise without requiring a retrieval model.

### 4.4 Optional: MuSiQue-Full

MuSiQue-Full adds paired unanswerable questions. This should be attempted only after answerable CSI works because answer sufficiency is not naturally produced by the same composition rule.

---

## 5. Scope

### Included in the minimum viable experiment

- MuSiQue-Ans train and development data;
- two-hop linear chains first;
- three-hop linear chains second;
- four-hop linear and branching graphs only after the pipeline is stable;
- support-only contexts;
- short answer generation;
- one small model with parameter-efficient or full small-model fine-tuning;
- two or three CSI rounds.

### Excluded initially

- open-domain retrieval;
- 20-paragraph full-context training;
- answerability classification;
- free-form rationales;
- large sampling budgets;
- external search tools;
- training on the source QA datasets.

---

## 6. Data acquisition, versioning, and leakage control

1. Download MuSiQue v1.0 using the official repository script.
2. Pin the repository commit and archive hashes.
3. Use the official MuSiQue-Ans train/dev/test split.
4. Preserve each full multihop example as the unit of splitting.
5. Record every constituent source-question identifier available in the release.
6. Deduplicate on normalized constituent question, support title, and answer.
7. Keep official dev and test examples untouched.

The official repository warns that MuSiQue was composed from SQuAD, T-REx, Natural Questions, MLQA, and Zero-Shot RE, and releases the source IDs used in MuSiQue dev/test. Do not add training data from those source datasets unless those IDs are explicitly removed.

### Critical anti-memorization rule

Do not extract labeled one-hop seed examples from the same full MuSiQue examples later used for pseudo-label construction. Otherwise the model can memorize the exact component answers.

Partition the MuSiQue training split at the **full-example level** before extracting any nodes:

| Partition | Suggested fraction | Use |
|---|---:|---|
| Seed-source | 30% | extract labeled one-hop components |
| Composition-source | 60% | hide all answers; construct 2–4-hop pseudo-labels |
| Internal audit/test | 10% | hidden evaluation and threshold tuning |

Stratify by hop count, graph shape, source dataset, answer type, and support-paragraph length. If constituent source IDs repeat across full examples, group them into one partition.

---

## 7. Canonical data representation

Normalize every example to an explicit graph:

```json
{
  "example_id": "...",
  "original_question": "...",
  "final_answer_hidden": "...",
  "answer_aliases_hidden": ["..."],
  "nodes": [
    {
      "node_id": 1,
      "question_template": "Who wrote Armageddon in Retrospect?",
      "parents": [],
      "support_title": "...",
      "support_text": "...",
      "gold_answer_hidden": "Kurt Vonnegut"
    },
    {
      "node_id": 2,
      "question_template": "What 1969 satire novel was #1 best known for?",
      "parents": [1],
      "support_title": "...",
      "support_text": "...",
      "gold_answer_hidden": "Slaughterhouse-Five"
    }
  ],
  "sink_id": 2,
  "graph_shape": "2hop_chain"
}
```

The loader should assert the exact field names of the downloaded v1.0 release rather than relying on this illustrative schema.

Store gold node answers and final answers in a separate audit file that the generation process cannot import.

---

## 8. Model interfaces

### 8.1 One-hop component mode

```text
[MODE: ONE_HOP]
Question: Who wrote Armageddon in Retrospect?
Passage: ...
Return only the short answer.
```

### 8.2 Visible sub-DAG mode

```text
[MODE: DAG]
Node 1 question: Who wrote Armageddon in Retrospect?
Node 1 passage: ...

Node 2 question: What 1969 satire novel was #1 best known for?
Node 2 passage: ...

Return the answer to Node 2.
```

The model may internally execute the two nodes or answer directly. The output remains one short answer.

### 8.3 End-to-end mode

```text
[MODE: END2END]
Question: Armageddon in Retrospect was written by the author who was best known for what 1969 satire novel?
Passages:
[1] ...
[2] ...
Return only the short answer.
```

Use explicit mode tokens so one checkpoint can support component, sub-DAG, and end-to-end training.

---

## 9. Composition algorithm

### 9.1 One-hop execution

For a source node with no parents:

\[
\hat a_v = M(q_v, p_v).
\]

### 9.2 Dependent-node execution

Replace every parent reference with the normalized parent prediction:

```text
What 1969 satire novel was #1 best known for?
```

becomes:

```text
What 1969 satire novel was Kurt Vonnegut best known for?
```

Then predict:

\[
\hat a_v = M(q_v[\hat a_{pa(v)}], p_v).
\]

### 9.3 Roundwise sub-DAG composition

At frontier `h_t`, let the current model directly solve any connected sub-DAG containing at most `h_t` nodes.

For a three-hop chain in round 2:

1. solve the two-hop prefix as one component;
2. substitute its sink answer into the final one-hop node;
3. solve the final node;
4. use the result as the full three-hop pseudo-label.

For a four-hop chain in round 3:

- prefer a 2+2 or 3+1 cut whose subgraphs are within the current frontier;
- solve upstream subgraphs first;
- instantiate all downstream references;
- solve the sink subgraph.

For branching graphs, independently solve each parent branch and substitute both answers into the merge node.

### 9.4 Composition trace

Store, but do not necessarily train on, a full trace:

```json
{
  "node_predictions": {
    "1": "Kurt Vonnegut",
    "2": "Slaughterhouse-Five"
  },
  "instantiated_questions": {
    "2": "What 1969 satire novel was Kurt Vonnegut best known for?"
  },
  "final_prediction": "Slaughterhouse-Five"
}
```

This trace is essential for auditing where pseudo-label errors arise.

---

## 10. Guarded aggregation

MuSiQue does not provide an exact symbolic verifier. The guard must therefore be conservative and its precision must be measured against hidden answers.

### 10.1 Local answer checks

Accept a node answer only when:

- it is non-empty and below a short token limit;
- it is not only punctuation, a stopword, or a generic phrase such as “the person”;
- it is an extractive span of the designated support paragraph or its title after normalization;
- dates and numbers normalize consistently;
- output contains no explanation or multiple alternatives.

### 10.2 Question-type checks

Infer a coarse expected type from the question:

- `who` → person or organization;
- `when` → date/year;
- `where` → place;
- `how many` → number;
- `what year` → four-digit year or date expression.

Use lightweight rules first. Named-entity recognition is optional and should not become a large external model dependency.

### 10.3 Bridge checks

For a predicted bridge answer:

- the substituted downstream question must contain no unresolved `#k` references;
- the answer or one normalized alias should occur in the downstream support title or paragraph when the graph construction expects an entity bridge;
- substitution must not produce obvious malformed text;
- the answer should not collapse two distinct parent slots to the same generic phrase unless the data supports it.

### 10.4 Agreement guard

For the strongest low-cost guard, require agreement between two harmless executions:

- original versus reversed support-paragraph order;
- plain prompt versus a second fixed prompt template;
- current checkpoint versus an exponential-moving-average checkpoint.

Do not use large self-consistency sampling. Two deterministic views are enough for an ablation.

### 10.5 Whole-graph checks

Accept the parent example only when:

- every node needed for the sink has an accepted answer;
- the graph executes in topological order;
- all references are resolved exactly once;
- no node answer violates the local checks;
- the final answer is a valid span in the sink support paragraph or title;
- the composed trace is serializable and deterministic.

### 10.6 Guard levels

Compare:

1. no guard;
2. span-only guard;
3. span + type + bridge guard;
4. full guard with two-view agreement.

Gold answers may be used to measure each guard's precision and recall after generation, never to accept individual examples.

---

## 11. Curriculum and pseudo-label pools

### Seed — one-hop

Extract one-hop nodes only from the seed-source partition. Train the model to answer their designated support paragraphs.

### Round 1 — two-hop

Start with linear two-hop graphs from the composition-source partition.

- execute two one-hop nodes;
- guard each answer and the bridge;
- train on accepted visible-DAG inputs;
- also train on the original two-hop question with the same final pseudo-label.

### Round 2 — three-hop

Use the round-1 model to solve two-hop sub-DAGs directly. Prefer cuts that create one two-hop component plus one one-hop component.

### Round 3 — four-hop

Start with linear four-hop chains. Add branching graphs only after linear pseudo-label precision is adequate.

### Suggested accepted-example targets

| Stage | Target accepted examples | Notes |
|---|---:|---|
| Seed | 4,000–8,000 one-hop nodes | gold labels from seed-source only |
| Round 1 | 3,000 two-hop examples | linear chains first |
| Round 2 | 2,000–3,000 three-hop examples | sub-DAG cut required |
| Round 3 | 1,500–2,500 four-hop examples | linear before branching |

The exact counts should be adjusted to the available graph-shape distribution and guard acceptance rate.

---

## 12. Training targets

### 12.1 Answer-only target

```text
Slaughterhouse-Five
```

This is the cheapest and should be the primary condition.

### 12.2 Compact trace target

```text
#1 = Kurt Vonnegut
#2 = Slaughterhouse-Five
FINAL = Slaughterhouse-Five
```

This can improve observability but increases exposure to noisy intermediate labels. Treat it as an ablation.

### 12.3 Multi-task mixture

At each round, train on:

- 30–40% one-hop seed replay;
- 20–30% prior-round DAG tasks;
- 20–30% current-round DAG tasks;
- 20–30% current-round end-to-end tasks.

Normalize sampling so one full example does not contribute an excessive number of nearly duplicate node prompts.

---

## 13. Model and compute configuration

Recommended starting point:

- 0.1B–1.5B model;
- T5-small/base or a sub-billion instruction model;
- LoRA for models above roughly 500M parameters;
- context limit 1,024 tokens for support-only experiments;
- answer limit 16–24 tokens;
- greedy decoding;
- no chain-of-thought generation;
- one or two epochs per round;
- batched pseudo-label generation.

Support-only contexts keep the experiment practical. If a support paragraph is unusually long, truncate only after confirming that the hidden gold answer span remains in view; this check may use metadata during data preparation but not during prediction.

---

## 14. Baselines

### B0 — One-hop seed, direct multihop inference

Train only on seed one-hop nodes and ask the model to answer full questions directly.

### B1 — Direct pseudo-labeling

Predict each full two-, three-, or four-hop question in one generation. Apply a matched final-answer span/type guard.

### B2 — Frozen atomic execution

Use the seed one-hop model to execute every graph node sequentially at all hop counts. Do not retrain.

### B3 — Frozen frontier execution

After each round, use the latest checkpoint as a sub-DAG executor but do not train a direct end-to-end student on the new parents.

### B4 — Unfiltered DAG composition

Execute the graph and retain every final answer.

### B5 — Guarded DAG composition with retraining

The proposed method.

### B6 — Gold-intermediate execution

Use gold intermediate answers but model-predict the sink. This diagnoses bridge-error accumulation and is not a valid self-improvement method.

### B7 — Gold-supervised multihop upper bound

Train on true final answers with the same example counts and curriculum.

### B8 — Direct model with gold decomposition at inference

Measure the value of visible structure independently of pseudo-labeling.

---

## 15. Evaluation

### Primary metrics

- answer exact match;
- token-level answer F1 using the official evaluator;
- accuracy/F1 by hop count;
- largest hop count with a pre-registered F1 threshold.

### Pseudo-label metrics

- final-answer exact match and F1 against hidden gold;
- intermediate-answer exact match and F1;
- guard acceptance rate;
- precision among accepted examples;
- node at which the first error occurs;
- precision as a function of graph depth.

### Structure slices

- linear versus branching graph;
- one-parent versus multi-parent sink;
- person, location, date, numeric, and other bridge answers;
- bridge answer present in downstream title versus body only;
- same-source versus cross-source constituent questions;
- support paragraph length;
- alias-sensitive examples.

### Transfer settings

1. original question + gold support paragraphs;
2. visible DAG + gold support paragraphs;
3. support plus one, three, or five distractors;
4. optional full 20-paragraph context.

### Efficiency

- model calls and generated tokens per accepted label;
- direct final inference latency;
- frozen executor latency;
- training tokens per round.

---

## 16. Required analyses

### 16.1 Hop-frontier heatmap

Rows are one through four hops; columns are seed and CSI rounds. Produce separate panels for visible-DAG and end-to-end inference.

### 16.2 Error propagation

Estimate:

\[
P(\text{final correct} \mid k)
\]

and compare it with the product of measured component accuracies. This shows whether errors are independent, correlated, or partly corrected downstream.

### 16.3 Guard tradeoff

Plot accepted-label precision against acceptance rate for the guard levels. The key question is whether a conservative bridge guard keeps enough examples to train.

### 16.4 Retraining versus execution

Compare direct CSI to frozen atomic and frontier executors at matched total generation budgets. This is essential: MuSiQue already supports stepwise execution, so the paper must show what retraining adds.

### 16.5 Rejected-slice generalization

Evaluate the final direct model on examples rejected because of:

- alias mismatch;
- bridge not found downstream;
- answer-type uncertainty;
- execution disagreement.

This mirrors the current paper's filtered-slice analysis, though the guard is imperfect and results must be interpreted cautiously.

---

## 17. Ablations

Prioritize:

1. direct versus composed pseudo-labels;
2. frozen execution versus retraining;
3. one-hop-only components versus frontier-sized sub-DAG components;
4. span-only versus bridge-aware guard;
5. answer-only versus compact-trace training;
6. support-only versus support plus distractors;
7. linear-only training versus adding branching graphs;
8. seed-source fraction;
9. accepted examples per round;
10. one versus two prompt views for agreement.

The most important ablation is **atomic execution versus frontier sub-DAG execution**. It tests the paper's iterative claim rather than merely showing that single-hop QA can be chained.

---

## 18. Statistical protocol

- Tune thresholds only on internal train/audit partitions.
- Use official development data for model selection only after the pipeline is fixed.
- Keep official test labels sealed until final evaluation.
- Match accepted-example counts across B1, B4, and B5.
- Use one development seed and three final seeds.
- Report paired bootstrap confidence intervals over full questions.
- For pseudo-label precision, bootstrap by full graph rather than by node.

---

## 19. Risks and mitigations

### Risk 1 — Frozen one-hop execution is already best

**Mitigation:** make this a central baseline. If retraining only compresses the executor without improving accuracy, report latency and direct-inference gains but do not claim frontier expansion.

### Risk 2 — Gold decomposition makes the task too assisted

**Mitigation:** clearly name the primary setting `structure-provided`. Evaluate the trained end-to-end model without decomposition. Do not claim automatic decomposition.

### Risk 3 — Span guards reject valid aliases

**Mitigation:** use normalization, title matching, and the released alias lists for evaluation. Do not use hidden gold aliases to accept an individual pseudo-label unless aliases are visible metadata at deployment; keep that as an oracle guard ablation.

### Risk 4 — Component seed memorizes composition examples

**Mitigation:** enforce full-example and source-ID disjointness between seed-source and composition-source partitions.

### Risk 5 — Three- and four-hop graphs do not admit clean frontier cuts

**Mitigation:** compute articulation points and subgraph sizes before training. Begin with graph shapes that admit a 2+1 or 2+2 cut and report coverage.

### Risk 6 — Context truncation removes the answer

**Mitigation:** use support-only paragraphs, validate tokenization during preprocessing, and exclude examples exceeding the configured context rather than silently truncating evidence.

### Risk 7 — Guard precision is too low

**Mitigation:** start with extractive person/date/place chains and linear graphs. Add broader answer types only after the pilot.

---

## 20. Go/no-go criteria

Proceed to three- and four-hop rounds only if:

- one-hop seed answer F1 is at least 75 on the held-out seed-source test;
- guarded two-hop pseudo-label F1 exceeds direct two-hop pseudo-label F1 by at least five points;
- accepted two-hop pseudo-label precision is at least 80% exact match or 85 F1;
- at least 1,500 two-hop examples pass the guard;
- round-1 direct two-hop F1 improves by at least three points over the seed model;
- no more than a two-point one-hop regression occurs after replay.

Keep MuSiQue as an exploratory result rather than a headline result if the guard cannot maintain both reasonable precision and sample volume.

---

## 21. Expected paper claims

A successful experiment can support:

> When a human-written multihop QA benchmark exposes its constituent question DAG, reliable short-answer predictions can be composed into cleaner supervision for larger graphs than direct frontier prediction. Retraining on those labels improves a direct model and can partially amortize stepwise execution.

The experiment does not by itself establish automatic decomposition or retrieval.

---

## 22. Implementation schedule

### Week 1 — Data and graph pipeline

- download and pin v1.0;
- inspect and normalize fields;
- partition by full example and source IDs;
- build graph parser and topological executor;
- reproduce official answer evaluation.

### Week 2 — One-hop seed and guards

- construct seed-source node dataset;
- train one-hop model;
- implement span, type, bridge, and agreement checks;
- audit two-hop pseudo-label precision.

### Week 3 — Two-hop CSI

- run direct, frozen, unfiltered, and guarded labelers;
- train visible-DAG and end-to-end round-1 models;
- evaluate internal and official development sets.

### Week 4 — Three/four-hop CSI

- implement sub-DAG cuts;
- run three-hop and four-hop rounds;
- add graph-shape and error-propagation analyses.

### Optional Week 5 — Distractor transfer

- add one/three/five distractors;
- finalize three-seed runs and paper figures.

---

## 23. Reproducibility checklist

- [ ] Official v1.0 hashes recorded.
- [ ] Seed-source and composition-source full examples are disjoint.
- [ ] Constituent source IDs grouped to prevent leakage.
- [ ] Gold node/final answers stored outside generation code.
- [ ] Every pseudo-label stores the complete execution trace.
- [ ] Guard thresholds fixed before official dev/test evaluation.
- [ ] Direct and composed baselines receive matched verifier budgets.
- [ ] Final end-to-end evaluation hides the decomposition.
- [ ] Results separated by hop count and graph shape.
- [ ] Source-dataset leakage warning documented in the paper.

---

## 24. Primary sources

- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2022). [MuSiQue: Multihop Questions via Single-hop Question Composition](https://aclanthology.org/2022.tacl-1.31/).
- Official repository and v1.0 download/evaluation scripts: [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique).
