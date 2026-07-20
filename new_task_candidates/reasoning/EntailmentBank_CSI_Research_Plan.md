# Research Plan: EntailmentBank-CSI

## Structure-provided compositional self-improvement for natural-language entailment trees

**Primary dataset:** EntailmentBank, Task 1 as the main experiment and Task 2 as distractor transfer.

**Core curriculum:** one-step textual entailment → two-step trees → three-step trees → four-or-more-step trees.

**Primary setting:** gold tree topology is available to the pseudo-label constructor, but intermediate conclusion text and full gold proofs are hidden.

**Main target:** direct generation of an entailment tree in the official proof format.

**Compute profile:** moderate but still single-GPU friendly. The dataset is small; the principal challenge is semantic guarding, not training scale.

**Risk level:** high. This should be treated as an exploratory natural-language result until pseudo-label precision is demonstrated.

---

## 1. Executive summary

EntailmentBank contains expert-authored explanation trees for grade-school science questions. Each tree combines known facts through several natural-language entailment steps until it reaches a hypothesis formed from the question and correct answer.

A simplified tree is:

```text
sent1: Astronauts are humans.
sent2: Humans are animals.
             |
             v
int1: Astronauts are animals.

int1 + sent3: Animals require oxygen to breathe.
             |
             v
int2: Astronauts require oxygen to breathe.

int2 + other facts
             |
             v
hypothesis
```

The proposed CSI system receives the **grouping topology** of the tree during pseudo-label construction: it knows which child nodes should be combined at each internal node, but it does not see the gold intermediate conclusion text. A local generator predicts an entailed intermediate conclusion from each child group. A separately trained one-step entailment verifier checks the proposal. Accepted conclusions are inserted into the next level, and the process continues bottom-up. If every step passes and the root premises entail the provided hypothesis, the assembled tree becomes a pseudo-label for the original full explanation task.

The seed supervision comes from one-step entailment nodes extracted only from a disjoint subset of training trees. Deeper training trees have their intermediate text and proof strings hidden. Round 1 constructs two-step trees, round 2 constructs three-step trees, and round 3 targets larger trees.

This setting is scientifically interesting because local conclusions can be individually valid yet fail to provide the abstraction needed by the parent step. That is a natural-language analogue of a structured composition error. A whole-tree guard can reject such granularity or interface mismatches.

The main limitation must be explicit: the experiment uses expert-provided tree topology. It therefore tests **self-improvement given a decomposition**, not automatic proof planning and not annotation-free reasoning from raw questions alone. The final direct model is evaluated without the topology, which tests whether the composed supervision is reusable.

---

## 2. Research question

> Given the topology of a natural-language entailment tree, can a small model compose verified one-step conclusions into sufficiently reliable deeper pseudo-proofs to improve direct explanation generation?

A positive result should establish that:

- local composition produces cleaner full-tree pseudo-labels than one-shot proof generation;
- a semantic guard materially improves pseudo-label precision;
- iterative retraining improves direct proof generation as tree depth grows;
- the final direct model improves on Task 1 without receiving topology at inference;
- some improvement transfers to Task 2 with distractor facts;
- the model can generalize to some guard-rejected composition slices.

---

## 3. Dataset facts

EntailmentBank contains 1,840 expert-authored entailment trees. The official paper reports:

| Split | Questions | Entailment reasoning steps |
|---|---:|---:|
| Train | 1,313 | 4,175 |
| Development | 187 | 597 |
| Test | 340 | 1,109 |
| Total | 1,840 | 5,881 |

The average tree contains about 7.6 nodes across 3.2 entailment steps. Each step typically combines multiple premises into a natural-language conclusion.

The benchmark defines three increasingly difficult tasks:

1. **Task 1:** all and only the relevant leaf facts are given;
2. **Task 2:** relevant leaves plus distractor facts are given;
3. **Task 3:** facts must be retrieved from a corpus.

Use Task 1 for the main CSI experiment. Task 2 is a transfer test. Task 3 is out of scope for the low-compute study.

---

## 4. What is composed

Let an internal tree node `v` have child texts `z_1, ..., z_m`. The local generator predicts:

\[
\hat z_v = G(z_1,\ldots,z_m).
\]

A guard `V` checks whether the child set entails the candidate conclusion:

\[
V(z_1,\ldots,z_m,\hat z_v) \in \{0,1\}.
\]

If accepted, `\hat z_v` is inserted as a premise for its parent. The final root conclusion is the provided hypothesis; the system must verify that the model-generated root premises entail it.

The full pseudo-label is a tree:

```text
sent2 & sent3 -> int1: <generated conclusion>;
int1 & sent1 -> hypothesis;
```

The tree topology determines which IDs connect. The model supplies the natural-language intermediate content.

---

## 5. Access levels and claims

Define three structure-access settings so the paper is transparent.

### S1 — Gold topology, hidden intermediate text

The main CSI pseudo-labeling condition. The constructor sees:

- leaf texts;
- parent/child groupings;
- root hypothesis;
- node IDs.

It does not see:

- intermediate conclusion text;
- gold proof serialization;
- gold alternative trees.

### S2 — No topology at inference

The final direct model receives the standard Task 1 or Task 2 input and must generate the full tree. This is the primary evaluation.

### S3 — Predicted topology extension

Optional future work: use a small premise-grouping model to propose child sets. Do not include this in the minimum experiment unless S1 succeeds.

The headline wording must be “structure-provided compositional self-improvement,” not automatic entailment-tree induction.

---

## 6. Data acquisition and versioning

1. Download the official EntailmentBank release and evaluation repository.
2. Pin the data version, repository commit, and file hashes.
3. Use the official Task 1 train/dev/test files.
4. Parse:
   - question and answer;
   - hypothesis;
   - leaf sentence IDs and texts;
   - intermediate IDs and texts;
   - proof edges/tree;
   - metadata and WorldTree provenance.
5. Reproduce the official evaluator on a small set of gold proofs.
6. Store hidden intermediate text and proof strings in an audit-only file.

Use the official development and test splits only for final model selection and evaluation. All pseudo-label construction occurs within the official training split.

---

## 7. Seed/composition partition

Because the training set is small, every labeled intermediate step must be carefully accounted for.

Partition official training trees before extracting any local steps:

| Partition | Suggested fraction | Use |
|---|---:|---|
| Atomic seed trees | 35–40% | extract labeled one-step entailment examples |
| Composition trees | 50–55% | hide intermediate text; construct deeper pseudo-labels |
| Internal audit trees | 10% | threshold tuning and hidden evaluation |

Stratify by:

- number of entailment steps;
- number of leaves;
- question topic/table;
- hypothesis length;
- average premises per step.

All steps from one tree remain in one partition. Deduplicate near-identical facts and hypotheses across partitions where possible.

### Why seed steps may come from deeper seed trees

The in-distribution seed unit is a **single textual entailment step**, not necessarily a complete one-step tree. It is acceptable to extract all local steps from the atomic seed-tree partition because those are the initial human-labeled components. No labeled step from a composition tree may be used for training.

---

## 8. Canonical representation

Store each composition tree as:

```json
{
  "id": "...",
  "hypothesis": "...",
  "leaf_nodes": {
    "sent1": "...",
    "sent2": "..."
  },
  "topology": [
    {"children": ["sent1", "sent2"], "parent": "int1"},
    {"children": ["int1", "sent3"], "parent": "hypothesis"}
  ],
  "hidden_intermediates": {
    "int1": "..."
  },
  "hidden_gold_proof": "...",
  "num_steps": 2
}
```

The pseudo-label generator receives the record with `hidden_intermediates` and `hidden_gold_proof` removed.

---

## 9. Local generation task

### Input

```text
[MODE: ENTAIL_STEP]
Premise 1: Astronauts are humans.
Premise 2: Humans are animals.
Generate one concise conclusion that follows from all premises and is useful for continued reasoning.
Return only the conclusion.
```

### Output

```text
Astronauts are animals.
```

### Constraints

- one sentence;
- no premise IDs or commentary;
- no unsupported new entity;
- preserve quantities, polarity, and modality;
- prefer the most specific shared conclusion rather than a vague paraphrase.

The phrase “useful for continued reasoning” is necessary because many locally entailed statements are too weak for the parent step. It does not solve the granularity problem, so the whole-tree guard remains essential.

---

## 10. One-step verifier

A purely symbolic verifier is unavailable. Use a separate, small entailment verifier trained only on atomic seed-tree steps.

### 10.1 Positive examples

Each gold local step from atomic seed trees:

```text
premises -> gold conclusion
```

### 10.2 Hard negatives

Generate negatives by:

- pairing the premises with a conclusion from another step in the same topic;
- swapping an entity, quantity, direction, or polarity;
- dropping a necessary premise and retaining the conclusion;
- using a sibling intermediate;
- replacing the conclusion with one premise;
- negating the gold conclusion;
- combining a correct subject with an incorrect predicate.

### 10.3 Verifier model

Use a compact encoder classifier or text-to-text `entailed/not_entail` model. Keep it frozen during CSI rounds to prevent self-confirmation drift.

Train the generator and verifier on disjoint splits of the atomic seed trees where feasible. At minimum, use different random initialization and no generated positives for the verifier.

### 10.4 Verifier input

```text
Premises: <p1> [SEP] <p2> ...
Hypothesis: <candidate conclusion>
```

### 10.5 Thresholding

Tune a high-precision threshold on held-out atomic seed steps and hard negatives. The target is precision, not recall.

A stronger external NLI model can be an ablation, but the main result should not depend on a large judge model.

---

## 11. Bottom-up composition algorithm

```text
text[node] = gold leaf text for each leaf
proof_steps = []

for layer in topological order:
    children = topology[layer].children
    parent = topology[layer].parent

    if parent == hypothesis:
        candidate = given hypothesis
    else:
        candidate = generator(text[children])

    if local_verifier(children, candidate) < threshold:
        reject tree

    if structural_checks_fail(children, candidate):
        reject tree

    text[parent] = candidate
    proof_steps.append(children -> parent: candidate)

if all steps pass:
    return serialized proof_steps
```

### 11.1 Whole-tree compatibility checks

A locally valid conclusion may be unusable at its parent. Therefore also require:

- every generated intermediate is actually consumed by a parent;
- the parent step passes the verifier after substitution;
- entities and variables remain referentially consistent across the path;
- quantities and comparisons are not weakened or reversed;
- no two generated intermediates contradict one another;
- the root step entails the exact given hypothesis;
- the final tree is acyclic and uses all required leaves under the provided topology.

The entire tree is rejected if any parent interface fails. This is the analogue of filtering boundary interactions.

---

## 12. Guard variants

### G0 — No guard

Retain every bottom-up generation.

### G1 — Surface guard

Check sentence length, formatting, entity introduction, and obvious copying only.

### G2 — Local entailment guard

Apply the frozen one-step verifier to each generated edge.

### G3 — Full tree guard

Local verifier plus parent-compatibility, contradiction, and root-hypothesis checks.

### G4 — Dual-verifier agreement

Optional: require agreement from two small verifiers trained on disjoint seed partitions.

The key experiment compares G0, G2, and G3 at matched retained-example counts.

---

## 13. Curriculum

### Seed — atomic steps

Train:

- local entailment generator;
- frozen local verifier;
- direct one-step/full-tree writer on seed-tree examples.

### Round 1 — two-step composition trees

Select composition trees with exactly two entailment steps. Generate the first intermediate, verify it, then verify the root step to the given hypothesis.

### Round 2 — three-step trees

Use the round-1 generator/direct model as the current component model. Compose in topological order and reject any interface failure.

### Round 3 — four-or-more-step trees

Begin with trees of four steps, then add larger trees if enough labels pass. Cap the number of generated candidates per intermediate to preserve the low-compute design.

### Suggested scale

The dataset is small, so use nearly all accepted composition trees rather than forcing thousands of examples.

| Stage | Expected source | Target retained examples |
|---|---|---:|
| Atomic seed | steps from 35–40% of train trees | all valid extracted steps |
| Round 1 | 2-step composition trees | 150–300, depending on distribution |
| Round 2 | 3-step composition trees | 150–300 |
| Round 3 | 4+ step composition trees | 100–250 |

Augment data through premise-order permutations and harmless sentence-ID renaming, not by generating new factual content.

---

## 14. Direct proof-writer training

The student receives the standard Task 1 input:

```text
hypothesis: ...
context: sent1: ... sent2: ... sent3: ...
```

and outputs a complete proof in canonical form:

```text
sent1 & sent2 -> int1: ...; int1 & sent3 -> hypothesis;
```

Train in a structured internal representation and deterministically serialize to the official DOT-like format.

### Training mixture

- 30–40% atomic seed step replay;
- 20–30% gold seed-tree full proofs;
- 30–50% accepted current/prior pseudo-trees.

Use tree-level sampling so large trees do not dominate merely because they contain more steps.

---

## 15. Model and compute configuration

Recommended:

- T5-small/base for generator and direct writer;
- compact encoder classifier for verifier;
- maximum input 512–768 tokens for Task 1;
- local conclusion maximum 32 tokens;
- full proof maximum 256–384 tokens;
- greedy local generation plus at most one alternate candidate;
- one or two epochs per CSI round;
- full fine-tuning for T5-small or LoRA for larger models.

The official dataset is small enough that training is inexpensive. The main resource cost is running several baselines and semantic verification, not data volume.

---

## 16. Baselines

### B0 — Atomic-seed direct writer

Train only on seed-tree supervision and evaluate at all tree sizes.

### B1 — Direct full-tree pseudo-labeling

Ask the current model to emit a complete proof for the next-depth tree in one generation. Apply the same full-tree verifier budget.

### B2 — Unfiltered bottom-up composition

Generate local conclusions and assemble every tree.

### B3 — Guarded bottom-up composition

The proposed method.

### B4 — Frozen bottom-up generator

Use the seed generator and verifier for all depths without retraining.

### B5 — Latest generator without direct-writer retraining

Update local generation only; do not train on complete parent trees.

### B6 — Gold topology + gold intermediate text

Supervised structural upper bound. This is not self-improvement.

### B7 — Gold-supervised full-tree curriculum

Train on true full proofs with matched tree counts.

### B8 — Topology-visible inference

Give the direct model the gold topology at test time to separate planning from conclusion generation.

---

## 17. Evaluation metrics

### Official EntailmentBank metrics

Use the official scorer for:

- leaf F1 and all-correct;
- step F1 and all-correct;
- intermediate F1 and all-correct;
- overall all-correct.

Run the full BLEURT-based official scoring only on frozen evaluation outputs if compute is limited.

### Validity-oriented metrics

Reference scoring can penalize a valid alternative tree. Also report:

- fraction of locally verifier-valid steps;
- fraction of trees whose every step passes the frozen verifier;
- root-hypothesis entailment rate;
- structural well-formedness;
- manual validity on a stratified sample.

### Pseudo-label metrics

- semantic similarity of generated intermediates to hidden gold;
- accepted-tree precision under the official scorer;
- manual validity of accepted trees;
- guard acceptance rate;
- first failing layer;
- error type: unsupported, too vague, entity drift, quantity drift, or interface mismatch.

### Transfer metrics

- Task 1 direct proof generation;
- Task 2 proof generation with distractors;
- performance by number of entailment steps;
- performance on guard-rejected tree slices.

---

## 18. Manual evaluation protocol

Because natural-language entailment admits valid alternatives, manually assess at least 100 pseudo-trees across methods.

For each local step, label:

- valid entailment;
- invalid entailment;
- valid but too weak for parent;
- valid paraphrase of gold;
- unclear.

For each full tree, label:

- reaches hypothesis soundly;
- structurally valid but semantically broken;
- valid alternative proof;
- invalid.

Use two annotators on a 30-example overlap and report agreement. This is a small, bounded annotation effort for analysis, not training.

---

## 19. Required analyses

### 19.1 Step-count frontier

Plot Task 1 accuracy/validity by number of gold entailment steps across CSI rounds.

### 19.2 Local-validity versus global-compatibility

Separate:

- trees rejected because a local step is not entailed;
- trees whose local steps are valid but a parent interface fails;
- trees whose root does not reach the hypothesis.

This is the central structured-error analysis.

### 19.3 Guard precision/coverage

Plot manual and hidden-gold precision against retained-tree count for G1–G4.

### 19.4 Retraining versus frozen composition

Compare direct final inference, frozen bottom-up construction, and CSI direct training at matched model calls and accepted labels.

### 19.5 Distractor transfer

Train only on Task 1 pseudo-trees and evaluate on Task 2. This tests whether reasoning learned from clean leaves survives irrelevant facts.

### 19.6 Rejected-slice generalization

Evaluate direct CSI on trees rejected for:

- entity drift;
- quantity/comparison mismatch;
- parent-interface failure;
- verifier disagreement;
- larger branching factor.

Interpret cautiously because the guard is learned rather than exact.

---

## 20. Ablations

1. no guard versus local versus full-tree guard;
2. direct versus bottom-up pseudo-labeling;
3. retraining versus frozen generator;
4. one versus two candidate conclusions per node;
5. separate versus shared generator/verifier data;
6. verifier hard-negative types;
7. gold topology versus noisy topology;
8. local-step replay ratio;
9. conclusion-only versus conclusion-plus-rationale local targets;
10. Task 1 only versus Task 1 plus synthetic distractors.

The highest-priority ablation is **local verifier only versus full parent-compatibility guard**. It tests whether the important errors truly occur at composition interfaces.

---

## 21. Statistical protocol

- Keep official dev/test untouched.
- Partition official train at tree level before extracting atomic steps.
- Tune verifier thresholds on internal atomic/audit trees.
- Match retained-tree counts across B1–B3.
- Use one development seed and three final seeds for B0–B4.
- Bootstrap by full question/tree.
- Report official metrics and manual validity separately.
- Freeze manual-evaluation rubric before inspecting final methods.

---

## 22. Risks and mitigations

### Risk 1 — Gold topology is too much supervision

**Mitigation:** state the access setting explicitly, evaluate the final model without topology, and frame the experiment as “given decomposition.” Include topology-visible upper bounds and optional noisy-topology tests.

### Risk 2 — Local generator produces many valid but unhelpful conclusions

**Mitigation:** add parent-compatibility checks, use at most one alternate candidate, and analyze this failure mode directly rather than hiding it.

### Risk 3 — Learned verifier shares the generator's biases

**Mitigation:** train on disjoint atomic trees, freeze it, create targeted hard negatives, and manually audit accepted pseudo-trees.

### Risk 4 — Too few accepted deeper trees

**Mitigation:** begin with two-step trees, use premise-order augmentation, lower candidate diversity rather than verifier precision, and report sample-volume limitations honestly.

### Risk 5 — Official scorer penalizes valid alternative explanations

**Mitigation:** supplement reference scores with local validity and manual full-tree judgments.

### Risk 6 — Direct pseudo-label baseline receives a weaker guard

**Mitigation:** parse the direct proof into local steps and run the same verifier and structural checks. Match retained counts.

### Risk 7 — Task 2 degradation dominates

**Mitigation:** keep Task 2 as transfer, not a required training result. Retrieval and distractor selection are separate problems.

---

## 23. Go/no-go criteria

Proceed to a full three-round study only if the pilot achieves:

- at least 85% verifier accuracy and at least 90% precision on held-out atomic hard negatives;
- at least 75% semantic validity for greedy one-step generation on hidden atomic steps;
- at least 70% manually valid accepted two-step pseudo-trees;
- at least 100 accepted two-step composition trees;
- guarded composition exceeds direct full-tree pseudo-labeling in accepted-tree precision;
- round-1 direct Task 1 performance improves on two-step development trees.

Do not use EntailmentBank as a headline result if the learned guard cannot achieve high precision without collapsing coverage.

---

## 24. Expected paper claim

A successful result can support:

> Given an expert-provided entailment-tree topology, model-generated one-step conclusions can be composed into useful deeper supervision when both local entailment and cross-step compatibility are guarded. Retraining on those trees improves direct explanation generation beyond the atomic seed regime.

It does not support automatic proof planning or fully annotation-free explanation generation.

---

## 25. Implementation schedule

### Week 1 — Data and official scorer

- pin release and repository;
- parse Task 1 trees and proof syntax;
- build topology-hidden records;
- reproduce official scoring on gold and corrupted proofs;
- create tree-level partitions.

### Week 2 — Atomic generator and verifier

- extract seed steps;
- build hard negatives;
- train generator and frozen verifier;
- tune high-precision threshold;
- perform initial manual audit.

### Week 3 — Two-step CSI

- implement bottom-up composer and full-tree checks;
- generate direct, unfiltered, and guarded labels;
- train round-1 direct writer;
- evaluate two-step internal and dev slices.

### Week 4 — Three- and four-step CSI

- generate deeper pseudo-trees;
- train later rounds;
- run frozen-composition and gold upper bounds;
- produce interface-error analyses.

### Week 5 — Transfer and final evaluation

- evaluate Task 1 and Task 2;
- run manual validity study;
- finalize three-seed results and paper figures.

---

## 26. Reproducibility checklist

- [ ] Official data version and hashes recorded.
- [ ] Official scorer reproduced.
- [ ] Train partition fixed before extracting steps.
- [ ] Hidden intermediate text inaccessible to generator/guard.
- [ ] Gold topology access disclosed in every result table.
- [ ] Generator and verifier training sets separated.
- [ ] Direct baseline receives the same verifier budget.
- [ ] Every pseudo-tree stores local candidates, scores, and rejection reason.
- [ ] Manual-evaluation sample and rubric pre-registered.
- [ ] Task 1 and Task 2 results reported separately.

---

## 27. Primary sources

- Dalvi, B., Jansen, P., Tafjord, O., Xie, Z., Smith, H., Pipatanangkura, L., and Clark, P. (2021). [Explaining Answers with Entailment Trees](https://aclanthology.org/2021.emnlp-main.585/).
- Official evaluation code and public dataset mirror: [allenai/entailment_bank](https://github.com/allenai/entailment_bank).
- Official data page: [AllenAI EntailmentBank](https://allenai.org/data/entailmentbank).
