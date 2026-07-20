# Research Plan: ProofWriter-CSI

## Verifier-backed compositional self-improvement for natural-language deductive proofs

**Primary dataset:** ProofWriter D0, D1, D2, D3, and D5, with the open-world-assumption data as the main setting.

**Core curriculum:** depth 0–1 labeled seed → depth 2 pseudo-proofs → depth 3 pseudo-proofs → depth 4–5 pseudo-proofs.

**Main target:** direct prediction of `answer + proof` from a natural-language theory and query.

**Composition mechanism:** iteratively generate one-step implications, verify them symbolically, add them to the theory closure, and assemble accepted local proof fragments.

**Compute profile:** low to moderate. Theories are short and the verifier is exact; the main cost is several short generations per pseudo-labeled theory.

---

## 1. Executive summary

ProofWriter provides the cleanest verifier-backed natural-language reasoning setting for compositional self-improvement. Each example contains natural-language facts and rules, a query, a truth label, and proof annotations. The D0, D1, D2, D3, and D5 collections each contain about 100,000 questions, with answers requiring progressively deeper reasoning.

The proposed system uses one model in two prompt modes:

1. **STEP mode:** propose one new implication together with the premises and rule that license it.
2. **SOLVE mode:** answer a complete theory/query instance and emit a proof.

During pseudo-label generation, STEP predictions are checked by an exact symbolic verifier using the dataset's underlying logical representation. Accepted conclusions are added to the current closure. If the query or its explicit negation is reached, the verified local proof fragments are assembled into a full pseudo-proof. The resulting full `(theory, query) → (answer, proof)` example becomes training data for the direct SOLVE model.

The curriculum begins with D0/D1 supervision, constructs D2 pseudo-proofs, retrains, then proceeds through D3 and D5. Unknown cases are initially rejected rather than pseudo-labeled because failure to find a proof is not evidence of non-entailment under the open-world assumption.

The central novelty must be stated carefully. The original ProofWriter work already introduced iterative one-step implication generation and proof assembly. This study is not claiming that mechanism as new. It asks a different question:

> Can a verified iterative executor be used as a pseudo-label generator whose outputs are recycled into training, causing a small direct model's reliable proof-depth frontier to expand and reducing its dependence on iterative inference?

The mandatory baselines are therefore the original-style frozen iterative executor, direct self-training, unverified composition, and a gold-supervised depth curriculum.

---

## 2. Research question

> Does iterative retraining on symbolically verified, model-generated proofs improve direct natural-language deduction at depths beyond the labeled seed regime?

A successful result should show that:

- verified composed pseudo-proofs have substantially higher precision than direct depth-frontier predictions;
- the direct model improves at depths 2, 3, and 4–5 over successive rounds;
- retraining improves over a frozen seed executor under matched generation budgets;
- proof validity remains high even when exact proof strings differ from one gold proof;
- short-depth performance and Unknown calibration do not collapse;
- gains transfer to at least one stylistically different ProofWriter evaluation set if available.

---

## 3. Dataset facts and choice of setting

ProofWriter's D* family contains five depth bands:

| Dataset | Maximum question proof depth | Approximate questions |
|---|---:|---:|
| D0 | 0 | 100k |
| D1 | 1 | 100k |
| D2 | 2 | 100k |
| D3 | 3 | 100k |
| D5 | 5 | 100k |

The release includes closed-world and open-world variants. In the open-world setting, labels are `True`, `False`, or `Unknown`; each example also contains all possible proofs for provable conclusions as auxiliary annotations.

### Recommended primary setting

Use the open-world-assumption data but pseudo-label only **provable** cases during the main CSI loop:

- if the query is derived, label `True`;
- if the explicit negation of the query is derived, label `False`;
- if neither is derived within the verified search budget, reject rather than label `Unknown`.

This avoids the invalid inference that “not found” means false.

### Optional simpler pilot

Use the closed-world variant or a provable-only subset to debug the pipeline. Do not make the closed-world result the only headline because it hides the most important abstention issue.

---

## 4. What is composed

A theory contains facts and rules such as:

```text
Charlie is kind.
If someone is kind then they are young.
If someone is young then they are quiet.
```

Local model predictions are proof fragments:

```json
{
  "premise_ids": ["fact1"],
  "rule_id": "rule1",
  "conclusion": "Charlie is young."
}
```

and then:

```json
{
  "premise_ids": ["int1"],
  "rule_id": "rule2",
  "conclusion": "Charlie is quiet."
}
```

The composition operator:

1. verifies each local rule application;
2. assigns a new intermediate ID;
3. adds the conclusion to the closure;
4. joins the proof edges into an acyclic proof DAG;
5. stops when the query or its negation is reached.

The full pseudo-label is:

```text
ANSWER: TRUE
PROOF: fact1 & rule1 -> int1: Charlie is young;
       int1 & rule2 -> hypothesis;
```

The exact serialization should follow the official release or a canonical internal format with a deterministic adapter to official evaluation.

---

## 5. Novelty boundary

The research plan must explicitly separate three systems:

1. **Iterative proof execution:** already studied in ProofWriter.
2. **Frozen iterative inference:** use the step model repeatedly but never learn from its deeper proofs.
3. **Compositional self-improvement:** verified deeper proofs become training data for an updated direct model, and the updated model is used in later construction rounds.

The publishable claim is about item 3 versus items 1–2. Do not claim novelty for one-step proof generation, forward chaining, or proof-fragment assembly.

A useful secondary outcome is inference amortization: after CSI, the SOLVE model may answer directly in one generation while retaining much of the verified executor's accuracy.

---

## 6. Data acquisition and versioning

1. Download the official ProofWriter release from the AllenAI data page.
2. Record release version and archive checksum.
3. Preserve D0/D1/D2/D3/D5, CWA/OWA, and train/dev/test distinctions.
4. Inspect the provided logical forms, proof fields, theory IDs, and derivative iterative examples.
5. Build a canonical theory representation with:
   - fact IDs and literals;
   - rule IDs, antecedents, and consequent;
   - natural-language surface forms;
   - query literal;
   - hidden label and proofs.
6. Store hidden answers and gold proofs in an audit-only table.

### Split by theory, not by question

Many questions share a theory. All questions derived from the same theory must remain in one partition. Question-level splitting would leak the complete fact/rule context and many conclusions.

Use the official splits for final evaluation. Within training, create deterministic theory-level partitions for seed, pseudo-label generation, and internal audit.

---

## 7. Canonical task formats

### 7.1 STEP mode

```text
[MODE: STEP]
Facts and known implications:
- fact1: Charlie is kind.
- int1: Charlie is young.

Rules:
- rule2: If someone is young then they are quiet.

Target query: Charlie is quiet.
Propose one new implication that moves toward the query.
Return JSON with premise_ids, rule_id, and conclusion.
```

The target query is included to reduce irrelevant implication enumeration. A target-free enumeration condition can be an ablation.

### 7.2 SOLVE mode

```text
[MODE: SOLVE]
Theory:
...
Question: Charlie is quiet.
Return ANSWER and a proof using fact/rule IDs.
```

### 7.3 Output schema

Use constrained JSON internally:

```json
{
  "answer": "true",
  "proof_steps": [
    {
      "premises": ["fact1"],
      "rule": "rule1",
      "conclusion_id": "int1",
      "conclusion": "Charlie is young."
    }
  ]
}
```

Convert to official proof syntax only for evaluation. This makes parsing and verification robust.

---

## 8. Exact symbolic verifier

The verifier is the central guard. It must never search for a proof or supply a missing conclusion; it only validates a model proposal.

For every proposed step:

1. parse `premise_ids`, `rule_id`, and `conclusion`;
2. confirm all premise IDs are in the current closure;
3. retrieve the selected rule;
4. unify the rule variables with the premise literals;
5. compute the licensed consequent;
6. confirm the proposed conclusion exactly matches that consequent after canonicalization;
7. reject duplicate or already known conclusions;
8. reject contradictory conclusions if the dataset semantics disallow the combination;
9. add the step only after all checks pass.

For a complete proof:

- every intermediate must be introduced before use;
- the proof graph must be acyclic;
- the root must equal the query or explicit negation;
- answer polarity must match the reached literal;
- no unsupported premise may appear;
- all IDs must resolve.

The verifier should use the dataset's logical representation, not an LLM judge.

---

## 9. Pseudo-proof generation algorithm

### 9.1 Target-directed iterative generation

```text
closure = initial facts
proof_graph = empty
for step in 1..budget:
    proposal = model(STEP prompt with closure, rules, query)
    if verifier accepts proposal:
        add conclusion to closure and proof_graph
    else:
        block the exact proposal and optionally retry once

    if query in closure:
        return TRUE + extracted verified proof
    if neg(query) in closure:
        return FALSE + extracted verified proof
return REJECT
```

### 9.2 Search budget

Start with:

- maximum 3 proposals for D2;
- maximum 5 for D3;
- maximum 8 for D5;
- at most one retry after a rejected step;
- deterministic decoding plus a blocked-duplicate list.

Increase only if acceptance is too low. Large beam search would turn the task into expensive proof search and obscure the low-compute claim.

### 9.3 Relevant-rule restriction

For efficiency, a symbolic preprocessor may identify rules whose consequents can unify with the query or with a backward dependency cone. It may restrict the visible rule list, but it must not instantiate the correct proof or conclusion for the model.

Report a no-restriction ablation to show how much the target-directed preprocessor contributes.

### 9.4 Proof extraction

When the target literal is reached, trace only the ancestors needed for that literal. Exclude irrelevant accepted implications from the final pseudo-proof.

---

## 10. Curriculum

### Seed stage — D0/D1

Train STEP and SOLVE modes using labeled D0/D1 theories. Use the release's derivative one-step training data where appropriate, while preserving theory-level splits.

### Round 1 — D2

- hide D2 train answers and proofs;
- generate verified proofs with the seed STEP model;
- retain only reached query/negated-query cases;
- train the direct SOLVE model on accepted D2 pseudo-proofs;
- preserve STEP-mode seed replay.

### Round 2 — D3

Use the round-1 checkpoint as the STEP and SOLVE model. Generate D3 pseudo-proofs under the same exact verifier.

### Round 3 — D5

Target depth-4 and depth-5 cases. Because acceptance may be lower, sample a balanced subset and cap generation cost.

### Suggested low-compute sample counts

| Stage | Candidate questions | Accepted target |
|---|---:|---:|
| Seed D0/D1 | 10k–20k labeled | all selected |
| Round 1 D2 | 10k candidates | 3k–5k accepted |
| Round 2 D3 | 10k candidates | 3k–5k accepted |
| Round 3 D5 | 10k–15k candidates | 3k–5k accepted |

Subsampling is appropriate because each D* set is much larger than needed for a mechanism test.

---

## 11. Unknown cases

Unknown is the most important semantic trap.

### Main rule

If neither the query nor its negation is derived, the pseudo-labeler returns `REJECT`, not `UNKNOWN`.

### Later extension

Pseudo-label Unknown only with a complete symbolic closure certificate:

- enumerate all applicable rule instances to fixpoint using a trusted symbolic engine;
- confirm neither polarity is derivable;
- use this only as an oracle-assisted ablation, because the engine then solves the task.

### Evaluation

Always evaluate the final model on Unknown questions to detect over-prediction of True/False. Report per-class accuracy and calibration.

---

## 12. Training mixture

Use one checkpoint with explicit mode tokens.

A starting mixture per round:

- 30% STEP seed replay;
- 20% SOLVE D0/D1 replay;
- 20% prior-round verified SOLVE examples;
- 30% current-round verified SOLVE examples.

If STEP accuracy degrades, increase STEP replay. Do not train STEP mode on unverified generated steps.

Targets may include only the minimal proof needed for the query. This reduces output length and avoids teaching irrelevant closure enumeration.

---

## 13. Model and compute configuration

Recommended:

- T5-small/base or a 0.5B–1.5B instruction model;
- maximum context 512–1,024 tokens;
- maximum proof output 256 tokens;
- constrained JSON generation if available;
- greedy decoding for the first proposal;
- one blocked retry at most;
- LoRA for larger models;
- one epoch per round with early stopping;
- mixed precision and batched STEP generation.

The verifier runs on CPU. Cache canonicalized theories and failed proposals so repeated baselines do not redo parsing work.

---

## 14. Baselines

### B0 — Seed-only direct SOLVE

Train on D0/D1 and evaluate directly on D2/D3/D5.

### B1 — Direct pseudo-labeling

Ask the current SOLVE model to answer and prove the complete next-depth instance in one generation. Apply the same complete-proof verifier.

### B2 — Unverified iterative composition

Assemble STEP outputs without symbolic checking.

### B3 — Frozen verified iterative executor

Use the seed STEP model and verifier at every depth, with no retraining.

### B4 — Latest-checkpoint executor without direct SOLVE retraining

Update only the STEP model or reuse the latest checkpoint for execution, but do not train on full parent proofs.

### B5 — Verified compositional self-improvement

The proposed method: verified executor outputs become direct SOLVE training data after every depth round.

### B6 — Answer-only CSI

Train on the composed answer but omit the proof target.

### B7 — Gold-supervised depth curriculum

Train on true answers and proofs with matched counts.

### B8 — Symbolic solver oracle

Exact upper bound and verifier sanity check; not a learning baseline.

---

## 15. Evaluation metrics

### Answer metrics

- exact answer accuracy;
- classwise True/False/Unknown accuracy;
- accuracy by minimum proof depth;
- largest depth above 90% accuracy;
- calibration or confidence by class.

### Proof metrics

- exact proof match where appropriate;
- official proof correctness;
- verifier validity rate;
- minimal-proof accuracy;
- proof depth and step count;
- proportion of correct answers with invalid proofs.

### Pseudo-label metrics

- answer precision against hidden gold;
- proof validity;
- proof semantic correctness against any gold proof;
- acceptance rate;
- generation proposals per accepted example;
- first failing depth or rule type.

### Efficiency

- generated tokens per accepted proof;
- direct SOLVE latency;
- frozen executor calls and latency;
- verifier CPU time;
- training tokens per round.

---

## 16. Required analyses

### 16.1 Depth-frontier heatmap

Rows are proof depth 0–5 and columns are CSI rounds. Provide answer and proof-validity heatmaps.

### 16.2 Noise versus depth

Measure component STEP error and composed proof error. Compare observed pseudo-label error with a union-bound or independence estimate.

### 16.3 Verifier effect

Compare unverified and verified composition at matched retained-example counts. This is the closest analogue to filtered versus unfiltered addition.

### 16.4 Retraining versus frozen execution

Plot accuracy and inference cost for:

- direct seed;
- frozen verified executor;
- CSI direct model;
- CSI executor.

A strong result is either better accuracy at matched budget or similar accuracy with substantially fewer inference calls.

### 16.5 Structured slices

Report:

- positive versus negated query;
- single- versus multi-premise rule;
- linear versus branching proof;
- one versus many possible proofs;
- repeated variable/entity;
- explicit negation;
- Unknown;
- hand-authored or paraphrased OOD sets when available.

### 16.6 Filter-rejected generalization

Evaluate the direct model on questions whose pseudo-proof generation was rejected because of:

- invalid local step;
- duplicate loop;
- budget exhaustion;
- contradictory proposals;
- no reached target.

This tests whether clean verified proofs teach beyond the accepted construction slice.

---

## 17. Ablations

1. direct versus composed pseudo-labels;
2. exact verifier versus syntax-only checking;
3. retraining versus frozen executor;
4. proof target versus answer-only target;
5. target-directed versus target-free STEP prompts;
6. one versus two proposal retries;
7. D0/D1 seed size;
8. 1k, 3k, and 5k accepted examples per round;
9. CWA versus OWA;
10. same-model versus separate STEP and SOLVE adapters.

A separate STEP adapter may reduce interference while sharing a frozen base model. It is a useful low-compute ablation.

---

## 18. Statistical protocol

- Split and sample by theory ID.
- Match accepted counts across direct and composition methods.
- Tune proposal budgets and thresholds on internal train/dev only.
- Use one development seed and three final seeds.
- Report paired bootstrap intervals by theory, not by question.
- Freeze the verifier implementation and canonicalization before final comparisons.
- Manually inspect a random sample of accepted proofs to verify the verifier/serialization adapter.

---

## 19. Risks and mitigations

### Risk 1 — The verifier or parser effectively solves the task

**Mitigation:** the verifier may only validate a model-proposed step. Log exactly what information it computes and include a symbolic-solver oracle separately.

### Risk 2 — This duplicates the original iterative ProofWriter contribution

**Mitigation:** center the research question on recycling verified deeper proofs into training, direct-model frontier expansion, and inference amortization. Cite the original iterative method prominently.

### Risk 3 — Frozen iterative execution is near perfect

**Mitigation:** use a deliberately small model and fixed low proposal budget. Report efficiency even if accuracy ties. Do not claim superiority if CSI merely imitates the executor.

### Risk 4 — Unknown performance collapses

**Mitigation:** preserve labeled Unknown replay from the seed or a small gold calibration set, and never pseudo-label Unknown from search failure.

### Risk 5 — STEP model generates only irrelevant implications

**Mitigation:** include the target query, restrict to a backward relevance cone, and use one blocked retry. Measure how much each aid contributes.

### Risk 6 — Proof strings are correct but fail official formatting

**Mitigation:** train and verify in structured JSON, then use a deterministic serializer for official evaluation.

### Risk 7 — Multiple proofs make exact match misleading

**Mitigation:** prioritize verifier validity and answer correctness; use any-gold-proof matching where possible.

---

## 20. Go/no-go criteria

Proceed beyond D2 only if:

- seed STEP validity is at least 95% on held-out one-step examples;
- verified D2 pseudo-label answer precision is at least 98%;
- at least 30% of sampled provable D2 questions are accepted within budget;
- round-1 direct D2 answer accuracy improves by at least three points over B0;
- verified CSI is better than direct pseudo-labeling at matched accepted counts;
- Unknown accuracy drops by no more than three points.

Make ProofWriter a diagnostic rather than a headline result if the frozen executor is perfect and retraining offers neither accuracy nor efficiency gains.

---

## 21. Expected paper claim

A successful experiment supports:

> A symbolically verified one-step natural-language reasoner can construct high-precision deeper proof supervision. Recycling those proofs into training expands a small direct model's reliable proof-depth regime and can amortize iterative proof execution.

It does not support a claim that one-step iterative proof generation is novel.

---

## 22. Implementation schedule

### Week 1 — Data and verifier

- download and pin release;
- parse logical and natural-language fields;
- implement exact step and full-proof verification;
- validate against gold proofs;
- create theory-level splits.

### Week 2 — Seed model

- build STEP and SOLVE seed tasks;
- train D0/D1 checkpoint;
- measure one-step validity and direct depth generalization;
- implement target-directed proposal loop.

### Week 3 — D2/D3 CSI

- generate direct, unverified, and verified pseudo-labels;
- train round-1 and round-2 models;
- run frozen-executor comparisons.

### Week 4 — D5 and analysis

- generate a bounded D5 subset;
- train final model;
- evaluate depth, negation, Unknown, and OOD slices;
- produce efficiency and frontier plots.

---

## 23. Reproducibility checklist

- [ ] Official release version and hashes recorded.
- [ ] Theory-level split enforced.
- [ ] Hidden labels/proofs inaccessible to generator and guard.
- [ ] Verifier only validates model proposals.
- [ ] Unknown never inferred from failed search.
- [ ] Proposal budgets matched across methods.
- [ ] Structured output and serializer tested separately.
- [ ] Every accepted pseudo-proof stores all proposals and verifier decisions.
- [ ] Frozen iterative executor included.
- [ ] Original ProofWriter novelty boundary stated explicitly.

---

## 24. Primary sources

- Tafjord, O., Dalvi, B., and Clark, P. (2021). [ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language](https://aclanthology.org/2021.findings-acl.317/).
- Official data page: [AllenAI ProofWriter](https://allenai.org/data/proofwriter).
