# Research Plan: BFCL-Comp

## Low-compute compositional self-improvement for parallel function calling

**Primary dataset:** Berkeley Function Calling Leaderboard (BFCL), restricted to the non-agentic BFCL V1 single-turn Python categories.

**Core curriculum:** one function call → two independent calls → four independent calls. An eight-call evaluation is optional and does not need a third training round.

**Intended resource regime:** one small instruction-tuned language model, parameter-efficient fine-tuning, short outputs, greedy decoding, no browser or tool-agent rollouts.

---

## 1. Executive summary

This study asks whether compositional self-improvement can teach a small language model to emit increasingly large sets of independent function calls without requiring the model to pseudo-label the full harder request directly.

The experiment starts with a model that is reliable on BFCL V1 **Simple** examples, each of which requires one function call. To construct a two-call training example, the current model solves two one-call subproblems separately. Their parseable predicted calls are combined into a single pseudo-label for a joined user request. After fine-tuning on accepted two-call examples, the process repeats: the new model solves two two-call subproblems, whose outputs are combined into four-call supervision. Stronger schema checks are evaluated as an ablation rather than assumed by the primary method.

The headline comparison is:

1. **Direct pseudo-labeling:** predict the complete next-frontier answer in one generation.
2. **Unfiltered composition:** independently predict the components and concatenate all outputs.
3. **Guarded composition:** independently predict the components, reject structurally unsafe outputs, then concatenate.
4. **Frozen composition:** recursively compose the seed model without retraining.

The main evaluation uses automatically generated held-out tasks at one, two, four, and eight calls, plus the original BFCL V1 **Parallel** and **Parallel Multiple** categories as external in-benchmark evaluation. This is not presented as an untouched BFCL leaderboard submission because BFCL V1 Simple data is used to build training examples.

### 1.1 Implemented pilot decision record (2026-07-19)

The first implemented experiment is a gated seed-7 pilot using the selected Qwen3.5-4B atomic adapter. Genuine cross-function composition is the primary condition. The original 60-row hidden-composition and test pools each contain 60 unique function names. Across all 400 Simple rows, no two rows have identical full function schemas; only five pairs remain compatible after reducing the signature to function name, argument names, required arguments, and recursive types. The corpus therefore cannot support the originally proposed balanced genuine same-function curriculum.

The implementation preserves the calibrated split and adds a separately reported auxiliary condition with 20% semi-synthetic repeated-function requests. These requests use conservative literal scalar mutations with disjoint train/test replacement registries. Constructing their inputs and hidden audit calls uses existing annotations, so this arm is explicitly **oracle-assisted semi-synthetic data**, not part of the label-free cross-function claim.

The implemented conditions are:

1. seed direct generation;
2. frozen recursive composition;
3. direct two-/four-call pseudo-labeling with G4;
4. component composition with G1;
5. component composition with G4;
6. G4 composition with a 20% synthetic-repeat mixture;
7. Oracle composition.

Oracle composition is an upper-bound control. It trains on the same composed prompts and matched data budget as the learned conditions, but its deterministic target concatenates the hidden true component calls. It is not self-improvement, and oracle calls are unavailable to model generation, G1/G4, and pseudo-label acceptance.

All generated data are persistent audit artifacts. Public candidates, hidden oracles, raw predictions, parsed calls, guard decisions, accepted and rejected rows, unique pseudo-labels, exact replay-expanded Trainer inputs, per-example evaluations, manifests, and checksums are kept under a timestamped `artifacts/runs/bfcl_compositional_pilot_*` directory.

The initial run completed Round 1 and was stopped by its predeclared gate. Post-run inspection showed that G1 and G4 accepted the same examples with the same targets, while condition-dependent materialization ordering exposed them to different training sequences and produced a misleading performance gap. The ordering key is now condition-invariant.

### 1.2 Revised fixed-curriculum decision (2026-07-20)

To match the manual addition experiment, the primary method is now G1 composition: require only enough JSON syntax and call shape to concatenate component predictions, without rejecting schema-valid semantic mistakes. G4 remains a structural-filter ablation. Hidden-oracle pseudo-label precision is reported after generation and never controls example acceptance or round progression.

The experiment runs the fixed manual 1→2→4 curriculum without a scientific promotion rule. Round 2 is submitted in the original Slurm DAG and starts after successful Round-1 jobs through `afterok` dependencies. Job failures and invalid or empty materializations remain operational stopping conditions.

---

## 2. Research question

> Can iterative composition of reliable single-call predictions expand a small model's function-calling frontier more reliably than direct self-training on multi-call requests?

A positive result should establish all of the following:

- multi-call accuracy increases beyond the seed regime;
- composition-based pseudo-labels are more accurate than direct pseudo-labels at the same frontier;
- retraining improves over simply composing the original seed model forever;
- schema-based guarding is measured as an ablation against the G1 primary condition;
- gains are not explained only by degradation on the original one-call task.

---

## 3. Why BFCL fits the framework

BFCL V1 provides a compact, executable function-calling setting without requiring long-horizon agents. The official release includes Python Simple, Parallel, Multiple, and Parallel Multiple categories, with both AST-based and executable subsets. The original dataset card reports:

| BFCL V1 Python category | AST examples | Executable examples | Role here |
|---|---:|---:|---|
| Simple | 400 | 100 | Atomic source and one-call evaluation |
| Parallel | 200 | 50 | Natural multi-call evaluation |
| Multiple | 200 | 50 | Optional distractor-function source |
| Parallel Multiple | 200 | 40 | Multi-call plus function-selection evaluation |

The experiment uses the same mechanism as the uploaded *Compositional Self-Improvement* draft:

\[
\text{decompose hard input}
\rightarrow
\text{predict easier components}
\rightarrow
\text{guard and compose outputs}
\rightarrow
\text{fine-tune}
\rightarrow
\text{repeat}.
\]

For this task:

- an atomic input is a user request plus function documentation that requires one call;
- input composition joins independent requests and unions their function documentation;
- output composition flattens the predicted function-call lists;
- the primary G1 condition checks JSON syntax and call shape, while the guard ablation adds membership, argument, count, and cross-component checks.

---

## 4. Proposed claims and hypotheses

### H1 — Composition produces cleaner frontier supervision

At two and four calls, guarded composition will have higher pseudo-label exact-match accuracy than direct prediction on the complete joined request.

### H2 — Iterative retraining expands the reliable call-count frontier

A model retrained after the two-call round will outperform both the seed model and frozen recursive composition on four-call tasks.

### H3 — The guard trades quantity for substantially higher quality

Guarded composition will retain fewer examples than unfiltered composition but will reduce systematic errors such as malformed calls, missing calls, duplicated calls, wrong function names, and invalid argument types.

### H4 — One-call ability is preserved

Replay of seed data will keep final one-call accuracy within two percentage points of the seed checkpoint.

### H5 — Benefits transfer beyond the exact prompt-conjunction templates

Improvements will remain visible on held-out joining templates and on the original BFCL Parallel and Parallel Multiple questions.

---

## 5. Scope

### Main scope

- BFCL V1 non-live Python data only.
- Single-turn generation only.
- Independent calls only.
- One consistent text serialization for calls.
- Two self-improvement rounds: 1 → 2 → 4 calls.
- Evaluation up to eight calls.

### Explicitly out of scope

- BFCL V3 multi-turn tasks.
- BFCL V4 web-search and memory agents.
- Executing arbitrary external APIs.
- Calls whose arguments depend on another call's returned value.
- Long natural-language rationales.
- Reinforcement learning.

This restriction is intentional: the experiment should isolate compositional self-improvement rather than agent engineering.

---

## 6. Data acquisition and versioning

1. Use the official Hugging Face dataset snapshot at revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`.
2. Download the non-live `simple`, `parallel`, and `parallel_multiple` question files and their `possible_answer` files directly as JSONL. Download executable files separately for the behavioral evaluation track.
3. Pin the official Gorilla evaluator commit independently and record both data and evaluator revisions in the experiment manifest.
4. Save the original JSONL files unchanged under `data/raw/`.
5. Build all derived examples under `data/processed/<source_commit>/`.
6. Record hashes of every source file and every generated split.

The BFCL dataset card notes that its files are organized as category-specific JSON files and should be loaded directly rather than through the standard Hugging Face `load_dataset` interface.

### 6.1 Initial pinned-data audit

The exploratory notebook [`coding_task_data_exploration.ipynb`](coding_task_data_exploration.ipynb) inspected the pinned AST files and found:

| Category | Rows | Reference-call count distribution | Rows with one accepted value per argument |
|---|---:|---|---:|
| Simple | 400 | `1: 400` | 137 |
| Parallel | 200 | `2: 110`, `3: 52`, `4: 35`, `6: 1`, `8: 2` | 77 |
| Parallel Multiple | 200 | `2: 64`, `3: 66`, `4: 69`, `5: 1` | 49 |

The last column does not define the usable pool. BFCL stores a set of accepted values for many arguments, including optional/default variants. The canonicalization policy below preserves those option sets for evaluation while selecting one deterministic target for SFT. This audit covers the AST files only; executable subsets remain a separate evaluator-backed track.

### 6.2 Atomic-supervision calibration

Before any compositional round, run the dedicated Qwen3.5-4B LoRA calibration implemented by `self.experiments.coding_atomic_sweep`. The pinned Simple rows produce 400 structurally usable examples, with one deterministic repair from an unqualified reference name to its unique namespaced schema. Under the model's non-thinking chat template, complete prompt-plus-target lengths range from 143 to 372 tokens, so the calibration uses a 512-token limit without truncation.

The staged sweep uses nested training subsets of 30, 60, 120, and 240 examples; learning rates `1e-5`, `5e-5`, and `2e-4`; and 10, 30, and 100 optimizer steps. It screens nine full-data cells, evaluates the two best schedules at the three smaller data sizes, and repeats the best three configurations with two additional seeds, for 21 LoRA runs. Hyperparameters are selected only on the atomic validation pool. The fixed atomic test pool is opened after selection, and the winning configuration is then evaluated on controlled 2/4/8-call tasks and the natural Parallel and Parallel Multiple files.

The executed zero-shot baseline exposed a sharp format-learning problem: strict format and exact accuracy were 0% on atomic validation and test and on all three controlled frontiers because Qwen wrapped otherwise plausible JSON arrays in Markdown fences. A one-update LoRA smoke run on 30 examples raised atomic validation to 77.5% exact with 100% format validity and training-slice accuracy to 93.3%, without an OOM at microbatch 16. This smoke result is diagnostic rather than a selected configuration, but it confirms that the sweep can distinguish rapid protocol acquisition from later semantic and compositional gains.

The selected atomic recipe is Qwen3.5-4B with ordinary LoRA rank 16, alpha 32, dropout 0, and all linear layers targeted; 240 atomic examples; 30 optimizer steps; learning rate `2e-4`; effective batch size 16; and maximum length 512. Across seeds 7, 23, and 42 it achieved 91.7% mean atomic test exact accuracy and 90% minimum-seed accuracy. Multi-call transfer was highly seed-sensitive: seed 7 reached 81%, 72%, and 49% on the controlled 2/4/8-call sets, whereas seeds 23 and 42 generally emitted one call and were near zero. The fixed-curriculum pilot therefore uses seed 7 first and treats the other seeds as later replication targets rather than launching a full three-seed cross-product immediately.

---

## 7. Canonical task format

Normalize every atomic item to:

```json
{
  "source_id": "simple_python_...",
  "question": "...",
  "functions": [
    {
      "name": "get_weather",
      "description": "...",
      "parameters": {"type": "object", "properties": {}, "required": []}
    }
  ],
  "hidden_reference_options": [
    {"name": "get_weather", "arguments": {"city": ["Paris"]}}
  ],
  "canonical_reference_calls": [
    {"name": "get_weather", "arguments": {"city": "Paris"}}
  ],
  "evaluation_mode": "ast_or_exec",
  "component_count": 1,
  "source_component_ids": ["simple_python_..."],
  "source_group_id": "simple_python_...",
  "evaluation_track": "seed",
  "composition_family": "atomic"
}
```

The model target is always serialized as a JSON list, even for a single call:

```json
[
  {"name": "get_weather", "arguments": {"city": "Paris"}}
]
```

For each argument, choose the first non-empty official accepted value for `canonical_reference_calls`; omit an optional argument when its accepted list contains only the empty string. Use this deterministic call list for SFT, but retain every accepted value in `hidden_reference_options` and use the complete option set for evaluation. Use one model-facing serialization throughout training and evaluation, and convert it to the official BFCL representation only inside the evaluation adapter.

Every derived example records `component_count`, `source_component_ids`, `source_group_id`, `evaluation_track` (`controlled`, `natural`, or `rejected` outside the seed pool), and `composition_family` (`same_function` or `cross_function`).

---

## 8. Atomic-pool construction

### 8.1 Required filters

Retain a Simple item only when:

- it has exactly one intended call;
- its function documentation is valid JSON Schema or can be deterministically normalized to it;
- the reference call parses;
- all required arguments are present in the reference call;
- the question is self-contained and does not rely on preceding dialogue;
- the prompt and target fit the chosen context and output limits;
- no function name collides with another incompatible schema in the same generated example.

### 8.2 Optional easy-first filter

For the minimum viable experiment, initially retain only:

- scalar, list, and shallow dictionary argument types;
- at most five required arguments;
- no highly nested arrays or dictionaries;
- questions shorter than a fixed threshold, such as 128 tokens.

After the pipeline works, add the excluded type-complexity slice as a harder evaluation band.

### 8.3 Atomic split

Split eligible atomic sources before constructing any composed examples. For the pinned 400-row AST Simple corpus, create four disjoint pools stratified by argument-type complexity:

- **Seed labeled pool:** used to fine-tune or calibrate the one-call seed model.
- **Hidden composition pool:** reference labels are stored for auditing but never used by pseudo-label generation.
- **Atomic validation pool:** used for seed hyperparameter selection and thresholds.
- **Atomic test pool:** used only after seed selection for final one-call retention and controlled-composition construction.

A fixed initial split is:

| Pool | Share | Count | Use |
|---|---:|---:|---|
| Seed labeled | 60% | 240 | Initial SFT and replay |
| Hidden composition | 15% | 60 | Source components for pseudo-label generation |
| Atomic validation | 10% | 40 | Seed selection and thresholds |
| Atomic test | 15% | 60 | Final atomic and controlled-composition evaluation |

An atomic `source_id` occurs in exactly one pool, regardless of how many generated combinations use it. If executable Simple sources are added later, partition them independently and retain the same source-isolation invariant rather than silently changing this AST split.

---

## 9. Generated task families

Generate one primary genuine family and one separately reported semi-synthetic auxiliary family.

### 9.1 Genuine same-function parallel — unavailable from Simple at useful scale

The intended family invokes one compatible function schema with different argument sets.

```text
Get the weather in Paris. Also get the weather in Rome.
```

```json
[
  {"name": "get_weather", "arguments": {"city": "Paris"}},
  {"name": "get_weather", "arguments": {"city": "Rome"}}
]
```

The pinned Simple corpus cannot supply this family at useful scale: it has zero identical-schema pairs and only five reduced-signature-compatible pairs across all 400 rows. Repartitioning would invalidate the calibrated seed adapters while leaving statistically empty train/test cells. Do not use those five pairs for a headline training or evaluation claim. Original BFCL Parallel remains the genuine repeated-function and compressed-language transfer test.

### 9.2 Cross-function parallel

Each clause invokes a different independent function.

```text
Get the weather in Paris. Also convert 100 USD to EUR.
```

The composed function set is the de-duplicated union of the component function documents.

This is the primary generated curriculum. The fixed hidden pool provides all 1,770 source-disjoint-within-example pairs and more than enough compatible four-source groups. Function names need not have appeared in seed SFT because every prompt supplies the exact schema and name; the atomic test already demonstrates high accuracy on mostly unseen names.

### 9.3 Semi-synthetic repeated-function auxiliary arm

For an atomic row whose scalar argument value appears exactly once in its question, deterministically replace one value and update the hidden audit call. Permit only bounded numeric changes, alternate schema enums, and curated place, country/state, color, or currency substitutions. Reject ambiguous spans, collections, free-form identifiers, and constraint violations. Use disjoint mutation values for hidden-pool training and test-pool evaluation.

The implemented hidden pool yields 135 distinct mutated pairs from 35 atomic sources. Render each semantic pair under two training templates so a 20% auxiliary mixture can be materialized while preserving the shared source-group provenance. This arm tests whether explicit repeated-function exposure improves original Parallel transfer; it must never be conflated with the label-free cross-function result.

### 9.4 Optional distractor-function track

Sample one to three irrelevant function documents from other BFCL items. Keep the target calls unchanged. This creates a selection-plus-multiplicity setting closer to Parallel Multiple.

Run this only after the main no-distractor pipeline is stable.

---

## 10. Input composition

### 10.1 Joining templates

Use deterministic joining templates; do not spend compute on an external paraphrasing model. Assign templates and clause order reproducibly from the experiment seed and candidate ID.

Training templates may include:

```text
{q1} Also, {q2}
```

```text
Complete both requests: (1) {q1} (2) {q2}
```

```text
Please do the following independently: {q1}; {q2}
```

```text
First request: {q1}\nSecond request: {q2}
```

The implemented training registry is `also`, `in_addition`, `then`, and `independent_semicolon`. The held-out registry is `numbered`, `bullets`, and `ordinal` (first/next/finally). It supports arbitrary arity and is disjoint by construction. Original BFCL remains harder than this lexical template shift because many Parallel prompts compress shared arguments with lists, “respectively,” or “repeat” constructions.

### 10.2 Clause normalization

- Strip terminal punctuation before joining.
- Lowercase only when required by the model tokenizer; otherwise preserve source text.
- Reject clauses with unresolved anaphora such as “it,” “that city,” or “the previous result,” unless a deterministic self-containment checker can resolve them.
- Randomize clause order while keeping target-call order aligned.

### 10.3 Output order

The training target follows prompt-clause order. The primary evaluator treats independent calls as a **multiset**, so a semantically correct permutation is not marked wrong.

---

## 11. Roundwise compositional self-improvement

Use binary composition at every round.

### Round 0 — seed

Train or select a checkpoint reliable on one-call tasks.

### Round 1 — two calls

For each target pair:

1. Query the current model separately on each one-call component.
2. Normalize each predicted one-call list.
3. Apply the component guard.
4. Flatten the two accepted lists into a two-call pseudo-label.
5. Pair it with the joined two-clause prompt.
6. Fine-tune on accepted two-call examples plus seed replay.

### Round 2 — four calls

For each target four-call example:

1. Partition it into two two-call subproblems.
2. Query the Round-1 model on each two-call subproblem.
3. Guard both predicted lists.
4. Flatten them into a four-call pseudo-label.
5. Fine-tune on accepted four-call examples, previous-round examples, and seed replay.

### Optional eight-call evaluation

Evaluate the final model directly on eight-call prompts. Also evaluate recursive composition of two predicted four-call subproblems. A third training round is optional rather than required.

---

## 12. Guard specification

The guard must not compare against hidden reference answers during pseudo-label construction.

### 12.1 Component-output guard

For each predicted subproblem output:

1. Parse as JSON.
2. Require a list of the expected length.
3. Require every item to contain exactly `name` and `arguments`, after harmless normalization.
4. Require `name` to match one of the function documents available to that subproblem.
5. Require `arguments` to be an object.
6. Require all schema-required arguments.
7. Reject unknown arguments unless the schema explicitly permits them.
8. Validate argument types recursively.
9. Reject duplicated identical calls within a component.
10. Enforce a short output-length ceiling.

### 12.2 Cross-component composition guard

After all components pass individually:

1. Reject incompatible duplicate function schemas with the same name.
2. Reject exact duplicate calls unless the prompt explicitly requests repetition.
3. Reject any component whose question contains a detected reference to another component's result.
4. Require the total number of calls to equal the sum of component call counts.
5. Canonicalize calls for training and evaluation.

### 12.3 Executable guard

On BFCL executable items, optionally execute the predicted call in the official sandbox. Use execution only as a structural/safety check during pseudo-label generation; do not compare the returned value against a hidden gold answer unless the same check is also available to the direct-pseudo-label baseline.

### 12.4 Guard ablation levels

Report a cumulative guard ladder:

| Level | Checks |
|---|---|
| G0 | No filtering |
| G1 | JSON parsing and output shape |
| G2 | Function membership and call count |
| G3 | Required arguments, no extras, recursive type validation |
| G4 | Duplicate/collision and dependency checks |
| G5 | Optional executable validation |

This reveals which checks provide the actual gain.

The pilot trains G1 and G4 composition arms and does not add a cross-view stability guard. G1 is the primary addition-matched condition. G4 is a compatibility and verification-signal ablation, not a semantic oracle: a prediction that supplies `"Rome"` instead of `"Paris"` can pass when both are valid strings. The hidden-reference audit reports accepted-label precision and false accepts alongside acceptance rate, but those audit labels do not filter training data or determine round progression.

---

## 13. Hidden references and pseudo-label auditing

Although source BFCL items contain accepted reference-value sets, the self-improvement generator must not use them. Keep `hidden_reference_options` and the derived canonical calls in a separate evaluation store that the generator cannot import.

For every generated task, construct an **oracle composite reference** by composing the hidden source option sets. The canonical call list may be used as a deterministic oracle SFT target only in the explicit upper-bound condition. Otherwise use the hidden references only to calculate:

- pseudo-label exact-match precision;
- function-name accuracy;
- argument-key and argument-value accuracy;
- error correlations across components;
- false-accept and false-reject rates of the guard.

This separation is essential: the experiment should measure whether model-produced components are reliable enough to bootstrap, not whether existing BFCL labels can be concatenated.

Persist the public and oracle streams separately. Public candidates contain prompts, schemas, component IDs, expected call count, template, and family but no target or accepted values. Oracle files contain canonical and accepted calls keyed by candidate ID. Only the explicit Oracle baseline and post-generation audit commands may open the oracle directory.

---

## 14. Train, validation, and test separation

Split atomic sources before composition, then apply complementary schema and template holdouts. The exact question/argument prohibition alone is insufficient because many generated tasks can reuse one atomic item.

### 14.1 Atomic-source and complexity split

The same function families may appear at different call counts, but no atomic `source_id` may appear across seed, hidden-composition, validation, or test pools. Construct every controlled evaluation item exclusively from atomic test sources. This tests easy-to-hard call-count generalization without source reuse across partitions.

### 14.2 Schema-disjoint split

Normalize each function schema to a signature hash containing the function name, argument names, required set, and recursive types. Reserve a subset of signature hashes for evaluation only.

This tests whether gains transfer to unseen function definitions rather than merely memorizing familiar APIs.

### 14.3 Template-disjoint split

Reserve joining templates for evaluation only.

### 14.4 Official BFCL evaluation

Evaluate on the original BFCL V1 Parallel and Parallel Multiple files. Report overlap statistics between training schemas and official evaluation schemas. Label the result clearly as **BFCL-derived evaluation**, not an untouched public leaderboard result.

---

## 15. Model and training protocol

### 15.1 Model regime

Use the calibrated Qwen3.5-4B checkpoint. The seed-7 adapter is the pilot checkpoint because it combines 90% atomic test accuracy with a nontrivial but incomplete multi-call frontier. Seeds 23 and 42 are reserved for replication after a positive pilot.

### 15.2 Parameter-efficient tuning

Implemented default:

- ordinary LoRA rank 16, alpha 32, dropout 0, targeting all linear layers;
- continue from the selected seed adapter rather than initializing a new adapter;
- learning rate `2e-4`, effective batch size 16, and automatic microbatch reduction on OOM;
- maximum training length 1,024 with over-length examples rejected rather than truncated;
- one materialized-dataset epoch per self-improvement round;
- greedy decoding with no sampling-based majority vote;
- at most 1,000 accepted new examples per round, matched across the main learned arms.

### 15.3 Replay mixture

At every round, mix:

- 30–40% seed one-call examples;
- 20–30% previous-frontier examples;
- 40–50% new accepted frontier examples.

Tune this mixture once on validation data, then freeze it across baselines.

### 15.4 Suggested data scale

| Stage | Accepted new examples | Notes |
|---|---:|---|
| Seed | 250–300 labeled atomic items | Existing BFCL labels |
| Round 1 | 1,000–2,000 two-call items | Sampled combinations from hidden atomic pool |
| Round 2 | 1,000–2,000 four-call items | Built from two-call subproblems |
| Optional Round 3 | 500–1,000 eight-call items | Only when the two-round result is strong |

The combinatorial source pool permits many generated examples without acquiring more annotations, but combinations that reuse atomic sources are statistically correlated. Record source IDs on every generated row and do not interpret the raw number of combinations as an equal number of independent annotations.

---

## 16. Baselines

All learned baselines use the same model, tokenizer, SFT hyperparameters, replay budget, and number of accepted training examples where possible.

### B0 — Seed only

No self-improvement.

### B1 — Direct pseudo-labeling

Generate the full two- or four-call output in one model call. Apply the same output-level JSON/schema guard used for the composition condition. This controls for filtering alone.

### B2 — Unfiltered composition

Predict components separately and flatten every parseable output, without schema or cross-component filtering beyond minimal serialization.

### B3 — Guarded composition

The proposed method.

### B4 — Frozen recursive composition

Use the seed model to solve atomic components recursively at every frontier, but never fine-tune. This tests whether retraining is necessary.

### B5 — Oracle-composed SFT upper bound

Train on generated multi-call prompts paired with composed hidden reference calls. This estimates the headroom remaining after pseudo-label noise is removed.

### B6 — Direct gold frontier upper bound, optional

Use human/reference multi-call labels for a small matched subset. This should remain an analysis condition, not the main comparison.

### Fairness controls

Report both:

- **matched training-example budget**, and
- **matched generation-token budget**.

Composition uses more component generations per hard example, so both views are informative.

---

## 17. Evaluation sets

### 17.1 Controlled generated frontier — primary

Construct fixed semi-synthetic evaluation sets exclusively from the atomic test pool. The atomic cell contains its 60 genuine rows. The controlled cross-function grid contains 200 examples at each call count `2`, `4`, and `8` under seen templates and another 200 per count under held-out templates. A balanced genuine same-function grid is impossible from Simple and is not fabricated.

Report the auxiliary mutation evaluation separately: up to 80 two-call repeated-function renders and 100 four-call paired-repeat tasks, with source-group-aware counts. Use original Parallel and Parallel Multiple—not the synthetic cell—as the external test of genuine repeated-function and compressed-language transfer.

Annotate, rather than fully cross, the following diagnostic factors:

- argument complexity: scalar, collection, nested;
- function-document count: relevant-only versus distractors;
- schema overlap: seen versus unseen;
- join template: seen versus held-out.

Report exact example counts and distinct atomic-source counts for every cell. Because test atoms may participate in multiple generated examples, compute 95% confidence intervals with an atomic-source-aware cluster bootstrap.

### 17.2 Original BFCL categories — natural secondary

Use:

- Simple Python for retention;
- Parallel for natural multi-call transfer, stratified by reference-call count;
- Parallel Multiple for selection plus multiplicity, stratified by reference-call count;
- executable subsets as a separate behavioral score.

The pinned AST audit supplies 174 natural two-call, 118 three-call, and 104 four-call examples across Parallel and Parallel Multiple, but only two eight-call examples. Therefore the natural track can support useful one-, two-, three-, and four-call reporting but cannot support a reliable eight-call frontier estimate. The three-call slice is a useful interpolation diagnostic even though it is not a training stage.

### 17.3 Guard-rejected interactions — diagnostic

Build a diagnostic set from examples rejected by the composition guard, including:

- unresolved anaphora or cross-clause references;
- incompatible duplicate function schemas;
- exact duplicate calls;
- outputs with nested-type edge cases;
- dependent-call templates in which a later argument must come from an earlier result, if a small synthetic set is added.

Do not train on this slice. Test whether clean independent-call supervision transfers at all to these harder interactions.

Every evaluation row stores `component_count`, `source_component_ids`, `source_group_id`, `evaluation_track`, and `composition_family` or a named rejection slice.

---

## 18. Metrics

### Primary

- **Controlled all-calls exact accuracy by call count:** the predicted multiset exactly matches the full oracle option set on the semi-synthetic frontier.
- **Official BFCL AST accuracy** on the applicable V1 categories, reported as natural transfer.
- **Official BFCL executable accuracy** on the executable subset, reported separately.

### Diagnostic

- valid-JSON rate;
- valid-schema rate;
- correct call-count rate;
- function-name micro/macro accuracy;
- argument-key F1;
- argument-value accuracy;
- per-call accuracy;
- pseudo-label precision before and after guarding;
- guard acceptance rate;
- guard false-accept and false-reject rates;
- one-call retention after each round;
- examples and distinct source groups per evaluation cell;
- source-grouped 95% confidence intervals;
- generated and training tokens;
- wall-clock time.

### Frontier summary

Define the reliable frontier as the largest call count with at least 90% exact accuracy, or use a lower threshold if no model reaches 90% on the one-call seed. Report the full accuracy-by-call-count curve regardless of threshold.

Apply the reliable-frontier definition to the controlled track. Treat natural cells with too few examples, especially eight calls, as descriptive only.

---

## 19. Main experimental matrix

The minimum publishable matrix is:

| Condition | Round 1 train | Round 2 train | Final evaluation |
|---|---|---|---|
| Seed only | — | — | 1/2/4/8 + BFCL |
| Direct | Direct 2-call labels | Direct 4-call labels | 1/2/4/8 + BFCL |
| Unfiltered composition | Composed 2-call labels | Composed 4-call labels | 1/2/4/8 + BFCL |
| Guarded composition | Guarded 2-call labels | Guarded 4-call labels | 1/2/4/8 + BFCL |
| Frozen composition | No training | No training | Recursive 2/4/8 |
| Oracle upper bound | Gold-composed 2-call | Gold-composed 4-call | 1/2/4/8 + BFCL |

The implemented pilot names these arms `direct_g4`, `compose_g1`, `compose_g4`, `compose_g4_repeat20`, and `oracle`. `compose_g1` is the primary condition. The repeat arm is an auxiliary comparison against `compose_g4`, not a replacement for it. Direct and G4 use the same output-level structural checks. Oracle uses true hidden targets only for its explicitly labeled upper-bound SFT data.

For a very limited budget, run one random seed for all conditions, then repeat only Seed, Direct, and Guarded Composition with two additional random seeds.

---

## 20. Ablations

Prioritize the following:

1. **Guard strength:** G0 through G4, with G5 only on executable items.
2. **Retained data size:** 250, 1,000, and 2,000 examples per round.
3. **Seed quality:** use two or three seed checkpoints with different one-call accuracy.
4. **Replay fraction:** no replay versus 30–40% seed replay.
5. **Composition arity:** binary 1→2→4 versus direct 1→4 composition.
6. **Same-function versus cross-function composition.**
7. **Seen versus unseen function schemas.**
8. **Seen versus held-out joining templates.**

Do not run every cross-product. Change one factor at a time around the default Guarded condition.

---

## 21. Error taxonomy

Manually inspect a fixed sample of errors and classify them as:

- invalid JSON or extraneous text;
- wrong number of calls;
- missing call;
- duplicate call;
- wrong function selection;
- missing required argument;
- hallucinated argument;
- wrong argument value;
- wrong nested type;
- cross-clause argument swap;
- order-only mismatch;
- official-evaluator adapter error.

The most important expected compositional failure is **cross-clause binding**: every call is individually plausible, but an argument from one request is attached to another function call.

---

## 22. Low-compute execution plan

### Minimum viable run

- Qwen3.5-4B with the selected 240-example seed-7 LoRA adapter;
- up to 1,000 accepted new examples per round;
- the full Round-1 and Round-2 DAG submitted upfront with `afterok` dependencies;
- one materialized-dataset epoch per round;
- maximum training context 1,024;
- greedy decoding;
- Direct G4, Composition G1, Composition G4, auxiliary G4+repeat20, and Oracle learned conditions;
- Seed-only and Frozen baselines require no additional SFT;
- one random seed initially.

### Expanded follow-up

- 2,000 accepted examples per round;
- two additional random seeds for the main comparison;
- distractor-function track;
- eight-call training round;
- executable guard ablation.

The study should be feasible without a powerful agent model because each generation is a short static JSON prediction.

---

## 23. Diagnostics and operational checks

There is no accuracy-based promotion rule. Every scheduled round runs when its prerequisite jobs finish successfully. The following are report-only diagnostics:

1. seed accuracy and the initial accuracy gap across call counts;
2. pseudo-label precision and acceptance for Direct, G1, G4, and the repeat arm;
3. frontier gain or loss after each training round;
4. atomic retention after each training round;
5. controlled-cell counts, source diversity, and template-stratified accuracy.

Operational checks still fail a job on missing component predictions required for materialization, an empty matched training budget, invalid persisted artifacts, non-finite training, or another runtime error. Slurm `afterok` dependencies then prevent downstream work. These checks ensure a runnable pipeline without using semantic correctness to select examples or rounds.

---

## 24. Risks and mitigations

### Risk: prompt joining is too synthetic

**Mitigation:** reserve joining templates, evaluate on original BFCL Parallel questions, and report template-disjoint performance.

### Risk: the task tests formatting more than reasoning

**Mitigation:** report argument binding and function selection separately; add distractor docs only after the base experiment works.

### Risk: schema overlap contaminates the external evaluation

**Mitigation:** publish schema-overlap statistics and include a schema-disjoint generated test.

### Risk: direct pseudo-labeling receives less inference compute

**Mitigation:** report matched-example and matched-generation-token comparisons.

### Risk: the guard accidentally uses reference answers

**Mitigation:** isolate hidden references in a separate file unavailable to the generation process; unit-test that the generator cannot import it.

### Risk: official BFCL files evolve

**Mitigation:** pin the repository commit and release all derived source IDs and hashes.

---

## 25. Expected paper figures and tables

1. **Accuracy heatmap:** call count × self-improvement round.
2. **Pseudo-label quality plot:** Direct, Unfiltered, and Guarded precision by frontier.
3. **Frontier curve:** exact accuracy at 1, 2, 4, and 8 calls.
4. **Guard tradeoff:** accepted examples versus pseudo-label precision.
5. **Original BFCL table:** Simple, Parallel, Parallel Multiple, AST and executable, stratified by natural call count.
6. **Error composition chart:** missing calls, binding errors, type errors, formatting errors.
7. **Compute table:** generation tokens, train tokens, and wall-clock time.

---

## 26. Four-week implementation schedule

### Week 1 — data and evaluator

- pin BFCL version;
- normalize schemas and references;
- create atomic splits;
- implement JSON/multiset evaluator;
- reproduce official BFCL scores for one test model output.

### Week 2 — composition and guard

- implement prompt joining;
- generate two- and four-call oracle test sets;
- implement G0–G4 guard ladder;
- audit 100 generated examples manually;
- train or select the seed checkpoint.

### Week 3 — main CSI runs

- generate Direct, Unfiltered, and Guarded pseudo-labels;
- audit hidden-reference pseudo-label precision;
- fine-tune Round 1 and Round 2 models;
- run controlled and official BFCL evaluation.

### Week 4 — ablations and writing

- seed-quality or data-size ablation;
- template- and schema-disjoint evaluation;
- error analysis;
- produce final figures, tables, and release manifest.

---

## 27. Deliverables

- pinned BFCL source manifest;
- scripts to build atomic and compositional splits;
- controlled, natural, and rejected evaluation manifests with source-group metadata;
- deterministic prompt-template registry;
- schema and composition guards;
- official-BFCL evaluation adapter;
- pseudo-label audit report;
- model checkpoints or LoRA adapters;
- one-command reproduction scripts;
- dataset card documenting that BFCL-derived training makes the result unsuitable as an untouched leaderboard submission.

### 27.1 Persistent audit artifact contract

Never discard composed data after training. A run keeps:

```text
data/
  public_candidates/
  oracle/
round_01/
  raw_predictions/
  conditions/<condition>/
    guard_decisions/
    parsed_predictions/
    pseudo_label_audit/
    composed_unique/
    training_materialized/
    evaluation/
round_02/
  ...same condition structure...
round_00/
  conditions/seed_direct/
  conditions/frozen_recursive/
manifest.json
data_checksums.json
checksums.json
summary.json
summary.csv
```

Keep raw predictions, parsed/component guard decisions, all accepted and rejected candidates, false-accept and false-reject audits, the unique selected pseudo-label set, and the exact replay-expanded Trainer input. Each row records candidate and parent IDs, source components, questions, schemas, joined prompt, template, family, condition, round, checkpoint, guard reasons, composed target, training selection, and replay instance. Round-2 data retain the complete 1→2→4 parent hierarchy.

The CLI audit view must sample accepted, rejected, false-accepted, and false-rejected rows and reconstruct a selected training example from its candidate ID. SHA-256 manifests make later notebook inspection and checkpoint-to-data provenance verifiable. Jobs write directly to the persistent run directory and use completion markers so timeout/resubmission does not erase partial artifacts.

---

## 28. Result interpretation

The primary G1 composition result is successful when it:

- clearly outperforms Direct pseudo-labeling at four calls;
- improves the largest reliably solved call count;
- remains better than Frozen recursive composition;
- preserves one-call accuracy;
- shows at least some transfer to original BFCL Parallel or Parallel Multiple prompts.

G4 succeeds as an auxiliary guard ablation only if its quantity–quality tradeoff improves downstream results over G1. Neither condition controls whether the scheduled curriculum runs.

The strongest possible result is not merely a higher average score. It is a clean demonstration that **short, reliable function-call predictions can be composed into supervision that teaches a small model to perform larger parallel call sets directly**.

---

## References

- [Official BFCL repository and evaluator](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [BFCL V1 release description and category counts](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [BFCL dataset card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- [Pinned BFCL data snapshot](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/tree/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda)
- [Coding-task data exploration notebook](coding_task_data_exploration.ipynb)
- Uploaded draft: `ICMLW26_Compositional_Self_Improvement (1).pdf`
