# Research Plan: CommitPack-Config

## Low-compute compositional self-improvement for YAML and JSON configuration editing

**Candidate dataset:** CommitPackFT, restricted to YAML and JSON file changes. This direction is paused as a primary benchmark; BFCL is the current research priority.

**Task:** Given a base configuration document and a natural-language edit request, output an RFC 6902-style JSON Patch.

**Core curriculum:** one independent edit → two edits → four edits. An eight-edit evaluation and interaction-heavy array edits are optional extensions.

**Intended resource regime:** CPU preprocessing plus one small instruction-tuned model with parameter-efficient fine-tuning; no repository build, agent loop, or expensive code execution.

---

## Decision memo — 2026-07-19

**Status:** Pause CommitPack-Config after the atomic-calibration study and focus the next implementation and experimental cycle on BFCL. Do not run the CommitPack compositional self-improvement pipeline in its current prompt formulation.

### What was actually tested

The completed Qwen3.5-4B study was an atomic-only LoRA calibration, not a compositional self-improvement run. It used one-edit gold supervision for training and evaluated the selected atomic model on repository-held-out one-edit examples plus natural 2/4/8-edit diagnostics. Across the full two-task sweep, 42 training cells were run; 21 belonged to CommitPack. The selected CommitPack configuration used 2,000 atomic examples, 150 optimizer steps, learning rate `5e-5`, LoRA rank 16/alpha 32, and effective batch size 16.

The selected model reached 98.5% three-seed mean exact accuracy on the 1,200-example atomic test set, with a worst seed of 98.3%. Without any multi-edit training, it also reached 97.3% on 200 natural two-edit examples and 97.7% on 74 natural four-edit examples. The 100% result on eleven eight-edit examples is descriptive only. These results establish that the current atomic task is learnable, but they do not establish compositional self-improvement.

### Why the current formulation is not a useful main frontier

The current prompt builder verbalizes the hidden structural diff one operation at a time. Each requested line exposes the operation type, full dotted path, and literal target value; multi-edit prompts enumerate those lines and therefore reveal both the decomposition and the required number of outputs. The target is largely a deterministic translation from this oracle-derived representation into JSON Patch syntax.

The retained edits are also unusually separable: they are scalar mapping-key `add`, `remove`, or `replace` operations; evaluated multi-edit examples contain no parent/child path dependencies; paths average fewer than two keys deep; and 72.5% of the two-edit examples modify keys under the same parent. Final-state evaluation appropriately accepts any behaviorally equivalent patch order. In this setting, atomic SFT mostly teaches bare-array formatting, operation serialization, and JSON Pointer escaping. Repeating that local translator several times is not the intended compositional reasoning problem.

This is not repository leakage: the train, validation, and test partitions are repository-disjoint. The shortcut comes from target-derived instruction construction. The current oracle-explicit formulation should therefore be retained only as an easy control for parsers, evaluators, formatting, JSON Pointer handling, and training-system smoke tests.

### Salvage path, if CommitPack is revisited

A CommitPack-v2 pilot should use the original commit subject/message plus the old document as input and keep the structural diff hidden. It must not expose operation count, paths, operation labels, target values absent from the natural message, or an enumerated decomposition. The retained pools have nonempty original subjects with median length around 39 characters and high subject diversity, so a salvage attempt is plausible; however, generic messages such as “Update dependencies” are not sufficiently grounded and must be rejected.

Before another full sweep, require all of the following:

- a human answerability audit of at least 100 message-grounded examples;
- at least 2,000 usable atomic examples, 200 two-edit examples, and preferably 100 four-edit examples after strict grounding filters;
- repository-disjoint splits and source-group-aware uncertainty;
- atomic SFT accuracy of at least 85%, but a post-atomic gap of at least 10–15 points on the powered multi-edit frontier;
- similar frontier behavior across seeds;
- a symbolic/template baseline that does not solve the main task;
- interaction strata beyond independent scalar edits, including arrays, container changes, cross-subtree changes, or dependent operations.

If message grounding leaves too few answerable examples or excessive label ambiguity, drop CommitPack as a primary benchmark rather than reconstructing the oracle decomposition in different words.

### Current priority

BFCL is the active direction. Its atomic-only calibration achieved 91.7% mean one-call test accuracy, while held-out multi-call behavior remained sharply below the atomic result and highly seed-sensitive because two seeds usually emitted only one call. That unresolved atomic-to-compositional gap is a substantially better target for the manual, guarded curriculum study. The remainder of this document is retained as an archival design and possible CommitPack-v2 starting point; this memo overrides any later wording that describes the current oracle-explicit formulation as the primary experiment.

---

## 1. Executive summary

This study converts real YAML and JSON changes from CommitPackFT into a controlled configuration-editing benchmark for compositional self-improvement.

Each source row contains a file before a commit, the file after the commit, a commit subject/message, repository metadata, language, and license. The official dataset card reports 702,062 total samples, including 114,320 YAML and 39,777 JSON samples. We parse the old and new documents, compute a structural patch, and retain a safe JSON-compatible subset.

A one-edit task asks the model to produce a short JSON Patch operation. A two-edit task joins two independent edit instructions over the same base document. Rather than pseudo-labeling that harder task directly, the current model predicts each smaller edit separately. A guard verifies that each predicted patch applies and that the component patches commute without overlapping or shifting one another. The accepted patches are concatenated and used as supervision for the multi-edit request.

The minimum viable benchmark deliberately excludes the hardest YAML features and array-index interactions. This keeps preprocessing deterministic and the verifier cheap while preserving a meaningful real-world artifact: a patch that changes a configuration file without collateral edits.

---

## 2. Research question

> Can guarded composition of small, locally predicted configuration edits teach a small language model to perform larger multi-edit requests more reliably than direct self-training?

The study should answer four subordinate questions:

1. Are model predictions on one-edit tasks accurate enough to produce useful two-edit pseudo-labels?
2. Does dynamic commutativity checking remove the dominant structured composition failures?
3. Does retraining on clean composed patches expand performance to four or eight requested edits?
4. Does training only on independent safe compositions improve any interaction-heavy edits that were rejected by the guard?

---

## 3. Why CommitPackFT fits the framework

CommitPackFT is a filtered collection of GitHub commits whose messages resemble natural-language instructions. Its records expose the exact information needed to derive an edit task:

- `old_file` and `new_file`;
- `old_contents` and `new_contents`;
- `subject` and `message`;
- language;
- repository license;
- repository identifier.

The dataset card reports:

| Split | Samples | Relevance here |
|---|---:|---|
| YAML | 114,320 | Primary source of deployment, CI, application, and tooling configurations |
| JSON | 39,777 | Cleaner parser-compatible source and initial debugging domain |
| Total CommitPackFT | 702,062 | Broader source corpus, not all needed for this study |

The framework mapping is:

- **Atomic input:** one requested structural edit to a local configuration context.
- **Composed input:** several edit clauses over a shared base document.
- **Atomic output:** one short JSON Patch sequence.
- **Output composition:** concatenate patches after canonicalization.
- **Guard:** parsing, patch application, path compatibility, and dynamic commutativity.
- **Frontier:** number of requested edits, document context size, and path depth.

This mirrors the uploaded *Compositional Self-Improvement* draft: larger supervision is built from predictions on smaller in-distribution subproblems, and unsafe compositions are rejected before retraining.

---

## 4. Proposed claims and hypotheses

### H1 — Guarded composition yields high-precision multi-edit pseudo-labels

The combination of local patch validation and commutativity checking will make two- and four-edit pseudo-labels substantially more accurate than direct full-request predictions.

### H2 — Iterative training expands the edit-count frontier

A model trained on guarded two-edit examples will improve on four-edit requests, and a second round will improve further over the seed model and frozen recursive composition.

### H3 — Structured interaction errors are the main failure mode

Unfiltered composition will fail disproportionately on overlapping paths, parent/child edits, and array-index changes, analogous to boundary interactions in the algorithmic experiments.

### H4 — Clean independent edits can transfer partially to rejected interactions

The final model may improve on array or overlapping-path examples even though those examples are excluded from pseudo-label training. This is an empirical question, not an assumed outcome.

### H5 — Improvements are not only template parsing

A simple symbolic instruction parser will not fully solve held-out naturalized prompts or original commit-message evaluation. If it does, the task generation must be redesigned before model training.

---

## 5. Scope

### Minimum viable benchmark

- JSON and JSON-compatible, single-document YAML.
- Mapping-key additions and removals.
- Scalar replacements.
- No comments or formatting preservation.
- No custom YAML tags, anchors, aliases, merge keys, or duplicate keys.
- No arbitrary code execution.
- No repository checkout.
- One, two, and four edits; optional eight-edit evaluation.
- Output as JSON Patch.

### Phase-2 extensions

- append-only array edits;
- stable array-element replacement;
- Kubernetes schema validation;
- original commit subjects as natural prompts;
- interaction-heavy rejected slices;
- larger context windows.

### Out of scope for the first paper result

- exact byte-level YAML rewriting;
- preserving comments and key order;
- multi-file commits;
- semantic repository tests;
- shell execution or deployment;
- agentic editing loops.

---

## 6. Dataset acquisition, licensing, and versioning

1. Download only the CommitPackFT `yaml` and `json` splits from revision `fc56fe33c030c6daa414c2b112c932b8eed085e6`.
2. Record the pinned revision and source-file hashes in the experiment manifest.
3. Save a manifest containing the source ID, repository, commit, file path, language, and license.
4. Exclude entries with `license == "unknown"` from any redistributed derived dataset.
5. Before public release, choose a conservative license allowlist with institutional guidance and preserve source attribution metadata.
6. Split by canonical repository before constructing any examples.
7. Record content hashes to detect duplicate files, forks, and repeated commits.

The source data is large enough that aggressive filtering is acceptable; quality and clear semantics matter more than retaining every row.

### 6.1 Initial bounded-data audit

The exploratory notebook [`coding_task_data_exploration.ipynb`](coding_task_data_exploration.ipynb) inspected the first 500 JSON and first 500 YAML rows using `jsonpatch==1.33` for a preliminary structural diff:

| Language and exploratory filter | One edit | Two edits | Four edits | Eight edits |
|---|---:|---:|---:|---:|
| JSON, all parsed exploratory diffs | 315 | 87 | 11 | 3 |
| JSON, `add`/`remove`/`replace` only | 309 | 87 | 11 | 2 |
| YAML, all parsed exploratory diffs | 256 | 70 | 23 | 5 |
| YAML, `add`/`remove`/`replace` only | 249 | 68 | 22 | 5 |

These are bounded prefix counts, not final yield estimates. They precede strict JSON-compatible YAML checks, repository splitting, license filtering, context limits, instruction auditing, static conflict checks, and dynamic commutativity. In particular, the seven observed MVP eight-edit rows are not enough to assume that a balanced held-out eight-edit evaluation will survive the complete pipeline.

### 6.2 Atomic-supervision calibration

Before composing edits, build and audit the scalar one-edit bank and run an independent Qwen3.5-4B LoRA calibration through `self.experiments.coding_atomic_sweep`. The pinned source contains 39,777 JSON and 114,320 YAML rows. The production filter accepts scalar `add`, `remove`, and `replace` operations only, retains at most one atomic example per source file row, uses deterministic repository-disjoint 80/10/10 partitions, and requires at least 10,000 deduplicated candidates plus 700 candidates in every language-by-operation stratum.

The completed production audit scanned all 154,097 source rows and retained 39,661 deduplicated atomic candidates: 32,056 in the repository-train partition, 3,874 in validation, and 3,731 in test. Global candidate counts were 4,987/1,789/20,036 JSON add/remove/replace examples and 3,804/1,043/8,002 YAML add/remove/replace examples. Thus every global stratum clears the gate, but the rarity of removals is real rather than a prefix-sampling artifact.

Because removals are materially rarer than replacements in the observed data, evaluation is balanced across the six language-by-operation strata as far as each repository-held-out partition permits, with a hard minimum of 50 examples per stratum rather than an artificial equal-count requirement. The calibration uses nested training subsets of 250, 500, 1,000, and 2,000 examples; learning rates `1e-5`, `5e-5`, and `2e-4`; and 50, 150, and 450 optimizer steps. It runs nine full-data cells, six data-size cells using the two best schedules, and six replications of the best three configurations, for 21 LoRA runs.

The selected 2,000-example train bank is balanced to 333–334 examples per stratum. Validation contains 99–101 examples per stratum. Test contains 97 YAML removes, 162 JSON removes, and 235–236 examples in the other four strata, for 1,200 examples total. Prompt-plus-target lengths have median 309 tokens and 95th percentile about 500 tokens; all selected examples fit the 1,024-token limit, with a maximum of 964.

Selection uses behavioral final-state accuracy on a fixed 600-example validation set. The fixed 1,200-example test set is opened only after selection. Atomic supervision is considered reliable when three-seed mean test accuracy is at least 85%, no seed is below 80%, valid/applicable patches are at least 98%, and the train-test gap is at most ten points.

The executed zero-shot baseline is already strong on atomic edits: 84.5% exact and 99.5% format validity on validation, and 86.6% exact and 99.2% format validity on test. Its powered natural two-edit frontier is lower at 77.5% exact; the 74-example four-edit slice is 77.0%, while the 10/11 eight-edit result remains descriptive because that cell is tiny. A one-update LoRA smoke run on 250 examples produced 84.8% validation exact and 99.7% format validity with no OOM at microbatch 4. The residual failures are dominated by unescaped JSON Pointer path segments and inapplicable paths, so the sweep should be interpreted primarily as a search for cheap, stable pointer/protocol improvement rather than basic task acquisition; the zero-shot baseline must remain in every comparison.

The strict natural held-out builder yielded 200 two-edit, 74 four-edit, and 11 eight-edit examples at the 2,048-token evaluation limit. Consequently, two edits are the powered secondary frontier for this first study. Four edits are reported as a smaller exploratory natural slice, and eight edits are descriptive only. A future controlled four-edit set may sample safe four-operation subsets from larger held-out source diffs, but it must retain source-group IDs and source-aware uncertainty rather than treating correlated subsets as independent examples.

---

## 7. Task definition

### Input

```text
Configuration document:
<normalized YAML or JSON context>

Requested changes:
1. Set the replicas field under spec to 3.
2. Change the image field under the first container to nginx:1.27.

Return only a JSON Patch array.
```

### Output

```json
[
  {"op": "replace", "path": "/spec/replicas", "value": 3},
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/image",
    "value": "nginx:1.27"
  }
]
```

### Behavioral correctness

The primary criterion is not exact patch-string match. Apply the predicted patch to the parsed old document and require the resulting document to equal the intended new state for the requested subset of edits, with no collateral changes.

Every derived example also records:

```json
{
  "component_count": 2,
  "source_component_ids": ["<commit>#<operation-1>", "<commit>#<operation-2>"],
  "source_group_id": "<canonical-repository>@<commit>",
  "evaluation_track": "controlled",
  "interaction_slice": "safe_independent"
}
```

Use `evaluation_track` values `controlled`, `natural`, and `rejected`. Use named `interaction_slice` values for safe compositions and each rejected interaction family.

---

## 8. Parsing and normalization pipeline

### 8.1 File-level filters

Retain a row only when:

- `old_file` and `new_file` refer to the same path, or the rename can be safely ignored;
- both old and new contents parse;
- the YAML input contains exactly one document;
- parsing produces only JSON-compatible mappings, arrays, strings, numbers, booleans, and nulls;
- there are no custom tags, unresolved aliases, merge keys, or duplicate mapping keys;
- serialized normalized documents fit an initial size ceiling;
- the old and new documents are not identical after normalization.

### 8.2 Canonical representation

Convert both JSON and YAML to a canonical in-memory JSON tree. Preserve source language as metadata but evaluate semantics on the normalized tree.

Recommended canonicalization:

- preserve mapping keys exactly as strings;
- preserve array order;
- normalize numeric types cautiously;
- reject non-finite floats;
- distinguish missing keys from keys with null values;
- use RFC 6901 JSON Pointer escaping for paths.

### 8.3 Initial context-size ceiling

For the first experiment, keep examples whose relevant context fits within approximately 768–1,024 model tokens. Do not include a huge full file merely because the changed path is local.

---

## 9. Structural diff extraction

Compute a deterministic structural diff between normalized old and new trees.

The notebook's `jsonpatch==1.33` output is only an exploratory preview. Before freezing the dataset, pin the production diff implementation and version, verify deterministic output on a fixed fixture corpus, and retain final-state equality as the authoritative correctness criterion.

### 9.1 Allowed operations in the MVP

- `replace` a scalar value;
- `add` a mapping key with a scalar or small JSON-compatible value;
- `remove` a mapping key;
- optionally replace a small complete subtree when it remains below a fixed size.

### 9.2 Initially excluded operations

- arbitrary array insertion or deletion;
- move and copy;
- test operations;
- replacement of a very large subtree;
- operations that change a node's container type and also require descendant edits;
- diffs whose result depends strongly on a library-specific minimal-patch heuristic.

### 9.3 Patch canonicalization

Store each atomic operation as:

```json
{
  "op": "replace",
  "path": "/spec/replicas",
  "value": 3,
  "old_value": 1,
  "source_commit": "...",
  "source_repo": "..."
}
```

`old_value` is evaluation metadata and is never part of the model target.

---

## 10. Building an atomic edit bank

A source commit containing several structural edits can yield several atomic tasks over the same old document.

For every extracted operation:

1. Apply it independently to the old document.
2. Confirm that it succeeds.
3. Confirm that it changes only the intended path or subtree.
4. Generate one or more deterministic instruction variants.
5. Store the source commit's full reference patch separately for auditing.

### 10.1 Instruction generation

Use controlled but naturalized templates.

For scalar replacement:

```text
Set the `replicas` field under `spec` to 3.
```

```text
Update `spec.replicas` so that its value is 3.
```

For mapping-key addition:

```text
Add a `timeout` field under `service` with value 30.
```

For removal:

```text
Remove the `deprecated` field from `features`.
```

Do not expose the exact JSON Pointer in every prompt. Naturalize paths through parent and leaf names so the model must ground the instruction in the document.

### 10.2 Ambiguity bands

Annotate automatically:

- **unique leaf:** the named leaf key appears once in the context;
- **repeated leaf:** the leaf key occurs in several locations and the parent context is needed;
- **deep path:** path depth exceeds a threshold;
- **large value:** inserted or replaced value is a list or object rather than a scalar.

Use the unique-leaf, scalar subset for the initial seed. Reserve more ambiguous bands for evaluation and later rounds.

---

## 11. Context extraction

Full files may be unnecessarily long. Build a local context for each task.

### 11.1 Atomic context

For a one-edit task, show:

- the smallest subtree that contains the target path;
- a configurable number of siblings at each ancestor;
- the absolute root prefix for that subtree.

Example:

```text
Context rooted at /spec/template/spec:
containers:
  - name: web
    image: nginx:1.26
```

The patch path remains absolute.

### 11.2 Composite context

For several edits, show the smallest common context containing all requested paths, again with bounded siblings. This makes the hard instance larger than each local subproblem while keeping total length controlled.

### 11.3 No-information-loss check

A deterministic checker must confirm that every target path and every value needed to interpret the instruction is present in the displayed context.

---

## 12. Safe composition from real commits

The cleanest source of multi-edit tasks is a single CommitPackFT row whose structural diff contains several operations on the same old document.

### 12.1 Candidate subsets

From a commit with operations \(p_1,\ldots,p_k\), sample subsets of size 2, 4, and optionally 8.

### 12.2 Input composition

Join the corresponding atomic instruction clauses in a randomized numbered list. The composite input uses the broader shared document context.

### 12.3 Output composition

Concatenate the predicted JSON Patch lists in instruction order, then canonicalize.

### 12.4 Why same-commit subsets are preferable

- every operation is grounded in a real old document;
- all intended changes co-occurred in an actual commit;
- no artificial grafting between unrelated repositories is required;
- a hidden oracle final state can be constructed exactly;
- path interactions can be measured rather than assumed.

---

## 13. Composition guard

The guard must use only the base document, displayed task structure, schemas, and predicted patches. It must not compare predictions with the hidden reference diff during pseudo-label generation.

### 13.1 Component patch guard

For each predicted component patch:

1. Parse as JSON.
2. Require a list with the expected maximum number of operations.
3. Restrict operations to the allowed set.
4. Validate every JSON Pointer.
5. Require `value` exactly when the operation requires it.
6. Reject extraneous fields unless explicitly allowed.
7. Apply the patch to a fresh copy of the base document.
8. Reject failed application, invalid container access, or non-JSON-compatible results.
9. Require the changed-path set to remain local to the component's displayed context.
10. Reject output longer than a fixed ceiling.

The locality check is not a reference-answer check: it only prevents a component from modifying unrelated parts of the document.

### 13.2 Static cross-component guard

For every pair of component patches:

- reject identical paths;
- reject ancestor/descendant path relationships;
- reject removal of a parent used by another patch;
- reject incompatible operations under the same array;
- reject duplicate additions of the same key;
- reject conflicting replacements.

### 13.3 Dynamic commutativity guard

For every pair \(P_i,P_j\):

1. apply \(P_i\) then \(P_j\) to the base document;
2. apply \(P_j\) then \(P_i\) to the base document;
3. require both orders to succeed;
4. require the resulting normalized documents to be equal.

For two or four components, optionally test all permutations. For eight components, use pairwise commutativity plus one or more randomized application orders.

Dynamic commutativity is the key analogue of checking that no boundary interaction invalidates the composition rule.

### 13.4 Result guard

After concatenation:

- apply the full patch to the base document;
- require successful parsing and serialization;
- require no duplicate JSON object keys;
- enforce the context and output-size limits;
- optionally validate against a domain schema.

### 13.5 Guard ladder

| Level | Checks |
|---|---|
| G0 | Concatenate every parseable output |
| G1 | JSON syntax and allowed operation fields |
| G2 | Individual patch application |
| G3 | Static path conflict checks |
| G4 | Dynamic pairwise commutativity |
| G5 | Optional domain schema validation |

---

## 14. Roundwise CSI protocol

Use binary composition at each round.

### Round 0 — one-edit seed

Train or select a checkpoint that is reliable on local one-edit tasks.

### Round 1 — two edits

For each safe two-operation source subset:

1. present each edit separately with its local context;
2. obtain two predicted patches from the seed model;
3. apply G1–G4;
4. concatenate accepted patches;
5. pair the pseudo-label with the broader two-edit input;
6. fine-tune on accepted examples plus one-edit replay.

### Round 2 — four edits

For each safe four-operation subset:

1. partition it into two two-edit subproblems;
2. query the Round-1 model on each subproblem;
3. guard each predicted patch list;
4. check cross-group commutativity;
5. concatenate into a four-edit pseudo-label;
6. fine-tune with replay from prior frontiers.

### Optional eight-edit evaluation

Evaluate the Round-2 model directly on eight-edit requests and compare it with recursive composition of two four-edit predictions. Add a third training round only when enough safe source commits remain.

---

## 15. Hidden reference labels and audit discipline

The structural diff extracted from `old_contents` and `new_contents` provides an oracle patch and oracle final document. Store these in an audit-only table unavailable to pseudo-label generation.

Use hidden references only to measure:

- pseudo-label final-state accuracy;
- operation precision and recall;
- collateral-change rate;
- false accepts and false rejects of the guard;
- error correlation across components;
- direct versus composed pseudo-label quality.

This is analogous to a controlled benchmark simulation: the existing commit provides ground truth for scientific evaluation, while the self-improvement procedure operates as though frontier labels were unavailable.

---

## 16. Data splits and leakage prevention

### 16.1 Repository-disjoint split

Split by canonical repository before extracting atomic operations or sampling any multi-edit subsets. All commits and forks linked to the same canonical repository remain in one split.

Suggested proportions:

- 80% repositories for seed and pseudo-label generation;
- 10% validation repositories;
- 10% test repositories.

### 16.2 Content deduplication

Deduplicate by hashes of:

- normalized old document;
- normalized new document;
- `(old_document, operation_set)`;
- instruction text after normalization.

### 16.3 Template split

Reserve some instruction templates for evaluation only.

### 16.4 Path-vocabulary split

Report both:

- seen key/path vocabulary; and
- held-out leaf-key vocabulary where possible.

The headline claim should be about edit-count expansion, so the main test may reuse common configuration concepts while excluding exact documents and commits.

Within the held-out repositories, permit at most one controlled evaluation example per `(canonical repository, commit, edit count)`. This prevents a large commit with many possible subsets from dominating a count cell.

---

## 17. Controlled and natural evaluation tracks

### 17.1 Controlled semi-synthetic frontier — primary

Construct safe one-, two-, and four-edit tasks from complete commits in held-out repositories. Select operation subsets only after the repository split, use deterministic naturalized instructions with held-out templates, and preserve the source commit's hidden final state for exact behavioral evaluation.

Target 200 JSON and 200 YAML examples at each of one, two, and four edits. Include a balanced eight-edit cell only when at least 100 distinct held-out source commits per language survive every parsing, context, license, locality, and commutativity filter. Otherwise report all available eight-edit examples as exploratory and define the reliable frontier only through four edits.

Balance the core cells by edit count and language. Annotate operation type, value type, path depth, context length, and path topology without constructing their full factorial cross-product.

### 17.2 Original commit-subject track — natural secondary

For commits with short, informative subjects and small structural diffs, use the original `subject` as the request and the complete extracted diff as reference. Stratify results by derived edit count but do not force artificial balance.

Because commit subjects are often incomplete, apply aggressive automatic filtering and manually audit a fixed evaluation sample. Report this track separately; do not let noisy subjects obscure the controlled CSI result.

### 17.3 Human paraphrase audit — optional

Paraphrase 100–200 controlled test instructions manually or through a small one-time annotation effort. This is useful only after the core experiment succeeds.

### 17.4 Guard-rejected interactions — diagnostic

Use held-out source commits to construct named slices for overlapping paths, parent/child edits, competing keys, array-index shifts, non-commutative orders, type changes, and supported advanced-YAML cases. Never train on these rows. Report each slice separately because some introduce new operation types in addition to interaction structure.

All three evaluation tracks report exact example counts and distinct `source_group_id` counts. Compute 95% confidence intervals by bootstrapping source commits rather than treating multiple derived subsets as independent observations.

---

## 18. Preventing a trivial template-parsing benchmark

A central risk is that deterministic instructions expose the path and value so directly that a rule-based parser solves the task.

### Required symbolic baseline

Implement a simple parser that:

- extracts quoted key names;
- recognizes verbs such as set, add, and remove;
- extracts literal values;
- searches the document for matching paths;
- emits a JSON Patch when the match is unique.

### Go/no-go rule

If this parser exceeds 90% final-state accuracy on four-edit held-out tasks, redesign the language generation before running the full model experiment.

Possible redesigns:

- omit exact dot paths and use parent/leaf descriptions;
- include repeated leaf keys requiring context;
- vary syntax and word order;
- describe values through local relations, such as “match the timeout used by the worker service” only when the referenced value is present in context;
- use original commit subjects for a larger share of evaluation;
- add a small manually paraphrased test set.

The goal is not to defeat all symbolic systems. It is to ensure that the measured effect is not merely string-to-pointer transduction.

---

## 19. Baselines

### B0 — Seed only

One-edit training only.

### B1 — Direct pseudo-labeling

Ask the current model to predict the complete two- or four-edit patch in one generation. Apply the same output-level syntax and patch-application guard available to the proposed method.

### B2 — Unfiltered composition

Predict components separately and concatenate all parseable patches without path-conflict or commutativity filtering.

### B3 — Guarded composition

The proposed method, using G1–G4.

### B4 — Frozen recursive composition

Use the seed model recursively on one-edit components but never fine-tune.

### B5 — Oracle-composed SFT upper bound

Train on the exact structural operations extracted from the hidden source diffs.

### B6 — Symbolic instruction parser

The required triviality check.

### B7 — Copy/full-document generation, optional

Ask the model to emit the entire edited YAML/JSON document rather than a patch. This tests whether the short patch representation is an important enabling choice.

### Fairness controls

Match:

- accepted training examples;
- total generated tokens;
- SFT tokens;
- guard strength available to Direct and Composition conditions.

---

## 20. Model and training protocol

### 20.1 Model regime

Use one instruction-tuned model in the 0.5B–1.5B range with reasonable JSON-generation ability.

### 20.2 Input and output limits

Recommended starting values:

- input context: 768–1,024 tokens;
- output limit: 128 tokens for one/two edits and 256 for four edits;
- document context selected locally rather than showing full files;
- greedy decoding during pseudo-label generation.

### 20.3 Parameter-efficient fine-tuning

- LoRA or 4-bit QLoRA;
- one epoch per round;
- modest rank, fixed across conditions;
- no reinforcement learning;
- no sampling-based search in the primary result.

### 20.4 Replay mixture

At each round, use approximately:

- 30–40% one-edit seed replay;
- 20–30% prior-frontier replay;
- 40–50% new accepted frontier examples.

### 20.5 Suggested data scale

| Stage | Target accepted examples |
|---|---:|
| One-edit seed | 1,000–2,000 |
| Round 1: two edits | 1,000–2,000 |
| Round 2: four edits | 1,000–2,000 |
| Optional eight-edit round | 500–1,000 |

The raw YAML and JSON pools are much larger than needed. Start small and audit quality.

---

## 21. Evaluation grid

The fixed core grid crosses only:

- edit count: 1, 2, and 4, with 8 conditional on the distinct-source gate above;
- file type: JSON versus YAML;

Record the following as diagnostic annotations or dedicated one-factor slices rather than a full cross-product:

- operation type: add, remove, replace, mixed;
- value type: scalar versus small list/object;
- path depth;
- document-context length;
- unique versus repeated leaf keys;
- cross-subtree versus same-parent edits;
- seen versus held-out prompt templates;
- repository/domain category within the held-out split;
- safe independent versus guard-rejected interaction cases.

The default controlled target is 200 examples per `(file type, edit count)` cell for one, two, and four edits, with at most one row per source commit and count. Publish the exact retained count whenever a cell falls short.

---

## 22. Primary metrics

### Behavioral metrics

- **Final-state exact match:** predicted patch applied to old document equals the intended document for the requested edit subset.
- **No-collateral-change accuracy:** all and only requested paths changed.
- **Patch application rate:** predicted patch applies without error.
- **Controlled frontier accuracy by edit count:** final-state exact match on the balanced semi-synthetic grid.

### Structural metrics

- valid JSON rate;
- allowed-operation rate;
- JSON Pointer validity;
- operation precision, recall, and F1 after canonicalization;
- exact operation-set match;
- correct edit-count rate;
- path accuracy;
- value accuracy.

### Self-improvement metrics

- pseudo-label final-state precision before and after guarding;
- guard acceptance rate;
- guard false-accept and false-reject rates;
- frontier accuracy by number of edits;
- one-edit retention after each round;
- examples and distinct source commits per evaluation cell;
- source-commit bootstrap 95% confidence intervals;
- generated tokens, SFT tokens, and wall-clock cost.

### Naturalness metrics

- performance on held-out templates;
- performance on original commit subjects;
- performance on the optional human-paraphrase set.

---

## 23. Filtered-out interaction slices

Build a separate diagnostic test set from source commits rejected by the safe-composition guard.

Candidate slices include:

- edits to the same array with index shifts;
- one edit removes a parent of another target;
- parent replacement plus descendant modification;
- two edits compete for the same key;
- type-changing edits;
- non-commutative patch orders;
- changes involving YAML anchors or merge keys, if a robust oracle representation is available.

These examples are never used for composed pseudo-label training. Evaluate whether the final model improves on them anyway, mirroring the paper's filtered-boundary analysis.

Interpretation must be cautious: array and advanced-YAML slices may also introduce new operation types, not only interactions. Report each slice separately.

---

## 24. Main experimental matrix

| Condition | Round 1 | Round 2 | Evaluation |
|---|---|---|---|
| Seed only | — | — | 1/2/4/8 edits |
| Direct | Direct two-edit pseudo-labels | Direct four-edit pseudo-labels | Controlled + natural tracks |
| Unfiltered composition | Concatenated atomic predictions | Concatenated two-edit predictions | Controlled + interaction slices |
| Guarded composition | G1–G4 accepted labels | G1–G4 accepted labels | Controlled + interaction slices |
| Frozen composition | No SFT | No SFT | Recursive patch composition |
| Oracle upper bound | Exact extracted two-edit labels | Exact extracted four-edit labels | Controlled + natural tracks |
| Symbolic parser | No SFT | No SFT | Controlled track |

For a minimal budget, run one random seed for all conditions and add two more seeds only for Seed, Direct, and Guarded Composition.

---

## 25. Priority ablations

1. **Guard ladder:** G0 through G4.
2. **Static path checks versus dynamic commutativity.**
3. **Retained examples per round:** 250, 1,000, 2,000.
4. **Seed quality:** low and high seed checkpoints.
5. **Replay:** none versus default replay mixture.
6. **Context representation:** full normalized document versus local context.
7. **Operation scope:** scalar-only versus small structured values.
8. **Language:** JSON-only versus JSON+YAML.
9. **Instruction naturalness:** explicit paths versus naturalized parent/leaf descriptions.
10. **Patch output versus full edited document.**

Avoid a full factorial design; use one-factor-at-a-time ablations around the default Guarded condition.

---

## 26. Error taxonomy

Manually inspect a fixed sample and label:

- invalid JSON;
- malformed JSON Patch object;
- invalid pointer escaping;
- wrong operation type;
- wrong path;
- right leaf but wrong repeated occurrence;
- wrong value or value type;
- omitted requested edit;
- duplicate edit;
- collateral edit;
- parent/child conflict;
- array-index shift;
- patch applies but produces wrong final state;
- instruction ambiguity;
- source diff or parser artifact.

The most important compositional failure is **individually plausible patches whose union is invalid or changes a different final state because the operations interact**.

---

## 27. Low-compute execution plan

### Minimum viable run

- stream only YAML and JSON source splits;
- sample and preprocess until at least 10,000 high-quality atomic operations are available;
- use 1,000 one-edit seed examples;
- retain 1,000 pseudo-labeled examples per round;
- run two rounds;
- use local contexts capped near 1,024 tokens;
- use one small model with LoRA/QLoRA;
- run Direct, Unfiltered, and Guarded learned conditions;
- evaluate on fixed 1/2/4/8-edit grids;
- run one random seed first.

### Expanded run after a positive pilot

- 2,000 examples per round;
- two additional seeds for the main comparison;
- JSON+YAML transfer analysis;
- original commit-subject evaluation;
- array-interaction slice;
- optional schema validation for Kubernetes-like manifests.

Most preprocessing and verification is deterministic CPU work. Model generation remains short and static.

---

## 28. Go/no-go pilot criteria

Proceed only when:

1. preprocessing yields at least 10,000 valid atomic edits and 2,000 safe two-edit groups across repository-disjoint splits;
2. manual audit finds at least 95% of controlled instructions unambiguous;
3. the symbolic parser remains below 90% four-edit final-state accuracy;
4. the seed model achieves at least 85% on one-edit tasks;
5. the seed model is at least 15 points worse on four-edit tasks, creating a meaningful frontier;
6. guarded two-edit pseudo-label precision is at least 95%, or at least 10 points above direct pseudo-label precision;
7. the G1–G4 guard retains at least 25–30% of candidate groups;
8. one self-improvement round improves two-edit performance without losing more than two points on one-edit tasks;
9. the held-out repositories yield 200 distinct source commits per language at each of one, two, and four edits after every filter, with no repository overlap with training.

If safe multi-operation commits are too rare, reduce the target to 1→2 edits and use four edits only for evaluation rather than grafting unrelated operations. Failure to reach 100 distinct held-out eight-edit source commits per language does not block the main experiment; it makes the eight-edit cell exploratory and removes it from the reliable-frontier claim.

---

## 29. Risks and mitigations

### Risk: YAML semantics are lost during normalization

**Mitigation:** define the benchmark explicitly over JSON-compatible YAML semantics; reject anchors, custom tags, merge keys, duplicate keys, and multi-document files.

### Risk: structural diff output depends on the library

**Mitigation:** treat the notebook's `jsonpatch==1.33` output as exploratory, pin the production diff implementation, publish deterministic fixture and canonicalization tests, and evaluate final document state rather than exact patch sequence.

### Risk: templated language makes the task trivial

**Mitigation:** require a symbolic baseline, hold out templates, add repeated-key cases, and include a secondary original-commit-subject evaluation.

### Risk: existing labels make self-improvement seem unnecessary

**Mitigation:** separate hidden oracle labels from generation and frame the dataset as a controlled simulation of unavailable frontier supervision. Include the oracle upper bound to quantify remaining pseudo-label noise.

### Risk: source leakage across repositories or forks

**Mitigation:** repository-disjoint split plus normalized-content hashing.

### Risk: strong filtering leaves too few multi-edit groups

**Mitigation:** start with two edits, allow small subtree replacements, and add four-edit training only after measuring yield.

### Risk: direct pseudo-labeling receives a weaker verifier

**Mitigation:** apply every output-level check available to both methods and isolate the extra benefit of component-wise checking and commutativity.

### Risk: license uncertainty

**Mitigation:** exclude unknown licenses, retain attribution metadata, use a conservative release allowlist, and obtain institutional review before redistributing derived contents.

---

## 30. Expected paper figures and tables

1. **Accuracy heatmap:** edit count × self-improvement round.
2. **Pseudo-label quality:** Direct, Unfiltered, and Guarded final-state precision.
3. **Guard tradeoff:** accepted sample count versus false-accept rate.
4. **Interaction plot:** safe independent versus rejected non-commutative edits.
5. **JSON/YAML transfer table.**
6. **Operation-type table:** add, remove, replace, mixed.
7. **Naturalness table:** seen templates, held-out templates, and original commit subjects stratified by derived edit count.
8. **Symbolic-baseline comparison.**
9. **Compute table:** preprocessing time, generation tokens, SFT tokens, and wall clock.

---

## 31. Five-week implementation schedule

### Week 1 — parser and source audit

- pin CommitPackFT revision;
- stream YAML and JSON samples;
- reproduce the bounded audit and measure full-scan post-filter yields by distinct source commit;
- implement strict JSON-compatible YAML parsing;
- profile parse rate, document size, licenses, and diff size;
- manually inspect 100 retained and 100 rejected rows.

### Week 2 — diff, task generation, and evaluator

- implement deterministic structural diff;
- create atomic operations and local contexts;
- implement instruction templates;
- implement patch application and final-state evaluator;
- implement repository-disjoint splits and deduplication.

### Week 3 — composition and pilot

- identify safe two- and four-operation subsets;
- implement G0–G4 guard ladder and commutativity tests;
- train/select the one-edit seed model;
- run the symbolic baseline and go/no-go audit.

### Week 4 — main CSI runs

- generate Direct, Unfiltered, and Guarded pseudo-label sets;
- audit against hidden oracle final states;
- fine-tune Round 1 and Round 2;
- evaluate all controlled test bands.

### Week 5 — ablations and natural evaluation

- guard and data-size ablations;
- original commit-subject subset;
- rejected interaction slices;
- error analysis, figures, and release documentation.

---

## 32. Deliverables

- pinned CommitPackFT source manifest;
- strict YAML/JSON parser and rejection log;
- deterministic structural-diff implementation;
- atomic and compositional task builder;
- local-context extractor;
- JSON Patch parser, applier, and final-state evaluator;
- static conflict and dynamic commutativity guards;
- symbolic baseline;
- repository-disjoint split files;
- controlled, natural, and rejected evaluation manifests with source-group metadata;
- pseudo-label audit report;
- LoRA adapters and reproducible training commands;
- derived-dataset card with licensing and semantic-normalization limitations.

---

## 33. Decision rule

The experiment is successful when Guarded Composition:

- yields more accurate two- and four-edit pseudo-labels than Direct prediction;
- advances the largest edit count solved reliably;
- outperforms Unfiltered and Frozen Composition;
- preserves one-edit performance;
- shows robustness to held-out instruction templates;
- produces valid, behaviorally correct patches rather than merely well-formed JSON;
- remains nontrivial relative to the symbolic parser.

Apply the reliable-frontier claim to the controlled track. Include eight edits in that claim only when the distinct-source gate is satisfied; otherwise report eight-edit results descriptively.

The strongest paper-level result would show that **real configuration changes can be decomposed into local edits, safely recombined through executable patch semantics, and used to teach a small model to perform larger edits directly—without an agent loop or large inference budget**.

---

## References

- [CommitPackFT dataset card](https://huggingface.co/datasets/bigcode/commitpackft)
- [Pinned CommitPackFT data snapshot](https://huggingface.co/datasets/bigcode/commitpackft/tree/fc56fe33c030c6daa414c2b112c932b8eed085e6)
- [OctoPack paper](https://arxiv.org/abs/2308.07124)
- [OctoPack data-construction repository](https://github.com/bigcode-project/octopack)
- [RFC 6902: JSON Patch](https://www.rfc-editor.org/rfc/rfc6902)
- [RFC 6901: JSON Pointer](https://www.rfc-editor.org/rfc/rfc6901)
- [Coding-task data exploration notebook](coding_task_data_exploration.ipynb)
- Uploaded draft: `ICMLW26_Compositional_Self_Improvement (1).pdf`
