# Workshop Plan: Compositional Self-Improvement as a Proof of Concept

## Scope

This plan is for the workshop version only.

In scope:

- fixed-capacity models
- self-improvement through compositional pseudo-labeling
- simple algorithmic tasks with controllable decomposition
- clear ablations and failure cases

Out of scope:

- meta-self-improvement
- model growth or capacity expansion
- adaptive architecture changes
- broad task suites with weak mechanistic grounding

## Core Question

Can a fixed model reliably improve on longer-horizon tasks by composing its own predictions on shorter subproblems, then training on those composed pseudo-labels?

The workshop version should answer this question with the smallest convincing set of tasks and controls.

## Main Thesis

Compositional self-improvement works when all three conditions hold:

1. The task admits a faithful decomposition into shorter subproblems.
2. The model is already competent on those shorter subproblems.
3. The composition operator preserves enough label quality to bootstrap longer-range training.

The paper should also show where this breaks:

1. when decomposition is incorrect,
2. when boundary interactions introduce systematic errors,
3. when pseudo-label quality falls below a usable threshold.

## Workshop Story

The simplest strong paper narrative is:

1. Reproduce the addition result as the anchor task.
2. Show the same idea on run-length, where composition is algebraically clean and associative.
3. Extend to multiplication, where composition is richer and noisier.
4. Demonstrate that success depends on correct composition rather than generic self-training.

This gives one clean task, two easy transfer tasks, and one ambitious stress test.

## Hypotheses

### H1: Composition beats naive self-training

For tasks with a correct decomposition rule, composed pseudo-labels should outperform direct self-labeling on long examples.

### H2: Algebraically clean tasks improve earlier and more reliably

Run-length should be easier than addition because its composition rule is exact and carry-free.

### H3: Boundary interactions determine the difficulty frontier

Addition and multiplication should show stronger gains on composition-safe slices than on boundary-interaction slices.

### H4: Multiplication needs staged composition

Unrestricted multiplication is probably too hard as a first workshop experiment. A staged version should work better:

- first structured partial products,
- then controlled carries,
- then full composition if the earlier stages succeed.

## Experimental Program

## Phase 1: Addition Reproduction

Goal:

- lock down the current self-improvement pipeline on `Qwen3-0.6B`
- regenerate the key figure cleanly
- verify the role of composition filtering and boundary-carry evaluation

Minimal deliverables:

- main self-improvement curve by round or max digits
- stitched no-boundary vs boundary-carry accuracy
- comparison of `with_carry` vs `with_carry_filtered`

Key control:

- base-only training without self-improvement

Why this matters:

- it establishes the benchmark task
- it validates the current infrastructure before adding new tasks

## Phase 2: Run-Length as a Clean Transfer Task

### Task Choice

Use bit-string summaries with explicit associative composition.

Recommended formulations:

- run-length:
  - input: a binary string of length `n`
  - target: `(max_run, prefix_run, suffix_run)`
  - composition: merge adjacent summaries using the exact boundary formula

### Why these are good next tasks

- the composition rules are exact
- there are no carry interactions
- decomposition is easy to formalize
- failures are easier to interpret than in multiplication

### Transfer Training Setup

Seed regime:

- train on short strings only
- evaluate on much longer strings

Pseudo-label generation:

- generate long strings by concatenating shorter strings
- ask the model for each component summary
- combine component predictions with the exact task-specific operator
- train on the composed pseudo-labels

Important baselines:

- supervised short-only
- naive self-training on direct long predictions
- compositional pseudo-labeling
- corrupted-composition control where one component summary is corrupted before composition

Expected result:

- run-length should provide the clean non-arithmetic proof of concept with a richer associative summary and no arithmetic carries

## Phase 3: Multiplication as the Stress Test

### Recommendation

Do not start with unrestricted decimal multiplication as the first multiplication experiment.

Start with a staged program:

1. single-digit by multi-digit multiplication
2. block multiplication with explicit partial products
3. controlled multi-block multiplication with carry-aware aggregation
4. full multiplication only if the first three stages look promising

### Why multiplication is hard

- composition creates overlapping partial products
- aggregation requires addition of shifted terms
- carries propagate across block boundaries
- pseudo-label errors can cascade more severely than in addition

### Candidate Decomposition

For chunked operands:

`A = sum_i 10^{s_i} A_i`

`B = sum_j 10^{t_j} B_j`

Then:

`A * B = sum_{i,j} 10^{s_i + t_j} (A_i * B_j)`

This suggests a natural compositional pipeline:

1. solve short block products,
2. form shifted partial products,
3. compose them into a final answer.

### Practical multiplication path

#### Stage M1: Single-digit x multi-digit

This isolates one source of carry while staying close to familiar arithmetic.

Success criterion:

- self-improvement extends to longer multiplicands than direct supervision alone

#### Stage M2: Blocked multiplication with exact composition

Use the model only for short block products.
Use exact code to sum shifted partial products into the final label.

Purpose:

- test whether short-product competence transfers at all
- avoid conflating block-product failure with final carry aggregation failure

This is still a valid workshop result because the self-improvement signal is compositional even if the final block-composition rule is exact.

#### Stage M3: Not required for workshop

We should not treat fully model-generated aggregation as a workshop requirement.

For the workshop version, blocked multiplication with model-predicted block
products plus exact aggregation is enough. The core question is whether local
competence on block products transfers through compositional pseudo-labeling.

If model-generated aggregation ever becomes relevant, it should be framed as
future work rather than a missing workshop component.

#### Stage M4: Carry-sensitive evaluation slices

As in addition, split examples into easier and harder subsets:

- low-overlap partial-product cases
- high-overlap partial-product cases
- low-carry aggregation cases
- high-carry aggregation cases

Expected result:

- gains should appear first on the easier slices

## Composition Design Principles Across Tasks

We should keep the workshop framing centered on one abstract recipe:

1. learn short problems,
2. compose short predictions into long pseudo-labels,
3. train on those pseudo-labels,
4. expand the solvable length range,
5. measure where composition succeeds and fails.

Every task should instantiate the same template with task-specific operators.

### Desired task properties

- exact or near-exact decomposition
- controllable difficulty by length
- interpretable boundary failures
- clear weak-to-strong generalization gap

### Undesired task properties

- trivial shortcuts
- decomposition that is only vaguely defined
- tasks where evaluation is easy but composition is not faithful
- tasks with uncontrolled natural-language ambiguity

## Required Baselines and Ablations

Every task should include:

1. supervised short-only baseline
2. direct self-training on long examples without composition
3. compositional self-improvement
4. corrupted-composition control
5. oracle-composition upper bound when feasible

For addition and multiplication, also include slice-based analysis:

1. boundary-safe
2. boundary-interaction
3. carry-light
4. carry-heavy

## Evaluation

Primary metrics:

- exact-match accuracy
- accuracy by length bucket
- best solved length at a fixed threshold

Secondary metrics:

- pseudo-label retention rate
- pseudo-label accuracy on held-out composed examples
- boundary-slice accuracy
- number of successful self-improvement rounds before collapse

Suggested threshold metric:

- maximum length with at least 90 percent exact-match accuracy

This is especially helpful for workshop figures.

## Figures for the Workshop Paper

Minimum figure set:

1. Addition self-improvement curve by round
2. Addition boundary-carry vs no-boundary stitched accuracy
3. Run-length generalization curve comparing short-only, naive self-training, and compositional self-improvement
4. Multiplication staged results, ideally by difficulty slice

If space is tight, the strongest compact figure set is:

1. one anchor figure for addition
2. one transfer figure for run-length
3. one stress-test figure for multiplication

## Implementation Plan

### Near-term

1. Finish and verify the addition reproduction on `Qwen3-0.6B`.
2. Refactor the current addition pipeline so task-specific logic is separable from the self-improvement loop.
3. Add a run-length task module with:
   - dataset generation
   - composition operator
   - evaluation by length
4. Run run-length baselines and confirm the method works in the clean setting.

### After clean transfer tasks

1. Add multiplication in staged form.
2. Start with exact composition of block products.
3. Then move toward fully compositional pseudo-label generation.

### Nice-to-have, not necessary for workshop

1. a third non-arithmetic task
2. broader model sweep
3. stronger architecture comparisons

## Risks

### Risk 1: Run-length is too easy

If the clean transfer task saturates immediately, it may not be persuasive on its own.

Response:

- use longer lengths
- reduce training coverage
- focus on extrapolation gap rather than absolute difficulty

### Risk 2: Multiplication is too unstable

This is the most likely failure mode.

Response:

- use staged multiplication
- report exact-composition intermediate results
- frame full multiplication as future work if needed

### Risk 3: Gains are actually from extra data, not composition

Response:

- include naive self-training and corrupted-composition controls

### Risk 4: The current pipeline is too addition-specific

Response:

- refactor early so the loop is task-agnostic and the task modules only define:
  - data generator
  - composition function
  - evaluation slices

## Success Criteria for the Workshop Version

The workshop version is strong enough if it shows:

1. a clean addition reproduction,
2. a successful transfer to run-length,
3. at least one credible multiplication result,
4. clear evidence that correct composition matters.

It does not need:

- full unrestricted multiplication,
- model growth,
- or a broad benchmark suite.

## Recommended Next Steps

1. Treat addition as the reference implementation.
2. Build run-length next because it is the fastest way to validate task transfer.
3. Approach multiplication in stages rather than as one monolithic benchmark.
4. Keep the paper claim narrow: compositional self-improvement can bootstrap fixed-capacity models on tasks with faithful decomposition.
