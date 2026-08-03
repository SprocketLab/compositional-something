# Research Plan: CLUTRR-Comp

## Compositional self-improvement on kinship relation chains

**Status (2026-08-03): discontinued.** The seed regime (.95 at k<=4) and the
frontier gap (.44 at k>=5) are real, and the kinship algebra is exact. The task
still fails, for a reason that took three routes to isolate: **CLUTRR's parts are
never in the seed regime.** A sub-chain of a long story scores .487 even when its
sentences are extracted, against .95 for a standalone short story, because the
story's entities interleave -- only 2.2% of extracted spans are free of off-chain
people. Composition consequently loses to direct prediction at every k (−.202
overall), with errors compounding near-independently, the same signature that
ended BFCL. Section 6b has the numbers and the pre-registered gate they failed.
Sections 5-6 below are kept as the record of how this was narrowed down.

---

## 1. Why CLUTRR, after BFCL

BFCL was discontinued (see `../coding/BFCL_Comp_Research_Plan.md` Section 39)
because its parallel calls are *independent*: per-call accuracy was flat across
frontiers (.897/.863/.899 at k=2/4/8) and exact-match tracked `p^k` to within a
few percent, so decomposition could not buy anything.

[CLUTRR](https://arxiv.org/abs/1908.06177) infers a kinship relation between two
people from a short story describing a chain of relations. Composition derives a
*new* relation rather than concatenating outputs, and the dataset ships a
train-short/test-long split by construction: train contains chains of 2-4 hops,
test contains 2-10.

Data: [CLUTRR/v1](https://huggingface.co/datasets/CLUTRR/v1), config
`gen_train234_test2to10`. 12,064 train rows (4050/3995/4019 at k=2/3/4) and
1,048 test rows spanning k=2-10. Closed label space of 18 relations, so
exact-match scoring is unambiguous.

---

## 2. The screening criterion, corrected

BFCL produced this screen: *can the model solve parts reliably while failing
composites?* That gap is the headroom decomposition exploits.

An earlier formulation -- "per-step retention must fall with k" -- is **wrong**
and should not be used. When accuracy is roughly flat in k, `acc(k)^(1/(k-1))`
mechanically climbs toward 1 regardless of what is happening, so the statistic
reports an artifact. CLUTRR's retention rises from .810 to .932 across k=5-10
while accuracy stays flat; by that rule CLUTRR would be rejected, yet it has a
50-point seed/frontier gap.

---

## 3. There is no seed regime without training

Base Qwen3.5-4B scores **.403** on a 300-row k=2-4 holdout, and 44.7% at k=2 in
an initial screen. The k-curve was flat and non-monotonic
(.447/.200/.150/.400/.300/.225/.350/.375/.400 at k=2..10) because it measured
noise around a weak baseline, not compositional degradation.

Screens on an untrained model are uninformative for this task. Train the seed
first.

---

## 4. Seed training works, and the frontier gap is large

LoRA rank 16, all-linear, 4,000 train examples at k=2-4, 300 steps, lr `2e-4`,
effective batch 16. Train loss .144.

| Holdout k=2-4 | Before | After |
|---|---:|---:|
| accuracy | .403 | **.950** |

Test accuracy by chain length:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 1.000 | .880 | .831 | .430 | .490 | .430 | .540 | .390 | .530 |

A sharp cliff at the training boundary: .83 at k=4 falls to .43 at k=5, then
plateaus near .45-.53 through k=10. Predictions stay diverse (sister, brother,
nephew, niece), so this is not majority-class collapse.

**This is the property BFCL lacked.** The seed is reliable inside its regime and
fails outside it, leaving roughly 50 points of headroom for decomposition to
recover. Note the failure is a *cliff*, not compounding decay -- accuracy does
not keep falling from k=5 to k=10 -- which suggests a distribution-shift failure
rather than accumulating per-step error.

---

## 5. The decomposition experiment is blocked on an incomplete composition rule

**Design.** For each test chain of k>4 hops, split into contiguous sub-chains of
at most 4 hops, ask the seed each sub-chain over the *unchanged story* (only the
queried pair moves), and combine the answers with a kinship composition rule.
Both arms use the same instances, so the comparison is paired. `<>` is a
deterministic lookup table, not a model call, mirroring concatenation in
addition and run-length.

**Result.** Decomposition is far worse than direct prediction at every k:

| k | n | chunks | direct | composed | delta | unresolved |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 100 | 2 | .430 | .150 | -.280 | .550 |
| 6 | 62 | 2 | .371 | .065 | -.306 | .661 |
| 7 | 82 | 2 | .476 | .085 | -.390 | .695 |
| 8 | 69 | 2 | .507 | .101 | -.406 | .710 |
| 9 | 48 | 3 | .396 | .042 | -.354 | .771 |
| 10 | 37 | 3 | .595 | .000 | -.595 | .946 |

**This measures a defect in the experiment, not the method.** Substituting
*gold* sub-chain relations for the model's predictions:

| outcome with perfect sub-chain answers | count |
|---|---:|
| sub-chain unfoldable (edge fold hits a missing pair) | 278 |
| gold parts compose to an illegal pair | 66 |
| gold parts produce the correct target | **85** |

The design has a ceiling near **20%** even with a flawless model, well below the
direct baseline of ~.45. The composed numbers above are therefore
uninterpretable.

**Cause.** The rule was mined from training data -- first from adjacent edges in
2-hop stories, then from every binary step in `proof_state` across all 12k rows.
Both yield exactly the same **62** of 324 possible relation pairs, because
CLUTRR's generator only ever composes along its own proof trees. Folding edges
left-to-right at arbitrary split points produces pairs the table never saw --
`nephew o brother`, `sister o aunt`, `niece o sister` -- which are perfectly
legal kinship compositions in reality (a nephew's brother is a nephew) but absent
from the observed algebra.

So the implemented `<>` is a partial observation of the kinship algebra, not the
algebra. It returns bottom on legal compositions, which is not a guard doing
useful work -- it is a missing rule.

---

## 5b. The algebra, fixed -- and a negative result that indicts the design again

**The algebra.** `reports/composition_screen/kinship.py` represents each relation
as a walk over parent/child/spouse steps (father `P`, son `C`, brother `PC`,
uncle `PPC`, nephew `PCC`, father-in-law `SP`, son-in-law `CS`, ...) and composes
by concatenation with three reductions: `CP` -> empty (a child's parent is
oneself), `SS` -> empty, and `SC` -> `C` (CLUTRR has no step-children).
Deliberately asymmetric: `SP` does *not* reduce, because a spouse's parent is an
in-law. This reproduces **all 62** observed rules with zero errors, against the
mined table's partial coverage.

Known wart: the rewriting is **not confluent**. `SCP` reduces to either empty or
`S` depending on rule order -- "my spouse's child's parent" is genuinely me or my
spouse. That ambiguity is real kinship and a legitimate bottom case.

**Adaptive splitting.** A fixed midpoint cut fails because 21.7% of sub-chains
fold to relations outside the 18-name answer space (great-grandmother), which the
model cannot express. Enumerating all contiguous cuts into <=4-hop chunks and
taking the first whose composition resolves finds a working cut for **99.1%** of
instances, and requires no gold -- an unusable cut announces itself when the rule
returns bottom.

**Result.** With the mechanics sound (unresolved <= .067), decomposition still
does not beat direct prediction:

| k | n | direct | composed | delta |
|---:|---:|---:|---:|---:|
| 5 | 60 | .467 | .433 | -.033 |
| 6 | 60 | .383 | .367 | -.017 |
| 7 | 60 | .517 | .467 | -.050 |
| 8 | 60 | .483 | .433 | -.050 |
| 9 | 48 | .396 | .396 | +.000 |
| 10 | 37 | .595 | .459 | -.135 |
| **all** | 325 | **.468** | **.425** | **-.043** |

Paired: 43 direct-only-correct against 29 composed-only-correct, McNemar
p ~ 0.13 -- a wash.

**Why, and why the experiment was still the wrong one.** The prediction was
`.83 x .88 x .88 = .64` at k=10, assuming a 4-hop sub-chain is as easy as a
standalone 4-hop instance. Backing out from the observed .459 over three chunks
gives implied per-chunk accuracy ~.77. Sub-chain questions were asked over the
*entire* k-hop story, so the model still had to locate four relevant sentences
among ten with every other entity as a distractor. That is not the analogue of
addition, where `x o x'` gives each sub-problem only its own digits.

Slicing the story per sub-chain is not available either: only **30%** of stories
have one sentence per edge, because a single sentence can encode two relations.

**The direction was backwards.** Compositional self-improvement *builds* hard
instances from easy ones; it never needs to decompose an arbitrary hard instance.
Addition concatenates two solved digit blocks; BFCL joined two atomic requests.
Taking CLUTRR's existing k=8 stories and cutting them up is the inverse operation
and demands machinery the method does not use.

---

## 6. What has to happen next

0. **Compose, do not decompose.** Build synthetic k=6-10 instances by
   concatenating two or three short-chain stories at a shared junction entity.
   Each part is then a *standalone short story* -- the seed regime at .95, not
   the ~.77 measured for sub-chains embedded in a long story. Predicted
   composed accuracy is roughly `.95 x .95 = .90` against ~.48 direct. Check
   first that the concatenated story reads coherently and that the junction
   entity is unambiguous; then compare composed against direct on the
   constructed set, and separately test transfer to CLUTRR's own k>=5 test
   split, which is the claim that actually matters.
1. ~~**Obtain the true kinship algebra.**~~ Done -- see Section 5b. Options, cheapest first: represent each
   relation as (generation delta, gender, lineage type) and compose analytically;
   infer the full 18x18 table by closure over the 62 known entries; or take the
   rule set from the original CLUTRR generator. Validate by checking that gold
   sub-chains fold and compose to the gold target on ~100% of test rows, which
   is the acceptance test the current rule fails at 20%.
2. **Rerun the decomposition experiment.** Only then does direct-versus-composed
   mean anything. Prediction from the seed's own numbers: sub-chains at k<=4
   score .83-1.00, so a 10-hop chain split 4+3+3 should reach roughly
   `.83 x .88 x .88 = .64` against .53-.60 direct.
3. **If decomposition wins, run Round 1.** Compose pseudo-labels for k=5-6 from
   seed predictions on sub-chains, retrain, and test whether the frontier moves.
4. **Watch for the guard question.** With a complete algebra `<>` becomes total,
   so CLUTRR would test frontier expansion but *not* the filtering story that
   addition's boundary carry supports. If a guard is wanted, ambiguity would have
   to come from elsewhere -- for example gender-underdetermined chains.

## 6b. Sentence extraction, and the measurement that ends the task

Section 6's step 0 and the decomposition route failed in opposite ways: concatenated
composites beat direct but were twice as hard as real CLUTRR chains at the same k
(.260 vs .507), while decomposing a real story tied direct (.425 vs .468). The
diagnosis offered for the latter was that sub-chain questions were asked over the
*entire* long story, so the model still had to find the relevant sentences. The
untried fix was to extract them -- the analogue of handing an addition sub-problem
its own digit block.

That fix has now been run end to end (`gate_b_extraction.py`, job 11954642) on 317
real k>=5 stories from `gen_train23_test2to10/test.csv`, a config with **zero story
overlap** with our evaluation split. Four arms, all paired on the same instances:

| k | n | direct | part (full story) | part (extracted) | composed (ext) | delta |
|---|---|---|---|---|---|---|
| 5 | 60 | .500 | .541 | .541 | .333 | −.167 |
| 6 | 60 | .600 | .511 | .532 | .350 | −.250 |
| 7 | 60 | .283 | .318 | .365 | .150 | −.133 |
| 8 | 60 | .383 | .390 | .488 | .217 | −.167 |
| 9 | 45 | .489 | .533 | .561 | .222 | −.267 |
| 10 | 32 | .406 | .316 | .395 | .125 | −.281 |
| **all** | **317** | **.445** | **.446** | **.487** | **.243** | **−.202** |

Extraction does what it was designed to do, and it is not enough. It helps parts by
**+.041** and never flips the sign: composition loses to direct at every single k.

**Why: the sub-problems are not in the seed regime.** This is the first *direct*
measurement of per-part accuracy -- the earlier "~.77" was inferred from composed
accuracy under a best-cut search, which selects the cuts that happen to resolve and
so overstates it. Measured, a 2-4 hop segment of a long story scores **.487**, against
**.95** for a standalone k=2-4 story. The method's load-bearing premise is that parts
land inside the reliable regime; here they do not, so there is nothing to compose from.

**Why extraction cannot fix that: CLUTRR stories are not separable.** Over 863
sub-chain spans, extraction keeps a median of 62% of sentences and loses 0% of
chains -- but only **2.2%** of spans are free of off-chain entities, with a mean of
**1.46** distractors remaining. Adjacent chain entities are described in interleaved
sentences, so the span that preserves a sub-chain's edges necessarily drags in people
outside it. A sub-story is therefore never the standalone short story the seed was
trained on. Unlike addition, where a digit block is genuinely independent, CLUTRR's
*relation* is compositional while its *surface form* is not.

**And the errors compound independently -- the BFCL signature.** With `p` the measured
extracted-part accuracy and `c` chunks per instance:

| k | chunks | p | p^c | composed | ratio |
|---|---|---|---|---|---|
| 5 | 2 | .541 | .292 | .333 | 1.14 |
| 6 | 2 | .532 | .283 | .350 | 1.24 |
| 7 | 2 | .365 | .133 | .150 | 1.13 |
| 8 | 2 | .488 | .238 | .217 | 0.91 |
| 9 | 3 | .561 | .176 | .222 | 1.26 |
| 10 | 3 | .395 | .062 | .125 | 2.03 |

Observed composed accuracy sits at 0.9-1.3x the independent-error prediction (2.03 at
k=10 on 32 instances). BFCL was discontinued on a ratio of 1.0. CLUTRR reaches the
same place by a different road: there the parts were already independent, here the
parts are dependent in principle but the model cannot isolate them in practice.

**Verdict: stop.** Gate B was specified in advance as extracted-part accuracy >=.80
*and* composed beating direct by >=5 points. It returned .487 and −.202. Round 1
(Gate C) was not run -- training on pseudo-labels that are correct .24 of the time
cannot move a .44 frontier. The Gate C script exists (`clutrr_round1_pool.py`, with
pseudo-replay and a pool/eval hash check) should the premise ever change.

**What would have to be true for CLUTRR to work.** Some route that yields parts inside
the seed regime *and* composites matching CLUTRR's own distribution. All three known
routes fail one or the other: concatenation gives easy parts but off-distribution
composites; decomposition and extraction give on-distribution composites but hard
parts. A fourth route -- generating both parts and composites with CLUTRR's own
generator -- would satisfy both, but that is supervised data construction, not
self-improvement from an unlabeled pool.

---

## 7. Practical notes

- Only **635 of 1,048** test rows are a clean `0->k` path with one name per node
  (`story_edges == [(0,1),(1,2),...]`, `query_edge == (0,k)`). The rest reuse
  entities or query a different pair, so node indices are not a usable handle for
  sub-chains. Usable counts per k: 131/62/82/69/48/37 at k=5-10.
- Entity names are recoverable by node index from the `genders` field, verified
  on all 1,048 rows.
- `sacct -j <id>` on this cluster can return an unrelated job's record when the
  queried job has no accounting entry yet. Validate the JobID field or read
  `squeue` for live state.

## References

- [CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text](https://arxiv.org/abs/1908.06177)
- [CLUTRR/v1 dataset](https://huggingface.co/datasets/CLUTRR/v1)
- Artifacts: `reports/composition_screen/` (screen, seed adapter, decomposition results,
  `gate_b_extraction.json`)
