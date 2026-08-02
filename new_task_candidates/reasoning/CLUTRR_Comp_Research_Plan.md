# Research Plan: CLUTRR-Comp

## Compositional self-improvement on kinship relation chains

**Status (2026-08-02):** candidate under evaluation. The seed regime and the
frontier gap are established. The decomposition experiment is **blocked** on an
implementation problem described in Section 5, and until it is fixed nothing
here speaks to whether the method works.

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

## 6. What has to happen next

1. **Obtain the true kinship algebra.** Options, cheapest first: represent each
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
- Artifacts: `reports/composition_screen/` (screen, seed adapter, decomposition results)
