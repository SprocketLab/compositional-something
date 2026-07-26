# BFCL call-order audit

Run: `/scratch/gpfs/BRENDEN/changho/compositional-something/artifacts/runs/bfcl_cumulative_size_sweep_20260721_132230`

Composed supervision is a valid instance of `f*(x1 o x2) = f*(x1) <> f*(x2)`
only when input composition and output composition use the same permutation.
The persisted sweep rendered the joined request from an independent shuffle of
the leaf clauses while concatenating target calls per component, so clause `k`
does not answer call `k` above two calls.

## 1. Persisted candidates (as trained on)

| candidate file | sampled | target in clause order | mean displaced calls | schema listed in clause order |
|---|---:|---:|---:|---:|
| `calls_2_cross` | 200 | 100.0% | 0.0% | 100.0% |
| `calls_4_cross` | 200 | 4.5% | 74.6% | 100.0% |
| `calls_8_cross` | 200 | 0.0% | 86.8% | 100.0% |
| `calls_2_repeat` | 200 | 100.0% | 0.0% | 100.0% |
| `calls_4_repeat` | 200 | 3.5% | 74.9% | 100.0% |
| `calls_8_repeat` | 200 | 5.5% | 72.5% | 100.0% |

## 2. The model's own ordering convention

| direct predictions | calls | resolved | emitted in clause order |
|---|---:|---:|---:|
| `seed` | 2 | 199 | 100.0% |
| `round_02_direct_g4` | 4 | 200 | 100.0% |
| `round_03_direct_g4` | 8 | 195 | 100.0% |

The model emits calls in request order essentially always, so a target in any
other order is unlearnable structure rather than a stylistic difference.

## 3. Materialized training targets by condition

| cell | regime | resolved | target in clause order |
|---|---|---:|---:|
| `n1000-compose_g1` | calls_2 | 200 | 100.0% |
| `n1000-compose_g1` | calls_4 | 200 | 4.5% |
| `n1000-compose_g1` | calls_8 | 200 | 0.0% |
| `n1000-compose_g4` | calls_2 | 200 | 100.0% |
| `n1000-compose_g4` | calls_4 | 200 | 4.5% |
| `n1000-compose_g4` | calls_8 | 200 | 0.0% |
| `n1000-compose_g4_repeat20` | calls_2 | 200 | 100.0% |
| `n1000-compose_g4_repeat20` | calls_4 | 200 | 4.0% |
| `n1000-compose_g4_repeat20` | calls_8 | 200 | 0.0% |
| `n1000-direct_g4` | calls_2 | 200 | 100.0% |
| `n1000-direct_g4` | calls_4 | 200 | 100.0% |
| `n1000-direct_g4` | calls_8 | 200 | 100.0% |
| `n1000-oracle` | calls_2 | 200 | 100.0% |
| `n1000-oracle` | calls_4 | 200 | 4.5% |
| `n1000-oracle` | calls_8 | 200 | 0.0% |
| `n2000-oracle` | calls_2 | 200 | 100.0% |
| `n2000-oracle` | calls_4 | 200 | 4.5% |
| `n2000-oracle` | calls_8 | 200 | 0.0% |
| `n3000-oracle` | calls_2 | 200 | 100.0% |
| `n3000-oracle` | calls_4 | 200 | 4.5% |
| `n3000-oracle` | calls_8 | 200 | 0.0% |

`direct_g4` targets are the model's own generations and stay in clause order;
every composed and Oracle target is permuted at four and eight calls.  The
confound therefore penalizes exactly the arms the study is about.

## 4. Candidates rebuilt with the corrected builders

| candidate file | sampled | target in clause order | mean displaced calls | schema listed in clause order |
|---|---:|---:|---:|---:|
| `calls_2_cross` | 200 | 100.0% | 0.0% | 56.5% |
| `calls_4_cross` | 200 | 100.0% | 0.0% | 1.5% |
| `calls_8_cross` | 200 | 100.0% | 0.0% | 0.0% |
| `calls_2_repeat` | 200 | 100.0% | 0.0% | 100.0% |

Clause and call order now agree by construction, and schema listing order is
an independent deterministic shuffle, so the positional shortcut documented in
the plan's schema-permutation audit is removed at construction time.

Chance agreement between schema order and clause order is `1/k!` for `k`
distinct functions: 50% at two calls, 4.2% at four, 0.002% at eight.  The
rebuilt columns sit at chance.  The repeat family reads 100% trivially because
all of its calls share one function name, so its schema list has one entry.

## Reproduce

```bash
python -m self.experiments.bfcl_call_order_audit --sample 200
```

## What this does and does not establish

It establishes that every composed and Oracle training target above two calls
asked for a permutation the model had no way to predict, and that the Direct
arm was exempt.  It does not by itself prove the permutation caused the
four/eight-call regressions; that requires retraining on corrected data with a
fixed update budget.  Until then, no conclusion about the four- or eight-call
frontier from the cumulative sweep should be treated as informative.
