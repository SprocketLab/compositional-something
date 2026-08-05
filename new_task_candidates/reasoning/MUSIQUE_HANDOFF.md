# MuSiQue — continuation guide

Practical handoff for picking this up on another cluster. The scientific record
is `MuSiQue_CSI_Research_Plan.md` §25–26; this file is what you need to *run*
things. Written 2026-08-04, at commit `ea8331d`.

---

## 1. Where we are in one paragraph

BFCL and CLUTRR were both discontinued. MuSiQue is the surviving candidate and
it **cleared its screen**: after training a one-hop seed, composing the model's
own sub-answers beats asking it the whole multi-hop question by **+.080**
(95% CI [+.037, +.123], McNemar p=.0009, 300 paired dev instances). Before
seeding the same contrast was +.023, CI spanning zero.

This is an **inference-time** advantage. No pseudo-labels have been generated,
nothing has been retrained on them, and no frontier has moved. Round 1 is the
actual claim and has not been run.

---

## 2. Assets

### Must survive the migration

| path | size | why |
|---|---|---|
| `reports/composition_screen/musique_seed/adapter/` | 144 MB | **the one-hop seed.** Round 1 continues from it. ~2.4 h of GPU to rebuild. **gitignored — rsync it explicitly** |
| `reports/composition_screen/data/musique_train.jsonl` | 230 MB | untracked; re-downloadable (§4) |
| `reports/composition_screen/data/musique_dev.jsonl` | 29 MB | tracked in git |

Everything else needed is tracked: `musique_seed.py`, `musique_isolation.py`,
`musique_shortcut.py`, `musique_retrieval_control.py`, their `.slurm` files, the
result JSONs, and `musique_seed/per_instance.jsonl`.

### Safe to leave behind

* `reports/composition_screen/logs/probe/` (178 MB) — throwaway from `memprobe.py`.
* `reports/composition_screen/round1/`, `round1_v2/` (144 MB each) — CLUTRR
  rounds, superseded; CLUTRR is discontinued.
* `reports/composition_screen/clutrr_seed/adapter/` (144 MB) — only needed to
  re-run discontinued CLUTRR work. Cheap to keep, not needed for new experiments.
* `artifacts/` (972 GB) — check `prepared_start.json` references before deleting
  anything in there; not needed for MuSiQue.

---

## 3. Reproducing the headline number

```bash
sbatch reports/composition_screen/musique_seed.slurm     # ~2.4 h, 1 GPU
```

Trains the seed from scratch and evaluates before/after. Writes
`musique_seed/musique_seed.json` and `per_instance.jsonl`. The **before** block
must reproduce `musique_retrieval_control.json` (+.020/+.150/−.100 per hop); if
it does not, the generation path has drifted and the after-numbers are not
comparable.

To evaluate the existing adapter without retraining, load it with
`load_adapter_for_evaluation(model, Path(".../musique_seed/adapter"))` — see
`musique_retrieval_control.py` for the arm structure.

---

## 4. Data

`musique_dev.jsonl` is MuSiQue-Ans v1.0 dev (2,417 rows). Train is untracked:

```bash
curl -sSL -o reports/composition_screen/data/musique_train.jsonl \
  "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_train.jsonl"
# expect 19,938 lines
```

`musique_seed.py` re-verifies leakage at startup and hard-exits on any overlap:
instance ids, multi-hop questions, and filled (sub-question, answer) pairs — the
last one matters because MuSiQue composes its items from a shared single-hop
pool. All three were 0 on the official split.

---

## 5. Next experiments, in the order I would run them

**(a) Self-proposed decomposition — the one that decides publishability.**
The composed arm currently reads `question_decomposition` from the release, i.e.
it is *handed* how to decompose. Plan §4.2 (End2End) is the matched arm. Until
this exists, the headline is "composition helps when someone tells you the
decomposition," which is a much weaker claim. Cheaper than Round 1.

> **RAN 2026-08-05 on neuronic (`musique_selfdecomp.py`) — negative.** Pooled
> self-composed − direct = +.017, CI [−.023, +.057], p=.51, vs +.070 (p=.0086)
> for the gold arm on the same 300 instances and adapter; self is significantly
> below gold (−.053, p=.036). Not an execution failure (2/300 parse fails, 0
> unfilled `#N` refs, 0 direct-arm drift): the proposals themselves fail, by
> anaphora ("this city") instead of `#N` so the filled step never names the
> entity, and by omitting the sink sub-question so the last answer is not the
> final answer. Both look prompt-fixable (format exemplar + "last step must
> answer the original question") and that iteration has NOT been run. Until it
> succeeds, the weaker headline stands.

**(b) Round 1.** Pseudo-label unlabeled multi-hop questions by composing seed
sub-answers, retrain, test whether *direct* multi-hop accuracy improves.
Prerequisite: partition the train split at full-example level per plan §6
(seed-source vs composition-source must be disjoint) or the pseudo-labels are
partly memorized. The current seed used all of train, so **a fresh seed on a
30% partition is needed before Round 1**, not the adapter above.

**(c) More seed capacity.** Part accuracy after seeding is .815/.770/.703 at
2/3/4 hops, well below CLUTRR's .95 seed regime, and headroom shrinks with hops
(+.120/+.090/+.030 — 4 hops is not individually significant). More seed data or
steps is the obvious untested lever, and it is what would make the 4-hop
frontier move.

---

## 6. Rules learned the hard way — do not relearn these

**Any arm consuming released gold structure needs a matched no-oracle arm.**
Giving sub-questions their own `paragraph_support_idx` paragraph inflated
headroom from +.017 to +.133. Roughly 86% of the apparent advantage was
retrieval, not composition. Plan §4.1 declares the gold-paragraph setting
primary; it would reproduce that artifact.

**Train the seed in the condition the composed arm faces at inference.** The
seed here trains on sub-questions over the *full 20-paragraph context*, not the
gold paragraph, because the composed arm must retrieve for itself. This
knowingly departs from plan §5, which excludes 20-paragraph training as too
expensive. On 48 GB cards that tension gets sharper — plan §4.3
(gold paragraph + a few distractors, ~1000 tokens) is the compromise worth
testing.

**The seed lifts direct prediction too** (+.120 at 3 hops). Baselines must use
the *seeded* direct model. Comparing against the base model overstates the gain.

**Log per-instance outcomes.** The retrieval control stored only aggregates, so
its odd non-monotonic pattern could not be tested. `musique_seed.py` writes
`per_instance.jsonl` and computes McNemar plus a paired bootstrap CI.

**The screening criterion, twice corrected.** Not "is the input separable"
(over-fit to CLUTRR) and not just "is the part easier than the whole" — the
part-vs-whole gap must also beat what the chain loses to cascading. MuSiQue's
gap is +.185/+.203/+.260 and it still bought nothing until the seed lifted part
accuracy.

---

## 7. Cluster notes

> **Migration landed 2026-08-05.** Site values live in
> `reports/composition_screen/cluster_env.sh`; submit any `*_portable.slurm`
> with `sbatch $(bash reports/composition_screen/sbatch_flags.sh) <script>`.
> The headline reproduced on an L40 (+.070 pooled, p=.0086, vs +.080 on Della;
> ~2.7× slower, 6h38m). Cross-cluster noise floor with identical weights:
> ~1.5% of per-instance outcomes flip. Neuronic artifacts are in
> `musique_seed_neuronic/`; the Della originals in `musique_seed/` are
> untouched and remain the reference.

* **48 GB is enough** for everything, including 8B, **with gradient
  checkpointing**. `train_lora(..., gradient_checkpointing=True)` — added in
  `ea8331d`, defaults off. Seed training at seq 4096 drops 86.6 GiB → 20.4 GiB.
  Full table in `reports/composition_screen/GPU_MEMORY_NOTES.md`.
* **Install `flash-linear-attention` and `causal-conv1d`.** Qwen3.5-4B is a
  hybrid linear-attention model running a torch fallback without them, which is
  why it costs *more* memory than Qwen3-8B unchecked. Should cut memory and step
  time; untested.
* Two GPUs buy throughput (DDP), not capacity. Raising `micro_batch_size` to 2
  fits on one card (32.4 GiB at seq 4096) and needs no distributed setup — try
  that before wiring up `torchrun`.
* Scripts must live on `/scratch/gpfs`, not node-local `/tmp`; a Slurm job that
  cannot see its script fails in ~1 s with an empty log.
* `train_lora` **rejects** over-length examples rather than truncating. Filter
  by true tokenized length before training (`musique_seed.py` does).
