# Self-Improving Composition Configs

## Research Hypothesis

We want to replace manually designed composition rules with a self-improvement loop where the model learns to propose its own data-generation configurations.

Inspired by SEAL, the model should generate self-edits or configurations, but here the edit is a composition rule for creating new self-labeled training data.

The core hypothesis is that a small LLM can learn an acquisition policy:

$$
s_t \rightarrow z_t \rightarrow \mathcal{D}_{z_t}^{\mathrm{pseudo}} \rightarrow M_{t+1}
$$

where:

- $s_t$: current learner state, including per-regime accuracy
- $z_t$: proposed composition config
- $\mathcal{D}_{z_t}^{\mathrm{pseudo}}$: composed data labeled by the current model
- $M_{t+1}$: candidate model after self-training

The desired learned behavior is:

$$
\pi_\theta(z \mid s_t)
$$

where the model proposes configurations likely to improve frontier regimes.

Default model: `Qwen/Qwen3-1.7B`.

Tasks:

- `addition`: initial digits roughly $3$-$7$, frontier starts at $8+$
- `run_length`: initial lengths roughly $8$-$16$, frontier starts at $17+$

We use self-labeling only. The model never calls tools or oracles during training.

## Config Setting

Current focus is config generation, not full program generation.

A config is:

```json
{
  "left": 5,
  "right": 3,
  "guard": "none"
}
```

The target regime is usually implied:

$$
\mathrm{target} = \mathrm{left} + \mathrm{right}
$$

We currently allow duplicate or already-seen choices because they may still improve the model, but we deduplicate equivalent actions before expensive candidate training.

## Current Pipeline

1. Seed train $M_0$ on base regimes.
2. Evaluate held-out accuracy by digit or length.
3. Build state $s_t$, including:
   - source regimes
   - frontier regimes
   - per-regime accuracy
   - current average/frontier accuracy
4. Prompt $M_t$ to sample $N$ proposal configs.
5. Parse and validate proposals.
6. Deduplicate equivalent valid actions.
7. For each unique valid config $z$:
   - compose examples from `left` and `right`
   - pseudolabel using current model $M_t$
   - train candidate $M_t^z$
   - evaluate held-out accuracy
8. Compute reward:

$$
R(z; s_t)
=
\Delta_{\mathrm{frontier}}(z)
+
\lambda \Delta_{\mathrm{avg-from-init}}(z)
$$

where:

$$
\Delta_{\mathrm{frontier}}(z)
=
\mathrm{Acc}_{\mathrm{frontier}}(M_t^z)
-
\mathrm{Acc}_{\mathrm{frontier}}(M_t)
$$

and:

$$
\Delta_{\mathrm{avg-from-init}}(z)
=
\mathrm{Acc}_{\mathrm{avg}}(M_t^z)
-
\mathrm{Acc}_{\mathrm{avg}}(M_0)
$$

with $\lambda \approx 0.1$.

9. Select the best candidate if it clears the threshold.
10. Replace current checkpoint with the selected candidate.
11. Update proposal behavior using proposal traces and reward signals.

## Proposal Training Tried

### 1. Selected-Trace SFT

Train on selected proposal traces.

Issue: improves imitation/format, but encourages copying the same action.

### 2. Oversampled Selected Traces

Oversample selected traces to prevent JSON collapse.

Issue: further strengthened repetition.

### 3. Outcome Traces

Store action plus observed outcome:

```json
{
  "action": {"left": 5, "right": 3, "guard": "none"},
  "observed": {
    "valid": true,
    "frontier_delta": 0.03,
    "avg_delta": 0.01
  }
}
```

Goal: teach the model a lightweight world model of data choice to improvement.

### 4. Merged Proposal Objective

Combined:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{GRPO}}
+
\alpha \mathcal{L}_{\mathrm{observation\ CE}}
+
\beta \mathcal{L}_{\mathrm{format\ CE}}
$$

with small weights such as:

$$
\alpha = 0.2,\quad \beta = 0.1
$$

### 5. Reward Variants

Tried:

- outcome reward
- rank reward
- validity reward
- zero-variance skip
- fixed-baseline variant
- history on/off

No clear fix for repetition.

## Main Empirical Findings

Initial config self-improvement showed some promise. A few selected rounds worked, and `run_length` sometimes generalized surprisingly well from few compositions.

But the system later broke down.

Failure progression:

1. Early runs: valid JSON collapsed after a few rounds.
2. Selected-trace SFT: format improved, but repeated configs increased.
3. GRPO-style updates: did not reliably improve acquisition behavior.
4. Prompt changes:
   - tagged reasoning/action/observation: often invalid
   - free reasoning: low valid rate
   - strict reasoning plus first JSON: high validity, low diversity

Latest prompt pilot:

| Setting | Task | Valid Proposals | Unique Actions | Main Failure |
| --- | ---: | ---: | ---: | --- |
| JSON reasoning first-JSON | addition | $24/24$ | $1$ | complete action collapse |
| JSON reasoning first-JSON | run_length | $6/24$ | $2$ | low validity and repeated actions |

For addition, all valid proposals were:

```text
left=5, right=3, guard=none, target=8
```

So the new prompt improved validity, but not the model's sense of improvement.

## Main Challenges

### Validity vs Diversity

The model can learn to emit valid JSON, but it collapses to one safe action.

### Missing Acquisition Function

The desired learned object is:

$$
f_\theta(s_t, z)
\approx
\mathbb{E}[R(z; s_t)]
$$

But current training mostly teaches:

```text
emit a valid config
```

not:

```text
choose the config likely to improve the frontier
```

### Selected-Trace SFT Is Temporally Mismatched

A selected action at state $s_t$ may not be appropriate at state $s_{t+1}$.

So SFT on selected traces induces copying:

$$
z_t \mapsto z_{t+1}
$$

rather than state-conditioned adaptation.

### GRPO May Be Mismatched

The reward is delayed, candidate-level, and expensive. Token-level policy gradients over small proposal batches may not teach the state-action-improvement relation.

### Coupled Training Is Unstable

Task SFT can damage proposal formatting. Proposal-format losses can dominate or conflict with task learning.

### Reward Is Expensive

Scoring one proposal requires:

$$
\mathrm{compose}
\rightarrow
\mathrm{pseudolabel}
\rightarrow
\mathrm{train}
\rightarrow
\mathrm{evaluate}
$$

This makes exploration costly and noisy.

## Open Research Question

How do we give the model a real sense of improvement?

The missing capability is a learned mapping:

$$
\mathrm{current\ accuracy\ profile}
+
\mathrm{source\ pool}
+
\mathrm{proposed\ composition}
\rightarrow
\mathrm{expected\ frontier\ improvement}
$$

Current evidence suggests validity learning is not enough, and selected-trace imitation may actively cause collapse. The core research problem is learning an acquisition policy for self-generated compositional data under noisy, delayed, self-labeled feedback.
