# Addition Self-Improvement Round 8 Model

This folder contains the final fixed-binary `with_carry_filtered` addition model
from `artifacts/runs/addition_fixedwidth_mixed_20260424_180824`.

Open `model_card.pdf` for:

- input/output format,
- a loading and inference code snippet,
- the held-out accuracy heatmap.

The model expects prompts like:

```text
Q: 104 + 2305 = ?
A:
```

and produces an unpadded integer answer.

Important: this checkpoint uses the repository's custom no-position LLaMA recipe
model and arithmetic character tokenizer. Load it through `self.core.recipes`
as shown in `model_card.pdf`.
