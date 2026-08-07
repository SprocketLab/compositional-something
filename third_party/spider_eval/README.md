# Vendored Spider evaluation files

`process_sql.py` and `evaluation.py` are copied verbatim from
https://github.com/taoyds/spider (Apache License 2.0), commit
`b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c`, fetched 2026-08-06.

They are used ONLY for the official hardness classifier
(`Evaluator.eval_hardness`) and the exact-set-match secondary metric, so that
hardness tiers reported for the SParC/Spider CSI task match the literature.
Test-suite execution scoring is reimplemented in
`self/coding/sparc_composition.py` (see its module docstring for the audit
that cross-checks the two).
