# Notebook Index

This folder is a mixed analysis area. Prefer editing the source notebook
`*.ipynb`; files named `*.executed.ipynb` are output snapshots from prior runs.

## Adaptive Config Self-Improvement

| Notebook | Main purpose | Primary run/log roots |
| --- | --- | --- |
| `adaptive_config_log_inspection.ipynb` | Print exact prompts, traces, and candidate logs for early trace-replay runs. | `artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205`; `artifacts/logs/adaptive-cand-addition-config-9346764.out`; `artifacts/logs/adaptive-cand-run-length-config-9346768.out` |
| `adaptive_candidate_trace_replay_analysis.ipynb` | Early 4-condition analysis, including config/program/policy/meta. Historical only. | `artifacts/figures/adaptive_candidate_trace_replay_analysis` plus older adaptive candidate run roots embedded in the notebook |
| `adaptive_config_replay_sweep_analysis.ipynb` | Selected proposal-trace replay sweep and validity/heatmaps. | `artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808`; `artifacts/runs/adaptive_candidate_trace_replay_full_bs16_20260607_121205` |
| `adaptive_outcome_validity_analysis.ipynb` | Outcome-trace and validity reward analysis. | `artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824` |
| `adaptive_grpo_zero_variance_analysis.ipynb` | GRPO zero-variance behavior: fixed baseline vs skip. | `artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654`; `artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824` |
| `adaptive_grpo_learning_dynamics.ipynb` | Learning dynamics, proposal reward traces, validity, and candidate updates. | `artifacts/runs/adaptive_config_grpo_zero_variance_25a_20260611_153654`; `artifacts/runs/adaptive_config_outcome_sweep_25a_20260610_151824` |
| `adaptive_config_post_rehearsal_slim_analysis.ipynb` | Historical post-task proposal rehearsal and slim-checkpoint runs. Historical only after cleanup. | `artifacts/runs/adaptive_config_post_rehearsal_grpo_slim_n8_n16_20260613_100113` |
| `adaptive_config_three_way_25attempts.ipynb` | Three-way comparison: initial config, selected trace replay, packed local system runs. | `artifacts/runs/adaptive_candidate_full_selected_rounds_20260604_140650`; `artifacts/runs/adaptive_config_replay_sweep_25r50a_20260610_002808`; `artifacts/runs/adaptive_config_packed_gpu_local_parallel_20260615_220549` |
| `adaptive_config_trial_summary.ipynb` | Broad trial history with tables and heatmaps across multiple adaptive config iterations. | Multiple run roots listed in the notebook, including the roots above |
| `adaptive_merged_agent_20260619_results.ipynb` | Merged-agent update runs and prompt-history plumbing check. | `artifacts/runs/adaptive_candidate_25a_seedfull_100steps_histfix_lp2_8job_20260620_233938` |
| `adaptive_1p7b_history_novelty_25a_analysis.ipynb` | History on/off and novelty beta 25-attempt sweep. | `artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947`; `artifacts/logs/adaptive_1p7b_history_novelty_25a_20260622_234947` |
| `adaptive_1p7b_history1_novelty0p05_results.ipynb` | Focused best-cell analysis for history on and novelty beta 0.05. | `artifacts/runs/adaptive_1p7b_history_novelty_25a_20260622_234947` |
| `adaptive_qwen3_model_compare_candidate_analysis.ipynb` | Qwen3 1.7B vs 4B 3-attempt candidate/proposal comparison. | `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528` |
| `adaptive_cached_seed_25a_proposal_dynamics.ipynb` | Cached seed, eval batch, and proposal dynamics comparison. | `artifacts/runs/adaptive_cached_seed_25a_evalbs128_20260622_193946` |
| `adaptive_recent_grpo_sweep_results.ipynb` | Recent LR/anchor/Dr.GRPO sweep analysis. Historical after simplification. | `artifacts/runs/adaptive_grpo_lr_anchor_sweep_20260623_105700`; `artifacts/runs/adaptive_grpo_lr_anchor_sweep_anchorfix_20260624_120855`; `artifacts/runs/adaptive_dr_grpo_default_cell_20260623_121407`; `artifacts/runs/adaptive_dr_grpo_default_cell_anchorfix_20260624_120907` |
| `adaptive_runtime_disaggregation.ipynb` | Runtime breakdown by seed training, candidate training, evaluation, and proposal update. | `artifacts/runs/adaptive_qwen3_model_compare_3a_20260622_132528` and related adaptive run roots |

## Workshop / Paper Experiments

| Notebook | Main purpose | Primary run/log roots |
| --- | --- | --- |
| `addition_exact_digits_fixed_binary_comparison.ipynb` | Addition exact-digits fixed-binary comparison. | `artifacts/runs/addition_exact_digits_fixed_binary_*`; `artifacts/runs/figure2_condition_sweep_20260419_212011/stage1/addition/...` |
| `addition_fixedwidth_mixed_comparison.ipynb` | Fixed-width mixed addition comparison. | `artifacts/runs/addition_fixedwidth_mixed_20260424_180824` |
| `addition_fixedwidth_moredata_heatmaps.ipynb` | Addition fixed-width more-data sweep heatmaps. | `artifacts/runs/addition_fixedwidth_moredata_sweep_20260424_214435`; `artifacts/runs/addition_fixedwidth_mixed_20260424_180824` |
| `figure3_real_seed_data_ablation.ipynb` | Figure 3 real seed-data ablation. | `artifacts/runs/figure3_real_seed_data_ablation_*` |
| `figure3_seed_quality_sample_size.ipynb` | Figure 3 seed-quality/sample-size analysis. | Figure 3 seed-quality run roots embedded in the notebook |
| `run_length_fixed_binary_comparison.ipynb` | Run-length fixed-binary comparison and rescue analysis. | `artifacts/runs/run_length_fixed_binary_20260425_150902`; `artifacts/runs/run_length_fixed_binary_round4_rescue_5seed_dryrun_20260425_174829`; `artifacts/runs/figure2_condition_sweep_20260419_212011/stage1/run_length/...` |
| `run_length_multisymbol_balanced_eval.ipynb` | Multisymbol balanced-eval run-length analysis. | `artifacts/runs/run_length_multisymbol_saved_balanced_20260423_075058` |
| `run_length_multisymbol_heatmaps.ipynb` | Multisymbol run-length heatmaps across alphabet/seed sweeps. | `artifacts/runs/run_length_multisymbol_anyrun_expand8_20260422_215338`; `artifacts/runs/run_length_alpha5_seedrange_sweep_20260422_235605`; `artifacts/runs/run_length_alpha5_seed6_10_self_improve_20260423_012154` |
| `run_length_symbol_pair_alpha10_seed_beam8_analysis.ipynb` | Alpha10 symbol-pair seed beam analysis. | `artifacts/runs/run_length_symbol_pair_alpha10_seed_beam8_20260424_105257` |
| `run_length_symbol_run_pair_alpha10_current_seed.ipynb` | Alpha10 current-seed pair analysis. | `artifacts/runs/run_length_multisymbol_pair_alpha10_currentseed_full_20260423_123229`; `artifacts/runs/run_length_multisymbol_pair_alpha10_seedcheck_20260423_114538` |
| `run_length_symbol_run_pair_alpha10_paper_heatmap.ipynb` | Alpha10 paper heatmap and round-4 seed sweep. | `artifacts/runs/run_length_multisymbol_pair_alpha10_strongseed_warmup500_schedulerfix_20260423_202440`; `artifacts/runs/run_length_symbol_pair_alpha10_round4_seed_sweep_20260423_215420` |
| `run_length_symbol_run_pair_alpha10_warmup300.ipynb` | Alpha10 warmup-300 analysis. | `artifacts/runs/run_length_multisymbol_pair_alpha10_strongseed_warmup300_20260423_182952` |
| `run_length_symbol_run_pair_heatmaps.ipynb` | All-round multisymbol pair heatmaps. | `artifacts/runs/run_length_multisymbol_pair_allrounds_20260423_095142` |

## Utility Files

| File | Purpose |
| --- | --- |
| `vis.py` | Shared plotting helpers for notebooks. |
| `eval_accuracy_over_rounds.png`, `per_digit_accuracy_over_rounds.png` | Static image outputs from earlier analyses. |
