from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "launchers" / "self" / "run_adaptive_candidate_training_ailab.sbatch"
SUBMITTER = ROOT / "launchers" / "self" / "submit_adaptive_candidate_training_ailab.sh"
BASE_CONFIG = ROOT / "launchers" / "self" / "config" / "adaptive_candidate_base.env"


def test_adaptive_candidate_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)


def test_adaptive_candidate_launcher_wires_packed_cached_local_workers(tmp_path: Path):
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[PYTHON_STUB]'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "TASK": "addition",
            "OUT_DIR": str(tmp_path / "run"),
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
            "RUN_COMPILE_CHECK": "0",
            "PYTHON_BIN": str(python_stub),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert f"[INFO] Loaded adaptive config: {BASE_CONFIG}" in stdout
    assert "Proposal sampling temp/top_p/force-unique/max-draws/batch-size: 0.9/0.95/1/0/8" in stdout
    assert "Proposal prompt action history/max items: 0/5" in stdout
    assert "Attempts/candidates: 100/8" in stdout
    assert "Max selected candidates: 0" in stdout
    assert "Train epochs/candidate max steps/seed max steps: 1/100/0" in stdout
    assert "Candidate local parallelism/pack/cache-base-state: 2/2/1" in stdout
    assert "Candidate eval backend: transformers" in stdout
    assert "vLLM python/gpu-memory-utilization/dtype: <current-python>/0.80/auto" in stdout
    assert "vLLM flashinfer/eager/max-model-len/max-seqs/max-batched-tokens: off/0/0/0/0" in stdout
    assert "Post-task proposal rehearsal/repeats/max examples: 0/64/256" in stdout
    assert "Proposal GRPO objective: grpo" in stdout
    assert "Proposal GRPO steps/lr/kl/anchor-kl/ref: 1/1e-6/0.01/0.01/adaptive_init" in stdout
    assert "Proposal GRPO grad clip/zero variance/baseline: 1.0/skip/0.5" in stdout
    assert "Proposal GRPO reward mode/outcome scale/dedup/novelty beta: outcome/0.05/1/0.05" in stdout
    assert "Source admission target-accuracy threshold: 0.80" in stdout
    assert "Proposal update loss/observation/format/replay/microbatch: merged_agent/0.2/0.02/256/8" in stdout
    assert "Proposal GRPO span: reasoning_action" in stdout
    assert "--candidate-local-pack-size 2" in stdout
    assert "--candidate-local-cache-base-state" in stdout
    assert "--candidate-eval-backend transformers" in stdout
    assert "--vllm-gpu-memory-utilization 0.80" in stdout
    assert "--vllm-dtype auto" in stdout
    assert "--vllm-flashinfer-sampler off" in stdout
    assert "--vllm-max-model-len 0" in stdout
    assert "--vllm-max-num-seqs 0" in stdout
    assert "--vllm-max-num-batched-tokens 0" in stdout
    assert "--max-selected-rounds 0" in stdout
    assert "--max-steps 100" in stdout
    assert "--seed-max-steps 0" in stdout
    assert "--proposal-temperature 0.9" in stdout
    assert "--proposal-top-p 0.95" in stdout
    assert "--no-proposal-prompt-action-history" in stdout
    assert "--proposal-prompt-action-history-max-items 5" in stdout
    assert "--proposal-sampling-batch-size 8" in stdout
    assert "--force-unique-proposals" in stdout
    assert "--proposal-unique-max-draws 0" in stdout
    assert "--num-rounds" not in stdout
    assert "--proposal-update-loss-mode merged_agent" in stdout
    assert "--proposal-update-microbatch-size 8" in stdout
    assert "--proposal-observation-loss-weight 0.2" in stdout
    assert "--proposal-grpo-objective grpo" in stdout
    assert "--proposal-grpo-span reasoning_action" in stdout
    assert "--proposal-grpo-anchor-kl-coef 0.01" in stdout
    assert "--proposal-grpo-anchor-kl-reference adaptive_init" in stdout
    assert "--proposal-grpo-deduplicate-actions" in stdout
    assert "--proposal-grpo-novelty-bonus-beta 0.05" in stdout
    assert "--source-admission-target-accuracy-threshold 0.80" in stdout
    assert "--proposal-history" not in stdout


def test_adaptive_candidate_submitter_dry_run_writes_matrix_manifest(tmp_path: Path):
    out_root = tmp_path / "adaptive_candidate_submit"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "LOG_DIR": str(tmp_path / "logs"),
            "PYTHON_BIN": sys.executable,
            "TASKS": "addition",
            "CONDITIONS": "config",
            "MODEL_NAME": "Qwen/Qwen3-4B",
            "PROPOSAL_MODEL_NAME": "current",
            "OUTCOME_TRACE_TARGET_MODES": "numeric",
            "PROPOSAL_GRPO_REWARD_MODES": "outcome rank",
            "PROPOSAL_GRPO_ZERO_VARIANCE_MODES": "fixed_baseline skip",
            "MAX_ATTEMPT_ROUNDS": "10",
            "NO_SELECTION_PATIENCE": "10",
            "NUM_CANDIDATES_LIST": "8 16",
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((out_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"] == ["addition"]
    assert manifest["conditions"] == ["config"]
    assert manifest["model_name"] == "Qwen/Qwen3-4B"
    assert manifest["proposal_model_name"] == "current"
    assert manifest["proposal_grpo_reward_modes"] == ["outcome", "rank"]
    assert manifest["proposal_grpo_zero_variance_modes"] == ["fixed_baseline", "skip"]
    assert manifest["num_candidates_list"] == [8, 16]
    assert manifest["proposal_grpo_objective"] == "grpo"
    assert manifest["proposal_grpo_learning_rates"] == ["1e-6"]
    assert manifest["proposal_grpo_kl_coef"] == "0.01"
    assert manifest["proposal_grpo_anchor_kl_coefs"] == ["0.01"]
    assert len(manifest["jobs"]) == 8
    assert (
        manifest["jobs"][
            "addition-config-numeric-n8-reward-outcome-grpo-fixed_baseline-obj-grpo-lr-1e-6-akl-0.01-eval-transformers"
        ][
            "job_id"
        ]
        == "dryrun-adaptive-cand-addition-config-numeric-n8-outcome-fixed-baseline-obj-grpo-lr-1em6-akl-0p01"
    )
    assert manifest["jobs"]["addition-config-numeric-n16-reward-rank-grpo-skip-obj-grpo-lr-1e-6-akl-0.01-eval-transformers"][
        "output_dir"
    ] == str(
        out_root / "addition-config-numeric-n16-reward-rank-grpo-skip-obj-grpo-lr-1em6-akl-0p01-eval-transformers"
    )

    combined = result.stdout + result.stderr
    assert "--job-name adaptive-cand-addition-config-numeric-n8-outcome-fixed-baseline-obj-grpo-lr-1em6-akl-0p01" in combined
    assert "--export" in combined
    assert "MODEL_NAME=Qwen/Qwen3-4B" in combined
    assert "PROPOSAL_MODEL_NAME=current" in combined
    assert "MAX_ATTEMPT_ROUNDS=10" in combined
    assert "MAX_SELECTED_ROUNDS=0" in combined
    assert "NUM_ROUNDS=" not in combined
    assert "NO_SELECTION_PATIENCE=10" in combined
    assert "MAX_STEPS=100" in combined
    assert "SEED_MAX_STEPS=0" in combined
    assert "PROPOSAL_TEMPERATURE=0.9" in combined
    assert "PROPOSAL_TOP_P=0.95" in combined
    assert "PROPOSAL_PROMPT_ACTION_HISTORY=0" in combined
    assert "PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS=5" in combined
    assert "PROPOSAL_SAMPLING_BATCH_SIZE=8" in combined
    assert "FORCE_UNIQUE_PROPOSALS=1" in combined
    assert "PROPOSAL_UNIQUE_MAX_DRAWS=0" in combined
    assert "CANDIDATE_LOCAL_PARALLELISM=2" in combined
    assert "CANDIDATE_LOCAL_PACK_SIZE=2" in combined
    assert "CANDIDATE_LOCAL_CACHE_BASE_STATE=1" in combined
    assert "CANDIDATE_EVAL_BACKEND=transformers" in combined
    assert "VLLM_GPU_MEMORY_UTILIZATION=0.80" in combined
    assert "VLLM_DTYPE=auto" in combined
    assert "VLLM_FLASHINFER_SAMPLER=off" in combined
    assert "VLLM_ENFORCE_EAGER=0" in combined
    assert "VLLM_MAX_MODEL_LEN=0" in combined
    assert "VLLM_MAX_NUM_SEQS=0" in combined
    assert "VLLM_MAX_NUM_BATCHED_TOKENS=0" in combined
    assert "PROPOSAL_GRPO_REWARD_MODE=rank" in combined
    assert "PROPOSAL_GRPO_DEDUPLICATE_ACTIONS=1" in combined
    assert "PROPOSAL_GRPO_OBJECTIVE=grpo" in combined
    assert "PROPOSAL_GRPO_LEARNING_RATE=1e-6" in combined
    assert "PROPOSAL_GRPO_KL_COEF=0.01" in combined
    assert "PROPOSAL_GRPO_ANCHOR_KL_COEF=0.01" in combined
    assert "PROPOSAL_GRPO_ANCHOR_KL_REFERENCE=adaptive_init" in combined
    assert "PROPOSAL_GRPO_NOVELTY_BONUS_BETA=0.05" in combined
    assert "SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD=0.80" in combined
    assert "PROPOSAL_UPDATE_LOSS_MODE=merged_agent" in combined
    assert "PROPOSAL_UPDATE_MICROBATCH_SIZE=8" in combined
    assert "PROPOSAL_HISTORY" not in combined
    assert "POST_TASK_PROPOSAL_REHEARSAL=0" in combined
    assert f"ADAPTIVE_CONFIG_FILES={BASE_CONFIG}" in combined


def test_adaptive_candidate_submitter_dry_run_expands_lr_anchor_kl_sweep(tmp_path: Path):
    out_root = tmp_path / "adaptive_candidate_lr_anchor_sweep"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "LOG_DIR": str(tmp_path / "logs"),
            "PYTHON_BIN": sys.executable,
            "TASKS": "addition run_length",
            "CONDITIONS": "config",
            "OUTCOME_TRACE_TARGET_MODES": "numeric",
            "PROPOSAL_GRPO_REWARD_MODES": "outcome",
            "PROPOSAL_GRPO_ZERO_VARIANCE_MODES": "skip",
            "NUM_CANDIDATES_LIST": "8",
            "PROPOSAL_GRPO_LEARNING_RATES": "1e-6 3e-6 5e-6",
            "PROPOSAL_GRPO_KL_COEF": "0",
            "PROPOSAL_GRPO_ANCHOR_KL_COEFS": "0 0.01 0.03",
            "MAX_ATTEMPT_ROUNDS": "25",
            "NO_SELECTION_PATIENCE": "25",
            "SBATCH_TIME": "5:00:00",
            "SBATCH_MEM": "48G",
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((out_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"] == ["addition", "run_length"]
    assert manifest["proposal_grpo_objective"] == "grpo"
    assert manifest["proposal_grpo_learning_rates"] == ["1e-6", "3e-6", "5e-6"]
    assert manifest["proposal_grpo_kl_coef"] == "0"
    assert manifest["proposal_grpo_anchor_kl_coefs"] == ["0", "0.01", "0.03"]
    assert len(manifest["jobs"]) == 18

    key = "run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5e-6-akl-0.03-eval-transformers"
    assert manifest["jobs"][key]["proposal_grpo_objective"] == "grpo"
    assert manifest["jobs"][key]["proposal_grpo_learning_rate"] == "5e-6"
    assert manifest["jobs"][key]["proposal_grpo_kl_coef"] == "0"
    assert manifest["jobs"][key]["proposal_grpo_anchor_kl_coef"] == "0.03"
    assert manifest["jobs"][key]["output_dir"] == str(
        out_root / "run_length-config-numeric-n8-reward-outcome-grpo-skip-obj-grpo-lr-5em6-akl-0p03-eval-transformers"
    )

    combined = result.stdout + result.stderr
    assert "--time 5:00:00" in combined
    assert "PROPOSAL_GRPO_OBJECTIVE=grpo" in combined
    assert "PROPOSAL_GRPO_LEARNING_RATE=5e-6" in combined
    assert "PROPOSAL_GRPO_KL_COEF=0" in combined
    assert "PROPOSAL_GRPO_ANCHOR_KL_COEF=0.03" in combined


def test_adaptive_candidate_submitter_dry_run_dr_grpo_two_job_default_cell(tmp_path: Path):
    out_root = tmp_path / "adaptive_candidate_dr_grpo"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "LOG_DIR": str(tmp_path / "logs"),
            "PYTHON_BIN": sys.executable,
            "TASKS": "addition run_length",
            "CONDITIONS": "config",
            "OUTCOME_TRACE_TARGET_MODES": "numeric",
            "PROPOSAL_GRPO_REWARD_MODES": "outcome",
            "PROPOSAL_GRPO_ZERO_VARIANCE_MODES": "skip",
            "NUM_CANDIDATES_LIST": "8",
            "PROPOSAL_GRPO_OBJECTIVE": "dr_grpo",
            "PROPOSAL_GRPO_LEARNING_RATES": "1e-6",
            "PROPOSAL_GRPO_KL_COEF": "0",
            "PROPOSAL_GRPO_ANCHOR_KL_COEFS": "0.01",
            "MAX_ATTEMPT_ROUNDS": "25",
            "NO_SELECTION_PATIENCE": "25",
            "SBATCH_TIME": "02:30:00",
            "SBATCH_MEM": "48G",
            "ADAPTIVE_CONFIG_FILES": str(BASE_CONFIG),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((out_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"] == ["addition", "run_length"]
    assert manifest["proposal_grpo_objective"] == "dr_grpo"
    assert manifest["proposal_grpo_learning_rates"] == ["1e-6"]
    assert manifest["proposal_grpo_kl_coef"] == "0"
    assert manifest["proposal_grpo_anchor_kl_coefs"] == ["0.01"]
    assert len(manifest["jobs"]) == 2

    key = "addition-config-numeric-n8-reward-outcome-grpo-skip-obj-dr_grpo-lr-1e-6-akl-0.01-eval-transformers"
    assert manifest["jobs"][key]["proposal_grpo_objective"] == "dr_grpo"
    assert manifest["jobs"][key]["proposal_grpo_learning_rate"] == "1e-6"
    assert manifest["jobs"][key]["proposal_grpo_kl_coef"] == "0"
    assert manifest["jobs"][key]["proposal_grpo_anchor_kl_coef"] == "0.01"

    combined = result.stdout + result.stderr
    assert "--time 02:30:00" in combined
    assert "PROPOSAL_GRPO_OBJECTIVE=dr_grpo" in combined
    assert "PROPOSAL_GRPO_LEARNING_RATE=1e-6" in combined
    assert "PROPOSAL_GRPO_KL_COEF=0" in combined
    assert "PROPOSAL_GRPO_ANCHOR_KL_COEF=0.01" in combined
