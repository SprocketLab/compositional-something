#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
source "${SCRIPT_DIR}/lib/figure2_recipe_common.sh"

self_cd_repo_root

FIGURE2_RECIPE_DEFAULT_CONFIG="${ROOT_DIR}/launchers/self/config/figure2_recipe_aggressive.env"
self_source_config_file "${FIGURE2_RECIPE_DEFAULT_CONFIG}" "Figure 2 recipe default config"
if [[ -n "${FIGURE2_RECIPE_CONFIG:-}" ]]; then
  self_source_config_files "${FIGURE2_RECIPE_CONFIG}" "Figure 2 recipe override config"
fi

FIGURE2_TASK_CONFIG="${FIGURE2_TASK_CONFIG:-${ROOT_DIR}/launchers/self/config/figure2_run_length.env}"
self_source_config_files "${FIGURE2_TASK_CONFIG}" "Figure 2 task config"

PAPER_SCHEDULE_ENV="${PAPER_SCHEDULE_ENV:-${ROOT_DIR}/artifacts/paper/paper_schedule_selection.env}"
if [[ -f "${PAPER_SCHEDULE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PAPER_SCHEDULE_ENV}"
fi

self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure2_recipe_aggressive_${TS}}"
FIGURE_DIR="${FIGURE_DIR:-${ROOT_DIR}/icmlw26_comp-self-improvement/figures}"
BASELINES="${FIGURE2_RECIPE_BASELINES_RAW}"
SEED="${SEED:-42}"
SCRATCH_MODEL_NAME="${SCRATCH_MODEL_NAME:-${ROOT_DIR}/meta/models/tiny_gpt2_8l_384d}"
RUN_FULLPACK_ONLY_IF_HEALTHY="${RUN_FULLPACK_ONLY_IF_HEALTHY:-1}"
RUN_PILOT_GATE="${RUN_PILOT_GATE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
MAX_STEPS_SEED="${MAX_STEPS_SEED:-}"
MAX_STEPS_ROUND="${MAX_STEPS_ROUND:-}"
INITIAL_TRAIN_PER_BIT="${INITIAL_TRAIN_PER_BIT:-100000}"
INITIAL_EVAL_PER_BIT="${INITIAL_EVAL_PER_BIT:-50}"
EXPAND_TRAIN_PER_BIT="${EXPAND_TRAIN_PER_BIT:-1200}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-8}"
EXPAND_NUM_BITS="${EXPAND_NUM_BITS:-}"
RUN_LENGTH_NUM_EXPAND_ROUNDS="${RUN_LENGTH_NUM_EXPAND_ROUNDS:-}"
RUN_LENGTH_EXPAND_NUM_BITS="${RUN_LENGTH_EXPAND_NUM_BITS:-}"
RUN_LENGTH_EXPAND_TRAIN_PER_BIT="${RUN_LENGTH_EXPAND_TRAIN_PER_BIT:-}"
COMPOSED_EVAL_PER_BIT="${COMPOSED_EVAL_PER_BIT:-50}"
BIT_COMPOSITION_PATH_MODE="${BIT_COMPOSITION_PATH_MODE:-random}"
RUN_LENGTH_BIT_COMPOSITION_PATH_MODE="${RUN_LENGTH_BIT_COMPOSITION_PATH_MODE:-}"
TARGET_MODE="${TARGET_MODE:-default}"
RUN_LENGTH_TARGET_MODE="${RUN_LENGTH_TARGET_MODE:-}"
FIGURE_STYLE="${FIGURE_STYLE:-paper}"
FIGURE_FONT_SCALE="${FIGURE_FONT_SCALE:-1.5}"
FIGURE_CANVAS_WIDTH="${FIGURE_CANVAS_WIDTH:-7.4}"
FIGURE_CANVAS_HEIGHT="${FIGURE_CANVAS_HEIGHT:-9.0}"

read -r -a TASK_LIST <<< "${TASKS}"
read -r -a BASELINE_LIST <<< "${BASELINES}"

mkdir -p "${OUT_ROOT}"

append_common_batch_overrides() {
  local -n ref="$1"
  if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
    ref+=(--per-device-train-batch-size "${TRAIN_BATCH_SIZE}")
  fi
  if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
    ref+=(--per-device-eval-batch-size "${EVAL_BATCH_SIZE}")
  fi
}

run_cmd() {
  local -a cmd=("$@")
  self_print_command_stdout "${cmd[@]}"
  if self_parse_bool "${DRY_RUN}"; then
    return 0
  fi
  "${cmd[@]}"
}

check_seed_gate() {
  local task="$1"
  local path="${OUT_ROOT}/${task}/seed/seed_fit_results.json"
  if self_parse_bool "${DRY_RUN}"; then
    return 0
  fi
  PYTHONPATH=. "${PYTHON_BIN}" - <<'PY' "${path}" "${task}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
payload = json.loads(path.read_text())
validation_min = payload.get("validation_min_per_size_accuracy")
test_min = payload.get("test_min_per_size_accuracy")
if validation_min is None or test_min is None or validation_min < 0.99 or test_min < 0.99:
    raise SystemExit(
        f"[ERROR] Seed gate failed for {task}: validation_min={validation_min} test_min={test_min}"
    )
print(
    f"[INFO] Seed gate passed for {task}: validation_min={validation_min:.4f} test_min={test_min:.4f}",
    flush=True,
)
PY
}

check_pilot_gate() {
  local task="$1"
  local pilot_path="${OUT_ROOT}/${task}/pilot/compose/self_improvement_results.json"
  local archived_path
  archived_path="$(figure2_archived_results_path "${task}")"
  if self_parse_bool "${DRY_RUN}"; then
    return 0
  fi
  PYTHONPATH=. "${PYTHON_BIN}" - <<'PY' "${pilot_path}" "${archived_path}" "${task}"
import json
import sys
from pathlib import Path

pilot_path = Path(sys.argv[1])
archived_path = Path(sys.argv[2])
task = sys.argv[3]

pilot = json.loads(pilot_path.read_text())
archived = json.loads(archived_path.read_text())

def max_size(row):
    value = row.get("max_bits")
    if value is None:
        value = row.get("max_size")
    return int(value)

pilot32 = next((row for row in pilot if max_size(row) == 32), None)
archived32 = next((row for row in archived if max_size(row) == 32), None)
pilot_final = max(pilot, key=max_size) if pilot else None
if pilot32 is None or archived32 is None or pilot_final is None:
    raise SystemExit(f"[ERROR] Missing 32-bit or final pilot rows while checking pilot gate for {task}.")

pilot32_acc = float(pilot32["eval_accuracy"])
archived32_acc = float(archived32["eval_accuracy"])
pilot_final_size = max_size(pilot_final)
pilot_final_acc = float(pilot_final["eval_accuracy"])

if pilot32_acc < archived32_acc - 0.02:
    raise SystemExit(
        f"[ERROR] Pilot gate failed for {task}: 32-bit acc {pilot32_acc:.4f} is below archived "
        f"{archived32_acc:.4f} by more than 0.02."
    )
if pilot_final_acc < 0.20:
    raise SystemExit(
        f"[ERROR] Pilot gate failed for {task}: final frontier acc at {pilot_final_size} bits "
        f"({pilot_final_acc:.4f}) is below the 0.20 non-collapse threshold."
    )
print(
    f"[INFO] Pilot gate passed for {task}: 32-bit acc={pilot32_acc:.4f} archived32={archived32_acc:.4f} "
    f"final frontier {pilot_final_size}-bit acc={pilot_final_acc:.4f}",
    flush=True,
)
PY
}

run_seed_fit_task() {
  local task="$1"
  local out_dir="${OUT_ROOT}/${task}/seed"
  local -a cmd=(
    "${PYTHON_BIN}" -m self.experiments.seed_fit_experiment
    --task "${task}"
    --model-name "${SCRATCH_MODEL_NAME}"
    --output-dir "${out_dir}"
    --format-version legacy
    --initial-min-size 8
    --initial-max-size 16
    --initial-train-per-size "${INITIAL_TRAIN_PER_BIT}"
    --initial-eval-per-size "${INITIAL_EVAL_PER_BIT}"
    --target-mode "$(figure2_resolve_target_mode_for_task "${task}")"
    --target-accuracy-threshold 0.99
    --init-from-scratch
    --recipe algorithmic_self_improve_v1
    --bucket-train-batches-by-size
    --save-model
    --seed "${SEED}"
  )
  append_common_batch_overrides cmd
  if [[ -n "${MAX_STEPS_SEED}" ]]; then
    cmd+=(--max-steps "${MAX_STEPS_SEED}")
  fi
  run_cmd "${cmd[@]}"
  check_seed_gate "${task}"
}

run_compose_pilot_task() {
  local task="$1"
  local seed_model="${OUT_ROOT}/${task}/seed/model"
  local out_dir="${OUT_ROOT}/${task}/pilot/compose"
  local module
  local num_expand_rounds
  local expand_num_bits
  local expand_train_per_bit
  local bit_composition_path_mode
  module="$(figure2_task_module "${task}")"
  num_expand_rounds="$(figure2_resolve_num_expand_rounds_for_task "${task}")"
  expand_num_bits="$(figure2_resolve_expand_num_bits_for_task "${task}")"
  expand_train_per_bit="$(figure2_resolve_expand_train_per_bit_for_task "${task}")"
  bit_composition_path_mode="$(figure2_resolve_bit_composition_path_mode_for_task "${task}")"
  if [[ ! -d "${seed_model}" ]]; then
    seed_model="$(figure2_resolve_seed_model_for_task "${task}")"
  fi
  if ! self_parse_bool "${DRY_RUN}" && [[ ! -d "${seed_model}" ]]; then
    echo "[ERROR] Missing seed model for ${task}: ${seed_model}" >&2
    exit 1
  fi
  local -a cmd=(
    "${PYTHON_BIN}" -m "${module}"
    --model-name "${seed_model}"
    --output-dir "${out_dir}"
    --format-version legacy
    --target-mode "$(figure2_resolve_target_mode_for_task "${task}")"
    --initial-min-bits 8
    --initial-max-bits 16
    --initial-train-per-bit "${INITIAL_TRAIN_PER_BIT}"
    --initial-eval-per-bit "${INITIAL_EVAL_PER_BIT}"
    --num-expand-rounds "${num_expand_rounds}"
    --expand-num-bits "${expand_num_bits}"
    --expand-train-per-bit "${expand_train_per_bit}"
    --eval-per-bit 50
    --composed-eval-per-bit "${COMPOSED_EVAL_PER_BIT}"
    --pseudo-label-mode compose
    --bit-composition-path-mode "${bit_composition_path_mode}"
    --recipe algorithmic_self_improve_v1
    --bucket-train-batches-by-bits
    --resume
    --seed "${SEED}"
  )
  append_common_batch_overrides cmd
  if [[ -n "${MAX_STEPS_ROUND}" ]]; then
    cmd+=(--max-steps "${MAX_STEPS_ROUND}")
  fi
  run_cmd "${cmd[@]}"
  if [[ "${RUN_PILOT_GATE}" == "1" ]]; then
    check_pilot_gate "${task}"
  else
    echo "[INFO] RUN_PILOT_GATE=0; skipping pilot gate for ${task}."
  fi
}

run_fullpack_task() {
  local task="$1"
  local seed_model="${OUT_ROOT}/${task}/seed/model"
  local module
  local num_expand_rounds
  local expand_num_bits
  local expand_train_per_bit
  local bit_composition_path_mode
  module="$(figure2_task_module "${task}")"
  num_expand_rounds="$(figure2_resolve_num_expand_rounds_for_task "${task}")"
  expand_num_bits="$(figure2_resolve_expand_num_bits_for_task "${task}")"
  expand_train_per_bit="$(figure2_resolve_expand_train_per_bit_for_task "${task}")"
  bit_composition_path_mode="$(figure2_resolve_bit_composition_path_mode_for_task "${task}")"
  if [[ ! -d "${seed_model}" ]]; then
    seed_model="$(figure2_resolve_seed_model_for_task "${task}")"
  fi
  if [[ "${RUN_FULLPACK_ONLY_IF_HEALTHY}" == "1" ]]; then
    check_pilot_gate "${task}"
  fi
  local baseline
  for baseline in "${BASELINE_LIST[@]}"; do
    local out_dir="${OUT_ROOT}/${task}/fullpack/${baseline}"
    local -a cmd=(
      "${PYTHON_BIN}" -m "${module}"
      --model-name "${seed_model}"
      --output-dir "${out_dir}"
      --format-version legacy
      --target-mode "$(figure2_resolve_target_mode_for_task "${task}")"
      --initial-min-bits 8
      --initial-max-bits 16
      --initial-train-per-bit "${INITIAL_TRAIN_PER_BIT}"
      --initial-eval-per-bit "${INITIAL_EVAL_PER_BIT}"
      --num-expand-rounds "${num_expand_rounds}"
      --expand-num-bits "${expand_num_bits}"
      --expand-train-per-bit "${expand_train_per_bit}"
      --eval-per-bit 50
      --composed-eval-per-bit "${COMPOSED_EVAL_PER_BIT}"
      --bit-composition-path-mode "${bit_composition_path_mode}"
      --recipe algorithmic_self_improve_v1
      --bucket-train-batches-by-bits
      --resume
      --seed "${SEED}"
    )
    case "${baseline}" in
      short_only) cmd+=(--pseudo-label-mode none) ;;
      direct) cmd+=(--pseudo-label-mode direct) ;;
      compose) cmd+=(--pseudo-label-mode compose) ;;
      compose_corrupt) cmd+=(--pseudo-label-mode compose_corrupt --corruption-rate 0.10) ;;
      *)
        echo "[ERROR] Unsupported baseline=${baseline}" >&2
        exit 1
        ;;
    esac
    append_common_batch_overrides cmd
    if [[ -n "${MAX_STEPS_ROUND}" ]]; then
      cmd+=(--max-steps "${MAX_STEPS_ROUND}")
    fi
    run_cmd "${cmd[@]}"
  done
}

render_figures_task() {
  local task="$1"
  local results_path="${OUT_ROOT}/${task}/fullpack/compose/self_improvement_results.json"
  if [[ ! -f "${results_path}" ]]; then
    results_path="${OUT_ROOT}/${task}/pilot/compose/self_improvement_results.json"
  fi
  if self_parse_bool "${DRY_RUN}"; then
    printf '[INFO] Figure refresh task=%s results=%s figure_dir=%s\n' "${task}" "${results_path}" "${FIGURE_DIR}"
    return 0
  fi
  mkdir -p "${FIGURE_DIR}"
  PYTHONPATH=. "${PYTHON_BIN}" - <<'PY' "${results_path}" "${task}" "${FIGURE_DIR}" "${FIGURE_STYLE}" "${FIGURE_FONT_SCALE}" "${FIGURE_CANVAS_WIDTH}" "${FIGURE_CANVAS_HEIGHT}"
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from self.analysis.training_curves import plot_per_size_accuracy_heatmap_from_results

results_path = Path(sys.argv[1])
task = sys.argv[2]
figure_dir = Path(sys.argv[3])
figure_style = sys.argv[4]
font_scale = float(sys.argv[5])
canvas_width = float(sys.argv[6])
canvas_height = float(sys.argv[7])

titles = {
    "run_length": "Per-bit accuracy heatmap -- run-length compose",
}
basenames = {
    "run_length": "run_length_self_improvement_heatmap",
}
paper_tick_stride = {
    "run_length": 4,
}
paper_dense_ticks_through = {
    "run_length": 16,
}
common_kwargs = {}
if figure_style == "paper":
    common_kwargs = {
        "annotate_mode": "sparse",
        "show_title": False,
        "show_round_max_labels": False,
        "y_tick_stride": paper_tick_stride[task],
        "dense_y_ticks_through": paper_dense_ticks_through[task],
        "font_scale": font_scale,
        "fixed_canvas_size": (canvas_width, canvas_height),
    }
fig = plot_per_size_accuracy_heatmap_from_results(
    results_path,
    task=task,
    mode="compose",
    title=titles[task],
    **common_kwargs,
)
base = figure_dir / basenames[task]
fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Wrote figure bundle to {base}", flush=True)
PY
}

run_stage_for_task() {
  local stage="$1"
  local task="$2"
  case "${stage}" in
    seed) run_seed_fit_task "${task}" ;;
    pilot) run_compose_pilot_task "${task}" ;;
    fullpack) run_fullpack_task "${task}" ;;
    figure) render_figures_task "${task}" ;;
    *)
      echo "[ERROR] Unsupported stage=${stage}" >&2
      exit 1
      ;;
  esac
}

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Figure dir" "${FIGURE_DIR}" \
  "Device target" "${DEVICE_TARGET}" \
  "Stage" "${STAGE}" \
  "Tasks" "${TASKS}" \
  "Baselines" "${BASELINES}" \
  "Run pilot gate" "${RUN_PILOT_GATE}" \
  "Dry run" "${DRY_RUN}" \
  "Paper schedule env" "${PAPER_SCHEDULE_ENV}" \
  "Initial train per bit" "${INITIAL_TRAIN_PER_BIT}" \
  "Initial eval per bit" "${INITIAL_EVAL_PER_BIT}" \
  "Composed eval per bit" "${COMPOSED_EVAL_PER_BIT}" \
  "Default bit composition path mode" "${BIT_COMPOSITION_PATH_MODE}" \
  "Default target mode" "${TARGET_MODE}" \
  "Figure style" "${FIGURE_STYLE}"
for task in "${TASK_LIST[@]}"; do
  echo "[INFO] Task schedule ${task}: seed_model=$(figure2_resolve_seed_model_for_task "${task}") num_expand_rounds=$(figure2_resolve_num_expand_rounds_for_task "${task}") expand_num_bits=$(figure2_resolve_expand_num_bits_for_task "${task}") expand_train_per_bit=$(figure2_resolve_expand_train_per_bit_for_task "${task}") bit_composition_path_mode=$(figure2_resolve_bit_composition_path_mode_for_task "${task}") target_mode=$(figure2_resolve_target_mode_for_task "${task}")"
done

case "${STAGE}" in
  seed|pilot|fullpack|figure)
    for task in "${TASK_LIST[@]}"; do
      run_stage_for_task "${STAGE}" "${task}"
    done
    ;;
  all)
    for task in "${TASK_LIST[@]}"; do
      run_seed_fit_task "${task}"
      run_compose_pilot_task "${task}"
      run_fullpack_task "${task}"
      render_figures_task "${task}"
    done
    ;;
  *)
    echo "[ERROR] Unsupported STAGE=${STAGE}" >&2
    exit 1
    ;;
esac

echo "[INFO] Finished Figure 2 recipe aggressive workflow."
