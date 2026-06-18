#!/usr/bin/env bash
# Shared task-default resolution for Figure 2 recipe launchers.

figure2_task_key() {
  case "$1" in
    run_length) echo "RUN_LENGTH" ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_get_required_task_var() {
  local task="$1"
  local suffix="$2"
  local key
  local var_name
  key="$(figure2_task_key "${task}")"
  var_name="FIGURE2_${key}_${suffix}"
  if [[ -z "${!var_name:-}" ]]; then
    echo "[ERROR] Missing Figure 2 task config ${var_name} for task=${task}." >&2
    return 1
  fi
  echo "${!var_name}"
}

figure2_task_module() {
  figure2_get_required_task_var "$1" "MODULE"
}

figure2_archived_results_path() {
  figure2_get_required_task_var "$1" "ARCHIVED_RESULTS_PATH"
}

figure2_resolve_seed_model_for_task() {
  case "$1" in
    run_length)
      echo "${RUN_LENGTH_SEED_MODEL:-$(figure2_get_required_task_var run_length SEED_MODEL_DEFAULT)}"
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_default_expand_num_bits_for_task() {
  figure2_get_required_task_var "$1" "DEFAULT_EXPAND_NUM_BITS"
}

figure2_resolve_expand_train_per_bit_for_task() {
  case "$1" in
    run_length)
      if [[ -n "${RUN_LENGTH_EXPAND_TRAIN_PER_BIT:-}" ]]; then
        echo "${RUN_LENGTH_EXPAND_TRAIN_PER_BIT}"
      else
        echo "${EXPAND_TRAIN_PER_BIT}"
      fi
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_resolve_num_expand_rounds_for_task() {
  case "$1" in
    run_length)
      echo "${RUN_LENGTH_NUM_EXPAND_ROUNDS:-${NUM_EXPAND_ROUNDS}}"
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_resolve_expand_num_bits_for_task() {
  case "$1" in
    run_length)
      if [[ -n "${RUN_LENGTH_EXPAND_NUM_BITS:-}" ]]; then
        echo "${RUN_LENGTH_EXPAND_NUM_BITS}"
      elif [[ -n "${EXPAND_NUM_BITS:-}" ]]; then
        echo "${EXPAND_NUM_BITS}"
      else
        figure2_default_expand_num_bits_for_task run_length
      fi
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_resolve_bit_composition_path_mode_for_task() {
  case "$1" in
    run_length)
      if [[ -n "${RUN_LENGTH_BIT_COMPOSITION_PATH_MODE:-}" ]]; then
        echo "${RUN_LENGTH_BIT_COMPOSITION_PATH_MODE}"
      else
        echo "${BIT_COMPOSITION_PATH_MODE}"
      fi
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

figure2_resolve_target_mode_for_task() {
  case "$1" in
    run_length)
      if [[ -n "${RUN_LENGTH_TARGET_MODE:-}" ]]; then
        echo "${RUN_LENGTH_TARGET_MODE}"
      else
        echo "${TARGET_MODE}"
      fi
      ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}
