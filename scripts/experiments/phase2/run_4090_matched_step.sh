#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_DIR}"

CONFIG="${CONFIG:-configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml}"
CONTROL_BASE_RUN="${CONTROL_BASE_RUN:-runs/phase2/bootstrap_fullrerun_3uav20gu_t250_k5_300u_seed45211_autodl_nocompile_20260605_001847}"
FIXEDTAU_BASE_RUN="${FIXEDTAU_BASE_RUN:-runs/phase2/bootstrap_fixed_tau1_from_u200_to_u300_seed45211_autodl_20260605_015958}"
MATCHED_PHASE="${MATCHED_PHASE:-calibration}"
RUN_ROOT="${RUN_ROOT:-runs/phase2/matched_step_4090_${MATCHED_PHASE}_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
SAVE_EVERY="${SAVE_EVERY:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"
NUM_ENVS="${NUM_ENVS:-32}"
ROLLOUT_ENV_STEPS="${ROLLOUT_ENV_STEPS:-250}"
RETURN_TARGET="${RETURN_TARGET:-bootstrap_gae}"
RETURN_TARGET_SCHEDULE="${RETURN_TARGET_SCHEDULE:-fixed}"
COLD_CRITIC_EPOCHS="${COLD_CRITIC_EPOCHS:-1}"
TRACKING_CRITIC_EPOCHS="${TRACKING_CRITIC_EPOCHS:-3}"
CRITIC_MINIBATCHES="${CRITIC_MINIBATCHES:-100}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-3}"
ACTOR_MINIBATCHES="${ACTOR_MINIBATCHES:-100}"
BW_SAMPLE_LIMIT="${BW_SAMPLE_LIMIT:-64}"
BW_K_STEPS="${BW_K_STEPS:-5}"
BW_BRANCH_HORIZONS="${BW_BRANCH_HORIZONS:-2,5,10}"
BW_LOCAL_HORIZON="${BW_LOCAL_HORIZON:-20}"
BW_LOCAL_SAMPLE_LIMIT="${BW_LOCAL_SAMPLE_LIMIT:-${BW_SAMPLE_LIMIT}}"
BACKTRACK_FACTORS="${BACKTRACK_FACTORS:-1,0.7,0.5,0.3,0.1,0.03,0.01,0.003}"
GUARD_MAX_CLIP="${GUARD_MAX_CLIP:-0.25}"
BRANCH_FALLBACK_MIN_PRODUCT="${BRANCH_FALLBACK_MIN_PRODUCT:--1e-4}"
METHOD_PRESET="${METHOD_PRESET:-auto}"
SCENARIO_PRESET="${SCENARIO_PRESET:-auto}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"
EVAL_AFTER="${EVAL_AFTER:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-930000}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-64}"
SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sagin_marl_mpl}"
export SAGIN_MARL_NATIVE_CUDA_CACHE="${SAGIN_MARL_NATIVE_CUDA_CACHE:-$HOME/.cache/sagin_marl_native_cuda_cache}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMP_NUM_THREADS="${TORCH_THREADS}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] CONFIG not found: ${CONFIG}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  for base_run in "${CONTROL_BASE_RUN}" "${FIXEDTAU_BASE_RUN}"; do
    if [[ ! -d "${base_run}" ]]; then
      echo "[ERROR] Base run not found: ${base_run}" >&2
      exit 1
    fi
  done
  _cuda_check="$("${PYTHON}" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
print(torch.cuda.device_count())
PY
)"
  if [[ "${_cuda_check}" != "1"* ]]; then
    echo "[ERROR] CUDA not available in python environment (${PYTHON})." >&2
    exit 1
  fi
fi

mkdir -p "${RUN_ROOT}"
MANIFEST_FILE="${RUN_ROOT}/matched_step_manifest.csv"

function csv_cell() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

function write_manifest_row() {
  local first=1
  for value in "$@"; do
    if [[ "${first}" == "0" ]]; then
      printf ',' >> "${MANIFEST_FILE}"
    fi
    csv_cell "${value}" >> "${MANIFEST_FILE}"
    first=0
  done
  printf '\n' >> "${MANIFEST_FILE}"
}

write_manifest_row \
  job_id phase scenario base_key base_update target_update fixed_tau method run_dir checkpoint command

if [[ "${METHOD_PRESET}" == "auto" ]]; then
  if [[ "${MATCHED_PHASE}" == "calibration" ]]; then
    METHODS=(
      ppo_bw1e5
      ppo_bw3e6
      ppo_bw1e6
      ppo_bw3e7
      klguard_bw1e5_kl003
      klguard_bw1e5_kl0015
      klguard_bw1e5_kl00075
      branchbest_bw1e5_kl003
      branchbest_bw1e5_kl0015
      branchbest_bw1e5_kl00075
      branchscore_bw1e5_kl003
      branchscore_bw1e5_kl0015
      branchscore_bw1e5_kl00075
      branchhard_bw1e5_kl0015
      bwlocal_branch_h20
      bwlocal_mix05_h20
      bwlocal_signgate_h20
      freeze_bw
    )
  elif [[ "${MATCHED_PHASE}" == "continuation" ]]; then
    METHODS=(
      ppo_bw1e6
      klguard_bw1e5_kl0015
      branchbest_bw1e5_kl0015
      branchscore_bw1e5_kl0015
      bwlocal_branch_h20
      bwlocal_mix05_h20
      freeze_bw
    )
  else
    echo "[ERROR] Unknown MATCHED_PHASE=${MATCHED_PHASE}; use calibration or continuation." >&2
    exit 1
  fi
else
  # shellcheck disable=SC2206
  METHODS=(${METHOD_PRESET})
fi

if [[ "${SCENARIO_PRESET}" == "auto" ]]; then
  if [[ "${MATCHED_PHASE}" == "calibration" ]]; then
    SCENARIOS=(
      "control_u250_to_u251|control|250|251|0"
      "fixedtau_u275_to_u276|fixedtau|275|276|1"
    )
  elif [[ "${MATCHED_PHASE}" == "continuation" ]]; then
    SCENARIOS=(
      "control_u250_to_u270|control|250|270|0"
      "fixedtau_u275_to_u300|fixedtau|275|300|1"
    )
  fi
else
  # shellcheck disable=SC2206
  SCENARIOS=(${SCENARIO_PRESET})
fi

function base_run_for_key() {
  local base_key="$1"
  case "${base_key}" in
    control) printf '%s\n' "${CONTROL_BASE_RUN}" ;;
    fixedtau) printf '%s\n' "${FIXEDTAU_BASE_RUN}" ;;
    *) printf '%s\n' "${base_key}" ;;
  esac
}

function guarded_common_args() {
  local lr="$1"
  local max_kl="$2"
  printf '%s\n' \
    --bw_actor_lr "${lr}" \
    --guarded_bw_update \
    --guarded_bw_backtrack_factors "${BACKTRACK_FACTORS}" \
    --guarded_bw_min_positive_credit -1 \
    --guarded_bw_min_credit_mean -1 \
    --guarded_bw_max_full_kl "${max_kl}" \
    --guarded_bw_max_full_clip "${GUARD_MAX_CLIP}"
}

function branch_common_args() {
  printf '%s\n' \
    --guarded_bw_branch_alignment \
    --guarded_bw_min_branch_delta_logprob_product 0 \
    --guarded_bw_min_branch_positive_logprob_up_frac 0 \
    --guarded_bw_min_branch_corr_delta_logprob -1
}

function method_args() {
  local method="$1"
  local -a out_ref=()

  out_ref+=(
    --accel_actor_lr 0
    --sat_actor_lr 0
    --bw_advantage_alignment_probe
    --bw_advantage_alignment_sample_limit "${BW_SAMPLE_LIMIT}"
    --bw_advantage_alignment_k_steps "${BW_K_STEPS}"
    --bw_advantage_alignment_true_mc_samples 0
    --bw_advantage_alignment_branch_samples 1
    --bw_advantage_alignment_branch_horizons "${BW_BRANCH_HORIZONS}"
  )

  case "${method}" in
    ppo_bw1e5) out_ref+=(--bw_actor_lr 1e-5) ;;
    ppo_bw3e6) out_ref+=(--bw_actor_lr 3e-6) ;;
    ppo_bw1e6) out_ref+=(--bw_actor_lr 1e-6) ;;
    ppo_bw3e7) out_ref+=(--bw_actor_lr 3e-7) ;;
    klguard_bw1e5_kl003)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.03)
      ;;
    klguard_bw1e5_kl0015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.015)
      ;;
    klguard_bw1e5_kl00075)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.0075)
      ;;
    branchhard_bw1e5_kl0015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      ;;
    branchbest_bw1e5_kl003)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.03)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_fallback best_safe --guarded_bw_branch_fallback_min_product="${BRANCH_FALLBACK_MIN_PRODUCT}")
      ;;
    branchbest_bw1e5_kl0015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_fallback best_safe --guarded_bw_branch_fallback_min_product="${BRANCH_FALLBACK_MIN_PRODUCT}")
      ;;
    branchbest_bw1e5_kl00075)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.0075)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_fallback best_safe --guarded_bw_branch_fallback_min_product="${BRANCH_FALLBACK_MIN_PRODUCT}")
      ;;
    branchscore_bw1e5_kl003)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.03)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_accept_mode best_score)
      ;;
    branchscore_bw1e5_kl0015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.015)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_accept_mode best_score)
      ;;
    branchscore_bw1e5_kl00075)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(guarded_common_args 1e-5 0.0075)
      while IFS= read -r line; do out_ref+=("${line}"); done < <(branch_common_args)
      out_ref+=(--guarded_bw_branch_accept_mode best_score)
      ;;
    bwlocal_branch_h20)
      out_ref+=(
        --bw_actor_lr 1e-5
        --bw_local_advantage_update
        --bw_local_advantage_mode branch
        --bw_local_advantage_horizon "${BW_LOCAL_HORIZON}"
        --bw_local_advantage_sample_limit "${BW_LOCAL_SAMPLE_LIMIT}"
        --bw_local_advantage_normalization scale
      )
      ;;
    bwlocal_mix05_h20)
      out_ref+=(
        --bw_actor_lr 1e-5
        --bw_local_advantage_update
        --bw_local_advantage_mode mix
        --bw_local_advantage_alpha 0.5
        --bw_local_advantage_horizon "${BW_LOCAL_HORIZON}"
        --bw_local_advantage_sample_limit "${BW_LOCAL_SAMPLE_LIMIT}"
        --bw_local_advantage_normalization scale
      )
      ;;
    bwlocal_signgate_h20)
      out_ref+=(
        --bw_actor_lr 1e-5
        --bw_local_advantage_update
        --bw_local_advantage_mode sign_gate
        --bw_local_advantage_horizon "${BW_LOCAL_HORIZON}"
        --bw_local_advantage_sample_limit "${BW_LOCAL_SAMPLE_LIMIT}"
        --bw_local_advantage_normalization scale
      )
      ;;
    freeze_bw) out_ref+=(--bw_actor_lr 0) ;;
    *)
      echo "[ERROR] Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  printf '%s\n' "${out_ref[@]}"
}

function maybe_eval_run() {
  local run_dir="$1"
  local label="$2"
  if [[ "${EVAL_AFTER}" != "1" ]]; then
    return
  fi
  if [[ ! -f "${run_dir}/final.pt" ]]; then
    echo "[WARN] final checkpoint missing; skip eval: ${run_dir}/final.pt" >&2
    return
  fi
  local seeds="${EVAL_SEEDS//,/ }"
  for seed in ${seeds}; do
    local eval_dir="${run_dir}/native_eval_seed${seed}"
    echo "[EVAL] ${label} seed=${seed}"
    "${PYTHON}" scripts/evaluate_structured_mixed_heads_native.py \
      --config "${CONFIG}" \
      --base_checkpoint "${run_dir}/final.pt" \
      --episodes "${EVAL_EPISODES}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --episode_seed_base "${seed}" \
      --policy_mode deterministic \
      --device cuda \
      --exec_accel_source policy \
      --exec_sat_source policy \
      --exec_bw_source policy \
      --access_bw_decision_interval 5 \
      --sat_decision_interval 1 \
      --out_dir "${eval_dir}" \
      --label "eval_seed${seed}"
  done
}

function run_case() {
  local scenario="$1"
  local base_key="$2"
  local base_update="$3"
  local target_update="$4"
  local fixed_tau="$5"
  local method="$6"
  local idx="$7"

  local base_run
  base_run="$(base_run_for_key "${base_key}")"
  local ckpt="${base_run}/checkpoint_update$(printf '%04d' "${base_update}").pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] checkpoint miss: ${ckpt} (command logged, skip actual run)" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      return
    fi
  fi

  local run_dir="${RUN_ROOT}/${scenario}_${method}_u$(printf '%04d' "${base_update}")to$(printf '%04d' "${target_update}")"
  local -a method_flags=()
  while IFS= read -r line; do
    method_flags+=("${line}")
  done < <(method_args "${method}")

  local -a args=(
    "${PYTHON}" scripts/train_joint_mcgae.py
    --config "${CONFIG}"
    --run_dir "${run_dir}"
    --resume "${ckpt}"
    --updates "${target_update}"
    --device cuda
    --num_envs "${NUM_ENVS}"
    --rollout_env_steps "${ROLLOUT_ENV_STEPS}"
    --return_target "${RETURN_TARGET}"
    --return_target_schedule "${RETURN_TARGET_SCHEDULE}"
    --cold_critic_epochs "${COLD_CRITIC_EPOCHS}"
    --tracking_critic_epochs "${TRACKING_CRITIC_EPOCHS}"
    --critic_minibatches "${CRITIC_MINIBATCHES}"
    --actor_epochs "${ACTOR_EPOCHS}"
    --actor_minibatches "${ACTOR_MINIBATCHES}"
    --save_every "${SAVE_EVERY}"
    --torch_threads "${TORCH_THREADS}"
    --disable_stage_best_save
    --structured_env_backend native
    --structured_env_tensor_backend cuda
  )

  if [[ "${fixed_tau}" == "1" ]]; then
    args+=(--bw_fixed_tau 1.0 --bw_fixed_kappa 32.0)
  fi
  if [[ "${DISABLE_TORCH_COMPILE}" == "1" ]]; then
    args+=(--disable_torch_compile)
  fi
  args+=("${method_flags[@]}")

  local cmd_str=""
  for token in "${args[@]}"; do
    cmd_str+="${cmd_str:+ }$(printf '%q' "${token}")"
  done

  write_manifest_row \
    "${idx}" "${MATCHED_PHASE}" "${scenario}" "${base_key}" "${base_update}" "${target_update}" \
    "${fixed_tau}" "${method}" "${run_dir}" "${ckpt}" "${cmd_str}"

  echo "[RUN ${idx}] phase=${MATCHED_PHASE} scenario=${scenario} method=${method} base=${base_key}:${base_update} target=${target_update} fixed_tau=${fixed_tau}"
  echo "       run_dir=${run_dir}"
  echo "       ckpt=${ckpt}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "       cmd=${cmd_str}"
    return
  fi

  mkdir -p "$(dirname "${run_dir}")"
  set -x
  "${args[@]}" 2>&1 | tee "${run_dir}.log"
  set +x
  maybe_eval_run "${run_dir}" "${scenario}_${method}"
}

job_counter=0
for scenario_entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r scenario_name base_key base_update target_update fixed_tau <<< "${scenario_entry}"
  for method in "${METHODS[@]}"; do
    ((job_counter += 1))
    run_case "${scenario_name}" "${base_key}" "${base_update}" "${target_update}" "${fixed_tau}" "${method}" "${job_counter}"
  done
done

echo "Matched-step ${MATCHED_PHASE} prepared in: ${RUN_ROOT}"
echo "Manifest: ${MANIFEST_FILE}"

if [[ "${SHUTDOWN_AFTER}" == "1" && "${DRY_RUN}" != "1" ]]; then
  echo "[INFO] SHUTDOWN_AFTER=1; requesting shutdown."
  /usr/bin/shutdown || true
fi
