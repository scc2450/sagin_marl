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
BASE_RUN="${BASE_RUN:-runs/phase2/bootstrap_fullrerun_3uav20gu_t250_k5_300u_seed45211_autodl_nocompile_20260605_001847}"
RUN_ROOT="${RUN_ROOT:-runs/phase2/matrix_4090_$(date +%Y%m%d_%H%M%S)}"
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
MATRIX_PRESET="${MATRIX_PRESET:-core}"
SCENARIO_PRESET="${SCENARIO_PRESET:-core}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sagin_marl_mpl}"
export SAGIN_MARL_NATIVE_CUDA_CACHE="${SAGIN_MARL_NATIVE_CUDA_CACHE:-$HOME/.cache/sagin_marl_native_cuda_cache}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMP_NUM_THREADS="${TORCH_THREADS}"

: "${DRY_RUN:?}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] CONFIG not found: ${CONFIG}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ! -d "${BASE_RUN}" ]]; then
  echo "[ERROR] BASE_RUN not found: ${BASE_RUN}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
MANIFEST_FILE="${RUN_ROOT}/matrix_manifest.csv"
echo "job_id,scenario,base_update,target_update,method,run_dir,checkpoint,command" > "${MANIFEST_FILE}"

if [[ "${MATRIX_PRESET}" == "core" ]]; then
  METHODS=(ppo_bw1e5 ppo_bw1e6 klguard_bw1e5 branchbest_bw1e5 freeze_bw)
elif [[ "${MATRIX_PRESET}" == "extended" ]]; then
  METHODS=(ppo_bw1e5 ppo_bw1e6 klguard_bw1e5 branchhard_bw1e5 branchbest_bw1e5 branchscore_bw1e5 freeze_bw)
else
  METHODS=(${MATRIX_PRESET})
fi

if [[ "${SCENARIO_PRESET}" == "core" ]]; then
  SCENARIOS=(
    "control_u240_to_u241|240|241|0"
    "control_u250_to_u260|250|260|0"
    "control_u250_to_u251|250|251|0"
    "control_u275_to_u300|275|300|0"
    "fixedtau_u275_to_u300|275|300|1"
  )
elif [[ "${SCENARIO_PRESET}" == "extended" ]]; then
  SCENARIOS=(
    "control_u200_to_u201|200|201|0"
    "control_u240_to_u241|240|241|0"
    "control_u240_to_u250|240|250|0"
    "control_u250_to_u251|250|251|0"
    "control_u250_to_u270|250|270|0"
    "control_u275_to_u300|275|300|0"
    "fixedtau_u250_to_u260|250|260|1"
    "fixedtau_u275_to_u300|275|300|1"
  )
else
  SCENARIOS=(${SCENARIO_PRESET})
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  _cuda_check="$("${PYTHON}" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
print(torch.cuda.device_count())
PY
)"
  if [[ "${_cuda_check}" != "1"* ]]; then
    echo "[ERROR] CUDA not available in python environment (${PYTHON})." >&2
    echo "Make sure this command runs on a CUDA machine and PYTHON points to the correct env." >&2
    exit 1
  fi
fi

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
    ppo_bw1e5)
      out_ref=( "${out_ref[@]}" --bw_actor_lr 1e-5 )
      ;;
    ppo_bw1e6)
      out_ref=( "${out_ref[@]}" --bw_actor_lr 1e-6 )
      ;;
    klguard_bw1e5)
      out_ref=(
        "${out_ref[@]}"
        --bw_actor_lr 1e-5
        --guarded_bw_update
        --guarded_bw_min_positive_credit -1
        --guarded_bw_min_credit_mean -1
        --guarded_bw_max_full_kl 0.03
        --guarded_bw_max_full_clip 0.25
      )
      ;;
    branchhard_bw1e5)
      out_ref=(
        "${out_ref[@]}"
        --bw_actor_lr 1e-5
        --guarded_bw_update
        --guarded_bw_min_positive_credit -1
        --guarded_bw_min_credit_mean -1
        --guarded_bw_max_full_kl 0.03
        --guarded_bw_max_full_clip 0.25
        --guarded_bw_branch_alignment
        --guarded_bw_min_branch_delta_logprob_product 0
        --guarded_bw_min_branch_positive_logprob_up_frac 0
        --guarded_bw_min_branch_corr_delta_logprob -1
      )
      ;;
    branchbest_bw1e5)
      out_ref=(
        "${out_ref[@]}"
        --bw_actor_lr 1e-5
        --guarded_bw_update
        --guarded_bw_min_positive_credit -1
        --guarded_bw_min_credit_mean -1
        --guarded_bw_max_full_kl 0.03
        --guarded_bw_max_full_clip 0.25
        --guarded_bw_branch_alignment
        --guarded_bw_min_branch_delta_logprob_product 0
        --guarded_bw_min_branch_positive_logprob_up_frac 0
        --guarded_bw_min_branch_corr_delta_logprob -1
        --guarded_bw_branch_fallback best_safe
        --guarded_bw_branch_fallback_min_product=-1e-4
      )
      ;;
    branchscore_bw1e5)
      out_ref=(
        "${out_ref[@]}"
        --bw_actor_lr 1e-5
        --guarded_bw_update
        --guarded_bw_min_positive_credit -1
        --guarded_bw_min_credit_mean -1
        --guarded_bw_max_full_kl 0.03
        --guarded_bw_max_full_clip 0.25
        --guarded_bw_branch_alignment
        --guarded_bw_min_branch_delta_logprob_product 0
        --guarded_bw_min_branch_positive_logprob_up_frac 0
        --guarded_bw_min_branch_corr_delta_logprob -1
        --guarded_bw_branch_accept_mode best_score
      )
      ;;
    freeze_bw)
      out_ref=( "${out_ref[@]}" --bw_actor_lr 0 )
      ;;
    *)
      echo "[ERROR] Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  printf '%s\n' "${out_ref[@]}"
}

function run_case() {
  local scenario="$1"
  local base_update="$2"
  local target_update="$3"
  local fixed_tau="$4"
  local method="$5"
  local idx="$6"

  local ckpt="${BASE_RUN}/checkpoint_update$(printf '%04d' "${base_update}").pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] checkpoint miss: ${ckpt} (command logged, skip actual run)" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      return
    fi
  fi

  local run_dir="${RUN_ROOT}/${scenario}_${method}_u$(printf '%04d' "${base_update}")to$(printf '%04d' "${target_update}")"

  local -a method_flags=()
  method_flags=()
  while IFS= read -r line; do
    method_flags+=( "${line}" )
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

  echo "${idx},${scenario},${base_update},${target_update},${method},${run_dir},${ckpt},${cmd_str}" >> "${MANIFEST_FILE}"

  echo "[RUN ${idx}] scenario=${scenario} method=${method} base=${base_update} target=${target_update} fixed_tau=${fixed_tau}"
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
}

job_counter=0
for scenario_entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r scenario_name base_update target_update fixed_tau <<< "${scenario_entry}"
  for method in "${METHODS[@]}"; do
    ((job_counter+=1))
    run_case "${scenario_name}" "${base_update}" "${target_update}" "${fixed_tau}" "${method}" "${job_counter}"
  done
done

echo "Matrix prepared in: ${RUN_ROOT}"
echo "Manifest: ${MANIFEST_FILE}"
