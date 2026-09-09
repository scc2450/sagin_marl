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
RUN_ROOT="${RUN_ROOT:-runs/phase2/h2_probe_4090_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-0}"

NUM_ENVS="${NUM_ENVS:-32}"
ROLLOUT_ENV_STEPS="${ROLLOUT_ENV_STEPS:-250}"
RETURN_TARGET="${RETURN_TARGET:-bootstrap_gae}"
RETURN_TARGET_SCHEDULE="${RETURN_TARGET_SCHEDULE:-fixed}"
COLD_CRITIC_EPOCHS="${COLD_CRITIC_EPOCHS:-1}"
TRACKING_CRITIC_EPOCHS="${TRACKING_CRITIC_EPOCHS:-3}"
CRITIC_MINIBATCHES="${CRITIC_MINIBATCHES:-100}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-1}"
ACTOR_MINIBATCHES="${ACTOR_MINIBATCHES:-100}"
SAVE_EVERY="${SAVE_EVERY:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"

BW_SAMPLE_LIMIT="${BW_SAMPLE_LIMIT:-256}"
BW_K_STEPS="${BW_K_STEPS:-20}"
BW_TRUE_MC_SAMPLES="${BW_TRUE_MC_SAMPLES:-2}"
BW_BRANCH_SAMPLES="${BW_BRANCH_SAMPLES:-1}"
BW_BRANCH_HORIZONS="${BW_BRANCH_HORIZONS:-2,5,10,20}"

# Default cases focus on checkpoints where BW degradation becomes visible.
# Format: case_name|base_key|base_update|target_update|fixed_tau|mode
# mode=probe_freeze: no actor learning; tests advantage-vs-counterfactual alignment only.
# mode=tiny_ppo: tiny BW PPO step; also tests whether delta_logprob follows branch_delta.
CASE_PRESET="${CASE_PRESET:-auto}"
MODE_PRESET="${MODE_PRESET:-probe_freeze tiny_ppo}"

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
  cuda_check="$("${PYTHON}" - <<'PY'
import torch
print('1' if torch.cuda.is_available() else '0')
print(torch.cuda.device_count())
PY
)"
  if [[ "${cuda_check}" != "1"* ]]; then
    echo "[ERROR] CUDA not available in python environment (${PYTHON})." >&2
    exit 1
  fi
fi

mkdir -p "${RUN_ROOT}"
MANIFEST_FILE="${RUN_ROOT}/h2_probe_manifest.csv"

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
  job_id case_name base_key base_update target_update fixed_tau mode run_dir checkpoint command

if [[ "${CASE_PRESET}" == "auto" ]]; then
  CASES=(
    "control_u150_to_u151|control|150|151|0"
    "control_u200_to_u201|control|200|201|0"
    "control_u250_to_u251|control|250|251|0"
    "fixedtau_u275_to_u276|fixedtau|275|276|1"
  )
else
  # shellcheck disable=SC2206
  CASES=(${CASE_PRESET})
fi

# shellcheck disable=SC2206
MODES=(${MODE_PRESET})

function base_run_for_key() {
  local base_key="$1"
  case "${base_key}" in
    control) printf '%s\n' "${CONTROL_BASE_RUN}" ;;
    fixedtau) printf '%s\n' "${FIXEDTAU_BASE_RUN}" ;;
    *) printf '%s\n' "${base_key}" ;;
  esac
}

function mode_args() {
  local mode="$1"
  local -a out=()
  out+=(
    --accel_actor_lr 0
    --sat_actor_lr 0
    --bw_advantage_alignment_probe
    --bw_advantage_alignment_sample_limit "${BW_SAMPLE_LIMIT}"
    --bw_advantage_alignment_k_steps "${BW_K_STEPS}"
    --bw_advantage_alignment_true_mc_samples "${BW_TRUE_MC_SAMPLES}"
    --bw_advantage_alignment_branch_samples "${BW_BRANCH_SAMPLES}"
    --bw_advantage_alignment_branch_horizons "${BW_BRANCH_HORIZONS}"
  )
  case "${mode}" in
    probe_freeze)
      out+=(--bw_actor_lr 0)
      ;;
    tiny_ppo)
      out+=(--bw_actor_lr 3e-7)
      ;;
    ppo_bw1e6)
      out+=(--bw_actor_lr 1e-6)
      ;;
    *)
      echo "[ERROR] Unknown mode: ${mode}" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "${out[@]}"
}

function run_case() {
  local case_name="$1"
  local base_key="$2"
  local base_update="$3"
  local target_update="$4"
  local fixed_tau="$5"
  local mode="$6"
  local idx="$7"

  local base_run
  base_run="$(base_run_for_key "${base_key}")"
  local ckpt="${base_run}/checkpoint_update$(printf '%04d' "${base_update}").pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] checkpoint missing: ${ckpt}" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      return
    fi
  fi

  local run_dir="${RUN_ROOT}/${case_name}_${mode}"
  local -a mode_flags=()
  while IFS= read -r line; do
    mode_flags+=("${line}")
  done < <(mode_args "${mode}")

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
  args+=("${mode_flags[@]}")

  local cmd_str=""
  for token in "${args[@]}"; do
    cmd_str+="${cmd_str:+ }$(printf '%q' "${token}")"
  done

  write_manifest_row \
    "${idx}" "${case_name}" "${base_key}" "${base_update}" "${target_update}" \
    "${fixed_tau}" "${mode}" "${run_dir}" "${ckpt}" "${cmd_str}"

  echo "[RUN ${idx}] case=${case_name} mode=${mode} base=${base_key}:${base_update} target=${target_update} fixed_tau=${fixed_tau}"
  echo "       run_dir=${run_dir}"
  echo "       ckpt=${ckpt}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "       cmd=${cmd_str}"
    return
  fi

  set -x
  "${args[@]}" 2>&1 | tee "${run_dir}.log"
  set +x
}

job_counter=0
for case_entry in "${CASES[@]}"; do
  IFS='|' read -r case_name base_key base_update target_update fixed_tau <<< "${case_entry}"
  for mode in "${MODES[@]}"; do
    ((job_counter += 1))
    run_case "${case_name}" "${base_key}" "${base_update}" "${target_update}" "${fixed_tau}" "${mode}" "${job_counter}"
  done
done

echo "H2 probe prepared in: ${RUN_ROOT}"
echo "Manifest: ${MANIFEST_FILE}"

if [[ "${SHUTDOWN_AFTER}" == "1" && "${DRY_RUN}" != "1" ]]; then
  echo "[INFO] SHUTDOWN_AFTER=1; requesting shutdown."
  /usr/bin/shutdown || true
fi
