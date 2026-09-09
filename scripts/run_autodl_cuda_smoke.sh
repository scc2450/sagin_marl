#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml}"
RUN_DIR="${RUN_DIR:-runs/autodl_cuda_smoke_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"
COMPILE="${COMPILE:-0}"
UPDATES="${UPDATES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
ROLLOUT_ENV_STEPS="${ROLLOUT_ENV_STEPS:-2}"
COLD_CRITIC_EPOCHS="${COLD_CRITIC_EPOCHS:-1}"
TRACKING_CRITIC_EPOCHS="${TRACKING_CRITIC_EPOCHS:-1}"
CRITIC_MINIBATCHES="${CRITIC_MINIBATCHES:-1}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-1}"
ACTOR_MINIBATCHES="${ACTOR_MINIBATCHES:-1}"
TORCH_THREADS="${TORCH_THREADS:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sagin_marl_mpl}"
export SAGIN_MARL_NATIVE_CUDA_CACHE="${SAGIN_MARL_NATIVE_CUDA_CACHE:-$HOME/.cache/sagin_marl_native_cuda_cache}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

EFFECTIVE_CONFIG="${CONFIG}"
TMP_CONFIG=""
cleanup() {
  if [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]]; then
    rm -f "${TMP_CONFIG}"
  fi
}
trap cleanup EXIT

if [[ "${COMPILE}" == "0" ]]; then
  TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/sagin_marl_autodl_nocompile.XXXXXX.yaml")"
  "${PYTHON}" - "${CONFIG}" "${TMP_CONFIG}" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8-sig")
for key in (
    "critic_compile_enabled",
    "critic_compile_fullgraph",
    "stage_actor_compile_enabled",
    "accel_actor_compile_enabled",
    "sat_actor_compile_enabled",
    "bw_actor_compile_enabled",
):
    pattern = rf"^({re.escape(key)}\s*:\s*).*$"
    text, count = re.subn(pattern, rf"\1false", text, flags=re.MULTILINE)
    if count == 0:
        text += f"\n{key}: false\n"
dst.write_text(text, encoding="utf-8")
PY
  EFFECTIVE_CONFIG="${TMP_CONFIG}"
fi

echo "CONFIG=${EFFECTIVE_CONFIG}"
echo "RUN_DIR=${RUN_DIR}"
echo "SAGIN_MARL_NATIVE_CUDA_CACHE=${SAGIN_MARL_NATIVE_CUDA_CACHE}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

"${PYTHON}" scripts/train_joint_mcgae.py \
  --config "${EFFECTIVE_CONFIG}" \
  --run_dir "${RUN_DIR}" \
  --device cuda \
  --num_envs "${NUM_ENVS}" \
  --rollout_env_steps "${ROLLOUT_ENV_STEPS}" \
  --updates "${UPDATES}" \
  --structured_env_backend native \
  --structured_env_tensor_backend cuda \
  --cold_critic_epochs "${COLD_CRITIC_EPOCHS}" \
  --tracking_critic_epochs "${TRACKING_CRITIC_EPOCHS}" \
  --critic_minibatches "${CRITIC_MINIBATCHES}" \
  --actor_epochs "${ACTOR_EPOCHS}" \
  --actor_minibatches "${ACTOR_MINIBATCHES}" \
  --torch_threads "${TORCH_THREADS}" \
  --disable_stage_best_save
