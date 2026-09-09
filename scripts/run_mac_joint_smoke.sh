#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml}"
RUN_DIR="${RUN_DIR:-runs/mac_joint_smoke}"
UPDATES="${UPDATES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
ROLLOUT_ENV_STEPS="${ROLLOUT_ENV_STEPS:-2}"
COLD_CRITIC_EPOCHS="${COLD_CRITIC_EPOCHS:-1}"
TRACKING_CRITIC_EPOCHS="${TRACKING_CRITIC_EPOCHS:-1}"
CRITIC_MINIBATCHES="${CRITIC_MINIBATCHES:-1}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-1}"
ACTOR_MINIBATCHES="${ACTOR_MINIBATCHES:-1}"
TORCH_THREADS="${TORCH_THREADS:-1}"
DISABLE_CHECKPOINT_EVAL="${DISABLE_CHECKPOINT_EVAL:-1}"

CHECKPOINT_EVAL_ARGS=()
if [[ "${DISABLE_CHECKPOINT_EVAL}" != "0" ]]; then
  CHECKPOINT_EVAL_ARGS+=(--disable_checkpoint_eval)
fi

conda run -n rl python scripts/train_joint_mcgae.py \
  --config "${CONFIG}" \
  --run_dir "${RUN_DIR}" \
  --device cpu \
  --num_envs "${NUM_ENVS}" \
  --rollout_env_steps "${ROLLOUT_ENV_STEPS}" \
  --updates "${UPDATES}" \
  --structured_env_backend python \
  --structured_env_tensor_backend cpu \
  --cold_critic_epochs "${COLD_CRITIC_EPOCHS}" \
  --tracking_critic_epochs "${TRACKING_CRITIC_EPOCHS}" \
  --critic_minibatches "${CRITIC_MINIBATCHES}" \
  --actor_epochs "${ACTOR_EPOCHS}" \
  --actor_minibatches "${ACTOR_MINIBATCHES}" \
  --torch_threads "${TORCH_THREADS}" \
  --disable_stage_best_save \
  "${CHECKPOINT_EVAL_ARGS[@]}"
