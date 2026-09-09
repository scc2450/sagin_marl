# Mac CPU Structured Rollout

This branch adds a portable Python/CPU structured rollout path for machines
without NVIDIA CUDA. It is intended for smoke tests, debugging, and small local
experiments on macOS. It does not replace the native CUDA rollout used for the
thesis-scale training run.

## What It Runs

- `structured_env_backend=python`
- `structured_env_tensor_backend=cpu`
- `PythonStructuredDriverGroup` with `SaginParallelEnv` and `StructuredControlDriver`
- `StructuredMAPPO.collect_env_steps(...)` through the three Python stages:
  accel policy, SAT subset policy, BW policy
- Joint MC-GAE training through `scripts/train_joint_mcgae.py`

The native CUDA safety shield is disabled automatically for Python/CPU runs,
because `safety_shield_solver=NATIVE_CUDA` only exists in the fused CUDA rollout.

## Quick Smoke

```bash
bash scripts/run_mac_joint_smoke.sh
```

The script uses the current thesis-style config and overrides only the runtime
backend and small smoke-test sizes.

You can scale it with environment variables:

```bash
UPDATES=1 ROLLOUT_ENV_STEPS=20 NUM_ENVS=1 bash scripts/run_mac_joint_smoke.sh
```

## Direct Command

```bash
conda run -n rl python scripts/train_joint_mcgae.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --run_dir runs/mac_joint_smoke \
  --device cpu \
  --num_envs 1 \
  --rollout_env_steps 20 \
  --updates 1 \
  --structured_env_backend python \
  --structured_env_tensor_backend cpu \
  --cold_critic_epochs 1 \
  --tracking_critic_epochs 1 \
  --critic_minibatches 1 \
  --actor_epochs 1 \
  --actor_minibatches 1 \
  --torch_threads 1 \
  --disable_stage_best_save
```

## Expected Local Speed

Measured on the current Mac CPU path:

- rollout only, `1 env x 20 steps`: about `0.32s`, roughly `62 env steps/s`
- one tiny training update, `1 env x 2 steps`, one critic/actor epoch: about `19s`
- one tiny training update, `1 env x 20 steps`, one critic/actor epoch: about `26s`

The full thesis recipe uses `64 envs x 250 steps x 300 updates` plus much larger
critic/actor schedules, so it should still be run on an NVIDIA CUDA machine.
