# AutoDL CUDA Runbook

This project now runs on the AutoDL RTX 4090 instance with the native CUDA
structured rollout path.

## Verified Instance

- GPU: NVIDIA RTX 4090, 24 GB VRAM
- Driver CUDA: 12.4
- PyTorch: 2.1.2+cu118
- Project path: `/root/autodl-tmp/sagin_marl`
- Verified command path: native CUDA rollout, actor/critic on CUDA

The native extension needed one portability fix: large rollout and actor ABI
structs are copied to CUDA constant symbols before kernel launches instead of
being passed by value. This avoids nvcc formal parameter space overflow on
Linux/RTX 4090.

## Quick Smoke

On the AutoDL machine:

```bash
cd /root/autodl-tmp/sagin_marl
PYTHON=/root/miniconda3/bin/python bash scripts/run_autodl_cuda_smoke.sh
```

The smoke defaults to:

- `COMPILE=0`
- `NUM_ENVS=1`
- `ROLLOUT_ENV_STEPS=2`
- `UPDATES=1`
- `structured_env_backend=native`
- `structured_env_tensor_backend=cuda`

Measured on the AutoDL 4090 after native extension cache was warm:

- `COMPILE=0`: about `1.2s` for `1 env x 2 steps x 1 update`
- `COMPILE=1`: about `91.8s` for the same tiny smoke because PyTorch graph
  compilation dominates small runs

## Longer Debug Run

```bash
cd /root/autodl-tmp/sagin_marl
PYTHON=/root/miniconda3/bin/python \
NUM_ENVS=4 \
ROLLOUT_ENV_STEPS=20 \
UPDATES=3 \
bash scripts/run_autodl_cuda_smoke.sh
```

## Compile Mode

Use the default `COMPILE=0` for smoke tests and quick debugging.

Use `COMPILE=1` when running long experiments where the initial compile cost can
be amortized. The script is still a smoke/debug entrypoint by default, so pass
the larger training schedule explicitly:

```bash
cd /root/autodl-tmp/sagin_marl
PYTHON=/root/miniconda3/bin/python \
COMPILE=1 \
NUM_ENVS=64 \
ROLLOUT_ENV_STEPS=250 \
UPDATES=300 \
COLD_CRITIC_EPOCHS=20 \
TRACKING_CRITIC_EPOCHS=5 \
CRITIC_MINIBATCHES=8 \
ACTOR_EPOCHS=5 \
ACTOR_MINIBATCHES=1 \
bash scripts/run_autodl_cuda_smoke.sh
```

## Cache

The script uses:

```bash
SAGIN_MARL_NATIVE_CUDA_CACHE=$HOME/.cache/sagin_marl_native_cuda_cache
TORCH_CUDA_ARCH_LIST=8.9
```

If the native CUDA sources change, either set a new cache path or remove the old
cache before re-running.
