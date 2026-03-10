# Line 3 Short-Train Validation

## Goal

Use short training runs to answer two questions before any long run:

1. Does `set_pool` beat the current `flat_mlp` actor under the same safety-followup setup?
2. Does the learned policy at least stop losing clearly to the `fixed` / `zero_accel` baseline?

Current code note:

- In [scripts/evaluate.py](d:/研三上/毕设/sagin_marl/scripts/evaluate.py), `fixed` and `zero_accel` are the same path.
- For this round, just use `--baseline fixed`.

## Metrics To Prioritize

For line 3 short validation, do not rank runs by `reward_sum` alone.

Primary metrics:

- `collision_rate_mean`
- `near_collision_ratio_mean`
- `min_inter_uav_dist_mean`
- `min_inter_uav_dist_min`
- `queue_total_active_mean`

Secondary metrics:

- `reward_sum_mean`
- `outflow_arrival_ratio_mean`
- `drop_ratio_mean`

The updated [summarize_policy_kpi.py](d:/研三上/毕设/sagin_marl/scripts/summarize_policy_kpi.py) now prints all of the metrics above.

## Run Groups

Use the same training length and the same evaluation seed base for all groups.

Recommended first pass:

- `updates=30`
- `save_interval=10`
- `episodes=20`
- `episode_seed_base=43000`

Run groups:

1. `flat30`
   - config: [configs/stage1_safety_followup/s1_safe_static_v2.yaml](d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v2.yaml)
   - purpose: current flat-MLP baseline
2. `setpool30`
   - config: [configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml](d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml)
   - purpose: new set-pooling actor, no actor warm start
3. `setpool30_initcritic`
   - config: [configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml](d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml)
   - purpose: isolate whether critic warm start helps the new actor

## Train Commands

Run from repo root after activating the virtual environment.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train.py --config configs/stage1_safety_followup/s1_safe_static_v2.yaml --updates 30 --save_interval 10 --run_dir runs/line3_short/flat30 --num_envs 12 --vec_backend subproc --torch_threads 2
```

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train.py --config configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml --updates 30 --save_interval 10 --run_dir runs/line3_short/setpool30 --num_envs 12 --vec_backend subproc --torch_threads 2
```

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train.py --config configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml --updates 30 --save_interval 10 --run_dir runs/line3_short/setpool30_initcritic --num_envs 12 --vec_backend subproc --torch_threads 2 --init_critic runs/ablation_followup/s1_safe_static_v2_warm250/critic.pt
```

Important:

- Do not use `--init_actor` with `setpool30` or `setpool30_initcritic`.
- The actor architecture changed; old flat checkpoints are not shape-compatible.

## Eval Commands

Evaluate the three learned runs with the same episode seeds, then run one shared `fixed` baseline CSV.

Why one shared `fixed` is enough in this round:

- `fixed` in [scripts/evaluate.py](d:/研三上/毕设/sagin_marl/scripts/evaluate.py) is the same as `zero_accel`.
- [configs/stage1_safety_followup/s1_safe_static_v2.yaml](d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v2.yaml) and [configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml](d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml) only differ in actor-encoder fields.
- The environment and reward setup are unchanged, and `fixed` does not use the actor.

### flat30

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate.py --config configs/stage1_safety_followup/s1_safe_static_v2.yaml --run_dir runs/line3_short/flat30 --episodes 20 --episode_seed_base 43000 --out runs/line3_short/flat30/eval_trained_seed43000.csv
```

### setpool30

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate.py --config configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml --run_dir runs/line3_short/setpool30 --episodes 20 --episode_seed_base 43000 --out runs/line3_short/setpool30/eval_trained_seed43000.csv
```

### setpool30_initcritic

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate.py --config configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml --run_dir runs/line3_short/setpool30_initcritic --episodes 20 --episode_seed_base 43000 --out runs/line3_short/setpool30_initcritic/eval_trained_seed43000.csv
```

### Shared fixed baseline

Use either `v2` or `v3` config. In this round they are equivalent for `fixed`.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate.py --config configs/stage1_safety_followup/s1_safe_static_v3_obs_encoder.yaml --baseline fixed --episodes 20 --episode_seed_base 43000 --out runs/line3_short/eval_fixed_shared_seed43000.csv
```

## Summary Commands

### Compare learned policies only

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/summarize_policy_kpi.py --input flat30=runs/line3_short/flat30/eval_trained_seed43000.csv setpool30=runs/line3_short/setpool30/eval_trained_seed43000.csv setpool30_initcritic=runs/line3_short/setpool30_initcritic/eval_trained_seed43000.csv
```

### Compare learned runs against the shared fixed baseline

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/summarize_policy_kpi.py --input flat30=runs/line3_short/flat30/eval_trained_seed43000.csv setpool30=runs/line3_short/setpool30/eval_trained_seed43000.csv setpool30_initcritic=runs/line3_short/setpool30_initcritic/eval_trained_seed43000.csv fixed=runs/line3_short/eval_fixed_shared_seed43000.csv
```

## Decision Rules

Use this order:

1. Safety first
   - Prefer lower `collision_rate_mean`
   - Prefer lower `near_collision_ratio_mean`
   - Prefer higher `min_inter_uav_dist_mean`
   - Prefer higher `min_inter_uav_dist_min`
2. Then queue stability
   - Prefer lower `queue_total_active_mean`
3. Then throughput and reward
   - Prefer higher `outflow_arrival_ratio_mean`
   - Prefer higher `reward_sum_mean`

Stop conditions for this round:

- If `set_pool` is still clearly worse than `fixed`, do not change reward or PPO settings yet.
- If `set_pool` beats `flat_mlp` on safety and is at least competitive on queue metrics, continue to a longer run.
- If `setpool30` and `setpool30_initcritic` are very close, keep the simpler no-warm-start setup.

## Episode-Level Follow-Up

This repo does not currently have a checked-in script for seed-locked debug CSV export.

What is available now:

- [scripts/evaluate.py](d:/研三上/毕设/sagin_marl/scripts/evaluate.py) for reproducible multi-episode CSV comparison
- [scripts/render_episode.py](d:/研三上/毕设/sagin_marl/scripts/render_episode.py) for a visual sanity check GIF
- [scripts/export_debug_episode.py](d:/研三上/毕设/sagin_marl/scripts/export_debug_episode.py) for seed-locked single-episode CSV + markdown summary export

Limit:

- `render_episode.py` does not support `episode_seed_base`, so it is not a replacement for the historical `debug_eval_epXX_seedXXXXbase.csv` workflow.
- `export_debug_episode.py` must replay resets from episode `0` up to the target index. Do not try to reproduce eval episode `k` by only using `seed=base+k` once.
  - Reason: the current env advances `episode_idx` on every `reset()`, and curriculum/adaptive episode state depends on that sequence.

For the current short-train gate, use the summary table first. When one run looks promising but still has ambiguous late-collision behavior, export the target episode with `export_debug_episode.py`.
