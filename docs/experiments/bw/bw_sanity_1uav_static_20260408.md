# 1 UAV Static BW Sanity Benchmark Notes (2026-04-08)

## Purpose

This note summarizes the simplified `BW`-only sanity benchmark built on 2026-04-08, why it was introduced, what exactly was changed between the two versions, and why the second version exposed a much more serious training failure than the first.

The original goal of this benchmark was:

- strip away `accel/sat` coupling
- remove movement and multi-UAV interaction
- keep only `BW` training
- check whether the current `BW` training stack can learn at all in a much simpler setting

This benchmark did **not** solve the original `BW` training problem. Instead, it showed that:

- the first simplified version was too easy to be informative
- once the task was made even modestly meaningful for `BW`, the current training stack collapsed badly

## Current simplified system

The current simplified benchmark is defined by [configs/structured_bw_sanity_1uav_static.yaml](/d:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static.yaml).

Main properties:

- `num_uav: 1`
- `num_gu: 5`
- `uav_spawn_mode: gu_centroid`
- `uav_init_speed_frac: 0.0`
- `train_accel: false`
- `train_sat: false`
- `train_bw: true`
- `exec_accel_source: zero`
- `exec_sat_source: zero`
- `exec_bw_source: policy`
- `fixed_satellite_strategy: true`
- `reward_w_relay: 0.0`
- `b_sat_total: 1.0e8`
- `sat_cpu_freq: 1.0e12`
- `structured_bw_parameterization: score_alpha_kappa_dirichlet`
- `structured_bw_per_uav_surrogate_enabled: true`

Interpretation:

- there is only one UAV
- the UAV does not move
- `accel` and `sat` are effectively removed from learning
- `sat` still exists in the environment, but satellite choice is fixed and backhaul reward is removed
- the only learned action is the `BW` share over the currently associated users

This is still **not** a pure queueing toy problem. It reuses the existing structured stack and environment, but tries to make `BW` the only trainable part.

## Files involved

Environment/config support added for this benchmark:

- [sagin_marl/env/config.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py)
  - added `uav_spawn_mode`
  - added `gu_init_num_clusters`
  - added `gu_init_cluster_std`
- [sagin_marl/env/sagin_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py)
  - added `gu_centroid` UAV spawning
  - made GU cluster count/std configurable at reset

Training/eval entrypoints reused:

- [scripts/train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py)
- [scripts/evaluate_structured.py](/d:/研三上/毕设/sagin_marl/scripts/evaluate_structured.py)

Run artifacts discussed below:

- [structured_bw_sanity_1uav_static_u20_20260408](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_u20_20260408)
- [structured_bw_sanity_1uav_static_v2_u30_20260408](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408)

## Version 1: too easy to be informative

Version 1 is captured by [runs/structured_bw_sanity_1uav_static_u20_20260408/config.yaml](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_u20_20260408/config.yaml).

Key settings:

- `traffic_model: homogeneous`
- `task_arrival_rate: 6.0e5`
- `gu_init_num_clusters: 1`
- `gu_init_cluster_std: 60.0`
- `b_acc: 1.0e7`

The checkpoint evaluation file is [checkpoint_eval.csv](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_u20_20260408/checkpoint_eval.csv).

Important numbers:

- fixed `queue_aware` reference during training:
  - `reward_sum = 50.5`
  - `processed_ratio_eval = 1.01`
  - `drop_ratio_eval = 0.0`
- learned checkpoint `u10`:
  - `reward_sum = 48.15`
  - `processed_ratio_eval = 0.9991`
  - `drop_ratio_eval = 0.0`
- learned checkpoint `u20`:
  - `reward_sum = 48.06`
  - `processed_ratio_eval = 0.9983`
  - `drop_ratio_eval = 0.0`

At first glance this looked "not too far" from the fixed reference.

That was misleading.

The real lesson from Version 1 is:

- this benchmark was so loose that `BW` barely mattered
- being close to the fixed reference here did **not** mean `BW` had learned a meaningful policy
- it mostly meant many `BW` allocations produced nearly the same outcome

In other words, Version 1 failed as a sanity benchmark because it did not create a stable, meaningful `BW` decision problem.

## Version 2: slightly harder, immediately exposes failure

Version 2 is captured by [runs/structured_bw_sanity_1uav_static_v2_u30_20260408/config.yaml](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408/config.yaml).

Only two meaningful changes were made relative to Version 1:

- `gu_init_cluster_std: 60.0 -> 250.0`
- `b_acc: 1.0e7 -> 5.0e6`

Intent:

- spread GU geometry so access conditions differ more
- tighten total access bandwidth so `BW` allocation matters more

Everything else important stayed the same:

- still `1` UAV
- still static
- still only train `BW`
- still fixed satellite strategy
- still no backhaul reward

## Fixed baselines in Version 2

The saved fixed-baseline comparison is [fixed_baselines_seed52000_ep24.json](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408/fixed_baselines_seed52000_ep24.json).

Results:

- `queue_aware`
  - `reward_sum = 46.78`
  - `processed_ratio_eval = 1.0044`
  - `drop_ratio_eval = 0.0`
  - `pre_backlog_steps_eval = 0.5536`
- `zero`
  - `reward_sum = 46.06`
  - `processed_ratio_eval = 0.9930`
  - `drop_ratio_eval = 0.00240`
  - `pre_backlog_steps_eval = 0.8523`

So the archived `24`-episode gap is only about `0.73` reward.

This means:

- `queue_aware` is better than `zero`
- but the gap is still small
- and from interactive checks during development, it was also unstable across seeds / episode budgets

So even Version 2 is **not yet** a clean "large-headroom" benchmark.

## Training collapse in Version 2

The key file is [checkpoint_eval.csv](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408/checkpoint_eval.csv).

Fixed `queue_aware` reference used during checkpoint eval:

- `reward_sum = 46.55`
- `processed_ratio_eval = 1.0036`
- `drop_ratio_eval = 0.0`
- `pre_backlog_steps_eval = 0.5826`

Learned policy checkpoints:

- `u10`
  - `reward_sum = 25.62`
  - `processed_ratio_eval = 0.8208`
  - `drop_ratio_eval = 0.0377`
  - `pre_backlog_steps_eval = 6.1150`
- `u20`
  - `reward_sum = 24.66`
  - `processed_ratio_eval = 0.8162`
  - `drop_ratio_eval = 0.0428`
  - `pre_backlog_steps_eval = 6.2102`
- `u30`
  - `reward_sum = 21.42`
  - `processed_ratio_eval = 0.7991`
  - `drop_ratio_eval = 0.0552`
  - `pre_backlog_steps_eval = 6.8352`

This is not "slightly worse".

This is a **strong collapse**:

- reward falls to less than half of the fixed reference
- processed ratio drops sharply
- drop ratio rises from `0.0` to `0.055`
- backlog explodes from about `0.58` to `6.84`

## Final deterministic / stochastic evaluation in Version 2

To check whether this was only a deterministic readout problem, the final actor was also evaluated in both deterministic and stochastic modes.

Files:

- [eval_det_seed53000_ep12_h128e32.csv](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408/eval_det_seed53000_ep12_h128e32.csv)
- [eval_stoch_seed53000_ep12_h128e32.csv](/d:/研三上/毕设/sagin_marl/runs/structured_bw_sanity_1uav_static_v2_u30_20260408/eval_stoch_seed53000_ep12_h128e32.csv)

Means over 12 episodes:

- deterministic reward mean: `22.04`
- stochastic reward mean: `16.86`

So Version 2 is **not** failing only because the deterministic readout is bad.

The policy itself is bad:

- deterministic is poor
- stochastic is even worse

## What changed between the two versions

The difference can be summarized very simply:

- Version 1 made `BW` almost irrelevant
- Version 2 made `BW` matter a bit more
- once `BW` started to matter, the current training stack failed badly

This is the central fact that needs to be preserved.

It is **not** correct to describe Version 1 as a near-success.
It is also **not** correct to describe Version 2 as just "harder".

Version 2 is the first version that meaningfully tested whether the current `BW` training stack can cope once allocation has some leverage.
The answer, so far, is **no**.

## What this does and does not show

What it does show:

- the original full-system `BW` failure is not just a fake artifact of multi-head coupling
- even in a drastically simplified setting, the current `BW` training stack can still fail badly
- once `BW` has even modest leverage, the current training pipeline can learn a policy much worse than a simple fixed rule

What it does **not** show:

- it does not isolate the single root cause
- it does not prove the benchmark itself is already ideal
- it does not prove the fixed `queue_aware` baseline has a large, stable advantage
- it does not prove whether the failure is mainly due to PPO signal quality, policy parameterization, normalization, objective scaling, or some hidden implementation issue

## Current diagnosis

The strongest current diagnosis from this sanity benchmark is:

- Version 1 was too easy to diagnose anything
- Version 2 is the first meaningful sanity check
- Version 2 reveals a serious training problem, because the learned policy collapses far below even a weakly better fixed baseline

So the right takeaway is not:

- "we just need to tune Version 2 more"

The right takeaway is:

- this simplified benchmark has now become strong enough to expose a real training failure
- that failure is severe enough that it deserves direct investigation
- it should not be hand-waved away as only a "small headroom" issue

## Recommended next use of this benchmark

This benchmark should now be treated as a **debug benchmark**, not a success benchmark.

It is useful for answering questions like:

- is there a hidden implementation bug in `BW` training?
- is PPO update quality for `BW` obviously broken even in a single-UAV case?
- does stochastic training collapse before deterministic deployment is even relevant?

It is **not yet** a good benchmark for claiming:

- `BW` is trainable from scratch
- learned `BW` can beat heuristic `BW`

That claim still has not been established.
