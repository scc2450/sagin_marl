# 1 UAV Static BW Sanity Follow-up (2026-04-08)

## 1. Purpose

This note summarizes the work done **after** introducing the simplified `BW`-only sanity benchmark, what was fixed, what new benchmark variants were added, what the current status is, and what problem remains.

This follow-up supersedes the earlier interpretation in:

- `docs/bw_sanity_1uav_static_20260408.md`

The most important update is:

- the earlier "learned collapses to half of the baseline" conclusion was heavily affected by real execution/evaluation bugs
- after fixing those bugs, the original simplified benchmark is **trainable**
- but the original simplified benchmark is also **too weak** to clearly expose training problems
- after strengthening the benchmark gap, the training problem becomes visible again

## 2. Simplified System Definition

The simplified benchmark family is built around:

- `configs/structured_bw_sanity_1uav_static.yaml`

Core semantics:

- `num_uav: 1`
- `num_gu: 5`
- `train_accel: false`
- `train_sat: false`
- `train_bw: true`
- `exec_accel_source: zero`
- `exec_sat_source: zero`
- `exec_bw_source: policy`
- `fixed_satellite_strategy: true`

Interpretation:

- only one UAV exists
- the UAV does not move
- `accel` and `sat` are not trained
- `BW` is the only learned head
- satellites remain in the environment, but the benchmark is intended to behave like "`BW`-only with fixed sat behavior"

## 3. Files Involved

### 3.1 Core config files

- `configs/structured_bw_sanity_1uav_static.yaml`
- `configs/structured_bw_sanity_1uav_static_gap.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug.yaml`

### 3.2 Core code paths

- `sagin_marl/env/config.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_driver.py`
- `sagin_marl/rl/structured_actor.py`
- `sagin_marl/rl/structured_eval.py`
- `sagin_marl/rl/structured_train.py`

### 3.3 Training / evaluation / debug scripts

- `scripts/train_structured.py`
- `scripts/evaluate_structured.py`
- `scripts/train_structured_bw_actoronly_debug.py`
- `scripts/search_bw_sanity_gap.py`

## 4. What Was Fixed

### 4.1 Structured driver did not honor `fixed_satellite_strategy`

This was a real bug.

Before the fix:

- the normal environment step path respected fixed satellite behavior
- but the structured stage path still required an explicit SAT-stage action
- in the sanity benchmark, `exec_sat_source: zero` could therefore break the intended "`fixed sat`" semantics

Fix location:

- `sagin_marl/env/structured_driver.py`

Effect:

- `StructuredControlDriver.run_sat_stage()` now mirrors the fixed-satellite behavior instead of silently relying on a zero/empty sat action

### 4.2 Structured actor evaluation ignored `exec_*_source`

This was also a real bug.

Before the fix:

- checkpoint eval effectively re-enabled actor accel/sat heads
- so `BW`-only sanity benchmark checkpoints were not being evaluated under the intended "`zero accel + zero sat + policy bw`" semantics

Fix locations:

- `sagin_marl/rl/structured_eval.py`
- `scripts/evaluate_structured.py`
- `scripts/train_structured.py`

Effect:

- fair checkpoint/model eval now supports the intended `exec_accel_source / exec_sat_source / exec_bw_source`

## 5. What the Original Simplified Benchmark Looked Like After Fixes

The main fair PPO rerun after the bug fixes is:

- `runs/structured_bw_sanity_1uav_static_v2_ppo_fix_u20_20260408`

Reference fixed baseline during checkpoint eval:

- `reward = 46.5519`
- `processed = 1.0036`
- `drop = 0.0000`
- `backlog = 0.5826`

From:

- `runs/structured_bw_sanity_1uav_static_v2_ppo_fix_u20_20260408/checkpoint_eval.csv`

Checkpoint eval:

- `u10`: `45.9616`
- `u20`: `47.5471`

Final fair eval:

- deterministic `46.1594`
- stochastic `44.7026`

From:

- `runs/structured_bw_sanity_1uav_static_v2_ppo_fix_u20_20260408/eval_det_fair_ep12.csv`
- `runs/structured_bw_sanity_1uav_static_v2_ppo_fix_u20_20260408/eval_stoch_fair_ep12.csv`

This changes the earlier conclusion substantially:

- the original "v2" simplified benchmark is **not** catastrophically broken once the execution/eval bugs are fixed
- standard PPO can train it to roughly the fixed baseline level

## 6. Actor-only Debug Path

To isolate whether the full PPO/critic path was the problem, an actor-only debug trainer was added:

- `scripts/train_structured_bw_actoronly_debug.py`

Run:

- `runs/structured_bw_sanity_1uav_static_v2_actoronly_u20_20260408`

Results from:

- `runs/structured_bw_sanity_1uav_static_v2_actoronly_u20_20260408/summary.json`

Key numbers:

- fixed `queue_aware_bw`: `45.9489`
- initial det: `41.9708`
- final det: `44.6399`
- final stoch: `43.2960`

Interpretation:

- actor-only debug did improve over initialization
- but it did **not** beat the corrected standard PPO rerun

So the simplified benchmark does **not** support the claim that "PPO/critic is the main cause" of failure here.

## 7. Why the Original Simplified Benchmark Was Still Not Good Enough

The problem with the corrected original benchmark is not catastrophic failure anymore.

The problem is that the benchmark gap is too small.

Under fair stochastic fixed-policy evaluation:

- `zero_bw` around `42.24`
- `queue_aware_bw` around `45.95`

This is only about `+3.7` reward on one representative seed block.

That means:

- even if training is healthy, the stochastic learning curve can still look flat
- this benchmark is not strong enough to decide whether the training stack is really healthy

This is why a benchmark-gap search was added.

## 8. Benchmark-gap Search

Script:

- `scripts/search_bw_sanity_gap.py`

Outputs:

- `runs/_tmp_bw_sanity_gap_search.csv`
- `runs/_tmp_bw_sanity_gap_search_seed53000.csv`
- `runs/_tmp_bw_sanity_gap_search_seed54000.csv`

### 8.1 Old weak benchmark (`base_v2`)

Reward gap `queue_aware_bw - zero_bw`:

- seed `52000`: `0.522`
- seed `53000`: `3.691`
- seed `54000`: `1.830`

### 8.2 Stronger benchmark (`sticky_hotspot_rho8_tightbw`)

Reward gap `queue_aware_bw - zero_bw`:

- seed `52000`: `33.622`
- seed `53000`: `27.177`
- seed `54000`: `34.076`

This is the first version that clearly satisfies the intended gate:

- the fixed-policy gap is large
- the gap is stable across multiple seeds

## 9. New High-gap Benchmark

New config:

- `configs/structured_bw_sanity_1uav_static_gap.yaml`

Main changes relative to the old simplified benchmark:

- `traffic_model: sticky_subset_hotspot`
- `arrival_base_hetero: 0.9`
- `hotspot_num_subsets: 4`
- `hotspot_subset_size: 2`
- `hotspot_rho: 8.0`
- `hotspot_on_mean_steps: 20`
- `hotspot_off_mean_steps: 5`
- `queue_init_gu_steps: 5.0`
- `gu_init_cluster_std: 320.0`
- `task_arrival_rate: 8.0e5`
- `b_acc: 4.0e6`

Interpretation:

- the benchmark now has large and persistent per-user pressure heterogeneity
- `BW` has materially more leverage
- if training is bad, it should now be visible

## 10. Debug Training Config on the High-gap Benchmark

Debug config:

- `configs/structured_bw_sanity_1uav_static_gap_debug.yaml`

Additional changes relative to `structured_bw_sanity_1uav_static_gap.yaml`:

- `checkpoint_eval_fixed_policy: queue_aware_bw`
- `checkpoint_eval_policy_mode: stochastic`
- `stagewise_advantage_norm_enabled: true`
- `structured_bw_loc_readout: fused`
- `entropy_coef: 0.0`
- `entropy_coef_bw: 0.0`
- `structured_bw_kappa_init: 32.0`
- `structured_bw_kappa_min: 16.0`
- `structured_bw_kappa_max: 64.0`

Purpose:

- use a stronger benchmark
- reduce policy noise
- use a stronger `BW` readout
- make checkpoint eval track the **same stochastic execution mode** as training
- use `queue_aware_bw` as the fixed BW-only reference

## 11. Current Debug Run and Result

Run:

- `runs/structured_bw_sanity_1uav_static_gap_debug_u10_20260408`

### 11.1 Fixed reference

From training console and:

- `runs/structured_bw_sanity_1uav_static_gap_debug_u10_20260408/checkpoint_eval.csv`

Fixed `queue_aware_bw` reference:

- `reward = 25.9332`
- `processed = 0.9109`
- `drop = 0.0430`
- `backlog = 8.0802`

### 11.2 Learned checkpoint

At `u10`:

- `reward = 1.5819`
- `processed = 0.7504`
- `drop = 0.1720`
- `backlog = 10.7007`

From:

- `runs/structured_bw_sanity_1uav_static_gap_debug_u10_20260408/checkpoint_eval.csv`

### 11.3 Training metrics

From:

- `runs/structured_bw_sanity_1uav_static_gap_debug_u10_20260408/metrics.csv`

Observations:

- `episode_reward` stays very low:
  - update 1: `0.3188`
  - update 5: `3.5876`
  - update 10: `4.7882`
- `episode_reward_std` stays huge:
  - roughly `21.9 ~ 23.6`
- `rollout_reward_per_step` stays near zero:
  - update 1: `0.0032`
  - update 10: `0.0111`

Interpretation:

- under this stronger benchmark, the current training stack is **clearly failing**
- this time the failure is no longer hidden by weak benchmark headroom
- the result is not ambiguous

## 12. Current Progress

What is already done:

1. built a very simplified `BW`-only benchmark
2. found and fixed two real execution/evaluation bugs
3. reran standard PPO and actor-only debug under fair semantics
4. showed that the corrected original simplified benchmark is trainable but too weak
5. added a fixed-policy benchmark-gap search
6. found a stronger benchmark regime with stable large `queue_aware_bw - zero_bw` gap
7. ran a first debug PPO training on that stronger benchmark

## 13. Current Problem

The current problem is now much more precise than before.

It is **not**:

- "maybe the curve is flat only because the benchmark gap is too small"

because the new benchmark already fixes that.

It is also **not yet proven to be**:

- "PPO/critic is the unique root cause"

because the old weak benchmark does not support that conclusion, and the new high-gap benchmark has only been run with standard PPO so far.

The current precise problem is:

- once the benchmark has enough true `BW` leverage,
- the current `BW` training setup still fails badly,
- even after fixing the structured sat/eval bugs,
- and even after making the checkpoint metric track the stochastic execution mode.

## 14. What This Follow-up Supports

This follow-up supports the following claims:

1. The earlier catastrophic collapse on the simplified benchmark was partly caused by real code bugs.
2. After fixing those bugs, the original simplified benchmark is trainable, but too weak to be a useful debug benchmark.
3. The strengthened high-gap benchmark is the first simplified benchmark that can reliably expose training problems.
4. Under that stronger benchmark, the current `BW` training still looks seriously wrong.

## 15. Current Best Use of This Benchmark

The recommended use of the current high-gap debug benchmark is:

- do **not** use it to claim "`BW` is solved"
- do use it as a **debug benchmark** for:
  - actor output inspection
  - PPO update path inspection
  - advantage / normalization ablations
  - readout / policy noise diagnosis

In short:

- `structured_bw_sanity_1uav_static.yaml` is now mostly a weak sanity check
- `structured_bw_sanity_1uav_static_gap.yaml` and `structured_bw_sanity_1uav_static_gap_debug.yaml` are the useful versions for diagnosing why `BW` still fails once the task has enough signal
