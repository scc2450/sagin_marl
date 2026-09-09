# Joint stage safety and training-speed check

## 1. Safety layer relation

The native CUDA accel step has two mutually replacing safety paths:

```text
regular avoidance:
  cfg.avoidance_enabled = true
  cfg.safety_shield_enabled = false

native safety shield:
  cfg.safety_shield_enabled = true
  cfg.safety_shield_solver = NATIVE_CUDA
  regular avoidance branch is bypassed
```

The CUDA branch point is:

```text
kParamSafetyShieldNative == 1 -> native_safety_project_env(...)
else                         -> accel_apply_avoidance(...)
```

So these should be treated as alternatives, not stacked safety layers.

## 2. Speed check

Command shape:

```text
scripts/benchmarks/profile_native_rollout_step_segments.py
  config = configs/tmp/structured_sat_mcgae_3uav_20gu_t250_positive_relcritic.yaml
  num_envs = 64
  rollout_env_steps = 40
  exec = accel policy, sat queue_aware, bw queue_aware
```

Results:

| mode | env steps/s | rollout time | total step ms/profiled step | accel_to_sat ms/profiled step |
| --- | ---: | ---: | ---: | ---: |
| regular avoidance | 574 | 4.46s | 9.60 | 2.16 |
| native safety shield | 560 | 4.57s | 9.38 | 2.06 |

Interpretation:

```text
native safety shield is not meaningfully faster in this setting.
The difference is within profiling noise / surrounding rollout variance.
Because native safety shield is the stronger safety layer and the speed is
similar, accel training should use native safety shield by default.
```

## 3. Three-stage ordinary PPO pressure check

Diagnostic config generated under:

```text
runs/diagnostics/joint_three_stage_pressure_20260509/joint_ppo_allstage_positive.yaml
```

Important scope:

```text
This was a train_structured.py ordinary PPO pressure test.
It was not the MC-GAE training recipe used by scripts/train_sat_mcgae.py /
scripts/train_stage_mcgae.py.

In the MC-GAE recipe, critic minibatches and actor minibatches are separate:
  critic_minibatches = 8
  actor_minibatches  = 1

In ordinary StructuredMAPPO.update(), num_mini_batch is shared by the critic
and actor update loops.
```

It enables:

```text
train_accel = true
train_sat   = true
train_bw    = true
exec all three stages = policy
reward_mode = positive_weighted_workload_level
native safety shield = true
regular avoidance = false
danger imitation = true, intervention_any
```

### Full-batch result

Command used:

```text
num_envs = 64
rollout_env_steps = 250
ppo_epochs = 1
num_mini_batch = 1
```

Result:

```text
CUDA OOM during critic value forward/backward.
Failing allocation included a [16000, 3, 20, 256] intermediate tensor.
```

Conclusion for ordinary PPO only:

```text
three-stage ordinary PPO with 64 env x 250 step and full-batch critic is not safe.
It can reproduce the previous "large batch blows up memory / becomes very slow" issue.
```

### Minibatched result

Command used:

```text
num_envs = 64
rollout_env_steps = 250
ppo_epochs = 1
num_mini_batch = 8
```

Result:

```text
no OOM
rollout_total_time_sec  = 14.8s
update_total_time_sec   = 258.4s
critic_train_sec        = 127.2s
actor_train_sec         = 31.2s
explained_variance_sec  = 51.7s
value_override_sec      = 32.6s
returns_sec             = 13.3s
```

Conclusion for ordinary PPO only:

```text
minibatching avoids OOM, but standard all-stage ordinary PPO is still too slow as-is.
The slow part is the critic/value path, not rollout.
```

This should not be used as a timing estimate for the intended MC-GAE recipe.
SAT MC-GAE timing is much lower after warmup because it does not run the same
ordinary PPO value override / train-GAE / full explained-variance path.

## 4. How to judge each stage in joint training

Use per-stage metrics, not only total reward:

```text
value_loss_accel / value_loss_sat / value_loss_bw
explained_variance_accel / explained_variance_sat / explained_variance_bw
entropy_accel / entropy_sat / entropy_bw
approx_kl_accel / approx_kl_sat / approx_kl_bw
clip_frac_accel / clip_frac_sat / clip_frac_bw
grad_norm_accel / grad_norm_sat / grad_norm_bw
danger_imitation_loss / danger_imitation_active_rate for accel
BW-specific kappa / log-ratio / auxiliary diagnostics when enabled
```

Important fix:

```text
PPO stage grad norm columns existed but were not filled in the normal PPO
stage-update branch. They are now populated from the stage actor update result.
```

Small smoke after the fix:

```text
num_envs = 8
rollout_env_steps = 20
num_mini_batch = 2

grad_norm_accel = 1.16
grad_norm_sat   = 0.77
grad_norm_bw    = 0.45
```

So the stage grad metrics can now be used as "is this head actually receiving
an update" checks.

## 5. Practical recommendation

For current experiments:

```text
do not run all-stage PPO with num_mini_batch = 1 at 64 x 250
do not treat ordinary train_structured.py joint PPO timing as MC-GAE timing
use the MC-GAE loop when testing the new critic/advantage recipe
judge learning with per-stage KL / clip / entropy / EV / grad norm
use native safety shield + danger imitation intervention_any for accel
keep regular avoidance disabled when native safety shield is enabled
```
