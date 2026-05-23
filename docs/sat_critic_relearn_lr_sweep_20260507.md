# SAT critic relearn LR sweep 2026-05-07

## Purpose

Continue the SAT fixed-policy critic audit at the point where it was interrupted:

```text
MC-trained critic -> normalized A_gae(V) full-stage SAT actor update -> collect pi1 rollout -> relearn critic on pi1 MC targets
```

This run only checks the post-actor-update critic relearn stability under different critic learning rates.  It skips the expensive Qpi branch probe.

## Shared setup

```text
config: configs/tmp/structured_single_sat_3uav_20gu_t250_ppo.yaml
stage: sat
reward_mode: positive_weighted_workload_level
num_envs: 64
rollout_env_steps: 250
target: MC
pre_train_rollouts: 1
pre_heldout_rollouts: 1
post_train_rollouts: 1
post_heldout_rollouts: 1
pre_critic_epochs: 20
refit_critic_on_actor_batch_epochs: 20
actor_update_kind: gae
actor_update_full_ppo: true
actor_update_epochs: 5
actor_update_minibatches: 1
actor_lr: 3e-4
actor_advantage_normalize_enabled: true
stagewise_advantage_norm_enabled: true
skip_actor_update_qpi_probe: true
Vhat probe: rows=8, policy_action_samples=2, continuations=2
```

The earlier `rows=24, actions=4, continuations=4` Vhat probe hit CUDA OOM in branch replay on the 64-env run, so this sweep uses the smaller Vhat probe above.

## Results

| relearn lr | epoch 0 Vhat EV | epoch 5 Vhat EV | epoch 10 Vhat EV | epoch 20 Vhat EV | epoch 30 Vhat EV | note |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-3` | 0.970 | 0.976 | 0.938 | 0.949 | 0.963 | stable enough |
| `5e-4` | 0.968 | 0.966 | 0.978 | 0.905 | 0.880 | late drop |
| `3e-4` | 0.971 | 0.963 | 0.963 | 0.959 | 0.962 | most stable |

Heldout MC EV also stayed high for all three:

| relearn lr | epoch 0 MC EV | epoch 5 MC EV | epoch 20 MC EV | epoch 30 MC EV |
| ---: | ---: | ---: | ---: | ---: |
| `1e-3` | 0.961 | 0.940 | 0.959 | 0.961 |
| `5e-4` | 0.956 | 0.957 | 0.941 | 0.941 |
| `3e-4` | 0.955 | 0.958 | 0.953 | 0.955 |

Mean/MAE behavior is the more important part for the LR decision:

| relearn lr | epoch | Vhat pred mean | Vhat target mean | residual mean | Vhat MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | 0 | 18.27 | 17.63 | -0.64 | 0.89 |
| `1e-3` | 1 | 15.15 | 17.63 | 2.48 | 2.48 |
| `1e-3` | 5 | 13.84 | 17.63 | 3.79 | 3.79 |
| `1e-3` | 10 | 21.38 | 17.63 | -3.76 | 3.76 |
| `1e-3` | 30 | 20.39 | 17.63 | -2.76 | 2.76 |
| `5e-4` | 0 | 18.35 | 17.77 | -0.58 | 0.87 |
| `5e-4` | 1 | 15.14 | 17.77 | 2.63 | 2.63 |
| `5e-4` | 5 | 17.92 | 17.77 | -0.15 | 0.78 |
| `5e-4` | 10 | 18.69 | 17.77 | -0.92 | 0.98 |
| `5e-4` | 30 | 19.73 | 17.77 | -1.96 | 2.05 |
| `3e-4` | 0 | 16.49 | 17.77 | 1.28 | 1.28 |
| `3e-4` | 1 | 18.58 | 17.77 | -0.82 | 1.13 |
| `3e-4` | 5 | 18.68 | 17.77 | -0.91 | 1.10 |
| `3e-4` | 10 | 18.83 | 17.77 | -1.06 | 1.19 |
| `3e-4` | 30 | 18.94 | 17.77 | -1.18 | 1.18 |

Raw files:

```text
runs/diagnostics/critic_relearn_20260507/sat_online64_actor5_relearn_lr1e3_vhat8.json
runs/diagnostics/critic_relearn_20260507/sat_online64_actor5_relearn_lr5e4_vhat8.json
runs/diagnostics/critic_relearn_20260507/sat_online64_actor5_relearn_lr3e4_vhat8.json
```

## Current conclusion

Under this reduced Vhat-probe budget, the post-update critic does not show a "cannot relearn after actor moves" failure.

The original concern about overshoot is supported by the mean/MAE table.  `1e-3` keeps high EV, but its value mean swings from under-prediction to over-prediction:

```text
epoch 1: pred mean 15.15 vs target 17.63
epoch 10: pred mean 21.38 vs target 17.63
```

So `1e-3` preserves ranking/structure, but it is too aggressive for a critic that is already close.

`3e-4` is the cleanest tracking LR in this sweep.  It does not have the large mean swings of `1e-3`, and its Vhat MAE stays around 1.1-1.3 after the first few epochs.

`5e-4` can work around epoch 5-10, but the later Vhat EV/MAE degradation makes it less clean than `3e-4`.

`1e-4` was not run because `3e-4` is already stable and `1e-4` is expected to be only slower for this specific relearn check.

Practical setting implied by this sweep:

```text
cold start / first refit: critic_lr = 1e-3, about 20 epochs
post-policy-update tracking: critic_lr = 3e-4, about 3-10 epochs
```

## Minibatch check

A small follow-up tested whether `critic_minibatches = 8` can be replaced by `1` for the SAT MC critic fit.

Shared setup:

```text
stage = sat
num_envs = 64
rollout_env_steps = 250
train_rollouts = 1
target = MC
critic_lr = 1e-3
critic_epochs = 20
Vhat probe = rows 8, action samples 2, continuations 2
```

Result:

| critic minibatches | outcome |
| ---: | --- |
| 8 | ran successfully; heldout MC EV = 0.941, Vhat EV = 0.913 |
| 1 | CUDA OOM during critic forward on the 16000-sample full batch |

So `critic_minibatches = 8` is not only an optimizer choice.  It is currently needed to keep the critic message-passing activations within the 6 GB GPU memory budget.

## Gradient accumulation check

To separate "minibatch for memory" from "minibatch optimizer steps", the script now supports:

```text
--critic_accumulate_full_batch
```

With this flag:

```text
critic_minibatches = 8
```

means:

```text
split 16000 samples into 8 microbatches
backward on each microbatch with loss weighted by microbatch_size / total_size
do exactly one Adam step at the end of the epoch
```

So one optimizer step sees the full 16000-sample average gradient, while memory use stays close to the 8-minibatch case.

### Result

Same SAT / MC / 64-env setup, 20 epochs:

| update style | critic lr | Adam steps per epoch | final heldout EV | note |
| --- | ---: | ---: | ---: | --- |
| regular `minibatches=8` | `1e-3` | 8 | 0.930 | works |
| accumulate full batch | `1e-3` | 1 | 0.333 | learns, but much slower |
| accumulate full batch | `3e-3` | 1 | 0.366 | still slow and less stable |
| accumulate full batch | `1e-2` | 1 | -5.952 | unstable / diverges |

The regular minibatch curve reaches high EV by about epoch 8-12:

```text
epoch 5:  heldout EV 0.689
epoch 8:  heldout EV 0.907
epoch 12: heldout EV 0.917
epoch 20: heldout EV 0.930
```

The accumulated full-batch `1e-3` curve is still rising at epoch 20:

```text
epoch 5:  heldout EV -0.027
epoch 10: heldout EV 0.118
epoch 15: heldout EV 0.236
epoch 20: heldout EV 0.333
```

Interpretation:

```text
The full-batch gradient is cleaner, but one Adam step per epoch moves too slowly at lr=1e-3.
Increasing lr to 3e-3 does not close the gap, and 1e-2 is unstable.
For the current SAT critic and MC target, repeated small-batch Adam steps are more practical than full-batch gradient accumulation.
```

This does not mean "large batch is theoretically wrong"; it means the current Adam/critic setup benefits from multiple small optimizer steps per epoch more than from a single exact full-batch step.

## Timing check

The recent audit runs were slow because they included extra diagnostic work: heldout rollout collection, per-epoch heldout evaluation, Vhat/Qpi branch replay, and actor-update probes.  A pure training-style timing was measured separately.

SAT / MC / positive reward, one rollout, no large branch probe:

| num envs | samples | rollout collect | critic train, 5 epochs | critic epoch avg |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2000 | 11.5 s | 1.23 s | 0.21-0.37 s |
| 64 | 16000 | 22.3 s | 6.42 s | 1.25-1.41 s |

So moving from 8 envs to 64 envs is not 8x slower.  Rollout collection is about 2x slower for 8x more samples, and critic training is about 5x slower.

Estimated online SAT-only cost if no Vhat/Qpi diagnostics are run:

```text
first update:
  collect 64-env rollout: about 22 s
  critic cold start, 20 epochs: about 25-26 s
  actor update: not yet separately timed, expected smaller than branch diagnostics

later updates:
  collect 64-env rollout: about 22 s
  critic tracking, 5 epochs: about 6-7 s
  actor update: extra, but should not include Vhat/Qpi replay
```

Therefore actual training should be much faster than the 10-15 minute audit runs.  If online training is still that slow, the first thing to check is whether diagnostic branch replay / heldout Vhat / repeated heldout evaluation accidentally remains in the hot path.

## 2026-05-09 100-update SAT MC-GAE training result

Run:

```text
config:
  configs/tmp/structured_sat_mcgae_3uav_20gu_t250_positive_relcritic.yaml

command:
  scripts/train_sat_mcgae.py
  --updates 100
  --num_envs 64
  --rollout_env_steps 250
  --reward_mode positive_weighted_workload_level

run dir:
  runs/diagnostics/sat_mcgae_positive_u100_20260509
```

Training setup:

```text
trained stage:
  SAT only

fixed execution:
  accel = cluster_center_queue_aware
  bw    = queue_aware

critic:
  target = finite-horizon MC return from the SAT stage rows
  update 1: lr=1e-3, 20 epochs
  later updates: lr=3e-4, 5 epochs
  minibatches=8
  update_microbatch_size=1000

actor:
  advantage = normalized A_gae(V) recomputed after critic fit
  actor lr = 3e-4
  actor epochs = 5
  actor minibatches = 1
```

Training curve summary:

| metric | update 1 | update 100 | last10 mean |
| --- | ---: | ---: | ---: |
| `sat_mc_return_mean` | 16.020 | 17.627 | 17.915 |
| `entropy_sat` | 7.060 | 5.870 | 5.956 |
| `critic_ev_after` | 0.922 | 0.957 | 0.954 |
| `approx_kl_sat` | 0.0004 | 0.0067 | 0.0047 |
| `clip_frac_sat` | 0.000 | 0.0547 | 0.0487 |

Timing after warmup and microbatch fix:

```text
last10 iteration_sec mean ≈ 25.4 s
last10 collect_sec mean   ≈ 8.35 s
last10 critic_sec mean    ≈ 12.90 s
last10 actor_sec mean     ≈ 4.12 s
```

The first update is much slower because it uses the cold critic fit:

```text
update 1 iteration_sec ≈ 83.3 s
```

### Final deterministic eval vs queue_aware SAT

Eval protocol:

```text
checkpoint:
  runs/diagnostics/sat_mcgae_positive_u100_20260509/checkpoint_final.pt

deterministic eval:
  episodes = 128 total
  seed bases = 900000 and 901000, 64 episodes each

fixed execution:
  accel = cluster_center_queue_aware
  bw    = queue_aware

comparison:
  learned SAT policy vs queue_aware SAT
```

Combined 128-episode result:

| metric | learned SAT | queue_aware SAT | learned - queue_aware |
| --- | ---: | ---: | ---: |
| `reward_sum` | 49.831 | 49.595 | +0.236 |
| `processed_ratio_eval` | 0.89699 | 0.89055 | +0.00643 |
| `drop_ratio_eval` | 0.07803 | 0.07803 | 0.00000 |
| `pre_backlog_steps_eval` | 5.540 | 5.870 | -0.331 |
| `D_sys_report` | 7.555 | 8.059 | -0.505 |
| `sat_overlap_eval` | 0.91489 | 0.76155 | +0.15333 |
| `collision_episode_fraction` | 0.0156 | 0.0156 | 0.0000 |

Paired episode counts:

```text
reward_sum:
  learned higher on 82 / 128 episodes

processed_ratio_eval:
  learned higher on 98 / 128 episodes
```

Raw eval files:

```text
runs/diagnostics/sat_mcgae_positive_u100_20260509/eval_vs_queue_aware_64ep_seed900000.json
runs/diagnostics/sat_mcgae_positive_u100_20260509/eval_vs_queue_aware_64ep_seed901000.json
```

Conclusion:

```text
SAT MC-GAE training produced a policy that is slightly but consistently better
than queue_aware SAT under the same accel/BW controllers.

The gain is not large, but it is not zero:
  reward_sum improves by about +0.24 per episode,
  processed ratio improves by about +0.0064,
  backlog/D_sys improve,
  drop and collision do not change.
```

## Can accel and BW use the same MC-GAE scheme?

Short answer:

```text
Yes in principle, but not by simply reusing scripts/train_sat_mcgae.py as-is.
The training idea is stage-generic; the current implementation is SAT-specific.
```

The stage-generic idea is:

```text
1. collect rollout with the target stage executed by policy
2. compute finite-horizon MC return for that stage's rows
3. fit the shared/stage critic on MC return
4. recompute A_gae(V) for that same stage
5. update only that stage actor with PPO ratio
6. sync native actor bindings before the next rollout
```

### Accel

Accel can use the same scheme.

The important differences from SAT are:

```text
action distribution:
  accel is continuous and currently uses the squashed/radial acceleration policy,
  so old/new logprob must use the accel distribution path.

stage row:
  use stage_id = 0 rows instead of stage_id = 1 rows.

fixed partners for accel-only training:
  sat = queue_aware
  bw  = queue_aware

safety:
  avoidance can stay enabled.
  danger imitation can be kept as an auxiliary term, but it is separate from
  the MC-GAE advantage path.
```

Why it is reasonable:

```text
Accel credit is long-horizon and state-dependent.
MC critic target avoids the broken early bootstrapped value target problem.
Then A_gae(V) uses the trained critic to reduce variance before the PPO actor update.
```

What must be implemented carefully:

```text
The SAT helper _sat_only_gae_from_mc_targets is not directly reusable.
It assumes SAT stage layout and terminal alignment.

A generic stage_mc_gae helper should take:
  stage_id
  stage transition indices
  env indices
  MC returns for that stage
  stage values
  terminated/truncated flags

and produce:
  stage returns
  stage advantages

without assuming "SAT then BW" specifically.
```

### BW

BW can also use the same scheme, if BW is trained as a stochastic PPO policy.

The important differences from SAT are:

```text
action distribution:
  BW is a masked per-UAV simplex action.
  PPO needs correct old/new logprob under the same masked simplex distribution.

stage row:
  use stage_id = 2 rows.

fixed partners for BW-only training:
  accel = cluster_center_queue_aware
  sat   = queue_aware

masks:
  invalid GU slots must remain excluded from logprob/entropy/advantage shaping.
```

Why it is less guaranteed than SAT:

```text
BW action leverage may be smaller in the current balanced regime,
and the simplex exploration distribution can make useful action differences
harder to sample.

MC-GAE fixes the critic target/advantage construction issue,
but it does not automatically guarantee that BW exploration is strong enough
or that the policy parameterization gives a large update signal.
```

Still, the right first BW test is:

```text
train BW with the same MC critic + recomputed A_gae(V) loop,
not the old clean teacher / train-time GAE target.
```

### Current code gap

Current working script:

```text
scripts/train_sat_mcgae.py
```

is intentionally SAT-only:

```text
STAGE_SAT = 1
_collect_sat_rollout(...)
_sat_only_gae_from_mc_targets(...)
_sat_actor_update_full_stage(...)
```

To train accel/BW the clean implementation should be:

```text
scripts/train_stage_mcgae.py

arguments:
  --stage accel|sat|bw
  --fixed_accel_source ...
  --fixed_sat_source ...
  --fixed_bw_source ...

shared steps:
  collect rollout
  choose stage batch by stage_id
  build MC target for stage rows
  train critic on that stage target
  compute generic stage A_gae(V)
  call stage-specific actor PPO update
```

Do not copy the SAT-specific assumptions into accel/BW.  The shared part should
be the MC target and critic/advantage pipeline; the actor logprob, masks, entropy,
and terminal alignment must remain stage-specific.

## 2026-05-09 implementation status

Implemented:

```text
scripts/train_stage_mcgae.py
```

This is the generic version of the SAT-only loop.  It supports:

```text
--stage accel
--stage sat
--stage bw
```

The shared pipeline is now:

```text
collect native rollout
select stage batch by stage_id
index finite-horizon MC return for those stage rows
train critic on that stage MC target
compute same-stage A_gae(V) from MC target and fitted V
normalize stage advantage
update that stage actor with PPO ratio
sync native actor bindings
```

Stage-specific pieces are intentionally not shared:

```text
accel:
  continuous radial-squash logprob path
  native CUDA safety shield is enabled for accel training
  regular avoidance is disabled because it is an alternative branch
  danger imitation auxiliary is enabled for accel training
  danger imitation trigger mode is forced to intervention_any

sat:
  subset categorical logprob path

bw:
  masked simplex logprob path and BW entropy normalization
```

The old SAT script is kept:

```text
scripts/train_sat_mcgae.py
```

so the already verified SAT run is not disturbed.  New experiments should use
`train_stage_mcgae.py`; old SAT reproduction can still use `train_sat_mcgae.py`.

Config additions:

```text
stage_mcgae_cold_critic_lr
stage_mcgae_cold_critic_epochs
stage_mcgae_tracking_critic_lr
stage_mcgae_tracking_critic_epochs
stage_mcgae_critic_minibatches
stage_mcgae_critic_update_microbatch_size
stage_mcgae_actor_lr
stage_mcgae_actor_epochs
stage_mcgae_actor_minibatches
```

If these are not set, `train_stage_mcgae.py` falls back to the existing
`sat_mcgae_*` values for compatibility.

Smoke checks completed:

```text
SAT:
  scripts/train_stage_mcgae.py --stage sat
  num_envs=2, rollout_env_steps=3, updates=1
  passed

accel:
  scripts/train_stage_mcgae.py --stage accel
  num_envs=2, rollout_env_steps=3, updates=1
  passed

BW:
  scripts/train_stage_mcgae.py --stage bw
  num_envs=2, rollout_env_steps=3, updates=1
  passed
```

The smoke checks only verify wiring:

```text
stage source override
rollout collection
MC target indexing
critic fit
generic same-stage GAE
stage-specific actor logprob/mask path
native actor sync
accel avoidance/danger-imitation flags are wired into the learner
```

They do not prove accel/BW will learn.  The next real checks should be:

```text
1. SAT reproduction with train_stage_mcgae.py for a short run.
2. accel-only run using:
     --stage accel
     exec_sat = queue_aware
     exec_bw = queue_aware
3. BW-only run using:
     --stage bw
     exec_accel = cluster_center_queue_aware
     exec_sat = queue_aware
```

For BW, keep `bw_flow_proxy_aux_enabled=false` unless the auxiliary loss is
explicitly added to the generic actor update.  The current generic script raises
if that auxiliary is enabled, to avoid silently changing the BW objective.

## 8. Joint MC-GAE Path

For all-stage training, do not use the ordinary `train_structured.py` PPO path
as a proxy for this experiment.  The intended path is:

```text
scripts/train_joint_mcgae.py
```

Its training loop is:

```text
1. collect one native rollout with accel/sat/bw all executed by policy
2. compute finite-horizon MC return for the full transition stream
3. index the same MC return into accel/sat/bw stage batches
4. train the shared critic sequentially on all three stage MC targets
5. re-evaluate the final critic for each stage
6. compute same-stage A_gae(V) from MC targets and final stage values
7. update accel/sat/bw actors with three independent optimizers
8. sync the native actor bindings once after all actor updates
```

Important defaults:

```text
reward_mode = positive_weighted_workload_level
actor_advantage_normalize_enabled = true
stagewise_advantage_norm_enabled = true
accel safety = safety_shield_enabled=true, safety_shield_solver=NATIVE_CUDA
regular avoidance = false
danger_imitation_enabled = true
danger_imitation_trigger_mode = intervention_any
```

This differs from ordinary PPO in two key ways:

```text
critic target:
  MC return, not one-shot train-time GAE target

actor advantage:
  A_gae(V) recomputed after the critic has been fit on that rollout
```

Smoke check completed:

```text
scripts/train_joint_mcgae.py
num_envs=2, rollout_env_steps=4, updates=1
cold_critic_epochs=1, critic_minibatches=1, actor_epochs=1
passed
```

The first smoke run spent most time in first-use `torch.compile` compilation of
the critic.  That timing should not be compared with steady-state training
speed.
