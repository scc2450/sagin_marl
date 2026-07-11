# Phase 4 Learning Ablation Runbook

Date: 2026-07-11

This runbook tracks the first learned ablation for Section 5:

> same staged actor, same training protocol, relational critic vs global-only
> critic.

The purpose is to test whether the relational centralized critic improves value
estimation and training quality. It is not an external MARL baseline and should
not be described as vanilla MAPPO.

## Branch

Implementation branch:

```bash
git switch erik/phase4-learning-ablation-baseline
```

## Configs

Full method anchor:

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

Global-only critic ablation:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml
```

Only intended config difference:

```yaml
critic_value_mode: relational
```

vs.

```yaml
critic_value_mode: global_only
```

Keep fixed:

- actor architecture and staged action factorization;
- environment scenario: 3 UAV, 20 GU, T=250;
- reward, queue, safety, and mask settings;
- MC-GAE/return-target settings;
- training seed list and evaluation seed bases;
- checkpoint selection rule.

## Algorithmic Delta

This ablation changes only the centralized critic value estimator. The learned
policy interface and execution logic remain the same.

Unchanged actor/execution components:

- acceleration actor;
- satellite subset actor;
- bandwidth allocation actor;
- staged action factorization: acceleration -> satellite selection ->
  bandwidth allocation;
- action masks and safety-aware execution;
- reward, queue dynamics, traffic model, and scenario parameters;
- PPO/MC-GAE training loop and stage-wise actor update order.

Full relational critic pathway:

```text
global scalars
  + GU/UAV/SAT node tokens
  + UAV-GU/UAV-SAT/UAV-UAV typed edge tokens
  + entity masks and local summaries
  -> typed relational message passing
  -> system token
  -> stage-specific value heads
```

Global-only critic pathway:

```text
global scalars
  -> global scalar encoder
  -> stage-specific value heads
```

Removed from the critic value pathway:

- GU/UAV/SAT node encoders;
- UAV-GU/UAV-SAT/UAV-UAV edge encoders;
- local summary encoders;
- typed relational message-passing blocks;
- entity/relation masks as critic-side relational structure.

Still present in `global_only`:

- trainable neural value functions;
- stage-specific value heads for acceleration, satellite selection, and
  bandwidth allocation;
- the same actor observations and action masks.

Therefore the comparison should be named:

```text
Relational critic vs global-only critic
```

It should not be named:

```text
Ours vs vanilla MAPPO
```

because the actor remains the same structured staged actor. This comparison
tests whether relational centralized value estimation improves learning and
evaluation quality under the same staged policy class.

## Local Smoke

Use the existing Mac/local smoke runner for wiring checks:

```bash
CONFIG=configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml \
RUN_DIR=runs/phase4_learning_ablation/smoke_global_only_critic \
UPDATES=1 \
NUM_ENVS=1 \
ROLLOUT_ENV_STEPS=2 \
scripts/run_mac_joint_smoke.sh
```

This is not evidence for the paper. It only checks that the config loads and the
joint training entrypoint accepts the ablation mode.

Verified locally on 2026-07-11:

```text
Update 1/1 completed with critic_value_mode=global_only.
```

Smoke artifacts were written under:

```text
runs/phase4_learning_ablation/smoke_global_only_critic
```

## Formal Run Shape

Use checkpoint-eval early stopping, not a fixed training length. The training
script still needs a budget cap, but `--max_updates` is only a hard upper bound.
The phase4 ablation should follow the same early-stop shape as the current
phase3 return-target runs: do not stop before 300 updates, validate every 25
updates, and stop on validation plateau before the 700-update cap when possible.

```text
checkpoint_eval_interval_updates = 25
checkpoint_eval_min_stop_update = 300
checkpoint_eval_episodes = 64
checkpoint_eval_reward_patience = 4
checkpoint_eval_reward_min_delta_rel = 0.005
hard_max_updates = 700
```

Suggested run directory naming:

```text
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211
```

Suggested first seed:

```text
45211
```

If compute allows, use at least two aligned seeds for the paper table.

Suggested remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/sgy/workspace/sagin_marl/.venv/bin/python scripts/train_joint_mcgae.py \
  --config configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml \
  --run_dir runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_<timestamp> \
  --device cuda \
  --num_envs 64 \
  --rollout_env_steps 250 \
  --max_updates 700 \
  --seed 45211 \
  --save_every 25
```

The selected checkpoint for downstream held-out evaluation should be
`best_checkpoint.pt`, chosen by source-scenario checkpoint evaluation. `final.pt`
is still useful for stability reporting, but it is not the selection rule.

Aborted run note:

```text
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_20260711_143556
```

This run was launched with fixed `--updates 300` and stopped around update 14.
Do not use it as a paper result.

## Metrics To Compare

Training side:

- validation reward curve;
- best/final validation reward;
- stage-wise critic loss;
- stage-wise explained variance, if logged;
- actor KL/clip/entropy or skipped-update indicators, if logged.

Evaluation side:

- reward;
- processed ratio;
- drop ratio;
- backlog or delay proxy;
- collision/early termination rate;
- runtime if easy to collect.

## Interpretation

If relational beats global-only:

- claim that relational value estimation improves learning quality or stability;
- tie the claim to critic/advantage quality, not to every actor-side design.

If global-only is close:

- keep the ablation and be honest;
- shift the main contribution emphasis toward staged hybrid-action scheduling,
  safety-aware execution, and protocol-level robustness.

If global-only wins:

- inspect critic overfitting, EV, and training horizon before treating the
  relational critic as final.
