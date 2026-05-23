# BW-only PPO T Sweep and Critic Value Diagnosis (2026-04-10)

## 1. Purpose

This note summarizes the 2026-04-10 follow-up on the simplified `1 UAV / BW-only` PPO issue.

The practical question is:

- why does the `BW-only + PPO` setup still work at short horizons,
- but become unreliable when `T` grows,
- and whether the immediate priority should be making the critic/value path more accurate.

Important framing:

- the current priority is **not** to introduce `action-dependent` or `counterfactual` credit first
- this is not because we have a preference for state-value critics
- it is because earlier action-dependent/counterfactual attempts did not solve the issue cleanly
- therefore the current line of investigation is: **how far can we get by making the standard PPO critic/value/advantage path accurate enough, and where exactly does it fail**

## 2. Original Symptom: T Sweep

The main simplified setting was:

- `num_uav: 1`
- `num_gu: 5`
- `train_accel: false`
- `train_sat: false`
- `train_bw: true`
- `exec_accel_source: zero`
- `exec_sat_source: zero`
- `exec_bw_source: policy`
- `BW-only + full PPO + env_reward`

The original `T=1/2/5/10` checkpoint eval showed a clear split:

| T | learned reward | heuristic reward | reward gap | processed | drop | backlog |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 0.682 | 0.576 | +0.106 | `1.632 > 1.427` | `0 = 0` | `4.368 < 4.573` |
| 2 | 1.133 | 1.048 | +0.085 | `1.401 > 1.319` | `0 = 0` | `4.372 < 4.467` |
| 5 | 2.212 | 2.249 | -0.036 | `1.152 < 1.163` | `0 = 0` | `4.390 > 4.314` |
| 10 | 3.344 | 3.939 | -0.595 | `0.949 < 1.046` | `0.0029 > 0` | `4.752 > 4.333` |

Training metrics at the final update were also consistent with this:

| T | env_reward_mean | approx_kl_bw | clip_frac_bw | bw_kappa_mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6388 | 0.0176 | 0.317 | 49.56 |
| 2 | 0.5441 | 0.0670 | 0.365 | 51.00 |
| 5 | 0.4546 | 0.0474 | 0.360 | 40.52 |
| 10 | 0.3252 | 0.00387 | 0.015 | 19.45 |

Initial conclusion:

- `T=1,2` can learn
- `T=5` starts to fall behind
- `T=10` is clearly bad
- the issue is not "one-step BW cannot be learned"
- the issue appears when the return depends on several consecutive BW decisions and queue feedback

## 3. Implementation Fixes Made During the Investigation

Several implementation details were checked and fixed before interpreting the critic results.

### 3.1 Time-limit bootstrap

The `T_steps` boundary is a training/simulation truncation of a continuing queue system, not a true terminal failure.

Therefore:

- true terminal, such as collision or energy depletion: no bootstrap
- time-limit truncation from `T_steps`: should bootstrap from value of the next state

Structured and non-structured code paths were updated so time-limit truncation is treated as bootstrap-able.

For BW-only structured training, the bootstrap value must use the BW value head:

- `train_bw: true`
- `train_accel: false`
- `train_sat: false`
- bootstrap should use `V_bw`

The BW-only structured learner now forces the step bootstrap stage to `bw` when only BW is trained.

### 3.2 Return mode naming

The old name `bw_return_mode: monte_carlo` was misleading.

It was not strict pure MC when `gae_lambda < 1`, because its future term could still contain lambda-return/value-bootstrap terms.

The code now treats old `monte_carlo`/`mc` as a compatibility alias for `step_lambda_return`, and the main standard PPO experiments use:

- `bw_return_mode: gae`
- `structured_step_bootstrap_stage: bw`

## 4. What Is Abnormal in the PPO Update

We separated two questions:

1. Does PPO follow its own advantage?
2. Does that advantage point in the direction of actual longer-horizon return?

The answer is:

- PPO often does follow its own advantage: `adv -> dlogp` is usually positive
- but at `T=10`, the advantage itself does not reliably align with true action-level return improvement

Representative 10-update GAE + `V_bw` chain-probe comparison:

| metric | T=1 | T=10 |
| --- | ---: | ---: |
| `delta_actor_reward_mean` | +0.0044 | +0.0001 |
| `actor_reward_improved_state_frac` | 0.883 | 0.517 |
| `value_loss_bw` | 0.105 | 0.731 |
| `approx_kl_bw` | 0.030 | 0.0076 |
| `clip_frac_bw` | 0.212 | 0.063 |
| `adv -> dlogp` | +0.325 | +0.199 |
| `true_adv -> dlogp` | +0.511 | +0.024 |
| `raw_adv -> true_adv` | +0.112 | +0.052 |
| `sign_agree_raw_adv_true` | 0.475 | 0.496 |
| `raw_advantage_abs_mean` | 0.397 | 0.719 |
| `true_adv_mc_abs_mean` | 0.034 | 0.030 |

Interpretation:

- `T=10` does not simply have a PPO sign bug
- the actor update still has some positive relation to the PPO advantage it was given
- the bigger issue is that `raw_advantage = return_target - V(s)` does not reliably represent the true action-level advantage
- the true action advantage is tiny, around `0.02-0.04`
- value/return residuals much larger than that can easily dominate the sign of the PPO advantage

## 5. Attempts to Improve the Critic / Advantage Path

### 5.1 Larger rollout batch

We tried increasing data per update with `num_envs=8`, because `num_envs` is practically capped around 8.

| setting | env steps/update | `Vpi_err` | `rawAdv -> trueAdv` | `trueAdv -> dlogp` | `dSelf` | eval gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T10 baseline | 100 | 3.534 | -0.382 | +0.037 | +0.004 | -0.524 |
| `num_envs=8, rollout=50` | 400 | 3.297 | -0.289 | -0.059 | +0.004 | -0.580 |
| `num_envs=8, rollout=125` | 1000 | 2.633 | -0.104 | +0.260 | -0.002 | -0.538 |

Conclusion:

- larger rollout batch is not useless
- it reduced some value/advantage error indicators
- but the improvement was not enough to make `rawAdv -> trueAdv` reliably positive
- it also did not make the actor's self-return consistently improve

### 5.2 More critic training / higher critic LR / higher GAE lambda

Short-run ablations at `T=10`, update 10:

| setting | `value_loss_bw` | `rawAdv -> trueAdv` | `trueAdv -> dlogp` | `adv -> dlogp` | `dSelf` | eval gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.077 | -0.382 | +0.037 | +0.187 | +0.004 | -0.524 |
| `critic_epochs=12` | 0.043 | -0.137 | -0.100 | +0.169 | +0.003 | -0.522 |
| `critic_epochs=16` | 0.035 | -0.111 | +0.283 | +0.290 | +0.006 | -0.517 |
| `critic_lr=1e-3` | 0.102 | -0.368 | +0.138 | +0.174 | +0.006 | -0.535 |
| `gae_lambda=0.99 + batch8x50` | 0.456 | -0.287 | -0.029 | +0.233 | +0.005 | -0.588 |

Conclusion:

- `critic_epochs=12/16` can reduce `value_loss_bw`
- but lower training value loss does not automatically produce a reliable action-level advantage
- `critic_lr=1e-3` did not help
- `gae_lambda=0.99` did not help in the tested short-run setting
- stronger critic fitting helps somewhat, but not enough

Important implementation detail:

- the extra `critic_epochs` in the current code do not mean "train critic first, recompute advantage, then train actor"
- the actor advantage snapshot is fixed before the PPO actor update
- extra critic epochs mainly improve the critic for later use, not the already-used actor advantage in the same update

## 6. Fixed-policy Critic Generalization Probe

To separate network capacity from online PPO noise, we added:

- `scripts/probe_structured_bw_value_generalization.py`

Protocol:

- freeze a trained actor
- collect train and heldout BW states
- estimate `V^pi(s)` by repeated MC rollouts
- train only the critic on the train states
- evaluate value prediction error on heldout states

Main finite-to-time-limit MC setting:

- policy mode: stochastic
- train states: 80
- heldout states: 40
- MC rollouts/state: 16
- horizon: 20
- target mode: finite to truncation

Results:

| setting | train RMSE | heldout RMSE | heldout corr | heldout MC SE |
| --- | ---: | ---: | ---: | ---: |
| T1 | 0.012 | 0.028 | 0.993 | 0.009 |
| T10 hotspot/hetero | 0.066 | 0.508 | 0.938 | 0.037 |

Interpretation:

- the critic architecture can fit the T10 train states well
- therefore the issue is not simply "the network cannot fit any T10 value target"
- the hard part is heldout generalization
- T10 heldout value error is still around `0.4-0.5`, much larger than action advantage magnitude `0.02-0.04`
- T1 is much easier: heldout RMSE around `0.028`, close to the action-advantage scale

This supports the current hypothesis:

- online PPO rollout data is not enough to train a state-value critic whose heldout error is small enough for T10 BW action-level credit

## 7. Arrival / Hotspot Ablation

We also tested whether the traffic process was a major source of value difficulty.

Original T10 setting:

- `traffic_model: sticky_subset_hotspot`
- `task_arrival_poisson: false`
- `arrival_base_hetero: 0.9`
- `hotspot_num_subsets: 4`
- `hotspot_rho: 8.0`

Note:

- Poisson arrival was already disabled
- the stochastic/heterogeneous part mainly came from the sticky hotspot state and per-GU heterogeneity

Constant-arrival ablation:

- `traffic_model: homogeneous`
- `task_arrival_poisson: false`
- `arrival_base_hetero: 0.0`
- `hotspot_num_subsets: 0`
- all GU have the same deterministic arrival rate

Short PPO probe:

| T10 setting | `rawAdv -> trueAdv` | `trueAdv -> dlogp` | `adv -> dlogp` | eval gap |
| --- | ---: | ---: | ---: | ---: |
| hotspot/hetero | -0.382 | +0.037 | +0.187 | -0.524 |
| constant arrival | +0.008 | +0.216 | +0.323 | -0.191 |

Fixed-policy value generalization:

| setting | heldout RMSE | heldout corr | heldout MC SE |
| --- | ---: | ---: | ---: |
| T10 hotspot/hetero | 0.508 | 0.938 | 0.037 |
| T10 constant arrival | 0.345 | 0.975 | 0.015 |
| T1 reference | 0.028 | 0.993 | 0.009 |

Conclusion:

- arrival/hotspot is a real difficulty amplifier
- removing hotspot/heterogeneity improves both PPO update diagnostics and value generalization
- but it does not fully solve the problem
- even constant-arrival T10 still has heldout value error around `0.345`, far larger than action advantage scale

So the traffic process is not the only cause.

The remaining difficulty is still the multi-step queue closed loop and the need to estimate a small action-level residual from a state-value baseline.

## 8. Why T10 Critic Is Hard

The critic is hard for several compounding reasons.

First, the useful BW action advantage is small:

- true action advantage is often around `0.02-0.04`
- value prediction error around `0.3-0.5` is already enough to swamp it

Second, T10 is a multi-step queue system:

- current BW changes GU queue outflow
- that changes UAV and satellite queues
- later BW decisions interact with those queues
- the reward effect can be delayed and partially masked by future policy samples

Third, each online rollout gives limited repeated coverage:

- larger batch gives more states
- but not many repeated futures from the same or nearby state
- state-value learning needs to average out future trajectory noise to estimate `V^pi(s)`

Fourth, the target is nonstationary online:

- the actor changes every update
- therefore `V^pi(s)` changes every update
- the critic is chasing a moving target

Fifth, traffic heterogeneity/hotspot increases difficulty:

- it changes which GU are important over time
- it increases value variance and generalization burden
- removing it helps, but not enough to make the T10 state-value advantage clean

## 9. Current Interpretation

The current best diagnosis is:

> At T10, PPO mostly follows the advantage it is given, but the state-value advantage is not accurate enough. The value error from `V(s)` is much larger than the BW action-level advantage, so `return_target - V(s)` does not reliably encode whether the sampled BW action was better than the current policy average.

This explains why:

- `adv -> dlogp` can be positive
- `trueAdv -> dlogp` can be weak or unstable
- increasing critic epochs lowers value loss but does not fully fix the actor update
- increasing rollout data helps somewhat but not enough
- constant arrival helps but still does not reach the needed precision

## 10. Recommended Next Priority

The next priority should still be the critic/value path, not immediately switching to action-dependent/counterfactual credit.

Reason:

- action-dependent/counterfactual variants were tried before and did not solve the issue cleanly
- state-value PPO is the standard baseline
- we now have evidence that the critic value error is a concrete bottleneck
- therefore the immediate next question should be: **what changes make heldout `V^pi(s)` error drop from `0.3-0.5` toward the action-advantage scale?**

Recommended critic-first experiments:

1. Add heldout value-generalization metrics to the debugging loop.

Do not judge critic only by `value_loss_bw` on the training batch.

Track:

- heldout value RMSE
- heldout value correlation
- heldout explained variance
- heldout bias
- RMSE relative to true action advantage scale

2. Test larger critic data coverage before larger network.

Useful variants:

- `num_envs=8`
- larger `rollout_env_steps`
- snapshot-bank style critic-only supervised data
- repeated futures from the same or similar BW states

The goal is not just more samples, but better coverage of the value function and lower heldout error.

3. Test critic target scaling / value normalization.

The current critic fits raw unnormalized returns.

Possible controlled tests:

- return target standardization for critic training
- PopArt-like value normalization
- reward/value scale normalization limited to the critic path

The success criterion should be heldout value accuracy and downstream advantage alignment, not only lower training MSE.

4. Test critic capacity only after data/target checks.

A larger critic, for example `hidden_dim=512, embed_dim=128`, is worth testing, but the criterion should be:

- does heldout RMSE drop substantially?
- or does train RMSE drop while heldout RMSE remain high?

If only train RMSE improves, network scale is not the main solution.

5. Clarify continuing-value vs finite-to-truncation target.

The queue system is conceptually continuing, while many probes use finite-to-time-limit MC for tractability.

We should keep both target definitions explicit:

- finite-to-truncation MC: cleaner, lower-variance diagnostic
- continue-through-truncation MC: closer to continuing value, but much higher variance and can leave the training time-index distribution

## 11. Fallback Direction

If critic-first work cannot reduce heldout `V^pi(s)` error to the needed scale, then we may need to revisit action-dependent credit.

But that should be framed as a fallback because:

- previous action-dependent/counterfactual attempts were not clearly successful
- we first need to know whether the standard critic path can be made accurate enough
- if it cannot, the reason should be documented quantitatively, not assumed

Possible fallback directions:

- `Q_bw(s, a_bw)` with careful heldout validation
- per-slot or per-user local value residuals
- counterfactual credit only on states with measurable BW leverage
- short-horizon local rollout/search targets used as auxiliary supervision rather than replacing PPO wholesale

## 12. Key Data Sources

Original T sweep:

- `runs/structured_bw_gap_onestep_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t2_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t5_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t10_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_onestep_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t2_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t5_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t10_envreward_ppo_u30/metrics.csv`

GAE + `V_bw` update-direction probes:

- `runs/structured/structured_bw_chainprobe_t1_gae_vbw_u10_12panel_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_chainprobe_t10_gae_vbw_u10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_batch8_roll50_probeu10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_batch8_roll125_probeu10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_criticepoch12_probeu10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_criticepoch16_probeu10_fix_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_criticlr1e3_probeu10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae099_vbw_batch8_roll50_probeu10_20260410/update_direction_probe.csv`

Fixed-policy value generalization probes:

- `runs/analysis/bw_value_gen_t1_policy_mc80x40_r16_20260410/summary.json`
- `runs/analysis/bw_value_gen_t10_policy_mc80x40_r16_20260410/summary.json`
- `runs/analysis/bw_value_gen_t10_constarrival_policy_mc80x40_r16_20260410/summary.json`
- `runs/analysis/bw_value_gen_t10_continue_mc40x20_r16_20260410/summary.json`

Constant-arrival T10 run:

- `configs/structured_bw_t10_gae_vbw_constarrival_probeu10.yaml`
- `runs/structured/structured_bw_t10_gae_vbw_constarrival_probeu10_20260410/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_gae_vbw_constarrival_probeu10_20260410/checkpoint_eval.csv`

Probe script:

- `scripts/probe_structured_bw_value_generalization.py`
