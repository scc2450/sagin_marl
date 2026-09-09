# 1 UAV Static BW Sanity: RL Update Follow-up (2026-04-09)

## 1. Purpose

This note summarizes the current state of the simplified `1 UAV / BW-only` debug benchmark after the 2026-04-09 investigation.

The goal of this note is to answer a narrower question than the earlier follow-up:

- if the benchmark is already simplified enough,
- and the actor can represent a good policy,
- why does RL training still fail?

The short answer is:

- this is **not** mainly an actor-capacity problem
- this is **not** reducible to a single obvious reward bug
- this is **not** explained by a trivial PPO implementation sign bug
- the strongest current evidence points to a problem in the **current BW RL update path itself**, especially the family of updates that optimize the log-probability of sampled simplex actions

This note continues from:

- `docs/bw_sanity_1uav_static_followup_20260408.md`

## 2. Scope

Unless otherwise stated, the main comparison setting is:

- `configs/structured_bw_sanity_1uav_static_gap_debug_weighted_scalar_mc.yaml`

Core semantics:

- `num_uav: 1`
- `num_gu: 5`
- `train_accel: false`
- `train_sat: false`
- `train_bw: true`
- `exec_accel_source: zero`
- `exec_sat_source: zero`
- `exec_bw_source: policy`
- `bw_train_target_mode: weighted_workload_delta`
- `bw_return_mode: monte_carlo`
- `structured_bw_per_slot_surrogate_enabled: false`
- `structured_bw_policy_update_mode: ppo`
- `structured_bw_parameterization: score_alpha_kappa_stickbreaking`

This isolates the current issue to:

- a single UAV
- no learned accel/sat control
- only the `BW` head being trained
- scalar `BW` actor update without per-slot surrogate

## 3. What Has Been Established

### 3.1 The actor can represent a strong policy

Imitation sanity run:

- `runs/structured_bw_imitation_queueaware_sanity/summary.json`

Key numbers:

- learned deterministic reward: `34.4164`
- learned stochastic reward: `34.2592`
- fixed `queue_aware_bw` reward: `34.5134`
- learned deterministic `bw_weighted_workload_delta_sum`: `296.7517`
- fixed `queue_aware_bw` `bw_weighted_workload_delta_sum`: `297.1524`

Interpretation:

- the current `BW` actor architecture can nearly reproduce a simple effective teacher
- therefore the main blocker is **not** that the actor is too weak to represent a useful bandwidth allocation rule

Important note:

- these imitation-sanity numbers are not directly comparable to the training checkpoint-eval numbers below
- they are used only as an actor-capacity sanity check

### 3.2 Simple reward substitutions did not solve the problem

Direct short-run comparison under the same `BW-only + scalar + MC + rollout=100 + updates=10` setup:

- baseline `weighted_workload_delta`:
  - `runs/structured_bw_gap_weighted_delta_scalar_mc_r100_u10_current/checkpoint_eval.csv`
- `weighted_workload_level = -W_after - drop_cost`:
  - `runs/structured_bw_gap_weighted_level_scalar_mc_r100_u10/checkpoint_eval.csv`

Main numbers:

| Mode | reward_sum | weighted_delta_sum | weighted_level_sum | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `weighted_workload_delta` | `3.9639` | `282.8691` | `-4337.1382` | `0.7661` | `0.1553` | `10.7287` |
| `weighted_workload_level` | `1.5226` | `280.9493` | `-4420.8036` | `0.7510` | `0.1726` | `10.7297` |
| fixed `queue_aware_bw` | `25.9332` | `295.1865` | `-2914.4041` | `0.9109` | `0.0430` | `8.0802` |

Interpretation:

- replacing `weighted_workload_delta` with a similarly simple `-W_after-drop` target did **not** solve the issue
- at least in this benchmark, the failure cannot be explained as "the old reward was wrong, and a nearby simple reward fixes it"

### 3.3 Longer BW-only Monte Carlo return did not solve the problem

True `BW`-only episode-MC run:

- `runs/structured_bw_gap_weighted_delta_scalar_bwepisodemc_r100_u10/checkpoint_eval.csv`

Compared to the baseline:

| Mode | reward_sum | weighted_delta_sum | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline `weighted_delta + shared MC` | `3.9639` | `282.8691` | `0.7661` | `0.1553` | `10.7287` |
| `weighted_delta + bw_episode_mc` | `1.4364` | `280.9904` | `0.7513` | `0.1727` | `10.7771` |

Interpretation:

- simply extending the `BW` return all the way to episode end did not rescue learning
- this rules out the strong claim that "the main problem is only that the BW return horizon is too short"

## 4. What the Credit Diagnostics Show

### 4.1 The current training signal does not rank actions the way longer-horizon outcomes do

Credit decomposition:

- `runs/diagnostics/credit_decomp_weighted_scalar_k20/summary.json`

For informative `BW` states, using the same `weighted_workload_delta` target:

- `immediate_vs_true` per-state Spearman mean: `-0.8383`
- `boot_vs_true` per-state Spearman mean: `-0.8383`
- `next_value_vs_true` per-state Spearman mean: `-0.1827`

Interpretation:

- within the same state, actions that look better to the immediate/boot training signal are often worse under the longer-horizon `k=20` outcome
- so the current `BW` update signal is not merely noisy; it is directionally misleading on these states

Important caveat:

- this still does **not** prove that a different reward formula alone would solve the benchmark
- it only proves that the current learning signal does not align well with longer-horizon action quality

### 4.2 But the current failure is not reducible to reward alone

If reward alone were the whole story, then switching to a different policy update family should not matter much.

However the update-family comparison below shows:

- `PPO` and `AWR` are **not identical**
- but both still fail

So the problem is better described as:

- a bad RL learning signal chain (`reward -> return -> advantage -> sampled-action update`)

not simply:

- "a single wrong reward formula"

## 5. PPO vs AWR: Current Best Evidence

### 5.1 Short training comparison

Baseline PPO run:

- `runs/structured_bw_gap_weighted_delta_scalar_mc_r100_u10_current/checkpoint_eval.csv`

AWR comparison run:

- `runs/structured_bw_gap_weighted_delta_scalar_awr_r100_u10/checkpoint_eval.csv`

Main numbers:

| Update mode | reward_sum | weighted_delta_sum | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PPO` | `3.9639` | `282.8691` | `0.7661` | `0.1553` | `10.7287` |
| `AWR` | `3.1607` | `282.3161` | `0.7629` | `0.1610` | `10.7605` |
| fixed `queue_aware_bw` | `25.9332` | `295.1865` | `0.9109` | `0.0430` | `8.0802` |

Interpretation:

- replacing `PPO` with `AWR` does change the optimization family
- but it does **not** rescue the benchmark
- therefore the problem is not "PPO only" in the narrow sense

### 5.2 Gradient-alignment diagnostics

PPO gradient-alignment run:

- `runs/diagnostics/bw_grad_align_weighted_delta_scalar_mc/summary.json`

AWR gradient-alignment run:

- `runs/diagnostics/bw_grad_align_weighted_delta_scalar_awr/summary.json`

Episode-MC PPO gradient-alignment run:

- `runs/diagnostics/bw_grad_align_weighted_delta_bwepisodemc/summary.json`

Key numbers:

| Update mode | grad cosine vs imitation | det L1 before | det L1 after | worsened frac |
| --- | ---: | ---: | ---: | ---: |
| `PPO` | `-0.9634` | `0.7607` | `0.7765` | `0.9100` |
| `AWR` | `-0.8184` | `0.7234` | `0.7362` | `0.8100` |
| `PPO + bw_episode_mc` | `-0.5160` | `0.8293` | `0.8333` | `0.8300` |

Interpretation:

- all three update variants move the policy away from the simple effective teacher on most samples
- `PPO` is the worst of the three
- `AWR` is less bad than PPO, but still directionally wrong
- true `BW` episode-MC return reduces the severity, but still does not fix the direction problem

This is currently the strongest evidence that:

- the failure is not just critic horizon
- the failure is not just actor capacity
- the failure lies in the **current sampled-action RL update path itself**

## 6. What Has Been Ruled Out vs What Has Not

### 6.1 Ruled out or strongly weakened

The following explanations are now weak:

- "the actor cannot represent a good policy"
- "a nearby simple reward replacement immediately fixes training"
- "the main issue is only critic bootstrap / short return horizon"
- "there is a trivial sign bug in the PPO ratio formula"

### 6.2 Not ruled out

The following are still open:

- the current simplex-action update family may be fundamentally mismatched to this `BW` problem
- the current advantage signal may still be semantically bad even when its arithmetic is correct
- the current distribution/update interface may be poor at moving the policy center toward better allocations
- the stick-breaking parameterization may still contribute part of the problem, although it is no longer the primary suspect

## 7. Current Best Diagnosis

The most precise current diagnosis is:

- the benchmark is simple enough
- the actor is expressive enough
- but the current `BW` RL training signal still fails to produce useful policy movement

More concretely:

- `PPO` and `AWR` both optimize the log-probability of sampled actions judged good by their weighting signal
- on this benchmark, those sampled-action updates do **not** move the policy toward a simple effective allocation rule
- `PPO` amplifies the problem more strongly than `AWR`
- but `AWR` still fails, so the issue is broader than PPO clipping alone

Therefore the strongest current statement is:

> The main problem is no longer best described as "reward only" or "critic only".  
> It is better described as a failure of the current `BW` sampled-action RL update family on this continuous simplex action.

## 8. Current PPO File Map

The following files are the main files involved in the current `BW` PPO path.

### 8.1 Config and top-level training entry

- `configs/structured_bw_sanity_1uav_static_gap_debug.yaml`
  - original high-gap debug config
- `configs/structured_bw_sanity_1uav_static_gap_debug_weighted_scalar_mc.yaml`
  - current scalar-PPO baseline config
- `configs/structured_bw_sanity_1uav_static_gap_debug_weighted_scalar_mc_awr.yaml`
  - AWR comparison config
- `configs/structured_bw_sanity_1uav_static_gap_debug_weighted_delta_scalar_bw_episode_mc.yaml`
  - true `BW` episode-MC comparison config
- `scripts/train_structured.py`
  - main training entrypoint
- `sagin_marl/env/config.py`
  - defines PPO/AWR/BW config fields

### 8.2 Training loop and rollout construction

- `sagin_marl/rl/structured_train.py`
  - rollout loop, env-step accounting, metrics, checkpoint eval scheduling
- `sagin_marl/rl/structured_buffer.py`
  - stores structured transitions and computes returns / advantages
- `sagin_marl/rl/structured_mappo.py`
  - core rollout collection, actor/critic update, PPO/AWR objective code

### 8.3 Actor, action distribution, and BW local-state construction

- `sagin_marl/rl/structured_factory.py`
  - builds actor / critic modules from config
- `sagin_marl/rl/structured_actor.py`
  - `BW` policy forward pass, deterministic readout, action evaluation
- `sagin_marl/rl/distributions.py`
  - simplex distributions including stick-breaking Beta and Dirichlet
- `sagin_marl/rl/structured_stage_builders.py`
  - builds local `BW` states from structured snapshots

### 8.4 Critic and world-state value path

- `sagin_marl/rl/structured_critic.py`
  - shared world-state critic with separate accel/sat/bw heads

### 8.5 Environment / reward / stage execution

- `sagin_marl/env/structured_driver.py`
  - structured stage execution, including `BW` reward bookkeeping
- `sagin_marl/env/sagin_env.py`
  - environment dynamics, rate computation, queue updates

### 8.6 Evaluation and diagnostics

- `sagin_marl/rl/structured_eval.py`
  - checkpoint/model evaluation logic
- `scripts/experiments/bw_training/train_structured_bw_imitation_sanity.py`
  - actor-capacity sanity via teacher imitation
- `scripts/diagnostics/diagnose/diagnose_structured_bw_policy_gradient_alignment.py`
  - gradient / one-step-update alignment diagnostics

## 9. What Is Still Missing

This note does **not** yet prove:

- whether the failure is specific to the stick-breaking parameterization
- whether the old Dirichlet parameterization suffers from exactly the same update pathology
- whether any non-sampled-action update objective can recover strong learning on the same benchmark

Those remain open.

## 10. Final Conclusion

As of 2026-04-09, the current simplified benchmark supports the following conclusion:

1. `BW` failure is **not** mainly due to actor expressivity.
2. `BW` failure is **not** fixed by nearby simple reward substitutions.
3. `BW` failure is **not** fixed by simply extending the `BW` return to episode-MC.
4. The strongest remaining problem is the current `BW` RL update path itself.
5. In particular, the current family of sampled-action log-prob updates (`PPO`, and to a lesser extent `AWR`) does not move the policy toward simple effective `BW` behavior on this benchmark.

That is the most precise current status.
