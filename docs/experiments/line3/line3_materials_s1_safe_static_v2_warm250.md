# Line 3 Materials for `s1_safe_static_v2_warm250`

Run directory: `runs/ablation_followup/s1_safe_static_v2_warm250`

Artifacts generated in this pass:

- `runs/ablation_followup/s1_safe_static_v2_warm250/eval_repro_43000.csv`
- `runs/ablation_followup/s1_safe_static_v2_warm250/eval_fixed_repro_43000.csv`
- `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep18_seed43000base.csv`
- `runs/ablation_followup/s1_safe_static_v2_warm250/obs_sample_eval_ep18_uav0.json`

## 1. Observation definition

Core code:

- `sagin_marl/env/sagin_env.py:218` `_build_spaces()`
- `sagin_marl/env/sagin_env.py:1422` `_get_obs()`
- `sagin_marl/env/sagin_env.py:1381` `_cache_sat_obs()`
- `sagin_marl/env/sagin_env.py:1482` `_build_global_state()`

Current best run dimensions:

- Actor obs dim: `207`
- Critic global state dim: `142`

Observation blocks:

| Block | Shape | Meaning | Range / normalization | Mask / padding |
| --- | --- | --- | --- | --- |
| `own` | `(7,)` | `[x, y, vx, vy, energy, uav_queue, t]` for this UAV | `x,y` divided by `map_size`; `vx,vy` divided by `v_max`; `energy` divided by `uav_energy_init`; `uav_queue` divided by `queue_max_uav`; `t` divided by `T_steps` | no extra mask |
| `users` | `(users_obs_max, 5)` = `(20, 5)` | per candidate GU: relative xy, GU queue, access spectral efficiency `eta`, previous association flag | relative xy divided by `map_size`; queue divided by `queue_max_gu`; `eta` is raw spectral efficiency, not clipped; prev-assoc is `0/1` | zero-padded rows |
| `users_mask` | `(20,)` | valid user rows | `0/1` | indicates padded rows |
| `sats` | `(sats_obs_max, 9)` = `(6, 9)` | per visible satellite: relative ECEF pos, relative ECEF vel, Doppler, spectral efficiency, satellite queue | rel pos / `(r_earth + sat_height)`; rel vel / `(r_earth + sat_height)`; Doppler / `nu_max`; spectral efficiency raw; sat queue / `queue_max_sat` | zero-padded rows |
| `sats_mask` | `(6,)` | valid satellite rows | `0/1` | indicates padded rows |
| `nbrs` | `(nbrs_obs_max, 4)` = `(4, 4)` | nearest other UAVs: relative xy and relative velocity xy | relative xy / `map_size`; relative velocity / `v_max` | zero-padded rows |
| `nbrs_mask` | `(4,)` | valid neighbor rows | `0/1` | indicates padded rows |

Ownership split:

- Own-state only: `own`
- User / queue / association info: `users`, `users_mask`
- Satellite / backhaul / Doppler / sat-queue info: `sats`, `sats_mask`
- Neighbor collision geometry: `nbrs`, `nbrs_mask`

Normalization / clipping / masking:

- Environment-side normalization is manual division by known scales inside `_get_obs()`.
- There is no environment-side clipping of obs values except natural zero padding.
- Masks are explicit for users, satellites, and neighbors.
- The actor applies `LayerNorm(obs_dim)` because this run has `input_norm_enabled: true`.

Per-field notes that matter for line 3:

- `users[..., 3]` is raw spectral efficiency and can be much larger than `1`.
- `sats[..., 6]` is `doppler / nu_max`, but in practice it is not clipped and can exceed `[-1, 1]` by a lot.
- `nbrs[..., 2:4]` uses relative velocity divided by `v_max`, so the theoretical component range is about `[-2, 2]`.

One real obs sample from `uav_0` at the start of the reproduced bad episode:

- File: `runs/ablation_followup/s1_safe_static_v2_warm250/obs_sample_eval_ep18_uav0.json`
- `own = [0.1195, 0.1462, 0.0, 0.0, 1.0, 0.02, 0.0]`
- `users_mask` is all `1`, so this agent sees a full `20`-GU candidate set.
- `sats_mask = [1, 1, 1, 0, 0, 0]`, so only `3` of `6` satellite slots are valid in that frame.
- Sample stats for that frame:
  - `own`: min `0.0`, max `1.0`
  - `users`: min `-0.118`, max `8.665`
  - `sats`: min `-23.283`, max `23.283`
  - `nbrs`: min `-0.017`, max `0.136`

## 2. Action space definition

Core code:

- `sagin_marl/env/sagin_env.py:218` `_build_spaces()`
- `sagin_marl/env/sagin_env.py:500` `_apply_uav_dynamics()`
- `sagin_marl/env/sagin_env.py:709` `_compute_access_rates()`
- `sagin_marl/env/sagin_env.py:847` `_select_satellites()`
- `sagin_marl/rl/action_assembler.py:8` `assemble_actions()`

Full environment action space:

- `accel`: continuous `Box(-1, 1, shape=(2,))`
- `bw_logits`: continuous `Box(-bw_logit_scale, bw_logit_scale, shape=(users_obs_max,))`
- `sat_logits`: continuous `Box(-sat_logit_scale, sat_logit_scale, shape=(sats_obs_max,))`

Current best run actually trains / executes:

- `train_accel: true`
- `train_bw: false`
- `train_sat: false`
- `enable_bw_action: false`
- `fixed_satellite_strategy: true`

So the effective control space for this run is only:

- `2D` continuous acceleration

Action semantics:

- `accel[0]`, `accel[1]`: normalized xy acceleration commands
- Physical mapping in env step: `clip(action, -1, 1) * a_max`
- Then velocity update: `uav_vel <- clip(uav_vel + accel * tau0, -v_max, v_max)`
- Then position update: `uav_pos <- uav_pos + uav_vel * tau0`
- Boundary handling: `reflect` mode plus final position clip to `[0, map_size]`

If bandwidth control is enabled:

- `bw_logits` is softmaxed only over valid associated candidate users
- Invalid / non-associated slots are masked to zero
- With `enable_bw_action: false`, env falls back to equal share over associated users

If satellite control is enabled:

- `sat_logits` is only read on the visible candidate subset
- Doppler-invalid satellites get logit `-1e9`
- Selection is top-k or sampling depending on `sat_select_mode`
- With `fixed_satellite_strategy: true`, env ignores `sat_logits` and chooses the nearest visible satellite

Safety / override / clamp:

- `safe-static` is action-side, inside `_apply_uav_dynamics()`, not obs-side.
- Avoidance layer:
  - builds repulsive acceleration `a_rep` when pair distance is below `avoidance_alert_factor * d_safe`
  - current run sets `avoidance_enabled: true`, `avoidance_eta: 4.0`, `avoidance_alert_factor: 2.0`
  - repulsion is added after policy accel scaling, then clipped to `[-a_max, a_max]`
- Energy safety layer exists but is off in this run because `energy_enabled: false`

## 3. Policy network structure

Core code:

- `sagin_marl/rl/policy.py:24` `flatten_obs()`
- `sagin_marl/rl/policy.py:64` `ActorNet`
- `sagin_marl/rl/policy.py:134` `ActorNet.act()`
- `sagin_marl/rl/policy.py:178` `ActorNet.evaluate_actions_parts()`
- `sagin_marl/rl/critic.py:8` `CriticNet`

Actor:

- Input encoding: simple flatten-and-concatenate of all obs blocks, no attention / RNN / set encoder
- Input dim for this run: `207`
- Optional input normalization: `LayerNorm(207)` is enabled
- Backbone: `Linear(207, 256) -> ReLU -> Linear(256, 256) -> ReLU`
- Accel head: `Linear(256, 2)` plus global learnable `log_std(2)`
- If enabled, extra heads exist:
  - `bw_head: Linear(256, users_obs_max)`
  - `sat_head: Linear(256, sats_obs_max)`
- Distribution: squashed Gaussian (`Normal` -> `tanh`)
- Current run only uses the accel head

Critic:

- Input is centralized global state from `_build_global_state()`
- State dim for this run: `142`
- Optional `LayerNorm(142)` is enabled
- Backbone: `Linear(142, 256) -> ReLU -> Linear(256, 256) -> ReLU`
- Value head: `Linear(256, 1)`

Parameter sharing:

- One shared actor is used for all UAV agents
- One shared centralized critic is used for all UAV agents
- There is no per-agent parameter specialization

## 4. Reward, termination, and safe-static position

Core code:

- `sagin_marl/env/sagin_env.py:414` `step()`
- `sagin_marl/env/sagin_env.py:999` `_compute_reward()`
- `sagin_marl/env/sagin_env.py:500` `_apply_uav_dynamics()`
- `sagin_marl/env/config.py:304` `ablation_flag()`

Reward structure in code:

- `raw_reward = term_service + term_drop + term_queue + term_q_delta + term_centroid + term_accel + term_energy`
- Final penalty added after that:
  - collision penalty
  - battery penalty

Current run effective weights from `config.yaml`:

- `eta_service = 0.0`
- `eta_drop = 1.3`
- `eta_drop_step = 10.0`
- `omega_q = 1.0`
- `eta_q_delta = 2.0`
- `eta_accel = 0.02`
- `eta_crash = 5.0`
- `eta_centroid = 0.0`
- `omega_e = 0.0`

Meaning for this run:

- Queue penalty and queue-delta are the main dense terms.
- Acceleration magnitude has a small penalty.
- Collision is a hard terminal penalty of `-5.0`.
- Energy term is inactive.
- Centroid term is inactive.

Termination / truncation:

- `terminated = collision or energy_depleted`
- `truncated = (t >= T_steps - 1)`
- With this run, termination is effectively collision-only because energy is disabled.

Where `safe-static` actually acts:

- Not before obs construction
- Not inside reward
- It acts after the policy action is produced and before dynamics are applied, inside `_apply_uav_dynamics()`

## 5. One reproduced failing episode

Source summary row:

- `runs/ablation_followup/s1_safe_static_v2_warm250/eval_n20_seed43000.csv`
- Episode `18`
- `reward_sum = -24.1075`
- `steps = 382`
- `termination_reason = collision`
- `min_inter_uav_dist = 12.4734`

Step-level dump reproduced from the same evaluation sequence:

- File: `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep18_seed43000base.csv`

Key facts from that dump:

- Queue-heavy only at the start:
  - step `0`: `queue_total_active = 3,068,838.5`, reward `-9.9011`
  - step `1`: `queue_total_active = 1,035,799.5`, reward `-9.9011`
  - step `2`: `queue_total_active = 144,799.4`, reward `+1.5144`
  - step `3`: `queue_total_active = 0.0`, reward `+0.2882`
- After step `3`, the episode is basically queue-cleared and most rewards are just a small accel penalty.
- Collision happens very late:
  - step `381`: `reward = -5.0026`
  - `reward_raw = -0.0026`
  - `collision_penalty = -5.0`
  - `min_inter_uav_dist = 12.4734`
- Safety takeover is rare but real:
  - nonzero action correction only at steps `230` and `381`
  - max correction norm for `uav_0` and `uav_1` is `1.8378`

Important inference from this episode:

- This is not a queue-collapse episode.
- It is a late-collision episode after the queue has already been driven to zero.
- So for this checkpoint, one failure mode is action / geometry stability, not reward backlog handling.

## 6. Fixed baseline interface comparison

Core code:

- `scripts/evaluate.py:109` `_baseline_actions()`
- `sagin_marl/rl/baselines.py:8` `zero_accel_policy()`
- `sagin_marl/rl/baselines.py:64` `queue_aware_policy()`

What `fixed` means in this repo:

- `baseline="fixed"` maps to `zero_accel_policy(num_agents)`
- It does not read obs at all
- It outputs `accel = [0, 0]` for every UAV
- BW and sat actions are absent / zero

Fairness vs learned policy in the current run:

- Same environment
- Same reward
- Same safety layer
- Same association and backhaul mechanics
- Learned policy has more information and more freedom than `fixed`, not less

Same-seed comparison on `20` episodes with base seed `43000`:

- Trained policy mean reward: `-4.6008`
- Fixed baseline mean reward: `-1.9437`
- Trained collision rate: `0.35`
- Fixed collision rate: `0.0`
- Trained mean active queue: `3795.21`
- Fixed mean active queue: `1866.91`
- Trained mean minimum inter-UAV distance: `38.06`
- Fixed mean minimum inter-UAV distance: `180.14`

Important inference:

- In this comparison, learned accel underperforms the fixed zero-accel baseline.
- That is not because `fixed` gets extra information. It gets less.

## 7. Training config

Primary files:

- `runs/ablation_followup/s1_safe_static_v2_warm250/config.yaml`
- `runs/ablation_followup/s1_safe_static_v2_warm250/config_source.yaml`

Most relevant settings:

- Action scope:
  - `enable_bw_action: false`
  - `fixed_satellite_strategy: true`
  - `train_accel: true`
  - `train_bw: false`
  - `train_sat: false`
- Obs / candidate selection:
  - `candidate_mode: nearest`
  - `candidate_k: 20`
  - `traffic_level: 1`
- Safety:
  - `avoidance_enabled: true`
  - `avoidance_eta: 4.0`
  - `avoidance_alert_factor: 2.0`
- Reward:
  - `queue_penalty_mode: linear`
  - `omega_q: 1.0`
  - `eta_drop: 1.3`
  - `eta_drop_step: 10.0`
  - `eta_q_delta: 2.0`
  - `eta_accel: 0.02`
- PPO:
  - `buffer_size: 400`
  - `num_mini_batch: 4`
  - `ppo_epochs: 5`
  - `gamma: 0.99`
  - `gae_lambda: 0.95`
  - `clip_ratio: 0.2`
  - `adv_clip: 5.0`
  - `actor_lr: 3e-4`
  - `critic_lr: 1e-4`
  - `entropy_coef: 0.001`
  - `value_coef: 0.5`
  - `input_norm_enabled: true`

## 8. Short takeaways for line 3

These are not full recommendations yet, just the strongest signals visible from the materials above:

- Obs scale mismatch is real.
  - User `eta` can be around `8+`.
  - Satellite Doppler normalized by `nu_max` still reaches about `23` in a real sample.
  - Everything is flattened into one MLP, so line 3 should treat representation scale seriously.
- Current failure is not only reward-side.
  - The reproduced bad episode clears queue by step `3` and still dies by collision at step `381`.
- Fixed zero-accel already beats the learned accel policy.
  - That points to action geometry / control stability as a first-class suspect.
