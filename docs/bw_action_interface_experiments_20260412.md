# BW Action Interface Experiments (2026-04-12)

This note summarizes the action-interface experiments run on the fixed
`deadline_mild + obs proxy` BW-only T=10 environment after the environment-side
leverage redesign work.

## Goal

The question was no longer "does the environment contain any BW leverage", but:

- can plain PPO learn that leverage more reliably if we reduce the BW action
  semantics from "absolute simplex over all GU" to a more local, structured
  redistribution interface?

All experiments below kept the environment fixed and only changed the BW actor
interface.

## Environment

- Config family: `structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild*.yaml`
- Common environment:
  - `1 UAV`
  - `5 GU`
  - `T=10`
  - `deadline_mild + obs proxy`
  - `train_accel=false`
  - `train_sat=false`
  - `train_bw=true`
- Common PPO defaults:
  - `actor_lr=3e-4`
  - `critic_lr=3e-4`
  - `ppo_epochs=4`
  - `num_mini_batch=4`
  - `exec_accel_source=zero`
  - `exec_sat_source=zero`
  - `exec_bw_source=policy`

Reference heuristic on this environment:

- reward `0.583`
- processed `1.075`
- drop `0.376`
- backlog `3.101`

## 1. Residual Mean-Only Over Heuristic

Config:

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_residual_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_residual_meanonly.yaml)

Implementation:

- parameterization: `score_residual_fixedkappa_dirichlet`
- base action: queue-aware heuristic reconstructed from structured BW local features
- actor learns a zero-sum bounded residual over the heuristic base
- `kappa` fixed at `24`
- per-GU floor `0.01`

Code:

- [structured_actor.py](/D:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py)
- [structured_factory.py](/D:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_factory.py)
- [config.py](/D:/研三上/毕设/sagin_marl/sagin_marl/env/config.py)

Runs:

- smoke: [structured_bw_t10_deadline_mild_residual_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_residual_meanonly_smoke_u1_20260412)
- train: [structured_bw_t10_deadline_mild_residual_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_residual_meanonly_u20_20260412)
- same-eval compare: [bw_deadline_mild_residual_meanonly_compare_20260412.json](/D:/研三上/毕设/sagin_marl/runs/analysis/bw_deadline_mild_residual_meanonly_compare_20260412.json)

Result:

- fresh init: reward `0.476`
- learned u20: reward `0.448`
- heuristic: reward `0.583`
- final `approx_kl_bw=0.0867`
- final `clip_frac_bw=0.065`

Takeaway:

- This change is too small as a learning intervention.
- It mainly injects a strong heuristic prior.
- PPO updates still happen, but training does not improve over that prior.

## 2. Top-2 Focus On Extra Mass

Config:

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_focus2_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_focus2_meanonly.yaml)

Implementation:

- parameterization: `score_focus2_fixedkappa_dirichlet`
- base action: queue-aware heuristic base
- actor picks a top-2 focus set and interpolates from the heuristic base toward a
  floor-preserving top-2 target
- `kappa` fixed at `24`
- per-GU floor `0.01`

Runs:

- smoke: [structured_bw_t10_deadline_mild_focus2_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_smoke_u1_20260412)
- u20: [structured_bw_t10_deadline_mild_focus2_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_u20_20260412)
- u20 compare: [bw_deadline_mild_focus2_meanonly_compare_20260412.json](/D:/研三上/毕设/sagin_marl/runs/analysis/bw_deadline_mild_focus2_meanonly_compare_20260412.json)
- u100: [structured_bw_t10_deadline_mild_focus2_meanonly_u100_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_u100_20260412)
- u100 compare: [bw_deadline_mild_focus2_meanonly_u100_compare_20260412.json](/D:/研三上/毕设/sagin_marl/runs/analysis/bw_deadline_mild_focus2_meanonly_u100_compare_20260412.json)

Result:

- u20 checkpoint reward `0.310`
- u20 same-eval learned reward `0.253`
- u20 same-eval heuristic reward `0.583`
- u20 final `approx_kl_bw=0.760`
- u20 final `clip_frac_bw=0.280`

- u100 checkpoint reward `0.435`
- u100 same-eval learned reward `0.454`
- u100 same-eval heuristic reward `0.583`
- u100 final `approx_kl_bw=0.099`
- u100 final `clip_frac_bw=0.195`

Takeaway:

- This is the only tested action redesign that produced a clear training gain.
- It is directionally helpful.
- But even after `u100`, plain PPO still stays below heuristic by about `0.13`.
- So action compression helps, but does not by itself solve the learning problem.

## 3. Donor / Receiver + Delta

Config:

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_transfer_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_transfer_meanonly.yaml)

Implementation:

- parameterization: `score_transfer_fixedkappa_dirichlet`
- base action: queue-aware heuristic base
- actor builds a donor/receiver transfer around that base
- `kappa` fixed at `24`
- per-GU floor `0.01`

Runs:

- smoke: [structured_bw_t10_deadline_mild_transfer_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_transfer_meanonly_smoke_u1_20260412)
- train: [structured_bw_t10_deadline_mild_transfer_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_transfer_meanonly_u20_20260412)
- same-eval compare: [bw_deadline_mild_transfer_meanonly_compare_20260412.json](/D:/研三上/毕设/sagin_marl/runs/analysis/bw_deadline_mild_transfer_meanonly_compare_20260412.json)

Result:

- fresh init: reward `0.323`
- learned u20: reward `0.102`
- heuristic: reward `0.583`
- final `approx_kl_bw=1.279`
- final `clip_frac_bw=0.262`

Takeaway:

- This interface is too brittle under the current PPO update.
- It over-compresses the action into a hard pair-transfer semantics.
- PPO pushes it hard, but learning becomes unstable and degrades from its own prior.

## Summary

If the success criterion is "plain PPO on this environment beats the heuristic",
all three tested action redesigns still fail.

But they do not fail in the same way:

- `residual mean-only`: too weak, mostly changes the prior
- `focus2`: directionally right, gives the strongest learning improvement
- `donor/receiver+delta`: too brittle under the current scalar-advantage PPO update

## Interpretation

These results suggest:

1. The action redesign direction is not wrong.
   Evidence: `focus2` clearly outperforms the original absolute-simplex PPO and
   keeps improving up to `u100`.

2. But the action redesign alone is not enough.
   All variants still use the same stage-level scalar advantage and the same
   critic/value path.

3. The remaining bottleneck is no longer "absolute simplex is too hard" alone.
   It is more like:
   - action semantics matter
   - but the update target is still too noisy / too global
   - so PPO cannot fully exploit the improved interface

## Working Hypothesis

The evidence now supports:

- environment leverage exists
- absolute-simplex action semantics were making PPO worse
- `focus2` is a better actor interface
- but plain state-value PPO still does not produce a clean enough learning signal
  to make that interface fully work

So the current best interpretation is:

- the direction of the action change is right
- the current change is still not sufficient
- the next likely step is **best action interface + better credit / teacher path**
  rather than either:
  - continuing to harden the environment
  - or continuing to tune the original absolute-simplex PPO
