# Heuristic Baselines (Greedy, Rule-Based)

This note explains the rough steps to build a greedy or other heuristic baseline.
Keep it simple, deterministic, and mask-aware so it is easy to compare with learned policies.

## What You Need
1. Per-agent observation dict from the env (`obs[agent]`).
2. The config (`cfg`) for sizes and feature flags.
3. A way to package actions (`assemble_actions`).

## Suggested Steps
1. Decide acceleration (`accel`).
Use the relative position in `users[:, 0:2]` and choose a target user.
Common targets are the max queue user (`users[:, 2]`) or max spectral efficiency user (`users[:, 3]`).
2. Decide bandwidth logits (`bw_logits`) if `cfg.enable_bw_action` is true.
Use a score such as `queue`, `queue * se`, or a one-hot on the best candidate.
3. Decide satellite logits (`sat_logits`) if `cfg.fixed_satellite_strategy` is false.
Use a score such as `sats[:, 7]` (SE) minus `sats[:, 8]` (queue), and zero out invalid entries.
4. Apply masks (`users_mask`, `sats_mask`) before argmax or scoring.
5. Use `assemble_actions(cfg, env.agents, accel, bw_logits, sat_logits)` to build the action dict.

## Minimal Skeleton
```python
import numpy as np
from sagin_marl.rl.action_assembler import assemble_actions


def greedy_baseline(obs_by_agent, cfg, agents):
    accel = np.zeros((len(agents), 2), dtype=np.float32)
    bw_logits = np.zeros((len(agents), cfg.users_obs_max), dtype=np.float32)
    sat_logits = np.zeros((len(agents), cfg.sats_obs_max), dtype=np.float32)

    for i, agent in enumerate(agents):
        obs = obs_by_agent[agent]

        users = obs["users"]
        users_mask = obs["users_mask"]
        if users_mask.any():
            scores = users[:, 2] * users_mask
            k = int(np.argmax(scores))
            rel = users[k, 0:2]
            norm = float(np.linalg.norm(rel) + 1e-6)
            accel[i] = rel / norm
            bw_logits[i] = scores

        if not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            sats_mask = obs["sats_mask"]
            if sats_mask.any():
                sat_logits[i] = (sats[:, 7] - sats[:, 8]) * sats_mask

    return assemble_actions(cfg, agents, accel, bw_logits=bw_logits, sat_logits=sat_logits)
```

## Current Structured/Native Evaluation

For the current joint MC-GAE mainline, prefer the structured native evaluator:

```bash
python scripts/evaluate_structured_mixed_heads_native.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --baseline_policy cluster_center_queue_aware \
  --episodes 64 \
  --num_envs 64 \
  --episode_seed_base 900000 \
  --device cuda \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --out_dir runs/diagnostics/<run_name>/native_eval_rule \
  --label rule
```

Supported current rule/MaxWeight baseline IDs include:

| ID | Exec sources |
|---|---|
| `static_uniform` | `zero`, `uniform`, `uniform` |
| `random_feasible` | `random`, `random`, `random` |
| `link_priority` | `zero`, `link_priority`, `link_priority` |
| `demand_priority` | `zero`, `demand_priority`, `demand_priority` |
| `queue_aware` | `queue_aware`, `queue_aware`, `queue_aware` |
| `cluster_center_queue_aware` | `cluster_center_queue_aware`, `queue_aware`, `queue_aware` |
| `maxweight_lyapunov` | `lyapunov`, `lyapunov`, `lyapunov` |
| `dpp_no_mobility` | `zero`, `lyapunov`, `lyapunov` |
| `dpp_equal_bw` | `lyapunov`, `lyapunov`, `uniform` |
| `dpp_greedy_sat` | `lyapunov`, `queue_aware`, `lyapunov` |
| `topology_dpp_bw` | `zero`, `zero`, `topology_dpp_bw` |
| `dpp_resource_bw` | legacy alias for `topology_dpp_bw` |
| `dpp_resource_hybrid_native` | `cluster_center_queue_aware`, `queue_aware`, `topology_dpp_bw` |
| `topology_dpp_native_bw_sat_cached` | `cluster_center_queue_aware`, `topology_dpp_sat`, `topology_dpp_bw` |
| `full_topology_dpp_joint` | `topology_dpp_accel`, `topology_dpp_sat`, `topology_dpp_bw` |
| `topology_dpp` | structured Python fallback; no native source triple yet |
| `dpp_resource_hybrid` | structured Python fallback; cluster-center accel with topology-DPP BW/SAT |

`lyapunov` is still accepted as a compatibility alias for `maxweight_lyapunov`.
`topology_dpp` and the legacy `dpp_resource_hybrid` are intentionally not added to `_FIXED_POLICY_EXEC_SOURCE_MAP`: they require Python-side topology/resource reasoning before choosing BW/SAT decisions. `dpp_resource_hybrid_native` is the fast staged-source variant: it keeps cluster-center accel and queue-aware SAT in native kernels, then uses the native `topology_dpp_bw` source for BW allocation. `topology_dpp_native_bw_sat_cached` is the first native SAT/BW-coupled step: accel remains cluster-center, SAT uses a topology-DPP queue-gap/backhaul proxy source, and BW uses native `topology_dpp_bw`. `full_topology_dpp_joint` adds native accel candidate enumeration before the same SAT/BW resource sources. `dpp_resource_bw` remains accepted as a legacy source/baseline alias.

## Integrating A New Native Baseline

1. Add or reuse a source mode in `sagin_marl/rl/structured_eval.py`.
2. If the baseline can be composed from existing native sources, only add an entry to `_FIXED_POLICY_EXEC_SOURCE_MAP`.
3. If it needs new behavior at runtime, add a Python prototype first, then add/validate the native kernel path.
4. Keep output fields unchanged so `scripts/evaluation/evaluate_thesis_native_methods.py` can compare CSV/JSON summaries.
