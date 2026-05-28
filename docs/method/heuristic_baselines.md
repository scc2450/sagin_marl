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

## Integrating Into Evaluation
1. Add a new `--baseline` choice in `scripts/evaluate.py`.
2. Call your heuristic function to generate actions when the baseline is selected.
3. Keep the output format unchanged so you can compare CSV results.
