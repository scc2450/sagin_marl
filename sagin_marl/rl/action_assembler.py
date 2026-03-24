from __future__ import annotations

from typing import Dict

import numpy as np


def assemble_actions(
    cfg,
    agents,
    accel_actions,
    bw_alloc=None,
    sat_select_mask=None,
    bw_logits=None,
    sat_logits=None,
) -> Dict[str, Dict]:
    bw_values = bw_alloc if bw_alloc is not None else bw_logits
    sat_values = sat_select_mask if sat_select_mask is not None else sat_logits
    actions = {}
    for i, agent in enumerate(agents):
        act = {
            "accel": accel_actions[i].astype(np.float32),
            "bw_alloc": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_select_mask": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        if cfg.enable_bw_action and bw_values is not None:
            act["bw_alloc"] = np.clip(bw_values[i].astype(np.float32), 0.0, 1.0)
        if not cfg.fixed_satellite_strategy and sat_values is not None:
            act["sat_select_mask"] = np.clip(sat_values[i].astype(np.float32), 0.0, 1.0)
        actions[agent] = act
    return actions
