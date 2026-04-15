from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def zero_accel_policy(num_agents: int) -> np.ndarray:
    return np.zeros((num_agents, 2), dtype=np.float32)


def random_accel_policy(
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=(num_agents, 2)).astype(np.float32)


def centroid_accel_policy(
    obs_list: List[Dict[str, np.ndarray]],
    gain: float = 2.0,
    queue_weighted: bool = True,
) -> np.ndarray:
    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        if not np.any(users_mask):
            continue
        rel = users[users_mask, 0:2]
        vec = np.mean(rel, axis=0)
        if queue_weighted and users.shape[1] >= 3:
            q = np.clip(users[users_mask, 2], 0.0, None)
            q_sum = float(np.sum(q))
            if q_sum > 1e-6:
                vec = (rel * q[:, None]).sum(axis=0) / (q_sum + 1e-9)
        accel[i] = np.clip(vec * gain, -1.0, 1.0).astype(np.float32)
    return accel


def _baseline_energy_term(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    accel_vec = np.zeros((2,), dtype=np.float32)
    energy_weight = float(getattr(cfg, "baseline_energy_weight", 1.0))
    if not cfg.energy_enabled or energy_weight <= 0.0:
        return accel_vec

    energy_low = float(getattr(cfg, "baseline_energy_low", 0.3))
    energy_norm = float(obs["own"][4])
    if energy_norm >= energy_low:
        return accel_vec

    vel = obs["own"][2:4].astype(np.float32)
    speed = float(np.linalg.norm(vel))
    if speed <= 1e-6:
        return accel_vec

    target_speed = min(cfg.uav_opt_speed / max(cfg.v_max, 1e-6), 1.0)
    delta = target_speed - speed
    if delta >= 0.0:
        return accel_vec

    scale = (energy_low - energy_norm) / max(energy_low, 1e-6)
    accel_vec = accel_vec + energy_weight * scale * (vel / speed) * delta
    return accel_vec.astype(np.float32, copy=False)


def _select_cluster_targets(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    cluster_centers: np.ndarray,
    cluster_counts: np.ndarray,
) -> np.ndarray:
    num_agents = len(obs_list)
    targets = np.full((num_agents,), -1, dtype=np.int32)
    if num_agents <= 0:
        return targets

    centers = np.asarray(cluster_centers, dtype=np.float32)
    counts = np.asarray(cluster_counts, dtype=np.float32).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 2 or counts.size != centers.shape[0]:
        return targets

    valid = np.flatnonzero(counts > 0.0)
    if valid.size == 0:
        return targets

    priority = valid[np.argsort(-counts[valid], kind="stable")]
    selected = priority[: min(num_agents, priority.size)]
    uav_pos = np.asarray([obs["own"][0:2] * cfg.map_size for obs in obs_list], dtype=np.float32)

    remaining_uavs = list(range(num_agents))
    for cluster_idx in selected:
        rem = np.asarray(remaining_uavs, dtype=np.int32)
        dists = np.linalg.norm(uav_pos[rem] - centers[cluster_idx], axis=1)
        best_uav = int(rem[int(np.argmin(dists))])
        targets[best_uav] = int(cluster_idx)
        remaining_uavs.remove(best_uav)

    if selected.size > 0:
        selected_centers = centers[selected]
        for u in remaining_uavs:
            dists = np.linalg.norm(selected_centers - uav_pos[u], axis=1)
            targets[u] = int(selected[int(np.argmin(dists))])

    return targets


def _cluster_tracking_term(obs: Dict[str, np.ndarray], cfg, target_abs: np.ndarray) -> np.ndarray:
    own = obs["own"]
    pos = own[0:2].astype(np.float32) * cfg.map_size
    vel = own[2:4].astype(np.float32) * cfg.v_max
    error = np.asarray(target_abs, dtype=np.float32) - pos
    dist = float(np.linalg.norm(error))
    speed = float(np.linalg.norm(vel))

    stop_radius = max(float(getattr(cfg, "baseline_cluster_stop_radius", 20.0) or 0.0), 0.0)
    speed_tol = max(float(getattr(cfg, "baseline_cluster_speed_tol", 2.0) or 0.0), 0.0)
    slow_radius_cfg = float(getattr(cfg, "baseline_cluster_slow_radius", 120.0) or 0.0)
    slow_radius = max(slow_radius_cfg, stop_radius + 1e-6)
    cruise_speed_cfg = getattr(cfg, "baseline_cluster_cruise_speed", None)
    cruise_speed = cfg.uav_opt_speed if cruise_speed_cfg is None else float(cruise_speed_cfg)
    cruise_speed = float(np.clip(cruise_speed, 0.0, cfg.v_max))
    vel_gain = max(float(getattr(cfg, "baseline_cluster_vel_gain", 1.0) or 0.0), 0.0)

    if dist <= stop_radius:
        if speed <= speed_tol:
            return np.zeros((2,), dtype=np.float32)
        desired_vel = np.zeros((2,), dtype=np.float32)
    else:
        direction = error / max(dist, 1e-6)
        desired_speed = cruise_speed * min(dist / slow_radius, 1.0)
        desired_vel = direction * desired_speed

    desired_accel = vel_gain * (desired_vel - vel) / max(cfg.tau0, 1e-6)
    action = desired_accel / max(cfg.a_max, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def cluster_center_accel_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    cluster_centers: np.ndarray | None,
    cluster_counts: np.ndarray | None,
) -> np.ndarray:
    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    if num_agents <= 0 or cluster_centers is None or cluster_counts is None:
        return accel

    centers = np.asarray(cluster_centers, dtype=np.float32)
    targets = _select_cluster_targets(obs_list, cfg, centers, np.asarray(cluster_counts, dtype=np.float32))
    for i, obs in enumerate(obs_list):
        accel_vec = np.zeros((2,), dtype=np.float32)
        target_idx = int(targets[i])
        if 0 <= target_idx < len(centers):
            accel_vec = accel_vec + _cluster_tracking_term(obs, cfg, centers[target_idx])
        accel_vec = accel_vec + _baseline_energy_term(obs, cfg)
        accel[i] = np.clip(accel_vec, -1.0, 1.0)
    return accel

def uniform_bw_policy(num_agents: int, users_obs_max: int) -> np.ndarray:
    if users_obs_max <= 0:
        return np.zeros((num_agents, 0), dtype=np.float32)
    return np.full((num_agents, users_obs_max), 1.0 / float(users_obs_max), dtype=np.float32)

def random_bw_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.random(size=(num_agents, cfg.users_obs_max)).astype(np.float32)

def uniform_sat_policy(num_agents: int, sats_obs_max: int) -> np.ndarray:
    return np.zeros((num_agents, sats_obs_max), dtype=np.float32)

def random_sat_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.random(size=(num_agents, cfg.sats_obs_max)) > 0.5).astype(np.float32)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return vals
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    if v_max - v_min <= 1e-9:
        return np.zeros_like(vals, dtype=np.float32)
    return ((vals - v_min) / (v_max - v_min)).astype(np.float32, copy=False)


def _sat_heuristic_score(sats: np.ndarray, mask: np.ndarray, cfg) -> np.ndarray:
    score = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    if not np.any(mask):
        return score

    sat_feat = sats[mask]
    se = np.asarray(sat_feat[:, 7], dtype=np.float32)
    qsat = np.asarray(sat_feat[:, 8], dtype=np.float32)
    load_norm = np.asarray(sat_feat[:, 9], dtype=np.float32)
    bw_ratio = np.asarray(sat_feat[:, 10], dtype=np.float32)
    stay = np.asarray(sat_feat[:, 11], dtype=np.float32)

    se_weight = float(getattr(cfg, "baseline_sat_se_weight", 1.0) or 0.0)
    q_penalty = float(getattr(cfg, "baseline_sat_queue_penalty", 0.5) or 0.0)
    load_penalty = float(getattr(cfg, "baseline_sat_load_penalty", 1.0) or 0.0)
    bw_reward = float(getattr(cfg, "baseline_sat_bw_reward", 0.75) or 0.0)
    stay_bonus = float(getattr(cfg, "baseline_sat_stay_bonus", 0.25) or 0.0)
    switch_margin = max(float(getattr(cfg, "baseline_sat_switch_margin", 0.15) or 0.0), 0.0)

    projected_count = 1.0 / np.clip(bw_ratio, 1e-6, 1.0)
    projected_load_term = np.log1p(projected_count)

    se_norm = _minmax_normalize(se)
    q_norm = _minmax_normalize(qsat)
    load_term_norm = _minmax_normalize(projected_load_term)
    bw_norm = _minmax_normalize(bw_ratio)

    score_slice = (
        se_weight * se_norm
        - q_penalty * q_norm
        - load_penalty * load_term_norm
        + bw_reward * bw_norm
        + stay_bonus * stay
    ).astype(np.float32, copy=False)

    current_idx = np.flatnonzero(stay > 0.5)
    if cfg.N_RF == 1 and current_idx.size == 1:
        cur = int(current_idx[0])
        best = int(np.argmax(score_slice))
        if best != cur and score_slice[best] <= score_slice[cur] + switch_margin:
            score_slice[cur] = score_slice[best] + 1e-3

    score[mask] = np.clip(score_slice, -cfg.sat_logit_scale, cfg.sat_logit_scale)
    return score


def _topk_select_mask(scores: np.ndarray, valid_mask: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0 or k <= 0:
        return out
    order = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    keep = order[: min(k, order.size)]
    out[keep] = 1.0
    return out


def _masked_softmax(scores: np.ndarray, valid_mask: np.ndarray, temperature: float) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    idx = np.flatnonzero(valid_mask)
    if idx.size == 0:
        return out
    temp = max(float(temperature), 1e-3)
    logits = np.asarray(scores[idx], dtype=np.float32) / temp
    logits = logits - float(np.max(logits))
    exps = np.exp(logits)
    denom = float(np.sum(exps))
    if denom <= 1e-9:
        out[idx] = 1.0 / float(idx.size)
        return out
    out[idx] = exps / denom
    return out


def _lyapunov_state_init(num_agents: int, cfg) -> Dict[str, np.ndarray]:
    return {
        "pressure_ema": np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32),
        "virtual_queue": np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32),
        "service_est": np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32),
        "prev_accel": np.zeros((num_agents, 2), dtype=np.float32),
        "dpp_access_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_backhaul_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_reg_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_objective_term": np.zeros((num_agents,), dtype=np.float32),
    }


def _baseline_repulse_term(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0
    if repulse_gain <= 0.0 or repulse_radius <= 0.0:
        return np.zeros((2,), dtype=np.float32)

    nbrs = obs["nbrs"]
    nbrs_mask = obs["nbrs_mask"] > 0.0
    if not np.any(nbrs_mask):
        return np.zeros((2,), dtype=np.float32)

    rel_nbr_pos = np.asarray(nbrs[nbrs_mask, 0:2], dtype=np.float32)
    rel_nbr_vel = np.asarray(nbrs[nbrs_mask, 2:4], dtype=np.float32)
    dist_norm = np.linalg.norm(rel_nbr_pos, axis=1)
    dist = dist_norm * cfg.map_size
    mask = (dist > 1e-6) & (dist < repulse_radius)
    if not np.any(mask):
        return np.zeros((2,), dtype=np.float32)

    rel_sel = rel_nbr_pos[mask]
    vel_sel = rel_nbr_vel[mask]
    dist_sel = dist[mask]
    dist_norm_sel = dist_norm[mask]

    direction = rel_sel / dist_norm_sel[:, None]
    approach_speed = np.sum(vel_sel * direction, axis=1)
    spring_strength = (1.0 / dist_sel - 1.0 / repulse_radius)
    damper_strength = np.where(approach_speed < 0.0, -approach_speed, 0.0)
    strength = spring_strength + damper_strength
    return (-repulse_gain * (direction * strength[:, None]).sum(axis=0)).astype(np.float32, copy=False)


def _generate_accel_candidates(cfg) -> np.ndarray:
    num = max(int(getattr(cfg, "dpp_accel_num_candidates", 9) or 9), 1)
    step = float(np.clip(getattr(cfg, "dpp_accel_step_scale", 0.6), 0.0, 1.0))
    if num == 1 or step <= 1e-9:
        return np.zeros((1, 2), dtype=np.float32)

    candidates = [np.zeros((2,), dtype=np.float32)]
    for k in range(num - 1):
        theta = 2.0 * np.pi * float(k) / float(num - 1)
        candidates.append(np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32) * step)
    return np.asarray(candidates, dtype=np.float32)


def _predict_users_rel_after_accel(obs: Dict[str, np.ndarray], cfg, accel_vec: np.ndarray) -> np.ndarray:
    users = np.asarray(obs["users"], dtype=np.float32)
    rel = users[:, 0:2].copy()
    own = np.asarray(obs["own"], dtype=np.float32)
    vel_abs = own[2:4] * cfg.v_max
    delta_pos_abs = vel_abs * cfg.tau0 + 0.5 * np.asarray(accel_vec, dtype=np.float32) * cfg.a_max * (cfg.tau0**2)
    delta_rel_norm = delta_pos_abs / max(float(cfg.map_size), 1e-6)
    rel = rel - delta_rel_norm[None, :]
    return rel.astype(np.float32, copy=False)


def _predict_topology_after_accel(
    obs: Dict[str, np.ndarray], 
    cfg, 
    accel_vec: np.ndarray,
    agent_id: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Complete topology snapshot after applying acceleration.
    
    Uses env_callbacks (stored temporarily in cfg._dpp_env_callbacks_temp) to 
    compute actual link qualities, visibility under new position. Falls back to 
    heuristic approximations if callbacks unavailable.
    
    Returns Dict with:
    - rel_next: new relative positions (K × 2)
    - eta: link quality per user (K,) - recomputed based on new position
    - rate_approx: approximate service rates per user (K,)
    - sat_visible_mask: valid satellites after move (L,)
    - sat_rate_per_user: backhaul rate if routed through each sat (K × L) approx
    """
    users = np.asarray(obs["users"], dtype=np.float32)
    rel_next = _predict_users_rel_after_accel(obs, cfg, accel_vec)
    
    # Get env_callbacks from cfg if available (set temporarily by lyapunov_queue_aware_policy_step)
    env_cbs = getattr(cfg, "_dpp_env_callbacks_temp", {}) or {}
    if not env_cbs:
        env_cbs = {}
    
    # If compute_access_rates callback available, use it for topology-aware link prediction
    if "compute_access_rates" in env_cbs and callable(env_cbs.get("compute_access_rates")):
        try:
            eta_new, rate_new = env_cbs["compute_access_rates"](agent_id, accel_vec, obs, rel_next)
            eta = np.clip(np.asarray(eta_new, dtype=np.float32), 0.0, 2.0)[:cfg.users_obs_max]
            rate_approx = np.clip(np.asarray(rate_new, dtype=np.float32), 0.0, None)[:cfg.users_obs_max]
        except (ValueError, IndexError, TypeError):
            eta = _approx_eta_from_distance(rel_next, cfg)
            rate_approx = 0.5 * eta
    else:
        # Fallback: approximate eta from distance
        eta = _approx_eta_from_distance(rel_next, cfg)
        rate_approx = 0.5 * eta
    
    # SAT visibility 
    if "check_sat_visibility" in env_cbs and callable(env_cbs.get("check_sat_visibility")):
        try:
            sat_visible = env_cbs["check_sat_visibility"](agent_id, accel_vec, obs)
            sat_visible_mask = np.asarray(sat_visible, dtype=bool)[:cfg.sats_obs_max]
        except (ValueError, IndexError, TypeError):
            sat_visible_mask = obs.get("sats_mask", np.ones((cfg.sats_obs_max,), dtype=bool)) > 0.5
    else:
        sat_visible_mask = obs.get("sats_mask", np.ones((cfg.sats_obs_max,), dtype=bool)) > 0.5
    
    # SAT rates
    if "compute_backhaul_rates" in env_cbs and callable(env_cbs.get("compute_backhaul_rates")):
        try:
            # Note: compute_backhaul rates would need sat selections, which we don't have yet
            # So we skip for now and use approximation
            sat_rate_per_user = np.ones((cfg.users_obs_max, cfg.sats_obs_max), dtype=np.float32) * 0.1
            sat_rate_per_user[:, ~sat_visible_mask] = 0.0
        except:
            sat_rate_per_user = np.ones((cfg.users_obs_max, cfg.sats_obs_max), dtype=np.float32) * 0.1
            sat_rate_per_user[:, ~sat_visible_mask] = 0.0
    else:
        sat_rate_per_user = np.ones((cfg.users_obs_max, cfg.sats_obs_max), dtype=np.float32) * 0.1
        sat_rate_per_user[:, ~sat_visible_mask] = 0.0
    
    return {
        "rel_next": rel_next.astype(np.float32, copy=False),
        "eta": eta.astype(np.float32, copy=False),
        "rate_approx": rate_approx.astype(np.float32, copy=False),
        "sat_visible_mask": sat_visible_mask.astype(bool, copy=False),
        "sat_rate_per_user": sat_rate_per_user.astype(np.float32, copy=False),
    }


def _approx_eta_from_distance(rel_next: np.ndarray, cfg) -> np.ndarray:
    """Heuristic approximation of link quality (eta) based on distance."""
    dist = np.linalg.norm(rel_next, axis=1)
    range_norm = np.clip(dist / 0.5, 0.0, 1.0)  # close = 0, far = 1
    eta = np.clip(0.8 * (1.0 - 0.9 * range_norm) + 0.1, 0.1, 1.0)
    return eta.astype(np.float32, copy=False)


def _dpp_allocate_bw(scores: np.ndarray, valid_mask: np.ndarray, cfg) -> np.ndarray:
    probs = _masked_softmax(scores, valid_mask, float(getattr(cfg, "dpp_bw_temp", 0.55) or 0.55))
    count = int(np.sum(valid_mask))
    if count <= 0:
        return probs

    floor = float(np.clip(getattr(cfg, "dpp_bw_floor", 0.0), 0.0, 0.2))
    floor = min(floor, 0.99 / float(count))
    if floor > 0.0:
        probs = (1.0 - floor * float(count)) * probs
        probs[valid_mask] = probs[valid_mask] + floor
        denom = float(np.sum(probs[valid_mask]))
        if denom > 1e-9:
            probs[valid_mask] = probs[valid_mask] / denom
    return probs.astype(np.float32, copy=False)


def _dpp_sat_selection(obs: Dict[str, np.ndarray], cfg, own_q_norm: float) -> Tuple[np.ndarray, float]:
    sat_sel = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    if cfg.fixed_satellite_strategy:
        return sat_sel, 0.0

    sats = obs["sats"]
    sats_mask = obs["sats_mask"] > 0.0
    sat_valid_mask = np.asarray(obs.get("sat_valid_mask", obs["sats_mask"]) > 0.0)
    if not np.any(sats_mask):
        return sat_sel, 0.0

    base_sat_scores = _sat_heuristic_score(sats, sats_mask, cfg)
    sat_scores = np.asarray(base_sat_scores, dtype=np.float32)
    qsat = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    qsat[sats_mask] = np.clip(np.asarray(sats[sats_mask, 8], dtype=np.float32), 0.0, None)
    se = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    bw_ratio = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    se[sats_mask] = np.clip(np.asarray(sats[sats_mask, 7], dtype=np.float32), 0.0, None)
    bw_ratio[sats_mask] = np.clip(np.asarray(sats[sats_mask, 10], dtype=np.float32), 0.0, 1.0)
    backhaul_proxy = 0.5 * _minmax_normalize(se) + 0.5 * bw_ratio
    gap = np.clip(own_q_norm - qsat, 0.0, None)

    gap_w = float(max(getattr(cfg, "dpp_sat_queue_gap_weight", 1.0) or 0.0, 0.0))
    sat_scores = sat_scores + gap_w * gap * backhaul_proxy

    topm = int(getattr(cfg, "dpp_sat_candidate_topm", cfg.sats_obs_max) or cfg.sats_obs_max)
    topm = max(min(topm, cfg.sats_obs_max), 1)
    valid_idx = np.flatnonzero(sat_valid_mask)
    if valid_idx.size > topm:
        order = valid_idx[np.argsort(sat_scores[valid_idx])[::-1]]
        keep = order[:topm]
        topm_mask = np.zeros_like(sat_valid_mask, dtype=bool)
        topm_mask[keep] = True
        sat_valid_mask = sat_valid_mask & topm_mask

    max_select = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)
    if max_select <= 0:
        return sat_sel, 0.0

    valid_idx = np.flatnonzero(sat_valid_mask)
    if valid_idx.size == 0:
        return sat_sel, 0.0

    enum_budget = max(int(getattr(cfg, "dpp_sat_enum_max_subsets", 64) or 64), 1)
    subset_penalty = float(max(getattr(cfg, "dpp_sat_subset_penalty", 0.02) or 0.0, 0.0))
    contention_w = float(max(getattr(cfg, "dpp_sat_contention_weight", 0.15) or 0.0, 0.0))

    candidate_subsets: List[Tuple[int, ...]] = []
    for k in range(1, min(max_select, valid_idx.size) + 1):
        for comb in combinations(valid_idx.tolist(), k):
            candidate_subsets.append(comb)
            if len(candidate_subsets) >= enum_budget:
                break
        if len(candidate_subsets) >= enum_budget:
            break

    if not candidate_subsets:
        sat_sel = _topk_select_mask(sat_scores, sat_valid_mask, max_select)
        backhaul_term = float(np.sum(gap * backhaul_proxy * sat_sel))
        return sat_sel.astype(np.float32, copy=False), backhaul_term

    best_subset: Tuple[int, ...] = tuple()
    best_score = -1e30
    best_backhaul = 0.0
    for subset in candidate_subsets:
        idx = np.asarray(subset, dtype=np.int32)
        sel = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
        sel[idx] = 1.0
        backhaul_term = float(np.sum(gap * backhaul_proxy * sel))
        contention_penalty = float(np.sum((1.0 - bw_ratio[idx]) * sel[idx]))
        score = float(np.sum(sat_scores[idx])) + backhaul_term
        score = score - subset_penalty * float(len(subset) ** 2)
        score = score - contention_w * contention_penalty
        if score > best_score:
            best_score = score
            best_subset = subset
            best_backhaul = backhaul_term

    if len(best_subset) == 0:
        return sat_sel, 0.0
    sat_sel[np.asarray(best_subset, dtype=np.int32)] = 1.0
    return sat_sel.astype(np.float32, copy=False), best_backhaul


def _dpp_one_agent(
    obs: Dict[str, np.ndarray],
    cfg,
    prev_accel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    accel_candidates = _generate_accel_candidates(cfg)
    users_mask = obs["users_mask"] > 0.0
    bw_valid_mask = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0)
    users = np.asarray(obs["users"], dtype=np.float32)
    own_q_norm = float(np.clip(obs["own"][5], 0.0, None))

    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    max_users = int(getattr(cfg, "dpp_gu_max_select", 6) or 6)
    max_users = max(min(max_users, cfg.users_obs_max), 1)
    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    dpp_v = float(max(getattr(cfg, "baseline_lyapunov_v", 2.0) or 0.0, 0.0))
    access_w = float(max(getattr(cfg, "dpp_access_weight", 1.0) or 0.0, 0.0))
    backhaul_w = float(max(getattr(cfg, "dpp_backhaul_weight", 1.0) or 0.0, 0.0))
    accel_cost = float(max(getattr(cfg, "dpp_accel_cost", 0.08) or 0.0, 0.0))
    smooth_w = float(max(getattr(cfg, "dpp_smoothness", 0.0) or 0.0, 0.0))
    dist_penalty = float(max(getattr(cfg, "dpp_dist_penalty", 0.1) or 0.0, 0.0))
    service_scale = float(max(getattr(cfg, "baseline_lyapunov_bw_service_scale", 1.0) or 0.0, 0.0))

    best_score = -1e30
    best_accel = np.zeros((2,), dtype=np.float32)
    best_bw = np.zeros((cfg.users_obs_max,), dtype=np.float32)
    best_sat = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    best_pressure = np.zeros((cfg.users_obs_max,), dtype=np.float32)
    best_service = np.zeros((cfg.users_obs_max,), dtype=np.float32)
    best_access_term = 0.0
    best_backhaul_term = 0.0
    best_reg_term = 0.0

    def _eval_candidate(cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        pressure = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        service = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        bw = np.zeros((cfg.users_obs_max,), dtype=np.float32)

        # IMPROVED: Use complete topology prediction which may recalculate link qualities
        #           if env_callbacks are available. Otherwise falls back to approximation.
        topo = _predict_topology_after_accel(obs, cfg, cand)
        rel_next = topo["rel_next"]
        eta_updated = topo["eta"]  # NEW: Updated link qualities after position change
        sat_visible_mask = topo["sat_visible_mask"]  # NEW: Updated sat visibility
        
        potential_score = - 5e5 * np.mean(np.linalg.norm(rel_next, axis=1))  # heuristic potential score based on average distance to users after move
        if np.any(users_mask):
            q = np.clip(np.asarray(users[users_mask, 2], dtype=np.float32), 0.0, None)
            # IMPROVED: Use updated eta from topology instead of stale obs["users"][:,3]
            eta_obs = np.clip(np.asarray(users[users_mask, 3], dtype=np.float32), 0.0, 1.0)
            eta_new = np.clip(eta_updated[users_mask], 0.0, 1.0)
            # Blend old and new (old has established signal, new accounts for move)
            eta_blend = 0.4 * eta_obs + 0.6 * eta_new
            prev_assoc = np.clip(np.asarray(users[users_mask, 4], dtype=np.float32), 0.0, 1.0)
            rel_dist = np.linalg.norm(rel_next[users_mask], axis=1)
            gap = np.clip(q - own_q_norm, 0.0, None)
            rate_proxy = np.clip(0.5 + eta_blend, 0.0, 2.0)
            assoc_term = 1.0 + assoc_bonus * prev_assoc

            score_slice = gap * rate_proxy * assoc_term - dist_penalty * rel_dist
            pressure_slice = q * rate_proxy * assoc_term
            pressure[users_mask] = np.clip(pressure_slice, 0.0, None)

            valid_users = users_mask & bw_valid_mask
            if np.any(valid_users):
                candidate_scores = np.full((cfg.users_obs_max,), -1e6, dtype=np.float32)
                candidate_scores[users_mask] = score_slice
                rank_mask = _topk_select_mask(candidate_scores, valid_users, max_users) > 0.0
                bw_scores = np.full((cfg.users_obs_max,), -1e6, dtype=np.float32)
                bw_scores[rank_mask] = dpp_v * candidate_scores[rank_mask]
                bw = _dpp_allocate_bw(bw_scores, rank_mask, cfg)

                eta_slot = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                gap_slot = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                eta_slot[users_mask] = rate_proxy
                gap_slot[users_mask] = gap
                service = service_scale * bw * eta_slot
                access_term = float(np.sum(gap_slot * service))
            else:
                access_term = potential_score
        else:
            access_term = potential_score

        # IMPROVED: Apply updated sat_visible_mask to SAT selection
        sat_sel, backhaul_term = _dpp_sat_selection(obs, cfg, own_q_norm)
        # Mask out non-visible satellites 
        if np.any(~sat_visible_mask):
            sat_sel[~sat_visible_mask] = 0.0

        reg_term = accel_cost * float(np.dot(cand, cand)) + smooth_w * float(np.sum((cand - prev_accel) ** 2))
        score = access_w * access_term + backhaul_w * backhaul_term - reg_term
        return bw, sat_sel, pressure, service, access_term, backhaul_term, reg_term, score

    use_shared_idx = bool(getattr(cfg, "dpp_shared_accel_index", False))
    selected_idx = -1
    if use_shared_idx:
        idx = int(np.clip(getattr(cfg, "dpp_shared_accel_selected_idx", -1), -1, len(accel_candidates) - 1))
        if idx >= 0:
            selected_idx = idx

    iter_candidates = enumerate(accel_candidates) if selected_idx < 0 else [(selected_idx, accel_candidates[selected_idx])]
    for cand_idx, cand in iter_candidates:
        bw, sat_sel, pressure, service, access_term, backhaul_term, reg_term, score = _eval_candidate(cand)

        if score > best_score:
            best_score = score
            best_accel = cand.astype(np.float32, copy=False)
            best_bw = bw.astype(np.float32, copy=False)
            best_sat = sat_sel.astype(np.float32, copy=False)
            best_pressure = pressure.astype(np.float32, copy=False)
            best_service = service.astype(np.float32, copy=False)
            best_access_term = access_term
            best_backhaul_term = backhaul_term
            best_reg_term = reg_term
    best_accel = best_accel * accel_gain
    return (
        best_accel,
        best_bw,
        best_sat,
        best_pressure,
        best_service,
        float(best_access_term),
        float(best_backhaul_term),
        float(best_reg_term),
        float(best_score),
    )


def lyapunov_queue_aware_policy_step(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    state: Dict[str, np.ndarray] | None = None,
    env_callbacks: Optional[Dict[str, Callable]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Lyapunov-inspired three-head heuristic baseline with per-episode state.

    The policy approximates one-step drift-plus-penalty using a per-slot
    virtual queue over user urgency and lightweight action cost shaping.
    
    If env_callbacks is provided, enables topology-aware DPP mode with complete
    link quality recalculation after position changes. Callbacks should include:
    - compute_access_rates(agent_id, accel_vec, obs) -> (eta, rate_vector)
    - compute_backhaul_rates(agent_id, accel_vec, obs, sat_selections) -> (rates, counts)
    - check_sat_visibility(agent_id, accel_vec, obs) -> valid_sat_mask
    """

    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    bw_alloc = np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32)
    sat_select_mask = np.zeros((num_agents, cfg.sats_obs_max), dtype=np.float32)

    if state is None:
        state = _lyapunov_state_init(num_agents, cfg)
    elif (
        state.get("pressure_ema") is None
        or state.get("virtual_queue") is None
        or state.get("service_est") is None
        or state.get("prev_accel") is None
        or state["pressure_ema"].shape != (num_agents, cfg.users_obs_max)
    ):
        state = _lyapunov_state_init(num_agents, cfg)

    pressure_ema = np.asarray(state["pressure_ema"], dtype=np.float32)
    virtual_queue = np.asarray(state["virtual_queue"], dtype=np.float32)
    service_est = np.asarray(state["service_est"], dtype=np.float32)
    prev_accel = np.asarray(state["prev_accel"], dtype=np.float32)
    dpp_access_term = np.asarray(state.get("dpp_access_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_backhaul_term = np.asarray(state.get("dpp_backhaul_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_reg_term = np.asarray(state.get("dpp_reg_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_objective_term = np.asarray(state.get("dpp_objective_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)

    mode = str(getattr(cfg, "baseline_lyapunov_mode", "urgency") or "urgency").strip().lower()
    if mode == "dpp":
        lyap_ema_beta = float(np.clip(getattr(cfg, "baseline_lyapunov_ema_beta", 0.6), 0.0, 0.999))
        shared_accel = bool(getattr(cfg, "dpp_shared_accel_index", False))
        
        # Store env_callbacks temporarily in cfg for _dpp_one_agent and _predict_topology_after_accel to use
        _dpp_env_callbacks = env_callbacks or {}
        setattr(cfg, "_dpp_env_callbacks_temp", _dpp_env_callbacks)
        
        if shared_accel:
            cands = _generate_accel_candidates(cfg)
            score_sum = np.zeros((len(cands),), dtype=np.float32)
            for k in range(len(cands)):
                setattr(cfg, "dpp_shared_accel_selected_idx", k)
                for i, obs in enumerate(obs_list):
                    _, _, _, _, _, _, _, _, score_i = _dpp_one_agent(obs, cfg, prev_accel[i])
                    score_sum[k] = score_sum[k] + float(score_i)
            best_k = int(np.argmax(score_sum)) if len(cands) > 0 else -1
            setattr(cfg, "dpp_shared_accel_selected_idx", best_k)

        for i, obs in enumerate(obs_list):
            accel_i, bw_i, sat_i, pressure_i, service_i, access_i, backhaul_i, reg_i, obj_i = _dpp_one_agent(
                obs,
                cfg,
                prev_accel[i],
            )

            pressure_ema[i] = lyap_ema_beta * pressure_ema[i] + (1.0 - lyap_ema_beta) * pressure_i
            virtual_queue[i] = np.clip(virtual_queue[i] + pressure_ema[i] - service_est[i], 0.0, None)
            service_est[i] = service_i
            dpp_access_term[i] = float(access_i)
            dpp_backhaul_term[i] = float(backhaul_i)
            dpp_reg_term[i] = float(reg_i)
            dpp_objective_term[i] = float(obj_i)

            accel_vec = accel_i + _baseline_repulse_term(obs, cfg) + _baseline_energy_term(obs, cfg)
            accel[i] = np.clip(accel_vec, -1.0, 1.0)
            if cfg.enable_bw_action:
                bw_alloc[i] = bw_i
            if not cfg.fixed_satellite_strategy:
                sat_select_mask[i] = sat_i
            prev_accel[i] = accel[i]

        next_state = {
            "pressure_ema": pressure_ema.astype(np.float32, copy=False),
            "virtual_queue": virtual_queue.astype(np.float32, copy=False),
            "service_est": service_est.astype(np.float32, copy=False),
            "prev_accel": prev_accel.astype(np.float32, copy=False),
            "dpp_access_term": dpp_access_term.astype(np.float32, copy=False),
            "dpp_backhaul_term": dpp_backhaul_term.astype(np.float32, copy=False),
            "dpp_reg_term": dpp_reg_term.astype(np.float32, copy=False),
            "dpp_objective_term": dpp_objective_term.astype(np.float32, copy=False),
        }
        if hasattr(cfg, "dpp_shared_accel_selected_idx"):
            setattr(cfg, "dpp_shared_accel_selected_idx", -1)
        return accel, bw_alloc, sat_select_mask, next_state

    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0

    lyap_v = max(float(getattr(cfg, "baseline_lyapunov_v", 2.0) or 0.0), 0.0)
    lyap_urgency_alpha = max(float(getattr(cfg, "baseline_lyapunov_urgency_alpha", 1.0) or 0.0), 0.0)
    lyap_drift_w = max(float(getattr(cfg, "baseline_lyapunov_drift_weight", 1.0) or 0.0), 0.0)
    lyap_action_cost = max(float(getattr(cfg, "baseline_lyapunov_action_cost", 0.05) or 0.0), 0.0)
    lyap_ema_beta = float(np.clip(getattr(cfg, "baseline_lyapunov_ema_beta", 0.6), 0.0, 0.999))
    lyap_bw_temp = max(float(getattr(cfg, "baseline_lyapunov_bw_temp", 0.6) or 0.0), 1e-3)
    lyap_bw_floor = float(np.clip(getattr(cfg, "baseline_lyapunov_bw_floor", 0.02), 0.0, 0.2))
    lyap_service_scale = max(float(getattr(cfg, "baseline_lyapunov_bw_service_scale", 1.0) or 0.0), 0.0)
    lyap_sat_drift_w = max(float(getattr(cfg, "baseline_lyapunov_sat_drift_weight", 0.6) or 0.0), 0.0)
    lyap_sat_switch_bias = max(float(getattr(cfg, "baseline_lyapunov_sat_switch_bias", 0.1) or 0.0), 0.0)

    for i, obs in enumerate(obs_list):
        accel_vec = np.zeros((2,), dtype=np.float32)
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        bw_valid_mask = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0)

        instant_pressure = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        if np.any(users_mask):
            rel = np.asarray(users[users_mask, 0:2], dtype=np.float32)
            q = np.clip(np.asarray(users[users_mask, 2], dtype=np.float32), 0.0, None)
            eta = np.clip(np.asarray(users[users_mask, 3], dtype=np.float32), 0.0, 1.0)
            prev_assoc = np.clip(np.asarray(users[users_mask, 4], dtype=np.float32), 0.0, 1.0)

            pressure_slice = q * (0.5 + eta)
            if assoc_bonus > 0.0:
                pressure_slice = pressure_slice * (0.01 + assoc_bonus * prev_assoc)
            instant_pressure[users_mask] = np.clip(pressure_slice, 0.0, None)

            pressure_ema[i] = lyap_ema_beta * pressure_ema[i] + (1.0 - lyap_ema_beta) * instant_pressure
            virtual_queue[i] = np.clip(virtual_queue[i] + pressure_ema[i] - service_est[i], 0.0, None)
            urgency = np.clip(pressure_ema[i] + lyap_drift_w * virtual_queue[i], 0.0, None)
            nbrs = obs["nbrs"]
            nbrs_mask = obs["nbrs_mask"] > 0.0
            if np.any(nbrs_mask):
                dist_gu = np.linalg.norm(rel, axis=1)
                rel_nbr = nbrs[nbrs_mask, 0:2]
                diff = rel[:, None, :] - rel_nbr[None, :, :]
                dist_nbrs_to_user = np.linalg.norm(diff, axis=-1)
                min_nbr_dist = np.min(dist_nbrs_to_user, axis=1)
                ratio = np.exp(lyap_urgency_alpha * (min_nbr_dist - dist_gu))
                responsibility = np.clip(ratio, 0.0, 1.0) 
                assert responsibility.shape == urgency[users_mask].shape
                urgency[users_mask] = urgency[users_mask] * responsibility 
                
            urgency_sum = float(np.sum(urgency[users_mask]))
            if urgency_sum > 1e-6:
                vec = (rel * urgency[users_mask, None]).sum(axis=0) / (urgency_sum + 1e-9)
                accel_vec = accel_vec + vec * accel_gain

            if cfg.enable_bw_action:
                base_scores = lyap_v * urgency - lyap_action_cost * (instant_pressure > 0.0).astype(np.float32)
                bw_scores = np.full((cfg.users_obs_max,), -1e6, dtype=np.float32)
                valid_slots = bw_valid_mask & users_mask
                bw_scores[valid_slots] = base_scores[valid_slots]
                probs = _masked_softmax(bw_scores, valid_slots, lyap_bw_temp)
                if lyap_bw_floor > 0.0 and np.any(valid_slots):
                    count = float(np.sum(valid_slots))
                    probs = (1.0 - lyap_bw_floor * count) * probs
                    probs[valid_slots] = probs[valid_slots] + lyap_bw_floor
                    denom = float(np.sum(probs[valid_slots]))
                    if denom > 1e-9:
                        probs[valid_slots] = probs[valid_slots] / denom
                bw_alloc[i] = probs.astype(np.float32, copy=False)

                eta_slot = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                eta_slot[users_mask] = 0.5 + eta
                service_est[i] = lyap_service_scale * bw_alloc[i] * eta_slot
            else:
                service_est[i].fill(0.0)
        else:
            pressure_ema[i].fill(0.0)
            virtual_queue[i].fill(0.0)
            service_est[i].fill(0.0)

        if not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            sats_mask = obs["sats_mask"] > 0.0
            sat_valid_mask = np.asarray(obs.get("sat_valid_mask", obs["sats_mask"]) > 0.0)
            if np.any(sats_mask):
                base_sat_scores = _sat_heuristic_score(sats, sats_mask, cfg)
                sat_scores = np.asarray(base_sat_scores, dtype=np.float32)
                q_active = float(np.mean(instant_pressure[users_mask])) if np.any(users_mask) else 0.0
                qsat = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
                qsat[sats_mask] = np.clip(np.asarray(sats[sats_mask, 8], dtype=np.float32), 0.0, None)
                sat_scores = sat_scores + lyap_sat_drift_w * (q_active - qsat)
                if lyap_sat_switch_bias > 0.0:
                    stay = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
                    stay[sats_mask] = np.clip(np.asarray(sats[sats_mask, 11], dtype=np.float32), 0.0, 1.0)
                    sat_scores = sat_scores + lyap_sat_switch_bias * stay
                sat_select_mask[i] = _topk_select_mask(
                    sat_scores,
                    sat_valid_mask,
                    max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0),
                )

        if repulse_gain > 0.0 and repulse_radius > 0.0:
            accel_vec = accel_vec + _baseline_repulse_term(obs, cfg)

        accel_vec = accel_vec + _baseline_energy_term(obs, cfg)
        # accel[i] = np.clip(accel_vec, -1.0, 1.0)
        accel[i] = np.sqrt(2) * accel_vec / max(np.linalg.norm(accel_vec), 1.0)
        prev_accel[i] = accel[i]

    next_state = {
        "pressure_ema": pressure_ema.astype(np.float32, copy=False),
        "virtual_queue": virtual_queue.astype(np.float32, copy=False),
        "service_est": service_est.astype(np.float32, copy=False),
        "prev_accel": prev_accel.astype(np.float32, copy=False),
        "dpp_access_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_backhaul_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_reg_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_objective_term": np.zeros((num_agents,), dtype=np.float32),
    }
    return accel, bw_alloc, sat_select_mask, next_state


def lyapunov_queue_aware_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    accel, bw_alloc, sat_select_mask, _ = lyapunov_queue_aware_policy_step(obs_list, cfg, state=None)
    return accel, bw_alloc, sat_select_mask

def queue_aware_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Queue + channel aware heuristic baseline.

    - Accel: move toward weighted centroid of high-queue users.
    - BW: softmax weights ~ queue * (0.5 + eta), with mild assoc bonus.
    - Sat: score by link quality, backlog, expected contention, and stay bonus.
    - Safety: repel from nearby neighbors.
    - Energy: slow down when energy is low (if enabled).
    """

    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    bw_alloc = np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32)
    sat_select_mask = np.zeros((num_agents, cfg.sats_obs_max), dtype=np.float32)

    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    energy_low = float(getattr(cfg, "baseline_energy_low", 0.3))
    energy_weight = float(getattr(cfg, "baseline_energy_weight", 1.0))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0

    for i, obs in enumerate(obs_list):
        accel_vec = np.zeros((2,), dtype=np.float32)
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        bw_valid_mask = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0)
        if np.any(users_mask):
            rel = users[users_mask, 0:2]
            q = users[users_mask, 2]
            eta = users[users_mask, 3]
            prev = users[users_mask, 4]
            weights = q * (0.5 + eta)
            if assoc_bonus > 0.0:
                weights = weights * (1.0 + assoc_bonus * prev)

            weight_sum = float(np.sum(weights))
            if weight_sum > 1e-6:
                vec = (rel * weights[:, None]).sum(axis=0) / (weight_sum + 1e-9)
                accel_vec = accel_vec + vec * accel_gain

            if cfg.enable_bw_action:
                slot_weights = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                slot_weights[users_mask] = np.clip(weights, 0.0, None)
                slot_weights = slot_weights * bw_valid_mask.astype(np.float32)
                denom = float(np.sum(slot_weights))
                if denom > 1e-6:
                    bw_alloc[i] = slot_weights / denom
                elif np.any(bw_valid_mask):
                    bw_alloc[i, bw_valid_mask] = 1.0 / float(np.sum(bw_valid_mask))

        if not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            sats_mask = obs["sats_mask"] > 0.0
            sat_valid_mask = np.asarray(obs.get("sat_valid_mask", obs["sats_mask"]) > 0.0)
            if np.any(sats_mask):
                sat_scores = _sat_heuristic_score(sats, sats_mask, cfg)
                sat_select_mask[i] = _topk_select_mask(
                    sat_scores,
                    sat_valid_mask,
                    max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0),
                )

        if repulse_gain > 0.0 and repulse_radius > 0.0:
            nbrs = obs["nbrs"]
            nbrs_mask = obs["nbrs_mask"] > 0.0
            if np.any(nbrs_mask):
                rel = nbrs[nbrs_mask, 0:2]
                dist_norm = np.linalg.norm(rel, axis=1)
                dist = dist_norm * cfg.map_size
                mask = (dist > 1e-6) & (dist < repulse_radius)
                if np.any(mask):
                    rel_sel = rel[mask]
                    dist_sel = dist[mask]
                    dist_norm_sel = dist_norm[mask]
                    direction = rel_sel / dist_norm_sel[:, None]
                    strength = (1.0 / dist_sel - 1.0 / repulse_radius)
                    accel_vec = accel_vec + repulse_gain * (direction * strength[:, None]).sum(axis=0)

        if cfg.energy_enabled and energy_weight > 0.0:
            energy_norm = float(obs["own"][4])
            if energy_norm < energy_low:
                vel = obs["own"][2:4].astype(np.float32)
                speed = float(np.linalg.norm(vel))
                if speed > 1e-6:
                    target_speed = min(cfg.uav_opt_speed / max(cfg.v_max, 1e-6), 1.0)
                    delta = target_speed - speed
                    if delta < 0.0:
                        scale = (energy_low - energy_norm) / max(energy_low, 1e-6)
                        accel_vec = accel_vec + energy_weight * scale * (vel / speed) * delta

        accel[i] = np.clip(accel_vec, -1.0, 1.0)

    return accel, bw_alloc, sat_select_mask


def cluster_center_queue_aware_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    cluster_centers: np.ndarray | None,
    cluster_counts: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    accel = cluster_center_accel_policy(obs_list, cfg, cluster_centers, cluster_counts)
    _, bw_alloc, sat_select_mask = queue_aware_policy(obs_list, cfg)
    return accel, bw_alloc, sat_select_mask

def queue_aware_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    _, bw_alloc, _ = queue_aware_policy(obs_list, cfg)
    return bw_alloc

def queue_aware_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    _, _, sat_select_mask = queue_aware_policy(obs_list, cfg)
    return sat_select_mask
