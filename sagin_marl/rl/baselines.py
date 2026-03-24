from __future__ import annotations

from typing import Dict, List, Tuple

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
