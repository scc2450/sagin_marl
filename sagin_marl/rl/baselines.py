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
    }


def lyapunov_queue_aware_policy_step(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    state: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Lyapunov-inspired three-head heuristic baseline with per-episode state.

    The policy approximates one-step drift-plus-penalty using a per-slot
    virtual queue over user urgency and lightweight action cost shaping.
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
        or state["pressure_ema"].shape != (num_agents, cfg.users_obs_max)
    ):
        state = _lyapunov_state_init(num_agents, cfg)

    pressure_ema = np.asarray(state["pressure_ema"], dtype=np.float32)
    virtual_queue = np.asarray(state["virtual_queue"], dtype=np.float32)
    service_est = np.asarray(state["service_est"], dtype=np.float32)

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
            nbrs = obs["nbrs"]
            nbrs_mask = obs["nbrs_mask"] > 0.0
            if np.any(nbrs_mask):
                rel_nbr_pos = nbrs[nbrs_mask, 0:2]
                rel_nbr_vel = nbrs[nbrs_mask, 2:4]
                dist_norm = np.linalg.norm(rel_nbr_pos, axis=1)
                dist = dist_norm * cfg.map_size
                mask = (dist > 1e-6) & (dist < repulse_radius)
                if np.any(mask):
                    rel_sel = rel_nbr_pos[mask]
                    vel_sel = rel_nbr_vel[mask]
                    dist_sel = dist[mask]
                    dist_norm_sel = dist_norm[mask]

                    direction = rel_sel / dist_norm_sel[:, None]
                    approach_speed = np.sum(vel_sel * direction, axis=1)
                    spring_strength = (1.0 / dist_sel - 1.0 / repulse_radius)
                    damper_strength = np.where(approach_speed < 0, -approach_speed, 0.0)
                    strength = spring_strength + damper_strength
                    accel_vec = accel_vec - repulse_gain * (direction * strength[:, None]).sum(axis=0)

        accel_vec = accel_vec + _baseline_energy_term(obs, cfg)
        # accel[i] = np.clip(accel_vec, -1.0, 1.0)
        accel[i] = np.sqrt(2) * accel_vec / max(np.linalg.norm(accel_vec), 1.0)

    next_state = {
        "pressure_ema": pressure_ema.astype(np.float32, copy=False),
        "virtual_queue": virtual_queue.astype(np.float32, copy=False),
        "service_est": service_est.astype(np.float32, copy=False),
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
