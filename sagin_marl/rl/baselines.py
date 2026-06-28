from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _project_normalized_accel_np(accel: np.ndarray) -> np.ndarray:
    arr = np.asarray(accel, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, 1.0 / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


def _project_normalized_accel_torch(accel: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(accel, dim=-1, keepdim=True)
    scale = torch.clamp(1.0 / norms.clamp_min(1.0e-8), max=1.0)
    return accel * scale


def zero_accel_policy(num_agents: int) -> np.ndarray:
    return np.zeros((num_agents, 2), dtype=np.float32)


def random_accel_policy(
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return _project_normalized_accel_np(rng.uniform(-1.0, 1.0, size=(num_agents, 2)))


def _stack_like(value, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _obs_batch_to_torch(
    obs_batch: Dict[str, np.ndarray | torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool, torch.device]:
    sample = next(iter(obs_batch.values()))
    if torch.is_tensor(sample):
        device = sample.device
        return (
            {key: _stack_like(value, device=device) for key, value in obs_batch.items()},
            True,
            device,
        )
    device = torch.device("cpu")
    return (
        {key: _stack_like(value, device=device) for key, value in obs_batch.items()},
        False,
        device,
    )


def _from_torch_outputs(
    outputs: tuple[torch.Tensor, ...],
    *,
    as_torch: bool,
) -> tuple[torch.Tensor, ...] | tuple[np.ndarray, ...]:
    if as_torch:
        return outputs
    return tuple(
        output.detach().cpu().numpy().astype(np.float32, copy=False)
        for output in outputs
    )


def _sat_action_select_k_from_cfg(cfg) -> int:
    raw = getattr(cfg, "sat_action_select_k", None)
    if raw is not None and int(raw) > 0:
        return int(raw)
    n_rf = max(int(getattr(cfg, "N_RF", 0) or 0), 1)
    num_sat = max(int(getattr(cfg, "num_sat", 0) or 0), 0)
    sat_select_cfg = getattr(cfg, "sat_num_select", None)
    sat_select = int(sat_select_cfg) if sat_select_cfg is not None and int(sat_select_cfg) > 0 else n_rf
    upper_sat = num_sat if num_sat > 0 else sat_select
    return max(min(int(upper_sat), int(n_rf), int(sat_select)), 1)


def _candidate_indices_np(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    raw = obs.get("candidate_indices")
    if raw is None:
        if int(cfg.users_obs_max) == int(cfg.num_gu):
            return np.arange(int(cfg.num_gu), dtype=np.int64)
        raise KeyError(
            "queue-aware BW baseline received candidate-slot user obs without candidate_indices. "
            "Either provide obs['candidate_indices'] for slot->GU mapping or use full-GU obs "
            "(users_obs_max == num_gu)."
        )
    out = np.asarray(raw, dtype=np.int64).reshape(-1)
    if int(out.shape[0]) < int(cfg.users_obs_max):
        padded = np.full((int(cfg.users_obs_max),), -1, dtype=np.int64)
        padded[: int(out.shape[0])] = out
        out = padded
    return out[: int(cfg.users_obs_max)]


def _slot_bw_to_full_gu_np(slot_bw: np.ndarray, obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    full = np.zeros((int(cfg.num_gu),), dtype=np.float32)
    if int(cfg.num_gu) <= 0:
        return full
    values = np.asarray(slot_bw, dtype=np.float32).reshape(-1)
    candidate_idx = _candidate_indices_np(obs, cfg)
    width = min(int(values.shape[0]), int(candidate_idx.shape[0]))
    for slot in range(width):
        gu_idx = int(candidate_idx[slot])
        if 0 <= gu_idx < int(cfg.num_gu):
            full[gu_idx] += float(values[slot])
    return full


def _slot_bw_to_full_gu_torch(
    slot_bw: torch.Tensor,
    obs_batch: Dict[str, torch.Tensor],
    cfg,
) -> torch.Tensor:
    batch_size, num_agents, slot_count = slot_bw.shape
    num_gu = int(cfg.num_gu)
    full = torch.zeros((batch_size, num_agents, num_gu), dtype=slot_bw.dtype, device=slot_bw.device)
    if num_gu <= 0 or slot_count <= 0:
        return full
    candidate_raw = obs_batch.get("candidate_indices")
    if candidate_raw is None:
        if slot_count != num_gu:
            raise KeyError(
                "queue-aware BW baseline received candidate-slot batched obs without candidate_indices. "
                "Either provide candidate_indices or use full-GU obs."
            )
        candidate = torch.arange(slot_count, dtype=torch.long, device=slot_bw.device).view(1, 1, slot_count)
        candidate = candidate.expand(batch_size, num_agents, slot_count)
    else:
        candidate = candidate_raw.to(device=slot_bw.device, dtype=torch.long)
        if candidate.ndim == 2:
            candidate = candidate.unsqueeze(0)
        if int(candidate.shape[-1]) < slot_count:
            padded = torch.full(
                (*candidate.shape[:-1], slot_count),
                -1,
                dtype=torch.long,
                device=slot_bw.device,
            )
            padded[..., : int(candidate.shape[-1])] = candidate
            candidate = padded
        candidate = candidate[..., :slot_count].expand(batch_size, num_agents, slot_count)
    valid = (candidate >= 0) & (candidate < num_gu)
    safe_candidate = candidate.clamp(min=0, max=max(num_gu - 1, 0))
    full.scatter_add_(2, safe_candidate, torch.where(valid, slot_bw, torch.zeros_like(slot_bw)))
    return full


def _baseline_energy_term_batch_torch(own: torch.Tensor, cfg) -> torch.Tensor:
    energy_weight = float(getattr(cfg, "baseline_energy_weight", 1.0))
    if not cfg.energy_enabled or energy_weight <= 0.0:
        return torch.zeros((*own.shape[:-1], 2), dtype=torch.float32, device=own.device)

    energy_low = float(getattr(cfg, "baseline_energy_low", 0.3))
    energy_norm = own[..., 4]
    vel = own[..., 2:4]
    speed = torch.linalg.norm(vel, dim=-1)
    target_speed = min(cfg.uav_opt_speed / max(cfg.v_max, 1e-6), 1.0)
    delta = target_speed - speed
    scale = (energy_low - energy_norm) / max(energy_low, 1e-6)
    safe_speed = speed.clamp_min(1.0e-6).unsqueeze(-1)
    term = energy_weight * scale.unsqueeze(-1) * (vel / safe_speed) * delta.unsqueeze(-1)
    active = (energy_norm < energy_low) & (speed > 1.0e-6) & (delta < 0.0)
    return torch.where(
        active.unsqueeze(-1),
        term,
        torch.zeros_like(term),
    )


def _select_cluster_targets_from_positions(
    uav_pos: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_counts: np.ndarray,
) -> np.ndarray:
    num_agents = int(np.asarray(uav_pos).shape[0])
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

    remaining_uavs = list(range(num_agents))
    for cluster_idx in selected:
        rem = np.asarray(remaining_uavs, dtype=np.int32)
        dists = np.linalg.norm(np.asarray(uav_pos[rem], dtype=np.float32) - centers[cluster_idx], axis=1)
        best_uav = int(rem[int(np.argmin(dists))])
        targets[best_uav] = int(cluster_idx)
        remaining_uavs.remove(best_uav)

    if selected.size > 0:
        selected_centers = centers[selected]
        for u in remaining_uavs:
            dists = np.linalg.norm(selected_centers - np.asarray(uav_pos[u], dtype=np.float32), axis=1)
            targets[u] = int(selected[int(np.argmin(dists))])

    return targets


def _torch_minmax_normalize_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    inf = torch.full_like(values, float("inf"))
    neg_inf = torch.full_like(values, float("-inf"))
    masked_min = torch.where(mask, values, inf).amin(dim=-1, keepdim=True)
    masked_max = torch.where(mask, values, neg_inf).amax(dim=-1, keepdim=True)
    span = masked_max - masked_min
    valid_range = span > 1.0e-9
    norm = (values - masked_min) / span.clamp_min(1.0e-9)
    return torch.where(mask & valid_range, norm, torch.zeros_like(values))


def _topk_select_mask_batch_torch(scores: torch.Tensor, valid_mask: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(scores, dtype=torch.float32)
    if k <= 0 or scores.shape[-1] <= 0:
        return out
    masked_scores = torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))
    topk = min(int(k), int(scores.shape[-1]))
    topk_idx = torch.argsort(masked_scores, dim=-1, descending=True)[..., :topk]
    topk_valid = valid_mask.gather(-1, topk_idx).to(dtype=torch.float32)
    out.scatter_(-1, topk_idx, topk_valid)
    return out


def _sat_heuristic_score_batch_torch(sats: torch.Tensor, mask: torch.Tensor, cfg) -> torch.Tensor:
    if sats.numel() == 0:
        return torch.zeros((*sats.shape[:-1],), dtype=torch.float32, device=sats.device)

    se = sats[..., 7]
    qsat = sats[..., 8]
    load_norm = sats[..., 9]
    bw_ratio = sats[..., 10]
    stay = sats[..., 11]

    se_weight = float(getattr(cfg, "baseline_sat_se_weight", 1.0) or 0.0)
    q_penalty = float(getattr(cfg, "baseline_sat_queue_penalty", 0.5) or 0.0)
    load_penalty = float(getattr(cfg, "baseline_sat_load_penalty", 1.0) or 0.0)
    bw_reward = float(getattr(cfg, "baseline_sat_bw_reward", 0.75) or 0.0)
    stay_bonus = float(getattr(cfg, "baseline_sat_stay_bonus", 0.25) or 0.0)
    switch_margin = max(float(getattr(cfg, "baseline_sat_switch_margin", 0.15) or 0.0), 0.0)

    projected_count = 1.0 / bw_ratio.clamp(1.0e-6, 1.0)
    projected_load_term = torch.log1p(projected_count)

    se_norm = _torch_minmax_normalize_masked(se, mask)
    q_norm = _torch_minmax_normalize_masked(qsat, mask)
    load_term_norm = _torch_minmax_normalize_masked(projected_load_term, mask)
    bw_norm = _torch_minmax_normalize_masked(bw_ratio, mask)

    score = (
        se_weight * se_norm
        - q_penalty * q_norm
        - load_penalty * load_term_norm
        + bw_reward * bw_norm
        + stay_bonus * stay
    )

    if int(getattr(cfg, "N_RF", 1)) == 1:
        current_mask = stay > 0.5
        current_count = current_mask.sum(dim=-1)
        current_idx = torch.argmax(stay, dim=-1)
        best_idx = torch.argmax(torch.where(mask, score, torch.full_like(score, float("-inf"))), dim=-1)
        cur_score = torch.take_along_dim(score, current_idx.unsqueeze(-1), dim=-1).squeeze(-1)
        best_score = torch.take_along_dim(score, best_idx.unsqueeze(-1), dim=-1).squeeze(-1)
        keep_current = (
            (current_count == 1)
            & (best_idx != current_idx)
            & (best_score <= cur_score + switch_margin)
        )
        adjusted = best_score + 1.0e-3
        score = score.scatter(
            -1,
            current_idx.unsqueeze(-1),
            torch.where(keep_current, adjusted, cur_score).unsqueeze(-1),
        )

    clipped = score.clamp(-float(cfg.sat_logit_scale), float(cfg.sat_logit_scale))
    return torch.where(mask, clipped, torch.zeros_like(clipped))


def queue_aware_policy_batch(
    obs_batch: Dict[str, np.ndarray | torch.Tensor],
    cfg,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    torch_batch, as_torch, _ = _obs_batch_to_torch(obs_batch)
    own = torch_batch["own"]
    users = torch_batch["users"]
    users_mask = torch_batch["users_mask"] > 0.0
    bw_valid_mask = (torch_batch.get("bw_valid_mask", torch_batch["users_mask"]) > 0.0)
    dtype = torch.float32
    device = own.device

    batch_size, num_agents = own.shape[:2]
    accel = torch.zeros((batch_size, num_agents, 2), dtype=dtype, device=device)
    bw_alloc = torch.zeros((batch_size, num_agents, int(cfg.num_gu)), dtype=dtype, device=device)
    sat_select_mask = torch.zeros(
        (batch_size, num_agents, int(torch_batch["sats"].shape[2])),
        dtype=dtype,
        device=device,
    )

    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0

    if users.shape[2] > 0:
        q = users[..., 2].clamp_min(0.0)
        eta = users[..., 3]
        prev = users[..., 4]
        weights = q * (0.5 + eta)
        if assoc_bonus > 0.0:
            weights = weights * (1.0 + assoc_bonus * prev)
        weights = torch.where(users_mask, weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=-1, keepdim=True)
        vec = (users[..., 0:2] * weights.unsqueeze(-1)).sum(dim=-2) / (weight_sum + 1.0e-9)
        accel = torch.where(
            users_mask.any(dim=-1, keepdim=True),
            vec * accel_gain,
            accel,
        )

        if cfg.enable_bw_action:
            slot_weights = weights.clamp_min(0.0) * bw_valid_mask.to(dtype=dtype)
            denom = slot_weights.sum(dim=-1, keepdim=True)
            slot_bw_alloc = torch.zeros_like(slot_weights)
            slot_bw_alloc = torch.where(
                denom > 1.0e-6,
                slot_weights / denom.clamp_min(1.0e-9),
                slot_bw_alloc,
            )
            uniform = bw_valid_mask.to(dtype=dtype) / bw_valid_mask.to(dtype=dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
            fallback = (denom <= 1.0e-6) & bw_valid_mask.any(dim=-1, keepdim=True)
            slot_bw_alloc = torch.where(fallback, uniform, slot_bw_alloc)
            width = min(int(cfg.num_gu), int(slot_bw_alloc.shape[-1]))
            if width > 0:
                bw_alloc[..., :width] = slot_bw_alloc[..., :width]

    if not cfg.fixed_satellite_strategy:
        sats = torch_batch["sats"]
        sat_valid_mask = torch_batch.get("sat_valid_mask", torch_batch["sats_mask"]) > 0.0
        if sats.shape[2] > 0:
            sat_scores = _sat_heuristic_score_batch_torch(sats, sat_valid_mask, cfg)
            sat_select_mask = _topk_select_mask_batch_torch(
                sat_scores,
                sat_valid_mask,
                _sat_action_select_k_from_cfg(cfg),
            )

    if repulse_gain > 0.0 and repulse_radius > 0.0 and int(torch_batch["nbrs"].shape[2]) > 0:
        nbrs = torch_batch["nbrs"]
        nbrs_mask = torch_batch["nbrs_mask"] > 0.0
        rel_nbr = nbrs[..., 0:2]
        dist_norm = torch.linalg.norm(rel_nbr, dim=-1)
        dist = dist_norm * float(cfg.map_size)
        active = nbrs_mask & (dist > 1.0e-6) & (dist < repulse_radius)
        direction = rel_nbr / dist_norm.clamp_min(1.0e-9).unsqueeze(-1)
        strength = (1.0 / dist.clamp_min(1.0e-9) - 1.0 / float(repulse_radius))
        repulse = (direction * strength.unsqueeze(-1) * active.unsqueeze(-1).to(dtype)).sum(dim=-2)
        accel = accel - repulse_gain * repulse

    accel = _project_normalized_accel_torch(accel + _baseline_energy_term_batch_torch(own, cfg))
    return _from_torch_outputs((accel, bw_alloc, sat_select_mask), as_torch=as_torch)


def cluster_center_accel_policy_batch(
    obs_batch: Dict[str, np.ndarray | torch.Tensor],
    cfg,
    cluster_centers: np.ndarray | torch.Tensor | None,
    cluster_counts: np.ndarray | torch.Tensor | None,
) -> np.ndarray | torch.Tensor:
    torch_batch, as_torch, device = _obs_batch_to_torch(obs_batch)
    own = torch_batch["own"]
    batch_size, num_agents = own.shape[:2]
    accel = torch.zeros((batch_size, num_agents, 2), dtype=torch.float32, device=device)
    if cluster_centers is None or cluster_counts is None:
        return _from_torch_outputs((accel,), as_torch=as_torch)[0]

    centers_t = _stack_like(cluster_centers, device=device)
    counts_t = _stack_like(cluster_counts, device=device)
    uav_pos = (own[..., 0:2] * float(cfg.map_size)).detach().cpu().numpy().astype(np.float32, copy=False)
    centers_np = centers_t.detach().cpu().numpy().astype(np.float32, copy=False)
    counts_np = counts_t.detach().cpu().numpy().astype(np.float32, copy=False)

    targets = np.full((batch_size, num_agents), -1, dtype=np.int64)
    for env_index in range(batch_size):
        targets[env_index] = _select_cluster_targets_from_positions(
            uav_pos[env_index],
            centers_np[env_index],
            counts_np[env_index],
        )
    target_idx = torch.as_tensor(targets, dtype=torch.long, device=device)
    valid_target = target_idx >= 0
    if bool(valid_target.any().item()):
        target_pos = torch.zeros((batch_size, num_agents, 2), dtype=torch.float32, device=device)
        env_ids, uav_ids = torch.nonzero(valid_target, as_tuple=True)
        target_pos[env_ids, uav_ids] = centers_t[env_ids, target_idx[env_ids, uav_ids]]

        pos = own[..., 0:2] * float(cfg.map_size)
        vel = own[..., 2:4] * float(cfg.v_max)
        error = target_pos - pos
        dist = torch.linalg.norm(error, dim=-1)
        speed = torch.linalg.norm(vel, dim=-1)

        stop_radius = max(float(getattr(cfg, "baseline_cluster_stop_radius", 20.0) or 0.0), 0.0)
        speed_tol = max(float(getattr(cfg, "baseline_cluster_speed_tol", 2.0) or 0.0), 0.0)
        slow_radius_cfg = float(getattr(cfg, "baseline_cluster_slow_radius", 120.0) or 0.0)
        slow_radius = max(slow_radius_cfg, stop_radius + 1.0e-6)
        cruise_speed_cfg = getattr(cfg, "baseline_cluster_cruise_speed", None)
        cruise_speed = cfg.uav_opt_speed if cruise_speed_cfg is None else float(cruise_speed_cfg)
        cruise_speed = float(np.clip(cruise_speed, 0.0, cfg.v_max))
        vel_gain = max(float(getattr(cfg, "baseline_cluster_vel_gain", 1.0) or 0.0), 0.0)

        desired_vel = torch.zeros_like(vel)
        outside_stop = valid_target & (dist > stop_radius)
        direction = error / dist.clamp_min(1.0e-6).unsqueeze(-1)
        desired_speed = cruise_speed * torch.clamp(dist / max(float(slow_radius), 1.0e-6), max=1.0)
        desired_vel = torch.where(
            outside_stop.unsqueeze(-1),
            direction * desired_speed.unsqueeze(-1),
            desired_vel,
        )
        desired_accel = vel_gain * (desired_vel - vel) / max(float(cfg.tau0), 1.0e-6)
        accel = desired_accel / max(float(cfg.a_max), 1.0e-6)
        stop_and_slow = valid_target & (dist <= stop_radius) & (speed <= speed_tol)
        accel = torch.where(stop_and_slow.unsqueeze(-1), torch.zeros_like(accel), accel)
        accel = torch.where(valid_target.unsqueeze(-1), accel, torch.zeros_like(accel))

    accel = _project_normalized_accel_torch(accel + _baseline_energy_term_batch_torch(own, cfg))
    return _from_torch_outputs((accel,), as_torch=as_torch)[0]


def cluster_center_queue_aware_policy_batch(
    obs_batch: Dict[str, np.ndarray | torch.Tensor],
    cfg,
    cluster_centers: np.ndarray | torch.Tensor | None,
    cluster_counts: np.ndarray | torch.Tensor | None,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    accel = cluster_center_accel_policy_batch(obs_batch, cfg, cluster_centers, cluster_counts)
    _, bw_alloc, sat_select_mask = queue_aware_policy_batch(obs_batch, cfg)
    return accel, bw_alloc, sat_select_mask


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
        accel[i] = _project_normalized_accel_np(vec * gain)
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
    uav_pos = np.asarray([obs["own"][0:2] * cfg.map_size for obs in obs_list], dtype=np.float32)
    return _select_cluster_targets_from_positions(uav_pos, cluster_centers, cluster_counts)


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
    return _project_normalized_accel_np(action)


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
        accel[i] = _project_normalized_accel_np(accel_vec)
    return accel

def uniform_bw_policy(num_agents: int, gu_count: int) -> np.ndarray:
    if gu_count <= 0:
        return np.zeros((num_agents, 0), dtype=np.float32)
    return np.full((num_agents, gu_count), 1.0 / float(gu_count), dtype=np.float32)

def random_bw_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.random(size=(num_agents, cfg.num_gu)).astype(np.float32)

def uniform_sat_policy(num_agents: int, sats_obs_max: int) -> np.ndarray:
    return np.zeros((num_agents, sats_obs_max), dtype=np.float32)

def random_sat_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.random(size=(num_agents, cfg.sats_obs_max)) > 0.5).astype(np.float32)


def _bw_slot_valid_mask(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    users_mask = np.asarray(obs.get("users_mask", np.zeros((int(cfg.users_obs_max),), dtype=np.float32))) > 0.0
    bw_valid = np.asarray(obs.get("bw_valid_mask", users_mask), dtype=np.float32).reshape(-1) > 0.0
    if bw_valid.shape[0] < int(cfg.users_obs_max):
        padded = np.zeros((int(cfg.users_obs_max),), dtype=bool)
        padded[: int(bw_valid.shape[0])] = bw_valid
        bw_valid = padded
    return (users_mask[: int(cfg.users_obs_max)] & bw_valid[: int(cfg.users_obs_max)]).astype(bool, copy=False)


def _slot_weight_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    weight_fn,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    bw_alloc = np.zeros((len(obs_list), int(cfg.num_gu)), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        valid = _bw_slot_valid_mask(obs, cfg)
        if not np.any(valid):
            continue
        raw_weights = np.asarray(weight_fn(obs, valid, rng), dtype=np.float32).reshape(-1)
        slot_weights = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        width = min(int(slot_weights.shape[0]), int(raw_weights.shape[0]))
        if width > 0:
            slot_weights[:width] = raw_weights[:width]
        slot_weights = np.where(valid, np.clip(slot_weights, 0.0, None), 0.0).astype(np.float32, copy=False)
        denom = float(np.sum(slot_weights, dtype=np.float32))
        if denom <= 1.0e-9:
            slot_weights[valid] = 1.0 / float(np.sum(valid))
        else:
            slot_weights = slot_weights / denom
        bw_alloc[i] = _slot_bw_to_full_gu_np(slot_weights, obs, cfg)
    return bw_alloc


def feasible_uniform_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    def _weights(_obs, valid, _rng):
        weights = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        weights[valid] = 1.0
        return weights

    return _slot_weight_bw_policy(obs_list, cfg, _weights)


def feasible_random_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()

    def _weights(_obs, valid, local_rng):
        weights = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        weights[valid] = local_rng.random(int(np.sum(valid))).astype(np.float32)
        return weights

    return _slot_weight_bw_policy(obs_list, cfg, _weights, rng=rng)


def link_priority_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    def _weights(obs, _valid, _rng):
        users = np.asarray(obs["users"], dtype=np.float32)
        weights = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        width = min(int(cfg.users_obs_max), int(users.shape[0]))
        if width > 0 and int(users.shape[1]) > 3:
            weights[:width] = np.clip(users[:width, 3], 0.0, None)
        return weights

    return _slot_weight_bw_policy(obs_list, cfg, _weights)


def demand_priority_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    def _weights(obs, _valid, _rng):
        users = np.asarray(obs["users"], dtype=np.float32)
        weights = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        width = min(int(cfg.users_obs_max), int(users.shape[0]))
        if width > 0 and int(users.shape[1]) > 2:
            weights[:width] = np.clip(users[:width, 2], 0.0, None)
        return weights

    return _slot_weight_bw_policy(obs_list, cfg, _weights)


def _sat_valid_mask(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    mask = np.asarray(obs.get("sat_valid_mask", obs.get("sats_mask", np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)))) > 0.0
    if mask.shape[0] < int(cfg.sats_obs_max):
        padded = np.zeros((int(cfg.sats_obs_max),), dtype=bool)
        padded[: int(mask.shape[0])] = mask
        mask = padded
    return mask[: int(cfg.sats_obs_max)].astype(bool, copy=False)


def feasible_uniform_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    sat_select_mask = np.zeros((len(obs_list), int(cfg.sats_obs_max)), dtype=np.float32)
    select_k = _sat_action_select_k_from_cfg(cfg)
    for i, obs in enumerate(obs_list):
        valid_idx = np.flatnonzero(_sat_valid_mask(obs, cfg))
        if valid_idx.size <= 0:
            continue
        keep_count = min(int(select_k), int(valid_idx.size))
        keep = rng.choice(valid_idx, size=keep_count, replace=False)
        sat_select_mask[i, keep] = 1.0
    return sat_select_mask


def feasible_random_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return feasible_uniform_sat_policy(obs_list, cfg, rng=rng)


def backhaul_rate_priority_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    sat_select_mask = np.zeros((len(obs_list), int(cfg.sats_obs_max)), dtype=np.float32)
    select_k = _sat_action_select_k_from_cfg(cfg)
    for i, obs in enumerate(obs_list):
        valid = _sat_valid_mask(obs, cfg)
        sats = np.asarray(obs["sats"], dtype=np.float32)
        scores = np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)
        width = min(int(scores.shape[0]), int(sats.shape[0]))
        if width > 0 and int(sats.shape[1]) > 7:
            spectral_efficiency = np.clip(sats[:width, 7], 0.0, None)
            bandwidth_ratio = (
                np.clip(sats[:width, 10], 0.0, None)
                if int(sats.shape[1]) > 10
                else np.ones((width,), dtype=np.float32)
            )
            scores[:width] = spectral_efficiency * bandwidth_ratio
        sat_select_mask[i] = _topk_select_mask(scores, valid, select_k)
    return sat_select_mask


def low_queue_priority_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    sat_select_mask = np.zeros((len(obs_list), int(cfg.sats_obs_max)), dtype=np.float32)
    select_k = _sat_action_select_k_from_cfg(cfg)
    for i, obs in enumerate(obs_list):
        valid = _sat_valid_mask(obs, cfg)
        sats = np.asarray(obs["sats"], dtype=np.float32)
        scores = np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)
        width = min(int(scores.shape[0]), int(sats.shape[0]))
        if width > 0 and int(sats.shape[1]) > 8:
            queue = np.clip(sats[:width, 8], 0.0, None)
            if int(sats.shape[1]) > 10:
                tie_link = np.clip(sats[:width, 7], 0.0, None) * np.clip(sats[:width, 10], 0.0, None)
            elif int(sats.shape[1]) > 7:
                tie_link = np.clip(sats[:width, 7], 0.0, None)
            else:
                tie_link = 0.0
            scores[:width] = -queue + 1.0e-4 * tie_link
        sat_select_mask[i] = _topk_select_mask(scores, valid, select_k)
    return sat_select_mask


def static_uniform_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        zero_accel_policy(len(obs_list)),
        feasible_uniform_bw_policy(obs_list, cfg),
        feasible_uniform_sat_policy(obs_list, cfg, rng=rng),
    )


def random_feasible_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    return (
        random_accel_policy(len(obs_list), rng=rng),
        feasible_random_bw_policy(obs_list, cfg, rng=rng),
        feasible_random_sat_policy(obs_list, cfg, rng=rng),
    )


def link_priority_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        zero_accel_policy(len(obs_list)),
        link_priority_bw_policy(obs_list, cfg),
        backhaul_rate_priority_sat_policy(obs_list, cfg),
    )


def demand_priority_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        zero_accel_policy(len(obs_list)),
        demand_priority_bw_policy(obs_list, cfg),
        low_queue_priority_sat_policy(obs_list, cfg),
    )


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


def _sat_channel_profile(sats: np.ndarray, mask: np.ndarray, cfg) -> Dict[str, np.ndarray]:
    se_abs = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    queue = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    load = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    doppler_margin = np.ones((cfg.sats_obs_max,), dtype=np.float32)
    relay_support = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    if not np.any(mask):
        return {
            "se_abs": se_abs,
            "queue": queue,
            "load": load,
            "doppler_margin": doppler_margin,
            "relay_support": relay_support,
        }

    sat_feat = np.asarray(sats[mask], dtype=np.float32)
    se = np.clip(np.asarray(sat_feat[:, 7], dtype=np.float32), 0.0, None)
    queue_slice = np.clip(np.asarray(sat_feat[:, 8], dtype=np.float32), 0.0, None)
    load_slice = np.clip(np.asarray(sat_feat[:, 9], dtype=np.float32), 0.0, None)
    nu_abs = np.clip(np.abs(np.asarray(sat_feat[:, 6], dtype=np.float32)), 0.0, None)

    # sat_obs[:, 7] already includes current backhaul attenuation, so this term
    # reacts to atmospheric/rain loss without needing extra environment access.
    se_abs_slice = np.tanh(se / 6.0).astype(np.float32, copy=False)
    doppler_margin_slice = (1.0 - np.clip(nu_abs, 0.0, 1.0)).astype(np.float32, copy=False)
    congestion = (1.0 / (1.0 + queue_slice + load_slice)).astype(np.float32, copy=False)
    relay_slice = np.clip(se_abs_slice * (0.5 + 0.5 * doppler_margin_slice) * congestion, 0.0, 1.0)

    se_abs[mask] = se_abs_slice
    queue[mask] = queue_slice
    load[mask] = load_slice
    doppler_margin[mask] = doppler_margin_slice
    relay_support[mask] = relay_slice
    return {
        "se_abs": se_abs,
        "queue": queue,
        "load": load,
        "doppler_margin": doppler_margin,
        "relay_support": relay_support,
    }


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


def _topology_dpp_state_init(num_agents: int, cfg) -> Dict[str, np.ndarray]:
    num_gu = int(getattr(cfg, "num_gu", 0))
    return {
        "pressure_ema": np.zeros((num_agents, num_gu), dtype=np.float32),
        "virtual_queue": np.zeros((num_agents, num_gu), dtype=np.float32),
        "service_est": np.zeros((num_agents, num_gu), dtype=np.float32),
        "prev_accel": np.zeros((num_agents, 2), dtype=np.float32),
        "dpp_access_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_backhaul_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_reg_term": np.zeros((num_agents,), dtype=np.float32),
        "dpp_objective_term": np.zeros((num_agents,), dtype=np.float32),
    }


def _topology_dpp_accel_candidates(cfg, *, allow_motion: bool = True) -> np.ndarray:
    if not allow_motion:
        return np.zeros((1, 2), dtype=np.float32)
    num = max(int(getattr(cfg, "topology_dpp_accel_num_candidates", 9) or 9), 1)
    step = float(np.clip(getattr(cfg, "topology_dpp_accel_step_scale", 0.6), 0.0, 1.0))
    if num <= 1 or step <= 1.0e-9:
        return np.zeros((1, 2), dtype=np.float32)
    candidates = [np.zeros((2,), dtype=np.float32)]
    for k in range(num - 1):
        theta = 2.0 * np.pi * float(k) / float(num - 1)
        candidates.append(np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32) * step)
    return np.asarray(candidates, dtype=np.float32)


def _baseline_repulse_term_np(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
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
    dist = dist_norm * float(cfg.map_size)
    mask = (dist > 1.0e-6) & (dist < repulse_radius)
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


def _predict_users_rel_after_accel(obs: Dict[str, np.ndarray], cfg, accel_vec: np.ndarray) -> np.ndarray:
    users = np.asarray(obs["users"], dtype=np.float32)
    rel = users[:, 0:2].copy()
    own = np.asarray(obs["own"], dtype=np.float32)
    vel_abs = own[2:4] * float(cfg.v_max)
    accel_abs = np.asarray(accel_vec, dtype=np.float32) * float(cfg.a_max)
    delta_pos_abs = vel_abs * float(cfg.tau0) + 0.5 * accel_abs * (float(cfg.tau0) ** 2)
    rel = rel - delta_pos_abs[None, :] / max(float(cfg.map_size), 1.0e-6)
    return rel.astype(np.float32, copy=False)


def _approx_eta_from_distance(rel_next: np.ndarray, cfg) -> np.ndarray:
    dist = np.linalg.norm(np.asarray(rel_next, dtype=np.float32), axis=1)
    range_norm = np.clip(dist / 0.5, 0.0, 1.0)
    eta = np.clip(0.8 * (1.0 - 0.9 * range_norm) + 0.1, 0.1, 1.0)
    return eta.astype(np.float32, copy=False)


def _predict_topology_after_accel(
    obs: Dict[str, np.ndarray],
    cfg,
    accel_vec: np.ndarray,
    *,
    agent_id: int = 0,
    env_callbacks: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    rel_next = _predict_users_rel_after_accel(obs, cfg, accel_vec)
    callbacks = env_callbacks or {}
    eta = None
    rate_approx = None
    compute_access = callbacks.get("compute_access_rates")
    if callable(compute_access):
        try:
            eta_new, rate_new = compute_access(agent_id, accel_vec, obs, rel_next)
            eta = np.asarray(eta_new, dtype=np.float32).reshape(-1)[: int(cfg.users_obs_max)]
            rate_approx = np.asarray(rate_new, dtype=np.float32).reshape(-1)[: int(cfg.users_obs_max)]
        except Exception:
            eta = None
            rate_approx = None
    if eta is None:
        eta = _approx_eta_from_distance(rel_next, cfg)
    if rate_approx is None or int(rate_approx.shape[0]) < int(cfg.users_obs_max):
        rate_approx = 0.5 * eta

    sat_visible = None
    check_sat = callbacks.get("check_sat_visibility")
    if callable(check_sat):
        try:
            sat_visible = check_sat(agent_id, accel_vec, obs)
        except Exception:
            sat_visible = None
    if sat_visible is None:
        sat_visible = obs.get("sat_valid_mask", obs.get("sats_mask", np.ones((int(cfg.sats_obs_max),), dtype=np.float32)))
    sat_visible_mask = np.asarray(sat_visible, dtype=np.float32).reshape(-1)[: int(cfg.sats_obs_max)] > 0.0
    if int(sat_visible_mask.shape[0]) < int(cfg.sats_obs_max):
        padded = np.zeros((int(cfg.sats_obs_max),), dtype=bool)
        padded[: int(sat_visible_mask.shape[0])] = sat_visible_mask
        sat_visible_mask = padded

    return {
        "rel_next": rel_next.astype(np.float32, copy=False),
        "eta": np.clip(eta, 0.0, None).astype(np.float32, copy=False),
        "rate_approx": np.clip(rate_approx, 0.0, None).astype(np.float32, copy=False),
        "sat_visible_mask": sat_visible_mask.astype(bool, copy=False),
    }


def _topology_dpp_allocate_bw(scores: np.ndarray, valid_mask: np.ndarray, cfg) -> np.ndarray:
    probs = _masked_softmax(
        scores,
        valid_mask,
        float(getattr(cfg, "topology_dpp_bw_temp", 0.55) or 0.55),
    )
    count = int(np.sum(valid_mask))
    if count <= 0:
        return probs.astype(np.float32, copy=False)
    floor = float(np.clip(getattr(cfg, "topology_dpp_bw_floor", 0.0), 0.0, 0.2))
    floor = min(floor, 0.99 / float(count))
    if floor > 0.0:
        probs = (1.0 - floor * float(count)) * probs
        probs[valid_mask] = probs[valid_mask] + floor
        denom = float(np.sum(probs[valid_mask]))
        if denom > 1.0e-9:
            probs[valid_mask] = probs[valid_mask] / denom
    return probs.astype(np.float32, copy=False)


def _topology_dpp_sat_selection(
    obs: Dict[str, np.ndarray],
    cfg,
    own_q_norm: float,
    *,
    sat_visible_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    sat_sel = np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)
    if cfg.fixed_satellite_strategy:
        return sat_sel, 0.0
    sats = obs["sats"]
    base_mask = np.asarray(obs.get("sat_valid_mask", obs.get("sats_mask", np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)))) > 0.0
    if sat_visible_mask is not None:
        visible = np.asarray(sat_visible_mask, dtype=bool).reshape(-1)
        if int(visible.shape[0]) < int(cfg.sats_obs_max):
            padded = np.zeros((int(cfg.sats_obs_max),), dtype=bool)
            padded[: int(visible.shape[0])] = visible
            visible = padded
        base_mask = base_mask & visible[: int(cfg.sats_obs_max)]
    if not np.any(base_mask):
        return sat_sel, 0.0

    sat_scores = _sat_heuristic_score(sats, base_mask, cfg)
    qsat = np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)
    se = np.zeros_like(qsat)
    bw_ratio = np.zeros_like(qsat)
    width = min(int(cfg.sats_obs_max), int(sats.shape[0]))
    if width > 0:
        qsat[:width] = np.clip(np.asarray(sats[:width, 8], dtype=np.float32), 0.0, None)
        se[:width] = np.clip(np.asarray(sats[:width, 7], dtype=np.float32), 0.0, None)
        if int(sats.shape[1]) > 10:
            bw_ratio[:width] = np.clip(np.asarray(sats[:width, 10], dtype=np.float32), 0.0, 1.0)
        else:
            bw_ratio[:width] = 1.0
    backhaul_proxy = 0.5 * _minmax_normalize(se) + 0.5 * bw_ratio
    gap = np.clip(float(own_q_norm) - qsat, 0.0, None)
    gap_w = float(max(getattr(cfg, "topology_dpp_sat_queue_gap_weight", 1.0) or 0.0, 0.0))
    sat_scores = sat_scores + gap_w * gap * backhaul_proxy

    topm = int(getattr(cfg, "topology_dpp_sat_candidate_topm", int(cfg.sats_obs_max)) or int(cfg.sats_obs_max))
    topm = max(min(topm, int(cfg.sats_obs_max)), 1)
    valid_idx = np.flatnonzero(base_mask)
    if valid_idx.size > topm:
        order = valid_idx[np.argsort(sat_scores[valid_idx])[::-1]]
        keep = order[:topm]
        topm_mask = np.zeros_like(base_mask, dtype=bool)
        topm_mask[keep] = True
        base_mask = base_mask & topm_mask
        valid_idx = np.flatnonzero(base_mask)
    if valid_idx.size == 0:
        return sat_sel, 0.0

    max_select = _sat_action_select_k_from_cfg(cfg)
    enum_budget = max(int(getattr(cfg, "topology_dpp_sat_enum_max_subsets", 64) or 64), 1)
    subset_penalty = float(max(getattr(cfg, "topology_dpp_sat_subset_penalty", 0.02) or 0.0, 0.0))
    contention_w = float(max(getattr(cfg, "topology_dpp_sat_contention_weight", 0.15) or 0.0, 0.0))

    candidate_subsets: List[Tuple[int, ...]] = []
    for k in range(1, min(max_select, int(valid_idx.size)) + 1):
        for comb in combinations(valid_idx.tolist(), k):
            candidate_subsets.append(comb)
            if len(candidate_subsets) >= enum_budget:
                break
        if len(candidate_subsets) >= enum_budget:
            break
    if not candidate_subsets:
        sat_sel = _topk_select_mask(sat_scores, base_mask, max_select)
        return sat_sel.astype(np.float32, copy=False), float(np.sum(gap * backhaul_proxy * sat_sel))

    best_subset: Tuple[int, ...] = tuple()
    best_score = -1.0e30
    best_backhaul = 0.0
    for subset in candidate_subsets:
        idx = np.asarray(subset, dtype=np.int32)
        backhaul_term = float(np.sum(gap[idx] * backhaul_proxy[idx]))
        contention_penalty = float(np.sum(1.0 - bw_ratio[idx]))
        score = float(np.sum(sat_scores[idx])) + backhaul_term
        score -= subset_penalty * float(len(subset) ** 2)
        score -= contention_w * contention_penalty
        if score > best_score:
            best_score = score
            best_subset = subset
            best_backhaul = backhaul_term
    if best_subset:
        sat_sel[np.asarray(best_subset, dtype=np.int32)] = 1.0
    return sat_sel.astype(np.float32, copy=False), float(best_backhaul)


def _topology_dpp_one_agent(
    obs: Dict[str, np.ndarray],
    cfg,
    prev_accel: np.ndarray,
    *,
    accel_candidates: np.ndarray,
    agent_id: int = 0,
    env_callbacks: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    users = np.asarray(obs["users"], dtype=np.float32)
    users_mask = np.asarray(obs["users_mask"], dtype=np.float32) > 0.0
    bw_valid_mask = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]), dtype=np.float32) > 0.0
    candidate_idx = _candidate_indices_np(obs, cfg)
    own_q_norm = float(np.clip(obs["own"][5], 0.0, None)) if len(obs["own"]) > 5 else 0.0

    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3) or 0.0)
    max_users = int(getattr(cfg, "topology_dpp_gu_max_select", 6) or 6)
    max_users = max(min(max_users, int(cfg.users_obs_max)), 1)
    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0) or 0.0)
    dpp_v = float(max(getattr(cfg, "baseline_lyapunov_v", 2.0) or 0.0, 0.0))
    access_w = float(max(getattr(cfg, "topology_dpp_access_weight", 1.0) or 0.0, 0.0))
    backhaul_w = float(max(getattr(cfg, "topology_dpp_backhaul_weight", 1.0) or 0.0, 0.0))
    mobility_w = float(max(getattr(cfg, "topology_dpp_mobility_weight", 0.75) or 0.0, 0.0))
    accel_cost = float(max(getattr(cfg, "topology_dpp_accel_cost", 0.08) or 0.0, 0.0))
    smooth_w = float(max(getattr(cfg, "topology_dpp_smoothness", 0.0) or 0.0, 0.0))
    dist_penalty = float(max(getattr(cfg, "topology_dpp_dist_penalty", 0.1) or 0.0, 0.0))
    service_scale = float(max(getattr(cfg, "baseline_lyapunov_bw_service_scale", 1.0) or 0.0, 0.0))

    best_score = -1.0e30
    best_accel = np.zeros((2,), dtype=np.float32)
    best_bw_full = np.zeros((int(cfg.num_gu),), dtype=np.float32)
    best_sat = np.zeros((int(cfg.sats_obs_max),), dtype=np.float32)
    best_pressure_full = np.zeros((int(cfg.num_gu),), dtype=np.float32)
    best_service_full = np.zeros((int(cfg.num_gu),), dtype=np.float32)
    best_terms = (0.0, 0.0, 0.0, -1.0e30)

    for cand in np.asarray(accel_candidates, dtype=np.float32):
        topo = _predict_topology_after_accel(
            obs,
            cfg,
            cand,
            agent_id=int(agent_id),
            env_callbacks=env_callbacks,
        )
        rel_next = topo["rel_next"]
        eta_updated = topo["eta"]
        sat_visible_mask = topo["sat_visible_mask"]
        slot_bw = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        pressure_slots = np.zeros_like(slot_bw)
        service_slots = np.zeros_like(slot_bw)

        access_term = 0.0
        mobility_term = 0.0
        if np.any(users_mask):
            q = np.clip(np.asarray(users[users_mask, 2], dtype=np.float32), 0.0, None)
            expected = np.zeros_like(q, dtype=np.float32)
            if int(users.shape[1]) > 5:
                expected = np.clip(np.asarray(users[users_mask, 5], dtype=np.float32), 0.0, None)
            eta_obs = np.clip(np.asarray(users[users_mask, 3], dtype=np.float32), 0.0, None)
            eta_new = np.clip(np.asarray(eta_updated[users_mask], dtype=np.float32), 0.0, None)
            eta_blend = 0.4 * eta_obs + 0.6 * eta_new
            prev_assoc = np.clip(np.asarray(users[users_mask, 4], dtype=np.float32), 0.0, 1.0)
            rel_dist = np.linalg.norm(rel_next[users_mask], axis=1)
            # GU access should not be suppressed by the UAV queue scale; backhaul
            # pressure is handled separately in SAT selection.
            demand_pressure = np.clip(q + expected, 0.0, None)
            rate_proxy = np.clip(0.5 + eta_blend, 0.0, 2.0)
            assoc_term = 1.0 + assoc_bonus * prev_assoc
            service_pressure = demand_pressure * rate_proxy * assoc_term
            score_slice = service_pressure - dist_penalty * rel_dist
            pressure_slice = service_pressure
            pressure_slots[users_mask] = np.clip(pressure_slice, 0.0, None)
            pressure_sum = float(np.sum(service_pressure))
            if pressure_sum > 1.0e-9:
                target_rel = (rel_next[users_mask] * service_pressure[:, None]).sum(axis=0) / (pressure_sum + 1.0e-9)
                desired_accel = _project_normalized_accel_np(target_rel * accel_gain)
                cand_accel = _project_normalized_accel_np(cand * accel_gain)
                mobility_term = float(np.dot(cand_accel, desired_accel))

            valid_slots = users_mask & bw_valid_mask & (candidate_idx >= 0) & (candidate_idx < int(cfg.num_gu))
            if np.any(valid_slots):
                candidate_scores = np.full((int(cfg.users_obs_max),), -1.0e6, dtype=np.float32)
                candidate_scores[users_mask] = score_slice
                rank_mask = _topk_select_mask(candidate_scores, valid_slots, max_users) > 0.0
                bw_scores = np.full((int(cfg.users_obs_max),), -1.0e6, dtype=np.float32)
                bw_scores[rank_mask] = dpp_v * candidate_scores[rank_mask]
                slot_bw = _topology_dpp_allocate_bw(bw_scores, rank_mask, cfg)
                eta_slot = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
                pressure_slot = np.zeros_like(eta_slot)
                eta_slot[users_mask] = rate_proxy
                pressure_slot[users_mask] = service_pressure
                service_slots = service_scale * slot_bw * eta_slot
                access_term = float(np.sum(pressure_slot * service_slots))
            else:
                access_term = -dist_penalty * float(np.mean(np.linalg.norm(rel_next[users_mask], axis=1)))
        elif rel_next.size > 0:
            access_term = -dist_penalty * float(np.mean(np.linalg.norm(rel_next, axis=1)))

        sat_sel, backhaul_term = _topology_dpp_sat_selection(
            obs,
            cfg,
            own_q_norm,
            sat_visible_mask=sat_visible_mask,
        )
        reg_term = accel_cost * float(np.dot(cand, cand)) + smooth_w * float(np.sum((cand - prev_accel) ** 2))
        score = access_w * access_term + backhaul_w * backhaul_term + mobility_w * mobility_term - reg_term
        if score > best_score:
            best_score = float(score)
            best_accel = _project_normalized_accel_np(cand * accel_gain)
            best_bw_full = _slot_bw_to_full_gu_np(slot_bw, obs, cfg)
            best_sat = sat_sel.astype(np.float32, copy=False)
            best_pressure_full = _slot_bw_to_full_gu_np(pressure_slots, obs, cfg)
            best_service_full = _slot_bw_to_full_gu_np(service_slots, obs, cfg)
            best_terms = (float(access_term), float(backhaul_term), float(reg_term), float(score))

    return (
        best_accel.astype(np.float32, copy=False),
        best_bw_full.astype(np.float32, copy=False),
        best_sat.astype(np.float32, copy=False),
        best_pressure_full.astype(np.float32, copy=False),
        best_service_full.astype(np.float32, copy=False),
        best_terms[0],
        best_terms[1],
        best_terms[2],
        best_terms[3],
    )


def topology_dpp_policy_step(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    state: Dict[str, np.ndarray] | None = None,
    *,
    compute_accel: bool = True,
    compute_bw: bool = True,
    compute_sat: bool = True,
    update_pressure: bool = True,
    update_service: bool = True,
    env_callbacks: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    bw_alloc = np.zeros((num_agents, int(cfg.num_gu)), dtype=np.float32)
    sat_select_mask = np.zeros((num_agents, int(cfg.sats_obs_max)), dtype=np.float32)
    if state is None or state.get("pressure_ema") is None or state["pressure_ema"].shape != (num_agents, int(cfg.num_gu)):
        state = _topology_dpp_state_init(num_agents, cfg)

    pressure_ema = np.asarray(state["pressure_ema"], dtype=np.float32)
    virtual_queue = np.asarray(state["virtual_queue"], dtype=np.float32)
    service_est = np.asarray(state["service_est"], dtype=np.float32)
    prev_accel = np.asarray(state["prev_accel"], dtype=np.float32)
    dpp_access_term = np.asarray(state.get("dpp_access_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_backhaul_term = np.asarray(state.get("dpp_backhaul_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_reg_term = np.asarray(state.get("dpp_reg_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    dpp_objective_term = np.asarray(state.get("dpp_objective_term", np.zeros((num_agents,), dtype=np.float32)), dtype=np.float32)
    ema_beta = float(np.clip(getattr(cfg, "baseline_lyapunov_ema_beta", 0.0), 0.0, 0.999))
    candidates = _topology_dpp_accel_candidates(cfg, allow_motion=bool(compute_accel))

    for i, obs in enumerate(obs_list):
        accel_i, bw_i, sat_i, pressure_i, service_i, access_i, backhaul_i, reg_i, obj_i = _topology_dpp_one_agent(
            obs,
            cfg,
            prev_accel[i],
            accel_candidates=candidates,
            agent_id=i,
            env_callbacks=env_callbacks,
        )
        if update_pressure:
            pressure_ema[i] = ema_beta * pressure_ema[i] + (1.0 - ema_beta) * pressure_i
            virtual_queue[i] = np.clip(virtual_queue[i] + pressure_ema[i] - service_est[i], 0.0, None)
        if update_service:
            service_est[i] = service_i
        dpp_access_term[i] = float(access_i)
        dpp_backhaul_term[i] = float(backhaul_i)
        dpp_reg_term[i] = float(reg_i)
        dpp_objective_term[i] = float(obj_i)
        if compute_accel:
            accel_vec = accel_i + _baseline_repulse_term_np(obs, cfg) + _baseline_energy_term(obs, cfg)
            accel[i] = _project_normalized_accel_np(accel_vec)
            prev_accel[i] = accel[i]
        if compute_bw and cfg.enable_bw_action:
            bw_alloc[i] = bw_i
        if compute_sat and not cfg.fixed_satellite_strategy:
            sat_select_mask[i] = sat_i

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
    return accel, bw_alloc, sat_select_mask, next_state


def topology_dpp_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    accel, bw_alloc, sat_select_mask, _ = topology_dpp_policy_step(obs_list, cfg, state=None)
    return accel, bw_alloc, sat_select_mask


def _lyapunov_state_init(num_agents: int, cfg) -> Dict[str, np.ndarray]:
    num_gu = int(getattr(cfg, "num_gu", 0))
    return {
        "pressure_ema": np.zeros((num_agents, num_gu), dtype=np.float32),
        "virtual_queue": np.zeros((num_agents, num_gu), dtype=np.float32),
        "service_est": np.zeros((num_agents, num_gu), dtype=np.float32),
    }


def lyapunov_queue_aware_policy_step(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
    state: Dict[str, np.ndarray] | None = None,
    *,
    compute_accel: bool = True,
    compute_bw: bool = True,
    compute_sat: bool = True,
    update_pressure: bool = True,
    update_service: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Lyapunov / MaxWeight style three-head heuristic baseline.

    Movement follows queue-pressure toward users with current and expected
    demand, bandwidth uses current queue times service ability, and satellite
    selection uses UAV-to-satellite backpressure.
    """

    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    bw_alloc = np.zeros((num_agents, cfg.num_gu), dtype=np.float32)
    sat_select_mask = np.zeros((num_agents, cfg.sats_obs_max), dtype=np.float32)

    if state is None:
        state = _lyapunov_state_init(num_agents, cfg)
    elif (
        state.get("pressure_ema") is None
        or state.get("virtual_queue") is None
        or state.get("service_est") is None
        or state["pressure_ema"].shape != (num_agents, int(cfg.num_gu))
    ):
        state = _lyapunov_state_init(num_agents, cfg)

    pressure_ema = np.asarray(state["pressure_ema"], dtype=np.float32)
    virtual_queue = np.asarray(state["virtual_queue"], dtype=np.float32)
    service_est = np.asarray(state["service_est"], dtype=np.float32)

    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0

    lyap_v = max(float(getattr(cfg, "baseline_lyapunov_v", 2.0) or 0.0), 0.0)
    lyap_urgency_alpha = max(float(getattr(cfg, "baseline_lyapunov_urgency_alpha", 1.0) or 0.0), 0.0)
    lyap_service_scale = max(float(getattr(cfg, "baseline_lyapunov_bw_service_scale", 1.0) or 0.0), 0.0)
    lyap_sat_drift_w = max(float(getattr(cfg, "baseline_lyapunov_sat_drift_weight", 0.6) or 0.0), 0.0)

    for i, obs in enumerate(obs_list):
        accel_vec = np.zeros((2,), dtype=np.float32)
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        candidate_idx = _candidate_indices_np(obs, cfg)
        bw_valid_mask = np.asarray(obs.get("bw_valid_mask", obs["users_mask"]) > 0.0)
        sat_valid_mask = np.asarray(obs.get("sat_valid_mask", obs["sats_mask"]) > 0.0)
        sat_profile = _sat_channel_profile(obs["sats"], sat_valid_mask, cfg)
        relay_gate = 1.0
        if not cfg.fixed_satellite_strategy:
            if np.any(sat_valid_mask):
                relay_gate = 0.5 + 0.5 * float(np.max(sat_profile["relay_support"][sat_valid_mask]))
            else:
                relay_gate = 0.5

        urgency_slots = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
        if np.any(users_mask):
            visible_slots = np.flatnonzero(users_mask)
            visible_gu = candidate_idx[visible_slots]
            mapped = (visible_gu >= 0) & (visible_gu < int(cfg.num_gu))
            visible_slots = visible_slots[mapped]
            visible_gu = visible_gu[mapped]

        if np.any(users_mask) and visible_slots.size > 0:
            rel = np.asarray(users[visible_slots, 0:2], dtype=np.float32)
            q = np.clip(np.asarray(users[visible_slots, 2], dtype=np.float32), 0.0, None)
            eta = np.clip(np.asarray(users[visible_slots, 3], dtype=np.float32), 0.0, None)
            expected = np.zeros_like(q, dtype=np.float32)
            if users.shape[1] > 5:
                expected = np.clip(np.asarray(users[visible_slots, 5], dtype=np.float32), 0.0, None)
            pressure_slice = (q + expected) * (0.5 + eta)
            pressure_slice = np.clip(pressure_slice, 0.0, None)

            if update_pressure:
                pressure_ema[i].fill(0.0)
                pressure_ema[i, visible_gu] = pressure_slice
                virtual_queue[i].fill(0.0)
            urgency_full = np.clip(pressure_ema[i], 0.0, None)
            urgency_slots[visible_slots] = urgency_full[visible_gu]
            nbrs = obs["nbrs"]
            nbrs_mask = obs["nbrs_mask"] > 0.0
            if compute_accel and np.any(nbrs_mask):
                dist_gu = np.linalg.norm(rel, axis=1)
                rel_nbr = nbrs[nbrs_mask, 0:2]
                diff = rel[:, None, :] - rel_nbr[None, :, :]
                dist_nbrs_to_user = np.linalg.norm(diff, axis=-1)
                min_nbr_dist = np.min(dist_nbrs_to_user, axis=1)
                ratio = np.exp(lyap_urgency_alpha * (min_nbr_dist - dist_gu))
                responsibility = np.clip(ratio, 0.0, 1.0) 
                urgency_slots[visible_slots] = urgency_slots[visible_slots] * responsibility
                
            move_weights = np.clip(urgency_slots[visible_slots], 0.0, None)
            urgency_sum = float(np.sum(move_weights))
            if compute_accel and urgency_sum > 1e-6:
                rel_target = (rel * move_weights[:, None]).sum(axis=0) / (urgency_sum + 1e-9)
                error = np.asarray(rel_target, dtype=np.float32) * float(cfg.map_size)
                dist = float(np.linalg.norm(error))
                vel = np.asarray(obs["own"][2:4], dtype=np.float32) * float(cfg.v_max)
                speed = float(np.linalg.norm(vel))
                stop_radius = max(float(getattr(cfg, "baseline_cluster_stop_radius", 20.0) or 0.0), 0.0)
                speed_tol = max(float(getattr(cfg, "baseline_cluster_speed_tol", 2.0) or 0.0), 0.0)
                slow_radius = max(
                    float(getattr(cfg, "baseline_cluster_slow_radius", 120.0) or 0.0),
                    stop_radius + 1e-6,
                )
                cruise_speed_cfg = getattr(cfg, "baseline_cluster_cruise_speed", None)
                cruise_speed = cfg.uav_opt_speed if cruise_speed_cfg is None else float(cruise_speed_cfg)
                cruise_speed = float(np.clip(cruise_speed, 0.0, cfg.v_max))
                vel_gain = max(float(getattr(cfg, "baseline_cluster_vel_gain", 1.0) or 0.0), 0.0) * accel_gain
                desired_vel = np.zeros((2,), dtype=np.float32)
                if dist > stop_radius:
                    desired_speed = cruise_speed * min(dist / max(slow_radius, 1e-6), 1.0)
                    desired_vel = error / max(dist, 1e-6) * desired_speed
                accel_vec = accel_vec + vel_gain * (desired_vel - vel) / max(float(cfg.tau0), 1e-6) / max(float(cfg.a_max), 1e-6)
                if dist <= stop_radius and speed <= speed_tol:
                    accel_vec = np.zeros((2,), dtype=np.float32)

            if compute_bw and cfg.enable_bw_action:
                service_slots = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
                service_slots[visible_slots] = 0.5 + eta
                queue_slots = np.zeros((int(cfg.users_obs_max),), dtype=np.float32)
                queue_slots[visible_slots] = q
                base_scores = lyap_v * relay_gate * queue_slots * service_slots
                bw_scores = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                valid_slots = bw_valid_mask & users_mask & (candidate_idx >= 0) & (candidate_idx < int(cfg.num_gu))
                bw_scores[valid_slots] = np.clip(base_scores[valid_slots], 1e-6, None)
                denom_raw = float(np.sum(bw_scores[valid_slots]))
                probs = np.zeros_like(bw_scores, dtype=np.float32)
                if denom_raw > 1e-9:
                    probs[valid_slots] = bw_scores[valid_slots] / denom_raw
                elif np.any(valid_slots):
                    probs[valid_slots] = 1.0 / float(np.sum(valid_slots))
                slot_bw = probs.astype(np.float32, copy=False)
                bw_alloc[i] = _slot_bw_to_full_gu_np(slot_bw, obs, cfg)

                if update_service:
                    service_est[i].fill(0.0)
                    full_service = _slot_bw_to_full_gu_np(slot_bw * service_slots, obs, cfg)
                    service_est[i] = lyap_service_scale * relay_gate * full_service
            elif update_service and compute_bw:
                service_est[i].fill(0.0)
        else:
            if update_pressure:
                pressure_ema[i].fill(0.0)
                virtual_queue[i].fill(0.0)
            if update_service and compute_bw:
                service_est[i].fill(0.0)

        if compute_sat and not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            if np.any(sat_valid_mask):
                uav_queue = float(np.clip(obs["own"][5], 0.0, None)) if len(obs["own"]) > 5 else 0.0
                qsat = sat_profile["queue"]
                sat_scores = lyap_sat_drift_w * (uav_queue - qsat) * sat_profile["relay_support"]
                sat_select_mask[i] = _topk_select_mask(
                    sat_scores,
                    sat_valid_mask,
                    _sat_action_select_k_from_cfg(cfg),
                )

        if compute_accel and repulse_gain > 0.0 and repulse_radius > 0.0:
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

        if compute_accel:
            accel_vec = accel_vec + _baseline_energy_term(obs, cfg)
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
    bw_alloc = np.zeros((num_agents, cfg.num_gu), dtype=np.float32)
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
                slot_bw = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                if denom > 1e-6:
                    slot_bw = slot_weights / denom
                elif np.any(bw_valid_mask):
                    slot_bw[bw_valid_mask] = 1.0 / float(np.sum(bw_valid_mask))
                width = min(int(cfg.num_gu), int(slot_bw.shape[0]))
                if width > 0:
                    bw_alloc[i, :width] = slot_bw[:width]

        if not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            sat_valid_mask = np.asarray(obs.get("sat_valid_mask", obs["sats_mask"]) > 0.0)
            if np.any(sat_valid_mask):
                sat_scores = _sat_heuristic_score(sats, sat_valid_mask, cfg)
                sat_select_mask[i] = _topk_select_mask(
                    sat_scores,
                    sat_valid_mask,
                    _sat_action_select_k_from_cfg(cfg),
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
                    accel_vec = accel_vec - repulse_gain * (direction * strength[:, None]).sum(axis=0)

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

        accel[i] = _project_normalized_accel_np(accel_vec)

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
