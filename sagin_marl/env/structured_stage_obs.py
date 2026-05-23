from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env import channel
from sagin_marl.env.numeric_guards import (
    LOG_RATIO_EPS,
    divide_or_default,
    log_ratio_argument,
    normalize_scale,
    relative_log_argument,
    require_positive_float,
)
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver


def refresh_stage_obs_cache(driver: StructuredControlDriver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    eta_feature = np.asarray(
        driver._get_stage_eta_ref_feature(driver._stage_assoc, driver._stage_candidates),
        dtype=np.float32,
    )
    eta_slots = np.zeros((env.cfg.num_uav, env.cfg.users_obs_max), dtype=np.float32)
    for u in range(env.cfg.num_uav):
        cand = driver._stage_candidates[u][: env.cfg.users_obs_max]
        for slot, gu_idx in enumerate(cand):
            gu_id = int(gu_idx)
            if 0 <= gu_id < env.cfg.num_gu:
                eta_slots[u, slot] = float(eta_feature[u, gu_id])
    env._store_cached_access_stage_context(
        driver._stage_assoc,
        driver._stage_candidates,
        eta=eta_slots,
        bw_valid_mask=driver._stage_bw_valid_mask,
        snapshot_step_t=int(env.t),
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        driver._refresh_stage_sat_obs_cache()
    driver._stage_obs_cache = current_obs_list_uncached(driver)


def _stage_assoc_array(driver: StructuredControlDriver) -> np.ndarray:
    env = driver.env
    cfg = env.cfg
    return (
        np.asarray(driver._stage_assoc, dtype=np.int32)
        if driver._stage_assoc is not None and np.asarray(driver._stage_assoc).shape == (cfg.num_gu,)
        else np.asarray(getattr(env, "_cached_assoc", env._associate_users()), dtype=np.int32)
    )


def _assoc_centroid_summary_from_stage(driver: StructuredControlDriver) -> tuple[np.ndarray, np.ndarray]:
    env = driver.env
    cfg = env.cfg
    assoc = _stage_assoc_array(driver)
    counts = np.zeros((cfg.num_uav,), dtype=np.float32)
    rel_centroids = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    map_scale = normalize_scale(float(cfg.map_size))
    for u in range(cfg.num_uav):
        idx = np.nonzero(assoc == u)[0]
        if idx.size <= 0:
            continue
        counts[u] = float(idx.size)
        centroid = np.mean(env.gu_pos[idx], axis=0)
        rel_centroids[u] = ((centroid - env.uav_pos[u]) / map_scale).astype(np.float32, copy=False)
    return counts, rel_centroids


def _neighbor_order_from_stage(driver: StructuredControlDriver) -> np.ndarray:
    env = driver.env
    cfg = env.cfg
    if driver._stage_uav_uav_dist is not None:
        dist = np.asarray(driver._stage_uav_uav_dist, dtype=np.float32).copy()
        if dist.shape == (cfg.num_uav, cfg.num_uav):
            np.fill_diagonal(dist, np.inf)
            return np.argsort(dist, axis=1)
    env._ensure_neighbor_cache()
    return np.asarray(env._cached_uav_neighbor_order, dtype=np.int64)


def _danger_neighbor_obs_from_stage(driver: StructuredControlDriver) -> np.ndarray:
    env = driver.env
    cfg = env.cfg
    danger = np.zeros((cfg.num_uav, env.danger_nbr_dim), dtype=np.float32)
    if cfg.num_uav <= 1:
        return danger
    if (
        driver._stage_uav_uav_rel_pos is None
        or driver._stage_uav_uav_rel_vel is None
        or driver._stage_uav_uav_dist is None
    ):
        return np.stack([env._danger_neighbor_obs(u) for u in range(cfg.num_uav)], axis=0).astype(np.float32, copy=False)

    rel_pos_all = np.asarray(driver._stage_uav_uav_rel_pos, dtype=np.float32)
    rel_vel_all = np.asarray(driver._stage_uav_uav_rel_vel, dtype=np.float32)
    dist_all = np.asarray(driver._stage_uav_uav_dist, dtype=np.float32)
    d_alert = float(cfg.avoidance_alert_factor) * float(cfg.d_safe) if bool(cfg.avoidance_enabled) else 0.0
    raw_prealert_factor = getattr(cfg, "avoidance_prealert_factor", None)
    trigger_dist = d_alert
    if raw_prealert_factor is not None:
        trigger_dist = max(float(raw_prealert_factor) * float(cfg.d_safe), d_alert)
    prealert_mode = str(getattr(cfg, "avoidance_prealert_mode", "distance") or "distance").strip().lower()
    if prealert_mode not in {"distance", "ttc"}:
        prealert_mode = "distance"
    if prealert_mode == "ttc":
        raw_prealert_dist_cap = getattr(cfg, "avoidance_prealert_dist_cap", None)
        if raw_prealert_dist_cap is not None:
            trigger_dist = max(float(raw_prealert_dist_cap), d_alert)
    closing_speed_thresh = max(float(getattr(cfg, "avoidance_prealert_closing_speed", 0.0) or 0.0), 0.0)
    prealert_ttc_limit = max(float(getattr(cfg, "avoidance_prealert_ttc", 0.0) or 0.0), 0.0)

    for u in range(cfg.num_uav):
        best = None
        best_key = None
        for j in range(cfg.num_uav):
            if j == u:
                continue
            rel_pos = rel_pos_all[u, j]
            dist = float(dist_all[u, j])
            if dist <= 1.0e-6:
                continue
            rel_vel = rel_vel_all[u, j]
            closing_speed = float(-(np.dot(rel_pos, rel_vel) / dist))
            closing_pos = max(closing_speed, 0.0)
            ttc_to_alert = float("inf")
            if d_alert > 0.0:
                if dist <= d_alert:
                    ttc_to_alert = 0.0
                elif closing_pos > 1.0e-6:
                    ttc_to_alert = (dist - d_alert) / closing_pos
            in_core_alert = bool(d_alert > 0.0 and dist < d_alert)
            if prealert_mode == "ttc":
                in_prealert = bool(
                    trigger_dist > d_alert
                    and dist < trigger_dist
                    and closing_speed > closing_speed_thresh
                    and prealert_ttc_limit > 0.0
                    and np.isfinite(ttc_to_alert)
                    and ttc_to_alert < prealert_ttc_limit
                )
            else:
                in_prealert = bool(
                    trigger_dist > d_alert
                    and dist < trigger_dist
                    and closing_speed > closing_speed_thresh
                )
            key = (
                1 if in_core_alert else 0,
                1 if in_prealert else 0,
                1 if closing_pos > 0.0 else 0,
                closing_pos,
                -dist,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (rel_pos, dist, closing_pos)
        if best is None:
            continue
        rel_pos, dist, closing_pos = best
        direction = rel_pos / normalize_scale(dist)
        danger[u] = np.asarray(
            [
                dist / normalize_scale(float(cfg.map_size)),
                np.clip(closing_pos / normalize_scale(float(cfg.v_max)), 0.0, 1.0),
                direction[0],
                direction[1],
                1.0,
            ],
            dtype=np.float32,
        )
    return danger


def _reward_aligned_feature_bundle(
    env: SaginParallelEnv,
    *,
    assoc: np.ndarray,
    sat_selection: list[list[int]] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cfg = env.cfg
    eps = env._bw_weighted_workload_eps()
    gu_ema, uav_ema, sat_ema = env._bw_weighted_workload_device_ema_vectors()
    sat_cost = (1.0 / np.maximum(np.asarray(sat_ema, dtype=np.float32), eps)).astype(np.float32, copy=False)
    sat_cost_fallback = float(np.mean(sat_cost)) if sat_cost.size > 0 else 0.0

    uav_downstream_cost = np.full((cfg.num_uav,), sat_cost_fallback, dtype=np.float32)
    if isinstance(sat_selection, list) and sat_cost.size > 0:
        for u in range(min(len(sat_selection), cfg.num_uav)):
            selected = np.asarray(sat_selection[u], dtype=np.int64).reshape(-1)
            valid = selected[(selected >= 0) & (selected < cfg.num_sat)]
            if valid.size > 0:
                uav_downstream_cost[u] = float(np.mean(sat_cost[valid]))
    uav_cost = (1.0 / np.maximum(np.asarray(uav_ema, dtype=np.float32), eps) + uav_downstream_cost).astype(
        np.float32,
        copy=False,
    )
    uav_cost_fallback = float(np.mean(uav_cost)) if uav_cost.size > 0 else 0.0

    assoc_arr = np.asarray(assoc, dtype=np.int32).reshape(-1)
    gu_downstream_cost = np.full((cfg.num_gu,), uav_cost_fallback, dtype=np.float32)
    if assoc_arr.shape == (cfg.num_gu,) and cfg.num_uav > 0:
        valid_assoc = (assoc_arr >= 0) & (assoc_arr < cfg.num_uav)
        if np.any(valid_assoc):
            gu_downstream_cost[valid_assoc] = uav_cost[assoc_arr[valid_assoc]]
    gu_cost = (1.0 / np.maximum(np.asarray(gu_ema, dtype=np.float32), eps) + gu_downstream_cost).astype(
        np.float32,
        copy=False,
    )

    assoc_uav_cost = np.full(
        (cfg.num_gu,),
        float(np.mean(uav_cost)) if uav_cost.size > 0 else 0.0,
        dtype=np.float32,
    )
    assoc_sat_cost_mean = np.full(
        (cfg.num_gu,),
        float(np.mean(sat_cost)) if sat_cost.size > 0 else 0.0,
        dtype=np.float32,
    )
    valid_assoc = (assoc_arr >= 0) & (assoc_arr < cfg.num_uav)
    if np.any(valid_assoc) and uav_cost.size > 0:
        assoc_uav_cost[valid_assoc] = uav_cost[assoc_arr[valid_assoc]]
    if isinstance(sat_selection, list) and sat_cost.size > 0:
        sat_mean_by_uav = np.full(
            (cfg.num_uav,),
            float(np.mean(sat_cost)),
            dtype=np.float32,
        )
        for u in range(min(len(sat_selection), cfg.num_uav)):
            selected = np.asarray(sat_selection[u], dtype=np.int64).reshape(-1)
            valid = selected[(selected >= 0) & (selected < cfg.num_sat)]
            if valid.size > 0:
                sat_mean_by_uav[u] = float(np.mean(sat_cost[valid]))
        if np.any(valid_assoc):
            assoc_sat_cost_mean[valid_assoc] = sat_mean_by_uav[assoc_arr[valid_assoc]]

    local_gu_service_cost = (1.0 / np.maximum(np.asarray(gu_ema, dtype=np.float32), eps)).astype(np.float32, copy=False)
    weighted_queue_cost = (
        np.asarray(gu_cost, dtype=np.float32) * np.asarray(env.gu_queue, dtype=np.float32)
    ).astype(np.float32, copy=False)
    weighted_queue_cost_relative = divide_or_default(
        weighted_queue_cost,
        float(np.mean(weighted_queue_cost, dtype=np.float32)),
        default=0.0,
    ).astype(np.float32, copy=False)
    _, _, _, gu_total_cost_ref = env._bw_weighted_workload_feature_refs()
    weighted_queue_ref = require_positive_float(
        float(env._arrival_ref()) * float(gu_total_cost_ref),
        name="weighted queue reward feature reference",
    )
    local_cost_mean = max(float(np.mean(local_gu_service_cost, dtype=np.float32)), LOG_RATIO_EPS)
    assoc_uav_cost_mean = max(float(np.mean(assoc_uav_cost, dtype=np.float32)), LOG_RATIO_EPS)
    assoc_sat_cost_mean_mean = max(float(np.mean(assoc_sat_cost_mean, dtype=np.float32)), LOG_RATIO_EPS)
    sat_cost_mean = max(float(np.mean(sat_cost, dtype=np.float32)), LOG_RATIO_EPS) if sat_cost.size > 0 else LOG_RATIO_EPS
    uav_cost_mean = max(float(np.mean(uav_cost, dtype=np.float32)), LOG_RATIO_EPS) if uav_cost.size > 0 else LOG_RATIO_EPS

    gu_reward_aligned = {
        "local_gu_service_cost": np.log(
            log_ratio_argument(local_gu_service_cost) / local_cost_mean
        ).astype(np.float32, copy=False),
        "assoc_uav_cost": np.log(
            log_ratio_argument(assoc_uav_cost) / assoc_uav_cost_mean
        ).astype(np.float32, copy=False),
        "assoc_sat_cost_mean": np.log(
            log_ratio_argument(assoc_sat_cost_mean) / assoc_sat_cost_mean_mean
        ).astype(np.float32, copy=False),
        "weighted_queue_cost": np.log1p(
            np.maximum(weighted_queue_cost, 0.0) / weighted_queue_ref
        ).astype(np.float32, copy=False),
        "weighted_queue_cost_relative": np.log(
            relative_log_argument(weighted_queue_cost_relative)
        ).astype(np.float32, copy=False),
    }
    uav_assoc_uav_cost = np.log(
        log_ratio_argument(np.asarray(uav_cost, dtype=np.float32)) / uav_cost_mean
    ).astype(np.float32, copy=False)
    sat_cost_norm = np.log(
        log_ratio_argument(np.asarray(sat_cost, dtype=np.float32)) / sat_cost_mean
    ).astype(np.float32, copy=False)
    return gu_reward_aligned, uav_assoc_uav_cost, sat_cost_norm


def _gu_proxy_feature_arrays_from_reward_aligned(
    env: SaginParallelEnv,
    *,
    assoc: np.ndarray,
    sat_selection: list[list[int]] | None,
    reward_aligned: dict[str, np.ndarray],
) -> list[np.ndarray]:
    del assoc, sat_selection
    cfg = env.cfg
    features: list[np.ndarray] = []
    base_arrival = require_positive_float(
        float(getattr(env, "effective_task_arrival_rate", cfg.task_arrival_rate)) * float(cfg.tau0),
        name="per-GU arrival reference bits per step",
    )
    if bool(getattr(cfg, "obs_user_include_arrival_rate", False)):
        arrival_rate = np.asarray(env._current_expected_gu_arrival_rates(), dtype=np.float32)
        features.append(arrival_rate / base_arrival)
    if bool(getattr(cfg, "obs_user_include_recent_arrival", False)):
        recent_arrival = np.asarray(
            getattr(env, "last_gu_arrival", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        features.append(recent_arrival / base_arrival)
    if bool(getattr(cfg, "obs_user_include_recent_service", False)):
        recent_service = np.asarray(
            getattr(env, "last_gu_outflow", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        features.append(recent_service / base_arrival)
    if bool(getattr(cfg, "obs_user_include_queue_headroom", False)):
        queue_headroom = 1.0 - (
            np.asarray(env.gu_queue, dtype=np.float32) / normalize_scale(float(cfg.queue_max_gu))
        )
        features.append(queue_headroom.astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)):
        features.append(np.asarray(reward_aligned["local_gu_service_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)):
        features.append(np.asarray(reward_aligned["assoc_uav_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)):
        features.append(np.asarray(reward_aligned["assoc_sat_cost_mean"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)):
        features.append(np.asarray(reward_aligned["weighted_queue_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)):
        features.append(np.asarray(reward_aligned["weighted_queue_cost_relative"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_urgency_risk", False)):
        features.append(
            np.asarray(getattr(env, "last_gu_urgency_risk", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)
        )
    if bool(getattr(cfg, "obs_user_include_downstream_pressure", False)):
        features.append(
            np.asarray(getattr(env, "last_gu_downstream_pressure", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)
        )
    if bool(getattr(cfg, "obs_user_include_service_gap", False)):
        cap_steps = max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6)
        service_gap = np.asarray(
            getattr(env, "last_gu_service_gap", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        features.append(service_gap / cap_steps)
    if bool(getattr(cfg, "obs_user_include_service_gap_risk", False)):
        features.append(
            np.asarray(getattr(env, "last_gu_service_gap_risk", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)
        )
    if bool(getattr(cfg, "obs_user_include_deadline_slack", False)):
        deadline_steps = np.maximum(
            np.asarray(getattr(env, "gu_deadline_steps", np.ones((cfg.num_gu,), dtype=np.float32)), dtype=np.float32),
            1.0e-6,
        )
        deadline_slack = np.asarray(
            getattr(env, "last_gu_deadline_slack", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        features.append(np.clip(deadline_slack / deadline_steps, -1.0, 1.0).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_deadline_risk", False)):
        deadline_risk = np.asarray(
            getattr(env, "last_gu_deadline_risk", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        features.append(np.clip(deadline_risk, 0.0, 2.0).astype(np.float32, copy=False))
    return features


def current_obs_list_uncached(driver: StructuredControlDriver) -> list[dict[str, np.ndarray]]:
    env = driver.env
    cfg = env.cfg
    assoc = _stage_assoc_array(driver)
    sat_selection = None if driver._stage_sat_selection is None else [list(sel) for sel in driver._stage_sat_selection]
    assoc_counts, assoc_rel_centroids = _assoc_centroid_summary_from_stage(driver)
    reward_aligned_needed = (
        bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False))
        or bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False))
        or bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False))
        or bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False))
        or bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False))
        or bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False))
    )
    gu_reward_aligned, uav_assoc_uav_cost, _ = (
        _reward_aligned_feature_bundle(
            env,
            assoc=assoc,
            sat_selection=sat_selection,
        )
        if reward_aligned_needed
        else (
            {
                "local_gu_service_cost": np.zeros((cfg.num_gu,), dtype=np.float32),
                "assoc_uav_cost": np.zeros((cfg.num_gu,), dtype=np.float32),
                "assoc_sat_cost_mean": np.zeros((cfg.num_gu,), dtype=np.float32),
                "weighted_queue_cost": np.zeros((cfg.num_gu,), dtype=np.float32),
                "weighted_queue_cost_relative": np.zeros((cfg.num_gu,), dtype=np.float32),
            },
            np.zeros((cfg.num_uav,), dtype=np.float32),
            np.zeros((cfg.num_sat,), dtype=np.float32),
        )
    )
    uav_reward_features = {"assoc_uav_cost": uav_assoc_uav_cost} if bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False)) else None
    gu_proxy_features = _gu_proxy_feature_arrays_from_reward_aligned(
        env,
        assoc=assoc,
        sat_selection=sat_selection,
        reward_aligned=gu_reward_aligned,
    )
    neighbor_order = _neighbor_order_from_stage(driver)
    uav_gu_rel = (
        np.asarray(driver._stage_uav_gu_rel, dtype=np.float32)
        if driver._stage_uav_gu_rel is not None
        else None
    )
    prev_assoc_flag = (
        np.asarray(driver._stage_prev_assoc_flag, dtype=np.float32)
        if driver._stage_prev_assoc_flag is not None
        else None
    )
    uav_uav_rel_pos = (
        np.asarray(driver._stage_uav_uav_rel_pos, dtype=np.float32)
        if driver._stage_uav_uav_rel_pos is not None
        else None
    )
    uav_uav_rel_vel = (
        np.asarray(driver._stage_uav_uav_rel_vel, dtype=np.float32)
        if driver._stage_uav_uav_rel_vel is not None
        else None
    )
    map_scale = normalize_scale(float(cfg.map_size))
    v_scale = normalize_scale(float(cfg.v_max))
    danger_obs = (
        _danger_neighbor_obs_from_stage(driver)
        if bool(getattr(cfg, "danger_nbr_enabled", False))
        else None
    )
    obs_list: list[dict[str, np.ndarray]] = []
    for u in range(len(env.agents)):
        assoc_count_norm = assoc_counts[u] / max(float(cfg.num_gu), 1.0)
        assoc_centroid_rel = assoc_rel_centroids[u]
        own_list = [
            env.uav_pos[u, 0] / map_scale,
            env.uav_pos[u, 1] / map_scale,
            env.uav_vel[u, 0] / v_scale,
            env.uav_vel[u, 1] / v_scale,
            env.uav_energy[u] / normalize_scale(float(cfg.uav_energy_init)),
            env.uav_queue[u] / normalize_scale(float(cfg.queue_max_uav)),
            0.0,  # Reserved: do not expose the artificial rollout horizon.
            assoc_count_norm,
            assoc_centroid_rel[0],
            assoc_centroid_rel[1],
        ]
        if uav_reward_features is not None:
            own_list.append(float(np.asarray(uav_reward_features["assoc_uav_cost"], dtype=np.float32)[u]))
        own = np.asarray(own_list, dtype=np.float32)

        users = np.zeros((cfg.users_obs_max, env.user_dim), dtype=np.float32)
        users_mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        candidate_indices = np.full((cfg.users_obs_max,), -1, dtype=np.int64)
        bw_valid_mask = env._cached_bw_valid_mask[u].copy()
        cand = env._cached_candidates[u] if env._cached_candidates else []
        for i, k in enumerate(cand[: cfg.users_obs_max]):
            candidate_indices[i] = int(k)
            if uav_gu_rel is not None:
                users[i, 0:2] = uav_gu_rel[u, k] / map_scale
            else:
                rel = env.gu_pos[k] - env.uav_pos[u]
                users[i, 0:2] = rel / map_scale
            users[i, 2] = env.gu_queue[k] / normalize_scale(float(cfg.queue_max_gu))
            users[i, 3] = env._cached_eta[u, i]
            users[i, 4] = (
                float(prev_assoc_flag[u, k])
                if prev_assoc_flag is not None
                else (1.0 if env.prev_association[k] == u else 0.0)
            )
            feat_col = 5
            for feature in gu_proxy_features:
                users[i, feat_col] = float(feature[k])
                feat_col += 1
            users_mask[i] = 1.0

        sats = env._cached_sat_obs[u].copy()
        sats_mask = env._cached_sat_mask[u].copy()
        sat_valid_mask = env._cached_sat_valid_mask[u].copy()

        nbrs = np.zeros((cfg.nbrs_obs_max, env.nbr_dim), dtype=np.float32)
        nbrs_mask = np.zeros((cfg.nbrs_obs_max,), dtype=np.float32)
        order = neighbor_order[u]
        count = 0
        for idx in order:
            if idx == u:
                continue
            if uav_uav_rel_pos is not None and uav_uav_rel_vel is not None:
                nbrs[count, 0:2] = uav_uav_rel_pos[u, idx] / map_scale
                nbrs[count, 2:4] = uav_uav_rel_vel[u, idx] / v_scale
            else:
                rel_pos = env.uav_pos[idx] - env.uav_pos[u]
                rel_vel = env.uav_vel[idx] - env.uav_vel[u]
                nbrs[count, 0:2] = rel_pos / map_scale
                nbrs[count, 2:4] = rel_vel / v_scale
            nbrs_mask[count] = 1.0
            count += 1
            if count >= cfg.nbrs_obs_max:
                break

        obs = {
            "own": own,
            "users": users,
            "users_mask": users_mask,
            "bw_valid_mask": bw_valid_mask,
            "candidate_indices": candidate_indices,
            "sats": sats,
            "sats_mask": sats_mask,
            "sat_valid_mask": sat_valid_mask,
            "nbrs": nbrs,
            "nbrs_mask": nbrs_mask,
        }
        if danger_obs is not None:
            obs["danger_nbr"] = danger_obs[u].copy()
        obs_list.append(obs)
    return obs_list


def current_obs_list(driver: StructuredControlDriver) -> list[dict[str, np.ndarray]]:
    if driver._stage_obs_cache is not None:
        return [
            {key: np.asarray(value).copy() for key, value in obs.items()}
            for obs in driver._stage_obs_cache
        ]
    return current_obs_list_uncached(driver)


def current_obs_batch(
    drivers: list[StructuredControlDriver] | tuple[StructuredControlDriver, ...],
    *,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    if not drivers:
        return {}
    obs_many = [current_obs_list(driver) for driver in drivers]
    sample_keys = tuple(obs_many[0][0].keys())
    stacked: dict[str, np.ndarray | torch.Tensor] = {}
    out_device = None if device is None else torch.device(device)
    for key in sample_keys:
        value = np.stack(
            [
                np.stack(
                    [np.asarray(obs[key], dtype=np.float32) for obs in obs_list],
                    axis=0,
                )
                for obs_list in obs_many
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if out_device is None:
            stacked[key] = value
        else:
            stacked[key] = torch.as_tensor(value, dtype=torch.float32, device=out_device)
    return stacked


def sat_mask_to_ids(driver: StructuredControlDriver, sat_mask: np.ndarray) -> np.ndarray:
    cfg = driver.env.cfg
    select_k = max(int(getattr(cfg, "sat_action_select_k", cfg.N_RF) or cfg.N_RF), 1)
    out = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    if driver._stage_visible is None:
        raise RuntimeError("run_accel_stage must be called before decoding sat masks")
    sat_mask_arr = np.asarray(sat_mask, dtype=np.float32)
    visible_width = max(
        min(
            int(getattr(cfg, "per_uav_visible_sat_token_max", cfg.sats_obs_max) or cfg.sats_obs_max),
            int(cfg.num_sat),
        ),
        0,
    )
    for u in range(cfg.num_uav):
        visible = driver._stage_visible[u][:visible_width]
        active_slots = np.flatnonzero(sat_mask_arr[u] > 0.5)
        mapped: list[int] = []
        for slot in active_slots.tolist():
            if 0 <= int(slot) < len(visible):
                sat_idx = int(visible[int(slot)])
                if sat_idx not in mapped:
                    mapped.append(sat_idx)
            if len(mapped) >= select_k:
                break
        if mapped:
            out[u, : len(mapped)] = np.asarray(mapped[:select_k], dtype=np.int64)
    return out
