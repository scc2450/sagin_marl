from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv

from .config import (
    SaginConfig,
    ablation_flag,
    access_carrier_freq_from_config as _access_carrier_freq_from_cfg,
    backhaul_carrier_freq_from_config as _backhaul_carrier_freq_from_cfg,
    ensure_structured_accel_actor_config,
)
from .topology import thomas_cluster_process
from .orbit import WalkerDeltaOrbitModel
from . import channel
from .safety_shield import solve_brake_distance_shield
from .numeric_guards import (
    GEOMETRY_DENOM_EPS,
    LOG_RATIO_EPS,
    NORMALIZATION_DENOM_EPS,
    RELATIVE_LOG_EPS,
    RUNTIME_RATIO_ZERO_TOL,
    geometry_denominator,
    divide_or_default,
    log_ratio_argument,
    normalize_scale,
    relative_log_argument,
    require_positive_float,
    scalar_divide_or_default,
    scalar_ratio_or_zero,
    ratio_or_zero,
    reward_ratio_denominator_scalar,
)


def _project_l2_ball_np(values: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    limit = max(float(max_norm), 0.0)
    if limit <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


@dataclass
class BwTransitionCoreResult:
    gu_queue_before: np.ndarray
    uav_queue_before: np.ndarray
    sat_queue_before: np.ndarray
    realized_arrival: np.ndarray
    rate_matrix: np.ndarray
    sat_loads: np.ndarray


@dataclass
class StepStatusCoreResult:
    reward: float
    reward_parts: Dict[str, Any]
    collision: bool
    terminated: bool
    truncated: bool


@dataclass
class StepMaterializationResult:
    obs: Dict[str, Dict[str, np.ndarray]]
    rewards: Dict[str, float]
    terminations: Dict[str, bool]
    truncations: Dict[str, bool]
    infos: Dict[str, Dict[str, object]]
    gu_proxy_features: List[np.ndarray] | None = None


@dataclass
class AccessChannelSnapshot:
    gain_matrix: np.ndarray


def _semantic_quantum(cfg, attr: str, default: float) -> float:
    value = getattr(cfg, attr, default)
    if value is None:
        return float(default)
    return max(float(value), 0.0)


def _quantize_queue_contract_np(
    cfg,
    value: np.ndarray | float,
    *,
    attr: str,
    default: float,
) -> np.ndarray:
    quantum = _semantic_quantum(cfg, attr, default)
    if quantum <= 0.0:
        return np.asarray(value, dtype=np.float32)
    work = np.asarray(value, dtype=np.float32)
    quantum32 = np.float32(quantum)
    return np.asarray(np.rint(work / quantum32).astype(np.float32) * quantum32, dtype=np.float32)


def _quantize_numeric_contract_np(
    cfg,
    value: np.ndarray | float,
    *,
    quantum: float,
    dtype=np.float32,
) -> np.ndarray:
    quantum_value = float(quantum)
    if quantum_value <= 0.0:
        return np.asarray(value, dtype=dtype)
    work = np.asarray(value, dtype=np.float32)
    quantum32 = np.float32(quantum_value)
    return np.asarray(np.rint(work / quantum32).astype(np.float32) * quantum32, dtype=dtype)


def _quantize_access_gain_snapshot_np(cfg, value: np.ndarray) -> np.ndarray:
    return _quantize_numeric_contract_np(
        cfg,
        np.asarray(value, dtype=np.float32),
        quantum=_semantic_quantum(cfg, "structured_access_gain_quantum", 5.0e-16),
        dtype=np.float32,
    )


def _access_interference_quantum_np(cfg) -> float:
    gain_quantum = _semantic_quantum(cfg, "structured_access_gain_quantum", 5.0e-16)
    if gain_quantum <= 0.0:
        return 0.0
    return abs(float(cfg.gu_tx_power)) * float(gain_quantum)


def compute_access_interference_beta_continuous(
    association: np.ndarray,
    access_gain_matrix: np.ndarray,
    gu_band_fraction: np.ndarray,
    *,
    gu_tx_power: float,
    num_uav: int | None = None,
    interference_enabled: bool = True,
) -> np.ndarray:
    assoc = np.asarray(association, dtype=np.int32).reshape(-1)
    gain = np.asarray(access_gain_matrix, dtype=np.float32)
    beta = np.asarray(gu_band_fraction, dtype=np.float32).reshape(-1)
    if gain.ndim != 2:
        raise ValueError("access_gain_matrix must be rank-2 [G,U].")
    if assoc.shape[0] != gain.shape[0] or beta.shape[0] != gain.shape[0]:
        raise ValueError("association, access_gain_matrix, and gu_band_fraction must share GU dimension.")
    uav_count = int(gain.shape[1] if num_uav is None else num_uav)
    if not interference_enabled or gain.size == 0 or uav_count <= 0:
        return np.zeros((max(uav_count, 0),), dtype=np.float32)

    active = (assoc >= 0) & (assoc < uav_count) & (beta > 0.0)
    if not np.any(active):
        return np.zeros((uav_count,), dtype=np.float32)

    active_idx = np.flatnonzero(active)
    serving_uav = assoc[active_idx]
    active_beta = beta[active_idx]
    active_gain = gain[active_idx, :uav_count]
    total_received = (
        np.float32(gu_tx_power)
        * np.sum(active_gain * active_beta[:, None], axis=0, dtype=np.float32)
    ).astype(np.float32, copy=False)
    same_cell = np.zeros((uav_count,), dtype=np.float32)
    np.add.at(
        same_cell,
        serving_uav,
        (
            np.float32(gu_tx_power)
            * gain[active_idx, serving_uav].astype(np.float32, copy=False)
            * active_beta
        ).astype(np.float32, copy=False),
    )
    return np.maximum(total_received - same_cell, np.float32(0.0)).astype(np.float32, copy=False)


def _quantize_access_pathloss_db_np(cfg, value: np.ndarray) -> np.ndarray:
    return _quantize_numeric_contract_np(
        cfg,
        np.asarray(value, dtype=np.float32),
        quantum=_semantic_quantum(cfg, "structured_access_pathloss_db_quantum", 1.0e-2),
        dtype=np.float32,
    )


def _quantize_metric_scalar(cfg, value: float, *, attr: str = "structured_summary_metric_quantum", default: float = 1.0e-5) -> float:
    quantum = _semantic_quantum(cfg, attr, default)
    if quantum <= 0.0:
        return float(value)
    return float(_quantize_numeric_contract_np(cfg, value, quantum=quantum, dtype=np.float32).reshape(()))


def _contract_sum_scalar_np(cfg, value: np.ndarray | float) -> float:
    return float(np.sum(np.asarray(value, dtype=np.float64), dtype=np.float64))


def _contract_add_scalar_np(*values: float) -> float:
    acc = 0.0
    for value in values:
        acc += float(value)
    return float(acc)


def _contract_ratio_scalar_np(
    numerator: float,
    denominator: float,
    *,
    min_denominator: float = NORMALIZATION_DENOM_EPS,
) -> float:
    return float(
        scalar_divide_or_default(
            numerator,
            denominator,
            eps=min_denominator,
            default=0.0,
        )
    )


def _structured_tensor_device_from_cfg(cfg) -> str:
    backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    if backend == "cpu":
        return "cpu"
    if backend in {"cuda", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def _normalize_sat_selection_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    matrix_arr = np.asarray(matrix, dtype=np.int64)
    out = np.full_like(matrix_arr, -1)
    for row_index in range(int(matrix_arr.shape[0])):
        seen: set[int] = set()
        write_index = 0
        for sat_idx in matrix_arr[row_index].tolist():
            sat_value = int(sat_idx)
            if sat_value < 0 or sat_value in seen:
                continue
            out[row_index, write_index] = sat_value
            seen.add(sat_value)
            write_index += 1
            if write_index >= int(matrix_arr.shape[1]):
                break
    return out


class SaginParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "sagin_parallel_v1"}

    def __init__(self, cfg: SaginConfig):
        ensure_structured_accel_actor_config(cfg)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.episode_idx = 0
        self.global_step = 0
        self.orbit = WalkerDeltaOrbitModel(
            cfg.num_sat,
            cfg.r_earth,
            cfg.sat_height,
            num_planes=cfg.walker_num_planes,
            inclination_deg=cfg.walker_inclination_deg,
            phase_factor=cfg.walker_phase_factor,
            earth_rotation_rate=cfg.earth_rotation_rate,
        )
        self._uav_height_sq = cfg.uav_height ** 2
        self._uav_orbit_radius = cfg.r_earth + cfg.uav_height
        self._sat_orbit_radius = cfg.r_earth + cfg.sat_height
        self._uav_orbit_radius_sq = self._uav_orbit_radius ** 2
        self._sat_orbit_radius_sq = self._sat_orbit_radius ** 2
        self._backhaul_gain_const = (
            (cfg.speed_of_light / (4.0 * math.pi * _backhaul_carrier_freq_from_cfg(cfg))) ** 2
            * cfg.uav_tx_gain
            * cfg.sat_rx_gain
        )
        self._orbit_pos_table, self._orbit_vel_table = self._precompute_orbit_lookup()

        self.agents = [f"uav_{i}" for i in range(cfg.num_uav)]
        self.possible_agents = list(self.agents)

        # Dimensions
        self.own_extra_dim = self._extra_own_feature_count()
        self.own_dim = 10 + self.own_extra_dim
        self.user_extra_dim = self._extra_user_feature_count()
        self.user_dim = 5 + self.user_extra_dim
        self.gu_node_dim = 3 + self.user_extra_dim
        self.sat_extra_dim = self._extra_sat_feature_count()
        self.sat_dim = 12 + self.sat_extra_dim
        self.nbr_dim = 4
        self.danger_nbr_dim = 5

        self._build_spaces()
        self.effective_task_arrival_rate = float(cfg.task_arrival_rate)
        raw_level = getattr(cfg, "traffic_level", 2)
        self.traffic_level = int(2 if raw_level is None else raw_level)
        self.traffic_level_ratio = 1.0
        self._set_effective_task_arrival_rate()
        self.avoidance_eta_eff = float(cfg.avoidance_eta)
        self.avoidance_collision_rate_ema = 0.0
        self.prev_episode_collision_rate = 0.0
        self.last_avoidance_eta_exec = float(cfg.avoidance_eta)
        self._episode_collision_count = 0
        self._episode_step_count = 0
        self._init_state()

    def _precompute_orbit_lookup(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        num_steps = max(int(cfg.T_steps), 0) + 1
        pos_table = np.zeros((num_steps, cfg.num_sat, 3), dtype=np.float32)
        vel_table = np.zeros((num_steps, cfg.num_sat, 3), dtype=np.float32)
        for t_idx in range(num_steps):
            pos_t, vel_t = self.orbit.get_states(float(t_idx) * cfg.tau0)
            pos_table[t_idx] = pos_t
            vel_table[t_idx] = vel_t
        return pos_table, vel_table

    @staticmethod
    def _candidate_lists_equal(left: List[List[int]] | None, right: List[List[int]] | None) -> bool:
        if left is None or right is None or len(left) != len(right):
            return False
        return all(list(a) == list(b) for a, b in zip(left, right))

    def _store_cached_eta(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        eta: np.ndarray,
        *,
        access_snapshot: AccessChannelSnapshot | np.ndarray | None = None,
        snapshot_step_t: int | None = None,
        uav_pos: np.ndarray | None = None,
        gu_pos: np.ndarray | None = None,
    ) -> None:
        self._cached_assoc = np.asarray(assoc, dtype=np.int32).copy()
        self._cached_candidates = [list(c) for c in candidates]
        self._cached_eta = np.asarray(eta, dtype=np.float32).copy()
        self._cached_eta_uav_pos = np.asarray(self.uav_pos if uav_pos is None else uav_pos, dtype=np.float32).copy()
        self._cached_eta_gu_pos = np.asarray(self.gu_pos if gu_pos is None else gu_pos, dtype=np.float32).copy()
        self._cached_access_gain_matrix = (
            None
            if access_snapshot is None
            else self._coerce_access_gain_matrix(access_snapshot).copy()
        )
        self._cached_access_snapshot_t = int(self.t if snapshot_step_t is None else snapshot_step_t)

    def _store_cached_access_stage_context(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        *,
        eta: np.ndarray,
        bw_valid_mask: np.ndarray | None = None,
        access_snapshot: AccessChannelSnapshot | np.ndarray | None = None,
        snapshot_step_t: int | None = None,
        uav_pos: np.ndarray | None = None,
        gu_pos: np.ndarray | None = None,
    ) -> None:
        if bw_valid_mask is not None:
            self._cached_bw_valid_mask = np.asarray(bw_valid_mask, dtype=np.float32).copy()
        self._store_cached_eta(
            assoc,
            candidates,
            eta,
            access_snapshot=access_snapshot,
            snapshot_step_t=snapshot_step_t,
            uav_pos=uav_pos,
            gu_pos=gu_pos,
        )

    def _cached_eta_matches(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        *,
        snapshot_step_t: int | None = None,
        uav_pos: np.ndarray | None = None,
        gu_pos: np.ndarray | None = None,
    ) -> bool:
        cached_assoc = getattr(self, "_cached_assoc", None)
        cached_eta = getattr(self, "_cached_eta", None)
        cached_uav_pos = getattr(self, "_cached_eta_uav_pos", None)
        cached_gu_pos = getattr(self, "_cached_eta_gu_pos", None)
        cached_snapshot_t = getattr(self, "_cached_access_snapshot_t", None)
        target_snapshot_t = int(self.t if snapshot_step_t is None else snapshot_step_t)
        current_uav_pos = np.asarray(self.uav_pos if uav_pos is None else uav_pos, dtype=np.float32)
        current_gu_pos = np.asarray(self.gu_pos if gu_pos is None else gu_pos, dtype=np.float32)
        return bool(
            cached_assoc is not None
            and cached_eta is not None
            and cached_uav_pos is not None
            and cached_gu_pos is not None
            and cached_snapshot_t is not None
            and int(cached_snapshot_t) == target_snapshot_t
            and np.asarray(cached_assoc).shape == assoc.shape
            and np.array_equal(np.asarray(cached_assoc), assoc)
            and self._candidate_lists_equal(getattr(self, "_cached_candidates", None), candidates)
            and np.array_equal(np.asarray(cached_uav_pos), current_uav_pos)
            and np.array_equal(np.asarray(cached_gu_pos), current_gu_pos)
        )

    def _cached_access_snapshot_matches(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        *,
        snapshot_step_t: int | None = None,
        uav_pos: np.ndarray | None = None,
        gu_pos: np.ndarray | None = None,
    ) -> bool:
        return bool(
            getattr(self, "_cached_access_gain_matrix", None) is not None
            and self._cached_eta_matches(
                assoc,
                candidates,
                snapshot_step_t=snapshot_step_t,
                uav_pos=uav_pos,
                gu_pos=gu_pos,
            )
        )

    def _doppler_precomp_enabled(self) -> bool:
        mode = str(getattr(self.cfg, "doppler_precomp_mode", "none") or "none").strip().lower()
        return mode in {"residual_hz", "residual_ppm"}

    def _doppler_residual_cap_hz(self) -> float:
        cfg = self.cfg
        mode = str(getattr(cfg, "doppler_precomp_mode", "none") or "none").strip().lower()
        if mode == "residual_hz":
            return max(float(getattr(cfg, "doppler_residual_hz", 0.0) or 0.0), 0.0)
        if mode == "residual_ppm":
            ppm = max(float(getattr(cfg, "doppler_residual_ppm", 0.0) or 0.0), 0.0)
            return _backhaul_carrier_freq_from_cfg(cfg) * ppm * 1e-6
        return 0.0

    def _reset_doppler_residual_state(self) -> None:
        cfg = self.cfg
        self._doppler_residual_state_hz = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        if not self._doppler_precomp_enabled():
            return
        cap = self._doppler_residual_cap_hz()
        if cap <= 0.0:
            return
        sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
        sigma = min(sigma, cap)
        if sigma <= 0.0:
            return
        rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
        init_std = sigma / math.sqrt(max(1.0 - rho * rho, 1e-6))
        init = self.rng.normal(loc=0.0, scale=init_std, size=self._doppler_residual_state_hz.shape)
        self._doppler_residual_state_hz = np.clip(init, -cap, cap).astype(np.float32, copy=False)

    def _advance_doppler_residual_state(self) -> None:
        if not self._doppler_precomp_enabled():
            return
        cap = self._doppler_residual_cap_hz()
        if cap <= 0.0:
            self._doppler_residual_state_hz.fill(0.0)
            return
        cfg = self.cfg
        rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
        sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
        sigma = min(sigma, cap)
        if sigma <= 0.0:
            self._doppler_residual_state_hz = np.clip(self._doppler_residual_state_hz, -cap, cap).astype(
                np.float32,
                copy=False,
            )
            return
        noise = self.rng.normal(loc=0.0, scale=sigma, size=self._doppler_residual_state_hz.shape)
        next_state = (
            np.float32(rho) * self._doppler_residual_state_hz.astype(np.float32, copy=False)
            + np.asarray(noise, dtype=np.float32)
        )
        self._doppler_residual_state_hz = np.clip(next_state, -cap, cap).astype(np.float32, copy=False)

    def _compute_centroid_stats(self) -> Tuple[float, float]:
        cfg = self.cfg
        centroid_reward = 0.0
        centroid_dist_mean = 0.0
        if cfg.num_gu > 0:
            q_weights = self.gu_queue / normalize_scale(cfg.queue_max_gu)
            w_sum = float(np.sum(q_weights))
            if w_sum <= NORMALIZATION_DENOM_EPS:
                # Keep a dense navigation signal even when all queues are empty.
                weights = np.full((cfg.num_gu,), 1.0 / max(cfg.num_gu, 1), dtype=np.float32)
            else:
                weights = (q_weights / w_sum).astype(np.float32, copy=False)
            centroid = np.sum(self.gu_pos * weights[:, None], axis=0)
            dists = np.linalg.norm(self.uav_pos - centroid[None, :], axis=1)
            centroid_dist_mean = float(np.mean(dists)) if dists.size else 0.0
            scale = max(float(getattr(cfg, "centroid_dist_scale", 1.0) or 1.0), 1e-6)
            centroid_reward = float(np.mean(np.exp(-dists / scale))) if dists.size else 0.0
        return centroid_reward, centroid_dist_mean

    def _assoc_centroid_summary(
        self,
        assoc: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        cfg = self.cfg
        assoc_arr = None if assoc is None else np.asarray(assoc, dtype=np.int32)
        if assoc_arr is None or assoc_arr.shape != (cfg.num_gu,):
            cached_assoc = getattr(self, "_cached_assoc", None)
            assoc_arr = np.asarray(cached_assoc, dtype=np.int32) if cached_assoc is not None else None
        if assoc_arr is None or assoc_arr.shape != (cfg.num_gu,):
            assoc_arr = self._associate_users()

        counts = np.zeros((cfg.num_uav,), dtype=np.float32)
        rel_centroids = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        dist_norms = np.zeros((cfg.num_uav,), dtype=np.float32)
        map_scale = normalize_scale(float(cfg.map_size))
        for u in range(cfg.num_uav):
            idx = np.nonzero(assoc_arr == u)[0]
            if idx.size <= 0:
                continue
            counts[u] = float(idx.size)
            centroid = np.mean(self.gu_pos[idx], axis=0)
            rel = (centroid - self.uav_pos[u]) / map_scale
            rel_centroids[u] = rel.astype(np.float32, copy=False)
            dist_norms[u] = float(np.linalg.norm(centroid - self.uav_pos[u]) / map_scale)

        valid_mask = counts > 0.0
        valid_count = float(np.count_nonzero(valid_mask))
        valid_frac = valid_count / max(float(cfg.num_uav), 1.0)
        mean_dist_norm = float(np.mean(dist_norms[valid_mask])) if np.any(valid_mask) else 0.0
        return counts, rel_centroids, dist_norms, valid_count, valid_frac, mean_dist_norm

    def _sat_overlap_summary(self) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        if cfg.num_uav <= 1:
            return np.zeros((cfg.num_uav,), dtype=np.float32), 0.0
        selections = getattr(self, "last_sat_selection", None)
        if not isinstance(selections, list) or len(selections) != cfg.num_uav:
            return np.zeros((cfg.num_uav,), dtype=np.float32), 0.0
        denom = max(float(cfg.num_uav - 1), 1.0)
        counts = np.zeros((cfg.num_sat,), dtype=np.float32)
        unique_selections: List[np.ndarray] = []
        for sel in selections:
            sel_arr = np.asarray(sel, dtype=np.int32).reshape(-1)
            if sel_arr.size <= 0:
                unique_selections.append(np.zeros((0,), dtype=np.int32))
                continue
            sel_unique = np.unique(sel_arr[(sel_arr >= 0) & (sel_arr < cfg.num_sat)])
            unique_selections.append(sel_unique.astype(np.int32, copy=False))
            if sel_unique.size > 0:
                counts[sel_unique] += 1.0
        overlap_u = np.zeros((cfg.num_uav,), dtype=np.float32)
        for u, sel_arr in enumerate(unique_selections):
            if sel_arr.size <= 0:
                continue
            overlap_u[u] = float(np.mean((counts[sel_arr] - 1.0) / denom))
        overlap_mean = float(np.mean(overlap_u)) if overlap_u.size > 0 else 0.0
        return overlap_u.astype(np.float32, copy=False), overlap_mean

    def _compute_sat_overlap_eval(self) -> float:
        _, overlap_mean = self._sat_overlap_summary()
        return overlap_mean

    def _queue_arrival_scale(self, arrival_sum: float) -> float:
        cfg = self.cfg
        queue_norm_k = normalize_scale(float(getattr(cfg, "queue_norm_K", 1.0) or 1.0))
        arrival_floor = float(getattr(cfg, "queue_norm_arrival_floor", 0.0) or 0.0)
        if arrival_floor <= 0.0:
            arrival_floor = (
                float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
                * float(cfg.num_gu)
                * float(cfg.tau0)
            )
        arrival_ref = max(float(arrival_sum), arrival_floor)
        arrival_ref = reward_ratio_denominator_scalar(arrival_ref, name="queue arrival normalization reference")
        return queue_norm_k * arrival_ref

    def _centroid_anneal_state(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        eta_start = float(getattr(cfg, "eta_centroid", 0.0) or 0.0)
        eta_final_cfg = getattr(cfg, "eta_centroid_final", None)
        eta_current = eta_start
        decay_steps = int(getattr(cfg, "eta_centroid_decay_steps", 0) or 0)
        if eta_final_cfg is not None and decay_steps > 0:
            progress = min(1.0, float(self.global_step) / float(max(decay_steps, 1)))
            eta_final = float(eta_final_cfg)
            eta_current = eta_start + (eta_final - eta_start) * progress
        decayed = max(eta_start - eta_current, 0.0)
        if eta_start > NORMALIZATION_DENOM_EPS:
            transfer_ratio = float(np.clip(decayed / eta_start, 0.0, 1.0))
        else:
            transfer_ratio = 0.0
        return eta_start, eta_current, transfer_ratio

    def _update_adaptive_avoidance_after_episode(self) -> None:
        cfg = self.cfg
        adaptive_enabled = bool(getattr(cfg, "avoidance_adaptive_enabled", False))
        eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
        eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
        eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
        eta_max = max(eta_min, eta_max)

        if not adaptive_enabled:
            self.avoidance_eta_eff = float(np.clip(float(cfg.avoidance_eta), eta_min, eta_max))
            return

        prev_steps = int(getattr(self, "_episode_step_count", 0))
        prev_collisions = int(getattr(self, "_episode_collision_count", 0))
        if prev_steps > 0:
            prev_rate = float(prev_collisions) / float(prev_steps)
            self.prev_episode_collision_rate = prev_rate
            beta = float(getattr(cfg, "avoidance_adaptive_ema_beta", 0.9) or 0.9)
            beta = float(np.clip(beta, 0.0, 0.9999))
            self.avoidance_collision_rate_ema = (
                beta * float(getattr(self, "avoidance_collision_rate_ema", 0.0))
                + (1.0 - beta) * prev_rate
            )
            target = float(getattr(cfg, "avoidance_collision_target", 0.05) or 0.05)
            gain = float(getattr(cfg, "avoidance_adaptive_gain", 1.0) or 1.0)
            eta_cur = float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta))
            eta_next = eta_cur + gain * (self.avoidance_collision_rate_ema - target) * float(cfg.a_max)
            self.avoidance_eta_eff = float(np.clip(eta_next, eta_min, eta_max))
        else:
            self.avoidance_eta_eff = float(
                np.clip(float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta)), eta_min, eta_max)
            )

    def _traffic_level_ratio(self) -> float:
        cfg = self.cfg
        raw_level = getattr(cfg, "traffic_level", 2)
        level = int(2 if raw_level is None else raw_level)
        level = int(np.clip(level, 0, 2))
        if level == 0:
            ratio = float(getattr(cfg, "traffic_level_nav_ratio", 0.08) or 0.08)
        elif level == 1:
            ratio = float(getattr(cfg, "traffic_level_easy_ratio", 0.5) or 0.5)
        else:
            ratio = float(getattr(cfg, "traffic_level_hard_ratio", 1.0) or 1.0)
        self.traffic_level = level
        self.traffic_level_ratio = float(np.clip(ratio, 0.0, 1.0))
        return self.traffic_level_ratio

    def _set_effective_task_arrival_rate(self) -> None:
        cfg = self.cfg
        base_rate = max(float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0), 0.0)
        ratio = self._traffic_level_ratio()
        self.effective_task_arrival_rate = base_rate * ratio
        # num_gu=0 is valid for satellite/link-only tests. The reward denominator
        # still needs a positive traffic scale, so use one nominal GU's arrival
        # when there are no GU entities rather than accepting an objective-scale 0.
        traffic_entity_count = max(float(cfg.num_gu), 1.0)
        self.arrival_ref_bits_per_step = reward_ratio_denominator_scalar(
            float(self.effective_task_arrival_rate) * traffic_entity_count * float(cfg.tau0),
            name="arrival_ref_bits_per_step",
        )

    def _arrival_ref(self) -> float:
        return reward_ratio_denominator_scalar(
            float(getattr(self, "arrival_ref_bits_per_step", 0.0) or 0.0),
            name="arrival_ref_bits_per_step",
        )

    def _bw_weighted_workload_sat_active_ref_count(self) -> float:
        cfg = self.cfg
        active_count = getattr(cfg, "queue_ref_sat_active_count", None)
        if active_count is not None:
            return max(float(active_count), 1.0)
        sat_k = max(int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", cfg.N_RF)) or cfg.N_RF), 0)
        if sat_k > 0 and cfg.num_uav > 0:
            return max(min(float(cfg.num_sat), float(sat_k * cfg.num_uav)), 1.0)
        return max(float(cfg.num_sat), 1.0)

    def _bw_weighted_workload_eps(self) -> float:
        cfg = self.cfg
        floor = getattr(cfg, "service_floor_bits_per_step", None)
        if floor is None:
            floor = getattr(cfg, "bw_weighted_workload_eps", 1.0)
        return max(float(floor or 0.0), float(NORMALIZATION_DENOM_EPS))

    def _bw_weighted_workload_device_ema_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg

        def _coerce_vec(attr_vec: str, attr_scalar: str, size: int, default_value: float) -> np.ndarray:
            if size <= 0:
                return np.zeros((0,), dtype=np.float32)
            vec = getattr(self, attr_vec, None)
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.shape == (size,):
                    return arr.astype(np.float32, copy=False)
            scalar = getattr(self, attr_scalar, None)
            if scalar is not None:
                per_entity = float(scalar) / max(float(size), 1.0)
                return np.full((size,), per_entity, dtype=np.float32)
            return np.full((size,), default_value, dtype=np.float32)

        arrival_ref = float(self._arrival_ref())
        gu_default = arrival_ref / max(float(cfg.num_gu), 1.0)
        uav_default = arrival_ref / max(float(cfg.num_uav), 1.0)
        sat_default = arrival_ref / max(float(self._bw_weighted_workload_sat_active_ref_count()), 1.0)
        gu_ema = _coerce_vec(
            "bw_weighted_workload_acc_ema_vec",
            "bw_weighted_workload_acc_ema",
            int(cfg.num_gu),
            gu_default,
        )
        uav_ema = _coerce_vec(
            "bw_weighted_workload_rel_ema_vec",
            "bw_weighted_workload_rel_ema",
            int(cfg.num_uav),
            uav_default,
        )
        sat_ema = _coerce_vec(
            "bw_weighted_workload_sat_ema_vec",
            "bw_weighted_workload_sat_ema",
            int(cfg.num_sat),
            sat_default,
        )
        return gu_ema, uav_ema, sat_ema

    def _bw_weighted_workload_device_costs(
        self,
        *,
        assoc_override: np.ndarray | None = None,
        sat_selection_override: List[List[int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        eps = self._bw_weighted_workload_eps()
        gu_ema, uav_ema, sat_ema = self._bw_weighted_workload_device_ema_vectors()

        sat_cost = (1.0 / np.maximum(sat_ema, eps)).astype(np.float32, copy=False)
        sat_cost_fallback = float(np.mean(sat_cost)) if sat_cost.size > 0 else 0.0

        uav_downstream_cost = np.full((cfg.num_uav,), sat_cost_fallback, dtype=np.float32)
        sat_selection = sat_selection_override if sat_selection_override is not None else self.last_sat_selection
        if isinstance(sat_selection, list):
            selection_matrix = self._sat_selection_matrix(sat_selection)
            for u in range(min(int(selection_matrix.shape[0]), cfg.num_uav)):
                selected = np.asarray(selection_matrix[u], dtype=np.int64).reshape(-1)
                valid = selected[(selected >= 0) & (selected < cfg.num_sat)]
                if valid.size > 0:
                    uav_downstream_cost[u] = float(np.mean(sat_cost[valid]))
        uav_cost = (1.0 / np.maximum(uav_ema, eps) + uav_downstream_cost).astype(np.float32, copy=False)
        uav_cost_fallback = float(np.mean(uav_cost)) if uav_cost.size > 0 else 0.0

        assoc_source = (
            assoc_override
            if assoc_override is not None
            else getattr(self, "last_association", np.full((cfg.num_gu,), -1, dtype=np.int32))
        )
        assoc = np.asarray(assoc_source, dtype=np.int32).reshape(-1)
        gu_downstream_cost = np.full((cfg.num_gu,), uav_cost_fallback, dtype=np.float32)
        if assoc.shape == (cfg.num_gu,) and cfg.num_uav > 0:
            valid_assoc = (assoc >= 0) & (assoc < cfg.num_uav)
            if np.any(valid_assoc):
                gu_downstream_cost[valid_assoc] = uav_cost[assoc[valid_assoc]]
        gu_cost = (1.0 / np.maximum(gu_ema, eps) + gu_downstream_cost).astype(np.float32, copy=False)
        return gu_cost, uav_cost, sat_cost

    @staticmethod
    def _bw_weighted_workload_total(
        *,
        gu_cost: np.ndarray,
        uav_cost: np.ndarray,
        sat_cost: np.ndarray,
        gu_queue: np.ndarray,
        uav_queue: np.ndarray,
        sat_queue: np.ndarray,
    ) -> float:
        total = 0.0
        if gu_cost.size > 0:
            total += float(
                np.sum(
                    gu_cost.astype(np.float32, copy=False) * np.asarray(gu_queue, dtype=np.float32),
                    dtype=np.float32,
                )
            )
        if uav_cost.size > 0:
            total += float(
                np.sum(
                    uav_cost.astype(np.float32, copy=False) * np.asarray(uav_queue, dtype=np.float32),
                    dtype=np.float32,
                )
            )
        if sat_cost.size > 0:
            total += float(
                np.sum(
                    sat_cost.astype(np.float32, copy=False) * np.asarray(sat_queue, dtype=np.float32),
                    dtype=np.float32,
                )
            )
        return float(total)

    def _update_bw_weighted_workload_service_ema(
        self,
        attr_vec: str,
        attr_sum: str,
        values: np.ndarray | float,
        size: int,
    ) -> None:
        if size <= 0:
            setattr(self, attr_vec, np.zeros((0,), dtype=np.float32))
            setattr(self, attr_sum, 0.0)
            return
        prev = np.asarray(getattr(self, attr_vec, np.zeros((size,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if prev.shape != (size,):
            scalar = float(getattr(self, attr_sum, 0.0) or 0.0)
            prev = np.full((size,), scalar / max(float(size), 1.0), dtype=np.float32)
        cur = np.asarray(values, dtype=np.float32)
        if cur.shape == ():
            cur = np.full((size,), float(cur), dtype=np.float32)
        else:
            cur = cur.reshape(-1).astype(np.float32, copy=False)
        if cur.shape != (size,):
            raise ValueError(f"{attr_vec} EMA update expects shape ({size},), got {tuple(cur.shape)}.")
        keep = float(np.clip(getattr(self.cfg, "bw_weighted_workload_ema_decay", 0.95), 0.0, 1.0))
        updated = (keep * prev + (1.0 - keep) * np.maximum(cur, 0.0)).astype(np.float32, copy=False)
        setattr(self, attr_vec, updated)
        setattr(self, attr_sum, float(np.sum(updated, dtype=np.float32)))

    def _reward_weighted_workload_delta(
        self,
        *,
        gu_queue_before: np.ndarray,
        uav_queue_before: np.ndarray,
        sat_queue_before: np.ndarray,
        realized_arrival: np.ndarray,
    ) -> float:
        gu_cost, uav_cost, sat_cost = self._bw_weighted_workload_device_costs()
        q_gu_before_service = np.asarray(gu_queue_before, dtype=np.float32) + np.asarray(realized_arrival, dtype=np.float32)
        workload_before = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=q_gu_before_service,
            uav_queue=np.asarray(uav_queue_before, dtype=np.float32),
            sat_queue=np.asarray(sat_queue_before, dtype=np.float32),
        )
        workload_after = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_queue, dtype=np.float32),
            uav_queue=np.asarray(self.uav_queue, dtype=np.float32),
            sat_queue=np.asarray(self.sat_queue, dtype=np.float32),
        )
        drop_cost = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_drop, dtype=np.float32),
            uav_queue=np.asarray(self.uav_drop, dtype=np.float32),
            sat_queue=np.asarray(self.sat_drop, dtype=np.float32),
        )
        return float(-(workload_after - workload_before) - drop_cost)

    def _reward_weighted_workload_level(self) -> float:
        gu_cost, uav_cost, sat_cost = self._bw_weighted_workload_device_costs()
        workload_after = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_queue, dtype=np.float32),
            uav_queue=np.asarray(self.uav_queue, dtype=np.float32),
            sat_queue=np.asarray(self.sat_queue, dtype=np.float32),
        )
        drop_cost = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_drop, dtype=np.float32),
            uav_queue=np.asarray(self.uav_drop, dtype=np.float32),
            sat_queue=np.asarray(self.sat_drop, dtype=np.float32),
        )
        return float(-workload_after - drop_cost)

    def _reward_positive_weighted_workload_level(self) -> float:
        gu_cost, uav_cost, sat_cost = self._bw_weighted_workload_device_costs()
        workload_after = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_queue, dtype=np.float32),
            uav_queue=np.asarray(self.uav_queue, dtype=np.float32),
            sat_queue=np.asarray(self.sat_queue, dtype=np.float32),
        )
        drop_cost = self._bw_weighted_workload_total(
            gu_cost=gu_cost,
            uav_cost=uav_cost,
            sat_cost=sat_cost,
            gu_queue=np.asarray(self.gu_drop, dtype=np.float32),
            uav_queue=np.asarray(self.uav_drop, dtype=np.float32),
            sat_queue=np.asarray(self.sat_drop, dtype=np.float32),
        )
        workload = max(float(workload_after + drop_cost), 0.0)
        return float(1.0 / (1.0 + np.log1p(workload)))

    def _reward_gu_queue_level(self) -> float:
        arrival_ref = self._arrival_ref()
        gu_backlog_after = float(np.sum(np.asarray(self.gu_queue, dtype=np.float32)))
        gu_drop = float(np.sum(np.asarray(self.gu_drop, dtype=np.float32)))
        return float(-(gu_backlog_after + gu_drop) / arrival_ref)

    def _reward_system_queue_level(self) -> float:
        arrival_ref = self._arrival_ref()
        q_total_after = (
            float(np.sum(np.asarray(self.gu_queue, dtype=np.float32)))
            + float(np.sum(np.asarray(self.uav_queue, dtype=np.float32)))
            + float(np.sum(np.asarray(self.sat_queue, dtype=np.float32)))
        )
        drop_total = (
            float(np.sum(np.asarray(self.gu_drop, dtype=np.float32)))
            + float(np.sum(np.asarray(self.uav_drop, dtype=np.float32)))
            + float(np.sum(np.asarray(self.sat_drop, dtype=np.float32)))
        )
        return float(-(q_total_after + drop_total) / arrival_ref)

    def _reward_gu_service_queue(self) -> float:
        arrival_ref = self._arrival_ref()
        gu_backlog_after = float(np.sum(np.asarray(self.gu_queue, dtype=np.float32)))
        gu_drop = float(np.sum(np.asarray(self.gu_drop, dtype=np.float32)))
        gu_outflow = float(np.sum(np.asarray(self.last_gu_outflow, dtype=np.float32)))
        return float((gu_outflow - gu_backlog_after - gu_drop) / arrival_ref)

    def _traffic_model(self) -> str:
        return str(getattr(self.cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()

    def _effective_b_backhaul_per_sat(self) -> float:
        legacy_bw = getattr(self.cfg, "b_sat_total", None)
        legacy_scale = getattr(self.cfg, "b_sat_total_scale", None)
        bandwidth = legacy_bw if legacy_bw is not None else getattr(self.cfg, "b_backhaul_per_sat", 0.0)
        scale_raw = legacy_scale if legacy_scale is not None else getattr(self.cfg, "b_backhaul_per_sat_scale", 1.0)
        scale = max(float(scale_raw or 1.0), 0.0)
        return float(bandwidth) * scale

    def _effective_b_sat_total(self) -> float:
        """Legacy alias for the per-SAT backhaul bandwidth pool."""
        return self._effective_b_backhaul_per_sat()

    def _effective_sat_cpu_freq(self) -> float:
        scale = max(float(getattr(self.cfg, "sat_cpu_freq_scale", 1.0) or 1.0), 0.0)
        return float(self.cfg.sat_cpu_freq) * scale

    def _current_task_arrival_rates(self, arrival_rate: float) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            return np.zeros((0,), dtype=np.float32)
        base_rate = max(float(arrival_rate), 0.0)
        model = self._traffic_model()
        if model != "sticky_subset_hotspot":
            return np.full((cfg.num_gu,), base_rate, dtype=np.float32)

        base_scale = np.asarray(
            getattr(self, "_arrival_base_scale", np.ones((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        if base_scale.shape != (cfg.num_gu,):
            base_scale = np.ones((cfg.num_gu,), dtype=np.float32)
        weights = base_scale.copy()
        hotspot_idx = int(getattr(self, "_hotspot_active_idx", -1))
        hotspot_mask = np.asarray(getattr(self, "_hotspot_member_mask", np.zeros((0, cfg.num_gu), dtype=bool)))
        if hotspot_idx >= 0 and hotspot_idx < hotspot_mask.shape[0]:
            rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
            if rho > 0.0:
                weights *= np.where(hotspot_mask[hotspot_idx], rho, 1.0)
        if bool(getattr(cfg, "arrival_mean_preserve", True)):
            mean_weight = float(np.mean(weights))
            if mean_weight > NORMALIZATION_DENOM_EPS:
                weights /= mean_weight
        rates = np.maximum(base_rate * weights, 0.0)
        return np.asarray(rates, dtype=np.float32)

    def _current_arrival_base_rate(self) -> float:
        cfg = self.cfg
        arrival_rate = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
        ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
        legacy_arrival_ramp = ramp_steps > 0
        use_arrival_ramp = ablation_flag(cfg, "use_arrival_ramp", default=False) or legacy_arrival_ramp
        if use_arrival_ramp and ramp_steps > 0:
            start = float(getattr(cfg, "arrival_ramp_start", 0.0) or 0.0)
            start = float(np.clip(start, 0.0, 1.0))
            use_global = bool(getattr(cfg, "arrival_ramp_use_global", False))
            t_ref = self.global_step if use_global else self.t
            progress = min(1.0, float(t_ref) / max(ramp_steps, 1))
            arrival_rate = arrival_rate * (start + (1.0 - start) * progress)
        return float(max(arrival_rate, 0.0))

    def _current_expected_gu_arrival_rates(self) -> np.ndarray:
        """Expected exogenous GU arrivals for the current decision step."""
        return self._current_task_arrival_rates(self._current_arrival_base_rate())

    def _advance_traffic_model_state(self) -> None:
        if self._traffic_model() != "sticky_subset_hotspot":
            return
        hotspot_mask = np.asarray(getattr(self, "_hotspot_member_mask", np.zeros((0, self.cfg.num_gu), dtype=bool)))
        num_subsets = int(hotspot_mask.shape[0])
        if num_subsets <= 0:
            self._hotspot_active_idx = -1
            self.last_hotspot_index = -1
            return
        on_mean = max(float(getattr(self.cfg, "hotspot_on_mean_steps", 15.0) or 15.0), 1.0)
        off_mean = max(float(getattr(self.cfg, "hotspot_off_mean_steps", 8.0) or 8.0), 1.0)
        hotspot_idx = int(getattr(self, "_hotspot_active_idx", -1))
        if hotspot_idx >= 0:
            if float(self.rng.random()) < (1.0 / on_mean):
                hotspot_idx = -1
        else:
            if float(self.rng.random()) < (1.0 / off_mean):
                hotspot_idx = int(self.rng.integers(num_subsets))
        self._hotspot_active_idx = hotspot_idx
        self.last_hotspot_index = hotspot_idx

    def _apply_traffic_model_state_override(self, state_override: dict | None) -> None:
        if state_override is None:
            return
        cfg = self.cfg
        hotspot_idx = int(state_override.get("_hotspot_active_idx", state_override.get("last_hotspot_index", -1)))
        self._hotspot_active_idx = hotspot_idx
        self.last_hotspot_index = int(state_override.get("last_hotspot_index", hotspot_idx))
        mask_value = state_override.get("last_hotspot_mask", None)
        if mask_value is None:
            hotspot_mask = np.asarray(getattr(self, "_hotspot_member_mask", np.zeros((0, cfg.num_gu), dtype=bool)))
            if 0 <= hotspot_idx < hotspot_mask.shape[0]:
                mask = hotspot_mask[hotspot_idx].astype(np.float32, copy=False)
            else:
                mask = np.zeros((cfg.num_gu,), dtype=np.float32)
        else:
            mask = np.asarray(mask_value, dtype=np.float32)
            if mask.shape != (cfg.num_gu,):
                raise ValueError(f"last_hotspot_mask override shape must be ({cfg.num_gu},), got {tuple(mask.shape)}.")
        self.last_hotspot_mask = mask.astype(np.float32, copy=True)

    def _build_hotspot_subsets(self, assoc: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        target_count = max(int(getattr(cfg, "hotspot_num_subsets", 0) or 0), 0)
        subset_size_cfg = max(int(getattr(cfg, "hotspot_subset_size", 0) or 0), 0)
        if target_count <= 0 or subset_size_cfg <= 0 or cfg.num_gu <= 0:
            return []

        gu_pos = np.asarray(self.gu_pos, dtype=np.float32)

        def _candidate_records(indices: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
            idx = np.asarray(indices, dtype=np.int32)
            if idx.size <= 0:
                return []
            local_size = min(subset_size_cfg, int(idx.size))
            if local_size <= 0:
                return []
            pos = gu_pos[idx]
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            records: list[tuple[tuple[int, ...], float]] = []
            seen: set[tuple[int, ...]] = set()
            for row in range(idx.size):
                order = np.argsort(dist[row], kind="stable")[:local_size]
                subset = tuple(sorted(int(x) for x in idx[order].tolist()))
                if subset in seen:
                    continue
                seen.add(subset)
                local_dist = dist[np.ix_(order, order)]
                score = float(np.mean(local_dist))
                records.append((subset, score))
            return records

        records: list[tuple[tuple[int, ...], float]] = []
        for u in range(cfg.num_uav):
            group = np.flatnonzero(np.asarray(assoc, dtype=np.int32) == int(u))
            records.extend(_candidate_records(group))
        if len(records) < target_count:
            records.extend(_candidate_records(np.arange(cfg.num_gu, dtype=np.int32)))

        dedup: dict[tuple[int, ...], float] = {}
        for subset, score in records:
            prev = dedup.get(subset)
            dedup[subset] = score if prev is None else min(prev, score)

        candidates = [
            (np.asarray(subset, dtype=np.int32), float(score))
            for subset, score in dedup.items()
            if len(subset) > 0
        ]
        candidates.sort(key=lambda item: (-int(item[0].size), item[1], tuple(int(x) for x in item[0].tolist())))
        if not candidates:
            return []

        usage = np.zeros((cfg.num_gu,), dtype=np.int32)
        selected: list[np.ndarray] = []
        remaining = list(candidates)
        while remaining and len(selected) < target_count:
            best_idx = min(
                range(len(remaining)),
                key=lambda idx: (
                    float(remaining[idx][1]),
                    int(np.sum(usage[remaining[idx][0]])),
                    tuple(int(x) for x in remaining[idx][0].tolist()),
                ),
            )
            subset, _ = remaining.pop(best_idx)
            selected.append(subset)
            usage[subset] += 1
        return selected

    def _build_bw_valid_mask(self, assoc: np.ndarray, candidates: List[List[int]]) -> np.ndarray:
        cfg = self.cfg
        mask = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        for u in range(cfg.num_uav):
            cand = candidates[u]
            if not cand:
                continue
            cand_idx = np.asarray(cand[: cfg.users_obs_max], dtype=np.int32)
            assoc_mask = (assoc[cand_idx] == u).astype(np.float32, copy=False)
            mask[u, : assoc_mask.shape[0]] = assoc_mask
        return mask

    def _uav_init_boundary_margin(self) -> float:
        cfg = self.cfg
        steps = max(float(getattr(cfg, "uav_init_boundary_margin_steps", 0.0) or 0.0), 0.0)
        margin = steps * float(cfg.v_max) * float(cfg.tau0)
        max_margin = max(0.0, 0.5 * float(cfg.map_size) - 1e-6)
        return float(min(margin, max_margin))

    def _sample_uav_safe_random_positions(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_uav <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        margin = self._uav_init_boundary_margin()
        low = float(margin)
        high = float(cfg.map_size) - float(margin)
        min_spacing_cfg = getattr(cfg, "uav_init_min_spacing", None)
        min_spacing = float(cfg.d_safe) if min_spacing_cfg is None else max(float(min_spacing_cfg), 0.0)
        max_tries = max(int(getattr(cfg, "uav_init_max_tries", 256) or 256), 1)

        positions = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        for i in range(cfg.num_uav):
            placed = False
            for _ in range(max_tries):
                candidate = self.rng.uniform(low, high, size=(2,)).astype(np.float32)
                if i > 0:
                    dist = np.linalg.norm(positions[:i] - candidate[None, :], axis=1)
                    if not np.all(dist >= min_spacing - 1e-6):
                        continue
                positions[i] = candidate
                placed = True
                break
            if not placed:
                raise RuntimeError(
                    "Could not sample UAV initial positions satisfying boundary margin "
                    "and minimum spacing constraints."
                )
        return positions

    def _sample_uav_initial_velocities(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_uav <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if not bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
            return np.zeros((cfg.num_uav, 2), dtype=np.float32)

        speed_frac = max(float(getattr(cfg, "uav_init_speed_frac", 0.0) or 0.0), 0.0)
        speed = min(speed_frac, 1.0) * float(cfg.v_max)
        if speed <= 0.0:
            return np.zeros((cfg.num_uav, 2), dtype=np.float32)

        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=(cfg.num_uav,))
        vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speed
        return vel.astype(np.float32, copy=False)

    def _sample_uav_positions(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_uav <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        spawn_mode = str(getattr(cfg, "uav_spawn_mode", "default") or "default").strip().lower()
        if spawn_mode == "gu_centroid":
            if cfg.num_gu > 0 and getattr(self, "gu_pos", None) is not None:
                center = np.mean(np.asarray(self.gu_pos, dtype=np.float32), axis=0).astype(np.float32, copy=False)
            else:
                center = np.array([cfg.map_size * 0.5, cfg.map_size * 0.5], dtype=np.float32)
            positions = np.repeat(center[None, :], cfg.num_uav, axis=0).astype(np.float32, copy=False)
            return np.clip(positions, 0.0, cfg.map_size)
        if spawn_mode == "gu_cluster_centers":
            centers = np.asarray(getattr(self, "gu_cluster_centers", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            counts = np.asarray(getattr(self, "gu_cluster_counts", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] <= 0:
                if cfg.num_gu > 0 and getattr(self, "gu_pos", None) is not None:
                    center = np.mean(np.asarray(self.gu_pos, dtype=np.float32), axis=0).astype(np.float32, copy=False)
                else:
                    center = np.array([cfg.map_size * 0.5, cfg.map_size * 0.5], dtype=np.float32)
                positions = np.repeat(center[None, :], cfg.num_uav, axis=0).astype(np.float32, copy=False)
                return np.clip(positions, 0.0, cfg.map_size)
            if counts.shape[0] != centers.shape[0]:
                order = np.arange(centers.shape[0], dtype=np.int64)
            else:
                order = np.argsort(-counts.astype(np.float32, copy=False), kind="stable")
            selected = centers[order[np.arange(cfg.num_uav) % max(order.size, 1)]].astype(np.float32, copy=False)
            return np.clip(selected, 0.0, cfg.map_size)
        if bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
            return self._sample_uav_safe_random_positions()
        use_curriculum_spawn = ablation_flag(
            cfg,
            "use_curriculum_spawn",
            fallback_attr="uav_spawn_curriculum_enabled",
            default=False,
        )
        if not use_curriculum_spawn:
            return self.rng.uniform(0.0, cfg.map_size, size=(cfg.num_uav, 2)).astype(np.float32)

        steps = int(getattr(cfg, "uav_spawn_curriculum_steps", 0) or 0)
        progress = 1.0 if steps <= 0 else min(1.0, float(max(self.episode_idx - 1, 0)) / steps)
        if progress >= 1.0 and getattr(cfg, "uav_spawn_full_random_final", True):
            return self.rng.uniform(0.0, cfg.map_size, size=(cfg.num_uav, 2)).astype(np.float32)

        radius_start = max(float(getattr(cfg, "uav_spawn_radius_start", 0.0) or 0.0), 0.0)
        radius_end = getattr(cfg, "uav_spawn_radius_end", None)
        if radius_end is None:
            radius_end = cfg.map_size * 0.5
        radius_end = max(float(radius_end), radius_start)
        radius = radius_start + (radius_end - radius_start) * progress

        if cfg.num_gu > 0:
            center = self.gu_pos[self.rng.integers(0, cfg.num_gu)].astype(np.float32, copy=False)
        else:
            center = np.array([cfg.map_size * 0.5, cfg.map_size * 0.5], dtype=np.float32)

        positions = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        for i in range(cfg.num_uav):
            pos = None
            for _ in range(20):
                ang = self.rng.uniform(0.0, 2.0 * math.pi)
                r = radius * math.sqrt(self.rng.uniform(0.0, 1.0))
                candidate = center + np.array([math.cos(ang) * r, math.sin(ang) * r], dtype=np.float32)
                if 0.0 <= candidate[0] <= cfg.map_size and 0.0 <= candidate[1] <= cfg.map_size:
                    pos = candidate
                    break
            if pos is None:
                if candidate is None:
                    candidate = center
                pos = np.clip(candidate, 0.0, cfg.map_size)
            positions[i] = pos
        return positions

    def _build_spaces(self) -> None:
        cfg = self.cfg
        obs_space = {
            "own": gym.spaces.Box(-np.inf, np.inf, shape=(self.own_dim,), dtype=np.float32),
            "users": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.users_obs_max, self.user_dim), dtype=np.float32),
            "users_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.users_obs_max,), dtype=np.float32),
            "bw_valid_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.users_obs_max,), dtype=np.float32),
            "candidate_indices": gym.spaces.Box(-1, cfg.num_gu, shape=(cfg.users_obs_max,), dtype=np.int64),
            "sats": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.sats_obs_max, self.sat_dim), dtype=np.float32),
            "sats_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.sats_obs_max,), dtype=np.float32),
            "sat_valid_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.sats_obs_max,), dtype=np.float32),
            "nbrs": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32),
            "nbrs_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.nbrs_obs_max,), dtype=np.float32),
        }
        if bool(getattr(cfg, "danger_nbr_enabled", False)):
            obs_space["danger_nbr"] = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.danger_nbr_dim,),
                dtype=np.float32,
            )
        self._obs_space = gym.spaces.Dict(obs_space)

        self._act_space = gym.spaces.Dict(
            {
                "accel": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "bw_alloc": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(cfg.num_gu,),
                    dtype=np.float32,
                ),
                "sat_select_mask": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(cfg.sats_obs_max,),
                    dtype=np.float32,
                ),
            }
        )

    def _extra_user_feature_count(self) -> int:
        cfg = self.cfg
        return (
            int(bool(getattr(cfg, "obs_user_include_arrival_rate", False)))
            + int(bool(getattr(cfg, "obs_user_include_recent_arrival", False)))
            + int(bool(getattr(cfg, "obs_user_include_recent_service", False)))
            + int(bool(getattr(cfg, "obs_user_include_queue_headroom", False)))
            + int(bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)))
            + int(bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)))
            + int(bool(getattr(cfg, "obs_user_include_urgency_risk", False)))
            + int(bool(getattr(cfg, "obs_user_include_downstream_pressure", False)))
            + int(bool(getattr(cfg, "obs_user_include_service_gap", False)))
            + int(bool(getattr(cfg, "obs_user_include_service_gap_risk", False)))
            + int(bool(getattr(cfg, "obs_user_include_deadline_slack", False)))
            + int(bool(getattr(cfg, "obs_user_include_deadline_risk", False)))
        )

    def _extra_own_feature_count(self) -> int:
        cfg = self.cfg
        return int(bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False)))

    def _extra_sat_feature_count(self) -> int:
        cfg = self.cfg
        return int(bool(getattr(cfg, "obs_sat_include_sat_cost", False)))

    def _bw_weighted_workload_feature_refs(self) -> tuple[float, float, float, float]:
        cfg = self.cfg
        eps = self._bw_weighted_workload_eps()
        arrival_ref = float(self._arrival_ref())
        gu_default = arrival_ref / max(float(cfg.num_gu), 1.0)
        uav_default = arrival_ref / max(float(cfg.num_uav), 1.0)
        sat_active_ref = float(self._bw_weighted_workload_sat_active_ref_count())
        sat_default = arrival_ref / max(sat_active_ref, 1.0)

        sat_cost_ref = 1.0 / max(float(sat_default), eps)
        uav_cost_ref = 1.0 / max(float(uav_default), eps) + sat_cost_ref
        gu_local_cost_ref = 1.0 / max(float(gu_default), eps)
        gu_total_cost_ref = gu_local_cost_ref + uav_cost_ref
        return (
            float(gu_local_cost_ref),
            float(uav_cost_ref),
            float(sat_cost_ref),
            float(gu_total_cost_ref),
        )

    def _gu_reward_aligned_feature_dict(
        self,
        *,
        normalized: bool = True,
        assoc_override: np.ndarray | None = None,
        sat_selection_override: List[List[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            zeros = np.zeros((0,), dtype=np.float32)
            return {
                "local_gu_service_cost": zeros.copy(),
                "assoc_uav_cost": zeros.copy(),
                "assoc_sat_cost_mean": zeros.copy(),
                "weighted_queue_cost": zeros.copy(),
            }

        eps = self._bw_weighted_workload_eps()
        gu_ema, _, _ = self._bw_weighted_workload_device_ema_vectors()
        gu_cost, uav_cost, sat_cost = self._bw_weighted_workload_device_costs(
            assoc_override=assoc_override,
            sat_selection_override=sat_selection_override,
        )
        assoc_source = (
            assoc_override
            if assoc_override is not None
            else getattr(self, "last_association", np.full((cfg.num_gu,), -1, dtype=np.int32))
        )
        assoc = np.asarray(assoc_source, dtype=np.int32).reshape(-1)
        sat_selection = sat_selection_override if sat_selection_override is not None else getattr(self, "last_sat_selection", None)

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
        valid_assoc = (assoc >= 0) & (assoc < cfg.num_uav)
        if np.any(valid_assoc) and uav_cost.size > 0:
            assoc_uav_cost[valid_assoc] = uav_cost[assoc[valid_assoc]]
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
                assoc_sat_cost_mean[valid_assoc] = sat_mean_by_uav[assoc[valid_assoc]]

        local_gu_service_cost = (1.0 / np.maximum(gu_ema, eps)).astype(np.float32, copy=False)
        weighted_queue_cost = (
            np.asarray(gu_cost, dtype=np.float32) * np.asarray(self.gu_queue, dtype=np.float32)
        ).astype(np.float32, copy=False)

        raw = {
            "local_gu_service_cost": np.asarray(local_gu_service_cost, dtype=np.float32),
            "assoc_uav_cost": np.asarray(assoc_uav_cost, dtype=np.float32),
            "assoc_sat_cost_mean": np.asarray(assoc_sat_cost_mean, dtype=np.float32),
            "weighted_queue_cost": np.asarray(weighted_queue_cost, dtype=np.float32),
        }
        weighted_queue_ratio = (
            raw["weighted_queue_cost"]
            / normalize_scale(float(np.mean(raw["weighted_queue_cost"], dtype=np.float32)))
        ).astype(np.float32, copy=False)
        raw["weighted_queue_cost_relative"] = weighted_queue_ratio
        if not bool(normalized):
            return raw

        _, _, _, gu_total_cost_ref = self._bw_weighted_workload_feature_refs()
        weighted_queue_ref = normalize_scale(float(self._arrival_ref()) * float(gu_total_cost_ref))
        local_cost_mean = max(float(np.mean(raw["local_gu_service_cost"], dtype=np.float32)), LOG_RATIO_EPS)
        assoc_uav_cost_mean = max(float(np.mean(raw["assoc_uav_cost"], dtype=np.float32)), LOG_RATIO_EPS)
        assoc_sat_cost_mean_mean = max(float(np.mean(raw["assoc_sat_cost_mean"], dtype=np.float32)), LOG_RATIO_EPS)

        return {
            "local_gu_service_cost": np.log(
                log_ratio_argument(raw["local_gu_service_cost"]) / local_cost_mean
            ).astype(np.float32, copy=False),
            "assoc_uav_cost": np.log(
                log_ratio_argument(raw["assoc_uav_cost"]) / assoc_uav_cost_mean
            ).astype(np.float32, copy=False),
            "assoc_sat_cost_mean": np.log(
                log_ratio_argument(raw["assoc_sat_cost_mean"]) / assoc_sat_cost_mean_mean
            ).astype(np.float32, copy=False),
            "weighted_queue_cost": np.log1p(
                np.maximum(raw["weighted_queue_cost"], 0.0) / weighted_queue_ref
            ).astype(np.float32, copy=False),
            "weighted_queue_cost_relative": np.log(
                relative_log_argument(raw["weighted_queue_cost_relative"])
            ).astype(np.float32, copy=False),
        }

    def _uav_reward_aligned_feature_dict(
        self,
        *,
        normalized: bool = True,
        assoc_override: np.ndarray | None = None,
        sat_selection_override: List[List[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        if cfg.num_uav <= 0:
            zeros = np.zeros((0,), dtype=np.float32)
            return {"assoc_uav_cost": zeros.copy()}
        _, uav_cost, _ = self._bw_weighted_workload_device_costs(
            assoc_override=assoc_override,
            sat_selection_override=sat_selection_override,
        )
        raw = {"assoc_uav_cost": np.asarray(uav_cost, dtype=np.float32)}
        if not bool(normalized):
            return raw
        assoc_uav_cost_mean = max(float(np.mean(raw["assoc_uav_cost"], dtype=np.float32)), LOG_RATIO_EPS)
        return {
            "assoc_uav_cost": np.log(
                log_ratio_argument(raw["assoc_uav_cost"]) / assoc_uav_cost_mean
            ).astype(np.float32, copy=False)
        }

    def _sat_reward_aligned_feature_dict(
        self,
        *,
        normalized: bool = True,
        assoc_override: np.ndarray | None = None,
        sat_selection_override: List[List[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        if cfg.num_sat <= 0:
            zeros = np.zeros((0,), dtype=np.float32)
            return {"sat_cost": zeros.copy()}
        _, _, sat_cost = self._bw_weighted_workload_device_costs(
            assoc_override=assoc_override,
            sat_selection_override=sat_selection_override,
        )
        raw = {"sat_cost": np.asarray(sat_cost, dtype=np.float32)}
        if not bool(normalized):
            return raw
        sat_cost_mean = max(float(np.mean(raw["sat_cost"], dtype=np.float32)), LOG_RATIO_EPS)
        return {
            "sat_cost": np.log(
                log_ratio_argument(raw["sat_cost"]) / sat_cost_mean
            ).astype(np.float32, copy=False)
        }

    def _compute_overflow_risk_proxy(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            return np.zeros((0,), dtype=np.float32)
        next_arrival_rates = np.asarray(self._current_expected_gu_arrival_rates(), dtype=np.float32)
        queue_cap = normalize_scale(float(cfg.queue_max_gu))
        threshold_frac = float(np.clip(getattr(cfg, "overflow_risk_threshold_frac", 0.75) or 0.75, 0.0, 0.999))
        arrival_coef = max(float(getattr(cfg, "overflow_risk_arrival_coef", 1.0) or 0.0), 0.0)
        service_coef = max(float(getattr(cfg, "overflow_risk_service_coef", 0.1) or 0.0), 0.0)
        q_norm = np.asarray(self.gu_queue, dtype=np.float32) / queue_cap
        base_arrival = require_positive_float(
            float(np.mean(next_arrival_rates, dtype=np.float32)) * float(cfg.tau0),
            name="per-GU arrival reference bits per step",
        )
        arrival_norm = (next_arrival_rates * float(cfg.tau0)) / base_arrival
        service_norm = np.asarray(getattr(self, "last_gu_outflow", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32) / base_arrival
        projected_pressure = q_norm + arrival_coef * (arrival_norm - 1.0) - service_coef * service_norm
        denom = normalize_scale(1.0 - threshold_frac)
        risk = np.maximum(projected_pressure - threshold_frac, 0.0) / denom
        return np.clip(risk, 0.0, 1.0).astype(np.float32, copy=False)

    def _compute_downstream_pressure_proxy(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            return np.zeros((0,), dtype=np.float32)
        if cfg.num_uav > 0:
            uav_fill = float(
                np.mean(np.asarray(self.uav_queue, dtype=np.float32) / normalize_scale(float(cfg.queue_max_uav)))
            )
        else:
            uav_fill = 0.0
        active_sat = np.asarray(
            getattr(self, "last_sat_connection_counts", np.zeros((cfg.num_sat,), dtype=np.float32)),
            dtype=np.float32,
        ) > 0.0
        if cfg.num_sat > 0 and np.any(active_sat):
            sat_fill = float(
                np.mean(
                    np.asarray(self.sat_queue, dtype=np.float32)[active_sat] / normalize_scale(float(cfg.queue_max_sat))
                )
            )
        elif cfg.num_sat > 0:
            sat_fill = float(np.mean(np.asarray(self.sat_queue, dtype=np.float32) / normalize_scale(float(cfg.queue_max_sat))))
        else:
            sat_fill = 0.0
        pressure = float(np.clip(max(uav_fill, sat_fill), 0.0, 1.0))
        return np.full((cfg.num_gu,), pressure, dtype=np.float32)

    def _update_service_gap_state(self, q_before: np.ndarray, outflow: np.ndarray, q_after: np.ndarray) -> None:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            self.last_gu_service_gap = np.zeros((0,), dtype=np.float32)
            return
        prev_gap = np.asarray(
            getattr(self, "last_gu_service_gap", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        increment = max(float(getattr(cfg, "service_gap_increment", 1.0) or 0.0), 0.0)
        relief_coef = max(float(getattr(cfg, "service_gap_relief_coef", 0.5) or 0.0), 0.0)
        cap_steps = max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6)
        base_arrival = require_positive_float(
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate)) * float(cfg.tau0),
            name="per-GU arrival reference bits per step",
        )
        q_before_arr = np.asarray(q_before, dtype=np.float32)
        q_after_arr = np.asarray(q_after, dtype=np.float32)
        outflow_arr = np.asarray(outflow, dtype=np.float32)
        backlogged = (q_before_arr > 1.0e-6).astype(np.float32, copy=False)
        service_relief = relief_coef * (outflow_arr / base_arrival)
        gap = prev_gap + increment * backlogged - service_relief
        gap = np.clip(gap, 0.0, cap_steps).astype(np.float32, copy=False)
        gap[q_after_arr <= 1.0e-6] = 0.0
        self.last_gu_service_gap = gap.astype(np.float32, copy=False)

    def _sample_deadline_steps(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            return np.zeros((0,), dtype=np.float32)
        base_steps = max(float(getattr(cfg, "deadline_base_steps", 4.0) or 0.0), 1.0)
        jitter_steps = max(float(getattr(cfg, "deadline_jitter_steps", 0.0) or 0.0), 0.0)
        if jitter_steps <= NORMALIZATION_DENOM_EPS:
            return np.full((cfg.num_gu,), base_steps, dtype=np.float32)
        low = max(base_steps - jitter_steps, 1.0)
        high = max(base_steps + jitter_steps, low)
        return self.rng.uniform(low, high, size=(cfg.num_gu,)).astype(np.float32, copy=False)

    def _update_deadline_state_and_expire(
        self,
        q_before: np.ndarray,
        outflow: np.ndarray,
        q_after_service: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        if cfg.num_gu <= 0 or not bool(getattr(cfg, "deadline_enabled", False)):
            zeros = np.zeros((cfg.num_gu,), dtype=np.float32)
            self.last_gu_deadline_age = zeros.copy()
            self.last_gu_deadline_slack = zeros.copy()
            self.last_gu_deadline_risk = zeros.copy()
            self.gu_expired = zeros.copy()
            return np.asarray(q_after_service, dtype=np.float32), zeros

        prev_age = np.asarray(
            getattr(self, "last_gu_deadline_age", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        deadline_steps = np.asarray(
            getattr(self, "gu_deadline_steps", np.full((cfg.num_gu,), 4.0, dtype=np.float32)),
            dtype=np.float32,
        )
        q_before_arr = np.asarray(q_before, dtype=np.float32)
        outflow_arr = np.asarray(outflow, dtype=np.float32)
        q_after_service_arr = np.asarray(q_after_service, dtype=np.float32)
        backlog_mask = q_before_arr > 1.0e-6
        service_frac = np.zeros((cfg.num_gu,), dtype=np.float32)
        service_frac[backlog_mask] = outflow_arr[backlog_mask] / np.maximum(q_before_arr[backlog_mask], 1.0e-6)
        age_increment = max(float(getattr(cfg, "deadline_age_increment", 1.0) or 0.0), 0.0)
        relief_coef = max(float(getattr(cfg, "deadline_service_relief_coef", 0.75) or 0.0), 0.0)
        age_cap = max(float(getattr(cfg, "deadline_age_cap_steps", 8.0) or 0.0), 1.0)
        age = prev_age + age_increment * backlog_mask.astype(np.float32, copy=False) - relief_coef * service_frac
        age = np.clip(age, 0.0, age_cap).astype(np.float32, copy=False)
        slack = (deadline_steps - age).astype(np.float32, copy=False)
        overdue = np.maximum(age - deadline_steps, 0.0).astype(np.float32, copy=False)
        expire_rate = max(float(getattr(cfg, "deadline_expire_rate", 0.35) or 0.0), 0.0)
        expire_frac = np.clip(expire_rate * overdue, 0.0, 1.0).astype(np.float32, copy=False)
        expire_amount = (expire_frac * q_after_service_arr).astype(np.float32, copy=False)
        q_after_final = np.maximum(q_after_service_arr - expire_amount, 0.0).astype(np.float32, copy=False)
        empty_mask = q_after_final <= 1.0e-6
        age[empty_mask] = 0.0
        slack[empty_mask] = deadline_steps[empty_mask]
        risk = np.clip(age / np.maximum(deadline_steps, 1.0e-6), 0.0, 2.0).astype(np.float32, copy=False)
        self.last_gu_deadline_age = age.astype(np.float32, copy=False)
        self.last_gu_deadline_slack = slack.astype(np.float32, copy=False)
        self.last_gu_deadline_risk = risk.astype(np.float32, copy=False)
        self.gu_expired = expire_amount.astype(np.float32, copy=False)
        return q_after_final, expire_amount

    def _compute_service_gap_risk_proxy(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0:
            return np.zeros((0,), dtype=np.float32)
        gap = np.asarray(
            getattr(self, "last_gu_service_gap", np.zeros((cfg.num_gu,), dtype=np.float32)),
            dtype=np.float32,
        )
        cap_steps = max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6)
        threshold_steps = float(getattr(cfg, "service_gap_risk_threshold_steps", 3.0) or 0.0)
        threshold_steps = float(np.clip(threshold_steps, 0.0, cap_steps - 1.0e-6))
        denom = max(cap_steps - threshold_steps, 1.0e-6)
        risk = np.maximum(gap - threshold_steps, 0.0) / denom
        return np.clip(risk, 0.0, 1.0).astype(np.float32, copy=False)

    def _refresh_bw_proxy_features(self) -> None:
        cfg = self.cfg
        self.last_gu_urgency_risk = self._compute_overflow_risk_proxy()
        self.last_gu_downstream_pressure = self._compute_downstream_pressure_proxy()
        self.last_gu_service_gap_risk = self._compute_service_gap_risk_proxy()
        if self.last_gu_urgency_risk.shape != (cfg.num_gu,):
            self.last_gu_urgency_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        if self.last_gu_downstream_pressure.shape != (cfg.num_gu,):
            self.last_gu_downstream_pressure = np.zeros((cfg.num_gu,), dtype=np.float32)
        if getattr(self, "last_gu_service_gap", np.zeros((0,), dtype=np.float32)).shape != (cfg.num_gu,):
            self.last_gu_service_gap = np.zeros((cfg.num_gu,), dtype=np.float32)
        if self.last_gu_service_gap_risk.shape != (cfg.num_gu,):
            self.last_gu_service_gap_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        if getattr(self, "last_gu_deadline_age", np.zeros((0,), dtype=np.float32)).shape != (cfg.num_gu,):
            self.last_gu_deadline_age = np.zeros((cfg.num_gu,), dtype=np.float32)
        if getattr(self, "last_gu_deadline_slack", np.zeros((0,), dtype=np.float32)).shape != (cfg.num_gu,):
            self.last_gu_deadline_slack = np.zeros((cfg.num_gu,), dtype=np.float32)
        if getattr(self, "last_gu_deadline_risk", np.zeros((0,), dtype=np.float32)).shape != (cfg.num_gu,):
            self.last_gu_deadline_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        if getattr(self, "gu_expired", np.zeros((0,), dtype=np.float32)).shape != (cfg.num_gu,):
            self.gu_expired = np.zeros((cfg.num_gu,), dtype=np.float32)

    def _gu_proxy_feature_arrays(
        self,
        *,
        assoc_override: np.ndarray | None = None,
        sat_selection_override: List[List[int]] | None = None,
    ) -> list[np.ndarray]:
        cfg = self.cfg
        features: list[np.ndarray] = []
        base_arrival = require_positive_float(
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate)) * float(cfg.tau0),
            name="per-GU arrival reference bits per step",
        )
        if bool(getattr(cfg, "obs_user_include_arrival_rate", False)):
            arrival_rate = np.asarray(self._current_expected_gu_arrival_rates(), dtype=np.float32)
            features.append(arrival_rate / base_arrival)
        if bool(getattr(cfg, "obs_user_include_recent_arrival", False)):
            recent_arrival = np.asarray(
                getattr(self, "last_gu_arrival", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(recent_arrival / base_arrival)
        if bool(getattr(cfg, "obs_user_include_recent_service", False)):
            recent_service = np.asarray(
                getattr(self, "last_gu_outflow", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(recent_service / base_arrival)
        if bool(getattr(cfg, "obs_user_include_queue_headroom", False)):
            queue_headroom = 1.0 - (
                np.asarray(self.gu_queue, dtype=np.float32) / normalize_scale(float(cfg.queue_max_gu))
            )
            features.append(queue_headroom.astype(np.float32, copy=False))
        reward_aligned = None
        if (
            bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False))
            or bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False))
            or bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False))
            or bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False))
            or bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False))
        ):
            reward_aligned = self._gu_reward_aligned_feature_dict(
                normalized=True,
                assoc_override=assoc_override,
                sat_selection_override=sat_selection_override,
            )
        if bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)):
            local_cost = (
                np.asarray(reward_aligned["local_gu_service_cost"], dtype=np.float32)
                if reward_aligned is not None
                else np.zeros((cfg.num_gu,), dtype=np.float32)
            )
            features.append(local_cost)
        if bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)):
            assoc_uav_cost = (
                np.asarray(reward_aligned["assoc_uav_cost"], dtype=np.float32)
                if reward_aligned is not None
                else np.zeros((cfg.num_gu,), dtype=np.float32)
            )
            features.append(assoc_uav_cost)
        if bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)):
            assoc_sat_cost = (
                np.asarray(reward_aligned["assoc_sat_cost_mean"], dtype=np.float32)
                if reward_aligned is not None
                else np.zeros((cfg.num_gu,), dtype=np.float32)
            )
            features.append(assoc_sat_cost)
        if bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)):
            weighted_queue_cost = (
                np.asarray(reward_aligned["weighted_queue_cost"], dtype=np.float32)
                if reward_aligned is not None
                else np.zeros((cfg.num_gu,), dtype=np.float32)
            )
            features.append(weighted_queue_cost)
        if bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)):
            weighted_queue_cost_relative = (
                np.asarray(reward_aligned["weighted_queue_cost_relative"], dtype=np.float32)
                if reward_aligned is not None
                else np.zeros((cfg.num_gu,), dtype=np.float32)
            )
            features.append(weighted_queue_cost_relative)
        if bool(getattr(cfg, "obs_user_include_urgency_risk", False)):
            urgency_risk = np.asarray(
                getattr(self, "last_gu_urgency_risk", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(urgency_risk)
        if bool(getattr(cfg, "obs_user_include_downstream_pressure", False)):
            downstream_pressure = np.asarray(
                getattr(self, "last_gu_downstream_pressure", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(downstream_pressure)
        if bool(getattr(cfg, "obs_user_include_service_gap", False)):
            cap_steps = max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6)
            service_gap = np.asarray(
                getattr(self, "last_gu_service_gap", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(service_gap / cap_steps)
        if bool(getattr(cfg, "obs_user_include_service_gap_risk", False)):
            service_gap_risk = np.asarray(
                getattr(self, "last_gu_service_gap_risk", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(service_gap_risk)
        if bool(getattr(cfg, "obs_user_include_deadline_slack", False)):
            deadline_steps = np.maximum(
                np.asarray(getattr(self, "gu_deadline_steps", np.ones((cfg.num_gu,), dtype=np.float32)), dtype=np.float32),
                1.0e-6,
            )
            deadline_slack = np.asarray(
                getattr(self, "last_gu_deadline_slack", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(np.clip(deadline_slack / deadline_steps, -1.0, 1.0).astype(np.float32, copy=False))
        if bool(getattr(cfg, "obs_user_include_deadline_risk", False)):
            deadline_risk = np.asarray(
                getattr(self, "last_gu_deadline_risk", np.zeros((cfg.num_gu,), dtype=np.float32)),
                dtype=np.float32,
            )
            features.append(np.clip(deadline_risk, 0.0, 2.0).astype(np.float32, copy=False))
        return [np.asarray(feature, dtype=np.float32) for feature in features]

    def _empty_step_profile(self) -> Dict[str, float]:
        return {
            "dynamics_time_sec": 0.0,
            "orbit_visible_time_sec": 0.0,
            "assoc_access_time_sec": 0.0,
            "backhaul_queue_time_sec": 0.0,
            "reward_time_sec": 0.0,
            "obs_time_sec": 0.0,
            "state_time_sec": 0.0,
            "step_total_time_sec": 0.0,
        }

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _dummy_actions(self) -> Dict[str, Dict[str, np.ndarray]]:
        assoc = self._associate_users()
        bw_rows: list[np.ndarray] = []
        for u in range(int(self.cfg.num_uav)):
            row = np.zeros(int(self.cfg.num_gu), dtype=np.float32)
            valid = assoc == u
            valid_count = int(np.sum(valid))
            if valid_count > 0:
                row[valid] = 1.0 / float(valid_count)
            bw_rows.append(row)
        return {
            agent: {
                "accel": np.zeros(2, dtype=np.float32),
                "bw_alloc": bw_rows[idx],
                "sat_select_mask": np.zeros(self.cfg.sats_obs_max, dtype=np.float32),
            }
            for idx, agent in enumerate(self.agents)
        }

    def _zero_bw_action_matrix(self) -> np.ndarray:
        return np.zeros((self.cfg.num_uav, self.cfg.num_gu), dtype=np.float32)

    def _sat_selection_matrix(self, selections: List[List[int]] | np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        cfg = self.cfg
        select_k = max(int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", cfg.N_RF)) or cfg.N_RF), 1)
        if isinstance(selections, np.ndarray):
            matrix = np.asarray(selections, dtype=np.int64)
            if matrix.ndim == 1:
                if int(cfg.num_uav) != 1:
                    raise ValueError(
                        f"1D sat selection matrix is only valid when num_uav=1, got num_uav={cfg.num_uav}."
                    )
                matrix = matrix.reshape(1, -1)
            if int(matrix.shape[0]) != int(cfg.num_uav):
                raise ValueError(
                    f"Sat selection matrix first dim must be {cfg.num_uav}, got {tuple(matrix.shape)}."
                )
            matrix_width = max(int(select_k), int(matrix.shape[1]))
            matrix_out = np.full((int(cfg.num_uav), matrix_width), -1, dtype=np.int64)
            copy_cols = min(matrix_width, int(matrix.shape[1]))
            if copy_cols > 0:
                matrix_out[:, :copy_cols] = matrix[:, :copy_cols]
            return _normalize_sat_selection_matrix_rows(matrix_out)

        selection_rows = [np.asarray(selected, dtype=np.int64).reshape(-1) for selected in selections]
        if selection_rows:
            select_k = max(select_k, max(int(row.size) for row in selection_rows))
        matrix = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
        for u, sat_idx in enumerate(selection_rows):
            if u >= int(cfg.num_uav):
                break
            if sat_idx.size <= 0:
                continue
            fill = min(int(sat_idx.size), int(select_k))
            matrix[u, :fill] = sat_idx[:fill]
        return _normalize_sat_selection_matrix_rows(matrix)

    def _sat_selection_lists(self, selections: List[List[int]] | np.ndarray | Sequence[Sequence[int]]) -> List[List[int]]:
        matrix = self._sat_selection_matrix(selections)
        out: List[List[int]] = []
        for u in range(int(matrix.shape[0])):
            selected = matrix[u]
            out.append([int(sat_idx) for sat_idx in selected[selected >= 0].tolist()])
        return out

    def _refresh_observation_cache_from_current_state(self) -> None:
        assoc = self._associate_users()
        candidates = self._build_candidate_users(assoc)
        self._cached_bw_valid_mask = self._build_bw_valid_mask(assoc, candidates)
        access_snapshot = self._sample_access_channel_snapshot()
        _, eta = self._compute_access_rates(
            assoc,
            candidates,
            self._zero_bw_action_matrix(),
            record_exec=False,
            access_snapshot=access_snapshot,
        )
        self._store_cached_access_stage_context(
            assoc,
            candidates,
            eta=eta,
            bw_valid_mask=self._cached_bw_valid_mask,
            access_snapshot=access_snapshot,
            snapshot_step_t=int(self.t),
        )
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        self._cached_global_state = None
        self._cached_obs_runtime_context = None

    def export_runtime_state(self) -> Dict[str, Any]:
        excluded = {
            "cfg",
            "rng",
            "orbit",
            "agents",
            "possible_agents",
            "_obs_space",
            "_act_space",
            "_orbit_pos_table",
            "_orbit_vel_table",
            "_cached_orbit_t",
            "_cached_orbit_pos",
            "_cached_orbit_vel",
            "_cached_elevation_t",
            "_cached_elevation_matrix",
            "_cached_backhaul_loss_t",
            "_cached_backhaul_loss_matrix",
            "_cached_uav_ecef",
            "_cached_uav_vel_ecef",
            "_cached_uav_neighbor_t",
            "_cached_uav_neighbor_order",
            "_cached_global_state",
            "_cached_assoc",
            "_cached_candidates",
            "_cached_eta",
            "_cached_eta_uav_pos",
            "_cached_eta_gu_pos",
            "_cached_access_snapshot_t",
            "_cached_access_gain_matrix",
            "_cached_bw_valid_mask",
            "_cached_sat_obs",
            "_cached_sat_mask",
            "_cached_sat_valid_mask",
            "_structured_native_driver_group",
        }
        state: Dict[str, Any] = {
            "rng_bit_generator_state": copy.deepcopy(self.rng.bit_generator.state),
        }
        for key, value in self.__dict__.items():
            if key in excluded:
                continue
            state[key] = copy.deepcopy(value)
        return state

    def load_runtime_state(
        self,
        state: Dict[str, Any],
        *,
        refresh_observation_cache: bool = True,
        refresh_global_state_cache: bool = True,
    ) -> None:
        state_dict = dict(state or {})
        rng_state = copy.deepcopy(state_dict.pop("rng_bit_generator_state", None))
        for key, value in state_dict.items():
            setattr(self, key, copy.deepcopy(value))
        self.rng = np.random.default_rng()
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self._invalidate_step_caches()
        self._refresh_uav_cache()
        if bool(refresh_observation_cache):
            self._refresh_observation_cache_from_current_state()
        if bool(refresh_global_state_cache):
            self._refresh_global_state_cache()

    def _queue_init_arrival_ref(self) -> float:
        cfg = self.cfg
        return (
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
            * float(cfg.num_gu)
            * float(cfg.tau0)
        )

    def _queue_init_layer_ref(self, layer: str) -> float:
        cfg = self.cfg
        ref_value = getattr(cfg, f"queue_ref_{layer}_per_step", None)
        if ref_value is not None:
            return max(float(ref_value), 0.0)
        return self._queue_init_arrival_ref()

    def _queue_init_entity_ref(self, layer: str) -> float:
        cfg = self.cfg
        total_ref = self._queue_init_layer_ref(layer)
        if layer == "gu":
            entities = max(float(cfg.num_gu), 1.0)
        elif layer == "uav":
            entities = max(float(cfg.num_uav), 1.0)
        else:
            active_count = getattr(cfg, "queue_ref_sat_active_count", None)
            entities = max(float(active_count), 1.0) if active_count is not None else max(float(cfg.num_sat), 1.0)
        return total_ref / entities

    def _resolve_queue_init_total(
        self,
        abs_attr: str,
        steps_attr: str,
        frac_attr: str,
        layer: str,
        total_cap: float,
    ) -> float:
        cfg = self.cfg
        abs_value = getattr(cfg, abs_attr, None)
        if abs_value is not None:
            return min(max(float(abs_value), 0.0), total_cap)

        steps_value = getattr(cfg, steps_attr, None)
        if steps_value is not None:
            total = max(float(steps_value), 0.0) * self._queue_init_layer_ref(layer)
            return min(total, total_cap)

        frac_value = max(float(getattr(cfg, frac_attr, 0.0) or 0.0), 0.0)
        return min(float(np.clip(frac_value, 0.0, 1.0)) * total_cap, total_cap)

    def _init_queues(self) -> None:
        cfg = self.cfg
        gu_total = self._resolve_queue_init_total(
            "queue_init_gu_abs",
            "queue_init_gu_steps",
            "queue_init_frac",
            "gu",
            float(cfg.num_gu) * float(cfg.queue_max_gu),
        )
        uav_total = self._resolve_queue_init_total(
            "queue_init_uav_abs",
            "queue_init_uav_steps",
            "queue_init_uav_frac",
            "uav",
            float(cfg.num_uav) * float(cfg.queue_max_uav),
        )
        sat_total = self._resolve_queue_init_total(
            "queue_init_sat_abs",
            "queue_init_sat_steps",
            "queue_init_sat_frac",
            "sat",
            float(cfg.num_sat) * float(cfg.queue_max_sat),
        )

        if cfg.num_gu > 0 and gu_total > 0.0:
            self.gu_queue = np.full((cfg.num_gu,), gu_total / float(cfg.num_gu), dtype=np.float32)
        if cfg.num_uav > 0 and uav_total > 0.0:
            self.uav_queue = np.full((cfg.num_uav,), uav_total / float(cfg.num_uav), dtype=np.float32)
        if cfg.num_sat > 0 and sat_total > 0.0:
            self.sat_queue = np.full((cfg.num_sat,), sat_total / float(cfg.num_sat), dtype=np.float32)

    def _init_traffic_model_state(self) -> None:
        cfg = self.cfg
        hetero = max(float(getattr(cfg, "arrival_base_hetero", 0.0) or 0.0), 0.0)
        low = max(1.0 - hetero, 1.0e-3)
        high = max(1.0 + hetero, low)
        if cfg.num_gu > 0:
            self._arrival_base_scale = self.rng.uniform(low, high, size=(cfg.num_gu,)).astype(np.float32, copy=False)
        else:
            self._arrival_base_scale = np.zeros((0,), dtype=np.float32)
        assoc = self._associate_users()
        subsets = self._build_hotspot_subsets(assoc) if self._traffic_model() == "sticky_subset_hotspot" else []
        self._hotspot_subsets = [np.asarray(subset, dtype=np.int32) for subset in subsets]
        self._hotspot_member_mask = np.zeros((len(self._hotspot_subsets), cfg.num_gu), dtype=bool)
        for idx, subset in enumerate(self._hotspot_subsets):
            self._hotspot_member_mask[idx, subset] = True
        self._hotspot_active_idx = -1
        self.last_hotspot_index = -1
        self.last_gu_arrival_rate_vec = np.full((cfg.num_gu,), float(self.effective_task_arrival_rate), dtype=np.float32)
        self.last_hotspot_mask = np.zeros((cfg.num_gu,), dtype=np.float32)

    def _apply_focus_preload(self) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "preload_enabled", False)):
            return
        if cfg.num_gu <= 0:
            return
        prob = float(np.clip(float(getattr(cfg, "preload_prob", 0.0) or 0.0), 0.0, 1.0))
        if prob <= 0.0 or float(self.rng.random()) > prob:
            return
        if len(getattr(self, "_hotspot_subsets", [])) <= 0:
            return

        hotspot_idx = int(self.rng.integers(len(self._hotspot_subsets)))
        self._hotspot_active_idx = hotspot_idx
        self.last_hotspot_index = hotspot_idx
        hot_mask = np.asarray(self._hotspot_member_mask[hotspot_idx], dtype=bool)
        self.last_hotspot_mask = hot_mask.astype(np.float32, copy=False)

        hot_gu_steps = max(float(getattr(cfg, "preload_hot_gu_steps", 0.0) or 0.0), 0.0)
        bg_gu_steps = max(float(getattr(cfg, "preload_bg_gu_steps", 0.0) or 0.0), 0.0)
        hot_uav_steps = max(float(getattr(cfg, "preload_hot_uav_steps", 0.0) or 0.0), 0.0)
        sat_steps = max(float(getattr(cfg, "preload_sat_steps", 0.0) or 0.0), 0.0)

        gu_bg_value = bg_gu_steps * self._queue_init_entity_ref("gu")
        gu_hot_value = hot_gu_steps * self._queue_init_entity_ref("gu")
        if gu_bg_value > 0.0:
            self.gu_queue = np.maximum(self.gu_queue, gu_bg_value).astype(np.float32, copy=False)
        if gu_hot_value > 0.0 and np.any(hot_mask):
            self.gu_queue[hot_mask] = np.maximum(self.gu_queue[hot_mask], gu_hot_value).astype(np.float32, copy=False)
        self.gu_queue = np.minimum(self.gu_queue, float(cfg.queue_max_gu)).astype(np.float32, copy=False)

        if hot_uav_steps > 0.0 and cfg.num_uav > 0:
            assoc = self._associate_users()
            hot_assoc = assoc[hot_mask]
            hot_uavs = np.unique(hot_assoc[hot_assoc >= 0])
            if hot_uavs.size > 0:
                uav_hot_value = hot_uav_steps * self._queue_init_entity_ref("uav")
                self.uav_queue[hot_uavs] = np.maximum(self.uav_queue[hot_uavs], uav_hot_value).astype(np.float32, copy=False)
                self.uav_queue = np.minimum(self.uav_queue, float(cfg.queue_max_uav)).astype(np.float32, copy=False)

        if sat_steps > 0.0 and cfg.num_sat > 0:
            sat_value = sat_steps * self._queue_init_entity_ref("sat")
            self.sat_queue = np.maximum(self.sat_queue, sat_value).astype(np.float32, copy=False)
            self.sat_queue = np.minimum(self.sat_queue, float(cfg.queue_max_sat)).astype(np.float32, copy=False)

    def _init_state(self) -> None:
        cfg = self.cfg
        self.t = 0
        self._episode_collision_count = 0
        self._episode_step_count = 0
        raw_num_clusters = getattr(cfg, "gu_init_num_clusters", None)
        if raw_num_clusters is None:
            num_clusters = max(1, cfg.num_gu // 5)
        else:
            num_clusters = max(1, int(raw_num_clusters))
        cluster_std = max(float(getattr(cfg, "gu_init_cluster_std", 80.0) or 0.0), 0.0)
        center_min_dist = max(float(getattr(cfg, "gu_init_cluster_center_min_dist", 0.0) or 0.0), 0.0)
        self.gu_pos, self.gu_cluster_centers, self.gu_cluster_counts = thomas_cluster_process(
            cfg.num_gu,
            cfg.map_size,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
            center_min_dist=center_min_dist,
            rng=self.rng,
            return_metadata=True,
        )
        self.uav_pos = self._sample_uav_positions()
        self.uav_vel = self._sample_uav_initial_velocities()
        self.uav_energy = np.full((cfg.num_uav,), cfg.uav_energy_init, dtype=np.float32)
        self.last_policy_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        self.last_exec_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        self.last_intervention_norm_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_close_risk_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_danger_imitation_mask = np.zeros((cfg.num_uav,), dtype=np.float32)

        self.gu_queue = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.uav_queue = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.sat_queue = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_association = np.full((cfg.num_gu,), -1, dtype=np.int32)
        self.prev_association = self.last_association.copy()
        self._init_traffic_model_state()
        self._init_queues()
        self._apply_focus_preload()
        self.prev_queue_sum = 0.0
        self.prev_queue_sum_active = 0.0
        self.prev_queue_sum_gu = 0.0
        self.prev_queue_sum_uav = 0.0
        self.prev_queue_sum_sat = 0.0
        self.prev_gu_queue_vec = np.asarray(self.gu_queue, dtype=np.float32).copy()
        self.prev_uav_queue_vec = np.asarray(self.uav_queue, dtype=np.float32).copy()
        self.prev_sat_queue_vec = np.asarray(self.sat_queue, dtype=np.float32).copy()
        arrival_ref = (
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
            * max(float(cfg.num_gu), 1.0)
            * float(cfg.tau0)
        )
        self.prev_arrival_sum = reward_ratio_denominator_scalar(arrival_ref, name="arrival_ref_bits_per_step")
        self.prev_q_norm_active = 0.0
        self.prev_centroid_dist_mean = self._compute_centroid_stats()[1]
        self.prev_d_min = 0.0
        self.last_gu_outflow = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_uav_outflow = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_access_interference_by_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_bw_fraction_by_uav_gu = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
        self.last_gu_to_uav_inflow_by_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_uav_to_sat_outflow_matrix = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        self.last_selected_mask_by_uav_sat = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        self.last_gu_arrival = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_arrival_rate_vec = np.asarray(
            getattr(
                self,
                "last_gu_arrival_rate_vec",
                np.full((cfg.num_gu,), float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate)), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        self.gu_deadline_steps = self._sample_deadline_steps()
        self.last_gu_urgency_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_downstream_pressure = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_service_gap = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_service_gap_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_deadline_age = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_deadline_slack = np.asarray(self.gu_deadline_steps, dtype=np.float32).copy()
        self.last_gu_deadline_risk = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.gu_drop = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.gu_expired = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.uav_drop = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.sat_drop = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_exec_bw_alloc = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
        self.last_exec_sat_select_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        sat_k = max(int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", cfg.N_RF)) or cfg.N_RF), 0)
        self.last_exec_sat_indices = np.full((cfg.num_uav, sat_k), -1, dtype=np.int64)
        self.last_energy_cost = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_sat_processed = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_sat_incoming = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_bw_align = 0.0
        self.last_sat_score = 0.0
        bw_workload_arrival_ref = self._arrival_ref()
        bw_acc_init = bw_workload_arrival_ref / max(float(cfg.num_gu), 1.0)
        bw_rel_init = bw_workload_arrival_ref / max(float(cfg.num_uav), 1.0)
        bw_sat_init = bw_workload_arrival_ref / self._bw_weighted_workload_sat_active_ref_count()
        self.bw_weighted_workload_acc_ema_vec = np.full((cfg.num_gu,), bw_acc_init, dtype=np.float32)
        self.bw_weighted_workload_rel_ema_vec = np.full((cfg.num_uav,), bw_rel_init, dtype=np.float32)
        self.bw_weighted_workload_sat_ema_vec = np.full((cfg.num_sat,), bw_sat_init, dtype=np.float32)
        self.bw_weighted_workload_acc_ema = float(np.sum(self.bw_weighted_workload_acc_ema_vec))
        self.bw_weighted_workload_rel_ema = float(np.sum(self.bw_weighted_workload_rel_ema_vec))
        self.bw_weighted_workload_sat_ema = float(np.sum(self.bw_weighted_workload_sat_ema_vec))
        self.last_sat_selection: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        self.last_sat_connection_counts = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_connected_sat_count = 0.0
        self.last_connected_sat_dist_mean = 0.0
        self.last_connected_sat_dist_p95 = 0.0
        self.last_connected_sat_elevation_deg_mean = 0.0
        self.last_connected_sat_elevation_deg_min = 0.0
        self.last_visible_raw_counts = np.zeros((cfg.num_uav,), dtype=np.int32)
        self.last_visible_kept_counts = np.zeros((cfg.num_uav,), dtype=np.int32)
        self.last_visible_raw_candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        self.last_visible_candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        self.last_visible_candidate_rank_values: List[List[float]] = [[] for _ in range(cfg.num_uav)]
        self.last_visible_candidate_scores: List[List[float]] = [[] for _ in range(cfg.num_uav)]
        self.last_visible_candidate_rank_gap_top1_top2 = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_score_gap_top1_top2 = np.zeros((cfg.num_uav,), dtype=np.float32)
        self._refresh_bw_proxy_features()
        self.last_visible_candidate_dist_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_elevation_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_se_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_queue_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_stats: Dict[str, float | str] = {}
        self.last_arrival_rate = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
        self._reset_doppler_residual_state()
        self.last_filter_active_ratio = 0.0
        self.last_projected_delta_norm_mean = 0.0
        self.last_fallback_count = 0.0
        self.last_boundary_filter_count = 0.0
        self.last_pairwise_filter_count = 0.0
        self.last_pairwise_filter_active_ratio = 0.0
        self.last_pairwise_projected_delta_norm = 0.0
        self.last_pairwise_fallback_count = 0.0
        self.last_pairwise_candidate_infeasible_count = 0.0
        self.last_safety_shield_active = 0.0
        self.last_safety_shield_feasible = 1.0
        self.last_safety_shield_delta_norm = 0.0
        self.last_safety_shield_delta_norm_max = 0.0
        self.last_safety_shield_min_margin_before = 0.0
        self.last_safety_shield_min_margin_after = 0.0
        self.last_safety_shield_min_distance_after = 0.0
        self.last_safety_shield_pair_count = 0.0
        self.last_safety_shield_status = "disabled"
        self.last_safety_shield_solver = ""
        self.last_step_profile = self._empty_step_profile()
        self.last_assoc_centroid_dist_norms = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_sat_overlap_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_reward_parts = {
            "service_ratio": 0.0,
            "drop_ratio": 0.0,
            "arrival_ref": self._arrival_ref(),
            "b_pre_steps": 0.0,
            "x_acc": 0.0,
            "x_rel": 0.0,
            "g_pre": 0.0,
            "d_pre": 0.0,
            "processed_ratio_eval": 0.0,
            "drop_ratio_eval": 0.0,
            "pre_backlog_steps_eval": 0.0,
            "sat_overlap_eval": 0.0,
            "assoc_centroid_dist_norm_mean": 0.0,
            "assoc_centroid_valid_uav_count": 0.0,
            "assoc_centroid_valid_uav_frac": 0.0,
            "D_sys_report": 0.0,
            "drop_sum": 0.0,
            "drop_sum_active": 0.0,
            "gu_drop_sum": 0.0,
            "uav_drop_sum": 0.0,
            "sat_drop_sum": 0.0,
            "drop_event": 0.0,
            "arrival_sum": 0.0,
            "outflow_sum": 0.0,
            "backhaul_sum": 0.0,
            "sat_processed_sum": 0.0,
            "service_norm": 0.0,
            "drop_norm": 0.0,
            "gu_drop_norm": 0.0,
            "uav_drop_norm": 0.0,
            "sat_drop_norm": 0.0,
            "throughput_access_norm": 0.0,
            "throughput_backhaul_norm": 0.0,
            "sat_processed_norm": 0.0,
            "outflow_arrival_ratio_step": 0.0,
            "sat_incoming_arrival_ratio_step": 0.0,
            "sat_processed_arrival_ratio_step": 0.0,
            "sat_processed_incoming_ratio_step": 0.0,
            "gu_drop_ratio_step": 0.0,
            "uav_drop_ratio_step": 0.0,
            "sat_drop_ratio_step": 0.0,
            "queue_pen": 0.0,
            "queue_pen_gu": 0.0,
            "queue_pen_uav": 0.0,
            "queue_pen_sat": 0.0,
            "gu_queue_fill_fraction": 0.0,
            "uav_queue_fill_fraction": 0.0,
            "sat_queue_fill_fraction": 0.0,
            "gu_queue_arrival_steps": 0.0,
            "uav_queue_arrival_steps": 0.0,
            "sat_queue_arrival_steps": 0.0,
            "queue_topk": 0.0,
            "assoc_ratio": 0.0,
            "assoc_unfair_max_gu_count": 0.0,
            "assoc_unfair_step": 0.0,
            "queue_delta": 0.0,
            "queue_delta_mode": "total",
            "queue_delta_gu": 0.0,
            "queue_delta_uav": 0.0,
            "queue_delta_sat": 0.0,
            "q_norm_active": 0.0,
            "prev_q_norm_active": 0.0,
            "q_norm_delta": 0.0,
            "q_norm_tail_q0": 0.0,
            "q_norm_tail_excess": 0.0,
            "queue_weight": 0.0,
            "q_delta_weight": 0.0,
            "crash_weight": 0.0,
            "centroid_transfer_ratio": 0.0,
            "centroid_eta": 0.0,
            "centroid_reward": 0.0,
            "centroid_dist_mean": 0.0,
            "bw_align": 0.0,
            "sat_score": 0.0,
            "dist_reward": 0.0,
            "dist_delta": 0.0,
            "energy_reward": 0.0,
            "collision_event": 0.0,
            "collision_penalty": 0.0,
            "battery_penalty": 0.0,
            "fail_penalty": 0.0,
            "avoidance_eta_eff": float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta)),
            "avoidance_eta_exec": float(getattr(self, "last_avoidance_eta_exec", cfg.avoidance_eta)),
            "avoidance_collision_rate_ema": float(getattr(self, "avoidance_collision_rate_ema", 0.0)),
            "avoidance_prev_episode_collision_rate": float(getattr(self, "prev_episode_collision_rate", 0.0)),
            "filter_active_ratio": 0.0,
            "projected_delta_norm_mean": 0.0,
            "fallback_count": 0.0,
            "boundary_filter_count": 0.0,
            "pairwise_filter_count": 0.0,
            "pairwise_filter_active_ratio": 0.0,
            "pairwise_projected_delta_norm": 0.0,
            "pairwise_fallback_count": 0.0,
            "pairwise_candidate_infeasible_count": 0.0,
            "safety_shield_active": 0.0,
            "safety_shield_feasible": 1.0,
            "safety_shield_delta_norm": 0.0,
            "safety_shield_delta_norm_max": 0.0,
            "safety_shield_min_margin_before": 0.0,
            "safety_shield_min_margin_after": 0.0,
            "safety_shield_min_distance_after": 0.0,
            "safety_shield_pair_count": 0.0,
            "term_service": 0.0,
            "term_drop": 0.0,
            "term_pre_drop": 0.0,
            "term_drop_gu": 0.0,
            "term_drop_uav": 0.0,
            "term_drop_sat": 0.0,
            "term_drop_step": 0.0,
            "term_queue": 0.0,
            "term_pre_backlog": 0.0,
            "term_topk": 0.0,
            "term_assoc": 0.0,
            "term_q_delta": 0.0,
            "term_throughput_access": 0.0,
            "term_throughput_backhaul": 0.0,
            "term_access": 0.0,
            "term_relay": 0.0,
            "term_queue_gu_arrival": 0.0,
            "term_centroid": 0.0,
            "term_bw_align": 0.0,
            "term_sat_score": 0.0,
            "term_dist": 0.0,
            "term_dist_delta": 0.0,
            "term_energy": 0.0,
            "term_accel": 0.0,
            "intervention_norm": 0.0,
            "intervention_rate": 0.0,
            "intervention_norm_top1": 0.0,
            "danger_imitation_active_rate": 0.0,
            "close_risk": 0.0,
            "term_close_risk": 0.0,
            "reward_raw": 0.0,
        }
        self._cached_candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        self._cached_assoc = -np.ones((cfg.num_gu,), dtype=np.int32)
        self._cached_eta = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        self._cached_eta_uav_pos = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        self._cached_eta_gu_pos = np.zeros((cfg.num_gu, 2), dtype=np.float32)
        self._cached_access_snapshot_t = -1
        self._cached_bw_valid_mask = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        self._cached_sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, self.sat_dim), dtype=np.float32)
        self._cached_sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        self._cached_sat_valid_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        self._cached_global_state = None
        self._cached_obs_runtime_context = None
        self._invalidate_step_caches()
        self._refresh_uav_cache()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._update_adaptive_avoidance_after_episode()
        self.episode_idx = int(getattr(self, "episode_idx", 0)) + 1
        self._set_effective_task_arrival_rate()
        self._init_state()
        # Prime candidates and eta for initial observations
        assoc = self._associate_users()
        candidates = self._build_candidate_users(assoc)
        self._cached_bw_valid_mask = self._build_bw_valid_mask(assoc, candidates)
        dummy_actions = self._dummy_actions()
        access_snapshot = self._sample_access_channel_snapshot()
        _, eta = self._compute_access_rates(
            assoc,
            candidates,
            dummy_actions,
            record_exec=False,
            access_snapshot=access_snapshot,
        )
        self._store_cached_access_stage_context(
            assoc,
            candidates,
            eta=eta,
            bw_valid_mask=self._cached_bw_valid_mask,
            access_snapshot=access_snapshot,
            snapshot_step_t=int(self.t),
        )
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        obs_context = self._build_obs_runtime_context()
        obs = self._build_all_obs_from_context(obs_context)
        infos = {
            agent: {
                "traffic_level": self.traffic_level,
                "traffic_level_ratio": self.traffic_level_ratio,
                "effective_task_arrival_rate": self.effective_task_arrival_rate,
                **self._agent_visible_info(idx),
            }
            for idx, agent in enumerate(self.agents)
        }
        self._refresh_global_state_cache(
            gu_proxy_features=[np.asarray(feature, dtype=np.float32) for feature in obs_context["gu_proxy_features"]],
        )
        return obs, infos

    def step(self, actions: Dict[str, Dict]):
        cfg = self.cfg
        step_start = time.perf_counter()
        step_profile = self._empty_step_profile()
        self.global_step = int(getattr(self, "global_step", 0)) + 1
        self.prev_queue_sum_gu = _contract_sum_scalar_np(cfg, self.gu_queue)
        self.prev_queue_sum_uav = _contract_sum_scalar_np(cfg, self.uav_queue)
        self.prev_queue_sum_sat = _contract_sum_scalar_np(cfg, self.sat_queue)
        self.prev_queue_sum = float(
            self.prev_queue_sum_gu + self.prev_queue_sum_uav + self.prev_queue_sum_sat
        )
        self.prev_queue_sum_active = float(self.prev_queue_sum_gu + self.prev_queue_sum_uav)
        self.prev_gu_queue_vec = np.asarray(self.gu_queue, dtype=np.float32).copy()
        self.prev_uav_queue_vec = np.asarray(self.uav_queue, dtype=np.float32).copy()
        self.prev_sat_queue_vec = np.asarray(self.sat_queue, dtype=np.float32).copy()
        prev_scale = self._queue_arrival_scale(float(getattr(self, "prev_arrival_sum", 0.0)))
        self.prev_q_norm_active = float(np.clip(self.prev_queue_sum_active / prev_scale, 0.0, 1.0))
        self.prev_centroid_dist_mean = self._compute_centroid_stats()[1]
        cached_assoc = getattr(self, "_cached_assoc", None)
        step_assoc = np.asarray(cached_assoc, dtype=np.int32) if cached_assoc is not None else np.zeros((0,), dtype=np.int32)
        if step_assoc.shape != (cfg.num_gu,):
            step_assoc = self._associate_users()
        step_candidates = [list(cand) for cand in getattr(self, "_cached_candidates", [[] for _ in range(cfg.num_uav)])]
        if len(step_candidates) != cfg.num_uav:
            step_candidates = self._build_candidate_users(step_assoc)
        step_visible = [list(vis) for vis in getattr(self, "last_visible_candidates", [[] for _ in range(cfg.num_uav)])]
        if len(step_visible) != cfg.num_uav:
            sat_pos_pre, _ = self._get_orbit_states()
            step_visible = self._visible_sats_sorted(sat_pos_pre)
        if cfg.num_gu > 0:
            d2d = np.linalg.norm(self.gu_pos - self.uav_pos[:, None, :], axis=2)
            self.prev_d_min = float(np.min(d2d))
        else:
            self.prev_d_min = 0.0
        profile_start = time.perf_counter()
        self._apply_uav_dynamics(actions)
        step_profile["dynamics_time_sec"] = time.perf_counter() - profile_start

        self.prev_association = self.last_association.copy()
        # Satellite states
        profile_start = time.perf_counter()
        sat_pos, sat_vel = self._get_orbit_states()
        step_profile["orbit_visible_time_sec"] = time.perf_counter() - profile_start

        # Compute associations and rates
        profile_start = time.perf_counter()
        access_snapshot = self._sample_access_channel_snapshot()
        access_rates, _ = self._compute_access_rates(
            step_assoc,
            step_candidates,
            actions,
            record_exec=True,
            access_snapshot=access_snapshot,
        )
        step_profile["assoc_access_time_sec"] = time.perf_counter() - profile_start

        # Update GU queues
        profile_start = time.perf_counter()
        sat_selection = self._select_satellites(sat_pos, sat_vel, actions, step_visible)
        self._apply_bw_transition_core(
            step_assoc,
            step_candidates,
            actions,
            sat_selection,
            sat_pos,
            sat_vel,
            access_rates=np.asarray(access_rates, dtype=np.float32),
        )
        step_profile["backhaul_queue_time_sec"] = time.perf_counter() - profile_start

        # Cache for obs
        profile_start = time.perf_counter()
        self._prepare_next_step_observation_cache(sat_pos, sat_vel, advance_doppler=True)
        step_profile["obs_time_sec"] = time.perf_counter() - profile_start

        # Rewards and done
        profile_start = time.perf_counter()
        step_status = self._finalize_post_bw_step()
        step_profile["reward_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        step_outputs = self._materialize_post_step_outputs(
            step_status,
            materialize_step_outputs=True,
            materialize_agent_dicts=True,
            refresh_global_state_cache=False,
        )
        step_profile["obs_time_sec"] += time.perf_counter() - profile_start
        profile_start = time.perf_counter()
        self._refresh_global_state_cache(gu_proxy_features=step_outputs.gu_proxy_features)
        step_profile["state_time_sec"] = time.perf_counter() - profile_start
        step_profile["step_total_time_sec"] = time.perf_counter() - step_start
        self.last_step_profile = step_profile
        return (
            step_outputs.obs,
            step_outputs.rewards,
            step_outputs.terminations,
            step_outputs.truncations,
            step_outputs.infos,
        )

    def _boundary_margin(self) -> float:
        cfg = self.cfg
        raw_margin = getattr(cfg, "boundary_margin", None)
        if raw_margin is None:
            margin = float(cfg.d_safe)
        else:
            margin = max(float(raw_margin), 0.0)
        return float(min(margin, max(0.0, 0.5 * float(cfg.map_size) - 1e-6)))

    def _project_axis_to_boundary(
        self,
        pos: float,
        vel: float,
        accel_cmd: float,
        lower: float,
        upper: float,
    ) -> Tuple[float, bool, bool]:
        cfg = self.cfg
        tau = max(float(cfg.tau0), 1e-6)
        a_max = float(cfg.a_max)
        v_max = float(cfg.v_max)
        accel_cmd = float(np.clip(accel_cmd, -a_max, a_max))
        vel_cmd = float(np.clip(vel + accel_cmd * tau, -v_max, v_max))
        pos_cmd = float(pos + vel_cmd * tau)
        if lower <= pos_cmd <= upper:
            return accel_cmd, False, False

        vel_low = max(-v_max, (lower - pos) / tau)
        vel_high = min(v_max, (upper - pos) / tau)
        if vel_low <= vel_high:
            target_vel = float(np.clip(vel_cmd, vel_low, vel_high))
            accel_proj = float(np.clip((target_vel - vel) / tau, -a_max, a_max))
            vel_next = float(np.clip(vel + accel_proj * tau, -v_max, v_max))
            pos_next = float(pos + vel_next * tau)
            if lower <= pos_next <= upper:
                return accel_proj, True, False

        if pos_cmd < lower or pos < lower:
            accel_fallback = a_max
        elif pos_cmd > upper or pos > upper:
            accel_fallback = -a_max
        else:
            center = 0.5 * (lower + upper)
            accel_fallback = a_max if pos < center else -a_max
        return float(np.clip(accel_fallback, -a_max, a_max)), True, True

    def _predict_next_from_accel(self, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        accel_arr = _project_l2_ball_np(accel, cfg.a_max)
        vel_next = _project_l2_ball_np(self.uav_vel + accel_arr * cfg.tau0, cfg.v_max)
        pos_next = self.uav_pos + vel_next * cfg.tau0
        return pos_next.astype(np.float32, copy=False), vel_next.astype(np.float32, copy=False)

    def _pairwise_hard_distance(self) -> float:
        cfg = self.cfg
        raw_distance = getattr(cfg, "pairwise_hard_distance", None)
        if raw_distance is None:
            return float(cfg.d_safe + 5.0)
        return max(float(raw_distance), float(cfg.d_safe))

    def _pairwise_trigger_mode(self) -> str:
        cfg = self.cfg
        mode = str(getattr(cfg, "pairwise_hard_trigger_mode", "distance") or "distance").strip().lower()
        if mode not in {"distance", "ttc"}:
            return "distance"
        return mode

    def _pairwise_trigger_ttc(self) -> float:
        cfg = self.cfg
        return max(float(getattr(cfg, "pairwise_hard_trigger_ttc", 2.0) or 0.0), 0.0)

    def _pairwise_trigger_distance(self, d_hard: float) -> float:
        cfg = self.cfg
        raw_distance = getattr(cfg, "pairwise_hard_trigger_distance", None)
        if raw_distance is not None:
            return max(float(raw_distance), d_hard)
        if self._pairwise_trigger_mode() == "ttc":
            return max(d_hard, d_hard + 2.0 * float(cfg.v_max) * self._pairwise_trigger_ttc())
        return d_hard

    def _pairwise_closing_speed_threshold(self) -> float:
        cfg = self.cfg
        return max(float(getattr(cfg, "pairwise_hard_closing_speed", 0.0) or 0.0), 0.0)

    def _pairwise_correction_direction(
        self,
        pair_diff_next: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        direction = np.asarray(pair_diff_next, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            return direction / norm
        direction = np.asarray(self.uav_pos[i] - self.uav_pos[j], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            return direction / norm
        direction = np.asarray(self.uav_vel[i] - self.uav_vel[j], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            return direction / norm
        return np.array([1.0, 0.0], dtype=np.float32)

    def _evaluate_pairwise_ttc_resolution(
        self,
        accel: np.ndarray,
        i: int,
        j: int,
        d_hard: float,
        direction: np.ndarray,
        dist_cur: float,
        ttc_limit: float,
        closing_speed_thresh: float,
    ) -> Dict[str, float | bool]:
        pos_next, vel_next = self._predict_next_from_accel(accel)
        dist_next = float(np.linalg.norm(pos_next[i] - pos_next[j]))
        rel_vel_next = np.asarray(vel_next[i] - vel_next[j], dtype=np.float32)
        radial_speed_next = float(np.dot(rel_vel_next, direction))
        closing_next = max(-radial_speed_next, 0.0)
        if dist_cur <= d_hard:
            allowed_closing = 0.0
        elif ttc_limit > 0.0:
            allowed_closing = max((dist_cur - d_hard) / ttc_limit, 0.0)
        else:
            allowed_closing = 0.0
        if dist_cur <= d_hard + 1e-6:
            ttc_safe = closing_next <= max(closing_speed_thresh, 1e-6)
        elif closing_next <= max(closing_speed_thresh, 1e-6):
            ttc_safe = True
        else:
            ttc_safe = closing_next <= max(allowed_closing, closing_speed_thresh) + 1e-6
        is_safe = (dist_next >= d_hard - 1e-6) and ttc_safe
        return {
            "dist_next": dist_next,
            "closing_next": closing_next,
            "allowed_closing": allowed_closing,
            "is_safe": is_safe,
        }

    def _select_pairwise_ttc_target(self, accel: np.ndarray, d_hard: float) -> Dict[str, object] | None:
        cfg = self.cfg
        trigger_dist = self._pairwise_trigger_distance(d_hard)
        ttc_limit = self._pairwise_trigger_ttc()
        closing_speed_thresh = self._pairwise_closing_speed_threshold()
        pos_next, vel_next = self._predict_next_from_accel(accel)
        best: Dict[str, object] | None = None

        for i in range(cfg.num_uav):
            for j in range(i + 1, cfg.num_uav):
                diff_cur = np.asarray(self.uav_pos[i] - self.uav_pos[j], dtype=np.float32)
                dist_cur = float(np.linalg.norm(diff_cur))
                diff_next = np.asarray(pos_next[i] - pos_next[j], dtype=np.float32)
                dist_next = float(np.linalg.norm(diff_next))
                direction = self._pairwise_correction_direction(diff_cur, i, j)
                rel_vel_next = np.asarray(vel_next[i] - vel_next[j], dtype=np.float32)
                closing_next = max(-float(np.dot(rel_vel_next, direction)), 0.0)
                immediate = dist_cur < d_hard or dist_next < d_hard
                ttc_to_hard = float("inf")
                triggered = immediate
                if not triggered and dist_cur <= trigger_dist and ttc_limit > 0.0 and closing_next > closing_speed_thresh:
                    ttc_to_hard = (dist_cur - d_hard) / max(closing_next, 1e-6)
                    triggered = ttc_to_hard < ttc_limit
                if not triggered:
                    continue
                priority = (0, dist_next, dist_cur) if immediate else (1, ttc_to_hard, dist_cur)
                candidate = {
                    "i": i,
                    "j": j,
                    "direction": direction,
                    "dist_cur": dist_cur,
                    "dist_next": dist_next,
                    "closing_next": closing_next,
                    "ttc_to_hard": ttc_to_hard,
                    "ttc_limit": ttc_limit,
                    "closing_speed_thresh": closing_speed_thresh,
                    "priority": priority,
                }
                if best is None or priority < best["priority"]:
                    best = candidate

        return best

    def _apply_boundary_hard_filter(
        self,
        accel: np.ndarray,
        indices: List[int] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.cfg
        zero_stats = {
            "filter_active_ratio": 0.0,
            "projected_delta_norm_mean": 0.0,
            "fallback_count": 0.0,
            "boundary_filter_count": 0.0,
            "pairwise_filter_count": 0.0,
        }
        if not bool(getattr(cfg, "boundary_hard_filter_enabled", False)):
            return accel, zero_stats

        margin = self._boundary_margin()
        lower = margin
        upper = float(cfg.map_size) - margin
        accel_safe = np.asarray(accel, dtype=np.float32).copy()
        if indices is None:
            target_indices = list(range(cfg.num_uav))
        else:
            target_indices = [int(idx) for idx in indices]
        delta_norms = np.zeros((len(target_indices),), dtype=np.float32)
        boundary_filter_count = 0
        fallback_count = 0

        for offset, i in enumerate(target_indices):
            accel_before = accel_safe[i].copy()
            adjusted = False
            fallback_used = False
            for axis in range(2):
                accel_axis, axis_adjusted, axis_fallback = self._project_axis_to_boundary(
                    float(self.uav_pos[i, axis]),
                    float(self.uav_vel[i, axis]),
                    float(accel_safe[i, axis]),
                    lower,
                    upper,
                )
                accel_safe[i, axis] = accel_axis
                adjusted = adjusted or axis_adjusted
                fallback_used = fallback_used or axis_fallback
            accel_safe[i] = _project_l2_ball_np(accel_safe[i], cfg.a_max)
            delta_norms[offset] = float(np.linalg.norm(accel_safe[i] - accel_before))
            if adjusted:
                boundary_filter_count += 1
            if fallback_used:
                fallback_count += 1

        stats = {
            "filter_active_ratio": float(boundary_filter_count) / float(max(len(target_indices), 1)),
            "projected_delta_norm_mean": float(np.mean(delta_norms)) if delta_norms.size else 0.0,
            "fallback_count": float(fallback_count),
            "boundary_filter_count": float(boundary_filter_count),
            "pairwise_filter_count": 0.0,
        }
        return accel_safe, stats

    def _resolve_pairwise_violation(
        self,
        accel: np.ndarray,
        i: int,
        j: int,
        d_hard: float,
    ) -> Tuple[np.ndarray, bool, bool, bool]:
        cfg = self.cfg
        pos_next, _ = self._predict_next_from_accel(accel)
        diff_next = np.asarray(pos_next[i] - pos_next[j], dtype=np.float32)
        dist_next = float(np.linalg.norm(diff_next))
        if dist_next >= d_hard:
            return accel, False, False, False

        direction = self._pairwise_correction_direction(diff_next, i, j)
        tau = max(float(cfg.tau0), 1e-6)
        gap = max(d_hard - dist_next, 0.0)
        required_push = gap / max(2.0 * tau * tau, 1e-6)

        accel_candidate = np.asarray(accel, dtype=np.float32).copy()
        accel_candidate[i] = _project_l2_ball_np(accel_candidate[i] + required_push * direction, cfg.a_max)
        accel_candidate[j] = _project_l2_ball_np(accel_candidate[j] - required_push * direction, cfg.a_max)
        accel_candidate, _ = self._apply_boundary_hard_filter(accel_candidate, indices=[i, j])
        pos_candidate, _ = self._predict_next_from_accel(accel_candidate)
        dist_candidate = float(np.linalg.norm(pos_candidate[i] - pos_candidate[j]))
        if dist_candidate >= d_hard:
            return accel_candidate, True, False, False

        accel_fallback = np.asarray(accel, dtype=np.float32).copy()
        accel_fallback[i] = _project_l2_ball_np(direction * cfg.a_max, cfg.a_max)
        accel_fallback[j] = _project_l2_ball_np(-direction * cfg.a_max, cfg.a_max)
        accel_fallback, _ = self._apply_boundary_hard_filter(accel_fallback, indices=[i, j])
        pos_fallback, _ = self._predict_next_from_accel(accel_fallback)
        dist_fallback = float(np.linalg.norm(pos_fallback[i] - pos_fallback[j]))
        if dist_fallback + 1e-6 >= dist_candidate:
            return accel_fallback, True, True, True
        return accel_candidate, True, True, False

    def _resolve_pairwise_ttc_violation(
        self,
        accel: np.ndarray,
        pair_info: Dict[str, object],
        d_hard: float,
    ) -> Tuple[np.ndarray, bool, bool, bool]:
        cfg = self.cfg
        i = int(pair_info["i"])
        j = int(pair_info["j"])
        direction = np.asarray(pair_info["direction"], dtype=np.float32)
        dist_cur = float(pair_info["dist_cur"])
        ttc_limit = float(pair_info["ttc_limit"])
        closing_speed_thresh = float(pair_info["closing_speed_thresh"])
        base_eval = self._evaluate_pairwise_ttc_resolution(
            accel,
            i,
            j,
            d_hard,
            direction,
            dist_cur,
            ttc_limit,
            closing_speed_thresh,
        )
        if bool(base_eval["is_safe"]):
            return accel, False, False, False

        tau = max(float(cfg.tau0), 1e-6)
        delta_closing = max(float(base_eval["closing_next"]) - float(base_eval["allowed_closing"]), 0.0)
        required_push = delta_closing / max(2.0 * tau, 1e-6)
        if float(base_eval["dist_next"]) < d_hard:
            gap = max(d_hard - float(base_eval["dist_next"]), 0.0)
            required_push = max(required_push, gap / max(2.0 * tau * tau, 1e-6))

        accel_candidate = np.asarray(accel, dtype=np.float32).copy()
        accel_candidate[i] = _project_l2_ball_np(accel_candidate[i] + required_push * direction, cfg.a_max)
        accel_candidate[j] = _project_l2_ball_np(accel_candidate[j] - required_push * direction, cfg.a_max)
        accel_candidate, _ = self._apply_boundary_hard_filter(accel_candidate, indices=[i, j])
        candidate_eval = self._evaluate_pairwise_ttc_resolution(
            accel_candidate,
            i,
            j,
            d_hard,
            direction,
            dist_cur,
            ttc_limit,
            closing_speed_thresh,
        )
        if bool(candidate_eval["is_safe"]):
            return accel_candidate, True, False, False

        accel_fallback = np.asarray(accel, dtype=np.float32).copy()
        accel_fallback[i] = _project_l2_ball_np(direction * cfg.a_max, cfg.a_max)
        accel_fallback[j] = _project_l2_ball_np(-direction * cfg.a_max, cfg.a_max)
        accel_fallback, _ = self._apply_boundary_hard_filter(accel_fallback, indices=[i, j])
        fallback_eval = self._evaluate_pairwise_ttc_resolution(
            accel_fallback,
            i,
            j,
            d_hard,
            direction,
            dist_cur,
            ttc_limit,
            closing_speed_thresh,
        )
        if bool(fallback_eval["is_safe"]):
            return accel_fallback, True, True, True
        if (
            float(fallback_eval["dist_next"]) > float(candidate_eval["dist_next"]) + 1e-6
            or (
                abs(float(fallback_eval["dist_next"]) - float(candidate_eval["dist_next"])) <= 1e-6
                and float(fallback_eval["closing_next"]) <= float(candidate_eval["closing_next"]) + 1e-6
            )
        ):
            return accel_fallback, True, True, True
        return accel_candidate, True, True, False

    def _apply_pairwise_hard_filter_distance(self, accel: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.cfg
        accel_in = np.asarray(accel, dtype=np.float32)
        accel_safe = accel_in.copy()
        d_hard = self._pairwise_hard_distance()
        max_passes = max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        total_pairs = max(cfg.num_uav * (cfg.num_uav - 1) // 2, 1)
        touched_pairs: set[Tuple[int, int]] = set()
        pairwise_filter_count = 0
        pairwise_fallback_count = 0
        pairwise_candidate_infeasible_count = 0

        for _ in range(max_passes):
            pos_next, _ = self._predict_next_from_accel(accel_safe)
            pair_order: List[Tuple[float, int, int]] = []
            for i in range(cfg.num_uav):
                for j in range(i + 1, cfg.num_uav):
                    dist_next = float(np.linalg.norm(pos_next[i] - pos_next[j]))
                    if dist_next < d_hard:
                        pair_order.append((dist_next, i, j))
            if not pair_order:
                break
            pair_order.sort(key=lambda item: item[0])
            changed_in_pass = False
            for _, i, j in pair_order:
                pos_cur, _ = self._predict_next_from_accel(accel_safe)
                dist_cur = float(np.linalg.norm(pos_cur[i] - pos_cur[j]))
                if dist_cur >= d_hard:
                    continue
                accel_next, adjusted, candidate_infeasible, used_fallback = self._resolve_pairwise_violation(
                    accel_safe,
                    i,
                    j,
                    d_hard,
                )
                if not adjusted:
                    continue
                if np.allclose(accel_next, accel_safe, atol=1e-6):
                    continue
                changed_in_pass = True
                accel_safe = accel_next
                pairwise_filter_count += 1
                pairwise_candidate_infeasible_count += int(candidate_infeasible)
                pairwise_fallback_count += int(used_fallback)
                touched_pairs.add((i, j))
            if not changed_in_pass:
                break

        delta_norm = float(np.mean(np.linalg.norm(accel_safe - accel_in, axis=1))) if cfg.num_uav > 0 else 0.0
        stats = {
            "pairwise_filter_count": float(pairwise_filter_count),
            "pairwise_filter_active_ratio": float(len(touched_pairs)) / float(total_pairs),
            "pairwise_projected_delta_norm": delta_norm,
            "pairwise_fallback_count": float(pairwise_fallback_count),
            "pairwise_candidate_infeasible_count": float(pairwise_candidate_infeasible_count),
        }
        return accel_safe, stats

    def _apply_pairwise_hard_filter_ttc(self, accel: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.cfg
        accel_in = np.asarray(accel, dtype=np.float32)
        accel_safe = accel_in.copy()
        d_hard = self._pairwise_hard_distance()
        single_pair_only = bool(getattr(cfg, "pairwise_hard_single_pair_only", True))
        max_passes = 1 if single_pair_only else max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        total_pairs = max(cfg.num_uav * (cfg.num_uav - 1) // 2, 1)
        touched_pairs: set[Tuple[int, int]] = set()
        pairwise_filter_count = 0
        pairwise_fallback_count = 0
        pairwise_candidate_infeasible_count = 0

        for _ in range(max_passes):
            pair_info = self._select_pairwise_ttc_target(accel_safe, d_hard)
            if pair_info is None:
                break
            i = int(pair_info["i"])
            j = int(pair_info["j"])
            accel_next, adjusted, candidate_infeasible, used_fallback = self._resolve_pairwise_ttc_violation(
                accel_safe,
                pair_info,
                d_hard,
            )
            if not adjusted or np.allclose(accel_next, accel_safe, atol=1e-6):
                break
            accel_safe = accel_next
            pairwise_filter_count += 1
            pairwise_candidate_infeasible_count += int(candidate_infeasible)
            pairwise_fallback_count += int(used_fallback)
            touched_pairs.add((i, j))
            if single_pair_only:
                break

        delta_norm = float(np.mean(np.linalg.norm(accel_safe - accel_in, axis=1))) if cfg.num_uav > 0 else 0.0
        stats = {
            "pairwise_filter_count": float(pairwise_filter_count),
            "pairwise_filter_active_ratio": float(len(touched_pairs)) / float(total_pairs),
            "pairwise_projected_delta_norm": delta_norm,
            "pairwise_fallback_count": float(pairwise_fallback_count),
            "pairwise_candidate_infeasible_count": float(pairwise_candidate_infeasible_count),
        }
        return accel_safe, stats

    def _apply_pairwise_hard_filter(self, accel: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.cfg
        zero_stats = {
            "pairwise_filter_count": 0.0,
            "pairwise_filter_active_ratio": 0.0,
            "pairwise_projected_delta_norm": 0.0,
            "pairwise_fallback_count": 0.0,
            "pairwise_candidate_infeasible_count": 0.0,
        }
        if not bool(getattr(cfg, "pairwise_hard_filter_enabled", False)):
            return accel, zero_stats
        if self._pairwise_trigger_mode() == "ttc":
            return self._apply_pairwise_hard_filter_ttc(accel)
        return self._apply_pairwise_hard_filter_distance(accel)

    def _reset_safety_shield_stats(self, *, status: str = "disabled") -> None:
        self.last_safety_shield_active = 0.0
        self.last_safety_shield_feasible = 1.0
        self.last_safety_shield_delta_norm = 0.0
        self.last_safety_shield_delta_norm_max = 0.0
        self.last_safety_shield_min_margin_before = 0.0
        self.last_safety_shield_min_margin_after = 0.0
        self.last_safety_shield_min_distance_after = 0.0
        self.last_safety_shield_pair_count = 0.0
        self.last_safety_shield_status = status
        self.last_safety_shield_solver = ""

    @staticmethod
    def _finite_safety_metric(value: float) -> float:
        metric = float(value)
        return metric if np.isfinite(metric) else 0.0

    def _apply_safety_shield(self, accel: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        accel_in = np.asarray(accel, dtype=np.float32)
        if not bool(getattr(cfg, "safety_shield_enabled", False)):
            self._reset_safety_shield_stats(status="disabled")
            return accel_in
        if int(getattr(cfg, "num_uav", 0) or 0) < 2:
            self._reset_safety_shield_stats(status="skipped")
            return accel_in

        result = solve_brake_distance_shield(
            pos=np.asarray(self.uav_pos, dtype=np.float32),
            vel=np.asarray(self.uav_vel, dtype=np.float32),
            nominal_accel=accel_in,
            cfg=cfg,
        )
        self.last_safety_shield_active = 1.0 if result.active else 0.0
        self.last_safety_shield_feasible = 1.0 if result.feasible else 0.0
        self.last_safety_shield_delta_norm = self._finite_safety_metric(result.delta_norm_mean)
        self.last_safety_shield_delta_norm_max = self._finite_safety_metric(result.delta_norm_max)
        self.last_safety_shield_min_margin_before = self._finite_safety_metric(result.min_margin_before)
        self.last_safety_shield_min_margin_after = self._finite_safety_metric(result.min_margin_after)
        self.last_safety_shield_min_distance_after = self._finite_safety_metric(result.min_distance_after)
        self.last_safety_shield_pair_count = float(result.pair_count)
        self.last_safety_shield_status = result.status
        self.last_safety_shield_solver = result.solver
        return np.asarray(result.accel, dtype=np.float32)

    def _apply_uav_dynamics(
        self,
        actions: Dict[str, Dict] | np.ndarray | Sequence[np.ndarray],
        *,
        refresh_cache: bool = True,
    ) -> None:
        cfg = self.cfg
        accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        policy_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        accel_action_matrix: np.ndarray | None = None
        if not isinstance(actions, dict):
            if isinstance(actions, np.ndarray):
                accel_action_matrix = np.asarray(actions, dtype=np.float32)
                if accel_action_matrix.ndim == 1:
                    if int(cfg.num_uav) != 1:
                        raise ValueError(
                            f"1D accel action matrix is only valid when num_uav=1, got num_uav={cfg.num_uav}."
                        )
                    accel_action_matrix = accel_action_matrix.reshape(1, -1)
            else:
                accel_action_matrix = np.stack(
                    [np.asarray(action, dtype=np.float32) for action in actions],
                    axis=0,
                )
            if accel_action_matrix.shape != (cfg.num_uav, 2):
                raise ValueError(f"accel action matrix shape must be ({cfg.num_uav}, 2)")
        use_avoidance = ablation_flag(
            cfg,
            "use_avoidance_layer",
            fallback_attr="avoidance_enabled",
            default=False,
        )
        use_energy_safety = ablation_flag(
            cfg,
            "use_energy_safety_layer",
            fallback_attr="energy_safety_enabled",
            default=False,
        )
        d_alert = cfg.avoidance_alert_factor * cfg.d_safe if use_avoidance else 0.0
        raw_prealert_factor = getattr(cfg, "avoidance_prealert_factor", None)
        d_prealert = 0.0
        if use_avoidance and raw_prealert_factor is not None:
            d_prealert = max(float(raw_prealert_factor) * cfg.d_safe, d_alert)
        prealert_mode = str(getattr(cfg, "avoidance_prealert_mode", "distance") or "distance").strip().lower()
        if prealert_mode not in {"distance", "ttc"}:
            prealert_mode = "distance"
        closing_speed_thresh = max(float(getattr(cfg, "avoidance_prealert_closing_speed", 0.0) or 0.0), 0.0)
        prealert_ttc_limit = max(float(getattr(cfg, "avoidance_prealert_ttc", 0.0) or 0.0), 0.0)
        raw_prealert_dist_cap = getattr(cfg, "avoidance_prealert_dist_cap", None)
        prealert_trigger_dist = d_prealert
        if use_avoidance and prealert_mode == "ttc":
            if raw_prealert_dist_cap is not None:
                prealert_trigger_dist = max(float(raw_prealert_dist_cap), d_alert)
            elif d_prealert > 0.0:
                prealert_trigger_dist = d_prealert
        repulse_mode = str(getattr(cfg, "avoidance_repulse_mode", "inverse") or "inverse").strip().lower()
        closing_gain_enabled = bool(getattr(cfg, "avoidance_closing_gain_enabled", False))
        closing_gain_cap = max(float(getattr(cfg, "avoidance_closing_gain_cap", 2.0) or 2.0), 1.0)
        closing_gain_top1_only = bool(getattr(cfg, "avoidance_closing_gain_top1_only", False))
        eta_avoid = float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta))
        _, _, centroid_transfer_ratio = self._centroid_anneal_state()
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        if cross_enabled:
            avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)
            eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * centroid_transfer_ratio)
            eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
            eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
            eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
            eta_max = max(eta_min, eta_max)
            eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))
        self.last_avoidance_eta_exec = float(eta_avoid)
        for i, agent in enumerate(self.agents):
            if accel_action_matrix is None:
                a = np.array(actions[agent]["accel"], dtype=np.float32)
            else:
                a = np.asarray(accel_action_matrix[i], dtype=np.float32)
            a = _project_l2_ball_np(np.clip(a, -1.0, 1.0), 1.0) * float(cfg.a_max)
            policy_accel[i] = a.copy()
            a_rep = np.zeros(2, dtype=np.float32)
            if use_avoidance and d_alert > 0.0:
                pair_terms = []
                for j in range(cfg.num_uav):
                    if i == j:
                        continue
                    diff = self.uav_pos[i] - self.uav_pos[j]
                    dist = float(np.linalg.norm(diff))
                    if dist <= 1e-6:
                        continue
                    rel_vel = self.uav_vel[i] - self.uav_vel[j]
                    closing_speed = float(-(np.dot(diff, rel_vel) / dist))
                    trigger_dist = d_alert
                    in_core_alert = dist < d_alert
                    in_prealert = False
                    ttc_to_alert = float("inf")
                    if prealert_mode == "ttc":
                        if (
                            prealert_trigger_dist > d_alert
                            and dist < prealert_trigger_dist
                            and closing_speed > closing_speed_thresh
                            and prealert_ttc_limit > 0.0
                        ):
                            ttc_to_alert = (dist - d_alert) / max(closing_speed, 1e-6)
                            in_prealert = ttc_to_alert < prealert_ttc_limit
                    else:
                        in_prealert = (
                            d_prealert > d_alert
                            and dist < d_prealert
                            and closing_speed > closing_speed_thresh
                        )
                    if in_prealert and not in_core_alert:
                        trigger_dist = prealert_trigger_dist if prealert_mode == "ttc" else d_prealert
                    if in_core_alert or in_prealert:
                        direction = diff / dist
                        if repulse_mode == "linear":
                            denom = max(trigger_dist - cfg.d_safe, 1e-6)
                            strength = float(np.clip((trigger_dist - dist) / denom, 0.0, 1.0))
                        elif repulse_mode == "quadratic":
                            denom = max(trigger_dist - cfg.d_safe, 1e-6)
                            base = float(np.clip((trigger_dist - dist) / denom, 0.0, 1.0))
                            strength = base * base
                        else:
                            strength = (1.0 / dist - 1.0 / trigger_dist)
                        closing_ratio_raw = 1.0
                        closing_gain = 1.0
                        if closing_gain_enabled and closing_speed_thresh > 1e-6 and closing_speed > closing_speed_thresh:
                            closing_ratio_raw = closing_speed / closing_speed_thresh
                            closing_gain = float(np.clip(closing_ratio_raw, 1.0, closing_gain_cap))
                        pair_terms.append(
                            {
                                "direction": direction,
                                "strength": strength,
                                "closing_gain": closing_gain,
                                "closing_ratio_raw": closing_ratio_raw,
                                "closing_bonus_score": strength * max(closing_ratio_raw - 1.0, 0.0),
                                "in_core_alert": in_core_alert,
                                "ttc_urgency": (1.0 / max(ttc_to_alert, 1e-6)) if np.isfinite(ttc_to_alert) else 0.0,
                                "dist": dist,
                            }
                        )
                top1_gain_idx = None
                if closing_gain_enabled and closing_gain_top1_only and pair_terms:
                    best_key = None
                    for idx, term in enumerate(pair_terms):
                        if float(term["closing_gain"]) <= 1.0:
                            continue
                        key = (
                            1 if bool(term["in_core_alert"]) else 0,
                            float(term["closing_bonus_score"]),
                            float(term["ttc_urgency"]),
                            float(term["strength"]),
                            -float(term["dist"]),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            top1_gain_idx = idx
                for idx, term in enumerate(pair_terms):
                    closing_gain = float(term["closing_gain"])
                    if closing_gain_top1_only and top1_gain_idx is not None and idx != top1_gain_idx:
                        closing_gain = 1.0
                    a_rep += eta_avoid * float(term["strength"]) * closing_gain * np.asarray(term["direction"], dtype=np.float32)
                if bool(getattr(cfg, "avoidance_repulse_clip", True)):
                    a_rep = _project_l2_ball_np(a_rep, cfg.a_max)
            if cfg.energy_enabled and use_energy_safety:
                v_next = self.uav_vel[i] + a * cfg.tau0
                speed_next = float(np.linalg.norm(v_next))
                est_energy = self.uav_energy[i] - float(self._fly_power(speed_next)) * cfg.tau0
                safe_threshold = cfg.energy_safe_threshold * cfg.uav_energy_init
                if est_energy < safe_threshold:
                    cur_speed = float(np.linalg.norm(self.uav_vel[i]))
                    if cur_speed > 1e-6:
                        direction = self.uav_vel[i] / cur_speed
                    else:
                        a_norm = float(np.linalg.norm(a))
                        if a_norm > 1e-6:
                            direction = a / a_norm
                        else:
                            direction = np.zeros(2, dtype=np.float32)
                    target_delta = cfg.uav_opt_speed - cur_speed
                    a = direction * np.clip(target_delta / max(cfg.tau0, 1e-6), -cfg.a_max, cfg.a_max)
            a = a + a_rep
            a = _project_l2_ball_np(a, cfg.a_max)
            accel[i] = a
        accel_before_hard_filter = accel.copy()
        accel, boundary_stats = self._apply_boundary_hard_filter(accel)
        accel, pairwise_stats = self._apply_pairwise_hard_filter(accel)
        accel_after_hard_filter = accel.copy()
        hard_delta_norm = (
            np.linalg.norm(accel_after_hard_filter - accel_before_hard_filter, axis=1)
            if cfg.num_uav > 0
            else np.zeros((0,), dtype=np.float32)
        )
        accel = self._apply_safety_shield(accel_after_hard_filter)
        hard_active_count = int(np.count_nonzero(hard_delta_norm > 1e-6))
        self.last_filter_active_ratio = float(hard_active_count) / float(max(cfg.num_uav, 1))
        self.last_projected_delta_norm_mean = float(np.mean(hard_delta_norm)) if hard_delta_norm.size else 0.0
        self.last_fallback_count = float(boundary_stats["fallback_count"] + pairwise_stats["pairwise_fallback_count"])
        self.last_boundary_filter_count = float(boundary_stats["boundary_filter_count"])
        self.last_pairwise_filter_count = float(pairwise_stats["pairwise_filter_count"])
        self.last_pairwise_filter_active_ratio = float(pairwise_stats["pairwise_filter_active_ratio"])
        self.last_pairwise_projected_delta_norm = float(pairwise_stats["pairwise_projected_delta_norm"])
        self.last_pairwise_fallback_count = float(pairwise_stats["pairwise_fallback_count"])
        self.last_pairwise_candidate_infeasible_count = float(pairwise_stats["pairwise_candidate_infeasible_count"])
        self.last_policy_accel = policy_accel
        self.last_exec_accel = accel.copy()
        uav_vel_next = _project_l2_ball_np(self.uav_vel + accel * cfg.tau0, cfg.v_max)
        uav_pos_next = (np.asarray(self.uav_pos, dtype=np.float32) + uav_vel_next * cfg.tau0).astype(
            np.float32,
            copy=False,
        )
        if cfg.boundary_mode == "reflect":
            for i in range(cfg.num_uav):
                for axis in range(2):
                    if uav_pos_next[i, axis] < 0.0:
                        uav_pos_next[i, axis] = -uav_pos_next[i, axis]
                        uav_vel_next[i, axis] = -uav_vel_next[i, axis]
                    elif uav_pos_next[i, axis] > cfg.map_size:
                        uav_pos_next[i, axis] = 2 * cfg.map_size - uav_pos_next[i, axis]
                        uav_vel_next[i, axis] = -uav_vel_next[i, axis]
        self.uav_vel = uav_vel_next
        self.uav_pos = np.clip(uav_pos_next, 0.0, cfg.map_size).astype(np.float32, copy=False)
        if refresh_cache:
            self._refresh_uav_cache()
        else:
            self._cached_uav_ecef = None
            self._cached_uav_vel_ecef = None
            self._cached_elevation_t = None
            self._cached_elevation_matrix = None
            self._cached_backhaul_loss_t = None
            self._cached_backhaul_loss_matrix = None
            self._cached_uav_neighbor_t = None
            self._cached_uav_neighbor_order = None
            self._cached_obs_runtime_context = None

    def _invalidate_step_caches(self) -> None:
        self._cached_orbit_t = None
        self._cached_orbit_pos = None
        self._cached_orbit_vel = None
        self._cached_elevation_t = None
        self._cached_elevation_matrix = None
        self._cached_backhaul_loss_t = None
        self._cached_backhaul_loss_matrix = None
        self._cached_uav_ecef = None
        self._cached_uav_vel_ecef = None
        self._cached_uav_neighbor_t = None
        self._cached_uav_neighbor_order = None
        self._cached_obs_runtime_context = None

    def _get_orbit_states(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cached_orbit_t != self.t or self._cached_orbit_pos is None:
            t_idx = int(self.t)
            if 0 <= t_idx < self._orbit_pos_table.shape[0]:
                self._cached_orbit_pos = self._orbit_pos_table[t_idx]
                self._cached_orbit_vel = self._orbit_vel_table[t_idx]
            else:
                self._cached_orbit_pos, self._cached_orbit_vel = self.orbit.get_states(self.t * self.cfg.tau0)
            self._cached_orbit_t = self.t
        return self._cached_orbit_pos, self._cached_orbit_vel

    def _refresh_uav_cache(self) -> None:
        cfg = self.cfg
        uav_ecef = np.zeros((cfg.num_uav, 3), dtype=np.float32)
        uav_vel_ecef = np.zeros((cfg.num_uav, 3), dtype=np.float32)
        for u in range(cfg.num_uav):
            x = float(self.uav_pos[u, 0])
            y = float(self.uav_pos[u, 1])
            lat, lon = self._local_to_latlon(x, y)
            r = cfg.r_earth + cfg.uav_height
            cos_lat = math.cos(lat)
            sin_lat = math.sin(lat)
            cos_lon = math.cos(lon)
            sin_lon = math.sin(lon)
            uav_ecef[u] = np.array(
                [
                    r * cos_lat * cos_lon,
                    r * cos_lat * sin_lon,
                    r * sin_lat,
                ],
                dtype=np.float32,
            )
            uav_vel_ecef[u] = self._enu_to_ecef(
                float(self.uav_vel[u, 0]),
                float(self.uav_vel[u, 1]),
                0.0,
                lat,
                lon,
            )
        self._cached_uav_ecef = uav_ecef
        self._cached_uav_vel_ecef = uav_vel_ecef
        self._cached_elevation_t = None
        self._cached_elevation_matrix = None
        self._cached_backhaul_loss_t = None
        self._cached_backhaul_loss_matrix = None
        self._cached_uav_neighbor_t = None
        self._cached_uav_neighbor_order = None

    def _ensure_neighbor_cache(self) -> None:
        if self._cached_uav_neighbor_t == self.t and self._cached_uav_neighbor_order is not None:
            return
        diff = self.uav_pos[:, None, :] - self.uav_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        self._cached_uav_neighbor_order = np.argsort(dist, axis=1)
        self._cached_uav_neighbor_t = self.t

    def _associate_users(self) -> np.ndarray:
        cfg = self.cfg
        K = cfg.num_gu
        assoc = np.full((K,), -1, dtype=np.int32)

        if K <= 0:
            return assoc

        diff = self.gu_pos[:, None, :] - self.uav_pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        best = np.argmin(dist2, axis=1)
        return best.astype(np.int32)

    def _build_candidate_users(self, assoc: np.ndarray) -> List[List[int]]:
        cfg = self.cfg
        candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        mode = str(getattr(cfg, "candidate_mode", "assoc")).lower()
        max_keep = int(getattr(cfg, "candidate_k", 0) or 0)
        if max_keep <= 0:
            max_keep = cfg.users_obs_max
        else:
            max_keep = min(max_keep, cfg.users_obs_max)
        if mode == "assoc":
            for k, u in enumerate(assoc):
                if u >= 0:
                    candidates[u].append(k)

            # Limit to max_keep by queue (descending)
            for u in range(cfg.num_uav):
                if len(candidates[u]) > max_keep:
                    qs = [(k, self.gu_queue[k]) for k in candidates[u]]
                    qs.sort(key=lambda x: x[1], reverse=True)
                    candidates[u] = [k for k, _ in qs[: max_keep]]
            return candidates

        if cfg.num_gu <= 0:
            return candidates

        use_radius = mode in ("radius", "dist", "distance")
        radius = getattr(cfg, "candidate_radius", None)
        for u in range(cfg.num_uav):
            d2d = np.linalg.norm(self.gu_pos - self.uav_pos[u], axis=1)
            if use_radius and radius is not None and radius > 0:
                idx = np.nonzero(d2d <= radius)[0]
                if idx.size > 0:
                    idx = idx[np.argsort(d2d[idx])]
                else:
                    idx = np.argsort(d2d)
            else:
                idx = np.argsort(d2d)
            if idx.size > max_keep:
                idx = idx[: max_keep]
            candidates[u] = idx.tolist()
        return candidates

    def _compute_access_link_gain_matrix(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_gu <= 0 or cfg.num_uav <= 0:
            return np.zeros((cfg.num_gu, cfg.num_uav), dtype=np.float32)

        diff = self.gu_pos[:, None, :] - self.uav_pos[None, :, :]
        d2d = np.linalg.norm(diff, axis=2)
        d3d = np.sqrt(d2d * d2d + self._uav_height_sq)
        phi = np.arcsin(np.clip(cfg.uav_height / geometry_denominator(d3d), -1.0, 1.0))
        pl = _quantize_access_pathloss_db_np(
            cfg,
            channel.pathloss_db(
                d3d,
                phi,
                cfg,
                carrier_freq_hz=_access_carrier_freq_from_cfg(cfg),
            ),
        )
        gain = 10 ** (-pl / 10.0)
        if cfg.fading_enabled and channel.access_fading_mode_from_config(cfg) == "iid_rician":
            gain = gain * channel.rician_power_gain(
                channel.rician_k_linear_from_config(cfg),
                size=gain.shape,
                rng=self.rng,
            )
        gain = _quantize_access_gain_snapshot_np(cfg, gain)
        return np.asarray(gain, dtype=np.float32)

    def _sample_access_channel_snapshot(self) -> AccessChannelSnapshot:
        return AccessChannelSnapshot(
            gain_matrix=np.asarray(self._compute_access_link_gain_matrix(), dtype=np.float32),
        )

    @staticmethod
    def _coerce_access_gain_matrix(
        access_snapshot: AccessChannelSnapshot | np.ndarray | None,
    ) -> np.ndarray | None:
        if access_snapshot is None:
            return None
        if isinstance(access_snapshot, AccessChannelSnapshot):
            return np.asarray(access_snapshot.gain_matrix, dtype=np.float32)
        return np.asarray(access_snapshot, dtype=np.float32)

    def _compute_access_interference_power(
        self,
        assoc: np.ndarray,
        gain_matrix: np.ndarray,
        gu_band_fraction: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        if not cfg.interference_enabled or gain_matrix.size == 0:
            return np.zeros((cfg.num_uav,), dtype=np.float32)

        interference = compute_access_interference_beta_continuous(
            assoc,
            gain_matrix,
            gu_band_fraction,
            gu_tx_power=float(cfg.gu_tx_power),
            num_uav=int(cfg.num_uav),
            interference_enabled=bool(cfg.interference_enabled),
        )
        return _quantize_numeric_contract_np(
            cfg,
            interference,
            quantum=_access_interference_quantum_np(cfg),
            dtype=np.float32,
        )

    def _compute_access_rates(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        actions: Dict[str, Dict] | np.ndarray | Sequence[np.ndarray],
        record_exec: bool = True,
        *,
        access_snapshot: AccessChannelSnapshot | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        gain_matrix = self._coerce_access_gain_matrix(access_snapshot)
        if gain_matrix is None:
            gain_matrix = self._sample_access_channel_snapshot().gain_matrix

        eta_quantum = _semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6)
        rate_quantum = _semantic_quantum(cfg, "structured_access_rate_quantum", 32.0)
        rates = np.zeros((cfg.num_gu,), dtype=np.float32)
        eta = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        bw_align_sum = 0.0
        bw_align_count = 0
        exec_bw = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
        gu_band_fraction = np.zeros((cfg.num_gu,), dtype=np.float32)
        cand_idx_by_u: List[np.ndarray] = [np.zeros((0,), dtype=np.int32) for _ in range(cfg.num_uav)]
        assoc_mask_by_u: List[np.ndarray] = [np.zeros((0,), dtype=bool) for _ in range(cfg.num_uav)]
        betas_by_u: List[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(cfg.num_uav)]
        valid_full_by_u: List[np.ndarray] = [np.zeros((cfg.num_gu,), dtype=bool) for _ in range(cfg.num_uav)]
        bw_action_matrix: np.ndarray | None = None
        if cfg.enable_bw_action:
            if isinstance(actions, dict):
                rows = []
                for u in range(cfg.num_uav):
                    action_data = actions[self.agents[u]]
                    if "bw_alloc" not in action_data and "bw_logits" not in action_data:
                        raise ValueError("BW action dict must provide full-G 'bw_alloc' for each UAV.")
                    bw_raw = action_data.get("bw_alloc", action_data.get("bw_logits"))
                    bw_row = np.asarray(bw_raw, dtype=np.float32).reshape(-1)
                    if int(bw_row.shape[0]) < int(cfg.num_gu):
                        raise ValueError(
                            f"BW action for UAV {u} must have at least num_gu={cfg.num_gu} entries, "
                            f"got {int(bw_row.shape[0])}."
                        )
                    rows.append(bw_row[: cfg.num_gu])
                bw_action_matrix = np.stack(rows, axis=0)
            elif isinstance(actions, np.ndarray):
                bw_action_matrix = np.asarray(actions, dtype=np.float32)
                if bw_action_matrix.ndim == 1:
                    if int(cfg.num_uav) != 1:
                        raise ValueError(
                            f"1D bw action matrix is only valid when num_uav=1, got num_uav={cfg.num_uav}."
                        )
                    bw_action_matrix = bw_action_matrix.reshape(1, -1)
            else:
                bw_action_matrix = np.stack(
                    [np.asarray(action, dtype=np.float32) for action in actions],
                    axis=0,
                )
            if bw_action_matrix.shape != (cfg.num_uav, cfg.num_gu):
                raise ValueError(
                    f"BW action matrix shape must be ({cfg.num_uav}, {cfg.num_gu}), "
                    f"got {tuple(bw_action_matrix.shape)}."
                )
            if record_exec:
                value_check_eps = 1.0e-7
                sum_eps = lambda n: max(1.0e-5, 8.0 * np.finfo(np.float32).eps * max(int(n), 1))
                if not np.all(np.isfinite(bw_action_matrix)):
                    raise ValueError("BW action contains non-finite values.")
                if float(np.min(bw_action_matrix)) < -value_check_eps:
                    raise ValueError("BW action contains negative values beyond tolerance.")
                for u in range(cfg.num_uav):
                    valid_full = assoc == u
                    invalid_mass = float(np.sum(np.clip(bw_action_matrix[u, ~valid_full], 0.0, None)))
                    valid_sum = float(np.sum(bw_action_matrix[u, valid_full]))
                    eps = sum_eps(int(np.sum(valid_full)))
                    if invalid_mass > eps:
                        raise ValueError(f"BW action invalid mass for UAV {u} is {invalid_mass}, tolerance {eps}.")
                    if np.any(valid_full):
                        if abs(valid_sum - 1.0) > eps:
                            raise ValueError(f"BW action valid sum for UAV {u} is {valid_sum}, tolerance {eps}.")
                    elif abs(valid_sum) > eps:
                        raise ValueError(f"BW action valid sum for empty UAV {u} is {valid_sum}, tolerance {eps}.")

        for u in range(cfg.num_uav):
            valid_full = (assoc == u)
            valid_full_by_u[u] = valid_full
            if cfg.enable_bw_action:
                if bw_action_matrix is None:
                    raise RuntimeError("BW action matrix was not materialized for enabled full-G BW action.")
                betas_full = bw_action_matrix[u].astype(np.float32, copy=False)
                betas_full = betas_full * valid_full.astype(np.float32, copy=False)
            else:
                betas_full = np.zeros((cfg.num_gu,), dtype=np.float32)
                if np.any(valid_full):
                    betas_full[valid_full] = 1.0 / float(np.sum(valid_full))
            exec_bw[u] = betas_full.astype(np.float32, copy=False)
            gu_band_fraction[valid_full] = betas_full[valid_full].astype(np.float32, copy=False)

            cand = candidates[u][: cfg.users_obs_max] if u < len(candidates) else []
            if not cand:
                continue
            cand_idx = np.asarray(cand, dtype=np.int32)
            assoc_mask = assoc[cand_idx] == u
            betas = betas_full[cand_idx].astype(np.float32, copy=False)
            cand_idx_by_u[u] = cand_idx
            assoc_mask_by_u[u] = assoc_mask
            betas_by_u[u] = betas

        interference_by_u = self._compute_access_interference_power(assoc, gain_matrix, gu_band_fraction)
        access_noise_figure_db = float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0)
        access_ergodic = bool(cfg.fading_enabled) and channel.access_fading_mode_from_config(cfg) == "ergodic_rician"
        access_k = channel.rician_k_linear_from_config(cfg)
        access_quad_points = int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16)

        for u in range(cfg.num_uav):
            cand_idx = cand_idx_by_u[u]
            assoc_mask = assoc_mask_by_u[u]
            betas = betas_by_u[u]
            interference = float(interference_by_u[u])

            if cand_idx.size > 0:
                gain = gain_matrix[cand_idx, u].astype(np.float32, copy=False)

                # Keep eta as a channel-quality feature on the reference full band.
                ref_snr = channel.snr_linear(
                    cfg.gu_tx_power,
                    gain,
                    cfg.noise_density,
                    cfg.b_acc,
                    interference,
                    noise_figure_db=access_noise_figure_db,
                )
                ref_se = (
                    channel.rician_ergodic_spectral_efficiency(
                        ref_snr,
                        access_k,
                        quadrature_points=access_quad_points,
                    )
                    if access_ergodic
                    else channel.spectral_efficiency(ref_snr)
                )
                se_values = _quantize_numeric_contract_np(
                    cfg,
                    ref_se,
                    quantum=eta_quantum,
                    dtype=np.float32,
                )
                eta[u, : cand_idx.size] = se_values

            assoc_idx = np.flatnonzero(valid_full_by_u[u])
            if assoc_idx.size > 0:
                gain_active = gain_matrix[assoc_idx, u].astype(np.float32, copy=False)
                beta_active = exec_bw[u, assoc_idx].astype(np.float32, copy=False)
                eff_bw = (beta_active * cfg.b_acc).astype(np.float32, copy=False)
                eff_interference = (
                    beta_active * interference if cfg.interference_enabled else np.zeros_like(beta_active)
                )
                eff_snr = channel.snr_linear(
                    cfg.gu_tx_power,
                    gain_active,
                    cfg.noise_density,
                    eff_bw,
                    eff_interference,
                    noise_figure_db=access_noise_figure_db,
                )
                eff_se = (
                    channel.rician_ergodic_spectral_efficiency(
                        eff_snr,
                        access_k,
                        quadrature_points=access_quad_points,
                    )
                    if access_ergodic
                    else channel.spectral_efficiency(eff_snr)
                )
                eff_rate = eff_bw * _quantize_numeric_contract_np(
                    cfg,
                    eff_se,
                    quantum=eta_quantum,
                    dtype=np.float32,
                )
                rates[assoc_idx] = _quantize_numeric_contract_np(
                    cfg,
                    eff_rate,
                    quantum=rate_quantum,
                    dtype=np.float32,
                )

                if cfg.enable_bw_action:
                    q_norm = self.gu_queue[assoc_idx] / normalize_scale(cfg.queue_max_gu)
                    assoc_bonus = 0.2
                    prev_mask = (self.prev_association[assoc_idx] == u).astype(np.float32)
                    ref_snr_align = channel.snr_linear(
                        cfg.gu_tx_power,
                        gain_active,
                        cfg.noise_density,
                        cfg.b_acc,
                        interference,
                        noise_figure_db=access_noise_figure_db,
                    )
                    ref_se_align = (
                        channel.rician_ergodic_spectral_efficiency(
                            ref_snr_align,
                            access_k,
                            quadrature_points=access_quad_points,
                        )
                        if access_ergodic
                        else channel.spectral_efficiency(ref_snr_align)
                    )
                    target_weights = q_norm * (0.5 + ref_se_align) * (1.0 + assoc_bonus * prev_mask)
                    denom = float(np.sum(target_weights))
                    if denom > 0:
                        target = target_weights / denom
                        l1 = float(np.sum(np.abs(beta_active - target)))
                        align = 1.0 - 0.5 * l1
                        bw_align_sum += align
                        bw_align_count += 1

        if record_exec:
            self.last_exec_bw_alloc = exec_bw
            self.last_bw_align = bw_align_sum / max(1, bw_align_count)
            self.last_bw_fraction_by_uav_gu = exec_bw.astype(np.float32, copy=True)
            self.last_access_interference_by_uav = np.asarray(interference_by_u, dtype=np.float32).reshape(cfg.num_uav)
        return rates, eta

    def _projected_sat_connection_count(self, u: int, sat_indices: np.ndarray) -> np.ndarray:
        sat_idx = np.asarray(sat_indices, dtype=np.int32)
        if sat_idx.size == 0:
            return np.zeros((0,), dtype=np.float32)

        load_count = self.last_sat_connection_counts[sat_idx].astype(np.float32, copy=False)
        if u < len(self.last_sat_selection):
            current = np.asarray(self.last_sat_selection[u], dtype=np.int32)
        else:
            current = np.zeros((0,), dtype=np.int32)
        projected_add = (~np.isin(sat_idx, current)).astype(np.float32, copy=False)
        return np.maximum(load_count + projected_add, 1.0).astype(np.float32, copy=False)

    def _projected_sat_bandwidth(self, u: int, sat_indices: np.ndarray) -> np.ndarray:
        projected_count = self._projected_sat_connection_count(u, sat_indices)
        if projected_count.size == 0:
            return projected_count
        return (self._effective_b_backhaul_per_sat() / projected_count).astype(np.float32, copy=False)

    def _update_gu_queues(self, access_rates: np.ndarray, assoc: np.ndarray) -> np.ndarray:
        arrival = self._sample_gu_arrival()
        return self._apply_gu_queue_transition_from_arrival(access_rates, assoc, arrival)

    def _sample_gu_arrival(self, arrival_override: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        arrival_rates = self._current_expected_gu_arrival_rates()
        self.last_arrival_rate = float(np.mean(arrival_rates)) if arrival_rates.size > 0 else 0.0
        self.last_gu_arrival_rate_vec = arrival_rates.astype(np.float32, copy=False)
        hotspot_idx = int(getattr(self, "_hotspot_active_idx", -1))
        hotspot_mask = np.asarray(getattr(self, "_hotspot_member_mask", np.zeros((0, cfg.num_gu), dtype=bool)))
        if hotspot_idx >= 0 and hotspot_idx < hotspot_mask.shape[0]:
            self.last_hotspot_mask = hotspot_mask[hotspot_idx].astype(np.float32, copy=False)
        else:
            self.last_hotspot_mask = np.zeros((cfg.num_gu,), dtype=np.float32)
        if arrival_override is not None:
            arrival = np.asarray(arrival_override, dtype=np.float32)
            if arrival.shape != (cfg.num_gu,):
                raise ValueError(f"arrival_override shape must be ({cfg.num_gu},), got {tuple(arrival.shape)}.")
        elif cfg.task_arrival_poisson:
            arrival = self.rng.poisson(arrival_rates).astype(np.float32)
        else:
            arrival = arrival_rates.astype(np.float32, copy=False)
        self.last_gu_arrival = arrival.astype(np.float32)
        return np.asarray(arrival, dtype=np.float32)

    def _apply_gu_queue_transition_from_arrival(
        self,
        access_rates: np.ndarray,
        assoc: np.ndarray,
        arrival: np.ndarray,
        *,
        traffic_state_after_override: dict | None = None,
        advance_traffic: bool = True,
    ) -> np.ndarray:
        cfg = self.cfg
        arrival_arr = np.asarray(arrival, dtype=np.float32)
        q_before = self.gu_queue + arrival_arr
        service_bits = _quantize_queue_contract_np(
            cfg,
            np.asarray(access_rates, dtype=np.float32) * float(cfg.tau0),
            attr="structured_flow_bits_quantum",
            default=32.0,
        )
        outflow = np.minimum(q_before, service_bits)
        q_after_service = q_before - outflow
        self._update_service_gap_state(q_before, outflow, q_after_service)
        self.last_gu_outflow = outflow.astype(np.float32)
        q_after_deadline, expire_amount = self._update_deadline_state_and_expire(q_before, outflow, q_after_service)
        overflow_drop = np.maximum(q_after_deadline - cfg.queue_max_gu, 0.0).astype(np.float32)
        self.gu_drop = (overflow_drop + expire_amount).astype(np.float32)
        q_after = np.minimum(q_after_deadline, cfg.queue_max_gu)
        q_after = _quantize_queue_contract_np(
            cfg,
            q_after,
            attr="structured_queue_state_quantum",
            default=128.0,
        )
        self.gu_queue = np.minimum(q_after, cfg.queue_max_gu).astype(np.float32)
        gu_outflow = outflow.astype(np.float32)
        self._update_bw_weighted_workload_service_ema(
            "bw_weighted_workload_acc_ema_vec",
            "bw_weighted_workload_acc_ema",
            service_bits,
            int(cfg.num_gu),
        )

        self.last_association = assoc.copy()
        if traffic_state_after_override is not None:
            self._apply_traffic_model_state_override(traffic_state_after_override)
        elif advance_traffic:
            self._advance_traffic_model_state()
        return gu_outflow

    def _select_satellites(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        actions: Dict[str, Dict],
        visible: List[List[int]],
    ) -> List[List[int]]:
        cfg = self.cfg
        selections: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        exec_sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        sat_k = max(int(getattr(cfg, "sat_action_select_k", getattr(cfg, "sat_num_select", cfg.N_RF)) or cfg.N_RF), 0)
        exec_sat_indices = np.full((cfg.num_uav, sat_k), -1, dtype=np.int64)
        for u in range(cfg.num_uav):
            vis = visible[u]
            if not vis:
                continue
            if cfg.fixed_satellite_strategy:
                # pick nearest visible
                vis_idx = np.asarray(vis, dtype=np.int32)
                dists = np.linalg.norm(sat_pos[vis_idx] - self._uav_ecef(u)[None, :], axis=1)
                sel = int(vis_idx[int(np.argmin(dists))])
                selections[u] = [sel]
                if sat_k > 0:
                    exec_sat_mask[u, 0] = 1.0
                    exec_sat_indices[u, 0] = 0
            else:
                cand = vis[: cfg.sats_obs_max]
                if not cand:
                    continue
                action_data = actions[self.agents[u]]
                sat_raw = action_data.get(
                    "sat_select_mask",
                    action_data.get("sat_logits", np.zeros((cfg.sats_obs_max,), dtype=np.float32)),
                )
                sat_raw = np.asarray(sat_raw, dtype=np.float32)[: len(cand)]
                valid_flags = np.ones((len(cand),), dtype=bool)
                if cfg.doppler_enabled:
                    cand_idx = np.asarray(cand, dtype=np.int32)
                    raw_nu = self._doppler_many(u, cand_idx, sat_pos, sat_vel)
                    nu_eff, _ = self._effective_doppler_array(u, cand_idx, raw_nu)
                    valid_flags = np.abs(nu_eff) <= cfg.nu_max

                valid_slots = np.flatnonzero(valid_flags)
                if valid_slots.size == 0:
                    continue
                chosen_slots = np.flatnonzero((sat_raw > 0.5) & valid_flags)
                if chosen_slots.size > cfg.N_RF:
                    order = np.argsort(-sat_raw[chosen_slots], kind="stable")
                    chosen_slots = chosen_slots[order[: cfg.N_RF]]
                if chosen_slots.size == 0:
                    best_slot = int(valid_slots[int(np.argmax(sat_raw[valid_slots]))])
                    chosen_slots = np.array([best_slot], dtype=np.int64)
                chosen_slots = chosen_slots[: cfg.N_RF]
                selections[u] = [cand[int(idx)] for idx in chosen_slots.tolist()]
                exec_sat_mask[u, chosen_slots] = 1.0
                if sat_k > 0:
                    fill = min(len(chosen_slots), sat_k)
                    exec_sat_indices[u, :fill] = chosen_slots[:fill]
        self.last_exec_sat_select_mask = exec_sat_mask
        self.last_exec_sat_indices = exec_sat_indices
        return selections

    def _compute_backhaul_rates(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        selections: List[List[int]] | np.ndarray | Sequence[Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        U = cfg.num_uav
        L = cfg.num_sat
        rate_matrix = np.zeros((U, L), dtype=np.float32)
        sat_score_sum = 0.0
        sat_score_count = 0
        loss_matrix = self._get_backhaul_loss_matrix(sat_pos)
        selection_matrix = self._sat_selection_matrix(selections)
        valid_selection = selection_matrix >= 0
        if np.any(selection_matrix[valid_selection] >= int(L)):
            raise ValueError(f"Satellite selection index out of range for num_sat={L}.")

        counts = np.zeros((L,), dtype=np.int32)
        if np.any(valid_selection):
            np.add.at(counts, selection_matrix[valid_selection].astype(np.int32, copy=False), 1)

        for u in range(U):
            sat_idx = selection_matrix[u]
            sat_idx = sat_idx[sat_idx >= 0].astype(np.int32, copy=False)
            if sat_idx.size == 0:
                continue
            count_u = counts[sat_idx]
            valid = count_u > 0
            if not np.any(valid):
                continue
            sat_idx = sat_idx[valid]
            count_u = count_u[valid]
            b_ul = (self._effective_b_backhaul_per_sat() / count_u).astype(np.float32, copy=False)
            rel = sat_pos[sat_idx] - self._uav_ecef(u)[None, :]
            d = geometry_denominator(np.linalg.norm(rel, axis=1))
            gain = self._backhaul_gain_const / geometry_denominator(d * d)
            if loss_matrix is not None:
                gain = gain * loss_matrix[u, sat_idx]

            snr = channel.snr_linear(
                cfg.uav_tx_power,
                gain,
                cfg.noise_density,
                b_ul,
                noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
            )
            if cfg.doppler_enabled or cfg.doppler_atten_enabled:
                raw_nu = self._doppler_many(u, sat_idx, sat_pos, sat_vel)
                nu_eff, _ = self._effective_doppler_array(u, sat_idx, raw_nu)
            else:
                nu_eff = np.zeros((sat_idx.size,), dtype=np.float32)
            if cfg.doppler_atten_enabled:
                snr = snr * channel.doppler_attenuation(nu_eff, cfg.subcarrier_spacing)

            se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
            rate = (b_ul * se).astype(np.float32, copy=False)

            if cfg.doppler_enabled:
                rate = np.where(np.abs(nu_eff) <= cfg.nu_max, rate, 0.0).astype(np.float32, copy=False)
            rate_matrix[u, sat_idx] = rate
            sat_score = se - 0.5 * (
                self.sat_queue[sat_idx] / normalize_scale(cfg.queue_max_sat)
            ).astype(np.float32, copy=False)
            sat_score_sum += float(np.sum(sat_score.astype(np.float32, copy=False), dtype=np.float32))
            sat_score_count += int(sat_score.size)

        self.last_sat_score = sat_score_sum / max(1, sat_score_count)
        rate_matrix = _quantize_numeric_contract_np(
            cfg,
            rate_matrix,
            quantum=_semantic_quantum(cfg, "structured_backhaul_rate_quantum", 32.0),
            dtype=np.float32,
        )
        return rate_matrix, counts

    def _update_uav_queues(self, gu_outflow: np.ndarray, rate_matrix: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        outflow_matrix = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        valid = self.last_association >= 0
        if np.any(valid):
            inflow = np.zeros((cfg.num_uav,), dtype=np.float32)
            np.add.at(
                inflow,
                self.last_association[valid].astype(np.int64, copy=False),
                np.asarray(gu_outflow[valid], dtype=np.float32),
            )
        else:
            inflow = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_gu_to_uav_inflow_by_uav = inflow.astype(np.float32, copy=True)
        q_before = self.uav_queue + inflow
        total_rate = np.sum(rate_matrix, axis=1).astype(np.float32)
        service_bits = _quantize_queue_contract_np(
            cfg,
            total_rate * float(cfg.tau0),
            attr="structured_flow_bits_quantum",
            default=32.0,
        )
        outflow = np.minimum(q_before, service_bits)
        q_after = q_before - outflow
        self.uav_drop = np.maximum(q_after - cfg.queue_max_uav, 0.0).astype(np.float32)
        q_after = np.minimum(q_after, cfg.queue_max_uav)
        q_after = _quantize_queue_contract_np(
            cfg,
            q_after,
            attr="structured_queue_state_quantum",
            default=128.0,
        )
        self.uav_queue = np.minimum(q_after, cfg.queue_max_uav).astype(np.float32)
        self.last_uav_outflow = outflow.astype(np.float32)
        self._update_bw_weighted_workload_service_ema(
            "bw_weighted_workload_rel_ema_vec",
            "bw_weighted_workload_rel_ema",
            service_bits,
            int(cfg.num_uav),
        )
        mask = total_rate > 0
        if np.any(mask):
            outflow_matrix[mask] = (rate_matrix[mask] / total_rate[mask, None]) * outflow[mask, None]
        self.last_uav_to_sat_outflow_matrix = outflow_matrix.astype(np.float32, copy=True)
        return outflow_matrix

    def _update_sat_queues(self, outflow_matrix: np.ndarray) -> None:
        cfg = self.cfg
        incoming = _quantize_queue_contract_np(
            cfg,
            np.sum(outflow_matrix, axis=0),
            attr="structured_flow_bits_quantum",
            default=32.0,
        )
        compute_rate = self._effective_sat_cpu_freq() / normalize_scale(float(cfg.task_cycles_per_bit))
        before = self.sat_queue.copy()
        q_before = before + incoming
        processed_cap = _quantize_queue_contract_np(
            cfg,
            float(compute_rate) * float(cfg.tau0),
            attr="structured_flow_bits_quantum",
            default=32.0,
        )
        processed = np.minimum(q_before, processed_cap)
        q_after = q_before - processed
        self.sat_drop = np.maximum(q_after - cfg.queue_max_sat, 0.0).astype(np.float32)
        q_after = np.minimum(q_after, cfg.queue_max_sat)
        q_after = _quantize_queue_contract_np(
            cfg,
            q_after,
            attr="structured_queue_state_quantum",
            default=128.0,
        )
        self.last_sat_processed = processed.astype(np.float32)
        self.last_sat_incoming = incoming.astype(np.float32)
        self.sat_queue = np.minimum(q_after, cfg.queue_max_sat).astype(np.float32)
        self._update_bw_weighted_workload_service_ema(
            "bw_weighted_workload_sat_ema_vec",
            "bw_weighted_workload_sat_ema",
            processed_cap,
            int(cfg.num_sat),
        )

    def _update_energy(self, selections: List[List[int]] | np.ndarray | Sequence[Sequence[int]]) -> None:
        cfg = self.cfg
        if not cfg.energy_enabled:
            self.last_energy_cost = np.zeros((cfg.num_uav,), dtype=np.float32)
            return
        speeds = np.linalg.norm(self.uav_vel, axis=1)
        p_fly = self._fly_power(speeds)
        selection_matrix = self._sat_selection_matrix(selections)
        link_counts = np.sum(selection_matrix >= 0, axis=1, dtype=np.int32).astype(np.float32, copy=False)
        p_comm = cfg.p_comm_link * link_counts
        self.last_energy_cost = p_fly + p_comm
        self.uav_energy = self.uav_energy - self.last_energy_cost * cfg.tau0
        self.uav_energy = np.maximum(self.uav_energy, 0.0)

    def _update_connected_sat_link_stats(
        self,
        sat_pos: np.ndarray,
        sat_selection: List[List[int]] | np.ndarray | Sequence[Sequence[int]],
    ) -> None:
        dist_values: List[float] = []
        elevation_deg_values: List[float] = []
        selection_matrix = self._sat_selection_matrix(sat_selection)
        if np.any(selection_matrix[selection_matrix >= 0] >= int(self.cfg.num_sat)):
            raise ValueError(f"Satellite selection index out of range for num_sat={self.cfg.num_sat}.")
        for u in range(int(selection_matrix.shape[0])):
            selected = selection_matrix[u]
            valid_selected = selected[selected >= 0]
            for l in valid_selected.tolist():
                sat_idx = int(l)
                rel_pos = sat_pos[sat_idx] - self._uav_ecef(u)
                dist_values.append(float(np.linalg.norm(rel_pos)))
                elevation_deg_values.append(math.degrees(self._elevation_angle(u, sat_idx, sat_pos)))
        if dist_values:
            dist_arr = np.asarray(dist_values, dtype=np.float32)
            elevation_arr = np.asarray(elevation_deg_values, dtype=np.float32)
            self.last_connected_sat_count = float(dist_arr.size)
            self.last_connected_sat_dist_mean = float(np.mean(dist_arr))
            self.last_connected_sat_dist_p95 = float(np.percentile(dist_arr, 95.0))
            self.last_connected_sat_elevation_deg_mean = float(np.mean(elevation_arr))
            self.last_connected_sat_elevation_deg_min = float(np.min(elevation_arr))
            return

        self.last_connected_sat_count = 0.0
        self.last_connected_sat_dist_mean = 0.0
        self.last_connected_sat_dist_p95 = 0.0
        self.last_connected_sat_elevation_deg_mean = 0.0
        self.last_connected_sat_elevation_deg_min = 0.0

    def _apply_bw_transition_core(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        bw_actions: Dict[str, Dict] | np.ndarray | Sequence[np.ndarray],
        sat_selection: List[List[int]] | np.ndarray | Sequence[Sequence[int]],
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        *,
        access_rates: np.ndarray | None = None,
        access_snapshot: AccessChannelSnapshot | np.ndarray | None = None,
        arrival_override: np.ndarray | None = None,
        traffic_state_after_override: dict | None = None,
        bw_link_transition_override: dict[str, Any] | None = None,
    ) -> BwTransitionCoreResult:
        assoc_arr = np.asarray(assoc, dtype=np.int32)
        sat_selection_matrix = self._sat_selection_matrix(sat_selection)
        gu_queue_before = np.asarray(self.gu_queue, dtype=np.float32).copy()
        uav_queue_before = np.asarray(self.uav_queue, dtype=np.float32).copy()
        sat_queue_before = np.asarray(self.sat_queue, dtype=np.float32).copy()
        if access_rates is None:
            access_rates_arr, _ = self._compute_access_rates(
                assoc_arr,
                candidates,
                bw_actions,
                record_exec=True,
                access_snapshot=access_snapshot,
            )
        else:
            access_rates_arr = np.asarray(access_rates, dtype=np.float32)

        realized_arrival = self._sample_gu_arrival(arrival_override=arrival_override)
        self._apply_gu_queue_transition_from_arrival(
            np.asarray(access_rates_arr, dtype=np.float32),
            assoc_arr,
            np.asarray(realized_arrival, dtype=np.float32),
            traffic_state_after_override=traffic_state_after_override,
        )

        self._refresh_uav_cache()
        uav_ecef = np.stack([self._uav_ecef(u) for u in range(self.cfg.num_uav)], axis=0).astype(np.float32)
        if bw_link_transition_override is None:
            self._update_energy(sat_selection_matrix)
            rate_matrix, sat_loads = self._compute_backhaul_rates(
                np.asarray(sat_pos, dtype=np.float32),
                np.asarray(sat_vel, dtype=np.float32),
                sat_selection_matrix,
            )
            self.last_sat_selection = self._sat_selection_lists(sat_selection_matrix)
            self.last_sat_connection_counts = np.asarray(sat_loads, dtype=np.float32)
            self._update_connected_sat_link_stats(np.asarray(sat_pos, dtype=np.float32), sat_selection_matrix)
            link_transition = {
                "uav_energy": np.asarray(self.uav_energy, dtype=np.float32)[None, ...],
                "last_energy_cost": np.asarray(self.last_energy_cost, dtype=np.float32)[None, ...],
                "rate_matrix": np.asarray(rate_matrix, dtype=np.float32)[None, ...],
                "sat_loads": np.asarray(sat_loads, dtype=np.float32)[None, ...],
                "last_sat_score": np.asarray([float(self.last_sat_score)], dtype=np.float32),
            }
        else:
            def _override_array(key: str, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
                if key not in bw_link_transition_override:
                    raise KeyError(f"BW link transition override is missing {key!r}.")
                arr = np.asarray(bw_link_transition_override[key], dtype=dtype)
                if arr.shape == (1,) + tuple(shape):
                    arr = arr[0]
                if arr.shape != tuple(shape):
                    raise ValueError(
                        f"BW link transition override {key!r} shape must be {shape}, got {arr.shape}."
                    )
                return arr.astype(dtype, copy=False)

            link_transition = {
                "uav_energy": _override_array("uav_energy", (int(self.cfg.num_uav),))[None, ...],
                "last_energy_cost": _override_array("last_energy_cost", (int(self.cfg.num_uav),))[None, ...],
                "rate_matrix": _override_array("rate_matrix", (int(self.cfg.num_uav), int(self.cfg.num_sat)))[None, ...],
                "sat_loads": _override_array("sat_loads", (int(self.cfg.num_sat),))[None, ...],
                "last_sat_score": np.asarray(
                    [_override_array("last_sat_score", (), dtype=np.float32).reshape(())[()]],
                    dtype=np.float32,
                ),
            }
        self.uav_energy = np.asarray(link_transition["uav_energy"][0], dtype=np.float32)
        self.last_energy_cost = np.asarray(link_transition["last_energy_cost"][0], dtype=np.float32)
        rate_matrix = np.asarray(link_transition["rate_matrix"][0], dtype=np.float32)
        sat_loads = np.asarray(link_transition["sat_loads"][0], dtype=np.float32)
        self.last_sat_score = float(np.asarray(link_transition["last_sat_score"], dtype=np.float32).reshape(-1)[0])
        self.last_sat_selection = self._sat_selection_lists(sat_selection_matrix)
        self.last_sat_connection_counts = sat_loads.astype(np.float32, copy=False)
        selected_mask = np.zeros((self.cfg.num_uav, self.cfg.num_sat), dtype=np.float32)
        for u in range(int(self.cfg.num_uav)):
            selected = sat_selection_matrix[u]
            selected = selected[(selected >= 0) & (selected < int(self.cfg.num_sat))]
            if selected.size > 0:
                selected_mask[u, selected.astype(np.int64, copy=False)] = 1.0
        self.last_selected_mask_by_uav_sat = selected_mask
        self._update_connected_sat_link_stats(np.asarray(sat_pos, dtype=np.float32), sat_selection_matrix)
        outflow_matrix = self._update_uav_queues(self.last_gu_outflow, rate_matrix)
        self._update_sat_queues(outflow_matrix)
        self._refresh_bw_proxy_features()
        return BwTransitionCoreResult(
            gu_queue_before=gu_queue_before,
            uav_queue_before=uav_queue_before,
            sat_queue_before=sat_queue_before,
            realized_arrival=realized_arrival,
            rate_matrix=np.asarray(rate_matrix, dtype=np.float32),
            sat_loads=np.asarray(sat_loads, dtype=np.float32),
        )

    def _prepare_next_step_stage_context(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        *,
        advance_doppler: bool,
        refresh_sat_obs: bool,
        doppler_residual_after_override: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        assoc_next = self._associate_users()
        candidate_lists_next = self._build_candidate_users(assoc_next)
        self._cached_bw_valid_mask = self._build_bw_valid_mask(assoc_next, candidate_lists_next)
        access_snapshot = self._sample_access_channel_snapshot()
        _, eta_next = self._compute_access_rates(
            assoc_next,
            candidate_lists_next,
            self._zero_bw_action_matrix(),
            record_exec=False,
            access_snapshot=access_snapshot,
        )
        self._store_cached_access_stage_context(
            assoc_next,
            candidate_lists_next,
            eta=eta_next,
            bw_valid_mask=self._cached_bw_valid_mask,
            access_snapshot=access_snapshot,
            snapshot_step_t=int(self.t) + 1,
        )
        if doppler_residual_after_override is not None:
            residual = np.asarray(doppler_residual_after_override, dtype=np.float32)
            expected_shape = (self.cfg.num_uav, self.cfg.num_sat)
            if residual.shape != expected_shape:
                raise ValueError(
                    f"doppler_residual_after_override shape must be {expected_shape}, got {tuple(residual.shape)}."
                )
            self._doppler_residual_state_hz = residual.copy()
        elif advance_doppler:
            self._advance_doppler_residual_state()
        sat_pos_arr = np.asarray(sat_pos, dtype=np.float32)
        sat_vel_arr = np.asarray(sat_vel, dtype=np.float32)
        visible_next: List[List[int]] | None = None
        if refresh_sat_obs:
            visible_next = self._visible_sats_sorted(sat_pos_arr)
            self._cache_sat_obs(sat_pos_arr, sat_vel_arr, visible_next)
        self._cached_obs_runtime_context = None
        return {
            "assoc": np.asarray(assoc_next, dtype=np.int32).copy(),
            "candidates": [list(c) for c in candidate_lists_next],
            "bw_valid_mask": np.asarray(self._cached_bw_valid_mask, dtype=np.float32).copy(),
            "eta_slots": np.asarray(eta_next, dtype=np.float32).copy(),
            "access_gain_matrix": self._coerce_access_gain_matrix(access_snapshot).copy(),
            "sat_pos": sat_pos_arr,
            "sat_vel": sat_vel_arr,
            "visible": None if visible_next is None else [list(v) for v in visible_next],
        }

    def _prepare_next_step_observation_cache(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        *,
        advance_doppler: bool,
        doppler_residual_after_override: np.ndarray | None = None,
    ) -> None:
        self._prepare_next_step_stage_context(
            sat_pos,
            sat_vel,
            advance_doppler=advance_doppler,
            refresh_sat_obs=True,
            doppler_residual_after_override=doppler_residual_after_override,
        )

    def _finalize_post_bw_step(self) -> StepStatusCoreResult:
        cfg = self.cfg
        reward = float(self._compute_reward())
        reward_parts = dict(getattr(self, "last_reward_parts", {}) or {})
        collision = bool(reward_parts.get("collision_event", 0.0) > 0.5)
        if not collision:
            collision = self._check_collision()
        self._episode_step_count = int(getattr(self, "_episode_step_count", 0)) + 1
        if collision:
            self._episode_collision_count = int(getattr(self, "_episode_collision_count", 0)) + 1
        energy_depleted = bool(cfg.energy_enabled and np.any(self.uav_energy <= 0.0))
        terminated = bool(collision or energy_depleted)
        truncated = bool(self.t >= (cfg.T_steps - 1))
        self.t += 1
        return StepStatusCoreResult(
            reward=reward,
            reward_parts=reward_parts,
            collision=collision,
            terminated=terminated,
            truncated=truncated,
        )

    def _materialize_post_step_outputs(
        self,
        step_status: StepStatusCoreResult,
        *,
        materialize_step_outputs: bool,
        materialize_agent_dicts: bool,
        refresh_global_state_cache: bool = True,
    ) -> StepMaterializationResult:
        obs_context = None
        gu_proxy_features = None
        if materialize_step_outputs:
            obs_context = self._build_obs_runtime_context()
            gu_proxy_features = [np.asarray(feature, dtype=np.float32) for feature in obs_context["gu_proxy_features"]]
        elif refresh_global_state_cache:
            gu_proxy_features = self._gu_proxy_feature_arrays()
        if materialize_step_outputs:
            obs = self._build_all_obs_from_context(obs_context if obs_context is not None else self._build_obs_runtime_context())
            infos = {agent: self._agent_visible_info(idx) for idx, agent in enumerate(self.agents)}
            if refresh_global_state_cache:
                self._refresh_global_state_cache(gu_proxy_features=gu_proxy_features)
        else:
            obs = {}
            infos = {}
            if refresh_global_state_cache:
                self._refresh_global_state_cache(gu_proxy_features=gu_proxy_features)
        if materialize_agent_dicts:
            rewards = {agent: float(step_status.reward) for agent in self.agents}
            terminations = {agent: bool(step_status.terminated) for agent in self.agents}
            truncations = {agent: bool(step_status.truncated) for agent in self.agents}
        else:
            rewards = {}
            terminations = {}
            truncations = {}
        return StepMaterializationResult(
            obs=obs,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            gu_proxy_features=gu_proxy_features,
        )

    def _compute_reward(self) -> float:
        cfg = self.cfg
        # Reward can use the legacy dense queue-aware shaping or a throughput-only objective.
        use_active_queue_delta = ablation_flag(
            cfg,
            "use_active_queue_delta",
            fallback_attr="queue_delta_use_active",
            default=False,
        )
        use_energy_reward = ablation_flag(cfg, "use_energy_reward", default=cfg.energy_enabled)
        use_reward_tanh = ablation_flag(
            cfg,
            "use_reward_tanh",
            fallback_attr="reward_tanh_enabled",
            default=False,
        )
        use_queue_log_smoothing = ablation_flag(
            cfg,
            "use_queue_log_smoothing",
            default=False,
        )
        queue_penalty_mode = str(getattr(cfg, "queue_penalty_mode", "quadratic") or "quadratic").lower()
        use_arrival_norm_queue = bool(getattr(cfg, "queue_reward_use_arrival_norm", False))
        reward_mode = str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower()
        if reward_mode not in {
            "dense",
            "throughput_only",
            "controllable_flow",
            "weighted_workload_delta",
            "relative_weighted_workload_delta",
            "weighted_workload_level",
            "positive_weighted_workload_level",
            "sat_relay_processed",
            "sat_backhaul_drop",
            "gu_queue_level",
            "system_queue_level",
            "gu_service_queue",
        }:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        if cfg.energy_enabled:
            p_max = self._energy_scale()
            r_energy = -float(np.mean(ratio_or_zero(self.last_energy_cost, p_max)))
        else:
            r_energy = 0.0

        q_gu = _contract_sum_scalar_np(cfg, self.gu_queue)
        q_uav = _contract_sum_scalar_np(cfg, self.uav_queue)
        q_sat = _contract_sum_scalar_np(cfg, self.sat_queue)
        q_total = _contract_add_scalar_np(q_gu, q_uav, q_sat)
        q_max_total = float(
            cfg.num_gu * cfg.queue_max_gu
            + cfg.num_uav * cfg.queue_max_uav
            + cfg.num_sat * cfg.queue_max_sat
        )
        q_max_total = normalize_scale(q_max_total)
        q_gu_max = normalize_scale(float(cfg.num_gu * cfg.queue_max_gu))
        q_uav_max = normalize_scale(float(cfg.num_uav * cfg.queue_max_uav))
        q_sat_max = normalize_scale(float(cfg.num_sat * cfg.queue_max_sat))
        q_gu_norm = _contract_ratio_scalar_np(q_gu, q_gu_max)
        q_uav_norm = _contract_ratio_scalar_np(q_uav, q_uav_max)
        q_sat_norm = _contract_ratio_scalar_np(q_sat, q_sat_max)
        q_total_active = _contract_add_scalar_np(q_gu, q_uav)

        arrival_sum = _contract_sum_scalar_np(cfg, self.last_gu_arrival)
        outflow_sum = _contract_sum_scalar_np(cfg, self.last_gu_outflow)
        backhaul_sum = _contract_sum_scalar_np(cfg, getattr(self, "last_sat_incoming", 0.0))
        sat_processed_sum = _contract_sum_scalar_np(cfg, getattr(self, "last_sat_processed", 0.0))
        expire_sum = _contract_sum_scalar_np(cfg, getattr(self, "gu_expired", 0.0))
        gu_drop_sum = _contract_sum_scalar_np(cfg, self.gu_drop)
        uav_drop_sum = _contract_sum_scalar_np(cfg, self.uav_drop)
        sat_drop_sum = _contract_sum_scalar_np(cfg, getattr(self, "sat_drop", 0.0))
        drop_sum_active = _contract_add_scalar_np(gu_drop_sum, uav_drop_sum)
        drop_sum = _contract_add_scalar_np(drop_sum_active, sat_drop_sum)
        service_ratio = _contract_ratio_scalar_np(outflow_sum, arrival_sum)
        drop_ratio = _contract_ratio_scalar_np(drop_sum, arrival_sum)
        service_ratio = float(np.clip(service_ratio, 0.0, 1.0))
        drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))
        if cfg.num_gu > 0:
            assoc_ratio = float(np.mean(self.last_association >= 0))
        else:
            assoc_ratio = 0.0
        assoc_unfair_max_gu_count = 0.0
        assoc_unfair_step = 0.0
        assoc_unfair_gu_threshold = max(int(getattr(cfg, "assoc_unfair_gu_threshold", 15) or 0), 0)
        if cfg.num_uav > 0 and cfg.num_gu > 0:
            assoc_valid = self.last_association[self.last_association >= 0]
            if assoc_valid.size > 0:
                assoc_counts = np.bincount(assoc_valid, minlength=cfg.num_uav).astype(np.float32, copy=False)
                assoc_unfair_max_gu_count = float(np.max(assoc_counts))
                if assoc_unfair_gu_threshold > 0 and assoc_unfair_max_gu_count >= float(assoc_unfair_gu_threshold):
                    assoc_unfair_step = 1.0

        arrival_ref = self._arrival_ref()
        arrival_scale = arrival_ref
        service_norm = _contract_ratio_scalar_np(outflow_sum, arrival_scale)
        drop_norm = _contract_ratio_scalar_np(drop_sum, arrival_scale)
        gu_drop_norm = _contract_ratio_scalar_np(gu_drop_sum, arrival_scale)
        uav_drop_norm = _contract_ratio_scalar_np(uav_drop_sum, arrival_scale)
        sat_drop_norm = _contract_ratio_scalar_np(sat_drop_sum, arrival_scale)
        drop_event = 1.0 if drop_sum > RUNTIME_RATIO_ZERO_TOL else 0.0
        throughput_access_norm = _contract_ratio_scalar_np(outflow_sum, arrival_scale)
        throughput_backhaul_norm = _contract_ratio_scalar_np(backhaul_sum, arrival_scale)
        sat_processed_norm = _contract_ratio_scalar_np(sat_processed_sum, arrival_scale)
        outflow_arrival_ratio_step = _contract_ratio_scalar_np(outflow_sum, arrival_sum)
        sat_incoming_arrival_ratio_step = _contract_ratio_scalar_np(backhaul_sum, arrival_sum)
        sat_processed_arrival_ratio_step = _contract_ratio_scalar_np(sat_processed_sum, arrival_sum)
        sat_processed_incoming_ratio_step = _contract_ratio_scalar_np(sat_processed_sum, backhaul_sum)
        gu_drop_ratio_step = _contract_ratio_scalar_np(gu_drop_sum, arrival_sum)
        uav_drop_ratio_step = _contract_ratio_scalar_np(uav_drop_sum, arrival_sum)
        sat_drop_ratio_step = _contract_ratio_scalar_np(sat_drop_sum, arrival_sum)
        expire_ratio_step = _contract_ratio_scalar_np(expire_sum, arrival_sum)
        b_pre_t = _contract_add_scalar_np(
            float(getattr(self, "prev_queue_sum_gu", q_gu)),
            float(getattr(self, "prev_queue_sum_uav", q_uav)),
        )
        b_pre_tp1 = q_total_active
        x_acc = _contract_ratio_scalar_np(outflow_sum, arrival_ref)
        x_rel = _contract_ratio_scalar_np(backhaul_sum, arrival_ref)
        g_pre = _contract_ratio_scalar_np(_contract_add_scalar_np(b_pre_tp1, -b_pre_t), arrival_ref)
        d_pre = _contract_ratio_scalar_np(_contract_add_scalar_np(gu_drop_sum, uav_drop_sum), arrival_ref)
        processed_ratio_eval = _contract_ratio_scalar_np(sat_processed_sum, arrival_ref)
        drop_ratio_eval = _contract_ratio_scalar_np(drop_sum, arrival_ref)
        pre_backlog_steps_eval = _contract_ratio_scalar_np(q_total_active, arrival_ref)
        b_pre_steps = pre_backlog_steps_eval
        D_sys_report = _contract_ratio_scalar_np(q_total, sat_processed_sum)
        (
            _assoc_centroid_counts,
            _assoc_centroid_rel,
            assoc_centroid_dist_norms,
            assoc_centroid_valid_uav_count,
            assoc_centroid_valid_uav_frac,
            assoc_centroid_dist_norm_mean,
        ) = self._assoc_centroid_summary()
        sat_overlap_uav, sat_overlap_eval = self._sat_overlap_summary()
        self.last_assoc_centroid_dist_norms = assoc_centroid_dist_norms.astype(np.float32, copy=False)
        self.last_sat_overlap_uav = sat_overlap_uav.astype(np.float32, copy=False)

        queue_norm_scale = self._queue_arrival_scale(arrival_sum)
        q_norm_active = float(np.clip(q_total_active / queue_norm_scale, 0.0, 1.0))
        prev_q_norm_active = float(getattr(self, "prev_q_norm_active", q_norm_active))
        q_norm_delta = float(prev_q_norm_active - q_norm_active)
        q_norm_tail_q0 = max(float(getattr(cfg, "q_norm_tail_q0", 0.0) or 0.0), 0.0)
        q_norm_tail_excess = 0.0
        prev_queue_sum_gu = float(getattr(self, "prev_queue_sum_gu", q_gu))
        prev_queue_sum_uav = float(getattr(self, "prev_queue_sum_uav", q_uav))
        prev_queue_sum_sat = float(getattr(self, "prev_queue_sum_sat", q_sat))
        queue_delta_mode = str(getattr(cfg, "queue_delta_mode", "total") or "total").strip().lower()
        if queue_delta_mode not in {"total", "weighted"}:
            queue_delta_mode = "total"
        queue_weight = float(cfg.omega_q)
        q_delta_weight = float(cfg.eta_q_delta)
        crash_weight = float(cfg.eta_crash)
        gu_queue_fill_fraction = q_gu_norm
        uav_queue_fill_fraction = q_uav_norm
        sat_queue_fill_fraction = q_sat_norm
        gu_queue_arrival_steps = q_gu / arrival_scale
        uav_queue_arrival_steps = q_uav / arrival_scale
        sat_queue_arrival_steps = q_sat / arrival_scale
        prev_gu_queue_arrival_steps = prev_queue_sum_gu / arrival_scale
        prev_uav_queue_arrival_steps = prev_queue_sum_uav / arrival_scale
        prev_sat_queue_arrival_steps = prev_queue_sum_sat / arrival_scale
        if use_arrival_norm_queue:
            queue_delta_gu = float(prev_gu_queue_arrival_steps - gu_queue_arrival_steps)
            queue_delta_uav = float(prev_uav_queue_arrival_steps - uav_queue_arrival_steps)
            queue_delta_sat = float(prev_sat_queue_arrival_steps - sat_queue_arrival_steps)
        else:
            prev_q_gu_norm = float(np.clip(prev_queue_sum_gu / q_gu_max, 0.0, 1.0))
            prev_q_uav_norm = float(np.clip(prev_queue_sum_uav / q_uav_max, 0.0, 1.0))
            prev_q_sat_norm = float(np.clip(prev_queue_sum_sat / q_sat_max, 0.0, 1.0))
            queue_delta_gu = float(np.clip(prev_q_gu_norm - q_gu_norm, -1.0, 1.0))
            queue_delta_uav = float(np.clip(prev_q_uav_norm - q_uav_norm, -1.0, 1.0))
            queue_delta_sat = float(np.clip(prev_q_sat_norm - q_sat_norm, -1.0, 1.0))

        def _queue_smooth(q_value: float) -> float:
            if use_arrival_norm_queue:
                q_value = max(float(q_value), 0.0)
                if use_queue_log_smoothing or queue_penalty_mode == "log":
                    k = float(getattr(cfg, "queue_log_k", 0.0) or 0.0)
                    if k > 0:
                        return math.log1p(k * q_value) / math.log1p(k)
                    return q_value
                if queue_penalty_mode == "linear":
                    return q_value
                return q_value * q_value
            q_norm = float(np.clip(q_value, 0.0, 1.0))
            if use_queue_log_smoothing or queue_penalty_mode == "log":
                k = float(getattr(cfg, "queue_log_k", 0.0) or 0.0)
                if k > 0:
                    return math.log1p(k * q_norm) / math.log1p(k)
                return q_norm
            if queue_penalty_mode == "linear":
                return q_norm
            # Default to quadratic queue penalty to amplify congestion gradients near full queues.
            return q_norm * q_norm

        if use_active_queue_delta:
            # Active queue (GU+UAV) is normalized by arrival scale for both
            # absolute penalty and delta reward to keep one consistent gradient scale.
            queue_delta_mode = "active"
            queue_gu = q_gu_norm
            queue_uav = q_uav_norm
            queue_sat = q_sat_norm
            if q_norm_tail_q0 > 0.0:
                q_norm_tail_excess = max(q_norm_active - q_norm_tail_q0, 0.0)
                queue_term = q_norm_tail_excess * q_norm_tail_excess
            else:
                queue_term = q_norm_active
            omega_q_tail = getattr(cfg, "omega_q_tail", None)
            queue_weight = float(cfg.omega_q if omega_q_tail is None else omega_q_tail)
            queue_delta = float(np.clip(q_norm_delta, -1.0, 1.0))
        else:
            queue_gu_raw = gu_queue_arrival_steps if use_arrival_norm_queue else q_gu_norm
            queue_uav_raw = uav_queue_arrival_steps if use_arrival_norm_queue else q_uav_norm
            queue_sat_raw = sat_queue_arrival_steps if use_arrival_norm_queue else q_sat_norm
            queue_gu = _queue_smooth(queue_gu_raw)
            queue_uav = _queue_smooth(queue_uav_raw)
            queue_sat = _queue_smooth(queue_sat_raw)
            w_gu = float(getattr(cfg, "omega_q_gu", 0.0) or 0.0)
            w_uav = float(getattr(cfg, "omega_q_uav", 0.0) or 0.0)
            w_sat = float(getattr(cfg, "omega_q_sat", 0.0) or 0.0)
            w_sum = abs(w_gu) + abs(w_uav) + abs(w_sat)
            if w_sum < NORMALIZATION_DENOM_EPS:
                queue_total_norm = q_total / arrival_scale if use_arrival_norm_queue else q_total / q_max_total
                queue_term = _queue_smooth(queue_total_norm)
            else:
                queue_term = (w_gu * queue_gu + w_uav * queue_uav + w_sat * queue_sat) / w_sum
            if queue_delta_mode == "weighted" and w_sum >= NORMALIZATION_DENOM_EPS:
                queue_delta = (w_gu * queue_delta_gu + w_uav * queue_delta_uav + w_sat * queue_delta_sat) / w_sum
                if not use_arrival_norm_queue:
                    queue_delta = float(np.clip(queue_delta, -1.0, 1.0))
                else:
                    queue_delta = float(queue_delta)
            else:
                queue_delta_mode = "total"
                prev_sum = self.prev_queue_sum
                cur_sum = q_total
                q_delta_den = arrival_scale if use_arrival_norm_queue else q_max_total
                queue_delta = scalar_ratio_or_zero(prev_sum - cur_sum, q_delta_den)
                if not use_arrival_norm_queue:
                    queue_delta = float(np.clip(queue_delta, -1.0, 1.0))
                else:
                    queue_delta = float(queue_delta)
            queue_weight = float(cfg.omega_q)

        if cfg.a_max > 0:
            accel_norm2 = scalar_ratio_or_zero(float(np.mean(np.sum(self.last_exec_accel**2, axis=1))), cfg.a_max**2)
        else:
            accel_norm2 = 0.0
        intervention_delta = np.asarray(self.last_exec_accel - self.last_policy_accel, dtype=np.float32)
        intervention_norms = np.linalg.norm(intervention_delta, axis=1) if intervention_delta.size else np.zeros((0,), dtype=np.float32)
        intervention_norms_uav = (
            ratio_or_zero(intervention_norms, cfg.a_max) if intervention_norms.size and cfg.a_max > 0 else np.zeros((0,), dtype=np.float32)
        )
        intervention_norm = (
            float(np.mean(intervention_norms_uav)) if intervention_norms_uav.size else 0.0
        )
        intervention_rate = float(np.mean(intervention_norms > 1e-6)) if intervention_norms.size else 0.0
        intervention_norm_top1 = (
            float(np.max(intervention_norms_uav)) if intervention_norms_uav.size else 0.0
        )
        danger_imitation_enabled = bool(getattr(cfg, "danger_imitation_enabled", False))
        danger_trigger_mode = str(
            getattr(cfg, "danger_imitation_trigger_mode", "intervention_any") or "intervention_any"
        ).strip().lower()
        if danger_trigger_mode not in {"risk_or_intervention", "intervention_any", "intervention_threshold"}:
            danger_trigger_mode = "intervention_any"
        need_close_risk_stats = bool(getattr(cfg, "close_risk_enabled", False)) or (
            danger_imitation_enabled and danger_trigger_mode == "risk_or_intervention"
        )
        if need_close_risk_stats:
            close_risk_value, close_risk_uav = self._compute_close_risk_stats(require_enabled=False)
        else:
            close_risk_value = 0.0
            close_risk_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        close_risk = close_risk_value if bool(getattr(cfg, "close_risk_enabled", False)) else 0.0
        danger_imitation_mask = np.zeros((cfg.num_uav,), dtype=np.float32)
        if danger_imitation_enabled:
            close_risk_thresh = max(float(getattr(cfg, "danger_imitation_close_risk_thresh", 0.05) or 0.0), 0.0)
            intervention_thresh = max(
                float(getattr(cfg, "danger_imitation_intervention_thresh", 0.05) or 0.0),
                0.0,
            )
            if danger_trigger_mode == "intervention_any":
                danger_active = intervention_norms > 1e-6
            elif danger_trigger_mode == "intervention_threshold":
                danger_active = intervention_norms_uav > intervention_thresh
            else:
                danger_active = (
                    (close_risk_uav > close_risk_thresh)
                    | (intervention_norms > 1e-6)
                )
            danger_imitation_mask = danger_active.astype(np.float32, copy=False)
        danger_imitation_active_rate = float(np.mean(danger_imitation_mask)) if danger_imitation_mask.size else 0.0
        self.last_intervention_norm_uav = intervention_norms_uav.astype(np.float32, copy=False)
        self.last_close_risk_uav = close_risk_uav.astype(np.float32, copy=False)
        self.last_danger_imitation_mask = danger_imitation_mask.astype(np.float32, copy=False)

        centroid_reward, centroid_dist_mean = self._compute_centroid_stats()
        centroid_eta_start, centroid_eta, centroid_transfer_ratio = self._centroid_anneal_state()
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        if cross_enabled:
            queue_gain = float(getattr(cfg, "centroid_cross_queue_gain", 0.0) or 0.0)
            q_delta_gain = float(getattr(cfg, "centroid_cross_q_delta_gain", 0.0) or 0.0)
            crash_gain = float(getattr(cfg, "centroid_cross_crash_gain", 0.0) or 0.0)
            queue_weight = max(0.0, queue_weight * (1.0 + queue_gain * centroid_transfer_ratio))
            q_delta_weight = max(0.0, q_delta_weight * (1.0 + q_delta_gain * centroid_transfer_ratio))
            crash_weight = max(0.0, crash_weight * (1.0 + crash_gain * centroid_transfer_ratio))
        dist_delta = 0.0
        queue_topk = 0.0
        bw_align = float(getattr(self, "last_bw_align", 0.0))
        sat_score = float(getattr(self, "last_sat_score", 0.0))
        tail_eta_accel = float(cfg.eta_accel)
        q_small = max(float(getattr(cfg, "tail_q_small", 0.0) or 0.0), 0.0)
        tail_eta_accel_gain = max(float(getattr(cfg, "tail_eta_accel_gain", 1.0) or 0.0), 0.0)
        if q_total_active <= q_small:
            tail_eta_accel = tail_eta_accel * tail_eta_accel_gain

        service_gap_risk_mean = (
            float(
                np.mean(
                    np.asarray(
                        getattr(self, "last_gu_service_gap_risk", np.zeros((cfg.num_gu,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    * (
                        np.asarray(self.gu_queue, dtype=np.float32)
                        / normalize_scale(float(cfg.queue_max_gu))
                    )
                )
            )
            if cfg.num_gu > 0
            else 0.0
        )
        gu_queue_before_vec = np.asarray(
            getattr(self, "prev_gu_queue_vec", np.asarray(self.gu_queue, dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        uav_queue_before_vec = np.asarray(
            getattr(self, "prev_uav_queue_vec", np.asarray(self.uav_queue, dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        sat_queue_before_vec = np.asarray(
            getattr(self, "prev_sat_queue_vec", np.asarray(self.sat_queue, dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        realized_arrival_vec = np.asarray(getattr(self, "last_gu_arrival", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32).copy()
        reward_weighted_workload_delta = self._reward_weighted_workload_delta(
            gu_queue_before=gu_queue_before_vec,
            uav_queue_before=uav_queue_before_vec,
            sat_queue_before=sat_queue_before_vec,
            realized_arrival=realized_arrival_vec,
        )
        reward_weighted_workload_level = self._reward_weighted_workload_level()
        workload_before_for_relative = max(
            float(reward_weighted_workload_delta - reward_weighted_workload_level),
            0.0,
        )
        reward_relative_weighted_workload_delta = float(
            reward_weighted_workload_delta / max(workload_before_for_relative, 1.0)
        )
        reward_positive_weighted_workload_level = self._reward_positive_weighted_workload_level()
        reward_gu_queue_level = self._reward_gu_queue_level()
        reward_system_queue_level = self._reward_system_queue_level()
        reward_gu_service_queue = self._reward_gu_service_queue()

        if reward_mode == "controllable_flow":
            term_service = 0.0
            term_throughput_access = float(getattr(cfg, "reward_w_access", 0.5) or 0.0) * x_acc
            term_throughput_backhaul = float(getattr(cfg, "reward_w_relay", 0.5) or 0.0) * x_rel
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = -float(getattr(cfg, "reward_w_pre_drop", 1.0) or 0.0) * d_pre
            term_queue = -float(getattr(cfg, "reward_w_pre_backlog", 0.08) or 0.0) * math.log1p(b_pre_steps)
            overflow_risk_mean = (
                float(np.mean(np.asarray(getattr(self, "last_gu_urgency_risk", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            )
            term_pre_overflow_risk = -float(getattr(cfg, "reward_w_pre_overflow_risk", 0.0) or 0.0) * overflow_risk_mean
            term_pre_service_gap = -float(getattr(cfg, "reward_w_pre_service_gap", 0.0) or 0.0) * service_gap_risk_mean
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            raw_reward = (
                term_throughput_access
                + term_throughput_backhaul
                + term_drop
                + term_queue
                + term_pre_overflow_risk
                + term_pre_service_gap
            )
        elif reward_mode == "sat_relay_processed":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.5 * x_rel
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = -drop_ratio_eval
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = 0.5 * x_rel + 0.5 * processed_ratio_eval - drop_ratio_eval - 0.05 * sat_overlap_eval
        elif reward_mode == "sat_backhaul_drop":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = x_rel
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = -d_pre
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = x_rel - d_pre - 0.05 * sat_overlap_eval
        elif reward_mode == "throughput_only":
            throughput_only_access_coef = float(getattr(cfg, "throughput_only_access_coef", 1.0) or 0.0)
            throughput_only_backhaul_coef = float(getattr(cfg, "throughput_only_backhaul_coef", 1.0) or 0.0)
            term_service = 0.0
            term_throughput_access = throughput_only_access_coef * throughput_access_norm
            term_throughput_backhaul = throughput_only_backhaul_coef * throughput_backhaul_norm
            throughput_only_gu_queue_coef = max(
                float(getattr(cfg, "throughput_only_gu_queue_coef", 0.0) or 0.0),
                0.0,
            )
            term_queue_gu_arrival = -throughput_only_gu_queue_coef * gu_queue_arrival_steps
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = term_throughput_access + term_throughput_backhaul + term_queue_gu_arrival
        elif reward_mode == "weighted_workload_delta":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_weighted_workload_delta
        elif reward_mode == "relative_weighted_workload_delta":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_relative_weighted_workload_delta
        elif reward_mode == "weighted_workload_level":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_weighted_workload_level
        elif reward_mode == "positive_weighted_workload_level":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_positive_weighted_workload_level
        elif reward_mode == "gu_queue_level":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_gu_queue_level
        elif reward_mode == "system_queue_level":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_system_queue_level
        elif reward_mode == "gu_service_queue":
            term_service = 0.0
            term_throughput_access = 0.0
            term_throughput_backhaul = 0.0
            term_queue_gu_arrival = 0.0
            eta_drop_default = 0.0
            eta_drop_gu = 0.0
            eta_drop_uav = 0.0
            eta_drop_sat = 0.0
            term_drop_gu = 0.0
            term_drop_uav = 0.0
            term_drop_sat = 0.0
            term_drop_step = 0.0
            term_drop = 0.0
            queue_weight = 0.0
            q_delta_weight = 0.0
            crash_weight = 0.0
            term_queue = 0.0
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = reward_gu_service_queue
        else:
            term_service = cfg.eta_service * service_norm
            term_throughput_access = float(getattr(cfg, "eta_throughput_access", 0.0) or 0.0) * throughput_access_norm
            term_throughput_backhaul = (
                float(getattr(cfg, "eta_throughput_backhaul", 0.0) or 0.0) * throughput_backhaul_norm
            )
            term_queue_gu_arrival = 0.0
            eta_drop_default = float(getattr(cfg, "eta_drop", 0.0) or 0.0)
            eta_drop_gu = float(
                eta_drop_default
                if getattr(cfg, "eta_drop_gu", None) is None
                else (getattr(cfg, "eta_drop_gu", 0.0) or 0.0)
            )
            eta_drop_uav = float(
                eta_drop_default
                if getattr(cfg, "eta_drop_uav", None) is None
                else (getattr(cfg, "eta_drop_uav", 0.0) or 0.0)
            )
            eta_drop_sat = float(
                eta_drop_default
                if getattr(cfg, "eta_drop_sat", None) is None
                else (getattr(cfg, "eta_drop_sat", 0.0) or 0.0)
            )
            term_drop_gu = -eta_drop_gu * gu_drop_norm
            term_drop_uav = -eta_drop_uav * uav_drop_norm
            term_drop_sat = -eta_drop_sat * sat_drop_norm
            term_drop_step = -float(getattr(cfg, "eta_drop_step", 0.0) or 0.0) * drop_event
            term_drop = term_drop_gu + term_drop_uav + term_drop_sat + term_drop_step
            term_queue = -queue_weight * queue_term
            term_pre_overflow_risk = 0.0
            term_pre_service_gap = 0.0
            term_q_delta = q_delta_weight * queue_delta
            term_centroid = centroid_eta * centroid_reward
            term_accel = -tail_eta_accel * accel_norm2
            term_close_risk = -max(float(getattr(cfg, "eta_close_risk", 0.0) or 0.0), 0.0) * close_risk
            term_energy = cfg.omega_e * r_energy if use_energy_reward else 0.0
            raw_reward = (
                term_service
                + term_throughput_access
                + term_throughput_backhaul
                + term_drop
                + term_queue
                + term_q_delta
                + term_centroid
                + term_accel
                + term_close_risk
                + term_energy
            )

        term_access = term_throughput_access
        term_relay = term_throughput_backhaul
        if reward_mode == "controllable_flow":
            term_pre_backlog = term_queue
            term_pre_drop = term_drop
        else:
            term_pre_backlog = 0.0
            term_pre_drop = 0.0

        collision_now = self._check_collision()
        if reward_mode in {
            "throughput_only",
            "controllable_flow",
            "weighted_workload_delta",
            "relative_weighted_workload_delta",
            "weighted_workload_level",
            "gu_queue_level",
            "system_queue_level",
            "gu_service_queue",
        }:
            collision_penalty = 0.0
            battery_penalty = 0.0
        else:
            collision_penalty = -crash_weight if collision_now else 0.0
            battery_penalty = -cfg.eta_batt if (cfg.energy_enabled and np.any(self.uav_energy <= 0.0)) else 0.0
        fail_penalty = collision_penalty + battery_penalty

        reward = raw_reward
        if reward_mode == "dense" and use_reward_tanh:
            reward = math.tanh(raw_reward)
        reward = reward + fail_penalty

        dist_reward = 0.0
        term_topk = 0.0
        term_assoc = 0.0
        term_dist = 0.0
        term_dist_delta = 0.0
        term_bw_align = 0.0
        term_sat_score = 0.0
        self.prev_arrival_sum = arrival_sum
        self.prev_q_norm_active = q_norm_active

        self.last_reward_parts = {
            "service_ratio": service_ratio,
            "drop_ratio": drop_ratio,
            "arrival_ref": arrival_ref,
            "b_pre_steps": b_pre_steps,
            "x_acc": x_acc,
            "x_rel": x_rel,
            "g_pre": g_pre,
            "d_pre": d_pre,
            "processed_ratio_eval": processed_ratio_eval,
            "drop_ratio_eval": drop_ratio_eval,
            "pre_backlog_steps_eval": pre_backlog_steps_eval,
            "sat_overlap_eval": sat_overlap_eval,
            "assoc_centroid_dist_norm_mean": assoc_centroid_dist_norm_mean,
            "assoc_centroid_valid_uav_count": assoc_centroid_valid_uav_count,
            "assoc_centroid_valid_uav_frac": assoc_centroid_valid_uav_frac,
            "D_sys_report": D_sys_report,
            "drop_sum": drop_sum,
            "drop_sum_active": drop_sum_active,
            "expire_sum": expire_sum,
            "gu_drop_sum": gu_drop_sum,
            "uav_drop_sum": uav_drop_sum,
            "sat_drop_sum": sat_drop_sum,
            "drop_event": drop_event,
            "arrival_sum": arrival_sum,
            "outflow_sum": outflow_sum,
            "backhaul_sum": backhaul_sum,
            "sat_processed_sum": sat_processed_sum,
            "service_norm": service_norm,
            "drop_norm": drop_norm,
            "gu_drop_norm": gu_drop_norm,
            "uav_drop_norm": uav_drop_norm,
            "sat_drop_norm": sat_drop_norm,
            "throughput_access_norm": throughput_access_norm,
            "throughput_backhaul_norm": throughput_backhaul_norm,
            "sat_processed_norm": sat_processed_norm,
            "outflow_arrival_ratio_step": outflow_arrival_ratio_step,
            "sat_incoming_arrival_ratio_step": sat_incoming_arrival_ratio_step,
            "sat_processed_arrival_ratio_step": sat_processed_arrival_ratio_step,
            "sat_processed_incoming_ratio_step": sat_processed_incoming_ratio_step,
            "gu_drop_ratio_step": gu_drop_ratio_step,
            "uav_drop_ratio_step": uav_drop_ratio_step,
            "sat_drop_ratio_step": sat_drop_ratio_step,
            "expire_ratio_step": expire_ratio_step,
            "queue_pen": queue_term,
            "queue_pen_gu": queue_gu,
            "queue_pen_uav": queue_uav,
            "queue_pen_sat": queue_sat,
            "gu_queue_fill_fraction": gu_queue_fill_fraction,
            "uav_queue_fill_fraction": uav_queue_fill_fraction,
            "sat_queue_fill_fraction": sat_queue_fill_fraction,
            "gu_queue_arrival_steps": gu_queue_arrival_steps,
            "uav_queue_arrival_steps": uav_queue_arrival_steps,
            "sat_queue_arrival_steps": sat_queue_arrival_steps,
            "queue_topk": queue_topk,
            "queue_total": q_total,
            "queue_total_active": q_total_active,
            "assoc_ratio": assoc_ratio,
            "assoc_unfair_max_gu_count": assoc_unfair_max_gu_count,
            "assoc_unfair_step": assoc_unfair_step,
            "queue_delta": queue_delta,
            "queue_delta_mode": queue_delta_mode,
            "queue_delta_gu": queue_delta_gu,
            "queue_delta_uav": queue_delta_uav,
            "queue_delta_sat": queue_delta_sat,
            "q_norm_active": q_norm_active,
            "prev_q_norm_active": prev_q_norm_active,
            "q_norm_delta": q_norm_delta,
            "q_norm_tail_q0": q_norm_tail_q0,
            "q_norm_tail_excess": q_norm_tail_excess,
            "queue_weight": queue_weight,
            "q_delta_weight": q_delta_weight,
            "crash_weight": crash_weight,
            "centroid_transfer_ratio": centroid_transfer_ratio,
            "centroid_eta": centroid_eta,
            "dist_reward": dist_reward,
            "dist_delta": dist_delta,
            "centroid_reward": centroid_reward,
            "centroid_dist_mean": centroid_dist_mean,
            "bw_align": bw_align,
            "sat_score": sat_score,
            "energy_reward": float(r_energy),
            "collision_event": 1.0 if collision_now else 0.0,
            "collision_penalty": collision_penalty,
            "battery_penalty": battery_penalty,
            "fail_penalty": fail_penalty,
            "arrival_rate_eff": float(getattr(self, "last_arrival_rate", cfg.task_arrival_rate)),
            "overflow_risk_mean": (
                float(np.mean(np.asarray(getattr(self, "last_gu_urgency_risk", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            ),
            "downstream_pressure_mean": (
                float(np.mean(np.asarray(getattr(self, "last_gu_downstream_pressure", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            ),
            "service_gap_mean": (
                float(np.mean(np.asarray(getattr(self, "last_gu_service_gap", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            ),
            "service_gap_risk_mean": float(service_gap_risk_mean),
            "deadline_slack_mean": (
                float(np.mean(np.asarray(getattr(self, "last_gu_deadline_slack", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            ),
            "deadline_risk_mean": (
                float(np.mean(np.asarray(getattr(self, "last_gu_deadline_risk", np.zeros((cfg.num_gu,), dtype=np.float32)), dtype=np.float32)))
                if cfg.num_gu > 0
                else 0.0
            ),
            "avoidance_eta_eff": float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta)),
            "avoidance_eta_exec": float(getattr(self, "last_avoidance_eta_exec", cfg.avoidance_eta)),
            "avoidance_collision_rate_ema": float(getattr(self, "avoidance_collision_rate_ema", 0.0)),
            "avoidance_prev_episode_collision_rate": float(getattr(self, "prev_episode_collision_rate", 0.0)),
            "filter_active_ratio": float(getattr(self, "last_filter_active_ratio", 0.0)),
            "projected_delta_norm_mean": float(getattr(self, "last_projected_delta_norm_mean", 0.0)),
            "fallback_count": float(getattr(self, "last_fallback_count", 0.0)),
            "boundary_filter_count": float(getattr(self, "last_boundary_filter_count", 0.0)),
            "pairwise_filter_count": float(getattr(self, "last_pairwise_filter_count", 0.0)),
            "pairwise_filter_active_ratio": float(getattr(self, "last_pairwise_filter_active_ratio", 0.0)),
            "pairwise_projected_delta_norm": float(getattr(self, "last_pairwise_projected_delta_norm", 0.0)),
            "pairwise_fallback_count": float(getattr(self, "last_pairwise_fallback_count", 0.0)),
            "pairwise_candidate_infeasible_count": float(
                getattr(self, "last_pairwise_candidate_infeasible_count", 0.0)
            ),
            "safety_shield_active": float(getattr(self, "last_safety_shield_active", 0.0)),
            "safety_shield_feasible": float(getattr(self, "last_safety_shield_feasible", 1.0)),
            "safety_shield_delta_norm": float(getattr(self, "last_safety_shield_delta_norm", 0.0)),
            "safety_shield_delta_norm_max": float(getattr(self, "last_safety_shield_delta_norm_max", 0.0)),
            "safety_shield_min_margin_before": float(getattr(self, "last_safety_shield_min_margin_before", 0.0)),
            "safety_shield_min_margin_after": float(getattr(self, "last_safety_shield_min_margin_after", 0.0)),
            "safety_shield_min_distance_after": float(
                getattr(self, "last_safety_shield_min_distance_after", 0.0)
            ),
            "safety_shield_pair_count": float(getattr(self, "last_safety_shield_pair_count", 0.0)),
            "term_service": term_service,
            "term_drop": term_drop,
            "term_pre_drop": term_pre_drop,
            "term_pre_overflow_risk": term_pre_overflow_risk,
            "term_pre_service_gap": term_pre_service_gap,
            "term_drop_gu": term_drop_gu,
            "term_drop_uav": term_drop_uav,
            "term_drop_sat": term_drop_sat,
            "term_drop_step": term_drop_step,
            "term_queue": term_queue,
            "term_pre_backlog": term_pre_backlog,
            "term_topk": term_topk,
            "term_assoc": term_assoc,
            "term_q_delta": term_q_delta,
            "term_throughput_access": term_throughput_access,
            "term_throughput_backhaul": term_throughput_backhaul,
            "term_access": term_access,
            "term_relay": term_relay,
            "term_queue_gu_arrival": term_queue_gu_arrival,
            "term_dist": term_dist,
            "term_dist_delta": term_dist_delta,
            "term_centroid": term_centroid,
            "term_bw_align": term_bw_align,
            "term_sat_score": term_sat_score,
            "term_energy": float(term_energy),
            "term_accel": term_accel,
            "intervention_norm": intervention_norm,
            "intervention_rate": intervention_rate,
            "intervention_norm_top1": intervention_norm_top1,
            "danger_imitation_active_rate": danger_imitation_active_rate,
            "close_risk": close_risk,
            "term_close_risk": term_close_risk,
            "reward_raw": raw_reward,
            "reward_mode_active": reward_mode,
            "bw_weighted_workload_delta_reward": float(reward_weighted_workload_delta),
            "bw_relative_weighted_workload_delta_reward": float(reward_relative_weighted_workload_delta),
            "bw_weighted_workload_level_reward": float(reward_weighted_workload_level),
            "bw_gu_queue_level_reward": float(reward_gu_queue_level),
            "bw_system_queue_level_reward": float(reward_system_queue_level),
            "bw_gu_service_queue_reward": float(reward_gu_service_queue),
        }
        metric_quantum = _semantic_quantum(cfg, "structured_summary_metric_quantum", 1.0e-5)
        for key, value in list(self.last_reward_parts.items()):
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if metric_quantum <= 0.0:
                    self.last_reward_parts[key] = float(value)
                else:
                    self.last_reward_parts[key] = float(
                        _quantize_numeric_contract_np(cfg, value, quantum=metric_quantum, dtype=np.float32).reshape(())
                    )
        return _quantize_metric_scalar(cfg, float(reward))

    def _compute_close_risk_stats(self, require_enabled: bool = True) -> Tuple[float, np.ndarray]:
        cfg = self.cfg
        if cfg.num_uav < 2:
            return 0.0, np.zeros((cfg.num_uav,), dtype=np.float32)
        if require_enabled and not bool(getattr(cfg, "close_risk_enabled", False)):
            return 0.0, np.zeros((cfg.num_uav,), dtype=np.float32)

        d_alert = float(cfg.avoidance_alert_factor) * float(cfg.d_safe)
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
        close_risk_cap = max(float(getattr(cfg, "close_risk_cap", 2.0) or 0.0), 0.0)
        dist_denom = max(trigger_dist - d_alert, 1e-6)
        close_scale = max(closing_speed_thresh, 1e-6)

        close_risk = 0.0
        pair_count = 0
        close_risk_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
        for i in range(cfg.num_uav):
            for j in range(i + 1, cfg.num_uav):
                diff = self.uav_pos[i] - self.uav_pos[j]
                dist = float(np.linalg.norm(diff))
                if dist <= 1e-6:
                    continue
                pair_count += 1
                rel_vel = self.uav_vel[i] - self.uav_vel[j]
                closing_speed = max(float(-(np.dot(diff, rel_vel) / dist)), 0.0)
                if closing_speed <= closing_speed_thresh or dist >= trigger_dist:
                    continue
                if prealert_mode == "ttc" and dist >= d_alert:
                    if prealert_ttc_limit <= 0.0:
                        continue
                    ttc_to_alert = (dist - d_alert) / max(closing_speed, 1e-6)
                    if ttc_to_alert >= prealert_ttc_limit:
                        continue

                dist_ratio = float(np.clip((trigger_dist - dist) / dist_denom, 0.0, 1.0))
                close_ratio = float(np.clip((closing_speed - closing_speed_thresh) / close_scale, 0.0, close_risk_cap))
                pair_risk = dist_ratio * close_ratio
                close_risk += pair_risk
                close_risk_uav[i] = max(float(close_risk_uav[i]), pair_risk)
                close_risk_uav[j] = max(float(close_risk_uav[j]), pair_risk)

        if pair_count <= 0:
            return 0.0, close_risk_uav
        return close_risk / float(pair_count), close_risk_uav

    def _compute_close_risk(self) -> float:
        close_risk, _ = self._compute_close_risk_stats(require_enabled=True)
        return close_risk

    def _check_collision(self) -> bool:
        cfg = self.cfg
        for i in range(cfg.num_uav):
            for j in range(i + 1, cfg.num_uav):
                if np.linalg.norm(self.uav_pos[i] - self.uav_pos[j]) < cfg.d_safe:
                    return True
        return False

    def _local_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        cfg = self.cfg
        lat0 = math.radians(cfg.ref_lat_deg)
        lon0 = math.radians(cfg.ref_lon_deg)
        lat = lat0 + y / cfg.r_earth
        lon = lon0 + x / normalize_scale(cfg.r_earth * math.cos(lat0))
        return lat, lon

    def _local_to_ecef(self, x: float, y: float, alt: float) -> np.ndarray:
        cfg = self.cfg
        lat, lon = self._local_to_latlon(x, y)
        r = cfg.r_earth + alt
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)
        return np.array(
            [
                r * cos_lat * cos_lon,
                r * cos_lat * sin_lon,
                r * sin_lat,
            ],
            dtype=np.float32,
        )

    def _enu_to_ecef(self, east: float, north: float, up: float, lat: float, lon: float) -> np.ndarray:
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        t = np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ],
            dtype=np.float32,
        )
        return t @ np.array([east, north, up], dtype=np.float32)

    def _elevation_angle(self, u: int, l: int, sat_pos: np.ndarray) -> float:
        q = self._uav_ecef(u)
        d = float(geometry_denominator(np.linalg.norm(sat_pos[l] - q)))
        arg = (self._sat_orbit_radius_sq - self._uav_orbit_radius_sq - d ** 2) / (
            normalize_scale(2.0 * self._uav_orbit_radius * d)
        )
        arg = np.clip(arg, -1.0, 1.0)
        return float(math.asin(arg))

    def _get_elevation_matrix(self, sat_pos: np.ndarray | None = None) -> np.ndarray:
        if self._cached_elevation_t == self.t and self._cached_elevation_matrix is not None:
            return self._cached_elevation_matrix

        if sat_pos is None:
            sat_pos, _ = self._get_orbit_states()
        if self._cached_uav_ecef is None:
            self._refresh_uav_cache()

        rel = sat_pos[None, :, :] - self._cached_uav_ecef[:, None, :]
        dist = geometry_denominator(np.linalg.norm(rel, axis=2))
        arg = (self._sat_orbit_radius_sq - self._uav_orbit_radius_sq - dist * dist) / (
            geometry_denominator(2.0 * self._uav_orbit_radius * dist)
        )
        np.clip(arg, -1.0, 1.0, out=arg)
        self._cached_elevation_matrix = np.arcsin(arg).astype(np.float32, copy=False)
        self._cached_elevation_t = self.t
        return self._cached_elevation_matrix

    def _fly_power(self, speed: np.ndarray | float) -> np.ndarray:
        cfg = self.cfg
        v = np.asarray(speed, dtype=np.float32)
        if cfg.energy_model == "rotor":
            p0 = cfg.rotor_p0
            pi = cfg.rotor_pi
            u_tip = cfg.rotor_u_tip
            v0 = cfg.rotor_v0
            d0 = cfg.rotor_d0
            rho = cfg.rotor_rho
            s = cfg.rotor_s
            area = cfg.rotor_area
            term1 = p0 * (1.0 + 3.0 * (v ** 2) / (u_tip ** 2))
            term2 = pi * np.sqrt(
                np.sqrt(1.0 + (v ** 4) / (4.0 * (v0 ** 4))) - (v ** 2) / (2.0 * (v0 ** 2))
            )
            term3 = 0.5 * d0 * rho * s * area * (v ** 3)
            return term1 + term2 + term3
        return cfg.p_fly_base + cfg.p_fly_coeff * (v ** 2)

    def _energy_scale(self) -> float:
        cfg = self.cfg
        p_fly = float(self._fly_power(cfg.v_max))
        p_comm = cfg.p_comm_link * max(1, cfg.N_RF)
        return p_fly + p_comm

    @staticmethod
    def _minmax_normalize(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=np.float32)
        if vals.size == 0:
            return vals
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        if v_max - v_min <= NORMALIZATION_DENOM_EPS:
            return np.zeros_like(vals, dtype=np.float32)
        return ((vals - v_min) / (v_max - v_min)).astype(np.float32, copy=False)

    @staticmethod
    def _topk_descending_stable(values: np.ndarray, k: int) -> np.ndarray:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        if vals.size == 0 or k <= 0:
            return np.zeros((0,), dtype=np.int64)
        if k >= vals.size:
            return np.argsort(-vals, kind="stable").astype(np.int64, copy=False)
        kth = float(np.partition(vals, vals.size - k)[vals.size - k])
        greater = np.flatnonzero(vals > kth)
        greater = greater[np.argsort(-vals[greater], kind="stable")]
        need = k - int(greater.size)
        if need <= 0:
            return greater[:k].astype(np.int64, copy=False)
        equal = np.flatnonzero(vals == kth)
        return np.concatenate((greater, equal[:need]), axis=0).astype(np.int64, copy=False)

    @staticmethod
    def _slice_rank_data(rank_data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        return {
            key: np.asarray(value, dtype=np.float32)[idx].astype(np.float32, copy=False)
            for key, value in rank_data.items()
        }

    def _sat_candidate_order_data(
        self,
        u: int,
        sat_indices: np.ndarray,
        sat_pos: np.ndarray,
        elev_values: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        mode = str(getattr(self.cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower()
        if mode not in {"elevation", "score"}:
            raise ValueError(f"Unsupported sat_candidate_mode='{self.cfg.sat_candidate_mode}'")
        if mode == "score":
            return self._sat_candidate_rank_data(u, sat_indices, sat_pos, elev_values=elev_values)

        sat_idx = np.asarray(sat_indices, dtype=np.int32)
        if sat_idx.size == 0:
            empty_f = np.zeros((0,), dtype=np.float32)
            return {"elevation": empty_f, "rank_value": empty_f}
        if elev_values is None:
            elev_matrix = self._get_elevation_matrix(sat_pos)
            elev = np.asarray(elev_matrix[u, sat_idx], dtype=np.float32)
        else:
            elev = np.asarray(elev_values, dtype=np.float32)
        return {
            "elevation": elev,
            "rank_value": elev.astype(np.float32, copy=False),
        }

    def _sat_candidate_rank_data(
        self,
        u: int,
        sat_indices: np.ndarray,
        sat_pos: np.ndarray,
        elev_values: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        sat_idx = np.asarray(sat_indices, dtype=np.int32)
        if sat_idx.size == 0:
            empty_f = np.zeros((0,), dtype=np.float32)
            return {
                "elevation": empty_f,
                "distance": empty_f,
                "spectral_efficiency": empty_f,
                "queue_norm": empty_f,
                "score": empty_f,
                "rank_value": empty_f,
            }

        if elev_values is None:
            elev_matrix = self._get_elevation_matrix(sat_pos)
            elev = np.asarray(elev_matrix[u, sat_idx], dtype=np.float32)
        else:
            elev = np.asarray(elev_values, dtype=np.float32)

        rel_pos = sat_pos[sat_idx] - self._uav_ecef(u)[None, :]
        dist = geometry_denominator(np.linalg.norm(rel_pos, axis=1))
        gain = self._backhaul_gain_const / geometry_denominator(dist * dist)
        if cfg.atm_loss_enabled:
            atm_loss = channel.atmospheric_loss_db(elev, cfg.atm_loss_db)
            gain = gain * (10 ** (-atm_loss / 10.0))
        projected_bw = self._projected_sat_bandwidth(u, sat_idx)
        snr = channel.snr_linear(
            cfg.uav_tx_power,
            gain,
            cfg.noise_density,
            projected_bw,
            noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
        )
        se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
        queue_norm = (self.sat_queue[sat_idx] / normalize_scale(cfg.queue_max_sat)).astype(np.float32, copy=False)

        elev_norm = self._minmax_normalize(elev)
        se_norm = self._minmax_normalize(se)
        score = (
            float(getattr(cfg, "sat_candidate_elevation_weight", 1.0) or 0.0) * elev_norm
            + float(getattr(cfg, "sat_candidate_se_weight", 1.0) or 0.0) * se_norm
            - float(getattr(cfg, "sat_candidate_queue_weight", 1.0) or 0.0) * queue_norm
        ).astype(np.float32, copy=False)

        mode = str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower()
        if mode not in {"elevation", "score"}:
            raise ValueError(f"Unsupported sat_candidate_mode='{cfg.sat_candidate_mode}'")
        rank_value = elev if mode == "elevation" else score
        return {
            "elevation": elev,
            "distance": dist.astype(np.float32, copy=False),
            "spectral_efficiency": se,
            "queue_norm": queue_norm,
            "score": score,
            "rank_value": rank_value.astype(np.float32, copy=False),
        }

    def _sat_candidate_order(self, rank_data: Dict[str, np.ndarray]) -> np.ndarray:
        rank_value = np.asarray(rank_data["rank_value"], dtype=np.float32)
        elev = np.asarray(rank_data["elevation"], dtype=np.float32)
        if rank_value.size == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.lexsort((-elev.astype(np.float32), -rank_value.astype(np.float32)))

    def _sat_candidate_topk_order(self, rank_data: Dict[str, np.ndarray], k: int) -> np.ndarray:
        rank_value = np.asarray(rank_data["rank_value"], dtype=np.float32)
        elev = np.asarray(rank_data["elevation"], dtype=np.float32)
        if rank_value.size == 0 or k <= 0:
            return np.zeros((0,), dtype=np.int64)
        if k >= rank_value.size:
            return self._sat_candidate_order(rank_data)

        threshold = float(np.partition(rank_value.astype(np.float32), rank_value.size - k)[rank_value.size - k])
        greater = np.flatnonzero(rank_value > threshold)
        need = k - int(greater.size)
        if greater.size > 0:
            greater_order = np.lexsort(
                (-elev[greater].astype(np.float32), -rank_value[greater].astype(np.float32))
            )
            greater = greater[greater_order]
        if need <= 0:
            return greater[:k].astype(np.int64, copy=False)

        tied = np.flatnonzero(rank_value == threshold)
        if tied.size > 0:
            tied_order = self._topk_descending_stable(elev[tied], need)
            tied = tied[tied_order]
        selected = np.concatenate((greater, tied[:need]), axis=0)
        selected_order = np.lexsort(
            (-elev[selected].astype(np.float32), -rank_value[selected].astype(np.float32))
        )
        return selected[selected_order].astype(np.int64, copy=False)

    def _summarize_visible_counts(self, name: str, counts: np.ndarray, out: Dict[str, float | str]) -> None:
        vals = np.asarray(counts, dtype=np.float32)
        if vals.size == 0:
            out[f"{name}_mean"] = 0.0
            out[f"{name}_p50"] = 0.0
            out[f"{name}_p90"] = 0.0
            out[f"{name}_fraction_le_1"] = 0.0
            out[f"{name}_fraction_ge_3"] = 0.0
            out[f"{name}_fraction_ge_5"] = 0.0
            return
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_p50"] = float(np.percentile(vals, 50))
        out[f"{name}_p90"] = float(np.percentile(vals, 90))
        out[f"{name}_fraction_le_1"] = float(np.mean(vals <= 1.0))
        out[f"{name}_fraction_ge_3"] = float(np.mean(vals >= 3.0))
        out[f"{name}_fraction_ge_5"] = float(np.mean(vals >= 5.0))

    def _agent_visible_info(self, u: int) -> Dict[str, object]:
        return {
            "visible_raw_count": int(self.last_visible_raw_counts[u]),
            "visible_kept_count": int(self.last_visible_kept_counts[u]),
            "visible_raw_candidates": list(self.last_visible_raw_candidates[u]),
            "visible_candidates": list(self.last_visible_candidates[u]),
            "visible_candidate_rank_values": list(self.last_visible_candidate_rank_values[u]),
            "visible_candidate_scores": list(self.last_visible_candidate_scores[u]),
            "visible_candidate_rank_gap_top1_top2": float(self.last_visible_candidate_rank_gap_top1_top2[u]),
            "visible_candidate_score_gap_top1_top2": float(self.last_visible_candidate_score_gap_top1_top2[u]),
            "visible_stats": dict(self.last_visible_stats),
        }

    def _visible_sats_sorted(self, sat_pos: np.ndarray, *, record_stats: bool = True) -> List[List[int]]:
        cfg = self.cfg
        elev_matrix = self._get_elevation_matrix(sat_pos)
        mode = str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower()
        if mode not in {"elevation", "score"}:
            raise ValueError(f"Unsupported sat_candidate_mode='{cfg.sat_candidate_mode}'")
        visible: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        if not record_stats:
            max_keep = cfg.visible_sats_max if cfg.visible_sats_max is not None else cfg.sats_obs_max
            max_keep = max(int(max_keep), 0)
            for u in range(cfg.num_uav):
                elev_u = elev_matrix[u]
                above = np.nonzero(elev_u >= cfg.theta_min_rad)[0].astype(np.int32, copy=False)
                raw_keep = min(max_keep, int(above.size))
                above_data = self._sat_candidate_order_data(u, above, sat_pos, elev_values=elev_u[above])
                above_order = self._sat_candidate_topk_order(above_data, raw_keep)
                visible[u] = above[above_order][:max_keep].tolist()
            return visible
        raw_candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        kept_rank_values: List[List[float]] = [[] for _ in range(cfg.num_uav)]
        kept_scores: List[List[float]] = [[] for _ in range(cfg.num_uav)]
        raw_counts = np.zeros((cfg.num_uav,), dtype=np.int32)
        kept_counts = np.zeros((cfg.num_uav,), dtype=np.int32)
        rank_gap_top1_top2 = np.zeros((cfg.num_uav,), dtype=np.float32)
        score_gap_top1_top2 = np.zeros((cfg.num_uav,), dtype=np.float32)
        dist_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        elev_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        se_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        queue_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        max_keep = cfg.visible_sats_max if cfg.visible_sats_max is not None else cfg.sats_obs_max
        max_keep = max(int(max_keep), 0)
        for u in range(cfg.num_uav):
            elev_u = elev_matrix[u]
            above = np.nonzero(elev_u >= cfg.theta_min_rad)[0].astype(np.int32, copy=False)
            raw_counts[u] = int(above.size)
            raw_keep = min(max_keep, int(above.size))
            above_data = self._sat_candidate_order_data(u, above, sat_pos, elev_values=elev_u[above])
            above_order = self._sat_candidate_topk_order(above_data, raw_keep)
            above_sorted = above[above_order]
            raw_candidates[u] = above_sorted.tolist()
            # Never backfill below-threshold satellites; unused obs slots stay zero-padded.
            kept = above_sorted[:max_keep]
            kept_counts[u] = int(kept.size)
            visible[u] = kept.tolist()
            if mode == "score" and kept.size > 0:
                kept_data = self._slice_rank_data(above_data, above_order[: kept.size])
            else:
                kept_data = self._sat_candidate_rank_data(u, kept, sat_pos, elev_values=elev_u[kept])
            kept_rank_values[u] = kept_data["rank_value"].astype(np.float32, copy=False).tolist()
            kept_scores[u] = kept_data["score"].astype(np.float32, copy=False).tolist()
            if kept.size >= 2:
                rank_gap_top1_top2[u] = float(kept_data["rank_value"][0] - kept_data["rank_value"][1])
                score_gap_top1_top2[u] = float(kept_data["score"][0] - kept_data["score"][1])
            if kept.size > 0:
                dist_std[u] = float(np.std(kept_data["distance"]))
                elev_std[u] = float(np.std(kept_data["elevation"]))
                se_std[u] = float(np.std(kept_data["spectral_efficiency"]))
                queue_std[u] = float(np.std(kept_data["queue_norm"]))

        self.last_visible_raw_counts = raw_counts
        self.last_visible_kept_counts = kept_counts
        self.last_visible_raw_candidates = raw_candidates
        self.last_visible_candidates = [list(v) for v in visible]
        self.last_visible_candidate_rank_values = kept_rank_values
        self.last_visible_candidate_scores = kept_scores
        self.last_visible_candidate_rank_gap_top1_top2 = rank_gap_top1_top2
        self.last_visible_candidate_score_gap_top1_top2 = score_gap_top1_top2
        self.last_visible_candidate_dist_std = dist_std
        self.last_visible_candidate_elevation_std = elev_std
        self.last_visible_candidate_se_std = se_std
        self.last_visible_candidate_queue_std = queue_std
        stats: Dict[str, float | str] = {
            "candidate_mode": mode,
            "visible_truncation_fraction": float(np.mean(raw_counts > kept_counts)) if raw_counts.size else 0.0,
            "candidate_dist_std_mean": float(np.mean(dist_std)) if dist_std.size else 0.0,
            "candidate_elevation_std_mean": float(np.mean(elev_std)) if elev_std.size else 0.0,
            "candidate_se_std_mean": float(np.mean(se_std)) if se_std.size else 0.0,
            "candidate_queue_std_mean": float(np.mean(queue_std)) if queue_std.size else 0.0,
            "candidate_rank_gap_top1_top2_mean": float(np.mean(rank_gap_top1_top2))
            if rank_gap_top1_top2.size
            else 0.0,
            "candidate_score_gap_top1_top2_mean": float(np.mean(score_gap_top1_top2))
            if score_gap_top1_top2.size
            else 0.0,
        }
        self._summarize_visible_counts("raw_visible_count", raw_counts, stats)
        self._summarize_visible_counts("kept_visible_count", kept_counts, stats)
        self.last_visible_stats = stats
        return visible

    def _uav_ecef(self, u: int) -> np.ndarray:
        if self._cached_uav_ecef is None:
            self._refresh_uav_cache()
        return self._cached_uav_ecef[u]

    def _uav_vel_ecef(self, u: int) -> np.ndarray:
        if self._cached_uav_vel_ecef is None:
            self._refresh_uav_cache()
        return self._cached_uav_vel_ecef[u]

    def _doppler(self, u: int, l: int, sat_pos: np.ndarray, sat_vel: np.ndarray) -> float:
        cfg = self.cfg
        r_vec = sat_pos[l] - self._uav_ecef(u)
        v_rel = sat_vel[l] - self._uav_vel_ecef(u)
        denom = geometry_denominator(np.linalg.norm(r_vec))
        proj = float(np.dot(v_rel, r_vec) / denom)
        return (_backhaul_carrier_freq_from_cfg(cfg) / cfg.speed_of_light) * proj

    def _doppler_many(self, u: int, sat_idx: np.ndarray, sat_pos: np.ndarray, sat_vel: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        sat_idx = np.asarray(sat_idx, dtype=np.int32)
        if sat_idx.size == 0:
            return np.zeros((0,), dtype=np.float32)
        r_vec = sat_pos[sat_idx] - self._uav_ecef(u)[None, :]
        v_rel = sat_vel[sat_idx] - self._uav_vel_ecef(u)[None, :]
        denom = geometry_denominator(np.linalg.norm(r_vec, axis=1))
        proj = np.einsum("ij,ij->i", v_rel, r_vec) / denom
        return ((_backhaul_carrier_freq_from_cfg(cfg) / cfg.speed_of_light) * proj).astype(np.float32, copy=False)

    def _effective_doppler(self, u: int, sat_idx: int, nu: float) -> Tuple[float, float]:
        if not self._doppler_precomp_enabled():
            return float(nu), 0.0
        nu_eff = float(self._doppler_residual_state_hz[int(u), int(sat_idx)])
        nu_hat = float(nu) - nu_eff
        return nu_eff, nu_hat

    def _effective_doppler_array(self, u: int, sat_idx: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nu_arr = np.asarray(nu, dtype=np.float32)
        sat_arr = np.asarray(sat_idx, dtype=np.int32)
        if nu_arr.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty
        if not self._doppler_precomp_enabled():
            return nu_arr.astype(np.float32, copy=False), np.zeros_like(nu_arr, dtype=np.float32)
        nu_eff = np.asarray(self._doppler_residual_state_hz[int(u), sat_arr], dtype=np.float32)
        nu_hat = (nu_arr - nu_eff).astype(np.float32, copy=False)
        return nu_eff, nu_hat

    def _backhaul_loss_factor(self, theta_rad: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        theta = np.asarray(theta_rad, dtype=np.float32)
        factor = np.ones_like(theta, dtype=np.float32)
        if cfg.atm_loss_enabled:
            atm_loss = channel.atmospheric_loss_db(theta, cfg.atm_loss_db)
            factor *= 10 ** (-atm_loss / 10.0)
        if getattr(cfg, "rain_loss_enabled", False):
            rain_loss = channel.rain_attenuation_db(
                theta,
                _backhaul_carrier_freq_from_cfg(cfg),
                cfg.rain_rate_001_mmph,
                cfg.rain_height_km,
                cfg.uav_height / 1000.0,
                cfg.ref_lat_deg if cfg.rain_lat_deg is None else cfg.rain_lat_deg,
                exceedance_pct=cfg.rain_exceedance_pct,
                polarization_tilt_deg=cfg.rain_polarization_tilt_deg,
            )
            factor *= 10 ** (-rain_loss / 10.0)
        return factor.astype(np.float32, copy=False)

    def _get_backhaul_loss_matrix(self, sat_pos: np.ndarray | None = None) -> np.ndarray | None:
        cfg = self.cfg
        if not (cfg.atm_loss_enabled or getattr(cfg, "rain_loss_enabled", False)):
            return None
        if self._cached_backhaul_loss_t == self.t and self._cached_backhaul_loss_matrix is not None:
            return self._cached_backhaul_loss_matrix
        elev_matrix = self._get_elevation_matrix(sat_pos)
        self._cached_backhaul_loss_matrix = self._backhaul_loss_factor(elev_matrix)
        self._cached_backhaul_loss_t = self.t
        return self._cached_backhaul_loss_matrix

    def _cache_sat_obs(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        visible: List[List[int]],
    ) -> None:
        cfg = self.cfg
        sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, self.sat_dim), dtype=np.float32)
        sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        sat_valid_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        elev_matrix = self._get_elevation_matrix(sat_pos)
        loss_matrix = self._get_backhaul_loss_matrix(sat_pos)
        for u in range(cfg.num_uav):
            sat_idx = np.asarray(visible[u][: cfg.sats_obs_max], dtype=np.int32)
            if sat_idx.size == 0:
                continue
            if u < len(self.last_sat_selection):
                current_sats = np.asarray(self.last_sat_selection[u], dtype=np.int32)
            else:
                current_sats = np.zeros((0,), dtype=np.int32)
            projected_count = self._projected_sat_connection_count(u, sat_idx)
            projected_bw = self._projected_sat_bandwidth(u, sat_idx)
            rel_pos = sat_pos[sat_idx] - self._uav_ecef(u)[None, :]
            rel_vel = sat_vel[sat_idx] - self._uav_vel_ecef(u)[None, :]
            d = geometry_denominator(np.linalg.norm(rel_pos, axis=1))
            gain = self._backhaul_gain_const / geometry_denominator(d * d)
            if loss_matrix is not None:
                gain = gain * loss_matrix[u, sat_idx]
            if cfg.doppler_enabled or cfg.doppler_atten_enabled or cfg.doppler_observed:
                raw_nu = self._doppler_many(u, sat_idx, sat_pos, sat_vel)
                nu_eff, _ = self._effective_doppler_array(u, sat_idx, raw_nu)
            else:
                nu_eff = np.zeros((sat_idx.size,), dtype=np.float32)
            snr = channel.snr_linear(
                cfg.uav_tx_power,
                gain,
                cfg.noise_density,
                projected_bw,
                noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
            )
            if cfg.doppler_observed and cfg.doppler_atten_enabled:
                snr = snr * channel.doppler_attenuation(nu_eff, cfg.subcarrier_spacing)
            sat_reward_features = None
            if bool(getattr(cfg, "obs_sat_include_sat_cost", False)):
                sat_reward_features = self._sat_reward_aligned_feature_dict(normalized=True)

            n = sat_idx.size
            sat_obs[u, :n, 0:3] = rel_pos / (cfg.r_earth + cfg.sat_height)
            sat_obs[u, :n, 3:6] = rel_vel / (cfg.r_earth + cfg.sat_height)
            sat_obs[u, :n, 6] = nu_eff / max(cfg.nu_max, 1.0)
            sat_obs[u, :n, 7] = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
            sat_obs[u, :n, 8] = self.sat_queue[sat_idx] / cfg.queue_max_sat
            sat_obs[u, :n, 9] = self.last_sat_connection_counts[sat_idx] / max(float(cfg.num_uav), 1.0)
            sat_obs[u, :n, 10] = 1.0 / projected_count
            sat_obs[u, :n, 11] = np.isin(sat_idx, current_sats).astype(np.float32, copy=False)
            feat_col = 12
            if sat_reward_features is not None:
                sat_obs[u, :n, feat_col] = np.asarray(
                    sat_reward_features["sat_cost"][sat_idx],
                    dtype=np.float32,
                )
                feat_col += 1
            sat_mask[u, :n] = 1.0
            elev_ok = elev_matrix[u, sat_idx] >= cfg.theta_min_rad
            if cfg.doppler_enabled:
                doppler_ok = np.abs(nu_eff) <= cfg.nu_max
                sat_valid_mask[u, :n] = (elev_ok & doppler_ok).astype(np.float32, copy=False)
            else:
                sat_valid_mask[u, :n] = elev_ok.astype(np.float32, copy=False)
        self._cached_sat_obs = sat_obs
        self._cached_sat_mask = sat_mask
        self._cached_sat_valid_mask = sat_valid_mask

    def _danger_neighbor_obs(self, u: int) -> np.ndarray:
        cfg = self.cfg
        feat = np.zeros((self.danger_nbr_dim,), dtype=np.float32)
        if cfg.num_uav <= 1:
            return feat

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

        best = None
        best_key = None
        for j in range(cfg.num_uav):
            if j == u:
                continue
            rel_pos = self.uav_pos[j] - self.uav_pos[u]
            dist = float(np.linalg.norm(rel_pos))
            if dist <= 1e-6:
                continue

            rel_vel = self.uav_vel[j] - self.uav_vel[u]
            closing_speed = float(-(np.dot(rel_pos, rel_vel) / dist))
            closing_pos = max(closing_speed, 0.0)
            ttc_to_alert = float("inf")
            if d_alert > 0.0:
                if dist <= d_alert:
                    ttc_to_alert = 0.0
                elif closing_pos > 1e-6:
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
                best = (rel_pos, dist, closing_pos, in_prealert, in_core_alert)

        if best is None:
            return feat

        rel_pos, dist, closing_pos, in_prealert, in_core_alert = best
        direction = rel_pos / max(dist, 1e-6)
        feat[:] = np.array(
            [
                dist / max(cfg.map_size, 1e-6),
                np.clip(closing_pos / max(cfg.v_max, 1e-6), 0.0, 1.0),
                direction[0],
                direction[1],
                1.0,
            ],
            dtype=np.float32,
        )
        return feat

    def _danger_neighbor_obs_batch(self) -> np.ndarray:
        cfg = self.cfg
        danger = np.zeros((cfg.num_uav, self.danger_nbr_dim), dtype=np.float32)
        if cfg.num_uav <= 1:
            return danger

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
        rel_pos_all = self.uav_pos[None, :, :] - self.uav_pos[:, None, :]
        rel_vel_all = self.uav_vel[None, :, :] - self.uav_vel[:, None, :]
        dist_all = np.linalg.norm(rel_pos_all, axis=-1).astype(np.float32, copy=False)

        for u in range(cfg.num_uav):
            best = None
            best_key = None
            for j in range(cfg.num_uav):
                if j == u:
                    continue
                rel_pos = rel_pos_all[u, j]
                dist = float(dist_all[u, j])
                if dist <= 1e-6:
                    continue

                rel_vel = rel_vel_all[u, j]
                closing_speed = float(-(np.dot(rel_pos, rel_vel) / dist))
                closing_pos = max(closing_speed, 0.0)
                ttc_to_alert = float("inf")
                if d_alert > 0.0:
                    if dist <= d_alert:
                        ttc_to_alert = 0.0
                    elif closing_pos > 1e-6:
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
            direction = rel_pos / max(dist, 1e-6)
            danger[u] = np.array(
                [
                    dist / max(cfg.map_size, 1e-6),
                    np.clip(closing_pos / max(cfg.v_max, 1e-6), 0.0, 1.0),
                    direction[0],
                    direction[1],
                    1.0,
                ],
                dtype=np.float32,
            )
        return danger

    def _build_obs_runtime_context(self) -> dict[str, object]:
        cached = getattr(self, "_cached_obs_runtime_context", None)
        if cached is not None and self._obs_runtime_context_is_current(cached):
            return cached
        cfg = self.cfg
        assoc_counts, assoc_rel_centroids, _, _, _, _ = self._assoc_centroid_summary()
        uav_reward_features = (
            self._uav_reward_aligned_feature_dict(normalized=True)
            if bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False))
            else None
        )
        gu_proxy_features = self._gu_proxy_feature_arrays()
        self._ensure_neighbor_cache()
        danger_obs = self._danger_neighbor_obs_batch() if bool(getattr(cfg, "danger_nbr_enabled", False)) else None
        context = {
            "assoc_counts": assoc_counts,
            "assoc_rel_centroids": assoc_rel_centroids,
            "uav_reward_features": uav_reward_features,
            "gu_proxy_features": gu_proxy_features,
            "neighbor_order": np.asarray(self._cached_uav_neighbor_order, dtype=np.int64),
            "danger_obs": danger_obs,
            "_signature": {
                "t": int(self.t),
                "uav_pos": np.asarray(self.uav_pos, dtype=np.float32).copy(),
                "uav_vel": np.asarray(self.uav_vel, dtype=np.float32).copy(),
                "uav_energy": np.asarray(self.uav_energy, dtype=np.float32).copy(),
                "uav_queue": np.asarray(self.uav_queue, dtype=np.float32).copy(),
                "gu_pos": np.asarray(self.gu_pos, dtype=np.float32).copy(),
                "gu_queue": np.asarray(self.gu_queue, dtype=np.float32).copy(),
                "prev_association": np.asarray(self.prev_association, dtype=np.int32).copy(),
                "cached_assoc": np.asarray(self._cached_assoc, dtype=np.int32).copy(),
                "cached_candidates": [list(c) for c in self._cached_candidates],
                "cached_eta": np.asarray(self._cached_eta, dtype=np.float32).copy(),
                "cached_bw_valid_mask": np.asarray(self._cached_bw_valid_mask, dtype=np.float32).copy(),
                "cached_sat_obs": np.asarray(self._cached_sat_obs, dtype=np.float32).copy(),
                "cached_sat_mask": np.asarray(self._cached_sat_mask, dtype=np.float32).copy(),
                "cached_sat_valid_mask": np.asarray(self._cached_sat_valid_mask, dtype=np.float32).copy(),
            },
        }
        self._cached_obs_runtime_context = context
        return context

    def _obs_runtime_context_is_current(self, context: dict[str, object]) -> bool:
        signature = context.get("_signature")
        if not isinstance(signature, dict):
            return False
        if int(signature.get("t", -1)) != int(self.t):
            return False
        if not np.array_equal(np.asarray(signature.get("uav_pos")), np.asarray(self.uav_pos)):
            return False
        if not np.array_equal(np.asarray(signature.get("uav_vel")), np.asarray(self.uav_vel)):
            return False
        if not np.array_equal(np.asarray(signature.get("uav_energy")), np.asarray(self.uav_energy)):
            return False
        if not np.array_equal(np.asarray(signature.get("uav_queue")), np.asarray(self.uav_queue)):
            return False
        if not np.array_equal(np.asarray(signature.get("gu_pos")), np.asarray(self.gu_pos)):
            return False
        if not np.array_equal(np.asarray(signature.get("gu_queue")), np.asarray(self.gu_queue)):
            return False
        if not np.array_equal(np.asarray(signature.get("prev_association")), np.asarray(self.prev_association)):
            return False
        if not np.array_equal(np.asarray(signature.get("cached_assoc")), np.asarray(self._cached_assoc)):
            return False
        if signature.get("cached_candidates") != [list(c) for c in self._cached_candidates]:
            return False
        if not np.array_equal(np.asarray(signature.get("cached_eta")), np.asarray(self._cached_eta)):
            return False
        if not np.array_equal(np.asarray(signature.get("cached_bw_valid_mask")), np.asarray(self._cached_bw_valid_mask)):
            return False
        if not np.array_equal(np.asarray(signature.get("cached_sat_obs")), np.asarray(self._cached_sat_obs)):
            return False
        if not np.array_equal(np.asarray(signature.get("cached_sat_mask")), np.asarray(self._cached_sat_mask)):
            return False
        if not np.array_equal(np.asarray(signature.get("cached_sat_valid_mask")), np.asarray(self._cached_sat_valid_mask)):
            return False
        return True

    def _get_obs_from_context(self, u: int, context: dict[str, object]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        assoc_counts = np.asarray(context["assoc_counts"], dtype=np.float32)
        assoc_rel_centroids = np.asarray(context["assoc_rel_centroids"], dtype=np.float32)
        uav_reward_features = context["uav_reward_features"]
        gu_proxy_features = [np.asarray(feature, dtype=np.float32) for feature in context["gu_proxy_features"]]
        neighbor_order = np.asarray(context["neighbor_order"], dtype=np.int64)
        danger_obs = context["danger_obs"]

        assoc_count_norm = assoc_counts[u] / max(float(cfg.num_gu), 1.0)
        assoc_centroid_rel = assoc_rel_centroids[u]
        own_list = [
            self.uav_pos[u, 0] / cfg.map_size,
            self.uav_pos[u, 1] / cfg.map_size,
            self.uav_vel[u, 0] / cfg.v_max,
            self.uav_vel[u, 1] / cfg.v_max,
            self.uav_energy[u] / normalize_scale(cfg.uav_energy_init),
            self.uav_queue[u] / cfg.queue_max_uav,
            0.0,  # Reserved: do not expose the artificial rollout horizon.
            assoc_count_norm,
            assoc_centroid_rel[0],
            assoc_centroid_rel[1],
        ]
        if uav_reward_features is not None:
            own_list.append(float(np.asarray(uav_reward_features["assoc_uav_cost"], dtype=np.float32)[u]))
        own = np.asarray(own_list, dtype=np.float32)

        users = np.zeros((cfg.users_obs_max, self.user_dim), dtype=np.float32)
        users_mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        bw_valid_mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        assoc = np.asarray(getattr(self, "_cached_assoc", np.full((cfg.num_gu,), -1, dtype=np.int32)), dtype=np.int32)
        eta_by_gu = np.zeros((cfg.num_gu,), dtype=np.float32)
        cand = self._cached_candidates[u] if self._cached_candidates else []
        for slot, k in enumerate(cand[: cfg.users_obs_max]):
            if 0 <= int(k) < cfg.num_gu:
                eta_by_gu[int(k)] = float(self._cached_eta[u, slot])
        for i, k in enumerate(range(min(cfg.num_gu, cfg.users_obs_max))):
            rel = self.gu_pos[k] - self.uav_pos[u]
            users[i, 0:2] = rel / cfg.map_size
            users[i, 2] = self.gu_queue[k] / cfg.queue_max_gu
            users[i, 3] = eta_by_gu[k]
            users[i, 4] = 1.0 if self.prev_association[k] == u else 0.0
            feat_col = 5
            for feature in gu_proxy_features:
                users[i, feat_col] = float(feature[k])
                feat_col += 1
            users_mask[i] = 1.0
            bw_valid_mask[i] = 1.0 if int(assoc[k]) == int(u) else 0.0

        sats = self._cached_sat_obs[u].copy()
        sats_mask = self._cached_sat_mask[u].copy()
        sat_valid_mask = self._cached_sat_valid_mask[u].copy()

        nbrs = np.zeros((cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32)
        nbrs_mask = np.zeros((cfg.nbrs_obs_max,), dtype=np.float32)
        order = neighbor_order[u]
        count = 0
        for idx in order:
            if idx == u:
                continue
            rel_pos = self.uav_pos[idx] - self.uav_pos[u]
            rel_vel = self.uav_vel[idx] - self.uav_vel[u]
            nbrs[count, 0:2] = rel_pos / cfg.map_size
            nbrs[count, 2:4] = rel_vel / cfg.v_max
            nbrs_mask[count] = 1.0
            count += 1
            if count >= cfg.nbrs_obs_max:
                break

        obs = {
            "own": own,
            "users": users,
            "users_mask": users_mask,
            "bw_valid_mask": bw_valid_mask,
            "sats": sats,
            "sats_mask": sats_mask,
            "sat_valid_mask": sat_valid_mask,
            "nbrs": nbrs,
            "nbrs_mask": nbrs_mask,
        }
        if danger_obs is not None:
            obs["danger_nbr"] = np.asarray(danger_obs, dtype=np.float32)[u].copy()
        return obs

    def _build_all_obs_from_context(self, context: dict[str, object]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            agent: self._get_obs_from_context(u, context)
            for u, agent in enumerate(self.agents)
        }

    def _build_all_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._build_all_obs_from_context(self._build_obs_runtime_context())

    def _get_obs(self, u: int) -> Dict[str, np.ndarray]:
        return self._get_obs_from_context(u, self._build_obs_runtime_context())

    def _build_global_state(self, *, gu_proxy_features: list[np.ndarray] | None = None) -> np.ndarray:
        cfg = self.cfg
        # Flatten global state for critic
        sat_pos, sat_vel = self._get_orbit_states()
        sat_idx = None
        if cfg.sat_state_max is not None and cfg.sat_state_max < cfg.num_sat:
            elev_matrix = self._get_elevation_matrix(sat_pos)
            scores = np.max(elev_matrix, axis=0)
            sat_idx = self._topk_descending_stable(scores, int(cfg.sat_state_max))
            sat_pos = sat_pos[sat_idx]
            sat_vel = sat_vel[sat_idx]
            sat_queue = self.sat_queue[sat_idx]
        else:
            sat_queue = self.sat_queue
        parts = [
            self.uav_pos.flatten() / cfg.map_size,
            self.uav_vel.flatten() / cfg.v_max,
            self.uav_queue / cfg.queue_max_uav,
            self.uav_energy / normalize_scale(cfg.uav_energy_init),
            self.gu_pos.flatten() / cfg.map_size,
            self.gu_queue / cfg.queue_max_gu,
            sat_pos.flatten() / (cfg.r_earth + cfg.sat_height),
            sat_vel.flatten() / (cfg.r_earth + cfg.sat_height),
            sat_queue / cfg.queue_max_sat,
            np.array([0.0], dtype=np.float32),  # Reserved: do not expose the artificial rollout horizon.
        ]
        parts.extend(self._gu_proxy_feature_arrays() if gu_proxy_features is None else gu_proxy_features)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _refresh_global_state_cache(self, *, gu_proxy_features: list[np.ndarray] | None = None) -> np.ndarray:
        self._cached_global_state = self._build_global_state(gu_proxy_features=gu_proxy_features)
        return self._cached_global_state

    def get_global_state(self) -> np.ndarray:
        if self._cached_global_state is None:
            return self._refresh_global_state_cache()
        return self._cached_global_state

    def render(self, mode="rgb_array"):
        if mode == "human":
            return None
        # Lazy import to avoid overhead in training
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.gu_pos[:, 0], self.gu_pos[:, 1], s=10, c="tab:blue", label="GU")
        ax.scatter(self.uav_pos[:, 0], self.uav_pos[:, 1], s=30, c="tab:red", label="UAV")
        ax.set_xlim(0, self.cfg.map_size)
        ax.set_ylim(0, self.cfg.map_size)
        ax.set_title(f"t={self.t}")
        ax.legend(loc="upper right")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # Matplotlib backends differ; prefer buffer_rgba and fall back to tostring_argb.
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgba = rgba.reshape((h, w, 4))
            buf = rgba[:, :, :3]
        except AttributeError:
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            argb = argb.reshape((h, w, 4))
            buf = argb[:, :, 1:4]
        plt.close(fig)
        return buf
