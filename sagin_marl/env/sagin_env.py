from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv

from .config import SaginConfig, ablation_flag
from .topology import thomas_cluster_process
from .orbit import WalkerDeltaOrbitModel
from . import channel


class SaginParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "sagin_parallel_v1"}

    def __init__(self, cfg: SaginConfig):
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
            (cfg.speed_of_light / (4.0 * math.pi * cfg.carrier_freq)) ** 2
            * cfg.uav_tx_gain
            * cfg.sat_rx_gain
        )

        self.agents = [f"uav_{i}" for i in range(cfg.num_uav)]
        self.possible_agents = list(self.agents)

        # Dimensions
        self.own_dim = 7
        self.user_dim = 5
        self.sat_dim = 12
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

    def _compute_centroid_stats(self) -> Tuple[float, float]:
        cfg = self.cfg
        centroid_reward = 0.0
        centroid_dist_mean = 0.0
        if cfg.num_gu > 0:
            q_weights = self.gu_queue / max(cfg.queue_max_gu, 1e-9)
            w_sum = float(np.sum(q_weights))
            if w_sum <= 1e-9:
                # Keep a dense navigation signal even when all queues are empty.
                weights = np.full((cfg.num_gu,), 1.0 / max(cfg.num_gu, 1), dtype=np.float32)
            else:
                weights = (q_weights / (w_sum + 1e-9)).astype(np.float32, copy=False)
            centroid = np.sum(self.gu_pos * weights[:, None], axis=0)
            dists = np.linalg.norm(self.uav_pos - centroid[None, :], axis=1)
            centroid_dist_mean = float(np.mean(dists)) if dists.size else 0.0
            scale = max(float(getattr(cfg, "centroid_dist_scale", 1.0) or 1.0), 1e-6)
            centroid_reward = float(np.mean(np.exp(-dists / scale))) if dists.size else 0.0
        return centroid_reward, centroid_dist_mean

    def _queue_arrival_scale(self, arrival_sum: float) -> float:
        cfg = self.cfg
        queue_norm_k = max(float(getattr(cfg, "queue_norm_K", 1.0) or 1.0), 1e-9)
        arrival_floor = float(getattr(cfg, "queue_norm_arrival_floor", 0.0) or 0.0)
        if arrival_floor <= 0.0:
            arrival_floor = (
                float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
                * float(cfg.num_gu)
                * float(cfg.tau0)
            )
        arrival_ref = max(float(arrival_sum), arrival_floor, 1e-9)
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
        if eta_start > 1e-9:
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
        self.arrival_ref_bits_per_step = max(
            float(self.effective_task_arrival_rate) * float(cfg.num_gu) * float(cfg.tau0),
            1e-9,
        )

    def _arrival_ref(self) -> float:
        return max(float(getattr(self, "arrival_ref_bits_per_step", 0.0) or 0.0), 1e-9)

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
                    shape=(cfg.users_obs_max,),
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
        return {
            agent: {
                "accel": np.zeros(2, dtype=np.float32),
                "bw_alloc": np.zeros(self.cfg.users_obs_max, dtype=np.float32),
                "sat_select_mask": np.zeros(self.cfg.sats_obs_max, dtype=np.float32),
            }
            for agent in self.agents
        }

    def _queue_init_arrival_ref(self) -> float:
        cfg = self.cfg
        return (
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
            * float(cfg.num_gu)
            * float(cfg.tau0)
        )

    def _resolve_queue_init_total(
        self,
        abs_attr: str,
        steps_attr: str,
        frac_attr: str,
        total_cap: float,
    ) -> float:
        cfg = self.cfg
        abs_value = getattr(cfg, abs_attr, None)
        if abs_value is not None:
            return min(max(float(abs_value), 0.0), total_cap)

        steps_value = getattr(cfg, steps_attr, None)
        if steps_value is not None:
            total = max(float(steps_value), 0.0) * self._queue_init_arrival_ref()
            return min(total, total_cap)

        frac_value = max(float(getattr(cfg, frac_attr, 0.0) or 0.0), 0.0)
        return min(float(np.clip(frac_value, 0.0, 1.0)) * total_cap, total_cap)

    def _init_queues(self) -> None:
        cfg = self.cfg
        gu_total = self._resolve_queue_init_total(
            "queue_init_gu_abs",
            "queue_init_gu_steps",
            "queue_init_frac",
            float(cfg.num_gu) * float(cfg.queue_max_gu),
        )
        uav_total = self._resolve_queue_init_total(
            "queue_init_uav_abs",
            "queue_init_uav_steps",
            "queue_init_uav_frac",
            float(cfg.num_uav) * float(cfg.queue_max_uav),
        )
        sat_total = self._resolve_queue_init_total(
            "queue_init_sat_abs",
            "queue_init_sat_steps",
            "queue_init_sat_frac",
            float(cfg.num_sat) * float(cfg.queue_max_sat),
        )

        if cfg.num_gu > 0 and gu_total > 0.0:
            self.gu_queue = np.full((cfg.num_gu,), gu_total / float(cfg.num_gu), dtype=np.float32)
        if cfg.num_uav > 0 and uav_total > 0.0:
            self.uav_queue = np.full((cfg.num_uav,), uav_total / float(cfg.num_uav), dtype=np.float32)
        if cfg.num_sat > 0 and sat_total > 0.0:
            self.sat_queue = np.full((cfg.num_sat,), sat_total / float(cfg.num_sat), dtype=np.float32)

    def _init_state(self) -> None:
        cfg = self.cfg
        self.t = 0
        self._episode_collision_count = 0
        self._episode_step_count = 0
        self.gu_pos, self.gu_cluster_centers, self.gu_cluster_counts = thomas_cluster_process(
            cfg.num_gu,
            cfg.map_size,
            num_clusters=max(1, cfg.num_gu // 5),
            cluster_std=80.0,
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
        self._init_queues()
        self.prev_queue_sum = 0.0
        self.prev_queue_sum_active = 0.0
        self.prev_queue_sum_gu = 0.0
        self.prev_queue_sum_uav = 0.0
        self.prev_queue_sum_sat = 0.0
        arrival_ref = (
            float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
            * float(cfg.num_gu)
            * float(cfg.tau0)
        )
        self.prev_arrival_sum = max(arrival_ref, 1e-9)
        self.prev_q_norm_active = 0.0
        self.prev_centroid_dist_mean = self._compute_centroid_stats()[1]
        self.prev_d_min = 0.0
        self.last_gu_outflow = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.last_gu_arrival = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.gu_drop = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.uav_drop = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.sat_drop = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_exec_bw_alloc = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        self.last_exec_sat_select_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        sat_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)
        self.last_exec_sat_indices = np.full((cfg.num_uav, sat_k), -1, dtype=np.int64)
        self.last_energy_cost = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_sat_processed = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_sat_incoming = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_bw_align = 0.0
        self.last_sat_score = 0.0
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
        self.last_visible_candidate_dist_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_elevation_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_se_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_candidate_queue_std = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_visible_stats: Dict[str, float | str] = {}
        self.last_arrival_rate = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
        self.last_filter_active_ratio = 0.0
        self.last_projected_delta_norm_mean = 0.0
        self.last_fallback_count = 0.0
        self.last_boundary_filter_count = 0.0
        self.last_pairwise_filter_count = 0.0
        self.last_pairwise_filter_active_ratio = 0.0
        self.last_pairwise_projected_delta_norm = 0.0
        self.last_pairwise_fallback_count = 0.0
        self.last_pairwise_candidate_infeasible_count = 0.0
        self.last_step_profile = self._empty_step_profile()
        self.last_reward_parts = {
            "service_ratio": 0.0,
            "drop_ratio": 0.0,
            "arrival_ref": self._arrival_ref(),
            "x_acc": 0.0,
            "x_rel": 0.0,
            "g_pre": 0.0,
            "d_pre": 0.0,
            "processed_ratio_eval": 0.0,
            "drop_ratio_eval": 0.0,
            "pre_backlog_steps_eval": 0.0,
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
            "term_service": 0.0,
            "term_drop": 0.0,
            "term_drop_gu": 0.0,
            "term_drop_uav": 0.0,
            "term_drop_sat": 0.0,
            "term_drop_step": 0.0,
            "term_queue": 0.0,
            "term_topk": 0.0,
            "term_assoc": 0.0,
            "term_q_delta": 0.0,
            "term_throughput_access": 0.0,
            "term_throughput_backhaul": 0.0,
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
        self._cached_bw_valid_mask = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        self._cached_sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, self.sat_dim), dtype=np.float32)
        self._cached_sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        self._cached_sat_valid_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        self._cached_global_state = None
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
        self._cached_assoc = assoc.copy()
        self._cached_candidates = self._build_candidate_users(assoc)
        self._cached_bw_valid_mask = self._build_bw_valid_mask(assoc, self._cached_candidates)
        dummy_actions = self._dummy_actions()
        _, self._cached_eta = self._compute_access_rates(
            assoc,
            self._cached_candidates,
            dummy_actions,
            record_exec=False,
        )
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        obs = {agent: self._get_obs(idx) for idx, agent in enumerate(self.agents)}
        infos = {
            agent: {
                "traffic_level": self.traffic_level,
                "traffic_level_ratio": self.traffic_level_ratio,
                "effective_task_arrival_rate": self.effective_task_arrival_rate,
                **self._agent_visible_info(idx),
            }
            for idx, agent in enumerate(self.agents)
        }
        self._refresh_global_state_cache()
        return obs, infos

    def step(self, actions: Dict[str, Dict]):
        cfg = self.cfg
        step_start = time.perf_counter()
        step_profile = self._empty_step_profile()
        self.global_step = int(getattr(self, "global_step", 0)) + 1
        self.prev_queue_sum = float(
            np.sum(self.gu_queue) + np.sum(self.uav_queue) + np.sum(self.sat_queue)
        )
        self.prev_queue_sum_active = float(np.sum(self.gu_queue) + np.sum(self.uav_queue))
        self.prev_queue_sum_gu = float(np.sum(self.gu_queue))
        self.prev_queue_sum_uav = float(np.sum(self.uav_queue))
        self.prev_queue_sum_sat = float(np.sum(self.sat_queue))
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
        visible = self._visible_sats_sorted(sat_pos)
        step_profile["orbit_visible_time_sec"] = time.perf_counter() - profile_start

        # Compute associations and rates
        profile_start = time.perf_counter()
        access_rates, _ = self._compute_access_rates(step_assoc, step_candidates, actions)
        step_profile["assoc_access_time_sec"] = time.perf_counter() - profile_start

        # Update GU queues
        profile_start = time.perf_counter()
        gu_outflow = self._update_gu_queues(access_rates, step_assoc)

        # Backhaul selection and rates
        sat_selection = self._select_satellites(sat_pos, sat_vel, actions, step_visible)
        self._update_energy(sat_selection)
        rate_matrix, sat_loads = self._compute_backhaul_rates(sat_pos, sat_vel, sat_selection)
        self.last_sat_selection = [list(sel) for sel in sat_selection]
        self.last_sat_connection_counts = sat_loads.astype(np.float32, copy=False)
        self._update_connected_sat_link_stats(sat_pos, sat_selection)

        # Update UAV queues and satellite queues
        outflow_matrix = self._update_uav_queues(gu_outflow, rate_matrix)
        self._update_sat_queues(outflow_matrix)
        step_profile["backhaul_queue_time_sec"] = time.perf_counter() - profile_start

        # Cache for obs
        profile_start = time.perf_counter()
        assoc_next = self._associate_users()
        candidate_lists_next = self._build_candidate_users(assoc_next)
        self._cached_assoc = assoc_next.copy()
        self._cached_candidates = candidate_lists_next
        self._cached_bw_valid_mask = self._build_bw_valid_mask(assoc_next, candidate_lists_next)
        _, eta_next = self._compute_access_rates(
            assoc_next,
            candidate_lists_next,
            self._dummy_actions(),
            record_exec=False,
        )
        self._cached_eta = eta_next
        visible_next = self._visible_sats_sorted(sat_pos)
        self._cache_sat_obs(sat_pos, sat_vel, visible_next)
        step_profile["obs_time_sec"] = time.perf_counter() - profile_start

        # Rewards and done
        profile_start = time.perf_counter()
        reward = self._compute_reward()
        collision = bool(getattr(self, "last_reward_parts", {}).get("collision_event", 0.0) > 0.5)
        if not collision:
            collision = self._check_collision()
        self._episode_step_count = int(getattr(self, "_episode_step_count", 0)) + 1
        if collision:
            self._episode_collision_count = int(getattr(self, "_episode_collision_count", 0)) + 1
        energy_depleted = cfg.energy_enabled and np.any(self.uav_energy <= 0.0)
        terminated = collision or energy_depleted
        truncated = self.t >= (cfg.T_steps - 1)

        self.t += 1
        step_profile["reward_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        obs = {agent: self._get_obs(idx) for idx, agent in enumerate(self.agents)}
        step_profile["obs_time_sec"] += time.perf_counter() - profile_start
        profile_start = time.perf_counter()
        self._refresh_global_state_cache()
        step_profile["state_time_sec"] = time.perf_counter() - profile_start
        step_profile["step_total_time_sec"] = time.perf_counter() - step_start
        self.last_step_profile = step_profile
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: self._agent_visible_info(idx) for idx, agent in enumerate(self.agents)}
        return obs, rewards, terminations, truncations, infos

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
        accel_arr = np.asarray(accel, dtype=np.float32)
        vel_next = np.clip(self.uav_vel + accel_arr * cfg.tau0, -cfg.v_max, cfg.v_max)
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
        accel_candidate[i] = np.clip(accel_candidate[i] + required_push * direction, -cfg.a_max, cfg.a_max)
        accel_candidate[j] = np.clip(accel_candidate[j] - required_push * direction, -cfg.a_max, cfg.a_max)
        accel_candidate, _ = self._apply_boundary_hard_filter(accel_candidate, indices=[i, j])
        pos_candidate, _ = self._predict_next_from_accel(accel_candidate)
        dist_candidate = float(np.linalg.norm(pos_candidate[i] - pos_candidate[j]))
        if dist_candidate >= d_hard:
            return accel_candidate, True, False, False

        accel_fallback = np.asarray(accel, dtype=np.float32).copy()
        accel_fallback[i] = np.clip(direction * cfg.a_max, -cfg.a_max, cfg.a_max)
        accel_fallback[j] = np.clip(-direction * cfg.a_max, -cfg.a_max, cfg.a_max)
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
        accel_candidate[i] = np.clip(accel_candidate[i] + required_push * direction, -cfg.a_max, cfg.a_max)
        accel_candidate[j] = np.clip(accel_candidate[j] - required_push * direction, -cfg.a_max, cfg.a_max)
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
        accel_fallback[i] = np.clip(direction * cfg.a_max, -cfg.a_max, cfg.a_max)
        accel_fallback[j] = np.clip(-direction * cfg.a_max, -cfg.a_max, cfg.a_max)
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

    def _apply_uav_dynamics(self, actions: Dict[str, Dict]) -> None:
        cfg = self.cfg
        accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        policy_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
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
            a = np.array(actions[agent]["accel"], dtype=np.float32)
            a = np.clip(a, -1.0, 1.0) * cfg.a_max
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
                    a_rep = np.clip(a_rep, -cfg.a_max, cfg.a_max)
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
            a = np.clip(a, -cfg.a_max, cfg.a_max)
            accel[i] = a
        accel_before_hard_filter = accel.copy()
        accel, boundary_stats = self._apply_boundary_hard_filter(accel)
        accel, pairwise_stats = self._apply_pairwise_hard_filter(accel)
        hard_delta_norm = (
            np.linalg.norm(accel - accel_before_hard_filter, axis=1) if cfg.num_uav > 0 else np.zeros((0,), dtype=np.float32)
        )
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
        self.uav_vel = np.clip(self.uav_vel + accel * cfg.tau0, -cfg.v_max, cfg.v_max)
        self.uav_pos = self.uav_pos + self.uav_vel * cfg.tau0
        if cfg.boundary_mode == "reflect":
            for i in range(cfg.num_uav):
                for axis in range(2):
                    if self.uav_pos[i, axis] < 0.0:
                        self.uav_pos[i, axis] = -self.uav_pos[i, axis]
                        self.uav_vel[i, axis] = -self.uav_vel[i, axis]
                    elif self.uav_pos[i, axis] > cfg.map_size:
                        self.uav_pos[i, axis] = 2 * cfg.map_size - self.uav_pos[i, axis]
                        self.uav_vel[i, axis] = -self.uav_vel[i, axis]
        self.uav_pos = np.clip(self.uav_pos, 0.0, cfg.map_size)
        self._refresh_uav_cache()

    def _invalidate_step_caches(self) -> None:
        self._cached_orbit_t = None
        self._cached_orbit_pos = None
        self._cached_orbit_vel = None
        self._cached_elevation_t = None
        self._cached_elevation_matrix = None
        self._cached_uav_ecef = None
        self._cached_uav_vel_ecef = None
        self._cached_uav_neighbor_t = None
        self._cached_uav_neighbor_order = None

    def _get_orbit_states(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cached_orbit_t != self.t or self._cached_orbit_pos is None:
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
        d2d = np.linalg.norm(diff, axis=2)
        d3d = np.sqrt(d2d * d2d + cfg.uav_height ** 2)
        phi = np.arcsin(cfg.uav_height / (d3d + 1e-9))
        pl = channel.pathloss_db(d3d, phi, cfg)
        best = np.argmin(pl, axis=1)
        best_pl = pl[np.arange(K), best]
        assoc = np.where(best_pl <= cfg.pl_threshold_db, best, -1).astype(np.int32)
        return assoc

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

    def _compute_access_rates(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        actions: Dict[str, Dict],
        record_exec: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        rates = np.zeros((cfg.num_gu,), dtype=np.float32)
        eta = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        bw_align_sum = 0.0
        bw_align_count = 0
        exec_bw = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)

        for u in range(cfg.num_uav):
            cand = candidates[u]
            if not cand:
                continue
            cand_idx = np.asarray(cand, dtype=np.int32)
            assoc_mask = assoc[cand_idx] == u
            if cfg.enable_bw_action:
                action_data = actions[self.agents[u]]
                bw_raw = action_data.get("bw_alloc", action_data.get("bw_logits", np.zeros((cfg.users_obs_max,), dtype=np.float32)))
                bw_raw = np.asarray(bw_raw, dtype=np.float32)[: cfg.users_obs_max]
                bw_slot = np.clip(bw_raw[: len(cand)], 0.0, None)
                betas = bw_slot * assoc_mask.astype(np.float32, copy=False)
                denom = float(np.sum(betas))
                if denom > 1e-9:
                    betas = betas / denom
                else:
                    betas = np.zeros((len(cand),), dtype=np.float32)
                    if np.any(assoc_mask):
                        betas[assoc_mask] = 1.0 / float(np.sum(assoc_mask))
            else:
                if np.any(assoc_mask):
                    betas = np.zeros((len(cand),), dtype=np.float32)
                    betas[assoc_mask] = 1.0 / float(np.sum(assoc_mask))
                else:
                    betas = np.zeros((len(cand),), dtype=np.float32)
            exec_bw[u, : len(cand)] = betas.astype(np.float32, copy=False)

            se_values = None
            # Preserve exact behavior when interference/fading are enabled (RNG order matters)
            if cfg.interference_enabled or cfg.fading_enabled:
                se_values = np.zeros((len(cand),), dtype=np.float32)
                for i, k in enumerate(cand):
                    d2d = np.linalg.norm(self.uav_pos[u] - self.gu_pos[k])
                    d3d = math.sqrt(d2d**2 + cfg.uav_height**2)
                    phi = math.asin(cfg.uav_height / (d3d + 1e-9))
                    pl = channel.pathloss_db(np.array([d3d]), np.array([phi]), cfg)[0]
                    gain = 10 ** (-pl / 10.0)
                    if cfg.fading_enabled:
                        gain *= channel.rician_power_gain(cfg.rician_K, size=1, rng=self.rng)[0]

                    interference = 0.0
                    if cfg.interference_enabled:
                        for j in range(cfg.num_gu):
                            if assoc[j] >= 0 and assoc[j] != u:
                                d2d_j = np.linalg.norm(self.uav_pos[u] - self.gu_pos[j])
                                d3d_j = math.sqrt(d2d_j**2 + cfg.uav_height**2)
                                phi_j = math.asin(cfg.uav_height / (d3d_j + 1e-9))
                                pl_j = channel.pathloss_db(np.array([d3d_j]), np.array([phi_j]), cfg)[0]
                                gain_j = 10 ** (-pl_j / 10.0)
                                if cfg.fading_enabled:
                                    gain_j *= channel.rician_power_gain(cfg.rician_K, size=1, rng=self.rng)[0]
                                interference += cfg.gu_tx_power * gain_j

                    snr = channel.snr_linear(cfg.gu_tx_power, gain, cfg.noise_density, cfg.b_acc, interference)
                    se = channel.spectral_efficiency(snr)
                    if assoc_mask[i]:
                        rate = betas[i] * cfg.b_acc * se
                        rates[k] = rate
                    eta[u, i] = se
                    se_values[i] = se
            else:
                gu_pos = self.gu_pos[cand_idx]
                uav_pos = self.uav_pos[u]
                d2d = np.linalg.norm(gu_pos - uav_pos, axis=1)
                d3d = np.sqrt(d2d * d2d + self._uav_height_sq)
                phi = np.arcsin(cfg.uav_height / (d3d + 1e-9))
                pl = channel.pathloss_db(d3d, phi, cfg)
                gain = 10 ** (-pl / 10.0)

                snr = channel.snr_linear(cfg.gu_tx_power, gain, cfg.noise_density, cfg.b_acc)
                se = channel.spectral_efficiency(snr)
                rate = betas * cfg.b_acc * se
                if np.any(assoc_mask):
                    rates[cand_idx[assoc_mask]] = rate[assoc_mask].astype(np.float32)
                eta[u, : len(cand)] = se.astype(np.float32)
                se_values = se.astype(np.float32)

            if cfg.enable_bw_action and se_values is not None and len(cand) > 0:
                if np.any(assoc_mask):
                    active_idx = cand_idx[assoc_mask]
                    q_norm = self.gu_queue[active_idx] / max(cfg.queue_max_gu, 1e-9)
                    assoc_bonus = 0.2
                    prev_mask = (self.prev_association[active_idx] == u).astype(np.float32)
                    target_weights = q_norm * (0.5 + se_values[assoc_mask]) * (1.0 + assoc_bonus * prev_mask)
                    denom = float(np.sum(target_weights))
                    if denom > 0:
                        target = target_weights / denom
                        betas_sel = betas[assoc_mask]
                        l1 = float(np.sum(np.abs(betas_sel - target)))
                        align = 1.0 - 0.5 * l1
                        bw_align_sum += align
                        bw_align_count += 1

        if record_exec:
            self.last_exec_bw_alloc = exec_bw
            self.last_bw_align = bw_align_sum / max(1, bw_align_count)
        return rates, eta

    def _update_gu_queues(self, access_rates: np.ndarray, assoc: np.ndarray) -> np.ndarray:
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
        self.last_arrival_rate = arrival_rate
        if cfg.task_arrival_poisson:
            arrival = self.rng.poisson(arrival_rate, size=cfg.num_gu).astype(np.float32)
        else:
            arrival = np.full((cfg.num_gu,), arrival_rate, dtype=np.float32)
        self.last_gu_arrival = arrival.astype(np.float32)
        q_before = self.gu_queue + arrival
        outflow = np.minimum(q_before, access_rates * cfg.tau0)
        q_after = q_before - outflow
        self.last_gu_outflow = outflow.astype(np.float32)
        self.gu_drop = np.maximum(q_after - cfg.queue_max_gu, 0.0).astype(np.float32)
        q_after = np.minimum(q_after, cfg.queue_max_gu)
        self.gu_queue = q_after.astype(np.float32)
        gu_outflow = outflow.astype(np.float32)

        self.last_association = assoc.copy()
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
        sat_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)
        exec_sat_indices = np.full((cfg.num_uav, sat_k), -1, dtype=np.int64)
        for u in range(cfg.num_uav):
            vis = visible[u]
            if not vis:
                continue
            if cfg.fixed_satellite_strategy:
                # pick nearest visible
                dists = [np.linalg.norm(sat_pos[l] - self._uav_ecef(u)) for l in vis]
                sel = vis[int(np.argmin(dists))]
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
                    for i, l in enumerate(cand):
                        nu = self._doppler(u, l, sat_pos, sat_vel)
                        if abs(nu) > cfg.nu_max:
                            valid_flags[i] = False

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

    def _compute_backhaul_rates(self, sat_pos: np.ndarray, sat_vel: np.ndarray, selections: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        U = cfg.num_uav
        L = cfg.num_sat
        rate_matrix = np.zeros((U, L), dtype=np.float32)
        sat_score_sum = 0.0
        sat_score_count = 0

        # count connections per sat
        counts = np.zeros((L,), dtype=np.int32)
        for u in range(U):
            for l in selections[u]:
                counts[l] += 1

        for u in range(U):
            for l in selections[u]:
                if counts[l] == 0:
                    continue
                b_ul = cfg.b_sat_total / counts[l]
                d = np.linalg.norm(sat_pos[l] - self._uav_ecef(u))
                gain = self._backhaul_gain_const / (d ** 2)
                if cfg.atm_loss_enabled:
                    theta = self._elevation_angle(u, l, sat_pos)
                    atm_loss = channel.atmospheric_loss_db(theta, cfg.atm_loss_db)
                    gain *= 10 ** (-atm_loss / 10.0)

                snr = channel.snr_linear(cfg.uav_tx_power, gain, cfg.noise_density, b_ul)
                nu = 0.0
                if cfg.doppler_enabled or cfg.doppler_atten_enabled:
                    nu = self._doppler(u, l, sat_pos, sat_vel)
                if cfg.doppler_atten_enabled:
                    chi = channel.doppler_attenuation(np.array([nu]), cfg.subcarrier_spacing)[0]
                    snr = snr * chi

                se = channel.spectral_efficiency(snr)
                rate = b_ul * se

                if cfg.doppler_enabled and abs(nu) > cfg.nu_max:
                    rate = 0.0
                rate_matrix[u, l] = rate
                sat_score = float(se) - 0.5 * float(self.sat_queue[l] / max(cfg.queue_max_sat, 1e-9))
                sat_score_sum += sat_score
                sat_score_count += 1

        self.last_sat_score = sat_score_sum / max(1, sat_score_count)
        return rate_matrix, counts

    def _update_uav_queues(self, gu_outflow: np.ndarray, rate_matrix: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        outflow_matrix = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        valid = self.last_association >= 0
        if np.any(valid):
            inflow = np.bincount(
                self.last_association[valid],
                weights=gu_outflow[valid],
                minlength=cfg.num_uav,
            ).astype(np.float32)
        else:
            inflow = np.zeros((cfg.num_uav,), dtype=np.float32)
        q_before = self.uav_queue + inflow
        total_rate = np.sum(rate_matrix, axis=1).astype(np.float32)
        outflow = np.minimum(q_before, total_rate * cfg.tau0)
        q_after = q_before - outflow
        self.uav_drop = np.maximum(q_after - cfg.queue_max_uav, 0.0).astype(np.float32)
        q_after = np.minimum(q_after, cfg.queue_max_uav)
        self.uav_queue = q_after.astype(np.float32)
        mask = total_rate > 0
        if np.any(mask):
            outflow_matrix[mask] = (rate_matrix[mask] / total_rate[mask, None]) * outflow[mask, None]
        return outflow_matrix

    def _update_sat_queues(self, outflow_matrix: np.ndarray) -> None:
        cfg = self.cfg
        incoming = np.sum(outflow_matrix, axis=0)
        compute_rate = cfg.sat_cpu_freq / cfg.task_cycles_per_bit
        before = self.sat_queue.copy()
        q_before = before + incoming
        processed = np.minimum(q_before, compute_rate * cfg.tau0)
        q_after = q_before - processed
        self.sat_drop = np.maximum(q_after - cfg.queue_max_sat, 0.0).astype(np.float32)
        q_after = np.minimum(q_after, cfg.queue_max_sat)
        self.last_sat_processed = processed.astype(np.float32)
        self.last_sat_incoming = incoming.astype(np.float32)
        self.sat_queue = q_after.astype(np.float32)

    def _update_energy(self, selections: List[List[int]]) -> None:
        cfg = self.cfg
        if not cfg.energy_enabled:
            self.last_energy_cost = np.zeros((cfg.num_uav,), dtype=np.float32)
            return
        speeds = np.linalg.norm(self.uav_vel, axis=1)
        p_fly = self._fly_power(speeds)
        link_counts = np.array([len(sats) for sats in selections], dtype=np.float32)
        p_comm = cfg.p_comm_link * link_counts
        self.last_energy_cost = p_fly + p_comm
        self.uav_energy = self.uav_energy - self.last_energy_cost * cfg.tau0
        self.uav_energy = np.maximum(self.uav_energy, 0.0)

    def _update_connected_sat_link_stats(self, sat_pos: np.ndarray, sat_selection: List[List[int]]) -> None:
        dist_values: List[float] = []
        elevation_deg_values: List[float] = []
        for u, selected in enumerate(sat_selection):
            for l in selected:
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
        if reward_mode not in {"dense", "throughput_only", "controllable_flow"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        if cfg.energy_enabled:
            p_max = self._energy_scale()
            r_energy = -np.mean(self.last_energy_cost / (p_max + 1e-9))
        else:
            r_energy = 0.0

        q_gu = float(np.sum(self.gu_queue))
        q_uav = float(np.sum(self.uav_queue))
        q_sat = float(np.sum(self.sat_queue))
        q_total = q_gu + q_uav + q_sat
        q_max_total = float(
            cfg.num_gu * cfg.queue_max_gu
            + cfg.num_uav * cfg.queue_max_uav
            + cfg.num_sat * cfg.queue_max_sat
        )
        q_max_total = max(q_max_total, 1e-9)
        q_gu_max = max(float(cfg.num_gu * cfg.queue_max_gu), 1e-9)
        q_uav_max = max(float(cfg.num_uav * cfg.queue_max_uav), 1e-9)
        q_sat_max = max(float(cfg.num_sat * cfg.queue_max_sat), 1e-9)
        q_gu_norm = q_gu / q_gu_max
        q_uav_norm = q_uav / q_uav_max
        q_sat_norm = q_sat / q_sat_max
        q_total_active = q_gu + q_uav

        arrival_sum = float(np.sum(self.last_gu_arrival))
        outflow_sum = float(np.sum(self.last_gu_outflow))
        backhaul_sum = float(np.sum(getattr(self, "last_sat_incoming", 0.0)))
        sat_processed_sum = float(np.sum(getattr(self, "last_sat_processed", 0.0)))
        gu_drop_sum = float(np.sum(self.gu_drop))
        uav_drop_sum = float(np.sum(self.uav_drop))
        sat_drop_sum = float(np.sum(getattr(self, "sat_drop", 0.0)))
        drop_sum_active = gu_drop_sum + uav_drop_sum
        drop_sum = drop_sum_active + sat_drop_sum
        service_ratio = outflow_sum / (arrival_sum + 1e-9)
        drop_ratio = drop_sum / (arrival_sum + 1e-9)
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
        service_norm = outflow_sum / arrival_scale
        drop_norm = drop_sum / arrival_scale
        gu_drop_norm = gu_drop_sum / arrival_scale
        uav_drop_norm = uav_drop_sum / arrival_scale
        sat_drop_norm = sat_drop_sum / arrival_scale
        drop_event = 1.0 if drop_sum > 1e-9 else 0.0
        throughput_access_norm = outflow_sum / arrival_scale
        throughput_backhaul_norm = backhaul_sum / arrival_scale
        sat_processed_norm = sat_processed_sum / arrival_scale
        outflow_arrival_ratio_step = outflow_sum / max(arrival_sum, 1e-9)
        sat_incoming_arrival_ratio_step = backhaul_sum / max(arrival_sum, 1e-9)
        sat_processed_arrival_ratio_step = sat_processed_sum / max(arrival_sum, 1e-9)
        sat_processed_incoming_ratio_step = sat_processed_sum / max(backhaul_sum, 1e-9)
        gu_drop_ratio_step = gu_drop_sum / max(arrival_sum, 1e-9)
        uav_drop_ratio_step = uav_drop_sum / max(arrival_sum, 1e-9)
        sat_drop_ratio_step = sat_drop_sum / max(arrival_sum, 1e-9)
        b_pre_t = float(getattr(self, "prev_queue_sum_gu", q_gu) + getattr(self, "prev_queue_sum_uav", q_uav))
        b_pre_tp1 = q_total_active
        x_acc = outflow_sum / arrival_ref
        x_rel = backhaul_sum / arrival_ref
        g_pre = (b_pre_tp1 - b_pre_t) / arrival_ref
        d_pre = (gu_drop_sum + uav_drop_sum) / arrival_ref
        processed_ratio_eval = sat_processed_sum / arrival_ref
        drop_ratio_eval = drop_sum / arrival_ref
        pre_backlog_steps_eval = q_total_active / arrival_ref
        D_sys_report = q_total / max(sat_processed_sum, 1e-9)

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
            if w_sum < 1e-9:
                queue_total_norm = q_total / arrival_scale if use_arrival_norm_queue else q_total / q_max_total
                queue_term = _queue_smooth(queue_total_norm)
            else:
                queue_term = (w_gu * queue_gu + w_uav * queue_uav + w_sat * queue_sat) / w_sum
            if queue_delta_mode == "weighted" and w_sum >= 1e-9:
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
                queue_delta = (prev_sum - cur_sum) / max(q_delta_den, 1e-9)
                if not use_arrival_norm_queue:
                    queue_delta = float(np.clip(queue_delta, -1.0, 1.0))
                else:
                    queue_delta = float(queue_delta)
            queue_weight = float(cfg.omega_q)

        if cfg.a_max > 0:
            accel_norm2 = float(np.mean(np.sum(self.last_exec_accel**2, axis=1))) / (cfg.a_max**2 + 1e-9)
        else:
            accel_norm2 = 0.0
        intervention_delta = np.asarray(self.last_exec_accel - self.last_policy_accel, dtype=np.float32)
        intervention_norms = np.linalg.norm(intervention_delta, axis=1) if intervention_delta.size else np.zeros((0,), dtype=np.float32)
        intervention_norms_uav = (
            intervention_norms / (cfg.a_max + 1e-9) if intervention_norms.size and cfg.a_max > 0 else np.zeros((0,), dtype=np.float32)
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
            getattr(cfg, "danger_imitation_trigger_mode", "risk_or_intervention") or "risk_or_intervention"
        ).strip().lower()
        if danger_trigger_mode not in {"risk_or_intervention", "intervention_any", "intervention_threshold"}:
            danger_trigger_mode = "risk_or_intervention"
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
            term_queue = -float(getattr(cfg, "reward_w_pre_growth", 0.2) or 0.0) * max(g_pre, 0.0)
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
            )
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
            term_q_delta = 0.0
            term_centroid = 0.0
            term_accel = 0.0
            term_close_risk = 0.0
            term_energy = 0.0
            raw_reward = term_throughput_access + term_throughput_backhaul + term_queue_gu_arrival
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

        collision_now = self._check_collision()
        if reward_mode in {"throughput_only", "controllable_flow"}:
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
            "x_acc": x_acc,
            "x_rel": x_rel,
            "g_pre": g_pre,
            "d_pre": d_pre,
            "processed_ratio_eval": processed_ratio_eval,
            "drop_ratio_eval": drop_ratio_eval,
            "pre_backlog_steps_eval": pre_backlog_steps_eval,
            "D_sys_report": D_sys_report,
            "drop_sum": drop_sum,
            "drop_sum_active": drop_sum_active,
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
            "term_service": term_service,
            "term_drop": term_drop,
            "term_drop_gu": term_drop_gu,
            "term_drop_uav": term_drop_uav,
            "term_drop_sat": term_drop_sat,
            "term_drop_step": term_drop_step,
            "term_queue": term_queue,
            "term_topk": term_topk,
            "term_assoc": term_assoc,
            "term_q_delta": term_q_delta,
            "term_throughput_access": term_throughput_access,
            "term_throughput_backhaul": term_throughput_backhaul,
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
        }
        return float(reward)

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
        lon = lon0 + x / (cfg.r_earth * math.cos(lat0) + 1e-9)
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
        d = np.linalg.norm(sat_pos[l] - q)
        arg = (self._sat_orbit_radius_sq - self._uav_orbit_radius_sq - d ** 2) / (
            2.0 * self._uav_orbit_radius * d + 1e-9
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
        dist = np.linalg.norm(rel, axis=2)
        arg = (self._sat_orbit_radius_sq - self._uav_orbit_radius_sq - dist * dist) / (
            2.0 * self._uav_orbit_radius * dist + 1e-9
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
        if v_max - v_min <= 1e-9:
            return np.zeros_like(vals, dtype=np.float32)
        return ((vals - v_min) / (v_max - v_min)).astype(np.float32, copy=False)

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
        dist = np.linalg.norm(rel_pos, axis=1) + 1e-9
        gain = self._backhaul_gain_const / np.maximum(dist * dist, 1e-9)
        if cfg.atm_loss_enabled:
            atm_loss = channel.atmospheric_loss_db(elev, cfg.atm_loss_db)
            gain = gain * (10 ** (-atm_loss / 10.0))
        snr = channel.snr_linear(cfg.uav_tx_power, gain, cfg.noise_density, cfg.b_sat_total)
        se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
        queue_norm = (self.sat_queue[sat_idx] / max(cfg.queue_max_sat, 1e-9)).astype(np.float32, copy=False)

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
        return np.lexsort((-elev.astype(np.float64), -rank_value.astype(np.float64)))

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

    def _visible_sats_sorted(self, sat_pos: np.ndarray) -> List[List[int]]:
        cfg = self.cfg
        elev_matrix = self._get_elevation_matrix(sat_pos)
        visible: List[List[int]] = [[] for _ in range(cfg.num_uav)]
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
        min_keep = cfg.visible_sats_min
        for u in range(cfg.num_uav):
            elev_u = elev_matrix[u]
            above = np.nonzero(elev_u >= cfg.theta_min_rad)[0].astype(np.int32, copy=False)
            raw_counts[u] = int(above.size)
            above_data = self._sat_candidate_rank_data(u, above, sat_pos, elev_values=elev_u[above])
            above_order = self._sat_candidate_order(above_data)
            above_sorted = above[above_order]
            raw_candidates[u] = above_sorted.tolist()
            selected = above_sorted
            if min_keep is not None and above.size < min_keep:
                needed = min_keep - int(above.size)
                extra = np.nonzero(elev_u < cfg.theta_min_rad)[0].astype(np.int32, copy=False)
                extra_data = self._sat_candidate_rank_data(u, extra, sat_pos, elev_values=elev_u[extra])
                extra_order = self._sat_candidate_order(extra_data)
                extra = extra[extra_order][:needed]
                if extra.size > 0:
                    selected = np.concatenate((selected, extra), axis=0)

            kept = selected[:max_keep]
            kept_counts[u] = int(kept.size)
            visible[u] = kept.tolist()
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
            "candidate_mode": str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower(),
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
        denom = np.linalg.norm(r_vec) + 1e-9
        proj = float(np.dot(v_rel, r_vec) / denom)
        return (cfg.carrier_freq / cfg.speed_of_light) * proj

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
        for u in range(cfg.num_uav):
            current_sats = set(self.last_sat_selection[u]) if u < len(self.last_sat_selection) else set()
            for i, l in enumerate(visible[u][: cfg.sats_obs_max]):
                rel_pos = sat_pos[l] - self._uav_ecef(u)
                rel_vel = sat_vel[l] - self._uav_vel_ecef(u)
                nu = self._doppler(u, l, sat_pos, sat_vel)
                d = np.linalg.norm(rel_pos) + 1e-9
                gain = (cfg.speed_of_light / (4.0 * math.pi * cfg.carrier_freq * d)) ** 2
                if cfg.atm_loss_enabled:
                    theta = self._elevation_angle(u, l, sat_pos)
                    atm_loss = channel.atmospheric_loss_db(theta, cfg.atm_loss_db)
                    gain *= 10 ** (-atm_loss / 10.0)
                gain *= cfg.uav_tx_gain * cfg.sat_rx_gain
                snr = channel.snr_linear(cfg.uav_tx_power, gain, cfg.noise_density, cfg.b_sat_total)
                if cfg.doppler_observed and cfg.doppler_atten_enabled:
                    chi = channel.doppler_attenuation(np.array([nu]), cfg.subcarrier_spacing)[0]
                    snr = snr * chi
                sat_obs[u, i, 0:3] = rel_pos / (cfg.r_earth + cfg.sat_height)
                sat_obs[u, i, 3:6] = rel_vel / (cfg.r_earth + cfg.sat_height)
                sat_obs[u, i, 6] = nu / max(cfg.nu_max, 1.0)
                sat_obs[u, i, 7] = channel.spectral_efficiency(snr)
                sat_obs[u, i, 8] = self.sat_queue[l] / cfg.queue_max_sat
                load_count = float(self.last_sat_connection_counts[l]) if l < self.last_sat_connection_counts.size else 0.0
                projected_count = max(load_count + (0.0 if l in current_sats else 1.0), 1.0)
                sat_obs[u, i, 9] = load_count / max(float(cfg.num_uav), 1.0)
                sat_obs[u, i, 10] = 1.0 / projected_count
                sat_obs[u, i, 11] = 1.0 if l in current_sats else 0.0
                sat_mask[u, i] = 1.0
                doppler_ok = (not cfg.doppler_enabled) or (abs(nu) <= cfg.nu_max)
                sat_valid_mask[u, i] = 1.0 if doppler_ok else 0.0
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

    def _get_obs(self, u: int) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        # own features
        own = np.array(
            [
                self.uav_pos[u, 0] / cfg.map_size,
                self.uav_pos[u, 1] / cfg.map_size,
                self.uav_vel[u, 0] / cfg.v_max,
                self.uav_vel[u, 1] / cfg.v_max,
                self.uav_energy[u] / max(cfg.uav_energy_init, 1e-9),
                self.uav_queue[u] / cfg.queue_max_uav,
                self.t / max(cfg.T_steps, 1),
            ],
            dtype=np.float32,
        )

        # users
        users = np.zeros((cfg.users_obs_max, self.user_dim), dtype=np.float32)
        users_mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        bw_valid_mask = self._cached_bw_valid_mask[u].copy()
        cand = self._cached_candidates[u] if self._cached_candidates else []
        for i, k in enumerate(cand[: cfg.users_obs_max]):
            rel = self.gu_pos[k] - self.uav_pos[u]
            users[i, 0:2] = rel / cfg.map_size
            users[i, 2] = self.gu_queue[k] / cfg.queue_max_gu
            users[i, 3] = self._cached_eta[u, i]
            users[i, 4] = 1.0 if self.prev_association[k] == u else 0.0
            users_mask[i] = 1.0

        # satellites (cached per step)
        sats = self._cached_sat_obs[u].copy()
        sats_mask = self._cached_sat_mask[u].copy()
        sat_valid_mask = self._cached_sat_valid_mask[u].copy()

        # neighbors
        nbrs = np.zeros((cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32)
        nbrs_mask = np.zeros((cfg.nbrs_obs_max,), dtype=np.float32)
        self._ensure_neighbor_cache()
        order = self._cached_uav_neighbor_order[u]
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
        if bool(getattr(cfg, "danger_nbr_enabled", False)):
            obs["danger_nbr"] = self._danger_neighbor_obs(u)
        return obs

    def _build_global_state(self) -> np.ndarray:
        cfg = self.cfg
        # Flatten global state for critic
        sat_pos, sat_vel = self._get_orbit_states()
        sat_idx = None
        if cfg.sat_state_max is not None and cfg.sat_state_max < cfg.num_sat:
            elev_matrix = self._get_elevation_matrix(sat_pos)
            scores = np.max(elev_matrix, axis=0)
            sat_idx = np.argsort(-scores, kind="stable")[: cfg.sat_state_max]
            sat_pos = sat_pos[sat_idx]
            sat_vel = sat_vel[sat_idx]
            sat_queue = self.sat_queue[sat_idx]
        else:
            sat_queue = self.sat_queue
        parts = [
            self.uav_pos.flatten() / cfg.map_size,
            self.uav_vel.flatten() / cfg.v_max,
            self.uav_queue / cfg.queue_max_uav,
            self.uav_energy / max(cfg.uav_energy_init, 1e-9),
            self.gu_pos.flatten() / cfg.map_size,
            self.gu_queue / cfg.queue_max_gu,
            sat_pos.flatten() / (cfg.r_earth + cfg.sat_height),
            sat_vel.flatten() / (cfg.r_earth + cfg.sat_height),
            sat_queue / cfg.queue_max_sat,
            np.array([self.t / max(cfg.T_steps, 1)], dtype=np.float32),
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _refresh_global_state_cache(self) -> np.ndarray:
        self._cached_global_state = self._build_global_state()
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
