from __future__ import annotations

import math
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
        self.sat_dim = 9
        self.nbr_dim = 4

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

    def _sample_uav_positions(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.num_uav <= 0:
            return np.zeros((0, 2), dtype=np.float32)
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
        self._obs_space = gym.spaces.Dict(
            {
                "own": gym.spaces.Box(-np.inf, np.inf, shape=(self.own_dim,), dtype=np.float32),
                "users": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.users_obs_max, self.user_dim), dtype=np.float32),
                "users_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.users_obs_max,), dtype=np.float32),
                "sats": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.sats_obs_max, self.sat_dim), dtype=np.float32),
                "sats_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.sats_obs_max,), dtype=np.float32),
                "nbrs": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32),
                "nbrs_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.nbrs_obs_max,), dtype=np.float32),
            }
        )

        self._act_space = gym.spaces.Dict(
            {
                "accel": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "bw_logits": gym.spaces.Box(
                    -cfg.bw_logit_scale, cfg.bw_logit_scale, shape=(cfg.users_obs_max,), dtype=np.float32
                ),
                "sat_logits": gym.spaces.Box(
                    -cfg.sat_logit_scale, cfg.sat_logit_scale, shape=(cfg.sats_obs_max,), dtype=np.float32
                ),
            }
        )

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _init_state(self) -> None:
        cfg = self.cfg
        self.t = 0
        self._episode_collision_count = 0
        self._episode_step_count = 0
        self.gu_pos = thomas_cluster_process(
            cfg.num_gu,
            cfg.map_size,
            num_clusters=max(1, cfg.num_gu // 5),
            cluster_std=80.0,
            rng=self.rng,
        )
        self.uav_pos = self._sample_uav_positions()
        self.uav_vel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        self.uav_energy = np.full((cfg.num_uav,), cfg.uav_energy_init, dtype=np.float32)
        self.last_exec_accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)

        self.gu_queue = np.zeros((cfg.num_gu,), dtype=np.float32)
        self.uav_queue = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.sat_queue = np.zeros((cfg.num_sat,), dtype=np.float32)
        init_gu_frac = float(getattr(cfg, "queue_init_frac", 0.0) or 0.0)
        init_uav_frac = float(getattr(cfg, "queue_init_uav_frac", 0.0) or 0.0)
        init_sat_frac = float(getattr(cfg, "queue_init_sat_frac", 0.0) or 0.0)
        if init_gu_frac > 0.0 and cfg.num_gu > 0:
            init_gu_frac = float(np.clip(init_gu_frac, 0.0, 1.0))
            self.gu_queue = np.full((cfg.num_gu,), cfg.queue_max_gu * init_gu_frac, dtype=np.float32)
        if init_uav_frac > 0.0 and cfg.num_uav > 0:
            init_uav_frac = float(np.clip(init_uav_frac, 0.0, 1.0))
            self.uav_queue = np.full((cfg.num_uav,), cfg.queue_max_uav * init_uav_frac, dtype=np.float32)
        if init_sat_frac > 0.0 and cfg.num_sat > 0:
            init_sat_frac = float(np.clip(init_sat_frac, 0.0, 1.0))
            self.sat_queue = np.full((cfg.num_sat,), cfg.queue_max_sat * init_sat_frac, dtype=np.float32)
        self.prev_queue_sum = 0.0
        self.prev_queue_sum_active = 0.0
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
        self.last_energy_cost = np.zeros((cfg.num_uav,), dtype=np.float32)
        self.last_sat_processed = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_sat_incoming = np.zeros((cfg.num_sat,), dtype=np.float32)
        self.last_bw_align = 0.0
        self.last_sat_score = 0.0
        self.last_arrival_rate = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))

        self.last_association = np.full((cfg.num_gu,), -1, dtype=np.int32)
        self.prev_association = self.last_association.copy()
        self.last_reward_parts = {
            "service_ratio": 0.0,
            "drop_ratio": 0.0,
            "drop_sum": 0.0,
            "drop_event": 0.0,
            "queue_pen": 0.0,
            "queue_pen_gu": 0.0,
            "queue_pen_uav": 0.0,
            "queue_pen_sat": 0.0,
            "queue_topk": 0.0,
            "assoc_ratio": 0.0,
            "queue_delta": 0.0,
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
            "term_service": 0.0,
            "term_drop": 0.0,
            "term_drop_step": 0.0,
            "term_queue": 0.0,
            "term_topk": 0.0,
            "term_assoc": 0.0,
            "term_q_delta": 0.0,
            "term_centroid": 0.0,
            "term_bw_align": 0.0,
            "term_sat_score": 0.0,
            "term_dist": 0.0,
            "term_dist_delta": 0.0,
            "term_energy": 0.0,
            "term_accel": 0.0,
            "reward_raw": 0.0,
        }
        self._cached_candidates: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        self._cached_eta = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        self._cached_sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, self.sat_dim), dtype=np.float32)
        self._cached_sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
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
        self._cached_candidates = self._build_candidate_users(assoc)
        dummy_actions = {
            agent: {
                "accel": np.zeros(2, dtype=np.float32),
                "bw_logits": np.zeros(self.cfg.users_obs_max, dtype=np.float32),
                "sat_logits": np.zeros(self.cfg.sats_obs_max, dtype=np.float32),
            }
            for agent in self.agents
        }
        _, self._cached_eta = self._compute_access_rates(assoc, self._cached_candidates, dummy_actions)
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        obs = {agent: self._get_obs(idx) for idx, agent in enumerate(self.agents)}
        infos = {
            agent: {
                "traffic_level": self.traffic_level,
                "traffic_level_ratio": self.traffic_level_ratio,
                "effective_task_arrival_rate": self.effective_task_arrival_rate,
            }
            for agent in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[str, Dict]):
        cfg = self.cfg
        self.global_step = int(getattr(self, "global_step", 0)) + 1
        self.prev_queue_sum = float(
            np.sum(self.gu_queue) + np.sum(self.uav_queue) + np.sum(self.sat_queue)
        )
        self.prev_queue_sum_active = float(np.sum(self.gu_queue) + np.sum(self.uav_queue))
        prev_scale = self._queue_arrival_scale(float(getattr(self, "prev_arrival_sum", 0.0)))
        self.prev_q_norm_active = float(np.clip(self.prev_queue_sum_active / prev_scale, 0.0, 1.0))
        self.prev_centroid_dist_mean = self._compute_centroid_stats()[1]
        if cfg.num_gu > 0:
            d2d = np.linalg.norm(self.gu_pos - self.uav_pos[:, None, :], axis=2)
            self.prev_d_min = float(np.min(d2d))
        else:
            self.prev_d_min = 0.0
        self._apply_uav_dynamics(actions)

        self.prev_association = self.last_association.copy()
        # Satellite states
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)

        # Compute associations and rates
        assoc = self._associate_users()
        candidate_lists = self._build_candidate_users(assoc)

        access_rates, eta = self._compute_access_rates(assoc, candidate_lists, actions)

        # Update GU queues
        gu_outflow = self._update_gu_queues(access_rates, assoc)

        # Backhaul selection and rates
        sat_selection = self._select_satellites(sat_pos, sat_vel, actions, visible)
        self._update_energy(sat_selection)
        rate_matrix, sat_loads = self._compute_backhaul_rates(sat_pos, sat_vel, sat_selection)

        # Update UAV queues and satellite queues
        outflow_matrix = self._update_uav_queues(gu_outflow, rate_matrix)
        self._update_sat_queues(outflow_matrix)

        # Cache for obs
        self._cached_candidates = candidate_lists
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        self._cached_eta = eta

        # Rewards and done
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

        obs = {agent: self._get_obs(idx) for idx, agent in enumerate(self.agents)}
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    def _apply_uav_dynamics(self, actions: Dict[str, Dict]) -> None:
        cfg = self.cfg
        accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
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
        repulse_mode = str(getattr(cfg, "avoidance_repulse_mode", "inverse") or "inverse").strip().lower()
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
            a_rep = np.zeros(2, dtype=np.float32)
            if use_avoidance and d_alert > 0.0:
                for j in range(cfg.num_uav):
                    if i == j:
                        continue
                    diff = self.uav_pos[i] - self.uav_pos[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < d_alert and dist > 1e-6:
                        direction = diff / dist
                        if repulse_mode == "linear":
                            denom = max(d_alert - cfg.d_safe, 1e-6)
                            strength = float(np.clip((d_alert - dist) / denom, 0.0, 1.0))
                        elif repulse_mode == "quadratic":
                            denom = max(d_alert - cfg.d_safe, 1e-6)
                            base = float(np.clip((d_alert - dist) / denom, 0.0, 1.0))
                            strength = base * base
                        else:
                            strength = (1.0 / dist - 1.0 / d_alert)
                        a_rep += eta_avoid * strength * direction
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        rates = np.zeros((cfg.num_gu,), dtype=np.float32)
        eta = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        bw_align_sum = 0.0
        bw_align_count = 0

        for u in range(cfg.num_uav):
            cand = candidates[u]
            if not cand:
                continue
            cand_idx = np.asarray(cand, dtype=np.int32)
            assoc_mask = assoc[cand_idx] == u
            if cfg.enable_bw_action:
                logits = np.array(actions[self.agents[u]]["bw_logits"], dtype=np.float32)
                mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                mask[: len(cand)] = assoc_mask.astype(np.float32)
                logits = logits[: cfg.users_obs_max]
                logits = logits - np.max(logits)
                weights = np.exp(logits) * mask
                denom = np.sum(weights)
                if denom > 0:
                    weights = weights / denom
                    betas = weights[: len(cand)]
                else:
                    betas = np.zeros((len(cand),), dtype=np.float32)
            else:
                if np.any(assoc_mask):
                    betas = np.zeros((len(cand),), dtype=np.float32)
                    betas[assoc_mask] = 1.0 / float(np.sum(assoc_mask))
                else:
                    betas = np.zeros((len(cand),), dtype=np.float32)

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
        for u in range(cfg.num_uav):
            vis = visible[u]
            if not vis:
                continue
            if cfg.fixed_satellite_strategy:
                # pick nearest visible
                dists = [np.linalg.norm(sat_pos[l] - self._uav_ecef(u)) for l in vis]
                sel = vis[int(np.argmin(dists))]
                selections[u] = [sel]
            else:
                logits = np.array(actions[self.agents[u]]["sat_logits"], dtype=np.float32)
                cand = vis[: cfg.sats_obs_max]
                if not cand:
                    continue
                valid_logits = logits[: len(cand)].copy()
                if cfg.doppler_enabled:
                    for i, l in enumerate(cand):
                        nu = self._doppler(u, l, sat_pos, sat_vel)
                        if abs(nu) > cfg.nu_max:
                            valid_logits[i] = -1e9

                if np.all(valid_logits <= -1e8):
                    continue

                if cfg.sat_select_mode == "sample":
                    probs = np.exp(valid_logits - np.max(valid_logits))
                    probs = probs / (np.sum(probs) + 1e-9)
                    chosen = []
                    probs = probs.copy()
                    for _ in range(min(cfg.N_RF, len(cand))):
                        if probs.sum() <= 0:
                            break
                        idx = self.rng.choice(len(cand), p=probs)
                        chosen.append(cand[idx])
                        probs[idx] = 0.0
                        probs = probs / (np.sum(probs) + 1e-9)
                    selections[u] = chosen
                else:
                    order = np.argsort(valid_logits)[::-1]
                    chosen = []
                    for idx in order:
                        if valid_logits[idx] <= -1e8:
                            continue
                        chosen.append(cand[idx])
                        if len(chosen) >= cfg.N_RF:
                            break
                    selections[u] = chosen
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
        processed = np.minimum(before + incoming, compute_rate * cfg.tau0)
        self.last_sat_processed = processed
        self.last_sat_incoming = incoming
        self.sat_queue = np.maximum(before + incoming - compute_rate * cfg.tau0, 0.0)

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

    def _compute_reward(self) -> float:
        cfg = self.cfg
        # Core dense reward for queueing: service/drop + absolute backlog + backlog trend.
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
        drop_sum = float(np.sum(self.gu_drop)) + float(np.sum(self.uav_drop))
        service_ratio = outflow_sum / (arrival_sum + 1e-9)
        drop_ratio = drop_sum / (arrival_sum + 1e-9)
        service_ratio = float(np.clip(service_ratio, 0.0, 1.0))
        drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))
        if cfg.num_gu > 0:
            assoc_ratio = float(np.mean(self.last_association >= 0))
        else:
            assoc_ratio = 0.0

        arrival_ref = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
        arrival_scale = max(arrival_ref * float(cfg.num_gu) * float(cfg.tau0), 1e-9)
        service_norm = outflow_sum / arrival_scale
        drop_norm = drop_sum / arrival_scale
        drop_event = 1.0 if drop_sum > 1e-9 else 0.0
        throughput_access_norm = outflow_sum / arrival_scale
        throughput_backhaul_norm = backhaul_sum / arrival_scale

        queue_norm_scale = self._queue_arrival_scale(arrival_sum)
        q_norm_active = float(np.clip(q_total_active / queue_norm_scale, 0.0, 1.0))
        prev_q_norm_active = float(getattr(self, "prev_q_norm_active", q_norm_active))
        q_norm_delta = float(prev_q_norm_active - q_norm_active)
        q_norm_tail_q0 = max(float(getattr(cfg, "q_norm_tail_q0", 0.0) or 0.0), 0.0)
        q_norm_tail_excess = 0.0
        queue_weight = float(cfg.omega_q)
        q_delta_weight = float(cfg.eta_q_delta)
        crash_weight = float(cfg.eta_crash)

        def _queue_smooth(q_norm: float) -> float:
            q_norm = float(np.clip(q_norm, 0.0, 1.0))
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
            queue_gu = _queue_smooth(q_gu_norm)
            queue_uav = _queue_smooth(q_uav_norm)
            queue_sat = _queue_smooth(q_sat_norm)
            w_gu = float(getattr(cfg, "omega_q_gu", 0.0) or 0.0)
            w_uav = float(getattr(cfg, "omega_q_uav", 0.0) or 0.0)
            w_sat = float(getattr(cfg, "omega_q_sat", 0.0) or 0.0)
            w_sum = abs(w_gu) + abs(w_uav) + abs(w_sat)
            if w_sum < 1e-9:
                queue_term = _queue_smooth(q_total / q_max_total)
            else:
                queue_term = (w_gu * queue_gu + w_uav * queue_uav + w_sat * queue_sat) / w_sum
            prev_sum = self.prev_queue_sum
            cur_sum = q_total
            q_delta_den = q_max_total
            queue_delta = (prev_sum - cur_sum) / max(q_delta_den, 1e-9)
            queue_delta = float(np.clip(queue_delta, -1.0, 1.0))
            queue_weight = float(cfg.omega_q)

        if cfg.a_max > 0:
            accel_norm2 = float(np.mean(np.sum(self.last_exec_accel**2, axis=1))) / (cfg.a_max**2 + 1e-9)
        else:
            accel_norm2 = 0.0

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
        term_service = cfg.eta_service * service_norm
        term_drop_step = -float(getattr(cfg, "eta_drop_step", 0.0) or 0.0) * drop_event
        term_drop = -cfg.eta_drop * drop_norm + term_drop_step
        term_queue = -queue_weight * queue_term
        term_q_delta = q_delta_weight * queue_delta
        term_centroid = centroid_eta * centroid_reward
        term_accel = -cfg.eta_accel * accel_norm2
        term_energy = cfg.omega_e * r_energy if use_energy_reward else 0.0
        raw_reward = (
            term_service
            + term_drop
            + term_queue
            + term_q_delta
            + term_centroid
            + term_accel
            + term_energy
        )

        collision_now = self._check_collision()
        collision_penalty = -crash_weight if collision_now else 0.0
        battery_penalty = -cfg.eta_batt if (cfg.energy_enabled and np.any(self.uav_energy <= 0.0)) else 0.0
        fail_penalty = collision_penalty + battery_penalty

        reward = raw_reward
        if use_reward_tanh:
            reward = math.tanh(raw_reward)
        reward = reward + fail_penalty

        dist_reward = 0.0
        term_topk = 0.0
        term_assoc = 0.0
        term_throughput_access = 0.0
        term_throughput_backhaul = 0.0
        term_dist = 0.0
        term_dist_delta = 0.0
        term_bw_align = 0.0
        term_sat_score = 0.0
        self.prev_arrival_sum = arrival_sum
        self.prev_q_norm_active = q_norm_active

        self.last_reward_parts = {
            "service_ratio": service_ratio,
            "drop_ratio": drop_ratio,
            "drop_sum": drop_sum,
            "drop_event": drop_event,
            "arrival_sum": arrival_sum,
            "outflow_sum": outflow_sum,
            "backhaul_sum": backhaul_sum,
            "service_norm": service_norm,
            "drop_norm": drop_norm,
            "throughput_access_norm": throughput_access_norm,
            "throughput_backhaul_norm": throughput_backhaul_norm,
            "queue_pen": queue_term,
            "queue_pen_gu": queue_gu,
            "queue_pen_uav": queue_uav,
            "queue_pen_sat": queue_sat,
            "queue_topk": queue_topk,
            "queue_total": q_total,
            "queue_total_active": q_total_active,
            "assoc_ratio": assoc_ratio,
            "queue_delta": queue_delta,
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
            "term_service": term_service,
            "term_drop": term_drop,
            "term_drop_step": term_drop_step,
            "term_queue": term_queue,
            "term_topk": term_topk,
            "term_assoc": term_assoc,
            "term_q_delta": term_q_delta,
            "term_throughput_access": term_throughput_access,
            "term_throughput_backhaul": term_throughput_backhaul,
            "term_dist": term_dist,
            "term_dist_delta": term_dist_delta,
            "term_centroid": term_centroid,
            "term_bw_align": term_bw_align,
            "term_sat_score": term_sat_score,
            "term_energy": float(term_energy),
            "term_accel": term_accel,
            "reward_raw": raw_reward,
        }
        return float(reward)

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
        cfg = self.cfg
        r_u = cfg.r_earth + cfg.uav_height
        r_s = cfg.r_earth + cfg.sat_height
        q = self._uav_ecef(u)
        d = np.linalg.norm(sat_pos[l] - q)
        arg = (r_s ** 2 - r_u ** 2 - d ** 2) / (2.0 * r_u * d + 1e-9)
        arg = np.clip(arg, -1.0, 1.0)
        return float(math.asin(arg))

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

    def _visible_sats_sorted(self, sat_pos: np.ndarray) -> List[List[int]]:
        cfg = self.cfg
        visible: List[List[int]] = [[] for _ in range(cfg.num_uav)]
        for u in range(cfg.num_uav):
            elev_list = []
            for l in range(cfg.num_sat):
                theta = self._elevation_angle(u, l, sat_pos)
                elev_list.append((l, theta))
            elev_list.sort(key=lambda x: x[1], reverse=True)

            above = [l for l, th in elev_list if th >= cfg.theta_min_rad]
            min_keep = cfg.visible_sats_min
            if min_keep is not None and len(above) < min_keep:
                # Include highest-elevation satellites even if below theta_min to enforce minimum visibility.
                needed = min_keep - len(above)
                extra = [l for l, th in elev_list if th < cfg.theta_min_rad][:needed]
                above.extend(extra)

            max_keep = cfg.visible_sats_max if cfg.visible_sats_max is not None else cfg.sats_obs_max
            visible[u] = above[: max_keep]
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
        for u in range(cfg.num_uav):
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
                sat_mask[u, i] = 1.0
        self._cached_sat_obs = sat_obs
        self._cached_sat_mask = sat_mask

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

        return {
            "own": own,
            "users": users,
            "users_mask": users_mask,
            "sats": sats,
            "sats_mask": sats_mask,
            "nbrs": nbrs,
            "nbrs_mask": nbrs_mask,
        }

    def get_global_state(self) -> np.ndarray:
        cfg = self.cfg
        # Flatten global state for critic
        sat_pos, sat_vel = self._get_orbit_states()
        sat_idx = None
        if cfg.sat_state_max is not None and cfg.sat_state_max < cfg.num_sat:
            scores = np.full((cfg.num_sat,), -np.inf, dtype=np.float32)
            for l in range(cfg.num_sat):
                max_theta = -1e9
                for u in range(cfg.num_uav):
                    theta = self._elevation_angle(u, l, sat_pos)
                    if theta > max_theta:
                        max_theta = theta
                scores[l] = max_theta
            sat_idx = np.argsort(scores)[::-1][: cfg.sat_state_max]
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
        return np.concatenate(parts).astype(np.float32)

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
