from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np
import torch

from sagin_marl.env import channel
from sagin_marl.env.numeric_guards import (
    GEOMETRY_DENOM_EPS,
    NORMALIZATION_DENOM_EPS,
    geometry_denominator,
    normalize_scale,
    ratio_or_zero,
)
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl.structured_types import BwStageSnapshot, SatStageSnapshot, StructuredWorldState

if TYPE_CHECKING:
    from sagin_marl.rl.structured_types import LocalAccelState, LocalBwState, LocalSatState


def _project_normalized_accel_np(accel: np.ndarray) -> np.ndarray:
    arr = np.asarray(accel, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, 1.0 / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


@dataclass
class StructuredStepResult:
    obs: Dict[str, Dict[str, np.ndarray]]
    rewards: Dict[str, float]
    terminations: Dict[str, bool]
    truncations: Dict[str, bool]
    infos: Dict[str, Dict[str, object]]
    danger_imitation_target: np.ndarray | None
    danger_imitation_mask: np.ndarray | None
    team_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    reward_parts: Dict[str, Any] | None = None
    bw_access_reward: float = 0.0
    bw_weighted_workload_delta_reward: float = 0.0
    bw_weighted_workload_level_reward: float = 0.0
    bw_gu_queue_level_reward: float = 0.0
    bw_system_queue_level_reward: float = 0.0
    bw_gu_service_queue_reward: float = 0.0
    bw_flow_proxy_scores: np.ndarray | None = None
    bw_flow_proxy_mask: np.ndarray | None = None
    bw_flow_proxy_deltas: np.ndarray | None = None
    next_accel_spec: Dict[str, Any] | None = None


@dataclass
class StructuredBatchStepResult:
    team_rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    danger_imitation_target: torch.Tensor | None = None
    danger_imitation_mask: torch.Tensor | None = None
    bw_access_rewards: torch.Tensor | None = None
    bw_weighted_workload_delta_rewards: torch.Tensor | None = None
    bw_weighted_workload_level_rewards: torch.Tensor | None = None
    bw_gu_queue_level_rewards: torch.Tensor | None = None
    bw_system_queue_level_rewards: torch.Tensor | None = None
    bw_gu_service_queue_rewards: torch.Tensor | None = None
    bw_flow_proxy_scores: torch.Tensor | None = None
    bw_flow_proxy_mask: torch.Tensor | None = None
    bw_flow_proxy_deltas: torch.Tensor | None = None
    reward_part_tensors: Dict[str, torch.Tensor] | None = None
    reward_mode_active: str | None = None

    @property
    def num_envs(self) -> int:
        return int(self.team_rewards.reshape(-1).shape[0])

    def __len__(self) -> int:
        return self.num_envs


@lru_cache(maxsize=32)
def _subset_member_spec_cpu(sat_count: int, max_select: int) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    subset_specs: list[tuple[int, ...]] = [tuple()]
    for k in range(1, min(int(max_select), int(sat_count)) + 1):
        subset_specs.extend(combinations(range(int(sat_count)), k))
    subset_sizes = tuple(len(spec) for spec in subset_specs)
    return tuple(subset_specs), subset_sizes


class StructuredControlDriver:
    STAGE_ACCEL = 0
    STAGE_SAT = 1
    STAGE_BW = 2

    def __init__(self, env, *, tensor_device: torch.device | str | None = None) -> None:
        self.env = env
        self._tensor_device = None if tensor_device is None else torch.device(tensor_device)
        self._step_open = False
        self._stage_assoc: np.ndarray | None = None
        self._stage_candidates: List[List[int]] | None = None
        self._stage_bw_valid_mask: np.ndarray | None = None
        self._stage_visible: List[List[int]] | None = None
        self._stage_sat_pos: np.ndarray | None = None
        self._stage_sat_vel: np.ndarray | None = None
        self._stage_sat_selection: List[List[int]] | None = None
        self._stage_sat_selection_matrix: np.ndarray | None = None
        self._stage_uav_ecef_all: np.ndarray | None = None
        self._stage_uav_vel_ecef_all: np.ndarray | None = None
        self._stage_access_gain_matrix: np.ndarray | None = None
        self._stage_uav_gu_rel: np.ndarray | None = None
        self._stage_candidate_flag: np.ndarray | None = None
        self._stage_bw_valid_flag: np.ndarray | None = None
        self._stage_prev_assoc_flag: np.ndarray | None = None
        self._stage_eta_ref_feature: np.ndarray | None = None
        self._stage_uav_uav_rel_pos: np.ndarray | None = None
        self._stage_uav_uav_rel_vel: np.ndarray | None = None
        self._stage_uav_uav_dist: np.ndarray | None = None
        self._stage_active_sat_ids: np.ndarray | None = None
        self._stage_sat_nodes_static: np.ndarray | None = None
        self._stage_sat_mask_cached: np.ndarray | None = None
        self._stage_us_rel_pos: np.ndarray | None = None
        self._stage_us_rel_vel: np.ndarray | None = None
        self._stage_us_gain: np.ndarray | None = None
        self._stage_us_nu_eff: np.ndarray | None = None
        self._stage_us_visible_flag: np.ndarray | None = None
        self._stage_us_valid_flag: np.ndarray | None = None
        self._stage_us_sat_queue: np.ndarray | None = None
        self._stage_obs_cache: list[dict[str, np.ndarray]] | None = None
        self._accel_stage_spec_cache: Dict[str, Any] | None = None

    @property
    def tensor_device(self) -> torch.device | None:
        return self._tensor_device

    def set_tensor_device(self, device: torch.device | str | None) -> None:
        self._tensor_device = None if device is None else torch.device(device)

    def _tensor_output_enabled(self) -> bool:
        return self._tensor_device is not None

    def _array_to_output_device(
        self,
        value: np.ndarray | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> np.ndarray | torch.Tensor:
        if not self._tensor_output_enabled():
            return value
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device=self._tensor_device)

    def _world_state_to_output_device(self, world_state: StructuredWorldState) -> StructuredWorldState:
        if not self._tensor_output_enabled():
            return world_state
        return StructuredWorldState(
            uav_nodes=self._array_to_output_device(world_state.uav_nodes, dtype=torch.float32),
            gu_nodes=self._array_to_output_device(world_state.gu_nodes, dtype=torch.float32),
            sat_nodes=self._array_to_output_device(world_state.sat_nodes, dtype=torch.float32),
            sat_ids=self._array_to_output_device(world_state.sat_ids, dtype=torch.long),
            uav_gu_edges=self._array_to_output_device(world_state.uav_gu_edges, dtype=torch.float32),
            uav_sat_edges=self._array_to_output_device(world_state.uav_sat_edges, dtype=torch.float32),
            uav_uav_edges=self._array_to_output_device(world_state.uav_uav_edges, dtype=torch.float32),
            global_scalars=self._array_to_output_device(world_state.global_scalars, dtype=torch.float32),
            gu_mask=self._array_to_output_device(world_state.gu_mask, dtype=torch.bool),
            sat_mask=self._array_to_output_device(world_state.sat_mask, dtype=torch.bool),
            uav_gu_mask=self._array_to_output_device(world_state.uav_gu_mask, dtype=torch.bool),
            uav_sat_mask=self._array_to_output_device(world_state.uav_sat_mask, dtype=torch.bool),
            uav_uav_mask=self._array_to_output_device(world_state.uav_uav_mask, dtype=torch.bool),
            stage_id=self._array_to_output_device(world_state.stage_id, dtype=torch.long),
        )

    def _ensure_step_started(self) -> None:
        if self._step_open:
            return
        env = self.env
        cfg = env.cfg
        env.global_step = int(getattr(env, "global_step", 0)) + 1
        env.prev_queue_sum = float(np.sum(env.gu_queue) + np.sum(env.uav_queue) + np.sum(env.sat_queue))
        env.prev_queue_sum_active = float(np.sum(env.gu_queue) + np.sum(env.uav_queue))
        env.prev_queue_sum_gu = float(np.sum(env.gu_queue))
        env.prev_queue_sum_uav = float(np.sum(env.uav_queue))
        env.prev_queue_sum_sat = float(np.sum(env.sat_queue))
        env.prev_gu_queue_vec = np.asarray(env.gu_queue, dtype=np.float32).copy()
        env.prev_uav_queue_vec = np.asarray(env.uav_queue, dtype=np.float32).copy()
        env.prev_sat_queue_vec = np.asarray(env.sat_queue, dtype=np.float32).copy()
        prev_scale = env._queue_arrival_scale(float(getattr(env, "prev_arrival_sum", 0.0)))
        env.prev_q_norm_active = float(np.clip(env.prev_queue_sum_active / prev_scale, 0.0, 1.0))
        env.prev_centroid_dist_mean = env._compute_centroid_stats()[1]
        if cfg.num_gu > 0:
            d2d = np.linalg.norm(env.gu_pos - env.uav_pos[:, None, :], axis=2)
            env.prev_d_min = float(np.min(d2d))
        else:
            env.prev_d_min = 0.0
        self._step_open = True

    def _clear_step(self) -> None:
        self._step_open = False
        self._stage_assoc = None
        self._stage_candidates = None
        self._stage_bw_valid_mask = None
        self._stage_visible = None
        self._stage_sat_pos = None
        self._stage_sat_vel = None
        self._stage_sat_selection = None
        self._stage_sat_selection_matrix = None
        self._stage_uav_ecef_all = None
        self._stage_uav_vel_ecef_all = None
        self._stage_access_gain_matrix = None
        self._stage_uav_gu_rel = None
        self._stage_candidate_flag = None
        self._stage_bw_valid_flag = None
        self._stage_prev_assoc_flag = None
        self._stage_eta_ref_feature = None
        self._stage_uav_uav_rel_pos = None
        self._stage_uav_uav_rel_vel = None
        self._stage_uav_uav_dist = None
        self._stage_active_sat_ids = None
        self._stage_sat_nodes_static = None
        self._stage_sat_mask_cached = None
        self._stage_us_rel_pos = None
        self._stage_us_rel_vel = None
        self._stage_us_gain = None
        self._stage_us_nu_eff = None
        self._stage_us_visible_flag = None
        self._stage_us_valid_flag = None
        self._stage_us_sat_queue = None
        self._stage_obs_cache = None
        self._accel_stage_spec_cache = None

    def _refresh_world_build_cache(
        self,
        *,
        access_gain_override: np.ndarray | None = None,
    ) -> None:
        if self._stage_assoc is None or self._stage_candidates is None or self._stage_visible is None:
            return
        env = self.env
        cfg = env.cfg
        self._stage_uav_ecef_all = np.stack([env._uav_ecef(u) for u in range(cfg.num_uav)], axis=0).astype(np.float32, copy=False)
        self._stage_uav_vel_ecef_all = np.stack([env._uav_vel_ecef(u) for u in range(cfg.num_uav)], axis=0).astype(np.float32, copy=False)
        if cfg.num_gu > 0:
            access_snapshot = (
                np.asarray(access_gain_override, dtype=np.float32)
                if access_gain_override is not None
                else env._sample_access_channel_snapshot()
            )
            self._stage_access_gain_matrix = np.asarray(
                env._coerce_access_gain_matrix(access_snapshot),
                dtype=np.float32,
            )
            self._stage_uav_gu_rel = (env.gu_pos[None, :, :] - env.uav_pos[:, None, :]).astype(np.float32, copy=False)
            candidate_flag = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
            bw_valid_flag = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
            for u in range(cfg.num_uav):
                cand = self._stage_candidates[u][: cfg.users_obs_max]
                for slot, gu_idx in enumerate(cand):
                    candidate_flag[u, int(gu_idx)] = 1.0
                bw_valid_flag[u] = (np.asarray(self._stage_assoc, dtype=np.int32) == int(u)).astype(np.float32)
            self._stage_candidate_flag = candidate_flag
            self._stage_bw_valid_flag = bw_valid_flag
            self._stage_prev_assoc_flag = (env.prev_association[None, :] == np.arange(cfg.num_uav, dtype=np.int64)[:, None]).astype(np.float32, copy=False)
            self._stage_eta_ref_feature = self._reference_eta_feature(
                self._stage_assoc,
                self._stage_candidates,
                access_snapshot=access_snapshot,
            )
        else:
            self._stage_access_gain_matrix = None
            self._stage_uav_gu_rel = None
            self._stage_candidate_flag = None
            self._stage_bw_valid_flag = None
            self._stage_prev_assoc_flag = None
            self._stage_eta_ref_feature = None

        rel_pos = (env.uav_pos[None, :, :] - env.uav_pos[:, None, :]).astype(np.float32, copy=False)
        rel_vel = (env.uav_vel[None, :, :] - env.uav_vel[:, None, :]).astype(np.float32, copy=False)
        dist = np.linalg.norm(rel_pos, axis=-1).astype(np.float32, copy=False)
        self._stage_uav_uav_rel_pos = rel_pos
        self._stage_uav_uav_rel_vel = rel_vel
        self._stage_uav_uav_dist = dist

        if self._stage_sat_pos is None or self._stage_sat_vel is None:
            self._stage_active_sat_ids = None
            self._stage_sat_nodes_static = None
            self._stage_sat_mask_cached = None
            self._stage_us_rel_pos = None
            self._stage_us_rel_vel = None
            self._stage_us_gain = None
            self._stage_us_nu_eff = None
            self._stage_us_visible_flag = None
            self._stage_us_valid_flag = None
            self._stage_us_sat_queue = None
            return

        active_sat_ids = self._active_sat_ids(self._stage_visible, None)
        self._stage_active_sat_ids = active_sat_ids
        sat_count = int(active_sat_ids.size)
        sat_node_dim = 8 + int(bool(getattr(cfg, "obs_sat_include_sat_cost", False)))
        if sat_count <= 0:
            self._stage_sat_nodes_static = np.zeros((1, 0, sat_node_dim), dtype=np.float32)
            self._stage_sat_mask_cached = np.zeros((1, 0), dtype=bool)
            self._stage_us_rel_pos = np.zeros((cfg.num_uav, 0, 3), dtype=np.float32)
            self._stage_us_rel_vel = np.zeros((cfg.num_uav, 0, 3), dtype=np.float32)
            self._stage_us_gain = np.zeros((cfg.num_uav, 0), dtype=np.float32)
            self._stage_us_nu_eff = np.zeros((cfg.num_uav, 0), dtype=np.float32)
            self._stage_us_visible_flag = np.zeros((cfg.num_uav, 0), dtype=np.float32)
            self._stage_us_valid_flag = np.zeros((cfg.num_uav, 0), dtype=np.float32)
            self._stage_us_sat_queue = np.zeros((0,), dtype=np.float32)
            return

        sat_pos = np.asarray(self._stage_sat_pos, dtype=np.float32)
        sat_vel = np.asarray(self._stage_sat_vel, dtype=np.float32)
        sat_nodes_static = np.zeros((1, sat_count, sat_node_dim), dtype=np.float32)
        sat_nodes_static[0, :, 0:3] = sat_pos[active_sat_ids] / (cfg.r_earth + cfg.sat_height)
        sat_nodes_static[0, :, 3:6] = sat_vel[active_sat_ids] / (cfg.r_earth + cfg.sat_height)
        sat_nodes_static[0, :, 6] = env.sat_queue[active_sat_ids] / normalize_scale(cfg.queue_max_sat)
        if bool(getattr(cfg, "obs_sat_include_sat_cost", False)):
            sat_reward_features = env._sat_reward_aligned_feature_dict(
                normalized=True,
                assoc_override=self._stage_assoc,
                sat_selection_override=self._stage_sat_selection,
            )
            sat_nodes_static[0, :, 8] = np.asarray(sat_reward_features["sat_cost"][active_sat_ids], dtype=np.float32)
        self._stage_sat_nodes_static = sat_nodes_static
        self._stage_sat_mask_cached = np.ones((1, sat_count), dtype=bool)

        rel_pos = sat_pos[active_sat_ids][None, :, :] - self._stage_uav_ecef_all[:, None, :]
        rel_vel = sat_vel[active_sat_ids][None, :, :] - self._stage_uav_vel_ecef_all[:, None, :]
        dist = geometry_denominator(np.linalg.norm(rel_pos, axis=-1))
        gain = (env._backhaul_gain_const / geometry_denominator(dist * dist)).astype(np.float32, copy=False)
        loss_matrix = env._get_backhaul_loss_matrix(sat_pos)
        elevation_matrix = env._get_elevation_matrix(sat_pos)
        if loss_matrix is not None:
            gain = gain * loss_matrix[:, active_sat_ids]
        if cfg.doppler_enabled or cfg.doppler_atten_enabled or cfg.doppler_observed:
            nu_eff = np.zeros((cfg.num_uav, sat_count), dtype=np.float32)
            for u in range(cfg.num_uav):
                raw_nu = env._doppler_many(u, active_sat_ids, sat_pos, sat_vel)
                nu_eff_u, _ = env._effective_doppler_array(u, active_sat_ids, raw_nu)
                nu_eff[u] = nu_eff_u.astype(np.float32, copy=False)
        else:
            nu_eff = np.zeros((cfg.num_uav, sat_count), dtype=np.float32)
        visible_flag = np.zeros((cfg.num_uav, sat_count), dtype=np.float32)
        valid_flag = np.zeros((cfg.num_uav, sat_count), dtype=np.float32)
        active_sat_ids_list = active_sat_ids.tolist()
        for u in range(cfg.num_uav):
            vis_set = set(self._stage_visible[u])
            visible_flag[u] = np.asarray([1.0 if sat_idx in vis_set else 0.0 for sat_idx in active_sat_ids_list], dtype=np.float32)
            valid = elevation_matrix[u, active_sat_ids] >= cfg.theta_min_rad
            if cfg.doppler_enabled:
                valid = valid & (np.abs(nu_eff[u]) <= cfg.nu_max)
            valid_flag[u] = valid.astype(np.float32, copy=False)
        self._stage_us_rel_pos = rel_pos.astype(np.float32, copy=False)
        self._stage_us_rel_vel = rel_vel.astype(np.float32, copy=False)
        self._stage_us_gain = gain.astype(np.float32, copy=False)
        self._stage_us_nu_eff = nu_eff
        self._stage_us_visible_flag = visible_flag
        self._stage_us_valid_flag = valid_flag
        self._stage_us_sat_queue = (env.sat_queue[active_sat_ids] / normalize_scale(cfg.queue_max_sat)).astype(np.float32, copy=False)

    @staticmethod
    def _candidate_lists_equal(left: List[List[int]] | None, right: List[List[int]] | None) -> bool:
        if left is None or right is None or len(left) != len(right):
            return False
        return all(list(a) == list(b) for a, b in zip(left, right))

    def _zero_bw_action_matrix(self) -> np.ndarray:
        cfg = self.env.cfg
        return np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)

    def _reference_eta_feature(
        self,
        assoc: np.ndarray,
        candidates: List[List[int]],
        *,
        access_snapshot=None,
    ) -> np.ndarray:
        env = self.env
        cfg = env.cfg
        cached_eta = getattr(env, "_cached_eta", None)
        use_cached = cached_eta is not None and bool(getattr(env, "_cached_eta_matches")(assoc, candidates))
        if use_cached:
            eta_slots = np.asarray(cached_eta, dtype=np.float32)
        else:
            if access_snapshot is None:
                use_stage_gain = (
                    self._stage_access_gain_matrix is not None
                    and self._stage_assoc is not None
                    and np.array_equal(self._stage_assoc, assoc)
                    and self._candidate_lists_equal(self._stage_candidates, candidates)
                )
                if use_stage_gain:
                    access_snapshot = np.asarray(self._stage_access_gain_matrix, dtype=np.float32)
            _, eta_slots = env._compute_access_rates(
                assoc,
                candidates,
                self._zero_bw_action_matrix(),
                record_exec=False,
                access_snapshot=access_snapshot,
            )
        eta_feature = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
        for u in range(cfg.num_uav):
            cand = candidates[u][: cfg.users_obs_max]
            for slot, gu_idx in enumerate(cand):
                gu_id = int(gu_idx)
                if 0 <= gu_id < cfg.num_gu:
                    eta_feature[u, gu_id] = float(eta_slots[u, slot])
        return eta_feature

    def _get_stage_eta_ref_feature(self, assoc: np.ndarray, candidates: List[List[int]]) -> np.ndarray:
        if (
            self._stage_eta_ref_feature is not None
            and self._stage_assoc is not None
            and np.array_equal(self._stage_assoc, assoc)
            and self._candidate_lists_equal(self._stage_candidates, candidates)
        ):
            return self._stage_eta_ref_feature
        return self._reference_eta_feature(assoc, candidates)

    def begin_step(self) -> StructuredWorldState:
        spec = self._prepare_accel_stage_spec()
        self._accel_stage_spec_cache = spec
        return self._build_world_state_from_spec(spec)

    def _prepare_accel_stage_spec(self) -> Dict[str, Any]:
        assoc = self.env._associate_users()
        candidates = self.env._build_candidate_users(assoc)
        bw_valid_mask = self.env._build_bw_valid_mask(assoc, candidates)
        sat_pos, sat_vel = self.env._get_orbit_states()
        visible = self.env._visible_sats_sorted(sat_pos, record_stats=False)
        access_snapshot = self.env._sample_access_channel_snapshot()
        access_gain_matrix = self.env._coerce_access_gain_matrix(access_snapshot)
        return {
            "env": self.env,
            "driver": self,
            "stage_id": self.STAGE_ACCEL,
            "assoc": assoc,
            "candidates": candidates,
            "sat_pos": sat_pos,
            "sat_vel": sat_vel,
            "visible": visible,
            "sat_selection": None,
            "active_sat_ids": self._active_sat_ids(visible, None),
            "sat_loads": self._compute_stage_sat_selection_counts(None),
            "bw_valid_mask": bw_valid_mask,
            "access_gain_matrix": np.asarray(access_gain_matrix, dtype=np.float32),
        }

    def _build_world_state_from_spec(self, spec: Dict[str, Any]) -> StructuredWorldState:
        self._stage_assoc = np.asarray(spec["assoc"], dtype=np.int32).copy()
        self._stage_candidates = [list(c) for c in spec["candidates"]]
        self._stage_bw_valid_mask = (
            None if spec.get("bw_valid_mask") is None else np.asarray(spec["bw_valid_mask"], dtype=np.float32).copy()
        )
        self._stage_visible = [list(v) for v in spec["visible"]]
        self._stage_sat_pos = np.asarray(spec["sat_pos"], dtype=np.float32).copy()
        self._stage_sat_vel = np.asarray(spec["sat_vel"], dtype=np.float32).copy()
        self._stage_sat_selection = (
            None if spec.get("sat_selection") is None else [list(sel) for sel in spec["sat_selection"]]
        )
        self._stage_obs_cache = None
        self._refresh_world_build_cache(access_gain_override=spec.get("access_gain_matrix"))
        return self._build_world_state(
            stage_id=int(spec["stage_id"]),
            assoc=np.asarray(spec["assoc"], dtype=np.int32),
            candidates=[list(c) for c in spec["candidates"]],
            sat_pos=np.asarray(spec["sat_pos"], dtype=np.float32),
            sat_vel=np.asarray(spec["sat_vel"], dtype=np.float32),
            visible=[list(v) for v in spec["visible"]],
            sat_selection=(
                None
                if spec.get("sat_selection") is None
                else [list(sel) for sel in spec["sat_selection"]]
            ),
        )

    def _prepare_sat_stage_spec(self) -> Dict[str, Any]:
        if self._stage_assoc is None or self._stage_visible is None or self._stage_sat_pos is None or self._stage_sat_vel is None:
            raise RuntimeError("run_accel_stage must be called before build_sat_stage_snapshot")
        if self._stage_access_gain_matrix is None:
            self._refresh_world_build_cache()
        return {
            "env": self.env,
            "driver": self,
            "stage_id": self.STAGE_SAT,
            "assoc": np.asarray(self._stage_assoc, dtype=np.int32),
            "candidates": [list(c) for c in self._stage_candidates or [[] for _ in range(self.env.cfg.num_uav)]],
            "sat_pos": np.asarray(self._stage_sat_pos, dtype=np.float32),
            "sat_vel": np.asarray(self._stage_sat_vel, dtype=np.float32),
            "visible": [list(v) for v in self._stage_visible],
            "sat_selection": None,
            "active_sat_ids": self._active_sat_ids(self._stage_visible, None),
            "sat_loads": self._compute_stage_sat_selection_counts(None),
            "bw_valid_mask": None if self._stage_bw_valid_mask is None else np.asarray(self._stage_bw_valid_mask, dtype=np.float32),
            "access_gain_matrix": np.asarray(self._stage_access_gain_matrix, dtype=np.float32),
        }

    def _apply_accel_action_in_place(
        self,
        accel_action: Sequence[np.ndarray] | np.ndarray,
        *,
        access_gain_override: np.ndarray | None = None,
    ) -> None:
        self._ensure_step_started()
        self._accel_stage_spec_cache = None
        env = self.env
        accel_arr = np.asarray(accel_action, dtype=np.float32)
        if accel_arr.shape != (env.cfg.num_uav, 2):
            raise ValueError(f"accel_action shape must be ({env.cfg.num_uav}, 2)")
        env.prev_association = env.last_association.copy()
        env._apply_uav_dynamics(accel_arr)
        sat_pos, sat_vel = env._get_orbit_states()
        assoc = env._associate_users()
        candidates = env._build_candidate_users(assoc)
        bw_valid_mask = env._build_bw_valid_mask(assoc, candidates)
        visible = env._visible_sats_sorted(sat_pos, record_stats=False)
        self._stage_assoc = assoc.copy()
        self._stage_candidates = [list(c) for c in candidates]
        self._stage_bw_valid_mask = bw_valid_mask.copy()
        self._stage_visible = [list(v) for v in visible]
        self._stage_sat_pos = sat_pos.copy()
        self._stage_sat_vel = sat_vel.copy()
        self._stage_sat_selection = None
        self._stage_obs_cache = None
        self._refresh_world_build_cache(access_gain_override=access_gain_override)

    def _run_accel_stage_to_spec(
        self,
        accel_action: Sequence[np.ndarray] | np.ndarray,
        *,
        access_gain_override: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        self._apply_accel_action_in_place(accel_action, access_gain_override=access_gain_override)
        return self._prepare_sat_stage_spec()

    def run_accel_stage(
        self,
        accel_action: Sequence[np.ndarray] | np.ndarray,
        *,
        access_gain_override: np.ndarray | None = None,
    ) -> StructuredWorldState:
        return self._build_world_state_from_spec(
            self._run_accel_stage_to_spec(
                accel_action,
                access_gain_override=access_gain_override,
            )
        )

    def build_local_accel_states(self, world_state: StructuredWorldState | None = None) -> List[LocalAccelState]:
        del world_state
        from sagin_marl.rl.structured_stage_builders import build_local_accel_states_from_spec

        if self._accel_stage_spec_cache is None:
            self._accel_stage_spec_cache = self._prepare_accel_stage_spec()
        return build_local_accel_states_from_spec(self._accel_stage_spec_cache, device=self._tensor_device)

    def build_sat_stage_snapshot(self, state_after_accel: StructuredWorldState | None = None) -> SatStageSnapshot:
        from sagin_marl.rl.structured_stage_builders import build_batched_local_sat_states_from_spec

        spec = self._prepare_sat_stage_spec()
        ws = state_after_accel if state_after_accel is not None else self._build_world_state_from_spec(spec)
        select_k = max(int(getattr(self.env.cfg, "sat_action_select_k", self.env.cfg.N_RF) or self.env.cfg.N_RF), 1)
        local_state = build_batched_local_sat_states_from_spec(spec, device=self._tensor_device)
        return SatStageSnapshot(world_state=ws, max_select=select_k, local_state=local_state)

    def build_local_sat_states(self, state_after_accel: StructuredWorldState | None = None) -> List[LocalSatState]:
        del state_after_accel
        from sagin_marl.rl.structured_stage_builders import build_local_sat_states_from_spec

        return build_local_sat_states_from_spec(self._prepare_sat_stage_spec(), device=self._tensor_device)

    def decode_sat_subset_actions(self, local_states: Sequence[LocalSatState], subset_indices: Sequence[int]) -> np.ndarray:
        del local_states
        _selections, actions = self._sat_selections_from_subset_indices(subset_indices)
        return actions

    def _sat_selections_from_subset_indices(self, subset_indices: Sequence[int]) -> tuple[list[list[int]], np.ndarray]:
        cfg = self.env.cfg
        select_k = max(int(getattr(cfg, "sat_action_select_k", cfg.N_RF) or cfg.N_RF), 1)
        if self._stage_visible is None:
            raise RuntimeError("run_accel_stage must be called before decode_sat_subset_actions")
        if len(subset_indices) != cfg.num_uav:
            raise ValueError(f"subset_indices length must be {cfg.num_uav}")
        actions = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
        selections: list[list[int]] = [[] for _ in range(cfg.num_uav)]
        for u, subset_index in enumerate(subset_indices):
            sat_ids = np.asarray(
                self._stage_visible[u][: int(getattr(cfg, "per_uav_visible_sat_token_max", cfg.sats_obs_max) or cfg.sats_obs_max)],
                dtype=np.int64,
            )
            if subset_index is None or int(subset_index) < 0 or sat_ids.size == 0:
                continue
            subset_specs, _ = _subset_member_spec_cpu(
                int(min(int(getattr(cfg, "per_uav_visible_sat_token_max", cfg.sats_obs_max) or cfg.sats_obs_max), int(cfg.num_sat))),
                int(select_k),
            )
            subset_idx = int(subset_index)
            if subset_idx < 0 or subset_idx >= len(subset_specs):
                continue
            local_subset = np.asarray(subset_specs[subset_idx], dtype=np.int64)
            if local_subset.size > 0 and np.any(local_subset >= sat_ids.size):
                continue
            mapped = sat_ids[local_subset] if local_subset.size > 0 else np.zeros((0,), dtype=np.int64)
            mapped = mapped[:select_k]
            actions[u, : mapped.size] = mapped
            selections[u] = mapped.tolist()
        return selections, actions

    def _sat_selections_from_action(
        self,
        sat_action: Sequence[Sequence[int]] | np.ndarray,
    ) -> tuple[list[list[int]], List[List[int]] | np.ndarray]:
        if self._stage_assoc is None or self._stage_visible is None or self._stage_sat_pos is None or self._stage_sat_vel is None:
            raise RuntimeError("run_accel_stage must be called before run_sat_stage")
        if bool(getattr(self.env.cfg, "fixed_satellite_strategy", False)):
            selections: List[List[int]] = []
            for u in range(self.env.cfg.num_uav):
                visible = list(self._stage_visible[u])
                if not visible:
                    selections.append([])
                    continue
                visible_idx = np.asarray(visible, dtype=np.int32)
                dists = np.linalg.norm(
                    np.asarray(self._stage_sat_pos, dtype=np.float32)[visible_idx]
                    - self.env._uav_ecef(u)[None, :],
                    axis=1,
                )
                nearest = int(visible_idx[int(np.argmin(dists))])
                selections.append([nearest])
            return selections, selections
        raw_action_arr = np.asarray(sat_action)
        if not np.issubdtype(raw_action_arr.dtype, np.integer):
            if not np.all(np.isfinite(raw_action_arr)):
                raise ValueError("sat_action must contain finite integer SAT ids or -1 padding")
            if not np.all(raw_action_arr == np.floor(raw_action_arr)):
                raise ValueError("sat_action must contain finite integer SAT ids or -1 padding")
        sat_action_arr = raw_action_arr.astype(np.int64, copy=False)
        select_k = max(int(getattr(self.env.cfg, "sat_action_select_k", self.env.cfg.N_RF) or self.env.cfg.N_RF), 1)
        if sat_action_arr.shape != (self.env.cfg.num_uav, select_k):
            raise ValueError(f"sat_action must have shape ({self.env.cfg.num_uav}, {select_k})")
        elevation_matrix = self.env._get_elevation_matrix(np.asarray(self._stage_sat_pos, dtype=np.float32))
        selections = []
        candidate_width = int(
            getattr(self.env.cfg, "per_uav_visible_sat_token_max", self.env.cfg.sats_obs_max)
            or self.env.cfg.sats_obs_max
        )
        candidate_width = max(min(candidate_width, int(self.env.cfg.num_sat)), 0)
        for u in range(self.env.cfg.num_uav):
            visible_candidates = [int(s) for s in self._stage_visible[u][:candidate_width]]
            visible_set = set(visible_candidates)
            valid_set: set[int] = set()
            for candidate in visible_candidates:
                if candidate < 0 or candidate >= int(self.env.cfg.num_sat):
                    continue
                if elevation_matrix[u, candidate] < float(self.env.cfg.theta_min_rad):
                    continue
                if bool(getattr(self.env.cfg, "doppler_enabled", False)):
                    raw_nu = self.env._doppler_many(
                        u,
                        np.asarray([candidate], dtype=np.int64),
                        np.asarray(self._stage_sat_pos, dtype=np.float32),
                        np.asarray(self._stage_sat_vel, dtype=np.float32),
                    )
                    nu_eff, _ = self.env._effective_doppler_array(
                        u,
                        np.asarray([candidate], dtype=np.int64),
                        raw_nu,
                    )
                    if abs(float(nu_eff[0])) > float(self.env.cfg.nu_max):
                        continue
                valid_set.add(candidate)
            chosen: List[int] = []
            for sat_idx in sat_action_arr[u].tolist():
                sat_idx = int(sat_idx)
                if sat_idx == -1:
                    continue
                if sat_idx < -1 or sat_idx >= int(self.env.cfg.num_sat):
                    raise ValueError(f"Selected satellite {sat_idx} is out of range for UAV {u}")
                if sat_idx not in visible_set:
                    raise ValueError(f"Selected satellite {sat_idx} is not a visible candidate for UAV {u}")
                if sat_idx in chosen:
                    raise ValueError(f"Selected satellite {sat_idx} is duplicated for UAV {u}")
                if elevation_matrix[u, sat_idx] < float(self.env.cfg.theta_min_rad):
                    raise ValueError(f"Selected satellite {sat_idx} is below elevation threshold for UAV {u}")
                if bool(getattr(self.env.cfg, "doppler_enabled", False)):
                    raw_nu = self.env._doppler_many(
                        u,
                        np.asarray([sat_idx], dtype=np.int64),
                        np.asarray(self._stage_sat_pos, dtype=np.float32),
                        np.asarray(self._stage_sat_vel, dtype=np.float32),
                    )
                    nu_eff, _ = self.env._effective_doppler_array(
                        u,
                        np.asarray([sat_idx], dtype=np.int64),
                        raw_nu,
                    )
                    if abs(float(nu_eff[0])) > float(self.env.cfg.nu_max):
                        raise ValueError(f"Selected satellite {sat_idx} violates Doppler limit for UAV {u}")
                chosen.append(sat_idx)
            if len(valid_set) > 0 and len(chosen) <= 0:
                raise ValueError(f"UAV {u} has valid SAT candidates but sat_action selected none")
            if len(chosen) > min(select_k, len(valid_set)):
                raise ValueError(f"UAV {u} selected too many SATs: {len(chosen)} > {min(select_k, len(valid_set))}")
            selections.append(chosen)
        return selections, sat_action_arr

    def _apply_sat_selection_to_spec(
        self,
        selections: Sequence[Sequence[int]],
        selection_matrix_source: List[List[int]] | np.ndarray | Sequence[Sequence[int]],
    ) -> Dict[str, Any]:
        self._set_sat_selection_in_place(selections, selection_matrix_source)
        return self._prepare_bw_stage_spec()

    def _set_sat_selection_in_place(
        self,
        selections: Sequence[Sequence[int]],
        selection_matrix_source: List[List[int]] | np.ndarray | Sequence[Sequence[int]],
    ) -> None:
        self._stage_sat_selection_matrix = self.env._sat_selection_matrix(selection_matrix_source).copy()
        self._stage_sat_selection = [list(sel) for sel in selections]
        self._stage_obs_cache = None

    def _refresh_stage_sat_obs_cache(self) -> None:
        if self._stage_sat_pos is None or self._stage_sat_vel is None or self._stage_visible is None:
            return
        env = self.env
        cfg = env.cfg
        use_stage_cache = all(
            value is not None
            for value in (
                self._stage_us_rel_pos,
                self._stage_us_rel_vel,
                self._stage_us_gain,
                self._stage_us_nu_eff,
                self._stage_us_valid_flag,
                self._stage_us_sat_queue,
            )
        )
        if not use_stage_cache:
            env._cache_sat_obs(self._stage_sat_pos, self._stage_sat_vel, self._stage_visible)
            return

        active_sat_ids = (
            np.asarray(self._stage_active_sat_ids, dtype=np.int32)
            if self._stage_active_sat_ids is not None
            else self._active_sat_ids(self._stage_visible, None)
        )
        if active_sat_ids.size <= 0:
            env._cached_sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, env.sat_dim), dtype=np.float32)
            env._cached_sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
            env._cached_sat_valid_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
            return

        sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, env.sat_dim), dtype=np.float32)
        sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        sat_valid_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        active_slot_by_sat = {int(sat_idx): slot for slot, sat_idx in enumerate(active_sat_ids.tolist())}
        current_selection = (
            self._stage_sat_selection
            if self._stage_sat_selection is not None
            else [list(sel) for sel in getattr(env, "last_sat_selection", [[] for _ in range(cfg.num_uav)])]
        )
        sat_loads = self._compute_stage_sat_selection_counts(self._stage_sat_selection)
        sat_cost_feature = None
        if bool(getattr(cfg, "obs_sat_include_sat_cost", False)):
            if self._stage_sat_nodes_static is not None and self._stage_sat_nodes_static.shape[1] == active_sat_ids.size:
                sat_cost_feature = np.asarray(self._stage_sat_nodes_static[0, :, 8], dtype=np.float32)
            else:
                sat_reward_features = env._sat_reward_aligned_feature_dict(
                    normalized=True,
                    assoc_override=self._stage_assoc,
                    sat_selection_override=self._stage_sat_selection,
                )
                sat_cost_feature = np.asarray(sat_reward_features["sat_cost"][active_sat_ids], dtype=np.float32)

        orbit_scale = normalize_scale(float(cfg.r_earth + cfg.sat_height))
        nu_scale = max(float(cfg.nu_max), 1.0)
        load_scale = max(float(cfg.num_uav), 1.0)
        effective_b_backhaul_per_sat = float(env._effective_b_backhaul_per_sat())
        rel_pos_all = np.asarray(self._stage_us_rel_pos, dtype=np.float32)
        rel_vel_all = np.asarray(self._stage_us_rel_vel, dtype=np.float32)
        gain_all = np.asarray(self._stage_us_gain, dtype=np.float32)
        nu_eff_all = np.asarray(self._stage_us_nu_eff, dtype=np.float32)
        valid_flag_all = np.asarray(self._stage_us_valid_flag, dtype=np.float32)
        sat_queue_feature_all = np.asarray(self._stage_us_sat_queue, dtype=np.float32)

        for u in range(cfg.num_uav):
            visible_ids = np.asarray(self._stage_visible[u][: cfg.sats_obs_max], dtype=np.int32)
            if visible_ids.size <= 0:
                continue
            slot_indices = np.asarray(
                [active_slot_by_sat.get(int(sat_idx), -1) for sat_idx in visible_ids.tolist()],
                dtype=np.int64,
            )
            valid_slots = slot_indices >= 0
            if not bool(np.any(valid_slots)):
                continue
            sat_idx = visible_ids[valid_slots]
            cache_slots = slot_indices[valid_slots]
            n = int(cache_slots.size)
            current = set(current_selection[u]) if u < len(current_selection) else set()
            projected_add = (~np.isin(sat_idx, np.fromiter(current, dtype=np.int32) if current else np.zeros((0,), dtype=np.int32))).astype(
                np.float32,
                copy=False,
            )
            projected_count = np.maximum(
                np.asarray(sat_loads[sat_idx], dtype=np.float32) + projected_add,
                1.0,
            ).astype(np.float32, copy=False)
            projected_bw = ratio_or_zero(effective_b_backhaul_per_sat, projected_count).astype(np.float32, copy=False)
            snr = channel.snr_linear(
                cfg.uav_tx_power,
                gain_all[u, cache_slots],
                cfg.noise_density,
                projected_bw,
                noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
            )
            if cfg.doppler_observed and cfg.doppler_atten_enabled:
                snr = snr * channel.doppler_attenuation(nu_eff_all[u, cache_slots], cfg.subcarrier_spacing)
            sat_obs[u, :n, 0:3] = rel_pos_all[u, cache_slots] / orbit_scale
            sat_obs[u, :n, 3:6] = rel_vel_all[u, cache_slots] / orbit_scale
            sat_obs[u, :n, 6] = nu_eff_all[u, cache_slots] / nu_scale
            sat_obs[u, :n, 7] = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
            sat_obs[u, :n, 8] = sat_queue_feature_all[cache_slots]
            sat_obs[u, :n, 9] = np.asarray(sat_loads[sat_idx], dtype=np.float32) / load_scale
            sat_obs[u, :n, 10] = 1.0 / projected_count
            sat_obs[u, :n, 11] = np.asarray(
                [1.0 if int(sat_value) in current else 0.0 for sat_value in sat_idx.tolist()],
                dtype=np.float32,
            )
            if sat_cost_feature is not None:
                sat_obs[u, :n, 12] = sat_cost_feature[cache_slots]
            sat_mask[u, :n] = 1.0
            sat_valid_mask[u, :n] = valid_flag_all[u, cache_slots]

        env._cached_sat_obs = sat_obs
        env._cached_sat_mask = sat_mask
        env._cached_sat_valid_mask = sat_valid_mask

    def _prepare_bw_stage_spec(self) -> Dict[str, Any]:
        if self._stage_assoc is None or self._stage_visible is None or self._stage_sat_pos is None or self._stage_sat_vel is None:
            raise RuntimeError("run_accel_stage must be called before run_sat_stage")
        if self._stage_sat_selection is None:
            raise RuntimeError("run_sat_stage must be called before build_bw_stage_snapshot")
        if self._stage_access_gain_matrix is None:
            self._refresh_world_build_cache()
        return {
            "env": self.env,
            "driver": self,
            "stage_id": self.STAGE_BW,
            "assoc": np.asarray(self._stage_assoc, dtype=np.int32),
            "candidates": [list(c) for c in self._stage_candidates or [[] for _ in range(self.env.cfg.num_uav)]],
            "sat_pos": np.asarray(self._stage_sat_pos, dtype=np.float32),
            "sat_vel": np.asarray(self._stage_sat_vel, dtype=np.float32),
            "visible": [list(v) for v in self._stage_visible],
            "sat_selection": [list(sel) for sel in self._stage_sat_selection],
            "active_sat_ids": self._active_sat_ids(self._stage_visible, self._stage_sat_selection),
            "sat_loads": self._compute_stage_sat_selection_counts(self._stage_sat_selection),
            "bw_valid_mask": None if self._stage_bw_valid_mask is None else np.asarray(self._stage_bw_valid_mask, dtype=np.float32),
            "access_gain_matrix": np.asarray(self._stage_access_gain_matrix, dtype=np.float32),
        }

    def _run_sat_stage_to_spec(self, sat_action: Sequence[Sequence[int]] | np.ndarray) -> Dict[str, Any]:
        selections, selection_matrix_source = self._sat_selections_from_action(sat_action)
        return self._apply_sat_selection_to_spec(selections, selection_matrix_source)

    def run_sat_stage(self, sat_action: Sequence[Sequence[int]] | np.ndarray) -> StructuredWorldState:
        return self._build_world_state_from_spec(self._run_sat_stage_to_spec(sat_action))

    def build_bw_valid_context(self, state_after_sat: StructuredWorldState | None = None) -> List[LocalBwState]:
        from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_spec

        del state_after_sat
        return build_local_bw_states_from_spec(self._prepare_bw_stage_spec(), device=self._tensor_device)

    def build_local_bw_states(self, state_after_sat: StructuredWorldState | None = None) -> List[LocalBwState]:
        return self.build_bw_valid_context(state_after_sat)

    def build_bw_stage_snapshot(self, state_after_sat: StructuredWorldState | None = None) -> BwStageSnapshot:
        from sagin_marl.rl.structured_stage_builders import build_batched_local_bw_states_from_spec

        spec = self._prepare_bw_stage_spec()
        ws = state_after_sat if state_after_sat is not None else self._build_world_state_from_spec(spec)
        local_state = build_batched_local_bw_states_from_spec(spec, device=self._tensor_device)
        cfg = self.env.cfg
        assoc = np.asarray(self._stage_assoc, dtype=np.int32).reshape(cfg.num_gu)
        selected = self.env._sat_selection_matrix(self._stage_sat_selection or [[] for _ in range(cfg.num_uav)])
        selected_mask = selected >= 0
        bw_valid_full = np.zeros((cfg.num_uav, cfg.num_gu), dtype=bool)
        for u in range(cfg.num_uav):
            bw_valid_full[u] = assoc == u
        if self._stage_access_gain_matrix is None:
            self._refresh_world_build_cache()
        return BwStageSnapshot(
            world_state=ws,
            assoc=self._array_to_output_device(assoc, dtype=torch.long),
            selected_sat_indices=self._array_to_output_device(selected, dtype=torch.long),
            selected_sat_mask=local_state.selected_sat_mask,
            access_gain_matrix=self._array_to_output_device(np.asarray(self._stage_access_gain_matrix, dtype=np.float32), dtype=torch.float32),
            bw_valid_mask_full=self._array_to_output_device(bw_valid_full, dtype=torch.bool),
            ego_features=local_state.ego_features,
            selected_sat_tokens=local_state.selected_sat_tokens,
            gu_tokens=local_state.gu_tokens,
            gu_mask=local_state.gu_mask,
            bw_valid_mask=local_state.bw_valid_mask,
        )

    def _export_stage_world_cache(self) -> Dict[str, Any]:
        return {
            "stage_uav_ecef_all": None if self._stage_uav_ecef_all is None else np.asarray(self._stage_uav_ecef_all, dtype=np.float32).copy(),
            "stage_uav_vel_ecef_all": None if self._stage_uav_vel_ecef_all is None else np.asarray(self._stage_uav_vel_ecef_all, dtype=np.float32).copy(),
            "stage_access_gain_matrix": None if self._stage_access_gain_matrix is None else np.asarray(self._stage_access_gain_matrix, dtype=np.float32).copy(),
            "stage_uav_gu_rel": None if self._stage_uav_gu_rel is None else np.asarray(self._stage_uav_gu_rel, dtype=np.float32).copy(),
            "stage_candidate_flag": None if self._stage_candidate_flag is None else np.asarray(self._stage_candidate_flag, dtype=np.float32).copy(),
            "stage_bw_valid_flag": None if self._stage_bw_valid_flag is None else np.asarray(self._stage_bw_valid_flag, dtype=np.float32).copy(),
            "stage_prev_assoc_flag": None if self._stage_prev_assoc_flag is None else np.asarray(self._stage_prev_assoc_flag, dtype=np.float32).copy(),
            "stage_eta_ref_feature": None if self._stage_eta_ref_feature is None else np.asarray(self._stage_eta_ref_feature, dtype=np.float32).copy(),
            "stage_uav_uav_rel_pos": None if self._stage_uav_uav_rel_pos is None else np.asarray(self._stage_uav_uav_rel_pos, dtype=np.float32).copy(),
            "stage_uav_uav_rel_vel": None if self._stage_uav_uav_rel_vel is None else np.asarray(self._stage_uav_uav_rel_vel, dtype=np.float32).copy(),
            "stage_uav_uav_dist": None if self._stage_uav_uav_dist is None else np.asarray(self._stage_uav_uav_dist, dtype=np.float32).copy(),
            "stage_active_sat_ids": None if self._stage_active_sat_ids is None else np.asarray(self._stage_active_sat_ids, dtype=np.int64).copy(),
            "stage_sat_nodes_static": None if self._stage_sat_nodes_static is None else np.asarray(self._stage_sat_nodes_static, dtype=np.float32).copy(),
            "stage_sat_mask_cached": None if self._stage_sat_mask_cached is None else np.asarray(self._stage_sat_mask_cached, dtype=bool).copy(),
            "stage_us_rel_pos": None if self._stage_us_rel_pos is None else np.asarray(self._stage_us_rel_pos, dtype=np.float32).copy(),
            "stage_us_rel_vel": None if self._stage_us_rel_vel is None else np.asarray(self._stage_us_rel_vel, dtype=np.float32).copy(),
            "stage_us_gain": None if self._stage_us_gain is None else np.asarray(self._stage_us_gain, dtype=np.float32).copy(),
            "stage_us_nu_eff": None if self._stage_us_nu_eff is None else np.asarray(self._stage_us_nu_eff, dtype=np.float32).copy(),
            "stage_us_visible_flag": None if self._stage_us_visible_flag is None else np.asarray(self._stage_us_visible_flag, dtype=np.float32).copy(),
            "stage_us_valid_flag": None if self._stage_us_valid_flag is None else np.asarray(self._stage_us_valid_flag, dtype=np.float32).copy(),
            "stage_us_sat_queue": None if self._stage_us_sat_queue is None else np.asarray(self._stage_us_sat_queue, dtype=np.float32).copy(),
        }

    def _restore_stage_world_cache(self, stage_world_cache: Dict[str, Any] | None) -> None:
        if not isinstance(stage_world_cache, dict):
            self._refresh_world_build_cache()
            return
        self._stage_uav_ecef_all = None if stage_world_cache.get("stage_uav_ecef_all") is None else np.asarray(stage_world_cache["stage_uav_ecef_all"], dtype=np.float32).copy()
        self._stage_uav_vel_ecef_all = None if stage_world_cache.get("stage_uav_vel_ecef_all") is None else np.asarray(stage_world_cache["stage_uav_vel_ecef_all"], dtype=np.float32).copy()
        self._stage_access_gain_matrix = None if stage_world_cache.get("stage_access_gain_matrix") is None else np.asarray(stage_world_cache["stage_access_gain_matrix"], dtype=np.float32).copy()
        self._stage_uav_gu_rel = None if stage_world_cache.get("stage_uav_gu_rel") is None else np.asarray(stage_world_cache["stage_uav_gu_rel"], dtype=np.float32).copy()
        self._stage_candidate_flag = None if stage_world_cache.get("stage_candidate_flag") is None else np.asarray(stage_world_cache["stage_candidate_flag"], dtype=np.float32).copy()
        self._stage_bw_valid_flag = None if stage_world_cache.get("stage_bw_valid_flag") is None else np.asarray(stage_world_cache["stage_bw_valid_flag"], dtype=np.float32).copy()
        self._stage_prev_assoc_flag = None if stage_world_cache.get("stage_prev_assoc_flag") is None else np.asarray(stage_world_cache["stage_prev_assoc_flag"], dtype=np.float32).copy()
        self._stage_eta_ref_feature = None if stage_world_cache.get("stage_eta_ref_feature") is None else np.asarray(stage_world_cache["stage_eta_ref_feature"], dtype=np.float32).copy()
        self._stage_uav_uav_rel_pos = None if stage_world_cache.get("stage_uav_uav_rel_pos") is None else np.asarray(stage_world_cache["stage_uav_uav_rel_pos"], dtype=np.float32).copy()
        self._stage_uav_uav_rel_vel = None if stage_world_cache.get("stage_uav_uav_rel_vel") is None else np.asarray(stage_world_cache["stage_uav_uav_rel_vel"], dtype=np.float32).copy()
        self._stage_uav_uav_dist = None if stage_world_cache.get("stage_uav_uav_dist") is None else np.asarray(stage_world_cache["stage_uav_uav_dist"], dtype=np.float32).copy()
        self._stage_active_sat_ids = None if stage_world_cache.get("stage_active_sat_ids") is None else np.asarray(stage_world_cache["stage_active_sat_ids"], dtype=np.int64).copy()
        self._stage_sat_nodes_static = None if stage_world_cache.get("stage_sat_nodes_static") is None else np.asarray(stage_world_cache["stage_sat_nodes_static"], dtype=np.float32).copy()
        self._stage_sat_mask_cached = None if stage_world_cache.get("stage_sat_mask_cached") is None else np.asarray(stage_world_cache["stage_sat_mask_cached"], dtype=bool).copy()
        self._stage_us_rel_pos = None if stage_world_cache.get("stage_us_rel_pos") is None else np.asarray(stage_world_cache["stage_us_rel_pos"], dtype=np.float32).copy()
        self._stage_us_rel_vel = None if stage_world_cache.get("stage_us_rel_vel") is None else np.asarray(stage_world_cache["stage_us_rel_vel"], dtype=np.float32).copy()
        self._stage_us_gain = None if stage_world_cache.get("stage_us_gain") is None else np.asarray(stage_world_cache["stage_us_gain"], dtype=np.float32).copy()
        self._stage_us_nu_eff = None if stage_world_cache.get("stage_us_nu_eff") is None else np.asarray(stage_world_cache["stage_us_nu_eff"], dtype=np.float32).copy()
        self._stage_us_visible_flag = None if stage_world_cache.get("stage_us_visible_flag") is None else np.asarray(stage_world_cache["stage_us_visible_flag"], dtype=np.float32).copy()
        self._stage_us_valid_flag = None if stage_world_cache.get("stage_us_valid_flag") is None else np.asarray(stage_world_cache["stage_us_valid_flag"], dtype=np.float32).copy()
        self._stage_us_sat_queue = None if stage_world_cache.get("stage_us_sat_queue") is None else np.asarray(stage_world_cache["stage_us_sat_queue"], dtype=np.float32).copy()

    def export_sat_stage_state(self) -> Dict[str, Any]:
        if self._stage_assoc is None or self._stage_candidates is None or self._stage_visible is None:
            raise RuntimeError("run_accel_stage must be called before export_sat_stage_state")
        stage_cached_eta = None
        cached_eta = getattr(self.env, "_cached_eta", None)
        if (
            cached_eta is not None
            and bool(getattr(self.env, "_cached_eta_matches")(self._stage_assoc, self._stage_candidates))
        ):
            stage_cached_eta = np.asarray(cached_eta, dtype=np.float32).copy()
        return {
            "env_state": self.env.export_runtime_state(),
            "step_open": bool(self._step_open),
            "stage_assoc": np.asarray(self._stage_assoc, dtype=np.int32).copy(),
            "stage_candidates": [list(c) for c in self._stage_candidates],
            "stage_bw_valid_mask": (
                None
                if self._stage_bw_valid_mask is None
                else np.asarray(self._stage_bw_valid_mask, dtype=np.float32).copy()
            ),
            "stage_visible": [list(v) for v in self._stage_visible],
            "stage_sat_pos": None if self._stage_sat_pos is None else np.asarray(self._stage_sat_pos, dtype=np.float32).copy(),
            "stage_sat_vel": None if self._stage_sat_vel is None else np.asarray(self._stage_sat_vel, dtype=np.float32).copy(),
            "stage_cached_eta": stage_cached_eta,
            "stage_world_cache": self._export_stage_world_cache(),
        }

    def export_bw_stage_state(self) -> Dict[str, Any]:
        if self._stage_assoc is None or self._stage_candidates is None or self._stage_sat_selection is None:
            raise RuntimeError("run_accel_stage and run_sat_stage must be called before export_bw_stage_state")
        stage_cached_eta = None
        cached_eta = getattr(self.env, "_cached_eta", None)
        if (
            cached_eta is not None
            and bool(getattr(self.env, "_cached_eta_matches")(self._stage_assoc, self._stage_candidates))
        ):
            stage_cached_eta = np.asarray(cached_eta, dtype=np.float32).copy()
        return {
            "env_state": self.env.export_runtime_state(),
            "step_open": bool(self._step_open),
            "stage_assoc": np.asarray(self._stage_assoc, dtype=np.int32).copy(),
            "stage_candidates": [list(c) for c in self._stage_candidates],
            "stage_bw_valid_mask": (
                None
                if self._stage_bw_valid_mask is None
                else np.asarray(self._stage_bw_valid_mask, dtype=np.float32).copy()
            ),
            "stage_visible": None if self._stage_visible is None else [list(v) for v in self._stage_visible],
            "stage_sat_pos": None if self._stage_sat_pos is None else np.asarray(self._stage_sat_pos, dtype=np.float32).copy(),
            "stage_sat_vel": None if self._stage_sat_vel is None else np.asarray(self._stage_sat_vel, dtype=np.float32).copy(),
            "stage_sat_selection": [list(sel) for sel in self._stage_sat_selection],
            "stage_cached_eta": stage_cached_eta,
            "stage_world_cache": self._export_stage_world_cache(),
        }

    def load_sat_stage_state(self, snapshot: Dict[str, Any]) -> SatStageSnapshot:
        payload = dict(snapshot or {})
        env_state = payload.get("env_state")
        if env_state is None:
            raise ValueError("snapshot missing env_state")
        self.env.load_runtime_state(
            env_state,
            refresh_observation_cache=False,
            refresh_global_state_cache=False,
        )
        self._clear_step()
        self._step_open = bool(payload.get("step_open", True))
        self._stage_assoc = np.asarray(payload["stage_assoc"], dtype=np.int32).copy()
        self._stage_candidates = [list(c) for c in payload["stage_candidates"]]
        stage_bw_valid_mask = payload.get("stage_bw_valid_mask")
        self._stage_bw_valid_mask = (
            None
            if stage_bw_valid_mask is None
            else np.asarray(stage_bw_valid_mask, dtype=np.float32).copy()
        )
        self._stage_visible = [list(v) for v in payload["stage_visible"]]
        stage_sat_pos = payload.get("stage_sat_pos")
        self._stage_sat_pos = None if stage_sat_pos is None else np.asarray(stage_sat_pos, dtype=np.float32).copy()
        stage_sat_vel = payload.get("stage_sat_vel")
        self._stage_sat_vel = None if stage_sat_vel is None else np.asarray(stage_sat_vel, dtype=np.float32).copy()
        self._stage_sat_selection = None
        env = self.env
        if self._stage_bw_valid_mask is not None:
            env._cached_bw_valid_mask = self._stage_bw_valid_mask.copy()
        self._restore_stage_world_cache(payload.get("stage_world_cache"))
        stage_cached_eta = payload.get("stage_cached_eta")
        if stage_cached_eta is not None:
            env._store_cached_eta(
                self._stage_assoc,
                self._stage_candidates,
                np.asarray(stage_cached_eta, dtype=np.float32),
                access_snapshot=self._stage_access_gain_matrix,
            )
        else:
            _, eta = env._compute_access_rates(
                self._stage_assoc,
                self._stage_candidates,
                self._zero_bw_action_matrix(),
                record_exec=False,
                access_snapshot=self._stage_access_gain_matrix,
            )
            env._store_cached_eta(
                self._stage_assoc,
                self._stage_candidates,
                eta,
                access_snapshot=self._stage_access_gain_matrix,
            )
        if self._stage_sat_pos is not None and self._stage_sat_vel is not None and self._stage_visible is not None:
            self._refresh_stage_sat_obs_cache()
        return self.build_sat_stage_snapshot()

    def load_bw_stage_state(self, snapshot: Dict[str, Any]) -> BwStageSnapshot:
        payload = dict(snapshot or {})
        env_state = payload.get("env_state")
        if env_state is None:
            raise ValueError("snapshot missing env_state")
        self.env.load_runtime_state(
            env_state,
            refresh_observation_cache=False,
            refresh_global_state_cache=False,
        )
        self._clear_step()
        self._step_open = bool(payload.get("step_open", True))
        self._stage_assoc = np.asarray(payload["stage_assoc"], dtype=np.int32).copy()
        self._stage_candidates = [list(c) for c in payload["stage_candidates"]]
        stage_bw_valid_mask = payload.get("stage_bw_valid_mask")
        self._stage_bw_valid_mask = (
            None
            if stage_bw_valid_mask is None
            else np.asarray(stage_bw_valid_mask, dtype=np.float32).copy()
        )
        stage_visible = payload.get("stage_visible")
        self._stage_visible = None if stage_visible is None else [list(v) for v in stage_visible]
        stage_sat_pos = payload.get("stage_sat_pos")
        self._stage_sat_pos = None if stage_sat_pos is None else np.asarray(stage_sat_pos, dtype=np.float32).copy()
        stage_sat_vel = payload.get("stage_sat_vel")
        self._stage_sat_vel = None if stage_sat_vel is None else np.asarray(stage_sat_vel, dtype=np.float32).copy()
        self._stage_sat_selection = [list(sel) for sel in payload["stage_sat_selection"]]
        self._stage_sat_selection_matrix = self.env._sat_selection_matrix(self._stage_sat_selection).copy()
        env = self.env
        if self._stage_bw_valid_mask is not None:
            env._cached_bw_valid_mask = self._stage_bw_valid_mask.copy()
        self._restore_stage_world_cache(payload.get("stage_world_cache"))
        stage_cached_eta = payload.get("stage_cached_eta")
        if stage_cached_eta is not None:
            env._store_cached_eta(
                self._stage_assoc,
                self._stage_candidates,
                np.asarray(stage_cached_eta, dtype=np.float32),
                access_snapshot=self._stage_access_gain_matrix,
            )
        else:
            _, eta = env._compute_access_rates(
                self._stage_assoc,
                self._stage_candidates,
                self._zero_bw_action_matrix(),
                record_exec=False,
                access_snapshot=self._stage_access_gain_matrix,
            )
            env._store_cached_eta(
                self._stage_assoc,
                self._stage_candidates,
                eta,
                access_snapshot=self._stage_access_gain_matrix,
            )
        if self._stage_sat_pos is not None and self._stage_sat_vel is not None and self._stage_visible is not None:
            self._refresh_stage_sat_obs_cache()
        return self.build_bw_stage_snapshot()

    def execute_stage_bw_and_step(
        self,
        bw_action: Sequence[np.ndarray] | np.ndarray,
        *,
        bw_proxy_base_action: Sequence[np.ndarray] | np.ndarray | None = None,
        arrival_override: np.ndarray | None = None,
        doppler_residual_after_override: np.ndarray | None = None,
        traffic_state_after_override: dict | None = None,
        bw_link_transition_override: dict[str, Any] | None = None,
        capture_auxiliary_outputs: bool = True,
        materialize_agent_dicts: bool = True,
    ) -> StructuredStepResult:
        return self._execute_stage_bw_core(
            bw_action,
            materialize_step_outputs=True,
            capture_auxiliary_outputs=capture_auxiliary_outputs,
            materialize_agent_dicts=materialize_agent_dicts,
            bw_proxy_base_action=bw_proxy_base_action,
            arrival_override=arrival_override,
            doppler_residual_after_override=doppler_residual_after_override,
            traffic_state_after_override=traffic_state_after_override,
            bw_link_transition_override=bw_link_transition_override,
        )

    def execute_stage_bw_and_prepare_next_accel(
        self,
        bw_action: Sequence[np.ndarray] | np.ndarray,
        *,
        bw_proxy_base_action: Sequence[np.ndarray] | np.ndarray | None = None,
        arrival_override: np.ndarray | None = None,
        doppler_residual_after_override: np.ndarray | None = None,
        traffic_state_after_override: dict | None = None,
        bw_link_transition_override: dict[str, Any] | None = None,
        capture_auxiliary_outputs: bool = True,
        materialize_agent_dicts: bool = True,
    ) -> tuple[StructuredStepResult, StructuredWorldState]:
        step_result = self._execute_stage_bw_core(
            bw_action,
            materialize_step_outputs=False,
            capture_next_accel_spec=True,
            capture_auxiliary_outputs=capture_auxiliary_outputs,
            materialize_agent_dicts=materialize_agent_dicts,
            bw_proxy_base_action=bw_proxy_base_action,
            arrival_override=arrival_override,
            doppler_residual_after_override=doppler_residual_after_override,
            traffic_state_after_override=traffic_state_after_override,
            bw_link_transition_override=bw_link_transition_override,
        )
        next_accel_spec = step_result.next_accel_spec
        next_world_state = self.begin_step() if next_accel_spec is None else self._build_world_state_from_spec(next_accel_spec)
        step_result.next_accel_spec = None
        return step_result, next_world_state

    @staticmethod
    def _reallocate_bw_toward_slot(
        base_action: np.ndarray,
        valid_mask: np.ndarray,
        target_slot: int,
        delta: float,
        *,
        eps: float = 1.0e-8,
    ) -> tuple[np.ndarray | None, float]:
        valid = np.asarray(valid_mask, dtype=bool)
        if not valid[int(target_slot)]:
            return None, 0.0
        donor_mask = valid.copy()
        donor_mask[int(target_slot)] = False
        donor_mass = float(np.sum(np.asarray(base_action, dtype=np.float32)[donor_mask], dtype=np.float32))
        if donor_mass <= eps:
            return None, 0.0
        used_delta = min(float(delta), 0.5 * donor_mass)
        if used_delta <= eps:
            return None, 0.0
        out = np.asarray(base_action, dtype=np.float32).copy()
        scale = float((donor_mass - used_delta) / max(donor_mass, eps))
        out[donor_mask] = out[donor_mask] * scale
        out[int(target_slot)] = float(out[int(target_slot)] + used_delta)
        out[~valid] = 0.0
        norm = float(np.sum(out[valid]))
        if norm <= eps:
            return None, 0.0
        out[valid] = out[valid] / norm
        return out.astype(np.float32, copy=False), float(used_delta)

    def _bw_flow_proxy_reward(
        self,
        bw_action: np.ndarray,
        *,
        realized_arrival: np.ndarray,
        rate_matrix: np.ndarray,
        gu_queue_before: np.ndarray,
        uav_queue_before: np.ndarray,
        sat_queue_before: np.ndarray,
    ) -> float:
        env = self.env
        cfg = env.cfg
        if self._stage_assoc is None or self._stage_candidates is None:
            raise RuntimeError("bw flow proxy reward requires accel-stage cache")
        bw_arr = np.asarray(bw_action, dtype=np.float32)
        access_rates, _ = env._compute_access_rates(
            self._stage_assoc,
            self._stage_candidates,
            bw_arr,
            record_exec=False,
            access_snapshot=self._stage_access_gain_matrix,
        )
        arrival = np.asarray(realized_arrival, dtype=np.float32)
        q_gu_before = np.asarray(gu_queue_before, dtype=np.float32) + arrival
        gu_outflow = np.minimum(q_gu_before, np.asarray(access_rates, dtype=np.float32) * float(cfg.tau0)).astype(
            np.float32
        )
        q_gu_after_raw = q_gu_before - gu_outflow
        gu_drop = np.maximum(q_gu_after_raw - float(cfg.queue_max_gu), 0.0).astype(np.float32)
        q_gu_after = np.minimum(q_gu_after_raw, float(cfg.queue_max_gu)).astype(np.float32)

        assoc = np.asarray(self._stage_assoc, dtype=np.int32)
        valid_assoc = assoc >= 0
        if np.any(valid_assoc):
            inflow_uav = np.zeros((int(cfg.num_uav),), dtype=np.float32)
            np.add.at(
                inflow_uav,
                assoc[valid_assoc].astype(np.int64, copy=False),
                np.asarray(gu_outflow[valid_assoc], dtype=np.float32),
            )
        else:
            inflow_uav = np.zeros((cfg.num_uav,), dtype=np.float32)

        q_uav_before = np.asarray(uav_queue_before, dtype=np.float32) + inflow_uav
        total_rate = np.sum(np.asarray(rate_matrix, dtype=np.float32), axis=1).astype(np.float32)
        uav_outflow = np.minimum(q_uav_before, total_rate * float(cfg.tau0)).astype(np.float32)
        q_uav_after_raw = q_uav_before - uav_outflow
        uav_drop = np.maximum(q_uav_after_raw - float(cfg.queue_max_uav), 0.0).astype(np.float32)
        q_uav_after = np.minimum(q_uav_after_raw, float(cfg.queue_max_uav)).astype(np.float32)

        outflow_matrix = np.zeros_like(rate_matrix, dtype=np.float32)
        active_uav = total_rate > 0.0
        if np.any(active_uav):
            outflow_matrix[active_uav] = (
                np.asarray(rate_matrix, dtype=np.float32)[active_uav] / total_rate[active_uav, None]
            ) * uav_outflow[active_uav, None]
        sat_incoming = np.sum(outflow_matrix, axis=0).astype(np.float32)
        compute_rate = float(env._effective_sat_cpu_freq()) / normalize_scale(float(cfg.task_cycles_per_bit))
        q_sat_before = np.asarray(sat_queue_before, dtype=np.float32) + sat_incoming
        sat_processed = np.minimum(q_sat_before, compute_rate * float(cfg.tau0)).astype(np.float32)

        q_sat_after_raw = q_sat_before - sat_processed
        sat_drop = np.maximum(q_sat_after_raw - float(cfg.queue_max_sat), 0.0).astype(np.float32)
        q_sat_after = np.minimum(q_sat_after_raw, float(cfg.queue_max_sat)).astype(np.float32)

        reward_mode = str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower()
        if reward_mode == "controllable_flow":
            arrival_ref = float(env._arrival_ref())
            x_acc = float(np.sum(gu_outflow) / arrival_ref)
            x_rel = float(np.sum(sat_incoming) / arrival_ref)
            d_pre = float((np.sum(gu_drop) + np.sum(uav_drop)) / arrival_ref)
            b_pre_steps = float((np.sum(q_gu_after) + np.sum(q_uav_after)) / arrival_ref)
            reward_proxy = (
                float(getattr(cfg, "reward_w_access", 0.5) or 0.0) * x_acc
                + float(getattr(cfg, "reward_w_relay", 0.5) or 0.0) * x_rel
                - float(getattr(cfg, "reward_w_pre_drop", 1.0) or 0.0) * d_pre
                - float(getattr(cfg, "reward_w_pre_backlog", 0.08) or 0.0) * float(np.log1p(b_pre_steps))
            )
            return float(reward_proxy)
        if reward_mode in {"weighted_workload_level", "weighted_workload_delta", "relative_weighted_workload_delta"}:
            gu_cost, uav_cost, sat_cost = self._bw_weighted_workload_device_costs()
            workload_after = self._bw_weighted_workload_total(
                gu_cost=gu_cost,
                uav_cost=uav_cost,
                sat_cost=sat_cost,
                gu_queue=q_gu_after,
                uav_queue=q_uav_after,
                sat_queue=q_sat_after,
            )
            drop_cost = self._bw_weighted_workload_total(
                gu_cost=gu_cost,
                uav_cost=uav_cost,
                sat_cost=sat_cost,
                gu_queue=gu_drop,
                uav_queue=uav_drop,
                sat_queue=sat_drop,
            )
            if reward_mode == "weighted_workload_level":
                return float(-workload_after - drop_cost)
            q_gu_before_service = np.asarray(gu_queue_before, dtype=np.float32) + np.asarray(realized_arrival, dtype=np.float32)
            workload_before = self._bw_weighted_workload_total(
                gu_cost=gu_cost,
                uav_cost=uav_cost,
                sat_cost=sat_cost,
                gu_queue=q_gu_before_service,
                uav_queue=np.asarray(uav_queue_before, dtype=np.float32),
                sat_queue=np.asarray(sat_queue_before, dtype=np.float32),
            )
            weighted_delta = float(-(workload_after - workload_before) - drop_cost)
            if reward_mode == "relative_weighted_workload_delta":
                return float(weighted_delta / max(float(workload_before), 1.0))
            return weighted_delta
        raise NotImplementedError(
            f"bw flow proxy reward currently does not support reward_mode={reward_mode!r}."
        )

    def _bw_weighted_workload_eps(self) -> float:
        cfg = self.env.cfg
        return max(float(getattr(cfg, "bw_weighted_workload_eps", 1.0) or 0.0), 1.0)

    def _bw_weighted_workload_device_ema_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env = self.env
        cfg = env.cfg

        def _coerce_vec(attr_vec: str, attr_scalar: str, size: int, default_value: float) -> np.ndarray:
            if size <= 0:
                return np.zeros((0,), dtype=np.float32)
            vec = getattr(env, attr_vec, None)
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.shape == (size,):
                    return arr.astype(np.float32, copy=False)
            scalar = getattr(env, attr_scalar, None)
            if scalar is not None:
                per_entity = float(scalar) / max(float(size), 1.0)
                return np.full((size,), per_entity, dtype=np.float32)
            return np.full((size,), default_value, dtype=np.float32)

        arrival_ref = float(env._arrival_ref())
        gu_default = arrival_ref / max(float(cfg.num_gu), 1.0)
        uav_default = arrival_ref / max(float(cfg.num_uav), 1.0)
        sat_active_ref = (
            float(env._bw_weighted_workload_sat_active_ref_count())
            if hasattr(env, "_bw_weighted_workload_sat_active_ref_count")
            else max(float(cfg.num_sat), 1.0)
        )
        sat_default = arrival_ref / max(sat_active_ref, 1.0)
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

    def _bw_weighted_workload_device_costs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env = self.env
        cfg = env.cfg
        eps = self._bw_weighted_workload_eps()
        gu_ema, uav_ema, sat_ema = self._bw_weighted_workload_device_ema_vectors()

        sat_cost = (1.0 / np.maximum(sat_ema, eps)).astype(np.float32, copy=False)
        sat_cost_fallback = float(np.mean(sat_cost)) if sat_cost.size > 0 else 0.0

        sat_selection = self._stage_sat_selection if self._stage_sat_selection is not None else getattr(env, "last_sat_selection", None)
        uav_downstream_cost = np.full((cfg.num_uav,), sat_cost_fallback, dtype=np.float32)
        if isinstance(sat_selection, list):
            for u in range(min(len(sat_selection), cfg.num_uav)):
                selected = np.asarray(sat_selection[u], dtype=np.int64).reshape(-1)
                valid = selected[(selected >= 0) & (selected < cfg.num_sat)]
                if valid.size > 0:
                    uav_downstream_cost[u] = float(np.mean(sat_cost[valid]))
        uav_cost = (1.0 / np.maximum(uav_ema, eps) + uav_downstream_cost).astype(np.float32, copy=False)
        uav_cost_fallback = float(np.mean(uav_cost)) if uav_cost.size > 0 else 0.0

        assoc_source = self._stage_assoc if self._stage_assoc is not None else getattr(
            env,
            "last_association",
            np.full((cfg.num_gu,), -1, dtype=np.int32),
        )
        assoc = np.asarray(assoc_source, dtype=np.int32).reshape(-1)
        gu_downstream_cost = np.full((cfg.num_gu,), uav_cost_fallback, dtype=np.float32)
        if assoc.shape == (cfg.num_gu,) and cfg.num_uav > 0:
            valid_assoc = (assoc >= 0) & (assoc < cfg.num_uav)
            if np.any(valid_assoc):
                gu_downstream_cost[valid_assoc] = uav_cost[assoc[valid_assoc]]
        gu_cost = (1.0 / np.maximum(gu_ema, eps) + gu_downstream_cost).astype(np.float32, copy=False)
        return gu_cost, uav_cost, sat_cost

    def _bw_weighted_workload_total(
        self,
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

    def _bw_weighted_workload_reward(
        self,
        *,
        gu_queue_before: np.ndarray,
        uav_queue_before: np.ndarray,
        sat_queue_before: np.ndarray,
        realized_arrival: np.ndarray,
    ) -> float:
        return float(
            self.env._reward_weighted_workload_delta(
                gu_queue_before=np.asarray(gu_queue_before, dtype=np.float32),
                uav_queue_before=np.asarray(uav_queue_before, dtype=np.float32),
                sat_queue_before=np.asarray(sat_queue_before, dtype=np.float32),
                realized_arrival=np.asarray(realized_arrival, dtype=np.float32),
            )
        )

    def _bw_weighted_workload_level_reward(self) -> float:
        return float(self.env._reward_weighted_workload_level())

    def _bw_gu_queue_level_reward(self) -> float:
        return float(self.env._reward_gu_queue_level())

    def _bw_system_queue_level_reward(self) -> float:
        return float(self.env._reward_system_queue_level())

    def _bw_gu_service_queue_reward(self) -> float:
        return float(self.env._reward_gu_service_queue())

    def _update_bw_weighted_workload_ema(self) -> None:
        env = self.env
        cfg = env.cfg
        decay = float(np.clip(float(getattr(cfg, "bw_weighted_workload_ema_decay", 0.95) or 0.0), 0.0, 1.0))
        keep = decay
        fresh = 1.0 - decay
        gu_ema_prev, uav_ema_prev, sat_ema_prev = self._bw_weighted_workload_device_ema_vectors()
        acc_outflow = np.asarray(env.last_gu_outflow, dtype=np.float32).reshape(-1)
        rel_outflow = np.asarray(
            getattr(env, "last_uav_outflow", np.zeros((cfg.num_uav,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        sat_outflow = np.asarray(env.last_sat_processed, dtype=np.float32).reshape(-1)
        env.bw_weighted_workload_acc_ema_vec = (
            keep * gu_ema_prev + fresh * acc_outflow
        ).astype(np.float32, copy=False)
        env.bw_weighted_workload_rel_ema_vec = (
            keep * uav_ema_prev + fresh * rel_outflow
        ).astype(np.float32, copy=False)
        env.bw_weighted_workload_sat_ema_vec = (
            keep * sat_ema_prev + fresh * sat_outflow
        ).astype(np.float32, copy=False)
        env.bw_weighted_workload_acc_ema = float(np.sum(env.bw_weighted_workload_acc_ema_vec))
        env.bw_weighted_workload_rel_ema = float(np.sum(env.bw_weighted_workload_rel_ema_vec))
        env.bw_weighted_workload_sat_ema = float(np.sum(env.bw_weighted_workload_sat_ema_vec))

    def _compute_bw_flow_proxy_scores(
        self,
        bw_proxy_base_action: np.ndarray,
        *,
        realized_arrival: np.ndarray,
        rate_matrix: np.ndarray,
        gu_queue_before: np.ndarray,
        uav_queue_before: np.ndarray,
        sat_queue_before: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        cfg = self.env.cfg
        if not (
            bool(getattr(cfg, "bw_flow_proxy_aux_enabled", False))
            or bool(getattr(cfg, "bw_counterfactual_credit_enabled", False))
            or bool(getattr(cfg, "bw_marginal_teacher_sample_enabled", False))
            or bool(getattr(cfg, "structured_bw_per_slot_surrogate_enabled", False))
        ):
            return None, None, None
        if self._stage_assoc is None:
            raise RuntimeError("bw flow proxy aux requires BW stage association cache")

        base_action = np.asarray(bw_proxy_base_action, dtype=np.float32)
        base_proxy_reward = self._bw_flow_proxy_reward(
            base_action,
            realized_arrival=np.asarray(realized_arrival, dtype=np.float32),
            rate_matrix=np.asarray(rate_matrix, dtype=np.float32),
            gu_queue_before=np.asarray(gu_queue_before, dtype=np.float32),
            uav_queue_before=np.asarray(uav_queue_before, dtype=np.float32),
            sat_queue_before=np.asarray(sat_queue_before, dtype=np.float32),
        )
        proxy_scores = np.zeros_like(base_action, dtype=np.float32)
        proxy_mask = np.zeros_like(base_action, dtype=np.float32)
        proxy_deltas = np.zeros_like(base_action, dtype=np.float32)
        valid_slots_all = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=bool)
        assoc = np.asarray(self._stage_assoc, dtype=np.int32).reshape(int(cfg.num_gu))
        for uav_idx in range(int(cfg.num_uav)):
            valid_slots_all[uav_idx] = assoc == uav_idx
        delta = max(float(getattr(cfg, "bw_flow_proxy_aux_delta", 0.05) or 0.0), 0.0)

        for u in range(int(cfg.num_uav)):
            valid_slots = np.asarray(valid_slots_all[u], dtype=bool)
            if int(np.sum(valid_slots)) <= 1:
                continue
            base_action_u = np.asarray(base_action[u], dtype=np.float32)
            target_slots = np.flatnonzero(valid_slots)
            for target_slot in target_slots.tolist():
                cf_action_u, used_delta = self._reallocate_bw_toward_slot(
                    base_action_u,
                    valid_slots,
                    int(target_slot),
                    delta,
                )
                if cf_action_u is None:
                    continue
                cf_action = np.asarray(base_action, dtype=np.float32).copy()
                cf_action[u] = cf_action_u
                cf_proxy_reward = self._bw_flow_proxy_reward(
                    cf_action,
                    realized_arrival=np.asarray(realized_arrival, dtype=np.float32),
                    rate_matrix=np.asarray(rate_matrix, dtype=np.float32),
                    gu_queue_before=np.asarray(gu_queue_before, dtype=np.float32),
                    uav_queue_before=np.asarray(uav_queue_before, dtype=np.float32),
                    sat_queue_before=np.asarray(sat_queue_before, dtype=np.float32),
                )
                proxy_scores[u, int(target_slot)] = float(
                    (cf_proxy_reward - base_proxy_reward) / max(float(used_delta), 1.0e-8)
                )
                proxy_mask[u, int(target_slot)] = 1.0
                proxy_deltas[u, int(target_slot)] = float(used_delta)
        return proxy_scores, proxy_mask, proxy_deltas

    def _execute_stage_bw_core(
        self,
        bw_action: Sequence[np.ndarray] | np.ndarray,
        *,
        materialize_step_outputs: bool,
        capture_next_accel_spec: bool = False,
        capture_auxiliary_outputs: bool = True,
        materialize_agent_dicts: bool = True,
        bw_proxy_base_action: Sequence[np.ndarray] | np.ndarray | None = None,
        arrival_override: np.ndarray | None = None,
        doppler_residual_after_override: np.ndarray | None = None,
        traffic_state_after_override: dict | None = None,
        bw_link_transition_override: dict[str, Any] | None = None,
    ) -> StructuredStepResult:
        if self._stage_assoc is None or self._stage_candidates is None or self._stage_sat_selection is None:
            raise RuntimeError("run_accel_stage and run_sat_stage must be called before execute_stage_bw_and_step")
        env = self.env
        cfg = env.cfg
        bw_arr = np.asarray(bw_action, dtype=np.float32)
        if bw_arr.shape != (cfg.num_uav, cfg.num_gu):
            raise ValueError(f"bw_action shape must be ({cfg.num_uav}, {cfg.num_gu})")
        if bw_proxy_base_action is None:
            proxy_base_arr = bw_arr
        else:
            proxy_base_arr = np.asarray(bw_proxy_base_action, dtype=np.float32)
            if proxy_base_arr.shape != (cfg.num_uav, cfg.num_gu):
                raise ValueError(f"bw_proxy_base_action shape must be ({cfg.num_uav}, {cfg.num_gu})")
        sat_pos = np.asarray(self._stage_sat_pos, dtype=np.float32)
        sat_vel = np.asarray(self._stage_sat_vel, dtype=np.float32)
        sat_selection_matrix = (
            np.asarray(self._stage_sat_selection_matrix, dtype=np.int64)
            if self._stage_sat_selection_matrix is not None
            else self.env._sat_selection_matrix(self._stage_sat_selection)
        )
        transition = env._apply_bw_transition_core(
            self._stage_assoc,
            self._stage_candidates,
            bw_arr,
            sat_selection_matrix,
            sat_pos,
            sat_vel,
            access_snapshot=self._stage_access_gain_matrix,
            arrival_override=arrival_override,
            traffic_state_after_override=traffic_state_after_override,
            bw_link_transition_override=bw_link_transition_override,
        )
        return self._finalize_bw_step_result_after_transition(
            transition,
            sat_pos=sat_pos,
            sat_vel=sat_vel,
            proxy_base_arr=proxy_base_arr,
            materialize_step_outputs=materialize_step_outputs,
            capture_next_accel_spec=capture_next_accel_spec,
            capture_auxiliary_outputs=capture_auxiliary_outputs,
            materialize_agent_dicts=materialize_agent_dicts,
            doppler_residual_after_override=doppler_residual_after_override,
        )

    def _finalize_bw_step_result_after_transition(
        self,
        transition,
        *,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        proxy_base_arr: np.ndarray,
        materialize_step_outputs: bool,
        capture_next_accel_spec: bool = False,
        capture_auxiliary_outputs: bool = True,
        materialize_agent_dicts: bool = True,
        step_status_override=None,
        bw_flow_proxy_override=None,
        danger_imitation_override=None,
        skip_workload_ema_update: bool = False,
        doppler_residual_after_override: np.ndarray | None = None,
    ) -> StructuredStepResult:
        env = self.env
        if capture_auxiliary_outputs:
            if bw_flow_proxy_override is None:
                bw_flow_proxy_scores, bw_flow_proxy_mask, bw_flow_proxy_deltas = self._compute_bw_flow_proxy_scores(
                    proxy_base_arr,
                    realized_arrival=np.asarray(transition.realized_arrival, dtype=np.float32),
                    rate_matrix=np.asarray(transition.rate_matrix, dtype=np.float32),
                    gu_queue_before=np.asarray(transition.gu_queue_before, dtype=np.float32),
                    uav_queue_before=np.asarray(transition.uav_queue_before, dtype=np.float32),
                    sat_queue_before=np.asarray(transition.sat_queue_before, dtype=np.float32),
                )
            else:
                bw_flow_proxy_scores, bw_flow_proxy_mask, bw_flow_proxy_deltas = bw_flow_proxy_override
        else:
            bw_flow_proxy_scores = None
            bw_flow_proxy_mask = None
            bw_flow_proxy_deltas = None
        env._prepare_next_step_stage_context(
            sat_pos,
            sat_vel,
            advance_doppler=True,
            refresh_sat_obs=materialize_step_outputs,
            doppler_residual_after_override=doppler_residual_after_override,
        )
        if not skip_workload_ema_update:
            self._update_bw_weighted_workload_ema()
        step_status = env._finalize_post_bw_step() if step_status_override is None else step_status_override
        reward = float(step_status.reward)
        reward_parts = dict(step_status.reward_parts)
        bw_access_reward = float(reward_parts.get("x_acc", 0.0) or 0.0)
        if "bw_weighted_workload_delta_reward" in reward_parts:
            bw_weighted_workload_delta_reward = float(reward_parts["bw_weighted_workload_delta_reward"] or 0.0)
        else:
            bw_weighted_workload_delta_reward = float(
                self._bw_weighted_workload_reward(
                    gu_queue_before=np.asarray(transition.gu_queue_before, dtype=np.float32),
                    uav_queue_before=np.asarray(transition.uav_queue_before, dtype=np.float32),
                    sat_queue_before=np.asarray(transition.sat_queue_before, dtype=np.float32),
                    realized_arrival=np.asarray(transition.realized_arrival, dtype=np.float32),
                )
                or 0.0
            )
        if "bw_weighted_workload_level_reward" in reward_parts:
            bw_weighted_workload_level_reward = float(reward_parts["bw_weighted_workload_level_reward"] or 0.0)
        else:
            bw_weighted_workload_level_reward = float(self._bw_weighted_workload_level_reward() or 0.0)
        if "bw_gu_queue_level_reward" in reward_parts:
            bw_gu_queue_level_reward = float(reward_parts["bw_gu_queue_level_reward"] or 0.0)
        else:
            bw_gu_queue_level_reward = float(self._bw_gu_queue_level_reward() or 0.0)
        if "bw_system_queue_level_reward" in reward_parts:
            bw_system_queue_level_reward = float(reward_parts["bw_system_queue_level_reward"] or 0.0)
        else:
            bw_system_queue_level_reward = float(self._bw_system_queue_level_reward() or 0.0)
        if "bw_gu_service_queue_reward" in reward_parts:
            bw_gu_service_queue_reward = float(reward_parts["bw_gu_service_queue_reward"] or 0.0)
        else:
            bw_gu_service_queue_reward = float(self._bw_gu_service_queue_reward() or 0.0)
        terminated = bool(step_status.terminated)
        truncated = bool(step_status.truncated)
        next_accel_spec = self._prepare_accel_stage_spec() if capture_next_accel_spec else None
        step_outputs = env._materialize_post_step_outputs(
            step_status,
            materialize_step_outputs=materialize_step_outputs,
            materialize_agent_dicts=materialize_agent_dicts,
            refresh_global_state_cache=materialize_step_outputs,
        )
        if capture_auxiliary_outputs:
            if danger_imitation_override is None:
                danger_imitation_target, danger_imitation_mask = self._build_danger_imitation_step_data()
            else:
                danger_imitation_target, danger_imitation_mask = danger_imitation_override
        else:
            danger_imitation_target = None
            danger_imitation_mask = None
        self._clear_step()
        return StructuredStepResult(
            obs=step_outputs.obs,
            rewards=step_outputs.rewards,
            terminations=step_outputs.terminations,
            truncations=step_outputs.truncations,
            infos=step_outputs.infos,
            danger_imitation_target=danger_imitation_target,
            danger_imitation_mask=danger_imitation_mask,
            team_reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            reward_parts=reward_parts,
            bw_access_reward=bw_access_reward,
            bw_weighted_workload_delta_reward=bw_weighted_workload_delta_reward,
            bw_weighted_workload_level_reward=bw_weighted_workload_level_reward,
            bw_gu_queue_level_reward=bw_gu_queue_level_reward,
            bw_system_queue_level_reward=bw_system_queue_level_reward,
            bw_gu_service_queue_reward=bw_gu_service_queue_reward,
            bw_flow_proxy_scores=bw_flow_proxy_scores,
            bw_flow_proxy_mask=bw_flow_proxy_mask,
            bw_flow_proxy_deltas=bw_flow_proxy_deltas,
            next_accel_spec=next_accel_spec,
        )

    def _dummy_action_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        action_dict = self.env._dummy_actions()
        return {agent: {key: np.asarray(value).copy() for key, value in data.items()} for agent, data in action_dict.items()}

    def _active_sat_ids(self, visible: List[List[int]], sat_selection: List[List[int]] | None) -> np.ndarray:
        active: List[int] = []
        for vis in visible:
            active.extend(int(idx) for idx in vis)
        if sat_selection is not None:
            for sel in sat_selection:
                active.extend(int(idx) for idx in sel)
        if not active:
            return np.zeros((0,), dtype=np.int32)
        return np.asarray(sorted(set(active)), dtype=np.int32)

    def _compute_stage_sat_selection_counts(self, sat_selection: List[List[int]] | None) -> np.ndarray:
        counts = np.asarray(self.env.last_sat_connection_counts, dtype=np.float32).copy()
        if sat_selection is None:
            return counts
        for u, selected in enumerate(sat_selection):
            current = set(self.env.last_sat_selection[u]) if u < len(self.env.last_sat_selection) else set()
            for sat_idx in selected:
                if sat_idx not in current:
                    counts[int(sat_idx)] += 1.0
        return counts

    def _build_danger_imitation_step_data(self) -> tuple[np.ndarray, np.ndarray]:
        env = self.env
        cfg = env.cfg
        accel_exec = np.asarray(
            getattr(env, "last_exec_accel", np.zeros((cfg.num_uav, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        accel_exec_norm = _project_normalized_accel_np(accel_exec / max(float(cfg.a_max), 1e-6))
        mask_uav = np.asarray(
            getattr(env, "last_danger_imitation_mask", np.zeros((cfg.num_uav,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(cfg.num_uav)
        mask_uav = (mask_uav > 0.5).astype(np.float32, copy=False)
        mask = np.repeat(mask_uav[:, None], 2, axis=1).astype(np.float32, copy=False)
        return accel_exec_norm, mask

    @staticmethod
    def _positive_cfg_scale(value: float, *, name: str) -> float:
        value_f = float(value)
        if value_f <= 0.0:
            raise ValueError(f"{name} must be positive for critic world normalization, got {value_f}.")
        return value_f

    @staticmethod
    def _log_ratio_np(value: np.ndarray, ref: float) -> np.ndarray:
        ref_f = max(float(ref), float(NORMALIZATION_DENOM_EPS))
        return np.log(np.maximum(np.asarray(value, dtype=np.float32), float(NORMALIZATION_DENOM_EPS)) / ref_f).astype(np.float32)

    @staticmethod
    def _log1p_nonnegative_np(value: np.ndarray | float) -> np.ndarray:
        return np.log1p(np.maximum(np.asarray(value, dtype=np.float32), 0.0)).astype(np.float32)

    def _critic_sat_ids_from_visible(self, visible: List[List[int]], sat_selection: List[List[int]] | None, stage_id: int) -> np.ndarray:
        cfg = self.env.cfg
        token_max = critic_schema.critic_sat_token_max_from_cfg(cfg)
        seen: set[int] = set()
        sat_ids: list[int] = []
        for vis in visible:
            for sat_idx_raw in vis:
                sat_idx = int(sat_idx_raw)
                if sat_idx < 0 or sat_idx >= int(cfg.num_sat) or sat_idx in seen:
                    continue
                seen.add(sat_idx)
                sat_ids.append(sat_idx)
        if len(sat_ids) > token_max:
            raise RuntimeError(
                "critic SAT candidate union exceeds critic_sat_token_max; "
                "visible_sats_max/sats_obs_max helper is inconsistent with _stage_visible."
            )
        if int(stage_id) == self.STAGE_BW and sat_selection is not None:
            missing: list[int] = []
            for selected in sat_selection:
                for sat_idx_raw in selected:
                    sat_idx = int(sat_idx_raw)
                    if sat_idx >= 0 and sat_idx < int(cfg.num_sat) and sat_idx not in seen:
                        missing.append(sat_idx)
            if missing:
                raise RuntimeError(
                    f"BW prefix selected SATs are not present in critic_sat_ids: {sorted(set(missing))}."
                )
        out = np.full((token_max,), -1, dtype=np.int64)
        if sat_ids:
            out[: len(sat_ids)] = np.asarray(sat_ids, dtype=np.int64)
        return out

    def _build_critic_world_state(
        self,
        stage_id: int,
        assoc: np.ndarray,
        candidates: List[List[int]],
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        visible: List[List[int]],
        sat_selection: List[List[int]] | None,
    ) -> StructuredWorldState:
        env = self.env
        cfg = env.cfg
        num_uav = int(cfg.num_uav)
        num_gu = int(cfg.num_gu)
        num_sat = int(cfg.num_sat)
        assoc = np.asarray(assoc, dtype=np.int32).reshape(num_gu)

        arrival_ref = self._positive_cfg_scale(env._arrival_ref(), name="arrival_ref_bits_per_step")
        gu_flow_ref = self._positive_cfg_scale(arrival_ref / float(num_gu), name="gu_flow_ref") if num_gu > 0 else 1.0
        uav_flow_ref = self._positive_cfg_scale(arrival_ref / float(num_uav), name="uav_flow_ref") if num_uav > 0 else 1.0
        sat_ref_count = self._positive_cfg_scale(env._bw_weighted_workload_sat_active_ref_count(), name="sat_workload_ref_count")
        sat_flow_ref = self._positive_cfg_scale(arrival_ref / sat_ref_count, name="sat_flow_ref")
        map_scale = self._positive_cfg_scale(float(cfg.map_size), name="map_size")
        v_scale = self._positive_cfg_scale(float(cfg.v_max), name="v_max")
        energy_scale = self._positive_cfg_scale(float(cfg.uav_energy_init), name="uav_energy_init")
        gu_queue_scale = self._positive_cfg_scale(float(cfg.queue_max_gu), name="queue_max_gu")
        uav_queue_scale = self._positive_cfg_scale(float(cfg.queue_max_uav), name="queue_max_uav")
        sat_queue_scale = self._positive_cfg_scale(float(cfg.queue_max_sat), name="queue_max_sat")
        orbit_pos_ref = self._positive_cfg_scale(float(cfg.r_earth + cfg.sat_height), name="orbit_pos_ref")
        sat_speed_ref = self._positive_cfg_scale(np.sqrt(3.986004418e14 / orbit_pos_ref), name="sat_speed_ref")
        service_floor = self._positive_cfg_scale(env._bw_weighted_workload_eps(), name="service_floor_bits_per_step")

        gu_flow_cost_ref = 1.0 / gu_flow_ref
        uav_flow_cost_ref = 1.0 / uav_flow_ref
        sat_cost_ref = 1.0 / sat_flow_ref
        uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref
        gu_total_cost_ref = gu_flow_cost_ref + uav_total_cost_ref
        gu_ema, uav_ema, sat_ema = env._bw_weighted_workload_device_ema_vectors()
        sat_cost = (1.0 / np.maximum(sat_ema, service_floor)).astype(np.float32, copy=False)
        local_gu_cost = (1.0 / np.maximum(gu_ema, service_floor)).astype(np.float32, copy=False)
        local_uav_cost = (1.0 / np.maximum(uav_ema, service_floor)).astype(np.float32, copy=False)
        last_assoc = np.asarray(getattr(env, "last_association", np.full((num_gu,), -1, dtype=np.int32)), dtype=np.int32)
        last_gu_cost, last_uav_cost, last_sat_cost = env._bw_weighted_workload_device_costs(
            assoc_override=last_assoc,
            sat_selection_override=getattr(env, "last_sat_selection", None),
        )
        prefix_known = int(stage_id) == self.STAGE_BW and sat_selection is not None
        if prefix_known:
            prefix_gu_cost, prefix_uav_cost, prefix_sat_cost = env._bw_weighted_workload_device_costs(
                assoc_override=assoc,
                sat_selection_override=sat_selection,
            )
        else:
            prefix_gu_cost = np.zeros((num_gu,), dtype=np.float32)
            prefix_uav_cost = np.zeros((num_uav,), dtype=np.float32)
            prefix_sat_cost = sat_cost.astype(np.float32, copy=False)

        sat_ids = self._critic_sat_ids_from_visible(visible, sat_selection, int(stage_id))
        sat_mask_vec = sat_ids >= 0
        safe_sat_ids = np.clip(sat_ids, 0, max(num_sat - 1, 0))
        sat_token_count = int(sat_ids.shape[0])
        sat_id_set = {int(s) for s in sat_ids[sat_mask_vec].tolist()}

        gu_nodes = np.zeros((1, num_gu, critic_schema.CRITIC_GU_NODE_DIM), dtype=np.float32)
        if num_gu > 0:
            gu_queue = np.asarray(env.gu_queue, dtype=np.float32)
            gu_expected = np.asarray(env._current_expected_gu_arrival_rates(), dtype=np.float32) * float(cfg.tau0)
            gu_last_arrival = np.asarray(getattr(env, "last_gu_arrival", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32)
            gu_last_outflow = np.asarray(getattr(env, "last_gu_outflow", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32)
            gu_drop = np.asarray(getattr(env, "gu_drop", np.zeros((num_gu,), dtype=np.float32)), dtype=np.float32)
            gu_nodes[0, :, critic_schema.GU_X : critic_schema.GU_Y + 1] = np.asarray(env.gu_pos, dtype=np.float32) / map_scale
            gu_nodes[0, :, critic_schema.GU_QUEUE_STEPS] = self._log1p_nonnegative_np(gu_queue / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_QUEUE_FILL] = gu_queue / gu_queue_scale
            gu_nodes[0, :, critic_schema.GU_EXPECTED_ARRIVAL_STEPS] = self._log1p_nonnegative_np(gu_expected / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_LAST_ARRIVAL_STEPS] = self._log1p_nonnegative_np(gu_last_arrival / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_LAST_OUTFLOW_STEPS] = self._log1p_nonnegative_np(gu_last_outflow / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_LAST_DROP_STEPS] = self._log1p_nonnegative_np(gu_drop / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_SERVICE_EMA_STEPS] = self._log1p_nonnegative_np(gu_ema / gu_flow_ref)
            gu_nodes[0, :, critic_schema.GU_LOCAL_COST_LOG_RATIO] = self._log_ratio_np(local_gu_cost, gu_flow_cost_ref)
            gu_nodes[0, :, critic_schema.GU_LAST_TOTAL_COST_LOG_RATIO] = self._log_ratio_np(last_gu_cost, gu_total_cost_ref)
            gu_nodes[0, :, critic_schema.GU_LAST_WORKLOAD_LOG1P] = np.log1p(np.maximum(last_gu_cost * gu_queue, 0.0)).astype(np.float32)
            if prefix_known:
                gu_nodes[0, :, critic_schema.GU_PREFIX_TOTAL_COST_LOG_RATIO] = self._log_ratio_np(prefix_gu_cost, gu_total_cost_ref)
                gu_nodes[0, :, critic_schema.GU_PREFIX_COST_KNOWN] = 1.0
                gu_nodes[0, :, critic_schema.GU_PREFIX_WORKLOAD_LOG1P] = np.log1p(np.maximum(prefix_gu_cost * gu_queue, 0.0)).astype(np.float32)

        uav_nodes = np.zeros((1, num_uav, critic_schema.CRITIC_UAV_NODE_DIM), dtype=np.float32)
        uav_queue = np.asarray(env.uav_queue, dtype=np.float32)
        uav_last_inflow = np.asarray(getattr(env, "last_gu_to_uav_inflow_by_uav", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32)
        uav_last_outflow = np.asarray(getattr(env, "last_uav_outflow", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32)
        uav_drop = np.asarray(getattr(env, "uav_drop", np.zeros((num_uav,), dtype=np.float32)), dtype=np.float32)
        access_noise_full_band = (
            self._positive_cfg_scale(float(cfg.noise_density), name="noise_density")
            * self._positive_cfg_scale(float(cfg.b_acc), name="b_acc")
            * (10.0 ** (float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0) / 10.0))
        )
        last_interference = np.asarray(
            getattr(env, "last_access_interference_by_uav", np.zeros((num_uav,), dtype=np.float32)),
            dtype=np.float32,
        )
        uav_nodes[0, :, critic_schema.UAV_X : critic_schema.UAV_Y + 1] = np.asarray(env.uav_pos, dtype=np.float32) / map_scale
        uav_nodes[0, :, critic_schema.UAV_VX : critic_schema.UAV_VY + 1] = np.asarray(env.uav_vel, dtype=np.float32) / v_scale
        uav_nodes[0, :, critic_schema.UAV_ENERGY] = np.asarray(env.uav_energy, dtype=np.float32) / energy_scale
        uav_nodes[0, :, critic_schema.UAV_QUEUE_STEPS] = self._log1p_nonnegative_np(uav_queue / uav_flow_ref)
        uav_nodes[0, :, critic_schema.UAV_QUEUE_FILL] = uav_queue / uav_queue_scale
        uav_nodes[0, :, critic_schema.UAV_LAST_INFLOW_STEPS] = self._log1p_nonnegative_np(uav_last_inflow / uav_flow_ref)
        uav_nodes[0, :, critic_schema.UAV_LAST_OUTFLOW_STEPS] = self._log1p_nonnegative_np(uav_last_outflow / uav_flow_ref)
        uav_nodes[0, :, critic_schema.UAV_LAST_DROP_STEPS] = self._log1p_nonnegative_np(uav_drop / uav_flow_ref)
        uav_nodes[0, :, critic_schema.UAV_SERVICE_EMA_STEPS] = self._log1p_nonnegative_np(uav_ema / uav_flow_ref)
        uav_nodes[0, :, critic_schema.UAV_LOCAL_COST_LOG_RATIO] = self._log_ratio_np(local_uav_cost, uav_flow_cost_ref)
        uav_nodes[0, :, critic_schema.UAV_LAST_TOTAL_COST_LOG_RATIO] = self._log_ratio_np(last_uav_cost, uav_total_cost_ref)
        uav_nodes[0, :, critic_schema.UAV_LAST_WORKLOAD_LOG1P] = np.log1p(np.maximum(last_uav_cost * uav_queue, 0.0)).astype(np.float32)
        uav_nodes[0, :, critic_schema.UAV_LAST_ACCESS_INTERFERENCE_LOG1P] = np.log1p(
            np.maximum(last_interference / access_noise_full_band, 0.0)
        ).astype(np.float32)
        if prefix_known:
            uav_nodes[0, :, critic_schema.UAV_PREFIX_TOTAL_COST_LOG_RATIO] = self._log_ratio_np(prefix_uav_cost, uav_total_cost_ref)
            uav_nodes[0, :, critic_schema.UAV_PREFIX_COST_KNOWN] = 1.0
            uav_nodes[0, :, critic_schema.UAV_PREFIX_WORKLOAD_LOG1P] = np.log1p(np.maximum(prefix_uav_cost * uav_queue, 0.0)).astype(np.float32)

        sat_nodes = np.zeros((1, sat_token_count, critic_schema.CRITIC_SAT_NODE_DIM), dtype=np.float32)
        if sat_token_count > 0:
            valid_ids = safe_sat_ids
            sat_queue = np.asarray(env.sat_queue, dtype=np.float32)
            sat_drop = np.asarray(getattr(env, "sat_drop", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32)
            sat_last_incoming = np.asarray(getattr(env, "last_sat_incoming", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32)
            sat_last_processed = np.asarray(getattr(env, "last_sat_processed", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32)
            sat_proc_capacity = (float(env._effective_sat_cpu_freq()) / self._positive_cfg_scale(float(cfg.task_cycles_per_bit), name="task_cycles_per_bit")) * float(cfg.tau0)
            sat_nodes[0, :, critic_schema.SAT_X : critic_schema.SAT_Z + 1] = sat_pos[valid_ids] / orbit_pos_ref
            sat_nodes[0, :, critic_schema.SAT_VX : critic_schema.SAT_VZ + 1] = sat_vel[valid_ids] / sat_speed_ref
            sat_nodes[0, :, critic_schema.SAT_QUEUE_STEPS] = self._log1p_nonnegative_np(sat_queue[valid_ids] / sat_flow_ref)
            sat_nodes[0, :, critic_schema.SAT_QUEUE_FILL] = sat_queue[valid_ids] / sat_queue_scale
            sat_nodes[0, :, critic_schema.SAT_LAST_INCOMING_STEPS] = self._log1p_nonnegative_np(sat_last_incoming[valid_ids] / sat_flow_ref)
            sat_nodes[0, :, critic_schema.SAT_LAST_PROCESSED_STEPS] = self._log1p_nonnegative_np(sat_last_processed[valid_ids] / sat_flow_ref)
            sat_nodes[0, :, critic_schema.SAT_LAST_DROP_STEPS] = self._log1p_nonnegative_np(sat_drop[valid_ids] / sat_flow_ref)
            sat_nodes[0, :, critic_schema.SAT_SERVICE_EMA_STEPS] = self._log1p_nonnegative_np(sat_ema[valid_ids] / sat_flow_ref)
            sat_nodes[0, :, critic_schema.SAT_COST_LOG_RATIO] = self._log_ratio_np(sat_cost[valid_ids], sat_cost_ref)
            sat_nodes[0, :, critic_schema.SAT_LAST_WORKLOAD_LOG1P] = np.log1p(np.maximum(sat_cost[valid_ids] * sat_queue[valid_ids], 0.0)).astype(np.float32)
            sat_nodes[0, :, critic_schema.SAT_PROC_CAPACITY_STEPS] = float(np.log1p(max(float(sat_proc_capacity / sat_flow_ref), 0.0)))
            last_selected_mask = np.asarray(getattr(env, "last_selected_mask_by_uav_sat", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32)
            sat_nodes[0, :, critic_schema.SAT_LAST_SELECTED_LOAD_FRAC] = np.sum(last_selected_mask[:, valid_ids], axis=0) / max(float(num_uav), 1.0)
            if prefix_known and sat_selection is not None:
                prefix_mask = np.zeros((num_uav, num_sat), dtype=np.float32)
                for u, selected in enumerate(sat_selection):
                    for sat_idx in selected:
                        if 0 <= int(sat_idx) < num_sat:
                            prefix_mask[u, int(sat_idx)] = 1.0
                sat_nodes[0, :, critic_schema.SAT_PREFIX_SELECTED_LOAD_FRAC] = np.sum(prefix_mask[:, valid_ids], axis=0) / max(float(num_uav), 1.0)
                sat_nodes[0, :, critic_schema.SAT_PREFIX_LOAD_KNOWN] = sat_mask_vec.astype(np.float32)
            sat_nodes *= sat_mask_vec.reshape(1, -1, 1).astype(np.float32)

        gu_mask = np.ones((1, num_gu), dtype=bool)
        sat_mask = sat_mask_vec.reshape(1, -1)
        uav_gu_mask = np.ones((1, num_uav, num_gu), dtype=bool)
        uav_sat_mask = np.broadcast_to(sat_mask[:, None, :], (1, num_uav, sat_token_count)).copy()

        uav_gu_edges = np.zeros((1, num_uav, num_gu, critic_schema.CRITIC_UAV_GU_EDGE_DIM), dtype=np.float32)
        if num_gu > 0:
            gu_rel = (np.asarray(env.gu_pos, dtype=np.float32)[None, :, :] - np.asarray(env.uav_pos, dtype=np.float32)[:, None, :]).astype(np.float32)
            horizontal = np.linalg.norm(gu_rel, axis=-1).astype(np.float32)
            access_gain = self._stage_access_gain_matrix if self._stage_access_gain_matrix is not None else env._compute_access_link_gain_matrix()
            access_gain = np.asarray(access_gain, dtype=np.float32).T
            access_snr = channel.snr_linear(
                cfg.gu_tx_power,
                access_gain,
                cfg.noise_density,
                cfg.b_acc,
                noise_figure_db=float(getattr(cfg, "access_noise_figure_db", 0.0) or 0.0),
            )
            if bool(cfg.fading_enabled) and channel.access_fading_mode_from_config(cfg) == "ergodic_rician":
                access_se = channel.rician_ergodic_spectral_efficiency(
                    access_snr,
                    channel.rician_k_linear_from_config(cfg),
                    quadrature_points=int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
                )
            else:
                access_se = channel.spectral_efficiency(access_snr)
            prefix_bw_valid = np.zeros((num_uav, num_gu), dtype=np.float32)
            if int(stage_id) in {self.STAGE_SAT, self.STAGE_BW}:
                for u in range(num_uav):
                    prefix_bw_valid[u] = (np.asarray(assoc, dtype=np.int32) == int(u)).astype(np.float32)
            uav_gu_edges[0, :, :, critic_schema.UG_REL_X : critic_schema.UG_REL_Y + 1] = gu_rel / map_scale
            uav_gu_edges[0, :, :, critic_schema.UG_HORIZONTAL_DIST] = horizontal / map_scale
            uav_gu_edges[0, :, :, critic_schema.UG_ELEVATION_NORM] = np.arctan2(float(cfg.uav_height), geometry_denominator(horizontal)) / (np.pi * 0.5)
            uav_gu_edges[0, :, :, critic_schema.UG_ACCESS_SE_REF] = np.asarray(access_se, dtype=np.float32)
            last_bw_fraction = np.asarray(
                getattr(env, "last_bw_fraction_by_uav_gu", np.zeros((num_uav, num_gu), dtype=np.float32)),
                dtype=np.float32,
            )
            uav_gu_edges[0, :, :, critic_schema.UG_LAST_BW_FRACTION] = last_bw_fraction
            uav_gu_edges[0, :, :, critic_schema.UG_LAST_SERVED_FLAG] = last_bw_fraction
            uav_gu_edges[0, :, :, critic_schema.UG_PREFIX_BW_VALID_FLAG] = prefix_bw_valid
            uav_gu_edges[0, :, :, critic_schema.UG_PREFIX_BW_VALID_KNOWN] = 1.0 if int(stage_id) in {self.STAGE_SAT, self.STAGE_BW} else 0.0
            if int(stage_id) in {self.STAGE_SAT, self.STAGE_BW}:
                uav_nodes[0, :, critic_schema.UAV_PREFIX_BW_VALID_COUNT_FRAC] = np.sum(prefix_bw_valid, axis=1) / max(float(num_gu), 1.0)

        uav_ecef_all = np.stack([env._uav_ecef(u) for u in range(num_uav)], axis=0).astype(np.float32)
        uav_vel_ecef_all = np.stack([env._uav_vel_ecef(u) for u in range(num_uav)], axis=0).astype(np.float32)
        uav_sat_edges = np.zeros((1, num_uav, sat_token_count, critic_schema.CRITIC_UAV_SAT_EDGE_DIM), dtype=np.float32)
        if sat_token_count > 0:
            valid_ids = safe_sat_ids
            rel_pos = sat_pos[valid_ids][None, :, :] - uav_ecef_all[:, None, :]
            rel_vel = sat_vel[valid_ids][None, :, :] - uav_vel_ecef_all[:, None, :]
            range_m = geometry_denominator(np.linalg.norm(rel_pos, axis=-1).astype(np.float32))
            radial_v = np.sum(rel_pos * rel_vel, axis=-1) / geometry_denominator(range_m * sat_speed_ref)
            loss_matrix = env._get_backhaul_loss_matrix(sat_pos)
            elevation_matrix = env._get_elevation_matrix(sat_pos)
            gain = (env._backhaul_gain_const / geometry_denominator(range_m * range_m)).astype(np.float32)
            if loss_matrix is not None:
                gain = gain * loss_matrix[:, valid_ids]
            b_backhaul_per_sat = self._positive_cfg_scale(env._effective_b_backhaul_per_sat(), name="b_backhaul_per_sat")
            backhaul_snr = channel.snr_linear(
                cfg.uav_tx_power,
                gain,
                cfg.noise_density,
                b_backhaul_per_sat,
                noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
            )
            nu_eff = np.zeros((num_uav, sat_token_count), dtype=np.float32)
            if cfg.doppler_enabled or cfg.doppler_atten_enabled or cfg.doppler_observed:
                for u in range(num_uav):
                    raw_nu = env._doppler_many(u, valid_ids, sat_pos, sat_vel)
                    nu_eff_u, _ = env._effective_doppler_array(u, valid_ids, raw_nu)
                    nu_eff[u] = nu_eff_u.astype(np.float32, copy=False)
            if cfg.doppler_observed and cfg.doppler_atten_enabled:
                backhaul_snr = backhaul_snr * channel.doppler_attenuation(nu_eff, cfg.subcarrier_spacing)
            backhaul_se_ref = np.asarray(channel.spectral_efficiency(backhaul_snr), dtype=np.float32)
            doppler_ref = float(getattr(cfg, "backhaul_carrier_freq", getattr(cfg, "carrier_freq", 1.0)) or 1.0) * sat_speed_ref / self._positive_cfg_scale(float(cfg.speed_of_light), name="speed_of_light")
            visible_flag = np.zeros((num_uav, sat_token_count), dtype=np.float32)
            valid_flag = np.zeros((num_uav, sat_token_count), dtype=np.float32)
            for u in range(num_uav):
                vis_set = {int(s) for s in visible[u]}
                visible_flag[u] = np.asarray([1.0 if int(s) in vis_set else 0.0 for s in valid_ids], dtype=np.float32)
                valid = elevation_matrix[u, valid_ids] >= cfg.theta_min_rad
                if cfg.doppler_enabled:
                    valid = valid & (np.abs(nu_eff[u]) <= cfg.nu_max)
                valid_flag[u] = valid.astype(np.float32)
            last_selected_mask = np.asarray(getattr(env, "last_selected_mask_by_uav_sat", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32)
            uav_sat_edges[0, :, :, critic_schema.US_REL_X : critic_schema.US_REL_Z + 1] = rel_pos / orbit_pos_ref
            uav_sat_edges[0, :, :, critic_schema.US_REL_VX : critic_schema.US_REL_VZ + 1] = rel_vel / sat_speed_ref
            uav_sat_edges[0, :, :, critic_schema.US_RADIAL_VELOCITY_NORM] = radial_v
            uav_sat_edges[0, :, :, critic_schema.US_RANGE_NORM] = range_m / orbit_pos_ref
            uav_sat_edges[0, :, :, critic_schema.US_ELEVATION_NORM] = elevation_matrix[:, valid_ids] / (np.pi * 0.5)
            uav_sat_edges[0, :, :, critic_schema.US_DOPPLER_NORM] = nu_eff / self._positive_cfg_scale(doppler_ref, name="doppler_ref")
            uav_sat_edges[0, :, :, critic_schema.US_DOPPLER_MARGIN] = nu_eff / max(float(cfg.nu_max), 1.0)
            uav_sat_edges[0, :, :, critic_schema.US_BACKHAUL_SE_REF] = backhaul_se_ref
            uav_sat_edges[0, :, :, critic_schema.US_VISIBLE_FLAG] = visible_flag
            uav_sat_edges[0, :, :, critic_schema.US_VALID_FLAG] = valid_flag
            uav_sat_edges[0, :, :, critic_schema.US_LAST_SELECTED_FLAG] = last_selected_mask[:, valid_ids]
            if prefix_known and sat_selection is not None:
                prefix_selected = np.zeros((num_uav, sat_token_count), dtype=np.float32)
                selected_load = np.zeros((num_sat,), dtype=np.float32)
                for u, selected in enumerate(sat_selection):
                    for sat_idx in selected:
                        sat_idx_i = int(sat_idx)
                        if 0 <= sat_idx_i < num_sat:
                            selected_load[sat_idx_i] += 1.0
                            if sat_idx_i in sat_id_set:
                                slot = int(np.flatnonzero(sat_ids == sat_idx_i)[0])
                                prefix_selected[u, slot] = 1.0
                uav_sat_edges[0, :, :, critic_schema.US_PREFIX_SELECTED_FLAG] = prefix_selected
                uav_sat_edges[0, :, :, critic_schema.US_PREFIX_SELECTED_KNOWN] = sat_mask_vec[None, :].astype(np.float32)
                load_on_token = np.maximum(selected_load[valid_ids], 1.0)
                b_share = b_backhaul_per_sat / load_on_token
                cap_snr = channel.snr_linear(
                    cfg.uav_tx_power,
                    gain,
                    cfg.noise_density,
                    b_share[None, :],
                    noise_figure_db=float(getattr(cfg, "backhaul_noise_figure_db", 0.0) or 0.0),
                )
                cap_se = np.asarray(channel.spectral_efficiency(cap_snr), dtype=np.float32)
                uav_sat_edges[0, :, :, critic_schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS] = self._log1p_nonnegative_np(
                    prefix_selected * cap_se * b_share[None, :] * float(cfg.tau0) / uav_flow_ref
                )
            uav_sat_edges *= sat_mask_vec.reshape(1, 1, -1, 1).astype(np.float32)

        uav_uav_edges = np.zeros((1, num_uav, num_uav, critic_schema.CRITIC_UAV_UAV_EDGE_DIM), dtype=np.float32)
        rel_uav_pos = np.asarray(env.uav_pos, dtype=np.float32)[None, :, :] - np.asarray(env.uav_pos, dtype=np.float32)[:, None, :]
        rel_uav_vel = np.asarray(env.uav_vel, dtype=np.float32)[None, :, :] - np.asarray(env.uav_vel, dtype=np.float32)[:, None, :]
        dist_uav = np.linalg.norm(rel_uav_pos, axis=-1).astype(np.float32)
        uav_uav_edges[0, :, :, critic_schema.UU_REL_X : critic_schema.UU_REL_Y + 1] = rel_uav_pos / map_scale
        uav_uav_edges[0, :, :, critic_schema.UU_REL_VX : critic_schema.UU_REL_VY + 1] = rel_uav_vel / v_scale
        uav_uav_edges[0, :, :, critic_schema.UU_DIST_NORM] = dist_uav / map_scale
        uav_uav_edges[0, :, :, critic_schema.UU_CLOSING_SPEED_NORM] = -np.sum(rel_uav_pos * rel_uav_vel, axis=-1) / geometry_denominator(dist_uav * v_scale)
        d_alert = float(cfg.avoidance_alert_factor) * float(cfg.d_safe) if bool(cfg.avoidance_enabled) else float(cfg.d_safe)
        uav_uav_edges[0, :, :, critic_schema.UU_ALERT_FLAG] = (dist_uav < d_alert).astype(np.float32)
        uav_uav_edges[0, :, :, critic_schema.UU_UNSAFE_FLAG] = (dist_uav < float(cfg.d_safe)).astype(np.float32)
        last_selected = np.asarray(getattr(env, "last_selected_mask_by_uav_sat", np.zeros((num_uav, num_sat), dtype=np.float32)), dtype=np.float32)
        last_shared = last_selected[:, None, :] * last_selected[None, :, :]
        sat_select_ref = max(float(getattr(cfg, "sat_action_select_k", cfg.N_RF) or cfg.N_RF), 1.0)
        uav_uav_edges[0, :, :, critic_schema.UU_LAST_SHARED_SAT_FRAC] = np.sum(last_shared, axis=-1) / sat_select_ref
        if prefix_known and sat_selection is not None:
            prefix_selected_full = np.zeros((num_uav, num_sat), dtype=np.float32)
            for u, selected in enumerate(sat_selection):
                for sat_idx in selected:
                    if 0 <= int(sat_idx) < num_sat:
                        prefix_selected_full[u, int(sat_idx)] = 1.0
            prefix_shared = prefix_selected_full[:, None, :] * prefix_selected_full[None, :, :]
            uav_uav_edges[0, :, :, critic_schema.UU_PREFIX_SHARED_SAT_FRAC] = np.sum(prefix_shared, axis=-1) / sat_select_ref
            uav_uav_edges[0, :, :, critic_schema.UU_PREFIX_SHARED_SAT_KNOWN] = 1.0
        uav_uav_mask = np.ones((1, num_uav, num_uav), dtype=bool)
        np.fill_diagonal(uav_uav_mask[0], False)
        for idx in (critic_schema.UU_ALERT_FLAG, critic_schema.UU_UNSAFE_FLAG):
            np.fill_diagonal(uav_uav_edges[0, :, :, idx], 0.0)

        global_scalars = np.zeros((1, critic_schema.CRITIC_GLOBAL_SCALAR_DIM), dtype=np.float32)
        sat_queue_all = np.asarray(env.sat_queue, dtype=np.float32)
        sat_drop_all = np.asarray(getattr(env, "sat_drop", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32)
        sat_last_processed_all = np.asarray(getattr(env, "last_sat_processed", np.zeros((num_sat,), dtype=np.float32)), dtype=np.float32)
        global_scalars[0, critic_schema.GLOBAL_TOTAL_GU_QUEUE_STEPS] = float(np.log1p(max(float(np.sum(env.gu_queue) / gu_flow_ref), 0.0))) if num_gu > 0 else 0.0
        global_scalars[0, critic_schema.GLOBAL_TOTAL_UAV_QUEUE_STEPS] = float(np.log1p(max(float(np.sum(uav_queue) / uav_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_SAT_QUEUE_STEPS] = float(np.log1p(max(float(np.sum(sat_queue_all) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_GU_DROP_STEPS] = float(np.log1p(max(float(np.sum(getattr(env, "gu_drop", 0.0)) / gu_flow_ref), 0.0))) if num_gu > 0 else 0.0
        global_scalars[0, critic_schema.GLOBAL_TOTAL_UAV_DROP_STEPS] = float(np.log1p(max(float(np.sum(uav_drop) / uav_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_SAT_DROP_STEPS] = float(np.log1p(max(float(np.sum(sat_drop_all) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS] = float(np.log1p(max(float(np.sum(env._current_expected_gu_arrival_rates() * float(cfg.tau0)) / gu_flow_ref), 0.0))) if num_gu > 0 else 0.0
        global_scalars[0, critic_schema.GLOBAL_TOTAL_LAST_GU_OUTFLOW_STEPS] = float(np.log1p(max(float(np.sum(getattr(env, "last_gu_outflow", 0.0)) / gu_flow_ref), 0.0))) if num_gu > 0 else 0.0
        global_scalars[0, critic_schema.GLOBAL_TOTAL_LAST_UAV_OUTFLOW_STEPS] = float(np.log1p(max(float(np.sum(uav_last_outflow) / uav_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_LAST_SAT_PROCESSED_STEPS] = float(np.log1p(max(float(np.sum(sat_last_processed_all) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_TOTAL_LAST_WEIGHTED_WORKLOAD_STEPS] = float(np.log1p(max(
            float(np.sum(last_gu_cost * env.gu_queue)) + float(np.sum(last_uav_cost * uav_queue)) + float(np.sum(last_sat_cost * sat_queue_all)),
            0.0,
        )))
        if prefix_known:
            global_scalars[0, critic_schema.GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS] = float(np.log1p(max(
                float(np.sum(prefix_gu_cost * env.gu_queue)) + float(np.sum(prefix_uav_cost * uav_queue)) + float(np.sum(prefix_sat_cost * sat_queue_all)),
                0.0,
            )))
            global_scalars[0, critic_schema.GLOBAL_PREFIX_WORKLOAD_KNOWN] = 1.0
            selected_loads = np.zeros((num_sat,), dtype=np.float32)
            for selected in sat_selection or []:
                for sat_idx in selected:
                    if 0 <= int(sat_idx) < num_sat:
                        selected_loads[int(sat_idx)] += 1.0
            active_loads = selected_loads[selected_loads > 0.0]
            if active_loads.size > 0:
                global_scalars[0, critic_schema.GLOBAL_SELECTED_SAT_LOAD_MEAN] = float(np.mean(active_loads / max(float(num_uav), 1.0)))
                global_scalars[0, critic_schema.GLOBAL_SELECTED_SAT_LOAD_MAX] = float(np.max(active_loads / max(float(num_uav), 1.0)))
            global_scalars[0, critic_schema.GLOBAL_SELECTED_SAT_LOAD_KNOWN] = 1.0
        global_scalars[0, critic_schema.GLOBAL_LAST_INTERFERENCE_MEAN] = float(np.mean(uav_nodes[0, :, critic_schema.UAV_LAST_ACCESS_INTERFERENCE_LOG1P])) if num_uav > 0 else 0.0
        global_scalars[0, critic_schema.GLOBAL_LAST_INTERFERENCE_MAX] = float(np.max(uav_nodes[0, :, critic_schema.UAV_LAST_ACCESS_INTERFERENCE_LOG1P])) if num_uav > 0 else 0.0
        last_loads = np.sum(last_selected, axis=0)
        last_active = last_loads[last_loads > 0.0]
        if last_active.size > 0:
            global_scalars[0, critic_schema.GLOBAL_LAST_SELECTED_SAT_LOAD_MEAN] = float(np.mean(last_active / max(float(num_uav), 1.0)))
            global_scalars[0, critic_schema.GLOBAL_LAST_SELECTED_SAT_LOAD_MAX] = float(np.max(last_active / max(float(num_uav), 1.0)))
        non_token_mask = np.ones((num_sat,), dtype=bool)
        if sat_id_set:
            non_token_mask[np.asarray(sorted(sat_id_set), dtype=np.int64)] = False
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_COUNT_FRAC] = float(np.sum(non_token_mask) / max(float(num_sat), 1.0))
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS] = float(np.log1p(max(float(np.sum(sat_queue_all[non_token_mask]) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_DROP_STEPS] = float(np.log1p(max(float(np.sum(sat_drop_all[non_token_mask]) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS] = float(np.log1p(max(float(np.sum(sat_last_processed_all[non_token_mask]) / sat_flow_ref), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS] = float(np.log1p(max(float(np.sum(sat_cost[non_token_mask] * sat_queue_all[non_token_mask])), 0.0)))
        global_scalars[0, critic_schema.GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS] = float(np.log1p(max(float(np.sum(sat_cost[non_token_mask] * sat_drop_all[non_token_mask])), 0.0)))
        t_steps = max(float(getattr(cfg, "T_steps", 1) or 1), 1.0)
        denom = max(t_steps - 1.0, 1.0)
        t_now = float(getattr(env, "t", 0) or 0)
        global_scalars[0, critic_schema.GLOBAL_REMAINING_HORIZON_FRAC] = float(np.clip((t_steps - 1.0 - t_now) / denom, 0.0, 1.0))

        world_state = StructuredWorldState(
            uav_nodes=uav_nodes.astype(np.float32, copy=False),
            gu_nodes=gu_nodes.astype(np.float32, copy=False),
            sat_nodes=sat_nodes.astype(np.float32, copy=False),
            sat_ids=sat_ids.reshape(1, -1).astype(np.int64, copy=False),
            uav_gu_edges=uav_gu_edges.astype(np.float32, copy=False),
            uav_sat_edges=uav_sat_edges.astype(np.float32, copy=False),
            uav_uav_edges=uav_uav_edges.astype(np.float32, copy=False),
            global_scalars=global_scalars.astype(np.float32, copy=False),
            gu_mask=gu_mask,
            sat_mask=sat_mask,
            uav_gu_mask=uav_gu_mask,
            uav_sat_mask=uav_sat_mask,
            uav_uav_mask=uav_uav_mask,
            stage_id=np.asarray([stage_id], dtype=np.int64),
        )
        return self._world_state_to_output_device(world_state)

    def _build_world_state(
        self,
        stage_id: int,
        assoc: np.ndarray,
        candidates: List[List[int]],
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        visible: List[List[int]],
        sat_selection: List[List[int]] | None,
    ) -> StructuredWorldState:
        return self._build_critic_world_state(
            stage_id,
            assoc,
            candidates,
            sat_pos,
            sat_vel,
            visible,
            sat_selection,
        )
