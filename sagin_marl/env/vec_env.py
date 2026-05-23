from __future__ import annotations

import copy
import itertools
import multiprocessing as mp
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import Dict, List, Sequence

import numpy as np

from .config import SaginConfig
from .sagin_env import SaginParallelEnv


def _done_from_flags(terms: Dict[str, bool], truncs: Dict[str, bool]) -> bool:
    if not terms:
        return False
    first_agent = next(iter(terms))
    return bool(terms[first_agent] or truncs[first_agent])


def _collect_step_stats(env: SaginParallelEnv) -> Dict[str, object]:
    def _safe_mean(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.mean(a)) if a.size > 0 else 0.0

    def _safe_max(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.max(a)) if a.size > 0 else 0.0

    def _safe_sum(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.sum(a)) if a.size > 0 else 0.0

    num_agents = len(getattr(env, 'agents', []))
    default_accel = np.zeros((num_agents, 2), dtype=np.float32)
    default_bw = np.zeros((num_agents, getattr(env.cfg, 'num_gu', 0)), dtype=np.float32)
    sat_k = max(int(getattr(env.cfg, 'sat_action_select_k', getattr(env.cfg, 'sat_num_select', env.cfg.N_RF)) or env.cfg.N_RF), 0)
    default_sat_mask = np.zeros((num_agents, getattr(env.cfg, 'sats_obs_max', 0)), dtype=np.float32)
    default_sat_indices = np.full((num_agents, sat_k), -1, dtype=np.int64)
    parts = getattr(env, 'last_reward_parts', None) or {}
    profile = getattr(env, 'last_step_profile', None) or {}
    sat_processed = getattr(env, 'last_sat_processed', None)
    sat_incoming = getattr(env, 'last_sat_incoming', None)

    return {
        'last_exec_accel': np.asarray(getattr(env, 'last_exec_accel', default_accel), dtype=np.float32),
        'last_exec_bw_alloc': np.asarray(getattr(env, 'last_exec_bw_alloc', default_bw), dtype=np.float32),
        'last_exec_sat_select_mask': np.asarray(
            getattr(env, 'last_exec_sat_select_mask', default_sat_mask),
            dtype=np.float32,
        ),
        'last_exec_sat_indices': np.asarray(
            getattr(env, 'last_exec_sat_indices', default_sat_indices),
            dtype=np.int64,
        ),
        'danger_imitation_mask': np.asarray(
            getattr(env, 'last_danger_imitation_mask', np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'visible_raw_counts': np.asarray(
            getattr(env, 'last_visible_raw_counts', np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'visible_kept_counts': np.asarray(
            getattr(env, 'last_visible_kept_counts', np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'visible_stats': dict(getattr(env, 'last_visible_stats', {}) or {}),
        'gu_queue_mean': _safe_mean(getattr(env, 'gu_queue', 0.0)),
        'uav_queue_mean': _safe_mean(getattr(env, 'uav_queue', 0.0)),
        'sat_queue_mean': _safe_mean(getattr(env, 'sat_queue', 0.0)),
        'gu_queue_max': _safe_max(getattr(env, 'gu_queue', 0.0)),
        'uav_queue_max': _safe_max(getattr(env, 'uav_queue', 0.0)),
        'sat_queue_max': _safe_max(getattr(env, 'sat_queue', 0.0)),
        'gu_drop_sum': _safe_sum(getattr(env, 'gu_drop', 0.0)),
        'uav_drop_sum': _safe_sum(getattr(env, 'uav_drop', 0.0)),
        'sat_drop_sum': _safe_sum(getattr(env, 'sat_drop', 0.0)),
        'sat_processed_sum': _safe_sum(sat_processed) if sat_processed is not None else 0.0,
        'sat_incoming_sum': _safe_sum(sat_incoming) if sat_incoming is not None else 0.0,
        'connected_sat_count': float(getattr(env, 'last_connected_sat_count', 0.0)),
        'connected_sat_dist_mean': float(getattr(env, 'last_connected_sat_dist_mean', 0.0)),
        'connected_sat_dist_p95': float(getattr(env, 'last_connected_sat_dist_p95', 0.0)),
        'connected_sat_elevation_deg_mean': float(
            getattr(env, 'last_connected_sat_elevation_deg_mean', 0.0)
        ),
        'connected_sat_elevation_deg_min': float(
            getattr(env, 'last_connected_sat_elevation_deg_min', 0.0)
        ),
        'assoc_centroid_dist_norms': np.asarray(
            getattr(env, 'last_assoc_centroid_dist_norms', np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'sat_overlap_uav': np.asarray(
            getattr(env, 'last_sat_overlap_uav', np.zeros((num_agents,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'energy_mean': _safe_mean(getattr(env, 'uav_energy', 0.0)),
        'dynamics_time_sec': float(profile.get('dynamics_time_sec', 0.0)),
        'orbit_visible_time_sec': float(profile.get('orbit_visible_time_sec', 0.0)),
        'assoc_access_time_sec': float(profile.get('assoc_access_time_sec', 0.0)),
        'backhaul_queue_time_sec': float(profile.get('backhaul_queue_time_sec', 0.0)),
        'reward_time_sec': float(profile.get('reward_time_sec', 0.0)),
        'obs_time_sec': float(profile.get('obs_time_sec', 0.0)),
        'state_time_sec': float(profile.get('state_time_sec', 0.0)),
        'step_total_time_sec': float(profile.get('step_total_time_sec', 0.0)),
        'reward_parts': dict(parts),
    }


def _copy_action_dict(actions: Dict[str, Dict]) -> Dict[str, Dict]:
    copied: Dict[str, Dict] = {}
    for agent, action in actions.items():
        copied_action: Dict[str, object] = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                copied_action[key] = np.asarray(value).copy()
            else:
                copied_action[key] = copy.deepcopy(value)
        copied[agent] = copied_action
    return copied


def _combo_mask(num_slots: int, combo: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((num_slots,), dtype=np.float32)
    for idx in combo:
        mask[int(idx)] = 1.0
    return mask


def _selected_sat_slots(
    env: SaginParallelEnv,
    u: int,
    cand: list[int],
    sat_raw: np.ndarray,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    cfg = env.cfg
    sat_raw = np.asarray(sat_raw, dtype=np.float32)[: len(cand)]
    valid_flags = np.ones((len(cand),), dtype=bool)
    if cfg.doppler_enabled and len(cand) > 0:
        cand_idx = np.asarray(cand, dtype=np.int32)
        raw_nu = env._doppler_many(u, cand_idx, sat_pos, sat_vel)
        nu_eff, _ = env._effective_doppler_array(u, cand_idx, raw_nu)
        valid_flags = np.abs(nu_eff) <= cfg.nu_max
    valid_slots = np.flatnonzero(valid_flags)
    if valid_slots.size == 0:
        return [], valid_flags
    chosen_slots = np.flatnonzero((sat_raw > 0.5) & valid_flags)
    if chosen_slots.size > cfg.N_RF:
        order = np.argsort(-sat_raw[chosen_slots], kind='stable')
        chosen_slots = chosen_slots[order[: cfg.N_RF]]
    if chosen_slots.size == 0:
        best_slot = int(valid_slots[int(np.argmax(sat_raw[valid_slots]))])
        chosen_slots = np.array([best_slot], dtype=np.int64)
    chosen_slots = chosen_slots[: cfg.N_RF]
    return [int(x) for x in chosen_slots.tolist()], valid_flags


def _collect_train_reward_stats(env: SaginParallelEnv) -> Dict[str, object]:
    num_uav = max(int(getattr(env.cfg, 'num_uav', 0) or 0), 0)
    return {
        'reward_parts': dict(getattr(env, 'last_reward_parts', {}) or {}),
        'assoc_centroid_dist_norms': np.asarray(
            getattr(env, 'last_assoc_centroid_dist_norms', np.zeros((num_uav,), dtype=np.float32)),
            dtype=np.float32,
        ),
        'sat_overlap_uav': np.asarray(
            getattr(env, 'last_sat_overlap_uav', np.zeros((num_uav,), dtype=np.float32)),
            dtype=np.float32,
        ),
    }


def _maybe_collect_sat_counterfactual_rows(
    env: SaginParallelEnv,
    actions: Dict[str, Dict],
    sample_prob: float,
) -> List[Dict[str, object]]:
    if sample_prob <= 0.0 or bool(getattr(env.cfg, 'fixed_satellite_strategy', False)):
        return []
    cfg = env.cfg
    sat_k = max(int(getattr(cfg, 'sat_action_select_k', getattr(cfg, 'sat_num_select', cfg.N_RF)) or cfg.N_RF), 0)
    if sat_k <= 0:
        return []
    sat_pos, sat_vel = env._get_orbit_states()
    visible = [list(v) for v in getattr(env, 'last_visible_candidates', [])]
    if len(visible) != cfg.num_uav:
        visible = env._visible_sats_sorted(sat_pos)

    from ..rl.policy import batch_flatten_obs

    rows: List[Dict[str, object]] = []
    for u, agent in enumerate(env.agents):
        if np.random.random() >= float(sample_prob):
            continue
        cand = visible[u][: cfg.sats_obs_max]
        if not cand:
            continue
        action_data = actions[agent]
        sat_raw = action_data.get(
            'sat_select_mask',
            action_data.get('sat_logits', np.zeros((cfg.sats_obs_max,), dtype=np.float32)),
        )
        chosen_slots, valid_flags = _selected_sat_slots(env, u, cand, sat_raw, sat_pos, sat_vel)
        valid_slots = np.flatnonzero(valid_flags)
        if valid_slots.size == 0:
            continue
        k = min(sat_k, int(valid_slots.size))
        if k <= 0 or len(chosen_slots) < k:
            continue
        selected_combo = tuple(int(x) for x in chosen_slots[:k])
        combos = list(itertools.combinations(valid_slots.tolist(), k))
        if selected_combo not in combos:
            continue
        alternatives: List[Dict[str, object]] = []
        for combo in combos:
            combo_tuple = tuple(int(x) for x in combo)
            if combo_tuple == selected_combo:
                continue
            actions_cf = _copy_action_dict(actions)
            actions_cf[agent]['sat_select_mask'] = _combo_mask(cfg.sats_obs_max, combo_tuple)
            env_cf = copy.deepcopy(env)
            try:
                obs_cf, rewards_cf, terms_cf, truncs_cf, _ = env_cf.step(actions_cf)
                obs_cf_step = batch_flatten_obs(list(obs_cf.values()), cfg).astype(np.float32, copy=False)
                alternatives.append(
                    {
                        'combo_slots': np.asarray(combo_tuple, dtype=np.int64),
                        'reward': float(list(rewards_cf.values())[0]),
                        'terminated': bool(list(terms_cf.values())[0]),
                        'truncated': bool(list(truncs_cf.values())[0]),
                        'next_state': np.asarray(env_cf.get_global_state(), dtype=np.float32),
                        'next_obs_step': obs_cf_step,
                        'train_reward_stats': _collect_train_reward_stats(env_cf),
                    }
                )
            finally:
                close_fn = getattr(env_cf, 'close', None)
                if callable(close_fn):
                    close_fn()
        if alternatives:
            rows.append(
                {
                    'uav': int(u),
                    'selected_slots': np.asarray(selected_combo, dtype=np.int64),
                    'alternatives': alternatives,
                }
            )
    return rows


def _worker(remote: Connection, cfg: SaginConfig, seed_offset: int) -> None:
    env_cfg = replace(cfg, seed=int(cfg.seed) + int(seed_offset))
    env = SaginParallelEnv(env_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'reset':
                seed = None if data is None else data.get('seed')
                obs, infos = env.reset(seed=seed)
                state = env.get_global_state()
                remote.send((obs, infos, state))
            elif cmd == 'step':
                actions = data['actions']
                auto_reset = bool(data.get('auto_reset', True))
                sat_cf_request = dict(data.get('sat_cf_request', {}) or {})
                sat_cf_rows = _maybe_collect_sat_counterfactual_rows(
                    env,
                    actions,
                    float(sat_cf_request.get('sample_prob', 0.0) or 0.0),
                )
                obs, rewards, terms, truncs, infos = env.step(actions)
                post_step_obs = copy.deepcopy(obs)
                stats = _collect_step_stats(env)
                if sat_cf_rows:
                    stats['sat_cf_credit_rows'] = sat_cf_rows
                stats['post_step_obs'] = post_step_obs
                stats['post_step_global_state'] = np.asarray(env.get_global_state(), dtype=np.float32)
                if auto_reset and _done_from_flags(terms, truncs):
                    obs, _ = env.reset()
                state = env.get_global_state()
                remote.send((obs, rewards, terms, truncs, infos, stats, state))
            elif cmd == 'get_state':
                remote.send(env.get_global_state())
            elif cmd == 'close':
                remote.close()
                return
            else:
                raise ValueError(f'Unknown worker command: {cmd}')
    finally:
        try:
            close_fn = getattr(env, 'close', None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass


class SyncVecSaginEnv:
    def __init__(self, cfg: SaginConfig, num_envs: int):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError('num_envs must be >= 1')
        self.envs = [SaginParallelEnv(replace(cfg, seed=int(cfg.seed) + i)) for i in range(self.num_envs)]
        self.agents = list(self.envs[0].agents)
        self.last_step_stats: List[Dict[str, object]] = [{} for _ in range(self.num_envs)]
        self.last_state_batch: np.ndarray | None = None

    def reset(self, seeds: Sequence[int] | None = None):
        obs_batch = []
        infos_batch = []
        state_batch = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else int(seeds[idx])
            obs, infos = env.reset(seed=seed)
            obs_batch.append(obs)
            infos_batch.append(infos)
            state_batch.append(np.asarray(env.get_global_state(), dtype=np.float32))
        self.last_state_batch = np.stack(state_batch, axis=0).astype(np.float32, copy=False)
        return obs_batch, infos_batch

    def step(self, action_batch, auto_reset: bool = True, sat_cf_request: Dict[str, object] | None = None):
        if len(action_batch) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} action dicts, got {len(action_batch)}')
        obs_batch = []
        rewards_batch = []
        terms_batch = []
        truncs_batch = []
        infos_batch = []
        stats_batch: List[Dict[str, object]] = []
        state_batch = []
        for env, actions in zip(self.envs, action_batch):
            sat_cf_rows = _maybe_collect_sat_counterfactual_rows(
                env,
                actions,
                float((sat_cf_request or {}).get('sample_prob', 0.0) or 0.0),
            )
            obs, rewards, terms, truncs, infos = env.step(actions)
            post_step_obs = copy.deepcopy(obs)
            stats = _collect_step_stats(env)
            if sat_cf_rows:
                stats['sat_cf_credit_rows'] = sat_cf_rows
            stats['post_step_obs'] = post_step_obs
            stats['post_step_global_state'] = np.asarray(env.get_global_state(), dtype=np.float32)
            if auto_reset and _done_from_flags(terms, truncs):
                obs, _ = env.reset()
            obs_batch.append(obs)
            rewards_batch.append(rewards)
            terms_batch.append(terms)
            truncs_batch.append(truncs)
            infos_batch.append(infos)
            stats_batch.append(stats)
            state_batch.append(np.asarray(env.get_global_state(), dtype=np.float32))
        self.last_step_stats = stats_batch
        self.last_state_batch = np.stack(state_batch, axis=0).astype(np.float32, copy=False)
        return obs_batch, rewards_batch, terms_batch, truncs_batch, infos_batch

    def get_global_state_batch(self) -> np.ndarray:
        if self.last_state_batch is not None:
            return np.asarray(self.last_state_batch, dtype=np.float32)
        states = [env.get_global_state() for env in self.envs]
        return np.stack(states, axis=0).astype(np.float32, copy=False)

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, 'close', None)
            if callable(close_fn):
                close_fn()


class SubprocVecSaginEnv:
    def __init__(self, cfg: SaginConfig, num_envs: int, start_method: str = 'spawn'):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError('num_envs must be >= 1')
        self.agents = [f'uav_{i}' for i in range(cfg.num_uav)]
        self.last_step_stats: List[Dict[str, object]] = [{} for _ in range(self.num_envs)]
        self.last_state_batch: np.ndarray | None = None

        ctx = mp.get_context(start_method)
        self._remotes: List[Connection] = []
        self._procs: List[mp.Process] = []
        for idx in range(self.num_envs):
            remote, work_remote = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(work_remote, cfg, idx), daemon=True)
            proc.start()
            work_remote.close()
            self._remotes.append(remote)
            self._procs.append(proc)

    def reset(self, seeds: Sequence[int] | None = None):
        for idx, remote in enumerate(self._remotes):
            seed = None if seeds is None else int(seeds[idx])
            remote.send(('reset', {'seed': seed}))
        results = [remote.recv() for remote in self._remotes]
        obs_batch, infos_batch, state_batch = zip(*results)
        self.last_state_batch = np.stack(state_batch, axis=0).astype(np.float32, copy=False)
        return list(obs_batch), list(infos_batch)

    def step(self, action_batch, auto_reset: bool = True, sat_cf_request: Dict[str, object] | None = None):
        if len(action_batch) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} action dicts, got {len(action_batch)}')
        for remote, actions in zip(self._remotes, action_batch):
            remote.send(
                (
                    'step',
                    {
                        'actions': actions,
                        'auto_reset': bool(auto_reset),
                        'sat_cf_request': dict(sat_cf_request or {}),
                    },
                )
            )
        results = [remote.recv() for remote in self._remotes]
        obs_batch, rewards_batch, terms_batch, truncs_batch, infos_batch, stats_batch, state_batch = zip(*results)
        self.last_step_stats = list(stats_batch)
        self.last_state_batch = np.stack(state_batch, axis=0).astype(np.float32, copy=False)
        return list(obs_batch), list(rewards_batch), list(terms_batch), list(truncs_batch), list(infos_batch)

    def get_global_state_batch(self) -> np.ndarray:
        if self.last_state_batch is not None:
            return np.asarray(self.last_state_batch, dtype=np.float32)
        for remote in self._remotes:
            remote.send(('get_state', None))
        states = [remote.recv() for remote in self._remotes]
        return np.stack(states, axis=0).astype(np.float32, copy=False)

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(('close', None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        self._remotes.clear()
        self._procs.clear()


def make_vec_env(cfg: SaginConfig, num_envs: int, backend: str = 'sync'):
    if int(num_envs) <= 1:
        raise ValueError('num_envs must be > 1 for vectorized environments.')
    backend_l = str(backend).lower()
    if backend_l == 'sync':
        return SyncVecSaginEnv(cfg, num_envs)
    if backend_l == 'subproc':
        return SubprocVecSaginEnv(cfg, num_envs)
    raise ValueError(f'Unknown vec backend: {backend}')
