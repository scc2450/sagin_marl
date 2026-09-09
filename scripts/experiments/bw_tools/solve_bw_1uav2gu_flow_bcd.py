from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import _quantize_queue_contract_np
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


DEFAULT_CONFIG = "configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_currentenv_competition_ppo.yaml"
DEFAULT_OUT_DIR = "artifacts/bw_oracle/flow_bcd"


@dataclass
class FlowState:
    gu_queue: np.ndarray
    uav_queue: np.ndarray
    sat_queue: np.ndarray
    gu_ema: np.ndarray
    uav_ema: np.ndarray
    sat_ema: np.ndarray


@dataclass
class StageModel:
    access_bits_by_grid: np.ndarray
    relay_cap_bits: float
    relay_share: np.ndarray
    sat_compute_cap_bits: np.ndarray
    arrival_bits: np.ndarray
    expected_arrival_bits: np.ndarray
    selected_sat_indices: np.ndarray
    candidate_indices: np.ndarray
    valid_mask: np.ndarray


@dataclass
class FlowProblem:
    stages: list[StageModel]
    initial_state: FlowState


@dataclass
class FlowRollout:
    reward_sum: float
    rewards: list[float]
    gu_queue: list[list[float]]
    uav_queue: list[list[float]]
    sat_queue_sum: list[float]
    gu_service: list[list[float]]
    uav_service: list[float]
    sat_processed_sum: list[float]
    drop_sum: list[float]


@dataclass
class NativeRollout:
    reward_sum: float
    level_reward_sum: float
    actions: list[list[float]]
    x_by_t: list[float]
    arrivals: list[list[float]]
    gu_queue: list[list[float]]
    reward_by_step: list[float]
    final_gu_queue: list[float]
    final_uav_queue: list[float]
    final_sat_queue_sum: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sat_pair_dummy(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", getattr(cfg, "N_RF", 1)) or 1), 1)
    return np.full((int(cfg.num_uav), select_k), -1, dtype=np.int64)


def _prepare_bw_stage(driver, cfg) -> None:
    driver.begin_step()
    driver.run_accel_stage(np.zeros((int(cfg.num_uav), 2), dtype=np.float32))
    driver.run_sat_stage(_sat_pair_dummy(cfg))


def _stage_candidate_arrays(driver, cfg) -> tuple[np.ndarray, np.ndarray]:
    candidates = getattr(driver, "_stage_candidates")
    assoc = np.asarray(getattr(driver, "_stage_assoc"), dtype=np.int32)
    candidate_indices = np.full((int(cfg.users_obs_max),), -1, dtype=np.int32)
    valid_mask = np.zeros((int(cfg.users_obs_max),), dtype=bool)
    for slot, gu_idx in enumerate(list(candidates[0])[: int(cfg.users_obs_max)]):
        candidate_indices[slot] = int(gu_idx)
        valid_mask[slot] = 0 <= int(gu_idx) < int(cfg.num_gu) and int(assoc[int(gu_idx)]) == 0
    return candidate_indices, valid_mask


def _action_from_x(driver, cfg, x: float) -> np.ndarray:
    candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    action = np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
    x0 = float(np.clip(float(x), 0.0, 1.0))
    shares = {0: x0, 1: 1.0 - x0}
    for slot, gu_idx in enumerate(candidate_indices):
        if bool(valid_mask[slot]) and int(gu_idx) in shares:
            action[0, slot] = float(shares[int(gu_idx)])
    denom = float(np.sum(action[0, valid_mask], dtype=np.float32)) if np.any(valid_mask) else 0.0
    if denom > 1.0e-9:
        action[0, valid_mask] /= denom
    elif np.any(valid_mask):
        action[0, valid_mask] = 1.0 / float(np.sum(valid_mask))
    return action


def _action_from_stage_model(cfg, stage: StageModel, x: float) -> np.ndarray:
    action = np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
    x0 = float(np.clip(float(x), 0.0, 1.0))
    shares = {0: x0, 1: 1.0 - x0}
    for slot, gu_idx in enumerate(stage.candidate_indices):
        if bool(stage.valid_mask[slot]) and int(gu_idx) in shares:
            action[0, slot] = float(shares[int(gu_idx)])
    denom = float(np.sum(action[0, stage.valid_mask], dtype=np.float32)) if np.any(stage.valid_mask) else 0.0
    if denom > 1.0e-9:
        action[0, stage.valid_mask] /= denom
    elif np.any(stage.valid_mask):
        action[0, stage.valid_mask] = 1.0 / float(np.sum(stage.valid_mask))
    return action


def _x_from_action(driver, cfg, action: np.ndarray) -> float:
    candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    act = np.asarray(action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
    for slot, gu_idx in enumerate(candidate_indices):
        if bool(valid_mask[slot]) and int(gu_idx) == 0:
            return float(act[0, slot])
    return 0.5


def _uniform_action(driver, cfg) -> np.ndarray:
    _candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    action = np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
    if np.any(valid_mask):
        action[0, valid_mask] = 1.0 / float(np.sum(valid_mask))
    return action


def _nearest_grid_indices(x_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(x_grid, dtype=np.float64).reshape(1, -1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return np.argmin(np.abs(vals - grid), axis=1).astype(np.int64)


def _expected_arrival_rates(env, cfg) -> np.ndarray:
    arrival_rate = float(getattr(env, "effective_task_arrival_rate", cfg.task_arrival_rate))
    ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
    if ramp_steps > 0:
        start = float(np.clip(float(getattr(cfg, "arrival_ramp_start", 0.0) or 0.0), 0.0, 1.0))
        use_global = bool(getattr(cfg, "arrival_ramp_use_global", False))
        t_ref = int(getattr(env, "global_step", 0)) if use_global else int(getattr(env, "t", 0))
        progress = min(1.0, float(t_ref) / max(ramp_steps, 1))
        arrival_rate = arrival_rate * (start + (1.0 - start) * progress)
    return np.asarray(env._current_task_arrival_rates(arrival_rate), dtype=np.float64)


def _flow_state_from_driver(driver) -> FlowState:
    gu_ema, uav_ema, sat_ema = driver._bw_weighted_workload_device_ema_vectors()
    env = driver.env
    return FlowState(
        gu_queue=np.asarray(env.gu_queue, dtype=np.float64).copy(),
        uav_queue=np.asarray(env.uav_queue, dtype=np.float64).copy(),
        sat_queue=np.asarray(env.sat_queue, dtype=np.float64).copy(),
        gu_ema=np.asarray(gu_ema, dtype=np.float64).copy(),
        uav_ema=np.asarray(uav_ema, dtype=np.float64).copy(),
        sat_ema=np.asarray(sat_ema, dtype=np.float64).copy(),
    )


def _build_stage_model(driver, cfg, x_grid: np.ndarray, *, arrival_bits: np.ndarray | None = None) -> StageModel:
    candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    access_bits = np.zeros((int(x_grid.size), int(cfg.num_gu)), dtype=np.float64)
    for i, x in enumerate(np.asarray(x_grid, dtype=np.float64).reshape(-1)):
        rates, _eta = driver.env._compute_access_rates(
            np.asarray(getattr(driver, "_stage_assoc"), dtype=np.int32),
            getattr(driver, "_stage_candidates"),
            _action_from_x(driver, cfg, float(x)),
            record_exec=False,
            access_snapshot=getattr(driver, "_stage_access_gain_matrix"),
        )
        bits = np.asarray(rates, dtype=np.float64) * float(cfg.tau0)
        access_bits[i] = np.asarray(
            _quantize_queue_contract_np(cfg, bits, attr="structured_flow_bits_quantum", default=32.0),
            dtype=np.float64,
        )

    rate_matrix, _sat_loads = driver.env._compute_backhaul_rates(
        np.asarray(getattr(driver, "_stage_sat_pos"), dtype=np.float32),
        np.asarray(getattr(driver, "_stage_sat_vel"), dtype=np.float32),
        np.asarray(getattr(driver, "_stage_sat_selection_matrix"), dtype=np.int64),
    )
    rates = np.asarray(rate_matrix, dtype=np.float64)[0]
    relay_bits = float(np.sum(rates, dtype=np.float64) * float(cfg.tau0))
    relay_cap = float(
        np.asarray(
            _quantize_queue_contract_np(cfg, relay_bits, attr="structured_flow_bits_quantum", default=32.0),
            dtype=np.float64,
        ).reshape(-1)[0]
    )
    if relay_bits > 0.0:
        relay_share = rates / float(np.sum(rates, dtype=np.float64))
    else:
        relay_share = np.zeros((int(cfg.num_sat),), dtype=np.float64)
    compute_rate = float(driver.env._effective_sat_cpu_freq()) / max(float(cfg.task_cycles_per_bit), 1.0e-9)
    compute_bits = float(compute_rate) * float(cfg.tau0)
    sat_compute_cap = np.full(
        (int(cfg.num_sat),),
        float(
            np.asarray(
                _quantize_queue_contract_np(cfg, compute_bits, attr="structured_flow_bits_quantum", default=32.0),
                dtype=np.float64,
            ).reshape(-1)[0]
        ),
        dtype=np.float64,
    )
    selection_matrix = np.asarray(getattr(driver, "_stage_sat_selection_matrix"), dtype=np.int64)
    selected = selection_matrix[0]
    selected = selected[(selected >= 0) & (selected < int(cfg.num_sat))].astype(np.int64, copy=False)
    expected_arrival = _expected_arrival_rates(driver.env, cfg)
    if arrival_bits is None:
        arrival = expected_arrival.copy()
    else:
        arrival = np.asarray(arrival_bits, dtype=np.float64).copy()
    return StageModel(
        access_bits_by_grid=access_bits,
        relay_cap_bits=relay_cap,
        relay_share=relay_share.astype(np.float64, copy=False),
        sat_compute_cap_bits=sat_compute_cap,
        arrival_bits=arrival,
        expected_arrival_bits=expected_arrival,
        selected_sat_indices=selected,
        candidate_indices=candidate_indices,
        valid_mask=valid_mask,
    )


def _device_costs_from_ema(cfg, state: FlowState, selected_sat_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = max(float(getattr(cfg, "bw_weighted_workload_eps", 1.0) or 0.0), 1.0)
    sat_cost = (1.0 / np.maximum(state.sat_ema, eps)).astype(np.float64, copy=False)
    sat_fallback = float(np.mean(sat_cost)) if sat_cost.size else 0.0
    selected = np.asarray(selected_sat_indices, dtype=np.int64)
    selected = selected[(selected >= 0) & (selected < int(cfg.num_sat))]
    uav_downstream = float(np.mean(sat_cost[selected])) if selected.size else sat_fallback
    uav_cost = np.asarray([1.0 / max(float(state.uav_ema[0]), eps) + uav_downstream], dtype=np.float64)
    gu_cost = (1.0 / np.maximum(state.gu_ema, eps) + float(uav_cost[0])).astype(np.float64, copy=False)
    return gu_cost, uav_cost, sat_cost


def _weighted_total(gu_cost: np.ndarray, uav_cost: np.ndarray, sat_cost: np.ndarray, gu: np.ndarray, uav: np.ndarray, sat: np.ndarray) -> float:
    return float(np.sum(gu_cost * gu, dtype=np.float64) + np.sum(uav_cost * uav, dtype=np.float64) + np.sum(sat_cost * sat, dtype=np.float64))


def simulate_flow(cfg, problem: FlowProblem, x_indices: np.ndarray) -> FlowRollout:
    state = FlowState(
        gu_queue=problem.initial_state.gu_queue.copy(),
        uav_queue=problem.initial_state.uav_queue.copy(),
        sat_queue=problem.initial_state.sat_queue.copy(),
        gu_ema=problem.initial_state.gu_ema.copy(),
        uav_ema=problem.initial_state.uav_ema.copy(),
        sat_ema=problem.initial_state.sat_ema.copy(),
    )
    rewards: list[float] = []
    gu_queues: list[list[float]] = []
    uav_queues: list[list[float]] = []
    sat_queue_sum: list[float] = []
    gu_services: list[list[float]] = []
    uav_services: list[float] = []
    sat_processed_sum: list[float] = []
    drop_sum_trace: list[float] = []
    for t, idx in enumerate(np.asarray(x_indices, dtype=np.int64).reshape(-1)):
        stage = problem.stages[min(t, len(problem.stages) - 1)]
        access_bits = np.asarray(stage.access_bits_by_grid[int(idx)], dtype=np.float64)
        arrival = np.asarray(stage.arrival_bits, dtype=np.float64)

        q_gu_before = state.gu_queue + arrival
        gu_service = np.minimum(q_gu_before, access_bits)
        q_gu_after_raw = q_gu_before - gu_service
        gu_drop = np.maximum(q_gu_after_raw - float(cfg.queue_max_gu), 0.0)
        state.gu_queue = np.asarray(
            _quantize_queue_contract_np(
                cfg,
                np.minimum(q_gu_after_raw, float(cfg.queue_max_gu)),
                attr="structured_queue_state_quantum",
                default=128.0,
            ),
            dtype=np.float64,
        )

        uav_in = np.asarray([float(np.sum(gu_service, dtype=np.float64))], dtype=np.float64)
        q_uav_before = state.uav_queue + uav_in
        uav_service = np.minimum(q_uav_before, np.asarray([stage.relay_cap_bits], dtype=np.float64))
        q_uav_after_raw = q_uav_before - uav_service
        uav_drop = np.maximum(q_uav_after_raw - float(cfg.queue_max_uav), 0.0)
        state.uav_queue = np.asarray(
            _quantize_queue_contract_np(
                cfg,
                np.minimum(q_uav_after_raw, float(cfg.queue_max_uav)),
                attr="structured_queue_state_quantum",
                default=128.0,
            ),
            dtype=np.float64,
        )

        sat_incoming = np.asarray(
            _quantize_queue_contract_np(
                cfg,
                stage.relay_share * float(uav_service[0]),
                attr="structured_flow_bits_quantum",
                default=32.0,
            ),
            dtype=np.float64,
        )
        q_sat_before = state.sat_queue + sat_incoming
        sat_processed = np.minimum(q_sat_before, stage.sat_compute_cap_bits)
        q_sat_after_raw = q_sat_before - sat_processed
        sat_drop = np.maximum(q_sat_after_raw - float(cfg.queue_max_sat), 0.0)
        state.sat_queue = np.asarray(
            _quantize_queue_contract_np(
                cfg,
                np.minimum(q_sat_after_raw, float(cfg.queue_max_sat)),
                attr="structured_queue_state_quantum",
                default=128.0,
            ),
            dtype=np.float64,
        )

        gu_cost, uav_cost, sat_cost = _device_costs_from_ema(cfg, state, stage.selected_sat_indices)
        workload_after = _weighted_total(gu_cost, uav_cost, sat_cost, state.gu_queue, state.uav_queue, state.sat_queue)
        drop_cost = _weighted_total(gu_cost, uav_cost, sat_cost, gu_drop, uav_drop, sat_drop)
        reward = float(-workload_after - drop_cost)

        decay = float(np.clip(float(getattr(cfg, "bw_weighted_workload_ema_decay", 0.95) or 0.0), 0.0, 1.0))
        fresh = 1.0 - decay
        state.gu_ema = decay * state.gu_ema + fresh * gu_service
        state.uav_ema = decay * state.uav_ema + fresh * uav_service
        state.sat_ema = decay * state.sat_ema + fresh * sat_processed

        rewards.append(reward)
        gu_queues.append(state.gu_queue.tolist())
        uav_queues.append(state.uav_queue.tolist())
        sat_queue_sum.append(float(np.sum(state.sat_queue, dtype=np.float64)))
        gu_services.append(gu_service.tolist())
        uav_services.append(float(uav_service[0]))
        sat_processed_sum.append(float(np.sum(sat_processed, dtype=np.float64)))
        drop_sum_trace.append(float(np.sum(gu_drop, dtype=np.float64) + np.sum(uav_drop, dtype=np.float64) + np.sum(sat_drop, dtype=np.float64)))

    return FlowRollout(
        reward_sum=float(np.sum(rewards, dtype=np.float64)),
        rewards=rewards,
        gu_queue=gu_queues,
        uav_queue=uav_queues,
        sat_queue_sum=sat_queue_sum,
        gu_service=gu_services,
        uav_service=uav_services,
        sat_processed_sum=sat_processed_sum,
        drop_sum=drop_sum_trace,
    )


def optimize_flow_bcd(
    cfg,
    problem: FlowProblem,
    x_grid: np.ndarray,
    starts: dict[str, np.ndarray],
    *,
    passes: int,
    block_size: int,
) -> dict[str, Any]:
    grid_n = int(x_grid.size)
    horizon = len(problem.stages)
    block = max(1, min(int(block_size), horizon))
    best_name = ""
    best_seq = None
    best_rollout = None
    traces: dict[str, list[dict[str, float]]] = {}
    for name, start in starts.items():
        seq = np.asarray(start, dtype=np.int64).reshape(-1).copy()
        if seq.size != horizon:
            seq = np.resize(seq, horizon).astype(np.int64, copy=False)
        seq = np.clip(seq, 0, grid_n - 1)
        current = simulate_flow(cfg, problem, seq)
        trace = [{"pass": -1.0, "pred_reward": float(current.reward_sum)}]
        for pass_idx in range(max(int(passes), 0)):
            improved_any = False
            for start_t in range(0, horizon):
                end_t = min(horizon, start_t + block)
                width = end_t - start_t
                local_best_seq = seq
                local_best = current
                for combo in itertools.product(range(grid_n), repeat=width):
                    if np.array_equal(seq[start_t:end_t], np.asarray(combo, dtype=np.int64)):
                        continue
                    candidate = seq.copy()
                    candidate[start_t:end_t] = np.asarray(combo, dtype=np.int64)
                    cand_rollout = simulate_flow(cfg, problem, candidate)
                    if cand_rollout.reward_sum > local_best.reward_sum + 1.0e-9:
                        local_best_seq = candidate
                        local_best = cand_rollout
                if local_best.reward_sum > current.reward_sum + 1.0e-9:
                    seq = local_best_seq
                    current = local_best
                    improved_any = True
            trace.append({"pass": float(pass_idx), "pred_reward": float(current.reward_sum)})
            if not improved_any:
                break
        traces[name] = trace
        if best_rollout is None or current.reward_sum > best_rollout.reward_sum:
            best_name = name
            best_seq = seq.copy()
            best_rollout = current
    if best_seq is None or best_rollout is None:
        raise RuntimeError("Flow BCD did not produce a candidate.")
    return {
        "best_start": best_name,
        "x_indices": best_seq,
        "x_values": np.asarray(x_grid, dtype=np.float64)[best_seq],
        "predicted": best_rollout,
        "traces": traces,
    }


def collect_full_horizon_problem(cfg, *, seed: int, x_grid: np.ndarray) -> FlowProblem:
    group = make_structured_driver_group(cfg, num_envs=1)
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        stages: list[StageModel] = []
        initial: FlowState | None = None
        for _t in range(int(cfg.T_steps)):
            _prepare_bw_stage(driver, cfg)
            if initial is None:
                initial = _flow_state_from_driver(driver)
            stage = _build_stage_model(driver, cfg, x_grid)
            uniform = _uniform_action(driver, cfg)
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                uniform,
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            stage.arrival_bits = np.asarray(driver.env.last_gu_arrival, dtype=np.float64).copy()
            stages.append(stage)
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
        if initial is None:
            raise RuntimeError("No BW stage collected.")
        return FlowProblem(stages=stages, initial_state=initial)
    finally:
        close_structured_env_group(group)


def replay_native_sequence(cfg, *, seed: int, stages: list[StageModel], x_values: np.ndarray) -> NativeRollout:
    group = make_structured_driver_group(cfg, num_envs=1)
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        reward_sum = 0.0
        level_sum = 0.0
        actions: list[list[float]] = []
        x_by_t: list[float] = []
        arrivals: list[list[float]] = []
        gu_queue: list[list[float]] = []
        reward_by_step: list[float] = []
        for t, x in enumerate(np.asarray(x_values, dtype=np.float64).reshape(-1)):
            _prepare_bw_stage(driver, cfg)
            action = _action_from_stage_model(cfg, stages[min(t, len(stages) - 1)], float(x))
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                action,
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            reward = float(step_result.team_reward)
            reward_sum += reward
            level_sum += float(step_result.bw_weighted_workload_level_reward)
            actions.append(np.asarray(action[0], dtype=np.float64).tolist())
            x_by_t.append(float(x))
            arrivals.append(np.asarray(driver.env.last_gu_arrival, dtype=np.float64).tolist())
            gu_queue.append(np.asarray(driver.env.gu_queue, dtype=np.float64).tolist())
            reward_by_step.append(reward)
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
        return NativeRollout(
            reward_sum=float(reward_sum),
            level_reward_sum=float(level_sum),
            actions=actions,
            x_by_t=x_by_t,
            arrivals=arrivals,
            gu_queue=gu_queue,
            reward_by_step=reward_by_step,
            final_gu_queue=np.asarray(driver.env.gu_queue, dtype=np.float64).tolist(),
            final_uav_queue=np.asarray(driver.env.uav_queue, dtype=np.float64).tolist(),
            final_sat_queue_sum=float(np.sum(np.asarray(driver.env.sat_queue, dtype=np.float64))),
        )
    finally:
        close_structured_env_group(group)


def replay_native_policy(cfg, *, seed: int, policy_name: str) -> NativeRollout:
    group = make_structured_driver_group(cfg, num_envs=1)
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        reward_sum = 0.0
        level_sum = 0.0
        actions: list[list[float]] = []
        x_by_t: list[float] = []
        arrivals: list[list[float]] = []
        gu_queue: list[list[float]] = []
        reward_by_step: list[float] = []
        for _t in range(int(cfg.T_steps)):
            _prepare_bw_stage(driver, cfg)
            if policy_name == "uniform":
                action = _uniform_action(driver, cfg)
            elif policy_name == "queue_aware_bw":
                obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
                action = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
            else:
                raise ValueError(f"Unknown native policy: {policy_name}")
            x_by_t.append(_x_from_action(driver, cfg, action))
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                action,
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            reward = float(step_result.team_reward)
            reward_sum += reward
            level_sum += float(step_result.bw_weighted_workload_level_reward)
            actions.append(np.asarray(action[0], dtype=np.float64).tolist())
            arrivals.append(np.asarray(driver.env.last_gu_arrival, dtype=np.float64).tolist())
            gu_queue.append(np.asarray(driver.env.gu_queue, dtype=np.float64).tolist())
            reward_by_step.append(reward)
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
        return NativeRollout(
            reward_sum=float(reward_sum),
            level_reward_sum=float(level_sum),
            actions=actions,
            x_by_t=x_by_t,
            arrivals=arrivals,
            gu_queue=gu_queue,
            reward_by_step=reward_by_step,
            final_gu_queue=np.asarray(driver.env.gu_queue, dtype=np.float64).tolist(),
            final_uav_queue=np.asarray(driver.env.uav_queue, dtype=np.float64).tolist(),
            final_sat_queue_sum=float(np.sum(np.asarray(driver.env.sat_queue, dtype=np.float64))),
        )
    finally:
        close_structured_env_group(group)


def replay_causal_mpc(
    cfg,
    *,
    seed: int,
    x_grid: np.ndarray,
    mpc_horizon: int,
    passes: int,
    block_size: int,
) -> NativeRollout:
    group = make_structured_driver_group(cfg, num_envs=1)
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        reward_sum = 0.0
        level_sum = 0.0
        actions: list[list[float]] = []
        x_by_t: list[float] = []
        arrivals: list[list[float]] = []
        gu_queue: list[list[float]] = []
        reward_by_step: list[float] = []
        uniform_idx = int(_nearest_grid_indices(x_grid, np.asarray([0.5]))[0])
        for _t in range(int(cfg.T_steps)):
            _prepare_bw_stage(driver, cfg)
            stage = _build_stage_model(driver, cfg, x_grid, arrival_bits=_expected_arrival_rates(driver.env, cfg))
            initial = _flow_state_from_driver(driver)
            horizon = max(1, min(int(mpc_horizon), int(cfg.T_steps) - int(driver.env.t)))
            problem = FlowProblem(stages=[stage for _ in range(horizon)], initial_state=initial)
            obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
            q_action = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
            q_x = _x_from_action(driver, cfg, q_action)
            q_idx = int(_nearest_grid_indices(x_grid, np.asarray([q_x]))[0])
            starts = {
                "uniform": np.full((horizon,), uniform_idx, dtype=np.int64),
                "queue_aware_repeat": np.full((horizon,), q_idx, dtype=np.int64),
            }
            opt = optimize_flow_bcd(
                cfg,
                problem,
                x_grid,
                starts,
                passes=passes,
                block_size=block_size,
            )
            x0 = float(np.asarray(opt["x_values"], dtype=np.float64).reshape(-1)[0])
            action = _action_from_x(driver, cfg, x0)
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                action,
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            reward = float(step_result.team_reward)
            reward_sum += reward
            level_sum += float(step_result.bw_weighted_workload_level_reward)
            actions.append(np.asarray(action[0], dtype=np.float64).tolist())
            x_by_t.append(x0)
            arrivals.append(np.asarray(driver.env.last_gu_arrival, dtype=np.float64).tolist())
            gu_queue.append(np.asarray(driver.env.gu_queue, dtype=np.float64).tolist())
            reward_by_step.append(reward)
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
        return NativeRollout(
            reward_sum=float(reward_sum),
            level_reward_sum=float(level_sum),
            actions=actions,
            x_by_t=x_by_t,
            arrivals=arrivals,
            gu_queue=gu_queue,
            reward_by_step=reward_by_step,
            final_gu_queue=np.asarray(driver.env.gu_queue, dtype=np.float64).tolist(),
            final_uav_queue=np.asarray(driver.env.uav_queue, dtype=np.float64).tolist(),
            final_sat_queue_sum=float(np.sum(np.asarray(driver.env.sat_queue, dtype=np.float64))),
        )
    finally:
        close_structured_env_group(group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow-model BCD oracle/MPC for the 1-UAV/2-GU BW sanity env.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--traffic-model", default="", help="Optional traffic_model override, e.g. homogeneous.")
    parser.add_argument("--tag", default="", help="Optional suffix for the output filename.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if str(args.traffic_model).strip():
        cfg.traffic_model = str(args.traffic_model).strip()
    if int(cfg.num_uav) != 1 or int(cfg.num_gu) != 2:
        raise ValueError(f"This script expects num_uav=1 and num_gu=2, got {cfg.num_uav}/{cfg.num_gu}.")
    if bool(getattr(cfg, "deadline_enabled", False)):
        raise ValueError("This flow model does not implement deadline expiry; use deadline_enabled=false.")
    step = float(args.grid_step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("--grid-step must be in (0, 1].")
    count = int(round(1.0 / step))
    x_grid = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)

    uniform = replay_native_policy(cfg, seed=int(args.seed), policy_name="uniform")
    queue_aware = replay_native_policy(cfg, seed=int(args.seed), policy_name="queue_aware_bw")
    full_problem = collect_full_horizon_problem(cfg, seed=int(args.seed), x_grid=x_grid)

    uniform_idx = int(_nearest_grid_indices(x_grid, np.asarray([0.5]))[0])
    queue_indices = _nearest_grid_indices(x_grid, np.asarray(queue_aware.x_by_t, dtype=np.float64))
    starts = {
        "uniform": np.full((len(full_problem.stages),), uniform_idx, dtype=np.int64),
        "queue_aware_bw": queue_indices,
    }
    full_opt = optimize_flow_bcd(
        cfg,
        full_problem,
        x_grid,
        starts,
        passes=int(args.passes),
        block_size=int(args.block_size),
    )
    full_native = replay_native_sequence(
        cfg,
        seed=int(args.seed),
        stages=full_problem.stages,
        x_values=np.asarray(full_opt["x_values"], dtype=np.float64),
    )
    causal_mpc = replay_causal_mpc(
        cfg,
        seed=int(args.seed),
        x_grid=x_grid,
        mpc_horizon=int(args.mpc_horizon),
        passes=int(args.passes),
        block_size=int(args.block_size),
    )

    payload = {
        "config": str(Path(args.config)),
        "seed": int(args.seed),
        "traffic_model": str(getattr(cfg, "traffic_model", "")),
        "method": {
            "grid": x_grid,
            "full_horizon": "flow-model block coordinate descent with fixed realized arrival tape and precomputed native access-rate curves",
            "causal_mpc": "receding-horizon flow BCD using current state/current channel curve and expected arrivals; only first action is executed",
            "final_evidence": "compare methods by native environment replay reward, not by flow predicted objective alone",
        },
        "uniform": uniform,
        "queue_aware_bw": queue_aware,
        "full_horizon_flow_bcd": {
            "predicted": full_opt["predicted"],
            "x_by_t": full_opt["x_values"],
            "best_start": full_opt["best_start"],
            "traces": full_opt["traces"],
            "native_replay": full_native,
        },
        "causal_flow_mpc": causal_mpc,
        "deltas": {
            "full_native_vs_queue_aware": float(full_native.reward_sum - queue_aware.reward_sum),
            "full_native_vs_uniform": float(full_native.reward_sum - uniform.reward_sum),
            "mpc_native_vs_queue_aware": float(causal_mpc.reward_sum - queue_aware.reward_sum),
            "mpc_native_vs_uniform": float(causal_mpc.reward_sum - uniform.reward_sum),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip()
    traffic_tag = str(getattr(cfg, "traffic_model", "traffic")).strip().replace(" ", "_")
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"flow_bcd_seed{int(args.seed)}_{traffic_tag}_grid{step:g}{suffix}.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
