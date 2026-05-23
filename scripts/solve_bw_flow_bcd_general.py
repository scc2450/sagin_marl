from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import _quantize_queue_contract_np
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_eval import (
    _evaluate_structured_action_replay_with_traces,
    _evaluate_structured_baseline_policy_with_traces,
    evaluate_structured_fixed_policy,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


DEFAULT_CONFIG = "configs/tmp/structured_bw_sanity_1uav_8gu_t10_quadhot_currentenv_competition_ppo.yaml"
DEFAULT_OUT_DIR = "artifacts/bw_oracle/flow_bcd_general"


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
    action_names: list[str]
    action_gid_library: np.ndarray
    access_bits_by_action: np.ndarray
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
    action_names: list[str]
    arrivals: list[list[float]]
    gu_queue: list[list[float]]
    reward_by_step: list[float]
    final_gu_queue: list[float]
    final_uav_queue: list[float]
    final_sat_queue_sum: float
    processed_ratio_eval: float = 0.0
    drop_ratio_eval: float = 0.0
    pre_backlog_steps_eval: float = 0.0


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


def _prepare_bw_stage(driver, cfg, *, access_gain_tape: np.ndarray | None = None) -> None:
    driver.begin_step()
    driver.run_accel_stage(
        np.zeros((int(cfg.num_uav), 2), dtype=np.float32),
        access_gain_override=access_gain_tape,
    )
    # With fixed_satellite_strategy=true, the staged driver ignores this dummy
    # action and selects the nearest visible satellite, matching sat zero.
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


def _normalize_gid_weights(cfg, weights: np.ndarray, valid_gids: np.ndarray) -> np.ndarray:
    out = np.zeros((int(cfg.num_gu),), dtype=np.float64)
    valid = np.asarray(valid_gids, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < int(cfg.num_gu))]
    if valid.size == 0:
        return out
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == int(cfg.num_gu):
        out[valid] = np.maximum(arr[valid], 0.0)
    elif arr.size == valid.size:
        out[valid] = np.maximum(arr, 0.0)
    denom = float(np.sum(out[valid], dtype=np.float64))
    if denom > 1.0e-12:
        out[valid] /= denom
    else:
        out[valid] = 1.0 / float(valid.size)
    return out


def _action_from_gid_weights(cfg, candidate_indices: np.ndarray, valid_mask: np.ndarray, gid_weights: np.ndarray) -> np.ndarray:
    action = np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
    weights = np.asarray(gid_weights, dtype=np.float64).reshape(-1)
    for slot, gu_idx in enumerate(np.asarray(candidate_indices, dtype=np.int64)):
        if bool(valid_mask[slot]) and 0 <= int(gu_idx) < weights.size:
            action[0, slot] = float(max(weights[int(gu_idx)], 0.0))
    denom = float(np.sum(action[0, valid_mask], dtype=np.float64)) if np.any(valid_mask) else 0.0
    if denom > 1.0e-12:
        action[0, valid_mask] /= denom
    elif np.any(valid_mask):
        action[0, valid_mask] = 1.0 / float(np.sum(valid_mask))
    return action


def _semantic_action_from_gid_weights(cfg, gid_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(gid_weights, dtype=np.float64).reshape(-1)
    semantic = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
    if int(cfg.num_uav) != 1:
        raise NotImplementedError("The general flow BCD oracle currently supports native replay for num_uav=1.")
    if weights.size == int(cfg.num_gu):
        row = np.maximum(weights, 0.0)
    else:
        row = np.zeros((int(cfg.num_gu),), dtype=np.float64)
        row[: min(row.size, weights.size)] = np.maximum(weights[: min(row.size, weights.size)], 0.0)
    denom = float(np.sum(row, dtype=np.float64))
    if denom > 1.0e-12:
        row = row / denom
    elif int(cfg.num_gu) > 0:
        row[:] = 1.0 / float(int(cfg.num_gu))
    semantic[0, : int(cfg.num_gu)] = row.astype(np.float32, copy=False)
    return semantic


def _uniform_semantic_action(cfg) -> np.ndarray:
    weights = np.ones((int(cfg.num_gu),), dtype=np.float64)
    return _semantic_action_from_gid_weights(cfg, weights)


def _gid_weights_from_action(driver, cfg, action: np.ndarray) -> np.ndarray:
    candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    act = np.asarray(action, dtype=np.float64).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
    weights = np.zeros((int(cfg.num_gu),), dtype=np.float64)
    for slot, gu_idx in enumerate(candidate_indices):
        if bool(valid_mask[slot]) and 0 <= int(gu_idx) < int(cfg.num_gu):
            weights[int(gu_idx)] += max(float(act[0, slot]), 0.0)
    valid_gids = candidate_indices[valid_mask]
    return _normalize_gid_weights(cfg, weights, valid_gids)


def _weights_from_values(cfg, valid_gids: np.ndarray, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.zeros((int(cfg.num_gu),), dtype=np.float64)
    valid = np.asarray(valid_gids, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < int(cfg.num_gu))]
    if valid.size == 0:
        return weights
    if arr.size == int(cfg.num_gu):
        weights[valid] = np.maximum(arr[valid], 0.0)
    else:
        weights[valid] = np.maximum(arr[: valid.size], 0.0)
    return _normalize_gid_weights(cfg, weights, valid)


def _add_action(actions: list[np.ndarray], names: list[str], name: str, weights: np.ndarray) -> None:
    rounded = np.round(np.asarray(weights, dtype=np.float64), decimals=7)
    for existing in actions:
        if np.allclose(rounded, np.round(existing, decimals=7), rtol=0.0, atol=1.0e-7):
            return
    actions.append(np.asarray(weights, dtype=np.float64).copy())
    names.append(str(name))


def _build_action_library(
    cfg,
    *,
    valid_gids: np.ndarray,
    expected_arrival: np.ndarray,
    queue: np.ndarray,
    queue_aware_weights: np.ndarray | None,
    realized_arrival: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    valid = np.asarray(valid_gids, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < int(cfg.num_gu))]
    actions: list[np.ndarray] = []
    names: list[str] = []
    if valid.size == 0:
        return ["zero"], np.zeros((1, int(cfg.num_gu)), dtype=np.float64)

    uniform = np.zeros((int(cfg.num_gu),), dtype=np.float64)
    uniform[valid] = 1.0 / float(valid.size)
    _add_action(actions, names, "uniform", uniform)

    if queue_aware_weights is not None:
        _add_action(actions, names, "queue_aware", _normalize_gid_weights(cfg, queue_aware_weights, valid))

    expected = np.asarray(expected_arrival, dtype=np.float64)
    _add_action(actions, names, "expected_arrival", _weights_from_values(cfg, valid, expected))
    _add_action(actions, names, "queue_plus_expected", _weights_from_values(cfg, valid, np.asarray(queue, dtype=np.float64) + expected))

    if realized_arrival is not None:
        realized = np.asarray(realized_arrival, dtype=np.float64)
        _add_action(actions, names, "realized_arrival", _weights_from_values(cfg, valid, realized))

    rank_values = np.asarray(realized_arrival if realized_arrival is not None else expected_arrival, dtype=np.float64)
    order = valid[np.argsort(-rank_values[valid], kind="stable")]
    for k in (1, 2, 4):
        kk = min(int(k), int(order.size))
        if kk <= 0:
            continue
        top = order[:kk]
        eq = np.zeros((int(cfg.num_gu),), dtype=np.float64)
        eq[top] = 1.0 / float(kk)
        _add_action(actions, names, f"top{kk}_equal", eq)
        prop = np.zeros((int(cfg.num_gu),), dtype=np.float64)
        prop[top] = np.maximum(rank_values[top], 0.0)
        _add_action(actions, names, f"top{kk}_prop", _normalize_gid_weights(cfg, prop, valid))

    for gid in valid.tolist():
        one = np.zeros((int(cfg.num_gu),), dtype=np.float64)
        one[int(gid)] = 1.0
        _add_action(actions, names, f"onehot_gu{int(gid)}", one)

    return names, np.stack(actions, axis=0).astype(np.float64, copy=False)


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


def _access_bits_for_actions(driver, cfg, stage: StageModel, action_gid_library: np.ndarray) -> np.ndarray:
    access_bits = np.zeros((int(action_gid_library.shape[0]), int(cfg.num_gu)), dtype=np.float64)
    for i, weights in enumerate(np.asarray(action_gid_library, dtype=np.float64)):
        action = _action_from_gid_weights(cfg, stage.candidate_indices, stage.valid_mask, weights)
        rates, _eta = driver.env._compute_access_rates(
            np.asarray(getattr(driver, "_stage_assoc"), dtype=np.int32),
            getattr(driver, "_stage_candidates"),
            action,
            record_exec=False,
            access_snapshot=getattr(driver, "_stage_access_gain_matrix"),
        )
        bits = np.asarray(rates, dtype=np.float64) * float(cfg.tau0)
        access_bits[i] = np.asarray(
            _quantize_queue_contract_np(cfg, bits, attr="structured_flow_bits_quantum", default=32.0),
            dtype=np.float64,
        )
    return access_bits


def _build_stage_model(
    driver,
    cfg,
    *,
    arrival_bits: np.ndarray | None = None,
    realized_action_library: bool = False,
) -> StageModel:
    candidate_indices, valid_mask = _stage_candidate_arrays(driver, cfg)
    valid_gids = candidate_indices[valid_mask]
    expected_arrival = _expected_arrival_rates(driver.env, cfg)
    arrival = expected_arrival.copy() if arrival_bits is None else np.asarray(arrival_bits, dtype=np.float64).copy()

    obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
    queue_aware = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
    queue_aware_weights = _gid_weights_from_action(driver, cfg, queue_aware)
    names, action_gid_library = _build_action_library(
        cfg,
        valid_gids=valid_gids,
        expected_arrival=expected_arrival,
        queue=np.asarray(driver.env.gu_queue, dtype=np.float64),
        queue_aware_weights=queue_aware_weights,
        realized_arrival=arrival if realized_action_library else None,
    )

    dummy_stage = StageModel(
        action_names=names,
        action_gid_library=action_gid_library,
        access_bits_by_action=np.zeros((int(action_gid_library.shape[0]), int(cfg.num_gu)), dtype=np.float64),
        relay_cap_bits=0.0,
        relay_share=np.zeros((int(cfg.num_sat),), dtype=np.float64),
        sat_compute_cap_bits=np.zeros((int(cfg.num_sat),), dtype=np.float64),
        arrival_bits=arrival,
        expected_arrival_bits=expected_arrival,
        selected_sat_indices=np.zeros((0,), dtype=np.int64),
        candidate_indices=candidate_indices,
        valid_mask=valid_mask,
    )
    access_bits = _access_bits_for_actions(driver, cfg, dummy_stage, action_gid_library)

    rate_matrix, _sat_loads = driver.env._compute_backhaul_rates(
        np.asarray(getattr(driver, "_stage_sat_pos"), dtype=np.float32),
        np.asarray(getattr(driver, "_stage_sat_vel"), dtype=np.float32),
        np.asarray(getattr(driver, "_stage_sat_selection_matrix"), dtype=np.int64),
    )
    rates = np.asarray(rate_matrix, dtype=np.float64)[0]
    relay_bits_raw = float(np.sum(rates, dtype=np.float64) * float(cfg.tau0))
    relay_cap = float(
        np.asarray(
            _quantize_queue_contract_np(cfg, relay_bits_raw, attr="structured_flow_bits_quantum", default=32.0),
            dtype=np.float64,
        ).reshape(-1)[0]
    )
    relay_share = rates / float(np.sum(rates, dtype=np.float64)) if relay_bits_raw > 0.0 else np.zeros((int(cfg.num_sat),), dtype=np.float64)

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

    return StageModel(
        action_names=names,
        action_gid_library=action_gid_library,
        access_bits_by_action=access_bits,
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


def _weighted_total(
    gu_cost: np.ndarray,
    uav_cost: np.ndarray,
    sat_cost: np.ndarray,
    gu: np.ndarray,
    uav: np.ndarray,
    sat: np.ndarray,
) -> float:
    return float(
        np.sum(gu_cost * gu, dtype=np.float64)
        + np.sum(uav_cost * uav, dtype=np.float64)
        + np.sum(sat_cost * sat, dtype=np.float64)
    )


def simulate_flow(cfg, problem: FlowProblem, action_indices: np.ndarray) -> FlowRollout:
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
    for t, idx in enumerate(np.asarray(action_indices, dtype=np.int64).reshape(-1)):
        stage = problem.stages[min(t, len(problem.stages) - 1)]
        action_idx = int(np.clip(int(idx), 0, int(stage.access_bits_by_action.shape[0]) - 1))
        access_bits = np.asarray(stage.access_bits_by_action[action_idx], dtype=np.float64)
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


def _idx_by_name(stage: StageModel, preferred: str) -> int:
    for i, name in enumerate(stage.action_names):
        if name == preferred:
            return i
    for i, name in enumerate(stage.action_names):
        if name.startswith(preferred):
            return i
    return 0


def optimize_flow_bcd(
    cfg,
    problem: FlowProblem,
    starts: dict[str, np.ndarray],
    *,
    passes: int,
) -> dict[str, Any]:
    horizon = len(problem.stages)
    best_name = ""
    best_seq = None
    best_rollout = None
    traces: dict[str, list[dict[str, float]]] = {}
    for name, start in starts.items():
        seq = np.asarray(start, dtype=np.int64).reshape(-1).copy()
        if seq.size != horizon:
            seq = np.resize(seq, horizon).astype(np.int64, copy=False)
        for t, stage in enumerate(problem.stages):
            seq[t] = int(np.clip(seq[t], 0, int(stage.access_bits_by_action.shape[0]) - 1))
        current = simulate_flow(cfg, problem, seq)
        trace = [{"pass": -1.0, "pred_reward": float(current.reward_sum)}]
        for pass_idx in range(max(int(passes), 0)):
            improved_any = False
            for t, stage in enumerate(problem.stages):
                local_best_seq = seq
                local_best = current
                for action_idx in range(int(stage.access_bits_by_action.shape[0])):
                    if int(seq[t]) == int(action_idx):
                        continue
                    candidate = seq.copy()
                    candidate[t] = int(action_idx)
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
        "action_indices": best_seq,
        "action_names": [problem.stages[t].action_names[int(best_seq[t])] for t in range(horizon)],
        "predicted": best_rollout,
        "traces": traces,
    }


def collect_full_horizon_problem(cfg, *, seed: int) -> FlowProblem:
    native_tape_rollout, _reset_rollouts = _native_uniform_tape_action_rollout(cfg, seed=int(seed))
    arrival_tape = [
        np.asarray(trace.get("arrival_tape"), dtype=np.float64).copy()
        for trace in native_tape_rollout
        if trace.get("arrival_tape") is not None
    ]

    group = make_structured_driver_group(cfg, num_envs=1)
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        stages: list[StageModel] = []
        initial: FlowState | None = None
        for _t in range(int(cfg.T_steps)):
            if _t >= len(native_tape_rollout):
                break
            trace = native_tape_rollout[_t]
            access_gain_tape = (
                None
                if trace.get("access_gain_tape") is None
                else np.asarray(trace.get("access_gain_tape"), dtype=np.float32).copy()
            )
            _prepare_bw_stage(driver, cfg, access_gain_tape=access_gain_tape)
            if initial is None:
                initial = _flow_state_from_driver(driver)
            if _t >= len(arrival_tape):
                break
            stage = _build_stage_model(
                driver,
                cfg,
                arrival_bits=arrival_tape[_t],
                realized_action_library=True,
            )
            stages.append(stage)
            uniform = _action_from_gid_weights(
                cfg,
                stage.candidate_indices,
                stage.valid_mask,
                stage.action_gid_library[_idx_by_name(stage, "uniform")],
            )
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                uniform,
                arrival_override=arrival_tape[_t],
                doppler_residual_after_override=trace.get("doppler_residual_after_tape"),
                traffic_state_after_override=trace.get("traffic_state_after_tape"),
                bw_link_transition_override=trace.get("bw_link_transition_tape"),
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
        if initial is None:
            raise RuntimeError("No BW stage collected.")
        return FlowProblem(stages=stages, initial_state=initial)
    finally:
        close_structured_env_group(group)


def _starts_for_problem(problem: FlowProblem) -> dict[str, np.ndarray]:
    starts: dict[str, np.ndarray] = {}
    for name in ("uniform", "queue_aware", "expected_arrival", "queue_plus_expected", "realized_arrival"):
        starts[name] = np.asarray([_idx_by_name(stage, name) for stage in problem.stages], dtype=np.int64)
    return starts


def _native_uniform_tape_action_rollout(cfg, *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
    (
        _base_summary,
        _base_rows,
        _base_traces,
        base_action_rollouts,
        reset_rollouts,
    ) = _evaluate_structured_baseline_policy_with_traces(
        cfg,
        baseline_policy="uniform_bw",
        episodes=1,
        episode_seed_base=int(seed),
        random_tape_cfg=cfg,
        exec_sources=("zero", "zero", "policy"),
    )
    if not base_action_rollouts:
        raise RuntimeError("Native random-tape baseline did not produce an action rollout.")
    return list(base_action_rollouts[0]), reset_rollouts


def _native_rollout_from_eval(
    cfg,
    *,
    summary: dict[str, float],
    trace_rows: list[dict[str, float]],
    semantic_actions: list[np.ndarray],
    action_names: list[str],
) -> NativeRollout:
    del cfg
    reward_by_step = [float(row.get("reward", 0.0)) for row in trace_rows]
    gu_queue = [[float(row.get("gu_queue_sum", 0.0))] for row in trace_rows]
    final_trace = trace_rows[-1] if trace_rows else {}
    return NativeRollout(
        reward_sum=float(summary.get("reward_sum", 0.0)),
        level_reward_sum=float(summary.get("bw_weighted_workload_level_sum", 0.0)),
        actions=[np.asarray(action[0], dtype=np.float64).tolist() for action in semantic_actions],
        action_names=[str(name) for name in action_names],
        arrivals=[],
        gu_queue=gu_queue,
        reward_by_step=reward_by_step,
        final_gu_queue=[float(final_trace.get("gu_queue_sum", 0.0))] if final_trace else [],
        final_uav_queue=[float(final_trace.get("uav_queue_sum", 0.0))] if final_trace else [],
        final_sat_queue_sum=float(final_trace.get("sat_queue_sum", 0.0) or 0.0),
        processed_ratio_eval=float(summary.get("processed_ratio_eval", 0.0)),
        drop_ratio_eval=float(summary.get("drop_ratio_eval", 0.0)),
        pre_backlog_steps_eval=float(summary.get("pre_backlog_steps_eval", 0.0)),
    )


def replay_native_semantic_sequence(
    cfg,
    *,
    seed: int,
    semantic_actions: list[np.ndarray],
    action_names: list[str],
) -> NativeRollout:
    """Replay fixed semantic GU-share actions through the official native kernel path."""
    base_rollout, reset_rollouts = _native_uniform_tape_action_rollout(cfg, seed=int(seed))
    horizon = min(len(base_rollout), len(semantic_actions))
    action_traces: list[dict[str, np.ndarray]] = []
    for step_index in range(horizon):
        trace = copy.deepcopy(base_rollout[step_index])
        trace["bw_semantic_action"] = np.asarray(semantic_actions[step_index], dtype=np.float32).reshape(
            int(cfg.num_uav),
            int(cfg.num_gu),
        )
        action_traces.append(trace)
    summary, _rows, traces = _evaluate_structured_action_replay_with_traces(
        cfg,
        action_rollouts=[action_traces],
        reset_rollouts=reset_rollouts,
        episode_seed_base=int(seed),
        vec_backend="sync",
        exec_sources=("zero", "zero", "policy"),
    )
    trace_rows = traces[0] if traces else []
    return _native_rollout_from_eval(
        cfg,
        summary=summary,
        trace_rows=trace_rows,
        semantic_actions=semantic_actions[:horizon],
        action_names=action_names[:horizon],
    )


def replay_native_sequence(cfg, *, seed: int, stages: list[StageModel], action_indices: np.ndarray) -> NativeRollout:
    semantic_actions: list[np.ndarray] = []
    action_names: list[str] = []
    for t, action_idx_raw in enumerate(np.asarray(action_indices, dtype=np.int64).reshape(-1)):
        stage = stages[min(t, len(stages) - 1)]
        action_idx = int(np.clip(int(action_idx_raw), 0, int(stage.action_gid_library.shape[0]) - 1))
        semantic_actions.append(_semantic_action_from_gid_weights(cfg, stage.action_gid_library[action_idx]))
        action_names.append(str(stage.action_names[action_idx]))
    return replay_native_semantic_sequence(
        cfg,
        seed=int(seed),
        semantic_actions=semantic_actions,
        action_names=action_names,
    )


def replay_native_policy(cfg, *, seed: int, policy_name: str) -> NativeRollout:
    policy = str(policy_name).strip().lower()
    if policy == "uniform":
        semantic_actions = [_uniform_semantic_action(cfg) for _ in range(int(cfg.T_steps))]
        return replay_native_semantic_sequence(
            cfg,
            seed=int(seed),
            semantic_actions=semantic_actions,
            action_names=["uniform" for _ in semantic_actions],
        )
    if policy == "queue_aware_bw":
        summary = evaluate_structured_fixed_policy(
            cfg,
            baseline_policy="queue_aware_bw",
            episodes=1,
            episode_seed_base=int(seed),
            num_envs=1,
        )
        return NativeRollout(
            reward_sum=float(summary.get("reward_sum", 0.0)),
            level_reward_sum=float(summary.get("bw_weighted_workload_level_sum", 0.0)),
            actions=[],
            action_names=["queue_aware_bw"],
            arrivals=[],
            gu_queue=[],
            reward_by_step=[],
            final_gu_queue=[],
            final_uav_queue=[],
            final_sat_queue_sum=0.0,
            processed_ratio_eval=float(summary.get("processed_ratio_eval", 0.0)),
            drop_ratio_eval=float(summary.get("drop_ratio_eval", 0.0)),
            pre_backlog_steps_eval=float(summary.get("pre_backlog_steps_eval", 0.0)),
        )
    raise ValueError(f"Unknown native policy: {policy_name}")


def replay_causal_mpc(cfg, *, seed: int, mpc_horizon: int, passes: int) -> NativeRollout:
    native_tape_rollout, _reset_rollouts = _native_uniform_tape_action_rollout(cfg, seed=int(seed))
    group = make_structured_driver_group(cfg, num_envs=1)
    semantic_actions: list[np.ndarray] = []
    action_names: list[str] = []
    try:
        group.reset_many([int(seed)])
        driver = group[0]
        for _t in range(int(cfg.T_steps)):
            if _t >= len(native_tape_rollout):
                break
            trace = native_tape_rollout[_t]
            access_gain_tape = (
                None
                if trace.get("access_gain_tape") is None
                else np.asarray(trace.get("access_gain_tape"), dtype=np.float32).copy()
            )
            _prepare_bw_stage(driver, cfg, access_gain_tape=access_gain_tape)
            expected_bits = (
                _expected_arrival_rates(driver.env, cfg)
                if trace.get("arrival_rate_tape") is None
                else np.asarray(trace.get("arrival_rate_tape"), dtype=np.float64).copy()
            )
            stage = _build_stage_model(driver, cfg, arrival_bits=expected_bits)
            initial = _flow_state_from_driver(driver)
            horizon = max(1, min(int(mpc_horizon), int(cfg.T_steps) - int(driver.env.t)))
            problem = FlowProblem(stages=[stage for _ in range(horizon)], initial_state=initial)
            starts = _starts_for_problem(problem)
            opt = optimize_flow_bcd(cfg, problem, starts, passes=passes)
            action_idx = int(np.asarray(opt["action_indices"], dtype=np.int64).reshape(-1)[0])
            semantic_actions.append(_semantic_action_from_gid_weights(cfg, stage.action_gid_library[action_idx]))
            action_names.append(str(stage.action_names[action_idx]))
            action = _action_from_gid_weights(cfg, stage.candidate_indices, stage.valid_mask, stage.action_gid_library[action_idx])
            step_result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(
                action,
                arrival_override=trace.get("arrival_tape"),
                doppler_residual_after_override=trace.get("doppler_residual_after_tape"),
                traffic_state_after_override=trace.get("traffic_state_after_tape"),
                bw_link_transition_override=trace.get("bw_link_transition_tape"),
                capture_auxiliary_outputs=False,
                materialize_agent_dicts=False,
            )
            if bool(step_result.terminated or step_result.truncated or driver.env.t >= int(cfg.T_steps)):
                break
    finally:
        close_structured_env_group(group)
    return replay_native_semantic_sequence(
        cfg,
        seed=int(seed),
        semantic_actions=semantic_actions,
        action_names=action_names,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General 1-UAV/K-GU BW flow BCD oracle/MPC.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--traffic-model", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if str(args.traffic_model).strip():
        cfg.traffic_model = str(args.traffic_model).strip()
    if int(cfg.num_uav) != 1:
        raise ValueError(f"This script expects num_uav=1, got {cfg.num_uav}.")
    if bool(getattr(cfg, "deadline_enabled", False)):
        raise ValueError("This flow model does not implement deadline expiry; use deadline_enabled=false.")

    uniform = replay_native_policy(cfg, seed=int(args.seed), policy_name="uniform")
    queue_aware = replay_native_policy(cfg, seed=int(args.seed), policy_name="queue_aware_bw")
    full_problem = collect_full_horizon_problem(cfg, seed=int(args.seed))
    full_opt = optimize_flow_bcd(cfg, full_problem, _starts_for_problem(full_problem), passes=int(args.passes))
    full_native = replay_native_sequence(
        cfg,
        seed=int(args.seed),
        stages=full_problem.stages,
        action_indices=np.asarray(full_opt["action_indices"], dtype=np.int64),
    )
    causal_mpc = replay_causal_mpc(
        cfg,
        seed=int(args.seed),
        mpc_horizon=int(args.mpc_horizon),
        passes=int(args.passes),
    )

    payload = {
        "config": str(Path(args.config)),
        "seed": int(args.seed),
        "traffic_model": str(getattr(cfg, "traffic_model", "")),
        "method": {
            "full_horizon": (
                "Python flow-model coordinate descent over a finite per-stage action library; "
                "reported native_replay is official native main-kernel action replay"
            ),
            "causal_mpc": (
                "Python receding-horizon flow coordinate descent; reported reward is official "
                "native main-kernel replay of the resulting semantic BW sequence"
            ),
            "action_library": "uniform, queue-aware, expected/queue proportional, realized-arrival proportional for oracle, top-k, and one-hot actions",
            "final_evidence": "official native main-kernel replay reward; queue_aware_bw uses evaluate_structured_fixed_policy",
        },
        "uniform": uniform,
        "queue_aware_bw": queue_aware,
        "full_horizon_flow_bcd": {
            "predicted": full_opt["predicted"],
            "action_indices": full_opt["action_indices"],
            "action_names": full_opt["action_names"],
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
    out_path = out_dir / f"flow_bcd_general_seed{int(args.seed)}_{traffic_tag}{suffix}.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
