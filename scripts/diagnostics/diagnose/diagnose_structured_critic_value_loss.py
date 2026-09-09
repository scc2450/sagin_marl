from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sagin_marl.rl.structured_mappo as structured_mappo
from sagin_marl.rl.structured_buffer import _index_dataclass_items
from sagin_marl.rl.structured_mappo import _to_device_dataclass


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1.0)),
        "p05": float(np.percentile(arr, 5.0)),
        "p25": float(np.percentile(arr, 25.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(arr.max()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--rollout_env_steps", type=int, required=True)
    parser.add_argument("--target_update", type=int, default=25)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--bw_train_target_mode", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    return parser


def _to_float_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _bootstrap_value_map(algo: Any, bootstrap_world_state: Any) -> dict[int, float]:
    if bootstrap_world_state is None:
        return {0: 0.0}
    if isinstance(bootstrap_world_state, dict):
        return {
            int(env_index): float(algo.bootstrap_value(world_state))
            for env_index, world_state in bootstrap_world_state.items()
        }
    return {0: float(algo.bootstrap_value(bootstrap_world_state))}


def _step_level_bw_return_decomposition(
    algo: Any,
    rollout_views: Any,
    bootstrap_world_state: Any,
) -> dict[int, dict[str, Any]]:
    training_stage_batches = getattr(rollout_views.training_view, "stage_batches", {})
    return_stage_batches = getattr(rollout_views.return_view, "stage_batches", {})
    accel_batch = training_stage_batches.get(0)
    sat_batch = training_stage_batches.get(1)
    bw_train_batch = training_stage_batches.get(2)
    bw_return_batch = return_stage_batches.get(2)
    if accel_batch is None or sat_batch is None or bw_train_batch is None or bw_return_batch is None:
        return {}
    bootstrap_values = _bootstrap_value_map(algo, bootstrap_world_state)
    time_limit_bootstrap_fn = getattr(algo, "_time_limit_bootstrap_values", None)
    truncated_bootstrap_values = (
        time_limit_bootstrap_fn(rollout_views.return_view)
        if callable(time_limit_bootstrap_fn)
        else {}
    )
    gamma = float(algo.gamma)
    gae_lambda = float(algo.gae_lambda)
    step_target_mode = str(algo.step_train_target_mode).strip().lower()
    bw_target_mode = str(algo.bw_train_target_mode).strip().lower()
    bw_return_mode = str(algo.bw_return_mode).strip().lower()
    if bw_return_mode in {"mc", "monte_carlo"}:
        bw_return_mode = "step_lambda_return"
    if str(algo.target_mode).strip().lower() != "step_level":
        return {}

    def step_train_reward_of(*, bw_access_reward: float, step_reward: float) -> float:
        if step_target_mode == "env_reward":
            return float(step_reward)
        if step_target_mode == "access_term":
            return float(algo.bw_reward_w_access) * float(bw_access_reward)
        return float(bw_access_reward)

    def bw_step_reward_of(
        *,
        bw_access_reward: float,
        bw_weighted_workload_delta_reward: float,
        bw_weighted_workload_level_reward: float,
        bw_gu_queue_level_reward: float,
        bw_system_queue_level_reward: float,
        bw_gu_service_queue_reward: float,
        step_reward: float,
        step_train_reward: float,
    ) -> float:
        if step_target_mode != "env_reward":
            return float(step_train_reward)
        if bw_target_mode == "env_reward":
            return float(step_reward)
        if bw_target_mode == "access_term":
            return float(algo.bw_reward_w_access) * float(bw_access_reward)
        if bw_target_mode == "weighted_workload_delta":
            return float(bw_weighted_workload_delta_reward)
        if bw_target_mode == "weighted_workload_level":
            return float(bw_weighted_workload_level_reward)
        if bw_target_mode == "gu_queue_level":
            return float(bw_gu_queue_level_reward)
        if bw_target_mode == "system_queue_level":
            return float(bw_system_queue_level_reward)
        if bw_target_mode == "gu_service_queue":
            return float(bw_gu_service_queue_reward)
        return float(bw_access_reward)

    def _positions_by_env(env_indices: Any) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for pos, env_index in enumerate(np.asarray(env_indices, dtype=np.int64).tolist()):
            out.setdefault(int(env_index), []).append(int(pos))
        return out

    accel_positions = _positions_by_env(accel_batch.env_indices)
    sat_positions = _positions_by_env(sat_batch.env_indices)
    bw_positions = _positions_by_env(bw_train_batch.env_indices)

    per_bw_idx: dict[int, dict[str, Any]] = {}
    for env_index, bw_env_positions in bw_positions.items():
        accel_env_positions = accel_positions.get(int(env_index), [])
        sat_env_positions = sat_positions.get(int(env_index), [])
        if not (
            len(accel_env_positions) == len(sat_env_positions) == len(bw_env_positions)
            and len(bw_env_positions) > 0
        ):
            continue
        next_step_return = 0.0
        next_step_accel_value = float(bootstrap_values.get(int(env_index), 0.0))
        next_bw_episode_return = 0.0
        env_records_by_step: dict[int, dict[str, Any]] = {}
        step_order: list[int] = []
        for group_offset in range(len(bw_env_positions) - 1, -1, -1):
            accel_pos = accel_env_positions[group_offset]
            bw_pos = bw_env_positions[group_offset]
            accel_idx = int(accel_batch.transition_indices[accel_pos])
            sat_idx = int(sat_batch.transition_indices[sat_env_positions[group_offset]])
            bw_idx = int(bw_train_batch.transition_indices[bw_pos])
            step_reward = float(bw_return_batch.rewards[bw_pos])
            bw_access_reward = float(bw_return_batch.bw_access_rewards[bw_pos])
            bw_weighted_workload_delta_reward = float(bw_return_batch.bw_weighted_workload_delta_rewards[bw_pos])
            bw_weighted_workload_level_reward = float(bw_return_batch.bw_weighted_workload_level_rewards[bw_pos])
            bw_gu_queue_level_reward = float(bw_return_batch.bw_gu_queue_level_rewards[bw_pos])
            bw_system_queue_level_reward = float(bw_return_batch.bw_system_queue_level_rewards[bw_pos])
            bw_gu_service_queue_reward = float(bw_return_batch.bw_gu_service_queue_rewards[bw_pos])
            step_train_reward = step_train_reward_of(
                bw_access_reward=bw_access_reward,
                step_reward=step_reward,
            )
            bw_step_reward = bw_step_reward_of(
                bw_access_reward=bw_access_reward,
                bw_weighted_workload_delta_reward=bw_weighted_workload_delta_reward,
                bw_weighted_workload_level_reward=bw_weighted_workload_level_reward,
                bw_gu_queue_level_reward=bw_gu_queue_level_reward,
                bw_system_queue_level_reward=bw_system_queue_level_reward,
                bw_gu_service_queue_reward=bw_gu_service_queue_reward,
                step_reward=step_reward,
                step_train_reward=step_train_reward,
            )
            terminated = bool(bw_return_batch.terminated[bw_pos])
            truncated = bool(bw_return_batch.truncated[bw_pos])
            timeout_bootstrap = float(truncated_bootstrap_values.get(int(bw_idx), 0.0))
            next_step_accel_value_in = float(next_step_accel_value)
            next_step_return_in = float(next_step_return)
            next_bw_episode_return_in = float(next_bw_episode_return)
            if terminated:
                step_return = float(step_train_reward)
                bw_step_return = float(bw_step_reward)
                contributions = {
                    "self_bw_reward": float(bw_step_reward),
                    "gamma_timeout_bootstrap": 0.0,
                    "gamma_1m_lambda_next_step_accel_value": 0.0,
                    "gamma_lambda_next_step_return": 0.0,
                    "gamma_next_step_return": 0.0,
                    "gamma_next_bw_episode_return": 0.0,
                }
                bootstrap_type = "terminated_zero"
                next_step_return = 0.0
                next_step_accel_value = 0.0
                next_bw_episode_return = 0.0
            elif truncated:
                step_return = float(step_train_reward + gamma * timeout_bootstrap)
                bw_step_return = float(bw_step_reward + gamma * timeout_bootstrap)
                contributions = {
                    "self_bw_reward": float(bw_step_reward),
                    "gamma_timeout_bootstrap": float(gamma * timeout_bootstrap),
                    "gamma_1m_lambda_next_step_accel_value": 0.0,
                    "gamma_lambda_next_step_return": 0.0,
                    "gamma_next_step_return": 0.0,
                    "gamma_next_bw_episode_return": 0.0,
                }
                bootstrap_type = "time_limit_bootstrap"
                next_step_return = 0.0
                next_step_accel_value = 0.0
                next_bw_episode_return = 0.0
            else:
                step_return = float(
                    step_train_reward
                    + gamma * ((1.0 - gae_lambda) * next_step_accel_value_in + gae_lambda * next_step_return_in)
                )
                if bw_return_mode == "gae":
                    bw_step_return = float(
                        bw_step_reward
                        + gamma * ((1.0 - gae_lambda) * next_step_accel_value_in + gae_lambda * next_step_return_in)
                    )
                    contributions = {
                        "self_bw_reward": float(bw_step_reward),
                        "gamma_timeout_bootstrap": 0.0,
                        "gamma_1m_lambda_next_step_accel_value": float(gamma * (1.0 - gae_lambda) * next_step_accel_value_in),
                        "gamma_lambda_next_step_return": float(gamma * gae_lambda * next_step_return_in),
                        "gamma_next_step_return": 0.0,
                        "gamma_next_bw_episode_return": 0.0,
                    }
                    bootstrap_type = "gae_mixed_with_next_step_return"
                elif bw_return_mode == "bw_episode_mc":
                    bw_step_return = float(bw_step_reward + gamma * next_bw_episode_return_in)
                    contributions = {
                        "self_bw_reward": float(bw_step_reward),
                        "gamma_timeout_bootstrap": 0.0,
                        "gamma_1m_lambda_next_step_accel_value": 0.0,
                        "gamma_lambda_next_step_return": 0.0,
                        "gamma_next_step_return": 0.0,
                        "gamma_next_bw_episode_return": float(gamma * next_bw_episode_return_in),
                    }
                    bootstrap_type = "bw_episode_mc"
                else:
                    bw_step_return = float(bw_step_reward + gamma * next_step_return_in)
                    contributions = {
                        "self_bw_reward": float(bw_step_reward),
                        "gamma_timeout_bootstrap": 0.0,
                        "gamma_1m_lambda_next_step_accel_value": 0.0,
                        "gamma_lambda_next_step_return": 0.0,
                        "gamma_next_step_return": float(gamma * next_step_return_in),
                        "gamma_next_bw_episode_return": 0.0,
                    }
                    bootstrap_type = "step_lambda_return"
                next_step_return = float(step_return)
                next_step_accel_value = float(accel_batch.values[accel_pos].reshape(-1)[0].item())
                next_bw_episode_return = float(bw_step_return)

            record = {
                "env_index": int(env_index),
                "env_step_index": int(group_offset),
                "buffer_indices": {
                    "accel": int(accel_idx),
                    "sat": int(sat_idx),
                    "bw": int(bw_idx),
                },
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_reward_env_total": float(step_reward),
                "step_train_reward": float(step_train_reward),
                "bw_step_reward": float(bw_step_reward),
                "timeout_bootstrap": float(timeout_bootstrap),
                "next_step_accel_value_input": float(next_step_accel_value_in),
                "next_step_return_input": float(next_step_return_in),
                "next_bw_episode_return_input": float(next_bw_episode_return_in),
                "step_return": float(step_return),
                "bw_step_return": float(bw_step_return),
                "bootstrap_type": str(bootstrap_type),
                "contributions": contributions,
            }
            env_records_by_step[int(group_offset)] = record
            step_order.append(int(group_offset))

        step_order.sort()
        for start_step in step_order:
            future_chain: list[dict[str, Any]] = []
            for step_idx in step_order:
                if step_idx < start_step:
                    continue
                rec = env_records_by_step[int(step_idx)]
                future_chain.append(
                    {
                        "env_step_index": int(rec["env_step_index"]),
                        "terminated": bool(rec["terminated"]),
                        "truncated": bool(rec["truncated"]),
                        "step_train_reward": float(rec["step_train_reward"]),
                        "bw_step_reward": float(rec["bw_step_reward"]),
                        "timeout_bootstrap": float(rec["timeout_bootstrap"]),
                        "next_step_accel_value_input": float(rec["next_step_accel_value_input"]),
                        "next_step_return_input": float(rec["next_step_return_input"]),
                        "step_return": float(rec["step_return"]),
                        "bw_step_return": float(rec["bw_step_return"]),
                        "bootstrap_type": str(rec["bootstrap_type"]),
                        "contributions": dict(rec["contributions"]),
                    }
                )
            env_records_by_step[int(start_step)]["future_chain_from_here"] = future_chain
            per_bw_idx[int(env_records_by_step[int(start_step)]["buffer_indices"]["bw"])] = env_records_by_step[int(start_step)]
    return per_bw_idx


def _decode_bw_action_by_gu(action: torch.Tensor, stage_candidates: list[list[int]], num_gu: int) -> dict[str, Any]:
    action_arr = np.asarray(action.detach().cpu(), dtype=np.float64)
    if action_arr.ndim == 1:
        action_arr = action_arr[None, :]
    per_uav: list[dict[str, Any]] = []
    gu_alloc = np.zeros((num_gu,), dtype=np.float64)
    for u in range(action_arr.shape[0]):
        slots = action_arr[u].reshape(-1)
        candidates_u = list(stage_candidates[u]) if u < len(stage_candidates) else []
        mapped_slots: list[dict[str, Any]] = []
        for slot, value in enumerate(slots.tolist()):
            gu_idx = int(candidates_u[slot]) if slot < len(candidates_u) else -1
            mapped_slots.append(
                {
                    "slot": int(slot),
                    "gu_idx": int(gu_idx),
                    "alloc": float(value),
                }
            )
            if 0 <= gu_idx < num_gu:
                gu_alloc[gu_idx] += float(value)
        per_uav.append({"uav": int(u), "slots": mapped_slots})
    return {
        "per_uav": per_uav,
        "gu_alloc": [float(x) for x in gu_alloc.tolist()],
    }


def _summarize_transition_state(
    *,
    world_state: Any,
    next_world_state: Any,
    bw_stage_state: Any,
    action: torch.Tensor,
    cfg: Any,
) -> dict[str, Any]:
    bw_stage_state = dict(bw_stage_state or {})
    env_state = dict(bw_stage_state.get("env_state") or {})

    gu_before_norm = (
        np.asarray(world_state.gu_nodes[0, :, 2], dtype=np.float64)
        if world_state.gu_nodes.shape[1] > 0
        else np.zeros((0,), dtype=np.float64)
    )
    gu_after_norm = (
        np.asarray(next_world_state.gu_nodes[0, :, 2], dtype=np.float64)
        if next_world_state.gu_nodes.shape[1] > 0
        else np.zeros((0,), dtype=np.float64)
    )
    gu_queue_before = np.asarray(
        env_state.get("gu_queue", gu_before_norm * float(cfg.queue_max_gu)),
        dtype=np.float64,
    ).reshape(-1)
    gu_queue_after = gu_after_norm * float(cfg.queue_max_gu)
    uav_queue_before = np.asarray(
        env_state.get(
            "uav_queue",
            np.asarray(world_state.uav_nodes[0, :, 5], dtype=np.float64) * float(cfg.queue_max_uav),
        ),
        dtype=np.float64,
    ).reshape(-1)
    uav_queue_after = np.asarray(next_world_state.uav_nodes[0, :, 5], dtype=np.float64).reshape(-1) * float(cfg.queue_max_uav)

    gu_obs_features: dict[str, list[float]] = {
        "queue_norm_before": _to_float_list(gu_before_norm),
        "queue_norm_after": _to_float_list(gu_after_norm),
        "queue_before": _to_float_list(gu_queue_before),
        "queue_after": _to_float_list(gu_queue_after),
    }
    feat_col = 3
    feature_specs = [
        ("obs_user_include_arrival_rate", "arrival_rate_proxy"),
        ("obs_user_include_recent_arrival", "recent_arrival_proxy"),
        ("obs_user_include_recent_service", "recent_service_proxy"),
        ("obs_user_include_queue_headroom", "queue_headroom_proxy"),
        ("obs_user_include_urgency_risk", "urgency_risk_proxy"),
        ("obs_user_include_downstream_pressure", "downstream_pressure_proxy"),
        ("obs_user_include_service_gap", "service_gap_proxy"),
        ("obs_user_include_service_gap_risk", "service_gap_risk_proxy"),
        ("obs_user_include_deadline_slack", "deadline_slack_proxy"),
        ("obs_user_include_deadline_risk", "deadline_risk_proxy"),
    ]
    for flag_name, feature_name in feature_specs:
        if bool(getattr(cfg, flag_name, False)):
            gu_obs_features[feature_name] = _to_float_list(world_state.gu_nodes[0, :, feat_col])
            feat_col += 1

    hot_mask = np.asarray(
        env_state.get("last_hotspot_mask", np.zeros((cfg.num_gu,), dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1)
    stage_candidates = bw_stage_state.get("stage_candidates") or []
    action_summary = _decode_bw_action_by_gu(action, stage_candidates, int(cfg.num_gu))

    return {
        "t_before": int(env_state.get("t", -1)),
        "time_frac_before": _to_float_list(world_state.uav_nodes[0, :, 6]),
        "hotspot_active_idx": int(env_state.get("_hotspot_active_idx", -1)),
        "hotspot_mask": _to_float_list(hot_mask),
        "association_before": [
            int(x)
            for x in np.asarray(
                env_state.get("last_association", np.full((cfg.num_gu,), -1, dtype=np.int32)),
                dtype=np.int64,
            ).reshape(-1).tolist()
        ],
        "prev_association_before": [
            int(x)
            for x in np.asarray(
                env_state.get("prev_association", np.full((cfg.num_gu,), -1, dtype=np.int32)),
                dtype=np.int64,
            ).reshape(-1).tolist()
        ],
        "uav_queue_before": _to_float_list(uav_queue_before),
        "uav_queue_after": _to_float_list(uav_queue_after),
        "last_gu_arrival_rate_vec": _to_float_list(
            env_state.get("last_gu_arrival_rate_vec", np.zeros((cfg.num_gu,), dtype=np.float32))
        ),
        "last_gu_arrival": _to_float_list(
            env_state.get("last_gu_arrival", np.zeros((cfg.num_gu,), dtype=np.float32))
        ),
        "last_gu_outflow_prev": _to_float_list(
            env_state.get("last_gu_outflow", np.zeros((cfg.num_gu,), dtype=np.float32))
        ),
        "stage_candidates": [[int(v) for v in row] for row in stage_candidates],
        "stage_bw_valid_mask": np.asarray(
            bw_stage_state.get("stage_bw_valid_mask", np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)),
            dtype=np.float64,
        ).tolist(),
        "bw_action": action_summary,
        "gu_state": gu_obs_features,
        "eta_ref_per_uav_gu": np.asarray(world_state.uav_gu_edges[0, :, :, 6], dtype=np.float64).tolist(),
        "candidate_flag_per_uav_gu": np.asarray(world_state.uav_gu_edges[0, :, :, 3], dtype=np.float64).tolist(),
        "bw_valid_flag_per_uav_gu": np.asarray(world_state.uav_gu_edges[0, :, :, 4], dtype=np.float64).tolist(),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        args.run_dir,
        f"critic_diag_update{int(args.target_update):04d}.json",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    original_update = structured_mappo.StructuredMAPPO.update
    update_counter = {"value": 0}

    def wrapped_update(self, buffer, bootstrap_world_state=None, *, rollout_views=None):
        update_counter["value"] += 1
        update_idx = int(update_counter["value"])
        if update_idx != int(args.target_update):
            return original_update(self, buffer, bootstrap_world_state, rollout_views=rollout_views)

        rollout_views = buffer.build_rollout_views(self.device) if rollout_views is None else rollout_views
        rollout_value_override = (
            self._rollout_value_override_from_training_view(rollout_views.training_view)
            if self.device.type == "cuda"
            and str(getattr(self, "structured_env_tensor_backend", "cpu") or "cpu").strip().lower() == "cuda"
            else None
        )
        gae = self.compute_returns_and_advantages(
            buffer,
            rollout_views.bootstrap_view if bootstrap_world_state is None else bootstrap_world_state,
            return_view=rollout_views.return_view,
            value_override=rollout_value_override,
        )
        returns_t = torch.from_numpy(gae["returns"]).to(self.device)
        bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
        bw_idx_np = (
            np.zeros((0,), dtype=np.int64)
            if bw_stage_batch is None
            else np.asarray(bw_stage_batch.transition_indices, dtype=np.int64)
        )
        if bw_idx_np.size == 0:
            payload = {
                "update": int(update_idx),
                "error": "no_bw_samples_in_target_update",
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return original_update(self, buffer, bootstrap_world_state, rollout_views=rollout_views)

        bw_idx_t = torch.as_tensor(bw_idx_np, device=self.device, dtype=torch.long)
        bw_returns_t = returns_t.index_select(0, bw_idx_t)
        values_for_diag = (
            torch.as_tensor(np.asarray(rollout_views.return_view.values, dtype=np.float32), device=self.device)
            if rollout_value_override is None
            else rollout_value_override.to(self.device, dtype=torch.float32)
        )
        bw_old_values_t = values_for_diag.index_select(0, bw_idx_t)
        world_batch = bw_stage_batch.world_batch
        with torch.no_grad():
            bw_pred_pre_t = self._stage_value_eval_from_batch(2, world_batch)

        bw_rewards = np.asarray(rollout_views.return_view.bw_weighted_workload_level_rewards, dtype=np.float64)[bw_idx_np]
        bw_returns = bw_returns_t.detach().cpu().numpy().astype(np.float64)
        bw_old_values = bw_old_values_t.detach().cpu().numpy().astype(np.float64)
        bw_pred_pre = bw_pred_pre_t.detach().cpu().numpy().astype(np.float64)
        err_old = bw_returns - bw_old_values
        err_pre = bw_returns - bw_pred_pre
        bw_return_decomp = _step_level_bw_return_decomposition(self, rollout_views, bootstrap_world_state)
        bw_return_batch = rollout_views.return_view.stage_batches[2]
        top_abs_error_indices = np.argsort(np.abs(err_pre))[::-1][:10]
        top_abs_error_samples: list[dict[str, Any]] = []
        for rank, rel_idx in enumerate(top_abs_error_indices, start=1):
            buffer_index = int(bw_idx_np[int(rel_idx)])
            top_abs_error_samples.append(
                {
                    "rank": int(rank),
                    "buffer_index": int(buffer_index),
                    "env_index": int(bw_stage_batch.env_indices[int(rel_idx)]),
                    "reward_step": float(bw_return_batch.rewards[int(rel_idx)]),
                    "bw_level_reward_step": float(bw_return_batch.bw_weighted_workload_level_rewards[int(rel_idx)]),
                    "return_target": float(bw_returns[int(rel_idx)]),
                    "value_old": float(bw_old_values[int(rel_idx)]),
                    "value_pred_pre": float(bw_pred_pre[int(rel_idx)]),
                    "error_old": float(err_old[int(rel_idx)]),
                    "error_pre": float(err_pre[int(rel_idx)]),
                    "abs_error_pre": float(abs(err_pre[int(rel_idx)])),
                    "state_summary": _summarize_transition_state(
                        world_state=_to_device_dataclass(
                            _index_dataclass_items(bw_stage_batch.world_batch, [int(rel_idx)]),
                            torch.device("cpu"),
                        ),
                        next_world_state=_to_device_dataclass(
                            _index_dataclass_items(bw_return_batch.next_world_batch, [int(rel_idx)]),
                            torch.device("cpu"),
                        ),
                        bw_stage_state=(
                            None
                            if bw_stage_batch.bw_stage_states is None
                            else bw_stage_batch.bw_stage_states[int(rel_idx)]
                        ),
                        action=bw_stage_batch.actions[int(rel_idx)],
                        cfg=self.cfg,
                    ),
                    "return_decomposition": bw_return_decomp.get(buffer_index),
                }
            )

        result = original_update(self, buffer, bootstrap_world_state, rollout_views=rollout_views)

        with torch.no_grad():
            bw_pred_post_t = self._stage_value_eval_from_batch(2, world_batch)
        bw_pred_post = bw_pred_post_t.detach().cpu().numpy().astype(np.float64)
        err_post = bw_returns - bw_pred_post

        payload = {
            "update": int(update_idx),
            "logged_value_loss": float(result.get("value_loss", 0.0)),
            "logged_value_loss_bw": float(result.get("value_loss_bw", 0.0)),
            "logged_explained_variance_bw": float(result.get("explained_variance_bw", 0.0)),
            "sample_count_bw": int(bw_idx_np.size),
            "bw_reward_step_stats": _stats(bw_rewards),
            "bw_return_stats": _stats(bw_returns),
            "bw_old_value_stats": _stats(bw_old_values),
            "bw_pred_pre_stats": _stats(bw_pred_pre),
            "bw_pred_post_stats": _stats(bw_pred_post),
            "bw_error_old_stats": _stats(err_old),
            "bw_error_pre_stats": _stats(err_pre),
            "bw_abs_error_pre_stats": _stats(np.abs(err_pre)),
            "bw_error_post_stats": _stats(err_post),
            "bw_abs_error_post_stats": _stats(np.abs(err_post)),
            "full_batch_mse_old_value": float(np.mean((bw_old_values - bw_returns) ** 2)),
            "full_batch_mse_pre": float(np.mean((bw_pred_pre - bw_returns) ** 2)),
            "full_batch_mse_post": float(np.mean((bw_pred_post - bw_returns) ** 2)),
            "top_abs_error_samples_pre": top_abs_error_samples,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote critic diagnostic to {out_path}")
        return result

    structured_mappo.StructuredMAPPO.update = wrapped_update
    try:
        import scripts.train_structured as train_structured

        sys.argv = [
            "train_structured.py",
            "--config",
            str(args.config),
            "--run_dir",
            str(args.run_dir),
            "--updates",
            str(int(args.updates)),
            "--rollout_env_steps",
            str(int(args.rollout_env_steps)),
            "--device",
            str(args.device),
            "--num_envs",
            str(int(args.num_envs)),
            "--vec_backend",
            str(args.vec_backend),
            "--torch_threads",
            str(int(args.torch_threads)),
        ]
        if args.bw_train_target_mode is not None:
            sys.argv.extend(["--bw_train_target_mode", str(args.bw_train_target_mode)])
        if args.hidden_dim is not None:
            sys.argv.extend(["--hidden_dim", str(int(args.hidden_dim))])
        if args.embed_dim is not None:
            sys.argv.extend(["--embed_dim", str(int(args.embed_dim))])
        train_structured.main()
    finally:
        structured_mappo.StructuredMAPPO.update = original_update


if __name__ == "__main__":
    main()
