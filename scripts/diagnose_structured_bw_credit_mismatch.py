from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_rollout_debug import append_single_env_step
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _collate_dataclass,
    _current_obs_list,
    _heuristic_accel,
    _heuristic_sat,
    _refresh_stage_obs_cache,
    _sat_mask_to_ids,
    _to_device_dataclass,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--actor-checkpoint", type=str, required=True)
    parser.add_argument("--critic-checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument("--policy-mode", type=str, default="stochastic", choices=["deterministic", "stochastic"])
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--max-contexts", type=int, default=300)
    parser.add_argument("--cf-delta", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "p50": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float | None:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return None
    pred_rank = _rankdata_desc(np.asarray(pred, dtype=np.float64))
    truth_rank = _rankdata_desc(np.asarray(truth, dtype=np.float64))
    corr = _pearson(pred_rank.tolist(), truth_rank.tolist())
    return None if corr is None else float(corr)


def _pairwise_concordance_desc(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0e-9) -> float | None:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return None
    total = 0.0
    hits = 0.0
    for i in range(pred.size):
        for j in range(i + 1, pred.size):
            pred_diff = float(pred[i] - pred[j])
            truth_diff = float(truth[i] - truth[j])
            if abs(pred_diff) <= eps and abs(truth_diff) <= eps:
                total += 1.0
                hits += 1.0
                continue
            if abs(pred_diff) <= eps or abs(truth_diff) <= eps:
                total += 1.0
                hits += 0.5
                continue
            total += 1.0
            if pred_diff * truth_diff > 0.0:
                hits += 1.0
    if total <= 0.0:
        return None
    return float(hits / total)


def _load_modules(
    cfg,
    actor_checkpoint: str,
    critic_checkpoint: str,
    *,
    device: torch.device,
    hidden_dim: int,
    embed_dim: int,
) -> tuple[StructuredMAPPO, Any, Any]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=hidden_dim, embed_dim=embed_dim)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    load_checkpoint_forgiving(actor, actor_checkpoint, map_location=device, strict=False)
    load_checkpoint_forgiving(critic, critic_checkpoint, map_location=device, strict=False)
    actor.eval()
    critic.eval()
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=float(getattr(cfg, "actor_lr", 1.0e-4) or 1.0e-4)),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), lr=float(getattr(cfg, "critic_lr", 1.0e-4) or 1.0e-4)
        ),
        device=device,
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        cfg=cfg,
        train_accel=bool(True if getattr(cfg, "train_accel", None) is None else getattr(cfg, "train_accel")),
        train_sat=bool(True if getattr(cfg, "train_sat", None) is None else getattr(cfg, "train_sat")),
        train_bw=bool(True if getattr(cfg, "train_bw", None) is None else getattr(cfg, "train_bw")),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    return learner, actor, critic


def _bw_valid_and_effective_loc(local_state: Any, loc_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.asarray(local_state.user_mask > 0.5, dtype=bool).reshape(-1)
        & np.asarray(local_state.bw_valid_mask > 0.5, dtype=bool).reshape(-1)
    )
    effective = np.zeros_like(np.asarray(loc_row, dtype=np.float32).reshape(-1))
    if int(np.sum(valid)) <= 0:
        return valid, effective
    latent = valid.copy()
    if int(np.sum(valid)) > 1:
        ref_idx = int(np.max(np.flatnonzero(valid)))
        latent[ref_idx] = False
    effective[latent] = np.asarray(loc_row, dtype=np.float32).reshape(-1)[latent]
    return valid, effective


def _reallocate_toward_slot(
    base_action: np.ndarray,
    valid_mask: np.ndarray,
    target_slot: int,
    delta: float,
    *,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray | None, float]:
    valid = np.asarray(valid_mask, dtype=bool)
    target = int(target_slot)
    if not valid[target]:
        return None, 0.0
    donor_mask = valid.copy()
    donor_mask[target] = False
    donor_mass = float(np.sum(np.asarray(base_action, dtype=np.float64)[donor_mask]))
    if donor_mass <= eps:
        return None, 0.0
    used_delta = min(float(delta), 0.5 * donor_mass)
    if used_delta <= eps:
        return None, 0.0
    out = np.asarray(base_action, dtype=np.float32).copy()
    scale = float((donor_mass - used_delta) / max(donor_mass, eps))
    out[donor_mask] = out[donor_mask] * scale
    out[target] = float(out[target] + used_delta)
    out[~valid] = 0.0
    norm = float(np.sum(out[valid]))
    if norm <= eps:
        return None, 0.0
    out[valid] = out[valid] / norm
    return out.astype(np.float32, copy=False), float(used_delta)


def _step_reward_and_parts(driver: StructuredControlDriver, bw_action: np.ndarray) -> tuple[float, dict[str, float]]:
    step_result = driver.execute_stage_bw_and_step(bw_action)
    reward = float(next(iter(step_result.rewards.values())))
    parts_raw = dict(getattr(driver.env, "last_reward_parts", {}) or {})
    parts: dict[str, float] = {}
    for key, value in parts_raw.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            parts[str(key)] = float(value)
    parts.setdefault("reward_raw", reward)
    parts.setdefault("processed_ratio_eval", 0.0)
    parts.setdefault("drop_ratio_eval", 0.0)
    parts.setdefault("pre_backlog_steps_eval", 0.0)
    return reward, parts


def _normalize_advantages_like_training(advantages: np.ndarray, stage_ids: np.ndarray, stagewise_norm: bool) -> np.ndarray:
    adv = torch.from_numpy(np.asarray(advantages, dtype=np.float32)).clone()
    if adv.numel() <= 1:
        return adv.cpu().numpy().astype(np.float32, copy=False)
    if stagewise_norm:
        stage_ids_np = np.asarray(stage_ids, dtype=np.int64)
        for stage_id in (0, 1, 2):
            stage_idx_np = np.flatnonzero(stage_ids_np == stage_id)
            if stage_idx_np.size <= 0:
                continue
            stage_idx = torch.as_tensor(stage_idx_np, dtype=torch.long)
            stage_adv = adv.index_select(0, stage_idx)
            stage_adv = (stage_adv - stage_adv.mean()) / stage_adv.std(unbiased=False).clamp_min(1.0e-8)
            adv.index_copy_(0, stage_idx, stage_adv)
    else:
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1.0e-8)
    return adv.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner, actor, _critic = _load_modules(
        cfg,
        args.actor_checkpoint,
        args.critic_checkpoint,
        device=device,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )

    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    buffer = StructuredRolloutBuffer()
    context_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    contexts_done = 0
    physical_step = 0
    deterministic = str(args.policy_mode).strip().lower() == "deterministic"

    try:
        for ep in range(int(args.episodes)):
            if contexts_done >= int(args.max_contexts):
                break
            seed = int(args.episode_seed_base) + ep
            env.reset(seed=seed)
            done = False
            while not done and contexts_done < int(args.max_contexts):
                z_accel = driver.begin_step()
                accel_world_batch = _to_device_dataclass(_collate_dataclass([z_accel], device), device)
                with torch.inference_mode():
                    v_accel = learner.critic.value_accel(accel_world_batch).detach().cpu().reshape(-1)[0]
                accel_local_states = driver.build_local_accel_states(z_accel)
                obs_start = _current_obs_list(driver)
                accel_action = _heuristic_accel(
                    obs_start,
                    cfg,
                    str(getattr(cfg, "exec_accel_source", "policy") or "policy").strip().lower(),
                    centers=getattr(driver.env, "gu_cluster_centers", None),
                    counts=getattr(driver.env, "gu_cluster_counts", None),
                )

                z_sat = driver.run_accel_stage(accel_action)
                sat_world_batch = _to_device_dataclass(_collate_dataclass([z_sat], device), device)
                with torch.inference_mode():
                    v_sat = learner.critic.value_sat(sat_world_batch).detach().cpu().reshape(-1)[0]
                sat_local_states = driver.build_sat_pair_candidates(z_sat)

                _refresh_stage_obs_cache(driver)
                obs_after_accel = _current_obs_list(driver)
                sat_mask = _heuristic_sat(
                    obs_after_accel,
                    cfg,
                    str(getattr(cfg, "exec_sat_source", "policy") or "policy").strip().lower(),
                )
                sat_action = _sat_mask_to_ids(driver, sat_mask)

                z_bw = driver.run_sat_stage(sat_action)
                bw_world_batch = _to_device_dataclass(_collate_dataclass([z_bw], device), device)
                with torch.inference_mode():
                    v_bw = learner.critic.value_bw(bw_world_batch).detach().cpu().reshape(-1)[0]

                bw_states = driver.build_bw_valid_context(z_bw)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.inference_mode():
                    bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
                bw_action = bw_out.action.detach().cpu().numpy().astype(np.float32, copy=False)
                bw_loc = bw_out.loc.detach().cpu().numpy().reshape(cfg.num_uav, -1)
                bw_logprob = float(bw_out.logprob.detach().cpu().reshape(-1).sum().item())

                pending_contexts: list[dict[str, Any]] = []
                if physical_step % max(int(args.step_stride), 1) == 0:
                    base_driver = copy.deepcopy(driver)
                    base_reward, base_parts = _step_reward_and_parts(base_driver, bw_action)
                    base_reward_raw = float(base_parts.get("reward_raw", base_reward))
                    base_processed = float(base_parts.get("processed_ratio_eval", 0.0))
                    base_drop = float(base_parts.get("drop_ratio_eval", 0.0))
                    base_backlog = float(base_parts.get("pre_backlog_steps_eval", 0.0))
                    for u in range(int(cfg.num_uav)):
                        if contexts_done >= int(args.max_contexts):
                            break
                        valid_mask, effective_loc = _bw_valid_and_effective_loc(bw_states[u], bw_loc[u])
                        valid_count = int(np.sum(valid_mask))
                        if valid_count <= 1:
                            continue
                        base_action_u = np.asarray(bw_action[u], dtype=np.float32)
                        loc_valid = np.asarray(effective_loc[valid_mask], dtype=np.float64)
                        action_valid = np.asarray(base_action_u[valid_mask], dtype=np.float64)
                        reward_credit: list[float] = []
                        reward_raw_credit: list[float] = []
                        processed_gain: list[float] = []
                        drop_gain: list[float] = []
                        backlog_gain: list[float] = []
                        deltas_used: list[float] = []
                        for target_slot in np.flatnonzero(valid_mask).tolist():
                            cf_action_u, used_delta = _reallocate_toward_slot(
                                base_action_u,
                                valid_mask,
                                int(target_slot),
                                float(args.cf_delta),
                            )
                            if cf_action_u is None or used_delta <= 0.0:
                                continue
                            cf_action = np.asarray(bw_action, dtype=np.float32).copy()
                            cf_action[u] = cf_action_u
                            cf_driver = copy.deepcopy(driver)
                            cf_reward, cf_parts = _step_reward_and_parts(cf_driver, cf_action)
                            reward_credit.append(float(cf_reward - base_reward))
                            reward_raw_credit.append(float(cf_parts.get("reward_raw", cf_reward) - base_reward_raw))
                            processed_gain.append(float(cf_parts.get("processed_ratio_eval", 0.0) - base_processed))
                            drop_gain.append(float(base_drop - cf_parts.get("drop_ratio_eval", 0.0)))
                            backlog_gain.append(float(base_backlog - cf_parts.get("pre_backlog_steps_eval", 0.0)))
                            deltas_used.append(float(used_delta))
                        if len(reward_credit) <= 1:
                            continue
                        reward_credit_arr = np.asarray(reward_credit, dtype=np.float64)
                        reward_raw_credit_arr = np.asarray(reward_raw_credit, dtype=np.float64)
                        processed_gain_arr = np.asarray(processed_gain, dtype=np.float64)
                        drop_gain_arr = np.asarray(drop_gain, dtype=np.float64)
                        backlog_gain_arr = np.asarray(backlog_gain, dtype=np.float64)
                        pending_contexts.append(
                            {
                                "episode": int(ep),
                                "seed": int(seed),
                                "step": int(physical_step),
                                "uav": int(u),
                                "valid_user_count": int(valid_count),
                                "policy_reward_step": float(base_reward),
                                "policy_reward_raw_step": float(base_reward_raw),
                                "policy_processed_step": float(base_processed),
                                "policy_drop_step": float(base_drop),
                                "policy_backlog_step": float(base_backlog),
                                "best_reward_step": float(base_reward + np.max(reward_credit_arr)),
                                "best_reward_raw_step": float(base_reward_raw + np.max(reward_raw_credit_arr)),
                                "best_processed_step": float(base_processed + np.max(processed_gain_arr)),
                                "best_drop_step": float(base_drop - np.max(drop_gain_arr)),
                                "best_backlog_step": float(base_backlog - np.max(backlog_gain_arr)),
                                "policy_local_reward_gap": float(np.max(reward_credit_arr)),
                                "policy_local_reward_raw_gap": float(np.max(reward_raw_credit_arr)),
                                "policy_local_processed_gap": float(np.max(processed_gain_arr)),
                                "policy_local_drop_gap": float(np.max(drop_gain_arr)),
                                "policy_local_backlog_gap": float(np.max(backlog_gain_arr)),
                                "positive_target_fraction": float(np.mean(reward_credit_arr > 0.0)),
                                "cf_delta_used_mean": _safe_mean(deltas_used),
                                "loc_reward_spearman": _safe_spearman_desc(loc_valid, reward_credit_arr),
                                "action_reward_spearman": _safe_spearman_desc(action_valid, reward_credit_arr),
                                "loc_reward_pairwise_acc": _pairwise_concordance_desc(loc_valid, reward_credit_arr),
                                "action_reward_pairwise_acc": _pairwise_concordance_desc(action_valid, reward_credit_arr),
                                "loc_reward_top1_hit": float(int(np.argmax(loc_valid)) == int(np.argmax(reward_credit_arr))),
                                "action_reward_top1_hit": float(
                                    int(np.argmax(action_valid)) == int(np.argmax(reward_credit_arr))
                                ),
                                "loc_reward_raw_spearman": _safe_spearman_desc(loc_valid, reward_raw_credit_arr),
                                "loc_processed_spearman": _safe_spearman_desc(loc_valid, processed_gain_arr),
                                "loc_drop_spearman": _safe_spearman_desc(loc_valid, drop_gain_arr),
                                "loc_backlog_spearman": _safe_spearman_desc(loc_valid, backlog_gain_arr),
                            }
                        )
                        contexts_done += 1

                if hasattr(driver, "execute_stage_bw_and_prepare_next_accel"):
                    step_result, next_world_state = driver.execute_stage_bw_and_prepare_next_accel(bw_action)
                else:
                    step_result = driver.execute_stage_bw_and_step(bw_action)
                    next_world_state = driver.begin_step()
                team_reward = float(next(iter(step_result.rewards.values())))
                terminated = bool(next(iter(step_result.terminations.values())))
                truncated = bool(next(iter(step_result.truncations.values())))
                zero_danger_target = torch.as_tensor(step_result.danger_imitation_target, dtype=torch.float32)
                zero_danger_mask = torch.zeros_like(torch.as_tensor(step_result.danger_imitation_mask, dtype=torch.float32))
                zero_danger_target = torch.zeros_like(zero_danger_target)
                append_single_env_step(
                    buffer,
                    env_index=0,
                    accel_world_state=z_accel,
                    sat_world_state=z_sat,
                    bw_world_state=z_bw,
                    next_world_state=next_world_state,
                    accel_local_actor_state=accel_local_states,
                    sat_local_actor_state=sat_local_states,
                    bw_local_actor_state=bw_states,
                    accel_action=torch.from_numpy(np.asarray(accel_action, dtype=np.float32)),
                    sat_action=torch.from_numpy(np.asarray(sat_action, dtype=np.int64)),
                    bw_action=torch.from_numpy(np.asarray(bw_action, dtype=np.float32)),
                    accel_old_logprob=torch.tensor(0.0, dtype=torch.float32),
                    sat_old_logprob=torch.tensor(0.0, dtype=torch.float32),
                    bw_old_logprob=torch.tensor(bw_logprob, dtype=torch.float32),
                    accel_value=torch.as_tensor(v_accel, dtype=torch.float32),
                    sat_value=torch.as_tensor(v_sat, dtype=torch.float32),
                    bw_value=torch.as_tensor(v_bw, dtype=torch.float32),
                    reward=team_reward,
                    terminated=terminated,
                    truncated=truncated,
                    accel_danger_imitation_target=zero_danger_target,
                    accel_danger_imitation_mask=zero_danger_mask,
                    bw_access_reward=float(getattr(step_result, "bw_access_reward", 0.0) or 0.0),
                    bw_flow_proxy_scores=(
                        None
                        if getattr(step_result, "bw_flow_proxy_scores", None) is None
                        else torch.as_tensor(step_result.bw_flow_proxy_scores, dtype=torch.float32)
                    ),
                    bw_flow_proxy_mask=(
                        None
                        if getattr(step_result, "bw_flow_proxy_mask", None) is None
                        else torch.as_tensor(step_result.bw_flow_proxy_mask, dtype=torch.float32)
                    ),
                    bw_old_logprob_per_agent=torch.as_tensor(
                        bw_out.logprob.detach().cpu().numpy(),
                        dtype=torch.float32,
                    ),
                )
                bw_transition_idx = len(buffer) - 1

                reward_parts = dict(getattr(driver.env, "last_reward_parts", {}) or {})
                rollout_rows.append(
                    {
                        "episode": int(ep),
                        "seed": int(seed),
                        "step": int(physical_step),
                        "transition_idx": int(bw_transition_idx),
                        "reward_sum": float(team_reward),
                        "processed_ratio_eval": float(reward_parts.get("processed_ratio_eval", 0.0)),
                        "drop_ratio_eval": float(reward_parts.get("drop_ratio_eval", 0.0)),
                        "pre_backlog_steps_eval": float(reward_parts.get("pre_backlog_steps_eval", 0.0)),
                    }
                )
                for row in pending_contexts:
                    row["transition_idx"] = int(bw_transition_idx)
                    context_rows.append(row)

                done = terminated or truncated
                physical_step += 1

        if len(buffer) <= 0:
            raise RuntimeError("No structured rollout transitions were collected.")
        rollout_views = buffer.build_rollout_views(learner.device)
        gae = learner.compute_returns_and_advantages(
            buffer,
            rollout_views.bootstrap_view,
            return_view=rollout_views.return_view,
        )
        adv_raw_np = np.asarray(gae["advantages"], dtype=np.float32)
        adv_final_np = _normalize_advantages_like_training(
            adv_raw_np,
            rollout_views.return_view.stage_ids,
            bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)),
        )
        returns_np = np.asarray(gae["returns"], dtype=np.float32)

        for row in context_rows:
            idx = int(row["transition_idx"])
            row["joint_adv_raw"] = float(adv_raw_np[idx])
            row["joint_adv_final"] = float(adv_final_np[idx])
            row["value_target"] = float(returns_np[idx])
            row["positive_joint_adv"] = int(row["joint_adv_final"] > 0.0)
            row["bad_local_reward_but_positive_adv"] = int(
                (float(row["policy_local_reward_gap"]) > 1.0e-6) and (float(row["joint_adv_final"]) > 0.0)
            )
            row["bad_local_processed_but_positive_adv"] = int(
                (float(row["policy_local_processed_gap"]) > 1.0e-6) and (float(row["joint_adv_final"]) > 0.0)
            )
            row["bad_local_drop_but_positive_adv"] = int(
                (float(row["policy_local_drop_gap"]) > 1.0e-6) and (float(row["joint_adv_final"]) > 0.0)
            )
            row["bad_local_backlog_but_positive_adv"] = int(
                (float(row["policy_local_backlog_gap"]) > 1.0e-6) and (float(row["joint_adv_final"]) > 0.0)
            )

        credit_mismatch = {
            "positive_joint_adv_fraction": _safe_mean([float(r["positive_joint_adv"]) for r in context_rows]),
            "bad_local_reward_but_positive_adv_fraction": _safe_mean(
                [float(r["bad_local_reward_but_positive_adv"]) for r in context_rows]
            ),
            "bad_local_processed_but_positive_adv_fraction": _safe_mean(
                [float(r["bad_local_processed_but_positive_adv"]) for r in context_rows]
            ),
            "bad_local_drop_but_positive_adv_fraction": _safe_mean(
                [float(r["bad_local_drop_but_positive_adv"]) for r in context_rows]
            ),
            "bad_local_backlog_but_positive_adv_fraction": _safe_mean(
                [float(r["bad_local_backlog_but_positive_adv"]) for r in context_rows]
            ),
            "corr_joint_adv_vs_policy_local_reward_gap": _pearson(
                [float(r["joint_adv_final"]) for r in context_rows],
                [float(r["policy_local_reward_gap"]) for r in context_rows],
            ),
            "corr_joint_adv_vs_policy_local_processed_gap": _pearson(
                [float(r["joint_adv_final"]) for r in context_rows],
                [float(r["policy_local_processed_gap"]) for r in context_rows],
            ),
            "corr_joint_adv_vs_policy_local_drop_gap": _pearson(
                [float(r["joint_adv_final"]) for r in context_rows],
                [float(r["policy_local_drop_gap"]) for r in context_rows],
            ),
            "corr_joint_adv_vs_policy_local_backlog_gap": _pearson(
                [float(r["joint_adv_final"]) for r in context_rows],
                [float(r["policy_local_backlog_gap"]) for r in context_rows],
            ),
        }
        local_alignment = {
            "loc_reward_spearman": _summarize(
                [float(r["loc_reward_spearman"]) for r in context_rows if r.get("loc_reward_spearman") is not None]
            ),
            "action_reward_spearman": _summarize(
                [float(r["action_reward_spearman"]) for r in context_rows if r.get("action_reward_spearman") is not None]
            ),
            "loc_reward_pairwise_acc": _summarize(
                [float(r["loc_reward_pairwise_acc"]) for r in context_rows if r.get("loc_reward_pairwise_acc") is not None]
            ),
            "action_reward_pairwise_acc": _summarize(
                [
                    float(r["action_reward_pairwise_acc"])
                    for r in context_rows
                    if r.get("action_reward_pairwise_acc") is not None
                ]
            ),
            "loc_reward_top1_hit": _summarize([float(r["loc_reward_top1_hit"]) for r in context_rows]),
            "action_reward_top1_hit": _summarize([float(r["action_reward_top1_hit"]) for r in context_rows]),
        }
        summary = {
            "config": str(Path(args.config).resolve()),
            "actor_checkpoint": str(Path(args.actor_checkpoint).resolve()),
            "critic_checkpoint": str(Path(args.critic_checkpoint).resolve()),
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base),
            "policy_mode": str(args.policy_mode),
            "step_stride": int(args.step_stride),
            "max_contexts": int(args.max_contexts),
            "cf_delta": float(args.cf_delta),
            "contexts_evaluated": int(len(context_rows)),
            "rollout_bw_steps": int(len(rollout_rows)),
            "credit_mismatch": credit_mismatch,
            "local_alignment": local_alignment,
            "rank_summary": {
                "policy_local_reward_gap": _summarize([float(r["policy_local_reward_gap"]) for r in context_rows]),
                "policy_local_processed_gap": _summarize([float(r["policy_local_processed_gap"]) for r in context_rows]),
                "policy_local_drop_gap": _summarize([float(r["policy_local_drop_gap"]) for r in context_rows]),
                "policy_local_backlog_gap": _summarize([float(r["policy_local_backlog_gap"]) for r in context_rows]),
                "valid_user_count": _summarize([float(r["valid_user_count"]) for r in context_rows]),
            },
            "worst_credit_mismatch_contexts": sorted(
                [row for row in context_rows if int(row.get("positive_joint_adv", 0)) == 1],
                key=lambda row: (
                    -float(row["policy_local_reward_gap"]),
                    -float(row["policy_local_drop_gap"]),
                    -float(row["policy_local_backlog_gap"]),
                ),
            )[:20],
            "context_csv": str((out_dir / "context_rows.csv").resolve()),
            "rollout_csv": str((out_dir / "rollout_rows.csv").resolve()),
            "summary_json": str((out_dir / "summary.json").resolve()),
        }

        with (out_dir / "context_rows.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(context_rows[0].keys()) if context_rows else ["episode", "seed", "step", "uav"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(context_rows)

        with (out_dir / "rollout_rows.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rollout_rows[0].keys()) if rollout_rows else ["episode", "seed", "step"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rollout_rows)

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
