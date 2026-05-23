from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import cluster_center_queue_aware_policy
from sagin_marl.rl.critic import CriticNet
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


CFG_PATH = Path(
    "configs/phase1_actions_curriculum_joint_3heads_fading_interference_vsat_precomp_satonly_nooverlap_satsup.yaml"
)
CURRENT_RUN_DIR = Path(
    "runs/phase1_actions/joint_3heads_fading_interference_ka_vsat_satonly_nooverlap_teacherab_satsup_u1200_subproc12_t2_20260329"
)
CURRENT_ACTOR_PATH = CURRENT_RUN_DIR / "actor_best.pt"
CURRENT_CRITIC_PATH = CURRENT_RUN_DIR / "critic_best.pt"
SATONLY_NOSUP_ACTOR_PATH = Path(
    "runs/phase1_actions/joint_3heads_fading_interference_ka_vsat_satonly_nooverlap_teacherab_u1200_subproc12_t2_20260329/actor_best.pt"
)
OLD_ACTOR_PATH = Path("runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/actor_best.pt")
OUTPUT_DIR = Path("runs/phase1_actions/frontend_critic_diag_20260330")

EPISODES = 8
SEED_BASE = 45100
ADV_CLIP_EPS = 1e-8


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "min": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 2:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float | None:
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = np.asarray(pred[mask], dtype=np.float64)
    target = np.asarray(target[mask], dtype=np.float64)
    if pred.size < 2:
        return None
    var_target = float(np.var(target))
    if var_target <= 1e-12:
        return None
    return float(1.0 - np.var(target - pred) / var_target)


def _get_state_batch(env: SaginParallelEnv) -> np.ndarray:
    cached_states = getattr(env, "last_state_batch", None)
    if cached_states is not None:
        return np.asarray(cached_states, dtype=np.float32)
    if hasattr(env, "get_global_state_batch"):
        return np.asarray(env.get_global_state_batch(), dtype=np.float32)
    return np.expand_dims(np.asarray(env.get_global_state(), dtype=np.float32), axis=0)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    episode_boundaries: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = int(rewards.shape[0])
    advantages = np.zeros((T,), dtype=np.float32)
    next_adv = 0.0
    for t in reversed(range(T)):
        delta = float(rewards[t]) + float(gamma) * float(bootstrap_values[t]) - float(values[t])
        continue_mask = 1.0 if (t < T - 1 and not bool(episode_boundaries[t])) else 0.0
        next_adv = delta + float(gamma) * float(lam) * continue_mask * next_adv
        advantages[t] = next_adv
    returns = advantages + values
    return advantages, returns


def _compute_mc_returns(rewards: np.ndarray, boundaries: np.ndarray, gamma: float) -> np.ndarray:
    T = int(rewards.shape[0])
    mc = np.zeros((T,), dtype=np.float32)
    next_ret = 0.0
    for t in reversed(range(T)):
        if t == T - 1 or bool(boundaries[t]):
            next_ret = 0.0
        next_ret = float(rewards[t]) + float(gamma) * next_ret
        mc[t] = next_ret
    return mc


def _make_models(cfg) -> tuple[int, int]:
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=0)
    obs_dim = int(batch_flatten_obs(list(obs.values()), cfg).shape[1])
    state_dim = int(_get_state_batch(env)[0].shape[0])
    return obs_dim, state_dim


def _load_actor(cfg, obs_dim: int, checkpoint_path: Path) -> ActorNet:
    actor = ActorNet(obs_dim, cfg).to(torch.device("cpu"))
    info = load_checkpoint_forgiving(actor, str(checkpoint_path), map_location="cpu", strict=True)
    adapted = len(info.get("adapted_keys", []))
    if adapted:
        print(f"Loaded actor {checkpoint_path} with {adapted} adapted keys.")
    actor.eval()
    return actor


def _load_critic(cfg, obs_dim: int, state_dim: int, checkpoint_path: Path) -> CriticNet:
    critic = CriticNet(state_dim, obs_dim, int(cfg.num_uav), cfg).to(torch.device("cpu"))
    info = load_checkpoint_forgiving(critic, str(checkpoint_path), map_location="cpu", strict=True)
    adapted = len(info.get("adapted_keys", []))
    if adapted:
        print(f"Loaded critic {checkpoint_path} with {adapted} adapted keys.")
    critic.eval()
    return critic


def _teacher_action_bundle(actor: ActorNet, obs_batch: np.ndarray, deterministic: bool) -> tuple[np.ndarray, np.ndarray | None]:
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        out = actor.act(obs_tensor, deterministic=deterministic)
    accel = out.accel.cpu().numpy() if out.accel is not None else np.zeros((obs_batch.shape[0], 2), dtype=np.float32)
    bw = out.bw_logits.cpu().numpy() if out.bw_logits is not None else None
    return accel, bw


def _policy_sat_bundle(actor: ActorNet, obs_batch: np.ndarray, deterministic: bool) -> np.ndarray | None:
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        out = actor.act(obs_tensor, deterministic=deterministic)
    return out.sat_logits.cpu().numpy() if out.sat_logits is not None else None


def _critic_value(critic: CriticNet, state_batch: np.ndarray, obs_step: np.ndarray) -> float:
    state_t = torch.tensor(state_batch, dtype=torch.float32, device=torch.device("cpu"))
    obs_t = torch.tensor(obs_step, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        value = critic(state_t, obs_t).detach().cpu().numpy().reshape(-1)
    return float(value[0])


def _strategy_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "current_best",
            "kind": "actor",
            "actor_path": CURRENT_ACTOR_PATH,
        },
        {
            "name": "satonly_nosup_best",
            "kind": "actor",
            "actor_path": SATONLY_NOSUP_ACTOR_PATH,
        },
        {
            "name": "old_actor_best_at_new_env",
            "kind": "actor",
            "actor_path": OLD_ACTOR_PATH,
        },
        {
            "name": "baseline_cluster_center_queue_aware",
            "kind": "baseline",
            "actor_path": None,
        },
    ]


def _build_actions(
    strategy: dict[str, Any],
    cfg,
    env: SaginParallelEnv,
    obs: dict[str, dict[str, np.ndarray]],
    actor_map: dict[str, ActorNet],
    teacher_actor: ActorNet,
) -> dict[str, dict[str, np.ndarray]]:
    if strategy["kind"] == "baseline":
        accel_actions, bw_logits, sat_logits = cluster_center_queue_aware_policy(
            list(obs.values()),
            cfg,
            getattr(env, "gu_cluster_centers", None),
            getattr(env, "gu_cluster_counts", None),
        )
        return assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)

    obs_batch = batch_flatten_obs(list(obs.values()), cfg)
    actor = actor_map[strategy["name"]]
    teacher_accel, teacher_bw = _teacher_action_bundle(
        teacher_actor,
        obs_batch,
        deterministic=bool(getattr(cfg, "exec_teacher_deterministic", True)),
    )
    policy_sat = _policy_sat_bundle(actor, obs_batch, deterministic=True)

    accel_actions = teacher_accel if str(getattr(cfg, "exec_accel_source", "policy")).strip().lower() == "teacher" else np.zeros((len(env.agents), 2), dtype=np.float32)
    bw_logits = teacher_bw if str(getattr(cfg, "exec_bw_source", "policy")).strip().lower() == "teacher" else None
    sat_logits = policy_sat if str(getattr(cfg, "exec_sat_source", "policy")).strip().lower() == "policy" else None
    return assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)


def _collect_rollout_rows(
    cfg,
    strategy: dict[str, Any],
    actor_map: dict[str, ActorNet],
    teacher_actor: ActorNet,
    critic: CriticNet,
    episodes: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    env = make_structured_env(cfg, mode="script")
    rows: list[dict[str, Any]] = []
    gamma = float(cfg.gamma)
    gae_lambda = float(cfg.gae_lambda)
    adv_clip = float(getattr(cfg, "adv_clip", 5.0) or 0.0)

    for ep in range(episodes):
        seed = seed_base + ep
        obs, _ = env.reset(seed=seed)
        done = False
        ep_rows: list[dict[str, Any]] = []
        step = 0
        while not done:
            state_batch = _get_state_batch(env)
            obs_flat = batch_flatten_obs(list(obs.values()), cfg).astype(np.float32, copy=False)
            obs_step = np.expand_dims(obs_flat, axis=0)
            value = _critic_value(critic, state_batch, obs_step)

            actions = _build_actions(strategy, cfg, env, obs, actor_map, teacher_actor)
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            reward = float(list(rewards.values())[0])
            terminated = bool(list(terms.values())[0])
            truncated = bool(list(truncs.values())[0])
            done = terminated or truncated

            next_state_batch = _get_state_batch(env)
            next_obs_flat = batch_flatten_obs(list(next_obs.values()), cfg).astype(np.float32, copy=False)
            next_obs_step = np.expand_dims(next_obs_flat, axis=0)
            next_value = 0.0 if terminated else _critic_value(critic, next_state_batch, next_obs_step)

            parts = dict(getattr(env, "last_reward_parts", {}) or {})
            assoc = np.asarray(getattr(env, "last_association", np.full((cfg.num_gu,), -1, dtype=np.int32)), dtype=np.int32)
            assoc_valid = assoc >= 0
            assoc_count = int(np.sum(assoc_valid))
            served_count = 0
            if hasattr(env, "last_gu_outflow"):
                gu_outflow = np.asarray(env.last_gu_outflow, dtype=np.float64)
                served_count = int(np.count_nonzero(gu_outflow > 1e-9))

            ep_rows.append(
                {
                    "strategy": strategy["name"],
                    "episode": ep,
                    "seed": seed,
                    "step": step,
                    "reward": reward,
                    "value_pred": value,
                    "next_value_pred": next_value,
                    "terminated": float(terminated),
                    "truncated": float(truncated),
                    "gu_drop_ratio_step": _to_float(parts.get("gu_drop_ratio_step")),
                    "g_pre": _to_float(parts.get("g_pre")),
                    "throughput_access_norm": _to_float(parts.get("throughput_access_norm")),
                    "assoc_ratio": _to_float(parts.get("assoc_ratio")),
                    "assoc_count": float(assoc_count),
                    "assoc_count_ratio": float(assoc_count) / max(float(cfg.num_gu), 1.0),
                    "served_user_count": float(served_count),
                    "served_user_ratio": float(served_count) / max(float(cfg.num_gu), 1.0),
                    "arrival_sum": _to_float(parts.get("arrival_sum")),
                    "outflow_sum": _to_float(parts.get("outflow_sum")),
                    "gu_queue_arrival_steps": _to_float(parts.get("gu_queue_arrival_steps")),
                }
            )
            obs = next_obs
            step += 1

        rewards_arr = np.asarray([row["reward"] for row in ep_rows], dtype=np.float32)
        values_arr = np.asarray([row["value_pred"] for row in ep_rows], dtype=np.float32)
        next_values_arr = np.asarray([row["next_value_pred"] for row in ep_rows], dtype=np.float32)
        boundaries_arr = np.asarray(
            [bool(row["terminated"] > 0.5 or row["truncated"] > 0.5) for row in ep_rows],
            dtype=bool,
        )
        advantages_raw, lambda_returns = _compute_gae(
            rewards_arr,
            values_arr,
            next_values_arr,
            boundaries_arr,
            gamma,
            gae_lambda,
        )
        mc_returns = _compute_mc_returns(rewards_arr, boundaries_arr, gamma)

        adv_mean = float(np.mean(advantages_raw))
        adv_std = float(np.std(advantages_raw))
        adv_preclip = (advantages_raw - adv_mean) / (adv_std + ADV_CLIP_EPS)
        if adv_clip > 0.0:
            advantages_final = np.clip(adv_preclip, -adv_clip, adv_clip)
        else:
            advantages_final = adv_preclip

        td_errors = rewards_arr + gamma * next_values_arr - values_arr
        td_errors = np.where(np.asarray([row["terminated"] > 0.5 for row in ep_rows], dtype=bool), rewards_arr - values_arr, td_errors)

        for idx, row in enumerate(ep_rows):
            row["adv_raw"] = float(advantages_raw[idx])
            row["adv_final"] = float(advantages_final[idx])
            row["lambda_return"] = float(lambda_returns[idx])
            row["mc_return"] = float(mc_returns[idx])
            row["td_error"] = float(td_errors[idx])

        rows.extend(ep_rows)

    return rows


def _quantile_bucket_rows(
    strategy: str,
    signal_name: str,
    signal_values: np.ndarray,
    adv_values: np.ndarray,
    return_values: np.ndarray,
    bucket_count: int = 5,
) -> list[dict[str, Any]]:
    mask = np.isfinite(signal_values) & np.isfinite(adv_values) & np.isfinite(return_values)
    signal_values = np.asarray(signal_values[mask], dtype=np.float64)
    adv_values = np.asarray(adv_values[mask], dtype=np.float64)
    return_values = np.asarray(return_values[mask], dtype=np.float64)
    if signal_values.size == 0:
        return []

    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    edges = np.quantile(signal_values, quantiles)
    bucket_rows: list[dict[str, Any]] = []
    for bucket_idx in range(bucket_count):
        lo = float(edges[bucket_idx])
        hi = float(edges[bucket_idx + 1])
        if bucket_idx == bucket_count - 1:
            bucket_mask = (signal_values >= lo) & (signal_values <= hi)
        else:
            bucket_mask = (signal_values >= lo) & (signal_values < hi)
        if not np.any(bucket_mask):
            continue
        bucket_rows.append(
            {
                "strategy": strategy,
                "signal": signal_name,
                "bucket": bucket_idx,
                "signal_low": lo,
                "signal_high": hi,
                "count": int(np.count_nonzero(bucket_mask)),
                "signal_mean": float(np.mean(signal_values[bucket_mask])),
                "adv_final_mean": float(np.mean(adv_values[bucket_mask])),
                "adv_final_p50": float(np.percentile(adv_values[bucket_mask], 50.0)),
                "mc_return_mean": float(np.mean(return_values[bucket_mask])),
                "mc_return_p50": float(np.percentile(return_values[bucket_mask], 50.0)),
            }
        )
    return bucket_rows


def _value_bucket_rows(strategy: str, values: np.ndarray, mc_returns: np.ndarray, bucket_count: int = 5) -> list[dict[str, Any]]:
    mask = np.isfinite(values) & np.isfinite(mc_returns)
    values = np.asarray(values[mask], dtype=np.float64)
    mc_returns = np.asarray(mc_returns[mask], dtype=np.float64)
    if values.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    edges = np.quantile(values, quantiles)
    rows: list[dict[str, Any]] = []
    for bucket_idx in range(bucket_count):
        lo = float(edges[bucket_idx])
        hi = float(edges[bucket_idx + 1])
        if bucket_idx == bucket_count - 1:
            bucket_mask = (values >= lo) & (values <= hi)
        else:
            bucket_mask = (values >= lo) & (values < hi)
        if not np.any(bucket_mask):
            continue
        rows.append(
            {
                "strategy": strategy,
                "bucket": bucket_idx,
                "value_low": lo,
                "value_high": hi,
                "count": int(np.count_nonzero(bucket_mask)),
                "value_mean": float(np.mean(values[bucket_mask])),
                "mc_return_mean": float(np.mean(mc_returns[bucket_mask])),
                "mc_return_p50": float(np.percentile(mc_returns[bucket_mask], 50.0)),
                "mc_return_p90": float(np.percentile(mc_returns[bucket_mask], 90.0)),
            }
        )
    return rows


def main() -> None:
    cfg = load_config(str(CFG_PATH))
    obs_dim, state_dim = _make_models(cfg)

    teacher_actor = _load_actor(cfg, obs_dim, Path(cfg.exec_teacher_actor_path))
    critic = _load_critic(cfg, obs_dim, state_dim, CURRENT_CRITIC_PATH)

    actor_map: dict[str, ActorNet] = {}
    for spec in _strategy_specs():
        if spec["kind"] == "actor":
            actor_map[spec["name"]] = _load_actor(cfg, obs_dim, Path(spec["actor_path"]))

    all_rows: list[dict[str, Any]] = []
    for spec in _strategy_specs():
        print(f"Collecting rollout for {spec['name']} ...")
        rows = _collect_rollout_rows(
            cfg,
            spec,
            actor_map,
            teacher_actor,
            critic,
            episodes=EPISODES,
            seed_base=SEED_BASE,
        )
        all_rows.extend(rows)

    step_fieldnames = [
        "strategy",
        "episode",
        "seed",
        "step",
        "reward",
        "value_pred",
        "next_value_pred",
        "adv_raw",
        "adv_final",
        "lambda_return",
        "mc_return",
        "td_error",
        "terminated",
        "truncated",
        "gu_drop_ratio_step",
        "g_pre",
        "throughput_access_norm",
        "assoc_ratio",
        "assoc_count",
        "assoc_count_ratio",
        "served_user_count",
        "served_user_ratio",
        "arrival_sum",
        "outflow_sum",
        "gu_queue_arrival_steps",
    ]
    _write_csv(OUTPUT_DIR / "frontend_critic_step_rows.csv", all_rows, step_fieldnames)

    frontend_signals = [
        "gu_drop_ratio_step",
        "g_pre",
        "throughput_access_norm",
        "assoc_ratio",
        "served_user_ratio",
        "gu_queue_arrival_steps",
    ]
    frontend_rows: list[dict[str, Any]] = []
    frontend_bucket_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    critic_bucket_rows: list[dict[str, Any]] = []
    summary_payload: dict[str, Any] = {"strategies": {}}

    for spec in _strategy_specs():
        strategy = spec["name"]
        strat_rows = [row for row in all_rows if row["strategy"] == strategy]
        adv_final = np.asarray([row["adv_final"] for row in strat_rows], dtype=np.float64)
        adv_raw = np.asarray([row["adv_raw"] for row in strat_rows], dtype=np.float64)
        mc_return = np.asarray([row["mc_return"] for row in strat_rows], dtype=np.float64)
        values = np.asarray([row["value_pred"] for row in strat_rows], dtype=np.float64)
        td_error = np.asarray([row["td_error"] for row in strat_rows], dtype=np.float64)

        summary_payload["strategies"][strategy] = {
            "num_env_steps": int(len(strat_rows)),
            "value_pred_dist": _summary(values.tolist()),
            "mc_return_dist": _summary(mc_return.tolist()),
            "adv_final_dist": _summary(adv_final.tolist()),
        }

        for signal in frontend_signals:
            signal_values = np.asarray([row[signal] for row in strat_rows], dtype=np.float64)
            frontend_rows.append(
                {
                    "strategy": strategy,
                    "signal": signal,
                    "count": int(signal_values.size),
                    "signal_mean": float(np.mean(signal_values)),
                    "signal_p50": float(np.percentile(signal_values, 50.0)),
                    "signal_p90": float(np.percentile(signal_values, 90.0)),
                    "corr_signal_adv_final_pearson": _pearson(signal_values, adv_final),
                    "corr_signal_adv_final_spearman": _spearman(signal_values, adv_final),
                    "corr_signal_adv_raw_pearson": _pearson(signal_values, adv_raw),
                    "corr_signal_adv_raw_spearman": _spearman(signal_values, adv_raw),
                    "corr_signal_mc_return_pearson": _pearson(signal_values, mc_return),
                    "corr_signal_mc_return_spearman": _spearman(signal_values, mc_return),
                }
            )
            frontend_bucket_rows.extend(
                _quantile_bucket_rows(strategy, signal, signal_values, adv_final, mc_return)
            )

        critic_rows.append(
            {
                "strategy": strategy,
                "num_env_steps": int(len(strat_rows)),
                "explained_variance_mc": _explained_variance(values, mc_return),
                "value_mae_mc": float(np.mean(np.abs(values - mc_return))),
                "value_rmse_mc": float(np.sqrt(np.mean(np.square(values - mc_return)))),
                "td_error_mean": float(np.mean(td_error)),
                "td_error_mae": float(np.mean(np.abs(td_error))),
                "td_error_rmse": float(np.sqrt(np.mean(np.square(td_error)))),
                "corr_value_mc_return_pearson": _pearson(values, mc_return),
                "corr_value_mc_return_spearman": _spearman(values, mc_return),
                "value_mean": float(np.mean(values)),
                "mc_return_mean": float(np.mean(mc_return)),
            }
        )
        critic_bucket_rows.extend(_value_bucket_rows(strategy, values, mc_return))

    _write_csv(
        OUTPUT_DIR / "frontend_adv_summary.csv",
        frontend_rows,
        [
            "strategy",
            "signal",
            "count",
            "signal_mean",
            "signal_p50",
            "signal_p90",
            "corr_signal_adv_final_pearson",
            "corr_signal_adv_final_spearman",
            "corr_signal_adv_raw_pearson",
            "corr_signal_adv_raw_spearman",
            "corr_signal_mc_return_pearson",
            "corr_signal_mc_return_spearman",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "frontend_adv_buckets.csv",
        frontend_bucket_rows,
        [
            "strategy",
            "signal",
            "bucket",
            "signal_low",
            "signal_high",
            "count",
            "signal_mean",
            "adv_final_mean",
            "adv_final_p50",
            "mc_return_mean",
            "mc_return_p50",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "critic_quality_summary.csv",
        critic_rows,
        [
            "strategy",
            "num_env_steps",
            "explained_variance_mc",
            "value_mae_mc",
            "value_rmse_mc",
            "td_error_mean",
            "td_error_mae",
            "td_error_rmse",
            "corr_value_mc_return_pearson",
            "corr_value_mc_return_spearman",
            "value_mean",
            "mc_return_mean",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "critic_value_buckets.csv",
        critic_bucket_rows,
        [
            "strategy",
            "bucket",
            "value_low",
            "value_high",
            "count",
            "value_mean",
            "mc_return_mean",
            "mc_return_p50",
            "mc_return_p90",
        ],
    )
    summary_payload["frontend_adv_summary_csv"] = str(OUTPUT_DIR / "frontend_adv_summary.csv")
    summary_payload["critic_quality_summary_csv"] = str(OUTPUT_DIR / "critic_quality_summary.csv")
    summary_payload["frontend_adv_buckets_csv"] = str(OUTPUT_DIR / "frontend_adv_buckets.csv")
    summary_payload["critic_value_buckets_csv"] = str(OUTPUT_DIR / "critic_value_buckets.csv")
    _write_json(OUTPUT_DIR / "summary.json", summary_payload)

    print(f"Wrote diagnostics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
