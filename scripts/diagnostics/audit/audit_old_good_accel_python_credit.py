from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _prepare_imports(repo_root: str) -> None:
    root = str(Path(repo_root).resolve())
    here_root = str(
        next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir()).resolve()
    )
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here_root]
    if root not in sys.path:
        sys.path.insert(0, root)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return 0.0
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _summ(x: Any) -> dict[str, float]:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "max": float(np.max(arr)),
    }


def _load_train_state(actor: torch.nn.Module, critic: torch.nn.Module, path: str | None, device: torch.device) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "actor_state_dict" in payload:
        actor.load_state_dict(payload["actor_state_dict"], strict=False)
    if isinstance(payload, dict) and "critic_state_dict" in payload:
        critic.load_state_dict(payload["critic_state_dict"], strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_state", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_rows", type=int, default=16)
    parser.add_argument("--policy_baseline_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _prepare_imports(args.repo_root)

    from dataclasses import replace

    import yaml

    from sagin_marl.env.config import SaginConfig, update_config
    from sagin_marl.env.sagin_env import SaginParallelEnv
    from sagin_marl.env.structured_driver import StructuredControlDriver
    from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
    from sagin_marl.rl.structured_factory import build_structured_modules_from_config
    from sagin_marl.rl.structured_mappo import (
        StructuredMAPPO,
        _collate_dataclass,
        _current_obs_list,
        _heuristic_bw,
        _heuristic_sat,
        _refresh_stage_obs_cache,
        _sat_mask_to_ids,
    )

    _set_seed(int(args.seed))
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f) or {}
    # Saved run configs may include bookkeeping keys that the older config
    # loader rejects.  Keep the actual training fields unchanged.
    cfg_data = {k: v for k, v in cfg_data.items() if not str(k).startswith("_")}
    cfg = update_config(SaginConfig(), cfg_data)
    device = torch.device(args.device)
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    _load_train_state(actor, critic, args.train_state, device)
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=cfg.clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        num_mini_batch=cfg.num_mini_batch,
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=float(cfg.actor_lr)),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr)),
        device=device,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )

    envs = [SaginParallelEnv(replace(cfg, seed=int(args.seed) + i)) for i in range(int(args.num_envs))]
    drivers = [StructuredControlDriver(env) for env in envs]
    for i, env in enumerate(envs):
        env.reset(seed=int(args.seed) + i)

    snapshots: list[dict[str, Any]] = []
    reset_counters = [0 for _ in range(int(args.num_envs))]
    buffer = StructuredRolloutBuffer()
    for _step in range(int(args.rollout_env_steps)):
        for driver in drivers:
            snapshots.append(copy.deepcopy(driver.env.export_runtime_state()))
        results = learner.collect_env_steps(drivers, buffer, deterministic=False)
        for env_index, result in enumerate(results):
            done = bool(next(iter(result.terminations.values()))) or bool(next(iter(result.truncations.values())))
            if done:
                reset_counters[env_index] += 1
                seed = int(args.seed) + reset_counters[env_index] * int(args.num_envs) + env_index
                envs[env_index].reset(seed=seed)
                drivers[env_index] = StructuredControlDriver(envs[env_index])

    batches = buffer.as_stage_batches()
    gae = learner.compute_returns_and_advantages(buffer, buffer.latest_next_world_states())
    stage_ids = np.asarray(batches["stage_ids"], dtype=np.int64)
    advantages = torch.from_numpy(gae["advantages"]).to(device)
    raw_advantages = advantages.clone()
    if advantages.numel() > 1:
        if bool(getattr(learner, "stagewise_advantage_norm_enabled", False)):
            advantages = advantages.clone()
            for sid in (0, 1, 2):
                idx_np = np.flatnonzero(stage_ids == sid)
                if idx_np.size <= 1:
                    continue
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                part = advantages.index_select(0, idx)
                part = (part - part.mean()) / part.std(unbiased=False).clamp_min(1e-8)
                advantages.index_copy_(0, idx, part)
        else:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = torch.from_numpy(gae["returns"]).to(device)
    values = torch.from_numpy(batches["values"]).to(device)
    accel_global_idx = np.flatnonzero(stage_ids == 0)
    sample_count = min(max(int(args.sample_rows), 1), int(accel_global_idx.size))
    rng = np.random.default_rng(int(args.seed) + 777)
    selected_stage_pos = np.sort(rng.choice(int(accel_global_idx.size), size=sample_count, replace=False))
    selected_global_idx = accel_global_idx[selected_stage_pos]
    local_states = [batches["local_actor_states"][int(i)] for i in selected_global_idx]
    sampled_actions = torch.stack(
        [
            action if torch.is_tensor(action) else torch.as_tensor(action)
            for action in (batches["actions"][int(i)] for i in selected_global_idx)
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    selected_adv = advantages.index_select(0, torch.as_tensor(selected_global_idx, dtype=torch.long, device=device))
    selected_raw_adv = raw_advantages.index_select(0, torch.as_tensor(selected_global_idx, dtype=torch.long, device=device))
    selected_returns = returns.index_select(0, torch.as_tensor(selected_global_idx, dtype=torch.long, device=device))
    selected_values = values.index_select(0, torch.as_tensor(selected_global_idx, dtype=torch.long, device=device))
    batch_local = _collate_dataclass(local_states, device)
    with torch.no_grad():
        old_logprob, _old_entropy, _old_out = learner._stage_actor_eval(0, local_states, [a for a in sampled_actions.detach().cpu()])
        policy_alt_actions = []
        for _ in range(max(int(args.policy_baseline_samples), 0)):
            alt = actor.act_accel(batch_local, deterministic=False)
            policy_alt_actions.append(alt.action.reshape(sample_count, int(cfg.num_uav), 2).detach().cpu())

    def one_manual_step(driver: Any, first_action_np: np.ndarray) -> tuple[float, bool]:
        driver.run_accel_stage(first_action_np)
        _refresh_stage_obs_cache(driver)
        obs_after_accel = _current_obs_list(driver)
        sat_mask = _heuristic_sat(obs_after_accel, cfg, learner.exec_source_by_stage[1])
        sat_action = _sat_mask_to_ids(driver, sat_mask)
        driver.run_sat_stage(sat_action)
        bw_action = _heuristic_bw(obs_after_accel, cfg, learner.exec_source_by_stage[2])
        result, _next_world = driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        reward = float(next(iter(result.rewards.values())))
        done = bool(next(iter(result.terminations.values()))) or bool(next(iter(result.truncations.values())))
        return reward, done

    branch_env = SaginParallelEnv(replace(cfg, seed=int(args.seed) + 99991))
    branch_driver = StructuredControlDriver(branch_env)

    def discounted_return(snapshot: dict[str, Any], action: torch.Tensor, horizon: int) -> float:
        branch_env.load_runtime_state(copy.deepcopy(snapshot))
        branch_driver._clear_step()
        total = 0.0
        discount = 1.0
        reward, done = one_manual_step(branch_driver, action.detach().cpu().numpy().astype(np.float32))
        total += discount * reward
        if done:
            return float(total)
        discount *= float(cfg.gamma)
        tmp_buffer = StructuredRolloutBuffer()
        for _ in range(max(int(horizon) - 1, 0)):
            result = learner.collect_env_step(branch_driver, tmp_buffer, deterministic=True)
            reward = float(next(iter(result.rewards.values())))
            total += discount * reward
            done = bool(next(iter(result.terminations.values()))) or bool(next(iter(result.truncations.values())))
            if done:
                break
            discount *= float(cfg.gamma)
        return float(total)

    history_rows = (selected_global_idx // 3).astype(np.int64)
    step_indices = history_rows // int(args.num_envs)
    horizons = np.maximum(int(cfg.T_steps) - step_indices, 1).astype(np.int64)
    sample_returns = np.zeros((sample_count,), dtype=np.float64)
    policy_mean_returns = np.zeros((sample_count,), dtype=np.float64)
    for i in range(sample_count):
        snap = snapshots[int(history_rows[i])]
        sample_returns[i] = discounted_return(snap, sampled_actions[i], int(horizons[i]))
        alt_returns = [
            discounted_return(snap, alt_actions[i], int(horizons[i]))
            for alt_actions in policy_alt_actions
        ]
        policy_mean_returns[i] = float(np.mean(alt_returns)) if alt_returns else 0.0
    policy_delta = sample_returns - policy_mean_returns

    update_metrics = learner.update(buffer, bootstrap_world_state=buffer.latest_next_world_states())
    with torch.no_grad():
        new_logprob, _new_entropy, _new_out = learner._stage_actor_eval(
            0,
            local_states,
            [a for a in sampled_actions.detach().cpu()],
        )
    delta_logprob = (new_logprob - old_logprob).detach().cpu().numpy()
    adv_np = selected_adv.detach().cpu().numpy()
    raw_adv_np = selected_raw_adv.detach().cpu().numpy()
    payload = {
        "repo_root": str(args.repo_root),
        "config": str(args.config),
        "train_state": str(args.train_state or ""),
        "sample_rows": int(sample_count),
        "update_metrics": {k: float(v) for k, v in update_metrics.items() if isinstance(v, (int, float))},
        "advantage": {
            "normalized": _summ(adv_np),
            "raw": _summ(raw_adv_np),
            "return": _summ(selected_returns),
            "value": _summ(selected_values),
        },
        "policy_delta": _summ(policy_delta),
        "sample_return": _summ(sample_returns),
        "policy_mean_return": _summ(policy_mean_returns),
        "delta_logprob": _summ(delta_logprob),
        "alignment": {
            "corr_norm_adv_vs_policy_delta": _corr(adv_np, policy_delta),
            "corr_raw_adv_vs_policy_delta": _corr(raw_adv_np, policy_delta),
            "sign_agree_norm_adv_policy_delta": float(np.mean(np.sign(adv_np) == np.sign(policy_delta))),
            "sign_agree_raw_adv_policy_delta": float(np.mean(np.sign(raw_adv_np) == np.sign(policy_delta))),
            "corr_norm_adv_vs_delta_logprob": _corr(adv_np, delta_logprob),
            "corr_policy_delta_vs_delta_logprob": _corr(policy_delta, delta_logprob),
            "mean_delta_logprob_when_policy_positive": float(np.mean(delta_logprob[policy_delta > 0.0])) if np.any(policy_delta > 0.0) else 0.0,
            "mean_delta_logprob_when_policy_negative": float(np.mean(delta_logprob[policy_delta < 0.0])) if np.any(policy_delta < 0.0) else 0.0,
            "policy_positive_frac": float(np.mean(policy_delta > 0.0)),
            "norm_adv_positive_frac": float(np.mean(adv_np > 0.0)),
        },
        "rows": [
            {
                "sample": int(selected_stage_pos[i]),
                "global_index": int(selected_global_idx[i]),
                "history_row": int(history_rows[i]),
                "step": int(step_indices[i]),
                "horizon": int(horizons[i]),
                "adv_norm": float(adv_np[i]),
                "adv_raw": float(raw_adv_np[i]),
                "return": float(selected_returns[i].detach().cpu().item()),
                "value": float(selected_values[i].detach().cpu().item()),
                "sample_return": float(sample_returns[i]),
                "policy_mean_return": float(policy_mean_returns[i]),
                "policy_delta": float(policy_delta[i]),
                "delta_logprob": float(delta_logprob[i]),
            }
            for i in range(sample_count)
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["alignment"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
