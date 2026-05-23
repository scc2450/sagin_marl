from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.distributions import atanh
from sagin_marl.rl.structured_accel_actor_schema import (
    PEER_ALERT_FLAG,
    PEER_CLOSING_SPEED,
    PEER_REL_VX,
    PEER_REL_VY,
    PEER_SAFE_DISTANCE_MARGIN,
    PEER_UNSAFE_FLAG,
)
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summary_np(x: Any) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def _corr_np(x: Any, y: Any) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) <= 2:
        return 0.0
    xa = xa[mask]
    ya = ya[mask]
    sx = float(np.std(xa))
    sy = float(np.std(ya))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _make_actor_stage_optimizers(actor: torch.nn.Module, actor_lr: float) -> dict[int, torch.optim.Optimizer]:
    modules = {
        0: getattr(actor, "accel_policy", None),
        1: getattr(actor, "sat_subset_policy", None),
        2: getattr(actor, "bw_policy", None),
    }
    out: dict[int, torch.optim.Optimizer] = {}
    for stage_id, module in modules.items():
        if module is None:
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            out[int(stage_id)] = torch.optim.Adam(params, lr=float(actor_lr))
    return out


def _build_learner(cfg: Any, device: torch.device) -> StructuredMAPPO:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(cfg.actor_lr))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.critic_lr))
    return StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        num_mini_batch=int(cfg.num_mini_batch),
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=actor_optimizer,
        actor_stage_optimizers=_make_actor_stage_optimizers(actor, float(cfg.actor_lr)),
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )


def _load_train_state(learner: StructuredMAPPO, path: str | None, device: torch.device) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "actor_state_dict" in payload:
        learner.actor.load_state_dict(payload["actor_state_dict"], strict=False)
    if isinstance(payload, dict) and "critic_state_dict" in payload:
        learner.critic.load_state_dict(payload["critic_state_dict"], strict=False)


def _done_distance(done_tn: np.ndarray) -> np.ndarray:
    steps, envs = done_tn.shape
    out = np.full((steps, envs), steps + 1, dtype=np.int64)
    for e in range(envs):
        next_dist = steps + 1
        for t in range(steps - 1, -1, -1):
            if bool(done_tn[t, e]):
                next_dist = 0
            else:
                next_dist = min(next_dist + 1, steps + 1)
            out[t, e] = next_dist
    return out


def _bucket_stats(values: np.ndarray, bucket: np.ndarray) -> dict[str, dict[str, float]]:
    buckets = {
        "done_this_step": bucket == 0,
        "done_1_5": (bucket >= 1) & (bucket <= 5),
        "done_6_20": (bucket >= 6) & (bucket <= 20),
        "done_gt20_or_none": bucket > 20,
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if int(mask.sum()) <= 0:
            out[name] = {"count": 0.0, "mean": 0.0, "std": 0.0}
        else:
            part = np.asarray(values)[mask]
            out[name] = {"count": float(mask.sum()), "mean": float(np.mean(part)), "std": float(np.std(part))}
    return out


def _signal_against_noise(name: str, target_row: np.ndarray, z_agent: np.ndarray) -> dict[str, float]:
    target_agent = np.repeat(np.asarray(target_row, dtype=np.float64).reshape(-1), z_agent.shape[1])
    z_flat = z_agent.reshape(-1, z_agent.shape[-1])
    prod = target_agent[:, None] * z_flat
    return {
        f"corr_{name}_z_x": _corr_np(target_agent, z_flat[:, 0]),
        f"corr_{name}_z_y": _corr_np(target_agent, z_flat[:, 1]),
        f"mean_{name}_z_norm": float(np.linalg.norm(np.mean(prod, axis=0))) if prod.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_state", default=None)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    device = torch.device(args.device)
    learner = _build_learner(cfg, device)
    _load_train_state(learner, args.train_state, device)

    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        group.reset_many([int(args.seed) + i for i in range(int(args.num_envs))])
        learner.bind_native_runtime_contract(group)
        learner.begin_native_rollout(group, rollout_env_steps=int(args.rollout_env_steps), num_envs=int(args.num_envs))
        buffer = StructuredRolloutBuffer()
        learner.collect_env_horizon_native_tensor_policy(
            group,
            buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
        rollout_views = buffer.build_rollout_views(device)
        batch_view = rollout_views.training_view
        return_view = rollout_views.return_view

        learner._refresh_actor_old_logprobs_from_training_view(batch_view)
        value_override = learner._rollout_value_override_from_training_view(batch_view)
        learner._apply_rollout_value_override_to_views(
            batch_view=batch_view,
            return_view=return_view,
            value_override=value_override,
        )
        gae = learner.compute_returns_and_advantages(
            buffer,
            rollout_views.bootstrap_view,
            return_view=return_view,
            value_override=value_override,
        )

        stage_ids = np.asarray(batch_view.stage_ids, dtype=np.int64)
        raw_adv_all = torch.from_numpy(gae["advantages"]).to(device=device, dtype=torch.float32)
        norm_adv_all = raw_adv_all.clone()
        if bool(getattr(learner, "actor_advantage_normalize_enabled", False)) and norm_adv_all.numel() > 1:
            if bool(getattr(learner, "stagewise_advantage_norm_enabled", False)):
                for sid in (0, 1, 2):
                    idx_np = np.flatnonzero(stage_ids == sid)
                    if idx_np.size <= 1:
                        continue
                    idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                    part = norm_adv_all.index_select(0, idx)
                    part = (part - part.mean()) / part.std(unbiased=False).clamp_min(1.0e-8)
                    norm_adv_all.index_copy_(0, idx, part)
            else:
                norm_adv_all = (norm_adv_all - norm_adv_all.mean()) / norm_adv_all.std(unbiased=False).clamp_min(1.0e-8)
        returns_all = torch.from_numpy(gae["returns"]).to(device=device, dtype=torch.float32)

        accel_batch = batch_view.stage_batches[0]
        num_samples = int(accel_batch.num_samples)
        num_agents = int(accel_batch.num_agents)
        transition_idx_np = np.asarray(accel_batch.transition_indices, dtype=np.int64).reshape(-1)
        transition_idx_t = torch.as_tensor(transition_idx_np, dtype=torch.long, device=device)

        local_rows = torch.arange(num_samples * num_agents, dtype=torch.long, device=device)
        local = _index_dataclass(accel_batch.local_batch, local_rows)
        with torch.no_grad():
            dist_out = learner.actor.act_accel(local, deterministic=True)
        actions = accel_batch.actions.to(device=device, dtype=torch.float32).reshape(num_samples * num_agents, -1)
        z = atanh(actions / float(learner.actor.accel_policy.action_scale))
        z_noise = ((z - dist_out.mean) / dist_out.std.clamp_min(1.0e-6)).reshape(num_samples, num_agents, -1)
        z_noise_np = z_noise.detach().cpu().numpy().astype(np.float64, copy=False)

        raw_adv = raw_adv_all.index_select(0, transition_idx_t).detach().cpu().numpy()
        norm_adv = norm_adv_all.index_select(0, transition_idx_t).detach().cpu().numpy()
        returns = returns_all.index_select(0, transition_idx_t).detach().cpu().numpy()
        same_bw_idx = transition_idx_np + 2
        immediate_reward = np.asarray(return_view.rewards, dtype=np.float32)[same_bw_idx]
        bw_access = np.asarray(return_view.bw_access_rewards, dtype=np.float32)[same_bw_idx]
        bw_level = np.asarray(return_view.bw_weighted_workload_level_rewards, dtype=np.float32)[same_bw_idx]
        bw_delta = np.asarray(return_view.bw_weighted_workload_delta_rewards, dtype=np.float32)[same_bw_idx]

        steps = int(args.rollout_env_steps)
        envs = int(args.num_envs)
        bw_return_batch = return_view.stage_batches[2]
        terminated_tn = np.asarray(bw_return_batch.terminated, dtype=bool).reshape(steps, envs)
        truncated_tn = np.asarray(bw_return_batch.truncated, dtype=bool).reshape(steps, envs)
        done_tn = terminated_tn | truncated_tn
        done_dist_tn = _done_distance(done_tn)
        history_row = (transition_idx_np // 3).astype(np.int64, copy=False)
        time_idx = history_row // envs
        env_idx = np.asarray(accel_batch.env_indices, dtype=np.int64).reshape(-1)
        steps_to_done = done_dist_tn[time_idx, env_idx]
        done_this_step = done_tn[time_idx, env_idx].astype(np.float32)
        terminated_this_step = terminated_tn[time_idx, env_idx].astype(np.float32)

        masks = accel_batch.danger_imitation_masks
        targets = accel_batch.danger_imitation_targets
        if masks is None:
            danger_mask_np = np.zeros((num_samples, num_agents, 2), dtype=np.float32)
        else:
            danger_mask_np = masks.detach().cpu().numpy().astype(np.float32, copy=False)
        if targets is None:
            danger_target_np = np.zeros((num_samples, num_agents, 2), dtype=np.float32)
        else:
            danger_target_np = targets.detach().cpu().numpy().astype(np.float32, copy=False)
        danger_agent_active = (danger_mask_np > 0.5).any(axis=-1)
        danger_row_active = danger_agent_active.any(axis=-1).astype(np.float32)
        danger_target_norm = np.linalg.norm(danger_target_np, axis=-1)

        peer = accel_batch.local_batch.peer_tokens.detach().cpu().numpy().astype(np.float32, copy=False)
        peer_mask = accel_batch.local_batch.peer_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        peer_valid = peer_mask > 0.5
        if peer_valid.any():
            peer_rel_vel_norm = np.linalg.norm(peer[..., PEER_REL_VX : PEER_REL_VY + 1], axis=-1)
            peer_stats = {
                "peer_valid_frac": float(peer_valid.mean()),
                "peer_rel_vel_norm_valid": _summary_np(peer_rel_vel_norm[peer_valid]),
                "peer_closing_speed_valid": _summary_np(peer[..., PEER_CLOSING_SPEED][peer_valid]),
                "peer_margin_valid": _summary_np(peer[..., PEER_SAFE_DISTANCE_MARGIN][peer_valid]),
                "peer_unsafe_rate_valid": float(np.mean(peer[..., PEER_UNSAFE_FLAG][peer_valid] > 0.5)),
                "peer_alert_rate_valid": float(np.mean(peer[..., PEER_ALERT_FLAG][peer_valid] > 0.5)),
            }
        else:
            peer_stats = {"peer_valid_frac": 0.0}

        reward_parts_payload: dict[str, Any] = {}
        part_corr_rows: list[dict[str, float | str]] = []
        for key, arr in sorted(return_view.reward_part_arrays.items()):
            part = np.asarray(arr, dtype=np.float32).reshape(-1)[same_bw_idx]
            reward_parts_payload[str(key)] = {
                "summary": _summary_np(part),
                "by_done_distance": _bucket_stats(part, steps_to_done),
            }
            corr_x = _corr_np(np.repeat(part, num_agents), z_noise_np.reshape(-1, 2)[:, 0])
            corr_y = _corr_np(np.repeat(part, num_agents), z_noise_np.reshape(-1, 2)[:, 1])
            part_corr_rows.append({"key": str(key), "corr_z_x": corr_x, "corr_z_y": corr_y, "abs_max": max(abs(corr_x), abs(corr_y))})
        top_part_corr = sorted(part_corr_rows, key=lambda item: float(item["abs_max"]), reverse=True)[:12]

        payload = {
            "config": str(args.config),
            "train_state": str(args.train_state or ""),
            "seed": int(args.seed),
            "num_envs": envs,
            "rollout_env_steps": steps,
            "samples": {
                "accel_rows": num_samples,
                "agent_rows": num_samples * num_agents,
                "terminated_step_rate": float(terminated_this_step.mean()),
                "done_step_rate": float(done_this_step.mean()),
            },
            "peer_input": peer_stats,
            "danger_imitation": {
                "agent_active_rate": float(danger_agent_active.mean()),
                "row_active_rate": float(danger_row_active.mean()),
                "target_norm": _summary_np(danger_target_norm),
                "row_active_by_done_distance": _bucket_stats(danger_row_active, steps_to_done),
                "target_norm_by_done_distance": _bucket_stats(danger_target_norm.max(axis=1), steps_to_done),
            },
            "done_distance": {
                "steps_to_done": _summary_np(steps_to_done),
                "terminated_by_done_distance": _bucket_stats(terminated_this_step, steps_to_done),
                "episode_length_proxy_done_count": float(done_tn.sum()),
            },
            "targets": {
                "return": _summary_np(returns),
                "raw_advantage": _summary_np(raw_adv),
                "normalized_advantage": _summary_np(norm_adv),
                "immediate_reward": _summary_np(immediate_reward),
                "bw_access_reward": _summary_np(bw_access),
                "bw_weighted_workload_level": _summary_np(bw_level),
                "bw_weighted_workload_delta": _summary_np(bw_delta),
                "return_by_done_distance": _bucket_stats(returns, steps_to_done),
                "norm_adv_by_done_distance": _bucket_stats(norm_adv, steps_to_done),
                "immediate_reward_by_done_distance": _bucket_stats(immediate_reward, steps_to_done),
            },
            "noise_signal": {
                **_signal_against_noise("return", returns, z_noise_np),
                **_signal_against_noise("raw_advantage", raw_adv, z_noise_np),
                **_signal_against_noise("normalized_advantage", norm_adv, z_noise_np),
                **_signal_against_noise("immediate_reward", immediate_reward, z_noise_np),
                **_signal_against_noise("bw_access", bw_access, z_noise_np),
                **_signal_against_noise("bw_level", bw_level, z_noise_np),
                **_signal_against_noise("done_this_step", done_this_step, z_noise_np),
                **_signal_against_noise("danger_row_active", danger_row_active, z_noise_np),
            },
            "reward_parts_top_noise_corr": top_part_corr,
            "reward_parts": reward_parts_payload,
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "samples": payload["samples"],
            "peer_input": payload["peer_input"],
            "danger_imitation": payload["danger_imitation"],
            "targets": payload["targets"],
            "noise_signal": payload["noise_signal"],
            "reward_parts_top_noise_corr": payload["reward_parts_top_noise_corr"],
            "out": str(out),
        }, ensure_ascii=False, indent=2)[:9000])
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
