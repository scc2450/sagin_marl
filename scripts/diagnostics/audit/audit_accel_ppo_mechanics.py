from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_eval import evaluate_structured_actor_exec_sources
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y_np = y.detach().cpu().numpy().astype(np.float64).reshape(-1)
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if int(mask.sum()) <= 1:
        return 0.0
    x_np = x_np[mask]
    y_np = y_np[mask]
    sx = float(np.std(x_np))
    sy = float(np.std(y_np))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x_np, y_np)[0, 1])


def _summ(x: torch.Tensor) -> dict[str, float]:
    arr = x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "max": float(np.max(arr)),
    }


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


def _build_learner(cfg: Any, device: torch.device, *, actor_lr: float | None = None) -> StructuredMAPPO:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    lr_a = float(cfg.actor_lr if actor_lr is None else actor_lr)
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=lr_a)
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
        actor_stage_optimizers=_make_actor_stage_optimizers(actor, lr_a),
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


def _load_state_dict_file(module: torch.nn.Module, path: str | None, device: torch.device, key: str | None = None) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    if key is not None and isinstance(payload, dict) and key in payload:
        payload = payload[key]
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    module.load_state_dict(payload, strict=False)


def _reset_optimizers(learner: StructuredMAPPO, cfg: Any, *, actor_lr: float | None = None) -> None:
    lr_a = float(cfg.actor_lr if actor_lr is None else actor_lr)
    learner.actor_optimizer = torch.optim.Adam([p for p in learner.actor.parameters() if p.requires_grad], lr=lr_a)
    learner.actor_stage_optimizers = _make_actor_stage_optimizers(learner.actor, lr_a)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=float(cfg.critic_lr))


def _accel_eval(learner: StructuredMAPPO, stage_batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(stage_batch.num_samples)
    a = int(stage_batch.num_agents)
    rows = torch.arange(n * a, dtype=torch.long, device=learner.device)
    local = _index_dataclass(stage_batch.local_batch, rows)
    actions = stage_batch.actions.to(device=learner.device, dtype=torch.float32)
    logp, ent, _ = learner._stage_actor_eval_from_batch(0, local, actions, a)
    return logp.reshape(-1), ent.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pure_lr", type=float, default=None)
    parser.add_argument("--actor_checkpoint", default=None)
    parser.add_argument("--critic_checkpoint", default=None)
    parser.add_argument("--train_state", default=None)
    parser.add_argument("--eval_episodes", type=int, default=0)
    parser.add_argument("--eval_seed_base", type=int, default=9000)
    parser.add_argument(
        "--official_minibatches",
        default=None,
        help="Comma-separated minibatch counts to compare on the same collected rollout buffer.",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    cfg = load_config(args.config)
    device = torch.device(args.device)
    learner = _build_learner(cfg, device, actor_lr=args.pure_lr)
    if args.train_state:
        _load_state_dict_file(learner.actor, args.train_state, device, key="actor_state_dict")
        _load_state_dict_file(learner.critic, args.train_state, device, key="critic_state_dict")
    _load_state_dict_file(learner.actor, args.actor_checkpoint, device)
    _load_state_dict_file(learner.critic, args.critic_checkpoint, device)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        def _eval_actor_current(label: str) -> dict[str, float]:
            if int(args.eval_episodes) <= 0:
                return {}
            summary, _rows = evaluate_structured_actor_exec_sources(
                cfg,
                learner.actor,
                device=device,
                episodes=int(args.eval_episodes),
                episode_seed_base=int(args.eval_seed_base),
                deterministic=True,
                num_envs=max(1, min(int(args.num_envs), int(args.eval_episodes))),
                vec_backend="sync",
                exec_accel_source="policy",
                exec_sat_source=str(getattr(cfg, "exec_sat_source", "queue_aware") or "queue_aware"),
                exec_bw_source=str(getattr(cfg, "exec_bw_source", "queue_aware") or "queue_aware"),
            )
            return {f"{label}_{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float))}

        if hasattr(group, "reset_many"):
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
        advantages = torch.from_numpy(gae["advantages"]).to(device)
        raw_advantages = advantages.clone()
        if bool(getattr(learner, "actor_advantage_normalize_enabled", False)) and advantages.numel() > 1:
            if bool(getattr(learner, "stagewise_advantage_norm_enabled", False)):
                advantages = advantages.clone()
                for sid in (0, 1, 2):
                    idx_np = np.flatnonzero(stage_ids == sid)
                    if idx_np.size <= 1:
                        continue
                    idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                    part = advantages.index_select(0, idx)
                    part = (part - part.mean()) / part.std(unbiased=False).clamp_min(1.0e-8)
                    advantages.index_copy_(0, idx, part)
            else:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)

        accel_batch = batch_view.stage_batches[0]
        accel_idx = torch.as_tensor(accel_batch.transition_indices, dtype=torch.long, device=device)
        adv = advantages.index_select(0, accel_idx).detach()
        raw_adv = raw_advantages.index_select(0, accel_idx).detach()
        old_logp = accel_batch.old_logprobs.to(device=device, dtype=torch.float32).reshape(-1).detach()
        pre_logp, entropy = _accel_eval(learner, accel_batch)
        replay_diff = pre_logp - old_logp

        base_actor_state = copy.deepcopy(learner.actor.state_dict())
        base_critic_state = copy.deepcopy(learner.critic.state_dict())
        base_eval = _eval_actor_current("base")
        pure_state = copy.deepcopy(base_actor_state)
        pure_opt = torch.optim.Adam(
            [p for p in learner.actor.accel_policy.parameters() if p.requires_grad],
            lr=float(args.pure_lr if args.pure_lr is not None else cfg.actor_lr),
        )
        ratio = torch.exp(torch.clamp(pre_logp - old_logp, min=-20.0, max=20.0))
        clipped = torch.clamp(ratio, 1.0 - float(learner.clip_ratio), 1.0 + float(learner.clip_ratio))
        pure_policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
        pure_opt.zero_grad()
        pure_policy_loss.backward()
        pure_grad_norm = torch.nn.utils.clip_grad_norm_(learner.actor.accel_policy.parameters(), float(learner.max_grad_norm))
        pure_opt.step()
        pure_post_logp, _ = _accel_eval(learner, accel_batch)
        pure_delta = (pure_post_logp - pre_logp).detach()
        pure_eval = _eval_actor_current("pure_ppo_one_step")

        pos = adv > 0.0
        neg = adv < 0.0
        mb_values = [int(learner.num_mini_batch)]
        if args.official_minibatches:
            mb_values = [
                max(1, int(part.strip()))
                for part in str(args.official_minibatches).split(",")
                if part.strip()
            ]
        official_payload: dict[str, Any] = {}
        for mb in mb_values:
            learner.actor.load_state_dict(base_actor_state)
            learner.critic.load_state_dict(base_critic_state)
            _reset_optimizers(learner, cfg, actor_lr=args.pure_lr)
            learner.num_mini_batch = int(mb)
            views_for_update = buffer.build_rollout_views(device)
            official_metrics = learner.update(buffer, rollout_views=views_for_update)
            official_post_logp, _ = _accel_eval(learner, accel_batch)
            official_delta = (official_post_logp - pre_logp).detach()
            official_eval = _eval_actor_current(f"official_mb{int(mb)}")
            official_payload[str(int(mb))] = {
                "metrics": {k: float(v) for k, v in official_metrics.items() if isinstance(v, (int, float))},
                "eval": official_eval,
                "delta_logprob": _summ(official_delta),
                "corr_adv_delta_logprob": _corr(adv, official_delta),
                "mean_delta_pos_adv": float(official_delta[pos].mean().detach().cpu().item()) if bool(pos.any()) else 0.0,
                "mean_delta_neg_adv": float(official_delta[neg].mean().detach().cpu().item()) if bool(neg.any()) else 0.0,
                "pos_adv_logprob_up_frac": float((official_delta[pos] > 0.0).float().mean().detach().cpu().item()) if bool(pos.any()) else 0.0,
                "neg_adv_logprob_down_frac": float((official_delta[neg] < 0.0).float().mean().detach().cpu().item()) if bool(neg.any()) else 0.0,
            }

        payload = {
            "config": str(args.config),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "accel_samples": int(adv.numel()),
            "old_logprob_replay_abs": _summ(replay_diff.abs()),
            "advantage": {"normalized": _summ(adv), "raw": _summ(raw_adv)},
            "pure_ppo": {
                "policy_loss": float(pure_policy_loss.detach().cpu().item()),
                "grad_norm": float(torch.as_tensor(pure_grad_norm).detach().cpu().item()),
                "eval": pure_eval,
                "delta_logprob": _summ(pure_delta),
                "corr_adv_delta_logprob": _corr(adv, pure_delta),
                "mean_delta_pos_adv": float(pure_delta[pos].mean().detach().cpu().item()) if bool(pos.any()) else 0.0,
                "mean_delta_neg_adv": float(pure_delta[neg].mean().detach().cpu().item()) if bool(neg.any()) else 0.0,
                "pos_adv_logprob_up_frac": float((pure_delta[pos] > 0.0).float().mean().detach().cpu().item()) if bool(pos.any()) else 0.0,
                "neg_adv_logprob_down_frac": float((pure_delta[neg] < 0.0).float().mean().detach().cpu().item()) if bool(neg.any()) else 0.0,
            },
            "base_eval": base_eval,
            "official_update_by_minibatch": official_payload,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("old_logprob_replay_abs", "pure_ppo", "official_update_by_minibatch")}, ensure_ascii=False, indent=2)[:6000])
        print(f"wrote {out}")
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
