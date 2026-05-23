from __future__ import annotations

import argparse
import json
import os
import random
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
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group, run_structured_training
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _done_from_step_result(step_result: Any) -> bool:
    return bool(
        list(step_result.terminations.values())[0]
        or list(step_result.truncations.values())[0]
    )


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _det_bw_action_from_snapshot(actor, snapshot: Any, device: torch.device) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=True)
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def _collect_panel(
    *,
    cfg,
    actor,
    device: torch.device,
    panel_states: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    try:
        while len(rows) < int(panel_states):
            env.reset(seed=int(seed_base) + int(episode_count))
            done = False
            while not done and len(rows) < int(panel_states):
                driver.begin_step()
                accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
                z1 = driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(z2)
                snapshot_state = driver.export_bw_stage_state()
                actor_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(snapshot_state or {}),
                        "actor_action": np.asarray(actor_action, dtype=np.float32),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(
                    np.asarray(actor_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                )
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _full_episode_return_from_snapshot(
    *,
    cfg,
    actor,
    device: torch.device,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
) -> float:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    try:
        probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
        total_reward = 0.0
        action = np.asarray(first_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        while True:
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            if _done_from_step_result(step_result):
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            z1 = probe_driver.run_accel_stage(accel_zero)
            z2 = probe_driver.run_sat_stage(_zero_sat_action(cfg))
            next_snapshot = probe_driver.build_bw_stage_snapshot(z2)
            action = _det_bw_action_from_snapshot(actor, next_snapshot, device).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        return float(total_reward)
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def _evaluate_panel(
    *,
    cfg,
    actor,
    device: torch.device,
    panel: list[dict[str, Any]],
) -> dict[str, Any]:
    returns: list[float] = []
    for row in panel:
        probe_env = make_structured_env(cfg, mode="script")
        probe_driver = as_structured_driver(probe_env)
        try:
            snapshot = probe_driver.load_bw_stage_state(dict(row["snapshot_state"] or {}))
            current_action = _det_bw_action_from_snapshot(actor, snapshot, device)
        finally:
            close_fn = getattr(probe_env, "close", None)
            if callable(close_fn):
                close_fn()
        returns.append(
            _full_episode_return_from_snapshot(
                cfg=cfg,
                actor=actor,
                device=device,
                snapshot_state=dict(row["snapshot_state"] or {}),
                first_action=current_action,
            )
        )
    return {"return": _safe_summary(np.asarray(returns, dtype=np.float32))}


def _build_learner(cfg, actor_ckpt: str | None, device: torch.device, *, actor_init: str = "checkpoint"):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device)
    actor_init_l = str(actor_init).strip().lower()
    if actor_init_l == "checkpoint":
        if not actor_ckpt:
            raise ValueError("--actor_checkpoint is required when --actor_init=checkpoint.")
        loc_readout = str(getattr(cfg, "structured_bw_loc_readout", "fused") or "fused").strip().lower()
        actor_arch = str(getattr(cfg, "structured_bw_actor_arch", "pooled_summary") or "pooled_summary").strip().lower()
        strict_load = (
            loc_readout not in {"user0_residual_fused", "fused_moe2", "fused_ctx_dot", "pairwise_comparator"}
            and actor_arch == "pooled_summary"
        )
        load_checkpoint_forgiving(actor, actor_ckpt, map_location=device, strict=bool(strict_load))
    elif actor_init_l != "scratch":
        raise ValueError(f"Unsupported actor_init={actor_init!r}")
    trainable_params = [param for param in actor.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable actor parameters remain after train-scope selection.")
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        actor_optimizer=torch.optim.Adam(trainable_params, lr=float(getattr(cfg, "actor_lr", 1.0e-3) or 1.0e-3)),
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    return actor, learner


def _configure_train_scope(actor, scope: str) -> str:
    scope_l = str(scope).strip().lower()
    for param in actor.parameters():
        param.requires_grad_(False)
    if scope_l in {"all", "actor", "full"}:
        for param in actor.parameters():
            param.requires_grad_(True)
        return "actor"
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")
    module_map = {
        "loc_head": "loc_head",
        "user_fusion": "user_fusion",
        "user_refine": "user_refine",
        "user_encoder": "user_encoder",
    }
    module_name = module_map.get(scope_l)
    if module_name is None:
        raise ValueError(f"Unsupported --train_scope: {scope}")
    module = getattr(bw_policy, module_name, None)
    if module is None:
        raise RuntimeError(f"bw_policy has no module {module_name!r}.")
    trainable_count = 0
    for param in module.parameters():
        param.requires_grad_(True)
        trainable_count += int(param.numel())
    if trainable_count <= 0:
        raise RuntimeError(f"No trainable parameters found for bw_policy.{module_name}.")
    return f"bw_policy.{module_name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", default=None)
    parser.add_argument(
        "--actor_init",
        choices=["checkpoint", "scratch"],
        default="checkpoint",
        help="Use checkpoint weights or initialize the actor from scratch.",
    )
    parser.add_argument("--panel_states", type=int, default=16)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--rollout_env_steps", type=int, default=80)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--trial_seed_base", type=int, default=46000)
    parser.add_argument(
        "--train_scope",
        default="all",
        help="Which actor parameters to update during the one-step online probe: all, loc_head, user_fusion, user_refine, user_encoder",
    )
    parser.add_argument(
        "--grad_aggregation",
        choices=["mean", "pcgrad"],
        default="mean",
        help="Clean BW gradient aggregation used during the one-step online probe.",
    )
    parser.add_argument(
        "--pcgrad_group_mode",
        choices=["random", "target_stats"],
        default=None,
        help="Optional override for bw_clean_pcgrad_group_mode.",
    )
    parser.add_argument(
        "--pcgrad_task_group_size",
        type=int,
        default=None,
        help="Optional override for bw_clean_pcgrad_task_group_size.",
    )
    parser.add_argument(
        "--loc_readout",
        choices=["fused", "fused_moe2", "fused_ctx_dot", "pairwise_comparator", "user_only", "user0", "user0_residual_fused"],
        default=None,
        help="Optional override for structured_bw_loc_readout.",
    )
    parser.add_argument(
        "--actor_arch",
        choices=["pooled_summary", "competition"],
        default=None,
        help="Optional override for structured_bw_actor_arch.",
    )
    parser.add_argument(
        "--traffic_model_override",
        default=None,
        help="Optional runtime override for cfg.traffic_model. Useful when the native diagnostic path cannot run hotspot tapes yet.",
    )
    parser.add_argument(
        "--competition_layers",
        type=int,
        default=None,
        help="Optional override for structured_bw_competition_layers.",
    )
    parser.add_argument(
        "--competition_heads",
        type=int,
        default=None,
        help="Optional override for structured_bw_competition_heads.",
    )
    parser.add_argument(
        "--clean_loss",
        choices=["huber", "masked_kl"],
        default=None,
        help="Optional override for bw_clean_per_user_loss.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setattr(cfg, "bw_clean_grad_aggregation", str(args.grad_aggregation))
    if args.pcgrad_group_mode is not None:
        setattr(cfg, "bw_clean_pcgrad_group_mode", str(args.pcgrad_group_mode))
    if args.pcgrad_task_group_size is not None:
        setattr(cfg, "bw_clean_pcgrad_task_group_size", max(int(args.pcgrad_task_group_size), 1))
    if args.loc_readout is not None:
        setattr(cfg, "structured_bw_loc_readout", str(args.loc_readout))
    if args.actor_arch is not None:
        setattr(cfg, "structured_bw_actor_arch", str(args.actor_arch))
    if args.traffic_model_override is not None:
        setattr(cfg, "traffic_model", str(args.traffic_model_override))
        if str(args.traffic_model_override).strip().lower() != "sticky_subset_hotspot":
            setattr(cfg, "hotspot_num_subsets", 0)
    if args.competition_layers is not None:
        setattr(cfg, "structured_bw_competition_layers", max(int(args.competition_layers), 1))
    if args.competition_heads is not None:
        setattr(cfg, "structured_bw_competition_heads", max(int(args.competition_heads), 1))
    if args.clean_loss is not None:
        setattr(cfg, "bw_clean_per_user_loss", str(args.clean_loss))
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    actor_checkpoint = None if args.actor_checkpoint is None else str(args.actor_checkpoint)
    if int(args.panel_states) > 0:
        panel_actor, _ = _build_learner(cfg, actor_checkpoint, device, actor_init=str(args.actor_init))
        panel_actor.eval()
        panel, episode_count = _collect_panel(
            cfg=cfg,
            actor=panel_actor,
            device=device,
            panel_states=int(args.panel_states),
            seed_base=int(args.panel_seed),
        )
    else:
        panel = []
        episode_count = 0

    trials_payload: list[dict[str, Any]] = []
    deltas: list[float] = []
    for trial in range(int(args.trials)):
        seed = int(args.trial_seed_base) + int(trial)
        _set_all_seeds(seed)
        actor, learner = _build_learner(cfg, actor_checkpoint, device, actor_init=str(args.actor_init))
        train_scope_used = _configure_train_scope(actor, str(args.train_scope))
        learner.actor_optimizer = torch.optim.Adam(
            [param for param in actor.parameters() if param.requires_grad],
            lr=float(getattr(cfg, "actor_lr", 1.0e-3) or 1.0e-3),
        )
        actor.eval()
        pre_eval = _evaluate_panel(cfg=cfg, actor=actor, device=device, panel=panel)
        env_group = make_structured_env_group(cfg, num_envs=int(args.num_envs), backend=str(args.vec_backend))
        try:
            history = run_structured_training(
                env_group,
                learner,
                num_updates=1,
                rollout_env_steps=int(args.rollout_env_steps),
                reset_seed=int(seed),
            )
        finally:
            close_structured_env_group(env_group)
        actor.eval()
        post_eval = _evaluate_panel(cfg=cfg, actor=actor, device=device, panel=panel)
        delta = float(post_eval["return"]["mean"] - pre_eval["return"]["mean"])
        deltas.append(delta)
        row = {
            "trial": int(trial),
            "seed": int(seed),
            "train_scope_used": str(train_scope_used),
            "pre_return_mean": float(pre_eval["return"]["mean"]),
            "post_return_mean": float(post_eval["return"]["mean"]),
            "delta_return_mean": float(delta),
        }
        if history:
            metrics = history[0]
            clean_group_debug = getattr(learner, "_bw_clean_last_group_debug", None)
            row["update_metrics"] = {
                "env_reward_mean": float(metrics.env_reward_mean),
                "episode_reward": float(metrics.episode_reward),
                "policy_loss": float(metrics.policy_loss),
                "clean_mean_target_gap": float(metrics.clean_mean_target_gap),
                "clean_mean_update_shift": float(metrics.clean_mean_update_shift),
                "clean_update_to_target_ratio": float(metrics.clean_update_to_target_ratio),
                "clean_target_beats_ref_frac": float(metrics.clean_target_beats_ref_frac),
                "clean_measured_kl": float(metrics.clean_measured_kl),
                "clean_kl_coef": float(metrics.clean_kl_coef),
            }
            if clean_group_debug is not None:
                row["update_metrics"]["clean_group_debug"] = clean_group_debug
        trials_payload.append(row)

    payload = {
        "config": os.path.abspath(args.config),
        "actor_init": str(args.actor_init),
        "actor_checkpoint": None if actor_checkpoint is None else os.path.abspath(actor_checkpoint),
        "actor_arch": str(getattr(cfg, "structured_bw_actor_arch", "pooled_summary")),
        "competition_layers": int(getattr(cfg, "structured_bw_competition_layers", 2) or 2),
        "competition_heads": int(getattr(cfg, "structured_bw_competition_heads", 4) or 4),
        "train_scope_requested": str(args.train_scope),
        "grad_aggregation": str(args.grad_aggregation),
        "pcgrad_group_mode": str(getattr(cfg, "bw_clean_pcgrad_group_mode", "random")),
        "pcgrad_task_group_size": int(getattr(cfg, "bw_clean_pcgrad_task_group_size", 32) or 1),
        "panel": {
            "states": int(len(panel)),
            "episodes": int(episode_count),
            "seed_base": int(args.panel_seed),
        },
        "trials": trials_payload,
        "delta_return_mean_summary": _safe_summary(np.asarray(deltas, dtype=np.float32)),
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
