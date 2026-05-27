from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_horizons(raw: list[str]) -> list[int]:
    values: list[int] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            horizon = int(part)
            if horizon <= 0:
                raise ValueError(f"horizon must be positive, got {horizon}")
            if horizon not in values:
                values.append(horizon)
    return values or [1]


def _parse_int_list(raw: str | None) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return []
    values: list[int] = []
    for part in str(raw).replace(",", " ").split():
        if not part:
            continue
        values.append(int(part))
    return values


def _to_float_list(value: torch.Tensor | np.ndarray | list[float]) -> list[float]:
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1).astype(float).tolist()


def _summary(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(_to_float_list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_f = x.detach().float().reshape(-1)
    y_f = y.detach().float().reshape(-1)
    if x_f.numel() <= 1 or y_f.numel() != x_f.numel():
        return 0.0
    x_c = x_f - x_f.mean()
    y_c = y_f - y_f.mean()
    denom = torch.sqrt((x_c.square().sum() * y_c.square().sum()).clamp_min(1.0e-20))
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((x_c * y_c).sum().div(denom).detach().cpu().item())


def _top_cases(
    *,
    acf: torch.Tensor,
    sampled_returns: torch.Tensor,
    baseline_returns: torch.Tensor,
    selected: torch.Tensor,
    selected_history_rows: list[int],
    num_envs: int,
    sampled_target: torch.Tensor,
    det_target: torch.Tensor,
    valid_mask: torch.Tensor,
    target_l1: torch.Tensor,
    target_linf: torch.Tensor,
    valid_count: torch.Tensor,
    limit: int,
) -> list[dict[str, Any]]:
    n = int(acf.numel())
    k = min(max(int(limit), 0), n)
    if k <= 0:
        return []
    order = torch.argsort(acf.detach().abs(), descending=True)[:k].detach().cpu().tolist()
    sampled_cpu = sampled_target.detach().float().cpu()
    det_cpu = det_target.detach().float().cpu()
    mask_cpu = valid_mask.detach().bool().cpu()
    selected_cpu = selected.detach().cpu().tolist()
    l1_cpu = target_l1.detach().float().cpu().tolist()
    linf_cpu = target_linf.detach().float().cpu().tolist()
    vc_cpu = valid_count.detach().float().cpu().tolist()
    acf_cpu = acf.detach().float().cpu().tolist()
    sr_cpu = sampled_returns.detach().float().cpu().tolist()
    br_cpu = baseline_returns.detach().float().cpu().tolist()
    out: list[dict[str, Any]] = []
    for local_idx in order:
        history_row = int(selected_history_rows[int(local_idx)])
        valid_indices = torch.nonzero(mask_cpu[int(local_idx)], as_tuple=False).reshape(-1).tolist()
        out.append(
            {
                "local_index": int(local_idx),
                "stage_sample_index": int(selected_cpu[int(local_idx)]),
                "history_row": history_row,
                "source_step": int(history_row // max(int(num_envs), 1)),
                "source_env": int(history_row % max(int(num_envs), 1)),
                "acf": float(acf_cpu[int(local_idx)]),
                "sampled_return": float(sr_cpu[int(local_idx)]),
                "deterministic_baseline_return": float(br_cpu[int(local_idx)]),
                "action_l1": float(l1_cpu[int(local_idx)]),
                "action_linf": float(linf_cpu[int(local_idx)]),
                "valid_count": float(vc_cpu[int(local_idx)]),
                "valid_indices": [int(v) for v in valid_indices],
                "sampled_action_valid": [
                    float(sampled_cpu[int(local_idx), int(v)].item())
                    for v in valid_indices
                ],
                "deterministic_action_valid": [
                    float(det_cpu[int(local_idx), int(v)].item())
                    for v in valid_indices
                ],
            }
        )
    return out


def _select_rows(
    *,
    bw_stage_batch: Any,
    valid_mask_3d: torch.Tensor,
    history_rows: torch.Tensor,
    num_envs: int,
    rollout_env_steps: int,
    max_horizon: int,
    uav_id: int,
    sample_budget: int,
    seed: int,
) -> torch.Tensor:
    num_samples = int(getattr(bw_stage_batch, "num_samples", 0) or 0)
    if num_samples <= 0:
        return torch.zeros((0,), dtype=torch.long, device=valid_mask_3d.device)
    choice_count = valid_mask_3d[:, int(uav_id), :].to(dtype=torch.float32).sum(dim=-1)
    source_step = history_rows.reshape(-1).to(device=valid_mask_3d.device, dtype=torch.long) // max(int(num_envs), 1)
    last_allowed_step = max(int(rollout_env_steps) - int(max_horizon), 0)
    enough_future_tape = source_step <= int(last_allowed_step)
    eligible = torch.nonzero((choice_count > 1.0) & enough_future_tape, as_tuple=False).reshape(-1)
    if eligible.numel() == 0:
        return eligible
    if int(sample_budget) <= 0 or eligible.numel() <= int(sample_budget):
        return eligible
    generator = torch.Generator(device=eligible.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(eligible.numel()), generator=generator, device=eligible.device)
    return eligible.index_select(0, perm[: int(sample_budget)])


@torch.no_grad()
def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))

    cfg = load_config(str(args.config))
    cfg.exec_bw_source = "policy_single_uav_queue_aware"
    cfg.bw_single_uav_policy_uav_id = int(args.uav_id)
    cfg.bw_clean_per_user_enabled = False
    cfg.bw_clean_native_branch_chunk_size = max(int(args.branch_chunk_size), 0)
    cfg.native_branch_replay_reuse_sub_workspace = not bool(args.disable_branch_workspace_reuse)
    cfg.checkpoint_eval_enabled = False
    cfg.checkpoint_eval_interval_updates = 0
    cfg.train_accel = False
    cfg.train_sat = False
    cfg.train_bw = True
    if args.exec_accel_source is not None:
        cfg.exec_accel_source = str(args.exec_accel_source)
    if args.exec_sat_source is not None:
        cfg.exec_sat_source = str(args.exec_sat_source)
    if bool(args.disable_interference):
        cfg.interference_enabled = False
    if args.rollout_env_steps is not None:
        cfg.buffer_size = int(args.rollout_env_steps)

    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    actor.eval()
    critic.eval()
    checkpoint_info: dict[str, Any] | None = None
    if args.actor_checkpoint:
        checkpoint_info = load_checkpoint_forgiving(
            actor,
            str(args.actor_checkpoint),
            map_location=device,
            strict=bool(args.strict_load),
        )

    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "cluster_center_queue_aware") or "cluster_center_queue_aware"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "zero") or "zero"),
        exec_bw_source="policy_single_uav_queue_aware",
    )

    env_group = make_structured_env_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        bind_native_contract = getattr(learner, "bind_native_runtime_contract", None)
        if callable(bind_native_contract):
            bind_native_contract(env_group)
        env_group.reset_many([int(args.env_seed) + i for i in range(int(args.num_envs))])
        buffer = StructuredRolloutBuffer()
        learner.begin_native_rollout(
            env_group,
            rollout_env_steps=int(args.rollout_env_steps),
            num_envs=int(args.num_envs),
        )
        results = learner.collect_env_horizon_native_tensor_policy(
            env_group,
            buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=False,
        )
        if len(results) != int(args.rollout_env_steps):
            raise RuntimeError(
                f"native rollout returned {len(results)} steps, expected {int(args.rollout_env_steps)}."
            )
        rollout_views = buffer.build_rollout_views(device)
        bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
        if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
            raise RuntimeError("rollout produced no BW stage samples.")

        transition_indices = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64).reshape(-1)
        if np.any((transition_indices - 2) % 3 != 0):
            raise RuntimeError("BW stage transition indices are not aligned to stage id 2.")
        history_rows_all = torch.as_tensor(
            ((transition_indices - 2) // 3).astype(np.int64, copy=False),
            device=device,
            dtype=torch.long,
        )
        sampled_actions = bw_stage_batch.actions.detach().to(device=device, dtype=torch.float32)
        if sampled_actions.ndim != 3:
            raise RuntimeError(f"expected BW actions [sample,uav,user], got {tuple(sampled_actions.shape)}")
        num_samples = int(sampled_actions.shape[0])
        num_agents = int(sampled_actions.shape[1])
        user_dim = int(sampled_actions.shape[2])
        uav_id = int(args.uav_id)
        if not (0 <= uav_id < num_agents):
            raise RuntimeError(f"uav_id={uav_id} out of range for num_agents={num_agents}")

        local_batch = bw_stage_batch.local_batch
        det_out = actor.act_bw(local_batch, deterministic=True)
        det_actions = det_out.action.detach().to(device=device, dtype=torch.float32)
        det_actions = det_actions.reshape(num_samples, num_agents, user_dim)
        valid_mask = ((local_batch.user_mask > 0.5) & (local_batch.bw_valid_mask > 0.5)).reshape(
            num_samples,
            num_agents,
            user_dim,
        )
        horizons = _parse_horizons(args.horizons)
        forced_stage_indices = _parse_int_list(args.stage_indices)
        if forced_stage_indices:
            forced = torch.as_tensor(forced_stage_indices, device=device, dtype=torch.long)
            if bool(((forced < 0) | (forced >= num_samples)).any().detach().cpu().item()):
                raise RuntimeError(
                    f"--stage_indices contains values outside [0, {num_samples - 1}]: {forced_stage_indices}"
                )
            selected = forced
        else:
            selected = _select_rows(
                bw_stage_batch=bw_stage_batch,
                valid_mask_3d=valid_mask,
                history_rows=history_rows_all,
                num_envs=int(args.num_envs),
                rollout_env_steps=int(args.rollout_env_steps),
                max_horizon=max(int(h) for h in horizons),
                uav_id=uav_id,
                sample_budget=int(args.sample_budget),
                seed=int(args.seed) + 17,
            )
        if selected.numel() == 0:
            raise RuntimeError(f"no eligible BW states with more than one valid user for UAV {uav_id}.")

        sampled_first_actions = sampled_actions.index_select(0, selected).contiguous()
        baseline_first_actions = sampled_first_actions.clone()
        baseline_first_actions[:, uav_id, :] = det_actions.index_select(0, selected)[:, uav_id, :]
        selected_history_rows = [int(x) for x in history_rows_all.index_select(0, selected).detach().cpu().tolist()]
        selected_valid_mask = valid_mask.index_select(0, selected)[:, uav_id, :]

        sampled_target = sampled_first_actions[:, uav_id, :]
        det_target = baseline_first_actions[:, uav_id, :]
        target_l1 = ((sampled_target - det_target).abs() * selected_valid_mask.to(sampled_target.dtype)).sum(dim=-1)
        target_linf = ((sampled_target - det_target).abs() * selected_valid_mask.to(sampled_target.dtype)).amax(dim=-1)
        valid_count = selected_valid_mask.to(dtype=torch.float32).sum(dim=-1)

        horizon_reports: dict[str, Any] = {}
        for horizon in horizons:
            if bool(args.separate_branch_calls):
                sampled_returns = learner._native_bw_clean_rollout_returns_from_history(
                    history_rows=selected_history_rows,
                    first_actions=sampled_first_actions,
                    horizon=int(horizon),
                )
                baseline_returns = learner._native_bw_clean_rollout_returns_from_history(
                    history_rows=selected_history_rows,
                    first_actions=baseline_first_actions,
                    horizon=int(horizon),
                )
            else:
                paired_actions = torch.stack(
                    (sampled_first_actions, baseline_first_actions),
                    dim=1,
                ).reshape(-1, num_agents, user_dim)
                paired_rows: list[int] = []
                for row in selected_history_rows:
                    paired_rows.extend([int(row), int(row)])
                paired_returns = learner._native_bw_clean_rollout_returns_from_history(
                    history_rows=paired_rows,
                    first_actions=paired_actions,
                    horizon=int(horizon),
                )
                paired_returns = paired_returns.reshape(-1, 2)
                sampled_returns = paired_returns[:, 0]
                baseline_returns = paired_returns[:, 1]
            acf = sampled_returns - baseline_returns
            abs_acf = acf.abs()
            horizon_reports[str(int(horizon))] = {
                "horizon": int(horizon),
                "acf": _summary(acf),
                "abs_acf": _summary(abs_acf),
                "sampled_return": _summary(sampled_returns),
                "deterministic_baseline_return": _summary(baseline_returns),
                "positive_frac": float((acf > 1.0e-8).float().mean().detach().cpu().item()),
                "negative_frac": float((acf < -1.0e-8).float().mean().detach().cpu().item()),
                "near_zero_frac_1e_6": float((abs_acf <= 1.0e-6).float().mean().detach().cpu().item()),
                "near_zero_frac_1e_4": float((abs_acf <= 1.0e-4).float().mean().detach().cpu().item()),
                "near_zero_frac_1e_3": float((abs_acf <= 1.0e-3).float().mean().detach().cpu().item()),
                "corr_acf_action_l1": _safe_corr(acf, target_l1),
                "corr_abs_acf_action_l1": _safe_corr(abs_acf, target_l1),
                "top_abs_cases": _top_cases(
                    acf=acf,
                    sampled_returns=sampled_returns,
                    baseline_returns=baseline_returns,
                    selected=selected,
                    selected_history_rows=selected_history_rows,
                    num_envs=int(args.num_envs),
                    sampled_target=sampled_target,
                    det_target=det_target,
                    valid_mask=selected_valid_mask,
                    target_l1=target_l1,
                    target_linf=target_linf,
                    valid_count=valid_count,
                    limit=int(args.top_cases),
                ),
            }

        output = {
            "config": str(args.config),
            "actor_checkpoint": None if not args.actor_checkpoint else str(args.actor_checkpoint),
            "checkpoint_info": checkpoint_info,
            "device": str(device),
            "seed": int(args.seed),
            "env_seed": int(args.env_seed),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "bw_stage_samples": int(num_samples),
            "selected_samples": int(selected.numel()),
            "uav_id": int(uav_id),
            "exec_sources": {
                "accel": str(getattr(cfg, "exec_accel_source", "")),
                "sat": str(getattr(cfg, "exec_sat_source", "")),
                "bw": "policy_single_uav_queue_aware",
            },
            "interference_enabled": bool(getattr(cfg, "interference_enabled", False)),
            "paired_branch_calls": not bool(args.separate_branch_calls),
            "branch_workspace_reuse": not bool(args.disable_branch_workspace_reuse),
            "action_delta_l1": _summary(target_l1),
            "action_delta_linf": _summary(target_linf),
            "valid_count": _summary(valid_count),
            "horizons": horizon_reports,
        }
        return output
    finally:
        close_structured_env_group(env_group)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose BW counterfactual advantage A_cf = G(sampled target-UAV BW, queue-aware others) "
            "- G(deterministic target-UAV BW, queue-aware others)."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor_checkpoint", type=Path, default=None)
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--sample_budget", type=int, default=128)
    parser.add_argument("--branch_chunk_size", type=int, default=16)
    parser.add_argument("--top_cases", type=int, default=8)
    parser.add_argument("--stage_indices", type=str, default=None)
    parser.add_argument("--separate_branch_calls", action="store_true")
    parser.add_argument("--disable_branch_workspace_reuse", action="store_true")
    parser.add_argument("--uav_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--env_seed", type=int, default=42000)
    parser.add_argument("--horizons", nargs="+", default=["1", "10"])
    parser.add_argument("--disable_interference", action="store_true")
    parser.add_argument("--exec_accel_source", type=str, default=None)
    parser.add_argument("--exec_sat_source", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_diagnostic(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
