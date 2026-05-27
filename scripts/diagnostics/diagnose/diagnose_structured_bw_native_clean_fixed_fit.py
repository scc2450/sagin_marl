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
import torch.nn.functional as F

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _index_dataclass, _to_device_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group, run_structured_training
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(raw: str) -> torch.device:
    value = str(raw or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(value)


def _safe_summary_tensor(values: torch.Tensor) -> dict[str, float]:
    arr = values.detach().float().reshape(-1).cpu()
    if int(arr.numel()) <= 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean().item()),
        "p50": float(torch.quantile(arr, 0.50).item()),
        "p90": float(torch.quantile(arr, 0.90).item()),
        "min": float(arr.min().item()),
        "max": float(arr.max().item()),
    }


def _build_actor(
    cfg: Any,
    *,
    actor_arch: str,
    device: torch.device,
    checkpoint: str | None = None,
):
    actor_cfg = copy.deepcopy(cfg)
    actor_cfg.structured_bw_actor_arch = str(actor_arch)
    bundle = build_structured_modules_from_config(actor_cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device)
    if checkpoint:
        strict = str(actor_arch).strip().lower() == "pooled_summary"
        load_checkpoint_forgiving(actor, str(checkpoint), map_location=device, strict=bool(strict))
    return actor


def _build_clean_learner(
    cfg: Any,
    actor,
    *,
    device: torch.device,
    actor_lr: float,
) -> StructuredMAPPO:
    params = [param for param in actor.parameters() if param.requires_grad]
    return StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=torch.optim.Adam(params, lr=float(actor_lr)),
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", False)),
        train_sat=bool(getattr(cfg, "train_sat", False)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )


def _masked_probs(raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(valid_mask, raw.clamp_min(0.0), torch.zeros_like(raw))
    denom = masked.sum(dim=1, keepdim=True)
    fallback = valid_mask.to(dtype=torch.float32) / valid_mask.to(dtype=torch.float32).sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)
    return torch.where(denom > 1.0e-8, masked / denom.clamp_min(1.0e-8), fallback)


def _det_action(actor, local_batch: Any, valid_mask: torch.Tensor) -> torch.Tensor:
    out = actor.act_bw(local_batch, deterministic=True)
    pred = getattr(out, "det_mean", None)
    if pred is None:
        pred = out.action
    return _masked_probs(pred.reshape(valid_mask.shape[0], -1).to(valid_mask.device, dtype=torch.float32), valid_mask)


@torch.no_grad()
def _fit_metrics(actor, local_batch: Any, target: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    pred = _det_action(actor, local_batch, valid_mask)
    valid_f = valid_mask.to(dtype=torch.float32)
    gap_sum = ((pred - target).abs() * valid_f).sum(dim=1)
    valid_l1_per_slot = ((pred - target).abs() * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    mse = (((pred - target).pow(2) * valid_f).sum() / valid_f.sum().clamp_min(1.0)).detach()
    invalid_mass = (pred * (~valid_mask).to(dtype=torch.float32)).abs().sum(dim=1).mean()
    simplex_error = (pred.sum(dim=1) - 1.0).abs().mean()
    return {
        "target_gap_sum_mean": float(gap_sum.mean().item()),
        "target_gap_sum_p90": float(torch.quantile(gap_sum.detach().float().cpu(), 0.90).item()) if int(gap_sum.numel()) > 0 else 0.0,
        "valid_l1_per_slot": float(valid_l1_per_slot.item()),
        "mse_per_valid_slot": float(mse.item()),
        "invalid_mass": float(invalid_mass.item()),
        "simplex_sum_abs_error": float(simplex_error.item()),
    }


def _collect_fixed_bw_batch(
    *,
    cfg: Any,
    actor,
    device: torch.device,
    num_envs: int,
    rollout_env_steps: int,
    seed: int,
) -> Any:
    learner = _build_clean_learner(cfg, actor, device=device, actor_lr=0.0)
    captured: dict[str, Any] = {}

    def _capture(buffer, _bootstrap_world_state) -> None:
        views = buffer.build_rollout_views(device)
        captured["bw_stage_batch"] = views.training_view.stage_batches.get(2)

    env = make_structured_env_group(cfg, num_envs=int(num_envs), backend="sync", mode="train")
    try:
        run_structured_training(
            env,
            learner,
            num_updates=1,
            rollout_env_steps=int(rollout_env_steps),
            reset_seed=int(seed),
            reset_on_start=True,
            episode_stat_window=10,
            before_update_callback=_capture,
        )
    finally:
        close_structured_env_group(env)
        close_probe = getattr(learner, "_close_bw_clean_probe_group", None)
        if callable(close_probe):
            close_probe()
    batch = captured.get("bw_stage_batch")
    if batch is None or int(getattr(batch, "num_samples", 0) or 0) <= 0:
        raise RuntimeError("Failed to capture a non-empty BW stage batch.")
    return batch


def _build_fixed_targets(
    *,
    learner: StructuredMAPPO,
    bw_stage_batch: Any,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    local_batch = _to_device_dataclass(bw_stage_batch.local_batch, device)
    num_samples = int(bw_stage_batch.num_samples)
    num_agents = int(bw_stage_batch.num_agents)
    ref_actions = bw_stage_batch.bw_ref_actions
    if ref_actions is None:
        ref_actions = bw_stage_batch.actions
    ref_t = ref_actions.to(device, dtype=torch.float32).reshape(num_samples * num_agents, -1)
    user_mask = getattr(local_batch, "user_mask", None)
    bw_valid_mask = getattr(local_batch, "bw_valid_mask", None)
    if user_mask is None or bw_valid_mask is None:
        raise RuntimeError("Captured BW batch is missing user_mask or bw_valid_mask.")
    valid_t = ((user_mask.to(device) > 0.5) & (bw_valid_mask.to(device) > 0.5)).reshape(num_samples * num_agents, -1)
    row_has_choice_t = valid_t.sum(dim=1) > 1
    ref_norm_t = _masked_probs(ref_t, valid_t)
    target_t, active_rows_t, target_gap_t, utility_l1_t = learner._build_native_bw_clean_per_user_targets(
        local_batch=local_batch,
        ref_probs=ref_norm_t,
        valid_mask=valid_t,
    )
    row_gap_t = (target_gap_t * valid_t.to(dtype=torch.float32)).sum(dim=1)
    active_rows_t = active_rows_t & row_has_choice_t & (row_gap_t > 1.0e-8)
    active_idx = torch.nonzero(active_rows_t, as_tuple=False).reshape(-1).to(device=device, dtype=torch.long)
    if int(active_idx.numel()) <= 0:
        raise RuntimeError("Fixed clean target builder produced no active rows.")
    local_active = _index_dataclass(local_batch, active_idx)
    target_active = target_t.index_select(0, active_idx).detach()
    valid_active = valid_t.index_select(0, active_idx).detach()
    summary = {
        "num_samples": int(num_samples),
        "num_agents": int(num_agents),
        "rows": int(num_samples * num_agents),
        "active_rows": int(active_idx.numel()),
        "active_frac": float(active_idx.numel() / max(num_samples * num_agents, 1)),
        "target_gap_sum": _safe_summary_tensor(row_gap_t.index_select(0, active_idx)),
        "utility_l1": _safe_summary_tensor(utility_l1_t.index_select(0, active_idx)),
        "valid_count": _safe_summary_tensor(valid_active.to(dtype=torch.float32).sum(dim=1)),
    }
    return local_active, target_active, valid_active, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline fixed-batch fit for native BW clean targets using the official native rollout path."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target_actor_checkpoint", default=None)
    parser.add_argument("--target_actor_arch", choices=["pooled_summary", "competition"], default="pooled_summary")
    parser.add_argument("--fit_actor_arch", choices=["pooled_summary", "competition"], default="competition")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--rollout_env_steps", type=int, default=16)
    parser.add_argument("--offline_steps", type=int, default=100)
    parser.add_argument("--actor_lr", type=float, default=1.0e-3)
    parser.add_argument("--eval_steps", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    device = _resolve_device(str(args.device))
    cfg = load_config(str(args.config))
    cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    cfg.train_accel = False
    cfg.train_sat = False
    cfg.train_bw = True
    cfg.bw_clean_per_user_enabled = True
    cfg.structured_bw_parameterization = "score_only_softmax"

    target_actor = _build_actor(
        cfg,
        actor_arch=str(args.target_actor_arch),
        device=device,
        checkpoint=None if args.target_actor_checkpoint is None else str(args.target_actor_checkpoint),
    )
    target_actor.eval()
    bw_stage_batch = _collect_fixed_bw_batch(
        cfg=cfg,
        actor=target_actor,
        device=device,
        num_envs=int(args.num_envs),
        rollout_env_steps=int(args.rollout_env_steps),
        seed=int(args.seed),
    )
    target_learner = _build_clean_learner(cfg, target_actor, device=device, actor_lr=0.0)
    local_active, target_active, valid_active, fixed_summary = _build_fixed_targets(
        learner=target_learner,
        bw_stage_batch=bw_stage_batch,
        device=device,
    )

    fit_actor = _build_actor(cfg, actor_arch=str(args.fit_actor_arch), device=device, checkpoint=None)
    fit_actor.train()
    optimizer = torch.optim.Adam(fit_actor.parameters(), lr=float(args.actor_lr))
    eval_steps = sorted({int(step) for step in args.eval_steps if 0 <= int(step) <= int(args.offline_steps)})
    history: list[dict[str, Any]] = []
    last_loss = 0.0
    for step in range(int(args.offline_steps) + 1):
        if step in eval_steps:
            fit_actor.eval()
            history.append(
                {
                    "offline_step": int(step),
                    **_fit_metrics(fit_actor, local_active, target_active, valid_active),
                    "last_train_loss": float(last_loss),
                }
            )
            fit_actor.train()
        if step == int(args.offline_steps):
            break
        pred = _det_action(fit_actor, local_active, valid_active)
        loss = F.smooth_l1_loss(pred[valid_active], target_active[valid_active])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fit_actor.parameters(), float(getattr(cfg, "max_grad_norm", 0.5) or 0.5))
        optimizer.step()
        last_loss = float(loss.detach().item())

    close_probe = getattr(target_learner, "_close_bw_clean_probe_group", None)
    if callable(close_probe):
        close_probe()
    payload = {
        "config": str(Path(args.config).resolve()),
        "traffic_model": str(getattr(cfg, "traffic_model", "")),
        "target_actor_checkpoint": None
        if args.target_actor_checkpoint is None
        else str(Path(args.target_actor_checkpoint).resolve()),
        "target_actor_arch": str(args.target_actor_arch),
        "fit_actor_arch": str(args.fit_actor_arch),
        "parameterization": str(getattr(cfg, "structured_bw_parameterization", "")),
        "device": str(device),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "rollout_env_steps": int(args.rollout_env_steps),
        "offline_steps": int(args.offline_steps),
        "actor_lr": float(args.actor_lr),
        "fixed_batch": fixed_summary,
        "history": history,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out_path = Path(str(args.json_out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
