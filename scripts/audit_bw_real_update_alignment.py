import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import stage_optimizer_params as _stage_optimizer_params
from sagin_marl.rl.structured_mappo import _slice_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache

from scripts.train_joint_mcgae import (
    STAGES,
    _actor_only_stage_batch,
    _collect_joint_rollout,
    _force_joint_config,
    _make_joint_learner,
    _resolve_training_hparams,
    _set_seed,
)
from scripts.train_stage_mcgae import (
    STAGE_NAME,
    _enable_strict_compile_global,
    _normalize_stage_advantage,
    _stage_actor_update_full_stage,
    _stage_gae_from_mc_targets,
    _sync_danger_imitation_to_learner,
    _train_stage_critic_on_stage,
)


BW_STAGE = 2
BW_MODES = ("legacy_fast", "new_fast", "current")


def _load_checkpoint(
    learner: Any,
    checkpoint: Path,
    *,
    device: torch.device,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizers: dict[int, torch.optim.Optimizer],
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    learner.actor.load_state_dict(state["actor"], strict=True)
    learner.critic.load_state_dict(state["critic"], strict=True)
    if "critic_optimizer" in state:
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    # Do not restore actor optimizer state here.  The audit intentionally asks
    # "what would one standardized BW update do from this actor state?", and
    # historical checkpoints may have Adam or SGD states depending on the
    # ablation.  Mixing optimizer classes can silently corrupt param groups.
    return state


def _eval_bw_params(
    learner: Any,
    stage_batch: Any,
    *,
    chunk_size: int = 1024,
) -> dict[str, torch.Tensor]:
    device = learner.device
    num_agents = int(stage_batch.num_agents)
    sample_count = int(stage_batch.num_samples)
    actions = stage_batch.actions.to(device=device)
    det_chunks: list[torch.Tensor] = []
    tau_chunks: list[torch.Tensor] = []
    kappa_chunks: list[torch.Tensor] = []
    valid_count_chunks: list[torch.Tensor] = []
    logprob_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, sample_count, int(chunk_size)):
            end = min(int(start) + int(chunk_size), sample_count)
            local_i = _slice_dataclass(stage_batch.local_batch, int(start) * num_agents, int(end) * num_agents)
            action_i = actions[int(start) : int(end)]
            logprob_i, entropy_i, out_i = learner._stage_actor_eval_from_batch(
                BW_STAGE,
                local_i,
                action_i,
                num_agents,
                compute_entropy=True,
                latent_actions=None,
            )
            det_chunks.append(out_i.det_mean.reshape(end - start, num_agents, -1).detach())
            logprob_chunks.append(logprob_i.reshape(-1).detach())
            entropy_chunks.append(entropy_i.reshape(-1).detach())
            tau = getattr(out_i, "tau", None)
            kappa = getattr(out_i, "kappa", None)
            valid_count = getattr(out_i, "valid_count", None)
            if tau is not None:
                tau_chunks.append(tau.reshape(end - start, num_agents).detach())
            if kappa is not None:
                kappa_chunks.append(kappa.reshape(end - start, num_agents).detach())
            if valid_count is not None:
                valid_count_chunks.append(valid_count.reshape(end - start, num_agents).detach())

    det_mean = torch.cat(det_chunks, dim=0)
    valid_mask = (
        stage_batch.local_batch.gu_mask.to(device=device, dtype=torch.bool)
        & stage_batch.local_batch.bw_valid_mask.to(device=device, dtype=torch.bool)
    ).reshape(sample_count, num_agents, -1)
    out: dict[str, torch.Tensor] = {
        "det_mean": det_mean,
        "valid_mask": valid_mask,
        "logprob": torch.cat(logprob_chunks, dim=0),
        "entropy": torch.cat(entropy_chunks, dim=0),
    }
    if tau_chunks:
        out["tau"] = torch.cat(tau_chunks, dim=0)
    if kappa_chunks:
        out["kappa"] = torch.cat(kappa_chunks, dim=0)
    if valid_count_chunks:
        out["valid_count"] = torch.cat(valid_count_chunks, dim=0)
    return out


def _tensor_stats(prefix: str, value: torch.Tensor) -> dict[str, float]:
    x = value.detach().to(dtype=torch.float32).reshape(-1)
    if int(x.numel()) == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan")}
    return {
        f"{prefix}_mean": float(x.mean().cpu().item()),
        f"{prefix}_std": float(x.std(unbiased=False).cpu().item()),
        f"{prefix}_min": float(x.min().cpu().item()),
        f"{prefix}_max": float(x.max().cpu().item()),
    }


def _uniform_gap(det_mean: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """KL(det_mean || uniform over valid slots), per sample-agent row."""

    valid = valid_mask.to(dtype=torch.bool)
    p = det_mean.to(dtype=torch.float32).masked_fill(~valid, 0.0).clamp_min(1.0e-12)
    valid_count = valid.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    gap = torch.log(valid_count) - entropy
    return torch.where(valid_count >= 2.0, gap, torch.zeros_like(gap))


def _uniform_gap_stats(prefix: str, det_mean: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    gap = _uniform_gap(det_mean, valid_mask)
    valid_rows = valid_mask.to(dtype=torch.float32).sum(dim=-1) >= 2.0
    x = gap[valid_rows].detach().to(dtype=torch.float32).reshape(-1)
    if int(x.numel()) == 0:
        return {f"{prefix}_uniform_gap_mean": float("nan")}
    return {
        f"{prefix}_uniform_gap_mean": float(x.mean().cpu().item()),
        f"{prefix}_uniform_gap_std": float(x.std(unbiased=False).cpu().item()),
        f"{prefix}_uniform_gap_p50": float(torch.quantile(x, 0.50).cpu().item()),
        f"{prefix}_uniform_gap_p90": float(torch.quantile(x, 0.90).cpu().item()),
    }


def _bw_grad_group_norms(actor: torch.nn.Module) -> dict[str, float]:
    groups = {
        "score_head": "bw_policy.score_head",
        "tau_head": "bw_policy.tau_head",
        "kappa_head": "bw_policy.kappa_head",
        "bw_policy": "bw_policy",
    }
    out: dict[str, float] = {}
    named = list(actor.named_parameters())
    for label, prefix in groups.items():
        sq = 0.0
        count = 0
        for name, param in named:
            if not name.startswith(prefix):
                continue
            grad = param.grad
            if grad is None:
                continue
            g = grad.detach().to(dtype=torch.float32)
            sq += float((g * g).sum().cpu().item())
            count += int(g.numel())
        out[f"probe_grad_norm_{label}"] = float(sq ** 0.5)
        out[f"probe_grad_count_{label}"] = float(count)
    return out


def _bucket_metrics(name: str, score: torch.Tensor, alignment: torch.Tensor) -> dict[str, float]:
    score_f = score.detach().to(dtype=torch.float32).reshape(-1)
    align_f = alignment.detach().to(dtype=torch.float32).reshape(-1)
    out: dict[str, float] = {}
    if int(score_f.numel()) < 4:
        return out
    try:
        qs = torch.quantile(score_f, torch.tensor([0.25, 0.5, 0.75], device=score_f.device))
    except RuntimeError:
        return out
    bounds = [
        (torch.full((), -float("inf"), device=score_f.device), qs[0], "q0"),
        (qs[0], qs[1], "q1"),
        (qs[1], qs[2], "q2"),
        (qs[2], torch.full((), float("inf"), device=score_f.device), "q3"),
    ]
    for lo, hi, suffix in bounds:
        mask = (score_f >= lo) & (score_f <= hi if torch.isinf(hi) else score_f < hi)
        if bool(mask.any().detach().cpu().item()):
            vals = align_f[mask]
            out[f"{name}_{suffix}_count"] = float(mask.sum().detach().cpu().item())
            out[f"{name}_{suffix}_alignment_mean"] = float(vals.mean().detach().cpu().item())
            out[f"{name}_{suffix}_positive_frac"] = float((vals > 0).to(dtype=torch.float32).mean().detach().cpu().item())
    return out


def _alignment_metrics(
    *,
    stage_batch: Any,
    adv: torch.Tensor,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    device = adv.device
    actions = stage_batch.actions.to(device=device, dtype=torch.float32)
    valid = before["valid_mask"].to(device=device, dtype=torch.float32)
    det_old = before["det_mean"].to(device=device, dtype=torch.float32)
    det_new = after["det_mean"].to(device=device, dtype=torch.float32)
    sample_dir = (actions - det_old) * valid
    delta_mean = (det_new - det_old) * valid
    per_agent_dot = (sample_dir * delta_mean).sum(dim=-1)
    per_agent_action_l1 = sample_dir.abs().sum(dim=-1)
    per_agent_delta_l1 = delta_mean.abs().sum(dim=-1)
    per_sample_dot = per_agent_dot.sum(dim=-1)
    per_sample_action_l1 = per_agent_action_l1.sum(dim=-1)
    per_sample_delta_l1 = per_agent_delta_l1.sum(dim=-1)
    adv_f = adv.detach().to(device=device, dtype=torch.float32).reshape(-1)
    alignment = adv_f * per_sample_dot
    denom = (adv_f.abs() * per_sample_action_l1 * per_sample_delta_l1).clamp_min(1.0e-12)
    cosine_like = alignment / denom
    out: dict[str, float] = {
        "alignment_mean": float(alignment.mean().detach().cpu().item()),
        "alignment_std": float(alignment.std(unbiased=False).detach().cpu().item()),
        "alignment_positive_frac": float((alignment > 0).to(dtype=torch.float32).mean().detach().cpu().item()),
        "dot_mean": float(per_sample_dot.mean().detach().cpu().item()),
        "dot_abs_mean": float(per_sample_dot.abs().mean().detach().cpu().item()),
        "action_det_l1_mean": float(per_sample_action_l1.mean().detach().cpu().item()),
        "action_det_l1_p90": float(torch.quantile(per_sample_action_l1, 0.90).detach().cpu().item()),
        "delta_mean_l1_mean": float(per_sample_delta_l1.mean().detach().cpu().item()),
        "delta_mean_l1_p90": float(torch.quantile(per_sample_delta_l1, 0.90).detach().cpu().item()),
        "alignment_cosine_like_mean": float(cosine_like.mean().detach().cpu().item()),
        "alignment_cosine_like_positive_frac": float((cosine_like > 0).to(dtype=torch.float32).mean().detach().cpu().item()),
        "adv_action_dir_cov": float((adv_f * per_sample_action_l1).mean().detach().cpu().item()),
        "samples": float(int(adv_f.numel())),
    }
    out.update(_tensor_stats("adv", adv_f))
    out.update(_tensor_stats("old_logprob_replay", before["logprob"]))
    out.update(_tensor_stats("entropy_replay", before["entropy"]))
    if "tau" in before:
        out.update(_tensor_stats("tau", before["tau"]))
    if "kappa" in before:
        out.update(_tensor_stats("kappa", before["kappa"]))
    if "valid_count" in before:
        out.update(_tensor_stats("valid_count", before["valid_count"].to(dtype=torch.float32)))
    out.update(_bucket_metrics("action_l1", per_sample_action_l1, alignment))
    out.update(_bucket_metrics("adv_abs", adv_f.abs(), alignment))
    if "valid_count" in before:
        out.update(_bucket_metrics("valid_count", before["valid_count"].to(dtype=torch.float32).mean(dim=1), alignment))
    if "kappa" in before:
        out.update(_bucket_metrics("kappa", before["kappa"].mean(dim=1), alignment))
    return out


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_checkpoint_for_mode(mode: str) -> str:
    if mode == "current":
        return "runs/joint_mcgae_3uav20gu_positive_dynlr_precise_u300_20260512/checkpoint_update0300.pt"
    if mode == "new_fast":
        return "runs/joint_mcgae_3uav20gu_positive_nativefix_diag_u300_rerun_20260511/checkpoint_update0300.pt"
    return "runs/joint_mcgae_3uav20gu_positive_diag_u300/checkpoint_update0200.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a real BW PPO update moves det_mean toward high-advantage sampled actions. "
            "Modes emulate the two historical native Dirichlet changes without changing PyTorch actor code."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=BW_MODES, action="append", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--reward_mode", default="positive_weighted_workload_level")
    parser.add_argument("--seed", type=int, default=84210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_epochs", type=int, default=None)
    parser.add_argument("--critic_minibatches", type=int, default=None)
    parser.add_argument("--critic_update_microbatch_size", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--actor_epochs", type=int, default=None)
    parser.add_argument("--actor_minibatches", type=int, default=None)
    parser.add_argument("--eval_chunk_size", type=int, default=1024)
    parser.add_argument(
        "--probe_sgd_lr",
        type=float,
        default=1.0e-3,
        help="Small one-step SGD probe lr used to distinguish estimator direction from Adam/update effects.",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="Disable torch.compile for this audit if debugging generated kernels.",
    )
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if device.type == "cuda":
        _enable_strict_compile_global()

    modes = list(args.mode or BW_MODES)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for mode_idx, mode in enumerate(modes):
        _set_seed(int(args.seed) + mode_idx)
        cfg = load_config(args.config)
        _force_joint_config(cfg, reward_mode=str(args.reward_mode))
        cfg.bw_native_dirichlet_diagnostic_mode = str(mode)
        cfg.stage_actor_logprob_parity_check_enabled = False
        cfg.checkpoint_eval_enabled = False
        cfg.train_trace_enabled = False
        cfg.update_direction_probe_branch_enabled = False
        if bool(args.disable_compile):
            cfg.critic_compile_enabled = False
            cfg.actor_compile_enabled = False
        report_torch_compile_cache(context=f"bw_real_update_alignment:{mode}", device=device, cfg=cfg)

        learner = _make_joint_learner(cfg, device=device)
        _sync_danger_imitation_to_learner(learner, cfg)
        hparam_args = argparse.Namespace(
            actor_lr=args.actor_lr,
            accel_actor_lr=None,
            sat_actor_lr=None,
            bw_actor_lr=args.actor_lr,
            cold_critic_lr=None,
            cold_critic_epochs=None,
            tracking_critic_lr=args.critic_lr,
            tracking_critic_epochs=args.critic_epochs,
            critic_minibatches=args.critic_minibatches,
            critic_update_microbatch_size=args.critic_update_microbatch_size,
            actor_epochs=args.actor_epochs,
            actor_minibatches=args.actor_minibatches,
        )
        hparams = _resolve_training_hparams(cfg, hparam_args)
        critic_optimizer = torch.optim.Adam(
            [p for p in learner.critic.parameters() if p.requires_grad],
            lr=float(hparams["tracking_critic_lr"]),
        )
        actor_optimizers = {
            stage_id: torch.optim.Adam(
                _stage_optimizer_params(learner.actor, int(stage_id)),
                lr=float(hparams[f"actor_lr_{STAGE_NAME[int(stage_id)]}"]),
            )
            for stage_id in STAGES
        }
        checkpoint = Path(args.checkpoint or _default_checkpoint_for_mode(mode))
        checkpoint_state = _load_checkpoint(
            learner,
            checkpoint,
            device=device,
            critic_optimizer=critic_optimizer,
            actor_optimizers=actor_optimizers,
        )

        group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
        try:
            learner.bind_native_runtime_contract(group)
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
            t_collect = time.perf_counter()
            views, _returns, stage_targets, reward_stats = _collect_joint_rollout(
                learner,
                group,
                rollout_env_steps=int(args.rollout_env_steps),
                device=device,
                seed=int(args.seed) + mode_idx * 100_000,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collect_sec = time.perf_counter() - t_collect
        finally:
            close_structured_env_group(group)

        critic_lr = float(hparams["tracking_critic_lr"])
        critic_epochs = int(hparams["tracking_critic_epochs"])
        if args.critic_lr is None and args.critic_epochs is None:
            # We are auditing a checkpoint after training has already started,
            # so use the normal tracking fit, not the cold-start 20-epoch fit.
            critic_lr = float(hparams["tracking_critic_lr"])
            critic_epochs = int(hparams["tracking_critic_epochs"])
        t_critic = time.perf_counter()
        stage_values_after_critic: dict[int, torch.Tensor] = {}
        critic_metrics: dict[str, float] = {}
        for stage_id in STAGES:
            stage_name = STAGE_NAME[int(stage_id)]
            stats, after_values, _trace = _train_stage_critic_on_stage(
                learner,
                stage_id=int(stage_id),
                stage_batch=views.training_view.stage_batches[int(stage_id)],
                target=stage_targets[int(stage_id)],
                optimizer=critic_optimizer,
                lr=critic_lr,
                epochs=critic_epochs,
                minibatches=int(hparams["critic_minibatches"]),
                update_microbatch_size=int(hparams["critic_update_microbatch_size"]),
                eval_before_enabled=False,
                diagnose_timing=False,
            )
            stage_values_after_critic[int(stage_id)] = after_values.detach()
            for k, v in stats.items():
                critic_metrics[f"{stage_name}_{k}"] = float(v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        critic_sec = time.perf_counter() - t_critic

        bw_stage_batch_raw = views.training_view.stage_batches[BW_STAGE]
        _ret, bw_adv_raw, _values = _stage_gae_from_mc_targets(
            learner,
            stage_id=BW_STAGE,
            stage_batch=bw_stage_batch_raw,
            mc_target=stage_targets[BW_STAGE],
            device=device,
            stage_values=stage_values_after_critic[BW_STAGE],
        )
        bw_adv = _normalize_stage_advantage(
            bw_adv_raw,
            enabled=bool(getattr(cfg, "actor_advantage_normalize_enabled", True)),
        )
        bw_stage_batch = _actor_only_stage_batch(bw_stage_batch_raw)
        del views
        del stage_targets
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        before = _eval_bw_params(learner, bw_stage_batch, chunk_size=int(args.eval_chunk_size))
        actor_state_before = {k: v.detach().clone() for k, v in learner.actor.state_dict().items()}
        probe_stats: dict[str, float] = {}
        probe_lr = max(float(args.probe_sgd_lr), 0.0)
        if probe_lr > 0.0:
            probe_optimizer = torch.optim.SGD(_stage_optimizer_params(learner.actor, BW_STAGE), lr=probe_lr)
            probe_stats_raw = _stage_actor_update_full_stage(
                learner,
                stage_id=BW_STAGE,
                stage_batch=bw_stage_batch,
                stage_advantages=bw_adv,
                optimizer=probe_optimizer,
                epochs=1,
                minibatches=1,
                parity_dump_dir=None,
                parity_dump_tag=None,
                parity_topk=0,
                kl_stop_threshold=None,
                clip_stop_threshold=None,
            )
            probe = _eval_bw_params(learner, bw_stage_batch, chunk_size=int(args.eval_chunk_size))
            probe_delta_l1 = (
                (probe["det_mean"].to(dtype=torch.float32) - before["det_mean"].to(dtype=torch.float32)).abs()
                * before["valid_mask"].to(dtype=torch.float32)
            ).sum(dim=-1).sum(dim=-1)
            probe_alignment = _alignment_metrics(stage_batch=bw_stage_batch, adv=bw_adv, before=before, after=probe)
            probe_stats.update({f"probe_sgd_{k}": float(v) for k, v in probe_stats_raw.items()})
            probe_stats.update({f"probe_sgd_{k}": float(v) for k, v in probe_alignment.items()})
            probe_stats["probe_sgd_delta_mean_l1_mean_direct"] = float(probe_delta_l1.mean().detach().cpu().item())
            probe_stats.update(_uniform_gap_stats("probe_sgd", probe["det_mean"], before["valid_mask"]))
            probe_stats.update(_bw_grad_group_norms(learner.actor))
            learner.actor.load_state_dict(actor_state_before, strict=True)
            for param in learner.actor.parameters():
                param.grad = None
            sync_native = getattr(learner, "_sync_native_actor_cuda_bindings_after_update", None)
            if callable(sync_native):
                sync_native()
        else:
            probe_stats["probe_sgd_skipped"] = 1.0
        t_actor = time.perf_counter()
        actor_stats = _stage_actor_update_full_stage(
            learner,
            stage_id=BW_STAGE,
            stage_batch=bw_stage_batch,
            stage_advantages=bw_adv,
            optimizer=actor_optimizers[BW_STAGE],
            epochs=int(hparams["actor_epochs"]),
            minibatches=int(hparams["actor_minibatches"]),
            parity_dump_dir=None,
            parity_dump_tag=None,
            parity_topk=0,
            kl_stop_threshold=None,
            clip_stop_threshold=None,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        actor_sec = time.perf_counter() - t_actor
        after = _eval_bw_params(learner, bw_stage_batch, chunk_size=int(args.eval_chunk_size))
        metrics = _alignment_metrics(stage_batch=bw_stage_batch, adv=bw_adv, before=before, after=after)
        metrics.update(_uniform_gap_stats("before", before["det_mean"], before["valid_mask"]))
        metrics.update(_uniform_gap_stats("after", after["det_mean"], before["valid_mask"]))

        row: dict[str, float | str] = {
            "mode": str(mode),
            "checkpoint": str(checkpoint),
            "checkpoint_update": float(checkpoint_state.get("update", float("nan"))),
            "collect_sec": float(collect_sec),
            "critic_sec": float(critic_sec),
            "actor_sec": float(actor_sec),
            "reward_transition_mean": float(reward_stats.get("transition_reward_mean", float("nan"))),
            "bw_mc_return_mean": float(reward_stats.get("bw_mc_return_mean", float("nan"))),
            "critic_lr": float(critic_lr),
            "critic_epochs": float(critic_epochs),
            "actor_lr_bw": float(actor_optimizers[BW_STAGE].param_groups[0].get("lr", float("nan"))),
            "actor_epochs": float(hparams["actor_epochs"]),
            "actor_minibatches": float(hparams["actor_minibatches"]),
        }
        row.update({k: float(v) for k, v in critic_metrics.items()})
        row.update({k: float(v) for k, v in probe_stats.items()})
        row.update({k: float(v) for k, v in actor_stats.items()})
        row.update({k: float(v) for k, v in metrics.items()})
        rows.append(row)

        mode_payload = {
            "mode": mode,
            "checkpoint": str(checkpoint),
            "row": row,
        }
        (run_dir / f"bw_real_update_alignment_{mode}.json").write_text(
            json.dumps(mode_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{mode}: align={row['alignment_mean']:.4g} pos={row['alignment_positive_frac']:.3f} "
            f"action_l1={row['action_det_l1_mean']:.4g} delta_l1={row['delta_mean_l1_mean']:.4g} "
            f"kl={row.get('approx_kl_bw', float('nan')):.4g} clip={row.get('clip_frac_bw', float('nan')):.3f}",
            flush=True,
        )

    _write_csv(run_dir / "bw_real_update_alignment.csv", rows)
    print(f"Wrote {run_dir / 'bw_real_update_alignment.csv'}", flush=True)


if __name__ == "__main__":
    main()
