from __future__ import annotations

import argparse
import csv
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

from scripts.diagnostics.audit.audit_bw_broad2local_offline import _collect_panel_bank, _rollout_panel_action_scores
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import world_state_to_torch
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    return _safe_corr(_rankdata_desc(np.asarray(pred, dtype=np.float64)), _rankdata_desc(np.asarray(truth, dtype=np.float64)))


def _pairwise_concordance_desc(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0e-9) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
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
    return float(hits / total) if total > 0.0 else 0.0


def _top1_hit_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    if float(np.std(pred)) <= 1.0e-12 or float(np.std(truth)) <= 1.0e-12:
        return 0.0
    return 1.0 if int(np.argmax(pred)) == int(np.argmax(truth)) else 0.0


def _load_actor_critic(
    run_dir: Path,
    update: int,
    device: torch.device,
    *,
    actor_checkpoint: str | None,
    critic_checkpoint: str | None,
) -> tuple[Any, Any, StructuredMAPPO, Any]:
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = Path(actor_checkpoint) if actor_checkpoint is not None else run_dir / f"actor_u{int(update):04d}.pt"
    critic_ckpt = Path(critic_checkpoint) if critic_checkpoint is not None else run_dir / f"critic_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    load_checkpoint_forgiving(bundle.critic, str(critic_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    bundle.critic.to(device).eval()
    learner = StructuredMAPPO(
        actor=bundle.actor,
        critic=bundle.critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        device=device,
        cfg=cfg,
    )
    return cfg, bundle.actor, learner, bundle.critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--critic_checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--panel_states", type=int, default=24)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument("--prefer_policy_gap_min", type=float, default=1.0e-3)
    parser.add_argument("--min_best_gap", type=float, default=0.02)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=22345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    cfg, actor, learner, critic = _load_actor_critic(
        run_dir,
        int(args.update),
        device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        critic_checkpoint=None if args.critic_checkpoint is None else str(args.critic_checkpoint),
    )
    bank = _collect_panel_bank(
        run_dir=run_dir,
        update=int(args.update),
        device=device,
        actor_checkpoint=None if args.actor_checkpoint is None else str(args.actor_checkpoint),
        output_dir=out_dir,
        episodes=int(args.episodes),
        panel_states=int(args.panel_states),
        policy_mode=str(args.policy_mode),
        panel_random_count=int(args.panel_random_count),
        heuristic_bw_source=str(args.heuristic_bw_source),
        k_steps=int(args.k_steps),
        seed=int(args.seed),
        prefer_policy_gap_min=float(args.prefer_policy_gap_min),
        min_best_gap=float(args.min_best_gap),
        num_envs=int(args.num_envs),
        bw_deterministic_readout="latent_mean_pushforward",
        bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
        bw_deterministic_step_size=float(args.bw_deterministic_step_size),
        bw_parameterization=None,
        bw_alpha_init_bias=None,
        bw_alpha_max=None,
        bw_tau_enabled=None,
        bw_tau_init_bias=None,
        bw_tau_min=None,
        bw_tau_max=None,
        vec_backend=str(args.vec_backend),
    )
    entries = list(bank["entries"])
    eval_group = make_structured_env_group(cfg, num_envs=max(1, int(args.num_envs)), backend=str(args.vec_backend))
    probe_driver = make_structured_driver(cfg, mode="script")
    pooled_true: list[float] = []
    pooled_proxy: list[float] = []
    pooled_boot: list[float] = []
    pooled_adv: list[float] = []
    pooled_proxy_active: list[float] = []

    state_rows: list[dict[str, Any]] = []
    proxy_spearman: list[float] = []
    proxy_pairwise: list[float] = []
    proxy_top1: list[float] = []
    boot_spearman: list[float] = []
    boot_pairwise: list[float] = []
    boot_top1: list[float] = []
    adv_spearman: list[float] = []
    adv_pairwise: list[float] = []
    adv_top1: list[float] = []
    proxy_active_frac: list[float] = []

    try:
        for idx, entry in enumerate(entries):
            local_state = _to_device_dataclass(entry["local_state"], device)
            sample_actions: list[np.ndarray] = []
            with torch.inference_mode():
                for _ in range(int(args.sample_count)):
                    sample_actions.append(
                        np.asarray(actor.act_bw(local_state, deterministic=False).action.detach().cpu().numpy(), dtype=np.float32)
                    )
            true_scores = _rollout_panel_action_scores(
                eval_group,
                entry["snapshot_state"],
                panel_actions=sample_actions,
                follow_actor=actor,
                follow_device=device,
                follow_deterministic=(str(args.policy_mode) == "deterministic"),
                bw_deterministic_readout="latent_mean_pushforward",
                bw_deterministic_opt_steps=int(args.bw_deterministic_opt_steps),
                bw_deterministic_step_size=float(args.bw_deterministic_step_size),
                gamma=float(cfg.gamma),
                k_steps=int(args.k_steps),
            )
            current_world = _to_device_dataclass(world_state_to_torch(entry["snapshot"].world_state), device)
            with torch.no_grad():
                current_bw_value = float(critic.value_bw(current_world).reshape(-1)[0].item())

            proxy_scores: list[float] = []
            boot_scores: list[float] = []
            adv_scores: list[float] = []
            active_flags: list[float] = []
            for action in sample_actions:
                probe_driver.load_bw_stage_state(entry["snapshot_state"])
                step_result, next_world_state = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
                immediate_reward = float(next(iter(step_result.rewards.values())))
                next_value = float(learner.bootstrap_value(next_world_state))
                boot_score = float(immediate_reward + float(cfg.gamma) * next_value)
                adv_score = float(boot_score - current_bw_value)
                proxy_score = 0.0
                is_active = 0.0
                if step_result.bw_flow_proxy_scores is not None and step_result.bw_flow_proxy_mask is not None:
                    sample_credit, sample_active, _metrics = learner._bw_counterfactual_credit_from_proxy(
                        torch.as_tensor(step_result.bw_flow_proxy_scores, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.as_tensor(step_result.bw_flow_proxy_mask, dtype=torch.float32, device=device).unsqueeze(0),
                    )
                    proxy_score = float(sample_credit.reshape(-1)[0].detach().cpu().item())
                    is_active = float(sample_active.reshape(-1)[0].detach().cpu().item())
                proxy_scores.append(proxy_score)
                boot_scores.append(boot_score)
                adv_scores.append(adv_score)
                active_flags.append(is_active)

            true_arr = np.asarray(true_scores, dtype=np.float64)
            proxy_arr = np.asarray(proxy_scores, dtype=np.float64)
            boot_arr = np.asarray(boot_scores, dtype=np.float64)
            adv_arr = np.asarray(adv_scores, dtype=np.float64)
            pooled_true.extend(true_arr.tolist())
            pooled_proxy.extend(proxy_arr.tolist())
            pooled_boot.extend(boot_arr.tolist())
            pooled_adv.extend(adv_arr.tolist())
            pooled_proxy_active.extend(active_flags)

            row = {
                "index": int(idx),
                "sample_count": int(len(sample_actions)),
                "true_score_mean": float(np.mean(true_arr)),
                "proxy_score_mean": float(np.mean(proxy_arr)),
                "boot_score_mean": float(np.mean(boot_arr)),
                "adv_score_mean": float(np.mean(adv_arr)),
                "proxy_active_frac": float(np.mean(np.asarray(active_flags, dtype=np.float64))),
                "proxy_spearman": _safe_spearman_desc(proxy_arr, true_arr),
                "proxy_pairwise_acc": _pairwise_concordance_desc(proxy_arr, true_arr),
                "proxy_top1_hit": _top1_hit_desc(proxy_arr, true_arr),
                "boot_spearman": _safe_spearman_desc(boot_arr, true_arr),
                "boot_pairwise_acc": _pairwise_concordance_desc(boot_arr, true_arr),
                "boot_top1_hit": _top1_hit_desc(boot_arr, true_arr),
                "adv_spearman": _safe_spearman_desc(adv_arr, true_arr),
                "adv_pairwise_acc": _pairwise_concordance_desc(adv_arr, true_arr),
                "adv_top1_hit": _top1_hit_desc(adv_arr, true_arr),
            }
            state_rows.append(row)
            proxy_spearman.append(float(row["proxy_spearman"]))
            proxy_pairwise.append(float(row["proxy_pairwise_acc"]))
            proxy_top1.append(float(row["proxy_top1_hit"]))
            boot_spearman.append(float(row["boot_spearman"]))
            boot_pairwise.append(float(row["boot_pairwise_acc"]))
            boot_top1.append(float(row["boot_top1_hit"]))
            adv_spearman.append(float(row["adv_spearman"]))
            adv_pairwise.append(float(row["adv_pairwise_acc"]))
            adv_top1.append(float(row["adv_top1_hit"]))
            proxy_active_frac.append(float(row["proxy_active_frac"]))
    finally:
        close_structured_env_group(eval_group)
        close_fn = getattr(probe_driver.env, "close", None)
        if callable(close_fn):
            close_fn()

    pooled_true_arr = np.asarray(pooled_true, dtype=np.float64)
    pooled_proxy_arr = np.asarray(pooled_proxy, dtype=np.float64)
    pooled_boot_arr = np.asarray(pooled_boot, dtype=np.float64)
    pooled_adv_arr = np.asarray(pooled_adv, dtype=np.float64)

    summary = {
        "run_dir": str(run_dir),
        "update": int(args.update),
        "actor_checkpoint": None if args.actor_checkpoint is None else str(Path(args.actor_checkpoint).resolve()),
        "critic_checkpoint": None if args.critic_checkpoint is None else str(Path(args.critic_checkpoint).resolve()),
        "policy_mode": str(args.policy_mode),
        "panel_state_count": int(len(state_rows)),
        "sample_count": int(args.sample_count),
        "k_steps": int(args.k_steps),
        "vec_backend": str(args.vec_backend),
        "device": str(device),
        "proxy_active_frac": _summarize(proxy_active_frac),
        "per_state": {
            "proxy_vs_true": {
                "spearman": _summarize(proxy_spearman),
                "pairwise_acc": _summarize(proxy_pairwise),
                "top1_hit_mean": float(np.mean(np.asarray(proxy_top1, dtype=np.float64))) if proxy_top1 else 0.0,
            },
            "boot_vs_true": {
                "spearman": _summarize(boot_spearman),
                "pairwise_acc": _summarize(boot_pairwise),
                "top1_hit_mean": float(np.mean(np.asarray(boot_top1, dtype=np.float64))) if boot_top1 else 0.0,
            },
            "adv_vs_true": {
                "spearman": _summarize(adv_spearman),
                "pairwise_acc": _summarize(adv_pairwise),
                "top1_hit_mean": float(np.mean(np.asarray(adv_top1, dtype=np.float64))) if adv_top1 else 0.0,
            },
        },
        "pooled": {
            "proxy_vs_true": {
                "pearson": _safe_corr(pooled_proxy_arr, pooled_true_arr),
                "spearman": _safe_spearman_desc(pooled_proxy_arr, pooled_true_arr),
                "pairwise_acc": _pairwise_concordance_desc(pooled_proxy_arr, pooled_true_arr),
                "top1_hit": _top1_hit_desc(pooled_proxy_arr, pooled_true_arr),
            },
            "boot_vs_true": {
                "pearson": _safe_corr(pooled_boot_arr, pooled_true_arr),
                "spearman": _safe_spearman_desc(pooled_boot_arr, pooled_true_arr),
                "pairwise_acc": _pairwise_concordance_desc(pooled_boot_arr, pooled_true_arr),
                "top1_hit": _top1_hit_desc(pooled_boot_arr, pooled_true_arr),
            },
            "adv_vs_true": {
                "pearson": _safe_corr(pooled_adv_arr, pooled_true_arr),
                "spearman": _safe_spearman_desc(pooled_adv_arr, pooled_true_arr),
                "pairwise_acc": _pairwise_concordance_desc(pooled_adv_arr, pooled_true_arr),
                "top1_hit": _top1_hit_desc(pooled_adv_arr, pooled_true_arr),
            },
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "per_state_rows.csv", state_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
