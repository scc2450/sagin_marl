from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.audit.audit_bw_broad2local_offline import _collect_panel_bank
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    order_pred = np.argsort(-pred, kind="mergesort")
    order_truth = np.argsort(-truth, kind="mergesort")
    rank_pred = np.empty_like(order_pred, dtype=np.float64)
    rank_truth = np.empty_like(order_truth, dtype=np.float64)
    rank_pred[order_pred] = np.arange(pred.size, dtype=np.float64)
    rank_truth[order_truth] = np.arange(truth.size, dtype=np.float64)
    if float(np.std(rank_pred)) <= 1.0e-12 or float(np.std(rank_truth)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(rank_pred, rank_truth)[0, 1])


def _pairwise_concordance_desc(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0e-9) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
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
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    return 1.0 if int(np.argmax(pred)) == int(np.argmax(truth)) else 0.0


def _load_actor(run_dir: Path, device: torch.device, actor_checkpoint: str | None):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = Path(actor_checkpoint) if actor_checkpoint is not None else run_dir / "actor_final.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    actor = bundle.actor.to(device).eval()
    return cfg, actor, actor_ckpt


def _bw_action_from_actor(actor, snapshot, device: torch.device, deterministic: bool) -> np.ndarray:
    bw_state = _to_device_dataclass(build_local_bw_states_from_snapshot(snapshot)[0], device)
    with torch.no_grad():
        out = actor.act_bw(bw_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _step_record(env, step_result) -> dict[str, float]:
    reward_parts = getattr(env, "last_reward_parts", {}) or {}
    arrival_ref = max(float(env._arrival_ref()), 1.0e-9)
    gu_q = float(np.sum(np.asarray(env.gu_queue, dtype=np.float32)) / arrival_ref)
    uav_q = float(np.sum(np.asarray(env.uav_queue, dtype=np.float32)) / arrival_ref)
    sat_q = float(np.sum(np.asarray(env.sat_queue, dtype=np.float32)) / arrival_ref)
    sys_q = gu_q + uav_q + sat_q
    gu_drop = float(np.sum(np.asarray(env.gu_drop, dtype=np.float32)) / arrival_ref)
    uav_drop = float(np.sum(np.asarray(env.uav_drop, dtype=np.float32)) / arrival_ref)
    sat_drop = float(np.sum(np.asarray(env.sat_drop, dtype=np.float32)) / arrival_ref)
    sys_drop = gu_drop + uav_drop + sat_drop
    gu_out = float(np.sum(np.asarray(env.last_gu_outflow, dtype=np.float32)) / arrival_ref)
    pre_backlog = float(reward_parts.get("pre_backlog_steps_eval", gu_q + uav_q))
    return {
        "env_reward": float(next(iter(step_result.rewards.values()))),
        "bw_access_reward": float(step_result.bw_access_reward),
        "bw_weighted_workload_delta_reward": float(step_result.bw_weighted_workload_delta_reward),
        "bw_weighted_workload_level_reward": float(step_result.bw_weighted_workload_level_reward),
        "bw_gu_queue_level_reward": float(step_result.bw_gu_queue_level_reward),
        "bw_system_queue_level_reward": float(step_result.bw_system_queue_level_reward),
        "bw_gu_service_queue_reward": float(step_result.bw_gu_service_queue_reward),
        "x_acc": float(reward_parts.get("x_acc", 0.0)),
        "x_rel": float(reward_parts.get("x_rel", 0.0)),
        "d_pre": float(reward_parts.get("d_pre", gu_drop + uav_drop)),
        "pre_b": pre_backlog,
        "log_pre_b": float(math.log1p(max(pre_backlog, 0.0))),
        "gu_q": gu_q,
        "uav_q": uav_q,
        "sat_q": sat_q,
        "sys_q": sys_q,
        "gu_drop": gu_drop,
        "sys_drop": sys_drop,
        "gu_out": gu_out,
    }


def _rollout_record_traj(
    *,
    snapshot_state: dict[str, object],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    follow_mode: str,
    follow_deterministic: bool,
) -> list[dict[str, float]]:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    follow_mode_l = str(follow_mode).strip().lower()
    if follow_mode_l not in {"queue_aware", "policy"}:
        raise ValueError(f"Unsupported follow_mode: {follow_mode}")
    try:
        probe_driver.load_bw_stage_state(snapshot_state)
        action = np.asarray(first_action, dtype=np.float32)
        records: list[dict[str, float]] = []
        for step_idx in range(int(k_steps)):
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            records.append(_step_record(probe_env, step_result))
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done or step_idx + 1 >= int(k_steps):
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = probe_driver.run_accel_stage(accel_zero)
            select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
            z2 = probe_driver.run_sat_stage(np.full((cfg.num_uav, select_k), -1, dtype=np.int64))
            if follow_mode_l == "queue_aware":
                obs = {agent: probe_env._get_obs(idx) for idx, agent in enumerate(probe_env.agents)}
                action = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
            else:
                bw_snapshot = probe_driver.build_bw_stage_snapshot(z2)
                action = _bw_action_from_actor(actor, bw_snapshot, device, deterministic=follow_deterministic)
        return records
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def _candidate_reward_fns() -> list[tuple[str, Callable[[dict[str, float]], float]]]:
    fns: list[tuple[str, Callable[[dict[str, float]], float]]] = [
        ("env_reward", lambda rec: float(rec["env_reward"])),
        ("bw_access_reward", lambda rec: float(rec["bw_access_reward"])),
        ("weighted_workload_delta", lambda rec: float(rec["bw_weighted_workload_delta_reward"])),
        ("weighted_workload_level", lambda rec: float(rec["bw_weighted_workload_level_reward"])),
        ("gu_queue_level", lambda rec: float(rec["bw_gu_queue_level_reward"])),
        ("system_queue_level", lambda rec: float(rec["bw_system_queue_level_reward"])),
        ("gu_service_queue", lambda rec: float(rec["bw_gu_service_queue_reward"])),
    ]
    a_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    b_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    rel_grid = [0.0, 0.25, 0.5, 1.0]
    for a in a_grid:
        fns.append((f"neg_guq_gudrop_a{a}", lambda rec, a=a: float(-rec["gu_q"] - a * rec["gu_drop"])))
        fns.append((f"neg_sysq_sysdrop_a{a}", lambda rec, a=a: float(-rec["sys_q"] - a * rec["sys_drop"])))
    for a in a_grid:
        for b in b_grid:
            fns.append(
                (
                    f"xacc_drop_logbacklog_a{a}_b{b}",
                    lambda rec, a=a, b=b: float(rec["x_acc"] - a * rec["d_pre"] - b * rec["log_pre_b"]),
                )
            )
            fns.append(
                (
                    f"xacc_drop_linbacklog_a{a}_b{b}",
                    lambda rec, a=a, b=b: float(rec["x_acc"] - a * rec["d_pre"] - b * rec["pre_b"]),
                )
            )
            fns.append(
                (
                    f"guout_gudrop_guq_a{a}_b{b}",
                    lambda rec, a=a, b=b: float(rec["gu_out"] - a * rec["gu_drop"] - b * rec["gu_q"]),
                )
            )
            fns.append(
                (
                    f"guout_sysdrop_sysq_a{a}_b{b}",
                    lambda rec, a=a, b=b: float(rec["gu_out"] - a * rec["sys_drop"] - b * rec["sys_q"]),
                )
            )
            for w_rel in rel_grid:
                fns.append(
                    (
                        f"xacc_rel_drop_logbacklog_w{w_rel}_a{a}_b{b}",
                        lambda rec, w_rel=w_rel, a=a, b=b: float(
                            rec["x_acc"] + w_rel * rec["x_rel"] - a * rec["d_pre"] - b * rec["log_pre_b"]
                        ),
                    )
                )
    return fns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--actor_checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--panel_states", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--seed", type=int, default=22345)
    parser.add_argument("--heuristic_bw_source", type=str, default="queue_aware")
    parser.add_argument("--k_steps", type=int, default=20)
    parser.add_argument("--follow_mode", choices=["queue_aware", "policy"], default="queue_aware")
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--panel_random_count", type=int, default=2)
    parser.add_argument("--prefer_policy_gap_min", type=float, default=1.0e-3)
    parser.add_argument("--min_best_gap", type=float, default=0.02)
    parser.add_argument("--bw_deterministic_opt_steps", type=int, default=8)
    parser.add_argument("--bw_deterministic_step_size", type=float, default=0.5)
    parser.add_argument("--topn", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed))
    run_dir = Path(args.run_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg, actor, ckpt_path = _load_actor(
        run_dir,
        device,
        None if args.actor_checkpoint is None else str(args.actor_checkpoint),
    )
    bank = _collect_panel_bank(
        run_dir=run_dir,
        update=0,
        device=device,
        actor_checkpoint=str(ckpt_path),
        output_dir=out_path.parent,
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
    state_action_trajs: list[tuple[list[str], list[list[dict[str, float]]]]] = []
    for entry in entries:
        candidate_names = list(entry["panel_names"]) + ["policy_det"]
        candidate_actions = [np.asarray(a, dtype=np.float32) for a in entry["panel_actions"]] + [
            np.asarray(entry["policy_det_action"], dtype=np.float32)
        ]
        trajectories = [
            _rollout_record_traj(
                snapshot_state=entry["snapshot_state"],
                first_action=action,
                cfg=cfg,
                actor=actor,
                device=device,
                k_steps=int(args.k_steps),
                follow_mode=str(args.follow_mode),
                follow_deterministic=(str(args.policy_mode) == "deterministic"),
            )
            for action in candidate_actions
        ]
        state_action_trajs.append((candidate_names, trajectories))

    gamma = float(cfg.gamma)
    results: list[dict[str, float | str]] = []
    for name, reward_fn in _candidate_reward_fns():
        spearman_values: list[float] = []
        pairwise_values: list[float] = []
        top1_values: list[float] = []
        for _candidate_names, trajectories in state_action_trajs:
            immediate_scores: list[float] = []
            true_scores: list[float] = []
            for traj in trajectories:
                reward_sequence = [float(reward_fn(record)) for record in traj]
                immediate_scores.append(float(reward_sequence[0]))
                total = 0.0
                discount = 1.0
                for reward_value in reward_sequence:
                    total += discount * float(reward_value)
                    discount *= gamma
                true_scores.append(float(total))
            immediate_arr = np.asarray(immediate_scores, dtype=np.float64)
            true_arr = np.asarray(true_scores, dtype=np.float64)
            spearman_values.append(_safe_spearman_desc(immediate_arr, true_arr))
            pairwise_values.append(_pairwise_concordance_desc(immediate_arr, true_arr))
            top1_values.append(_top1_hit_desc(immediate_arr, true_arr))
        results.append(
            {
                "name": name,
                "spearman_mean": float(np.mean(np.asarray(spearman_values, dtype=np.float64))),
                "pairwise_mean": float(np.mean(np.asarray(pairwise_values, dtype=np.float64))),
                "top1_mean": float(np.mean(np.asarray(top1_values, dtype=np.float64))),
            }
        )

    results.sort(
        key=lambda row: (
            float(row["spearman_mean"]),
            float(row["pairwise_mean"]),
            float(row["top1_mean"]),
        ),
        reverse=True,
    )
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "follow_mode": str(args.follow_mode),
        "policy_mode": str(args.policy_mode),
        "states": int(len(entries)),
        "k_steps": int(args.k_steps),
        "top_results": results[: int(args.topn)],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
