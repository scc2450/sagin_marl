from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy,
    cluster_center_queue_aware_policy,
    centroid_accel_policy,
    queue_aware_policy,
    queue_aware_bw_policy,
    queue_aware_sat_policy,
    random_accel_policy,
    random_bw_policy,
    random_sat_policy,
    uniform_bw_policy,
    uniform_sat_policy,
    zero_accel_policy,
)
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.rl.structured_train import make_structured_env
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.utils.progress import Progress


def _baseline_actions(
    baseline: str,
    obs_list,
    cfg,
    num_agents: int,
    env=None,
    rng: np.random.Generator | None = None,
):
    if baseline in ("zero_accel", "fixed"):
        return zero_accel_policy(num_agents), None, None
    if baseline == "random_accel":
        return random_accel_policy(num_agents, rng=rng), None, None
    if baseline == "cluster_center":
        centers = None if env is None else getattr(env, "gu_cluster_centers", None)
        counts = None if env is None else getattr(env, "gu_cluster_counts", None)
        return cluster_center_accel_policy(obs_list, cfg, centers, counts), None, None
    if baseline == "cluster_center_queue_aware":
        centers = None if env is None else getattr(env, "gu_cluster_centers", None)
        counts = None if env is None else getattr(env, "gu_cluster_counts", None)
        return cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
    if baseline == "centroid":
        gain = float(getattr(cfg, "baseline_centroid_gain", 2.0))
        queue_weighted = bool(getattr(cfg, "baseline_centroid_queue_weighted", True))
        return centroid_accel_policy(obs_list, gain=gain, queue_weighted=queue_weighted), None, None
    if baseline == "uniform_bw":
        return zero_accel_policy(num_agents), uniform_bw_policy(num_agents, cfg.users_obs_max), None
    if baseline == "random_bw":
        return zero_accel_policy(num_agents), random_bw_policy(num_agents, cfg, rng=rng), None
    if baseline == "queue_aware_bw":
        return zero_accel_policy(num_agents), queue_aware_bw_policy(obs_list, cfg), None
    if baseline == "uniform_sat":
        return zero_accel_policy(num_agents), None, uniform_sat_policy(num_agents, cfg.sats_obs_max)
    if baseline == "random_sat":
        return zero_accel_policy(num_agents), None, random_sat_policy(num_agents, cfg, rng=rng)
    if baseline == "queue_aware_sat":
        return zero_accel_policy(num_agents), None, queue_aware_sat_policy(obs_list, cfg)
    if baseline == "queue_aware":
        return queue_aware_policy(obs_list, cfg)
    raise ValueError(f"Unknown baseline: {baseline}")


def _normalize_exec_source(raw: str | None) -> str:
    src = str("policy" if raw is None else raw).strip().lower()
    allowed = {"policy", "teacher", "heuristic", "zero"}
    if src not in allowed:
        raise ValueError(f"Invalid exec source '{raw}'. Allowed: {sorted(allowed)}")
    return src


def _select_exec_values(
    source: str,
    policy_values: np.ndarray | None,
    teacher_values: np.ndarray | None,
    heuristic_values: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if source == "policy" and policy_values is not None:
        return np.asarray(policy_values, dtype=np.float32)
    if source == "teacher" and teacher_values is not None:
        return np.asarray(teacher_values, dtype=np.float32)
    if source == "heuristic" and heuristic_values is not None:
        return np.asarray(heuristic_values, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _hybrid_bw_sat_actions(
    mode: str,
    obs_list,
    cfg,
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "none":
        return None, None
    if mode in {"queue_aware", "queue_aware_bw"}:
        _, bw_logits, sat_logits = queue_aware_policy(obs_list, cfg)
        return bw_logits, sat_logits
    if mode == "uniform_bw":
        return uniform_bw_policy(num_agents, cfg.users_obs_max), None
    if mode == "random_bw":
        return random_bw_policy(num_agents, cfg, rng=rng), None
    if mode == "uniform_sat":
        return None, uniform_sat_policy(num_agents, cfg.sats_obs_max)
    if mode == "random_sat":
        return None, random_sat_policy(num_agents, cfg, rng=rng)
    if mode == "queue_aware_sat":
        return None, queue_aware_sat_policy(obs_list, cfg)
    raise ValueError(f"Unknown hybrid_bw_sat mode: {mode}")


def _resolve_paths(
    run_dir: str | None,
    checkpoint: str | None,
    out: str | None,
    baseline: str,
) -> tuple[str, str]:
    use_baseline = baseline != "none"
    if run_dir:
        checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        if out is None:
            suffix = f"reward_scale_{baseline}.csv" if use_baseline else "reward_scale_trained.csv"
            out = os.path.join(run_dir, suffix)
    else:
        checkpoint = checkpoint or "runs/phase1/actor.pt"
        out = out or ("runs/phase1/reward_scale_baseline.csv" if use_baseline else "runs/phase1/reward_scale_trained.csv")
    return checkpoint, out


def _safe_array(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stat_summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "abs_median": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    abs_arr = np.abs(arr)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_median": float(np.median(abs_arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _abs_stat_summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "abs_mean": 0.0,
            "abs_std": 0.0,
            "abs_median": 0.0,
            "abs_p10": 0.0,
            "abs_p50": 0.0,
            "abs_p90": 0.0,
        }
    abs_arr = np.abs(arr)
    return {
        "abs_mean": float(np.mean(abs_arr)),
        "abs_std": float(np.std(abs_arr)),
        "abs_median": float(np.median(abs_arr)),
        "abs_p10": float(np.percentile(abs_arr, 10)),
        "abs_p50": float(np.percentile(abs_arr, 50)),
        "abs_p90": float(np.percentile(abs_arr, 90)),
    }


def _append(metric_signals, metric_contribs, group: str, name: str, signal: float, contrib: float) -> None:
    key = f"{group}:{name}"
    metric_signals[key].append(float(signal))
    metric_contribs[key].append(float(contrib))


def _queue_split_contribs(cfg, parts: dict) -> Dict[str, float]:
    queue_mode = str(parts.get("queue_delta_mode", "total") or "total").strip().lower()
    queue_weight = float(parts.get("queue_weight", getattr(cfg, "omega_q", 0.0)))
    q_delta_weight = float(parts.get("q_delta_weight", getattr(cfg, "eta_q_delta", 0.0)))
    w_gu = float(getattr(cfg, "omega_q_gu", 0.0) or 0.0)
    w_uav = float(getattr(cfg, "omega_q_uav", 0.0) or 0.0)
    w_sat = float(getattr(cfg, "omega_q_sat", 0.0) or 0.0)
    w_sum = abs(w_gu) + abs(w_uav) + abs(w_sat)
    split = {
        "queue_gu": 0.0,
        "queue_uav": 0.0,
        "queue_sat": 0.0,
        "qdelta_gu": 0.0,
        "qdelta_uav": 0.0,
        "qdelta_sat": 0.0,
    }
    if w_sum < 1e-9:
        return split
    split["queue_gu"] = -queue_weight * (w_gu / w_sum) * float(parts.get("queue_pen_gu", 0.0))
    split["queue_uav"] = -queue_weight * (w_uav / w_sum) * float(parts.get("queue_pen_uav", 0.0))
    split["queue_sat"] = -queue_weight * (w_sat / w_sum) * float(parts.get("queue_pen_sat", 0.0))
    if queue_mode == "weighted":
        split["qdelta_gu"] = q_delta_weight * (w_gu / w_sum) * float(parts.get("queue_delta_gu", 0.0))
        split["qdelta_uav"] = q_delta_weight * (w_uav / w_sum) * float(parts.get("queue_delta_uav", 0.0))
        split["qdelta_sat"] = q_delta_weight * (w_sat / w_sum) * float(parts.get("queue_delta_sat", 0.0))
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=[
            "none",
            "fixed",
            "zero_accel",
            "random_accel",
            "cluster_center",
            "cluster_center_queue_aware",
            "centroid",
            "queue_aware",
            "uniform_bw",
            "random_bw",
            "queue_aware_bw",
            "uniform_sat",
            "random_sat",
            "queue_aware_sat",
        ],
    )
    parser.add_argument(
        "--hybrid_bw_sat",
        type=str,
        default="none",
        choices=[
            "none",
            "queue_aware",
            "uniform_bw",
            "random_bw",
            "queue_aware_bw",
            "uniform_sat",
            "random_sat",
            "queue_aware_sat",
        ],
    )
    parser.add_argument("--episode_seed_base", type=int, default=None)
    args = parser.parse_args()

    args.checkpoint, args.out = _resolve_paths(args.run_dir, args.checkpoint, args.out, args.baseline)

    cfg = load_config(args.config)
    if args.baseline in {"cluster_center", "cluster_center_queue_aware"}:
        cfg.avoidance_enabled = True
        cfg.pairwise_hard_filter_enabled = True
    env = make_structured_env(cfg, backend="sync")
    use_baseline = args.baseline != "none"
    use_hybrid = args.hybrid_bw_sat != "none"
    if use_baseline and use_hybrid:
        raise ValueError("Hybrid bw/sat is only valid when baseline=none.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = None
    teacher_actor = None
    exec_accel_source = _normalize_exec_source(getattr(cfg, "exec_accel_source", "policy"))
    exec_bw_source = _normalize_exec_source(getattr(cfg, "exec_bw_source", "policy"))
    exec_sat_source = _normalize_exec_source(getattr(cfg, "exec_sat_source", "policy"))
    teacher_deterministic = bool(getattr(cfg, "exec_teacher_deterministic", True))
    need_teacher_exec = "teacher" in {exec_accel_source, exec_bw_source, exec_sat_source}
    need_heuristic_exec = "heuristic" in {exec_accel_source, exec_bw_source, exec_sat_source}

    if not use_baseline:
        obs, _ = env.reset()
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        info = load_checkpoint_forgiving(actor, args.checkpoint, map_location=device, strict=not use_hybrid)
        if info.get("adapted_keys"):
            print(f"Loaded actor with adapted tensors from {args.checkpoint}: {len(info['adapted_keys'])}")
        actor.eval()
        if need_teacher_exec:
            teacher_path = getattr(cfg, "exec_teacher_actor_path", None) or args.checkpoint
            teacher_actor = ActorNet(obs_dim, cfg).to(device)
            info = load_checkpoint_forgiving(teacher_actor, teacher_path, map_location=device)
            if info.get("adapted_keys"):
                print(f"Loaded teacher actor with adapted tensors from {teacher_path}: {len(info['adapted_keys'])}")
            teacher_actor.eval()

    metric_signals: dict[str, list[float]] = defaultdict(list)
    metric_contribs: dict[str, list[float]] = defaultdict(list)
    progress = Progress(args.episodes, desc="RewardScale")
    total_steps = 0

    for ep in range(args.episodes):
        ep_seed = None if args.episode_seed_base is None else int(args.episode_seed_base) + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        while not done:
            if use_baseline:
                obs_list = list(obs.values())
                accel_actions, bw_logits, sat_logits = _baseline_actions(
                    args.baseline, obs_list, cfg, len(env.agents), env=env, rng=env.rng
                )
                actions = assemble_actions(
                    cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits
                )
            else:
                obs_list = list(obs.values())
                obs_batch = batch_flatten_obs(obs_list, cfg)
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    policy_out = actor.act(obs_tensor, deterministic=True)
                    teacher_accel = None
                    teacher_bw = None
                    teacher_sat = None
                    if teacher_actor is not None:
                        teacher_out = teacher_actor.act(obs_tensor, deterministic=teacher_deterministic)
                        teacher_accel = teacher_out.accel.cpu().numpy()
                        teacher_bw = (
                            teacher_out.bw_logits.cpu().numpy() if teacher_out.bw_logits is not None else None
                        )
                        teacher_sat = (
                            teacher_out.sat_logits.cpu().numpy() if teacher_out.sat_logits is not None else None
                        )
                heur_accel = None
                heur_bw = None
                heur_sat = None
                if need_heuristic_exec:
                    heur_accel, heur_bw, heur_sat = queue_aware_policy(obs_list, cfg)
                policy_accel = policy_out.accel.cpu().numpy()
                policy_bw = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
                policy_sat = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
                if use_hybrid:
                    hybrid_bw, hybrid_sat = _hybrid_bw_sat_actions(
                        args.hybrid_bw_sat,
                        obs_list,
                        cfg,
                        len(env.agents),
                        rng=env.rng,
                    )
                    if hybrid_bw is not None:
                        policy_bw = hybrid_bw
                    if hybrid_sat is not None:
                        policy_sat = hybrid_sat
                accel_actions = _select_exec_values(
                    exec_accel_source,
                    policy_accel,
                    teacher_accel,
                    heur_accel,
                    (len(env.agents), 2),
                )
                bw_logits = None
                sat_logits = None
                if cfg.enable_bw_action:
                    bw_logits = _select_exec_values(
                        exec_bw_source,
                        policy_bw,
                        teacher_bw,
                        heur_bw,
                        (len(env.agents), cfg.users_obs_max),
                    )
                if not cfg.fixed_satellite_strategy:
                    sat_logits = _select_exec_values(
                        exec_sat_source,
                        policy_sat,
                        teacher_sat,
                        heur_sat,
                        (len(env.agents), cfg.sats_obs_max),
                    )
                actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)

            obs, rewards, terms, truncs, _ = env.step(actions)
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            reward_scalar = float(list(rewards.values())[0])
            parts = dict(getattr(env, "last_reward_parts", {}) or {})
            split = _queue_split_contribs(cfg, parts)
            close_risk_raw, _ = env._compute_close_risk_stats(require_enabled=False)
            if cfg.a_max > 0:
                accel_norm2 = float(np.mean(np.sum(np.asarray(env.last_exec_accel, dtype=np.float64) ** 2, axis=1))) / (
                    float(cfg.a_max) ** 2 + 1e-9
                )
            else:
                accel_norm2 = 0.0
            battery_event = 1.0 if (cfg.energy_enabled and np.any(env.uav_energy <= 0.0)) else 0.0

            _append(metric_signals, metric_contribs, "throughput", "service_norm", parts.get("service_norm", 0.0), parts.get("term_service", 0.0))
            _append(metric_signals, metric_contribs, "throughput", "throughput_access_norm", parts.get("throughput_access_norm", 0.0), parts.get("term_throughput_access", 0.0))
            _append(metric_signals, metric_contribs, "throughput", "throughput_backhaul_norm", parts.get("throughput_backhaul_norm", 0.0), parts.get("term_throughput_backhaul", 0.0))
            _append(metric_signals, metric_contribs, "drop", "gu_drop_norm", parts.get("gu_drop_norm", 0.0), parts.get("term_drop_gu", 0.0))
            _append(metric_signals, metric_contribs, "drop", "uav_drop_norm", parts.get("uav_drop_norm", 0.0), parts.get("term_drop_uav", 0.0))
            _append(metric_signals, metric_contribs, "drop", "sat_drop_norm", parts.get("sat_drop_norm", 0.0), parts.get("term_drop_sat", 0.0))
            _append(metric_signals, metric_contribs, "drop", "drop_event", parts.get("drop_event", 0.0), parts.get("term_drop_step", 0.0))
            _append(metric_signals, metric_contribs, "drop", "drop_total", parts.get("drop_norm", 0.0), parts.get("term_drop", 0.0))
            _append(metric_signals, metric_contribs, "queue", "queue_pen_gu", parts.get("queue_pen_gu", 0.0), split["queue_gu"])
            _append(metric_signals, metric_contribs, "queue", "queue_pen_uav", parts.get("queue_pen_uav", 0.0), split["queue_uav"])
            _append(metric_signals, metric_contribs, "queue", "queue_pen_sat", parts.get("queue_pen_sat", 0.0), split["queue_sat"])
            _append(metric_signals, metric_contribs, "queue", "queue_pen_total", parts.get("queue_pen", 0.0), parts.get("term_queue", 0.0))
            _append(metric_signals, metric_contribs, "queue_delta", "queue_delta_gu", parts.get("queue_delta_gu", 0.0), split["qdelta_gu"])
            _append(metric_signals, metric_contribs, "queue_delta", "queue_delta_uav", parts.get("queue_delta_uav", 0.0), split["qdelta_uav"])
            _append(metric_signals, metric_contribs, "queue_delta", "queue_delta_sat", parts.get("queue_delta_sat", 0.0), split["qdelta_sat"])
            _append(metric_signals, metric_contribs, "queue_delta", "queue_delta_total", parts.get("queue_delta", 0.0), parts.get("term_q_delta", 0.0))
            _append(metric_signals, metric_contribs, "shape", "centroid_reward", parts.get("centroid_reward", 0.0), parts.get("term_centroid", 0.0))
            _append(metric_signals, metric_contribs, "shape", "bw_align", parts.get("bw_align", 0.0), parts.get("term_bw_align", 0.0))
            _append(metric_signals, metric_contribs, "shape", "sat_score", parts.get("sat_score", 0.0), parts.get("term_sat_score", 0.0))
            _append(metric_signals, metric_contribs, "shape", "dist_reward", parts.get("dist_reward", 0.0), parts.get("term_dist", 0.0))
            _append(metric_signals, metric_contribs, "shape", "dist_delta", parts.get("dist_delta", 0.0), parts.get("term_dist_delta", 0.0))
            _append(metric_signals, metric_contribs, "energy", "energy_reward", parts.get("energy_reward", 0.0), parts.get("term_energy", 0.0))
            _append(metric_signals, metric_contribs, "safety", "accel_norm2", accel_norm2, parts.get("term_accel", 0.0))
            _append(metric_signals, metric_contribs, "safety", "close_risk", close_risk_raw, parts.get("term_close_risk", 0.0))
            _append(metric_signals, metric_contribs, "safety", "collision_event", parts.get("collision_event", 0.0), parts.get("collision_penalty", 0.0))
            _append(metric_signals, metric_contribs, "safety", "battery_event", battery_event, parts.get("battery_penalty", 0.0))
            _append(metric_signals, metric_contribs, "safety", "fail_penalty_total", parts.get("fail_penalty", 0.0), parts.get("fail_penalty", 0.0))
            _append(metric_signals, metric_contribs, "reward", "reward_raw", parts.get("reward_raw", 0.0), parts.get("reward_raw", 0.0))
            _append(metric_signals, metric_contribs, "reward", "reward_total", reward_scalar, reward_scalar)

            total_steps += 1
        progress.update(ep + 1)
    progress.close()

    rows = []
    for key in sorted(metric_signals.keys()):
        group, name = key.split(":", 1)
        signal_arr = _safe_array(metric_signals[key])
        contrib_arr = _safe_array(metric_contribs[key])
        signal_stats = _stat_summary(signal_arr)
        contrib_signed_stats = _stat_summary(contrib_arr)
        contrib_abs_stats = _abs_stat_summary(contrib_arr)
        row = {
            "group": group,
            "name": name,
            "count": int(signal_stats["count"]),
            "signal_mean": signal_stats["mean"],
            "signal_std": signal_stats["std"],
            "signal_abs_median": signal_stats["abs_median"],
            "signal_p10": signal_stats["p10"],
            "signal_p50": signal_stats["p50"],
            "signal_p90": signal_stats["p90"],
            "contrib_mean": contrib_signed_stats["mean"],
            "contrib_std": contrib_signed_stats["std"],
            "contrib_abs_mean": contrib_abs_stats["abs_mean"],
            "contrib_abs_std": contrib_abs_stats["abs_std"],
            "contrib_abs_median": contrib_abs_stats["abs_median"],
            "contrib_abs_p10": contrib_abs_stats["abs_p10"],
            "contrib_abs_p50": contrib_abs_stats["abs_p50"],
            "contrib_abs_p90": contrib_abs_stats["abs_p90"],
        }
        rows.append(row)

    rows.sort(key=lambda item: (-float(item["contrib_abs_mean"]), item["group"], item["name"]))
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved reward-scale summary to {args.out}")
    print(f"Total episodes: {args.episodes}, total steps: {total_steps}")
    print("Top terms by contrib_abs_mean:")
    for row in rows[:12]:
        print(
            f"  {row['group']}/{row['name']}: "
            f"signal_mean={row['signal_mean']:.6g}, "
            f"signal_p50={row['signal_p50']:.6g}, "
            f"contrib_abs_mean={row['contrib_abs_mean']:.6g}, "
            f"contrib_abs_p90={row['contrib_abs_p90']:.6g}"
        )
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


if __name__ == "__main__":
    main()
