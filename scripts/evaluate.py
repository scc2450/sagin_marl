from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    centroid_accel_policy,
    queue_aware_policy,
    random_accel_policy,
    zero_accel_policy,
    uniform_bw_policy,
    random_bw_policy,
    queue_aware_bw_policy,
    uniform_sat_policy,
    random_sat_policy,
    queue_aware_sat_policy,
)
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.progress import Progress


def _init_eval_tb_layout(writer: SummaryWriter, tag_prefix: str) -> None:
    def t(name: str) -> str:
        return f"{tag_prefix}/{name}"

    layout = {
        "Eval/Reward": {
            "RewardSum": ["Multiline", [t("reward_sum")]],
            "RewardRaw": ["Multiline", [t("reward_raw")]],
            "Steps": ["Multiline", [t("steps")]],
        },
        "Eval/Queues": {
            "QueueMean": ["Multiline", [t("gu_queue_mean"), t("uav_queue_mean"), t("sat_queue_mean")]],
            "QueueMax": ["Multiline", [t("gu_queue_max"), t("uav_queue_max"), t("sat_queue_max")]],
        },
        "Eval/Service": {
            "ServiceNorm": ["Multiline", [t("service_norm")]],
            "DropNorm": ["Multiline", [t("drop_norm")]],
        },
        "Eval/Distance": {
            "CentroidDist": ["Multiline", [t("centroid_dist_mean")]],
        },
        "Eval/Drops": {
            "Drops": ["Multiline", [t("gu_drop_sum"), t("uav_drop_sum")]],
        },
        "Eval/Association": {
            "AssocRatio": ["Multiline", [t("assoc_ratio_mean")]],
            "AssocDist": ["Multiline", [t("assoc_dist_mean")]],
        },
        "Eval/Satellite": {
            "SatFlow": ["Multiline", [t("sat_incoming_sum"), t("sat_processed_sum")]],
        },
        "Eval/Performance": {
            "Speed": ["Multiline", [t("steps_per_sec")]],
            "EpisodeTime": ["Multiline", [t("episode_time_sec")]],
        },
        "Eval/Energy": {
            "EnergyMean": ["Multiline", [t("energy_mean")]],
        },
    }

    other = None
    if tag_prefix == "eval/trained":
        other = "eval/baseline"
    elif tag_prefix == "eval/baseline":
        other = "eval/trained"

    if other is not None:
        layout["Eval/Compare"] = {
            "RewardSum": ["Multiline", [f"{tag_prefix}/reward_sum", f"{other}/reward_sum"]],
            "QueueMean": [
                "Multiline",
                [
                    f"{tag_prefix}/gu_queue_mean",
                    f"{tag_prefix}/uav_queue_mean",
                    f"{tag_prefix}/sat_queue_mean",
                    f"{other}/gu_queue_mean",
                    f"{other}/uav_queue_mean",
                    f"{other}/sat_queue_mean",
                ],
            ],
            "Drops": [
                "Multiline",
                [f"{tag_prefix}/gu_drop_sum", f"{tag_prefix}/uav_drop_sum", f"{other}/gu_drop_sum", f"{other}/uav_drop_sum"],
            ],
            "SatFlow": [
                "Multiline",
                [f"{tag_prefix}/sat_incoming_sum", f"{tag_prefix}/sat_processed_sum", f"{other}/sat_incoming_sum", f"{other}/sat_processed_sum"],
            ],
        }

    writer.add_custom_scalars(layout)


def _baseline_actions(
    baseline: str,
    obs_list,
    cfg,
    num_agents: int,
    rng: np.random.Generator | None = None,
):
    if baseline in ("zero_accel", "fixed"):
        return zero_accel_policy(num_agents), None, None
    if baseline == "random_accel":
        return random_accel_policy(num_agents, rng=rng), None, None
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


def _resolve_eval_paths(
    run_dir: str | None,
    checkpoint: str | None,
    out: str | None,
    tb_dir: str | None,
    baseline: str,
) -> tuple[str, str, str]:
    use_baseline = baseline != "none"
    if run_dir:
        checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        if out is None:
            filename = "eval_baseline.csv" if use_baseline else "eval_trained.csv"
            out = os.path.join(run_dir, filename)
    else:
        checkpoint = checkpoint or "runs/phase1/actor.pt"
        out = out or "runs/phase1/eval_trained.csv"

    out_dir = os.path.dirname(out) or "."
    tb_dir = tb_dir or os.path.join(out_dir, "eval_tb")
    return checkpoint, out, tb_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory that contains checkpoints and evaluation outputs.",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default=None,
        help="TensorBoard log dir for evaluation. Default: <out_dir>/eval_tb",
    )
    parser.add_argument(
        "--tb_tag",
        type=str,
        default=None,
        help="TensorBoard tag prefix. Default: eval/trained or eval/baseline",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=[
            "none", 
            "fixed", 
            "zero_accel", 
            "random_accel", 
            "centroid", 
            "queue_aware", 
            "uniform_bw", 
            "random_bw", 
            "queue_aware_bw", 
            "uniform_sat",
            "random_sat",
            "queue_aware_sat"
        ],
        help="Use a baseline policy instead of a trained model.",
    )
    parser.add_argument(
        "--hybrid_bw_sat",
        type=str,
        default="none",
        choices=["none", "queue_aware"],
        help="Use queue_aware for bw/sat while keeping accel from the trained actor.",
    )
    parser.add_argument(
        "--episode_seed_base",
        type=int,
        default=None,
        help="If set, episode e resets with seed=episode_seed_base+e for reproducible cross-policy comparisons.",
    )
    args = parser.parse_args()

    args.checkpoint, args.out, args.tb_dir = _resolve_eval_paths(
        args.run_dir, args.checkpoint, args.out, args.tb_dir, args.baseline
    )

    cfg = load_config(args.config)
    env = SaginParallelEnv(cfg)

    use_baseline = args.baseline != "none"
    use_hybrid = args.hybrid_bw_sat != "none"
    if use_hybrid and use_baseline:
        raise ValueError("Hybrid bw/sat is only valid when evaluating a trained model (baseline=none).")
    out_dir = os.path.dirname(args.out) or "."
    tb_dir = args.tb_dir or os.path.join(out_dir, "eval_tb")
    tb_tag = args.tb_tag or ("eval/baseline" if use_baseline else "eval/trained")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)
    _init_eval_tb_layout(tb_writer, tb_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = None
    if not use_baseline:
        obs, _ = env.reset()
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        state = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(state, strict=not use_hybrid)
        actor.eval()

    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "episode",
        "reward_sum",
        "reward_raw",
        "steps",
        "episode_time_sec",
        "steps_per_sec",
        "service_norm",
        "drop_norm",
        "queue_total_active",
        "arrival_sum",
        "outflow_sum",
        "outflow_arrival_ratio",
        "drop_sum",
        "drop_ratio",
        "drop_ratio_step_mean",
        "centroid_dist_mean",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
        "gu_queue_max",
        "uav_queue_max",
        "sat_queue_max",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_processed_sum",
        "sat_incoming_sum",
        "energy_mean",
        "assoc_ratio_mean",
        "assoc_dist_mean",
    ]
    progress = Progress(args.episodes, desc="Eval")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for ep in range(args.episodes):
            ep_seed = None
            if args.episode_seed_base is not None:
                ep_seed = int(args.episode_seed_base) + ep
            obs, _ = env.reset(seed=ep_seed)
            done = False
            reward_sum = 0.0
            steps = 0
            gu_queue_sum = 0.0
            uav_queue_sum = 0.0
            sat_queue_sum = 0.0
            gu_queue_max = 0.0
            uav_queue_max = 0.0
            sat_queue_max = 0.0
            gu_drop_sum = 0.0
            uav_drop_sum = 0.0
            sat_processed_sum = 0.0
            sat_incoming_sum = 0.0
            energy_mean_sum = 0.0
            assoc_ratio_sum = 0.0
            assoc_dist_sum = 0.0
            reward_raw_sum = 0.0
            service_norm_sum = 0.0
            drop_norm_sum = 0.0
            queue_total_active_sum = 0.0
            arrival_sum_ep = 0.0
            outflow_sum_ep = 0.0
            drop_sum_ep = 0.0
            drop_ratio_step_sum = 0.0
            centroid_dist_sum = 0.0
            ep_start = time.perf_counter()
            while not done:
                if use_baseline:
                    obs_list = list(obs.values())
                    accel_actions, bw_logits, sat_logits = _baseline_actions(
                        args.baseline, obs_list, cfg, len(env.agents), rng=env.rng
                    )
                    actions = assemble_actions(
                        cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits
                    )
                else:
                    obs_batch = batch_flatten_obs(list(obs.values()), cfg)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        policy_out = actor.act(obs_tensor, deterministic=True)
                    accel_actions = policy_out.accel.cpu().numpy()
                    bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
                    sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
                    if use_hybrid:
                        obs_list = list(obs.values())
                        _, bw_logits, sat_logits = queue_aware_policy(obs_list, cfg)
                    actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
                obs, rewards, terms, truncs, _ = env.step(actions)
                reward_sum += list(rewards.values())[0]
                done = list(terms.values())[0] or list(truncs.values())[0]
                steps += 1
                gu_queue_sum += float(np.mean(env.gu_queue))
                uav_queue_sum += float(np.mean(env.uav_queue))
                sat_queue_sum += float(np.mean(env.sat_queue))
                gu_queue_max = max(gu_queue_max, float(np.max(env.gu_queue)))
                uav_queue_max = max(uav_queue_max, float(np.max(env.uav_queue)))
                sat_queue_max = max(sat_queue_max, float(np.max(env.sat_queue)))
                gu_drop_sum += float(np.sum(env.gu_drop))
                uav_drop_sum += float(np.sum(env.uav_drop))
                if hasattr(env, "last_sat_processed"):
                    sat_processed_sum += float(np.sum(env.last_sat_processed))
                if hasattr(env, "last_sat_incoming"):
                    sat_incoming_sum += float(np.sum(env.last_sat_incoming))
                if cfg.energy_enabled:
                    energy_mean_sum += float(np.mean(env.uav_energy))
                parts = getattr(env, "last_reward_parts", None)
                if parts:
                    reward_raw_sum += float(parts.get("reward_raw", 0.0))
                    service_norm_sum += float(parts.get("service_norm", 0.0))
                    drop_norm_sum += float(parts.get("drop_norm", 0.0))
                    queue_total_active_sum += float(parts.get("queue_total_active", 0.0))
                    arrival_sum_ep += float(parts.get("arrival_sum", 0.0))
                    outflow_sum_ep += float(parts.get("outflow_sum", 0.0))
                    drop_sum_ep += float(parts.get("drop_sum", 0.0))
                    drop_ratio_step_sum += float(parts.get("drop_ratio", 0.0))
                    centroid_dist_sum += float(parts.get("centroid_dist_mean", 0.0))
                assoc_ratio = 0.0
                assoc_dist = 0.0
                if cfg.num_gu > 0 and hasattr(env, "last_association"):
                    assoc = env.last_association
                    mask = assoc >= 0
                    if mask.size > 0:
                        assoc_ratio = float(np.mean(mask))
                        if np.any(mask):
                            gu_pos = env.gu_pos[mask]
                            u_idx = assoc[mask].astype(np.int32)
                            uav_pos = env.uav_pos[u_idx]
                            d2d = np.linalg.norm(gu_pos - uav_pos, axis=1)
                            assoc_dist = float(np.mean(d2d)) if d2d.size else 0.0
                assoc_ratio_sum += assoc_ratio
                assoc_dist_sum += assoc_dist
            ep_time = time.perf_counter() - ep_start
            steps = max(1, steps)
            metrics = {
                "reward_sum": reward_sum,
                "reward_raw": reward_raw_sum / steps,
                "steps": steps,
                "episode_time_sec": ep_time,
                "steps_per_sec": steps / max(1e-9, ep_time),
                "service_norm": service_norm_sum / steps,
                "drop_norm": drop_norm_sum / steps,
                "queue_total_active": queue_total_active_sum / steps,
                "arrival_sum": arrival_sum_ep,
                "outflow_sum": outflow_sum_ep,
                "outflow_arrival_ratio": outflow_sum_ep / max(arrival_sum_ep, 1e-9),
                "drop_sum": drop_sum_ep,
                "drop_ratio": drop_sum_ep / max(arrival_sum_ep, 1e-9),
                "drop_ratio_step_mean": drop_ratio_step_sum / steps,
                "centroid_dist_mean": centroid_dist_sum / steps,
                "gu_queue_mean": gu_queue_sum / steps,
                "uav_queue_mean": uav_queue_sum / steps,
                "sat_queue_mean": sat_queue_sum / steps,
                "gu_queue_max": gu_queue_max,
                "uav_queue_max": uav_queue_max,
                "sat_queue_max": sat_queue_max,
                "gu_drop_sum": gu_drop_sum,
                "uav_drop_sum": uav_drop_sum,
                "sat_processed_sum": sat_processed_sum,
                "sat_incoming_sum": sat_incoming_sum,
                "energy_mean": (energy_mean_sum / steps) if cfg.energy_enabled else 0.0,
                "assoc_ratio_mean": assoc_ratio_sum / steps,
                "assoc_dist_mean": assoc_dist_sum / steps,
            }
            writer.writerow(
                [
                    ep,
                    metrics["reward_sum"],
                    metrics["reward_raw"],
                    metrics["steps"],
                    metrics["episode_time_sec"],
                    metrics["steps_per_sec"],
                    metrics["service_norm"],
                    metrics["drop_norm"],
                    metrics["queue_total_active"],
                    metrics["arrival_sum"],
                    metrics["outflow_sum"],
                    metrics["outflow_arrival_ratio"],
                    metrics["drop_sum"],
                    metrics["drop_ratio"],
                    metrics["drop_ratio_step_mean"],
                    metrics["centroid_dist_mean"],
                    metrics["gu_queue_mean"],
                    metrics["uav_queue_mean"],
                    metrics["sat_queue_mean"],
                    metrics["gu_queue_max"],
                    metrics["uav_queue_max"],
                    metrics["sat_queue_max"],
                    metrics["gu_drop_sum"],
                    metrics["uav_drop_sum"],
                    metrics["sat_processed_sum"],
                    metrics["sat_incoming_sum"],
                    metrics["energy_mean"],
                    metrics["assoc_ratio_mean"],
                    metrics["assoc_dist_mean"],
                ]
            )
            for key, val in metrics.items():
                tb_writer.add_scalar(f"{tb_tag}/{key}", val, ep)
            progress.update(ep + 1)
    progress.close()
    tb_writer.close()
    print(f"Saved evaluation metrics to {args.out}")


if __name__ == "__main__":
    main()
