from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import queue_aware_sat_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


@dataclass(frozen=True)
class PolicySpec:
    name: str
    label: str
    config_path: str
    checkpoint_path: str
    hybrid_bw_sat: str = "none"


def _build_actor(spec: PolicySpec, device: torch.device) -> ActorNet:
    cfg = load_config(spec.config_path)
    env = make_structured_env(cfg, mode="script")
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(device)
    info = load_checkpoint_forgiving(actor, spec.checkpoint_path, map_location=device)
    adapted = len(info.get("adapted_keys", []))
    missing = len(info.get("missing_keys", []))
    print(f"[load] {spec.name}: adapted={adapted}, missing={missing}")
    actor.eval()
    return actor


def _episode_metrics(env: SaginParallelEnv, reward_sum: float, steps: int, sat_processed_sum: float, collision_any: float,
                     processed_ratio_eval_sum: float, pre_backlog_steps_eval_sum: float, d_sys_queue_sum: float,
                     throughput_access_norm_sum: float, throughput_backhaul_norm_sum: float,
                     sat_processed_norm_sum: float, sat_overlap_eval_sum: float) -> Dict[str, float]:
    steps = max(steps, 1)
    return {
        "reward_sum": reward_sum,
        "processed_ratio_eval": processed_ratio_eval_sum / steps,
        "pre_backlog_steps_eval": pre_backlog_steps_eval_sum / steps,
        "D_sys_report": d_sys_queue_sum / max(sat_processed_sum, 1e-9),
        "collision": collision_any,
        "throughput_access_norm": throughput_access_norm_sum / steps,
        "throughput_backhaul_norm": throughput_backhaul_norm_sum / steps,
        "sat_processed_norm": sat_processed_norm_sum / steps,
        "sat_overlap_eval": sat_overlap_eval_sum / steps,
        "steps": float(steps),
        "terminated_early": 1.0 if steps < int(env.cfg.T_steps) else 0.0,
    }


def _mean_summary(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [k for k in rows[0].keys() if k not in {"episode"}]
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}


def _evaluate_policy(
    spec: PolicySpec,
    actor: ActorNet,
    param_name: str,
    param_value: float,
    scale: float,
    episodes: int,
    seed_base: int,
    device: torch.device,
) -> tuple[List[Dict[str, float]], Dict[str, float], float]:
    cfg = load_config(spec.config_path)
    base_value = float(getattr(cfg, param_name))
    setattr(cfg, param_name, float(param_value))
    env = make_structured_env(cfg, mode="script")
    episode_rows: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        reward_sum = 0.0
        steps = 0
        sat_processed_sum = 0.0
        collision_any = 0.0
        processed_ratio_eval_sum = 0.0
        pre_backlog_steps_eval_sum = 0.0
        throughput_access_norm_sum = 0.0
        throughput_backhaul_norm_sum = 0.0
        sat_processed_norm_sum = 0.0
        sat_overlap_eval_sum = 0.0

        while not done:
            obs_list = list(obs.values())
            obs_batch = batch_flatten_obs(obs_list, cfg)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_out = actor.act(obs_tensor, deterministic=True)

            accel = policy_out.accel.cpu().numpy()
            bw = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
            sat = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None

            if spec.hybrid_bw_sat == "queue_aware_sat":
                sat = queue_aware_sat_policy(obs_list, cfg)

            actions = assemble_actions(cfg, env.agents, accel, bw_logits=bw, sat_logits=sat)
            obs, rewards, terms, truncs, _ = env.step(actions)

            reward_sum += float(list(rewards.values())[0])
            done = bool(list(terms.values())[0] or list(truncs.values())[0])
            steps += 1

            if hasattr(env, "last_sat_processed"):
                sat_processed_sum += float(np.sum(env.last_sat_processed))

            parts = getattr(env, "last_reward_parts", {})
            processed_ratio_eval_sum += float(parts.get("processed_ratio_eval", 0.0))
            pre_backlog_steps_eval_sum += float(parts.get("pre_backlog_steps_eval", 0.0))
            throughput_access_norm_sum += float(parts.get("throughput_access_norm", 0.0))
            throughput_backhaul_norm_sum += float(parts.get("throughput_backhaul_norm", 0.0))
            sat_processed_norm_sum += float(parts.get("sat_processed_norm", 0.0))
            sat_overlap_eval_sum += float(parts.get("sat_overlap_eval", 0.0))
            collision_any = max(collision_any, float(parts.get("collision_event", 0.0)))

        d_sys_queue_sum = float(np.sum(env.gu_queue) + np.sum(env.uav_queue) + np.sum(env.sat_queue))
        metrics = _episode_metrics(
            env=env,
            reward_sum=reward_sum,
            steps=steps,
            sat_processed_sum=sat_processed_sum,
            collision_any=collision_any,
            processed_ratio_eval_sum=processed_ratio_eval_sum,
            pre_backlog_steps_eval_sum=pre_backlog_steps_eval_sum,
            d_sys_queue_sum=d_sys_queue_sum,
            throughput_access_norm_sum=throughput_access_norm_sum,
            throughput_backhaul_norm_sum=throughput_backhaul_norm_sum,
            sat_processed_norm_sum=sat_processed_norm_sum,
            sat_overlap_eval_sum=sat_overlap_eval_sum,
        )
        metrics["episode"] = float(ep)
        episode_rows.append(metrics)

    summary = _mean_summary(episode_rows)
    summary.update(
        {
            "param_name": param_name,
            "scale": scale,
            "value": float(param_value),
            "base_value": base_value,
            "policy": spec.name,
            "policy_label": spec.label,
            "hybrid_bw_sat": spec.hybrid_bw_sat,
        }
    )
    return episode_rows, summary, base_value


def _write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rank_rows(summary_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[tuple[str, float], List[Dict[str, float]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["param_name"]), float(row["scale"])), []).append(row)

    ranked_rows: List[Dict[str, float]] = []
    for (_, _), rows in grouped.items():
        reward_sorted = sorted(rows, key=lambda r: (-float(r["reward_sum"]), float(r["pre_backlog_steps_eval"])))
        backlog_sorted = sorted(rows, key=lambda r: (float(r["pre_backlog_steps_eval"]), -float(r["reward_sum"])))
        reward_rank = {id(row): idx + 1 for idx, row in enumerate(reward_sorted)}
        backlog_rank = {id(row): idx + 1 for idx, row in enumerate(backlog_sorted)}
        reward_order = " > ".join(row["policy_label"] for row in reward_sorted)
        backlog_order = " < ".join(row["policy_label"] for row in backlog_sorted)
        for row in rows:
            out = dict(row)
            out["rank_reward_sum"] = reward_rank[id(row)]
            out["rank_pre_backlog_steps_eval"] = backlog_rank[id(row)]
            out["reward_order"] = reward_order
            out["backlog_order"] = backlog_order
            ranked_rows.append(out)
    return sorted(ranked_rows, key=lambda r: (str(r["param_name"]), float(r["scale"]), int(r["rank_reward_sum"])))


def _write_markdown(path: Path, ranked_rows: List[Dict[str, float]], params: List[str], scales: List[float]) -> None:
    grouped: Dict[tuple[str, float], List[Dict[str, float]]] = {}
    for row in ranked_rows:
        grouped.setdefault((str(row["param_name"]), float(row["scale"])), []).append(row)

    lines: List[str] = ["# Eval-Only Param Sweep", ""]
    for param in params:
        lines.append(f"## {param}")
        lines.append("")
        lines.append("| scale | reward ranking | backlog ranking |")
        lines.append("|---|---|---|")
        for scale in scales:
            rows = grouped[(param, scale)]
            reward_order = rows[0]["reward_order"]
            backlog_order = rows[0]["backlog_order"]
            lines.append(f"| {scale:.1f}x | {reward_order} | {backlog_order} |")
        lines.append("")
        lines.append("| scale | policy | reward_sum | processed_ratio | pre_backlog_steps | D_sys_report | collision |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for scale in scales:
            rows = sorted(grouped[(param, scale)], key=lambda r: int(r["rank_reward_sum"]))
            for row in rows:
                lines.append(
                    f"| {scale:.1f}x | {row['policy_label']} | {float(row['reward_sum']):.3f} | "
                    f"{float(row['processed_ratio_eval']):.4f} | {float(row['pre_backlog_steps_eval']):.4f} | "
                    f"{float(row['D_sys_report']):.5f} | {float(row['collision']):.2f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_seed_base", type=int, default=82000)
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument(
        "--params",
        type=str,
        default="task_arrival_rate,b_acc,b_backhaul_per_sat,sat_cpu_freq",
        help="Comma-separated env params to sweep.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="stage1_best,stage2_best,stage2_plus_queue_aware_sat,stage3_best",
        help="Comma-separated policy ids to evaluate.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/phase1_actions/eval_param_sweep_20260325",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(int(args.torch_threads), 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_policies = [
        PolicySpec(
            name="stage1_best",
            label="Stage 1 best",
            config_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage1_accel/config_source.yaml",
            checkpoint_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage1_accel/actor_best.pt",
        ),
        PolicySpec(
            name="stage2_best",
            label="Stage 2 best",
            config_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage2_bw/config_source.yaml",
            checkpoint_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage2_bw/actor_best.pt",
        ),
        PolicySpec(
            name="stage2_plus_queue_aware_sat",
            label="Stage 2 + queue_aware_sat",
            config_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage3_sat/config_source.yaml",
            checkpoint_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage2_bw/actor_best.pt",
            hybrid_bw_sat="queue_aware_sat",
        ),
        PolicySpec(
            name="stage3_best",
            label="Stage 3 best",
            config_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage3_sat/config_source.yaml",
            checkpoint_path="runs/phase1_actions/curriculum_formal_u1500_subproc12_t2/stage3_sat/actor_best.pt",
        ),
    ]
    requested_policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    requested_params = [item.strip() for item in str(args.params).split(",") if item.strip()]
    policy_map = {spec.name: spec for spec in all_policies}
    unknown_policies = [name for name in requested_policies if name not in policy_map]
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}")
    policies = [policy_map[name] for name in requested_policies]
    valid_params = {"task_arrival_rate", "b_acc", "b_backhaul_per_sat", "b_sat_total", "sat_cpu_freq"}
    unknown_params = [name for name in requested_params if name not in valid_params]
    if unknown_params:
        raise ValueError(f"Unknown params: {unknown_params}")
    params = requested_params
    scales = [0.5, 1.0, 2.0]

    actors = {spec.name: _build_actor(spec, device) for spec in policies}
    detail_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    total_jobs = len(params) * len(scales) * len(policies)
    job_idx = 0
    sweep_start = time.perf_counter()

    for param_name in params:
        ref_cfg = load_config(policies[0].config_path)
        base_value = float(getattr(ref_cfg, param_name))
        for scale in scales:
            param_value = base_value * scale
            for spec in policies:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] {param_name}={param_value} ({scale:.1f}x) | {spec.name}")
                ep_rows, summary, _ = _evaluate_policy(
                    spec=spec,
                    actor=actors[spec.name],
                    param_name=param_name,
                    param_value=param_value,
                    scale=scale,
                    episodes=int(args.episodes),
                    seed_base=int(args.episode_seed_base),
                    device=device,
                )
                for row in ep_rows:
                    row.update(
                        {
                            "param_name": param_name,
                            "scale": scale,
                            "value": float(param_value),
                            "base_value": base_value,
                            "policy": spec.name,
                            "policy_label": spec.label,
                            "hybrid_bw_sat": spec.hybrid_bw_sat,
                        }
                    )
                detail_rows.extend(ep_rows)
                summary_rows.append(summary)

    ranked_rows = _rank_rows(summary_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_fieldnames = [
        "param_name",
        "scale",
        "value",
        "base_value",
        "policy",
        "policy_label",
        "hybrid_bw_sat",
        "episode",
        "reward_sum",
        "processed_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "collision",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "sat_processed_norm",
        "sat_overlap_eval",
        "steps",
        "terminated_early",
    ]
    summary_fieldnames = [
        "param_name",
        "scale",
        "value",
        "base_value",
        "policy",
        "policy_label",
        "hybrid_bw_sat",
        "reward_sum",
        "processed_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "collision",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "sat_processed_norm",
        "sat_overlap_eval",
        "steps",
        "terminated_early",
    ]
    ranked_fieldnames = summary_fieldnames + ["rank_reward_sum", "rank_pre_backlog_steps_eval", "reward_order", "backlog_order"]

    _write_csv(out_dir / "detail.csv", detail_rows, detail_fieldnames)
    _write_csv(out_dir / "summary.csv", summary_rows, summary_fieldnames)
    _write_csv(out_dir / "ranked_summary.csv", ranked_rows, ranked_fieldnames)
    _write_markdown(out_dir / "summary.md", ranked_rows, params, scales)

    elapsed = time.perf_counter() - sweep_start
    print(f"Done in {elapsed:.1f}s. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
