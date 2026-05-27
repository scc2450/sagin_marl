from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import cluster_center_queue_aware_policy, queue_aware_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


@dataclass(frozen=True)
class CellSpec:
    name: str
    bw_label: str
    partner_label: str
    config_path: str
    primary_checkpoint: str
    exec_accel_source: str
    exec_bw_source: str
    exec_sat_source: str
    teacher_actor_path: str | None = None


def _normalize_exec_source(raw: str | None) -> str:
    src = str("policy" if raw is None else raw).strip().lower()
    if src == "heuristic_residual":
        src = "policy"
    allowed = {"policy", "teacher", "heuristic", "cluster_center_queue_aware", "zero"}
    if src not in allowed:
        raise ValueError(f"Invalid exec source '{raw}'. Allowed: {sorted(allowed)}")
    return src


def _source_needs_heuristic(source: str) -> bool:
    return source in {"heuristic", "cluster_center_queue_aware"}


def _resolve_exec_heuristic_triplet(obs_list, cfg, env, accel_source: str, bw_source: str, sat_source: str):
    queue_bundle = (None, None, None)
    cluster_bundle = (None, None, None)
    sources = {accel_source, bw_source, sat_source}

    if "heuristic" in sources:
        queue_bundle = queue_aware_policy(obs_list, cfg)

    if "cluster_center_queue_aware" in sources:
        centers = getattr(env, "gu_cluster_centers", None)
        counts = getattr(env, "gu_cluster_counts", None)
        cluster_bundle = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)

    def pick(source: str):
        if source == "cluster_center_queue_aware":
            return cluster_bundle
        if source == "heuristic":
            return queue_bundle
        return (None, None, None)

    accel_triplet = pick(accel_source)
    bw_triplet = pick(bw_source)
    sat_triplet = pick(sat_source)
    return accel_triplet[0], bw_triplet[1], sat_triplet[2]


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
    if source in {"heuristic", "cluster_center_queue_aware"} and heuristic_values is not None:
        return np.asarray(heuristic_values, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _int_histogram(values: list[float]) -> dict[str, int]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    uniq, counts = np.unique(arr, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def _summarize_row_metrics(rows: list[dict[str, Any]], exclude_keys: set[str] | None = None) -> dict[str, float | None]:
    if not rows:
        return {}
    exclude = set() if exclude_keys is None else set(exclude_keys)
    out: dict[str, float | None] = {}
    for key in rows[0].keys():
        if key in exclude:
            continue
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                values.append(float(value))
        if not values:
            continue
        for stat_name, stat_value in _summarize(values).items():
            out[f"{key}_{stat_name}"] = stat_value
    return out


def _episode_metric_row(
    cell: CellSpec,
    episode: int,
    episode_seed: int | None,
    reward_sum: float,
    steps: int,
    processed_ratio_eval_sum: float,
    drop_ratio_eval_sum: float,
    pre_backlog_steps_eval_sum: float,
    bw_valid_counts: list[float],
    elapsed_sec: float,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_denom = max(int(steps), 1)
    row = {
        "cell": cell.name,
        "bw_label": cell.bw_label,
        "partner_label": cell.partner_label,
        "episode": int(episode),
        "episode_seed": None if episode_seed is None else int(episode_seed),
        "reward_sum": float(reward_sum),
        "steps": int(steps),
        "processed_ratio_eval": float(processed_ratio_eval_sum / step_denom),
        "drop_ratio_eval": float(drop_ratio_eval_sum / step_denom),
        "pre_backlog_steps_eval": float(pre_backlog_steps_eval_sum / step_denom),
        "elapsed_sec": float(elapsed_sec),
    }
    bw_summary = _summarize(bw_valid_counts)
    row["bw_valid_count_mean"] = bw_summary["mean"]
    row["bw_valid_count_p50"] = bw_summary["p50"]
    row["bw_valid_count_p90"] = bw_summary["p90"]
    row["bw_valid_count_max"] = bw_summary["max"]
    if extra_metrics:
        row.update(extra_metrics)
    return row


def _build_checkpoint_swap_cells(args) -> list[CellSpec]:
    bwonly_config = str(Path(args.bwonly_run) / "config_source.yaml")
    joint_config = str(Path(args.joint_run) / "config_source.yaml")
    return [
        CellSpec(
            name="bwonly_bw__cluster_partner",
            bw_label="bwonly_u0050",
            partner_label="cluster_center_queue_aware",
            config_path=bwonly_config,
            primary_checkpoint=args.bwonly_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="bwonly_bw__joint_partner",
            bw_label="bwonly_u0050",
            partner_label="joint_u0050_teacher_accel_sat",
            config_path=bwonly_config,
            primary_checkpoint=args.bwonly_checkpoint,
            exec_accel_source="teacher",
            exec_bw_source="policy",
            exec_sat_source="teacher",
            teacher_actor_path=args.joint_checkpoint,
        ),
        CellSpec(
            name="joint_bw__cluster_partner",
            bw_label="joint_u0050",
            partner_label="cluster_center_queue_aware",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="joint_bw__joint_partner",
            bw_label="joint_u0050",
            partner_label="joint_u0050_native",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="policy",
            exec_bw_source="policy",
            exec_sat_source="policy",
        ),
    ]


def _build_partner_component_split_cells(args) -> list[CellSpec]:
    if args.fixed_bw_source == "joint":
        config_path = str(Path(args.joint_run) / "config_source.yaml")
        primary_checkpoint = args.joint_checkpoint
        bw_label = "joint_u0050"
        joint_exec_source = "policy"
        teacher_actor_path = None
    else:
        config_path = str(Path(args.bwonly_run) / "config_source.yaml")
        primary_checkpoint = args.bwonly_checkpoint
        bw_label = "bwonly_u0050"
        joint_exec_source = "teacher"
        teacher_actor_path = args.joint_checkpoint

    return [
        CellSpec(
            name=f"{args.fixed_bw_source}_bw__cluster_accel_cluster_sat",
            bw_label=bw_label,
            partner_label="cluster_accel__cluster_sat",
            config_path=config_path,
            primary_checkpoint=primary_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name=f"{args.fixed_bw_source}_bw__joint_accel_cluster_sat",
            bw_label=bw_label,
            partner_label="joint_accel__cluster_sat",
            config_path=config_path,
            primary_checkpoint=primary_checkpoint,
            exec_accel_source=joint_exec_source,
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
            teacher_actor_path=teacher_actor_path,
        ),
        CellSpec(
            name=f"{args.fixed_bw_source}_bw__cluster_accel_joint_sat",
            bw_label=bw_label,
            partner_label="cluster_accel__joint_sat",
            config_path=config_path,
            primary_checkpoint=primary_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source=joint_exec_source,
            teacher_actor_path=teacher_actor_path,
        ),
        CellSpec(
            name=f"{args.fixed_bw_source}_bw__joint_accel_joint_sat",
            bw_label=bw_label,
            partner_label="joint_accel__joint_sat",
            config_path=config_path,
            primary_checkpoint=primary_checkpoint,
            exec_accel_source=joint_exec_source,
            exec_bw_source="policy",
            exec_sat_source=joint_exec_source,
            teacher_actor_path=teacher_actor_path,
        ),
    ]


def _build_bw_execution_swap_cells(args) -> list[CellSpec]:
    joint_config = str(Path(args.joint_run) / "config_source.yaml")
    bwonly_config = str(Path(args.bwonly_run) / "config_source.yaml")
    return [
        CellSpec(
            name="cluster_accel_cluster_sat__heuristic_bw",
            bw_label="heuristic_bw",
            partner_label="cluster_accel__cluster_sat",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="cluster_center_queue_aware",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="cluster_accel_cluster_sat__bwonly_bw",
            bw_label="bwonly_u0050",
            partner_label="cluster_accel__cluster_sat",
            config_path=bwonly_config,
            primary_checkpoint=args.bwonly_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="cluster_accel_cluster_sat__joint_bw",
            bw_label="joint_u0050",
            partner_label="cluster_accel__cluster_sat",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="cluster_center_queue_aware",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="joint_accel_cluster_sat__heuristic_bw",
            bw_label="heuristic_bw",
            partner_label="joint_accel__cluster_sat",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="policy",
            exec_bw_source="cluster_center_queue_aware",
            exec_sat_source="cluster_center_queue_aware",
        ),
        CellSpec(
            name="joint_accel_cluster_sat__bwonly_bw",
            bw_label="bwonly_u0050",
            partner_label="joint_accel__cluster_sat",
            config_path=bwonly_config,
            primary_checkpoint=args.bwonly_checkpoint,
            exec_accel_source="teacher",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
            teacher_actor_path=args.joint_checkpoint,
        ),
        CellSpec(
            name="joint_accel_cluster_sat__joint_bw",
            bw_label="joint_u0050",
            partner_label="joint_accel__cluster_sat",
            config_path=joint_config,
            primary_checkpoint=args.joint_checkpoint,
            exec_accel_source="policy",
            exec_bw_source="policy",
            exec_sat_source="cluster_center_queue_aware",
        ),
    ]


def _build_cells(args) -> list[CellSpec]:
    if args.matrix == "checkpoint_swap":
        return _build_checkpoint_swap_cells(args)
    if args.matrix == "partner_component_split":
        return _build_partner_component_split_cells(args)
    if args.matrix == "bw_execution_swap":
        return _build_bw_execution_swap_cells(args)
    raise ValueError(f"Unknown matrix mode: {args.matrix}")


def evaluate_cell(
    cell: CellSpec,
    episodes: int,
    episode_seed_base: int | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_config(cell.config_path)
    cfg.exec_accel_source = _normalize_exec_source(cell.exec_accel_source)
    cfg.exec_bw_source = _normalize_exec_source(cell.exec_bw_source)
    cfg.exec_sat_source = _normalize_exec_source(cell.exec_sat_source)
    cfg.exec_teacher_actor_path = cell.teacher_actor_path
    cfg.exec_teacher_deterministic = True

    env = make_structured_env(cfg, mode="script")
    try:
        obs, _ = env.reset(seed=episode_seed_base)
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]

        actor = ActorNet(obs_dim, cfg).to(device)
        actor_info = load_checkpoint_forgiving(actor, cell.primary_checkpoint, map_location=device)
        actor.eval()

        teacher_actor = None
        teacher_info: dict[str, Any] | None = None
        need_teacher_exec = "teacher" in {
            cfg.exec_accel_source,
            cfg.exec_bw_source,
            cfg.exec_sat_source,
        }
        need_heuristic_exec = any(
            _source_needs_heuristic(src) for src in (cfg.exec_accel_source, cfg.exec_bw_source, cfg.exec_sat_source)
        )
        if need_teacher_exec:
            if not cell.teacher_actor_path:
                raise ValueError(f"Cell {cell.name} requires teacher execution but teacher_actor_path is missing.")
            teacher_actor = ActorNet(obs_dim, cfg).to(device)
            teacher_info = load_checkpoint_forgiving(teacher_actor, cell.teacher_actor_path, map_location=device)
            teacher_actor.eval()

        rows: list[dict[str, Any]] = []
        all_bw_valid_counts: list[float] = []
        all_bw_valid_step_max_values: list[float] = []
        all_bw_valid_step_std_values: list[float] = []
        all_assoc_count_values: list[float] = []
        all_assoc_count_step_max_values: list[float] = []
        all_assoc_count_step_std_values: list[float] = []

        for episode in range(episodes):
            episode_seed = None if episode_seed_base is None else int(episode_seed_base) + int(episode)
            obs, _ = env.reset(seed=episode_seed)
            done = False
            reward_sum = 0.0
            steps = 0
            processed_ratio_eval_sum = 0.0
            drop_ratio_eval_sum = 0.0
            pre_backlog_steps_eval_sum = 0.0
            throughput_access_norm_sum = 0.0
            throughput_backhaul_norm_sum = 0.0
            sat_processed_norm_sum = 0.0
            assoc_ratio_sum = 0.0
            assoc_dist_sum = 0.0
            assoc_unfair_max_gu_count_sum = 0.0
            assoc_unfair_step_sum = 0.0
            assoc_centroid_dist_norm_sum = 0.0
            centroid_dist_sum = 0.0
            intervention_norm_sum = 0.0
            intervention_rate_sum = 0.0
            connected_sat_count_sum = 0.0
            connected_sat_dist_mean_sum = 0.0
            arrival_sum_ep = 0.0
            outflow_sum_ep = 0.0
            active_drop_sum_ep = 0.0
            drop_sum_ep = 0.0
            sat_incoming_sum = 0.0
            sat_processed_sum = 0.0
            gu_queue_sum = 0.0
            uav_queue_sum = 0.0
            sat_queue_sum = 0.0
            gu_drop_sum = 0.0
            uav_drop_sum = 0.0
            sat_drop_sum = 0.0
            min_inter_uav_dist = float("inf")
            near_collision_steps = 0.0
            layer_queue_sum_steps = {"gu": [], "uav": [], "sat": []}
            layer_queue_start_sum = {
                "gu": float(np.sum(env.gu_queue)),
                "uav": float(np.sum(env.uav_queue)),
                "sat": float(np.sum(env.sat_queue)),
            }
            episode_bw_valid_counts: list[float] = []
            episode_bw_valid_step_max_values: list[float] = []
            episode_bw_valid_step_std_values: list[float] = []
            episode_assoc_count_values: list[float] = []
            episode_assoc_count_step_max_values: list[float] = []
            episode_assoc_count_step_std_values: list[float] = []
            t0 = time.perf_counter()

            while not done:
                obs_list = list(obs.values())
                obs_batch = batch_flatten_obs(obs_list, cfg)
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    policy_out = actor.act(obs_tensor, deterministic=True)
                    teacher_accel = None
                    teacher_bw = None
                    teacher_sat = None
                    if teacher_actor is not None:
                        teacher_out = teacher_actor.act(obs_tensor, deterministic=True)
                        teacher_accel = teacher_out.accel.cpu().numpy() if teacher_out.accel is not None else None
                        teacher_bw = teacher_out.bw_logits.cpu().numpy() if teacher_out.bw_logits is not None else None
                        teacher_sat = teacher_out.sat_logits.cpu().numpy() if teacher_out.sat_logits is not None else None

                policy_accel = policy_out.accel.cpu().numpy() if policy_out.accel is not None else None
                policy_bw = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
                policy_sat = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None

                heur_accel = None
                heur_bw = None
                heur_sat = None
                if need_heuristic_exec:
                    heur_accel, heur_bw, heur_sat = _resolve_exec_heuristic_triplet(
                        obs_list,
                        cfg,
                        env,
                        cfg.exec_accel_source,
                        cfg.exec_bw_source,
                        cfg.exec_sat_source,
                    )

                bw_valid_tensor = None if policy_out.dist_out is None else policy_out.dist_out.get("bw_valid_count")
                if bw_valid_tensor is not None:
                    bw_valid_np = bw_valid_tensor.detach().cpu().numpy().astype(np.float64)
                    episode_bw_valid_counts.extend(bw_valid_np.tolist())
                    all_bw_valid_counts.extend(bw_valid_np.tolist())
                    episode_bw_valid_step_max_values.append(float(np.max(bw_valid_np)) if bw_valid_np.size else 0.0)
                    episode_bw_valid_step_std_values.append(float(np.std(bw_valid_np)) if bw_valid_np.size else 0.0)
                    all_bw_valid_step_max_values.append(float(np.max(bw_valid_np)) if bw_valid_np.size else 0.0)
                    all_bw_valid_step_std_values.append(float(np.std(bw_valid_np)) if bw_valid_np.size else 0.0)

                accel_actions = _select_exec_values(
                    cfg.exec_accel_source,
                    policy_accel,
                    teacher_accel,
                    heur_accel,
                    (len(env.agents), 2),
                )
                bw_logits = None
                sat_logits = None
                if cfg.enable_bw_action:
                    bw_logits = _select_exec_values(
                        cfg.exec_bw_source,
                        policy_bw,
                        teacher_bw,
                        heur_bw,
                        (len(env.agents), cfg.users_obs_max),
                    )
                if not cfg.fixed_satellite_strategy:
                    sat_logits = _select_exec_values(
                        cfg.exec_sat_source,
                        policy_sat,
                        teacher_sat,
                        heur_sat,
                        (len(env.agents), cfg.sats_obs_max),
                    )
                actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
                obs, rewards, terms, truncs, _ = env.step(actions)
                reward_sum += float(list(rewards.values())[0])
                done = bool(list(terms.values())[0] or list(truncs.values())[0])
                steps += 1
                gu_queue_sum += float(np.mean(env.gu_queue))
                uav_queue_sum += float(np.mean(env.uav_queue))
                sat_queue_sum += float(np.mean(env.sat_queue))
                layer_queue_sum_steps["gu"].append(float(np.sum(env.gu_queue)))
                layer_queue_sum_steps["uav"].append(float(np.sum(env.uav_queue)))
                layer_queue_sum_steps["sat"].append(float(np.sum(env.sat_queue)))
                gu_drop_sum += float(np.sum(env.gu_drop))
                uav_drop_sum += float(np.sum(env.uav_drop))
                if hasattr(env, "sat_drop"):
                    sat_drop_sum += float(np.sum(env.sat_drop))
                if hasattr(env, "last_sat_processed"):
                    sat_processed_sum += float(np.sum(env.last_sat_processed))
                if hasattr(env, "last_sat_incoming"):
                    sat_incoming_sum += float(np.sum(env.last_sat_incoming))

                if cfg.num_uav > 1 and hasattr(env, "uav_pos"):
                    diff = env.uav_pos[:, None, :] - env.uav_pos[None, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dists, np.inf)
                    cur_min_dist = float(np.min(dists))
                    min_inter_uav_dist = min(min_inter_uav_dist, cur_min_dist)
                    if cur_min_dist < float(cfg.avoidance_alert_factor) * float(cfg.d_safe):
                        near_collision_steps += 1.0

                connected_sat_count_sum += float(getattr(env, "last_connected_sat_count", 0.0))
                connected_sat_dist_mean_sum += float(getattr(env, "last_connected_sat_dist_mean", 0.0))

                assoc_ratio = 0.0
                assoc_dist = 0.0
                assoc_counts = np.zeros((cfg.num_uav,), dtype=np.float32)
                if cfg.num_gu > 0 and hasattr(env, "last_association"):
                    assoc = np.asarray(env.last_association, dtype=np.int32)
                    mask = assoc >= 0
                    if mask.size > 0:
                        assoc_ratio = float(np.mean(mask))
                        if np.any(mask):
                            assoc_counts = np.bincount(assoc[mask], minlength=cfg.num_uav).astype(np.float32)
                            gu_pos = env.gu_pos[mask]
                            u_idx = assoc[mask].astype(np.int32)
                            uav_pos = env.uav_pos[u_idx]
                            d2d = np.linalg.norm(gu_pos - uav_pos, axis=1)
                            assoc_dist = float(np.mean(d2d)) if d2d.size else 0.0
                assoc_ratio_sum += assoc_ratio
                assoc_dist_sum += assoc_dist
                episode_assoc_count_values.extend(assoc_counts.tolist())
                all_assoc_count_values.extend(assoc_counts.tolist())
                assoc_count_step_max = float(np.max(assoc_counts)) if assoc_counts.size else 0.0
                assoc_count_step_std = float(np.std(assoc_counts)) if assoc_counts.size else 0.0
                episode_assoc_count_step_max_values.append(assoc_count_step_max)
                episode_assoc_count_step_std_values.append(assoc_count_step_std)
                all_assoc_count_step_max_values.append(assoc_count_step_max)
                all_assoc_count_step_std_values.append(assoc_count_step_std)

                parts = getattr(env, "last_reward_parts", None)
                if parts:
                    processed_ratio_eval_sum += float(parts.get("processed_ratio_eval", 0.0))
                    drop_ratio_eval_sum += float(parts.get("drop_ratio_eval", 0.0))
                    pre_backlog_steps_eval_sum += float(parts.get("pre_backlog_steps_eval", 0.0))
                    throughput_access_norm_sum += float(parts.get("throughput_access_norm", 0.0))
                    throughput_backhaul_norm_sum += float(parts.get("throughput_backhaul_norm", 0.0))
                    sat_processed_norm_sum += float(parts.get("sat_processed_norm", 0.0))
                    arrival_sum_ep += float(parts.get("arrival_sum", 0.0))
                    outflow_sum_ep += float(parts.get("outflow_sum", 0.0))
                    active_drop_sum_ep += float(parts.get("drop_sum_active", 0.0))
                    drop_sum_ep += float(parts.get("drop_sum", 0.0))
                    assoc_unfair_max_gu_count_sum += float(parts.get("assoc_unfair_max_gu_count", 0.0))
                    assoc_unfair_step_sum += float(parts.get("assoc_unfair_step", 0.0))
                    assoc_centroid_dist_norm_sum += float(parts.get("assoc_centroid_dist_norm_mean", 0.0))
                    centroid_dist_sum += float(parts.get("centroid_dist_mean", 0.0))
                    intervention_norm_sum += float(parts.get("intervention_norm", 0.0))
                    intervention_rate_sum += float(parts.get("intervention_rate", 0.0))

            steps = max(steps, 1)
            arrival_per_step = arrival_sum_ep / float(steps)
            arrival_denom = max(arrival_per_step, 1e-9)
            layer_queue_end_sum = {
                "gu": float(np.sum(env.gu_queue)),
                "uav": float(np.sum(env.uav_queue)),
                "sat": float(np.sum(env.sat_queue)),
            }
            layer_queue_arrival_steps_p95: dict[str, float] = {}
            layer_queue_drift_ratio: dict[str, float] = {}
            for layer in ("gu", "uav", "sat"):
                queue_sum_steps = layer_queue_sum_steps[layer]
                queue_sum_p95 = float(np.percentile(queue_sum_steps, 95)) if queue_sum_steps else 0.0
                layer_queue_arrival_steps_p95[layer] = queue_sum_p95 / arrival_denom
                layer_queue_drift_ratio[layer] = (
                    (layer_queue_end_sum[layer] - layer_queue_start_sum[layer]) / float(steps)
                ) / arrival_denom

            assoc_count_all_summary = _summarize(episode_assoc_count_values)
            bw_valid_count_step_max_summary = _summarize(episode_bw_valid_step_max_values)
            bw_valid_count_step_std_summary = _summarize(episode_bw_valid_step_std_values)
            assoc_count_step_max_summary = _summarize(episode_assoc_count_step_max_values)
            assoc_count_step_std_summary = _summarize(episode_assoc_count_step_std_values)
            extra_metrics = {
                "throughput_access_norm": throughput_access_norm_sum / float(steps),
                "throughput_backhaul_norm": throughput_backhaul_norm_sum / float(steps),
                "sat_processed_norm": sat_processed_norm_sum / float(steps),
                "gu_queue_mean": gu_queue_sum / float(steps),
                "uav_queue_mean": uav_queue_sum / float(steps),
                "sat_queue_mean": sat_queue_sum / float(steps),
                "gu_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["gu"],
                "uav_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["uav"],
                "sat_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["sat"],
                "gu_queue_drift_ratio": layer_queue_drift_ratio["gu"],
                "uav_queue_drift_ratio": layer_queue_drift_ratio["uav"],
                "sat_queue_drift_ratio": layer_queue_drift_ratio["sat"],
                "gu_drop_ratio": gu_drop_sum / max(arrival_sum_ep, 1e-9),
                "uav_drop_ratio": uav_drop_sum / max(arrival_sum_ep, 1e-9),
                "sat_drop_ratio": sat_drop_sum / max(arrival_sum_ep, 1e-9),
                "assoc_ratio_mean": assoc_ratio_sum / float(steps),
                "assoc_dist_mean": assoc_dist_sum / float(steps),
                "assoc_unfair_max_gu_count_mean": assoc_unfair_max_gu_count_sum / float(steps),
                "assoc_unfair_step_mean": assoc_unfair_step_sum / float(steps),
                "assoc_centroid_dist_norm_mean": assoc_centroid_dist_norm_sum / float(steps),
                "centroid_dist_mean": centroid_dist_sum / float(steps),
                "intervention_norm_mean": intervention_norm_sum / float(steps),
                "intervention_rate_mean": intervention_rate_sum / float(steps),
                "near_collision_ratio": near_collision_steps / float(steps),
                "min_inter_uav_dist": 0.0 if not np.isfinite(min_inter_uav_dist) else min_inter_uav_dist,
                "connected_sat_count_mean": connected_sat_count_sum / float(steps),
                "connected_sat_dist_mean": connected_sat_dist_mean_sum / float(steps),
                "arrival_sum": arrival_sum_ep,
                "outflow_sum": outflow_sum_ep,
                "active_drop_sum": active_drop_sum_ep,
                "drop_sum": drop_sum_ep,
                "sat_incoming_sum": sat_incoming_sum,
                "sat_processed_sum": sat_processed_sum,
                "active_net_drift_per_step": (arrival_sum_ep - sat_incoming_sum - active_drop_sum_ep) / float(steps),
                "sat_net_drift_per_step": (sat_incoming_sum - sat_processed_sum - sat_drop_sum) / float(steps),
                "total_net_drift_per_step": (arrival_sum_ep - sat_processed_sum - drop_sum_ep) / float(steps),
                "assoc_count_all_mean": assoc_count_all_summary["mean"],
                "assoc_count_all_p90": assoc_count_all_summary["p90"],
                "assoc_count_all_max": assoc_count_all_summary["max"],
                "assoc_count_step_max_mean": assoc_count_step_max_summary["mean"],
                "assoc_count_step_std_mean": assoc_count_step_std_summary["mean"],
                "bw_valid_count_step_max_mean": bw_valid_count_step_max_summary["mean"],
                "bw_valid_count_step_std_mean": bw_valid_count_step_std_summary["mean"],
            }

            rows.append(
                _episode_metric_row(
                    cell=cell,
                    episode=episode,
                    episode_seed=episode_seed,
                    reward_sum=reward_sum,
                    steps=steps,
                    processed_ratio_eval_sum=processed_ratio_eval_sum,
                    drop_ratio_eval_sum=drop_ratio_eval_sum,
                    pre_backlog_steps_eval_sum=pre_backlog_steps_eval_sum,
                    bw_valid_counts=episode_bw_valid_counts,
                    elapsed_sec=time.perf_counter() - t0,
                    extra_metrics=extra_metrics,
                )
            )
        summary = {
            "cell": cell.name,
            "bw_label": cell.bw_label,
            "partner_label": cell.partner_label,
            "config_path": os.path.abspath(cell.config_path),
            "primary_checkpoint": os.path.abspath(cell.primary_checkpoint),
            "teacher_actor_path": None if cell.teacher_actor_path is None else os.path.abspath(cell.teacher_actor_path),
            "exec_accel_source": cfg.exec_accel_source,
            "exec_bw_source": cfg.exec_bw_source,
            "exec_sat_source": cfg.exec_sat_source,
            "episodes": int(episodes),
            "episode_seed_base": None if episode_seed_base is None else int(episode_seed_base),
            "actor_adapted_key_count": int(len(actor_info.get("adapted_keys", []))),
            "teacher_adapted_key_count": None
            if teacher_info is None
            else int(len(teacher_info.get("adapted_keys", []))),
            "bw_valid_count_all_summary": _summarize(all_bw_valid_counts),
            "bw_valid_count_hist": _int_histogram(all_bw_valid_counts),
            "bw_valid_count_step_max_summary": _summarize(all_bw_valid_step_max_values),
            "bw_valid_count_step_std_summary": _summarize(all_bw_valid_step_std_values),
            "assoc_count_all_summary": _summarize(all_assoc_count_values),
            "assoc_count_hist": _int_histogram(all_assoc_count_values),
            "assoc_count_step_max_summary": _summarize(all_assoc_count_step_max_values),
            "assoc_count_step_std_summary": _summarize(all_assoc_count_step_std_values),
        }
        summary.update(
            _summarize_row_metrics(
                rows,
                exclude_keys={"cell", "bw_label", "partner_label", "episode", "episode_seed"},
            )
        )
        return rows, summary
    finally:
        env.close()


def _contrast(summary_map: dict[str, dict[str, Any]], a: str, b: str, metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cell_a": a,
        "cell_b": b,
    }
    for metric in metrics:
        av = summary_map[a].get(f"{metric}_mean")
        bv = summary_map[b].get(f"{metric}_mean")
        out[f"{metric}_a_minus_b"] = None if av is None or bv is None else float(av) - float(bv)
    return out


def _build_contrasts(args, summary_map: dict[str, dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    if args.matrix == "checkpoint_swap":
        return {
            "bwonly_partner_swap_cluster_minus_joint_partner": _contrast(
                summary_map,
                "bwonly_bw__cluster_partner",
                "bwonly_bw__joint_partner",
                metrics,
            ),
            "jointbw_partner_swap_cluster_minus_joint_partner": _contrast(
                summary_map,
                "joint_bw__cluster_partner",
                "joint_bw__joint_partner",
                metrics,
            ),
            "cluster_partner_bwonly_minus_jointbw": _contrast(
                summary_map,
                "bwonly_bw__cluster_partner",
                "joint_bw__cluster_partner",
                metrics,
            ),
            "joint_partner_bwonly_minus_jointbw": _contrast(
                summary_map,
                "bwonly_bw__joint_partner",
                "joint_bw__joint_partner",
                metrics,
            ),
        }
    if args.matrix == "bw_execution_swap":
        return {
            "cluster_partner_heuristic_minus_bwonly": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__heuristic_bw",
                "cluster_accel_cluster_sat__bwonly_bw",
                metrics,
            ),
            "cluster_partner_heuristic_minus_jointbw": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__heuristic_bw",
                "cluster_accel_cluster_sat__joint_bw",
                metrics,
            ),
            "cluster_partner_bwonly_minus_jointbw": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__bwonly_bw",
                "cluster_accel_cluster_sat__joint_bw",
                metrics,
            ),
            "jointaccel_partner_heuristic_minus_bwonly": _contrast(
                summary_map,
                "joint_accel_cluster_sat__heuristic_bw",
                "joint_accel_cluster_sat__bwonly_bw",
                metrics,
            ),
            "jointaccel_partner_heuristic_minus_jointbw": _contrast(
                summary_map,
                "joint_accel_cluster_sat__heuristic_bw",
                "joint_accel_cluster_sat__joint_bw",
                metrics,
            ),
            "jointaccel_partner_bwonly_minus_jointbw": _contrast(
                summary_map,
                "joint_accel_cluster_sat__bwonly_bw",
                "joint_accel_cluster_sat__joint_bw",
                metrics,
            ),
            "heuristic_bw_clusteraccel_minus_jointaccel": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__heuristic_bw",
                "joint_accel_cluster_sat__heuristic_bw",
                metrics,
            ),
            "bwonly_bw_clusteraccel_minus_jointaccel": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__bwonly_bw",
                "joint_accel_cluster_sat__bwonly_bw",
                metrics,
            ),
            "jointbw_clusteraccel_minus_jointaccel": _contrast(
                summary_map,
                "cluster_accel_cluster_sat__joint_bw",
                "joint_accel_cluster_sat__joint_bw",
                metrics,
            ),
        }

    prefix = f"{args.fixed_bw_source}_bw__"
    cluster_cluster = f"{prefix}cluster_accel_cluster_sat"
    joint_cluster = f"{prefix}joint_accel_cluster_sat"
    cluster_joint = f"{prefix}cluster_accel_joint_sat"
    joint_joint = f"{prefix}joint_accel_joint_sat"
    interaction: dict[str, float | None] = {}
    for metric in metrics:
        cc = summary_map[cluster_cluster].get(f"{metric}_mean")
        jc = summary_map[joint_cluster].get(f"{metric}_mean")
        cj = summary_map[cluster_joint].get(f"{metric}_mean")
        jj = summary_map[joint_joint].get(f"{metric}_mean")
        if None in {cc, jc, cj, jj}:
            interaction[f"{metric}_joint_joint_minus_additive"] = None
        else:
            interaction[f"{metric}_joint_joint_minus_additive"] = float(jj) - float(jc) - float(cj) + float(cc)
    return {
        "joint_accel_gain_over_cluster_given_cluster_sat": _contrast(
            summary_map,
            joint_cluster,
            cluster_cluster,
            metrics,
        ),
        "joint_sat_gain_over_cluster_given_cluster_accel": _contrast(
            summary_map,
            cluster_joint,
            cluster_cluster,
            metrics,
        ),
        "joint_sat_gain_over_cluster_given_joint_accel": _contrast(
            summary_map,
            joint_joint,
            joint_cluster,
            metrics,
        ),
        "joint_accel_gain_over_cluster_given_joint_sat": _contrast(
            summary_map,
            joint_joint,
            cluster_joint,
            metrics,
        ),
        "interaction": interaction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=str,
        default="checkpoint_swap",
        choices=["checkpoint_swap", "partner_component_split", "bw_execution_swap"],
    )
    parser.add_argument(
        "--fixed-bw-source",
        type=str,
        default="joint",
        choices=["joint", "bwonly"],
    )
    parser.add_argument(
        "--bwonly-run",
        type=str,
        default=r"runs\phase1_actions\bwonly_clustercenterexec_u600_env12_subproc_20260401",
    )
    parser.add_argument(
        "--joint-run",
        type=str,
        default=r"runs\bw_geom_credit_50u\a0_joint",
    )
    parser.add_argument(
        "--bwonly-checkpoint",
        type=str,
        default=r"runs\phase1_actions\bwonly_clustercenterexec_u600_env12_subproc_20260401\actor_u0050.pt",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=str,
        default=r"runs\bw_geom_credit_50u\a0_joint\actor_u0050.pt",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=42000)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"runs\partner_swap_matrix_u0050_20260401",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cells = _build_cells(args)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        print(f"[partner-swap] evaluating {cell.name}")
        rows, summary = evaluate_cell(
            cell=cell,
            episodes=int(args.episodes),
            episode_seed_base=args.episode_seed_base,
            device=device,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    summary_map = {item["cell"]: item for item in summaries}
    contrast_metrics = [
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "bw_valid_count_mean",
        "bw_valid_count_p90",
    ]
    contrasts = _build_contrasts(args, summary_map, contrast_metrics)

    csv_path = out_dir / "episode_metrics.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    summary_csv_path = out_dir / "cell_summary.csv"
    if summaries:
        summary_csv_fields = [
            "cell",
            "bw_label",
            "partner_label",
            "episodes",
            "episode_seed_base",
            "reward_sum_mean",
            "reward_sum_std",
            "processed_ratio_eval_mean",
            "drop_ratio_eval_mean",
            "pre_backlog_steps_eval_mean",
            "throughput_access_norm_mean",
            "throughput_backhaul_norm_mean",
            "gu_queue_arrival_steps_p95_mean",
            "uav_queue_arrival_steps_p95_mean",
            "sat_queue_arrival_steps_p95_mean",
            "gu_drop_ratio_mean",
            "uav_drop_ratio_mean",
            "sat_drop_ratio_mean",
            "assoc_ratio_mean_mean",
            "assoc_dist_mean_mean",
            "assoc_unfair_max_gu_count_mean_mean",
            "assoc_unfair_step_mean_mean",
            "assoc_count_step_max_mean_mean",
            "assoc_count_step_std_mean_mean",
            "bw_valid_count_mean_mean",
            "bw_valid_count_p90_mean",
            "bw_valid_count_step_max_mean_mean",
            "bw_valid_count_all_summary",
            "assoc_count_all_summary",
        ]
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_csv_fields)
            writer.writeheader()
            for row in summaries:
                out_row = {k: row.get(k) for k in summary_csv_fields}
                out_row["bw_valid_count_all_summary"] = json.dumps(
                    row.get("bw_valid_count_all_summary", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                out_row["assoc_count_all_summary"] = json.dumps(
                    row.get("assoc_count_all_summary", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                writer.writerow(out_row)

    summary_json_path = out_dir / "summary.json"
    payload = {
        "meta": {
            "matrix": args.matrix,
            "fixed_bw_source": args.fixed_bw_source,
            "episodes": int(args.episodes),
            "episode_seed_base": int(args.episode_seed_base) if args.episode_seed_base is not None else None,
            "device": str(device),
            "cells": [asdict(cell) for cell in cells],
        },
        "cells": summaries,
        "contrasts": contrasts,
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote episode metrics to {csv_path}")
    print(f"Wrote cell summary to {summary_csv_path}")
    print(f"Wrote summary json to {summary_json_path}")


if __name__ == "__main__":
    main()
