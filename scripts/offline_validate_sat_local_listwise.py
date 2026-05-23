from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    _collate_dataclass,
    _current_obs_list,
    _heuristic_accel,
    _heuristic_bw,
    _refresh_stage_obs_cache,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml",
    )
    parser.add_argument(
        "--init-actor",
        type=str,
        default="runs/tmp_sat_clean_smoke/actor_final.pt",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-seed-base", type=int, default=61000)
    parser.add_argument("--step-stride", type=int, default=40)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--max-panel-size", type=int, default=10)
    parser.add_argument("--holdout-size", type=int, default=4)
    parser.add_argument("--min-best-gap", type=float, default=0.0)
    parser.add_argument("--target-temp", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--accel-source", type=str, default="cluster_center_queue_aware")
    parser.add_argument("--bw-source", type=str, default="queue_aware")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _cluster_centers_and_counts(env: SaginParallelEnv) -> tuple[np.ndarray | None, np.ndarray | None]:
    centers = getattr(env, "gu_cluster_centers", None)
    counts = getattr(env, "gu_cluster_counts", None)
    centers_arr = None if centers is None else np.asarray(centers, dtype=np.float32)
    counts_arr = None if counts is None else np.asarray(counts, dtype=np.float32)
    return centers_arr, counts_arr


def _sat_action_ids_to_text(action_ids: np.ndarray) -> str:
    rows: list[str] = []
    for row in np.asarray(action_ids, dtype=np.int64):
        valid = [str(int(x)) for x in row.tolist() if int(x) >= 0]
        rows.append("-" if not valid else "|".join(valid))
    return ";".join(rows)


def _cpu_dataclass(batch: Any) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_cpu_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        kwargs[field.name] = value.detach().cpu()
    return type(batch)(**kwargs)


def _to_device_dataclass(batch: Any, device: torch.device) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_to_device_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        kwargs[field.name] = value.to(device=device)
    return type(batch)(**kwargs)


def _repeat_dataclass(batch: Any, repeats: int) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_repeat_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        view = value.unsqueeze(0).expand(int(repeats), *value.shape)
        kwargs[field.name] = view.reshape(int(repeats) * int(value.shape[0]), *value.shape[1:])
    return type(batch)(**kwargs)


def _tail_discounted_returns(step_rewards: list[float], gamma: float) -> list[float]:
    tail = [0.0 for _ in step_rewards]
    running = 0.0
    for idx in range(len(step_rewards) - 1, -1, -1):
        running = float(step_rewards[idx]) + float(gamma) * float(running)
        tail[idx] = float(running)
    return tail


def _canonicalize_sat_action_row(
    row_sat_ids: list[int],
    *,
    visible_ids: list[int],
    select_k: int,
) -> np.ndarray:
    visible_rank = {int(sat_id): idx for idx, sat_id in enumerate(visible_ids)}
    unique_valid = [int(sat_id) for sat_id in row_sat_ids if int(sat_id) >= 0]
    unique_valid = sorted(set(unique_valid), key=lambda sat_id: visible_rank.get(int(sat_id), 10**9))
    out = np.full((int(select_k),), -1, dtype=np.int64)
    keep = unique_valid[: int(select_k)]
    if keep:
        out[: len(keep)] = np.asarray(keep, dtype=np.int64)
    return out


def _enumerate_single_swap_candidates(
    *,
    current_sat_action_ids: np.ndarray,
    visible_per_uav: list[list[int]],
    select_k: int,
) -> list[dict[str, Any]]:
    current = np.asarray(current_sat_action_ids, dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for u in range(int(current.shape[0])):
        visible_ids = [int(x) for x in list(visible_per_uav[u])]
        current_row = [int(x) for x in current[u].tolist() if int(x) >= 0]
        if not current_row:
            continue
        alternatives = [sat_id for sat_id in visible_ids if sat_id not in set(current_row)]
        if not alternatives:
            continue
        for pos, old_sat in enumerate(current_row):
            for new_sat in alternatives:
                cand = np.asarray(current, dtype=np.int64).copy()
                swapped = list(current_row)
                swapped[pos] = int(new_sat)
                cand[u] = _canonicalize_sat_action_row(
                    swapped,
                    visible_ids=visible_ids,
                    select_k=int(select_k),
                )
                key = tuple(tuple(int(x) for x in row) for row in cand.tolist())
                if key in seen or np.array_equal(cand, current):
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "action_ids": cand,
                        "uav": int(u),
                        "drop_sat_id": int(old_sat),
                        "add_sat_id": int(new_sat),
                    }
                )
    return candidates


def _action_ids_to_pair_indices(
    local_states: list[Any],
    visible_per_uav: list[list[int]],
    action_ids: np.ndarray,
) -> list[int] | None:
    pair_indices: list[int] = []
    action_arr = np.asarray(action_ids, dtype=np.int64)
    for u, state in enumerate(local_states):
        visible = [int(x) for x in list(visible_per_uav[u])]
        visible_rank = {int(sat_id): idx for idx, sat_id in enumerate(visible)}
        subset_mask = np.asarray(state.subset_mask[0].detach().cpu().numpy() > 0.5, dtype=bool)
        subset_members = np.asarray(state.subset_members[0].detach().cpu().numpy(), dtype=np.int64)
        wanted = tuple(
            sorted(
                [int(x) for x in action_arr[u].tolist() if int(x) >= 0],
                key=lambda sat_id: visible_rank.get(int(sat_id), 10**9),
            )
        )
        match = -1
        for subset_idx in np.flatnonzero(subset_mask).tolist():
            member_slots = subset_members[int(subset_idx)]
            sat_ids = []
            for slot in member_slots.tolist():
                if int(slot) < 0 or int(slot) >= len(visible):
                    continue
                sat_ids.append(int(visible[int(slot)]))
            sat_key = tuple(sorted(set(sat_ids), key=lambda sat_id: visible_rank.get(int(sat_id), 10**9)))
            if sat_key == wanted:
                match = int(subset_idx)
                break
        if match < 0:
            return None
        pair_indices.append(match)
    return pair_indices


def _actor_sat_action(
    actor,
    local_states: list[Any],
    driver: StructuredControlDriver,
    device: torch.device,
) -> tuple[list[int], np.ndarray, Any]:
    local_batch = _collate_dataclass(local_states, device)
    with torch.inference_mode():
        out = actor.sat_pair_policy(local_batch, deterministic=True)
    pair_indices = out.subset_index.detach().cpu().numpy().astype(np.int64, copy=False).tolist()
    action_ids = driver.decode_sat_pair_actions(local_states, pair_indices)
    return pair_indices, np.asarray(action_ids, dtype=np.int64), local_batch


def _rollout_single_sat_action_return(
    *,
    probe_driver: StructuredControlDriver,
    actor,
    snapshot_state: dict[str, Any],
    first_sat_action_ids: np.ndarray,
    gamma: float,
    cfg,
    accel_source: str,
    bw_source: str,
    device: torch.device,
) -> float:
    probe_driver.load_sat_stage_state(snapshot_state)
    total = 0.0
    discount = 1.0
    max_steps = max(int(getattr(cfg, "T_steps", 1)), 1)
    centers, counts = _cluster_centers_and_counts(probe_driver.env)

    probe_driver.run_sat_stage(np.asarray(first_sat_action_ids, dtype=np.int64))
    _refresh_stage_obs_cache(probe_driver)
    bw_obs = _current_obs_list(probe_driver)
    bw_action = _heuristic_bw(bw_obs, cfg, bw_source).astype(np.float32, copy=False)
    step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
    reward = float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values()))))
    total += float(discount) * reward
    done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
    if done:
        return float(total)
    discount *= float(gamma)

    for _step in range(1, max_steps):
        obs_before = _current_obs_list(probe_driver)
        accel_action = _heuristic_accel(
            obs_before,
            cfg,
            accel_source,
            centers=centers,
            counts=counts,
        ).astype(np.float32, copy=False)
        sat_world = probe_driver.run_accel_stage(accel_action)
        del sat_world
        _refresh_stage_obs_cache(probe_driver)
        local_states = probe_driver.build_sat_pair_candidates()
        _pair_indices, sat_action_ids, _local_batch = _actor_sat_action(actor, local_states, probe_driver, device)
        probe_driver.run_sat_stage(sat_action_ids)
        _refresh_stage_obs_cache(probe_driver)
        bw_obs = _current_obs_list(probe_driver)
        bw_action = _heuristic_bw(bw_obs, cfg, bw_source).astype(np.float32, copy=False)
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
        reward = float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values()))))
        total += float(discount) * reward
        done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
        if done:
            break
        discount *= float(gamma)
    return float(total)


def _build_panel_candidates(
    *,
    local_states: list[Any],
    visible_per_uav: list[list[int]],
    current_pair_indices: list[int],
    current_action_ids: np.ndarray,
    select_k: int,
    max_panel_size: int,
) -> tuple[list[list[int]], list[np.ndarray], list[str], list[dict[str, Any]]]:
    pair_panel: list[list[int]] = [list(int(x) for x in current_pair_indices)]
    action_panel: list[np.ndarray] = [np.asarray(current_action_ids, dtype=np.int64).copy()]
    text_panel: list[str] = [_sat_action_ids_to_text(np.asarray(current_action_ids, dtype=np.int64))]
    meta_panel: list[dict[str, Any]] = [{"kind": "executed"}]
    swap_candidates = _enumerate_single_swap_candidates(
        current_sat_action_ids=np.asarray(current_action_ids, dtype=np.int64),
        visible_per_uav=visible_per_uav,
        select_k=int(select_k),
    )
    limit = max(int(max_panel_size) - 1, 0)
    for cand in swap_candidates[:limit]:
        pair_indices = _action_ids_to_pair_indices(local_states, visible_per_uav, cand["action_ids"])
        if pair_indices is None:
            continue
        pair_key = tuple(int(x) for x in pair_indices)
        if pair_key in {tuple(int(x) for x in row) for row in pair_panel}:
            continue
        pair_panel.append([int(x) for x in pair_indices])
        action_panel.append(np.asarray(cand["action_ids"], dtype=np.int64).copy())
        text_panel.append(_sat_action_ids_to_text(np.asarray(cand["action_ids"], dtype=np.int64)))
        meta_panel.append(
            {
                "kind": "single_swap",
                "uav": int(cand["uav"]),
                "drop_sat_id": int(cand["drop_sat_id"]),
                "add_sat_id": int(cand["add_sat_id"]),
            }
        )
    return pair_panel, action_panel, text_panel, meta_panel


def _collect_dataset(
    *,
    actor,
    cfg,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    gamma = float(getattr(cfg, "gamma", 1.0) or 1.0)
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    live_env = make_structured_env(cfg, mode="script")
    probe_env = make_structured_env(cfg, mode="script")
    live_driver = as_structured_driver(live_env)
    probe_driver = as_structured_driver(probe_env)
    entries: list[dict[str, Any]] = []
    try:
        for episode in range(int(args.episodes)):
            if len(entries) >= int(args.max_contexts):
                break
            seed = int(args.episode_seed_base) + int(episode)
            live_env.reset(seed=seed)
            actor.eval()
            done = False
            step = 0
            step_rewards: list[float] = []
            pending_contexts: list[dict[str, Any]] = []
            while not done and len(entries) + len(pending_contexts) < int(args.max_contexts):
                _ = live_driver.begin_step()
                obs_before = _current_obs_list(live_driver)
                centers, counts = _cluster_centers_and_counts(live_env)
                accel_action = _heuristic_accel(
                    obs_before,
                    cfg,
                    str(args.accel_source),
                    centers=centers,
                    counts=counts,
                ).astype(np.float32, copy=False)
                _ = live_driver.run_accel_stage(accel_action)
                _refresh_stage_obs_cache(live_driver)
                local_states = live_driver.build_sat_pair_candidates()
                current_pair_indices, current_sat_action_ids, local_batch = _actor_sat_action(
                    actor,
                    local_states,
                    live_driver,
                    device,
                )
                sampled = (step % max(int(args.step_stride), 1)) == 0
                if sampled:
                    visible_per_uav = [list(v) for v in (live_driver._stage_visible or [[] for _ in range(cfg.num_uav)])]
                    pair_panel, action_panel, text_panel, meta_panel = _build_panel_candidates(
                        local_states=local_states,
                        visible_per_uav=visible_per_uav,
                        current_pair_indices=current_pair_indices,
                        current_action_ids=current_sat_action_ids,
                        select_k=int(select_k),
                        max_panel_size=int(args.max_panel_size),
                    )
                    if len(pair_panel) >= 2:
                        pending_contexts.append(
                            {
                                "episode": int(episode),
                                "seed": int(seed),
                                "step": int(step),
                                "snapshot_state": live_driver.export_sat_stage_state(),
                                "local_batch_cpu": _cpu_dataclass(_to_device_dataclass(local_batch, torch.device("cpu"))),
                                "pair_panel": pair_panel,
                                "action_panel": action_panel,
                                "text_panel": text_panel,
                                "meta_panel": meta_panel,
                            }
                        )
                live_driver.run_sat_stage(current_sat_action_ids)
                _refresh_stage_obs_cache(live_driver)
                bw_obs = _current_obs_list(live_driver)
                bw_action = _heuristic_bw(bw_obs, cfg, str(args.bw_source)).astype(np.float32, copy=False)
                step_result, _next_world = live_driver.execute_stage_bw_and_prepare_next_accel(bw_action)
                reward = float(getattr(step_result, "bw_weighted_workload_level_reward", next(iter(step_result.rewards.values()))))
                step_rewards.append(reward)
                done = bool(next(iter(step_result.terminations.values())) or next(iter(step_result.truncations.values())))
                step += 1

            tail_returns = _tail_discounted_returns(step_rewards, gamma)
            for pending in pending_contexts:
                baseline_score = float(tail_returns[int(pending["step"])]) if pending["step"] < len(tail_returns) else 0.0
                panel_scores: list[float] = [baseline_score]
                for alt_action_ids in pending["action_panel"][1:]:
                    alt_score = _rollout_single_sat_action_return(
                        probe_driver=probe_driver,
                        actor=actor,
                        snapshot_state=pending["snapshot_state"],
                        first_sat_action_ids=np.asarray(alt_action_ids, dtype=np.int64),
                        gamma=gamma,
                        cfg=cfg,
                        accel_source=str(args.accel_source),
                        bw_source=str(args.bw_source),
                        device=device,
                    )
                    panel_scores.append(float(alt_score))
                scores_arr = np.asarray(panel_scores, dtype=np.float64)
                best_idx = int(np.argmax(scores_arr))
                best_gap = float(scores_arr[best_idx] - scores_arr[0])
                if float(best_gap) + 1.0e-12 < float(args.min_best_gap):
                    continue
                entry = {
                    "context_id": int(len(entries)),
                    "episode": int(pending["episode"]),
                    "seed": int(pending["seed"]),
                    "step": int(pending["step"]),
                    "executed_idx": 0,
                    "best_idx": int(best_idx),
                    "best_gap": float(best_gap),
                    "baseline_score": float(scores_arr[0]),
                    "panel_scores": [float(x) for x in scores_arr.tolist()],
                    "panel_pair_indices": [[int(x) for x in row] for row in pending["pair_panel"]],
                    "panel_action_text": [str(x) for x in pending["text_panel"]],
                    "panel_meta": pending["meta_panel"],
                    "candidate_count": int(len(scores_arr)),
                    "local_batch_cpu": pending["local_batch_cpu"],
                }
                entries.append(entry)
                if len(entries) >= int(args.max_contexts):
                    break
            if len(entries) >= int(args.max_contexts):
                break
    finally:
        for env in (live_env, probe_env):
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
    return entries


def _panel_logprob(actor, local_batch_cpu: Any, panel_pair_indices: list[list[int]], device: torch.device) -> torch.Tensor:
    local_batch = _to_device_dataclass(local_batch_cpu, device)
    joint_actions = torch.as_tensor(panel_pair_indices, dtype=torch.long, device=device)
    num_candidates = int(joint_actions.shape[0])
    num_agents = int(joint_actions.shape[1])
    repeated_batch = _repeat_dataclass(local_batch, num_candidates)
    eval_out = actor.evaluate_sat_pair(repeated_batch, joint_actions.reshape(-1))
    joint_logprob = eval_out.logprob.reshape(num_candidates, num_agents).sum(dim=-1)
    return joint_logprob


def _entry_listwise_loss(actor, entry: dict[str, Any], device: torch.device, target_temp: float) -> torch.Tensor:
    scores = torch.as_tensor(entry["panel_scores"], dtype=torch.float32, device=device)
    joint_logprob = _panel_logprob(actor, entry["local_batch_cpu"], entry["panel_pair_indices"], device)
    target_logits = (scores - scores.max()) / max(float(target_temp), 1.0e-6)
    target_dist = torch.softmax(target_logits, dim=0)
    return -(target_dist * torch.log_softmax(joint_logprob, dim=0)).sum()


def _evaluate_entries(
    actor,
    entries: list[dict[str, Any]],
    device: torch.device,
    target_temp: float,
) -> dict[str, float]:
    if not entries:
        return {
            "count": 0.0,
            "listwise_loss": 0.0,
            "top1_hit": 0.0,
            "expected_gap_to_best": 0.0,
            "det_gap_to_best": 0.0,
            "expected_gain_vs_executed": 0.0,
            "det_gain_vs_executed": 0.0,
            "oracle_gap_best_executed": 0.0,
            "candidate_count_mean": 0.0,
        }
    actor.eval()
    listwise_losses: list[float] = []
    top1_hits: list[float] = []
    expected_gap_to_best: list[float] = []
    det_gap_to_best: list[float] = []
    expected_gain_vs_executed: list[float] = []
    det_gain_vs_executed: list[float] = []
    oracle_gap_best_executed: list[float] = []
    candidate_count_mean: list[float] = []
    with torch.inference_mode():
        for entry in entries:
            scores = np.asarray(entry["panel_scores"], dtype=np.float64)
            joint_logprob = _panel_logprob(actor, entry["local_batch_cpu"], entry["panel_pair_indices"], device)
            panel_prob = torch.softmax(joint_logprob, dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
            pred_idx = int(np.argmax(panel_prob))
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            executed_score = float(scores[int(entry["executed_idx"])])
            expected_score = float(np.sum(panel_prob * scores, dtype=np.float64))
            det_score = float(scores[pred_idx])
            loss = float(_entry_listwise_loss(actor, entry, device, target_temp).item())
            listwise_losses.append(loss)
            top1_hits.append(float(pred_idx == best_idx))
            expected_gap_to_best.append(float(best_score - expected_score))
            det_gap_to_best.append(float(best_score - det_score))
            expected_gain_vs_executed.append(float(expected_score - executed_score))
            det_gain_vs_executed.append(float(det_score - executed_score))
            oracle_gap_best_executed.append(float(best_score - executed_score))
            candidate_count_mean.append(float(len(scores)))
    return {
        "count": float(len(entries)),
        "listwise_loss": _safe_mean(listwise_losses),
        "top1_hit": _safe_mean(top1_hits),
        "expected_gap_to_best": _safe_mean(expected_gap_to_best),
        "det_gap_to_best": _safe_mean(det_gap_to_best),
        "expected_gain_vs_executed": _safe_mean(expected_gain_vs_executed),
        "det_gain_vs_executed": _safe_mean(det_gain_vs_executed),
        "oracle_gap_best_executed": _safe_mean(oracle_gap_best_executed),
        "candidate_count_mean": _safe_mean(candidate_count_mean),
    }


def _train_offline_listwise(
    actor,
    train_entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for param in actor.parameters():
        param.requires_grad_(False)
    for param in actor.sat_pair_policy.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(
        actor.sat_pair_policy.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, Any]] = []
    best_holdout = _evaluate_entries(actor, holdout_entries, device, float(args.target_temp))
    best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    best_metric = float(best_holdout["listwise_loss"])
    for epoch in range(1, max(int(args.epochs), 1) + 1):
        actor.train()
        order = list(range(len(train_entries)))
        random.shuffle(order)
        batch_losses: list[float] = []
        for start in range(0, len(order), max(int(args.batch_size), 1)):
            batch_ids = order[start : start + max(int(args.batch_size), 1)]
            optimizer.zero_grad(set_to_none=True)
            loss_terms: list[torch.Tensor] = []
            for idx in batch_ids:
                loss_terms.append(_entry_listwise_loss(actor, train_entries[int(idx)], device, float(args.target_temp)))
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.sat_pair_policy.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_losses.append(float(loss.item()))
        if epoch == 1 or epoch == int(args.epochs) or (int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0):
            train_metrics = _evaluate_entries(actor, train_entries, device, float(args.target_temp))
            holdout_metrics = _evaluate_entries(actor, holdout_entries, device, float(args.target_temp))
            record = {
                "epoch": int(epoch),
                "train_loss_epoch_mean": _safe_mean(batch_losses),
                "train_metrics": train_metrics,
                "holdout_metrics": holdout_metrics,
            }
            history.append(record)
            if holdout_entries and float(holdout_metrics["listwise_loss"]) < float(best_metric):
                best_metric = float(holdout_metrics["listwise_loss"])
                best_holdout = holdout_metrics
                best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    actor.load_state_dict(best_state, strict=True)
    final_train = _evaluate_entries(actor, train_entries, device, float(args.target_temp))
    final_holdout = _evaluate_entries(actor, holdout_entries, device, float(args.target_temp))
    return history, {
        "best_holdout": best_holdout,
        "final_train": final_train,
        "final_holdout": final_holdout,
    }


def _split_entries(entries: list[dict[str, Any]], holdout_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order = list(range(len(entries)))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    holdout_n = max(0, min(int(holdout_size), len(order)))
    holdout_idx = set(order[:holdout_n])
    train_entries = [entry for idx, entry in enumerate(entries) if idx not in holdout_idx]
    holdout_entries = [entry for idx, entry in enumerate(entries) if idx in holdout_idx]
    return train_entries, holdout_entries


def _write_entries_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = {
            "context_id": int(entry["context_id"]),
            "episode": int(entry["episode"]),
            "seed": int(entry["seed"]),
            "step": int(entry["step"]),
            "candidate_count": int(entry["candidate_count"]),
            "executed_idx": int(entry["executed_idx"]),
            "best_idx": int(entry["best_idx"]),
            "best_gap": float(entry["best_gap"]),
            "baseline_score": float(entry["baseline_score"]),
            "panel_scores": json.dumps(entry["panel_scores"], ensure_ascii=False),
            "panel_action_text": json.dumps(entry["panel_action_text"], ensure_ascii=False),
            "panel_meta": json.dumps(entry["panel_meta"], ensure_ascii=False),
        }
        rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["context_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_all_seeds(int(args.seed))
    device = torch.device(str(args.device))

    cfg = load_config(str(args.config))
    bundle = build_structured_modules_from_config(cfg, build_critic=False)
    actor_init = bundle.actor.to(device=device)
    checkpoint_info = None
    init_actor_path = Path(str(args.init_actor))
    if str(args.init_actor).strip():
        if not init_actor_path.exists():
            raise FileNotFoundError(f"init actor checkpoint not found: {init_actor_path}")
        checkpoint_info = load_checkpoint_forgiving(actor_init, str(init_actor_path), map_location=device)
    actor_collect = actor_init
    entries = _collect_dataset(actor=actor_collect, cfg=cfg, args=args, device=device)
    if len(entries) < 2:
        raise RuntimeError(f"offline validation requires at least 2 contexts, got {len(entries)}")

    train_entries, holdout_entries = _split_entries(entries, int(args.holdout_size), int(args.seed))
    actor_train = build_structured_modules_from_config(cfg, build_critic=False).actor.to(device=device)
    if checkpoint_info is not None:
        _ = load_checkpoint_forgiving(actor_train, str(init_actor_path), map_location=device)
    before_train = _evaluate_entries(actor_train, train_entries, device, float(args.target_temp))
    before_holdout = _evaluate_entries(actor_train, holdout_entries, device, float(args.target_temp))
    history, final_metrics = _train_offline_listwise(
        actor_train,
        train_entries,
        holdout_entries,
        args=args,
        device=device,
    )
    torch.save(actor_train.state_dict(), out_dir / "actor_offline_listwise.pt")
    _write_entries_csv(out_dir / "entries.csv", entries)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    summary = {
        "config": str(args.config),
        "init_actor": str(args.init_actor),
        "checkpoint_info": checkpoint_info,
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "step_stride": int(args.step_stride),
        "max_contexts": int(args.max_contexts),
        "max_panel_size": int(args.max_panel_size),
        "contexts_collected": int(len(entries)),
        "train_contexts": int(len(train_entries)),
        "holdout_contexts": int(len(holdout_entries)),
        "min_best_gap": float(args.min_best_gap),
        "target_temp": float(args.target_temp),
        "candidate_count": _summarize([float(entry["candidate_count"]) for entry in entries]),
        "oracle_best_gap": _summarize([float(entry["best_gap"]) for entry in entries]),
        "before_train": before_train,
        "before_holdout": before_holdout,
        "final_train": final_metrics["final_train"],
        "final_holdout": final_metrics["final_holdout"],
        "best_holdout": final_metrics["best_holdout"],
        "entries_csv": str(out_dir / "entries.csv"),
        "history_json": str(out_dir / "history.json"),
        "actor_path": str(out_dir / "actor_offline_listwise.pt"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
