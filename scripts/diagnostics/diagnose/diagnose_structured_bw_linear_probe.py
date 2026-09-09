from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_actor import _attend
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_types import LocalBwState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    corr = _safe_corr(_rankdata_desc(pred).tolist(), _rankdata_desc(truth).tolist())
    return float(corr or 0.0)


def _pairwise_accuracy_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size <= 1 or truth.size <= 1 or pred.size != truth.size:
        return 0.0
    correct = 0.0
    total = 0.0
    for i in range(pred.size):
        for j in range(i + 1, pred.size):
            truth_gap = float(truth[i] - truth[j])
            if abs(truth_gap) <= 1.0e-12:
                continue
            pred_gap = float(pred[i] - pred[j])
            total += 1.0
            if abs(pred_gap) <= 1.0e-12:
                correct += 0.5
            elif pred_gap * truth_gap > 0.0:
                correct += 1.0
    if total <= 0.0:
        return 0.0
    return float(correct / total)


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


def _bucket_mean(counts: list[int], values: list[float]) -> list[dict[str, float]]:
    buckets: dict[int, list[float]] = {}
    for count, value in zip(counts, values):
        buckets.setdefault(int(count), []).append(float(value))
    rows: list[dict[str, float]] = []
    for count in sorted(buckets):
        arr = np.asarray(buckets[count], dtype=np.float64)
        rows.append({"count": float(count), "n": float(arr.size), "mean": float(np.mean(arr))})
    return rows


def _refresh_stage_obs_cache(driver: StructuredControlDriver) -> None:
    if driver._stage_assoc is None or driver._stage_candidates is None:
        raise RuntimeError("run_accel_stage must be called before refreshing stage obs cache")
    env = driver.env
    env._cached_assoc = driver._stage_assoc.copy()
    env._cached_candidates = [list(c) for c in driver._stage_candidates]
    if driver._stage_bw_valid_mask is not None:
        env._cached_bw_valid_mask = driver._stage_bw_valid_mask.copy()
    dummy_actions = env._dummy_actions()
    _, env._cached_eta = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        dummy_actions,
        record_exec=False,
    )
    if driver._stage_sat_pos is not None and driver._stage_sat_vel is not None and driver._stage_visible is not None:
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)


def _current_obs_list(env: SaginParallelEnv) -> list[dict[str, np.ndarray]]:
    return [env._get_obs(i) for i in range(len(env.agents))]


def _load_actor(run_dir: Path, update: int, device: torch.device):
    cfg = load_config(str(run_dir / "config_source.yaml"))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor_ckpt = run_dir / f"actor_u{int(update):04d}.pt"
    load_checkpoint_forgiving(bundle.actor, str(actor_ckpt), map_location=device, strict=True)
    bundle.actor.to(device).eval()
    return cfg, bundle.actor


def _queue_eta_prev_target(obs: dict[str, np.ndarray], assoc_bonus: float) -> np.ndarray:
    users = np.asarray(obs["users"], dtype=np.float32)
    valid = np.asarray(obs["bw_valid_mask"] > 0.0, dtype=bool)
    q = np.clip(users[:, 2], 0.0, None)
    eta_ref = np.clip(users[:, 3], 0.0, None)
    prev = np.clip(users[:, 4], 0.0, 1.0)
    target = q * (0.5 + eta_ref) * (1.0 + float(assoc_bonus) * prev)
    return target[valid]


def _normalize_target(target: np.ndarray) -> np.ndarray | None:
    if target.ndim != 1 or target.size <= 1:
        return None
    target_log = np.log(np.clip(target, 1.0e-8, None))
    centered = target_log - float(np.mean(target_log))
    scale = float(np.std(centered))
    if scale <= 1.0e-8:
        return None
    return (centered / scale).astype(np.float32)


def _bw_intermediates(bw_policy, local_state: LocalBwState) -> dict[str, torch.Tensor]:
    valid_mask = (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    ego_0 = bw_policy.ego_encoder(local_state.ego_uav_after_sat)
    sat_0 = bw_policy.sat_encoder(torch.cat([local_state.sat_nodes, local_state.sat_edges], dim=-1))
    user_raw = torch.cat([local_state.user_nodes, local_state.user_edges], dim=-1)
    user_0 = bw_policy.user_encoder(user_raw)

    sat_ctx_1 = _attend(ego_0, sat_0, local_state.sat_mask)
    query_1 = bw_policy.query_proj_1(torch.cat([ego_0, sat_ctx_1], dim=-1))
    user_ctx_1 = _attend(query_1, user_0, valid_mask)

    sat_1 = sat_0 + bw_policy.sat_refine(torch.cat([sat_0, ego_0.unsqueeze(1).expand_as(sat_0)], dim=-1))
    user_1 = user_0 + bw_policy.user_refine(
        torch.cat(
            [
                user_0,
                ego_0.unsqueeze(1).expand_as(user_0),
                sat_ctx_1.unsqueeze(1).expand_as(user_0),
                user_ctx_1.unsqueeze(1).expand_as(user_0),
            ],
            dim=-1,
        )
    )

    query_2 = bw_policy.query_proj_2(torch.cat([ego_0, sat_ctx_1, user_ctx_1], dim=-1))
    sat_ctx_2 = _attend(query_2, sat_1, local_state.sat_mask)
    user_ctx_2 = _attend(query_2, user_1, valid_mask)
    fused = bw_policy.user_fusion(
        torch.cat(
            [
                ego_0.unsqueeze(1).expand_as(user_1),
                sat_ctx_2.unsqueeze(1).expand_as(user_1),
                user_ctx_2.unsqueeze(1).expand_as(user_1),
                user_1,
            ],
            dim=-1,
        )
    )
    return {
        "raw": user_raw,
        "user0": user_0,
        "user1": user_1,
        "fused": fused,
    }


def _collect_probe_rows(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    episode_seed_base: int,
    deterministic: bool,
    device: torch.device,
) -> dict[str, Any]:
    cfg, actor = _load_actor(run_dir, int(update), device)
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    env = make_structured_env(cfg, mode="script")
    rows: list[dict[str, Any]] = []
    skipped_singleton = 0
    skipped_flat_target = 0
    total_states = 0

    try:
        for ep in range(int(episodes)):
            env.reset(seed=int(episode_seed_base) + ep)
            driver = as_structured_driver(env)
            done = False
            while not done:
                z0 = driver.begin_step()
                accel_states = driver.build_local_accel_states(z0)
                accel_batch = _collate_dataclass(accel_states, device)
                with torch.inference_mode():
                    accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
                z1 = driver.run_accel_stage(accel_out.action.detach().cpu().numpy())
                _refresh_stage_obs_cache(driver)
                obs_after_accel = _current_obs_list(env)

                sat_states = driver.build_sat_pair_candidates(z1)
                sat_batch = _collate_dataclass(sat_states, device)
                with torch.inference_mode():
                    sat_out = actor.act_sat_pair(sat_batch, deterministic=deterministic)
                sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
                z2 = driver.run_sat_stage(sat_action)

                bw_states = driver.build_bw_valid_context(z2)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.inference_mode():
                    reprs = _bw_intermediates(actor.bw_policy, bw_batch)
                    bw_out = actor.act_bw(bw_batch, deterministic=deterministic)

                valid_mask = ((bw_batch.user_mask > 0.5) & (bw_batch.bw_valid_mask > 0.5)).detach().cpu().numpy()
                repr_np = {name: tensor.detach().cpu().numpy() for name, tensor in reprs.items()}

                for u in range(int(cfg.num_uav)):
                    total_states += 1
                    valid = np.asarray(valid_mask[u], dtype=bool)
                    valid_count = int(np.sum(valid))
                    if valid_count <= 1:
                        skipped_singleton += 1
                        continue
                    target = _queue_eta_prev_target(obs_after_accel[u], assoc_bonus=assoc_bonus)
                    target_z = _normalize_target(target)
                    if target_z is None:
                        skipped_flat_target += 1
                        continue
                    row: dict[str, Any] = {
                        "episode": int(ep),
                        "valid_user_count": int(valid_count),
                        "target": target.astype(np.float32, copy=True),
                        "target_z": target_z.astype(np.float32, copy=True),
                    }
                    for name, values in repr_np.items():
                        row[name] = np.asarray(values[u, valid, :], dtype=np.float32)
                    rows.append(row)

                step = driver.execute_stage_bw_and_step(bw_out.action.detach().cpu().numpy())
                done = bool(any(step.terminations.values()) or any(step.truncations.values()))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "cfg": cfg,
        "loc_readout": str(getattr(actor.bw_policy, "loc_readout", "fused")),
        "rows": rows,
        "total_states": int(total_states),
        "skipped_singleton": int(skipped_singleton),
        "skipped_flat_target": int(skipped_flat_target),
    }


def _split_rows(rows: list[dict[str, Any]], train_episode_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if int(row["episode"]) < int(train_episode_count)]
    test_rows = [row for row in rows if int(row["episode"]) >= int(train_episode_count)]
    return train_rows, test_rows


def _flatten_rows(rows: list[dict[str, Any]], layer_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    x = np.concatenate([np.asarray(row[layer_name], dtype=np.float32) for row in rows], axis=0)
    y = np.concatenate([np.asarray(row["target_z"], dtype=np.float32) for row in rows], axis=0)
    return x, y


def _fit_linear_probe(train_rows: list[dict[str, Any]], layer_name: str, ridge_lambda: float) -> dict[str, Any]:
    x_train, y_train = _flatten_rows(train_rows, layer_name)
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError(f"No training samples available for layer {layer_name}")
    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = x_train.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std > 1.0e-6, feature_std, 1.0)
    x_norm = (x_train - feature_mean) / feature_std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)], axis=1)
    gram = x_aug.T @ x_aug
    reg = np.eye(x_aug.shape[1], dtype=np.float32) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ y_train
    try:
        weight = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        weight = np.linalg.pinv(gram + reg) @ rhs
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "weight": weight.astype(np.float32),
        "feature_dim": int(x_train.shape[1]),
        "train_user_count": int(x_train.shape[0]),
    }


def _predict_linear_probe(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    x_norm = (x_arr - model["feature_mean"]) / model["feature_std"]
    weight = np.asarray(model["weight"], dtype=np.float32)
    return x_norm @ weight[:-1] + weight[-1]


def _evaluate_rows(rows: list[dict[str, Any]], layer_name: str, model: dict[str, Any]) -> dict[str, Any]:
    valid_counts: list[int] = []
    spearman_values: list[float] = []
    top1_hit_values: list[float] = []
    pairwise_values: list[float] = []
    global_pred: list[float] = []
    global_target_z: list[float] = []

    for row in rows:
        pred = _predict_linear_probe(model, np.asarray(row[layer_name], dtype=np.float32))
        target = np.asarray(row["target"], dtype=np.float32)
        target_z = np.asarray(row["target_z"], dtype=np.float32)
        valid_count = int(row["valid_user_count"])
        valid_counts.append(valid_count)
        spearman_values.append(_safe_spearman_desc(pred.astype(np.float64), target.astype(np.float64)))
        top1_hit_values.append(float(int(np.argmax(pred)) == int(np.argmax(target))))
        pairwise_values.append(_pairwise_accuracy_desc(pred.astype(np.float64), target.astype(np.float64)))
        global_pred.extend([float(x) for x in pred.tolist()])
        global_target_z.extend([float(x) for x in target_z.tolist()])

    mse = 0.0
    corr = None
    r2 = None
    if global_pred and global_target_z:
        pred_np = np.asarray(global_pred, dtype=np.float64)
        target_np = np.asarray(global_target_z, dtype=np.float64)
        mse = float(np.mean((pred_np - target_np) ** 2))
        corr = _safe_corr(global_pred, global_target_z)
        denom = float(np.sum((target_np - float(np.mean(target_np))) ** 2))
        if denom > 1.0e-12:
            r2 = float(1.0 - np.sum((pred_np - target_np) ** 2) / denom)

    return {
        "state_count": int(len(rows)),
        "user_count": int(len(global_pred)),
        "valid_user_count": _summarize([float(x) for x in valid_counts]),
        "spearman_vs_queue_eta_prev": _summarize(spearman_values),
        "top1_hit_vs_queue_eta_prev": _summarize(top1_hit_values),
        "pairwise_acc_vs_queue_eta_prev": _summarize(pairwise_values),
        "global_mse_to_target_z": float(mse),
        "global_corr_to_target_z": corr,
        "global_r2_to_target_z": r2,
        "spearman_by_valid_user_count": _bucket_mean(valid_counts, spearman_values),
        "top1_hit_by_valid_user_count": _bucket_mean(valid_counts, top1_hit_values),
        "pairwise_acc_by_valid_user_count": _bucket_mean(valid_counts, pairwise_values),
    }


def diagnose_update(
    run_dir: Path,
    update: int,
    *,
    episodes: int,
    train_episode_count: int,
    episode_seed_base: int,
    deterministic: bool,
    ridge_lambda: float,
    device: torch.device,
) -> dict[str, Any]:
    collected = _collect_probe_rows(
        run_dir,
        int(update),
        episodes=int(episodes),
        episode_seed_base=int(episode_seed_base),
        deterministic=deterministic,
        device=device,
    )
    rows = collected["rows"]
    train_rows, test_rows = _split_rows(rows, int(train_episode_count))
    if not train_rows or not test_rows:
        raise ValueError(
            f"Need both train and test rows, got train={len(train_rows)} test={len(test_rows)}. "
            "Increase episodes or adjust train_episode_count."
        )

    layers: dict[str, Any] = {}
    best_layer = None
    best_spearman = None
    for layer_name in ("raw", "user0", "user1", "fused"):
        model = _fit_linear_probe(train_rows, layer_name, ridge_lambda=float(ridge_lambda))
        train_metrics = _evaluate_rows(train_rows, layer_name, model)
        test_metrics = _evaluate_rows(test_rows, layer_name, model)
        layers[layer_name] = {
            "feature_dim": int(model["feature_dim"]),
            "weight_norm": float(np.linalg.norm(model["weight"][:-1])),
            "train": train_metrics,
            "test": test_metrics,
        }
        test_spearman = float(test_metrics["spearman_vs_queue_eta_prev"]["mean"])
        if best_spearman is None or test_spearman > best_spearman:
            best_spearman = test_spearman
            best_layer = layer_name

    return {
        "update": int(update),
        "episodes": int(episodes),
        "train_episode_count": int(train_episode_count),
        "test_episode_count": int(episodes - train_episode_count),
        "episode_seed_base": int(episode_seed_base),
        "deterministic": bool(deterministic),
        "ridge_lambda": float(ridge_lambda),
        "loc_readout": str(collected["loc_readout"]),
        "state_counts": {
            "total_bw_states_seen": int(collected["total_states"]),
            "retained_states": int(len(rows)),
            "skipped_singleton": int(collected["skipped_singleton"]),
            "skipped_flat_target": int(collected["skipped_flat_target"]),
            "train_states": int(len(train_rows)),
            "test_states": int(len(test_rows)),
        },
        "best_layer_by_test_spearman": {
            "layer": str(best_layer),
            "spearman_mean": 0.0 if best_spearman is None else float(best_spearman),
        },
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100])
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--episode-seed-base", type=int, default=91000)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--out-name", type=str, default="structured_bw_linear_probe.json")
    args = parser.parse_args()

    train_episode_count = (
        int(args.train_episodes)
        if args.train_episodes is not None
        else max(1, min(int(args.episodes) - 1, int(round(int(args.episodes) * 0.6))))
    )
    if train_episode_count <= 0 or train_episode_count >= int(args.episodes):
        raise ValueError("train_episode_count must be between 1 and episodes - 1")

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "train_episode_count": int(train_episode_count),
        "episode_seed_base": int(args.episode_seed_base),
        "policy_mode": str(args.policy_mode),
        "ridge_lambda": float(args.ridge_lambda),
        "updates": {},
    }
    for update in args.updates:
        update_summary = diagnose_update(
            run_dir,
            int(update),
            episodes=int(args.episodes),
            train_episode_count=int(train_episode_count),
            episode_seed_base=int(args.episode_seed_base) + int(update) * 1000,
            deterministic=(args.policy_mode == "deterministic"),
            ridge_lambda=float(args.ridge_lambda),
            device=device,
        )
        summary["updates"][f"u{int(update):04d}"] = update_summary
        print(
            json.dumps(
                {
                    "update": int(update),
                    "loc_readout": update_summary["loc_readout"],
                    "best_layer": update_summary["best_layer_by_test_spearman"]["layer"],
                    "best_test_spearman": update_summary["best_layer_by_test_spearman"]["spearman_mean"],
                    "user1_test_spearman": update_summary["layers"]["user1"]["test"]["spearman_vs_queue_eta_prev"]["mean"],
                    "fused_test_spearman": update_summary["layers"]["fused"]["test"]["spearman_vs_queue_eta_prev"]["mean"],
                    "user1_test_top1": update_summary["layers"]["user1"]["test"]["top1_hit_vs_queue_eta_prev"]["mean"],
                    "fused_test_top1": update_summary["layers"]["fused"]["test"]["top1_hit_vs_queue_eta_prev"]["mean"],
                }
            )
        )

    out_path = run_dir / str(args.out_name)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
