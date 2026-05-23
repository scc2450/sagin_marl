from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_actor import _attend
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass, _to_cpu_tensor, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.count_nonzero(finite)) <= 1:
        return 0.0
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if float(np.std(x_arr)) <= 1.0e-12 or float(np.std(y_arr)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _done_from_step_result(step_result: Any) -> bool:
    return bool(
        list(step_result.terminations.values())[0]
        or list(step_result.truncations.values())[0]
    )


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _det_bw_action_from_snapshot(actor, snapshot: Any, device: torch.device) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=True)
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def _collect_panel_rows(
    *,
    cfg,
    actor,
    device: torch.device,
    panel_states: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    try:
        while len(rows) < int(panel_states):
            env.reset(seed=int(seed_base) + int(episode_count))
            done = False
            while not done and len(rows) < int(panel_states):
                driver.begin_step()
                accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
                driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(z2)
                ref_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(driver.export_bw_stage_state() or {}),
                        "local_state": build_local_bw_states_from_snapshot(snapshot)[0],
                        "ref_action": np.asarray(ref_action, dtype=np.float32).reshape(-1),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(
                    np.asarray(ref_action, dtype=np.float32).reshape(
                        int(cfg.num_uav),
                        int(cfg.users_obs_max),
                    )
                )
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _make_learner(cfg, actor_ckpt: str, device: torch.device) -> tuple[Any, StructuredMAPPO]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device)
    load_checkpoint_forgiving(actor, actor_ckpt, map_location=device, strict=True)
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=int(getattr(cfg, "ppo_epochs", 1) or 1),
        num_mini_batch=int(getattr(cfg, "num_mini_batch", 1) or 1),
        actor_optimizer=torch.optim.Adam(
            actor.parameters(),
            lr=float(getattr(cfg, "actor_lr", 1.0e-3) or 1.0e-3),
        ),
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    actor.eval()
    return actor, learner


def _compute_target_beats_flags(
    *,
    learner: StructuredMAPPO,
    cfg,
    snapshot_states: list[dict[str, Any]],
    ref_actions: np.ndarray,
    target_actions: np.ndarray,
    valid_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = int(target_actions.shape[0])
    valid_mask_f = np.asarray(valid_masks, dtype=np.float32)
    target_gap = np.sum(
        np.abs(np.asarray(target_actions, dtype=np.float32) - np.asarray(ref_actions, dtype=np.float32)) * valid_mask_f,
        axis=-1,
        dtype=np.float64,
    )
    target_beats_flags = np.zeros((num_samples,), dtype=np.float32)
    active_target_idx = np.flatnonzero(target_gap > 1.0e-8)
    if active_target_idx.size <= 0:
        return target_beats_flags, target_gap.astype(np.float32, copy=False)
    ref_returns_np = getattr(learner, "_bw_clean_last_ref_returns", None)
    if ref_returns_np is None or int(len(ref_returns_np)) != num_samples:
        ref_returns_np = np.full((num_samples,), np.nan, dtype=np.float32)
    else:
        ref_returns_np = np.asarray(ref_returns_np, dtype=np.float32).copy()
    missing_ref_mask = ~np.isfinite(ref_returns_np[active_target_idx])
    if bool(np.any(missing_ref_mask)):
        missing_idx = active_target_idx[missing_ref_mask]
        missing_ref_returns = learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(snapshot_states[int(idx)] or {}) for idx in missing_idx.tolist()],
            first_actions=[
                np.asarray(ref_actions[int(idx)], dtype=np.float32).reshape(
                    int(cfg.num_uav),
                    int(cfg.users_obs_max),
                )
                for idx in missing_idx.tolist()
            ],
        )
        ref_returns_np[missing_idx] = np.asarray(missing_ref_returns, dtype=np.float32)
    target_returns = learner._bw_clean_rollout_returns_parallel(
        snapshot_states=[dict(snapshot_states[int(idx)] or {}) for idx in active_target_idx.tolist()],
        first_actions=[
            np.asarray(target_actions[int(idx)], dtype=np.float32).reshape(
                int(cfg.num_uav),
                int(cfg.users_obs_max),
            )
            for idx in active_target_idx.tolist()
        ],
    )
    target_beats_flags[active_target_idx] = (
        np.asarray(target_returns, dtype=np.float32) > (ref_returns_np[active_target_idx] + 1.0e-6)
    ).astype(np.float32, copy=False)
    return target_beats_flags, target_gap.astype(np.float32, copy=False)


def _flatten_local_state_batch(local_batch: Any) -> np.ndarray:
    if not is_dataclass(local_batch):
        raise TypeError("_flatten_local_state_batch expects a dataclass batch.")
    flat_parts: list[np.ndarray] = []
    for field in fields(local_batch):
        value = getattr(local_batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        tensor = value.detach().to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        flat_parts.append(_to_cpu_tensor(tensor).numpy().reshape(tensor.shape[0], -1))
    if not flat_parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(flat_parts, axis=-1).astype(np.float32, copy=False)


def _flatten_masked_rep_batch(rep: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    rep_t = rep.detach().to(dtype=torch.float32)
    mask_t = mask.detach().to(dtype=torch.float32).unsqueeze(-1)
    rep_t = rep_t * mask_t
    return _to_cpu_tensor(rep_t).numpy().reshape(rep_t.shape[0], -1).astype(np.float32, copy=False)


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std = np.where(std > 1.0e-6, std, 1.0)
    return ((values - mean) / std).astype(np.float32, copy=False)


def _extract_bw_representations(actor: Any, local_batch: Any) -> dict[str, torch.Tensor]:
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy.")
    if str(getattr(bw_policy, "score_model", "neural")).strip().lower() != "neural":
        raise RuntimeError("Representation ambiguity diagnostic requires neural BW score_model.")

    valid_mask = ((local_batch.user_mask > 0.5) & (local_batch.bw_valid_mask > 0.5))
    ego_0 = bw_policy.ego_encoder(bw_policy.ego_input_norm(local_batch.ego_uav_after_sat))
    sat_0 = bw_policy.sat_encoder(
        bw_policy.sat_input_norm(torch.cat([local_batch.sat_nodes, local_batch.sat_edges], dim=-1))
    )
    user_0 = bw_policy.user_encoder(
        bw_policy.user_input_norm(torch.cat([local_batch.user_nodes, local_batch.user_edges], dim=-1))
    )

    sat_ctx_1 = _attend(ego_0, sat_0, local_batch.sat_mask)
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
    sat_ctx_2 = _attend(query_2, sat_1, local_batch.sat_mask)
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
    if str(getattr(bw_policy, "loc_readout", "fused")).strip().lower() == "user_only":
        loc_src = user_1
    elif str(getattr(bw_policy, "loc_readout", "fused")).strip().lower() == "user0":
        loc_src = user_0
    else:
        loc_src = fused
    return {
        "user_0": user_0,
        "user_1": user_1,
        "fused": fused,
        "loc_src": loc_src,
        "valid_mask": valid_mask,
    }


def _pair_metrics(
    state_vectors_z: np.ndarray,
    target_actions: np.ndarray,
    ref_actions: np.ndarray,
    valid_masks: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    num_samples = int(state_vectors_z.shape[0])
    if num_samples <= 1:
        return {
            "pair_count": 0,
            "state_target_corr": 0.0,
            "state_ref_corr": 0.0,
            "state_dist": _safe_summary(np.zeros((0,), dtype=np.float32)),
            "target_l1": _safe_summary(np.zeros((0,), dtype=np.float32)),
            "ref_l1": _safe_summary(np.zeros((0,), dtype=np.float32)),
            "near_state_far_target_frac": 0.0,
            "nearest_neighbor_target_l1": _safe_summary(np.zeros((0,), dtype=np.float32)),
            "nearest_neighbor_ref_l1": _safe_summary(np.zeros((0,), dtype=np.float32)),
            "examples": [],
        }
    state_vectors_t = torch.as_tensor(state_vectors_z, dtype=torch.float32)
    pairwise_state = torch.cdist(state_vectors_t, state_vectors_t, p=2).cpu().numpy().astype(np.float32, copy=False)
    state_dist_values: list[float] = []
    target_diff_values: list[float] = []
    ref_diff_values: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    nearest_target_l1: list[float] = []
    nearest_ref_l1: list[float] = []
    for idx in range(num_samples):
        state_row = pairwise_state[idx].copy()
        state_row[idx] = np.inf
        nn_idx = int(np.argmin(state_row))
        mask_union_nn = np.asarray(valid_masks[idx] | valid_masks[nn_idx], dtype=np.float32)
        nn_target_l1 = float(np.sum(np.abs(target_actions[idx] - target_actions[nn_idx]) * mask_union_nn, dtype=np.float64))
        nn_ref_l1 = float(np.sum(np.abs(ref_actions[idx] - ref_actions[nn_idx]) * mask_union_nn, dtype=np.float64))
        nearest_target_l1.append(nn_target_l1)
        nearest_ref_l1.append(nn_ref_l1)
    for left in range(num_samples):
        for right in range(left + 1, num_samples):
            mask_union = np.asarray(valid_masks[left] | valid_masks[right], dtype=np.float32)
            state_dist = float(pairwise_state[left, right])
            target_l1 = float(np.sum(np.abs(target_actions[left] - target_actions[right]) * mask_union, dtype=np.float64))
            ref_l1 = float(np.sum(np.abs(ref_actions[left] - ref_actions[right]) * mask_union, dtype=np.float64))
            state_dist_values.append(state_dist)
            target_diff_values.append(target_l1)
            ref_diff_values.append(ref_l1)
            pair_rows.append(
                {
                    "left": int(left),
                    "right": int(right),
                    "state_dist": float(state_dist),
                    "target_l1": float(target_l1),
                    "ref_l1": float(ref_l1),
                }
            )
    state_dist_np = np.asarray(state_dist_values, dtype=np.float32)
    target_diff_np = np.asarray(target_diff_values, dtype=np.float32)
    ref_diff_np = np.asarray(ref_diff_values, dtype=np.float32)
    near_state_thresh = float(np.percentile(state_dist_np, 10.0)) if state_dist_np.size > 0 else 0.0
    far_target_thresh = float(np.percentile(target_diff_np, 90.0)) if target_diff_np.size > 0 else 0.0
    candidate_examples = [
        pair
        for pair in pair_rows
        if pair["state_dist"] <= near_state_thresh and pair["target_l1"] >= far_target_thresh
    ]
    candidate_examples.sort(key=lambda row: (-float(row["target_l1"]), float(row["state_dist"])))
    examples: list[dict[str, Any]] = []
    for pair in candidate_examples[:8]:
        left = int(pair["left"])
        right = int(pair["right"])
        examples.append(
            {
                "left": {
                    "index": int(left),
                    "episode": int(rows[left]["episode"]),
                    "t": int(rows[left]["t"]),
                },
                "right": {
                    "index": int(right),
                    "episode": int(rows[right]["episode"]),
                    "t": int(rows[right]["t"]),
                },
                "state_dist": float(pair["state_dist"]),
                "target_l1": float(pair["target_l1"]),
                "ref_l1": float(pair["ref_l1"]),
            }
        )
    near_state_far_target_frac = float(
        np.mean(
            (
                (state_dist_np <= near_state_thresh)
                & (target_diff_np >= far_target_thresh)
            ).astype(np.float32)
        )
    ) if state_dist_np.size > 0 else 0.0
    return {
        "pair_count": int(len(pair_rows)),
        "state_target_corr": float(_safe_corr(state_dist_np, target_diff_np)),
        "state_ref_corr": float(_safe_corr(state_dist_np, ref_diff_np)),
        "state_dist": _safe_summary(state_dist_np),
        "target_l1": _safe_summary(target_diff_np),
        "ref_l1": _safe_summary(ref_diff_np),
        "near_state_threshold_p10": float(near_state_thresh),
        "far_target_threshold_p90": float(far_target_thresh),
        "near_state_far_target_frac": float(near_state_far_target_frac),
        "nearest_neighbor_target_l1": _safe_summary(np.asarray(nearest_target_l1, dtype=np.float32)),
        "nearest_neighbor_ref_l1": _safe_summary(np.asarray(nearest_ref_l1, dtype=np.float32)),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=64)
    parser.add_argument("--panel_seed", type=int, default=54000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    cfg = load_config(str(args.config))
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    actor, learner = _make_learner(cfg, str(args.actor_checkpoint), device)
    rows, episode_count = _collect_panel_rows(
        cfg=cfg,
        actor=actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )
    local_states = [row["local_state"] for row in rows]
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in rows]
    local_batch = _collate_dataclass(local_states, learner.device)
    flat_state_vectors_z = _zscore_rows(_flatten_local_state_batch(local_batch))
    rep_tensors = _extract_bw_representations(actor, local_batch)
    valid_mask = rep_tensors.pop("valid_mask")
    rep_vectors_z = {
        "raw_local_state": flat_state_vectors_z,
    }
    for rep_name, rep_value in rep_tensors.items():
        rep_vectors_z[str(rep_name)] = _zscore_rows(_flatten_masked_rep_batch(rep_value, valid_mask))

    with torch.inference_mode():
        ref_out = learner.actor.act_bw(local_batch, deterministic=True)
    ref_actions = _to_cpu_tensor(ref_out.action).numpy().astype(np.float32, copy=False)
    valid_masks_np = _to_cpu_tensor(valid_mask).numpy().astype(bool, copy=False)

    target_actions_np, rho_values_np, utility_l1_values_np = learner._bw_clean_target_actions_parallel(
        snapshot_states=[dict(state or {}) for state in snapshot_states],
        ref_actions=ref_actions,
        valid_masks=valid_masks_np,
    )
    target_beats_flags, target_gap_np = _compute_target_beats_flags(
        learner=learner,
        cfg=cfg,
        snapshot_states=snapshot_states,
        ref_actions=ref_actions,
        target_actions=np.asarray(target_actions_np, dtype=np.float32),
        valid_masks=valid_masks_np,
    )

    improving_idx = np.flatnonzero(target_beats_flags > 0.5)
    if improving_idx.size >= 2:
        used_idx = improving_idx
        subset_name = "improving_only"
    else:
        used_idx = np.arange(len(rows), dtype=np.int64)
        subset_name = "all_rows_fallback"
    subset_rows = [rows[int(idx)] for idx in used_idx.tolist()]
    pair_metrics_by_rep: dict[str, Any] = {}
    for rep_name, rep_vectors in rep_vectors_z.items():
        pair_metrics_by_rep[str(rep_name)] = _pair_metrics(
            state_vectors_z=np.asarray(rep_vectors[used_idx], dtype=np.float32),
            target_actions=np.asarray(target_actions_np[used_idx], dtype=np.float32),
            ref_actions=np.asarray(ref_actions[used_idx], dtype=np.float32),
            valid_masks=np.asarray(valid_masks_np[used_idx], dtype=bool),
            rows=subset_rows,
        )
    payload = {
        "config": str(args.config),
        "actor_checkpoint": str(args.actor_checkpoint),
        "device": str(device),
        "loc_readout": str(getattr(getattr(actor, "bw_policy", None), "loc_readout", "unknown")),
        "collection": {
            "panel_states_requested": int(args.panel_states),
            "panel_states_collected": int(len(rows)),
            "panel_seed": int(args.panel_seed),
            "episodes_spanned": int(episode_count),
        },
        "target_stats": {
            "clean_target_beats_ref_frac": float(np.mean(target_beats_flags, dtype=np.float64)) if len(rows) > 0 else 0.0,
            "target_gap": _safe_summary(np.asarray(target_gap_np, dtype=np.float32)),
            "rho": _safe_summary(np.asarray(rho_values_np, dtype=np.float32)),
            "utility_l1": _safe_summary(np.asarray(utility_l1_values_np, dtype=np.float32)),
        },
        "subset_used": {
            "name": str(subset_name),
            "rows": int(len(used_idx)),
        },
        "pair_metrics_by_rep": pair_metrics_by_rep,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(json_text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text, encoding="utf-8")


if __name__ == "__main__":
    main()
