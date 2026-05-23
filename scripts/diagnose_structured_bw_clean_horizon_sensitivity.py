from __future__ import annotations

import argparse
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

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _to_device_dataclass
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
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size != ya.size or xa.size < 2:
        return 0.0
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
        return 0.0
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


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
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _collect_panel(
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
                z1 = driver.run_accel_stage(accel_zero)
                z2 = driver.run_sat_stage(_zero_sat_action(cfg))
                snapshot = driver.build_bw_stage_snapshot(z2)
                snapshot_state = driver.export_bw_stage_state()
                ref_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(snapshot_state or {}),
                        "ref_action": np.asarray(ref_action, dtype=np.float32),
                        "valid_mask": valid_mask,
                    }
                )
                step_result = driver.execute_stage_bw_and_step(ref_action)
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _make_learner(cfg, actor, device: torch.device) -> StructuredMAPPO:
    return StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
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


def _masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if a_arr.ndim == 1:
        a_arr = a_arr.reshape(1, -1)
    else:
        a_arr = a_arr.reshape(a_arr.shape[0], -1)
    if b_arr.ndim == 1:
        b_arr = b_arr.reshape(1, -1)
    else:
        b_arr = b_arr.reshape(b_arr.shape[0], -1)
    if mask_arr.ndim == 1:
        mask_arr = mask_arr.reshape(1, -1)
    else:
        mask_arr = mask_arr.reshape(mask_arr.shape[0], -1)
    return np.sum(
        np.abs(a_arr - b_arr) * mask_arr,
        axis=-1,
        dtype=np.float64,
    )


def _analyze_horizons(
    *,
    learner: StructuredMAPPO,
    snapshot_states: list[dict[str, Any]],
    ref_actions: np.ndarray,
    valid_masks: np.ndarray,
    horizons: list[int],
) -> dict[str, Any]:
    eps = 1.0e-8
    target_by_h: dict[int, np.ndarray] = {}
    build_summary: dict[str, Any] = {}

    for horizon in horizons:
        learner.bw_clean_per_user_horizon = int(horizon)
        target_actions, rho_values, utility_l1_values = learner._bw_clean_target_actions_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            ref_actions=np.asarray(ref_actions, dtype=np.float32),
            valid_masks=np.asarray(valid_masks, dtype=bool),
        )
        ref_returns = np.asarray(getattr(learner, "_bw_clean_last_ref_returns", np.zeros((len(snapshot_states),), dtype=np.float32)))
        target_by_h[int(horizon)] = np.asarray(target_actions, dtype=np.float32)
        rho_values = np.asarray(rho_values, dtype=np.float32)
        utility_l1_values = np.asarray(utility_l1_values, dtype=np.float32)
        ref_returns = np.asarray(ref_returns, dtype=np.float32)

        target_gap = _masked_l1(target_actions, ref_actions, valid_masks)
        active = target_gap > eps
        build_summary[str(int(horizon))] = {
            "active_target_frac": float(np.mean(active.astype(np.float32))) if active.size > 0 else 0.0,
            "target_gap": _safe_summary(target_gap),
            "rho": _safe_summary(np.asarray(rho_values, dtype=np.float64)),
            "utility_l1": _safe_summary(np.asarray(utility_l1_values, dtype=np.float64)),
            "ref_return_build_h": _safe_summary(np.asarray(ref_returns, dtype=np.float64)),
        }

    common_eval: dict[str, Any] = {}
    gain_arrays: dict[tuple[int, int], np.ndarray] = {}
    for eval_h in horizons:
        learner.bw_clean_per_user_horizon = int(eval_h)
        ref_eval_returns = learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(ref_actions[idx], dtype=np.float32).reshape(int(learner.cfg.num_uav), int(learner.cfg.users_obs_max))
                for idx in range(ref_actions.shape[0])
            ],
        )
        ref_eval_returns_np = np.asarray(ref_eval_returns, dtype=np.float32)
        eval_rows: dict[str, Any] = {}
        for build_h in horizons:
            target_actions = target_by_h[int(build_h)]
            target_returns = learner._bw_clean_rollout_returns_parallel(
                snapshot_states=[dict(state or {}) for state in snapshot_states],
                first_actions=[
                    np.asarray(target_actions[idx], dtype=np.float32).reshape(int(learner.cfg.num_uav), int(learner.cfg.users_obs_max))
                    for idx in range(target_actions.shape[0])
                ],
            )
            target_returns_np = np.asarray(target_returns, dtype=np.float32)
            gain = np.asarray(target_returns_np - ref_eval_returns_np, dtype=np.float32)
            gain_arrays[(int(eval_h), int(build_h))] = gain.copy()
            eval_rows[str(int(build_h))] = {
                "gain": _safe_summary(gain),
                "beats_ref_frac": float(np.mean((gain > 1.0e-6).astype(np.float32))) if gain.size > 0 else 0.0,
            }
        common_eval[str(int(eval_h))] = eval_rows

    pairwise_targets: dict[str, Any] = {}
    for idx, left_h in enumerate(horizons):
        for right_h in horizons[idx + 1 :]:
            left_target = target_by_h[int(left_h)]
            right_target = target_by_h[int(right_h)]
            target_l1_diff = _masked_l1(left_target, right_target, valid_masks)
            left_gap = _masked_l1(left_target, ref_actions, valid_masks)
            right_gap = _masked_l1(right_target, ref_actions, valid_masks)
            denom = 0.5 * (left_gap + right_gap)
            relative_diff = np.where(denom > eps, target_l1_diff / denom, 0.0)
            pairwise_targets[f"{int(left_h)}_vs_{int(right_h)}"] = {
                "target_l1_diff": _safe_summary(target_l1_diff),
                "relative_target_diff_to_ref_gap": _safe_summary(relative_diff),
                "self_eval_gain_corr": _safe_corr(
                    gain_arrays[(int(left_h), int(left_h))],
                    gain_arrays[(int(right_h), int(right_h))],
                ),
            }

    cross_eval_advantage: dict[str, Any] = {}
    for eval_h in horizons:
        eval_rows = common_eval[str(int(eval_h))]
        for idx, left_h in enumerate(horizons):
            for right_h in horizons[idx + 1 :]:
                left_gain = gain_arrays[(int(eval_h), int(left_h))]
                right_gain = gain_arrays[(int(eval_h), int(right_h))]
                left_gain_mean = float(eval_rows[str(int(left_h))]["gain"]["mean"])
                right_gain_mean = float(eval_rows[str(int(right_h))]["gain"]["mean"])
                cross_eval_advantage[f"eval{int(eval_h)}_{int(left_h)}_vs_{int(right_h)}"] = {
                    "mean_gain_delta": float(right_gain_mean - left_gain_mean),
                    "left_beats_ref_frac": float(eval_rows[str(int(left_h))]["beats_ref_frac"]),
                    "right_beats_ref_frac": float(eval_rows[str(int(right_h))]["beats_ref_frac"]),
                    "per_sample_gain_corr": _safe_corr(left_gain, right_gain),
                    "right_beats_left_frac": (
                        float(np.mean((right_gain > left_gain + 1.0e-6).astype(np.float32)))
                        if right_gain.size > 0
                        else 0.0
                    ),
                }

    return {
        "build_summary": build_summary,
        "common_eval": common_eval,
        "pairwise_targets": pairwise_targets,
        "cross_eval_advantage": cross_eval_advantage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=48)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    horizons = sorted({max(int(h), 1) for h in list(args.horizons)})
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device).eval()
    load_checkpoint_forgiving(actor, args.actor_checkpoint, map_location=device, strict=True)

    panel_rows, episode_count = _collect_panel(
        cfg=cfg,
        actor=actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in panel_rows]
    ref_actions = np.stack([np.asarray(row["ref_action"], dtype=np.float32) for row in panel_rows], axis=0)
    valid_masks = np.stack([np.asarray(row["valid_mask"], dtype=bool) for row in panel_rows], axis=0)

    learner = _make_learner(cfg, actor, device)
    try:
        analysis = _analyze_horizons(
            learner=learner,
            snapshot_states=snapshot_states,
            ref_actions=ref_actions,
            valid_masks=valid_masks,
            horizons=horizons,
        )
    finally:
        learner._close_bw_clean_probe_group()

    payload = {
        "config": os.path.abspath(args.config),
        "actor_checkpoint": os.path.abspath(args.actor_checkpoint),
        "panel": {
            "states": int(len(panel_rows)),
            "episodes": int(episode_count),
            "seed_base": int(args.panel_seed),
            "horizons": [int(h) for h in horizons],
        },
        "analysis": analysis,
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
