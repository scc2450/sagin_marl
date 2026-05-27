from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _collect_panel_with_actor(
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
                panel_action = _det_bw_action_from_snapshot(actor, snapshot, device)
                rows.append(
                    {
                        "episode": int(episode_count),
                        "t": int(getattr(env, "t", 0)),
                        "snapshot_state": dict(snapshot_state or {}),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(panel_action)
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _make_learner(cfg, actor, device: torch.device, teacher_horizon: int) -> StructuredMAPPO:
    learner = StructuredMAPPO(
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
    learner.bw_clean_per_user_horizon = int(teacher_horizon)
    return learner


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


def _full_episode_return_from_snapshot(
    *,
    cfg,
    actor,
    device: torch.device,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
) -> float:
    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    try:
        probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
        total_reward = 0.0
        action = np.asarray(first_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        while True:
            step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
            total_reward += float(next(iter(step_result.rewards.values())))
            done = _done_from_step_result(step_result)
            if done:
                break
            probe_driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            z1 = probe_driver.run_accel_stage(accel_zero)
            z2 = probe_driver.run_sat_stage(_zero_sat_action(cfg))
            next_snapshot = probe_driver.build_bw_stage_snapshot(z2)
            action = _det_bw_action_from_snapshot(actor, next_snapshot, device).reshape(
                int(cfg.num_uav),
                int(cfg.users_obs_max),
            )
        return float(total_reward)
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()


def _analyze_checkpoint(
    *,
    cfg,
    actor_ckpt: str,
    device: torch.device,
    snapshot_states: list[dict[str, Any]],
    teacher_horizon: int,
) -> dict[str, Any]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device).eval()
    load_checkpoint_forgiving(actor, actor_ckpt, map_location=device, strict=True)
    learner = _make_learner(cfg, actor, device, teacher_horizon=int(teacher_horizon))
    try:
        ref_actions: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        probe_env = make_structured_env(cfg, mode="script")
        probe_driver = as_structured_driver(probe_env)
        try:
            for snapshot_state in snapshot_states:
                snapshot = probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
                ref_actions.append(_det_bw_action_from_snapshot(actor, snapshot, device))
                valid_masks.append(np.asarray(snapshot.bw_valid_mask, dtype=bool))
        finally:
            close_fn = getattr(probe_env, "close", None)
            if callable(close_fn):
                close_fn()

        ref_actions_np = np.stack(ref_actions, axis=0).astype(np.float32, copy=False)
        valid_masks_np = np.stack(valid_masks, axis=0).astype(bool, copy=False)

        target_actions_np, rho_np, utility_l1_np = learner._bw_clean_target_actions_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            ref_actions=ref_actions_np,
            valid_masks=valid_masks_np,
        )
        local_ref_returns = np.asarray(
            getattr(learner, "_bw_clean_last_ref_returns", np.zeros((len(snapshot_states),), dtype=np.float32)),
            dtype=np.float32,
        )
        local_target_returns = learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(target_actions_np[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                for idx in range(target_actions_np.shape[0])
            ],
        )
        local_target_returns_np = np.asarray(local_target_returns, dtype=np.float32)
        local_gain = np.asarray(local_target_returns_np - local_ref_returns, dtype=np.float32)

        full_ref_returns: list[float] = []
        full_target_returns: list[float] = []
        for idx, snapshot_state in enumerate(snapshot_states):
            full_ref_returns.append(
                _full_episode_return_from_snapshot(
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    snapshot_state=dict(snapshot_state or {}),
                    first_action=np.asarray(ref_actions_np[idx], dtype=np.float32),
                )
            )
            full_target_returns.append(
                _full_episode_return_from_snapshot(
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    snapshot_state=dict(snapshot_state or {}),
                    first_action=np.asarray(target_actions_np[idx], dtype=np.float32),
                )
            )
        full_ref_returns_np = np.asarray(full_ref_returns, dtype=np.float32)
        full_target_returns_np = np.asarray(full_target_returns, dtype=np.float32)
        full_gain = np.asarray(full_target_returns_np - full_ref_returns_np, dtype=np.float32)

        target_gap = _masked_l1(target_actions_np, ref_actions_np, valid_masks_np)
        improving_local = local_gain > 1.0e-6
        improving_full = full_gain > 1.0e-6

        return {
            "actor_checkpoint": os.path.abspath(actor_ckpt),
            "teacher_horizon": int(teacher_horizon),
            "target_gap": _safe_summary(target_gap),
            "rho": _safe_summary(np.asarray(rho_np, dtype=np.float64)),
            "utility_l1": _safe_summary(np.asarray(utility_l1_np, dtype=np.float64)),
            "local_gain": _safe_summary(np.asarray(local_gain, dtype=np.float64)),
            "full_episode_gain": _safe_summary(np.asarray(full_gain, dtype=np.float64)),
            "local_beats_ref_frac": float(np.mean(improving_local.astype(np.float32))) if improving_local.size > 0 else 0.0,
            "full_beats_ref_frac": float(np.mean(improving_full.astype(np.float32))) if improving_full.size > 0 else 0.0,
            "local_full_gain_corr": _safe_corr(local_gain, full_gain),
            "local_positive_full_nonpositive_frac": (
                float(np.mean((improving_local & ~improving_full).astype(np.float32)))
                if improving_local.size > 0
                else 0.0
            ),
            "full_positive_local_nonpositive_frac": (
                float(np.mean((improving_full & ~improving_local).astype(np.float32)))
                if improving_full.size > 0
                else 0.0
            ),
        }
    finally:
        learner._close_bw_clean_probe_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel_actor_checkpoint", required=True)
    parser.add_argument("--actor_checkpoints", nargs="+", required=True)
    parser.add_argument("--checkpoint_labels", nargs="*", default=None)
    parser.add_argument("--panel_states", type=int, default=32)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--teacher_horizon", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    panel_bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    panel_actor = panel_bundle.actor.to(device).eval()
    load_checkpoint_forgiving(panel_actor, args.panel_actor_checkpoint, map_location=device, strict=True)
    panel_rows, episode_count = _collect_panel_with_actor(
        cfg=cfg,
        actor=panel_actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in panel_rows]

    labels = list(args.checkpoint_labels) if args.checkpoint_labels else []
    if len(labels) != len(args.actor_checkpoints):
        labels = [Path(path).stem for path in args.actor_checkpoints]

    results: dict[str, Any] = {}
    for label, ckpt in zip(labels, args.actor_checkpoints):
        results[str(label)] = _analyze_checkpoint(
            cfg=cfg,
            actor_ckpt=str(ckpt),
            device=device,
            snapshot_states=snapshot_states,
            teacher_horizon=int(args.teacher_horizon),
        )

    payload = {
        "config": os.path.abspath(args.config),
        "panel_actor_checkpoint": os.path.abspath(args.panel_actor_checkpoint),
        "panel": {
            "states": int(len(panel_rows)),
            "episodes": int(episode_count),
            "seed_base": int(args.panel_seed),
            "teacher_horizon": int(args.teacher_horizon),
        },
        "results": results,
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
