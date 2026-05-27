from __future__ import annotations

import argparse
import copy
import json
import os
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
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _index_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving

from scripts.diagnostics.diagnose.diagnose_structured_bw_fixed_teacher_fit import (
    _collect_panel_bank,
    _configure_train_scope,
    _det_bw_action_from_snapshot,
    _evaluate_panel,
    _full_episode_return_from_snapshot,
    _make_learner,
    _masked_l1,
    _offline_fit_one_step,
    _safe_summary,
    _set_all_seeds,
)
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _build_actor(cfg, actor_checkpoint: str, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device).eval()
    strict_load = str(getattr(cfg, "structured_bw_loc_readout", "fused") or "fused").strip().lower() not in {
        "user0_residual_fused",
        "fused_moe2",
        "fused_ctx_dot",
    }
    load_checkpoint_forgiving(actor, actor_checkpoint, map_location=device, strict=bool(strict_load))
    return actor


def _normalize_bw_action(action: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32)
    valid_arr = np.asarray(valid_mask, dtype=bool)
    out = np.zeros_like(action_arr, dtype=np.float32)
    if action_arr.ndim == 1:
        action_arr = action_arr.reshape(1, -1)
        valid_arr = valid_arr.reshape(1, -1)
        out = out.reshape(1, -1)
    for agent_idx in range(int(action_arr.shape[0])):
        valid = valid_arr[agent_idx]
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            continue
        row = np.clip(np.asarray(action_arr[agent_idx], dtype=np.float32)[valid], 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1.0e-12:
            out[agent_idx, valid] = 1.0 / float(valid_count)
        else:
            out[agent_idx, valid] = row / row_sum
    return out.reshape(np.asarray(action, dtype=np.float32).shape)


def _interp_bw_action(current_action: np.ndarray, target_action: np.ndarray, valid_mask: np.ndarray, alpha: float) -> np.ndarray:
    mixed = (1.0 - float(alpha)) * np.asarray(current_action, dtype=np.float32) + float(alpha) * np.asarray(target_action, dtype=np.float32)
    mixed = np.where(np.asarray(valid_mask, dtype=bool), mixed, 0.0)
    return _normalize_bw_action(mixed, valid_mask)


def _capture_user0_offline_states(
    *,
    cfg,
    actor_checkpoint: str,
    device: torch.device,
    snapshot_states: list[dict[str, Any]],
    local_states: list[Any],
    target_actions_np: np.ndarray,
    valid_masks: np.ndarray,
    improving_mask: np.ndarray,
    actor_lr: float,
    teacher_horizon: int,
    capture_steps: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    user0_cfg = copy.deepcopy(cfg)
    setattr(user0_cfg, "structured_bw_loc_readout", "user0")
    actor = _build_actor(user0_cfg, actor_checkpoint, device)
    _configure_train_scope(actor, "loc_head")
    learner = _make_learner(user0_cfg, actor, device, float(actor_lr))
    learner.bw_clean_per_user_horizon = int(teacher_horizon)

    local_batch = _collate_dataclass(local_states, device)
    train_idx_np = np.flatnonzero(np.asarray(improving_mask, dtype=np.bool_))
    train_idx_t = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    local_batch_train = _index_dataclass(local_batch, train_idx_t)
    target_actions_train_np = np.asarray(target_actions_np[train_idx_np], dtype=np.float32)
    valid_masks_train_np = np.asarray(valid_masks[train_idx_np], dtype=bool)
    target_actions_t = torch.as_tensor(target_actions_train_np, dtype=torch.float32, device=device)
    valid_masks_train = torch.as_tensor(valid_masks_train_np, dtype=torch.float32, device=device) > 0.5

    captured: dict[int, dict[str, torch.Tensor]] = {}
    capture_set = {int(step) for step in capture_steps if int(step) >= 0}
    if 0 in capture_set:
        captured[0] = {k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()}
    for step in range(1, max(capture_set) + 1):
        _offline_fit_one_step(
            learner=learner,
            local_batch_train=local_batch_train,
            target_actions=target_actions_t,
            valid_masks_train=valid_masks_train,
        )
        if step in capture_set:
            captured[step] = {k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()}
    return captured


def _evaluate_interpolation_for_actor(
    *,
    cfg,
    actor_state: dict[str, torch.Tensor],
    actor_checkpoint: str,
    device: torch.device,
    snapshot_states: list[dict[str, Any]],
    fixed_target_actions: np.ndarray,
    valid_masks: np.ndarray,
    alphas: list[float],
) -> dict[str, Any]:
    user0_cfg = copy.deepcopy(cfg)
    setattr(user0_cfg, "structured_bw_loc_readout", "user0")
    actor = _build_actor(user0_cfg, actor_checkpoint, device)
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device).eval()

    per_alpha_returns: dict[str, list[float]] = {f"{float(alpha):.2f}": [] for alpha in alphas}
    per_alpha_l1_to_target: dict[str, list[float]] = {f"{float(alpha):.2f}": [] for alpha in alphas}
    per_alpha_l1_to_current: dict[str, list[float]] = {f"{float(alpha):.2f}": [] for alpha in alphas}
    per_state_rows: list[dict[str, Any]] = []

    for idx, snapshot_state in enumerate(snapshot_states):
        probe_actor = _build_actor(user0_cfg, actor_checkpoint, device)
        probe_actor.load_state_dict(actor_state, strict=True)
        probe_actor.to(device).eval()
        probe_env = make_structured_env(cfg, mode="script")
        probe_driver = as_structured_driver(probe_env)
        try:
            snapshot = probe_driver.load_bw_stage_state(dict(snapshot_state or {}))
            current_action = _det_bw_action_from_snapshot(probe_actor, snapshot, device)
        finally:
            close_fn = getattr(probe_env, "close", None)
            if callable(close_fn):
                close_fn()
        current_action_2d = np.asarray(current_action, dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        target_action_2d = np.asarray(fixed_target_actions[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        valid_mask_2d = np.asarray(valid_masks[idx], dtype=bool).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
        alpha_rows: list[dict[str, float]] = []
        for alpha in alphas:
            first_action = _interp_bw_action(current_action_2d, target_action_2d, valid_mask_2d, float(alpha))
            selected_return = _full_episode_return_from_snapshot(
                cfg=cfg,
                actor=probe_actor,
                device=device,
                snapshot_state=dict(snapshot_state or {}),
                first_action=first_action,
            )
            alpha_key = f"{float(alpha):.2f}"
            l1_to_target = float(_masked_l1(first_action, target_action_2d, valid_mask_2d)[0])
            l1_to_current = float(_masked_l1(first_action, current_action_2d, valid_mask_2d)[0])
            per_alpha_returns[alpha_key].append(float(selected_return))
            per_alpha_l1_to_target[alpha_key].append(l1_to_target)
            per_alpha_l1_to_current[alpha_key].append(l1_to_current)
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "selected_return": float(selected_return),
                    "l1_to_target": l1_to_target,
                    "l1_to_current": l1_to_current,
                }
            )
        per_state_rows.append({"state_index": int(idx), "alpha_rows": alpha_rows})

    curve = {
        alpha_key: {
            "selected_return": _safe_summary(per_alpha_returns[alpha_key]),
            "l1_to_target": _safe_summary(per_alpha_l1_to_target[alpha_key]),
            "l1_to_current": _safe_summary(per_alpha_l1_to_current[alpha_key]),
        }
        for alpha_key in per_alpha_returns
    }
    return {
        "alphas": [float(alpha) for alpha in alphas],
        "alpha_curve": curve,
        "state_rows": per_state_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=8)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--teacher_horizon", type=int, default=5)
    parser.add_argument("--offline_steps", type=int, default=20)
    parser.add_argument("--capture_steps", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--actor_lr", type=float, default=1.0e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setattr(cfg, "structured_bw_loc_readout", "fused")
    setattr(cfg, "bw_clean_per_user_loss", "huber")
    setattr(cfg, "bw_clean_grad_aggregation", "mean")
    device = torch.device(args.device)
    _set_all_seeds(int(getattr(cfg, "seed", 0) or 0))

    base_actor = _build_actor(cfg, args.actor_checkpoint, device)
    panel_rows, episode_count = _collect_panel_bank(
        cfg=cfg,
        actor=base_actor,
        device=device,
        panel_states=int(args.panel_states),
        seed_base=int(args.panel_seed),
    )
    snapshot_states = [dict(row["snapshot_state"] or {}) for row in panel_rows]
    local_states = [row["local_state"] for row in panel_rows]
    valid_masks = np.stack([np.asarray(row["valid_mask"], dtype=bool) for row in panel_rows], axis=0)
    fixed_ref_actions = np.stack([np.asarray(row["ref_action"], dtype=np.float32) for row in panel_rows], axis=0)

    teacher_learner = _make_learner(cfg, base_actor, device, float(args.actor_lr))
    teacher_learner.bw_clean_per_user_horizon = int(args.teacher_horizon)
    target_actions_np, rho_np, utility_np = teacher_learner._bw_clean_target_actions_parallel(
        snapshot_states=[dict(state or {}) for state in snapshot_states],
        ref_actions=fixed_ref_actions,
        valid_masks=valid_masks,
    )
    ref_returns_np = np.asarray(
        getattr(teacher_learner, "_bw_clean_last_ref_returns", np.zeros((len(snapshot_states),), dtype=np.float32)),
        dtype=np.float32,
    )
    target_returns_np = np.asarray(
        teacher_learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(target_actions_np[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                for idx in range(target_actions_np.shape[0])
            ],
        ),
        dtype=np.float32,
    )
    improving_mask = target_returns_np > (ref_returns_np + 1.0e-6)

    captured = _capture_user0_offline_states(
        cfg=cfg,
        actor_checkpoint=args.actor_checkpoint,
        device=device,
        snapshot_states=snapshot_states,
        local_states=local_states,
        target_actions_np=target_actions_np,
        valid_masks=valid_masks,
        improving_mask=improving_mask,
        actor_lr=float(args.actor_lr),
        teacher_horizon=int(args.teacher_horizon),
        capture_steps=[int(step) for step in args.capture_steps],
    )

    steps_summary: dict[str, Any] = {}
    for step in [int(step) for step in args.capture_steps]:
        steps_summary[f"step_{int(step)}"] = _evaluate_interpolation_for_actor(
            cfg=cfg,
            actor_state=captured[int(step)],
            actor_checkpoint=args.actor_checkpoint,
            device=device,
            snapshot_states=snapshot_states,
            fixed_target_actions=target_actions_np,
            valid_masks=valid_masks,
            alphas=[float(alpha) for alpha in args.alphas],
        )

    result = {
        "config": str(Path(args.config).resolve()),
        "actor_checkpoint": str(Path(args.actor_checkpoint).resolve()),
        "panel": {
            "states": int(len(snapshot_states)),
            "episodes": int(episode_count),
            "seed_base": int(args.panel_seed),
        },
        "teacher_horizon": int(args.teacher_horizon),
        "offline_steps": int(args.offline_steps),
        "capture_steps": [int(step) for step in args.capture_steps],
        "fixed_bank": {
            "improving_frac": float(np.mean(improving_mask.astype(np.float32))) if improving_mask.size > 0 else 0.0,
            "train_samples": int(np.sum(improving_mask.astype(np.int32))),
            "ref_return": _safe_summary(ref_returns_np),
            "target_return": _safe_summary(target_returns_np),
            "target_minus_ref": _safe_summary(np.asarray(target_returns_np - ref_returns_np, dtype=np.float32)),
            "rho": _safe_summary(np.asarray(rho_np, dtype=np.float32)),
            "utility_l1": _safe_summary(np.asarray(utility_np, dtype=np.float32)),
        },
        "steps": steps_summary,
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text, encoding="utf-8")


if __name__ == "__main__":
    main()
