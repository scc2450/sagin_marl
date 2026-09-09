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
from sagin_marl.rl.structured_mappo import _collate_dataclass

from scripts.diagnostics.diagnose.diagnose_structured_bw_fixed_target_interpolation import _build_actor, _capture_user0_offline_states
from scripts.diagnostics.diagnose.diagnose_structured_bw_fixed_teacher_fit import (
    _collect_panel_bank,
    _make_learner,
    _masked_l1,
    _safe_summary,
    _set_all_seeds,
)


def _build_user0_actor_with_state(
    *,
    cfg,
    actor_checkpoint: str,
    actor_state: dict[str, torch.Tensor],
    device: torch.device,
):
    user0_cfg = copy.deepcopy(cfg)
    setattr(user0_cfg, "structured_bw_loc_readout", "user0")
    actor = _build_actor(user0_cfg, actor_checkpoint, device)
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device).eval()
    return user0_cfg, actor


def _compute_teacher_bundle(
    *,
    cfg,
    actor,
    device: torch.device,
    actor_lr: float,
    teacher_horizon: int,
    snapshot_states: list[dict[str, Any]],
    local_states: list[Any],
    valid_masks: np.ndarray,
) -> dict[str, Any]:
    learner = _make_learner(cfg, actor, device, float(actor_lr))
    learner.bw_clean_per_user_horizon = int(teacher_horizon)
    local_batch = _collate_dataclass(local_states, device)
    with torch.inference_mode():
        actor_out = actor.act_bw(local_batch, deterministic=True)
    ref_actions_np = np.asarray(actor_out.action.detach().cpu().numpy(), dtype=np.float32)
    target_actions_np, rho_np, utility_np = learner._bw_clean_target_actions_parallel(
        snapshot_states=[dict(state or {}) for state in snapshot_states],
        ref_actions=ref_actions_np,
        valid_masks=valid_masks,
    )
    ref_returns_np = np.asarray(
        getattr(learner, "_bw_clean_last_ref_returns", np.zeros((len(snapshot_states),), dtype=np.float32)),
        dtype=np.float32,
    )
    target_returns_np = np.asarray(
        learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(target_actions_np[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                for idx in range(target_actions_np.shape[0])
            ],
        ),
        dtype=np.float32,
    )
    return {
        "cfg": cfg,
        "actor": actor,
        "learner": learner,
        "ref_actions": ref_actions_np,
        "target_actions": np.asarray(target_actions_np, dtype=np.float32),
        "valid_masks": np.asarray(valid_masks, dtype=bool),
        "rho": np.asarray(rho_np, dtype=np.float32),
        "utility_l1": np.asarray(utility_np, dtype=np.float32),
        "ref_returns": ref_returns_np,
        "target_returns": target_returns_np,
        "self_gain": np.asarray(target_returns_np - ref_returns_np, dtype=np.float32),
    }


def _cross_eval(
    *,
    snapshot_states: list[dict[str, Any]],
    source_targets: np.ndarray,
    target_bundle: dict[str, Any],
) -> dict[str, Any]:
    learner = target_bundle["learner"]
    cfg = target_bundle["cfg"]
    cross_returns = np.asarray(
        learner._bw_clean_rollout_returns_parallel(
            snapshot_states=[dict(state or {}) for state in snapshot_states],
            first_actions=[
                np.asarray(source_targets[idx], dtype=np.float32).reshape(int(cfg.num_uav), int(cfg.users_obs_max))
                for idx in range(source_targets.shape[0])
            ],
        ),
        dtype=np.float32,
    )
    cross_gain = np.asarray(cross_returns - target_bundle["ref_returns"], dtype=np.float32)
    return {
        "cross_return": _safe_summary(cross_returns),
        "cross_gain": _safe_summary(cross_gain),
        "cross_beats_ref_frac": float(np.mean((cross_returns > target_bundle["ref_returns"] + 1.0e-6).astype(np.float32))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=16)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--teacher_horizon", type=int, default=5)
    parser.add_argument("--offline_steps", type=int, default=20)
    parser.add_argument("--capture_steps", type=int, nargs="+", default=[10, 20])
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
    target_actions_np, _rho_np, _utility_np = teacher_learner._bw_clean_target_actions_parallel(
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

    capture_steps = sorted({int(step) for step in args.capture_steps if int(step) >= 0})
    captured = _capture_user0_offline_states(
        cfg=cfg,
        actor_checkpoint=args.actor_checkpoint,
        device=device,
        snapshot_states=snapshot_states,
        local_states=local_states,
        target_actions_np=np.asarray(target_actions_np, dtype=np.float32),
        valid_masks=valid_masks,
        improving_mask=np.asarray(improving_mask, dtype=np.bool_),
        actor_lr=float(args.actor_lr),
        teacher_horizon=int(args.teacher_horizon),
        capture_steps=capture_steps,
    )

    bundles: dict[int, dict[str, Any]] = {}
    for step in capture_steps:
        user0_cfg, actor = _build_user0_actor_with_state(
            cfg=cfg,
            actor_checkpoint=args.actor_checkpoint,
            actor_state=captured[int(step)],
            device=device,
        )
        bundles[int(step)] = _compute_teacher_bundle(
            cfg=user0_cfg,
            actor=actor,
            device=device,
            actor_lr=float(args.actor_lr),
            teacher_horizon=int(args.teacher_horizon),
            snapshot_states=snapshot_states,
            local_states=local_states,
            valid_masks=valid_masks,
        )

    comparisons: list[dict[str, Any]] = []
    for step_a in capture_steps:
        for step_b in capture_steps:
            if step_a >= step_b:
                continue
            bundle_a = bundles[int(step_a)]
            bundle_b = bundles[int(step_b)]
            target_shift = _masked_l1(
                bundle_a["target_actions"],
                bundle_b["target_actions"],
                valid_masks,
            )
            comparisons.append(
                {
                    "steps": [int(step_a), int(step_b)],
                    "target_shift_l1": _safe_summary(target_shift),
                    f"self_gain_step_{int(step_a)}": _safe_summary(bundle_a["self_gain"]),
                    f"self_gain_step_{int(step_b)}": _safe_summary(bundle_b["self_gain"]),
                    f"cross_{int(step_a)}_target_under_{int(step_b)}": _cross_eval(
                        snapshot_states=snapshot_states,
                        source_targets=bundle_a["target_actions"],
                        target_bundle=bundle_b,
                    ),
                    f"cross_{int(step_b)}_target_under_{int(step_a)}": _cross_eval(
                        snapshot_states=snapshot_states,
                        source_targets=bundle_b["target_actions"],
                        target_bundle=bundle_a,
                    ),
                }
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
        "capture_steps": capture_steps,
        "per_step": {
            f"step_{int(step)}": {
                "ref_return": _safe_summary(bundles[int(step)]["ref_returns"]),
                "target_return": _safe_summary(bundles[int(step)]["target_returns"]),
                "self_gain": _safe_summary(bundles[int(step)]["self_gain"]),
                "self_beats_ref_frac": float(
                    np.mean((bundles[int(step)]["target_returns"] > bundles[int(step)]["ref_returns"] + 1.0e-6).astype(np.float32))
                ),
                "rho": _safe_summary(bundles[int(step)]["rho"]),
                "utility_l1": _safe_summary(bundles[int(step)]["utility_l1"]),
            }
            for step in capture_steps
        },
        "comparisons": comparisons,
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text, encoding="utf-8")


if __name__ == "__main__":
    main()
