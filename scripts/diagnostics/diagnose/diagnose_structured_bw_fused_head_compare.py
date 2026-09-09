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
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _index_dataclass
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving

from scripts.diagnostics.diagnose.diagnose_structured_bw_fixed_teacher_fit import (
    _collect_panel_bank,
    _configure_train_scope,
    _evaluate_panel,
    _make_learner,
    _masked_l1,
    _offline_fit_one_step,
    _safe_summary,
    _set_all_seeds,
)


def _build_actor(cfg, actor_checkpoint: str, device: torch.device):
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    actor = bundle.actor.to(device).eval()
    strict_load = str(getattr(cfg, "structured_bw_loc_readout", "fused") or "fused").strip().lower() not in {
        "user0_residual_fused",
        "fused_moe2",
        "fused_ctx_dot",
        "pairwise_comparator",
    }
    load_checkpoint_forgiving(actor, actor_checkpoint, map_location=device, strict=bool(strict_load))
    return actor


def _run_variant(
    *,
    variant_name: str,
    base_cfg,
    actor_checkpoint: str,
    device: torch.device,
    snapshot_states: list[dict[str, Any]],
    local_states: list[Any],
    target_actions_np: np.ndarray,
    valid_masks: np.ndarray,
    improving_mask: np.ndarray,
    panel_episodes: int,
    teacher_horizon: int,
    offline_steps: int,
    eval_schedule: list[int],
    actor_lr: float,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    setattr(cfg, "structured_bw_loc_readout", str(variant_name))
    actor = _build_actor(cfg, actor_checkpoint, device)
    train_scope_used = _configure_train_scope(actor, "loc_head")
    learner = _make_learner(cfg, actor, device, float(actor_lr))
    learner.bw_clean_per_user_horizon = int(teacher_horizon)

    local_batch = _collate_dataclass(local_states, device)
    train_idx_np = np.flatnonzero(np.asarray(improving_mask, dtype=np.bool_))
    train_idx_t = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    local_batch_train = _index_dataclass(local_batch, train_idx_t)
    target_actions_train_np = np.asarray(target_actions_np[train_idx_np], dtype=np.float32)
    valid_masks_train_np = np.asarray(valid_masks[train_idx_np], dtype=bool)
    target_actions_t = torch.as_tensor(target_actions_train_np, dtype=torch.float32, device=device)
    valid_masks_train = torch.as_tensor(valid_masks_train_np, dtype=torch.float32, device=device) > 0.5

    history: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    try:
        for step in range(0, int(offline_steps) + 1):
            if step in eval_schedule:
                panel_eval = _evaluate_panel(
                    cfg=cfg,
                    actor=learner.actor,
                    device=device,
                    snapshot_states=snapshot_states,
                    fixed_target_actions=target_actions_np,
                )
                with torch.inference_mode():
                    current_out = learner.actor.act_bw(local_batch_train, deterministic=True)
                current_gap = _masked_l1(
                    current_out.action.detach().cpu().numpy(),
                    target_actions_train_np,
                    valid_masks_train_np,
                )
                history.append(
                    {
                        "offline_step": int(step),
                        "panel_current_return": panel_eval["current_return"],
                        "panel_fixed_target_return": panel_eval["fixed_target_return"],
                        "panel_target_minus_current": panel_eval["target_minus_current"],
                        "fixed_target_beats_current_frac": float(panel_eval["fixed_target_beats_current_frac"]),
                        "current_target_gap": _safe_summary(current_gap),
                    }
                )
            if step >= int(offline_steps):
                break
            update_stats = _offline_fit_one_step(
                learner=learner,
                local_batch_train=local_batch_train,
                target_actions=target_actions_t,
                valid_masks_train=valid_masks_train,
                fit_loss=str(getattr(cfg, "bw_clean_per_user_loss", "huber") or "huber"),
            )
            update_stats["offline_step"] = int(step + 1)
            updates.append(update_stats)
    finally:
        del learner
        del actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "variant": str(variant_name),
        "train_scope_used": str(train_scope_used),
        "history": history,
        "updates": updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--panel_states", type=int, default=16)
    parser.add_argument("--panel_seed", type=int, default=35000)
    parser.add_argument("--teacher_horizon", type=int, default=5)
    parser.add_argument("--offline_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20])
    parser.add_argument("--actor_lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["fused", "fused_moe2"],
        help="Loc-readout variants to compare on the same fixed bank and fixed target.",
    )
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

    eval_schedule = sorted({int(step) for step in list(args.eval_steps) if 0 <= int(step) <= int(args.offline_steps)})
    variants = []
    for variant_name in [str(v) for v in args.variants]:
        variants.append(
            _run_variant(
                variant_name=variant_name,
                base_cfg=cfg,
                actor_checkpoint=args.actor_checkpoint,
                device=device,
                snapshot_states=snapshot_states,
                local_states=local_states,
                target_actions_np=target_actions_np,
                valid_masks=valid_masks,
                improving_mask=improving_mask,
                panel_episodes=int(episode_count),
                teacher_horizon=int(args.teacher_horizon),
                offline_steps=int(args.offline_steps),
                eval_schedule=eval_schedule,
                actor_lr=float(args.actor_lr),
            )
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
        "eval_steps": eval_schedule,
        "fixed_bank": {
            "improving_frac": float(np.mean(improving_mask.astype(np.float32))) if improving_mask.size > 0 else 0.0,
            "train_samples": int(np.sum(improving_mask.astype(np.int32))),
            "ref_return": _safe_summary(ref_returns_np),
            "target_return": _safe_summary(target_returns_np),
            "target_minus_ref": _safe_summary(np.asarray(target_returns_np - ref_returns_np, dtype=np.float32)),
            "rho": _safe_summary(np.asarray(rho_np, dtype=np.float32)),
            "utility_l1": _safe_summary(np.asarray(utility_np, dtype=np.float32)),
        },
        "variants": variants,
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text, encoding="utf-8")


if __name__ == "__main__":
    main()
