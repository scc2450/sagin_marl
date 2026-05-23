from __future__ import annotations

import os
from typing import Any, Dict, List

import torch

from sagin_marl.utils.checkpoint import load_state_dict_forgiving


def save_structured_train_state(
    log_dir: str,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    *,
    actor_stage_optimizers: Dict[int, Any] | None = None,
    update: int,
    planned_total_updates: int,
    total_env_steps: int,
    history_rows: List[Dict[str, float]],
    total_time_sec: float,
    checkpoint_eval_state: Dict[str, float] | None = None,
    checkpoint_eval_fixed_summary: Dict[str, float] | None = None,
    suffix: str | None = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    state_name = "train_state.pt" if suffix is None else f"train_state_{suffix}.pt"
    critic_enabled = critic is not None
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_enabled": bool(critic_enabled),
        "critic_state_dict": None if not critic_enabled else critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "actor_stage_optimizer_state_dicts": {
            str(int(stage_id)): optimizer.state_dict()
            for stage_id, optimizer in dict(actor_stage_optimizers or {}).items()
            if optimizer is not None
        },
        "critic_optimizer_state_dict": None if critic_optimizer is None else critic_optimizer.state_dict(),
        "actor_scheduler_state_dict": None,
        "critic_scheduler_state_dict": None,
        "reward_rms": None,
        "update": int(update),
        "planned_total_updates": int(planned_total_updates),
        "total_env_steps": int(total_env_steps),
        "reward_history": [float(row.get("env_reward_mean", 0.0)) for row in history_rows],
        "history_rows": [dict(row) for row in history_rows],
        "best_ma": float("nan"),
        "no_improve": 0,
        "checkpoint_eval_state": {
            str(k): float(v) for k, v in dict(checkpoint_eval_state or {}).items()
        },
        "checkpoint_eval_fixed_summary": checkpoint_eval_fixed_summary,
        "total_time_sec": float(total_time_sec),
    }
    out_path = os.path.join(log_dir, state_name)
    torch.save(payload, out_path)
    return out_path


def load_structured_train_state(
    path: str,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    *,
    actor_stage_optimizers: Dict[int, Any] | None = None,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Train state '{path}' did not contain a dictionary payload.")
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if not isinstance(actor_state, dict):
        raise KeyError(f"Train state '{path}' is missing 'actor_state_dict'.")
    actor_info = load_state_dict_forgiving(actor, actor_state, strict=False)
    critic_info = None
    if critic is not None and isinstance(critic_state, dict):
        critic_info = load_state_dict_forgiving(critic, critic_state, strict=False)
    actor_optimizer_state = payload.get("actor_optimizer_state_dict")
    actor_stage_optimizer_states = payload.get("actor_stage_optimizer_state_dicts")
    critic_optimizer_state = payload.get("critic_optimizer_state_dict")
    if actor_optimizer_state is not None:
        try:
            actor_optimizer.load_state_dict(actor_optimizer_state)
        except ValueError as exc:
            print(f"Warning: failed to load actor optimizer state from {path}: {exc}")
    if isinstance(actor_stage_optimizer_states, dict) and actor_stage_optimizers:
        for stage_id, optimizer in dict(actor_stage_optimizers).items():
            state = actor_stage_optimizer_states.get(str(int(stage_id)))
            if state is None:
                continue
            try:
                optimizer.load_state_dict(state)
            except ValueError as exc:
                print(
                    "Warning: failed to load actor stage optimizer "
                    f"{int(stage_id)} state from {path}: {exc}"
                )
    if critic_optimizer is not None and critic_optimizer_state is not None:
        try:
            critic_optimizer.load_state_dict(critic_optimizer_state)
        except ValueError as exc:
            print(f"Warning: failed to load critic optimizer state from {path}: {exc}")
    return {
        "path": path,
        "actor_info": actor_info,
        "critic_info": critic_info,
        "update": int(payload.get("update", 0) or 0),
        "planned_total_updates": int(payload.get("planned_total_updates", 0) or 0),
        "total_env_steps": int(payload.get("total_env_steps", 0) or 0),
        "reward_history": [float(x) for x in (payload.get("reward_history", []) or [])],
        "history_rows": [dict(row) for row in (payload.get("history_rows", []) or [])],
        "critic_enabled": bool(payload.get("critic_enabled", isinstance(critic_state, dict))),
        "checkpoint_eval_state": {
            str(k): float(v) for k, v in dict(payload.get("checkpoint_eval_state", {}) or {}).items()
        },
        "checkpoint_eval_fixed_summary": (
            dict(payload["checkpoint_eval_fixed_summary"])
            if payload.get("checkpoint_eval_fixed_summary") is not None
            else None
        ),
        "total_time_sec": float(payload.get("total_time_sec", 0.0) or 0.0),
    }
