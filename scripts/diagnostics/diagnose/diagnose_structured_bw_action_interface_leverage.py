from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from itertools import combinations
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
from sagin_marl.rl.structured_bw_update_direction import collect_bw_snapshot_panel
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if text:
            out.append(int(text))
    return out


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if text:
            out.append(float(text))
    return out


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.mean(finite))


def _safe_abs_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite))


def _safe_frac(mask: list[bool] | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr.astype(np.float64)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _rollout_from_snapshot_with_heuristic_tail(
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    k_steps: int,
    gamma: float,
) -> dict[str, float]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    try:
        driver.load_bw_stage_state(snapshot_state)
        action = np.asarray(first_action, dtype=np.float32)
        discounted_reward = 0.0
        discount = 1.0
        for step in range(int(k_steps)):
            step_result, _ = driver.execute_stage_bw_and_prepare_next_accel(action)
            reward = float(next(iter(step_result.rewards.values())))
            discounted_reward += discount * reward
            done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
            if done or step == int(k_steps) - 1:
                break
            driver.begin_step()
            accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
            _ = driver.run_accel_stage(accel_zero)
            _ = driver.run_sat_stage(_zero_sat_action(cfg))
            obs = {agent: env._get_obs(idx) for idx, agent in enumerate(env.agents)}
            from sagin_marl.rl.baselines import queue_aware_bw_policy

            action = np.asarray(queue_aware_bw_policy(list(obs.values()), cfg), dtype=np.float32)
            discount *= float(gamma)
        return {"reward": float(discounted_reward)}
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _safe_action_key(action: np.ndarray, valid_mask: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    clipped = np.where(valid, arr, 0.0)
    return tuple(np.round(clipped, decimals=decimals).tolist())


def _mass_shift(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if int(np.sum(valid)) <= 0:
        return 0.0
    return float(0.5 * np.sum(np.abs(aa[valid] - bb[valid])))


def _build_local_state(snapshot: Any, device: torch.device):
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    return _to_device_dataclass(bw_states[0], device)


def _bw_valid_mask(local_state: Any) -> torch.Tensor:
    return (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)


def _make_score_tensor(
    shape: torch.Size,
    *,
    valid_mask: torch.Tensor,
    default: float = 0.0,
    hi_slots: list[int] | None = None,
    lo_slots: list[int] | None = None,
    hi_value: float = 6.0,
    lo_value: float = -6.0,
) -> torch.Tensor:
    score = torch.full(shape, float(default), dtype=torch.float32, device=valid_mask.device)
    hi_slots = [] if hi_slots is None else hi_slots
    lo_slots = [] if lo_slots is None else lo_slots
    for slot in hi_slots:
        score[:, int(slot)] = float(hi_value)
    for slot in lo_slots:
        score[:, int(slot)] = float(lo_value)
    return torch.where(valid_mask, score, torch.zeros_like(score))


def _alpha_tensor(alpha: float, local_state: Any) -> torch.Tensor:
    return torch.full(
        (local_state.user_nodes.shape[0],),
        float(alpha),
        dtype=local_state.user_nodes.dtype,
        device=local_state.user_nodes.device,
    )


def _interface_base_action_np(bw_policy: Any, local_state: Any, valid_mask: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        base = bw_policy._queue_aware_base_mean(local_state, valid_mask)
    return np.asarray(base.detach().cpu().numpy(), dtype=np.float32)


def _residual_candidates(
    *,
    bw_policy: Any,
    local_state: Any,
    valid_mask: torch.Tensor,
    alpha_grid: list[float],
) -> list[dict[str, Any]]:
    valid_slots = [int(idx) for idx in torch.nonzero(valid_mask[0], as_tuple=False).flatten().tolist()]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    base_np = _interface_base_action_np(bw_policy, local_state, valid_mask)
    valid_mask_np = np.asarray(valid_mask.detach().cpu().numpy()[0], dtype=bool)
    for alpha in alpha_grid:
        alpha_t = _alpha_tensor(float(alpha), local_state)
        for target in valid_slots:
            score = _make_score_tensor(
                local_state.user_nodes[..., 2].shape,
                valid_mask=valid_mask,
                default=-3.0,
                hi_slots=[int(target)],
                hi_value=6.0,
                lo_value=-6.0,
            )
            with torch.no_grad():
                _base_mean, residual_mean = bw_policy._residual_det_mean(
                    local_state=local_state,
                    score=score,
                    alpha=alpha_t,
                    valid_mask=valid_mask,
                )
                action = bw_policy._apply_simplex_floor(residual_mean, valid_mask, bw_policy.residual_floor)
            action_np = np.asarray(action.detach().cpu().numpy(), dtype=np.float32)
            key = _safe_action_key(action_np[0], valid_mask_np)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "family": "residual",
                    "kind": "target_push",
                    "alpha": float(alpha),
                    "target_slot": int(target),
                    "donor_slot": -1,
                    "pair_slot_a": int(target),
                    "pair_slot_b": -1,
                    "lambda": -1.0,
                    "action": action_np,
                    "mass_shift_from_base": _mass_shift(action_np[0], base_np[0], valid_mask_np),
                }
            )
        for donor in valid_slots:
            for receiver in valid_slots:
                if donor == receiver:
                    continue
                score = _make_score_tensor(
                    local_state.user_nodes[..., 2].shape,
                    valid_mask=valid_mask,
                    default=0.0,
                    hi_slots=[int(receiver)],
                    lo_slots=[int(donor)],
                    hi_value=6.0,
                    lo_value=-6.0,
                )
                with torch.no_grad():
                    _base_mean, residual_mean = bw_policy._residual_det_mean(
                        local_state=local_state,
                        score=score,
                        alpha=alpha_t,
                        valid_mask=valid_mask,
                    )
                    action = bw_policy._apply_simplex_floor(residual_mean, valid_mask, bw_policy.residual_floor)
                action_np = np.asarray(action.detach().cpu().numpy(), dtype=np.float32)
                key = _safe_action_key(action_np[0], valid_mask_np)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "family": "residual",
                        "kind": "pair_shift",
                        "alpha": float(alpha),
                        "target_slot": int(receiver),
                        "donor_slot": int(donor),
                        "pair_slot_a": int(receiver),
                        "pair_slot_b": int(donor),
                        "lambda": -1.0,
                        "action": action_np,
                        "mass_shift_from_base": _mass_shift(action_np[0], base_np[0], valid_mask_np),
                    }
                )
    return candidates


def _focus2_candidates(
    *,
    bw_policy: Any,
    local_state: Any,
    valid_mask: torch.Tensor,
    alpha_grid: list[float],
    lambda_grid: list[float],
) -> list[dict[str, Any]]:
    valid_slots = [int(idx) for idx in torch.nonzero(valid_mask[0], as_tuple=False).flatten().tolist()]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    base_np = _interface_base_action_np(bw_policy, local_state, valid_mask)
    valid_mask_np = np.asarray(valid_mask.detach().cpu().numpy()[0], dtype=bool)
    for alpha in alpha_grid:
        alpha_t = _alpha_tensor(float(alpha), local_state)
        for slot_a, slot_b in combinations(valid_slots, 2):
            for lam in lambda_grid:
                lam_clip = float(np.clip(float(lam), 1.0e-3, 1.0 - 1.0e-3))
                score = torch.full(
                    local_state.user_nodes[..., 2].shape,
                    -12.0,
                    dtype=torch.float32,
                    device=valid_mask.device,
                )
                score[:, int(slot_a)] = float(np.log(lam_clip))
                score[:, int(slot_b)] = float(np.log(1.0 - lam_clip))
                score = torch.where(valid_mask, score, torch.zeros_like(score))
                with torch.no_grad():
                    _base_mean, _focus_target, dist_mean, _support_mask = bw_policy._focus_topk_det_mean(
                        local_state=local_state,
                        score=score,
                        alpha=alpha_t,
                        valid_mask=valid_mask,
                        focus_k=2,
                    )
                    action = bw_policy._apply_simplex_floor(dist_mean, valid_mask, bw_policy.residual_floor)
                action_np = np.asarray(action.detach().cpu().numpy(), dtype=np.float32)
                key = _safe_action_key(action_np[0], valid_mask_np)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "family": "focus2",
                        "kind": "pair_focus",
                        "alpha": float(alpha),
                        "target_slot": int(slot_a if lam >= 0.5 else slot_b),
                        "donor_slot": -1,
                        "pair_slot_a": int(slot_a),
                        "pair_slot_b": int(slot_b),
                        "lambda": float(lam),
                        "action": action_np,
                        "mass_shift_from_base": _mass_shift(action_np[0], base_np[0], valid_mask_np),
                    }
                )
    return candidates


def _transfer_candidates(
    *,
    bw_policy: Any,
    local_state: Any,
    valid_mask: torch.Tensor,
    alpha_grid: list[float],
) -> list[dict[str, Any]]:
    valid_slots = [int(idx) for idx in torch.nonzero(valid_mask[0], as_tuple=False).flatten().tolist()]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    base_np = _interface_base_action_np(bw_policy, local_state, valid_mask)
    valid_mask_np = np.asarray(valid_mask.detach().cpu().numpy()[0], dtype=bool)
    for alpha in alpha_grid:
        alpha_t = _alpha_tensor(float(alpha), local_state)
        for donor in valid_slots:
            for receiver in valid_slots:
                if donor == receiver:
                    continue
                score = _make_score_tensor(
                    local_state.user_nodes[..., 2].shape,
                    valid_mask=valid_mask,
                    default=0.0,
                    hi_slots=[int(receiver)],
                    lo_slots=[int(donor)],
                    hi_value=6.0,
                    lo_value=-6.0,
                )
                with torch.no_grad():
                    _base_mean, det_mean, _dist_mean, _pair_mask = bw_policy._transfer_det_mean(
                        local_state=local_state,
                        score=score,
                        alpha=alpha_t,
                        valid_mask=valid_mask,
                    )
                action_np = np.asarray(det_mean.detach().cpu().numpy(), dtype=np.float32)
                key = _safe_action_key(action_np[0], valid_mask_np)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "family": "transfer",
                        "kind": "donor_receiver",
                        "alpha": float(alpha),
                        "target_slot": int(receiver),
                        "donor_slot": int(donor),
                        "pair_slot_a": int(receiver),
                        "pair_slot_b": int(donor),
                        "lambda": -1.0,
                        "action": action_np,
                        "mass_shift_from_base": _mass_shift(action_np[0], base_np[0], valid_mask_np),
                    }
                )
    return candidates


def _generate_candidates(
    *,
    parameterization: str,
    bw_policy: Any,
    local_state: Any,
    valid_mask: torch.Tensor,
    alpha_grid: list[float],
    lambda_grid: list[float],
) -> list[dict[str, Any]]:
    if parameterization == "score_residual_fixedkappa_dirichlet":
        return _residual_candidates(
            bw_policy=bw_policy,
            local_state=local_state,
            valid_mask=valid_mask,
            alpha_grid=alpha_grid,
        )
    if parameterization == "score_focus2_fixedkappa_dirichlet":
        return _focus2_candidates(
            bw_policy=bw_policy,
            local_state=local_state,
            valid_mask=valid_mask,
            alpha_grid=alpha_grid,
            lambda_grid=lambda_grid,
        )
    if parameterization == "score_transfer_fixedkappa_dirichlet":
        return _transfer_candidates(
            bw_policy=bw_policy,
            local_state=local_state,
            valid_mask=valid_mask,
            alpha_grid=alpha_grid,
        )
    raise ValueError(f"Unsupported BW parameterization for interface leverage probe: {parameterization!r}")


def _analyze_config(
    *,
    cfg,
    config_path: Path,
    panel: list[dict[str, Any]],
    horizons: list[int],
    alpha_grid: list[float],
    lambda_grid: list[float],
    out_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cpu")
    bundle = build_structured_modules_from_config(cfg)
    actor = bundle.actor
    actor.eval()
    bw_policy = actor.bw_policy
    parameterization = str(getattr(cfg, "structured_bw_parameterization", "legacy") or "legacy").strip().lower()

    horizon_metrics: dict[int, dict[str, list[float]]] = {
        int(h): {
            "base_gap_vs_heuristic": [],
            "best_gap_vs_heuristic": [],
            "best_gap_vs_base": [],
            "mean_abs_delta_vs_base": [],
            "mean_abs_delta_vs_heuristic": [],
            "candidate_gap": [],
            "best_mass_shift": [],
            "base_mass_shift_vs_heuristic": [],
            "best_positive_vs_heuristic": [],
            "best_positive_vs_base": [],
            "candidate_count": [],
        }
        for h in horizons
    }
    state_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for state_idx, entry in enumerate(panel):
        snapshot = entry["snapshot"]
        snapshot_state = entry["snapshot_state"]
        heuristic_action = np.asarray(entry["heuristic_action"], dtype=np.float32)
        local_state = _build_local_state(snapshot, device)
        valid_mask_t = _bw_valid_mask(local_state)
        valid_mask_np = np.asarray(valid_mask_t.detach().cpu().numpy()[0], dtype=bool)
        valid_slots = [int(idx) for idx in np.flatnonzero(valid_mask_np).tolist()]
        if len(valid_slots) <= 1:
            continue
        base_action = _interface_base_action_np(bw_policy, local_state, valid_mask_t)
        candidates = _generate_candidates(
            parameterization=parameterization,
            bw_policy=bw_policy,
            local_state=local_state,
            valid_mask=valid_mask_t,
            alpha_grid=alpha_grid,
            lambda_grid=lambda_grid,
        )

        state_row: dict[str, Any] = {
            "config_name": config_path.stem,
            "state_index": int(state_idx),
            "episode": int(entry["episode"]),
            "t": int(entry["t"]),
            "valid_count": int(len(valid_slots)),
            "candidate_count_total": int(len(candidates)),
            "base_mass_shift_vs_heuristic": _mass_shift(base_action[0], heuristic_action[0], valid_mask_np),
        }

        for horizon in horizons:
            heuristic_reward = float(
                _rollout_from_snapshot_with_heuristic_tail(
                    snapshot_state=snapshot_state,
                    first_action=heuristic_action,
                    cfg=cfg,
                    k_steps=int(horizon),
                    gamma=float(cfg.gamma),
                )["reward"]
            )
            base_reward = float(
                _rollout_from_snapshot_with_heuristic_tail(
                    snapshot_state=snapshot_state,
                    first_action=base_action,
                    cfg=cfg,
                    k_steps=int(horizon),
                    gamma=float(cfg.gamma),
                )["reward"]
            )
            candidate_rewards: list[float] = []
            candidate_base_deltas: list[float] = []
            candidate_heur_deltas: list[float] = []
            best_reward = base_reward
            best_base_delta = 0.0
            best_heur_delta = base_reward - heuristic_reward
            best_mass_shift = 0.0
            best_kind = "base"
            best_target = -1
            best_donor = -1
            best_pair_a = -1
            best_pair_b = -1
            best_alpha = -1.0
            best_lambda = -1.0

            for candidate_idx, candidate in enumerate(candidates):
                action = np.asarray(candidate["action"], dtype=np.float32)
                reward = float(
                    _rollout_from_snapshot_with_heuristic_tail(
                        snapshot_state=snapshot_state,
                        first_action=action,
                        cfg=cfg,
                        k_steps=int(horizon),
                        gamma=float(cfg.gamma),
                    )["reward"]
                )
                delta_vs_base = float(reward - base_reward)
                delta_vs_heuristic = float(reward - heuristic_reward)
                candidate_rewards.append(reward)
                candidate_base_deltas.append(delta_vs_base)
                candidate_heur_deltas.append(delta_vs_heuristic)
                candidate_rows.append(
                    {
                        "config_name": config_path.stem,
                        "state_index": int(state_idx),
                        "episode": int(entry["episode"]),
                        "t": int(entry["t"]),
                        "horizon": int(horizon),
                        "candidate_index": int(candidate_idx),
                        "family": str(candidate["family"]),
                        "kind": str(candidate["kind"]),
                        "alpha": float(candidate["alpha"]),
                        "lambda": float(candidate["lambda"]),
                        "target_slot": int(candidate["target_slot"]),
                        "donor_slot": int(candidate["donor_slot"]),
                        "pair_slot_a": int(candidate["pair_slot_a"]),
                        "pair_slot_b": int(candidate["pair_slot_b"]),
                        "reward": float(reward),
                        "delta_vs_base": float(delta_vs_base),
                        "delta_vs_heuristic": float(delta_vs_heuristic),
                        "mass_shift_from_base": float(candidate["mass_shift_from_base"]),
                    }
                )
                if reward > best_reward + 1.0e-9:
                    best_reward = float(reward)
                    best_base_delta = float(delta_vs_base)
                    best_heur_delta = float(delta_vs_heuristic)
                    best_mass_shift = float(candidate["mass_shift_from_base"])
                    best_kind = str(candidate["kind"])
                    best_target = int(candidate["target_slot"])
                    best_donor = int(candidate["donor_slot"])
                    best_pair_a = int(candidate["pair_slot_a"])
                    best_pair_b = int(candidate["pair_slot_b"])
                    best_alpha = float(candidate["alpha"])
                    best_lambda = float(candidate["lambda"])

            candidate_gap = (max(candidate_rewards) - min(candidate_rewards)) if len(candidate_rewards) >= 2 else 0.0
            horizon_metrics[int(horizon)]["base_gap_vs_heuristic"].append(float(base_reward - heuristic_reward))
            horizon_metrics[int(horizon)]["best_gap_vs_heuristic"].append(float(best_heur_delta))
            horizon_metrics[int(horizon)]["best_gap_vs_base"].append(float(best_base_delta))
            horizon_metrics[int(horizon)]["mean_abs_delta_vs_base"].append(_safe_abs_mean(candidate_base_deltas))
            horizon_metrics[int(horizon)]["mean_abs_delta_vs_heuristic"].append(_safe_abs_mean(candidate_heur_deltas))
            horizon_metrics[int(horizon)]["candidate_gap"].append(float(candidate_gap))
            horizon_metrics[int(horizon)]["best_mass_shift"].append(float(best_mass_shift))
            horizon_metrics[int(horizon)]["base_mass_shift_vs_heuristic"].append(
                float(_mass_shift(base_action[0], heuristic_action[0], valid_mask_np))
            )
            horizon_metrics[int(horizon)]["best_positive_vs_heuristic"].append(bool(best_heur_delta > 1.0e-9))
            horizon_metrics[int(horizon)]["best_positive_vs_base"].append(bool(best_base_delta > 1.0e-9))
            horizon_metrics[int(horizon)]["candidate_count"].append(float(len(candidates)))

            state_row[f"h{int(horizon)}_heuristic_reward"] = float(heuristic_reward)
            state_row[f"h{int(horizon)}_base_reward"] = float(base_reward)
            state_row[f"h{int(horizon)}_base_gap_vs_heuristic"] = float(base_reward - heuristic_reward)
            state_row[f"h{int(horizon)}_best_reward"] = float(best_reward)
            state_row[f"h{int(horizon)}_best_gap_vs_heuristic"] = float(best_heur_delta)
            state_row[f"h{int(horizon)}_best_gap_vs_base"] = float(best_base_delta)
            state_row[f"h{int(horizon)}_mean_abs_delta_vs_base"] = _safe_abs_mean(candidate_base_deltas)
            state_row[f"h{int(horizon)}_mean_abs_delta_vs_heuristic"] = _safe_abs_mean(candidate_heur_deltas)
            state_row[f"h{int(horizon)}_candidate_gap"] = float(candidate_gap)
            state_row[f"h{int(horizon)}_best_mass_shift"] = float(best_mass_shift)
            state_row[f"h{int(horizon)}_best_kind"] = str(best_kind)
            state_row[f"h{int(horizon)}_best_target_slot"] = int(best_target)
            state_row[f"h{int(horizon)}_best_donor_slot"] = int(best_donor)
            state_row[f"h{int(horizon)}_best_pair_slot_a"] = int(best_pair_a)
            state_row[f"h{int(horizon)}_best_pair_slot_b"] = int(best_pair_b)
            state_row[f"h{int(horizon)}_best_alpha"] = float(best_alpha)
            state_row[f"h{int(horizon)}_best_lambda"] = float(best_lambda)
        state_rows.append(state_row)

    summary: dict[str, Any] = {
        "config_name": config_path.stem,
        "config_path": str(config_path),
        "parameterization": str(parameterization),
        "states_collected": int(len(panel)),
        "states_analyzed": int(len(state_rows)),
        "horizons": [int(h) for h in horizons],
        "alpha_grid": [float(v) for v in alpha_grid],
        "lambda_grid": [float(v) for v in lambda_grid],
        "horizon_metrics": {},
    }
    for horizon in horizons:
        metrics = horizon_metrics[int(horizon)]
        summary["horizon_metrics"][f"h{int(horizon)}"] = {
            "base_gap_vs_heuristic_mean": _safe_mean(metrics["base_gap_vs_heuristic"]),
            "base_gap_vs_heuristic_std": _safe_std(metrics["base_gap_vs_heuristic"]),
            "best_gap_vs_heuristic_mean": _safe_mean(metrics["best_gap_vs_heuristic"]),
            "best_gap_vs_heuristic_std": _safe_std(metrics["best_gap_vs_heuristic"]),
            "best_gap_vs_base_mean": _safe_mean(metrics["best_gap_vs_base"]),
            "best_gap_vs_base_std": _safe_std(metrics["best_gap_vs_base"]),
            "mean_abs_delta_vs_base": _safe_mean(metrics["mean_abs_delta_vs_base"]),
            "mean_abs_delta_vs_heuristic": _safe_mean(metrics["mean_abs_delta_vs_heuristic"]),
            "candidate_gap_mean": _safe_mean(metrics["candidate_gap"]),
            "best_mass_shift_mean": _safe_mean(metrics["best_mass_shift"]),
            "base_mass_shift_vs_heuristic_mean": _safe_mean(metrics["base_mass_shift_vs_heuristic"]),
            "best_positive_vs_heuristic_frac": _safe_frac(metrics["best_positive_vs_heuristic"]),
            "best_positive_vs_base_frac": _safe_frac(metrics["best_positive_vs_base"]),
            "candidate_count_mean": _safe_mean(metrics["candidate_count"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "state_metrics.csv", state_rows)
    _write_csv(out_dir / "candidate_metrics.csv", candidate_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose interface-conditioned BW leverage on fixed snapshot panels.")
    parser.add_argument("--configs", type=str, required=True, help="Comma-separated config paths.")
    parser.add_argument("--panel_config", type=str, default="", help="Optional config used only to collect the snapshot panel.")
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--horizons", type=str, default="2,5,10")
    parser.add_argument("--alpha_grid", type=str, default="0.33,0.67,1.0")
    parser.add_argument("--lambda_grid", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    _set_all_seeds(int(args.seed))
    config_paths = [Path(part.strip()) for part in str(args.configs).split(",") if part.strip()]
    if not config_paths:
        raise ValueError("No config paths provided.")
    panel_config_path = Path(str(args.panel_config).strip()) if str(args.panel_config).strip() else config_paths[0]
    panel_cfg = load_config(str(panel_config_path))
    panel = collect_bw_snapshot_panel(panel_cfg, episodes=int(args.episodes), states=int(args.states), seed=int(args.seed))
    if not panel:
        raise RuntimeError(f"No BW snapshots collected from panel config: {panel_config_path}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    horizons = _parse_int_list(args.horizons)
    alpha_grid = _parse_float_list(args.alpha_grid)
    lambda_grid = _parse_float_list(args.lambda_grid)

    summaries: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        cfg = load_config(str(config_path))
        cfg_out_dir = out_root / config_path.stem
        summary = _analyze_config(
            cfg=cfg,
            config_path=config_path,
            panel=panel,
            horizons=horizons,
            alpha_grid=alpha_grid,
            lambda_grid=lambda_grid,
            out_dir=cfg_out_dir,
        )
        summaries.append(summary)
        row: dict[str, Any] = {
            "config_name": str(summary["config_name"]),
            "parameterization": str(summary["parameterization"]),
            "states_analyzed": int(summary["states_analyzed"]),
        }
        for horizon_key, metrics in summary["horizon_metrics"].items():
            row[f"{horizon_key}_base_gap_vs_heuristic_mean"] = float(metrics["base_gap_vs_heuristic_mean"])
            row[f"{horizon_key}_best_gap_vs_heuristic_mean"] = float(metrics["best_gap_vs_heuristic_mean"])
            row[f"{horizon_key}_best_gap_vs_base_mean"] = float(metrics["best_gap_vs_base_mean"])
            row[f"{horizon_key}_mean_abs_delta_vs_base"] = float(metrics["mean_abs_delta_vs_base"])
            row[f"{horizon_key}_candidate_gap_mean"] = float(metrics["candidate_gap_mean"])
            row[f"{horizon_key}_best_mass_shift_mean"] = float(metrics["best_mass_shift_mean"])
            row[f"{horizon_key}_best_positive_vs_heuristic_frac"] = float(metrics["best_positive_vs_heuristic_frac"])
        combined_rows.append(row)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "panel_config": str(panel_config_path),
                "states_collected": int(len(panel)),
                "runs": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    _write_csv(out_root / "summary.csv", combined_rows)


if __name__ == "__main__":
    main()
