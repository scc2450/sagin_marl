from __future__ import annotations

import atexit
import csv
import json
import os
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_mappo import _collate_dataclass, _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group


_BRANCH_PROBE_GROUP_CACHE: dict[str, Any] = {
    "signature": None,
    "group": None,
}


def _close_cached_branch_probe_group() -> None:
    group = _BRANCH_PROBE_GROUP_CACHE.get("group")
    if group is None:
        return
    close_fn = getattr(group, "close", None)
    if callable(close_fn):
        close_fn()
    _BRANCH_PROBE_GROUP_CACHE["group"] = None
    _BRANCH_PROBE_GROUP_CACHE["signature"] = None


atexit.register(_close_cached_branch_probe_group)


def _make_probe_driver(cfg):
    del cfg
    raise RuntimeError("legacy single-driver BW probes have been removed; use native batch tensor probes.")


def _close_probe_driver(driver) -> None:
    close_structured_env_group(driver)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) <= 1 or len(y) <= 1 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xa) & np.isfinite(ya)
    if int(np.sum(finite)) <= 1:
        return 0.0
    xa = xa[finite]
    ya = ya[finite]
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.mean(finite))


def _safe_frac(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr.astype(np.float64)))


def _sign_agree_frac(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size <= 0 or ya.size <= 0 or xa.size != ya.size:
        return 0.0
    finite = np.isfinite(xa) & np.isfinite(ya) & (np.abs(xa) > 1.0e-12) & (np.abs(ya) > 1.0e-12)
    if int(np.sum(finite)) <= 0:
        return 0.0
    return float(np.mean((xa[finite] * ya[finite]) > 0.0))


def _true_advantage_snr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0
    std = float(np.std(finite))
    return float(np.mean(np.abs(finite)) / max(std, 1.0e-12))


def _select_dataclass_batch(batch: Any, positions: list[int], device: torch.device) -> Any:
    if not is_dataclass(batch):
        return batch
    if not positions:
        raise ValueError("positions must be non-empty")
    index_cpu = torch.as_tensor(positions, dtype=torch.long)
    kwargs: dict[str, Any] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            kwargs[field.name] = value.index_select(0, index_cpu.to(value.device))
        elif isinstance(value, np.ndarray):
            kwargs[field.name] = torch.from_numpy(value[np.asarray(positions, dtype=np.int64)]).to(device)
        elif isinstance(value, list):
            kwargs[field.name] = [value[int(i)] for i in positions]
        else:
            kwargs[field.name] = value
    return type(batch)(**kwargs)


def _snapshot_step(snapshot_state: dict[str, Any]) -> int:
    env_state = snapshot_state.get("env_state", {}) if isinstance(snapshot_state, dict) else {}
    if isinstance(env_state, dict) and "t" in env_state:
        return int(env_state.get("t", 0) or 0)
    if isinstance(snapshot_state, dict) and "t" in snapshot_state:
        return int(snapshot_state.get("t", 0) or 0)
    return 0


def _summarize_true_advantage_alignment(
    *,
    ppo_advantages: np.ndarray,
    true_advantages: np.ndarray,
    delta_logprob: np.ndarray,
) -> dict[str, float]:
    ppo_adv = np.asarray(ppo_advantages, dtype=np.float64)
    true_adv = np.asarray(true_advantages, dtype=np.float64)
    dlogp = np.asarray(delta_logprob, dtype=np.float64)
    if ppo_adv.size <= 0 or true_adv.size <= 0 or dlogp.size <= 0:
        return {
            "corr_advantage_vs_true_adv_mc": 0.0,
            "sign_agree_advantage_true_adv_mc": 0.0,
            "corr_true_adv_mc_vs_delta_logprob": 0.0,
            "true_adv_mc_abs_mean": 0.0,
            "true_adv_mc_std": 0.0,
            "true_adv_mc_snr": 0.0,
            "mean_true_adv_mc_pos_adv": 0.0,
            "mean_true_adv_mc_neg_adv": 0.0,
            "ppo_pos_true_adv_positive_frac": 0.0,
            "ppo_neg_true_adv_negative_frac": 0.0,
            "mean_delta_logprob_true_adv_pos": 0.0,
            "mean_delta_logprob_true_adv_neg": 0.0,
            "true_adv_pos_logprob_up_frac": 0.0,
            "true_adv_neg_logprob_down_frac": 0.0,
            "mean_true_adv_mc_times_delta_logprob": 0.0,
            "mean_abs_delta_logprob": 0.0,
        }
    pos_ppo = ppo_adv > 0.0
    neg_ppo = ppo_adv < 0.0
    pos_true = true_adv > 0.0
    neg_true = true_adv < 0.0
    return {
        "corr_advantage_vs_true_adv_mc": _safe_corr(ppo_adv.tolist(), true_adv.tolist()),
        "sign_agree_advantage_true_adv_mc": _sign_agree_frac(ppo_adv, true_adv),
        "corr_true_adv_mc_vs_delta_logprob": _safe_corr(true_adv.tolist(), dlogp.tolist()),
        "true_adv_mc_abs_mean": float(np.mean(np.abs(true_adv))) if true_adv.size > 0 else 0.0,
        "true_adv_mc_std": float(np.std(true_adv)) if true_adv.size > 0 else 0.0,
        "true_adv_mc_snr": _true_advantage_snr(true_adv),
        "mean_true_adv_mc_pos_adv": _safe_mean(true_adv[pos_ppo]),
        "mean_true_adv_mc_neg_adv": _safe_mean(true_adv[neg_ppo]),
        "ppo_pos_true_adv_positive_frac": _safe_frac(true_adv[pos_ppo] > 0.0),
        "ppo_neg_true_adv_negative_frac": _safe_frac(true_adv[neg_ppo] < 0.0),
        "mean_delta_logprob_true_adv_pos": _safe_mean(dlogp[pos_true]),
        "mean_delta_logprob_true_adv_neg": _safe_mean(dlogp[neg_true]),
        "true_adv_pos_logprob_up_frac": _safe_frac(dlogp[pos_true] > 0.0),
        "true_adv_neg_logprob_down_frac": _safe_frac(dlogp[neg_true] < 0.0),
        "mean_true_adv_mc_times_delta_logprob": _safe_mean(true_adv * dlogp),
        "mean_abs_delta_logprob": float(np.mean(np.abs(dlogp))) if dlogp.size > 0 else 0.0,
    }


def _summarize_advantage_decomposition(
    *,
    norm_advantages: np.ndarray,
    raw_advantages: np.ndarray,
    return_targets: np.ndarray,
    value_preds: np.ndarray,
    sampled_q_mc: np.ndarray,
    policy_q_mc: np.ndarray,
    delta_logprob: np.ndarray,
) -> dict[str, float]:
    norm_adv = np.asarray(norm_advantages, dtype=np.float64)
    raw_adv = np.asarray(raw_advantages, dtype=np.float64)
    returns = np.asarray(return_targets, dtype=np.float64)
    values = np.asarray(value_preds, dtype=np.float64)
    sampled_q = np.asarray(sampled_q_mc, dtype=np.float64)
    policy_q = np.asarray(policy_q_mc, dtype=np.float64)
    dlogp = np.asarray(delta_logprob, dtype=np.float64)
    true_adv = sampled_q - policy_q
    if (
        norm_adv.size <= 0
        or raw_adv.size <= 0
        or returns.size <= 0
        or values.size <= 0
        or sampled_q.size <= 0
        or policy_q.size <= 0
        or dlogp.size <= 0
    ):
        return {
            "corr_norm_advantage_vs_raw_advantage": 0.0,
            "sign_agree_norm_raw_advantage": 0.0,
            "corr_raw_advantage_vs_true_adv_mc": 0.0,
            "sign_agree_raw_advantage_true_adv_mc": 0.0,
            "corr_raw_advantage_vs_delta_logprob": 0.0,
            "raw_advantage_abs_mean": 0.0,
            "raw_advantage_std": 0.0,
            "raw_advantage_snr": 0.0,
            "corr_return_target_vs_sampled_q_mc": 0.0,
            "return_target_sampled_q_error_mean": 0.0,
            "return_target_sampled_q_error_abs_mean": 0.0,
            "corr_value_vs_policy_q_mc": 0.0,
            "value_policy_q_error_mean": 0.0,
            "value_policy_q_error_abs_mean": 0.0,
            "return_target_std": 0.0,
            "sampled_q_mc_std": 0.0,
            "value_pred_std": 0.0,
            "policy_q_mc_std": 0.0,
        }
    return_error = returns - sampled_q
    value_error = values - policy_q
    return {
        "corr_norm_advantage_vs_raw_advantage": _safe_corr(norm_adv.tolist(), raw_adv.tolist()),
        "sign_agree_norm_raw_advantage": _sign_agree_frac(norm_adv, raw_adv),
        "corr_raw_advantage_vs_true_adv_mc": _safe_corr(raw_adv.tolist(), true_adv.tolist()),
        "sign_agree_raw_advantage_true_adv_mc": _sign_agree_frac(raw_adv, true_adv),
        "corr_raw_advantage_vs_delta_logprob": _safe_corr(raw_adv.tolist(), dlogp.tolist()),
        "raw_advantage_abs_mean": float(np.mean(np.abs(raw_adv))) if raw_adv.size > 0 else 0.0,
        "raw_advantage_std": float(np.std(raw_adv)) if raw_adv.size > 0 else 0.0,
        "raw_advantage_snr": _true_advantage_snr(raw_adv),
        "corr_return_target_vs_sampled_q_mc": _safe_corr(returns.tolist(), sampled_q.tolist()),
        "return_target_sampled_q_error_mean": _safe_mean(return_error),
        "return_target_sampled_q_error_abs_mean": float(np.mean(np.abs(return_error))) if return_error.size > 0 else 0.0,
        "corr_value_vs_policy_q_mc": _safe_corr(values.tolist(), policy_q.tolist()),
        "value_policy_q_error_mean": _safe_mean(value_error),
        "value_policy_q_error_abs_mean": float(np.mean(np.abs(value_error))) if value_error.size > 0 else 0.0,
        "return_target_std": float(np.std(returns)) if returns.size > 0 else 0.0,
        "sampled_q_mc_std": float(np.std(sampled_q)) if sampled_q.size > 0 else 0.0,
        "value_pred_std": float(np.std(values)) if values.size > 0 else 0.0,
        "policy_q_mc_std": float(np.std(policy_q)) if policy_q.size > 0 else 0.0,
    }


def _summarize_branch_alignment(
    *,
    norm_advantages: np.ndarray,
    raw_advantages: np.ndarray,
    branch_deltas: np.ndarray,
    delta_logprob: np.ndarray,
) -> dict[str, float]:
    norm_adv = np.asarray(norm_advantages, dtype=np.float64)
    raw_adv = np.asarray(raw_advantages, dtype=np.float64)
    branch = np.asarray(branch_deltas, dtype=np.float64)
    dlogp = np.asarray(delta_logprob, dtype=np.float64)
    if norm_adv.size <= 0 or raw_adv.size <= 0 or branch.size <= 0 or dlogp.size <= 0:
        return {
            "sample_count": 0.0,
            "corr_advantage_vs_branch_delta": 0.0,
            "corr_raw_advantage_vs_branch_delta": 0.0,
            "sign_agree_raw_advantage_branch_delta": 0.0,
            "corr_branch_delta_vs_delta_logprob": 0.0,
            "branch_delta_abs_mean": 0.0,
            "branch_delta_std": 0.0,
            "branch_delta_snr": 0.0,
            "mean_branch_delta_pos_adv": 0.0,
            "mean_branch_delta_neg_adv": 0.0,
            "mean_branch_delta_pos_raw_adv": 0.0,
            "mean_branch_delta_neg_raw_adv": 0.0,
            "raw_pos_branch_delta_positive_frac": 0.0,
            "raw_neg_branch_delta_negative_frac": 0.0,
            "branch_delta_pos_logprob_up_frac": 0.0,
            "branch_delta_neg_logprob_down_frac": 0.0,
            "mean_branch_delta_times_delta_logprob": 0.0,
            "mean_abs_delta_logprob": 0.0,
        }
    finite = np.isfinite(norm_adv) & np.isfinite(raw_adv) & np.isfinite(branch) & np.isfinite(dlogp)
    norm_adv = norm_adv[finite]
    raw_adv = raw_adv[finite]
    branch = branch[finite]
    dlogp = dlogp[finite]
    if branch.size <= 0:
        return _summarize_branch_alignment(
            norm_advantages=np.asarray([], dtype=np.float64),
            raw_advantages=np.asarray([], dtype=np.float64),
            branch_deltas=np.asarray([], dtype=np.float64),
            delta_logprob=np.asarray([], dtype=np.float64),
        )
    pos_adv = norm_adv > 0.0
    neg_adv = norm_adv < 0.0
    pos_raw = raw_adv > 0.0
    neg_raw = raw_adv < 0.0
    pos_branch = branch > 0.0
    neg_branch = branch < 0.0
    return {
        "sample_count": float(branch.size),
        "corr_advantage_vs_branch_delta": _safe_corr(norm_adv.tolist(), branch.tolist()),
        "corr_raw_advantage_vs_branch_delta": _safe_corr(raw_adv.tolist(), branch.tolist()),
        "sign_agree_raw_advantage_branch_delta": _sign_agree_frac(raw_adv, branch),
        "corr_branch_delta_vs_delta_logprob": _safe_corr(branch.tolist(), dlogp.tolist()),
        "branch_delta_abs_mean": float(np.mean(np.abs(branch))) if branch.size > 0 else 0.0,
        "branch_delta_std": float(np.std(branch)) if branch.size > 0 else 0.0,
        "branch_delta_snr": _true_advantage_snr(branch),
        "mean_branch_delta_pos_adv": _safe_mean(branch[pos_adv]),
        "mean_branch_delta_neg_adv": _safe_mean(branch[neg_adv]),
        "mean_branch_delta_pos_raw_adv": _safe_mean(branch[pos_raw]),
        "mean_branch_delta_neg_raw_adv": _safe_mean(branch[neg_raw]),
        "raw_pos_branch_delta_positive_frac": _safe_frac(branch[pos_raw] > 0.0),
        "raw_neg_branch_delta_negative_frac": _safe_frac(branch[neg_raw] < 0.0),
        "branch_delta_pos_logprob_up_frac": _safe_frac(dlogp[pos_branch] > 0.0),
        "branch_delta_neg_logprob_down_frac": _safe_frac(dlogp[neg_branch] < 0.0),
        "mean_branch_delta_times_delta_logprob": _safe_mean(branch * dlogp),
        "mean_abs_delta_logprob": float(np.mean(np.abs(dlogp))) if dlogp.size > 0 else 0.0,
    }


def _normalize_advantages_like_training(
    advantages: np.ndarray,
    stage_ids: np.ndarray,
    *,
    stagewise_enabled: bool,
    normalize_enabled: bool = True,
) -> np.ndarray:
    adv = np.asarray(advantages, dtype=np.float32).copy()
    if adv.size <= 1:
        return adv
    if not bool(normalize_enabled):
        return adv
    if bool(stagewise_enabled):
        for stage_id in (0, 1, 2):
            idx = np.flatnonzero(np.asarray(stage_ids, dtype=np.int64) == int(stage_id))
            if idx.size <= 1:
                continue
            stage_adv = adv[idx]
            stage_std = float(np.std(stage_adv))
            adv[idx] = (stage_adv - float(np.mean(stage_adv))) / max(stage_std, 1.0e-8)
        return adv
    adv_std = float(np.std(adv))
    return (adv - float(np.mean(adv))) / max(adv_std, 1.0e-8)


def _default_sat_action(driver: StructuredControlDriver, cfg) -> np.ndarray:
    sat_states = driver.build_local_sat_states()
    subset_indices: list[int] = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if int(valid.numel()) > 0 else 0)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def _make_snapshot(driver: StructuredControlDriver, cfg) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    driver.begin_step()
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    z1 = driver.run_accel_stage(accel_zero)
    z2 = driver.run_sat_stage(_default_sat_action(driver, cfg))
    snapshot = driver.build_bw_stage_snapshot(z2)
    snapshot_state = driver.export_bw_stage_state()
    obs = {agent: driver.env._get_obs(idx) for idx, agent in enumerate(driver.env.agents)}
    return snapshot_state, snapshot, list(obs.values())


def _heuristic_action(obs_list, cfg) -> np.ndarray:
    return np.asarray(queue_aware_bw_policy(obs_list, cfg), dtype=np.float32)


@contextmanager
def _temporary_torch_seed(seed: int | None, device: torch.device):
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        cuda_states = torch.cuda.get_rng_state_all()
    torch.manual_seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _bw_action_from_actor(
    actor,
    snapshot,
    device: torch.device,
    deterministic: bool,
    *,
    sample_seed: int | None = None,
) -> np.ndarray:
    bw_states = build_local_bw_states_from_snapshot(snapshot)
    local_state = _to_device_dataclass(bw_states[0], device)
    with _temporary_torch_seed(sample_seed, device), torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _branch_parallel_signature(cfg, *, workers: int, backend: str) -> tuple[Any, ...]:
    return (
        int(getattr(cfg, "seed", 0) or 0),
        int(getattr(cfg, "T_steps", 0) or 0),
        int(getattr(cfg, "num_uav", 0) or 0),
        int(getattr(cfg, "num_gu", 0) or 0),
        int(getattr(cfg, "num_sat", 0) or 0),
        int(workers),
        str(backend or "sync").strip().lower(),
    )


def _get_cached_branch_probe_group(cfg, *, workers: int, backend: str):
    del cfg, workers, backend
    raise RuntimeError("legacy remote-driver BW probes have been removed; use native batch tensor probes.")


def _bw_local_batch_from_snapshots(
    snapshots: list[Any],
    device: torch.device,
):
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    local_rows = []
    for snapshot in snapshots:
        local_rows.extend(build_local_bw_states_from_snapshot(snapshot))
    local_batch_cpu = _collate_dataclass(local_rows, torch.device("cpu"))
    return _to_device_dataclass(local_batch_cpu, device)


def _sample_masked_dirichlet_numpy(
    mean: np.ndarray,
    *,
    kappa: float,
    mask: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1.0e-8,
) -> np.ndarray:
    action = np.zeros_like(mean, dtype=np.float32)
    valid_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if valid_idx.size <= 0:
        return action
    if valid_idx.size == 1:
        action[valid_idx[0]] = 1.0
        return action
    probs = np.asarray(mean[valid_idx], dtype=np.float64)
    probs = np.clip(probs, float(eps), None)
    probs = probs / max(float(probs.sum()), float(eps))
    concentration = np.clip(probs * float(max(kappa, eps)), float(eps), None)
    gamma = rng.gamma(shape=concentration, scale=1.0)
    gamma_sum = float(np.sum(gamma))
    if gamma_sum <= float(eps):
        action[valid_idx] = np.asarray(probs, dtype=np.float32)
    else:
        action[valid_idx] = np.asarray(gamma / gamma_sum, dtype=np.float32)
    return action


def _bw_seeded_actions_from_local_batch(
    actor,
    local_batch,
    *,
    sample_seeds: list[int | None],
) -> np.ndarray | None:
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        return None
    params_fn = getattr(bw_policy, "_params", None)
    if not callable(params_fn):
        return None

    with torch.inference_mode():
        _score, det_mean, _alpha, kappa, _tau, _valid_count, _latent_count, valid_mask = params_fn(local_batch)

    valid_mask_np = np.asarray(valid_mask.detach().cpu().numpy(), dtype=bool)
    row_count, user_dim = valid_mask_np.shape
    if len(sample_seeds) != row_count:
        raise ValueError(f"sample_seeds length {len(sample_seeds)} must match BW rows {row_count}.")

    kappa_np = np.asarray(kappa.detach().cpu().numpy(), dtype=np.float64).reshape(row_count)
    mean_np = np.asarray(det_mean.detach().cpu().numpy(), dtype=np.float32).reshape(row_count, user_dim)

    actions = np.zeros((row_count, user_dim), dtype=np.float32)
    for row_idx, seed in enumerate(sample_seeds):
        rng = np.random.default_rng(None if seed is None else int(seed))
        row_action = _sample_masked_dirichlet_numpy(
            mean_np[row_idx],
            kappa=float(kappa_np[row_idx]),
            mask=valid_mask_np[row_idx],
            rng=rng,
        )
        actions[row_idx] = np.asarray(row_action, dtype=np.float32)
    return actions


def _bw_actions_from_snapshots_batch(
    actor,
    snapshots: list[Any],
    device: torch.device,
    *,
    deterministic: bool,
    sample_seed: int | None = None,
    sample_seeds: list[int | None] | None = None,
) -> list[np.ndarray]:
    if not snapshots:
        return []
    if getattr(snapshots[0], "ego_features", None) is None:
        raise RuntimeError(
            "BW update-direction probes require BwStageSnapshot objects carrying redesigned "
            "LocalBwState tensors. Create them with StructuredControlDriver.build_bw_stage_snapshot()."
        )
    num_uav = int(np.asarray(getattr(snapshots[0], "ego_features")).shape[0])
    flat_sample_seeds = None
    if sample_seeds is not None:
        if len(sample_seeds) == len(snapshots):
            flat_sample_seeds = [
                seed
                for seed in sample_seeds
                for _ in range(max(int(num_uav), 1))
            ]
        elif len(sample_seeds) == len(snapshots) * max(int(num_uav), 1):
            flat_sample_seeds = list(sample_seeds)
        else:
            raise ValueError(
                f"sample_seeds length {len(sample_seeds)} must match snapshots ({len(snapshots)}) "
                f"or flattened BW rows ({len(snapshots) * max(int(num_uav), 1)})."
            )
    local_batch = _bw_local_batch_from_snapshots(snapshots, device)
    actions = None
    if not bool(deterministic) and flat_sample_seeds is not None:
        actions = _bw_seeded_actions_from_local_batch(
            actor,
            local_batch,
            sample_seeds=flat_sample_seeds,
        )
    if actions is None:
        with _temporary_torch_seed(sample_seed, device), torch.inference_mode():
            out = actor.act_bw(local_batch, deterministic=bool(deterministic))
        actions = np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)
    user_dim = int(actions.shape[-1])
    if actions.shape[0] != len(snapshots) * num_uav:
        raise ValueError(
            f"Unexpected BW action batch shape {tuple(actions.shape)} for {len(snapshots)} snapshots and num_uav={num_uav}."
        )
    actions = actions.reshape(len(snapshots), num_uav, user_dim)
    return [np.asarray(actions[idx], dtype=np.float32) for idx in range(actions.shape[0])]


def _bw_det_actions_from_snapshot_states_parallel(
    *,
    snapshot_states: list[dict[str, Any]],
    cfg,
    actor,
    device: torch.device,
    workers: int,
    backend: str,
) -> list[np.ndarray]:
    if not snapshot_states:
        return []
    if int(workers) <= 1 or str(backend or "sync").strip().lower() != "subproc":
        probe_driver = _make_probe_driver(cfg)
        try:
            actions: list[np.ndarray] = []
            for snapshot_state in snapshot_states:
                snapshot = probe_driver.load_bw_stage_state(snapshot_state)
                actions.append(_bw_action_from_actor(actor, snapshot, device, deterministic=True))
            return actions
        finally:
            _close_probe_driver(probe_driver)
    group = _get_cached_branch_probe_group(cfg, workers=int(min(workers, len(snapshot_states))), backend=str(backend))
    actions: list[np.ndarray] = []
    worker_count = len(group)
    for start in range(0, len(snapshot_states), worker_count):
        chunk_states = snapshot_states[start : start + worker_count]
        chunk_slots = list(range(len(chunk_states)))
        snapshots = group.load_bw_stage_state_many(chunk_states, indices=chunk_slots)
        actions.extend(
            _bw_actions_from_snapshots_batch(
                actor,
                list(snapshots),
                device,
                deterministic=True,
            )
        )
    return actions


def _bw_probe_actor_state_dict_cpu(actor) -> dict[str, torch.Tensor]:
    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None:
        raise RuntimeError("Structured actor is missing bw_policy for branch probe sync.")
    return {
        str(name): tensor.detach().cpu()
        for name, tensor in bw_policy.state_dict().items()
    }


def _rollout_branch_pairs_fused_parallel(
    *,
    snapshot_states: list[dict[str, Any]],
    sampled_actions: list[np.ndarray],
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
    sample_seed_bases: list[int] | None,
    workers: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_rollouts = int(len(snapshot_states))
    sampled_rewards = np.zeros((num_rollouts,), dtype=np.float32)
    ref_rewards = np.zeros((num_rollouts,), dtype=np.float32)
    if num_rollouts <= 0:
        return sampled_rewards, ref_rewards
    if int(workers) <= 1 or str(backend or "sync").strip().lower() != "subproc":
        for idx in range(num_rollouts):
            sample_seed_base = None if sample_seed_bases is None else int(sample_seed_bases[idx])
            snapshot_state = snapshot_states[idx]
            sampled_action = np.asarray(sampled_actions[idx], dtype=np.float32)
            probe_driver = _make_probe_driver(cfg)
            try:
                snapshot = probe_driver.load_bw_stage_state(snapshot_state)
                ref_action = _bw_action_from_actor(actor, snapshot, device, deterministic=True)
                sampled_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=probe_driver,
                    snapshot_state=snapshot_state,
                    first_action=sampled_action,
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    k_steps=int(k_steps),
                    gamma=float(gamma),
                    follow_deterministic=bool(follow_deterministic),
                    sample_seed_base=sample_seed_base,
                )
                ref_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=probe_driver,
                    snapshot_state=snapshot_state,
                    first_action=ref_action,
                    cfg=cfg,
                    actor=actor,
                    device=device,
                    k_steps=int(k_steps),
                    gamma=float(gamma),
                    follow_deterministic=bool(follow_deterministic),
                    sample_seed_base=sample_seed_base,
                )
                sampled_rewards[idx] = float(sampled_roll["reward"])
                ref_rewards[idx] = float(ref_roll["reward"])
            finally:
                _close_probe_driver(probe_driver)
        return sampled_rewards, ref_rewards

    worker_count = int(min(int(workers), num_rollouts))
    group = _get_cached_branch_probe_group(cfg, workers=worker_count, backend=str(backend))
    group.set_bw_probe_actor_state_many(_bw_probe_actor_state_dict_cpu(actor), indices=list(range(worker_count)))
    for start in range(0, num_rollouts, worker_count):
        chunk_states = snapshot_states[start : start + worker_count]
        chunk_actions = [np.asarray(action, dtype=np.float32) for action in sampled_actions[start : start + worker_count]]
        chunk_seed_bases = None if sample_seed_bases is None else sample_seed_bases[start : start + worker_count]
        chunk_len = len(chunk_states)
        chunk_slots = list(range(chunk_len))
        task_batches: list[list[dict[str, Any]]] = []
        for local_idx in range(chunk_len):
            task_batches.append(
                [
                    {
                        "snapshot_state": chunk_states[local_idx],
                        "sampled_action": chunk_actions[local_idx],
                        "k_steps": int(k_steps),
                        "gamma": float(gamma),
                        "follow_deterministic": bool(follow_deterministic),
                        "sample_seed_base": None if chunk_seed_bases is None else int(chunk_seed_bases[local_idx]),
                    }
                ]
            )
        chunk_results = group.rollout_bw_branch_tasks_many(task_batches, indices=chunk_slots)
        for local_idx, worker_results in enumerate(chunk_results):
            if not worker_results:
                continue
            result = worker_results[0]
            sampled_rewards[start + local_idx] = float(result["sampled_reward"])
            ref_rewards[start + local_idx] = float(result["ref_reward"])
    return sampled_rewards, ref_rewards


def _rollout_from_snapshot_states_parallel(
    *,
    snapshot_states: list[dict[str, Any]],
    first_actions: list[np.ndarray],
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
    sample_seed_bases: list[int] | None,
    workers: int,
    backend: str,
) -> np.ndarray:
    num_rollouts = int(len(snapshot_states))
    if num_rollouts <= 0:
        return np.zeros((0,), dtype=np.float32)
    if int(workers) <= 1 or str(backend or "sync").strip().lower() != "subproc":
        rewards = np.zeros((num_rollouts,), dtype=np.float32)
        for idx, snapshot_state in enumerate(snapshot_states):
            roll = _rollout_from_snapshot(
                snapshot_state=snapshot_state,
                first_action=np.asarray(first_actions[idx], dtype=np.float32),
                cfg=cfg,
                actor=actor,
                device=device,
                k_steps=int(k_steps),
                gamma=float(gamma),
                follow_deterministic=bool(follow_deterministic),
                sample_seed_base=None if sample_seed_bases is None else int(sample_seed_bases[idx]),
            )
            rewards[idx] = float(roll["reward"])
        return rewards

    raise RuntimeError(
        "subproc BW branch rollout used the deleted legacy stage-many API; "
        "use the native main-kernel branch rollout path instead."
    )


def _rollout_from_snapshot(
    *,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
    sample_seed_base: int | None = None,
) -> dict[str, float]:
    def _run_with_driver(probe_driver: StructuredControlDriver) -> dict[str, float]:
        return _rollout_from_snapshot_with_driver(
            probe_driver=probe_driver,
            snapshot_state=snapshot_state,
            first_action=first_action,
            cfg=cfg,
            actor=actor,
            device=device,
            k_steps=k_steps,
            gamma=gamma,
            follow_deterministic=follow_deterministic,
            sample_seed_base=sample_seed_base,
        )

    probe_driver = _make_probe_driver(cfg)
    try:
        return _run_with_driver(probe_driver)
    finally:
        _close_probe_driver(probe_driver)


def _rollout_from_snapshot_with_driver(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    k_steps: int,
    gamma: float,
    follow_deterministic: bool,
    sample_seed_base: int | None = None,
) -> dict[str, float]:
    probe_driver.load_bw_stage_state(snapshot_state)
    discounted_reward = 0.0
    discounted_weighted = 0.0
    discount = 1.0
    action = np.asarray(first_action, dtype=np.float32)
    for step in range(int(k_steps)):
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
        reward = float(next(iter(step_result.rewards.values())))
        weighted = float(getattr(step_result, "bw_weighted_workload_delta_reward", 0.0) or 0.0)
        discounted_reward += discount * reward
        discounted_weighted += discount * weighted
        done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
        if done or step == int(k_steps) - 1:
            break
        probe_driver.begin_step()
        accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        z1 = probe_driver.run_accel_stage(accel_zero)
        z2 = probe_driver.run_sat_stage(_default_sat_action(probe_driver, cfg))
        action = _bw_action_from_actor(
            actor,
            probe_driver.build_bw_stage_snapshot(z2),
            device,
            deterministic=follow_deterministic,
            sample_seed=None if follow_deterministic or sample_seed_base is None else int(sample_seed_base) + int(step) + 1,
        )
        discount *= float(gamma)
    return {
        "reward": float(discounted_reward),
        "weighted": float(discounted_weighted),
    }


def collect_bw_snapshot_panel(cfg, *, episodes: int, states: int, seed: int) -> list[dict[str, Any]]:
    if int(cfg.num_uav) != 1:
        raise ValueError("BW update-direction probe currently requires num_uav == 1.")
    driver = _make_probe_driver(cfg)
    entries: list[dict[str, Any]] = []
    try:
        for ep in range(int(episodes)):
            driver.env.reset(seed=int(seed) + ep)
            done = False
            while not done and len(entries) < int(states):
                snapshot_state, snapshot, obs_list = _make_snapshot(driver, cfg)
                heuristic_action = _heuristic_action(obs_list, cfg)
                entries.append(
                    {
                        "episode": int(ep),
                        "t": int(snapshot_state.get("env_state", {}).get("t", 0)),
                        "snapshot_state": snapshot_state,
                        "snapshot": snapshot,
                        "heuristic_action": heuristic_action,
                    }
                )
                step_result = driver.execute_stage_bw_and_step(heuristic_action)
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
                if len(entries) >= int(states):
                    break
    finally:
        _close_probe_driver(driver)
    return entries


def evaluate_bw_snapshot_panel(
    actor,
    panel: list[dict[str, Any]],
    cfg,
    device: torch.device,
    *,
    k_steps: int,
    deterministic: bool = True,
    actor_policy_samples: int = 1,
    seed_base: int | None = None,
) -> dict[str, Any]:
    reward_values: list[float] = []
    weighted_values: list[float] = []
    heuristic_reward_values: list[float] = []
    heuristic_weighted_values: list[float] = []
    l1_to_heur_values: list[float] = []
    heuristic_beats_actor: list[float] = []
    state_actor_reward_means: list[float] = []
    state_actor_weighted_means: list[float] = []
    state_heuristic_reward_means: list[float] = []
    state_heuristic_weighted_means: list[float] = []
    rows: list[dict[str, Any]] = []
    policy_samples = 1 if bool(deterministic) else max(int(actor_policy_samples), 1)
    for entry_idx, entry in enumerate(panel):
        snapshot = entry["snapshot"]
        snapshot_state = entry["snapshot_state"]
        heuristic_action = np.asarray(entry["heuristic_action"], dtype=np.float32)
        actor_roll_rewards: list[float] = []
        actor_roll_weighted: list[float] = []
        heuristic_roll_rewards: list[float] = []
        heuristic_roll_weighted: list[float] = []
        actor_action_ref = None
        for sample_idx in range(int(policy_samples)):
            sample_seed = None
            if not bool(deterministic):
                sample_seed = int(0 if seed_base is None else seed_base) + int(entry_idx) * 100_000 + int(sample_idx) * 1_000
            actor_action = _bw_action_from_actor(
                actor,
                snapshot,
                device,
                deterministic=bool(deterministic),
                sample_seed=sample_seed,
            )
            if actor_action_ref is None:
                actor_action_ref = np.asarray(actor_action, dtype=np.float32)
            actor_roll = _rollout_from_snapshot(
                snapshot_state=snapshot_state,
                first_action=actor_action,
                cfg=cfg,
                actor=actor,
                device=device,
                k_steps=int(k_steps),
                gamma=float(cfg.gamma),
                follow_deterministic=bool(deterministic),
                sample_seed_base=None if bool(deterministic) else sample_seed,
            )
            heuristic_roll = _rollout_from_snapshot(
                snapshot_state=snapshot_state,
                first_action=heuristic_action,
                cfg=cfg,
                actor=actor,
                device=device,
                k_steps=int(k_steps),
                gamma=float(cfg.gamma),
                follow_deterministic=bool(deterministic),
                sample_seed_base=None if bool(deterministic) else sample_seed,
            )
            actor_roll_rewards.append(float(actor_roll["reward"]))
            actor_roll_weighted.append(float(actor_roll["weighted"]))
            heuristic_roll_rewards.append(float(heuristic_roll["reward"]))
            heuristic_roll_weighted.append(float(heuristic_roll["weighted"]))
        actor_reward_mean = float(np.mean(np.asarray(actor_roll_rewards, dtype=np.float64))) if actor_roll_rewards else 0.0
        actor_weighted_mean = float(np.mean(np.asarray(actor_roll_weighted, dtype=np.float64))) if actor_roll_weighted else 0.0
        heuristic_reward_mean = float(np.mean(np.asarray(heuristic_roll_rewards, dtype=np.float64))) if heuristic_roll_rewards else 0.0
        heuristic_weighted_mean = (
            float(np.mean(np.asarray(heuristic_roll_weighted, dtype=np.float64))) if heuristic_roll_weighted else 0.0
        )
        valid_mask = np.asarray(snapshot.bw_valid_mask, dtype=bool)
        actor_action_l1 = heuristic_action if actor_action_ref is None else actor_action_ref
        l1_value = float(np.abs(actor_action_l1 - heuristic_action)[valid_mask].sum()) if np.any(valid_mask) else 0.0
        reward_values.append(actor_reward_mean)
        weighted_values.append(actor_weighted_mean)
        heuristic_reward_values.append(heuristic_reward_mean)
        heuristic_weighted_values.append(heuristic_weighted_mean)
        state_actor_reward_means.append(actor_reward_mean)
        state_actor_weighted_means.append(actor_weighted_mean)
        state_heuristic_reward_means.append(heuristic_reward_mean)
        state_heuristic_weighted_means.append(heuristic_weighted_mean)
        l1_to_heur_values.append(l1_value)
        heuristic_beats_actor.append(float(heuristic_reward_mean > actor_reward_mean + 1.0e-9))
        if len(rows) < 6:
            rows.append(
                {
                    "episode": int(entry["episode"]),
                    "t": int(entry["t"]),
                    "actor_reward": float(actor_reward_mean),
                    "heuristic_reward": float(heuristic_reward_mean),
                    "actor_weighted": float(actor_weighted_mean),
                    "heuristic_weighted": float(heuristic_weighted_mean),
                    "l1_to_heur": float(l1_value),
                }
            )
    return {
        "actor_reward": _summarize(reward_values),
        "actor_weighted": _summarize(weighted_values),
        "heuristic_reward": _summarize(heuristic_reward_values),
        "heuristic_weighted": _summarize(heuristic_weighted_values),
        "l1_to_heur": _summarize(l1_to_heur_values),
        "heuristic_beats_actor_frac": float(np.mean(heuristic_beats_actor)) if heuristic_beats_actor else 0.0,
        "actor_policy_mode": "deterministic" if bool(deterministic) else "stochastic",
        "actor_policy_samples": int(policy_samples),
        "state_actor_reward_means": state_actor_reward_means,
        "state_actor_weighted_means": state_actor_weighted_means,
        "state_heuristic_reward_means": state_heuristic_reward_means,
        "state_heuristic_weighted_means": state_heuristic_weighted_means,
        "examples": rows,
    }


def build_bw_advantage_probe_context(
    learner,
    buffer,
    bootstrap_world_state,
    *,
    sample_limit: int,
    sample_seed: int,
) -> dict[str, Any] | None:
    rollout_views = buffer.build_rollout_views(device=learner.device)
    training_view = rollout_views.training_view
    return_view = rollout_views.return_view
    bw_batch = training_view.stage_batches.get(2)
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    if stage_ids.size == 0:
        return None
    gae = learner.compute_returns_and_advantages(buffer, bootstrap_world_state)
    advantages = _normalize_advantages_like_training(
        np.asarray(gae["advantages"], dtype=np.float32),
        stage_ids,
        stagewise_enabled=bool(getattr(learner, "stagewise_advantage_norm_enabled", False)),
        normalize_enabled=bool(getattr(learner, "actor_advantage_normalize_enabled", True)),
    )
    if bw_batch is None or bw_batch.num_samples <= 0:
        return None
    values_for_advantage = np.asarray(return_view.values, dtype=np.float32)
    returns_for_advantage = np.asarray(gae["returns"], dtype=np.float32)
    actor_advantage_override = getattr(learner, "last_update_actor_advantages_np", None)
    actor_values_override = getattr(learner, "last_update_actor_values_np", None)
    actor_returns_override = getattr(learner, "last_update_returns_np", None)
    if actor_advantage_override is not None:
        override_adv = np.asarray(actor_advantage_override, dtype=np.float32)
        if override_adv.shape == advantages.shape:
            advantages = override_adv
            if actor_values_override is not None:
                override_values = np.asarray(actor_values_override, dtype=np.float32)
                if override_values.shape == values_for_advantage.shape:
                    values_for_advantage = override_values
            if actor_returns_override is not None:
                override_returns = np.asarray(actor_returns_override, dtype=np.float32)
                if override_returns.shape == returns_for_advantage.shape:
                    returns_for_advantage = override_returns
    bw_indices = [int(idx) for idx in np.asarray(bw_batch.transition_indices, dtype=np.int64).tolist()]

    def _build_probe_batch(indices: list[int]) -> dict[str, Any] | None:
        if not indices:
            return None
        position_lookup = {int(transition_idx): pos for pos, transition_idx in enumerate(bw_indices)}
        positions = [int(position_lookup[int(index)]) for index in indices]
        pos_tensor = torch.as_tensor(positions, dtype=torch.long, device=bw_batch.actions.device)
        local_batch = _select_dataclass_batch(bw_batch.local_batch, positions, learner.device)
        joint_actions = bw_batch.actions.index_select(0, pos_tensor).to(learner.device)
        old_logprobs = bw_batch.old_logprobs.index_select(0, pos_tensor.to(bw_batch.old_logprobs.device)).to(learner.device)
        index_array = np.asarray(indices, dtype=np.int64)
        return {
            "indices": list(indices),
            "sample_count": int(len(indices)),
            "num_agents": int(joint_actions.shape[1]),
            "local_batch": local_batch,
            "joint_actions": joint_actions,
            "old_logprobs": old_logprobs,
            "advantages": torch.from_numpy(advantages[index_array]).to(learner.device),
            "values": torch.from_numpy(values_for_advantage[index_array]).to(learner.device),
            "returns": torch.from_numpy(returns_for_advantage[index_array]).to(learner.device),
        }

    logprob_batch = _build_probe_batch(bw_indices)
    if logprob_batch is None:
        return None

    candidate_indices = [
        int(transition_idx)
        for transition_idx, snapshot_state in zip(bw_indices, bw_batch.bw_stage_states or [])
        if snapshot_state is not None
    ]
    selected: list[int] = []
    action_gap_batch = None
    if candidate_indices:
        rng = np.random.default_rng(int(sample_seed))
        sample_count = min(max(int(sample_limit), 1), len(candidate_indices))
        selected = sorted(rng.choice(candidate_indices, size=sample_count, replace=False).astype(np.int64).tolist())
        action_gap_batch = _build_probe_batch(selected)
    return {
        "selected_indices": list(bw_indices),
        "sample_count": int(logprob_batch["sample_count"]),
        "num_agents": int(logprob_batch["num_agents"]),
        "local_batch": logprob_batch["local_batch"],
        "joint_actions": logprob_batch["joint_actions"],
        "old_logprobs": logprob_batch["old_logprobs"],
        "advantages": logprob_batch["advantages"],
        "values": logprob_batch["values"],
        "returns": logprob_batch["returns"],
        "logprob_sample_count": int(logprob_batch["sample_count"]),
        "action_gap_selected_indices": list(selected),
        "action_gap_sample_count": 0 if action_gap_batch is None else int(action_gap_batch["sample_count"]),
        "action_gap_joint_actions": None if action_gap_batch is None else action_gap_batch["joint_actions"],
        "action_gap_advantages": None if action_gap_batch is None else action_gap_batch["advantages"],
        "action_gap_values": None if action_gap_batch is None else action_gap_batch["values"],
        "action_gap_returns": None if action_gap_batch is None else action_gap_batch["returns"],
        "snapshot_states": [
            (bw_batch.bw_stage_states or [])[int(bw_indices.index(int(i)))]
            for i in selected
        ],
        "action_gap_positions": [int(bw_indices.index(int(i))) for i in selected],
    }


def compute_bw_branch_advantage_override(
    *,
    learner,
    buffer,
    bootstrap_world_state,
    cfg,
    device: torch.device,
    horizon: int,
    branch_samples: int,
    branch_seed: int = 0,
    ref_mode: str = "deterministic",
    follow_policy_mode: str = "stochastic",
    normalize: bool = False,
    include_default_advantage_stats: bool = True,
) -> dict[str, Any] | None:
    horizon_eff = max(int(horizon), 1)
    ref_mode_l = str(ref_mode or "deterministic").strip().lower()
    if ref_mode_l not in {"deterministic"}:
        raise ValueError("bw_actor_branch_ref_mode currently supports only 'deterministic'.")
    follow_mode_l = str(follow_policy_mode or "stochastic").strip().lower()
    if follow_mode_l not in {"deterministic", "stochastic"}:
        raise ValueError("bw_actor_branch_follow_policy_mode must be one of {'deterministic', 'stochastic'}.")
    branch_samples_eff = max(int(branch_samples), 1)
    if follow_mode_l == "deterministic" and branch_samples_eff > 1:
        branch_samples_eff = 1

    rollout_views = buffer.build_rollout_views(device=device)
    training_view = rollout_views.training_view
    return_view = rollout_views.return_view
    bw_batch = training_view.stage_batches.get(2)
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    if stage_ids.size == 0:
        return None
    if bw_batch is None or bw_batch.num_samples <= 0:
        return None
    bw_indices = [int(idx) for idx in np.asarray(bw_batch.transition_indices, dtype=np.int64).tolist()]
    snapshot_states = list(bw_batch.bw_stage_states or [])
    if any(state is None for state in snapshot_states):
        missing = int(sum(state is None for state in snapshot_states))
        raise RuntimeError(
            "branch_delta actor override requires BW stage snapshots in the rollout buffer, "
            f"but {missing} BW samples were missing snapshot state."
        )

    default_bw_adv = np.zeros((len(bw_indices),), dtype=np.float32)
    default_bw_raw_adv = np.zeros((len(bw_indices),), dtype=np.float32)
    if bool(include_default_advantage_stats):
        gae = learner.compute_returns_and_advantages(buffer, bootstrap_world_state)
        default_adv = _normalize_advantages_like_training(
            np.asarray(gae["advantages"], dtype=np.float32),
            stage_ids,
            stagewise_enabled=bool(getattr(learner, "stagewise_advantage_norm_enabled", False)),
            normalize_enabled=bool(getattr(learner, "actor_advantage_normalize_enabled", True)),
        )
        default_raw_adv = np.asarray(gae["advantages"], dtype=np.float32)
        bw_index_array = np.asarray(bw_indices, dtype=np.int64)
        default_bw_adv = default_adv[bw_index_array]
        default_bw_raw_adv = default_raw_adv[bw_index_array]
    branch_raw = np.zeros((len(bw_indices),), dtype=np.float32)
    sampled_q = np.zeros_like(branch_raw)
    ref_q = np.zeros_like(branch_raw)
    parallel_envs = max(int(getattr(cfg, "bw_actor_branch_parallel_envs", 0) or 0), 0)
    parallel_backend = str(getattr(cfg, "bw_actor_branch_parallel_backend", "sync") or "sync").strip().lower()
    use_worker_fused_rollout = bool(getattr(cfg, "bw_actor_branch_worker_fused_rollout_enabled", False))
    sampled_actions = [
        np.asarray(bw_batch.actions[row_idx].detach().cpu(), dtype=np.float32)
        for row_idx in range(bw_batch.num_samples)
    ]
    det_actions = None if use_worker_fused_rollout else _bw_det_actions_from_snapshot_states_parallel(
        snapshot_states=snapshot_states,
        cfg=cfg,
        actor=learner.actor,
        device=device,
        workers=int(parallel_envs),
        backend=str(parallel_backend),
    )
    sampled_q_acc = np.zeros_like(branch_raw, dtype=np.float64)
    ref_q_acc = np.zeros_like(branch_raw, dtype=np.float64)
    for branch_sample_idx in range(int(branch_samples_eff)):
        branch_seed_bases = [
            int(branch_seed)
            + int(row_idx) * 1_000_000
            + int(horizon_eff) * 10_000
            + int(branch_sample_idx) * 100
            for row_idx in range(len(bw_indices))
        ]
        if use_worker_fused_rollout:
            sampled_rewards, ref_rewards = _rollout_branch_pairs_fused_parallel(
                snapshot_states=snapshot_states,
                sampled_actions=sampled_actions,
                cfg=cfg,
                actor=learner.actor,
                device=device,
                k_steps=int(horizon_eff),
                gamma=float(cfg.gamma),
                follow_deterministic=(follow_mode_l == "deterministic"),
                sample_seed_bases=None if follow_mode_l == "deterministic" else branch_seed_bases,
                workers=int(parallel_envs),
                backend=str(parallel_backend),
            )
        else:
            sampled_rewards = _rollout_from_snapshot_states_parallel(
                snapshot_states=snapshot_states,
                first_actions=sampled_actions,
                cfg=cfg,
                actor=learner.actor,
                device=device,
                k_steps=int(horizon_eff),
                gamma=float(cfg.gamma),
                follow_deterministic=(follow_mode_l == "deterministic"),
                sample_seed_bases=None if follow_mode_l == "deterministic" else branch_seed_bases,
                workers=int(parallel_envs),
                backend=str(parallel_backend),
            )
            ref_rewards = _rollout_from_snapshot_states_parallel(
                snapshot_states=snapshot_states,
                first_actions=det_actions,
                cfg=cfg,
                actor=learner.actor,
                device=device,
                k_steps=int(horizon_eff),
                gamma=float(cfg.gamma),
                follow_deterministic=(follow_mode_l == "deterministic"),
                sample_seed_bases=None if follow_mode_l == "deterministic" else branch_seed_bases,
                workers=int(parallel_envs),
                backend=str(parallel_backend),
            )
        sampled_q_acc += np.asarray(sampled_rewards, dtype=np.float64)
        ref_q_acc += np.asarray(ref_rewards, dtype=np.float64)
    sampled_q = (sampled_q_acc / float(branch_samples_eff)).astype(np.float32, copy=False)
    ref_q = (ref_q_acc / float(branch_samples_eff)).astype(np.float32, copy=False)
    branch_raw = (sampled_q - ref_q).astype(np.float32, copy=False)

    branch_adv = branch_raw.copy()
    if bool(normalize) and branch_adv.size > 1:
        branch_adv = _normalize_advantages_like_training(
            branch_adv,
            np.full(branch_adv.shape, 2, dtype=np.int64),
            stagewise_enabled=False,
            normalize_enabled=True,
        )

    return {
        "stage_id": 2,
        "transition_indices": list(bw_indices),
        "advantages": branch_adv.astype(np.float32, copy=False),
        "raw_branch_advantages": branch_raw.astype(np.float32, copy=False),
        "sampled_q": sampled_q.astype(np.float32, copy=False),
        "ref_q": ref_q.astype(np.float32, copy=False),
        "default_advantages": default_bw_adv.astype(np.float32, copy=False),
        "default_raw_advantages": default_bw_raw_adv.astype(np.float32, copy=False),
        "sample_count": int(len(bw_indices)),
        "branch_samples": int(branch_samples_eff),
        "horizon": int(horizon_eff),
        "normalize": bool(normalize),
        "corr_default_vs_branch": _safe_corr(default_bw_adv.tolist(), branch_raw.tolist()),
        "corr_raw_default_vs_branch": _safe_corr(default_bw_raw_adv.tolist(), branch_raw.tolist()),
        "sign_agree_default_vs_branch": _sign_agree_frac(default_bw_adv, branch_raw),
        "sign_agree_raw_default_vs_branch": _sign_agree_frac(default_bw_raw_adv, branch_raw),
        "branch_abs_mean": float(np.mean(np.abs(branch_raw))) if branch_raw.size > 0 else 0.0,
        "branch_std": float(np.std(branch_raw)) if branch_raw.size > 0 else 0.0,
        "branch_mean": float(np.mean(branch_raw)) if branch_raw.size > 0 else 0.0,
    }


def compute_bw_true_advantage_override(
    *,
    learner,
    buffer,
    bootstrap_world_state,
    cfg,
    device: torch.device,
    k_steps: int,
    true_mc_samples: int,
    true_mc_seed: int = 0,
) -> dict[str, Any] | None:
    horizon_eff = max(int(k_steps), 1)
    mc_samples_eff = max(int(true_mc_samples), 1)

    rollout_views = buffer.build_rollout_views(device=device)
    training_view = rollout_views.training_view
    return_view = rollout_views.return_view
    bw_batch = training_view.stage_batches.get(2)
    stage_ids = np.asarray(return_view.stage_ids, dtype=np.int64)
    if stage_ids.size == 0:
        return None
    if bw_batch is None or bw_batch.num_samples <= 0:
        return None
    bw_indices = [int(idx) for idx in np.asarray(bw_batch.transition_indices, dtype=np.int64).tolist()]
    snapshot_states = list(bw_batch.bw_stage_states or [])
    if any(state is None for state in snapshot_states):
        missing = int(sum(state is None for state in snapshot_states))
        raise RuntimeError(
            "true_adv_mc actor override requires BW stage snapshots in the rollout buffer, "
            f"but {missing} BW samples were missing snapshot state."
        )

    gae = learner.compute_returns_and_advantages(buffer, bootstrap_world_state)
    default_adv = _normalize_advantages_like_training(
        np.asarray(gae["advantages"], dtype=np.float32),
        stage_ids,
        stagewise_enabled=bool(getattr(learner, "stagewise_advantage_norm_enabled", False)),
        normalize_enabled=bool(getattr(learner, "actor_advantage_normalize_enabled", True)),
    )
    default_raw_adv = np.asarray(gae["advantages"], dtype=np.float32)
    true_adv_raw = np.zeros((len(bw_indices),), dtype=np.float32)
    sampled_q = np.zeros_like(true_adv_raw)
    policy_q = np.zeros_like(true_adv_raw)

    probe_driver = _make_probe_driver(cfg)
    try:
        for row_idx, snapshot_state in enumerate(snapshot_states):
            snapshot = probe_driver.load_bw_stage_state(snapshot_state)
            sampled_action = np.asarray(bw_batch.actions[row_idx].detach().cpu(), dtype=np.float32)
            sampled_qs: list[float] = []
            policy_qs: list[float] = []
            for mc_idx in range(int(mc_samples_eff)):
                mc_seed = int(true_mc_seed) + int(row_idx) * 100_000 + int(mc_idx) * 1_000
                sampled_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=probe_driver,
                    snapshot_state=snapshot_state,
                    first_action=np.asarray(sampled_action, dtype=np.float32),
                    cfg=cfg,
                    actor=learner.actor,
                    device=device,
                    k_steps=int(horizon_eff),
                    gamma=float(cfg.gamma),
                    follow_deterministic=False,
                    sample_seed_base=int(mc_seed),
                )
                policy_action = _bw_action_from_actor(
                    learner.actor,
                    snapshot,
                    device,
                    deterministic=False,
                    sample_seed=int(mc_seed),
                )
                policy_roll = _rollout_from_snapshot_with_driver(
                    probe_driver=probe_driver,
                    snapshot_state=snapshot_state,
                    first_action=np.asarray(policy_action, dtype=np.float32),
                    cfg=cfg,
                    actor=learner.actor,
                    device=device,
                    k_steps=int(horizon_eff),
                    gamma=float(cfg.gamma),
                    follow_deterministic=False,
                    sample_seed_base=int(mc_seed),
                )
                sampled_qs.append(float(sampled_roll["reward"]))
                policy_qs.append(float(policy_roll["reward"]))
            sampled_q[row_idx] = float(_safe_mean(np.asarray(sampled_qs, dtype=np.float64)))
            policy_q[row_idx] = float(_safe_mean(np.asarray(policy_qs, dtype=np.float64)))
            true_adv_raw[row_idx] = float(sampled_q[row_idx] - policy_q[row_idx])
    finally:
        _close_probe_driver(probe_driver)

    bw_index_array = np.asarray(bw_indices, dtype=np.int64)
    default_bw_adv = default_adv[bw_index_array]
    default_bw_raw_adv = default_raw_adv[bw_index_array]
    return {
        "stage_id": 2,
        "transition_indices": list(bw_indices),
        "advantages": true_adv_raw.astype(np.float32, copy=False),
        "raw_true_advantages": true_adv_raw.astype(np.float32, copy=False),
        "sampled_q": sampled_q.astype(np.float32, copy=False),
        "policy_q": policy_q.astype(np.float32, copy=False),
        "default_advantages": default_bw_adv.astype(np.float32, copy=False),
        "default_raw_advantages": default_bw_raw_adv.astype(np.float32, copy=False),
        "sample_count": int(len(bw_indices)),
        "true_mc_samples": int(mc_samples_eff),
        "horizon": int(horizon_eff),
        "corr_default_vs_true_adv": _safe_corr(default_bw_adv.tolist(), true_adv_raw.tolist()),
        "corr_raw_default_vs_true_adv": _safe_corr(default_bw_raw_adv.tolist(), true_adv_raw.tolist()),
        "sign_agree_default_vs_true_adv": _sign_agree_frac(default_bw_adv, true_adv_raw),
        "sign_agree_raw_default_vs_true_adv": _sign_agree_frac(default_bw_raw_adv, true_adv_raw),
        "true_adv_abs_mean": float(np.mean(np.abs(true_adv_raw))) if true_adv_raw.size > 0 else 0.0,
        "true_adv_std": float(np.std(true_adv_raw)) if true_adv_raw.size > 0 else 0.0,
        "true_adv_mean": float(np.mean(true_adv_raw)) if true_adv_raw.size > 0 else 0.0,
    }


def evaluate_bw_advantage_alignment(
    *,
    pre_actor,
    post_actor,
    probe_context: dict[str, Any],
    cfg,
    device: torch.device,
    k_steps: int,
    true_mc_enabled: bool = False,
    true_mc_samples: int = 0,
    true_mc_seed: int = 0,
    branch_enabled: bool = False,
    branch_horizons: list[int] | tuple[int, ...] | None = None,
    branch_samples: int = 0,
    branch_seed: int = 0,
    branch_ref_mode: str = "deterministic",
    branch_follow_policy_mode: str = "stochastic",
) -> dict[str, Any]:
    local_batch = probe_context["local_batch"]
    joint_actions = probe_context["joint_actions"]
    old_logprobs = probe_context["old_logprobs"]
    advantages_t = probe_context["advantages"]
    num_samples = int(joint_actions.shape[0])
    num_agents = int(probe_context["num_agents"])
    flat_action = joint_actions.to(device).reshape(num_samples * num_agents, -1)
    with torch.inference_mode():
        pre_eval = pre_actor.evaluate_bw(local_batch, flat_action)
        post_eval = post_actor.evaluate_bw(local_batch, flat_action)
    pre_joint_logprob = pre_eval.logprob.reshape(num_samples, num_agents).sum(dim=1)
    post_joint_logprob = post_eval.logprob.reshape(num_samples, num_agents).sum(dim=1)
    replay_abs_diff = (pre_joint_logprob - old_logprobs).abs().detach().cpu().numpy().astype(np.float64)
    delta_logprob = (post_joint_logprob - old_logprobs).detach().cpu().numpy().astype(np.float64)
    advantages = advantages_t.detach().cpu().numpy().astype(np.float64)
    positive_mask = advantages > 0.0
    negative_mask = advantages < 0.0

    action_gap_values: list[float] = []
    true_adv_mc_values: list[float] = []
    true_adv_mc_sampled_q_values: list[float] = []
    true_adv_mc_policy_q_values: list[float] = []
    true_adv_mc_steps: list[int] = []
    branch_ref_mode_l = str(branch_ref_mode or "deterministic").strip().lower()
    if branch_ref_mode_l not in {"deterministic"}:
        raise ValueError("branch_ref_mode currently supports only 'deterministic'.")
    branch_follow_mode_l = str(branch_follow_policy_mode or "stochastic").strip().lower()
    if branch_follow_mode_l not in {"deterministic", "stochastic"}:
        raise ValueError("branch_follow_policy_mode must be one of {'deterministic', 'stochastic'}.")
    branch_horizon_values = sorted(
        {
            int(value)
            for value in (branch_horizons if branch_horizons is not None else [])
            if int(value) > 0
        }
    )
    branch_samples_eff = max(int(branch_samples), 0)
    if branch_follow_mode_l == "deterministic" and branch_samples_eff > 1:
        branch_samples_eff = 1
    branch_records_by_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in branch_horizon_values}
    sample_rows: list[dict[str, float]] = []
    action_gap_joint_actions = probe_context.get("action_gap_joint_actions")
    action_gap_advantages_t = probe_context.get("action_gap_advantages")
    action_gap_value_preds_t = probe_context.get("action_gap_values")
    action_gap_returns_t = probe_context.get("action_gap_returns")
    action_gap_positions = [int(pos) for pos in probe_context.get("action_gap_positions", [])]
    action_gap_advantages = (
        np.asarray([], dtype=np.float64)
        if action_gap_advantages_t is None
        else action_gap_advantages_t.detach().cpu().numpy().astype(np.float64)
    )
    action_gap_value_preds = (
        np.asarray([], dtype=np.float64)
        if action_gap_value_preds_t is None
        else action_gap_value_preds_t.detach().cpu().numpy().astype(np.float64)
    )
    action_gap_returns = (
        np.asarray([], dtype=np.float64)
        if action_gap_returns_t is None
        else action_gap_returns_t.detach().cpu().numpy().astype(np.float64)
    )
    action_gap_raw_advantages = action_gap_returns - action_gap_value_preds
    action_gap_delta_logprob = (
        np.asarray([], dtype=np.float64)
        if not action_gap_positions
        else delta_logprob[np.asarray(action_gap_positions, dtype=np.int64)]
    )
    action_gap_positive_mask = action_gap_advantages > 0.0
    action_gap_negative_mask = action_gap_advantages < 0.0
    if action_gap_joint_actions is not None:
        for row_idx, (snapshot_state, sampled_action, advantage, delta_lp, raw_adv, return_target, value_pred) in enumerate(
            zip(
                probe_context["snapshot_states"],
                action_gap_joint_actions.detach().cpu().numpy(),
                action_gap_advantages.tolist(),
                action_gap_delta_logprob.tolist(),
                action_gap_raw_advantages.tolist(),
                action_gap_returns.tolist(),
                action_gap_value_preds.tolist(),
            )
        ):
            probe_driver = _make_probe_driver(cfg)
            try:
                snapshot = probe_driver.load_bw_stage_state(snapshot_state)
                det_action = _bw_action_from_actor(pre_actor, snapshot, device, deterministic=True)
            finally:
                _close_probe_driver(probe_driver)
            sampled_roll = _rollout_from_snapshot(
                snapshot_state=snapshot_state,
                first_action=np.asarray(sampled_action, dtype=np.float32),
                cfg=cfg,
                actor=pre_actor,
                device=device,
                k_steps=int(k_steps),
                gamma=float(cfg.gamma),
                follow_deterministic=True,
            )
            det_roll = _rollout_from_snapshot(
                snapshot_state=snapshot_state,
                first_action=np.asarray(det_action, dtype=np.float32),
                cfg=cfg,
                actor=pre_actor,
                device=device,
                k_steps=int(k_steps),
                gamma=float(cfg.gamma),
                follow_deterministic=True,
            )
            action_gap = float(sampled_roll["reward"] - det_roll["reward"])
            action_gap_values.append(action_gap)
            branch_summary_for_row: dict[int, float] = {}
            if bool(branch_enabled) and branch_horizon_values and branch_samples_eff > 0:
                for horizon in branch_horizon_values:
                    sampled_branch_qs: list[float] = []
                    ref_branch_qs: list[float] = []
                    for branch_sample_idx in range(int(branch_samples_eff)):
                        branch_sample_seed = (
                            int(branch_seed)
                            + int(row_idx) * 1_000_000
                            + int(horizon) * 10_000
                            + int(branch_sample_idx) * 100
                        )
                        sampled_branch_roll = _rollout_from_snapshot(
                            snapshot_state=snapshot_state,
                            first_action=np.asarray(sampled_action, dtype=np.float32),
                            cfg=cfg,
                            actor=pre_actor,
                            device=device,
                            k_steps=int(horizon),
                            gamma=float(cfg.gamma),
                            follow_deterministic=(branch_follow_mode_l == "deterministic"),
                            sample_seed_base=None if branch_follow_mode_l == "deterministic" else int(branch_sample_seed),
                        )
                        ref_branch_roll = _rollout_from_snapshot(
                            snapshot_state=snapshot_state,
                            first_action=np.asarray(det_action, dtype=np.float32),
                            cfg=cfg,
                            actor=pre_actor,
                            device=device,
                            k_steps=int(horizon),
                            gamma=float(cfg.gamma),
                            follow_deterministic=(branch_follow_mode_l == "deterministic"),
                            sample_seed_base=None if branch_follow_mode_l == "deterministic" else int(branch_sample_seed),
                        )
                        sampled_branch_qs.append(float(sampled_branch_roll["reward"]))
                        ref_branch_qs.append(float(ref_branch_roll["reward"]))
                    sampled_branch_q = _safe_mean(np.asarray(sampled_branch_qs, dtype=np.float64))
                    ref_branch_q = _safe_mean(np.asarray(ref_branch_qs, dtype=np.float64))
                    branch_delta = float(sampled_branch_q - ref_branch_q)
                    branch_summary_for_row[int(horizon)] = branch_delta
                    branch_records_by_horizon[int(horizon)].append(
                        {
                            "sample": float(row_idx),
                            "t": float(_snapshot_step(snapshot_state)),
                            "horizon": float(horizon),
                            "advantage": float(advantage),
                            "raw_advantage": float(raw_adv),
                            "delta_logprob": float(delta_lp),
                            "sampled_q": float(sampled_branch_q),
                            "ref_q": float(ref_branch_q),
                            "delta": float(branch_delta),
                        }
                    )
            true_adv_mc = 0.0
            sampled_q_mc = 0.0
            policy_q_mc = 0.0
            if bool(true_mc_enabled) and int(true_mc_samples) > 0:
                sampled_qs: list[float] = []
                policy_qs: list[float] = []
                mc_samples = max(int(true_mc_samples), 1)
                for mc_idx in range(mc_samples):
                    mc_seed = int(true_mc_seed) + int(row_idx) * 100_000 + int(mc_idx) * 1_000
                    sampled_mc_roll = _rollout_from_snapshot(
                        snapshot_state=snapshot_state,
                        first_action=np.asarray(sampled_action, dtype=np.float32),
                        cfg=cfg,
                        actor=pre_actor,
                        device=device,
                        k_steps=int(k_steps),
                        gamma=float(cfg.gamma),
                        follow_deterministic=False,
                        sample_seed_base=int(mc_seed),
                    )
                    policy_action = _bw_action_from_actor(
                        pre_actor,
                        snapshot,
                        device,
                        deterministic=False,
                        sample_seed=int(mc_seed),
                    )
                    policy_mc_roll = _rollout_from_snapshot(
                        snapshot_state=snapshot_state,
                        first_action=np.asarray(policy_action, dtype=np.float32),
                        cfg=cfg,
                        actor=pre_actor,
                        device=device,
                        k_steps=int(k_steps),
                        gamma=float(cfg.gamma),
                        follow_deterministic=False,
                        sample_seed_base=int(mc_seed),
                    )
                    sampled_qs.append(float(sampled_mc_roll["reward"]))
                    policy_qs.append(float(policy_mc_roll["reward"]))
                sampled_q_mc = _safe_mean(np.asarray(sampled_qs, dtype=np.float64))
                policy_q_mc = _safe_mean(np.asarray(policy_qs, dtype=np.float64))
                true_adv_mc = float(sampled_q_mc - policy_q_mc)
                true_adv_mc_values.append(true_adv_mc)
                true_adv_mc_sampled_q_values.append(sampled_q_mc)
                true_adv_mc_policy_q_values.append(policy_q_mc)
                true_adv_mc_steps.append(_snapshot_step(snapshot_state))
            if len(sample_rows) < 8:
                sample_row = {
                    "sample": float(row_idx),
                    "t": float(_snapshot_step(snapshot_state)),
                    "advantage": float(advantage),
                    "raw_advantage": float(raw_adv),
                    "return_target": float(return_target),
                    "value_pred": float(value_pred),
                    "delta_logprob": float(delta_lp),
                    "action_gap_k": float(action_gap),
                }
                for horizon, value in branch_summary_for_row.items():
                    sample_row[f"branch_delta_h{int(horizon)}"] = float(value)
                if bool(true_mc_enabled) and int(true_mc_samples) > 0:
                    sample_row.update(
                        {
                            "true_adv_mc": float(true_adv_mc),
                            "sampled_q_mc": float(sampled_q_mc),
                            "policy_q_mc": float(policy_q_mc),
                        }
                    )
                sample_rows.append(sample_row)

    pos_logprob = delta_logprob[positive_mask]
    neg_logprob = delta_logprob[negative_mask]
    pos_gap = np.asarray([gap for gap, keep in zip(action_gap_values, action_gap_positive_mask.tolist()) if keep], dtype=np.float64)
    neg_gap = np.asarray([gap for gap, keep in zip(action_gap_values, action_gap_negative_mask.tolist()) if keep], dtype=np.float64)
    true_mc_advantages = np.asarray(true_adv_mc_values, dtype=np.float64)
    true_mc_sampled_q = np.asarray(true_adv_mc_sampled_q_values, dtype=np.float64)
    true_mc_policy_q = np.asarray(true_adv_mc_policy_q_values, dtype=np.float64)
    true_mc_steps = np.asarray(true_adv_mc_steps, dtype=np.int64)
    true_mc_ppo_advantages = action_gap_advantages[: true_mc_advantages.size]
    true_mc_raw_advantages = action_gap_raw_advantages[: true_mc_advantages.size]
    true_mc_return_targets = action_gap_returns[: true_mc_advantages.size]
    true_mc_value_preds = action_gap_value_preds[: true_mc_advantages.size]
    true_mc_delta_logprob = action_gap_delta_logprob[: true_mc_advantages.size]
    true_action_alignment = _summarize_true_advantage_alignment(
        ppo_advantages=true_mc_ppo_advantages,
        true_advantages=true_mc_advantages,
        delta_logprob=true_mc_delta_logprob,
    )
    advantage_decomposition = _summarize_advantage_decomposition(
        norm_advantages=true_mc_ppo_advantages,
        raw_advantages=true_mc_raw_advantages,
        return_targets=true_mc_return_targets,
        value_preds=true_mc_value_preds,
        sampled_q_mc=true_mc_sampled_q,
        policy_q_mc=true_mc_policy_q,
        delta_logprob=true_mc_delta_logprob,
    )
    true_action_alignment.update(
        {
            "enabled": bool(true_mc_enabled),
            "sample_count": int(true_mc_advantages.size),
            "policy_samples_per_action": int(max(int(true_mc_samples), 0)) if bool(true_mc_enabled) else 0,
            "sampled_q_mc": _summarize(true_mc_sampled_q.tolist()),
            "policy_q_mc": _summarize(true_mc_policy_q.tolist()),
        }
    )
    true_mc_buckets: list[dict[str, float]] = []
    if true_mc_advantages.size > 0:
        for step in sorted({int(value) for value in true_mc_steps.tolist()}):
            keep = true_mc_steps == int(step)
            bucket_summary = _summarize_true_advantage_alignment(
                ppo_advantages=true_mc_ppo_advantages[keep],
                true_advantages=true_mc_advantages[keep],
                delta_logprob=true_mc_delta_logprob[keep],
            )
            bucket_summary["decomposition"] = _summarize_advantage_decomposition(
                norm_advantages=true_mc_ppo_advantages[keep],
                raw_advantages=true_mc_raw_advantages[keep],
                return_targets=true_mc_return_targets[keep],
                value_preds=true_mc_value_preds[keep],
                sampled_q_mc=true_mc_sampled_q[keep],
                policy_q_mc=true_mc_policy_q[keep],
                delta_logprob=true_mc_delta_logprob[keep],
            )
            bucket_summary.update(
                {
                    "t": float(step),
                    "remaining_steps": float(max(int(getattr(cfg, "T_steps", 0) or 0) - int(step), 0)),
                    "sample_count": float(int(np.sum(keep))),
                }
            )
            true_mc_buckets.append(bucket_summary)
    true_action_alignment["buckets_by_t"] = true_mc_buckets
    true_action_alignment["advantage_decomposition"] = advantage_decomposition

    branch_horizon_summaries: list[dict[str, Any]] = []
    branch_primary_summary = _summarize_branch_alignment(
        norm_advantages=np.asarray([], dtype=np.float64),
        raw_advantages=np.asarray([], dtype=np.float64),
        branch_deltas=np.asarray([], dtype=np.float64),
        delta_logprob=np.asarray([], dtype=np.float64),
    )
    branch_primary_horizon = 0
    for horizon in branch_horizon_values:
        records = branch_records_by_horizon.get(int(horizon), [])
        norm_adv_h = np.asarray([row["advantage"] for row in records], dtype=np.float64)
        raw_adv_h = np.asarray([row["raw_advantage"] for row in records], dtype=np.float64)
        branch_delta_h = np.asarray([row["delta"] for row in records], dtype=np.float64)
        dlogp_h = np.asarray([row["delta_logprob"] for row in records], dtype=np.float64)
        summary_h = _summarize_branch_alignment(
            norm_advantages=norm_adv_h,
            raw_advantages=raw_adv_h,
            branch_deltas=branch_delta_h,
            delta_logprob=dlogp_h,
        )
        summary_h.update(
            {
                "horizon": float(horizon),
                "sampled_q": _summarize([row["sampled_q"] for row in records]),
                "ref_q": _summarize([row["ref_q"] for row in records]),
                "delta": _summarize([row["delta"] for row in records]),
            }
        )
        branch_horizon_summaries.append(summary_h)
        if int(horizon) == max(branch_horizon_values, default=0):
            branch_primary_horizon = int(horizon)
            branch_primary_summary = summary_h
    branch_alignment = {
        "enabled": bool(branch_enabled),
        "ref_mode": str(branch_ref_mode_l),
        "follow_policy_mode": str(branch_follow_mode_l),
        "samples_per_action": int(branch_samples_eff) if bool(branch_enabled) else 0,
        "primary_horizon": int(branch_primary_horizon),
        "primary": branch_primary_summary,
        "by_horizon": branch_horizon_summaries,
        "examples": {
            f"h{int(horizon)}": branch_records_by_horizon.get(int(horizon), [])[:8]
            for horizon in branch_horizon_values
        },
    }

    critic_signal_good = (
        _safe_corr(action_gap_advantages.tolist(), action_gap_values) > 0.05
        and (float(np.mean(pos_gap)) if pos_gap.size > 0 else 0.0) >= (float(np.mean(neg_gap)) if neg_gap.size > 0 else 0.0)
    )
    logprob_signal_good = (
        _safe_corr(advantages.tolist(), delta_logprob.tolist()) > 0.05
        and (float(np.mean(pos_logprob)) if pos_logprob.size > 0 else 0.0) >= (float(np.mean(neg_logprob)) if neg_logprob.size > 0 else 0.0)
        and float(np.mean(replay_abs_diff)) <= 1.0e-4
    )
    if critic_signal_good and not logprob_signal_good:
        judgement = "logprob_interface_suspect"
    elif (not critic_signal_good) and logprob_signal_good:
        judgement = "critic_advantage_suspect"
    elif (not critic_signal_good) and (not logprob_signal_good):
        judgement = "both_suspect"
    else:
        judgement = "inconclusive_or_consistent"

    return {
        "sample_count": int(num_samples),
        "action_gap_sample_count": int(len(action_gap_values)),
        "advantage": {
            "mean": float(np.mean(advantages)) if advantages.size > 0 else 0.0,
            "abs_mean": float(np.mean(np.abs(advantages))) if advantages.size > 0 else 0.0,
            "positive_frac": float(np.mean(positive_mask.astype(np.float64))) if advantages.size > 0 else 0.0,
        },
        "logprob_alignment": {
            "sample_count": int(num_samples),
            "old_logprob_replay_abs_diff": _summarize(replay_abs_diff.tolist()),
            "old_logprob_exact_replay_frac_1e-5": float(np.mean(replay_abs_diff <= 1.0e-5)) if replay_abs_diff.size > 0 else 0.0,
            "delta_logprob": _summarize(delta_logprob.tolist()),
            "corr_advantage_vs_delta_logprob": _safe_corr(advantages.tolist(), delta_logprob.tolist()),
            "mean_delta_logprob_pos_adv": float(np.mean(pos_logprob)) if pos_logprob.size > 0 else 0.0,
            "mean_delta_logprob_neg_adv": float(np.mean(neg_logprob)) if neg_logprob.size > 0 else 0.0,
            "pos_adv_logprob_up_frac": float(np.mean(pos_logprob > 0.0)) if pos_logprob.size > 0 else 0.0,
            "neg_adv_logprob_down_frac": float(np.mean(neg_logprob < 0.0)) if neg_logprob.size > 0 else 0.0,
        },
        "critic_alignment": {
            "sample_count": int(len(action_gap_values)),
            "action_gap_k": _summarize(action_gap_values),
            "corr_advantage_vs_action_gap_k": _safe_corr(action_gap_advantages.tolist(), action_gap_values),
            "mean_action_gap_k_pos_adv": float(np.mean(pos_gap)) if pos_gap.size > 0 else 0.0,
            "mean_action_gap_k_neg_adv": float(np.mean(neg_gap)) if neg_gap.size > 0 else 0.0,
            "pos_adv_action_gap_positive_frac": float(np.mean(pos_gap > 0.0)) if pos_gap.size > 0 else 0.0,
            "neg_adv_action_gap_negative_frac": float(np.mean(neg_gap < 0.0)) if neg_gap.size > 0 else 0.0,
        },
        "true_action_alignment": true_action_alignment,
        "branch_alignment": branch_alignment,
        "judgement": str(judgement),
        "examples": sample_rows,
    }


def bw_update_direction_probe_fieldnames() -> List[str]:
    return [
        "update",
        "panel_states",
        "k_steps",
        "actor_policy_mode",
        "actor_policy_samples",
        "pre_actor_reward_mean",
        "post_actor_reward_mean",
        "delta_actor_reward_mean",
        "actor_reward_improved_state_frac",
        "pre_actor_weighted_mean",
        "post_actor_weighted_mean",
        "delta_actor_weighted_mean",
        "pre_heuristic_reward_mean",
        "post_heuristic_reward_mean",
        "pre_heuristic_gap_mean",
        "post_heuristic_gap_mean",
        "delta_heuristic_gap_mean",
        "pre_l1_to_heur_mean",
        "post_l1_to_heur_mean",
        "delta_l1_to_heur_mean",
        "pre_heuristic_beats_actor_frac",
        "post_heuristic_beats_actor_frac",
        "delta_heuristic_beats_actor_frac",
        "probe_batch_samples",
        "probe_action_gap_samples",
        "adv_positive_frac",
        "old_logprob_replay_abs_diff_mean",
        "old_logprob_exact_replay_frac_1e-5",
        "corr_advantage_vs_delta_logprob",
        "mean_delta_logprob_pos_adv",
        "mean_delta_logprob_neg_adv",
        "pos_adv_logprob_up_frac",
        "neg_adv_logprob_down_frac",
        "corr_advantage_vs_action_gap_k",
        "mean_action_gap_k_pos_adv",
        "mean_action_gap_k_neg_adv",
        "pos_adv_action_gap_positive_frac",
        "neg_adv_action_gap_negative_frac",
        "branch_enabled",
        "branch_ref_mode",
        "branch_follow_policy_mode",
        "branch_samples_per_action",
        "branch_primary_horizon",
        "branch_sample_count",
        "corr_advantage_vs_branch_delta",
        "corr_raw_advantage_vs_branch_delta",
        "sign_agree_raw_advantage_branch_delta",
        "corr_branch_delta_vs_delta_logprob",
        "branch_delta_abs_mean",
        "branch_delta_std",
        "branch_delta_snr",
        "mean_branch_delta_pos_adv",
        "mean_branch_delta_neg_adv",
        "mean_branch_delta_pos_raw_adv",
        "mean_branch_delta_neg_raw_adv",
        "raw_pos_branch_delta_positive_frac",
        "raw_neg_branch_delta_negative_frac",
        "branch_delta_pos_logprob_up_frac",
        "branch_delta_neg_logprob_down_frac",
        "corr_raw_advantage_vs_branch_delta_h2",
        "corr_branch_delta_vs_delta_logprob_h2",
        "branch_delta_abs_mean_h2",
        "corr_raw_advantage_vs_branch_delta_h5",
        "corr_branch_delta_vs_delta_logprob_h5",
        "branch_delta_abs_mean_h5",
        "corr_raw_advantage_vs_branch_delta_h10",
        "corr_branch_delta_vs_delta_logprob_h10",
        "branch_delta_abs_mean_h10",
        "true_mc_sample_count",
        "true_mc_policy_samples_per_action",
        "corr_advantage_vs_true_adv_mc",
        "sign_agree_advantage_true_adv_mc",
        "corr_true_adv_mc_vs_delta_logprob",
        "true_adv_mc_abs_mean",
        "true_adv_mc_std",
        "true_adv_mc_snr",
        "mean_true_adv_mc_pos_adv",
        "mean_true_adv_mc_neg_adv",
        "ppo_pos_true_adv_positive_frac",
        "ppo_neg_true_adv_negative_frac",
        "mean_delta_logprob_true_adv_pos",
        "mean_delta_logprob_true_adv_neg",
        "true_adv_pos_logprob_up_frac",
        "true_adv_neg_logprob_down_frac",
        "corr_norm_advantage_vs_raw_advantage",
        "sign_agree_norm_raw_advantage",
        "corr_raw_advantage_vs_true_adv_mc",
        "sign_agree_raw_advantage_true_adv_mc",
        "corr_raw_advantage_vs_delta_logprob",
        "raw_advantage_abs_mean",
        "raw_advantage_std",
        "raw_advantage_snr",
        "corr_return_target_vs_sampled_q_mc",
        "return_target_sampled_q_error_mean",
        "return_target_sampled_q_error_abs_mean",
        "corr_value_vs_policy_q_mc",
        "value_policy_q_error_mean",
        "value_policy_q_error_abs_mean",
        "return_target_std",
        "sampled_q_mc_std",
        "value_pred_std",
        "policy_q_mc_std",
        "alignment_judgement",
        "env_reward_mean",
        "bw_train_reward_mean",
        "policy_loss",
        "value_loss_bw",
        "approx_kl_bw",
        "clip_frac_bw",
        "entropy_bw",
    ]


def append_bw_update_direction_probe_row(path: str, row: Dict[str, object]) -> None:
    fieldnames = bw_update_direction_probe_fieldnames()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_bw_update_direction_probe_payload(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
