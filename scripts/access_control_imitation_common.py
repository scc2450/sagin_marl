from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sagin_marl.env import channel

from evaluate_structured_access_control_oracle import (
    _assoc_candidate_users,
    _feasible_overlap_candidate_lists,
    _reference_eta_matrix,
)


@dataclass
class AccessUserGroup:
    features: np.ndarray
    target_index: int
    weight: float


class AccessBidScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ACCESS_RULES: tuple[dict[str, float], ...] = (
    {
        "name": "overlap_bid_q_eta",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.0,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_prev",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.3,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_only",
        "eta_bias": 1.0,
        "eta_weight": 0.0,
        "prev_bonus": 0.0,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_dist",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.15,
        "dist_penalty": 0.75,
        "own_load_penalty": 0.0,
    },
    {
        "name": "overlap_bid_q_eta_load",
        "eta_bias": 0.5,
        "eta_weight": 1.0,
        "prev_bonus": 0.15,
        "dist_penalty": 0.0,
        "own_load_penalty": 0.75,
    },
)


def get_access_rule(rule_name: str) -> dict[str, float]:
    for rule in ACCESS_RULES:
        if str(rule["name"]) == str(rule_name):
            return dict(rule)
    raise KeyError(f"Unknown access rule: {rule_name}")


def pair_feature_names() -> list[str]:
    return [
        "rel_x",
        "rel_y",
        "q_norm",
        "eta",
        "prev_assoc_to_u",
        "base_assoc_to_u",
        "dist_norm",
        "own_assoc_count_norm",
        "own_queue_norm",
        "own_energy_norm",
        "time_norm",
    ]


def _assoc_candidate_users_from_state(
    cfg,
    gu_queue: np.ndarray,
    assoc: np.ndarray,
    *,
    max_keep: int,
) -> list[list[int]]:
    candidates: list[list[int]] = [[] for _ in range(int(cfg.num_uav))]
    for gu_idx, u in enumerate(np.asarray(assoc, dtype=np.int32).tolist()):
        if u >= 0:
            candidates[int(u)].append(int(gu_idx))
    for u in range(int(cfg.num_uav)):
        if len(candidates[u]) > int(max_keep):
            candidates[u].sort(key=lambda idx: float(gu_queue[int(idx)]), reverse=True)
            candidates[u] = candidates[u][: int(max_keep)]
    return candidates


def _feasible_overlap_candidate_lists_from_state(
    cfg,
    *,
    uav_pos: np.ndarray,
    gu_pos: np.ndarray,
    gu_queue: np.ndarray,
    per_uav_keep: int,
    per_user_max_uav: int,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    num_gu = int(cfg.num_gu)
    num_uav = int(cfg.num_uav)
    if num_gu <= 0 or num_uav <= 0:
        return [[] for _ in range(num_uav)], np.zeros((num_gu, num_uav), dtype=np.float32), np.zeros((num_gu, num_uav), dtype=np.float32)
    diff = np.asarray(gu_pos, dtype=np.float32)[:, None, :] - np.asarray(uav_pos, dtype=np.float32)[None, :, :]
    d2d = np.linalg.norm(diff, axis=2)
    d3d = np.sqrt(d2d * d2d + float(cfg.uav_height) ** 2)
    phi = np.arcsin(float(cfg.uav_height) / (d3d + 1.0e-9))
    pathloss = channel.pathloss_db(d3d, phi, cfg).astype(np.float32, copy=False)
    feasible = pathloss <= float(cfg.pl_threshold_db)
    candidates: list[list[int]] = [[] for _ in range(num_uav)]
    for gu_idx in range(num_gu):
        feasible_u = np.flatnonzero(feasible[gu_idx])
        if feasible_u.size <= 0:
            continue
        ordered = feasible_u[np.argsort(pathloss[gu_idx, feasible_u], kind="stable")]
        keep_u = ordered[: max(int(per_user_max_uav), 1)]
        for u in keep_u.tolist():
            candidates[int(u)].append(int(gu_idx))
    for u in range(num_uav):
        if len(candidates[u]) > int(per_uav_keep):
            candidates[u].sort(key=lambda idx: float(gu_queue[int(idx)]), reverse=True)
            candidates[u] = candidates[u][: int(per_uav_keep)]
    return candidates, pathloss.astype(np.float32, copy=False), d2d.astype(np.float32, copy=False)


def _reference_eta_matrix_from_state(
    cfg,
    *,
    uav_pos: np.ndarray,
    gu_pos: np.ndarray,
) -> np.ndarray:
    diff = np.asarray(gu_pos, dtype=np.float32)[:, None, :] - np.asarray(uav_pos, dtype=np.float32)[None, :, :]
    d2d = np.linalg.norm(diff, axis=2)
    d3d = np.sqrt(d2d * d2d + float(cfg.uav_height) ** 2)
    phi = np.arcsin(float(cfg.uav_height) / (d3d + 1.0e-9))
    pl = channel.pathloss_db(d3d, phi, cfg)
    gain = np.asarray(10.0 ** (-pl / 10.0), dtype=np.float32)
    ref_snr = channel.snr_linear(
        float(cfg.gu_tx_power),
        gain,
        float(cfg.noise_density),
        float(cfg.b_acc),
    )
    return np.asarray(channel.spectral_efficiency(ref_snr), dtype=np.float32)


def _match_assoc_from_state_bids(
    cfg,
    *,
    base_assoc: np.ndarray,
    overlap_candidates: list[list[int]],
    gu_queue: np.ndarray,
    prev_association: np.ndarray,
    eta_ref: np.ndarray,
    d2d: np.ndarray,
    pathloss: np.ndarray,
    uav_queue: np.ndarray,
    uav_energy: np.ndarray,
    rule: dict[str, float],
) -> np.ndarray:
    assoc = np.asarray(base_assoc, dtype=np.int32).copy()
    own_assoc_count = np.bincount(
        np.asarray(base_assoc, dtype=np.int32)[np.asarray(base_assoc, dtype=np.int32) >= 0],
        minlength=int(cfg.num_uav),
    ).astype(np.float32, copy=False)
    contenders: dict[int, list[tuple[float, float, int]]] = {}
    for u, cand_list in enumerate(overlap_candidates):
        for gu_idx in cand_list:
            q_norm = float(gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9))
            eta = float(eta_ref[int(gu_idx), int(u)])
            prev = 1.0 if int(prev_association[int(gu_idx)]) == int(u) else 0.0
            dist_norm = float(d2d[int(gu_idx), int(u)] / max(float(cfg.map_size), 1.0e-9))
            own_load = float(own_assoc_count[int(u)] / max(float(cfg.num_gu), 1.0))
            score = q_norm * (float(rule["eta_bias"]) + float(rule["eta_weight"]) * eta)
            score = score * (1.0 + float(rule["prev_bonus"]) * prev)
            score = score / (1.0 + float(rule["dist_penalty"]) * dist_norm)
            score = score / (1.0 + float(rule["own_load_penalty"]) * own_load)
            contenders.setdefault(int(gu_idx), []).append((float(score), -float(pathloss[int(gu_idx), int(u)]), int(u)))
    for gu_idx, items in contenders.items():
        best_score, _neg_pl, best_u = max(items, key=lambda item: (float(item[0]), float(item[1]), -int(item[2])))
        if best_score > 0.0:
            assoc[int(gu_idx)] = int(best_u)
    return assoc.astype(np.int32, copy=False)


def build_overlap_user_choices(
    env,
    *,
    base_assoc: np.ndarray,
    selected_assoc: np.ndarray,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
) -> tuple[list[AccessUserGroup], dict[str, Any]]:
    cfg = env.cfg
    overlap_candidates, _pathloss, d2d = _feasible_overlap_candidate_lists(
        env,
        per_uav_keep=int(overlap_per_uav_keep),
        per_user_max_uav=int(overlap_per_user_max_uav),
    )
    eta_ref = _reference_eta_matrix(env)
    own_assoc_count = np.bincount(
        np.asarray(base_assoc, dtype=np.int32)[np.asarray(base_assoc, dtype=np.int32) >= 0],
        minlength=int(cfg.num_uav),
    ).astype(np.float32, copy=False)
    per_user_uavs: dict[int, list[int]] = {}
    for u, cand_list in enumerate(overlap_candidates):
        for gu_idx in cand_list:
            per_user_uavs.setdefault(int(gu_idx), []).append(int(u))
    groups: list[AccessUserGroup] = []
    skipped_users = 0
    for gu_idx in range(int(cfg.num_gu)):
        target_u = int(selected_assoc[gu_idx])
        if target_u < 0:
            skipped_users += 1
            continue
        cand_uavs = sorted(set(int(u) for u in per_user_uavs.get(int(gu_idx), [])) | {target_u})
        if not cand_uavs:
            skipped_users += 1
            continue
        rel_rows: list[list[float]] = []
        target_index = None
        for idx, u in enumerate(cand_uavs):
            rel = (env.gu_pos[int(gu_idx)] - env.uav_pos[int(u)]) / max(float(cfg.map_size), 1.0e-9)
            feat = [
                float(rel[0]),
                float(rel[1]),
                float(env.gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9)),
                float(eta_ref[int(gu_idx), int(u)]),
                1.0 if int(env.prev_association[int(gu_idx)]) == int(u) else 0.0,
                1.0 if int(base_assoc[int(gu_idx)]) == int(u) else 0.0,
                float(d2d[int(gu_idx), int(u)] / max(float(cfg.map_size), 1.0e-9)),
                float(own_assoc_count[int(u)] / max(float(cfg.num_gu), 1.0)),
                float(env.uav_queue[int(u)] / max(float(cfg.queue_max_uav), 1.0e-9)),
                float(env.uav_energy[int(u)] / max(float(cfg.uav_energy_init), 1.0e-9)),
                float(env.t / max(float(cfg.T_steps), 1.0)),
            ]
            rel_rows.append(feat)
            if int(u) == target_u:
                target_index = int(idx)
        if target_index is None:
            skipped_users += 1
            continue
        groups.append(
            AccessUserGroup(
                features=np.asarray(rel_rows, dtype=np.float32),
                target_index=int(target_index),
                weight=1.0,
            )
        )
    meta = {
        "group_count": int(len(groups)),
        "skipped_users": int(skipped_users),
        "feature_dim": int(len(pair_feature_names())),
        "overlap_candidates": overlap_candidates,
        "eta_ref": eta_ref,
        "d2d": d2d,
        "base_assoc": np.asarray(base_assoc, dtype=np.int32).copy(),
        "own_assoc_count": own_assoc_count,
    }
    return groups, meta


def predict_access_assoc_from_snapshot(
    snapshot_state: dict[str, Any],
    cfg,
    model: AccessBidScorer,
    device: torch.device,
    *,
    base_assoc: np.ndarray | None = None,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
) -> tuple[np.ndarray, list[list[int]]]:
    env_state = dict(snapshot_state["env_state"])
    uav_pos = np.asarray(env_state["uav_pos"], dtype=np.float32)
    gu_pos = np.asarray(env_state["gu_pos"], dtype=np.float32)
    gu_queue = np.asarray(env_state["gu_queue"], dtype=np.float32)
    uav_queue = np.asarray(env_state["uav_queue"], dtype=np.float32)
    uav_energy = np.asarray(env_state["uav_energy"], dtype=np.float32)
    prev_association = np.asarray(env_state["prev_association"], dtype=np.int32)
    t = int(env_state["t"])
    if base_assoc is None:
        base_assoc = np.asarray(snapshot_state["stage_assoc"], dtype=np.int32)
    else:
        base_assoc = np.asarray(base_assoc, dtype=np.int32)
    overlap_candidates, _pathloss, d2d = _feasible_overlap_candidate_lists_from_state(
        cfg,
        uav_pos=uav_pos,
        gu_pos=gu_pos,
        gu_queue=gu_queue,
        per_uav_keep=int(overlap_per_uav_keep),
        per_user_max_uav=int(overlap_per_user_max_uav),
    )
    eta_ref = _reference_eta_matrix_from_state(cfg, uav_pos=uav_pos, gu_pos=gu_pos)
    own_assoc_count = np.bincount(
        np.asarray(base_assoc, dtype=np.int32)[np.asarray(base_assoc, dtype=np.int32) >= 0],
        minlength=int(cfg.num_uav),
    ).astype(np.float32, copy=False)
    per_user_uavs: dict[int, list[int]] = {}
    for u, cand_list in enumerate(overlap_candidates):
        for gu_idx in cand_list:
            per_user_uavs.setdefault(int(gu_idx), []).append(int(u))
    assoc = np.asarray(base_assoc, dtype=np.int32).copy()
    model.eval()
    with torch.inference_mode():
        for gu_idx in range(int(cfg.num_gu)):
            cand_uavs = sorted(set(int(u) for u in per_user_uavs.get(int(gu_idx), [])))
            if not cand_uavs:
                continue
            rows: list[list[float]] = []
            for u in cand_uavs:
                rel = (gu_pos[int(gu_idx)] - uav_pos[int(u)]) / max(float(cfg.map_size), 1.0e-9)
                rows.append(
                    [
                        float(rel[0]),
                        float(rel[1]),
                        float(gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9)),
                        float(eta_ref[int(gu_idx), int(u)]),
                        1.0 if int(prev_association[int(gu_idx)]) == int(u) else 0.0,
                        1.0 if int(base_assoc[int(gu_idx)]) == int(u) else 0.0,
                        float(d2d[int(gu_idx), int(u)] / max(float(cfg.map_size), 1.0e-9)),
                        float(own_assoc_count[int(u)] / max(float(cfg.num_gu), 1.0)),
                        float(uav_queue[int(u)] / max(float(cfg.queue_max_uav), 1.0e-9)),
                        float(uav_energy[int(u)] / max(float(cfg.uav_energy_init), 1.0e-9)),
                        float(t / max(float(cfg.T_steps), 1.0)),
                    ]
                )
            logits = model(torch.as_tensor(np.asarray(rows, dtype=np.float32), device=device)).squeeze(-1)
            best_idx = int(torch.argmax(logits).item())
            assoc[int(gu_idx)] = int(cand_uavs[best_idx])
    candidates = _assoc_candidate_users_from_state(
        cfg,
        gu_queue,
        assoc,
        max_keep=int(cfg.users_obs_max),
    )
    return assoc.astype(np.int32, copy=False), candidates


def predict_access_assoc_fixedrule_from_snapshot(
    snapshot_state: dict[str, Any],
    cfg,
    *,
    rule_name: str,
    base_assoc: np.ndarray | None = None,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
) -> tuple[np.ndarray, list[list[int]]]:
    env_state = dict(snapshot_state["env_state"])
    uav_pos = np.asarray(env_state["uav_pos"], dtype=np.float32)
    gu_pos = np.asarray(env_state["gu_pos"], dtype=np.float32)
    gu_queue = np.asarray(env_state["gu_queue"], dtype=np.float32)
    uav_queue = np.asarray(env_state["uav_queue"], dtype=np.float32)
    uav_energy = np.asarray(env_state["uav_energy"], dtype=np.float32)
    prev_association = np.asarray(env_state["prev_association"], dtype=np.int32)
    if base_assoc is None:
        base_assoc = np.asarray(snapshot_state["stage_assoc"], dtype=np.int32)
    else:
        base_assoc = np.asarray(base_assoc, dtype=np.int32)
    overlap_candidates, pathloss, d2d = _feasible_overlap_candidate_lists_from_state(
        cfg,
        uav_pos=uav_pos,
        gu_pos=gu_pos,
        gu_queue=gu_queue,
        per_uav_keep=int(overlap_per_uav_keep),
        per_user_max_uav=int(overlap_per_user_max_uav),
    )
    eta_ref = _reference_eta_matrix_from_state(cfg, uav_pos=uav_pos, gu_pos=gu_pos)
    assoc = _match_assoc_from_state_bids(
        cfg,
        base_assoc=base_assoc,
        overlap_candidates=overlap_candidates,
        gu_queue=gu_queue,
        prev_association=prev_association,
        eta_ref=eta_ref,
        d2d=d2d,
        pathloss=pathloss,
        uav_queue=uav_queue,
        uav_energy=uav_energy,
        rule=get_access_rule(rule_name),
    )
    candidates = _assoc_candidate_users_from_state(
        cfg,
        gu_queue,
        assoc,
        max_keep=int(cfg.users_obs_max),
    )
    return assoc.astype(np.int32, copy=False), candidates


def queue_aware_bw_policy_from_snapshot(
    snapshot_state: dict[str, Any],
    cfg,
    *,
    assoc: np.ndarray,
    candidates: list[list[int]],
) -> np.ndarray:
    env_state = dict(snapshot_state["env_state"])
    uav_pos = np.asarray(env_state["uav_pos"], dtype=np.float32)
    gu_pos = np.asarray(env_state["gu_pos"], dtype=np.float32)
    gu_queue = np.asarray(env_state["gu_queue"], dtype=np.float32)
    prev_association = np.asarray(env_state["prev_association"], dtype=np.int32)
    eta_ref = _reference_eta_matrix_from_state(cfg, uav_pos=uav_pos, gu_pos=gu_pos)
    bw_alloc = np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    assoc_arr = np.asarray(assoc, dtype=np.int32)
    for u in range(int(cfg.num_uav)):
        cand = [int(idx) for idx in candidates[int(u)][: int(cfg.users_obs_max)]]
        if not cand:
            continue
        weights = np.zeros((len(cand),), dtype=np.float32)
        valid_mask = np.zeros((len(cand),), dtype=bool)
        for slot, gu_idx in enumerate(cand):
            valid = int(assoc_arr[int(gu_idx)]) == int(u)
            valid_mask[slot] = bool(valid)
            if not valid:
                continue
            q = float(gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9))
            eta = float(eta_ref[int(gu_idx), int(u)])
            prev = 1.0 if int(prev_association[int(gu_idx)]) == int(u) else 0.0
            score = q * (0.5 + eta)
            if assoc_bonus > 0.0:
                score = score * (1.0 + assoc_bonus * prev)
            weights[slot] = max(float(score), 0.0)
        denom = float(np.sum(weights))
        if denom > 1.0e-6:
            bw_alloc[int(u), : len(cand)] = weights / denom
        elif np.any(valid_mask):
            bw_alloc[int(u), np.flatnonzero(valid_mask)] = 1.0 / float(np.sum(valid_mask))
    return bw_alloc.astype(np.float32, copy=False)


def predict_access_assoc(
    env,
    model: AccessBidScorer,
    device: torch.device,
    *,
    base_assoc: np.ndarray,
    overlap_per_uav_keep: int,
    overlap_per_user_max_uav: int,
) -> tuple[np.ndarray, list[list[int]]]:
    cfg = env.cfg
    overlap_candidates, _pathloss, d2d = _feasible_overlap_candidate_lists(
        env,
        per_uav_keep=int(overlap_per_uav_keep),
        per_user_max_uav=int(overlap_per_user_max_uav),
    )
    eta_ref = _reference_eta_matrix(env)
    own_assoc_count = np.bincount(
        np.asarray(base_assoc, dtype=np.int32)[np.asarray(base_assoc, dtype=np.int32) >= 0],
        minlength=int(cfg.num_uav),
    ).astype(np.float32, copy=False)
    per_user_uavs: dict[int, list[int]] = {}
    for u, cand_list in enumerate(overlap_candidates):
        for gu_idx in cand_list:
            per_user_uavs.setdefault(int(gu_idx), []).append(int(u))
    assoc = np.asarray(base_assoc, dtype=np.int32).copy()
    model.eval()
    with torch.inference_mode():
        for gu_idx in range(int(cfg.num_gu)):
            cand_uavs = sorted(set(int(u) for u in per_user_uavs.get(int(gu_idx), [])))
            if not cand_uavs:
                continue
            rows: list[list[float]] = []
            for u in cand_uavs:
                rel = (env.gu_pos[int(gu_idx)] - env.uav_pos[int(u)]) / max(float(cfg.map_size), 1.0e-9)
                rows.append(
                    [
                        float(rel[0]),
                        float(rel[1]),
                        float(env.gu_queue[int(gu_idx)] / max(float(cfg.queue_max_gu), 1.0e-9)),
                        float(eta_ref[int(gu_idx), int(u)]),
                        1.0 if int(env.prev_association[int(gu_idx)]) == int(u) else 0.0,
                        1.0 if int(base_assoc[int(gu_idx)]) == int(u) else 0.0,
                        float(d2d[int(gu_idx), int(u)] / max(float(cfg.map_size), 1.0e-9)),
                        float(own_assoc_count[int(u)] / max(float(cfg.num_gu), 1.0)),
                        float(env.uav_queue[int(u)] / max(float(cfg.queue_max_uav), 1.0e-9)),
                        float(env.uav_energy[int(u)] / max(float(cfg.uav_energy_init), 1.0e-9)),
                        float(env.t / max(float(cfg.T_steps), 1.0)),
                    ]
                )
            logits = model(torch.as_tensor(np.asarray(rows, dtype=np.float32), device=device)).squeeze(-1)
            best_idx = int(torch.argmax(logits).item())
            assoc[int(gu_idx)] = int(cand_uavs[best_idx])
    candidates = _assoc_candidate_users(env, assoc, max_keep=int(cfg.users_obs_max))
    return assoc.astype(np.int32, copy=False), candidates
