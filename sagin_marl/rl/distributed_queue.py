"""Observation-only distributed queue-aware receding-horizon movement.

Rows are independent UAV observations. No cluster labels, global GU state,
other agents' observations, or intent messages are accepted by this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class DQSettings:
    horizon_steps: int = 5
    movement_weight: float = 0.02
    switch_weight: float = 0.01
    overlap_weight: float = 0.25
    safety_margin: float = 5.0
    risk_weight: float = 0.25


def _risk_adjustment(obs, cfg, actions, trajectories, base_score, boundary_ok, settings):
    """One-step radial robust screen plus a soft two-step lookahead cost.

    The bound assumes observed peer velocity, bounded executed acceleration and
    straight interpolation within a simulator step; it is not a global guarantee.
    """
    ego, mask = obs.ego_features, obs.peer_mask
    if obs.peer_tokens.shape[1] == 0:
        clear = torch.full_like(base_score, float("inf"))
        return base_score, boundary_ok, None, clear
    dt, amax = float(cfg.tau0), float(cfg.a_max)
    peer = torch.where(mask[..., None], obs.peer_tokens, 0.0)
    # Observation contract is ego minus peer for both relative fields.
    rel = -peer[..., :2] * float(cfg.map_size)
    dist = rel.norm(dim=-1).clamp_min(1e-6)
    normal = rel / dist[..., None]
    own_pos = ego[:, :2] * float(cfg.map_size)
    own_vel = ego[:, 2:4] * float(cfg.v_max)
    peer_vel = own_vel[:, None] - peer[..., 2:4] * float(cfg.v_max)
    own_delta = trajectories[0] - own_pos[:, None]
    nominal_rel = rel[:, None] + peer_vel[:, None] * dt - own_delta[:, :, None]
    # Semi-implicit simulator motion: acceleration contributes amax * dt**2,
    # not 0.5 * amax * dt**2, to the next position.
    endpoint = (nominal_rel * normal[:, None]).sum(-1) - amax * dt * dt
    endpoint = endpoint.masked_fill(~mask[:, None], float("inf"))
    current = dist.masked_fill(~mask, float("inf"))
    robust_clear = torch.minimum(current[:, None], endpoint).amin(-1)
    physical = float(cfg.d_safe)
    # Starting within the soft buffer is allowed. Already-violated physical
    # separation instead requires predicted non-worsening recovery.
    required = torch.minimum(current, torch.full_like(current, physical))
    permitted = (endpoint >= required[:, None]).all(-1) & boundary_ok
    h = min(2, len(trajectories))
    future_rel = rel[:, None] + peer_vel[:, None] * (h * dt) - (
        trajectories[h - 1] - own_pos[:, None])[:, :, None]
    uncertainty = amax * dt * dt * h * (h + 1) / 2
    future_clear = (future_rel.norm(dim=-1) - uncertainty).masked_fill(
        ~mask[:, None], float("inf"))
    buffer = physical + settings.safety_margin
    risk = ((buffer - future_clear) / max(buffer, 1)).clamp_min(0).square().amax(-1)
    score = base_score - settings.risk_weight * risk
    # Recovery prioritizes next-step separation, then lower closing speed.
    endpoint_min = endpoint.amin(-1)
    best_endpoint = endpoint_min.masked_fill(~boundary_ok, -float("inf")).amax(-1, keepdim=True)
    recoverable = boundary_ok & (endpoint_min >= best_endpoint - 1e-4)
    own_next_vel = own_delta / dt
    closing = (-(peer_vel[:, None] - own_next_vel[:, :, None]) * normal[:, None]).sum(-1)
    closing = closing.masked_fill(~mask[:, None], -float("inf")).amax(-1).nan_to_num(neginf=0)
    recovery_score = -closing - 0.01 * actions.square().sum(-1)
    fallback = recovery_score.masked_fill(~recoverable, -float("inf")).argmax(-1)
    fallback = torch.where(boundary_ok.any(-1), fallback, torch.full_like(fallback, actions.shape[1] - 1))
    return score, permitted, fallback, robust_clear


@torch.no_grad()
def distributed_queue_action(obs, cfg, variant="c", settings=DQSettings()):
    if variant not in {"a", "b", "c"}:
        raise ValueError(f"Unknown distributed queue variant: {variant}")
    if settings.horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    ego = obs.ego_features
    gu = torch.where(obs.gu_mask[..., None], obs.gu_tokens, 0.0)
    peer = torch.where(obs.peer_mask[..., None], obs.peer_tokens, 0.0)
    valid = obs.gu_mask
    n = ego.shape[0]
    dt, vmax, amax = float(cfg.tau0), float(cfg.v_max), float(cfg.a_max)
    map_size, height = float(cfg.map_size), float(cfg.uav_height)
    pos, vel = ego[:, :2] * map_size, ego[:, 2:4] * vmax
    angles = torch.arange(8, device=ego.device, dtype=ego.dtype) * (math.pi / 4)
    directions = torch.stack((angles.cos(), angles.sin()), -1)
    fixed = torch.cat((directions * 0.5, directions, directions[:1] * 0), 0)
    brake = -vel / (amax * dt)
    brake = brake / brake.norm(dim=-1, keepdim=True).clamp_min(1)
    actions = torch.cat((fixed[None].expand(n, -1, -1), brake[:, None]), 1)
    k = actions.shape[1]
    p = pos[:, None].expand(-1, k, -1).clone()
    v = vel[:, None].expand(-1, k, -1).clone()
    distance = torch.zeros((n, k), device=ego.device, dtype=ego.dtype)
    clearance = torch.full_like(distance, float("inf"))
    boundary_ok = torch.ones_like(distance, dtype=torch.bool)
    trajectories = []
    # Peer observations encode ego minus peer, for position and velocity.
    peer_pos = pos[:, None] - peer[..., :2] * map_size
    peer_vel = vel[:, None] - peer[..., 2:4] * vmax
    for step in range(settings.horizon_steps):
        old_p = p
        v = v + actions * (amax * dt)
        v = v * (vmax / v.norm(dim=-1, keepdim=True).clamp_min(vmax))
        p = p + v * dt
        distance += (p - old_p).norm(dim=-1)
        boundary_ok &= ((p >= 0) & (p <= map_size)).all(-1)
        trajectories.append(p)
        # Minimum relative distance along each piecewise-linear prediction segment.
        rel0 = peer_pos[:, None] + peer_vel[:, None] * (step * dt) - old_p[:, :, None]
        rel_delta = peer_vel[:, None] * dt - (p - old_p)[:, :, None]
        t = (-(rel0 * rel_delta).sum(-1) / rel_delta.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
        sep = (rel0 + t[..., None] * rel_delta).norm(dim=-1)
        sep = sep.masked_fill(~obs.peer_mask[:, None], float("inf"))
        if sep.shape[-1]:
            clearance = torch.minimum(clearance, sep.amin(-1))

    # Decode the current token schema. Expected arrival is log1p-normalized.
    # Homogeneous configured mean is public, not another UAV's private state.
    level = int(cfg.traffic_level)
    ratio = (cfg.traffic_level_nav_ratio if level <= 0 else
             cfg.traffic_level_easy_ratio if level == 1 else cfg.traffic_level_hard_ratio)
    flow_ref = float(cfg.task_arrival_rate) * float(ratio) * dt
    queued = gu[..., 3].clamp_min(0) * float(cfg.queue_max_gu)
    arrival = gu[..., 4].clamp(0, 20).expm1() * flow_ref
    demand = (queued + settings.horizon_steps * arrival) * valid
    beta = demand / demand.sum(-1, keepdim=True).clamp_min(1)
    gu_pos = pos[:, None] + gu[..., 12:14] * map_size
    initial_d2 = (gu[..., 12:14] * map_size).square().sum(-1) + height**2
    # Local link extrapolation: observed full-band SE, fixed interference,
    # inverse-square distance ratio. This is a proxy, not the full channel model.
    snr0 = torch.exp2(gu[..., 15].clamp(0, 30)) - 1
    service_cap = torch.zeros((n, k, gu.shape[1]), device=ego.device, dtype=ego.dtype)
    for p_t in trajectories:
        d2 = (gu_pos[:, None] - p_t[:, :, None]).square().sum(-1) + height**2
        se = torch.log2(1 + snr0[:, None] * initial_d2[:, None] / d2.clamp_min(1))
        service_cap += beta[:, None] * float(cfg.b_acc) * se * dt
    served = torch.minimum(demand[:, None], service_cap)
    denom = demand.sum(-1, keepdim=True).clamp_min(1)
    benefit = served.sum(-1) / denom
    overlap = torch.zeros_like(benefit)
    if variant in {"b", "c"} and peer.shape[1]:
        predicted_peer = peer_pos + peer_vel * (settings.horizon_steps * dt)
        peer_d = (gu_pos[:, :, None] - predicted_peer[:, None]).norm(dim=-1)
        peer_d = peer_d.masked_fill(~obs.peer_mask[:, None], float("inf")).amin(-1)
        own_d = (gu_pos[:, None] - p[:, :, None]).norm(dim=-1)
        # A soft ownership preference; never remove users merely because a peer exists.
        competing = ((own_d - peer_d[:, None]) / max(height, 1)).sigmoid()
        overlap = (served * competing).sum(-1) / denom
    switch = (actions - ego[:, None, 20:22]).square().sum(-1)
    score = (benefit - settings.overlap_weight * overlap
             - settings.movement_weight * distance / (vmax * dt * settings.horizon_steps)
             - settings.switch_weight * switch)
    allowed = boundary_ok
    risk_fallback = None
    robust_clear = clearance
    base_score = score
    if variant == "c":
        score, allowed, risk_fallback, robust_clear = _risk_adjustment(
            obs, cfg, actions, trajectories, score, boundary_ok, settings)
    has_allowed = allowed.any(-1)
    best = score.masked_fill(~allowed, -float("inf")).argmax(-1)
    # If all predictions are unsafe, maximize clearance inside map bounds.
    # No safety guarantee under unobserved neighbor acceleration is claimed.
    fallback_clearance = clearance.nan_to_num(posinf=map_size)
    fallback = fallback_clearance.masked_fill(~boundary_ok, -float("inf")).argmax(-1)
    fallback = torch.where(boundary_ok.any(-1), fallback, torch.full_like(fallback, k - 1))
    if risk_fallback is not None:
        fallback = risk_fallback
    best = torch.where(has_allowed, best, fallback)
    chosen = actions[torch.arange(n, device=ego.device), best]
    return chosen, {"predicted_clearance": clearance.gather(1, best[:, None]).squeeze(1),
                    "fallback": ~has_allowed, "score": score.gather(1, best[:, None]).squeeze(1),
                    "candidate_actions": actions, "candidate_scores": score,
                    "candidate_clearance": clearance, "candidate_allowed": allowed,
                    "candidate_boundary_ok": boundary_ok, "selected": best,
                    "first_positions": trajectories[0],
                    "base_scores": base_score, "robust_first_clearance": robust_clear}
