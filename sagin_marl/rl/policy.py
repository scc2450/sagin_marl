from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

OWN_OBS_DIM = 7
USER_OBS_DIM = 5
SAT_OBS_DIM = 9
NBR_OBS_DIM = 4
DANGER_NBR_OBS_DIM = 5


@dataclass
class PolicyOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    accel: torch.Tensor
    bw_logits: torch.Tensor | None = None
    sat_logits: torch.Tensor | None = None
    dist_out: Dict[str, torch.Tensor] | None = None


def flat_obs_dim(cfg) -> int:
    dim = (
        OWN_OBS_DIM
        + cfg.users_obs_max * USER_OBS_DIM
        + cfg.users_obs_max
        + cfg.sats_obs_max * SAT_OBS_DIM
        + cfg.sats_obs_max
        + cfg.nbrs_obs_max * NBR_OBS_DIM
        + cfg.nbrs_obs_max
    )
    if bool(getattr(cfg, "danger_nbr_enabled", False)):
        dim += DANGER_NBR_OBS_DIM
    return dim


def flatten_obs(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    parts = [
        obs["own"].ravel(),
    ]
    if bool(getattr(cfg, "danger_nbr_enabled", False)):
        parts.append(obs["danger_nbr"].ravel())
    parts.extend(
        [
            obs["users"].ravel(),
            obs["users_mask"].ravel(),
            obs["sats"].ravel(),
            obs["sats_mask"].ravel(),
            obs["nbrs"].ravel(),
            obs["nbrs_mask"].ravel(),
        ]
    )
    return np.concatenate(parts).astype(np.float32)


def batch_flatten_obs(obs_batch: list[Dict[str, np.ndarray]], cfg) -> np.ndarray:
    # obs_batch is a list/dict of per-agent observations
    obs_list = [flatten_obs(obs, cfg) for obs in obs_batch]
    return np.stack(obs_list, axis=0)


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_action(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(z) * scale


def _logprob_from_squashed(dist: Normal, action: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    eps = 1e-4
    t = action / scale
    t = torch.clamp(t, -1 + eps, 1 - eps)
    z = atanh(t)
    logprob = dist.log_prob(z) - torch.log(1 - t.pow(2) + eps)
    if scale != 1.0:
        logprob = logprob - math.log(scale)
    return logprob.sum(-1)


def _make_encoder(in_dim: int, hidden_dim: int, use_input_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    if use_input_norm:
        layers.append(nn.LayerNorm(in_dim))
    layers.extend(
        [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
    )
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.enable_bw = cfg.enable_bw_action
        self.enable_sat = not cfg.fixed_satellite_strategy
        self.bw_scale = float(cfg.bw_logit_scale)
        self.sat_scale = float(cfg.sat_logit_scale)
        self.danger_nbr_enabled = bool(getattr(cfg, "danger_nbr_enabled", False))
        self.obs_dim = int(obs_dim)
        self.expected_obs_dim = flat_obs_dim(cfg)
        if self.obs_dim != self.expected_obs_dim:
            raise ValueError(f"Actor obs_dim={self.obs_dim} does not match expected flat obs dim={self.expected_obs_dim}")
        self.users_obs_max = int(cfg.users_obs_max)
        self.sats_obs_max = int(cfg.sats_obs_max)
        self.nbrs_obs_max = int(cfg.nbrs_obs_max)
        self._users_obs_size = self.users_obs_max * USER_OBS_DIM
        self._sats_obs_size = self.sats_obs_max * SAT_OBS_DIM
        self._nbrs_obs_size = self.nbrs_obs_max * NBR_OBS_DIM

        idx = 0
        self._own_slice = slice(idx, idx + OWN_OBS_DIM)
        idx += OWN_OBS_DIM
        self._danger_nbr_slice = None
        if self.danger_nbr_enabled:
            self._danger_nbr_slice = slice(idx, idx + DANGER_NBR_OBS_DIM)
            idx += DANGER_NBR_OBS_DIM
        self._users_slice = slice(idx, idx + self._users_obs_size)
        idx += self._users_obs_size
        self._users_mask_slice = slice(idx, idx + self.users_obs_max)
        idx += self.users_obs_max
        self._sats_slice = slice(idx, idx + self._sats_obs_size)
        idx += self._sats_obs_size
        self._sats_mask_slice = slice(idx, idx + self.sats_obs_max)
        idx += self.sats_obs_max
        self._nbrs_slice = slice(idx, idx + self._nbrs_obs_size)
        idx += self._nbrs_obs_size
        self._nbrs_mask_slice = slice(idx, idx + self.nbrs_obs_max)
        idx += self.nbrs_obs_max
        if idx != self.expected_obs_dim:
            raise ValueError(f"Cached obs slices end at {idx}, expected {self.expected_obs_dim}")

        self.encoder_type = str(getattr(cfg, "actor_encoder_type", "flat_mlp")).strip().lower()
        if self.encoder_type not in {"flat_mlp", "set_pool"}:
            raise ValueError(f"Unsupported actor_encoder_type='{self.encoder_type}'")

        if self.encoder_type == "flat_mlp":
            self.obs_norm = nn.LayerNorm(obs_dim) if getattr(cfg, "input_norm_enabled", False) else nn.Identity()
            self.fc1 = nn.Linear(obs_dim, cfg.actor_hidden)
            self.fc2 = nn.Linear(cfg.actor_hidden, cfg.actor_hidden)
            self.own_encoder = None
            self.danger_nbr_encoder = None
            self.users_encoder = None
            self.sats_encoder = None
            self.nbrs_encoder = None
            self.fusion_fc1 = None
            self.fusion_fc2 = None
        else:
            embed_dim = int(getattr(cfg, "actor_set_embed_dim", 64))
            if embed_dim <= 0:
                raise ValueError("actor_set_embed_dim must be positive")
            use_input_norm = bool(getattr(cfg, "input_norm_enabled", False))
            self.obs_norm = nn.Identity()
            self.fc1 = None
            self.fc2 = None
            self.own_encoder = _make_encoder(OWN_OBS_DIM, embed_dim, use_input_norm)
            self.danger_nbr_encoder = (
                _make_encoder(DANGER_NBR_OBS_DIM, embed_dim, use_input_norm) if self.danger_nbr_enabled else None
            )
            self.users_encoder = _make_encoder(USER_OBS_DIM, embed_dim, use_input_norm)
            self.sats_encoder = _make_encoder(SAT_OBS_DIM, embed_dim, use_input_norm)
            self.nbrs_encoder = _make_encoder(NBR_OBS_DIM, embed_dim, use_input_norm)
            fusion_in_dim = 2 * embed_dim + 3 * (2 * embed_dim) if self.danger_nbr_enabled else embed_dim + 3 * (2 * embed_dim)
            self.fusion_fc1 = nn.Linear(fusion_in_dim, cfg.actor_hidden)
            self.fusion_fc2 = nn.Linear(cfg.actor_hidden, cfg.actor_hidden)

        self.mu_head = nn.Linear(cfg.actor_hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        if self.enable_bw:
            self.bw_head = nn.Linear(cfg.actor_hidden, cfg.users_obs_max)
            self.bw_log_std = nn.Parameter(torch.zeros(cfg.users_obs_max))
        else:
            self.bw_head = None
            self.bw_log_std = None

        if self.enable_sat:
            self.sat_head = nn.Linear(cfg.actor_hidden, cfg.sats_obs_max)
            self.sat_log_std = nn.Parameter(torch.zeros(cfg.sats_obs_max))
        else:
            self.sat_head = None
            self.sat_log_std = None

    def backbone_modules(self) -> Tuple[nn.Module, ...]:
        if self.encoder_type == "flat_mlp":
            return (self.obs_norm, self.fc1, self.fc2)
        modules: list[nn.Module] = [
            self.own_encoder,
        ]
        if self.danger_nbr_encoder is not None:
            modules.append(self.danger_nbr_encoder)
        modules.extend(
            [
                self.users_encoder,
                self.sats_encoder,
                self.nbrs_encoder,
                self.fusion_fc1,
                self.fusion_fc2,
            ]
        )
        return tuple(modules)

    def _split_flat_obs(
        self, obs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if obs.ndim != 2:
            raise ValueError(f"Expected obs tensor with shape [B, D], got {tuple(obs.shape)}")
        if obs.shape[1] != self.expected_obs_dim:
            raise ValueError(f"Expected obs dim {self.expected_obs_dim}, got {obs.shape[1]}")

        batch_size = obs.shape[0]
        own = obs[:, self._own_slice]
        danger_nbr = obs[:, self._danger_nbr_slice] if self._danger_nbr_slice is not None else None
        users = obs[:, self._users_slice].reshape(batch_size, self.users_obs_max, USER_OBS_DIM)
        users_mask = obs[:, self._users_mask_slice]
        sats = obs[:, self._sats_slice].reshape(batch_size, self.sats_obs_max, SAT_OBS_DIM)
        sats_mask = obs[:, self._sats_mask_slice]
        nbrs = obs[:, self._nbrs_slice].reshape(batch_size, self.nbrs_obs_max, NBR_OBS_DIM)
        nbrs_mask = obs[:, self._nbrs_mask_slice]
        return own, danger_nbr, users, users_mask, sats, sats_mask, nbrs, nbrs_mask

    def _masked_pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = (mask > 0.0).unsqueeze(-1)
        mask_float = mask_bool.to(encoded.dtype)
        mask_sum = mask_float.sum(dim=1)
        mean_feat = (encoded * mask_float).sum(dim=1) / mask_sum.clamp_min(1.0)

        max_feat = encoded.masked_fill(~mask_bool, float("-inf")).amax(dim=1)
        has_any = mask_sum.squeeze(-1) > 0.0
        max_feat = torch.where(has_any.unsqueeze(-1), max_feat, torch.zeros_like(max_feat))
        return torch.cat([mean_feat, max_feat], dim=-1)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "flat_mlp":
            x = self.obs_norm(obs)
            x = F.relu(self.fc1(x))
            return F.relu(self.fc2(x))

        own, danger_nbr, users, users_mask, sats, sats_mask, nbrs, nbrs_mask = self._split_flat_obs(obs)
        own_feat = self.own_encoder(own)
        fused_parts = [own_feat]
        if self.danger_nbr_encoder is not None:
            if danger_nbr is None:
                raise ValueError("danger_nbr slice is missing while danger_nbr_enabled=True")
            fused_parts.append(self.danger_nbr_encoder(danger_nbr))
        users_feat = self._masked_pool(self.users_encoder(users), users_mask)
        sats_feat = self._masked_pool(self.sats_encoder(sats), sats_mask)
        nbrs_feat = self._masked_pool(self.nbrs_encoder(nbrs), nbrs_mask)
        fused = torch.cat([*fused_parts, users_feat, sats_feat, nbrs_feat], dim=-1)
        fused = F.relu(self.fusion_fc1(fused))
        return F.relu(self.fusion_fc2(fused))

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._encode_obs(obs)
        mu = self.mu_head(x)
        out = {"mu": mu}
        if self.bw_head is not None:
            out["bw_mu"] = self.bw_head(x)
        if self.sat_head is not None:
            out["sat_mu"] = self.sat_head(x)
        return out

    def _concat_actions(
        self,
        accel: torch.Tensor,
        bw: torch.Tensor | None,
        sat: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [accel]
        if self.enable_bw and bw is not None:
            parts.append(bw)
        if self.enable_sat and sat is not None:
            parts.append(sat)
        return torch.cat(parts, dim=-1)

    def _split_actions(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        idx = 0
        accel = action[:, idx : idx + 2]
        idx += 2
        bw = None
        sat = None
        if self.enable_bw:
            bw = action[:, idx : idx + self.cfg.users_obs_max]
            idx += self.cfg.users_obs_max
        if self.enable_sat:
            sat = action[:, idx : idx + self.cfg.sats_obs_max]
        return accel, bw, sat

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        out = self.forward(obs)
        mu = out["mu"]
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            z = mu
        else:
            z = dist.rsample()
        accel = _squash_action(z, scale=1.0)
        logprob = _logprob_from_squashed(dist, accel, scale=1.0)

        bw_logits = None
        sat_logits = None

        if self.enable_bw:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            z_bw = bw_mu if deterministic else bw_dist.rsample()
            bw_logits = _squash_action(z_bw, scale=self.bw_scale)
            logprob = logprob + _logprob_from_squashed(bw_dist, bw_logits, scale=self.bw_scale)

        if self.enable_sat:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            z_sat = sat_mu if deterministic else sat_dist.rsample()
            sat_logits = _squash_action(z_sat, scale=self.sat_scale)
            logprob = logprob + _logprob_from_squashed(sat_dist, sat_logits, scale=self.sat_scale)

        action = self._concat_actions(accel, bw_logits, sat_logits)
        return PolicyOutput(
            action=action,
            logprob=logprob,
            accel=accel,
            bw_logits=bw_logits,
            sat_logits=sat_logits,
            dist_out=out,
        )

    def evaluate_actions_parts(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not torch.isfinite(obs).all():
            print("NaN/Inf detected in obs passed to evaluate_actions_parts")
            raise ValueError("obs contains NaN/Inf")
        if out is None:
            out = self.forward(obs)
        accel_action, bw_action, sat_action = self._split_actions(action)

        mu = out["mu"]
        if not torch.isfinite(mu).all():
            print("NaN/Inf detected in actor mu inside evaluate_actions_parts")
            raise ValueError("actor mu contains NaN/Inf")
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        if not torch.isfinite(std).all():
            print("NaN/Inf detected in actor std inside evaluate_actions_parts")
            raise ValueError("actor std contains NaN/Inf")
        dist = Normal(mu, std)
        logprob_parts: Dict[str, torch.Tensor] = {"accel": _logprob_from_squashed(dist, accel_action, scale=1.0)}
        entropy_parts: Dict[str, torch.Tensor] = {"accel": dist.entropy().sum(-1)}

        if self.enable_bw and bw_action is not None:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            logprob_parts["bw"] = _logprob_from_squashed(bw_dist, bw_action, scale=self.bw_scale)
            entropy_parts["bw"] = bw_dist.entropy().sum(-1)

        if self.enable_sat and sat_action is not None:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            logprob_parts["sat"] = _logprob_from_squashed(sat_dist, sat_action, scale=self.sat_scale)
            entropy_parts["sat"] = sat_dist.entropy().sum(-1)

        return logprob_parts, entropy_parts

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logprob_parts, entropy_parts = self.evaluate_actions_parts(obs, action, out=out)
        logprob = sum(logprob_parts.values())
        entropy = sum(entropy_parts.values())
        return logprob, entropy
