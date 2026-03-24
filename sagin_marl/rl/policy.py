from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import HybridActionDist

OWN_OBS_DIM = 7
USER_OBS_DIM = 5
SAT_OBS_DIM = 12
NBR_OBS_DIM = 4
DANGER_NBR_OBS_DIM = 5


@dataclass
class PolicyOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    accel: torch.Tensor
    bw_action: torch.Tensor | None = None
    sat_select_mask: torch.Tensor | None = None
    sat_indices: torch.Tensor | None = None
    dist_out: Dict[str, torch.Tensor] | None = None

    @property
    def bw_logits(self) -> torch.Tensor | None:
        return self.bw_action

    @property
    def sat_logits(self) -> torch.Tensor | None:
        return self.sat_select_mask


def flat_obs_dim(cfg) -> int:
    dim = (
        OWN_OBS_DIM
        + cfg.users_obs_max * USER_OBS_DIM
        + cfg.users_obs_max
        + cfg.users_obs_max
        + cfg.sats_obs_max * SAT_OBS_DIM
        + cfg.sats_obs_max
        + cfg.sats_obs_max
        + cfg.nbrs_obs_max * NBR_OBS_DIM
        + cfg.nbrs_obs_max
    )
    if bool(getattr(cfg, "danger_nbr_enabled", False)):
        dim += DANGER_NBR_OBS_DIM
    return dim


def flatten_obs(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    bw_valid_mask = obs.get("bw_valid_mask", obs["users_mask"])
    sat_valid_mask = obs.get("sat_valid_mask", obs["sats_mask"])
    parts = [
        obs["own"].ravel(),
    ]
    if bool(getattr(cfg, "danger_nbr_enabled", False)):
        parts.append(obs["danger_nbr"].ravel())
    parts.extend(
        [
            obs["users"].ravel(),
            obs["users_mask"].ravel(),
            np.asarray(bw_valid_mask, dtype=np.float32).ravel(),
            obs["sats"].ravel(),
            obs["sats_mask"].ravel(),
            np.asarray(sat_valid_mask, dtype=np.float32).ravel(),
            obs["nbrs"].ravel(),
            obs["nbrs_mask"].ravel(),
        ]
    )
    return np.concatenate(parts).astype(np.float32)


def batch_flatten_obs(obs_batch: list[Dict[str, np.ndarray]], cfg) -> np.ndarray:
    obs_list = [flatten_obs(obs, cfg) for obs in obs_batch]
    return np.stack(obs_list, axis=0)


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


def _make_scorer(in_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.enable_bw = bool(cfg.enable_bw_action)
        self.enable_sat = not bool(cfg.fixed_satellite_strategy)
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
        self.sat_num_select = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 0)

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
        self._bw_valid_mask_slice = slice(idx, idx + self.users_obs_max)
        idx += self.users_obs_max
        self._sats_slice = slice(idx, idx + self._sats_obs_size)
        idx += self._sats_obs_size
        self._sats_mask_slice = slice(idx, idx + self.sats_obs_max)
        idx += self.sats_obs_max
        self._sat_valid_mask_slice = slice(idx, idx + self.sats_obs_max)
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

        use_input_norm = bool(getattr(cfg, "input_norm_enabled", False))
        if self.encoder_type == "flat_mlp":
            self.obs_norm = nn.LayerNorm(obs_dim) if use_input_norm else nn.Identity()
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
            fusion_in_dim = (
                2 * embed_dim + 3 * (2 * embed_dim)
                if self.danger_nbr_enabled
                else embed_dim + 3 * (2 * embed_dim)
            )
            self.fusion_fc1 = nn.Linear(fusion_in_dim, cfg.actor_hidden)
            self.fusion_fc2 = nn.Linear(cfg.actor_hidden, cfg.actor_hidden)

        elem_embed_dim = int(getattr(cfg, "actor_set_embed_dim", 64))
        elem_embed_dim = max(elem_embed_dim, 16)
        self.mu_head = nn.Linear(cfg.actor_hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        if self.enable_bw:
            self.bw_user_encoder = _make_encoder(USER_OBS_DIM, elem_embed_dim, use_input_norm)
            self.bw_scorer = _make_scorer(cfg.actor_hidden + elem_embed_dim + USER_OBS_DIM)
            if bool(getattr(cfg, "bw_head_zero_init", False)):
                final = self.bw_scorer[-1]
                if isinstance(final, nn.Linear):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
        else:
            self.bw_user_encoder = None
            self.bw_scorer = None

        if self.enable_sat:
            self.sat_action_encoder = _make_encoder(SAT_OBS_DIM, elem_embed_dim, use_input_norm)
            self.sat_scorer = _make_scorer(cfg.actor_hidden + elem_embed_dim + SAT_OBS_DIM)
        else:
            self.sat_action_encoder = None
            self.sat_scorer = None

    def backbone_modules(self) -> Tuple[nn.Module, ...]:
        if self.encoder_type == "flat_mlp":
            return (self.obs_norm, self.fc1, self.fc2)
        modules: list[nn.Module] = [self.own_encoder]
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
        bw_valid_mask = obs[:, self._bw_valid_mask_slice]
        sats = obs[:, self._sats_slice].reshape(batch_size, self.sats_obs_max, SAT_OBS_DIM)
        sats_mask = obs[:, self._sats_mask_slice]
        sat_valid_mask = obs[:, self._sat_valid_mask_slice]
        nbrs = obs[:, self._nbrs_slice].reshape(batch_size, self.nbrs_obs_max, NBR_OBS_DIM)
        nbrs_mask = obs[:, self._nbrs_mask_slice]
        return own, danger_nbr, users, users_mask, bw_valid_mask, sats, sats_mask, sat_valid_mask, nbrs, nbrs_mask

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

        own, danger_nbr, users, users_mask, _, sats, sats_mask, _, nbrs, nbrs_mask = self._split_flat_obs(obs)
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
        own, danger_nbr, users, users_mask, bw_valid_mask, sats, sats_mask, sat_valid_mask, nbrs, nbrs_mask = self._split_flat_obs(obs)
        del own, danger_nbr, users_mask, sats_mask, nbrs, nbrs_mask
        out: Dict[str, torch.Tensor] = {
            "ctx": x,
            "mu": self.mu_head(x),
            "users": users,
            "bw_valid_mask": bw_valid_mask,
            "sats": sats,
            "sat_valid_mask": sat_valid_mask,
        }
        if self.enable_bw and self.bw_user_encoder is not None and self.bw_scorer is not None:
            user_emb = self.bw_user_encoder(users)
            ctx_u = x.unsqueeze(1).expand(-1, self.users_obs_max, -1)
            bw_in = torch.cat([ctx_u, user_emb, users], dim=-1)
            bw_score = self.bw_scorer(bw_in).squeeze(-1)
            alpha_floor = max(float(getattr(self.cfg, "bw_alpha_floor", 0.2) or 0.0), 1e-4)
            out["bw_alpha"] = F.softplus(bw_score) + alpha_floor
        if self.enable_sat and self.sat_action_encoder is not None and self.sat_scorer is not None:
            sat_emb = self.sat_action_encoder(sats)
            ctx_s = x.unsqueeze(1).expand(-1, self.sats_obs_max, -1)
            sat_in = torch.cat([ctx_s, sat_emb, sats], dim=-1)
            out["sat_logits"] = self.sat_scorer(sat_in).squeeze(-1)
        return out

    def _build_hybrid_dist(self, out: Dict[str, torch.Tensor]) -> HybridActionDist:
        return HybridActionDist(
            accel_mu=out["mu"],
            accel_log_std=self.log_std,
            bw_alpha=out.get("bw_alpha"),
            bw_mask=out.get("bw_valid_mask"),
            sat_logits=out.get("sat_logits"),
            sat_mask=out.get("sat_valid_mask"),
            sat_num_select=self.sat_num_select,
        )

    def _split_env_action(
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
        hybrid_dist = self._build_hybrid_dist(out)
        sample = hybrid_dist.sample(deterministic=deterministic)
        logprob = sum(sample.logprob_parts.values())
        return PolicyOutput(
            action=sample.env_action,
            logprob=logprob,
            accel=sample.accel,
            bw_action=sample.bw_action,
            sat_select_mask=sample.sat_select_mask,
            sat_indices=sample.sat_indices,
            dist_out=out,
        )

    def evaluate_actions_parts(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        sat_indices: torch.Tensor | None = None,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not torch.isfinite(obs).all():
            raise ValueError("obs contains NaN/Inf")
        if out is None:
            out = self.forward(obs)
        accel_action, bw_action, _ = self._split_env_action(action)
        hybrid_dist = self._build_hybrid_dist(out)
        logprob_parts = hybrid_dist.log_prob(accel_action, bw_action=bw_action, sat_indices=sat_indices)
        entropy_parts = hybrid_dist.entropy()
        return logprob_parts, entropy_parts

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        sat_indices: torch.Tensor | None = None,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logprob_parts, entropy_parts = self.evaluate_actions_parts(obs, action, sat_indices=sat_indices, out=out)
        logprob = sum(logprob_parts.values())
        entropy = sum(entropy_parts.values())
        return logprob, entropy
