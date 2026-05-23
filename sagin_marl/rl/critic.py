from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import (
    DANGER_NBR_OBS_DIM,
    NBR_OBS_DIM,
    OWN_OBS_DIM,
    SAT_OBS_DIM,
    USER_OBS_DIM,
    flat_obs_dim,
)


VALUE_HEAD_NAMES = ("accel", "bw", "sat")


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


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, obs_dim: int, num_agents: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.num_agents = int(num_agents)
        self.expected_obs_dim = flat_obs_dim(cfg)
        if self.obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"Critic obs_dim={self.obs_dim} does not match expected flat obs dim={self.expected_obs_dim}"
            )

        self.danger_nbr_enabled = bool(getattr(cfg, "danger_nbr_enabled", False))
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
            raise ValueError(f"Cached critic obs slices end at {idx}, expected {self.expected_obs_dim}")

        use_input_norm = bool(getattr(cfg, "input_norm_enabled", False))
        hidden_dim = int(cfg.critic_hidden)
        embed_dim = int(getattr(cfg, "critic_set_embed_dim", getattr(cfg, "actor_set_embed_dim", 64)))
        if embed_dim <= 0:
            raise ValueError("critic_set_embed_dim must be positive")
        self.value_agg = str(getattr(cfg, "critic_value_agg", "agent_mean") or "agent_mean").strip().lower()
        if self.value_agg not in ("agent_mean", "pooled_feature"):
            raise ValueError("critic_value_agg must be one of: agent_mean, pooled_feature")
        self.multihead_value_enabled = bool(getattr(cfg, "critic_multihead_value_enabled", False))
        self.head_feature_mode = str(getattr(cfg, "critic_head_feature_mode", "shared_full") or "shared_full")
        self.head_feature_mode = self.head_feature_mode.strip().lower()
        if self.head_feature_mode not in ("shared_full",):
            raise ValueError("critic_head_feature_mode must be one of: shared_full")

        self.state_norm = nn.LayerNorm(state_dim) if use_input_norm else nn.Identity()
        self.global_fc1 = nn.Linear(state_dim, hidden_dim)
        self.global_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.own_encoder = _make_encoder(OWN_OBS_DIM, embed_dim, use_input_norm)
        self.danger_nbr_encoder = (
            _make_encoder(DANGER_NBR_OBS_DIM, embed_dim, use_input_norm) if self.danger_nbr_enabled else None
        )
        self.users_encoder = _make_encoder(USER_OBS_DIM, embed_dim, use_input_norm)
        self.sats_encoder = _make_encoder(SAT_OBS_DIM, embed_dim, use_input_norm)
        self.nbrs_encoder = _make_encoder(NBR_OBS_DIM, embed_dim, use_input_norm)

        local_in_dim = embed_dim + 5 * (2 * embed_dim)
        if self.danger_nbr_enabled:
            local_in_dim += embed_dim
        self.local_fc1 = nn.Linear(local_in_dim, hidden_dim)
        self.local_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.derived_dim = 25
        self.value_fc1 = nn.Linear(2 * hidden_dim + self.derived_dim, hidden_dim)
        if self.multihead_value_enabled:
            self.value_feature_gates = nn.ModuleDict({name: nn.Identity() for name in VALUE_HEAD_NAMES})
            self.value_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in VALUE_HEAD_NAMES})
        else:
            self.value_fc2 = nn.Linear(hidden_dim, 1)
        if self.value_agg == "pooled_feature":
            self.value_pool_fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
            if not self.multihead_value_enabled:
                self.value_pool_fc2 = nn.Linear(hidden_dim, 1)

    def _split_obs_step(
        self, obs_step: torch.Tensor
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
        if obs_step.ndim != 3:
            raise ValueError(f"Expected obs_step tensor with shape [B, N, D], got {tuple(obs_step.shape)}")
        if obs_step.shape[1] != self.num_agents:
            raise ValueError(f"Expected obs_step agent dim {self.num_agents}, got {obs_step.shape[1]}")
        if obs_step.shape[2] != self.expected_obs_dim:
            raise ValueError(f"Expected obs dim {self.expected_obs_dim}, got {obs_step.shape[2]}")

        batch_size, num_agents, _ = obs_step.shape
        own = obs_step[:, :, self._own_slice]
        danger_nbr = obs_step[:, :, self._danger_nbr_slice] if self._danger_nbr_slice is not None else None
        users = obs_step[:, :, self._users_slice].reshape(batch_size, num_agents, self.users_obs_max, USER_OBS_DIM)
        users_mask = obs_step[:, :, self._users_mask_slice]
        bw_valid_mask = obs_step[:, :, self._bw_valid_mask_slice]
        sats = obs_step[:, :, self._sats_slice].reshape(batch_size, num_agents, self.sats_obs_max, SAT_OBS_DIM)
        sats_mask = obs_step[:, :, self._sats_mask_slice]
        sat_valid_mask = obs_step[:, :, self._sat_valid_mask_slice]
        nbrs = obs_step[:, :, self._nbrs_slice].reshape(batch_size, num_agents, self.nbrs_obs_max, NBR_OBS_DIM)
        nbrs_mask = obs_step[:, :, self._nbrs_mask_slice]
        return own, danger_nbr, users, users_mask, bw_valid_mask, sats, sats_mask, sat_valid_mask, nbrs, nbrs_mask

    @staticmethod
    def _encode_features(encoder: nn.Module, feat: torch.Tensor) -> torch.Tensor:
        flat = feat.reshape(-1, feat.shape[-1])
        encoded = encoder(flat)
        return encoded.reshape(*feat.shape[:-1], encoded.shape[-1])

    @staticmethod
    def _masked_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = (mask > 0.0).unsqueeze(-1)
        mask_float = mask_bool.to(encoded.dtype)
        mask_sum = mask_float.sum(dim=2)
        mean_feat = (encoded * mask_float).sum(dim=2) / mask_sum.clamp_min(1.0)

        max_feat = encoded.masked_fill(~mask_bool, float("-inf")).amax(dim=2)
        has_any = mask_sum.squeeze(-1) > 0.0
        max_feat = torch.where(has_any.unsqueeze(-1), max_feat, torch.zeros_like(max_feat))
        return torch.cat([mean_feat, max_feat], dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = (mask > 0.0).to(values.dtype)
        denom = mask_float.sum(dim=2).clamp_min(1.0)
        return (values * mask_float).sum(dim=2) / denom

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask > 0.0
        max_values = values.masked_fill(~mask_bool, float("-inf")).amax(dim=2)
        return torch.where(mask_bool.any(dim=2), max_values, torch.zeros_like(max_values))

    @staticmethod
    def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask > 0.0
        min_values = values.masked_fill(~mask_bool, float("inf")).amin(dim=2)
        return torch.where(mask_bool.any(dim=2), min_values, torch.zeros_like(min_values))

    @staticmethod
    def _masked_top1_gap(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask > 0.0
        topk = torch.topk(values.masked_fill(~mask_bool, float("-inf")), k=min(2, values.shape[2]), dim=2).values
        gap = topk[..., 0] - topk[..., 1] if values.shape[2] >= 2 else torch.zeros_like(topk[..., 0])
        valid_two = mask_bool.sum(dim=2) >= 2
        return torch.where(valid_two, gap, torch.zeros_like(gap))

    def _derived_stats(
        self,
        own: torch.Tensor,
        danger_nbr: torch.Tensor | None,
        users: torch.Tensor,
        users_mask: torch.Tensor,
        bw_valid_mask: torch.Tensor,
        sats: torch.Tensor,
        sats_mask: torch.Tensor,
        sat_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_eta = users[..., 3]
        user_queue = users[..., 2]
        user_prev_assoc = users[..., 4]

        sat_se = sats[..., 7]
        sat_queue = sats[..., 8]
        sat_load = sats[..., 9]
        sat_projected = 1.0 / sats[..., 10].clamp_min(1e-6)
        sat_stay = sats[..., 11]

        visible_user_ratio = users_mask.sum(dim=2) / max(float(self.users_obs_max), 1.0)
        valid_user_ratio = bw_valid_mask.sum(dim=2) / max(float(self.users_obs_max), 1.0)
        user_eta_mean = self._masked_mean(user_eta, bw_valid_mask)
        user_eta_max = self._masked_max(user_eta, bw_valid_mask)
        user_eta_gap = self._masked_top1_gap(user_eta, bw_valid_mask)
        user_queue_mean = self._masked_mean(user_queue, bw_valid_mask)
        user_queue_max = self._masked_max(user_queue, bw_valid_mask)
        user_prev_assoc_ratio = self._masked_mean(user_prev_assoc, users_mask)

        visible_sat_ratio = sats_mask.sum(dim=2) / max(float(self.sats_obs_max), 1.0)
        valid_sat_ratio = sat_valid_mask.sum(dim=2) / max(float(self.sats_obs_max), 1.0)
        sat_se_mean = self._masked_mean(sat_se, sat_valid_mask)
        sat_se_max = self._masked_max(sat_se, sat_valid_mask)
        sat_se_gap = self._masked_top1_gap(sat_se, sat_valid_mask)
        sat_queue_mean = self._masked_mean(sat_queue, sat_valid_mask)
        sat_queue_min = self._masked_min(sat_queue, sat_valid_mask)
        sat_load_mean = self._masked_mean(sat_load, sat_valid_mask)
        sat_load_max = self._masked_max(sat_load, sat_valid_mask)
        sat_projected_mean = self._masked_mean(sat_projected, sat_valid_mask)
        sat_projected_min = self._masked_min(sat_projected, sat_valid_mask)
        sat_stay_ratio = self._masked_mean(sat_stay, sats_mask)

        assoc_count = own[..., 7]
        assoc_centroid_x = own[..., 8]
        assoc_centroid_y = own[..., 9]
        if danger_nbr is None:
            danger_dist = torch.zeros_like(assoc_count)
            danger_closing = torch.zeros_like(assoc_count)
        else:
            danger_dist = danger_nbr[..., 0]
            danger_closing = danger_nbr[..., 1]

        stats = [
            assoc_count,
            assoc_centroid_x,
            assoc_centroid_y,
            danger_dist,
            danger_closing,
            visible_user_ratio,
            valid_user_ratio,
            user_eta_mean,
            user_eta_max,
            user_eta_gap,
            user_queue_mean,
            user_queue_max,
            user_prev_assoc_ratio,
            visible_sat_ratio,
            valid_sat_ratio,
            sat_se_mean,
            sat_se_max,
            sat_se_gap,
            sat_queue_mean,
            sat_queue_min,
            sat_load_mean,
            sat_load_max,
            sat_projected_mean,
            sat_projected_min,
            sat_stay_ratio,
        ]
        return torch.stack(stats, dim=-1)

    def _build_agent_value_features(self, state: torch.Tensor, obs_step: torch.Tensor) -> torch.Tensor:
        own, danger_nbr, users, users_mask, bw_valid_mask, sats, sats_mask, sat_valid_mask, nbrs, nbrs_mask = (
            self._split_obs_step(obs_step)
        )

        g_ctx = self.state_norm(state)
        g_ctx = F.relu(self.global_fc1(g_ctx))
        g_ctx = F.relu(self.global_fc2(g_ctx))

        users_encoded = self._encode_features(self.users_encoder, users)
        sats_encoded = self._encode_features(self.sats_encoder, sats)
        nbrs_encoded = self._encode_features(self.nbrs_encoder, nbrs)

        local_parts = [self._encode_features(self.own_encoder, own)]
        if self.danger_nbr_encoder is not None:
            if danger_nbr is None:
                raise ValueError("danger_nbr slice is missing while danger_nbr_enabled=True")
            local_parts.append(self._encode_features(self.danger_nbr_encoder, danger_nbr))
        local_parts.extend(
            [
                self._masked_pool(users_encoded, users_mask),
                self._masked_pool(users_encoded, bw_valid_mask),
                self._masked_pool(sats_encoded, sats_mask),
                self._masked_pool(sats_encoded, sat_valid_mask),
                self._masked_pool(nbrs_encoded, nbrs_mask),
            ]
        )
        local_ctx = torch.cat(local_parts, dim=-1)
        local_ctx = F.relu(self.local_fc1(local_ctx))
        local_ctx = F.relu(self.local_fc2(local_ctx))

        derived = self._derived_stats(
            own,
            danger_nbr,
            users,
            users_mask,
            bw_valid_mask,
            sats,
            sats_mask,
            sat_valid_mask,
        )
        g_ctx_expanded = g_ctx.unsqueeze(1).expand(-1, obs_step.shape[1], -1)
        value_input = torch.cat([g_ctx_expanded, local_ctx, derived], dim=-1)
        return F.relu(self.value_fc1(value_input))

    def _apply_head_feature_gate(self, head_name: str, agent_value_features: torch.Tensor) -> torch.Tensor:
        if head_name not in VALUE_HEAD_NAMES:
            raise KeyError(f"Unknown value head: {head_name}")
        if self.head_feature_mode == "shared_full":
            return self.value_feature_gates[head_name](agent_value_features)
        raise ValueError(f"Unsupported critic_head_feature_mode: {self.head_feature_mode}")

    def _forward_scalar_value_from_features(self, agent_value_features: torch.Tensor) -> torch.Tensor:
        if self.value_agg == "agent_mean":
            agent_values = self.value_fc2(agent_value_features).squeeze(-1)
            return agent_values.mean(dim=1)

        pooled_features = torch.cat(
            [agent_value_features.mean(dim=1), agent_value_features.amax(dim=1)],
            dim=-1,
        )
        pooled_hidden = F.relu(self.value_pool_fc1(pooled_features))
        return self.value_pool_fc2(pooled_hidden).squeeze(-1)

    def _forward_head_values_from_features(self, agent_value_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_values: Dict[str, torch.Tensor] = {}
        for head_name in VALUE_HEAD_NAMES:
            gated_features = self._apply_head_feature_gate(head_name, agent_value_features)
            if self.value_agg == "agent_mean":
                agent_values = self.value_heads[head_name](gated_features).squeeze(-1)
                head_values[head_name] = agent_values.mean(dim=1)
                continue

            pooled_features = torch.cat(
                [gated_features.mean(dim=1), gated_features.amax(dim=1)],
                dim=-1,
            )
            pooled_hidden = F.relu(self.value_pool_fc1(pooled_features))
            head_values[head_name] = self.value_heads[head_name](pooled_hidden).squeeze(-1)
        return head_values

    def forward_heads(self, state: torch.Tensor, obs_step: torch.Tensor) -> Dict[str, torch.Tensor]:
        agent_value_features = self._build_agent_value_features(state, obs_step)
        if not self.multihead_value_enabled:
            shared_value = self._forward_scalar_value_from_features(agent_value_features)
            return {head_name: shared_value for head_name in VALUE_HEAD_NAMES}
        return self._forward_head_values_from_features(agent_value_features)

    def forward(self, state: torch.Tensor, obs_step: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"Expected state tensor with shape [B, S], got {tuple(state.shape)}")
        if obs_step.ndim != 3:
            raise ValueError(f"Expected obs_step tensor with shape [B, N, D], got {tuple(obs_step.shape)}")
        if state.shape[0] != obs_step.shape[0]:
            raise ValueError(
                f"State batch size {state.shape[0]} does not match obs_step batch size {obs_step.shape[0]}"
            )
        agent_value_features = self._build_agent_value_features(state, obs_step)
        if not self.multihead_value_enabled:
            return self._forward_scalar_value_from_features(agent_value_features)

        head_values = self._forward_head_values_from_features(agent_value_features)
        return torch.stack([head_values[head_name] for head_name in VALUE_HEAD_NAMES], dim=-1).mean(dim=-1)
