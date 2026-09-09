from __future__ import annotations

import math
from itertools import combinations
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .distributions import MaskedMeanConcentrationDirichlet, squash_action, squashed_logprob
from . import structured_bw_actor_schema as bw_schema
from . import structured_sat_actor_schema as sat_schema
from .structured_types import (
    AccelPolicyOutput,
    BwPolicyOutput,
    LocalAccelState,
    LocalBwState,
    LocalSatState,
    SatSubsetPolicyOutput,
)


NEG_INF = -1e9


@lru_cache(maxsize=32)
def _subset_member_spec_cpu(sat_count: int, max_select: int) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    subset_specs: list[tuple[int, ...]] = [tuple()]
    for k in range(1, min(int(max_select), int(sat_count)) + 1):
        subset_specs.extend(combinations(range(int(sat_count)), k))
    subset_sizes = tuple(len(spec) for spec in subset_specs)
    return tuple(subset_specs), subset_sizes


_SUBSET_MEMBER_TENSOR_CACHE: dict[tuple[int, int, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}


def _subset_member_tensor(sat_count: int, max_select: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (int(sat_count), int(max_select), str(device.type), int(device.index) if device.index is not None else None)
    cached = _SUBSET_MEMBER_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        members, sizes = cached
        if members.device == device and sizes.device == device:
            return members, sizes
    subset_specs, subset_sizes = _subset_member_spec_cpu(int(sat_count), int(max_select))
    subset_members = torch.full(
        (len(subset_specs), int(max_select)),
        -1,
        dtype=torch.long,
        device=device,
    )
    for subset_idx, members in enumerate(subset_specs):
        if members:
            subset_members[subset_idx, : len(members)] = torch.as_tensor(members, dtype=torch.long, device=device)
    subset_sizes_t = torch.as_tensor(subset_sizes, dtype=torch.long, device=device)
    _SUBSET_MEMBER_TENSOR_CACHE[cache_key] = (subset_members, subset_sizes_t)
    return subset_members, subset_sizes_t


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int | None = None,
    *,
    num_layers: int = 2,
    activation_factory: type[nn.Module] = nn.ReLU,
    final_activation: bool | None = None,
) -> nn.Sequential:
    out_dim = hidden_dim if out_dim is None else out_dim
    layer_count = max(int(num_layers), 1)
    modules: list[nn.Module] = []
    if layer_count == 1:
        modules.append(nn.Linear(int(in_dim), int(out_dim)))
    else:
        modules.extend([nn.Linear(int(in_dim), int(hidden_dim)), activation_factory()])
        for _ in range(layer_count - 2):
            modules.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), activation_factory()])
        modules.append(nn.Linear(int(hidden_dim), int(out_dim)))
    use_final_activation = (int(out_dim) == int(hidden_dim)) if final_activation is None else bool(final_activation)
    modules.append(activation_factory() if use_final_activation else nn.Identity())
    return nn.Sequential(*modules)


def _make_output_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_layers: int = 1,
    activation_factory: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    return _make_mlp(
        int(in_dim),
        int(hidden_dim),
        int(out_dim),
        num_layers=max(int(num_layers), 1),
        activation_factory=activation_factory,
        final_activation=False,
    )


def _make_raw_input_norm(input_dim: int, enabled: bool | None) -> nn.Module:
    return nn.LayerNorm(int(input_dim)) if bool(enabled) else nn.Identity()


class AccelInteractionBlock(nn.Module):
    """Masked token self-attention block shared by accel GU/peer/SAT token sets."""

    def __init__(self, *, embed_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        head_count = max(int(num_heads), 1)
        if int(embed_dim) % head_count != 0:
            raise ValueError(
                f"accel_attention_heads={head_count} must divide actor embed_dim={int(embed_dim)}."
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=head_count,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(int(embed_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )
        self.norm_ffn = nn.LayerNorm(int(embed_dim))

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if int(h.shape[-2]) == 0:
            return h
        valid = valid_mask.to(dtype=torch.bool)
        valid_f = valid.unsqueeze(-1).to(dtype=h.dtype)
        h = h * valid_f
        has_valid = valid.any(dim=-1, keepdim=True)
        key_padding_mask = torch.where(has_valid, ~valid, torch.zeros_like(valid))
        attn_out, _weights = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm_attn(h + attn_out) * valid_f
        h = self.norm_ffn(h + self.ffn(h)) * valid_f
        return h


def _safe_categorical_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = logits.masked_fill(~mask, NEG_INF)
    has_valid = mask.any(dim=-1, keepdim=True)
    return torch.where(has_valid, safe, torch.zeros_like(safe))


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.to(scores.dtype)
    safe_scores = torch.where(mask, scores, torch.full_like(scores, NEG_INF))
    max_scores = safe_scores.amax(dim=dim, keepdim=True)
    has_valid = mask.any(dim=dim, keepdim=True)
    max_scores = torch.where(has_valid, max_scores, torch.zeros_like(max_scores))
    shifted = safe_scores - max_scores
    exp_scores = torch.exp(shifted) * mask_f
    norm = exp_scores.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    probs = exp_scores / norm
    return torch.where(has_valid, probs, torch.zeros_like(probs))


def _multi_query_attention(queries: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-2] == 0:
        return queries.new_zeros(queries.shape)
    mask_q = mask.unsqueeze(-2).expand(*queries.shape[:-1], tokens.shape[-2])
    scores = (queries.unsqueeze(-2) * tokens.unsqueeze(-3)).sum(dim=-1) / max(float(tokens.shape[-1]) ** 0.5, 1.0)
    weights = _masked_softmax(scores, mask_q, dim=-1)
    return (weights.unsqueeze(-1) * tokens.unsqueeze(-3)).sum(dim=-2)


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-2] == 0:
        return tokens.new_zeros(tokens.shape[:-2] + (tokens.shape[-1],))
    mask_f = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=-2).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=-2) / denom


def _masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-2] == 0:
        return tokens.new_zeros(tokens.shape[:-2] + (tokens.shape[-1],))
    safe = tokens.masked_fill(~mask.unsqueeze(-1), NEG_INF)
    out = safe.amax(dim=-2)
    return torch.where(mask.any(dim=-1, keepdim=True), out, torch.zeros_like(out))


def _fit_width_last(value: torch.Tensor, width: int, *, fill_value: float = 0.0) -> torch.Tensor:
    target_width = max(int(width), 0)
    current_width = int(value.shape[-1])
    if current_width == target_width:
        return value
    if current_width > target_width:
        return value[..., :target_width]
    pad_shape = value.shape[:-1] + (target_width - current_width,)
    pad = value.new_full(pad_shape, float(fill_value))
    return torch.cat([value, pad], dim=-1)


def _fixed_masked_token_features(tokens: torch.Tensor, mask: torch.Tensor, token_count: int) -> torch.Tensor:
    count = max(int(token_count), 0)
    row_count = int(tokens.shape[0])
    if count == 0:
        return tokens.new_zeros((row_count, 0))
    token_dim = int(tokens.shape[-1])
    mask_bool = mask.to(dtype=torch.bool)
    token_fixed = _fit_width_last(
        (tokens * mask_bool.unsqueeze(-1).to(dtype=tokens.dtype)).transpose(-1, -2),
        count,
    ).transpose(-1, -2)
    mask_fixed = _fit_width_last(mask_bool.to(dtype=tokens.dtype), count)
    return torch.cat([token_fixed.reshape(row_count, count * token_dim), mask_fixed.reshape(row_count, count)], dim=-1)


def _attend(query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Legacy single-query masked dot-product attention helper for diagnostics."""
    if tokens.shape[-2] == 0:
        return query.new_zeros(query.shape)
    query_t = query.unsqueeze(-2) if query.ndim == tokens.ndim - 1 else query
    scores = (query_t * tokens).sum(dim=-1) / max(float(tokens.shape[-1]) ** 0.5, 1.0)
    weights = _masked_softmax(scores, mask.to(dtype=torch.bool), dim=-1)
    context = (weights.unsqueeze(-1) * tokens).sum(dim=-2)
    return context.squeeze(-2) if query.ndim == tokens.ndim - 1 else context


def _gather_member_tokens(tokens: torch.Tensor, members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_members = members.clamp_min(0)
    batch_size, subset_count, member_slots = safe_members.shape
    embed_dim = tokens.shape[-1]
    expanded = tokens.unsqueeze(1).expand(-1, subset_count, -1, -1)
    gather_index = safe_members.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
    gathered = torch.gather(expanded, 2, gather_index)
    member_mask = members >= 0
    return gathered, member_mask


def _masked_member_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=-2).clamp_min(1.0)
    return (values * mask_f).sum(dim=-2) / denom


def _copy_tensor_out(target: torch.Tensor, value: torch.Tensor) -> None:
    target.copy_(value.to(dtype=target.dtype) if value.dtype != target.dtype else value)


def _copy_group_logprob_sum_out(target: torch.Tensor, value: torch.Tensor) -> None:
    grouped = value.reshape(int(target.shape[0]), -1).sum(dim=1)
    target.copy_(grouped.to(dtype=target.dtype) if grouped.dtype != target.dtype else grouped)


def _normal_latent_logprob(mean: torch.Tensor, std: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    latent_t = latent.to(device=mean.device, dtype=mean.dtype)
    return (
        -0.5 * ((latent_t - mean) / std).pow(2)
        - torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)


def _reshape_group_agent_tensor_for_history(
    value: torch.Tensor,
    *,
    num_envs: int,
    num_agents: int,
    name: str,
) -> torch.Tensor:
    expected_envs = int(num_envs)
    expected_agents = int(num_agents)
    if value.ndim >= 2 and tuple(value.shape[:2]) == (expected_envs, expected_agents):
        return value
    if value.ndim >= 1 and int(value.shape[0]) == expected_envs * expected_agents:
        return value.reshape(expected_envs, expected_agents, *value.shape[1:])
    raise ValueError(
        f"History target shape for {name} does not match grouped ({expected_envs}, {expected_agents}, ... ) "
        f"canonical ABI from value shape {tuple(value.shape)}."
    )


def _history_slot_row_indices(
    *,
    history_slot_t: torch.Tensor | None,
    base_row_ids_t: torch.Tensor | None,
) -> torch.Tensor | None:
    if not torch.is_tensor(history_slot_t) or not torch.is_tensor(base_row_ids_t):
        return None
    if base_row_ids_t.ndim != 1:
        raise RuntimeError("history row ids must be 1-D.")
    slot_scalar_t = history_slot_t.reshape(-1)
    if int(slot_scalar_t.numel()) <= 0:
        raise RuntimeError("history slot tensor must contain one slot index.")
    return base_row_ids_t.to(dtype=torch.long) + slot_scalar_t[0].to(
        device=base_row_ids_t.device,
        dtype=torch.long,
    ) * int(base_row_ids_t.shape[0])


def _write_history_slot_tensor_out(
    target: torch.Tensor | None,
    *,
    row_indices_t: torch.Tensor | None,
    value: torch.Tensor,
) -> None:
    if target is None or row_indices_t is None:
        return
    target.index_copy_(0, row_indices_t, value.to(dtype=target.dtype) if value.dtype != target.dtype else value)


def _reshape_group_agent_tensor_for_target(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    num_envs: int,
    num_agents: int,
    name: str,
) -> torch.Tensor:
    if tuple(value.shape) == tuple(target.shape):
        return value
    expected_envs = int(num_envs)
    expected_agents = int(num_agents)
    if (
        value.ndim >= 1
        and target.ndim >= 2
        and int(value.shape[0]) == expected_envs * expected_agents
        and int(target.shape[0]) == expected_envs
        and int(target.shape[1]) == expected_agents
        and tuple(value.shape[1:]) == tuple(target.shape[2:])
    ):
        return value.reshape(expected_envs, expected_agents, *value.shape[1:])
    raise ValueError(
        f"Target shape {tuple(target.shape)} does not match {name} shape {tuple(value.shape)} "
        f"for grouped ({expected_envs}, {expected_agents}, ... ) canonical ABI."
    )


class AccelPolicy(nn.Module):
    def __init__(
        self,
        ego_dim: int,
        cell_dim: int,
        gu_token_dim: int,
        peer_token_dim: int,
        sat_token_dim: int,
        *,
        gu_query_count: int = 4,
        peer_query_count: int = 2,
        sat_query_count: int = 2,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        encoder_mlp_layers: int = 2,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 1,
        interaction_layers: int = 0,
        attention_heads: int = 4,
        action_scale: float = 1.0,
        input_norm_enabled: bool | None = False,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.gu_query_count = int(gu_query_count)
        self.peer_query_count = int(peer_query_count)
        self.sat_query_count = int(sat_query_count)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.encoder_mlp_layers = max(int(encoder_mlp_layers), 1)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        self.interaction_layers = max(int(interaction_layers), 0)
        self.attention_heads = max(int(attention_heads), 1)

        self.ego_norm = _make_raw_input_norm(ego_dim, input_norm_enabled)
        self.cell_norm = _make_raw_input_norm(cell_dim, input_norm_enabled)
        self.gu_norm = _make_raw_input_norm(gu_token_dim, input_norm_enabled)
        self.peer_norm = _make_raw_input_norm(peer_token_dim, input_norm_enabled)
        self.sat_norm = _make_raw_input_norm(sat_token_dim, input_norm_enabled)

        self.ego_encoder = _make_mlp(ego_dim, hidden_dim, embed_dim, num_layers=self.encoder_mlp_layers)
        self.cell_encoder = _make_mlp(cell_dim, hidden_dim, embed_dim, num_layers=self.encoder_mlp_layers)
        self.gu_encoder = _make_mlp(gu_token_dim, hidden_dim, embed_dim, num_layers=self.encoder_mlp_layers)
        self.peer_encoder = _make_mlp(peer_token_dim, hidden_dim, embed_dim, num_layers=self.encoder_mlp_layers)
        self.sat_encoder = _make_mlp(sat_token_dim, hidden_dim, embed_dim, num_layers=self.encoder_mlp_layers)
        self.interaction_blocks = nn.ModuleList(
            [
                AccelInteractionBlock(
                    embed_dim=self.embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=self.attention_heads,
                )
                for _ in range(self.interaction_layers)
            ]
        )

        query_dim = 2 * embed_dim
        self.gu_query = nn.Linear(query_dim, self.gu_query_count * embed_dim)
        self.peer_query = nn.Linear(query_dim, self.peer_query_count * embed_dim)
        self.sat_query = nn.Linear(query_dim, self.sat_query_count * embed_dim)

        fusion_dim = (
            2 * embed_dim
            + (self.gu_query_count + 2) * embed_dim
            + (self.peer_query_count + 2) * embed_dim
            + (self.sat_query_count + 2) * embed_dim
        )
        self.fusion = _make_mlp(fusion_dim, hidden_dim, hidden_dim, num_layers=self.context_mlp_layers)
        if self.head_mlp_layers <= 1:
            self.mu_head = nn.Linear(hidden_dim, 2)
        else:
            self.mu_head = _make_output_mlp(hidden_dim, hidden_dim, 2, num_layers=self.head_mlp_layers)
        self.log_std = nn.Parameter(torch.zeros(2))

    def _encode_tokens(
        self,
        token: torch.Tensor,
        norm: nn.Module,
        encoder: nn.Module,
    ) -> torch.Tensor:
        if token.shape[-2] == 0:
            return token.new_zeros(token.shape[:-1] + (self.embed_dim,))
        return encoder(norm(token))

    def _context(self, local_state: LocalAccelState) -> torch.Tensor:
        ego_emb = self.ego_encoder(self.ego_norm(local_state.ego_features))
        cell_emb = self.cell_encoder(self.cell_norm(local_state.ego_cell))
        gu_emb = self._encode_tokens(local_state.gu_tokens, self.gu_norm, self.gu_encoder)
        peer_emb = self._encode_tokens(local_state.peer_tokens, self.peer_norm, self.peer_encoder)
        sat_emb = self._encode_tokens(local_state.sat_tokens, self.sat_norm, self.sat_encoder)
        for block in self.interaction_blocks:
            gu_emb = block(gu_emb, local_state.gu_mask)
            peer_emb = block(peer_emb, local_state.peer_mask)
            sat_emb = block(sat_emb, local_state.sat_mask)

        query_src = torch.cat([ego_emb, cell_emb], dim=-1)
        batch = int(query_src.shape[0])
        gu_queries = self.gu_query(query_src).view(batch, self.gu_query_count, self.embed_dim)
        peer_queries = self.peer_query(query_src).view(batch, self.peer_query_count, self.embed_dim)
        sat_queries = self.sat_query(query_src).view(batch, self.sat_query_count, self.embed_dim)

        gu_attn = _multi_query_attention(gu_queries, gu_emb, local_state.gu_mask)
        peer_attn = _multi_query_attention(peer_queries, peer_emb, local_state.peer_mask)
        sat_attn = _multi_query_attention(sat_queries, sat_emb, local_state.sat_mask)

        z = torch.cat(
            [
                ego_emb,
                cell_emb,
                gu_attn.reshape(batch, self.gu_query_count * self.embed_dim),
                _masked_mean(gu_emb, local_state.gu_mask),
                _masked_max(gu_emb, local_state.gu_mask),
                peer_attn.reshape(batch, self.peer_query_count * self.embed_dim),
                _masked_mean(peer_emb, local_state.peer_mask),
                _masked_max(peer_emb, local_state.peer_mask),
                sat_attn.reshape(batch, self.sat_query_count * self.embed_dim),
                _masked_mean(sat_emb, local_state.sat_mask),
                _masked_max(sat_emb, local_state.sat_mask),
            ],
            dim=-1,
        )
        return self.fusion(z)

    def forward(self, local_state: LocalAccelState, deterministic: bool = False) -> AccelPolicyOutput:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        latent = mean if deterministic else dist.rsample()
        action = squash_action(latent, self.action_scale)
        logprob = _normal_latent_logprob(mean, std, latent)
        entropy = dist.entropy().sum(dim=-1)
        return AccelPolicyOutput(action=action, logprob=logprob, entropy=entropy, mean=mean, std=std, latent_action=latent)

    def act_into(
        self,
        local_state: LocalAccelState,
        *,
        action_out: torch.Tensor | None = None,
        latent_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_latent_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().reshape((1,) * max(mean.ndim - 1, 0) + (2,)).expand_as(mean)
        dist = Normal(mean, std)
        latent = mean if deterministic else dist.rsample()
        action = squash_action(latent, self.action_scale)
        latent_canonical = _reshape_group_agent_tensor_for_history(
            latent,
            num_envs=num_envs,
            num_agents=num_agents,
            name="accel latent action",
        )
        if action_out is not None:
            action_canonical = _reshape_group_agent_tensor_for_target(
                action,
                action_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="accel action",
            )
            _copy_tensor_out(action_out, action_canonical)
        else:
            action_canonical = _reshape_group_agent_tensor_for_history(
                action,
                num_envs=num_envs,
                num_agents=num_agents,
                name="accel action",
            )
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        _write_history_slot_tensor_out(
            history_action_out,
            row_indices_t=history_row_indices_t,
            value=action_canonical,
        )
        if latent_action_out is not None:
            _copy_tensor_out(latent_action_out, latent_canonical)
        _write_history_slot_tensor_out(
            history_latent_action_out,
            row_indices_t=history_row_indices_t,
            value=latent_canonical,
        )
        if logprob_out is not None:
            if tuple(logprob_out.shape) != (int(num_envs),):
                raise ValueError(f"Target shape {tuple(logprob_out.shape)} does not match ({int(action.shape[0])},).")
            logprob = _normal_latent_logprob(mean, std, latent)
            _copy_group_logprob_sum_out(logprob_out, logprob)
            grouped_logprob = logprob.reshape(int(num_envs), -1).sum(dim=1)
            _write_history_slot_tensor_out(
                history_logprob_out,
                row_indices_t=history_row_indices_t,
                value=grouped_logprob,
            )

    def evaluate_actions(
        self,
        local_state: LocalAccelState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
        latent_action: torch.Tensor | None = None,
    ) -> AccelPolicyOutput:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        logprob = (
            _normal_latent_logprob(mean, std, latent_action)
            if latent_action is not None
            else squashed_logprob(dist, action, self.action_scale)
        )
        entropy = dist.entropy().sum(dim=-1) if bool(compute_entropy) else torch.zeros_like(logprob)
        return AccelPolicyOutput(action=action, logprob=logprob, entropy=entropy, mean=mean, std=std, latent_action=latent_action)


class FlatAccelPolicy(nn.Module):
    """Flat MLP accel head for MAPPO-like baselines.

    This policy keeps the same action distribution and safety-compatible output
    contract as AccelPolicy, but removes token encoders, attention, and
    hand-designed set aggregation.
    """

    def __init__(
        self,
        ego_dim: int,
        cell_dim: int,
        gu_token_dim: int,
        peer_token_dim: int,
        sat_token_dim: int,
        *,
        gu_token_count: int,
        peer_token_count: int,
        sat_token_count: int,
        hidden_dim: int = 256,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 1,
        action_scale: float = 1.0,
        input_norm_enabled: bool | None = False,
    ) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        self.gu_token_count = max(int(gu_token_count), 0)
        self.peer_token_count = max(int(peer_token_count), 0)
        self.sat_token_count = max(int(sat_token_count), 0)
        self.hidden_dim = int(hidden_dim)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        input_dim = (
            int(ego_dim)
            + int(cell_dim)
            + self.gu_token_count * (int(gu_token_dim) + 1)
            + self.peer_token_count * (int(peer_token_dim) + 1)
            + self.sat_token_count * (int(sat_token_dim) + 1)
        )
        self.input_norm = _make_raw_input_norm(input_dim, input_norm_enabled)
        self.trunk = _make_mlp(input_dim, self.hidden_dim, self.hidden_dim, num_layers=self.context_mlp_layers)
        if self.head_mlp_layers <= 1:
            self.mu_head = nn.Linear(self.hidden_dim, 2)
        else:
            self.mu_head = _make_output_mlp(self.hidden_dim, self.hidden_dim, 2, num_layers=self.head_mlp_layers)
        self.log_std = nn.Parameter(torch.zeros(2))

    def _features(self, local_state: LocalAccelState) -> torch.Tensor:
        return torch.cat(
            [
                local_state.ego_features,
                local_state.ego_cell,
                _fixed_masked_token_features(local_state.gu_tokens, local_state.gu_mask, self.gu_token_count),
                _fixed_masked_token_features(local_state.peer_tokens, local_state.peer_mask, self.peer_token_count),
                _fixed_masked_token_features(local_state.sat_tokens, local_state.sat_mask, self.sat_token_count),
            ],
            dim=-1,
        )

    def _context(self, local_state: LocalAccelState) -> torch.Tensor:
        return self.trunk(self.input_norm(self._features(local_state)))

    def forward(self, local_state: LocalAccelState, deterministic: bool = False) -> AccelPolicyOutput:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        latent = mean if deterministic else dist.rsample()
        action = squash_action(latent, self.action_scale)
        logprob = _normal_latent_logprob(mean, std, latent)
        entropy = dist.entropy().sum(dim=-1)
        return AccelPolicyOutput(action=action, logprob=logprob, entropy=entropy, mean=mean, std=std, latent_action=latent)

    def act_into(
        self,
        local_state: LocalAccelState,
        *,
        action_out: torch.Tensor | None = None,
        latent_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_latent_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().reshape((1,) * max(mean.ndim - 1, 0) + (2,)).expand_as(mean)
        dist = Normal(mean, std)
        latent = mean if deterministic else dist.rsample()
        action = squash_action(latent, self.action_scale)
        latent_canonical = _reshape_group_agent_tensor_for_history(
            latent,
            num_envs=num_envs,
            num_agents=num_agents,
            name="accel latent action",
        )
        if action_out is not None:
            action_canonical = _reshape_group_agent_tensor_for_target(
                action,
                action_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="accel action",
            )
            _copy_tensor_out(action_out, action_canonical)
        else:
            action_canonical = _reshape_group_agent_tensor_for_history(
                action,
                num_envs=num_envs,
                num_agents=num_agents,
                name="accel action",
            )
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        _write_history_slot_tensor_out(history_action_out, row_indices_t=history_row_indices_t, value=action_canonical)
        if latent_action_out is not None:
            _copy_tensor_out(latent_action_out, latent_canonical)
        _write_history_slot_tensor_out(
            history_latent_action_out,
            row_indices_t=history_row_indices_t,
            value=latent_canonical,
        )
        if logprob_out is not None:
            if tuple(logprob_out.shape) != (int(num_envs),):
                raise ValueError(f"Target shape {tuple(logprob_out.shape)} does not match ({int(action.shape[0])},).")
            logprob = _normal_latent_logprob(mean, std, latent)
            _copy_group_logprob_sum_out(logprob_out, logprob)
            grouped_logprob = logprob.reshape(int(num_envs), -1).sum(dim=1)
            _write_history_slot_tensor_out(
                history_logprob_out,
                row_indices_t=history_row_indices_t,
                value=grouped_logprob,
            )

    def evaluate_actions(
        self,
        local_state: LocalAccelState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
        latent_action: torch.Tensor | None = None,
    ) -> AccelPolicyOutput:
        ctx = self._context(local_state)
        mean = self.mu_head(ctx)
        std = torch.clamp(self.log_std, -5.0, 2.0).exp().unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        logprob = (
            _normal_latent_logprob(mean, std, latent_action)
            if latent_action is not None
            else squashed_logprob(dist, action, self.action_scale)
        )
        entropy = dist.entropy().sum(dim=-1) if bool(compute_entropy) else torch.zeros_like(logprob)
        return AccelPolicyOutput(action=action, logprob=logprob, entropy=entropy, mean=mean, std=std, latent_action=latent_action)


def _make_sat_mlp2(in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2) -> nn.Sequential:
    return _make_mlp(
        int(in_dim),
        int(hidden_dim),
        int(out_dim),
        num_layers=max(int(num_layers), 1),
        activation_factory=nn.ReLU,
        final_activation=False,
    )


class SatSelfAttentionBlock(nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = max(int(num_heads), 1)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"SAT attention heads={self.num_heads} must divide embed_dim={self.embed_dim}.")
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = _make_sat_mlp2(self.embed_dim, int(hidden_dim), self.embed_dim)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if int(h.shape[-2]) == 0:
            return h
        valid = valid_mask.to(dtype=torch.bool)
        valid_f = valid.unsqueeze(-1).to(dtype=h.dtype)
        h = h * valid_f
        x = self.attn_norm(h)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        batch_size, token_count, _ = q.shape
        q = q.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, NEG_INF)
        query_has_valid = valid.any(dim=-1)
        safe_scores = torch.where(
            query_has_valid[:, None, None, None],
            scores,
            torch.zeros_like(scores),
        )
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * key_mask.to(dtype=weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        weights = weights / denom
        attn = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, token_count, self.embed_dim)
        attn = self.out_proj(attn) * valid_f
        h = (h + attn) * valid_f
        h = (h + self.ffn(self.ffn_norm(h))) * valid_f
        return h


class SatSubsetPolicy(nn.Module):
    def __init__(
        self,
        *,
        ego_dim: int = sat_schema.SAT_EGO_DIM,
        demand_dim: int = sat_schema.SAT_DEMAND_DIM,
        role_dim: int = sat_schema.SAT_ROLE_DIM,
        sat_token_dim: int = sat_schema.SAT_TOKEN_DIM,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        encoder_mlp_layers: int = 2,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 2,
        sat_competition_layers: int = 2,
        sat_attention_heads: int = 4,
        sat_action_select_k: int = 1,
        per_uav_visible_sat_token_max: int = 1,
        input_norm_enabled: bool | None = False,
    ) -> None:
        super().__init__()
        self.ego_dim = int(ego_dim)
        self.demand_dim = int(demand_dim)
        self.role_dim = int(role_dim)
        self.sat_token_dim = int(sat_token_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.encoder_mlp_layers = max(int(encoder_mlp_layers), 1)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        self.sat_action_select_k = int(sat_action_select_k)
        self.per_uav_visible_sat_token_max = int(per_uav_visible_sat_token_max)
        self.sat_attention_heads = int(sat_attention_heads)
        self.sat_competition_layers = max(int(sat_competition_layers), 1)
        if self.sat_action_select_k <= 0:
            raise ValueError("sat_action_select_k must be positive.")
        if self.per_uav_visible_sat_token_max <= 0:
            raise ValueError("per_uav_visible_sat_token_max must be positive.")
        if self.ego_dim != sat_schema.SAT_EGO_DIM:
            raise ValueError(f"SAT ego_dim must be {sat_schema.SAT_EGO_DIM}, got {self.ego_dim}.")
        if self.demand_dim != sat_schema.SAT_DEMAND_DIM:
            raise ValueError(f"SAT demand_dim must be {sat_schema.SAT_DEMAND_DIM}, got {self.demand_dim}.")
        if self.role_dim != sat_schema.SAT_ROLE_DIM:
            raise ValueError(f"SAT role_dim must be {sat_schema.SAT_ROLE_DIM}, got {self.role_dim}.")
        if self.sat_token_dim != sat_schema.SAT_TOKEN_DIM:
            raise ValueError(f"SAT sat_token_dim must be {sat_schema.SAT_TOKEN_DIM}, got {self.sat_token_dim}.")

        self.ego_input_norm = _make_raw_input_norm(self.ego_dim, input_norm_enabled)
        self.demand_input_norm = _make_raw_input_norm(self.demand_dim, input_norm_enabled)
        self.sat_input_norm = _make_raw_input_norm(self.sat_token_dim, input_norm_enabled)
        self.ego_encoder = _make_sat_mlp2(self.ego_dim, self.hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.demand_encoder = _make_sat_mlp2(self.demand_dim, self.hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.role_encoder = _make_sat_mlp2(self.role_dim, self.hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.sat_encoder = _make_sat_mlp2(self.sat_token_dim, self.hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.ctx_encoder = _make_sat_mlp2(3 * self.embed_dim, self.hidden_dim, self.embed_dim, num_layers=self.context_mlp_layers)
        self.sat_context_fusion = _make_sat_mlp2(2 * self.embed_dim, self.hidden_dim, self.embed_dim, num_layers=self.context_mlp_layers)
        self.sat_self_attention_blocks = nn.ModuleList(
            [
                SatSelfAttentionBlock(
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=int(sat_attention_heads),
                )
                for _ in range(self.sat_competition_layers)
            ]
        )
        self.sat_logit_head = _make_sat_mlp2(self.embed_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)
        self.count_logit_head = _make_sat_mlp2(
            self.embed_dim,
            self.hidden_dim,
            self.sat_action_select_k + 1,
            num_layers=self.head_mlp_layers,
        )

    @staticmethod
    def _canonical_members(state: LocalSatState) -> torch.Tensor:
        members = state.subset_members.to(dtype=torch.long)
        if members.ndim == 3:
            return members[0]
        if members.ndim != 2:
            raise ValueError(f"subset_members must have shape [M,K] or [R,M,K], got {tuple(members.shape)}.")
        return members

    def _compute_logits(self, state: LocalSatState) -> torch.Tensor:
        valid_sat_mask = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
        ego_emb = self.ego_encoder(self.ego_input_norm(state.ego_features))
        demand_emb = self.demand_encoder(self.demand_input_norm(state.demand_features))
        role_emb = self.role_encoder(state.role_features)
        sat_emb = self.sat_encoder(self.sat_input_norm(state.sat_tokens))
        ctx0 = self.ctx_encoder(torch.cat([ego_emb, demand_emb, role_emb], dim=-1))
        sat_h = self.sat_context_fusion(
            torch.cat([sat_emb, ctx0.unsqueeze(1).expand(-1, sat_emb.shape[1], -1)], dim=-1)
        )
        sat_h = sat_h * valid_sat_mask.unsqueeze(-1).to(dtype=sat_h.dtype)
        for block in self.sat_self_attention_blocks:
            sat_h = block(sat_h, valid_sat_mask)
        item_logits = self.sat_logit_head(sat_h).squeeze(-1)
        item_logits = item_logits.masked_fill(~valid_sat_mask, 0.0)
        count_logits = self.count_logit_head(ctx0)

        members = self._canonical_members(state).to(device=item_logits.device)
        member_mask = members >= 0
        safe_members = members.clamp_min(0)
        row_count = int(item_logits.shape[0])
        flat = safe_members.reshape(-1)
        gathered = item_logits.index_select(1, flat).reshape(row_count, int(members.shape[0]), int(members.shape[1]))
        item_sum = (gathered * member_mask.unsqueeze(0).to(dtype=item_logits.dtype)).sum(dim=-1)
        sizes = member_mask.sum(dim=-1).clamp(max=self.sat_action_select_k).to(dtype=torch.long)
        count_part = count_logits.index_select(1, sizes)
        logits = item_sum + count_part
        return logits

    def _legal_subset_mask(self, state: LocalSatState, logits: torch.Tensor) -> torch.Tensor:
        valid_sat_mask = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
        members = self._canonical_members(state).to(device=logits.device)
        member_mask = members >= 0
        safe_members = members.clamp_min(0)
        valid_members = valid_sat_mask.index_select(1, safe_members.reshape(-1)).reshape(
            int(logits.shape[0]),
            int(members.shape[0]),
            int(members.shape[1]),
        )
        member_ok = valid_members | ~member_mask.unsqueeze(0)
        mask = member_ok.all(dim=-1)
        sizes = member_mask.sum(dim=-1).view(1, -1)
        valid_count = valid_sat_mask.sum(dim=-1, keepdim=True)
        mask = mask & (sizes <= valid_count.clamp(max=self.sat_action_select_k))
        mask = torch.where(valid_count > 0, mask & (sizes > 0), mask & (sizes == 0))
        return mask

    def _select(
        self,
        state: LocalSatState,
        subset_index: torch.Tensor | None,
        deterministic: bool,
        *,
        compute_entropy: bool = True,
    ) -> SatSubsetPolicyOutput:
        logits = self._compute_logits(state)
        subset_mask = self._legal_subset_mask(state, logits)
        if subset_mask.shape != logits.shape:
            raise ValueError(f"subset_mask shape {tuple(subset_mask.shape)} does not match logits {tuple(logits.shape)}.")
        safe_logits = _safe_categorical_logits(logits, subset_mask)
        legal_count = subset_mask.sum(dim=-1)
        dist = Categorical(logits=safe_logits)
        if subset_index is None:
            chosen = safe_logits.argmax(dim=-1) if deterministic else dist.sample()
        else:
            chosen = subset_index.to(device=logits.device, dtype=torch.long).reshape(-1)
            if int(chosen.shape[0]) != int(logits.shape[0]):
                raise ValueError(f"subset_index length {int(chosen.shape[0])} does not match row count {int(logits.shape[0])}.")
            row_ids = torch.arange(int(chosen.shape[0]), device=logits.device, dtype=torch.long)
            chosen_in_range = (chosen >= 0) & (chosen < int(logits.shape[1]))
            chosen_safe = chosen.clamp(min=0, max=max(int(logits.shape[1]) - 1, 0))
            no_legal_fallback = (legal_count <= 0) & (chosen_safe == 0)
            chosen_legal = chosen_in_range & (subset_mask[row_ids, chosen_safe] | no_legal_fallback)
            if not bool(chosen_legal.all().item()):
                bad = torch.nonzero(~chosen_legal, as_tuple=False).flatten()[0]
                bad_i = int(bad.item())
                valid_mask_dbg = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
                legal_idx_dbg = torch.nonzero(subset_mask[bad_i], as_tuple=False).flatten()[:8].detach().cpu().tolist()
                candidate_dbg = (
                    state.candidate_sat_ids[bad_i].detach().cpu().tolist()
                    if torch.is_tensor(getattr(state, "candidate_sat_ids", None))
                    else []
                )
                raise ValueError(
                    f"subset_index contains an illegal SAT subset at row {bad_i}: "
                    f"{int(chosen[bad].item())}; "
                    f"legal_count={int(legal_count[bad_i].item())}, "
                    f"valid_count={int(valid_mask_dbg[bad_i].sum().item())}, "
                    f"legal_head={legal_idx_dbg}, candidates={candidate_dbg}."
                )
        chosen = torch.where(legal_count > 0, chosen.clamp_min(0), torch.zeros_like(chosen))
        members_base = self._canonical_members(state).to(device=logits.device)
        selected_members = members_base.index_select(0, chosen.clamp(max=members_base.shape[0] - 1))
        selected_members = torch.where(
            (legal_count > 0).unsqueeze(-1),
            selected_members,
            torch.full_like(selected_members, -1),
        )
        candidate_ids = state.candidate_sat_ids.to(device=logits.device, dtype=torch.long)
        safe_selected = selected_members.clamp_min(0)
        selected_global = torch.gather(candidate_ids, 1, safe_selected.clamp(max=max(int(candidate_ids.shape[1]) - 1, 0)))
        selected_global = torch.where(selected_members >= 0, selected_global, torch.full_like(selected_global, -1))
        logprob_raw = dist.log_prob(chosen)
        entropy_raw = dist.entropy() if bool(compute_entropy) else torch.zeros_like(logprob_raw)
        multi_legal = legal_count > 1
        logprob = torch.where(multi_legal, logprob_raw, torch.zeros_like(logprob_raw))
        entropy = torch.where(multi_legal, entropy_raw, torch.zeros_like(entropy_raw))
        subset_out = torch.where(legal_count > 0, chosen, torch.zeros_like(chosen))
        return SatSubsetPolicyOutput(
            selected_sat_indices=selected_global,
            subset_index=subset_out,
            subset_members=selected_members,
            logprob=logprob,
            entropy=entropy,
            logits=logits,
        )

    def forward(self, local_state: LocalSatState, deterministic: bool = False) -> SatSubsetPolicyOutput:
        return self._select(local_state, subset_index=None, deterministic=bool(deterministic))

    def act_into(
        self,
        local_state: LocalSatState,
        *,
        subset_index_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        out = self.forward(local_state, deterministic=deterministic)
        subset_index = out.subset_index
        if subset_index_out is not None:
            subset_index_canonical = _reshape_group_agent_tensor_for_target(
                subset_index,
                subset_index_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat subset index",
            )
            _copy_tensor_out(subset_index_out, subset_index_canonical)
        else:
            subset_index_canonical = _reshape_group_agent_tensor_for_history(
                subset_index,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat subset index",
            )
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        _write_history_slot_tensor_out(
            history_action_out,
            row_indices_t=history_row_indices_t,
            value=subset_index_canonical,
        )
        logprob = out.logprob
        logprob_grouped = logprob.reshape(int(num_envs), -1)
        if logprob_per_agent_out is not None:
            target = _reshape_group_agent_tensor_for_target(
                logprob,
                logprob_per_agent_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat logprob per agent",
            )
            _copy_tensor_out(logprob_per_agent_out, target)
        _write_history_slot_tensor_out(
            history_logprob_per_agent_out,
            row_indices_t=history_row_indices_t,
            value=logprob_grouped,
        )
        if logprob_out is not None:
            if tuple(logprob_out.shape) != (int(num_envs),):
                raise ValueError(f"Target shape {tuple(logprob_out.shape)} does not match ({int(num_envs)},).")
            _copy_group_logprob_sum_out(logprob_out, logprob)
            _write_history_slot_tensor_out(
                history_logprob_out,
                row_indices_t=history_row_indices_t,
                value=logprob_grouped.sum(dim=1),
            )

    def evaluate_actions(
        self,
        local_state: LocalSatState,
        subset_index: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> SatSubsetPolicyOutput:
        return self._select(
            local_state,
            subset_index=subset_index,
            deterministic=True,
            compute_entropy=bool(compute_entropy),
        )

    def topk_legal_subset_indices(
        self,
        local_state: LocalSatState,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = max(int(k), 0)
        logits = self._compute_logits(local_state)
        batch_size = int(logits.shape[0])
        if topk <= 0:
            empty_idx = torch.full((batch_size, 0), -1, dtype=torch.long, device=logits.device)
            empty_logits = torch.empty((batch_size, 0), dtype=logits.dtype, device=logits.device)
            return empty_idx, empty_logits
        subset_mask = self._legal_subset_mask(local_state, logits)
        safe_logits = logits.masked_fill(~subset_mask, float("-inf"))
        topk = min(topk, int(logits.shape[-1]))
        topk_values, topk_indices = torch.topk(safe_logits, k=topk, dim=-1)
        valid_counts = subset_mask.sum(dim=-1, keepdim=True)
        rank = torch.arange(topk, device=logits.device, dtype=torch.long).view(1, -1)
        keep_mask = rank < valid_counts
        topk_indices = torch.where(
            keep_mask,
            topk_indices,
            torch.full_like(topk_indices, -1),
        )
        topk_values = torch.where(
            keep_mask,
            topk_values,
            torch.full_like(topk_values, float("-inf")),
        )
        return topk_indices, topk_values


class FlatSatSubsetPolicy(nn.Module):
    """Flat MLP SAT subset head for MAPPO-like baselines."""

    def __init__(
        self,
        *,
        ego_dim: int = sat_schema.SAT_EGO_DIM,
        demand_dim: int = sat_schema.SAT_DEMAND_DIM,
        role_dim: int = sat_schema.SAT_ROLE_DIM,
        sat_token_dim: int = sat_schema.SAT_TOKEN_DIM,
        hidden_dim: int = 128,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 2,
        sat_action_select_k: int = 1,
        per_uav_visible_sat_token_max: int = 1,
        input_norm_enabled: bool | None = False,
    ) -> None:
        super().__init__()
        self.ego_dim = int(ego_dim)
        self.demand_dim = int(demand_dim)
        self.role_dim = int(role_dim)
        self.sat_token_dim = int(sat_token_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        self.sat_action_select_k = int(sat_action_select_k)
        self.per_uav_visible_sat_token_max = int(per_uav_visible_sat_token_max)
        if self.sat_action_select_k <= 0:
            raise ValueError("sat_action_select_k must be positive.")
        if self.per_uav_visible_sat_token_max <= 0:
            raise ValueError("per_uav_visible_sat_token_max must be positive.")
        subset_specs, _subset_sizes = _subset_member_spec_cpu(self.per_uav_visible_sat_token_max, self.sat_action_select_k)
        self.subset_count = int(len(subset_specs))
        input_dim = (
            self.ego_dim
            + self.demand_dim
            + self.role_dim
            + self.per_uav_visible_sat_token_max * (self.sat_token_dim + 2)
        )
        self.input_norm = _make_raw_input_norm(input_dim, input_norm_enabled)
        self.trunk = _make_sat_mlp2(input_dim, self.hidden_dim, self.hidden_dim, num_layers=self.context_mlp_layers)
        self.logit_head = _make_sat_mlp2(self.hidden_dim, self.hidden_dim, self.subset_count, num_layers=self.head_mlp_layers)

    @staticmethod
    def _canonical_members(state: LocalSatState) -> torch.Tensor:
        return SatSubsetPolicy._canonical_members(state)

    def _features(self, state: LocalSatState) -> torch.Tensor:
        valid_mask = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
        sat_token_features = _fixed_masked_token_features(
            state.sat_tokens,
            state.sat_mask,
            self.per_uav_visible_sat_token_max,
        )
        sat_valid_features = _fit_width_last(valid_mask.to(dtype=state.sat_tokens.dtype), self.per_uav_visible_sat_token_max)
        return torch.cat(
            [
                state.ego_features,
                state.demand_features,
                state.role_features,
                sat_token_features,
                sat_valid_features,
            ],
            dim=-1,
        )

    def _compute_logits(self, state: LocalSatState) -> torch.Tensor:
        logits = self.logit_head(self.trunk(self.input_norm(self._features(state))))
        members = self._canonical_members(state)
        return _fit_width_last(logits, int(members.shape[0]))

    def _legal_subset_mask(self, state: LocalSatState, logits: torch.Tensor) -> torch.Tensor:
        valid_sat_mask = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
        members = self._canonical_members(state).to(device=logits.device)
        member_mask = members >= 0
        safe_members = members.clamp_min(0)
        valid_members = valid_sat_mask.index_select(1, safe_members.reshape(-1)).reshape(
            int(logits.shape[0]),
            int(members.shape[0]),
            int(members.shape[1]),
        )
        member_ok = valid_members | ~member_mask.unsqueeze(0)
        mask = member_ok.all(dim=-1)
        sizes = member_mask.sum(dim=-1).view(1, -1)
        valid_count = valid_sat_mask.sum(dim=-1, keepdim=True)
        mask = mask & (sizes <= valid_count.clamp(max=self.sat_action_select_k))
        mask = torch.where(valid_count > 0, mask & (sizes > 0), mask & (sizes == 0))
        return mask

    def _select(
        self,
        state: LocalSatState,
        subset_index: torch.Tensor | None,
        deterministic: bool,
        *,
        compute_entropy: bool = True,
    ) -> SatSubsetPolicyOutput:
        logits = self._compute_logits(state)
        subset_mask = self._legal_subset_mask(state, logits)
        if subset_mask.shape != logits.shape:
            raise ValueError(f"subset_mask shape {tuple(subset_mask.shape)} does not match logits {tuple(logits.shape)}.")
        safe_logits = _safe_categorical_logits(logits, subset_mask)
        legal_count = subset_mask.sum(dim=-1)
        dist = Categorical(logits=safe_logits)
        if subset_index is None:
            chosen = safe_logits.argmax(dim=-1) if deterministic else dist.sample()
        else:
            chosen = subset_index.to(device=logits.device, dtype=torch.long).reshape(-1)
            if int(chosen.shape[0]) != int(logits.shape[0]):
                raise ValueError(f"subset_index length {int(chosen.shape[0])} does not match row count {int(logits.shape[0])}.")
            row_ids = torch.arange(int(chosen.shape[0]), device=logits.device, dtype=torch.long)
            chosen_in_range = (chosen >= 0) & (chosen < int(logits.shape[1]))
            chosen_safe = chosen.clamp(min=0, max=max(int(logits.shape[1]) - 1, 0))
            no_legal_fallback = (legal_count <= 0) & (chosen_safe == 0)
            chosen_legal = chosen_in_range & (subset_mask[row_ids, chosen_safe] | no_legal_fallback)
            if not bool(chosen_legal.all().item()):
                bad = torch.nonzero(~chosen_legal, as_tuple=False).flatten()[0]
                bad_i = int(bad.item())
                valid_mask_dbg = state.sat_mask.to(dtype=torch.bool) & state.sat_valid_mask.to(dtype=torch.bool)
                legal_idx_dbg = torch.nonzero(subset_mask[bad_i], as_tuple=False).flatten()[:8].detach().cpu().tolist()
                candidate_dbg = (
                    state.candidate_sat_ids[bad_i].detach().cpu().tolist()
                    if torch.is_tensor(getattr(state, "candidate_sat_ids", None))
                    else []
                )
                raise ValueError(
                    f"subset_index contains an illegal SAT subset at row {bad_i}: "
                    f"{int(chosen[bad].item())}; "
                    f"legal_count={int(legal_count[bad_i].item())}, "
                    f"valid_count={int(valid_mask_dbg[bad_i].sum().item())}, "
                    f"legal_head={legal_idx_dbg}, candidates={candidate_dbg}."
                )
        chosen = torch.where(legal_count > 0, chosen.clamp_min(0), torch.zeros_like(chosen))
        members_base = self._canonical_members(state).to(device=logits.device)
        selected_members = members_base.index_select(0, chosen.clamp(max=members_base.shape[0] - 1))
        selected_members = torch.where(
            (legal_count > 0).unsqueeze(-1),
            selected_members,
            torch.full_like(selected_members, -1),
        )
        candidate_ids = state.candidate_sat_ids.to(device=logits.device, dtype=torch.long)
        safe_selected = selected_members.clamp_min(0)
        selected_global = torch.gather(candidate_ids, 1, safe_selected.clamp(max=max(int(candidate_ids.shape[1]) - 1, 0)))
        selected_global = torch.where(selected_members >= 0, selected_global, torch.full_like(selected_global, -1))
        logprob_raw = dist.log_prob(chosen)
        entropy_raw = dist.entropy() if bool(compute_entropy) else torch.zeros_like(logprob_raw)
        multi_legal = legal_count > 1
        logprob = torch.where(multi_legal, logprob_raw, torch.zeros_like(logprob_raw))
        entropy = torch.where(multi_legal, entropy_raw, torch.zeros_like(entropy_raw))
        subset_out = torch.where(legal_count > 0, chosen, torch.zeros_like(chosen))
        return SatSubsetPolicyOutput(
            selected_sat_indices=selected_global,
            subset_index=subset_out,
            subset_members=selected_members,
            logprob=logprob,
            entropy=entropy,
            logits=logits,
        )

    def forward(self, local_state: LocalSatState, deterministic: bool = False) -> SatSubsetPolicyOutput:
        return self._select(local_state, subset_index=None, deterministic=bool(deterministic))

    def act_into(
        self,
        local_state: LocalSatState,
        *,
        subset_index_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        out = self.forward(local_state, deterministic=deterministic)
        subset_index = out.subset_index
        if subset_index_out is not None:
            subset_index_canonical = _reshape_group_agent_tensor_for_target(
                subset_index,
                subset_index_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat subset index",
            )
            _copy_tensor_out(subset_index_out, subset_index_canonical)
        else:
            subset_index_canonical = _reshape_group_agent_tensor_for_history(
                subset_index,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat subset index",
            )
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        _write_history_slot_tensor_out(history_action_out, row_indices_t=history_row_indices_t, value=subset_index_canonical)
        logprob = out.logprob
        logprob_grouped = logprob.reshape(int(num_envs), -1)
        if logprob_per_agent_out is not None:
            target = _reshape_group_agent_tensor_for_target(
                logprob,
                logprob_per_agent_out,
                num_envs=num_envs,
                num_agents=num_agents,
                name="sat logprob per agent",
            )
            _copy_tensor_out(logprob_per_agent_out, target)
        _write_history_slot_tensor_out(
            history_logprob_per_agent_out,
            row_indices_t=history_row_indices_t,
            value=logprob_grouped,
        )
        if logprob_out is not None:
            if tuple(logprob_out.shape) != (int(num_envs),):
                raise ValueError(f"Target shape {tuple(logprob_out.shape)} does not match ({int(num_envs)},).")
            _copy_group_logprob_sum_out(logprob_out, logprob)
            _write_history_slot_tensor_out(
                history_logprob_out,
                row_indices_t=history_row_indices_t,
                value=logprob_grouped.sum(dim=1),
            )

    def evaluate_actions(
        self,
        local_state: LocalSatState,
        subset_index: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> SatSubsetPolicyOutput:
        return self._select(
            local_state,
            subset_index=subset_index,
            deterministic=True,
            compute_entropy=bool(compute_entropy),
        )

    def topk_legal_subset_indices(
        self,
        local_state: LocalSatState,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = max(int(k), 0)
        logits = self._compute_logits(local_state)
        batch_size = int(logits.shape[0])
        if topk <= 0:
            empty_idx = torch.full((batch_size, 0), -1, dtype=torch.long, device=logits.device)
            empty_logits = torch.empty((batch_size, 0), dtype=logits.dtype, device=logits.device)
            return empty_idx, empty_logits
        subset_mask = self._legal_subset_mask(local_state, logits)
        safe_logits = logits.masked_fill(~subset_mask, float("-inf"))
        topk = min(topk, int(logits.shape[-1]))
        topk_values, topk_indices = torch.topk(safe_logits, k=topk, dim=-1)
        valid_counts = subset_mask.sum(dim=-1, keepdim=True)
        rank = torch.arange(topk, device=logits.device, dtype=torch.long).view(1, -1)
        keep_mask = rank < valid_counts
        topk_indices = torch.where(keep_mask, topk_indices, torch.full_like(topk_indices, -1))
        topk_values = torch.where(keep_mask, topk_values, torch.full_like(topk_values, float("-inf")))
        return topk_indices, topk_values


class BwCompetitionBlock(nn.Module):
    """Masked user self-attention block for competition-aware BW allocation."""

    def __init__(
        self,
        *,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        manual_attention_enabled: bool = False,
    ) -> None:
        super().__init__()
        head_count = max(int(num_heads), 1)
        if int(embed_dim) % head_count != 0:
            raise ValueError(
                f"BW competition heads={head_count} must divide embed_dim={int(embed_dim)}."
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=head_count,
            batch_first=True,
        )
        self.manual_attention_enabled = bool(manual_attention_enabled)
        self.norm_attn = nn.LayerNorm(int(embed_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )
        self.norm_ffn = nn.LayerNorm(int(embed_dim))

    def _manual_attention(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        batch, token_count, embed_dim = q.shape
        head_count = int(self.attn.num_heads)
        head_dim = int(embed_dim) // max(head_count, 1)
        q = q.view(batch, token_count, head_count, head_dim).transpose(1, 2)
        k = k.view(batch, token_count, head_count, head_dim).transpose(1, 2)
        v = v.view(batch, token_count, head_count, head_dim).transpose(1, 2)
        attn_mask = torch.where(
            key_padding_mask[:, None, None, :],
            torch.full((), float("-inf"), dtype=h.dtype, device=h.device),
            torch.zeros((), dtype=h.dtype, device=h.device),
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch, token_count, embed_dim)
        return self.attn.out_proj(attn)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if int(h.shape[-2]) == 0:
            return h
        valid = valid_mask.to(dtype=torch.bool)
        valid_f = valid.unsqueeze(-1).to(dtype=h.dtype)
        h = h * valid_f
        has_valid = valid.any(dim=-1, keepdim=True)
        key_padding_mask = torch.where(
            has_valid,
            ~valid,
            torch.zeros_like(valid),
        )
        if bool(self.manual_attention_enabled):
            attn_out = self._manual_attention(h, key_padding_mask)
        else:
            attn_out, _weights = self.attn(
                h,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        h = self.norm_attn(h + attn_out)
        h = h * valid_f
        h = self.norm_ffn(h + self.ffn(h))
        h = h * valid_f
        return h


def _make_mlp2(in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2) -> nn.Sequential:
    return _make_mlp(
        int(in_dim),
        int(hidden_dim),
        int(out_dim),
        num_layers=max(int(num_layers), 1),
        activation_factory=nn.SiLU,
        final_activation=False,
    )


class BwPolicy(nn.Module):
    def __init__(
        self,
        *,
        ego_dim: int = bw_schema.BW_EGO_DIM,
        sat_token_dim: int = bw_schema.BW_SAT_TOKEN_DIM,
        gu_token_dim: int = bw_schema.BW_GU_TOKEN_DIM,
        embed_dim: int,
        hidden_dim: int,
        down_query_count: int = 2,
        num_competition_layers: int = 2,
        num_heads: int = 4,
        encoder_mlp_layers: int = 2,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 2,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        kappa_min: float = 0.5,
        kappa_max: float = 32.0,
        fixed_tau: float | None = None,
        fixed_kappa: float | None = None,
        native_dirichlet_diagnostic_mode: str = "current",
        manual_competition_attention_enabled: bool = False,
        input_norm_enabled: bool | None = False,
    ) -> None:
        super().__init__()
        if int(ego_dim) != bw_schema.BW_EGO_DIM:
            raise ValueError(f"BW ego_dim must be {bw_schema.BW_EGO_DIM}, got {ego_dim}.")
        if int(sat_token_dim) != bw_schema.BW_SAT_TOKEN_DIM:
            raise ValueError(f"BW sat_token_dim must be {bw_schema.BW_SAT_TOKEN_DIM}, got {sat_token_dim}.")
        if int(gu_token_dim) != bw_schema.BW_GU_TOKEN_DIM:
            raise ValueError(f"BW gu_token_dim must be {bw_schema.BW_GU_TOKEN_DIM}, got {gu_token_dim}.")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.down_query_count = int(down_query_count)
        self.num_competition_layers = int(num_competition_layers)
        self.num_heads = int(num_heads)
        self.encoder_mlp_layers = max(int(encoder_mlp_layers), 1)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.fixed_tau = None if fixed_tau is None else float(fixed_tau)
        self.fixed_kappa = None if fixed_kappa is None else float(fixed_kappa)
        self.native_dirichlet_diagnostic_mode = str(native_dirichlet_diagnostic_mode or "current").strip().lower()
        self.manual_competition_attention_enabled = bool(manual_competition_attention_enabled)
        if self.down_query_count < 1:
            raise ValueError("bw_down_query_count must be >= 1.")
        if self.num_competition_layers < 1:
            raise ValueError("bw_competition_layers must be >= 1.")
        if self.embed_dim % max(self.num_heads, 1) != 0:
            raise ValueError("bw_attention_heads must divide actor_embed_dim.")
        if not (0.0 < self.tau_min < self.tau_max):
            raise ValueError("BW tau range must satisfy 0 < tau_min < tau_max.")
        if not (0.0 < self.kappa_min < self.kappa_max):
            raise ValueError("BW kappa range must satisfy 0 < kappa_min < kappa_max.")
        if self.fixed_tau is not None and not (self.fixed_tau > 0.0):
            raise ValueError("fixed_tau must be positive when set.")
        if self.fixed_kappa is not None and not (self.fixed_kappa > 0.0):
            raise ValueError("fixed_kappa must be positive when set.")
        if self.native_dirichlet_diagnostic_mode not in {"current", "new_fast", "legacy_fast"}:
            raise ValueError(
                "native_dirichlet_diagnostic_mode must be one of "
                "{'current', 'new_fast', 'legacy_fast'}."
            )

        self.ego_input_norm = _make_raw_input_norm(bw_schema.BW_EGO_DIM, input_norm_enabled)
        self.sat_input_norm = _make_raw_input_norm(bw_schema.BW_SAT_TOKEN_DIM, input_norm_enabled)
        self.gu_input_norm = _make_raw_input_norm(bw_schema.BW_GU_TOKEN_DIM, input_norm_enabled)
        self.ego_encoder = _make_mlp2(
            bw_schema.BW_EGO_DIM,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.sat_encoder = _make_mlp2(
            bw_schema.BW_SAT_TOKEN_DIM,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.gu_encoder = _make_mlp2(
            bw_schema.BW_GU_TOKEN_DIM,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.down_query_proj = nn.Linear(self.embed_dim, self.down_query_count * self.embed_dim)
        self.sat_add_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.down_context_encoder = _make_mlp2(
            (self.down_query_count + 1) * self.embed_dim,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.context_mlp_layers,
        )
        self.ctx0_encoder = _make_mlp2(
            2 * self.embed_dim,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.context_mlp_layers,
        )
        self.gu_context_fusion = _make_mlp2(
            2 * self.embed_dim,
            self.hidden_dim,
            self.embed_dim,
            num_layers=self.context_mlp_layers,
        )
        self.competition_blocks = nn.ModuleList(
            [
                BwCompetitionBlock(
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    manual_attention_enabled=self.manual_competition_attention_enabled,
                )
                for _ in range(self.num_competition_layers)
            ]
        )
        self.score_head = _make_mlp2(self.embed_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)
        self.tau_head = _make_mlp2(self.embed_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)
        self.kappa_head = _make_mlp2(self.embed_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)

    @staticmethod
    def _masked_sum(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if int(tokens.shape[-2]) == 0:
            return tokens.new_zeros(tokens.shape[:-2] + (tokens.shape[-1],))
        mask_f = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * mask_f).sum(dim=-2)

    @staticmethod
    def _objective_denominator(valid_count: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.clamp(valid_count.to(dtype=dtype) - 1.0, min=1.0)

    def _params(
        self,
        local_state: LocalBwState,
        *,
        include_alpha: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        valid = local_state.gu_mask.to(dtype=torch.bool) & local_state.bw_valid_mask.to(dtype=torch.bool)
        valid_count = valid.sum(dim=-1).to(dtype=torch.long)
        latent_count = torch.clamp(valid_count - 1, min=0)
        row_count = int(local_state.ego_features.shape[0])
        embed_dim = self.embed_dim

        ego_emb = self.ego_encoder(self.ego_input_norm(local_state.ego_features))
        sat_emb = self.sat_encoder(self.sat_input_norm(local_state.selected_sat_tokens))
        gu_emb = self.gu_encoder(self.gu_input_norm(local_state.gu_tokens))

        queries = self.down_query_proj(ego_emb).view(row_count, self.down_query_count, embed_dim)
        sat_mask = local_state.selected_sat_mask.to(dtype=torch.bool)
        down_attn = _multi_query_attention(queries, sat_emb, sat_mask)
        sat_add_pool = self._masked_sum(self.sat_add_proj(sat_emb), sat_mask)
        down_ctx = self.down_context_encoder(torch.cat([down_attn.flatten(1), sat_add_pool], dim=-1))

        ctx0 = self.ctx0_encoder(torch.cat([ego_emb, down_ctx], dim=-1))
        ctx_expand = ctx0[:, None, :].expand(-1, int(gu_emb.shape[1]), -1)
        gu_h = self.gu_context_fusion(torch.cat([gu_emb, ctx_expand], dim=-1))
        for block in self.competition_blocks:
            gu_h = block(gu_h, valid)

        raw_score = self.score_head(gu_h).squeeze(-1)
        score = raw_score.masked_fill(~valid, NEG_INF)
        if self.fixed_tau is None:
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_head(ctx0)).squeeze(-1)
        else:
            tau = ctx0.new_full((row_count,), float(self.fixed_tau))
        det_mean = _masked_softmax(score / tau[:, None], valid, dim=-1)
        det_mean = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(det_mean.dtype), det_mean)
        det_mean = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(det_mean), det_mean)
        if self.fixed_kappa is None:
            kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.kappa_head(ctx0)).squeeze(-1)
        else:
            kappa = ctx0.new_full((row_count,), float(self.fixed_kappa))
        if bool(include_alpha):
            alpha = (det_mean * kappa[:, None]).masked_fill(~valid, 0.0)
        else:
            alpha = torch.empty((0,), dtype=det_mean.dtype, device=det_mean.device)
        return score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid

    def _output(
        self,
        *,
        action: torch.Tensor,
        logprob_raw: torch.Tensor,
        entropy_raw: torch.Tensor,
        score: torch.Tensor,
        det_mean: torch.Tensor,
        alpha: torch.Tensor,
        kappa: torch.Tensor,
        tau: torch.Tensor,
        valid_count: torch.Tensor,
        latent_count: torch.Tensor,
    ) -> BwPolicyOutput:
        denom = self._objective_denominator(valid_count, logprob_raw.dtype)
        return BwPolicyOutput(
            action=action,
            logprob=logprob_raw / denom,
            entropy=entropy_raw / denom,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            valid_count=valid_count,
            latent_count=latent_count,
            tau=tau,
        )

    def forward(self, local_state: LocalBwState, deterministic: bool = False) -> BwPolicyOutput:
        score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid = self._params(local_state)
        dist = MaskedMeanConcentrationDirichlet(mean=det_mean, kappa=kappa, mask=valid)
        action = det_mean if deterministic else dist.rsample()
        action = action.masked_fill(~valid, 0.0)
        action = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(action.dtype), action)
        action = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(action), action)
        logprob_raw = dist.log_prob(action)
        entropy_raw = dist.entropy()
        return self._output(
            action=action,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            tau=tau,
            valid_count=valid_count,
            latent_count=latent_count,
        )

    def evaluate_actions(
        self,
        local_state: LocalBwState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> BwPolicyOutput:
        score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid = self._params(local_state)
        dist = MaskedMeanConcentrationDirichlet(mean=det_mean, kappa=kappa, mask=valid)
        action_eval = action.to(dtype=det_mean.dtype, device=det_mean.device).masked_fill(~valid, 0.0)
        action_eval = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(action_eval.dtype), action_eval)
        action_eval = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(action_eval), action_eval)
        logprob_raw = dist.log_prob(action_eval)
        entropy_raw = dist.entropy() if bool(compute_entropy) else torch.zeros_like(logprob_raw)
        return self._output(
            action=action_eval,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            tau=tau,
            valid_count=valid_count,
            latent_count=latent_count,
        )

    def act_into(
        self,
        local_state: LocalBwState,
        *,
        action_out: torch.Tensor | None = None,
        ref_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        entropy_per_agent_out: torch.Tensor | None = None,
        logprob_raw_per_agent_out: torch.Tensor | None = None,
        entropy_raw_per_agent_out: torch.Tensor | None = None,
        tau_out: torch.Tensor | None = None,
        kappa_out: torch.Tensor | None = None,
        valid_count_out: torch.Tensor | None = None,
        latent_count_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_ref_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        out = self.forward(local_state, deterministic=deterministic)
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        action_grouped = out.action.reshape(int(num_envs), int(num_agents), -1)
        logprob_grouped = out.logprob.reshape(int(num_envs), int(num_agents))
        entropy_grouped = out.entropy.reshape(int(num_envs), int(num_agents))
        logprob_raw_grouped = out.logprob_raw.reshape(int(num_envs), int(num_agents))
        entropy_raw_grouped = out.entropy_raw.reshape(int(num_envs), int(num_agents))
        tau_grouped = out.tau.reshape(int(num_envs), int(num_agents))
        kappa_grouped = out.kappa.reshape(int(num_envs), int(num_agents))
        valid_count_grouped = out.valid_count.reshape(int(num_envs), int(num_agents))
        latent_count_grouped = out.latent_count.reshape(int(num_envs), int(num_agents))
        det_grouped = out.det_mean.reshape(int(num_envs), int(num_agents), -1)
        if action_out is not None:
            _copy_tensor_out(action_out, _reshape_group_agent_tensor_for_target(out.action, action_out, num_envs=num_envs, num_agents=num_agents, name="bw action"))
        if ref_action_out is not None:
            _copy_tensor_out(ref_action_out, _reshape_group_agent_tensor_for_target(out.det_mean, ref_action_out, num_envs=num_envs, num_agents=num_agents, name="bw ref action"))
        if logprob_per_agent_out is not None:
            _copy_tensor_out(logprob_per_agent_out, _reshape_group_agent_tensor_for_target(out.logprob, logprob_per_agent_out, num_envs=num_envs, num_agents=num_agents, name="bw per-agent logprob"))
        if entropy_per_agent_out is not None:
            _copy_tensor_out(entropy_per_agent_out, entropy_grouped)
        if logprob_raw_per_agent_out is not None:
            _copy_tensor_out(logprob_raw_per_agent_out, logprob_raw_grouped)
        if entropy_raw_per_agent_out is not None:
            _copy_tensor_out(entropy_raw_per_agent_out, entropy_raw_grouped)
        if tau_out is not None:
            _copy_tensor_out(tau_out, tau_grouped)
        if kappa_out is not None:
            _copy_tensor_out(kappa_out, kappa_grouped)
        if valid_count_out is not None:
            _copy_tensor_out(valid_count_out, valid_count_grouped)
        if latent_count_out is not None:
            _copy_tensor_out(latent_count_out, latent_count_grouped)
        if logprob_out is not None:
            _copy_tensor_out(logprob_out, logprob_grouped.sum(dim=1))
        _write_history_slot_tensor_out(history_action_out, row_indices_t=history_row_indices_t, value=action_grouped)
        _write_history_slot_tensor_out(history_ref_action_out, row_indices_t=history_row_indices_t, value=det_grouped)
        _write_history_slot_tensor_out(history_logprob_per_agent_out, row_indices_t=history_row_indices_t, value=logprob_grouped)
        _write_history_slot_tensor_out(history_logprob_out, row_indices_t=history_row_indices_t, value=logprob_grouped.sum(dim=1))


class FlatBwPolicy(nn.Module):
    """Flat MLP BW allocation head for MAPPO-like baselines."""

    def __init__(
        self,
        *,
        ego_dim: int = bw_schema.BW_EGO_DIM,
        sat_token_dim: int = bw_schema.BW_SAT_TOKEN_DIM,
        gu_token_dim: int = bw_schema.BW_GU_TOKEN_DIM,
        selected_sat_token_count: int,
        gu_token_count: int,
        hidden_dim: int,
        context_mlp_layers: int = 2,
        head_mlp_layers: int = 2,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        kappa_min: float = 0.5,
        kappa_max: float = 32.0,
        fixed_tau: float | None = None,
        fixed_kappa: float | None = None,
        native_dirichlet_diagnostic_mode: str = "current",
        input_norm_enabled: bool | None = False,
    ) -> None:
        super().__init__()
        if int(ego_dim) != bw_schema.BW_EGO_DIM:
            raise ValueError(f"BW ego_dim must be {bw_schema.BW_EGO_DIM}, got {ego_dim}.")
        if int(sat_token_dim) != bw_schema.BW_SAT_TOKEN_DIM:
            raise ValueError(f"BW sat_token_dim must be {bw_schema.BW_SAT_TOKEN_DIM}, got {sat_token_dim}.")
        if int(gu_token_dim) != bw_schema.BW_GU_TOKEN_DIM:
            raise ValueError(f"BW gu_token_dim must be {bw_schema.BW_GU_TOKEN_DIM}, got {gu_token_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.selected_sat_token_count = max(int(selected_sat_token_count), 0)
        self.gu_token_count = max(int(gu_token_count), 1)
        self.context_mlp_layers = max(int(context_mlp_layers), 1)
        self.head_mlp_layers = max(int(head_mlp_layers), 1)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.fixed_tau = None if fixed_tau is None else float(fixed_tau)
        self.fixed_kappa = None if fixed_kappa is None else float(fixed_kappa)
        self.native_dirichlet_diagnostic_mode = str(native_dirichlet_diagnostic_mode or "current").strip().lower()
        if not (0.0 < self.tau_min < self.tau_max):
            raise ValueError("BW tau range must satisfy 0 < tau_min < tau_max.")
        if not (0.0 < self.kappa_min < self.kappa_max):
            raise ValueError("BW kappa range must satisfy 0 < kappa_min < kappa_max.")
        if self.fixed_tau is not None and not (self.fixed_tau > 0.0):
            raise ValueError("fixed_tau must be positive when set.")
        if self.fixed_kappa is not None and not (self.fixed_kappa > 0.0):
            raise ValueError("fixed_kappa must be positive when set.")
        if self.native_dirichlet_diagnostic_mode not in {"current", "new_fast", "legacy_fast"}:
            raise ValueError(
                "native_dirichlet_diagnostic_mode must be one of "
                "{'current', 'new_fast', 'legacy_fast'}."
            )
        input_dim = (
            bw_schema.BW_EGO_DIM
            + self.selected_sat_token_count * (bw_schema.BW_SAT_TOKEN_DIM + 1)
            + self.gu_token_count * (bw_schema.BW_GU_TOKEN_DIM + 2)
        )
        self.input_norm = _make_raw_input_norm(input_dim, input_norm_enabled)
        self.trunk = _make_mlp2(input_dim, self.hidden_dim, self.hidden_dim, num_layers=self.context_mlp_layers)
        self.score_head = _make_mlp2(self.hidden_dim, self.hidden_dim, self.gu_token_count, num_layers=self.head_mlp_layers)
        self.tau_head = _make_mlp2(self.hidden_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)
        self.kappa_head = _make_mlp2(self.hidden_dim, self.hidden_dim, 1, num_layers=self.head_mlp_layers)

    @staticmethod
    def _objective_denominator(valid_count: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.clamp(valid_count.to(dtype=dtype) - 1.0, min=1.0)

    def _features(self, local_state: LocalBwState) -> torch.Tensor:
        gu_mask = local_state.gu_mask.to(dtype=torch.bool)
        bw_valid = local_state.bw_valid_mask.to(dtype=torch.bool)
        gu_features = _fixed_masked_token_features(local_state.gu_tokens, gu_mask, self.gu_token_count)
        valid_features = _fit_width_last(bw_valid.to(dtype=local_state.gu_tokens.dtype), self.gu_token_count)
        sat_features = _fixed_masked_token_features(
            local_state.selected_sat_tokens,
            local_state.selected_sat_mask,
            self.selected_sat_token_count,
        )
        return torch.cat([local_state.ego_features, sat_features, gu_features, valid_features], dim=-1)

    def _params(
        self,
        local_state: LocalBwState,
        *,
        include_alpha: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        valid = local_state.gu_mask.to(dtype=torch.bool) & local_state.bw_valid_mask.to(dtype=torch.bool)
        valid_count = valid.sum(dim=-1).to(dtype=torch.long)
        latent_count = torch.clamp(valid_count - 1, min=0)
        row_count = int(local_state.ego_features.shape[0])
        ctx = self.trunk(self.input_norm(self._features(local_state)))
        raw_score = _fit_width_last(self.score_head(ctx), int(local_state.gu_tokens.shape[1]))
        score = raw_score.masked_fill(~valid, NEG_INF)
        if self.fixed_tau is None:
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_head(ctx)).squeeze(-1)
        else:
            tau = ctx.new_full((row_count,), float(self.fixed_tau))
        det_mean = _masked_softmax(score / tau[:, None], valid, dim=-1)
        det_mean = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(det_mean.dtype), det_mean)
        det_mean = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(det_mean), det_mean)
        if self.fixed_kappa is None:
            kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.kappa_head(ctx)).squeeze(-1)
        else:
            kappa = ctx.new_full((row_count,), float(self.fixed_kappa))
        if bool(include_alpha):
            alpha = (det_mean * kappa[:, None]).masked_fill(~valid, 0.0)
        else:
            alpha = torch.empty((0,), dtype=det_mean.dtype, device=det_mean.device)
        return score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid

    def _output(
        self,
        *,
        action: torch.Tensor,
        logprob_raw: torch.Tensor,
        entropy_raw: torch.Tensor,
        score: torch.Tensor,
        det_mean: torch.Tensor,
        alpha: torch.Tensor,
        kappa: torch.Tensor,
        tau: torch.Tensor,
        valid_count: torch.Tensor,
        latent_count: torch.Tensor,
    ) -> BwPolicyOutput:
        denom = self._objective_denominator(valid_count, logprob_raw.dtype)
        return BwPolicyOutput(
            action=action,
            logprob=logprob_raw / denom,
            entropy=entropy_raw / denom,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            valid_count=valid_count,
            latent_count=latent_count,
            tau=tau,
        )

    def forward(self, local_state: LocalBwState, deterministic: bool = False) -> BwPolicyOutput:
        score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid = self._params(local_state)
        dist = MaskedMeanConcentrationDirichlet(mean=det_mean, kappa=kappa, mask=valid)
        action = det_mean if deterministic else dist.rsample()
        action = action.masked_fill(~valid, 0.0)
        action = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(action.dtype), action)
        action = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(action), action)
        logprob_raw = dist.log_prob(action)
        entropy_raw = dist.entropy()
        return self._output(
            action=action,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            tau=tau,
            valid_count=valid_count,
            latent_count=latent_count,
        )

    def evaluate_actions(
        self,
        local_state: LocalBwState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> BwPolicyOutput:
        score, det_mean, alpha, kappa, tau, valid_count, latent_count, valid = self._params(local_state)
        dist = MaskedMeanConcentrationDirichlet(mean=det_mean, kappa=kappa, mask=valid)
        action_eval = action.to(dtype=det_mean.dtype, device=det_mean.device).masked_fill(~valid, 0.0)
        action_eval = torch.where(valid_count.unsqueeze(-1) == 1, valid.to(action_eval.dtype), action_eval)
        action_eval = torch.where(valid_count.unsqueeze(-1) <= 0, torch.zeros_like(action_eval), action_eval)
        logprob_raw = dist.log_prob(action_eval)
        entropy_raw = dist.entropy() if bool(compute_entropy) else torch.zeros_like(logprob_raw)
        return self._output(
            action=action_eval,
            logprob_raw=logprob_raw,
            entropy_raw=entropy_raw,
            score=score,
            det_mean=det_mean,
            alpha=alpha,
            kappa=kappa,
            tau=tau,
            valid_count=valid_count,
            latent_count=latent_count,
        )

    def act_into(
        self,
        local_state: LocalBwState,
        *,
        action_out: torch.Tensor | None = None,
        ref_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        entropy_per_agent_out: torch.Tensor | None = None,
        logprob_raw_per_agent_out: torch.Tensor | None = None,
        entropy_raw_per_agent_out: torch.Tensor | None = None,
        tau_out: torch.Tensor | None = None,
        kappa_out: torch.Tensor | None = None,
        valid_count_out: torch.Tensor | None = None,
        latent_count_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_ref_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        out = self.forward(local_state, deterministic=deterministic)
        history_row_indices_t = _history_slot_row_indices(
            history_slot_t=history_slot_t,
            base_row_ids_t=history_env_row_ids_t,
        )
        action_grouped = out.action.reshape(int(num_envs), int(num_agents), -1)
        logprob_grouped = out.logprob.reshape(int(num_envs), int(num_agents))
        entropy_grouped = out.entropy.reshape(int(num_envs), int(num_agents))
        logprob_raw_grouped = out.logprob_raw.reshape(int(num_envs), int(num_agents))
        entropy_raw_grouped = out.entropy_raw.reshape(int(num_envs), int(num_agents))
        tau_grouped = out.tau.reshape(int(num_envs), int(num_agents))
        kappa_grouped = out.kappa.reshape(int(num_envs), int(num_agents))
        valid_count_grouped = out.valid_count.reshape(int(num_envs), int(num_agents))
        latent_count_grouped = out.latent_count.reshape(int(num_envs), int(num_agents))
        det_grouped = out.det_mean.reshape(int(num_envs), int(num_agents), -1)
        if action_out is not None:
            _copy_tensor_out(action_out, _reshape_group_agent_tensor_for_target(out.action, action_out, num_envs=num_envs, num_agents=num_agents, name="bw action"))
        if ref_action_out is not None:
            _copy_tensor_out(ref_action_out, _reshape_group_agent_tensor_for_target(out.det_mean, ref_action_out, num_envs=num_envs, num_agents=num_agents, name="bw ref action"))
        if logprob_per_agent_out is not None:
            _copy_tensor_out(logprob_per_agent_out, _reshape_group_agent_tensor_for_target(out.logprob, logprob_per_agent_out, num_envs=num_envs, num_agents=num_agents, name="bw per-agent logprob"))
        if entropy_per_agent_out is not None:
            _copy_tensor_out(entropy_per_agent_out, entropy_grouped)
        if logprob_raw_per_agent_out is not None:
            _copy_tensor_out(logprob_raw_per_agent_out, logprob_raw_grouped)
        if entropy_raw_per_agent_out is not None:
            _copy_tensor_out(entropy_raw_per_agent_out, entropy_raw_grouped)
        if tau_out is not None:
            _copy_tensor_out(tau_out, tau_grouped)
        if kappa_out is not None:
            _copy_tensor_out(kappa_out, kappa_grouped)
        if valid_count_out is not None:
            _copy_tensor_out(valid_count_out, valid_count_grouped)
        if latent_count_out is not None:
            _copy_tensor_out(latent_count_out, latent_count_grouped)
        if logprob_out is not None:
            _copy_tensor_out(logprob_out, logprob_grouped.sum(dim=1))
        _write_history_slot_tensor_out(history_action_out, row_indices_t=history_row_indices_t, value=action_grouped)
        _write_history_slot_tensor_out(history_ref_action_out, row_indices_t=history_row_indices_t, value=det_grouped)
        _write_history_slot_tensor_out(history_logprob_per_agent_out, row_indices_t=history_row_indices_t, value=logprob_grouped)
        _write_history_slot_tensor_out(history_logprob_out, row_indices_t=history_row_indices_t, value=logprob_grouped.sum(dim=1))


class StructuredActor(nn.Module):
    def __init__(
        self,
        accel_policy: AccelPolicy,
        sat_subset_policy: SatSubsetPolicy,
        bw_policy: BwPolicy,
    ):
        super().__init__()
        self.accel_policy = accel_policy
        self.sat_subset_policy = sat_subset_policy
        self.bw_policy = bw_policy

    def act_accel(self, local_state: LocalAccelState, deterministic: bool = False) -> AccelPolicyOutput:
        return self.accel_policy(local_state, deterministic=deterministic)

    def act_sat(self, local_state: LocalSatState, deterministic: bool = False) -> SatSubsetPolicyOutput:
        return self.sat_subset_policy(local_state, deterministic=deterministic)

    def act_bw(self, local_state: LocalBwState, deterministic: bool = False) -> BwPolicyOutput:
        return self.bw_policy(local_state, deterministic=deterministic)

    def act_accel_into(
        self,
        local_state: LocalAccelState,
        *,
        action_out: torch.Tensor | None = None,
        latent_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_latent_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        self.accel_policy.act_into(
            local_state,
            action_out=action_out,
            latent_action_out=latent_action_out,
            logprob_out=logprob_out,
            history_action_out=history_action_out,
            history_latent_action_out=history_latent_action_out,
            history_logprob_out=history_logprob_out,
            history_slot_t=history_slot_t,
            history_env_row_ids_t=history_env_row_ids_t,
            deterministic=deterministic,
            num_envs=num_envs,
            num_agents=num_agents,
        )

    def act_sat_into(
        self,
        local_state: LocalSatState,
        *,
        subset_index_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        self.sat_subset_policy.act_into(
            local_state,
            subset_index_out=subset_index_out,
            logprob_out=logprob_out,
            logprob_per_agent_out=logprob_per_agent_out,
            history_action_out=history_action_out,
            history_logprob_out=history_logprob_out,
            history_logprob_per_agent_out=history_logprob_per_agent_out,
            history_slot_t=history_slot_t,
            history_env_row_ids_t=history_env_row_ids_t,
            deterministic=deterministic,
            num_envs=num_envs,
            num_agents=num_agents,
        )

    def act_bw_into(
        self,
        local_state: LocalBwState,
        *,
        action_out: torch.Tensor | None = None,
        ref_action_out: torch.Tensor | None = None,
        logprob_out: torch.Tensor | None = None,
        logprob_per_agent_out: torch.Tensor | None = None,
        entropy_per_agent_out: torch.Tensor | None = None,
        logprob_raw_per_agent_out: torch.Tensor | None = None,
        entropy_raw_per_agent_out: torch.Tensor | None = None,
        tau_out: torch.Tensor | None = None,
        kappa_out: torch.Tensor | None = None,
        valid_count_out: torch.Tensor | None = None,
        latent_count_out: torch.Tensor | None = None,
        history_action_out: torch.Tensor | None = None,
        history_ref_action_out: torch.Tensor | None = None,
        history_logprob_out: torch.Tensor | None = None,
        history_logprob_per_agent_out: torch.Tensor | None = None,
        history_slot_t: torch.Tensor | None = None,
        history_env_row_ids_t: torch.Tensor | None = None,
        deterministic: bool = False,
        num_envs: int,
        num_agents: int,
    ) -> None:
        self.bw_policy.act_into(
            local_state,
            action_out=action_out,
            ref_action_out=ref_action_out,
            logprob_out=logprob_out,
            logprob_per_agent_out=logprob_per_agent_out,
            entropy_per_agent_out=entropy_per_agent_out,
            logprob_raw_per_agent_out=logprob_raw_per_agent_out,
            entropy_raw_per_agent_out=entropy_raw_per_agent_out,
            tau_out=tau_out,
            kappa_out=kappa_out,
            valid_count_out=valid_count_out,
            latent_count_out=latent_count_out,
            history_action_out=history_action_out,
            history_ref_action_out=history_ref_action_out,
            history_logprob_out=history_logprob_out,
            history_logprob_per_agent_out=history_logprob_per_agent_out,
            history_slot_t=history_slot_t,
            history_env_row_ids_t=history_env_row_ids_t,
            deterministic=deterministic,
            num_envs=num_envs,
            num_agents=num_agents,
        )

    def evaluate_accel(
        self,
        local_state: LocalAccelState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
        latent_action: torch.Tensor | None = None,
    ) -> AccelPolicyOutput:
        return self.accel_policy.evaluate_actions(
            local_state,
            action,
            compute_entropy=bool(compute_entropy),
            latent_action=latent_action,
        )

    def evaluate_sat(
        self,
        local_state: LocalSatState,
        subset_index: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> SatSubsetPolicyOutput:
        return self.sat_subset_policy.evaluate_actions(
            local_state,
            subset_index,
            compute_entropy=bool(compute_entropy),
        )

    def sat_topk_legal_subset_indices(
        self,
        local_state: LocalSatState,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sat_subset_policy.topk_legal_subset_indices(local_state, k)

    def evaluate_bw(
        self,
        local_state: LocalBwState,
        action: torch.Tensor,
        *,
        compute_entropy: bool = True,
    ) -> BwPolicyOutput:
        return self.bw_policy.evaluate_actions(
            local_state,
            action,
            compute_entropy=bool(compute_entropy),
        )
