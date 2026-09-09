from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import structured_critic_schema as schema
from .structured_types import LocalBwState, StructuredWorldState


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int | None = None,
    *,
    num_layers: int = 2,
    final_activation: bool | None = None,
) -> nn.Sequential:
    out_dim = hidden_dim if out_dim is None else out_dim
    layer_count = max(int(num_layers), 1)
    modules: list[nn.Module] = []
    if layer_count == 1:
        modules.append(nn.Linear(int(in_dim), int(out_dim)))
    else:
        modules.extend([nn.Linear(int(in_dim), int(hidden_dim)), nn.ReLU()])
        for _ in range(layer_count - 2):
            modules.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU()])
        modules.append(nn.Linear(int(hidden_dim), int(out_dim)))
    use_final_activation = (int(out_dim) == int(hidden_dim)) if final_activation is None else bool(final_activation)
    modules.append(nn.ReLU() if use_final_activation else nn.Identity())
    return nn.Sequential(*modules)


def _make_input_norm(input_dim: int, enabled: bool) -> nn.Module:
    return nn.LayerNorm(input_dim) if enabled else nn.Identity()


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    safe_scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(safe_scores, dim=dim)
    weights = weights * mask.to(weights.dtype)
    norm = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights / norm


def _masked_stats(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_f = mask.to(dtype=values.dtype)
    raw_count = mask_f.sum(dim=dims)
    count = raw_count.clamp_min(1.0)
    mean = (values * mask_f).sum(dim=dims) / count
    centered = torch.where(mask, values - mean.view([-1] + [1] * (values.ndim - 1)), torch.zeros_like(values))
    std = ((centered.pow(2) * mask_f).sum(dim=dims) / count).sqrt()
    masked_values = torch.where(mask, values, torch.full_like(values, -1.0e9))
    max_value = masked_values.amax(dim=dims)
    max_value = torch.where(raw_count > 0.5, max_value, torch.zeros_like(max_value))
    return mean, std, max_value


def _masked_mean_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-2] == 0:
        return tokens.new_zeros(tokens.shape[:-2] + (tokens.shape[-1],))
    mask_f = mask.to(dtype=tokens.dtype)
    denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (tokens * mask_f.unsqueeze(-1)).sum(dim=-2) / denom


def _masked_max_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-2] == 0:
        return tokens.new_zeros(tokens.shape[:-2] + (tokens.shape[-1],))
    masked = torch.where(mask.unsqueeze(-1), tokens, torch.full_like(tokens, -1.0e9))
    max_token = masked.amax(dim=-2)
    has_valid = mask.any(dim=-1, keepdim=True)
    return torch.where(has_valid, max_token, torch.zeros_like(max_token))


class ZeroStructuredCritic(nn.Module):
    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__()
        target_device = torch.device("cpu" if device is None else device)
        self.register_buffer("_device_anchor", torch.zeros((), dtype=torch.float32, device=target_device))
        self.popart_enabled = False

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _zeros(self, world_state_batch) -> torch.Tensor:
        batch = int(world_state_batch.uav_nodes.shape[0])
        return torch.zeros((batch,), dtype=torch.float32, device=self.device)

    def value_accel(self, world_state_batch) -> torch.Tensor:
        return self._zeros(world_state_batch)

    def value_sat(self, world_state_batch) -> torch.Tensor:
        return self._zeros(world_state_batch)

    def value_bw(self, world_state_batch) -> torch.Tensor:
        return self._zeros(world_state_batch)

    def delta_bw(
        self,
        local_state: LocalBwState,
        sampled_action: torch.Tensor,
        ref_action: torch.Tensor,
        *,
        num_agents: int = 1,
    ) -> torch.Tensor:
        del local_state, sampled_action, ref_action, num_agents
        raise RuntimeError("ZeroStructuredCritic does not provide BW delta estimates.")


def _masked_sum_tokens(tokens: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if tokens.shape[dim] == 0:
        out_shape = list(tokens.shape)
        del out_shape[dim]
        return tokens.new_zeros(out_shape)
    return (tokens * mask.to(tokens.dtype).unsqueeze(-1)).sum(dim=dim)


class _TypedRelationalBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        edge_dim: int,
        system_dim: int,
        global_dim: int,
        hidden_dim: int,
        *,
        message_mlp_layers: int = 2,
        fixed_bounded_relations_enabled: bool = False,
    ) -> None:
        super().__init__()
        mlp_layers = max(int(message_mlp_layers), 1)
        self.fixed_bounded_relations_enabled = bool(fixed_bounded_relations_enabled)
        self.ug_to_g = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.ug_to_u = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.us_to_u = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.us_to_s = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.uu_to_u = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)

        self.gu_update = _make_mlp(token_dim * 2, hidden_dim, token_dim, num_layers=mlp_layers)
        self.uav_update = _make_mlp(token_dim * 5, hidden_dim, token_dim, num_layers=mlp_layers)
        self.sat_update = _make_mlp(token_dim * 3, hidden_dim, token_dim, num_layers=mlp_layers)

        self.gu_to_system = _make_mlp(token_dim + system_dim, hidden_dim, system_dim, num_layers=mlp_layers)
        self.uav_to_system = _make_mlp(token_dim + system_dim, hidden_dim, system_dim, num_layers=mlp_layers)
        self.sat_to_system = _make_mlp(token_dim + system_dim, hidden_dim, system_dim, num_layers=mlp_layers)
        self.system_update = _make_mlp(system_dim * 4 + global_dim, hidden_dim, system_dim, num_layers=mlp_layers)

    @staticmethod
    def _is_fast_pair_mlp(mlp: nn.Sequential) -> bool:
        return (
            len(mlp) == 4
            and isinstance(mlp[0], nn.Linear)
            and isinstance(mlp[1], nn.ReLU)
            and isinstance(mlp[2], nn.Linear)
        )

    @staticmethod
    def _can_sum_before_pair_output(mlp: nn.Sequential) -> bool:
        return _TypedRelationalBlock._is_fast_pair_mlp(mlp) and isinstance(mlp[3], nn.Identity)

    @staticmethod
    def _mlp3_pair(
        mlp: nn.Sequential,
        left: torch.Tensor,
        right: torch.Tensor,
        edge: torch.Tensor,
        *,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        """Evaluate MLP([left, right, edge]), using the fast path for 2-layer MLPs."""
        if not _TypedRelationalBlock._is_fast_pair_mlp(mlp):
            leading = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1], edge.shape[:-1])
            left_e = left.expand(*leading, int(left.shape[-1]))
            right_e = right.expand(*leading, int(right.shape[-1]))
            edge_e = edge.expand(*leading, int(edge.shape[-1]))
            return mlp(torch.cat([left_e, right_e, edge_e], dim=-1))
        lin0 = mlp[0]
        lin1 = mlp[2]
        if not isinstance(lin0, nn.Linear) or not isinstance(lin1, nn.Linear):
            raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
        w = lin0.weight
        left_w = w[:, :left_dim]
        right_w = w[:, left_dim : left_dim + right_dim]
        edge_w = w[:, left_dim + right_dim :]
        hidden = (
            F.linear(left, left_w, None)
            + F.linear(right, right_w, None)
            + F.linear(edge, edge_w, lin0.bias)
        )
        hidden = torch.relu(hidden)
        out = lin1(hidden)
        return mlp[3](out)

    @staticmethod
    def _mlp3_pair_sparse_sum(
        mlp: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
        *,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        """Compute sum_s mask[t,s] * MLP([source_s, target_t, edge[t,s]]).

        This is algebraically equivalent to the dense pair MLP followed by
        `_masked_sum_tokens(..., dim=2)`, but avoids running the pair MLP on
        padded/invisible UAV-SAT edges.
        """
        if mask.ndim != 3:
            raise ValueError(f"sparse pair aggregation expects a [B,T,S] mask, got {tuple(mask.shape)}")
        batch_size, target_count, source_count = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        out_dim = int(mlp[-2].out_features) if isinstance(mlp[-2], nn.Linear) else int(source.shape[-1])
        if batch_size <= 0 or target_count <= 0 or source_count <= 0:
            return target.new_zeros((batch_size, target_count, out_dim))
        valid = torch.nonzero(mask.reshape(-1).to(dtype=torch.bool), as_tuple=False).flatten()
        if int(valid.numel()) <= 0:
            return target.new_zeros((batch_size, target_count, out_dim))
        src_idx = valid.remainder(source_count)
        tmp = torch.div(valid, source_count, rounding_mode="floor")
        tgt_idx = tmp.remainder(target_count)
        batch_idx = torch.div(tmp, target_count, rounding_mode="floor")
        source_v = source[batch_idx, src_idx]
        target_v = target[batch_idx, tgt_idx]
        edge_v = edge.reshape(batch_size * target_count * source_count, int(edge.shape[-1])).index_select(0, valid)
        msg = _TypedRelationalBlock._mlp3_pair(
            mlp,
            source_v,
            target_v,
            edge_v,
            left_dim=left_dim,
            right_dim=right_dim,
        )
        out = target.new_zeros((batch_size * target_count, int(msg.shape[-1])))
        out.index_add_(0, batch_idx * target_count + tgt_idx, msg)
        return out.view(batch_size, target_count, int(msg.shape[-1]))

    @staticmethod
    def _sparse_pair_indices(mask: torch.Tensor) -> tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask.ndim != 3:
            raise ValueError(f"sparse pair aggregation expects a [B,T,S] mask, got {tuple(mask.shape)}")
        batch_size, target_count, source_count = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        valid = torch.nonzero(mask.reshape(-1).to(dtype=torch.bool), as_tuple=False).flatten()
        if int(valid.numel()) <= 0:
            empty = valid
            return batch_size, target_count, source_count, empty, empty, empty, empty
        src_idx = valid.remainder(source_count)
        tmp = torch.div(valid, source_count, rounding_mode="floor")
        tgt_idx = tmp.remainder(target_count)
        batch_idx = torch.div(tmp, target_count, rounding_mode="floor")
        target_flat_idx = batch_idx * target_count + tgt_idx
        return batch_size, target_count, source_count, valid, batch_idx, src_idx, target_flat_idx

    @staticmethod
    def _mlp3_pair_sparse_sum_from_indices(
        mlp: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        edge: torch.Tensor,
        *,
        batch_size: int,
        target_count: int,
        source_count: int,
        valid: torch.Tensor,
        batch_idx: torch.Tensor,
        src_idx: torch.Tensor,
        target_flat_idx: torch.Tensor,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        out_dim = int(mlp[-2].out_features) if isinstance(mlp[-2], nn.Linear) else int(source.shape[-1])
        if batch_size <= 0 or target_count <= 0 or source_count <= 0 or int(valid.numel()) <= 0:
            return target.new_zeros((batch_size, target_count, out_dim))
        tgt_idx = target_flat_idx.remainder(target_count)
        source_v = source[batch_idx, src_idx]
        target_v = target[batch_idx, tgt_idx]
        edge_v = edge.reshape(batch_size * target_count * source_count, int(edge.shape[-1])).index_select(0, valid)
        msg = _TypedRelationalBlock._mlp3_pair(
            mlp,
            source_v,
            target_v,
            edge_v,
            left_dim=left_dim,
            right_dim=right_dim,
        )
        out = target.new_zeros((batch_size * target_count, int(msg.shape[-1])))
        out.index_add_(0, target_flat_idx, msg)
        return out.view(batch_size, target_count, int(msg.shape[-1]))

    @staticmethod
    def _split_mlp3_pair_first_layer(
        mlp: nn.Sequential,
        left_dim: int,
        right_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        lin0 = mlp[0]
        if not isinstance(lin0, nn.Linear):
            raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
        w = lin0.weight
        return (
            w[:, :left_dim],
            w[:, left_dim : left_dim + right_dim],
            w[:, left_dim + right_dim :],
            lin0.bias,
        )

    @staticmethod
    def _finish_fast_pair_mlp(mlp: nn.Sequential, hidden: torch.Tensor) -> torch.Tensor:
        lin1 = mlp[2]
        if not isinstance(lin1, nn.Linear):
            raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
        return mlp[3](lin1(torch.relu(hidden)))

    @staticmethod
    def _finish_fast_pair_mlp_sum(mlp: nn.Sequential, hidden_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        lin1 = mlp[2]
        if not isinstance(lin1, nn.Linear) or not isinstance(mlp[3], nn.Identity):
            raise RuntimeError("pre-output aggregation requires Linear/ReLU/Linear/Identity layout.")
        out = F.linear(hidden_sum, lin1.weight, None)
        if lin1.bias is not None:
            out = out + count.to(dtype=out.dtype) * lin1.bias
        return out

    @staticmethod
    def _mlp3_bidirectional_dense_sums(
        mlp_source_to_target: nn.Sequential,
        mlp_target_to_source: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
        *,
        left_dim: int,
        right_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both directions for a dense [B,target,source] relation.

        The result is exactly:
        - sum_source MLP_st([source, target, edge]) into target
        - sum_target MLP_ts([target, source, edge]) into source
        The fast path reuses the first-layer token projections for both
        directions; fallback keeps the old dense math.
        """
        if (
            not _TypedRelationalBlock._is_fast_pair_mlp(mlp_source_to_target)
            or not _TypedRelationalBlock._is_fast_pair_mlp(mlp_target_to_source)
        ):
            msg_s_to_t = _TypedRelationalBlock._mlp3_pair(
                mlp_source_to_target,
                source.unsqueeze(1),
                target.unsqueeze(2),
                edge,
                left_dim=left_dim,
                right_dim=right_dim,
            )
            msg_t_to_s = _TypedRelationalBlock._mlp3_pair(
                mlp_target_to_source,
                target.unsqueeze(2),
                source.unsqueeze(1),
                edge,
                left_dim=right_dim,
                right_dim=left_dim,
            )
            return (
                _masked_sum_tokens(msg_s_to_t, mask, dim=2),
                _masked_sum_tokens(msg_t_to_s.transpose(1, 2), mask.transpose(1, 2), dim=2),
            )

        s_left_st, t_right_st, edge_w_st, bias_st = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp_source_to_target,
            left_dim,
            right_dim,
        )
        t_left_ts, s_right_ts, edge_w_ts, bias_ts = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp_target_to_source,
            right_dim,
            left_dim,
        )
        source_proj = F.linear(source, torch.cat([s_left_st, s_right_ts], dim=0), None)
        source_st, source_ts = torch.split(source_proj, [s_left_st.shape[0], s_right_ts.shape[0]], dim=-1)
        target_proj = F.linear(target, torch.cat([t_right_st, t_left_ts], dim=0), None)
        target_st, target_ts = torch.split(target_proj, [t_right_st.shape[0], t_left_ts.shape[0]], dim=-1)
        edge_bias = torch.cat([bias_st, bias_ts], dim=0) if bias_st is not None and bias_ts is not None else None
        edge_proj = F.linear(edge, torch.cat([edge_w_st, edge_w_ts], dim=0), edge_bias)
        edge_st, edge_ts = torch.split(edge_proj, [edge_w_st.shape[0], edge_w_ts.shape[0]], dim=-1)

        hidden_st = source_st.unsqueeze(1) + target_st.unsqueeze(2) + edge_st
        hidden_ts = target_ts.unsqueeze(2) + source_ts.unsqueeze(1) + edge_ts
        if (
            _TypedRelationalBlock._can_sum_before_pair_output(mlp_source_to_target)
            and _TypedRelationalBlock._can_sum_before_pair_output(mlp_target_to_source)
        ):
            mask_st = mask.to(dtype=hidden_st.dtype).unsqueeze(-1)
            mask_ts = mask.transpose(1, 2).to(dtype=hidden_ts.dtype).unsqueeze(-1)
            hidden_st_sum = (torch.relu(hidden_st) * mask_st).sum(dim=2)
            hidden_ts_sum = (torch.relu(hidden_ts).transpose(1, 2) * mask_ts).sum(dim=2)
            return (
                _TypedRelationalBlock._finish_fast_pair_mlp_sum(
                    mlp_source_to_target,
                    hidden_st_sum,
                    mask_st.sum(dim=2),
                ),
                _TypedRelationalBlock._finish_fast_pair_mlp_sum(
                    mlp_target_to_source,
                    hidden_ts_sum,
                    mask_ts.sum(dim=2),
                ),
            )
        msg_s_to_t = _TypedRelationalBlock._finish_fast_pair_mlp(mlp_source_to_target, hidden_st)
        msg_t_to_s = _TypedRelationalBlock._finish_fast_pair_mlp(mlp_target_to_source, hidden_ts)
        return (
            _masked_sum_tokens(msg_s_to_t, mask, dim=2),
            _masked_sum_tokens(msg_t_to_s.transpose(1, 2), mask.transpose(1, 2), dim=2),
        )

    @staticmethod
    def _mlp3_bidirectional_sparse_sums_from_indices(
        mlp_source_to_target: nn.Sequential,
        mlp_target_to_source: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        edge: torch.Tensor,
        *,
        batch_size: int,
        target_count: int,
        source_count: int,
        valid: torch.Tensor,
        batch_idx: torch.Tensor,
        src_idx: torch.Tensor,
        target_flat_idx: torch.Tensor,
        left_dim: int,
        right_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not _TypedRelationalBlock._is_fast_pair_mlp(mlp_source_to_target)
            or not _TypedRelationalBlock._is_fast_pair_mlp(mlp_target_to_source)
        ):
            target_to_source_valid = batch_idx * (source_count * target_count) + src_idx * target_count + target_flat_idx.remainder(target_count)
            target_to_source_flat = batch_idx * source_count + src_idx
            return (
                _TypedRelationalBlock._mlp3_pair_sparse_sum_from_indices(
                    mlp_source_to_target,
                    source,
                    target,
                    edge,
                    batch_size=batch_size,
                    target_count=target_count,
                    source_count=source_count,
                    valid=valid,
                    batch_idx=batch_idx,
                    src_idx=src_idx,
                    target_flat_idx=target_flat_idx,
                    left_dim=left_dim,
                    right_dim=right_dim,
                ),
                _TypedRelationalBlock._mlp3_pair_sparse_sum_from_indices(
                    mlp_target_to_source,
                    target,
                    source,
                    edge.transpose(1, 2),
                    batch_size=batch_size,
                    target_count=source_count,
                    source_count=target_count,
                    valid=target_to_source_valid,
                    batch_idx=batch_idx,
                    src_idx=target_flat_idx.remainder(target_count),
                    target_flat_idx=target_to_source_flat,
                    left_dim=right_dim,
                    right_dim=left_dim,
                ),
            )

        out_dim_st = int(mlp_source_to_target[-2].out_features)
        out_dim_ts = int(mlp_target_to_source[-2].out_features)
        if batch_size <= 0 or target_count <= 0 or source_count <= 0 or int(valid.numel()) <= 0:
            return (
                target.new_zeros((batch_size, target_count, out_dim_st)),
                source.new_zeros((batch_size, source_count, out_dim_ts)),
            )

        tgt_idx = target_flat_idx.remainder(target_count)
        s_left_st, t_right_st, edge_w_st, bias_st = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp_source_to_target,
            left_dim,
            right_dim,
        )
        t_left_ts, s_right_ts, edge_w_ts, bias_ts = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp_target_to_source,
            right_dim,
            left_dim,
        )
        source_proj = F.linear(source, torch.cat([s_left_st, s_right_ts], dim=0), None)
        source_st, source_ts = torch.split(source_proj, [s_left_st.shape[0], s_right_ts.shape[0]], dim=-1)
        target_proj = F.linear(target, torch.cat([t_right_st, t_left_ts], dim=0), None)
        target_st, target_ts = torch.split(target_proj, [t_right_st.shape[0], t_left_ts.shape[0]], dim=-1)
        edge_v = edge.reshape(batch_size * target_count * source_count, int(edge.shape[-1])).index_select(0, valid)
        edge_bias = torch.cat([bias_st, bias_ts], dim=0) if bias_st is not None and bias_ts is not None else None
        edge_proj = F.linear(edge_v, torch.cat([edge_w_st, edge_w_ts], dim=0), edge_bias)
        edge_st, edge_ts = torch.split(edge_proj, [edge_w_st.shape[0], edge_w_ts.shape[0]], dim=-1)

        hidden_st = source_st[batch_idx, src_idx] + target_st[batch_idx, tgt_idx] + edge_st
        hidden_ts = target_ts[batch_idx, tgt_idx] + source_ts[batch_idx, src_idx] + edge_ts
        if (
            _TypedRelationalBlock._can_sum_before_pair_output(mlp_source_to_target)
            and _TypedRelationalBlock._can_sum_before_pair_output(mlp_target_to_source)
        ):
            hidden_st = torch.relu(hidden_st)
            hidden_ts = torch.relu(hidden_ts)
            out_h_t = target.new_zeros((batch_size * target_count, int(hidden_st.shape[-1])))
            out_h_s = source.new_zeros((batch_size * source_count, int(hidden_ts.shape[-1])))
            out_h_t.index_add_(0, target_flat_idx, hidden_st)
            out_h_s.index_add_(0, batch_idx * source_count + src_idx, hidden_ts)
            ones = hidden_st.new_ones((int(hidden_st.shape[0]), 1))
            count_t = target.new_zeros((batch_size * target_count, 1))
            count_s = source.new_zeros((batch_size * source_count, 1))
            count_t.index_add_(0, target_flat_idx, ones)
            count_s.index_add_(0, batch_idx * source_count + src_idx, ones)
            return (
                _TypedRelationalBlock._finish_fast_pair_mlp_sum(
                    mlp_source_to_target,
                    out_h_t,
                    count_t,
                ).view(batch_size, target_count, out_dim_st),
                _TypedRelationalBlock._finish_fast_pair_mlp_sum(
                    mlp_target_to_source,
                    out_h_s,
                    count_s,
                ).view(batch_size, source_count, out_dim_ts),
            )
        msg_s_to_t = _TypedRelationalBlock._finish_fast_pair_mlp(mlp_source_to_target, hidden_st)
        msg_t_to_s = _TypedRelationalBlock._finish_fast_pair_mlp(mlp_target_to_source, hidden_ts)
        out_t = target.new_zeros((batch_size * target_count, int(msg_s_to_t.shape[-1])))
        out_s = source.new_zeros((batch_size * source_count, int(msg_t_to_s.shape[-1])))
        out_t.index_add_(0, target_flat_idx, msg_s_to_t)
        out_s.index_add_(0, batch_idx * source_count + src_idx, msg_t_to_s)
        return (
            out_t.view(batch_size, target_count, int(msg_s_to_t.shape[-1])),
            out_s.view(batch_size, source_count, int(msg_t_to_s.shape[-1])),
        )

    @staticmethod
    def _mlp3_self_sparse_sum_from_indices(
        mlp: nn.Sequential,
        tokens: torch.Tensor,
        edge: torch.Tensor,
        *,
        batch_size: int,
        target_count: int,
        source_count: int,
        valid: torch.Tensor,
        batch_idx: torch.Tensor,
        src_idx: torch.Tensor,
        target_flat_idx: torch.Tensor,
        token_dim: int,
    ) -> torch.Tensor:
        if not _TypedRelationalBlock._is_fast_pair_mlp(mlp):
            return _TypedRelationalBlock._mlp3_pair_sparse_sum_from_indices(
                mlp,
                tokens,
                tokens,
                edge,
                batch_size=batch_size,
                target_count=target_count,
                source_count=source_count,
                valid=valid,
                batch_idx=batch_idx,
                src_idx=src_idx,
                target_flat_idx=target_flat_idx,
                left_dim=token_dim,
                right_dim=token_dim,
            )
        out_dim = int(mlp[-2].out_features)
        if batch_size <= 0 or target_count <= 0 or source_count <= 0 or int(valid.numel()) <= 0:
            return tokens.new_zeros((batch_size, target_count, out_dim))
        left_w, right_w, edge_w, bias = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp,
            token_dim,
            token_dim,
        )
        token_proj = F.linear(tokens, torch.cat([left_w, right_w], dim=0), None)
        token_left, token_right = torch.split(token_proj, [left_w.shape[0], right_w.shape[0]], dim=-1)
        tgt_idx = target_flat_idx.remainder(target_count)
        edge_v = edge.reshape(batch_size * target_count * source_count, int(edge.shape[-1])).index_select(0, valid)
        hidden = token_left[batch_idx, src_idx] + token_right[batch_idx, tgt_idx] + F.linear(edge_v, edge_w, bias)
        if _TypedRelationalBlock._can_sum_before_pair_output(mlp):
            hidden = torch.relu(hidden)
            out_h = tokens.new_zeros((batch_size * target_count, int(hidden.shape[-1])))
            out_h.index_add_(0, target_flat_idx, hidden)
            count = tokens.new_zeros((batch_size * target_count, 1))
            count.index_add_(0, target_flat_idx, hidden.new_ones((int(hidden.shape[0]), 1)))
            return _TypedRelationalBlock._finish_fast_pair_mlp_sum(mlp, out_h, count).view(batch_size, target_count, out_dim)
        msg = _TypedRelationalBlock._finish_fast_pair_mlp(mlp, hidden)
        out = tokens.new_zeros((batch_size * target_count, int(msg.shape[-1])))
        out.index_add_(0, target_flat_idx, msg)
        return out.view(batch_size, target_count, int(msg.shape[-1]))

    @staticmethod
    def _mlp3_self_dense_sum(
        mlp: nn.Sequential,
        tokens: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
        *,
        token_dim: int,
    ) -> torch.Tensor:
        if not _TypedRelationalBlock._is_fast_pair_mlp(mlp):
            msg = _TypedRelationalBlock._mlp3_pair(
                mlp,
                tokens.unsqueeze(1),
                tokens.unsqueeze(2),
                edge,
                left_dim=token_dim,
                right_dim=token_dim,
            )
            return _masked_sum_tokens(msg, mask, dim=2)
        out_dim = int(mlp[-2].out_features)
        if int(tokens.shape[1]) <= 0:
            return tokens.new_zeros((int(tokens.shape[0]), int(tokens.shape[1]), out_dim))
        left_w, right_w, edge_w, bias = _TypedRelationalBlock._split_mlp3_pair_first_layer(
            mlp,
            token_dim,
            token_dim,
        )
        token_proj = F.linear(tokens, torch.cat([left_w, right_w], dim=0), None)
        token_left, token_right = torch.split(token_proj, [left_w.shape[0], right_w.shape[0]], dim=-1)
        hidden = token_left.unsqueeze(1) + token_right.unsqueeze(2) + F.linear(edge, edge_w, bias)
        if _TypedRelationalBlock._can_sum_before_pair_output(mlp):
            mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            hidden_sum = (torch.relu(hidden) * mask_f).sum(dim=2)
            return _TypedRelationalBlock._finish_fast_pair_mlp_sum(
                mlp,
                hidden_sum,
                mask_f.sum(dim=2),
            )
        msg = _TypedRelationalBlock._finish_fast_pair_mlp(mlp, hidden)
        return _masked_sum_tokens(msg, mask, dim=2)

    @staticmethod
    def _mlp2_pair_sparse_sum(
        mlp: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"sparse token aggregation expects a [B,N] mask, got {tuple(mask.shape)}")
        batch_size, item_count = (int(mask.shape[0]), int(mask.shape[1]))
        out_dim = int(mlp[-2].out_features) if isinstance(mlp[-2], nn.Linear) else int(target.shape[-1])
        if batch_size <= 0 or item_count <= 0:
            return target.new_zeros((batch_size, out_dim))
        valid = torch.nonzero(mask.reshape(-1).to(dtype=torch.bool), as_tuple=False).flatten()
        if int(valid.numel()) <= 0:
            return target.new_zeros((batch_size, out_dim))
        item_idx = valid.remainder(item_count)
        batch_idx = torch.div(valid, item_count, rounding_mode="floor")
        if _TypedRelationalBlock._can_sum_before_pair_output(mlp):
            lin0 = mlp[0]
            if not isinstance(lin0, nn.Linear):
                raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
            w = lin0.weight
            hidden = (
                F.linear(source[batch_idx, item_idx], w[:, :left_dim], None)
                + F.linear(target[batch_idx], w[:, left_dim : left_dim + right_dim], lin0.bias)
            )
            hidden = torch.relu(hidden)
            hidden_sum = target.new_zeros((batch_size, int(hidden.shape[-1])))
            hidden_sum.index_add_(0, batch_idx, hidden)
            count = target.new_zeros((batch_size, 1))
            count.index_add_(0, batch_idx, hidden.new_ones((int(hidden.shape[0]), 1)))
            return _TypedRelationalBlock._finish_fast_pair_mlp_sum(mlp, hidden_sum, count)
        msg = _TypedRelationalBlock._mlp2_pair(
            mlp,
            source[batch_idx, item_idx],
            target[batch_idx],
            left_dim=left_dim,
            right_dim=right_dim,
        )
        out = target.new_zeros((batch_size, int(msg.shape[-1])))
        out.index_add_(0, batch_idx, msg)
        return out

    @staticmethod
    def _mlp2_pair_dense_sum(
        mlp: nn.Sequential,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        if not _TypedRelationalBlock._can_sum_before_pair_output(mlp):
            msg = _TypedRelationalBlock._mlp2_pair(
                mlp,
                source,
                target.unsqueeze(1),
                left_dim=left_dim,
                right_dim=right_dim,
            )
            if mask is None:
                return msg.sum(dim=1)
            return _masked_sum_tokens(msg, mask, dim=1)
        lin0 = mlp[0]
        if not isinstance(lin0, nn.Linear):
            raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
        w = lin0.weight
        hidden = (
            F.linear(source, w[:, :left_dim], None)
            + F.linear(target, w[:, left_dim : left_dim + right_dim], lin0.bias).unsqueeze(1)
        )
        hidden = torch.relu(hidden)
        if mask is None:
            hidden_sum = hidden.sum(dim=1)
            count = hidden.new_full((int(hidden.shape[0]), 1), float(int(hidden.shape[1])))
        else:
            mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            hidden_sum = (hidden * mask_f).sum(dim=1)
            count = mask_f.sum(dim=1)
        return _TypedRelationalBlock._finish_fast_pair_mlp_sum(mlp, hidden_sum, count)

    @staticmethod
    def _mlp2_pair(
        mlp: nn.Sequential,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        left_dim: int,
        right_dim: int,
    ) -> torch.Tensor:
        """Evaluate MLP([left, right]), using the fast path for 2-layer MLPs."""
        if not _TypedRelationalBlock._is_fast_pair_mlp(mlp):
            leading = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
            left_e = left.expand(*leading, int(left.shape[-1]))
            right_e = right.expand(*leading, int(right.shape[-1]))
            return mlp(torch.cat([left_e, right_e], dim=-1))
        lin0 = mlp[0]
        lin1 = mlp[2]
        if not isinstance(lin0, nn.Linear) or not isinstance(lin1, nn.Linear):
            raise RuntimeError("pair MLP optimization expects _make_mlp's Linear/ReLU/Linear layout.")
        w = lin0.weight
        hidden = (
            F.linear(left, w[:, :left_dim], None)
            + F.linear(right, w[:, left_dim : left_dim + right_dim], lin0.bias)
        )
        hidden = torch.relu(hidden)
        out = lin1(hidden)
        return mlp[3](out)

    @staticmethod
    def _pair_ug(
        uav_tokens: torch.Tensor,
        gu_tokens: torch.Tensor,
        ug_edges: torch.Tensor,
    ) -> torch.Tensor:
        uav = uav_tokens.unsqueeze(2).expand(-1, -1, gu_tokens.shape[1], -1)
        gu = gu_tokens.unsqueeze(1).expand(-1, uav_tokens.shape[1], -1, -1)
        return torch.cat([uav, gu, ug_edges], dim=-1)

    @staticmethod
    def _pair_us(
        uav_tokens: torch.Tensor,
        sat_tokens: torch.Tensor,
        us_edges: torch.Tensor,
    ) -> torch.Tensor:
        uav = uav_tokens.unsqueeze(2).expand(-1, -1, sat_tokens.shape[1], -1)
        sat = sat_tokens.unsqueeze(1).expand(-1, uav_tokens.shape[1], -1, -1)
        return torch.cat([uav, sat, us_edges], dim=-1)

    @staticmethod
    def _pair_uu(
        uav_tokens: torch.Tensor,
        uu_edges: torch.Tensor,
    ) -> torch.Tensor:
        target = uav_tokens.unsqueeze(2).expand(-1, -1, uav_tokens.shape[1], -1)
        source = uav_tokens.unsqueeze(1).expand(-1, uav_tokens.shape[1], -1, -1)
        return torch.cat([source, target, uu_edges], dim=-1)

    def forward(
        self,
        gu_tokens: torch.Tensor,
        uav_tokens: torch.Tensor,
        sat_tokens: torch.Tensor,
        ug_edges: torch.Tensor,
        us_edges: torch.Tensor,
        uu_edges: torch.Tensor,
        gu_mask: torch.Tensor,
        sat_mask: torch.Tensor,
        uav_gu_mask: torch.Tensor,
        uav_sat_mask: torch.Tensor,
        uav_uav_mask: torch.Tensor,
        uav_local_summary: torch.Tensor,
        sat_local_summary: torch.Tensor,
        system_token: torch.Tensor,
        global_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_dim = int(uav_tokens.shape[-1])
        agg_g_to_u, agg_u_to_g = self._mlp3_bidirectional_dense_sums(
            self.ug_to_u,
            self.ug_to_g,
            gu_tokens,
            uav_tokens,
            ug_edges,
            uav_gu_mask,
            left_dim=token_dim,
            right_dim=token_dim,
        )

        if self.fixed_bounded_relations_enabled:
            agg_s_to_u, agg_u_to_s = self._mlp3_bidirectional_dense_sums(
                self.us_to_u,
                self.us_to_s,
                sat_tokens,
                uav_tokens,
                us_edges,
                uav_sat_mask,
                left_dim=token_dim,
                right_dim=token_dim,
            )
        else:
            us_b, us_t, us_s, us_valid, us_batch, us_src, us_target = self._sparse_pair_indices(uav_sat_mask)
            agg_s_to_u, agg_u_to_s = self._mlp3_bidirectional_sparse_sums_from_indices(
                self.us_to_u,
                self.us_to_s,
                sat_tokens,
                uav_tokens,
                us_edges,
                batch_size=us_b,
                target_count=us_t,
                source_count=us_s,
                valid=us_valid,
                batch_idx=us_batch,
                src_idx=us_src,
                target_flat_idx=us_target,
                left_dim=token_dim,
                right_dim=token_dim,
            )

        if self.fixed_bounded_relations_enabled:
            agg_u_to_u = self._mlp3_self_dense_sum(
                self.uu_to_u,
                uav_tokens,
                uu_edges,
                uav_uav_mask,
                token_dim=token_dim,
            )
        else:
            uu_b, uu_t, uu_s, uu_valid, uu_batch, uu_src, uu_target = self._sparse_pair_indices(uav_uav_mask)
            agg_u_to_u = self._mlp3_self_sparse_sum_from_indices(
                self.uu_to_u,
                uav_tokens,
                uu_edges,
                batch_size=uu_b,
                target_count=uu_t,
                source_count=uu_s,
                valid=uu_valid,
                batch_idx=uu_batch,
                src_idx=uu_src,
                target_flat_idx=uu_target,
                token_dim=token_dim,
            )

        gu_tokens = gu_tokens + self.gu_update(torch.cat([gu_tokens, agg_u_to_g], dim=-1))
        uav_tokens = uav_tokens + self.uav_update(
            torch.cat([uav_tokens, agg_g_to_u, agg_s_to_u, agg_u_to_u, uav_local_summary], dim=-1)
        )
        sat_tokens = sat_tokens + self.sat_update(torch.cat([sat_tokens, agg_u_to_s, sat_local_summary], dim=-1))

        system_dim = int(system_token.shape[-1])
        agg_g_sys = self._mlp2_pair_dense_sum(
            self.gu_to_system,
            gu_tokens,
            system_token,
            gu_mask,
            left_dim=token_dim,
            right_dim=system_dim,
        )
        agg_u_sys = self._mlp2_pair_dense_sum(
            self.uav_to_system,
            uav_tokens,
            system_token,
            None,
            left_dim=token_dim,
            right_dim=system_dim,
        )
        if self.fixed_bounded_relations_enabled:
            agg_s_sys = self._mlp2_pair_dense_sum(
                self.sat_to_system,
                sat_tokens,
                system_token,
                sat_mask,
                left_dim=token_dim,
                right_dim=system_dim,
            )
        else:
            agg_s_sys = self._mlp2_pair_sparse_sum(
                self.sat_to_system,
                sat_tokens,
                system_token,
                sat_mask,
                left_dim=token_dim,
                right_dim=system_dim,
            )
        system_token = system_token + self.system_update(
            torch.cat([system_token, agg_g_sys, agg_u_sys, agg_s_sys, global_embed], dim=-1)
        )
        return gu_tokens, uav_tokens, sat_tokens, system_token


class _SatRelationalBlock(nn.Module):
    """SAT-stage value block: only UAV/SAT tokens and UAV-SAT edges interact."""

    def __init__(
        self,
        token_dim: int,
        edge_dim: int,
        system_dim: int,
        global_dim: int,
        hidden_dim: int,
        *,
        message_mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        mlp_layers = max(int(message_mlp_layers), 1)
        self.us_to_u = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.us_to_s = _make_mlp(token_dim * 2 + edge_dim, hidden_dim, token_dim, num_layers=mlp_layers)
        self.us_to_system = _make_mlp(
            token_dim * 2 + edge_dim + system_dim,
            hidden_dim,
            system_dim,
            num_layers=mlp_layers,
        )
        self.uav_update = _make_mlp(token_dim * 3, hidden_dim, token_dim, num_layers=mlp_layers)
        self.sat_update = _make_mlp(token_dim * 3, hidden_dim, token_dim, num_layers=mlp_layers)
        self.uav_to_system = _make_mlp(token_dim + system_dim, hidden_dim, system_dim, num_layers=mlp_layers)
        self.sat_to_system = _make_mlp(token_dim + system_dim, hidden_dim, system_dim, num_layers=mlp_layers)
        self.system_update = _make_mlp(system_dim * 4 + global_dim, hidden_dim, system_dim, num_layers=mlp_layers)

    def forward(
        self,
        uav_tokens: torch.Tensor,
        sat_tokens: torch.Tensor,
        us_edges: torch.Tensor,
        sat_mask: torch.Tensor,
        uav_sat_mask: torch.Tensor,
        system_token: torch.Tensor,
        global_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_dim = int(uav_tokens.shape[-1])
        uav_for_us = uav_tokens.unsqueeze(2)
        sat_for_us = sat_tokens.unsqueeze(1)
        msg_s_to_u = _TypedRelationalBlock._mlp3_pair(
            self.us_to_u,
            sat_for_us,
            uav_for_us,
            us_edges,
            left_dim=token_dim,
            right_dim=token_dim,
        )
        msg_u_to_s = _TypedRelationalBlock._mlp3_pair(
            self.us_to_s,
            uav_for_us,
            sat_for_us,
            us_edges,
            left_dim=token_dim,
            right_dim=token_dim,
        )
        agg_s_to_u = _masked_sum_tokens(msg_s_to_u, uav_sat_mask, dim=2)
        agg_u_to_s = _masked_sum_tokens(msg_u_to_s.transpose(1, 2), uav_sat_mask.transpose(1, 2), dim=2)
        sys_u = system_token.unsqueeze(1).expand(-1, int(uav_tokens.shape[1]), -1)
        sys_s = system_token.unsqueeze(1).expand(-1, int(sat_tokens.shape[1]), -1)
        uav_tokens = uav_tokens + self.uav_update(torch.cat([uav_tokens, agg_s_to_u, sys_u], dim=-1))
        sat_tokens = sat_tokens + self.sat_update(torch.cat([sat_tokens, agg_u_to_s, sys_s], dim=-1))

        sys = system_token.unsqueeze(1)
        system_dim = int(system_token.shape[-1])
        agg_u_sys = _TypedRelationalBlock._mlp2_pair(
            self.uav_to_system,
            uav_tokens,
            sys,
            left_dim=token_dim,
            right_dim=system_dim,
        ).sum(dim=1)
        agg_s_sys = _masked_sum_tokens(
            _TypedRelationalBlock._mlp2_pair(
                self.sat_to_system,
                sat_tokens,
                sys,
                left_dim=token_dim,
                right_dim=system_dim,
            ),
            sat_mask,
            dim=1,
        )
        sys_us = system_token[:, None, None, :].expand(
            -1,
            int(uav_tokens.shape[1]),
            int(sat_tokens.shape[1]),
            -1,
        )
        uav_for_sys = uav_for_us.expand(-1, -1, int(sat_tokens.shape[1]), -1)
        sat_for_sys = sat_for_us.expand(-1, int(uav_tokens.shape[1]), -1, -1)
        us_sys_msg = self.us_to_system(torch.cat([uav_for_sys, sat_for_sys, us_edges, sys_us], dim=-1))
        us_mask_f = uav_sat_mask.to(dtype=us_sys_msg.dtype).unsqueeze(-1)
        us_count = us_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        agg_us_sys = (us_sys_msg * us_mask_f).sum(dim=(1, 2)) / us_count
        system_token = system_token + self.system_update(
            torch.cat([system_token, agg_u_sys, agg_s_sys, agg_us_sys, global_embed], dim=-1)
        )
        return uav_tokens, sat_tokens, system_token


class StructuredCritic(nn.Module):
    def __init__(
        self,
        uav_dim: int,
        gu_dim: int,
        sat_dim: int,
        uav_gu_edge_dim: int,
        uav_sat_edge_dim: int,
        uav_uav_edge_dim: int,
        hidden_dim: int = schema.CRITIC_HIDDEN_DIM,
        embed_dim: int = schema.CRITIC_EMBED_DIM,
        edge_embed_dim: int | None = None,
        global_embed_dim: int | None = None,
        system_token_dim: int | None = None,
        message_layers: int | None = None,
        encoder_mlp_layers: int = 2,
        message_mlp_layers: int = 2,
        fixed_bounded_relations_enabled: bool = False,
        value_head_hidden_dim: int | None = None,
        value_head_layers: int = 2,
        value_mode: str = "relational",
        input_norm_enabled: bool = False,
        global_feature_enabled: bool = False,
        popart_enabled: bool = False,
        popart_beta: float = 0.999,
        popart_min_std: float = 1.0e-4,
        bw_local_ego_dim: int | None = None,
        bw_local_sat_node_dim: int | None = None,
        bw_local_sat_edge_dim: int | None = None,
        bw_local_user_node_dim: int | None = None,
        bw_local_user_edge_dim: int | None = None,
        bw_action_dim: int | None = None,
        stage_specific_paths_enabled: bool = False,
        sat_hidden_dim: int | None = None,
        sat_embed_dim: int | None = None,
        sat_message_layers: int | None = None,
        sat_encoder_mlp_layers: int | None = None,
        sat_message_mlp_layers: int | None = None,
        sat_value_head_hidden_dim: int | None = None,
        sat_value_head_layers: int | None = None,
        flat_num_uav: int | None = None,
        flat_num_gu: int | None = None,
        flat_num_sat: int | None = None,
    ):
        super().__init__()
        expected = {
            "uav_dim": (uav_dim, schema.CRITIC_UAV_NODE_DIM),
            "gu_dim": (gu_dim, schema.CRITIC_GU_NODE_DIM),
            "sat_dim": (sat_dim, schema.CRITIC_SAT_NODE_DIM),
            "uav_gu_edge_dim": (uav_gu_edge_dim, schema.CRITIC_UAV_GU_EDGE_DIM),
            "uav_sat_edge_dim": (uav_sat_edge_dim, schema.CRITIC_UAV_SAT_EDGE_DIM),
            "uav_uav_edge_dim": (uav_uav_edge_dim, schema.CRITIC_UAV_UAV_EDGE_DIM),
        }
        bad = [f"{name}={got} (expected {want})" for name, (got, want) in expected.items() if int(got) != int(want)]
        if bad:
            raise ValueError("StructuredCritic requires the fixed critic schema: " + ", ".join(bad))

        self.embed_dim = int(embed_dim)
        self.edge_embed_dim = int(self.embed_dim if edge_embed_dim is None else edge_embed_dim)
        self.global_embed_dim = int(self.embed_dim if global_embed_dim is None else global_embed_dim)
        self.system_token_dim = int(self.embed_dim if system_token_dim is None else system_token_dim)
        self.message_layers = int(schema.CRITIC_MESSAGE_LAYERS if message_layers is None else message_layers)
        self.value_head_hidden_dim = int(hidden_dim if value_head_hidden_dim is None else value_head_hidden_dim)
        self.encoder_mlp_layers = max(int(encoder_mlp_layers), 1)
        self.message_mlp_layers = max(int(message_mlp_layers), 1)
        self.fixed_bounded_relations_enabled = bool(fixed_bounded_relations_enabled)
        self.value_head_layers = max(int(value_head_layers), 1)
        if self.global_embed_dim != self.system_token_dim:
            raise ValueError("critic_global_embed_dim must equal critic_system_token_dim for additive system-token init.")
        if self.embed_dim != self.edge_embed_dim or self.embed_dim != self.system_token_dim:
            raise ValueError("current StructuredCritic implementation requires token, edge, and system dims to match.")
        value_mode_l = str(value_mode or "relational").strip().lower()
        if value_mode_l in {"full", "graph", "system", "system_token"}:
            value_mode_l = "relational"
        if value_mode_l in {"global", "global_scalars", "global-scalar", "global-only"}:
            value_mode_l = "global_only"
        if value_mode_l in {"global_linear", "global-linear", "linear_global", "linear-global"}:
            value_mode_l = "global_linear"
        if value_mode_l in {"flat", "flat_mlp", "flat-mlp", "flat_global", "flat-global", "shared_mlp", "shared-mlp"}:
            value_mode_l = "flat_mlp"
        if value_mode_l not in {"relational", "global_only", "global_linear", "flat_mlp"}:
            raise ValueError("critic_value_mode must be one of {'relational', 'global_only', 'global_linear', 'flat_mlp'}.")
        self.value_mode = value_mode_l
        del global_feature_enabled
        if self.value_mode == "global_linear" and bool(popart_enabled):
            raise ValueError("critic_popart_enabled is not supported with critic_value_mode='global_linear'.")
        self.popart_enabled = bool(popart_enabled)
        self.popart_beta = float(min(max(float(popart_beta), 0.0), 0.99999))
        self.popart_min_std = float(max(float(popart_min_std), 1.0e-8))
        self.stage_specific_paths_enabled = bool(stage_specific_paths_enabled)
        self.sat_path_enabled = self.stage_specific_paths_enabled and self.value_mode == "relational"
        use_input_norm = bool(input_norm_enabled)
        self.flat_num_uav = int(flat_num_uav or 0)
        self.flat_num_gu = int(flat_num_gu or 0)
        self.flat_num_sat = int(flat_num_sat or 0)
        if self.value_mode == "flat_mlp":
            if self.flat_num_uav <= 0 or self.flat_num_gu < 0 or self.flat_num_sat <= 0:
                raise ValueError(
                    "critic_value_mode='flat_mlp' requires flat_num_uav > 0, "
                    "flat_num_gu >= 0, and flat_num_sat > 0."
                )
            flat_input_dim = (
                self.flat_num_uav * int(uav_dim)
                + self.flat_num_gu * int(gu_dim)
                + self.flat_num_sat * int(sat_dim)
                + self.flat_num_sat
                + self.flat_num_uav * self.flat_num_gu * int(uav_gu_edge_dim)
                + self.flat_num_uav * self.flat_num_sat * int(uav_sat_edge_dim)
                + self.flat_num_uav * self.flat_num_uav * int(uav_uav_edge_dim)
                + schema.CRITIC_GLOBAL_SCALAR_DIM
                + self.flat_num_gu
                + self.flat_num_sat
                + self.flat_num_uav * self.flat_num_gu
                + self.flat_num_uav * self.flat_num_sat
                + self.flat_num_uav * self.flat_num_uav
            )
            self.flat_input_norm = _make_input_norm(flat_input_dim, use_input_norm)
            self.flat_context_encoder = _make_mlp(
                flat_input_dim,
                hidden_dim,
                self.system_token_dim,
                num_layers=self.encoder_mlp_layers,
            )

        self.uav_input_norm = _make_input_norm(uav_dim, use_input_norm)
        self.gu_input_norm = _make_input_norm(gu_dim, use_input_norm)
        self.sat_input_norm = _make_input_norm(sat_dim, use_input_norm)
        self.ug_edge_input_norm = _make_input_norm(uav_gu_edge_dim, use_input_norm)
        self.us_edge_input_norm = _make_input_norm(uav_sat_edge_dim, use_input_norm)
        self.uu_edge_input_norm = _make_input_norm(uav_uav_edge_dim, use_input_norm)
        self.global_input_norm = _make_input_norm(schema.CRITIC_GLOBAL_SCALAR_DIM, use_input_norm)
        self.uav_encoder = _make_mlp(uav_dim, hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.gu_encoder = _make_mlp(gu_dim, hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.sat_encoder = _make_mlp(sat_dim, hidden_dim, self.embed_dim, num_layers=self.encoder_mlp_layers)
        self.ug_edge_encoder = _make_mlp(uav_gu_edge_dim, hidden_dim, self.edge_embed_dim, num_layers=self.encoder_mlp_layers)
        self.us_edge_encoder = _make_mlp(uav_sat_edge_dim, hidden_dim, self.edge_embed_dim, num_layers=self.encoder_mlp_layers)
        self.uu_edge_encoder = _make_mlp(uav_uav_edge_dim, hidden_dim, self.edge_embed_dim, num_layers=self.encoder_mlp_layers)
        self.global_scalar_encoder = _make_mlp(
            schema.CRITIC_GLOBAL_SCALAR_DIM,
            hidden_dim,
            self.global_embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.register_buffer(
            "global_linear_coeff",
            torch.zeros((3, schema.CRITIC_GLOBAL_SCALAR_DIM + 1), dtype=torch.float32),
        )
        self.register_buffer(
            "global_linear_xtx",
            torch.zeros((3, schema.CRITIC_GLOBAL_SCALAR_DIM + 1, schema.CRITIC_GLOBAL_SCALAR_DIM + 1), dtype=torch.float64),
        )
        self.register_buffer(
            "global_linear_xty",
            torch.zeros((3, schema.CRITIC_GLOBAL_SCALAR_DIM + 1), dtype=torch.float64),
        )
        self.register_buffer("global_linear_initialized", torch.zeros(3, dtype=torch.float32))
        self.uav_local_summary_encoder = _make_mlp(
            schema.CRITIC_UAV_LOCAL_SUM_DIM,
            hidden_dim,
            self.embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.sat_local_summary_encoder = _make_mlp(
            schema.CRITIC_SAT_LOCAL_SUM_DIM,
            hidden_dim,
            self.embed_dim,
            num_layers=self.encoder_mlp_layers,
        )
        self.system_token = nn.Parameter(torch.zeros(self.system_token_dim, dtype=torch.float32))
        self.relational_blocks = nn.ModuleList(
            [
                _TypedRelationalBlock(
                    token_dim=self.embed_dim,
                    edge_dim=self.edge_embed_dim,
                    system_dim=self.system_token_dim,
                    global_dim=self.global_embed_dim,
                    hidden_dim=hidden_dim,
                    message_mlp_layers=self.message_mlp_layers,
                    fixed_bounded_relations_enabled=self.fixed_bounded_relations_enabled,
                )
                for _ in range(max(self.message_layers, 0))
            ]
        )
        self.value_accel_head = _make_mlp(
            self.system_token_dim,
            self.value_head_hidden_dim,
            1,
            num_layers=self.value_head_layers,
            final_activation=False,
        )
        self.value_sat_head = _make_mlp(
            self.system_token_dim,
            self.value_head_hidden_dim,
            1,
            num_layers=self.value_head_layers,
            final_activation=False,
        )
        self.value_bw_head = _make_mlp(
            self.system_token_dim,
            self.value_head_hidden_dim,
            1,
            num_layers=self.value_head_layers,
            final_activation=False,
        )
        if self.sat_path_enabled:
            self.sat_path_hidden_dim = int(hidden_dim if sat_hidden_dim is None else sat_hidden_dim)
            self.sat_path_embed_dim = int(self.embed_dim if sat_embed_dim is None else sat_embed_dim)
            self.sat_path_message_layers = int(self.message_layers if sat_message_layers is None else sat_message_layers)
            self.sat_path_encoder_layers = max(
                int(self.encoder_mlp_layers if sat_encoder_mlp_layers is None else sat_encoder_mlp_layers),
                1,
            )
            self.sat_path_message_mlp_layers = max(
                int(self.message_mlp_layers if sat_message_mlp_layers is None else sat_message_mlp_layers),
                1,
            )
            self.sat_path_value_head_hidden = int(
                self.value_head_hidden_dim if sat_value_head_hidden_dim is None else sat_value_head_hidden_dim
            )
            self.sat_path_value_head_layers = max(
                int(self.value_head_layers if sat_value_head_layers is None else sat_value_head_layers),
                1,
            )
            self.sat_path_uav_input_norm = _make_input_norm(uav_dim, use_input_norm)
            self.sat_path_sat_input_norm = _make_input_norm(sat_dim, use_input_norm)
            self.sat_path_us_edge_input_norm = _make_input_norm(uav_sat_edge_dim, use_input_norm)
            self.sat_path_global_input_norm = _make_input_norm(schema.CRITIC_GLOBAL_SCALAR_DIM, use_input_norm)
            self.sat_path_summary_dim = 30
            self.sat_path_uav_encoder = _make_mlp(
                uav_dim,
                self.sat_path_hidden_dim,
                self.sat_path_embed_dim,
                num_layers=self.sat_path_encoder_layers,
            )
            self.sat_path_sat_encoder = _make_mlp(
                sat_dim,
                self.sat_path_hidden_dim,
                self.sat_path_embed_dim,
                num_layers=self.sat_path_encoder_layers,
            )
            self.sat_path_us_edge_encoder = _make_mlp(
                uav_sat_edge_dim,
                self.sat_path_hidden_dim,
                self.sat_path_embed_dim,
                num_layers=self.sat_path_encoder_layers,
            )
            self.sat_path_global_encoder = _make_mlp(
                schema.CRITIC_GLOBAL_SCALAR_DIM,
                self.sat_path_hidden_dim,
                self.sat_path_embed_dim,
                num_layers=self.sat_path_encoder_layers,
            )
            self.sat_path_summary_encoder = _make_mlp(
                self.sat_path_summary_dim,
                self.sat_path_hidden_dim,
                self.sat_path_embed_dim,
                num_layers=self.sat_path_encoder_layers,
            )
            self.sat_path_system_token = nn.Parameter(torch.zeros(self.sat_path_embed_dim, dtype=torch.float32))
            self.sat_path_blocks = nn.ModuleList(
                [
                    _SatRelationalBlock(
                        token_dim=self.sat_path_embed_dim,
                        edge_dim=self.sat_path_embed_dim,
                        system_dim=self.sat_path_embed_dim,
                        global_dim=self.sat_path_embed_dim,
                        hidden_dim=self.sat_path_hidden_dim,
                        message_mlp_layers=self.sat_path_message_mlp_layers,
                    )
                    for _ in range(max(self.sat_path_message_layers, 0))
                ]
            )
            self.sat_path_value_head = _make_mlp(
                self.sat_path_embed_dim,
                self.sat_path_value_head_hidden,
                1,
                num_layers=self.sat_path_value_head_layers,
                final_activation=False,
            )
        self.bw_delta_available = all(
            value is not None
            for value in (
                bw_local_ego_dim,
                bw_local_sat_node_dim,
                bw_local_sat_edge_dim,
                bw_local_user_node_dim,
                bw_local_user_edge_dim,
                bw_action_dim,
            )
        )
        if self.bw_delta_available:
            action_slot_dim = 4
            sat_context_dim = int(bw_local_sat_node_dim) + int(bw_local_sat_edge_dim or 0)
            user_pair_dim = int(bw_local_user_node_dim) + int(bw_local_user_edge_dim or 0) + action_slot_dim
            self.bw_delta_ego_input_norm = _make_input_norm(int(bw_local_ego_dim), use_input_norm)
            self.bw_delta_sat_input_norm = _make_input_norm(sat_context_dim, use_input_norm)
            self.bw_delta_user_input_norm = _make_input_norm(user_pair_dim, use_input_norm)
            self.bw_delta_ego_encoder = _make_mlp(
                int(bw_local_ego_dim),
                hidden_dim,
                embed_dim,
                num_layers=self.encoder_mlp_layers,
            )
            self.bw_delta_sat_encoder = _make_mlp(
                sat_context_dim,
                hidden_dim,
                embed_dim,
                num_layers=self.encoder_mlp_layers,
            )
            self.bw_delta_user_encoder = _make_mlp(
                user_pair_dim,
                hidden_dim,
                embed_dim,
                num_layers=self.encoder_mlp_layers,
            )
            self.bw_delta_agent_fusion = _make_mlp(
                embed_dim * 5 + 4,
                hidden_dim,
                hidden_dim,
                num_layers=self.message_mlp_layers,
            )
            self.bw_delta_team_fusion = _make_mlp(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_layers=self.message_mlp_layers,
            )
            self.bw_delta_head = nn.Linear(hidden_dim, 1)
        if self.popart_enabled:
            self.register_buffer("popart_mean", torch.zeros(3, dtype=torch.float32))
            self.register_buffer("popart_var", torch.ones(3, dtype=torch.float32))
            self.register_buffer("popart_initialized", torch.zeros(3, dtype=torch.float32))

    @staticmethod
    def _stage_idx(stage: str | int) -> int:
        if isinstance(stage, str):
            if stage == "accel":
                return 0
            if stage == "sat":
                return 1
            if stage == "bw":
                return 2
            raise ValueError(f"Unsupported stage: {stage}")
        stage_idx = int(stage)
        if stage_idx not in {0, 1, 2}:
            raise ValueError(f"Unsupported stage_id={stage_idx}")
        return stage_idx

    def _stage_head_module_by_idx(self, stage_idx: int) -> nn.Module:
        if stage_idx == 0:
            return self.value_accel_head
        if stage_idx == 1:
            if self.sat_path_enabled:
                return self.sat_path_value_head
            return self.value_sat_head
        if stage_idx == 2:
            return self.value_bw_head
        raise ValueError(f"Unsupported stage_id={stage_idx}")

    def _stage_head_by_idx(self, stage_idx: int) -> nn.Linear:
        head = self._stage_head_module_by_idx(stage_idx)
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential):
            for layer in reversed(head):
                if isinstance(layer, nn.Linear):
                    return layer
        raise RuntimeError("stage value head must end with a Linear layer for PopArt rescaling.")

    def popart_stats(self, stage: str | int, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.popart_enabled:
            mean = torch.as_tensor(0.0, dtype=dtype or torch.float32, device=device)
            std = torch.as_tensor(1.0, dtype=dtype or torch.float32, device=device)
            return mean, std
        stage_idx = self._stage_idx(stage)
        mean = self.popart_mean[stage_idx]
        std = self.popart_var[stage_idx].clamp_min(self.popart_min_std ** 2).sqrt()
        if dtype is not None or device is not None:
            mean = mean.to(dtype=dtype or mean.dtype, device=device or mean.device)
            std = std.to(dtype=dtype or std.dtype, device=device or std.device)
        return mean, std

    def popart_normalize(self, stage: str | int, value: torch.Tensor) -> torch.Tensor:
        if not self.popart_enabled:
            return value
        mean, std = self.popart_stats(stage, dtype=value.dtype, device=value.device)
        return (value - mean) / std.clamp_min(self.popart_min_std)

    @torch.no_grad()
    def update_popart_stats(self, stage: str | int, targets: torch.Tensor) -> None:
        if not self.popart_enabled or targets.numel() <= 0:
            return
        stage_idx = self._stage_idx(stage)
        target = targets.detach().to(dtype=self.popart_mean.dtype, device=self.popart_mean.device).reshape(-1)
        batch_mean = target.mean()
        batch_var = target.var(unbiased=False).clamp_min(self.popart_min_std ** 2)
        old_mean = self.popart_mean[stage_idx].clone()
        old_var = self.popart_var[stage_idx].clone().clamp_min(self.popart_min_std ** 2)
        old_std = old_var.sqrt().clamp_min(self.popart_min_std)
        if float(self.popart_initialized[stage_idx].item()) <= 0.5:
            new_mean = batch_mean
            new_var = batch_var
            self.popart_initialized[stage_idx] = 1.0
        else:
            beta = torch.as_tensor(self.popart_beta, dtype=target.dtype, device=target.device)
            batch_second_moment = batch_var + batch_mean.square()
            old_second_moment = old_var + old_mean.square()
            new_mean = beta * old_mean + (1.0 - beta) * batch_mean
            new_second_moment = beta * old_second_moment + (1.0 - beta) * batch_second_moment
            new_var = (new_second_moment - new_mean.square()).clamp_min(self.popart_min_std ** 2)
        new_std = new_var.sqrt().clamp_min(self.popart_min_std)
        head = self._stage_head_by_idx(stage_idx)
        scale = (old_std / new_std).to(dtype=head.weight.dtype, device=head.weight.device)
        old_mean_h = old_mean.to(dtype=head.bias.dtype, device=head.bias.device)
        old_std_h = old_std.to(dtype=head.bias.dtype, device=head.bias.device)
        new_mean_h = new_mean.to(dtype=head.bias.dtype, device=head.bias.device)
        new_std_h = new_std.to(dtype=head.bias.dtype, device=head.bias.device)
        head.weight.mul_(scale)
        head.bias.copy_(((old_std_h * head.bias + old_mean_h - new_mean_h) / new_std_h).reshape_as(head.bias))
        self.popart_mean[stage_idx] = new_mean
        self.popart_var[stage_idx] = new_var

    def _local_summaries(self, world_state: StructuredWorldState) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = world_state.uav_nodes.dtype
        ug = world_state.uav_gu_edges
        us = world_state.uav_sat_edges
        gu = world_state.gu_nodes
        uav = world_state.uav_nodes
        prefix_gu_mask = (
            ug[..., schema.UG_PREFIX_BW_VALID_FLAG]
            * ug[..., schema.UG_PREFIX_BW_VALID_KNOWN]
            * world_state.uav_gu_mask.to(dtype)
        )
        last_bw_weight = ug[..., schema.UG_LAST_SERVED_FLAG] * world_state.uav_gu_mask.to(dtype)
        gu_queue = gu[..., schema.GU_QUEUE_STEPS].unsqueeze(1)
        gu_expected = gu[..., schema.GU_EXPECTED_ARRIVAL_STEPS].unsqueeze(1)
        gu_last_outflow = gu[..., schema.GU_LAST_OUTFLOW_STEPS].unsqueeze(1)
        gu_last_drop = gu[..., schema.GU_LAST_DROP_STEPS].unsqueeze(1)
        uav_local = torch.stack(
            [
                (prefix_gu_mask * gu_queue).sum(dim=-1),
                (prefix_gu_mask * gu_expected).sum(dim=-1),
                (prefix_gu_mask * gu_last_outflow).sum(dim=-1),
                (prefix_gu_mask * gu_last_drop).sum(dim=-1),
                (last_bw_weight * gu_queue).sum(dim=-1),
                (last_bw_weight * gu_expected).sum(dim=-1),
                (last_bw_weight * gu_last_outflow).sum(dim=-1),
                (last_bw_weight * gu_last_drop).sum(dim=-1),
            ],
            dim=-1,
        )

        prefix_sat_mask = (
            us[..., schema.US_PREFIX_SELECTED_FLAG]
            * us[..., schema.US_PREFIX_SELECTED_KNOWN]
            * world_state.uav_sat_mask.to(dtype)
        )
        last_sat_mask = us[..., schema.US_LAST_SELECTED_FLAG] * world_state.uav_sat_mask.to(dtype)
        uav_queue = uav[..., schema.UAV_QUEUE_STEPS].unsqueeze(2)
        uav_last_inflow = uav[..., schema.UAV_LAST_INFLOW_STEPS].unsqueeze(2)
        uav_last_outflow = uav[..., schema.UAV_LAST_OUTFLOW_STEPS].unsqueeze(2)
        uav_last_drop = uav[..., schema.UAV_LAST_DROP_STEPS].unsqueeze(2)
        prefix_cap = us[..., schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS] * prefix_sat_mask
        sat_local = torch.stack(
            [
                (prefix_sat_mask * uav_queue).sum(dim=1),
                (prefix_sat_mask * uav_last_inflow).sum(dim=1),
                (prefix_sat_mask * uav_last_outflow).sum(dim=1),
                (prefix_sat_mask * uav_last_drop).sum(dim=1),
                prefix_cap.sum(dim=1),
                (last_sat_mask * uav_queue).sum(dim=1),
                (last_sat_mask * uav_last_outflow).sum(dim=1),
                (last_sat_mask * uav_last_drop).sum(dim=1),
            ],
            dim=-1,
        )
        return uav_local, sat_local

    @staticmethod
    def _summary_sum_mean_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_b = mask.to(dtype=torch.bool)
        mask_f = mask_b.to(dtype=values.dtype).unsqueeze(-1)
        raw_count = mask_f.sum(dim=1).clamp_min(1.0)
        total = (values * mask_f).sum(dim=1)
        mean = total / raw_count
        masked = torch.where(mask_b.unsqueeze(-1), values, torch.full_like(values, -1.0e9))
        max_value = masked.amax(dim=1)
        has_valid = mask_b.any(dim=1, keepdim=True)
        max_value = torch.where(has_valid, max_value, torch.zeros_like(max_value))
        return torch.cat([total, mean, max_value], dim=-1)

    def _sat_summary_features(self, world_state: StructuredWorldState) -> torch.Tensor:
        batch_size = int(world_state.uav_nodes.shape[0])
        uav_mask = torch.ones(
            (batch_size, int(world_state.uav_nodes.shape[1])),
            dtype=torch.bool,
            device=world_state.uav_nodes.device,
        )
        uav_summary_source = torch.stack(
            [
                world_state.uav_nodes[..., schema.UAV_QUEUE_STEPS],
                world_state.uav_nodes[..., schema.UAV_LAST_INFLOW_STEPS],
                world_state.uav_nodes[..., schema.UAV_LAST_OUTFLOW_STEPS],
                world_state.uav_nodes[..., schema.UAV_LAST_DROP_STEPS],
            ],
            dim=-1,
        )
        sat_summary_source = torch.stack(
            [
                world_state.sat_nodes[..., schema.SAT_QUEUE_STEPS],
                world_state.sat_nodes[..., schema.SAT_LAST_INCOMING_STEPS],
                world_state.sat_nodes[..., schema.SAT_LAST_PROCESSED_STEPS],
                world_state.sat_nodes[..., schema.SAT_LAST_DROP_STEPS],
                world_state.sat_nodes[..., schema.SAT_LAST_SELECTED_LOAD_FRAC],
                world_state.sat_nodes[..., schema.SAT_PROC_CAPACITY_STEPS],
            ],
            dim=-1,
        )
        return torch.cat(
            [
                self._summary_sum_mean_max(uav_summary_source, uav_mask),
                self._summary_sum_mean_max(sat_summary_source, world_state.sat_mask),
            ],
            dim=-1,
        )

    def _sat_system_context(self, world_state: StructuredWorldState) -> torch.Tensor:
        global_embed = self.sat_path_global_encoder(
            self.sat_path_global_input_norm(world_state.global_scalars)
        )
        summary_embed = self.sat_path_summary_encoder(self._sat_summary_features(world_state))
        uav_tokens = self.sat_path_uav_encoder(self.sat_path_uav_input_norm(world_state.uav_nodes))
        sat_tokens = self.sat_path_sat_encoder(self.sat_path_sat_input_norm(world_state.sat_nodes))
        us_edges = self.sat_path_us_edge_encoder(self.sat_path_us_edge_input_norm(world_state.uav_sat_edges))
        sat_mask = world_state.sat_mask.to(dtype=torch.bool)
        uav_sat_mask = world_state.uav_sat_mask.to(dtype=torch.bool)
        batch_size = int(world_state.uav_nodes.shape[0])
        system_token = self.sat_path_system_token.view(1, -1).expand(batch_size, -1) + global_embed + summary_embed
        for block in self.sat_path_blocks:
            uav_tokens, sat_tokens, system_token = block(
                uav_tokens,
                sat_tokens,
                us_edges,
                sat_mask,
                uav_sat_mask,
                system_token,
                global_embed,
            )
        return system_token

    def _system_context(self, world_state: StructuredWorldState) -> torch.Tensor:
        global_embed = self.global_scalar_encoder(self.global_input_norm(world_state.global_scalars))
        if self.value_mode == "global_only":
            return global_embed
        if self.value_mode == "flat_mlp":
            return self.flat_context_encoder(self.flat_input_norm(self._flat_world_features(world_state)))

        gu_mask = world_state.gu_mask.to(dtype=torch.bool)
        sat_mask = world_state.sat_mask.to(dtype=torch.bool)
        uav_gu_mask = world_state.uav_gu_mask.to(dtype=torch.bool)
        uav_sat_mask = world_state.uav_sat_mask.to(dtype=torch.bool)
        uav_uav_mask = world_state.uav_uav_mask.to(dtype=torch.bool)
        gu_tokens = self.gu_encoder(self.gu_input_norm(world_state.gu_nodes))
        uav_tokens = self.uav_encoder(self.uav_input_norm(world_state.uav_nodes))
        sat_tokens = self.sat_encoder(self.sat_input_norm(world_state.sat_nodes))
        ug_edges = self.ug_edge_encoder(self.ug_edge_input_norm(world_state.uav_gu_edges))
        us_edges = self.us_edge_encoder(self.us_edge_input_norm(world_state.uav_sat_edges))
        uu_edges = self.uu_edge_encoder(self.uu_edge_input_norm(world_state.uav_uav_edges))
        uav_local_summary, sat_local_summary = self._local_summaries(world_state)
        uav_local = self.uav_local_summary_encoder(uav_local_summary)
        sat_local = self.sat_local_summary_encoder(sat_local_summary)
        batch_size = int(world_state.uav_nodes.shape[0])
        system_token = self.system_token.view(1, -1).expand(batch_size, -1) + global_embed
        for block in self.relational_blocks:
            gu_tokens, uav_tokens, sat_tokens, system_token = block(
                gu_tokens,
                uav_tokens,
                sat_tokens,
                ug_edges,
                us_edges,
                uu_edges,
                gu_mask,
                sat_mask,
                uav_gu_mask,
                uav_sat_mask,
                uav_uav_mask,
                uav_local,
                sat_local,
                system_token,
                global_embed,
            )
        return system_token

    def _flat_world_features(self, world_state: StructuredWorldState) -> torch.Tensor:
        batch_size = int(world_state.global_scalars.shape[0])

        def _expect_shape(value: torch.Tensor, tail: tuple[int, ...], name: str) -> torch.Tensor:
            if tuple(value.shape[1:]) != tuple(tail):
                raise ValueError(
                    f"flat_mlp critic expected {name} shape [B, {tail}], got {tuple(value.shape)}."
                )
            return value

        def _flat(value: torch.Tensor, tail: tuple[int, ...], name: str) -> torch.Tensor:
            tensor = _expect_shape(value, tail, name).to(dtype=world_state.global_scalars.dtype)
            return tensor.reshape(batch_size, -1)

        features = [
            _flat(world_state.uav_nodes, (self.flat_num_uav, schema.CRITIC_UAV_NODE_DIM), "uav_nodes"),
            _flat(world_state.gu_nodes, (self.flat_num_gu, schema.CRITIC_GU_NODE_DIM), "gu_nodes"),
            _flat(world_state.sat_nodes, (self.flat_num_sat, schema.CRITIC_SAT_NODE_DIM), "sat_nodes"),
            _flat(world_state.sat_ids.unsqueeze(-1), (self.flat_num_sat, 1), "sat_ids"),
            _flat(
                world_state.uav_gu_edges,
                (self.flat_num_uav, self.flat_num_gu, schema.CRITIC_UAV_GU_EDGE_DIM),
                "uav_gu_edges",
            ),
            _flat(
                world_state.uav_sat_edges,
                (self.flat_num_uav, self.flat_num_sat, schema.CRITIC_UAV_SAT_EDGE_DIM),
                "uav_sat_edges",
            ),
            _flat(
                world_state.uav_uav_edges,
                (self.flat_num_uav, self.flat_num_uav, schema.CRITIC_UAV_UAV_EDGE_DIM),
                "uav_uav_edges",
            ),
            _flat(world_state.global_scalars, (schema.CRITIC_GLOBAL_SCALAR_DIM,), "global_scalars"),
            _flat(world_state.gu_mask, (self.flat_num_gu,), "gu_mask"),
            _flat(world_state.sat_mask, (self.flat_num_sat,), "sat_mask"),
            _flat(world_state.uav_gu_mask, (self.flat_num_uav, self.flat_num_gu), "uav_gu_mask"),
            _flat(world_state.uav_sat_mask, (self.flat_num_uav, self.flat_num_sat), "uav_sat_mask"),
            _flat(world_state.uav_uav_mask, (self.flat_num_uav, self.flat_num_uav), "uav_uav_mask"),
        ]
        return torch.cat(features, dim=-1)

    def _global_linear_value(self, world_state: StructuredWorldState, stage_idx: int) -> torch.Tensor:
        features = world_state.global_scalars
        coeff = self.global_linear_coeff[int(stage_idx)].to(device=features.device, dtype=features.dtype)
        bias = coeff[0]
        weight = coeff[1:]
        return features.matmul(weight) + bias

    @torch.no_grad()
    def fit_global_linear(
        self,
        stage: str | int,
        world_state: StructuredWorldState,
        targets: torch.Tensor,
        *,
        ridge: float = 1.0e-6,
        decay: float = 0.90,
    ) -> dict[str, float]:
        stage_idx = self._stage_idx(stage)
        features = world_state.global_scalars.detach().to(dtype=torch.float64)
        target = targets.detach().to(device=features.device, dtype=torch.float64).reshape(-1)
        if features.ndim != 2 or int(features.shape[-1]) != schema.CRITIC_GLOBAL_SCALAR_DIM:
            raise ValueError(
                "fit_global_linear expects world_state.global_scalars with shape "
                f"[B, {schema.CRITIC_GLOBAL_SCALAR_DIM}], got {tuple(features.shape)}"
            )
        if int(features.shape[0]) != int(target.numel()):
            raise ValueError(
                f"fit_global_linear batch mismatch: features={int(features.shape[0])}, targets={int(target.numel())}"
            )
        finite = torch.isfinite(features).all(dim=1) & torch.isfinite(target)
        if int(finite.sum().item()) <= 0:
            self.global_linear_coeff[int(stage_idx)].zero_()
            self.global_linear_initialized[int(stage_idx)] = 0.0
            return {"sample_count": 0.0, "mse": 0.0, "ev": 0.0}
        x = features.index_select(0, torch.nonzero(finite, as_tuple=False).flatten())
        y = target.index_select(0, torch.nonzero(finite, as_tuple=False).flatten())
        ones = torch.ones((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
        design = torch.cat([ones, x], dim=1)
        batch_xtx = design.transpose(0, 1).matmul(design)
        batch_xty = design.transpose(0, 1).matmul(y)
        decay_f = min(max(float(decay), 0.0), 0.999999)
        stage_i = int(stage_idx)
        if float(self.global_linear_initialized[stage_i].item()) > 0.5 and decay_f > 0.0:
            old_xtx = self.global_linear_xtx[stage_i].to(device=batch_xtx.device, dtype=batch_xtx.dtype)
            old_xty = self.global_linear_xty[stage_i].to(device=batch_xty.device, dtype=batch_xty.dtype)
            xtx = decay_f * old_xtx + batch_xtx
            xty = decay_f * old_xty + batch_xty
        else:
            xtx = batch_xtx
            xty = batch_xty
        reg_strength = max(float(ridge), 0.0)
        gram = xtx
        if reg_strength > 0.0:
            reg = torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device) * reg_strength
            reg[0, 0] = 0.0
            gram = gram + reg
        try:
            coef = torch.linalg.solve(gram, xty)
        except RuntimeError:
            coef = torch.linalg.lstsq(gram, xty).solution
        pred = design.matmul(coef)
        err = y - pred
        mse = float(err.pow(2).mean().item())
        var_y = float(y.var(unbiased=False).item())
        ev = 0.0 if var_y <= 1.0e-12 else float(1.0 - float(err.var(unbiased=False).item()) / var_y)
        self.global_linear_coeff[stage_i].copy_(coef.to(device=self.global_linear_coeff.device, dtype=self.global_linear_coeff.dtype))
        self.global_linear_xtx[stage_i].copy_(xtx.to(device=self.global_linear_xtx.device, dtype=self.global_linear_xtx.dtype))
        self.global_linear_xty[stage_i].copy_(xty.to(device=self.global_linear_xty.device, dtype=self.global_linear_xty.dtype))
        self.global_linear_initialized[stage_i] = 1.0
        return {"sample_count": float(y.numel()), "mse": mse, "ev": ev}

    def _value(self, world_state: StructuredWorldState, stage: str) -> torch.Tensor:
        stage_idx = self._stage_idx(stage)
        if self.value_mode == "global_linear":
            return self._global_linear_value(world_state, stage_idx)
        value_head = self._stage_head_module_by_idx(stage_idx)
        context = self._sat_system_context(world_state) if stage_idx == 1 and self.sat_path_enabled else self._system_context(world_state)
        normalized_value = value_head(context).squeeze(-1)
        if not self.popart_enabled:
            return normalized_value
        mean, std = self.popart_stats(stage_idx, dtype=normalized_value.dtype, device=normalized_value.device)
        return normalized_value * std + mean

    def value_accel(self, world_state: StructuredWorldState) -> torch.Tensor:
        return self._value(world_state, stage="accel")

    def value_sat(self, world_state: StructuredWorldState) -> torch.Tensor:
        return self._value(world_state, stage="sat")

    def value_bw(self, world_state: StructuredWorldState) -> torch.Tensor:
        return self._value(world_state, stage="bw")

    def delta_bw(
        self,
        local_state: LocalBwState,
        sampled_action: torch.Tensor,
        ref_action: torch.Tensor,
        *,
        num_agents: int = 1,
    ) -> torch.Tensor:
        if not self.bw_delta_available:
            raise RuntimeError("BW delta critic head is unavailable because local BW dimensions were not provided.")
        if int(num_agents) <= 0:
            raise ValueError("num_agents must be positive for BW delta critic evaluation.")
        sampled_action = sampled_action.to(device=local_state.ego_features.device, dtype=local_state.ego_features.dtype)
        ref_action = ref_action.to(device=sampled_action.device, dtype=sampled_action.dtype)
        row_count = int(sampled_action.shape[0])
        if row_count != int(ref_action.shape[0]) or row_count != int(local_state.ego_features.shape[0]):
            raise ValueError("BW delta critic expects local_state, sampled_action, and ref_action to share batch size.")
        if row_count % int(num_agents) != 0:
            raise ValueError("BW delta critic row count must be divisible by num_agents.")
        sample_count = row_count // int(num_agents)

        user_mask = (local_state.gu_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
        sat_mask = local_state.selected_sat_mask > 0.5
        action_delta = sampled_action - ref_action
        abs_action_delta = action_delta.abs()

        ego_token = self.bw_delta_ego_encoder(self.bw_delta_ego_input_norm(local_state.ego_features))
        sat_token = self.bw_delta_sat_encoder(self.bw_delta_sat_input_norm(local_state.selected_sat_tokens))
        sat_mean = _masked_mean_tokens(sat_token, sat_mask)
        sat_max = _masked_max_tokens(sat_token, sat_mask)

        action_slot_feat = torch.stack(
            [
                sampled_action,
                ref_action,
                action_delta,
                abs_action_delta,
            ],
            dim=-1,
        )
        user_pair = torch.cat([local_state.gu_tokens, action_slot_feat], dim=-1)
        user_token = self.bw_delta_user_encoder(self.bw_delta_user_input_norm(user_pair))
        user_mean = _masked_mean_tokens(user_token, user_mask)
        user_max = _masked_max_tokens(user_token, user_mask)

        user_mask_f = user_mask.to(dtype=sampled_action.dtype)
        active_count = user_mask_f.sum(dim=-1, keepdim=True)
        l1_delta = (abs_action_delta * user_mask_f).sum(dim=-1, keepdim=True)
        max_delta = torch.where(user_mask, abs_action_delta, torch.zeros_like(abs_action_delta)).amax(dim=-1, keepdim=True)
        sampled_active_mean = (sampled_action * user_mask_f).sum(dim=-1, keepdim=True) / active_count.clamp_min(1.0)
        ref_active_mean = (ref_action * user_mask_f).sum(dim=-1, keepdim=True) / active_count.clamp_min(1.0)

        agent_feat = torch.cat(
            [
                ego_token,
                sat_mean,
                sat_max,
                user_mean,
                user_max,
                l1_delta,
                max_delta,
                sampled_active_mean,
                ref_active_mean,
            ],
            dim=-1,
        )
        agent_hidden = self.bw_delta_agent_fusion(agent_feat)
        team_hidden = agent_hidden.reshape(sample_count, int(num_agents), -1).mean(dim=1)
        return self.bw_delta_head(self.bw_delta_team_fusion(team_hidden)).squeeze(-1)

    def forward(self, world_state: StructuredWorldState) -> dict[str, torch.Tensor]:
        return {
            "accel": self.value_accel(world_state),
            "sat": self.value_sat(world_state),
            "bw": self.value_bw(world_state),
        }
