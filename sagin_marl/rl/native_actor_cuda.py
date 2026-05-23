from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sagin_marl.env import native_cuda
from sagin_marl.rl import structured_accel_actor_schema as accel_schema
BW_COMPETITION_MAX_LAYERS = 56
SAT_COMPETITION_MAX_LAYERS = 4
ACCEL_INTERACTION_MAX_LAYERS = 8
NATIVE_ACTOR_CUDA_MAX_HIDDEN = 512
NATIVE_ACTOR_CUDA_MAX_EMBED = 256
NATIVE_ACTOR_CUDA_MAX_MLP_LAYERS = 4


ACTOR_WEIGHT_NAMES: tuple[str, ...] = (
    # Accel policy.
    "accel_policy.log_std",
    "accel_policy.ego_norm.weight",
    "accel_policy.ego_norm.bias",
    "accel_policy.cell_norm.weight",
    "accel_policy.cell_norm.bias",
    "accel_policy.gu_norm.weight",
    "accel_policy.gu_norm.bias",
    "accel_policy.peer_norm.weight",
    "accel_policy.peer_norm.bias",
    "accel_policy.sat_norm.weight",
    "accel_policy.sat_norm.bias",
    "accel_policy.ego_encoder.0.weight",
    "accel_policy.ego_encoder.0.bias",
    "accel_policy.ego_encoder.2.weight",
    "accel_policy.ego_encoder.2.bias",
    "accel_policy.cell_encoder.0.weight",
    "accel_policy.cell_encoder.0.bias",
    "accel_policy.cell_encoder.2.weight",
    "accel_policy.cell_encoder.2.bias",
    "accel_policy.gu_encoder.0.weight",
    "accel_policy.gu_encoder.0.bias",
    "accel_policy.gu_encoder.2.weight",
    "accel_policy.gu_encoder.2.bias",
    "accel_policy.peer_encoder.0.weight",
    "accel_policy.peer_encoder.0.bias",
    "accel_policy.peer_encoder.2.weight",
    "accel_policy.peer_encoder.2.bias",
    "accel_policy.sat_encoder.0.weight",
    "accel_policy.sat_encoder.0.bias",
    "accel_policy.sat_encoder.2.weight",
    "accel_policy.sat_encoder.2.bias",
    "accel_policy.gu_query.weight",
    "accel_policy.gu_query.bias",
    "accel_policy.peer_query.weight",
    "accel_policy.peer_query.bias",
    "accel_policy.sat_query.weight",
    "accel_policy.sat_query.bias",
    "accel_policy.fusion.0.weight",
    "accel_policy.fusion.0.bias",
    "accel_policy.fusion.2.weight",
    "accel_policy.fusion.2.bias",
    "accel_policy.mu_head.weight",
    "accel_policy.mu_head.bias",
    # Redesigned SAT subset policy fixed base weights. Keep this block length
    # aligned with the legacy SAT ABI slot range so BW weight indices stay stable.
    "sat_subset_policy.ego_input_norm.weight",
    "sat_subset_policy.ego_input_norm.bias",
    "sat_subset_policy.demand_input_norm.weight",
    "sat_subset_policy.demand_input_norm.bias",
    "sat_subset_policy.ego_encoder.0.weight",
    "sat_subset_policy.ego_encoder.0.bias",
    "sat_subset_policy.ego_encoder.2.weight",
    "sat_subset_policy.ego_encoder.2.bias",
    "sat_subset_policy.demand_encoder.0.weight",
    "sat_subset_policy.demand_encoder.0.bias",
    "sat_subset_policy.demand_encoder.2.weight",
    "sat_subset_policy.demand_encoder.2.bias",
    "sat_subset_policy.role_encoder.0.weight",
    "sat_subset_policy.role_encoder.0.bias",
    "sat_subset_policy.role_encoder.2.weight",
    "sat_subset_policy.role_encoder.2.bias",
    "sat_subset_policy.sat_input_norm.weight",
    "sat_subset_policy.sat_input_norm.bias",
    "sat_subset_policy.sat_encoder.0.weight",
    "sat_subset_policy.sat_encoder.0.bias",
    "sat_subset_policy.sat_encoder.2.weight",
    "sat_subset_policy.sat_encoder.2.bias",
    "sat_subset_policy.ctx_encoder.0.weight",
    "sat_subset_policy.ctx_encoder.0.bias",
    "sat_subset_policy.ctx_encoder.2.weight",
    "sat_subset_policy.ctx_encoder.2.bias",
    "sat_subset_policy.sat_context_fusion.0.weight",
    "sat_subset_policy.sat_context_fusion.0.bias",
    "sat_subset_policy.sat_context_fusion.2.weight",
    "sat_subset_policy.sat_context_fusion.2.bias",
    "sat_subset_policy.sat_logit_head.0.weight",
    "sat_subset_policy.sat_logit_head.0.bias",
    "sat_subset_policy.sat_logit_head.2.weight",
    "sat_subset_policy.sat_logit_head.2.bias",
    "sat_subset_policy.count_logit_head.0.weight",
    "sat_subset_policy.count_logit_head.0.bias",
    "sat_subset_policy.count_logit_head.2.weight",
    "sat_subset_policy.count_logit_head.2.bias",
    "sat_subset_policy._unused_native_sat_slot_0",
    "sat_subset_policy._unused_native_sat_slot_1",
    "sat_subset_policy._unused_native_sat_slot_2",
    "sat_subset_policy._unused_native_sat_slot_3",
    # BW policy common.
    "bw_policy.ego_input_norm.weight",
    "bw_policy.ego_input_norm.bias",
    "bw_policy.sat_input_norm.weight",
    "bw_policy.sat_input_norm.bias",
    "bw_policy.gu_input_norm.weight",
    "bw_policy.gu_input_norm.bias",
    "bw_policy.ego_encoder.0.weight",
    "bw_policy.ego_encoder.0.bias",
    "bw_policy.ego_encoder.2.weight",
    "bw_policy.ego_encoder.2.bias",
    "bw_policy.sat_encoder.0.weight",
    "bw_policy.sat_encoder.0.bias",
    "bw_policy.sat_encoder.2.weight",
    "bw_policy.sat_encoder.2.bias",
    "bw_policy.gu_encoder.0.weight",
    "bw_policy.gu_encoder.0.bias",
    "bw_policy.gu_encoder.2.weight",
    "bw_policy.gu_encoder.2.bias",
    "bw_policy.sat_add_proj.weight",
    "bw_policy.sat_add_proj.bias",
    "bw_policy.down_query_proj.weight",
    "bw_policy.down_query_proj.bias",
    "bw_policy.down_context_encoder.0.weight",
    "bw_policy.down_context_encoder.0.bias",
    "bw_policy.down_context_encoder.2.weight",
    "bw_policy.down_context_encoder.2.bias",
    "bw_policy.ctx0_encoder.0.weight",
    "bw_policy.ctx0_encoder.0.bias",
    "bw_policy.ctx0_encoder.2.weight",
    "bw_policy.ctx0_encoder.2.bias",
    "bw_policy.gu_context_fusion.0.weight",
    "bw_policy.gu_context_fusion.0.bias",
    "bw_policy.gu_context_fusion.2.weight",
    "bw_policy.gu_context_fusion.2.bias",
    "bw_policy.score_head.0.weight",
    "bw_policy.score_head.0.bias",
    "bw_policy.score_head.2.weight",
    "bw_policy.score_head.2.bias",
    "bw_policy.tau_head.0.weight",
    "bw_policy.tau_head.0.bias",
    "bw_policy.tau_head.2.weight",
    "bw_policy.tau_head.2.bias",
    "bw_policy.kappa_head.0.weight",
    "bw_policy.kappa_head.0.bias",
    "bw_policy.kappa_head.2.weight",
    "bw_policy.kappa_head.2.bias",
)

_BW_COMPETITION_BLOCK_SUFFIXES: tuple[str, ...] = (
    "attn.in_proj_weight",
    "attn.in_proj_bias",
    "attn.out_proj.weight",
    "attn.out_proj.bias",
    "norm_attn.weight",
    "norm_attn.bias",
    "ffn.0.weight",
    "ffn.0.bias",
    "ffn.2.weight",
    "ffn.2.bias",
    "norm_ffn.weight",
    "norm_ffn.bias",
)

ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + tuple(
    f"bw_policy.competition_blocks.{layer}.{suffix}"
    for layer in range(BW_COMPETITION_MAX_LAYERS)
    for suffix in _BW_COMPETITION_BLOCK_SUFFIXES
)

SAT_BLOCK_WEIGHT_BASE = len(ACTOR_WEIGHT_NAMES)
_SAT_COMPETITION_BLOCK_SUFFIXES: tuple[str, ...] = (
    "attn_norm.weight",
    "attn_norm.bias",
    "qkv_proj.weight",
    "qkv_proj.bias",
    "out_proj.weight",
    "out_proj.bias",
    "ffn_norm.weight",
    "ffn_norm.bias",
    "ffn.0.weight",
    "ffn.0.bias",
    "ffn.2.weight",
    "ffn.2.bias",
)

ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + tuple(
    f"sat_subset_policy.sat_self_attention_blocks.{layer}.{suffix}"
    for layer in range(SAT_COMPETITION_MAX_LAYERS)
    for suffix in _SAT_COMPETITION_BLOCK_SUFFIXES
)

_ACTOR_EXTRA_MLP_PREFIXES: tuple[str, ...] = (
    "accel_policy.ego_encoder",
    "accel_policy.cell_encoder",
    "accel_policy.gu_encoder",
    "accel_policy.peer_encoder",
    "accel_policy.sat_encoder",
    "accel_policy.fusion",
    "sat_subset_policy.ego_encoder",
    "sat_subset_policy.demand_encoder",
    "sat_subset_policy.role_encoder",
    "sat_subset_policy.sat_encoder",
    "sat_subset_policy.ctx_encoder",
    "sat_subset_policy.sat_context_fusion",
    "sat_subset_policy.sat_logit_head",
    "sat_subset_policy.count_logit_head",
    "bw_policy.ego_encoder",
    "bw_policy.sat_encoder",
    "bw_policy.gu_encoder",
    "bw_policy.down_context_encoder",
    "bw_policy.ctx0_encoder",
    "bw_policy.gu_context_fusion",
    "bw_policy.score_head",
    "bw_policy.tau_head",
    "bw_policy.kappa_head",
)
_ACTOR_EXTRA_MLP_SUFFIXES: tuple[str, ...] = (
    "4.weight",
    "4.bias",
    "6.weight",
    "6.bias",
)
ACTOR_EXTRA_MLP_WEIGHT_BASE = len(ACTOR_WEIGHT_NAMES)
ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + tuple(
    f"{prefix}.{suffix}"
    for prefix in _ACTOR_EXTRA_MLP_PREFIXES
    for suffix in _ACTOR_EXTRA_MLP_SUFFIXES
)

ACCEL_BLOCK_WEIGHT_BASE = len(ACTOR_WEIGHT_NAMES)
_ACCEL_INTERACTION_BLOCK_SUFFIXES: tuple[str, ...] = _BW_COMPETITION_BLOCK_SUFFIXES
ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + tuple(
    f"accel_policy.interaction_blocks.{layer}.{suffix}"
    for layer in range(ACCEL_INTERACTION_MAX_LAYERS)
    for suffix in _ACCEL_INTERACTION_BLOCK_SUFFIXES
)

ACCEL_MU_HEAD_MLP_WEIGHT_BASE = len(ACTOR_WEIGHT_NAMES)
_ACCEL_MU_HEAD_MLP_SUFFIXES: tuple[str, ...] = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
    "6.weight",
    "6.bias",
)
ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + tuple(
    f"accel_policy.mu_head.{suffix}" for suffix in _ACCEL_MU_HEAD_MLP_SUFFIXES
)

BW_TAU_KAPPA_PACKED_WEIGHT_BASE = len(ACTOR_WEIGHT_NAMES)
_BW_TAU_KAPPA_PACKED_NAMES: tuple[str, ...] = (
    "bw_policy._native_tau_kappa_head.0.weight",
    "bw_policy._native_tau_kappa_head.0.bias",
    "bw_policy._native_tau_kappa_head.2.weight",
    "bw_policy._native_tau_kappa_head.2.bias",
)
ACTOR_WEIGHT_NAMES = ACTOR_WEIGHT_NAMES + _BW_TAU_KAPPA_PACKED_NAMES

_RAW_INPUT_NORM_WEIGHT_NAMES: frozenset[str] = frozenset(
    {
        "accel_policy.ego_norm.weight",
        "accel_policy.ego_norm.bias",
        "accel_policy.cell_norm.weight",
        "accel_policy.cell_norm.bias",
        "accel_policy.gu_norm.weight",
        "accel_policy.gu_norm.bias",
        "accel_policy.peer_norm.weight",
        "accel_policy.peer_norm.bias",
        "accel_policy.sat_norm.weight",
        "accel_policy.sat_norm.bias",
        "sat_subset_policy.ego_input_norm.weight",
        "sat_subset_policy.ego_input_norm.bias",
        "sat_subset_policy.demand_input_norm.weight",
        "sat_subset_policy.demand_input_norm.bias",
        "sat_subset_policy.sat_input_norm.weight",
        "sat_subset_policy.sat_input_norm.bias",
        "bw_policy.ego_input_norm.weight",
        "bw_policy.ego_input_norm.bias",
        "bw_policy.sat_input_norm.weight",
        "bw_policy.sat_input_norm.bias",
        "bw_policy.gu_input_norm.weight",
        "bw_policy.gu_input_norm.bias",
    }
)


def _build_bw_tau_kappa_packed_tensor(
    name: str,
    state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    head_layers: int,
) -> torch.Tensor | None:
    """Build native-only packed tau/kappa head tensors for the 2-layer case."""

    if int(head_layers) != 2 or name not in _BW_TAU_KAPPA_PACKED_NAMES:
        return None
    tau0_w = state.get("bw_policy.tau_head.0.weight")
    tau0_b = state.get("bw_policy.tau_head.0.bias")
    tau2_w = state.get("bw_policy.tau_head.2.weight")
    tau2_b = state.get("bw_policy.tau_head.2.bias")
    kap0_w = state.get("bw_policy.kappa_head.0.weight")
    kap0_b = state.get("bw_policy.kappa_head.0.bias")
    kap2_w = state.get("bw_policy.kappa_head.2.weight")
    kap2_b = state.get("bw_policy.kappa_head.2.bias")
    tensors = (tau0_w, tau0_b, tau2_w, tau2_b, kap0_w, kap0_b, kap2_w, kap2_b)
    if any(t is None for t in tensors):
        return None
    assert tau0_w is not None and tau0_b is not None and tau2_w is not None and tau2_b is not None
    assert kap0_w is not None and kap0_b is not None and kap2_w is not None and kap2_b is not None
    if name.endswith(".0.weight"):
        return torch.cat([tau0_w, kap0_w], dim=0).detach().to(device=device, dtype=torch.float32).contiguous()
    if name.endswith(".0.bias"):
        return torch.cat([tau0_b, kap0_b], dim=0).detach().to(device=device, dtype=torch.float32).contiguous()
    if name.endswith(".2.weight"):
        hidden = int(tau2_w.shape[-1])
        packed = torch.zeros((2, hidden * 2), dtype=torch.float32, device=device)
        packed[0, :hidden].copy_(tau2_w.detach().to(device=device, dtype=torch.float32).reshape(-1))
        packed[1, hidden : hidden * 2].copy_(kap2_w.detach().to(device=device, dtype=torch.float32).reshape(-1))
        return packed.contiguous()
    if name.endswith(".2.bias"):
        return torch.cat([tau2_b, kap2_b], dim=0).detach().to(device=device, dtype=torch.float32).contiguous()
    return None


_REQUIRED_BW_BASE_WEIGHT_NAMES: tuple[str, ...] = (
    "bw_policy.ego_encoder.0.weight",
    "bw_policy.ego_encoder.0.bias",
    "bw_policy.ego_encoder.2.weight",
    "bw_policy.ego_encoder.2.bias",
    "bw_policy.sat_encoder.0.weight",
    "bw_policy.sat_encoder.0.bias",
    "bw_policy.sat_encoder.2.weight",
    "bw_policy.sat_encoder.2.bias",
    "bw_policy.gu_encoder.0.weight",
    "bw_policy.gu_encoder.0.bias",
    "bw_policy.gu_encoder.2.weight",
    "bw_policy.gu_encoder.2.bias",
    "bw_policy.down_query_proj.weight",
    "bw_policy.down_query_proj.bias",
    "bw_policy.sat_add_proj.weight",
    "bw_policy.sat_add_proj.bias",
    "bw_policy.down_context_encoder.0.weight",
    "bw_policy.down_context_encoder.0.bias",
    "bw_policy.down_context_encoder.2.weight",
    "bw_policy.down_context_encoder.2.bias",
    "bw_policy.ctx0_encoder.0.weight",
    "bw_policy.ctx0_encoder.0.bias",
    "bw_policy.ctx0_encoder.2.weight",
    "bw_policy.ctx0_encoder.2.bias",
    "bw_policy.gu_context_fusion.0.weight",
    "bw_policy.gu_context_fusion.0.bias",
    "bw_policy.gu_context_fusion.2.weight",
    "bw_policy.gu_context_fusion.2.bias",
    "bw_policy.score_head.0.weight",
    "bw_policy.score_head.0.bias",
    "bw_policy.score_head.2.weight",
    "bw_policy.score_head.2.bias",
    "bw_policy.tau_head.0.weight",
    "bw_policy.tau_head.0.bias",
    "bw_policy.tau_head.2.weight",
    "bw_policy.tau_head.2.bias",
    "bw_policy.kappa_head.0.weight",
    "bw_policy.kappa_head.0.bias",
    "bw_policy.kappa_head.2.weight",
    "bw_policy.kappa_head.2.bias",
)


def _require_actor_mlp_layer_range(policy: Any, attr: str, owner: str) -> int:
    value = int(getattr(policy, attr, 2) or 2)
    if value < 2 or value > NATIVE_ACTOR_CUDA_MAX_MLP_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports {owner}.{attr} in [2, {NATIVE_ACTOR_CUDA_MAX_MLP_LAYERS}], "
            f"got {value}."
        )
    return value


def _extra_mlp_names(prefixes: tuple[str, ...], layers: int) -> set[str]:
    names: set[str] = set()
    if int(layers) >= 3:
        for prefix in prefixes:
            names.add(f"{prefix}.4.weight")
            names.add(f"{prefix}.4.bias")
    if int(layers) >= 4:
        for prefix in prefixes:
            names.add(f"{prefix}.6.weight")
            names.add(f"{prefix}.6.bias")
    return names


def _contiguous_mlp_names(prefix: str, layers: int) -> set[str]:
    names: set[str] = set()
    if int(layers) >= 2:
        names.update({f"{prefix}.0.weight", f"{prefix}.0.bias", f"{prefix}.2.weight", f"{prefix}.2.bias"})
    if int(layers) >= 3:
        names.update({f"{prefix}.4.weight", f"{prefix}.4.bias"})
    if int(layers) >= 4:
        names.update({f"{prefix}.6.weight", f"{prefix}.6.bias"})
    return names


def _required_bw_weight_names(bw_policy: Any) -> set[str]:
    names = set(_REQUIRED_BW_BASE_WEIGHT_NAMES)
    encoder_layers = _require_actor_mlp_layer_range(bw_policy, "encoder_mlp_layers", "bw_policy")
    context_layers = _require_actor_mlp_layer_range(bw_policy, "context_mlp_layers", "bw_policy")
    head_layers = _require_actor_mlp_layer_range(bw_policy, "head_mlp_layers", "bw_policy")
    names.update(_extra_mlp_names(("bw_policy.ego_encoder", "bw_policy.sat_encoder", "bw_policy.gu_encoder"), encoder_layers))
    names.update(_extra_mlp_names(("bw_policy.down_context_encoder", "bw_policy.ctx0_encoder", "bw_policy.gu_context_fusion"), context_layers))
    names.update(_extra_mlp_names(("bw_policy.score_head", "bw_policy.tau_head", "bw_policy.kappa_head"), head_layers))
    competition_blocks = getattr(bw_policy, "competition_blocks", None)
    layer_count = 0 if competition_blocks is None else len(competition_blocks)
    for layer in range(layer_count):
        for suffix in _BW_COMPETITION_BLOCK_SUFFIXES:
            names.add(f"bw_policy.competition_blocks.{layer}.{suffix}")
    return names


_REQUIRED_SAT_BASE_WEIGHT_NAMES: tuple[str, ...] = tuple(
    name
    for name in ACTOR_WEIGHT_NAMES[:85]
    if name.startswith("sat_subset_policy.")
    and "._unused_native_sat_slot_" not in name
    and name not in _RAW_INPUT_NORM_WEIGHT_NAMES
)


def _required_sat_weight_names(sat_policy: Any) -> set[str]:
    names = set(_REQUIRED_SAT_BASE_WEIGHT_NAMES)
    encoder_layers = _require_actor_mlp_layer_range(sat_policy, "encoder_mlp_layers", "sat_subset_policy")
    context_layers = _require_actor_mlp_layer_range(sat_policy, "context_mlp_layers", "sat_subset_policy")
    head_layers = _require_actor_mlp_layer_range(sat_policy, "head_mlp_layers", "sat_subset_policy")
    names.update(_extra_mlp_names(("sat_subset_policy.ego_encoder", "sat_subset_policy.demand_encoder", "sat_subset_policy.role_encoder", "sat_subset_policy.sat_encoder"), encoder_layers))
    names.update(_extra_mlp_names(("sat_subset_policy.ctx_encoder", "sat_subset_policy.sat_context_fusion"), context_layers))
    names.update(_extra_mlp_names(("sat_subset_policy.sat_logit_head", "sat_subset_policy.count_logit_head"), head_layers))
    blocks = getattr(sat_policy, "sat_self_attention_blocks", None)
    layer_count = 0 if blocks is None else len(blocks)
    if layer_count > SAT_COMPETITION_MAX_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports up to {SAT_COMPETITION_MAX_LAYERS} SAT competition layers, "
            f"got {layer_count}."
        )
    for layer in range(layer_count):
        for suffix in _SAT_COMPETITION_BLOCK_SUFFIXES:
            names.add(f"sat_subset_policy.sat_self_attention_blocks.{layer}.{suffix}")
    return names


_REQUIRED_ACCEL_BASE_WEIGHT_NAMES: tuple[str, ...] = (
    "accel_policy.log_std",
    "accel_policy.ego_encoder.0.weight",
    "accel_policy.ego_encoder.0.bias",
    "accel_policy.ego_encoder.2.weight",
    "accel_policy.ego_encoder.2.bias",
    "accel_policy.cell_encoder.0.weight",
    "accel_policy.cell_encoder.0.bias",
    "accel_policy.cell_encoder.2.weight",
    "accel_policy.cell_encoder.2.bias",
    "accel_policy.gu_encoder.0.weight",
    "accel_policy.gu_encoder.0.bias",
    "accel_policy.gu_encoder.2.weight",
    "accel_policy.gu_encoder.2.bias",
    "accel_policy.peer_encoder.0.weight",
    "accel_policy.peer_encoder.0.bias",
    "accel_policy.peer_encoder.2.weight",
    "accel_policy.peer_encoder.2.bias",
    "accel_policy.sat_encoder.0.weight",
    "accel_policy.sat_encoder.0.bias",
    "accel_policy.sat_encoder.2.weight",
    "accel_policy.sat_encoder.2.bias",
    "accel_policy.gu_query.weight",
    "accel_policy.gu_query.bias",
    "accel_policy.peer_query.weight",
    "accel_policy.peer_query.bias",
    "accel_policy.sat_query.weight",
    "accel_policy.sat_query.bias",
    "accel_policy.fusion.0.weight",
    "accel_policy.fusion.0.bias",
    "accel_policy.fusion.2.weight",
    "accel_policy.fusion.2.bias",
)


def _required_accel_weight_names(accel_policy: Any) -> set[str]:
    names = set(_REQUIRED_ACCEL_BASE_WEIGHT_NAMES)
    encoder_layers = _require_actor_mlp_layer_range(accel_policy, "encoder_mlp_layers", "accel_policy")
    context_layers = _require_actor_mlp_layer_range(accel_policy, "context_mlp_layers", "accel_policy")
    head_layers = int(getattr(accel_policy, "head_mlp_layers", 1) or 1)
    if head_layers <= 1:
        names.add("accel_policy.mu_head.weight")
        names.add("accel_policy.mu_head.bias")
    else:
        _require_actor_mlp_layer_range(accel_policy, "head_mlp_layers", "accel_policy")
        names.update(_contiguous_mlp_names("accel_policy.mu_head", head_layers))
    names.update(_extra_mlp_names(("accel_policy.ego_encoder", "accel_policy.cell_encoder", "accel_policy.gu_encoder", "accel_policy.peer_encoder", "accel_policy.sat_encoder"), encoder_layers))
    names.update(_extra_mlp_names(("accel_policy.fusion",), context_layers))
    blocks = getattr(accel_policy, "interaction_blocks", None)
    layer_count = 0 if blocks is None else len(blocks)
    if layer_count > ACCEL_INTERACTION_MAX_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports up to {ACCEL_INTERACTION_MAX_LAYERS} accel interaction layers, "
            f"got {layer_count}."
        )
    for layer in range(layer_count):
        for suffix in _ACCEL_INTERACTION_BLOCK_SUFFIXES:
            names.add(f"accel_policy.interaction_blocks.{layer}.{suffix}")
    return names


@dataclass
class NativeActorCudaBinding:
    actor: Any
    abi: native_cuda.NativeActorCudaABI
    _owned_weights: tuple[torch.Tensor, ...]
    _empty_weight: torch.Tensor

    def sync_from_module(self) -> None:
        state = self.actor.state_dict()
        bw_policy = getattr(self.actor, "bw_policy", None)
        bw_head_layers = _require_actor_mlp_layer_range(bw_policy, "head_mlp_layers", "bw_policy")
        required_accel = _required_accel_weight_names(getattr(self.actor, "accel_policy", None))
        required_sat = _required_sat_weight_names(getattr(self.actor, "sat_subset_policy", None))
        required_bw = _required_bw_weight_names(bw_policy)
        with torch.no_grad():
            for name, target in zip(ACTOR_WEIGHT_NAMES, self._owned_weights):
                packed = _build_bw_tau_kappa_packed_tensor(
                    name,
                    state,
                    device=target.device if target.numel() > 0 else self._empty_weight.device,
                    head_layers=bw_head_layers,
                )
                if packed is not None:
                    if tuple(packed.shape) != tuple(target.shape):
                        raise RuntimeError(
                            f"native actor CUDA ABI derived weight {name!r} shape changed from "
                            f"{tuple(target.shape)} to {tuple(packed.shape)}."
                        )
                    target.copy_(packed)
                    continue
                source = state.get(name)
                if source is None:
                    if name in _BW_TAU_KAPPA_PACKED_NAMES:
                        continue
                    if name in _RAW_INPUT_NORM_WEIGHT_NAMES:
                        continue
                    if name in required_accel or name in required_bw or name in required_sat:
                        raise RuntimeError(f"native actor CUDA ABI missing required weight {name!r}.")
                    continue
                if tuple(source.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"native actor CUDA ABI weight {name!r} shape changed from "
                        f"{tuple(target.shape)} to {tuple(source.shape)}."
                    )
                target.copy_(source.detach().to(device=target.device, dtype=torch.float32))


def build_native_actor_cuda_binding(actor: Any, *, device: torch.device | str) -> NativeActorCudaBinding:
    tensor_device = torch.device(device)
    if tensor_device.type != "cuda":
        raise RuntimeError("native actor CUDA binding requires a CUDA device.")
    state = actor.state_dict()
    accel_policy = getattr(actor, "accel_policy", None)
    sat_policy = getattr(actor, "sat_subset_policy", None)
    bw_policy = getattr(actor, "bw_policy", None)
    if accel_policy is None or sat_policy is None or bw_policy is None:
        raise RuntimeError("native actor CUDA binding requires StructuredActor policies.")
    required_accel = _required_accel_weight_names(accel_policy)
    required_bw = _required_bw_weight_names(bw_policy)
    required_sat = _required_sat_weight_names(sat_policy)
    empty = torch.empty((0,), dtype=torch.float32, device=tensor_device)
    owned: list[torch.Tensor] = []
    for name in ACTOR_WEIGHT_NAMES:
        packed = _build_bw_tau_kappa_packed_tensor(
            name,
            state,
            device=tensor_device,
            head_layers=int(getattr(bw_policy, "head_mlp_layers", 2) or 2),
        )
        if packed is not None:
            owned.append(packed.clone())
            continue
        source = state.get(name)
        if source is None:
            if name in _BW_TAU_KAPPA_PACKED_NAMES:
                owned.append(empty)
                continue
            if name in _RAW_INPUT_NORM_WEIGHT_NAMES:
                owned.append(empty)
                continue
            if name in required_accel or name in required_bw or name in required_sat:
                raise RuntimeError(f"native actor CUDA ABI missing required weight {name!r}.")
            owned.append(empty)
            continue
        owned.append(source.detach().to(device=tensor_device, dtype=torch.float32).contiguous().clone())

    accel_hidden = int(getattr(accel_policy, "hidden_dim", 0) or 0)
    sat_hidden = int(getattr(sat_policy, "hidden_dim", 0) or 0)
    bw_hidden = int(getattr(bw_policy, "hidden_dim", 0) or 0)
    accel_embed = int(getattr(accel_policy, "embed_dim", 0) or 0)
    sat_embed = int(getattr(sat_policy, "embed_dim", 0) or 0)
    bw_embed = int(getattr(bw_policy, "embed_dim", 0) or 0)
    for label, value in (("accel", accel_hidden), ("sat", sat_hidden), ("bw", bw_hidden)):
        if value <= 0 or value > NATIVE_ACTOR_CUDA_MAX_HIDDEN:
            raise RuntimeError(
                f"native actor CUDA {label} hidden_dim must satisfy "
                f"0 < hidden_dim <= {NATIVE_ACTOR_CUDA_MAX_HIDDEN}, got {value}."
            )
    for label, value in (("accel", accel_embed), ("sat", sat_embed), ("bw", bw_embed)):
        if value <= 0 or value > NATIVE_ACTOR_CUDA_MAX_EMBED:
            raise RuntimeError(
                f"native actor CUDA {label} embed_dim must satisfy "
                f"0 < embed_dim <= {NATIVE_ACTOR_CUDA_MAX_EMBED}, got {value}."
            )
    hidden_dim = max(accel_hidden, sat_hidden, bw_hidden)
    embed_dim = max(accel_embed, sat_embed, bw_embed)
    bw_competition_blocks = getattr(bw_policy, "competition_blocks", None)
    competition_layers = 0 if bw_competition_blocks is None else len(bw_competition_blocks)
    if competition_layers > BW_COMPETITION_MAX_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports up to {BW_COMPETITION_MAX_LAYERS} BW competition layers, "
            f"got {competition_layers}."
        )
    sat_competition_blocks = getattr(sat_policy, "sat_self_attention_blocks", None)
    sat_competition_layers = 0 if sat_competition_blocks is None else len(sat_competition_blocks)
    if sat_competition_layers > SAT_COMPETITION_MAX_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports up to {SAT_COMPETITION_MAX_LAYERS} SAT competition layers, "
            f"got {sat_competition_layers}."
        )
    accel_interaction_blocks = getattr(accel_policy, "interaction_blocks", None)
    accel_interaction_layers = 0 if accel_interaction_blocks is None else len(accel_interaction_blocks)
    if accel_interaction_layers > ACCEL_INTERACTION_MAX_LAYERS:
        raise RuntimeError(
            f"native actor CUDA supports up to {ACCEL_INTERACTION_MAX_LAYERS} accel interaction layers, "
            f"got {accel_interaction_layers}."
        )
    accel_encoder_layers = _require_actor_mlp_layer_range(accel_policy, "encoder_mlp_layers", "accel_policy")
    accel_context_layers = _require_actor_mlp_layer_range(accel_policy, "context_mlp_layers", "accel_policy")
    accel_head_layers = int(getattr(accel_policy, "head_mlp_layers", 1) or 1)
    if accel_head_layers > 1:
        _require_actor_mlp_layer_range(accel_policy, "head_mlp_layers", "accel_policy")
    accel_gu_query_count = int(getattr(accel_policy, "gu_query_count", accel_schema.ACCEL_GU_QUERY_COUNT) or 0)
    accel_peer_query_count = int(getattr(accel_policy, "peer_query_count", accel_schema.ACCEL_PEER_QUERY_COUNT) or 0)
    accel_sat_query_count = int(getattr(accel_policy, "sat_query_count", accel_schema.ACCEL_SAT_QUERY_COUNT) or 0)
    for label, value in (
        ("accel gu_query_count", accel_gu_query_count),
        ("accel peer_query_count", accel_peer_query_count),
        ("accel sat_query_count", accel_sat_query_count),
    ):
        if value <= 0:
            raise RuntimeError(f"native actor CUDA {label} must be positive, got {value}.")
    sat_encoder_layers = _require_actor_mlp_layer_range(sat_policy, "encoder_mlp_layers", "sat_subset_policy")
    sat_context_layers = _require_actor_mlp_layer_range(sat_policy, "context_mlp_layers", "sat_subset_policy")
    sat_head_layers = _require_actor_mlp_layer_range(sat_policy, "head_mlp_layers", "sat_subset_policy")
    bw_encoder_layers = _require_actor_mlp_layer_range(bw_policy, "encoder_mlp_layers", "bw_policy")
    bw_context_layers = _require_actor_mlp_layer_range(bw_policy, "context_mlp_layers", "bw_policy")
    bw_head_layers = _require_actor_mlp_layer_range(bw_policy, "head_mlp_layers", "bw_policy")
    rng_seed = int(getattr(actor, "native_cuda_rng_seed", 0xA123B456C789D012)) & 0xFFFFFFFFFFFFFFFF
    bw_mode_name = str(getattr(bw_policy, "native_dirichlet_diagnostic_mode", "current") or "current").strip().lower()
    bw_mode_code = {"current": 0, "new_fast": 1, "legacy_fast": 2}.get(bw_mode_name)
    if bw_mode_code is None:
        raise RuntimeError(f"Unsupported native BW Dirichlet diagnostic mode: {bw_mode_name!r}.")
    int_params = (
        int(hidden_dim),
        int(embed_dim),
        int(bool(getattr(sat_policy, "use_sat_tokens", True))),
        int(getattr(bw_policy, "num_heads", getattr(bw_policy, "competition_heads", 1))),
        int(competition_layers),
        int(rng_seed & 0xFFFFFFFF),
        int((rng_seed >> 32) & 0xFFFFFFFF),
        accel_schema.ACCEL_EGO_DIM,
        accel_schema.ACCEL_CELL_DIM,
        accel_schema.ACCEL_GU_TOKEN_DIM,
        accel_schema.ACCEL_PEER_TOKEN_DIM,
        accel_schema.ACCEL_SAT_TOKEN_DIM,
        int(accel_gu_query_count),
        int(accel_peer_query_count),
        int(accel_sat_query_count),
        int(getattr(bw_policy, "down_query_count", 1)),
        int(getattr(sat_policy, "sat_attention_heads", getattr(sat_policy, "num_heads", 1))),
        int(sat_competition_layers),
        int(SAT_BLOCK_WEIGHT_BASE),
        int(len(_SAT_COMPETITION_BLOCK_SUFFIXES)),
        int(accel_encoder_layers),
        int(accel_context_layers),
        int(accel_interaction_layers),
        int(getattr(accel_policy, "attention_heads", 1)),
        int(ACCEL_BLOCK_WEIGHT_BASE),
        int(len(_ACCEL_INTERACTION_BLOCK_SUFFIXES)),
        int(sat_encoder_layers),
        int(sat_context_layers),
        int(sat_head_layers),
        int(bw_encoder_layers),
        int(bw_context_layers),
        int(bw_head_layers),
        int(ACTOR_EXTRA_MLP_WEIGHT_BASE),
        int(len(_ACTOR_EXTRA_MLP_SUFFIXES)),
        int(accel_head_layers),
        int(ACCEL_MU_HEAD_MLP_WEIGHT_BASE),
        int(accel_hidden),
        int(accel_embed),
        int(sat_hidden),
        int(sat_embed),
        int(bw_hidden),
        int(bw_embed),
        int(BW_TAU_KAPPA_PACKED_WEIGHT_BASE),
        int(bw_mode_code),
    )
    float_params = (
        float(getattr(accel_policy, "action_scale", 1.0)),
        float(getattr(bw_policy, "tau_min", 0.7)),
        float(getattr(bw_policy, "tau_max", 1.3)),
        float(getattr(bw_policy, "kappa_min", 0.5)),
        float(getattr(bw_policy, "kappa_max", 32.0)),
        -1.0 if getattr(bw_policy, "fixed_tau", None) is None else float(getattr(bw_policy, "fixed_tau")),
        -1.0 if getattr(bw_policy, "fixed_kappa", None) is None else float(getattr(bw_policy, "fixed_kappa")),
    )
    abi = native_cuda.NativeActorCudaABI(
        weight_tensors=tuple(owned),
        int_tensors=(),
        int_params=tuple(int_params),
        float_params=tuple(float_params),
    )
    return NativeActorCudaBinding(
        actor=actor,
        abi=abi,
        _owned_weights=tuple(owned),
        _empty_weight=empty,
    )
