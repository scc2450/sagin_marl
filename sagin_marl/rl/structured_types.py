from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    ArrayLike = np.ndarray | torch.Tensor
else:
    ArrayLike = np.ndarray


@dataclass
class StructuredWorldState:
    uav_nodes: ArrayLike
    gu_nodes: ArrayLike
    sat_nodes: ArrayLike
    sat_ids: ArrayLike
    uav_gu_edges: ArrayLike
    uav_sat_edges: ArrayLike
    uav_uav_edges: ArrayLike
    global_scalars: ArrayLike
    gu_mask: ArrayLike
    sat_mask: ArrayLike
    uav_gu_mask: ArrayLike
    uav_sat_mask: ArrayLike
    uav_uav_mask: ArrayLike
    stage_id: ArrayLike


@dataclass
class LocalAccelState:
    ego_features: torch.Tensor
    ego_cell: torch.Tensor
    gu_tokens: torch.Tensor
    gu_mask: torch.Tensor
    peer_tokens: torch.Tensor
    peer_mask: torch.Tensor
    sat_tokens: torch.Tensor
    sat_mask: torch.Tensor


@dataclass
class LocalSatState:
    ego_features: torch.Tensor
    demand_features: torch.Tensor
    role_features: torch.Tensor
    sat_tokens: torch.Tensor
    sat_mask: torch.Tensor
    sat_valid_mask: torch.Tensor
    subset_members: torch.Tensor
    subset_mask: torch.Tensor
    candidate_sat_ids: torch.Tensor


@dataclass
class LocalBwState:
    ego_features: torch.Tensor
    selected_sat_tokens: torch.Tensor
    selected_sat_mask: torch.Tensor
    gu_tokens: torch.Tensor
    gu_mask: torch.Tensor
    bw_valid_mask: torch.Tensor


@dataclass
class AccelStageSnapshot:
    world_state: StructuredWorldState


@dataclass
class SatStageSnapshot:
    world_state: StructuredWorldState
    max_select: int
    local_state: LocalSatState | None = None


@dataclass
class BwStageSnapshot:
    world_state: StructuredWorldState
    assoc: ArrayLike
    selected_sat_indices: ArrayLike
    selected_sat_mask: ArrayLike
    access_gain_matrix: ArrayLike
    bw_valid_mask_full: ArrayLike
    ego_features: ArrayLike | None = None
    selected_sat_tokens: ArrayLike | None = None
    gu_tokens: ArrayLike | None = None
    gu_mask: ArrayLike | None = None
    bw_valid_mask: ArrayLike | None = None


@dataclass
class AccelPolicyOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    latent_action: torch.Tensor | None = None


@dataclass
class SatSubsetPolicyOutput:
    selected_sat_indices: torch.Tensor
    subset_index: torch.Tensor
    subset_members: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    logits: torch.Tensor


@dataclass
class BwPolicyOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    logprob_raw: torch.Tensor
    entropy_raw: torch.Tensor
    score: torch.Tensor
    det_mean: torch.Tensor
    alpha: torch.Tensor
    kappa: torch.Tensor
    valid_count: torch.Tensor
    latent_count: torch.Tensor
    tau: torch.Tensor
