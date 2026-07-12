from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _collate_dataclass, _to_device_dataclass
from sagin_marl.utils.checkpoint import load_state_dict_forgiving
from sagin_marl.rl.structured_train import as_structured_drivers, close_structured_env_group, make_structured_env_group


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_module_checkpoint(
    module: torch.nn.Module,
    path: str,
    *,
    state_key: str,
    map_location: str | torch.device,
    strict: bool = True,
) -> dict[str, object]:
    # Historical training checkpoints are trusted local artifacts and store
    # actor/critic under top-level keys instead of as a bare state_dict.
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint '{path}' did not contain a state_dict dictionary.")
    module_state = state.get(state_key, state)
    if not isinstance(module_state, dict):
        raise TypeError(f"Checkpoint '{path}' key '{state_key}' was not a state_dict dictionary.")
    info = load_state_dict_forgiving(module, module_state, strict=strict)
    info["path"] = path
    info["state_key"] = state_key if module_state is not state else ""
    return info


def _checkpoint_stage_actor_optimizer_state(path: str, stage_id: int) -> dict[str, Any] | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    optimizers = state.get("actor_optimizers")
    if not isinstance(optimizers, dict):
        return None
    opt_state = optimizers.get(int(stage_id), optimizers.get(str(int(stage_id))))
    return opt_state if isinstance(opt_state, dict) else None


def _checkpoint_stage_actor_lr(path: str, stage_id: int, fallback: float) -> float:
    opt_state = _checkpoint_stage_actor_optimizer_state(path, stage_id)
    if opt_state is None:
        return float(fallback)
    param_groups = opt_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        return float(fallback)
    lr = param_groups[0].get("lr")
    try:
        return float(lr)
    except Exception:
        return float(fallback)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _is_native_driver_group(obj: Any) -> bool:
    native_required = (
        "reset_many",
        "reset_at",
        "native_rollout_program",
        "begin_native_main_kernel_rollout",
        "native_rollout_runtime",
    )
    return all(hasattr(obj, name) for name in native_required)


def _teacher_bw_from_local_state(local_state: Any, assoc_bonus: float) -> torch.Tensor:
    valid_mask = (local_state.gu_mask > 0.5) & (local_state.bw_valid_mask > 0.5)
    if hasattr(local_state, "user_nodes") and hasattr(local_state, "user_edges"):
        q = torch.clamp(local_state.user_nodes[..., 2], min=0.0)
        eta = torch.clamp(local_state.user_edges[..., 6], min=0.0)
        prev = torch.clamp(local_state.user_edges[..., 5], min=0.0)
    else:
        # Current LocalBwState uses compact GU tokens. Index 0 tracks queue pressure;
        # index 10 tracks ego access rate under full-BW reference.
        q = torch.clamp(local_state.gu_tokens[..., 0], min=0.0)
        eta = torch.clamp(local_state.gu_tokens[..., 10], min=0.0)
        prev = torch.zeros_like(q)
    weights = q * (0.5 + eta)
    if assoc_bonus > 0.0:
        weights = weights * (1.0 + assoc_bonus * prev)
    weights = torch.where(valid_mask, torch.clamp(weights, min=0.0), torch.zeros_like(weights))
    denom = weights.sum(dim=-1, keepdim=True)
    teacher = torch.where(
        denom > 1.0e-8,
        weights / denom.clamp_min(1.0e-8),
        torch.zeros_like(weights),
    )
    valid_count = valid_mask.to(dtype=teacher.dtype).sum(dim=-1, keepdim=True)
    uniform = torch.where(
        valid_mask,
        torch.ones_like(teacher),
        torch.zeros_like(teacher),
    )
    uniform = torch.where(
        valid_count > 0.5,
        uniform / valid_count.clamp_min(1.0),
        torch.zeros_like(uniform),
    )
    teacher = torch.where(denom > 1.0e-8, teacher, uniform)
    return teacher


def _masked_simplex_kl(
    target: torch.Tensor,
    mode: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    target_valid = torch.where(valid_mask, target.clamp_min(eps), torch.zeros_like(target))
    mode_valid = torch.where(valid_mask, mode.clamp_min(eps), torch.zeros_like(mode))
    target_valid = target_valid / target_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    mode_valid = mode_valid / mode_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target_valid * (torch.log(target_valid.clamp_min(eps)) - torch.log(mode_valid.clamp_min(eps)))
    return (kl * valid_mask.to(dtype=kl.dtype)).sum(dim=-1)


def _flatten_grads(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    flat_parts: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_parts.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            flat_parts.append(grad.detach().to(dtype=torch.float32).reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32, device=loss.device)
    return torch.cat(flat_parts, dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 1.0e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().mean().item()) if value.numel() > 0 else 0.0


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().reshape(-1).to(dtype=torch.float32)
    y = y.detach().reshape(-1).to(dtype=torch.float32, device=x.device)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = x_centered.pow(2).mean().sqrt()
    y_std = y_centered.pow(2).mean().sqrt()
    if float(x_std.item()) <= 1.0e-8 or float(y_std.item()) <= 1.0e-8:
        return 0.0
    return float(((x_centered * y_centered).mean() / (x_std * y_std).clamp_min(1.0e-8)).item())


def _masked_l1_per_row(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((a - b).abs() * mask.to(dtype=a.dtype)).sum(dim=-1)


def _mean_masked_cosine(delta: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    delta = delta * mask.to(dtype=delta.dtype)
    target = target * mask.to(dtype=target.dtype)
    delta_norm = delta.pow(2).sum(dim=-1).sqrt()
    target_norm = target.pow(2).sum(dim=-1).sqrt()
    valid = (delta_norm > 1.0e-12) & (target_norm > 1.0e-12)
    if not torch.any(valid):
        return 0.0
    cosine = (delta * target).sum(dim=-1) / (delta_norm * target_norm).clamp_min(1.0e-12)
    return float(cosine[valid].mean().item())


def _direction_stats(adv: torch.Tensor, delta_logprob: torch.Tensor) -> dict[str, float]:
    adv_f = adv.detach().to(dtype=torch.float32).reshape(-1)
    delta_f = delta_logprob.detach().to(dtype=torch.float32).reshape(-1)
    if int(adv_f.numel()) <= 0:
        return {
            "adv_positive_frac": 0.0,
            "adv_negative_frac": 0.0,
            "pos_adv_delta_positive_frac": 0.0,
            "neg_adv_delta_negative_frac": 0.0,
            "direction_agree_frac": 0.0,
            "direction_wrong_frac": 0.0,
            "delta_logprob_mean": 0.0,
            "delta_logprob_abs_mean": 0.0,
            "delta_logprob_when_adv_positive_mean": 0.0,
            "delta_logprob_when_adv_positive_p10": 0.0,
            "delta_logprob_when_adv_positive_p50": 0.0,
            "delta_logprob_when_adv_positive_p90": 0.0,
            "delta_logprob_when_adv_negative_mean": 0.0,
            "delta_logprob_when_adv_negative_p10": 0.0,
            "delta_logprob_when_adv_negative_p50": 0.0,
            "delta_logprob_when_adv_negative_p90": 0.0,
            "pos_adv_delta_positive_abs_mean": 0.0,
            "pos_adv_delta_negative_abs_mean": 0.0,
            "pos_adv_delta_positive_abs_sum": 0.0,
            "pos_adv_delta_negative_abs_sum": 0.0,
            "pos_adv_up_abs_share": 0.0,
            "neg_adv_delta_negative_abs_mean": 0.0,
            "neg_adv_delta_positive_abs_mean": 0.0,
            "neg_adv_delta_negative_abs_sum": 0.0,
            "neg_adv_delta_positive_abs_sum": 0.0,
            "neg_adv_down_abs_share": 0.0,
            "direction_correct_abs_sum": 0.0,
            "direction_wrong_abs_sum": 0.0,
            "direction_correct_abs_share": 0.0,
            "credit_mean": 0.0,
            "credit_when_adv_positive_mean": 0.0,
            "credit_when_adv_negative_mean": 0.0,
            "credit_positive_abs_sum": 0.0,
            "credit_negative_abs_sum": 0.0,
            "credit_positive_abs_share": 0.0,
        }
    pos = adv_f > 0.0
    neg = adv_f < 0.0
    nonzero = adv_f.abs() > 1.0e-8
    credit = adv_f * delta_f

    def _frac(mask: torch.Tensor, denom: torch.Tensor) -> float:
        denom_count = int(denom.to(dtype=torch.bool).sum().detach().cpu().item())
        if denom_count <= 0:
            return 0.0
        return float((mask & denom).to(dtype=torch.float32).sum().detach().cpu().item() / float(denom_count))

    def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.to(dtype=torch.bool).sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(values[mask].mean().detach().cpu().item())

    def _quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
        if int(mask.to(dtype=torch.bool).sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(torch.quantile(values[mask].detach().to(dtype=torch.float32), float(q)).detach().cpu().item())

    def _abs_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.to(dtype=torch.bool).sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(values[mask].detach().abs().sum().cpu().item())

    def _abs_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.to(dtype=torch.bool).sum().detach().cpu().item()) <= 0:
            return 0.0
        return float(values[mask].detach().abs().mean().cpu().item())

    agree = ((pos & (delta_f > 0.0)) | (neg & (delta_f < 0.0))) & nonzero
    wrong = ((pos & (delta_f < 0.0)) | (neg & (delta_f > 0.0))) & nonzero
    pos_up = pos & (delta_f > 0.0)
    pos_down = pos & (delta_f < 0.0)
    neg_down = neg & (delta_f < 0.0)
    neg_up = neg & (delta_f > 0.0)
    correct_abs_sum = _abs_sum(delta_f, agree)
    wrong_abs_sum = _abs_sum(delta_f, wrong)
    pos_up_abs_sum = _abs_sum(delta_f, pos_up)
    pos_down_abs_sum = _abs_sum(delta_f, pos_down)
    neg_down_abs_sum = _abs_sum(delta_f, neg_down)
    neg_up_abs_sum = _abs_sum(delta_f, neg_up)
    credit_positive = credit > 0.0
    credit_negative = credit < 0.0
    credit_positive_abs_sum = _abs_sum(credit, credit_positive)
    credit_negative_abs_sum = _abs_sum(credit, credit_negative)
    nonzero_count = int(nonzero.sum().detach().cpu().item())
    return {
        "adv_positive_frac": float(pos.to(dtype=torch.float32).mean().detach().cpu().item()),
        "adv_negative_frac": float(neg.to(dtype=torch.float32).mean().detach().cpu().item()),
        "pos_adv_delta_positive_frac": _frac(delta_f > 0.0, pos),
        "neg_adv_delta_negative_frac": _frac(delta_f < 0.0, neg),
        "direction_agree_frac": float(
            agree.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(nonzero_count, 1))
        ),
        "direction_wrong_frac": float(
            wrong.to(dtype=torch.float32).sum().detach().cpu().item() / float(max(nonzero_count, 1))
        ),
        "delta_logprob_mean": float(delta_f.mean().detach().cpu().item()),
        "delta_logprob_abs_mean": float(delta_f.abs().mean().detach().cpu().item()),
        "delta_logprob_when_adv_positive_mean": _mean(delta_f, pos),
        "delta_logprob_when_adv_positive_p10": _quantile(delta_f, pos, 0.10),
        "delta_logprob_when_adv_positive_p50": _quantile(delta_f, pos, 0.50),
        "delta_logprob_when_adv_positive_p90": _quantile(delta_f, pos, 0.90),
        "delta_logprob_when_adv_negative_mean": _mean(delta_f, neg),
        "delta_logprob_when_adv_negative_p10": _quantile(delta_f, neg, 0.10),
        "delta_logprob_when_adv_negative_p50": _quantile(delta_f, neg, 0.50),
        "delta_logprob_when_adv_negative_p90": _quantile(delta_f, neg, 0.90),
        "pos_adv_delta_positive_abs_mean": _abs_mean(delta_f, pos_up),
        "pos_adv_delta_negative_abs_mean": _abs_mean(delta_f, pos_down),
        "pos_adv_delta_positive_abs_sum": float(pos_up_abs_sum),
        "pos_adv_delta_negative_abs_sum": float(pos_down_abs_sum),
        "pos_adv_up_abs_share": float(pos_up_abs_sum / max(pos_up_abs_sum + pos_down_abs_sum, 1.0e-12)),
        "neg_adv_delta_negative_abs_mean": _abs_mean(delta_f, neg_down),
        "neg_adv_delta_positive_abs_mean": _abs_mean(delta_f, neg_up),
        "neg_adv_delta_negative_abs_sum": float(neg_down_abs_sum),
        "neg_adv_delta_positive_abs_sum": float(neg_up_abs_sum),
        "neg_adv_down_abs_share": float(neg_down_abs_sum / max(neg_down_abs_sum + neg_up_abs_sum, 1.0e-12)),
        "direction_correct_abs_sum": float(correct_abs_sum),
        "direction_wrong_abs_sum": float(wrong_abs_sum),
        "direction_correct_abs_share": float(correct_abs_sum / max(correct_abs_sum + wrong_abs_sum, 1.0e-12)),
        "credit_mean": float(credit.mean().detach().cpu().item()),
        "credit_when_adv_positive_mean": _mean(credit, pos),
        "credit_when_adv_negative_mean": _mean(credit, neg),
        "credit_positive_abs_sum": float(credit_positive_abs_sum),
        "credit_negative_abs_sum": float(credit_negative_abs_sum),
        "credit_positive_abs_share": float(
            credit_positive_abs_sum / max(credit_positive_abs_sum + credit_negative_abs_sum, 1.0e-12)
        ),
    }



def _standardize_features(x: torch.Tensor) -> torch.Tensor:
    x_f = x.detach().to(dtype=torch.float32)
    if x_f.ndim == 1:
        x_f = x_f.unsqueeze(-1)
    finite = torch.isfinite(x_f)
    x_f = torch.where(finite, x_f, torch.zeros_like(x_f))
    if int(x_f.shape[0]) <= 1:
        return torch.zeros_like(x_f)
    mean = x_f.mean(dim=0, keepdim=True)
    std = x_f.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    return (x_f - mean) / std


def _pairwise_lower_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or int(matrix.shape[0]) <= 1:
        return torch.empty((0,), dtype=torch.float32, device=matrix.device)
    idx = torch.tril_indices(int(matrix.shape[0]), int(matrix.shape[1]), offset=-1, device=matrix.device)
    return matrix[idx[0], idx[1]]


def _pairwise_cosine_values(features: torch.Tensor) -> torch.Tensor:
    x = features.detach().to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if int(x.shape[0]) <= 1:
        return torch.empty((0,), dtype=torch.float32, device=x.device)
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1.0e-12)
    unit = x / norm
    return _pairwise_lower_values(unit @ unit.T)


def _pairwise_l1_values(features: torch.Tensor) -> torch.Tensor:
    x = features.detach().to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if int(x.shape[0]) <= 1:
        return torch.empty((0,), dtype=torch.float32, device=x.device)
    dist = (x[:, None, :] - x[None, :, :]).abs().mean(dim=-1)
    return _pairwise_lower_values(dist)


def _pairwise_l2_values(features: torch.Tensor) -> torch.Tensor:
    x = features.detach().to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if int(x.shape[0]) <= 1:
        return torch.empty((0,), dtype=torch.float32, device=x.device)
    dist = torch.linalg.vector_norm(x[:, None, :] - x[None, :, :], dim=-1) / max(float(x.shape[-1]) ** 0.5, 1.0)
    return _pairwise_lower_values(dist)


def _mean_where(values: torch.Tensor, mask: torch.Tensor) -> float:
    values_f = values.detach().to(dtype=torch.float32).reshape(-1)
    mask_b = mask.detach().to(dtype=torch.bool).reshape(-1)
    if int(mask_b.sum().item()) <= 0:
        return 0.0
    return float(values_f[mask_b].mean().item())


def _kmeans_labels(features: torch.Tensor, k: int = 3, iters: int = 25) -> torch.Tensor:
    x = _standardize_features(features.detach().to(dtype=torch.float32).cpu())
    n = int(x.shape[0])
    k_i = min(max(int(k), 1), n)
    if n <= 0:
        return torch.empty((0,), dtype=torch.long)
    if k_i <= 1:
        return torch.zeros((n,), dtype=torch.long)
    centers_idx = [0]
    for _ in range(1, k_i):
        center_stack = x[torch.as_tensor(centers_idx, dtype=torch.long)]
        dist = torch.cdist(x, center_stack).min(dim=1).values
        for used in centers_idx:
            dist[int(used)] = -1.0
        centers_idx.append(int(torch.argmax(dist).item()))
    centers = x[torch.as_tensor(centers_idx, dtype=torch.long)].clone()
    labels = torch.zeros((n,), dtype=torch.long)
    for _ in range(max(int(iters), 1)):
        labels = torch.cdist(x, centers).argmin(dim=1)
        new_centers = centers.clone()
        for cluster_id in range(k_i):
            mask = labels == cluster_id
            if int(mask.sum().item()) > 0:
                new_centers[cluster_id] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1.0e-5, rtol=1.0e-5):
            centers = new_centers
            break
        centers = new_centers
    return labels


def _cluster_gradient_summary(
    *,
    grad_matrix: torch.Tensor,
    cluster_features: torch.Tensor,
    push_features: torch.Tensor,
    action_features: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    if int(grad_matrix.numel()) <= 0 or int(grad_matrix.shape[0]) <= 1:
        return {
            f"{prefix}_cluster_count": 0,
            f"{prefix}_cluster_sizes": [],
        }
    g = grad_matrix.detach().to(dtype=torch.float32).cpu()
    row_norm = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1.0e-12)
    grad_cos = _pairwise_lower_values((g / row_norm) @ (g / row_norm).T)
    labels = _kmeans_labels(cluster_features, k=3)
    idx = torch.tril_indices(int(g.shape[0]), int(g.shape[0]), offset=-1)
    same = labels[idx[0]] == labels[idx[1]]
    different = ~same
    push_cos = _pairwise_cosine_values(push_features.detach().cpu())
    action_l1 = _pairwise_l1_values(action_features.detach().cpu())
    sizes = [int((labels == cluster_id).sum().item()) for cluster_id in range(int(labels.max().item()) + 1)]
    return {
        f"{prefix}_cluster_count": int(len(sizes)),
        f"{prefix}_cluster_sizes": sizes,
        f"{prefix}_within_pair_frac": float(same.to(dtype=torch.float32).mean().item()) if int(same.numel()) > 0 else 0.0,
        f"{prefix}_within_grad_cos_mean": _mean_where(grad_cos, same),
        f"{prefix}_between_grad_cos_mean": _mean_where(grad_cos, different),
        f"{prefix}_within_grad_neg_frac": _mean_where((grad_cos < 0.0).to(dtype=torch.float32), same),
        f"{prefix}_between_grad_neg_frac": _mean_where((grad_cos < 0.0).to(dtype=torch.float32), different),
        f"{prefix}_within_push_cos_mean": _mean_where(push_cos, same),
        f"{prefix}_between_push_cos_mean": _mean_where(push_cos, different),
        f"{prefix}_within_action_l1_mean": _mean_where(action_l1, same),
        f"{prefix}_between_action_l1_mean": _mean_where(action_l1, different),
    }


def _pairwise_feature_summary(
    *,
    grad_matrix: torch.Tensor,
    selected_adv: torch.Tensor,
    action_features: torch.Tensor,
    det_features: torch.Tensor,
    push_features: torch.Tensor,
    state_features: torch.Tensor,
    scalar_features: dict[str, torch.Tensor],
) -> dict[str, float]:
    if int(grad_matrix.numel()) <= 0 or int(grad_matrix.shape[0]) <= 1:
        return {"sample_count": float(int(grad_matrix.shape[0]) if grad_matrix.ndim >= 1 else 0)}
    g = grad_matrix.detach().to(dtype=torch.float32).cpu()
    row_norm = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1.0e-12)
    grad_cos = _pairwise_lower_values((g / row_norm) @ (g / row_norm).T)
    conflict = grad_cos < 0.0
    non_conflict = grad_cos >= 0.0

    action_l1 = _pairwise_l1_values(action_features.detach().cpu())
    det_l1 = _pairwise_l1_values(det_features.detach().cpu())
    push_cos = _pairwise_cosine_values(push_features.detach().cpu())
    state_l2 = _pairwise_l2_values(_standardize_features(state_features.detach().cpu()))
    adv_diff = _pairwise_l1_values(selected_adv.detach().cpu().reshape(-1, 1))

    summary: dict[str, float] = {
        "sample_count": float(int(g.shape[0])),
        "pair_count": float(int(grad_cos.numel())),
        "grad_cos_mean": float(grad_cos.mean().item()) if int(grad_cos.numel()) > 0 else 0.0,
        "grad_cos_neg_frac": float(conflict.to(dtype=torch.float32).mean().item()) if int(grad_cos.numel()) > 0 else 0.0,
        "grad_cos_vs_action_l1_corr": _safe_corr(grad_cos, action_l1),
        "grad_cos_vs_det_l1_corr": _safe_corr(grad_cos, det_l1),
        "grad_cos_vs_push_cos_corr": _safe_corr(grad_cos, push_cos),
        "grad_cos_vs_state_l2_corr": _safe_corr(grad_cos, state_l2),
        "grad_cos_vs_adv_diff_corr": _safe_corr(grad_cos, adv_diff),
        "action_l1_conflict_mean": _mean_where(action_l1, conflict),
        "action_l1_nonconflict_mean": _mean_where(action_l1, non_conflict),
        "det_l1_conflict_mean": _mean_where(det_l1, conflict),
        "det_l1_nonconflict_mean": _mean_where(det_l1, non_conflict),
        "push_cos_conflict_mean": _mean_where(push_cos, conflict),
        "push_cos_nonconflict_mean": _mean_where(push_cos, non_conflict),
        "state_l2_conflict_mean": _mean_where(state_l2, conflict),
        "state_l2_nonconflict_mean": _mean_where(state_l2, non_conflict),
        "adv_diff_conflict_mean": _mean_where(adv_diff, conflict),
        "adv_diff_nonconflict_mean": _mean_where(adv_diff, non_conflict),
    }

    batch_grad = g.mean(dim=0)
    batch_norm = torch.linalg.vector_norm(batch_grad).clamp_min(1.0e-12)
    row_batch_cos = (g @ batch_grad) / (row_norm.squeeze(1) * batch_norm)
    summary.update(
        {
            "row_selected_batch_cos_mean": float(row_batch_cos.mean().item()),
            "row_selected_batch_conflict_frac": float((row_batch_cos < 0.0).to(dtype=torch.float32).mean().item()),
        }
    )
    for feature_name, values in scalar_features.items():
        values_f = values.detach().to(dtype=torch.float32).cpu().reshape(-1)
        if int(values_f.numel()) != int(row_batch_cos.numel()):
            continue
        summary[f"row_batch_cos_vs_{feature_name}_corr"] = _safe_corr(row_batch_cos, values_f)
        summary[f"{feature_name}_row_conflict_mean"] = _mean_where(values_f, row_batch_cos < 0.0)
        summary[f"{feature_name}_row_nonconflict_mean"] = _mean_where(values_f, row_batch_cos >= 0.0)
    summary.update(
        _cluster_gradient_summary(
            grad_matrix=g,
            cluster_features=push_features.detach().cpu(),
            push_features=push_features.detach().cpu(),
            action_features=action_features.detach().cpu(),
            prefix="push",
        )
    )
    summary.update(
        _cluster_gradient_summary(
            grad_matrix=g,
            cluster_features=action_features.detach().cpu(),
            push_features=push_features.detach().cpu(),
            action_features=action_features.detach().cpu(),
            prefix="action",
        )
    )
    summary.update(
        _cluster_gradient_summary(
            grad_matrix=g,
            cluster_features=state_features.detach().cpu(),
            push_features=push_features.detach().cpu(),
            action_features=action_features.detach().cpu(),
            prefix="state",
        )
    )
    return summary

def _gradient_conflict_summary(
    *,
    grad_matrix: torch.Tensor,
    selected_adv: torch.Tensor,
    batch_grad: torch.Tensor,
) -> dict[str, float]:
    if int(grad_matrix.numel()) <= 0:
        return {
            "sample_count": 0.0,
            "mean_row_grad_norm": 0.0,
            "batch_grad_norm": 0.0,
            "cancel_ratio": 0.0,
            "row_batch_cos_mean": 0.0,
            "row_batch_cos_pos_adv_mean": 0.0,
            "row_batch_cos_neg_adv_mean": 0.0,
            "row_batch_conflict_frac": 0.0,
            "row_batch_conflict_pos_adv_frac": 0.0,
            "row_batch_conflict_neg_adv_frac": 0.0,
            "pairwise_cos_mean": 0.0,
            "pairwise_neg_cos_frac": 0.0,
            "pairwise_pos_neg_cos_mean": 0.0,
            "pairwise_pos_neg_neg_cos_frac": 0.0,
        }

    g = grad_matrix.detach().to(dtype=torch.float32).cpu()
    batch = batch_grad.detach().to(dtype=torch.float32).cpu().reshape(-1)
    adv_np = selected_adv.detach().to(dtype=torch.float32).cpu().reshape(-1).numpy()
    row_norm = torch.linalg.vector_norm(g, dim=1).clamp_min(1.0e-12)
    batch_norm = torch.linalg.vector_norm(batch).clamp_min(1.0e-12)
    row_batch_cos = (g @ batch) / (row_norm * batch_norm)
    pos_mask = torch.as_tensor(adv_np > 0.0, dtype=torch.bool)
    neg_mask = torch.as_tensor(adv_np < 0.0, dtype=torch.bool)

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.sum().item()) <= 0:
            return 0.0
        return float(values[mask].mean().item())

    def _masked_frac(mask: torch.Tensor, denom: torch.Tensor) -> float:
        denom_count = int(denom.sum().item())
        if denom_count <= 0:
            return 0.0
        return float((mask & denom).to(dtype=torch.float32).sum().item() / float(denom_count))

    unit = g / row_norm.unsqueeze(1)
    cos = unit @ unit.T
    n = int(cos.shape[0])
    pair_values: list[float] = []
    pair_neg_count = 0
    pos_neg_values: list[float] = []
    pos_neg_neg_count = 0
    pos_np = adv_np > 0.0
    neg_np = adv_np < 0.0
    for i in range(n):
        for j in range(i + 1, n):
            value = float(cos[i, j].item())
            pair_values.append(value)
            if value < 0.0:
                pair_neg_count += 1
            if (bool(pos_np[i]) and bool(neg_np[j])) or (bool(neg_np[i]) and bool(pos_np[j])):
                pos_neg_values.append(value)
                if value < 0.0:
                    pos_neg_neg_count += 1

    mean_grad = g.mean(dim=0)
    cancel_ratio = float(
        torch.linalg.vector_norm(mean_grad).item() / max(float(row_norm.mean().item()), 1.0e-12)
    )
    return {
        "sample_count": float(n),
        "mean_row_grad_norm": float(row_norm.mean().item()),
        "batch_grad_norm": float(batch_norm.item()),
        "cancel_ratio": float(cancel_ratio),
        "row_batch_cos_mean": float(row_batch_cos.mean().item()),
        "row_batch_cos_pos_adv_mean": _masked_mean(row_batch_cos, pos_mask),
        "row_batch_cos_neg_adv_mean": _masked_mean(row_batch_cos, neg_mask),
        "row_batch_conflict_frac": float((row_batch_cos < 0.0).to(dtype=torch.float32).mean().item()),
        "row_batch_conflict_pos_adv_frac": _masked_frac(row_batch_cos < 0.0, pos_mask),
        "row_batch_conflict_neg_adv_frac": _masked_frac(row_batch_cos < 0.0, neg_mask),
        "pairwise_cos_mean": float(np.mean(pair_values, dtype=np.float64)) if pair_values else 0.0,
        "pairwise_neg_cos_frac": float(pair_neg_count / max(len(pair_values), 1)),
        "pairwise_pos_neg_cos_mean": float(np.mean(pos_neg_values, dtype=np.float64)) if pos_neg_values else 0.0,
        "pairwise_pos_neg_neg_cos_frac": float(pos_neg_neg_count / max(len(pos_neg_values), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--actor_path", type=str, required=True)
    parser.add_argument("--critic_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--rollout_env_steps", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--vec_backend", type=str, default="sync")
    parser.add_argument("--structured_env_backend", type=str, default="")
    parser.add_argument("--structured_env_tensor_backend", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--grad_sample_rows", type=int, default=64)
    parser.add_argument("--grad_sample_seed", type=int, default=99001)
    parser.add_argument("--single_control_rows", type=int, default=12)
    parser.add_argument("--conflict_rows", type=int, default=64)
    parser.add_argument("--one_step_lr", type=float, default=None)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    _set_all_seeds(int(args.seed))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    if str(args.structured_env_backend).strip():
        setattr(cfg, "structured_env_backend", str(args.structured_env_backend).strip())
    if str(args.structured_env_tensor_backend).strip():
        setattr(cfg, "structured_env_tensor_backend", str(args.structured_env_tensor_backend).strip())
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    _load_module_checkpoint(actor, str(args.actor_path), state_key="actor", map_location=device, strict=True)
    _load_module_checkpoint(critic, str(args.critic_path), state_key="critic", map_location=device, strict=True)
    actor.train()
    critic.eval()

    trainer = StructuredMAPPO(
        actor,
        critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(cfg.clip_ratio),
        value_coef=float(cfg.value_coef),
        entropy_coef=float(cfg.entropy_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        num_mini_batch=int(cfg.num_mini_batch),
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        joint_stage_updates=False,
        cfg=cfg,
        train_accel=bool(getattr(cfg, "train_accel", True)),
        train_sat=bool(getattr(cfg, "train_sat", True)),
        train_bw=bool(getattr(cfg, "train_bw", True)),
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy")),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy")),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy")),
    )

    env_group = make_structured_env_group(
        cfg,
        num_envs=max(int(args.num_envs), 1),
        backend=str(args.vec_backend),
        mode="script",
    )
    buffer = StructuredRolloutBuffer()
    try:
        drivers = as_structured_drivers(env_group)
        actual_num_envs = len(drivers)
        reset_many = getattr(drivers, "reset_many", None)
        if not callable(reset_many):
            raise RuntimeError("diagnostic rollout requires a structured driver group with reset_many")
        reset_many([int(args.seed) + env_index for env_index in range(actual_num_envs)])
        collect_horizon = getattr(trainer, "collect_env_horizon_native_tensor_policy", None)
        begin_native_rollout = getattr(trainer, "begin_native_rollout", None)
        if _is_native_driver_group(drivers) and callable(begin_native_rollout) and callable(collect_horizon):
            begin_native_rollout(
                drivers,
                rollout_env_steps=int(args.rollout_env_steps),
                num_envs=int(actual_num_envs),
            )
            collect_horizon(
                drivers,
                buffer=buffer,
                horizon=int(args.rollout_env_steps),
                deterministic=False,
            )
        else:
            for _ in range(int(args.rollout_env_steps)):
                trainer.collect_env_steps(drivers, buffer, deterministic=False)
        bootstrap_world_state = buffer.build_bootstrap_view()
        if bootstrap_world_state is not None and getattr(bootstrap_world_state, "next_world_batch", None) is not None:
            bootstrap_world_state = type(bootstrap_world_state)(
                env_indices=bootstrap_world_state.env_indices,
                next_world_batch=_to_device_dataclass(bootstrap_world_state.next_world_batch, device),
            )
        gae = trainer.compute_returns_and_advantages(buffer, bootstrap_world_state)
    finally:
        close_structured_env_group(env_group)

    rollout_views = buffer.build_rollout_views(device)
    bw_stage_batch = rollout_views.training_view.stage_batches.get(2)
    if bw_stage_batch is None or int(bw_stage_batch.num_samples) <= 0:
        raise RuntimeError("No BW transitions collected.")
    bw_transition_idx = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64).reshape(-1)
    advantages_np = np.asarray(gae["advantages"], dtype=np.float32).reshape(-1)
    if int(bw_transition_idx.size) != int(bw_stage_batch.num_samples):
        raise RuntimeError("BW transition index count does not match BW stage sample count.")
    bw_adv = torch.from_numpy(advantages_np[bw_transition_idx]).to(device=device, dtype=torch.float32)
    if bw_adv.numel() > 1 and bool(getattr(cfg, "stagewise_advantage_norm_enabled", False)):
        bw_adv = (bw_adv - bw_adv.mean()) / bw_adv.std(unbiased=False).clamp_min(1.0e-8)
    elif bw_adv.numel() > 1:
        bw_adv = (bw_adv - bw_adv.mean()) / bw_adv.std(unbiased=False).clamp_min(1.0e-8)

    bw_local_batch = bw_stage_batch.local_batch
    bw_world_batch = bw_stage_batch.world_batch
    joint_actions = bw_stage_batch.actions.to(device)
    old_logprobs = bw_stage_batch.old_logprobs.to(device)

    num_samples = int(joint_actions.shape[0])
    num_agents = int(joint_actions.shape[1])
    flat_local_batch = bw_local_batch
    new_logprob, entropy, actor_out = trainer._stage_actor_eval_from_batch(
        2,
        flat_local_batch,
        joint_actions,
        num_agents,
    )
    cache = {
        "advantages": bw_adv,
        "old_logprobs": old_logprobs,
        "num_agents": num_agents,
        "local_batch": bw_local_batch,
        "flat_indices": torch.arange(
            num_samples * num_agents,
            device=device,
            dtype=torch.long,
        ).reshape(num_samples, num_agents),
    }
    (
        ppo_policy_loss,
        entropy_mean,
        approx_kl,
        clip_frac,
        corr_valid,
        corr_latent,
        kappa_mean,
        kappa_p10,
        kappa_p90,
        kappa_hi_frac,
    ) = trainer._stage_policy_terms(
        2,
        actor_out=actor_out,
        cache=cache,
        mb_rel=torch.arange(num_samples, device=device, dtype=torch.long),
        joint_actions_mb=joint_actions,
        new_logprob=new_logprob,
        entropy=entropy,
    )

    valid_mask = (bw_local_batch.gu_mask > 0.5) & (bw_local_batch.bw_valid_mask > 0.5)
    teacher_action = _teacher_bw_from_local_state(
        bw_local_batch,
        assoc_bonus=float(getattr(cfg, "baseline_assoc_bonus", 0.3) or 0.0),
    )
    det_action = actor.act_bw(bw_local_batch, deterministic=True).action
    det_actor_out = actor.evaluate_bw(bw_local_batch, det_action)
    det_weighted_logprob_loss = -(bw_adv * det_actor_out.logprob.reshape(num_samples, num_agents).sum(dim=1)).mean()
    imitation_loss = _masked_simplex_kl(teacher_action, det_action, valid_mask).mean()
    imitation_l1 = (
        (det_action - teacher_action).abs() * valid_mask.to(dtype=det_action.dtype)
    ).sum(dim=-1).mean()
    sample_teacher_l1_per_row = (
        (joint_actions.reshape(num_samples * num_agents, -1) - teacher_action).abs()
        * valid_mask.to(dtype=det_action.dtype)
    ).sum(dim=-1)
    sample_teacher_l1_per_sample = sample_teacher_l1_per_row.reshape(num_samples, num_agents).mean(dim=1)
    sample_teacher_l1 = sample_teacher_l1_per_sample.mean()
    positive_adv = bw_adv > 0.0
    negative_adv = bw_adv < 0.0

    bw_params = [param for param in actor.bw_policy.parameters() if param.requires_grad]
    score_head_params = [param for param in actor.bw_policy.score_head.parameters() if param.requires_grad]
    grad_ppo_all = _flatten_grads(ppo_policy_loss, bw_params)
    grad_detlogprob_all = _flatten_grads(det_weighted_logprob_loss, bw_params)
    grad_imitation_all = _flatten_grads(imitation_loss, bw_params)
    grad_ppo_score = _flatten_grads(ppo_policy_loss, score_head_params)
    grad_detlogprob_score = _flatten_grads(det_weighted_logprob_loss, score_head_params)
    grad_imitation_score = _flatten_grads(imitation_loss, score_head_params)

    fallback_bw_lr = float(getattr(cfg, "stage_mcgae_bw_actor_lr", getattr(cfg, "actor_lr", 3.0e-4)) or 3.0e-4)
    one_step_lr = (
        float(args.one_step_lr)
        if args.one_step_lr is not None
        else _checkpoint_stage_actor_lr(str(args.actor_path), 2, fallback_bw_lr)
    )
    optimizer_state = _checkpoint_stage_actor_optimizer_state(str(args.actor_path), 2)

    def _logprob_for_actor(actor_model: Any) -> tuple[torch.Tensor, Any]:
        actor_out_model = actor_model.evaluate_bw(
            bw_local_batch,
            joint_actions.reshape(num_samples * num_agents, -1),
        )
        logprob_model = actor_out_model.logprob.reshape(num_samples, num_agents).sum(dim=1)
        return logprob_model, actor_out_model

    def _ppo_loss_for_actor(actor_model: Any) -> tuple[torch.Tensor, Any]:
        new_logprob_model, actor_out_model = _logprob_for_actor(actor_model)
        entropy_model = actor_out_model.entropy.reshape(num_samples, num_agents).sum(dim=1)
        policy_loss_model, *_rest = trainer._stage_policy_terms(
            2,
            actor_out=actor_out_model,
            cache=cache,
            mb_rel=torch.arange(num_samples, device=device, dtype=torch.long),
            joint_actions_mb=joint_actions,
            new_logprob=new_logprob_model,
            entropy=entropy_model,
        )
        return policy_loss_model, actor_out_model

    def _subset_ppo_loss_for_actor(
        actor_model: Any,
        selected_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logprob_model, _actor_out_model = _logprob_for_actor(actor_model)
        selected_logprob = logprob_model.index_select(0, selected_idx)
        selected_old_logprob = old_logprobs.index_select(0, selected_idx).detach()
        selected_adv = bw_adv.index_select(0, selected_idx).detach()
        log_ratio = torch.clamp(selected_logprob - selected_old_logprob, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - float(trainer.clip_ratio), 1.0 + float(trainer.clip_ratio))
        policy_loss = -torch.minimum(ratio * selected_adv, clipped * selected_adv).mean()
        return policy_loss, logprob_model

    def _new_bw_step_actor(
        *,
        load_optimizer_state: bool = True,
    ) -> tuple[Any, list[torch.nn.Parameter], torch.optim.Optimizer, bool, str]:
        actor_model = copy.deepcopy(actor).to(device)
        actor_model.train()
        params = [param for param in actor_model.bw_policy.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params, lr=float(one_step_lr))
        optimizer_state_loaded = False
        optimizer_state_error = ""
        if load_optimizer_state and optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                for group in optimizer.param_groups:
                    group["lr"] = float(one_step_lr)
                _move_optimizer_state_to_device(optimizer, device)
                optimizer_state_loaded = True
            except Exception as exc:
                optimizer_state_error = str(exc)
        for group in optimizer.param_groups:
            group["lr"] = float(one_step_lr)
        return actor_model, params, optimizer, optimizer_state_loaded, optimizer_state_error

    def _selected_one_step_control(selected_idx: torch.Tensor) -> dict[str, Any]:
        selected_idx = selected_idx.to(device=device, dtype=torch.long).reshape(-1)
        if int(selected_idx.numel()) <= 0:
            zero_delta = torch.zeros_like(new_logprob.detach())
            return {
                "selected_count": 0,
                "optimizer_state_loaded": False,
                "optimizer_state_error": "empty selected set",
                "loss_before": 0.0,
                "loss_after": 0.0,
                "selected_alignment": _direction_stats(torch.empty(0, device=device), torch.empty(0, device=device)),
                "all_alignment": _direction_stats(bw_adv, zero_delta),
            }

        actor_model, params, optimizer, opt_loaded, opt_error = _new_bw_step_actor()
        loss_before, _ = _subset_ppo_loss_for_actor(actor_model, selected_idx)
        optimizer.zero_grad(set_to_none=True)
        loss_before.backward()
        torch.nn.utils.clip_grad_norm_(params, float(getattr(cfg, "max_grad_norm", 0.5) or 0.5))
        optimizer.step()
        loss_after, logprob_after = _subset_ppo_loss_for_actor(actor_model, selected_idx)
        with torch.no_grad():
            delta_logprob = logprob_after.detach() - new_logprob.detach()
            selected_delta = delta_logprob.index_select(0, selected_idx)
            selected_adv = bw_adv.index_select(0, selected_idx)
        return {
            "selected_count": int(selected_idx.numel()),
            "optimizer_state_loaded": bool(opt_loaded),
            "optimizer_state_error": str(opt_error),
            "loss_before": float(loss_before.detach().item()),
            "loss_after": float(loss_after.detach().item()),
            "selected_alignment": _direction_stats(selected_adv, selected_delta),
            "all_alignment": _direction_stats(bw_adv, delta_logprob),
        }

    def _single_row_self_control(
        candidate_idx: torch.Tensor,
        *,
        max_rows: int,
        load_optimizer_state: bool,
    ) -> dict[str, Any]:
        candidate_idx = candidate_idx.to(device=device, dtype=torch.long).reshape(-1)
        max_rows = max(int(max_rows), 0)
        if max_rows > 0 and int(candidate_idx.numel()) > max_rows:
            candidate_idx = candidate_idx[:max_rows]
        deltas: list[torch.Tensor] = []
        advs: list[torch.Tensor] = []
        opt_loaded_count = 0
        opt_error_count = 0
        for row_idx in candidate_idx:
            selected_idx = row_idx.reshape(1)
            actor_model, params, optimizer, opt_loaded, opt_error = _new_bw_step_actor(
                load_optimizer_state=load_optimizer_state,
            )
            opt_loaded_count += int(bool(opt_loaded))
            opt_error_count += int(bool(opt_error))
            loss_before, _ = _subset_ppo_loss_for_actor(actor_model, selected_idx)
            optimizer.zero_grad(set_to_none=True)
            loss_before.backward()
            torch.nn.utils.clip_grad_norm_(params, float(getattr(cfg, "max_grad_norm", 0.5) or 0.5))
            optimizer.step()
            with torch.no_grad():
                logprob_after, _ = _logprob_for_actor(actor_model)
                deltas.append((logprob_after.index_select(0, selected_idx) - new_logprob.detach().index_select(0, selected_idx)).reshape(()))
                advs.append(bw_adv.index_select(0, selected_idx).detach().reshape(()))
        if not deltas:
            adv_tensor = torch.empty(0, device=device)
            delta_tensor = torch.empty(0, device=device)
        else:
            adv_tensor = torch.stack(advs).to(device=device)
            delta_tensor = torch.stack(deltas).to(device=device)
        return {
            "selected_count": int(candidate_idx.numel()),
            "optimizer_state_requested": bool(load_optimizer_state),
            "optimizer_state_loaded_count": int(opt_loaded_count),
            "optimizer_state_error_count": int(opt_error_count),
            "self_alignment": _direction_stats(adv_tensor, delta_tensor),
        }

    actor_step, actor_step_params, actor_step_optimizer, one_step_optimizer_state_loaded, one_step_optimizer_state_error = _new_bw_step_actor()

    det_before = actor_step.act_bw(bw_local_batch, deterministic=True).action
    det_teacher_l1_before = _masked_l1_per_row(det_before, teacher_action, valid_mask)
    det_teacher_kl_before = _masked_simplex_kl(teacher_action, det_before, valid_mask)
    ppo_step_loss_before, _ = _ppo_loss_for_actor(actor_step)

    actor_step_optimizer.zero_grad(set_to_none=True)
    ppo_step_loss_before.backward()
    torch.nn.utils.clip_grad_norm_(actor_step_params, float(getattr(cfg, "max_grad_norm", 0.5) or 0.5))
    actor_step_optimizer.step()

    det_after = actor_step.act_bw(bw_local_batch, deterministic=True).action
    det_teacher_l1_after = _masked_l1_per_row(det_after, teacher_action, valid_mask)
    det_teacher_kl_after = _masked_simplex_kl(teacher_action, det_after, valid_mask)
    ppo_step_loss_after, _ = _ppo_loss_for_actor(actor_step)
    with torch.no_grad():
        actor_step_out_after = actor_step.evaluate_bw(
            bw_local_batch,
            joint_actions.reshape(num_samples * num_agents, -1),
        )
        one_step_logprob_after = actor_step_out_after.logprob.reshape(num_samples, num_agents).sum(dim=1)
        one_step_delta_logprob = one_step_logprob_after - new_logprob.detach()
    pos_idx = torch.nonzero(positive_adv, as_tuple=False).reshape(-1)
    neg_idx = torch.nonzero(negative_adv, as_tuple=False).reshape(-1)
    pos_only_control = _selected_one_step_control(pos_idx)
    neg_only_control = _selected_one_step_control(neg_idx)
    pos_single_cold_control = _single_row_self_control(
        pos_idx,
        max_rows=int(args.single_control_rows),
        load_optimizer_state=False,
    )
    neg_single_cold_control = _single_row_self_control(
        neg_idx,
        max_rows=int(args.single_control_rows),
        load_optimizer_state=False,
    )
    pos_single_adam_control = _single_row_self_control(
        pos_idx,
        max_rows=int(args.single_control_rows),
        load_optimizer_state=True,
    )
    neg_single_adam_control = _single_row_self_control(
        neg_idx,
        max_rows=int(args.single_control_rows),
        load_optimizer_state=True,
    )

    det_delta = det_after - det_before
    teacher_delta = teacher_action - det_before
    improved_mask = det_teacher_l1_after < det_teacher_l1_before
    worsened_mask = det_teacher_l1_after > det_teacher_l1_before

    action_flat = joint_actions.reshape(num_samples * num_agents, -1).detach()
    det_flat = det_action.detach()
    valid_flat = valid_mask.to(dtype=action_flat.dtype)
    action_features = (action_flat * valid_flat).reshape(num_samples, num_agents, -1).reshape(num_samples, -1)
    det_features = (det_flat * valid_flat).reshape(num_samples, num_agents, -1).reshape(num_samples, -1)
    push_features = ((action_flat - det_flat) * valid_flat).reshape(num_samples, num_agents, -1).reshape(num_samples, -1)
    ego_features = bw_local_batch.ego_features.detach().reshape(num_samples, num_agents, -1).reshape(num_samples, -1)
    sat_tokens = bw_local_batch.selected_sat_tokens.detach()
    sat_mask = bw_local_batch.selected_sat_mask.detach().to(dtype=sat_tokens.dtype).unsqueeze(-1)
    sat_features = (sat_tokens * sat_mask).reshape(num_samples, num_agents, -1).reshape(num_samples, -1)
    gu_tokens = bw_local_batch.gu_tokens.detach()
    gu_features = (gu_tokens * valid_mask.to(dtype=gu_tokens.dtype).unsqueeze(-1)).reshape(
        num_samples,
        num_agents,
        -1,
    ).reshape(num_samples, -1)
    state_features = torch.cat([ego_features, sat_features, gu_features], dim=-1)
    sample_det_l1_per_row = ((action_flat - det_flat).abs() * valid_flat).sum(dim=-1)
    sample_det_l1_per_sample = sample_det_l1_per_row.reshape(num_samples, num_agents).mean(dim=1)
    sample_action_hhi = ((action_flat.pow(2)) * valid_flat).sum(dim=-1).reshape(num_samples, num_agents).mean(dim=1)
    det_action_hhi = ((det_flat.pow(2)) * valid_flat).sum(dim=-1).reshape(num_samples, num_agents).mean(dim=1)

    def _metric_per_sample(value: torch.Tensor) -> torch.Tensor:
        value_f = value.detach().to(dtype=torch.float32).reshape(-1)
        if int(value_f.numel()) == int(num_samples * num_agents):
            return value_f.reshape(num_samples, num_agents).mean(dim=1)
        if int(value_f.numel()) == int(num_samples):
            return value_f.reshape(num_samples)
        if int(value_f.numel()) % int(num_samples) == 0:
            return value_f.reshape(num_samples, -1).mean(dim=1)
        return value_f.new_full((num_samples,), float(value_f.mean().item()) if int(value_f.numel()) > 0 else 0.0)

    entropy_per_sample = _metric_per_sample(entropy)
    kappa_per_sample = _metric_per_sample(actor_out.kappa)
    tau_per_sample = _metric_per_sample(actor_out.tau)
    valid_count_per_sample = _metric_per_sample(actor_out.valid_count)

    def _row_grad_matrix_for_indices(selected_idx: torch.Tensor) -> torch.Tensor:
        selected_idx = selected_idx.to(device=device, dtype=torch.long).reshape(-1)
        if int(selected_idx.numel()) <= 0:
            return torch.zeros((0, 0), dtype=torch.float32)
        per_row: list[torch.Tensor] = []
        for row_idx_t in selected_idx:
            row_idx = int(row_idx_t.detach().cpu().item())
            log_ratio_i = torch.clamp(new_logprob[row_idx] - old_logprobs[row_idx].detach(), min=-20.0, max=20.0)
            ratio_i = torch.exp(log_ratio_i)
            clipped_i = torch.clamp(ratio_i, 1.0 - float(trainer.clip_ratio), 1.0 + float(trainer.clip_ratio))
            adv_i = bw_adv[row_idx].detach()
            loss_i = -torch.minimum(ratio_i * adv_i, clipped_i * adv_i)
            per_row.append(_flatten_grads(loss_i, bw_params).detach().cpu())
        return torch.stack(per_row, dim=0) if per_row else torch.zeros((0, 0), dtype=torch.float32)

    conflict_rows = max(int(args.conflict_rows), 0)
    pos_conflict_idx = pos_idx[:conflict_rows] if conflict_rows > 0 else pos_idx[:0]
    neg_conflict_idx = neg_idx[:conflict_rows] if conflict_rows > 0 else neg_idx[:0]
    scalar_feature_map = {
        "adv_abs": bw_adv.detach().abs(),
        "sample_det_l1": sample_det_l1_per_sample.detach(),
        "sample_teacher_l1": sample_teacher_l1_per_sample.detach(),
        "sample_action_hhi": sample_action_hhi.detach(),
        "det_action_hhi": det_action_hhi.detach(),
        "entropy": entropy_per_sample.detach(),
        "kappa": kappa_per_sample.detach(),
        "tau": tau_per_sample.detach(),
        "valid_count": valid_count_per_sample.detach(),
    }

    agent_action_features = action_flat * valid_flat
    agent_det_features = det_flat * valid_flat
    agent_push_features = (action_flat - det_flat) * valid_flat
    agent_sat_features = (sat_tokens * sat_mask).reshape(num_samples * num_agents, -1)
    agent_gu_features = (gu_tokens * valid_mask.to(dtype=gu_tokens.dtype).unsqueeze(-1)).reshape(
        num_samples * num_agents,
        -1,
    )
    agent_state_features = torch.cat(
        [bw_local_batch.ego_features.detach(), agent_sat_features, agent_gu_features],
        dim=-1,
    )
    agent_adv = bw_adv.detach().repeat_interleave(num_agents)
    agent_ids = torch.arange(num_agents, device=device, dtype=torch.long).repeat(num_samples)

    def _metric_per_agent(value: torch.Tensor) -> torch.Tensor:
        value_f = value.detach().to(dtype=torch.float32).reshape(-1)
        if int(value_f.numel()) == int(num_samples * num_agents):
            return value_f.reshape(num_samples * num_agents)
        if int(value_f.numel()) == int(num_samples):
            return value_f.reshape(num_samples).repeat_interleave(num_agents)
        if int(value_f.numel()) % int(num_samples * num_agents) == 0:
            return value_f.reshape(num_samples * num_agents, -1).mean(dim=1)
        if int(value_f.numel()) % int(num_samples) == 0:
            return value_f.reshape(num_samples, -1).mean(dim=1).repeat_interleave(num_agents)
        fallback = float(value_f.mean().item()) if int(value_f.numel()) > 0 else 0.0
        return value_f.new_full((num_samples * num_agents,), fallback)

    agent_scalar_feature_map = {
        "adv_abs": agent_adv.abs(),
        "sample_det_l1": sample_det_l1_per_row.detach(),
        "sample_teacher_l1": sample_teacher_l1_per_row.detach(),
        "sample_action_hhi": ((action_flat.pow(2)) * valid_flat).sum(dim=-1).detach(),
        "det_action_hhi": ((det_flat.pow(2)) * valid_flat).sum(dim=-1).detach(),
        "entropy": _metric_per_agent(actor_out.entropy),
        "kappa": _metric_per_agent(actor_out.kappa),
        "tau": _metric_per_agent(actor_out.tau),
        "valid_count": _metric_per_agent(actor_out.valid_count),
        "agent_id": agent_ids.to(dtype=torch.float32),
    }
    agent_logprob_flat = actor_out.logprob.reshape(-1)
    if int(agent_logprob_flat.numel()) != int(num_samples * num_agents):
        raise RuntimeError(
            "Per-agent BW conflict diagnostic requires actor_out.logprob to be flat agent-level logprobs."
        )

    def _agent_flat_indices_from_samples(selected_idx: torch.Tensor) -> torch.Tensor:
        selected_idx = selected_idx.to(device=device, dtype=torch.long).reshape(-1)
        if int(selected_idx.numel()) <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        offsets = torch.arange(num_agents, device=device, dtype=torch.long).reshape(1, num_agents)
        return (selected_idx.reshape(-1, 1) * num_agents + offsets).reshape(-1)

    def _agent_grad_matrix_for_flat_indices(selected_flat_idx: torch.Tensor) -> torch.Tensor:
        selected_flat_idx = selected_flat_idx.to(device=device, dtype=torch.long).reshape(-1)
        if int(selected_flat_idx.numel()) <= 0:
            return torch.zeros((0, 0), dtype=torch.float32)
        per_agent: list[torch.Tensor] = []
        for flat_idx_t in selected_flat_idx:
            flat_idx = int(flat_idx_t.detach().cpu().item())
            sample_idx_i = flat_idx // num_agents
            loss_i = -(bw_adv[sample_idx_i].detach() * agent_logprob_flat[flat_idx])
            per_agent.append(_flatten_grads(loss_i, bw_params).detach().cpu())
        return torch.stack(per_agent, dim=0) if per_agent else torch.zeros((0, 0), dtype=torch.float32)

    def _agent_conflict_attribution_for_flat_indices(selected_flat_idx: torch.Tensor) -> dict[str, Any]:
        grad_matrix = _agent_grad_matrix_for_flat_indices(selected_flat_idx)
        selected_flat_cpu = selected_flat_idx.detach().cpu().to(dtype=torch.long).reshape(-1)
        scalar_selected = {
            key: value.detach().cpu().index_select(0, selected_flat_cpu)
            for key, value in agent_scalar_feature_map.items()
        }
        return _pairwise_feature_summary(
            grad_matrix=grad_matrix,
            selected_adv=agent_adv.detach().cpu().index_select(0, selected_flat_cpu),
            action_features=agent_action_features.detach().cpu().index_select(0, selected_flat_cpu),
            det_features=agent_det_features.detach().cpu().index_select(0, selected_flat_cpu),
            push_features=agent_push_features.detach().cpu().index_select(0, selected_flat_cpu),
            state_features=agent_state_features.detach().cpu().index_select(0, selected_flat_cpu),
            scalar_features=scalar_selected,
        )

    def _agent_id_conflict_attribution(selected_flat_idx: torch.Tensor) -> dict[str, Any]:
        result: dict[str, Any] = {}
        selected_flat_idx = selected_flat_idx.to(device=device, dtype=torch.long).reshape(-1)
        selected_agent_ids = agent_ids.index_select(0, selected_flat_idx) if int(selected_flat_idx.numel()) > 0 else agent_ids[:0]
        for agent_id_i in range(num_agents):
            mask = selected_agent_ids == int(agent_id_i)
            result[f"agent_{agent_id_i}"] = _agent_conflict_attribution_for_flat_indices(selected_flat_idx[mask])
        return result

    def _conflict_attribution_for_indices(selected_idx: torch.Tensor) -> dict[str, float]:
        grad_matrix = _row_grad_matrix_for_indices(selected_idx)
        selected_idx_cpu = selected_idx.detach().cpu().to(dtype=torch.long).reshape(-1)
        scalar_selected = {
            key: value.detach().cpu().index_select(0, selected_idx_cpu)
            for key, value in scalar_feature_map.items()
        }
        return _pairwise_feature_summary(
            grad_matrix=grad_matrix,
            selected_adv=bw_adv.detach().cpu().index_select(0, selected_idx_cpu),
            action_features=action_features.detach().cpu().index_select(0, selected_idx_cpu),
            det_features=det_features.detach().cpu().index_select(0, selected_idx_cpu),
            push_features=push_features.detach().cpu().index_select(0, selected_idx_cpu),
            state_features=state_features.detach().cpu().index_select(0, selected_idx_cpu),
            scalar_features=scalar_selected,
        )

    positive_conflict_attribution = _conflict_attribution_for_indices(pos_conflict_idx)
    negative_conflict_attribution = _conflict_attribution_for_indices(neg_conflict_idx)
    pos_agent_flat_idx = _agent_flat_indices_from_samples(pos_conflict_idx)
    neg_agent_flat_idx = _agent_flat_indices_from_samples(neg_conflict_idx)
    positive_agent_conflict_attribution = _agent_conflict_attribution_for_flat_indices(pos_agent_flat_idx)
    negative_agent_conflict_attribution = _agent_conflict_attribution_for_flat_indices(neg_agent_flat_idx)
    positive_agent_id_conflict_attribution = _agent_id_conflict_attribution(pos_agent_flat_idx)
    negative_agent_id_conflict_attribution = _agent_id_conflict_attribution(neg_agent_flat_idx)

    grad_conflict: dict[str, float] = {}
    grad_sample_rows = min(max(int(args.grad_sample_rows), 0), int(num_samples))
    if grad_sample_rows > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.grad_sample_seed))
        sample_idx = torch.randperm(num_samples, device=device, generator=generator)[:grad_sample_rows]
        per_row_grads: list[torch.Tensor] = []
        for row_idx_t in sample_idx:
            row_idx = int(row_idx_t.detach().cpu().item())
            log_ratio_i = torch.clamp(new_logprob[row_idx] - old_logprobs[row_idx].detach(), min=-20.0, max=20.0)
            ratio_i = torch.exp(log_ratio_i)
            clipped_i = torch.clamp(ratio_i, 1.0 - float(trainer.clip_ratio), 1.0 + float(trainer.clip_ratio))
            adv_i = bw_adv[row_idx].detach()
            loss_i = -torch.minimum(ratio_i * adv_i, clipped_i * adv_i)
            per_row_grads.append(_flatten_grads(loss_i, bw_params).detach().cpu())
        grad_matrix = torch.stack(per_row_grads, dim=0) if per_row_grads else torch.zeros((0, 0), dtype=torch.float32)
        grad_conflict = _gradient_conflict_summary(
            grad_matrix=grad_matrix,
            selected_adv=bw_adv.index_select(0, sample_idx).detach(),
            batch_grad=grad_ppo_all.detach(),
        )

    summary = {
        "config": str(args.config),
        "actor_path": str(args.actor_path),
        "critic_path": str(args.critic_path),
        "num_bw_samples": int(num_samples),
        "ppo_policy_loss": float(ppo_policy_loss.detach().item()),
        "det_weighted_logprob_loss": float(det_weighted_logprob_loss.detach().item()),
        "imitation_loss": float(imitation_loss.detach().item()),
        "imitation_l1": float(imitation_l1.detach().item()),
        "sample_teacher_l1": float(sample_teacher_l1.detach().item()),
        "adv_teacher_l1_corr": _safe_corr(bw_adv, sample_teacher_l1_per_sample),
        "sample_teacher_l1_pos_adv": (
            float(sample_teacher_l1_per_sample[positive_adv].mean().item()) if torch.any(positive_adv) else 0.0
        ),
        "sample_teacher_l1_neg_adv": (
            float(sample_teacher_l1_per_sample[negative_adv].mean().item()) if torch.any(negative_adv) else 0.0
        ),
        "approx_kl_bw": float(approx_kl.detach().item()),
        "clip_frac_bw": float(clip_frac.detach().item()),
        "entropy_bw": float(entropy_mean.detach().item()),
        "bw_kappa_mean": float(kappa_mean.detach().item()),
        "bw_kappa_p10": float(kappa_p10.detach().item()),
        "bw_kappa_p90": float(kappa_p90.detach().item()),
        "bw_kappa_hi_frac": float(kappa_hi_frac.detach().item()),
        "bw_abs_log_ratio_corr_valid_count": float(corr_valid.detach().item()),
        "bw_abs_log_ratio_corr_latent_count": float(corr_latent.detach().item()),
        "grad_cosine_all": _cosine(grad_ppo_all, grad_imitation_all),
        "grad_cosine_score_head": _cosine(grad_ppo_score, grad_imitation_score),
        "grad_cosine_detlogprob_all": _cosine(grad_detlogprob_all, grad_imitation_all),
        "grad_cosine_detlogprob_score_head": _cosine(grad_detlogprob_score, grad_imitation_score),
        "grad_norm_ppo_all": float(grad_ppo_all.norm().item()),
        "grad_norm_detlogprob_all": float(grad_detlogprob_all.norm().item()),
        "grad_norm_imitation_all": float(grad_imitation_all.norm().item()),
        "grad_norm_ppo_score_head": float(grad_ppo_score.norm().item()),
        "grad_norm_detlogprob_score_head": float(grad_detlogprob_score.norm().item()),
        "grad_norm_imitation_score_head": float(grad_imitation_score.norm().item()),
        "grad_abs_mean_ppo_all": _mean_abs(grad_ppo_all),
        "grad_abs_mean_imitation_all": _mean_abs(grad_imitation_all),
        "one_step_det_teacher_l1_before": float(det_teacher_l1_before.mean().item()),
        "one_step_det_teacher_l1_after": float(det_teacher_l1_after.mean().item()),
        "one_step_det_teacher_kl_before": float(det_teacher_kl_before.mean().item()),
        "one_step_det_teacher_kl_after": float(det_teacher_kl_after.mean().item()),
        "one_step_det_teacher_l1_delta": float((det_teacher_l1_after - det_teacher_l1_before).mean().item()),
        "one_step_det_teacher_kl_delta": float((det_teacher_kl_after - det_teacher_kl_before).mean().item()),
        "one_step_det_improved_frac": float(improved_mask.to(dtype=torch.float32).mean().item()),
        "one_step_det_worsened_frac": float(worsened_mask.to(dtype=torch.float32).mean().item()),
        "one_step_det_toward_teacher_cosine": _mean_masked_cosine(det_delta, teacher_delta, valid_mask),
        "one_step_lr": float(one_step_lr),
        "one_step_optimizer_state_loaded": bool(one_step_optimizer_state_loaded),
        "one_step_optimizer_state_error": str(one_step_optimizer_state_error),
        "one_step_ppo_loss_before": float(ppo_step_loss_before.detach().item()),
        "one_step_ppo_loss_after": float(ppo_step_loss_after.detach().item()),
        "one_step_logprob_alignment": _direction_stats(bw_adv, one_step_delta_logprob),
        "one_step_pos_only_control": pos_only_control,
        "one_step_neg_only_control": neg_only_control,
        "one_step_pos_single_cold_control": pos_single_cold_control,
        "one_step_neg_single_cold_control": neg_single_cold_control,
        "one_step_pos_single_adam_control": pos_single_adam_control,
        "one_step_neg_single_adam_control": neg_single_adam_control,
        "one_step_gradient_conflict": grad_conflict,
        "positive_conflict_attribution": positive_conflict_attribution,
        "negative_conflict_attribution": negative_conflict_attribution,
        "positive_agent_conflict_attribution": positive_agent_conflict_attribution,
        "negative_agent_conflict_attribution": negative_agent_conflict_attribution,
        "positive_agent_id_conflict_attribution": positive_agent_id_conflict_attribution,
        "negative_agent_id_conflict_attribution": negative_agent_id_conflict_attribution,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
