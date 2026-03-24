from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gamma, Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(z) * scale


def squashed_logprob(dist: Normal, action: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    eps = 1e-4
    t = action / scale
    t = torch.clamp(t, -1 + eps, 1 - eps)
    z = atanh(t)
    logprob = dist.log_prob(z) - torch.log(1 - t.pow(2) + eps)
    if scale != 1.0:
        logprob = logprob - torch.log(torch.full_like(logprob, scale))
    return logprob.sum(-1)


class MaskedDirichlet:
    def __init__(self, alpha: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        self.alpha = alpha.clamp_min(eps)
        self.mask = mask > 0.5
        self.eps = eps
        self.mask_f = self.mask.to(self.alpha.dtype)
        self.valid_count = self.mask_f.sum(dim=-1)

    def _masked_alpha(self) -> torch.Tensor:
        return torch.where(self.mask, self.alpha, torch.ones_like(self.alpha))

    def sample(self) -> torch.Tensor:
        gamma = Gamma(self.alpha, torch.ones_like(self.alpha)).sample()
        gamma = gamma * self.mask_f
        denom = gamma.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, gamma / denom.clamp_min(self.eps), torch.zeros_like(gamma))
        single_mask = self.valid_count == 1
        if torch.any(single_mask):
            action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        if torch.any(no_mask):
            action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def mode(self) -> torch.Tensor:
        alpha_masked = self.alpha * self.mask_f
        denom = alpha_masked.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, alpha_masked / denom.clamp_min(self.eps), torch.zeros_like(alpha_masked))
        single_mask = self.valid_count == 1
        if torch.any(single_mask):
            action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        if torch.any(no_mask):
            action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        masked_alpha = self._masked_alpha()
        action_safe = torch.where(self.mask, action.clamp_min(self.eps), torch.ones_like(action))
        alpha0 = (self.alpha * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        logprob = (
            torch.lgamma(alpha0)
            - torch.lgamma(masked_alpha).sum(dim=-1)
            + ((masked_alpha - 1.0) * torch.log(action_safe)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, logprob, torch.zeros_like(logprob))

    def entropy(self) -> torch.Tensor:
        masked_alpha = self._masked_alpha()
        alpha0 = (self.alpha * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        k_valid = self.valid_count
        log_beta = torch.lgamma(masked_alpha).sum(dim=-1) - torch.lgamma(alpha0)
        entropy = (
            log_beta
            + (alpha0 - k_valid) * torch.digamma(alpha0)
            - ((masked_alpha - 1.0) * torch.digamma(masked_alpha)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, entropy, torch.zeros_like(entropy))


@dataclass
class MaskedSequentialCategoricalSample:
    indices: torch.Tensor
    select_mask: torch.Tensor


class MaskedSequentialCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor, k: int = 2):
        self.logits = logits
        self.mask = mask > 0.5
        self.k = max(int(k), 0)
        self.num_choices = int(logits.shape[-1])

    def _safe_logits(self, mask: torch.Tensor) -> torch.Tensor:
        return self.logits.masked_fill(~mask, -1e9)

    def _empty_result(self) -> MaskedSequentialCategoricalSample:
        batch_shape = self.logits.shape[:-1]
        return MaskedSequentialCategoricalSample(
            indices=torch.full((*batch_shape, self.k), -1, dtype=torch.int64, device=self.logits.device),
            select_mask=torch.zeros((*batch_shape, self.num_choices), dtype=self.logits.dtype, device=self.logits.device),
        )

    def sample(self) -> MaskedSequentialCategoricalSample:
        if self.k <= 0:
            return self._empty_result()
        batch_shape = self.logits.shape[:-1]
        indices = torch.full((*batch_shape, self.k), -1, dtype=torch.int64, device=self.logits.device)
        select_mask = torch.zeros((*batch_shape, self.num_choices), dtype=self.logits.dtype, device=self.logits.device)
        current_mask = self.mask.clone()
        for step in range(self.k):
            remaining = current_mask.sum(dim=-1)
            active = remaining > 0
            if not torch.any(active):
                break
            safe_logits = self._safe_logits(current_mask)
            safe_logits = torch.where(active.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
            dist = Categorical(logits=safe_logits)
            sampled = dist.sample()
            sampled = torch.where(active, sampled, torch.full_like(sampled, -1))
            indices[..., step] = sampled
            chosen_mask = F.one_hot(sampled.clamp_min(0), num_classes=self.num_choices).to(torch.bool)
            chosen_mask = chosen_mask & active.unsqueeze(-1) & current_mask
            select_mask = select_mask + chosen_mask.to(select_mask.dtype)
            current_mask = current_mask & ~chosen_mask
        return MaskedSequentialCategoricalSample(indices=indices, select_mask=select_mask)

    def mode(self) -> MaskedSequentialCategoricalSample:
        if self.k <= 0:
            return self._empty_result()
        safe_logits = self._safe_logits(self.mask)
        topk = safe_logits.topk(k=min(self.k, self.num_choices), dim=-1).indices
        valid_count = self.mask.sum(dim=-1)
        rank_idx = torch.arange(topk.shape[-1], device=topk.device).view(*([1] * (topk.ndim - 1)), -1)
        valid_rank = rank_idx < valid_count.unsqueeze(-1)
        indices = torch.where(valid_rank, topk, torch.full_like(topk, -1))
        if topk.shape[-1] < self.k:
            pad = torch.full((*indices.shape[:-1], self.k - topk.shape[-1]), -1, dtype=indices.dtype, device=indices.device)
            indices = torch.cat([indices, pad], dim=-1)
        select_mask = torch.zeros_like(self.logits)
        valid_choice_mask = indices >= 0
        if torch.any(valid_choice_mask):
            one_hot = F.one_hot(indices.clamp_min(0), num_classes=self.num_choices).to(select_mask.dtype)
            one_hot = one_hot * valid_choice_mask.unsqueeze(-1).to(select_mask.dtype)
            select_mask = one_hot.sum(dim=-2)
        return MaskedSequentialCategoricalSample(indices=indices, select_mask=select_mask)

    def log_prob(self, indices: torch.Tensor) -> torch.Tensor:
        if self.k <= 0:
            return torch.zeros(self.logits.shape[:-1], dtype=self.logits.dtype, device=self.logits.device)
        flat_logits = self.logits.reshape(-1, self.num_choices)
        flat_mask = self.mask.reshape(-1, self.num_choices)
        flat_indices = indices.reshape(-1, self.k)
        logprob = torch.zeros((flat_logits.shape[0],), dtype=flat_logits.dtype, device=flat_logits.device)
        current_mask = flat_mask.clone()
        for step in range(self.k):
            remaining = current_mask.sum(dim=-1)
            active = remaining > 1
            if torch.any(active):
                safe_logits = flat_logits.masked_fill(~current_mask, -1e9)
                safe_logits = torch.where(active.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
                dist = Categorical(logits=safe_logits)
                step_indices = flat_indices[:, step].clamp_min(0)
                logprob = logprob + torch.where(active, dist.log_prob(step_indices), torch.zeros_like(logprob))
            chosen_mask = F.one_hot(flat_indices[:, step].clamp_min(0), num_classes=self.num_choices).to(torch.bool)
            chosen_mask = chosen_mask & current_mask & (flat_indices[:, step] >= 0).unsqueeze(-1)
            current_mask = current_mask & ~chosen_mask
        return logprob.reshape(self.logits.shape[:-1])

    def entropy(self) -> torch.Tensor:
        batch_shape = self.logits.shape[:-1]
        if self.k <= 0:
            return torch.zeros(batch_shape, dtype=self.logits.dtype, device=self.logits.device)
        safe_logits = self._safe_logits(self.mask)
        has_any = self.mask.any(dim=-1)
        safe_logits = torch.where(has_any.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
        dist1 = Categorical(logits=safe_logits)
        entropy = torch.where(has_any, dist1.entropy(), torch.zeros_like(dist1.entropy()))
        if self.k == 1:
            return entropy
        if self.k != 2:
            return entropy
        probs1 = dist1.probs
        expected_h2 = torch.zeros_like(entropy)
        for choice in range(self.num_choices):
            choice_mask = self.mask.clone()
            choice_mask[..., choice] = False
            remaining = choice_mask.sum(dim=-1)
            active = self.mask[..., choice] & (remaining > 0)
            if not torch.any(active):
                continue
            logits2 = self._safe_logits(choice_mask)
            logits2 = torch.where(active.unsqueeze(-1), logits2, torch.zeros_like(logits2))
            dist2 = Categorical(logits=logits2)
            h2 = torch.where(active, dist2.entropy(), torch.zeros_like(entropy))
            expected_h2 = expected_h2 + probs1[..., choice] * h2
        return entropy + expected_h2


@dataclass
class HybridActionSample:
    env_action: torch.Tensor
    accel: torch.Tensor
    bw_action: torch.Tensor | None
    sat_indices: torch.Tensor | None
    sat_select_mask: torch.Tensor | None
    logprob_parts: Dict[str, torch.Tensor]
    entropy_parts: Dict[str, torch.Tensor]


class HybridActionDist:
    def __init__(
        self,
        accel_mu: torch.Tensor,
        accel_log_std: torch.Tensor,
        bw_alpha: torch.Tensor | None = None,
        bw_mask: torch.Tensor | None = None,
        sat_logits: torch.Tensor | None = None,
        sat_mask: torch.Tensor | None = None,
        sat_num_select: int = 2,
    ):
        self.accel_mu = accel_mu
        self.accel_log_std = torch.clamp(accel_log_std, -5.0, 2.0)
        self.accel_std = torch.exp(self.accel_log_std)
        self.accel_dist = Normal(self.accel_mu, self.accel_std)
        self.bw_dist = None if bw_alpha is None or bw_mask is None else MaskedDirichlet(bw_alpha, bw_mask)
        self.sat_dist = (
            None
            if sat_logits is None or sat_mask is None
            else MaskedSequentialCategorical(sat_logits, sat_mask, k=sat_num_select)
        )

    def sample(self, deterministic: bool = False) -> HybridActionSample:
        accel_z = self.accel_mu if deterministic else self.accel_dist.rsample()
        accel = squash_action(accel_z, scale=1.0)
        logprob_parts: Dict[str, torch.Tensor] = {
            "accel": squashed_logprob(self.accel_dist, accel, scale=1.0)
        }
        entropy_parts: Dict[str, torch.Tensor] = {
            "accel": self.accel_dist.entropy().sum(dim=-1)
        }
        env_parts = [accel]
        bw_action = None
        if self.bw_dist is not None:
            bw_action = self.bw_dist.mode() if deterministic else self.bw_dist.sample()
            env_parts.append(bw_action)
            logprob_parts["bw"] = self.bw_dist.log_prob(bw_action)
            entropy_parts["bw"] = self.bw_dist.entropy()
        sat_indices = None
        sat_select_mask = None
        if self.sat_dist is not None:
            sat_sample = self.sat_dist.mode() if deterministic else self.sat_dist.sample()
            sat_indices = sat_sample.indices
            sat_select_mask = sat_sample.select_mask
            env_parts.append(sat_select_mask)
            logprob_parts["sat"] = self.sat_dist.log_prob(sat_indices)
            entropy_parts["sat"] = self.sat_dist.entropy()
        return HybridActionSample(
            env_action=torch.cat(env_parts, dim=-1),
            accel=accel,
            bw_action=bw_action,
            sat_indices=sat_indices,
            sat_select_mask=sat_select_mask,
            logprob_parts=logprob_parts,
            entropy_parts=entropy_parts,
        )

    def log_prob(
        self,
        accel: torch.Tensor,
        bw_action: torch.Tensor | None = None,
        sat_indices: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "accel": squashed_logprob(self.accel_dist, accel, scale=1.0)
        }
        if self.bw_dist is not None and bw_action is not None:
            out["bw"] = self.bw_dist.log_prob(bw_action)
        if self.sat_dist is not None and sat_indices is not None:
            out["sat"] = self.sat_dist.log_prob(sat_indices)
        return out

    def entropy(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "accel": self.accel_dist.entropy().sum(dim=-1)
        }
        if self.bw_dist is not None:
            out["bw"] = self.bw_dist.entropy()
        if self.sat_dist is not None:
            out["sat"] = self.sat_dist.entropy()
        return out
