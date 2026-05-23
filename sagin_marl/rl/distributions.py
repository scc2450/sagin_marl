from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Categorical, Gamma, Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    radius = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    squashed_radius = torch.tanh(radius)
    direction_scale = torch.where(
        radius > 1e-8,
        squashed_radius / radius.clamp_min(1e-8),
        torch.ones_like(radius),
    )
    return z * direction_scale * scale


def squashed_logprob(dist: Normal, action: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    eps = 1e-4
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}.")
    t = action / scale
    action_dim = int(t.shape[-1])
    action_radius = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    squashed_radius = action_radius.clamp(max=1.0 - eps)
    raw_radius = atanh(squashed_radius)
    direction = torch.where(
        action_radius > eps,
        t / action_radius.clamp_min(eps),
        t,
    )
    z = direction * raw_radius
    logprob_z = dist.log_prob(z).sum(dim=-1)
    radius_ratio = torch.where(
        squashed_radius > eps,
        squashed_radius / raw_radius.clamp_min(eps),
        torch.ones_like(squashed_radius),
    )
    log_det = (
        action_dim * torch.log(torch.as_tensor(scale, dtype=t.dtype, device=t.device))
        + (action_dim - 1) * torch.log(radius_ratio.clamp_min(eps))
        + torch.log((1.0 - squashed_radius.pow(2)).clamp_min(eps))
    ).squeeze(-1)
    return logprob_z - log_det


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
        action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def mode(self) -> torch.Tensor:
        alpha_masked = self.alpha * self.mask_f
        denom = alpha_masked.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, alpha_masked / denom.clamp_min(self.eps), torch.zeros_like(alpha_masked))
        single_mask = self.valid_count == 1
        action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
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


class MaskedMeanConcentrationDirichlet:
    """Masked simplex Dirichlet parameterized by mean and scalar concentration.

    The deterministic BW action is defined by the caller as ``mean`` itself.
    This class only provides the stochastic simplex distribution around that
    mean for sampling and log-prob evaluation.
    """

    def __init__(self, mean: torch.Tensor, kappa: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        if mean.shape != mask.shape:
            raise ValueError("mean and mask must share the same shape")
        if kappa.ndim == mean.ndim:
            if kappa.shape[-1] != 1:
                raise ValueError("kappa must be scalar per row when it has the same ndim as mean")
            kappa_scalar = kappa.squeeze(-1)
        elif kappa.ndim == mean.ndim - 1:
            kappa_scalar = kappa
        else:
            raise ValueError("kappa must have shape [...,] or [..., 1] matching mean batch dims")
        if tuple(kappa_scalar.shape) != tuple(mean.shape[:-1]):
            raise ValueError("kappa batch shape must match mean batch shape")
        self.mask = mask > 0.5
        self.eps = float(eps)
        self.mask_f = self.mask.to(mean.dtype)
        self.valid_count = self.mask_f.sum(dim=-1)
        mean_valid = torch.where(self.mask, mean.clamp_min(self.eps), torch.zeros_like(mean))
        mean_sum = mean_valid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        normalized_mean = torch.where(self.mask, mean_valid / mean_sum, torch.zeros_like(mean_valid))
        single_mask = self.valid_count == 1
        normalized_mean = torch.where(single_mask.unsqueeze(-1), self.mask_f, normalized_mean)
        no_mask = self.valid_count <= 0
        normalized_mean = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(normalized_mean), normalized_mean)
        self.mean = normalized_mean
        self.kappa = kappa_scalar.clamp_min(self.eps)
        self.concentration = self.mean * self.kappa.unsqueeze(-1)

    def _masked_concentration(self) -> torch.Tensor:
        return torch.where(self.mask, self.concentration.clamp_min(self.eps), torch.ones_like(self.concentration))

    def sample(self) -> torch.Tensor:
        concentration = self._masked_concentration()
        gamma = Gamma(concentration, torch.ones_like(concentration)).sample()
        gamma = gamma * self.mask_f
        denom = gamma.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, gamma / denom.clamp_min(self.eps), torch.zeros_like(gamma))
        single_mask = self.valid_count == 1
        action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def rsample(self) -> torch.Tensor:
        return self.sample()

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.mean.shape:
            raise ValueError("action must have the same shape as mean")
        masked_concentration = self._masked_concentration()
        action_valid = torch.where(self.mask, action.clamp_min(self.eps), torch.zeros_like(action))
        action_sum = action_valid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        probs = torch.where(self.mask, action_valid / action_sum, torch.zeros_like(action_valid))
        action_safe = torch.where(self.mask, probs.clamp_min(self.eps), torch.ones_like(probs))
        alpha0 = (self.concentration * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        logprob = (
            torch.lgamma(alpha0)
            - torch.lgamma(masked_concentration).sum(dim=-1)
            + ((masked_concentration - 1.0) * torch.log(action_safe)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, logprob, torch.zeros_like(logprob))

    def entropy(self) -> torch.Tensor:
        masked_concentration = self._masked_concentration()
        alpha0 = (self.concentration * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        k_valid = self.valid_count
        log_beta = torch.lgamma(masked_concentration).sum(dim=-1) - torch.lgamma(alpha0)
        entropy = (
            log_beta
            + (alpha0 - k_valid) * torch.digamma(alpha0)
            - ((masked_concentration - 1.0) * torch.digamma(masked_concentration)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, entropy, torch.zeros_like(entropy))


class MaskedStickBreakingBeta:
    """Masked stick-breaking distribution with independent Beta latent factors.

    The action on the simplex is produced by sequentially allocating a fraction
    of the remaining mass to each valid slot, leaving the final valid slot as
    the deterministic residual. Log-prob ratios for PPO are computed in the
    latent Beta-factor space, where the stick-breaking Jacobian cancels between
    old and new policies for the same realized action.
    """

    def __init__(self, mean: torch.Tensor, kappa: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6):
        if mean.shape != mask.shape:
            raise ValueError("mean and mask must share the same shape")
        if kappa.ndim == mean.ndim:
            if kappa.shape[-1] != 1:
                raise ValueError("kappa must be scalar per row when it has the same ndim as mean")
            kappa_scalar = kappa.squeeze(-1)
        elif kappa.ndim == mean.ndim - 1:
            kappa_scalar = kappa
        else:
            raise ValueError("kappa must have shape [...,] or [..., 1] matching mean batch dims")
        if tuple(kappa_scalar.shape) != tuple(mean.shape[:-1]):
            raise ValueError("kappa batch shape must match mean batch shape")
        self.eps = float(eps)
        self.mask = mask > 0.5
        self.mask_f = self.mask.to(mean.dtype)
        self.valid_count = self.mask_f.sum(dim=-1)
        mean_valid = torch.where(self.mask, mean.clamp_min(self.eps), torch.zeros_like(mean))
        mean_sum = mean_valid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        normalized_mean = torch.where(self.mask, mean_valid / mean_sum, torch.zeros_like(mean_valid))
        single_mask = self.valid_count == 1
        normalized_mean = torch.where(single_mask.unsqueeze(-1), self.mask_f, normalized_mean)
        no_mask = self.valid_count <= 0
        normalized_mean = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(normalized_mean), normalized_mean)
        self.mean = normalized_mean
        self.kappa = kappa_scalar.clamp_min(self.eps)
        self._flat_mean = self.mean.reshape(-1, self.mean.shape[-1])
        self._flat_mask = self.mask.reshape(-1, self.mask.shape[-1])
        self._batch_shape = self.mean.shape[:-1]
        self._event_dim = int(self.mean.shape[-1])
        index_grid = torch.arange(self._event_dim, device=self.mean.device, dtype=torch.long).unsqueeze(0)
        self._flat_last_idx = torch.where(
            self._flat_mask,
            index_grid.expand_as(self._flat_mask),
            torch.full_like(self._flat_mask, -1, dtype=torch.long),
        ).amax(dim=-1)
        active_rows = self._flat_mask.sum(dim=-1) > 1
        last_mask = index_grid.expand_as(self._flat_mask) == self._flat_last_idx.unsqueeze(-1)
        self._flat_latent_mask = self._flat_mask & ~(active_rows.unsqueeze(-1) & last_mask)
        self._flat_last_mask = self._flat_mask & ~self._flat_latent_mask
        self.latent_mask = self._flat_latent_mask.reshape_as(self.mask)
        self.latent_count = self.latent_mask.to(dtype=mean.dtype).sum(dim=-1)
        factor_mean = self._flat_mean_to_factors(self._flat_mean)
        flat_kappa = self.kappa.reshape(-1, 1).expand_as(factor_mean)
        self._flat_factor_mean = torch.where(
            self._flat_latent_mask,
            factor_mean.clamp(min=self.eps, max=1.0 - self.eps),
            torch.full_like(factor_mean, 0.5),
        )
        self._flat_alpha = torch.where(
            self._flat_latent_mask,
            (self._flat_factor_mean * flat_kappa).clamp_min(self.eps),
            torch.ones_like(self._flat_factor_mean),
        )
        self._flat_beta = torch.where(
            self._flat_latent_mask,
            ((1.0 - self._flat_factor_mean) * flat_kappa).clamp_min(self.eps),
            torch.ones_like(self._flat_factor_mean),
        )
        self._beta_dist = Beta(self._flat_alpha, self._flat_beta)

    def _flat_mean_to_factors(self, flat_probs: torch.Tensor) -> torch.Tensor:
        factor_cols: list[torch.Tensor] = []
        remaining = torch.ones((flat_probs.shape[0],), dtype=flat_probs.dtype, device=flat_probs.device)
        for slot in range(self._event_dim):
            slot_mass = torch.where(self._flat_mask[:, slot], flat_probs[:, slot], torch.zeros_like(remaining))
            factor = torch.where(
                self._flat_latent_mask[:, slot],
                (slot_mass / remaining.clamp_min(self.eps)).clamp(min=self.eps, max=1.0 - self.eps),
                torch.zeros_like(remaining),
            )
            factor_cols.append(factor)
            remaining = torch.where(
                self._flat_mask[:, slot],
                (remaining - slot_mass).clamp_min(self.eps),
                remaining,
            )
        if not factor_cols:
            return torch.zeros_like(flat_probs)
        return torch.stack(factor_cols, dim=-1)

    def _flat_factors_to_action(self, flat_factors: torch.Tensor) -> torch.Tensor:
        action_cols: list[torch.Tensor] = []
        remaining = torch.ones((flat_factors.shape[0],), dtype=flat_factors.dtype, device=flat_factors.device)
        for slot in range(self._event_dim):
            factor = torch.where(
                self._flat_latent_mask[:, slot],
                flat_factors[:, slot].clamp(min=self.eps, max=1.0 - self.eps),
                torch.zeros_like(remaining),
            )
            latent_alloc = torch.where(self._flat_latent_mask[:, slot], remaining * factor, torch.zeros_like(remaining))
            last_alloc = torch.where(self._flat_last_mask[:, slot], remaining, torch.zeros_like(remaining))
            action_cols.append(latent_alloc + last_alloc)
            remaining = torch.where(self._flat_latent_mask[:, slot], remaining * (1.0 - factor), remaining)
            remaining = torch.where(self._flat_last_mask[:, slot], torch.zeros_like(remaining), remaining)
        if not action_cols:
            return torch.zeros_like(flat_factors)
        return torch.stack(action_cols, dim=-1)

    def _flat_action_to_factors(self, flat_action: torch.Tensor) -> torch.Tensor:
        flat_valid = torch.where(self._flat_mask, flat_action.clamp_min(self.eps), torch.zeros_like(flat_action))
        valid_sum = flat_valid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        probs = torch.where(self._flat_mask, flat_valid / valid_sum, torch.zeros_like(flat_valid))
        return self._flat_mean_to_factors(probs)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if len(sample_shape) > 1:
            raise ValueError("Only one-dimensional sample_shape is supported")
        sample_count = int(sample_shape[0]) if len(sample_shape) == 1 else 1
        factor_sample = self._beta_dist.rsample((sample_count,))
        factor_sample = torch.where(
            self._flat_latent_mask.unsqueeze(0),
            factor_sample,
            torch.zeros_like(factor_sample),
        )
        flat_actions = self._flat_factors_to_action(factor_sample.reshape(-1, self._event_dim)).reshape(
            sample_count,
            *self._batch_shape,
            self._event_dim,
        )
        if len(sample_shape) == 0:
            return flat_actions[0]
        return flat_actions

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob_parts(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.mean.shape:
            raise ValueError("action must have the same shape as mean")
        flat_factors = self._flat_action_to_factors(action.reshape(-1, self._event_dim))
        logprob = self._beta_dist.log_prob(flat_factors.clamp(min=self.eps, max=1.0 - self.eps))
        logprob = torch.where(self._flat_latent_mask, logprob, torch.zeros_like(logprob))
        return logprob.reshape(*self._batch_shape, self._event_dim)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.log_prob_parts(action).sum(dim=-1)

    def entropy_parts(self) -> torch.Tensor:
        entropy = self._beta_dist.entropy()
        entropy = torch.where(self._flat_latent_mask, entropy, torch.zeros_like(entropy))
        return entropy.reshape(*self._batch_shape, self._event_dim)

    def entropy(self) -> torch.Tensor:
        return self.entropy_parts().sum(dim=-1)


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
        self._bit_weights = (1 << torch.arange(self.num_choices, device=logits.device, dtype=torch.int64))

    def _safe_logits(self, mask: torch.Tensor) -> torch.Tensor:
        return self.logits.masked_fill(~mask, -1e9)

    def _indices_to_select_mask(self, indices: torch.Tensor) -> torch.Tensor:
        select_mask = torch.zeros((*indices.shape[:-1], self.num_choices), dtype=self.logits.dtype, device=self.logits.device)
        valid_choice_mask = indices >= 0
        if self.num_choices <= 0:
            return select_mask
        one_hot = F.one_hot(indices.clamp_min(0), num_classes=self.num_choices).to(select_mask.dtype)
        one_hot = one_hot * valid_choice_mask.unsqueeze(-1).to(select_mask.dtype)
        select_mask = one_hot.sum(dim=-2)
        return select_mask

    def _pad_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] >= self.k:
            return indices
        pad = torch.full(
            (*indices.shape[:-1], self.k - indices.shape[-1]),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        return torch.cat([indices, pad], dim=-1)

    def _empty_result(self) -> MaskedSequentialCategoricalSample:
        batch_shape = self.logits.shape[:-1]
        return MaskedSequentialCategoricalSample(
            indices=torch.full((*batch_shape, self.k), -1, dtype=torch.int64, device=self.logits.device),
            select_mask=torch.zeros((*batch_shape, self.num_choices), dtype=self.logits.dtype, device=self.logits.device),
        )

    def sample(self) -> MaskedSequentialCategoricalSample:
        if self.k <= 0:
            return self._empty_result()
        max_k = min(self.k, self.num_choices)
        if max_k <= 0:
            return self._empty_result()
        valid_count = self.mask.sum(dim=-1)
        safe_logits = self.logits.masked_fill(~self.mask, float("-inf"))
        gumbel_u = torch.rand_like(self.logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(gumbel_u))
        sampled_scores = safe_logits + gumbel
        topk = sampled_scores.topk(k=max_k, dim=-1).indices
        rank_idx = torch.arange(max_k, device=topk.device).view(*([1] * (topk.ndim - 1)), -1)
        valid_rank = rank_idx < valid_count.unsqueeze(-1)
        indices = torch.where(valid_rank, topk, torch.full_like(topk, -1))
        indices = self._pad_indices(indices)
        select_mask = self._indices_to_select_mask(indices)
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
        indices = self._pad_indices(indices)
        select_mask = self._indices_to_select_mask(indices)
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
            safe_logits = flat_logits.masked_fill(~current_mask, -1e9)
            safe_logits = torch.where(active.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
            step_indices = flat_indices[:, step].clamp_min(0)
            step_log_probs = F.log_softmax(safe_logits, dim=-1)
            gathered = step_log_probs.gather(1, step_indices.unsqueeze(-1)).squeeze(-1)
            step_valid = active & (flat_indices[:, step] >= 0) & current_mask.gather(
                1, step_indices.unsqueeze(-1)
            ).squeeze(-1)
            logprob = logprob + torch.where(step_valid, gathered, torch.zeros_like(logprob))
            step_valid_idx = flat_indices[:, step] >= 0
            chosen_mask = F.one_hot(
                flat_indices[:, step].clamp_min(0),
                num_classes=self.num_choices,
            ).to(torch.bool)
            chosen_mask = chosen_mask & step_valid_idx.unsqueeze(-1)
            current_mask = current_mask & ~chosen_mask
        return logprob.reshape(self.logits.shape[:-1])

    def _entropy_exact_state_dp(self, steps: int) -> torch.Tensor:
        batch_shape = self.logits.shape[:-1]
        if self.num_choices <= 0 or int(steps) <= 0:
            return torch.zeros(batch_shape, dtype=self.logits.dtype, device=self.logits.device)
        if self.num_choices > 12:
            safe_logits = self._safe_logits(self.mask)
            has_any = self.mask.any(dim=-1)
            safe_logits = torch.where(has_any.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
            one_step_entropy = torch.where(has_any, Categorical(logits=safe_logits).entropy(), torch.zeros(batch_shape, dtype=self.logits.dtype, device=self.logits.device))
            return one_step_entropy * float(min(int(steps), self.num_choices))

        flat_logits = self.logits.reshape(-1, self.num_choices)
        flat_mask = self.mask.reshape(-1, self.num_choices)
        batch = int(flat_logits.shape[0])
        state_count = 1 << int(self.num_choices)
        state_ids = torch.arange(state_count, device=self.logits.device, dtype=torch.long)
        bit_weights = (1 << torch.arange(self.num_choices, device=self.logits.device, dtype=torch.long))
        state_masks = (state_ids.unsqueeze(-1).bitwise_and(bit_weights.view(1, -1)) != 0)
        initial_ids = (flat_mask.to(torch.long) * bit_weights.view(1, -1)).sum(dim=-1)
        state_probs = F.one_hot(initial_ids, num_classes=state_count).to(dtype=self.logits.dtype)
        total_entropy = torch.zeros((batch,), dtype=self.logits.dtype, device=self.logits.device)
        state_valid_counts = state_masks.sum(dim=-1)
        for _step in range(min(int(steps), int(self.num_choices))):
            active_states = state_valid_counts > 1
            logits_state = flat_logits[:, None, :].masked_fill(~state_masks.view(1, state_count, self.num_choices), -1e9)
            logits_state = torch.where(active_states.view(1, state_count, 1), logits_state, torch.zeros_like(logits_state))
            dist_state = Categorical(logits=logits_state)
            entropy_state = torch.where(
                active_states.view(1, state_count),
                dist_state.entropy(),
                torch.zeros((batch, state_count), dtype=self.logits.dtype, device=self.logits.device),
            )
            total_entropy = total_entropy + (state_probs * entropy_state).sum(dim=-1)
            choice_probs = torch.where(
                active_states.view(1, state_count, 1),
                dist_state.probs * state_masks.view(1, state_count, self.num_choices).to(self.logits.dtype),
                torch.zeros((batch, state_count, self.num_choices), dtype=self.logits.dtype, device=self.logits.device),
            )
            next_state_ids = state_ids.view(state_count, 1).bitwise_and(~bit_weights.view(1, self.num_choices))
            next_probs = torch.zeros_like(state_probs)
            expanded_next = next_state_ids.view(1, state_count, self.num_choices).expand(batch, -1, -1)
            next_probs.scatter_add_(1, expanded_next.reshape(batch, -1), (state_probs.unsqueeze(-1) * choice_probs).reshape(batch, -1))
            inactive_mass = state_probs * (~active_states).view(1, state_count).to(state_probs.dtype)
            state_probs = next_probs + inactive_mass
        return total_entropy.reshape(batch_shape)

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
            return self._entropy_exact_state_dp(int(self.k))
        probs1 = dist1.probs
        expected_h2 = torch.zeros_like(entropy)
        for choice in range(self.num_choices):
            choice_mask = self.mask & ~F.one_hot(
                torch.full(batch_shape, int(choice), dtype=torch.long, device=self.logits.device),
                num_classes=self.num_choices,
            ).to(torch.bool)
            remaining = choice_mask.sum(dim=-1)
            active = self.mask[..., choice] & (remaining > 0)
            logits2 = self._safe_logits(choice_mask)
            logits2 = torch.where(active.unsqueeze(-1), logits2, torch.zeros_like(logits2))
            dist2 = Categorical(logits=logits2)
            h2 = torch.where(active, dist2.entropy(), torch.zeros_like(entropy))
            expected_h2 = expected_h2 + probs1[..., choice] * h2
        return entropy + expected_h2


@dataclass
class HybridActionSample:
    env_action: torch.Tensor
    accel: torch.Tensor | None
    bw_action: torch.Tensor | None
    sat_indices: torch.Tensor | None
    sat_select_mask: torch.Tensor | None
    logprob_parts: Dict[str, torch.Tensor]
    entropy_parts: Dict[str, torch.Tensor]


def _normalize_requested_heads(heads: Iterable[str] | None) -> set[str] | None:
    if heads is None:
        return None
    allowed = {"accel", "bw", "sat"}
    out = {str(head).strip().lower() for head in heads}
    return out & allowed


class HybridActionDist:
    def __init__(
        self,
        accel_mu: torch.Tensor | None,
        accel_log_std: torch.Tensor | None,
        bw_alpha: torch.Tensor | None = None,
        bw_mask: torch.Tensor | None = None,
        sat_logits: torch.Tensor | None = None,
        sat_mask: torch.Tensor | None = None,
        sat_num_select: int = 2,
    ):
        batch_source = accel_mu if accel_mu is not None else bw_alpha
        if batch_source is None:
            batch_source = sat_logits
        if batch_source is None:
            raise ValueError("HybridActionDist requires at least one action head.")
        self.batch_shape = tuple(batch_source.shape[:-1])
        self.dtype = batch_source.dtype
        self.device = batch_source.device
        self.accel_mu = accel_mu
        self.accel_log_std = None if accel_log_std is None else torch.clamp(accel_log_std, -5.0, 2.0)
        self.accel_std = None if self.accel_log_std is None else torch.exp(self.accel_log_std)
        self.accel_dist = (
            None if self.accel_mu is None or self.accel_std is None else Normal(self.accel_mu, self.accel_std)
        )
        self.bw_dist = None if bw_alpha is None or bw_mask is None else MaskedDirichlet(bw_alpha, bw_mask)
        self.sat_dist = (
            None
            if sat_logits is None or sat_mask is None
            else MaskedSequentialCategorical(sat_logits, sat_mask, k=sat_num_select)
        )

    def _zero_batch(self) -> torch.Tensor:
        return torch.zeros(self.batch_shape, dtype=self.dtype, device=self.device)

    def _head_enabled(self, head_set: set[str] | None, head: str) -> bool:
        return head_set is None or head in head_set

    def sample(
        self,
        deterministic: bool = False,
        compute_logprob: bool = False,
        compute_entropy: bool = False,
        stat_heads: Iterable[str] | None = None,
    ) -> HybridActionSample:
        stat_head_set = _normalize_requested_heads(stat_heads)
        logprob_parts: Dict[str, torch.Tensor] = {}
        entropy_parts: Dict[str, torch.Tensor] = {}
        env_parts: list[torch.Tensor] = []
        accel = None
        if self.accel_dist is not None and self.accel_mu is not None:
            accel_z = self.accel_mu if deterministic else self.accel_dist.rsample()
            accel = squash_action(accel_z, scale=1.0)
            env_parts.append(accel)
            if compute_logprob and self._head_enabled(stat_head_set, "accel"):
                logprob_parts["accel"] = squashed_logprob(self.accel_dist, accel, scale=1.0)
            if compute_entropy and self._head_enabled(stat_head_set, "accel"):
                entropy_parts["accel"] = self.accel_dist.entropy().sum(dim=-1)
        bw_action = None
        if self.bw_dist is not None:
            bw_action = self.bw_dist.mode() if deterministic else self.bw_dist.sample()
            env_parts.append(bw_action)
            if compute_logprob and self._head_enabled(stat_head_set, "bw"):
                logprob_parts["bw"] = self.bw_dist.log_prob(bw_action)
            if compute_entropy and self._head_enabled(stat_head_set, "bw"):
                entropy_parts["bw"] = self.bw_dist.entropy()
        sat_indices = None
        sat_select_mask = None
        if self.sat_dist is not None:
            sat_sample = self.sat_dist.mode() if deterministic else self.sat_dist.sample()
            sat_indices = sat_sample.indices
            sat_select_mask = sat_sample.select_mask
            env_parts.append(sat_select_mask)
            if compute_logprob and self._head_enabled(stat_head_set, "sat"):
                logprob_parts["sat"] = self.sat_dist.log_prob(sat_indices)
            if compute_entropy and self._head_enabled(stat_head_set, "sat"):
                entropy_parts["sat"] = self.sat_dist.entropy()
        env_action = torch.cat(env_parts, dim=-1) if env_parts else torch.zeros(
            (*self.batch_shape, 0), dtype=self.dtype, device=self.device
        )
        return HybridActionSample(
            env_action=env_action,
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
        heads: Iterable[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        head_set = _normalize_requested_heads(heads)
        out: Dict[str, torch.Tensor] = {}
        if self.accel_dist is not None and self._head_enabled(head_set, "accel"):
            out["accel"] = squashed_logprob(self.accel_dist, accel, scale=1.0)
        if self.bw_dist is not None and bw_action is not None and self._head_enabled(head_set, "bw"):
            out["bw"] = self.bw_dist.log_prob(bw_action)
        if self.sat_dist is not None and sat_indices is not None and self._head_enabled(head_set, "sat"):
            out["sat"] = self.sat_dist.log_prob(sat_indices)
        return out

    def entropy(self, heads: Iterable[str] | None = None) -> Dict[str, torch.Tensor]:
        head_set = _normalize_requested_heads(heads)
        out: Dict[str, torch.Tensor] = {}
        if self.accel_dist is not None and self._head_enabled(head_set, "accel"):
            out["accel"] = self.accel_dist.entropy().sum(dim=-1)
        if self.bw_dist is not None and self._head_enabled(head_set, "bw"):
            out["bw"] = self.bw_dist.entropy()
        if self.sat_dist is not None and self._head_enabled(head_set, "sat"):
            out["sat"] = self.sat_dist.entropy()
        return out
