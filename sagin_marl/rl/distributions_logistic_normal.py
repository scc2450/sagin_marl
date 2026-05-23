from __future__ import annotations

import math

import torch


LOG_2PI = math.log(2.0 * math.pi)


class MaskedLogisticNormal:
    """Additive-log-ratio logistic-normal on a masked simplex.

    The last valid component is used as the ALR reference component.
    Rows with zero valid components return the zero vector, and rows with a
    single valid component collapse to a deterministic one-hot action.
    """

    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        if loc.shape != log_scale.shape or loc.shape != mask.shape:
            raise ValueError("loc, log_scale, and mask must share the same shape")
        self.loc = loc
        self.log_scale = log_scale.clamp(min=math.log(eps))
        self.mask = mask > 0.5
        self.eps = float(eps)
        self._flat_loc = self.loc.reshape(-1, self.loc.shape[-1])
        self._flat_log_scale = self.log_scale.reshape(-1, self.log_scale.shape[-1])
        self._flat_scale = self._flat_log_scale.exp()
        self._flat_mask = self.mask.reshape(-1, self.mask.shape[-1])
        self._event_dim = int(self.loc.shape[-1])
        self._batch_shape = self.loc.shape[:-1]
        self._flat_counts = self._flat_mask.sum(dim=-1)
        self._flat_has_valid = self._flat_counts > 0
        self._flat_has_latent = self._flat_counts > 1
        index_grid = torch.arange(self._event_dim, device=self.loc.device, dtype=torch.long).unsqueeze(0)
        self._flat_ref_idx = torch.where(
            self._flat_mask,
            index_grid.expand_as(self._flat_mask),
            torch.full_like(self._flat_mask, -1, dtype=torch.long),
        ).amax(dim=-1)
        ref_mask = index_grid.expand_as(self._flat_mask) == self._flat_ref_idx.unsqueeze(-1)
        self._flat_latent_mask = self._flat_mask & ~(self._flat_has_latent.unsqueeze(-1) & ref_mask)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size(self._batch_shape)

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size((self._event_dim,))

    def _flat_probs_from_latent(self, flat_latent: torch.Tensor) -> torch.Tensor:
        if flat_latent.shape != self._flat_loc.shape:
            raise ValueError("flat_latent must have flattened loc shape")
        logits = torch.where(self._flat_latent_mask, flat_latent, torch.zeros_like(flat_latent))
        safe_logits = logits.masked_fill(~self._flat_mask, float("-inf"))
        safe_logits = torch.where(self._flat_has_valid.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
        probs = torch.softmax(safe_logits, dim=-1)
        probs = probs * self._flat_mask.to(probs.dtype)
        norm = probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return probs / norm

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if len(sample_shape) > 1:
            raise ValueError("Only one-dimensional sample_shape is supported")
        sample_count = int(sample_shape[0]) if len(sample_shape) == 1 else 1
        latent = self._flat_loc.unsqueeze(0) + self._flat_scale.unsqueeze(0) * torch.randn(
            (sample_count, *self._flat_loc.shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        latent = latent * self._flat_latent_mask.unsqueeze(0).to(latent.dtype)
        stacked = self._flat_probs_from_latent(latent.reshape(-1, self._event_dim)).reshape(
            sample_count,
            *self._batch_shape,
            self._event_dim,
        )
        if len(sample_shape) == 0:
            return stacked[0]
        return stacked

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def mode(self) -> torch.Tensor:
        latent = self._flat_loc * self._flat_latent_mask.to(self.loc.dtype)
        return self._flat_probs_from_latent(latent).reshape(*self._batch_shape, self._event_dim)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape != self.loc.shape:
            raise ValueError("value must have the same shape as loc")
        flat_value = value.reshape(-1, self._event_dim)
        valid_value = torch.where(self._flat_mask, flat_value.clamp_min(self.eps), torch.zeros_like(flat_value))
        valid_sum = valid_value.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        probs = torch.where(self._flat_mask, valid_value / valid_sum, torch.zeros_like(valid_value))
        out = torch.zeros((flat_value.shape[0],), dtype=self.loc.dtype, device=self.loc.device)
        active = self._flat_has_latent
        ref_prob = probs.gather(1, self._flat_ref_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1).clamp_min(self.eps)
        latent = torch.log(probs.clamp_min(self.eps)) - torch.log(ref_prob.unsqueeze(-1))
        latent_z = (latent - self._flat_loc) / self._flat_scale.clamp_min(self.eps)
        normal_logprob = -0.5 * (latent_z.pow(2) + LOG_2PI) - self._flat_log_scale
        normal_logprob = (normal_logprob * self._flat_latent_mask.to(normal_logprob.dtype)).sum(dim=-1)
        log_det = (torch.log(probs.clamp_min(self.eps)) * self._flat_mask.to(probs.dtype)).sum(dim=-1)
        out = torch.where(active, normal_logprob - log_det, out)
        return out.reshape(*self._batch_shape)

    def entropy(self, num_samples: int = 1) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        out = torch.zeros((self._flat_loc.shape[0],), dtype=self.loc.dtype, device=self.loc.device)
        active = self._flat_has_latent
        base_entropy = (
            (0.5 * (1.0 + LOG_2PI) + self._flat_log_scale)
            * self._flat_latent_mask.to(self.loc.dtype)
        ).sum(dim=-1)
        latent = self._flat_loc.unsqueeze(0) + self._flat_scale.unsqueeze(0) * torch.randn(
            (num_samples, *self._flat_loc.shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        latent = latent * self._flat_latent_mask.unsqueeze(0).to(latent.dtype)
        probs = self._flat_probs_from_latent(latent.reshape(-1, self._event_dim)).reshape(
            num_samples,
            self._flat_loc.shape[0],
            self._event_dim,
        )
        log_abs_det = (
            torch.log(probs.clamp_min(self.eps))
            * self._flat_mask.unsqueeze(0).to(probs.dtype)
        ).sum(dim=-1)
        out = torch.where(active, base_entropy + log_abs_det.mean(dim=0), out)
        return out.reshape(*self._batch_shape)
