from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _normalize_dims(ndim: int, dim: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(dim, int):
        dims = (int(dim),)
    else:
        dims = tuple(int(axis) for axis in dim)
    normalized: list[int] = []
    for axis in dims:
        normalized_axis = axis if axis >= 0 else ndim + axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise IndexError(f"Dimension out of range for ndim={ndim}: {axis}")
        normalized.append(normalized_axis)
    return tuple(normalized)


def canonical_sum_torch(
    value_t: torch.Tensor,
    *,
    dim: int | Iterable[int],
    out_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dims = _normalize_dims(value_t.ndim, dim)
    if not dims:
        return value_t.to(dtype=out_dtype)
    work_t = value_t.to(dtype=torch.float32).to(dtype=out_dtype)
    for axis in sorted(dims, reverse=True):
        work_t = work_t.sum(dim=axis, dtype=out_dtype)
    return work_t.to(dtype=out_dtype)


def canonical_sum_numpy(
    value: torch.Tensor | np.ndarray,
    *,
    dim: int | Iterable[int],
    out_dtype: np.dtype | type[np.generic] = np.float64,
) -> np.ndarray:
    tensor = value.detach().to(device="cpu", dtype=torch.float32) if torch.is_tensor(value) else torch.as_tensor(
        np.asarray(value, dtype=np.float32),
        dtype=torch.float32,
        device="cpu",
    )
    reduced_t = canonical_sum_torch(tensor, dim=dim, out_dtype=torch.float64)
    reduced_np = reduced_t.detach().cpu().numpy()
    return reduced_np.astype(out_dtype, copy=False)
