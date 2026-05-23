from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _merge_tensor_with_overlap(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    merged = target.detach().clone()
    source_cast = source.detach().to(device=merged.device, dtype=merged.dtype)
    slices = tuple(slice(0, min(int(s), int(t))) for s, t in zip(source_cast.shape, merged.shape))
    if not slices:
        return merged
    merged[slices] = source_cast[slices]
    return merged


def _remap_legacy_key(key: str, current_state: Dict[str, torch.Tensor]) -> str:
    return key


def load_state_dict_forgiving(
    module: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, object]:
    current_state = module.state_dict()
    adapted_state: Dict[str, torch.Tensor] = {}
    adapted_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    skipped_keys: list[tuple[str, tuple[int, ...] | None, tuple[int, ...] | None]] = []

    for key, value in state_dict.items():
        target_key = _remap_legacy_key(key, current_state)
        if target_key not in current_state:
            continue
        target = current_state[target_key]
        if not isinstance(value, torch.Tensor):
            skipped_keys.append((target_key, None, tuple(int(v) for v in target.shape)))
            continue
        if tuple(value.shape) == tuple(target.shape):
            adapted_state[target_key] = value.to(device=target.device, dtype=target.dtype)
            continue
        if value.ndim == target.ndim:
            adapted_state[target_key] = _merge_tensor_with_overlap(target, value)
            adapted_keys.append((target_key, tuple(int(v) for v in value.shape), tuple(int(v) for v in target.shape)))
            continue
        skipped_keys.append((target_key, tuple(int(v) for v in value.shape), tuple(int(v) for v in target.shape)))

    missing_keys, unexpected_keys = module.load_state_dict(adapted_state, strict=False)
    info = {
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "adapted_keys": adapted_keys,
        "skipped_keys": skipped_keys,
    }
    if strict and (info["missing_keys"] or info["unexpected_keys"] or info["skipped_keys"]):
        raise RuntimeError(
            "Strict checkpoint load failed: "
            f"missing={info['missing_keys']}, unexpected={info['unexpected_keys']}, skipped={info['skipped_keys']}"
        )
    return info


def load_checkpoint_forgiving(
    module: nn.Module,
    path: str,
    map_location: str | torch.device | None = None,
    strict: bool = False,
) -> Dict[str, object]:
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint '{path}' did not contain a state_dict dictionary.")
    info = load_state_dict_forgiving(module, state, strict=strict)
    info["path"] = path
    return info
