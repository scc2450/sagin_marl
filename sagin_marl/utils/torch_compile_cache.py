"""Utilities for making torch.compile cache usage explicit at train startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


RECOMMENDED_INDUCTOR_CACHE = r"D:\sagin_cache\torchinductor"
RECOMMENDED_TRITON_CACHE = r"D:\sagin_cache\triton"


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _is_c_drive(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).drive.lower() == "c:"


def _actual_inductor_cache_dir() -> str | None:
    try:
        import torch._inductor.codecache as codecache

        return str(codecache.cache_dir())
    except Exception:
        return None


def _compile_cache_expected(cfg: Any | None) -> bool:
    if cfg is None:
        return True
    compile_flags = (
        "critic_compile_enabled",
        "stage_actor_compile_enabled",
        "accel_actor_compile_enabled",
        "sat_actor_compile_enabled",
        "bw_actor_compile_enabled",
    )
    return any(bool(getattr(cfg, name, False)) for name in compile_flags)


def report_torch_compile_cache(
    *,
    context: str,
    device: Any | None = None,
    cfg: Any | None = None,
    warn_if_unset: bool = True,
    warn_if_c_drive: bool = True,
) -> None:
    """Print cache paths and warn about paths that are bad for this project.

    This intentionally does not set environment variables.  Cache locations are
    process-level/user-level deployment settings; silently changing them in code
    would hide an important requirement when moving the project to another
    machine.
    """

    device_type = str(getattr(device, "type", device) or "")
    if device_type and device_type != "cuda":
        return
    if not _compile_cache_expected(cfg):
        return

    inductor_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    triton_env = os.environ.get("TRITON_CACHE_DIR")
    actual_inductor = _actual_inductor_cache_dir()

    for raw_path in (inductor_env, triton_env):
        if raw_path:
            Path(raw_path).mkdir(parents=True, exist_ok=True)

    print(
        f"[torch-compile-cache:{context}] "
        f"TORCHINDUCTOR_CACHE_DIR={inductor_env or '<unset>'} "
        f"TRITON_CACHE_DIR={triton_env or '<unset>'} "
        f"torchinductor_cache_dir={actual_inductor or '<unknown>'}"
    )

    warnings: list[str] = []
    if warn_if_unset and not inductor_env:
        warnings.append(
            "TORCHINDUCTOR_CACHE_DIR is unset; PyTorch will use its default cache directory."
        )
    if warn_if_unset and not triton_env:
        warnings.append("TRITON_CACHE_DIR is unset; Triton will use its default cache directory.")

    for label, raw_path in (
        ("TORCHINDUCTOR_CACHE_DIR", inductor_env),
        ("TRITON_CACHE_DIR", triton_env),
        ("torchinductor_cache_dir", actual_inductor),
    ):
        if raw_path and not _is_ascii(str(raw_path)):
            warnings.append(f"{label} contains non-ASCII characters; Triton/Inductor may fail on Windows.")
        if warn_if_c_drive and raw_path and _is_c_drive(str(raw_path)):
            warnings.append(f"{label} is on C:; this project recommends a D: ASCII cache path.")

    if warnings:
        print(
            f"[torch-compile-cache:{context}] WARNING: "
            + " ".join(warnings)
            + f" Recommended PowerShell setup: "
            + f"$env:TORCHINDUCTOR_CACHE_DIR='{RECOMMENDED_INDUCTOR_CACHE}'; "
            + f"$env:TRITON_CACHE_DIR='{RECOMMENDED_TRITON_CACHE}'"
        )
