from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
from typing import Sequence

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None
_CACHE_ROOT = Path(os.environ.get("SAGIN_MARL_NATIVE_CUDA_CACHE", r"D:\sagin_marl_native_cuda_cache"))
_NINJA_READY = False
_MSVC_READY = False
_MSVC_VERSION_READY = False

SOURCE_POLICY = 0
SOURCE_ZERO = 1
SOURCE_UNIFORM = 2
SOURCE_RANDOM = 3
SOURCE_LINK_PRIORITY = 4
SOURCE_DEMAND_PRIORITY = 5
SOURCE_QUEUE_AWARE = 6
SOURCE_CLUSTER_CENTER_QUEUE_AWARE = 7
SOURCE_LYAPUNOV = 8

FLOW_BASE_EXECUTED = 0
FLOW_BASE_DETERMINISTIC = 1
FLOW_BASE_EXTERNAL_LIVE_OVERRIDE = 2

_ACTOR_RUNTIME_FLOAT_TENSOR_COUNT = 413
_ACTOR_RUNTIME_LONG_TENSOR_COUNT = 58
_ACTOR_RUNTIME_BOOL_TENSOR_COUNT = 55
_ACTOR_RUNTIME_INT_TENSOR_COUNT = 18


@dataclass(frozen=True)
class NativeCudaRuntimeABI:
    """Fixed native CUDA rollout ABI built once with runtime-owned tensors.

    Tensor vector positions are semantic enum slots understood by kernels.cu.
    This deliberately is not a task/copy table; kernels interpret these tensors
    as the rollout state, live actor I/O, stage buffers, history ring, and static
    constants needed by the final native env kernels.
    """

    float_tensors: tuple[torch.Tensor, ...]
    long_tensors: tuple[torch.Tensor, ...]
    bool_tensors: tuple[torch.Tensor, ...]
    int_tensors: tuple[torch.Tensor, ...]
    int_params: tuple[int, ...]
    float_params: tuple[float, ...]
    call_args: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[int, ...],
        tuple[float, ...],
    ] = field(init=False, repr=False)
    actor_call_args: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[int, ...],
        tuple[float, ...],
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "call_args",
            (
                self.float_tensors,
                self.long_tensors,
                self.bool_tensors,
                self.int_tensors,
                self.int_params,
                self.float_params,
            ),
        )
        object.__setattr__(
            self,
            "actor_call_args",
            (
                self.float_tensors[:_ACTOR_RUNTIME_FLOAT_TENSOR_COUNT],
                self.long_tensors[:_ACTOR_RUNTIME_LONG_TENSOR_COUNT],
                self.bool_tensors[:_ACTOR_RUNTIME_BOOL_TENSOR_COUNT],
                self.int_tensors[:_ACTOR_RUNTIME_INT_TENSOR_COUNT],
                self.int_params,
                self.float_params,
            ),
        )


@dataclass(frozen=True)
class NativeActorCudaABI:
    """Fixed native CUDA actor ABI built from a StructuredActor module.

    The weight tensor order is defined by sagin_marl.rl.native_actor_cuda and
    mirrored by actor_kernels.cu. These tensors are runtime-owned CUDA buffers;
    training code synchronizes them from PyTorch modules outside the rollout
    step hot path.
    """

    weight_tensors: tuple[torch.Tensor, ...]
    int_tensors: tuple[torch.Tensor, ...]
    int_params: tuple[int, ...]
    float_params: tuple[float, ...]
    call_args: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[int, ...],
        tuple[float, ...],
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "call_args",
            (
                self.weight_tensors,
                self.int_tensors,
                self.int_params,
                self.float_params,
            ),
        )


def _prepend_path_once(path: str) -> None:
    if not path:
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if path not in parts:
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _activate_msvc_environment_if_needed() -> None:
    global _MSVC_READY
    if os.name != "nt":
        return
    if _MSVC_READY and shutil.which("cl"):
        return
    os.environ.setdefault("VSLANG", "1033")
    if shutil.which("cl"):
        _MSVC_READY = True
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return
    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=False,
        capture_output=True,
    )
    stdout = result.stdout.decode(errors="replace") if isinstance(result.stdout, bytes) else str(result.stdout or "")
    install_path = stdout.strip().splitlines()[0] if stdout.strip() else ""
    if not install_path:
        return
    vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        cl_matches = sorted(Path(install_path).glob("VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if cl_matches:
            _prepend_path_once(str(cl_matches[-1].parent))
            if shutil.which("cl"):
                _MSVC_READY = True
                return
        return
    env_cmd = f'call "{vcvars}" >nul && set'
    env_result = subprocess.run(env_cmd, shell=True, check=False, capture_output=True)
    env_stdout = env_result.stdout.decode(errors="replace") if isinstance(env_result.stdout, bytes) else str(env_result.stdout or "")
    env_stderr = env_result.stderr.decode(errors="replace") if isinstance(env_result.stderr, bytes) else str(env_result.stderr or "")
    if env_result.returncode != 0:
        raise RuntimeError("Failed to activate the MSVC build environment with vcvars64.bat: " + env_stderr.strip())
    for line in env_stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value
    if not shutil.which("cl"):
        raise RuntimeError("MSVC cl.exe was not found after activating vcvars64.bat.")
    _MSVC_READY = True


def _ensure_ninja_on_path() -> None:
    global _NINJA_READY
    if _NINJA_READY and shutil.which("ninja"):
        return
    if shutil.which("ninja"):
        _NINJA_READY = True
        return
    try:
        import ninja  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch CUDA extensions require ninja in the active .venv.") from exc
    ninja_bin_dir = Path(str(getattr(ninja, "BIN_DIR", "")))
    ninja_exe = ninja_bin_dir / ("ninja.exe" if os.name == "nt" else "ninja")
    if not ninja_exe.exists():
        raise RuntimeError(f"ninja package is installed but executable was not found at {ninja_exe}.")
    _prepend_path_once(str(ninja_bin_dir))
    if not shutil.which("ninja"):
        raise RuntimeError("ninja executable was not found after adding the package BIN_DIR to PATH.")
    _NINJA_READY = True


def _validate_msvc_version_and_skip_broken_torch_decoder() -> None:
    global _MSVC_VERSION_READY
    if os.name != "nt":
        return
    if _MSVC_VERSION_READY:
        return
    compiler = shutil.which("cl")
    if not compiler:
        raise RuntimeError("MSVC cl.exe is required to build native CUDA kernels.")
    result = subprocess.run([compiler], check=False, capture_output=True)
    output = b"\n".join(part for part in (result.stdout, result.stderr) if part)
    text = output.decode(errors="replace")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        raise RuntimeError("Could not parse MSVC cl.exe version from compiler output.")
    version = tuple(int(part) for part in match.groups())
    if version < (19, 0, 24215):
        raise RuntimeError(f"MSVC cl.exe version {version!r} is too old for the PyTorch native CUDA extension ABI.")
    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
    _MSVC_VERSION_READY = True


def _ensure_ascii_venv_junction() -> Path:
    if os.name != "nt":
        return Path(sys.prefix)
    venv_root = Path(sys.prefix)
    link_root = _CACHE_ROOT / "venv"
    if link_root.exists():
        if not link_root.is_dir():
            raise RuntimeError(f"native CUDA cache path exists but is not a directory: {link_root}")
        return link_root
    link_root.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(f'mklink /J "{link_root}" "{venv_root}"', shell=True, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr or "")
        stdout = result.stdout.decode(errors="replace") if isinstance(result.stdout, bytes) else str(result.stdout or "")
        raise RuntimeError("Failed to create ASCII junction for the active .venv: " + (stderr.strip() or stdout.strip()))
    return link_root


def _ascii_torch_paths() -> tuple[list[str], list[str]]:
    if os.name != "nt":
        return ([], [])
    venv_link = _ensure_ascii_venv_junction()
    torch_root = venv_link / "Lib" / "site-packages" / "torch"
    include_paths = [
        str(torch_root / "include"),
        str(torch_root / "include" / "torch" / "csrc" / "api" / "include"),
    ]
    library_paths = [str(torch_root / "lib")]
    return include_paths, library_paths


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    _ensure_ninja_on_path()
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        _prepend_path_once(str(Path(cuda_home) / "bin"))
    _activate_msvc_environment_if_needed()
    _validate_msvc_version_and_skip_broken_torch_decoder()
    root = Path(__file__).resolve().parent
    source_root = _CACHE_ROOT / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    torch_extensions_dir = _CACHE_ROOT / "torch_extensions"
    torch_extensions_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
    extra_include_paths, extra_library_paths = _ascii_torch_paths()
    build_sources: list[str] = []
    for name in ("kernels.cpp", "kernels.cu", "actor_kernels.cu"):
        src = root / name
        dst = source_root / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
        build_sources.append(str(dst))
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    if os.name == "nt":
        extra_cuda_cflags.append("-allow-unsupported-compiler")
    extra_ldflags = [f"/LIBPATH:{path}" for path in extra_library_paths] if os.name == "nt" else [f"-L{path}" for path in extra_library_paths]
    _EXTENSION = load(
        name="sagin_marl_native_cuda_live",
        sources=build_sources,
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=bool(int(os.environ.get("SAGIN_MARL_NATIVE_CUDA_VERBOSE", "0") or "0")),
    )
    return _EXTENSION


def _call_args(
    abi: NativeCudaRuntimeABI,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[int, ...], tuple[float, ...]]:
    if not isinstance(abi, NativeCudaRuntimeABI):
        raise TypeError("native CUDA live binding requires a NativeCudaRuntimeABI.")
    return abi.call_args


def _actor_runtime_call_args(
    abi: NativeCudaRuntimeABI,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[int, ...], tuple[float, ...]]:
    if not isinstance(abi, NativeCudaRuntimeABI):
        raise TypeError("native CUDA actor binding requires a NativeCudaRuntimeABI.")
    return abi.actor_call_args


def _actor_call_args(
    abi: NativeActorCudaABI,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[int, ...], tuple[float, ...]]:
    if not isinstance(abi, NativeActorCudaABI):
        raise TypeError("native CUDA actor binding requires a NativeActorCudaABI.")
    return abi.call_args


def prepare_initial_accel_live(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int = 0,
    active_idx: int = 0,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.prepare_initial_accel_live(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        False,
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def accel_to_sat_live(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int,
    active_idx: int,
    accel_source_mode: int,
    sat_source_mode: int,
    bw_source_mode: int,
) -> None:
    ext = _load_extension()
    ext.accel_to_sat_live(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        False,
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def queue_aware_accel_live(
    abi: NativeCudaRuntimeABI,
    *,
    active_idx: int,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.queue_aware_accel_live(
        *_call_args(abi),
        int(active_idx),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def cluster_center_accel_live(
    abi: NativeCudaRuntimeABI,
    *,
    active_idx: int,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.cluster_center_accel_live(
        *_call_args(abi),
        int(active_idx),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def baseline_accel_live(
    abi: NativeCudaRuntimeABI,
    *,
    active_idx: int,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.baseline_accel_live(
        *_call_args(abi),
        int(active_idx),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def queue_aware_sat_live(
    abi: NativeCudaRuntimeABI,
    *,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.queue_aware_sat_live(
        *_call_args(abi),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def baseline_sat_live(
    abi: NativeCudaRuntimeABI,
    *,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.baseline_sat_live(
        *_call_args(abi),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def queue_aware_bw_live(
    abi: NativeCudaRuntimeABI,
    *,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.queue_aware_bw_live(
        *_call_args(abi),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def baseline_bw_live(
    abi: NativeCudaRuntimeABI,
    *,
    accel_source_mode: int = SOURCE_POLICY,
    sat_source_mode: int = SOURCE_POLICY,
    bw_source_mode: int = SOURCE_POLICY,
) -> None:
    ext = _load_extension()
    ext.baseline_bw_live(
        *_call_args(abi),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def sat_to_bw_live(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int,
    active_idx: int,
    accel_source_mode: int,
    sat_source_mode: int,
    bw_source_mode: int,
) -> None:
    ext = _load_extension()
    ext.sat_to_bw_live(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        False,
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def finish_commit_prepare_live(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int,
    active_idx: int,
    rollout_tail: bool,
    accel_source_mode: int,
    sat_source_mode: int,
    bw_source_mode: int,
) -> None:
    ext = _load_extension()
    ext.finish_commit_prepare_live(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        bool(rollout_tail),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def apply_bw_macro_live(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int,
    active_idx: int,
    rollout_tail: bool,
    accel_source_mode: int,
    sat_source_mode: int,
    bw_source_mode: int,
) -> None:
    ext = _load_extension()
    ext.apply_bw_macro_live(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        bool(rollout_tail),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
    )


def finish_commit_prepare_live_profiled(
    abi: NativeCudaRuntimeABI,
    *,
    slot: int,
    active_idx: int,
    rollout_tail: bool,
    accel_source_mode: int,
    sat_source_mode: int,
    bw_source_mode: int,
    profile_out: torch.Tensor,
) -> None:
    if not torch.is_tensor(profile_out):
        raise TypeError("finish internal profile requires a torch.Tensor profile_out.")
    ext = _load_extension()
    ext.finish_commit_prepare_live_profiled(
        *_call_args(abi),
        int(slot),
        int(active_idx),
        bool(rollout_tail),
        int(accel_source_mode),
        int(sat_source_mode),
        int(bw_source_mode),
        profile_out,
    )


def prepare_branch_replay_from_history(
    source_abi: NativeCudaRuntimeABI,
    target_abi: NativeCudaRuntimeABI,
    *,
    history_rows: torch.Tensor,
    stage_id: int,
) -> None:
    if not isinstance(source_abi, NativeCudaRuntimeABI) or not isinstance(target_abi, NativeCudaRuntimeABI):
        raise TypeError("native CUDA branch prepare requires source and target NativeCudaRuntimeABI objects.")
    if not torch.is_tensor(history_rows):
        raise TypeError("native CUDA branch prepare requires a history_rows tensor.")
    rows = history_rows.to(device=source_abi.float_tensors[0].device, dtype=torch.long).contiguous()
    ext = _load_extension()
    ext.prepare_branch_replay_from_history(
        *source_abi.call_args,
        *target_abi.call_args,
        rows,
        int(stage_id),
    )


def stage_mc_gae(
    values: torch.Tensor,
    mc_returns: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    returns_out: torch.Tensor,
    advantages_out: torch.Tensor,
    *,
    num_steps: int,
    num_envs: int,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Run native MC-return GAE for a dense [step, env] stage sequence."""
    ext = _load_extension()
    ext.stage_mc_gae(
        values,
        mc_returns,
        terminated,
        truncated,
        returns_out,
        advantages_out,
        int(num_steps),
        int(num_envs),
        float(gamma),
        float(gae_lambda),
    )


def actor_accel_live(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    active_idx: int,
    deterministic: bool,
    rng_step: int,
) -> None:
    ext = _load_extension()
    ext.actor_accel_live(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        int(active_idx),
        bool(deterministic),
        int(rng_step),
    )


def actor_accel_live_fused(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    active_idx: int,
    deterministic: bool,
    rng_step: int,
) -> None:
    ext = _load_extension()
    ext.actor_accel_live_fused(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        int(active_idx),
        bool(deterministic),
        int(rng_step),
    )


def actor_sat_live(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    deterministic: bool,
    rng_step: int,
) -> None:
    ext = _load_extension()
    ext.actor_sat_live(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        bool(deterministic),
        int(rng_step),
    )


def actor_sat_live_fused(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    deterministic: bool,
    rng_step: int,
) -> None:
    ext = _load_extension()
    ext.actor_sat_live_fused(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        bool(deterministic),
        int(rng_step),
    )


def actor_bw_live(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    deterministic: bool,
    rng_step: int,
    history_slot: int = -1,
) -> None:
    ext = _load_extension()
    ext.actor_bw_live(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        bool(deterministic),
        int(rng_step),
        int(history_slot),
    )


def actor_bw_live_fused(
    runtime_abi: NativeCudaRuntimeABI,
    actor_abi: NativeActorCudaABI,
    *,
    deterministic: bool,
    rng_step: int,
    history_slot: int = -1,
) -> None:
    ext = _load_extension()
    ext.actor_bw_live_fused(
        *_actor_runtime_call_args(runtime_abi),
        *_actor_call_args(actor_abi),
        bool(deterministic),
        int(rng_step),
        int(history_slot),
    )
