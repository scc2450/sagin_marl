from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Callable
import warnings

import torch

_CUDA_COMPILE_BACKEND_AVAILABLE: bool | None = None

# Env rollout kernels are launched through sagin_marl.env.native_cuda. This
# runtime remains available for non-rollout tensor helpers and actor compile
# support, but it must not own final native env segment names.
_CUDA_GRAPH_DIRECT_INPUT_KERNELS: set[str] = set()


@dataclass
class StructuredKernelCompileStats:
    enabled: set[str] = field(default_factory=set)
    disabled: set[str] = field(default_factory=set)
    fallback: dict[str, str] = field(default_factory=dict)


class StructuredKernelRuntime:
    def __init__(self, cfg: Any, *, device: torch.device | str | None) -> None:
        self._cfg = cfg
        self._device = None if device is None else torch.device(device)
        self._cache: dict[str, Callable[..., Any]] = {}
        self.stats = StructuredKernelCompileStats()

    @property
    def compile_enabled(self) -> bool:
        mode = str(getattr(self._cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower()
        if mode not in {"auto", "compile"}:
            return False
        if not hasattr(torch, "compile"):
            return False
        if self._device is None:
            return mode == "compile"
        if self._device.type == "cuda":
            if not self._cuda_compile_backend_available() and self._compile_backend() != "cudagraphs":
                self.stats.disabled.add("cuda_compile_backend_unavailable")
                return False
            return True
        return mode == "compile"

    def _compile_backend(self) -> str | None:
        backend = str(getattr(self._cfg, "structured_kernel_compile_backend", "auto") or "auto").strip().lower()
        if backend in {"", "default", "inductor"}:
            return None
        if backend in {"cudagraph", "cuda_graph", "cuda-graphs"}:
            return "cudagraphs"
        if backend != "auto":
            return backend
        if self._device is not None and self._device.type == "cuda" and not self._cuda_compile_backend_available():
            return "cudagraphs"
        return None

    @staticmethod
    def _cuda_compile_backend_available() -> bool:
        global _CUDA_COMPILE_BACKEND_AVAILABLE
        if _CUDA_COMPILE_BACKEND_AVAILABLE is None:
            _CUDA_COMPILE_BACKEND_AVAILABLE = importlib.util.find_spec("triton") is not None
        return bool(_CUDA_COMPILE_BACKEND_AVAILABLE)

    def _auto_compile_min_numel(self) -> int:
        mode = str(getattr(self._cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower()
        if mode != "auto":
            return 0
        if self._device is None or self._device.type != "cuda":
            return 0
        return max(int(getattr(self._cfg, "structured_kernel_compile_min_numel", 0) or 0), 0)

    def _auto_compile_warmup_calls(self) -> int:
        mode = str(getattr(self._cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower()
        if mode != "auto":
            return 0
        if self._device is None or self._device.type != "cuda":
            return 0
        default_warmup = 4 if self._compile_backend() == "cudagraphs" else 8
        return max(int(getattr(self._cfg, "structured_kernel_auto_compile_warmup_calls", default_warmup) or 0), 0)

    def _mark_cudagraph_step_begin(self) -> None:
        if not bool(getattr(self._cfg, "structured_kernel_compile_cudagraphs", False)):
            return
        if not bool(getattr(self._cfg, "structured_kernel_cudagraph_mark_step_begin", False)):
            return
        if self._device is None or self._device.type != "cuda":
            return
        mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if callable(mark_step):
            mark_step()

    def _compile_allowlist(self) -> set[str] | None:
        raw = getattr(self._cfg, "structured_kernel_compile_allowlist", None)
        if raw is None:
            return None
        if isinstance(raw, str):
            text = raw.strip()
            if not text or text.lower() in {"*", "all"}:
                return None
            return {item.strip() for item in text.split(",") if item.strip()}
        try:
            return {str(item).strip() for item in raw if str(item).strip()}
        except TypeError:
            return None

    def _required_compile_names(self) -> set[str] | None:
        raw = getattr(self._cfg, "structured_kernel_compile_required", None)
        if raw is None:
            return set()
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return set()
            if text.lower() in {"*", "all"}:
                return None
            return {item.strip() for item in text.split(",") if item.strip()}
        try:
            return {str(item).strip() for item in raw if str(item).strip()}
        except TypeError:
            return set()

    def _compile_required(self, name: str) -> bool:
        if self._device is None or self._device.type != "cuda":
            return False
        required = self._required_compile_names()
        return required is None or str(name) in required

    def _strict_compile_error(self, name: str, reason: str) -> RuntimeError:
        return RuntimeError(
            f"CUDA native kernel {name!r} is marked as compile-required but {reason}. "
            "The official GPU live path must not silently fall back to eager Python tensor-op dispatch."
        )

    def _raise_on_strict_cudagraph_warning(self, name: str, captured: list[warnings.WarningMessage]) -> None:
        for record in captured:
            message = str(record.message)
            lowered = message.lower()
            if (
                "cuda graph is empty" in lowered
                or "skipping cudagraph" in lowered
                or ("cudagraph" in lowered and "skip" in lowered)
            ):
                raise self._strict_compile_error(name, f"cudagraph capture emitted warning: {message}")

    def _call_compiled_kernel(
        self,
        *,
        name: str,
        compiled_fn: Callable[..., Any],
        strict_required: bool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self._mark_cudagraph_step_begin()
        if strict_required and self._compile_backend() == "cudagraphs":
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                value = compiled_fn(*args, **kwargs)
            self._raise_on_strict_cudagraph_warning(name, captured)
            return self._maybe_clone_cudagraph_outputs(name, value)
        return self._maybe_clone_cudagraph_outputs(name, compiled_fn(*args, **kwargs))

    @staticmethod
    def _tensor_numel(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel())
        if isinstance(value, dict):
            return sum(StructuredKernelRuntime._tensor_numel(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(StructuredKernelRuntime._tensor_numel(item) for item in value)
        return 0

    @classmethod
    def _call_numel(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        return sum(cls._tensor_numel(item) for item in args) + sum(cls._tensor_numel(item) for item in kwargs.values())

    @staticmethod
    def _tensor_field_names(value: Any) -> tuple[str, ...] | None:
        field_names = getattr(value, "_tensor_fields", None)
        if field_names is None:
            return None
        return tuple(str(name) for name in field_names)

    @staticmethod
    def _rebuild_tensor_field_object(value: Any, field_values: dict[str, Any]) -> Any:
        return type(value)(**field_values)

    @staticmethod
    def _clone_cudagraph_outputs(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: StructuredKernelRuntime._clone_cudagraph_outputs(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(StructuredKernelRuntime._clone_cudagraph_outputs(item) for item in value)
        if isinstance(value, list):
            return [StructuredKernelRuntime._clone_cudagraph_outputs(item) for item in value]
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            return StructuredKernelRuntime._rebuild_tensor_field_object(
                value,
                {
                    field_name: StructuredKernelRuntime._clone_cudagraph_outputs(getattr(value, field_name))
                    for field_name in tensor_fields
                },
            )
        return value

    @staticmethod
    def _empty_like_output_buffers(value: Any) -> Any:
        if torch.is_tensor(value):
            return torch.empty_like(value)
        if isinstance(value, dict):
            return {key: StructuredKernelRuntime._empty_like_output_buffers(item) for key, item in value.items()}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(StructuredKernelRuntime._empty_like_output_buffers(item) for item in value))
        if isinstance(value, tuple):
            return tuple(StructuredKernelRuntime._empty_like_output_buffers(item) for item in value)
        if isinstance(value, list):
            return [StructuredKernelRuntime._empty_like_output_buffers(item) for item in value]
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            return StructuredKernelRuntime._rebuild_tensor_field_object(
                value,
                {
                    field_name: StructuredKernelRuntime._empty_like_output_buffers(getattr(value, field_name))
                    for field_name in tensor_fields
                },
            )
        return value

    @staticmethod
    def _copy_into_output_buffers_(target: Any, value: Any) -> Any:
        if torch.is_tensor(value):
            if not torch.is_tensor(target):
                raise TypeError("compiled CUDA output buffer structure does not match tensor output.")
            if tuple(target.shape) != tuple(value.shape) or target.dtype != value.dtype or target.device != value.device:
                raise RuntimeError(
                    "compiled CUDA output shape/dtype/device changed after capture; "
                    "official native main-kernel segments require fixed output buffers."
                )
            target.copy_(value)
            return target
        if isinstance(value, dict):
            if not isinstance(target, dict) or set(target.keys()) != set(value.keys()):
                raise TypeError("compiled CUDA output buffer structure does not match dict output.")
            return {
                key: StructuredKernelRuntime._copy_into_output_buffers_(target[key], item)
                for key, item in value.items()
            }
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            if type(target) is not type(value) or len(target) != len(value):
                raise TypeError("compiled CUDA output buffer structure does not match namedtuple output.")
            return type(value)(
                *(
                    StructuredKernelRuntime._copy_into_output_buffers_(target_item, value_item)
                    for target_item, value_item in zip(target, value)
                )
            )
        if isinstance(value, tuple):
            if not isinstance(target, tuple) or len(target) != len(value):
                raise TypeError("compiled CUDA output buffer structure does not match tuple output.")
            return tuple(
                StructuredKernelRuntime._copy_into_output_buffers_(target_item, value_item)
                for target_item, value_item in zip(target, value)
            )
        if isinstance(value, list):
            if not isinstance(target, list) or len(target) != len(value):
                raise TypeError("compiled CUDA output buffer structure does not match list output.")
            return [
                StructuredKernelRuntime._copy_into_output_buffers_(target_item, value_item)
                for target_item, value_item in zip(target, value)
            ]
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            if type(target) is not type(value):
                raise TypeError("compiled CUDA output buffer structure does not match tensor-field output.")
            return StructuredKernelRuntime._rebuild_tensor_field_object(
                value,
                {
                    field_name: StructuredKernelRuntime._copy_into_output_buffers_(
                        getattr(target, field_name),
                        getattr(value, field_name),
                    )
                    for field_name in tensor_fields
                },
            )
        return value

    @staticmethod
    def _static_cuda_graph_inputs(value: Any, *, direct_inputs: bool = False) -> Any:
        if torch.is_tensor(value):
            if value.device.type != "cuda":
                raise TypeError("strict CUDA graph segment inputs must be CUDA tensors.")
            if direct_inputs:
                return value.detach()
            return torch.empty_like(value).copy_(value)
        if isinstance(value, dict):
            return {
                key: StructuredKernelRuntime._static_cuda_graph_inputs(item, direct_inputs=direct_inputs)
                for key, item in value.items()
            }
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(
                *(StructuredKernelRuntime._static_cuda_graph_inputs(item, direct_inputs=direct_inputs) for item in value)
            )
        if isinstance(value, tuple):
            return tuple(
                StructuredKernelRuntime._static_cuda_graph_inputs(item, direct_inputs=direct_inputs)
                for item in value
            )
        if isinstance(value, list):
            return [
                StructuredKernelRuntime._static_cuda_graph_inputs(item, direct_inputs=direct_inputs)
                for item in value
            ]
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            return StructuredKernelRuntime._rebuild_tensor_field_object(
                value,
                {
                    field_name: StructuredKernelRuntime._static_cuda_graph_inputs(
                        getattr(value, field_name),
                        direct_inputs=direct_inputs,
                    )
                    for field_name in tensor_fields
                },
            )
        return value

    @staticmethod
    def _copy_into_static_cuda_graph_inputs_(target: Any, value: Any) -> Any:
        if torch.is_tensor(value):
            if not torch.is_tensor(target):
                raise TypeError("strict CUDA graph input structure changed from tensor to non-tensor.")
            if tuple(target.shape) != tuple(value.shape) or target.dtype != value.dtype or target.device != value.device:
                raise RuntimeError(
                    "strict CUDA graph input shape/dtype/device changed after capture; "
                    "official native main-kernel segments require fixed input buffers."
                )
            target.copy_(value)
            return target
        if isinstance(value, dict):
            if not isinstance(target, dict) or set(target.keys()) != set(value.keys()):
                raise TypeError("strict CUDA graph input dict structure changed after capture.")
            for key, item in value.items():
                StructuredKernelRuntime._copy_into_static_cuda_graph_inputs_(target[key], item)
            return target
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            if type(target) is not type(value) or len(target) != len(value):
                raise TypeError("strict CUDA graph input namedtuple structure changed after capture.")
            for target_item, value_item in zip(target, value):
                StructuredKernelRuntime._copy_into_static_cuda_graph_inputs_(target_item, value_item)
            return target
        if isinstance(value, tuple):
            if not isinstance(target, tuple) or len(target) != len(value):
                raise TypeError("strict CUDA graph input tuple structure changed after capture.")
            for target_item, value_item in zip(target, value):
                StructuredKernelRuntime._copy_into_static_cuda_graph_inputs_(target_item, value_item)
            return target
        if isinstance(value, list):
            if not isinstance(target, list) or len(target) != len(value):
                raise TypeError("strict CUDA graph input list structure changed after capture.")
            for target_item, value_item in zip(target, value):
                StructuredKernelRuntime._copy_into_static_cuda_graph_inputs_(target_item, value_item)
            return target
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            if type(target) is not type(value):
                raise TypeError("strict CUDA graph input tensor-field structure changed after capture.")
            target_fields = StructuredKernelRuntime._tensor_field_names(target)
            if target_fields != tensor_fields:
                raise TypeError("strict CUDA graph input tensor-field names changed after capture.")
            for field_name in tensor_fields:
                StructuredKernelRuntime._copy_into_static_cuda_graph_inputs_(
                    getattr(target, field_name),
                    getattr(value, field_name),
                )
            return target
        if target != value:
            raise RuntimeError(
                "strict CUDA graph non-tensor input changed after capture; "
                "configuration and scalar control values must be frozen during graph build."
            )
        return target

    @staticmethod
    def _validate_static_cuda_graph_inputs_(target: Any, value: Any, *, path: str = "input") -> Any:
        if torch.is_tensor(value):
            if not torch.is_tensor(target):
                raise TypeError(f"strict CUDA graph direct {path} changed from tensor to non-tensor.")
            if tuple(target.shape) != tuple(value.shape) or target.dtype != value.dtype or target.device != value.device:
                raise RuntimeError(
                    f"strict CUDA graph direct {path} shape/dtype/device changed after capture; "
                    "official native main-kernel segments require fixed input buffers."
                )
            if target.data_ptr() != value.data_ptr():
                raise RuntimeError(
                    f"strict CUDA graph direct {path} data pointer changed after capture; "
                    "official native main-kernel segments must use persistent input buffers instead of replay-time copies."
                )
            return target
        if isinstance(value, dict):
            if not isinstance(target, dict) or set(target.keys()) != set(value.keys()):
                raise TypeError(f"strict CUDA graph direct {path} dict structure changed after capture.")
            for key, item in value.items():
                StructuredKernelRuntime._validate_static_cuda_graph_inputs_(
                    target[key],
                    item,
                    path=f"{path}.{key}",
                )
            return target
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            if type(target) is not type(value) or len(target) != len(value):
                raise TypeError(f"strict CUDA graph direct {path} namedtuple structure changed after capture.")
            for field_name, target_item, value_item in zip(value._fields, target, value):
                StructuredKernelRuntime._validate_static_cuda_graph_inputs_(
                    target_item,
                    value_item,
                    path=f"{path}.{field_name}",
                )
            return target
        if isinstance(value, tuple):
            if not isinstance(target, tuple) or len(target) != len(value):
                raise TypeError(f"strict CUDA graph direct {path} tuple structure changed after capture.")
            for index, (target_item, value_item) in enumerate(zip(target, value)):
                StructuredKernelRuntime._validate_static_cuda_graph_inputs_(
                    target_item,
                    value_item,
                    path=f"{path}[{index}]",
                )
            return target
        if isinstance(value, list):
            if not isinstance(target, list) or len(target) != len(value):
                raise TypeError(f"strict CUDA graph direct {path} list structure changed after capture.")
            for index, (target_item, value_item) in enumerate(zip(target, value)):
                StructuredKernelRuntime._validate_static_cuda_graph_inputs_(
                    target_item,
                    value_item,
                    path=f"{path}[{index}]",
                )
            return target
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            if type(target) is not type(value):
                raise TypeError(f"strict CUDA graph direct {path} tensor-field structure changed after capture.")
            target_fields = StructuredKernelRuntime._tensor_field_names(target)
            if target_fields != tensor_fields:
                raise TypeError(f"strict CUDA graph direct {path} tensor-field names changed after capture.")
            for field_name in tensor_fields:
                StructuredKernelRuntime._validate_static_cuda_graph_inputs_(
                    getattr(target, field_name),
                    getattr(value, field_name),
                    path=f"{path}.{field_name}",
                )
            return target
        if target != value:
            raise RuntimeError(
                f"strict CUDA graph direct non-tensor {path} changed after capture; "
                "configuration and scalar control values must be frozen during graph build."
            )
        return target

    @staticmethod
    def _collect_unique_cuda_tensors(value: Any, out: dict[int, torch.Tensor]) -> None:
        if torch.is_tensor(value):
            if value.device.type == "cuda":
                out.setdefault(int(value.data_ptr()), value)
            return
        if isinstance(value, dict):
            for item in value.values():
                StructuredKernelRuntime._collect_unique_cuda_tensors(item, out)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                StructuredKernelRuntime._collect_unique_cuda_tensors(item, out)
            return
        tensor_fields = getattr(value, "_tensor_fields", None)
        if tensor_fields is not None:
            for field_name in tensor_fields:
                StructuredKernelRuntime._collect_unique_cuda_tensors(getattr(value, field_name, None), out)

    @staticmethod
    def _cuda_graph_input_signature(value: Any) -> Any:
        if torch.is_tensor(value):
            return (
                "tensor",
                str(value.device),
                str(value.dtype),
                tuple(int(dim) for dim in value.shape),
                int(value.data_ptr()),
            )
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (key, StructuredKernelRuntime._cuda_graph_input_signature(value[key]))
                    for key in sorted(value.keys())
                ),
            )
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return (
                "namedtuple",
                type(value).__name__,
                tuple(
                    (field_name, StructuredKernelRuntime._cuda_graph_input_signature(item))
                    for field_name, item in zip(value._fields, value)
                ),
            )
        if isinstance(value, tuple):
            return ("tuple", tuple(StructuredKernelRuntime._cuda_graph_input_signature(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(StructuredKernelRuntime._cuda_graph_input_signature(item) for item in value))
        tensor_fields = StructuredKernelRuntime._tensor_field_names(value)
        if tensor_fields is not None:
            return (
                "tensor_fields",
                type(value).__name__,
                tuple(
                    (field_name, StructuredKernelRuntime._cuda_graph_input_signature(getattr(value, field_name)))
                    for field_name in tensor_fields
                ),
            )
        if value is None or isinstance(value, (bool, int, float, str)):
            return ("scalar", type(value).__name__, value)
        return ("object", type(value).__name__, id(value))

    @classmethod
    def _snapshot_cuda_tensors(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        tensors: dict[int, torch.Tensor] = {}
        cls._collect_unique_cuda_tensors(args, tensors)
        cls._collect_unique_cuda_tensors(kwargs, tensors)
        return [(tensor, tensor.detach().clone()) for tensor in tensors.values()]

    @staticmethod
    def _restore_cuda_tensor_snapshots_(snapshots: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for target, snapshot in snapshots:
            target.copy_(snapshot)

    def _cudagraph_clone_output_kernels(self) -> set[str]:
        raw = getattr(self._cfg, "structured_kernel_cudagraph_clone_output_kernels", None)
        if raw is None:
            return {"visible_sats"}
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return set()
            if text.lower() in {"*", "all"}:
                return {"*"}
            return {item.strip() for item in text.split(",") if item.strip()}
        try:
            return {str(item).strip() for item in raw if str(item).strip()}
        except TypeError:
            return {"visible_sats"}

    def _maybe_clone_cudagraph_outputs(self, name: str, value: Any) -> Any:
        if not bool(getattr(self._cfg, "structured_kernel_compile_cudagraphs", False)):
            return value
        if self._device is None or self._device.type != "cuda":
            return value
        clone_kernels = self._cudagraph_clone_output_kernels()
        if "*" not in clone_kernels and name not in clone_kernels:
            return value
        return self._clone_cudagraph_outputs(value)

    def compile_kernel(self, name: str, eager_fn: Callable[..., Any]) -> Callable[..., Any]:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        strict_required = self._compile_required(name)
        if not self.compile_enabled:
            if strict_required:
                raise self._strict_compile_error(name, "kernel compilation is disabled or unavailable")
            self.stats.disabled.add(name)
            self._cache[name] = eager_fn
            return eager_fn
        mode = str(getattr(self._cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower()
        allowlist = self._compile_allowlist()
        if mode == "auto" and allowlist is not None and name not in allowlist:
            if strict_required:
                raise self._strict_compile_error(name, "it is missing from structured_kernel_compile_allowlist")
            self.stats.disabled.add(f"{name}:not_allowlisted")
            self._cache[name] = eager_fn
            return eager_fn
        if (
            self._device is not None
            and self._device.type == "cuda"
            and str(name) in _CUDA_GRAPH_DIRECT_INPUT_KERNELS
            and not strict_required
        ):
            self.stats.disabled.add(f"{name}:stateful_direct_requires_explicit_compile")
            self._cache[name] = eager_fn
            return eager_fn

        min_numel = 0 if strict_required else self._auto_compile_min_numel()
        warmup_calls = 0 if strict_required else self._auto_compile_warmup_calls()
        if min_numel > 0 or warmup_calls > 0:
            state: dict[str, Any] = {"compiled_fn": None, "enabled": True, "eligible_calls": 0}

            def _lazy_runner(*args, **kwargs):
                if not state["enabled"]:
                    return eager_fn(*args, **kwargs)
                if self._call_numel(args, kwargs) < min_numel:
                    return eager_fn(*args, **kwargs)
                state["eligible_calls"] = int(state["eligible_calls"]) + 1
                if int(state["eligible_calls"]) <= warmup_calls:
                    return eager_fn(*args, **kwargs)
                compiled_fn = state.get("compiled_fn")
                if compiled_fn is None:
                    self._configure_inductor_runtime()
                    compile_mode = str(
                        getattr(self._cfg, "structured_kernel_compile_mode", "reduce-overhead") or "reduce-overhead"
                    ).strip()
                    fullgraph = bool(getattr(self._cfg, "structured_kernel_compile_fullgraph", False))
                    if strict_required:
                        fullgraph = bool(
                            getattr(self._cfg, "structured_kernel_compile_required_fullgraph", False)
                        ) or fullgraph
                    dynamic = bool(getattr(self._cfg, "structured_kernel_compile_dynamic", False))
                    try:
                        compile_backend = self._compile_backend()
                        if compile_backend == "cudagraphs":
                            compiled_fn = torch.compile(eager_fn, backend=compile_backend)
                        elif compile_backend is None:
                            compiled_fn = torch.compile(
                                eager_fn,
                                mode=compile_mode,
                                fullgraph=fullgraph,
                                dynamic=dynamic,
                            )
                        else:
                            compiled_fn = torch.compile(
                                eager_fn,
                                backend=compile_backend,
                                mode=compile_mode,
                                fullgraph=fullgraph,
                                dynamic=dynamic,
                            )
                    except Exception as exc:  # pragma: no cover - runtime fallback
                        if strict_required:
                            raise self._strict_compile_error(
                                name,
                                f"compile initialization failed with {type(exc).__name__}",
                            ) from exc
                        state["enabled"] = False
                        self.stats.disabled.add(name)
                        self.stats.fallback[name] = f"compile_init_failed:{type(exc).__name__}"
                        return eager_fn(*args, **kwargs)
                    state["compiled_fn"] = compiled_fn
                    self.stats.enabled.add(name)
                try:
                    return self._call_compiled_kernel(
                        name=name,
                        compiled_fn=compiled_fn,
                        strict_required=strict_required,
                        args=args,
                        kwargs=kwargs,
                    )
                except Exception as exc:  # pragma: no cover - runtime fallback
                    if strict_required and "is marked as compile-required" in str(exc):
                        raise
                    if strict_required:
                        raise self._strict_compile_error(
                            name,
                            f"compiled execution failed with {type(exc).__name__}",
                        ) from exc
                    state["enabled"] = False
                    self.stats.disabled.add(name)
                    self.stats.fallback[name] = f"compile_runtime_failed:{type(exc).__name__}"
                    return eager_fn(*args, **kwargs)

            reason = "auto_lazy"
            if min_numel > 0:
                reason += f":min_numel={min_numel}"
            if warmup_calls > 0:
                reason += f":warmup_calls={warmup_calls}"
            self.stats.disabled.add(f"{name}:{reason}")
            self._cache[name] = _lazy_runner
            return _lazy_runner

        compile_backend = self._compile_backend()
        if (
            strict_required
            and self._device is not None
            and self._device.type == "cuda"
            and str(name) in _CUDA_GRAPH_DIRECT_INPUT_KERNELS
        ):
            compile_backend = "cudagraphs"
        if strict_required and compile_backend == "cudagraphs":
            direct_inputs = bool(
                getattr(self._cfg, "structured_kernel_cudagraph_direct_inputs", False)
                and str(name) in _CUDA_GRAPH_DIRECT_INPUT_KERNELS
            )
            state: dict[str, Any] = {
                "graph": None,
                "args": None,
                "kwargs": None,
                "buffers": None,
                "variants": {},
            }

            def _buffered_runner(*args, **kwargs):
                active_state = state
                if direct_inputs:
                    signature = (
                        StructuredKernelRuntime._cuda_graph_input_signature(args),
                        StructuredKernelRuntime._cuda_graph_input_signature(kwargs),
                    )
                    variants = state.setdefault("variants", {})
                    active_state = variants.get(signature)
                    if active_state is None:
                        if len(variants) >= 8:
                            raise self._strict_compile_error(
                                name,
                                "manual cudagraph direct-input variants exceeded 8; "
                                "official native main-kernel segments must use a bounded set of persistent buffers",
                            )
                        active_state = {
                            "graph": None,
                            "args": None,
                            "kwargs": None,
                            "buffers": None,
                        }
                        variants[signature] = active_state
                graph = active_state.get("graph")
                if graph is None:
                    snapshots = self._snapshot_cuda_tensors(args, kwargs) if direct_inputs else []
                    try:
                        with torch.no_grad():
                            sample_output = eager_fn(*args, **kwargs)
                        output_buffers = self._empty_like_output_buffers(sample_output)
                        static_args = self._static_cuda_graph_inputs(args, direct_inputs=direct_inputs)
                        static_kwargs = self._static_cuda_graph_inputs(kwargs, direct_inputs=direct_inputs)

                        def _buffered_eager_fn():
                            with torch.no_grad():
                                value = eager_fn(*static_args, **static_kwargs)
                            return StructuredKernelRuntime._copy_into_output_buffers_(output_buffers, value)

                        capture_stream = torch.cuda.Stream()
                        capture_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(capture_stream):
                            _buffered_eager_fn()
                        torch.cuda.current_stream().wait_stream(capture_stream)
                        graph_local = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph_local):
                            _buffered_eager_fn()
                    except Exception as exc:  # pragma: no cover - runtime fallback
                        self._restore_cuda_tensor_snapshots_(snapshots)
                        raise self._strict_compile_error(
                            name,
                            f"manual cudagraph capture initialization failed with {type(exc).__name__}",
                        ) from exc
                    self._restore_cuda_tensor_snapshots_(snapshots)
                    active_state["args"] = static_args
                    active_state["kwargs"] = static_kwargs
                    active_state["buffers"] = output_buffers
                    active_state["graph"] = graph_local
                    graph = graph_local
                try:
                    if direct_inputs:
                        self._validate_static_cuda_graph_inputs_(active_state["args"], args)
                        self._validate_static_cuda_graph_inputs_(active_state["kwargs"], kwargs)
                    else:
                        self._copy_into_static_cuda_graph_inputs_(active_state["args"], args)
                        self._copy_into_static_cuda_graph_inputs_(active_state["kwargs"], kwargs)
                    graph.replay()
                    return active_state["buffers"]
                except Exception as exc:  # pragma: no cover - runtime fallback
                    raise self._strict_compile_error(
                        name,
                        f"manual cudagraph replay failed with {type(exc).__name__}",
                    ) from exc

            self.stats.enabled.add(name)
            self._cache[name] = _buffered_runner
            return _buffered_runner

        self._configure_inductor_runtime()
        compile_mode = str(
            getattr(self._cfg, "structured_kernel_compile_mode", "reduce-overhead") or "reduce-overhead"
        ).strip()
        fullgraph = bool(getattr(self._cfg, "structured_kernel_compile_fullgraph", False))
        if strict_required:
            fullgraph = bool(getattr(self._cfg, "structured_kernel_compile_required_fullgraph", False)) or fullgraph
        dynamic = bool(getattr(self._cfg, "structured_kernel_compile_dynamic", False))
        try:
            if compile_backend == "cudagraphs":
                compiled_fn = torch.compile(eager_fn, backend=compile_backend)
            elif compile_backend is None:
                compiled_fn = torch.compile(
                    eager_fn,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            else:
                compiled_fn = torch.compile(
                    eager_fn,
                    backend=compile_backend,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
        except Exception as exc:  # pragma: no cover - runtime fallback
            if strict_required:
                raise self._strict_compile_error(
                    name,
                    f"compile initialization failed with {type(exc).__name__}",
                ) from exc
            self.stats.disabled.add(name)
            self.stats.fallback[name] = f"compile_init_failed:{type(exc).__name__}"
            self._cache[name] = eager_fn
            return eager_fn

        state = {"enabled": True}
        def _runner(*args, **kwargs):
            if not state["enabled"]:
                return eager_fn(*args, **kwargs)
            try:
                return self._call_compiled_kernel(
                    name=name,
                    compiled_fn=compiled_fn,
                    strict_required=strict_required,
                    args=args,
                    kwargs=kwargs,
                )
            except Exception as exc:  # pragma: no cover - runtime fallback
                if strict_required and "is marked as compile-required" in str(exc):
                    raise
                if strict_required:
                    raise self._strict_compile_error(
                        name,
                        f"compiled execution failed with {type(exc).__name__}",
                    ) from exc
                state["enabled"] = False
                self.stats.disabled.add(name)
                self.stats.fallback[name] = f"compile_runtime_failed:{type(exc).__name__}"
                return eager_fn(*args, **kwargs)

        self.stats.enabled.add(name)
        self._cache[name] = _runner
        return _runner

    def _configure_inductor_runtime(self) -> None:
        if bool(getattr(self._cfg, "structured_kernel_compile_cudagraphs", False)):
            return
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
        except Exception:
            return


def get_structured_kernel_runtime(
    cfg: Any,
    *,
    device: torch.device | str | None,
) -> StructuredKernelRuntime:
    runtime_cache = getattr(cfg, "_structured_kernel_runtime_cache", None)
    if not isinstance(runtime_cache, dict):
        runtime_cache = {}
        setattr(cfg, "_structured_kernel_runtime_cache", runtime_cache)
    device_key = "none" if device is None else str(torch.device(device))
    runtime = runtime_cache.get(device_key)
    if isinstance(runtime, StructuredKernelRuntime):
        return runtime
    runtime = StructuredKernelRuntime(cfg, device=device)
    runtime_cache[device_key] = runtime
    return runtime
