import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import stage_optimizer_params as _stage_optimizer_params
from sagin_marl.rl.structured_mappo import _slice_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group

from scripts.train_joint_mcgae import (
    _collect_joint_rollout,
    _force_joint_config,
    _make_joint_learner,
)
from scripts.train_stage_mcgae import (
    _BwActorLossChunkModule,
    _strict_chunk_size,
    _train_stage_critic_on_stage,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_inductor_trace(trace_dir: Path, *, keep_caches: bool = False) -> None:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.error_on_recompile = True
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        torch._dynamo.config.fail_on_recompile_limit_hit = True

    trace_dir.mkdir(parents=True, exist_ok=True)
    inductor_config.trace.enabled = True
    inductor_config.trace.debug_dir = str(trace_dir)
    inductor_config.trace.output_code = True
    inductor_config.trace.fx_graph = True
    inductor_config.trace.ir_pre_fusion = True
    inductor_config.trace.ir_post_fusion = True
    inductor_config.trace.log_autotuning_results = True
    # Diagnostics need a fresh dump for each process; otherwise Inductor can
    # silently reuse a disk-cached graph and leave the trace directory empty.
    if keep_caches:
        return
    if hasattr(inductor_config, "fx_graph_cache"):
        inductor_config.fx_graph_cache = False
    if hasattr(inductor_config, "autotune_local_cache"):
        inductor_config.autotune_local_cache = False
    if hasattr(inductor_config, "autotune_remote_cache"):
        inductor_config.autotune_remote_cache = False


def _make_bw_chunk(stage_batch: Any, *, device: torch.device, chunk_size: int) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_agents = int(stage_batch.num_agents)
    sample_count = int(stage_batch.num_samples)
    chunk = min(int(chunk_size), sample_count)
    local_i = _slice_dataclass(stage_batch.local_batch, 0, int(chunk) * int(num_agents))
    action_i = stage_batch.actions.to(device=device)[0:int(chunk)]
    old_i = torch.zeros((int(chunk),), dtype=torch.float32, device=device)
    # Deterministic pseudo-random advantages make the chunk non-degenerate while
    # staying independent of the critic path we are testing.
    gen = torch.Generator(device=device)
    gen.manual_seed(123456)
    adv_i = torch.randn((int(chunk),), dtype=torch.float32, device=device, generator=gen)
    danger_targets_i = torch.zeros((int(chunk), int(num_agents), 2), dtype=torch.float32, device=device)
    danger_masks_i = torch.zeros_like(danger_targets_i)
    return local_i, action_i, old_i, adv_i, danger_targets_i, danger_masks_i


def _clone_tensor_dataclass(batch: Any) -> Any:
    field_names = getattr(batch, "_tensor_fields", None)
    if field_names is None:
        from dataclasses import fields, is_dataclass

        if not is_dataclass(batch):
            raise TypeError("_clone_tensor_dataclass expects a tensor dataclass")
        field_names = tuple(field.name for field in fields(batch))
    kwargs = {}
    for field_name in field_names:
        value = getattr(batch, field_name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field_name}: {type(value)!r}")
        kwargs[str(field_name)] = value.detach().clone()
    return type(batch)(**kwargs)


def _bench_bw_compiled(
    learner: Any,
    stage_batch: Any | None,
    *,
    prebuilt_inputs: tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    sample_count_override: int | None = None,
    num_agents_override: int | None = None,
    trace_dir: Path,
    profile_path: Path,
    iters: int,
    profile_iters: int,
    keep_inductor_caches: bool = False,
) -> dict[str, Any]:
    device = learner.device
    _configure_inductor_trace(trace_dir, keep_caches=bool(keep_inductor_caches))
    if stage_batch is None and (prebuilt_inputs is None or sample_count_override is None or num_agents_override is None):
        raise ValueError("stage_batch or complete prebuilt input metadata is required")
    num_agents = int(num_agents_override if num_agents_override is not None else stage_batch.num_agents)
    sample_count = int(sample_count_override if sample_count_override is not None else stage_batch.num_samples)
    chunk_size = _strict_chunk_size(sample_count, 2048)
    inputs = prebuilt_inputs if prebuilt_inputs is not None else _make_bw_chunk(stage_batch, device=device, chunk_size=chunk_size)
    module = _BwActorLossChunkModule(
        learner.actor.bw_policy,
        num_agents=int(num_agents),
        need_entropy=False,
        entropy_coef=0.0,
        clip_ratio=float(learner.clip_ratio),
    )
    compiled = torch.compile(
        module,
        fullgraph=True,
        options={
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        },
    )
    params = _stage_optimizer_params(learner.actor, 2)

    def one_backward() -> float:
        for p in params:
            p.grad = None
        loss = compiled(*inputs)[0]
        loss.backward()
        _sync(device)
        return float(loss.detach().cpu().item())

    # Compile + warmup.
    warm_losses = [one_backward() for _ in range(2)]
    times = []
    losses = []
    for _ in range(max(int(iters), 1)):
        t0 = time.perf_counter()
        losses.append(one_backward())
        times.append(time.perf_counter() - t0)

    profile_text = ""
    if int(profile_iters) > 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
            for _ in range(int(profile_iters)):
                one_backward()
        profile_text = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(profile_text, encoding="utf-8")

    return {
        "chunk_size": int(chunk_size),
        "sample_count": int(sample_count),
        "warm_losses": warm_losses,
        "losses": losses,
        "times_sec": times,
        "mean_sec": float(sum(times) / max(len(times), 1)),
        "min_sec": float(min(times) if times else 0.0),
        "max_sec": float(max(times) if times else 0.0),
        "profile_path": str(profile_path),
    }


def _simple_mm_bench(device: torch.device, *, repeats: int = 20) -> dict[str, float]:
    a = torch.randn((6000, 128), dtype=torch.float32, device=device)
    b = torch.randn((128, 256), dtype=torch.float32, device=device)
    # Warmup.
    for _ in range(3):
        _ = a @ b
    _sync(device)
    times = []
    for _ in range(max(int(repeats), 1)):
        t0 = time.perf_counter()
        _ = a @ b
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {
        "simple_mm_mean_ms": float(1000.0 * sum(times) / max(len(times), 1)),
        "simple_mm_min_ms": float(1000.0 * min(times) if times else 0.0),
        "simple_mm_max_ms": float(1000.0 * max(times) if times else 0.0),
    }


def _tensor_brief(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach()
    xf = x.to(dtype=torch.float32)
    return {
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "is_contiguous": bool(x.is_contiguous()),
        "mean": float(xf.mean().detach().cpu().item()) if x.numel() else 0.0,
        "std": float(xf.std(unbiased=False).detach().cpu().item()) if x.numel() else 0.0,
        "min": float(xf.min().detach().cpu().item()) if x.numel() else 0.0,
        "max": float(xf.max().detach().cpu().item()) if x.numel() else 0.0,
    }


def _bw_chunk_brief(inputs: tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, Any]:
    local_i, action_i, _old_i, adv_i, _danger_targets_i, _danger_masks_i = inputs
    out: dict[str, Any] = {
        "action": _tensor_brief(action_i),
        "adv": _tensor_brief(adv_i),
        "local_type": type(local_i).__name__,
        "local_fields": {},
    }
    names = getattr(local_i, "_tensor_fields", None)
    if names is None:
        from dataclasses import fields, is_dataclass

        names = tuple(field.name for field in fields(local_i)) if is_dataclass(local_i) else ()
    for name in names:
        value = getattr(local_i, str(name))
        if torch.is_tensor(value):
            brief = _tensor_brief(value)
            if value.dtype == torch.bool:
                brief["true_frac"] = float(value.to(dtype=torch.float32).mean().detach().cpu().item()) if value.numel() else 0.0
            out["local_fields"][str(name)] = brief
    return out


def _save_bw_inputs(path: Path, inputs: tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    local_i, action_i, old_i, adv_i, danger_targets_i, danger_masks_i = inputs
    names = getattr(local_i, "_tensor_fields", None)
    if names is None:
        from dataclasses import fields, is_dataclass

        names = tuple(field.name for field in fields(local_i)) if is_dataclass(local_i) else ()
    payload = {
        "local_type": type(local_i).__name__,
        "local_fields": {str(name): getattr(local_i, str(name)).detach().cpu() for name in names},
        "action": action_i.detach().cpu(),
        "old": old_i.detach().cpu(),
        "adv": adv_i.detach().cpu(),
        "danger_targets": danger_targets_i.detach().cpu(),
        "danger_masks": danger_masks_i.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_bw_inputs(path: Path, *, local_type: type, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    local_fields = {str(k): v.to(device=device) for k, v in dict(payload["local_fields"]).items()}
    local_i = local_type(**local_fields)
    return (
        local_i,
        payload["action"].to(device=device),
        payload["old"].to(device=device),
        payload["adv"].to(device=device),
        payload["danger_targets"].to(device=device),
        payload["danger_masks"].to(device=device),
    )


def _summarize_trace(trace_dir: Path) -> dict[str, Any]:
    files = [p for p in trace_dir.rglob("*") if p.is_file()]
    suffix_counts: dict[str, int] = {}
    total_bytes = 0
    for p in files:
        suffix_counts[p.suffix or "<none>"] = suffix_counts.get(p.suffix or "<none>", 0) + 1
        total_bytes += int(p.stat().st_size)
    largest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:20]
    return {
        "trace_dir": str(trace_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "suffix_counts": suffix_counts,
        "largest_files": [
            {"path": str(p.relative_to(trace_dir)), "bytes": int(p.stat().st_size)}
            for p in largest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether compiled critic affects compiled BW actor loss.")
    parser.add_argument("--config", default="configs/tmp/structured_sat_mcgae_3uav_20gu_t250_positive_relcritic.yaml")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--mode", choices=["clean_bw", "after_critic"], required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=94001)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile_iters", type=int, default=1)
    parser.add_argument("--critic_stage", type=int, default=1)
    parser.add_argument("--critic_epochs", type=int, default=1)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--clone_bw_chunk_and_drop_views", action="store_true")
    parser.add_argument("--close_group_before_bench", action="store_true")
    parser.add_argument("--bench_sample_count_override", type=int, default=0)
    parser.add_argument("--save_chunk_path", default="")
    parser.add_argument("--load_chunk_path", default="")
    parser.add_argument("--keep_inductor_caches", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    _force_joint_config(cfg, reward_mode=str(getattr(cfg, "reward_mode", "positive_weighted_workload_level")))
    cfg.critic_compile_enabled = True
    cfg.critic_compile_fullgraph = True
    learner = _make_joint_learner(cfg, device=device)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), mode="train")
    group_closed = False

    try:
        collect_t0 = time.perf_counter()
        views, _returns, stage_targets, reward_stats = _collect_joint_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            seed=int(args.seed),
        )
        _sync(device)
        collect_sec = time.perf_counter() - collect_t0
        bw_stage_batch = views.training_view.stage_batches[2]
        bw_inputs = None
        bw_sample_count = int(bw_stage_batch.num_samples)
        if int(args.bench_sample_count_override) > 0:
            bw_sample_count = int(args.bench_sample_count_override)
        bw_num_agents = int(bw_stage_batch.num_agents)

        critic_stats = {}
        if args.mode == "after_critic":
            opt = torch.optim.Adam(learner.critic.parameters(), lr=1.0e-3)
            t0 = time.perf_counter()
            critic_stats, _values, _trace = _train_stage_critic_on_stage(
                learner,
                stage_id=int(args.critic_stage),
                stage_batch=views.training_view.stage_batches[int(args.critic_stage)],
                target=stage_targets[int(args.critic_stage)],
                optimizer=opt,
                lr=1.0e-3,
                epochs=int(args.critic_epochs),
                minibatches=int(args.critic_minibatches),
                update_microbatch_size=0,
                diagnose_timing=False,
            )
            _sync(device)
            critic_stats["critic_wall_sec"] = time.perf_counter() - t0
            del opt
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        if bool(args.close_group_before_bench):
            close_structured_env_group(group)
            group_closed = True
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

        if bool(args.clone_bw_chunk_and_drop_views):
            chunk_size = _strict_chunk_size(bw_sample_count, 2048)
            raw_inputs = _make_bw_chunk(bw_stage_batch, device=device, chunk_size=chunk_size)
            bw_inputs = (
                _clone_tensor_dataclass(raw_inputs[0]),
                raw_inputs[1].detach().clone(),
                raw_inputs[2].detach().clone(),
                raw_inputs[3].detach().clone(),
                raw_inputs[4].detach().clone(),
                raw_inputs[5].detach().clone(),
            )
            del raw_inputs
            if str(args.save_chunk_path):
                _save_bw_inputs(Path(args.save_chunk_path), bw_inputs)
            if str(args.load_chunk_path):
                bw_inputs = _load_bw_inputs(Path(args.load_chunk_path), local_type=type(bw_inputs[0]), device=device)
            del bw_stage_batch
            del views
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        elif str(args.save_chunk_path) or str(args.load_chunk_path):
            raise ValueError("--save_chunk_path/--load_chunk_path require --clone_bw_chunk_and_drop_views")

        trace_dir = run_dir / f"inductor_trace_{args.mode}"
        profile_path = run_dir / f"bw_profile_{args.mode}.txt"
        bw_stats = _bench_bw_compiled(
            learner,
            None if bw_inputs is not None else bw_stage_batch,
            prebuilt_inputs=bw_inputs,
            sample_count_override=bw_sample_count if bw_inputs is not None else None,
            num_agents_override=bw_num_agents if bw_inputs is not None else None,
            trace_dir=trace_dir,
            profile_path=profile_path,
            iters=int(args.iters),
            profile_iters=int(args.profile_iters),
            keep_inductor_caches=bool(args.keep_inductor_caches),
        )
        _sync(device)
        mm_stats = _simple_mm_bench(device)
        if bw_inputs is not None:
            chunk_brief = _bw_chunk_brief(bw_inputs)
        else:
            chunk_brief = _bw_chunk_brief(_make_bw_chunk(bw_stage_batch, device=device, chunk_size=_strict_chunk_size(bw_sample_count, 2048)))

        summary = {
            "mode": args.mode,
            "device": str(device),
            "collect_sec": collect_sec,
            "reward_stats": reward_stats,
            "critic_stats": critic_stats,
            "bw_stats": bw_stats,
            "mm_stats": mm_stats,
            "chunk_brief": chunk_brief,
            "trace_summary": _summarize_trace(trace_dir),
        }
        out_path = run_dir / f"summary_{args.mode}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if not group_closed:
            close_structured_env_group(group)


if __name__ == "__main__":
    main()
