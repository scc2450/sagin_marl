from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.stage_mcgae import (
    make_learner as _make_learner,
    stage_optimizer_params as _stage_optimizer_params,
)
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache

from scripts.train_joint_mcgae import _actor_only_stage_batch, _collect_joint_rollout
from scripts.train_stage_mcgae import (
    _cuda_mem,
    _enable_strict_compile_global,
    _normalize_stage_advantage,
    _stage_actor_update_full_stage,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _clone_adv(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().to(dtype=torch.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and isolate intermittent BW actor update slowdowns.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--actor_epochs", type=int, default=5)
    parser.add_argument("--actor_minibatches", type=int, default=1)
    parser.add_argument("--actor_lr", type=float, default=3.0e-4)
    parser.add_argument("--adv_mode", choices=("random", "mc_return"), default="random")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--empty_cache_between_repeats", action="store_true")
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.reward_mode = "positive_weighted_workload_level"
    cfg.train_accel = True
    cfg.train_sat = True
    cfg.train_bw = True
    cfg.exec_accel_source = "policy"
    cfg.exec_sat_source = "policy"
    cfg.exec_bw_source = "policy"
    cfg.actor_advantage_normalize_enabled = True
    cfg.stagewise_advantage_norm_enabled = True
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False

    report_torch_compile_cache(context="diagnose_bw_actor_update_repro", device=device, cfg=cfg)
    if device.type == "cuda":
        _enable_strict_compile_global()

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=2)
    learner.bind_native_runtime_contract(None)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    rows: list[dict[str, float]] = []
    try:
        learner.bind_native_runtime_contract(group)
        collect_t0 = time.perf_counter()
        views, _all_returns, stage_targets, reward_stats = _collect_joint_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            seed=int(args.seed),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        collect_sec = time.perf_counter() - collect_t0
        bw_stage_batch = _actor_only_stage_batch(views.training_view.stage_batches[2])
        sample_count = int(bw_stage_batch.num_samples)
        if str(args.adv_mode) == "mc_return":
            adv = _normalize_stage_advantage(
                stage_targets[2].detach().to(device=device, dtype=torch.float32),
                enabled=True,
            )
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.seed) + 9191)
            adv = torch.randn(sample_count, dtype=torch.float32, device=device, generator=gen)
            adv = _normalize_stage_advantage(adv, enabled=True)
        adv = _clone_adv(adv)

        # Release rollout/history views; the target of this script is the
        # compact actor batch path, not native storage-view path.
        del views
        del stage_targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        initial_actor_state = deepcopy(learner.actor.state_dict())
        print(
            f"BW actor repro | samples={sample_count} repeats={int(args.repeats)} "
            f"collect_sec={collect_sec:.3f} adv={args.adv_mode}",
            flush=True,
        )

        for rep in range(max(int(args.repeats), 1)):
            learner.actor.load_state_dict(initial_actor_state)
            params = _stage_optimizer_params(learner.actor, 2)
            optimizer = torch.optim.Adam(params, lr=float(args.actor_lr))
            if bool(args.empty_cache_between_repeats) and device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            rep_t0 = time.perf_counter()
            stats = _stage_actor_update_full_stage(
                learner,
                stage_id=2,
                stage_batch=bw_stage_batch,
                stage_advantages=adv,
                optimizer=optimizer,
                epochs=int(args.actor_epochs),
                minibatches=int(args.actor_minibatches),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "repeat": float(rep + 1),
                "collect_sec": float(collect_sec),
                "repeat_total_sec": float(time.perf_counter() - rep_t0),
                **{f"reward_{k}": float(v) for k, v in reward_stats.items()},
                **stats,
                **_cuda_mem("repeat_end", device),
            }
            rows.append(row)
            _write_csv(run_dir / "metrics.csv", rows)
            print(
                f"repeat {rep + 1}/{int(args.repeats)} "
                f"old={row['actor_old_logprob_sec']:.3f}s "
                f"loop={row['actor_update_loop_sec']:.3f}s "
                f"total={row['repeat_total_sec']:.3f}s "
                f"kl={row.get('approx_kl_bw', 0.0):.4f}",
                flush=True,
            )
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
