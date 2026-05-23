from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_mappo import _index_dataclass
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache

from scripts.audit_stage_critic_only_fit import _make_learner
from scripts.train_joint_mcgae import _collect_joint_rollout
from scripts.train_stage_mcgae import _enable_strict_compile_global


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one joint MC-GAE critic minibatch forward/backward.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--stage", choices=("accel", "sat", "bw"), default="sat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--row_limit", type=int, default=40)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_id = {"accel": 0, "sat": 1, "bw": 2}[str(args.stage)]
    cfg = load_config(args.config)
    cfg.reward_mode = "positive_weighted_workload_level"
    cfg.exec_accel_source = "policy"
    cfg.exec_sat_source = "policy"
    cfg.exec_bw_source = "policy"
    cfg.train_accel = True
    cfg.train_sat = True
    cfg.train_bw = True
    cfg.checkpoint_eval_enabled = False
    cfg.train_trace_enabled = False
    report_torch_compile_cache(context=f"profile_joint_critic_step:{args.stage}", device=device, cfg=cfg)
    if device.type == "cuda":
        _enable_strict_compile_global()

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        t0 = time.perf_counter()
        views, _all_returns, stage_targets, _reward_stats = _collect_joint_rollout(
            learner,
            group,
            rollout_env_steps=int(args.rollout_env_steps),
            device=device,
            seed=int(args.seed),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"collect_sec={time.perf_counter() - t0:.3f}", flush=True)

        stage_batch = views.training_view.stage_batches[int(stage_id)]
        n = int(stage_batch.num_samples)
        b = min(max(int(args.batch_size), 1), n)
        idx = torch.arange(0, b, dtype=torch.long, device=device)
        world = _index_dataclass(stage_batch.world_batch, idx)
        target = stage_targets[int(stage_id)].detach().to(device=device, dtype=torch.float32).reshape(-1)[:b]
        params = [p for p in learner.critic.parameters() if p.requires_grad]
        for p in params:
            p.grad = None

        # Warm compiled graph once outside the profiler.
        pred = learner._stage_value_eval_from_batch(int(stage_id), world)
        loss = F.mse_loss(pred, target)
        loss.backward()
        for p in params:
            p.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            pred = learner._stage_value_eval_from_batch(int(stage_id), world)
            loss = F.mse_loss(pred, target)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
            row_limit=max(int(args.row_limit), 1),
        )
        out = run_dir / f"critic_{args.stage}_profile.txt"
        out.write_text(table, encoding="utf-8")
        print(table, flush=True)
        print(f"profile_path={out}", flush=True)
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
