from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.env.vec_env import SyncVecSaginEnv, make_vec_env
from sagin_marl.rl.mappo import train
from sagin_marl.utils.seeding import set_seed


def _build_env(cfg, num_envs: int, vec_backend: str):
    if num_envs > 1:
        return make_vec_env(cfg, num_envs=num_envs, backend=vec_backend)
    return SyncVecSaginEnv(cfg, 1)


def _disable_benchmark_overheads(cfg) -> None:
    cfg.checkpoint_eval_enabled = False
    cfg.checkpoint_eval_interval_updates = 0
    cfg.checkpoint_eval_start_update = 0
    cfg.checkpoint_eval_episodes = 0
    cfg.checkpoint_eval_early_stop_enabled = False
    cfg.checkpoint_eval_reward_early_stop_enabled = False
    cfg.early_stop_enabled = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--disable_checkpoint_eval", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--cprofile_out", type=str, default=None)
    parser.add_argument("--cprofile_topn", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.disable_checkpoint_eval:
        _disable_benchmark_overheads(cfg)

    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)

    set_seed(cfg.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    env = _build_env(cfg, args.num_envs, args.vec_backend)
    try:
        if args.cprofile_out:
            profiler = cProfile.Profile()
            profiler.enable()
            train(env, cfg, args.run_dir, total_updates=args.updates, save_interval_updates=0)
            profiler.disable()
            profiler.dump_stats(args.cprofile_out)
            stats_path = Path(args.cprofile_out)
            text_path = stats_path.with_suffix(stats_path.suffix + ".txt")
            with open(text_path, "w", encoding="utf-8") as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
                stats.print_stats(args.cprofile_topn)
        else:
            train(env, cfg, args.run_dir, total_updates=args.updates, save_interval_updates=0)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
