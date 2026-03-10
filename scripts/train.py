from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train
from sagin_marl.utils.seeding import set_seed


def _resolve_log_dir(log_dir: str, run_dir: str | None, run_id: str | None) -> str:
    if run_dir:
        return run_dir
    if run_id:
        if run_id == "auto":
            run_id = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, run_id)
    return log_dir


def _save_config(log_dir: str, cfg, config_path: str) -> None:
    try:
        data = asdict(cfg)
        data["_config_source"] = config_path
        out_path = os.path.join(log_dir, "config.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        if os.path.isfile(config_path):
            shutil.copy2(config_path, os.path.join(log_dir, "config_source.yaml"))
    except Exception as exc:
        print(f"Warning: failed to save config in {log_dir}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--log_dir", type=str, default="runs/phase1")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Full run directory. Overrides --log_dir/--run_id.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Subdirectory name under --log_dir. Use 'auto' for timestamp.",
    )
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Save actor_uXXXX.pt/critic_uXXXX.pt every N updates. <=0 disables periodic saves.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environment instances for rollout collection.",
    )
    parser.add_argument(
        "--vec_backend",
        type=str,
        default="subproc",
        choices=["sync", "subproc"],
        help="Vectorized env backend when --num_envs > 1.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="Set torch intra-op threads (<=0 keeps default).",
    )
    parser.add_argument("--init_actor", type=str, default=None, help="Init actor checkpoint path.")
    parser.add_argument("--init_critic", type=str, default=None, help="Init critic checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")

    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)

    set_seed(cfg.seed)
    if args.num_envs > 1:
        from sagin_marl.env.vec_env import make_vec_env

        env = make_vec_env(cfg, num_envs=args.num_envs, backend=args.vec_backend)
    else:
        env = SaginParallelEnv(cfg)
    log_dir = _resolve_log_dir(args.log_dir, args.run_dir, args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    _save_config(log_dir, cfg, os.path.abspath(args.config))
    print(f"Run dir: {log_dir}")
    print(f"Rollout envs: {args.num_envs} ({args.vec_backend if args.num_envs > 1 else 'single'})")
    try:
        train(
            env,
            cfg,
            log_dir,
            total_updates=args.updates,
            save_interval_updates=args.save_interval,
            init_actor_path=args.init_actor,
            init_critic_path=args.init_critic,
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
