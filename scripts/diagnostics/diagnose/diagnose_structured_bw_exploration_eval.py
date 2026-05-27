from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import evaluate_structured_actor, evaluate_structured_actor_exec_sources
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _clone_with_kappa_scale(actor, scale: float):
    cloned = copy.deepcopy(actor)
    bw_policy = getattr(cloned, "bw_policy", None)
    if bw_policy is None or not hasattr(bw_policy, "kappa_min") or not hasattr(bw_policy, "kappa_max"):
        raise RuntimeError("Actor does not expose BW kappa_min/kappa_max for exploration scaling.")
    bw_policy.kappa_min = float(bw_policy.kappa_min) * float(scale)
    bw_policy.kappa_max = float(bw_policy.kappa_max) * float(scale)
    return cloned


def _evaluate(
    cfg,
    actor,
    *,
    episodes: int,
    episode_seed_base: int | None,
    deterministic: bool,
    num_envs: int,
    vec_backend: str,
    respect_exec_sources: bool,
):
    device = next(actor.parameters()).device
    if respect_exec_sources:
        summary, _ = evaluate_structured_actor_exec_sources(
            cfg,
            actor,
            device=device,
            episodes=int(episodes),
            episode_seed_base=episode_seed_base,
            deterministic=bool(deterministic),
            num_envs=int(num_envs),
            vec_backend=str(vec_backend),
        )
    else:
        summary, _ = evaluate_structured_actor(
            cfg,
            actor,
            device=device,
            episodes=int(episodes),
            episode_seed_base=episode_seed_base,
            deterministic=bool(deterministic),
            num_envs=int(num_envs),
            vec_backend=str(vec_backend),
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--episode_seed_base", type=int, default=91000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--respect_exec_sources", action="store_true")
    parser.add_argument("--kappa_scales", type=float, nargs="*", default=[1.0, 0.25, 0.125])
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = args.config or str(run_dir / "config_source.yaml")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "actor_final.pt"
    cfg = load_config(config_path)
    bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))
    actor = bundle.actor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint_forgiving(actor, str(checkpoint_path), map_location=device, strict=True)
    actor = actor.to(device).eval()
    base_bw = getattr(actor, "bw_policy", None)
    base_kappa_min = None if base_bw is None or not hasattr(base_bw, "kappa_min") else float(base_bw.kappa_min)
    base_kappa_max = None if base_bw is None or not hasattr(base_bw, "kappa_max") else float(base_bw.kappa_max)

    results: list[dict[str, object]] = []
    det_summary = _evaluate(
        cfg,
        actor,
        episodes=int(args.episodes),
        episode_seed_base=int(args.episode_seed_base),
        deterministic=True,
        num_envs=int(args.num_envs),
        vec_backend=str(args.vec_backend),
        respect_exec_sources=bool(args.respect_exec_sources),
    )
    results.append(
        {
            "mode": "deterministic",
            "kappa_scale": 1.0,
            "kappa_min": base_kappa_min,
            "kappa_max": base_kappa_max,
            **{k: float(v) for k, v in det_summary.items()},
        }
    )

    for scale in args.kappa_scales:
        scale = float(scale)
        eval_actor = actor if abs(scale - 1.0) <= 1.0e-12 else _clone_with_kappa_scale(actor, scale).to(device).eval()
        stoch_summary = _evaluate(
            cfg,
            eval_actor,
            episodes=int(args.episodes),
            episode_seed_base=int(args.episode_seed_base),
            deterministic=False,
            num_envs=int(args.num_envs),
            vec_backend=str(args.vec_backend),
            respect_exec_sources=bool(args.respect_exec_sources),
        )
        bw_policy = getattr(eval_actor, "bw_policy", None)
        results.append(
            {
                "mode": "stochastic",
                "kappa_scale": scale,
                "kappa_min": None if bw_policy is None or not hasattr(bw_policy, "kappa_min") else float(bw_policy.kappa_min),
                "kappa_max": None if bw_policy is None or not hasattr(bw_policy, "kappa_max") else float(bw_policy.kappa_max),
                **{k: float(v) for k, v in stoch_summary.items()},
            }
        )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "respect_exec_sources": bool(args.respect_exec_sources),
        "episodes": int(args.episodes),
        "episode_seed_base": int(args.episode_seed_base),
        "num_envs": int(args.num_envs),
        "vec_backend": str(args.vec_backend),
        "results": results,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
