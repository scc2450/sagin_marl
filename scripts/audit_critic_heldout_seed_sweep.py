from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_driver_group
from scripts.audit_stage_critic_only_fit import (
    _collect_stage_bank,
    _eval_critic,
    _make_learner,
    _train_critic_only,
)
from scripts.audit_stage_ppo_credit_alignment import STAGE_ID, _force_single_stage_config, _set_seed


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_ID), default="sat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--target", choices=["mc", "train_gae"], default="mc")
    parser.add_argument("--train_seed_base", type=int, default=55210)
    parser.add_argument("--train_rollouts", type=int, default=8)
    parser.add_argument("--heldout_seed_bases", default="245210,9045210,19045210,29045210")
    parser.add_argument("--heldout_rollouts", type=int, default=2)
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_minibatches", type=int, default=8)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_message_layers", type=int, default=None)
    parser.add_argument("--init_seed", type=int, default=45210)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    _set_seed(int(args.init_seed))
    cfg = load_config(args.config)
    stage_id = STAGE_ID[str(args.stage)]
    _force_single_stage_config(cfg, stage_id=stage_id, reward_mode=args.reward_mode)
    if args.critic_lr is not None:
        cfg.critic_lr = float(args.critic_lr)
    if args.critic_message_layers is not None:
        cfg.critic_message_layers = int(args.critic_message_layers)

    learner, _actor, _critic = _make_learner(cfg, device=device, stage_id=stage_id)
    group = make_structured_driver_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        learner.bind_native_runtime_contract(group)
        train_world, train_target, train_summaries = _collect_stage_bank(
            learner,
            group,
            stage_id=stage_id,
            rollout_env_steps=int(args.rollout_env_steps),
            rollouts=int(args.train_rollouts),
            seed_base=int(args.train_seed_base),
            device=device,
            target=str(args.target),
        )
        history = _train_critic_only(
            learner,
            stage_id=stage_id,
            train_world=train_world,
            train_target=train_target,
            heldout_world=train_world,
            heldout_target=train_target,
            epochs=int(args.critic_epochs),
            minibatches=int(args.critic_minibatches),
            lr=float(cfg.critic_lr),
            max_grad_norm=float(cfg.max_grad_norm),
            eval_every=max(1, int(args.critic_epochs) // 3),
        )
        _pred_train, train_eval = _eval_critic(
            learner,
            stage_id=stage_id,
            world_bank=train_world,
            target=train_target,
            batch_size=max(1024, int(train_target.numel()) // max(int(args.critic_minibatches), 1)),
        )
        del _pred_train

        heldout_results = []
        for seed_base in _parse_int_list(str(args.heldout_seed_bases)):
            heldout_world, heldout_target, heldout_summaries = _collect_stage_bank(
                learner,
                group,
                stage_id=stage_id,
                rollout_env_steps=int(args.rollout_env_steps),
                rollouts=int(args.heldout_rollouts),
                seed_base=int(seed_base),
                device=device,
                target=str(args.target),
            )
            _pred, stats = _eval_critic(
                learner,
                stage_id=stage_id,
                world_bank=heldout_world,
                target=heldout_target,
                batch_size=max(1024, int(heldout_target.numel()) // max(int(args.critic_minibatches), 1)),
            )
            del _pred
            heldout_results.append(
                {
                    "seed_base": int(seed_base),
                    "rollouts": int(args.heldout_rollouts),
                    "summaries": heldout_summaries,
                    "stats": stats,
                }
            )

        result = {
            "config": {
                "stage": str(args.stage),
                "stage_id": int(stage_id),
                "config": str(args.config),
                "reward_mode": str(args.reward_mode or cfg.reward_mode),
                "target": str(args.target),
                "num_envs": int(args.num_envs),
                "rollout_env_steps": int(args.rollout_env_steps),
                "train_seed_base": int(args.train_seed_base),
                "train_rollouts": int(args.train_rollouts),
                "heldout_seed_bases": _parse_int_list(str(args.heldout_seed_bases)),
                "heldout_rollouts": int(args.heldout_rollouts),
                "critic_epochs": int(args.critic_epochs),
                "critic_lr": float(cfg.critic_lr),
                "critic_message_layers": int(getattr(cfg, "critic_message_layers", -1)),
                "init_seed": int(args.init_seed),
            },
            "train_summaries": train_summaries,
            "train_history": history,
            "train_eval": train_eval,
            "heldout_results": heldout_results,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["config"], ensure_ascii=False, indent=2))
        print(f"train EV={train_eval['ev']:.4f} corr={train_eval['corr']:.4f}")
        for row in heldout_results:
            stats = row["stats"]
            print(
                f"heldout seed_base={row['seed_base']} EV={stats['ev']:.4f} "
                f"corr={stats['corr']:.4f} mean={stats['target']['mean']:.4f} std={stats['target']['std']:.4f}"
            )
        print(f"wrote {out_path}")
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
