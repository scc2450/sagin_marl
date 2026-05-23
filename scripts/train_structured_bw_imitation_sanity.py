from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.baselines import queue_aware_bw_policy
from sagin_marl.rl.structured_eval import (
    evaluate_structured_actor_exec_sources,
    evaluate_structured_fixed_policy,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _refresh_stage_obs_cache
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cpu_dataclass(batch: Any) -> Any:
    if not is_dataclass(batch):
        raise TypeError("_cpu_dataclass expects a dataclass instance")
    kwargs = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported field type for {field.name}: {type(value)!r}")
        kwargs[field.name] = value.detach().cpu()
    return type(batch)(**kwargs)


def _valid_bw_mask(local_state) -> torch.Tensor:
    return (local_state.user_mask > 0.5) & (local_state.bw_valid_mask > 0.5)


def _masked_simplex_kl(
    target: torch.Tensor,
    mode: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    target_valid = torch.where(valid_mask, target.clamp_min(eps), torch.zeros_like(target))
    mode_valid = torch.where(valid_mask, mode.clamp_min(eps), torch.zeros_like(mode))
    target_valid = target_valid / target_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    mode_valid = mode_valid / mode_valid.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target_valid * (torch.log(target_valid.clamp_min(eps)) - torch.log(mode_valid.clamp_min(eps)))
    return (kl * valid_mask.to(dtype=kl.dtype)).sum(dim=-1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_teacher_dataset(
    cfg,
    *,
    episodes: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    samples: list[dict[str, Any]] = []
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    sat_zero = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    try:
        for ep in range(int(episodes)):
            env.reset(seed=int(seed_base) + ep)
            done = False
            while not done:
                driver.begin_step()
                z1 = driver.run_accel_stage(accel_zero)
                _refresh_stage_obs_cache(driver)
                obs_list = [env._get_obs(idx) for idx in range(len(env.agents))]
                z2 = driver.run_sat_stage(sat_zero)
                bw_states = driver.build_bw_valid_context(z2)
                teacher_action = np.asarray(queue_aware_bw_policy(obs_list, cfg), dtype=np.float32)
                for u, local_state in enumerate(bw_states):
                    samples.append(
                        {
                            "local_state": _cpu_dataclass(local_state),
                            "target_action": np.asarray(teacher_action[u : u + 1], dtype=np.float32),
                        }
                    )
                step_result = driver.execute_stage_bw_and_step(teacher_action)
                done = bool(list(step_result.terminations.values())[0] or list(step_result.truncations.values())[0])
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return samples


def _make_batch(entries: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    local_states_cpu = [entry["local_state"] for entry in entries]
    local_state = _collate_dataclass(local_states_cpu, device)
    target_action = torch.cat(
        [torch.as_tensor(entry["target_action"], dtype=torch.float32) for entry in entries],
        dim=0,
    ).to(device)
    return {
        "local_state": local_state,
        "target_action": target_action,
    }


def _evaluate_imitation_split(
    actor,
    entries: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if not entries:
        return {
            "sample_count": 0.0,
            "kl_mean": 0.0,
            "l1_mean": 0.0,
            "mass_mean": 0.0,
        }
    actor.eval()
    kl_values: list[float] = []
    l1_values: list[float] = []
    mass_values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(entries), int(batch_size)):
            batch_entries = entries[start : start + int(batch_size)]
            batch = _make_batch(batch_entries, device)
            local_state = batch["local_state"]
            target_action = batch["target_action"]
            valid_mask = _valid_bw_mask(local_state)
            student_action = actor.bw_policy.deterministic_action(
                local_state,
                readout="latent_mean_pushforward",
            )
            kl = _masked_simplex_kl(target_action, student_action, valid_mask)
            l1 = (
                (student_action - target_action).abs() * valid_mask.to(dtype=student_action.dtype)
            ).sum(dim=-1)
            mass = (student_action * valid_mask.to(dtype=student_action.dtype)).sum(dim=-1)
            kl_values.extend(kl.detach().cpu().tolist())
            l1_values.extend(l1.detach().cpu().tolist())
            mass_values.extend(mass.detach().cpu().tolist())
    return {
        "sample_count": float(len(entries)),
        "kl_mean": float(np.mean(np.asarray(kl_values, dtype=np.float64))),
        "l1_mean": float(np.mean(np.asarray(l1_values, dtype=np.float64))),
        "mass_mean": float(np.mean(np.asarray(mass_values, dtype=np.float64))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--train_episodes", type=int, default=24)
    parser.add_argument("--val_episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_num_envs", type=int, default=4)
    parser.add_argument("--eval_vec_backend", choices=["sync", "subproc"], default="sync")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init_actor", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    _set_all_seeds(int(args.seed))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )
    actor = bundle.actor.to(device)
    actor.eval()
    if args.init_actor:
        load_checkpoint_forgiving(actor, str(args.init_actor), map_location=device, strict=True)

    train_samples = _collect_teacher_dataset(cfg, episodes=int(args.train_episodes), seed_base=int(args.seed) * 100)
    val_samples = _collect_teacher_dataset(cfg, episodes=int(args.val_episodes), seed_base=int(args.seed) * 1000)
    optimizer = torch.optim.Adam(actor.bw_policy.parameters(), lr=float(args.lr))

    history: list[dict[str, Any]] = []
    best_val_kl = float("inf")
    best_actor_path = run_dir / "actor_best.pt"
    final_actor_path = run_dir / "actor_final.pt"

    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        perm = np.random.permutation(len(train_samples)).tolist()
        train_losses: list[float] = []
        train_l1s: list[float] = []
        for start in range(0, len(perm), int(args.batch_size)):
            batch_indices = perm[start : start + int(args.batch_size)]
            batch_entries = [train_samples[int(i)] for i in batch_indices]
            batch = _make_batch(batch_entries, device)
            local_state = batch["local_state"]
            target_action = batch["target_action"]
            valid_mask = _valid_bw_mask(local_state)
            student_action = actor.bw_policy.deterministic_action(
                local_state,
                readout="latent_mean_pushforward",
            )
            kl = _masked_simplex_kl(target_action, student_action, valid_mask).mean()
            l1 = (
                ((student_action - target_action).abs() * valid_mask.to(dtype=student_action.dtype)).sum(dim=-1).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            kl.backward()
            torch.nn.utils.clip_grad_norm_(actor.bw_policy.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(kl.detach().cpu().item()))
            train_l1s.append(float(l1.detach().cpu().item()))

        train_summary = _evaluate_imitation_split(
            actor,
            train_samples,
            device=device,
            batch_size=int(args.batch_size),
        )
        val_summary = _evaluate_imitation_split(
            actor,
            val_samples,
            device=device,
            batch_size=int(args.batch_size),
        )
        row = {
            "epoch": int(epoch),
            "train_loss_batch_mean": float(np.mean(np.asarray(train_losses, dtype=np.float64))) if train_losses else 0.0,
            "train_l1_batch_mean": float(np.mean(np.asarray(train_l1s, dtype=np.float64))) if train_l1s else 0.0,
            "train_kl_eval": float(train_summary["kl_mean"]),
            "train_l1_eval": float(train_summary["l1_mean"]),
            "val_kl_eval": float(val_summary["kl_mean"]),
            "val_l1_eval": float(val_summary["l1_mean"]),
        }
        history.append(row)
        if float(val_summary["kl_mean"]) < best_val_kl:
            best_val_kl = float(val_summary["kl_mean"])
            torch.save(actor.state_dict(), best_actor_path)
        print(
            f"Epoch {epoch:03d} | train_kl={row['train_kl_eval']:.5f} train_l1={row['train_l1_eval']:.5f} "
            f"| val_kl={row['val_kl_eval']:.5f} val_l1={row['val_l1_eval']:.5f}"
        )

    torch.save(actor.state_dict(), final_actor_path)
    _write_csv(run_dir / "metrics.csv", history)

    if best_actor_path.exists():
        load_checkpoint_forgiving(actor, str(best_actor_path), map_location=device, strict=True)
    actor.eval()
    learned_det_summary, _ = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.eval_episodes),
        episode_seed_base=int(args.seed) * 10000,
        deterministic=True,
        num_envs=int(args.eval_num_envs),
        vec_backend=str(args.eval_vec_backend),
    )
    learned_stoch_summary, _ = evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=int(args.eval_episodes),
        episode_seed_base=int(args.seed) * 10000,
        deterministic=False,
        num_envs=int(args.eval_num_envs),
        vec_backend=str(args.eval_vec_backend),
    )
    fixed_summary = evaluate_structured_fixed_policy(
        cfg,
        baseline_policy="queue_aware_bw",
        episodes=int(args.eval_episodes),
        episode_seed_base=int(args.seed) * 10000,
    )

    summary = {
        "config": str(args.config),
        "run_dir": str(run_dir),
        "train_samples": int(len(train_samples)),
        "val_samples": int(len(val_samples)),
        "best_val_kl": float(best_val_kl),
        "final_train": history[-1] if history else {},
        "learned_deterministic": learned_det_summary,
        "learned_stochastic": learned_stoch_summary,
        "fixed_queue_aware_bw": fixed_summary,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
