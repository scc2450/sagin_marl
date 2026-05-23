from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_float_list(raw: list[str]) -> list[float]:
    values: list[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            value = float(part)
            if value <= 0.0:
                raise ValueError(f"delta values must be positive, got {value}.")
            if value not in values:
                values.append(value)
    return values


def _summary_tensor(values: torch.Tensor) -> dict[str, float]:
    arr = values.detach().float().reshape(-1).cpu()
    if int(arr.numel()) <= 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(arr.numel()),
        "mean": float(arr.mean().item()),
        "std": float(arr.std(unbiased=False).item()) if int(arr.numel()) > 1 else 0.0,
        "min": float(arr.min().item()),
        "p05": float(torch.quantile(arr, 0.05).item()),
        "p25": float(torch.quantile(arr, 0.25).item()),
        "p50": float(torch.quantile(arr, 0.50).item()),
        "p75": float(torch.quantile(arr, 0.75).item()),
        "p95": float(torch.quantile(arr, 0.95).item()),
        "max": float(arr.max().item()),
    }


def _masked_probs(raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_f = valid_mask.to(dtype=torch.float32)
    masked = torch.where(valid_mask, raw.to(dtype=torch.float32).clamp_min(0.0), torch.zeros_like(raw, dtype=torch.float32))
    denom = masked.sum(dim=-1, keepdim=True)
    fallback = valid_f / valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(denom > 1.0e-8, masked / denom.clamp_min(1.0e-8), fallback)


def _select_flat_rows(
    *,
    valid_mask: torch.Tensor,
    history_rows: torch.Tensor,
    num_agents: int,
    num_envs: int,
    rollout_env_steps: int,
    horizon: int,
    sample_budget: int,
    seed: int,
) -> torch.Tensor:
    row_count = int(valid_mask.shape[0])
    if row_count <= 0:
        return torch.zeros((0,), dtype=torch.long, device=valid_mask.device)
    sample_rows = torch.arange(row_count, dtype=torch.long, device=valid_mask.device) // max(int(num_agents), 1)
    source_steps = history_rows.index_select(0, sample_rows) // max(int(num_envs), 1)
    enough_future = source_steps <= max(int(rollout_env_steps) - int(horizon), 0)
    eligible = torch.nonzero((valid_mask.sum(dim=1) > 1) & enough_future, as_tuple=False).reshape(-1)
    if int(eligible.numel()) <= 0:
        return eligible
    if int(sample_budget) <= 0 or int(eligible.numel()) <= int(sample_budget):
        return eligible
    generator = torch.Generator(device=eligible.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(eligible.numel()), generator=generator, device=eligible.device)
    return eligible.index_select(0, perm[: int(sample_budget)])


def _row_best_summary(
    *,
    selected_rows: torch.Tensor,
    candidate_rows: torch.Tensor,
    candidate_diffs: torch.Tensor,
    eps: float,
) -> dict[str, Any]:
    if int(selected_rows.numel()) <= 0:
        return {"row_count": 0.0, "row_improved_frac": 0.0, "best_diff": _summary_tensor(candidate_diffs)}
    best = torch.full((int(selected_rows.numel()),), -float("inf"), dtype=torch.float32, device=candidate_diffs.device)
    row_to_local = {int(row): idx for idx, row in enumerate(selected_rows.detach().cpu().tolist())}
    for idx, row in enumerate(candidate_rows.detach().cpu().tolist()):
        local = row_to_local.get(int(row))
        if local is None:
            continue
        best[local] = torch.maximum(best[local], candidate_diffs[int(idx)].to(dtype=torch.float32))
    finite = torch.isfinite(best)
    finite_best = best[finite]
    improved = finite_best > float(eps)
    return {
        "row_count": float(selected_rows.numel()),
        "row_with_candidate_frac": float(finite.to(dtype=torch.float32).mean().item()) if int(finite.numel()) else 0.0,
        "row_improved_frac": float(improved.to(dtype=torch.float32).mean().item()) if int(improved.numel()) else 0.0,
        "best_diff": _summary_tensor(finite_best),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    _set_seed(int(args.seed))

    cfg = load_config(str(args.config))
    cfg.structured_env_tensor_backend = "cuda" if device.type == "cuda" else "cpu"
    cfg.train_accel = False
    cfg.train_sat = False
    cfg.train_bw = True
    cfg.exec_accel_source = str(args.exec_accel_source)
    cfg.exec_sat_source = str(args.exec_sat_source)
    cfg.exec_bw_source = "policy"
    cfg.checkpoint_eval_enabled = False
    cfg.checkpoint_eval_interval_updates = 0
    cfg.update_direction_probe_enabled = False
    cfg.update_direction_probe_branch_enabled = False
    cfg.bw_clean_per_user_enabled = False
    cfg.bw_clean_native_branch_chunk_size = max(int(args.branch_chunk_size), 0)
    cfg.native_branch_replay_reuse_sub_workspace = not bool(args.disable_branch_workspace_reuse)
    if bool(args.force_score_only_softmax):
        cfg.structured_bw_parameterization = "score_only_softmax"
    if args.rollout_env_steps is not None:
        cfg.buffer_size = int(args.rollout_env_steps)
    if args.disable_interference:
        cfg.interference_enabled = False

    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
    )
    actor = bundle.actor.to(device)
    critic = bundle.critic.to(device)
    actor.eval()
    critic.eval()
    checkpoint_info: dict[str, Any] | None = None
    if args.actor_checkpoint:
        checkpoint_info = load_checkpoint_forgiving(
            actor,
            str(args.actor_checkpoint),
            map_location=device,
            strict=bool(args.strict_load),
        )

    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=float(getattr(cfg, "value_coef", 0.5) or 0.5),
        entropy_coef=float(getattr(cfg, "entropy_coef", 0.0) or 0.0),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=False,
        train_sat=False,
        train_bw=True,
        exec_accel_source=str(args.exec_accel_source),
        exec_sat_source=str(args.exec_sat_source),
        exec_bw_source="policy",
    )

    env_group = make_structured_env_group(cfg, num_envs=int(args.num_envs), backend="sync", mode="train")
    try:
        bind = getattr(learner, "bind_native_runtime_contract", None)
        if callable(bind):
            bind(env_group)
        env_group.reset_many([int(args.env_seed) + i for i in range(int(args.num_envs))])
        buffer = StructuredRolloutBuffer()
        learner.begin_native_rollout(
            env_group,
            rollout_env_steps=int(args.rollout_env_steps),
            num_envs=int(args.num_envs),
        )
        results = learner.collect_env_horizon_native_tensor_policy(
            env_group,
            buffer,
            horizon=int(args.rollout_env_steps),
            deterministic=True,
        )
        if len(results) != int(args.rollout_env_steps):
            raise RuntimeError(f"native rollout returned {len(results)} steps, expected {int(args.rollout_env_steps)}.")
        views = buffer.build_rollout_views(device)
        bw_stage_batch = views.training_view.stage_batches.get(2)
        if bw_stage_batch is None or int(getattr(bw_stage_batch, "num_samples", 0) or 0) <= 0:
            raise RuntimeError("rollout produced no BW stage samples.")

        num_samples = int(bw_stage_batch.num_samples)
        num_agents = int(bw_stage_batch.num_agents)
        user_dim = int(getattr(cfg, "users_obs_max"))
        transition_indices = np.asarray(bw_stage_batch.transition_indices, dtype=np.int64).reshape(-1)
        if np.any((transition_indices - 2) % 3 != 0):
            raise RuntimeError("BW transition indices are not aligned to stage id 2.")
        history_rows = torch.as_tensor(
            ((transition_indices - 2) // 3).astype(np.int64, copy=False),
            dtype=torch.long,
            device=device,
        )
        local_batch = bw_stage_batch.local_batch
        user_mask = getattr(local_batch, "user_mask", None)
        bw_valid_mask = getattr(local_batch, "bw_valid_mask", None)
        if user_mask is None or bw_valid_mask is None:
            raise RuntimeError("BW batch is missing user_mask or bw_valid_mask.")
        valid_mask = ((user_mask.to(device) > 0.5) & (bw_valid_mask.to(device) > 0.5)).reshape(
            num_samples * num_agents,
            user_dim,
        )
        ref_actions = bw_stage_batch.bw_ref_actions
        if ref_actions is None:
            ref_actions = bw_stage_batch.actions
        ref_action_3d = ref_actions.detach().to(device=device, dtype=torch.float32).reshape(
            num_samples,
            num_agents,
            user_dim,
        )
        ref_probs = _masked_probs(ref_action_3d.reshape(num_samples * num_agents, user_dim), valid_mask)
        ref_action_3d = ref_probs.reshape(num_samples, num_agents, user_dim)

        selected = _select_flat_rows(
            valid_mask=valid_mask,
            history_rows=history_rows,
            num_agents=num_agents,
            num_envs=int(args.num_envs),
            rollout_env_steps=int(args.rollout_env_steps),
            horizon=int(args.horizon),
            sample_budget=int(args.sample_budget),
            seed=int(args.seed) + 17,
        )
        if int(selected.numel()) <= 0:
            raise RuntimeError("no eligible BW rows with more than one valid user.")

        selected_samples = sorted({int(row) // num_agents for row in selected.detach().cpu().tolist()})
        selected_samples_t = torch.as_tensor(selected_samples, dtype=torch.long, device=device)
        ref_returns_unique = learner._native_bw_clean_rollout_returns_from_history(
            history_rows=[int(history_rows[int(sample)].item()) for sample in selected_samples],
            first_actions=ref_action_3d.index_select(0, selected_samples_t),
            horizon=int(args.horizon),
        )
        ref_return_by_sample = {
            int(sample): ref_returns_unique[idx]
            for idx, sample in enumerate(selected_samples)
        }
        ref_returns_flat = torch.stack(
            [ref_return_by_sample[int(row) // num_agents] for row in selected.detach().cpu().tolist()],
            dim=0,
        ).to(device=device)

        deltas = _parse_float_list(args.deltas)
        cand_actions: list[torch.Tensor] = []
        cand_history_rows: list[int] = []
        cand_flat_rows: list[int] = []
        cand_family: list[str] = []
        cand_delta: list[float] = []
        cand_user: list[int] = []
        cand_l1: list[float] = []

        selected_cpu = selected.detach().cpu().tolist()
        for flat_row in selected_cpu:
            flat_row_i = int(flat_row)
            sample = flat_row_i // num_agents
            agent = flat_row_i % num_agents
            valid_users = torch.nonzero(valid_mask[flat_row_i], as_tuple=False).reshape(-1).detach().cpu().tolist()
            if len(valid_users) <= 1:
                continue
            ref_row = ref_probs[flat_row_i]
            donor = int(torch.where(valid_mask[flat_row_i], ref_row, torch.full_like(ref_row, -1.0)).argmax().item())
            donor_mass = float(ref_row[donor].detach().item())
            for delta_raw in deltas:
                delta = min(float(delta_raw), donor_mass)
                if delta <= 1.0e-8:
                    continue
                for user in valid_users:
                    user_i = int(user)
                    if user_i == donor:
                        continue
                    action_row = ref_action_3d[int(sample)].detach().clone()
                    action_row[int(agent), user_i] += float(delta)
                    action_row[int(agent), donor] -= float(delta)
                    cand_actions.append(action_row)
                    cand_history_rows.append(int(history_rows[int(sample)].item()))
                    cand_flat_rows.append(flat_row_i)
                    cand_family.append(f"delta_{float(delta_raw):g}")
                    cand_delta.append(float(delta))
                    cand_user.append(user_i)
                    cand_l1.append(float(2.0 * delta))
            if bool(args.include_onehot):
                for user in valid_users:
                    user_i = int(user)
                    action_row = ref_action_3d[int(sample)].detach().clone()
                    action_row[int(agent)].zero_()
                    action_row[int(agent), user_i] = 1.0
                    cand_actions.append(action_row)
                    cand_history_rows.append(int(history_rows[int(sample)].item()))
                    cand_flat_rows.append(flat_row_i)
                    cand_family.append("onehot")
                    cand_delta.append(float("nan"))
                    cand_user.append(user_i)
                    cand_l1.append(float((action_row[int(agent)] - ref_probs[flat_row_i]).abs().sum().item()))
            if bool(args.include_uniform):
                action_row = ref_action_3d[int(sample)].detach().clone()
                action_row[int(agent)].zero_()
                uniform_value = 1.0 / float(len(valid_users))
                for user in valid_users:
                    action_row[int(agent), int(user)] = uniform_value
                cand_actions.append(action_row)
                cand_history_rows.append(int(history_rows[int(sample)].item()))
                cand_flat_rows.append(flat_row_i)
                cand_family.append("uniform")
                cand_delta.append(float("nan"))
                cand_user.append(-1)
                cand_l1.append(float((action_row[int(agent)] - ref_probs[flat_row_i]).abs().sum().item()))

        if not cand_actions:
            raise RuntimeError("candidate sweep produced no candidate actions.")
        cand_action_t = torch.stack(cand_actions, dim=0).to(device=device, dtype=torch.float32)
        cand_returns = learner._native_bw_clean_rollout_returns_from_history(
            history_rows=cand_history_rows,
            first_actions=cand_action_t,
            horizon=int(args.horizon),
        )
        selected_index_by_row = {int(row): idx for idx, row in enumerate(selected_cpu)}
        cand_ref_returns = torch.stack(
            [ref_returns_flat[selected_index_by_row[int(row)]] for row in cand_flat_rows],
            dim=0,
        ).to(device=device)
        cand_diffs = cand_returns - cand_ref_returns
        cand_rows_t = torch.as_tensor(cand_flat_rows, dtype=torch.long, device=device)
        cand_l1_t = torch.as_tensor(cand_l1, dtype=torch.float32, device=device)
        eps = float(args.gate_eps)

        family_reports: dict[str, Any] = {}
        for family in sorted(set(cand_family)):
            indices = [idx for idx, item in enumerate(cand_family) if item == family]
            idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            diffs = cand_diffs.index_select(0, idx_t)
            rows = cand_rows_t.index_select(0, idx_t)
            family_reports[family] = {
                "candidate_count": float(len(indices)),
                "candidate_positive_frac": float((diffs > eps).to(dtype=torch.float32).mean().item()),
                "return_diff": _summary_tensor(diffs),
                "action_l1": _summary_tensor(cand_l1_t.index_select(0, idx_t)),
                "row_best": _row_best_summary(
                    selected_rows=selected,
                    candidate_rows=rows,
                    candidate_diffs=diffs,
                    eps=eps,
                ),
            }

        best_by_row = torch.full((int(selected.numel()),), -float("inf"), dtype=torch.float32, device=device)
        best_idx_by_row = torch.full((int(selected.numel()),), -1, dtype=torch.long, device=device)
        for idx, row in enumerate(cand_flat_rows):
            local = selected_index_by_row[int(row)]
            diff = cand_diffs[int(idx)].to(dtype=torch.float32)
            if bool(diff > best_by_row[local]):
                best_by_row[local] = diff
                best_idx_by_row[local] = int(idx)
        finite_best = best_by_row[torch.isfinite(best_by_row)]
        improved_mask = best_by_row > eps
        ref_negative_mask = ref_returns_flat < -eps
        ref_zeroish_mask = ~ref_negative_mask
        improved_negative = improved_mask[ref_negative_mask] if int(ref_negative_mask.numel()) else improved_mask[:0]
        improved_zeroish = improved_mask[ref_zeroish_mask] if int(ref_zeroish_mask.numel()) else improved_mask[:0]
        best_cases: list[dict[str, Any]] = []
        top_k = min(int(args.top_cases), int(selected.numel()))
        if top_k > 0:
            order = torch.argsort(best_by_row, descending=True)[:top_k].detach().cpu().tolist()
            for local in order:
                cand_idx = int(best_idx_by_row[int(local)].detach().item())
                flat_row = int(selected[int(local)].detach().item())
                sample = flat_row // num_agents
                agent = flat_row % num_agents
                best_cases.append(
                    {
                        "flat_row": flat_row,
                        "sample": sample,
                        "agent": agent,
                        "history_row": int(history_rows[int(sample)].detach().item()),
                        "source_step": int(int(history_rows[int(sample)].detach().item()) // max(int(args.num_envs), 1)),
                        "source_env": int(int(history_rows[int(sample)].detach().item()) % max(int(args.num_envs), 1)),
                        "best_family": cand_family[cand_idx] if cand_idx >= 0 else "",
                        "best_user": int(cand_user[cand_idx]) if cand_idx >= 0 else -1,
                        "best_return_diff": float(best_by_row[int(local)].detach().cpu().item()),
                        "best_return": float(cand_returns[cand_idx].detach().cpu().item()) if cand_idx >= 0 else 0.0,
                        "ref_return": float(ref_returns_flat[int(local)].detach().cpu().item()),
                        "best_action_l1": float(cand_l1[cand_idx]) if cand_idx >= 0 else 0.0,
                        "valid_count": float(valid_mask[flat_row].to(dtype=torch.float32).sum().detach().cpu().item()),
                    }
                )

        return {
            "config": str(args.config),
            "checkpoint_info": checkpoint_info,
            "device": str(device),
            "seed": int(args.seed),
            "env_seed": int(args.env_seed),
            "num_envs": int(args.num_envs),
            "rollout_env_steps": int(args.rollout_env_steps),
            "horizon": int(args.horizon),
            "bw_stage_samples": int(num_samples),
            "flat_rows": int(num_samples * num_agents),
            "eligible_selected_rows": int(selected.numel()),
            "candidate_count": int(len(cand_actions)),
            "exec_sources": {
                "accel": str(args.exec_accel_source),
                "sat": str(args.exec_sat_source),
                "bw": "policy",
            },
            "parameterization": str(getattr(cfg, "structured_bw_parameterization", "")),
            "interference_enabled": bool(getattr(cfg, "interference_enabled", False)),
            "selected_valid_count": _summary_tensor(valid_mask.index_select(0, selected).to(dtype=torch.float32).sum(dim=1)),
            "ref_return": _summary_tensor(ref_returns_flat),
            "overall": {
                "candidate_positive_frac": float((cand_diffs > eps).to(dtype=torch.float32).mean().item()),
                "row_improved_frac": float(improved_mask.to(dtype=torch.float32).mean().item()),
                "ref_negative_frac": float(ref_negative_mask.to(dtype=torch.float32).mean().item()),
                "row_improved_frac_given_ref_negative": float(improved_negative.to(dtype=torch.float32).mean().item())
                if int(improved_negative.numel()) > 0
                else 0.0,
                "row_improved_frac_given_ref_zeroish": float(improved_zeroish.to(dtype=torch.float32).mean().item())
                if int(improved_zeroish.numel()) > 0
                else 0.0,
                "best_diff": _summary_tensor(finite_best),
                "best_diff_given_ref_negative": _summary_tensor(best_by_row[ref_negative_mask]),
                "all_candidate_diff": _summary_tensor(cand_diffs),
            },
            "families": family_reports,
            "top_best_cases": best_cases,
        }
    finally:
        close_structured_env_group(env_group)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native exact replay sweep for BW clean-teacher deltas and finite candidate actions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor_checkpoint", default=None)
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--env_seed", type=int, default=42000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--sample_budget", type=int, default=128)
    parser.add_argument("--branch_chunk_size", type=int, default=128)
    parser.add_argument("--deltas", nargs="*", default=["0.02", "0.05", "0.1", "0.2", "0.5"])
    parser.add_argument("--include_onehot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_uniform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate_eps", type=float, default=1.0e-6)
    parser.add_argument("--exec_accel_source", default="cluster_center_queue_aware")
    parser.add_argument("--exec_sat_source", default="zero")
    parser.add_argument("--disable_interference", action="store_true")
    parser.add_argument("--disable_branch_workspace_reuse", action="store_true")
    parser.add_argument("--force_score_only_softmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--top_cases", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = run(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
