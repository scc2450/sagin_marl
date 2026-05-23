from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_types import LocalBwState


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def _first_linear_in_features(module: torch.nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            return int(child.in_features)
    raise RuntimeError(f"Could not infer input dimension from {module.__class__.__name__}.")


def _infer_bw_dims(bw_policy: torch.nn.Module, cfg: Any) -> dict[str, int]:
    ego_dim = _first_linear_in_features(getattr(bw_policy, "ego_encoder"))
    sat_pair_dim = _first_linear_in_features(getattr(bw_policy, "sat_encoder"))
    user_pair_dim = _first_linear_in_features(getattr(bw_policy, "user_encoder"))
    sat_edge_dim = 14
    user_edge_dim = 7
    sat_node_dim = int(sat_pair_dim) - int(sat_edge_dim)
    user_node_dim = int(user_pair_dim) - int(user_edge_dim)
    if user_node_dim <= 2:
        raise RuntimeError("workload proportional fit requires user_nodes[..., 2] to exist.")
    if sat_node_dim <= 0:
        raise RuntimeError("Could not infer a positive BW sat node dimension.")
    return {
        "ego_dim": int(ego_dim),
        "sat_node_dim": int(sat_node_dim),
        "sat_edge_dim": int(sat_edge_dim),
        "sat_count": int(getattr(cfg, "sats_obs_max", 6) or 6),
        "user_node_dim": int(user_node_dim),
        "user_edge_dim": int(user_edge_dim),
        "user_count": int(getattr(cfg, "users_obs_max", 20) or 20),
    }


def _make_batch(
    *,
    batch_size: int,
    dims: dict[str, int],
    device: torch.device,
    generator: torch.Generator,
) -> tuple[LocalBwState, torch.Tensor, torch.Tensor]:
    batch = int(batch_size)
    user_count = int(dims["user_count"])
    sat_count = int(dims["sat_count"])
    user_node_dim = int(dims["user_node_dim"])
    user_edge_dim = int(dims["user_edge_dim"])

    ego = torch.zeros((batch, int(dims["ego_dim"])), dtype=torch.float32, device=device)
    sat_nodes = torch.zeros((batch, sat_count, int(dims["sat_node_dim"])), dtype=torch.float32, device=device)
    sat_edges = torch.zeros((batch, sat_count, int(dims["sat_edge_dim"])), dtype=torch.float32, device=device)
    sat_mask = torch.ones((batch, sat_count), dtype=torch.bool, device=device)

    valid_count = torch.randint(1, user_count + 1, (batch,), generator=generator, device=device)
    random_scores = torch.rand((batch, user_count), generator=generator, device=device)
    ranks = random_scores.argsort(dim=-1)
    user_mask = torch.zeros((batch, user_count), dtype=torch.bool, device=device)
    for row in range(batch):
        user_mask[row, ranks[row, : int(valid_count[row].item())]] = True
    bw_valid_mask = user_mask.clone()

    workload = torch.rand((batch, user_count), generator=generator, device=device).pow(2.0)
    workload = workload * user_mask.to(workload.dtype)
    workload_sum = workload.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = workload / workload_sum

    user_nodes = torch.zeros((batch, user_count, user_node_dim), dtype=torch.float32, device=device)
    user_edges = torch.zeros((batch, user_count, user_edge_dim), dtype=torch.float32, device=device)
    user_nodes[..., 2] = workload
    user_edges[..., 3] = user_mask.to(torch.float32)
    user_edges[..., 4] = bw_valid_mask.to(torch.float32)

    state = LocalBwState(
        ego_uav_after_sat=ego,
        sat_nodes=sat_nodes,
        sat_edges=sat_edges,
        sat_mask=sat_mask,
        user_nodes=user_nodes,
        user_edges=user_edges,
        user_mask=user_mask,
        bw_valid_mask=bw_valid_mask,
    )
    return state, target, bw_valid_mask


@torch.no_grad()
def _evaluate(
    bw_policy: torch.nn.Module,
    *,
    dims: dict[str, int],
    device: torch.device,
    batch_size: int,
    batches: int,
    generator: torch.Generator,
) -> dict[str, float]:
    bw_policy.eval()
    losses: list[float] = []
    l1s: list[float] = []
    invalid_masses: list[float] = []
    sum_errors: list[float] = []
    for _ in range(int(batches)):
        state, target, mask = _make_batch(
            batch_size=batch_size,
            dims=dims,
            device=device,
            generator=generator,
        )
        action = bw_policy(state, deterministic=True).action
        losses.append(float(F.mse_loss(action, target).item()))
        valid_denom = mask.to(action.dtype).sum().clamp_min(1.0)
        l1s.append(float(((action - target).abs() * mask.to(action.dtype)).sum().div(valid_denom).item()))
        invalid_masses.append(float((action * (~mask).to(action.dtype)).abs().sum(dim=-1).mean().item()))
        sum_errors.append(float((action.sum(dim=-1) - 1.0).abs().mean().item()))
    return {
        "mse": float(sum(losses) / max(len(losses), 1)),
        "valid_l1_per_slot": float(sum(l1s) / max(len(l1s), 1)),
        "invalid_mass": float(sum(invalid_masses) / max(len(invalid_masses), 1)),
        "simplex_sum_abs_error": float(sum(sum_errors) / max(len(sum_errors), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Actor-only synthetic fit: target BW allocation is proportional to user_nodes[..., 2] workload."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--actor_arch", choices=["pooled_summary", "competition"], default="competition")
    parser.add_argument("--competition_layers", type=int, default=2)
    parser.add_argument("--competition_heads", type=int, default=4)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _device(str(args.device))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 17)

    cfg = load_config(str(args.config))
    cfg.traffic_model = "homogeneous"
    cfg.hotspot_num_subsets = 0
    cfg.structured_bw_actor_arch = str(args.actor_arch)
    cfg.structured_bw_competition_layers = max(int(args.competition_layers), 1)
    cfg.structured_bw_competition_heads = max(int(args.competition_heads), 1)
    cfg.structured_bw_parameterization = "score_only_softmax"

    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        build_critic=False,
    )
    bw_policy = bundle.actor.bw_policy.to(device)
    dims = _infer_bw_dims(bw_policy, cfg)
    optimizer = torch.optim.Adam(bw_policy.parameters(), lr=float(args.lr))

    before = _evaluate(
        bw_policy,
        dims=dims,
        device=device,
        batch_size=int(args.batch_size),
        batches=int(args.eval_batches),
        generator=generator,
    )

    bw_policy.train()
    last_loss = 0.0
    for _step in range(int(args.steps)):
        state, target, _mask = _make_batch(
            batch_size=int(args.batch_size),
            dims=dims,
            device=device,
            generator=generator,
        )
        action = bw_policy(state, deterministic=True).action
        loss = F.mse_loss(action, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    after = _evaluate(
        bw_policy,
        dims=dims,
        device=device,
        batch_size=int(args.batch_size),
        batches=int(args.eval_batches),
        generator=generator,
    )

    payload = {
        "ok": True,
        "config": str(args.config),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "actor_arch": str(args.actor_arch),
        "competition_layers": int(cfg.structured_bw_competition_layers),
        "competition_heads": int(cfg.structured_bw_competition_heads),
        "parameterization": str(cfg.structured_bw_parameterization),
        "target_rule": "user_nodes[..., 2] / sum_valid(user_nodes[..., 2])",
        "dims": dims,
        "last_train_mse": last_loss,
        "before": before,
        "after": after,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(str(args.json_out), "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
