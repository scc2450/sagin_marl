from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(ROOT, "sagin_marl")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not locate repository root.")
    ROOT = parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_actor import (
    _attend,
    _gather_member_tokens,
    _make_mlp,
    _masked_member_mean,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass, _padcat_tensors
from sagin_marl.rl.structured_types import LocalSatState
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    n = int(pred.size)
    if n <= 1:
        return 1.0
    pred_rank = np.argsort(np.argsort(-pred, kind="mergesort"), kind="mergesort").astype(np.float64)
    true_rank = np.argsort(np.argsort(-truth, kind="mergesort"), kind="mergesort").astype(np.float64)
    d2 = np.sum((pred_rank - true_rank) ** 2)
    return float(1.0 - (6.0 * d2) / (n * (n * n - 1.0)))


def _argmax_rank01(pred: np.ndarray, truth: np.ndarray) -> float:
    n = int(pred.size)
    if n <= 1:
        return 0.0
    chosen = int(np.argmax(pred))
    true_order = np.argsort(-truth, kind="mergesort")
    rank = int(np.flatnonzero(true_order == chosen)[0])
    return float(rank) / float(max(n - 1, 1))


def _local_subset_quality(local_state: LocalSatState) -> torch.Tensor:
    member_edges, member_mask = _gather_member_tokens(local_state.sat_edges, local_state.subset_members)
    se = member_edges[..., 7]
    projected_bw = member_edges[..., 10]
    return (se * projected_bw * member_mask.to(se.dtype)).sum(dim=-1)


def _normalize_quality(quality: torch.Tensor, subset_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(quality)
    valid = subset_mask > 0.5
    for row_idx in range(int(quality.shape[0])):
        row_valid = valid[row_idx]
        if not torch.any(row_valid):
            continue
        row_values = quality[row_idx, row_valid]
        mean = row_values.mean()
        std = row_values.std(unbiased=False).clamp_min(1.0e-6)
        out[row_idx, row_valid] = (row_values - mean) / std
    return out


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, subset_mask: torch.Tensor) -> torch.Tensor:
    valid = subset_mask > 0.5
    if not torch.any(valid):
        return pred.new_zeros(())
    diff = pred[valid] - target[valid]
    return torch.mean(diff * diff)


@dataclass
class SatProbeSample:
    local_state: LocalSatState
    quality: torch.Tensor
    quality_norm: torch.Tensor
    policy_logits: torch.Tensor


class SatProbeDataset(Dataset):
    def __init__(self, samples: Sequence[SatProbeSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SatProbeSample:
        return self.samples[idx]


def _collate_probe_samples(samples: Sequence[SatProbeSample]):
    local_state = _collate_dataclass([sample.local_state for sample in samples], torch.device("cpu"))
    quality = _padcat_tensors([sample.quality for sample in samples], torch.device("cpu"))
    quality_norm = _padcat_tensors([sample.quality_norm for sample in samples], torch.device("cpu"))
    policy_logits = _padcat_tensors([sample.policy_logits for sample in samples], torch.device("cpu"))
    return local_state, quality, quality_norm, policy_logits


class StructuredSatQualityProbe(nn.Module):
    def __init__(self, ego_dim: int, sat_node_dim: int, sat_edge_dim: int, hidden_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.ego_encoder = _make_mlp(ego_dim, hidden_dim, embed_dim)
        self.query_proj_1 = _make_mlp(embed_dim, hidden_dim, embed_dim)
        self.query_proj_2 = _make_mlp(embed_dim, hidden_dim, embed_dim)
        self.sat_encoder = _make_mlp(sat_node_dim + sat_edge_dim, hidden_dim, embed_dim)
        self.sat_refine = _make_mlp(embed_dim * 2, hidden_dim, embed_dim)
        self.ego_fusion = _make_mlp(embed_dim * 2, hidden_dim, embed_dim)
        self.subset_projector = _make_mlp(embed_dim * 2 + 1, hidden_dim, embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, local_state: LocalSatState) -> torch.Tensor:
        ego_0 = self.ego_encoder(local_state.ego_uav_after_accel)
        sat_0 = self.sat_encoder(torch.cat([local_state.sat_nodes, local_state.sat_edges], dim=-1))
        sat_ctx_1 = _attend(self.query_proj_1(ego_0), sat_0, local_state.sat_mask)
        ego_1 = ego_0 + self.ego_fusion(torch.cat([ego_0, sat_ctx_1], dim=-1))
        sat_1 = sat_0 + self.sat_refine(torch.cat([sat_0, ego_1.unsqueeze(1).expand_as(sat_0)], dim=-1))
        sat_ctx_2 = _attend(self.query_proj_2(ego_1), sat_1, local_state.sat_mask)
        member_tokens, member_mask = _gather_member_tokens(sat_1, local_state.subset_members)
        subset_mean = _masked_member_mean(member_tokens, member_mask)
        member_mask_f = member_mask.to(member_tokens.dtype).unsqueeze(-1)
        subset_dev = ((member_tokens - subset_mean.unsqueeze(-2)).abs() * member_mask_f).amax(dim=-2)
        subset_size = member_mask.to(member_tokens.dtype).sum(dim=-1, keepdim=True) / max(
            float(local_state.subset_members.shape[-1]), 1.0
        )
        subset_repr = self.subset_projector(torch.cat([subset_mean, subset_dev, subset_size], dim=-1))
        fused = torch.cat([ego_1.unsqueeze(1).expand(-1, subset_repr.shape[1], -1), sat_ctx_2.unsqueeze(1).expand(-1, subset_repr.shape[1], -1), subset_repr], dim=-1)
        return self.scorer(fused).squeeze(-1)


def _split_seeds(seed_base: int, count: int, offset: int) -> list[int]:
    return [int(seed_base + offset + idx) for idx in range(int(count))]


def _collect_split(cfg_path: str, actor, device: torch.device, seeds: Sequence[int]) -> list[SatProbeSample]:
    cfg = load_config(cfg_path)
    env = make_structured_env(cfg, mode="script")
    actor.eval()
    samples: list[SatProbeSample] = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            del obs
            driver = as_structured_driver(env)
            done = False
            while not done:
                z0 = driver.begin_step()
                accel_states = driver.build_local_accel_states(z0)
                accel_batch = _collate_dataclass(accel_states, device)
                with torch.no_grad():
                    accel_out = actor.act_accel(accel_batch, deterministic=True)
                z1 = driver.run_accel_stage(accel_out.action.cpu().numpy())
                sat_states = driver.build_sat_pair_candidates(z1)
                sat_batch = _collate_dataclass(sat_states, device)
                with torch.no_grad():
                    sat_out = actor.act_sat_pair(sat_batch, deterministic=True)
                    quality = _local_subset_quality(sat_batch)
                    quality_norm = _normalize_quality(quality, sat_batch.subset_mask)
                for sample_idx, sample_state in enumerate(sat_states):
                    subset_count = int(sample_state.subset_mask.shape[1])
                    samples.append(
                        SatProbeSample(
                            local_state=sample_state,
                            quality=quality[sample_idx : sample_idx + 1, :subset_count].cpu(),
                            quality_norm=quality_norm[sample_idx : sample_idx + 1, :subset_count].cpu(),
                            policy_logits=sat_out.logits[sample_idx : sample_idx + 1, :subset_count].cpu(),
                        )
                    )
                sat_action = driver.decode_sat_pair_actions(sat_states, sat_out.pair_index.cpu().tolist())
                z2 = driver.run_sat_stage(sat_action)
                bw_states = driver.build_bw_valid_context(z2)
                bw_batch = _collate_dataclass(bw_states, device)
                with torch.no_grad():
                    bw_out = actor.act_bw(bw_batch, deterministic=True)
                step = driver.execute_stage_bw_and_step(bw_out.action.cpu().numpy())
                done = bool(any(step.terminations.values()) or any(step.truncations.values()))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return samples


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    spearman_values: list[float] = []
    argmax_rank_values: list[float] = []
    with torch.no_grad():
        for local_state, quality, quality_norm, _policy_logits in loader:
            local_state = _move_dataclass_to(local_state, device)
            quality = quality.to(device)
            quality_norm = quality_norm.to(device)
            pred = model(local_state)
            losses.append(float(_masked_mse(pred, quality_norm, local_state.subset_mask).item()))
            subset_mask = local_state.subset_mask > 0.5
            pred_np = pred.cpu().numpy()
            quality_np = quality.cpu().numpy()
            mask_np = subset_mask.cpu().numpy()
            for row_idx in range(int(pred_np.shape[0])):
                valid = mask_np[row_idx]
                if not np.any(valid):
                    continue
                pred_row = pred_np[row_idx, valid]
                quality_row = quality_np[row_idx, valid]
                spearman_values.append(_spearman_desc(pred_row, quality_row))
                argmax_rank_values.append(_argmax_rank01(pred_row, quality_row))
    return {
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "spearman_mean": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "argmax_rank01_mean": float(np.mean(argmax_rank_values)) if argmax_rank_values else 0.0,
    }


def _evaluate_policy(loader: DataLoader) -> dict[str, float]:
    spearman_values: list[float] = []
    argmax_rank_values: list[float] = []
    for local_state, quality, _quality_norm, policy_logits in loader:
        mask_np = (local_state.subset_mask > 0.5).cpu().numpy()
        quality_np = quality.cpu().numpy()
        logits_np = policy_logits.cpu().numpy()
        for row_idx in range(int(logits_np.shape[0])):
            valid = mask_np[row_idx]
            if not np.any(valid):
                continue
            pred_row = logits_np[row_idx, valid]
            quality_row = quality_np[row_idx, valid]
            spearman_values.append(_spearman_desc(pred_row, quality_row))
            argmax_rank_values.append(_argmax_rank01(pred_row, quality_row))
    return {
        "spearman_mean": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "argmax_rank01_mean": float(np.mean(argmax_rank_values)) if argmax_rank_values else 0.0,
    }


def _move_dataclass_to(local_state: LocalSatState, device: torch.device) -> LocalSatState:
    return LocalSatState(
        ego_uav_after_accel=local_state.ego_uav_after_accel.to(device),
        sat_nodes=local_state.sat_nodes.to(device),
        sat_edges=local_state.sat_edges.to(device),
        sat_mask=local_state.sat_mask.to(device),
        subset_tokens=local_state.subset_tokens.to(device),
        subset_mask=local_state.subset_mask.to(device),
        subset_members=local_state.subset_members.to(device),
    )


def _train_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = deepcopy(probe.state_dict())
    best_metrics = {"loss_mean": float("inf"), "spearman_mean": -1.0, "argmax_rank01_mean": 1.0}
    for _epoch in range(int(epochs)):
        probe.train()
        for local_state, _quality, quality_norm, _policy_logits in train_loader:
            local_state = _move_dataclass_to(local_state, device)
            quality_norm = quality_norm.to(device)
            pred = probe(local_state)
            loss = _masked_mse(pred, quality_norm, local_state.subset_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_metrics = _evaluate_model(probe, val_loader, device)
        if val_metrics["spearman_mean"] > best_metrics["spearman_mean"]:
            best_state = deepcopy(probe.state_dict())
            best_metrics = dict(val_metrics)
    probe.load_state_dict(best_state)
    return probe, best_metrics


def _build_probe_from_dataset(dataset: SatProbeDataset, hidden_dim: int, embed_dim: int) -> StructuredSatQualityProbe:
    if len(dataset) <= 0:
        raise ValueError("Probe dataset is empty")
    sample = dataset[0].local_state
    return StructuredSatQualityProbe(
        ego_dim=int(sample.ego_uav_after_accel.shape[-1]),
        sat_node_dim=int(sample.sat_nodes.shape[-1]),
        sat_edge_dim=int(sample.sat_edges.shape[-1]),
        hidden_dim=int(hidden_dim),
        embed_dim=int(embed_dim),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--val-episodes", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--episode-seed-base", type=int, default=52000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out-name", type=str, default="structured_sat_probe_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config_source.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config_source.yaml or config.yaml under {run_dir}")
    cfg = load_config(str(cfg_path))
    device = torch.device("cpu")
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64)
    actor = bundle.actor.to(device)
    actor.eval()

    train_seeds = _split_seeds(int(args.episode_seed_base), int(args.train_episodes), 0)
    val_seeds = _split_seeds(int(args.episode_seed_base), int(args.val_episodes), 1000)
    test_seeds = _split_seeds(int(args.episode_seed_base), int(args.test_episodes), 2000)

    results: dict[str, object] = {
        "run_dir": str(run_dir),
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "updates": {},
    }

    for update in args.updates:
        checkpoint_path = run_dir / f"actor_u{int(update):04d}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        load_checkpoint_forgiving(actor, str(checkpoint_path), map_location=device, strict=False)
        actor.eval()

        train_samples = _collect_split(str(cfg_path), actor, device, train_seeds)
        val_samples = _collect_split(str(cfg_path), actor, device, val_seeds)
        test_samples = _collect_split(str(cfg_path), actor, device, test_seeds)

        train_dataset = SatProbeDataset(train_samples)
        val_dataset = SatProbeDataset(val_samples)
        test_dataset = SatProbeDataset(test_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=_collate_probe_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=_collate_probe_samples,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=_collate_probe_samples,
        )

        probe = _build_probe_from_dataset(train_dataset, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim)).to(device)
        probe, best_val = _train_probe(
            probe,
            train_loader,
            val_loader,
            device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        test_metrics = _evaluate_model(probe, test_loader, device)
        policy_metrics = _evaluate_policy(test_loader)
        results["updates"][f"u{int(update):04d}"] = {
            "checkpoint": str(checkpoint_path),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "policy_test": policy_metrics,
            "probe_val_best": best_val,
            "probe_test": test_metrics,
        }
        print(
            json.dumps(
                {
                    "update": int(update),
                    "policy_test": policy_metrics,
                    "probe_test": test_metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
