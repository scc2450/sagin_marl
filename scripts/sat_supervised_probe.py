from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sagin_marl.env import channel
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import cluster_center_queue_aware_policy
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


@dataclass
class SplitData:
    own: np.ndarray
    sats: np.ndarray
    valid: np.ndarray
    rates: np.ndarray
    targets: np.ndarray


class SatProbeDataset(Dataset):
    def __init__(self, split: SplitData):
        self.own = torch.from_numpy(split.own).float()
        self.sats = torch.from_numpy(split.sats).float()
        self.valid = torch.from_numpy(split.valid).float()
        self.rates = torch.from_numpy(split.rates).float()
        self.targets = torch.from_numpy(split.targets).float()

    def __len__(self) -> int:
        return int(self.own.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.own[idx],
            self.sats[idx],
            self.valid[idx],
            self.rates[idx],
            self.targets[idx],
        )


class SatRateProbe(nn.Module):
    def __init__(self, own_dim: int, sat_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.own_mlp = nn.Sequential(
            nn.Linear(own_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.sat_mlp = nn.Sequential(
            nn.Linear(sat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + sat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, own: torch.Tensor, sats: torch.Tensor) -> torch.Tensor:
        own_ctx = self.own_mlp(own)
        sat_ctx = self.sat_mlp(sats)
        own_exp = own_ctx.unsqueeze(1).expand(-1, sats.shape[1], -1)
        x = torch.cat([own_exp, sat_ctx, sats], dim=-1)
        return self.head(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase1_actions_curriculum_joint_3heads_fading_interference_vsat_precomp_satonly_nooverlap.yaml',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='runs/phase1_actions/sat_supervised_probe_20260329',
    )
    parser.add_argument('--train_episodes', type=int, default=24)
    parser.add_argument('--val_episodes', type=int, default=8)
    parser.add_argument('--test_episodes', type=int, default=8)
    parser.add_argument('--seed_base', type=int, default=52000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--torch_threads', type=int, default=2)
    return parser.parse_args()


def current_visible(env: SaginParallelEnv, sat_pos: np.ndarray) -> list[list[int]]:
    visible = [list(v) for v in getattr(env, 'last_visible_candidates', [])]
    if len(visible) != env.cfg.num_uav:
        visible = env._visible_sats_sorted(sat_pos)
    return [list(v[: env.cfg.sats_obs_max]) for v in visible]


def projected_rate_labels(env: SaginParallelEnv, sat_pos: np.ndarray, sat_vel: np.ndarray, visible: list[list[int]]) -> np.ndarray:
    cfg = env.cfg
    labels = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
    elev_matrix = env._get_elevation_matrix(sat_pos)
    loss_matrix = env._get_backhaul_loss_matrix(sat_pos)
    for u in range(cfg.num_uav):
        sat_idx = np.asarray(visible[u][: cfg.sats_obs_max], dtype=np.int32)
        if sat_idx.size == 0:
            continue
        rel_pos = sat_pos[sat_idx] - env._uav_ecef(u)[None, :]
        d = np.linalg.norm(rel_pos, axis=1) + 1e-9
        gain = env._backhaul_gain_const / np.maximum(d * d, 1e-9)
        if loss_matrix is not None:
            gain = gain * loss_matrix[u, sat_idx]
        bw = env._projected_sat_bandwidth(u, sat_idx)
        if cfg.doppler_enabled or cfg.doppler_atten_enabled or cfg.doppler_observed:
            raw_nu = env._doppler_many(u, sat_idx, sat_pos, sat_vel)
            nu_eff, _ = env._effective_doppler_array(u, sat_idx, raw_nu)
        else:
            nu_eff = np.zeros((sat_idx.size,), dtype=np.float32)
        snr = channel.snr_linear(cfg.uav_tx_power, gain, cfg.noise_density, bw)
        if cfg.doppler_observed and cfg.doppler_atten_enabled:
            snr = snr * channel.doppler_attenuation(nu_eff, cfg.subcarrier_spacing)
        se = np.asarray(channel.spectral_efficiency(snr), dtype=np.float32)
        rate = (bw * se).astype(np.float32, copy=False)
        elev_ok = elev_matrix[u, sat_idx] >= cfg.theta_min_rad
        valid = elev_ok.astype(bool)
        if cfg.doppler_enabled:
            valid = valid & (np.abs(nu_eff) <= cfg.nu_max)
        rate = np.where(valid, rate, 0.0).astype(np.float32, copy=False)
        labels[u, : sat_idx.size] = rate
    return labels


def collect_split(cfg_path: str, seeds: list[int]) -> SplitData:
    cfg = load_config(cfg_path)
    env = make_structured_env(cfg, mode="script")
    own_list: list[np.ndarray] = []
    sats_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    rates_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            sat_pos, sat_vel = env._get_orbit_states()
            visible = current_visible(env, sat_pos)
            label_rates = projected_rate_labels(env, sat_pos, sat_vel, visible)
            for agent_idx, agent in enumerate(env.agents):
                agent_obs = obs[agent]
                own = np.asarray(agent_obs['own'], dtype=np.float32)
                sats = np.asarray(agent_obs['sats'], dtype=np.float32)
                valid = np.asarray(agent_obs.get('sat_valid_mask', agent_obs['sats_mask']), dtype=np.float32)
                rates = np.asarray(label_rates[agent_idx], dtype=np.float32)
                targets = np.log1p(rates / 1.0e6).astype(np.float32, copy=False)
                own_list.append(own)
                sats_list.append(sats)
                valid_list.append(valid)
                rates_list.append(rates)
                targets_list.append(targets)

            obs_list = list(obs.values())
            accel_actions, bw_logits, sat_logits = cluster_center_queue_aware_policy(
                obs_list,
                cfg,
                getattr(env, 'gu_cluster_centers', None),
                getattr(env, 'gu_cluster_counts', None),
            )
            actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
            obs, _, terms, truncs, _ = env.step(actions)
            done = bool(list(terms.values())[0] or list(truncs.values())[0])

    return SplitData(
        own=np.asarray(own_list, dtype=np.float32),
        sats=np.asarray(sats_list, dtype=np.float32),
        valid=np.asarray(valid_list, dtype=np.float32),
        rates=np.asarray(rates_list, dtype=np.float32),
        targets=np.asarray(targets_list, dtype=np.float32),
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid > 0.5
    if not torch.any(mask):
        return pred.new_zeros(())
    diff = pred[mask] - target[mask]
    return torch.mean(diff * diff)


def spearman_desc(pred: np.ndarray, truth: np.ndarray) -> float:
    n = pred.size
    if n <= 1:
        return 1.0
    pred_rank = np.argsort(np.argsort(-pred, kind='mergesort'), kind='mergesort').astype(np.float64)
    true_rank = np.argsort(np.argsort(-truth, kind='mergesort'), kind='mergesort').astype(np.float64)
    d2 = np.sum((pred_rank - true_rank) ** 2)
    return float(1.0 - (6.0 * d2) / (n * (n * n - 1.0)))


def evaluate_scores(scores: np.ndarray, split: SplitData, cfg) -> dict[str, float]:
    top1_hits = 0.0
    top2_overlap = 0.0
    exact_top2 = 0.0
    regret_vals: list[float] = []
    rho_vals: list[float] = []
    se_invcount_vals: list[float] = []
    total = split.own.shape[0]
    heuristic_top1_hits = 0.0
    heuristic_top2_overlap = 0.0
    heuristic_exact_top2 = 0.0
    heuristic_regret_vals: list[float] = []
    for i in range(total):
        valid_idx = np.flatnonzero(split.valid[i] > 0.5)
        if valid_idx.size == 0:
            continue
        truth = split.rates[i, valid_idx].astype(np.float64)
        pred = scores[i, valid_idx].astype(np.float64)
        truth_order = valid_idx[np.argsort(-truth, kind='mergesort')]
        pred_order = valid_idx[np.argsort(-pred, kind='mergesort')]
        k = min(int(cfg.N_RF), int(valid_idx.size))
        truth_top = truth_order[:k]
        pred_top = pred_order[:k]
        top1_hits += 1.0 if int(pred_order[0]) == int(truth_order[0]) else 0.0
        overlap = len(set(pred_top.tolist()) & set(truth_top.tolist()))
        top2_overlap += float(overlap) / max(k, 1)
        exact_top2 += 1.0 if set(pred_top.tolist()) == set(truth_top.tolist()) else 0.0
        truth_best_value = float(np.sum(split.rates[i, truth_top]))
        pred_value = float(np.sum(split.rates[i, pred_top]))
        regret_vals.append((truth_best_value - pred_value) / max(truth_best_value, 1e-9))
        rho_vals.append(spearman_desc(pred, truth))

        sats = split.sats[i]
        heuristic = (sats[:, 7] * sats[:, 10]).astype(np.float64)
        heur_valid = heuristic[valid_idx]
        heuristic_order = valid_idx[np.argsort(-heur_valid, kind='mergesort')]
        heur_top = heuristic_order[:k]
        heuristic_top1_hits += 1.0 if int(heuristic_order[0]) == int(truth_order[0]) else 0.0
        heur_overlap = len(set(heur_top.tolist()) & set(truth_top.tolist()))
        heuristic_top2_overlap += float(heur_overlap) / max(k, 1)
        heuristic_exact_top2 += 1.0 if set(heur_top.tolist()) == set(truth_top.tolist()) else 0.0
        heur_value = float(np.sum(split.rates[i, heur_top]))
        heuristic_regret_vals.append((truth_best_value - heur_value) / max(truth_best_value, 1e-9))
        se_invcount_vals.append(float(np.sum(split.rates[i, heur_top])))

    denom = max(total, 1)
    out = {
        'samples': int(total),
        'top1_accuracy': top1_hits / denom,
        'top2_overlap_rate': top2_overlap / denom,
        'exact_top2_match_rate': exact_top2 / denom,
        'mean_regret': float(np.mean(regret_vals)) if regret_vals else 0.0,
        'p90_regret': float(np.quantile(regret_vals, 0.9)) if regret_vals else 0.0,
        'mean_spearman': float(np.mean(rho_vals)) if rho_vals else 0.0,
        'heuristic_se_x_invcount_top1_accuracy': heuristic_top1_hits / denom,
        'heuristic_se_x_invcount_top2_overlap_rate': heuristic_top2_overlap / denom,
        'heuristic_se_x_invcount_exact_top2_match_rate': heuristic_exact_top2 / denom,
        'heuristic_se_x_invcount_mean_regret': float(np.mean(heuristic_regret_vals)) if heuristic_regret_vals else 0.0,
    }
    return out


def predict_scores(model: nn.Module, split: SplitData, batch_size: int, device: torch.device) -> np.ndarray:
    ds = SatProbeDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for own, sats, valid, _, _ in loader:
            own = own.to(device)
            sats = sats.to(device)
            valid = valid.to(device)
            score = model(own, sats)
            score = score.masked_fill(valid <= 0.5, -1.0e9)
            preds.append(score.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 0), dtype=np.float32)


def train_probe(train_split: SplitData, val_split: SplitData, cfg, epochs: int, batch_size: int, lr: float, weight_decay: float, device: torch.device):
    model = SatRateProbe(train_split.own.shape[1], train_split.sats.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(SatProbeDataset(train_split), batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for own, sats, valid, _rates, targets in train_loader:
            own = own.to(device)
            sats = sats.to(device)
            valid = valid.to(device)
            targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(own, sats)
            loss = masked_mse(pred, targets, valid)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_scores = predict_scores(model, val_split, batch_size=batch_size, device=device)
        val_metrics = evaluate_scores(val_scores, val_split, cfg)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_top1_accuracy': val_metrics['top1_accuracy'],
            'val_exact_top2_match_rate': val_metrics['exact_top2_match_rate'],
            'val_mean_regret': val_metrics['mean_regret'],
            'val_mean_spearman': val_metrics['mean_spearman'],
        }
        history.append(record)
        key = (-val_metrics['exact_top2_match_rate'], val_metrics['mean_regret'], -val_metrics['top1_accuracy'])
        if best_val is None or key < best_val:
            best_val = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_history(path: str, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    train_seeds = list(range(args.seed_base, args.seed_base + args.train_episodes))
    val_base = args.seed_base + args.train_episodes
    val_seeds = list(range(val_base, val_base + args.val_episodes))
    test_base = val_base + args.val_episodes
    test_seeds = list(range(test_base, test_base + args.test_episodes))

    train_split = collect_split(args.config, train_seeds)
    val_split = collect_split(args.config, val_seeds)
    test_split = collect_split(args.config, test_seeds)

    model, history = train_probe(
        train_split,
        val_split,
        cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    save_history(os.path.join(args.out_dir, 'train_history.csv'), history)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'probe_best.pt'))

    train_scores = predict_scores(model, train_split, args.batch_size, device)
    val_scores = predict_scores(model, val_split, args.batch_size, device)
    test_scores = predict_scores(model, test_split, args.batch_size, device)

    summary = {
        'config': os.path.abspath(args.config),
        'device': str(device),
        'episodes': {
            'train': args.train_episodes,
            'val': args.val_episodes,
            'test': args.test_episodes,
        },
        'seed_base': args.seed_base,
        'train_samples': int(train_split.own.shape[0]),
        'val_samples': int(val_split.own.shape[0]),
        'test_samples': int(test_split.own.shape[0]),
        'metrics': {
            'train': evaluate_scores(train_scores, train_split, cfg),
            'val': evaluate_scores(val_scores, val_split, cfg),
            'test': evaluate_scores(test_scores, test_split, cfg),
        },
        'best_epoch': int(max(history, key=lambda x: (x['val_exact_top2_match_rate'], -x['val_mean_regret'], x['val_top1_accuracy']))['epoch']) if history else 0,
    }

    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
