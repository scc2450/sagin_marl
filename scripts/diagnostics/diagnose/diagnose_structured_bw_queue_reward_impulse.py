from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _to_device_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.rl.structured_train import as_structured_driver, as_structured_drivers, make_structured_driver, make_structured_env


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _done_from_step_result(step_result) -> bool:
    return bool(
        list(step_result.terminations.values())[0]
        or list(step_result.truncations.values())[0]
    )


def _zero_sat_action(cfg) -> np.ndarray:
    select_k = max(int(getattr(cfg, "sat_num_select", cfg.N_RF) or cfg.N_RF), 1)
    return np.full((cfg.num_uav, select_k), -1, dtype=np.int64)


def _load_cfg_and_actor(run_dir: Path, device: torch.device, checkpoint_name: str | None):
    config_path = run_dir / "config_source.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    cfg = load_config(str(config_path))
    bundle = build_structured_modules_from_config(cfg, hidden_dim=256, embed_dim=64, build_critic=False)
    checkpoint_path = run_dir / (checkpoint_name or "actor_final.pt")
    load_checkpoint_forgiving(bundle.actor, str(checkpoint_path), map_location=device, strict=True)
    actor = bundle.actor.to(device).eval()
    return cfg, actor, checkpoint_path, config_path


def _bw_action_from_actor(actor, snapshot, device: torch.device, deterministic: bool) -> np.ndarray:
    local_state = _to_device_dataclass(build_local_bw_states_from_snapshot(snapshot)[0], device)
    with torch.inference_mode():
        out = actor.act_bw(local_state, deterministic=bool(deterministic))
    return np.asarray(out.action.detach().cpu().numpy(), dtype=np.float32)


def _prepare_bw_snapshot(driver: StructuredControlDriver, cfg) -> tuple[dict[str, Any], Any, int]:
    driver.begin_step()
    accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
    _ = driver.run_accel_stage(accel_zero)
    bw_world = driver.run_sat_stage(_zero_sat_action(cfg))
    snapshot = driver.build_bw_stage_snapshot(bw_world)
    snapshot_state = driver.export_bw_stage_state()
    env_state = snapshot_state.get("env_state", {}) if isinstance(snapshot_state, dict) else {}
    t = int(env_state.get("t", 0) or 0) if isinstance(env_state, dict) else 0
    return snapshot_state, snapshot, t


def _collect_policy_bw_samples(
    *,
    cfg,
    actor,
    device: torch.device,
    num_samples: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_structured_env(cfg, mode="script")
    driver = as_structured_driver(env)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    try:
        while len(rows) < int(num_samples):
            env.reset(seed=int(seed_base) + int(episode_count))
            done = False
            while not done and len(rows) < int(num_samples):
                snapshot_state, snapshot, t = _prepare_bw_snapshot(driver, cfg)
                sampled_action = _bw_action_from_actor(actor, snapshot, device, deterministic=False)
                det_action = _bw_action_from_actor(actor, snapshot, device, deterministic=True)
                rows.append(
                    {
                        "sample_index": int(len(rows)),
                        "episode": int(episode_count),
                        "t": int(t),
                        "snapshot_state": snapshot_state,
                        "sampled_action": np.asarray(sampled_action, dtype=np.float32).tolist(),
                        "det_action": np.asarray(det_action, dtype=np.float32).tolist(),
                    }
                )
                step_result = driver.execute_stage_bw_and_step(np.asarray(sampled_action, dtype=np.float32))
                done = _done_from_step_result(step_result)
            episode_count += 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    return rows, int(episode_count)


def _timeline_step_record(env, k: int, step_result) -> dict[str, Any]:
    reward = float(next(iter(step_result.rewards.values())))
    return {
        "k": int(k),
        "reward": float(reward),
        "gu_queue": np.asarray(env.gu_queue, dtype=np.float32).tolist(),
        "last_arrival": np.asarray(getattr(env, "last_gu_arrival", np.zeros((env.cfg.num_gu,), dtype=np.float32)), dtype=np.float32).tolist(),
        "last_outflow": np.asarray(getattr(env, "last_gu_outflow", np.zeros((env.cfg.num_gu,), dtype=np.float32)), dtype=np.float32).tolist(),
    }


def _rollout_timeline_with_driver(
    *,
    probe_driver: StructuredControlDriver,
    snapshot_state: dict[str, Any],
    first_action: np.ndarray,
    cfg,
    actor,
    device: torch.device,
    max_k: int,
    follow_deterministic: bool,
) -> list[dict[str, Any]]:
    probe_driver.load_bw_stage_state(snapshot_state)
    action = np.asarray(first_action, dtype=np.float32)
    timeline: list[dict[str, Any]] = []
    for step_idx in range(int(max_k)):
        step_result, _next_world = probe_driver.execute_stage_bw_and_prepare_next_accel(action)
        timeline.append(_timeline_step_record(probe_driver.env, step_idx + 1, step_result))
        if _done_from_step_result(step_result) or step_idx + 1 >= int(max_k):
            break
        probe_driver.begin_step()
        accel_zero = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        _ = probe_driver.run_accel_stage(accel_zero)
        bw_world = probe_driver.run_sat_stage(_zero_sat_action(cfg))
        snapshot = probe_driver.build_bw_stage_snapshot(bw_world)
        action = _bw_action_from_actor(actor, snapshot, device, deterministic=bool(follow_deterministic))
    return timeline


def _discounted_prefix(values: list[float], gamma: float) -> list[float]:
    total = 0.0
    discount = 1.0
    out: list[float] = []
    for value in values:
        total += discount * float(value)
        out.append(float(total))
        discount *= float(gamma)
    return out


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _build_summary(rows: list[dict[str, Any]], *, max_k: int, gamma: float) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for k in range(1, int(max_k) + 1):
        eligible = [row for row in rows if len(row["queue_l1_by_k"]) >= int(k)]
        queue_vals = [float(row["queue_l1_by_k"][k - 1]) for row in eligible]
        reward_deltas = [float(row["reward_delta_by_k"][k - 1]) for row in eligible]
        cum_deltas = [
            float(_discounted_prefix(row["reward_delta_by_k"], gamma)[k - 1])
            for row in eligible
        ]
        summary.append(
            {
                "k": int(k),
                "count": int(len(eligible)),
                "mean_queue_l1": float(np.mean(np.asarray(queue_vals, dtype=np.float64))) if queue_vals else 0.0,
                "median_queue_l1": _safe_percentile(queue_vals, 50.0),
                "p90_queue_l1": _safe_percentile(queue_vals, 90.0),
                "frac_queue_l1_gt_1e5": float(np.mean((np.asarray(queue_vals, dtype=np.float64) > 1.0e5).astype(np.float64))) if queue_vals else 0.0,
                "frac_queue_l1_gt_5e5": float(np.mean((np.asarray(queue_vals, dtype=np.float64) > 5.0e5).astype(np.float64))) if queue_vals else 0.0,
                "mean_reward_delta": float(np.mean(np.asarray(reward_deltas, dtype=np.float64))) if reward_deltas else 0.0,
                "mean_abs_reward_delta": float(np.mean(np.abs(np.asarray(reward_deltas, dtype=np.float64)))) if reward_deltas else 0.0,
                "mean_discounted_cum_delta": float(np.mean(np.asarray(cum_deltas, dtype=np.float64))) if cum_deltas else 0.0,
                "mean_abs_discounted_cum_delta": float(np.mean(np.abs(np.asarray(cum_deltas, dtype=np.float64)))) if cum_deltas else 0.0,
            }
        )
    return summary


def _pick_representative_indices(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    queue_mass = [float(np.sum(np.asarray(row["queue_l1_by_k"], dtype=np.float64))) for row in rows]
    zero_like = int(np.argmin(np.asarray(queue_mass, dtype=np.float64)))

    fade_candidates = [
        idx
        for idx, row in enumerate(rows)
        if row["queue_l1_by_k"]
        and float(row["queue_l1_by_k"][0]) > 0.0
        and any(float(value) <= 1.0e-6 for value in row["queue_l1_by_k"][1:3])
    ]
    if fade_candidates:
        fade_like = max(
            fade_candidates,
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][0]),
        )
    else:
        fade_like = max(
            range(len(rows)),
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][0]) if rows[idx]["queue_l1_by_k"] else 0.0,
        )

    tail_candidates = [
        idx
        for idx, row in enumerate(rows)
        if len(row["queue_l1_by_k"]) >= 5 and all(float(value) > 1.0e-6 for value in row["queue_l1_by_k"][:5])
    ]
    if tail_candidates:
        tail_like = max(
            tail_candidates,
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][4]),
        )
    else:
        tail_like = max(
            range(len(rows)),
            key=lambda idx: float(rows[idx]["queue_l1_by_k"][-1]) if rows[idx]["queue_l1_by_k"] else 0.0,
        )

    chosen: list[int] = []
    for idx in (zero_like, fade_like, tail_like):
        if int(idx) not in chosen:
            chosen.append(int(idx))
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=3042)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_all_seeds(int(args.seed_base))
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg, actor, checkpoint_path, config_path = _load_cfg_and_actor(
        run_dir,
        device,
        None if args.checkpoint_name is None else str(args.checkpoint_name),
    )

    sample_rows, episode_count = _collect_policy_bw_samples(
        cfg=cfg,
        actor=actor,
        device=device,
        num_samples=int(args.num_samples),
        seed_base=int(args.seed_base),
    )

    probe_env = make_structured_env(cfg, mode="script")
    probe_driver = as_structured_driver(probe_env)
    analyzed_rows: list[dict[str, Any]] = []
    try:
        for row in sample_rows:
            snapshot_state = row["snapshot_state"]
            sampled_action = np.asarray(row["sampled_action"], dtype=np.float32)
            det_action = np.asarray(row["det_action"], dtype=np.float32)
            timeline_sampled = _rollout_timeline_with_driver(
                probe_driver=probe_driver,
                snapshot_state=snapshot_state,
                first_action=sampled_action,
                cfg=cfg,
                actor=actor,
                device=device,
                max_k=int(args.max_k),
                follow_deterministic=True,
            )
            timeline_det = _rollout_timeline_with_driver(
                probe_driver=probe_driver,
                snapshot_state=snapshot_state,
                first_action=det_action,
                cfg=cfg,
                actor=actor,
                device=device,
                max_k=int(args.max_k),
                follow_deterministic=True,
            )
            common_len = min(len(timeline_sampled), len(timeline_det))
            queue_l1_by_k: list[float] = []
            reward_delta_by_k: list[float] = []
            for step_idx in range(common_len):
                queue_sampled = np.asarray(timeline_sampled[step_idx]["gu_queue"], dtype=np.float64)
                queue_det = np.asarray(timeline_det[step_idx]["gu_queue"], dtype=np.float64)
                queue_l1_by_k.append(float(np.sum(np.abs(queue_sampled - queue_det))))
                reward_delta_by_k.append(
                    float(timeline_sampled[step_idx]["reward"]) - float(timeline_det[step_idx]["reward"])
                )
            analyzed_rows.append(
                {
                    "rel_idx": int(row["sample_index"]),
                    "sample_index": int(row["sample_index"]),
                    "episode": int(row["episode"]),
                    "t": int(row["t"]),
                    "sampled_action": row["sampled_action"],
                    "det_action": row["det_action"],
                    "timeline_sampled": timeline_sampled,
                    "timeline_det": timeline_det,
                    "queue_l1_by_k": queue_l1_by_k,
                    "reward_delta_by_k": reward_delta_by_k,
                }
            )
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()

    summary = _build_summary(analyzed_rows, max_k=int(args.max_k), gamma=float(cfg.gamma))
    representative_indices = _pick_representative_indices(analyzed_rows)
    representative_samples = [analyzed_rows[idx] for idx in representative_indices]

    payload = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "checkpoint": str(checkpoint_path),
        "seed_base": int(args.seed_base),
        "num_bw_samples": int(len(analyzed_rows)),
        "episodes_used": int(episode_count),
        "follow_policy_mode": "deterministic",
        "collection_policy_mode": "stochastic",
        "compare": "sampled_action vs current_actor_deterministic_action",
        "reward_mode": str(getattr(cfg, "reward_mode", "")),
        "gamma": float(cfg.gamma),
        "summary": summary,
        "representative_samples": representative_samples,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
