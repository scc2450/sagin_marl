from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_eval import _as_driver_list
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO
from sagin_marl.rl.structured_parallel_eval import looks_like_driver_group, reset_many
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_state_dict_forgiving


def _font_properties() -> fm.FontProperties:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            return fm.FontProperties(fname=str(path))
    return fm.FontProperties()


def _actor_state_from_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint {path} is not a dictionary")
    actor_state = ckpt.get("actor")
    if isinstance(actor_state, dict):
        return dict(actor_state)
    if all(isinstance(key, str) for key in ckpt):
        return dict(ckpt)
    raise KeyError(f"checkpoint {path} does not contain an actor state_dict")


def _actor_state_from_stage_checkpoints(
    *,
    accel_checkpoint: Path,
    sat_checkpoint: Path,
    bw_checkpoint: Path,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for prefix, path in [
        ("accel_policy.", accel_checkpoint),
        ("sat_subset_policy.", sat_checkpoint),
        ("bw_policy.", bw_checkpoint),
    ]:
        state = _actor_state_from_checkpoint(path)
        bad_keys = [key for key in state if not str(key).startswith(prefix)]
        if bad_keys:
            raise ValueError(f"{path} contains keys outside prefix {prefix!r}: {bad_keys[:5]}")
        overlap = set(merged).intersection(state)
        if overlap:
            raise ValueError(f"stage checkpoints contain overlapping keys: {sorted(overlap)[:5]}")
        merged.update({key: value.detach().clone() for key, value in state.items()})
    return merged


def _xy_from_state(state: dict[str, Any], key: str, rows: int) -> np.ndarray:
    value = state.get(key)
    if value is None:
        return np.zeros((rows, 2), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 2 and arr.shape[0] == rows and arr.shape[1] >= 2:
        return arr[:, :2].copy()
    out = np.zeros((rows, 2), dtype=np.float32)
    flat = arr.reshape(-1)
    out.reshape(-1)[: min(out.size, flat.size)] = flat[: min(out.size, flat.size)]
    return out


def _done_from_step_result(step_result: Any) -> bool:
    terminated = getattr(step_result, "terminated", None)
    truncated = getattr(step_result, "truncated", None)
    if torch.is_tensor(terminated) and torch.is_tensor(truncated):
        done = terminated.reshape(-1)[0] | truncated.reshape(-1)[0]
        return bool(done.detach().cpu().item())
    if terminated is not None and truncated is not None:
        return bool(terminated or truncated)
    return False


def _min_pairwise_distance(path: np.ndarray) -> float:
    if path.shape[1] < 2:
        return float("nan")
    min_dist = float("inf")
    for positions in path:
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist[dist <= 0.0] = np.inf
        min_dist = min(min_dist, float(np.min(dist)))
    return min_dist


def _reshape_history_tensor(tensor: torch.Tensor, *, steps: int, slots: int) -> np.ndarray:
    arr = tensor[: steps * slots].detach().cpu().numpy()
    return arr.reshape((steps, slots) + tuple(arr.shape[1:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--accel_checkpoint", required=True)
    parser.add_argument("--sat_checkpoint", required=True)
    parser.add_argument("--bw_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episode_seed", type=int, default=903043)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--access_bw_decision_interval", type=int, default=5)
    parser.add_argument("--sat_decision_interval", type=int, default=1)
    args = parser.parse_args()

    font_prop = _font_properties()
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    cfg = load_config(args.config)
    cfg.access_bw_decision_interval = max(int(args.access_bw_decision_interval), 1)
    cfg.sat_decision_interval = max(int(args.sat_decision_interval), 1)
    cfg.structured_native_history_snapshots_enabled = True
    map_size = float(getattr(cfg, "map_size", 1000.0) or 1000.0)

    requested = str(args.device).lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    else:
        device = torch.device(requested)

    bundle = build_structured_modules_from_config(cfg)
    actor_state = _actor_state_from_stage_checkpoints(
        accel_checkpoint=Path(args.accel_checkpoint),
        sat_checkpoint=Path(args.sat_checkpoint),
        bw_checkpoint=Path(args.bw_checkpoint),
    )
    info = load_state_dict_forgiving(bundle.actor, actor_state, strict=True)
    if info.get("missing_keys") or info.get("unexpected_keys"):
        raise RuntimeError(f"checkpoint load mismatch: {info}")
    actor = bundle.actor.to(device).eval()

    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(getattr(cfg, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95) or 0.95),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=True,
        train_sat=True,
        train_bw=True,
        exec_accel_source="policy",
        exec_sat_source="policy",
        exec_bw_source="policy",
    )

    env_group = None
    done = False
    try:
        env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
        drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
        learner.bind_native_runtime_contract(env_group)
        reset_many(drivers, [int(args.episode_seed)])
        num_uav = int(getattr(cfg, "num_uav", 0) or 0)
        num_gu = int(getattr(cfg, "num_gu", 0) or 0)

        initial_state = env_group.export_runtime_state_batch(indices=[0])[0]
        initial_uav_pos = _xy_from_state(initial_state, "uav_pos", rows=num_uav)
        gu_pos = _xy_from_state(initial_state, "gu_pos", rows=num_gu)

        horizon = max(int(args.max_steps), 1)
        buffer = StructuredRolloutBuffer()
        learner.begin_native_rollout(drivers, rollout_env_steps=horizon, num_envs=1)
        results = learner.collect_env_horizon_native_tensor_policy(
            drivers,
            buffer,
            horizon=horizon,
            deterministic=True,
        )
        done = any(_done_from_step_result(result) for result in results)
        runtime = getattr(drivers, "native_rollout_runtime", None)
        if runtime is None or getattr(runtime, "history", None) is None:
            raise RuntimeError("native runtime history is unavailable")
        world = runtime.history.bw_stage.world_batch
        uav_nodes = _reshape_history_tensor(world.uav_nodes, steps=horizon, slots=1)[:, 0]
        uav_pos_hist = uav_nodes[..., critic_schema.UAV_X : critic_schema.UAV_Y + 1] * map_size
        path_arr = np.concatenate([initial_uav_pos[None, :, :], uav_pos_hist], axis=0)
    finally:
        close_structured_env_group(env_group)
    colors = ["#3E6FB6", "#D98634", "#6AA84F", "#8E5EA2", "#C44E52"]

    fig, ax = plt.subplots(figsize=(5.8, 5.35))
    if gu_pos.size:
        ax.scatter(
            gu_pos[:, 0],
            gu_pos[:, 1],
            s=22,
            c="#B8B8B8",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            label="地面用户",
            zorder=1,
        )
    for uav_idx in range(path_arr.shape[1]):
        xy = path_arr[:, uav_idx, :]
        color = colors[uav_idx % len(colors)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, alpha=0.95, label=f"无人机 {uav_idx + 1}", zorder=3)
        ax.scatter(xy[0, 0], xy[0, 1], s=78, facecolors="white", edgecolors=color, linewidths=1.8, marker="o", zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=96, c=color, edgecolors="black", linewidths=0.8, marker="^", zorder=5)
    ax.scatter([], [], s=78, facecolors="white", edgecolors="#333333", linewidths=1.4, marker="o", label="初始位置")
    ax.scatter([], [], s=90, c="#666666", edgecolors="black", linewidths=0.7, marker="^", label="终止位置")

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("横向位置 / m", fontproperties=font_prop, fontsize=10)
    ax.set_ylabel("纵向位置 / m", fontproperties=font_prop, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.3)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.5)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_fontsize(8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout(pad=0.8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig4-9-typical-uav-trajectory-new-preview.pdf"
    png_path = out_dir / "fig4-9-typical-uav-trajectory-new-preview.png"
    csv_path = out_dir / "fig4-9-typical-uav-trajectory-new-preview.csv"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("step,uav,x,y\n")
        for step_idx, positions in enumerate(path_arr):
            for uav_idx, xy in enumerate(positions):
                f.write(f"{step_idx},{uav_idx},{float(xy[0])},{float(xy[1])}\n")

    min_dist = _min_pairwise_distance(path_arr)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    print(f"saved {csv_path}")
    print(f"steps={path_arr.shape[0] - 1}, done={done}, min_pairwise_distance={min_dist:.3f} m")


if __name__ == "__main__":
    main()
