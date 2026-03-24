from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy,
    centroid_accel_policy,
    queue_aware_policy,
    queue_aware_bw_policy,
    queue_aware_sat_policy,
    random_accel_policy,
    random_bw_policy,
    random_sat_policy,
    uniform_bw_policy,
    uniform_sat_policy,
    zero_accel_policy,
)
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


_UNASSOCIATED_GU_COLOR = np.array([0.65, 0.65, 0.65, 0.85], dtype=np.float32)


def _resolve_render_paths(
    run_dir: str | None,
    checkpoint: str | None,
    out: str | None,
    baseline: str = "none",
) -> tuple[str | None, str]:
    use_baseline = baseline != "none"
    if run_dir:
        if not use_baseline:
            checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        if out is None:
            filename = "episode.gif" if not use_baseline else f"episode_{baseline}.gif"
            out = os.path.join(run_dir, filename)
    else:
        if not use_baseline:
            checkpoint = checkpoint or "runs/phase1/actor.pt"
        if out is None:
            filename = "episode.gif" if not use_baseline else f"episode_{baseline}.gif"
            out = os.path.join("runs/phase1", filename)
    return checkpoint, out


def _avoid_overwrite_path(path: str, overwrite: bool) -> tuple[str, bool]:
    if overwrite or not os.path.exists(path):
        return path, False

    root, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate, True
        idx += 1


def _baseline_actions(
    baseline: str,
    obs_list,
    cfg,
    num_agents: int,
    env=None,
    rng: np.random.Generator | None = None,
):
    if baseline in ("zero_accel", "fixed"):
        return zero_accel_policy(num_agents), None, None
    if baseline == "random_accel":
        return random_accel_policy(num_agents, rng=rng), None, None
    if baseline == "cluster_center":
        centers = None if env is None else getattr(env, "gu_cluster_centers", None)
        counts = None if env is None else getattr(env, "gu_cluster_counts", None)
        return cluster_center_accel_policy(obs_list, cfg, centers, counts), None, None
    if baseline == "centroid":
        gain = float(getattr(cfg, "baseline_centroid_gain", 2.0))
        queue_weighted = bool(getattr(cfg, "baseline_centroid_queue_weighted", True))
        return centroid_accel_policy(obs_list, gain=gain, queue_weighted=queue_weighted), None, None
    if baseline == "uniform_bw":
        return zero_accel_policy(num_agents), uniform_bw_policy(num_agents, cfg.users_obs_max), None
    if baseline == "random_bw":
        return zero_accel_policy(num_agents), random_bw_policy(num_agents, cfg, rng=rng), None
    if baseline == "queue_aware_bw":
        return zero_accel_policy(num_agents), queue_aware_bw_policy(obs_list, cfg), None
    if baseline == "uniform_sat":
        return zero_accel_policy(num_agents), None, uniform_sat_policy(num_agents, cfg.sats_obs_max)
    if baseline == "random_sat":
        return zero_accel_policy(num_agents), None, random_sat_policy(num_agents, cfg, rng=rng)
    if baseline == "queue_aware_sat":
        return zero_accel_policy(num_agents), None, queue_aware_sat_policy(obs_list, cfg)
    if baseline == "queue_aware":
        return queue_aware_policy(obs_list, cfg)
    raise ValueError(f"Unknown baseline: {baseline}")


def _build_uav_colors(num_uav: int) -> np.ndarray:
    import matplotlib.pyplot as plt

    if num_uav <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    if num_uav <= 10:
        cmap_name = "tab10"
    elif num_uav <= 20:
        cmap_name = "tab20"
    else:
        cmap_name = "gist_rainbow"
    cmap = plt.get_cmap(cmap_name, num_uav)
    return np.asarray([cmap(i) for i in range(num_uav)], dtype=np.float32)


def _render_colored_frame(env: SaginParallelEnv) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = env.cfg
    assoc = (
        env._associate_users()
        if cfg.num_gu > 0 and cfg.num_uav > 0
        else np.full((cfg.num_gu,), -1, dtype=np.int32)
    )
    uav_colors = _build_uav_colors(cfg.num_uav)

    fig, ax = plt.subplots(figsize=(5, 5))
    if cfg.num_gu > 0 and cfg.num_uav > 0:
        connected_mask = assoc >= 0
        for gu_idx, uav_idx in enumerate(assoc):
            if uav_idx < 0:
                continue
            gu_xy = env.gu_pos[gu_idx]
            uav_xy = env.uav_pos[uav_idx]
            ax.plot(
                [gu_xy[0], uav_xy[0]],
                [gu_xy[1], uav_xy[1]],
                color=uav_colors[uav_idx],
                linewidth=0.8,
                alpha=0.18,
                zorder=1,
            )

        for uav_idx in range(cfg.num_uav):
            gu_mask = assoc == uav_idx
            if not np.any(gu_mask):
                continue
            gu_xy = env.gu_pos[gu_mask]
            ax.scatter(
                gu_xy[:, 0],
                gu_xy[:, 1],
                s=18,
                c=[uav_colors[uav_idx]],
                alpha=0.85,
                edgecolors="none",
                zorder=2,
            )

        if np.any(~connected_mask):
            gu_xy = env.gu_pos[~connected_mask]
            ax.scatter(
                gu_xy[:, 0],
                gu_xy[:, 1],
                s=18,
                c=[_UNASSOCIATED_GU_COLOR],
                alpha=float(_UNASSOCIATED_GU_COLOR[3]),
                edgecolors="none",
                zorder=2,
            )
    elif cfg.num_gu > 0:
        ax.scatter(
            env.gu_pos[:, 0],
            env.gu_pos[:, 1],
            s=18,
            c=[_UNASSOCIATED_GU_COLOR],
            alpha=float(_UNASSOCIATED_GU_COLOR[3]),
            edgecolors="none",
            zorder=2,
        )

    legend_handles = []
    for uav_idx in range(cfg.num_uav):
        uav_xy = env.uav_pos[uav_idx]
        color = uav_colors[uav_idx]
        ax.scatter(
            [uav_xy[0]],
            [uav_xy[1]],
            s=90,
            c=[color],
            marker="^",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            f"U{uav_idx}",
            (uav_xy[0], uav_xy[1]),
            xytext=(5, 4),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                linestyle="None",
                label=f"UAV {uav_idx}",
            )
        )

    if cfg.num_gu > 0 and np.any(assoc < 0):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_UNASSOCIATED_GU_COLOR,
                markeredgecolor="none",
                markersize=5,
                linestyle="None",
                label="Unassociated GU",
            )
        )

    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    served = int(np.count_nonzero(assoc >= 0)) if cfg.num_gu > 0 else 0
    ax.set_title(f"t={env.t} | assoc={served}/{cfg.num_gu}")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgba = rgba.reshape((h, w, 4))
        buf = rgba[:, :, :3]
    except AttributeError:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        argb = argb.reshape((h, w, 4))
        buf = argb[:, :, 1:4]
    plt.close(fig)
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory that contains checkpoints and render outputs.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=[
            "none",
            "fixed",
            "zero_accel",
            "random_accel",
            "cluster_center",
            "centroid",
            "queue_aware",
            "uniform_bw",
            "random_bw",
            "queue_aware_bw",
            "uniform_sat",
            "random_sat",
            "queue_aware_sat",
        ],
        help="Render a baseline policy instead of a trained checkpoint.",
    )
    parser.add_argument(
        "--episode_seed",
        type=int,
        default=None,
        help="Reset seed for the rendered episode. Match evaluate.py --episode_seed_base + episode index to replay an eval rollout.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists. Default behavior keeps the existing file and writes to a suffixed path instead.",
    )
    args = parser.parse_args()

    args.checkpoint, args.out = _resolve_render_paths(
        args.run_dir, args.checkpoint, args.out, args.baseline
    )
    args.out, redirected = _avoid_overwrite_path(args.out, args.overwrite)
    if redirected:
        print(f"Output exists, writing to {args.out} instead of overwriting the existing file.")

    cfg = load_config(args.config)
    if args.baseline == "cluster_center":
        cfg.avoidance_enabled = True
        cfg.pairwise_hard_filter_enabled = True
    env = SaginParallelEnv(cfg)
    use_baseline = args.baseline != "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset(seed=args.episode_seed)
    actor = None
    if not use_baseline:
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        info = load_checkpoint_forgiving(actor, args.checkpoint, map_location=device, strict=True)
        if info.get("adapted_keys"):
            print(f"Loaded actor with adapted tensors from {args.checkpoint}: {len(info['adapted_keys'])}")
        actor.eval()

    frames = []
    done = False
    while not done:
        frame = _render_colored_frame(env)
        frames.append(frame)

        obs_list = list(obs.values())
        if use_baseline:
            accel_actions, bw_logits, sat_logits = _baseline_actions(
                args.baseline,
                obs_list,
                cfg,
                len(env.agents),
                env=env,
                rng=env.rng,
            )
        else:
            obs_batch = batch_flatten_obs(obs_list, cfg)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_out = actor.act(obs_tensor, deterministic=True)
            accel_actions = policy_out.accel.cpu().numpy()
            bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
            sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
        actions = assemble_actions(
            cfg,
            env.agents,
            accel_actions,
            bw_logits=bw_logits,
            sat_logits=sat_logits,
        )
        obs, rewards, terms, truncs, _ = env.step(actions)
        done = list(terms.values())[0] or list(truncs.values())[0]

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved render to {args.out}")


if __name__ == "__main__":
    main()
