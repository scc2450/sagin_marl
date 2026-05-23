from __future__ import annotations

import argparse
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO, _normalize_exec_source
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


_UNASSOCIATED_GU_COLOR = np.array([0.65, 0.65, 0.65, 0.85], dtype=np.float32)


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    return torch.device(requested)


def _resolve_paths(
    *,
    run_dir: str | None,
    config: str | None,
    checkpoint: str | None,
    out: str | None,
) -> tuple[str, str | None, str]:
    if run_dir:
        config = config or os.path.join(run_dir, "config_source.yaml")
        checkpoint = checkpoint or os.path.join(run_dir, "actor_final.pt")
        out = out or os.path.join(run_dir, "episode_structured.gif")
    else:
        config = config or "configs/phase1.yaml"
        checkpoint = checkpoint or "runs/structured/actor_final.pt"
        out = out or "runs/structured/episode_structured.gif"
    return config, checkpoint, out


def _avoid_overwrite_path(path: str, overwrite: bool) -> tuple[str, bool]:
    if overwrite or not os.path.exists(path):
        return path, False
    root, ext = os.path.splitext(path)
    suffix = 1
    while True:
        candidate = f"{root}_{suffix}{ext}"
        if not os.path.exists(candidate):
            return candidate, True
        suffix += 1


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


def _state_array(
    state: dict[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    fill: float | int = 0,
) -> np.ndarray:
    value = state.get(name)
    if value is None:
        return np.full(shape, fill, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape).copy()
    out = np.full(shape, fill, dtype=dtype)
    flat = arr.reshape(-1)
    out.reshape(-1)[: min(out.size, flat.size)] = flat[: min(out.size, flat.size)]
    return out


def _state_xy_array(state: dict[str, Any], name: str, *, rows: int) -> np.ndarray:
    value = state.get(name)
    if value is None:
        return np.zeros((rows, 2), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 2 and int(arr.shape[0]) == int(rows) and int(arr.shape[1]) >= 2:
        return arr[:, :2].copy()
    return _state_array(state, name, shape=(rows, 2), dtype=np.float32)


def _render_state_frame(cfg: Any, state: dict[str, Any], *, queue_label_limit: int) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    num_uav = max(int(getattr(cfg, "num_uav", 0) or 0), 0)
    num_gu = max(int(getattr(cfg, "num_gu", 0) or 0), 0)
    map_size = float(getattr(cfg, "map_size", 1000.0) or 1000.0)
    uav_pos = _state_xy_array(state, "uav_pos", rows=num_uav)
    gu_pos = _state_xy_array(state, "gu_pos", rows=num_gu)
    gu_queue = _state_array(state, "gu_queue", shape=(num_gu,), dtype=np.float32)
    assoc = _state_array(state, "last_association", shape=(num_gu,), dtype=np.int32, fill=-1)
    assoc = np.where((assoc >= 0) & (assoc < max(num_uav, 1)), assoc, -1).astype(np.int32, copy=False)
    uav_colors = _build_uav_colors(num_uav)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    if num_gu > 0 and num_uav > 0:
        connected_mask = assoc >= 0
        for gu_idx, uav_idx in enumerate(assoc):
            if uav_idx < 0:
                continue
            gu_xy = gu_pos[gu_idx]
            uav_xy = uav_pos[uav_idx]
            ax.plot(
                [gu_xy[0], uav_xy[0]],
                [gu_xy[1], uav_xy[1]],
                color=uav_colors[uav_idx],
                linewidth=0.65,
                alpha=0.14,
                zorder=1,
            )

        for uav_idx in range(num_uav):
            gu_mask = assoc == uav_idx
            if not np.any(gu_mask):
                continue
            gu_xy = gu_pos[gu_mask]
            ax.scatter(
                gu_xy[:, 0],
                gu_xy[:, 1],
                s=14,
                c=[uav_colors[uav_idx]],
                alpha=0.82,
                edgecolors="none",
                zorder=2,
            )
            if num_gu <= int(queue_label_limit):
                queues = gu_queue[gu_mask]
                for idx in range(len(gu_xy)):
                    ax.annotate(
                        f"{queues[idx] / 1.0e6:.1f}M",
                        (gu_xy[idx, 0], gu_xy[idx, 1]),
                        xytext=(5, 4),
                        textcoords="offset points",
                        color=uav_colors[uav_idx],
                        fontsize=7,
                        fontweight="bold",
                        zorder=4,
                    )
        if np.any(~connected_mask):
            gu_xy = gu_pos[~connected_mask]
            ax.scatter(
                gu_xy[:, 0],
                gu_xy[:, 1],
                s=14,
                c=[_UNASSOCIATED_GU_COLOR],
                alpha=float(_UNASSOCIATED_GU_COLOR[3]),
                edgecolors="none",
                zorder=2,
            )
    elif num_gu > 0:
        ax.scatter(
            gu_pos[:, 0],
            gu_pos[:, 1],
            s=14,
            c=[_UNASSOCIATED_GU_COLOR],
            alpha=float(_UNASSOCIATED_GU_COLOR[3]),
            edgecolors="none",
            zorder=2,
        )

    legend_handles = []
    for uav_idx in range(num_uav):
        uav_xy = uav_pos[uav_idx]
        color = uav_colors[uav_idx]
        ax.scatter(
            [uav_xy[0]],
            [uav_xy[1]],
            s=92,
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
        if num_uav <= 16:
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

    if num_gu > 0 and np.any(assoc < 0):
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

    step_t = int(state.get("t", 0) or 0)
    served = int(np.count_nonzero(assoc >= 0)) if num_gu > 0 else 0
    queue_mbits = float(np.sum(gu_queue, dtype=np.float64) / 1.0e6)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"t={step_t} | assoc={served}/{num_gu} | GU queue={queue_mbits:.1f}M")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((height, width, 4))
        frame = rgba[:, :, :3].copy()
    except AttributeError:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))
        frame = argb[:, :, 1:4].copy()
    plt.close(fig)
    return frame


def _source_or_cfg(raw: str | None, cfg: Any, field: str) -> str:
    return _normalize_exec_source(raw if raw is not None else getattr(cfg, field, "policy"))


def _load_actor_checkpoint(actor: torch.nn.Module, path: str, *, device: torch.device, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    info = load_checkpoint_forgiving(actor, path, map_location=device, strict=False)
    if info.get("adapted_keys"):
        print(f"Loaded {label} with adapted tensors from {path}: {len(info['adapted_keys'])}")


def _done_from_step_result(step_result: Any) -> bool:
    terminated = getattr(step_result, "terminated", None)
    truncated = getattr(step_result, "truncated", None)
    if torch.is_tensor(terminated) and torch.is_tensor(truncated):
        done = terminated.reshape(-1)[0] | truncated.reshape(-1)[0]
        return bool(done.detach().cpu().item())
    if terminated is not None and truncated is not None:
        return bool(terminated or truncated)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--episode_seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--queue_label_limit", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--policy_mode", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--structured_env_tensor_backend",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--exec_accel_source", type=str, default=None)
    parser.add_argument("--exec_sat_source", type=str, default=None)
    parser.add_argument("--exec_bw_source", type=str, default=None)
    args = parser.parse_args()

    config_path, checkpoint_path, out_path = _resolve_paths(
        run_dir=args.run_dir,
        config=args.config,
        checkpoint=args.checkpoint,
        out=args.out,
    )
    out_path, redirected = _avoid_overwrite_path(out_path, args.overwrite)
    if redirected:
        print(f"Output exists, writing to {out_path} instead of overwriting the existing file.")

    cfg = load_config(config_path)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)
    device = _resolve_torch_device(args.device)
    accel_source = _source_or_cfg(args.exec_accel_source, cfg, "exec_accel_source")
    sat_source = _source_or_cfg(args.exec_sat_source, cfg, "exec_sat_source")
    bw_source = _source_or_cfg(args.exec_bw_source, cfg, "exec_bw_source")
    sources = (accel_source, sat_source, bw_source)

    bundle = build_structured_modules_from_config(cfg, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim)
    actor = bundle.actor.to(device)
    if any(source == "policy" for source in sources):
        _load_actor_checkpoint(actor, checkpoint_path, device=device, label="actor")
    actor.eval()

    teacher_actor = None
    if any(source == "teacher" for source in sources):
        teacher_path = str(getattr(cfg, "exec_teacher_actor_path", "") or checkpoint_path or "")
        teacher_bundle = build_structured_modules_from_config(cfg, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim)
        teacher_actor = teacher_bundle.actor.to(device)
        _load_actor_checkpoint(teacher_actor, teacher_path, device=device, label="teacher actor")
        teacher_actor.eval()

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
        train_accel=accel_source == "policy",
        train_sat=sat_source == "policy",
        train_bw=bw_source == "policy",
        exec_accel_source=accel_source,
        exec_sat_source=sat_source,
        exec_bw_source=bw_source,
        teacher_actor=teacher_actor,
    )

    frames: list[np.ndarray] = []
    env_group = None
    try:
        env_group = make_structured_env_group(cfg, num_envs=1, backend=args.vec_backend, mode="eval")
        learner.bind_native_runtime_contract(env_group)
        env_group.reset_many([args.episode_seed])
        horizon = int(args.max_steps) if args.max_steps is not None else int(getattr(cfg, "T_steps", 0) or 0)
        horizon = max(horizon, 1)
        frame_stride = max(int(args.frame_stride), 1)
        for step_idx in range(horizon):
            if step_idx % frame_stride == 0:
                state = env_group.export_runtime_state_batch(indices=[0])[0]
                frames.append(
                    _render_state_frame(
                        cfg,
                        state,
                        queue_label_limit=max(int(args.queue_label_limit), 0),
                    )
                )
            step_result = learner.collect_env_steps(
                env_group,
                None,
                deterministic=args.policy_mode != "stochastic",
            )
            if _done_from_step_result(step_result):
                break
        if not frames:
            state = env_group.export_runtime_state_batch(indices=[0])[0]
            frames.append(_render_state_frame(cfg, state, queue_label_limit=max(int(args.queue_label_limit), 0)))
    finally:
        close_structured_env_group(env_group)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    imageio.mimsave(out_path, frames, fps=max(int(args.fps), 1))
    print(f"Saved structured render to {out_path}")


if __name__ == "__main__":
    main()
