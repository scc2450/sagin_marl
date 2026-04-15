from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    cluster_center_accel_policy,
    cluster_center_queue_aware_policy,
    centroid_accel_policy,
    lyapunov_queue_aware_policy_step,
    queue_aware_policy,
    random_accel_policy,
    zero_accel_policy,
    uniform_bw_policy,
    random_bw_policy,
    queue_aware_bw_policy,
    uniform_sat_policy,
    random_sat_policy,
    queue_aware_sat_policy,
)
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.utils.progress import Progress


def _init_eval_tb_layout(writer: SummaryWriter, tag_prefix: str) -> None:
    def t(name: str) -> str:
        return f"{tag_prefix}/{name}"

    layout = {
        "Eval/Main": {
            "RewardSum": ["Multiline", [t("reward_sum")]],
            "Throughput": [
                "Multiline",
                [
                    t("throughput_access_norm"),
                    t("throughput_backhaul_norm"),
                ],
            ],
            "QueueMean": ["Multiline", [t("gu_queue_mean"), t("uav_queue_mean")]],
            "QueueTail": [
                "Multiline",
                [
                    t("gu_queue_arrival_steps_p95"),
                    t("uav_queue_arrival_steps_p95"),
                ],
            ],
            "Safety": [
                "Multiline",
                [
                    t("terminated_early"),
                    t("collision"),
                ],
            ],
        },
        "Eval/Drift": {
            "QueueDrift": [
                "Multiline",
                [
                    t("gu_queue_drift_ratio"),
                    t("uav_queue_drift_ratio"),
                    t("sat_processed_norm"),
                ],
            ],
        },
    }

    writer.add_custom_scalars(layout)


def _build_lyapunov_env_callbacks(env, cfg) -> dict:
    """Build environment callbacks for topology-aware DPP baseline.
    
    These callbacks allow the DPP baseline to recompute link qualities and 
    visibility after simulating acceleration effects on UAV position.
    """
    if env is None:
        return {}
    
    def compute_access_rates(agent_id: int, accel_vec: np.ndarray, obs: dict, rel_next: np.ndarray):
        """
        Compute GU-UAV link qualities and rates after acceleration.
        
        Returns:
        - eta: (K,) spectral efficiency per GU
        - rates: (K,) achievable rate per GU
        """
        try:
            # Simulate new position after acceleration
            own = np.asarray(obs["own"], dtype=np.float32)
            rel_original = obs["users"][:cfg.users_obs_max, :2]
            original_eta = obs["users"][:cfg.users_obs_max, 3]
            vel_abs = own[2:4] * cfg.v_max
            delta_pos_abs = vel_abs * cfg.tau0 + 0.5 * np.asarray(accel_vec, dtype=np.float32) * cfg.a_max * (cfg.tau0**2)
            
            H = getattr(cfg, "uav_height", 100.0) # 无人机高度
            d2_old = np.sum(rel_original**2, axis=1) + H**2
            d2_new = np.sum(rel_next**2, axis=1) + H**2
            # Get current GU association to compute rates
            users = obs["users"]
            candidates = []
            for g in range(cfg.num_gu):
                if g < len(users) and users[g, 4] == agent_id:  # user[g].assoc == agent_id
                    candidates.append(g)
            
            if not candidates:
                candidates = list(range(min(cfg.num_gu, cfg.users_obs_max)))
            
            snr_old = 2.0**(original_eta) - 1.0
            snr_new = (d2_old / np.maximum(d2_new, 1e-9)) * np.maximum(snr_old, 0.0)
            
            eta = np.log2(1.0 + snr_new)
            rates = 0.5 * eta
            return eta, rates
        except:
            # Fallback: return obs values
            return np.clip(obs["users"][:cfg.users_obs_max, 3], 0.0, 1.0), None
    
    def check_sat_visibility(agent_id: int, accel_vec: np.ndarray, obs: dict):
        """
        Check which satellites remain visible after acceleration.
        
        Returns:
        - valid_sat_mask: (L,) boolean mask of visible satellites
        """
        try:
            # Placeholder: return current visibility mask
            sat_mask = obs.get("sats_mask", np.ones((cfg.sats_obs_max,), dtype=bool))
            return sat_mask > 0.5
        except:
            return np.ones((cfg.sats_obs_max,), dtype=bool)
    
    return {
        "compute_access_rates": compute_access_rates,
        "check_sat_visibility": check_sat_visibility,
    }


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
    if baseline == "cluster_center_queue_aware":
        centers = None if env is None else getattr(env, "gu_cluster_centers", None)
        counts = None if env is None else getattr(env, "gu_cluster_counts", None)
        return cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
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


def _normalize_exec_source(raw: str | None) -> str:
    src = str("policy" if raw is None else raw).strip().lower()
    if src == "heuristic_residual":
        src = "policy"
    allowed = {"policy", "teacher", "heuristic", "zero"}
    if src not in allowed:
        raise ValueError(f"Invalid exec source '{raw}'. Allowed: {sorted(allowed)}")
    return src


def _select_exec_values(
    source: str,
    policy_values: np.ndarray | None,
    teacher_values: np.ndarray | None,
    heuristic_values: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if source == "policy" and policy_values is not None:
        return np.asarray(policy_values, dtype=np.float32)
    if source == "teacher" and teacher_values is not None:
        return np.asarray(teacher_values, dtype=np.float32)
    if source == "heuristic" and heuristic_values is not None:
        return np.asarray(heuristic_values, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _source_needs_heuristic(source: str) -> bool:
    return source == "heuristic"


def _compose_bw_exec_values(
    source: str,
    policy_values: np.ndarray | None,
    teacher_values: np.ndarray | None,
    heuristic_values: np.ndarray | None,
    shape: tuple[int, int],
    cfg,
) -> np.ndarray:
    del cfg
    return _select_exec_values(source, policy_values, teacher_values, heuristic_values, shape)


def _hybrid_bw_sat_actions(
    mode: str,
    obs_list,
    cfg,
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "none":
        return None, None
    if mode in {"queue_aware", "queue_aware_bw"}:
        _, bw_logits, sat_logits = queue_aware_policy(obs_list, cfg)
        return bw_logits, sat_logits
    if mode == "uniform_bw":
        return uniform_bw_policy(num_agents, cfg.users_obs_max), None
    if mode == "random_bw":
        return random_bw_policy(num_agents, cfg, rng=rng), None
    if mode == "uniform_sat":
        return None, uniform_sat_policy(num_agents, cfg.sats_obs_max)
    if mode == "random_sat":
        return None, random_sat_policy(num_agents, cfg, rng=rng)
    if mode == "queue_aware_sat":
        return None, queue_aware_sat_policy(obs_list, cfg)
    raise ValueError(f"Unknown hybrid_bw_sat mode: {mode}")


def _resolve_eval_paths(
    run_dir: str | None,
    checkpoint: str | None,
    out: str | None,
    tb_dir: str | None,
    baseline: str,
) -> tuple[str, str, str]:
    use_baseline = baseline != "none"
    if run_dir:
        checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        if out is None:
            filename = "eval_baseline.csv" if use_baseline else "eval_trained.csv"
            out = os.path.join(run_dir, filename)
    else:
        checkpoint = checkpoint or "runs/phase1/actor.pt"
        out = out or "runs/phase1/eval_trained.csv"

    out_dir = os.path.dirname(out) or "."
    tb_dir = tb_dir or os.path.join(out_dir, "eval_tb")
    return checkpoint, out, tb_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--checkpoint_strict",
        action="store_true",
        help="Require exact key match when loading the actor checkpoint.",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="How to sample the trained policy when baseline=none.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory that contains checkpoints and evaluation outputs.",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default=None,
        help="TensorBoard log dir for evaluation. Default: <out_dir>/eval_tb",
    )
    parser.add_argument(
        "--tb_tag",
        type=str,
        default=None,
        help="TensorBoard tag prefix. Default: eval/trained or eval/baseline",
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
            "cluster_center_queue_aware",
            "centroid", 
            "queue_aware", 
            "lyapunov",
            "uniform_bw", 
            "random_bw", 
            "queue_aware_bw", 
            "uniform_sat",
            "random_sat",
            "queue_aware_sat"
        ],
        help="Use a baseline policy instead of a trained model.",
    )
    parser.add_argument(
        "--hybrid_bw_sat",
        type=str,
        default="none",
        choices=[
            "none",
            "queue_aware",
            "uniform_bw",
            "random_bw",
            "queue_aware_bw",
            "uniform_sat",
            "random_sat",
            "queue_aware_sat",
        ],
        help="Override bw/sat while keeping trained-model evaluation for the other action heads.",
    )
    parser.add_argument(
        "--episode_seed_base",
        type=int,
        default=None,
        help="If set, episode e resets with seed=episode_seed_base+e for reproducible cross-policy comparisons.",
    )
    args = parser.parse_args()

    args.checkpoint, args.out, args.tb_dir = _resolve_eval_paths(
        args.run_dir, args.checkpoint, args.out, args.tb_dir, args.baseline
    )

    cfg = load_config(args.config)
    if args.baseline in {"cluster_center", "cluster_center_queue_aware"}:
        cfg.avoidance_enabled = True
        cfg.pairwise_hard_filter_enabled = True
    env = SaginParallelEnv(cfg)

    use_baseline = args.baseline != "none"
    use_hybrid = args.hybrid_bw_sat != "none"
    if use_hybrid and use_baseline:
        raise ValueError("Hybrid bw/sat is only valid when evaluating a trained model (baseline=none).")
    out_dir = os.path.dirname(args.out) or "."
    tb_dir = args.tb_dir or os.path.join(out_dir, "eval_tb")
    tb_tag = args.tb_tag or ("eval/baseline" if use_baseline else "eval/trained")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)
    _init_eval_tb_layout(tb_writer, tb_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_deterministic = args.policy_mode != "stochastic"
    actor = None
    teacher_actor = None
    exec_accel_source = _normalize_exec_source(getattr(cfg, "exec_accel_source", "policy"))
    exec_bw_source = _normalize_exec_source(getattr(cfg, "exec_bw_source", "policy"))
    exec_sat_source = _normalize_exec_source(getattr(cfg, "exec_sat_source", "policy"))
    teacher_deterministic = bool(getattr(cfg, "exec_teacher_deterministic", True))
    need_teacher_exec = "teacher" in {exec_accel_source, exec_bw_source, exec_sat_source}
    need_heuristic_exec = any(
        _source_needs_heuristic(src) for src in (exec_accel_source, exec_bw_source, exec_sat_source)
    )
    if not use_baseline:
        obs, _ = env.reset()
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        info = load_checkpoint_forgiving(actor, args.checkpoint, map_location=device, strict=args.checkpoint_strict)
        if info.get("adapted_keys"):
            print(f"Loaded actor with adapted tensors from {args.checkpoint}: {len(info['adapted_keys'])}")
        actor.eval()
        if need_teacher_exec:
            teacher_path = getattr(cfg, "exec_teacher_actor_path", None) or args.checkpoint
            teacher_actor = ActorNet(obs_dim, cfg).to(device)
            info = load_checkpoint_forgiving(teacher_actor, teacher_path, map_location=device)
            if info.get("adapted_keys"):
                print(f"Loaded teacher actor with adapted tensors from {teacher_path}: {len(info['adapted_keys'])}")
            teacher_actor.eval()

    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "episode",
        "reward_sum",
        "steps",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "sat_processed_norm",
        "gu_queue_mean",
        "uav_queue_mean",
        "gu_queue_arrival_steps_p95",
        "uav_queue_arrival_steps_p95",
        "terminated_early",
        "collision",
        "gu_queue_drift_ratio",
        "uav_queue_drift_ratio",
    ]
    tb_fieldnames = [
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "x_acc_mean",
        "x_rel_mean",
        "throughput_access_norm",
        "throughput_backhaul_norm",
        "gu_queue_mean",
        "uav_queue_mean",
        "gu_queue_arrival_steps_p95",
        "uav_queue_arrival_steps_p95",
        "terminated_early",
        "collision",
        "gu_queue_drift_ratio",
        "uav_queue_drift_ratio",
        "sat_processed_norm",
    ]
    tb_field_set = set(tb_fieldnames)
    q_zero_eps = 1e-9
    layer_total_capacity = {
        "gu": float(cfg.num_gu) * float(cfg.queue_max_gu),
        "uav": float(cfg.num_uav) * float(cfg.queue_max_uav),
        "sat": float(cfg.num_sat) * float(cfg.queue_max_sat),
    }
    progress = Progress(args.episodes, desc="Eval")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for ep in range(args.episodes):
            ep_seed = None
            if args.episode_seed_base is not None:
                ep_seed = int(args.episode_seed_base) + ep
            obs, _ = env.reset(seed=ep_seed)
            done = False
            reward_sum = 0.0
            steps = 0
            layer_queue_sum_steps = {"gu": [], "uav": [], "sat": []}
            layer_zero_steps = {"gu": 0.0, "uav": 0.0, "sat": 0.0}
            layer_queue_start_sum = {
                "gu": float(np.sum(env.gu_queue)),
                "uav": float(np.sum(env.uav_queue)),
                "sat": float(np.sum(env.sat_queue)),
            }
            q_active_zero_steps = 0.0
            all_layers_nonzero_steps = 0.0
            gu_queue_sum = 0.0
            uav_queue_sum = 0.0
            sat_queue_sum = 0.0
            gu_queue_max = 0.0
            uav_queue_max = 0.0
            sat_queue_max = 0.0
            gu_drop_sum = 0.0
            uav_drop_sum = 0.0
            sat_drop_sum = 0.0
            sat_processed_sum = 0.0
            sat_incoming_sum = 0.0
            connected_sat_count_sum = 0.0
            connected_sat_dist_mean_sum = 0.0
            connected_sat_dist_p95_sum = 0.0
            connected_sat_elevation_deg_mean_sum = 0.0
            connected_sat_elevation_deg_min_episode = float("inf")
            energy_mean_sum = 0.0
            assoc_ratio_sum = 0.0
            assoc_dist_sum = 0.0
            uav_assoc_gu_count_steps = [[] for _ in range(cfg.num_uav)]
            uav_access_outflow_steps = [[] for _ in range(cfg.num_uav)]
            uav_access_outflow_sum = np.zeros((cfg.num_uav,), dtype=np.float64)
            reward_raw_sum = 0.0
            service_norm_sum = 0.0
            throughput_access_norm_sum = 0.0
            throughput_backhaul_norm_sum = 0.0
            sat_processed_norm_sum = 0.0
            processed_ratio_eval_sum = 0.0
            drop_ratio_eval_sum = 0.0
            pre_backlog_steps_eval_sum = 0.0
            x_acc_sum = 0.0
            x_rel_sum = 0.0
            g_pre_sum = 0.0
            d_pre_sum = 0.0
            drop_norm_sum = 0.0
            queue_total_active_sum = 0.0
            queue_total_active_steps: list[float] = []
            arrival_sum_ep = 0.0
            outflow_sum_ep = 0.0
            active_drop_sum_ep = 0.0
            drop_sum_ep = 0.0
            drop_ratio_step_sum = 0.0
            drop_event_steps = 0.0
            term_drop_sum = 0.0
            term_drop_gu_sum = 0.0
            term_drop_uav_sum = 0.0
            term_drop_sat_sum = 0.0
            term_drop_step_sum = 0.0
            collision_any = 0.0
            min_inter_uav_dist = float("inf")
            near_collision_steps = 0.0
            danger_imitation_active_rate_sum = 0.0
            intervention_norm_sum = 0.0
            intervention_rate_sum = 0.0
            intervention_norm_top1_sum = 0.0
            close_risk_sum = 0.0
            term_close_risk_sum = 0.0
            centroid_dist_sum = 0.0
            ep_start = time.perf_counter()
            baseline_state = None
            while not done:
                if use_baseline:
                    obs_list = list(obs.values())
                    if args.baseline == "lyapunov":
                        # Build environment callbacks for topology-aware DPP baseline
                        env_callbacks = _build_lyapunov_env_callbacks(env, cfg) if env is not None else None
                        accel_actions, bw_logits, sat_logits, baseline_state = lyapunov_queue_aware_policy_step(
                            obs_list,
                            cfg,
                            state=baseline_state,
                            env_callbacks=env_callbacks,
                        )
                    else:
                        accel_actions, bw_logits, sat_logits = _baseline_actions(
                            args.baseline,
                            obs_list,
                            cfg,
                            len(env.agents),
                            env=env,
                            rng=env.rng,
                        )
                    actions = assemble_actions(
                        cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits
                    )
                else:
                    obs_list = list(obs.values())
                    obs_batch = batch_flatten_obs(list(obs.values()), cfg)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        policy_out = actor.act(obs_tensor, deterministic=policy_deterministic)
                        teacher_accel = None
                        teacher_bw = None
                        teacher_sat = None
                        if teacher_actor is not None:
                            teacher_out = teacher_actor.act(obs_tensor, deterministic=teacher_deterministic)
                            teacher_accel = teacher_out.accel.cpu().numpy()
                            teacher_bw = teacher_out.bw_logits.cpu().numpy() if teacher_out.bw_logits is not None else None
                            teacher_sat = teacher_out.sat_logits.cpu().numpy() if teacher_out.sat_logits is not None else None
                    heur_accel = None
                    heur_bw = None
                    heur_sat = None
                    if need_heuristic_exec:
                        heur_accel, heur_bw, heur_sat = queue_aware_policy(obs_list, cfg)
                    policy_accel = policy_out.accel.cpu().numpy()
                    policy_bw = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
                    policy_sat = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
                    if use_hybrid:
                        hybrid_bw, hybrid_sat = _hybrid_bw_sat_actions(
                            args.hybrid_bw_sat,
                            obs_list,
                            cfg,
                            len(env.agents),
                            rng=env.rng,
                        )
                        if hybrid_bw is not None:
                            policy_bw = hybrid_bw
                        if hybrid_sat is not None:
                            policy_sat = hybrid_sat
                    accel_actions = _select_exec_values(
                        exec_accel_source,
                        policy_accel,
                        teacher_accel,
                        heur_accel,
                        (len(env.agents), 2),
                    )
                    bw_logits = None
                    sat_logits = None
                    if cfg.enable_bw_action:
                        bw_logits = _compose_bw_exec_values(
                            exec_bw_source,
                            policy_bw,
                            teacher_bw,
                            heur_bw,
                            (len(env.agents), cfg.users_obs_max),
                            cfg,
                        )
                    if not cfg.fixed_satellite_strategy:
                        sat_logits = _select_exec_values(
                            exec_sat_source,
                            policy_sat,
                            teacher_sat,
                            heur_sat,
                            (len(env.agents), cfg.sats_obs_max),
                        )
                    actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
                obs, rewards, terms, truncs, _ = env.step(actions)
                reward_sum += list(rewards.values())[0]
                done = list(terms.values())[0] or list(truncs.values())[0]
                steps += 1
                gu_queue_sum += float(np.mean(env.gu_queue))
                uav_queue_sum += float(np.mean(env.uav_queue))
                sat_queue_sum += float(np.mean(env.sat_queue))
                gu_queue_max = max(gu_queue_max, float(np.max(env.gu_queue)))
                uav_queue_max = max(uav_queue_max, float(np.max(env.uav_queue)))
                sat_queue_max = max(sat_queue_max, float(np.max(env.sat_queue)))
                queues = {
                    "gu": env.gu_queue,
                    "uav": env.uav_queue,
                    "sat": env.sat_queue,
                }
                for layer, queue in queues.items():
                    queue_sum = float(np.sum(queue))
                    layer_queue_sum_steps[layer].append(queue_sum)
                    if queue_sum <= q_zero_eps:
                        layer_zero_steps[layer] += 1.0
                q_active_now = float(np.sum(env.gu_queue) + np.sum(env.uav_queue))
                if q_active_now <= q_zero_eps:
                    q_active_zero_steps += 1.0
                if all(float(np.sum(queue)) > q_zero_eps for queue in queues.values()):
                    all_layers_nonzero_steps += 1.0
                gu_drop_sum += float(np.sum(env.gu_drop))
                uav_drop_sum += float(np.sum(env.uav_drop))
                if hasattr(env, "sat_drop"):
                    sat_drop_sum += float(np.sum(env.sat_drop))
                if hasattr(env, "last_sat_processed"):
                    sat_processed_sum += float(np.sum(env.last_sat_processed))
                if hasattr(env, "last_sat_incoming"):
                    sat_incoming_sum += float(np.sum(env.last_sat_incoming))
                connected_sat_count_step = float(getattr(env, "last_connected_sat_count", 0.0))
                connected_sat_count_sum += connected_sat_count_step
                connected_sat_dist_mean_sum += float(getattr(env, "last_connected_sat_dist_mean", 0.0))
                connected_sat_dist_p95_sum += float(getattr(env, "last_connected_sat_dist_p95", 0.0))
                connected_sat_elevation_deg_mean_sum += float(
                    getattr(env, "last_connected_sat_elevation_deg_mean", 0.0)
                )
                if connected_sat_count_step > 0.0:
                    connected_sat_elevation_deg_min_episode = min(
                        connected_sat_elevation_deg_min_episode,
                        float(getattr(env, "last_connected_sat_elevation_deg_min", 0.0)),
                    )
                if cfg.energy_enabled:
                    energy_mean_sum += float(np.mean(env.uav_energy))
                parts = getattr(env, "last_reward_parts", None)
                if parts:
                    reward_raw_sum += float(parts.get("reward_raw", 0.0))
                    service_norm_sum += float(parts.get("service_norm", 0.0))
                    throughput_access_norm_sum += float(parts.get("throughput_access_norm", 0.0))
                    throughput_backhaul_norm_sum += float(parts.get("throughput_backhaul_norm", 0.0))
                    sat_processed_norm_sum += float(parts.get("sat_processed_norm", 0.0))
                    processed_ratio_eval_sum += float(parts.get("processed_ratio_eval", 0.0))
                    drop_ratio_eval_sum += float(parts.get("drop_ratio_eval", 0.0))
                    pre_backlog_steps_eval_sum += float(parts.get("pre_backlog_steps_eval", 0.0))
                    x_acc_sum += float(parts.get("x_acc", 0.0))
                    x_rel_sum += float(parts.get("x_rel", 0.0))
                    g_pre_sum += float(parts.get("g_pre", 0.0))
                    d_pre_sum += float(parts.get("d_pre", 0.0))
                    drop_norm_sum += float(parts.get("drop_norm", 0.0))
                    queue_total_active_step = float(parts.get("queue_total_active", 0.0))
                    queue_total_active_sum += queue_total_active_step
                    queue_total_active_steps.append(queue_total_active_step)
                    arrival_sum_ep += float(parts.get("arrival_sum", 0.0))
                    outflow_sum_ep += float(parts.get("outflow_sum", 0.0))
                    active_drop_sum_ep += float(parts.get("drop_sum_active", 0.0))
                    drop_sum_ep += float(parts.get("drop_sum", 0.0))
                    drop_ratio_step_sum += float(parts.get("drop_ratio", 0.0))
                    drop_event_steps += float(parts.get("drop_event", 0.0))
                    term_drop_sum += float(parts.get("term_drop", 0.0))
                    term_drop_gu_sum += float(parts.get("term_drop_gu", 0.0))
                    term_drop_uav_sum += float(parts.get("term_drop_uav", 0.0))
                    term_drop_sat_sum += float(parts.get("term_drop_sat", 0.0))
                    term_drop_step_sum += float(parts.get("term_drop_step", 0.0))
                    collision_any = max(collision_any, float(parts.get("collision_event", 0.0)))
                    danger_imitation_active_rate_sum += float(parts.get("danger_imitation_active_rate", 0.0))
                    intervention_norm_sum += float(parts.get("intervention_norm", 0.0))
                    intervention_rate_sum += float(parts.get("intervention_rate", 0.0))
                    intervention_norm_top1_sum += float(parts.get("intervention_norm_top1", 0.0))
                    close_risk_sum += float(parts.get("close_risk", 0.0))
                    term_close_risk_sum += float(parts.get("term_close_risk", 0.0))
                    centroid_dist_sum += float(parts.get("centroid_dist_mean", 0.0))
                if cfg.num_uav > 1 and hasattr(env, "uav_pos"):
                    diff = env.uav_pos[:, None, :] - env.uav_pos[None, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dists, np.inf)
                    cur_min_dist = float(np.min(dists))
                    min_inter_uav_dist = min(min_inter_uav_dist, cur_min_dist)
                    if cur_min_dist < float(cfg.avoidance_alert_factor) * float(cfg.d_safe):
                        near_collision_steps += 1.0
                assoc_ratio = 0.0
                assoc_dist = 0.0
                assoc_counts = np.zeros((cfg.num_uav,), dtype=np.float32)
                access_outflow_per_uav = np.zeros((cfg.num_uav,), dtype=np.float32)
                if cfg.num_gu > 0 and hasattr(env, "last_association"):
                    assoc = np.asarray(env.last_association, dtype=np.int32)
                    mask = assoc >= 0
                    if mask.size > 0:
                        assoc_ratio = float(np.mean(mask))
                        if np.any(mask):
                            assoc_counts = np.bincount(assoc[mask], minlength=cfg.num_uav).astype(np.float32)
                            if hasattr(env, "last_gu_outflow"):
                                gu_outflow = np.asarray(env.last_gu_outflow, dtype=np.float32)
                                access_outflow_per_uav = np.bincount(
                                    assoc[mask],
                                    weights=gu_outflow[mask],
                                    minlength=cfg.num_uav,
                                ).astype(np.float32)
                            gu_pos = env.gu_pos[mask]
                            u_idx = assoc[mask].astype(np.int32)
                            uav_pos = env.uav_pos[u_idx]
                            d2d = np.linalg.norm(gu_pos - uav_pos, axis=1)
                            assoc_dist = float(np.mean(d2d)) if d2d.size else 0.0
                assoc_ratio_sum += assoc_ratio
                assoc_dist_sum += assoc_dist
                for u in range(cfg.num_uav):
                    uav_assoc_gu_count_steps[u].append(float(assoc_counts[u]))
                    uav_access_outflow_steps[u].append(float(access_outflow_per_uav[u]))
                    uav_access_outflow_sum[u] += float(access_outflow_per_uav[u])
            ep_time = time.perf_counter() - ep_start
            steps = max(1, steps)
            layer_queue_end_sum = {
                "gu": float(np.sum(env.gu_queue)),
                "uav": float(np.sum(env.uav_queue)),
                "sat": float(np.sum(env.sat_queue)),
            }
            arrival_per_step = arrival_sum_ep / max(steps, 1)
            arrival_denom = max(arrival_per_step, 1e-9)
            layer_queue_arrival_steps_mean = {}
            layer_queue_arrival_steps_p95 = {}
            layer_queue_fill_fraction_mean = {}
            layer_queue_fill_fraction_p95 = {}
            layer_queue_nonzero_step_fraction = {}
            layer_queue_drift_ratio = {}
            for layer in ("gu", "uav", "sat"):
                queue_sum_steps = layer_queue_sum_steps[layer]
                queue_sum_mean = float(np.mean(queue_sum_steps)) if queue_sum_steps else 0.0
                queue_sum_p95 = float(np.percentile(queue_sum_steps, 95)) if queue_sum_steps else 0.0
                capacity = max(layer_total_capacity[layer], 1e-9)
                layer_queue_arrival_steps_mean[layer] = queue_sum_mean / arrival_denom
                layer_queue_arrival_steps_p95[layer] = queue_sum_p95 / arrival_denom
                layer_queue_fill_fraction_mean[layer] = queue_sum_mean / capacity
                layer_queue_fill_fraction_p95[layer] = queue_sum_p95 / capacity
                layer_queue_nonzero_step_fraction[layer] = (
                    max(steps - layer_zero_steps[layer], 0.0) / max(steps, 1)
                )
                layer_queue_drift_ratio[layer] = (
                    (layer_queue_end_sum[layer] - layer_queue_start_sum[layer]) / max(steps, 1)
                ) / arrival_denom
            active_net_drift_per_step = (arrival_sum_ep - sat_incoming_sum - active_drop_sum_ep) / max(steps, 1)
            sat_net_drift_per_step = (sat_incoming_sum - sat_processed_sum - sat_drop_sum) / max(steps, 1)
            total_net_drift_per_step = (arrival_sum_ep - sat_processed_sum - drop_sum_ep) / max(steps, 1)
            if collision_any >= 0.5:
                termination_reason = "collision"
            elif steps < int(cfg.T_steps):
                termination_reason = "energy" if cfg.energy_enabled else "early_non_collision"
            else:
                termination_reason = "time_limit"
            metrics = {
                "episode": ep,
                "reward_sum": reward_sum,
                "reward_raw": reward_raw_sum / steps,
                "steps": steps,
                "processed_ratio_eval": processed_ratio_eval_sum / steps,
                "drop_ratio_eval": drop_ratio_eval_sum / steps,
                "pre_backlog_steps_eval": pre_backlog_steps_eval_sum / steps,
                "D_sys_report": (
                    float(np.sum(env.gu_queue) + np.sum(env.uav_queue) + np.sum(env.sat_queue))
                    / max(sat_processed_sum, 1e-9)
                ),
                "x_acc_mean": x_acc_sum / steps,
                "x_rel_mean": x_rel_sum / steps,
                "g_pre_mean": g_pre_sum / steps,
                "d_pre_mean": d_pre_sum / steps,
                "episode_time_sec": ep_time,
                "steps_per_sec": steps / max(1e-9, ep_time),
                "service_norm": service_norm_sum / steps,
                "throughput_access_norm": throughput_access_norm_sum / steps,
                "throughput_backhaul_norm": throughput_backhaul_norm_sum / steps,
                "sat_processed_norm": sat_processed_norm_sum / steps,
                "drop_norm": drop_norm_sum / steps,
                "queue_total_active": queue_total_active_sum / steps,
                "queue_total_active_excl_step0": (
                    sum(queue_total_active_steps[1:]) / len(queue_total_active_steps[1:])
                    if len(queue_total_active_steps) > 1
                    else 0.0
                ),
                "queue_total_active_max": max(queue_total_active_steps) if queue_total_active_steps else 0.0,
                "queue_total_active_p95_step": (
                    float(np.percentile(queue_total_active_steps, 95)) if queue_total_active_steps else 0.0
                ),
                "arrival_sum": arrival_sum_ep,
                "outflow_sum": outflow_sum_ep,
                "outflow_arrival_ratio": outflow_sum_ep / max(arrival_sum_ep, 1e-9),
                "sat_incoming_sum": sat_incoming_sum,
                "sat_processed_sum": sat_processed_sum,
                "sat_incoming_arrival_ratio": sat_incoming_sum / max(arrival_sum_ep, 1e-9),
                "sat_processed_arrival_ratio": sat_processed_sum / max(arrival_sum_ep, 1e-9),
                "sat_processed_incoming_ratio": sat_processed_sum / max(sat_incoming_sum, 1e-9),
                "drop_sum": drop_sum_ep,
                "drop_ratio": drop_sum_ep / max(arrival_sum_ep, 1e-9),
                "drop_ratio_step_mean": drop_ratio_step_sum / steps,
                "drop_sum_step_mean": drop_sum_ep / steps,
                "active_drop_sum": active_drop_sum_ep,
                "active_drop_ratio": active_drop_sum_ep / max(arrival_sum_ep, 1e-9),
                "active_drop_sum_step_mean": active_drop_sum_ep / steps,
                "gu_drop_sum": gu_drop_sum,
                "uav_drop_sum": uav_drop_sum,
                "sat_drop_sum": sat_drop_sum,
                "sat_drop_sum_step_mean": sat_drop_sum / steps,
                "gu_drop_ratio": gu_drop_sum / max(arrival_sum_ep, 1e-9),
                "uav_drop_ratio": uav_drop_sum / max(arrival_sum_ep, 1e-9),
                "sat_drop_ratio": sat_drop_sum / max(arrival_sum_ep, 1e-9),
                "drop_event_steps": drop_event_steps,
                "drop_event_step_fraction": drop_event_steps / steps,
                "term_drop_mean": term_drop_sum / steps,
                "term_drop_gu_mean": term_drop_gu_sum / steps,
                "term_drop_uav_mean": term_drop_uav_sum / steps,
                "term_drop_sat_mean": term_drop_sat_sum / steps,
                "term_drop_step_mean": term_drop_step_sum / steps,
                "terminated_early": 1.0 if steps < int(cfg.T_steps) else 0.0,
                "termination_reason": termination_reason,
                "collision": collision_any,
                "min_inter_uav_dist": 0.0 if not np.isfinite(min_inter_uav_dist) else min_inter_uav_dist,
                "near_collision_steps": near_collision_steps,
                "near_collision_ratio": near_collision_steps / steps,
                "danger_imitation_active_rate_mean": danger_imitation_active_rate_sum / steps,
                "intervention_norm_mean": intervention_norm_sum / steps,
                "intervention_rate_mean": intervention_rate_sum / steps,
                "intervention_norm_top1_mean": intervention_norm_top1_sum / steps,
                "close_risk_mean": close_risk_sum / steps,
                "term_close_risk_mean": term_close_risk_sum / steps,
                "centroid_dist_mean": centroid_dist_sum / steps,
                "gu_queue_mean": gu_queue_sum / steps,
                "uav_queue_mean": uav_queue_sum / steps,
                "sat_queue_mean": sat_queue_sum / steps,
                "gu_queue_max": gu_queue_max,
                "uav_queue_max": uav_queue_max,
                "sat_queue_max": sat_queue_max,
                "gu_queue_arrival_steps_mean": layer_queue_arrival_steps_mean["gu"],
                "uav_queue_arrival_steps_mean": layer_queue_arrival_steps_mean["uav"],
                "sat_queue_arrival_steps_mean": layer_queue_arrival_steps_mean["sat"],
                "gu_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["gu"],
                "uav_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["uav"],
                "sat_queue_arrival_steps_p95": layer_queue_arrival_steps_p95["sat"],
                "gu_queue_fill_fraction_mean": layer_queue_fill_fraction_mean["gu"],
                "uav_queue_fill_fraction_mean": layer_queue_fill_fraction_mean["uav"],
                "sat_queue_fill_fraction_mean": layer_queue_fill_fraction_mean["sat"],
                "gu_queue_fill_fraction_p95": layer_queue_fill_fraction_p95["gu"],
                "uav_queue_fill_fraction_p95": layer_queue_fill_fraction_p95["uav"],
                "sat_queue_fill_fraction_p95": layer_queue_fill_fraction_p95["sat"],
                "active_queue_empty_step_fraction": q_active_zero_steps / max(steps, 1),
                "all_layers_nonempty_step_fraction": all_layers_nonzero_steps / max(steps, 1),
                "gu_queue_nonzero_step_fraction": layer_queue_nonzero_step_fraction["gu"],
                "uav_queue_nonzero_step_fraction": layer_queue_nonzero_step_fraction["uav"],
                "sat_queue_nonzero_step_fraction": layer_queue_nonzero_step_fraction["sat"],
                "gu_queue_drift_ratio": layer_queue_drift_ratio["gu"],
                "uav_queue_drift_ratio": layer_queue_drift_ratio["uav"],
                "sat_queue_drift_ratio": layer_queue_drift_ratio["sat"],
                "active_net_drift_per_step": active_net_drift_per_step,
                "sat_net_drift_per_step": sat_net_drift_per_step,
                "total_net_drift_per_step": total_net_drift_per_step,
                "connected_sat_count_mean": connected_sat_count_sum / steps,
                "connected_sat_dist_mean": connected_sat_dist_mean_sum / steps,
                "connected_sat_dist_p95_mean": connected_sat_dist_p95_sum / steps,
                "connected_sat_elevation_deg_mean": connected_sat_elevation_deg_mean_sum / steps,
                "connected_sat_elevation_deg_min": (
                    0.0 if not np.isfinite(connected_sat_elevation_deg_min_episode)
                    else connected_sat_elevation_deg_min_episode
                ),
                "energy_mean": (energy_mean_sum / steps) if cfg.energy_enabled else 0.0,
                "assoc_ratio_mean": assoc_ratio_sum / steps,
                "assoc_dist_mean": assoc_dist_sum / steps,
            }
            writer.writerow([metrics.get(name, "") for name in fieldnames])
            for key, val in metrics.items():
                if key in tb_field_set and isinstance(val, (int, float, np.floating, np.integer)):
                    tb_writer.add_scalar(f"{tb_tag}/{key}", float(val), ep)
            progress.update(ep + 1)
    progress.close()
    tb_writer.close()
    print(f"Saved evaluation metrics to {args.out}")


if __name__ == "__main__":
    main()
