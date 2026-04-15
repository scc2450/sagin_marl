from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import re

import math

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class AblationConfig:
    """Toggle non-standard mechanisms for ablation studies."""

    # RL-side additions over vanilla PPO
    use_imitation_loss: bool = False
    use_heuristic_mask: bool = False
    use_magic_decay: bool = False  # Reserved for future schedules.

    # Environment curricula / safety layers
    use_curriculum_spawn: bool = False
    use_arrival_ramp: bool = False
    use_avoidance_layer: bool = False
    use_energy_safety_layer: bool = False

    # Reward shaping components
    use_throughput_reward: bool = True
    use_potential_shaping: bool = True
    use_legacy_queue_penalty: bool = False
    use_queue_log_smoothing: bool = False
    use_queue_topk_penalty: bool = False
    use_queue_delta_reward: bool = False
    use_active_queue_delta: bool = False
    use_centroid_reward: bool = False
    use_bw_align_reward: bool = False
    use_sat_score_reward: bool = False
    use_dist_delta_reward: bool = False
    use_energy_reward: bool = False
    use_reward_tanh: bool = False


@dataclass
class SaginConfig:
    seed: int = 42
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Map and timing
    map_size: float = 1000.0
    tau0: float = 1.0
    T_steps: int = 400

    # Entity counts
    num_uav: int = 3
    num_gu: int = 20
    num_sat: int = 6

    # Observation limits
    users_obs_max: int = 20
    sats_obs_max: int = 6
    nbrs_obs_max: int = 4
    danger_nbr_enabled: bool = False
    visible_sats_max: int | None = None
    visible_sats_min: int | None = None
    sat_candidate_mode: str = "elevation"  # "elevation" or "score"
    sat_candidate_elevation_weight: float = 1.0
    sat_candidate_se_weight: float = 1.0
    sat_candidate_queue_weight: float = 1.0

    # Geometry
    uav_height: float = 100.0
    sat_height: float = 500_000.0
    r_earth: float = 6_371_000.0
    theta_min_deg: float = 10.0
    ref_lat_deg: float = 0.0
    ref_lon_deg: float = 0.0

    # Walker-Delta constellation
    walker_num_planes: int = 3
    walker_inclination_deg: float = 53.0
    walker_phase_factor: int = 1
    earth_rotation_rate: float = 7.2921159e-5  # rad/s

    # UAV dynamics
    v_max: float = 30.0
    a_max: float = 5.0
    d_safe: float = 20.0
    boundary_mode: str = "clip"  # "clip" or "reflect"
    boundary_hard_filter_enabled: bool = False
    boundary_margin: float | None = None
    pairwise_hard_filter_enabled: bool = False
    pairwise_hard_distance: float | None = None
    pairwise_hard_max_passes: int = 2
    pairwise_hard_trigger_mode: str = "distance"  # "distance" or "ttc"
    pairwise_hard_trigger_ttc: float = 2.0
    pairwise_hard_trigger_distance: float | None = None
    pairwise_hard_closing_speed: float = 0.0
    pairwise_hard_single_pair_only: bool = True
    uav_spawn_curriculum_enabled: bool = False
    uav_spawn_radius_start: float = 100.0
    uav_spawn_radius_end: float | None = None
    uav_spawn_curriculum_steps: int = 0
    uav_spawn_full_random_final: bool = True
    uav_safe_random_init_enabled: bool = False
    uav_init_boundary_margin_steps: float = 3.0
    uav_init_speed_frac: float = 0.2
    uav_init_min_spacing: float | None = None
    uav_init_max_tries: int = 256

    # Queues (bits)
    queue_max_gu: float = 5e6
    queue_max_uav: float = 1e7
    queue_max_sat: float = 5e7
    queue_max_gu_steps: float | None = None
    queue_max_uav_steps: float | None = None
    queue_max_sat_steps: float | None = None
    queue_init_frac: float = 0.0
    queue_init_uav_frac: float = 0.0
    queue_init_sat_frac: float = 0.0
    queue_init_gu_abs: float | None = None
    queue_init_uav_abs: float | None = None
    queue_init_sat_abs: float | None = None
    queue_init_gu_steps: float | None = None
    queue_init_uav_steps: float | None = None
    queue_init_sat_steps: float | None = None
    queue_ref_gu_per_step: float | None = None
    queue_ref_uav_per_step: float | None = None
    queue_ref_sat_per_step: float | None = None
    queue_ref_sat_active_count: float | None = None

    # Task arrivals
    task_arrival_rate: float = 2e5  # bits per slot (mean)
    task_arrival_poisson: bool = True
    traffic_level: int = 2
    traffic_level_nav_ratio: float = 0.08
    traffic_level_easy_ratio: float = 0.5
    traffic_level_hard_ratio: float = 1.0
    arrival_ramp_steps: int = 0
    arrival_ramp_start: float = 0.0
    arrival_ramp_use_global: bool = False

    # Communication
    b_acc: float = 5e6
    b_sat_total: float = 20e6
    gu_tx_power: float = 0.2  # Watts
    uav_tx_power: float = 1.0  # Watts
    uav_tx_gain: float = 300.0
    sat_rx_gain: float = 300.0
    noise_density: float = 4e-21  # W/Hz (thermal noise at ~290K)
    carrier_freq: float = 2e9
    speed_of_light: float = 3e8
    pathloss_const_db: float = 32.4
    los_a: float = 9.61
    los_b: float = 0.16
    xi_los: float = 1.0
    xi_nlos: float = 20.0
    pl_threshold_db: float = 140.0
    pathloss_mode: str = "prob_los"  # "prob_los" or "free_space"
    rician_K: float = 10.0
    atm_loss_enabled: bool = False
    atm_loss_db: float = 2.0
    subcarrier_spacing: float = 15e3

    # Satellite compute
    sat_cpu_freq: float = 1e10  # cycles/s
    task_cycles_per_bit: float = 1000.0  # cycles/bit

    # Doppler
    nu_max: float = 2000.0
    doppler_observed: bool = True
    doppler_atten_enabled: bool = False

    # Phase toggles
    doppler_enabled: bool = False
    energy_enabled: bool = False
    fading_enabled: bool = False
    interference_enabled: bool = False
    enable_bw_action: bool = False
    fixed_satellite_strategy: bool = True
    N_RF: int = 1
    sat_select_mode: str = "topk"
    sat_state_max: int | None = None
    append_action_masks_to_obs: bool = True

    # Collision avoidance (optional safety layer)
    avoidance_enabled: bool = False
    avoidance_eta: float = 100.0
    avoidance_alert_factor: float = 1.5
    avoidance_prealert_factor: float | None = None
    avoidance_prealert_closing_speed: float = 0.0
    avoidance_prealert_mode: str = "distance"  # "distance" or "ttc"
    avoidance_prealert_ttc: float | None = None
    avoidance_prealert_dist_cap: float | None = None
    avoidance_repulse_mode: str = "inverse"  # "inverse", "linear", "quadratic"
    avoidance_repulse_clip: bool = True
    avoidance_closing_gain_enabled: bool = False
    avoidance_closing_gain_cap: float = 2.0
    avoidance_closing_gain_top1_only: bool = False
    avoidance_adaptive_enabled: bool = False
    avoidance_collision_target: float = 0.05
    avoidance_adaptive_gain: float = 1.0
    avoidance_adaptive_ema_beta: float = 0.9
    avoidance_eta_min: float = 0.0
    avoidance_eta_max: float | None = None

    # Energy placeholders
    uav_energy_init: float = 1.0
    p_fly_base: float = 0.01
    p_fly_coeff: float = 0.001
    p_comm_link: float = 0.01
    energy_model: str = "simple"  # "simple" or "rotor"
    energy_safety_enabled: bool = False
    energy_safe_threshold: float = 0.2  # fraction of init energy
    uav_opt_speed: float = 10.0

    # Rotorcraft power model params (for energy_model="rotor")
    rotor_p0: float = 79.86
    rotor_pi: float = 88.63
    rotor_u_tip: float = 120.0
    rotor_v0: float = 4.03
    rotor_d0: float = 0.6
    rotor_rho: float = 1.225
    rotor_s: float = 0.05
    rotor_area: float = 0.503

    # Action logit scales
    bw_logit_scale: float = 5.0
    sat_logit_scale: float = 5.0
    bw_residual_alpha: float = 0.5
    bw_residual_clip: float = 1.0
    bw_residual_l2_coef: float = 0.0
    bw_head_zero_init: bool = False
    bw_log_std_init: float = 0.0
    bw_log_std_trainable: bool = True
    bw_policy: str = "dirichlet"
    bw_alpha_floor: float = 0.2
    sat_policy: str = "masked_categorical"
    sat_num_select: int | None = None

    # Baseline heuristics (queue_aware)
    baseline_accel_gain: float = 2.0
    baseline_assoc_bonus: float = 0.3
    baseline_sat_se_weight: float = 0.75
    baseline_sat_queue_penalty: float = 0.25
    baseline_sat_load_penalty: float = 0.75
    baseline_sat_bw_reward: float = 0.75
    baseline_sat_stay_bonus: float = 1.0
    baseline_sat_switch_margin: float = 0.35
    baseline_repulse_gain: float = 1.0
    baseline_repulse_radius_factor: float = 1.5
    baseline_energy_low: float = 0.3
    baseline_energy_weight: float = 1.0
    baseline_cluster_cruise_speed: float | None = None
    baseline_cluster_slow_radius: float = 120.0
    baseline_cluster_stop_radius: float = 20.0
    baseline_cluster_speed_tol: float = 2.0
    baseline_cluster_vel_gain: float = 1.0
    baseline_lyapunov_v: float = 2.0
    baseline_lyapunov_urgency_alpha: float = 1.0
    baseline_lyapunov_drift_weight: float = 1.2
    baseline_lyapunov_action_cost: float = 0.03
    baseline_lyapunov_ema_beta: float = 0.6
    baseline_lyapunov_bw_temp: float = 0.55
    baseline_lyapunov_bw_floor: float = 0.01
    baseline_lyapunov_bw_service_scale: float = 1.0
    baseline_lyapunov_sat_drift_weight: float = 0.6
    baseline_lyapunov_sat_switch_bias: float = 0.1
    baseline_lyapunov_mode: str = "urgency"  # "urgency" or "dpp"
    dpp_accel_num_candidates: int = 9
    dpp_accel_step_scale: float = 0.6
    dpp_shared_accel_index: bool = False
    dpp_gu_max_select: int = 6
    dpp_access_weight: float = 1.0
    dpp_backhaul_weight: float = 1.0
    dpp_accel_cost: float = 0.08
    dpp_smoothness: float = 0.05
    dpp_dist_penalty: float = 0.10
    dpp_bw_temp: float = 0.55
    dpp_bw_floor: float = 0.01
    dpp_sat_queue_gap_weight: float = 1.0
    dpp_sat_candidate_topm: int = 4
    dpp_sat_enum_max_subsets: int = 64
    dpp_sat_subset_penalty: float = 0.02
    dpp_sat_contention_weight: float = 0.15

    # Reward shaping
    reward_mode: str = "dense"  # "controllable_flow", "dense", or "throughput_only"
    arrival_ref_mode: str = "expected_arrival"
    use_queue_max_norm: bool = False
    reward_w_access: float = 0.5
    reward_w_relay: float = 0.5
    reward_w_pre_backlog: float = 0.08
    reward_w_pre_drop: float = 1.0
    reward_w_pre_growth: float = 0.0
    throughput_only_access_coef: float = 1.0
    throughput_only_backhaul_coef: float = 1.0
    throughput_only_gu_queue_coef: float = 0.0
    omega_q: float = 0.6
    omega_q_gu: float = 1.0
    omega_q_uav: float = 0.0
    omega_q_sat: float = 0.0
    omega_q_topk: float = 0.0
    omega_e: float = 0.0
    eta_crash: float = 5.0
    eta_batt: float = 5.0
    eta_drop: float = 1.0
    eta_drop_gu: float | None = None
    eta_drop_uav: float | None = None
    eta_drop_sat: float | None = None
    eta_drop_step: float = 10.0
    eta_cong: float = 0.1
    eta_service: float = 0.0
    eta_assoc: float = 0.2
    eta_q_delta: float = 0.6
    eta_throughput_access: float = 0.0
    eta_throughput_backhaul: float = 0.0
    eta_accel: float = 0.02
    close_risk_enabled: bool = False
    eta_close_risk: float = 0.0
    close_risk_cap: float = 2.0
    tail_q_small: float = 0.0
    tail_eta_accel_gain: float = 1.0
    eta_centroid: float = 0.6
    eta_centroid_final: float | None = None
    eta_centroid_decay_steps: int = 0
    centroid_cross_anneal_enabled: bool = False
    centroid_cross_queue_gain: float = 0.0
    centroid_cross_q_delta_gain: float = 0.0
    centroid_cross_crash_gain: float = 0.0
    centroid_cross_avoidance_gain: float = 0.0
    centroid_dist_scale: float = 800.0
    eta_bw_align: float = 0.3
    eta_sat_score: float = 0.1
    eta_dist: float = 0.0
    eta_dist_delta: float = 0.0
    dist_reward_scale: float = 0.0
    queue_penalty_mode: str = "quadratic"  # "quadratic", "linear", or "log"
    queue_log_k: float = 0.0
    queue_norm_K: float = 1.0
    queue_norm_arrival_floor: float = 0.0
    queue_reward_use_arrival_norm: bool = False
    q_norm_tail_q0: float = 0.0
    omega_q_tail: float | None = None
    queue_topk_k: int = 0
    queue_topk_local: bool = False
    queue_delta_use_active: bool = False
    queue_delta_mode: str = "total"  # "total" or "weighted" when active queue delta is disabled
    candidate_mode: str = "assoc"  # "assoc", "nearest", "radius"
    candidate_radius: float | None = None
    candidate_k: int | None = None
    assoc_unfair_gu_threshold: int = 15
    queue_th_gu: float | None = None
    queue_th_uav: float | None = None
    queue_th_gu_frac: float = 0.8
    queue_th_uav_frac: float = 0.8

    # PPO defaults (hardware aware)
    buffer_size: int = 4000
    num_mini_batch: int = 32
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    adv_clip: float = 5.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    train_episode_stat_window: int = 100
    reward_norm_enabled: bool = True
    reward_norm_clip: float = 10.0
    reward_tanh_enabled: bool = False
    lr_decay_enabled: bool = True
    lr_final_factor: float = 0.1
    input_norm_enabled: bool = True
    kl_coef: float = 0.0
    target_kl: float = 0.0
    kl_stop: bool = False
    imitation_enabled: bool = False
    imitation_coef: float = 0.0
    imitation_coef_final: float | None = None
    imitation_coef_decay_start_update: int = 0
    imitation_coef_decay_updates: int = 0
    imitation_accel: bool = True
    imitation_bw: bool = True
    imitation_sat: bool = False
    danger_imitation_enabled: bool = False
    danger_imitation_coef: float = 0.0
    danger_imitation_trigger_mode: str = "risk_or_intervention"
    danger_imitation_close_risk_thresh: float = 0.05
    danger_imitation_intervention_thresh: float = 0.05
    train_accel: bool | None = None
    train_bw: bool | None = None
    train_sat: bool | None = None
    train_shared_backbone: bool = True
    train_fusion: bool = False
    train_fusion_last_layer: bool = False
    exec_accel_source: str = "policy"  # policy|teacher|heuristic|zero
    exec_bw_source: str = "policy"  # policy|teacher|heuristic|zero|heuristic_residual
    exec_sat_source: str = "policy"  # policy|teacher|heuristic|zero
    exec_teacher_actor_path: str | None = None
    exec_teacher_deterministic: bool = True
    reward_stage1_assoc_centroid_enabled: bool = False
    reward_stage1_assoc_centroid_weight_init: float = 0.15
    reward_stage1_assoc_centroid_weight_mid: float = 0.05
    reward_stage1_assoc_centroid_weight_floor: float = 0.0
    reward_stage1_assoc_centroid_hold_ratio: float = 0.30
    reward_stage1_assoc_centroid_mid_ratio: float = 0.70
    reward_stage3_sat_overlap_enabled: bool = False
    reward_stage3_sat_overlap_weight_init: float = 0.10
    reward_stage3_sat_overlap_weight_mid: float = 0.03
    reward_stage3_sat_overlap_weight_floor: float = 0.0
    reward_stage3_sat_overlap_hold_ratio: float = 0.40
    reward_stage3_sat_overlap_mid_ratio: float = 0.80

    actor_hidden: int = 256
    actor_encoder_type: str = "flat_mlp"  # "flat_mlp" or "set_pool"
    actor_set_embed_dim: int = 64
    critic_hidden: int = 256

    # Early stopping (convergence)
    early_stop_enabled: bool = True
    early_stop_min_updates: int = 20
    early_stop_window: int = 5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3
    checkpoint_eval_enabled: bool = False
    checkpoint_eval_interval_updates: int = 0
    checkpoint_eval_start_update: int = 0
    checkpoint_eval_episodes: int = 20
    checkpoint_eval_episode_seed_base: int | None = None
    checkpoint_eval_fixed_policy: str = "zero"  # "zero" | "queue_aware" | "teacher_accel_queue_aware"
    checkpoint_eval_sat_drop_early_stop_enabled: bool = True
    checkpoint_eval_sat_drop_worsen_delta: float = 5e-4
    checkpoint_eval_front_queue_rel_improve_tol: float = 0.05
    checkpoint_eval_worsen_patience: int = 2
    checkpoint_eval_early_stop_enabled: bool = True
    checkpoint_eval_reward_early_stop_enabled: bool = False
    checkpoint_eval_reward_patience: int = 5
    checkpoint_eval_reward_min_delta_rel: float = 0.0
    checkpoint_eval_reward_collision_threshold: float = 1.0
    checkpoint_eval_model_collision_threshold: float = 0.0
    checkpoint_eval_reward_tie_rel_tol: float = 0.05
    checkpoint_eval_use_sat_overlap: bool = False

    @property
    def theta_min_rad(self) -> float:
        return math.radians(self.theta_min_deg)


def ablation_flag(cfg: SaginConfig, name: str, fallback_attr: str | None = None, default: bool = False) -> bool:
    """Read an ablation flag with optional legacy fallback."""
    ablation = getattr(cfg, "ablation", None)
    ablation_value: bool | None = None
    if ablation is not None and hasattr(ablation, name):
        ablation_value = bool(getattr(ablation, name))
    fallback_value: bool | None = None
    if fallback_attr and hasattr(cfg, fallback_attr):
        fallback_value = bool(getattr(cfg, fallback_attr))
    if ablation_value is None and fallback_value is None:
        return bool(default)
    # Backward compatibility: legacy toggles (e.g., imitation_enabled) should
    # still enable the feature unless explicitly turned off there as well.
    return bool((ablation_value or False) or (fallback_value or False))


def load_config(path: str) -> SaginConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return update_config(SaginConfig(), data)


def update_config(cfg: SaginConfig, updates: Dict[str, Any]) -> SaginConfig:
    if not isinstance(updates, dict):
        raise TypeError("Config updates must be a mapping.")
    _apply_updates(cfg, updates)
    _finalize_config(cfg)
    return cfg


def _apply_updates(target: Any, updates: Dict[str, Any], path: str = "") -> None:
    for key, value in updates.items():
        k = "ablation" if key == "ablation_flags" else key
        full_key = f"{path}.{k}" if path else k
        if not hasattr(target, k):
            raise KeyError(f"Unknown config key: {full_key}")

        current = getattr(target, k)
        if _is_dataclass_instance(current):
            if not isinstance(value, dict):
                raise TypeError(f"Expected mapping for nested config key: {full_key}")
            _apply_updates(current, value, full_key)
            continue

        coerced = _coerce_scalar(value, current)
        if isinstance(current, bool) and isinstance(coerced, int):
            coerced = bool(coerced)
        if isinstance(current, int) and isinstance(coerced, float) and coerced.is_integer():
            coerced = int(coerced)
        setattr(target, k, coerced)


def _is_dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def _coerce_scalar(value: Any, current: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    low = s.lower()

    if isinstance(current, bool) or low in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False

    if isinstance(current, (int, float)) or current is None:
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return value

    return value


def _traffic_level_ratio(cfg: SaginConfig) -> float:
    raw_level = getattr(cfg, "traffic_level", 2)
    level = int(2 if raw_level is None else raw_level)
    if level <= 0:
        ratio = float(getattr(cfg, "traffic_level_nav_ratio", 0.08) or 0.08)
    elif level == 1:
        ratio = float(getattr(cfg, "traffic_level_easy_ratio", 0.5) or 0.5)
    else:
        ratio = float(getattr(cfg, "traffic_level_hard_ratio", 1.0) or 1.0)
    return min(max(ratio, 0.0), 1.0)


def _default_queue_ref_per_step(cfg: SaginConfig) -> float:
    base_rate = max(float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0), 0.0)
    return base_rate * _traffic_level_ratio(cfg) * float(cfg.num_gu) * float(cfg.tau0)


def _queue_ref_per_step(cfg: SaginConfig, layer: str) -> float:
    ref_value = getattr(cfg, f"queue_ref_{layer}_per_step", None)
    if ref_value is not None:
        return max(float(ref_value), 0.0)
    return _default_queue_ref_per_step(cfg)


def _queue_ref_entities(cfg: SaginConfig, layer: str) -> float:
    if layer == "gu":
        return max(float(cfg.num_gu), 1.0)
    if layer == "uav":
        return max(float(cfg.num_uav), 1.0)
    active_count = getattr(cfg, "queue_ref_sat_active_count", None)
    if active_count is not None:
        return max(float(active_count), 1.0)
    return max(float(cfg.num_sat), 1.0)


def _finalize_config(cfg: SaginConfig) -> None:
    for layer in ("gu", "uav", "sat"):
        steps_value = getattr(cfg, f"queue_max_{layer}_steps", None)
        if steps_value is None:
            continue
        total_cap = max(float(steps_value), 0.0) * _queue_ref_per_step(cfg, layer)
        per_entity_cap = total_cap / _queue_ref_entities(cfg, layer)
        setattr(cfg, f"queue_max_{layer}", per_entity_cap)
