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
    per_uav_visible_sat_token_max: int | None = None
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
    safety_shield_enabled: bool = False
    safety_shield_solver: str = "CLARABEL"
    safety_shield_julia_exe: str = ""
    safety_shield_julia_project: str = ""
    safety_shield_julia_depot: str = ""
    safety_shield_brake_rho: float = 0.8
    safety_shield_a_safe: float = 0.0
    safety_shield_distance_buffer: float = 2.0
    safety_shield_tolerance: float = 1.0e-5
    safety_shield_relax_on_infeasible: bool = True
    safety_shield_relax_slack_weight: float = 1.0e4
    safety_shield_relax_action_weight: float = 1.0
    safety_shield_tensor_iters: int = 8
    safety_shield_tensor_step_gain: float = 1.0
    safety_shield_verbose: bool = False
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
    uav_spawn_mode: str = "default"  # "default" | "gu_centroid" | "gu_cluster_centers"
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

    # Optional resource auto-scaling from a reference system footprint.
    resource_scale_enabled: bool = False
    resource_scale_ref_num_uav: float | None = None
    resource_scale_ref_num_gu: float | None = None
    resource_scale_ref_task_arrival_rate: float | None = None
    resource_scale_sat_active_count: float | None = None
    resource_scale_ref_sat_active_count: float | None = None
    resource_scale_b_acc_multiplier: float = 1.0

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
    traffic_model: str = "homogeneous"
    arrival_base_hetero: float = 0.0
    gu_init_num_clusters: int | None = None
    gu_init_cluster_std: float = 80.0
    gu_init_cluster_center_min_dist: float = 0.0
    hotspot_num_subsets: int = 0
    hotspot_subset_size: int = 4
    hotspot_rho: float = 4.0
    hotspot_on_mean_steps: float = 15.0
    hotspot_off_mean_steps: float = 8.0
    arrival_mean_preserve: bool = True

    # Communication
    b_acc: float = 5e6
    b_backhaul_per_sat: float = 20e6
    b_backhaul_per_sat_scale: float = 1.0
    # Legacy aliases kept for existing YAML files/checkpoints. These are
    # synchronized in _finalize_config; new configs should use
    # b_backhaul_per_sat(_scale).
    b_sat_total: float | None = None
    b_sat_total_scale: float | None = None
    gu_tx_power: float = 0.2  # Watts
    uav_tx_power: float = 1.0  # Watts
    uav_tx_gain: float = 300.0
    sat_rx_gain: float = 300.0
    noise_density: float = 4e-21  # W/Hz (thermal noise at ~290K)
    carrier_freq: float = 2e9
    access_carrier_freq: float = 2e9
    backhaul_carrier_freq: float | None = None
    access_noise_figure_db: float = 5.0
    backhaul_noise_figure_db: float = 3.0
    speed_of_light: float = 3e8
    pathloss_const_db: float = 32.4
    los_a: float = 9.61
    los_b: float = 0.16
    xi_los: float = 1.0
    xi_nlos: float = 20.0
    pl_threshold_db: float = 140.0
    pathloss_mode: str = "prob_los"  # "prob_los" or "free_space"
    rician_K: float = 10.0
    access_fading_mode: str = "ergodic_rician"  # "large_scale", "ergodic_rician", or "iid_rician"
    access_rician_k_db: float | None = None
    access_ergodic_rician_quadrature_points: int = 16
    atm_loss_enabled: bool = False
    atm_loss_db: float = 2.0
    rain_loss_enabled: bool = False
    rain_rate_001_mmph: float = 0.0
    rain_height_km: float = 5.0
    rain_exceedance_pct: float = 0.1
    rain_polarization_tilt_deg: float = 45.0
    rain_lat_deg: float | None = None
    subcarrier_spacing: float = 15e3

    # Satellite compute
    sat_cpu_freq: float = 1e10  # cycles/s
    sat_cpu_freq_scale: float = 1.0
    task_cycles_per_bit: float = 1000.0  # cycles/bit

    # Localized queue preload for bw-focus environments
    preload_enabled: bool = False
    preload_prob: float = 0.0
    preload_hot_gu_steps: float = 0.0
    preload_bg_gu_steps: float = 0.0
    preload_hot_uav_steps: float = 0.0
    preload_sat_steps: float = 0.0

    # Doppler
    nu_max: float = 2000.0
    doppler_observed: bool = True
    doppler_atten_enabled: bool = False
    doppler_precomp_mode: str = "none"  # "none", "residual_hz", or "residual_ppm"
    doppler_residual_hz: float = 0.0
    doppler_residual_ppm: float = 0.0
    doppler_residual_ar_rho: float = 0.98
    doppler_residual_sigma_hz: float = 100.0

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
    obs_own_include_assoc_uav_cost: bool = False
    obs_user_include_arrival_rate: bool = False
    obs_user_include_recent_arrival: bool = False
    obs_user_include_recent_service: bool = False
    obs_user_include_queue_headroom: bool = False
    obs_user_include_local_gu_service_cost: bool = False
    obs_user_include_assoc_uav_cost: bool = False
    obs_user_include_assoc_sat_cost_mean: bool = False
    obs_user_include_weighted_queue_cost: bool = False
    obs_user_include_weighted_queue_cost_relative: bool = False
    obs_user_include_urgency_risk: bool = False
    obs_user_include_downstream_pressure: bool = False
    obs_user_include_service_gap: bool = False
    obs_user_include_service_gap_risk: bool = False
    obs_user_include_deadline_slack: bool = False
    obs_user_include_deadline_risk: bool = False
    obs_sat_include_sat_cost: bool = False

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
    accel_log_std_init: float = 0.0
    accel_log_std_trainable: bool = True
    bw_log_std_init: float = 0.0
    bw_log_std_trainable: bool = True
    bw_policy: str = "dirichlet"
    structured_bw_objective_norm: str = "per_latent_dim"
    bw_down_query_count: int = 2
    bw_competition_layers: int = 2
    bw_attention_heads: int = 4
    bw_manual_competition_attention_enabled: bool = False
    bw_tau_min: float = 0.5
    bw_tau_max: float = 2.0
    bw_kappa_min: float = 0.5
    bw_kappa_max: float = 32.0
    bw_fixed_tau: float | None = None
    bw_fixed_kappa: float | None = None
    bw_native_dirichlet_diagnostic_mode: str = "current"
    structured_env_backend: str = "native"  # "native" | "legacy" | "auto"
    structured_env_tensor_backend: str = "cuda"  # "cpu" | "cuda" | "auto"
    structured_native_cuda_empty_cache_after_update: bool = True
    structured_native_numeric_contract: str = "tensor_float32"
    structured_backhaul_rate_quantum: float = 32.0
    structured_flow_bits_quantum: float = 0.0
    structured_queue_state_quantum: float = 128.0
    structured_summary_metric_quantum: float = 0.0
    structured_kernel_operator_mode: str = "auto"  # "auto" | "eager" | "compile"
    structured_kernel_compile_backend: str = "auto"
    structured_kernel_compile_mode: str = "reduce-overhead"
    structured_kernel_compile_fullgraph: bool = False
    structured_kernel_compile_dynamic: bool = False
    structured_kernel_compile_cudagraphs: bool = False
    structured_kernel_cudagraph_mark_step_begin: bool = False
    structured_kernel_cudagraph_clone_output_kernels: str = "visible_sats"
    structured_kernel_cudagraph_direct_inputs: bool = True
    structured_kernel_compile_min_numel: int = 4096
    structured_kernel_auto_compile_warmup_calls: int = 8
    structured_kernel_compile_required: str = ""
    structured_kernel_compile_required_fullgraph: bool = False
    structured_kernel_compile_allowlist: str = (
        "access_eta_slots,gu_queue_transition,bw_queue_transition,"
        "close_risk_and_danger,visible_sats,bw_link_transition_cached,"
        "bw_link_transition_active_cached,access_rate,bw_post_stats,"
        "prepare_native_stage_batch,local_accel_components,local_bw_user_components,"
        "local_accel_full_obs,local_sat_full_obs,local_bw_full_obs,"
        "bw_workload_costs,bw_workload_rewards"
    )
    structured_native_reset_tape_chunk_rows: int = 8
    structured_native_main_kernel_require_compiled_segments: bool = False
    structured_native_main_kernel_required_compile_names: str = ""
    structured_native_history_snapshots_enabled: str = "auto"  # "auto" | "true" | "false"
    structured_acceptance_matrix_steps: int = 8
    structured_rollout_actor_compile_enabled: bool = False
    structured_rollout_actor_compile_mode: str = "default"
    structured_native_main_kernel_actor_compile_enabled: bool = False
    structured_native_accel_actor_fused: bool = True
    structured_native_sat_actor_fused: bool = True
    structured_native_bw_actor_fused: bool = True
    structured_bw_per_uav_surrogate_enabled: bool = False
    structured_bw_per_slot_surrogate_enabled: bool = False
    structured_bw_entropy_norm_mode: str = "none"  # "none", "per_latent_count", or "per_simplex_dim"
    bw_clean_per_user_enabled: bool = False
    bw_clean_per_user_horizon: int = 10
    bw_clean_per_user_delta_probe: float = 0.02
    bw_clean_per_user_beta: float = 0.5
    bw_clean_per_user_loss: str = "huber"  # "huber" or "masked_kl"
    bw_clean_candidate_select_enabled: bool = False
    bw_clean_candidate_select_deltas: str = "0.02,0.05,0.1,0.2,0.5"
    bw_clean_candidate_select_include_onehot: bool = True
    bw_clean_candidate_select_include_uniform: bool = True
    bw_clean_candidate_select_row_filter_mode: str = "ref_negative"  # "ref_negative" | "none"
    bw_clean_candidate_select_ref_return_eps: float = 1.0e-6
    bw_clean_candidate_select_gate_eps: float = 1.0e-6
    bw_clean_row_sample_enabled: bool = False
    bw_clean_row_sample_budget: int = 0
    bw_clean_row_sample_seed: int | None = None
    bw_clean_probe_engine: str = "worker_group_batched"  # "worker_group_batched" | "worker_grouped" | "roundtrip"
    bw_clean_grad_aggregation: str = "mean"  # "mean" | "pcgrad"
    bw_clean_pcgrad_task_group_size: int = 32
    bw_clean_pcgrad_group_mode: str = "random"  # "random" | "target_stats"
    bw_clean_trust_region_enabled: bool = False
    bw_clean_trust_region_target_kl: float = 0.01
    bw_clean_trust_region_kl_coef_init: float = 0.1
    bw_clean_trust_region_kl_coef_min: float = 1.0e-4
    bw_clean_trust_region_kl_coef_max: float = 1.0e3
    bw_clean_trust_region_backtrack_factor: float = 0.5
    bw_clean_trust_region_max_backtracks: int = 4
    bw_slot_advantage_weight: float = 1.0
    bw_slot_advantage_base_mode: str = "sample"  # "sample" | "zero"
    bw_alpha_floor: float = 0.2
    sat_policy: str = "masked_categorical"
    sat_num_select: int | None = None
    sat_action_select_k: int = 1
    sat_competition_layers: int = 2
    sat_attention_heads: int = 4

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
    baseline_lyapunov_drift_weight: float = 0.0
    baseline_lyapunov_action_cost: float = 0.0
    baseline_lyapunov_ema_beta: float = 0.0
    baseline_lyapunov_bw_temp: float = 1.0
    baseline_lyapunov_bw_floor: float = 0.0
    baseline_lyapunov_bw_service_scale: float = 1.0
    baseline_lyapunov_sat_drift_weight: float = 0.6
    baseline_lyapunov_sat_switch_bias: float = 0.0
    baseline_lyapunov_sat_abs_se_weight: float = 0.0
    baseline_lyapunov_sat_doppler_penalty: float = 0.0
    topology_dpp_accel_num_candidates: int = 9
    topology_dpp_accel_step_scale: float = 0.6
    topology_dpp_gu_max_select: int = 6
    topology_dpp_access_weight: float = 1.0
    topology_dpp_backhaul_weight: float = 1.0
    topology_dpp_mobility_weight: float = 0.75
    topology_dpp_accel_cost: float = 0.08
    topology_dpp_smoothness: float = 0.05
    topology_dpp_accel_safety_weight: float = 4.0
    topology_dpp_accel_role_weight: float = 0.35
    topology_dpp_dist_penalty: float = 0.10
    topology_dpp_bw_temp: float = 0.55
    topology_dpp_bw_floor: float = 0.01
    topology_dpp_sat_queue_gap_weight: float = 1.0
    topology_dpp_sat_candidate_topm: int = 4
    topology_dpp_sat_enum_max_subsets: int = 64
    topology_dpp_sat_subset_penalty: float = 0.02
    topology_dpp_sat_contention_weight: float = 0.15

    # Reward shaping
    reward_mode: str = "dense"  # "controllable_flow" | "dense" | "throughput_only" | "weighted_workload_delta" | "relative_weighted_workload_delta" | "weighted_workload_level" | "positive_weighted_workload_level" | "sat_relay_processed" | "sat_backhaul_drop" | "gu_queue_level" | "system_queue_level" | "gu_service_queue"
    arrival_ref_mode: str = "expected_arrival"
    use_queue_max_norm: bool = False
    reward_w_access: float = 0.5
    reward_w_relay: float = 0.5
    reward_w_pre_backlog: float = 0.08
    reward_w_pre_drop: float = 1.0
    reward_w_pre_overflow_risk: float = 0.0
    reward_w_pre_service_gap: float = 0.0
    reward_w_pre_growth: float = 0.0
    overflow_risk_threshold_frac: float = 0.75
    overflow_risk_arrival_coef: float = 1.0
    overflow_risk_service_coef: float = 0.1
    service_gap_increment: float = 1.0
    service_gap_relief_coef: float = 0.5
    service_gap_cap_steps: float = 8.0
    service_gap_risk_threshold_steps: float = 3.0
    deadline_enabled: bool = False
    deadline_base_steps: float = 4.0
    deadline_jitter_steps: float = 0.0
    deadline_age_increment: float = 1.0
    deadline_service_relief_coef: float = 0.75
    deadline_expire_rate: float = 0.35
    deadline_age_cap_steps: float = 8.0
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
    entropy_coef_accel: float | None = None
    entropy_coef_sat: float | None = None
    entropy_coef_bw: float | None = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    train_episode_stat_window: int = 100
    reward_norm_enabled: bool = True
    reward_norm_clip: float = 10.0
    reward_tanh_enabled: bool = False
    lr_decay_enabled: bool = True
    lr_final_factor: float = 0.1
    # Legacy flat actor/critic networks may still use this switch. Structured
    # schema networks use the explicit switches below so raw physical features
    # are not normalized across heterogeneous fields by accident.
    input_norm_enabled: bool = True
    structured_actor_input_norm_enabled: bool = False
    structured_critic_input_norm_enabled: bool = False
    structured_actor_backbone: str = "topology_aware"  # "topology_aware" | "flat_mlp"
    kl_coef: float = 0.0
    target_kl: float = 0.0
    kl_stop: bool = False
    ppo_headwise_surrogate_enabled: bool = False
    ppo_headwise_kl_stop_enabled: bool = True
    ppo_per_head_advantage_enabled: bool = False
    bw_danger_tighter_clip_enabled: bool = False
    bw_danger_clip_ratio: float = 0.1
    bw_danger_adv_quantile: float = 0.5
    bw_danger_valid_count_quantile: float = 0.5
    bw_danger_queue_max_quantile: float = 0.5
    bw_false_positive_mask_enabled: bool = False
    bw_false_positive_horizon: int = 5
    bw_false_positive_bad_tau: float = 1.0
    bw_continuous_credit_enabled: bool = False
    bw_continuous_credit_alpha: float = 0.0
    bw_continuous_credit_horizon: int = 5
    bw_continuous_credit_renorm_enabled: bool = True
    bw_head_grad_scale: float = 1.0
    bw_actor_advantage_override_mode: str = "gae"  # "gae" | "branch_delta" | "true_adv_mc" | "delta_critic" | "delta_teacher_student"
    bw_actor_branch_horizon: int = 10
    bw_actor_branch_samples: int = 1
    bw_actor_branch_ref_mode: str = "deterministic"  # "deterministic"
    bw_actor_branch_follow_policy_mode: str = "stochastic"  # "deterministic" | "stochastic"
    bw_actor_branch_normalize: bool = False
    bw_actor_branch_parallel_envs: int = 0
    bw_actor_branch_parallel_backend: str = "sync"
    bw_actor_branch_worker_fused_rollout_enabled: bool = False
    bw_actor_true_mc_horizon: int = 10
    bw_actor_true_mc_samples: int = 4
    bw_delta_critic_enabled: bool = False
    bw_delta_critic_horizon: int = 3
    bw_delta_critic_samples: int = 1
    bw_delta_critic_ref_mode: str = "deterministic"  # "deterministic"
    bw_delta_critic_follow_policy_mode: str = "deterministic"  # "deterministic" | "stochastic"
    bw_delta_critic_loss_coef: float = 1.0
    bw_delta_critic_warmup_epochs: int = 1
    bw_delta_teacher_student_corr_low: float = 0.6
    bw_delta_teacher_student_corr_high: float = 0.9
    bw_delta_teacher_student_mix_power: float = 1.0
    bw_delta_teacher_student_disable_dense_teacher_when_ready: bool = False
    bw_delta_teacher_student_ready_patience: int = 2
    bw_delta_teacher_student_probe_interval_updates: int = 0
    bw_delta_teacher_student_probe_reenable_corr: float = 0.85
    structured_bw_freeze_kappa: bool = False
    bw_actor_branch_gate_snr_threshold: float = 0.0
    critic_epochs: int | None = None
    critic_warmup_before_actor_epochs: int = 0
    critic_warmup_recompute_advantages: bool = False
    critic_warmup_recompute_mode: str = "advantage_only"  # "advantage_only" | "gae"
    critic_loss_target_standardize: bool = False
    critic_loss_running_standardize: bool = False
    critic_loss_running_standardize_decay: float = 0.99
    critic_replay_bank_enabled: bool = False
    critic_replay_bank_capacity: int = 0
    critic_popart_enabled: bool = False
    critic_popart_beta: float = 0.999
    critic_kl_stop_coupled: bool = True
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
    danger_imitation_trigger_mode: str = "intervention_any"
    danger_imitation_close_risk_thresh: float = 0.05
    danger_imitation_intervention_thresh: float = 0.05
    train_accel: bool | None = None
    train_bw: bool | None = None
    train_sat: bool | None = None
    sat_clean_joint_enabled: bool = False
    sat_clean_topm_per_uav: int = 4
    sat_clean_contexts_per_update: int = 16
    sat_clean_entropy_topk_per_env: int = 2
    sat_clean_uniform_contexts_per_env: int = 1
    sat_clean_parallel_envs: int = 0
    sat_clean_parallel_backend: str = "sync"
    sat_clean_tasks_per_worker: int = 8
    sat_clean_positive_gap_eps: float = 1.0e-6
    sat_supervision_enabled: bool = False
    sat_supervision_coef: float = 0.0
    sat_supervision_top1_coef: float = 1.0
    sat_supervision_topk_coef: float = 0.5
    sat_counterfactual_credit_enabled: bool = False
    sat_counterfactual_credit_samples_per_update: int = 0
    sat_counterfactual_credit_weight: float = 1.0
    bw_counterfactual_credit_enabled: bool = False
    bw_counterfactual_credit_weight: float = 1.0
    bw_marginal_teacher_sample_enabled: bool = False
    bw_marginal_teacher_sample_weight: float = 1.0
    structured_bw_policy_update_mode: str = "ppo"  # "ppo" | "awr" | "det_awr" | "vmpo_lite"
    structured_actor_update_mode: str = "ppo"  # "ppo" | "vs_ref"
    accel_update_mode: str | None = None  # None inherits structured_actor_update_mode
    sat_update_mode: str | None = None
    bw_update_mode: str | None = None
    vs_ref_rows_per_update: int = 32
    vs_ref_samples_per_row: int = 1
    vs_ref_horizon_mode: str = "episode_remaining"  # currently only finite episode remainder
    vs_ref_advantage_normalize: str = "stage"  # "none" | "stage"
    vs_ref_ref_policy: str = "deterministic_current"
    vs_ref_follow_policy: str = "deterministic_current"
    vs_ref_disable_critic: bool = True
    vs_ref_sampling_mode: str = "uniform"  # "uniform" | "active_mixture"
    vs_ref_sampling_alpha: float = 0.6
    vs_ref_sampling_random_frac: float = 0.25
    vs_ref_sampling_time_frac: float = 0.25
    vs_ref_sampling_leverage_frac: float = 0.25
    vs_ref_sampling_uncertainty_frac: float = 0.25
    vs_ref_sampling_cost_power: float = 0.5
    structured_bw_awr_temperature: float = 0.5
    structured_bw_awr_max_weight: float = 20.0
    structured_bw_awr_normalize_weights: bool = True
    structured_bw_awr_kl_coef: float = 0.0
    structured_bw_vmpo_temperature: float = 0.5
    structured_bw_vmpo_top_frac: float = 0.5
    structured_bw_vmpo_kl_coef: float = 0.01
    bw_flow_proxy_aux_enabled: bool = False
    bw_flow_proxy_aux_coef: float = 0.02
    bw_flow_proxy_aux_delta: float = 0.05
    bw_flow_proxy_base_action_mode: str = "executed"  # "executed" | "deterministic"; external override needs a fixed native producer
    bw_flow_proxy_aux_min_gap: float = 1e-4
    bw_flow_proxy_aux_regression_coef: float = 0.0
    bw_flow_proxy_grad_diagnostics_enabled: bool = False
    step_train_target_mode: str = "env_reward"  # "env_reward" | "access_term" | "access_raw"
    bw_train_target_mode: str = "env_reward"  # "env_reward" | "access_term" | "access_raw" | "weighted_workload_delta" | "weighted_workload_level" | "gu_queue_level" | "system_queue_level" | "gu_service_queue"
    # Access association and BW allocation macro-decision interval in primitive
    # environment steps.  A value of 1 still goes through the macro path and
    # should reduce exactly to the current per-step behavior.
    access_bw_decision_interval: int = 1
    # SAT selection macro-decision interval in primitive environment steps.
    # Like access_bw_decision_interval, K=1 still exercises the macro path.
    sat_decision_interval: int = 1
    # BW-only step-level return target. "step_lambda_return" is the old
    # "monte_carlo"/"mc" mode: BW uses r_t + gamma * next_step_lambda_return,
    # so future terms can still contain value bootstrap when gae_lambda < 1.
    # Use "bw_episode_mc" for the BW reward chain with no intermediate lambda
    # bootstrap except rollout-tail bootstrap. By default T_steps is treated
    # as a finite-horizon episode boundary, so time-limit truncation does not
    # bootstrap into the next reset episode. Set time_limit_bootstrap_enabled
    # only for continuing-task TimeLimit experiments.
    # "bw_nstep" for a short-horizon BW state value target:
    #   sum_{i=0}^{h-1} gamma^i r_{t+i}^{bw} + gamma^h V_bw(s_{t+h}).
    bw_return_mode: str = "gae"  # "gae" | "bw_gae" | "step_lambda_return" | "bw_episode_mc" | "bw_nstep"
    bw_nstep_horizon: int = 3
    time_limit_bootstrap_enabled: bool = False
    structured_step_bootstrap_stage: str = "accel"  # "accel" | "bw"
    bw_weighted_workload_ema_decay: float = 0.95
    service_floor_bits_per_step: float | None = None
    bw_weighted_workload_eps: float = 1.0
    actor_advantage_normalize_enabled: bool = True
    stagewise_advantage_norm_enabled: bool = True
    # Generic single-stage MC-critic + A_gae(V) training loop knobs.  These are
    # read by scripts/train_stage_mcgae.py; the regular PPO trainer ignores
    # them.  The stage-specific SAT keys below remain for compatibility with
    # scripts/train_sat_mcgae.py and older temporary configs.
    stage_mcgae_cold_critic_lr: float | None = None
    stage_mcgae_cold_critic_epochs: int | None = None
    stage_mcgae_tracking_critic_lr: float | None = None
    stage_mcgae_tracking_critic_epochs: int | None = None
    stage_mcgae_critic_minibatches: int | None = None
    stage_mcgae_critic_update_microbatch_size: int | None = None
    stage_mcgae_critic_eval_before_enabled: bool | None = None
    stage_mcgae_critic_ev_gate_enabled: bool | None = None
    stage_mcgae_critic_ev_soft_target: float | None = None
    stage_mcgae_critic_ev_hard_floor: float | None = None
    stage_mcgae_critic_ev_extra_epochs: int | None = None
    stage_mcgae_critic_ev_max_retries: int | None = None
    stage_mcgae_actor_lr: float | None = None
    stage_mcgae_accel_actor_lr: float | None = None
    stage_mcgae_sat_actor_lr: float | None = None
    stage_mcgae_bw_actor_lr: float | None = None
    stage_mcgae_actor_epochs: int | None = None
    stage_mcgae_actor_minibatches: int | None = None
    stage_actor_logprob_parity_check_enabled: bool = True
    stage_actor_logprob_parity_abs_tol: float = 1.0e-3
    stage_actor_logprob_parity_rel_tol: float = 1.0e-4
    stage_actor_kl_early_stop_enabled: bool = False
    stage_actor_target_kl: float = 0.02
    stage_actor_target_kl_accel: float = 0.02
    stage_actor_target_kl_sat: float = 0.02
    stage_actor_target_kl_bw: float = 0.02
    stage_actor_kl_stop_multiplier: float = 1.5
    stage_actor_dynamic_lr_enabled: bool = False
    stage_actor_lr_decay_factor: float = 0.5
    stage_actor_lr_grow_factor: float = 1.25
    stage_actor_lr_decay_patience: int = 3
    stage_actor_lr_grow_patience: int = 10
    stage_actor_lr_high_kl_frac: float = 1.5
    stage_actor_lr_low_kl_frac: float = 0.3
    stage_actor_lr_high_clip: float = 0.3
    stage_actor_lr_low_clip: float = 0.05
    stage_actor_lr_hard_kl_frac: float = 3.0
    stage_actor_lr_hard_clip: float = 0.6
    stage_actor_lr_min: float = 1.0e-5
    stage_actor_lr_min_accel: float = 3.0e-6
    stage_actor_lr_min_sat: float = 1.0e-5
    stage_actor_lr_min_bw: float = 1.0e-5
    stage_actor_lr_max: float = 2.0e-3
    stage_actor_lr_max_accel: float = 6.0e-4
    stage_actor_lr_max_sat: float = 2.0e-3
    stage_actor_lr_max_bw: float = 2.0e-3
    # Dedicated SAT MC-critic + A_gae(V) training loop knobs.
    sat_mcgae_cold_critic_lr: float = 1.0e-3
    sat_mcgae_cold_critic_epochs: int = 20
    sat_mcgae_tracking_critic_lr: float = 3.0e-4
    sat_mcgae_tracking_critic_epochs: int = 5
    sat_mcgae_critic_minibatches: int = 8
    sat_mcgae_critic_update_microbatch_size: int = 0
    sat_mcgae_critic_eval_before_enabled: bool = False
    sat_mcgae_actor_lr: float = 3.0e-4
    sat_mcgae_actor_epochs: int = 5
    sat_mcgae_actor_minibatches: int = 1
    stage_actor_compile_enabled: bool = True
    accel_actor_compile_enabled: bool = True
    sat_actor_compile_enabled: bool = True
    bw_actor_compile_enabled: bool = False
    # Internal memory microbatch for PPO actor evaluation/update. This does not
    # change PPO minibatch semantics: gradients are accumulated over these
    # chunks and optimizer.step() is still called once per PPO minibatch.
    actor_update_microbatch_size: int = 1024
    train_trace_enabled: bool = True
    train_trace_rollout_interval: int = 0
    train_shared_backbone: bool = True
    train_fusion: bool = False
    train_fusion_last_layer: bool = False
    obs_own_include_uav_id_norm: bool = False
    exec_accel_source: str = "policy"  # policy|teacher|queue_aware|cluster_center_queue_aware|zero
    exec_bw_source: str = "policy"  # policy|teacher|queue_aware|cluster_center_queue_aware|zero
    exec_sat_source: str = "policy"  # policy|teacher|queue_aware|cluster_center_queue_aware|zero
    bw_single_uav_policy_uav_id: int = 0
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
    aux_schedule_use_planned_total_updates: bool = False

    actor_hidden: int = 256
    actor_encoder_type: str = "flat_mlp"  # "flat_mlp" or "set_pool"
    actor_set_embed_dim: int = 128
    accel_hidden: int = 0
    accel_embed_dim: int = 0
    sat_hidden: int = 0
    sat_embed_dim: int = 0
    bw_hidden: int = 0
    bw_embed_dim: int = 0
    actor_encoder_mlp_layers: int = 2
    actor_context_mlp_layers: int = 2
    actor_head_mlp_layers: int = 2
    accel_encoder_mlp_layers: int = 0
    accel_context_mlp_layers: int = 0
    accel_head_mlp_layers: int = 1
    accel_interaction_layers: int = 0
    accel_attention_heads: int = 4
    accel_gu_query_count: int = 0
    accel_peer_query_count: int = 0
    accel_sat_query_count: int = 0
    sat_encoder_mlp_layers: int = 0
    sat_context_mlp_layers: int = 0
    sat_head_mlp_layers: int = 0
    bw_encoder_mlp_layers: int = 0
    bw_context_mlp_layers: int = 0
    bw_head_mlp_layers: int = 0
    bw_private_trunk_enabled: bool = False
    bw_private_trunk_init_from_shared: bool = True
    critic_embed_dim: int = 128
    critic_edge_embed_dim: int = 128
    critic_global_embed_dim: int = 128
    critic_system_token_dim: int = 128
    critic_hidden: int = 256
    critic_encoder_mlp_layers: int = 2
    critic_message_mlp_layers: int = 2
    critic_message_layers: int = 1
    critic_fixed_bounded_relations_enabled: bool = True
    critic_compile_enabled: bool = True
    critic_compile_fullgraph: bool = True
    critic_value_head_hidden: int = 256
    critic_value_head_layers: int = 2
    critic_value_mode: str = "relational"  # "relational" | "global_only" | "global_linear" | "flat_mlp"
    critic_stage_specific_paths_enabled: bool = False
    critic_sat_hidden: int = 0
    critic_sat_embed_dim: int = 0
    critic_sat_encoder_mlp_layers: int = 0
    critic_sat_message_mlp_layers: int = 0
    critic_sat_message_layers: int = 0
    critic_sat_value_head_hidden: int = 0
    critic_sat_value_head_layers: int = 0
    critic_global_linear_fit_ridge: float = 1.0e-6
    critic_global_linear_fit_decay: float = 0.90
    critic_global_linear_fit_recompute_advantages: bool = True
    critic_value_agg: str = "agent_mean"  # "agent_mean" or "pooled_feature"
    critic_multihead_value_enabled: bool = False
    critic_head_feature_mode: str = "shared_full"
    critic_global_feature_enabled: bool = False

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
    checkpoint_eval_fixed_policy: str = "zero"  # "zero" | "queue_aware" | "queue_aware_bw" | "cluster_center_queue_aware" | "teacher_accel_queue_aware" | "stage2_exec_fixed_sat"
    checkpoint_eval_policy_mode: str = "deterministic"  # "deterministic" | "stochastic"
    checkpoint_eval_min_stop_update: int = 0
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
    checkpoint_eval_stop_on_early_stop: bool = False
    checkpoint_eval_save_best_models: bool = False
    checkpoint_eval_use_best_as_final: bool = False
    update_direction_probe_enabled: bool = False
    update_direction_probe_interval_updates: int = 0
    update_direction_probe_start_update: int = 0
    update_direction_probe_panel_episodes: int = 6
    update_direction_probe_panel_states: int = 16
    update_direction_probe_panel_seed: int = 35000
    update_direction_probe_k_steps: int = 20
    update_direction_probe_bw_sample_limit: int = 16
    update_direction_probe_actor_policy_mode: str = "deterministic"  # "deterministic" | "stochastic"
    update_direction_probe_actor_policy_samples: int = 1
    update_direction_probe_true_mc_enabled: bool = False
    update_direction_probe_true_mc_samples: int = 4
    update_direction_probe_branch_enabled: bool = False
    update_direction_probe_branch_horizons: list[int] = field(default_factory=lambda: [2, 5, 10])
    update_direction_probe_branch_samples: int = 2
    update_direction_probe_branch_ref_mode: str = "deterministic"  # "deterministic"
    update_direction_probe_branch_follow_policy_mode: str = "stochastic"  # "deterministic" | "stochastic"

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Compiled structured-kernel callables are process-local and may capture
        # non-picklable closures. Never propagate that cache across subprocesses.
        state.pop("_structured_kernel_runtime_cache", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def theta_min_rad(self) -> float:
        return math.radians(self.theta_min_deg)


def access_carrier_freq_from_config(cfg: Any) -> float:
    fallback = float(getattr(cfg, "carrier_freq", 2.0e9) or 2.0e9)
    return max(float(getattr(cfg, "access_carrier_freq", fallback) or fallback), 1.0)


def backhaul_carrier_freq_from_config(cfg: Any) -> float:
    fallback = float(getattr(cfg, "carrier_freq", 2.0e9) or 2.0e9)
    return max(float(getattr(cfg, "backhaul_carrier_freq", fallback) or fallback), 1.0)


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
        if not path and k == "stage_actor_lr_clip_threshold":
            k = "stage_actor_lr_high_clip"
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
    return base_rate * _traffic_level_ratio(cfg) * max(float(cfg.num_gu), 1.0) * float(cfg.tau0)


def _require_positive_config_float(value: float, name: str) -> float:
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value_f!r}.")
    return value_f


def _effective_b_backhaul_per_sat_from_cfg(cfg: SaginConfig) -> float:
    bandwidth = getattr(cfg, "b_backhaul_per_sat", 0.0)
    scale = getattr(cfg, "b_backhaul_per_sat_scale", 1.0)
    return float(bandwidth) * float(scale)


def _normalize_safety_shield_config(cfg: SaginConfig) -> None:
    cfg.safety_shield_enabled = bool(getattr(cfg, "safety_shield_enabled", False))
    cfg.safety_shield_solver = str(getattr(cfg, "safety_shield_solver", "CLARABEL") or "CLARABEL").strip().upper()
    cfg.safety_shield_julia_exe = str(getattr(cfg, "safety_shield_julia_exe", "") or "").strip()
    cfg.safety_shield_julia_project = str(getattr(cfg, "safety_shield_julia_project", "") or "").strip()
    cfg.safety_shield_julia_depot = str(getattr(cfg, "safety_shield_julia_depot", "") or "").strip()
    cfg.safety_shield_brake_rho = max(float(getattr(cfg, "safety_shield_brake_rho", 0.8) or 0.8), 1.0e-6)
    cfg.safety_shield_a_safe = max(float(getattr(cfg, "safety_shield_a_safe", 0.0) or 0.0), 0.0)
    cfg.safety_shield_distance_buffer = max(
        float(getattr(cfg, "safety_shield_distance_buffer", 0.0) or 0.0),
        0.0,
    )
    cfg.safety_shield_tolerance = max(float(getattr(cfg, "safety_shield_tolerance", 1.0e-5) or 1.0e-5), 0.0)
    cfg.safety_shield_relax_on_infeasible = bool(getattr(cfg, "safety_shield_relax_on_infeasible", True))
    cfg.safety_shield_relax_slack_weight = max(
        float(getattr(cfg, "safety_shield_relax_slack_weight", 1.0e4) or 1.0e4),
        1.0e-6,
    )
    cfg.safety_shield_relax_action_weight = max(
        float(getattr(cfg, "safety_shield_relax_action_weight", 1.0) or 1.0),
        0.0,
    )
    cfg.safety_shield_tensor_iters = max(int(getattr(cfg, "safety_shield_tensor_iters", 8) or 8), 1)
    cfg.safety_shield_tensor_step_gain = max(
        float(getattr(cfg, "safety_shield_tensor_step_gain", 1.0) or 1.0),
        0.0,
    )
    cfg.safety_shield_verbose = bool(getattr(cfg, "safety_shield_verbose", False))


def ensure_structured_accel_actor_config(cfg: SaginConfig) -> None:
    """Normalize and validate config fields required by the structured accel actor."""
    if getattr(cfg, "b_sat_total", None) is not None:
        cfg.b_backhaul_per_sat = float(cfg.b_sat_total)
    if getattr(cfg, "b_sat_total_scale", None) is not None:
        cfg.b_backhaul_per_sat_scale = float(cfg.b_sat_total_scale)
    cfg.b_backhaul_per_sat = max(float(getattr(cfg, "b_backhaul_per_sat", 0.0) or 0.0), 0.0)
    cfg.b_backhaul_per_sat_scale = max(float(getattr(cfg, "b_backhaul_per_sat_scale", 1.0) or 1.0), 0.0)
    cfg.b_sat_total = float(cfg.b_backhaul_per_sat)
    cfg.b_sat_total_scale = float(cfg.b_backhaul_per_sat_scale)
    cfg.access_carrier_freq = access_carrier_freq_from_config(cfg)
    cfg.backhaul_carrier_freq = backhaul_carrier_freq_from_config(cfg)
    cfg.access_noise_figure_db = max(float(getattr(cfg, "access_noise_figure_db", 5.0) or 0.0), 0.0)
    cfg.backhaul_noise_figure_db = max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0)
    cfg.noise_density = _require_positive_config_float(float(getattr(cfg, "noise_density", 0.0) or 0.0), "noise_density")
    if getattr(cfg, "service_floor_bits_per_step", None) is None:
        cfg.service_floor_bits_per_step = float(getattr(cfg, "bw_weighted_workload_eps", 1.0) or 1.0)
    cfg.service_floor_bits_per_step = _require_positive_config_float(
        float(getattr(cfg, "service_floor_bits_per_step", 1.0) or 0.0),
        "service_floor_bits_per_step",
    )
    cfg.bw_weighted_workload_eps = float(cfg.service_floor_bits_per_step)

    if int(getattr(cfg, "num_uav", 0)) <= 0:
        raise ValueError(f"num_uav must be positive, got {getattr(cfg, 'num_uav', None)!r}.")
    if int(getattr(cfg, "num_gu", 0)) < 0:
        raise ValueError(f"num_gu must be non-negative, got {getattr(cfg, 'num_gu', None)!r}.")
    if int(getattr(cfg, "num_sat", 0)) <= 0:
        raise ValueError(f"num_sat must be positive, got {getattr(cfg, 'num_sat', None)!r}.")
    cfg.users_obs_max = int(getattr(cfg, "users_obs_max", 0) or 0)
    cfg.sats_obs_max = int(getattr(cfg, "sats_obs_max", 0) or 0)
    if cfg.users_obs_max <= 0:
        raise ValueError(f"users_obs_max must be positive, got {cfg.users_obs_max!r}.")
    if cfg.users_obs_max < int(cfg.num_gu):
        cfg.users_obs_max = int(cfg.num_gu)
    raw_candidate_k = getattr(cfg, "candidate_k", None)
    if raw_candidate_k is not None and int(raw_candidate_k) > 0 and int(raw_candidate_k) < int(cfg.num_gu):
        cfg.candidate_k = int(cfg.num_gu)
        raw_candidate_k = cfg.candidate_k
    effective_candidate_k = cfg.users_obs_max if raw_candidate_k is None or int(raw_candidate_k) <= 0 else min(int(raw_candidate_k), cfg.users_obs_max)
    if effective_candidate_k <= 0:
        raise ValueError(f"effective candidate_k must be positive, got {effective_candidate_k!r}.")
    if effective_candidate_k < int(cfg.num_gu):
        raise ValueError(
            "effective candidate_k must cover all GU for structured accel actor; "
            f"got effective_candidate_k={effective_candidate_k}, num_gu={cfg.num_gu}."
        )
    candidate_mode = str(getattr(cfg, "candidate_mode", "assoc") or "assoc").strip().lower()
    if candidate_mode not in {"assoc", "nearest"}:
        raise ValueError(f'candidate_mode must be "assoc" or "nearest" for structured accel actor, got {candidate_mode!r}.')
    cfg.candidate_mode = candidate_mode

    cfg.N_RF = int(getattr(cfg, "N_RF", 0) or 0)
    if cfg.N_RF <= 0:
        raise ValueError(f"N_RF must be positive, got {cfg.N_RF!r}.")
    sat_select_cfg = getattr(cfg, "sat_num_select", None)
    sat_select_count = int(sat_select_cfg) if sat_select_cfg is not None and int(sat_select_cfg) > 0 else int(cfg.N_RF)
    sat_select_ref_count = min(int(cfg.num_sat), int(cfg.N_RF), sat_select_count)
    if sat_select_ref_count <= 0:
        raise ValueError("sat_select_ref_count must be positive.")
    cfg.sat_action_select_k = int(sat_select_ref_count)

    visible_ref = getattr(cfg, "visible_sats_max", None)
    visible_width = int(visible_ref) if visible_ref is not None else int(cfg.sats_obs_max)
    derived_sat_width = min(int(cfg.num_sat), visible_width)
    cfg.per_uav_visible_sat_token_max = derived_sat_width
    if int(getattr(cfg, "per_uav_visible_sat_token_max", 0) or 0) <= 0:
        raise ValueError(
            "per_uav_visible_sat_token_max must be positive; "
            f"got {cfg.per_uav_visible_sat_token_max!r} from num_sat={cfg.num_sat}, visible_sats_max={visible_ref!r}, sats_obs_max={cfg.sats_obs_max}."
        )

    _require_positive_config_float(_default_queue_ref_per_step(cfg), "arrival_ref_bits_per_step")
    for attr in (
        "map_size",
        "v_max",
        "a_max",
        "uav_energy_init",
        "queue_max_gu",
        "queue_max_uav",
        "queue_max_sat",
        "b_acc",
        "task_cycles_per_bit",
        "speed_of_light",
    ):
        _require_positive_config_float(float(getattr(cfg, attr, 0.0) or 0.0), attr)
    if float(getattr(cfg, "tau0", 0.0) or 0.0) <= 0.0:
        raise ValueError(f"tau0 must be positive, got {getattr(cfg, 'tau0', None)!r}.")
    _normalize_safety_shield_config(cfg)
    _require_positive_config_float(float(cfg.r_earth + cfg.sat_height), "orbit_pos_ref")
    _require_positive_config_float(_effective_b_backhaul_per_sat_from_cfg(cfg), "effective_b_backhaul_per_sat")
    if bool(getattr(cfg, "doppler_enabled", False) or getattr(cfg, "doppler_atten_enabled", False) or getattr(cfg, "doppler_observed", False)):
        _require_positive_config_float(float(getattr(cfg, "nu_max", 0.0) or 0.0), "nu_max")


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


def _resource_scale_active_sat_count(cfg: SaginConfig, reference: bool = False) -> float:
    if reference:
        active_count = getattr(cfg, "resource_scale_ref_sat_active_count", None)
        if active_count is not None:
            return max(float(active_count), 1.0)
    else:
        active_count = getattr(cfg, "resource_scale_sat_active_count", None)
        if active_count is not None:
            return max(float(active_count), 1.0)
        active_count = getattr(cfg, "queue_ref_sat_active_count", None)
        if active_count is not None:
            return max(float(active_count), 1.0)

    sat_num_select = getattr(cfg, "sat_num_select", None)
    if sat_num_select is not None and int(sat_num_select) > 0:
        sat_k_raw = sat_num_select
    else:
        sat_k_raw = getattr(cfg, "sat_action_select_k", None)
        if sat_k_raw is None or int(sat_k_raw) <= 0:
            sat_k_raw = getattr(cfg, "N_RF", 0)
    sat_k = max(int(sat_k_raw or getattr(cfg, "N_RF", 0)), 0)
    if sat_k > 0 and cfg.num_uav > 0:
        return max(min(float(cfg.num_sat), float(sat_k * cfg.num_uav)), 1.0)
    return max(float(cfg.num_sat), 1.0)


def _apply_resource_scaling(cfg: SaginConfig) -> None:
    if not bool(getattr(cfg, "resource_scale_enabled", False)):
        return

    ref_num_uav = max(float(getattr(cfg, "resource_scale_ref_num_uav", None) or cfg.num_uav), 1.0)
    ref_num_gu = max(float(getattr(cfg, "resource_scale_ref_num_gu", None) or cfg.num_gu), 1.0)
    ref_arrival = max(
        float(getattr(cfg, "resource_scale_ref_task_arrival_rate", None) or getattr(cfg, "task_arrival_rate", 0.0) or 0.0),
        0.0,
    )
    cur_arrival = max(float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0), 0.0)

    cur_gu_per_uav = cur_arrival * max(float(cfg.num_gu), 0.0) / max(float(cfg.num_uav), 1.0)
    ref_gu_per_uav = ref_arrival * ref_num_gu / ref_num_uav
    ref_gu_per_uav = _require_positive_config_float(ref_gu_per_uav, "resource_scale reference GU arrival per UAV")
    acc_scale = cur_gu_per_uav / ref_gu_per_uav
    acc_multiplier = max(float(getattr(cfg, "resource_scale_b_acc_multiplier", 1.0) or 0.0), 0.0)
    cfg.b_acc = max(float(cfg.b_acc), 0.0) * acc_scale * acc_multiplier

    cur_active_sat = _resource_scale_active_sat_count(cfg, reference=False)
    ref_active_sat = _resource_scale_active_sat_count(cfg, reference=True)
    cur_gu_per_sat = cur_arrival * max(float(cfg.num_gu), 0.0) / max(cur_active_sat, 1.0)
    ref_gu_per_sat = ref_arrival * ref_num_gu / max(ref_active_sat, 1.0)
    ref_gu_per_sat = _require_positive_config_float(ref_gu_per_sat, "resource_scale reference GU arrival per active satellite")
    sat_scale = cur_gu_per_sat / ref_gu_per_sat
    cfg.b_backhaul_per_sat = max(float(cfg.b_backhaul_per_sat), 0.0) * sat_scale
    cfg.sat_cpu_freq = max(float(cfg.sat_cpu_freq), 0.0) * sat_scale


def _finalize_config(cfg: SaginConfig) -> None:
    if getattr(cfg, "b_sat_total", None) is not None:
        cfg.b_backhaul_per_sat = float(cfg.b_sat_total)
    if getattr(cfg, "b_sat_total_scale", None) is not None:
        cfg.b_backhaul_per_sat_scale = float(cfg.b_sat_total_scale)
    cfg.b_backhaul_per_sat = max(float(getattr(cfg, "b_backhaul_per_sat", 0.0) or 0.0), 0.0)
    cfg.b_backhaul_per_sat_scale = max(float(getattr(cfg, "b_backhaul_per_sat_scale", 1.0) or 1.0), 0.0)
    # Keep legacy attributes readable for older tests/scripts, but do not use
    # them as the semantic source of truth after finalization.
    cfg.b_sat_total = float(cfg.b_backhaul_per_sat)
    cfg.b_sat_total_scale = float(cfg.b_backhaul_per_sat_scale)
    cfg.access_carrier_freq = access_carrier_freq_from_config(cfg)
    cfg.backhaul_carrier_freq = backhaul_carrier_freq_from_config(cfg)
    cfg.access_noise_figure_db = max(float(getattr(cfg, "access_noise_figure_db", 5.0) or 0.0), 0.0)
    cfg.backhaul_noise_figure_db = max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0)
    cfg.noise_density = _require_positive_config_float(float(getattr(cfg, "noise_density", 0.0) or 0.0), "noise_density")
    if getattr(cfg, "service_floor_bits_per_step", None) is None:
        cfg.service_floor_bits_per_step = float(getattr(cfg, "bw_weighted_workload_eps", 1.0) or 1.0)
    cfg.service_floor_bits_per_step = _require_positive_config_float(
        float(getattr(cfg, "service_floor_bits_per_step", 1.0) or 0.0),
        "service_floor_bits_per_step",
    )
    cfg.bw_weighted_workload_eps = float(cfg.service_floor_bits_per_step)
    actor_backbone = str(getattr(cfg, "structured_actor_backbone", "topology_aware") or "topology_aware").strip().lower()
    if actor_backbone in {"structured", "staged", "topology", "topology-aware", "topology_aware_staged"}:
        actor_backbone = "topology_aware"
    if actor_backbone in {"flat", "mappo_like", "mappo-like", "flat_actor"}:
        actor_backbone = "flat_mlp"
    if actor_backbone not in {"topology_aware", "flat_mlp"}:
        raise ValueError("structured_actor_backbone must be one of {'topology_aware', 'flat_mlp'}.")
    cfg.structured_actor_backbone = actor_backbone
    mode = str(getattr(cfg, "access_fading_mode", "ergodic_rician") or "ergodic_rician").strip().lower()
    if mode in {"none", "off", "disabled"}:
        mode = "large_scale"
    if mode not in {"large_scale", "ergodic_rician", "iid_rician"}:
        raise ValueError(f"Unsupported access_fading_mode: {mode}")
    cfg.access_fading_mode = mode
    cfg.access_ergodic_rician_quadrature_points = max(
        int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
        1,
    )
    _require_positive_config_float(_default_queue_ref_per_step(cfg), "arrival_ref_bits_per_step")
    _apply_resource_scaling(cfg)
    cfg.b_sat_total = float(cfg.b_backhaul_per_sat)
    cfg.b_sat_total_scale = float(cfg.b_backhaul_per_sat_scale)
    if int(getattr(cfg, "num_uav", 0)) <= 0:
        raise ValueError(f"num_uav must be positive, got {getattr(cfg, 'num_uav', None)!r}.")
    if int(getattr(cfg, "num_gu", 0)) < 0:
        raise ValueError(f"num_gu must be non-negative, got {getattr(cfg, 'num_gu', None)!r}.")
    if int(getattr(cfg, "num_sat", 0)) <= 0:
        raise ValueError(f"num_sat must be positive, got {getattr(cfg, 'num_sat', None)!r}.")
    cfg.users_obs_max = int(getattr(cfg, "users_obs_max", 0) or 0)
    cfg.sats_obs_max = int(getattr(cfg, "sats_obs_max", 0) or 0)
    if cfg.users_obs_max <= 0:
        raise ValueError(f"users_obs_max must be positive, got {cfg.users_obs_max!r}.")
    if cfg.users_obs_max < int(cfg.num_gu):
        cfg.users_obs_max = int(cfg.num_gu)
    raw_candidate_k = getattr(cfg, "candidate_k", None)
    if raw_candidate_k is not None and int(raw_candidate_k) > 0 and int(raw_candidate_k) < int(cfg.num_gu):
        cfg.candidate_k = int(cfg.num_gu)
        raw_candidate_k = cfg.candidate_k
    effective_candidate_k = cfg.users_obs_max if raw_candidate_k is None or int(raw_candidate_k) <= 0 else min(int(raw_candidate_k), cfg.users_obs_max)
    if effective_candidate_k <= 0:
        raise ValueError(f"effective candidate_k must be positive, got {effective_candidate_k!r}.")
    if effective_candidate_k < int(cfg.num_gu):
        raise ValueError(
            "effective candidate_k must cover all GU for structured accel actor; "
            f"got effective_candidate_k={effective_candidate_k}, num_gu={cfg.num_gu}."
        )
    candidate_mode = str(getattr(cfg, "candidate_mode", "assoc") or "assoc").strip().lower()
    if candidate_mode not in {"assoc", "nearest"}:
        raise ValueError(f'candidate_mode must be "assoc" or "nearest" for structured accel actor, got {candidate_mode!r}.')
    cfg.candidate_mode = candidate_mode
    actor_update_mode = str(getattr(cfg, "structured_actor_update_mode", "ppo") or "ppo").strip().lower()
    if actor_update_mode not in {"ppo", "vs_ref"}:
        raise ValueError("structured_actor_update_mode must be one of {'ppo', 'vs_ref'}.")
    cfg.structured_actor_update_mode = actor_update_mode
    for attr in ("accel_update_mode", "sat_update_mode", "bw_update_mode"):
        raw_mode = getattr(cfg, attr, None)
        if raw_mode is None or str(raw_mode).strip() == "":
            setattr(cfg, attr, None)
            continue
        mode_value = str(raw_mode).strip().lower()
        if mode_value not in {"ppo", "vs_ref"}:
            raise ValueError(f"{attr} must be one of {{'ppo', 'vs_ref'}} when set.")
        setattr(cfg, attr, mode_value)
    cfg.vs_ref_rows_per_update = max(int(getattr(cfg, "vs_ref_rows_per_update", 32) or 32), 1)
    cfg.vs_ref_samples_per_row = max(int(getattr(cfg, "vs_ref_samples_per_row", 1) or 1), 1)
    cfg.vs_ref_horizon_mode = str(
        getattr(cfg, "vs_ref_horizon_mode", "episode_remaining") or "episode_remaining"
    ).strip().lower()
    if cfg.vs_ref_horizon_mode not in {"episode_remaining"}:
        raise ValueError("vs_ref_horizon_mode currently supports only 'episode_remaining'.")
    cfg.vs_ref_advantage_normalize = str(
        getattr(cfg, "vs_ref_advantage_normalize", "stage") or "stage"
    ).strip().lower()
    if cfg.vs_ref_advantage_normalize not in {"none", "stage"}:
        raise ValueError("vs_ref_advantage_normalize must be one of {'none', 'stage'}.")
    for attr in ("vs_ref_ref_policy", "vs_ref_follow_policy"):
        value = str(getattr(cfg, attr, "deterministic_current") or "deterministic_current").strip().lower()
        if value not in {"deterministic_current"}:
            raise ValueError(f"{attr} currently supports only 'deterministic_current'.")
        setattr(cfg, attr, value)
    cfg.vs_ref_sampling_mode = str(
        getattr(cfg, "vs_ref_sampling_mode", "uniform") or "uniform"
    ).strip().lower()
    if cfg.vs_ref_sampling_mode not in {"uniform", "active_mixture"}:
        raise ValueError("vs_ref_sampling_mode must be one of {'uniform', 'active_mixture'}.")
    cfg.vs_ref_sampling_alpha = min(
        max(float(getattr(cfg, "vs_ref_sampling_alpha", 0.6) or 0.0), 0.0),
        4.0,
    )
    for attr in (
        "vs_ref_sampling_random_frac",
        "vs_ref_sampling_time_frac",
        "vs_ref_sampling_leverage_frac",
        "vs_ref_sampling_uncertainty_frac",
    ):
        setattr(cfg, attr, max(float(getattr(cfg, attr, 0.25) or 0.0), 0.0))
    cfg.vs_ref_sampling_cost_power = max(float(getattr(cfg, "vs_ref_sampling_cost_power", 0.5) or 0.0), 0.0)
    cfg.structured_bw_objective_norm = str(
        getattr(cfg, "structured_bw_objective_norm", "per_latent_dim") or "per_latent_dim"
    ).strip().lower()
    if cfg.structured_bw_objective_norm != "per_latent_dim":
        raise ValueError('structured_bw_objective_norm is fixed to "per_latent_dim" for redesigned BW actor.')
    for attr in (
        "actor_encoder_mlp_layers",
        "actor_context_mlp_layers",
        "actor_head_mlp_layers",
        "critic_encoder_mlp_layers",
        "critic_message_mlp_layers",
        "critic_value_head_layers",
    ):
        value = int(getattr(cfg, attr, 2) or 2)
        if value < 1:
            raise ValueError(f"{attr} must be >= 1.")
        setattr(cfg, attr, value)
    cfg.critic_stage_specific_paths_enabled = bool(getattr(cfg, "critic_stage_specific_paths_enabled", False))
    for attr, fallback_attr in (
        ("critic_sat_hidden", "critic_hidden"),
        ("critic_sat_embed_dim", "critic_embed_dim"),
        ("critic_sat_encoder_mlp_layers", "critic_encoder_mlp_layers"),
        ("critic_sat_message_mlp_layers", "critic_message_mlp_layers"),
        ("critic_sat_message_layers", "critic_message_layers"),
        ("critic_sat_value_head_hidden", "critic_value_head_hidden"),
        ("critic_sat_value_head_layers", "critic_value_head_layers"),
    ):
        fallback = int(getattr(cfg, fallback_attr))
        value = int(getattr(cfg, attr, 0) or 0)
        if value <= 0:
            value = fallback
        if value < 1:
            raise ValueError(f"{attr} must be >= 1 when set.")
        setattr(cfg, attr, value)
    cfg.actor_hidden = max(int(getattr(cfg, "actor_hidden", 256) or 256), 1)
    cfg.actor_set_embed_dim = max(int(getattr(cfg, "actor_set_embed_dim", 128) or 128), 1)
    for prefix in ("accel", "sat", "bw"):
        hidden_attr = f"{prefix}_hidden"
        embed_attr = f"{prefix}_embed_dim"
        hidden_value = int(getattr(cfg, hidden_attr, 0) or 0)
        embed_value = int(getattr(cfg, embed_attr, 0) or 0)
        if hidden_value <= 0:
            hidden_value = int(cfg.actor_hidden)
        if embed_value <= 0:
            embed_value = int(cfg.actor_set_embed_dim)
        if hidden_value < 1:
            raise ValueError(f"{hidden_attr} must be >= 1 when set.")
        if embed_value < 1:
            raise ValueError(f"{embed_attr} must be >= 1 when set.")
        setattr(cfg, hidden_attr, hidden_value)
        setattr(cfg, embed_attr, embed_value)
    for prefix in ("accel", "sat", "bw"):
        for part, fallback_attr in (
            ("encoder", "actor_encoder_mlp_layers"),
            ("context", "actor_context_mlp_layers"),
            ("head", "actor_head_mlp_layers"),
        ):
            attr = f"{prefix}_{part}_mlp_layers"
            fallback = int(getattr(cfg, fallback_attr, 2) or 2)
            value = int(getattr(cfg, attr, 0) or 0)
            if value <= 0:
                value = fallback
            if value < 1:
                raise ValueError(f"{attr} must be >= 1.")
            if value > 4:
                raise ValueError(f"{attr} must be <= 4 for native CUDA actor parity, got {value}.")
            setattr(cfg, attr, value)
    cfg.accel_interaction_layers = int(getattr(cfg, "accel_interaction_layers", 0) or 0)
    cfg.accel_attention_heads = int(getattr(cfg, "accel_attention_heads", 4) or 4)
    for attr, fallback in (
        ("accel_gu_query_count", 4),
        ("accel_peer_query_count", 2),
        ("accel_sat_query_count", 2),
    ):
        value = int(getattr(cfg, attr, 0) or 0)
        if value <= 0:
            value = int(fallback)
        if value < 1:
            raise ValueError(f"{attr} must be >= 1.")
        setattr(cfg, attr, value)
    if cfg.accel_interaction_layers < 0:
        raise ValueError("accel_interaction_layers must be >= 0.")
    if cfg.accel_attention_heads < 1:
        raise ValueError("accel_attention_heads must be >= 1.")
    cfg.bw_down_query_count = int(getattr(cfg, "bw_down_query_count", 2) or 2)
    cfg.bw_competition_layers = int(getattr(cfg, "bw_competition_layers", 2) or 2)
    cfg.bw_attention_heads = int(getattr(cfg, "bw_attention_heads", 4) or 4)
    if cfg.bw_down_query_count < 1:
        raise ValueError("bw_down_query_count must be >= 1.")
    if cfg.bw_competition_layers < 1:
        raise ValueError("bw_competition_layers must be >= 1.")
    if cfg.bw_attention_heads < 1:
        raise ValueError("bw_attention_heads must be >= 1.")
    cfg.bw_tau_min = float(getattr(cfg, "bw_tau_min", 0.5) or 0.5)
    cfg.bw_tau_max = float(getattr(cfg, "bw_tau_max", 2.0) or 2.0)
    cfg.bw_kappa_min = float(getattr(cfg, "bw_kappa_min", 0.5) or 0.5)
    cfg.bw_kappa_max = float(getattr(cfg, "bw_kappa_max", 32.0) or 32.0)
    fixed_tau = getattr(cfg, "bw_fixed_tau", None)
    cfg.bw_fixed_tau = None if fixed_tau is None else float(fixed_tau)
    fixed_kappa = getattr(cfg, "bw_fixed_kappa", None)
    cfg.bw_fixed_kappa = None if fixed_kappa is None else float(fixed_kappa)
    if not (0.0 < cfg.bw_tau_min < cfg.bw_tau_max):
        raise ValueError("BW tau range must satisfy 0 < bw_tau_min < bw_tau_max.")
    if not (0.0 < cfg.bw_kappa_min < cfg.bw_kappa_max):
        raise ValueError("BW kappa range must satisfy 0 < bw_kappa_min < bw_kappa_max.")
    if cfg.bw_fixed_tau is not None and not (cfg.bw_fixed_tau > 0.0):
        raise ValueError("bw_fixed_tau must be positive when set.")
    if cfg.bw_fixed_kappa is not None and not (cfg.bw_fixed_kappa > 0.0):
        raise ValueError("bw_fixed_kappa must be positive when set.")
    bw_native_dirichlet_diagnostic_mode = str(
        getattr(cfg, "bw_native_dirichlet_diagnostic_mode", "current") or "current"
    ).strip().lower()
    if bw_native_dirichlet_diagnostic_mode not in {"current", "new_fast", "legacy_fast"}:
        raise ValueError(
            "bw_native_dirichlet_diagnostic_mode must be one of "
            "{'current', 'new_fast', 'legacy_fast'}."
        )
    cfg.bw_native_dirichlet_diagnostic_mode = bw_native_dirichlet_diagnostic_mode
    cfg.N_RF = int(getattr(cfg, "N_RF", 0) or 0)
    if cfg.N_RF <= 0:
        raise ValueError(f"N_RF must be positive, got {cfg.N_RF!r}.")
    sat_select_cfg = getattr(cfg, "sat_num_select", None)
    sat_select_count = int(sat_select_cfg) if sat_select_cfg is not None and int(sat_select_cfg) > 0 else int(cfg.N_RF)
    sat_select_ref_count = min(int(cfg.num_sat), int(cfg.N_RF), sat_select_count)
    if sat_select_ref_count <= 0:
        raise ValueError("sat_select_ref_count must be positive.")
    cfg.sat_action_select_k = int(sat_select_ref_count)
    cfg.sat_competition_layers = int(getattr(cfg, "sat_competition_layers", 2) or 2)
    cfg.sat_attention_heads = int(getattr(cfg, "sat_attention_heads", 4) or 4)
    if cfg.sat_competition_layers < 1:
        raise ValueError("sat_competition_layers must be >= 1.")
    if cfg.sat_attention_heads < 1:
        raise ValueError("sat_attention_heads must be >= 1.")
    visible_ref = getattr(cfg, "visible_sats_max", None)
    visible_width = int(visible_ref) if visible_ref is not None else int(cfg.sats_obs_max)
    cfg.per_uav_visible_sat_token_max = min(int(cfg.num_sat), visible_width)
    if int(cfg.per_uav_visible_sat_token_max) <= 0:
        raise ValueError(
            "per_uav_visible_sat_token_max must be positive; "
            f"got {cfg.per_uav_visible_sat_token_max!r} from num_sat={cfg.num_sat}, visible_sats_max={visible_ref!r}, sats_obs_max={cfg.sats_obs_max}."
        )
    for attr in (
        "map_size",
        "v_max",
        "a_max",
        "uav_energy_init",
        "queue_max_gu",
        "queue_max_uav",
        "queue_max_sat",
        "b_acc",
        "task_cycles_per_bit",
        "speed_of_light",
    ):
        _require_positive_config_float(float(getattr(cfg, attr, 0.0) or 0.0), attr)
    _normalize_safety_shield_config(cfg)
    _require_positive_config_float(float(cfg.r_earth + cfg.sat_height), "orbit_pos_ref")
    _require_positive_config_float(_effective_b_backhaul_per_sat_from_cfg(cfg), "effective_b_backhaul_per_sat")
    if bool(getattr(cfg, "doppler_enabled", False) or getattr(cfg, "doppler_atten_enabled", False) or getattr(cfg, "doppler_observed", False)):
        _require_positive_config_float(float(getattr(cfg, "nu_max", 0.0) or 0.0), "nu_max")
    for layer in ("gu", "uav", "sat"):
        steps_value = getattr(cfg, f"queue_max_{layer}_steps", None)
        if steps_value is None:
            continue
        total_cap = max(float(steps_value), 0.0) * _queue_ref_per_step(cfg, layer)
        per_entity_cap = total_cap / _queue_ref_entities(cfg, layer)
        setattr(cfg, f"queue_max_{layer}", per_entity_cap)
    for attr in ("queue_max_gu", "queue_max_uav", "queue_max_sat"):
        _require_positive_config_float(float(getattr(cfg, attr, 0.0) or 0.0), attr)
