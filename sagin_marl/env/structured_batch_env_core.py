from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import copy
from dataclasses import dataclass, is_dataclass, replace
import math
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sagin_marl.env import channel
from sagin_marl.env.numeric_guards import (
    ANGLE_RAD_EPS,
    FREQUENCY_GHZ_EPS,
    GEOMETRY_DENOM_EPS,
    LOG_RATIO_EPS,
    MIN_RAIN_REDUCTION_FACTOR,
    NORMALIZATION_DENOM_EPS,
    POSITIVE_COEFF_EPS,
    PROBABILITY_EQUALITY_TOL,
    RELATIVE_LOG_EPS,
    RUNTIME_RATIO_ZERO_TOL,
    TRIG_DENOM_EPS,
    geometry_denominator,
    log_ratio_argument,
    normalize_scale,
    ratio_or_zero,
    require_positive_array,
    require_positive_float,
    reward_ratio_denominator_scalar,
)
from sagin_marl.env.config import (
    ablation_flag,
    access_carrier_freq_from_config as _access_carrier_freq_from_cfg,
    backhaul_carrier_freq_from_config as _backhaul_carrier_freq_from_cfg,
)
from sagin_marl.env.orbit import WalkerDeltaOrbitModel
from sagin_marl.env.structured_kernel_runtime import get_structured_kernel_runtime
from sagin_marl.env.sagin_env import BwTransitionCoreResult, SaginParallelEnv, StepStatusCoreResult
from sagin_marl.env.topology import thomas_cluster_process
from sagin_marl.env.structured_stage_obs import (
    current_obs_batch,
    current_obs_list,
    refresh_stage_obs_cache,
    sat_mask_to_ids,
)
from sagin_marl.env.structured_driver import (
    StructuredBatchStepResult,
    StructuredControlDriver,
    _subset_member_spec_cpu,
)
from sagin_marl.env.structured_gpu_rollout_runtime import (
    StructuredGpuAccelObsView,
    StructuredGpuBwRuntimeCacheBuffers,
    StructuredGpuBwObsView,
    StructuredGpuNativeRolloutProgram,
    StructuredGpuNativeRuntimeStepProgram,
    StructuredGpuRewardPartBuffers,
    StructuredGpuSatObsView,
    StructuredGpuRolloutRuntime,
    _BW_REWARD_PART_KEYS,
    _tensor_field_names,
)
from sagin_marl.env.native_cuda import bindings as native_cuda
from sagin_marl.rl import structured_accel_actor_schema as accel_schema
from sagin_marl.rl import structured_bw_actor_schema as bw_schema
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl import structured_sat_actor_schema as sat_schema
from sagin_marl.rl.structured_types import LocalAccelState, LocalBwState, LocalSatState, StructuredWorldState


_NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH = 0


def _project_l2_ball_np(values: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    limit = max(float(max_norm), 0.0)
    if limit <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


def _project_l2_ball_torch(values: torch.Tensor, max_norm: float) -> torch.Tensor:
    arr = values.to(dtype=torch.float32)
    limit = max(float(max_norm), 0.0)
    if limit <= 0.0:
        return torch.zeros_like(arr)
    norms = torch.linalg.vector_norm(arr, dim=-1, keepdim=True)
    scale = torch.clamp(limit / norms.clamp_min(1.0e-8), max=1.0)
    return arr * scale


class NativeMainKernelHotReplayGuard:
    """Guard used by structural tests for official CUDA hot replay.

    Capture/build is allowed to read the frozen static spec. Hot replay is not:
    if a compiled/captured segment falls back to Python and reads this spec, the
    guard fails immediately instead of letting the old config-object boundary
    hide inside the strict path.
    """

    def __enter__(self):
        global _NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH
        _NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        global _NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH
        _NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH = max(_NATIVE_MAIN_KERNEL_HOT_REPLAY_GUARD_DEPTH - 1, 0)
        return False


@dataclass(frozen=True)
class _NativeAccessRateStaticParams:
    eta_quantum: float
    rate_quantum: float
    interference_quantum: float
    fast_float32: bool
    enable_bw_action: bool
    interference_enabled: bool
    gu_tx_power: float
    noise_density: float
    noise_figure_db: float
    b_acc: float
    queue_max_gu: float
    fading_mode_code: int
    rician_k: float
    ergodic_rician_quadrature_points: int


@dataclass(frozen=True)
class _NativeBwWorkloadStaticParams:
    eps: float
    num_gu: int
    num_uav: int
    num_sat: int
    sat_active_ref_count: float


@dataclass(frozen=True)
class _NativeChannelStaticParams:
    fast_float32: bool
    access_gain_quantum: float
    access_pathloss_db_quantum: float
    pathloss_const_db: float
    carrier_freq: float
    xi_los: float
    xi_nlos: float
    los_a: float
    los_b: float
    pathloss_mode: str
    atm_loss_enabled: bool
    atm_loss_db: float
    rain_loss_enabled: bool
    rain_rate_001_mmph: float
    rain_height_km: float
    station_height_km: float
    latitude_deg: float
    rain_exceedance_pct: float
    rain_polarization_tilt_deg: float


@dataclass(frozen=True)
class _NativeCandidateStaticParams:
    fast_float32: bool
    num_gu: int
    num_uav: int
    users_obs_max: int
    uav_height: float
    pl_threshold_db: float
    candidate_mode: str
    candidate_k: int
    candidate_radius: float | None


@dataclass(frozen=True)
class _NativeShapeAbiStaticParams:
    num_envs: int
    num_uav: int
    num_gu: int
    num_sat: int
    users_obs_max: int
    sats_obs_max: int
    visible_sats_max: int
    sat_num_select: int
    n_rf: int


@dataclass(frozen=True)
class _NativeNumericStaticParams:
    fast_float32: bool
    access_eta_quantum: float
    access_rate_quantum: float
    flow_bits_quantum: float
    queue_state_quantum: float
    summary_metric_quantum: float


@dataclass(frozen=True)
class _NativeSatGeometryStaticParams:
    num_uav: int
    num_sat: int
    n_rf: int
    sat_num_select: int
    sats_obs_max: int
    visible_sats_max: int
    ref_lat_deg: float
    ref_lon_deg: float
    r_earth: float
    uav_height: float
    theta_min_rad: float
    carrier_freq: float
    speed_of_light: float
    nu_max: float
    queue_max_sat: float
    uav_tx_power: float
    noise_density: float
    noise_figure_db: float
    subcarrier_spacing: float
    sat_candidate_mode: str
    sat_candidate_elevation_weight: float
    sat_candidate_queue_weight: float
    sat_candidate_se_weight: float
    doppler_enabled: bool
    doppler_atten_enabled: bool
    doppler_observed: bool
    doppler_precomp_enabled: bool


@dataclass(frozen=True)
class _NativeLocalObsStaticParams:
    num_uav: int
    num_gu: int
    num_sat: int
    tau0: float
    map_size: float
    v_max: float
    a_max: float
    uav_energy_init: float
    queue_max_gu: float
    queue_max_uav: float
    queue_max_sat: float
    b_acc: float
    r_earth: float
    sat_height: float
    uav_height: float
    gu_tx_power: float
    uav_tx_power: float
    noise_density: float
    noise_figure_db: float
    access_noise_figure_db: float
    backhaul_noise_figure_db: float
    backhaul_carrier_freq: float
    speed_of_light: float
    service_floor_bits_per_step: float
    sat_active_ref_count: float
    sat_select_ref_count: int
    nu_max: float
    subcarrier_spacing: float
    obs_own_include_assoc_uav_cost: bool
    obs_own_include_uav_id_norm: bool
    obs_sat_include_sat_cost: bool
    obs_user_include_assoc_uav_cost: bool
    obs_user_include_assoc_sat_cost_mean: bool
    obs_user_include_local_gu_service_cost: bool
    obs_user_include_weighted_queue_cost: bool
    obs_user_include_weighted_queue_cost_relative: bool
    obs_user_include_queue_headroom: bool
    obs_user_include_arrival_rate: bool
    obs_user_include_recent_arrival: bool
    obs_user_include_recent_service: bool
    obs_user_include_urgency_risk: bool
    obs_user_include_downstream_pressure: bool
    obs_user_include_service_gap: bool
    obs_user_include_service_gap_risk: bool
    obs_user_include_deadline_slack: bool
    obs_user_include_deadline_risk: bool
    service_gap_cap_steps: float
    avoidance_enabled: bool
    avoidance_alert_factor: float
    d_safe: float
    doppler_atten_enabled: bool
    doppler_observed: bool


@dataclass(frozen=True)
class _NativeGlobalStateStaticParams:
    num_gu: int
    num_sat: int
    map_size: float
    v_max: float
    uav_energy_init: float
    queue_max_gu: float
    queue_max_uav: float
    queue_max_sat: float
    r_earth: float
    sat_height: float
    uav_height: float
    t_steps: float


@dataclass(frozen=True)
class _NativeAccelSafetyStaticParams:
    num_uav: int
    a_max: float
    v_max: float
    tau0: float
    map_size: float
    d_safe: float
    uav_energy_init: float
    uav_opt_speed: float
    energy_enabled: bool
    energy_safe_threshold: float
    use_avoidance: bool
    use_energy_safety: bool
    boundary_hard_filter_enabled: bool
    pairwise_hard_filter_enabled: bool
    boundary_mode: str
    boundary_margin: float
    avoidance_alert_factor: float
    avoidance_prealert_factor: float | None
    avoidance_prealert_mode: str
    avoidance_prealert_closing_speed: float
    avoidance_prealert_ttc: float
    avoidance_prealert_dist_cap: float | None
    avoidance_repulse_mode: str
    avoidance_repulse_clip: float
    avoidance_closing_gain_enabled: bool
    avoidance_closing_gain_cap: float
    avoidance_closing_gain_top1_only: bool
    avoidance_eta_min: float
    avoidance_eta_max: float
    avoidance_eta: float
    centroid_cross_anneal_enabled: bool
    eta_centroid: float
    eta_centroid_final: float | None
    eta_centroid_decay_steps: int
    centroid_cross_avoidance_gain: float
    pairwise_hard_distance: float | None
    pairwise_hard_trigger_mode: str
    pairwise_hard_trigger_distance: float | None
    pairwise_hard_trigger_ttc: float
    pairwise_hard_closing_speed: float
    pairwise_hard_max_passes: int
    pairwise_hard_single_pair_only: bool
    safety_shield_native_enabled: bool
    safety_shield_tensor_iters: int
    safety_shield_brake_rho: float
    safety_shield_a_safe: float
    safety_shield_distance_buffer: float
    safety_shield_step_gain: float
    safety_shield_tolerance: float


@dataclass(frozen=True)
class _NativeBwQueueDeadlineStaticParams:
    tau0: float
    queue_max_gu: float
    queue_max_uav: float
    queue_max_sat: float
    flow_bits_quantum: float
    queue_state_quantum: float
    fast_float32: bool
    service_gap_increment: float
    service_gap_relief_coef: float
    service_gap_cap_steps: float
    deadline_enabled: bool
    deadline_age_increment: float
    deadline_service_relief_coef: float
    deadline_age_cap_steps: float
    deadline_expire_rate: float


@dataclass(frozen=True)
class _NativeBwLinkStaticParams:
    num_sat: int
    tau0: float
    backhaul_rate_quantum: float
    b_backhaul_per_sat: float
    b_backhaul_per_sat_scale: float
    uav_tx_power: float
    noise_density: float
    noise_figure_db: float
    p_comm_link: float
    queue_max_sat: float
    energy_enabled: bool
    doppler_enabled: bool
    doppler_atten_enabled: bool
    subcarrier_spacing: float
    energy_model: str
    p_fly_base: float
    p_fly_coeff: float
    rotor_p0: float
    rotor_pi: float
    rotor_u_tip: float
    rotor_v0: float
    rotor_d0: float
    rotor_rho: float
    rotor_s: float
    rotor_area: float


@dataclass(frozen=True)
class _NativeRewardMetricsStaticParams:
    reward_mode_default: str
    fast_float32: bool
    num_uav: int
    num_gu: int
    num_sat: int
    t_steps: int
    tau0: float
    a_max: float
    v_max: float
    uav_energy_init: float
    n_rf: int
    queue_max_gu: float
    queue_max_uav: float
    queue_max_sat: float
    summary_metric_quantum: float
    bw_weighted_workload_ema_decay: float
    energy_enabled: bool
    energy_model: str
    p_fly_base: float
    p_fly_coeff: float
    p_comm_link: float
    rotor_p0: float
    rotor_pi: float
    rotor_u_tip: float
    rotor_v0: float
    rotor_d0: float
    rotor_rho: float
    rotor_s: float
    rotor_area: float
    omega_q: float
    omega_e: float
    eta_service: float
    eta_q_delta: float
    eta_batt: float
    eta_crash: float
    eta_accel: float
    eta_drop: float
    eta_drop_step: float
    eta_drop_gu: float
    eta_drop_uav: float
    eta_drop_sat: float
    eta_throughput_access: float
    eta_throughput_backhaul: float
    eta_close_risk: float
    reward_w_access: float
    reward_w_relay: float
    reward_w_pre_backlog: float
    reward_w_pre_drop: float
    reward_w_pre_service_gap: float
    reward_w_pre_overflow_risk: float
    throughput_only_access_coef: float
    throughput_only_backhaul_coef: float
    throughput_only_gu_queue_coef: float
    queue_penalty_mode: str
    queue_delta_mode: str
    queue_norm_k: float
    queue_norm_arrival_floor: float
    queue_reward_use_arrival_norm: bool
    queue_log_k: float
    omega_q_gu: float
    omega_q_uav: float
    omega_q_sat: float
    omega_q_tail: float
    q_norm_tail_q0: float
    tail_q_small: float
    tail_eta_accel_gain: float
    centroid_cross_anneal_enabled: bool
    eta_centroid: float
    eta_centroid_final: float | None
    eta_centroid_decay_steps: int
    centroid_dist_scale: float
    centroid_cross_queue_gain: float
    centroid_cross_q_delta_gain: float
    centroid_cross_crash_gain: float
    use_queue_log_smoothing: bool
    use_active_queue_delta: bool
    use_energy_reward: bool
    use_reward_tanh: bool


@dataclass(frozen=True)
class _NativeBwFlowProxyStaticParams:
    enabled: bool
    reward_mode_code: int
    aux_delta: float
    eps: float


@dataclass(frozen=True)
class _NativePostStatsSafetyStaticParams:
    num_uav: int
    num_gu: int
    num_sat: int
    tau0: float
    queue_max_gu: float
    queue_max_uav: float
    queue_max_sat: float
    overflow_risk_threshold_frac: float
    overflow_risk_arrival_coef: float
    overflow_risk_service_coef: float
    service_gap_cap_steps: float
    service_gap_risk_threshold_steps: float
    a_max: float
    d_safe: float
    avoidance_alert_factor: float
    close_risk_enabled: bool
    close_risk_cap: float
    avoidance_prealert_factor: float | None
    avoidance_prealert_mode: str
    avoidance_prealert_closing_speed: float
    avoidance_prealert_ttc: float
    avoidance_prealert_dist_cap: float | None
    danger_imitation_enabled: bool
    danger_imitation_trigger_mode: str
    danger_imitation_close_risk_thresh: float
    danger_imitation_intervention_thresh: float


@dataclass(frozen=True)
class _NativeRandomTapeStaticParams:
    num_uav: int
    num_gu: int
    num_sat: int
    deadline_enabled: bool
    doppler_precomp_mode: str
    doppler_residual_sigma_hz: float
    doppler_residual_rho: float
    doppler_residual_cap_hz: float


@dataclass(frozen=True)
class _NativeHistoryOutputStaticParams:
    copy_graph_outputs: bool
    strict_cuda_contract: bool
    cudagraph_direct_inputs: bool
    runtime_snapshots_enabled: bool


@dataclass(frozen=True)
class _NativeCompileCaptureStaticParams:
    require_compiled_segments: bool
    cudagraph_direct_inputs: bool
    operator_mode: str
    compile_backend: str


@dataclass(frozen=True)
class _NativeMainKernelTypedDomains:
    shape: _NativeShapeAbiStaticParams
    numeric: _NativeNumericStaticParams
    channel: _NativeChannelStaticParams
    candidate: _NativeCandidateStaticParams
    access_rate: _NativeAccessRateStaticParams
    sat_geometry: _NativeSatGeometryStaticParams
    local_obs: _NativeLocalObsStaticParams
    global_state: _NativeGlobalStateStaticParams
    accel_safety: _NativeAccelSafetyStaticParams
    bw_queue_deadline: _NativeBwQueueDeadlineStaticParams
    bw_link: _NativeBwLinkStaticParams
    bw_workload: _NativeBwWorkloadStaticParams
    reward_metrics: _NativeRewardMetricsStaticParams
    bw_flow_proxy: _NativeBwFlowProxyStaticParams
    post_stats_safety: _NativePostStatsSafetyStaticParams
    random_tape: _NativeRandomTapeStaticParams
    history_output: _NativeHistoryOutputStaticParams
    compile_capture: _NativeCompileCaptureStaticParams


def _access_fading_mode_code_from_cfg(cfg: Any) -> int:
    if not bool(getattr(cfg, "fading_enabled", False)):
        return 0
    mode = channel.access_fading_mode_from_config(cfg)
    return {"large_scale": 0, "ergodic_rician": 1, "iid_rician": 2}.get(mode, 1)


def _native_history_snapshots_required_from_cfg(cfg: Any) -> bool:
    raw = getattr(cfg, "structured_native_history_snapshots_enabled", "auto")
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw or "auto").strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "always"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "never"}:
        return False
    # Auto keeps snapshots only for code paths that restore historical states
    # for branch replay / clean teacher diagnostics. Plain PPO or MC-GAE
    # training consumes local obs, actions, rewards, dones, and world batches,
    # not the full runtime/stage snapshots.
    stage_modes = {
        str(getattr(cfg, "structured_actor_update_mode", "ppo") or "ppo").strip().lower(),
        str(getattr(cfg, "accel_update_mode", "") or "").strip().lower(),
        str(getattr(cfg, "sat_update_mode", "") or "").strip().lower(),
        str(getattr(cfg, "bw_update_mode", "") or "").strip().lower(),
    }
    if "vs_ref" in stage_modes:
        return True
    bw_override = str(getattr(cfg, "bw_actor_advantage_override_mode", "gae") or "gae").strip().lower()
    if bw_override in {"branch_delta", "true_adv_mc", "delta_critic", "delta_teacher_student"}:
        return True
    for attr in (
        "bw_clean_per_user_enabled",
        "bw_clean_candidate_select_enabled",
        "bw_delta_critic_enabled",
        "bw_marginal_teacher_sample_enabled",
        "sat_clean_joint_enabled",
        "update_direction_probe_branch_enabled",
    ):
        if bool(getattr(cfg, attr, False)):
            return True
    return False


def _native_access_rate_static_params_from_cfg(cfg: Any) -> _NativeAccessRateStaticParams:
    access_gain_quantum = float(_semantic_quantum(cfg, "structured_access_gain_quantum", 5.0e-16))
    return _NativeAccessRateStaticParams(
        eta_quantum=float(_semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6)),
        rate_quantum=float(_semantic_quantum(cfg, "structured_access_rate_quantum", 32.0)),
        interference_quantum=(
            0.0 if access_gain_quantum <= 0.0 else abs(float(cfg.gu_tx_power)) * access_gain_quantum
        ),
        fast_float32=True,
        enable_bw_action=bool(getattr(cfg, "enable_bw_action", False)),
        interference_enabled=bool(getattr(cfg, "interference_enabled", False)),
        gu_tx_power=float(cfg.gu_tx_power),
        noise_density=float(cfg.noise_density),
        noise_figure_db=max(float(getattr(cfg, "access_noise_figure_db", 5.0) or 0.0), 0.0),
        b_acc=float(cfg.b_acc),
        queue_max_gu=float(cfg.queue_max_gu),
        fading_mode_code=_access_fading_mode_code_from_cfg(cfg),
        rician_k=channel.rician_k_linear_from_config(cfg),
        ergodic_rician_quadrature_points=max(
            int(getattr(cfg, "access_ergodic_rician_quadrature_points", 16) or 16),
            1,
        ),
    )


def _native_bw_workload_static_params_from_cfg(cfg: Any) -> _NativeBwWorkloadStaticParams:
    return _NativeBwWorkloadStaticParams(
        eps=max(
            float(
                getattr(
                    cfg,
                    "service_floor_bits_per_step",
                    getattr(cfg, "bw_weighted_workload_eps", 1.0),
                )
                or 0.0
            ),
            float(NORMALIZATION_DENOM_EPS),
        ),
        num_gu=int(cfg.num_gu),
        num_uav=int(cfg.num_uav),
        num_sat=int(cfg.num_sat),
        sat_active_ref_count=float(_bw_weighted_workload_sat_active_ref_count(cfg)),
    )


def _native_channel_static_params_from_cfg(cfg: Any) -> _NativeChannelStaticParams:
    rain_lat = cfg.ref_lat_deg if getattr(cfg, "rain_lat_deg", None) is None else cfg.rain_lat_deg
    return _NativeChannelStaticParams(
        fast_float32=True,
        access_gain_quantum=float(_semantic_quantum(cfg, "structured_access_gain_quantum", 5.0e-16)),
        access_pathloss_db_quantum=float(_semantic_quantum(cfg, "structured_access_pathloss_db_quantum", 1.0e-2)),
        pathloss_const_db=float(cfg.pathloss_const_db),
        carrier_freq=_access_carrier_freq_from_cfg(cfg),
        xi_los=float(cfg.xi_los),
        xi_nlos=float(cfg.xi_nlos),
        los_a=float(cfg.los_a),
        los_b=float(cfg.los_b),
        pathloss_mode=str(getattr(cfg, "pathloss_mode", "prob_los") or "prob_los").strip().lower(),
        atm_loss_enabled=bool(getattr(cfg, "atm_loss_enabled", False)),
        atm_loss_db=float(cfg.atm_loss_db),
        rain_loss_enabled=bool(getattr(cfg, "rain_loss_enabled", False)),
        rain_rate_001_mmph=float(cfg.rain_rate_001_mmph),
        rain_height_km=float(cfg.rain_height_km),
        station_height_km=float(cfg.uav_height) / 1000.0,
        latitude_deg=float(rain_lat),
        rain_exceedance_pct=float(cfg.rain_exceedance_pct),
        rain_polarization_tilt_deg=float(cfg.rain_polarization_tilt_deg),
    )


def _native_candidate_static_params_from_cfg(cfg: Any) -> _NativeCandidateStaticParams:
    raw_mode = getattr(cfg, "candidate_mode", None)
    if raw_mode is None:
        raw_mode = getattr(cfg, "candidate_users_mode", "assoc")
    mode = str(raw_mode or "assoc").strip().lower()
    users_obs_max = int(cfg.users_obs_max)
    candidate_k_raw = int(getattr(cfg, "candidate_k", 0) or 0)
    candidate_k = users_obs_max if candidate_k_raw <= 0 else min(candidate_k_raw, users_obs_max)
    raw_radius = getattr(cfg, "candidate_radius", None)
    radius = None if raw_radius is None else float(raw_radius)
    return _NativeCandidateStaticParams(
        fast_float32=True,
        num_gu=int(cfg.num_gu),
        num_uav=int(cfg.num_uav),
        users_obs_max=users_obs_max,
        uav_height=float(cfg.uav_height),
        pl_threshold_db=float(cfg.pl_threshold_db),
        candidate_mode=mode,
        candidate_k=int(candidate_k),
        candidate_radius=radius,
    )


def _sat_action_select_k_from_config(cfg: Any) -> int:
    n_rf = max(int(getattr(cfg, "N_RF", 0) or 0), 1)
    num_sat = max(int(getattr(cfg, "num_sat", 0) or 0), 0)
    sat_num_select_cfg = getattr(cfg, "sat_num_select", None)
    sat_num_select = (
        int(sat_num_select_cfg)
        if sat_num_select_cfg is not None and int(sat_num_select_cfg) > 0
        else n_rf
    )
    upper_sat = num_sat if num_sat > 0 else sat_num_select
    select_k = max(min(int(upper_sat), int(n_rf), int(sat_num_select)), 1)
    try:
        setattr(cfg, "sat_action_select_k", int(select_k))
    except Exception:
        pass
    return int(select_k)


def _sat_visible_width_from_config(cfg: Any) -> int:
    raw_width = getattr(cfg, "per_uav_visible_sat_token_max", None)
    if raw_width is None:
        raw_width = getattr(cfg, "visible_sats_max", None)
    if raw_width is None:
        raw_width = getattr(cfg, "sats_obs_max", None)
    if raw_width is None:
        raise ValueError("per_uav_visible_sat_token_max or sats_obs_max must be configured.")
    width = int(raw_width)
    num_sat = max(int(getattr(cfg, "num_sat", 0) or 0), 0)
    if num_sat > 0:
        width = min(width, num_sat)
    if width <= 0:
        raise ValueError("per_uav_visible_sat_token_max must be positive.")
    try:
        setattr(cfg, "per_uav_visible_sat_token_max", int(width))
    except Exception:
        pass
    return int(width)


def _native_typed_domains_from_cfg(cfg: Any, *, num_envs: int) -> _NativeMainKernelTypedDomains:
    visible_sats_max = _sat_visible_width_from_config(cfg)
    sat_num_select = _sat_action_select_k_from_config(cfg)
    shape = _NativeShapeAbiStaticParams(
        num_envs=int(num_envs),
        num_uav=int(cfg.num_uav),
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        users_obs_max=int(cfg.users_obs_max),
        sats_obs_max=int(cfg.sats_obs_max),
        visible_sats_max=int(visible_sats_max),
        sat_num_select=int(sat_num_select),
        n_rf=int(cfg.N_RF),
    )
    numeric = _NativeNumericStaticParams(
        fast_float32=True,
        access_eta_quantum=float(_semantic_quantum(cfg, "structured_access_eta_quantum", 1.0e-6)),
        access_rate_quantum=float(_semantic_quantum(cfg, "structured_access_rate_quantum", 32.0)),
        flow_bits_quantum=float(_semantic_quantum(cfg, "structured_flow_bits_quantum", 0.0)),
        queue_state_quantum=float(_semantic_quantum(cfg, "structured_queue_state_quantum", 128.0)),
        summary_metric_quantum=float(_semantic_quantum(cfg, "structured_summary_metric_quantum", 0.0)),
    )
    channel_p = _native_channel_static_params_from_cfg(cfg)
    candidate_p = _native_candidate_static_params_from_cfg(cfg)
    access_p = _native_access_rate_static_params_from_cfg(cfg)
    bw_workload_p = _native_bw_workload_static_params_from_cfg(cfg)
    sat_p = _NativeSatGeometryStaticParams(
        num_uav=int(cfg.num_uav),
        num_sat=int(cfg.num_sat),
        n_rf=int(cfg.N_RF),
        sat_num_select=shape.sat_num_select,
        sats_obs_max=int(cfg.sats_obs_max),
        visible_sats_max=shape.visible_sats_max,
        ref_lat_deg=float(cfg.ref_lat_deg),
        ref_lon_deg=float(cfg.ref_lon_deg),
        r_earth=float(cfg.r_earth),
        uav_height=float(cfg.uav_height),
        theta_min_rad=float(cfg.theta_min_rad),
        carrier_freq=_backhaul_carrier_freq_from_cfg(cfg),
        speed_of_light=float(cfg.speed_of_light),
        nu_max=float(cfg.nu_max),
        queue_max_sat=float(cfg.queue_max_sat),
        uav_tx_power=float(cfg.uav_tx_power),
        noise_density=float(cfg.noise_density),
        noise_figure_db=max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0),
        subcarrier_spacing=float(getattr(cfg, "subcarrier_spacing", 15e3) or 15e3),
        sat_candidate_mode=str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower(),
        sat_candidate_elevation_weight=float(getattr(cfg, "sat_candidate_elevation_weight", 1.0) or 0.0),
        sat_candidate_queue_weight=float(getattr(cfg, "sat_candidate_queue_weight", 0.0) or 0.0),
        sat_candidate_se_weight=float(getattr(cfg, "sat_candidate_se_weight", 0.0) or 0.0),
        doppler_enabled=bool(getattr(cfg, "doppler_enabled", False)),
        doppler_atten_enabled=bool(getattr(cfg, "doppler_atten_enabled", False)),
        doppler_observed=bool(getattr(cfg, "doppler_observed", False)),
        doppler_precomp_enabled=bool(_doppler_precomp_enabled_from_cfg(cfg)),
    )
    local_obs_p = _NativeLocalObsStaticParams(
        num_uav=int(cfg.num_uav),
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        tau0=float(cfg.tau0),
        map_size=float(cfg.map_size),
        v_max=float(cfg.v_max),
        a_max=float(cfg.a_max),
        uav_energy_init=float(cfg.uav_energy_init),
        queue_max_gu=float(cfg.queue_max_gu),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        b_acc=float(cfg.b_acc),
        r_earth=float(cfg.r_earth),
        sat_height=float(cfg.sat_height),
        uav_height=float(cfg.uav_height),
        gu_tx_power=float(cfg.gu_tx_power),
        uav_tx_power=float(cfg.uav_tx_power),
        noise_density=float(cfg.noise_density),
        noise_figure_db=max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0),
        access_noise_figure_db=max(float(getattr(cfg, "access_noise_figure_db", 5.0) or 0.0), 0.0),
        backhaul_noise_figure_db=max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0),
        backhaul_carrier_freq=_backhaul_carrier_freq_from_cfg(cfg),
        speed_of_light=float(cfg.speed_of_light),
        service_floor_bits_per_step=float(getattr(cfg, "service_floor_bits_per_step", getattr(cfg, "bw_weighted_workload_eps", 1.0)) or 1.0),
        sat_active_ref_count=float(_bw_weighted_workload_sat_active_ref_count(cfg)),
        sat_select_ref_count=int(sat_num_select),
        nu_max=float(cfg.nu_max),
        subcarrier_spacing=float(getattr(cfg, "subcarrier_spacing", 15e3) or 15e3),
        obs_own_include_assoc_uav_cost=bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False)),
        obs_own_include_uav_id_norm=bool(getattr(cfg, "obs_own_include_uav_id_norm", False)),
        obs_sat_include_sat_cost=bool(getattr(cfg, "obs_sat_include_sat_cost", False)),
        obs_user_include_assoc_uav_cost=bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)),
        obs_user_include_assoc_sat_cost_mean=bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)),
        obs_user_include_local_gu_service_cost=bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)),
        obs_user_include_weighted_queue_cost=bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)),
        obs_user_include_weighted_queue_cost_relative=bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)),
        obs_user_include_queue_headroom=bool(getattr(cfg, "obs_user_include_queue_headroom", False)),
        obs_user_include_arrival_rate=bool(getattr(cfg, "obs_user_include_arrival_rate", False)),
        obs_user_include_recent_arrival=bool(getattr(cfg, "obs_user_include_recent_arrival", False)),
        obs_user_include_recent_service=bool(getattr(cfg, "obs_user_include_recent_service", False)),
        obs_user_include_urgency_risk=bool(getattr(cfg, "obs_user_include_urgency_risk", False)),
        obs_user_include_downstream_pressure=bool(getattr(cfg, "obs_user_include_downstream_pressure", False)),
        obs_user_include_service_gap=bool(getattr(cfg, "obs_user_include_service_gap", False)),
        obs_user_include_service_gap_risk=bool(getattr(cfg, "obs_user_include_service_gap_risk", False)),
        obs_user_include_deadline_slack=bool(getattr(cfg, "obs_user_include_deadline_slack", False)),
        obs_user_include_deadline_risk=bool(getattr(cfg, "obs_user_include_deadline_risk", False)),
        service_gap_cap_steps=max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6),
        avoidance_enabled=bool(getattr(cfg, "avoidance_enabled", False)),
        avoidance_alert_factor=float(cfg.avoidance_alert_factor),
        d_safe=float(cfg.d_safe),
        doppler_atten_enabled=bool(getattr(cfg, "doppler_atten_enabled", False)),
        doppler_observed=bool(getattr(cfg, "doppler_observed", False)),
    )
    global_state_p = _NativeGlobalStateStaticParams(
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        map_size=float(cfg.map_size),
        v_max=float(cfg.v_max),
        uav_energy_init=float(cfg.uav_energy_init),
        queue_max_gu=float(cfg.queue_max_gu),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        r_earth=float(cfg.r_earth),
        sat_height=float(cfg.sat_height),
        uav_height=float(cfg.uav_height),
        t_steps=max(float(cfg.T_steps), 1.0),
    )
    bw_queue_p = _NativeBwQueueDeadlineStaticParams(
        tau0=float(cfg.tau0),
        queue_max_gu=float(cfg.queue_max_gu),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        flow_bits_quantum=numeric.flow_bits_quantum,
        queue_state_quantum=numeric.queue_state_quantum,
        fast_float32=numeric.fast_float32,
        service_gap_increment=max(float(getattr(cfg, "service_gap_increment", 1.0) or 0.0), 0.0),
        service_gap_relief_coef=max(float(getattr(cfg, "service_gap_relief_coef", 0.5) or 0.0), 0.0),
        service_gap_cap_steps=max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6),
        deadline_enabled=bool(getattr(cfg, "deadline_enabled", False)),
        deadline_age_increment=max(float(getattr(cfg, "deadline_age_increment", 1.0) or 0.0), 0.0),
        deadline_service_relief_coef=max(float(getattr(cfg, "deadline_service_relief_coef", 0.75) or 0.0), 0.0),
        deadline_age_cap_steps=max(float(getattr(cfg, "deadline_age_cap_steps", 8.0) or 0.0), 1.0),
        deadline_expire_rate=max(float(getattr(cfg, "deadline_expire_rate", 0.35) or 0.0), 0.0),
    )
    accel_p = _NativeAccelSafetyStaticParams(
        num_uav=int(cfg.num_uav),
        a_max=float(cfg.a_max),
        v_max=float(cfg.v_max),
        tau0=float(bw_queue_p.tau0),
        map_size=float(cfg.map_size),
        d_safe=float(cfg.d_safe),
        uav_energy_init=float(cfg.uav_energy_init),
        uav_opt_speed=float(cfg.uav_opt_speed),
        energy_enabled=bool(cfg.energy_enabled),
        energy_safe_threshold=float(cfg.energy_safe_threshold),
        use_avoidance=bool(ablation_flag(cfg, "use_avoidance_layer", fallback_attr="avoidance_enabled", default=False)),
        use_energy_safety=bool(cfg.energy_enabled)
        and bool(ablation_flag(cfg, "use_energy_safety_layer", fallback_attr="energy_safety_enabled", default=False)),
        boundary_hard_filter_enabled=bool(getattr(cfg, "boundary_hard_filter_enabled", False)),
        pairwise_hard_filter_enabled=bool(getattr(cfg, "pairwise_hard_filter_enabled", False)),
        boundary_mode=str(getattr(cfg, "boundary_mode", "clip") or "clip").strip().lower(),
        boundary_margin=max(float(getattr(cfg, "boundary_margin", 0.0) or 0.0), 0.0),
        avoidance_alert_factor=float(cfg.avoidance_alert_factor),
        avoidance_prealert_factor=(
            None
            if getattr(cfg, "avoidance_prealert_factor", None) is None
            else float(getattr(cfg, "avoidance_prealert_factor"))
        ),
        avoidance_prealert_mode=str(getattr(cfg, "avoidance_prealert_mode", "distance") or "distance").strip().lower(),
        avoidance_prealert_closing_speed=max(float(getattr(cfg, "avoidance_prealert_closing_speed", 0.0) or 0.0), 0.0),
        avoidance_prealert_ttc=max(float(getattr(cfg, "avoidance_prealert_ttc", 0.0) or 0.0), 0.0),
        avoidance_prealert_dist_cap=(
            None if getattr(cfg, "avoidance_prealert_dist_cap", None) is None else float(getattr(cfg, "avoidance_prealert_dist_cap"))
        ),
        avoidance_repulse_mode=str(getattr(cfg, "avoidance_repulse_mode", "inverse") or "inverse").strip().lower(),
        avoidance_repulse_clip=max(float(getattr(cfg, "avoidance_repulse_clip", 1.0e3) or 1.0e3), 1.0e-6),
        avoidance_closing_gain_enabled=bool(getattr(cfg, "avoidance_closing_gain_enabled", False)),
        avoidance_closing_gain_cap=max(float(getattr(cfg, "avoidance_closing_gain_cap", 2.0) or 2.0), 1.0),
        avoidance_closing_gain_top1_only=bool(getattr(cfg, "avoidance_closing_gain_top1_only", False)),
        avoidance_eta_min=max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0),
        avoidance_eta_max=max(
            max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0),
            float(cfg.a_max if getattr(cfg, "avoidance_eta_max", None) is None else getattr(cfg, "avoidance_eta_max")),
        ),
        avoidance_eta=float(getattr(cfg, "avoidance_eta", 0.0) or 0.0),
        centroid_cross_anneal_enabled=bool(getattr(cfg, "centroid_cross_anneal_enabled", False)),
        eta_centroid=float(getattr(cfg, "eta_centroid", 0.0) or 0.0),
        eta_centroid_final=(
            None if getattr(cfg, "eta_centroid_final", None) is None else float(getattr(cfg, "eta_centroid_final"))
        ),
        eta_centroid_decay_steps=int(getattr(cfg, "eta_centroid_decay_steps", 0) or 0),
        centroid_cross_avoidance_gain=float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0),
        pairwise_hard_distance=(
            None if getattr(cfg, "pairwise_hard_distance", None) is None else float(getattr(cfg, "pairwise_hard_distance"))
        ),
        pairwise_hard_trigger_mode=str(getattr(cfg, "pairwise_hard_trigger_mode", "distance") or "distance").strip().lower(),
        pairwise_hard_trigger_distance=(
            None
            if getattr(cfg, "pairwise_hard_trigger_distance", None) is None
            else float(getattr(cfg, "pairwise_hard_trigger_distance"))
        ),
        pairwise_hard_trigger_ttc=max(float(getattr(cfg, "pairwise_hard_trigger_ttc", 0.0) or 0.0), 0.0),
        pairwise_hard_closing_speed=max(float(getattr(cfg, "pairwise_hard_closing_speed", 0.0) or 0.0), 0.0),
        pairwise_hard_max_passes=max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1),
        pairwise_hard_single_pair_only=bool(getattr(cfg, "pairwise_hard_single_pair_only", True)),
        safety_shield_native_enabled=bool(getattr(cfg, "safety_shield_enabled", False))
        and str(getattr(cfg, "safety_shield_solver", "") or "").strip().upper() == "NATIVE_CUDA",
        safety_shield_tensor_iters=max(int(getattr(cfg, "safety_shield_tensor_iters", 8) or 8), 1),
        safety_shield_brake_rho=max(float(getattr(cfg, "safety_shield_brake_rho", 0.8) or 0.8), 1.0e-6),
        safety_shield_a_safe=max(float(getattr(cfg, "safety_shield_a_safe", 0.0) or 0.0), 0.0),
        safety_shield_distance_buffer=max(float(getattr(cfg, "safety_shield_distance_buffer", 0.0) or 0.0), 0.0),
        safety_shield_step_gain=max(float(getattr(cfg, "safety_shield_tensor_step_gain", 1.0) or 1.0), 0.0),
        safety_shield_tolerance=max(float(getattr(cfg, "safety_shield_tolerance", 1.0e-5) or 1.0e-5), 0.0),
    )
    bw_link_p = _NativeBwLinkStaticParams(
        num_sat=int(cfg.num_sat),
        tau0=float(cfg.tau0),
        backhaul_rate_quantum=float(_semantic_quantum(cfg, "structured_backhaul_rate_quantum", 32.0)),
        b_backhaul_per_sat=float(
            getattr(cfg, "b_sat_total", None)
            if getattr(cfg, "b_sat_total", None) is not None
            else getattr(cfg, "b_backhaul_per_sat", 0.0)
        ),
        b_backhaul_per_sat_scale=normalize_scale(
            float(
                (
                    getattr(cfg, "b_sat_total_scale", None)
                    if getattr(cfg, "b_sat_total_scale", None) is not None
                    else getattr(cfg, "b_backhaul_per_sat_scale", 1.0)
                )
                or 1.0
            )
        ),
        uav_tx_power=float(cfg.uav_tx_power),
        noise_density=float(cfg.noise_density),
        noise_figure_db=max(float(getattr(cfg, "backhaul_noise_figure_db", 3.0) or 0.0), 0.0),
        p_comm_link=float(cfg.p_comm_link),
        queue_max_sat=float(cfg.queue_max_sat),
        energy_enabled=bool(getattr(cfg, "energy_enabled", False)),
        doppler_enabled=bool(getattr(cfg, "doppler_enabled", False)),
        doppler_atten_enabled=bool(getattr(cfg, "doppler_atten_enabled", False)),
        subcarrier_spacing=float(getattr(cfg, "subcarrier_spacing", 15e3) or 15e3),
        energy_model=str(getattr(cfg, "energy_model", "simple") or "simple").strip().lower(),
        p_fly_base=float(cfg.p_fly_base),
        p_fly_coeff=float(cfg.p_fly_coeff),
        rotor_p0=float(cfg.rotor_p0),
        rotor_pi=float(cfg.rotor_pi),
        rotor_u_tip=float(cfg.rotor_u_tip),
        rotor_v0=float(cfg.rotor_v0),
        rotor_d0=float(cfg.rotor_d0),
        rotor_rho=float(cfg.rotor_rho),
        rotor_s=float(cfg.rotor_s),
        rotor_area=float(cfg.rotor_area),
    )
    reward_p = _NativeRewardMetricsStaticParams(
        reward_mode_default=str(getattr(cfg, "reward_mode", "") or ""),
        fast_float32=numeric.fast_float32,
        num_uav=int(cfg.num_uav),
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        t_steps=int(cfg.T_steps),
        tau0=float(cfg.tau0),
        a_max=float(cfg.a_max),
        v_max=float(cfg.v_max),
        uav_energy_init=float(cfg.uav_energy_init),
        n_rf=int(cfg.N_RF),
        queue_max_gu=float(cfg.queue_max_gu),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        summary_metric_quantum=numeric.summary_metric_quantum,
        bw_weighted_workload_ema_decay=min(max(float(getattr(cfg, "bw_weighted_workload_ema_decay", 0.95) or 0.0), 0.0), 1.0),
        energy_enabled=bool(cfg.energy_enabled),
        energy_model=str(getattr(cfg, "energy_model", "simple") or "simple").strip().lower(),
        p_fly_base=float(cfg.p_fly_base),
        p_fly_coeff=float(cfg.p_fly_coeff),
        p_comm_link=float(cfg.p_comm_link),
        rotor_p0=float(cfg.rotor_p0),
        rotor_pi=float(cfg.rotor_pi),
        rotor_u_tip=float(cfg.rotor_u_tip),
        rotor_v0=float(cfg.rotor_v0),
        rotor_d0=float(cfg.rotor_d0),
        rotor_rho=float(cfg.rotor_rho),
        rotor_s=float(cfg.rotor_s),
        rotor_area=float(cfg.rotor_area),
        omega_q=float(cfg.omega_q),
        omega_e=float(cfg.omega_e),
        eta_service=float(cfg.eta_service),
        eta_q_delta=float(cfg.eta_q_delta),
        eta_batt=float(cfg.eta_batt),
        eta_crash=float(cfg.eta_crash),
        eta_accel=float(cfg.eta_accel),
        eta_drop=float(getattr(cfg, "eta_drop", 0.0) or 0.0),
        eta_drop_step=float(getattr(cfg, "eta_drop_step", 0.0) or 0.0),
        eta_drop_gu=float(getattr(cfg, "eta_drop_gu", 0.0) or 0.0),
        eta_drop_uav=float(getattr(cfg, "eta_drop_uav", 0.0) or 0.0),
        eta_drop_sat=float(getattr(cfg, "eta_drop_sat", 0.0) or 0.0),
        eta_throughput_access=float(getattr(cfg, "eta_throughput_access", 0.0) or 0.0),
        eta_throughput_backhaul=float(getattr(cfg, "eta_throughput_backhaul", 0.0) or 0.0),
        eta_close_risk=float(getattr(cfg, "eta_close_risk", 0.0) or 0.0),
        reward_w_access=float(getattr(cfg, "reward_w_access", 0.0) or 0.0),
        reward_w_relay=float(getattr(cfg, "reward_w_relay", 0.0) or 0.0),
        reward_w_pre_backlog=float(getattr(cfg, "reward_w_pre_backlog", 0.0) or 0.0),
        reward_w_pre_drop=float(getattr(cfg, "reward_w_pre_drop", 0.0) or 0.0),
        reward_w_pre_service_gap=float(getattr(cfg, "reward_w_pre_service_gap", 0.0) or 0.0),
        reward_w_pre_overflow_risk=float(getattr(cfg, "reward_w_pre_overflow_risk", 0.0) or 0.0),
        throughput_only_access_coef=float(getattr(cfg, "throughput_only_access_coef", 1.0) or 0.0),
        throughput_only_backhaul_coef=float(getattr(cfg, "throughput_only_backhaul_coef", 1.0) or 0.0),
        throughput_only_gu_queue_coef=float(getattr(cfg, "throughput_only_gu_queue_coef", 1.0) or 0.0),
        queue_penalty_mode=str(getattr(cfg, "queue_penalty_mode", "linear") or "linear").strip().lower(),
        queue_delta_mode=str(getattr(cfg, "queue_delta_mode", "delta") or "delta").strip().lower(),
        queue_norm_k=float(getattr(cfg, "queue_norm_K", 1.0) or 1.0),
        queue_norm_arrival_floor=float(getattr(cfg, "queue_norm_arrival_floor", 1.0) or 1.0),
        queue_reward_use_arrival_norm=bool(getattr(cfg, "queue_reward_use_arrival_norm", False)),
        queue_log_k=float(getattr(cfg, "queue_log_k", 1.0) or 1.0),
        omega_q_gu=float(getattr(cfg, "omega_q_gu", 0.0) or 0.0),
        omega_q_uav=float(getattr(cfg, "omega_q_uav", 0.0) or 0.0),
        omega_q_sat=float(getattr(cfg, "omega_q_sat", 0.0) or 0.0),
        omega_q_tail=float(getattr(cfg, "omega_q_tail", 0.0) or 0.0),
        q_norm_tail_q0=float(getattr(cfg, "q_norm_tail_q0", 1.0) or 1.0),
        tail_q_small=float(getattr(cfg, "tail_q_small", 1.0) or 1.0),
        tail_eta_accel_gain=float(getattr(cfg, "tail_eta_accel_gain", 0.0) or 0.0),
        centroid_cross_anneal_enabled=bool(getattr(cfg, "centroid_cross_anneal_enabled", False)),
        eta_centroid=float(getattr(cfg, "eta_centroid", 0.0) or 0.0),
        eta_centroid_final=(
            None if getattr(cfg, "eta_centroid_final", None) is None else float(getattr(cfg, "eta_centroid_final"))
        ),
        eta_centroid_decay_steps=int(getattr(cfg, "eta_centroid_decay_steps", 0) or 0),
        centroid_dist_scale=float(getattr(cfg, "centroid_dist_scale", 1.0) or 1.0),
        centroid_cross_queue_gain=float(getattr(cfg, "centroid_cross_queue_gain", 0.0) or 0.0),
        centroid_cross_q_delta_gain=float(getattr(cfg, "centroid_cross_q_delta_gain", 0.0) or 0.0),
        centroid_cross_crash_gain=float(getattr(cfg, "centroid_cross_crash_gain", 0.0) or 0.0),
        use_queue_log_smoothing=bool(ablation_flag(cfg, "use_queue_log_smoothing", default=False)),
        use_active_queue_delta=bool(
            ablation_flag(
                cfg,
                "use_active_queue_delta",
                fallback_attr="queue_delta_use_active",
                default=False,
            )
        ),
        use_energy_reward=bool(ablation_flag(cfg, "use_energy_reward", default=bool(cfg.energy_enabled))),
        use_reward_tanh=bool(
            ablation_flag(cfg, "use_reward_tanh", fallback_attr="reward_tanh_enabled", default=False)
        ),
    )
    reward_mode_norm = str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower()
    bw_flow_proxy_mode_code = {
        "controllable_flow": 1,
        "weighted_workload_level": 2,
        "weighted_workload_delta": 3,
        "relative_weighted_workload_delta": 4,
    }.get(reward_mode_norm, 0)
    bw_flow_proxy_p = _NativeBwFlowProxyStaticParams(
        enabled=bool(
            bool(getattr(cfg, "bw_flow_proxy_aux_enabled", False))
            or bool(getattr(cfg, "bw_counterfactual_credit_enabled", False))
            or bool(getattr(cfg, "bw_marginal_teacher_sample_enabled", False))
            or bool(getattr(cfg, "structured_bw_per_slot_surrogate_enabled", False))
        ),
        reward_mode_code=int(bw_flow_proxy_mode_code),
        aux_delta=max(float(getattr(cfg, "bw_flow_proxy_aux_delta", 0.05) or 0.0), 0.0),
        eps=1.0e-8,
    )
    post_p = _NativePostStatsSafetyStaticParams(
        num_uav=int(cfg.num_uav),
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        tau0=float(cfg.tau0),
        queue_max_gu=float(cfg.queue_max_gu),
        queue_max_uav=float(cfg.queue_max_uav),
        queue_max_sat=float(cfg.queue_max_sat),
        overflow_risk_threshold_frac=float(getattr(cfg, "overflow_risk_threshold_frac", 0.85) or 0.85),
        overflow_risk_arrival_coef=float(getattr(cfg, "overflow_risk_arrival_coef", 1.0) or 0.0),
        overflow_risk_service_coef=float(getattr(cfg, "overflow_risk_service_coef", 1.0) or 0.0),
        service_gap_cap_steps=bw_queue_p.service_gap_cap_steps,
        service_gap_risk_threshold_steps=float(getattr(cfg, "service_gap_risk_threshold_steps", 4.0) or 4.0),
        a_max=float(cfg.a_max),
        d_safe=float(cfg.d_safe),
        avoidance_alert_factor=float(cfg.avoidance_alert_factor),
        close_risk_enabled=bool(getattr(cfg, "close_risk_enabled", True)),
        close_risk_cap=max(float(getattr(cfg, "close_risk_cap", 1.0) or 1.0), 1.0e-6),
        avoidance_prealert_factor=accel_p.avoidance_prealert_factor,
        avoidance_prealert_mode=accel_p.avoidance_prealert_mode,
        avoidance_prealert_closing_speed=accel_p.avoidance_prealert_closing_speed,
        avoidance_prealert_ttc=accel_p.avoidance_prealert_ttc,
        avoidance_prealert_dist_cap=accel_p.avoidance_prealert_dist_cap,
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_trigger_mode=str(
            getattr(cfg, "danger_imitation_trigger_mode", "intervention_any") or "intervention_any"
        ).strip().lower(),
        danger_imitation_close_risk_thresh=float(getattr(cfg, "danger_imitation_close_risk_thresh", 0.0) or 0.0),
        danger_imitation_intervention_thresh=float(getattr(cfg, "danger_imitation_intervention_thresh", 0.0) or 0.0),
    )
    random_p = _NativeRandomTapeStaticParams(
        num_uav=int(cfg.num_uav),
        num_gu=int(cfg.num_gu),
        num_sat=int(cfg.num_sat),
        deadline_enabled=bool(getattr(cfg, "deadline_enabled", False)),
        doppler_precomp_mode=str(getattr(cfg, "doppler_precomp_mode", "none") or "none").strip().lower(),
        doppler_residual_sigma_hz=float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0),
        doppler_residual_rho=float(getattr(cfg, "doppler_residual_rho", 0.0) or 0.0),
        doppler_residual_cap_hz=float(getattr(cfg, "doppler_residual_cap_hz", 0.0) or 0.0),
    )
    history_p = _NativeHistoryOutputStaticParams(
        copy_graph_outputs=bool(
            getattr(cfg, "structured_kernel_compile_cudagraphs", False)
            and not bool(getattr(cfg, "structured_kernel_cudagraph_direct_inputs", False))
        ),
        strict_cuda_contract=bool(getattr(cfg, "structured_native_main_kernel_require_compiled_segments", False)),
        cudagraph_direct_inputs=bool(getattr(cfg, "structured_kernel_cudagraph_direct_inputs", False)),
        runtime_snapshots_enabled=_native_history_snapshots_required_from_cfg(cfg),
    )
    compile_p = _NativeCompileCaptureStaticParams(
        require_compiled_segments=bool(getattr(cfg, "structured_native_main_kernel_require_compiled_segments", False)),
        cudagraph_direct_inputs=bool(getattr(cfg, "structured_kernel_cudagraph_direct_inputs", False)),
        operator_mode=str(getattr(cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower(),
        compile_backend=str(getattr(cfg, "structured_kernel_compile_backend", "") or ""),
    )
    return _NativeMainKernelTypedDomains(
        shape=shape,
        numeric=numeric,
        channel=channel_p,
        candidate=candidate_p,
        access_rate=access_p,
        sat_geometry=sat_p,
        local_obs=local_obs_p,
        global_state=global_state_p,
        accel_safety=accel_p,
        bw_queue_deadline=bw_queue_p,
        bw_link=bw_link_p,
        bw_workload=bw_workload_p,
        reward_metrics=reward_p,
        bw_flow_proxy=bw_flow_proxy_p,
        post_stats_safety=post_p,
        random_tape=random_p,
        history_output=history_p,
        compile_capture=compile_p,
    )


def _native_local_obs_user_proxy_feature_dim(local_obs_params: _NativeLocalObsStaticParams) -> int:
    p = local_obs_params
    return int(
        int(bool(p.obs_user_include_arrival_rate))
        + int(bool(p.obs_user_include_recent_arrival))
        + int(bool(p.obs_user_include_recent_service))
        + int(bool(p.obs_user_include_queue_headroom))
        + int(bool(p.obs_user_include_local_gu_service_cost))
        + int(bool(p.obs_user_include_assoc_uav_cost))
        + int(bool(p.obs_user_include_assoc_sat_cost_mean))
        + int(bool(p.obs_user_include_weighted_queue_cost))
        + int(bool(p.obs_user_include_weighted_queue_cost_relative))
        + int(bool(p.obs_user_include_urgency_risk))
        + int(bool(p.obs_user_include_downstream_pressure))
        + int(bool(p.obs_user_include_service_gap))
        + int(bool(p.obs_user_include_service_gap_risk))
        + int(bool(p.obs_user_include_deadline_slack))
        + int(bool(p.obs_user_include_deadline_risk))
    )


@dataclass(frozen=True)
class StructuredNativeModuleShapeSpec:
    accel_ego_dim: int
    accel_cell_dim: int
    accel_gu_token_dim: int
    accel_peer_token_dim: int
    accel_sat_token_dim: int
    accel_sat_width: int
    accel_gu_query_count: int
    accel_peer_query_count: int
    accel_sat_query_count: int
    uav_node_dim: int
    gu_node_dim: int
    sat_node_dim: int
    uav_gu_edge_dim: int
    uav_sat_edge_dim: int
    uav_uav_edge_dim: int
    user_node_dim: int
    user_edge_dim: int
    sat_edge_dim: int
    sats_obs_width: int
    users_obs_max: int
    visible_sats_max: int
    sat_num_select: int
    bw_action_dim: int


def native_module_shape_spec_from_config(cfg: Any) -> StructuredNativeModuleShapeSpec:
    """Static actor/critic shape ABI for the native tensor main kernel.

    This mirrors the native rollout workspace dimensions without creating an
    environment, sampling random tapes, or running a staged prepare probe.
    """

    accel_sat_width = _accel_sat_width_from_config(cfg)
    domains = _native_typed_domains_from_cfg(cfg, num_envs=1)
    shape_p = domains.shape
    uav_node_dim = critic_schema.CRITIC_UAV_NODE_DIM
    user_node_dim = critic_schema.CRITIC_GU_NODE_DIM
    sat_node_dim = critic_schema.CRITIC_SAT_NODE_DIM
    active_width = min(
        int(shape_p.num_sat),
        max(int(shape_p.visible_sats_max), 0) * int(shape_p.num_uav),
    )
    sat_obs_width = min(int(active_width), max(int(shape_p.visible_sats_max), 0))
    return StructuredNativeModuleShapeSpec(
        accel_ego_dim=accel_schema.ACCEL_EGO_DIM,
        accel_cell_dim=accel_schema.ACCEL_CELL_DIM,
        accel_gu_token_dim=accel_schema.ACCEL_GU_TOKEN_DIM,
        accel_peer_token_dim=accel_schema.ACCEL_PEER_TOKEN_DIM,
        accel_sat_token_dim=accel_schema.ACCEL_SAT_TOKEN_DIM,
        accel_sat_width=int(accel_sat_width),
        accel_gu_query_count=int(getattr(cfg, "accel_gu_query_count", accel_schema.ACCEL_GU_QUERY_COUNT)),
        accel_peer_query_count=int(getattr(cfg, "accel_peer_query_count", accel_schema.ACCEL_PEER_QUERY_COUNT)),
        accel_sat_query_count=int(getattr(cfg, "accel_sat_query_count", accel_schema.ACCEL_SAT_QUERY_COUNT)),
        uav_node_dim=int(uav_node_dim),
        gu_node_dim=int(user_node_dim),
        sat_node_dim=int(sat_node_dim),
        uav_gu_edge_dim=critic_schema.CRITIC_UAV_GU_EDGE_DIM,
        uav_sat_edge_dim=critic_schema.CRITIC_UAV_SAT_EDGE_DIM,
        uav_uav_edge_dim=critic_schema.CRITIC_UAV_UAV_EDGE_DIM,
        user_node_dim=int(user_node_dim),
        user_edge_dim=critic_schema.CRITIC_UAV_GU_EDGE_DIM,
        sat_edge_dim=critic_schema.CRITIC_UAV_SAT_EDGE_DIM,
        sats_obs_width=int(sat_obs_width),
        users_obs_max=int(shape_p.users_obs_max),
        visible_sats_max=int(shape_p.visible_sats_max),
        sat_num_select=int(shape_p.sat_num_select),
        bw_action_dim=int(shape_p.num_gu),
    )


class _NativeStageTensorFields(NamedTuple):
    stage_id: torch.Tensor
    effective_b_backhaul_per_sat: torch.Tensor
    uav_pos: torch.Tensor
    uav_vel: torch.Tensor
    uav_energy: torch.Tensor
    uav_queue: torch.Tensor
    gu_pos: torch.Tensor
    gu_queue: torch.Tensor
    arrival_ref_bits_per_step: torch.Tensor
    expected_arrival_rate_vec: torch.Tensor
    gu_ema: torch.Tensor
    uav_ema: torch.Tensor
    gu_drop: torch.Tensor
    uav_drop: torch.Tensor
    last_gu_arrival: torch.Tensor
    last_gu_outflow: torch.Tensor
    last_gu_to_uav_inflow_by_uav: torch.Tensor
    last_uav_to_sat_outflow_matrix: torch.Tensor
    last_bw_fraction_by_uav_gu: torch.Tensor
    last_access_interference_by_uav: torch.Tensor
    sat_queue: torch.Tensor
    sat_loads: torch.Tensor
    sat_ema: torch.Tensor
    sat_drop: torch.Tensor
    last_sat_processed: torch.Tensor
    last_selected_mask_by_uav_sat: torch.Tensor
    sat_pos: torch.Tensor
    sat_vel: torch.Tensor
    assoc: torch.Tensor
    prev_association: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    bw_valid_mask: torch.Tensor
    sat_selection_matrix: torch.Tensor
    candidate_flag: torch.Tensor
    bw_valid_flag: torch.Tensor
    prev_assoc_flag: torch.Tensor
    eta_ref_feature: torch.Tensor
    eta_slots: torch.Tensor
    gu_proxy_features: torch.Tensor
    uav_assoc_uav_cost: torch.Tensor
    sat_cost_norm: torch.Tensor
    access_gain_matrix: torch.Tensor
    visible_ids: torch.Tensor
    visible_mask: torch.Tensor
    visible_flag_all: torch.Tensor
    elevation_matrix: torch.Tensor
    uav_ecef_all: torch.Tensor
    uav_vel_ecef_all: torch.Tensor
    active_sat_ids: torch.Tensor
    sat_pos_active: torch.Tensor
    sat_vel_active: torch.Tensor
    sat_queue_active: torch.Tensor
    sat_load_active: torch.Tensor
    sat_cost_norm_active: torch.Tensor
    us_rel_pos_active: torch.Tensor
    us_rel_vel_active: torch.Tensor
    us_gain_active: torch.Tensor
    us_nu_eff_active: torch.Tensor
    visible_flag_active: torch.Tensor
    us_valid_flag_active: torch.Tensor
    us_rel_pos_all: torch.Tensor | None = None
    us_rel_vel_all: torch.Tensor | None = None
    us_gain_all: torch.Tensor | None = None
    us_nu_eff_all: torch.Tensor | None = None
    us_valid_flag_all: torch.Tensor | None = None
    us_sat_queue_all: torch.Tensor | None = None


@dataclass
class _NativeTrainingWorldTensorFields:
    uav_nodes: torch.Tensor
    gu_nodes: torch.Tensor
    sat_nodes: torch.Tensor
    sat_ids: torch.Tensor
    uav_gu_edges: torch.Tensor
    uav_sat_edges: torch.Tensor
    uav_uav_edges: torch.Tensor
    global_scalars: torch.Tensor
    gu_mask: torch.Tensor
    sat_mask: torch.Tensor
    uav_gu_mask: torch.Tensor
    uav_sat_mask: torch.Tensor
    uav_uav_mask: torch.Tensor


class _NativeAccelTrainingHistoryOutBuffers(NamedTuple):
    world_batch: _NativeTrainingWorldTensorFields
    ego_features: torch.Tensor
    ego_cell: torch.Tensor
    gu_tokens: torch.Tensor
    gu_mask: torch.Tensor
    peer_tokens: torch.Tensor
    peer_mask: torch.Tensor
    sat_tokens: torch.Tensor
    sat_mask: torch.Tensor
    danger_imitation_targets: torch.Tensor | None = None
    danger_imitation_masks: torch.Tensor | None = None


class _NativeSatTrainingHistoryOutBuffers(NamedTuple):
    world_batch: _NativeTrainingWorldTensorFields
    ego_features: torch.Tensor
    demand_features: torch.Tensor
    role_features: torch.Tensor
    sat_tokens: torch.Tensor
    sat_mask: torch.Tensor
    sat_valid_mask: torch.Tensor
    candidate_sat_ids: torch.Tensor
    subset_mask: torch.Tensor
    subset_members: torch.Tensor


class _NativeBwTrainingHistoryOutBuffers(NamedTuple):
    world_batch: _NativeTrainingWorldTensorFields
    ego_features: torch.Tensor
    selected_sat_tokens: torch.Tensor
    selected_sat_mask: torch.Tensor
    gu_tokens: torch.Tensor
    gu_mask: torch.Tensor
    bw_valid_mask: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_next_world_mask: torch.Tensor | None = None
    bw_access_rewards: torch.Tensor | None = None
    bw_weighted_workload_delta_rewards: torch.Tensor | None = None
    bw_weighted_workload_level_rewards: torch.Tensor | None = None
    bw_gu_queue_level_rewards: torch.Tensor | None = None
    bw_system_queue_level_rewards: torch.Tensor | None = None
    bw_gu_service_queue_rewards: torch.Tensor | None = None
    bw_flow_proxy_scores: torch.Tensor | None = None
    bw_flow_proxy_masks: torch.Tensor | None = None
    bw_flow_proxy_deltas: torch.Tensor | None = None


def _accel_sat_width_from_config(cfg: Any) -> int:
    raw_width = getattr(cfg, "per_uav_visible_sat_token_max", None)
    if raw_width is None:
        raise ValueError("per_uav_visible_sat_token_max must be finalized before building accel actor shapes.")
    width = int(raw_width)
    if width <= 0:
        raise ValueError("per_uav_visible_sat_token_max must be positive for accel actor.")
    visible_ref = getattr(cfg, "visible_sats_max", None)
    visible_width = int(visible_ref) if visible_ref is not None else int(getattr(cfg, "sats_obs_max", 0) or 0)
    derived = min(int(getattr(cfg, "num_sat", 0) or 0), visible_width)
    if width != derived:
        raise ValueError(
            "per_uav_visible_sat_token_max must equal min(num_sat, visible_sats_max or sats_obs_max) "
            f"for accel actor; got {width}, expected {derived}."
        )
    return width


def _native_actor_scratch_width_from_config(cfg: Any, *, sat_obs_width: int, subset_count: int, accel_sat_width: int) -> int:
    actor_hidden = max(int(getattr(cfg, "actor_hidden", 256) or 256), 1)
    actor_embed = max(int(getattr(cfg, "actor_set_embed_dim", 128) or 128), 1)
    accel_hidden = max(int(getattr(cfg, "accel_hidden", 0) or actor_hidden), 1)
    accel_embed = max(int(getattr(cfg, "accel_embed_dim", 0) or actor_embed), 1)
    sat_hidden = max(int(getattr(cfg, "sat_hidden", 0) or actor_hidden), 1)
    sat_embed = max(int(getattr(cfg, "sat_embed_dim", 0) or actor_embed), 1)
    bw_hidden = max(int(getattr(cfg, "bw_hidden", 0) or actor_hidden), 1)
    bw_embed = max(int(getattr(cfg, "bw_embed_dim", 0) or actor_embed), 1)
    hidden_dim = max(accel_hidden, sat_hidden, bw_hidden)
    embed_dim = max(accel_embed, sat_embed, bw_embed, 128)
    peer_count = max(int(cfg.num_uav) - 1, 0)
    accel_gu_query_count = max(int(getattr(cfg, "accel_gu_query_count", accel_schema.ACCEL_GU_QUERY_COUNT) or accel_schema.ACCEL_GU_QUERY_COUNT), 1)
    accel_peer_query_count = max(int(getattr(cfg, "accel_peer_query_count", accel_schema.ACCEL_PEER_QUERY_COUNT) or accel_schema.ACCEL_PEER_QUERY_COUNT), 1)
    accel_sat_query_count = max(int(getattr(cfg, "accel_sat_query_count", accel_schema.ACCEL_SAT_QUERY_COUNT) or accel_schema.ACCEL_SAT_QUERY_COUNT), 1)
    accel_max_count = max(int(cfg.num_gu), peer_count, int(accel_sat_width), 1)
    accel_attention_heads = max(int(getattr(cfg, "accel_attention_heads", 1) or 1), 1)
    sat_attention_heads = max(int(getattr(cfg, "sat_attention_heads", 4) or 4), 1)
    bw_attention_heads = max(int(getattr(cfg, "structured_bw_competition_heads", getattr(cfg, "bw_attention_heads", 1)) or 1), 1)
    accel_max_in = max(
        accel_schema.ACCEL_EGO_DIM,
        accel_schema.ACCEL_CELL_DIM,
        accel_schema.ACCEL_GU_TOKEN_DIM,
        accel_schema.ACCEL_PEER_TOKEN_DIM,
        accel_schema.ACCEL_SAT_TOKEN_DIM,
        2 * accel_embed,
    )
    accel_fusion_dim = (
        2 * accel_embed
        + (accel_gu_query_count + 2) * accel_embed
        + (accel_peer_query_count + 2) * accel_embed
        + (accel_sat_query_count + 2) * accel_embed
    )
    accel_width = (
        2 * 128
        + accel_schema.ACCEL_EGO_DIM
        + accel_schema.ACCEL_CELL_DIM
        + 4 * accel_embed
        + 2
        * (
            accel_gu_query_count
            + accel_peer_query_count
            + accel_sat_query_count
        )
        * accel_embed
        + 6 * accel_embed
        + accel_fusion_dim
        + accel_hidden
        + 2 * accel_max_count * accel_hidden
        + 5 * accel_max_count * accel_embed
        + accel_max_count * accel_attention_heads * accel_max_count
        + 2 * accel_max_count * accel_max_in
        + max(peer_count, 1) * accel_embed
        + max(int(cfg.num_gu), 1) * accel_embed
        + max(int(accel_sat_width), 1) * accel_embed
        + 1024
    )
    legacy_max_items = max(
        int(cfg.num_uav),
        int(cfg.num_gu),
        int(cfg.users_obs_max),
        int(sat_obs_width),
        int(subset_count),
        1,
    )
    legacy_width = (
        int(legacy_max_items) * int(embed_dim) * 10
        + int(max(int(cfg.num_gu), int(sat_obs_width), int(accel_sat_width), 1))
        * int(max(accel_attention_heads, sat_attention_heads, bw_attention_heads, 1))
        * int(max(int(cfg.num_gu), int(sat_obs_width), int(accel_sat_width), 1))
        + int(legacy_max_items) * 16
        + int(legacy_max_items) * int(hidden_dim)
        + int(hidden_dim) * 12
        + int(embed_dim) * 96
        + 4096
    )
    return int(max(accel_width, legacy_width))


def _allocate_native_main_kernel_stage_fields(
    *,
    cfg,
    local_obs_params: _NativeLocalObsStaticParams,
    batch_size: int,
    active_width: int,
    max_keep: int,
    select_k: int,
    device: torch.device | str,
) -> _NativeStageTensorFields:
    tensor_device = torch.device(device)
    b = int(batch_size)
    u = int(cfg.num_uav)
    g = int(cfg.num_gu)
    s = int(cfg.num_sat)
    c = int(cfg.users_obs_max)
    k = int(select_k)
    a = int(active_width)
    m = int(max_keep)
    proxy_dim = _native_local_obs_user_proxy_feature_dim(local_obs_params)
    full_geometry = str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower() != "elevation"

    def f32(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float32, device=tensor_device)

    def long(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.long, device=tensor_device)

    def bool_t(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.bool, device=tensor_device)

    fields = _NativeStageTensorFields(
        stage_id=long((b,)),
        effective_b_backhaul_per_sat=f32((b,)),
        uav_pos=f32((b, u, 2)),
        uav_vel=f32((b, u, 2)),
        uav_energy=f32((b, u)),
        uav_queue=f32((b, u)),
        gu_pos=f32((b, g, 2)),
        gu_queue=f32((b, g)),
        arrival_ref_bits_per_step=f32((b,)),
        expected_arrival_rate_vec=f32((b, g)),
        gu_ema=f32((b, g)),
        uav_ema=f32((b, u)),
        gu_drop=f32((b, g)),
        uav_drop=f32((b, u)),
        last_gu_arrival=f32((b, g)),
        last_gu_outflow=f32((b, g)),
        last_gu_to_uav_inflow_by_uav=f32((b, u)),
        last_uav_to_sat_outflow_matrix=f32((b, u, s)),
        last_bw_fraction_by_uav_gu=f32((b, u, g)),
        last_access_interference_by_uav=f32((b, u)),
        sat_queue=f32((b, s)),
        sat_loads=f32((b, s)),
        sat_ema=f32((b, s)),
        sat_drop=f32((b, s)),
        last_sat_processed=f32((b, s)),
        last_selected_mask_by_uav_sat=f32((b, u, s)),
        sat_pos=f32((b, s, 3)),
        sat_vel=f32((b, s, 3)),
        assoc=long((b, g)),
        prev_association=long((b, g)),
        candidate_indices=long((b, u, c)),
        candidate_mask=bool_t((b, u, c)),
        bw_valid_mask=f32((b, u, c)),
        sat_selection_matrix=long((b, u, k)),
        candidate_flag=f32((b, u, g)),
        bw_valid_flag=f32((b, u, g)),
        prev_assoc_flag=f32((b, u, g)),
        eta_ref_feature=f32((b, u, g)),
        eta_slots=f32((b, u, c)),
        gu_proxy_features=f32((b, g, proxy_dim)),
        uav_assoc_uav_cost=f32((b, u)),
        sat_cost_norm=f32((b, s)),
        access_gain_matrix=f32((b, g, u)),
        visible_ids=long((b, u, m)),
        visible_mask=bool_t((b, u, m)),
        visible_flag_all=f32((b, u, s)),
        elevation_matrix=f32((b, u, s)),
        uav_ecef_all=f32((b, u, 3)),
        uav_vel_ecef_all=f32((b, u, 3)),
        active_sat_ids=long((b, a)),
        sat_pos_active=f32((b, a, 3)),
        sat_vel_active=f32((b, a, 3)),
        sat_queue_active=f32((b, a)),
        sat_load_active=f32((b, a)),
        sat_cost_norm_active=f32((b, a)),
        us_rel_pos_active=f32((b, u, a, 3)),
        us_rel_vel_active=f32((b, u, a, 3)),
        us_gain_active=f32((b, u, a)),
        us_nu_eff_active=f32((b, u, a)),
        visible_flag_active=f32((b, u, a)),
        us_valid_flag_active=f32((b, u, a)),
        us_rel_pos_all=f32((b, u, s, 3)) if full_geometry else None,
        us_rel_vel_all=f32((b, u, s, 3)) if full_geometry else None,
        us_gain_all=f32((b, u, s)) if full_geometry else None,
        us_nu_eff_all=f32((b, u, s)) if full_geometry else None,
        us_valid_flag_all=f32((b, u, s)) if full_geometry else None,
        us_sat_queue_all=f32((b, u, s)) if full_geometry else None,
    )

    service_floor = float(
        getattr(
            cfg,
            "service_floor_bits_per_step",
            getattr(cfg, "bw_weighted_workload_eps", 1.0),
        )
        or 1.0
    )
    fields.sat_ema.fill_(max(service_floor, 1.0e-12))
    fields.arrival_ref_bits_per_step.fill_(max(float(g) * float(getattr(cfg, "tau0", 1.0) or 1.0), 1.0e-12))
    fields.expected_arrival_rate_vec.zero_()
    fields.gu_ema.fill_(max(service_floor, 1.0e-12))
    fields.uav_ema.fill_(max(service_floor, 1.0e-12))
    fields.gu_drop.zero_()
    fields.uav_drop.zero_()
    fields.last_gu_arrival.zero_()
    fields.last_gu_outflow.zero_()
    fields.last_gu_to_uav_inflow_by_uav.zero_()
    fields.last_uav_to_sat_outflow_matrix.zero_()
    fields.last_bw_fraction_by_uav_gu.zero_()
    fields.last_access_interference_by_uav.zero_()
    fields.sat_drop.zero_()
    fields.last_sat_processed.zero_()
    fields.last_selected_mask_by_uav_sat.zero_()
    return fields


def _allocate_native_training_world_tensor_fields(
    *,
    local_obs_params: _NativeLocalObsStaticParams,
    fields_obj: _NativeStageTensorFields,
) -> _NativeTrainingWorldTensorFields:
    if not _is_native_stage_fields(fields_obj):
        raise RuntimeError("native training world allocation requires fixed stage field tensors.")
    p = local_obs_params
    batch_size = int(fields_obj.uav_pos.shape[0])
    num_uav = int(p.num_uav)
    num_gu = int(p.num_gu)
    max_sat_count = int(fields_obj.active_sat_ids.shape[1]) if fields_obj.active_sat_ids.ndim >= 2 else 0
    device = fields_obj.uav_pos.device
    return _NativeTrainingWorldTensorFields(
        uav_nodes=torch.empty((batch_size, num_uav, critic_schema.CRITIC_UAV_NODE_DIM), dtype=torch.float32, device=device),
        gu_nodes=torch.empty((batch_size, num_gu, critic_schema.CRITIC_GU_NODE_DIM), dtype=torch.float32, device=device),
        sat_nodes=torch.empty((batch_size, max_sat_count, critic_schema.CRITIC_SAT_NODE_DIM), dtype=torch.float32, device=device),
        sat_ids=torch.empty((batch_size, max_sat_count), dtype=torch.long, device=device),
        uav_gu_edges=torch.empty((batch_size, num_uav, num_gu, critic_schema.CRITIC_UAV_GU_EDGE_DIM), dtype=torch.float32, device=device),
        uav_sat_edges=torch.empty((batch_size, num_uav, max_sat_count, critic_schema.CRITIC_UAV_SAT_EDGE_DIM), dtype=torch.float32, device=device),
        uav_uav_edges=torch.empty((batch_size, num_uav, num_uav, critic_schema.CRITIC_UAV_UAV_EDGE_DIM), dtype=torch.float32, device=device),
        global_scalars=torch.empty((batch_size, critic_schema.CRITIC_GLOBAL_SCALAR_DIM), dtype=torch.float32, device=device),
        gu_mask=torch.empty((batch_size, num_gu), dtype=torch.bool, device=device),
        sat_mask=torch.empty((batch_size, max_sat_count), dtype=torch.bool, device=device),
        uav_gu_mask=torch.empty((batch_size, num_uav, num_gu), dtype=torch.bool, device=device),
        uav_sat_mask=torch.empty((batch_size, num_uav, max_sat_count), dtype=torch.bool, device=device),
        uav_uav_mask=torch.empty((batch_size, num_uav, num_uav), dtype=torch.bool, device=device),
    )


class _NativeBwDirectInputTensorFields(NamedTuple):
    assoc: torch.Tensor
    prev_association: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    bw_valid_mask: torch.Tensor
    access_gain_matrix: torch.Tensor
    sat_selection_matrix: torch.Tensor
    active_sat_ids: torch.Tensor
    gain_active: torch.Tensor
    nu_eff_active: torch.Tensor
    valid_flag_active: torch.Tensor
    sat_pos: torch.Tensor
    uav_ecef: torch.Tensor
    uav_pos: torch.Tensor
    uav_vel: torch.Tensor
    gu_pos: torch.Tensor


class _NativeBwLinkTransitionTensorFields(NamedTuple):
    uav_energy: torch.Tensor
    last_energy_cost: torch.Tensor
    rate_matrix: torch.Tensor
    sat_loads: torch.Tensor
    last_sat_score: torch.Tensor

    def __bool__(self) -> bool:
        return True

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._fields

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return tuple.__getitem__(self, key)

    def items(self):
        for key in self._fields:
            yield key, getattr(self, key)


class _NativeBwMetricsTensorFields(NamedTuple):
    gu_ema: torch.Tensor
    uav_ema: torch.Tensor
    sat_ema: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    arrival_sum: torch.Tensor
    arrival_ref: torch.Tensor
    outflow_sum: torch.Tensor
    backhaul_sum: torch.Tensor
    sat_processed_sum: torch.Tensor
    expire_sum: torch.Tensor
    gu_drop_sum: torch.Tensor
    uav_drop_sum: torch.Tensor
    sat_drop_sum: torch.Tensor
    drop_sum_active: torch.Tensor
    drop_sum: torch.Tensor
    q_gu: torch.Tensor
    q_uav: torch.Tensor
    q_sat: torch.Tensor
    q_total: torch.Tensor
    q_total_active: torch.Tensor
    service_ratio: torch.Tensor
    drop_ratio: torch.Tensor
    x_acc: torch.Tensor
    x_rel: torch.Tensor
    b_pre_t: torch.Tensor
    g_pre: torch.Tensor
    d_pre: torch.Tensor
    processed_ratio_eval: torch.Tensor
    drop_ratio_eval: torch.Tensor
    pre_backlog_steps_eval: torch.Tensor
    d_sys_report: torch.Tensor
    sat_overlap_eval: torch.Tensor
    overflow_risk_mean: torch.Tensor
    downstream_pressure_mean: torch.Tensor
    service_gap_mean: torch.Tensor
    service_gap_risk_mean: torch.Tensor
    bw_weighted_workload_delta_reward: torch.Tensor
    bw_weighted_workload_level_reward: torch.Tensor
    bw_gu_queue_level_reward: torch.Tensor
    bw_system_queue_level_reward: torch.Tensor
    bw_gu_service_queue_reward: torch.Tensor
    intervention_norm_uav: torch.Tensor
    close_risk_uav: torch.Tensor
    danger_imitation_mask: torch.Tensor
    intervention_norm: torch.Tensor
    intervention_rate: torch.Tensor
    intervention_norm_top1: torch.Tensor
    close_risk: torch.Tensor
    danger_imitation_active_rate: torch.Tensor
    collision: torch.Tensor
    reward_raw: torch.Tensor
    centroid_dist_mean: torch.Tensor
    centroid_reward: torch.Tensor
    q_norm_active: torch.Tensor
    prev_q_norm_active: torch.Tensor
    queue_delta: torch.Tensor
    term_service: torch.Tensor
    term_drop: torch.Tensor
    term_queue: torch.Tensor
    term_q_delta: torch.Tensor
    term_centroid: torch.Tensor
    term_accel: torch.Tensor
    term_close_risk: torch.Tensor
    term_energy: torch.Tensor
    collision_penalty: torch.Tensor
    battery_penalty: torch.Tensor

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key) if key in self._fields else default

    def items(self):
        for key in self._fields:
            yield key, getattr(self, key)


class _NativeBwWorkloadRewardTensorFields(NamedTuple):
    delta: torch.Tensor
    level: torch.Tensor
    positive_level: torch.Tensor
    gu_queue_level: torch.Tensor
    system_queue_level: torch.Tensor
    gu_service_queue: torch.Tensor


class _NativeCloseRiskSafetyTensorFields(NamedTuple):
    intervention_norm_uav: torch.Tensor
    intervention_norm: torch.Tensor
    intervention_rate: torch.Tensor
    intervention_norm_top1: torch.Tensor
    close_risk: torch.Tensor
    close_risk_uav: torch.Tensor
    danger_imitation_mask: torch.Tensor
    danger_imitation_active_rate: torch.Tensor
    collision: torch.Tensor


class _NativeVisibleSatScoreTensorFields(NamedTuple):
    elevation: torch.Tensor
    above_mask: torch.Tensor
    score: torch.Tensor


class _NativeRewardAlignedGuFeatureTensorFields(NamedTuple):
    local_gu_service_cost: torch.Tensor
    assoc_uav_cost: torch.Tensor
    assoc_sat_cost_mean: torch.Tensor
    weighted_queue_cost: torch.Tensor
    weighted_queue_cost_relative: torch.Tensor


class _NativePreparedSatGeometryTensorFields(NamedTuple):
    top_idx: torch.Tensor
    top_mask: torch.Tensor
    active_ids: torch.Tensor
    visible_flag: torch.Tensor
    elevation: torch.Tensor
    uav_ecef: torch.Tensor
    uav_vel_ecef: torch.Tensor
    sat_pos_active: torch.Tensor
    sat_vel_active: torch.Tensor
    sat_queue_active: torch.Tensor
    sat_load_active: torch.Tensor
    us_rel_pos_active: torch.Tensor
    us_rel_vel_active: torch.Tensor
    us_gain_active: torch.Tensor
    us_nu_eff_active: torch.Tensor
    visible_flag_active: torch.Tensor
    us_valid_flag_active: torch.Tensor


class _NativeBwRuntimeStateOutBuffers(NamedTuple):
    uav_pos: torch.Tensor
    uav_vel: torch.Tensor
    gu_pos: torch.Tensor
    prev_queue_sum_gu: torch.Tensor
    prev_queue_sum_uav: torch.Tensor
    prev_queue_sum_sat: torch.Tensor
    prev_gu_queue_vec: torch.Tensor
    prev_uav_queue_vec: torch.Tensor
    prev_sat_queue_vec: torch.Tensor
    gu_queue: torch.Tensor
    uav_queue: torch.Tensor
    sat_queue: torch.Tensor
    uav_energy: torch.Tensor
    last_association: torch.Tensor
    last_sat_selection_matrix: torch.Tensor
    last_sat_connection_counts: torch.Tensor
    last_gu_service_gap: torch.Tensor
    last_gu_deadline_age: torch.Tensor
    last_gu_arrival: torch.Tensor
    last_gu_arrival_rate_vec: torch.Tensor
    last_gu_outflow: torch.Tensor
    last_gu_deadline_slack: torch.Tensor
    last_gu_deadline_risk: torch.Tensor
    last_gu_urgency_risk: torch.Tensor
    last_gu_downstream_pressure: torch.Tensor
    last_gu_service_gap_risk: torch.Tensor
    last_exec_accel: torch.Tensor
    last_policy_accel: torch.Tensor
    gu_workload_ema: torch.Tensor
    uav_workload_ema: torch.Tensor
    sat_workload_ema: torch.Tensor
    arrival_ref_bits_per_step: torch.Tensor
    effective_task_arrival_rate: torch.Tensor
    gu_deadline_steps: torch.Tensor
    arrival_base_scale: torch.Tensor
    hotspot_active_idx: torch.Tensor
    hotspot_subset_count: torch.Tensor
    hotspot_member_mask: torch.Tensor
    traffic_reset_step: torch.Tensor
    traffic_reset_ordinal: torch.Tensor
    episode_idx: torch.Tensor
    t: torch.Tensor
    prev_q_norm_active: torch.Tensor
    doppler_residual: torch.Tensor | None = None
    sat_pos: torch.Tensor | None = None
    sat_vel: torch.Tensor | None = None


class _NativeBwRewardPartOutBuffers(NamedTuple):
    service_ratio: torch.Tensor
    drop_ratio: torch.Tensor
    arrival_ref: torch.Tensor
    b_pre_steps: torch.Tensor
    x_acc: torch.Tensor
    x_rel: torch.Tensor
    g_pre: torch.Tensor
    d_pre: torch.Tensor
    processed_ratio_eval: torch.Tensor
    drop_ratio_eval: torch.Tensor
    pre_backlog_steps_eval: torch.Tensor
    sat_overlap_eval: torch.Tensor
    D_sys_report: torch.Tensor
    drop_sum: torch.Tensor
    gu_queue_sum: torch.Tensor
    uav_queue_sum: torch.Tensor
    sat_queue_sum: torch.Tensor
    queue_total_sum: torch.Tensor
    drop_sum_active: torch.Tensor
    expire_sum: torch.Tensor
    gu_drop_sum: torch.Tensor
    uav_drop_sum: torch.Tensor
    sat_drop_sum: torch.Tensor
    arrival_sum: torch.Tensor
    outflow_sum: torch.Tensor
    backhaul_sum: torch.Tensor
    sat_processed_sum: torch.Tensor
    collision_event: torch.Tensor
    overflow_risk_mean: torch.Tensor
    downstream_pressure_mean: torch.Tensor
    service_gap_mean: torch.Tensor
    service_gap_risk_mean: torch.Tensor
    bw_weighted_workload_delta_reward: torch.Tensor
    bw_weighted_workload_level_reward: torch.Tensor
    bw_gu_queue_level_reward: torch.Tensor
    bw_system_queue_level_reward: torch.Tensor
    bw_gu_service_queue_reward: torch.Tensor
    intervention_norm: torch.Tensor
    intervention_rate: torch.Tensor
    intervention_norm_top1: torch.Tensor
    danger_imitation_active_rate: torch.Tensor
    close_risk: torch.Tensor
    term_close_risk: torch.Tensor
    reward_raw: torch.Tensor


@dataclass
class StructuredBatchRuntimeTensorState:
    uav_pos: torch.Tensor
    uav_vel: torch.Tensor
    uav_energy: torch.Tensor
    uav_queue: torch.Tensor
    gu_pos: torch.Tensor
    gu_cluster_centers: torch.Tensor
    gu_cluster_counts: torch.Tensor
    gu_queue: torch.Tensor
    sat_queue: torch.Tensor
    sat_pos: torch.Tensor
    sat_vel: torch.Tensor
    prev_association: torch.Tensor
    last_association: torch.Tensor
    last_sat_selection_matrix: torch.Tensor
    last_sat_connection_counts: torch.Tensor
    arrival_ref_bits_per_step: torch.Tensor
    effective_task_arrival_rate: torch.Tensor
    arrival_base_scale: torch.Tensor
    hotspot_active_idx: torch.Tensor
    hotspot_subset_count: torch.Tensor
    hotspot_member_mask: torch.Tensor
    traffic_reset_step: torch.Tensor
    traffic_reset_ordinal: torch.Tensor
    episode_idx: torch.Tensor
    gu_workload_ema: torch.Tensor
    uav_workload_ema: torch.Tensor
    sat_workload_ema: torch.Tensor
    last_gu_arrival_rate_vec: torch.Tensor
    gu_deadline_steps: torch.Tensor
    last_gu_arrival: torch.Tensor
    last_gu_outflow: torch.Tensor
    gu_drop: torch.Tensor
    uav_drop: torch.Tensor
    sat_drop: torch.Tensor
    last_access_interference_by_uav: torch.Tensor
    last_bw_fraction_by_uav_gu: torch.Tensor
    last_gu_to_uav_inflow_by_uav: torch.Tensor
    last_uav_to_sat_outflow_matrix: torch.Tensor
    last_selected_mask_by_uav_sat: torch.Tensor
    last_sat_processed: torch.Tensor
    last_gu_urgency_risk: torch.Tensor
    last_gu_downstream_pressure: torch.Tensor
    last_gu_service_gap_risk: torch.Tensor
    last_gu_deadline_slack: torch.Tensor
    last_gu_deadline_risk: torch.Tensor
    last_gu_service_gap: torch.Tensor
    last_gu_deadline_age: torch.Tensor
    last_exec_accel: torch.Tensor
    last_policy_accel: torch.Tensor
    avoidance_eta_eff: torch.Tensor
    last_avoidance_eta_exec: torch.Tensor
    doppler_residual: torch.Tensor
    prev_queue_sum_gu: torch.Tensor
    prev_queue_sum_uav: torch.Tensor
    prev_queue_sum_sat: torch.Tensor
    prev_q_norm_active: torch.Tensor
    prev_gu_queue_vec: torch.Tensor
    prev_uav_queue_vec: torch.Tensor
    prev_sat_queue_vec: torch.Tensor
    t: torch.Tensor
    global_step: torch.Tensor


class _NativeMainKernelWorkspaceExecutor:
    """Executor proxy that runs a native program against a private workspace."""

    def __init__(
        self,
        core: "StructuredBatchEnvCore",
        *,
        runtime: StructuredGpuRolloutRuntime,
        tensor_state: StructuredBatchRuntimeTensorState,
        cfg: Any,
        rng: torch.Generator,
        bound_kernels: dict[str, Any] | None = None,
        num_envs_override: int | None = None,
        slot_state_payloads: list[dict[str, Any]] | None = None,
        slot_rngs: list[np.random.Generator] | None = None,
    ) -> None:
        self._core = core
        self._runtime = runtime
        self._tensor_state = tensor_state
        self._cfg = cfg
        self._rng = rng
        self._typed_domains = getattr(core, "_native_main_kernel_typed_domains_obj", None)
        self._bound_kernels: dict[str, Any] = bound_kernels if bound_kernels is not None else {}
        self._num_envs_override = None if num_envs_override is None else int(num_envs_override)
        self._slot_state_payloads = slot_state_payloads
        self._slot_rngs = slot_rngs

    @contextmanager
    def _active(self):
        with self._core._native_main_kernel_workspace_context(
            runtime=self._runtime,
            tensor_state=self._tensor_state,
            cfg=self._cfg,
            rng=self._rng,
            bound_kernels=self._bound_kernels,
            num_envs_override=self._num_envs_override,
            slot_state_payloads=self._slot_state_payloads,
            slot_rngs=self._slot_rngs,
        ):
            if self._typed_domains is not None:
                setattr(self._core, "_native_main_kernel_typed_domains_obj", self._typed_domains)
            if self._bound_kernels is not None:
                setattr(self._core, "_native_main_kernel_bound_kernels", self._bound_kernels)
            try:
                yield
            finally:
                self._typed_domains = getattr(self._core, "_native_main_kernel_typed_domains_obj", None)
                current_bound = getattr(self._core, "_native_main_kernel_bound_kernels", None)
                if isinstance(current_bound, dict) and current_bound is not self._bound_kernels:
                    self._bound_kernels.clear()
                    self._bound_kernels.update(current_bound)
                elif isinstance(current_bound, dict):
                    self._bound_kernels = current_bound

    def _runtime_step_select_rollout_storage(self, *, num_envs: int) -> None:
        with self._active():
            self._core._runtime_step_select_rollout_storage(num_envs=num_envs)

    def _runtime_begin_horizon(self, *, num_steps: int) -> None:
        with self._active():
            self._core._runtime_begin_horizon(num_steps=num_steps)

    def _runtime_step_begin_accel_obs(self):
        with self._active():
            return self._core._runtime_step_begin_accel_obs()

    def _runtime_step_publish_sat_obs(self, **kwargs):
        with self._active():
            return self._core._runtime_step_publish_sat_obs(**kwargs)

    def _runtime_step_publish_bw_obs(self, **kwargs):
        with self._active():
            return self._core._runtime_step_publish_bw_obs(**kwargs)

    def _runtime_step_apply_bw_macro_live(self, **kwargs):
        with self._active():
            return self._core._runtime_step_apply_bw_macro_live(**kwargs)

    def _runtime_step_finish_bw(self, **kwargs):
        with self._active():
            return self._core._runtime_step_finish_bw(**kwargs)

_SAT_SUBSET_MEMBER_TENSOR_CACHE: dict[tuple[int, int, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}


def _is_native_stage_fields(value: Any) -> bool:
    return isinstance(value, _NativeStageTensorFields)




def _stage_fields_batch_size(fields: _NativeStageTensorFields) -> int:
    for field_name in fields._fields:
        value = getattr(fields, field_name)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("native stage fields do not contain a batched tensor.")


def _stage_fields_sample_tensor(fields: _NativeStageTensorFields) -> torch.Tensor | None:
    for field_name in fields._fields:
        value = getattr(fields, field_name)
        if torch.is_tensor(value):
            return value
    return None


def _stage_fields_with_updates(
    fields: _NativeStageTensorFields,
    updates: Mapping[str, torch.Tensor] | None,
) -> _NativeStageTensorFields:
    if not updates:
        return fields
    valid_updates = {
        str(key): value
        for key, value in updates.items()
        if str(key) in fields._fields and torch.is_tensor(value)
    }
    if not valid_updates:
        return fields
    return fields._replace(**valid_updates)


def _native_stage_fields_match_like(target: _NativeStageTensorFields | None, source: _NativeStageTensorFields) -> bool:
    if not isinstance(target, _NativeStageTensorFields):
        return False
    for field_name in source._fields:
        src = getattr(source, field_name)
        dst = getattr(target, field_name)
        if src is None:
            if dst is not None:
                return False
            continue
        if not torch.is_tensor(src):
            continue
        if not torch.is_tensor(dst):
            return False
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype or dst.device != src.device:
            return False
    return True


def _sat_subset_member_tensor(
    sat_count: int,
    max_select: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_device = torch.device(device)
    cache_key = (
        int(sat_count),
        int(max_select),
        str(tensor_device.type),
        int(tensor_device.index) if tensor_device.index is not None else None,
    )
    cached = _SAT_SUBSET_MEMBER_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        members_t, sizes_t = cached
        if members_t.device == tensor_device and sizes_t.device == tensor_device:
            return members_t, sizes_t

    subset_specs, subset_sizes = _subset_member_spec_cpu(int(sat_count), int(max_select))
    subset_members_t = torch.full(
        (len(subset_specs), int(max_select)),
        -1,
        dtype=torch.long,
        device=tensor_device,
    )
    for subset_idx, members in enumerate(subset_specs):
        if members:
            subset_members_t[subset_idx, : len(members)] = torch.as_tensor(
                members,
                dtype=torch.long,
                device=tensor_device,
            )
    subset_sizes_t = torch.as_tensor(subset_sizes, dtype=torch.float32, device=tensor_device)
    _SAT_SUBSET_MEMBER_TENSOR_CACHE[cache_key] = (subset_members_t, subset_sizes_t)
    return subset_members_t, subset_sizes_t


class _StructuredCoreEnvSequence(Sequence[Any]):
    def __init__(self, core: "StructuredBatchEnvCore") -> None:
        self._core = core

    def __len__(self) -> int:
        return self._core.num_envs

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._core._env_for_slot(slot) for slot in range(*index.indices(len(self)))]
        return self._core._env_for_slot(int(index))

    def __iter__(self):
        for slot in range(len(self)):
            yield self._core._env_for_slot(slot)


class _StructuredCoreDriverSequence(Sequence[StructuredControlDriver]):
    def __init__(self, core: "StructuredBatchEnvCore") -> None:
        self._core = core

    def __len__(self) -> int:
        return self._core.num_envs

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._core._driver_for_slot(slot) for slot in range(*index.indices(len(self)))]
        return self._core._driver_for_slot(int(index))

    def __iter__(self):
        for slot in range(len(self)):
            yield self._core._driver_for_slot(slot)


class _StructuredNativeSlotView:
    _RUNTIME_FIELD_MAP = {
        "uav_pos": ("uav_pos", np.float32),
        "uav_vel": ("uav_vel", np.float32),
        "uav_energy": ("uav_energy", np.float32),
        "uav_queue": ("uav_queue", np.float32),
        "gu_pos": ("gu_pos", np.float32),
        "gu_cluster_centers": ("gu_cluster_centers", np.float32),
        "gu_cluster_counts": ("gu_cluster_counts", np.float32),
        "gu_queue": ("gu_queue", np.float32),
        "sat_queue": ("sat_queue", np.float32),
        "sat_pos": ("sat_pos", np.float32),
        "sat_vel": ("sat_vel", np.float32),
        "prev_association": ("prev_association", np.int32),
        "last_association": ("last_association", np.int32),
        "arrival_ref_bits_per_step": ("arrival_ref_bits_per_step", np.float32),
        "effective_task_arrival_rate": ("effective_task_arrival_rate", np.float32),
        "_arrival_base_scale": ("arrival_base_scale", np.float32),
        "_hotspot_active_idx": ("hotspot_active_idx", np.int32),
        "_hotspot_member_mask": ("hotspot_member_mask", np.float32),
        "last_gu_service_gap": ("last_gu_service_gap", np.float32),
        "last_gu_deadline_age": ("last_gu_deadline_age", np.float32),
        "last_exec_accel": ("last_exec_accel", np.float32),
        "last_policy_accel": ("last_policy_accel", np.float32),
        "avoidance_eta_eff": ("avoidance_eta_eff", np.float32),
        "last_avoidance_eta_exec": ("last_avoidance_eta_exec", np.float32),
        "_doppler_residual_state_hz": ("doppler_residual", np.float32),
        "prev_queue_sum_gu": ("prev_queue_sum_gu", np.float32),
        "prev_queue_sum_uav": ("prev_queue_sum_uav", np.float32),
        "prev_queue_sum_sat": ("prev_queue_sum_sat", np.float32),
        "prev_gu_queue_vec": ("prev_gu_queue_vec", np.float32),
        "prev_uav_queue_vec": ("prev_uav_queue_vec", np.float32),
        "prev_sat_queue_vec": ("prev_sat_queue_vec", np.float32),
        "gu_drop": ("gu_drop", np.float32),
        "uav_drop": ("uav_drop", np.float32),
        "sat_drop": ("sat_drop", np.float32),
        "last_access_interference_by_uav": ("last_access_interference_by_uav", np.float32),
        "last_bw_fraction_by_uav_gu": ("last_bw_fraction_by_uav_gu", np.float32),
        "last_gu_to_uav_inflow_by_uav": ("last_gu_to_uav_inflow_by_uav", np.float32),
        "last_uav_to_sat_outflow_matrix": ("last_uav_to_sat_outflow_matrix", np.float32),
        "last_selected_mask_by_uav_sat": ("last_selected_mask_by_uav_sat", np.float32),
        "last_sat_processed": ("last_sat_processed", np.float32),
        "t": ("t", np.int32),
        "global_step": ("global_step", np.int32),
        "bw_weighted_workload_acc_ema_vec": ("gu_workload_ema", np.float32),
        "bw_weighted_workload_rel_ema_vec": ("uav_workload_ema", np.float32),
        "bw_weighted_workload_sat_ema_vec": ("sat_workload_ema", np.float32),
        "last_gu_arrival_rate_vec": ("last_gu_arrival_rate_vec", np.float32),
        "gu_deadline_steps": ("gu_deadline_steps", np.float32),
        "last_gu_arrival": ("last_gu_arrival", np.float32),
        "last_gu_outflow": ("last_gu_outflow", np.float32),
        "last_gu_urgency_risk": ("last_gu_urgency_risk", np.float32),
        "last_gu_downstream_pressure": ("last_gu_downstream_pressure", np.float32),
        "last_gu_service_gap_risk": ("last_gu_service_gap_risk", np.float32),
        "last_gu_deadline_slack": ("last_gu_deadline_slack", np.float32),
        "last_gu_deadline_risk": ("last_gu_deadline_risk", np.float32),
    }

    def __init__(self, core: "StructuredBatchEnvCore", slot: int) -> None:
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_slot", int(slot))
        cfg = core.cfg
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "agents", [f"uav_{i}" for i in range(int(cfg.num_uav))])
        object.__setattr__(self, "possible_agents", list(self.agents))
        own_extra_dim = int(bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False)))
        user_extra_dim = (
            int(bool(getattr(cfg, "obs_user_include_arrival_rate", False)))
            + int(bool(getattr(cfg, "obs_user_include_recent_arrival", False)))
            + int(bool(getattr(cfg, "obs_user_include_recent_service", False)))
            + int(bool(getattr(cfg, "obs_user_include_queue_headroom", False)))
            + int(bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)))
            + int(bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)))
            + int(bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)))
            + int(bool(getattr(cfg, "obs_user_include_urgency_risk", False)))
            + int(bool(getattr(cfg, "obs_user_include_downstream_pressure", False)))
            + int(bool(getattr(cfg, "obs_user_include_service_gap", False)))
            + int(bool(getattr(cfg, "obs_user_include_service_gap_risk", False)))
            + int(bool(getattr(cfg, "obs_user_include_deadline_slack", False)))
            + int(bool(getattr(cfg, "obs_user_include_deadline_risk", False)))
        )
        sat_extra_dim = int(bool(getattr(cfg, "obs_sat_include_sat_cost", False)))
        object.__setattr__(self, "own_extra_dim", own_extra_dim)
        object.__setattr__(self, "own_dim", 10 + own_extra_dim)
        object.__setattr__(self, "user_extra_dim", user_extra_dim)
        object.__setattr__(self, "user_dim", 5 + user_extra_dim)
        object.__setattr__(self, "gu_node_dim", 3 + user_extra_dim)
        object.__setattr__(self, "sat_extra_dim", sat_extra_dim)
        object.__setattr__(self, "sat_dim", 12 + sat_extra_dim)
        object.__setattr__(self, "nbr_dim", 4)
        object.__setattr__(self, "danger_nbr_dim", 5)
        object.__setattr__(self, "_uav_height_sq", float(cfg.uav_height) ** 2)
        object.__setattr__(self, "_uav_orbit_radius", float(cfg.r_earth + cfg.uav_height))
        object.__setattr__(self, "_sat_orbit_radius", float(cfg.r_earth + cfg.sat_height))
        object.__setattr__(self, "_uav_orbit_radius_sq", float(cfg.r_earth + cfg.uav_height) ** 2)
        object.__setattr__(self, "_sat_orbit_radius_sq", float(cfg.r_earth + cfg.sat_height) ** 2)
        object.__setattr__(
            self,
            "_backhaul_gain_const",
            float(
                (cfg.speed_of_light / (4.0 * math.pi * _backhaul_carrier_freq_from_cfg(cfg))) ** 2
                * cfg.uav_tx_gain
                * cfg.sat_rx_gain
            ),
        )

    @property
    def _meta(self) -> dict[str, Any]:
        return self._core._slot_state_payloads[self._slot]

    @property
    def rng(self) -> np.random.Generator:
        return self._core._slot_rngs[self._slot]

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._core._slot_rngs[self._slot] = value

    def _default_meta_value(self, name: str):
        cfg = self.cfg
        if name in {
            "_cached_elevation_t",
            "_cached_elevation_matrix",
            "_cached_backhaul_loss_t",
            "_cached_backhaul_loss_matrix",
            "_cached_uav_ecef",
            "_cached_uav_vel_ecef",
            "_cached_uav_neighbor_t",
            "_cached_uav_neighbor_order",
            "_cached_global_state",
            "_cached_obs_runtime_context",
        }:
            return None
        if name == "_cached_orbit_t":
            return int(self.t)
        if name == "_cached_orbit_pos":
            return _compat_numpy_array(
                self._core._runtime_tensor_state.sat_pos[self._slot],
                dtype=np.float32,
            ).copy()
        if name == "_cached_orbit_vel":
            return _compat_numpy_array(
                self._core._runtime_tensor_state.sat_vel[self._slot],
                dtype=np.float32,
            ).copy()
        if name == "_cached_assoc":
            return np.full((int(cfg.num_gu),), -1, dtype=np.int32)
        if name == "_cached_candidates":
            return [[] for _ in range(int(cfg.num_uav))]
        if name == "_cached_bw_valid_mask":
            return np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
        if name == "_cached_eta":
            return np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32)
        if name == "_cached_eta_uav_pos":
            return np.asarray(self.uav_pos, dtype=np.float32).copy()
        if name == "_cached_eta_gu_pos":
            return np.asarray(self.gu_pos, dtype=np.float32).copy()
        if name == "_cached_access_gain_matrix":
            return None
        if name == "_cached_sat_obs":
            return np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max), int(self.sat_dim)), dtype=np.float32)
        if name == "_cached_sat_mask":
            return np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32)
        if name == "_cached_sat_valid_mask":
            return np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32)
        raise AttributeError(name)

    def __getattr__(self, name: str):
        runtime_entry = self._RUNTIME_FIELD_MAP.get(name)
        if runtime_entry is not None:
            runtime_name, dtype = runtime_entry
            tensor = getattr(self._core._runtime_tensor_state, runtime_name)[self._slot]
            if tensor.ndim == 0:
                scalar = tensor.detach().item()
                if np.issubdtype(np.dtype(dtype), np.integer):
                    return int(scalar)
                return float(scalar)
            return _compat_numpy_array(tensor, dtype=dtype).copy()
        if name == "last_sat_selection":
            matrix = _compat_numpy_array(
                self._core._runtime_tensor_state.last_sat_selection_matrix[self._slot],
                dtype=np.int64,
            )
            return _sat_selection_lists_from_matrix(matrix)
        if name == "last_sat_connection_counts":
            return _compat_numpy_array(
                self._core._runtime_tensor_state.last_sat_connection_counts[self._slot],
                dtype=np.float32,
            )
        if name == "_orbit_pos_table":
            return self._core._orbit_pos_table
        if name == "_orbit_vel_table":
            return self._core._orbit_vel_table
        if name == "orbit":
            return self._core._orbit_model
        if name == "bw_weighted_workload_acc_ema":
            return float(
                self._core._runtime_tensor_state.gu_workload_ema[self._slot].sum(dtype=torch.float32).detach().item()
            )
        if name == "bw_weighted_workload_rel_ema":
            return float(
                self._core._runtime_tensor_state.uav_workload_ema[self._slot].sum(dtype=torch.float32).detach().item()
            )
        if name == "bw_weighted_workload_sat_ema":
            return float(
                self._core._runtime_tensor_state.sat_workload_ema[self._slot].sum(dtype=torch.float32).detach().item()
            )
        meta = self._meta
        if name in meta:
            return meta[name]
        try:
            default_value = self._default_meta_value(name)
        except AttributeError:
            default_value = None
        else:
            meta[name] = self._copy_meta_value(default_value)
            return meta[name]
        env_descriptor = SaginParallelEnv.__dict__.get(name)
        if env_descriptor is not None:
            bound = env_descriptor.__get__(self, type(self))
            if callable(bound):
                return bound
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_core", "_slot", "cfg", "agents", "possible_agents", "own_extra_dim", "own_dim", "user_extra_dim", "user_dim", "gu_node_dim", "sat_extra_dim", "sat_dim", "nbr_dim", "danger_nbr_dim", "_uav_height_sq", "_uav_orbit_radius", "_sat_orbit_radius", "_uav_orbit_radius_sq", "_sat_orbit_radius_sq", "_backhaul_gain_const"}:
            object.__setattr__(self, name, value)
            return
        runtime_entry = self._RUNTIME_FIELD_MAP.get(name)
        if runtime_entry is not None:
            runtime_name, dtype = runtime_entry
            target = getattr(self._core._runtime_tensor_state, runtime_name)
            if name == "_hotspot_member_mask":
                row = target[self._slot]
                row.zero_()
                value_t = _as_kernel_tensor(value, dtype=target.dtype, device=target.device)
                if value_t.ndim == 1:
                    value_t = value_t.view(1, -1)
                copy_rows = min(int(row.shape[0]), int(value_t.shape[0])) if value_t.ndim >= 2 else 0
                copy_cols = min(int(row.shape[1]), int(value_t.shape[1])) if value_t.ndim >= 2 else 0
                if copy_rows > 0 and copy_cols > 0:
                    row[:copy_rows, :copy_cols].copy_(value_t[:copy_rows, :copy_cols])
                del dtype
                return
            target[self._slot] = _as_kernel_tensor(
                value,
                dtype=target.dtype,
                device=target.device,
            )
            del dtype
            return
        if name == "last_sat_selection":
            matrix = _sat_selection_matrix_from_values(self.cfg, value)
            target = self._core._runtime_tensor_state.last_sat_selection_matrix
            target[self._slot] = _as_kernel_tensor(matrix, dtype=target.dtype, device=target.device)
            return
        if name == "last_sat_connection_counts":
            target = self._core._runtime_tensor_state.last_sat_connection_counts
            target[self._slot] = _as_kernel_tensor(value, dtype=target.dtype, device=target.device)
            return
        self._meta[name] = self._store_meta_value(value)

    @staticmethod
    def _copy_meta_value(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {key: _StructuredNativeSlotView._copy_meta_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_StructuredNativeSlotView._copy_meta_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_StructuredNativeSlotView._copy_meta_value(item) for item in value)
        return copy.deepcopy(value)

    @staticmethod
    def _store_meta_value(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {key: _StructuredNativeSlotView._store_meta_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_StructuredNativeSlotView._store_meta_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_StructuredNativeSlotView._store_meta_value(item) for item in value)
        return value

    @staticmethod
    def _minmax_normalize(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size <= 0:
            return arr
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
        if v_max - v_min <= NORMALIZATION_DENOM_EPS:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - v_min) / (v_max - v_min)).astype(np.float32, copy=False)

    def _queue_arrival_scale(self, arrival_sum: float) -> float:
        cfg = self.cfg
        queue_norm_k = normalize_scale(float(getattr(cfg, "queue_norm_K", 1.0) or 1.0))
        arrival_floor = float(getattr(cfg, "queue_norm_arrival_floor", 0.0) or 0.0)
        if arrival_floor <= 0.0:
            arrival_floor = float(self.effective_task_arrival_rate) * float(cfg.num_gu) * float(cfg.tau0)
        arrival_ref = reward_ratio_denominator_scalar(
            max(float(arrival_sum), arrival_floor),
            name="queue arrival normalization reference",
        )
        return queue_norm_k * arrival_ref

    def _compute_centroid_stats(self) -> tuple[float, float]:
        cfg = self.cfg
        centroid_reward = 0.0
        centroid_dist_mean = 0.0
        if int(cfg.num_gu) > 0:
            q_weights = np.asarray(self.gu_queue, dtype=np.float32) / normalize_scale(float(cfg.queue_max_gu))
            w_sum = float(np.sum(q_weights))
            if w_sum <= NORMALIZATION_DENOM_EPS:
                weights = np.full((int(cfg.num_gu),), 1.0 / max(int(cfg.num_gu), 1), dtype=np.float32)
            else:
                weights = (q_weights / w_sum).astype(np.float32, copy=False)
            centroid = np.sum(np.asarray(self.gu_pos, dtype=np.float32) * weights[:, None], axis=0)
            dists = np.linalg.norm(np.asarray(self.uav_pos, dtype=np.float32) - centroid[None, :], axis=1)
            centroid_dist_mean = float(np.mean(dists)) if dists.size else 0.0
            scale = max(float(getattr(cfg, "centroid_dist_scale", 1.0) or 1.0), 1.0e-6)
            centroid_reward = float(np.mean(np.exp(-dists / scale))) if dists.size else 0.0
        return centroid_reward, centroid_dist_mean

    def _centroid_anneal_state(self) -> tuple[float, float, float]:
        cfg = self.cfg
        eta_start = float(getattr(cfg, "eta_centroid", 0.0) or 0.0)
        eta_final_cfg = getattr(cfg, "eta_centroid_final", None)
        eta_current = eta_start
        decay_steps = int(getattr(cfg, "eta_centroid_decay_steps", 0) or 0)
        if eta_final_cfg is not None and decay_steps > 0:
            progress = min(1.0, float(self.global_step) / float(max(decay_steps, 1)))
            eta_final = float(eta_final_cfg)
            eta_current = eta_start + (eta_final - eta_start) * progress
        decayed = max(eta_start - eta_current, 0.0)
        transfer_ratio = float(np.clip(decayed / eta_start, 0.0, 1.0)) if eta_start > NORMALIZATION_DENOM_EPS else 0.0
        return eta_start, eta_current, transfer_ratio

    def _arrival_ref(self) -> float:
        return reward_ratio_denominator_scalar(
            float(self.arrival_ref_bits_per_step),
            name="arrival_ref_bits_per_step",
        )

    def _bw_weighted_workload_eps(self) -> float:
        floor = getattr(self.cfg, "service_floor_bits_per_step", None)
        if floor is None:
            floor = getattr(self.cfg, "bw_weighted_workload_eps", 1.0)
        return max(float(floor or 0.0), float(NORMALIZATION_DENOM_EPS))

    def _bw_weighted_workload_sat_active_ref_count(self) -> float:
        return _bw_weighted_workload_sat_active_ref_count(self.cfg)

    def _bw_weighted_workload_device_ema_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.bw_weighted_workload_acc_ema_vec, dtype=np.float32),
            np.asarray(self.bw_weighted_workload_rel_ema_vec, dtype=np.float32),
            np.asarray(self.bw_weighted_workload_sat_ema_vec, dtype=np.float32),
        )

    def _bw_weighted_workload_feature_refs(self) -> tuple[float, float, float, float]:
        cfg = self.cfg
        eps = self._bw_weighted_workload_eps()
        arrival_ref = float(self._arrival_ref())
        gu_default = arrival_ref / max(float(cfg.num_gu), 1.0)
        uav_default = arrival_ref / max(float(cfg.num_uav), 1.0)
        sat_default = arrival_ref / max(float(self._bw_weighted_workload_sat_active_ref_count()), 1.0)
        sat_cost_ref = 1.0 / max(float(sat_default), eps)
        uav_cost_ref = 1.0 / max(float(uav_default), eps) + sat_cost_ref
        gu_local_cost_ref = 1.0 / max(float(gu_default), eps)
        gu_total_cost_ref = gu_local_cost_ref + uav_cost_ref
        return float(gu_local_cost_ref), float(uav_cost_ref), float(sat_cost_ref), float(gu_total_cost_ref)

    def _effective_b_backhaul_per_sat(self) -> float:
        return _effective_b_backhaul_per_sat_from_cfg(self.cfg)

    def _effective_b_sat_total(self) -> float:
        """Legacy alias for the per-SAT backhaul bandwidth pool."""
        return self._effective_b_backhaul_per_sat()

    def _effective_sat_cpu_freq(self) -> float:
        return _effective_sat_cpu_freq_from_cfg(self.cfg)

    def _doppler_precomp_enabled(self) -> bool:
        mode = str(getattr(self.cfg, "doppler_precomp_mode", "none") or "none").strip().lower()
        return mode in {"residual_hz", "residual_ppm"}

    def _sat_selection_lists(self, selections: list[list[int]] | np.ndarray | Sequence[Sequence[int]]) -> list[list[int]]:
        return _sat_selection_lists_from_matrix(_sat_selection_matrix_from_values(self.cfg, selections))

    def _sat_selection_matrix(self, selections: list[list[int]] | np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        return _sat_selection_matrix_from_values(self.cfg, selections)

    def _uav_ecef(self, u: int) -> np.ndarray:
        uav_pos = np.asarray(self.uav_pos[int(u)], dtype=np.float32)
        lat0 = math.radians(float(self.cfg.ref_lat_deg))
        lon0 = math.radians(float(self.cfg.ref_lon_deg))
        lat = lat0 + float(uav_pos[1]) / normalize_scale(float(self.cfg.r_earth))
        lon = lon0 + float(uav_pos[0]) / normalize_scale(float(self.cfg.r_earth) * math.cos(lat0))
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)
        return np.asarray(
            [
                self._uav_orbit_radius * cos_lat * cos_lon,
                self._uav_orbit_radius * cos_lat * sin_lon,
                self._uav_orbit_radius * sin_lat,
            ],
            dtype=np.float32,
        )

    def _uav_vel_ecef(self, u: int) -> np.ndarray:
        cfg = self.cfg
        uav_pos = np.asarray(self.uav_pos[int(u)], dtype=np.float32)
        uav_vel = np.asarray(self.uav_vel[int(u)], dtype=np.float32)
        lat0 = math.radians(float(cfg.ref_lat_deg))
        lon0 = math.radians(float(cfg.ref_lon_deg))
        lat = lat0 + float(uav_pos[1]) / normalize_scale(float(cfg.r_earth))
        lon = lon0 + float(uav_pos[0]) / normalize_scale(float(cfg.r_earth) * math.cos(lat0))
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        transform = np.asarray(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ],
            dtype=np.float32,
        )
        enu_velocity = np.asarray([float(uav_vel[0]), float(uav_vel[1]), 0.0], dtype=np.float32)
        return (transform @ enu_velocity).astype(np.float32, copy=False)

    def _get_elevation_matrix(self, sat_pos: np.ndarray | None = None) -> np.ndarray:
        sat_pos_arr = np.asarray(self.sat_pos if sat_pos is None else sat_pos, dtype=np.float32)
        uav_ecef = np.stack([self._uav_ecef(u) for u in range(int(self.cfg.num_uav))], axis=0)
        rel = sat_pos_arr[None, :, :] - uav_ecef[:, None, :]
        dist = geometry_denominator(np.linalg.norm(rel, axis=-1))
        arg = (float(self._sat_orbit_radius_sq) - float(self._uav_orbit_radius_sq) - dist * dist) / (
            geometry_denominator(2.0 * float(self._uav_orbit_radius) * dist)
        )
        np.clip(arg, -1.0, 1.0, out=arg)
        return np.arcsin(arg).astype(np.float32, copy=False)

    def _visible_sats_sorted(self, sat_pos: np.ndarray, *, record_stats: bool = True) -> list[list[int]]:
        visible, _, _ = _visible_sats_sorted_batch(
            [self],
            np.asarray(sat_pos, dtype=np.float32),
            typed_domains=self._core._native_main_kernel_typed_domains(),
        )
        return visible[0] if visible else [[] for _ in range(int(self.cfg.num_uav))]

    @staticmethod
    def _topk_descending_stable(values: np.ndarray, k: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size <= 0 or int(k) <= 0:
            return np.zeros((0,), dtype=np.int64)
        order = np.argsort(-arr.astype(np.float32), kind="stable")
        return order[: int(k)].astype(np.int64, copy=False)

    def _current_task_arrival_rates(self, arrival_rate: float) -> np.ndarray:
        return _current_task_arrival_rates_batch([self], [float(arrival_rate)])[0]

    def _advance_traffic_model_state(self) -> None:
        if str(getattr(self.cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower() != "sticky_subset_hotspot":
            return
        hotspot_mask = np.asarray(getattr(self, "_hotspot_member_mask", np.zeros((0, int(self.cfg.num_gu)), dtype=bool)))
        num_subsets = int(hotspot_mask.shape[0])
        if num_subsets <= 0:
            self._hotspot_active_idx = -1
            self.last_hotspot_index = -1
            return
        on_mean = max(float(getattr(self.cfg, "hotspot_on_mean_steps", 15.0) or 15.0), 1.0)
        off_mean = max(float(getattr(self.cfg, "hotspot_off_mean_steps", 8.0) or 8.0), 1.0)
        hotspot_idx = int(getattr(self, "_hotspot_active_idx", -1))
        if hotspot_idx >= 0:
            if float(self.rng.random()) < (1.0 / on_mean):
                hotspot_idx = -1
        else:
            if float(self.rng.random()) < (1.0 / off_mean):
                hotspot_idx = int(self.rng.integers(num_subsets))
        self._hotspot_active_idx = hotspot_idx
        self.last_hotspot_index = hotspot_idx

    def export_runtime_state(self) -> dict[str, Any]:
        return self._core._export_runtime_state_from_core_slot(self._slot)

    def load_runtime_state(
        self,
        state: dict[str, Any],
        *,
        refresh_observation_cache: bool = True,
        refresh_global_state_cache: bool = True,
    ) -> None:
        del refresh_observation_cache
        del refresh_global_state_cache
        self._core.load_runtime_state_batch([state], indices=[self._slot])

    def close(self) -> None:
        return None

def _as_kernel_tensor(
    value: torch.Tensor | np.ndarray | Sequence[float] | Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    kernel_device = torch.device("cpu") if device is None else torch.device(device)
    if torch.is_tensor(value):
        if value.device == kernel_device and value.dtype == dtype:
            return value
        return value.to(device=kernel_device, dtype=dtype)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=dtype, device=kernel_device)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            value_len = len(value)
        except TypeError:
            value_len = 0
        if value_len > 0:
            first = value[0]
            if torch.is_tensor(first):
                return torch.stack(
                    [
                        item.to(device=kernel_device, dtype=dtype)
                        if torch.is_tensor(item)
                        else torch.as_tensor(item, dtype=dtype, device=kernel_device)
                        for item in value
                    ],
                    dim=0,
                )
            if isinstance(first, np.ndarray):
                value = np.asarray(value)
    return torch.as_tensor(value, dtype=dtype, device=kernel_device)


def _resolve_structured_tensor_device(
    cfg,
    tensor_device: torch.device | str | None,
) -> torch.device:
    if tensor_device is not None:
        return torch.device(tensor_device)
    backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    if backend == "cpu":
        return torch.device("cpu")
    if backend in {"cuda", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _same_structured_tensor_device(
    left: torch.device | str | None,
    right: torch.device | str | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_device = torch.device(left)
    right_device = torch.device(right)
    if left_device == right_device:
        return True
    if left_device.type == right_device.type == "cuda":
        return (
            left_device.index is None
            or right_device.index is None
            or int(left_device.index) == int(right_device.index)
        )
    return False


def _compat_numpy_array(value: torch.Tensor | np.ndarray, *, dtype=None) -> np.ndarray:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _legacy_float32_sum(value: torch.Tensor | np.ndarray) -> float:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value, dtype=np.float32))
    return float(tensor.to(dtype=torch.float32).sum(dim=tuple(range(tensor.ndim)), dtype=torch.float32).item())


def _visible_lists_from_spec(spec: dict[str, Any], cfg) -> list[list[int]]:
    visible = spec.get("visible")
    if visible is not None:
        return [list(values) for values in visible]
    visible_ids = spec.get("visible_ids_tensor")
    visible_mask = spec.get("visible_mask_tensor")
    if visible_ids is not None and visible_mask is not None:
        ids_arr = _compat_numpy_array(visible_ids, dtype=np.int64)
        mask_arr = _compat_numpy_array(visible_mask, dtype=bool)
        out: list[list[int]] = []
        for uav_index in range(int(cfg.num_uav)):
            ids_u = ids_arr[uav_index]
            mask_u = mask_arr[uav_index]
            out.append([int(value) for value in ids_u[mask_u].tolist()])
        return out
    return [[] for _ in range(int(cfg.num_uav))]


def _indexed_tensor_compat_item(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    index: int,
):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value[int(index)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[int(index)]
    raise TypeError(f"Unsupported indexed value type: {type(value)!r}")


def _tensor_index_numpy(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    index: int,
    *,
    dtype=None,
) -> np.ndarray:
    item = _indexed_tensor_compat_item(value, index)
    arr = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _tensor_compat_value(
    value: torch.Tensor | np.ndarray,
    *,
    dtype,
):
    np_dtype = np.dtype(dtype)
    if torch.is_tensor(value):
        arr = _compat_numpy_array(value, dtype=np_dtype)
        if arr.ndim == 0:
            if np.issubdtype(arr.dtype, np.bool_):
                return bool(arr.item())
            if np.issubdtype(arr.dtype, np.integer):
                return int(arr.item())
            return float(arr.item())
        return arr
    arr = np.asarray(value, dtype=np_dtype)
    if arr.ndim == 0:
        if np.issubdtype(arr.dtype, np.bool_):
            return bool(arr.item())
        if np.issubdtype(arr.dtype, np.integer):
            return int(arr.item())
        return float(arr.item())
    return arr


def _tensor_index_compat(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    index: int,
    *,
    dtype,
):
    item = _indexed_tensor_compat_item(value, index)
    if torch.is_tensor(item):
        return _tensor_compat_value(item, dtype=dtype)
    return _tensor_compat_value(np.asarray(item), dtype=dtype)


def _tensor_index_scalar(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    index: int,
    *,
    dtype=None,
):
    item = _indexed_tensor_compat_item(value, index)
    if torch.is_tensor(item):
        item = item.detach()
        if dtype is None:
            return item.item()
        np_dtype = np.dtype(dtype)
        scalar = item.item()
        if np.issubdtype(np_dtype, np.bool_):
            return bool(scalar)
        if np.issubdtype(np_dtype, np.integer):
            return int(np_dtype.type(scalar))
        return float(np_dtype.type(scalar))
    return _tensor_index_numpy([item], 0, dtype=dtype).item()


def _resolve_kernel_callable(
    *,
    name: str,
    eager_fn: Callable[..., Any],
    compile_cfg,
    device: torch.device | str | None,
) -> Callable[..., Any]:
    if compile_cfg is None:
        return eager_fn
    runtime = get_structured_kernel_runtime(compile_cfg, device=device)
    return runtime.compile_kernel(name, eager_fn)


def _kernel_name_set_from_config(raw) -> set[str] | None:
    if raw is None:
        return set()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return set()
        if text.lower() in {"*", "all"}:
            return None
        return {item.strip() for item in text.split(",") if item.strip()}
    try:
        return {str(item).strip() for item in raw if str(item).strip()}
    except TypeError:
        return set()


def _los_probability_torch(phi_deg_t: torch.Tensor, *, a: float, b: float) -> torch.Tensor:
    a_value = float(a)
    b_value = float(b)
    return 1.0 / (1.0 + a_value * torch.exp(-b_value * (phi_deg_t - a_value)))


def _torch_positive(value_t: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(value_t, min=float(eps))


def _torch_divide_or_default(
    numerator_t: torch.Tensor,
    denominator_t: torch.Tensor | float,
    *,
    eps: float = NORMALIZATION_DENOM_EPS,
    default: float = 0.0,
) -> torch.Tensor:
    denominator_work_t = torch.as_tensor(denominator_t, dtype=numerator_t.dtype, device=numerator_t.device)
    numerator_work_t, denominator_work_t = torch.broadcast_tensors(numerator_t, denominator_work_t)
    default_t = numerator_work_t * 0.0 + float(default)
    return torch.where(
        torch.abs(denominator_work_t) > float(eps),
        numerator_work_t / denominator_work_t,
        default_t,
    )


def _torch_ratio_or_zero(numerator_t: torch.Tensor, denominator_t: torch.Tensor | float) -> torch.Tensor:
    return _torch_divide_or_default(
        numerator_t,
        denominator_t,
        eps=RUNTIME_RATIO_ZERO_TOL,
        default=0.0,
    )


def _torch_log1p_nonnegative(value_t: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(value_t.to(dtype=torch.float32), min=0.0))


def _torch_require_positive_reward_ref(value_t: torch.Tensor, *, name: str) -> torch.Tensor:
    value_work_t = value_t.reshape(-1)
    torch._assert(torch.all(torch.isfinite(value_work_t) & (value_work_t > 0.0)), f"{name} must be positive and finite.")
    return value_t


def _validate_full_g_bw_action_tensor(
    *,
    bw_action_matrix_t: torch.Tensor,
    assoc_t: torch.Tensor,
    num_uav: int,
    context: str,
) -> torch.Tensor:
    if bw_action_matrix_t.ndim != 3:
        raise RuntimeError(f"{context} requires full-G BW action tensor with shape [B, U, G].")
    batch_size = int(bw_action_matrix_t.shape[0])
    uav_count = int(num_uav)
    gu_count = int(bw_action_matrix_t.shape[2])
    if uav_count <= 0 or gu_count <= 0:
        raise RuntimeError(f"{context} requires positive U and G dimensions.")
    if int(bw_action_matrix_t.shape[1]) != uav_count:
        raise RuntimeError(
            f"{context} BW action U dimension must be {uav_count}, got {int(bw_action_matrix_t.shape[1])}."
        )
    if tuple(assoc_t.shape) != (batch_size, gu_count):
        raise RuntimeError(
            f"{context} association tensor must have shape {(batch_size, gu_count)}, "
            f"got {tuple(assoc_t.shape)}."
        )
    work_t = bw_action_matrix_t.to(dtype=torch.float32)
    value_check_eps = 1.0e-7
    fp32_eps = torch.finfo(torch.float32).eps
    torch._assert(torch.all(torch.isfinite(work_t)), f"{context} BW action contains non-finite values.")
    torch._assert(
        torch.amin(work_t) >= -value_check_eps,
        f"{context} BW action contains negative values beyond tolerance.",
    )
    uav_ids_t = torch.arange(uav_count, dtype=torch.long, device=work_t.device).view(1, uav_count, 1)
    assoc_long_t = assoc_t.to(device=work_t.device, dtype=torch.long).reshape(batch_size, gu_count)
    valid_t = (assoc_long_t[:, None, :] == uav_ids_t) & (assoc_long_t[:, None, :] >= 0) & (
        assoc_long_t[:, None, :] < uav_count
    )
    valid_count_t = valid_t.to(dtype=torch.float32).sum(dim=-1)
    valid_sum_t = torch.where(valid_t, work_t, work_t * 0.0).sum(dim=-1)
    invalid_mass_t = torch.clamp(torch.where(valid_t, work_t * 0.0, work_t), min=0.0).sum(dim=-1)
    sum_eps_t = torch.clamp(8.0 * fp32_eps * torch.clamp(valid_count_t, min=1.0), min=1.0e-5)
    sum_error_t = torch.where(valid_count_t > 0.0, torch.abs(valid_sum_t - 1.0), torch.abs(valid_sum_t))
    torch._assert(
        torch.all(invalid_mass_t <= sum_eps_t),
        f"{context} BW action has positive mass on invalid GU entries.",
    )
    torch._assert(
        torch.all(sum_error_t <= sum_eps_t),
        f"{context} BW action valid-GU simplex sum is outside tolerance.",
    )
    return valid_t


def _pathloss_db_torch(
    *,
    d_t: torch.Tensor,
    phi_rad_t: torch.Tensor,
    channel_params: _NativeChannelStaticParams,
) -> torch.Tensor:
    channel_p = channel_params
    work_dtype = torch.float32 if bool(channel_p.fast_float32) else torch.float64
    work_d_t = d_t.to(dtype=work_dtype)
    work_phi_t = phi_rad_t.to(dtype=work_dtype)
    safe_d_t = _torch_positive(work_d_t, GEOMETRY_DENOM_EPS)
    pl_base = float(channel_p.pathloss_const_db) + 20.0 * math.log10(float(channel_p.carrier_freq) / 1.0e9)
    pl_los_t = pl_base + float(channel_p.xi_los)
    pl_los_t = pl_los_t + 20.0 * torch.log10(safe_d_t)
    pl_nlos_t = pl_base + float(channel_p.xi_nlos)
    pl_nlos_t = pl_nlos_t + 20.0 * torch.log10(safe_d_t)
    if str(channel_p.pathloss_mode) == "free_space":
        return pl_los_t
    phi_deg_t = work_phi_t * (180.0 / math.pi)
    p_los_t = _los_probability_torch(phi_deg_t, a=float(channel_p.los_a), b=float(channel_p.los_b)).to(dtype=work_dtype)
    return p_los_t * pl_los_t + (1.0 - p_los_t) * pl_nlos_t


def _atmospheric_loss_db_torch(theta_rad_t: torch.Tensor, *, base_loss_db: float) -> torch.Tensor:
    sin_el_t = torch.clamp(torch.sin(theta_rad_t), min=1.0e-3)
    return float(base_loss_db) / sin_el_t


def _p838_curve_torch(
    log_f_t: torch.Tensor,
    *,
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> torch.Tensor:
    out_t = log_f_t.to(dtype=torch.float32) * 0.0
    for ai, bi, ci in zip(a, b, c, strict=True):
        out_t = out_t + float(ai) * torch.exp(-((log_f_t - float(bi)) / abs(float(ci))) ** 2)
    return out_t


def _rain_frequency_terms_from_logf_torch(
    log_f_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log10_kh_t = _p838_curve_torch(
        log_f_t,
        a=(-5.33980, -0.35351, -0.23789, -0.94158),
        b=(-0.10008, 1.26970, 0.86036, 0.64552),
        c=(1.13098, 0.45400, 0.15354, 0.16817),
    ) - 0.18961 * log_f_t + 0.71147
    log10_kv_t = _p838_curve_torch(
        log_f_t,
        a=(-3.80595, -3.44965, -0.39902, 0.50167),
        b=(0.56934, -0.22911, 0.73042, 1.07319),
        c=(0.81061, 0.51059, 0.11899, 0.27195),
    ) - 0.16398 * log_f_t + 0.63297
    alpha_h_t = _p838_curve_torch(
        log_f_t,
        a=(-0.14318, 0.29591, 0.32177, -5.37610, 16.1721),
        b=(1.82442, 0.77564, 0.63773, -0.96230, -3.29980),
        c=(-0.55187, 0.19822, 0.13164, 1.47828, 3.43990),
    ) + 0.67849 * log_f_t - 1.95537
    alpha_v_t = _p838_curve_torch(
        log_f_t,
        a=(-0.07771, 0.56727, -0.20238, -48.2991, 48.5833),
        b=(2.33840, 0.95545, 1.14520, 0.791669, 0.791459),
        c=(-0.76284, 0.54039, 0.26809, 0.116226, 0.116479),
    ) - 0.053739 * log_f_t + 0.83433
    return 10.0 ** log10_kh_t, 10.0 ** log10_kv_t, alpha_h_t, alpha_v_t


def _rain_specific_attenuation_db_per_km_torch(
    *,
    theta_rad_t: torch.Tensor,
    carrier_freq_hz: float,
    rain_rate_mmph: float,
    polarization_tilt_deg: float = 45.0,
) -> torch.Tensor:
    work_dtype = theta_rad_t.dtype if theta_rad_t.is_floating_point() else torch.float32
    theta_t = theta_rad_t.to(dtype=work_dtype)
    rain_rate = max(float(rain_rate_mmph), 0.0)
    if rain_rate <= 0.0:
        return theta_t * 0.0
    freq_ghz = max(float(carrier_freq_hz) / 1.0e9, FREQUENCY_GHZ_EPS)
    log_f_t = theta_t * 0.0 + math.log10(freq_ghz)
    k_h_t, k_v_t, alpha_h_t, alpha_v_t = _rain_frequency_terms_from_logf_torch(log_f_t)
    tau = math.radians(float(polarization_tilt_deg))
    cos_term_t = torch.cos(theta_t).square() * math.cos(2.0 * tau)
    k_t = 0.5 * (k_h_t + k_v_t + (k_h_t - k_v_t) * cos_term_t)
    alpha_t = 0.5 * (
        k_h_t * alpha_h_t
        + k_v_t * alpha_v_t
        + (k_h_t * alpha_h_t - k_v_t * alpha_v_t) * cos_term_t
    ) / torch.clamp(k_t, min=POSITIVE_COEFF_EPS)
    return k_t * (float(rain_rate) ** alpha_t)


def _rain_attenuation_db_torch(
    *,
    theta_rad_t: torch.Tensor,
    carrier_freq_hz: float,
    rain_rate_001_mmph: float,
    rain_height_km: float,
    station_height_km: float,
    latitude_deg: float,
    exceedance_pct: float = 0.1,
    polarization_tilt_deg: float = 45.0,
) -> torch.Tensor:
    work_dtype = theta_rad_t.dtype if theta_rad_t.is_floating_point() else torch.float32
    rain_rate = max(float(rain_rate_001_mmph), 0.0)
    if rain_rate <= 0.0 or float(rain_height_km) <= float(station_height_km):
        return theta_rad_t.to(dtype=work_dtype) * 0.0

    theta_t = torch.clamp(theta_rad_t.to(dtype=work_dtype), min=ANGLE_RAD_EPS)
    theta_deg_t = torch.rad2deg(theta_t)
    sin_el_t = torch.clamp(torch.sin(theta_t), min=TRIG_DENOM_EPS)
    cos_el_t = torch.cos(theta_t)
    freq_ghz = max(float(carrier_freq_hz) / 1.0e9, FREQUENCY_GHZ_EPS)

    gamma_r_t = _rain_specific_attenuation_db_per_km_torch(
        theta_rad_t=theta_t,
        carrier_freq_hz=float(carrier_freq_hz),
        rain_rate_mmph=rain_rate,
        polarization_tilt_deg=polarization_tilt_deg,
    )

    delta_h = max(float(rain_height_km) - float(station_height_km), 0.0)
    l_s_t = float(delta_h) / sin_el_t
    l_g_t = l_s_t * cos_el_t
    r_001_t = 1.0 / (
        1.0
        + 0.78 * torch.sqrt(torch.clamp(l_g_t * gamma_r_t / float(freq_ghz), min=0.0))
        - 0.38 * (1.0 - torch.exp(-2.0 * torch.clamp(l_g_t, min=0.0)))
    )
    r_001_t = torch.clamp(r_001_t, min=MIN_RAIN_REDUCTION_FACTOR)

    zeta_t = torch.atan2(
        theta_t * 0.0 + float(delta_h),
        torch.clamp(l_g_t * r_001_t, min=GEOMETRY_DENOM_EPS),
    )
    l_r_t = torch.where(
        zeta_t > theta_t,
        torch.clamp(l_g_t, min=0.0) * r_001_t / torch.clamp(cos_el_t, min=TRIG_DENOM_EPS),
        l_s_t,
    )

    chi = max(36.0 - abs(float(latitude_deg)), 0.0)
    v_001_t = 1.0 / (
        1.0
        + torch.sqrt(sin_el_t)
        * (
            31.0
            * (1.0 - torch.exp(-theta_deg_t / (1.0 + float(chi))))
            * torch.sqrt(torch.clamp(l_r_t * gamma_r_t, min=0.0))
            / (float(freq_ghz) ** 2)
            - 0.45
        )
    )
    v_001_t = torch.clamp(v_001_t, min=MIN_RAIN_REDUCTION_FACTOR)
    l_e_t = l_r_t * v_001_t
    a_001_t = gamma_r_t * l_e_t

    p = min(max(float(exceedance_pct), 0.001), 5.0)
    if abs(p - 0.01) < PROBABILITY_EQUALITY_TOL:
        return a_001_t

    lat_abs = abs(float(latitude_deg))
    if p >= 1.0 or lat_abs >= 36.0:
        beta_t = theta_t * 0.0
    else:
        beta_t = torch.where(
            theta_deg_t >= 25.0,
            theta_t * 0.0 + (-0.005 * (lat_abs - 36.0)),
            theta_t * 0.0 + (-0.005 * (lat_abs - 36.0) + 1.8) - 4.25 * sin_el_t,
        )
    exponent_t = -(
        0.655
        + 0.033 * math.log(p)
        - 0.045 * torch.log(torch.clamp(a_001_t, min=RELATIVE_LOG_EPS))
        - beta_t * (1.0 - p) * sin_el_t
    )
    return a_001_t * ((p / 0.01) ** exponent_t)


def _backhaul_loss_factor_torch(
    *,
    channel_params: _NativeChannelStaticParams,
    carrier_freq_hz: float,
    elevation_batch_t: torch.Tensor,
) -> torch.Tensor | None:
    channel_p = channel_params
    if not (bool(channel_p.atm_loss_enabled) or bool(channel_p.rain_loss_enabled)):
        return None
    work_dtype = torch.float32 if bool(channel_p.fast_float32) else torch.float64
    theta_t = elevation_batch_t.to(dtype=work_dtype)
    factor_t = theta_t * 0.0 + 1.0
    if bool(channel_p.atm_loss_enabled):
        atm_loss_t = _atmospheric_loss_db_torch(theta_t, base_loss_db=float(channel_p.atm_loss_db)).to(dtype=work_dtype)
        factor_t = factor_t * torch.pow(10.0, -atm_loss_t / 10.0)
    if bool(channel_p.rain_loss_enabled):
        rain_loss_t = _rain_attenuation_db_torch(
            theta_rad_t=theta_t,
            carrier_freq_hz=float(carrier_freq_hz),
            rain_rate_001_mmph=float(channel_p.rain_rate_001_mmph),
            rain_height_km=float(channel_p.rain_height_km),
            station_height_km=float(channel_p.station_height_km),
            latitude_deg=float(channel_p.latitude_deg),
            exceedance_pct=float(channel_p.rain_exceedance_pct),
            polarization_tilt_deg=float(channel_p.rain_polarization_tilt_deg),
        )
        rain_loss_t = rain_loss_t.to(dtype=work_dtype)
        factor_t = factor_t * torch.pow(10.0, -rain_loss_t / 10.0)
    return factor_t.to(dtype=torch.float32)


def _snr_linear_torch(
    *,
    power: float,
    gain_t: torch.Tensor,
    noise_density: float,
    bandwidth_t: torch.Tensor,
    interference_t: torch.Tensor | None = None,
    noise_figure_db: float = 0.0,
) -> torch.Tensor:
    gain_work_t = gain_t.to(dtype=torch.float32 if gain_t.dtype != torch.float64 else torch.float64)
    bandwidth_work_t = bandwidth_t.to(device=gain_work_t.device, dtype=gain_work_t.dtype)
    noise_figure_linear = 10.0 ** (max(float(noise_figure_db), 0.0) / 10.0)
    denom_t = float(noise_density) * float(noise_figure_linear) * bandwidth_work_t
    if interference_t is not None:
        denom_t = denom_t + interference_t.to(device=gain_work_t.device, dtype=gain_work_t.dtype)
    signal_t = float(power) * gain_work_t
    valid_t = (bandwidth_work_t > 0.0) & (denom_t > 0.0)
    denom_safe_t = torch.where(valid_t, denom_t, torch.ones_like(denom_t))
    snr_t = signal_t / denom_safe_t
    return torch.where(valid_t, snr_t, torch.zeros_like(snr_t))


def _spectral_efficiency_torch(snr_t: torch.Tensor) -> torch.Tensor:
    return torch.log2(1.0 + snr_t)


def _spectral_efficiency_reference_like_torch(snr_t: torch.Tensor) -> torch.Tensor:
    work_dtype = torch.float32 if snr_t.dtype != torch.float64 else torch.float64
    snr_work_t = snr_t.to(dtype=work_dtype)
    return torch.log2(1.0 + snr_work_t)


_RICIAN_HERMGAUSS_TORCH_CACHE: dict[tuple[int, str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _hermgauss_nodes_weights_torch(
    points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(int(points), 1)
    key = (n, str(device.type), int(device.index) if device.index is not None else None, dtype)
    cached = _RICIAN_HERMGAUSS_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n)
    nodes_t = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights_t = torch.as_tensor(weights_np, device=device, dtype=dtype)
    _RICIAN_HERMGAUSS_TORCH_CACHE[key] = (nodes_t, weights_t)
    return nodes_t, weights_t


def _rician_ergodic_spectral_efficiency_torch(
    snr_t: torch.Tensor,
    *,
    k_factor: float,
    quadrature_points: int,
) -> torch.Tensor:
    work_dtype = torch.float32 if snr_t.dtype != torch.float64 else torch.float64
    snr_work_t = torch.clamp(snr_t.to(dtype=work_dtype), min=0.0)
    nodes_t, weights_t = _hermgauss_nodes_weights_torch(
        quadrature_points,
        device=snr_work_t.device,
        dtype=work_dtype,
    )
    k_value = max(float(k_factor), 0.0)
    sigma = math.sqrt(1.0 / (2.0 * (k_value + 1.0)))
    mean_real = math.sqrt(k_value / (k_value + 1.0))
    real_t = mean_real + sigma * math.sqrt(2.0) * nodes_t[:, None]
    imag_t = sigma * math.sqrt(2.0) * nodes_t[None, :]
    gain_t = real_t.square() + imag_t.square()
    weight_t = (weights_t[:, None] * weights_t[None, :]) / math.pi
    expanded_snr_t = snr_work_t.unsqueeze(-1).unsqueeze(-1)
    se_t = torch.log2(1.0 + expanded_snr_t * gain_t)
    return (se_t * weight_t).sum(dim=(-1, -2))


def _access_spectral_efficiency_torch(
    snr_t: torch.Tensor,
    *,
    params: _NativeAccessRateStaticParams,
) -> torch.Tensor:
    if int(params.fading_mode_code) == 1:
        return _rician_ergodic_spectral_efficiency_torch(
            snr_t,
            k_factor=float(params.rician_k),
            quadrature_points=int(params.ergodic_rician_quadrature_points),
        )
    return _spectral_efficiency_reference_like_torch(snr_t)


def _semantic_quantum(cfg, attr: str, default: float) -> float:
    value = getattr(cfg, attr, default)
    if value is None:
        return float(default)
    return max(float(value), 0.0)


def _quantize_semantic_tensor(
    value_t: torch.Tensor,
    *,
    quantum: float,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    quantum_value = float(quantum)
    if quantum_value <= 0.0:
        return value_t.to(dtype=out_dtype)
    work_dtype = torch.float32 if out_dtype != torch.float64 else torch.float64
    work_t = value_t.to(dtype=work_dtype)
    return (torch.round(work_t / quantum_value) * quantum_value).to(dtype=out_dtype)


def _sum_tensor_float32_semantics(
    value_t: torch.Tensor,
    *,
    dim: int | tuple[int, ...],
) -> torch.Tensor:
    return value_t.to(dtype=torch.float32).sum(dim=dim, dtype=torch.float32)


def _sum_tensor_hot_semantics(
    value_t: torch.Tensor,
    *,
    dim: int | tuple[int, ...],
    cfg,
) -> torch.Tensor:
    del cfg
    return _sum_tensor_float32_semantics(value_t, dim=dim)


def _quantize_access_gain_snapshot_numpy(cfg, gain: np.ndarray) -> np.ndarray:
    quantum = float(_semantic_quantum(cfg, "structured_access_gain_quantum", 5.0e-16))
    if quantum <= 0.0:
        return np.asarray(gain, dtype=np.float32)
    gain32 = np.asarray(gain, dtype=np.float32)
    quantum32 = np.float32(quantum)
    return np.asarray(np.rint(gain32 / quantum32).astype(np.float32) * quantum32, dtype=np.float32)


def _quantize_access_pathloss_db_numpy(cfg, pathloss: np.ndarray) -> np.ndarray:
    quantum = float(_semantic_quantum(cfg, "structured_access_pathloss_db_quantum", 1.0e-2))
    if quantum <= 0.0:
        return np.asarray(pathloss, dtype=np.float32)
    pathloss32 = np.asarray(pathloss, dtype=np.float32)
    quantum32 = np.float32(quantum)
    return np.asarray(np.rint(pathloss32 / quantum32).astype(np.float32) * quantum32, dtype=np.float32)


def _quantize_access_gain_snapshot_tensor(
    gain_t: torch.Tensor,
    *,
    quantum: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return _quantize_semantic_tensor(
        gain_t,
        quantum=float(quantum),
        out_dtype=out_dtype,
    )


def _quantize_access_pathloss_db_tensor(cfg, pathloss_t: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
    return _quantize_semantic_tensor(
        pathloss_t,
        quantum=_semantic_quantum(cfg, "structured_access_pathloss_db_quantum", 1.0e-2),
        out_dtype=out_dtype,
    )


def _doppler_attenuation_torch(nu_t: torch.Tensor, *, subcarrier_spacing: float) -> torch.Tensor:
    if float(subcarrier_spacing) <= 0.0:
        return nu_t.to(dtype=torch.float32) * 0.0 + 1.0
    return torch.sinc(nu_t / float(subcarrier_spacing)).square()


def _compute_access_link_gain_matrix_tensor_impl(
    *,
    gu_pos_t: torch.Tensor,
    uav_pos_t: torch.Tensor,
    channel_params: _NativeChannelStaticParams,
    candidate_params: _NativeCandidateStaticParams,
) -> torch.Tensor:
    work_dtype = torch.float32 if bool(channel_params.fast_float32) else torch.float64
    gu_pos_work_t = gu_pos_t.to(dtype=work_dtype)
    uav_pos_work_t = uav_pos_t.to(dtype=work_dtype)
    diff_t = gu_pos_work_t[:, :, None, :] - uav_pos_work_t[:, None, :, :]
    d2d_t = torch.linalg.norm(diff_t, dim=-1)
    height_value = float(candidate_params.uav_height)
    pl_base_value = float(channel_params.pathloss_const_db) + 20.0 * math.log10(float(channel_params.carrier_freq) / 1.0e9)
    d3d_t = torch.sqrt(d2d_t * d2d_t + height_value * height_value)
    safe_d_t = _torch_positive(d3d_t, GEOMETRY_DENOM_EPS)
    phi_t = torch.asin(torch.clamp(height_value / safe_d_t, -1.0, 1.0))
    pl_los_t = pl_base_value + float(channel_params.xi_los) + 20.0 * torch.log10(safe_d_t)
    if str(channel_params.pathloss_mode) == "free_space":
        pathloss_t = pl_los_t
    else:
        pl_nlos_t = pl_base_value + float(channel_params.xi_nlos) + 20.0 * torch.log10(safe_d_t)
        phi_deg_t = phi_t * (180.0 / math.pi)
        los_a = float(channel_params.los_a)
        los_b = float(channel_params.los_b)
        p_los_t = 1.0 / (1.0 + los_a * torch.exp(-los_b * (phi_deg_t - los_a)))
        pathloss_t = p_los_t * pl_los_t + (1.0 - p_los_t) * pl_nlos_t
    pathloss_t = _quantize_semantic_tensor(
        pathloss_t,
        quantum=float(channel_params.access_pathloss_db_quantum),
        out_dtype=work_dtype,
    )
    return torch.exp((-pathloss_t / 10.0) * math.log(10.0)).to(dtype=torch.float32)


def _sat_selection_matrix_or_default_batch(
    *,
    cfg,
    batch_size: int,
    sat_selection_matrix: np.ndarray | None,
) -> np.ndarray:
    select_k = _sat_action_select_k_from_config(cfg)
    if sat_selection_matrix is None:
        return np.full((batch_size, int(cfg.num_uav), select_k), -1, dtype=np.int64)
    matrix = np.asarray(sat_selection_matrix, dtype=np.int64)
    expected = (batch_size, int(cfg.num_uav), select_k)
    if tuple(matrix.shape) != expected:
        raise ValueError(f"sat_selection_matrix must have shape {expected}, got {tuple(matrix.shape)}.")
    return matrix


def _reward_aligned_feature_bundle_batch_from_state(
    *,
    cfg,
    gu_queue_batch: np.ndarray,
    arrival_ref_batch: np.ndarray,
    gu_ema_batch: np.ndarray,
    uav_ema_batch: np.ndarray,
    sat_ema_batch: np.ndarray,
    assoc_batch: np.ndarray,
    sat_selection_matrix_batch: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    gu_queue_arr = np.asarray(gu_queue_batch, dtype=np.float32)
    arrival_ref_arr = require_positive_array(
        np.asarray(arrival_ref_batch, dtype=np.float32).reshape(-1),
        name="arrival_ref_bits_per_step",
    )
    gu_ema_arr = np.asarray(gu_ema_batch, dtype=np.float32)
    uav_ema_arr = np.asarray(uav_ema_batch, dtype=np.float32)
    sat_ema_arr = np.asarray(sat_ema_batch, dtype=np.float32)
    assoc_arr = np.asarray(assoc_batch, dtype=np.int32)
    batch_size = int(gu_queue_arr.shape[0])

    eps_value = getattr(cfg, "service_floor_bits_per_step", None)
    if eps_value is None:
        eps_value = getattr(cfg, "bw_weighted_workload_eps", 1.0)
    eps = max(float(eps_value or 0.0), float(NORMALIZATION_DENOM_EPS))
    sat_cost = (1.0 / np.maximum(sat_ema_arr, eps)).astype(np.float32, copy=False)
    sat_cost_fallback = (
        np.mean(sat_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False)
        if sat_cost.shape[1] > 0
        else np.zeros((batch_size,), dtype=np.float32)
    )

    sat_selection_arr = _sat_selection_matrix_or_default_batch(
        cfg=cfg,
        batch_size=batch_size,
        sat_selection_matrix=sat_selection_matrix_batch,
    )
    sat_selected = _sat_selection_presence_batch(sat_selection_arr, num_sat=int(cfg.num_sat))
    sat_selected_count = np.sum(sat_selected, axis=2, dtype=np.float32)
    sat_cost_sum = np.sum(sat_selected * sat_cost[:, None, :], axis=2, dtype=np.float32)
    uav_downstream_cost = np.where(
        sat_selected_count > 0.0,
        ratio_or_zero(sat_cost_sum, sat_selected_count),
        sat_cost_fallback[:, None],
    ).astype(np.float32, copy=False)

    uav_cost = (1.0 / np.maximum(uav_ema_arr, eps) + uav_downstream_cost).astype(np.float32, copy=False)
    uav_cost_fallback = (
        np.mean(uav_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False)
        if uav_cost.shape[1] > 0
        else np.zeros((batch_size,), dtype=np.float32)
    )

    gu_downstream_cost = np.broadcast_to(uav_cost_fallback[:, None], assoc_arr.shape).copy()
    if int(cfg.num_uav) > 0 and int(cfg.num_gu) > 0:
        valid_assoc = (assoc_arr >= 0) & (assoc_arr < int(cfg.num_uav))
        if np.any(valid_assoc):
            batch_coords = np.broadcast_to(np.arange(batch_size, dtype=np.int64)[:, None], assoc_arr.shape)
            gu_downstream_cost[valid_assoc] = uav_cost[batch_coords[valid_assoc], assoc_arr[valid_assoc]]
    gu_cost = (1.0 / np.maximum(gu_ema_arr, eps) + gu_downstream_cost).astype(np.float32, copy=False)

    assoc_uav_cost = np.broadcast_to(uav_cost_fallback[:, None], assoc_arr.shape).copy()
    if int(cfg.num_uav) > 0 and int(cfg.num_gu) > 0:
        valid_assoc = (assoc_arr >= 0) & (assoc_arr < int(cfg.num_uav))
        if np.any(valid_assoc):
            batch_coords = np.broadcast_to(np.arange(batch_size, dtype=np.int64)[:, None], assoc_arr.shape)
            assoc_uav_cost[valid_assoc] = uav_cost[batch_coords[valid_assoc], assoc_arr[valid_assoc]]

    assoc_sat_cost_mean = np.broadcast_to(sat_cost_fallback[:, None], assoc_arr.shape).copy()
    if int(cfg.num_uav) > 0 and int(cfg.num_gu) > 0:
        valid_assoc = (assoc_arr >= 0) & (assoc_arr < int(cfg.num_uav))
        if np.any(valid_assoc):
            batch_coords = np.broadcast_to(np.arange(batch_size, dtype=np.int64)[:, None], assoc_arr.shape)
            assoc_sat_cost_mean[valid_assoc] = uav_downstream_cost[batch_coords[valid_assoc], assoc_arr[valid_assoc]]

    local_gu_service_cost = (1.0 / np.maximum(gu_ema_arr, eps)).astype(np.float32, copy=False)
    weighted_queue_cost = (gu_cost * gu_queue_arr).astype(np.float32, copy=False)
    weighted_queue_cost_mean = np.maximum(
        np.mean(weighted_queue_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        NORMALIZATION_DENOM_EPS,
    )
    weighted_queue_cost_relative = (
        weighted_queue_cost / weighted_queue_cost_mean[:, None]
    ).astype(np.float32, copy=False)

    gu_default = arrival_ref_arr / max(float(cfg.num_gu), 1.0)
    uav_default = arrival_ref_arr / max(float(cfg.num_uav), 1.0)
    sat_default = arrival_ref_arr / max(float(_bw_weighted_workload_sat_active_ref_count(cfg)), 1.0)
    sat_cost_ref = 1.0 / np.maximum(sat_default, eps)
    uav_cost_ref = 1.0 / np.maximum(uav_default, eps) + sat_cost_ref
    gu_local_cost_ref = 1.0 / np.maximum(gu_default, eps)
    gu_total_cost_ref = gu_local_cost_ref + uav_cost_ref
    weighted_queue_ref = require_positive_array(
        arrival_ref_arr * gu_total_cost_ref,
        name="weighted queue reward feature reference",
    )

    local_cost_mean = np.maximum(
        np.mean(local_gu_service_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        LOG_RATIO_EPS,
    )
    assoc_uav_cost_mean = np.maximum(
        np.mean(assoc_uav_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        LOG_RATIO_EPS,
    )
    assoc_sat_cost_mean_mean = np.maximum(
        np.mean(assoc_sat_cost_mean, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        LOG_RATIO_EPS,
    )
    sat_cost_mean = np.maximum(
        np.mean(sat_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        LOG_RATIO_EPS,
    )
    uav_cost_mean = np.maximum(
        np.mean(uav_cost, axis=1, dtype=np.float32).astype(np.float32, copy=False),
        LOG_RATIO_EPS,
    )

    gu_reward_aligned = {
        "local_gu_service_cost": np.log(log_ratio_argument(local_gu_service_cost) / local_cost_mean[:, None]).astype(np.float32, copy=False),
        "assoc_uav_cost": np.log(log_ratio_argument(assoc_uav_cost) / assoc_uav_cost_mean[:, None]).astype(np.float32, copy=False),
        "assoc_sat_cost_mean": np.log(log_ratio_argument(assoc_sat_cost_mean) / assoc_sat_cost_mean_mean[:, None]).astype(np.float32, copy=False),
        "weighted_queue_cost": np.log1p(np.maximum(weighted_queue_cost, 0.0) / weighted_queue_ref[:, None]).astype(np.float32, copy=False),
        "weighted_queue_cost_relative": np.log(np.maximum(weighted_queue_cost_relative, RELATIVE_LOG_EPS)).astype(np.float32, copy=False),
    }
    uav_assoc_uav_cost = np.log(
        log_ratio_argument(uav_cost) / uav_cost_mean[:, None]
    ).astype(np.float32, copy=False)
    sat_cost_norm = np.log(
        log_ratio_argument(sat_cost) / sat_cost_mean[:, None]
    ).astype(np.float32, copy=False)
    return gu_reward_aligned, uav_assoc_uav_cost, sat_cost_norm


def _gu_proxy_feature_arrays_batch_from_state(
    *,
    cfg,
    gu_queue_batch: np.ndarray,
    arrival_rate_vec_batch: np.ndarray,
    recent_arrival_batch: np.ndarray,
    recent_service_batch: np.ndarray,
    urgency_risk_batch: np.ndarray,
    downstream_pressure_batch: np.ndarray,
    service_gap_batch: np.ndarray,
    service_gap_risk_batch: np.ndarray,
    deadline_steps_batch: np.ndarray,
    deadline_slack_batch: np.ndarray,
    deadline_risk_batch: np.ndarray,
    reward_aligned: dict[str, np.ndarray],
) -> list[np.ndarray]:
    gu_queue_arr = np.asarray(gu_queue_batch, dtype=np.float32)
    arrival_rate_vec_arr = np.asarray(arrival_rate_vec_batch, dtype=np.float32)
    recent_arrival_arr = np.asarray(recent_arrival_batch, dtype=np.float32)
    recent_service_arr = np.asarray(recent_service_batch, dtype=np.float32)
    urgency_risk_arr = np.asarray(urgency_risk_batch, dtype=np.float32)
    downstream_pressure_arr = np.asarray(downstream_pressure_batch, dtype=np.float32)
    service_gap_arr = np.asarray(service_gap_batch, dtype=np.float32)
    service_gap_risk_arr = np.asarray(service_gap_risk_batch, dtype=np.float32)
    deadline_steps_arr = np.maximum(np.asarray(deadline_steps_batch, dtype=np.float32), 1.0e-6)
    deadline_slack_arr = np.asarray(deadline_slack_batch, dtype=np.float32)
    deadline_risk_arr = np.asarray(deadline_risk_batch, dtype=np.float32)
    base_arrival_arr = require_positive_array(
        arrival_rate_vec_arr.mean(axis=1, keepdims=True) * float(cfg.tau0),
        name="per-GU arrival reference bits per step",
    )

    features: list[np.ndarray] = []
    if bool(getattr(cfg, "obs_user_include_arrival_rate", False)):
        features.append((arrival_rate_vec_arr / base_arrival_arr).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_recent_arrival", False)):
        features.append((recent_arrival_arr / base_arrival_arr).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_recent_service", False)):
        features.append((recent_service_arr / base_arrival_arr).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_queue_headroom", False)):
        queue_headroom = 1.0 - (gu_queue_arr / normalize_scale(float(cfg.queue_max_gu)))
        features.append(queue_headroom.astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_local_gu_service_cost", False)):
        features.append(np.asarray(reward_aligned["local_gu_service_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_assoc_uav_cost", False)):
        features.append(np.asarray(reward_aligned["assoc_uav_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_assoc_sat_cost_mean", False)):
        features.append(np.asarray(reward_aligned["assoc_sat_cost_mean"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_weighted_queue_cost", False)):
        features.append(np.asarray(reward_aligned["weighted_queue_cost"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_weighted_queue_cost_relative", False)):
        features.append(np.asarray(reward_aligned["weighted_queue_cost_relative"], dtype=np.float32))
    if bool(getattr(cfg, "obs_user_include_urgency_risk", False)):
        features.append(urgency_risk_arr.astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_downstream_pressure", False)):
        features.append(downstream_pressure_arr.astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_service_gap", False)):
        cap_steps = max(float(getattr(cfg, "service_gap_cap_steps", 8.0) or 0.0), 1.0e-6)
        features.append((service_gap_arr / cap_steps).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_service_gap_risk", False)):
        features.append(service_gap_risk_arr.astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_deadline_slack", False)):
        features.append(np.clip(deadline_slack_arr / deadline_steps_arr, -1.0, 1.0).astype(np.float32, copy=False))
    if bool(getattr(cfg, "obs_user_include_deadline_risk", False)):
        features.append(np.clip(deadline_risk_arr, 0.0, 2.0).astype(np.float32, copy=False))
    return features


def _as_grouped_numpy_batch(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    *,
    expected_envs: int,
    trailing_shape: Sequence[int],
    dtype,
) -> np.ndarray:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        arr = _compat_numpy_array(value, dtype=dtype)
    else:
        arr = np.stack([np.asarray(item, dtype=dtype) for item in value], axis=0)
    expected_shape = (int(expected_envs), *tuple(int(dim) for dim in trailing_shape))
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"Expected grouped batch shape {expected_shape}, got {tuple(arr.shape)}.")
    return arr


def _as_grouped_tensor_batch(
    value: torch.Tensor | np.ndarray | Sequence[Any],
    *,
    expected_envs: int,
    trailing_shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    expected_shape = (int(expected_envs), *tuple(int(dim) for dim in trailing_shape))
    tensor = _as_kernel_tensor(value, dtype=dtype, device=device)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected grouped batch tensor shape {expected_shape}, got {tuple(tensor.shape)}.")
    return tensor


def _traffic_level_state_from_cfg(cfg) -> tuple[int, float, float, float]:
    raw_level = getattr(cfg, "traffic_level", 2)
    level = int(2 if raw_level is None else raw_level)
    level = int(np.clip(level, 0, 2))
    if level == 0:
        ratio = float(getattr(cfg, "traffic_level_nav_ratio", 0.08) or 0.08)
    elif level == 1:
        ratio = float(getattr(cfg, "traffic_level_easy_ratio", 0.5) or 0.5)
    else:
        ratio = float(getattr(cfg, "traffic_level_hard_ratio", 1.0) or 1.0)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    effective_rate = max(float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0), 0.0) * ratio
    arrival_ref = reward_ratio_denominator_scalar(
        effective_rate * float(cfg.num_gu) * float(cfg.tau0),
        name="arrival_ref_bits_per_step",
    )
    return level, ratio, effective_rate, arrival_ref


def _uav_init_boundary_margin_from_cfg(cfg) -> float:
    steps = max(float(getattr(cfg, "uav_init_boundary_margin_steps", 0.0) or 0.0), 0.0)
    margin = steps * float(cfg.v_max) * float(cfg.tau0)
    max_margin = max(0.0, 0.5 * float(cfg.map_size) - 1.0e-6)
    return float(min(margin, max_margin))


def _sample_uav_safe_random_positions_native(cfg, rng: np.random.Generator) -> np.ndarray:
    if int(cfg.num_uav) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    margin = _uav_init_boundary_margin_from_cfg(cfg)
    low = float(margin)
    high = float(cfg.map_size) - float(margin)
    min_spacing_cfg = getattr(cfg, "uav_init_min_spacing", None)
    min_spacing = float(cfg.d_safe) if min_spacing_cfg is None else max(float(min_spacing_cfg), 0.0)
    max_tries = max(int(getattr(cfg, "uav_init_max_tries", 256) or 256), 1)
    positions = np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
    for uav_index in range(int(cfg.num_uav)):
        placed = False
        for _ in range(max_tries):
            candidate = rng.uniform(low, high, size=(2,)).astype(np.float32, copy=False)
            if uav_index > 0:
                dist = np.linalg.norm(positions[:uav_index] - candidate[None, :], axis=1)
                if not np.all(dist >= min_spacing - 1.0e-6):
                    continue
            positions[uav_index] = candidate
            placed = True
            break
        if not placed:
            raise RuntimeError(
                "Could not sample UAV initial positions satisfying boundary margin "
                "and minimum spacing constraints."
            )
    return positions


def _sample_uav_initial_velocities_native(cfg, rng: np.random.Generator) -> np.ndarray:
    if int(cfg.num_uav) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if not bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
        return np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
    speed_frac = max(float(getattr(cfg, "uav_init_speed_frac", 0.0) or 0.0), 0.0)
    speed = min(speed_frac, 1.0) * float(cfg.v_max)
    if speed <= 0.0:
        return np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(int(cfg.num_uav),))
    vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speed
    return vel.astype(np.float32, copy=False)


def _sample_uav_positions_native(
    cfg,
    rng: np.random.Generator,
    *,
    gu_pos: np.ndarray,
    gu_cluster_centers: np.ndarray | None = None,
    gu_cluster_counts: np.ndarray | None = None,
    episode_idx: int,
) -> np.ndarray:
    if int(cfg.num_uav) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    spawn_mode = str(getattr(cfg, "uav_spawn_mode", "default") or "default").strip().lower()
    if spawn_mode == "gu_centroid":
        if int(cfg.num_gu) > 0 and np.asarray(gu_pos).shape == (int(cfg.num_gu), 2):
            center = np.mean(np.asarray(gu_pos, dtype=np.float32), axis=0).astype(np.float32, copy=False)
        else:
            center = np.array([float(cfg.map_size) * 0.5, float(cfg.map_size) * 0.5], dtype=np.float32)
        positions = np.repeat(center[None, :], int(cfg.num_uav), axis=0)
        return np.clip(positions, 0.0, float(cfg.map_size)).astype(np.float32, copy=False)
    if spawn_mode == "gu_cluster_centers":
        centers_raw = np.zeros((0, 2), dtype=np.float32) if gu_cluster_centers is None else np.asarray(gu_cluster_centers, dtype=np.float32)
        counts_raw = np.zeros((0,), dtype=np.float32) if gu_cluster_counts is None else np.asarray(gu_cluster_counts, dtype=np.float32).reshape(-1)
        cluster_count = min(int(centers_raw.shape[0]) if centers_raw.ndim == 2 else 0, int(counts_raw.shape[0]))
        if cluster_count > 0:
            centers = centers_raw[:cluster_count].reshape(cluster_count, 2)
            counts = counts_raw[:cluster_count]
            order = np.argsort(-counts, kind="stable")
            selected = centers[order[np.arange(int(cfg.num_uav)) % max(order.size, 1)]]
            return np.clip(selected, 0.0, float(cfg.map_size)).astype(np.float32, copy=False)
        if int(cfg.num_gu) > 0 and np.asarray(gu_pos).shape == (int(cfg.num_gu), 2):
            center = np.mean(np.asarray(gu_pos, dtype=np.float32), axis=0).astype(np.float32, copy=False)
        else:
            center = np.array([float(cfg.map_size) * 0.5, float(cfg.map_size) * 0.5], dtype=np.float32)
        positions = np.repeat(center[None, :], int(cfg.num_uav), axis=0)
        return np.clip(positions, 0.0, float(cfg.map_size)).astype(np.float32, copy=False)
    if bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
        return _sample_uav_safe_random_positions_native(cfg, rng)
    use_curriculum_spawn = ablation_flag(
        cfg,
        "use_curriculum_spawn",
        fallback_attr="uav_spawn_curriculum_enabled",
        default=False,
    )
    if not use_curriculum_spawn:
        return rng.uniform(0.0, float(cfg.map_size), size=(int(cfg.num_uav), 2)).astype(np.float32, copy=False)
    steps = int(getattr(cfg, "uav_spawn_curriculum_steps", 0) or 0)
    progress = 1.0 if steps <= 0 else min(1.0, float(max(int(episode_idx) - 1, 0)) / float(steps))
    if progress >= 1.0 and bool(getattr(cfg, "uav_spawn_full_random_final", True)):
        return rng.uniform(0.0, float(cfg.map_size), size=(int(cfg.num_uav), 2)).astype(np.float32, copy=False)
    radius_start = max(float(getattr(cfg, "uav_spawn_radius_start", 0.0) or 0.0), 0.0)
    radius_end_cfg = getattr(cfg, "uav_spawn_radius_end", None)
    radius_end = float(cfg.map_size) * 0.5 if radius_end_cfg is None else float(radius_end_cfg)
    radius_end = max(radius_end, radius_start)
    radius = radius_start + (radius_end - radius_start) * progress
    if int(cfg.num_gu) > 0 and np.asarray(gu_pos).shape == (int(cfg.num_gu), 2):
        center = np.asarray(gu_pos[int(rng.integers(0, int(cfg.num_gu)))], dtype=np.float32)
    else:
        center = np.array([float(cfg.map_size) * 0.5, float(cfg.map_size) * 0.5], dtype=np.float32)
    positions = np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
    for uav_index in range(int(cfg.num_uav)):
        pos = None
        candidate = None
        for _ in range(20):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            radius_sample = radius * math.sqrt(float(rng.uniform(0.0, 1.0)))
            candidate = center + np.array(
                [math.cos(ang) * radius_sample, math.sin(ang) * radius_sample],
                dtype=np.float32,
            )
            if 0.0 <= candidate[0] <= float(cfg.map_size) and 0.0 <= candidate[1] <= float(cfg.map_size):
                pos = candidate
                break
        if pos is None:
            pos = np.clip(center if candidate is None else candidate, 0.0, float(cfg.map_size))
        positions[uav_index] = np.asarray(pos, dtype=np.float32)
    return positions


def _sample_deadline_steps_native(cfg, rng: np.random.Generator) -> np.ndarray:
    if int(cfg.num_gu) <= 0:
        return np.zeros((0,), dtype=np.float32)
    base_steps = max(float(getattr(cfg, "deadline_base_steps", 4.0) or 0.0), 1.0)
    jitter_steps = max(float(getattr(cfg, "deadline_jitter_steps", 0.0) or 0.0), 0.0)
    if jitter_steps <= NORMALIZATION_DENOM_EPS:
        return np.full((int(cfg.num_gu),), base_steps, dtype=np.float32)
    low = max(base_steps - jitter_steps, 1.0)
    high = max(base_steps + jitter_steps, low)
    return rng.uniform(low, high, size=(int(cfg.num_gu),)).astype(np.float32, copy=False)


def _queue_init_arrival_ref_from_rate(cfg, effective_task_arrival_rate: float) -> float:
    return float(effective_task_arrival_rate) * float(cfg.num_gu) * float(cfg.tau0)


def _queue_init_layer_ref_from_rate(cfg, effective_task_arrival_rate: float, layer: str) -> float:
    ref_value = getattr(cfg, f"queue_ref_{layer}_per_step", None)
    if ref_value is not None:
        return max(float(ref_value), 0.0)
    return _queue_init_arrival_ref_from_rate(cfg, effective_task_arrival_rate)


def _queue_init_entity_ref_from_rate(cfg, effective_task_arrival_rate: float, layer: str) -> float:
    total_ref = _queue_init_layer_ref_from_rate(cfg, effective_task_arrival_rate, layer)
    if layer == "gu":
        entities = max(float(cfg.num_gu), 1.0)
    elif layer == "uav":
        entities = max(float(cfg.num_uav), 1.0)
    else:
        active_count = getattr(cfg, "queue_ref_sat_active_count", None)
        entities = max(float(active_count), 1.0) if active_count is not None else max(float(cfg.num_sat), 1.0)
    return total_ref / entities


def _resolve_queue_init_total_native(
    cfg,
    *,
    abs_attr: str,
    steps_attr: str,
    frac_attr: str,
    layer: str,
    total_cap: float,
    effective_task_arrival_rate: float,
) -> float:
    abs_value = getattr(cfg, abs_attr, None)
    if abs_value is not None:
        return min(max(float(abs_value), 0.0), float(total_cap))
    steps_value = getattr(cfg, steps_attr, None)
    if steps_value is not None:
        total = max(float(steps_value), 0.0) * _queue_init_layer_ref_from_rate(cfg, effective_task_arrival_rate, layer)
        return min(total, float(total_cap))
    frac_value = max(float(getattr(cfg, frac_attr, 0.0) or 0.0), 0.0)
    return min(float(np.clip(frac_value, 0.0, 1.0)) * float(total_cap), float(total_cap))


def _build_hotspot_subsets_from_assoc(cfg, gu_pos: np.ndarray, assoc: np.ndarray) -> list[np.ndarray]:
    target_count = max(int(getattr(cfg, "hotspot_num_subsets", 0) or 0), 0)
    subset_size_cfg = max(int(getattr(cfg, "hotspot_subset_size", 0) or 0), 0)
    if target_count <= 0 or subset_size_cfg <= 0 or int(cfg.num_gu) <= 0:
        return []
    gu_pos_arr = np.asarray(gu_pos, dtype=np.float32)

    def _candidate_records(indices: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size <= 0:
            return []
        local_size = min(subset_size_cfg, int(idx.size))
        if local_size <= 0:
            return []
        pos = gu_pos_arr[idx]
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        records: list[tuple[tuple[int, ...], float]] = []
        seen: set[tuple[int, ...]] = set()
        for row in range(int(idx.size)):
            order = np.argsort(dist[row], kind="stable")[:local_size]
            subset = tuple(sorted(int(x) for x in idx[order].tolist()))
            if subset in seen:
                continue
            seen.add(subset)
            local_dist = dist[np.ix_(order, order)]
            records.append((subset, float(np.mean(local_dist))))
        return records

    assoc_arr = np.asarray(assoc, dtype=np.int32)
    records: list[tuple[tuple[int, ...], float]] = []
    for uav_index in range(int(cfg.num_uav)):
        group = np.flatnonzero(assoc_arr == int(uav_index)).astype(np.int32, copy=False)
        records.extend(_candidate_records(group))
    if len(records) < target_count:
        records.extend(_candidate_records(np.arange(int(cfg.num_gu), dtype=np.int32)))
    dedup: dict[tuple[int, ...], float] = {}
    for subset, score in records:
        prev = dedup.get(subset)
        dedup[subset] = score if prev is None else min(prev, score)
    candidates = [
        (np.asarray(subset, dtype=np.int32), float(score))
        for subset, score in dedup.items()
        if len(subset) > 0
    ]
    candidates.sort(key=lambda item: (-int(item[0].size), item[1], tuple(int(x) for x in item[0].tolist())))
    if not candidates:
        return []
    usage = np.zeros((int(cfg.num_gu),), dtype=np.int32)
    selected: list[np.ndarray] = []
    remaining = list(candidates)
    while remaining and len(selected) < target_count:
        best_index = min(
            range(len(remaining)),
            key=lambda idx: (
                float(remaining[idx][1]),
                int(np.sum(usage[remaining[idx][0]])),
                tuple(int(x) for x in remaining[idx][0].tolist()),
            ),
        )
        subset, _ = remaining.pop(best_index)
        selected.append(subset)
        usage[subset] += 1
    return selected


def _build_native_reset_queue_state(
    cfg,
    *,
    effective_task_arrival_rate: float,
    gu_pos: np.ndarray,
    uav_pos: np.ndarray,
    assoc_init: np.ndarray,
    rng: np.random.Generator | None = None,
    preload_draw: float | None = None,
    preload_choice: int | None = None,
) -> dict[str, Any]:
    gu_queue = np.zeros((int(cfg.num_gu),), dtype=np.float32)
    uav_queue = np.zeros((int(cfg.num_uav),), dtype=np.float32)
    sat_queue = np.zeros((int(cfg.num_sat),), dtype=np.float32)
    init_totals = {
        "gu": _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_gu_abs",
            steps_attr="queue_init_gu_steps",
            frac_attr="queue_init_frac",
            layer="gu",
            total_cap=float(cfg.num_gu) * float(cfg.queue_max_gu),
            effective_task_arrival_rate=effective_task_arrival_rate,
        ),
        "uav": _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_uav_abs",
            steps_attr="queue_init_uav_steps",
            frac_attr="queue_init_uav_frac",
            layer="uav",
            total_cap=float(cfg.num_uav) * float(cfg.queue_max_uav),
            effective_task_arrival_rate=effective_task_arrival_rate,
        ),
        "sat": _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_sat_abs",
            steps_attr="queue_init_sat_steps",
            frac_attr="queue_init_sat_frac",
            layer="sat",
            total_cap=float(cfg.num_sat) * float(cfg.queue_max_sat),
            effective_task_arrival_rate=effective_task_arrival_rate,
        ),
    }
    if int(cfg.num_gu) > 0 and init_totals["gu"] > 0.0:
        gu_queue.fill(float(init_totals["gu"]) / max(float(cfg.num_gu), 1.0))
    if int(cfg.num_uav) > 0 and init_totals["uav"] > 0.0:
        uav_queue.fill(float(init_totals["uav"]) / max(float(cfg.num_uav), 1.0))
    if int(cfg.num_sat) > 0 and init_totals["sat"] > 0.0:
        sat_queue.fill(float(init_totals["sat"]) / max(float(cfg.num_sat), 1.0))

    if str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower() == "sticky_subset_hotspot":
        hotspot_subsets = _build_hotspot_subsets_from_assoc(cfg, gu_pos, assoc_init)
    else:
        hotspot_subsets = []
    hotspot_member_mask = np.zeros((len(hotspot_subsets), int(cfg.num_gu)), dtype=bool)
    for subset_index, subset in enumerate(hotspot_subsets):
        hotspot_member_mask[subset_index, np.asarray(subset, dtype=np.int32)] = True
    hotspot_active_idx = -1
    last_hotspot_index = -1
    last_hotspot_mask = np.zeros((int(cfg.num_gu),), dtype=np.float32)

    if bool(getattr(cfg, "preload_enabled", False)) and int(cfg.num_gu) > 0:
        prob = float(np.clip(float(getattr(cfg, "preload_prob", 0.0) or 0.0), 0.0, 1.0))
        if prob > 0.0:
            draw = float(rng.random()) if rng is not None else float(1.0 if preload_draw is None else preload_draw)
            if draw <= prob and hotspot_member_mask.shape[0] > 0:
                if rng is not None:
                    hotspot_active_idx = int(rng.integers(hotspot_member_mask.shape[0]))
                else:
                    hotspot_active_idx = int(0 if preload_choice is None else preload_choice) % int(hotspot_member_mask.shape[0])
                last_hotspot_index = hotspot_active_idx
                hot_mask = np.asarray(hotspot_member_mask[hotspot_active_idx], dtype=bool)
                last_hotspot_mask = hot_mask.astype(np.float32, copy=False)
                hot_gu_steps = max(float(getattr(cfg, "preload_hot_gu_steps", 0.0) or 0.0), 0.0)
                bg_gu_steps = max(float(getattr(cfg, "preload_bg_gu_steps", 0.0) or 0.0), 0.0)
                hot_uav_steps = max(float(getattr(cfg, "preload_hot_uav_steps", 0.0) or 0.0), 0.0)
                sat_steps = max(float(getattr(cfg, "preload_sat_steps", 0.0) or 0.0), 0.0)
                gu_bg_value = bg_gu_steps * _queue_init_entity_ref_from_rate(cfg, effective_task_arrival_rate, "gu")
                gu_hot_value = hot_gu_steps * _queue_init_entity_ref_from_rate(cfg, effective_task_arrival_rate, "gu")
                if gu_bg_value > 0.0:
                    gu_queue = np.maximum(gu_queue, gu_bg_value).astype(np.float32, copy=False)
                if gu_hot_value > 0.0 and bool(np.any(hot_mask)):
                    gu_queue[hot_mask] = np.maximum(gu_queue[hot_mask], gu_hot_value).astype(np.float32, copy=False)
                gu_queue = np.minimum(gu_queue, float(cfg.queue_max_gu)).astype(np.float32, copy=False)
                if hot_uav_steps > 0.0 and int(cfg.num_uav) > 0:
                    hot_assoc = np.asarray(assoc_init, dtype=np.int32)[hot_mask]
                    hot_uavs = np.unique(hot_assoc[hot_assoc >= 0])
                    if hot_uavs.size > 0:
                        uav_hot_value = hot_uav_steps * _queue_init_entity_ref_from_rate(
                            cfg,
                            effective_task_arrival_rate,
                            "uav",
                        )
                        uav_queue[hot_uavs] = np.maximum(uav_queue[hot_uavs], uav_hot_value).astype(
                            np.float32,
                            copy=False,
                        )
                        uav_queue = np.minimum(uav_queue, float(cfg.queue_max_uav)).astype(np.float32, copy=False)
                if sat_steps > 0.0 and int(cfg.num_sat) > 0:
                    sat_value = sat_steps * _queue_init_entity_ref_from_rate(cfg, effective_task_arrival_rate, "sat")
                    sat_queue = np.maximum(sat_queue, sat_value).astype(np.float32, copy=False)
                    sat_queue = np.minimum(sat_queue, float(cfg.queue_max_sat)).astype(np.float32, copy=False)

    return {
        "gu_queue": np.asarray(gu_queue, dtype=np.float32),
        "uav_queue": np.asarray(uav_queue, dtype=np.float32),
        "sat_queue": np.asarray(sat_queue, dtype=np.float32),
        "hotspot_subsets": [np.asarray(subset, dtype=np.int32) for subset in hotspot_subsets],
        "hotspot_member_mask": np.asarray(hotspot_member_mask, dtype=bool),
        "hotspot_active_idx": int(hotspot_active_idx),
        "last_hotspot_index": int(last_hotspot_index),
        "last_hotspot_mask": np.asarray(last_hotspot_mask, dtype=np.float32),
    }


def _build_native_reset_arrival_rate_vec(
    cfg,
    *,
    effective_task_arrival_rate: float,
    arrival_base_scale: np.ndarray,
    hotspot_mask: np.ndarray | None = None,
) -> np.ndarray:
    num_gu = int(cfg.num_gu)
    if num_gu <= 0:
        return np.zeros((0,), dtype=np.float32)
    traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
    if traffic_model != "sticky_subset_hotspot":
        return np.full((num_gu,), float(effective_task_arrival_rate), dtype=np.float32)
    base_scale = np.asarray(arrival_base_scale, dtype=np.float32)
    if base_scale.shape != (num_gu,):
        base_scale = np.ones((num_gu,), dtype=np.float32)
    weights = base_scale.astype(np.float32, copy=True)
    mask = np.zeros((num_gu,), dtype=bool) if hotspot_mask is None else np.asarray(hotspot_mask, dtype=bool).reshape(num_gu)
    rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
    if rho > 0.0 and bool(np.any(mask)):
        weights *= np.where(mask, rho, 1.0).astype(np.float32, copy=False)
    if bool(getattr(cfg, "arrival_mean_preserve", True)):
        mean_weight = float(np.mean(weights, dtype=np.float32))
        if mean_weight > NORMALIZATION_DENOM_EPS:
            weights = (weights / mean_weight).astype(np.float32, copy=False)
    return np.maximum(float(effective_task_arrival_rate) * weights, 0.0).astype(np.float32, copy=False)


def _compute_centroid_dist_mean(cfg, gu_pos: np.ndarray, gu_queue: np.ndarray, uav_pos: np.ndarray) -> float:
    if int(cfg.num_gu) <= 0:
        return 0.0
    gu_pos_arr = np.asarray(gu_pos, dtype=np.float32)
    gu_queue_arr = np.asarray(gu_queue, dtype=np.float32)
    q_weights = gu_queue_arr / normalize_scale(float(cfg.queue_max_gu))
    weight_sum = float(np.sum(q_weights, dtype=np.float32))
    if weight_sum <= NORMALIZATION_DENOM_EPS:
        weights = np.full((int(cfg.num_gu),), 1.0 / max(int(cfg.num_gu), 1), dtype=np.float32)
    else:
        weights = (q_weights / weight_sum).astype(np.float32, copy=False)
    centroid = np.sum(gu_pos_arr * weights[:, None], axis=0)
    dists = np.linalg.norm(np.asarray(uav_pos, dtype=np.float32) - centroid[None, :], axis=1)
    return float(np.mean(dists)) if dists.size > 0 else 0.0


def _doppler_precomp_enabled_from_cfg(cfg) -> bool:
    mode = str(getattr(cfg, "doppler_precomp_mode", "none") or "none").strip().lower()
    return mode in {"residual_hz", "residual_ppm"}


def _doppler_residual_cap_hz_from_cfg(cfg) -> float:
    mode = str(getattr(cfg, "doppler_precomp_mode", "none") or "none").strip().lower()
    if mode == "residual_hz":
        return max(float(getattr(cfg, "doppler_residual_hz", 0.0) or 0.0), 0.0)
    if mode == "residual_ppm":
        ppm = max(float(getattr(cfg, "doppler_residual_ppm", 0.0) or 0.0), 0.0)
        return _backhaul_carrier_freq_from_cfg(cfg) * ppm * 1.0e-6
    return 0.0


def _effective_b_backhaul_per_sat_from_cfg(cfg) -> float:
    legacy_bw = getattr(cfg, "b_sat_total", None)
    legacy_scale = getattr(cfg, "b_sat_total_scale", None)
    scale = max(
        float((legacy_scale if legacy_scale is not None else getattr(cfg, "b_backhaul_per_sat_scale", 1.0)) or 1.0),
        0.0,
    )
    bandwidth = float(legacy_bw if legacy_bw is not None else getattr(cfg, "b_backhaul_per_sat", 0.0))
    return bandwidth * scale


def _effective_b_sat_total_from_cfg(cfg) -> float:
    """Legacy alias for the per-SAT backhaul bandwidth pool."""
    return _effective_b_backhaul_per_sat_from_cfg(cfg)


def _effective_sat_cpu_freq_from_cfg(cfg) -> float:
    scale = max(float(getattr(cfg, "sat_cpu_freq_scale", 1.0) or 1.0), 0.0)
    return float(cfg.sat_cpu_freq) * scale


def _initial_doppler_residual_state(cfg, rng: np.random.Generator) -> np.ndarray:
    residual = np.zeros((int(cfg.num_uav), int(cfg.num_sat)), dtype=np.float32)
    if not _doppler_precomp_enabled_from_cfg(cfg):
        return residual
    cap = _doppler_residual_cap_hz_from_cfg(cfg)
    if cap <= 0.0:
        return residual
    sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
    sigma = min(sigma, cap)
    if sigma <= 0.0:
        return residual
    rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
    init_std = sigma / math.sqrt(max(1.0 - rho * rho, 1.0e-6))
    init = rng.normal(loc=0.0, scale=init_std, size=residual.shape)
    return np.clip(init, -cap, cap).astype(np.float32, copy=False)


def _initial_reward_parts_from_existing(
    existing: dict[str, Any] | None,
    *,
    arrival_ref: float,
    avoidance_eta_eff: float,
    last_avoidance_eta_exec: float,
    avoidance_collision_rate_ema: float,
    prev_episode_collision_rate: float,
) -> dict[str, Any]:
    reward_parts = copy.deepcopy(existing or {})
    for key, value in list(reward_parts.items()):
        if key == "queue_delta_mode":
            reward_parts[key] = "total"
        elif isinstance(value, (bool, int, float, np.integer, np.floating)):
            reward_parts[key] = 0.0
    reward_parts["arrival_ref"] = float(arrival_ref)
    reward_parts["queue_delta_mode"] = "total"
    reward_parts["avoidance_eta_eff"] = float(avoidance_eta_eff)
    reward_parts["avoidance_eta_exec"] = float(last_avoidance_eta_exec)
    reward_parts["avoidance_collision_rate_ema"] = float(avoidance_collision_rate_ema)
    reward_parts["avoidance_prev_episode_collision_rate"] = float(prev_episode_collision_rate)
    return reward_parts


def _apply_batched_bw_queue_transition_tensor_impl(
    *,
    uav_queue_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    gu_outflow_t: torch.Tensor,
    assoc_t: torch.Tensor,
    rate_matrix_t: torch.Tensor,
    tau0: float,
    queue_max_uav: float,
    queue_max_sat: float,
    sat_compute_rates_t: torch.Tensor,
    flow_bits_quantum: float = 32.0,
    queue_state_quantum: float = 128.0,
    fast_float32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_device = uav_queue_before_t.device
    batch_size, num_uav = uav_queue_before_t.shape
    work_dtype = torch.float32 if bool(fast_float32) else torch.float64
    uav_queue_before_work_t = uav_queue_before_t.to(dtype=work_dtype)
    sat_queue_before_work_t = sat_queue_before_t.to(dtype=work_dtype)
    gu_outflow_work_t = gu_outflow_t.to(dtype=work_dtype)
    rate_matrix_work_t = rate_matrix_t.to(dtype=work_dtype)
    sat_compute_rates_work_t = sat_compute_rates_t.to(dtype=work_dtype)
    valid_assoc_t = (assoc_t >= 0) & (assoc_t < int(num_uav))
    assoc_clamped_t = torch.clamp(assoc_t, min=0, max=max(int(num_uav) - 1, 0))
    assoc_one_hot_t = torch.nn.functional.one_hot(
        assoc_clamped_t,
        num_classes=int(num_uav),
    ).to(dtype=work_dtype)
    assoc_one_hot_t = assoc_one_hot_t * valid_assoc_t.unsqueeze(-1).to(dtype=work_dtype)
    inflow_t = (assoc_one_hot_t * gu_outflow_work_t.unsqueeze(-1)).sum(dim=1)

    q_uav_before_t = uav_queue_before_work_t + inflow_t
    total_rate_t = rate_matrix_work_t.sum(dim=-1)
    uav_service_t = _quantize_semantic_tensor(
        total_rate_t * float(tau0),
        quantum=flow_bits_quantum,
        out_dtype=work_dtype,
    )
    uav_outflow_t = torch.minimum(q_uav_before_t, uav_service_t)
    q_uav_after_raw_t = q_uav_before_t - uav_outflow_t
    uav_drop_t = torch.clamp(q_uav_after_raw_t - float(queue_max_uav), min=0.0)
    uav_queue_after_t = _quantize_semantic_tensor(
        torch.clamp(q_uav_after_raw_t, max=float(queue_max_uav)),
        quantum=float(queue_state_quantum),
        out_dtype=work_dtype,
    )

    safe_total_rate_t = torch.where(total_rate_t > 0.0, total_rate_t, total_rate_t * 0.0 + 1.0)
    outflow_matrix_t = (rate_matrix_work_t / safe_total_rate_t.unsqueeze(-1)) * uav_outflow_t.unsqueeze(-1)
    outflow_matrix_t = torch.where(total_rate_t.unsqueeze(-1) > 0.0, outflow_matrix_t, outflow_matrix_t * 0.0)

    sat_incoming_t = _quantize_semantic_tensor(
        outflow_matrix_t.sum(dim=1),
        quantum=float(flow_bits_quantum),
        out_dtype=work_dtype,
    )
    q_sat_before_t = sat_queue_before_work_t + sat_incoming_t
    sat_service_t = _quantize_semantic_tensor(
        sat_compute_rates_work_t.unsqueeze(-1) * float(tau0),
        quantum=flow_bits_quantum,
        out_dtype=work_dtype,
    )
    sat_processed_t = torch.minimum(q_sat_before_t, sat_service_t)
    q_sat_after_raw_t = q_sat_before_t - sat_processed_t
    sat_drop_t = torch.clamp(q_sat_after_raw_t - float(queue_max_sat), min=0.0)
    sat_queue_after_t = _quantize_semantic_tensor(
        torch.clamp(q_sat_after_raw_t, max=float(queue_max_sat)),
        quantum=float(queue_state_quantum),
        out_dtype=work_dtype,
    )
    return (
        uav_queue_after_t.to(dtype=torch.float32),
        uav_drop_t.to(dtype=torch.float32),
        uav_outflow_t.to(dtype=torch.float32),
        sat_queue_after_t.to(dtype=torch.float32),
        sat_drop_t.to(dtype=torch.float32),
        sat_incoming_t.to(dtype=torch.float32),
        sat_processed_t.to(dtype=torch.float32),
    )


def _apply_batched_gu_queue_transition_tensor_impl(
    *,
    gu_queue_before_t: torch.Tensor,
    arrivals_t: torch.Tensor,
    access_rates_t: torch.Tensor,
    prev_service_gap_t: torch.Tensor,
    prev_deadline_age_t: torch.Tensor,
    deadline_steps_t: torch.Tensor,
    base_arrival_steps_t: torch.Tensor,
    tau0: float,
    queue_max_gu: float,
    service_gap_increment: float,
    service_gap_relief_coef: float,
    service_gap_cap_steps: float,
    deadline_enabled: bool,
    deadline_age_increment: float,
    deadline_service_relief_coef: float,
    deadline_age_cap_steps: float,
    deadline_expire_rate: float,
    flow_bits_quantum: float = 32.0,
    queue_state_quantum: float = 128.0,
    fast_float32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    work_dtype = torch.float32 if bool(fast_float32) else torch.float64
    gu_queue_before_work_t = gu_queue_before_t.to(dtype=work_dtype)
    arrivals_work_t = arrivals_t.to(dtype=work_dtype)
    access_rates_work_t = access_rates_t.to(dtype=work_dtype)
    prev_service_gap_work_t = prev_service_gap_t.to(dtype=work_dtype)
    prev_deadline_age_work_t = prev_deadline_age_t.to(dtype=work_dtype)
    deadline_steps_work_t = deadline_steps_t.to(dtype=work_dtype)
    base_arrival_steps_work_t = base_arrival_steps_t.to(dtype=work_dtype)
    q_before_t = gu_queue_before_work_t + arrivals_work_t
    service_bits_t = _quantize_semantic_tensor(
        access_rates_work_t * float(tau0),
        quantum=flow_bits_quantum,
        out_dtype=work_dtype,
    )
    outflow_t = torch.minimum(q_before_t, service_bits_t)
    q_after_service_t = q_before_t - outflow_t

    backlogged_t = q_before_t > 1.0e-6
    service_relief_t = float(service_gap_relief_coef) * (
        _torch_ratio_or_zero(outflow_t, base_arrival_steps_work_t)
    )
    service_gap_t = prev_service_gap_work_t + float(service_gap_increment) * backlogged_t.to(work_dtype) - service_relief_t
    service_gap_t = torch.clamp(service_gap_t, min=0.0, max=float(service_gap_cap_steps))
    service_gap_t = torch.where(q_after_service_t <= 1.0e-6, service_gap_t * 0.0, service_gap_t)

    if deadline_enabled:
        service_frac_t = torch.where(
            backlogged_t,
            outflow_t / torch.clamp(q_before_t, min=1.0e-6),
            outflow_t * 0.0,
        )
        deadline_age_t = prev_deadline_age_work_t + float(deadline_age_increment) * backlogged_t.to(work_dtype)
        deadline_age_t = deadline_age_t - float(deadline_service_relief_coef) * service_frac_t
        deadline_age_t = torch.clamp(deadline_age_t, min=0.0, max=float(deadline_age_cap_steps))
        deadline_slack_t = deadline_steps_work_t - deadline_age_t
        overdue_t = torch.clamp(deadline_age_t - deadline_steps_work_t, min=0.0)
        expire_frac_t = torch.clamp(float(deadline_expire_rate) * overdue_t, min=0.0, max=1.0)
        gu_expired_t = expire_frac_t * q_after_service_t
        q_after_deadline_t = torch.clamp(q_after_service_t - gu_expired_t, min=0.0)
        empty_t = q_after_deadline_t <= 1.0e-6
        deadline_age_t = torch.where(empty_t, deadline_age_t * 0.0, deadline_age_t)
        deadline_slack_t = torch.where(empty_t, deadline_steps_work_t, deadline_slack_t)
        deadline_risk_t = torch.clamp(
            deadline_age_t / torch.clamp(deadline_steps_work_t, min=1.0e-6),
            min=0.0,
            max=2.0,
        )
    else:
        gu_expired_t = q_after_service_t * 0.0
        q_after_deadline_t = q_after_service_t
        deadline_age_t = q_after_service_t * 0.0
        deadline_slack_t = q_after_service_t * 0.0
        deadline_risk_t = q_after_service_t * 0.0

    overflow_drop_t = torch.clamp(q_after_deadline_t - float(queue_max_gu), min=0.0)
    gu_drop_t = overflow_drop_t + gu_expired_t
    gu_queue_after_t = _quantize_semantic_tensor(
        torch.clamp(q_after_deadline_t, max=float(queue_max_gu)),
        quantum=float(queue_state_quantum),
        out_dtype=work_dtype,
    )
    return (
        gu_queue_after_t.to(dtype=torch.float32),
        gu_drop_t.to(dtype=torch.float32),
        outflow_t.to(dtype=torch.float32),
        service_gap_t.to(dtype=torch.float32),
        deadline_age_t.to(dtype=torch.float32),
        deadline_slack_t.to(dtype=torch.float32),
        deadline_risk_t.to(dtype=torch.float32),
        gu_expired_t.to(dtype=torch.float32),
    )


def _current_task_arrival_rates_batch(
    envs: Sequence[SaginParallelEnv],
    base_rates: Sequence[float],
) -> np.ndarray:
    if not envs:
        return np.zeros((0, 0), dtype=np.float32)
    cfg = envs[0].cfg
    batch_size = len(envs)
    if cfg.num_gu <= 0:
        return np.zeros((batch_size, 0), dtype=np.float32)

    base_rates_arr = np.maximum(np.asarray(base_rates, dtype=np.float32).reshape(batch_size), 0.0)
    traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
    if traffic_model != "sticky_subset_hotspot":
        return np.repeat(base_rates_arr[:, None], cfg.num_gu, axis=1).astype(np.float32, copy=False)

    preserve_mean = bool(getattr(cfg, "arrival_mean_preserve", True))
    hotspot_rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
    rates = np.zeros((batch_size, cfg.num_gu), dtype=np.float32)
    default_scale = np.ones((cfg.num_gu,), dtype=np.float32)
    for env_index, env in enumerate(envs):
        base_scale = np.asarray(
            getattr(env, "_arrival_base_scale", default_scale.astype(np.float32)),
            dtype=np.float32,
        )
        if base_scale.shape != (cfg.num_gu,):
            base_scale = default_scale
        weights = base_scale.copy()
        hotspot_idx = int(getattr(env, "_hotspot_active_idx", -1))
        hotspot_mask = np.asarray(
            getattr(env, "_hotspot_member_mask", np.zeros((0, cfg.num_gu), dtype=bool))
        )
        if hotspot_rho > 0.0 and 0 <= hotspot_idx < hotspot_mask.shape[0]:
            weights *= np.where(hotspot_mask[hotspot_idx], hotspot_rho, 1.0)
        if preserve_mean:
            mean_weight = float(np.mean(weights))
            if mean_weight > NORMALIZATION_DENOM_EPS:
                weights /= mean_weight
        rates[env_index] = np.maximum(base_rates_arr[env_index] * weights, 0.0).astype(np.float32, copy=False)
    return rates


def _associate_users_batch(
    envs: Sequence[SaginParallelEnv],
    *,
    gu_pos_batch: np.ndarray | None = None,
    uav_pos_batch: np.ndarray | None = None,
) -> list[np.ndarray]:
    if not envs:
        return []
    cfg = envs[0].cfg
    if cfg.num_gu <= 0:
        return [np.full((0,), -1, dtype=np.int32) for _ in envs]
    gu_pos = (
        np.asarray(gu_pos_batch, dtype=np.float32)
        if gu_pos_batch is not None
        else np.stack([np.asarray(env.gu_pos, dtype=np.float32) for env in envs], axis=0)
    )
    uav_pos = (
        np.asarray(uav_pos_batch, dtype=np.float32)
        if uav_pos_batch is not None
        else np.stack([np.asarray(env.uav_pos, dtype=np.float32) for env in envs], axis=0)
    )
    diff = gu_pos[:, :, None, :] - uav_pos[:, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    best = np.argmin(dist2, axis=2)
    assoc_batch = best.astype(np.int32, copy=False)
    return [np.asarray(assoc_batch[idx], dtype=np.int32).copy() for idx in range(len(envs))]


def _associate_users_and_access_base_gain_tensor_impl(
    *,
    channel_params: _NativeChannelStaticParams,
    candidate_params: _NativeCandidateStaticParams,
    gu_pos_t: torch.Tensor,
    uav_pos_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_p = candidate_params
    if int(candidate_p.num_gu) <= 0:
        raise RuntimeError("strict association segment requires at least one GU.")
    work_dtype = torch.float32 if bool(candidate_p.fast_float32) else torch.float64
    gu_pos_work_t = gu_pos_t.to(dtype=work_dtype)
    uav_pos_work_t = uav_pos_t.to(dtype=work_dtype)
    diff_t = gu_pos_work_t[:, :, None, :] - uav_pos_work_t[:, None, :, :]
    d2_t = torch.sum(diff_t * diff_t, dim=-1)
    d2d_t = torch.sqrt(d2_t)
    height_value = float(candidate_p.uav_height)
    d3d_t = torch.sqrt(d2d_t * d2d_t + height_value * height_value)
    phi_t = torch.asin(torch.clamp(height_value / _torch_positive(d3d_t, GEOMETRY_DENOM_EPS), -1.0, 1.0))
    pl_t = _pathloss_db_torch(d_t=d3d_t, phi_rad_t=phi_t, channel_params=channel_params)
    best_t = torch.argmin(d2_t, dim=2).to(dtype=torch.long)
    assoc_t = best_t.to(dtype=torch.int32)
    pathloss_quant_t = _quantize_semantic_tensor(
        pl_t,
        quantum=float(channel_params.access_pathloss_db_quantum),
        out_dtype=work_dtype,
    )
    base_gain_t = torch.exp((-pathloss_quant_t / 10.0) * math.log(10.0)).to(dtype=torch.float32)
    return assoc_t, base_gain_t


def _build_candidate_users_batch(
    envs: Sequence[SaginParallelEnv],
    assoc_batch: Sequence[np.ndarray],
    *,
    gu_queue_batch: np.ndarray | None = None,
    gu_pos_batch: np.ndarray | None = None,
    uav_pos_batch: np.ndarray | None = None,
) -> list[list[list[int]]]:
    if not envs:
        return []
    cfg = envs[0].cfg
    raw_mode = getattr(cfg, "candidate_mode", None)
    if raw_mode is None:
        raw_mode = getattr(cfg, "candidate_users_mode", "assoc")
    mode = str(raw_mode or "assoc").strip().lower()
    max_keep = int(getattr(cfg, "candidate_k", 0) or 0)
    if max_keep <= 0:
        max_keep = int(cfg.users_obs_max)
    else:
        max_keep = min(max_keep, int(cfg.users_obs_max))
    candidate_groups: list[list[list[int]]] = []
    if max_keep <= 0:
        return [[[] for _ in range(cfg.num_uav)] for _ in envs]

    if mode in {"associated", "assoc"}:
        if gu_queue_batch is None:
            gu_queue_batch = np.stack([np.asarray(env.gu_queue, dtype=np.float32) for env in envs], axis=0)
        else:
            gu_queue_batch = np.asarray(gu_queue_batch, dtype=np.float32)
        for env_index, assoc in enumerate(assoc_batch):
            assoc_arr = np.asarray(assoc, dtype=np.int32)
            candidates: list[list[int]] = [[] for _ in range(cfg.num_uav)]
            for uav_idx in range(cfg.num_uav):
                gu_idx = np.flatnonzero(assoc_arr == uav_idx).astype(np.int64, copy=False)
                if gu_idx.size > max_keep:
                    order = np.argsort(-np.asarray(gu_queue_batch[env_index, gu_idx], dtype=np.float32), kind="stable")
                    gu_idx = gu_idx[order[:max_keep]]
                candidates[uav_idx] = gu_idx.astype(np.int64, copy=False).tolist()
            candidate_groups.append(candidates)
        return candidate_groups

    if mode in {"associated_queue_topk", "assoc_queue_topk"}:
        if gu_queue_batch is None:
            gu_queue_batch = np.stack([np.asarray(env.gu_queue, dtype=np.float32) for env in envs], axis=0)
        else:
            gu_queue_batch = np.asarray(gu_queue_batch, dtype=np.float32)
        for env_index, (env, assoc) in enumerate(zip(envs, assoc_batch)):
            assoc_arr = np.asarray(assoc, dtype=np.int32)
            candidates: list[list[int]] = [[] for _ in range(cfg.num_uav)]
            for uav_idx in range(cfg.num_uav):
                cand_arr = np.flatnonzero(assoc_arr == uav_idx).astype(np.int64, copy=False)
                if cand_arr.size > 0:
                    order = np.argsort(-np.asarray(gu_queue_batch[env_index, cand_arr], dtype=np.float32), kind="stable")
                    candidates[uav_idx] = cand_arr[order[:max_keep]].astype(np.int64, copy=False).tolist()
            candidate_groups.append(candidates)
        return candidate_groups

    if cfg.num_gu <= 0:
        return [[[] for _ in range(cfg.num_uav)] for _ in envs]

    use_radius = mode in {"radius", "dist", "distance"}
    radius = getattr(cfg, "candidate_radius", None)
    gu_pos = (
        np.asarray(gu_pos_batch, dtype=np.float32)
        if gu_pos_batch is not None
        else np.stack([np.asarray(env.gu_pos, dtype=np.float32) for env in envs], axis=0)
    )
    uav_pos = (
        np.asarray(uav_pos_batch, dtype=np.float32)
        if uav_pos_batch is not None
        else np.stack([np.asarray(env.uav_pos, dtype=np.float32) for env in envs], axis=0)
    )
    d2d = np.linalg.norm(gu_pos[:, None, :, :] - uav_pos[:, :, None, :], axis=-1)
    for env_index in range(len(envs)):
        env_candidates: list[list[int]] = [[] for _ in range(cfg.num_uav)]
        for uav_idx in range(cfg.num_uav):
            d2d_row = np.asarray(d2d[env_index, uav_idx], dtype=np.float32)
            ordered = np.argsort(d2d_row).astype(np.int64, copy=False)
            if use_radius and radius is not None and float(radius) > 0.0:
                within = ordered[np.asarray(d2d_row[ordered], dtype=np.float32) <= float(radius)]
                ordered = within if int(within.size) > 0 else ordered
            env_candidates[uav_idx] = ordered[:max_keep].astype(np.int64, copy=False).tolist()
        candidate_groups.append(env_candidates)
    return candidate_groups


def _build_candidate_index_mask_tensor_impl(
    *,
    candidate_params: _NativeCandidateStaticParams,
    assoc_t: torch.Tensor,
    gu_queue_t: torch.Tensor,
    gu_pos_t: torch.Tensor,
    uav_pos_t: torch.Tensor,
    candidate_slot_ids_t: torch.Tensor,
    candidate_gu_ids_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
    out_idx: torch.Tensor,
    out_mask: torch.Tensor,
) -> None:
    candidate_p = candidate_params
    batch_size = int(assoc_t.shape[0])
    num_uav = int(candidate_p.num_uav)
    num_gu = int(candidate_p.num_gu)
    users_obs_max = int(candidate_p.users_obs_max)
    expected_shape = (batch_size, num_uav, users_obs_max)
    if tuple(out_idx.shape) != expected_shape or out_idx.dtype != torch.long or out_idx.device != assoc_t.device:
        raise RuntimeError("candidate index output buffer violates the fixed tensor ABI.")
    if tuple(out_mask.shape) != expected_shape or out_mask.dtype != torch.bool or out_mask.device != assoc_t.device:
        raise RuntimeError("candidate mask output buffer violates the fixed tensor ABI.")
    out_idx.fill_(-1)
    out_mask.zero_()
    if batch_size <= 0 or num_uav <= 0 or num_gu <= 0 or users_obs_max <= 0:
        return None

    mode = str(candidate_p.candidate_mode)
    keep = int(candidate_p.candidate_k)
    keep = min(int(keep), num_gu)
    if keep <= 0:
        return None

    slot_ids_t = candidate_slot_ids_t[:, :, :keep].to(device=assoc_t.device, dtype=torch.long)
    gu_ids_t = candidate_gu_ids_t[:num_gu].to(device=assoc_t.device, dtype=torch.long)
    uav_ids_t = candidate_uav_ids_t[:, :num_uav, :].to(device=assoc_t.device, dtype=torch.long)
    assoc_by_uav_t = assoc_t.to(dtype=torch.long).unsqueeze(1) == uav_ids_t

    if mode in {"associated", "assoc", "associated_queue_topk", "assoc_queue_topk"}:
        assoc_count_t = assoc_by_uav_t.sum(dim=2)
        asc_key_t = torch.where(
            assoc_by_uav_t,
            gu_ids_t.view(1, 1, num_gu).to(dtype=torch.float32),
            float("inf"),
        )
        asc_idx_t = torch.argsort(asc_key_t, dim=2, stable=True)[:, :, :keep]
        queue_key_t = torch.where(
            assoc_by_uav_t,
            -gu_queue_t.to(dtype=torch.float32).unsqueeze(1),
            float("inf"),
        )
        queue_idx_t = torch.argsort(queue_key_t, dim=2, stable=True)[:, :, :keep]
        if mode in {"associated_queue_topk", "assoc_queue_topk"}:
            selected_t = queue_idx_t
        else:
            selected_t = torch.where((assoc_count_t > keep).unsqueeze(-1), queue_idx_t, asc_idx_t)
        valid_count_t = torch.clamp(assoc_count_t, max=keep)
        valid_mask_t = slot_ids_t < valid_count_t.unsqueeze(-1)
    else:
        d2d_t = torch.linalg.vector_norm(
            gu_pos_t[:, None, :, :].to(dtype=torch.float32)
            - uav_pos_t[:, :, None, :].to(dtype=torch.float32),
            dim=-1,
        )
        all_idx_t = torch.argsort(d2d_t, dim=2, stable=True)[:, :, :keep]
        if mode in {"radius", "dist", "distance"}:
            radius = candidate_p.candidate_radius
            if radius is not None and float(radius) > 0.0:
                within_t = d2d_t <= float(radius)
                masked_key_t = torch.where(within_t, d2d_t, float("inf"))
                radius_idx_t = torch.argsort(masked_key_t, dim=2, stable=True)[:, :, :keep]
                within_count_t = within_t.sum(dim=2)
                has_within_t = within_count_t > 0
                selected_t = torch.where(has_within_t.unsqueeze(-1), radius_idx_t, all_idx_t)
                valid_count_t = torch.where(
                    has_within_t,
                    torch.clamp(within_count_t, max=keep),
                    within_count_t * 0 + keep,
                )
            else:
                selected_t = all_idx_t
                valid_count_t = assoc_t[:, :num_uav].to(dtype=torch.long) * 0 + keep
        else:
            selected_t = all_idx_t
            valid_count_t = assoc_t[:, :num_uav].to(dtype=torch.long) * 0 + keep
        valid_mask_t = slot_ids_t < valid_count_t.unsqueeze(-1)

    out_idx[:, :, :keep] = torch.where(valid_mask_t, selected_t.to(dtype=torch.long), selected_t.to(dtype=torch.long) * 0 - 1)
    out_mask[:, :, :keep] = valid_mask_t
    return None


def _build_bw_valid_mask_batch(
    cfg,
    assoc_batch: Sequence[np.ndarray],
    candidate_groups: Sequence[Sequence[Sequence[int]]],
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for assoc, candidates in zip(assoc_batch, candidate_groups):
        assoc_arr = np.asarray(assoc, dtype=np.int32)
        mask = np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        for uav_idx in range(cfg.num_uav):
            cand = list(candidates[uav_idx])[: cfg.users_obs_max]
            if not cand:
                continue
            cand_idx = np.asarray(cand, dtype=np.int32)
            assoc_mask = (assoc_arr[cand_idx] == uav_idx).astype(np.float32, copy=False)
            mask[uav_idx, : assoc_mask.shape[0]] = assoc_mask
        masks.append(mask)
    return masks


def _candidate_flag_bundle_from_index_mask_tensor_impl(
    *,
    candidate_params: _NativeCandidateStaticParams,
    assoc_t: torch.Tensor,
    candidate_indices_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    prev_association_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
    bw_valid_slots_out: torch.Tensor,
    candidate_flag_out: torch.Tensor,
    bw_valid_flag_out: torch.Tensor,
    prev_assoc_flag_out: torch.Tensor,
) -> None:
    candidate_p = candidate_params
    batch_size = int(candidate_indices_t.shape[0])
    num_uav = int(candidate_p.num_uav)
    num_gu = int(candidate_p.num_gu)
    expected_flag_shape = (batch_size, num_uav, num_gu)
    if tuple(candidate_flag_out.shape) != expected_flag_shape or candidate_flag_out.device != candidate_indices_t.device:
        raise RuntimeError("candidate flag output buffer violates the fixed tensor ABI.")
    if batch_size <= 0 or num_uav <= 0 or num_gu <= 0:
        bw_valid_slots_out.zero_()
        candidate_flag_out.zero_()
        bw_valid_flag_out.zero_()
        prev_assoc_flag_out.zero_()
        return None
    safe_idx_t = torch.clamp(candidate_indices_t.to(dtype=torch.long), min=0, max=max(num_gu - 1, 0))
    slot_mask_t = candidate_mask_t.to(dtype=torch.bool) & (candidate_indices_t >= 0)
    candidate_flag_out.zero_()
    candidate_flag_out.scatter_(2, safe_idx_t, slot_mask_t.to(dtype=torch.float32))
    gathered_assoc_t = torch.gather(
        assoc_t.to(dtype=torch.long).unsqueeze(1).expand(-1, num_uav, -1),
        2,
        safe_idx_t,
    )
    uav_ids_t = candidate_uav_ids_t[:, :num_uav, :].to(device=candidate_indices_t.device, dtype=torch.long)
    bw_valid_slots_t = (slot_mask_t & (gathered_assoc_t == uav_ids_t)).to(dtype=torch.float32)
    bw_valid_slots_out.copy_(bw_valid_slots_t.to(dtype=bw_valid_slots_out.dtype))
    bw_valid_flag_out.zero_()
    bw_valid_flag_out.scatter_(2, safe_idx_t, bw_valid_slots_t)
    prev_assoc_flag_out.copy_(
        (
        prev_association_t.to(dtype=torch.long).unsqueeze(1)
        == uav_ids_t
        ).to(dtype=prev_assoc_flag_out.dtype)
    )
    return None


def _candidate_flag_bundle_from_candidates(
    cfg,
    candidates: Sequence[Sequence[int]],
    bw_valid_mask: np.ndarray | None,
    prev_association: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_flag = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
    bw_valid_flag = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
    bw_valid_slots = (
        np.zeros((cfg.num_uav, cfg.users_obs_max), dtype=np.float32)
        if bw_valid_mask is None
        else np.asarray(bw_valid_mask, dtype=np.float32)
    )
    for uav_idx in range(cfg.num_uav):
        cand = list(candidates[uav_idx])[: cfg.users_obs_max]
        for slot, gu_idx in enumerate(cand):
            gu_id = int(gu_idx)
            if 0 <= gu_id < cfg.num_gu:
                candidate_flag[uav_idx, gu_id] = 1.0
                if bw_valid_slots[uav_idx, slot] > 0.5:
                    bw_valid_flag[uav_idx, gu_id] = 1.0
    prev_assoc_flag = (
        np.asarray(prev_association, dtype=np.int64)[None, :]
        == np.arange(cfg.num_uav, dtype=np.int64)[:, None]
    ).astype(np.float32, copy=False)
    return candidate_flag, bw_valid_flag, prev_assoc_flag


def _candidate_index_mask_batch(
    cfg,
    candidate_groups: Sequence[Sequence[Sequence[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_indices = np.full(
        (len(candidate_groups), cfg.num_uav, cfg.users_obs_max),
        -1,
        dtype=np.int64,
    )
    candidate_mask = np.zeros(
        (len(candidate_groups), cfg.num_uav, cfg.users_obs_max),
        dtype=bool,
    )
    for env_index, candidates in enumerate(candidate_groups):
        for uav_idx in range(cfg.num_uav):
            cand = list(candidates[uav_idx])[: cfg.users_obs_max]
            if not cand:
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            candidate_indices[env_index, uav_idx, : cand_arr.size] = cand_arr
            candidate_mask[env_index, uav_idx, : cand_arr.size] = True
    return candidate_indices, candidate_mask


def _sat_selection_matrix_from_values(
    cfg,
    selections: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    select_k = _sat_action_select_k_from_config(cfg)

    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        matrix_arr = np.asarray(matrix, dtype=np.int64)
        out = np.full_like(matrix_arr, -1)
        for row_index in range(int(matrix_arr.shape[0])):
            seen: set[int] = set()
            write_index = 0
            for sat_idx in matrix_arr[row_index].tolist():
                sat_value = int(sat_idx)
                if sat_value < 0 or sat_value in seen:
                    continue
                out[row_index, write_index] = sat_value
                seen.add(sat_value)
                write_index += 1
                if write_index >= int(matrix_arr.shape[1]):
                    break
        return out

    if isinstance(selections, np.ndarray):
        matrix = np.asarray(selections, dtype=np.int64)
        if matrix.ndim == 1:
            if int(cfg.num_uav) != 1:
                raise ValueError(
                    f"1D sat selection matrix is only valid when num_uav=1, got num_uav={cfg.num_uav}."
                )
            matrix = matrix.reshape(1, -1)
        if int(matrix.shape[0]) != int(cfg.num_uav):
            raise ValueError(
                f"Sat selection matrix first dim must be {cfg.num_uav}, got {tuple(matrix.shape)}."
            )
        matrix_out = np.full((int(cfg.num_uav), int(select_k)), -1, dtype=np.int64)
        copy_cols = min(int(select_k), int(matrix.shape[1]))
        if copy_cols > 0:
            matrix_out[:, :copy_cols] = matrix[:, :copy_cols]
        return _normalize_rows(matrix_out)

    matrix = np.full((int(cfg.num_uav), select_k), -1, dtype=np.int64)
    for uav_index, selected in enumerate(selections):
        if uav_index >= int(cfg.num_uav):
            break
        sat_idx = np.asarray(selected, dtype=np.int64).reshape(-1)
        if sat_idx.size <= 0:
            continue
        fill = min(int(sat_idx.size), int(select_k))
        matrix[uav_index, :fill] = sat_idx[:fill]
    return _normalize_rows(matrix)


def _sat_selection_lists_from_matrix(selection_matrix: np.ndarray) -> list[list[int]]:
    matrix = np.asarray(selection_matrix, dtype=np.int64)
    return [
        [int(sat_idx) for sat_idx in matrix[uav_index][matrix[uav_index] >= 0].tolist()]
        for uav_index in range(int(matrix.shape[0]))
    ]


def _active_sat_ids_from_visible_and_selection(
    visible: Sequence[Sequence[int]],
    sat_selection_matrix: np.ndarray | None,
) -> np.ndarray:
    active: list[int] = []
    for visible_ids in visible:
        active.extend(int(sat_idx) for sat_idx in visible_ids)
    if sat_selection_matrix is not None:
        sat_selection_arr = np.asarray(sat_selection_matrix, dtype=np.int64)
        active.extend(int(sat_idx) for sat_idx in sat_selection_arr[sat_selection_arr >= 0].tolist())
    if not active:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(sorted(set(active)), dtype=np.int32)


def _compute_uav_cache_arrays(
    cfg,
    uav_pos_batch: np.ndarray,
    uav_vel_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uav_pos = np.asarray(uav_pos_batch, dtype=np.float32)
    uav_vel = np.asarray(uav_vel_batch, dtype=np.float32)
    num_envs = int(uav_pos.shape[0])
    num_uav = int(uav_pos.shape[1]) if uav_pos.ndim >= 2 else 0
    if num_uav <= 0:
        empty = np.zeros((num_envs, 0, 3), dtype=np.float32)
        return empty, empty

    lat0 = math.radians(float(cfg.ref_lat_deg))
    lon0 = math.radians(float(cfg.ref_lon_deg))
    cos_lat0 = math.cos(lat0)
    denom_lon = normalize_scale(float(cfg.r_earth) * cos_lat0)

    lat = lat0 + uav_pos[:, :, 1] / float(cfg.r_earth)
    lon = lon0 + uav_pos[:, :, 0] / denom_lon
    r = float(cfg.r_earth + cfg.uav_height)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    uav_ecef = np.zeros((num_envs, num_uav, 3), dtype=np.float32)
    uav_ecef[:, :, 0] = r * cos_lat * cos_lon
    uav_ecef[:, :, 1] = r * cos_lat * sin_lon
    uav_ecef[:, :, 2] = r * sin_lat

    east = np.asarray(uav_vel[:, :, 0], dtype=np.float32)
    north = np.asarray(uav_vel[:, :, 1], dtype=np.float32)
    up = np.zeros_like(east, dtype=np.float32)
    uav_vel_ecef = np.zeros((num_envs, num_uav, 3), dtype=np.float32)
    uav_vel_ecef[:, :, 0] = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    uav_vel_ecef[:, :, 1] = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    uav_vel_ecef[:, :, 2] = cos_lat * north + sin_lat * up
    return uav_ecef, uav_vel_ecef


def _refresh_uav_cache_batch(
    envs: Sequence[SaginParallelEnv],
    *,
    uav_pos_batch: torch.Tensor | np.ndarray | None = None,
    uav_vel_batch: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not envs:
        empty = np.zeros((0, 0, 3), dtype=np.float32)
        return empty, empty
    cfg = envs[0].cfg
    num_envs = len(envs)
    num_uav = int(cfg.num_uav)
    if num_uav <= 0:
        for env in envs:
            env._cached_uav_ecef = np.zeros((0, 3), dtype=np.float32)
            env._cached_uav_vel_ecef = np.zeros((0, 3), dtype=np.float32)
            env._cached_elevation_t = None
            env._cached_elevation_matrix = None
            env._cached_backhaul_loss_t = None
            env._cached_backhaul_loss_matrix = None
            env._cached_uav_neighbor_t = None
            env._cached_uav_neighbor_order = None
        empty = np.zeros((num_envs, 0, 3), dtype=np.float32)
        return empty, empty

    uav_pos = (
        _compat_numpy_array(uav_pos_batch, dtype=np.float32)
        if uav_pos_batch is not None
        else np.stack([np.asarray(env.uav_pos, dtype=np.float32) for env in envs], axis=0)
    )
    uav_vel = (
        _compat_numpy_array(uav_vel_batch, dtype=np.float32)
        if uav_vel_batch is not None
        else np.stack([np.asarray(env.uav_vel, dtype=np.float32) for env in envs], axis=0)
    )
    uav_ecef, uav_vel_ecef = _compute_uav_cache_arrays(cfg, uav_pos, uav_vel)

    for env_index, env in enumerate(envs):
        env._cached_uav_ecef = np.asarray(uav_ecef[env_index], dtype=np.float32).copy()
        env._cached_uav_vel_ecef = np.asarray(uav_vel_ecef[env_index], dtype=np.float32).copy()
        env._cached_elevation_t = None
        env._cached_elevation_matrix = None
        env._cached_backhaul_loss_t = None
        env._cached_backhaul_loss_matrix = None
        env._cached_uav_neighbor_t = None
        env._cached_uav_neighbor_order = None
        env._cached_obs_runtime_context = None
    return uav_ecef, uav_vel_ecef


def _can_use_simple_accel_batch(cfg) -> bool:
    return bool(
        str(getattr(cfg, "boundary_mode", "clip") or "clip").strip().lower() == "clip"
        and not bool(getattr(cfg, "boundary_hard_filter_enabled", False))
        and not bool(getattr(cfg, "pairwise_hard_filter_enabled", False))
        and not ablation_flag(cfg, "use_avoidance_layer", fallback_attr="avoidance_enabled", default=False)
        and not ablation_flag(cfg, "use_energy_safety_layer", fallback_attr="energy_safety_enabled", default=False)
    )


def _apply_uav_dynamics_batch_simple(
    envs: Sequence[SaginParallelEnv],
    accel_actions: Sequence[np.ndarray],
) -> None:
    if not envs:
        return
    cfg = envs[0].cfg
    action_batch = np.stack([np.asarray(action, dtype=np.float32) for action in accel_actions], axis=0)
    if action_batch.shape != (len(envs), cfg.num_uav, 2):
        raise ValueError(f"accel action batch shape must be ({len(envs)}, {cfg.num_uav}, 2)")

    exec_accel = _project_l2_ball_np(np.clip(action_batch, -1.0, 1.0), 1.0) * float(cfg.a_max)
    uav_vel_batch = np.stack([np.asarray(env.uav_vel, dtype=np.float32) for env in envs], axis=0)
    uav_pos_batch = np.stack([np.asarray(env.uav_pos, dtype=np.float32) for env in envs], axis=0)
    uav_vel_batch = _project_l2_ball_np(uav_vel_batch + exec_accel * float(cfg.tau0), cfg.v_max)
    uav_pos_batch = np.clip(
        uav_pos_batch + uav_vel_batch * float(cfg.tau0),
        0.0,
        float(cfg.map_size),
    ).astype(np.float32, copy=False)

    eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
    eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
    eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
    eta_max = max(eta_min, eta_max)
    cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
    avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)

    for env_index, env in enumerate(envs):
        eta_avoid = float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta))
        if cross_enabled:
            _, _, transfer_ratio = env._centroid_anneal_state()
            eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * transfer_ratio)
        eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))
        env.last_avoidance_eta_exec = float(eta_avoid)
        env.last_policy_accel = np.asarray(exec_accel[env_index], dtype=np.float32).copy()
        env.last_exec_accel = np.asarray(exec_accel[env_index], dtype=np.float32).copy()
        env.last_filter_active_ratio = 0.0
        env.last_projected_delta_norm_mean = 0.0
        env.last_fallback_count = 0.0
        env.last_boundary_filter_count = 0.0
        env.last_pairwise_filter_count = 0.0
        env.last_pairwise_filter_active_ratio = 0.0
        env.last_pairwise_projected_delta_norm = 0.0
        env.last_pairwise_fallback_count = 0.0
        env.last_pairwise_candidate_infeasible_count = 0.0
        env.uav_vel = np.asarray(uav_vel_batch[env_index], dtype=np.float32).copy()
        env.uav_pos = np.asarray(uav_pos_batch[env_index], dtype=np.float32).copy()


def _can_use_native_accel_batch(cfg) -> bool:
    return str(getattr(cfg, "boundary_mode", "clip") or "clip").strip().lower() in {"clip", "reflect"}


def _predict_next_from_accel_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accel_arr = _project_l2_ball_np(accel, cfg.a_max)
    vel_next = _project_l2_ball_np(np.asarray(uav_vel, dtype=np.float32) + accel_arr * float(cfg.tau0), cfg.v_max)
    pos_next = np.asarray(uav_pos, dtype=np.float32) + vel_next * float(cfg.tau0)
    return pos_next.astype(np.float32, copy=False), vel_next.astype(np.float32, copy=False)


def _project_axis_to_boundary_from_state(
    cfg,
    pos: float,
    vel: float,
    accel_cmd: float,
    lower: float,
    upper: float,
) -> tuple[float, bool, bool]:
    tau = max(float(cfg.tau0), 1.0e-6)
    a_max = float(cfg.a_max)
    v_max = float(cfg.v_max)
    accel_cmd = float(np.clip(accel_cmd, -a_max, a_max))
    vel_cmd = float(np.clip(vel + accel_cmd * tau, -v_max, v_max))
    pos_cmd = float(pos + vel_cmd * tau)
    if lower <= pos_cmd <= upper:
        return accel_cmd, False, False

    vel_low = max(-v_max, (lower - pos) / tau)
    vel_high = min(v_max, (upper - pos) / tau)
    if vel_low <= vel_high:
        target_vel = float(np.clip(vel_cmd, vel_low, vel_high))
        accel_proj = float(np.clip((target_vel - vel) / tau, -a_max, a_max))
        vel_next = float(np.clip(vel + accel_proj * tau, -v_max, v_max))
        pos_next = float(pos + vel_next * tau)
        if lower <= pos_next <= upper:
            return accel_proj, True, False

    if pos_cmd < lower or pos < lower:
        accel_fallback = a_max
    elif pos_cmd > upper or pos > upper:
        accel_fallback = -a_max
    else:
        center = 0.5 * (lower + upper)
        accel_fallback = a_max if pos < center else -a_max
    return float(np.clip(accel_fallback, -a_max, a_max)), True, True


def _apply_boundary_hard_filter_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    zero_stats = {
        "filter_active_ratio": 0.0,
        "projected_delta_norm_mean": 0.0,
        "fallback_count": 0.0,
        "boundary_filter_count": 0.0,
        "pairwise_filter_count": 0.0,
    }
    if not bool(getattr(cfg, "boundary_hard_filter_enabled", False)):
        return np.asarray(accel, dtype=np.float32).copy(), zero_stats

    margin = max(float(getattr(cfg, "boundary_margin", 0.0) or 0.0), 0.0)
    lower = margin
    upper = float(cfg.map_size) - margin
    accel_safe = np.asarray(accel, dtype=np.float32).copy()
    pos_arr = np.asarray(uav_pos, dtype=np.float32)
    vel_arr = np.asarray(uav_vel, dtype=np.float32)
    target_indices = list(range(int(cfg.num_uav))) if indices is None else [int(idx) for idx in indices]
    delta_norms = np.zeros((len(target_indices),), dtype=np.float32)
    boundary_filter_count = 0
    fallback_count = 0
    for offset, i in enumerate(target_indices):
        accel_before = accel_safe[i].copy()
        adjusted = False
        fallback_used = False
        for axis in range(2):
            accel_axis, axis_adjusted, axis_fallback = _project_axis_to_boundary_from_state(
                cfg,
                float(pos_arr[i, axis]),
                float(vel_arr[i, axis]),
                float(accel_safe[i, axis]),
                lower,
                upper,
            )
            accel_safe[i, axis] = accel_axis
            adjusted = adjusted or axis_adjusted
            fallback_used = fallback_used or axis_fallback
        accel_safe[i] = _project_l2_ball_np(accel_safe[i], cfg.a_max)
        delta_norms[offset] = float(np.linalg.norm(accel_safe[i] - accel_before))
        if adjusted:
            boundary_filter_count += 1
        if fallback_used:
            fallback_count += 1
    return accel_safe, {
        "filter_active_ratio": float(boundary_filter_count) / float(max(len(target_indices), 1)),
        "projected_delta_norm_mean": float(np.mean(delta_norms)) if delta_norms.size else 0.0,
        "fallback_count": float(fallback_count),
        "boundary_filter_count": float(boundary_filter_count),
        "pairwise_filter_count": 0.0,
    }


def _pairwise_hard_distance_from_cfg(cfg) -> float:
    raw_distance = getattr(cfg, "pairwise_hard_distance", None)
    if raw_distance is None:
        return float(cfg.d_safe + 5.0)
    return max(float(raw_distance), float(cfg.d_safe))


def _pairwise_trigger_mode_from_cfg(cfg) -> str:
    mode = str(getattr(cfg, "pairwise_hard_trigger_mode", "distance") or "distance").strip().lower()
    return mode if mode in {"distance", "ttc"} else "distance"


def _pairwise_trigger_ttc_from_cfg(cfg) -> float:
    return max(float(getattr(cfg, "pairwise_hard_trigger_ttc", 2.0) or 0.0), 0.0)


def _pairwise_trigger_distance_from_cfg(cfg, d_hard: float) -> float:
    raw_distance = getattr(cfg, "pairwise_hard_trigger_distance", None)
    if raw_distance is not None:
        return max(float(raw_distance), d_hard)
    if _pairwise_trigger_mode_from_cfg(cfg) == "ttc":
        return max(d_hard, d_hard + 2.0 * float(cfg.v_max) * _pairwise_trigger_ttc_from_cfg(cfg))
    return d_hard


def _pairwise_closing_speed_threshold_from_cfg(cfg) -> float:
    return max(float(getattr(cfg, "pairwise_hard_closing_speed", 0.0) or 0.0), 0.0)


def _pairwise_correction_direction_from_state(
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    pair_diff_next: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    direction = np.asarray(pair_diff_next, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-6:
        return direction / norm
    direction = np.asarray(uav_pos[i] - uav_pos[j], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-6:
        return direction / norm
    direction = np.asarray(uav_vel[i] - uav_vel[j], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-6:
        return direction / norm
    return np.asarray([1.0, 0.0], dtype=np.float32)


def _evaluate_pairwise_ttc_resolution_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
    i: int,
    j: int,
    d_hard: float,
    direction: np.ndarray,
    dist_cur: float,
    ttc_limit: float,
    closing_speed_thresh: float,
) -> dict[str, float | bool]:
    pos_next, vel_next = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel)
    dist_next = float(np.linalg.norm(pos_next[i] - pos_next[j]))
    rel_vel_next = np.asarray(vel_next[i] - vel_next[j], dtype=np.float32)
    radial_speed_next = float(np.dot(rel_vel_next, direction))
    closing_next = max(-radial_speed_next, 0.0)
    if dist_cur <= d_hard:
        allowed_closing = 0.0
    elif ttc_limit > 0.0:
        allowed_closing = max((dist_cur - d_hard) / ttc_limit, 0.0)
    else:
        allowed_closing = 0.0
    if dist_cur <= d_hard + 1.0e-6:
        ttc_safe = closing_next <= max(closing_speed_thresh, 1.0e-6)
    elif closing_next <= max(closing_speed_thresh, 1.0e-6):
        ttc_safe = True
    else:
        ttc_safe = closing_next <= max(allowed_closing, closing_speed_thresh) + 1.0e-6
    return {
        "dist_next": dist_next,
        "closing_next": closing_next,
        "allowed_closing": allowed_closing,
        "is_safe": bool((dist_next >= d_hard - 1.0e-6) and ttc_safe),
    }


def _select_pairwise_ttc_target_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
    d_hard: float,
) -> dict[str, Any] | None:
    trigger_dist = _pairwise_trigger_distance_from_cfg(cfg, d_hard)
    ttc_limit = _pairwise_trigger_ttc_from_cfg(cfg)
    closing_speed_thresh = _pairwise_closing_speed_threshold_from_cfg(cfg)
    pos_next, vel_next = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel)
    best: dict[str, Any] | None = None
    for i in range(int(cfg.num_uav)):
        for j in range(i + 1, int(cfg.num_uav)):
            diff_cur = np.asarray(uav_pos[i] - uav_pos[j], dtype=np.float32)
            dist_cur = float(np.linalg.norm(diff_cur))
            diff_next = np.asarray(pos_next[i] - pos_next[j], dtype=np.float32)
            dist_next = float(np.linalg.norm(diff_next))
            direction = _pairwise_correction_direction_from_state(uav_pos, uav_vel, diff_cur, i, j)
            rel_vel_next = np.asarray(vel_next[i] - vel_next[j], dtype=np.float32)
            closing_next = max(-float(np.dot(rel_vel_next, direction)), 0.0)
            immediate = dist_cur < d_hard or dist_next < d_hard
            ttc_to_hard = float("inf")
            triggered = immediate
            if (not triggered) and dist_cur <= trigger_dist and ttc_limit > 0.0 and closing_next > closing_speed_thresh:
                ttc_to_hard = (dist_cur - d_hard) / max(closing_next, 1.0e-6)
                triggered = ttc_to_hard < ttc_limit
            if not triggered:
                continue
            priority = (0, dist_next, dist_cur) if immediate else (1, ttc_to_hard, dist_cur)
            candidate = {
                "i": i,
                "j": j,
                "direction": direction,
                "dist_cur": dist_cur,
                "dist_next": dist_next,
                "closing_next": closing_next,
                "ttc_to_hard": ttc_to_hard,
                "ttc_limit": ttc_limit,
                "closing_speed_thresh": closing_speed_thresh,
                "priority": priority,
            }
            if best is None or priority < best["priority"]:
                best = candidate
    return best


def _resolve_pairwise_violation_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
    i: int,
    j: int,
    d_hard: float,
) -> tuple[np.ndarray, bool, bool, bool]:
    pos_next, _ = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel)
    diff_next = np.asarray(pos_next[i] - pos_next[j], dtype=np.float32)
    dist_next = float(np.linalg.norm(diff_next))
    if dist_next >= d_hard:
        return np.asarray(accel, dtype=np.float32), False, False, False

    direction = _pairwise_correction_direction_from_state(uav_pos, uav_vel, diff_next, i, j)
    tau = max(float(cfg.tau0), 1.0e-6)
    gap = max(d_hard - dist_next, 0.0)
    required_push = gap / max(2.0 * tau * tau, 1.0e-6)

    accel_candidate = np.asarray(accel, dtype=np.float32).copy()
    accel_candidate[i] = _project_l2_ball_np(accel_candidate[i] + required_push * direction, cfg.a_max)
    accel_candidate[j] = _project_l2_ball_np(accel_candidate[j] - required_push * direction, cfg.a_max)
    accel_candidate, _ = _apply_boundary_hard_filter_from_state(cfg, uav_pos, uav_vel, accel_candidate, indices=[i, j])
    pos_candidate, _ = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel_candidate)
    dist_candidate = float(np.linalg.norm(pos_candidate[i] - pos_candidate[j]))
    if dist_candidate >= d_hard:
        return accel_candidate, True, False, False

    accel_fallback = np.asarray(accel, dtype=np.float32).copy()
    accel_fallback[i] = _project_l2_ball_np(direction * float(cfg.a_max), cfg.a_max)
    accel_fallback[j] = _project_l2_ball_np(-direction * float(cfg.a_max), cfg.a_max)
    accel_fallback, _ = _apply_boundary_hard_filter_from_state(cfg, uav_pos, uav_vel, accel_fallback, indices=[i, j])
    pos_fallback, _ = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel_fallback)
    dist_fallback = float(np.linalg.norm(pos_fallback[i] - pos_fallback[j]))
    if dist_fallback + 1.0e-6 >= dist_candidate:
        return accel_fallback, True, True, True
    return accel_candidate, True, True, False


def _resolve_pairwise_ttc_violation_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
    pair_info: dict[str, Any],
    d_hard: float,
) -> tuple[np.ndarray, bool, bool, bool]:
    i = int(pair_info["i"])
    j = int(pair_info["j"])
    direction = np.asarray(pair_info["direction"], dtype=np.float32)
    dist_cur = float(pair_info["dist_cur"])
    ttc_limit = float(pair_info["ttc_limit"])
    closing_speed_thresh = float(pair_info["closing_speed_thresh"])
    base_eval = _evaluate_pairwise_ttc_resolution_from_state(
        cfg,
        uav_pos,
        uav_vel,
        accel,
        i,
        j,
        d_hard,
        direction,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(base_eval["is_safe"]):
        return np.asarray(accel, dtype=np.float32), False, False, False

    tau = max(float(cfg.tau0), 1.0e-6)
    delta_closing = max(float(base_eval["closing_next"]) - float(base_eval["allowed_closing"]), 0.0)
    required_push = delta_closing / max(2.0 * tau, 1.0e-6)
    if float(base_eval["dist_next"]) < d_hard:
        gap = max(d_hard - float(base_eval["dist_next"]), 0.0)
        required_push = max(required_push, gap / max(2.0 * tau * tau, 1.0e-6))

    accel_candidate = np.asarray(accel, dtype=np.float32).copy()
    accel_candidate[i] = _project_l2_ball_np(accel_candidate[i] + required_push * direction, cfg.a_max)
    accel_candidate[j] = _project_l2_ball_np(accel_candidate[j] - required_push * direction, cfg.a_max)
    accel_candidate, _ = _apply_boundary_hard_filter_from_state(cfg, uav_pos, uav_vel, accel_candidate, indices=[i, j])
    candidate_eval = _evaluate_pairwise_ttc_resolution_from_state(
        cfg,
        uav_pos,
        uav_vel,
        accel_candidate,
        i,
        j,
        d_hard,
        direction,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(candidate_eval["is_safe"]):
        return accel_candidate, True, False, False

    accel_fallback = np.asarray(accel, dtype=np.float32).copy()
    accel_fallback[i] = _project_l2_ball_np(direction * float(cfg.a_max), cfg.a_max)
    accel_fallback[j] = _project_l2_ball_np(-direction * float(cfg.a_max), cfg.a_max)
    accel_fallback, _ = _apply_boundary_hard_filter_from_state(cfg, uav_pos, uav_vel, accel_fallback, indices=[i, j])
    fallback_eval = _evaluate_pairwise_ttc_resolution_from_state(
        cfg,
        uav_pos,
        uav_vel,
        accel_fallback,
        i,
        j,
        d_hard,
        direction,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(fallback_eval["is_safe"]):
        return accel_fallback, True, True, True
    if (
        float(fallback_eval["dist_next"]) > float(candidate_eval["dist_next"]) + 1.0e-6
        or (
            abs(float(fallback_eval["dist_next"]) - float(candidate_eval["dist_next"])) <= 1.0e-6
            and float(fallback_eval["closing_next"]) <= float(candidate_eval["closing_next"]) + 1.0e-6
        )
    ):
        return accel_fallback, True, True, True
    return accel_candidate, True, True, False


def _apply_pairwise_hard_filter_from_state(
    cfg,
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    accel: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    zero_stats = {
        "pairwise_filter_count": 0.0,
        "pairwise_filter_active_ratio": 0.0,
        "pairwise_projected_delta_norm": 0.0,
        "pairwise_fallback_count": 0.0,
        "pairwise_candidate_infeasible_count": 0.0,
    }
    if not bool(getattr(cfg, "pairwise_hard_filter_enabled", False)):
        return np.asarray(accel, dtype=np.float32).copy(), zero_stats

    accel_in = np.asarray(accel, dtype=np.float32)
    accel_safe = accel_in.copy()
    d_hard = _pairwise_hard_distance_from_cfg(cfg)
    total_pairs = max(int(cfg.num_uav) * (int(cfg.num_uav) - 1) // 2, 1)
    touched_pairs: set[tuple[int, int]] = set()
    pairwise_filter_count = 0
    pairwise_fallback_count = 0
    pairwise_candidate_infeasible_count = 0

    if _pairwise_trigger_mode_from_cfg(cfg) == "ttc":
        single_pair_only = bool(getattr(cfg, "pairwise_hard_single_pair_only", True))
        max_passes = 1 if single_pair_only else max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        for _ in range(max_passes):
            pair_info = _select_pairwise_ttc_target_from_state(cfg, uav_pos, uav_vel, accel_safe, d_hard)
            if pair_info is None:
                break
            i = int(pair_info["i"])
            j = int(pair_info["j"])
            accel_next, adjusted, candidate_infeasible, used_fallback = _resolve_pairwise_ttc_violation_from_state(
                cfg,
                uav_pos,
                uav_vel,
                accel_safe,
                pair_info,
                d_hard,
            )
            if (not adjusted) or np.allclose(accel_next, accel_safe, atol=1.0e-6):
                break
            accel_safe = accel_next
            pairwise_filter_count += 1
            pairwise_candidate_infeasible_count += int(candidate_infeasible)
            pairwise_fallback_count += int(used_fallback)
            touched_pairs.add((i, j))
            if single_pair_only:
                break
    else:
        max_passes = max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        for _ in range(max_passes):
            pos_next, _ = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel_safe)
            pair_order: list[tuple[float, int, int]] = []
            for i in range(int(cfg.num_uav)):
                for j in range(i + 1, int(cfg.num_uav)):
                    dist_next = float(np.linalg.norm(pos_next[i] - pos_next[j]))
                    if dist_next < d_hard:
                        pair_order.append((dist_next, i, j))
            if not pair_order:
                break
            pair_order.sort(key=lambda item: item[0])
            changed_in_pass = False
            for _, i, j in pair_order:
                pos_cur, _ = _predict_next_from_accel_from_state(cfg, uav_pos, uav_vel, accel_safe)
                dist_cur = float(np.linalg.norm(pos_cur[i] - pos_cur[j]))
                if dist_cur >= d_hard:
                    continue
                accel_next, adjusted, candidate_infeasible, used_fallback = _resolve_pairwise_violation_from_state(
                    cfg,
                    uav_pos,
                    uav_vel,
                    accel_safe,
                    i,
                    j,
                    d_hard,
                )
                if (not adjusted) or np.allclose(accel_next, accel_safe, atol=1.0e-6):
                    continue
                changed_in_pass = True
                accel_safe = accel_next
                pairwise_filter_count += 1
                pairwise_candidate_infeasible_count += int(candidate_infeasible)
                pairwise_fallback_count += int(used_fallback)
                touched_pairs.add((i, j))
            if not changed_in_pass:
                break

    delta_norm = float(np.mean(np.linalg.norm(accel_safe - accel_in, axis=1))) if int(cfg.num_uav) > 0 else 0.0
    return accel_safe, {
        "pairwise_filter_count": float(pairwise_filter_count),
        "pairwise_filter_active_ratio": float(len(touched_pairs)) / float(total_pairs),
        "pairwise_projected_delta_norm": delta_norm,
        "pairwise_fallback_count": float(pairwise_fallback_count),
        "pairwise_candidate_infeasible_count": float(pairwise_candidate_infeasible_count),
    }


def _predict_next_from_accel_tensor_from_state(
    cfg,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accel_cmd_t = _project_l2_ball_torch(accel_t, cfg.a_max)
    vel_next_t = _project_l2_ball_torch(
        uav_vel_t.to(dtype=torch.float32) + accel_cmd_t * float(cfg.tau0),
        cfg.v_max,
    )
    pos_next_t = uav_pos_t.to(dtype=torch.float32) + vel_next_t * float(cfg.tau0)
    return pos_next_t, vel_next_t


def _project_axis_to_boundary_tensor_from_state(
    accel_safety_params: _NativeAccelSafetyStaticParams,
    pos_t: torch.Tensor,
    vel_t: torch.Tensor,
    accel_cmd_t: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = max(float(accel_safety_params.tau0), 1.0e-6)
    a_max = float(accel_safety_params.a_max)
    v_max = float(accel_safety_params.v_max)
    pos_work_t = pos_t.to(dtype=torch.float32)
    lower_t = pos_work_t * 0.0 + float(lower)
    upper_t = pos_work_t * 0.0 + float(upper)
    accel_cmd_t = torch.clamp(accel_cmd_t.to(dtype=torch.float32), min=-a_max, max=a_max)
    vel_work_t = vel_t.to(dtype=torch.float32)
    vel_cmd_t = torch.clamp(vel_work_t + accel_cmd_t * tau, min=-v_max, max=v_max)
    pos_cmd_t = pos_work_t + vel_cmd_t * tau
    within_t = (pos_cmd_t >= lower_t) & (pos_cmd_t <= upper_t)

    vel_low_t = torch.maximum(pos_work_t * 0.0 - v_max, (lower_t - pos_work_t) / tau)
    vel_high_t = torch.minimum(pos_work_t * 0.0 + v_max, (upper_t - pos_work_t) / tau)
    proj_feasible_t = vel_low_t <= vel_high_t
    target_vel_t = torch.minimum(torch.maximum(vel_cmd_t, vel_low_t), vel_high_t)
    accel_proj_t = torch.clamp((target_vel_t - vel_work_t) / tau, min=-a_max, max=a_max)
    vel_next_t = torch.clamp(vel_work_t + accel_proj_t * tau, min=-v_max, max=v_max)
    pos_next_t = pos_work_t + vel_next_t * tau
    proj_ok_t = proj_feasible_t & (pos_next_t >= lower_t) & (pos_next_t <= upper_t)

    center = 0.5 * (lower + upper)
    fallback_t = torch.where(
        (pos_cmd_t < lower_t) | (pos_work_t < lower_t),
        accel_cmd_t * 0.0 + a_max,
        torch.where(
            (pos_cmd_t > upper_t) | (pos_work_t > upper_t),
            accel_cmd_t * 0.0 - a_max,
            torch.where(
                pos_work_t < float(center),
                accel_cmd_t * 0.0 + a_max,
                accel_cmd_t * 0.0 - a_max,
            ),
        ),
    )
    accel_safe_t = torch.where(within_t, accel_cmd_t, torch.where(proj_ok_t, accel_proj_t, fallback_t))
    adjusted_t = ~within_t
    fallback_used_t = adjusted_t & ~proj_ok_t
    return accel_safe_t, adjusted_t, fallback_used_t


def _apply_boundary_hard_filter_tensor_from_state(
    accel_safety_params: _NativeAccelSafetyStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero_stats = {
        "filter_active_ratio": 0.0,
        "projected_delta_norm_mean": 0.0,
        "fallback_count": 0.0,
        "boundary_filter_count": 0.0,
        "pairwise_filter_count": 0.0,
    }
    accel_p = accel_safety_params
    if not bool(accel_p.boundary_hard_filter_enabled):
        return accel_t.to(dtype=torch.float32).clone(), zero_stats

    margin = max(float(accel_p.boundary_margin), 0.0)
    lower = margin
    upper = float(accel_p.map_size) - margin
    accel_safe_t = accel_t.to(dtype=torch.float32).clone()
    target_indices = list(range(int(accel_p.num_uav))) if indices is None else [int(idx) for idx in indices]
    if not target_indices:
        return accel_safe_t, zero_stats
    target_index_t = torch.as_tensor(target_indices, dtype=torch.long, device=accel_safe_t.device)
    accel_before_t = accel_safe_t.index_select(0, target_index_t)
    pos_sel_t = uav_pos_t.to(dtype=torch.float32).index_select(0, target_index_t)
    vel_sel_t = uav_vel_t.to(dtype=torch.float32).index_select(0, target_index_t)
    accel_sel_t = accel_before_t.clone()
    adjusted_any_t = torch.zeros((len(target_indices),), dtype=torch.bool, device=accel_safe_t.device)
    fallback_any_t = torch.zeros((len(target_indices),), dtype=torch.bool, device=accel_safe_t.device)
    for axis in range(2):
        accel_axis_t, axis_adjusted_t, axis_fallback_t = _project_axis_to_boundary_tensor_from_state(
            accel_p,
            pos_sel_t[:, axis],
            vel_sel_t[:, axis],
            accel_sel_t[:, axis],
            lower,
            upper,
        )
        accel_sel_t[:, axis] = accel_axis_t
        adjusted_any_t |= axis_adjusted_t
        fallback_any_t |= axis_fallback_t
    accel_sel_t = _project_l2_ball_torch(accel_sel_t, accel_p.a_max)
    delta_norm_t = torch.linalg.vector_norm(accel_sel_t - accel_before_t, dim=1)
    accel_safe_t.index_copy_(0, target_index_t, accel_sel_t)
    boundary_count = int(adjusted_any_t.to(dtype=torch.int32).sum().item())
    fallback_count = int(fallback_any_t.to(dtype=torch.int32).sum().item())
    return accel_safe_t, {
        "filter_active_ratio": float(boundary_count) / float(max(len(target_indices), 1)),
        "projected_delta_norm_mean": float(delta_norm_t.mean().item()) if delta_norm_t.numel() > 0 else 0.0,
        "fallback_count": float(fallback_count),
        "boundary_filter_count": float(boundary_count),
        "pairwise_filter_count": 0.0,
    }


def _pairwise_correction_direction_tensor_from_state(
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    pair_diff_next_t: torch.Tensor,
    i: int,
    j: int,
) -> torch.Tensor:
    for candidate_t in (
        pair_diff_next_t.to(dtype=torch.float32),
        (uav_pos_t[i] - uav_pos_t[j]).to(dtype=torch.float32),
        (uav_vel_t[i] - uav_vel_t[j]).to(dtype=torch.float32),
    ):
        norm_t = torch.linalg.vector_norm(candidate_t)
        if float(norm_t.item()) > 1.0e-6:
            return candidate_t / norm_t
    return torch.tensor([1.0, 0.0], dtype=torch.float32, device=uav_pos_t.device)


def _evaluate_pairwise_ttc_resolution_tensor_from_state(
    cfg,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
    i: int,
    j: int,
    d_hard: float,
    direction_t: torch.Tensor,
    dist_cur: float,
    ttc_limit: float,
    closing_speed_thresh: float,
) -> dict[str, float | bool]:
    pos_next_t, vel_next_t = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_t)
    dist_next = float(torch.linalg.vector_norm(pos_next_t[i] - pos_next_t[j]).item())
    rel_vel_next_t = (vel_next_t[i] - vel_next_t[j]).to(dtype=torch.float32)
    radial_speed_next = float(torch.dot(rel_vel_next_t, direction_t.to(dtype=torch.float32)).item())
    closing_next = max(-radial_speed_next, 0.0)
    if dist_cur <= d_hard:
        allowed_closing = 0.0
    elif ttc_limit > 0.0:
        allowed_closing = max((dist_cur - d_hard) / ttc_limit, 0.0)
    else:
        allowed_closing = 0.0
    if dist_cur <= d_hard + 1.0e-6:
        ttc_safe = closing_next <= max(closing_speed_thresh, 1.0e-6)
    elif closing_next <= max(closing_speed_thresh, 1.0e-6):
        ttc_safe = True
    else:
        ttc_safe = closing_next <= max(allowed_closing, closing_speed_thresh) + 1.0e-6
    return {
        "dist_next": dist_next,
        "closing_next": closing_next,
        "allowed_closing": allowed_closing,
        "is_safe": bool((dist_next >= d_hard - 1.0e-6) and ttc_safe),
    }


def _select_pairwise_ttc_target_tensor_from_state(
    cfg,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
    d_hard: float,
) -> dict[str, Any] | None:
    trigger_dist = _pairwise_trigger_distance_from_cfg(cfg, d_hard)
    ttc_limit = _pairwise_trigger_ttc_from_cfg(cfg)
    closing_speed_thresh = _pairwise_closing_speed_threshold_from_cfg(cfg)
    pos_next_t, vel_next_t = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_t)
    best: dict[str, Any] | None = None
    for i in range(int(cfg.num_uav)):
        for j in range(i + 1, int(cfg.num_uav)):
            diff_cur_t = (uav_pos_t[i] - uav_pos_t[j]).to(dtype=torch.float32)
            dist_cur = float(torch.linalg.vector_norm(diff_cur_t).item())
            diff_next_t = (pos_next_t[i] - pos_next_t[j]).to(dtype=torch.float32)
            dist_next = float(torch.linalg.vector_norm(diff_next_t).item())
            direction_t = _pairwise_correction_direction_tensor_from_state(uav_pos_t, uav_vel_t, diff_cur_t, i, j)
            rel_vel_next_t = (vel_next_t[i] - vel_next_t[j]).to(dtype=torch.float32)
            closing_next = max(-float(torch.dot(rel_vel_next_t, direction_t).item()), 0.0)
            immediate = dist_cur < d_hard or dist_next < d_hard
            ttc_to_hard = float("inf")
            triggered = immediate
            if (not triggered) and dist_cur <= trigger_dist and ttc_limit > 0.0 and closing_next > closing_speed_thresh:
                ttc_to_hard = (dist_cur - d_hard) / max(closing_next, 1.0e-6)
                triggered = ttc_to_hard < ttc_limit
            if not triggered:
                continue
            priority = (0, dist_next, dist_cur) if immediate else (1, ttc_to_hard, dist_cur)
            candidate = {
                "i": i,
                "j": j,
                "direction": direction_t,
                "dist_cur": dist_cur,
                "dist_next": dist_next,
                "closing_next": closing_next,
                "ttc_to_hard": ttc_to_hard,
                "ttc_limit": ttc_limit,
                "closing_speed_thresh": closing_speed_thresh,
                "priority": priority,
            }
            if best is None or priority < best["priority"]:
                best = candidate
    return best


def _resolve_pairwise_violation_tensor_from_state(
    cfg,
    accel_safety_params: _NativeAccelSafetyStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
    i: int,
    j: int,
    d_hard: float,
) -> tuple[torch.Tensor, bool, bool, bool]:
    pos_next_t, _ = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_t)
    diff_next_t = (pos_next_t[i] - pos_next_t[j]).to(dtype=torch.float32)
    dist_next = float(torch.linalg.vector_norm(diff_next_t).item())
    if dist_next >= d_hard:
        return accel_t.to(dtype=torch.float32).clone(), False, False, False

    direction_t = _pairwise_correction_direction_tensor_from_state(uav_pos_t, uav_vel_t, diff_next_t, i, j)
    tau = max(float(cfg.tau0), 1.0e-6)
    gap = max(d_hard - dist_next, 0.0)
    required_push = gap / max(2.0 * tau * tau, 1.0e-6)

    accel_candidate_t = accel_t.to(dtype=torch.float32).clone()
    accel_candidate_t[i] = _project_l2_ball_torch(accel_candidate_t[i] + required_push * direction_t, cfg.a_max)
    accel_candidate_t[j] = _project_l2_ball_torch(accel_candidate_t[j] - required_push * direction_t, cfg.a_max)
    accel_candidate_t, _ = _apply_boundary_hard_filter_tensor_from_state(
        accel_safety_params,
        uav_pos_t,
        uav_vel_t,
        accel_candidate_t,
        indices=[i, j],
    )
    pos_candidate_t, _ = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_candidate_t)
    dist_candidate = float(torch.linalg.vector_norm(pos_candidate_t[i] - pos_candidate_t[j]).item())
    if dist_candidate >= d_hard:
        return accel_candidate_t, True, False, False

    accel_fallback_t = accel_t.to(dtype=torch.float32).clone()
    accel_fallback_t[i] = _project_l2_ball_torch(direction_t * float(cfg.a_max), cfg.a_max)
    accel_fallback_t[j] = _project_l2_ball_torch(-direction_t * float(cfg.a_max), cfg.a_max)
    accel_fallback_t, _ = _apply_boundary_hard_filter_tensor_from_state(
        accel_safety_params,
        uav_pos_t,
        uav_vel_t,
        accel_fallback_t,
        indices=[i, j],
    )
    pos_fallback_t, _ = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_fallback_t)
    dist_fallback = float(torch.linalg.vector_norm(pos_fallback_t[i] - pos_fallback_t[j]).item())
    if dist_fallback + 1.0e-6 >= dist_candidate:
        return accel_fallback_t, True, True, True
    return accel_candidate_t, True, True, False


def _resolve_pairwise_ttc_violation_tensor_from_state(
    cfg,
    accel_safety_params: _NativeAccelSafetyStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
    pair_info: dict[str, Any],
    d_hard: float,
) -> tuple[torch.Tensor, bool, bool, bool]:
    i = int(pair_info["i"])
    j = int(pair_info["j"])
    direction_t = pair_info["direction"].to(dtype=torch.float32)
    dist_cur = float(pair_info["dist_cur"])
    ttc_limit = float(pair_info["ttc_limit"])
    closing_speed_thresh = float(pair_info["closing_speed_thresh"])
    base_eval = _evaluate_pairwise_ttc_resolution_tensor_from_state(
        cfg,
        uav_pos_t,
        uav_vel_t,
        accel_t,
        i,
        j,
        d_hard,
        direction_t,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(base_eval["is_safe"]):
        return accel_t.to(dtype=torch.float32).clone(), False, False, False

    tau = max(float(cfg.tau0), 1.0e-6)
    delta_closing = max(float(base_eval["closing_next"]) - float(base_eval["allowed_closing"]), 0.0)
    required_push = delta_closing / max(2.0 * tau, 1.0e-6)
    if float(base_eval["dist_next"]) < d_hard:
        gap = max(d_hard - float(base_eval["dist_next"]), 0.0)
        required_push = max(required_push, gap / max(2.0 * tau * tau, 1.0e-6))

    accel_candidate_t = accel_t.to(dtype=torch.float32).clone()
    accel_candidate_t[i] = _project_l2_ball_torch(accel_candidate_t[i] + required_push * direction_t, cfg.a_max)
    accel_candidate_t[j] = _project_l2_ball_torch(accel_candidate_t[j] - required_push * direction_t, cfg.a_max)
    accel_candidate_t, _ = _apply_boundary_hard_filter_tensor_from_state(
        accel_safety_params,
        uav_pos_t,
        uav_vel_t,
        accel_candidate_t,
        indices=[i, j],
    )
    candidate_eval = _evaluate_pairwise_ttc_resolution_tensor_from_state(
        cfg,
        uav_pos_t,
        uav_vel_t,
        accel_candidate_t,
        i,
        j,
        d_hard,
        direction_t,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(candidate_eval["is_safe"]):
        return accel_candidate_t, True, False, False

    accel_fallback_t = accel_t.to(dtype=torch.float32).clone()
    accel_fallback_t[i] = _project_l2_ball_torch(direction_t * float(cfg.a_max), cfg.a_max)
    accel_fallback_t[j] = _project_l2_ball_torch(-direction_t * float(cfg.a_max), cfg.a_max)
    accel_fallback_t, _ = _apply_boundary_hard_filter_tensor_from_state(
        accel_safety_params,
        uav_pos_t,
        uav_vel_t,
        accel_fallback_t,
        indices=[i, j],
    )
    fallback_eval = _evaluate_pairwise_ttc_resolution_tensor_from_state(
        cfg,
        uav_pos_t,
        uav_vel_t,
        accel_fallback_t,
        i,
        j,
        d_hard,
        direction_t,
        dist_cur,
        ttc_limit,
        closing_speed_thresh,
    )
    if bool(fallback_eval["is_safe"]):
        return accel_fallback_t, True, True, True
    if (
        float(fallback_eval["dist_next"]) > float(candidate_eval["dist_next"]) + 1.0e-6
        or (
            abs(float(fallback_eval["dist_next"]) - float(candidate_eval["dist_next"])) <= 1.0e-6
            and float(fallback_eval["closing_next"]) <= float(candidate_eval["closing_next"]) + 1.0e-6
        )
    ):
        return accel_fallback_t, True, True, True
    return accel_candidate_t, True, True, False


def _apply_pairwise_hard_filter_tensor_from_state(
    cfg,
    accel_safety_params: _NativeAccelSafetyStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    accel_t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero_stats = {
        "pairwise_filter_count": 0.0,
        "pairwise_filter_active_ratio": 0.0,
        "pairwise_projected_delta_norm": 0.0,
        "pairwise_fallback_count": 0.0,
        "pairwise_candidate_infeasible_count": 0.0,
    }
    if not bool(getattr(cfg, "pairwise_hard_filter_enabled", False)):
        return accel_t.to(dtype=torch.float32).clone(), zero_stats

    accel_in_t = accel_t.to(dtype=torch.float32)
    accel_safe_t = accel_in_t.clone()
    d_hard = _pairwise_hard_distance_from_cfg(cfg)
    total_pairs = max(int(cfg.num_uav) * (int(cfg.num_uav) - 1) // 2, 1)
    touched_pairs: set[tuple[int, int]] = set()
    pairwise_filter_count = 0
    pairwise_fallback_count = 0
    pairwise_candidate_infeasible_count = 0

    if _pairwise_trigger_mode_from_cfg(cfg) == "ttc":
        single_pair_only = bool(getattr(cfg, "pairwise_hard_single_pair_only", True))
        max_passes = 1 if single_pair_only else max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        for _ in range(max_passes):
            pair_info = _select_pairwise_ttc_target_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_safe_t, d_hard)
            if pair_info is None:
                break
            i = int(pair_info["i"])
            j = int(pair_info["j"])
            accel_next_t, adjusted, candidate_infeasible, used_fallback = _resolve_pairwise_ttc_violation_tensor_from_state(
                cfg,
                accel_safety_params,
                uav_pos_t,
                uav_vel_t,
                accel_safe_t,
                pair_info,
                d_hard,
            )
            if (not adjusted) or torch.allclose(accel_next_t, accel_safe_t, atol=1.0e-6, rtol=0.0):
                break
            accel_safe_t = accel_next_t
            pairwise_filter_count += 1
            pairwise_candidate_infeasible_count += int(candidate_infeasible)
            pairwise_fallback_count += int(used_fallback)
            touched_pairs.add((i, j))
            if single_pair_only:
                break
    else:
        max_passes = max(int(getattr(cfg, "pairwise_hard_max_passes", 2) or 2), 1)
        for _ in range(max_passes):
            pos_next_t, _ = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_safe_t)
            pair_order: list[tuple[float, int, int]] = []
            for i in range(int(cfg.num_uav)):
                for j in range(i + 1, int(cfg.num_uav)):
                    dist_next = float(torch.linalg.vector_norm(pos_next_t[i] - pos_next_t[j]).item())
                    if dist_next < d_hard:
                        pair_order.append((dist_next, i, j))
            if not pair_order:
                break
            pair_order.sort(key=lambda item: item[0])
            changed_in_pass = False
            for _, i, j in pair_order:
                pos_cur_t, _ = _predict_next_from_accel_tensor_from_state(cfg, uav_pos_t, uav_vel_t, accel_safe_t)
                dist_cur = float(torch.linalg.vector_norm(pos_cur_t[i] - pos_cur_t[j]).item())
                if dist_cur >= d_hard:
                    continue
                accel_next_t, adjusted, candidate_infeasible, used_fallback = _resolve_pairwise_violation_tensor_from_state(
                    cfg,
                    accel_safety_params,
                    uav_pos_t,
                    uav_vel_t,
                    accel_safe_t,
                    i,
                    j,
                    d_hard,
                )
                if (not adjusted) or torch.allclose(accel_next_t, accel_safe_t, atol=1.0e-6, rtol=0.0):
                    continue
                changed_in_pass = True
                accel_safe_t = accel_next_t
                pairwise_filter_count += 1
                pairwise_candidate_infeasible_count += int(candidate_infeasible)
                pairwise_fallback_count += int(used_fallback)
                touched_pairs.add((i, j))
            if not changed_in_pass:
                break

    delta_norm_t = torch.linalg.vector_norm(accel_safe_t - accel_in_t, dim=1) if int(cfg.num_uav) > 0 else torch.zeros((0,), dtype=torch.float32, device=accel_t.device)
    return accel_safe_t, {
        "pairwise_filter_count": float(pairwise_filter_count),
        "pairwise_filter_active_ratio": float(len(touched_pairs)) / float(total_pairs),
        "pairwise_projected_delta_norm": float(delta_norm_t.mean().item()) if delta_norm_t.numel() > 0 else 0.0,
        "pairwise_fallback_count": float(pairwise_fallback_count),
        "pairwise_candidate_infeasible_count": float(pairwise_candidate_infeasible_count),
    }


def _visible_sats_sorted_batch(
    envs: Sequence[SaginParallelEnv],
    sat_pos: np.ndarray,
    *,
    typed_domains: _NativeMainKernelTypedDomains,
    uav_ecef_batch: np.ndarray | None = None,
    sat_queue_batch: np.ndarray | None = None,
    sat_load_batch: np.ndarray | None = None,
) -> tuple[list[list[list[int]]], list[np.ndarray], list[np.ndarray]]:
    if not envs:
        return [], [], []
    cfg = envs[0].cfg
    max_keep = max(int(typed_domains.shape.visible_sats_max), 0)
    mode = str(getattr(cfg, "sat_candidate_mode", "elevation") or "elevation").strip().lower()
    if mode not in {"elevation", "score"}:
        raise ValueError(f"Unsupported sat_candidate_mode='{cfg.sat_candidate_mode}'")

    if uav_ecef_batch is None:
        for env in envs:
            if env._cached_uav_ecef is None:
                env._refresh_uav_cache()
        uav_ecef_batch = np.stack([np.asarray(env._cached_uav_ecef, dtype=np.float32) for env in envs], axis=0)
    else:
        uav_ecef_batch = np.asarray(uav_ecef_batch, dtype=np.float32)
    ref_env = envs[0]
    kernel_backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    kernel_device = torch.device("cuda" if kernel_backend == "cuda" and torch.cuda.is_available() else "cpu")

    visible_groups: list[list[list[int]]] = []
    visible_flag_groups: list[np.ndarray] = []
    elev_groups: list[np.ndarray] = []

    current_sel_mask = np.zeros(
        (len(envs), cfg.num_uav, cfg.num_sat),
        dtype=bool,
    )
    if mode == "score":
        if sat_queue_batch is None:
            sat_queue_batch = np.stack([np.asarray(env.sat_queue, dtype=np.float32) for env in envs], axis=0)
        else:
            sat_queue_batch = np.asarray(sat_queue_batch, dtype=np.float32)
        if sat_load_batch is None:
            sat_load_batch = np.stack(
                [np.asarray(env.last_sat_connection_counts, dtype=np.float32) for env in envs],
                axis=0,
            )
        else:
            sat_load_batch = np.asarray(sat_load_batch, dtype=np.float32)
        for env_index, env in enumerate(envs):
            for uav_idx in range(min(len(env.last_sat_selection), cfg.num_uav)):
                selected = np.asarray(env.last_sat_selection[uav_idx], dtype=np.int32).reshape(-1)
                valid = selected[(selected >= 0) & (selected < cfg.num_sat)]
                if valid.size > 0:
                    current_sel_mask[env_index, uav_idx, valid] = True
    else:
        if sat_queue_batch is None:
            sat_queue_batch = np.zeros((len(envs), cfg.num_sat), dtype=np.float32)
        else:
            sat_queue_batch = np.asarray(sat_queue_batch, dtype=np.float32)
        if sat_load_batch is None:
            sat_load_batch = np.zeros((len(envs), cfg.num_sat), dtype=np.float32)
        else:
            sat_load_batch = np.asarray(sat_load_batch, dtype=np.float32)

    visible_kernel_out = _visible_sats_batch_tensor_impl(
        sat_pos_t=_as_kernel_tensor(np.asarray(sat_pos, dtype=np.float32)[None, :, :].repeat(len(envs), axis=0), dtype=torch.float32, device=kernel_device),
        uav_ecef_t=_as_kernel_tensor(uav_ecef_batch, dtype=torch.float32, device=kernel_device),
        sat_queue_t=_as_kernel_tensor(sat_queue_batch, dtype=torch.float32, device=kernel_device),
        sat_load_t=_as_kernel_tensor(sat_load_batch, dtype=torch.float32, device=kernel_device),
        current_sel_mask_t=_as_kernel_tensor(current_sel_mask, dtype=torch.bool, device=kernel_device),
        channel_params=typed_domains.channel,
        sat_geometry_params=typed_domains.sat_geometry,
        sat_orbit_radius_sq=float(ref_env._sat_orbit_radius_sq),
        uav_orbit_radius_sq=float(ref_env._uav_orbit_radius_sq),
        uav_orbit_radius=float(ref_env._uav_orbit_radius),
        backhaul_gain_const=float(ref_env._backhaul_gain_const),
        effective_b_backhaul_per_sat=float(ref_env._effective_b_backhaul_per_sat()),
    )
    visible_elevation_t = visible_kernel_out.elevation
    visible_above_mask_t = visible_kernel_out.above_mask
    visible_score_t = visible_kernel_out.score

    if mode == "elevation":
        elev_t = visible_elevation_t
        above_mask_t = visible_above_mask_t
        elev_batch = elev_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if max_keep > 0:
            top_idx_t, top_mask_t = _masked_topk_sat_ids_desc(elev_t, above_mask_t, k=max_keep)
            top_idx_batch = top_idx_t.detach().cpu().numpy().astype(np.int32, copy=False)
            top_mask_batch = top_mask_t.detach().cpu().numpy().astype(bool, copy=False)
        else:
            top_idx_batch = np.zeros((len(envs), cfg.num_uav, 0), dtype=np.int32)
            top_mask_batch = np.zeros((len(envs), cfg.num_uav, 0), dtype=bool)
        for env_index, env in enumerate(envs):
            elev_matrix = elev_batch[env_index]
            env._cached_elevation_matrix = elev_matrix
            env._cached_elevation_t = env.t
            visible = [[] for _ in range(cfg.num_uav)]
            visible_flag_all = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
            for uav_idx in range(cfg.num_uav):
                kept = top_idx_batch[env_index, uav_idx][top_mask_batch[env_index, uav_idx]]
                visible[uav_idx] = kept.astype(np.int32, copy=False).tolist()
                if kept.size > 0:
                    visible_flag_all[uav_idx, kept] = 1.0
            visible_groups.append(visible)
            visible_flag_groups.append(visible_flag_all)
            elev_groups.append(elev_matrix)
        return visible_groups, visible_flag_groups, elev_groups

    for env_index, env in enumerate(envs):
        elev_matrix = _tensor_index_compat(visible_elevation_t, env_index, dtype=np.float32)
        env._cached_elevation_matrix = elev_matrix
        env._cached_elevation_t = env.t
        visible = [[] for _ in range(cfg.num_uav)]
        visible_flag_all = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
        for uav_idx in range(cfg.num_uav):
            elev_u = np.asarray(elev_matrix[uav_idx], dtype=np.float32)
            above_mask_u = _tensor_index_compat(visible_above_mask_t[env_index], uav_idx, dtype=bool)
            above = np.nonzero(np.asarray(above_mask_u, dtype=bool))[0].astype(np.int32, copy=False)
            if int(above.size) <= 0:
                continue
            if mode == "score":
                score_u = _tensor_index_compat(visible_score_t[env_index], uav_idx, dtype=np.float32)
                score = np.asarray(score_u[above], dtype=np.float32)
                elev = np.asarray(elev_u[above], dtype=np.float32)
                order = np.lexsort((-elev.astype(np.float32), -score.astype(np.float32)))
            else:
                order = np.argsort(-np.asarray(elev_u[above], dtype=np.float32), kind="stable")
            kept = above[order[:max_keep]].astype(np.int32, copy=False)
            visible[uav_idx] = kept.tolist()
            if kept.size > 0:
                visible_flag_all[uav_idx, kept] = 1.0
        visible_groups.append(visible)
        visible_flag_groups.append(visible_flag_all)
        elev_groups.append(elev_matrix)
    return visible_groups, visible_flag_groups, elev_groups


def _sample_gu_arrival_batch(envs: Sequence[SaginParallelEnv]) -> np.ndarray:
    if not envs:
        return np.zeros((0, 0), dtype=np.float32)
    cfg = envs[0].cfg
    batch_size = len(envs)
    if cfg.num_gu <= 0:
        return np.zeros((batch_size, 0), dtype=np.float32)

    ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
    legacy_arrival_ramp = ramp_steps > 0
    use_arrival_ramp = ablation_flag(cfg, "use_arrival_ramp", default=False) or legacy_arrival_ramp
    traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
    native_slot_batch = all(isinstance(env, _StructuredNativeSlotView) for env in envs)
    if native_slot_batch and (not use_arrival_ramp) and traffic_model != "sticky_subset_hotspot":
        core = envs[0]._core
        slots = [int(env._slot) for env in envs]
        selected_np = np.asarray(slots, dtype=np.int64)
        tensor_state = core._runtime_tensor_state
        if selected_np.size == int(tensor_state.effective_task_arrival_rate.shape[0]) and np.array_equal(
            selected_np,
            np.arange(selected_np.size, dtype=np.int64),
        ):
            effective_rates_t = tensor_state.effective_task_arrival_rate
        else:
            selected_t = torch.as_tensor(
                selected_np,
                dtype=torch.long,
                device=tensor_state.effective_task_arrival_rate.device,
            )
            effective_rates_t = tensor_state.effective_task_arrival_rate.index_select(0, selected_t)
        base_rates_arr = np.maximum(
            effective_rates_t.detach().cpu().numpy().astype(np.float32, copy=False),
            0.0,
        ).reshape(batch_size)
        arrival_rates_batch = np.repeat(base_rates_arr[:, None], int(cfg.num_gu), axis=1).astype(np.float32, copy=False)
        arrivals_batch = np.zeros_like(arrival_rates_batch, dtype=np.float32)
        for env_index, slot in enumerate(slots):
            rates = np.asarray(arrival_rates_batch[env_index], dtype=np.float32)
            rng = core._slot_rngs[int(slot)]
            arrivals_batch[env_index] = (
                rng.poisson(rates).astype(np.float32)
                if bool(cfg.task_arrival_poisson)
                else rates.astype(np.float32, copy=False)
            )
            meta = core._slot_state_payloads[int(slot)]
            meta["last_arrival_rate"] = float(np.mean(rates)) if rates.size > 0 else 0.0
            meta["last_hotspot_mask"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        kernel_device = tensor_state.last_gu_arrival.device
        arrivals_t = torch.as_tensor(arrivals_batch, dtype=torch.float32, device=kernel_device)
        rates_t = torch.as_tensor(arrival_rates_batch, dtype=torch.float32, device=kernel_device)
        if selected_np.size == int(tensor_state.last_gu_arrival.shape[0]) and np.array_equal(
            selected_np,
            np.arange(selected_np.size, dtype=np.int64),
        ):
            tensor_state.last_gu_arrival.copy_(arrivals_t)
            tensor_state.last_gu_arrival_rate_vec.copy_(rates_t)
        else:
            selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=kernel_device)
            tensor_state.last_gu_arrival[selected_t] = arrivals_t
            tensor_state.last_gu_arrival_rate_vec[selected_t] = rates_t
        return arrivals_batch

    arrival_base_rates: list[float] = []
    for env in envs:
        arrival_rate = float(getattr(env, "effective_task_arrival_rate", cfg.task_arrival_rate))
        if use_arrival_ramp and ramp_steps > 0:
            start = float(getattr(cfg, "arrival_ramp_start", 0.0) or 0.0)
            start = float(np.clip(start, 0.0, 1.0))
            use_global = bool(getattr(cfg, "arrival_ramp_use_global", False))
            t_ref = env.global_step if use_global else env.t
            progress = min(1.0, float(t_ref) / max(ramp_steps, 1))
            arrival_rate = arrival_rate * (start + (1.0 - start) * progress)
        arrival_base_rates.append(arrival_rate)

    arrival_rates_batch = _current_task_arrival_rates_batch(envs, arrival_base_rates)
    arrivals_batch = np.zeros_like(arrival_rates_batch, dtype=np.float32)
    for env_index, env in enumerate(envs):
        arrival_rates = np.asarray(arrival_rates_batch[env_index], dtype=np.float32)
        env.last_arrival_rate = float(np.mean(arrival_rates)) if arrival_rates.size > 0 else 0.0
        env.last_gu_arrival_rate_vec = arrival_rates.astype(np.float32, copy=False)
        hotspot_idx = int(getattr(env, "_hotspot_active_idx", -1))
        hotspot_mask = np.asarray(getattr(env, "_hotspot_member_mask", np.zeros((0, cfg.num_gu), dtype=bool)))
        if 0 <= hotspot_idx < hotspot_mask.shape[0]:
            env.last_hotspot_mask = hotspot_mask[hotspot_idx].astype(np.float32, copy=False)
        else:
            env.last_hotspot_mask = np.zeros((cfg.num_gu,), dtype=np.float32)
        if bool(cfg.task_arrival_poisson):
            arrival = env.rng.poisson(arrival_rates).astype(np.float32)
        else:
            arrival = arrival_rates.astype(np.float32, copy=False)
        env.last_gu_arrival = arrival.astype(np.float32, copy=False)
        arrivals_batch[env_index] = np.asarray(arrival, dtype=np.float32)
    return arrivals_batch


def _arrival_ref_batch_from_runtime_tensor_state(
    runtime_tensor_state: StructuredBatchRuntimeTensorState,
    indices: Sequence[int],
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    kernel_device = runtime_tensor_state.arrival_ref_bits_per_step.device if device is None else torch.device(device)
    selected_np = np.asarray(indices, dtype=np.int64)
    if selected_np.size == int(runtime_tensor_state.arrival_ref_bits_per_step.shape[0]) and np.array_equal(
        selected_np,
        np.arange(selected_np.size, dtype=np.int64),
    ):
        selected_values = runtime_tensor_state.arrival_ref_bits_per_step
    else:
        selected = torch.as_tensor(selected_np, dtype=torch.long, device=runtime_tensor_state.arrival_ref_bits_per_step.device)
        selected_values = runtime_tensor_state.arrival_ref_bits_per_step.index_select(0, selected)
    return selected_values.to(
        device=kernel_device,
        dtype=torch.float32,
    )


def _arrival_ref_batch(envs: Sequence[SaginParallelEnv]) -> np.ndarray:
    if not envs:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(
        [
            reward_ratio_denominator_scalar(
                float(getattr(env, "arrival_ref_bits_per_step", 0.0) or 0.0),
                name="arrival_ref_bits_per_step",
            )
            for env in envs
        ],
        dtype=np.float32,
    )


def _bw_weighted_workload_sat_active_ref_count(cfg) -> float:
    active_count = getattr(cfg, "queue_ref_sat_active_count", None)
    if active_count is not None:
        return max(float(active_count), 1.0)
    sat_k = _sat_action_select_k_from_config(cfg)
    if sat_k > 0 and int(cfg.num_uav) > 0:
        return max(min(float(cfg.num_sat), float(sat_k * cfg.num_uav)), 1.0)
    return max(float(cfg.num_sat), 1.0)


def _bw_weighted_workload_device_ema_vectors_batch(
    envs: Sequence[SaginParallelEnv],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not envs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )
    cfg = envs[0].cfg
    batch_size = len(envs)

    def _coerce_vec(env: SaginParallelEnv, attr_vec: str, attr_scalar: str, size: int, default_value: float) -> np.ndarray:
        if size <= 0:
            return np.zeros((0,), dtype=np.float32)
        vec = getattr(env, attr_vec, None)
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape == (size,):
                return arr.astype(np.float32, copy=False)
        scalar = getattr(env, attr_scalar, None)
        if scalar is not None:
            per_entity = float(scalar) / max(float(size), 1.0)
            return np.full((size,), per_entity, dtype=np.float32)
        return np.full((size,), default_value, dtype=np.float32)

    gu_ema = np.zeros((batch_size, int(cfg.num_gu)), dtype=np.float32)
    uav_ema = np.zeros((batch_size, int(cfg.num_uav)), dtype=np.float32)
    sat_ema = np.zeros((batch_size, int(cfg.num_sat)), dtype=np.float32)
    sat_active_ref_count = _bw_weighted_workload_sat_active_ref_count(cfg)
    for env_index, env in enumerate(envs):
        arrival_ref = reward_ratio_denominator_scalar(
            float(getattr(env, "arrival_ref_bits_per_step", 0.0) or 0.0),
            name="arrival_ref_bits_per_step",
        )
        gu_default = arrival_ref / max(float(cfg.num_gu), 1.0)
        uav_default = arrival_ref / max(float(cfg.num_uav), 1.0)
        sat_default = arrival_ref / max(float(sat_active_ref_count), 1.0)
        gu_ema[env_index] = _coerce_vec(
            env,
            "bw_weighted_workload_acc_ema_vec",
            "bw_weighted_workload_acc_ema",
            int(cfg.num_gu),
            gu_default,
        )
        uav_ema[env_index] = _coerce_vec(
            env,
            "bw_weighted_workload_rel_ema_vec",
            "bw_weighted_workload_rel_ema",
            int(cfg.num_uav),
            uav_default,
        )
        sat_ema[env_index] = _coerce_vec(
            env,
            "bw_weighted_workload_sat_ema_vec",
            "bw_weighted_workload_sat_ema",
            int(cfg.num_sat),
            sat_default,
        )
    return gu_ema, uav_ema, sat_ema


def _sat_selection_presence_batch(
    sat_selection_matrix: np.ndarray,
    *,
    num_sat: int,
) -> np.ndarray:
    sat_selection_arr = np.asarray(sat_selection_matrix, dtype=np.int64)
    if sat_selection_arr.ndim != 3:
        raise ValueError("sat_selection_matrix must have shape [B, U, K]")
    batch_size, num_uav, select_k = sat_selection_arr.shape
    if num_sat <= 0:
        return np.zeros((batch_size, num_uav, 0), dtype=np.float32)
    selected = np.zeros((batch_size, num_uav, int(num_sat)), dtype=np.float32)
    batch_coords = np.broadcast_to(np.arange(batch_size, dtype=np.int64)[:, None], (batch_size, num_uav))
    uav_coords = np.broadcast_to(np.arange(num_uav, dtype=np.int64)[None, :], (batch_size, num_uav))
    for slot in range(select_k):
        sat_idx = sat_selection_arr[:, :, slot]
        valid = (sat_idx >= 0) & (sat_idx < int(num_sat))
        if np.any(valid):
            selected[batch_coords[valid], uav_coords[valid], sat_idx[valid]] = 1.0
    return selected


def _sat_overlap_eval_batch(
    sat_selection_matrix: np.ndarray,
    *,
    num_sat: int,
) -> np.ndarray:
    sat_selected = _sat_selection_presence_batch(sat_selection_matrix, num_sat=num_sat)
    if sat_selected.size <= 0:
        return np.zeros((sat_selected.shape[0],), dtype=np.float32)
    _, num_uav, _ = sat_selected.shape
    if num_uav <= 1:
        return np.zeros((sat_selected.shape[0],), dtype=np.float32)
    counts = np.sum(sat_selected, axis=1, dtype=np.float32)
    denom = max(float(num_uav - 1), 1.0)
    selected_count = np.sum(sat_selected, axis=2, dtype=np.float32)
    overlap_sum = np.sum((counts[:, None, :] - 1.0) * sat_selected, axis=2, dtype=np.float32)
    overlap_u = np.where(selected_count > 0.0, ratio_or_zero(overlap_sum, selected_count * denom), 0.0)
    return np.mean(overlap_u, axis=1, dtype=np.float32).astype(np.float32, copy=False)


def _bw_weighted_workload_device_costs_batch(
    envs: Sequence[SaginParallelEnv],
    *,
    associations: np.ndarray,
    sat_selection_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not envs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )
    cfg = envs[0].cfg
    batch_size = len(envs)
    eps_value = getattr(cfg, "service_floor_bits_per_step", None)
    if eps_value is None:
        eps_value = getattr(cfg, "bw_weighted_workload_eps", 1.0)
    eps = max(float(eps_value or 0.0), float(NORMALIZATION_DENOM_EPS))
    gu_ema, uav_ema, sat_ema = _bw_weighted_workload_device_ema_vectors_batch(envs)
    sat_cost = (1.0 / np.maximum(sat_ema, eps)).astype(np.float32, copy=False)
    sat_cost_fallback = (
        np.mean(sat_cost, axis=1, dtype=np.float32)
        if sat_cost.shape[1] > 0
        else np.zeros((batch_size,), dtype=np.float32)
    )
    sat_selected = _sat_selection_presence_batch(sat_selection_matrix, num_sat=int(cfg.num_sat))
    sat_selected_count = np.sum(sat_selected, axis=2, dtype=np.float32)
    sat_cost_sum = np.sum(sat_selected * sat_cost[:, None, :], axis=2, dtype=np.float32)
    uav_downstream_cost = np.where(
        sat_selected_count > 0.0,
        ratio_or_zero(sat_cost_sum, sat_selected_count),
        sat_cost_fallback[:, None],
    ).astype(np.float32, copy=False)

    uav_cost = (1.0 / np.maximum(uav_ema, eps) + uav_downstream_cost).astype(np.float32, copy=False)
    uav_cost_fallback = (
        np.mean(uav_cost, axis=1, dtype=np.float32)
        if uav_cost.shape[1] > 0
        else np.zeros((batch_size,), dtype=np.float32)
    )

    assoc_arr = np.asarray(associations, dtype=np.int32).reshape(batch_size, int(cfg.num_gu))
    gu_downstream_cost = np.broadcast_to(uav_cost_fallback[:, None], (batch_size, int(cfg.num_gu))).copy()
    if int(cfg.num_uav) > 0 and int(cfg.num_gu) > 0:
        valid_assoc = (assoc_arr >= 0) & (assoc_arr < int(cfg.num_uav))
        if np.any(valid_assoc):
            batch_coords = np.broadcast_to(
                np.arange(batch_size, dtype=np.int64)[:, None],
                assoc_arr.shape,
            )
            gu_downstream_cost[valid_assoc] = uav_cost[batch_coords[valid_assoc], assoc_arr[valid_assoc]]
    gu_cost = (1.0 / np.maximum(gu_ema, eps) + gu_downstream_cost).astype(np.float32, copy=False)
    return gu_cost, uav_cost, sat_cost


def _backhaul_loss_factor_batch(
    *,
    cfg,
    elevation_batch: np.ndarray,
) -> np.ndarray | None:
    if not (bool(getattr(cfg, "atm_loss_enabled", False)) or bool(getattr(cfg, "rain_loss_enabled", False))):
        return None
    theta = np.asarray(elevation_batch, dtype=np.float32)
    factor = np.ones_like(theta, dtype=np.float32)
    if bool(getattr(cfg, "atm_loss_enabled", False)):
        atm_loss = channel.atmospheric_loss_db(theta, cfg.atm_loss_db)
        factor *= 10.0 ** (-atm_loss / 10.0)
    if bool(getattr(cfg, "rain_loss_enabled", False)):
        rain_loss = channel.rain_attenuation_db(
            theta,
            _backhaul_carrier_freq_from_cfg(cfg),
            cfg.rain_rate_001_mmph,
            cfg.rain_height_km,
            cfg.uav_height / 1000.0,
            cfg.ref_lat_deg if cfg.rain_lat_deg is None else cfg.rain_lat_deg,
            exceedance_pct=cfg.rain_exceedance_pct,
            polarization_tilt_deg=cfg.rain_polarization_tilt_deg,
        )
        factor *= 10.0 ** (-rain_loss / 10.0)
    return factor.astype(np.float32, copy=False)


def _compute_uav_cache_tensors_impl(
    *,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    sat_geometry_params: _NativeSatGeometryStaticParams,
    uav_ecef_out_t: torch.Tensor,
    uav_vel_ecef_out_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sat_p = sat_geometry_params
    if (
        not torch.is_tensor(uav_ecef_out_t)
        or not torch.is_tensor(uav_vel_ecef_out_t)
        or uav_ecef_out_t.device != uav_pos_t.device
        or uav_vel_ecef_out_t.device != uav_vel_t.device
        or tuple(uav_ecef_out_t.shape) != (*uav_pos_t.shape[:-1], 3)
        or tuple(uav_vel_ecef_out_t.shape) != (*uav_vel_t.shape[:-1], 3)
    ):
        raise RuntimeError("strict UAV geometry segment requires preallocated ECEF output buffers.")
    lat0 = math.radians(float(sat_p.ref_lat_deg))
    lon0 = math.radians(float(sat_p.ref_lon_deg))
    cos_lat0 = math.cos(lat0)
    denom_lon = normalize_scale(float(sat_p.r_earth) * cos_lat0)

    lat_t = lat0 + uav_pos_t[..., 1] / float(sat_p.r_earth)
    lon_t = lon0 + uav_pos_t[..., 0] / denom_lon
    r = float(sat_p.r_earth + sat_p.uav_height)
    cos_lat_t = torch.cos(lat_t)
    sin_lat_t = torch.sin(lat_t)
    cos_lon_t = torch.cos(lon_t)
    sin_lon_t = torch.sin(lon_t)

    uav_ecef_t = uav_ecef_out_t
    uav_ecef_t[..., 0] = r * cos_lat_t * cos_lon_t
    uav_ecef_t[..., 1] = r * cos_lat_t * sin_lon_t
    uav_ecef_t[..., 2] = r * sin_lat_t

    east_t = uav_vel_t[..., 0]
    north_t = uav_vel_t[..., 1]
    up_t = east_t.to(dtype=torch.float32) * 0.0
    uav_vel_ecef_t = uav_vel_ecef_out_t
    uav_vel_ecef_t[..., 0] = -sin_lon_t * east_t - sin_lat_t * cos_lon_t * north_t + cos_lat_t * cos_lon_t * up_t
    uav_vel_ecef_t[..., 1] = cos_lon_t * east_t - sin_lat_t * sin_lon_t * north_t + cos_lat_t * sin_lon_t * up_t
    uav_vel_ecef_t[..., 2] = cos_lat_t * north_t + sin_lat_t * up_t
    return uav_ecef_t, uav_vel_ecef_t


def _elevation_matrix_from_positions_tensor_impl(
    *,
    sat_pos_t: torch.Tensor,
    uav_ecef_t: torch.Tensor,
    sat_orbit_radius_sq: float,
    uav_orbit_radius_sq: float,
    uav_orbit_radius: float,
) -> torch.Tensor:
    rel_t = sat_pos_t[:, None, :, :] - uav_ecef_t[:, :, None, :]
    dist_t = _torch_positive(torch.linalg.vector_norm(rel_t, dim=-1), GEOMETRY_DENOM_EPS)
    arg_t = (float(sat_orbit_radius_sq) - float(uav_orbit_radius_sq) - dist_t.square()) / (
        _torch_positive(2.0 * float(uav_orbit_radius) * dist_t, GEOMETRY_DENOM_EPS)
    )
    arg_t = torch.clamp(arg_t, -1.0, 1.0)
    return torch.asin(arg_t).to(dtype=torch.float32)


def _minmax_normalize_masked_torch(
    values_t: torch.Tensor,
    valid_mask_t: torch.Tensor,
) -> torch.Tensor:
    if values_t.numel() <= 0:
        return values_t.to(dtype=torch.float32)
    valid_any_t = valid_mask_t.any(dim=-1, keepdim=True)
    pos_inf = values_t * 0.0 + float("inf")
    neg_inf = values_t * 0.0 - float("inf")
    min_t = torch.where(valid_mask_t, values_t, pos_inf).amin(dim=-1, keepdim=True)
    max_t = torch.where(valid_mask_t, values_t, neg_inf).amax(dim=-1, keepdim=True)
    min_t = torch.where(valid_any_t, min_t, min_t * 0.0)
    max_t = torch.where(valid_any_t, max_t, max_t * 0.0)
    span_t = max_t - min_t
    normalized_t = _torch_divide_or_default(values_t - min_t, span_t, eps=NORMALIZATION_DENOM_EPS, default=0.0)
    return torch.where(valid_mask_t & (span_t > NORMALIZATION_DENOM_EPS), normalized_t, values_t * 0.0).to(dtype=torch.float32)


def _sat_selection_presence_tensor_impl(
    sat_selection_matrix_t: torch.Tensor,
    *,
    num_sat: int,
) -> torch.Tensor:
    if sat_selection_matrix_t.ndim != 3:
        raise ValueError("sat_selection_matrix must have shape [B, U, K]")
    batch_size, num_uav, _select_k = sat_selection_matrix_t.shape
    if int(num_sat) <= 0:
        raise RuntimeError("strict satellite selection presence requires num_sat > 0.")
    sat_idx_t = sat_selection_matrix_t.to(dtype=torch.long)
    valid_t = (sat_idx_t >= 0) & (sat_idx_t < int(num_sat))
    clamped_t = torch.clamp(sat_idx_t, min=0, max=max(int(num_sat) - 1, 0))
    selected_t = F.one_hot(clamped_t, num_classes=int(num_sat)).to(dtype=torch.float32)
    selected_t = selected_t * valid_t.unsqueeze(-1).to(dtype=torch.float32)
    return torch.amax(selected_t, dim=2)


def _reward_aligned_feature_bundle_batch_from_state_tensor_impl(
    *,
    bw_workload_static_params: _NativeBwWorkloadStaticParams,
    gu_queue_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    gu_ema_t: torch.Tensor,
    uav_ema_t: torch.Tensor,
    sat_ema_t: torch.Tensor,
    assoc_t: torch.Tensor,
    sat_selection_matrix_t: torch.Tensor,
) -> tuple[_NativeRewardAlignedGuFeatureTensorFields, torch.Tensor, torch.Tensor]:
    params = bw_workload_static_params
    arrival_ref_vec_t = _torch_require_positive_reward_ref(
        arrival_ref_t.reshape(-1),
        name="arrival_ref_bits_per_step",
    )
    eps = float(params.eps)

    sat_cost_t = 1.0 / torch.clamp(sat_ema_t, min=eps)
    if sat_cost_t.shape[1] <= 0:
        raise RuntimeError("strict workload features require at least one satellite.")
    sat_cost_fallback_t = sat_cost_t.mean(dim=1)
    sat_selected_t = _sat_selection_presence_tensor_impl(sat_selection_matrix_t, num_sat=int(params.num_sat))
    sat_selected_count_t = sat_selected_t.sum(dim=2)
    sat_cost_sum_t = (sat_selected_t * sat_cost_t[:, None, :]).sum(dim=2)
    uav_downstream_cost_t = torch.where(
        sat_selected_count_t > 0.0,
        _torch_ratio_or_zero(sat_cost_sum_t, sat_selected_count_t),
        sat_cost_fallback_t[:, None],
    ).to(dtype=torch.float32)

    uav_cost_t = (1.0 / torch.clamp(uav_ema_t, min=eps) + uav_downstream_cost_t).to(dtype=torch.float32)
    if uav_cost_t.shape[1] <= 0:
        raise RuntimeError("strict workload features require at least one UAV.")
    uav_cost_fallback_t = uav_cost_t.mean(dim=1)

    assoc_long_t = assoc_t.to(dtype=torch.long)
    if int(params.num_uav) > 0 and int(params.num_gu) > 0:
        valid_assoc_t = (assoc_long_t >= 0) & (assoc_long_t < int(params.num_uav))
        assoc_clamped_t = torch.clamp(assoc_long_t, min=0, max=max(int(params.num_uav) - 1, 0))
        assoc_uav_cost_t = uav_cost_t.gather(1, assoc_clamped_t)
        assoc_sat_cost_t = uav_downstream_cost_t.gather(1, assoc_clamped_t)
        gu_downstream_cost_t = torch.where(valid_assoc_t, assoc_uav_cost_t, uav_cost_fallback_t[:, None])
        assoc_uav_cost_t = torch.where(valid_assoc_t, assoc_uav_cost_t, uav_cost_fallback_t[:, None])
        assoc_sat_cost_t = torch.where(valid_assoc_t, assoc_sat_cost_t, sat_cost_fallback_t[:, None])
    else:
        gu_downstream_cost_t = uav_cost_fallback_t[:, None].expand(-1, int(params.num_gu))
        assoc_uav_cost_t = gu_downstream_cost_t
        assoc_sat_cost_t = sat_cost_fallback_t[:, None].expand(-1, int(params.num_gu))
    gu_cost_t = (1.0 / torch.clamp(gu_ema_t, min=eps) + gu_downstream_cost_t).to(dtype=torch.float32)

    local_gu_service_cost_t = (1.0 / torch.clamp(gu_ema_t, min=eps)).to(dtype=torch.float32)
    weighted_queue_cost_t = (gu_cost_t * gu_queue_t).to(dtype=torch.float32)
    weighted_queue_cost_mean_t = torch.clamp(weighted_queue_cost_t.mean(dim=1), min=NORMALIZATION_DENOM_EPS)
    weighted_queue_cost_relative_t = _torch_divide_or_default(weighted_queue_cost_t, weighted_queue_cost_mean_t[:, None])

    gu_default_t = arrival_ref_vec_t / max(float(params.num_gu), 1.0)
    uav_default_t = arrival_ref_vec_t / max(float(params.num_uav), 1.0)
    sat_default_t = arrival_ref_vec_t / max(float(params.sat_active_ref_count), 1.0)
    sat_cost_ref_t = 1.0 / torch.clamp(sat_default_t, min=eps)
    uav_cost_ref_t = 1.0 / torch.clamp(uav_default_t, min=eps) + sat_cost_ref_t
    gu_local_cost_ref_t = 1.0 / torch.clamp(gu_default_t, min=eps)
    weighted_queue_ref_t = _torch_require_positive_reward_ref(
        arrival_ref_vec_t * (gu_local_cost_ref_t + uav_cost_ref_t),
        name="weighted queue reward feature reference",
    )

    local_cost_mean_t = torch.clamp(local_gu_service_cost_t.mean(dim=1), min=LOG_RATIO_EPS)
    assoc_uav_cost_mean_t = torch.clamp(assoc_uav_cost_t.mean(dim=1), min=LOG_RATIO_EPS)
    assoc_sat_cost_mean_t = torch.clamp(assoc_sat_cost_t.mean(dim=1), min=LOG_RATIO_EPS)
    sat_cost_mean_t = torch.clamp(sat_cost_t.mean(dim=1), min=LOG_RATIO_EPS)
    uav_cost_mean_t = torch.clamp(uav_cost_t.mean(dim=1), min=LOG_RATIO_EPS)

    gu_reward_aligned_t = _NativeRewardAlignedGuFeatureTensorFields(
        local_gu_service_cost=torch.log(torch.clamp(local_gu_service_cost_t, min=LOG_RATIO_EPS) / local_cost_mean_t[:, None]).to(dtype=torch.float32),
        assoc_uav_cost=torch.log(torch.clamp(assoc_uav_cost_t, min=LOG_RATIO_EPS) / assoc_uav_cost_mean_t[:, None]).to(dtype=torch.float32),
        assoc_sat_cost_mean=torch.log(torch.clamp(assoc_sat_cost_t, min=LOG_RATIO_EPS) / assoc_sat_cost_mean_t[:, None]).to(dtype=torch.float32),
        weighted_queue_cost=torch.log1p(torch.clamp(weighted_queue_cost_t, min=0.0) / weighted_queue_ref_t[:, None]).to(dtype=torch.float32),
        weighted_queue_cost_relative=torch.log(torch.clamp(weighted_queue_cost_relative_t, min=RELATIVE_LOG_EPS)).to(dtype=torch.float32),
    )
    uav_assoc_uav_cost_t = torch.log(torch.clamp(uav_cost_t, min=LOG_RATIO_EPS) / uav_cost_mean_t[:, None]).to(dtype=torch.float32)
    sat_cost_norm_t = torch.log(torch.clamp(sat_cost_t, min=LOG_RATIO_EPS) / sat_cost_mean_t[:, None]).to(dtype=torch.float32)
    return gu_reward_aligned_t, uav_assoc_uav_cost_t, sat_cost_norm_t


def _gu_proxy_feature_arrays_batch_from_state_tensor_impl(
    *,
    local_obs_params: _NativeLocalObsStaticParams,
    gu_queue_t: torch.Tensor,
    arrival_rate_vec_t: torch.Tensor,
    recent_arrival_t: torch.Tensor,
    recent_service_t: torch.Tensor,
    urgency_risk_t: torch.Tensor,
    downstream_pressure_t: torch.Tensor,
    service_gap_t: torch.Tensor,
    service_gap_risk_t: torch.Tensor,
    deadline_steps_t: torch.Tensor,
    deadline_slack_t: torch.Tensor,
    deadline_risk_t: torch.Tensor,
    reward_aligned: _NativeRewardAlignedGuFeatureTensorFields,
    out_t: torch.Tensor,
) -> None:
    local_p = local_obs_params
    if not torch.is_tensor(out_t):
        raise RuntimeError("strict GU proxy feature segment requires a runtime-owned output tensor.")
    out_t.zero_()
    feature_idx = 0

    def _write_feature(feature_t: torch.Tensor) -> None:
        nonlocal feature_idx
        if feature_idx >= int(out_t.shape[-1]):
            raise RuntimeError("GU proxy feature output ABI is smaller than the configured feature set.")
        out_t[..., int(feature_idx)].copy_(feature_t.to(device=out_t.device, dtype=torch.float32))
        feature_idx += 1

    deadline_steps_safe_t = torch.clamp(deadline_steps_t, min=1.0e-6)
    base_arrival_t = _torch_require_positive_reward_ref(
        arrival_rate_vec_t.mean(dim=1, keepdim=True) * float(local_p.tau0),
        name="per-GU arrival reference bits per step",
    )
    if bool(local_p.obs_user_include_arrival_rate):
        _write_feature((arrival_rate_vec_t / base_arrival_t).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_recent_arrival):
        _write_feature((recent_arrival_t / base_arrival_t).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_recent_service):
        _write_feature((recent_service_t / base_arrival_t).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_queue_headroom):
        _write_feature((1.0 - (gu_queue_t / normalize_scale(float(local_p.queue_max_gu)))).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_local_gu_service_cost):
        _write_feature(reward_aligned.local_gu_service_cost.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_assoc_uav_cost):
        _write_feature(reward_aligned.assoc_uav_cost.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_assoc_sat_cost_mean):
        _write_feature(reward_aligned.assoc_sat_cost_mean.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_weighted_queue_cost):
        _write_feature(reward_aligned.weighted_queue_cost.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_weighted_queue_cost_relative):
        _write_feature(reward_aligned.weighted_queue_cost_relative.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_urgency_risk):
        _write_feature(urgency_risk_t.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_downstream_pressure):
        _write_feature(downstream_pressure_t.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_service_gap):
        cap_steps = max(float(local_p.service_gap_cap_steps), 1.0e-6)
        _write_feature((service_gap_t / cap_steps).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_service_gap_risk):
        _write_feature(service_gap_risk_t.to(dtype=torch.float32))
    if bool(local_p.obs_user_include_deadline_slack):
        _write_feature(torch.clamp(deadline_slack_t / deadline_steps_safe_t, -1.0, 1.0).to(dtype=torch.float32))
    if bool(local_p.obs_user_include_deadline_risk):
        _write_feature(torch.clamp(deadline_risk_t, 0.0, 2.0).to(dtype=torch.float32))
    if feature_idx != int(out_t.shape[-1]):
        raise RuntimeError("GU proxy feature output ABI does not match the configured feature set.")
    return None


def _bw_weighted_workload_device_costs_static_tensor_impl(
    *,
    params: _NativeBwWorkloadStaticParams,
    gu_ema_t: torch.Tensor,
    uav_ema_t: torch.Tensor,
    sat_ema_t: torch.Tensor,
    assoc_t: torch.Tensor,
    sat_selection_matrix_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = float(params.eps)
    sat_cost_t = (1.0 / torch.clamp(sat_ema_t, min=eps)).to(dtype=torch.float32)
    if sat_cost_t.shape[1] <= 0:
        raise RuntimeError("strict workload cost segment requires at least one satellite.")
    sat_cost_fallback_t = sat_cost_t.mean(dim=1)
    sat_selected_t = _sat_selection_presence_tensor_impl(sat_selection_matrix_t, num_sat=int(params.num_sat))
    sat_selected_count_t = sat_selected_t.sum(dim=2)
    sat_cost_sum_t = (sat_selected_t * sat_cost_t[:, None, :]).sum(dim=2)
    uav_downstream_cost_t = torch.where(
        sat_selected_count_t > 0.0,
        _torch_ratio_or_zero(sat_cost_sum_t, sat_selected_count_t),
        sat_cost_fallback_t[:, None],
    ).to(dtype=torch.float32)
    uav_cost_t = (1.0 / torch.clamp(uav_ema_t, min=eps) + uav_downstream_cost_t).to(dtype=torch.float32)
    if uav_cost_t.shape[1] <= 0:
        raise RuntimeError("strict workload cost segment requires at least one UAV.")
    uav_cost_fallback_t = uav_cost_t.mean(dim=1)
    assoc_long_t = assoc_t.to(dtype=torch.long)
    if int(params.num_uav) > 0 and int(params.num_gu) > 0:
        valid_assoc_t = (assoc_long_t >= 0) & (assoc_long_t < int(params.num_uav))
        assoc_clamped_t = torch.clamp(assoc_long_t, min=0, max=max(int(params.num_uav) - 1, 0))
        gu_downstream_cost_t = torch.where(
            valid_assoc_t,
            uav_cost_t.gather(1, assoc_clamped_t),
            uav_cost_fallback_t[:, None],
        )
    else:
        gu_downstream_cost_t = uav_cost_fallback_t[:, None].expand(-1, int(params.num_gu))
    gu_cost_t = (1.0 / torch.clamp(gu_ema_t, min=eps) + gu_downstream_cost_t).to(dtype=torch.float32)
    return gu_cost_t, uav_cost_t, sat_cost_t


def _bw_weighted_workload_rewards_tensor_impl(
    *,
    params: _NativeBwWorkloadStaticParams,
    gu_cost_t: torch.Tensor,
    uav_cost_t: torch.Tensor,
    sat_cost_t: torch.Tensor,
    active_sat_ids_t: torch.Tensor | None = None,
    gu_queue_before_t: torch.Tensor,
    uav_queue_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    realized_arrival_t: torch.Tensor,
    gu_queue_after_t: torch.Tensor,
    uav_queue_after_t: torch.Tensor,
    sat_queue_after_t: torch.Tensor,
    gu_drop_t: torch.Tensor,
    uav_drop_t: torch.Tensor,
    sat_drop_t: torch.Tensor,
    last_gu_outflow_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
) -> _NativeBwWorkloadRewardTensorFields:
    arrival_ref_vec_t = _torch_require_positive_reward_ref(
        arrival_ref_t.reshape(-1),
        name="arrival_ref_bits_per_step",
    )
    reward_dtype = torch.float64
    gu_cost_work_t = gu_cost_t.to(dtype=reward_dtype)
    uav_cost_work_t = uav_cost_t.to(dtype=reward_dtype)
    sat_cost_work_t = sat_cost_t.to(dtype=reward_dtype)
    q_gu_before_service_t = (gu_queue_before_t + realized_arrival_t).to(dtype=reward_dtype)
    uav_queue_before_work_t = uav_queue_before_t.to(dtype=reward_dtype)
    sat_queue_before_work_t = sat_queue_before_t.to(dtype=reward_dtype)
    gu_queue_after_work_t = gu_queue_after_t.to(dtype=reward_dtype)
    uav_queue_after_work_t = uav_queue_after_t.to(dtype=reward_dtype)
    sat_queue_after_work_t = sat_queue_after_t.to(dtype=reward_dtype)
    gu_drop_work_t = gu_drop_t.to(dtype=reward_dtype)
    uav_drop_work_t = uav_drop_t.to(dtype=reward_dtype)
    sat_drop_work_t = sat_drop_t.to(dtype=reward_dtype)
    last_gu_outflow_work_t = last_gu_outflow_t.to(dtype=reward_dtype)
    workload_before_t = (
        (gu_cost_work_t * q_gu_before_service_t).sum(dim=1, dtype=reward_dtype)
        + (uav_cost_work_t * uav_queue_before_work_t).sum(dim=1, dtype=reward_dtype)
        + (sat_cost_work_t * sat_queue_before_work_t).sum(dim=1, dtype=reward_dtype)
    )
    workload_after_t = (
        (gu_cost_work_t * gu_queue_after_work_t).sum(dim=1, dtype=reward_dtype)
        + (uav_cost_work_t * uav_queue_after_work_t).sum(dim=1, dtype=reward_dtype)
        + (sat_cost_work_t * sat_queue_after_work_t).sum(dim=1, dtype=reward_dtype)
    )
    drop_cost_t = (
        (gu_cost_work_t * gu_drop_work_t).sum(dim=1, dtype=reward_dtype)
        + (uav_cost_work_t * uav_drop_work_t).sum(dim=1, dtype=reward_dtype)
        + (sat_cost_work_t * sat_drop_work_t).sum(dim=1, dtype=reward_dtype)
    )
    del active_sat_ids_t, params
    workload_t = torch.clamp(workload_after_t + drop_cost_t, min=0.0)
    positive_level_t = 1.0 / (1.0 + torch.log1p(workload_t))
    gu_backlog_after_t = gu_queue_after_work_t.sum(dim=1, dtype=reward_dtype)
    q_total_after_t = (
        gu_queue_after_work_t.sum(dim=1, dtype=reward_dtype)
        + uav_queue_after_work_t.sum(dim=1, dtype=reward_dtype)
        + sat_queue_after_work_t.sum(dim=1, dtype=reward_dtype)
    )
    gu_drop_sum_t = gu_drop_work_t.sum(dim=1, dtype=reward_dtype)
    drop_total_t = (
        gu_drop_work_t.sum(dim=1, dtype=reward_dtype)
        + uav_drop_work_t.sum(dim=1, dtype=reward_dtype)
        + sat_drop_work_t.sum(dim=1, dtype=reward_dtype)
    )
    gu_outflow_sum_t = last_gu_outflow_work_t.sum(dim=1, dtype=reward_dtype)
    return _NativeBwWorkloadRewardTensorFields(
        delta=(-(workload_after_t - workload_before_t) - drop_cost_t).to(dtype=torch.float32),
        level=(-workload_after_t - drop_cost_t).to(dtype=torch.float32),
        positive_level=positive_level_t.to(dtype=torch.float32),
        gu_queue_level=(-(gu_backlog_after_t + gu_drop_sum_t) / arrival_ref_vec_t).to(dtype=torch.float32),
        system_queue_level=(-(q_total_after_t + drop_total_t) / arrival_ref_vec_t).to(dtype=torch.float32),
        gu_service_queue=((gu_outflow_sum_t - gu_backlog_after_t - gu_drop_sum_t) / arrival_ref_vec_t).to(dtype=torch.float32),
    )


def _sat_overlap_eval_tensor_impl(
    sat_selection_matrix_t: torch.Tensor,
    *,
    num_sat: int,
) -> torch.Tensor:
    sat_selected_t = _sat_selection_presence_tensor_impl(sat_selection_matrix_t, num_sat=num_sat)
    if sat_selected_t.numel() <= 0:
        raise RuntimeError("strict satellite overlap metric requires non-empty selection tensors.")
    num_uav = int(sat_selected_t.shape[1])
    if num_uav <= 1:
        return sat_selected_t[:, 0].sum(dim=-1).to(dtype=torch.float32) * 0.0
    counts_t = sat_selected_t.sum(dim=1)
    denom = max(float(num_uav - 1), 1.0)
    selected_count_t = sat_selected_t.sum(dim=2)
    overlap_sum_t = ((counts_t[:, None, :] - 1.0) * sat_selected_t).sum(dim=2)
    overlap_u_t = torch.where(
        selected_count_t > 0.0,
        _torch_ratio_or_zero(overlap_sum_t, selected_count_t * denom),
        selected_count_t.to(dtype=torch.float32) * 0.0,
    )
    return overlap_u_t.mean(dim=1).to(dtype=torch.float32)


def _visible_sats_batch_tensor_impl(
    *,
    sat_pos_t: torch.Tensor,
    uav_ecef_t: torch.Tensor,
    sat_queue_t: torch.Tensor,
    sat_load_t: torch.Tensor,
    current_sel_mask_t: torch.Tensor,
    channel_params: _NativeChannelStaticParams,
    sat_geometry_params: _NativeSatGeometryStaticParams,
    sat_orbit_radius_sq: float,
    uav_orbit_radius_sq: float,
    uav_orbit_radius: float,
    backhaul_gain_const: float,
    effective_b_backhaul_per_sat: float,
) -> _NativeVisibleSatScoreTensorFields:
    sat_p = sat_geometry_params
    channel_p = channel_params
    elev_t = _elevation_matrix_from_positions_tensor_impl(
        sat_pos_t=sat_pos_t,
        uav_ecef_t=uav_ecef_t,
        sat_orbit_radius_sq=sat_orbit_radius_sq,
        uav_orbit_radius_sq=uav_orbit_radius_sq,
        uav_orbit_radius=uav_orbit_radius,
    )
    above_mask_t = elev_t >= float(sat_p.theta_min_rad)
    mode = str(sat_p.sat_candidate_mode)
    score_t = elev_t
    if mode == "score":
        rel_t = sat_pos_t[:, None, :, :] - uav_ecef_t[:, :, None, :]
        dist_t = _torch_positive(torch.linalg.vector_norm(rel_t, dim=-1), GEOMETRY_DENOM_EPS)
        projected_count_t = torch.clamp(
            sat_load_t[:, None, :] + (~current_sel_mask_t).to(dtype=torch.float32),
            min=1.0,
        )
        gain_t = float(backhaul_gain_const) / _torch_positive(dist_t.square(), GEOMETRY_DENOM_EPS)
        if channel_p is not None:
            loss_t = _backhaul_loss_factor_torch(
                channel_params=channel_p,
                carrier_freq_hz=float(sat_p.carrier_freq),
                elevation_batch_t=elev_t,
            )
            if loss_t is not None:
                gain_t = gain_t * loss_t.to(dtype=torch.float32)
        projected_bw_t = _torch_ratio_or_zero(projected_count_t * 0.0 + float(effective_b_backhaul_per_sat), projected_count_t)
        snr_t = _snr_linear_torch(
            power=float(sat_p.uav_tx_power),
            gain_t=gain_t,
            noise_density=float(sat_p.noise_density),
            bandwidth_t=projected_bw_t,
            noise_figure_db=float(sat_p.noise_figure_db),
        )
        se_t = _spectral_efficiency_torch(snr_t).to(dtype=torch.float32)
        queue_norm_t = sat_queue_t[:, None, :] / normalize_scale(float(sat_p.queue_max_sat))
        elev_norm_t = _minmax_normalize_masked_torch(elev_t, above_mask_t)
        se_norm_t = _minmax_normalize_masked_torch(se_t, above_mask_t)
        score_t = (
            float(sat_p.sat_candidate_elevation_weight) * elev_norm_t
            + float(sat_p.sat_candidate_se_weight) * se_norm_t
            - float(sat_p.sat_candidate_queue_weight) * queue_norm_t
        ).to(dtype=torch.float32)
    return _NativeVisibleSatScoreTensorFields(
        elevation=elev_t.to(dtype=torch.float32),
        above_mask=above_mask_t,
        score=score_t.to(dtype=torch.float32),
    )


def _masked_topk_sat_ids_desc(
    score_t: torch.Tensor,
    mask_t: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_shape = tuple(int(dim) for dim in score_t.shape[:-1])
    num_sat = int(score_t.shape[-1])
    k_i = max(min(int(k), num_sat), 0)
    if k_i <= 0:
        raise RuntimeError("strict satellite top-k segment requires k > 0 and non-empty satellites.")
    masked_score_t = torch.where(
        mask_t.to(dtype=torch.bool),
        score_t.to(dtype=torch.float32),
        score_t.to(dtype=torch.float32) * 0.0 - float("inf"),
    )
    top_score_t, top_idx_t = torch.topk(masked_score_t, k=k_i, dim=-1, largest=True, sorted=False)
    # Deterministic tie policy: lower satellite id wins among equal scores. Sort
    # the small top-k window by id first, then stable-sort by score descending.
    id_order_t = torch.argsort(top_idx_t, dim=-1, stable=True)
    idx_by_id_t = torch.gather(top_idx_t, -1, id_order_t)
    score_by_id_t = torch.gather(top_score_t, -1, id_order_t)
    score_order_t = torch.argsort(-score_by_id_t, dim=-1, stable=True)
    top_idx_ordered_t = torch.gather(idx_by_id_t, -1, score_order_t)
    top_mask_t = torch.gather(mask_t.to(dtype=torch.bool), -1, top_idx_ordered_t)
    return top_idx_ordered_t.to(dtype=torch.long), top_mask_t.to(dtype=torch.bool)


def _active_sat_ids_from_visible_ids_tensor(
    visible_ids_t: torch.Tensor,
    visible_mask_t: torch.Tensor,
    *,
    active_width: int,
    num_sat: int,
) -> torch.Tensor:
    width_i = max(min(int(active_width), int(num_sat)), 0)
    if width_i <= 0:
        raise RuntimeError("strict active satellite segment requires a positive active width.")
    if (
        not torch.is_tensor(visible_ids_t)
        or visible_ids_t.dtype != torch.long
        or visible_ids_t.ndim != 3
        or not torch.is_tensor(visible_mask_t)
        or visible_mask_t.device != visible_ids_t.device
        or tuple(visible_mask_t.shape) != tuple(visible_ids_t.shape)
    ):
        raise RuntimeError("strict active satellite segment requires visible id/mask tensors with matching [B,U,K] shape.")
    batch_size = int(visible_ids_t.shape[0])
    flat_ids_t = visible_ids_t.reshape(batch_size, -1).to(dtype=torch.long)
    flat_valid_t = (
        visible_mask_t.reshape(batch_size, -1).to(dtype=torch.bool)
        & (flat_ids_t >= 0)
        & (flat_ids_t < int(num_sat))
    )
    item_count = int(flat_ids_t.shape[1])
    if item_count <= 0:
        return torch.full((batch_size, width_i), -1, dtype=torch.long, device=visible_ids_t.device)
    pos_t = torch.arange(item_count, dtype=torch.long, device=visible_ids_t.device)
    same_id_t = flat_ids_t[:, :, None] == flat_ids_t[:, None, :]
    previous_t = pos_t.view(1, item_count, 1) > pos_t.view(1, 1, item_count)
    duplicate_before_t = (same_id_t & previous_t & flat_valid_t[:, None, :]).any(dim=2)
    first_valid_t = flat_valid_t & ~duplicate_before_t
    order_key_t = torch.where(first_valid_t, pos_t.view(1, item_count), torch.full_like(flat_ids_t, item_count))
    order_t = torch.argsort(order_key_t, dim=1, stable=True)[:, :width_i]
    active_ids_t = torch.gather(flat_ids_t, 1, order_t)
    active_mask_t = torch.gather(first_valid_t, 1, order_t)
    return torch.where(active_mask_t, active_ids_t, torch.full_like(active_ids_t, -1)).to(dtype=torch.long)


def _prepare_native_stage_batch_tensor_impl(
    *,
    channel_params: _NativeChannelStaticParams,
    candidate_params: _NativeCandidateStaticParams,
    sat_geometry_params: _NativeSatGeometryStaticParams,
    local_obs_params: _NativeLocalObsStaticParams,
    access_rate_static_params: _NativeAccessRateStaticParams,
    bw_workload_static_params: _NativeBwWorkloadStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    uav_energy_t: torch.Tensor,
    uav_queue_t: torch.Tensor,
    gu_pos_t: torch.Tensor,
    gu_queue_t: torch.Tensor,
    sat_queue_t: torch.Tensor,
    prev_association_t: torch.Tensor,
    last_sat_selection_matrix_t: torch.Tensor,
    last_sat_connection_counts_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    gu_ema_t: torch.Tensor,
    uav_ema_t: torch.Tensor,
    sat_ema_t: torch.Tensor,
    expected_arrival_rate_vec_t: torch.Tensor,
    last_gu_arrival_t: torch.Tensor,
    last_gu_outflow_t: torch.Tensor,
    last_gu_urgency_risk_t: torch.Tensor,
    last_gu_downstream_pressure_t: torch.Tensor,
    last_gu_service_gap_t: torch.Tensor,
    last_gu_service_gap_risk_t: torch.Tensor,
    gu_deadline_steps_t: torch.Tensor,
    last_gu_deadline_slack_t: torch.Tensor,
    last_gu_deadline_risk_t: torch.Tensor,
    doppler_residual_t: torch.Tensor,
    sat_pos_base_t: torch.Tensor,
    sat_vel_base_t: torch.Tensor,
    fading_gain_t: torch.Tensor,
    sat_orbit_radius_sq: float,
    uav_orbit_radius_sq: float,
    uav_orbit_radius: float,
    backhaul_gain_const: float,
    effective_b_backhaul_per_sat: float,
    candidate_slot_ids_t: torch.Tensor,
    candidate_gu_ids_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
    candidate_env_ids_t: torch.Tensor,
    sat_all_ids_t: torch.Tensor,
    stage_out: _NativeStageTensorFields,
    gu_drop_t: torch.Tensor | None = None,
    uav_drop_t: torch.Tensor | None = None,
    sat_drop_t: torch.Tensor | None = None,
    last_gu_to_uav_inflow_by_uav_t: torch.Tensor | None = None,
    last_uav_to_sat_outflow_matrix_t: torch.Tensor | None = None,
    last_bw_fraction_by_uav_gu_t: torch.Tensor | None = None,
    last_access_interference_by_uav_t: torch.Tensor | None = None,
    last_sat_processed_t: torch.Tensor | None = None,
    last_selected_mask_by_uav_sat_t: torch.Tensor | None = None,
) -> None:
    sat_p = sat_geometry_params
    batch_size = int(uav_pos_t.shape[0])
    num_uav = int(sat_p.num_uav)
    num_sat = int(sat_p.num_sat)
    select_k = max(int(sat_p.sat_num_select), 1)
    device = uav_pos_t.device
    if not _is_native_stage_fields(stage_out):
        raise RuntimeError("prepare stage live segment requires preallocated runtime-owned stage buffers.")

    assoc_t, access_base_gain_t = _associate_users_and_access_base_gain_tensor_impl(
        channel_params=channel_params,
        candidate_params=candidate_params,
        gu_pos_t=gu_pos_t.to(dtype=torch.float32),
        uav_pos_t=uav_pos_t.to(dtype=torch.float32),
    )
    _build_candidate_index_mask_tensor_impl(
        candidate_params=candidate_params,
        assoc_t=assoc_t,
        gu_queue_t=gu_queue_t.to(dtype=torch.float32),
        gu_pos_t=gu_pos_t.to(dtype=torch.float32),
        uav_pos_t=uav_pos_t.to(dtype=torch.float32),
        candidate_slot_ids_t=slot_ids_t,
        candidate_gu_ids_t=candidate_gu_ids_t,
        candidate_uav_ids_t=candidate_uav_ids_t,
        out_idx=stage_out.candidate_indices,
        out_mask=stage_out.candidate_mask,
    )
    candidate_indices_t = stage_out.candidate_indices
    candidate_mask_t = stage_out.candidate_mask
    _candidate_flag_bundle_from_index_mask_tensor_impl(
        candidate_params=candidate_params,
        assoc_t=assoc_t,
        candidate_indices_t=candidate_indices_t,
        candidate_mask_t=candidate_mask_t,
        prev_association_t=prev_association_t.to(dtype=torch.int32),
        candidate_uav_ids_t=candidate_uav_ids_t,
        bw_valid_slots_out=stage_out.bw_valid_mask,
        candidate_flag_out=stage_out.candidate_flag,
        bw_valid_flag_out=stage_out.bw_valid_flag,
        prev_assoc_flag_out=stage_out.prev_assoc_flag,
    )
    bw_valid_mask_t = stage_out.bw_valid_mask
    candidate_flag_t = stage_out.candidate_flag
    bw_valid_flag_t = stage_out.bw_valid_flag
    prev_assoc_flag_t = stage_out.prev_assoc_flag

    sat_selection_matrix_t = stage_out.sat_selection_matrix
    sat_selection_matrix_t.fill_(-1)
    reward_aligned_t, uav_assoc_uav_cost_t, sat_cost_norm_t = _reward_aligned_feature_bundle_batch_from_state_tensor_impl(
        bw_workload_static_params=bw_workload_static_params,
        gu_queue_t=gu_queue_t.to(dtype=torch.float32),
        arrival_ref_t=arrival_ref_t.to(dtype=torch.float32),
        gu_ema_t=gu_ema_t.to(dtype=torch.float32),
        uav_ema_t=uav_ema_t.to(dtype=torch.float32),
        sat_ema_t=sat_ema_t.to(dtype=torch.float32),
        assoc_t=assoc_t.to(dtype=torch.long),
        sat_selection_matrix_t=sat_selection_matrix_t,
    )
    gu_proxy_features_t = stage_out.gu_proxy_features
    _gu_proxy_feature_arrays_batch_from_state_tensor_impl(
        local_obs_params=local_obs_params,
        gu_queue_t=gu_queue_t.to(dtype=torch.float32),
        arrival_rate_vec_t=expected_arrival_rate_vec_t.to(dtype=torch.float32),
        recent_arrival_t=last_gu_arrival_t.to(dtype=torch.float32),
        recent_service_t=last_gu_outflow_t.to(dtype=torch.float32),
        urgency_risk_t=last_gu_urgency_risk_t.to(dtype=torch.float32),
        downstream_pressure_t=last_gu_downstream_pressure_t.to(dtype=torch.float32),
        service_gap_t=last_gu_service_gap_t.to(dtype=torch.float32),
        service_gap_risk_t=last_gu_service_gap_risk_t.to(dtype=torch.float32),
        deadline_steps_t=gu_deadline_steps_t.to(dtype=torch.float32),
        deadline_slack_t=last_gu_deadline_slack_t.to(dtype=torch.float32),
        deadline_risk_t=last_gu_deadline_risk_t.to(dtype=torch.float32),
        reward_aligned=reward_aligned_t,
        out_t=gu_proxy_features_t,
    )

    access_gain_input_t = access_base_gain_t.to(dtype=torch.float32)
    if int(access_rate_static_params.fading_mode_code) == 2:
        access_gain_input_t = access_gain_input_t * fading_gain_t.to(dtype=torch.float32)
    gain_matrix_t = _quantize_access_gain_snapshot_tensor(
        access_gain_input_t,
        quantum=float(channel_params.access_gain_quantum),
        out_dtype=torch.float32,
    )
    eta_slots_t = _apply_batched_access_eta_slots_tensor_impl(
        gain_matrix_t=gain_matrix_t,
        assoc_t=assoc_t.to(dtype=torch.long),
        candidate_indices_t=candidate_indices_t.to(dtype=torch.long),
        candidate_mask_t=candidate_mask_t.to(dtype=torch.bool),
        params=access_rate_static_params,
        candidate_uav_ids_t=candidate_uav_ids_t,
    )
    _copy_tensor_out_(stage_out.eta_slots, eta_slots_t.to(dtype=torch.float32))
    eta_feature_t = stage_out.eta_ref_feature
    eta_feature_t.zero_()
    eta_feature_t.scatter_add_(
        2,
        torch.clamp(candidate_indices_t.to(dtype=torch.long), min=0),
        torch.where(
            candidate_mask_t.to(dtype=torch.bool),
            eta_slots_t.to(dtype=torch.float32),
            eta_slots_t.to(dtype=torch.float32) * 0.0,
        ),
    )

    sat_pos_base_f_t = sat_pos_base_t.to(device=device, dtype=torch.float32)
    sat_vel_base_f_t = sat_vel_base_t.to(device=device, dtype=torch.float32)
    sat_pos_t = (
        sat_pos_base_f_t
        if sat_pos_base_f_t.ndim == 3
        else sat_pos_base_f_t.unsqueeze(0).expand(batch_size, -1, -1)
    )
    sat_vel_t = (
        sat_vel_base_f_t
        if sat_vel_base_f_t.ndim == 3
        else sat_vel_base_f_t.unsqueeze(0).expand(batch_size, -1, -1)
    )
    full_rel_pos_t: torch.Tensor | None = None
    full_rel_vel_t: torch.Tensor | None = None
    full_gain_t: torch.Tensor | None = None
    full_nu_eff_t: torch.Tensor | None = None
    full_valid_flag_t: torch.Tensor | None = None
    full_sat_queue_feature_t: torch.Tensor | None = None
    candidate_mode_l = str(sat_p.sat_candidate_mode)
    if candidate_mode_l == "elevation":
        geom_out = _prepare_full_sat_geometry_tensor_impl(
            channel_params=channel_params,
            sat_geometry_params=sat_geometry_params,
            uav_pos_t=uav_pos_t.to(dtype=torch.float32),
            uav_vel_t=uav_vel_t.to(dtype=torch.float32),
            sat_pos_base_t=sat_pos_base_t.to(device=device, dtype=torch.float32),
            sat_vel_base_t=sat_vel_base_t.to(device=device, dtype=torch.float32),
            sat_queue_t=sat_queue_t.to(dtype=torch.float32),
            sat_load_t=last_sat_connection_counts_t.to(dtype=torch.float32),
            doppler_residual_t=doppler_residual_t.to(dtype=torch.float32),
            sat_orbit_radius_sq=float(sat_orbit_radius_sq),
            uav_orbit_radius_sq=float(uav_orbit_radius_sq),
            uav_orbit_radius=float(uav_orbit_radius),
            backhaul_gain_const=float(backhaul_gain_const),
            sat_all_ids_t=sat_all_ids_t,
            uav_ecef_out_t=stage_out.uav_ecef_all,
            uav_vel_ecef_out_t=stage_out.uav_vel_ecef_all,
        )
        visible_ids_t = geom_out.top_idx.to(dtype=torch.long)
        visible_mask_t = geom_out.top_mask.to(dtype=torch.bool)
        active_ids_t = geom_out.active_ids.to(dtype=torch.long)
        visible_flag_t = geom_out.visible_flag.to(dtype=torch.float32)
        elev_t = geom_out.elevation.to(dtype=torch.float32)
        uav_ecef_t = geom_out.uav_ecef.to(dtype=torch.float32)
        uav_vel_ecef_t = geom_out.uav_vel_ecef.to(dtype=torch.float32)
        sat_pos_active_t = geom_out.sat_pos_active.to(dtype=torch.float32)
        sat_vel_active_t = geom_out.sat_vel_active.to(dtype=torch.float32)
        sat_queue_active_t = geom_out.sat_queue_active.to(dtype=torch.float32)
        sat_load_active_t = geom_out.sat_load_active.to(dtype=torch.float32)
        us_rel_pos_active_t = geom_out.us_rel_pos_active.to(dtype=torch.float32)
        us_rel_vel_active_t = geom_out.us_rel_vel_active.to(dtype=torch.float32)
        us_gain_active_t = geom_out.us_gain_active.to(dtype=torch.float32)
        us_nu_eff_active_t = geom_out.us_nu_eff_active.to(dtype=torch.float32)
        visible_flag_active_t = geom_out.visible_flag_active.to(dtype=torch.float32)
        us_valid_flag_active_t = geom_out.us_valid_flag_active.to(dtype=torch.float32)
        active_mask_t = active_ids_t >= 0
        if int(active_ids_t.shape[1]) > 0:
            active_safe_t = torch.clamp(active_ids_t, min=0, max=max(num_sat - 1, 0))
            sat_cost_norm_active_t = torch.gather(
                sat_cost_norm_t.to(dtype=torch.float32),
                1,
                active_safe_t,
            ) * active_mask_t.to(dtype=torch.float32)
        else:
            sat_cost_norm_active_t = stage_out.sat_cost_norm_active.zero_()
    else:
        uav_ecef_t, uav_vel_ecef_t = _compute_uav_cache_tensors_impl(
            uav_pos_t=uav_pos_t.to(dtype=torch.float32),
            uav_vel_t=uav_vel_t.to(dtype=torch.float32),
            sat_geometry_params=sat_p,
            uav_ecef_out_t=stage_out.uav_ecef_all,
            uav_vel_ecef_out_t=stage_out.uav_vel_ecef_all,
        )
        visible_out = _visible_sats_batch_tensor_impl(
            sat_pos_t=sat_pos_t,
            uav_ecef_t=uav_ecef_t,
            sat_queue_t=sat_queue_t.to(dtype=torch.float32),
            sat_load_t=last_sat_connection_counts_t.to(dtype=torch.float32),
            current_sel_mask_t=(
                _sat_selection_presence_tensor_impl(
                    last_sat_selection_matrix_t.to(device=device, dtype=torch.long),
                    num_sat=num_sat,
                )
                > 0.5
            ),
            channel_params=channel_params,
            sat_geometry_params=sat_geometry_params,
            sat_orbit_radius_sq=float(sat_orbit_radius_sq),
            uav_orbit_radius_sq=float(uav_orbit_radius_sq),
            uav_orbit_radius=float(uav_orbit_radius),
            backhaul_gain_const=float(backhaul_gain_const),
            effective_b_backhaul_per_sat=float(effective_b_backhaul_per_sat),
        )
        elev_t = visible_out.elevation.to(dtype=torch.float32)
        above_t = visible_out.above_mask.to(dtype=torch.bool)
        max_keep = max(int(sat_p.visible_sats_max), 0)
        if max_keep > 0:
            visible_ids_t, visible_mask_t = _masked_topk_sat_ids_desc(
                visible_out.score.to(dtype=torch.float32),
                above_t,
                k=max_keep,
            )
        else:
            visible_ids_t = stage_out.visible_ids
            visible_ids_t.fill_(-1)
            visible_mask_t = stage_out.visible_mask
            visible_mask_t.zero_()
        visible_flag_t = stage_out.visible_flag_all
        visible_flag_t.zero_()
        if max_keep > 0:
            visible_flag_t.scatter_(2, visible_ids_t, visible_mask_t.to(dtype=torch.float32))
        active_width = min(num_sat, max_keep * num_uav)
        if active_width > 0:
            active_ids_t = _active_sat_ids_from_visible_ids_tensor(
                visible_ids_t,
                visible_mask_t,
                active_width=active_width,
                num_sat=num_sat,
            )
            active_mask_t = active_ids_t >= 0
        else:
            active_ids_t = stage_out.active_sat_ids
            active_ids_t.fill_(-1)
            active_mask_t = active_ids_t >= 0
        full_rel_pos_t = sat_pos_t[:, None, :, :] - uav_ecef_t[:, :, None, :]
        full_rel_vel_t = sat_vel_t[:, None, :, :] - uav_vel_ecef_t[:, :, None, :]
        dist_t = _torch_positive(torch.linalg.vector_norm(full_rel_pos_t, dim=-1), GEOMETRY_DENOM_EPS)
        full_gain_t = (float(backhaul_gain_const) / _torch_positive(dist_t * dist_t, GEOMETRY_DENOM_EPS)).to(dtype=torch.float32)
        loss_t = _backhaul_loss_factor_torch(
            channel_params=channel_params,
            carrier_freq_hz=float(sat_p.carrier_freq),
            elevation_batch_t=elev_t,
        )
        if loss_t is not None:
            full_gain_t = full_gain_t * loss_t.to(dtype=torch.float32)
        raw_nu_t = (
            (float(sat_p.carrier_freq) / float(sat_p.speed_of_light))
            * torch.sum(full_rel_vel_t * full_rel_pos_t, dim=-1)
            / _torch_positive(dist_t, GEOMETRY_DENOM_EPS)
        ).to(dtype=torch.float32)
        if bool(sat_p.doppler_enabled) or bool(sat_p.doppler_atten_enabled) or bool(sat_p.doppler_observed):
            if bool(sat_p.doppler_precomp_enabled):
                full_nu_eff_t = doppler_residual_t.to(dtype=torch.float32)
            else:
                full_nu_eff_t = raw_nu_t
        else:
            full_nu_eff_t = raw_nu_t * 0.0
        full_valid_flag_t = above_t
        if bool(sat_p.doppler_enabled):
            full_valid_flag_t = full_valid_flag_t & (torch.abs(full_nu_eff_t) <= float(sat_p.nu_max))
        full_sat_queue_feature_t = (
            sat_queue_t.to(dtype=torch.float32) / normalize_scale(float(sat_p.queue_max_sat))
        )[:, None, :].expand(-1, num_uav, -1)
        active_mask_t = active_ids_t >= 0
        active_mask_float_t = active_mask_t.to(dtype=torch.float32)
        active_safe_t = torch.clamp(active_ids_t, min=0, max=max(num_sat - 1, 0))
        if active_width > 0:
            sat_feature_idx_t = active_safe_t.unsqueeze(-1).expand(-1, -1, 3)
            us_feature_idx_t = active_safe_t[:, None, :, None].expand(-1, num_uav, -1, 3)
            us_scalar_idx_t = active_safe_t[:, None, :].expand(-1, num_uav, -1)
            sat_pos_active_t = torch.gather(sat_pos_t, 1, sat_feature_idx_t).to(dtype=torch.float32)
            sat_vel_active_t = torch.gather(sat_vel_t, 1, sat_feature_idx_t).to(dtype=torch.float32)
            sat_queue_active_t = torch.gather(sat_queue_t.to(dtype=torch.float32), 1, active_safe_t)
            sat_load_active_t = torch.gather(
                last_sat_connection_counts_t.to(dtype=torch.float32),
                1,
                active_safe_t,
            )
            sat_cost_norm_active_t = torch.gather(sat_cost_norm_t.to(dtype=torch.float32), 1, active_safe_t)
            us_rel_pos_active_t = torch.gather(full_rel_pos_t, 2, us_feature_idx_t).to(dtype=torch.float32)
            us_rel_vel_active_t = torch.gather(full_rel_vel_t, 2, us_feature_idx_t).to(dtype=torch.float32)
            us_gain_active_t = torch.gather(full_gain_t, 2, us_scalar_idx_t).to(dtype=torch.float32)
            us_nu_eff_active_t = torch.gather(full_nu_eff_t, 2, us_scalar_idx_t).to(dtype=torch.float32)
            visible_flag_active_t = torch.gather(visible_flag_t, 2, us_scalar_idx_t).to(dtype=torch.float32)
            us_valid_flag_active_t = torch.gather(full_valid_flag_t.to(dtype=torch.float32), 2, us_scalar_idx_t)
            sat_pos_active_t = sat_pos_active_t * active_mask_float_t.unsqueeze(-1)
            sat_vel_active_t = sat_vel_active_t * active_mask_float_t.unsqueeze(-1)
            sat_queue_active_t = sat_queue_active_t * active_mask_float_t
            sat_load_active_t = sat_load_active_t * active_mask_float_t
            sat_cost_norm_active_t = sat_cost_norm_active_t * active_mask_float_t
            us_active_mask_t = active_mask_float_t[:, None, :]
            us_rel_pos_active_t = us_rel_pos_active_t * us_active_mask_t.unsqueeze(-1)
            us_rel_vel_active_t = us_rel_vel_active_t * us_active_mask_t.unsqueeze(-1)
            us_gain_active_t = us_gain_active_t * us_active_mask_t
            us_nu_eff_active_t = us_nu_eff_active_t * us_active_mask_t
            visible_flag_active_t = visible_flag_active_t * us_active_mask_t
            us_valid_flag_active_t = us_valid_flag_active_t * us_active_mask_t
        else:
            sat_pos_active_t = stage_out.sat_pos_active.zero_()
            sat_vel_active_t = stage_out.sat_vel_active.zero_()
            sat_queue_active_t = stage_out.sat_queue_active.zero_()
            sat_load_active_t = stage_out.sat_load_active.zero_()
            sat_cost_norm_active_t = stage_out.sat_cost_norm_active.zero_()
            us_rel_pos_active_t = stage_out.us_rel_pos_active.zero_()
            us_rel_vel_active_t = stage_out.us_rel_vel_active.zero_()
            us_gain_active_t = stage_out.us_gain_active.zero_()
            us_nu_eff_active_t = stage_out.us_nu_eff_active.zero_()
            visible_flag_active_t = stage_out.visible_flag_active.zero_()
            us_valid_flag_active_t = stage_out.us_valid_flag_active.zero_()

    stage_out.stage_id.fill_(int(StructuredControlDriver.STAGE_ACCEL))
    stage_out.effective_b_backhaul_per_sat.fill_(float(effective_b_backhaul_per_sat))
    _copy_tensor_out_(stage_out.uav_pos, uav_pos_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_vel, uav_vel_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_energy, uav_energy_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_queue, uav_queue_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.gu_pos, gu_pos_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.gu_queue, gu_queue_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.arrival_ref_bits_per_step, arrival_ref_t.to(dtype=torch.float32).reshape(batch_size))
    _copy_tensor_out_(stage_out.expected_arrival_rate_vec, expected_arrival_rate_vec_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.gu_ema, gu_ema_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_ema, uav_ema_t.to(dtype=torch.float32))
    if torch.is_tensor(gu_drop_t):
        _copy_tensor_out_(stage_out.gu_drop, gu_drop_t.to(dtype=torch.float32))
    else:
        stage_out.gu_drop.zero_()
    if torch.is_tensor(uav_drop_t):
        _copy_tensor_out_(stage_out.uav_drop, uav_drop_t.to(dtype=torch.float32))
    else:
        stage_out.uav_drop.zero_()
    _copy_tensor_out_(stage_out.last_gu_arrival, last_gu_arrival_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.last_gu_outflow, last_gu_outflow_t.to(dtype=torch.float32))
    if torch.is_tensor(last_gu_to_uav_inflow_by_uav_t):
        _copy_tensor_out_(stage_out.last_gu_to_uav_inflow_by_uav, last_gu_to_uav_inflow_by_uav_t.to(dtype=torch.float32))
    else:
        stage_out.last_gu_to_uav_inflow_by_uav.zero_()
    if torch.is_tensor(last_uav_to_sat_outflow_matrix_t):
        _copy_tensor_out_(stage_out.last_uav_to_sat_outflow_matrix, last_uav_to_sat_outflow_matrix_t.to(dtype=torch.float32))
    else:
        stage_out.last_uav_to_sat_outflow_matrix.zero_()
    if torch.is_tensor(last_bw_fraction_by_uav_gu_t):
        _copy_tensor_out_(stage_out.last_bw_fraction_by_uav_gu, last_bw_fraction_by_uav_gu_t.to(dtype=torch.float32))
    else:
        stage_out.last_bw_fraction_by_uav_gu.zero_()
    if torch.is_tensor(last_access_interference_by_uav_t):
        _copy_tensor_out_(stage_out.last_access_interference_by_uav, last_access_interference_by_uav_t.to(dtype=torch.float32))
    else:
        stage_out.last_access_interference_by_uav.zero_()
    _copy_tensor_out_(stage_out.sat_queue, sat_queue_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_loads, last_sat_connection_counts_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_ema, sat_ema_t.to(dtype=torch.float32))
    if torch.is_tensor(sat_drop_t):
        _copy_tensor_out_(stage_out.sat_drop, sat_drop_t.to(dtype=torch.float32))
    else:
        stage_out.sat_drop.zero_()
    if torch.is_tensor(last_sat_processed_t):
        _copy_tensor_out_(stage_out.last_sat_processed, last_sat_processed_t.to(dtype=torch.float32))
    else:
        stage_out.last_sat_processed.zero_()
    if torch.is_tensor(last_selected_mask_by_uav_sat_t):
        _copy_tensor_out_(stage_out.last_selected_mask_by_uav_sat, last_selected_mask_by_uav_sat_t.to(dtype=torch.float32))
    else:
        _copy_tensor_out_(
            stage_out.last_selected_mask_by_uav_sat,
            _sat_selection_presence_tensor_impl(
                last_sat_selection_matrix_t.to(device=device, dtype=torch.long),
                num_sat=num_sat,
            ).to(dtype=torch.float32),
        )
    _copy_tensor_out_(stage_out.sat_pos, sat_pos_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_vel, sat_vel_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.assoc, assoc_t.to(dtype=torch.long))
    _copy_tensor_out_(stage_out.prev_association, prev_association_t.to(dtype=torch.long))
    _copy_tensor_out_(stage_out.uav_assoc_uav_cost, uav_assoc_uav_cost_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_cost_norm, sat_cost_norm_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.access_gain_matrix, gain_matrix_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.visible_ids, visible_ids_t.to(dtype=torch.long))
    _copy_tensor_out_(stage_out.visible_mask, visible_mask_t.to(dtype=torch.bool))
    _copy_tensor_out_(stage_out.visible_flag_all, visible_flag_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.elevation_matrix, elev_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_ecef_all, uav_ecef_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.uav_vel_ecef_all, uav_vel_ecef_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.active_sat_ids, active_ids_t.to(dtype=torch.long))
    _copy_tensor_out_(stage_out.sat_pos_active, sat_pos_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_vel_active, sat_vel_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_queue_active, sat_queue_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_load_active, sat_load_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.sat_cost_norm_active, sat_cost_norm_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.us_rel_pos_active, us_rel_pos_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.us_rel_vel_active, us_rel_vel_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.us_gain_active, us_gain_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.us_nu_eff_active, us_nu_eff_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.visible_flag_active, visible_flag_active_t.to(dtype=torch.float32))
    _copy_tensor_out_(stage_out.us_valid_flag_active, us_valid_flag_active_t.to(dtype=torch.float32))
    for target_t, source_t in (
        (stage_out.us_rel_pos_all, full_rel_pos_t),
        (stage_out.us_rel_vel_all, full_rel_vel_t),
        (stage_out.us_gain_all, full_gain_t),
        (stage_out.us_nu_eff_all, full_nu_eff_t),
        (stage_out.us_valid_flag_all, full_valid_flag_t),
        (stage_out.us_sat_queue_all, full_sat_queue_feature_t),
    ):
        if torch.is_tensor(target_t):
            if torch.is_tensor(source_t):
                _copy_tensor_out_(target_t, source_t.to(dtype=target_t.dtype))
            else:
                target_t.zero_()
    return None


def _prepare_full_sat_geometry_tensor_impl(
    *,
    channel_params: _NativeChannelStaticParams,
    sat_geometry_params: _NativeSatGeometryStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    sat_pos_base_t: torch.Tensor,
    sat_vel_base_t: torch.Tensor,
    sat_queue_t: torch.Tensor,
    sat_load_t: torch.Tensor,
    doppler_residual_t: torch.Tensor,
    sat_orbit_radius_sq: float,
    uav_orbit_radius_sq: float,
    uav_orbit_radius: float,
    backhaul_gain_const: float,
    sat_all_ids_t: torch.Tensor,
    uav_ecef_out_t: torch.Tensor,
    uav_vel_ecef_out_t: torch.Tensor,
) -> _NativePreparedSatGeometryTensorFields:
    sat_p = sat_geometry_params
    batch_size = int(uav_pos_t.shape[0])
    num_uav = int(sat_p.num_uav)
    num_sat = int(sat_p.num_sat)
    device = uav_pos_t.device
    uav_ecef_t, uav_vel_ecef_t = _compute_uav_cache_tensors_impl(
        uav_pos_t=uav_pos_t,
        uav_vel_t=uav_vel_t,
        sat_geometry_params=sat_p,
        uav_ecef_out_t=uav_ecef_out_t,
        uav_vel_ecef_out_t=uav_vel_ecef_out_t,
    )
    sat_pos_base_f_t = sat_pos_base_t.to(dtype=torch.float32)
    sat_vel_base_f_t = sat_vel_base_t.to(dtype=torch.float32)
    sat_pos_group_t = (
        sat_pos_base_f_t
        if sat_pos_base_f_t.ndim == 3
        else sat_pos_base_f_t.unsqueeze(0).expand(batch_size, -1, -1)
    )
    sat_vel_group_t = (
        sat_vel_base_f_t
        if sat_vel_base_f_t.ndim == 3
        else sat_vel_base_f_t.unsqueeze(0).expand(batch_size, -1, -1)
    )
    elev_t = _elevation_matrix_from_positions_tensor_impl(
        sat_pos_t=sat_pos_group_t,
        uav_ecef_t=uav_ecef_t,
        sat_orbit_radius_sq=sat_orbit_radius_sq,
        uav_orbit_radius_sq=uav_orbit_radius_sq,
        uav_orbit_radius=uav_orbit_radius,
    ).to(dtype=torch.float32)
    above_t = elev_t >= float(sat_p.theta_min_rad)
    max_keep = max(int(sat_p.visible_sats_max), 0)
    if max_keep > 0:
        top_idx_t, top_mask_t = _masked_topk_sat_ids_desc(elev_t, above_t, k=max_keep)
    else:
        top_idx_t = sat_queue_t[:, None, :0].expand(batch_size, num_uav, 0).to(dtype=torch.long)
        top_mask_t = above_t[:, :, :0]
    visible_flag_t = elev_t * 0.0
    if max_keep > 0:
        visible_flag_t.scatter_(2, top_idx_t, top_mask_t.to(dtype=torch.float32))

    active_width = min(num_sat, max_keep * num_uav)
    if active_width > 0:
        active_ids_t = _active_sat_ids_from_visible_ids_tensor(
            top_idx_t,
            top_mask_t,
            active_width=active_width,
            num_sat=num_sat,
        )
        active_mask_t = active_ids_t >= 0
    else:
        active_ids_t = top_idx_t[:, 0, :0]
        active_mask_t = active_ids_t >= 0

    active_safe_t = torch.clamp(active_ids_t, min=0, max=max(num_sat - 1, 0))
    active_xyz_idx_t = active_safe_t.unsqueeze(-1).expand(-1, -1, 3)
    active_scalar_idx_t = active_safe_t[:, None, :].expand(-1, num_uav, -1)
    active_mask_float_t = active_mask_t.to(dtype=torch.float32)
    us_active_mask_float_t = active_mask_float_t[:, None, :]
    sat_pos_active_t = torch.gather(sat_pos_group_t, 1, active_xyz_idx_t).to(dtype=torch.float32)
    sat_vel_active_t = torch.gather(sat_vel_group_t, 1, active_xyz_idx_t).to(dtype=torch.float32)
    sat_queue_active_t = torch.gather(sat_queue_t, 1, active_safe_t).to(dtype=torch.float32)
    sat_load_active_t = torch.gather(sat_load_t, 1, active_safe_t).to(dtype=torch.float32)
    elev_active_t = torch.gather(elev_t, 2, active_scalar_idx_t).to(dtype=torch.float32)
    visible_flag_active_t = torch.gather(visible_flag_t, 2, active_scalar_idx_t).to(dtype=torch.float32)
    rel_pos_active_t = sat_pos_active_t[:, None, :, :] - uav_ecef_t[:, :, None, :]
    rel_vel_active_t = sat_vel_active_t[:, None, :, :] - uav_vel_ecef_t[:, :, None, :]
    dist_active_t = _torch_positive(torch.linalg.vector_norm(rel_pos_active_t, dim=-1), GEOMETRY_DENOM_EPS)
    gain_active_t = (
        float(backhaul_gain_const)
        / _torch_positive(dist_active_t * dist_active_t, GEOMETRY_DENOM_EPS)
    ).to(dtype=torch.float32)
    loss_active_t = _backhaul_loss_factor_torch(
        channel_params=channel_params,
        carrier_freq_hz=float(sat_p.carrier_freq),
        elevation_batch_t=elev_active_t,
    )
    if loss_active_t is not None:
        gain_active_t = gain_active_t * loss_active_t.to(dtype=torch.float32)
    raw_nu_active_t = (
        (float(sat_p.carrier_freq) / float(sat_p.speed_of_light))
        * torch.sum(rel_vel_active_t * rel_pos_active_t, dim=-1)
        / _torch_positive(dist_active_t, GEOMETRY_DENOM_EPS)
    ).to(dtype=torch.float32)
    if bool(sat_p.doppler_enabled) or bool(sat_p.doppler_atten_enabled) or bool(sat_p.doppler_observed):
        if bool(sat_p.doppler_precomp_enabled):
            nu_eff_active_t = torch.gather(doppler_residual_t.to(dtype=torch.float32), 2, active_scalar_idx_t)
        else:
            nu_eff_active_t = raw_nu_active_t
    else:
        nu_eff_active_t = raw_nu_active_t * 0.0
    valid_flag_active_t = visible_flag_active_t > 0.5
    if bool(sat_p.doppler_enabled):
        valid_flag_active_t = valid_flag_active_t & (torch.abs(nu_eff_active_t) <= float(sat_p.nu_max))

    return _NativePreparedSatGeometryTensorFields(
        top_idx=top_idx_t,
        top_mask=top_mask_t,
        active_ids=active_ids_t,
        visible_flag=visible_flag_t,
        elevation=elev_t,
        uav_ecef=uav_ecef_t.to(dtype=torch.float32),
        uav_vel_ecef=uav_vel_ecef_t.to(dtype=torch.float32),
        sat_pos_active=sat_pos_active_t * active_mask_float_t.unsqueeze(-1),
        sat_vel_active=sat_vel_active_t * active_mask_float_t.unsqueeze(-1),
        sat_queue_active=sat_queue_active_t * active_mask_float_t,
        sat_load_active=sat_load_active_t * active_mask_float_t,
        us_rel_pos_active=rel_pos_active_t * us_active_mask_float_t.unsqueeze(-1),
        us_rel_vel_active=rel_vel_active_t * us_active_mask_float_t.unsqueeze(-1),
        us_gain_active=gain_active_t * us_active_mask_float_t,
        us_nu_eff_active=nu_eff_active_t * us_active_mask_float_t,
        visible_flag_active=visible_flag_active_t * us_active_mask_float_t,
        us_valid_flag_active=valid_flag_active_t.to(dtype=torch.float32) * us_active_mask_float_t,
    )


def _build_global_state_tensor_impl(
    *,
    global_state_params: _NativeGlobalStateStaticParams,
    sat_geometry_params: _NativeSatGeometryStaticParams,
    bw_workload_static_params: _NativeBwWorkloadStaticParams,
    local_obs_params: _NativeLocalObsStaticParams,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    uav_queue_t: torch.Tensor,
    uav_energy_t: torch.Tensor,
    gu_pos_t: torch.Tensor,
    gu_queue_t: torch.Tensor,
    sat_pos_t: torch.Tensor,
    sat_vel_t: torch.Tensor,
    sat_queue_t: torch.Tensor,
    t_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    gu_ema_t: torch.Tensor,
    uav_ema_t: torch.Tensor,
    sat_ema_t: torch.Tensor,
    assoc_t: torch.Tensor,
    sat_selection_matrix_t: torch.Tensor,
    arrival_rate_vec_t: torch.Tensor,
    recent_arrival_t: torch.Tensor,
    recent_service_t: torch.Tensor,
    urgency_risk_t: torch.Tensor,
    downstream_pressure_t: torch.Tensor,
    service_gap_t: torch.Tensor,
    service_gap_risk_t: torch.Tensor,
    deadline_steps_t: torch.Tensor,
    deadline_slack_t: torch.Tensor,
    deadline_risk_t: torch.Tensor,
    sat_state_max: int | None,
) -> torch.Tensor:
    batch_size = int(uav_pos_t.shape[0])
    if batch_size <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=uav_pos_t.device)
    global_p = global_state_params
    map_scale = normalize_scale(float(global_p.map_size))
    v_scale = normalize_scale(float(global_p.v_max))
    sat_orbit_scale = normalize_scale(float(global_p.r_earth + global_p.sat_height))

    sat_pos_view_t = sat_pos_t
    sat_vel_view_t = sat_vel_t
    sat_queue_view_t = sat_queue_t
    if sat_state_max is not None and int(sat_state_max) < int(global_p.num_sat):
        uav_ecef_t, _uav_vel_ecef_t = _compute_uav_cache_tensors_impl(
            uav_pos_t=uav_pos_t,
            uav_vel_t=uav_vel_t,
            sat_geometry_params=sat_geometry_params,
            uav_ecef_out_t=uav_pos_t.new_empty((*uav_pos_t.shape[:-1], 3), dtype=torch.float32),
            uav_vel_ecef_out_t=uav_vel_t.new_empty((*uav_vel_t.shape[:-1], 3), dtype=torch.float32),
        )
        elev_t = _elevation_matrix_from_positions_tensor_impl(
            sat_pos_t=sat_pos_t,
            uav_ecef_t=uav_ecef_t,
            sat_orbit_radius_sq=float(global_p.r_earth + global_p.sat_height) ** 2,
            uav_orbit_radius_sq=float(global_p.r_earth + global_p.uav_height) ** 2,
            uav_orbit_radius=float(global_p.r_earth + global_p.uav_height),
        )
        sat_score_t = torch.amax(elev_t, dim=1)
        sat_idx_t, _sat_idx_mask_t = _masked_topk_sat_ids_desc(
            sat_score_t,
            torch.ones_like(sat_score_t, dtype=torch.bool),
            k=int(sat_state_max),
        )
        sat_idx_xyz_t = sat_idx_t.unsqueeze(-1).expand(-1, -1, 3)
        sat_pos_view_t = sat_pos_t.gather(1, sat_idx_xyz_t)
        sat_vel_view_t = sat_vel_t.gather(1, sat_idx_xyz_t)
        sat_queue_view_t = sat_queue_t.gather(1, sat_idx_t)

    reward_aligned_t, _uav_assoc_uav_cost_t, _sat_cost_norm_t = _reward_aligned_feature_bundle_batch_from_state_tensor_impl(
        bw_workload_static_params=bw_workload_static_params,
        gu_queue_t=gu_queue_t,
        arrival_ref_t=arrival_ref_t,
        gu_ema_t=gu_ema_t,
        uav_ema_t=uav_ema_t,
        sat_ema_t=sat_ema_t,
        assoc_t=assoc_t,
        sat_selection_matrix_t=sat_selection_matrix_t,
    )
    gu_proxy_feature_dim = _native_local_obs_user_proxy_feature_dim(local_obs_params)
    gu_proxy_t = gu_queue_t.new_empty((batch_size, int(global_p.num_gu), int(gu_proxy_feature_dim)), dtype=torch.float32)
    _gu_proxy_feature_arrays_batch_from_state_tensor_impl(
        local_obs_params=local_obs_params,
        gu_queue_t=gu_queue_t,
        arrival_rate_vec_t=arrival_rate_vec_t,
        recent_arrival_t=recent_arrival_t,
        recent_service_t=recent_service_t,
        urgency_risk_t=urgency_risk_t,
        downstream_pressure_t=downstream_pressure_t,
        service_gap_t=service_gap_t,
        service_gap_risk_t=service_gap_risk_t,
        deadline_steps_t=deadline_steps_t,
        deadline_slack_t=deadline_slack_t,
        deadline_risk_t=deadline_risk_t,
        reward_aligned=reward_aligned_t,
        out_t=gu_proxy_t,
    )
    parts_t = [
        uav_pos_t.reshape(batch_size, -1) / map_scale,
        uav_vel_t.reshape(batch_size, -1) / v_scale,
        uav_queue_t / normalize_scale(float(global_p.queue_max_uav)),
        uav_energy_t / normalize_scale(float(global_p.uav_energy_init)),
        gu_pos_t.reshape(batch_size, -1) / map_scale,
        gu_queue_t / normalize_scale(float(global_p.queue_max_gu)),
        sat_pos_view_t.reshape(batch_size, -1) / sat_orbit_scale,
        sat_vel_view_t.reshape(batch_size, -1) / sat_orbit_scale,
        sat_queue_view_t.reshape(batch_size, -1) / normalize_scale(float(global_p.queue_max_sat)),
        (t_t.reshape(batch_size).to(dtype=torch.float32) / max(float(global_p.t_steps), 1.0)).unsqueeze(-1),
    ]
    if int(gu_proxy_feature_dim) > 0:
        parts_t.append(gu_proxy_t.reshape(batch_size, -1).to(dtype=torch.float32))
    return torch.cat(parts_t, dim=1).to(dtype=torch.float32)


def _build_world_from_packed_specs_tensor_impl(
    *,
    local_obs_params: _NativeLocalObsStaticParams,
    stage_ids_t: torch.Tensor | None = None,
    effective_b_backhaul_per_sat_t: torch.Tensor,
    uav_pos_t: torch.Tensor,
    uav_vel_t: torch.Tensor,
    uav_energy_t: torch.Tensor,
    uav_queue_t: torch.Tensor,
    uav_assoc_uav_cost_t: torch.Tensor,
    gu_pos_t: torch.Tensor,
    gu_queue_t: torch.Tensor,
    gu_proxy_features_t: torch.Tensor,
    assoc_t: torch.Tensor,
    candidate_flag_t: torch.Tensor,
    bw_valid_flag_t: torch.Tensor,
    prev_assoc_flag_t: torch.Tensor,
    eta_ref_feature_t: torch.Tensor,
    sat_pos_active_t: torch.Tensor,
    sat_vel_active_t: torch.Tensor,
    sat_queue_active_t: torch.Tensor,
    sat_load_active_t: torch.Tensor,
    sat_cost_norm_active_t: torch.Tensor,
    sat_active_mask_t: torch.Tensor,
    rel_pos_active_t: torch.Tensor,
    rel_vel_active_t: torch.Tensor,
    gain_active_t: torch.Tensor,
    nu_eff_active_t: torch.Tensor,
    visible_flag_active_t: torch.Tensor,
    valid_flag_active_t: torch.Tensor,
    current_sel_flag_active_t: torch.Tensor,
    t_t: torch.Tensor | None = None,
    history_world_out: _NativeTrainingWorldTensorFields | None = None,
    history_row_indices_t: torch.Tensor | None = None,
    uav_index_t: torch.Tensor | None = None,
    offdiag_mask_t: torch.Tensor | None = None,
    active_sat_ids_t: torch.Tensor | None = None,
    arrival_ref_t: torch.Tensor | None = None,
    expected_arrival_rate_vec_t: torch.Tensor | None = None,
    gu_ema_t: torch.Tensor | None = None,
    uav_ema_t: torch.Tensor | None = None,
    sat_ema_active_t: torch.Tensor | None = None,
    last_gu_arrival_t: torch.Tensor | None = None,
    last_gu_outflow_t: torch.Tensor | None = None,
    gu_drop_t: torch.Tensor | None = None,
    uav_drop_t: torch.Tensor | None = None,
    sat_drop_active_t: torch.Tensor | None = None,
    last_gu_to_uav_inflow_by_uav_t: torch.Tensor | None = None,
    last_uav_to_sat_outflow_active_t: torch.Tensor | None = None,
    last_bw_fraction_by_uav_gu_t: torch.Tensor | None = None,
    last_access_interference_by_uav_t: torch.Tensor | None = None,
    last_selected_flag_active_t: torch.Tensor | None = None,
    last_sat_processed_active_t: torch.Tensor | None = None,
    sat_proc_capacity_active_t: torch.Tensor | None = None,
    sat_queue_full_t: torch.Tensor | None = None,
    sat_ema_full_t: torch.Tensor | None = None,
    sat_drop_full_t: torch.Tensor | None = None,
    last_sat_processed_full_t: torch.Tensor | None = None,
    last_selected_flag_full_t: torch.Tensor | None = None,
    access_gain_matrix_t: torch.Tensor | None = None,
    access_rate_static_params: _NativeAccessRateStaticParams | None = None,
) -> Mapping[str, torch.Tensor] | None:
    p = local_obs_params
    batch_size = int(uav_pos_t.shape[0])
    kernel_device = uav_pos_t.device
    num_uav = int(p.num_uav)
    num_gu = int(p.num_gu)
    max_sat_count = int(sat_pos_active_t.shape[1])
    gu_node_dim = critic_schema.CRITIC_GU_NODE_DIM
    uav_node_dim = critic_schema.CRITIC_UAV_NODE_DIM
    sat_node_dim = critic_schema.CRITIC_SAT_NODE_DIM
    ug_edge_dim = critic_schema.CRITIC_UAV_GU_EDGE_DIM
    us_edge_dim = critic_schema.CRITIC_UAV_SAT_EDGE_DIM
    uu_edge_dim = critic_schema.CRITIC_UAV_UAV_EDGE_DIM

    map_scale = normalize_scale(float(p.map_size))
    v_scale = normalize_scale(float(p.v_max))
    uav_energy_scale = normalize_scale(float(p.uav_energy_init))
    gu_queue_scale = normalize_scale(float(p.queue_max_gu))
    uav_queue_scale = normalize_scale(float(p.queue_max_uav))
    sat_queue_scale = normalize_scale(float(p.queue_max_sat))
    orbit_scale = normalize_scale(float(p.r_earth + p.sat_height))
    d_alert = (
        float(p.avoidance_alert_factor) * float(p.d_safe)
        if bool(p.avoidance_enabled)
        else float(p.d_safe)
    )
    tau0 = normalize_scale(float(p.tau0))
    service_floor = normalize_scale(float(getattr(p, "service_floor_bits_per_step", 1.0)))
    if arrival_ref_t is None:
        arrival_ref_vec_t = torch.full(
            (batch_size,),
            max(float(num_gu), 1.0) * tau0,
            dtype=torch.float32,
            device=kernel_device,
        )
    else:
        arrival_ref_vec_t = arrival_ref_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size)
    arrival_ref_vec_t = _torch_require_positive_reward_ref(arrival_ref_vec_t, name="arrival_ref_bits_per_step")
    gu_flow_ref_t = arrival_ref_vec_t / max(float(num_gu), 1.0)
    uav_flow_ref_t = arrival_ref_vec_t / max(float(num_uav), 1.0)
    sat_ref_count = normalize_scale(float(getattr(p, "sat_active_ref_count", max(float(num_uav), 1.0))))
    sat_flow_ref_t = arrival_ref_vec_t / sat_ref_count
    gu_flow_cost_ref_t = 1.0 / gu_flow_ref_t
    uav_flow_cost_ref_t = 1.0 / uav_flow_ref_t
    sat_cost_ref_t = 1.0 / sat_flow_ref_t
    uav_total_cost_ref_t = uav_flow_cost_ref_t + sat_cost_ref_t
    gu_total_cost_ref_t = gu_flow_cost_ref_t + uav_total_cost_ref_t

    def _optional_float_tensor(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        *,
        fill: float = 0.0,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(shape, float(fill), dtype=torch.float32, device=kernel_device)
        return value.to(device=kernel_device, dtype=torch.float32).reshape(shape)

    def _log_ratio_tensor(value_t: torch.Tensor, ref_t: torch.Tensor) -> torch.Tensor:
        value_safe_t = torch.clamp(value_t.to(dtype=torch.float32), min=float(NORMALIZATION_DENOM_EPS))
        ref_safe_t = torch.clamp(ref_t.to(device=value_safe_t.device, dtype=torch.float32), min=float(NORMALIZATION_DENOM_EPS))
        return torch.log(value_safe_t / ref_safe_t)

    gu_ema_work_t = _optional_float_tensor(gu_ema_t, (batch_size, num_gu), fill=1.0)
    uav_ema_work_t = _optional_float_tensor(uav_ema_t, (batch_size, num_uav), fill=1.0)
    sat_ema_active_work_t = _optional_float_tensor(sat_ema_active_t, (batch_size, max_sat_count), fill=1.0)
    expected_arrival_work_t = (
        _optional_float_tensor(expected_arrival_rate_vec_t, (batch_size, num_gu), fill=0.0) * tau0
    )
    last_gu_arrival_work_t = _optional_float_tensor(last_gu_arrival_t, (batch_size, num_gu), fill=0.0)
    last_gu_outflow_work_t = _optional_float_tensor(last_gu_outflow_t, (batch_size, num_gu), fill=0.0)
    gu_drop_work_t = _optional_float_tensor(gu_drop_t, (batch_size, num_gu), fill=0.0)
    uav_drop_work_t = _optional_float_tensor(uav_drop_t, (batch_size, num_uav), fill=0.0)
    sat_drop_active_work_t = _optional_float_tensor(sat_drop_active_t, (batch_size, max_sat_count), fill=0.0)
    last_uav_inflow_work_t = _optional_float_tensor(last_gu_to_uav_inflow_by_uav_t, (batch_size, num_uav), fill=0.0)
    last_uav_sat_outflow_active_work_t = _optional_float_tensor(
        last_uav_to_sat_outflow_active_t,
        (batch_size, num_uav, max_sat_count),
        fill=0.0,
    )
    last_bw_fraction_work_t = _optional_float_tensor(last_bw_fraction_by_uav_gu_t, (batch_size, num_uav, num_gu), fill=0.0)
    last_access_interference_work_t = _optional_float_tensor(
        last_access_interference_by_uav_t,
        (batch_size, num_uav),
        fill=0.0,
    )
    last_selected_active_work_t = _optional_float_tensor(
        last_selected_flag_active_t,
        (batch_size, num_uav, max_sat_count),
        fill=0.0,
    )
    last_sat_processed_active_work_t = _optional_float_tensor(last_sat_processed_active_t, (batch_size, max_sat_count), fill=0.0)
    sat_proc_capacity_active_work_t = _optional_float_tensor(sat_proc_capacity_active_t, (batch_size, max_sat_count), fill=0.0)
    sat_active_mask_float_t = sat_active_mask_t.to(device=kernel_device, dtype=torch.float32)
    sat_cost_active_t = (1.0 / torch.clamp(sat_ema_active_work_t, min=service_floor)).to(dtype=torch.float32)
    sat_active_count_t = sat_active_mask_float_t.sum(dim=1)
    sat_cost_fallback_t = torch.where(
        sat_active_count_t > 0.0,
        _torch_ratio_or_zero((sat_cost_active_t * sat_active_mask_float_t).sum(dim=1), sat_active_count_t),
        sat_cost_ref_t,
    ).to(dtype=torch.float32)
    local_uav_cost_t = (1.0 / torch.clamp(uav_ema_work_t, min=service_floor)).to(dtype=torch.float32)
    local_gu_cost_t = (1.0 / torch.clamp(gu_ema_work_t, min=service_floor)).to(dtype=torch.float32)

    def _uav_cost_from_active_selection(selection_active_t: torch.Tensor) -> torch.Tensor:
        if max_sat_count <= 0:
            downstream_t = sat_cost_fallback_t[:, None].expand(-1, num_uav)
            return (local_uav_cost_t + downstream_t).to(dtype=torch.float32)
        sel_t = selection_active_t.to(dtype=torch.float32) * sat_active_mask_float_t[:, None, :]
        selected_count_t = sel_t.sum(dim=2)
        selected_cost_t = (sel_t * sat_cost_active_t[:, None, :]).sum(dim=2)
        downstream_t = torch.where(
            selected_count_t > 0.0,
            _torch_ratio_or_zero(selected_cost_t, selected_count_t),
            sat_cost_fallback_t[:, None],
        )
        return (local_uav_cost_t + downstream_t).to(dtype=torch.float32)

    last_uav_total_cost_t = _uav_cost_from_active_selection(last_selected_active_work_t)
    prefix_uav_total_cost_t = _uav_cost_from_active_selection(current_sel_flag_active_t.to(device=kernel_device, dtype=torch.float32))

    prev_assoc_weight_t = prev_assoc_flag_t.to(device=kernel_device, dtype=torch.float32)
    prev_assoc_count_t = prev_assoc_weight_t.sum(dim=1)
    last_uav_cost_fallback_t = last_uav_total_cost_t.mean(dim=1)
    last_gu_downstream_t = torch.where(
        prev_assoc_count_t > 0.0,
        _torch_ratio_or_zero((prev_assoc_weight_t * last_uav_total_cost_t[:, :, None]).sum(dim=1), prev_assoc_count_t),
        last_uav_cost_fallback_t[:, None],
    )
    last_gu_total_cost_t = (local_gu_cost_t + last_gu_downstream_t).to(dtype=torch.float32)

    prefix_uav_cost_fallback_t = prefix_uav_total_cost_t.mean(dim=1)
    assoc_long_t = assoc_t.to(device=kernel_device, dtype=torch.long)
    if num_uav > 0 and num_gu > 0:
        valid_assoc_t = (assoc_long_t >= 0) & (assoc_long_t < num_uav)
        assoc_clamped_t = torch.clamp(assoc_long_t, min=0, max=max(num_uav - 1, 0))
        prefix_gu_downstream_t = torch.where(
            valid_assoc_t,
            prefix_uav_total_cost_t.gather(1, assoc_clamped_t),
            prefix_uav_cost_fallback_t[:, None],
        )
    else:
        prefix_gu_downstream_t = prefix_uav_cost_fallback_t[:, None].expand(-1, num_gu)
    prefix_gu_total_cost_t = (local_gu_cost_t + prefix_gu_downstream_t).to(dtype=torch.float32)

    sat_queue_global_t = (
        sat_queue_full_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size, int(p.num_sat))
        if sat_queue_full_t is not None
        else None
    )
    sat_ema_global_t = (
        sat_ema_full_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size, int(p.num_sat))
        if sat_ema_full_t is not None
        else None
    )
    sat_drop_global_t = (
        sat_drop_full_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size, int(p.num_sat))
        if sat_drop_full_t is not None
        else None
    )
    last_sat_processed_global_t = (
        last_sat_processed_full_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size, int(p.num_sat))
        if last_sat_processed_full_t is not None
        else None
    )
    last_selected_global_t = (
        last_selected_flag_full_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size, num_uav, int(p.num_sat))
        if last_selected_flag_full_t is not None
        else None
    )
    access_se_ref_feature_t = eta_ref_feature_t.to(device=kernel_device, dtype=torch.float32)
    if access_gain_matrix_t is not None and access_rate_static_params is not None:
        gain_by_uav_t = access_gain_matrix_t.to(device=kernel_device, dtype=torch.float32).permute(0, 2, 1)
        access_snr_ref_t = _snr_linear_torch(
            power=float(access_rate_static_params.gu_tx_power),
            gain_t=gain_by_uav_t,
            noise_density=float(access_rate_static_params.noise_density),
            bandwidth_t=torch.full_like(gain_by_uav_t, float(access_rate_static_params.b_acc)),
            noise_figure_db=float(access_rate_static_params.noise_figure_db),
        )
        access_se_ref_feature_t = _access_spectral_efficiency_torch(
            access_snr_ref_t,
            params=access_rate_static_params,
        ).to(dtype=torch.float32)
    if active_sat_ids_t is None:
        active_ids_for_global_t = torch.arange(max_sat_count, dtype=torch.long, device=kernel_device).view(1, max_sat_count).expand(batch_size, -1)
    else:
        active_ids_for_global_t = active_sat_ids_t.to(device=kernel_device, dtype=torch.long).reshape(batch_size, max_sat_count)
    token_sat_mask_full_t: torch.Tensor | None = None
    if max_sat_count > 0 and int(p.num_sat) > 0:
        active_valid_for_global_t = (active_ids_for_global_t >= 0) & (active_ids_for_global_t < int(p.num_sat))
        token_sat_hits_t = torch.zeros((batch_size, int(p.num_sat)), dtype=torch.float32, device=kernel_device)
        token_sat_hits_t.scatter_add_(
            1,
            torch.clamp(active_ids_for_global_t, min=0, max=max(int(p.num_sat) - 1, 0)),
            active_valid_for_global_t.to(dtype=torch.float32),
        )
        token_sat_mask_full_t = token_sat_hits_t > 0.5

    return_dict = history_world_out is None
    if history_world_out is None:
        history_world_out = _NativeTrainingWorldTensorFields(
            uav_nodes=torch.empty((batch_size, num_uav, uav_node_dim), dtype=torch.float32, device=kernel_device),
            gu_nodes=torch.empty((batch_size, num_gu, gu_node_dim), dtype=torch.float32, device=kernel_device),
            sat_nodes=torch.empty((batch_size, max_sat_count, sat_node_dim), dtype=torch.float32, device=kernel_device),
            sat_ids=torch.empty((batch_size, max_sat_count), dtype=torch.long, device=kernel_device),
            uav_gu_edges=torch.empty((batch_size, num_uav, num_gu, ug_edge_dim), dtype=torch.float32, device=kernel_device),
            uav_sat_edges=torch.empty((batch_size, num_uav, max_sat_count, us_edge_dim), dtype=torch.float32, device=kernel_device),
            uav_uav_edges=torch.empty((batch_size, num_uav, num_uav, uu_edge_dim), dtype=torch.float32, device=kernel_device),
            global_scalars=torch.empty((batch_size, critic_schema.CRITIC_GLOBAL_SCALAR_DIM), dtype=torch.float32, device=kernel_device),
            gu_mask=torch.empty((batch_size, num_gu), dtype=torch.bool, device=kernel_device),
            sat_mask=torch.empty((batch_size, max_sat_count), dtype=torch.bool, device=kernel_device),
            uav_gu_mask=torch.empty((batch_size, num_uav, num_gu), dtype=torch.bool, device=kernel_device),
            uav_sat_mask=torch.empty((batch_size, num_uav, max_sat_count), dtype=torch.bool, device=kernel_device),
            uav_uav_mask=torch.empty((batch_size, num_uav, num_uav), dtype=torch.bool, device=kernel_device),
        )
    if history_row_indices_t is None:
        history_row_indices_t = torch.arange(batch_size, dtype=torch.long, device=kernel_device)
    if stage_ids_t is None:
        stage_ids_runtime_t = torch.zeros((batch_size,), dtype=torch.long, device=kernel_device)
    else:
        stage_ids_runtime_t = stage_ids_t.to(device=kernel_device, dtype=torch.long).reshape(batch_size)
    prefix_bw_known_t = (stage_ids_runtime_t >= int(critic_schema.CRITIC_STAGE_SAT)).to(dtype=torch.float32)
    prefix_sat_known_t = (stage_ids_runtime_t >= int(critic_schema.CRITIC_STAGE_BW)).to(dtype=torch.float32)

    if (
        history_row_indices_t.dtype != torch.long
        or tuple(history_row_indices_t.shape) != (batch_size,)
        or history_world_out.uav_nodes.device != uav_pos_t.device
        or tuple(history_world_out.uav_nodes.shape[1:]) != (num_uav, uav_node_dim)
        or tuple(history_world_out.gu_nodes.shape[1:]) != (num_gu, gu_node_dim)
        or tuple(history_world_out.sat_nodes.shape[1:]) != (max_sat_count, sat_node_dim)
        or tuple(history_world_out.sat_ids.shape[1:]) != (max_sat_count,)
        or tuple(history_world_out.uav_gu_edges.shape[1:]) != (num_uav, num_gu, ug_edge_dim)
        or tuple(history_world_out.uav_sat_edges.shape[1:]) != (num_uav, max_sat_count, us_edge_dim)
        or tuple(history_world_out.uav_uav_edges.shape[1:]) != (num_uav, num_uav, uu_edge_dim)
        or tuple(history_world_out.global_scalars.shape[1:]) != (critic_schema.CRITIC_GLOBAL_SCALAR_DIM,)
        or tuple(history_world_out.gu_mask.shape[1:]) != (num_gu,)
        or tuple(history_world_out.sat_mask.shape[1:]) != (max_sat_count,)
        or tuple(history_world_out.uav_gu_mask.shape[1:]) != (num_uav, num_gu)
        or tuple(history_world_out.uav_sat_mask.shape[1:]) != (num_uav, max_sat_count)
        or tuple(history_world_out.uav_uav_mask.shape[1:]) != (num_uav, num_uav)
    ):
        raise RuntimeError("native history world buffers do not match the fixed world ABI.")
    history_world_out.uav_nodes.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.gu_nodes.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.sat_nodes.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.sat_ids.index_fill_(0, history_row_indices_t, -1)
    history_world_out.uav_gu_edges.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.uav_sat_edges.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.uav_uav_edges.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.global_scalars.index_fill_(0, history_row_indices_t, 0.0)
    history_world_out.gu_mask.index_fill_(0, history_row_indices_t, True)
    history_world_out.sat_mask.index_fill_(0, history_row_indices_t, False)
    history_world_out.uav_gu_mask.index_fill_(0, history_row_indices_t, True)
    history_world_out.uav_sat_mask.index_fill_(0, history_row_indices_t, False)
    history_world_out.uav_uav_mask.index_fill_(0, history_row_indices_t, False)

    if offdiag_mask_t is not None:
        if (
            not torch.is_tensor(offdiag_mask_t)
            or offdiag_mask_t.dtype != torch.bool
            or offdiag_mask_t.device != uav_pos_t.device
            or tuple(offdiag_mask_t.shape) != (1, num_uav, num_uav)
        ):
            raise RuntimeError("native world builder requires precomputed offdiag UAV mask constants.")
        offdiag_mask_batch_t = offdiag_mask_t if int(batch_size) == 1 else offdiag_mask_t.expand(batch_size, -1, -1)
    else:
        diag_mask_t = torch.eye(num_uav, dtype=torch.bool, device=uav_pos_t.device).unsqueeze(0)
        offdiag_mask_batch_t = (~diag_mask_t) if int(batch_size) == 1 else (~diag_mask_t).expand(batch_size, -1, -1)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_X : critic_schema.UAV_Y + 1] = (uav_pos_t / map_scale).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_VX : critic_schema.UAV_VY + 1] = (uav_vel_t / v_scale).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_ENERGY] = (uav_energy_t / uav_energy_scale).to(dtype=history_world_out.uav_nodes.dtype)
    uav_queue_steps_t = _torch_log1p_nonnegative(uav_queue_t.to(dtype=torch.float32) / uav_flow_ref_t[:, None])
    uav_queue_fill_t = uav_queue_t.to(dtype=torch.float32) / uav_queue_scale
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_QUEUE_STEPS] = uav_queue_steps_t.to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_QUEUE_FILL] = uav_queue_fill_t.to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_INFLOW_STEPS] = _torch_log1p_nonnegative(last_uav_inflow_work_t / uav_flow_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    last_uav_outflow_work_t = last_uav_sat_outflow_active_work_t.sum(dim=2)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_OUTFLOW_STEPS] = _torch_log1p_nonnegative(last_uav_outflow_work_t / uav_flow_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_DROP_STEPS] = _torch_log1p_nonnegative(uav_drop_work_t / uav_flow_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_SERVICE_EMA_STEPS] = _torch_log1p_nonnegative(uav_ema_work_t / uav_flow_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LOCAL_COST_LOG_RATIO] = _log_ratio_tensor(local_uav_cost_t, uav_flow_cost_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_TOTAL_COST_LOG_RATIO] = _log_ratio_tensor(last_uav_total_cost_t, uav_total_cost_ref_t[:, None]).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_WORKLOAD_LOG1P] = torch.log1p(torch.clamp(last_uav_total_cost_t * uav_queue_t.to(dtype=torch.float32), min=0.0)).to(dtype=history_world_out.uav_nodes.dtype)
    prefix_workload_known_uav_t = prefix_sat_known_t[:, None]
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_PREFIX_TOTAL_COST_LOG_RATIO] = (_log_ratio_tensor(prefix_uav_total_cost_t, uav_total_cost_ref_t[:, None]) * prefix_workload_known_uav_t).to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_PREFIX_COST_KNOWN] = prefix_workload_known_uav_t.to(dtype=history_world_out.uav_nodes.dtype)
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_PREFIX_WORKLOAD_LOG1P] = (
        torch.log1p(torch.clamp(prefix_uav_total_cost_t * uav_queue_t.to(dtype=torch.float32), min=0.0)) * prefix_workload_known_uav_t
    ).to(dtype=history_world_out.uav_nodes.dtype)
    access_nf_linear = 10.0 ** (max(float(getattr(p, "access_noise_figure_db", getattr(p, "noise_figure_db", 0.0))), 0.0) / 10.0)
    access_noise_full_band = normalize_scale(float(p.noise_density) * access_nf_linear * float(p.b_acc))
    history_world_out.uav_nodes[history_row_indices_t, :, critic_schema.UAV_LAST_ACCESS_INTERFERENCE_LOG1P] = torch.log1p(
        torch.clamp(last_access_interference_work_t / access_noise_full_band, min=0.0)
    ).to(dtype=history_world_out.uav_nodes.dtype)

    if num_gu > 0:
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_X : critic_schema.GU_Y + 1] = (gu_pos_t / map_scale).to(dtype=history_world_out.gu_nodes.dtype)
        gu_queue_steps_t = _torch_log1p_nonnegative(gu_queue_t.to(dtype=torch.float32) / gu_flow_ref_t[:, None])
        gu_queue_fill_t = gu_queue_t.to(dtype=torch.float32) / gu_queue_scale
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_QUEUE_STEPS] = gu_queue_steps_t.to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_QUEUE_FILL] = gu_queue_fill_t.to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_EXPECTED_ARRIVAL_STEPS] = _torch_log1p_nonnegative(expected_arrival_work_t / gu_flow_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LAST_ARRIVAL_STEPS] = _torch_log1p_nonnegative(last_gu_arrival_work_t / gu_flow_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LAST_OUTFLOW_STEPS] = _torch_log1p_nonnegative(last_gu_outflow_work_t / gu_flow_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LAST_DROP_STEPS] = _torch_log1p_nonnegative(gu_drop_work_t / gu_flow_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_SERVICE_EMA_STEPS] = _torch_log1p_nonnegative(gu_ema_work_t / gu_flow_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LOCAL_COST_LOG_RATIO] = _log_ratio_tensor(local_gu_cost_t, gu_flow_cost_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LAST_TOTAL_COST_LOG_RATIO] = _log_ratio_tensor(last_gu_total_cost_t, gu_total_cost_ref_t[:, None]).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_LAST_WORKLOAD_LOG1P] = torch.log1p(torch.clamp(last_gu_total_cost_t * gu_queue_t.to(dtype=torch.float32), min=0.0)).to(dtype=history_world_out.gu_nodes.dtype)
        prefix_workload_known_gu_t = prefix_sat_known_t[:, None]
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_PREFIX_TOTAL_COST_LOG_RATIO] = (_log_ratio_tensor(prefix_gu_total_cost_t, gu_total_cost_ref_t[:, None]) * prefix_workload_known_gu_t).to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_PREFIX_COST_KNOWN] = prefix_workload_known_gu_t.to(dtype=history_world_out.gu_nodes.dtype)
        history_world_out.gu_nodes[history_row_indices_t, :, critic_schema.GU_PREFIX_WORKLOAD_LOG1P] = (
            torch.log1p(torch.clamp(prefix_gu_total_cost_t * gu_queue_t.to(dtype=torch.float32), min=0.0)) * prefix_workload_known_gu_t
        ).to(dtype=history_world_out.gu_nodes.dtype)
        gu_rel_t = gu_pos_t[:, None, :, :] - uav_pos_t[:, :, None, :]
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_REL_X : critic_schema.UG_REL_Y + 1] = (gu_rel_t / map_scale).to(dtype=history_world_out.uav_gu_edges.dtype)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_HORIZONTAL_DIST] = (
            torch.linalg.vector_norm(gu_rel_t, dim=-1) / map_scale
        ).to(dtype=history_world_out.uav_gu_edges.dtype)
        horizontal_t = torch.linalg.vector_norm(gu_rel_t, dim=-1).clamp_min(float(GEOMETRY_DENOM_EPS))
        elevation_t = torch.atan2(torch.full_like(horizontal_t, float(p.uav_height)), horizontal_t)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_ELEVATION_NORM] = (
            elevation_t / (math.pi * 0.5)
        ).to(dtype=history_world_out.uav_gu_edges.dtype)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_ACCESS_SE_REF] = access_se_ref_feature_t.to(dtype=history_world_out.uav_gu_edges.dtype)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_LAST_BW_FRACTION] = last_bw_fraction_work_t.to(dtype=history_world_out.uav_gu_edges.dtype)
        last_bw_weight_t = last_bw_fraction_work_t.to(device=kernel_device, dtype=torch.float32)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_LAST_SERVED_FLAG] = last_bw_weight_t.to(dtype=history_world_out.uav_gu_edges.dtype)
        prefix_bw_known_ugg_t = prefix_bw_known_t[:, None, None]
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_PREFIX_BW_VALID_FLAG] = (
            bw_valid_flag_t.to(dtype=torch.float32) * prefix_bw_known_ugg_t
        ).to(dtype=history_world_out.uav_gu_edges.dtype)
        history_world_out.uav_gu_edges[history_row_indices_t, :, :, critic_schema.UG_PREFIX_BW_VALID_KNOWN] = prefix_bw_known_ugg_t.to(dtype=history_world_out.uav_gu_edges.dtype)

    rel_uav_pos_t = uav_pos_t[:, None, :, :] - uav_pos_t[:, :, None, :]
    rel_uav_vel_t = uav_vel_t[:, None, :, :] - uav_vel_t[:, :, None, :]
    dist_uav_t = torch.linalg.vector_norm(rel_uav_pos_t, dim=-1)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_REL_X : critic_schema.UU_REL_Y + 1] = (rel_uav_pos_t / map_scale).to(dtype=history_world_out.uav_uav_edges.dtype)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_REL_VX : critic_schema.UU_REL_VY + 1] = (rel_uav_vel_t / v_scale).to(dtype=history_world_out.uav_uav_edges.dtype)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_DIST_NORM] = (dist_uav_t / map_scale).to(dtype=history_world_out.uav_uav_edges.dtype)
    closing_den_t = torch.clamp(dist_uav_t * float(p.v_max), min=float(GEOMETRY_DENOM_EPS))
    closing_t = -(rel_uav_pos_t * rel_uav_vel_t).sum(dim=-1) / closing_den_t
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_CLOSING_SPEED_NORM] = closing_t.to(dtype=history_world_out.uav_uav_edges.dtype)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_ALERT_FLAG] = (dist_uav_t < d_alert).to(dtype=history_world_out.uav_uav_edges.dtype)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_UNSAFE_FLAG] = (dist_uav_t < float(p.d_safe)).to(dtype=history_world_out.uav_uav_edges.dtype)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_ALERT_FLAG].masked_fill_(~offdiag_mask_batch_t, 0.0)
    history_world_out.uav_uav_edges[history_row_indices_t, :, :, critic_schema.UU_UNSAFE_FLAG].masked_fill_(~offdiag_mask_batch_t, 0.0)
    history_world_out.uav_uav_mask[history_row_indices_t] = offdiag_mask_batch_t

    if max_sat_count > 0:
        if active_sat_ids_t is None:
            slot_sat_ids_t = torch.arange(max_sat_count, dtype=torch.long, device=kernel_device).view(1, max_sat_count).expand(batch_size, -1)
        else:
            slot_sat_ids_t = active_sat_ids_t.to(device=kernel_device, dtype=torch.long).reshape(batch_size, max_sat_count)
        history_world_out.sat_ids[history_row_indices_t] = torch.where(sat_active_mask_t, slot_sat_ids_t, torch.full_like(slot_sat_ids_t, -1))
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_X : critic_schema.SAT_Z + 1] = ((sat_pos_active_t / orbit_scale) * sat_active_mask_float_t.unsqueeze(-1)).to(dtype=history_world_out.sat_nodes.dtype)
        sat_speed_scale = math.sqrt(3.986004418e14 / max(float(p.r_earth + p.sat_height), float(GEOMETRY_DENOM_EPS)))
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_VX : critic_schema.SAT_VZ + 1] = ((sat_vel_active_t / sat_speed_scale) * sat_active_mask_float_t.unsqueeze(-1)).to(dtype=history_world_out.sat_nodes.dtype)
        sat_queue_steps_t = _torch_log1p_nonnegative(sat_queue_active_t.to(dtype=torch.float32) / sat_flow_ref_t[:, None]) * sat_active_mask_float_t
        sat_queue_fill_t = (sat_queue_active_t.to(dtype=torch.float32) / sat_queue_scale) * sat_active_mask_float_t
        sat_last_incoming_t = last_uav_sat_outflow_active_work_t.sum(dim=1)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_QUEUE_STEPS] = sat_queue_steps_t.to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_QUEUE_FILL] = sat_queue_fill_t.to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_LAST_INCOMING_STEPS] = (_torch_log1p_nonnegative(sat_last_incoming_t / sat_flow_ref_t[:, None]) * sat_active_mask_float_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_LAST_PROCESSED_STEPS] = (_torch_log1p_nonnegative(last_sat_processed_active_work_t / sat_flow_ref_t[:, None]) * sat_active_mask_float_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_LAST_DROP_STEPS] = (_torch_log1p_nonnegative(sat_drop_active_work_t / sat_flow_ref_t[:, None]) * sat_active_mask_float_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_SERVICE_EMA_STEPS] = (_torch_log1p_nonnegative(sat_ema_active_work_t / sat_flow_ref_t[:, None]) * sat_active_mask_float_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_COST_LOG_RATIO] = (_log_ratio_tensor(sat_cost_active_t, sat_cost_ref_t[:, None]) * sat_active_mask_float_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_LAST_WORKLOAD_LOG1P] = (
            torch.log1p(torch.clamp(sat_cost_active_t * sat_queue_active_t.to(dtype=torch.float32), min=0.0)) * sat_active_mask_float_t
        ).to(dtype=history_world_out.sat_nodes.dtype)
        prefix_sat_known_s_t = prefix_sat_known_t[:, None]
        prefix_selected_count_active_t = current_sel_flag_active_t.to(dtype=torch.float32).sum(dim=1) * sat_active_mask_float_t
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_PREFIX_SELECTED_LOAD_FRAC] = (
            (prefix_selected_count_active_t / max(float(p.num_uav), 1.0)) * prefix_sat_known_s_t
        ).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_PREFIX_LOAD_KNOWN] = (sat_active_mask_float_t * prefix_sat_known_s_t).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_LAST_SELECTED_LOAD_FRAC] = (
            last_selected_active_work_t.sum(dim=1) / max(float(p.num_uav), 1.0) * sat_active_mask_float_t
        ).to(dtype=history_world_out.sat_nodes.dtype)
        history_world_out.sat_nodes[history_row_indices_t, :, critic_schema.SAT_PROC_CAPACITY_STEPS] = (
            _torch_log1p_nonnegative(sat_proc_capacity_active_work_t / sat_flow_ref_t[:, None]) * sat_active_mask_float_t
        ).to(dtype=history_world_out.sat_nodes.dtype)

        projected_count_t = torch.clamp(prefix_selected_count_active_t, min=1.0)
        projected_bw_t = effective_b_backhaul_per_sat_t[:, None] / projected_count_t
        snr_ref_t = _snr_linear_torch(
            power=float(p.uav_tx_power),
            gain_t=gain_active_t,
            noise_density=float(p.noise_density),
            bandwidth_t=effective_b_backhaul_per_sat_t[:, None, None],
            noise_figure_db=float(getattr(p, "backhaul_noise_figure_db", p.noise_figure_db)),
        )
        snr_prefix_t = _snr_linear_torch(
            power=float(p.uav_tx_power),
            gain_t=gain_active_t,
            noise_density=float(p.noise_density),
            bandwidth_t=projected_bw_t[:, None, :],
            noise_figure_db=float(getattr(p, "backhaul_noise_figure_db", p.noise_figure_db)),
        )
        if bool(p.doppler_observed) and bool(p.doppler_atten_enabled):
            attenuation_t = _doppler_attenuation_torch(
                nu_eff_active_t,
                subcarrier_spacing=float(p.subcarrier_spacing),
            )
            snr_ref_t = snr_ref_t * attenuation_t
            snr_prefix_t = snr_prefix_t * attenuation_t
        se_ref_t = _spectral_efficiency_torch(snr_ref_t).to(dtype=torch.float32)
        se_prefix_t = _spectral_efficiency_torch(snr_prefix_t).to(dtype=torch.float32)
        active_mask_float_t = sat_active_mask_t[:, None, :].to(dtype=torch.float32)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_REL_X : critic_schema.US_REL_Z + 1] = ((rel_pos_active_t / orbit_scale) * active_mask_float_t.unsqueeze(-1)).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_REL_VX : critic_schema.US_REL_VZ + 1] = ((rel_vel_active_t / sat_speed_scale) * active_mask_float_t.unsqueeze(-1)).to(dtype=history_world_out.uav_sat_edges.dtype)
        range_t = torch.linalg.vector_norm(rel_pos_active_t, dim=-1).clamp_min(float(GEOMETRY_DENOM_EPS))
        radial_t = (rel_pos_active_t * rel_vel_active_t).sum(dim=-1) / (range_t * sat_speed_scale)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_RADIAL_VELOCITY_NORM] = (radial_t * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_RANGE_NORM] = ((range_t / orbit_scale) * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        doppler_ref = max(float(getattr(p, "backhaul_carrier_freq", 1.0)) * sat_speed_scale / max(float(getattr(p, "speed_of_light", 299792458.0)), 1.0), 1.0)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_DOPPLER_NORM] = ((nu_eff_active_t / doppler_ref) * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_DOPPLER_MARGIN] = ((nu_eff_active_t / max(float(p.nu_max), 1.0)) * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_BACKHAUL_SE_REF] = (se_ref_t * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_VISIBLE_FLAG] = (visible_flag_active_t.to(dtype=torch.float32) * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_VALID_FLAG] = (valid_flag_active_t.to(dtype=torch.float32) * active_mask_float_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_LAST_SELECTED_FLAG] = (
            last_selected_active_work_t * active_mask_float_t
        ).to(dtype=history_world_out.uav_sat_edges.dtype)
        prefix_sat_known_uss_t = prefix_sat_known_t[:, None, None]
        prefix_sel_t = current_sel_flag_active_t.to(dtype=torch.float32) * active_mask_float_t * prefix_sat_known_uss_t
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_PREFIX_SELECTED_FLAG] = prefix_sel_t.to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_PREFIX_SELECTED_KNOWN] = (active_mask_float_t * prefix_sat_known_uss_t).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.uav_sat_edges[history_row_indices_t, :, :, critic_schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS] = (
            _torch_log1p_nonnegative(projected_bw_t[:, None, :] * se_prefix_t * tau0 / uav_flow_ref_t[:, None, None])
            * active_mask_float_t
            * prefix_sel_t
        ).to(dtype=history_world_out.uav_sat_edges.dtype)
        history_world_out.sat_mask[history_row_indices_t] = sat_active_mask_t.to(dtype=torch.bool)
        history_world_out.uav_sat_mask[history_row_indices_t] = sat_active_mask_t[:, None, :].expand(-1, num_uav, -1).to(dtype=torch.bool)
    gs = history_world_out.global_scalars
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_GU_QUEUE_STEPS] = _torch_log1p_nonnegative(
        gu_queue_t.to(dtype=torch.float32).sum(dim=1) / gu_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_UAV_QUEUE_STEPS] = _torch_log1p_nonnegative(
        uav_queue_t.to(dtype=torch.float32).sum(dim=1) / uav_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_SAT_QUEUE_STEPS] = _torch_log1p_nonnegative(
        (
            sat_queue_global_t.sum(dim=1)
            if sat_queue_global_t is not None
            else sat_queue_active_t.to(dtype=torch.float32).sum(dim=1)
        )
        / sat_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_GU_DROP_STEPS] = _torch_log1p_nonnegative(
        gu_drop_work_t.sum(dim=1) / gu_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_UAV_DROP_STEPS] = _torch_log1p_nonnegative(
        uav_drop_work_t.sum(dim=1) / uav_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_SAT_DROP_STEPS] = _torch_log1p_nonnegative(
        (
            sat_drop_global_t.sum(dim=1)
            if sat_drop_global_t is not None
            else sat_drop_active_work_t.sum(dim=1)
        )
        / sat_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS] = _torch_log1p_nonnegative(
        expected_arrival_work_t.sum(dim=1) / gu_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_LAST_GU_OUTFLOW_STEPS] = _torch_log1p_nonnegative(
        last_gu_outflow_work_t.sum(dim=1) / gu_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_LAST_UAV_OUTFLOW_STEPS] = _torch_log1p_nonnegative(
        last_uav_sat_outflow_active_work_t.sum(dim=(1, 2)) / uav_flow_ref_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_LAST_SAT_PROCESSED_STEPS] = _torch_log1p_nonnegative(
        (
            last_sat_processed_global_t.sum(dim=1)
            if last_sat_processed_global_t is not None
            else last_sat_processed_active_work_t.sum(dim=1)
        )
        / sat_flow_ref_t
    ).to(dtype=gs.dtype)
    if sat_queue_global_t is not None and sat_ema_global_t is not None:
        sat_cost_global_t = 1.0 / torch.clamp(sat_ema_global_t, min=service_floor)
        sat_workload_last_t = (sat_cost_global_t * sat_queue_global_t).sum(dim=1)
        sat_workload_prefix_t = sat_workload_last_t
    else:
        sat_cost_global_t = None
        sat_workload_last_t = ((1.0 / torch.clamp(sat_ema_active_work_t, min=service_floor)) * sat_queue_active_t.to(dtype=torch.float32)).sum(dim=1)
        sat_workload_prefix_t = sat_workload_last_t
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_LAST_WEIGHTED_WORKLOAD_STEPS] = _torch_log1p_nonnegative(
        (last_gu_total_cost_t * gu_queue_t.to(dtype=torch.float32)).sum(dim=1)
        + (last_uav_total_cost_t * uav_queue_t.to(dtype=torch.float32)).sum(dim=1)
        + sat_workload_last_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS] = (
        _torch_log1p_nonnegative(
            (prefix_gu_total_cost_t * gu_queue_t.to(dtype=torch.float32)).sum(dim=1)
            + (prefix_uav_total_cost_t * uav_queue_t.to(dtype=torch.float32)).sum(dim=1)
            + sat_workload_prefix_t
        )
        * prefix_sat_known_t.to(dtype=torch.float32)
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_PREFIX_WORKLOAD_KNOWN] = prefix_sat_known_t.to(dtype=gs.dtype)
    interference_log_t = torch.log1p(torch.clamp(last_access_interference_work_t / access_noise_full_band, min=0.0))
    gs[history_row_indices_t, critic_schema.GLOBAL_LAST_INTERFERENCE_MEAN] = interference_log_t.mean(dim=1).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_LAST_INTERFERENCE_MAX] = interference_log_t.amax(dim=1).to(dtype=gs.dtype)
    prefix_selected_count_active_t = current_sel_flag_active_t.to(dtype=torch.float32).sum(dim=1) * sat_active_mask_t.to(dtype=torch.float32)
    selected_load_frac_t = torch.where(
        (prefix_selected_count_active_t > 0.0) & sat_active_mask_t,
        prefix_selected_count_active_t / max(float(num_uav), 1.0),
        torch.zeros_like(sat_load_active_t),
    )
    selected_count_t = ((prefix_selected_count_active_t > 0.0) & sat_active_mask_t).to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
    selected_known_t = prefix_sat_known_t.to(dtype=torch.float32)
    gs[history_row_indices_t, critic_schema.GLOBAL_SELECTED_SAT_LOAD_MEAN] = (
        (selected_load_frac_t.sum(dim=1) / selected_count_t) * selected_known_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_SELECTED_SAT_LOAD_MAX] = (
        selected_load_frac_t.amax(dim=1) * selected_known_t
    ).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_SELECTED_SAT_LOAD_KNOWN] = selected_known_t.to(dtype=gs.dtype)
    last_load_source_t = (
        last_selected_global_t.sum(dim=1)
        if last_selected_global_t is not None
        else last_selected_active_work_t.sum(dim=1) * sat_active_mask_t.to(dtype=torch.float32)
    )
    last_load_frac_t = last_load_source_t / max(float(num_uav), 1.0)
    last_count_t = (last_load_frac_t > 0.0).to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
    gs[history_row_indices_t, critic_schema.GLOBAL_LAST_SELECTED_SAT_LOAD_MEAN] = (last_load_frac_t.sum(dim=1) / last_count_t).to(dtype=gs.dtype)
    gs[history_row_indices_t, critic_schema.GLOBAL_LAST_SELECTED_SAT_LOAD_MAX] = last_load_frac_t.amax(dim=1).to(dtype=gs.dtype)
    if token_sat_mask_full_t is not None and sat_queue_global_t is not None:
        non_token_mask_t = ~token_sat_mask_full_t
        non_token_f_t = non_token_mask_t.to(dtype=torch.float32)
        gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_COUNT_FRAC] = (
            non_token_f_t.sum(dim=1) / max(float(int(p.num_sat)), 1.0)
        ).to(dtype=gs.dtype)
        gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS] = _torch_log1p_nonnegative(
            (sat_queue_global_t * non_token_f_t).sum(dim=1) / sat_flow_ref_t
        ).to(dtype=gs.dtype)
        if sat_drop_global_t is not None:
            gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_DROP_STEPS] = _torch_log1p_nonnegative(
                (sat_drop_global_t * non_token_f_t).sum(dim=1) / sat_flow_ref_t
            ).to(dtype=gs.dtype)
        if last_sat_processed_global_t is not None:
            gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS] = _torch_log1p_nonnegative(
                (last_sat_processed_global_t * non_token_f_t).sum(dim=1) / sat_flow_ref_t
            ).to(dtype=gs.dtype)
        if sat_cost_global_t is not None:
            gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS] = _torch_log1p_nonnegative(
                (sat_cost_global_t * sat_queue_global_t * non_token_f_t).sum(dim=1)
            ).to(dtype=gs.dtype)
            if sat_drop_global_t is not None:
                gs[history_row_indices_t, critic_schema.GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS] = _torch_log1p_nonnegative(
                    (sat_cost_global_t * sat_drop_global_t * non_token_f_t).sum(dim=1)
                ).to(dtype=gs.dtype)
    if t_t is not None:
        t_steps = max(float(getattr(p, "t_steps", 1.0)), 1.0)
        denom = max(t_steps - 1.0, 1.0)
        remaining_t = torch.clamp(
            (t_steps - 1.0 - t_t.to(device=kernel_device, dtype=torch.float32).reshape(batch_size)) / denom,
            min=0.0,
            max=1.0,
        )
        gs[history_row_indices_t, critic_schema.GLOBAL_REMAINING_HORIZON_FRAC] = remaining_t.to(dtype=gs.dtype)
    if return_dict:
        return {
            "uav_nodes": history_world_out.uav_nodes,
            "gu_nodes": history_world_out.gu_nodes,
            "sat_nodes": history_world_out.sat_nodes,
            "sat_ids": history_world_out.sat_ids,
            "uav_gu_edges": history_world_out.uav_gu_edges,
            "uav_sat_edges": history_world_out.uav_sat_edges,
            "uav_uav_edges": history_world_out.uav_uav_edges,
            "global_scalars": history_world_out.global_scalars,
            "gu_mask": history_world_out.gu_mask,
            "sat_mask": history_world_out.sat_mask,
            "uav_gu_mask": history_world_out.uav_gu_mask,
            "uav_sat_mask": history_world_out.uav_sat_mask,
            "uav_uav_mask": history_world_out.uav_uav_mask,
            "stage_ids": stage_ids_runtime_t,
        }
    return None


def _structured_world_state_from_tensor_dict(
    world_out: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str | None,
) -> StructuredWorldState:
    target_device = None if device is None else torch.device(device)

    def _world_float(name: str) -> torch.Tensor:
        tensor = world_out[name]
        return tensor.to(dtype=torch.float32) if target_device is None else tensor.to(device=target_device, dtype=torch.float32)

    def _world_bool(name: str) -> torch.Tensor:
        tensor = world_out[name]
        return tensor.to(dtype=torch.bool) if target_device is None else tensor.to(device=target_device, dtype=torch.bool)

    stage_ids_t = world_out["stage_ids"]
    return StructuredWorldState(
        uav_nodes=_world_float("uav_nodes"),
        gu_nodes=_world_float("gu_nodes"),
        sat_nodes=_world_float("sat_nodes"),
        sat_ids=world_out["sat_ids"].to(dtype=torch.long) if target_device is None else world_out["sat_ids"].to(device=target_device, dtype=torch.long),
        uav_gu_edges=_world_float("uav_gu_edges"),
        uav_sat_edges=_world_float("uav_sat_edges"),
        uav_uav_edges=_world_float("uav_uav_edges"),
        global_scalars=_world_float("global_scalars"),
        gu_mask=_world_bool("gu_mask"),
        sat_mask=_world_bool("sat_mask"),
        uav_gu_mask=_world_bool("uav_gu_mask"),
        uav_sat_mask=_world_bool("uav_sat_mask"),
        uav_uav_mask=_world_bool("uav_uav_mask"),
        stage_id=stage_ids_t.to(dtype=torch.long) if target_device is None else stage_ids_t.to(device=target_device, dtype=torch.long),
    )


def _build_world_from_stage_fields_direct_tensor_impl(
    *,
    local_obs_params: _NativeLocalObsStaticParams,
    fields_obj: _NativeStageTensorFields,
    stage_id: int,
    t_t: torch.Tensor | None = None,
    access_rate_static_params: _NativeAccessRateStaticParams | None = None,
    device: torch.device | str | None = None,
) -> StructuredWorldState:
    """Build the training world tensors directly from native stage field buffers.

    This is the direct live replacement for the old stage-view/world-ref
    materialization chain. It accepts only runtime-owned tensor fields and
    typed params; reference adapters may still materialize their own worlds
    outside the live core.
    """
    if not _is_native_stage_fields(fields_obj):
        raise RuntimeError("native direct world history requires fixed stage field tensors.")
    p = local_obs_params
    kernel_device = fields_obj.uav_pos.device
    batch_size = int(fields_obj.uav_pos.shape[0])
    active_ids_t = fields_obj.active_sat_ids.to(device=kernel_device, dtype=torch.long)
    active_mask_t = active_ids_t >= 0
    active_safe_t = torch.clamp(active_ids_t, min=0, max=max(int(p.num_sat) - 1, 0))
    active_mask_float_t = active_mask_t.to(dtype=torch.float32)
    sat_selection_t = fields_obj.sat_selection_matrix.to(device=kernel_device, dtype=torch.long)
    max_sat_count = int(active_ids_t.shape[1]) if active_ids_t.ndim >= 2 else 0
    if max_sat_count > 0:
        current_sel_flag_active_t = (
            active_safe_t[:, None, None, :] == sat_selection_t.unsqueeze(-1)
        ).any(dim=2).to(dtype=torch.float32) * active_mask_float_t[:, None, :]
    else:
        current_sel_flag_active_t = fields_obj.us_gain_active.new_zeros((batch_size, int(p.num_uav), 0))
    if max_sat_count > 0:
        active_scalar_idx_t = active_safe_t
        active_selected_idx_t = active_safe_t[:, None, :].expand(batch_size, int(p.num_uav), max_sat_count)
        sat_ema_active_t = torch.gather(
            fields_obj.sat_ema.to(device=kernel_device, dtype=torch.float32),
            1,
            active_scalar_idx_t,
        ) * active_mask_float_t
        sat_drop_active_t = torch.gather(
            fields_obj.sat_drop.to(device=kernel_device, dtype=torch.float32),
            1,
            active_scalar_idx_t,
        ) * active_mask_float_t
        last_sat_processed_active_t = torch.gather(
            fields_obj.last_sat_processed.to(device=kernel_device, dtype=torch.float32),
            1,
            active_scalar_idx_t,
        ) * active_mask_float_t
        last_selected_active_t = torch.gather(
            fields_obj.last_selected_mask_by_uav_sat.to(device=kernel_device, dtype=torch.float32),
            2,
            active_selected_idx_t,
        ) * active_mask_float_t[:, None, :]
        last_uav_to_sat_outflow_active_t = torch.gather(
            fields_obj.last_uav_to_sat_outflow_matrix.to(device=kernel_device, dtype=torch.float32),
            2,
            active_selected_idx_t,
        ) * active_mask_float_t[:, None, :]
    else:
        sat_ema_active_t = fields_obj.sat_queue_active.new_zeros((batch_size, 0), dtype=torch.float32)
        sat_drop_active_t = fields_obj.sat_queue_active.new_zeros((batch_size, 0), dtype=torch.float32)
        last_sat_processed_active_t = fields_obj.sat_queue_active.new_zeros((batch_size, 0), dtype=torch.float32)
        last_selected_active_t = fields_obj.us_gain_active.new_zeros((batch_size, int(p.num_uav), 0), dtype=torch.float32)
        last_uav_to_sat_outflow_active_t = fields_obj.us_gain_active.new_zeros((batch_size, int(p.num_uav), 0), dtype=torch.float32)
    stage_ids_t = fields_obj.stage_id.to(device=kernel_device, dtype=torch.long).reshape(batch_size).clone()
    if int(stage_id) >= 0:
        stage_ids_t.fill_(int(stage_id))
    temp_world_out = _allocate_native_training_world_tensor_fields(
        local_obs_params=p,
        fields_obj=fields_obj,
    )
    history_row_indices_t = torch.arange(batch_size, dtype=torch.long, device=kernel_device)
    _build_world_from_packed_specs_tensor_impl(
        local_obs_params=p,
        stage_ids_t=stage_ids_t,
        effective_b_backhaul_per_sat_t=fields_obj.effective_b_backhaul_per_sat.to(device=kernel_device, dtype=torch.float32).reshape(batch_size),
        uav_pos_t=fields_obj.uav_pos.to(device=kernel_device, dtype=torch.float32),
        uav_vel_t=fields_obj.uav_vel.to(device=kernel_device, dtype=torch.float32),
        uav_energy_t=fields_obj.uav_energy.to(device=kernel_device, dtype=torch.float32),
        uav_queue_t=fields_obj.uav_queue.to(device=kernel_device, dtype=torch.float32),
        uav_assoc_uav_cost_t=fields_obj.uav_assoc_uav_cost.to(device=kernel_device, dtype=torch.float32),
        gu_pos_t=fields_obj.gu_pos.to(device=kernel_device, dtype=torch.float32),
        gu_queue_t=fields_obj.gu_queue.to(device=kernel_device, dtype=torch.float32),
        gu_proxy_features_t=fields_obj.gu_proxy_features.to(device=kernel_device, dtype=torch.float32),
        assoc_t=fields_obj.assoc.to(device=kernel_device, dtype=torch.long),
        candidate_flag_t=fields_obj.candidate_flag.to(device=kernel_device, dtype=torch.float32),
        bw_valid_flag_t=fields_obj.bw_valid_flag.to(device=kernel_device, dtype=torch.float32),
        prev_assoc_flag_t=fields_obj.prev_assoc_flag.to(device=kernel_device, dtype=torch.float32),
        eta_ref_feature_t=fields_obj.eta_ref_feature.to(device=kernel_device, dtype=torch.float32),
        sat_pos_active_t=fields_obj.sat_pos_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t.unsqueeze(-1),
        sat_vel_active_t=fields_obj.sat_vel_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t.unsqueeze(-1),
        sat_queue_active_t=fields_obj.sat_queue_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t,
        sat_load_active_t=fields_obj.sat_load_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t,
        sat_cost_norm_active_t=fields_obj.sat_cost_norm_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t,
        sat_active_mask_t=active_mask_t,
        rel_pos_active_t=fields_obj.us_rel_pos_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :, None],
        rel_vel_active_t=fields_obj.us_rel_vel_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :, None],
        gain_active_t=fields_obj.us_gain_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :],
        nu_eff_active_t=fields_obj.us_nu_eff_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :],
        visible_flag_active_t=fields_obj.visible_flag_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :],
        valid_flag_active_t=fields_obj.us_valid_flag_active.to(device=kernel_device, dtype=torch.float32) * active_mask_float_t[:, None, :],
        current_sel_flag_active_t=current_sel_flag_active_t,
        t_t=t_t,
        history_world_out=temp_world_out,
        history_row_indices_t=history_row_indices_t,
        active_sat_ids_t=active_ids_t,
        arrival_ref_t=fields_obj.arrival_ref_bits_per_step.to(device=kernel_device, dtype=torch.float32),
        expected_arrival_rate_vec_t=fields_obj.expected_arrival_rate_vec.to(device=kernel_device, dtype=torch.float32),
        gu_ema_t=fields_obj.gu_ema.to(device=kernel_device, dtype=torch.float32),
        uav_ema_t=fields_obj.uav_ema.to(device=kernel_device, dtype=torch.float32),
        sat_ema_active_t=sat_ema_active_t,
        last_gu_arrival_t=fields_obj.last_gu_arrival.to(device=kernel_device, dtype=torch.float32),
        last_gu_outflow_t=fields_obj.last_gu_outflow.to(device=kernel_device, dtype=torch.float32),
        gu_drop_t=fields_obj.gu_drop.to(device=kernel_device, dtype=torch.float32),
        uav_drop_t=fields_obj.uav_drop.to(device=kernel_device, dtype=torch.float32),
        sat_drop_active_t=sat_drop_active_t,
        last_gu_to_uav_inflow_by_uav_t=fields_obj.last_gu_to_uav_inflow_by_uav.to(device=kernel_device, dtype=torch.float32),
        last_uav_to_sat_outflow_active_t=last_uav_to_sat_outflow_active_t,
        last_bw_fraction_by_uav_gu_t=fields_obj.last_bw_fraction_by_uav_gu.to(device=kernel_device, dtype=torch.float32),
        last_access_interference_by_uav_t=fields_obj.last_access_interference_by_uav.to(device=kernel_device, dtype=torch.float32),
        last_selected_flag_active_t=last_selected_active_t,
        last_sat_processed_active_t=last_sat_processed_active_t,
        sat_queue_full_t=fields_obj.sat_queue.to(device=kernel_device, dtype=torch.float32),
        sat_ema_full_t=fields_obj.sat_ema.to(device=kernel_device, dtype=torch.float32),
        sat_drop_full_t=fields_obj.sat_drop.to(device=kernel_device, dtype=torch.float32),
        last_sat_processed_full_t=fields_obj.last_sat_processed.to(device=kernel_device, dtype=torch.float32),
        last_selected_flag_full_t=fields_obj.last_selected_mask_by_uav_sat.to(device=kernel_device, dtype=torch.float32),
        access_gain_matrix_t=fields_obj.access_gain_matrix.to(device=kernel_device, dtype=torch.float32),
        access_rate_static_params=access_rate_static_params,
    )
    if max_sat_count > 0:
        temp_world_out.sat_ids[history_row_indices_t] = torch.where(
            active_mask_t,
            active_safe_t,
            torch.full_like(active_safe_t, -1),
        )
    return _structured_world_state_from_tensor_dict(
        {
            "uav_nodes": temp_world_out.uav_nodes,
            "gu_nodes": temp_world_out.gu_nodes,
            "sat_nodes": temp_world_out.sat_nodes,
            "sat_ids": temp_world_out.sat_ids,
            "uav_gu_edges": temp_world_out.uav_gu_edges,
            "uav_sat_edges": temp_world_out.uav_sat_edges,
            "uav_uav_edges": temp_world_out.uav_uav_edges,
            "global_scalars": temp_world_out.global_scalars,
            "gu_mask": temp_world_out.gu_mask,
            "sat_mask": temp_world_out.sat_mask,
            "uav_gu_mask": temp_world_out.uav_gu_mask,
            "uav_sat_mask": temp_world_out.uav_sat_mask,
            "uav_uav_mask": temp_world_out.uav_uav_mask,
            "stage_ids": stage_ids_t.to(dtype=torch.int64),
        },
        device=device,
    )




def _build_fast_bw_step_metrics_tensor_impl(
    *,
    reward_metrics_params: _NativeRewardMetricsStaticParams,
    bw_workload_static_params: _NativeBwWorkloadStaticParams,
    reward_mode: str,
    gu_ema_prev_t: torch.Tensor,
    uav_ema_prev_t: torch.Tensor,
    sat_ema_prev_t: torch.Tensor,
    gu_outflow_t: torch.Tensor,
    uav_outflow_t: torch.Tensor,
    sat_processed_t: torch.Tensor,
    assoc_t: torch.Tensor,
    sat_selection_matrix_t: torch.Tensor,
    gu_queue_before_t: torch.Tensor,
    uav_queue_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    arrivals_t: torch.Tensor,
    gu_queue_after_t: torch.Tensor,
    uav_queue_after_t: torch.Tensor,
    sat_queue_after_t: torch.Tensor,
    gu_drop_t: torch.Tensor,
    gu_expired_t: torch.Tensor,
    uav_drop_t: torch.Tensor,
    sat_drop_t: torch.Tensor,
    last_sat_incoming_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    prev_queue_sum_gu_t: torch.Tensor,
    prev_queue_sum_uav_t: torch.Tensor,
    prev_queue_sum_sat_t: torch.Tensor,
    effective_task_arrival_rate_t: torch.Tensor,
    gu_pos_t: torch.Tensor,
    uav_pos_t: torch.Tensor,
    last_exec_accel_t: torch.Tensor,
    last_energy_cost_t: torch.Tensor,
    global_step_t: torch.Tensor,
    prev_q_norm_active_t: torch.Tensor,
    gu_urgency_risk_t: torch.Tensor,
    downstream_pressure_t: torch.Tensor,
    service_gap_t: torch.Tensor,
    service_gap_risk_t: torch.Tensor,
    intervention_norm_uav_t: torch.Tensor,
    close_risk_uav_t: torch.Tensor,
    danger_imitation_mask_t: torch.Tensor,
    intervention_norm_t: torch.Tensor,
    intervention_rate_t: torch.Tensor,
    intervention_norm_top1_t: torch.Tensor,
    close_risk_t: torch.Tensor,
    danger_imitation_active_rate_t: torch.Tensor,
    collision_t: torch.Tensor,
    t_t: torch.Tensor,
    uav_energy_after_t: torch.Tensor,
    gu_service_capacity_t: torch.Tensor | None = None,
    uav_service_capacity_t: torch.Tensor | None = None,
    sat_service_capacity_t: torch.Tensor | None = None,
    active_sat_ids_t: torch.Tensor | None = None,
) -> _NativeBwMetricsTensorFields:
    reward_p = reward_metrics_params
    ema_decay_raw = float(reward_p.bw_weighted_workload_ema_decay)
    ema_decay = min(max(ema_decay_raw, 0.0), 1.0)
    ema_keep = ema_decay
    ema_fresh = 1.0 - ema_decay
    gu_ema_source_t = gu_outflow_t if gu_service_capacity_t is None else gu_service_capacity_t
    uav_ema_source_t = uav_outflow_t if uav_service_capacity_t is None else uav_service_capacity_t
    sat_ema_source_t = sat_processed_t if sat_service_capacity_t is None else sat_service_capacity_t
    gu_ema_t = (ema_keep * gu_ema_prev_t + ema_fresh * gu_ema_source_t).to(dtype=torch.float32)
    uav_ema_t = (ema_keep * uav_ema_prev_t + ema_fresh * uav_ema_source_t).to(dtype=torch.float32)
    sat_ema_t = (ema_keep * sat_ema_prev_t + ema_fresh * sat_ema_source_t).to(dtype=torch.float32)
    gu_cost_t, uav_cost_t, sat_cost_t = _bw_weighted_workload_device_costs_static_tensor_impl(
        params=bw_workload_static_params,
        gu_ema_t=gu_ema_t,
        uav_ema_t=uav_ema_t,
        sat_ema_t=sat_ema_t,
        assoc_t=assoc_t,
        sat_selection_matrix_t=sat_selection_matrix_t,
    )
    workload_rewards_t = _bw_weighted_workload_rewards_tensor_impl(
        params=bw_workload_static_params,
        gu_cost_t=gu_cost_t,
        uav_cost_t=uav_cost_t,
        sat_cost_t=sat_cost_t,
        active_sat_ids_t=active_sat_ids_t,
        gu_queue_before_t=gu_queue_before_t,
        uav_queue_before_t=uav_queue_before_t,
        sat_queue_before_t=sat_queue_before_t,
        realized_arrival_t=arrivals_t,
        gu_queue_after_t=gu_queue_after_t,
        uav_queue_after_t=uav_queue_after_t,
        sat_queue_after_t=sat_queue_after_t,
        gu_drop_t=gu_drop_t,
        uav_drop_t=uav_drop_t,
        sat_drop_t=sat_drop_t,
        last_gu_outflow_t=gu_outflow_t,
        arrival_ref_t=arrival_ref_t,
    )

    metric_dtype = torch.float32 if bool(reward_p.fast_float32) else torch.float64
    arrival_sum_t = _sum_tensor_float32_semantics(arrivals_t, dim=1)
    arrival_ref_vec_t = _torch_require_positive_reward_ref(
        arrival_ref_t.reshape(-1).to(dtype=metric_dtype),
        name="arrival_ref_bits_per_step",
    )
    outflow_sum_t = _sum_tensor_float32_semantics(gu_outflow_t, dim=1)
    backhaul_sum_t = _sum_tensor_float32_semantics(last_sat_incoming_t, dim=1)
    sat_processed_sum_t = _sum_tensor_float32_semantics(sat_processed_t, dim=1)
    expire_sum_t = _sum_tensor_float32_semantics(gu_expired_t, dim=1)
    gu_drop_sum_t = _sum_tensor_float32_semantics(gu_drop_t, dim=1)
    uav_drop_sum_t = _sum_tensor_float32_semantics(uav_drop_t, dim=1)
    sat_drop_sum_t = _sum_tensor_float32_semantics(sat_drop_t, dim=1)
    drop_sum_active_t = gu_drop_sum_t + uav_drop_sum_t
    drop_sum_t = drop_sum_active_t + sat_drop_sum_t
    q_gu_t = _sum_tensor_float32_semantics(gu_queue_after_t, dim=1)
    q_uav_t = _sum_tensor_float32_semantics(uav_queue_after_t, dim=1)
    q_sat_t = _sum_tensor_float32_semantics(sat_queue_after_t, dim=1)
    q_total_t = q_gu_t + q_uav_t + q_sat_t
    q_total_active_t = q_gu_t + q_uav_t
    service_ratio_t = torch.clamp(_torch_ratio_or_zero(outflow_sum_t, arrival_sum_t), 0.0, 1.0)
    drop_ratio_t = torch.clamp(_torch_ratio_or_zero(drop_sum_t, arrival_sum_t), 0.0, 1.0)
    x_acc_raw_t = outflow_sum_t / arrival_ref_vec_t
    x_rel_raw_t = backhaul_sum_t / arrival_ref_vec_t
    b_pre_t_t = (
        prev_queue_sum_gu_t.to(dtype=torch.float32)
        + prev_queue_sum_uav_t.to(dtype=torch.float32)
    ).to(dtype=metric_dtype)
    g_pre_raw_t = (q_total_active_t - b_pre_t_t) / arrival_ref_vec_t
    d_pre_raw_t = (gu_drop_sum_t + uav_drop_sum_t) / arrival_ref_vec_t
    processed_ratio_eval_raw_t = sat_processed_sum_t / arrival_ref_vec_t
    drop_ratio_eval_raw_t = drop_sum_t / arrival_ref_vec_t
    pre_backlog_steps_eval_raw_t = q_total_active_t / arrival_ref_vec_t
    d_sys_report_raw_t = _torch_ratio_or_zero(q_total_t, sat_processed_sum_t)
    metric_quantum = float(reward_p.summary_metric_quantum)
    x_acc_t = _quantize_semantic_tensor(x_acc_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    x_rel_t = _quantize_semantic_tensor(x_rel_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    g_pre_t = _quantize_semantic_tensor(g_pre_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    d_pre_t = _quantize_semantic_tensor(d_pre_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    processed_ratio_eval_t = _quantize_semantic_tensor(processed_ratio_eval_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    drop_ratio_eval_t = _quantize_semantic_tensor(drop_ratio_eval_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    pre_backlog_steps_eval_t = _quantize_semantic_tensor(pre_backlog_steps_eval_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    d_sys_report_t = _quantize_semantic_tensor(d_sys_report_raw_t, quantum=metric_quantum, out_dtype=metric_dtype)
    sat_overlap_eval_t = _quantize_semantic_tensor(
        _sat_overlap_eval_tensor_impl(sat_selection_matrix_t, num_sat=int(reward_p.num_sat)),
        quantum=metric_quantum,
        out_dtype=metric_dtype,
    )
    overflow_risk_mean_t = (
        gu_urgency_risk_t.mean(dim=1).to(dtype=torch.float32)
        if int(reward_p.num_gu) > 0
        else arrival_sum_t.to(dtype=torch.float32) * 0.0
    )
    downstream_pressure_mean_t = (
        downstream_pressure_t.mean(dim=1).to(dtype=torch.float32)
        if int(reward_p.num_gu) > 0
        else arrival_sum_t.to(dtype=torch.float32) * 0.0
    )
    service_gap_mean_t = (
        service_gap_t.mean(dim=1).to(dtype=torch.float32)
        if int(reward_p.num_gu) > 0
        else arrival_sum_t.to(dtype=torch.float32) * 0.0
    )
    if int(reward_p.num_gu) > 0:
        queue_norm_t = gu_queue_after_t / normalize_scale(float(reward_p.queue_max_gu))
        service_gap_risk_mean_t = (service_gap_risk_t * queue_norm_t).mean(dim=1).to(dtype=torch.float32)
    else:
        service_gap_risk_mean_t = arrival_sum_t.to(dtype=torch.float32) * 0.0

    if reward_mode == "controllable_flow":
        reward_t = (
            float(reward_p.reward_w_access) * x_acc_raw_t
            + float(reward_p.reward_w_relay) * x_rel_raw_t
            - float(reward_p.reward_w_pre_drop) * d_pre_raw_t
            - float(reward_p.reward_w_pre_backlog) * torch.log1p(pre_backlog_steps_eval_raw_t)
            - float(reward_p.reward_w_pre_overflow_risk) * overflow_risk_mean_t
            - float(reward_p.reward_w_pre_service_gap) * service_gap_risk_mean_t
        ).to(dtype=torch.float32)
    elif reward_mode == "sat_relay_processed":
        reward_t = (
            0.5 * x_rel_raw_t
            + 0.5 * processed_ratio_eval_raw_t
            - drop_ratio_eval_raw_t
            - 0.05 * sat_overlap_eval_t.to(dtype=metric_dtype)
        ).to(dtype=torch.float32)
    elif reward_mode == "sat_backhaul_drop":
        reward_t = (
            x_rel_raw_t
            - d_pre_raw_t
            - 0.05 * sat_overlap_eval_t.to(dtype=metric_dtype)
        ).to(dtype=torch.float32)
    elif reward_mode == "throughput_only":
        reward_t = (
            float(reward_p.throughput_only_access_coef) * x_acc_raw_t
            + float(reward_p.throughput_only_backhaul_coef) * x_rel_raw_t
            - max(float(reward_p.throughput_only_gu_queue_coef), 0.0)
            * (q_gu_t / arrival_ref_vec_t)
        ).to(dtype=torch.float32)
    elif reward_mode == "weighted_workload_delta":
        reward_t = workload_rewards_t.delta.to(dtype=torch.float32)
    elif reward_mode == "relative_weighted_workload_delta":
        workload_before_t = torch.clamp(
            workload_rewards_t.delta.to(dtype=torch.float32) - workload_rewards_t.level.to(dtype=torch.float32),
            min=0.0,
        )
        reward_t = (
            workload_rewards_t.delta.to(dtype=torch.float32) / torch.clamp(workload_before_t, min=1.0)
        ).to(dtype=torch.float32)
    elif reward_mode == "weighted_workload_level":
        reward_t = workload_rewards_t.level.to(dtype=torch.float32)
    elif reward_mode == "positive_weighted_workload_level":
        reward_t = workload_rewards_t.positive_level.to(dtype=torch.float32)
    elif reward_mode == "gu_queue_level":
        reward_t = workload_rewards_t.gu_queue_level.to(dtype=torch.float32)
    elif reward_mode == "system_queue_level":
        reward_t = workload_rewards_t.system_queue_level.to(dtype=torch.float32)
    elif reward_mode == "gu_service_queue":
        reward_t = workload_rewards_t.gu_service_queue.to(dtype=torch.float32)
    else:
        q_max_total = max(
            float(reward_p.num_gu) * float(reward_p.queue_max_gu)
            + float(reward_p.num_uav) * float(reward_p.queue_max_uav)
            + float(reward_p.num_sat) * float(reward_p.queue_max_sat),
            NORMALIZATION_DENOM_EPS,
        )
        q_gu_max = normalize_scale(float(reward_p.num_gu) * float(reward_p.queue_max_gu))
        q_uav_max = normalize_scale(float(reward_p.num_uav) * float(reward_p.queue_max_uav))
        q_sat_max = normalize_scale(float(reward_p.num_sat) * float(reward_p.queue_max_sat))
        q_gu_norm_t = q_gu_t / q_gu_max
        q_uav_norm_t = q_uav_t / q_uav_max
        q_sat_norm_t = q_sat_t / q_sat_max
        gu_queue_arrival_steps_t = q_gu_t / arrival_ref_vec_t
        uav_queue_arrival_steps_t = q_uav_t / arrival_ref_vec_t
        sat_queue_arrival_steps_t = q_sat_t / arrival_ref_vec_t
        prev_gu_queue_arrival_steps_t = prev_queue_sum_gu_t.to(dtype=metric_dtype) / arrival_ref_vec_t
        prev_uav_queue_arrival_steps_t = prev_queue_sum_uav_t.to(dtype=metric_dtype) / arrival_ref_vec_t
        prev_sat_queue_arrival_steps_t = prev_queue_sum_sat_t.to(dtype=metric_dtype) / arrival_ref_vec_t
        use_arrival_norm_queue = bool(reward_p.queue_reward_use_arrival_norm)
        use_queue_log_smoothing = bool(reward_p.use_queue_log_smoothing)
        queue_penalty_mode = str(reward_p.queue_penalty_mode or "quadratic").lower()

        def _queue_smooth_tensor(q_value_t: torch.Tensor) -> torch.Tensor:
            if use_arrival_norm_queue:
                q_work_t = torch.clamp(q_value_t.to(dtype=metric_dtype), min=0.0)
                if use_queue_log_smoothing or queue_penalty_mode == "log":
                    k = float(reward_p.queue_log_k)
                    if k > 0.0:
                        return torch.log1p(float(k) * q_work_t) / math.log1p(float(k))
                    return q_work_t
                if queue_penalty_mode == "linear":
                    return q_work_t
                return q_work_t * q_work_t
            q_norm_t = torch.clamp(q_value_t.to(dtype=metric_dtype), min=0.0, max=1.0)
            if use_queue_log_smoothing or queue_penalty_mode == "log":
                k = float(reward_p.queue_log_k)
                if k > 0.0:
                    return torch.log1p(float(k) * q_norm_t) / math.log1p(float(k))
                return q_norm_t
            if queue_penalty_mode == "linear":
                return q_norm_t
            return q_norm_t * q_norm_t

        queue_norm_k = normalize_scale(float(reward_p.queue_norm_k))
        arrival_floor_cfg = float(reward_p.queue_norm_arrival_floor)
        if arrival_floor_cfg > 0.0:
            arrival_floor_t = arrival_sum_t.to(dtype=metric_dtype) * 0.0 + arrival_floor_cfg
        else:
            arrival_floor_t = (
                effective_task_arrival_rate_t.reshape(-1).to(dtype=metric_dtype)
                * float(reward_p.num_gu)
                * float(reward_p.tau0)
            )
        queue_norm_scale_t = float(queue_norm_k) * _torch_require_positive_reward_ref(
            torch.maximum(arrival_sum_t.to(dtype=metric_dtype), arrival_floor_t),
            name="queue arrival normalization reference",
        )
        q_norm_active_t = torch.clamp(q_total_active_t / queue_norm_scale_t, min=0.0, max=1.0)
        prev_q_norm_t = torch.clamp(prev_q_norm_active_t.reshape(-1).to(dtype=metric_dtype), min=0.0, max=1.0)
        q_norm_delta_t = torch.clamp(prev_q_norm_t - q_norm_active_t, min=-1.0, max=1.0)

        prev_q_gu_norm_t = torch.clamp(prev_queue_sum_gu_t.to(dtype=metric_dtype) / q_gu_max, min=0.0, max=1.0)
        prev_q_uav_norm_t = torch.clamp(prev_queue_sum_uav_t.to(dtype=metric_dtype) / q_uav_max, min=0.0, max=1.0)
        prev_q_sat_norm_t = torch.clamp(prev_queue_sum_sat_t.to(dtype=metric_dtype) / q_sat_max, min=0.0, max=1.0)
        if use_arrival_norm_queue:
            queue_delta_gu_t = prev_gu_queue_arrival_steps_t - gu_queue_arrival_steps_t
            queue_delta_uav_t = prev_uav_queue_arrival_steps_t - uav_queue_arrival_steps_t
            queue_delta_sat_t = prev_sat_queue_arrival_steps_t - sat_queue_arrival_steps_t
        else:
            queue_delta_gu_t = torch.clamp(prev_q_gu_norm_t - q_gu_norm_t, min=-1.0, max=1.0)
            queue_delta_uav_t = torch.clamp(prev_q_uav_norm_t - q_uav_norm_t, min=-1.0, max=1.0)
            queue_delta_sat_t = torch.clamp(prev_q_sat_norm_t - q_sat_norm_t, min=-1.0, max=1.0)

        use_active_queue_delta = bool(reward_p.use_active_queue_delta)
        if use_active_queue_delta:
            q_norm_tail_q0 = max(float(reward_p.q_norm_tail_q0), 0.0)
            if q_norm_tail_q0 > 0.0:
                q_norm_tail_excess_t = torch.clamp(q_norm_active_t - q_norm_tail_q0, min=0.0)
                queue_term_t = q_norm_tail_excess_t * q_norm_tail_excess_t
            else:
                queue_term_t = q_norm_active_t
            omega_q_tail = float(reward_p.omega_q_tail)
            queue_weight_base = float(reward_p.omega_q if abs(omega_q_tail) < LOG_RATIO_EPS else omega_q_tail)
            q_delta_weight_base = float(reward_p.eta_q_delta)
            queue_delta_t = q_norm_delta_t
        else:
            queue_gu_raw_t = gu_queue_arrival_steps_t if use_arrival_norm_queue else q_gu_norm_t
            queue_uav_raw_t = uav_queue_arrival_steps_t if use_arrival_norm_queue else q_uav_norm_t
            queue_sat_raw_t = sat_queue_arrival_steps_t if use_arrival_norm_queue else q_sat_norm_t
            queue_gu_t = _queue_smooth_tensor(queue_gu_raw_t)
            queue_uav_t = _queue_smooth_tensor(queue_uav_raw_t)
            queue_sat_t = _queue_smooth_tensor(queue_sat_raw_t)
            w_gu = float(reward_p.omega_q_gu)
            w_uav = float(reward_p.omega_q_uav)
            w_sat = float(reward_p.omega_q_sat)
            w_sum = abs(w_gu) + abs(w_uav) + abs(w_sat)
            if w_sum < NORMALIZATION_DENOM_EPS:
                queue_total_norm_t = q_total_t / arrival_ref_vec_t if use_arrival_norm_queue else q_total_t / q_max_total
                queue_term_t = _queue_smooth_tensor(queue_total_norm_t)
            else:
                queue_term_t = (w_gu * queue_gu_t + w_uav * queue_uav_t + w_sat * queue_sat_t) / w_sum
            queue_delta_mode = str(reward_p.queue_delta_mode or "total").strip().lower()
            if queue_delta_mode == "weighted" and w_sum >= NORMALIZATION_DENOM_EPS:
                queue_delta_t = (w_gu * queue_delta_gu_t + w_uav * queue_delta_uav_t + w_sat * queue_delta_sat_t) / w_sum
                if not use_arrival_norm_queue:
                    queue_delta_t = torch.clamp(queue_delta_t, min=-1.0, max=1.0)
            else:
                prev_sum_t = (
                    prev_queue_sum_gu_t.to(dtype=metric_dtype)
                    + prev_queue_sum_uav_t.to(dtype=metric_dtype)
                    + prev_queue_sum_sat_t.to(dtype=metric_dtype)
                )
                q_delta_den = arrival_ref_vec_t if use_arrival_norm_queue else q_total_t.to(dtype=metric_dtype) * 0.0 + q_max_total
                queue_delta_t = _torch_ratio_or_zero(prev_sum_t - q_total_t, q_delta_den)
                if not use_arrival_norm_queue:
                    queue_delta_t = torch.clamp(queue_delta_t, min=-1.0, max=1.0)
            queue_weight_base = float(reward_p.omega_q)
            q_delta_weight_base = float(reward_p.eta_q_delta)

        if float(reward_p.a_max) > 0.0:
            accel_norm2_t = (
                last_exec_accel_t.to(dtype=metric_dtype).square().sum(dim=2).mean(dim=1)
                / normalize_scale(float(reward_p.a_max) ** 2)
            )
        else:
            accel_norm2_t = q_total_t.to(dtype=metric_dtype) * 0.0

        if int(reward_p.num_gu) > 0:
            q_weights_t = gu_queue_after_t.to(dtype=torch.float32) / normalize_scale(float(reward_p.queue_max_gu))
            w_sum_t = q_weights_t.sum(dim=1, dtype=torch.float32)
            uniform_weights_t = q_weights_t * 0.0 + (1.0 / max(int(reward_p.num_gu), 1))
            normalized_weights_t = _torch_divide_or_default(q_weights_t, w_sum_t[:, None])
            weights_t = torch.where((w_sum_t > NORMALIZATION_DENOM_EPS)[:, None], normalized_weights_t, uniform_weights_t)
            centroid_t = (gu_pos_t.to(dtype=torch.float32) * weights_t[:, :, None]).sum(dim=1, dtype=torch.float32)
            centroid_dists_t = torch.linalg.vector_norm(
                uav_pos_t.to(dtype=torch.float32) - centroid_t[:, None, :],
                dim=2,
            )
            centroid_dist_mean_t = centroid_dists_t.mean(dim=1, dtype=torch.float32).to(dtype=metric_dtype)
            centroid_reward_t = torch.exp(
                -centroid_dists_t / max(float(reward_p.centroid_dist_scale), 1.0e-6)
            ).mean(dim=1, dtype=torch.float32).to(dtype=metric_dtype)
            centroid_dist_mean_t = centroid_dist_mean_t.to(dtype=metric_dtype)
        else:
            centroid_dist_mean_t = q_total_t.to(dtype=metric_dtype) * 0.0
            centroid_reward_t = q_total_t.to(dtype=metric_dtype) * 0.0

        eta_start = float(reward_p.eta_centroid)
        eta_final_cfg = reward_p.eta_centroid_final
        decay_steps = int(reward_p.eta_centroid_decay_steps)
        if eta_final_cfg is not None and decay_steps > 0:
            progress_t = torch.clamp(global_step_t.reshape(-1).to(dtype=metric_dtype) / float(max(decay_steps, 1)), max=1.0)
            centroid_eta_t = eta_start + (float(eta_final_cfg) - eta_start) * progress_t
        else:
            centroid_eta_t = q_total_t.to(dtype=metric_dtype) * 0.0 + eta_start
        if eta_start > NORMALIZATION_DENOM_EPS:
            centroid_transfer_ratio_t = torch.clamp((eta_start - centroid_eta_t) / eta_start, min=0.0, max=1.0)
        else:
            centroid_transfer_ratio_t = q_total_t.to(dtype=metric_dtype) * 0.0
        queue_weight_t = q_total_t.to(dtype=metric_dtype) * 0.0 + queue_weight_base
        q_delta_weight_t = q_total_t.to(dtype=metric_dtype) * 0.0 + q_delta_weight_base
        crash_weight_t = q_total_t.to(dtype=metric_dtype) * 0.0 + float(reward_p.eta_crash)
        if bool(reward_p.centroid_cross_anneal_enabled):
            queue_weight_t = torch.clamp(
                queue_weight_t
                * (1.0 + float(reward_p.centroid_cross_queue_gain) * centroid_transfer_ratio_t),
                min=0.0,
            )
            q_delta_weight_t = torch.clamp(
                q_delta_weight_t
                * (1.0 + float(reward_p.centroid_cross_q_delta_gain) * centroid_transfer_ratio_t),
                min=0.0,
            )
            crash_weight_t = torch.clamp(
                crash_weight_t
                * (1.0 + float(reward_p.centroid_cross_crash_gain) * centroid_transfer_ratio_t),
                min=0.0,
            )
        q_small = max(float(reward_p.tail_q_small), 0.0)
        tail_eta_accel = float(reward_p.eta_accel)
        if q_small > 0.0:
            tail_eta_accel_t = torch.where(
                q_total_active_t <= q_small,
                q_total_t.to(dtype=metric_dtype) * 0.0 + tail_eta_accel * max(float(reward_p.tail_eta_accel_gain), 0.0),
                q_total_t.to(dtype=metric_dtype) * 0.0 + tail_eta_accel,
            )
        else:
            tail_eta_accel_t = q_total_t.to(dtype=metric_dtype) * 0.0 + tail_eta_accel
        term_service_t = float(reward_p.eta_service) * (outflow_sum_t / arrival_ref_vec_t)
        eta_drop_gu = float(reward_p.eta_drop_gu)
        eta_drop_uav = float(reward_p.eta_drop_uav)
        eta_drop_sat = float(reward_p.eta_drop_sat)
        drop_event_t = (drop_sum_t > RUNTIME_RATIO_ZERO_TOL).to(dtype=metric_dtype)
        term_drop_gu_t = -eta_drop_gu * (gu_drop_sum_t / arrival_ref_vec_t)
        term_drop_uav_t = -eta_drop_uav * (uav_drop_sum_t / arrival_ref_vec_t)
        term_drop_sat_t = -eta_drop_sat * (sat_drop_sum_t / arrival_ref_vec_t)
        term_drop_step_t = -float(reward_p.eta_drop_step) * drop_event_t
        term_drop_t = term_drop_gu_t + term_drop_uav_t + term_drop_sat_t + term_drop_step_t
        term_queue_t = -queue_weight_t * queue_term_t
        term_q_delta_t = q_delta_weight_t * queue_delta_t
        term_centroid_t = centroid_eta_t * centroid_reward_t
        term_accel_t = -tail_eta_accel_t * accel_norm2_t
        term_close_risk_t = -max(float(reward_p.eta_close_risk), 0.0) * close_risk_t.to(dtype=metric_dtype)
        if bool(reward_p.energy_enabled) and bool(reward_p.use_energy_reward):
            if str(reward_p.energy_model or "simple") == "rotor":
                v_max = float(reward_p.v_max)
                p_fly = (
                    float(reward_p.rotor_p0) * (1.0 + 3.0 * (v_max**2) / (float(reward_p.rotor_u_tip) ** 2))
                    + float(reward_p.rotor_pi)
                    * math.sqrt(
                        math.sqrt(1.0 + (v_max**4) / (4.0 * (float(reward_p.rotor_v0) ** 4)))
                        - (v_max**2) / (2.0 * (float(reward_p.rotor_v0) ** 2))
                    )
                    + 0.5
                    * float(reward_p.rotor_d0)
                    * float(reward_p.rotor_rho)
                    * float(reward_p.rotor_s)
                    * float(reward_p.rotor_area)
                    * (v_max**3)
                )
            else:
                p_fly = float(reward_p.p_fly_base) + float(reward_p.p_fly_coeff) * (float(reward_p.v_max) ** 2)
            p_max = p_fly + float(reward_p.p_comm_link) * max(1, int(reward_p.n_rf))
            r_energy_t = -_torch_ratio_or_zero(last_energy_cost_t.to(dtype=metric_dtype).mean(dim=1), p_max)
            term_energy_t = float(reward_p.omega_e) * r_energy_t
        else:
            term_energy_t = q_total_t.to(dtype=metric_dtype) * 0.0
        raw_reward_t = (
            term_service_t
            + float(reward_p.eta_throughput_access) * (outflow_sum_t / arrival_ref_vec_t)
            + float(reward_p.eta_throughput_backhaul) * (backhaul_sum_t / arrival_ref_vec_t)
            + term_drop_t
            + term_queue_t
            + term_q_delta_t
            + term_centroid_t
            + term_accel_t
            + term_close_risk_t
            + term_energy_t
        )
        collision_penalty_t = torch.where(
            collision_t.to(dtype=torch.bool),
            -crash_weight_t,
            crash_weight_t * 0.0,
        )
        battery_penalty_t = torch.where(
            bool(reward_p.energy_enabled) & torch.any(uav_energy_after_t <= 0.0, dim=1),
            raw_reward_t * 0.0 - float(reward_p.eta_batt),
            raw_reward_t * 0.0,
        )
        reward_t = raw_reward_t
        if bool(reward_p.use_reward_tanh):
            reward_t = torch.tanh(reward_t)
        reward_t = (reward_t + collision_penalty_t + battery_penalty_t).to(dtype=torch.float32)
        reward_raw_t = raw_reward_t.to(dtype=torch.float32)
        centroid_dist_mean_out_t = centroid_dist_mean_t.to(dtype=torch.float32)
        centroid_reward_out_t = centroid_reward_t.to(dtype=torch.float32)
        q_norm_active_out_t = q_norm_active_t.to(dtype=torch.float32)
        prev_q_norm_active_out_t = prev_q_norm_t.to(dtype=torch.float32)
        queue_delta_out_t = queue_delta_t.to(dtype=torch.float32)
        term_service_out_t = term_service_t.to(dtype=torch.float32)
        term_drop_out_t = term_drop_t.to(dtype=torch.float32)
        term_queue_out_t = term_queue_t.to(dtype=torch.float32)
        term_q_delta_out_t = term_q_delta_t.to(dtype=torch.float32)
        term_centroid_out_t = term_centroid_t.to(dtype=torch.float32)
        term_accel_out_t = term_accel_t.to(dtype=torch.float32)
        term_close_risk_out_t = term_close_risk_t.to(dtype=torch.float32)
        term_energy_out_t = term_energy_t.to(dtype=torch.float32)
        collision_penalty_out_t = collision_penalty_t.to(dtype=torch.float32)
        battery_penalty_out_t = battery_penalty_t.to(dtype=torch.float32)
    reward_t = _quantize_semantic_tensor(reward_t, quantum=metric_quantum, out_dtype=torch.float32)
    if reward_mode != "dense":
        zero_extra_t = reward_t.to(dtype=torch.float32) * 0.0
        reward_raw_t = reward_t.to(dtype=torch.float32)
        centroid_dist_mean_out_t = zero_extra_t
        centroid_reward_out_t = zero_extra_t
        q_norm_active_out_t = zero_extra_t
        prev_q_norm_active_out_t = zero_extra_t
        queue_delta_out_t = zero_extra_t
        term_service_out_t = zero_extra_t
        term_drop_out_t = zero_extra_t
        term_queue_out_t = zero_extra_t
        term_q_delta_out_t = zero_extra_t
        term_centroid_out_t = zero_extra_t
        term_accel_out_t = zero_extra_t
        term_close_risk_out_t = zero_extra_t
        term_energy_out_t = zero_extra_t
        collision_penalty_out_t = zero_extra_t
        battery_penalty_out_t = zero_extra_t

    energy_depleted_t = (
        bool(reward_p.energy_enabled)
        and torch.any(uav_energy_after_t <= 0.0, dim=1)
    )
    terminated_t = collision_t.to(dtype=torch.bool) | energy_depleted_t
    truncated_t = (t_t.to(dtype=torch.float32) >= float(reward_p.t_steps - 1))
    return _NativeBwMetricsTensorFields(
        gu_ema=gu_ema_t,
        uav_ema=uav_ema_t,
        sat_ema=sat_ema_t,
        reward=reward_t,
        terminated=terminated_t.to(dtype=torch.bool),
        truncated=truncated_t.to(dtype=torch.bool),
        arrival_sum=arrival_sum_t,
        arrival_ref=arrival_ref_vec_t,
        outflow_sum=outflow_sum_t,
        backhaul_sum=backhaul_sum_t,
        sat_processed_sum=sat_processed_sum_t,
        expire_sum=expire_sum_t,
        gu_drop_sum=gu_drop_sum_t,
        uav_drop_sum=uav_drop_sum_t,
        sat_drop_sum=sat_drop_sum_t,
        drop_sum_active=drop_sum_active_t,
        drop_sum=drop_sum_t,
        q_gu=q_gu_t,
        q_uav=q_uav_t,
        q_sat=q_sat_t,
        q_total=q_total_t,
        q_total_active=q_total_active_t,
        service_ratio=service_ratio_t,
        drop_ratio=drop_ratio_t,
        x_acc=x_acc_t,
        x_rel=x_rel_t,
        b_pre_t=b_pre_t_t,
        g_pre=g_pre_t,
        d_pre=d_pre_t,
        processed_ratio_eval=processed_ratio_eval_t,
        drop_ratio_eval=drop_ratio_eval_t,
        pre_backlog_steps_eval=pre_backlog_steps_eval_t,
        d_sys_report=d_sys_report_t,
        sat_overlap_eval=sat_overlap_eval_t,
        overflow_risk_mean=overflow_risk_mean_t.to(dtype=metric_dtype),
        downstream_pressure_mean=downstream_pressure_mean_t.to(dtype=metric_dtype),
        service_gap_mean=service_gap_mean_t.to(dtype=metric_dtype),
        service_gap_risk_mean=service_gap_risk_mean_t.to(dtype=metric_dtype),
        bw_weighted_workload_delta_reward=workload_rewards_t.delta.to(dtype=torch.float32),
        bw_weighted_workload_level_reward=workload_rewards_t.level.to(dtype=torch.float32),
        bw_gu_queue_level_reward=workload_rewards_t.gu_queue_level.to(dtype=torch.float32),
        bw_system_queue_level_reward=workload_rewards_t.system_queue_level.to(dtype=torch.float32),
        bw_gu_service_queue_reward=workload_rewards_t.gu_service_queue.to(dtype=torch.float32),
        intervention_norm_uav=intervention_norm_uav_t.to(dtype=torch.float32),
        close_risk_uav=close_risk_uav_t.to(dtype=torch.float32),
        danger_imitation_mask=danger_imitation_mask_t.to(dtype=torch.float32),
        intervention_norm=intervention_norm_t.to(dtype=torch.float32),
        intervention_rate=intervention_rate_t.to(dtype=torch.float32),
        intervention_norm_top1=intervention_norm_top1_t.to(dtype=torch.float32),
        close_risk=close_risk_t.to(dtype=torch.float32),
        danger_imitation_active_rate=danger_imitation_active_rate_t.to(dtype=torch.float32),
        collision=collision_t.to(dtype=torch.bool),
        reward_raw=reward_raw_t,
        centroid_dist_mean=centroid_dist_mean_out_t,
        centroid_reward=centroid_reward_out_t,
        q_norm_active=q_norm_active_out_t,
        prev_q_norm_active=prev_q_norm_active_out_t,
        queue_delta=queue_delta_out_t,
        term_service=term_service_out_t,
        term_drop=term_drop_out_t,
        term_queue=term_queue_out_t,
        term_q_delta=term_q_delta_out_t,
        term_centroid=term_centroid_out_t,
        term_accel=term_accel_out_t,
        term_close_risk=term_close_risk_out_t,
        term_energy=term_energy_out_t,
        collision_penalty=collision_penalty_out_t,
        battery_penalty=battery_penalty_out_t,
    )


def _fly_power_torch(
    speed_t: torch.Tensor,
    *,
    params: _NativeBwLinkStaticParams,
) -> torch.Tensor:
    if str(params.energy_model) == "rotor":
        p0 = float(params.rotor_p0)
        pi = float(params.rotor_pi)
        u_tip = float(params.rotor_u_tip)
        v0 = float(params.rotor_v0)
        d0 = float(params.rotor_d0)
        rho = float(params.rotor_rho)
        s = float(params.rotor_s)
        area = float(params.rotor_area)
        speed_sq_t = speed_t.square()
        speed_fourth_t = speed_sq_t.square()
        term1_t = p0 * (1.0 + 3.0 * speed_sq_t / normalize_scale(u_tip * u_tip))
        inner_t = torch.sqrt(1.0 + speed_fourth_t / normalize_scale(4.0 * (v0**4))) - speed_sq_t / normalize_scale(
            2.0 * (v0**2),
        )
        inner_t = torch.clamp(inner_t, min=0.0)
        term2_t = pi * torch.sqrt(inner_t)
        term3_t = 0.5 * d0 * rho * s * area * speed_t * speed_sq_t
        return term1_t + term2_t + term3_t
    return float(params.p_fly_base) + float(params.p_fly_coeff) * speed_t.square()


def _apply_batched_access_rate_static_tensor_impl(
    *,
    params: _NativeAccessRateStaticParams,
    gain_matrix_t: torch.Tensor,
    assoc_t: torch.Tensor,
    candidate_indices_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    bw_action_matrix_t: torch.Tensor,
    gu_queue_before_t: torch.Tensor,
    prev_association_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_device = gain_matrix_t.device
    eta_quantum = float(params.eta_quantum)
    rate_quantum = float(params.rate_quantum)
    work_dtype = torch.float32 if bool(params.fast_float32) else torch.float64
    gain_matrix_work_t = gain_matrix_t.to(dtype=work_dtype)
    bw_action_matrix_work_t = bw_action_matrix_t.to(dtype=work_dtype)
    gu_queue_before_work_t = gu_queue_before_t.to(dtype=work_dtype)
    batch_size, num_gu, num_uav = gain_matrix_t.shape
    del candidate_indices_t, candidate_mask_t
    if batch_size <= 0 or num_uav <= 0 or num_gu <= 0:
        raise RuntimeError("strict access-rate segment requires non-empty full-batch tensors.")
    expected_action_shape = (int(batch_size), int(num_uav), int(num_gu))
    if tuple(bw_action_matrix_work_t.shape) != expected_action_shape:
        raise RuntimeError(
            "BW access-rate tensor path requires full-G actions with shape "
            f"{expected_action_shape}, got {tuple(bw_action_matrix_work_t.shape)}."
        )
    if (
        not torch.is_tensor(candidate_uav_ids_t)
        or candidate_uav_ids_t.dtype != torch.long
        or candidate_uav_ids_t.device != kernel_device
        or tuple(candidate_uav_ids_t.shape) != (1, int(num_uav), 1)
    ):
        raise RuntimeError("strict access-rate segment requires precomputed UAV id constants.")

    uav_ids_t = candidate_uav_ids_t
    assoc_expanded_t = assoc_t.to(dtype=torch.long).unsqueeze(1).expand(-1, num_uav, -1)
    assoc_match_t = (assoc_expanded_t == uav_ids_t) & (assoc_expanded_t >= 0) & (assoc_expanded_t < int(num_uav))
    assoc_count_t = assoc_match_t.to(work_dtype).sum(dim=-1, keepdim=True)
    beta_uniform_t = torch.where(
        assoc_count_t > 0.0,
        assoc_match_t.to(work_dtype) / torch.clamp(assoc_count_t, min=1.0),
        assoc_match_t.to(work_dtype) * 0.0,
    )
    if bool(params.enable_bw_action):
        _validate_full_g_bw_action_tensor(
            bw_action_matrix_t=bw_action_matrix_work_t,
            assoc_t=assoc_t,
            num_uav=int(num_uav),
            context="BW access-rate tensor path",
        )
        betas_t = bw_action_matrix_work_t * assoc_match_t.to(work_dtype)
    else:
        betas_t = beta_uniform_t
    exec_bw_t = torch.where(assoc_match_t, betas_t, betas_t * 0.0)
    beta_active_t = torch.where(assoc_match_t, exec_bw_t, exec_bw_t * 0.0)
    gu_band_fraction_t = beta_active_t.sum(dim=1)

    if bool(params.interference_enabled) and num_gu > 0:
        active_gu_t = (assoc_t >= 0) & (assoc_t < int(num_uav)) & (gu_band_fraction_t > 0.0)
        total_received_t = float(params.gu_tx_power) * (
            gain_matrix_work_t * gu_band_fraction_t.unsqueeze(-1).to(work_dtype) * active_gu_t.unsqueeze(-1).to(work_dtype)
        ).sum(dim=1)
        serving_uav_t = torch.clamp(assoc_t, min=0, max=max(int(num_uav) - 1, 0))
        serving_gain_t = torch.gather(
            gain_matrix_work_t,
            2,
            serving_uav_t.unsqueeze(-1),
        ).squeeze(-1)
        serving_gain_t = torch.where(active_gu_t, serving_gain_t, serving_gain_t * 0.0)
        same_cell_t = total_received_t * 0.0
        same_cell_t.scatter_add_(
            1,
            serving_uav_t,
            float(params.gu_tx_power)
            * torch.where(active_gu_t, serving_gain_t * gu_band_fraction_t, serving_gain_t * 0.0),
        )
        interference_by_uav_t = torch.clamp(total_received_t - same_cell_t, min=0.0)
        interference_by_uav_t = _quantize_semantic_tensor(
            interference_by_uav_t,
            quantum=float(params.interference_quantum),
            out_dtype=work_dtype,
        )
    else:
        interference_by_uav_t = gain_matrix_work_t.sum(dim=1) * 0.0

    gain_by_uav_t = gain_matrix_work_t.permute(0, 2, 1)
    gain_slots_t = gain_by_uav_t
    gain_slots_t = torch.where(assoc_match_t, gain_slots_t, gain_slots_t * 0.0)
    interference_slots_t = interference_by_uav_t.unsqueeze(-1)
    ref_snr_t = _snr_linear_torch(
        power=float(params.gu_tx_power),
        gain_t=gain_slots_t,
        noise_density=float(params.noise_density),
        bandwidth_t=torch.full_like(gain_slots_t, float(params.b_acc)),
        interference_t=interference_slots_t,
        noise_figure_db=float(params.noise_figure_db),
    )
    se_values_t = _quantize_semantic_tensor(
        _access_spectral_efficiency_torch(ref_snr_t, params=params),
        quantum=eta_quantum,
        out_dtype=work_dtype,
    )
    eta_slots_t = torch.where(assoc_match_t, se_values_t, se_values_t * 0.0)

    eff_bw_t = beta_active_t * float(params.b_acc)
    eff_interference_t = (
        beta_active_t * interference_slots_t
        if bool(params.interference_enabled)
        else beta_active_t * 0.0
    )
    eff_snr_t = _snr_linear_torch(
        power=float(params.gu_tx_power),
        gain_t=gain_slots_t,
        noise_density=float(params.noise_density),
        bandwidth_t=eff_bw_t,
        interference_t=eff_interference_t,
        noise_figure_db=float(params.noise_figure_db),
    )
    eff_rate_t = eff_bw_t * _quantize_semantic_tensor(
        _access_spectral_efficiency_torch(eff_snr_t, params=params),
        quantum=eta_quantum,
        out_dtype=work_dtype,
    )
    eff_rate_t = _quantize_semantic_tensor(
        eff_rate_t,
        quantum=rate_quantum,
        out_dtype=work_dtype,
    )
    rates_t = torch.where(assoc_match_t, eff_rate_t, eff_rate_t * 0.0).sum(dim=1)

    if bool(params.enable_bw_action):
        prev_assoc_expanded_t = prev_association_t.to(dtype=torch.long).unsqueeze(1).expand(-1, num_uav, -1)
        q_norm_t = gu_queue_before_work_t.unsqueeze(1).expand(-1, num_uav, -1) / normalize_scale(float(params.queue_max_gu))
        q_norm_t = torch.where(assoc_match_t, q_norm_t, q_norm_t * 0.0)
        prev_assoc_match_t = (prev_assoc_expanded_t == uav_ids_t) & assoc_match_t
        target_weights_t = q_norm_t * (0.5 + eta_slots_t) * (
            1.0 + 0.2 * prev_assoc_match_t.to(work_dtype)
        )
        target_weights_t = torch.where(assoc_match_t, target_weights_t, target_weights_t * 0.0)
        target_denom_t = target_weights_t.sum(dim=-1, keepdim=True)
        target_t = torch.where(
            target_denom_t > 0.0,
            _torch_divide_or_default(target_weights_t, target_denom_t, eps=LOG_RATIO_EPS),
            target_weights_t * 0.0,
        )
        align_t = 1.0 - 0.5 * torch.abs(beta_active_t - target_t).sum(dim=-1)
        align_valid_t = assoc_match_t.any(dim=-1)
        bw_align_sum_t = (align_t * align_valid_t.to(work_dtype)).sum(dim=1)
        bw_align_count_t = align_valid_t.to(work_dtype).sum(dim=1)
    else:
        bw_align_sum_t = gain_matrix_work_t.sum(dim=(1, 2)) * 0.0
        bw_align_count_t = bw_align_sum_t

    if bool(params.enable_bw_action):
        bw_align_t = bw_align_sum_t / torch.clamp(bw_align_count_t, min=1.0)
    else:
        bw_align_t = bw_align_sum_t
    return (
        rates_t.to(dtype=torch.float32),
        exec_bw_t.to(dtype=torch.float32),
        bw_align_t.to(dtype=torch.float32),
        eta_slots_t.to(dtype=torch.float32),
    )


def _apply_batched_access_eta_slots_tensor_impl(
    *,
    gain_matrix_t: torch.Tensor,
    assoc_t: torch.Tensor,
    candidate_indices_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    params: _NativeAccessRateStaticParams,
    candidate_uav_ids_t: torch.Tensor,
) -> torch.Tensor:
    kernel_device = gain_matrix_t.device
    work_dtype = torch.float32 if bool(params.fast_float32) else torch.float64
    eta_quantum = float(params.eta_quantum)
    gain_matrix_work_t = gain_matrix_t.to(dtype=work_dtype)
    batch_size, num_gu, num_uav = gain_matrix_t.shape
    _, _, users_obs_max = candidate_indices_t.shape
    if batch_size <= 0 or num_uav <= 0 or users_obs_max <= 0 or num_gu <= 0:
        raise RuntimeError("strict access eta segment requires non-empty full-batch tensors.")
    if (
        not torch.is_tensor(candidate_uav_ids_t)
        or candidate_uav_ids_t.dtype != torch.long
        or candidate_uav_ids_t.device != kernel_device
        or tuple(candidate_uav_ids_t.shape) != (1, int(num_uav), 1)
    ):
        raise RuntimeError("strict access eta segment requires precomputed UAV id constants.")

    candidate_valid_t = candidate_mask_t.to(dtype=torch.bool) & (candidate_indices_t >= 0) & (candidate_indices_t < int(num_gu))
    candidate_indices_clamped_t = torch.clamp(candidate_indices_t.to(dtype=torch.long), min=0, max=max(int(num_gu) - 1, 0))
    uav_ids_t = candidate_uav_ids_t
    assoc_expanded_t = assoc_t.to(dtype=torch.long).unsqueeze(1).expand(-1, num_uav, -1)
    candidate_assoc_t = torch.gather(assoc_expanded_t, 2, candidate_indices_clamped_t)
    assoc_match_t = candidate_valid_t & (candidate_assoc_t == uav_ids_t)
    assoc_count_t = assoc_match_t.to(work_dtype).sum(dim=-1, keepdim=True)
    betas_t = torch.where(
        assoc_count_t > 0.0,
        assoc_match_t.to(work_dtype) / torch.clamp(assoc_count_t, min=1.0),
        assoc_match_t.to(work_dtype) * 0.0,
    )
    gu_band_fraction_t = gain_matrix_work_t.sum(dim=-1) * 0.0
    gu_band_fraction_t.scatter_add_(
        1,
        candidate_indices_clamped_t.reshape(batch_size, -1),
        torch.where(assoc_match_t, betas_t, betas_t * 0.0).reshape(batch_size, -1),
    )

    if bool(params.interference_enabled) and num_gu > 0:
        active_gu_t = (assoc_t >= 0) & (assoc_t < int(num_uav)) & (gu_band_fraction_t > 0.0)
        total_received_t = float(params.gu_tx_power) * (
            gain_matrix_work_t
            * gu_band_fraction_t.unsqueeze(-1).to(work_dtype)
            * active_gu_t.unsqueeze(-1).to(work_dtype)
        ).sum(dim=1)
        serving_uav_t = torch.clamp(assoc_t, min=0, max=max(int(num_uav) - 1, 0))
        serving_gain_t = torch.gather(gain_matrix_work_t, 2, serving_uav_t.unsqueeze(-1)).squeeze(-1)
        serving_gain_t = torch.where(active_gu_t, serving_gain_t, serving_gain_t * 0.0)
        same_cell_t = total_received_t * 0.0
        same_cell_t.scatter_add_(
            1,
            serving_uav_t,
            float(params.gu_tx_power)
            * torch.where(active_gu_t, serving_gain_t * gu_band_fraction_t, serving_gain_t * 0.0),
        )
        interference_by_uav_t = torch.clamp(total_received_t - same_cell_t, min=0.0)
        interference_by_uav_t = _quantize_semantic_tensor(
            interference_by_uav_t,
            quantum=float(params.interference_quantum),
            out_dtype=work_dtype,
        )
    else:
        interference_by_uav_t = gain_matrix_work_t.sum(dim=1) * 0.0

    gain_by_uav_t = gain_matrix_work_t.permute(0, 2, 1)
    gain_slots_t = torch.gather(gain_by_uav_t, 2, candidate_indices_clamped_t)
    gain_slots_t = torch.where(candidate_valid_t, gain_slots_t, gain_slots_t * 0.0)
    ref_snr_t = _snr_linear_torch(
        power=float(params.gu_tx_power),
        gain_t=gain_slots_t,
        noise_density=float(params.noise_density),
        bandwidth_t=torch.full_like(gain_slots_t, float(params.b_acc)),
        interference_t=interference_by_uav_t.unsqueeze(-1),
        noise_figure_db=float(params.noise_figure_db),
    )
    se_values_t = _quantize_semantic_tensor(
        _access_spectral_efficiency_torch(ref_snr_t, params=params),
        quantum=eta_quantum,
        out_dtype=work_dtype,
    )
    return torch.where(candidate_valid_t, se_values_t, se_values_t * 0.0).to(dtype=torch.float32)


def _apply_batched_bw_link_transition_from_active_cached_tensor_impl(
    *,
    selection_t: torch.Tensor,
    active_sat_ids_t: torch.Tensor,
    gain_active_t: torch.Tensor,
    nu_eff_active_t: torch.Tensor,
    valid_flag_active_t: torch.Tensor,
    uav_vel_xy_t: torch.Tensor,
    uav_energy_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    params: _NativeBwLinkStaticParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_device = selection_t.device
    batch_size, num_uav, max_links = selection_t.shape
    num_sat = int(params.num_sat)
    valid_selection_t = selection_t >= 0

    if bool(params.energy_enabled):
        speed_t = torch.linalg.norm(uav_vel_xy_t, dim=-1)
        p_fly_t = _fly_power_torch(speed_t, params=params)
        link_counts_t = valid_selection_t.to(torch.float32).sum(dim=-1)
        last_energy_cost_t = p_fly_t + float(params.p_comm_link) * link_counts_t
        uav_energy_t = torch.clamp(uav_energy_before_t - last_energy_cost_t * float(params.tau0), min=0.0)
    else:
        last_energy_cost_t = uav_energy_before_t * 0.0
        uav_energy_t = uav_energy_before_t

    selection_clamped_t = torch.clamp(selection_t, min=0, max=max(num_sat - 1, 0))
    selection_one_hot_t = torch.nn.functional.one_hot(
        selection_clamped_t,
        num_classes=num_sat,
    ).to(dtype=torch.float32)
    selection_one_hot_t = selection_one_hot_t * valid_selection_t.unsqueeze(-1).to(dtype=torch.float32)
    counts_t = selection_one_hot_t.sum(dim=(1, 2))

    selected_count_t = counts_t.gather(1, selection_clamped_t.reshape(batch_size, -1)).reshape(batch_size, num_uav, max_links)
    safe_count_t = torch.clamp(selected_count_t, min=1.0)
    b_ul_t = float(params.b_backhaul_per_sat) * max(float(params.b_backhaul_per_sat_scale), 0.0) / safe_count_t

    active_match_t = (active_sat_ids_t[:, None, None, :] == selection_clamped_t.unsqueeze(-1)) & valid_selection_t.unsqueeze(-1)
    active_match_f_t = active_match_t.to(dtype=torch.float32)
    selected_gain_t = (gain_active_t.unsqueeze(2) * active_match_f_t).sum(dim=-1)
    selected_nu_eff_t = (nu_eff_active_t.unsqueeze(2) * active_match_f_t).sum(dim=-1)
    selected_valid_t = ((valid_flag_active_t.unsqueeze(2) > 0.5) & active_match_t).any(dim=-1)

    snr_t = _snr_linear_torch(
        power=float(params.uav_tx_power),
        gain_t=selected_gain_t,
        noise_density=float(params.noise_density),
        bandwidth_t=b_ul_t,
        noise_figure_db=float(params.noise_figure_db),
    )
    if bool(params.doppler_atten_enabled):
        subcarrier_spacing = float(params.subcarrier_spacing)
        if subcarrier_spacing > 0.0:
            snr_t = snr_t * torch.sinc(selected_nu_eff_t / subcarrier_spacing).square()

    se_t = torch.log2(1.0 + snr_t)
    rate_t = b_ul_t * se_t
    if bool(params.doppler_enabled):
        rate_t = torch.where(selected_valid_t, rate_t, rate_t * 0.0)
    rate_t = torch.where(valid_selection_t, rate_t, rate_t * 0.0)

    rate_matrix_t = (selection_one_hot_t * rate_t.unsqueeze(-1).to(dtype=torch.float32)).sum(dim=2)
    rate_matrix_t = _quantize_semantic_tensor(
        rate_matrix_t,
        quantum=float(params.backhaul_rate_quantum),
        out_dtype=torch.float32,
    )

    selected_sat_queue_t = sat_queue_before_t.gather(1, selection_clamped_t.reshape(batch_size, -1)).reshape(batch_size, num_uav, max_links)
    sat_score_t = se_t - 0.5 * (
        selected_sat_queue_t / normalize_scale(float(params.queue_max_sat))
    )
    sat_score_t = torch.where(valid_selection_t, sat_score_t, sat_score_t * 0.0)
    sat_score_sum_t = sat_score_t.sum(dim=(1, 2))
    sat_score_count_t = valid_selection_t.to(torch.float32).sum(dim=(1, 2))
    last_sat_score_t = sat_score_sum_t / torch.clamp(sat_score_count_t, min=1.0)
    return uav_energy_t, last_energy_cost_t, rate_matrix_t, counts_t, last_sat_score_t


def _build_ephemeral_reference_envs_from_runtime_states(
    cfg,
    slot_indices: Sequence[int],
    slot_states: Sequence[dict[str, Any]],
) -> list[SaginParallelEnv]:
    envs: list[SaginParallelEnv] = []
    base_seed_raw = getattr(cfg, "seed", 0)
    try:
        base_seed = int(base_seed_raw)
    except (TypeError, ValueError):
        base_seed = 0
    for slot, state in zip(slot_indices, slot_states):
        env_cfg = replace(cfg, seed=base_seed + int(slot))
        env = SaginParallelEnv(env_cfg)
        env.load_runtime_state(
            copy.deepcopy(state),
            refresh_observation_cache=False,
            refresh_global_state_cache=False,
        )
        envs.append(env)
    return envs


def _apply_batched_bw_post_stats_tensor_impl(
    *,
    sat_pos_t: torch.Tensor,
    selection_t: torch.Tensor,
    uav_ecef_t: torch.Tensor,
    gu_queue_t: torch.Tensor,
    last_gu_outflow_t: torch.Tensor,
    next_arrival_rates_t: torch.Tensor,
    uav_queue_t: torch.Tensor,
    sat_queue_t: torch.Tensor,
    sat_connection_counts_t: torch.Tensor,
    last_gu_service_gap_t: torch.Tensor,
    params: _NativePostStatsSafetyStaticParams,
    uav_orbit_radius: float,
    uav_orbit_radius_sq: float,
    sat_orbit_radius_sq: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    post_p = params
    kernel_device = sat_pos_t.device
    batch_size, num_uav, max_links = selection_t.shape
    valid_selection_t = selection_t >= 0
    selection_clamped_t = torch.clamp(selection_t, min=0)
    selected_sat_pos_t = sat_pos_t.gather(
        1,
        selection_clamped_t.reshape(batch_size, -1, 1).expand(-1, -1, int(sat_pos_t.shape[-1])),
    ).reshape(batch_size, num_uav, max_links, int(sat_pos_t.shape[-1]))
    rel_t = selected_sat_pos_t - uav_ecef_t.unsqueeze(2)
    dist_t = _torch_positive(torch.linalg.norm(rel_t, dim=-1), GEOMETRY_DENOM_EPS)
    arg_t = (
        float(sat_orbit_radius_sq)
        - float(uav_orbit_radius_sq)
        - dist_t.square()
    ) / _torch_positive(2.0 * float(uav_orbit_radius) * dist_t, GEOMETRY_DENOM_EPS)
    arg_t = torch.clamp(arg_t, -1.0, 1.0)
    elev_deg_t = torch.rad2deg(torch.asin(arg_t))

    flat_valid_t = valid_selection_t.reshape(batch_size, -1)
    flat_dist_t = dist_t.reshape(batch_size, -1)
    flat_elev_t = elev_deg_t.reshape(batch_size, -1)
    valid_count_t = flat_valid_t.to(torch.float32).sum(dim=1)
    valid_count_safe_t = torch.clamp(valid_count_t, min=1.0)
    dist_sum_t = (flat_dist_t * flat_valid_t.to(torch.float32)).sum(dim=1)
    elev_sum_t = (flat_elev_t * flat_valid_t.to(torch.float32)).sum(dim=1)
    dist_mean_t = dist_sum_t / valid_count_safe_t
    elev_mean_t = elev_sum_t / valid_count_safe_t
    dist_min_t = torch.where(
        valid_count_t > 0.0,
        flat_dist_t.masked_fill(~flat_valid_t, float("inf")).min(dim=1).values,
        valid_count_t.to(dtype=flat_dist_t.dtype) * 0.0,
    )
    elev_min_t = torch.where(
        valid_count_t > 0.0,
        flat_elev_t.masked_fill(~flat_valid_t, float("inf")).min(dim=1).values,
        valid_count_t.to(dtype=flat_elev_t.dtype) * 0.0,
    )
    sorted_dist_t = flat_dist_t.masked_fill(~flat_valid_t, float("inf")).sort(dim=1).values
    quantile_pos_t = torch.clamp(0.95 * (valid_count_t - 1.0), min=0.0)
    q_lo_t = torch.floor(quantile_pos_t).to(torch.long)
    q_hi_t = torch.ceil(quantile_pos_t).to(torch.long)
    q_w_t = (quantile_pos_t - q_lo_t.to(torch.float32)).unsqueeze(1)
    q_lo_v_t = torch.gather(sorted_dist_t, 1, q_lo_t.unsqueeze(1))
    q_hi_v_t = torch.gather(sorted_dist_t, 1, q_hi_t.unsqueeze(1))
    dist_p95_t = (1.0 - q_w_t) * q_lo_v_t + q_w_t * q_hi_v_t
    dist_p95_t = torch.where(
        valid_count_t.unsqueeze(1) > 0.0,
        dist_p95_t,
        dist_p95_t * 0.0,
    ).squeeze(1)

    if int(post_p.num_gu) > 0:
        queue_cap = normalize_scale(float(post_p.queue_max_gu))
        threshold_frac_raw = float(post_p.overflow_risk_threshold_frac)
        threshold_frac = min(max(threshold_frac_raw, 0.0), 0.999)
        arrival_coef = max(float(post_p.overflow_risk_arrival_coef), 0.0)
        service_coef = max(float(post_p.overflow_risk_service_coef), 0.0)
        q_norm_t = gu_queue_t / queue_cap
        base_rate_t = torch.clamp(next_arrival_rates_t.mean(dim=1, keepdim=True), min=0.0)
        base_arrival_steps_t = _torch_require_positive_reward_ref(
            base_rate_t * float(post_p.tau0),
            name="per-GU arrival reference bits per step",
        )
        arrival_norm_t = (next_arrival_rates_t * float(post_p.tau0)) / base_arrival_steps_t
        service_norm_t = last_gu_outflow_t / base_arrival_steps_t
        projected_pressure_t = q_norm_t + arrival_coef * (arrival_norm_t - 1.0) - service_coef * service_norm_t
        denom = normalize_scale(1.0 - threshold_frac)
        last_gu_urgency_risk_t = torch.clamp((projected_pressure_t - threshold_frac) / denom, min=0.0, max=1.0)

        if int(post_p.num_uav) > 0:
            uav_fill_t = (uav_queue_t / normalize_scale(float(post_p.queue_max_uav))).mean(dim=1)
        else:
            uav_fill_t = valid_count_t * 0.0
        if int(post_p.num_sat) > 0:
            sat_fill_all_t = sat_queue_t / normalize_scale(float(post_p.queue_max_sat))
            active_sat_t = sat_connection_counts_t > 0.0
            active_count_t = active_sat_t.to(torch.float32).sum(dim=1)
            sat_fill_active_t = (sat_fill_all_t * active_sat_t.to(torch.float32)).sum(dim=1) / torch.clamp(active_count_t, min=1.0)
            sat_fill_mean_t = sat_fill_all_t.mean(dim=1)
            sat_fill_t = torch.where(active_count_t > 0.0, sat_fill_active_t, sat_fill_mean_t)
        else:
            sat_fill_t = valid_count_t * 0.0
        pressure_t = torch.clamp(torch.maximum(uav_fill_t, sat_fill_t), min=0.0, max=1.0)
        last_gu_downstream_pressure_t = pressure_t.unsqueeze(1).expand_as(gu_queue_t)

        cap_steps = max(float(post_p.service_gap_cap_steps), 1.0e-6)
        threshold_steps = float(post_p.service_gap_risk_threshold_steps)
        threshold_steps = min(max(threshold_steps, 0.0), cap_steps - 1.0e-6)
        gap_denom = max(cap_steps - threshold_steps, 1.0e-6)
        last_gu_service_gap_risk_t = torch.clamp(
            (last_gu_service_gap_t - threshold_steps) / gap_denom,
            min=0.0,
            max=1.0,
        )
    else:
        last_gu_urgency_risk_t = gu_queue_t * 0.0
        last_gu_downstream_pressure_t = gu_queue_t * 0.0
        last_gu_service_gap_risk_t = gu_queue_t * 0.0
    return (
        valid_count_t,
        dist_mean_t,
        dist_p95_t,
        elev_mean_t,
        elev_min_t,
        last_gu_urgency_risk_t,
        last_gu_downstream_pressure_t,
        last_gu_service_gap_risk_t,
    )


def _apply_batched_close_risk_and_danger_tensor_impl(
    *,
    pos_t: torch.Tensor,
    vel_t: torch.Tensor,
    exec_accel_t: torch.Tensor,
    policy_accel_t: torch.Tensor,
    params: _NativePostStatsSafetyStaticParams,
    upper_pair_mask_t: torch.Tensor,
) -> _NativeCloseRiskSafetyTensorFields:
    post_p = params
    kernel_device = pos_t.device
    batch_size, num_uav, _ = pos_t.shape
    intervention_delta_t = exec_accel_t - policy_accel_t
    intervention_norms_t = torch.linalg.vector_norm(intervention_delta_t, dim=-1)
    if float(post_p.a_max) > 0.0:
        intervention_norm_uav_t = _torch_ratio_or_zero(intervention_norms_t, float(post_p.a_max))
    else:
        intervention_norm_uav_t = intervention_norms_t * 0.0
    intervention_norm_t = (
        intervention_norm_uav_t.mean(dim=-1)
        if num_uav > 0
        else intervention_norms_t.sum(dim=-1).to(dtype=torch.float32) * 0.0
    )
    intervention_rate_t = (
        (intervention_norms_t > 1.0e-6).to(torch.float32).mean(dim=-1)
        if num_uav > 0
        else intervention_norm_t * 0.0
    )
    intervention_norm_top1_t = (
        intervention_norm_uav_t.max(dim=-1).values
        if num_uav > 0
        else intervention_norm_t * 0.0
    )

    close_risk_t = intervention_norm_t * 0.0
    close_risk_uav_t = intervention_norms_t.to(dtype=torch.float32) * 0.0
    collision_t = close_risk_t.to(dtype=torch.bool)

    if num_uav >= 2:
        if (
            not torch.is_tensor(upper_pair_mask_t)
            or upper_pair_mask_t.dtype != torch.bool
            or upper_pair_mask_t.device != kernel_device
            or tuple(upper_pair_mask_t.shape) != (1, num_uav, num_uav)
        ):
            raise RuntimeError("strict close-risk segment requires a precomputed upper-pair mask.")
        diff_t = pos_t.unsqueeze(2) - pos_t.unsqueeze(1)
        dist_t = torch.linalg.vector_norm(diff_t, dim=-1)
        pair_mask_t = upper_pair_mask_t
        valid_pair_t = pair_mask_t & (dist_t > 1.0e-6)
        collision_t = (valid_pair_t & (dist_t < float(post_p.d_safe))).any(dim=(1, 2))

        danger_imitation_enabled = bool(post_p.danger_imitation_enabled)
        danger_trigger_mode = str(post_p.danger_imitation_trigger_mode or "intervention_any").strip().lower()
        if danger_trigger_mode not in {"risk_or_intervention", "intervention_any", "intervention_threshold"}:
            danger_trigger_mode = "intervention_any"
        need_close_risk_stats = bool(post_p.close_risk_enabled) or (
            danger_imitation_enabled and danger_trigger_mode == "risk_or_intervention"
        )

        if need_close_risk_stats:
            d_alert = float(post_p.avoidance_alert_factor) * float(post_p.d_safe)
            raw_prealert_factor = post_p.avoidance_prealert_factor
            trigger_dist = d_alert
            if raw_prealert_factor is not None:
                trigger_dist = max(float(raw_prealert_factor) * float(post_p.d_safe), d_alert)

            prealert_mode = str(post_p.avoidance_prealert_mode or "distance").strip().lower()
            if prealert_mode not in {"distance", "ttc"}:
                prealert_mode = "distance"
            if prealert_mode == "ttc":
                raw_prealert_dist_cap = post_p.avoidance_prealert_dist_cap
                if raw_prealert_dist_cap is not None:
                    trigger_dist = max(float(raw_prealert_dist_cap), d_alert)

            closing_speed_thresh = max(float(post_p.avoidance_prealert_closing_speed), 0.0)
            prealert_ttc_limit = max(float(post_p.avoidance_prealert_ttc), 0.0)
            close_risk_cap = max(float(post_p.close_risk_cap), 0.0)
            dist_denom = max(trigger_dist - d_alert, 1.0e-6)
            close_scale = max(closing_speed_thresh, 1.0e-6)

            rel_vel_t = vel_t.unsqueeze(2) - vel_t.unsqueeze(1)
            closing_speed_t = torch.clamp(
                -torch.sum(diff_t * rel_vel_t, dim=-1) / torch.clamp(dist_t, min=1.0e-6),
                min=0.0,
            )
            active_pair_t = valid_pair_t & (closing_speed_t > float(closing_speed_thresh)) & (
                dist_t < float(trigger_dist)
            )
            if prealert_mode == "ttc":
                if prealert_ttc_limit <= 0.0:
                    active_pair_t = active_pair_t & (dist_t < float(d_alert))
                else:
                    ttc_to_alert_t = (dist_t - float(d_alert)) / torch.clamp(closing_speed_t, min=1.0e-6)
                    active_pair_t = active_pair_t & (
                        (dist_t < float(d_alert)) | (ttc_to_alert_t < float(prealert_ttc_limit))
                    )
            dist_ratio_t = torch.clamp((float(trigger_dist) - dist_t) / float(dist_denom), min=0.0, max=1.0)
            close_ratio_t = torch.clamp(
                (closing_speed_t - float(closing_speed_thresh)) / float(close_scale),
                min=0.0,
                max=float(close_risk_cap),
            )
            pair_risk_t = torch.where(active_pair_t, dist_ratio_t * close_ratio_t, dist_ratio_t * 0.0)
            pair_count_t = valid_pair_t.to(torch.float32).sum(dim=(1, 2))
            close_risk_t = torch.where(
                pair_count_t > 0.0,
                pair_risk_t.sum(dim=(1, 2)) / torch.clamp(pair_count_t, min=1.0),
                pair_count_t * 0.0,
            )
            close_risk_uav_t = (pair_risk_t + pair_risk_t.transpose(1, 2)).max(dim=-1).values

    danger_imitation_enabled = bool(post_p.danger_imitation_enabled)
    danger_trigger_mode = str(post_p.danger_imitation_trigger_mode or "intervention_any").strip().lower()
    if danger_trigger_mode not in {"risk_or_intervention", "intervention_any", "intervention_threshold"}:
        danger_trigger_mode = "intervention_any"
    if danger_imitation_enabled:
        close_risk_thresh = max(float(post_p.danger_imitation_close_risk_thresh), 0.0)
        intervention_thresh = max(float(post_p.danger_imitation_intervention_thresh), 0.0)
        if danger_trigger_mode == "intervention_any":
            danger_mask_t = (intervention_norms_t > 1.0e-6).to(torch.float32)
        elif danger_trigger_mode == "intervention_threshold":
            danger_mask_t = (intervention_norm_uav_t > float(intervention_thresh)).to(torch.float32)
        else:
            danger_mask_t = (
                (close_risk_uav_t > float(close_risk_thresh)) | (intervention_norms_t > 1.0e-6)
            ).to(torch.float32)
    else:
        danger_mask_t = close_risk_uav_t * 0.0
    danger_imitation_active_rate_t = (
        danger_mask_t.mean(dim=-1)
        if num_uav > 0
        else close_risk_t * 0.0
    )

    return _NativeCloseRiskSafetyTensorFields(
        intervention_norm_uav=intervention_norm_uav_t,
        intervention_norm=intervention_norm_t,
        intervention_rate=intervention_rate_t,
        intervention_norm_top1=intervention_norm_top1_t,
        close_risk=close_risk_t,
        close_risk_uav=close_risk_uav_t,
        danger_imitation_mask=danger_mask_t,
        danger_imitation_active_rate=danger_imitation_active_rate_t,
        collision=collision_t,
    )



def _copy_tensor_out_(target: torch.Tensor | None, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
    if target is None:
        return
    out_dtype = dtype if dtype is not None else target.dtype
    target.copy_(value.to(device=target.device, dtype=out_dtype))


def _copy_training_world_rows_(
    *,
    target_world: _NativeTrainingWorldTensorFields | None,
    source_world: _NativeTrainingWorldTensorFields | None,
    target_rows_t: torch.Tensor | None,
    source_rows_t: torch.Tensor | None,
) -> None:
    if target_world is None or source_world is None:
        return
    if not torch.is_tensor(target_rows_t) or not torch.is_tensor(source_rows_t):
        return
    if int(target_rows_t.numel()) <= 0 or int(source_rows_t.numel()) <= 0:
        return
    for field_name in _tensor_field_names(source_world):
        src = getattr(source_world, field_name, None)
        dst = getattr(target_world, field_name, None)
        if not torch.is_tensor(src) or not torch.is_tensor(dst):
            continue
        if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
            raise RuntimeError(
                f"training world row copy shape mismatch for {field_name}: "
                f"source trailing shape {tuple(src.shape[1:])} != target trailing shape {tuple(dst.shape[1:])}."
            )
        dst.index_copy_(
            0,
            target_rows_t.to(device=dst.device, dtype=torch.long),
            src.index_select(0, source_rows_t.to(device=src.device, dtype=torch.long)).to(device=dst.device, dtype=dst.dtype),
        )


def _copy_tensor_dataclass_rows_(
    *,
    target_state: Any,
    source_state: Any,
    target_rows_t: torch.Tensor,
    source_rows_t: torch.Tensor,
) -> None:
    if target_state is None or source_state is None:
        return
    if not torch.is_tensor(target_rows_t) or not torch.is_tensor(source_rows_t):
        return
    if int(target_rows_t.numel()) <= 0 or int(source_rows_t.numel()) <= 0:
        return
    if int(target_rows_t.numel()) != int(source_rows_t.numel()):
        raise RuntimeError("runtime state row copy requires matching target/source row counts.")
    for field_name in _tensor_field_names(source_state):
        src = getattr(source_state, field_name, None)
        dst = getattr(target_state, field_name, None)
        if not torch.is_tensor(src) or not torch.is_tensor(dst):
            continue
        if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
            raise RuntimeError(
                f"runtime state row copy shape mismatch for {field_name}: "
                f"source trailing shape {tuple(src.shape[1:])} != target trailing shape {tuple(dst.shape[1:])}."
            )
        dst.index_copy_(
            0,
            target_rows_t.to(device=dst.device, dtype=torch.long),
            src.index_select(0, source_rows_t.to(device=src.device, dtype=torch.long)).to(
                device=dst.device,
                dtype=dst.dtype,
            ),
        )


def _copy_training_world_rows_where_(
    *,
    target_world: _NativeTrainingWorldTensorFields | None,
    source_world: _NativeTrainingWorldTensorFields | None,
    target_rows_t: torch.Tensor | None,
    source_rows_t: torch.Tensor | None,
    mask_t: torch.Tensor | None,
) -> None:
    if target_world is None or source_world is None:
        return
    if not torch.is_tensor(target_rows_t) or not torch.is_tensor(source_rows_t) or not torch.is_tensor(mask_t):
        return
    if int(target_rows_t.numel()) <= 0 or int(source_rows_t.numel()) <= 0:
        return
    target_rows_l = target_rows_t.to(dtype=torch.long)
    source_rows_l = source_rows_t.to(dtype=torch.long)
    for field_name in _tensor_field_names(source_world):
        src = getattr(source_world, field_name, None)
        dst = getattr(target_world, field_name, None)
        if not torch.is_tensor(src) or not torch.is_tensor(dst):
            continue
        if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
            raise RuntimeError(
                f"conditional training world row copy shape mismatch for {field_name}: "
                f"source trailing shape {tuple(src.shape[1:])} != target trailing shape {tuple(dst.shape[1:])}."
            )
        dst_rows = dst.index_select(0, target_rows_l.to(device=dst.device))
        src_rows = src.index_select(0, source_rows_l.to(device=src.device)).to(device=dst.device, dtype=dst.dtype)
        row_mask = mask_t.reshape(-1).to(device=dst.device, dtype=torch.bool)
        row_mask = row_mask.reshape((int(row_mask.numel()),) + (1,) * (src_rows.ndim - 1))
        merged = torch.where(row_mask, src_rows, dst_rows)
        dst.index_copy_(0, target_rows_l.to(device=dst.device), merged)


def _history_slot_row_indices_tensor_impl(
    *,
    history_slot_t: torch.Tensor | None,
    base_row_ids_t: torch.Tensor | None,
    slot_offset: int = 0,
) -> torch.Tensor | None:
    if not torch.is_tensor(history_slot_t) or not torch.is_tensor(base_row_ids_t):
        return None
    if base_row_ids_t.ndim != 1:
        raise RuntimeError("history row-id tensor must be 1-D.")
    slot_scalar_t = history_slot_t.reshape(-1)
    if int(slot_scalar_t.numel()) <= 0:
        raise RuntimeError("history slot tensor must contain one scalar slot index.")
    slot_t = slot_scalar_t[0].to(
        device=base_row_ids_t.device,
        dtype=torch.long,
    ) + int(slot_offset)
    return base_row_ids_t.to(dtype=torch.long) + slot_t * int(base_row_ids_t.shape[0])

def _read_history_ring_tensor_rows_(
    source: torch.Tensor | None,
    row_indices_t: torch.Tensor | None,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if source is None or row_indices_t is None:
        return None
    value = source.index_select(0, row_indices_t.to(device=source.device, dtype=torch.long))
    if dtype is not None and value.dtype != dtype:
        value = value.to(dtype=dtype)
    return value


def _copy_native_stage_fields_out_(target: _NativeStageTensorFields | None, source: _NativeStageTensorFields) -> None:
    if target is None:
        return
    for field_name in source._fields:
        src = getattr(source, field_name)
        dst = getattr(target, field_name)
        if torch.is_tensor(src) and torch.is_tensor(dst):
            _copy_tensor_out_(dst, src)


def _select_bw_arrival_tape_step_tensors(
    *,
    arrivals_t: torch.Tensor,
    arrival_rates_t: torch.Tensor,
    arrival_rollout_tape_t: torch.Tensor | None,
    arrival_rate_rollout_tape_t: torch.Tensor | None,
    arrival_tape_step_t: torch.Tensor | None,
    hotspot_active_after_rollout_tape_t: torch.Tensor | None = None,
    reset_followup_arrival_rollout_tape_t: torch.Tensor | None = None,
    reset_followup_arrival_rate_rollout_tape_t: torch.Tensor | None = None,
    reset_followup_hotspot_active_after_rollout_tape_t: torch.Tensor | None = None,
    traffic_reset_step_t: torch.Tensor | None = None,
    traffic_reset_ordinal_t: torch.Tensor | None = None,
    env_row_ids_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if (
        not torch.is_tensor(arrival_rollout_tape_t)
        or not torch.is_tensor(arrival_rate_rollout_tape_t)
        or not torch.is_tensor(arrival_tape_step_t)
    ):
        return arrivals_t, arrival_rates_t, None
    max_step = max(int(arrival_rollout_tape_t.shape[0]) - 1, 0)
    step_idx_t = torch.clamp(arrival_tape_step_t.reshape(-1)[0].to(dtype=torch.long), min=0, max=max_step).reshape(1)
    arrivals_step_t = arrival_rollout_tape_t.index_select(0, step_idx_t).squeeze(0).to(dtype=torch.float32)
    rates_step_t = arrival_rate_rollout_tape_t.index_select(0, step_idx_t).squeeze(0).to(dtype=torch.float32)
    active_after_step_t = None
    if torch.is_tensor(hotspot_active_after_rollout_tape_t):
        active_after_step_t = hotspot_active_after_rollout_tape_t.index_select(0, step_idx_t).squeeze(0).to(dtype=torch.int32)
    if (
        torch.is_tensor(reset_followup_arrival_rollout_tape_t)
        and torch.is_tensor(reset_followup_arrival_rate_rollout_tape_t)
        and torch.is_tensor(traffic_reset_step_t)
        and torch.is_tensor(traffic_reset_ordinal_t)
        and torch.is_tensor(env_row_ids_t)
        and int(reset_followup_arrival_rollout_tape_t.ndim) == 4
    ):
        tape_device = reset_followup_arrival_rollout_tape_t.device
        max_reset_ordinal = max(int(reset_followup_arrival_rollout_tape_t.shape[0]) - 1, 0)
        max_followup_step = max(int(reset_followup_arrival_rollout_tape_t.shape[1]) - 1, 0)
        reset_step_t = traffic_reset_step_t.reshape(-1).to(device=tape_device, dtype=torch.long)
        reset_ordinal_t = traffic_reset_ordinal_t.reshape(-1).to(device=tape_device, dtype=torch.long)
        env_ids_t = env_row_ids_t.reshape(-1).to(device=tape_device, dtype=torch.long)
        reset_idx_t = torch.clamp(reset_ordinal_t, min=0, max=max_reset_ordinal)
        step_scalar_t = step_idx_t[0].to(device=tape_device, dtype=torch.long)
        followup_step_t = torch.clamp(step_scalar_t - reset_step_t - 1, min=0, max=max_followup_step)
        reset_arrivals_t = reset_followup_arrival_rollout_tape_t[reset_idx_t, followup_step_t, env_ids_t].to(dtype=torch.float32)
        reset_rates_t = reset_followup_arrival_rate_rollout_tape_t[reset_idx_t, followup_step_t, env_ids_t].to(dtype=torch.float32)
        use_reset_t = (reset_step_t >= 0) & (reset_ordinal_t >= 0)
        use_reset_t = use_reset_t & (reset_step_t < step_scalar_t)
        arrivals_step_t = torch.where(use_reset_t.view(-1, 1), reset_arrivals_t, arrivals_step_t)
        rates_step_t = torch.where(use_reset_t.view(-1, 1), reset_rates_t, rates_step_t)
        if torch.is_tensor(reset_followup_hotspot_active_after_rollout_tape_t):
            reset_active_t = reset_followup_hotspot_active_after_rollout_tape_t[reset_idx_t, followup_step_t, env_ids_t].to(dtype=torch.int32)
            if active_after_step_t is None:
                active_after_step_t = reset_active_t
            else:
                active_after_step_t = torch.where(use_reset_t, reset_active_t, active_after_step_t.reshape(-1).to(dtype=torch.int32))
    return arrivals_step_t, rates_step_t, active_after_step_t


def _select_rollout_tape_step_tensor(
    *,
    fallback_t: torch.Tensor,
    rollout_tape_t: torch.Tensor | None,
    step_t: torch.Tensor | None,
    step_offset: int = 0,
) -> torch.Tensor:
    if not torch.is_tensor(rollout_tape_t) or not torch.is_tensor(step_t):
        return fallback_t
    max_step = max(int(rollout_tape_t.shape[0]) - 1, 0)
    step_idx_t = torch.clamp(
        step_t.reshape(-1)[0].to(dtype=torch.long) + int(step_offset),
        min=0,
        max=max_step,
    ).reshape(1)
    return rollout_tape_t.index_select(0, step_idx_t).squeeze(0).to(dtype=torch.float32)


def _select_rollout_tape_step_tensor_as(
    *,
    rollout_tape_t: torch.Tensor,
    step_t: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(rollout_tape_t) or not torch.is_tensor(step_t):
        raise RuntimeError("native GPU done-aware reset requires rollout reset tapes and a rollout step tensor.")
    max_step = max(int(rollout_tape_t.shape[0]) - 1, 0)
    step_idx_t = torch.clamp(step_t.reshape(-1)[0].to(dtype=torch.long), min=0, max=max_step).reshape(1)
    return rollout_tape_t.index_select(0, step_idx_t).squeeze(0).to(dtype=dtype)


def _select_reset_ordinal_tape_rows_as(
    *,
    rollout_tape_t: torch.Tensor,
    reset_ordinal_t: torch.Tensor,
    env_row_ids_t: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(rollout_tape_t) or not torch.is_tensor(reset_ordinal_t):
        raise RuntimeError("native GPU done-aware reset requires rollout reset tapes and per-env reset ordinals.")
    if int(rollout_tape_t.ndim) < 2:
        raise RuntimeError("native GPU reset ordinal tape must have shape [reset_ordinal, env, ...].")
    env_count = int(rollout_tape_t.shape[1])
    reset_ordinal_t = reset_ordinal_t.reshape(-1).to(device=rollout_tape_t.device, dtype=torch.long)
    if env_row_ids_t is None:
        env_ids_t = torch.arange(int(reset_ordinal_t.numel()), device=rollout_tape_t.device, dtype=torch.long)
    else:
        env_ids_t = env_row_ids_t.reshape(-1).to(device=rollout_tape_t.device, dtype=torch.long)
    if int(env_ids_t.numel()) != int(reset_ordinal_t.numel()):
        raise RuntimeError("native GPU reset ordinal tape env ids must match batch size.")
    max_reset_ordinal = max(int(rollout_tape_t.shape[0]) - 1, 0)
    reset_idx_t = torch.clamp(reset_ordinal_t, min=0, max=max_reset_ordinal)
    env_ids_t = torch.clamp(env_ids_t, min=0, max=max(env_count - 1, 0))
    return rollout_tape_t[reset_idx_t, env_ids_t].to(dtype=dtype)


def _where_reset_rows(mask_t: torch.Tensor, reset_t: torch.Tensor, post_t: torch.Tensor) -> torch.Tensor:
    mask = mask_t.to(device=post_t.device, dtype=torch.bool).reshape(-1)
    view_shape = (int(mask.shape[0]),) + (1,) * (post_t.ndim - 1)
    return torch.where(mask.view(view_shape), reset_t.to(device=post_t.device, dtype=post_t.dtype), post_t)


def _copy_bw_reward_parts_out_(
    reward_part_out_tensors: _NativeBwRewardPartOutBuffers | None,
    metrics: _NativeBwMetricsTensorFields,
) -> None:
    if reward_part_out_tensors is None:
        return
    _copy_tensor_out_(reward_part_out_tensors.service_ratio, metrics.service_ratio.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.drop_ratio, metrics.drop_ratio.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.arrival_ref, metrics.arrival_ref.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.b_pre_steps, metrics.pre_backlog_steps_eval.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.x_acc, metrics.x_acc.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.x_rel, metrics.x_rel.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.g_pre, metrics.g_pre.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.d_pre, metrics.d_pre.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.processed_ratio_eval, metrics.processed_ratio_eval.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.drop_ratio_eval, metrics.drop_ratio_eval.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.pre_backlog_steps_eval, metrics.pre_backlog_steps_eval.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.sat_overlap_eval, metrics.sat_overlap_eval.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.D_sys_report, metrics.d_sys_report.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.drop_sum, metrics.drop_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.gu_queue_sum, metrics.q_gu.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.uav_queue_sum, metrics.q_uav.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.sat_queue_sum, metrics.q_sat.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.queue_total_sum, metrics.q_total.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.drop_sum_active, metrics.drop_sum_active.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.expire_sum, metrics.expire_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.gu_drop_sum, metrics.gu_drop_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.uav_drop_sum, metrics.uav_drop_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.sat_drop_sum, metrics.sat_drop_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.arrival_sum, metrics.arrival_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.outflow_sum, metrics.outflow_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.backhaul_sum, metrics.backhaul_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.sat_processed_sum, metrics.sat_processed_sum.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.collision_event, metrics.collision.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.overflow_risk_mean, metrics.overflow_risk_mean.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.downstream_pressure_mean, metrics.downstream_pressure_mean.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.service_gap_mean, metrics.service_gap_mean.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.service_gap_risk_mean, metrics.service_gap_risk_mean.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(
        reward_part_out_tensors.bw_weighted_workload_delta_reward,
        metrics.bw_weighted_workload_delta_reward.reshape(-1),
        dtype=torch.float32,
    )
    _copy_tensor_out_(
        reward_part_out_tensors.bw_weighted_workload_level_reward,
        metrics.bw_weighted_workload_level_reward.reshape(-1),
        dtype=torch.float32,
    )
    _copy_tensor_out_(reward_part_out_tensors.bw_gu_queue_level_reward, metrics.bw_gu_queue_level_reward.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(
        reward_part_out_tensors.bw_system_queue_level_reward,
        metrics.bw_system_queue_level_reward.reshape(-1),
        dtype=torch.float32,
    )
    _copy_tensor_out_(reward_part_out_tensors.bw_gu_service_queue_reward, metrics.bw_gu_service_queue_reward.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.intervention_norm, metrics.intervention_norm.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.intervention_rate, metrics.intervention_rate.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.intervention_norm_top1, metrics.intervention_norm_top1.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(
        reward_part_out_tensors.danger_imitation_active_rate,
        metrics.danger_imitation_active_rate.reshape(-1),
        dtype=torch.float32,
    )
    _copy_tensor_out_(reward_part_out_tensors.close_risk, metrics.close_risk.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.term_close_risk, metrics.term_close_risk.reshape(-1), dtype=torch.float32)
    _copy_tensor_out_(reward_part_out_tensors.reward_raw, metrics.reward_raw.reshape(-1), dtype=torch.float32)




def _compute_bw_flow_proxy_rewards_tensor_impl(
    *,
    access_rate_static_params: _NativeAccessRateStaticParams,
    bw_queue_deadline_params: _NativeBwQueueDeadlineStaticParams,
    reward_metrics_params: _NativeRewardMetricsStaticParams,
    bw_flow_proxy_params: _NativeBwFlowProxyStaticParams,
    bw_action_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    gain_matrix_t: torch.Tensor,
    assoc_t: torch.Tensor,
    candidate_indices_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    gu_queue_before_t: torch.Tensor,
    prev_association_t: torch.Tensor,
    realized_arrival_t: torch.Tensor,
    uav_queue_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    rate_matrix_t: torch.Tensor,
    sat_compute_rates_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
    candidate_env_ids_t: torch.Tensor,
    candidate_slot_ids_t: torch.Tensor,
    gu_costs_t: torch.Tensor | None = None,
    uav_costs_t: torch.Tensor | None = None,
    sat_costs_t: torch.Tensor | None = None,
) -> torch.Tensor:
    bw_queue_p = bw_queue_deadline_params
    reward_p = reward_metrics_params
    proxy_p = bw_flow_proxy_params
    action_t = bw_action_t.to(dtype=torch.float32)
    batch_size = int(action_t.shape[0])
    if batch_size <= 0:
        raise RuntimeError("strict flow-proxy reward segment requires non-empty batches.")
    access_rates_t, _exec_bw_t, _bw_align_t, _eta_slots_t = _apply_batched_access_rate_static_tensor_impl(
        params=access_rate_static_params,
        gain_matrix_t=gain_matrix_t.to(dtype=torch.float32),
        assoc_t=assoc_t.to(dtype=torch.long),
        candidate_indices_t=candidate_indices_t.to(dtype=torch.long),
        candidate_mask_t=candidate_mask_t.to(dtype=torch.bool),
        bw_action_matrix_t=action_t,
        gu_queue_before_t=gu_queue_before_t.to(dtype=torch.float32),
        prev_association_t=prev_association_t.to(dtype=torch.long),
        candidate_uav_ids_t=candidate_uav_ids_t,
    )

    flow_bits_quantum = float(bw_queue_p.flow_bits_quantum)
    tau0 = float(bw_queue_p.tau0)
    arrival_t = realized_arrival_t.to(dtype=torch.float32)
    gu_queue_before_f = gu_queue_before_t.to(dtype=torch.float32)
    q_gu_before_t = gu_queue_before_f + arrival_t
    gu_service_t = _quantize_semantic_tensor(
        access_rates_t.to(dtype=torch.float32) * tau0,
        quantum=flow_bits_quantum,
        out_dtype=torch.float32,
    )
    gu_outflow_t = torch.minimum(q_gu_before_t, gu_service_t)
    q_gu_after_raw_t = q_gu_before_t - gu_outflow_t
    gu_drop_t = torch.clamp(q_gu_after_raw_t - float(bw_queue_p.queue_max_gu), min=0.0)
    q_gu_after_t = torch.clamp(q_gu_after_raw_t, max=float(bw_queue_p.queue_max_gu))

    assoc_long_t = assoc_t.to(dtype=torch.long)
    num_uav = int(action_t.shape[1])
    valid_assoc_t = (assoc_long_t >= 0) & (assoc_long_t < num_uav)
    safe_assoc_t = assoc_long_t.clamp(min=0, max=max(num_uav - 1, 0))
    inflow_uav_t = (
        F.one_hot(safe_assoc_t, num_classes=max(num_uav, 1)).to(dtype=torch.float32)
        * torch.where(valid_assoc_t, gu_outflow_t, gu_outflow_t * 0.0).unsqueeze(-1)
    ).sum(dim=1)

    uav_queue_before_f = uav_queue_before_t.to(dtype=torch.float32)
    rate_matrix_f = rate_matrix_t.to(dtype=torch.float32)
    q_uav_before_t = uav_queue_before_f + inflow_uav_t
    total_rate_t = rate_matrix_f.sum(dim=-1)
    uav_service_t = _quantize_semantic_tensor(
        total_rate_t * tau0,
        quantum=flow_bits_quantum,
        out_dtype=torch.float32,
    )
    uav_outflow_t = torch.minimum(q_uav_before_t, uav_service_t)
    q_uav_after_raw_t = q_uav_before_t - uav_outflow_t
    uav_drop_t = torch.clamp(q_uav_after_raw_t - float(bw_queue_p.queue_max_uav), min=0.0)
    q_uav_after_t = torch.clamp(q_uav_after_raw_t, max=float(bw_queue_p.queue_max_uav))

    safe_total_rate_t = torch.where(total_rate_t > 0.0, total_rate_t, total_rate_t * 0.0 + 1.0)
    outflow_matrix_t = (rate_matrix_f / safe_total_rate_t.unsqueeze(-1)) * uav_outflow_t.unsqueeze(-1)
    outflow_matrix_t = torch.where(total_rate_t.unsqueeze(-1) > 0.0, outflow_matrix_t, outflow_matrix_t * 0.0)
    sat_incoming_t = outflow_matrix_t.sum(dim=1)
    sat_queue_before_f = sat_queue_before_t.to(dtype=torch.float32)
    q_sat_before_t = sat_queue_before_f + sat_incoming_t
    sat_compute_f = sat_compute_rates_t.to(dtype=torch.float32)
    if sat_compute_f.ndim == 1:
        sat_compute_f = sat_compute_f.unsqueeze(-1)
    sat_service_t = _quantize_semantic_tensor(
        sat_compute_f * tau0,
        quantum=flow_bits_quantum,
        out_dtype=torch.float32,
    )
    sat_processed_t = torch.minimum(q_sat_before_t, sat_service_t)
    q_sat_after_raw_t = q_sat_before_t - sat_processed_t
    sat_drop_t = torch.clamp(q_sat_after_raw_t - float(bw_queue_p.queue_max_sat), min=0.0)
    q_sat_after_t = torch.clamp(q_sat_after_raw_t, max=float(bw_queue_p.queue_max_sat))

    arrival_ref_f = arrival_ref_t.to(dtype=torch.float32).reshape(-1)
    if int(proxy_p.reward_mode_code) == 1:
        x_acc_t = gu_outflow_t.sum(dim=-1) / arrival_ref_f
        x_rel_t = sat_incoming_t.sum(dim=-1) / arrival_ref_f
        d_pre_t = (gu_drop_t.sum(dim=-1) + uav_drop_t.sum(dim=-1)) / arrival_ref_f
        b_pre_steps_t = (q_gu_after_t.sum(dim=-1) + q_uav_after_t.sum(dim=-1)) / arrival_ref_f
        return (
            float(reward_p.reward_w_access) * x_acc_t
            + float(reward_p.reward_w_relay) * x_rel_t
            - float(reward_p.reward_w_pre_drop) * d_pre_t
            - float(reward_p.reward_w_pre_backlog) * torch.log1p(b_pre_steps_t)
        ).to(dtype=torch.float32)

    if int(proxy_p.reward_mode_code) in {2, 3}:
        if gu_costs_t is None or uav_costs_t is None or sat_costs_t is None:
            raise RuntimeError("weighted workload proxy reward requires tensor cost inputs.")
        gu_cost_t = gu_costs_t.to(dtype=torch.float32)
        uav_cost_t = uav_costs_t.to(dtype=torch.float32)
        sat_cost_t = sat_costs_t.to(dtype=torch.float32)
        workload_after_t = (
            (gu_cost_t * q_gu_after_t).sum(dim=-1)
            + (uav_cost_t * q_uav_after_t).sum(dim=-1)
            + (sat_cost_t * q_sat_after_t).sum(dim=-1)
        )
        drop_cost_t = (
            (gu_cost_t * gu_drop_t).sum(dim=-1)
            + (uav_cost_t * uav_drop_t).sum(dim=-1)
            + (sat_cost_t * sat_drop_t).sum(dim=-1)
        )
        if int(proxy_p.reward_mode_code) == 2:
            return (-(workload_after_t + drop_cost_t)).to(dtype=torch.float32)
        workload_before_t = (
            (gu_cost_t * (gu_queue_before_f + arrival_t)).sum(dim=-1)
            + (uav_cost_t * uav_queue_before_f).sum(dim=-1)
            + (sat_cost_t * sat_queue_before_f).sum(dim=-1)
        )
        return (-(workload_after_t - workload_before_t) - drop_cost_t).to(dtype=torch.float32)

    return action_t.sum(dim=(1, 2)).to(dtype=torch.float32) * 0.0


def _apply_batched_bw_flow_proxy_scores_tensor_impl(
    *,
    access_rate_static_params: _NativeAccessRateStaticParams,
    bw_queue_deadline_params: _NativeBwQueueDeadlineStaticParams,
    reward_metrics_params: _NativeRewardMetricsStaticParams,
    bw_flow_proxy_params: _NativeBwFlowProxyStaticParams,
    proxy_base_actions_t: torch.Tensor,
    valid_masks_t: torch.Tensor,
    arrival_ref_t: torch.Tensor,
    gain_matrix_t: torch.Tensor,
    assoc_t: torch.Tensor,
    candidate_indices_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    gu_queue_before_t: torch.Tensor,
    prev_association_t: torch.Tensor,
    realized_arrival_t: torch.Tensor,
    uav_queue_before_t: torch.Tensor,
    sat_queue_before_t: torch.Tensor,
    rate_matrix_t: torch.Tensor,
    sat_compute_rates_t: torch.Tensor,
    candidate_uav_ids_t: torch.Tensor,
    candidate_env_ids_t: torch.Tensor,
    candidate_slot_ids_t: torch.Tensor,
    gu_costs_t: torch.Tensor | None = None,
    uav_costs_t: torch.Tensor | None = None,
    sat_costs_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    proxy_p = bw_flow_proxy_params
    proxy_base_t = proxy_base_actions_t.to(dtype=torch.float32)
    if int(proxy_base_t.numel()) <= 0 or not bool(proxy_p.enabled) or int(proxy_p.reward_mode_code) <= 0:
        return None
    valid_t = valid_masks_t.to(dtype=torch.bool)
    batch_size, num_uav, num_slots = proxy_base_t.shape
    slot_ids_t = torch.arange(num_slots, dtype=torch.long, device=proxy_base_t.device).view(1, 1, num_slots)
    if (
        not torch.is_tensor(candidate_env_ids_t)
        or candidate_env_ids_t.dtype != torch.long
        or candidate_env_ids_t.device != proxy_base_t.device
        or tuple(candidate_env_ids_t.shape) != (batch_size, 1, 1)
        or not torch.is_tensor(candidate_uav_ids_t)
        or candidate_uav_ids_t.dtype != torch.long
        or candidate_uav_ids_t.device != proxy_base_t.device
        or tuple(candidate_uav_ids_t.shape) != (1, num_uav, 1)
    ):
        raise RuntimeError("strict flow-proxy segment requires precomputed env/UAV id constants.")
    proxy_scores_t = proxy_base_t * 0.0
    proxy_mask_t = proxy_base_t * 0.0
    proxy_deltas_t = proxy_base_t * 0.0
    eps = float(proxy_p.eps)
    delta_value = float(proxy_p.aux_delta)
    if delta_value <= eps:
        return proxy_scores_t, proxy_mask_t, proxy_deltas_t

    base_rewards_t = _compute_bw_flow_proxy_rewards_tensor_impl(
        access_rate_static_params=access_rate_static_params,
        bw_queue_deadline_params=bw_queue_deadline_params,
        reward_metrics_params=reward_metrics_params,
        bw_flow_proxy_params=proxy_p,
        bw_action_t=proxy_base_t,
        arrival_ref_t=arrival_ref_t,
        gain_matrix_t=gain_matrix_t,
        assoc_t=assoc_t,
        candidate_indices_t=candidate_indices_t,
        candidate_mask_t=candidate_mask_t,
        gu_queue_before_t=gu_queue_before_t,
        prev_association_t=prev_association_t,
        realized_arrival_t=realized_arrival_t,
        uav_queue_before_t=uav_queue_before_t,
        sat_queue_before_t=sat_queue_before_t,
        rate_matrix_t=rate_matrix_t,
        sat_compute_rates_t=sat_compute_rates_t,
        candidate_uav_ids_t=candidate_uav_ids_t,
        candidate_env_ids_t=candidate_env_ids_t,
        candidate_slot_ids_t=slot_ids_t,
        gu_costs_t=gu_costs_t,
        uav_costs_t=uav_costs_t,
        sat_costs_t=sat_costs_t,
    )

    env_flat_t = candidate_env_ids_t.expand(batch_size, num_uav, num_slots).reshape(-1)
    uav_flat_t = candidate_uav_ids_t.expand(batch_size, num_uav, num_slots).reshape(-1)
    slot_flat_t = slot_ids_t.expand(batch_size, num_uav, num_slots).reshape(-1)
    num_cf = int(env_flat_t.numel())

    base_u_t = proxy_base_t[env_flat_t, uav_flat_t, :]
    valid_u_t = valid_t[env_flat_t, uav_flat_t, :]
    slot_grid_t = slot_ids_t.reshape(1, num_slots)
    target_mask_t = slot_grid_t == slot_flat_t.view(num_cf, 1)
    donor_mask_t = valid_u_t & (~target_mask_t)
    donor_mass_t = torch.sum(torch.where(donor_mask_t, base_u_t, base_u_t * 0.0), dim=-1)
    valid_count_t = valid_t.to(dtype=torch.int32).sum(dim=-1).unsqueeze(-1).expand_as(valid_t).reshape(-1)
    target_valid_t = valid_t.reshape(-1)
    used_delta_t = torch.minimum(
        donor_mass_t * 0.0 + delta_value,
        donor_mass_t * 0.5,
    )
    active_t = target_valid_t & (valid_count_t > 1) & (donor_mass_t > eps) & (used_delta_t > eps)
    used_delta_t = torch.where(active_t, used_delta_t, used_delta_t * 0.0)
    scale_t = (donor_mass_t - used_delta_t) / torch.clamp(donor_mass_t, min=eps)
    cf_u_t = torch.where(donor_mask_t, base_u_t * scale_t.unsqueeze(-1), base_u_t)
    cf_u_t = torch.where(target_mask_t, cf_u_t + used_delta_t.unsqueeze(-1), cf_u_t)
    cf_u_t = torch.where(valid_u_t, cf_u_t, cf_u_t * 0.0)
    cf_norm_t = torch.clamp(
        torch.sum(torch.where(valid_u_t, cf_u_t, cf_u_t * 0.0), dim=-1),
        min=eps,
    )
    cf_u_t = torch.where(valid_u_t, cf_u_t / cf_norm_t.unsqueeze(-1), cf_u_t * 0.0)

    cf_actions_base_t = proxy_base_t.index_select(0, env_flat_t)
    uav_update_mask_t = F.one_hot(uav_flat_t, num_classes=max(num_uav, 1)).to(dtype=torch.bool).unsqueeze(-1)
    cf_actions_t = torch.where(uav_update_mask_t, cf_u_t[:, None, :], cf_actions_base_t)

    cf_gu_costs_t = None if gu_costs_t is None else gu_costs_t.index_select(0, env_flat_t)
    cf_uav_costs_t = None if uav_costs_t is None else uav_costs_t.index_select(0, env_flat_t)
    cf_sat_costs_t = None if sat_costs_t is None else sat_costs_t.index_select(0, env_flat_t)
    cf_rewards_t = _compute_bw_flow_proxy_rewards_tensor_impl(
        access_rate_static_params=access_rate_static_params,
        bw_queue_deadline_params=bw_queue_deadline_params,
        reward_metrics_params=reward_metrics_params,
        bw_flow_proxy_params=proxy_p,
        bw_action_t=cf_actions_t,
        arrival_ref_t=arrival_ref_t.index_select(0, env_flat_t),
        gain_matrix_t=gain_matrix_t.index_select(0, env_flat_t),
        assoc_t=assoc_t.index_select(0, env_flat_t),
        candidate_indices_t=candidate_indices_t.index_select(0, env_flat_t),
        candidate_mask_t=candidate_mask_t.index_select(0, env_flat_t),
        gu_queue_before_t=gu_queue_before_t.index_select(0, env_flat_t),
        prev_association_t=prev_association_t.index_select(0, env_flat_t),
        realized_arrival_t=realized_arrival_t.index_select(0, env_flat_t),
        uav_queue_before_t=uav_queue_before_t.index_select(0, env_flat_t),
        sat_queue_before_t=sat_queue_before_t.index_select(0, env_flat_t),
        rate_matrix_t=rate_matrix_t.index_select(0, env_flat_t),
        sat_compute_rates_t=sat_compute_rates_t.index_select(0, env_flat_t),
        candidate_uav_ids_t=candidate_uav_ids_t,
        candidate_env_ids_t=candidate_env_ids_t.index_select(0, env_flat_t),
        candidate_slot_ids_t=candidate_slot_ids_t,
        gu_costs_t=cf_gu_costs_t,
        uav_costs_t=cf_uav_costs_t,
        sat_costs_t=cf_sat_costs_t,
    )
    base_rewards_flat_t = base_rewards_t.index_select(0, env_flat_t)
    score_flat_t = (cf_rewards_t - base_rewards_flat_t) / torch.clamp(used_delta_t, min=eps)
    score_flat_t = torch.where(active_t, score_flat_t, score_flat_t * 0.0)
    return (
        score_flat_t.reshape(batch_size, num_uav, num_slots).to(dtype=torch.float32),
        active_t.to(dtype=torch.float32).reshape(batch_size, num_uav, num_slots),
        used_delta_t.reshape(batch_size, num_uav, num_slots).to(dtype=torch.float32),
    )


class StructuredBatchEnvCore:
    def __init__(
        self,
        cfg,
        num_envs: int | None = None,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> None:
        if num_envs is None:
            raise ValueError("num_envs is required for native batch core")
        self._cfg = cfg
        self._num_envs = int(num_envs)
        self._native_driver_slots: list[StructuredControlDriver | None] = [None] * self._num_envs
        self._drivers = _StructuredCoreDriverSequence(self)
        self._envs = _StructuredCoreEnvSequence(self)
        self._driver_slots: dict[int, int] = {}
        self._tensor_device = _resolve_structured_tensor_device(cfg, tensor_device)
        self._native_torch_rng = self._make_native_torch_rng()
        self._orbit_model = self._build_orbit_model()
        self._orbit_pos_table, self._orbit_vel_table = self._build_orbit_lookup_tables()
        self._orbit_pos_table_tensor: torch.Tensor | None = None
        self._orbit_vel_table_tensor: torch.Tensor | None = None
        self._refresh_orbit_lookup_tensors()
        self._native_rician_noise_buffer: torch.Tensor | None = None
        self._native_rician_gain_buffer: torch.Tensor | None = None
        self._native_doppler_noise_buffer: torch.Tensor | None = None
        self._native_fading_unity_buffer: torch.Tensor | None = None
        self._native_stage_id_buffers: dict[tuple[int, int, str, int | None], torch.Tensor] = {}
        self._native_scalar_float_buffers: dict[tuple[str, int, str, int | None], torch.Tensor] = {}
        self._native_zero_float_buffers: dict[tuple[str, tuple[int, ...], str, int | None], torch.Tensor] = {}
        self._native_override_input_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype, str, int | None],
            torch.Tensor,
        ] = {}
        self._last_accel_batch_runtime_synced = False
        self._last_sat_batch_runtime_synced = False
        self._runtime_tensor_state = self._allocate_runtime_tensor_state()
        self._native_rollout_fast_random = False
        self._slot_state_payloads: list[dict[str, Any]] = [self._default_slot_state_payload(slot) for slot in range(self._num_envs)]
        self._slot_rngs: list[np.random.Generator] = [self._default_slot_rng(slot) for slot in range(self._num_envs)]
        self._native_rollout_runtime: StructuredGpuRolloutRuntime | None = None
        self._native_hot_replay_runtime: StructuredGpuRolloutRuntime | None = None
        self._native_hot_replay_tensor_state: StructuredBatchRuntimeTensorState | None = None
        self._native_hot_replay_cfg: Any | None = None
        self._native_hot_replay_torch_rng: torch.Generator | None = None
        self._native_hot_replay_bound_kernels: dict[str, Any] | None = None
        self._native_hot_replay_selected_indices: tuple[int, ...] | None = None
        self._native_hot_replay_slot_state_payloads: list[dict[str, Any]] | None = None
        self._native_hot_replay_slot_rngs: list[np.random.Generator] | None = None
        self._native_sub_batch_runtime: StructuredGpuRolloutRuntime | None = None
        self._native_sub_batch_tensor_state: StructuredBatchRuntimeTensorState | None = None
        self._native_sub_batch_cfg: Any | None = None
        self._native_sub_batch_torch_rng: torch.Generator | None = None
        self._native_sub_batch_bound_kernels: dict[str, Any] | None = None
        self._native_sub_batch_selected_indices: tuple[int, ...] | None = None
        self._native_sub_batch_slot_state_payloads: list[dict[str, Any]] | None = None
        self._native_sub_batch_slot_rngs: list[np.random.Generator] | None = None
        self._sync_runtime_orbit_state_from_t()

    @property
    def tensor_device(self) -> torch.device | None:
        return self._tensor_device

    @property
    def drivers(self) -> list[StructuredControlDriver]:
        return self._drivers

    @property
    def envs(self) -> list[SaginParallelEnv]:
        return self._envs

    @property
    def cfg(self):
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def _configure_driver(self, driver: StructuredControlDriver, slot: int) -> None:
        setattr(driver, "_structured_batch_slot_index", int(slot))

    def _env_for_slot(self, slot: int) -> Any:
        return self._slot_view(int(slot))

    def _driver_for_slot(self, slot: int) -> StructuredControlDriver:
        idx = int(slot)
        driver = self._native_driver_slots[idx]
        if driver is None:
            driver = StructuredControlDriver(self._slot_view(idx))
            self._native_driver_slots[idx] = driver
        self._configure_driver(driver, idx)
        return driver

    def _default_slot_rng(self, slot: int) -> np.random.Generator:
        try:
            base_seed = int(getattr(self._cfg, "seed", 0))
        except (TypeError, ValueError):
            base_seed = 0
        return np.random.default_rng(base_seed + int(slot))

    def _make_native_torch_rng(self) -> torch.Generator:
        device = torch.device("cpu") if self._tensor_device is None else torch.device(self._tensor_device)
        generator = torch.Generator(device=device)
        try:
            seed = int(getattr(self._cfg, "seed", 0))
        except (TypeError, ValueError):
            seed = 0
        generator.manual_seed(seed)
        return generator

    def _default_slot_state_payload(self, slot: int) -> dict[str, Any]:
        cfg = self._cfg
        select_k = _sat_action_select_k_from_config(cfg)
        return {
            "episode_idx": 0,
            "traffic_level": int(np.clip(int(getattr(cfg, "traffic_level", 2) or 2), 0, 2)),
            "traffic_level_ratio": float(getattr(cfg, "traffic_level_hard_ratio", 1.0) or 1.0),
            "avoidance_eta_eff": float(getattr(cfg, "avoidance_eta", 0.0) or 0.0),
            "avoidance_collision_rate_ema": 0.0,
            "prev_episode_collision_rate": 0.0,
            "last_avoidance_eta_exec": float(getattr(cfg, "avoidance_eta", 0.0) or 0.0),
            "_episode_collision_count": 0,
            "_episode_step_count": 0,
            "gu_cluster_centers": np.zeros((0, 2), dtype=np.float32),
            "gu_cluster_counts": np.zeros((0,), dtype=np.int32),
            "_arrival_base_scale": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "_hotspot_subsets": [],
            "_hotspot_member_mask": np.zeros((0, int(cfg.num_gu)), dtype=bool),
            "_hotspot_active_idx": -1,
            "last_hotspot_index": -1,
            "last_hotspot_mask": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "gu_deadline_steps": np.ones((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_arrival_rate_vec": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_arrival": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_outflow": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_deadline_slack": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_deadline_risk": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_urgency_risk": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_downstream_pressure": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "last_gu_service_gap_risk": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "gu_drop": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "gu_expired": np.zeros((int(cfg.num_gu),), dtype=np.float32),
            "uav_drop": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "sat_drop": np.zeros((int(cfg.num_sat),), dtype=np.float32),
            "last_exec_bw_alloc": np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32),
            "last_exec_sat_select_mask": np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32),
            "last_exec_sat_indices": np.full((int(cfg.num_uav), select_k), -1, dtype=np.int64),
            "last_energy_cost": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_sat_processed": np.zeros((int(cfg.num_sat),), dtype=np.float32),
            "last_sat_incoming": np.zeros((int(cfg.num_sat),), dtype=np.float32),
            "last_bw_align": 0.0,
            "last_sat_score": 0.0,
            "last_connected_sat_count": 0.0,
            "last_connected_sat_dist_mean": 0.0,
            "last_connected_sat_dist_p95": 0.0,
            "last_connected_sat_elevation_deg_mean": 0.0,
            "last_connected_sat_elevation_deg_min": 0.0,
            "last_visible_raw_counts": np.zeros((int(cfg.num_uav),), dtype=np.int32),
            "last_visible_kept_counts": np.zeros((int(cfg.num_uav),), dtype=np.int32),
            "last_visible_raw_candidates": [[] for _ in range(int(cfg.num_uav))],
            "last_visible_candidates": [[] for _ in range(int(cfg.num_uav))],
            "last_visible_candidate_rank_values": [[] for _ in range(int(cfg.num_uav))],
            "last_visible_candidate_scores": [[] for _ in range(int(cfg.num_uav))],
            "last_visible_candidate_rank_gap_top1_top2": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_candidate_score_gap_top1_top2": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_candidate_dist_std": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_candidate_elevation_std": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_candidate_se_std": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_candidate_queue_std": np.zeros((int(cfg.num_uav),), dtype=np.float32),
            "last_visible_stats": {},
            "last_arrival_rate": float(getattr(cfg, "task_arrival_rate", 0.0) or 0.0),
            "last_filter_active_ratio": 0.0,
            "last_projected_delta_norm_mean": 0.0,
            "last_fallback_count": 0.0,
            "last_boundary_filter_count": 0.0,
            "last_pairwise_filter_count": 0.0,
            "last_pairwise_filter_active_ratio": 0.0,
            "last_pairwise_projected_delta_norm": 0.0,
            "last_pairwise_fallback_count": 0.0,
            "last_pairwise_candidate_infeasible_count": 0.0,
            "last_reward_parts": {},
            "_cached_assoc": np.full((int(cfg.num_gu),), -1, dtype=np.int32),
            "_cached_candidates": [[] for _ in range(int(cfg.num_uav))],
            "_cached_bw_valid_mask": np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32),
            "_cached_eta": np.zeros((int(cfg.num_uav), int(cfg.users_obs_max)), dtype=np.float32),
            "_cached_eta_uav_pos": np.zeros((int(cfg.num_uav), 2), dtype=np.float32),
            "_cached_eta_gu_pos": np.zeros((int(cfg.num_gu), 2), dtype=np.float32),
            "_cached_access_gain_matrix": None,
            "_cached_uav_ecef": np.zeros((int(cfg.num_uav), 3), dtype=np.float32),
            "_cached_uav_vel_ecef": np.zeros((int(cfg.num_uav), 3), dtype=np.float32),
            "_cached_elevation_t": None,
            "_cached_elevation_matrix": None,
            "_cached_backhaul_loss_t": None,
            "_cached_backhaul_loss_matrix": None,
            "_cached_uav_neighbor_t": None,
            "_cached_uav_neighbor_order": None,
            "_cached_sat_obs": np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max), 12 + int(bool(getattr(cfg, "obs_sat_include_sat_cost", False)))), dtype=np.float32),
            "_cached_sat_mask": np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32),
            "_cached_sat_valid_mask": np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32),
            "_cached_global_state": None,
            "_cached_obs_runtime_context": None,
            "_cached_orbit_t": 0,
            "_cached_orbit_pos": _compat_numpy_array(
                self._runtime_tensor_state.sat_pos[int(slot)],
                dtype=np.float32,
            ).copy(),
            "_cached_orbit_vel": _compat_numpy_array(
                self._runtime_tensor_state.sat_vel[int(slot)],
                dtype=np.float32,
            ).copy(),
        }

    def _load_slot_state_payloads(self, indices: Sequence[int], states: Sequence[dict[str, Any]]) -> None:
        if len(indices) != len(states):
            raise ValueError(f"Expected {len(indices)} slot payloads, got {len(states)}.")
        for index, state in zip(indices, states):
            slot = int(index)
            payload = copy.deepcopy(dict(state or {}))
            rng_state = copy.deepcopy(payload.get("rng_bit_generator_state"))
            if rng_state is None:
                rng = self._default_slot_rng(slot)
            else:
                rng = np.random.default_rng()
                rng.bit_generator.state = rng_state
            self._slot_rngs[slot] = rng
            self._slot_state_payloads[slot] = payload

    def _slot_view(self, slot: int) -> _StructuredNativeSlotView:
        return _StructuredNativeSlotView(self, int(slot))

    def _slot_views(self, indices: Sequence[int]) -> list[_StructuredNativeSlotView]:
        return [self._slot_view(int(index)) for index in indices]

    def _native_batch_enabled(self, indices: Sequence[int] | None = None) -> bool:
        del indices
        return True

    @property
    def runtime_state(self) -> StructuredBatchRuntimeTensorState:
        return self._runtime_tensor_state

    @property
    def runtime_tensor_state(self) -> StructuredBatchRuntimeTensorState:
        return self._runtime_tensor_state

    def _sample_gu_arrival_batch_tensor_native(
        self,
        selected_indices: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not selected_indices:
            empty = torch.zeros((0, 0), dtype=torch.float32, device=device)
            return empty, empty
        cfg = self._cfg
        ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
        legacy_arrival_ramp = ramp_steps > 0
        use_arrival_ramp = ablation_flag(cfg, "use_arrival_ramp", default=False) or legacy_arrival_ramp
        traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
        if int(cfg.num_gu) <= 0:
            return None
        full_batch_selected = bool(
            len(selected_indices) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected_indices))
        )
        selected = None
        if not full_batch_selected:
            selected = torch.as_tensor(
                [int(slot) for slot in selected_indices],
                dtype=torch.long,
                device=self._runtime_tensor_state.effective_task_arrival_rate.device,
            )
        rates = None
        if (not use_arrival_ramp) and traffic_model != "sticky_subset_hotspot":
            rates_base = (
                self._runtime_tensor_state.effective_task_arrival_rate
                if full_batch_selected
                else self._runtime_tensor_state.effective_task_arrival_rate.index_select(0, selected)
            )
            rates = rates_base.clamp_min(0.0).to(device=device, dtype=torch.float32).view(-1, 1).expand(-1, int(cfg.num_gu))
        runtime = self.native_rollout_runtime
        arrivals = None
        rollout_arrival_tape = None if runtime is None else runtime.random.arrival_rollout_tape
        rollout_rate_tape = None if runtime is None else runtime.random.arrival_rate_rollout_tape
        if (
            torch.is_tensor(rollout_arrival_tape)
            and torch.is_tensor(rollout_rate_tape)
            and int(runtime.random.step) < int(rollout_arrival_tape.shape[0])
            and tuple(rollout_arrival_tape.shape[1:]) == (int(self._num_envs), int(cfg.num_gu))
        ):
            tape_step = int(runtime.random.step)
            arrivals_full = rollout_arrival_tape[tape_step].to(device=device, dtype=torch.float32)
            rates_full = rollout_rate_tape[tape_step].to(device=device, dtype=torch.float32)
            if full_batch_selected:
                arrivals = arrivals_full
                rates = rates_full
            else:
                if selected is None:
                    raise RuntimeError("native partial-batch arrival selection tensor was not initialized.")
                selected_on_device = selected.to(device=device)
                arrivals = arrivals_full.index_select(0, selected_on_device)
                rates = rates_full.index_select(0, selected_on_device)
        elif use_arrival_ramp or traffic_model == "sticky_subset_hotspot":
            return None
        if arrivals is None:
            if rates is None:
                return None
            if bool(cfg.task_arrival_poisson):
                arrivals = torch.poisson(rates, generator=self._native_torch_rng).to(dtype=torch.float32)
            else:
                arrivals = rates
        bind_direct = bool(
            full_batch_selected
            and runtime is not None
            and bool(getattr(runtime.main, "copy_graph_outputs", False))
        )
        if not bind_direct:
            target_device = self._runtime_tensor_state.last_gu_arrival.device
            if full_batch_selected:
                self._runtime_tensor_state.last_gu_arrival.copy_(arrivals.to(device=target_device, dtype=torch.float32))
                self._runtime_tensor_state.last_gu_arrival_rate_vec.copy_(rates.to(device=target_device, dtype=torch.float32))
            else:
                if selected is None:
                    raise RuntimeError("native partial-batch arrival selection tensor was not initialized.")
                self._runtime_tensor_state.last_gu_arrival[selected] = arrivals.to(device=target_device, dtype=torch.float32)
                self._runtime_tensor_state.last_gu_arrival_rate_vec[selected] = rates.to(device=target_device, dtype=torch.float32)
        if runtime is not None:
            runtime_device = torch.device(runtime.device)
            arrival_device = torch.device(arrivals.device)
            same_runtime_device = arrival_device == runtime_device or (
                arrival_device.type == runtime_device.type == "cuda"
                and (runtime_device.index is None or arrival_device.index in {None, runtime_device.index})
            )
            if same_runtime_device:
                random_buffers = runtime.write_random_arrival_tape(
                    arrivals=arrivals.to(dtype=torch.float32),
                    rates=rates.to(device=arrivals.device, dtype=torch.float32),
                )
                if device.type == "cuda" and torch.is_tensor(random_buffers.arrivals) and torch.is_tensor(
                    random_buffers.arrival_rates
                ):
                    return (
                        random_buffers.arrivals.to(device=device, dtype=torch.float32),
                        random_buffers.arrival_rates.to(device=device, dtype=torch.float32),
                    )
        if device.type == "cuda":
            return arrivals.to(device=device, dtype=torch.float32), rates.to(device=device, dtype=torch.float32)
        rate_mean = rates.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        hotspot_mask_batch = None
        rollout_hotspot_mask_tape = None if runtime is None else runtime.random.hotspot_mask_rollout_tape
        if (
            torch.is_tensor(rollout_hotspot_mask_tape)
            and int(runtime.random.step) < int(rollout_hotspot_mask_tape.shape[0])
            and tuple(rollout_hotspot_mask_tape.shape[1:]) == (int(self._num_envs), int(cfg.num_gu))
        ):
            hotspot_full = rollout_hotspot_mask_tape[int(runtime.random.step)].to(device=device, dtype=torch.float32)
            hotspot_mask_batch = (
                hotspot_full
                if full_batch_selected
                else hotspot_full.index_select(0, selected.to(device=device))
            ).detach().cpu().numpy().astype(np.float32, copy=False)
        for env_index, slot in enumerate(selected_indices):
            meta = self._slot_state_payloads[int(slot)]
            meta["last_arrival_rate"] = float(rate_mean[int(env_index)])
            if hotspot_mask_batch is not None:
                meta["last_hotspot_mask"] = np.asarray(hotspot_mask_batch[int(env_index)], dtype=np.float32).copy()
            else:
                meta["last_hotspot_mask"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        return arrivals.to(device=device, dtype=torch.float32), rates.to(device=device, dtype=torch.float32)

    def _current_expected_arrival_rate_vec_tensor(
        self,
        selected_indices: Sequence[int],
        selected_tensor: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        cfg = self._cfg
        row_count = len(selected_indices)
        if row_count <= 0 or int(cfg.num_gu) <= 0:
            return torch.zeros((row_count, 0), dtype=torch.float32, device=device)

        runtime = self.native_rollout_runtime
        rollout_rate_tape = None if runtime is None else runtime.random.arrival_rate_rollout_tape
        if (
            torch.is_tensor(rollout_rate_tape)
            and int(runtime.random.step) < int(rollout_rate_tape.shape[0])
            and tuple(rollout_rate_tape.shape[1:]) == (int(self._num_envs), int(cfg.num_gu))
        ):
            rates_full_t = rollout_rate_tape[int(runtime.random.step)].to(device=device, dtype=torch.float32)
            if row_count == int(self._num_envs) and all(int(index) == pos for pos, index in enumerate(selected_indices)):
                return rates_full_t
            return rates_full_t.index_select(0, selected_tensor.to(device=device, dtype=torch.long))

        tensor_state = self._runtime_tensor_state
        selected_state_t = selected_tensor.to(device=tensor_state.effective_task_arrival_rate.device, dtype=torch.long)
        base_rate_t = tensor_state.effective_task_arrival_rate.index_select(0, selected_state_t).to(
            device=device,
            dtype=torch.float32,
        ).clamp_min(0.0)

        ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
        use_arrival_ramp = ablation_flag(cfg, "use_arrival_ramp", default=False) or ramp_steps > 0
        if use_arrival_ramp and ramp_steps > 0:
            start = float(getattr(cfg, "arrival_ramp_start", 0.0) or 0.0)
            start = float(np.clip(start, 0.0, 1.0))
            use_global = bool(getattr(cfg, "arrival_ramp_use_global", False))
            counter_source_t = tensor_state.global_step if use_global else tensor_state.t
            counter_t = counter_source_t.index_select(0, selected_state_t).to(device=device, dtype=torch.float32)
            progress_t = torch.clamp(counter_t / float(max(ramp_steps, 1)), max=1.0)
            base_rate_t = base_rate_t * (float(start) + (1.0 - float(start)) * progress_t)

        traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
        if traffic_model != "sticky_subset_hotspot":
            return base_rate_t.view(row_count, 1).expand(row_count, int(cfg.num_gu)).contiguous()

        base_scale_t = tensor_state.arrival_base_scale.index_select(0, selected_state_t).to(
            device=device,
            dtype=torch.float32,
        )
        if tuple(base_scale_t.shape) != (row_count, int(cfg.num_gu)):
            base_scale_t = torch.ones((row_count, int(cfg.num_gu)), dtype=torch.float32, device=device)
        member_mask_t = tensor_state.hotspot_member_mask.index_select(0, selected_state_t).to(
            device=device,
            dtype=torch.float32,
        )
        subset_count_t = tensor_state.hotspot_subset_count.index_select(0, selected_state_t).to(
            device=device,
            dtype=torch.long,
        )
        active_t = tensor_state.hotspot_active_idx.index_select(0, selected_state_t).to(
            device=device,
            dtype=torch.long,
        )
        max_subsets = int(member_mask_t.shape[1]) if member_mask_t.ndim >= 3 else 0
        if max_subsets <= 0:
            return (base_rate_t.view(row_count, 1) * base_scale_t).clamp_min(0.0)

        env_rows_t = torch.arange(row_count, dtype=torch.long, device=device)
        valid_active_t = (active_t >= 0) & (active_t < subset_count_t)
        active_clamped_t = torch.clamp(active_t, min=0, max=max_subsets - 1)
        hot_mask_t = member_mask_t[env_rows_t, active_clamped_t]
        hot_mask_t = torch.where(valid_active_t.view(row_count, 1), hot_mask_t, torch.zeros_like(hot_mask_t))

        rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
        weights_t = base_scale_t * torch.where(
            hot_mask_t > 0.5,
            torch.full_like(base_scale_t, float(rho)),
            torch.ones_like(base_scale_t),
        )
        if bool(getattr(cfg, "arrival_mean_preserve", True)):
            weights_t = _torch_divide_or_default(weights_t, weights_t.mean(dim=1, keepdim=True))
        return (base_rate_t.view(row_count, 1) * weights_t).clamp_min(0.0).to(dtype=torch.float32)

    def set_native_rollout_fast_random(self, enabled: bool) -> None:
        self._native_rollout_fast_random = bool(enabled)

    def clear_native_main_kernel(self) -> None:
        runtime = self.native_rollout_runtime
        if runtime is not None:
            runtime.clear_main_kernel()

    def _native_main_kernel_workspace_active(self, runtime: StructuredGpuRolloutRuntime | None) -> bool:
        if runtime is None:
            return False
        history = getattr(runtime, "rollout_history", getattr(runtime, "history", None))
        main = getattr(runtime, "main", None)
        if history is None or main is None:
            return False
        try:
            history_capacity = int(getattr(history, "capacity", 0) or 0)
            history_num_envs = int(getattr(history, "num_envs", 0) or 0)
            main_num_envs = int(getattr(main, "num_envs", 0) or 0)
        except (TypeError, ValueError):
            return False
        return (
            history_capacity > 0
            and history_num_envs == int(self._num_envs)
            and main_num_envs == int(self._num_envs)
        )

    def begin_native_main_kernel_rollout(self, *, capacity: int, num_envs: int) -> None:
        runtime = self.native_rollout_runtime
        if runtime is None:
            raise RuntimeError("native main-kernel rollout requires a native tensor rollout runtime.")
        if self._tensor_device is None or torch.device(self._tensor_device).type != "cuda":
            raise RuntimeError("final native CUDA main-kernel rollout requires structured_env_tensor_backend='cuda'.")
        selected = list(range(int(self._num_envs)))
        self._refresh_native_main_kernel_typed_domains()
        self._activate_native_main_kernel_strict_cuda_graph_contract(torch.device(self._tensor_device))
        self._require_native_main_kernel_direct_path(selected, context="rollout_begin")
        reuse_workspace = self._native_main_kernel_workspace_active(runtime)
        runtime.begin_rollout_training_ring(capacity=int(capacity), num_envs=int(num_envs))
        if reuse_workspace:
            runtime.begin_main_kernel_rollout()
        else:
            runtime.clear_main_kernel()
        runtime.main.selected_env_mapping = torch.arange(
            int(num_envs),
            dtype=torch.long,
            device=torch.device(self._tensor_device),
        )
        runtime.main.flow_proxy_base_action_mode_code = self._native_flow_proxy_base_action_mode_code(
            getattr(self._cfg, "bw_flow_proxy_base_action_mode", "executed")
        )
        self._refresh_native_main_kernel_typed_domains()
        history_params = self._native_main_kernel_typed_domains().history_output
        runtime.main.copy_graph_outputs = bool(history_params.copy_graph_outputs)
        self._publish_runtime_reset_random_tape(selected)
        self._native_rollout_base_episode_idx_t = self._runtime_tensor_state.episode_idx.detach().clone()
        reset_tape_chunk_rows = max(
            min(
                int(capacity),
                int(getattr(self._cfg, "structured_native_reset_tape_chunk_rows", 8) or 8),
            ),
            1,
        )
        self._prepare_runtime_rollout_random_tapes(capacity=int(capacity), reset_rows=reset_tape_chunk_rows)
        self._runtime_tensor_state.traffic_reset_step.fill_(-1)
        self._runtime_tensor_state.traffic_reset_ordinal.fill_(-1)
        if torch.is_tensor(runtime.random.reset_count):
            runtime.random.reset_count.zero_()
        if not reuse_workspace and bool(getattr(self._cfg, "structured_kernel_compile_cudagraphs", False)) and bool(
            getattr(self._cfg, "structured_kernel_cudagraph_direct_inputs", False)
        ):
            runtime_cache = getattr(self._cfg, "_structured_kernel_runtime_cache", None)
            if isinstance(runtime_cache, dict):
                runtime_cache.clear()
        if not reuse_workspace:
            self._preallocate_native_main_kernel_workspace()

    def _activate_native_main_kernel_strict_cuda_graph_contract(self, tensor_device: torch.device) -> None:
        del tensor_device
        return

    def _preallocate_native_main_kernel_workspace(self) -> None:
        if self._tensor_device is None:
            return
        device = torch.device(self._tensor_device)
        batch_size = int(self._num_envs)
        cfg = self._cfg
        self._native_fading_unity_tensor((batch_size, int(cfg.num_gu), int(cfg.num_uav)), device=device)
        self._native_stage_id_tensor(int(StructuredControlDriver.STAGE_ACCEL), batch_size, device=device)
        self._native_stage_id_tensor(int(StructuredControlDriver.STAGE_SAT), batch_size, device=device)
        self._native_stage_id_tensor(int(StructuredControlDriver.STAGE_BW), batch_size, device=device)
        self._native_zero_float_tensor(
            "access_override_zero_bw_action",
            (batch_size, int(cfg.num_uav), int(cfg.num_gu)),
            device=device,
        )
        self._native_zero_float_tensor(
            "access_override_eta_feature",
            (batch_size, int(cfg.num_uav), int(cfg.num_gu)),
            device=device,
        )
        self._native_zero_float_tensor(
            "doppler_residual_zero",
            (batch_size, int(cfg.num_uav), int(cfg.num_sat)),
            device=device,
        )
        self._native_zero_float_tensor(
            "reward_part_term_close_risk_zero",
            (batch_size,),
            device=device,
        )
        native_sat_compute_rate = _effective_sat_cpu_freq_from_cfg(cfg) / normalize_scale(
            float(cfg.task_cycles_per_bit)
        )
        runtime_sat_compute_rates = self._native_scalar_float_tensor(
            "sat_compute_rate",
            native_sat_compute_rate,
            batch_size,
            device=device,
        )
        bw_link_transition_batch = _NativeBwLinkTransitionTensorFields(
            uav_energy=torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device),
            last_energy_cost=torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device),
            rate_matrix=torch.empty(
                (batch_size, int(cfg.num_uav), int(cfg.num_sat)),
                dtype=torch.float32,
                device=device,
            ),
            sat_loads=torch.empty((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=device),
            last_sat_score=torch.empty((batch_size,), dtype=torch.float32, device=device),
        )
        runtime = self.native_rollout_runtime
        if runtime is not None:
            typed_domains = self._native_main_kernel_typed_domains()
            flow_proxy_params = typed_domains.bw_flow_proxy
            local_obs_params = typed_domains.local_obs
            runtime.main.bw_link_transition = bw_link_transition_batch
            bw_link_transition_override = _NativeBwLinkTransitionTensorFields(
                uav_energy=torch.empty_like(bw_link_transition_batch.uav_energy),
                last_energy_cost=torch.empty_like(bw_link_transition_batch.last_energy_cost),
                rate_matrix=torch.empty_like(bw_link_transition_batch.rate_matrix),
                sat_loads=torch.empty_like(bw_link_transition_batch.sat_loads),
                last_sat_score=torch.empty_like(bw_link_transition_batch.last_sat_score),
            )
            for _override_tensor in bw_link_transition_override:
                _override_tensor.zero_()
            runtime.main.bw_link_transition_override = bw_link_transition_override
            runtime.main.bw_link_transition_override_active = torch.empty((batch_size,), dtype=torch.float32, device=device)
            runtime.main.bw_link_transition_override_active.zero_()
            select_k = _sat_action_select_k_from_config(cfg)
            max_keep = _sat_visible_width_from_config(cfg)
            active_width = min(int(cfg.num_sat), max(int(max_keep), 0) * int(cfg.num_uav))
            sat_obs_width = min(int(active_width), max(int(max_keep), 0))
            row_count = int(batch_size) * int(cfg.num_uav)
            runtime.main.num_envs = int(batch_size)
            runtime.main.num_uav = int(cfg.num_uav)
            runtime.main.num_gu = int(cfg.num_gu)
            runtime.main.num_sat = int(cfg.num_sat)
            runtime.main.users_obs_max = int(cfg.users_obs_max)
            runtime.main.sats_obs_max = int(cfg.sats_obs_max)
            runtime.main.visible_sats_max = int(max_keep)
            runtime.main.sat_num_select = int(select_k)
            runtime.main.sat_max_select = int(select_k)
            runtime.main.accel_direct_supported = bool(
                _can_use_native_accel_batch(cfg)
                and str(getattr(cfg, "boundary_mode", "clip") or "clip").strip().lower() in {"clip", "reflect"}
            )
            runtime.main.accel_offdiag_mask = ~torch.eye(
                int(cfg.num_uav),
                dtype=torch.bool,
                device=device,
            ).view(1, int(cfg.num_uav), int(cfg.num_uav))
            runtime.main.uav_pair_upper_mask = torch.triu(
                torch.ones((int(cfg.num_uav), int(cfg.num_uav)), dtype=torch.bool, device=device),
                diagonal=1,
            ).view(1, int(cfg.num_uav), int(cfg.num_uav))
            runtime.main.accel_uav_index_order = torch.arange(int(cfg.num_uav), dtype=torch.long, device=device).view(
                1,
                1,
                int(cfg.num_uav),
            )
            if int(cfg.num_uav) > 1:
                _nbr_base_t = torch.arange(int(cfg.num_uav) - 1, dtype=torch.long, device=device).view(1, int(cfg.num_uav) - 1)
                _ego_ids_t = torch.arange(int(cfg.num_uav), dtype=torch.long, device=device).view(int(cfg.num_uav), 1)
                runtime.main.accel_neighbor_indices = (_nbr_base_t + (_nbr_base_t >= _ego_ids_t).to(dtype=torch.long)).view(
                    1,
                    int(cfg.num_uav),
                    int(cfg.num_uav) - 1,
                )
            else:
                runtime.main.accel_neighbor_indices = runtime.main.accel_uav_index_order[:, :, :0]
            runtime.main.candidate_slot_ids = torch.arange(int(cfg.users_obs_max), dtype=torch.long, device=device).view(
                1,
                1,
                int(cfg.users_obs_max),
            )
            runtime.main.candidate_env_ids = torch.arange(batch_size, dtype=torch.long, device=device).view(batch_size, 1, 1)
            runtime.main.candidate_gu_ids = torch.arange(int(cfg.num_gu), dtype=torch.long, device=device)
            runtime.main.candidate_uav_ids = torch.arange(int(cfg.num_uav), dtype=torch.long, device=device).view(
                1,
                int(cfg.num_uav),
                1,
            )
            runtime.main.sat_all_ids = torch.arange(int(cfg.num_sat), dtype=torch.long, device=device).view(1, int(cfg.num_sat))
            uav_node_dim = critic_schema.CRITIC_UAV_NODE_DIM
            sat_node_dim = critic_schema.CRITIC_SAT_NODE_DIM
            user_node_dim = critic_schema.CRITIC_GU_NODE_DIM
            uav_orbit_radius = float(cfg.r_earth + cfg.uav_height)
            sat_orbit_radius = float(cfg.r_earth + cfg.sat_height)
            doppler_cap = 0.0
            doppler_rho = 0.0
            doppler_sigma = 0.0
            if bool(getattr(cfg, "doppler_precomp_mode", "none") in {"residual_hz", "residual_ppm"}):
                doppler_cap = _doppler_residual_cap_hz_from_cfg(cfg)
            if doppler_cap > 0.0:
                doppler_rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
                doppler_sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
            runtime.main.bw_action_shape = (batch_size, int(cfg.num_uav), int(cfg.num_gu))
            runtime.main.sat_bw_empty_gu_proxy = self._native_zero_float_tensor(
                "sat_bw_empty_gu_proxy",
                (batch_size, int(cfg.num_gu), 0),
                device=device,
            )
            accel_action_template = torch.empty((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=device)
            accel_logprob_template = torch.empty((batch_size,), dtype=torch.float32, device=device)
            accel_live_logprob_template = torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device)
            sat_subset_index_template = torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.long, device=device)
            sat_action_indices_template = torch.empty((batch_size, int(cfg.num_uav), int(select_k)), dtype=torch.long, device=device)
            sat_logprob_template = torch.empty((batch_size,), dtype=torch.float32, device=device)
            sat_live_logprob_template = torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device)
            bw_action_template = torch.empty(runtime.main.bw_action_shape, dtype=torch.float32, device=device)
            bw_ref_action_template = torch.empty(runtime.main.bw_action_shape, dtype=torch.float32, device=device)
            bw_logprob_template = torch.empty((batch_size,), dtype=torch.float32, device=device)
            runtime.main.bw_reward_mode_active = str(getattr(cfg, "reward_mode", "dense") or "dense").strip().lower()
            runtime.main.bw_flow_proxy_enabled = bool(flow_proxy_params.enabled)
            runtime.main.bw_flow_proxy_reward_mode_code = int(flow_proxy_params.reward_mode_code)
            runtime.main.bw_fading_enabled = bool(cfg.fading_enabled) and _access_fading_mode_code_from_cfg(cfg) == 2
            runtime.main.bw_uav_orbit_radius = uav_orbit_radius
            runtime.main.bw_uav_orbit_radius_sq = float(uav_orbit_radius * uav_orbit_radius)
            runtime.main.bw_sat_orbit_radius_sq = float(sat_orbit_radius * sat_orbit_radius)
            runtime.main.bw_backhaul_gain_const = float(
                (float(cfg.speed_of_light) / (4.0 * math.pi * _backhaul_carrier_freq_from_cfg(cfg))) ** 2
                * float(cfg.uav_tx_gain)
                * float(cfg.sat_rx_gain)
            )
            runtime.main.bw_effective_b_backhaul_per_sat = _effective_b_backhaul_per_sat_from_cfg(cfg)
            runtime.main.bw_inv_a_max = 1.0 / max(float(cfg.a_max), 1.0e-6)
            runtime.main.bw_doppler_cap = float(doppler_cap)
            runtime.main.bw_doppler_rho = float(doppler_rho)
            runtime.main.bw_doppler_sigma = float(doppler_sigma)
            runtime.main.bw_assoc = torch.empty((batch_size, int(cfg.num_gu)), dtype=torch.long, device=device)
            runtime.main.bw_prev_association = torch.empty((batch_size, int(cfg.num_gu)), dtype=torch.long, device=device)
            runtime.main.bw_candidate_indices = torch.empty(
                (batch_size, int(cfg.num_uav), int(cfg.users_obs_max)),
                dtype=torch.long,
                device=device,
            )
            runtime.main.bw_candidate_mask = torch.empty(
                (batch_size, int(cfg.num_uav), int(cfg.users_obs_max)),
                dtype=torch.bool,
                device=device,
            )
            runtime.main.bw_valid_mask = torch.empty(
                (batch_size, int(cfg.num_uav), int(cfg.num_gu)),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.bw_access_gain_matrix = torch.empty(
                (batch_size, int(cfg.num_gu), int(cfg.num_uav)),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.bw_sat_selection_matrix = torch.empty(
                (batch_size, int(cfg.num_uav), int(select_k)),
                dtype=torch.long,
                device=device,
            )
            runtime.main.bw_active_sat_ids = torch.empty((batch_size, int(active_width)), dtype=torch.long, device=device)
            runtime.main.bw_gain_active = torch.empty(
                (batch_size, int(cfg.num_uav), int(active_width)),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.bw_nu_eff_active = torch.empty_like(runtime.main.bw_gain_active)
            runtime.main.bw_valid_flag_active = torch.empty_like(runtime.main.bw_gain_active)
            runtime.main.bw_sat_pos = torch.empty(
                (batch_size, int(cfg.num_sat), 3),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.bw_uav_ecef = torch.empty((batch_size, int(cfg.num_uav), 3), dtype=torch.float32, device=device)
            runtime.main.bw_uav_pos = torch.empty((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=device)
            runtime.main.bw_uav_vel = torch.empty((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=device)
            runtime.main.bw_gu_pos = torch.empty((batch_size, int(cfg.num_gu), 2), dtype=torch.float32, device=device)
            runtime.main.bw_sat_compute_rates = runtime_sat_compute_rates
            subset_members_base_t, _subset_sizes_t = _sat_subset_member_tensor(
                int(sat_obs_width),
                int(select_k),
                device,
            )
            runtime.main.sat_subset_members_base = subset_members_base_t
            runtime.main.sat_subset_sizes = _subset_sizes_t
            runtime.main.sat_visible_width = int(sat_obs_width)
            subset_count = int(subset_members_base_t.shape[0])
            accel_sat_width = _accel_sat_width_from_config(cfg)
            accel_obs_view = StructuredGpuAccelObsView(
                ego_features=torch.empty((row_count, accel_schema.ACCEL_EGO_DIM), dtype=torch.float32, device=device),
                ego_cell=torch.empty((row_count, accel_schema.ACCEL_CELL_DIM), dtype=torch.float32, device=device),
                gu_tokens=torch.empty((row_count, int(cfg.num_gu), accel_schema.ACCEL_GU_TOKEN_DIM), dtype=torch.float32, device=device),
                gu_mask=torch.empty((row_count, int(cfg.num_gu)), dtype=torch.bool, device=device),
                peer_tokens=torch.empty((row_count, max(int(cfg.num_uav) - 1, 0), accel_schema.ACCEL_PEER_TOKEN_DIM), dtype=torch.float32, device=device),
                peer_mask=torch.empty((row_count, max(int(cfg.num_uav) - 1, 0)), dtype=torch.bool, device=device),
                sat_tokens=torch.empty((row_count, accel_sat_width, accel_schema.ACCEL_SAT_TOKEN_DIM), dtype=torch.float32, device=device),
                sat_mask=torch.empty((row_count, accel_sat_width), dtype=torch.bool, device=device),
            )
            accel_obs_view_next = StructuredGpuAccelObsView(
                **{
                    field_name: torch.empty_like(getattr(accel_obs_view, field_name))
                    for field_name in StructuredGpuAccelObsView._tensor_fields
                }
            )
            runtime.main.accel_stage_fields = _allocate_native_main_kernel_stage_fields(
                cfg=cfg,
                local_obs_params=local_obs_params,
                batch_size=batch_size,
                active_width=active_width,
                max_keep=max_keep,
                select_k=select_k,
                device=device,
            )
            accel_stage_fields_next = _allocate_native_main_kernel_stage_fields(
                cfg=cfg,
                local_obs_params=local_obs_params,
                batch_size=batch_size,
                active_width=active_width,
                max_keep=max_keep,
                select_k=select_k,
                device=device,
            )
            runtime.main.accel_stage_field_buffers = (
                runtime.main.accel_stage_fields,
                accel_stage_fields_next,
            )
            runtime.main.accel_active_idx = 0
            runtime.main.accel_live_obs_buffers = (accel_obs_view, accel_obs_view_next)
            runtime.main.live_accel_action = torch.empty_like(accel_action_template)
            runtime.main.live_accel_action.zero_()
            runtime.main.live_accel_latent_action = torch.empty_like(accel_action_template)
            runtime.main.live_accel_latent_action.zero_()
            runtime.main.live_accel_old_logprob = torch.empty_like(accel_live_logprob_template)
            runtime.main.live_accel_old_logprob.zero_()
            sat_obs_view = StructuredGpuSatObsView(
                ego_features=torch.empty((row_count, sat_schema.SAT_EGO_DIM), dtype=torch.float32, device=device),
                demand_features=torch.empty((row_count, sat_schema.SAT_DEMAND_DIM), dtype=torch.float32, device=device),
                role_features=torch.empty((row_count, sat_schema.SAT_ROLE_DIM), dtype=torch.float32, device=device),
                sat_tokens=torch.empty((row_count, sat_obs_width, sat_schema.SAT_TOKEN_DIM), dtype=torch.float32, device=device),
                sat_mask=torch.empty((row_count, sat_obs_width), dtype=torch.bool, device=device),
                sat_valid_mask=torch.empty((row_count, sat_obs_width), dtype=torch.bool, device=device),
                candidate_sat_ids=torch.empty((row_count, sat_obs_width), dtype=torch.long, device=device),
                subset_mask=torch.zeros((row_count, subset_count), dtype=torch.bool, device=device),
                subset_members=subset_members_base_t.unsqueeze(0).expand(row_count, -1, -1),
            )
            runtime.main.live_sat_obs = sat_obs_view
            runtime.main.live_sat_subset_index = torch.empty_like(sat_subset_index_template)
            runtime.main.live_sat_subset_index.fill_(-1)
            runtime.main.live_sat_action_indices = torch.empty_like(sat_action_indices_template)
            runtime.main.live_sat_action_indices.fill_(-1)
            runtime.main.live_sat_old_logprobs_per_agent = torch.empty_like(sat_live_logprob_template)
            runtime.main.live_sat_old_logprobs_per_agent.zero_()
            runtime.main.live_sat_entropy_per_agent = torch.empty_like(sat_live_logprob_template)
            runtime.main.live_sat_entropy_per_agent.zero_()
            runtime.main.sat_stage_fields = _allocate_native_main_kernel_stage_fields(
                cfg=cfg,
                local_obs_params=local_obs_params,
                batch_size=batch_size,
                active_width=active_width,
                max_keep=max_keep,
                select_k=select_k,
                device=device,
            )
            bw_obs_view = StructuredGpuBwObsView(
                ego_features=torch.empty((row_count, bw_schema.BW_EGO_DIM), dtype=torch.float32, device=device),
                selected_sat_tokens=torch.empty(
                    (row_count, int(select_k), bw_schema.BW_SAT_TOKEN_DIM),
                    dtype=torch.float32,
                    device=device,
                ),
                selected_sat_mask=torch.empty((row_count, int(select_k)), dtype=torch.bool, device=device),
                gu_tokens=torch.empty((row_count, int(cfg.num_gu), bw_schema.BW_GU_TOKEN_DIM), dtype=torch.float32, device=device),
                gu_mask=torch.empty((row_count, int(cfg.num_gu)), dtype=torch.bool, device=device),
                bw_valid_mask=torch.empty((row_count, int(cfg.num_gu)), dtype=torch.bool, device=device),
            )
            runtime.main.live_bw_obs = bw_obs_view
            runtime.main.live_bw_action = torch.empty_like(bw_action_template)
            runtime.main.live_bw_action.zero_()
            runtime.main.live_bw_ref_action = torch.empty_like(bw_ref_action_template)
            runtime.main.live_bw_ref_action.zero_()
            runtime.main.live_bw_flow_proxy_override_action = torch.empty_like(bw_action_template)
            runtime.main.live_bw_flow_proxy_override_action.zero_()
            runtime.main.live_bw_old_logprob = torch.empty_like(bw_logprob_template)
            runtime.main.live_bw_old_logprob.zero_()
            runtime.main.live_bw_old_logprobs_per_agent = torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device)
            runtime.main.live_bw_old_logprobs_per_agent.zero_()
            bw_agent_shape = (batch_size, int(cfg.num_uav))
            runtime.main.live_bw_entropy_per_agent = torch.zeros(bw_agent_shape, dtype=torch.float32, device=device)
            runtime.main.live_bw_logprob_raw_per_agent = torch.zeros(bw_agent_shape, dtype=torch.float32, device=device)
            runtime.main.live_bw_entropy_raw_per_agent = torch.zeros(bw_agent_shape, dtype=torch.float32, device=device)
            runtime.main.live_bw_tau = torch.zeros(bw_agent_shape, dtype=torch.float32, device=device)
            runtime.main.live_bw_kappa = torch.zeros(bw_agent_shape, dtype=torch.float32, device=device)
            runtime.main.live_bw_valid_count = torch.zeros(bw_agent_shape, dtype=torch.long, device=device)
            runtime.main.live_bw_latent_count = torch.zeros(bw_agent_shape, dtype=torch.long, device=device)
            lyapunov_state_shape = (batch_size, int(cfg.num_uav), int(cfg.num_gu))
            runtime.main.lyapunov_pressure_ema = torch.zeros(
                lyapunov_state_shape,
                dtype=torch.float32,
                device=device,
            )
            runtime.main.lyapunov_virtual_queue = torch.zeros_like(runtime.main.lyapunov_pressure_ema)
            runtime.main.lyapunov_service_est = torch.zeros_like(runtime.main.lyapunov_pressure_ema)
            runtime.main.lyapunov_instant_pressure = torch.zeros_like(runtime.main.lyapunov_pressure_ema)
            actor_scratch_width = _native_actor_scratch_width_from_config(
                cfg,
                sat_obs_width=int(sat_obs_width),
                subset_count=int(subset_count),
                accel_sat_width=int(accel_sat_width),
            )
            runtime.main.native_actor_scratch = torch.empty(
                (row_count, int(actor_scratch_width)),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.native_cuda_marker = torch.zeros((1,), dtype=torch.int32, device=device)
            runtime.main.native_cuda_empty_float = torch.empty((0,), dtype=torch.float32, device=device)
            runtime.main.native_cuda_empty_long = torch.empty((0,), dtype=torch.long, device=device)
            runtime.main.native_cuda_empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
            runtime.main.native_cuda_empty_int = torch.empty((0,), dtype=torch.int32, device=device)
            bw_sat_loads_t = torch.empty((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=device)
            bw_sat_load_active_t = torch.empty((batch_size, int(active_width)), dtype=torch.float32, device=device)
            runtime.main.bw_stage_fields = _stage_fields_with_updates(
                runtime.main.sat_stage_fields,
                {
                    "stage_id": self._native_stage_id_tensor(
                        int(StructuredControlDriver.STAGE_BW),
                        batch_size,
                        device=device,
                    ),
                    "sat_selection_matrix": runtime.main.bw_sat_selection_matrix,
                    "sat_loads": bw_sat_loads_t,
                    "sat_load_active": bw_sat_load_active_t,
                },
            )
            runtime.main.bw_direct_input_fields = _NativeBwDirectInputTensorFields(
                assoc=runtime.main.bw_assoc,
                prev_association=runtime.main.bw_prev_association,
                candidate_indices=runtime.main.bw_candidate_indices,
                candidate_mask=runtime.main.bw_candidate_mask,
                bw_valid_mask=runtime.main.bw_valid_mask,
                access_gain_matrix=runtime.main.bw_access_gain_matrix,
                sat_selection_matrix=runtime.main.bw_sat_selection_matrix,
                active_sat_ids=runtime.main.bw_active_sat_ids,
                gain_active=runtime.main.bw_gain_active,
                nu_eff_active=runtime.main.bw_nu_eff_active,
                valid_flag_active=runtime.main.bw_valid_flag_active,
                sat_pos=runtime.main.bw_sat_pos,
                uav_ecef=runtime.main.bw_uav_ecef,
                uav_pos=runtime.main.bw_uav_pos,
                uav_vel=runtime.main.bw_uav_vel,
                gu_pos=runtime.main.bw_gu_pos,
            )
            runtime.random.arrivals = torch.empty((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=device)
            runtime.random.arrival_rates = torch.empty((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=device)
            runtime.random.fading_gain = torch.empty(
                (batch_size, int(cfg.num_gu), int(cfg.num_uav)),
                dtype=torch.float32,
                device=device,
            )
            runtime.random.step_tensor = torch.zeros((2,), dtype=torch.int32, device=device)
            runtime.random.reset_count = torch.zeros((batch_size,), dtype=torch.int32, device=device)
            runtime.main.bw_fading_gain_unity = self._native_fading_unity_tensor(
                tuple(runtime.random.fading_gain.shape),
                device=device,
            )
            runtime.random.doppler_noise = torch.empty(
                (batch_size, int(cfg.num_uav), int(cfg.num_sat)),
                dtype=torch.float32,
                device=device,
            )
            runtime.main.bw_doppler_noise_zero = self._native_zero_float_tensor(
                "doppler_noise_zero",
                tuple(runtime.random.doppler_noise.shape),
                device=device,
            )
            runtime.preallocate_step_result_buffers(
                num_envs=batch_size,
                num_uav=int(cfg.num_uav),
                bw_action_shape=runtime.main.bw_action_shape,
                device=device,
                reward_mode_active=runtime.main.bw_reward_mode_active,
                expose_auxiliary_outputs=bool(typed_domains.post_stats_safety.danger_imitation_enabled),
                expose_reward_parts=True,
                expose_flow_proxy=bool(flow_proxy_params.enabled) and int(flow_proxy_params.reward_mode_code) > 0,
            )
            result_buffers = runtime.result
            reward_parts = result_buffers.reward_parts
            if not isinstance(reward_parts, StructuredGpuRewardPartBuffers):
                raise RuntimeError("BW reward-part output buffers must be preallocated at rollout begin.")
            runtime.main.bw_reward_part_out = _NativeBwRewardPartOutBuffers(
                service_ratio=reward_parts.service_ratio,
                drop_ratio=reward_parts.drop_ratio,
                arrival_ref=reward_parts.arrival_ref,
                b_pre_steps=reward_parts.b_pre_steps,
                x_acc=reward_parts.x_acc,
                x_rel=reward_parts.x_rel,
                g_pre=reward_parts.g_pre,
                d_pre=reward_parts.d_pre,
                processed_ratio_eval=reward_parts.processed_ratio_eval,
                drop_ratio_eval=reward_parts.drop_ratio_eval,
                pre_backlog_steps_eval=reward_parts.pre_backlog_steps_eval,
                sat_overlap_eval=reward_parts.sat_overlap_eval,
                D_sys_report=reward_parts.D_sys_report,
                drop_sum=reward_parts.drop_sum,
                gu_queue_sum=reward_parts.gu_queue_sum,
                uav_queue_sum=reward_parts.uav_queue_sum,
                sat_queue_sum=reward_parts.sat_queue_sum,
                queue_total_sum=reward_parts.queue_total_sum,
                drop_sum_active=reward_parts.drop_sum_active,
                expire_sum=reward_parts.expire_sum,
                gu_drop_sum=reward_parts.gu_drop_sum,
                uav_drop_sum=reward_parts.uav_drop_sum,
                sat_drop_sum=reward_parts.sat_drop_sum,
                arrival_sum=reward_parts.arrival_sum,
                outflow_sum=reward_parts.outflow_sum,
                backhaul_sum=reward_parts.backhaul_sum,
                sat_processed_sum=reward_parts.sat_processed_sum,
                collision_event=reward_parts.collision_event,
                overflow_risk_mean=reward_parts.overflow_risk_mean,
                downstream_pressure_mean=reward_parts.downstream_pressure_mean,
                service_gap_mean=reward_parts.service_gap_mean,
                service_gap_risk_mean=reward_parts.service_gap_risk_mean,
                bw_weighted_workload_delta_reward=reward_parts.bw_weighted_workload_delta_reward,
                bw_weighted_workload_level_reward=reward_parts.bw_weighted_workload_level_reward,
                bw_gu_queue_level_reward=reward_parts.bw_gu_queue_level_reward,
                bw_system_queue_level_reward=reward_parts.bw_system_queue_level_reward,
                bw_gu_service_queue_reward=reward_parts.bw_gu_service_queue_reward,
                intervention_norm=reward_parts.intervention_norm,
                intervention_rate=reward_parts.intervention_rate,
                intervention_norm_top1=reward_parts.intervention_norm_top1,
                danger_imitation_active_rate=reward_parts.danger_imitation_active_rate,
                close_risk=reward_parts.close_risk,
                term_close_risk=reward_parts.term_close_risk,
                reward_raw=reward_parts.reward_raw,
            )
            runtime.main.bw_state_out = _NativeBwRuntimeStateOutBuffers(
                uav_pos=self._runtime_tensor_state.uav_pos,
                uav_vel=self._runtime_tensor_state.uav_vel,
                gu_pos=self._runtime_tensor_state.gu_pos,
                prev_queue_sum_gu=self._runtime_tensor_state.prev_queue_sum_gu,
                prev_queue_sum_uav=self._runtime_tensor_state.prev_queue_sum_uav,
                prev_queue_sum_sat=self._runtime_tensor_state.prev_queue_sum_sat,
                prev_gu_queue_vec=self._runtime_tensor_state.prev_gu_queue_vec,
                prev_uav_queue_vec=self._runtime_tensor_state.prev_uav_queue_vec,
                prev_sat_queue_vec=self._runtime_tensor_state.prev_sat_queue_vec,
                gu_queue=self._runtime_tensor_state.gu_queue,
                uav_queue=self._runtime_tensor_state.uav_queue,
                sat_queue=self._runtime_tensor_state.sat_queue,
                uav_energy=self._runtime_tensor_state.uav_energy,
                last_association=self._runtime_tensor_state.last_association,
                last_sat_selection_matrix=self._runtime_tensor_state.last_sat_selection_matrix,
                last_sat_connection_counts=self._runtime_tensor_state.last_sat_connection_counts,
                last_gu_service_gap=self._runtime_tensor_state.last_gu_service_gap,
                last_gu_deadline_age=self._runtime_tensor_state.last_gu_deadline_age,
                last_gu_arrival=self._runtime_tensor_state.last_gu_arrival,
                last_gu_arrival_rate_vec=self._runtime_tensor_state.last_gu_arrival_rate_vec,
                last_gu_outflow=self._runtime_tensor_state.last_gu_outflow,
                last_gu_deadline_slack=self._runtime_tensor_state.last_gu_deadline_slack,
                last_gu_deadline_risk=self._runtime_tensor_state.last_gu_deadline_risk,
                last_gu_urgency_risk=self._runtime_tensor_state.last_gu_urgency_risk,
                last_gu_downstream_pressure=self._runtime_tensor_state.last_gu_downstream_pressure,
                last_gu_service_gap_risk=self._runtime_tensor_state.last_gu_service_gap_risk,
                last_exec_accel=self._runtime_tensor_state.last_exec_accel,
                last_policy_accel=self._runtime_tensor_state.last_policy_accel,
                gu_workload_ema=self._runtime_tensor_state.gu_workload_ema,
                uav_workload_ema=self._runtime_tensor_state.uav_workload_ema,
                sat_workload_ema=self._runtime_tensor_state.sat_workload_ema,
                arrival_ref_bits_per_step=self._runtime_tensor_state.arrival_ref_bits_per_step,
                effective_task_arrival_rate=self._runtime_tensor_state.effective_task_arrival_rate,
                gu_deadline_steps=self._runtime_tensor_state.gu_deadline_steps,
                arrival_base_scale=self._runtime_tensor_state.arrival_base_scale,
                hotspot_active_idx=self._runtime_tensor_state.hotspot_active_idx,
                hotspot_subset_count=self._runtime_tensor_state.hotspot_subset_count,
                hotspot_member_mask=self._runtime_tensor_state.hotspot_member_mask,
                traffic_reset_step=self._runtime_tensor_state.traffic_reset_step,
                traffic_reset_ordinal=self._runtime_tensor_state.traffic_reset_ordinal,
                episode_idx=self._runtime_tensor_state.episode_idx,
                t=self._runtime_tensor_state.t,
                prev_q_norm_active=self._runtime_tensor_state.prev_q_norm_active,
                doppler_residual=self._runtime_tensor_state.doppler_residual,
                sat_pos=self._runtime_tensor_state.sat_pos,
                sat_vel=self._runtime_tensor_state.sat_vel,
            )
            accel_world_template = _allocate_native_training_world_tensor_fields(
                local_obs_params=local_obs_params,
                fields_obj=runtime.main.accel_stage_fields,
            )
            sat_world_template = _allocate_native_training_world_tensor_fields(
                local_obs_params=local_obs_params,
                fields_obj=runtime.main.sat_stage_fields,
            )
            bw_world_template = _allocate_native_training_world_tensor_fields(
                local_obs_params=local_obs_params,
                fields_obj=runtime.main.bw_stage_fields,
            )
            next_world_template = _allocate_native_training_world_tensor_fields(
                local_obs_params=local_obs_params,
                fields_obj=runtime.main.accel_stage_field_buffers[1],
            )
            history_snapshots_enabled = bool(
                self._native_main_kernel_typed_domains().history_output.runtime_snapshots_enabled
            )
            if not history_snapshots_enabled:
                history = runtime.rollout_history
                history.accel_runtime_state = None
                history.accel_runtime_stage = None
                history.sat_runtime_state = None
                history.sat_runtime_stage = None
                history.bw_runtime_state = None
                history.bw_runtime_cache = None
                history.bw_runtime_stage = None
            rollout_scalar_template = torch.empty((batch_size,), dtype=torch.float32, device=device)
            bw_agent_logprob_template = torch.empty((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=device)
            runtime.preallocate_native_main_kernel_training_ring_buffers(
                accel_world_batch=accel_world_template,
                sat_world_batch=sat_world_template,
                bw_world_batch=bw_world_template,
                next_world_batch=next_world_template,
                accel_runtime_state=self._runtime_tensor_state if history_snapshots_enabled else None,
                accel_runtime_stage=runtime.main.accel_stage_field_buffers[0] if history_snapshots_enabled else None,
                sat_runtime_state=self._runtime_tensor_state if history_snapshots_enabled else None,
                sat_runtime_stage=runtime.main.sat_stage_fields if history_snapshots_enabled else None,
                bw_runtime_state=self._runtime_tensor_state if history_snapshots_enabled else None,
                bw_runtime_cache=StructuredGpuBwRuntimeCacheBuffers(
                    candidate_indices=runtime.main.bw_candidate_indices,
                    valid_mask=runtime.main.bw_valid_mask,
                    assoc=runtime.main.bw_assoc,
                    prev_association=runtime.main.bw_prev_association,
                    candidate_mask=runtime.main.bw_candidate_mask,
                    access_gain_matrix=runtime.main.bw_access_gain_matrix,
                    sat_selection_matrix=runtime.main.bw_sat_selection_matrix,
                    active_sat_ids=runtime.main.bw_active_sat_ids,
                    gain_active=runtime.main.bw_gain_active,
                    nu_eff_active=runtime.main.bw_nu_eff_active,
                    valid_flag_active=runtime.main.bw_valid_flag_active,
                    sat_pos=runtime.main.bw_sat_pos,
                    uav_ecef=runtime.main.bw_uav_ecef,
                    uav_pos=runtime.main.bw_uav_pos,
                    uav_vel=runtime.main.bw_uav_vel,
                    gu_pos=runtime.main.bw_gu_pos,
                ) if history_snapshots_enabled else None,
                bw_runtime_stage=runtime.main.bw_stage_fields if history_snapshots_enabled else None,
                accel_local_batch=accel_obs_view,
                sat_local_batch=sat_obs_view,
                bw_local_batch=bw_obs_view,
                accel_actions=accel_action_template,
                accel_latent_actions=accel_action_template,
                sat_actions=sat_subset_index_template,
                sat_action_indices=sat_action_indices_template,
                bw_actions=bw_action_template,
                accel_old_logprobs=accel_logprob_template,
                sat_old_logprobs=sat_logprob_template,
                sat_old_logprobs_per_agent=sat_live_logprob_template,
                bw_old_logprobs=bw_logprob_template,
                accel_values=rollout_scalar_template,
                sat_values=rollout_scalar_template,
                bw_values=rollout_scalar_template,
                rewards=result_buffers.team_rewards,
                terminated=result_buffers.terminated,
                truncated=result_buffers.truncated,
                accel_danger_imitation_targets=result_buffers.danger_imitation_target,
                accel_danger_imitation_masks=result_buffers.danger_imitation_mask,
                bw_access_rewards=result_buffers.bw_access_rewards,
                bw_weighted_workload_delta_rewards=result_buffers.bw_weighted_workload_delta_rewards,
                bw_weighted_workload_level_rewards=result_buffers.bw_weighted_workload_level_rewards,
                bw_gu_queue_level_rewards=result_buffers.bw_gu_queue_level_rewards,
                bw_system_queue_level_rewards=result_buffers.bw_system_queue_level_rewards,
                bw_gu_service_queue_rewards=result_buffers.bw_gu_service_queue_rewards,
                bw_flow_proxy_scores=result_buffers.bw_flow_proxy_scores,
                bw_flow_proxy_masks=result_buffers.bw_flow_proxy_mask,
                bw_flow_proxy_deltas=result_buffers.bw_flow_proxy_deltas,
                bw_ref_actions=bw_ref_action_template,
                bw_old_logprobs_per_agent=bw_agent_logprob_template,
                bw_entropy_per_agent=bw_agent_logprob_template,
                bw_logprob_raw_per_agent=bw_agent_logprob_template,
                bw_entropy_raw_per_agent=bw_agent_logprob_template,
                bw_tau=bw_agent_logprob_template,
                bw_kappa=bw_agent_logprob_template,
                bw_valid_count=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.long, device=device),
                bw_latent_count=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.long, device=device),
            )
            self._bind_native_main_kernel_history_outputs(runtime, runtime.history)
            runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(runtime)
            runtime.capture_native_main_kernel_base_views()

    def _native_main_kernel_runtime(self) -> StructuredGpuRolloutRuntime:
        runtime = self.native_rollout_runtime
        if runtime is None:
            raise RuntimeError("native main-kernel execution requires a native tensor rollout runtime.")
        return runtime

    def _build_native_cuda_runtime_abi(self, runtime: StructuredGpuRolloutRuntime) -> native_cuda.NativeCudaRuntimeABI:
        main = runtime.main
        domains = self._native_main_kernel_typed_domains()
        cfg = self._cfg
        tensor_state = self._runtime_tensor_state
        marker = main.native_cuda_marker
        if not torch.is_tensor(marker) or marker.dtype != torch.int32 or marker.device.type != "cuda":
            raise RuntimeError("native CUDA typed ABI requires a persistent CUDA int32 marker/state tensor.")
        empty_float = main.native_cuda_empty_float
        empty_long = main.native_cuda_empty_long
        empty_bool = main.native_cuda_empty_bool
        empty_int = main.native_cuda_empty_int
        if (
            not torch.is_tensor(empty_float)
            or not torch.is_tensor(empty_long)
            or not torch.is_tensor(empty_bool)
            or not torch.is_tensor(empty_int)
        ):
            raise RuntimeError("native CUDA typed ABI requires persistent empty fallback tensors.")
        float_tensors: list[torch.Tensor] = []
        long_tensors: list[torch.Tensor] = []
        bool_tensors: list[torch.Tensor] = []
        int_tensors: list[torch.Tensor] = [marker]

        def _float(value: torch.Tensor | None, name: str) -> None:
            if value is None:
                float_tensors.append(empty_float)
                return
            if not torch.is_tensor(value) or value.dtype != torch.float32 or value.device.type != "cuda":
                raise RuntimeError(f"native CUDA ABI float tensor {name!r} is missing or has the wrong dtype/device.")
            if not value.is_contiguous():
                raise RuntimeError(f"native CUDA ABI float tensor {name!r} must be contiguous.")
            float_tensors.append(value)

        def _long(value: torch.Tensor | None, name: str) -> None:
            if value is None:
                long_tensors.append(empty_long)
                return
            if not torch.is_tensor(value) or value.dtype != torch.long or value.device.type != "cuda":
                raise RuntimeError(f"native CUDA ABI long tensor {name!r} is missing or has the wrong dtype/device.")
            if not value.is_contiguous():
                raise RuntimeError(f"native CUDA ABI long tensor {name!r} must be contiguous.")
            long_tensors.append(value)

        def _bool(value: torch.Tensor | None, name: str) -> None:
            if value is None:
                bool_tensors.append(empty_bool)
                return
            if not torch.is_tensor(value) or value.dtype != torch.bool or value.device.type != "cuda":
                raise RuntimeError(f"native CUDA ABI bool tensor {name!r} is missing or has the wrong dtype/device.")
            if not value.is_contiguous():
                raise RuntimeError(f"native CUDA ABI bool tensor {name!r} must be contiguous.")
            bool_tensors.append(value)

        def _int(value: torch.Tensor | None, name: str) -> None:
            if value is None:
                int_tensors.append(empty_int)
                return
            if not torch.is_tensor(value) or value.dtype != torch.int32 or value.device.type != "cuda":
                raise RuntimeError(f"native CUDA ABI int tensor {name!r} is missing or has the wrong dtype/device.")
            if not value.is_contiguous():
                raise RuntimeError(f"native CUDA ABI int tensor {name!r} must be contiguous.")
            int_tensors.append(value)

        stage_float_fields = (
            "effective_b_backhaul_per_sat",
            "uav_pos",
            "uav_vel",
            "uav_energy",
            "uav_queue",
            "gu_pos",
            "gu_queue",
            "sat_queue",
            "sat_loads",
            "sat_pos",
            "sat_vel",
            "bw_valid_mask",
            "candidate_flag",
            "bw_valid_flag",
            "prev_assoc_flag",
            "eta_ref_feature",
            "eta_slots",
            "gu_proxy_features",
            "uav_assoc_uav_cost",
            "sat_cost_norm",
            "access_gain_matrix",
            "visible_flag_all",
            "elevation_matrix",
            "uav_ecef_all",
            "uav_vel_ecef_all",
            "sat_pos_active",
            "sat_vel_active",
            "sat_queue_active",
            "sat_load_active",
            "sat_cost_norm_active",
            "us_rel_pos_active",
            "us_rel_vel_active",
            "us_gain_active",
            "us_nu_eff_active",
            "visible_flag_active",
            "us_valid_flag_active",
            "us_rel_pos_all",
            "us_rel_vel_all",
            "us_gain_all",
            "us_nu_eff_all",
            "us_valid_flag_all",
            "us_sat_queue_all",
        )
        stage_long_fields = (
            "stage_id",
            "assoc",
            "prev_association",
            "candidate_indices",
            "sat_selection_matrix",
            "visible_ids",
            "active_sat_ids",
        )
        stage_bool_fields = ("candidate_mask", "visible_mask")

        def _stage_fields(fields_obj: _NativeStageTensorFields | None, prefix: str) -> None:
            if not _is_native_stage_fields(fields_obj):
                raise RuntimeError(f"native CUDA ABI requires {prefix} stage fields.")
            for field_name in stage_float_fields:
                _float(getattr(fields_obj, field_name), f"{prefix}.{field_name}")
            for field_name in stage_long_fields:
                _long(getattr(fields_obj, field_name), f"{prefix}.{field_name}")
            for field_name in stage_bool_fields:
                _bool(getattr(fields_obj, field_name), f"{prefix}.{field_name}")

        def _training_world(world_obj: _NativeTrainingWorldTensorFields | None, prefix: str) -> None:
            if not isinstance(world_obj, _NativeTrainingWorldTensorFields):
                raise RuntimeError(f"native CUDA ABI requires {prefix} world history tensors.")
            for field_name in (
                "uav_nodes",
                "gu_nodes",
                "sat_nodes",
                "uav_gu_edges",
                "uav_sat_edges",
                "uav_uav_edges",
            ):
                _float(getattr(world_obj, field_name), f"{prefix}.{field_name}")
            for field_name in ("gu_mask", "sat_mask", "uav_gu_mask", "uav_sat_mask", "uav_uav_mask"):
                _bool(getattr(world_obj, field_name), f"{prefix}.{field_name}")

        def _obs_accel(obs_obj: StructuredGpuAccelObsView | _NativeAccelTrainingHistoryOutBuffers | None, prefix: str) -> None:
            if obs_obj is None:
                raise RuntimeError(f"native CUDA ABI requires {prefix} accel obs tensors.")
            for field_name in ("ego_features", "ego_cell", "gu_tokens", "peer_tokens", "sat_tokens"):
                _float(getattr(obs_obj, field_name), f"{prefix}.{field_name}")
            for field_name in ("gu_mask", "peer_mask", "sat_mask"):
                _bool(getattr(obs_obj, field_name), f"{prefix}.{field_name}")

        def _obs_sat(obs_obj: StructuredGpuSatObsView | _NativeSatTrainingHistoryOutBuffers | None, prefix: str) -> None:
            if obs_obj is None:
                raise RuntimeError(f"native CUDA ABI requires {prefix} SAT obs tensors.")
            for field_name in ("ego_features", "demand_features", "role_features", "sat_tokens"):
                _float(getattr(obs_obj, field_name), f"{prefix}.{field_name}")
            for field_name in ("sat_mask", "sat_valid_mask"):
                _bool(getattr(obs_obj, field_name), f"{prefix}.{field_name}")
            if prefix == "live.sat_obs":
                _long(None, f"{prefix}.subset_members_unused")
            else:
                _long(getattr(obs_obj, "subset_members"), f"{prefix}.subset_members")

        def _obs_bw(obs_obj: StructuredGpuBwObsView | _NativeBwTrainingHistoryOutBuffers | None, prefix: str) -> None:
            if obs_obj is None:
                raise RuntimeError(f"native CUDA ABI requires {prefix} BW obs tensors.")
            _float(getattr(obs_obj, "ego_features"), f"{prefix}.ego_features")
            _float(getattr(obs_obj, "selected_sat_tokens"), f"{prefix}.selected_sat_tokens")
            _float(getattr(obs_obj, "gu_tokens"), f"{prefix}.gu_tokens")
            for field_name in ("selected_sat_mask", "gu_mask", "bw_valid_mask"):
                _bool(getattr(obs_obj, field_name), f"{prefix}.{field_name}")

        def _reward_parts(parts_obj: StructuredGpuRewardPartBuffers | None, prefix: str) -> None:
            if parts_obj is None:
                for key in _BW_REWARD_PART_KEYS:
                    _float(None, f"{prefix}.{key}")
                return
            for key in _BW_REWARD_PART_KEYS:
                _float(getattr(parts_obj, key), f"{prefix}.{key}")

        def _assert_history_snapshot_layout(
            history_obj: Any,
            source_obj: Any,
            field_names: tuple[str, ...],
            *,
            history_prefix: str,
            source_prefix: str,
        ) -> None:
            if history_obj is None or source_obj is None:
                return
            for field_name in field_names:
                hist = getattr(history_obj, field_name, None)
                src = getattr(source_obj, field_name, None)
                if hist is None or src is None:
                    continue
                if not torch.is_tensor(hist) or not torch.is_tensor(src):
                    continue
                if hist.ndim < 1 or src.ndim < 1:
                    raise RuntimeError(
                        f"native runtime snapshot field {history_prefix}.{field_name} and "
                        f"{source_prefix}.{field_name} must be row-major tensors."
                    )
                hist_tail = tuple(hist.shape[1:])
                src_tail = tuple(src.shape[1:])
                if hist_tail != src_tail:
                    raise RuntimeError(
                        f"native runtime snapshot row-width mismatch for {field_name}: "
                        f"{history_prefix}.{field_name} trailing shape {hist_tail} != "
                        f"{source_prefix}.{field_name} trailing shape {src_tail}."
                    )

        # Runtime state tensors.
        for field_name in (
            "uav_pos",
            "uav_vel",
            "uav_energy",
            "uav_queue",
            "gu_pos",
            "gu_queue",
            "sat_queue",
            "sat_pos",
            "sat_vel",
            "arrival_ref_bits_per_step",
            "effective_task_arrival_rate",
            "arrival_base_scale",
            "gu_workload_ema",
            "uav_workload_ema",
            "sat_workload_ema",
            "last_gu_arrival_rate_vec",
            "gu_deadline_steps",
            "last_gu_arrival",
            "last_gu_outflow",
            "last_gu_urgency_risk",
            "last_gu_downstream_pressure",
            "last_gu_service_gap_risk",
            "last_gu_deadline_slack",
            "last_gu_deadline_risk",
            "last_gu_service_gap",
            "last_gu_deadline_age",
            "last_exec_accel",
            "last_policy_accel",
            "avoidance_eta_eff",
            "last_avoidance_eta_exec",
            "doppler_residual",
            "prev_queue_sum_gu",
            "prev_queue_sum_uav",
            "prev_queue_sum_sat",
            "prev_q_norm_active",
            "prev_gu_queue_vec",
            "prev_uav_queue_vec",
            "prev_sat_queue_vec",
        ):
            _float(getattr(tensor_state, field_name), f"state.{field_name}")
        _int(getattr(tensor_state, "prev_association"), "state.prev_association")
        _int(getattr(tensor_state, "last_association"), "state.last_association")
        _long(getattr(tensor_state, "last_sat_selection_matrix"), "state.last_sat_selection_matrix")
        _int(None, "state.last_sat_connection_counts_int_unused")
        for field_name in (
            "hotspot_active_idx",
            "hotspot_subset_count",
            "traffic_reset_step",
            "traffic_reset_ordinal",
            "episode_idx",
            "t",
            "global_step",
        ):
            _int(getattr(tensor_state, field_name), f"state.{field_name}")
        _bool(None, "state.hotspot_member_mask_bool_unused")

        # Random rollout/reset tapes and per-step random buffers.
        random = runtime.random
        for field_name in (
            "arrivals",
            "arrival_rates",
            "fading_gain",
            "doppler_noise",
            "arrival_rollout_tape",
            "arrival_rate_rollout_tape",
            "fading_gain_rollout_tape",
            "doppler_noise_rollout_tape",
            "reset_gu_pos_rollout_tape",
            "reset_uav_pos_rollout_tape",
            "reset_uav_vel_rollout_tape",
            "reset_gu_queue_rollout_tape",
            "reset_uav_queue_rollout_tape",
            "reset_sat_queue_rollout_tape",
            "reset_arrival_base_scale_rollout_tape",
            "reset_deadline_steps_rollout_tape",
            "reset_doppler_residual_rollout_tape",
            "reset_effective_arrival_rate_rollout_tape",
            "reset_arrival_rate_vec_rollout_tape",
            "reset_arrival_ref_rollout_tape",
            "reset_followup_arrival_rollout_tape",
            "reset_followup_arrival_rate_rollout_tape",
        ):
            _float(getattr(random, field_name, None), f"random.{field_name}")
        for field_name in (
            "hotspot_active_after_rollout_tape",
            "reset_followup_hotspot_active_after_rollout_tape",
            "reset_episode_idx_rollout_tape",
            "reset_hotspot_active_idx_rollout_tape",
            "reset_hotspot_subset_count_rollout_tape",
        ):
            _int(getattr(random, field_name, None), f"random.{field_name}")
        _bool(None, "random.reset_hotspot_member_mask_rollout_tape_bool_unused")
        _bool(None, "random.hotspot_mask_rollout_tape_bool_unused")
        _int(getattr(random, "step_tensor", None), "random.step_tensor")
        _int(getattr(random, "reset_count", None), "random.reset_count")

        if not isinstance(main.accel_stage_field_buffers, tuple) or len(main.accel_stage_field_buffers) != 2:
            raise RuntimeError("native CUDA ABI requires two accel stage field buffers.")
        _stage_fields(main.accel_stage_field_buffers[0], "stage.accel0")
        _stage_fields(main.accel_stage_field_buffers[1], "stage.accel1")
        _stage_fields(main.sat_stage_fields, "stage.sat")
        _stage_fields(main.bw_stage_fields, "stage.bw")

        if not isinstance(main.accel_live_obs_buffers, tuple) or len(main.accel_live_obs_buffers) != 2:
            raise RuntimeError("native CUDA ABI requires two accel live obs buffers.")
        _obs_accel(main.accel_live_obs_buffers[0], "live.accel_obs0")
        _obs_accel(main.accel_live_obs_buffers[1], "live.accel_obs1")
        _obs_sat(main.live_sat_obs, "live.sat_obs")
        _obs_bw(main.live_bw_obs, "live.bw_obs")

        for name in (
            "live_accel_action",
            "live_bw_action",
            "live_bw_ref_action",
            "live_bw_flow_proxy_override_action",
            "live_accel_old_logprob",
            "live_sat_old_logprobs_per_agent",
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
        ):
            _float(getattr(main, name), f"main.{name}")
        _long(main.live_sat_subset_index, "main.live_sat_subset_index")

        bw_input = main.bw_direct_input_fields
        if bw_input is None:
            raise RuntimeError("native CUDA ABI requires BW direct input fields.")
        for name in ("bw_valid_mask", "access_gain_matrix", "gain_active", "nu_eff_active", "valid_flag_active", "sat_pos", "uav_ecef", "uav_pos", "uav_vel", "gu_pos"):
            _float(getattr(bw_input, name), f"bw_input.{name}")
        for name in ("assoc", "prev_association", "candidate_indices", "sat_selection_matrix", "active_sat_ids"):
            _long(getattr(bw_input, name), f"bw_input.{name}")
        _bool(getattr(bw_input, "candidate_mask"), "bw_input.candidate_mask")

        _float(main.sat_subset_sizes, "main.sat_subset_sizes")
        for name in (
            "sat_subset_members_base",
            "candidate_slot_ids",
            "candidate_env_ids",
            "candidate_gu_ids",
            "candidate_uav_ids",
            "sat_all_ids",
            "accel_uav_index_order",
            "accel_neighbor_indices",
        ):
            _long(getattr(main, name), f"main.{name}")
        _bool(main.accel_offdiag_mask, "main.accel_offdiag_mask")
        _bool(main.uav_pair_upper_mask, "main.uav_pair_upper_mask")

        history = runtime.history
        _training_world(history.accel_stage.world_batch, "history.accel_world")
        _training_world(history.sat_stage.world_batch, "history.sat_world")
        _training_world(history.bw_stage.world_batch, "history.bw_world")
        _training_world(history.terminal_next_world, "history.terminal_next_world")
        _obs_accel(history.accel_stage, "history.accel_local")
        _obs_sat(history.sat_stage, "history.sat_local")
        _obs_bw(history.bw_stage, "history.bw_local")
        for name in (
            "actions",
            "old_logprobs",
            "values",
        ):
            _float(getattr(history.accel_stage, name) if name != "actions" else history.accel_stage.actions, f"history.accel_stage.{name}")
        _long(history.sat_stage.actions, "history.sat_stage.actions")
        _float(history.sat_stage.old_logprobs, "history.sat_stage.old_logprobs")
        _float(history.sat_stage.values, "history.sat_stage.values")
        for name in (
            "actions",
            "old_logprobs",
            "values",
            "rewards",
            "bw_access_rewards",
            "bw_weighted_workload_delta_rewards",
            "bw_weighted_workload_level_rewards",
            "bw_gu_queue_level_rewards",
            "bw_system_queue_level_rewards",
            "bw_gu_service_queue_rewards",
            "bw_flow_proxy_scores",
            "bw_flow_proxy_masks",
            "bw_flow_proxy_deltas",
            "bw_ref_actions",
            "bw_old_logprobs_per_agent",
        ):
            _float(getattr(history.bw_stage, name, None), f"history.bw_stage.{name}")
        _bool(history.terminated, "history.terminated")
        _bool(history.truncated, "history.truncated")
        _bool(history.terminal_next_world_mask, "history.terminal_next_world_mask")
        _float(history.accel_stage.danger_imitation_targets, "history.accel_stage.danger_imitation_targets")
        _float(history.accel_stage.danger_imitation_masks, "history.accel_stage.danger_imitation_masks")
        _reward_parts(history.bw_stage.reward_parts, "history.bw_stage.reward_parts")

        link = main.bw_link_transition
        if link is not None:
            for name in ("uav_energy", "last_energy_cost", "rate_matrix", "sat_loads", "last_sat_score"):
                _float(getattr(link, name), f"main.bw_link_transition.{name}")
        else:
            for name in ("uav_energy", "last_energy_cost", "rate_matrix", "sat_loads", "last_sat_score"):
                _float(None, f"main.bw_link_transition.{name}")
        link_override = main.bw_link_transition_override
        if link_override is not None:
            for name in ("uav_energy", "last_energy_cost", "rate_matrix", "sat_loads", "last_sat_score"):
                _float(getattr(link_override, name), f"main.bw_link_transition_override.{name}")
        else:
            for name in ("uav_energy", "last_energy_cost", "rate_matrix", "sat_loads", "last_sat_score"):
                _float(None, f"main.bw_link_transition_override.{name}")
        _float(main.bw_link_transition_override_active, "main.bw_link_transition_override_active")
        _float(main.bw_sat_compute_rates, "main.bw_sat_compute_rates")
        _float(main.bw_fading_gain_unity, "main.bw_fading_gain_unity")
        _float(main.bw_doppler_noise_zero, "main.bw_doppler_noise_zero")
        _float(tensor_state.last_sat_connection_counts, "state.last_sat_connection_counts")
        _float(tensor_state.gu_cluster_centers, "state.gu_cluster_centers")
        _float(tensor_state.gu_cluster_counts, "state.gu_cluster_counts")
        _float(getattr(random, "reset_gu_cluster_centers_rollout_tape", None), "random.reset_gu_cluster_centers_rollout_tape")
        _float(getattr(random, "reset_gu_cluster_counts_rollout_tape", None), "random.reset_gu_cluster_counts_rollout_tape")
        _float(self._orbit_pos_table_tensor, "orbit.pos_table")
        _float(self._orbit_vel_table_tensor, "orbit.vel_table")
        _float(getattr(main, "native_actor_scratch", None), "main.native_actor_scratch")
        for field_name in (
            "gu_drop",
            "uav_drop",
            "sat_drop",
            "last_access_interference_by_uav",
            "last_bw_fraction_by_uav_gu",
            "last_gu_to_uav_inflow_by_uav",
            "last_uav_to_sat_outflow_matrix",
            "last_selected_mask_by_uav_sat",
            "last_sat_processed",
        ):
            _float(getattr(tensor_state, field_name), f"state.{field_name}")
        _float(history.accel_stage.world_batch.global_scalars, "history.accel_world.global_scalars")
        _float(history.sat_stage.world_batch.global_scalars, "history.sat_world.global_scalars")
        _float(history.bw_stage.world_batch.global_scalars, "history.bw_world.global_scalars")
        _float(history.terminal_next_world.global_scalars, "history.terminal_next_world.global_scalars")
        for name in (
            "live_bw_entropy_per_agent",
            "live_bw_logprob_raw_per_agent",
            "live_bw_entropy_raw_per_agent",
            "live_bw_tau",
            "live_bw_kappa",
        ):
            _float(getattr(main, name, None), f"main.{name}")
        for name in (
            "bw_entropy_per_agent",
            "bw_logprob_raw_per_agent",
            "bw_entropy_raw_per_agent",
            "bw_tau",
            "bw_kappa",
        ):
            _float(getattr(history.bw_stage, name, None), f"history.bw_stage.{name}")
        _float(getattr(main, "live_sat_entropy_per_agent", None), "main.live_sat_entropy_per_agent")
        _float(getattr(main, "live_accel_latent_action", None), "main.live_accel_latent_action")
        _float(getattr(history.accel_stage, "latent_actions", None), "history.accel_stage.latent_actions")
        _float(getattr(history.sat_stage, "old_logprobs_per_agent", None), "history.sat_stage.old_logprobs_per_agent")
        _float(getattr(tensor_state, "hotspot_member_mask", None), "state.hotspot_member_mask")
        _float(getattr(random, "reset_hotspot_member_mask_rollout_tape", None), "random.reset_hotspot_member_mask_rollout_tape")
        _long(getattr(main, "selected_env_mapping", None), "main.selected_env_mapping")
        _long(history.accel_stage.world_batch.sat_ids, "history.accel_world.sat_ids")
        _long(history.sat_stage.world_batch.sat_ids, "history.sat_world.sat_ids")
        _long(history.bw_stage.world_batch.sat_ids, "history.bw_world.sat_ids")
        _long(history.terminal_next_world.sat_ids, "history.terminal_next_world.sat_ids")
        _long(getattr(main, "live_bw_valid_count", None), "main.live_bw_valid_count")
        _long(getattr(main, "live_bw_latent_count", None), "main.live_bw_latent_count")
        _long(getattr(history.bw_stage, "bw_valid_count", None), "history.bw_stage.bw_valid_count")
        _long(getattr(history.bw_stage, "bw_latent_count", None), "history.bw_stage.bw_latent_count")
        _long(getattr(main.live_sat_obs, "candidate_sat_ids", None), "live.sat_obs.candidate_sat_ids")
        _long(getattr(history.sat_stage, "candidate_sat_ids", None), "history.sat_local.candidate_sat_ids")
        _long(getattr(main, "live_sat_action_indices", None), "main.live_sat_action_indices")
        _long(getattr(history.sat_stage, "action_indices", None), "history.sat_stage.action_indices")
        _bool(None, "live.sat_obs.subset_mask_unused")
        _bool(getattr(history.sat_stage, "subset_mask", None), "history.sat_local.subset_mask")

        history_bw_state = getattr(history, "bw_runtime_state", None)
        bw_runtime_state_float_fields = (
            "uav_pos",
            "uav_vel",
            "uav_energy",
            "uav_queue",
            "gu_pos",
            "gu_cluster_centers",
            "gu_cluster_counts",
            "gu_queue",
            "sat_queue",
            "sat_pos",
            "sat_vel",
            "last_sat_connection_counts",
            "arrival_ref_bits_per_step",
            "effective_task_arrival_rate",
            "arrival_base_scale",
            "gu_workload_ema",
            "uav_workload_ema",
            "sat_workload_ema",
            "last_gu_arrival_rate_vec",
            "gu_deadline_steps",
            "last_gu_arrival",
            "last_gu_outflow",
            "gu_drop",
            "uav_drop",
            "sat_drop",
            "last_access_interference_by_uav",
            "last_bw_fraction_by_uav_gu",
            "last_gu_to_uav_inflow_by_uav",
            "last_uav_to_sat_outflow_matrix",
            "last_selected_mask_by_uav_sat",
            "last_sat_processed",
            "last_gu_urgency_risk",
            "last_gu_downstream_pressure",
            "last_gu_service_gap_risk",
            "last_gu_deadline_slack",
            "last_gu_deadline_risk",
            "last_gu_service_gap",
            "last_gu_deadline_age",
            "last_exec_accel",
            "last_policy_accel",
            "avoidance_eta_eff",
            "last_avoidance_eta_exec",
            "doppler_residual",
            "prev_queue_sum_gu",
            "prev_queue_sum_uav",
            "prev_queue_sum_sat",
            "prev_q_norm_active",
            "prev_gu_queue_vec",
            "prev_uav_queue_vec",
            "prev_sat_queue_vec",
            "hotspot_member_mask",
        )
        _assert_history_snapshot_layout(
            history_bw_state,
            tensor_state,
            bw_runtime_state_float_fields,
            history_prefix="history.bw_runtime_state",
            source_prefix="state",
        )
        for field_name in bw_runtime_state_float_fields:
            _float(getattr(history_bw_state, field_name, None), f"history.bw_runtime_state.{field_name}")
        _assert_history_snapshot_layout(
            history_bw_state,
            tensor_state,
            ("last_sat_selection_matrix",),
            history_prefix="history.bw_runtime_state",
            source_prefix="state",
        )
        _long(getattr(history_bw_state, "last_sat_selection_matrix", None), "history.bw_runtime_state.last_sat_selection_matrix")
        bw_runtime_state_int_fields = (
            "prev_association",
            "last_association",
            "hotspot_active_idx",
            "hotspot_subset_count",
            "traffic_reset_step",
            "traffic_reset_ordinal",
            "episode_idx",
            "t",
            "global_step",
        )
        _assert_history_snapshot_layout(
            history_bw_state,
            tensor_state,
            bw_runtime_state_int_fields,
            history_prefix="history.bw_runtime_state",
            source_prefix="state",
        )
        for field_name in bw_runtime_state_int_fields:
            _int(getattr(history_bw_state, field_name, None), f"history.bw_runtime_state.{field_name}")
        _bool(None, "history.bw_runtime_state.hotspot_member_mask_bool_unused")

        history_bw_cache = getattr(history, "bw_runtime_cache", None)
        bw_runtime_cache_float_fields = (
            "valid_mask",
            "access_gain_matrix",
            "gain_active",
            "nu_eff_active",
            "valid_flag_active",
            "sat_pos",
            "uav_ecef",
            "uav_pos",
            "uav_vel",
            "gu_pos",
        )
        _assert_history_snapshot_layout(
            history_bw_cache,
            bw_input,
            bw_runtime_cache_float_fields,
            history_prefix="history.bw_runtime_cache",
            source_prefix="bw_input",
        )
        for field_name in bw_runtime_cache_float_fields:
            _float(getattr(history_bw_cache, field_name, None), f"history.bw_runtime_cache.{field_name}")
        bw_runtime_cache_long_fields = (
            "candidate_indices",
            "assoc",
            "prev_association",
            "sat_selection_matrix",
            "active_sat_ids",
        )
        _assert_history_snapshot_layout(
            history_bw_cache,
            bw_input,
            bw_runtime_cache_long_fields,
            history_prefix="history.bw_runtime_cache",
            source_prefix="bw_input",
        )
        for field_name in bw_runtime_cache_long_fields:
            _long(getattr(history_bw_cache, field_name, None), f"history.bw_runtime_cache.{field_name}")
        _assert_history_snapshot_layout(
            history_bw_cache,
            bw_input,
            ("candidate_mask",),
            history_prefix="history.bw_runtime_cache",
            source_prefix="bw_input",
        )
        _bool(getattr(history_bw_cache, "candidate_mask", None), "history.bw_runtime_cache.candidate_mask")

        history_bw_stage = getattr(history, "bw_runtime_stage", None)
        if _is_native_stage_fields(history_bw_stage):
            _assert_history_snapshot_layout(
                history_bw_stage,
                main.bw_stage_fields,
                stage_float_fields,
                history_prefix="history.bw_runtime_stage",
                source_prefix="stage.bw",
            )
            _assert_history_snapshot_layout(
                history_bw_stage,
                main.bw_stage_fields,
                stage_long_fields,
                history_prefix="history.bw_runtime_stage",
                source_prefix="stage.bw",
            )
            _assert_history_snapshot_layout(
                history_bw_stage,
                main.bw_stage_fields,
                stage_bool_fields,
                history_prefix="history.bw_runtime_stage",
                source_prefix="stage.bw",
            )
            _stage_fields(history_bw_stage, "history.bw_runtime_stage")
        else:
            for field_name in stage_float_fields:
                _float(None, f"history.bw_runtime_stage.{field_name}")
            for field_name in stage_long_fields:
                _long(None, f"history.bw_runtime_stage.{field_name}")
            for field_name in stage_bool_fields:
                _bool(None, f"history.bw_runtime_stage.{field_name}")

        for snapshot_name, history_state, source_stage in (
            ("accel", getattr(history, "accel_runtime_state", None), main.accel_stage_field_buffers[0]),
            ("sat", getattr(history, "sat_runtime_state", None), main.sat_stage_fields),
        ):
            _assert_history_snapshot_layout(
                history_state,
                tensor_state,
                bw_runtime_state_float_fields,
                history_prefix=f"history.{snapshot_name}_runtime_state",
                source_prefix="state",
            )
            for field_name in bw_runtime_state_float_fields:
                _float(getattr(history_state, field_name, None), f"history.{snapshot_name}_runtime_state.{field_name}")
            _assert_history_snapshot_layout(
                history_state,
                tensor_state,
                ("last_sat_selection_matrix",),
                history_prefix=f"history.{snapshot_name}_runtime_state",
                source_prefix="state",
            )
            _long(
                getattr(history_state, "last_sat_selection_matrix", None),
                f"history.{snapshot_name}_runtime_state.last_sat_selection_matrix",
            )
            _assert_history_snapshot_layout(
                history_state,
                tensor_state,
                bw_runtime_state_int_fields,
                history_prefix=f"history.{snapshot_name}_runtime_state",
                source_prefix="state",
            )
            for field_name in bw_runtime_state_int_fields:
                _int(getattr(history_state, field_name, None), f"history.{snapshot_name}_runtime_state.{field_name}")
            _bool(None, f"history.{snapshot_name}_runtime_state.hotspot_member_mask_bool_unused")

            history_stage = getattr(history, f"{snapshot_name}_runtime_stage", None)
            if _is_native_stage_fields(history_stage):
                _assert_history_snapshot_layout(
                    history_stage,
                    source_stage,
                    stage_float_fields,
                    history_prefix=f"history.{snapshot_name}_runtime_stage",
                    source_prefix=f"stage.{snapshot_name}",
                )
                _assert_history_snapshot_layout(
                    history_stage,
                    source_stage,
                    stage_long_fields,
                    history_prefix=f"history.{snapshot_name}_runtime_stage",
                    source_prefix=f"stage.{snapshot_name}",
                )
                _assert_history_snapshot_layout(
                    history_stage,
                    source_stage,
                    stage_bool_fields,
                    history_prefix=f"history.{snapshot_name}_runtime_stage",
                    source_prefix=f"stage.{snapshot_name}",
                )
                _stage_fields(history_stage, f"history.{snapshot_name}_runtime_stage")
            else:
                for field_name in stage_float_fields:
                    _float(None, f"history.{snapshot_name}_runtime_stage.{field_name}")
                for field_name in stage_long_fields:
                    _long(None, f"history.{snapshot_name}_runtime_stage.{field_name}")
                for field_name in stage_bool_fields:
                    _bool(None, f"history.{snapshot_name}_runtime_stage.{field_name}")

        for name in (
            "lyapunov_pressure_ema",
            "lyapunov_virtual_queue",
            "lyapunov_service_est",
            "lyapunov_instant_pressure",
        ):
            _float(getattr(main, name, None), f"main.{name}")

        candidate_mode_code = {
            "associated": 0,
            "assoc": 0,
            "associated_queue_topk": 1,
            "assoc_queue_topk": 1,
            "radius": 2,
            "dist": 2,
            "distance": 2,
            "nearest": 3,
        }.get(str(domains.candidate.candidate_mode), 0)
        sat_candidate_mode_code = 0 if str(domains.sat_geometry.sat_candidate_mode) == "elevation" else 1
        boundary_mode_code = 1 if str(domains.accel_safety.boundary_mode) == "reflect" else 0
        reward_mode_code = {
            "dense": 0,
            "controllable_flow": 1,
            "throughput_only": 2,
            "weighted_workload_delta": 3,
            "weighted_workload_level": 4,
            "positive_weighted_workload_level": 5,
            "gu_queue_level": 6,
            "system_queue_level": 7,
            "gu_service_queue": 8,
            "relative_weighted_workload_delta": 9,
            "sat_relay_processed": 10,
            "sat_backhaul_drop": 11,
        }.get(str(main.bw_reward_mode_active), 0)
        traffic_model_code = 1 if str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower() == "sticky_subset_hotspot" else 0
        pathloss_mode_code = 1 if str(domains.channel.pathloss_mode) == "free_space" else 0
        energy_model_code = 1 if str(domains.bw_link.energy_model) == "rotor" else 0
        post_p = domains.post_stats_safety
        danger_trigger_mode_code = {
            "risk_or_intervention": 0,
            "or": 0,
            "intervention_any": 1,
            "intervention_threshold": 2,
        }.get(str(post_p.danger_imitation_trigger_mode), 1)
        int_params = (
            int(main.num_envs),
            int(main.num_uav),
            int(main.num_gu),
            int(main.num_sat),
            int(main.users_obs_max),
            int(main.sats_obs_max),
            int(main.visible_sats_max),
            int(main.sat_num_select),
            int(runtime.history.capacity),
            int(main.accel_actor_source_mode_code),
            int(main.sat_actor_source_mode_code),
            int(main.bw_actor_source_mode_code),
            int(main.flow_proxy_base_action_mode_code),
            int(candidate_mode_code),
            int(sat_candidate_mode_code),
            int(boundary_mode_code),
            int(reward_mode_code),
            int(traffic_model_code),
            int(pathloss_mode_code),
            int(energy_model_code),
            int(domains.candidate.candidate_k),
            0 if domains.candidate.candidate_radius is None else 1,
            int(bool(domains.access_rate.enable_bw_action)),
            int(bool(domains.access_rate.interference_enabled)),
            int(bool(domains.sat_geometry.doppler_enabled)),
            int(bool(domains.sat_geometry.doppler_atten_enabled)),
            int(bool(domains.sat_geometry.doppler_observed)),
            int(bool(domains.sat_geometry.doppler_precomp_enabled)),
            int(bool(domains.accel_safety.use_avoidance)),
            int(bool(domains.accel_safety.use_energy_safety)),
            int(bool(domains.accel_safety.boundary_hard_filter_enabled)),
            int(bool(domains.accel_safety.pairwise_hard_filter_enabled)),
            int(bool(domains.bw_queue_deadline.deadline_enabled)),
            int(bool(domains.bw_link.energy_enabled)),
            int(bool(main.bw_flow_proxy_enabled)),
            int(main.bw_flow_proxy_reward_mode_code),
            int(bool(domains.local_obs.obs_own_include_assoc_uav_cost)),
            int(bool(domains.local_obs.obs_own_include_uav_id_norm)),
            int(bool(domains.local_obs.obs_sat_include_sat_cost)),
            int(_native_local_obs_user_proxy_feature_dim(domains.local_obs)),
            int(domains.local_obs.num_uav),
            int(domains.local_obs.num_gu),
            int(domains.local_obs.num_sat),
            int(main.sat_visible_width),
            int(main.sat_subset_members_base.shape[0]) if torch.is_tensor(main.sat_subset_members_base) else 0,
            sat_schema.SAT_EGO_DIM,
            int(critic_schema.CRITIC_GU_NODE_DIM),
            sat_schema.SAT_TOKEN_DIM,
            {"linear": 0, "log": 1, "quadratic": 2}.get(str(domains.reward_metrics.queue_penalty_mode), 0),
            {"total": 0, "weighted": 1, "delta": 0}.get(str(domains.reward_metrics.queue_delta_mode), 0),
            int(bool(domains.reward_metrics.queue_reward_use_arrival_norm)),
            int(bool(domains.reward_metrics.use_queue_log_smoothing)),
            int(bool(domains.reward_metrics.use_active_queue_delta)),
            int(bool(domains.reward_metrics.use_energy_reward)),
            int(bool(domains.reward_metrics.use_reward_tanh)),
            int(bool(domains.reward_metrics.centroid_cross_anneal_enabled)),
            int(domains.reward_metrics.eta_centroid_final is not None),
            int(bool(post_p.close_risk_enabled)),
            int(bool(post_p.danger_imitation_enabled)),
            int(danger_trigger_mode_code),
            int(post_p.avoidance_prealert_factor is not None),
            int(str(post_p.avoidance_prealert_mode) == "ttc"),
            int(post_p.avoidance_prealert_dist_cap is not None),
            int(bool(domains.channel.atm_loss_enabled)),
            int({"inverse": 0, "linear": 1, "quadratic": 2}.get(str(domains.accel_safety.avoidance_repulse_mode), 0)),
            int(bool(getattr(cfg, "avoidance_repulse_clip", True))),
            int(bool(domains.accel_safety.avoidance_closing_gain_enabled)),
            int(bool(domains.accel_safety.avoidance_closing_gain_top1_only)),
            int(bool(domains.channel.rain_loss_enabled)),
            int(bool(domains.local_obs.obs_user_include_arrival_rate)),
            int(bool(domains.local_obs.obs_user_include_recent_arrival)),
            int(bool(domains.local_obs.obs_user_include_recent_service)),
            int(bool(domains.local_obs.obs_user_include_queue_headroom)),
            int(bool(domains.local_obs.obs_user_include_local_gu_service_cost)),
            int(bool(domains.local_obs.obs_user_include_assoc_uav_cost)),
            int(bool(domains.local_obs.obs_user_include_assoc_sat_cost_mean)),
            int(bool(domains.local_obs.obs_user_include_weighted_queue_cost)),
            int(bool(domains.local_obs.obs_user_include_weighted_queue_cost_relative)),
            int(bool(domains.local_obs.obs_user_include_urgency_risk)),
            int(bool(domains.local_obs.obs_user_include_downstream_pressure)),
            int(bool(domains.local_obs.obs_user_include_service_gap)),
            int(bool(domains.local_obs.obs_user_include_service_gap_risk)),
            int(bool(domains.local_obs.obs_user_include_deadline_slack)),
            int(bool(domains.local_obs.obs_user_include_deadline_risk)),
            int(critic_schema.CRITIC_UAV_GU_EDGE_DIM),
            int(critic_schema.CRITIC_UAV_SAT_EDGE_DIM),
            int(critic_schema.CRITIC_UAV_UAV_EDGE_DIM),
            int(getattr(main, "accel_live_obs_buffers", (None,))[0].sat_tokens.shape[1])
            if getattr(main, "accel_live_obs_buffers", None) is not None
            else _accel_sat_width_from_config(cfg),
            int(bool(domains.accel_safety.safety_shield_native_enabled)),
            int(domains.accel_safety.safety_shield_tensor_iters),
            int(bool(domains.history_output.runtime_snapshots_enabled)),
            max(int(getattr(cfg, "access_bw_decision_interval", 1) or 1), 1),
            max(int(getattr(cfg, "sat_decision_interval", 1) or 1), 1),
        )
        float_params = (
            float(domains.numeric.access_eta_quantum),
            float(domains.numeric.access_rate_quantum),
            float(domains.numeric.flow_bits_quantum),
            float(domains.numeric.queue_state_quantum),
            float(domains.numeric.summary_metric_quantum),
            float(domains.channel.access_gain_quantum),
            float(domains.channel.access_pathloss_db_quantum),
            float(domains.channel.pathloss_const_db),
            float(domains.channel.carrier_freq),
            float(domains.channel.xi_los),
            float(domains.channel.xi_nlos),
            float(domains.channel.los_a),
            float(domains.channel.los_b),
            float(domains.candidate.uav_height),
            float(domains.candidate.pl_threshold_db),
            0.0 if domains.candidate.candidate_radius is None else float(domains.candidate.candidate_radius),
            float(domains.sat_geometry.ref_lat_deg),
            float(domains.sat_geometry.ref_lon_deg),
            float(domains.sat_geometry.r_earth),
            float(domains.sat_geometry.uav_height),
            float(domains.sat_geometry.theta_min_rad),
            float(domains.sat_geometry.carrier_freq),
            float(domains.sat_geometry.speed_of_light),
            float(domains.sat_geometry.nu_max),
            float(domains.sat_geometry.queue_max_sat),
            float(domains.sat_geometry.uav_tx_power),
            float(domains.sat_geometry.noise_density),
            float(domains.sat_geometry.subcarrier_spacing),
            float(domains.sat_geometry.sat_candidate_elevation_weight),
            float(domains.sat_geometry.sat_candidate_queue_weight),
            float(domains.sat_geometry.sat_candidate_se_weight),
            float(domains.local_obs.tau0),
            float(domains.local_obs.map_size),
            float(domains.local_obs.v_max),
            float(domains.local_obs.uav_energy_init),
            float(domains.local_obs.queue_max_gu),
            float(domains.local_obs.queue_max_uav),
            float(domains.local_obs.queue_max_sat),
            float(domains.local_obs.r_earth),
            float(domains.local_obs.sat_height),
            float(domains.local_obs.uav_tx_power),
            float(domains.local_obs.noise_density),
            float(domains.local_obs.nu_max),
            float(domains.local_obs.subcarrier_spacing),
            float(domains.local_obs.service_gap_cap_steps),
            float(domains.local_obs.avoidance_alert_factor),
            float(domains.local_obs.d_safe),
            float(domains.accel_safety.a_max),
            float(domains.accel_safety.v_max),
            float(domains.accel_safety.tau0),
            float(domains.accel_safety.map_size),
            float(domains.accel_safety.d_safe),
            float(domains.accel_safety.uav_energy_init),
            float(domains.accel_safety.uav_opt_speed),
            float(domains.accel_safety.energy_safe_threshold),
            float(domains.accel_safety.boundary_margin),
            float(domains.bw_queue_deadline.tau0),
            float(domains.bw_queue_deadline.queue_max_gu),
            float(domains.bw_queue_deadline.queue_max_uav),
            float(domains.bw_queue_deadline.queue_max_sat),
            float(domains.bw_queue_deadline.service_gap_increment),
            float(domains.bw_queue_deadline.service_gap_relief_coef),
            float(domains.bw_queue_deadline.service_gap_cap_steps),
            float(domains.bw_queue_deadline.deadline_age_increment),
            float(domains.bw_queue_deadline.deadline_service_relief_coef),
            float(domains.bw_queue_deadline.deadline_age_cap_steps),
            float(domains.bw_queue_deadline.deadline_expire_rate),
            float(domains.bw_link.tau0),
            float(domains.bw_link.backhaul_rate_quantum),
            float(domains.bw_link.b_backhaul_per_sat),
            float(domains.bw_link.b_backhaul_per_sat_scale),
            float(domains.bw_link.uav_tx_power),
            float(domains.bw_link.noise_density),
            float(domains.bw_link.p_comm_link),
            float(domains.bw_link.queue_max_sat),
            float(domains.reward_metrics.t_steps),
            float(domains.reward_metrics.omega_q),
            float(domains.reward_metrics.omega_e),
            float(domains.reward_metrics.eta_service),
            float(domains.reward_metrics.eta_q_delta),
            float(domains.reward_metrics.eta_batt),
            float(domains.reward_metrics.eta_crash),
            float(domains.reward_metrics.eta_accel),
            float(domains.reward_metrics.eta_drop),
            float(domains.reward_metrics.eta_drop_step),
            float(domains.reward_metrics.eta_drop_gu),
            float(domains.reward_metrics.eta_drop_uav),
            float(domains.reward_metrics.eta_drop_sat),
            float(domains.reward_metrics.reward_w_access),
            float(domains.reward_metrics.reward_w_relay),
            float(domains.reward_metrics.reward_w_pre_backlog),
            float(domains.reward_metrics.reward_w_pre_drop),
            float(domains.reward_metrics.reward_w_pre_service_gap),
            float(domains.reward_metrics.reward_w_pre_overflow_risk),
            float(main.bw_uav_orbit_radius),
            float(main.bw_uav_orbit_radius_sq),
            float(main.bw_sat_orbit_radius_sq),
            float(main.bw_backhaul_gain_const),
            float(main.bw_effective_b_backhaul_per_sat),
            float(main.bw_inv_a_max),
            float(main.bw_doppler_cap),
            float(main.bw_doppler_rho),
            float(main.bw_doppler_sigma),
            float(domains.access_rate.gu_tx_power),
            float(domains.access_rate.noise_density),
            float(domains.access_rate.b_acc),
            float(domains.access_rate.interference_quantum),
            float(domains.bw_workload.eps),
            float(domains.reward_metrics.bw_weighted_workload_ema_decay),
            float(domains.reward_metrics.eta_throughput_access),
            float(domains.reward_metrics.eta_throughput_backhaul),
            float(domains.reward_metrics.eta_close_risk),
            float(domains.reward_metrics.throughput_only_access_coef),
            float(domains.reward_metrics.throughput_only_backhaul_coef),
            float(domains.reward_metrics.throughput_only_gu_queue_coef),
            float(domains.reward_metrics.queue_norm_k),
            float(domains.reward_metrics.queue_norm_arrival_floor),
            float(domains.reward_metrics.queue_log_k),
            float(domains.reward_metrics.omega_q_gu),
            float(domains.reward_metrics.omega_q_uav),
            float(domains.reward_metrics.omega_q_sat),
            float(domains.reward_metrics.omega_q_tail),
            float(domains.reward_metrics.q_norm_tail_q0),
            float(domains.reward_metrics.tail_q_small),
            float(domains.reward_metrics.tail_eta_accel_gain),
            float(domains.reward_metrics.eta_centroid),
            float(0.0 if domains.reward_metrics.eta_centroid_final is None else domains.reward_metrics.eta_centroid_final),
            float(domains.reward_metrics.eta_centroid_decay_steps),
            float(domains.reward_metrics.centroid_dist_scale),
            float(domains.reward_metrics.centroid_cross_queue_gain),
            float(domains.reward_metrics.centroid_cross_q_delta_gain),
            float(domains.reward_metrics.centroid_cross_crash_gain),
            float(domains.reward_metrics.p_fly_base),
            float(domains.reward_metrics.p_fly_coeff),
            float(domains.reward_metrics.rotor_p0),
            float(domains.reward_metrics.rotor_pi),
            float(domains.reward_metrics.rotor_u_tip),
            float(domains.reward_metrics.rotor_v0),
            float(domains.reward_metrics.rotor_d0),
            float(domains.reward_metrics.rotor_rho),
            float(domains.reward_metrics.rotor_s),
            float(domains.reward_metrics.rotor_area),
            float(domains.reward_metrics.n_rf),
            float(domains.bw_flow_proxy.aux_delta),
            float(domains.bw_flow_proxy.eps),
            0.0 if post_p.avoidance_prealert_factor is None else float(post_p.avoidance_prealert_factor),
            float(post_p.avoidance_prealert_closing_speed),
            float(post_p.avoidance_prealert_ttc),
            0.0 if post_p.avoidance_prealert_dist_cap is None else float(post_p.avoidance_prealert_dist_cap),
            float(post_p.close_risk_cap),
            float(post_p.danger_imitation_close_risk_thresh),
            float(post_p.danger_imitation_intervention_thresh),
            float(domains.channel.atm_loss_db),
            float(2.0 if getattr(cfg, "baseline_accel_gain", None) is None else getattr(cfg, "baseline_accel_gain")),
            float(getattr(cfg, "baseline_assoc_bonus", 0.3) or 0.0),
            float(getattr(cfg, "baseline_repulse_gain", 0.0) or 0.0),
            float(getattr(cfg, "baseline_repulse_radius_factor", 1.5) or 0.0),
            float(getattr(cfg, "baseline_energy_weight", 1.0) or 0.0),
            float(getattr(cfg, "baseline_energy_low", 0.3) or 0.3),
            float(getattr(cfg, "baseline_sat_se_weight", 1.0) or 0.0),
            float(getattr(cfg, "baseline_sat_queue_penalty", 0.5) or 0.0),
            float(getattr(cfg, "baseline_sat_load_penalty", 1.0) or 0.0),
            float(getattr(cfg, "baseline_sat_bw_reward", 0.75) or 0.0),
            float(getattr(cfg, "baseline_sat_stay_bonus", 0.25) or 0.0),
            float(getattr(cfg, "baseline_sat_switch_margin", 0.15) or 0.0),
            float(getattr(cfg, "baseline_cluster_stop_radius", 20.0) or 0.0),
            float(getattr(cfg, "baseline_cluster_speed_tol", 2.0) or 0.0),
            float(getattr(cfg, "baseline_cluster_slow_radius", 120.0) or 0.0),
            float(cfg.uav_opt_speed if getattr(cfg, "baseline_cluster_cruise_speed", None) is None else getattr(cfg, "baseline_cluster_cruise_speed")),
            float(getattr(cfg, "baseline_cluster_vel_gain", 1.0) or 0.0),
            float(getattr(cfg, "sat_logit_scale", 1.0e9) or 1.0e9),
            float(domains.accel_safety.avoidance_closing_gain_cap),
            float(domains.accel_safety.avoidance_eta),
            float(domains.bw_workload.sat_active_ref_count),
            float(10.0 ** (max(float(domains.access_rate.noise_figure_db), 0.0) / 10.0)),
            float(10.0 ** (max(float(domains.sat_geometry.noise_figure_db), 0.0) / 10.0)),
            float(10.0 ** (max(float(domains.bw_link.noise_figure_db), 0.0) / 10.0)),
            float(domains.access_rate.fading_mode_code),
            float(domains.access_rate.rician_k),
            float(domains.channel.rain_rate_001_mmph),
            float(domains.channel.rain_height_km),
            float(domains.channel.station_height_km),
            float(domains.channel.latitude_deg),
            float(domains.channel.rain_exceedance_pct),
            float(domains.channel.rain_polarization_tilt_deg),
            float(domains.accel_safety.safety_shield_distance_buffer),
            float(domains.accel_safety.safety_shield_a_safe),
            float(domains.accel_safety.safety_shield_step_gain),
            float(domains.accel_safety.safety_shield_tolerance),
            float(domains.accel_safety.safety_shield_brake_rho),
            float(getattr(cfg, "baseline_lyapunov_v", 2.0) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_urgency_alpha", 1.0) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_drift_weight", 1.0) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_action_cost", 0.05) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_ema_beta", 0.6) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_bw_temp", 0.6) or 0.6),
            float(getattr(cfg, "baseline_lyapunov_bw_floor", 0.02) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_bw_service_scale", 1.0) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_sat_drift_weight", 0.6) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_sat_switch_bias", 0.1) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_sat_abs_se_weight", 0.5) or 0.0),
            float(getattr(cfg, "baseline_lyapunov_sat_doppler_penalty", 0.35) or 0.0),
        )
        return native_cuda.NativeCudaRuntimeABI(
            float_tensors=tuple(float_tensors),
            long_tensors=tuple(long_tensors),
            bool_tensors=tuple(bool_tensors),
            int_tensors=tuple(int_tensors),
            int_params=int_params,
            float_params=float_params,
        )

    @contextmanager
    def _native_main_kernel_workspace_context(
        self,
        *,
        runtime: StructuredGpuRolloutRuntime,
        tensor_state: StructuredBatchRuntimeTensorState,
        cfg: Any,
        rng: torch.Generator,
        bound_kernels: dict[str, Any],
        num_envs_override: int | None = None,
        slot_state_payloads: list[dict[str, Any]] | None = None,
        slot_rngs: list[np.random.Generator] | None = None,
    ):
        old_runtime = self._native_rollout_runtime
        old_tensor_state = self._runtime_tensor_state
        old_cfg = self._cfg
        old_rng = self._native_torch_rng
        old_num_envs = self._num_envs
        old_slot_state_payloads = self._slot_state_payloads
        old_slot_rngs = self._slot_rngs
        old_domains = getattr(self, "_native_main_kernel_typed_domains_obj", None)
        old_bound = getattr(self, "_native_main_kernel_bound_kernels", None)
        self._native_rollout_runtime = runtime
        self._runtime_tensor_state = tensor_state
        self._cfg = cfg
        self._native_torch_rng = rng
        self._native_main_kernel_bound_kernels = bound_kernels
        if num_envs_override is not None:
            self._num_envs = int(num_envs_override)
        if slot_state_payloads is not None:
            self._slot_state_payloads = slot_state_payloads
        if slot_rngs is not None:
            self._slot_rngs = slot_rngs
        try:
            yield
        finally:
            self._native_rollout_runtime = old_runtime
            self._runtime_tensor_state = old_tensor_state
            self._cfg = old_cfg
            self._native_torch_rng = old_rng
            self._num_envs = old_num_envs
            self._slot_state_payloads = old_slot_state_payloads
            self._slot_rngs = old_slot_rngs
            if old_domains is not None:
                setattr(self, "_native_main_kernel_typed_domains_obj", old_domains)
            elif hasattr(self, "_native_main_kernel_typed_domains_obj"):
                delattr(self, "_native_main_kernel_typed_domains_obj")
            if old_bound is not None:
                setattr(self, "_native_main_kernel_bound_kernels", old_bound)
            elif hasattr(self, "_native_main_kernel_bound_kernels"):
                delattr(self, "_native_main_kernel_bound_kernels")

    def _bind_native_main_kernel_history_outputs(
        self,
        runtime: StructuredGpuRolloutRuntime,
        history,
    ) -> None:
        runtime.main.accel_history_out = _NativeAccelTrainingHistoryOutBuffers(
            world_batch=history.accel_stage.world_batch,
            ego_features=history.accel_stage.ego_features,
            ego_cell=history.accel_stage.ego_cell,
            gu_tokens=history.accel_stage.gu_tokens,
            gu_mask=history.accel_stage.gu_mask,
            peer_tokens=history.accel_stage.peer_tokens,
            peer_mask=history.accel_stage.peer_mask,
            sat_tokens=history.accel_stage.sat_tokens,
            sat_mask=history.accel_stage.sat_mask,
            danger_imitation_targets=history.accel_stage.danger_imitation_targets,
            danger_imitation_masks=history.accel_stage.danger_imitation_masks,
        )
        runtime.main.sat_history_out = _NativeSatTrainingHistoryOutBuffers(
            world_batch=history.sat_stage.world_batch,
            ego_features=history.sat_stage.ego_features,
            demand_features=history.sat_stage.demand_features,
            role_features=history.sat_stage.role_features,
            sat_tokens=history.sat_stage.sat_tokens,
            sat_mask=history.sat_stage.sat_mask,
            sat_valid_mask=history.sat_stage.sat_valid_mask,
            candidate_sat_ids=history.sat_stage.candidate_sat_ids,
            subset_mask=history.sat_stage.subset_mask,
            subset_members=history.sat_stage.subset_members,
        )
        runtime.main.bw_history_out = _NativeBwTrainingHistoryOutBuffers(
            world_batch=history.bw_stage.world_batch,
            ego_features=history.bw_stage.ego_features,
            selected_sat_tokens=history.bw_stage.selected_sat_tokens,
            selected_sat_mask=history.bw_stage.selected_sat_mask,
            gu_tokens=history.bw_stage.gu_tokens,
            gu_mask=history.bw_stage.gu_mask,
            bw_valid_mask=history.bw_stage.bw_valid_mask,
            rewards=history.bw_stage.rewards,
            terminated=history.terminated,
            truncated=history.truncated,
            terminal_next_world_mask=history.terminal_next_world_mask,
            bw_access_rewards=history.bw_stage.bw_access_rewards,
            bw_weighted_workload_delta_rewards=history.bw_stage.bw_weighted_workload_delta_rewards,
            bw_weighted_workload_level_rewards=history.bw_stage.bw_weighted_workload_level_rewards,
            bw_gu_queue_level_rewards=history.bw_stage.bw_gu_queue_level_rewards,
            bw_system_queue_level_rewards=history.bw_stage.bw_system_queue_level_rewards,
            bw_gu_service_queue_rewards=history.bw_stage.bw_gu_service_queue_rewards,
            bw_flow_proxy_scores=history.bw_stage.bw_flow_proxy_scores,
            bw_flow_proxy_masks=history.bw_stage.bw_flow_proxy_masks,
            bw_flow_proxy_deltas=history.bw_stage.bw_flow_proxy_deltas,
        )
        runtime.main.next_accel_history_world_out = history.accel_stage.world_batch
        runtime.main.terminal_next_history_world_out = history.terminal_next_world
        runtime.main.terminal_next_history_mask_out = history.terminal_next_world_mask

    def _runtime_step_select_rollout_storage(self, *, num_envs: int) -> None:
        runtime = self._native_main_kernel_runtime()
        runtime.activate_rollout_history()
        runtime.history.num_envs = int(num_envs)
        self._bind_native_main_kernel_history_outputs(runtime, runtime.history)

    def _refresh_native_main_kernel_typed_domains(self) -> _NativeMainKernelTypedDomains:
        domains = _native_typed_domains_from_cfg(self._cfg, num_envs=int(self._num_envs))
        setattr(
            self,
            "_native_main_kernel_typed_domains_obj",
            domains,
        )
        setattr(self, "_native_main_kernel_bound_kernels", {})
        return domains

    def _native_main_kernel_typed_domains(self) -> _NativeMainKernelTypedDomains:
        domains = getattr(self, "_native_main_kernel_typed_domains_obj", None)
        if not isinstance(domains, _NativeMainKernelTypedDomains):
            self._refresh_native_main_kernel_typed_domains()
            domains = getattr(self, "_native_main_kernel_typed_domains_obj", None)
        if not isinstance(domains, _NativeMainKernelTypedDomains):
            raise RuntimeError("native main-kernel typed domains were not initialized.")
        return domains

    def _native_main_kernel_kernel_specs(self) -> dict[str, Callable[..., Any]]:
        return {}

    def _native_main_kernel_required_compile_names(self) -> set[str] | None:
        raw = getattr(self._cfg, "structured_native_main_kernel_required_compile_names", None)
        names = _kernel_name_set_from_config(raw)
        if names:
            return names
        return set(self._native_main_kernel_kernel_specs().keys())

    def _assert_native_main_kernel_compiled_segment_contract(self, tensor_device: torch.device) -> None:
        if tensor_device.type != "cuda":
            return
        if not bool(getattr(self._cfg, "structured_native_main_kernel_require_compiled_segments", False)):
            return
        if not self._native_main_kernel_direct_required(range(int(self._num_envs))):
            raise RuntimeError(
                "official CUDA native main-kernel compiled-segment contract requires the direct full-batch "
                "tensor path. This configuration cannot fall back to compat stage/object execution."
            )
        segment_names = set(self._native_main_kernel_kernel_specs().keys())
        required_names = self._native_main_kernel_required_compile_names()
        if required_names is not None:
            missing_segments = sorted(segment_names.difference(required_names))
            if missing_segments:
                raise RuntimeError(
                    "native CUDA main-kernel requires every official env segment to be compile-required; "
                    "missing from structured_native_main_kernel_required_compile_names: "
                    + ", ".join(missing_segments)
                )
        strict_names = _kernel_name_set_from_config(getattr(self._cfg, "structured_kernel_compile_required", None))
        if strict_names is not None:
            missing_strict = sorted((segment_names if required_names is None else required_names).difference(strict_names))
            if missing_strict:
                raise RuntimeError(
                    "native CUDA main-kernel compiled segment contract is enabled, but these segments are "
                    "not present in structured_kernel_compile_required: "
                    + ", ".join(missing_strict)
                )
        mode = str(getattr(self._cfg, "structured_kernel_operator_mode", "auto") or "auto").strip().lower()
        if mode not in {"auto", "compile"}:
            raise RuntimeError(
                "native CUDA main-kernel compiled segment contract requires "
                "structured_kernel_operator_mode='auto' or 'compile'."
            )

    def _native_main_kernel_cache_key(self, name: str, device: torch.device | str) -> tuple[str, str, int | None]:
        tensor_device = torch.device(device)
        device_index = tensor_device.index
        if tensor_device.type == "cuda" and device_index is None and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        return (str(name), str(tensor_device.type), int(device_index) if device_index is not None else None)

    def _prebind_native_main_kernel_kernels(self) -> None:
        return

    def _native_main_kernel_callable(
        self,
        *,
        name: str,
        eager_fn: Callable[..., Any],
        device: torch.device | str,
    ) -> Callable[..., Any]:
        tensor_device = torch.device(device)
        cache = getattr(self, "_native_main_kernel_bound_kernels", None)
        key = self._native_main_kernel_cache_key(name, tensor_device)
        if cache is not None and key in cache:
            return cache[key]
        if tensor_device.type == "cuda" and self._native_main_kernel_strict_cuda_contract_enabled():
            if not self._native_main_kernel_direct_required(range(int(self._num_envs))):
                raise RuntimeError(
                    f"native main-kernel callable {name!r} requires the official CUDA direct full-batch "
                    "tensor path; compat stage/object fallback is forbidden."
                )
        if tensor_device.type == "cuda" and self._native_main_kernel_direct_required(range(int(self._num_envs))):
            raise RuntimeError(
                f"native main-kernel callable {name!r} was not pre-bound; "
                "official CUDA direct live path must resolve kernels before the rollout hot loop."
            )
        raise RuntimeError(
            f"native main-kernel callable {name!r} was removed. "
            "Official native rollout env stages are launched through sagin_marl.env.native_cuda bindings."
        )

    @staticmethod
    def _native_flow_proxy_base_action_mode_code(mode: object) -> int:
        mode_s = str(mode or "executed").strip().lower()
        if mode_s == "executed":
            return int(native_cuda.FLOW_BASE_EXECUTED)
        if mode_s == "deterministic":
            return int(native_cuda.FLOW_BASE_DETERMINISTIC)
        if mode_s == "external_live_override":
            return int(native_cuda.FLOW_BASE_EXTERNAL_LIVE_OVERRIDE)
        raise RuntimeError(f"native rollout flow proxy base action mode {mode_s!r} is not supported.")

    @staticmethod
    def _bind_native_flow_proxy_override_constant(runtime: StructuredGpuRolloutRuntime) -> None:
        main = runtime.main
        if int(getattr(main, "flow_proxy_base_action_mode_code", 0) or 0) != int(
            native_cuda.FLOW_BASE_EXTERNAL_LIVE_OVERRIDE
        ):
            return
        override_action = getattr(main, "live_bw_flow_proxy_override_action", None)
        if not torch.is_tensor(override_action):
            raise RuntimeError(
                "native rollout flow proxy base action mode 'external_live_override' requires "
                "runtime.main.live_bw_flow_proxy_override_action to be a fixed CUDA tensor."
            )
        override_action.zero_()

    def native_rollout_program(self) -> StructuredGpuNativeRolloutProgram:
        """Build the single native GPU rollout program for official live paths."""
        runtime = self.native_rollout_runtime
        if runtime is None:
            raise RuntimeError("native rollout program requires a persistent native rollout runtime.")
        visible_cfg = getattr(self._cfg, "visible_sats_max", None)
        if visible_cfg is None:
            visible_cfg = getattr(self._cfg, "sats_obs_max", None)
        runtime.main.flow_proxy_base_action_mode_code = self._native_flow_proxy_base_action_mode_code(
            getattr(self._cfg, "bw_flow_proxy_base_action_mode", "executed")
        )
        self._bind_native_flow_proxy_override_constant(runtime)
        self._refresh_native_main_kernel_typed_domains()
        self._publish_runtime_reset_random_tape(range(int(self._num_envs)))
        step_program = StructuredGpuNativeRuntimeStepProgram(
            runtime=runtime,
            executor=self,
            num_envs=int(self._num_envs),
            fixed_visible_sat_width=None if visible_cfg is None else max(int(visible_cfg), 0),
        )
        return StructuredGpuNativeRolloutProgram(
            step_program=step_program,
        )

    def debug_native_hot_replay_program(
        self,
        *,
        capacity: int = 3,
        num_envs: int | None = None,
        selected_indices: Sequence[int] | None = None,
    ) -> StructuredGpuNativeRolloutProgram:
        """Build the historical isolated replay workspace for explicit debug tests only."""
        official_runtime = self.native_rollout_runtime
        if official_runtime is None:
            raise RuntimeError("native hot replay program requires a persistent native rollout runtime.")
        if selected_indices is None:
            replay_envs = int(self._num_envs if num_envs is None else num_envs)
            if replay_envs < 0 or replay_envs > int(self._num_envs):
                raise ValueError(f"native hot replay num_envs must be in [0, {int(self._num_envs)}], got {replay_envs}.")
            selected_tuple = tuple(range(replay_envs))
        else:
            selected_tuple = tuple(int(index) for index in selected_indices)
            if num_envs is not None and int(num_envs) != len(selected_tuple):
                raise ValueError(
                    f"native hot replay selected_indices has {len(selected_tuple)} envs, "
                    f"but num_envs={int(num_envs)} was requested."
                )
            if any(index < 0 or index >= int(self._num_envs) for index in selected_tuple):
                raise ValueError(
                    f"native hot replay selected_indices must be in [0, {int(self._num_envs)}), "
                    f"got {list(selected_tuple)!r}."
                )
            if len(set(selected_tuple)) != len(selected_tuple):
                raise ValueError("native hot replay selected_indices must not contain duplicates.")
            replay_envs = len(selected_tuple)
        visible_cfg = getattr(self._cfg, "visible_sats_max", None)
        if visible_cfg is None:
            visible_cfg = getattr(self._cfg, "sats_obs_max", None)
        flow_proxy_base_action_mode_code = self._native_flow_proxy_base_action_mode_code(
            getattr(self._cfg, "bw_flow_proxy_base_action_mode", "executed")
        )
        hot_runtime = self._native_hot_replay_runtime
        hot_tensor_state = self._native_hot_replay_tensor_state
        hot_cfg = self._native_hot_replay_cfg
        hot_rng = self._native_hot_replay_torch_rng
        hot_bound_kernels = self._native_hot_replay_bound_kernels
        hot_selected_indices = self._native_hot_replay_selected_indices
        hot_slot_state_payloads = self._native_hot_replay_slot_state_payloads
        hot_slot_rngs = self._native_hot_replay_slot_rngs
        hot_history = None if hot_runtime is None else getattr(hot_runtime, "rollout_history", getattr(hot_runtime, "history", None))
        hot_main = None if hot_runtime is None else getattr(hot_runtime, "main", None)
        reuse_hot_workspace = bool(
            hot_runtime is not None
            and hot_tensor_state is not None
            and hot_cfg is not None
            and hot_rng is not None
            and isinstance(hot_bound_kernels, dict)
            and hot_history is not None
            and hot_main is not None
            and hot_selected_indices == selected_tuple
            and hot_slot_state_payloads is not None
            and hot_slot_rngs is not None
            and _same_structured_tensor_device(hot_runtime.device, official_runtime.device)
            and int(getattr(hot_tensor_state.uav_pos, "shape", (0,))[0]) == int(replay_envs)
            and int(getattr(hot_history, "capacity", 0) or 0) >= max(int(capacity), 1)
            and int(getattr(hot_history, "num_envs", 0) or 0) == int(replay_envs)
            and int(getattr(hot_main, "num_envs", 0) or 0) == int(replay_envs)
            and int(getattr(hot_history, "cursor", 0) or 0) < int(getattr(hot_history, "capacity", 0) or 0)
        )
        if not reuse_hot_workspace:
            hot_runtime = StructuredGpuRolloutRuntime(device=torch.device(official_runtime.device))
            hot_tensor_state = self._clone_runtime_tensor_state(self._runtime_tensor_state, indices=selected_tuple)
            hot_cfg = copy.copy(self._cfg)
            setattr(hot_cfg, "_structured_kernel_runtime_cache", {})
            hot_rng = self._make_native_torch_rng()
            hot_bound_kernels = {}
            hot_slot_state_payloads = [
                copy.deepcopy(self._slot_state_payloads[int(index)])
                for index in selected_tuple
            ]
            hot_slot_rngs = [
                self._clone_numpy_generator_state(self._slot_rngs[int(index)])
                for index in selected_tuple
            ]
            try:
                hot_rng.set_state(self._native_torch_rng.get_state())
            except RuntimeError:
                pass
            self._native_hot_replay_runtime = hot_runtime
            self._native_hot_replay_tensor_state = hot_tensor_state
            self._native_hot_replay_cfg = hot_cfg
            self._native_hot_replay_torch_rng = hot_rng
            self._native_hot_replay_bound_kernels = hot_bound_kernels
            self._native_hot_replay_selected_indices = selected_tuple
            self._native_hot_replay_slot_state_payloads = hot_slot_state_payloads
            self._native_hot_replay_slot_rngs = hot_slot_rngs
        executor = _NativeMainKernelWorkspaceExecutor(
            self,
            runtime=hot_runtime,
            tensor_state=hot_tensor_state,
            cfg=hot_cfg,
            rng=hot_rng,
            bound_kernels=hot_bound_kernels,
            num_envs_override=replay_envs,
            slot_state_payloads=hot_slot_state_payloads,
            slot_rngs=hot_slot_rngs,
        )
        if not reuse_hot_workspace:
            with executor._active():
                self.begin_native_main_kernel_rollout(capacity=max(int(capacity), 1), num_envs=replay_envs)
        hot_runtime.main.flow_proxy_base_action_mode_code = int(flow_proxy_base_action_mode_code)
        self._bind_native_flow_proxy_override_constant(hot_runtime)
        hot_runtime.main.selected_env_mapping = torch.as_tensor(
            selected_tuple,
            dtype=torch.long,
            device=torch.device(hot_runtime.device),
        )
        with executor._active():
            hot_runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(hot_runtime)
        step_program = StructuredGpuNativeRuntimeStepProgram(
            runtime=hot_runtime,
            executor=executor,
            num_envs=replay_envs,
            fixed_visible_sat_width=None if visible_cfg is None else max(int(visible_cfg), 0),
        )
        return StructuredGpuNativeRolloutProgram(step_program=step_program)

    @staticmethod
    def _select_runtime_tape_env_rows(
        value: torch.Tensor | None,
        selected_indices: Sequence[int],
        *,
        env_dim: int = 1,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(value):
            return None
        env_dim_i = int(env_dim)
        if value.ndim < 2:
            return value
        if env_dim_i < 0:
            env_dim_i += int(value.ndim)
        if env_dim_i < 0 or env_dim_i >= int(value.ndim):
            raise ValueError(f"runtime rollout tape env_dim={env_dim} is invalid for shape={tuple(value.shape)}.")
        selected_tuple = tuple(int(index) for index in selected_indices)
        axis_size = int(value.shape[env_dim_i])
        invalid = [index for index in selected_tuple if index < 0 or index >= axis_size]
        if invalid:
            raise ValueError(
                "runtime rollout tape selected env index out of bounds: "
                f"indices={invalid!r}, env_dim={env_dim_i}, axis_size={axis_size}, shape={tuple(value.shape)}."
            )
        selected = torch.as_tensor(selected_tuple, dtype=torch.long, device=value.device)
        return value.index_select(env_dim_i, selected).contiguous()

    def _copy_selected_random_rollout_tapes(
        self,
        *,
        source_runtime: StructuredGpuRolloutRuntime,
        target_runtime: StructuredGpuRolloutRuntime,
        selected_indices: Sequence[int],
    ) -> None:
        source_random = source_runtime.random
        target_runtime.write_random_rollout_tapes(
            arrival_tape=self._select_runtime_tape_env_rows(source_random.arrival_rollout_tape, selected_indices),
            arrival_rate_tape=self._select_runtime_tape_env_rows(source_random.arrival_rate_rollout_tape, selected_indices),
            hotspot_active_after_tape=self._select_runtime_tape_env_rows(
                source_random.hotspot_active_after_rollout_tape,
                selected_indices,
            ),
            reset_followup_arrival_tape=self._select_runtime_tape_env_rows(
                source_random.reset_followup_arrival_rollout_tape,
                selected_indices,
                env_dim=2,
            ),
            reset_followup_arrival_rate_tape=self._select_runtime_tape_env_rows(
                source_random.reset_followup_arrival_rate_rollout_tape,
                selected_indices,
                env_dim=2,
            ),
            reset_followup_hotspot_active_after_tape=self._select_runtime_tape_env_rows(
                source_random.reset_followup_hotspot_active_after_rollout_tape,
                selected_indices,
                env_dim=2,
            ),
            hotspot_active_tape=self._select_runtime_tape_env_rows(source_random.hotspot_active_rollout_tape, selected_indices),
            hotspot_mask_tape=self._select_runtime_tape_env_rows(source_random.hotspot_mask_rollout_tape, selected_indices),
            fading_gain_tape=self._select_runtime_tape_env_rows(source_random.fading_gain_rollout_tape, selected_indices),
            doppler_noise_tape=self._select_runtime_tape_env_rows(source_random.doppler_noise_rollout_tape, selected_indices),
        )
        reset_gu_pos = self._select_runtime_tape_env_rows(source_random.reset_gu_pos_rollout_tape, selected_indices)
        reset_uav_pos = self._select_runtime_tape_env_rows(source_random.reset_uav_pos_rollout_tape, selected_indices)
        reset_uav_vel = self._select_runtime_tape_env_rows(source_random.reset_uav_vel_rollout_tape, selected_indices)
        reset_gu_cluster_centers = self._select_runtime_tape_env_rows(
            source_random.reset_gu_cluster_centers_rollout_tape,
            selected_indices,
        )
        reset_gu_cluster_counts = self._select_runtime_tape_env_rows(
            source_random.reset_gu_cluster_counts_rollout_tape,
            selected_indices,
        )
        reset_arrival_base_scale = self._select_runtime_tape_env_rows(
            source_random.reset_arrival_base_scale_rollout_tape,
            selected_indices,
        )
        reset_deadline_steps = self._select_runtime_tape_env_rows(
            source_random.reset_deadline_steps_rollout_tape,
            selected_indices,
        )
        reset_doppler_residual = self._select_runtime_tape_env_rows(
            source_random.reset_doppler_residual_rollout_tape,
            selected_indices,
        )
        reset_effective_arrival_rate = self._select_runtime_tape_env_rows(
            source_random.reset_effective_arrival_rate_rollout_tape,
            selected_indices,
        )
        reset_episode_idx = self._select_runtime_tape_env_rows(source_random.reset_episode_idx_rollout_tape, selected_indices)
        if all(
            value is not None
            for value in (
                reset_gu_pos,
                reset_uav_pos,
                reset_uav_vel,
                reset_gu_cluster_centers,
                reset_gu_cluster_counts,
                reset_arrival_base_scale,
                reset_deadline_steps,
                reset_doppler_residual,
                reset_effective_arrival_rate,
                reset_episode_idx,
            )
        ):
            target_runtime.write_random_reset_rollout_tapes(
                gu_pos_tape=reset_gu_pos,
                uav_pos_tape=reset_uav_pos,
                uav_vel_tape=reset_uav_vel,
                gu_cluster_centers_tape=reset_gu_cluster_centers,
                gu_cluster_counts_tape=reset_gu_cluster_counts,
                gu_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_gu_queue_rollout_tape, selected_indices),
                uav_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_uav_queue_rollout_tape, selected_indices),
                sat_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_sat_queue_rollout_tape, selected_indices),
                arrival_base_scale_tape=reset_arrival_base_scale,
                deadline_steps_tape=reset_deadline_steps,
                doppler_residual_tape=reset_doppler_residual,
                effective_arrival_rate_tape=reset_effective_arrival_rate,
                arrival_rate_vec_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_arrival_rate_vec_rollout_tape,
                    selected_indices,
                ),
                arrival_ref_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_arrival_ref_rollout_tape,
                    selected_indices,
                ),
                episode_idx_tape=reset_episode_idx,
                hotspot_active_idx_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_active_idx_rollout_tape,
                    selected_indices,
                ),
                hotspot_subset_count_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_subset_count_rollout_tape,
                    selected_indices,
                ),
                hotspot_member_mask_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_member_mask_rollout_tape,
                    selected_indices,
                ),
            )

    @staticmethod
    def _gather_runtime_tape_time_env_window(
        value: torch.Tensor | None,
        *,
        source_steps: Sequence[int],
        source_envs: Sequence[int],
        horizon: int,
        field_name: str,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(value):
            return None
        horizon_i = max(int(horizon), 0)
        branch_count = len(tuple(source_steps))
        if horizon_i <= 0 or branch_count <= 0:
            return value.new_empty((0, branch_count) + tuple(value.shape[2:]))
        if int(value.ndim) < 2:
            return value
        step_tuple = tuple(int(step) for step in source_steps)
        env_tuple = tuple(int(env) for env in source_envs)
        if len(step_tuple) != len(env_tuple):
            raise RuntimeError("branch rollout tape gather requires matching source step/env counts.")
        max_step = max(step_tuple) + horizon_i - 1
        if max_step >= int(value.shape[0]):
            raise RuntimeError(
                f"branch rollout tape {field_name!r} is too short for horizon={horizon_i}: "
                f"max_source_step={max(step_tuple)}, tape_steps={int(value.shape[0])}."
            )
        invalid_envs = [env for env in env_tuple if env < 0 or env >= int(value.shape[1])]
        if invalid_envs:
            raise RuntimeError(
                f"branch rollout tape {field_name!r} env index out of bounds: "
                f"indices={invalid_envs!r}, env_count={int(value.shape[1])}."
            )
        base_steps_t = torch.as_tensor(step_tuple, dtype=torch.long, device=value.device).view(1, branch_count)
        offsets_t = torch.arange(horizon_i, dtype=torch.long, device=value.device).view(horizon_i, 1)
        step_index_t = base_steps_t + offsets_t
        env_index_t = torch.as_tensor(env_tuple, dtype=torch.long, device=value.device).view(1, branch_count).expand(
            horizon_i,
            branch_count,
        )
        return value[step_index_t, env_index_t].contiguous()

    @staticmethod
    def _truncate_reset_followup_tape(
        value: torch.Tensor | None,
        *,
        horizon: int,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(value) or int(value.ndim) < 2:
            return value
        return value[:, : max(int(horizon), 0)].contiguous()

    def _copy_branch_random_rollout_tapes(
        self,
        *,
        source_runtime: StructuredGpuRolloutRuntime,
        target_runtime: StructuredGpuRolloutRuntime,
        source_steps: Sequence[int],
        source_envs: Sequence[int],
        horizon: int,
    ) -> None:
        source_random = source_runtime.random
        selected_envs = tuple(int(env) for env in source_envs)
        horizon_i = max(int(horizon), 1)

        def _window(value: torch.Tensor | None, *, steps: int, name: str) -> torch.Tensor | None:
            return self._gather_runtime_tape_time_env_window(
                value,
                source_steps=source_steps,
                source_envs=source_envs,
                horizon=max(int(steps), 0),
                field_name=name,
            )

        target_runtime.write_random_rollout_tapes(
            arrival_tape=_window(source_random.arrival_rollout_tape, steps=horizon_i, name="arrival_rollout_tape"),
            arrival_rate_tape=_window(
                source_random.arrival_rate_rollout_tape,
                steps=horizon_i,
                name="arrival_rate_rollout_tape",
            ),
            hotspot_active_after_tape=_window(
                source_random.hotspot_active_after_rollout_tape,
                steps=horizon_i,
                name="hotspot_active_after_rollout_tape",
            ),
            reset_followup_arrival_tape=self._truncate_reset_followup_tape(
                self._select_runtime_tape_env_rows(
                    source_random.reset_followup_arrival_rollout_tape,
                    selected_envs,
                    env_dim=2,
                ),
                horizon=horizon_i,
            ),
            reset_followup_arrival_rate_tape=self._truncate_reset_followup_tape(
                self._select_runtime_tape_env_rows(
                    source_random.reset_followup_arrival_rate_rollout_tape,
                    selected_envs,
                    env_dim=2,
                ),
                horizon=horizon_i,
            ),
            reset_followup_hotspot_active_after_tape=self._truncate_reset_followup_tape(
                self._select_runtime_tape_env_rows(
                    source_random.reset_followup_hotspot_active_after_rollout_tape,
                    selected_envs,
                    env_dim=2,
                ),
                horizon=horizon_i,
            ),
            hotspot_active_tape=_window(
                source_random.hotspot_active_rollout_tape,
                steps=horizon_i,
                name="hotspot_active_rollout_tape",
            ),
            hotspot_mask_tape=_window(
                source_random.hotspot_mask_rollout_tape,
                steps=horizon_i,
                name="hotspot_mask_rollout_tape",
            ),
            fading_gain_tape=_window(
                source_random.fading_gain_rollout_tape,
                steps=horizon_i + 1,
                name="fading_gain_rollout_tape",
            ),
            doppler_noise_tape=_window(
                source_random.doppler_noise_rollout_tape,
                steps=horizon_i,
                name="doppler_noise_rollout_tape",
            ),
        )
        reset_gu_pos = self._select_runtime_tape_env_rows(source_random.reset_gu_pos_rollout_tape, selected_envs)
        reset_uav_pos = self._select_runtime_tape_env_rows(source_random.reset_uav_pos_rollout_tape, selected_envs)
        reset_uav_vel = self._select_runtime_tape_env_rows(source_random.reset_uav_vel_rollout_tape, selected_envs)
        reset_gu_cluster_centers = self._select_runtime_tape_env_rows(
            source_random.reset_gu_cluster_centers_rollout_tape,
            selected_envs,
        )
        reset_gu_cluster_counts = self._select_runtime_tape_env_rows(
            source_random.reset_gu_cluster_counts_rollout_tape,
            selected_envs,
        )
        reset_arrival_base_scale = self._select_runtime_tape_env_rows(
            source_random.reset_arrival_base_scale_rollout_tape,
            selected_envs,
        )
        reset_deadline_steps = self._select_runtime_tape_env_rows(
            source_random.reset_deadline_steps_rollout_tape,
            selected_envs,
        )
        reset_doppler_residual = self._select_runtime_tape_env_rows(
            source_random.reset_doppler_residual_rollout_tape,
            selected_envs,
        )
        reset_effective_arrival_rate = self._select_runtime_tape_env_rows(
            source_random.reset_effective_arrival_rate_rollout_tape,
            selected_envs,
        )
        reset_episode_idx = self._select_runtime_tape_env_rows(source_random.reset_episode_idx_rollout_tape, selected_envs)
        if all(
            value is not None
            for value in (
                reset_gu_pos,
                reset_uav_pos,
                reset_uav_vel,
                reset_gu_cluster_centers,
                reset_gu_cluster_counts,
                reset_arrival_base_scale,
                reset_deadline_steps,
                reset_doppler_residual,
                reset_effective_arrival_rate,
                reset_episode_idx,
            )
        ):
            target_runtime.write_random_reset_rollout_tapes(
                gu_pos_tape=reset_gu_pos,
                uav_pos_tape=reset_uav_pos,
                uav_vel_tape=reset_uav_vel,
                gu_cluster_centers_tape=reset_gu_cluster_centers,
                gu_cluster_counts_tape=reset_gu_cluster_counts,
                gu_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_gu_queue_rollout_tape, selected_envs),
                uav_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_uav_queue_rollout_tape, selected_envs),
                sat_queue_tape=self._select_runtime_tape_env_rows(source_random.reset_sat_queue_rollout_tape, selected_envs),
                arrival_base_scale_tape=reset_arrival_base_scale,
                deadline_steps_tape=reset_deadline_steps,
                doppler_residual_tape=reset_doppler_residual,
                effective_arrival_rate_tape=reset_effective_arrival_rate,
                arrival_rate_vec_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_arrival_rate_vec_rollout_tape,
                    selected_envs,
                ),
                arrival_ref_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_arrival_ref_rollout_tape,
                    selected_envs,
                ),
                episode_idx_tape=reset_episode_idx,
                hotspot_active_idx_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_active_idx_rollout_tape,
                    selected_envs,
                ),
                hotspot_subset_count_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_subset_count_rollout_tape,
                    selected_envs,
                ),
                hotspot_member_mask_tape=self._select_runtime_tape_env_rows(
                    source_random.reset_hotspot_member_mask_rollout_tape,
                    selected_envs,
                ),
            )
        if torch.is_tensor(target_runtime.random.step_tensor):
            target_runtime.random.step_tensor.zero_()
        if torch.is_tensor(target_runtime.random.reset_count):
            target_runtime.random.reset_count.zero_()
        target_runtime.random.step = 0

    def prepare_native_branch_replay_from_history(
        self,
        *,
        history_rows: Sequence[int],
        horizon: int,
        stage_id: int = 2,
        future_random_mode: str = "copy",
        future_random_seed: int | None = None,
    ) -> None:
        official_runtime = self.native_rollout_runtime
        sub_runtime = self._native_sub_batch_runtime
        sub_tensor_state = self._native_sub_batch_tensor_state
        sub_cfg = self._native_sub_batch_cfg
        sub_rng = self._native_sub_batch_torch_rng
        sub_bound_kernels = self._native_sub_batch_bound_kernels
        sub_slot_state_payloads = self._native_sub_batch_slot_state_payloads
        sub_slot_rngs = self._native_sub_batch_slot_rngs
        if official_runtime is None or sub_runtime is None or sub_tensor_state is None:
            raise RuntimeError("native branch replay requires native_sub_batch_rollout_program() to be built first.")
        if sub_cfg is None or sub_rng is None or sub_bound_kernels is None:
            raise RuntimeError("native branch replay sub-batch workspace is incomplete.")
        stage_id_i = int(stage_id)
        if stage_id_i not in {0, 1, 2}:
            raise RuntimeError(f"native branch replay stage_id must be 0, 1, or 2, got {stage_id_i}.")
        future_random_mode_i = str(future_random_mode or "copy").strip().lower()
        if future_random_mode_i not in {"copy", "resample"}:
            raise RuntimeError(
                "native branch replay future_random_mode must be 'copy' or 'resample', "
                f"got {future_random_mode!r}."
            )
        stage_name = ("accel", "sat", "bw")[stage_id_i]
        history_state = getattr(official_runtime.history, f"{stage_name}_runtime_state", None)
        if history_state is None:
            raise RuntimeError(
                f"native branch replay requires {stage_name}_runtime_state history snapshots. "
                "Set structured_native_history_snapshots_enabled=true, or collect the rollout with a mode that "
                "keeps snapshots automatically (vs_ref / clean teacher / branch probe)."
            )
        history_cache = getattr(official_runtime.history, "bw_runtime_cache", None) if stage_id_i == 2 else None
        if stage_id_i == 2 and history_cache is None:
            raise RuntimeError(
                "native BW clean teacher requires bw_runtime_cache history snapshots. "
                "Set structured_native_history_snapshots_enabled=true before collecting the rollout."
            )
        history_rows_tuple = tuple(int(row) for row in history_rows)
        branch_count = len(history_rows_tuple)
        if branch_count <= 0:
            return
        source_num_envs = int(getattr(official_runtime.history, "num_envs", 0) or self._num_envs)
        if source_num_envs <= 0:
            raise RuntimeError("native branch replay cannot infer source env count.")
        capacity_rows = int(getattr(official_runtime.history, "capacity", 0) or 0) * source_num_envs
        invalid_rows = [row for row in history_rows_tuple if row < 0 or row >= capacity_rows]
        if invalid_rows:
            raise RuntimeError(
                f"native branch replay history row out of bounds: rows={invalid_rows!r}, capacity_rows={capacity_rows}."
            )
        source_envs = tuple(int(row % source_num_envs) for row in history_rows_tuple)
        source_steps = tuple(int(row // source_num_envs) for row in history_rows_tuple)
        source_rows_t = torch.as_tensor(
            history_rows_tuple,
            dtype=torch.long,
            device=getattr(history_state, "uav_pos").device,
        )
        history_stage = getattr(official_runtime.history, f"{stage_name}_runtime_stage", None)
        if stage_id_i == 0:
            target_stage = sub_runtime.main.accel_stage_field_buffers[0]
        elif stage_id_i == 1:
            target_stage = getattr(sub_runtime.main, "sat_stage_fields", None)
        else:
            target_stage = getattr(sub_runtime.main, "bw_stage_fields", None)
        if history_stage is None:
            raise RuntimeError(
                f"native branch replay requires {stage_name}_runtime_stage history snapshots. "
                "Set structured_native_history_snapshots_enabled=true before collecting the rollout."
            )
        if not _is_native_stage_fields(target_stage):
            raise RuntimeError(f"native branch replay sub-batch workspace is missing {stage_name} stage fields.")
        horizon_i = max(int(horizon), 1)
        if future_random_mode_i == "copy":
            self._copy_branch_random_rollout_tapes(
                source_runtime=official_runtime,
                target_runtime=sub_runtime,
                source_steps=source_steps,
                source_envs=source_envs,
                horizon=horizon_i,
            )
        executor = _NativeMainKernelWorkspaceExecutor(
            self,
            runtime=sub_runtime,
            tensor_state=sub_tensor_state,
            cfg=sub_cfg,
            rng=sub_rng,
            bound_kernels=sub_bound_kernels,
            num_envs_override=branch_count,
            slot_state_payloads=sub_slot_state_payloads,
            slot_rngs=sub_slot_rngs,
        )
        with executor._active():
            sub_runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(sub_runtime)
        source_abi = getattr(getattr(official_runtime, "main", None), "native_cuda_abi", None)
        target_abi = getattr(getattr(sub_runtime, "main", None), "native_cuda_abi", None)
        if not isinstance(source_abi, native_cuda.NativeCudaRuntimeABI):
            source_abi = self._build_native_cuda_runtime_abi(official_runtime)
            official_runtime.main.native_cuda_abi = source_abi
        if not isinstance(target_abi, native_cuda.NativeCudaRuntimeABI):
            raise RuntimeError("native branch replay target ABI was not initialized.")
        native_cuda.prepare_branch_replay_from_history(
            source_abi,
            target_abi,
            history_rows=source_rows_t,
            stage_id=stage_id_i,
        )
        if future_random_mode_i == "resample":
            if future_random_seed is not None:
                try:
                    sub_rng.manual_seed(int(future_random_seed))
                except RuntimeError:
                    pass
            with executor._active():
                self._prepare_runtime_rollout_random_tapes(capacity=horizon_i, reset_rows=horizon_i)
                sub_runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(sub_runtime)

    def native_sub_batch_rollout_program(
        self,
        *,
        capacity: int = 3,
        selected_indices: Sequence[int],
        allow_duplicate_indices: bool = False,
    ) -> StructuredGpuNativeRolloutProgram:
        """Build the official native selected-env replay program.

        The selected-env program owns a persistent sub-batch tensor workspace and
        runs the same native actor/env ABI as the full-batch program. The public
        name is intentionally separate from the historical hot-replay helper so
        callers do not rely on partial legacy stage APIs.
        """

        official_runtime = self.native_rollout_runtime
        if official_runtime is None:
            raise RuntimeError("native selected-env sub-batch program requires a persistent native rollout runtime.")
        selected_tuple = tuple(int(index) for index in selected_indices)
        if any(index < 0 or index >= int(self._num_envs) for index in selected_tuple):
            raise ValueError(
                f"native selected-env sub-batch indices must be in [0, {int(self._num_envs)}), "
                f"got {list(selected_tuple)!r}."
            )
        if (not bool(allow_duplicate_indices)) and len(set(selected_tuple)) != len(selected_tuple):
            raise ValueError("native selected-env sub-batch indices must not contain duplicates.")
        replay_envs = len(selected_tuple)
        visible_cfg = getattr(self._cfg, "visible_sats_max", None)
        if visible_cfg is None:
            visible_cfg = getattr(self._cfg, "sats_obs_max", None)
        flow_proxy_base_action_mode_code = self._native_flow_proxy_base_action_mode_code(
            getattr(self._cfg, "bw_flow_proxy_base_action_mode", "executed")
        )
        sub_runtime = self._native_sub_batch_runtime
        sub_tensor_state = self._native_sub_batch_tensor_state
        sub_cfg = self._native_sub_batch_cfg
        sub_rng = self._native_sub_batch_torch_rng
        sub_bound_kernels = self._native_sub_batch_bound_kernels
        sub_selected_indices = self._native_sub_batch_selected_indices
        sub_slot_state_payloads = self._native_sub_batch_slot_state_payloads
        sub_slot_rngs = self._native_sub_batch_slot_rngs
        sub_history = None if sub_runtime is None else getattr(sub_runtime, "rollout_history", getattr(sub_runtime, "history", None))
        sub_main = None if sub_runtime is None else getattr(sub_runtime, "main", None)
        reuse_sub_workspace = bool(
            sub_runtime is not None
            and sub_tensor_state is not None
            and sub_cfg is not None
            and sub_rng is not None
            and isinstance(sub_bound_kernels, dict)
            and sub_history is not None
            and sub_main is not None
            and sub_selected_indices == selected_tuple
            and sub_slot_state_payloads is not None
            and sub_slot_rngs is not None
            and _same_structured_tensor_device(sub_runtime.device, official_runtime.device)
            and int(getattr(sub_tensor_state.uav_pos, "shape", (0,))[0]) == int(replay_envs)
            and int(getattr(sub_history, "capacity", 0) or 0) >= max(int(capacity), 1)
            and int(getattr(sub_history, "num_envs", 0) or 0) == int(replay_envs)
            and int(getattr(sub_main, "num_envs", 0) or 0) == int(replay_envs)
            and int(getattr(sub_history, "cursor", 0) or 0) < int(getattr(sub_history, "capacity", 0) or 0)
        )
        if not bool(getattr(self._cfg, "native_branch_replay_reuse_sub_workspace", True)):
            reuse_sub_workspace = False
        if not reuse_sub_workspace:
            sub_runtime = StructuredGpuRolloutRuntime(device=torch.device(official_runtime.device))
            sub_tensor_state = self._clone_runtime_tensor_state(self._runtime_tensor_state, indices=selected_tuple)
            sub_cfg = copy.copy(self._cfg)
            setattr(sub_cfg, "_structured_kernel_runtime_cache", {})
            sub_rng = self._make_native_torch_rng()
            sub_bound_kernels = {}
            sub_slot_state_payloads = [
                copy.deepcopy(self._slot_state_payloads[int(index)])
                for index in selected_tuple
            ]
            sub_slot_rngs = [
                self._clone_numpy_generator_state(self._slot_rngs[int(index)])
                for index in selected_tuple
            ]
            try:
                sub_rng.set_state(self._native_torch_rng.get_state())
            except RuntimeError:
                pass
            self._native_sub_batch_runtime = sub_runtime
            self._native_sub_batch_tensor_state = sub_tensor_state
            self._native_sub_batch_cfg = sub_cfg
            self._native_sub_batch_torch_rng = sub_rng
            self._native_sub_batch_bound_kernels = sub_bound_kernels
            self._native_sub_batch_selected_indices = selected_tuple
            self._native_sub_batch_slot_state_payloads = sub_slot_state_payloads
            self._native_sub_batch_slot_rngs = sub_slot_rngs
        else:
            # Branch replay advances the sub-batch Python-side slot payloads/RNGs.
            # Reusing the tensor workspace is fine, but the per-slot mutable
            # payloads must be refreshed for every replay program; otherwise a
            # previous branch for the same selected env tuple can leak into the
            # next branch.
            sub_slot_state_payloads = [
                copy.deepcopy(self._slot_state_payloads[int(index)])
                for index in selected_tuple
            ]
            sub_slot_rngs = [
                self._clone_numpy_generator_state(self._slot_rngs[int(index)])
                for index in selected_tuple
            ]
            try:
                sub_rng.set_state(self._native_torch_rng.get_state())
            except RuntimeError:
                pass
            self._native_sub_batch_slot_state_payloads = sub_slot_state_payloads
            self._native_sub_batch_slot_rngs = sub_slot_rngs
        executor = _NativeMainKernelWorkspaceExecutor(
            self,
            runtime=sub_runtime,
            tensor_state=sub_tensor_state,
            cfg=sub_cfg,
            rng=sub_rng,
            bound_kernels=sub_bound_kernels,
            num_envs_override=replay_envs,
            slot_state_payloads=sub_slot_state_payloads,
            slot_rngs=sub_slot_rngs,
        )
        if not reuse_sub_workspace:
            with executor._active():
                self.begin_native_main_kernel_rollout(capacity=max(int(capacity), 1), num_envs=replay_envs)
            self._copy_selected_random_rollout_tapes(
                source_runtime=official_runtime,
                target_runtime=sub_runtime,
                selected_indices=selected_tuple,
            )
        else:
            self._copy_selected_random_rollout_tapes(
                source_runtime=official_runtime,
                target_runtime=sub_runtime,
                selected_indices=selected_tuple,
            )
        sub_runtime.main.flow_proxy_base_action_mode_code = int(flow_proxy_base_action_mode_code)
        self._bind_native_flow_proxy_override_constant(sub_runtime)
        sub_runtime.main.selected_env_mapping = torch.as_tensor(
            selected_tuple,
            dtype=torch.long,
            device=torch.device(sub_runtime.device),
        )
        with executor._active():
            sub_runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(sub_runtime)
        step_program = StructuredGpuNativeRuntimeStepProgram(
            runtime=sub_runtime,
            executor=executor,
            num_envs=replay_envs,
            fixed_visible_sat_width=None if visible_cfg is None else max(int(visible_cfg), 0),
        )
        return StructuredGpuNativeRolloutProgram(step_program=step_program)

    @staticmethod
    def _runtime_native_cuda_source_modes(runtime: StructuredGpuRolloutRuntime) -> tuple[int, int, int]:
        main = runtime.main
        return (
            int(main.accel_actor_source_mode_code),
            int(main.sat_actor_source_mode_code),
            int(main.bw_actor_source_mode_code),
        )

    def _runtime_native_cuda_abi(self, runtime: StructuredGpuRolloutRuntime) -> native_cuda.NativeCudaRuntimeABI:
        abi = runtime.main.native_cuda_abi
        if not isinstance(abi, native_cuda.NativeCudaRuntimeABI):
            raise RuntimeError("native CUDA typed runtime ABI was not initialized.")
        return abi

    def _runtime_begin_horizon(self, *, num_steps: int) -> None:
        runtime = self._native_main_kernel_runtime()
        runtime.activate_rollout_history()
        runtime.history.num_envs = int(runtime.main.num_envs)
        self._bind_native_main_kernel_history_outputs(runtime, runtime.history)
        runtime.begin_horizon(num_steps=int(num_steps), num_envs=int(runtime.main.num_envs))
        runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(runtime)
        abi = self._runtime_native_cuda_abi(runtime)
        accel_mode, sat_mode, bw_mode = self._runtime_native_cuda_source_modes(runtime)
        runtime.main.accel_active_idx = 0
        native_cuda.prepare_initial_accel_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=accel_mode,
            sat_source_mode=sat_mode,
            bw_source_mode=bw_mode,
        )

    def _runtime_step_begin_accel_obs(self):
        runtime = self._native_main_kernel_runtime()
        live_buffers = runtime.main.accel_live_obs_buffers
        if live_buffers is None:
            raise RuntimeError("native runtime step requires accel live obs buffers.")
        active_idx = int(runtime.main.accel_active_idx)
        if active_idx not in {0, 1}:
            raise RuntimeError("native accel active index must be 0 or 1.")
        return live_buffers[active_idx]

    def _runtime_step_publish_sat_obs(
        self,
        *,
        indices: Sequence[int] | None = None,
        max_visible: int | None = None,
    ) -> tuple[Any, int]:
        if indices is not None:
            self._require_native_main_kernel_full_batch(indices, context="accel_to_sat_live")
        del max_visible
        runtime = self._native_main_kernel_runtime()
        abi = self._runtime_native_cuda_abi(runtime)
        active_idx = int(runtime.main.accel_active_idx)
        accel_mode, sat_mode, bw_mode = self._runtime_native_cuda_source_modes(runtime)
        native_cuda.accel_to_sat_live(
            abi,
            slot=int(runtime.history.cursor),
            active_idx=active_idx,
            accel_source_mode=accel_mode,
            sat_source_mode=sat_mode,
            bw_source_mode=bw_mode,
        )
        if runtime.main.live_sat_obs is None:
            raise RuntimeError("native runtime step requires SAT live obs buffer.")
        return runtime.main.live_sat_obs, int(runtime.main.sat_max_select)

    def _copy_current_bw_runtime_state_to_history(
        self,
        *,
        runtime: StructuredGpuRolloutRuntime,
        slot: int,
    ) -> None:
        history_state = getattr(runtime.history, "bw_runtime_state", None)
        if history_state is None:
            return
        num_envs = int(getattr(runtime.main, "num_envs", 0) or 0)
        if num_envs <= 0:
            return
        slot_i = int(slot)
        if slot_i < 0 or slot_i >= int(getattr(runtime.history, "capacity", 0) or 0):
            raise RuntimeError("native BW runtime snapshot requires an in-range history slot.")
        source_rows_t = getattr(runtime.main, "candidate_env_ids", None)
        if not torch.is_tensor(source_rows_t):
            raise RuntimeError("native BW runtime snapshot requires preallocated runtime row indices.")
        source_rows_t = source_rows_t.reshape(-1)
        if int(source_rows_t.numel()) < num_envs:
            raise RuntimeError(
                "native BW runtime snapshot row index buffer is smaller than the live runtime batch "
                f"({int(source_rows_t.numel())} < {num_envs})."
            )
        source_rows_t = source_rows_t[:num_envs]
        target_rows_t = source_rows_t + int(slot_i) * int(num_envs)
        _copy_tensor_dataclass_rows_(
            target_state=history_state,
            source_state=self._runtime_tensor_state,
            target_rows_t=target_rows_t,
            source_rows_t=source_rows_t,
        )
        history_cache = getattr(runtime.history, "bw_runtime_cache", None)
        if history_cache is not None:
            _copy_tensor_dataclass_rows_(
                target_state=history_cache,
                source_state=StructuredGpuBwRuntimeCacheBuffers(
                    candidate_indices=runtime.main.bw_candidate_indices,
                    valid_mask=runtime.main.bw_valid_mask,
                    assoc=runtime.main.bw_assoc,
                    prev_association=runtime.main.bw_prev_association,
                    candidate_mask=runtime.main.bw_candidate_mask,
                    access_gain_matrix=runtime.main.bw_access_gain_matrix,
                    sat_selection_matrix=runtime.main.bw_sat_selection_matrix,
                    active_sat_ids=runtime.main.bw_active_sat_ids,
                    gain_active=runtime.main.bw_gain_active,
                    nu_eff_active=runtime.main.bw_nu_eff_active,
                    valid_flag_active=runtime.main.bw_valid_flag_active,
                    sat_pos=runtime.main.bw_sat_pos,
                    uav_ecef=runtime.main.bw_uav_ecef,
                    uav_pos=runtime.main.bw_uav_pos,
                    uav_vel=runtime.main.bw_uav_vel,
                    gu_pos=runtime.main.bw_gu_pos,
                ),
                target_rows_t=target_rows_t,
                source_rows_t=source_rows_t,
            )
        history_stage = getattr(runtime.history, "bw_runtime_stage", None)
        bw_stage_fields = getattr(runtime.main, "bw_stage_fields", None)
        if history_stage is not None and _is_native_stage_fields(bw_stage_fields):
            _copy_tensor_dataclass_rows_(
                target_state=history_stage,
                source_state=bw_stage_fields,
                target_rows_t=target_rows_t,
                source_rows_t=source_rows_t,
            )

    def _runtime_step_publish_bw_obs(
        self,
        *,
        indices: Sequence[int] | None = None,
        max_visible: int | None = None,
    ) -> Any:
        if indices is not None:
            self._require_native_main_kernel_full_batch(indices, context="sat_to_bw_live")
        del max_visible
        runtime = self._native_main_kernel_runtime()
        abi = self._runtime_native_cuda_abi(runtime)
        active_idx = int(runtime.main.accel_active_idx)
        accel_mode, sat_mode, bw_mode = self._runtime_native_cuda_source_modes(runtime)
        native_cuda.sat_to_bw_live(
            abi,
            slot=int(runtime.history.cursor),
            active_idx=active_idx,
            accel_source_mode=accel_mode,
            sat_source_mode=sat_mode,
            bw_source_mode=bw_mode,
        )
        # The BW runtime snapshot is written inside the native sat->bw kernel.
        # Keeping the legacy Python copy here launches per-field index ops.
        if runtime.main.live_bw_obs is None:
            raise RuntimeError("native runtime step requires BW live obs buffer.")
        return runtime.main.live_bw_obs

    def _runtime_step_apply_bw_macro_live(
        self,
        *,
        indices: Sequence[int] | None = None,
        max_visible: int | None = None,
        rollout_tail: bool = False,
    ) -> None:
        if indices is not None:
            self._require_native_main_kernel_full_batch(indices, context="apply_bw_macro_live")
        del max_visible
        runtime = self._native_main_kernel_runtime()
        slot = int(runtime.history.cursor)
        if slot < 0 or slot >= int(runtime.history.capacity):
            raise RuntimeError("native CUDA BW macro apply requires an in-range history slot.")
        abi = self._runtime_native_cuda_abi(runtime)
        active_idx = int(runtime.main.accel_active_idx)
        accel_mode, sat_mode, bw_mode = self._runtime_native_cuda_source_modes(runtime)
        native_cuda.apply_bw_macro_live(
            abi,
            slot=slot,
            active_idx=active_idx,
            rollout_tail=bool(rollout_tail),
            accel_source_mode=accel_mode,
            sat_source_mode=sat_mode,
            bw_source_mode=bw_mode,
        )

    def _runtime_step_finish_bw(
        self,
        *,
        indices: Sequence[int] | None = None,
        max_visible: int | None = None,
        rollout_tail: bool = False,
    ) -> StructuredBatchStepResult:
        if indices is not None:
            self._require_native_main_kernel_full_batch(indices, context="finish_commit_prepare_live")
        del max_visible
        runtime = self._native_main_kernel_runtime()
        slot = int(runtime.history.cursor)
        if slot < 0 or slot >= int(runtime.history.capacity):
            raise RuntimeError("native CUDA finish requires an in-range history slot.")
        if slot >= len(runtime.result.step_result_views):
            raise RuntimeError("native CUDA finish requires a prebound step result view.")
        abi = self._runtime_native_cuda_abi(runtime)
        active_idx = int(runtime.main.accel_active_idx)
        accel_mode, sat_mode, bw_mode = self._runtime_native_cuda_source_modes(runtime)
        if (
            bool(self._native_main_kernel_typed_domains().access_rate.enable_bw_action)
            and int(bw_mode) == 0
            and bool(getattr(self._cfg, "structured_native_validate_full_g_bw_action_python", False))
        ):
            bw_stage_fields = runtime.main.bw_stage_fields
            if not _is_native_stage_fields(bw_stage_fields):
                raise RuntimeError("native CUDA finish requires BW stage fields for full-G BW action validation.")
            _validate_full_g_bw_action_tensor(
                bw_action_matrix_t=runtime.main.live_bw_action,
                assoc_t=bw_stage_fields.assoc,
                num_uav=int(runtime.main.num_uav),
                context="native CUDA finish",
            )
        native_cuda.finish_commit_prepare_live(
            abi,
            slot=slot,
            active_idx=active_idx,
            rollout_tail=bool(rollout_tail),
            accel_source_mode=accel_mode,
            sat_source_mode=sat_mode,
            bw_source_mode=bw_mode,
        )
        runtime.main.step_result_view = runtime.result.step_result_views[slot]
        runtime.history.cursor = slot + 1
        runtime.random.step = int(runtime.random.step) + 1
        if not bool(rollout_tail):
            runtime.main.accel_active_idx = 1 - active_idx
        return runtime.main.step_result_view

    def _native_main_kernel_direct_required(self, selected_indices: Sequence[int]) -> bool:
        return bool(self._native_runtime_full_batch_tensor_path(selected_indices, require_cuda=False))

    def _native_main_kernel_strict_cuda_contract_enabled(self) -> bool:
        if self._tensor_device is None:
            return False
        return bool(
            torch.device(self._tensor_device).type == "cuda"
            and bool(getattr(self._cfg, "structured_native_main_kernel_require_compiled_segments", False))
        )

    def _native_main_kernel_live_rollout_active(self) -> bool:
        runtime = self.native_rollout_runtime
        if runtime is None:
            return False
        main = runtime.main
        return bool(int(getattr(main, "num_envs", 0) or 0) > 0)

    def _reject_native_main_kernel_legacy_adapter(self, api_name: str) -> None:
        if not self._native_main_kernel_strict_cuda_contract_enabled() or not self._native_main_kernel_live_rollout_active():
            return
        raise RuntimeError(
            f"official CUDA native main-kernel does not expose legacy adapter API {api_name}(). "
            "Use tensor batch/result ABI or an outer reference/debug adapter; NumPy/list/single-env "
            "materialization must not enter the live core."
        )

    def _require_native_main_kernel_full_batch(self, selected_indices: Sequence[int], *, context: str) -> None:
        if not self._native_main_kernel_strict_cuda_contract_enabled():
            return
        selected_tuple = tuple(int(index) for index in selected_indices)
        full_batch_selected = bool(
            selected_tuple
            and len(selected_tuple) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected_tuple))
        )
        if not full_batch_selected:
            raise RuntimeError(
                f"official CUDA native main-kernel {context} requires full-batch env execution; "
                "partial-batch CUDA execution must use a separate persistent sub-batch kernel instead of "
                "falling back to compat stage/object paths."
            )

    def _require_native_main_kernel_direct_path(self, selected_indices: Sequence[int], *, context: str) -> None:
        self._require_native_main_kernel_full_batch(selected_indices, context=context)
        if not self._native_main_kernel_strict_cuda_contract_enabled():
            return
        if self._native_main_kernel_direct_required(selected_indices):
            return
        raise RuntimeError(
            f"official CUDA native main-kernel {context} requires the direct full-batch tensor path; "
            "direct_required=False cannot fall back to compat stage/object execution. Check tensor device, "
            "native tensor runtime, and full-batch selection."
        )

    def _native_runtime_full_batch_tensor_path(
        self,
        selected_indices: Sequence[int],
        *,
        require_cuda: bool = False,
    ) -> bool:
        selected_tuple = tuple(int(index) for index in selected_indices)
        cfg = self._cfg
        if self._tensor_device is None:
            return False
        tensor_device = torch.device(self._tensor_device)
        if tensor_device.type not in {"cpu", "cuda"}:
            return False
        if bool(require_cuda) and tensor_device.type != "cuda":
            return False
        return bool(
            selected_tuple
            and len(selected_tuple) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected_tuple))
        )

    def _native_fading_unity_tensor(self, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        target_device = torch.device(device)
        if (
            self._native_fading_unity_buffer is None
            or tuple(self._native_fading_unity_buffer.shape) != tuple(shape)
            or self._native_fading_unity_buffer.dtype != torch.float32
            or not _same_structured_tensor_device(self._native_fading_unity_buffer.device, target_device)
        ):
            self._native_fading_unity_buffer = torch.empty(shape, dtype=torch.float32, device=target_device)
            self._native_fading_unity_buffer.fill_(1.0)
        return self._native_fading_unity_buffer

    def _native_stage_id_tensor(self, stage_id: int, batch_size: int, *, device: torch.device) -> torch.Tensor:
        target_device = torch.device(device)
        key = (
            int(stage_id),
            int(batch_size),
            str(target_device.type),
            int(target_device.index) if target_device.index is not None else None,
        )
        value = self._native_stage_id_buffers.get(key)
        if value is None or value.device != target_device or value.dtype != torch.long:
            value = torch.empty((int(batch_size),), dtype=torch.long, device=target_device)
            value.fill_(int(stage_id))
            self._native_stage_id_buffers[key] = value
        return value

    def _native_scalar_float_tensor(
        self,
        name: str,
        value: float,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        key = (
            str(name),
            int(batch_size),
            str(target_device.type),
            int(target_device.index) if target_device.index is not None else None,
        )
        tensor = self._native_scalar_float_buffers.get(key)
        if tensor is None or tensor.device != target_device or tensor.dtype != torch.float32:
            tensor = torch.empty((int(batch_size),), dtype=torch.float32, device=target_device)
            self._native_scalar_float_buffers[key] = tensor
        tensor.fill_(float(value))
        return tensor

    def _native_zero_float_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        shape_t = tuple(int(dim) for dim in shape)
        key = (
            str(name),
            shape_t,
            str(target_device.type),
            int(target_device.index) if target_device.index is not None else None,
        )
        tensor = self._native_zero_float_buffers.get(key)
        if tensor is None or tensor.device != target_device or tensor.dtype != torch.float32:
            tensor = torch.empty(shape_t, dtype=torch.float32, device=target_device)
            self._native_zero_float_buffers[key] = tensor
        tensor.zero_()
        return tensor

    def _native_override_input_tensor(
        self,
        name: str,
        value: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        value_t = value.to(device=target_device, dtype=dtype)
        shape_t = tuple(int(dim) for dim in value_t.shape)
        key = (
            str(name),
            shape_t,
            dtype,
            str(target_device.type),
            int(target_device.index) if target_device.index is not None else None,
        )
        buffer_t = self._native_override_input_buffers.get(key)
        if (
            buffer_t is None
            or tuple(buffer_t.shape) != shape_t
            or buffer_t.dtype != dtype
            or not _same_structured_tensor_device(buffer_t.device, target_device)
        ):
            buffer_t = torch.empty(shape_t, dtype=dtype, device=target_device)
            self._native_override_input_buffers[key] = buffer_t
        buffer_t.copy_(value_t)
        return buffer_t

    def _rician_power_gain_tensor(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        step_offset: int = 0,
        ) -> torch.Tensor:
        cfg = self._cfg
        target_device = torch.device(device)
        if not (bool(getattr(cfg, "fading_enabled", False)) and _access_fading_mode_code_from_cfg(cfg) == 2):
            return self._native_fading_unity_tensor(shape, device=target_device)
        def _same_device(value: torch.Tensor | None) -> bool:
            if value is None:
                return False
            value_device = torch.device(value.device)
            if value_device == target_device:
                return True
            return (
                value_device.type == target_device.type == "cuda"
                and (target_device.index is None or value_device.index in {None, target_device.index})
            )
        k_value = channel.rician_k_linear_from_config(cfg)
        s_value = math.sqrt(k_value / (k_value + 1.0))
        sigma_value = math.sqrt(1.0 / (2.0 * (k_value + 1.0)))
        runtime = getattr(self, "_native_rollout_runtime", None)
        rollout_tape = None if runtime is None else runtime.random.fading_gain_rollout_tape
        if (
            torch.is_tensor(rollout_tape)
            and tuple(rollout_tape.shape[1:]) == tuple(shape)
            and int(runtime.random.step) + int(step_offset) < int(rollout_tape.shape[0])
            and (
                torch.device(rollout_tape.device) == target_device
                or (
                    torch.device(rollout_tape.device).type == target_device.type == "cuda"
                    and (target_device.index is None or torch.device(rollout_tape.device).index in {None, target_device.index})
                )
            )
        ):
            gain_t = rollout_tape[int(runtime.random.step) + int(step_offset)]
            random_buffers = runtime.write_random_fading_tape(gain_t)
            if torch.is_tensor(random_buffers.fading_gain):
                return random_buffers.fading_gain
            return gain_t
        noise_shape = (2, *shape)
        if (
            self._native_rician_noise_buffer is None
            or tuple(self._native_rician_noise_buffer.shape) != tuple(noise_shape)
            or not _same_device(self._native_rician_noise_buffer)
            or self._native_rician_noise_buffer.dtype != torch.float32
        ):
            self._native_rician_noise_buffer = torch.empty(noise_shape, dtype=torch.float32, device=target_device)
        if (
            self._native_rician_gain_buffer is None
            or tuple(self._native_rician_gain_buffer.shape) != tuple(shape)
            or not _same_device(self._native_rician_gain_buffer)
            or self._native_rician_gain_buffer.dtype != torch.float32
        ):
            self._native_rician_gain_buffer = torch.empty(shape, dtype=torch.float32, device=target_device)
        noise = self._native_rician_noise_buffer
        gain = self._native_rician_gain_buffer
        noise.normal_(generator=self._native_torch_rng)
        real = noise[0]
        imag = noise[1]
        real = real.mul(float(sigma_value)).add(float(s_value))
        imag = imag.mul(float(sigma_value))
        gain.copy_(real).mul_(real).addcmul_(imag, imag)
        if runtime is not None:
            runtime_device = torch.device(runtime.device)
            if target_device == runtime_device or (
                target_device.type == runtime_device.type == "cuda"
                and (runtime_device.index is None or target_device.index in {None, runtime_device.index})
            ):
                runtime.write_random_fading_tape(gain)
        return gain

    def _native_doppler_noise_tensor(self, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        target_device = torch.device(device)
        runtime = getattr(self, "_native_rollout_runtime", None)
        rollout_tape = None if runtime is None else runtime.random.doppler_noise_rollout_tape
        if (
            torch.is_tensor(rollout_tape)
            and tuple(rollout_tape.shape[1:]) == tuple(shape)
            and int(runtime.random.step) < int(rollout_tape.shape[0])
            and (
                torch.device(rollout_tape.device) == target_device
                or (
                    torch.device(rollout_tape.device).type == target_device.type == "cuda"
                    and (target_device.index is None or torch.device(rollout_tape.device).index in {None, target_device.index})
                )
            )
        ):
            noise_t = rollout_tape[int(runtime.random.step)]
            random_buffers = runtime.write_random_doppler_noise_tape(noise_t)
            if torch.is_tensor(random_buffers.doppler_noise):
                return random_buffers.doppler_noise
            return noise_t
        if (
            self._native_doppler_noise_buffer is None
            or tuple(self._native_doppler_noise_buffer.shape) != tuple(shape)
            or self._native_doppler_noise_buffer.dtype != torch.float32
            or not (
                torch.device(self._native_doppler_noise_buffer.device) == target_device
                or (
                    torch.device(self._native_doppler_noise_buffer.device).type == target_device.type == "cuda"
                    and (target_device.index is None or self._native_doppler_noise_buffer.device.index in {None, target_device.index})
                )
            )
        ):
            self._native_doppler_noise_buffer = torch.empty(shape, dtype=torch.float32, device=target_device)
        self._native_doppler_noise_buffer.normal_(generator=self._native_torch_rng)
        if runtime is not None:
            runtime_device = torch.device(runtime.device)
            noise_device = torch.device(self._native_doppler_noise_buffer.device)
            if noise_device == runtime_device or (
                noise_device.type == runtime_device.type == "cuda"
                and (runtime_device.index is None or noise_device.index in {None, runtime_device.index})
            ):
                runtime.write_random_doppler_noise_tape(self._native_doppler_noise_buffer)
        return self._native_doppler_noise_buffer

    @property
    def native_rollout_runtime(self) -> StructuredGpuRolloutRuntime | None:
        device = self._tensor_device
        if device is None:
            return None
        device = torch.device(device)
        if self._native_rollout_runtime is None or self._native_rollout_runtime.device != device:
            self._native_rollout_runtime = StructuredGpuRolloutRuntime(device=device)
        return self._native_rollout_runtime

    def set_tensor_device(self, device: torch.device | str | None) -> None:
        target_device = None if device is None else torch.device(device)
        current_device = self._tensor_device
        if _same_structured_tensor_device(current_device, target_device):
            runtime_device = torch.device("cpu") if target_device is None else target_device
            if _same_structured_tensor_device(self._runtime_tensor_state.uav_pos.device, runtime_device):
                return
        self._tensor_device = target_device
        self._native_torch_rng = self._make_native_torch_rng()
        if self._tensor_device is None:
            self._native_rollout_runtime = None
            self._native_hot_replay_runtime = None
            self._native_hot_replay_tensor_state = None
            self._native_hot_replay_cfg = None
            self._native_hot_replay_torch_rng = None
            self._native_hot_replay_bound_kernels = None
            self._native_sub_batch_runtime = None
            self._native_sub_batch_tensor_state = None
            self._native_sub_batch_cfg = None
            self._native_sub_batch_torch_rng = None
            self._native_sub_batch_bound_kernels = None
        elif self._native_rollout_runtime is not None and not _same_structured_tensor_device(
            self._native_rollout_runtime.device,
            self._tensor_device,
        ):
            self._native_rollout_runtime = None
            self._native_hot_replay_runtime = None
            self._native_hot_replay_tensor_state = None
            self._native_hot_replay_cfg = None
            self._native_hot_replay_torch_rng = None
            self._native_hot_replay_bound_kernels = None
            self._native_sub_batch_runtime = None
            self._native_sub_batch_tensor_state = None
            self._native_sub_batch_cfg = None
            self._native_sub_batch_torch_rng = None
            self._native_sub_batch_bound_kernels = None
        self._refresh_orbit_lookup_tensors()
        previous_state = self._runtime_tensor_state
        self._runtime_tensor_state = self._clone_runtime_tensor_state(previous_state)

    def _slot_indices_for_drivers(self, drivers: Sequence[StructuredControlDriver]) -> list[int]:
        slots: list[int] = []
        for driver in drivers:
            slot = getattr(driver, "_structured_batch_slot_index", None)
            if slot is None:
                driver_id = id(driver)
                if driver_id not in self._driver_slots:
                    for candidate_slot in range(self._num_envs):
                        candidate = self._drivers[candidate_slot]
                        self._driver_slots[id(candidate)] = candidate_slot
                    if driver_id not in self._driver_slots:
                        raise KeyError("Driver is not registered with this batch core.")
                slot = self._driver_slots[driver_id]
            slots.append(int(slot))
        return slots

    def _build_orbit_model(self) -> WalkerDeltaOrbitModel | None:
        if self._num_envs <= 0:
            return None
        cfg = self._cfg
        return WalkerDeltaOrbitModel(
            cfg.num_sat,
            cfg.r_earth,
            cfg.sat_height,
            num_planes=cfg.walker_num_planes,
            inclination_deg=cfg.walker_inclination_deg,
            phase_factor=cfg.walker_phase_factor,
            earth_rotation_rate=cfg.earth_rotation_rate,
        )

    def _build_orbit_lookup_tables(self) -> tuple[np.ndarray, np.ndarray]:
        if self._num_envs <= 0 or self._orbit_model is None:
            return (
                np.zeros((0, 0, 3), dtype=np.float32),
                np.zeros((0, 0, 3), dtype=np.float32),
            )
        cfg = self._cfg
        num_steps = max(int(cfg.T_steps), 0) + 1
        pos_table = np.zeros((num_steps, int(cfg.num_sat), 3), dtype=np.float32)
        vel_table = np.zeros((num_steps, int(cfg.num_sat), 3), dtype=np.float32)
        for t_idx in range(num_steps):
            pos_t, vel_t = self._orbit_model.get_states(float(t_idx) * float(cfg.tau0))
            pos_table[t_idx] = np.asarray(pos_t, dtype=np.float32)
            vel_table[t_idx] = np.asarray(vel_t, dtype=np.float32)
        return pos_table, vel_table

    def _refresh_orbit_lookup_tensors(self) -> None:
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        self._orbit_pos_table_tensor = torch.as_tensor(self._orbit_pos_table, dtype=torch.float32, device=kernel_device)
        self._orbit_vel_table_tensor = torch.as_tensor(self._orbit_vel_table, dtype=torch.float32, device=kernel_device)

    def _orbit_states_from_time_indices(self, t_values: Sequence[int] | np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self._num_envs <= 0:
            return (
                np.zeros((0, 0, 3), dtype=np.float32),
                np.zeros((0, 0, 3), dtype=np.float32),
            )
        cfg = self._cfg
        t_arr = _compat_numpy_array(t_values, dtype=np.int64).reshape(-1)
        sat_shape = (t_arr.shape[0], int(cfg.num_sat), 3)
        if t_arr.size == 0:
            return np.zeros(sat_shape, dtype=np.float32), np.zeros(sat_shape, dtype=np.float32)
        pos = np.empty(sat_shape, dtype=np.float32)
        vel = np.empty(sat_shape, dtype=np.float32)
        valid = (t_arr >= 0) & (t_arr < self._orbit_pos_table.shape[0])
        if bool(np.any(valid)):
            valid_idx = t_arr[valid].astype(np.int64, copy=False)
            pos[valid] = np.asarray(self._orbit_pos_table[valid_idx], dtype=np.float32)
            vel[valid] = np.asarray(self._orbit_vel_table[valid_idx], dtype=np.float32)
        if bool(np.any(~valid)):
            if self._orbit_model is None:
                raise RuntimeError("Orbit model is unavailable for out-of-range orbit lookup.")
            invalid_slots = np.nonzero(~valid)[0]
            for slot in invalid_slots.tolist():
                pos_t, vel_t = self._orbit_model.get_states(float(t_arr[slot]) * float(cfg.tau0))
                pos[slot] = np.asarray(pos_t, dtype=np.float32)
                vel[slot] = np.asarray(vel_t, dtype=np.float32)
        return pos, vel

    def _orbit_states_from_time_indices_tensor(
        self,
        t_values: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_device = torch.device("cpu") if device is None else torch.device(device)
        if self._num_envs <= 0:
            zero = torch.zeros((0, 0, 3), dtype=torch.float32, device=kernel_device)
            return zero, zero.clone()
        if (
            torch.is_tensor(t_values)
            and self._orbit_pos_table_tensor is not None
            and self._orbit_vel_table_tensor is not None
        ):
            index_t = t_values.reshape(-1).to(device=self._orbit_pos_table_tensor.device, dtype=torch.long)
            pos_t = self._orbit_pos_table_tensor.index_select(0, index_t)
            vel_t = self._orbit_vel_table_tensor.index_select(0, index_t)
            if pos_t.device != kernel_device:
                pos_t = pos_t.to(device=kernel_device)
                vel_t = vel_t.to(device=kernel_device)
            return pos_t, vel_t
        t_arr = _compat_numpy_array(t_values, dtype=np.int64).reshape(-1)
        if (
            t_arr.size > 0
            and bool(np.all((t_arr >= 0) & (t_arr < self._orbit_pos_table.shape[0])))
            and self._orbit_pos_table_tensor is not None
            and self._orbit_vel_table_tensor is not None
        ):
            index_t = torch.as_tensor(t_arr, dtype=torch.long, device=self._orbit_pos_table_tensor.device)
            pos_t = self._orbit_pos_table_tensor.index_select(0, index_t)
            vel_t = self._orbit_vel_table_tensor.index_select(0, index_t)
            if pos_t.device != kernel_device:
                pos_t = pos_t.to(device=kernel_device)
                vel_t = vel_t.to(device=kernel_device)
            return pos_t, vel_t
        pos, vel = self._orbit_states_from_time_indices(t_arr)
        return (
            _as_kernel_tensor(pos, dtype=torch.float32, device=kernel_device),
            _as_kernel_tensor(vel, dtype=torch.float32, device=kernel_device),
        )

    def _sync_runtime_orbit_state_from_t(
        self,
        indices: Sequence[int] | None = None,
        *,
        t_values: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        selected = self.resolve_indices(indices)
        if not selected:
            return
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        full_batch_selected = bool(
            len(selected) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected))
        )
        selected_tensor = None
        if not full_batch_selected:
            selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=kernel_device)
        if t_values is None:
            if full_batch_selected:
                t_values = self._runtime_tensor_state.t
            else:
                if selected_tensor is None:
                    raise RuntimeError("native partial-batch orbit selection tensor was not initialized.")
                t_values = self._runtime_tensor_state.t.index_select(
                    0,
                    selected_tensor.to(device=self._runtime_tensor_state.t.device),
                )
        if torch.is_tensor(t_values):
            t_count = int(t_values.reshape(-1).numel())
        else:
            t_count = int(np.asarray(t_values).reshape(-1).shape[0])
        if t_count != len(selected):
            raise ValueError(f"Expected {len(selected)} time indices, got {t_count}.")
        sat_pos_t, sat_vel_t = self._orbit_states_from_time_indices_tensor(t_values, device=kernel_device)
        if full_batch_selected:
            self._runtime_tensor_state.sat_pos.copy_(sat_pos_t.to(device=self._runtime_tensor_state.sat_pos.device))
            self._runtime_tensor_state.sat_vel.copy_(sat_vel_t.to(device=self._runtime_tensor_state.sat_vel.device))
        else:
            if selected_tensor is None:
                raise RuntimeError("native partial-batch orbit selection tensor was not initialized.")
            self._runtime_tensor_state.sat_pos[selected_tensor] = sat_pos_t
            self._runtime_tensor_state.sat_vel[selected_tensor] = sat_vel_t

    def _advance_native_doppler_residual_batch(self, indices: Sequence[int]) -> None:
        selected = self.resolve_indices(indices)
        if not selected:
            return
        cfg = self._cfg
        full_batch_selected = bool(
            len(selected) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected))
        )
        cap = 0.0
        if bool(getattr(cfg, "doppler_precomp_mode", "none") in {"residual_hz", "residual_ppm"}):
            cap = _doppler_residual_cap_hz_from_cfg(cfg)
        if cap <= 0.0:
            if full_batch_selected:
                self._runtime_tensor_state.doppler_residual.zero_()
                return
            zero_t = self._native_zero_float_tensor(
                "doppler_residual_zero",
                (len(selected), int(cfg.num_uav), int(cfg.num_sat)),
                device=self._runtime_tensor_state.doppler_residual.device,
            )
            selected_tensor = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=self._runtime_tensor_state.doppler_residual.device,
            )
            self._runtime_tensor_state.doppler_residual[selected_tensor] = zero_t
            return
        rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
        sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
        selected_tensor = None
        if not full_batch_selected:
            selected_tensor = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=self._runtime_tensor_state.doppler_residual.device,
            )
        prev_t = (
            self._runtime_tensor_state.doppler_residual
            if full_batch_selected
            else self._runtime_tensor_state.doppler_residual.index_select(0, selected_tensor)
        ).to(dtype=torch.float32)
        if sigma <= 0.0:
            next_t = prev_t.clamp(min=-float(cap), max=float(cap))
        else:
            noise_t = self._native_doppler_noise_tensor(
                tuple(prev_t.shape),
                device=prev_t.device,
            )
            next_t = (prev_t * float(rho) + noise_t * float(sigma)).clamp(
                min=-float(cap),
                max=float(cap),
            )
        next_t = next_t.to(device=self._runtime_tensor_state.doppler_residual.device, dtype=torch.float32)
        if full_batch_selected:
            self._runtime_tensor_state.doppler_residual.copy_(next_t)
        else:
            if selected_tensor is None:
                raise RuntimeError("native partial-batch doppler selection tensor was not initialized.")
            self._runtime_tensor_state.doppler_residual[selected_tensor] = next_t

    @staticmethod
    def _to_output(value: np.ndarray | torch.Tensor, *, device: torch.device | None, dtype: torch.dtype | None = None):
        if torch.is_tensor(value):
            if device is None:
                return _compat_numpy_array(value, dtype=None)
            return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
        if device is None:
            return value
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _clone_numpy_generator_state(rng: np.random.Generator) -> np.random.Generator:
        cloned = np.random.default_rng()
        cloned.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        return cloned

    def _clone_runtime_tensor_state(
        self,
        source: StructuredBatchRuntimeTensorState,
        *,
        num_envs: int | None = None,
        indices: Sequence[int] | None = None,
    ) -> StructuredBatchRuntimeTensorState:
        if indices is not None:
            selected = tuple(int(index) for index in indices)
            target = self._allocate_runtime_tensor_state(num_envs=len(selected))
        else:
            selected = None
            target = self._allocate_runtime_tensor_state(num_envs=num_envs)
        selected_cache: dict[torch.device, torch.Tensor] = {}
        for field_name in StructuredBatchRuntimeTensorState.__dataclass_fields__:
            target_tensor = getattr(target, field_name)
            source_tensor = getattr(source, field_name)
            rows = int(target_tensor.shape[0]) if target_tensor.ndim > 0 else 0
            if selected is not None and source_tensor.ndim > 0 and rows == len(selected):
                device = source_tensor.device
                selected_tensor = selected_cache.get(device)
                if selected_tensor is None:
                    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
                    selected_cache[device] = selected_tensor
                source_slice = source_tensor.index_select(0, selected_tensor)
            else:
                source_slice = source_tensor[:rows] if source_tensor.ndim > 0 else source_tensor
            target_tensor.copy_(source_slice.to(device=target_tensor.device, dtype=target_tensor.dtype))
        return target

    def _allocate_runtime_tensor_state(self, *, num_envs: int | None = None) -> StructuredBatchRuntimeTensorState:
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        batch_size = int(self._num_envs if num_envs is None else num_envs)
        if batch_size <= 0:
            zero_f = torch.zeros((0,), dtype=torch.float32, device=kernel_device)
            zero_i32 = torch.zeros((0,), dtype=torch.int32, device=kernel_device)
            zero_i64 = torch.zeros((0,), dtype=torch.int64, device=kernel_device)
            return StructuredBatchRuntimeTensorState(
                uav_pos=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                uav_vel=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                uav_energy=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                uav_queue=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                gu_pos=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                gu_cluster_centers=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                gu_cluster_counts=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                gu_queue=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                sat_queue=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                sat_pos=torch.zeros((0, 0, 3), dtype=torch.float32, device=kernel_device),
                sat_vel=torch.zeros((0, 0, 3), dtype=torch.float32, device=kernel_device),
                prev_association=torch.zeros((0, 0), dtype=torch.int32, device=kernel_device),
                last_association=torch.zeros((0, 0), dtype=torch.int32, device=kernel_device),
                last_sat_selection_matrix=torch.zeros((0, 0, 0), dtype=torch.int64, device=kernel_device),
                last_sat_connection_counts=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                arrival_ref_bits_per_step=zero_f.clone(),
                effective_task_arrival_rate=zero_f.clone(),
                arrival_base_scale=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                hotspot_active_idx=zero_i32.clone(),
                hotspot_subset_count=zero_i32.clone(),
                hotspot_member_mask=torch.zeros((0, 0, 0), dtype=torch.float32, device=kernel_device),
                traffic_reset_step=zero_i32.clone(),
                traffic_reset_ordinal=zero_i32.clone(),
                episode_idx=zero_i32.clone(),
                gu_workload_ema=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                uav_workload_ema=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                sat_workload_ema=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_arrival_rate_vec=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                gu_deadline_steps=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_arrival=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_outflow=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                gu_drop=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                uav_drop=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                sat_drop=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_access_interference_by_uav=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_bw_fraction_by_uav_gu=torch.zeros((0, 0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_to_uav_inflow_by_uav=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_uav_to_sat_outflow_matrix=torch.zeros((0, 0, 0), dtype=torch.float32, device=kernel_device),
                last_selected_mask_by_uav_sat=torch.zeros((0, 0, 0), dtype=torch.float32, device=kernel_device),
                last_sat_processed=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_urgency_risk=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_downstream_pressure=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_service_gap_risk=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_deadline_slack=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_deadline_risk=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_service_gap=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_gu_deadline_age=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                last_exec_accel=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                last_policy_accel=torch.zeros((0, 0, 2), dtype=torch.float32, device=kernel_device),
                avoidance_eta_eff=zero_f.clone(),
                last_avoidance_eta_exec=zero_f.clone(),
                doppler_residual=torch.zeros((0, 0, 0), dtype=torch.float32, device=kernel_device),
                prev_queue_sum_gu=zero_f.clone(),
                prev_queue_sum_uav=zero_f.clone(),
                prev_queue_sum_sat=zero_f.clone(),
                prev_q_norm_active=zero_f.clone(),
                prev_gu_queue_vec=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                prev_uav_queue_vec=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                prev_sat_queue_vec=torch.zeros((0, 0), dtype=torch.float32, device=kernel_device),
                t=zero_i32.clone(),
                global_step=zero_i32.clone(),
            )
        cfg = self._cfg
        select_k = _sat_action_select_k_from_config(cfg)
        max_hotspot_subsets = max(int(getattr(cfg, "hotspot_num_subsets", 0) or 0), 1)
        raw_num_clusters = getattr(cfg, "gu_init_num_clusters", None)
        num_clusters = max(1, int(cfg.num_gu // 5)) if raw_num_clusters is None else max(1, int(raw_num_clusters))
        return StructuredBatchRuntimeTensorState(
            uav_pos=torch.zeros((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=kernel_device),
            uav_vel=torch.zeros((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=kernel_device),
            uav_energy=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            uav_queue=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            gu_pos=torch.zeros((batch_size, int(cfg.num_gu), 2), dtype=torch.float32, device=kernel_device),
            gu_cluster_centers=torch.zeros((batch_size, int(num_clusters), 2), dtype=torch.float32, device=kernel_device),
            gu_cluster_counts=torch.zeros((batch_size, int(num_clusters)), dtype=torch.float32, device=kernel_device),
            gu_queue=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            sat_queue=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            sat_pos=torch.zeros((batch_size, int(cfg.num_sat), 3), dtype=torch.float32, device=kernel_device),
            sat_vel=torch.zeros((batch_size, int(cfg.num_sat), 3), dtype=torch.float32, device=kernel_device),
            prev_association=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.int32, device=kernel_device),
            last_association=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.int32, device=kernel_device),
            last_sat_selection_matrix=torch.full((batch_size, int(cfg.num_uav), select_k), -1, dtype=torch.int64, device=kernel_device),
            last_sat_connection_counts=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            arrival_ref_bits_per_step=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            effective_task_arrival_rate=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            arrival_base_scale=torch.ones((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            hotspot_active_idx=torch.full((batch_size,), -1, dtype=torch.int32, device=kernel_device),
            hotspot_subset_count=torch.zeros((batch_size,), dtype=torch.int32, device=kernel_device),
            hotspot_member_mask=torch.zeros(
                (batch_size, max_hotspot_subsets, int(cfg.num_gu)),
                dtype=torch.float32,
                device=kernel_device,
            ),
            traffic_reset_step=torch.full((batch_size,), -1, dtype=torch.int32, device=kernel_device),
            traffic_reset_ordinal=torch.full((batch_size,), -1, dtype=torch.int32, device=kernel_device),
            episode_idx=torch.zeros((batch_size,), dtype=torch.int32, device=kernel_device),
            gu_workload_ema=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            uav_workload_ema=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            sat_workload_ema=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            last_gu_arrival_rate_vec=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            gu_deadline_steps=torch.ones((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_arrival=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_outflow=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            gu_drop=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            uav_drop=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            sat_drop=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            last_access_interference_by_uav=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            last_bw_fraction_by_uav_gu=torch.zeros((batch_size, int(cfg.num_uav), int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_to_uav_inflow_by_uav=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            last_uav_to_sat_outflow_matrix=torch.zeros((batch_size, int(cfg.num_uav), int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            last_selected_mask_by_uav_sat=torch.zeros((batch_size, int(cfg.num_uav), int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            last_sat_processed=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            last_gu_urgency_risk=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_downstream_pressure=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_service_gap_risk=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_deadline_slack=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_deadline_risk=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_service_gap=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_gu_deadline_age=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            last_exec_accel=torch.zeros((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=kernel_device),
            last_policy_accel=torch.zeros((batch_size, int(cfg.num_uav), 2), dtype=torch.float32, device=kernel_device),
            avoidance_eta_eff=torch.full(
                (batch_size,),
                float(getattr(cfg, "avoidance_eta", 0.0) or 0.0),
                dtype=torch.float32,
                device=kernel_device,
            ),
            last_avoidance_eta_exec=torch.full(
                (batch_size,),
                float(getattr(cfg, "avoidance_eta", 0.0) or 0.0),
                dtype=torch.float32,
                device=kernel_device,
            ),
            doppler_residual=torch.zeros((batch_size, int(cfg.num_uav), int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            prev_queue_sum_gu=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            prev_queue_sum_uav=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            prev_queue_sum_sat=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            prev_q_norm_active=torch.zeros((batch_size,), dtype=torch.float32, device=kernel_device),
            prev_gu_queue_vec=torch.zeros((batch_size, int(cfg.num_gu)), dtype=torch.float32, device=kernel_device),
            prev_uav_queue_vec=torch.zeros((batch_size, int(cfg.num_uav)), dtype=torch.float32, device=kernel_device),
            prev_sat_queue_vec=torch.zeros((batch_size, int(cfg.num_sat)), dtype=torch.float32, device=kernel_device),
            t=torch.zeros((batch_size,), dtype=torch.int32, device=kernel_device),
            global_step=torch.zeros((batch_size,), dtype=torch.int32, device=kernel_device),
        )

    def _sync_runtime_state(self, indices: Sequence[int] | None = None) -> None:
        selected_indices = self.resolve_indices(indices)
        if not selected_indices:
            return
        envs = self._slot_views(selected_indices)
        cfg = self._cfg
        tensor_state = self._runtime_tensor_state
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=kernel_device)
        uav_pos = np.stack([np.asarray(env.uav_pos, dtype=np.float32) for env in envs], axis=0)
        uav_vel = np.stack([np.asarray(env.uav_vel, dtype=np.float32) for env in envs], axis=0)
        uav_energy = np.stack([np.asarray(env.uav_energy, dtype=np.float32) for env in envs], axis=0)
        uav_queue = np.stack([np.asarray(env.uav_queue, dtype=np.float32) for env in envs], axis=0)
        gu_pos = np.stack([np.asarray(env.gu_pos, dtype=np.float32) for env in envs], axis=0)
        cluster_width = int(tensor_state.gu_cluster_centers.shape[1])
        gu_cluster_centers = np.zeros((len(envs), cluster_width, 2), dtype=np.float32)
        gu_cluster_counts = np.zeros((len(envs), cluster_width), dtype=np.float32)
        for row, env in enumerate(envs):
            centers = np.asarray(getattr(env, "gu_cluster_centers", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
            counts = np.asarray(getattr(env, "gu_cluster_counts", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            count = min(int(cluster_width), int(centers.shape[0]), int(counts.shape[0]))
            if count > 0:
                gu_cluster_centers[row, :count, :] = centers[:count]
                gu_cluster_counts[row, :count] = counts[:count]
        gu_queue = np.stack([np.asarray(env.gu_queue, dtype=np.float32) for env in envs], axis=0)
        sat_queue = np.stack([np.asarray(env.sat_queue, dtype=np.float32) for env in envs], axis=0)
        tensor_state.uav_pos[selected_tensor] = _as_kernel_tensor(uav_pos, dtype=torch.float32, device=kernel_device)
        tensor_state.uav_vel[selected_tensor] = _as_kernel_tensor(uav_vel, dtype=torch.float32, device=kernel_device)
        tensor_state.uav_energy[selected_tensor] = _as_kernel_tensor(uav_energy, dtype=torch.float32, device=kernel_device)
        tensor_state.uav_queue[selected_tensor] = _as_kernel_tensor(uav_queue, dtype=torch.float32, device=kernel_device)
        tensor_state.gu_pos[selected_tensor] = _as_kernel_tensor(gu_pos, dtype=torch.float32, device=kernel_device)
        tensor_state.gu_cluster_centers[selected_tensor] = _as_kernel_tensor(gu_cluster_centers, dtype=torch.float32, device=kernel_device)
        tensor_state.gu_cluster_counts[selected_tensor] = _as_kernel_tensor(gu_cluster_counts, dtype=torch.float32, device=kernel_device)
        tensor_state.gu_queue[selected_tensor] = _as_kernel_tensor(gu_queue, dtype=torch.float32, device=kernel_device)
        tensor_state.sat_queue[selected_tensor] = _as_kernel_tensor(sat_queue, dtype=torch.float32, device=kernel_device)
        prev_association = np.stack(
            [np.asarray(env.prev_association, dtype=np.int32) for env in envs],
            axis=0,
        )
        last_association = np.stack(
            [np.asarray(getattr(env, "last_association", np.full((env.cfg.num_gu,), -1, dtype=np.int32)), dtype=np.int32) for env in envs],
            axis=0,
        )
        last_sat_selection_matrix = np.stack(
            [
                env._sat_selection_matrix(getattr(env, "last_sat_selection", [[] for _ in range(env.cfg.num_uav)]))
                for env in envs
            ],
            axis=0,
        ).astype(np.int64, copy=False)
        last_sat_connection_counts = np.stack(
            [
                np.asarray(
                    getattr(env, "last_sat_connection_counts", np.zeros((env.cfg.num_sat,), dtype=np.float32)),
                    dtype=np.float32,
                )
                for env in envs
            ],
            axis=0,
        )
        tensor_state.prev_association[selected_tensor] = _as_kernel_tensor(prev_association, dtype=torch.int32, device=kernel_device)
        tensor_state.last_association[selected_tensor] = _as_kernel_tensor(last_association, dtype=torch.int32, device=kernel_device)
        tensor_state.last_sat_selection_matrix[selected_tensor] = _as_kernel_tensor(last_sat_selection_matrix, dtype=torch.int64, device=kernel_device)
        tensor_state.last_sat_connection_counts[selected_tensor] = _as_kernel_tensor(last_sat_connection_counts, dtype=torch.float32, device=kernel_device)
        arrival_ref_bits_per_step = _arrival_ref_batch(envs)
        effective_task_arrival_rate = np.asarray(
            [float(getattr(env, "effective_task_arrival_rate", env.cfg.task_arrival_rate)) for env in envs],
            dtype=np.float32,
        )
        tensor_state.arrival_ref_bits_per_step[selected_tensor] = _as_kernel_tensor(arrival_ref_bits_per_step, dtype=torch.float32, device=kernel_device)
        tensor_state.effective_task_arrival_rate[selected_tensor] = _as_kernel_tensor(effective_task_arrival_rate, dtype=torch.float32, device=kernel_device)
        max_hotspot_subsets = int(tensor_state.hotspot_member_mask.shape[1])
        arrival_base_scale = np.stack(
            [
                np.asarray(
                    getattr(env, "_arrival_base_scale", np.ones((env.cfg.num_gu,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(int(env.cfg.num_gu))
                for env in envs
            ],
            axis=0,
        )
        hotspot_active_idx = np.asarray(
            [int(getattr(env, "_hotspot_active_idx", -1)) for env in envs],
            dtype=np.int32,
        )
        hotspot_subset_count = np.zeros((len(envs),), dtype=np.int32)
        hotspot_member_mask = np.zeros(
            (len(envs), max_hotspot_subsets, int(cfg.num_gu)),
            dtype=np.float32,
        )
        for env_index, env in enumerate(envs):
            raw_mask = np.asarray(
                getattr(env, "_hotspot_member_mask", np.zeros((0, int(cfg.num_gu)), dtype=bool)),
                dtype=bool,
            ).reshape(-1, int(cfg.num_gu))
            count = min(int(raw_mask.shape[0]), max_hotspot_subsets)
            hotspot_subset_count[int(env_index)] = int(count)
            if count > 0:
                hotspot_member_mask[int(env_index), :count, :] = raw_mask[:count].astype(np.float32, copy=False)
        tensor_state.arrival_base_scale[selected_tensor] = _as_kernel_tensor(arrival_base_scale, dtype=torch.float32, device=kernel_device)
        tensor_state.hotspot_active_idx[selected_tensor] = _as_kernel_tensor(hotspot_active_idx, dtype=torch.int32, device=kernel_device)
        tensor_state.hotspot_subset_count[selected_tensor] = _as_kernel_tensor(hotspot_subset_count, dtype=torch.int32, device=kernel_device)
        tensor_state.hotspot_member_mask[selected_tensor] = _as_kernel_tensor(hotspot_member_mask, dtype=torch.float32, device=kernel_device)
        tensor_state.traffic_reset_step[selected_tensor] = torch.full(
            (len(envs),),
            -1,
            dtype=torch.int32,
            device=kernel_device,
        )
        tensor_state.traffic_reset_ordinal[selected_tensor] = torch.full(
            (len(envs),),
            -1,
            dtype=torch.int32,
            device=kernel_device,
        )
        episode_idx = np.asarray(
            [
                int(self._slot_state_payloads[int(slot)].get("episode_idx", getattr(env, "episode_idx", 0)))
                for slot, env in zip(selected_indices, envs)
            ],
            dtype=np.int32,
        )
        tensor_state.episode_idx[selected_tensor] = _as_kernel_tensor(episode_idx, dtype=torch.int32, device=kernel_device)
        gu_ema, uav_ema, sat_ema = _bw_weighted_workload_device_ema_vectors_batch(envs)
        tensor_state.gu_workload_ema[selected_tensor] = _as_kernel_tensor(gu_ema, dtype=torch.float32, device=kernel_device)
        tensor_state.uav_workload_ema[selected_tensor] = _as_kernel_tensor(uav_ema, dtype=torch.float32, device=kernel_device)
        tensor_state.sat_workload_ema[selected_tensor] = _as_kernel_tensor(sat_ema, dtype=torch.float32, device=kernel_device)
        def _stack_gu_meta(name: str, default: float = 0.0) -> np.ndarray:
            return np.stack(
                [
                    np.asarray(
                        getattr(env, name, np.full((env.cfg.num_gu,), default, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            )

        tensor_state.last_gu_arrival_rate_vec[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_arrival_rate_vec"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.gu_deadline_steps[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("gu_deadline_steps", default=1.0),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_arrival[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_arrival"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_outflow[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_outflow"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.gu_drop[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("gu_drop"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.uav_drop[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(getattr(env, "uav_drop", np.zeros((env.cfg.num_uav,), dtype=np.float32)), dtype=np.float32)
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.sat_drop[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(getattr(env, "sat_drop", np.zeros((env.cfg.num_sat,), dtype=np.float32)), dtype=np.float32)
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_access_interference_by_uav[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_access_interference_by_uav", np.zeros((env.cfg.num_uav,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_bw_fraction_by_uav_gu[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_bw_fraction_by_uav_gu", np.zeros((env.cfg.num_uav, env.cfg.num_gu), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_to_uav_inflow_by_uav[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_gu_to_uav_inflow_by_uav", np.zeros((env.cfg.num_uav,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_uav_to_sat_outflow_matrix[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_uav_to_sat_outflow_matrix", np.zeros((env.cfg.num_uav, env.cfg.num_sat), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_selected_mask_by_uav_sat[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_selected_mask_by_uav_sat", np.zeros((env.cfg.num_uav, env.cfg.num_sat), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_sat_processed[selected_tensor] = _as_kernel_tensor(
            np.stack(
                [
                    np.asarray(
                        getattr(env, "last_sat_processed", np.zeros((env.cfg.num_sat,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for env in envs
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_urgency_risk[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_urgency_risk"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_downstream_pressure[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_downstream_pressure"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_service_gap_risk[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_service_gap_risk"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_deadline_slack[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_deadline_slack"),
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_gu_deadline_risk[selected_tensor] = _as_kernel_tensor(
            _stack_gu_meta("last_gu_deadline_risk"),
            dtype=torch.float32,
            device=kernel_device,
        )
        last_gu_service_gap = np.stack(
            [
                np.asarray(getattr(env, "last_gu_service_gap", np.zeros((env.cfg.num_gu,), dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        last_gu_deadline_age = np.stack(
            [
                np.asarray(getattr(env, "last_gu_deadline_age", np.zeros((env.cfg.num_gu,), dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        tensor_state.last_gu_service_gap[selected_tensor] = _as_kernel_tensor(last_gu_service_gap, dtype=torch.float32, device=kernel_device)
        tensor_state.last_gu_deadline_age[selected_tensor] = _as_kernel_tensor(last_gu_deadline_age, dtype=torch.float32, device=kernel_device)
        last_exec_accel = np.stack(
            [
                np.asarray(getattr(env, "last_exec_accel", np.zeros((env.cfg.num_uav, 2), dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        last_policy_accel = np.stack(
            [
                np.asarray(getattr(env, "last_policy_accel", np.zeros((env.cfg.num_uav, 2), dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        doppler_residual = np.stack(
            [
                np.asarray(getattr(env, "_doppler_residual_state_hz", np.zeros((env.cfg.num_uav, env.cfg.num_sat), dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        tensor_state.last_exec_accel[selected_tensor] = _as_kernel_tensor(last_exec_accel, dtype=torch.float32, device=kernel_device)
        tensor_state.last_policy_accel[selected_tensor] = _as_kernel_tensor(last_policy_accel, dtype=torch.float32, device=kernel_device)
        avoidance_eta_eff = np.asarray(
            [float(getattr(env, "avoidance_eta_eff", env.cfg.avoidance_eta)) for env in envs],
            dtype=np.float32,
        )
        last_avoidance_eta_exec = np.asarray(
            [float(getattr(env, "last_avoidance_eta_exec", env.cfg.avoidance_eta)) for env in envs],
            dtype=np.float32,
        )
        tensor_state.avoidance_eta_eff[selected_tensor] = _as_kernel_tensor(
            avoidance_eta_eff,
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.last_avoidance_eta_exec[selected_tensor] = _as_kernel_tensor(
            last_avoidance_eta_exec,
            dtype=torch.float32,
            device=kernel_device,
        )
        tensor_state.doppler_residual[selected_tensor] = _as_kernel_tensor(doppler_residual, dtype=torch.float32, device=kernel_device)
        prev_queue_sum_gu = np.asarray(
            [float(getattr(env, "prev_queue_sum_gu", np.sum(env.gu_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_queue_sum_uav = np.asarray(
            [float(getattr(env, "prev_queue_sum_uav", np.sum(env.uav_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_queue_sum_sat = np.asarray(
            [float(getattr(env, "prev_queue_sum_sat", np.sum(env.sat_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_q_norm_active = np.asarray(
            [float(getattr(env, "prev_q_norm_active", 0.0)) for env in envs],
            dtype=np.float32,
        )
        prev_gu_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_gu_queue_vec", np.asarray(env.gu_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        prev_uav_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_uav_queue_vec", np.asarray(env.uav_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        prev_sat_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_sat_queue_vec", np.asarray(env.sat_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        t = np.asarray([int(env.t) for env in envs], dtype=np.int32)
        global_step = np.asarray([int(env.global_step) for env in envs], dtype=np.int32)
        tensor_state.prev_queue_sum_gu[selected_tensor] = _as_kernel_tensor(prev_queue_sum_gu, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_queue_sum_uav[selected_tensor] = _as_kernel_tensor(prev_queue_sum_uav, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_queue_sum_sat[selected_tensor] = _as_kernel_tensor(prev_queue_sum_sat, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_q_norm_active[selected_tensor] = _as_kernel_tensor(prev_q_norm_active, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_gu_queue_vec[selected_tensor] = _as_kernel_tensor(prev_gu_queue_vec, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_uav_queue_vec[selected_tensor] = _as_kernel_tensor(prev_uav_queue_vec, dtype=torch.float32, device=kernel_device)
        tensor_state.prev_sat_queue_vec[selected_tensor] = _as_kernel_tensor(prev_sat_queue_vec, dtype=torch.float32, device=kernel_device)
        tensor_state.t[selected_tensor] = _as_kernel_tensor(t, dtype=torch.int32, device=kernel_device)
        tensor_state.global_step[selected_tensor] = _as_kernel_tensor(global_step, dtype=torch.int32, device=kernel_device)
        self._sync_runtime_orbit_state_from_t(selected_indices, t_values=t)

    def _write_runtime_state_from_serialized_states(
        self,
        indices: Sequence[int],
        states: Sequence[dict[str, Any]],
    ) -> None:
        selected_indices = [int(index) for index in indices]
        if not selected_indices:
            return
        if len(states) != len(selected_indices):
            raise ValueError(f"Expected {len(selected_indices)} runtime states, got {len(states)}.")
        envs = self._slot_views(selected_indices)
        cfg = self._cfg
        runtime_tensor_state = self._runtime_tensor_state
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=kernel_device)

        def _stack(key: str, *, dtype, fallback_attr: str | None = None) -> np.ndarray:
            rows: list[np.ndarray] = []
            for payload, env in zip(states, envs):
                raw_value = payload.get(key)
                if raw_value is None and fallback_attr is not None:
                    raw_value = getattr(env, fallback_attr)
                rows.append(np.asarray(raw_value, dtype=dtype))
            return np.stack(rows, axis=0)

        uav_pos = _stack("uav_pos", dtype=np.float32, fallback_attr="uav_pos")
        uav_vel = _stack("uav_vel", dtype=np.float32, fallback_attr="uav_vel")
        uav_energy = _stack("uav_energy", dtype=np.float32, fallback_attr="uav_energy")
        uav_queue = _stack("uav_queue", dtype=np.float32, fallback_attr="uav_queue")
        gu_pos = _stack("gu_pos", dtype=np.float32, fallback_attr="gu_pos")
        cluster_width = int(runtime_tensor_state.gu_cluster_centers.shape[1])
        gu_cluster_centers = np.zeros((len(states), cluster_width, 2), dtype=np.float32)
        gu_cluster_counts = np.zeros((len(states), cluster_width), dtype=np.float32)
        for row, (payload, env) in enumerate(zip(states, envs)):
            centers = np.asarray(
                payload.get("gu_cluster_centers", getattr(env, "gu_cluster_centers", np.zeros((0, 2), dtype=np.float32))),
                dtype=np.float32,
            ).reshape(-1, 2)
            counts = np.asarray(
                payload.get("gu_cluster_counts", getattr(env, "gu_cluster_counts", np.zeros((0,), dtype=np.float32))),
                dtype=np.float32,
            ).reshape(-1)
            count = min(int(cluster_width), int(centers.shape[0]), int(counts.shape[0]))
            if count > 0:
                gu_cluster_centers[row, :count, :] = centers[:count]
                gu_cluster_counts[row, :count] = counts[:count]
        gu_queue = _stack("gu_queue", dtype=np.float32, fallback_attr="gu_queue")
        sat_queue = _stack("sat_queue", dtype=np.float32, fallback_attr="sat_queue")
        prev_association = _stack("prev_association", dtype=np.int32, fallback_attr="prev_association")
        last_association = _stack("last_association", dtype=np.int32, fallback_attr="last_association")
        last_sat_selection_matrix = np.stack(
            [
                _sat_selection_matrix_from_values(
                    cfg,
                    payload.get("last_sat_selection", getattr(env, "last_sat_selection", [[] for _ in range(cfg.num_uav)])),
                )
                for payload, env in zip(states, envs)
            ],
            axis=0,
        ).astype(np.int64, copy=False)
        last_sat_connection_counts = _stack(
            "last_sat_connection_counts",
            dtype=np.float32,
            fallback_attr="last_sat_connection_counts",
        )
        arrival_ref_bits_per_step = np.asarray(
            [
                float(payload.get("arrival_ref_bits_per_step", getattr(env, "arrival_ref_bits_per_step", 0.0)))
                for payload, env in zip(states, envs)
            ],
            dtype=np.float32,
        )
        arrival_ref_bits_per_step = require_positive_array(
            arrival_ref_bits_per_step,
            name="arrival_ref_bits_per_step",
        )
        effective_task_arrival_rate = np.asarray(
            [
                float(payload.get("effective_task_arrival_rate", getattr(env, "effective_task_arrival_rate", cfg.task_arrival_rate)))
                for payload, env in zip(states, envs)
            ],
            dtype=np.float32,
        )
        arrival_base_scale = _stack("_arrival_base_scale", dtype=np.float32, fallback_attr="_arrival_base_scale")
        max_hotspot_subsets = int(runtime_tensor_state.hotspot_member_mask.shape[1])
        hotspot_active_idx = np.asarray(
            [int(payload.get("_hotspot_active_idx", getattr(env, "_hotspot_active_idx", -1))) for payload, env in zip(states, envs)],
            dtype=np.int32,
        )
        hotspot_subset_count = np.zeros((len(states),), dtype=np.int32)
        hotspot_member_mask = np.zeros((len(states), max_hotspot_subsets, int(cfg.num_gu)), dtype=np.float32)
        for row, (payload, env) in enumerate(zip(states, envs)):
            raw_mask = np.asarray(
                payload.get(
                    "_hotspot_member_mask",
                    getattr(env, "_hotspot_member_mask", np.zeros((0, int(cfg.num_gu)), dtype=bool)),
                ),
                dtype=bool,
            ).reshape(-1, int(cfg.num_gu))
            count = min(int(raw_mask.shape[0]), max_hotspot_subsets)
            hotspot_subset_count[int(row)] = int(count)
            if count > 0:
                hotspot_member_mask[int(row), :count, :] = raw_mask[:count].astype(np.float32, copy=False)
        gu_ema = _stack("bw_weighted_workload_acc_ema_vec", dtype=np.float32, fallback_attr="bw_weighted_workload_acc_ema_vec")
        uav_ema = _stack("bw_weighted_workload_rel_ema_vec", dtype=np.float32, fallback_attr="bw_weighted_workload_rel_ema_vec")
        sat_ema = _stack("bw_weighted_workload_sat_ema_vec", dtype=np.float32, fallback_attr="bw_weighted_workload_sat_ema_vec")
        last_gu_arrival_rate_vec = _stack("last_gu_arrival_rate_vec", dtype=np.float32, fallback_attr="last_gu_arrival_rate_vec")
        gu_deadline_steps = _stack("gu_deadline_steps", dtype=np.float32, fallback_attr="gu_deadline_steps")
        last_gu_arrival = _stack("last_gu_arrival", dtype=np.float32, fallback_attr="last_gu_arrival")
        last_gu_outflow = _stack("last_gu_outflow", dtype=np.float32, fallback_attr="last_gu_outflow")
        gu_drop = _stack("gu_drop", dtype=np.float32, fallback_attr="gu_drop")
        uav_drop = _stack("uav_drop", dtype=np.float32, fallback_attr="uav_drop")
        sat_drop = _stack("sat_drop", dtype=np.float32, fallback_attr="sat_drop")
        last_access_interference_by_uav = _stack(
            "last_access_interference_by_uav",
            dtype=np.float32,
            fallback_attr="last_access_interference_by_uav",
        )
        last_bw_fraction_by_uav_gu = _stack(
            "last_bw_fraction_by_uav_gu",
            dtype=np.float32,
            fallback_attr="last_bw_fraction_by_uav_gu",
        )
        last_gu_to_uav_inflow_by_uav = _stack(
            "last_gu_to_uav_inflow_by_uav",
            dtype=np.float32,
            fallback_attr="last_gu_to_uav_inflow_by_uav",
        )
        last_uav_to_sat_outflow_matrix = _stack(
            "last_uav_to_sat_outflow_matrix",
            dtype=np.float32,
            fallback_attr="last_uav_to_sat_outflow_matrix",
        )
        last_selected_mask_by_uav_sat = _stack(
            "last_selected_mask_by_uav_sat",
            dtype=np.float32,
            fallback_attr="last_selected_mask_by_uav_sat",
        )
        last_sat_processed = _stack("last_sat_processed", dtype=np.float32, fallback_attr="last_sat_processed")
        last_gu_urgency_risk = _stack("last_gu_urgency_risk", dtype=np.float32, fallback_attr="last_gu_urgency_risk")
        last_gu_downstream_pressure = _stack("last_gu_downstream_pressure", dtype=np.float32, fallback_attr="last_gu_downstream_pressure")
        last_gu_service_gap_risk = _stack("last_gu_service_gap_risk", dtype=np.float32, fallback_attr="last_gu_service_gap_risk")
        last_gu_deadline_slack = _stack("last_gu_deadline_slack", dtype=np.float32, fallback_attr="last_gu_deadline_slack")
        last_gu_deadline_risk = _stack("last_gu_deadline_risk", dtype=np.float32, fallback_attr="last_gu_deadline_risk")
        last_gu_service_gap = _stack("last_gu_service_gap", dtype=np.float32, fallback_attr="last_gu_service_gap")
        last_gu_deadline_age = _stack("last_gu_deadline_age", dtype=np.float32, fallback_attr="last_gu_deadline_age")
        last_exec_accel = _stack("last_exec_accel", dtype=np.float32, fallback_attr="last_exec_accel")
        last_policy_accel = _stack("last_policy_accel", dtype=np.float32, fallback_attr="last_policy_accel")
        doppler_residual = _stack("_doppler_residual_state_hz", dtype=np.float32, fallback_attr="_doppler_residual_state_hz")
        avoidance_eta_eff = np.asarray(
            [
                float(payload.get("avoidance_eta_eff", getattr(env, "avoidance_eta_eff", cfg.avoidance_eta)))
                for payload, env in zip(states, envs)
            ],
            dtype=np.float32,
        )
        last_avoidance_eta_exec = np.asarray(
            [
                float(payload.get("last_avoidance_eta_exec", getattr(env, "last_avoidance_eta_exec", cfg.avoidance_eta)))
                for payload, env in zip(states, envs)
            ],
            dtype=np.float32,
        )
        prev_queue_sum_gu = np.asarray(
            [float(payload.get("prev_queue_sum_gu", getattr(env, "prev_queue_sum_gu", 0.0))) for payload, env in zip(states, envs)],
            dtype=np.float32,
        )
        prev_queue_sum_uav = np.asarray(
            [float(payload.get("prev_queue_sum_uav", getattr(env, "prev_queue_sum_uav", 0.0))) for payload, env in zip(states, envs)],
            dtype=np.float32,
        )
        prev_queue_sum_sat = np.asarray(
            [float(payload.get("prev_queue_sum_sat", getattr(env, "prev_queue_sum_sat", 0.0))) for payload, env in zip(states, envs)],
            dtype=np.float32,
        )
        prev_q_norm_active = np.asarray(
            [float(payload.get("prev_q_norm_active", getattr(env, "prev_q_norm_active", 0.0))) for payload, env in zip(states, envs)],
            dtype=np.float32,
        )
        prev_gu_queue_vec = _stack("prev_gu_queue_vec", dtype=np.float32, fallback_attr="prev_gu_queue_vec")
        prev_uav_queue_vec = _stack("prev_uav_queue_vec", dtype=np.float32, fallback_attr="prev_uav_queue_vec")
        prev_sat_queue_vec = _stack("prev_sat_queue_vec", dtype=np.float32, fallback_attr="prev_sat_queue_vec")
        t = np.asarray(
            [int(payload.get("t", getattr(env, "t", 0))) for payload, env in zip(states, envs)],
            dtype=np.int32,
        )
        global_step = np.asarray(
            [int(payload.get("global_step", getattr(env, "global_step", 0))) for payload, env in zip(states, envs)],
            dtype=np.int32,
        )

        runtime_tensor_state.uav_pos[selected_tensor] = _as_kernel_tensor(uav_pos, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.uav_vel[selected_tensor] = _as_kernel_tensor(uav_vel, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.uav_energy[selected_tensor] = _as_kernel_tensor(uav_energy, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.uav_queue[selected_tensor] = _as_kernel_tensor(uav_queue, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_pos[selected_tensor] = _as_kernel_tensor(gu_pos, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_cluster_centers[selected_tensor] = _as_kernel_tensor(gu_cluster_centers, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_cluster_counts[selected_tensor] = _as_kernel_tensor(gu_cluster_counts, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_queue[selected_tensor] = _as_kernel_tensor(gu_queue, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.sat_queue[selected_tensor] = _as_kernel_tensor(sat_queue, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_association[selected_tensor] = _as_kernel_tensor(prev_association, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.last_association[selected_tensor] = _as_kernel_tensor(last_association, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.last_sat_selection_matrix[selected_tensor] = _as_kernel_tensor(last_sat_selection_matrix, dtype=torch.int64, device=kernel_device)
        runtime_tensor_state.last_sat_connection_counts[selected_tensor] = _as_kernel_tensor(last_sat_connection_counts, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.arrival_ref_bits_per_step[selected_tensor] = _as_kernel_tensor(arrival_ref_bits_per_step, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.effective_task_arrival_rate[selected_tensor] = _as_kernel_tensor(effective_task_arrival_rate, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.arrival_base_scale[selected_tensor] = _as_kernel_tensor(arrival_base_scale, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.hotspot_active_idx[selected_tensor] = _as_kernel_tensor(hotspot_active_idx, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.hotspot_subset_count[selected_tensor] = _as_kernel_tensor(hotspot_subset_count, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.hotspot_member_mask[selected_tensor] = _as_kernel_tensor(hotspot_member_mask, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.traffic_reset_step[selected_tensor] = torch.full(
            (len(states),),
            -1,
            dtype=torch.int32,
            device=kernel_device,
        )
        runtime_tensor_state.traffic_reset_ordinal[selected_tensor] = torch.full(
            (len(states),),
            -1,
            dtype=torch.int32,
            device=kernel_device,
        )
        episode_idx = np.asarray(
            [
                int(payload.get("episode_idx", getattr(env, "episode_idx", 0)))
                for payload, env in zip(states, envs)
            ],
            dtype=np.int32,
        )
        runtime_tensor_state.episode_idx[selected_tensor] = _as_kernel_tensor(episode_idx, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.gu_workload_ema[selected_tensor] = _as_kernel_tensor(gu_ema, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.uav_workload_ema[selected_tensor] = _as_kernel_tensor(uav_ema, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.sat_workload_ema[selected_tensor] = _as_kernel_tensor(sat_ema, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_arrival_rate_vec[selected_tensor] = _as_kernel_tensor(last_gu_arrival_rate_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_deadline_steps[selected_tensor] = _as_kernel_tensor(gu_deadline_steps, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_arrival[selected_tensor] = _as_kernel_tensor(last_gu_arrival, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_outflow[selected_tensor] = _as_kernel_tensor(last_gu_outflow, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.gu_drop[selected_tensor] = _as_kernel_tensor(gu_drop, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.uav_drop[selected_tensor] = _as_kernel_tensor(uav_drop, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.sat_drop[selected_tensor] = _as_kernel_tensor(sat_drop, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_access_interference_by_uav[selected_tensor] = _as_kernel_tensor(last_access_interference_by_uav, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_bw_fraction_by_uav_gu[selected_tensor] = _as_kernel_tensor(last_bw_fraction_by_uav_gu, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_to_uav_inflow_by_uav[selected_tensor] = _as_kernel_tensor(last_gu_to_uav_inflow_by_uav, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_uav_to_sat_outflow_matrix[selected_tensor] = _as_kernel_tensor(last_uav_to_sat_outflow_matrix, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_selected_mask_by_uav_sat[selected_tensor] = _as_kernel_tensor(last_selected_mask_by_uav_sat, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_sat_processed[selected_tensor] = _as_kernel_tensor(last_sat_processed, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_urgency_risk[selected_tensor] = _as_kernel_tensor(last_gu_urgency_risk, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_downstream_pressure[selected_tensor] = _as_kernel_tensor(last_gu_downstream_pressure, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_service_gap_risk[selected_tensor] = _as_kernel_tensor(last_gu_service_gap_risk, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_deadline_slack[selected_tensor] = _as_kernel_tensor(last_gu_deadline_slack, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_deadline_risk[selected_tensor] = _as_kernel_tensor(last_gu_deadline_risk, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_service_gap[selected_tensor] = _as_kernel_tensor(last_gu_service_gap, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_gu_deadline_age[selected_tensor] = _as_kernel_tensor(last_gu_deadline_age, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_exec_accel[selected_tensor] = _as_kernel_tensor(last_exec_accel, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.last_policy_accel[selected_tensor] = _as_kernel_tensor(last_policy_accel, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.avoidance_eta_eff[selected_tensor] = _as_kernel_tensor(
            avoidance_eta_eff,
            dtype=torch.float32,
            device=kernel_device,
        )
        runtime_tensor_state.last_avoidance_eta_exec[selected_tensor] = _as_kernel_tensor(
            last_avoidance_eta_exec,
            dtype=torch.float32,
            device=kernel_device,
        )
        runtime_tensor_state.doppler_residual[selected_tensor] = _as_kernel_tensor(doppler_residual, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_queue_sum_gu[selected_tensor] = _as_kernel_tensor(prev_queue_sum_gu, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_queue_sum_uav[selected_tensor] = _as_kernel_tensor(prev_queue_sum_uav, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_queue_sum_sat[selected_tensor] = _as_kernel_tensor(prev_queue_sum_sat, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_q_norm_active[selected_tensor] = _as_kernel_tensor(prev_q_norm_active, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_gu_queue_vec[selected_tensor] = _as_kernel_tensor(prev_gu_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_uav_queue_vec[selected_tensor] = _as_kernel_tensor(prev_uav_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_sat_queue_vec[selected_tensor] = _as_kernel_tensor(prev_sat_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.t[selected_tensor] = _as_kernel_tensor(t, dtype=torch.int32, device=kernel_device)
        runtime_tensor_state.global_step[selected_tensor] = _as_kernel_tensor(global_step, dtype=torch.int32, device=kernel_device)
        self._sync_runtime_orbit_state_from_t(selected_indices, t_values=t)

    def _publish_runtime_reset_random_tape(self, indices: Sequence[int]) -> None:
        runtime = self.native_rollout_runtime
        if runtime is None:
            return
        selected_indices = [int(index) for index in indices]
        if not selected_indices:
            return
        runtime_device = torch.device(runtime.device)
        tensor_state = self._runtime_tensor_state
        state_device = torch.device(tensor_state.gu_pos.device)
        same_device = state_device == runtime_device or (
            state_device.type == runtime_device.type == "cuda"
            and (runtime_device.index is None or state_device.index in {None, runtime_device.index})
        )
        if not same_device:
            raise RuntimeError(
                "native reset random tape requires runtime tensor state on the rollout runtime device."
            )
        selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=state_device)
        full_batch_selected = bool(
            len(selected_indices) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected_indices))
        )

        def _select_rows(value: torch.Tensor) -> torch.Tensor:
            return value if full_batch_selected else value.index_select(0, selected_tensor)

        runtime.write_random_reset_tape(
            gu_pos=_select_rows(tensor_state.gu_pos).to(dtype=torch.float32),
            uav_pos=_select_rows(tensor_state.uav_pos).to(dtype=torch.float32),
            uav_vel=_select_rows(tensor_state.uav_vel).to(dtype=torch.float32),
            gu_cluster_centers=_select_rows(tensor_state.gu_cluster_centers).to(dtype=torch.float32),
            gu_cluster_counts=_select_rows(tensor_state.gu_cluster_counts).to(dtype=torch.float32),
            arrival_base_scale=_select_rows(tensor_state.arrival_base_scale).to(dtype=torch.float32),
            deadline_steps=_select_rows(tensor_state.gu_deadline_steps).to(dtype=torch.float32),
            doppler_residual=_select_rows(tensor_state.doppler_residual).to(dtype=torch.float32),
            effective_arrival_rate=_select_rows(tensor_state.effective_task_arrival_rate).to(dtype=torch.float32),
        )

    def _prepare_runtime_rollout_random_tapes(self, *, capacity: int, reset_rows: int | None = None) -> None:
        runtime = self.native_rollout_runtime
        if runtime is None:
            return
        steps = max(int(capacity), 0)
        if steps <= 0:
            return
        cfg = self._cfg
        runtime_device = torch.device(runtime.device)
        tensor_state = self._runtime_tensor_state
        state_device = torch.device(tensor_state.uav_pos.device)
        same_device = state_device == runtime_device or (
            state_device.type == runtime_device.type == "cuda"
            and (runtime_device.index is None or state_device.index in {None, runtime_device.index})
        )
        if not same_device:
            raise RuntimeError("native rollout random tape requires runtime tensor state on the rollout device.")

        arrival_tape = None
        arrival_rate_tape = None
        hotspot_active_tape = None
        hotspot_active_after_tape = None
        hotspot_mask_tape = None
        ramp_steps = int(getattr(cfg, "arrival_ramp_steps", 0) or 0)
        use_arrival_ramp = ablation_flag(cfg, "use_arrival_ramp", default=False) or ramp_steps > 0
        traffic_model = str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower()
        if int(cfg.num_gu) > 0:
            effective_rates = tensor_state.effective_task_arrival_rate.clamp_min(0.0).to(
                device=runtime_device,
                dtype=torch.float32,
            )
            base_rate_tape = effective_rates.view(1, int(self._num_envs)).expand(steps, int(self._num_envs)).contiguous()
            if use_arrival_ramp and ramp_steps > 0:
                start = float(getattr(cfg, "arrival_ramp_start", 0.0) or 0.0)
                start = float(np.clip(start, 0.0, 1.0))
                use_global = bool(getattr(cfg, "arrival_ramp_use_global", False))
                counter_t = (
                    tensor_state.global_step if use_global else tensor_state.t
                ).to(device=runtime_device, dtype=torch.float32)
                step_offset_t = torch.arange(steps, dtype=torch.float32, device=runtime_device).view(steps, 1)
                progress_t = torch.clamp((counter_t.view(1, int(self._num_envs)) + step_offset_t) / float(max(ramp_steps, 1)), max=1.0)
                ramp_factor_t = float(start) + (1.0 - float(start)) * progress_t
                base_rate_tape = base_rate_tape * ramp_factor_t

            if traffic_model == "sticky_subset_hotspot":
                base_scale_t = tensor_state.arrival_base_scale.to(device=runtime_device, dtype=torch.float32)
                member_mask_t = tensor_state.hotspot_member_mask.to(device=runtime_device, dtype=torch.float32)
                subset_counts_t = tensor_state.hotspot_subset_count.to(device=runtime_device, dtype=torch.long)
                max_subsets = int(max(int(member_mask_t.shape[1]), 1))
                active_t = tensor_state.hotspot_active_idx.to(device=runtime_device, dtype=torch.long)
                env_ids_t = torch.arange(int(self._num_envs), dtype=torch.long, device=runtime_device)
                arrival_rate_tape = torch.empty((steps, int(self._num_envs), int(cfg.num_gu)), dtype=torch.float32, device=runtime_device)
                hotspot_active_tape = torch.empty((steps, int(self._num_envs)), dtype=torch.long, device=runtime_device)
                hotspot_active_after_tape = torch.empty((steps, int(self._num_envs)), dtype=torch.int32, device=runtime_device)
                hotspot_mask_tape = torch.empty((steps, int(self._num_envs), int(cfg.num_gu)), dtype=torch.float32, device=runtime_device)
                rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
                preserve_mean = bool(getattr(cfg, "arrival_mean_preserve", True))
                on_prob = 1.0 / max(float(getattr(cfg, "hotspot_on_mean_steps", 15.0) or 15.0), 1.0)
                off_prob = 1.0 / max(float(getattr(cfg, "hotspot_off_mean_steps", 8.0) or 8.0), 1.0)
                for step_index in range(steps):
                    valid_active_t = (active_t >= 0) & (active_t < subset_counts_t)
                    active_clamped_t = torch.clamp(active_t, min=0, max=max_subsets - 1)
                    step_hot_mask_t = member_mask_t[env_ids_t, active_clamped_t]
                    step_hot_mask_t = torch.where(
                        valid_active_t.view(int(self._num_envs), 1),
                        step_hot_mask_t,
                        torch.zeros_like(step_hot_mask_t),
                    )
                    weights_t = base_scale_t * torch.where(
                        step_hot_mask_t > 0.5,
                        torch.full_like(base_scale_t, float(rho)),
                        torch.ones_like(base_scale_t),
                    )
                    if preserve_mean:
                        weights_t = _torch_divide_or_default(weights_t, weights_t.mean(dim=1, keepdim=True))
                    arrival_rate_tape[step_index].copy_(base_rate_tape[step_index].view(int(self._num_envs), 1) * weights_t)
                    hotspot_active_tape[step_index].copy_(active_t)
                    hotspot_mask_tape[step_index].copy_(step_hot_mask_t)

                    has_subsets_t = subset_counts_t > 0
                    if bool(has_subsets_t.any().item()):
                        on_rand_t = torch.rand((int(self._num_envs),), dtype=torch.float32, device=runtime_device, generator=self._native_torch_rng)
                        off_rand_t = torch.rand((int(self._num_envs),), dtype=torch.float32, device=runtime_device, generator=self._native_torch_rng)
                        chosen_t = torch.randint(
                            low=0,
                            high=max_subsets,
                            size=(int(self._num_envs),),
                            dtype=torch.long,
                            device=runtime_device,
                            generator=self._native_torch_rng,
                        ) % subset_counts_t.clamp_min(1)
                        turn_off_t = valid_active_t & (on_rand_t < float(on_prob))
                        inactive_t = (~valid_active_t) & has_subsets_t
                        turn_on_t = inactive_t & (off_rand_t < float(off_prob))
                        active_t = torch.where(turn_off_t, torch.full_like(active_t, -1), active_t)
                        active_t = torch.where(turn_on_t, chosen_t, active_t)
                    hotspot_active_after_tape[step_index].copy_(active_t.to(dtype=torch.int32))

                final_active_np = active_t.detach().cpu().numpy().astype(np.int64, copy=False)
                final_mask_np = hotspot_mask_tape[-1].detach().cpu().numpy().astype(np.float32, copy=False)
                for slot in range(int(self._num_envs)):
                    meta = self._slot_state_payloads[int(slot)]
                    final_active = int(final_active_np[int(slot)])
                    meta["_hotspot_active_idx"] = final_active
                    meta["last_hotspot_index"] = final_active
                    meta["last_hotspot_mask"] = np.asarray(final_mask_np[int(slot)], dtype=np.float32).copy()
            else:
                arrival_rate_tape = (
                    base_rate_tape.view(steps, int(self._num_envs), 1)
                    .expand(steps, int(self._num_envs), int(cfg.num_gu))
                    .contiguous()
                )

            if bool(cfg.task_arrival_poisson):
                arrival_tape = torch.poisson(arrival_rate_tape, generator=self._native_torch_rng).to(dtype=torch.float32)
            else:
                arrival_tape = arrival_rate_tape.clone()

        fading_gain_tape = None
        if (
            bool(cfg.fading_enabled)
            and _access_fading_mode_code_from_cfg(cfg) == 2
            and int(cfg.num_gu) > 0
            and int(cfg.num_uav) > 0
        ):
            k_value = channel.rician_k_linear_from_config(cfg)
            s_value = math.sqrt(k_value / (k_value + 1.0))
            sigma_value = math.sqrt(1.0 / (2.0 * (k_value + 1.0)))
            noise = torch.empty(
                (2, steps + 1, int(self._num_envs), int(cfg.num_gu), int(cfg.num_uav)),
                dtype=torch.float32,
                device=runtime_device,
            )
            noise.normal_(generator=self._native_torch_rng)
            real = noise[0].mul(float(sigma_value)).add(float(s_value))
            imag = noise[1].mul(float(sigma_value))
            fading_gain_tape = real.mul(real).addcmul(imag, imag)

        doppler_noise_tape = None
        cap = _doppler_residual_cap_hz_from_cfg(cfg)
        sigma = max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0)
        if (
            _doppler_precomp_enabled_from_cfg(cfg)
            and cap > 0.0
            and sigma > 0.0
            and int(cfg.num_uav) > 0
            and int(cfg.num_sat) > 0
        ):
            doppler_noise_tape = torch.empty(
                (steps, int(self._num_envs), int(cfg.num_uav), int(cfg.num_sat)),
                dtype=torch.float32,
                device=runtime_device,
            )
            doppler_noise_tape.normal_(generator=self._native_torch_rng)

        _traffic_level, _traffic_ratio, reset_effective_rate, reset_arrival_ref = _traffic_level_state_from_cfg(cfg)
        reset_row_count = max(int(reset_rows) if reset_rows is not None else steps, 1)
        reset_gu_pos_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_gu), 2), dtype=torch.float32, device=runtime_device)
        reset_uav_pos_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_uav), 2), dtype=torch.float32, device=runtime_device)
        reset_uav_vel_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_uav), 2), dtype=torch.float32, device=runtime_device)
        cluster_width = int(tensor_state.gu_cluster_centers.shape[1])
        reset_gu_cluster_centers_tape = torch.empty((reset_row_count, int(self._num_envs), cluster_width, 2), dtype=torch.float32, device=runtime_device)
        reset_gu_cluster_counts_tape = torch.empty((reset_row_count, int(self._num_envs), cluster_width), dtype=torch.float32, device=runtime_device)
        reset_arrival_base_scale_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_gu)), dtype=torch.float32, device=runtime_device)
        reset_deadline_steps_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_gu)), dtype=torch.float32, device=runtime_device)
        reset_doppler_residual_tape = torch.empty((reset_row_count, int(self._num_envs), int(cfg.num_uav), int(cfg.num_sat)), dtype=torch.float32, device=runtime_device)
        reset_effective_rate_tape = torch.full((reset_row_count, int(self._num_envs)), float(reset_effective_rate), dtype=torch.float32, device=runtime_device)
        reset_arrival_ref_tape = torch.full((reset_row_count, int(self._num_envs)), float(reset_arrival_ref), dtype=torch.float32, device=runtime_device)
        reset_arrival_rate_vec_tape = torch.full(
            (reset_row_count, int(self._num_envs), int(cfg.num_gu)),
            float(reset_effective_rate),
            dtype=torch.float32,
            device=runtime_device,
        )
        max_reset_hotspot_subsets = max(int(getattr(cfg, "hotspot_num_subsets", 0) or 0), 1)
        reset_hotspot_active_idx_tape = torch.full(
            (reset_row_count, int(self._num_envs)),
            -1,
            dtype=torch.int32,
            device=runtime_device,
        )
        reset_hotspot_subset_count_tape = torch.zeros((reset_row_count, int(self._num_envs)), dtype=torch.int32, device=runtime_device)
        reset_hotspot_member_mask_tape = torch.zeros(
            (reset_row_count, int(self._num_envs), max_reset_hotspot_subsets, int(cfg.num_gu)),
            dtype=torch.float32,
            device=runtime_device,
        )
        reset_episode_idx_tape = torch.empty((reset_row_count, int(self._num_envs)), dtype=torch.int32, device=runtime_device)
        gu_queue_total = _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_gu_abs",
            steps_attr="queue_init_gu_steps",
            frac_attr="queue_init_frac",
            layer="gu",
            total_cap=float(cfg.num_gu) * float(cfg.queue_max_gu),
            effective_task_arrival_rate=float(reset_effective_rate),
        )
        uav_queue_total = _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_uav_abs",
            steps_attr="queue_init_uav_steps",
            frac_attr="queue_init_uav_frac",
            layer="uav",
            total_cap=float(cfg.num_uav) * float(cfg.queue_max_uav),
            effective_task_arrival_rate=float(reset_effective_rate),
        )
        sat_queue_total = _resolve_queue_init_total_native(
            cfg,
            abs_attr="queue_init_sat_abs",
            steps_attr="queue_init_sat_steps",
            frac_attr="queue_init_sat_frac",
            layer="sat",
            total_cap=float(cfg.num_sat) * float(cfg.queue_max_sat),
            effective_task_arrival_rate=float(reset_effective_rate),
        )
        reset_gu_queue_tape = torch.full(
            (reset_row_count, int(self._num_envs), int(cfg.num_gu)),
            float(gu_queue_total) / max(float(cfg.num_gu), 1.0),
            dtype=torch.float32,
            device=runtime_device,
        )
        reset_uav_queue_tape = torch.full(
            (reset_row_count, int(self._num_envs), int(cfg.num_uav)),
            float(uav_queue_total) / max(float(cfg.num_uav), 1.0),
            dtype=torch.float32,
            device=runtime_device,
        )
        reset_sat_queue_tape = torch.full(
            (reset_row_count, int(self._num_envs), int(cfg.num_sat)),
            float(sat_queue_total) / max(float(cfg.num_sat), 1.0),
            dtype=torch.float32,
            device=runtime_device,
        )
        hetero = max(float(getattr(cfg, "arrival_base_hetero", 0.0) or 0.0), 0.0)
        arrival_low = max(1.0 - hetero, 1.0e-3)
        arrival_high = max(1.0 + hetero, arrival_low)
        deadline_base = max(float(getattr(cfg, "deadline_base_steps", 4.0) or 0.0), 1.0)
        deadline_jitter = max(float(getattr(cfg, "deadline_jitter_steps", 0.0) or 0.0), 0.0)
        doppler_cap_reset = _doppler_residual_cap_hz_from_cfg(cfg)
        doppler_sigma_reset = min(max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0), doppler_cap_reset)
        doppler_rho_reset = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
        doppler_init_std = (
            doppler_sigma_reset / math.sqrt(max(1.0 - doppler_rho_reset * doppler_rho_reset, 1.0e-6))
            if _doppler_precomp_enabled_from_cfg(cfg) and doppler_cap_reset > 0.0 and doppler_sigma_reset > 0.0
            else 0.0
        )
        stored_base_episode_idx = getattr(self, "_native_rollout_base_episode_idx_t", None)
        if torch.is_tensor(stored_base_episode_idx) and tuple(stored_base_episode_idx.shape) == tuple(tensor_state.episode_idx.shape):
            base_episode_idx = stored_base_episode_idx.to(device=runtime_device, dtype=torch.int32)
        else:
            base_episode_idx = tensor_state.episode_idx.to(device=runtime_device, dtype=torch.int32)
        base_episode_idx_cpu = base_episode_idx.detach().cpu().numpy().astype(np.int32, copy=False)
        for reset_ordinal in range(reset_row_count):
            for slot in range(int(self._num_envs)):
                gu_pos_t, gu_centers_t, gu_counts_t = self._sample_native_reset_gu_tape(
                    cfg,
                    generator=self._native_torch_rng,
                    device=runtime_device,
                )
                episode_idx = int(base_episode_idx_cpu[int(slot)]) + int(reset_ordinal) + 1
                uav_pos_t = self._sample_native_reset_uav_positions_tape(
                    cfg,
                    generator=self._native_torch_rng,
                    device=runtime_device,
                    gu_pos=gu_pos_t,
                    gu_cluster_centers=gu_centers_t,
                    gu_cluster_counts=gu_counts_t,
                    episode_idx=episode_idx,
                )
                uav_vel_t = self._sample_native_reset_uav_vel_tape(cfg, generator=self._native_torch_rng, device=runtime_device)
                reset_gu_pos_tape[reset_ordinal, slot].copy_(gu_pos_t)
                reset_uav_pos_tape[reset_ordinal, slot].copy_(uav_pos_t)
                reset_uav_vel_tape[reset_ordinal, slot].copy_(uav_vel_t)
                reset_gu_cluster_centers_tape[reset_ordinal, slot].zero_()
                reset_gu_cluster_counts_tape[reset_ordinal, slot].zero_()
                cluster_count = min(int(cluster_width), int(gu_centers_t.shape[0]), int(gu_counts_t.shape[0]))
                if cluster_count > 0:
                    reset_gu_cluster_centers_tape[reset_ordinal, slot, :cluster_count].copy_(gu_centers_t[:cluster_count])
                    reset_gu_cluster_counts_tape[reset_ordinal, slot, :cluster_count].copy_(gu_counts_t[:cluster_count].to(dtype=torch.float32))
            if int(cfg.num_gu) > 0:
                reset_arrival_base_scale_tape[reset_ordinal].copy_(
                    torch.rand(
                        (int(self._num_envs), int(cfg.num_gu)),
                        generator=self._native_torch_rng,
                        device=runtime_device,
                        dtype=torch.float32,
                    ).mul(float(arrival_high - arrival_low)).add(float(arrival_low))
                )
                max_choice = max(int(getattr(cfg, "hotspot_num_subsets", 0) or 0), 1)
                use_reset_queue_state = bool(getattr(cfg, "preload_enabled", False)) or traffic_model == "sticky_subset_hotspot"
                for slot in range(int(self._num_envs)):
                    arrival_base_scale_np = (
                        reset_arrival_base_scale_tape[reset_ordinal, slot]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    hotspot_mask_np = np.zeros((int(cfg.num_gu),), dtype=np.float32)
                    if use_reset_queue_state:
                        preload_draw = None
                        preload_choice = None
                        if bool(getattr(cfg, "preload_enabled", False)):
                            preload_draw = float(
                                torch.rand(
                                    (),
                                    generator=self._native_torch_rng,
                                    device=runtime_device,
                                    dtype=torch.float32,
                                ).item()
                            )
                            preload_choice = int(
                                torch.randint(
                                    low=0,
                                    high=max_choice,
                                    size=(1,),
                                    generator=self._native_torch_rng,
                                    device=runtime_device,
                                    dtype=torch.long,
                                ).item()
                            )
                        gu_pos_np = (
                            reset_gu_pos_tape[reset_ordinal, slot]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        uav_pos_np = (
                            reset_uav_pos_tape[reset_ordinal, slot]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        assoc_init = _associate_users_batch(
                            [self._slot_view(int(slot))],
                            gu_pos_batch=gu_pos_np[None, ...],
                            uav_pos_batch=uav_pos_np[None, ...],
                        )[0]
                        queue_state = _build_native_reset_queue_state(
                            cfg,
                            effective_task_arrival_rate=float(reset_effective_rate),
                            gu_pos=gu_pos_np,
                            uav_pos=uav_pos_np,
                            assoc_init=assoc_init,
                            preload_draw=preload_draw,
                            preload_choice=preload_choice,
                        )
                        hotspot_mask_np = np.asarray(queue_state["last_hotspot_mask"], dtype=np.float32)
                        member_mask_np = np.asarray(queue_state["hotspot_member_mask"], dtype=bool).reshape(-1, int(cfg.num_gu))
                        subset_count = min(int(member_mask_np.shape[0]), max_reset_hotspot_subsets)
                        reset_hotspot_subset_count_tape[reset_ordinal, slot] = int(subset_count)
                        reset_hotspot_active_idx_tape[reset_ordinal, slot] = int(queue_state["hotspot_active_idx"])
                        if subset_count > 0:
                            reset_hotspot_member_mask_tape[reset_ordinal, slot, :subset_count].copy_(
                                torch.as_tensor(
                                    member_mask_np[:subset_count].astype(np.float32, copy=False),
                                    dtype=torch.float32,
                                    device=runtime_device,
                                )
                            )
                        reset_gu_queue_tape[reset_ordinal, slot].copy_(
                            torch.as_tensor(queue_state["gu_queue"], dtype=torch.float32, device=runtime_device)
                        )
                        reset_uav_queue_tape[reset_ordinal, slot].copy_(
                            torch.as_tensor(queue_state["uav_queue"], dtype=torch.float32, device=runtime_device)
                        )
                        reset_sat_queue_tape[reset_ordinal, slot].copy_(
                            torch.as_tensor(queue_state["sat_queue"], dtype=torch.float32, device=runtime_device)
                        )
                    reset_arrival_rate_vec_tape[reset_ordinal, slot].copy_(
                        torch.as_tensor(
                            _build_native_reset_arrival_rate_vec(
                                cfg,
                                effective_task_arrival_rate=float(reset_effective_rate),
                                arrival_base_scale=arrival_base_scale_np,
                                hotspot_mask=hotspot_mask_np,
                            ),
                            dtype=torch.float32,
                            device=runtime_device,
                        )
                    )
                if deadline_jitter <= NORMALIZATION_DENOM_EPS:
                    reset_deadline_steps_tape[reset_ordinal].fill_(float(deadline_base))
                else:
                    d_low = max(deadline_base - deadline_jitter, 1.0)
                    d_high = max(deadline_base + deadline_jitter, d_low)
                    reset_deadline_steps_tape[reset_ordinal].copy_(
                        torch.rand(
                            (int(self._num_envs), int(cfg.num_gu)),
                            generator=self._native_torch_rng,
                            device=runtime_device,
                            dtype=torch.float32,
                        ).mul(float(d_high - d_low)).add(float(d_low))
                    )
            else:
                reset_arrival_base_scale_tape[reset_ordinal].zero_()
                reset_deadline_steps_tape[reset_ordinal].zero_()
                reset_arrival_rate_vec_tape[reset_ordinal].zero_()
            if doppler_init_std > 0.0 and int(cfg.num_uav) > 0 and int(cfg.num_sat) > 0:
                reset_doppler_residual_tape[reset_ordinal].copy_(
                    torch.randn(
                        (int(self._num_envs), int(cfg.num_uav), int(cfg.num_sat)),
                        generator=self._native_torch_rng,
                        device=runtime_device,
                        dtype=torch.float32,
                    ).mul(float(doppler_init_std)).clamp(-float(doppler_cap_reset), float(doppler_cap_reset))
                )
            else:
                reset_doppler_residual_tape[reset_ordinal].zero_()
            reset_episode_idx_tape[reset_ordinal].copy_(base_episode_idx + int(reset_ordinal) + 1)

        reset_followup_arrival_rate_tape = None
        reset_followup_arrival_tape = None
        reset_followup_hotspot_active_after_tape = None
        if traffic_model == "sticky_subset_hotspot" and int(cfg.num_gu) > 0 and arrival_rate_tape is not None:
            env_count = int(self._num_envs)
            gu_count = int(cfg.num_gu)
            reset_row_ids_t = torch.arange(reset_row_count, dtype=torch.long, device=runtime_device).view(reset_row_count, 1).expand(reset_row_count, env_count)
            env_ids_t = torch.arange(env_count, dtype=torch.long, device=runtime_device).view(1, env_count).expand(reset_row_count, env_count)
            active_t = reset_hotspot_active_idx_tape.to(dtype=torch.long)
            subset_counts_t = reset_hotspot_subset_count_tape.to(dtype=torch.long)
            member_mask_t = reset_hotspot_member_mask_tape.to(dtype=torch.float32)
            base_scale_t = reset_arrival_base_scale_tape.to(dtype=torch.float32)
            reset_followup_arrival_rate_tape = torch.empty(
                (reset_row_count, steps, env_count, gu_count),
                dtype=torch.float32,
                device=runtime_device,
            )
            reset_followup_hotspot_active_after_tape = torch.empty(
                (reset_row_count, steps, env_count),
                dtype=torch.int32,
                device=runtime_device,
            )
            rho = max(float(getattr(cfg, "hotspot_rho", 4.0) or 0.0), 0.0)
            preserve_mean = bool(getattr(cfg, "arrival_mean_preserve", True))
            on_prob = 1.0 / max(float(getattr(cfg, "hotspot_on_mean_steps", 15.0) or 15.0), 1.0)
            off_prob = 1.0 / max(float(getattr(cfg, "hotspot_off_mean_steps", 8.0) or 8.0), 1.0)
            max_subsets = int(max(int(member_mask_t.shape[2]), 1))
            for future_step in range(steps):
                valid_active_t = (active_t >= 0) & (active_t < subset_counts_t)
                active_clamped_t = torch.clamp(active_t, min=0, max=max_subsets - 1)
                step_hot_mask_t = member_mask_t[reset_row_ids_t, env_ids_t, active_clamped_t]
                step_hot_mask_t = torch.where(
                    valid_active_t.view(reset_row_count, env_count, 1),
                    step_hot_mask_t,
                    step_hot_mask_t * 0.0,
                )
                weights_t = base_scale_t * torch.where(
                    step_hot_mask_t > 0.5,
                    base_scale_t * 0.0 + float(rho),
                    base_scale_t * 0.0 + 1.0,
                )
                if preserve_mean:
                    weights_t = _torch_divide_or_default(weights_t, weights_t.mean(dim=2, keepdim=True))
                rate_t = weights_t * float(reset_effective_rate)
                reset_followup_arrival_rate_tape[:, future_step].copy_(rate_t)
                has_subsets_t = subset_counts_t > 0
                if bool(has_subsets_t.any().item()):
                    on_rand_t = torch.rand((reset_row_count, env_count), dtype=torch.float32, device=runtime_device, generator=self._native_torch_rng)
                    off_rand_t = torch.rand((reset_row_count, env_count), dtype=torch.float32, device=runtime_device, generator=self._native_torch_rng)
                    chosen_t = torch.randint(
                        low=0,
                        high=max_subsets,
                        size=(reset_row_count, env_count),
                        dtype=torch.long,
                        device=runtime_device,
                        generator=self._native_torch_rng,
                    ) % subset_counts_t.clamp_min(1)
                    update_t = has_subsets_t
                    turn_off_t = update_t & valid_active_t & (on_rand_t < float(on_prob))
                    inactive_t = update_t & (~valid_active_t)
                    turn_on_t = inactive_t & (off_rand_t < float(off_prob))
                    active_t = torch.where(turn_off_t, active_t * 0 - 1, active_t)
                    active_t = torch.where(turn_on_t, chosen_t, active_t)
                reset_followup_hotspot_active_after_tape[:, future_step].copy_(active_t.to(dtype=torch.int32))
            reset_followup_arrival_tape = (
                torch.poisson(reset_followup_arrival_rate_tape, generator=self._native_torch_rng).to(dtype=torch.float32)
                if bool(cfg.task_arrival_poisson)
                else reset_followup_arrival_rate_tape.clone()
            )

        runtime.write_random_rollout_tapes(
            arrival_tape=arrival_tape,
            arrival_rate_tape=arrival_rate_tape,
            hotspot_active_after_tape=hotspot_active_after_tape,
            reset_followup_arrival_tape=reset_followup_arrival_tape,
            reset_followup_arrival_rate_tape=reset_followup_arrival_rate_tape,
            reset_followup_hotspot_active_after_tape=reset_followup_hotspot_active_after_tape,
            hotspot_active_tape=hotspot_active_tape,
            hotspot_mask_tape=hotspot_mask_tape,
            fading_gain_tape=fading_gain_tape,
            doppler_noise_tape=doppler_noise_tape,
        )
        runtime.write_random_reset_rollout_tapes(
            gu_pos_tape=reset_gu_pos_tape,
            uav_pos_tape=reset_uav_pos_tape,
            uav_vel_tape=reset_uav_vel_tape,
            gu_cluster_centers_tape=reset_gu_cluster_centers_tape,
            gu_cluster_counts_tape=reset_gu_cluster_counts_tape,
            gu_queue_tape=reset_gu_queue_tape,
            uav_queue_tape=reset_uav_queue_tape,
            sat_queue_tape=reset_sat_queue_tape,
            arrival_base_scale_tape=reset_arrival_base_scale_tape,
            deadline_steps_tape=reset_deadline_steps_tape,
            doppler_residual_tape=reset_doppler_residual_tape,
            effective_arrival_rate_tape=reset_effective_rate_tape,
            arrival_rate_vec_tape=reset_arrival_rate_vec_tape,
            arrival_ref_tape=reset_arrival_ref_tape,
            episode_idx_tape=reset_episode_idx_tape,
            hotspot_active_idx_tape=reset_hotspot_active_idx_tape,
            hotspot_subset_count_tape=reset_hotspot_subset_count_tape,
            hotspot_member_mask_tape=reset_hotspot_member_mask_tape,
        )
        if torch.is_tensor(runtime.random.step_tensor):
            runtime.random.step_tensor.zero_()
        runtime.random.step = 0

    @staticmethod
    def _runtime_reset_rollout_tape_rows(runtime: StructuredGpuRolloutRuntime) -> int:
        random = runtime.random
        for field_name in (
            "reset_arrival_ref_rollout_tape",
            "reset_gu_pos_rollout_tape",
            "reset_episode_idx_rollout_tape",
        ):
            value = getattr(random, field_name, None)
            if torch.is_tensor(value) and int(value.ndim) >= 1:
                return max(int(value.shape[0]), 0)
        return 0

    @staticmethod
    def _restore_runtime_random_tensor(random: Any, field_name: str, value: torch.Tensor | None) -> None:
        if value is None:
            setattr(random, field_name, None)
            return
        current = getattr(random, field_name, None)
        if torch.is_tensor(current) and tuple(current.shape) == tuple(value.shape) and current.dtype == value.dtype and current.device == value.device:
            current.copy_(value)
        else:
            setattr(random, field_name, value)

    @staticmethod
    def _preserve_runtime_random_tape_prefix(random: Any, field_name: str, old_value: torch.Tensor | None) -> None:
        if not torch.is_tensor(old_value):
            return
        new_value = getattr(random, field_name, None)
        if not torch.is_tensor(new_value) or int(new_value.ndim) <= 0 or int(old_value.ndim) <= 0:
            return
        rows = min(int(old_value.shape[0]), int(new_value.shape[0]))
        if rows <= 0:
            return
        new_value[:rows].copy_(old_value[:rows].to(device=new_value.device, dtype=new_value.dtype))

    def ensure_native_rollout_reset_tape_capacity(self, *, chunk_rows: int | None = None) -> bool:
        runtime = self.native_rollout_runtime
        if runtime is None:
            return False
        step_tensor = runtime.random.step_tensor
        if torch.is_tensor(step_tensor) and int(step_tensor.numel()) > 1:
            reset_used = bool(int(step_tensor[1].item()) != 0)
            if not reset_used:
                return False
            step_tensor[1].zero_()
        reset_count = runtime.random.reset_count
        if not torch.is_tensor(reset_count) or int(reset_count.numel()) <= 0:
            return False
        current_rows = self._runtime_reset_rollout_tape_rows(runtime)
        if current_rows <= 0:
            current_rows = 1
        max_used = int(reset_count.max().item())
        chunk_rows_i = max(int(chunk_rows) if chunk_rows is not None else 1, 1)
        needed_rows = max(max_used + 1, 1)
        if needed_rows <= current_rows:
            return False
        needed_rows = max(needed_rows, current_rows + chunk_rows_i)
        arrival_tape = runtime.random.arrival_rollout_tape
        capacity = int(arrival_tape.shape[0]) if torch.is_tensor(arrival_tape) and int(arrival_tape.ndim) >= 1 else int(runtime.history.capacity)
        capacity = max(capacity, 1)

        base_fields = (
            "arrival_rollout_tape",
            "arrival_rate_rollout_tape",
            "hotspot_active_after_rollout_tape",
            "hotspot_active_rollout_tape",
            "hotspot_mask_rollout_tape",
            "fading_gain_rollout_tape",
            "doppler_noise_rollout_tape",
        )
        reset_fields = (
            "reset_gu_pos_rollout_tape",
            "reset_uav_pos_rollout_tape",
            "reset_uav_vel_rollout_tape",
            "reset_gu_cluster_centers_rollout_tape",
            "reset_gu_cluster_counts_rollout_tape",
            "reset_gu_queue_rollout_tape",
            "reset_uav_queue_rollout_tape",
            "reset_sat_queue_rollout_tape",
            "reset_arrival_base_scale_rollout_tape",
            "reset_deadline_steps_rollout_tape",
            "reset_doppler_residual_rollout_tape",
            "reset_effective_arrival_rate_rollout_tape",
            "reset_arrival_rate_vec_rollout_tape",
            "reset_arrival_ref_rollout_tape",
            "reset_followup_arrival_rollout_tape",
            "reset_followup_arrival_rate_rollout_tape",
            "reset_followup_hotspot_active_after_rollout_tape",
            "reset_episode_idx_rollout_tape",
            "reset_hotspot_active_idx_rollout_tape",
            "reset_hotspot_subset_count_rollout_tape",
            "reset_hotspot_member_mask_rollout_tape",
        )
        saved_base = {
            name: (getattr(runtime.random, name).detach().clone() if torch.is_tensor(getattr(runtime.random, name, None)) else None)
            for name in base_fields
        }
        saved_reset = {name: getattr(runtime.random, name, None) for name in reset_fields}
        saved_step_tensor = runtime.random.step_tensor.detach().clone() if torch.is_tensor(runtime.random.step_tensor) else None
        saved_reset_count = reset_count.detach().clone()
        saved_step = int(runtime.random.step)

        self._prepare_runtime_rollout_random_tapes(capacity=capacity, reset_rows=needed_rows)

        for name, value in saved_base.items():
            self._restore_runtime_random_tensor(runtime.random, name, value)
        for name, old_value in saved_reset.items():
            self._preserve_runtime_random_tape_prefix(runtime.random, name, old_value)
        if saved_step_tensor is not None and torch.is_tensor(runtime.random.step_tensor):
            runtime.random.step_tensor.copy_(saved_step_tensor.to(device=runtime.random.step_tensor.device, dtype=runtime.random.step_tensor.dtype))
        if torch.is_tensor(runtime.random.reset_count):
            runtime.random.reset_count.copy_(saved_reset_count.to(device=runtime.random.reset_count.device, dtype=runtime.random.reset_count.dtype))
        runtime.random.step = saved_step
        runtime.main.native_cuda_abi = self._build_native_cuda_runtime_abi(runtime)
        return True

    def _sync_runtime_step_scalars_from_envs(
        self,
        indices: Sequence[int],
        envs: Sequence[Any],
        *,
        sync_t_and_orbit: bool,
    ) -> None:
        selected_indices = [int(index) for index in indices]
        if not selected_indices:
            return
        if len(envs) != len(selected_indices):
            raise ValueError(f"Expected {len(selected_indices)} envs, got {len(envs)}.")
        runtime_tensor_state = self._runtime_tensor_state
        kernel_device = torch.device("cpu") if self._tensor_device is None else self._tensor_device
        selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=kernel_device)
        prev_queue_sum_gu = np.asarray(
            [float(getattr(env, "prev_queue_sum_gu", np.sum(env.gu_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_queue_sum_uav = np.asarray(
            [float(getattr(env, "prev_queue_sum_uav", np.sum(env.uav_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_queue_sum_sat = np.asarray(
            [float(getattr(env, "prev_queue_sum_sat", np.sum(env.sat_queue, dtype=np.float32))) for env in envs],
            dtype=np.float32,
        )
        prev_q_norm_active = np.asarray(
            [float(getattr(env, "prev_q_norm_active", 0.0)) for env in envs],
            dtype=np.float32,
        )
        prev_gu_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_gu_queue_vec", np.asarray(env.gu_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        prev_uav_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_uav_queue_vec", np.asarray(env.uav_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        prev_sat_queue_vec = np.stack(
            [
                np.asarray(getattr(env, "prev_sat_queue_vec", np.asarray(env.sat_queue, dtype=np.float32)), dtype=np.float32)
                for env in envs
            ],
            axis=0,
        )
        global_step = np.asarray([int(getattr(env, "global_step", 0)) for env in envs], dtype=np.int32)
        runtime_tensor_state.prev_queue_sum_gu[selected_tensor] = _as_kernel_tensor(prev_queue_sum_gu, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_queue_sum_uav[selected_tensor] = _as_kernel_tensor(prev_queue_sum_uav, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_queue_sum_sat[selected_tensor] = _as_kernel_tensor(prev_queue_sum_sat, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_q_norm_active[selected_tensor] = _as_kernel_tensor(prev_q_norm_active, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_gu_queue_vec[selected_tensor] = _as_kernel_tensor(prev_gu_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_uav_queue_vec[selected_tensor] = _as_kernel_tensor(prev_uav_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.prev_sat_queue_vec[selected_tensor] = _as_kernel_tensor(prev_sat_queue_vec, dtype=torch.float32, device=kernel_device)
        runtime_tensor_state.global_step[selected_tensor] = _as_kernel_tensor(global_step, dtype=torch.int32, device=kernel_device)
        if bool(sync_t_and_orbit):
            t = np.asarray([int(getattr(env, "t", 0)) for env in envs], dtype=np.int32)
            runtime_tensor_state.t[selected_tensor] = _as_kernel_tensor(t, dtype=torch.int32, device=kernel_device)
            self._sync_runtime_orbit_state_from_t(selected_indices, t_values=t)

    def _native_reset_generator(self, seed: int | None) -> torch.Generator:
        device = torch.device("cpu") if self._tensor_device is None else torch.device(self._tensor_device)
        if seed is None:
            return self._native_torch_rng
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    def _sample_native_reset_gu_tape(
        self,
        cfg,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_gu = int(cfg.num_gu)
        raw_num_clusters = getattr(cfg, "gu_init_num_clusters", None)
        num_clusters = max(1, int(cfg.num_gu // 5)) if raw_num_clusters is None else max(1, int(raw_num_clusters))
        if num_gu <= 0:
            return (
                torch.zeros((0, 2), dtype=torch.float32, device=device),
                torch.zeros((num_clusters, 2), dtype=torch.float32, device=device),
                torch.zeros((num_clusters,), dtype=torch.int32, device=device),
            )
        map_size = float(cfg.map_size)
        cluster_std = max(float(getattr(cfg, "gu_init_cluster_std", 80.0) or 0.0), 0.0)
        center_min_dist = max(float(getattr(cfg, "gu_init_cluster_center_min_dist", 0.0) or 0.0), 0.0)
        center_low = 0.1 * map_size
        center_span = 0.8 * map_size
        if center_min_dist <= 0.0 or num_clusters <= 1:
            centers = (
                torch.rand((num_clusters, 2), generator=generator, device=device, dtype=torch.float32)
                .mul_(center_span)
                .add_(center_low)
            )
        else:
            candidate_count = 512
            candidates = (
                torch.rand((candidate_count, num_clusters, 2), generator=generator, device=device, dtype=torch.float32)
                .mul_(center_span)
                .add_(center_low)
            )
            diff_t = candidates[:, :, None, :] - candidates[:, None, :, :]
            eye_t = torch.eye(num_clusters, dtype=torch.bool, device=device).view(1, num_clusters, num_clusters)
            dist_t = torch.linalg.vector_norm(diff_t, dim=-1).masked_fill(eye_t, float("inf"))
            min_dist_t = dist_t.amin(dim=(1, 2))
            valid_t = min_dist_t >= float(center_min_dist)
            first_valid_t = torch.argmax(valid_t.to(dtype=torch.int64), dim=0)
            best_t = torch.argmax(min_dist_t, dim=0)
            selected_t = torch.where(valid_t.any(), first_valid_t, best_t)
            centers = candidates.index_select(0, selected_t.reshape(1)).squeeze(0)
        cluster_ids = torch.randint(num_clusters, (num_gu,), generator=generator, device=device, dtype=torch.long)
        offsets = torch.randn((num_gu, 2), generator=generator, device=device, dtype=torch.float32).mul_(cluster_std)
        gu_pos = centers.index_select(0, cluster_ids).add_(offsets).clamp_(0.0, map_size)
        counts = torch.bincount(cluster_ids, minlength=num_clusters).to(dtype=torch.int32)
        return gu_pos, centers, counts

    def _sample_native_reset_uav_positions_tape(
        self,
        cfg,
        *,
        generator: torch.Generator,
        device: torch.device,
        gu_pos: torch.Tensor,
        gu_cluster_centers: torch.Tensor | None = None,
        gu_cluster_counts: torch.Tensor | None = None,
        episode_idx: int,
    ) -> torch.Tensor:
        num_uav = int(cfg.num_uav)
        if num_uav <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=device)
        map_size = float(cfg.map_size)
        spawn_mode = str(getattr(cfg, "uav_spawn_mode", "default") or "default").strip().lower()
        if spawn_mode == "gu_centroid":
            if int(cfg.num_gu) > 0 and tuple(gu_pos.shape) == (int(cfg.num_gu), 2):
                center = gu_pos.to(dtype=torch.float32).mean(dim=0)
            else:
                center = torch.full((2,), map_size * 0.5, dtype=torch.float32, device=device)
            return center.view(1, 2).expand(num_uav, 2).clone().clamp_(0.0, map_size)
        if spawn_mode == "gu_cluster_centers":
            centers_t = (
                torch.zeros((0, 2), dtype=torch.float32, device=device)
                if gu_cluster_centers is None
                else gu_cluster_centers.to(device=device, dtype=torch.float32)
            )
            counts_t = (
                torch.zeros((0,), dtype=torch.float32, device=device)
                if gu_cluster_counts is None
                else gu_cluster_counts.to(device=device, dtype=torch.float32).reshape(-1)
            )
            cluster_count = min(
                int(centers_t.shape[0]) if centers_t.ndim == 2 and int(centers_t.shape[1]) == 2 else 0,
                int(counts_t.numel()),
            )
            if cluster_count > 0:
                counts_valid_t = counts_t[:cluster_count]
                valid_cluster_t = torch.nonzero(counts_valid_t > 0.0, as_tuple=False).flatten()
                if int(valid_cluster_t.numel()) > 0:
                    order_rel_t = torch.argsort(counts_valid_t.index_select(0, valid_cluster_t), descending=True)
                    order_t = valid_cluster_t.index_select(0, order_rel_t)
                    selected_t = order_t.index_select(
                        0,
                        torch.arange(num_uav, dtype=torch.long, device=device) % int(order_t.numel()),
                    )
                    return centers_t[:cluster_count].index_select(0, selected_t).clone().clamp_(0.0, map_size)
            if int(cfg.num_gu) > 0 and tuple(gu_pos.shape) == (int(cfg.num_gu), 2):
                center = gu_pos.to(dtype=torch.float32).mean(dim=0)
            else:
                center = torch.full((2,), map_size * 0.5, dtype=torch.float32, device=device)
            return center.view(1, 2).expand(num_uav, 2).clone().clamp_(0.0, map_size)
        if bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
            margin_steps = max(float(getattr(cfg, "uav_init_boundary_margin_steps", 0.0) or 0.0), 0.0)
            margin = margin_steps * float(cfg.v_max) * float(cfg.tau0)
            low = max(0.0, margin)
            high = min(map_size, map_size - margin)
            if high < low:
                low = 0.0
                high = map_size
            min_spacing = max(float(getattr(cfg, "uav_init_min_spacing", cfg.d_safe) or 0.0), 0.0)
            max_tries = max(int(getattr(cfg, "uav_init_max_tries", 200) or 200), 1)
            positions = torch.zeros((num_uav, 2), dtype=torch.float32, device=device)
            for uav_index in range(num_uav):
                placed = False
                for _ in range(max_tries):
                    candidate = torch.rand((2,), generator=generator, device=device, dtype=torch.float32)
                    candidate = candidate.mul(float(high - low)).add(float(low))
                    if uav_index > 0:
                        dist = torch.linalg.norm(positions[:uav_index] - candidate.view(1, 2), dim=1)
                        if not bool(torch.all(dist >= (min_spacing - 1.0e-6)).item()):
                            continue
                    positions[uav_index].copy_(candidate)
                    placed = True
                    break
                if not placed:
                    raise RuntimeError(
                        "native reset tape could not sample UAV positions satisfying spacing constraints."
                    )
            return positions
        use_curriculum_spawn = ablation_flag(
            cfg,
            "use_curriculum_spawn",
            fallback_attr="uav_spawn_curriculum_enabled",
            default=False,
        )
        if not use_curriculum_spawn:
            return torch.rand((num_uav, 2), generator=generator, device=device, dtype=torch.float32).mul_(map_size)
        steps = int(getattr(cfg, "uav_spawn_curriculum_steps", 0) or 0)
        progress = 1.0 if steps <= 0 else min(1.0, float(max(int(episode_idx) - 1, 0)) / float(steps))
        if progress >= 1.0 and bool(getattr(cfg, "uav_spawn_full_random_final", True)):
            return torch.rand((num_uav, 2), generator=generator, device=device, dtype=torch.float32).mul_(map_size)
        radius_start = max(float(getattr(cfg, "uav_spawn_radius_start", 0.0) or 0.0), 0.0)
        radius_end_cfg = getattr(cfg, "uav_spawn_radius_end", None)
        radius_end = map_size * 0.5 if radius_end_cfg is None else float(radius_end_cfg)
        radius = radius_start + (max(radius_end, radius_start) - radius_start) * progress
        if int(cfg.num_gu) > 0 and tuple(gu_pos.shape) == (int(cfg.num_gu), 2):
            center_idx = int(torch.randint(int(cfg.num_gu), (1,), generator=generator, device=device).item())
            center = gu_pos[center_idx]
        else:
            center = torch.full((2,), map_size * 0.5, dtype=torch.float32, device=device)
        angles = torch.rand((num_uav,), generator=generator, device=device, dtype=torch.float32).mul_(2.0 * math.pi)
        radii = torch.rand((num_uav,), generator=generator, device=device, dtype=torch.float32).sqrt_().mul_(radius)
        offsets = torch.stack([torch.cos(angles).mul(radii), torch.sin(angles).mul(radii)], dim=1)
        return center.view(1, 2).add(offsets).clamp_(0.0, map_size)

    def _sample_native_reset_uav_vel_tape(
        self,
        cfg,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        num_uav = int(cfg.num_uav)
        if num_uav <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=device)
        if not bool(getattr(cfg, "uav_safe_random_init_enabled", False)):
            return torch.zeros((num_uav, 2), dtype=torch.float32, device=device)
        speed_frac = max(float(getattr(cfg, "uav_init_speed_frac", 0.0) or 0.0), 0.0)
        speed = min(speed_frac, 1.0) * float(cfg.v_max)
        if speed <= 0.0:
            return torch.zeros((num_uav, 2), dtype=torch.float32, device=device)
        angles = torch.rand((num_uav,), generator=generator, device=device, dtype=torch.float32).mul_(2.0 * math.pi)
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).mul_(float(speed))

    def _build_native_reset_tapes(
        self,
        selected: Sequence[int],
        seeds: Sequence[int | None],
    ) -> list[dict[str, Any]] | None:
        runtime = self.native_rollout_runtime
        if runtime is None:
            return None
        cfg = self._cfg
        device = torch.device(runtime.device)
        tapes: list[dict[str, Any]] = []
        traffic_level, traffic_ratio, effective_task_arrival_rate, arrival_ref = _traffic_level_state_from_cfg(cfg)
        hetero = max(float(getattr(cfg, "arrival_base_hetero", 0.0) or 0.0), 0.0)
        low = max(1.0 - hetero, 1.0e-3)
        high = max(1.0 + hetero, low)
        base_steps = max(float(getattr(cfg, "deadline_base_steps", 4.0) or 0.0), 1.0)
        jitter_steps = max(float(getattr(cfg, "deadline_jitter_steps", 0.0) or 0.0), 0.0)
        doppler_cap = _doppler_residual_cap_hz_from_cfg(cfg)
        doppler_sigma = min(max(float(getattr(cfg, "doppler_residual_sigma_hz", 0.0) or 0.0), 0.0), doppler_cap)
        doppler_rho = float(np.clip(float(getattr(cfg, "doppler_residual_ar_rho", 0.98) or 0.98), 0.0, 0.9999))
        doppler_init_std = (
            doppler_sigma / math.sqrt(max(1.0 - doppler_rho * doppler_rho, 1.0e-6))
            if _doppler_precomp_enabled_from_cfg(cfg) and doppler_cap > 0.0 and doppler_sigma > 0.0
            else 0.0
        )
        for slot, seed in zip(selected, seeds):
            env = self._slot_view(int(slot))
            episode_idx = int(getattr(env, "episode_idx", 0)) + 1
            generator = self._native_reset_generator(seed)
            gu_pos_t, gu_centers_t, gu_counts_t = self._sample_native_reset_gu_tape(
                cfg,
                generator=generator,
                device=device,
            )
            uav_pos_t = self._sample_native_reset_uav_positions_tape(
                cfg,
                generator=generator,
                device=device,
                gu_pos=gu_pos_t,
                gu_cluster_centers=gu_centers_t,
                gu_cluster_counts=gu_counts_t,
                episode_idx=episode_idx,
            )
            uav_vel_t = self._sample_native_reset_uav_vel_tape(cfg, generator=generator, device=device)
            if int(cfg.num_gu) > 0:
                arrival_base_scale_t = torch.rand(
                    (int(cfg.num_gu),),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                ).mul(float(high - low)).add(float(low))
                if jitter_steps <= NORMALIZATION_DENOM_EPS:
                    deadline_t = torch.full((int(cfg.num_gu),), base_steps, dtype=torch.float32, device=device)
                else:
                    d_low = max(base_steps - jitter_steps, 1.0)
                    d_high = max(base_steps + jitter_steps, d_low)
                    deadline_t = torch.rand(
                        (int(cfg.num_gu),),
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    ).mul(float(d_high - d_low)).add(float(d_low))
            else:
                arrival_base_scale_t = torch.zeros((0,), dtype=torch.float32, device=device)
                deadline_t = torch.zeros((0,), dtype=torch.float32, device=device)
            if doppler_init_std > 0.0 and int(cfg.num_uav) > 0 and int(cfg.num_sat) > 0:
                doppler_t = torch.randn(
                    (int(cfg.num_uav), int(cfg.num_sat)),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                ).mul(float(doppler_init_std)).clamp_(-float(doppler_cap), float(doppler_cap))
            else:
                doppler_t = torch.zeros((int(cfg.num_uav), int(cfg.num_sat)), dtype=torch.float32, device=device)
            tapes.append(
                {
                    "episode_idx": episode_idx,
                    "traffic_level": traffic_level,
                    "traffic_level_ratio": traffic_ratio,
                    "effective_task_arrival_rate": effective_task_arrival_rate,
                    "arrival_ref_bits_per_step": arrival_ref,
                    "gu_pos": gu_pos_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "gu_cluster_centers": gu_centers_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "gu_cluster_counts": gu_counts_t.detach().cpu().numpy().astype(np.int32, copy=True),
                    "uav_pos": uav_pos_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "uav_vel": uav_vel_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "arrival_base_scale": arrival_base_scale_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "gu_deadline_steps": deadline_t.detach().cpu().numpy().astype(np.float32, copy=True),
                    "doppler_residual": doppler_t.detach().cpu().numpy().astype(np.float32, copy=True),
                }
            )
        return tapes

    def _build_native_reset_state(
        self,
        slot: int,
        seed: int | None,
        *,
        reset_tape: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self._slot_view(int(slot))
        cfg = env.cfg
        state = env.export_runtime_state()

        if seed is None:
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(env.rng.bit_generator.state)
        else:
            rng = np.random.default_rng(int(seed))

        eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
        eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
        eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
        eta_max = max(eta_min, eta_max)
        adaptive_enabled = bool(getattr(cfg, "avoidance_adaptive_enabled", False))
        prev_steps = int(getattr(env, "_episode_step_count", 0))
        prev_collisions = int(getattr(env, "_episode_collision_count", 0))
        prev_episode_collision_rate = float(getattr(env, "prev_episode_collision_rate", 0.0))
        avoidance_collision_rate_ema = float(getattr(env, "avoidance_collision_rate_ema", 0.0))
        if not adaptive_enabled:
            avoidance_eta_eff = float(np.clip(float(cfg.avoidance_eta), eta_min, eta_max))
        elif prev_steps > 0:
            prev_episode_collision_rate = float(prev_collisions) / float(prev_steps)
            beta = float(getattr(cfg, "avoidance_adaptive_ema_beta", 0.9) or 0.9)
            beta = float(np.clip(beta, 0.0, 0.9999))
            avoidance_collision_rate_ema = (
                beta * float(getattr(env, "avoidance_collision_rate_ema", 0.0))
                + (1.0 - beta) * prev_episode_collision_rate
            )
            target = float(getattr(cfg, "avoidance_collision_target", 0.05) or 0.05)
            gain = float(getattr(cfg, "avoidance_adaptive_gain", 1.0) or 1.0)
            eta_cur = float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta))
            avoidance_eta_eff = float(
                np.clip(eta_cur + gain * (avoidance_collision_rate_ema - target) * float(cfg.a_max), eta_min, eta_max)
            )
        else:
            avoidance_eta_eff = float(
                np.clip(float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta)), eta_min, eta_max)
            )

        traffic_level, traffic_ratio, effective_task_arrival_rate, arrival_ref = _traffic_level_state_from_cfg(cfg)
        episode_idx = int(getattr(env, "episode_idx", 0)) + 1
        if reset_tape is not None:
            episode_idx = int(reset_tape["episode_idx"])
            traffic_level = int(reset_tape["traffic_level"])
            traffic_ratio = float(reset_tape["traffic_level_ratio"])
            effective_task_arrival_rate = float(reset_tape["effective_task_arrival_rate"])
            arrival_ref = reward_ratio_denominator_scalar(
                float(reset_tape["arrival_ref_bits_per_step"]),
                name="arrival_ref_bits_per_step",
            )
            gu_pos = np.asarray(reset_tape["gu_pos"], dtype=np.float32)
            gu_cluster_centers = np.asarray(reset_tape["gu_cluster_centers"], dtype=np.float32)
            gu_cluster_counts = np.asarray(reset_tape["gu_cluster_counts"], dtype=np.int32)
            uav_pos = np.asarray(reset_tape["uav_pos"], dtype=np.float32)
            uav_vel = np.asarray(reset_tape["uav_vel"], dtype=np.float32)
        else:
            raw_num_clusters = getattr(cfg, "gu_init_num_clusters", None)
            num_clusters = max(1, int(cfg.num_gu // 5)) if raw_num_clusters is None else max(1, int(raw_num_clusters))
            cluster_std = max(float(getattr(cfg, "gu_init_cluster_std", 80.0) or 0.0), 0.0)
            center_min_dist = max(float(getattr(cfg, "gu_init_cluster_center_min_dist", 0.0) or 0.0), 0.0)
            gu_pos, gu_cluster_centers, gu_cluster_counts = thomas_cluster_process(
                int(cfg.num_gu),
                float(cfg.map_size),
                num_clusters=num_clusters,
                cluster_std=cluster_std,
                center_min_dist=center_min_dist,
                rng=rng,
                return_metadata=True,
            )
            uav_pos = _sample_uav_positions_native(
                cfg,
                rng,
                gu_pos=gu_pos,
                gu_cluster_centers=gu_cluster_centers,
                gu_cluster_counts=gu_cluster_counts,
                episode_idx=episode_idx,
            )
            uav_vel = _sample_uav_initial_velocities_native(cfg, rng)
        gu_queue = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        uav_queue = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        sat_queue = np.zeros((int(cfg.num_sat),), dtype=np.float32)
        assoc_init = _associate_users_batch([env], gu_pos_batch=gu_pos[None, ...], uav_pos_batch=uav_pos[None, ...])[0]

        hetero = max(float(getattr(cfg, "arrival_base_hetero", 0.0) or 0.0), 0.0)
        low = max(1.0 - hetero, 1.0e-3)
        high = max(1.0 + hetero, low)
        if reset_tape is not None:
            arrival_base_scale = np.asarray(reset_tape["arrival_base_scale"], dtype=np.float32)
        elif int(cfg.num_gu) > 0:
            arrival_base_scale = rng.uniform(low, high, size=(int(cfg.num_gu),)).astype(np.float32, copy=False)
        else:
            arrival_base_scale = np.zeros((0,), dtype=np.float32)
        queue_state = _build_native_reset_queue_state(
            cfg,
            effective_task_arrival_rate=effective_task_arrival_rate,
            gu_pos=gu_pos,
            uav_pos=uav_pos,
            assoc_init=assoc_init,
            rng=rng,
        )
        gu_queue = np.asarray(queue_state["gu_queue"], dtype=np.float32)
        uav_queue = np.asarray(queue_state["uav_queue"], dtype=np.float32)
        sat_queue = np.asarray(queue_state["sat_queue"], dtype=np.float32)
        hotspot_subsets = queue_state["hotspot_subsets"]
        hotspot_member_mask = np.asarray(queue_state["hotspot_member_mask"], dtype=bool)
        hotspot_active_idx = int(queue_state["hotspot_active_idx"])
        last_hotspot_index = int(queue_state["last_hotspot_index"])
        last_hotspot_mask = np.asarray(queue_state["last_hotspot_mask"], dtype=np.float32)
        reset_arrival_rate_vec = _build_native_reset_arrival_rate_vec(
            cfg,
            effective_task_arrival_rate=effective_task_arrival_rate,
            arrival_base_scale=arrival_base_scale,
            hotspot_mask=last_hotspot_mask,
        )

        gu_deadline_steps = (
            np.asarray(reset_tape["gu_deadline_steps"], dtype=np.float32)
            if reset_tape is not None
            else _sample_deadline_steps_native(cfg, rng)
        )
        bw_acc_init = arrival_ref / max(float(cfg.num_gu), 1.0)
        bw_rel_init = arrival_ref / max(float(cfg.num_uav), 1.0)
        bw_sat_init = arrival_ref / _bw_weighted_workload_sat_active_ref_count(cfg)
        last_avoidance_eta_exec = float(getattr(env, "last_avoidance_eta_exec", cfg.avoidance_eta))

        state["episode_idx"] = episode_idx
        state["traffic_level"] = traffic_level
        state["traffic_level_ratio"] = traffic_ratio
        state["effective_task_arrival_rate"] = effective_task_arrival_rate
        state["arrival_ref_bits_per_step"] = arrival_ref
        state["avoidance_eta_eff"] = avoidance_eta_eff
        state["avoidance_collision_rate_ema"] = avoidance_collision_rate_ema
        state["prev_episode_collision_rate"] = prev_episode_collision_rate
        state["t"] = 0
        state["_episode_collision_count"] = 0
        state["_episode_step_count"] = 0
        state["gu_pos"] = np.asarray(gu_pos, dtype=np.float32)
        state["gu_cluster_centers"] = np.asarray(gu_cluster_centers, dtype=np.float32)
        state["gu_cluster_counts"] = np.asarray(gu_cluster_counts, dtype=np.int32)
        state["uav_pos"] = np.asarray(uav_pos, dtype=np.float32)
        state["uav_vel"] = np.asarray(uav_vel, dtype=np.float32)
        state["uav_energy"] = np.full((int(cfg.num_uav),), float(cfg.uav_energy_init), dtype=np.float32)
        state["last_policy_accel"] = np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
        state["last_exec_accel"] = np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
        state["last_intervention_norm_uav"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_close_risk_uav"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_danger_imitation_mask"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["gu_queue"] = np.asarray(gu_queue, dtype=np.float32)
        state["uav_queue"] = np.asarray(uav_queue, dtype=np.float32)
        state["sat_queue"] = np.asarray(sat_queue, dtype=np.float32)
        state["last_association"] = np.full((int(cfg.num_gu),), -1, dtype=np.int32)
        state["prev_association"] = np.full((int(cfg.num_gu),), -1, dtype=np.int32)
        state["_arrival_base_scale"] = np.asarray(arrival_base_scale, dtype=np.float32)
        state["_hotspot_subsets"] = [np.asarray(subset, dtype=np.int32) for subset in hotspot_subsets]
        state["_hotspot_member_mask"] = np.asarray(hotspot_member_mask, dtype=bool)
        state["_hotspot_active_idx"] = int(hotspot_active_idx)
        state["last_hotspot_index"] = int(last_hotspot_index)
        state["last_gu_arrival_rate_vec"] = np.asarray(reset_arrival_rate_vec, dtype=np.float32)
        state["last_hotspot_mask"] = np.asarray(last_hotspot_mask, dtype=np.float32)
        prev_queue_sum_gu = float(np.sum(state["gu_queue"]))
        prev_queue_sum_uav = float(np.sum(state["uav_queue"]))
        prev_queue_sum_sat = float(np.sum(state["sat_queue"]))
        prev_queue_sum_active = float(prev_queue_sum_gu + prev_queue_sum_uav)
        prev_queue_sum_total = float(prev_queue_sum_active + prev_queue_sum_sat)
        queue_norm_k = normalize_scale(float(getattr(cfg, "queue_norm_K", 1.0) or 1.0))
        queue_norm_arrival_floor = float(getattr(cfg, "queue_norm_arrival_floor", 0.0) or 0.0)
        if queue_norm_arrival_floor <= 0.0:
            queue_norm_arrival_floor = (
                float(effective_task_arrival_rate) * float(cfg.num_gu) * float(cfg.tau0)
            )
        prev_queue_norm_ref = reward_ratio_denominator_scalar(
            max(float(arrival_ref), queue_norm_arrival_floor),
            name="queue arrival normalization reference",
        )
        prev_queue_norm_scale = queue_norm_k * prev_queue_norm_ref
        state["prev_queue_sum"] = prev_queue_sum_total
        state["prev_queue_sum_active"] = prev_queue_sum_active
        state["prev_queue_sum_gu"] = prev_queue_sum_gu
        state["prev_queue_sum_uav"] = prev_queue_sum_uav
        state["prev_queue_sum_sat"] = prev_queue_sum_sat
        state["prev_gu_queue_vec"] = np.asarray(gu_queue, dtype=np.float32).copy()
        state["prev_uav_queue_vec"] = np.asarray(uav_queue, dtype=np.float32).copy()
        state["prev_sat_queue_vec"] = np.asarray(sat_queue, dtype=np.float32).copy()
        state["prev_arrival_sum"] = float(arrival_ref)
        state["prev_q_norm_active"] = float(np.clip(prev_queue_sum_active / prev_queue_norm_scale, 0.0, 1.0))
        state["prev_centroid_dist_mean"] = _compute_centroid_dist_mean(cfg, gu_pos, gu_queue, uav_pos)
        state["prev_d_min"] = 0.0
        state["last_gu_outflow"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_uav_outflow"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_gu_arrival"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["gu_deadline_steps"] = np.asarray(gu_deadline_steps, dtype=np.float32)
        state["last_gu_urgency_risk"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_gu_downstream_pressure"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_gu_service_gap"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_gu_service_gap_risk"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_gu_deadline_age"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["last_gu_deadline_slack"] = np.asarray(gu_deadline_steps, dtype=np.float32).copy()
        state["last_gu_deadline_risk"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["gu_drop"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["gu_expired"] = np.zeros((int(cfg.num_gu),), dtype=np.float32)
        state["uav_drop"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["sat_drop"] = np.zeros((int(cfg.num_sat),), dtype=np.float32)
        state["last_access_interference_by_uav"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_bw_fraction_by_uav_gu"] = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
        state["last_gu_to_uav_inflow_by_uav"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_uav_to_sat_outflow_matrix"] = np.zeros((int(cfg.num_uav), int(cfg.num_sat)), dtype=np.float32)
        state["last_selected_mask_by_uav_sat"] = np.zeros((int(cfg.num_uav), int(cfg.num_sat)), dtype=np.float32)
        state["last_exec_bw_alloc"] = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
        state["last_exec_sat_select_mask"] = np.zeros((int(cfg.num_uav), int(cfg.sats_obs_max)), dtype=np.float32)
        sat_select_k = _sat_action_select_k_from_config(cfg)
        state["last_exec_sat_indices"] = np.full((int(cfg.num_uav), sat_select_k), -1, dtype=np.int64)
        state["last_energy_cost"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_sat_processed"] = np.zeros((int(cfg.num_sat),), dtype=np.float32)
        state["last_sat_incoming"] = np.zeros((int(cfg.num_sat),), dtype=np.float32)
        state["last_bw_align"] = 0.0
        state["last_sat_score"] = 0.0
        state["bw_weighted_workload_acc_ema_vec"] = np.full((int(cfg.num_gu),), bw_acc_init, dtype=np.float32)
        state["bw_weighted_workload_rel_ema_vec"] = np.full((int(cfg.num_uav),), bw_rel_init, dtype=np.float32)
        state["bw_weighted_workload_sat_ema_vec"] = np.full((int(cfg.num_sat),), bw_sat_init, dtype=np.float32)
        state["bw_weighted_workload_acc_ema"] = float(np.sum(state["bw_weighted_workload_acc_ema_vec"], dtype=np.float32))
        state["bw_weighted_workload_rel_ema"] = float(np.sum(state["bw_weighted_workload_rel_ema_vec"], dtype=np.float32))
        state["bw_weighted_workload_sat_ema"] = float(np.sum(state["bw_weighted_workload_sat_ema_vec"], dtype=np.float32))
        state["last_sat_selection"] = [[] for _ in range(int(cfg.num_uav))]
        state["last_sat_connection_counts"] = np.zeros((int(cfg.num_sat),), dtype=np.float32)
        state["last_connected_sat_count"] = 0.0
        state["last_connected_sat_dist_mean"] = 0.0
        state["last_connected_sat_dist_p95"] = 0.0
        state["last_connected_sat_elevation_deg_mean"] = 0.0
        state["last_connected_sat_elevation_deg_min"] = 0.0
        state["last_visible_raw_counts"] = np.zeros((int(cfg.num_uav),), dtype=np.int32)
        state["last_visible_kept_counts"] = np.zeros((int(cfg.num_uav),), dtype=np.int32)
        state["last_visible_raw_candidates"] = [[] for _ in range(int(cfg.num_uav))]
        state["last_visible_candidates"] = [[] for _ in range(int(cfg.num_uav))]
        state["last_visible_candidate_rank_values"] = [[] for _ in range(int(cfg.num_uav))]
        state["last_visible_candidate_scores"] = [[] for _ in range(int(cfg.num_uav))]
        state["last_visible_candidate_rank_gap_top1_top2"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_candidate_score_gap_top1_top2"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_candidate_dist_std"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_candidate_elevation_std"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_candidate_se_std"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_candidate_queue_std"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_visible_stats"] = {}
        state["last_arrival_rate"] = (
            float(np.mean(reset_arrival_rate_vec, dtype=np.float32))
            if int(cfg.num_gu) > 0
            else float(effective_task_arrival_rate)
        )
        state["_doppler_residual_state_hz"] = (
            np.asarray(reset_tape["doppler_residual"], dtype=np.float32)
            if reset_tape is not None
            else _initial_doppler_residual_state(cfg, rng)
        )
        state["last_filter_active_ratio"] = 0.0
        state["last_projected_delta_norm_mean"] = 0.0
        state["last_fallback_count"] = 0.0
        state["last_boundary_filter_count"] = 0.0
        state["last_pairwise_filter_count"] = 0.0
        state["last_pairwise_filter_active_ratio"] = 0.0
        state["last_pairwise_projected_delta_norm"] = 0.0
        state["last_pairwise_fallback_count"] = 0.0
        state["last_pairwise_candidate_infeasible_count"] = 0.0
        state["last_step_profile"] = {
            "dynamics_time_sec": 0.0,
            "orbit_visible_time_sec": 0.0,
            "assoc_access_time_sec": 0.0,
            "backhaul_queue_time_sec": 0.0,
            "reward_time_sec": 0.0,
            "obs_time_sec": 0.0,
            "state_time_sec": 0.0,
            "step_total_time_sec": 0.0,
        }
        state["last_assoc_centroid_dist_norms"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_sat_overlap_uav"] = np.zeros((int(cfg.num_uav),), dtype=np.float32)
        state["last_reward_parts"] = _initial_reward_parts_from_existing(
            state.get("last_reward_parts"),
            arrival_ref=arrival_ref,
            avoidance_eta_eff=avoidance_eta_eff,
            last_avoidance_eta_exec=last_avoidance_eta_exec,
            avoidance_collision_rate_ema=avoidance_collision_rate_ema,
            prev_episode_collision_rate=prev_episode_collision_rate,
        )
        state["rng_bit_generator_state"] = copy.deepcopy(rng.bit_generator.state)
        return state

    def _export_runtime_state_from_core_slot(self, index: int) -> dict[str, Any]:
        slot = int(index)
        tensor_state = self._runtime_tensor_state

        def _tensor_np(name: str, *, dtype) -> np.ndarray:
            return _compat_numpy_array(getattr(tensor_state, name)[slot], dtype=dtype).copy()

        def _tensor_scalar(name: str, *, dtype) -> float:
            return float(_compat_numpy_array(getattr(tensor_state, name)[slot], dtype=dtype).reshape(()))

        exported = copy.deepcopy(self._slot_state_payloads[slot])
        exported["rng_bit_generator_state"] = copy.deepcopy(self._slot_rngs[slot].bit_generator.state)
        exported["uav_pos"] = _tensor_np("uav_pos", dtype=np.float32)
        exported["uav_vel"] = _tensor_np("uav_vel", dtype=np.float32)
        exported["uav_energy"] = _tensor_np("uav_energy", dtype=np.float32)
        exported["uav_queue"] = _tensor_np("uav_queue", dtype=np.float32)
        exported["gu_pos"] = _tensor_np("gu_pos", dtype=np.float32)
        exported["gu_cluster_centers"] = _tensor_np("gu_cluster_centers", dtype=np.float32)
        exported["gu_cluster_counts"] = _tensor_np("gu_cluster_counts", dtype=np.float32)
        exported["gu_queue"] = _tensor_np("gu_queue", dtype=np.float32)
        exported["sat_queue"] = _tensor_np("sat_queue", dtype=np.float32)
        exported["prev_association"] = _tensor_np("prev_association", dtype=np.int32)
        exported["last_association"] = _tensor_np("last_association", dtype=np.int32)
        exported["last_sat_selection"] = _sat_selection_lists_from_matrix(
            _tensor_np("last_sat_selection_matrix", dtype=np.int64)
        )
        exported["last_sat_connection_counts"] = _tensor_np("last_sat_connection_counts", dtype=np.float32)
        exported["arrival_ref_bits_per_step"] = _tensor_scalar("arrival_ref_bits_per_step", dtype=np.float32)
        exported["effective_task_arrival_rate"] = _tensor_scalar("effective_task_arrival_rate", dtype=np.float32)
        exported["bw_weighted_workload_acc_ema_vec"] = _tensor_np("gu_workload_ema", dtype=np.float32)
        exported["bw_weighted_workload_rel_ema_vec"] = _tensor_np("uav_workload_ema", dtype=np.float32)
        exported["bw_weighted_workload_sat_ema_vec"] = _tensor_np("sat_workload_ema", dtype=np.float32)
        exported["bw_weighted_workload_acc_ema"] = float(np.sum(exported["bw_weighted_workload_acc_ema_vec"], dtype=np.float32))
        exported["bw_weighted_workload_rel_ema"] = float(np.sum(exported["bw_weighted_workload_rel_ema_vec"], dtype=np.float32))
        exported["bw_weighted_workload_sat_ema"] = float(np.sum(exported["bw_weighted_workload_sat_ema_vec"], dtype=np.float32))
        exported["last_gu_arrival_rate_vec"] = _tensor_np("last_gu_arrival_rate_vec", dtype=np.float32)
        exported["gu_deadline_steps"] = _tensor_np("gu_deadline_steps", dtype=np.float32)
        exported["last_gu_arrival"] = _tensor_np("last_gu_arrival", dtype=np.float32)
        exported["last_gu_outflow"] = _tensor_np("last_gu_outflow", dtype=np.float32)
        exported["gu_drop"] = _tensor_np("gu_drop", dtype=np.float32)
        exported["uav_drop"] = _tensor_np("uav_drop", dtype=np.float32)
        exported["sat_drop"] = _tensor_np("sat_drop", dtype=np.float32)
        exported["last_access_interference_by_uav"] = _tensor_np("last_access_interference_by_uav", dtype=np.float32)
        exported["last_bw_fraction_by_uav_gu"] = _tensor_np("last_bw_fraction_by_uav_gu", dtype=np.float32)
        exported["last_gu_to_uav_inflow_by_uav"] = _tensor_np("last_gu_to_uav_inflow_by_uav", dtype=np.float32)
        exported["last_uav_to_sat_outflow_matrix"] = _tensor_np("last_uav_to_sat_outflow_matrix", dtype=np.float32)
        exported["last_selected_mask_by_uav_sat"] = _tensor_np("last_selected_mask_by_uav_sat", dtype=np.float32)
        exported["last_sat_processed"] = _tensor_np("last_sat_processed", dtype=np.float32)
        exported["last_gu_urgency_risk"] = _tensor_np("last_gu_urgency_risk", dtype=np.float32)
        exported["last_gu_downstream_pressure"] = _tensor_np("last_gu_downstream_pressure", dtype=np.float32)
        exported["last_gu_service_gap_risk"] = _tensor_np("last_gu_service_gap_risk", dtype=np.float32)
        exported["last_gu_deadline_slack"] = _tensor_np("last_gu_deadline_slack", dtype=np.float32)
        exported["last_gu_deadline_risk"] = _tensor_np("last_gu_deadline_risk", dtype=np.float32)
        exported["last_gu_service_gap"] = _tensor_np("last_gu_service_gap", dtype=np.float32)
        exported["last_gu_deadline_age"] = _tensor_np("last_gu_deadline_age", dtype=np.float32)
        exported["last_exec_accel"] = _tensor_np("last_exec_accel", dtype=np.float32)
        exported["last_policy_accel"] = _tensor_np("last_policy_accel", dtype=np.float32)
        exported["_doppler_residual_state_hz"] = _tensor_np("doppler_residual", dtype=np.float32)
        exported["prev_queue_sum_gu"] = _tensor_scalar("prev_queue_sum_gu", dtype=np.float32)
        exported["prev_queue_sum_uav"] = _tensor_scalar("prev_queue_sum_uav", dtype=np.float32)
        exported["prev_queue_sum_sat"] = _tensor_scalar("prev_queue_sum_sat", dtype=np.float32)
        exported["prev_gu_queue_vec"] = _tensor_np("prev_gu_queue_vec", dtype=np.float32)
        exported["prev_uav_queue_vec"] = _tensor_np("prev_uav_queue_vec", dtype=np.float32)
        exported["prev_sat_queue_vec"] = _tensor_np("prev_sat_queue_vec", dtype=np.float32)
        exported["episode_idx"] = int(_tensor_scalar("episode_idx", dtype=np.int32))
        exported["t"] = int(_tensor_scalar("t", dtype=np.int32))
        exported["global_step"] = int(_tensor_scalar("global_step", dtype=np.int32))
        return exported


    def resolve_indices(self, indices: Sequence[int] | None = None) -> list[int]:
        if indices is None:
            return list(range(len(self._drivers)))
        return [int(index) for index in indices]

    def select_drivers(self, indices: Sequence[int] | None = None) -> tuple[list[int], list[StructuredControlDriver]]:
        selected_indices = self.resolve_indices(indices)
        return selected_indices, [self._drivers[index] for index in selected_indices]

    def output_device(self, device: torch.device | str | None = None) -> torch.device | None:
        if device is None:
            return self._tensor_device
        return torch.device(device)



    def _eta_feature_from_slots(
        cfg,
        candidates: Sequence[Sequence[int]],
        eta_slots: np.ndarray,
    ) -> np.ndarray:
        eta_feature = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
        eta_slots_arr = _compat_numpy_array(eta_slots, dtype=np.float32)
        for uav_idx in range(int(cfg.num_uav)):
            cand = list(candidates[uav_idx])[: int(cfg.users_obs_max)]
            for slot, gu_idx in enumerate(cand):
                gu_id = int(gu_idx)
                if 0 <= gu_id < int(cfg.num_gu):
                    eta_feature[uav_idx, gu_id] = float(eta_slots_arr[uav_idx, slot])
        return eta_feature

    def _ensure_step_started_native(self, indices: Sequence[int], *, tensor_only_meta: bool = False) -> None:
        if not indices:
            return
        cfg = self._cfg
        runtime_tensor_state = self._runtime_tensor_state
        selected = [int(index) for index in indices]
        kernel_device = runtime_tensor_state.global_step.device
        full_batch_selected = bool(
            len(selected) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected))
        )
        selected_tensor = None
        if not full_batch_selected:
            selected_arr = np.asarray(selected, dtype=np.int64)
            selected_tensor = torch.as_tensor(selected_arr, dtype=torch.long, device=kernel_device)

        def _select_runtime_rows(tensor: torch.Tensor) -> torch.Tensor:
            if full_batch_selected:
                return tensor
            if selected_tensor is None:
                raise RuntimeError("native partial-batch runtime selection tensor was not initialized.")
            return tensor[selected_tensor]

        def _assign_runtime_rows(target: torch.Tensor, value: torch.Tensor) -> None:
            if full_batch_selected:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
            else:
                if selected_tensor is None:
                    raise RuntimeError("native partial-batch runtime selection tensor was not initialized.")
                target[selected_tensor] = value.to(device=target.device, dtype=target.dtype)

        gpu_native_hot_meta = bool(
            tensor_only_meta
            or (
                self._native_rollout_fast_random
                and self._tensor_device is not None
                and self._tensor_device.type == "cuda"
            )
        )

        if full_batch_selected:
            runtime_tensor_state.global_step.add_(1)
        else:
            if selected_tensor is None:
                raise RuntimeError("native partial-batch runtime selection tensor was not initialized.")
            runtime_tensor_state.global_step[selected_tensor] = runtime_tensor_state.global_step[selected_tensor] + 1
        prev_queue_sum_gu = _sum_tensor_float32_semantics(
            _select_runtime_rows(runtime_tensor_state.gu_queue),
            dim=1,
        ).to(dtype=torch.float32)
        prev_queue_sum_uav = _sum_tensor_float32_semantics(
            _select_runtime_rows(runtime_tensor_state.uav_queue),
            dim=1,
        ).to(dtype=torch.float32)
        prev_queue_sum_sat = _select_runtime_rows(runtime_tensor_state.sat_queue).sum(dim=1, dtype=torch.float32)
        _assign_runtime_rows(runtime_tensor_state.prev_queue_sum_gu, prev_queue_sum_gu)
        _assign_runtime_rows(runtime_tensor_state.prev_queue_sum_uav, prev_queue_sum_uav)
        _assign_runtime_rows(runtime_tensor_state.prev_queue_sum_sat, prev_queue_sum_sat)
        _assign_runtime_rows(runtime_tensor_state.prev_gu_queue_vec, _select_runtime_rows(runtime_tensor_state.gu_queue))
        _assign_runtime_rows(runtime_tensor_state.prev_uav_queue_vec, _select_runtime_rows(runtime_tensor_state.uav_queue))
        _assign_runtime_rows(runtime_tensor_state.prev_sat_queue_vec, _select_runtime_rows(runtime_tensor_state.sat_queue))

        if gpu_native_hot_meta:
            return

        gu_pos_batch_t = _select_runtime_rows(runtime_tensor_state.gu_pos).to(dtype=torch.float32)
        uav_pos_batch_t = _select_runtime_rows(runtime_tensor_state.uav_pos).to(dtype=torch.float32)
        gu_queue_batch_t = _select_runtime_rows(runtime_tensor_state.gu_queue).to(dtype=torch.float32)
        effective_arrival_batch_t = _select_runtime_rows(runtime_tensor_state.effective_task_arrival_rate).to(dtype=torch.float32)
        global_step_batch = _select_runtime_rows(runtime_tensor_state.global_step).detach().cpu().tolist()
        prev_arrival_sum_t = torch.as_tensor(
            [float(self._slot_state_payloads[int(slot)].get("prev_arrival_sum", 0.0)) for slot in selected],
            dtype=torch.float32,
            device=kernel_device,
        )
        queue_norm_k = normalize_scale(float(getattr(cfg, "queue_norm_K", 1.0) or 1.0))
        arrival_floor_cfg = float(getattr(cfg, "queue_norm_arrival_floor", 0.0) or 0.0)
        if arrival_floor_cfg > 0.0:
            arrival_floor_t = torch.full_like(prev_arrival_sum_t, arrival_floor_cfg, dtype=torch.float32)
        else:
            arrival_floor_t = effective_arrival_batch_t * float(cfg.num_gu) * float(cfg.tau0)
        prev_scale_t = float(queue_norm_k) * _torch_require_positive_reward_ref(
            torch.maximum(prev_arrival_sum_t, arrival_floor_t),
            name="queue arrival normalization reference",
        )
        if int(cfg.num_gu) > 0:
            q_weights_t = gu_queue_batch_t / normalize_scale(float(cfg.queue_max_gu))
            w_sum_t = q_weights_t.sum(dim=1, dtype=torch.float32)
            uniform_weights_t = torch.full(
                (len(selected), int(cfg.num_gu)),
                1.0 / max(int(cfg.num_gu), 1),
                dtype=torch.float32,
                device=kernel_device,
            )
            normalized_weights_t = _torch_divide_or_default(q_weights_t, w_sum_t[:, None])
            weights_t = torch.where((w_sum_t > NORMALIZATION_DENOM_EPS)[:, None], normalized_weights_t, uniform_weights_t)
            centroids_t = (gu_pos_batch_t * weights_t[:, :, None]).sum(dim=1, dtype=torch.float32)
            centroid_dists_t = torch.linalg.vector_norm(uav_pos_batch_t - centroids_t[:, None, :], dim=2)
            centroid_dist_mean_t = (
                centroid_dists_t.mean(dim=1, dtype=torch.float32)
                if centroid_dists_t.numel() > 0
                else torch.zeros((len(selected),), dtype=torch.float32, device=kernel_device)
            )
            d2d_all_t = torch.linalg.vector_norm(
                gu_pos_batch_t[:, None, :, :] - uav_pos_batch_t[:, :, None, :],
                dim=3,
            )
            prev_d_min_t = d2d_all_t.amin(dim=(1, 2)).to(dtype=torch.float32)
        else:
            centroid_dist_mean_t = torch.zeros((len(selected),), dtype=torch.float32, device=kernel_device)
            prev_d_min_t = torch.zeros((len(selected),), dtype=torch.float32, device=kernel_device)
        meta_metric_values = torch.stack(
            (
                prev_queue_sum_gu,
                prev_queue_sum_uav,
                prev_queue_sum_sat,
                _torch_ratio_or_zero(prev_queue_sum_gu + prev_queue_sum_uav, prev_scale_t),
                centroid_dist_mean_t,
                prev_d_min_t,
            ),
            dim=1,
        ).to(dtype=torch.float32).detach().cpu().numpy()

        for local_index, slot in enumerate(selected):
            meta = self._slot_state_payloads[int(slot)]
            metric_row = meta_metric_values[local_index]
            prev_queue_sum_gu_value = float(metric_row[0])
            prev_queue_sum_uav_value = float(metric_row[1])
            prev_queue_sum_sat_value = float(metric_row[2])
            prev_queue_sum_active = float(prev_queue_sum_gu_value + prev_queue_sum_uav_value)
            prev_queue_sum_total = float(prev_queue_sum_active + prev_queue_sum_sat_value)
            meta["global_step"] = int(global_step_batch[local_index])
            meta["prev_queue_sum"] = prev_queue_sum_total
            meta["prev_queue_sum_active"] = prev_queue_sum_active
            meta["prev_queue_sum_gu"] = prev_queue_sum_gu_value
            meta["prev_queue_sum_uav"] = prev_queue_sum_uav_value
            meta["prev_queue_sum_sat"] = prev_queue_sum_sat_value
            meta["prev_q_norm_active"] = float(np.clip(float(metric_row[3]), 0.0, 1.0))
            meta["prev_centroid_dist_mean"] = float(metric_row[4])
            meta["prev_d_min"] = float(metric_row[5])

    def _apply_native_accel_stage_batch_runtime_native(
        self,
        indices: Sequence[int],
        accel_actions: Sequence[np.ndarray | torch.Tensor],
    ) -> None:
        if not indices:
            return
        cfg = self._cfg
        typed_domains = self._native_main_kernel_typed_domains()
        accel_safety_p = typed_domains.accel_safety
        selected_indices = [int(index) for index in indices]
        selected_arr = np.asarray(selected_indices, dtype=np.int64)
        runtime_tensor_state = self._runtime_tensor_state
        kernel_device = runtime_tensor_state.uav_pos.device
        selected_tensor = torch.as_tensor(selected_arr, dtype=torch.long, device=kernel_device)
        full_batch_selected = bool(
            len(selected_indices) == int(self._num_envs)
            and all(int(index) == pos for pos, index in enumerate(selected_indices))
        )

        def _select_runtime_rows(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if full_batch_selected else tensor[selected_tensor]

        def _assign_runtime_rows(target: torch.Tensor, value: torch.Tensor) -> None:
            if full_batch_selected:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
            else:
                target[selected_tensor] = value.to(device=target.device, dtype=target.dtype)

        self._ensure_step_started_native(selected_indices)

        action_t = _as_grouped_tensor_batch(
            accel_actions,
            expected_envs=len(selected_indices),
            trailing_shape=(int(cfg.num_uav), 2),
            dtype=torch.float32,
            device=kernel_device,
        )
        policy_accel_t = _project_l2_ball_torch(torch.clamp(action_t, min=-1.0, max=1.0), 1.0) * float(cfg.a_max)
        exec_accel_t = policy_accel_t.clone()
        prev_association_t = _select_runtime_rows(runtime_tensor_state.last_association).clone()
        uav_pos_t = _select_runtime_rows(runtime_tensor_state.uav_pos).clone()
        uav_vel_t = _select_runtime_rows(runtime_tensor_state.uav_vel).clone()
        uav_energy_t = _select_runtime_rows(runtime_tensor_state.uav_energy).clone()

        use_avoidance = ablation_flag(
            cfg,
            "use_avoidance_layer",
            fallback_attr="avoidance_enabled",
            default=False,
        )
        use_energy_safety = ablation_flag(
            cfg,
            "use_energy_safety_layer",
            fallback_attr="energy_safety_enabled",
            default=False,
        )
        bw_link_p = typed_domains.bw_link if bool(cfg.energy_enabled) and bool(use_energy_safety) else None
        boundary_hard_enabled = bool(getattr(cfg, "boundary_hard_filter_enabled", False))
        pairwise_hard_enabled = bool(getattr(cfg, "pairwise_hard_filter_enabled", False))
        boundary_mode = str(getattr(cfg, "boundary_mode", "clip") or "clip").strip().lower()
        if boundary_mode not in {"clip", "reflect"}:
            raise ValueError(f"Unsupported boundary_mode={boundary_mode!r} for native accel runtime")

        d_alert = float(cfg.avoidance_alert_factor) * float(cfg.d_safe) if use_avoidance else 0.0
        raw_prealert_factor = getattr(cfg, "avoidance_prealert_factor", None)
        d_prealert = 0.0
        if use_avoidance and raw_prealert_factor is not None:
            d_prealert = max(float(raw_prealert_factor) * float(cfg.d_safe), d_alert)
        prealert_mode = str(getattr(cfg, "avoidance_prealert_mode", "distance") or "distance").strip().lower()
        if prealert_mode not in {"distance", "ttc"}:
            prealert_mode = "distance"
        closing_speed_thresh = max(float(getattr(cfg, "avoidance_prealert_closing_speed", 0.0) or 0.0), 0.0)
        prealert_ttc_limit = max(float(getattr(cfg, "avoidance_prealert_ttc", 0.0) or 0.0), 0.0)
        raw_prealert_dist_cap = getattr(cfg, "avoidance_prealert_dist_cap", None)
        prealert_trigger_dist = d_prealert
        if use_avoidance and prealert_mode == "ttc":
            if raw_prealert_dist_cap is not None:
                prealert_trigger_dist = max(float(raw_prealert_dist_cap), d_alert)
            elif d_prealert > 0.0:
                prealert_trigger_dist = d_prealert
        repulse_mode = str(getattr(cfg, "avoidance_repulse_mode", "inverse") or "inverse").strip().lower()
        closing_gain_enabled = bool(getattr(cfg, "avoidance_closing_gain_enabled", False))
        closing_gain_cap = max(float(getattr(cfg, "avoidance_closing_gain_cap", 2.0) or 2.0), 1.0)
        closing_gain_top1_only = bool(getattr(cfg, "avoidance_closing_gain_top1_only", False))
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)
        eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
        eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
        eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
        eta_max = max(eta_min, eta_max)

        safe_threshold = float(cfg.energy_safe_threshold) * float(cfg.uav_energy_init)
        map_limit = float(cfg.map_size)
        tau = float(cfg.tau0)
        a_max = float(cfg.a_max)
        v_max = float(cfg.v_max)
        standard_vector_accel_path = bool(
            use_avoidance
            and d_alert > 0.0
            and int(cfg.num_uav) > 1
            and not (bool(cfg.energy_enabled) and use_energy_safety)
            and not boundary_hard_enabled
            and not pairwise_hard_enabled
            and boundary_mode in {"clip", "reflect"}
            and not cross_enabled
        )
        hot_tensor_main = bool(
            self._native_rollout_fast_random
            and full_batch_selected
            and kernel_device.type == "cuda"
        )
        if standard_vector_accel_path:
            if hot_tensor_main:
                eta_t = torch.full(
                    (len(selected_indices),),
                    float(np.clip(float(getattr(cfg, "avoidance_eta", 0.0) or 0.0), eta_min, eta_max)),
                    dtype=torch.float32,
                    device=kernel_device,
                )
                eta_values = None
            else:
                eta_values = []
                for slot in selected_indices:
                    meta = self._slot_state_payloads[int(slot)]
                    eta_value = float(meta.get("avoidance_eta_eff", cfg.avoidance_eta))
                    eta_values.append(float(np.clip(eta_value, eta_min, eta_max)))
                eta_t = torch.as_tensor(eta_values, dtype=torch.float32, device=kernel_device)
            uav_count = int(cfg.num_uav)
            diff_t = uav_pos_t[:, :, None, :] - uav_pos_t[:, None, :, :]
            dist_t = torch.linalg.vector_norm(diff_t, dim=-1)
            offdiag_mask_t = ~torch.eye(uav_count, dtype=torch.bool, device=kernel_device).view(1, uav_count, uav_count)
            valid_pair_t = offdiag_mask_t & (dist_t > 1.0e-6)
            rel_vel_t = uav_vel_t[:, :, None, :] - uav_vel_t[:, None, :, :]
            closing_speed_t = -torch.sum(diff_t * rel_vel_t, dim=-1) / _torch_positive(dist_t, GEOMETRY_DENOM_EPS)
            in_core_alert_t = valid_pair_t & (dist_t < float(d_alert))
            if prealert_mode == "ttc":
                ttc_to_alert_t = (dist_t - float(d_alert)) / torch.clamp(closing_speed_t, min=1.0e-6)
                in_prealert_t = (
                    valid_pair_t
                    & (float(prealert_trigger_dist) > float(d_alert))
                    & (dist_t < float(prealert_trigger_dist))
                    & (closing_speed_t > float(closing_speed_thresh))
                    & (float(prealert_ttc_limit) > 0.0)
                    & (ttc_to_alert_t < float(prealert_ttc_limit))
                )
                ttc_urgency_t = torch.where(
                    torch.isfinite(ttc_to_alert_t),
                    1.0 / torch.clamp(ttc_to_alert_t, min=1.0e-6),
                    torch.zeros_like(ttc_to_alert_t),
                )
            else:
                in_prealert_t = (
                    valid_pair_t
                    & (float(d_prealert) > float(d_alert))
                    & (dist_t < float(d_prealert))
                    & (closing_speed_t > float(closing_speed_thresh))
                )
                ttc_urgency_t = torch.zeros_like(dist_t)
            active_pair_t = in_core_alert_t | in_prealert_t
            prealert_trigger_value = float(prealert_trigger_dist if prealert_mode == "ttc" else d_prealert)
            trigger_dist_t = torch.where(
                in_prealert_t & ~in_core_alert_t,
                torch.full_like(dist_t, prealert_trigger_value),
                torch.full_like(dist_t, float(d_alert)),
            )
            if repulse_mode == "linear":
                denom = torch.clamp(trigger_dist_t - float(cfg.d_safe), min=1.0e-6)
                strength_t = torch.clamp((trigger_dist_t - dist_t) / denom, min=0.0, max=1.0)
            elif repulse_mode == "quadratic":
                denom = torch.clamp(trigger_dist_t - float(cfg.d_safe), min=1.0e-6)
                base_t = torch.clamp((trigger_dist_t - dist_t) / denom, min=0.0, max=1.0)
                strength_t = base_t * base_t
            else:
                strength_t = 1.0 / torch.clamp(dist_t, min=1.0e-6) - 1.0 / torch.clamp(trigger_dist_t, min=1.0e-6)
            strength_t = torch.where(active_pair_t, strength_t, torch.zeros_like(strength_t))
            closing_ratio_raw_t = torch.ones_like(strength_t)
            closing_gain_t = torch.ones_like(strength_t)
            if closing_gain_enabled and closing_speed_thresh > 1.0e-6:
                raw_ratio_t = closing_speed_t / float(closing_speed_thresh)
                gain_candidate_t = torch.clamp(raw_ratio_t, min=1.0, max=float(closing_gain_cap))
                gain_active_t = active_pair_t & (closing_speed_t > float(closing_speed_thresh))
                closing_ratio_raw_t = torch.where(gain_active_t, raw_ratio_t, closing_ratio_raw_t)
                closing_gain_t = torch.where(gain_active_t, gain_candidate_t, closing_gain_t)
            if closing_gain_enabled and closing_gain_top1_only:
                eligible_t = active_pair_t & (closing_gain_t > 1.0)
                neg_inf_t = torch.full_like(strength_t, -float("inf"))
                key0_t = torch.where(eligible_t, in_core_alert_t.to(dtype=torch.float32), neg_inf_t)
                max0_t = key0_t.amax(dim=2, keepdim=True)
                mask_t = eligible_t & (key0_t == max0_t)
                closing_bonus_t = strength_t * torch.clamp(closing_ratio_raw_t - 1.0, min=0.0)
                key1_t = torch.where(mask_t, closing_bonus_t, neg_inf_t)
                max1_t = key1_t.amax(dim=2, keepdim=True)
                mask_t = mask_t & (key1_t == max1_t)
                key2_t = torch.where(mask_t, ttc_urgency_t, neg_inf_t)
                max2_t = key2_t.amax(dim=2, keepdim=True)
                mask_t = mask_t & (key2_t == max2_t)
                key3_t = torch.where(mask_t, strength_t, neg_inf_t)
                max3_t = key3_t.amax(dim=2, keepdim=True)
                mask_t = mask_t & (key3_t == max3_t)
                key4_t = torch.where(mask_t, -dist_t, neg_inf_t)
                best_idx_t = key4_t.argmax(dim=2)
                has_best_t = eligible_t.any(dim=2)
                selected_t = (
                    torch.arange(uav_count, device=kernel_device).view(1, 1, uav_count)
                    == best_idx_t.unsqueeze(-1)
                ) & has_best_t.unsqueeze(-1)
                effective_closing_gain_t = torch.where(selected_t, closing_gain_t, torch.ones_like(closing_gain_t))
            else:
                effective_closing_gain_t = closing_gain_t
            direction_t = diff_t / torch.clamp(dist_t, min=1.0e-6).unsqueeze(-1)
            repulse_t = (
                eta_t.view(-1, 1, 1, 1)
                * strength_t.unsqueeze(-1)
                * effective_closing_gain_t.unsqueeze(-1)
                * direction_t.to(dtype=torch.float32)
            ).sum(dim=2)
            if bool(getattr(cfg, "avoidance_repulse_clip", True)):
                repulse_t = _project_l2_ball_torch(repulse_t, a_max)
            exec_accel_t = _project_l2_ball_torch(policy_accel_t + repulse_t, a_max)
            uav_vel_next_t = _project_l2_ball_torch(uav_vel_t + exec_accel_t * tau, v_max)
            uav_pos_next_t = uav_pos_t + uav_vel_next_t * tau
            if boundary_mode == "reflect":
                below_t = uav_pos_next_t < 0.0
                uav_pos_next_t = torch.where(below_t, -uav_pos_next_t, uav_pos_next_t)
                uav_vel_next_t = torch.where(below_t, -uav_vel_next_t, uav_vel_next_t)
                above_t = uav_pos_next_t > map_limit
                uav_pos_next_t = torch.where(above_t, 2.0 * map_limit - uav_pos_next_t, uav_pos_next_t)
                uav_vel_next_t = torch.where(above_t, -uav_vel_next_t, uav_vel_next_t)
            uav_pos_next_t = torch.clamp(uav_pos_next_t, min=0.0, max=map_limit)
            _assign_runtime_rows(runtime_tensor_state.prev_association, prev_association_t)
            _assign_runtime_rows(runtime_tensor_state.uav_vel, uav_vel_next_t)
            _assign_runtime_rows(runtime_tensor_state.uav_pos, uav_pos_next_t)
            _assign_runtime_rows(runtime_tensor_state.last_exec_accel, exec_accel_t)
            _assign_runtime_rows(runtime_tensor_state.last_policy_accel, policy_accel_t)
            if not hot_tensor_main:
                for local_index, slot in enumerate(selected_indices):
                    meta = self._slot_state_payloads[int(slot)]
                    meta["last_avoidance_eta_exec"] = float(eta_values[local_index])
                    meta["last_filter_active_ratio"] = 0.0
                    meta["last_projected_delta_norm_mean"] = 0.0
                    meta["last_fallback_count"] = 0.0
                    meta["last_boundary_filter_count"] = 0.0
                    meta["last_pairwise_filter_count"] = 0.0
                    meta["last_pairwise_filter_active_ratio"] = 0.0
                    meta["last_pairwise_projected_delta_norm"] = 0.0
                    meta["last_pairwise_fallback_count"] = 0.0
                    meta["last_pairwise_candidate_infeasible_count"] = 0.0
                    meta["_cached_uav_ecef"] = None
                    meta["_cached_uav_vel_ecef"] = None
                    meta["_cached_elevation_t"] = None
                    meta["_cached_elevation_matrix"] = None
                    meta["_cached_backhaul_loss_t"] = None
                    meta["_cached_backhaul_loss_matrix"] = None
                    meta["_cached_uav_neighbor_t"] = None
                    meta["_cached_uav_neighbor_order"] = None
                    meta["_cached_global_state"] = None
                    meta["_cached_obs_runtime_context"] = None
            return

        for local_index, slot in enumerate(selected_indices):
            env = self._slot_view(slot)
            eta_avoid = float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta))
            if cross_enabled:
                _, _, transfer_ratio = env._centroid_anneal_state()
                eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * transfer_ratio)
                eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))

            accel_env_t = policy_accel_t[local_index].clone()
            if use_avoidance and d_alert > 0.0 and int(cfg.num_uav) > 1:
                repulse_t = torch.zeros_like(accel_env_t)
                for i in range(int(cfg.num_uav)):
                    pair_terms: list[dict[str, Any]] = []
                    for j in range(int(cfg.num_uav)):
                        if i == j:
                            continue
                        diff_t = uav_pos_t[local_index, i] - uav_pos_t[local_index, j]
                        dist_t = torch.linalg.vector_norm(diff_t)
                        dist = float(dist_t.item())
                        if dist <= 1.0e-6:
                            continue
                        rel_vel_t = uav_vel_t[local_index, i] - uav_vel_t[local_index, j]
                        closing_speed = float((-(diff_t * rel_vel_t).sum() / _torch_positive(dist_t, GEOMETRY_DENOM_EPS)).item())
                        trigger_dist = d_alert
                        in_core_alert = dist < d_alert
                        in_prealert = False
                        ttc_to_alert = float("inf")
                        if prealert_mode == "ttc":
                            if (
                                prealert_trigger_dist > d_alert
                                and dist < prealert_trigger_dist
                                and closing_speed > closing_speed_thresh
                                and prealert_ttc_limit > 0.0
                            ):
                                ttc_to_alert = (dist - d_alert) / max(closing_speed, 1.0e-6)
                                in_prealert = ttc_to_alert < prealert_ttc_limit
                        else:
                            in_prealert = (
                                d_prealert > d_alert
                                and dist < d_prealert
                                and closing_speed > closing_speed_thresh
                            )
                        if in_prealert and not in_core_alert:
                            trigger_dist = prealert_trigger_dist if prealert_mode == "ttc" else d_prealert
                        if not (in_core_alert or in_prealert):
                            continue
                        direction_t = diff_t / torch.clamp(dist_t, min=1.0e-6)
                        if repulse_mode == "linear":
                            denom = max(trigger_dist - float(cfg.d_safe), 1.0e-6)
                            strength = float(np.clip((trigger_dist - dist) / denom, 0.0, 1.0))
                        elif repulse_mode == "quadratic":
                            denom = max(trigger_dist - float(cfg.d_safe), 1.0e-6)
                            base = float(np.clip((trigger_dist - dist) / denom, 0.0, 1.0))
                            strength = base * base
                        else:
                            strength = (1.0 / dist - 1.0 / max(trigger_dist, 1.0e-6))
                        closing_ratio_raw = 1.0
                        closing_gain = 1.0
                        if closing_gain_enabled and closing_speed_thresh > 1.0e-6 and closing_speed > closing_speed_thresh:
                            closing_ratio_raw = closing_speed / closing_speed_thresh
                            closing_gain = float(np.clip(closing_ratio_raw, 1.0, closing_gain_cap))
                        pair_terms.append(
                            {
                                "direction": direction_t,
                                "strength": strength,
                                "closing_gain": closing_gain,
                                "closing_ratio_raw": closing_ratio_raw,
                                "closing_bonus_score": strength * max(closing_ratio_raw - 1.0, 0.0),
                                "in_core_alert": in_core_alert,
                                "ttc_urgency": (1.0 / max(ttc_to_alert, 1.0e-6)) if math.isfinite(ttc_to_alert) else 0.0,
                                "dist": dist,
                            }
                        )
                    top1_gain_idx = None
                    if closing_gain_enabled and closing_gain_top1_only and pair_terms:
                        best_key = None
                        for term_index, term in enumerate(pair_terms):
                            if float(term["closing_gain"]) <= 1.0:
                                continue
                            key = (
                                1 if bool(term["in_core_alert"]) else 0,
                                float(term["closing_bonus_score"]),
                                float(term["ttc_urgency"]),
                                float(term["strength"]),
                                -float(term["dist"]),
                            )
                            if best_key is None or key > best_key:
                                best_key = key
                                top1_gain_idx = term_index
                    repulse_i_t = torch.zeros((2,), dtype=torch.float32, device=kernel_device)
                    for term_index, term in enumerate(pair_terms):
                        closing_gain = float(term["closing_gain"])
                        if closing_gain_top1_only and top1_gain_idx is not None and term_index != top1_gain_idx:
                            closing_gain = 1.0
                        repulse_i_t = repulse_i_t + (
                            float(eta_avoid) * float(term["strength"]) * float(closing_gain)
                        ) * term["direction"].to(dtype=torch.float32)
                    if bool(getattr(cfg, "avoidance_repulse_clip", True)):
                        repulse_i_t = _project_l2_ball_torch(repulse_i_t, a_max)
                    repulse_t[i] = repulse_i_t
                accel_env_t = accel_env_t + repulse_t

            if bool(cfg.energy_enabled) and use_energy_safety:
                for i in range(int(cfg.num_uav)):
                    v_next_t = uav_vel_t[local_index, i] + accel_env_t[i] * tau
                    speed_next_t = torch.linalg.vector_norm(v_next_t)
                    est_energy_t = (
                        uav_energy_t[local_index, i]
                        - _fly_power_torch(speed_next_t.reshape(1), params=bw_link_p).reshape(())
                        * tau
                    )
                    if float(est_energy_t.item()) < safe_threshold:
                        cur_speed_t = torch.linalg.vector_norm(uav_vel_t[local_index, i])
                        cur_speed = float(cur_speed_t.item())
                        if cur_speed > 1.0e-6:
                            direction_t = uav_vel_t[local_index, i] / cur_speed_t
                        else:
                            accel_norm_t = torch.linalg.vector_norm(accel_env_t[i])
                            accel_norm = float(accel_norm_t.item())
                            if accel_norm > 1.0e-6:
                                direction_t = accel_env_t[i] / accel_norm_t
                            else:
                                direction_t = torch.zeros((2,), dtype=torch.float32, device=kernel_device)
                        target_delta = float(cfg.uav_opt_speed) - cur_speed
                        safe_accel = np.clip(target_delta / max(tau, 1.0e-6), -a_max, a_max)
                        accel_env_t[i] = direction_t.to(dtype=torch.float32) * float(safe_accel)

            accel_env_t = _project_l2_ball_torch(accel_env_t, a_max)
            boundary_stats = {
                "filter_active_ratio": 0.0,
                "projected_delta_norm_mean": 0.0,
                "fallback_count": 0.0,
                "boundary_filter_count": 0.0,
            }
            pairwise_stats = {
                "pairwise_filter_count": 0.0,
                "pairwise_filter_active_ratio": 0.0,
                "pairwise_projected_delta_norm": 0.0,
                "pairwise_fallback_count": 0.0,
                "pairwise_candidate_infeasible_count": 0.0,
            }
            hard_active_ratio = 0.0
            hard_projected_delta_norm_mean = 0.0
            hard_fallback_count = 0.0
            if boundary_hard_enabled or pairwise_hard_enabled:
                accel_before_t = accel_env_t.clone()
                accel_safe_t = accel_before_t
                uav_pos_env_t = uav_pos_t[local_index]
                uav_vel_env_t = uav_vel_t[local_index]
                if boundary_hard_enabled:
                    accel_safe_t, boundary_stats = _apply_boundary_hard_filter_tensor_from_state(
                        accel_safety_p,
                        uav_pos_env_t,
                        uav_vel_env_t,
                        accel_safe_t,
                    )
                if pairwise_hard_enabled:
                    accel_safe_t, pairwise_stats = _apply_pairwise_hard_filter_tensor_from_state(
                        cfg,
                        accel_safety_p,
                        uav_pos_env_t,
                        uav_vel_env_t,
                        accel_safe_t,
                    )
                hard_delta_norm_t = torch.linalg.vector_norm(accel_safe_t - accel_before_t, dim=1)
                hard_active_ratio = float(
                    (hard_delta_norm_t > 1.0e-6).to(dtype=torch.float32).mean().item()
                ) if hard_delta_norm_t.numel() > 0 else 0.0
                hard_projected_delta_norm_mean = float(hard_delta_norm_t.mean().item()) if hard_delta_norm_t.numel() > 0 else 0.0
                hard_fallback_count = float(boundary_stats["fallback_count"] + pairwise_stats["pairwise_fallback_count"])
                accel_env_t = accel_safe_t.to(dtype=torch.float32)

            uav_vel_env_t = _project_l2_ball_torch(uav_vel_t[local_index] + accel_env_t * tau, v_max)
            uav_pos_env_t = uav_pos_t[local_index] + uav_vel_env_t * tau
            if boundary_mode == "reflect":
                for axis in range(2):
                    below_mask_t = uav_pos_env_t[:, axis] < 0.0
                    if bool(torch.any(below_mask_t)):
                        uav_pos_env_t[below_mask_t, axis] = -uav_pos_env_t[below_mask_t, axis]
                        uav_vel_env_t[below_mask_t, axis] = -uav_vel_env_t[below_mask_t, axis]
                    above_mask_t = uav_pos_env_t[:, axis] > map_limit
                    if bool(torch.any(above_mask_t)):
                        uav_pos_env_t[above_mask_t, axis] = 2.0 * map_limit - uav_pos_env_t[above_mask_t, axis]
                        uav_vel_env_t[above_mask_t, axis] = -uav_vel_env_t[above_mask_t, axis]
            uav_pos_env_t = torch.clamp(uav_pos_env_t, min=0.0, max=map_limit)

            exec_accel_t[local_index] = accel_env_t
            uav_pos_t[local_index] = uav_pos_env_t
            uav_vel_t[local_index] = uav_vel_env_t

            env.last_avoidance_eta_exec = float(eta_avoid)
            env.last_filter_active_ratio = float(hard_active_ratio)
            env.last_projected_delta_norm_mean = float(hard_projected_delta_norm_mean)
            env.last_fallback_count = float(hard_fallback_count)
            env.last_boundary_filter_count = float(boundary_stats["boundary_filter_count"])
            env.last_pairwise_filter_count = float(pairwise_stats["pairwise_filter_count"])
            env.last_pairwise_filter_active_ratio = float(pairwise_stats["pairwise_filter_active_ratio"])
            env.last_pairwise_projected_delta_norm = float(pairwise_stats["pairwise_projected_delta_norm"])
            env.last_pairwise_fallback_count = float(pairwise_stats["pairwise_fallback_count"])
            env.last_pairwise_candidate_infeasible_count = float(pairwise_stats["pairwise_candidate_infeasible_count"])
            env._cached_uav_ecef = None
            env._cached_uav_vel_ecef = None
            env._cached_elevation_t = None
            env._cached_elevation_matrix = None
            env._cached_backhaul_loss_t = None
            env._cached_backhaul_loss_matrix = None
            env._cached_uav_neighbor_t = None
            env._cached_uav_neighbor_order = None
            env._cached_global_state = None
            env._cached_obs_runtime_context = None

        _assign_runtime_rows(runtime_tensor_state.prev_association, prev_association_t)
        _assign_runtime_rows(runtime_tensor_state.uav_vel, uav_vel_t)
        _assign_runtime_rows(runtime_tensor_state.uav_pos, uav_pos_t)
        _assign_runtime_rows(runtime_tensor_state.last_exec_accel, exec_accel_t)
        _assign_runtime_rows(runtime_tensor_state.last_policy_accel, policy_accel_t)

    def _apply_simple_accel_stage_batch_runtime_native(
        self,
        indices: Sequence[int],
        accel_actions: Sequence[np.ndarray],
    ) -> None:
        if not indices:
            return
        cfg = self._cfg
        selected_indices = [int(index) for index in indices]
        selected_arr = np.asarray(selected_indices, dtype=np.int64)
        runtime_tensor_state = self._runtime_tensor_state
        kernel_device = runtime_tensor_state.uav_pos.device
        selected_tensor = torch.as_tensor(selected_arr, dtype=torch.long, device=kernel_device)
        self._ensure_step_started_native(selected_indices)

        action_t = _as_grouped_tensor_batch(
            accel_actions,
            expected_envs=len(selected_indices),
            trailing_shape=(int(cfg.num_uav), 2),
            dtype=torch.float32,
            device=kernel_device,
        )
        exec_accel_t = _project_l2_ball_torch(torch.clamp(action_t, min=-1.0, max=1.0), 1.0) * float(cfg.a_max)
        prev_association_t = runtime_tensor_state.last_association[selected_tensor].clone()
        uav_vel_t = _project_l2_ball_torch(
            runtime_tensor_state.uav_vel[selected_tensor] + exec_accel_t * float(cfg.tau0),
            cfg.v_max,
        )
        uav_pos_t = torch.clamp(
            runtime_tensor_state.uav_pos[selected_tensor] + uav_vel_t * float(cfg.tau0),
            min=0.0,
            max=float(cfg.map_size),
        )

        runtime_tensor_state.prev_association[selected_tensor] = prev_association_t
        runtime_tensor_state.uav_vel[selected_tensor] = uav_vel_t
        runtime_tensor_state.uav_pos[selected_tensor] = uav_pos_t
        runtime_tensor_state.last_exec_accel[selected_tensor] = exec_accel_t
        runtime_tensor_state.last_policy_accel[selected_tensor] = exec_accel_t

        eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
        eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
        eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
        eta_max = max(eta_min, eta_max)
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)
        for local_index, slot in enumerate(selected_indices):
            env = self._slot_view(slot)
            eta_avoid = float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta))
            if cross_enabled:
                _, _, transfer_ratio = env._centroid_anneal_state()
                eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * transfer_ratio)
            eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))
            env.last_avoidance_eta_exec = float(eta_avoid)
            env.last_filter_active_ratio = 0.0
            env.last_projected_delta_norm_mean = 0.0
            env.last_fallback_count = 0.0
            env.last_boundary_filter_count = 0.0
            env.last_pairwise_filter_count = 0.0
            env.last_pairwise_filter_active_ratio = 0.0
            env.last_pairwise_projected_delta_norm = 0.0
            env.last_pairwise_fallback_count = 0.0
            env.last_pairwise_candidate_infeasible_count = 0.0
            env.prev_association = prev_association_t[local_index]
            env.last_policy_accel = runtime_tensor_state.last_policy_accel[selected_tensor][local_index]
            env.last_exec_accel = runtime_tensor_state.last_exec_accel[selected_tensor][local_index]
            env.uav_vel = uav_vel_t[local_index]
            env.uav_pos = uav_pos_t[local_index]
            env._cached_global_state = None
            env._cached_obs_runtime_context = None

    def _apply_simple_accel_stage_batch_runtime(
        self,
        drivers: Sequence[StructuredControlDriver],
        accel_actions: Sequence[np.ndarray],
    ) -> None:
        if not drivers:
            return
        cfg = drivers[0].env.cfg
        selected_indices = self._slot_indices_for_drivers(drivers)
        selected_arr = np.asarray(selected_indices, dtype=np.int64)
        runtime_tensor_state = self._runtime_tensor_state
        kernel_device = runtime_tensor_state.uav_pos.device
        selected_tensor = torch.as_tensor(selected_arr, dtype=torch.long, device=kernel_device)
        for driver in drivers:
            driver._ensure_step_started()

        action_t = _as_grouped_tensor_batch(
            accel_actions,
            expected_envs=len(drivers),
            trailing_shape=(int(cfg.num_uav), 2),
            dtype=torch.float32,
            device=kernel_device,
        )
        exec_accel_t = _project_l2_ball_torch(torch.clamp(action_t, min=-1.0, max=1.0), 1.0) * float(cfg.a_max)
        prev_association_t = runtime_tensor_state.last_association[selected_tensor].clone()
        uav_vel_t = _project_l2_ball_torch(
            runtime_tensor_state.uav_vel[selected_tensor] + exec_accel_t * float(cfg.tau0),
            cfg.v_max,
        )
        uav_pos_t = torch.clamp(
            runtime_tensor_state.uav_pos[selected_tensor] + uav_vel_t * float(cfg.tau0),
            min=0.0,
            max=float(cfg.map_size),
        )

        runtime_tensor_state.prev_association[selected_tensor] = prev_association_t
        runtime_tensor_state.uav_vel[selected_tensor] = uav_vel_t
        runtime_tensor_state.uav_pos[selected_tensor] = uav_pos_t
        runtime_tensor_state.last_exec_accel[selected_tensor] = exec_accel_t
        runtime_tensor_state.last_policy_accel[selected_tensor] = exec_accel_t

        eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
        eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
        eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
        eta_max = max(eta_min, eta_max)
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)

        for local_index, (slot, driver) in enumerate(zip(selected_indices, drivers)):
            env = driver.env
            eta_avoid = float(getattr(env, "avoidance_eta_eff", cfg.avoidance_eta))
            if cross_enabled:
                _, _, transfer_ratio = env._centroid_anneal_state()
                eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * transfer_ratio)
                eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))
            env.last_avoidance_eta_exec = float(eta_avoid)
            env.last_filter_active_ratio = 0.0
            env.last_projected_delta_norm_mean = 0.0
            env.last_fallback_count = 0.0
            env.last_boundary_filter_count = 0.0
            env.last_pairwise_filter_count = 0.0
            env.last_pairwise_filter_active_ratio = 0.0
            env.last_pairwise_projected_delta_norm = 0.0
            env.last_pairwise_fallback_count = 0.0
            env.last_pairwise_candidate_infeasible_count = 0.0
            env.prev_association = _tensor_index_compat(prev_association_t, local_index, dtype=np.int32)
            env.last_policy_accel = _tensor_index_compat(
                runtime_tensor_state.last_policy_accel[selected_tensor],
                local_index,
                dtype=np.float32,
            )
            env.last_exec_accel = _tensor_index_compat(
                runtime_tensor_state.last_exec_accel[selected_tensor],
                local_index,
                dtype=np.float32,
            )
            env.uav_vel = _tensor_index_compat(uav_vel_t, local_index, dtype=np.float32)
            env.uav_pos = _tensor_index_compat(uav_pos_t, local_index, dtype=np.float32)
            env._cached_global_state = None
            env._cached_obs_runtime_context = None

        _refresh_uav_cache_batch(
            [driver.env for driver in drivers],
            uav_pos_batch=uav_pos_t,
            uav_vel_batch=uav_vel_t,
        )

    def get_global_state_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | np.ndarray:
        self._reject_native_main_kernel_legacy_adapter("get_global_state_batch")
        out_device = self.output_device(device)
        selected_indices = self.resolve_indices(indices)
        if not selected_indices:
            empty = np.zeros((0, 0), dtype=np.float32)
            if out_device is None:
                return empty
            return torch.as_tensor(empty, dtype=torch.float32, device=out_device)
        runtime_tensor_state = self._runtime_tensor_state
        selected_envs = self._slot_views(selected_indices)
        cached_states: list[np.ndarray] = []
        cached_state_tensors: list[torch.Tensor] = []
        for env in selected_envs:
            cached_state = getattr(env, "_cached_global_state", None)
            if cached_state is None:
                cached_states = []
                cached_state_tensors = []
                break
            if torch.is_tensor(cached_state):
                cached_state_tensors.append(cached_state)
                continue
            if isinstance(cached_state, np.ndarray) and cached_state.dtype == np.float32:
                cached_states.append(cached_state)
            else:
                cached_states.append(np.asarray(cached_state, dtype=np.float32))
        if cached_state_tensors and len(cached_state_tensors) == len(selected_envs):
            batch_tensor = torch.stack(
                [
                    state.to(device=out_device, dtype=torch.float32)
                    if out_device is not None
                    else state.to(dtype=torch.float32)
                    for state in cached_state_tensors
                ],
                dim=0,
            )
            if out_device is None:
                return batch_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            return batch_tensor
        if cached_states:
            batch = np.stack(cached_states, axis=0)
            if out_device is None:
                return batch
            return torch.as_tensor(batch, dtype=torch.float32, device=out_device)
        sat_state_max = getattr(self._cfg, "sat_state_max", None) if self._num_envs > 0 else None
        use_sat_subset = self._num_envs > 0 and sat_state_max is not None and int(sat_state_max) < int(self._cfg.num_sat)
        cfg = self._cfg if self._num_envs > 0 else None
        kernel_device = self.tensor_device
        selected_tensor = torch.as_tensor(
            selected_indices,
            dtype=torch.long,
            device=runtime_tensor_state.uav_pos.device,
        )
        typed_domains = self._native_main_kernel_typed_domains()
        global_state_params = typed_domains.global_state
        sat_geometry_params = typed_domains.sat_geometry
        bw_workload_params = typed_domains.bw_workload
        local_obs_params = typed_domains.local_obs

        def _global_state_adapter(**kwargs):
            return _build_global_state_tensor_impl(
                global_state_params=global_state_params,
                sat_geometry_params=sat_geometry_params,
                bw_workload_static_params=bw_workload_params,
                local_obs_params=local_obs_params,
                **kwargs,
            )

        global_state_kernel = _resolve_kernel_callable(
            name="global_state_batch",
            eager_fn=_global_state_adapter,
            compile_cfg=self._cfg,
            device=kernel_device,
        )
        expected_arrival_rate_vec_t = self._current_expected_arrival_rate_vec_tensor(
            selected_indices,
            selected_tensor,
            device=kernel_device,
        )
        batch_tensor = global_state_kernel(
            uav_pos_t=runtime_tensor_state.uav_pos.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            uav_vel_t=runtime_tensor_state.uav_vel.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            uav_queue_t=runtime_tensor_state.uav_queue.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            uav_energy_t=runtime_tensor_state.uav_energy.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            gu_pos_t=runtime_tensor_state.gu_pos.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            gu_queue_t=runtime_tensor_state.gu_queue.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            sat_pos_t=runtime_tensor_state.sat_pos.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            sat_vel_t=runtime_tensor_state.sat_vel.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            sat_queue_t=runtime_tensor_state.sat_queue.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            t_t=runtime_tensor_state.t.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            arrival_ref_t=runtime_tensor_state.arrival_ref_bits_per_step.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            gu_ema_t=runtime_tensor_state.gu_workload_ema.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            uav_ema_t=runtime_tensor_state.uav_workload_ema.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            sat_ema_t=runtime_tensor_state.sat_workload_ema.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            assoc_t=runtime_tensor_state.last_association.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.long),
            sat_selection_matrix_t=runtime_tensor_state.last_sat_selection_matrix.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.long),
            arrival_rate_vec_t=expected_arrival_rate_vec_t,
            recent_arrival_t=runtime_tensor_state.last_gu_arrival.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            recent_service_t=runtime_tensor_state.last_gu_outflow.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            urgency_risk_t=runtime_tensor_state.last_gu_urgency_risk.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            downstream_pressure_t=runtime_tensor_state.last_gu_downstream_pressure.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            service_gap_t=runtime_tensor_state.last_gu_service_gap.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            service_gap_risk_t=runtime_tensor_state.last_gu_service_gap_risk.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            deadline_steps_t=runtime_tensor_state.gu_deadline_steps.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            deadline_slack_t=runtime_tensor_state.last_gu_deadline_slack.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            deadline_risk_t=runtime_tensor_state.last_gu_deadline_risk.index_select(0, selected_tensor).to(device=kernel_device, dtype=torch.float32),
            sat_state_max=(int(sat_state_max) if use_sat_subset else None),
        )
        if out_device is None:
            return batch_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return batch_tensor.to(device=out_device, dtype=torch.float32)

    def export_runtime_state_batch(self, *, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
        self._reject_native_main_kernel_legacy_adapter("export_runtime_state_batch")
        selected = self.resolve_indices(indices)
        return [self._export_runtime_state_from_core_slot(index) for index in selected]

    def load_runtime_state_batch(
        self,
        states: Sequence[dict[str, Any]],
        *,
        indices: Sequence[int] | None = None,
        refresh_observation_cache: bool = True,
        refresh_global_state_cache: bool = True,
    ) -> None:
        self._reject_native_main_kernel_legacy_adapter("load_runtime_state_batch")
        selected = self.resolve_indices(indices)
        if len(states) != len(selected):
            raise ValueError(f"Expected {len(selected)} runtime states, got {len(states)}.")
        runtime = self.native_rollout_runtime
        if runtime is not None:
            runtime.clear_main_kernel()
        self._load_slot_state_payloads(selected, states)
        self._write_runtime_state_from_serialized_states(selected, states)
        if refresh_observation_cache:
            self._refresh_native_observation_payloads(selected)
        elif refresh_global_state_cache:
            for slot in selected:
                self._slot_state_payloads[int(slot)]["_cached_global_state"] = None

    def refresh_stage_obs_cache_many(self, indices: Sequence[int] | None = None) -> None:
        self._reject_native_main_kernel_legacy_adapter("refresh_stage_obs_cache_many")
        selected_indices = self.resolve_indices(indices)
        if self._native_batch_enabled(selected_indices):
            return
        _, selected_drivers = self.select_drivers(indices)
        for driver in selected_drivers:
            refresh_stage_obs_cache(driver)

    def _native_observation_payload_current(self, env: _StructuredNativeSlotView) -> bool:
        try:
            cached_t = int(getattr(env, "_cached_access_snapshot_t", -1))
            if cached_t != int(env.t):
                return False
            if int(getattr(env, "_cached_orbit_t", -1)) != int(env.t):
                return False
            if not np.array_equal(np.asarray(env._cached_eta_uav_pos), np.asarray(env.uav_pos, dtype=np.float32)):
                return False
            if not np.array_equal(np.asarray(env._cached_eta_gu_pos), np.asarray(env.gu_pos, dtype=np.float32)):
                return False
            if np.asarray(env._cached_bw_valid_mask).shape != (int(env.cfg.num_uav), int(env.cfg.users_obs_max)):
                return False
            if np.asarray(env._cached_eta).shape != (int(env.cfg.num_uav), int(env.cfg.users_obs_max)):
                return False
            if np.asarray(env._cached_sat_obs).shape[:2] != (int(env.cfg.num_uav), int(env.cfg.sats_obs_max)):
                return False
            if np.asarray(env._cached_sat_mask).shape != (int(env.cfg.num_uav), int(env.cfg.sats_obs_max)):
                return False
            if np.asarray(env._cached_sat_valid_mask).shape != (int(env.cfg.num_uav), int(env.cfg.sats_obs_max)):
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def _refresh_native_observation_payloads(self, selected: Sequence[int], *, force: bool = True) -> None:
        for slot in selected:
            payload = self._slot_state_payloads[int(slot)]
            env = self._slot_view(int(slot))
            if not bool(force) and self._native_observation_payload_current(env):
                continue
            payload["_cached_elevation_t"] = None
            payload["_cached_elevation_matrix"] = None
            payload["_cached_backhaul_loss_t"] = None
            payload["_cached_backhaul_loss_matrix"] = None
            payload["_cached_uav_ecef"] = None
            payload["_cached_uav_vel_ecef"] = None
            payload["_cached_uav_neighbor_t"] = None
            payload["_cached_uav_neighbor_order"] = None
            payload["_cached_orbit_t"] = None
            payload["_cached_orbit_pos"] = None
            payload["_cached_orbit_vel"] = None
            env._refresh_observation_cache_from_current_state()

    def _build_native_obs_env_arrays(self, env: _StructuredNativeSlotView) -> dict[str, np.ndarray]:
        cfg = env.cfg
        num_uav = int(cfg.num_uav)
        assoc_counts, assoc_rel_centroids, _, _, _, _ = env._assoc_centroid_summary()
        uav_reward_features = (
            env._uav_reward_aligned_feature_dict(normalized=True)
            if bool(getattr(cfg, "obs_own_include_assoc_uav_cost", False))
            else None
        )
        gu_proxy_features = [np.asarray(feature, dtype=np.float32) for feature in env._gu_proxy_feature_arrays()]
        env._ensure_neighbor_cache()
        neighbor_order = np.asarray(env._cached_uav_neighbor_order, dtype=np.int64)
        danger_obs = (
            np.asarray(env._danger_neighbor_obs_batch(), dtype=np.float32)
            if bool(getattr(cfg, "danger_nbr_enabled", False))
            else None
        )

        map_scale = normalize_scale(float(cfg.map_size))
        v_scale = normalize_scale(float(cfg.v_max))
        uav_energy_scale = normalize_scale(float(cfg.uav_energy_init))
        gu_queue_scale = normalize_scale(float(cfg.queue_max_gu))
        uav_queue_scale = normalize_scale(float(cfg.queue_max_uav))
        prev_assoc = np.asarray(env.prev_association, dtype=np.int32)
        uav_pos = np.asarray(env.uav_pos, dtype=np.float32)
        uav_vel = np.asarray(env.uav_vel, dtype=np.float32)
        uav_energy = np.asarray(env.uav_energy, dtype=np.float32)
        uav_queue = np.asarray(env.uav_queue, dtype=np.float32)
        gu_pos = np.asarray(env.gu_pos, dtype=np.float32)
        gu_queue = np.asarray(env.gu_queue, dtype=np.float32)

        own = np.zeros((num_uav, int(env.own_dim)), dtype=np.float32)
        own[:, 0:2] = uav_pos / map_scale
        own[:, 2:4] = uav_vel / v_scale
        own[:, 4] = uav_energy / uav_energy_scale
        own[:, 5] = uav_queue / uav_queue_scale
        # Reserved: keep compatible shape without exposing the artificial horizon.
        own[:, 6] = 0.0
        own[:, 7] = np.asarray(assoc_counts, dtype=np.float32) / max(float(cfg.num_gu), 1.0)
        own[:, 8:10] = np.asarray(assoc_rel_centroids, dtype=np.float32)
        if uav_reward_features is not None:
            own[:, 10] = np.asarray(uav_reward_features["assoc_uav_cost"], dtype=np.float32)

        users = np.zeros((num_uav, int(cfg.users_obs_max), int(env.user_dim)), dtype=np.float32)
        users_mask = np.zeros((num_uav, int(cfg.users_obs_max)), dtype=np.float32)
        bw_valid_mask = np.zeros((num_uav, int(cfg.users_obs_max)), dtype=np.float32)
        uav_gu_rel = gu_pos[None, :, :] - uav_pos[:, None, :]
        cached_eta = np.asarray(env._cached_eta, dtype=np.float32)
        cached_candidates = env._cached_candidates if env._cached_candidates else [[] for _ in range(num_uav)]
        cached_assoc = np.asarray(getattr(env, "_cached_assoc", np.full((int(cfg.num_gu),), -1, dtype=np.int32)), dtype=np.int32)
        for uav_index in range(num_uav):
            eta_by_gu = np.zeros((int(cfg.num_gu),), dtype=np.float32)
            cand = cached_candidates[uav_index][: int(cfg.users_obs_max)]
            if cand:
                cand_arr = np.asarray(cand, dtype=np.int64)
                valid_cand = (cand_arr >= 0) & (cand_arr < int(cfg.num_gu))
                eta_by_gu[cand_arr[valid_cand]] = cached_eta[uav_index, : int(cand_arr.size)][valid_cand]
            count = min(int(cfg.num_gu), int(cfg.users_obs_max))
            if count <= 0:
                continue
            gu_ids = np.arange(count, dtype=np.int64)
            users[uav_index, :count, 0:2] = uav_gu_rel[uav_index, gu_ids] / map_scale
            users[uav_index, :count, 2] = gu_queue[gu_ids] / gu_queue_scale
            users[uav_index, :count, 3] = eta_by_gu[gu_ids]
            users[uav_index, :count, 4] = (prev_assoc[gu_ids] == uav_index).astype(np.float32, copy=False)
            feat_col = 5
            for feature in gu_proxy_features:
                users[uav_index, :count, feat_col] = feature[gu_ids]
                feat_col += 1
            users_mask[uav_index, :count] = 1.0
            bw_valid_mask[uav_index, :count] = (cached_assoc[gu_ids] == uav_index).astype(np.float32, copy=False)

        sats = np.asarray(env._cached_sat_obs, dtype=np.float32).copy()
        sats_mask = np.asarray(env._cached_sat_mask, dtype=np.float32).copy()
        sat_valid_mask = np.asarray(env._cached_sat_valid_mask, dtype=np.float32).copy()

        nbrs = np.zeros((num_uav, int(cfg.nbrs_obs_max), int(env.nbr_dim)), dtype=np.float32)
        nbrs_mask = np.zeros((num_uav, int(cfg.nbrs_obs_max)), dtype=np.float32)
        uav_rel_pos = uav_pos[None, :, :] - uav_pos[:, None, :]
        uav_rel_vel = uav_vel[None, :, :] - uav_vel[:, None, :]
        for uav_index in range(num_uav):
            order = neighbor_order[uav_index]
            valid = order[order != uav_index][: int(cfg.nbrs_obs_max)]
            count = int(valid.size)
            if count <= 0:
                continue
            nbrs[uav_index, :count, 0:2] = uav_rel_pos[uav_index, valid] / map_scale
            nbrs[uav_index, :count, 2:4] = uav_rel_vel[uav_index, valid] / v_scale
            nbrs_mask[uav_index, :count] = 1.0

        obs = {
            "own": own,
            "users": users,
            "users_mask": users_mask,
            "bw_valid_mask": bw_valid_mask,
            "sats": sats,
            "sats_mask": sats_mask,
            "sat_valid_mask": sat_valid_mask,
            "nbrs": nbrs,
            "nbrs_mask": nbrs_mask,
        }
        if danger_obs is not None:
            obs["danger_nbr"] = danger_obs.copy()
        return obs

    @staticmethod
    def _native_obs_many_from_arrays(obs_arrays: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        num_agents = int(obs_arrays["own"].shape[0])
        obs_list: list[dict[str, np.ndarray]] = []
        for agent_index in range(num_agents):
            obs_list.append(
                {
                    key: np.asarray(value[agent_index], dtype=np.float32).copy()
                    for key, value in obs_arrays.items()
                }
            )
        return obs_list

    def current_obs_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        self._reject_native_main_kernel_legacy_adapter("current_obs_batch")
        out_device = self.output_device(device)
        selected_indices = self.resolve_indices(indices)
        if self._native_batch_enabled(selected_indices):
            if not selected_indices:
                return {}
            self._refresh_native_observation_payloads(selected_indices, force=False)
            env_arrays = [self._build_native_obs_env_arrays(env) for env in self._slot_views(selected_indices)]
            sample_keys = tuple(env_arrays[0].keys())
            stacked = {
                key: np.stack([obs[key] for obs in env_arrays], axis=0).astype(np.float32, copy=False)
                for key in sample_keys
            }
            if out_device is None:
                return stacked
            return {
                key: torch.as_tensor(value, dtype=torch.float32, device=out_device)
                for key, value in stacked.items()
            }
        _, selected_drivers = self.select_drivers(indices)
        return current_obs_batch(selected_drivers, device=out_device)

    def current_obs_many(self, indices: Sequence[int] | None = None):
        self._reject_native_main_kernel_legacy_adapter("current_obs_many")
        selected_indices = self.resolve_indices(indices)
        if self._native_batch_enabled(selected_indices):
            self._refresh_native_observation_payloads(selected_indices, force=False)
            return [
                self._native_obs_many_from_arrays(self._build_native_obs_env_arrays(self._slot_view(int(slot))))
                for slot in selected_indices
            ]
        _, selected_drivers = self.select_drivers(indices)
        return [current_obs_list(driver) for driver in selected_drivers]

    def cluster_meta_batch(
        self,
        *,
        indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor | None]:
        self._reject_native_main_kernel_legacy_adapter("cluster_meta_batch")
        out_device = self.output_device(device)
        selected_indices = self.resolve_indices(indices)
        if not selected_indices:
            return {"centers": None, "counts": None}
        centers: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for index in selected_indices:
            env = self._slot_view(int(index))
            center = getattr(env, "gu_cluster_centers", None)
            count = getattr(env, "gu_cluster_counts", None)
            if center is None or count is None:
                return {"centers": None, "counts": None}
            centers.append(np.asarray(center, dtype=np.float32))
            counts.append(np.asarray(count, dtype=np.float32))
        center_batch = np.stack(centers, axis=0).astype(np.float32, copy=False)
        count_batch = np.stack(counts, axis=0).astype(np.float32, copy=False)
        if out_device is None:
            return {"centers": center_batch, "counts": count_batch}
        return {
            "centers": torch.as_tensor(center_batch, dtype=torch.float32, device=out_device),
            "counts": torch.as_tensor(count_batch, dtype=torch.float32, device=out_device),
        }

    def cluster_meta_many(self, indices: Sequence[int] | None = None):
        self._reject_native_main_kernel_legacy_adapter("cluster_meta_many")
        selected_indices = self.resolve_indices(indices)
        out = []
        for index in selected_indices:
            env = self._slot_view(int(index))
            out.append(
                {
                    "centers": None if getattr(env, "gu_cluster_centers", None) is None else np.asarray(env.gu_cluster_centers),
                    "counts": None if getattr(env, "gu_cluster_counts", None) is None else np.asarray(env.gu_cluster_counts),
                }
            )
        return out

    def sat_mask_to_ids_many(
        self,
        sat_masks: Sequence[np.ndarray],
        indices: Sequence[int] | None = None,
    ):
        self._reject_native_main_kernel_legacy_adapter("sat_mask_to_ids_many")
        selected_indices = self.resolve_indices(indices)
        if self._native_batch_enabled(selected_indices):
            if not selected_indices:
                return []
            cfg = self._cfg
            select_k = _sat_action_select_k_from_config(cfg)
            runtime = self.native_rollout_runtime
            stage_fields = None if runtime is None else runtime.main.sat_stage_fields
            if not _is_native_stage_fields(stage_fields):
                raise RuntimeError("native SAT mask decode requires fixed SAT stage fields.")
            visible_ids_t = stage_fields.visible_ids.to(device=self.tensor_device, dtype=torch.long)
            visible_mask_t = stage_fields.visible_mask.to(device=self.tensor_device, dtype=torch.bool)
            if visible_ids_t is None or visible_mask_t is None:
                raise RuntimeError("native SAT mask decode requires visible id/mask tensors.")
            sat_mask_t = _as_grouped_tensor_batch(
                sat_masks,
                expected_envs=len(selected_indices),
                trailing_shape=(int(cfg.num_uav), int(cfg.sats_obs_max)),
                dtype=torch.float32,
                device=visible_ids_t.device,
            )
            width = min(int(cfg.sats_obs_max), int(visible_ids_t.shape[-1]), int(sat_mask_t.shape[-1]))
            active_t = (sat_mask_t[:, :, :width] > 0.5) & visible_mask_t[:, :, :width]
            slot_order_t = torch.arange(width, dtype=torch.long, device=visible_ids_t.device)
            sort_key_t = torch.where(
                active_t,
                slot_order_t.view(1, 1, width),
                torch.full((len(selected_indices), int(cfg.num_uav), width), width, dtype=torch.long, device=visible_ids_t.device),
            )
            top_slots_t = torch.argsort(sort_key_t, dim=2, stable=True)[:, :, : int(select_k)]
            selected_ids_t = torch.gather(visible_ids_t[:, :, :width], 2, top_slots_t)
            selected_valid_t = torch.gather(active_t, 2, top_slots_t)
            decoded_t = torch.where(selected_valid_t, selected_ids_t, torch.full_like(selected_ids_t, -1))
            return [
                decoded_t[env_index].detach().cpu().numpy().astype(np.int64, copy=False)
                for env_index in range(int(decoded_t.shape[0]))
            ]
        _, selected_drivers = self.select_drivers(indices)
        return [sat_mask_to_ids(driver, sat_mask) for driver, sat_mask in zip(selected_drivers, sat_masks)]

    def build_native_world_from_stage_fields(
        self,
        fields_obj: _NativeStageTensorFields,
        *,
        stage_id: int,
        device: torch.device | str | None = None,
    ) -> StructuredWorldState:
        domains = self._native_main_kernel_typed_domains()
        runtime = self.native_rollout_runtime
        selected_mapping = None if runtime is None else getattr(runtime.main, "selected_env_mapping", None)
        t_for_world_t = None
        if torch.is_tensor(selected_mapping):
            batch_size = _stage_fields_batch_size(fields_obj)
            if int(selected_mapping.numel()) == int(batch_size):
                state = self._runtime_tensor_state
                selected_t = selected_mapping.to(device=state.sat_queue.device, dtype=torch.long).reshape(batch_size)
                t_for_world_t = state.t.index_select(0, selected_t)
                fields_obj = _stage_fields_with_updates(
                    fields_obj,
                    {
                        "arrival_ref_bits_per_step": state.arrival_ref_bits_per_step.index_select(0, selected_t),
                        "expected_arrival_rate_vec": state.last_gu_arrival_rate_vec.index_select(0, selected_t),
                        "gu_ema": state.gu_workload_ema.index_select(0, selected_t),
                        "uav_ema": state.uav_workload_ema.index_select(0, selected_t),
                        "gu_drop": state.gu_drop.index_select(0, selected_t),
                        "uav_drop": state.uav_drop.index_select(0, selected_t),
                        "last_gu_arrival": state.last_gu_arrival.index_select(0, selected_t),
                        "last_gu_outflow": state.last_gu_outflow.index_select(0, selected_t),
                        "last_gu_to_uav_inflow_by_uav": state.last_gu_to_uav_inflow_by_uav.index_select(0, selected_t),
                        "last_uav_to_sat_outflow_matrix": state.last_uav_to_sat_outflow_matrix.index_select(0, selected_t),
                        "last_bw_fraction_by_uav_gu": state.last_bw_fraction_by_uav_gu.index_select(0, selected_t),
                        "last_access_interference_by_uav": state.last_access_interference_by_uav.index_select(0, selected_t),
                        "sat_ema": state.sat_workload_ema.index_select(0, selected_t),
                        "sat_drop": state.sat_drop.index_select(0, selected_t),
                        "last_sat_processed": state.last_sat_processed.index_select(0, selected_t),
                        "last_selected_mask_by_uav_sat": state.last_selected_mask_by_uav_sat.index_select(0, selected_t),
                    },
                )
        else:
            state = self._runtime_tensor_state
            if state is not None and torch.is_tensor(getattr(state, "t", None)):
                batch_size = _stage_fields_batch_size(fields_obj)
                if int(state.t.numel()) == int(batch_size):
                    t_for_world_t = state.t
        return _build_world_from_stage_fields_direct_tensor_impl(
            local_obs_params=domains.local_obs,
            fields_obj=fields_obj,
            stage_id=int(stage_id),
            t_t=t_for_world_t,
            access_rate_static_params=domains.access_rate,
            device=self.output_device(device),
        )

    def reset_many(self, seeds: Sequence[int | None] | None = None, indices: Sequence[int] | None = None) -> None:
        selected = self.resolve_indices(indices)
        runtime = self.native_rollout_runtime
        self._native_hot_replay_runtime = None
        self._native_hot_replay_tensor_state = None
        self._native_hot_replay_cfg = None
        self._native_hot_replay_torch_rng = None
        self._native_hot_replay_bound_kernels = None
        self._native_sub_batch_runtime = None
        self._native_sub_batch_tensor_state = None
        self._native_sub_batch_cfg = None
        self._native_sub_batch_torch_rng = None
        self._native_sub_batch_bound_kernels = None
        if runtime is not None and not self._native_main_kernel_workspace_active(runtime):
            runtime.clear_main_kernel()
        if seeds is None:
            seeds = [None] * len(selected)
        if len(seeds) != len(selected):
            raise ValueError(f"Expected {len(selected)} reset seeds, got {len(seeds)}.")
        if selected and all(seed is not None for seed in seeds):
            try:
                self._native_torch_rng.manual_seed(int(seeds[0]))
            except (TypeError, ValueError):
                pass
        reset_tapes = self._build_native_reset_tapes(selected, seeds)
        if reset_tapes is not None and len(reset_tapes) != len(selected):
            raise RuntimeError("native reset tape count does not match selected env count.")
        states = [
            self._build_native_reset_state(
                int(slot),
                seed,
                reset_tape=None if reset_tapes is None else reset_tapes[offset],
            )
            for offset, (slot, seed) in enumerate(zip(selected, seeds))
        ]
        self._load_slot_state_payloads(selected, states)
        self._write_runtime_state_from_serialized_states(selected, states)
        self._refresh_native_observation_payloads(selected)
        if self.native_rollout_runtime is not None:
            self._publish_runtime_reset_random_tape(selected)

    def reset_at(self, index: int, seed: int | None = None) -> None:
        self.reset_many([seed], indices=[int(index)])

    def close(self) -> None:
        return None
