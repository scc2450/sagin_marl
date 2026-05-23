from __future__ import annotations

BW_EGO_DIM = 11
BW_SAT_TOKEN_DIM = 9
BW_GU_TOKEN_DIM = 13

BW_OBJECTIVE_NORM = "per_latent_dim"

BW_EGO_FIELDS = (
    "uav_queue_steps",
    "uav_queue_fill",
    "uav_last_inflow_steps",
    "uav_last_outflow_steps",
    "uav_last_drop_steps",
    "uav_service_ema_steps",
    "uav_local_cost_log_ratio",
    "uav_last_total_cost_log_ratio",
    "uav_last_workload_log1p",
    "uav_last_access_interference_log1p",
    "remaining_horizon_frac",
)

BW_EGO_REMAINING_HORIZON_FRAC = 10

BW_SAT_TOKEN_FIELDS = (
    "prefix_backhaul_capacity_steps",
    "sat_queue_steps",
    "sat_queue_fill",
    "sat_last_incoming_steps",
    "sat_last_processed_steps",
    "sat_last_drop_steps",
    "sat_service_ema_steps",
    "sat_cost_log_ratio",
    "sat_last_workload_log1p",
)

BW_GU_TOKEN_FIELDS = (
    "gu_queue_steps",
    "gu_queue_fill",
    "gu_expected_arrival_steps",
    "gu_last_arrival_steps",
    "gu_last_outflow_steps",
    "gu_last_drop_steps",
    "gu_service_ema_steps",
    "gu_local_cost_log_ratio",
    "gu_last_total_cost_log_ratio",
    "gu_last_workload_log1p",
    "access_rate_full_bw_ref_steps",
    "cross_interference_mean_log1p",
    "cross_interference_max_log1p",
)

assert len(BW_EGO_FIELDS) == BW_EGO_DIM
assert len(BW_SAT_TOKEN_FIELDS) == BW_SAT_TOKEN_DIM
assert len(BW_GU_TOKEN_FIELDS) == BW_GU_TOKEN_DIM
