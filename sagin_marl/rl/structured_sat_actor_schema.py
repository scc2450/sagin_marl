from __future__ import annotations

SAT_EGO_DIM = 13
SAT_DEMAND_DIM = 8
SAT_ROLE_DIM = 1
SAT_TOKEN_DIM = 26

(
    EGO_UAV_QUEUE_STEPS,
    EGO_UAV_QUEUE_FILL,
    EGO_UAV_LAST_INFLOW_STEPS,
    EGO_UAV_LAST_OUTFLOW_STEPS,
    EGO_UAV_LAST_DROP_STEPS,
    EGO_UAV_SERVICE_EMA_STEPS,
    EGO_UAV_LOCAL_COST_LOG_RATIO,
    EGO_UAV_LAST_TOTAL_COST_LOG_RATIO,
    EGO_UAV_LAST_WORKLOAD_LOG1P,
    EGO_UAV_LAST_ACCESS_INTERFERENCE_LOG1P,
    EGO_LAST_SELECTED_COUNT_FRAC,
    EGO_LAST_BACKHAUL_OUTFLOW_STEPS,
    EGO_REMAINING_HORIZON_FRAC,
) = range(SAT_EGO_DIM)

SAT_EGO_FIELDS = (
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
    "last_selected_count_frac",
    "last_backhaul_outflow_steps",
    "remaining_horizon_frac",
)

(
    DEMAND_CELL_GU_COUNT_FRAC,
    DEMAND_CELL_QUEUE_STEPS_SUM,
    DEMAND_CELL_EXPECTED_ARRIVAL_STEPS_SUM,
    DEMAND_CELL_LAST_ARRIVAL_STEPS_SUM,
    DEMAND_CELL_LAST_OUTFLOW_STEPS_SUM,
    DEMAND_CELL_LAST_DROP_STEPS_SUM,
    DEMAND_CELL_LAST_WORKLOAD_LOG1P_SUM,
    DEMAND_CELL_ACCESS_RATE_FULL_BW_REF_SUM,
) = range(SAT_DEMAND_DIM)

SAT_DEMAND_FIELDS = (
    "cell_gu_count_frac",
    "cell_queue_steps_sum",
    "cell_expected_arrival_steps_sum",
    "cell_last_arrival_steps_sum",
    "cell_last_outflow_steps_sum",
    "cell_last_drop_steps_sum",
    "cell_last_workload_log1p_sum",
    "cell_access_rate_full_bw_ref_sum",
)

(
    SAT_QUEUE_STEPS,
    SAT_QUEUE_FILL,
    SAT_LAST_INCOMING_STEPS,
    SAT_LAST_PROCESSED_STEPS,
    SAT_LAST_DROP_STEPS,
    SAT_SERVICE_EMA_STEPS,
    SAT_COST_LOG_RATIO,
    SAT_LAST_WORKLOAD_LOG1P,
    SAT_LAST_SELECTED_LOAD_FRAC,
    SAT_PROC_CAPACITY_STEPS,
    US_REL_X_NORM,
    US_REL_Y_NORM,
    US_REL_Z_NORM,
    US_REL_VX_NORM,
    US_REL_VY_NORM,
    US_REL_VZ_NORM,
    US_RANGE_NORM,
    US_RADIAL_VELOCITY_NORM,
    US_ELEVATION_NORM,
    US_DOPPLER_NORM,
    US_DOPPLER_MARGIN,
    US_BACKHAUL_SE_REF,
    US_VISIBLE_FLAG,
    US_VALID_FLAG,
    US_LAST_SELECTED_FLAG,
    US_LAST_OUTFLOW_STEPS,
) = range(SAT_TOKEN_DIM)

SAT_TOKEN_FIELDS = (
    "sat_queue_steps",
    "sat_queue_fill",
    "sat_last_incoming_steps",
    "sat_last_processed_steps",
    "sat_last_drop_steps",
    "sat_service_ema_steps",
    "sat_cost_log_ratio",
    "sat_last_workload_log1p",
    "sat_last_selected_load_frac",
    "sat_proc_capacity_steps",
    "us_rel_x_norm",
    "us_rel_y_norm",
    "us_rel_z_norm",
    "us_rel_vx_norm",
    "us_rel_vy_norm",
    "us_rel_vz_norm",
    "us_range_norm",
    "us_radial_velocity_norm",
    "us_elevation_norm",
    "us_doppler_norm",
    "us_doppler_margin",
    "us_backhaul_se_ref",
    "us_visible_flag",
    "us_valid_flag",
    "us_last_selected_flag",
    "us_last_outflow_steps",
)

assert len(SAT_EGO_FIELDS) == SAT_EGO_DIM
assert len(SAT_DEMAND_FIELDS) == SAT_DEMAND_DIM
assert len(SAT_TOKEN_FIELDS) == SAT_TOKEN_DIM
