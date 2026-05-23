from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.structured_batch_env_core import (
    _allocate_native_main_kernel_stage_fields,
    _active_sat_ids_from_visible_ids_tensor,
    _build_world_from_stage_fields_direct_tensor_impl,
    _build_world_from_packed_specs_tensor_impl,
    _native_typed_domains_from_cfg,
)
from sagin_marl.rl import structured_critic_schema as schema


def _world_inputs(device: torch.device):
    cfg = SaginConfig(num_uav=2, num_gu=3, num_sat=2, users_obs_max=3, sats_obs_max=2, visible_sats_max=2, sat_num_select=1)
    cfg.fading_enabled = False
    domains = _native_typed_domains_from_cfg(cfg, num_envs=1)
    batch = 1
    num_uav = int(cfg.num_uav)
    num_gu = int(cfg.num_gu)
    num_sat = int(cfg.num_sat)
    active = 2
    arrival_ref = torch.tensor([600.0], dtype=torch.float32, device=device)
    kwargs = dict(
        local_obs_params=domains.local_obs,
        stage_ids_t=torch.tensor([schema.CRITIC_STAGE_BW], dtype=torch.long, device=device),
        effective_b_backhaul_per_sat_t=torch.tensor([float(cfg.b_backhaul_per_sat)], dtype=torch.float32, device=device),
        uav_pos_t=torch.tensor([[[0.0, 0.0], [100.0, 0.0]]], dtype=torch.float32, device=device),
        uav_vel_t=torch.tensor([[[1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32, device=device),
        uav_energy_t=torch.full((batch, num_uav), float(cfg.uav_energy_init), dtype=torch.float32, device=device),
        uav_queue_t=torch.tensor([[300.0, 600.0]], dtype=torch.float32, device=device),
        uav_assoc_uav_cost_t=torch.zeros((batch, num_uav), dtype=torch.float32, device=device),
        gu_pos_t=torch.tensor([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]], dtype=torch.float32, device=device),
        gu_queue_t=torch.tensor([[200.0, 400.0, 0.0]], dtype=torch.float32, device=device),
        gu_proxy_features_t=torch.zeros((batch, num_gu, 0), dtype=torch.float32, device=device),
        assoc_t=torch.tensor([[0, 0, 1]], dtype=torch.long, device=device),
        candidate_flag_t=torch.ones((batch, num_uav, num_gu), dtype=torch.float32, device=device),
        bw_valid_flag_t=torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device),
        prev_assoc_flag_t=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float32, device=device),
        eta_ref_feature_t=torch.full((batch, num_uav, num_gu), 2.5, dtype=torch.float32, device=device),
        sat_pos_active_t=torch.tensor([[[7.0e6, 0.0, 0.0], [0.0, 7.0e6, 0.0]]], dtype=torch.float32, device=device),
        sat_vel_active_t=torch.tensor([[[0.0, 7500.0, 0.0], [-7500.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
        sat_queue_active_t=torch.tensor([[300.0, 900.0]], dtype=torch.float32, device=device),
        sat_load_active_t=torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device),
        sat_cost_norm_active_t=torch.zeros((batch, active), dtype=torch.float32, device=device),
        sat_active_mask_t=torch.ones((batch, active), dtype=torch.bool, device=device),
        rel_pos_active_t=torch.full((batch, num_uav, active, 3), 1.0e6, dtype=torch.float32, device=device),
        rel_vel_active_t=torch.full((batch, num_uav, active, 3), 100.0, dtype=torch.float32, device=device),
        gain_active_t=torch.full((batch, num_uav, active), 1.0e-12, dtype=torch.float32, device=device),
        nu_eff_active_t=torch.zeros((batch, num_uav, active), dtype=torch.float32, device=device),
        visible_flag_active_t=torch.ones((batch, num_uav, active), dtype=torch.float32, device=device),
        valid_flag_active_t=torch.ones((batch, num_uav, active), dtype=torch.float32, device=device),
        current_sel_flag_active_t=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, device=device),
        active_sat_ids_t=torch.tensor([[0, 1]], dtype=torch.long, device=device),
        arrival_ref_t=arrival_ref,
        expected_arrival_rate_vec_t=torch.tensor([[20.0, 40.0, 0.0]], dtype=torch.float32, device=device),
        gu_ema_t=torch.full((batch, num_gu), 200.0, dtype=torch.float32, device=device),
        uav_ema_t=torch.full((batch, num_uav), 300.0, dtype=torch.float32, device=device),
        sat_ema_active_t=torch.full((batch, active), 300.0, dtype=torch.float32, device=device),
        last_gu_arrival_t=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device),
        last_gu_outflow_t=torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32, device=device),
        gu_drop_t=torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float32, device=device),
        uav_drop_t=torch.tensor([[10.0, 11.0]], dtype=torch.float32, device=device),
        sat_drop_active_t=torch.tensor([[12.0, 13.0]], dtype=torch.float32, device=device),
        last_gu_to_uav_inflow_by_uav_t=torch.tensor([[14.0, 15.0]], dtype=torch.float32, device=device),
        last_uav_to_sat_outflow_active_t=torch.tensor([[[16.0, 0.0], [0.0, 17.0]]], dtype=torch.float32, device=device),
        last_bw_fraction_by_uav_gu_t=torch.tensor([[[0.7, 0.3, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device),
        last_access_interference_by_uav_t=torch.tensor([[1.0e-12, 2.0e-12]], dtype=torch.float32, device=device),
        last_selected_flag_active_t=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, device=device),
        last_sat_processed_active_t=torch.tensor([[18.0, 19.0]], dtype=torch.float32, device=device),
        sat_proc_capacity_active_t=torch.tensor([[20.0, 21.0]], dtype=torch.float32, device=device),
    )
    return cfg, kwargs


def test_native_active_sat_ids_preserve_visible_dedup_order() -> None:
    visible_ids = torch.tensor(
        [
            [[4, 1, 3], [1, 2, 4]],
            [[0, 2, 1], [2, 5, 3]],
        ],
        dtype=torch.long,
    )
    visible_mask = torch.tensor(
        [
            [[True, True, False], [True, True, True]],
            [[False, True, True], [True, False, True]],
        ],
        dtype=torch.bool,
    )
    active = _active_sat_ids_from_visible_ids_tensor(
        visible_ids,
        visible_mask,
        active_width=5,
        num_sat=6,
    )
    expected = torch.tensor(
        [
            [4, 1, 2, -1, -1],
            [2, 1, 3, -1, -1],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(active, expected)


def test_tensor_native_world_builder_uses_reward_aligned_flow_scales() -> None:
    device = torch.device("cpu")
    cfg, kwargs = _world_inputs(device)
    world = _build_world_from_packed_specs_tensor_impl(**kwargs)
    gu_flow_ref = 600.0 / float(cfg.num_gu)
    uav_flow_ref = 600.0 / float(cfg.num_uav)
    sat_flow_ref = 600.0 / 2.0

    np.testing.assert_allclose(
        world["gu_nodes"][0, :, schema.GU_QUEUE_STEPS].numpy(),
        np.asarray([200.0, 400.0, 0.0], dtype=np.float32) / gu_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["uav_nodes"][0, :, schema.UAV_QUEUE_STEPS].numpy(),
        np.asarray([300.0, 600.0], dtype=np.float32) / uav_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["sat_nodes"][0, :, schema.SAT_QUEUE_STEPS].numpy(),
        np.asarray([300.0, 900.0], dtype=np.float32) / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(world["uav_gu_edges"][0, :, :, schema.UG_ACCESS_SE_REF].numpy(), 2.5, atol=1.0e-6)
    np.testing.assert_allclose(
        world["uav_gu_edges"][0, :, :, schema.UG_LAST_BW_FRACTION].numpy(),
        kwargs["last_bw_fraction_by_uav_gu_t"].numpy()[0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["uav_gu_edges"][0, :, :, schema.UG_LAST_SERVED_FLAG].numpy(),
        kwargs["last_bw_fraction_by_uav_gu_t"].numpy()[0],
        atol=1.0e-6,
    )
    assert world["global_scalars"][0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN].item() == 1.0
    assert world["uav_sat_edges"][0, 0, 0, schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS].item() > 0.0


def test_tensor_native_world_builder_stage_prefix_flags_gate_prefix_fields() -> None:
    device = torch.device("cpu")
    _cfg, kwargs = _world_inputs(device)
    kwargs["stage_ids_t"] = torch.tensor([schema.CRITIC_STAGE_SAT], dtype=torch.long, device=device)
    world = _build_world_from_packed_specs_tensor_impl(**kwargs)
    assert torch.all(world["uav_gu_edges"][..., schema.UG_PREFIX_BW_VALID_KNOWN] == 1.0)
    assert torch.all(world["uav_sat_edges"][..., schema.US_PREFIX_SELECTED_KNOWN] == 0.0)
    assert torch.all(world["uav_sat_edges"][..., schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS] == 0.0)
    assert world["global_scalars"][0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN].item() == 0.0


def test_direct_stage_fields_world_builder_uses_stage_when_filling_prefix_fields() -> None:
    device = torch.device("cpu")
    cfg, kwargs = _world_inputs(device)
    domains = _native_typed_domains_from_cfg(cfg, num_envs=1)
    fields = _allocate_native_main_kernel_stage_fields(
        cfg=cfg,
        local_obs_params=domains.local_obs,
        batch_size=1,
        active_width=2,
        max_keep=2,
        select_k=1,
        device=device,
    )
    fields.stage_id.fill_(schema.CRITIC_STAGE_BW)
    fields.effective_b_backhaul_per_sat.copy_(kwargs["effective_b_backhaul_per_sat_t"])
    fields.uav_pos.copy_(kwargs["uav_pos_t"])
    fields.uav_vel.copy_(kwargs["uav_vel_t"])
    fields.uav_energy.copy_(kwargs["uav_energy_t"])
    fields.uav_queue.copy_(kwargs["uav_queue_t"])
    fields.gu_pos.copy_(kwargs["gu_pos_t"])
    fields.gu_queue.copy_(kwargs["gu_queue_t"])
    fields.arrival_ref_bits_per_step.copy_(kwargs["arrival_ref_t"])
    fields.expected_arrival_rate_vec.copy_(kwargs["expected_arrival_rate_vec_t"])
    fields.gu_ema.copy_(kwargs["gu_ema_t"])
    fields.uav_ema.copy_(kwargs["uav_ema_t"])
    fields.gu_drop.copy_(kwargs["gu_drop_t"])
    fields.uav_drop.copy_(kwargs["uav_drop_t"])
    fields.last_gu_arrival.copy_(kwargs["last_gu_arrival_t"])
    fields.last_gu_outflow.copy_(kwargs["last_gu_outflow_t"])
    fields.last_gu_to_uav_inflow_by_uav.copy_(kwargs["last_gu_to_uav_inflow_by_uav_t"])
    fields.last_uav_to_sat_outflow_matrix.copy_(torch.tensor([[[16.0, 0.0], [0.0, 17.0]]], dtype=torch.float32, device=device))
    fields.last_bw_fraction_by_uav_gu.copy_(kwargs["last_bw_fraction_by_uav_gu_t"])
    fields.last_access_interference_by_uav.copy_(kwargs["last_access_interference_by_uav_t"])
    fields.sat_queue.copy_(kwargs["sat_queue_active_t"])
    fields.sat_loads.copy_(kwargs["sat_load_active_t"])
    fields.sat_ema.copy_(torch.tensor([[100.0, 1000.0]], dtype=torch.float32, device=device))
    fields.sat_drop.copy_(kwargs["sat_drop_active_t"])
    fields.last_sat_processed.copy_(kwargs["last_sat_processed_active_t"])
    fields.last_selected_mask_by_uav_sat.copy_(kwargs["last_selected_flag_active_t"])
    fields.sat_pos.copy_(kwargs["sat_pos_active_t"])
    fields.sat_vel.copy_(kwargs["sat_vel_active_t"])
    fields.assoc.copy_(kwargs["assoc_t"])
    fields.prev_association.copy_(torch.tensor([[0, 1, 1]], dtype=torch.long, device=device))
    fields.candidate_indices.copy_(torch.tensor([[[0, 1, 2], [0, 1, 2]]], dtype=torch.long, device=device))
    fields.candidate_mask.fill_(True)
    fields.bw_valid_mask.copy_(torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device))
    fields.sat_selection_matrix.copy_(torch.tensor([[[0], [1]]], dtype=torch.long, device=device))
    fields.candidate_flag.copy_(kwargs["candidate_flag_t"])
    fields.bw_valid_flag.copy_(kwargs["bw_valid_flag_t"])
    fields.prev_assoc_flag.copy_(kwargs["prev_assoc_flag_t"])
    fields.eta_ref_feature.copy_(kwargs["eta_ref_feature_t"])
    fields.eta_slots.zero_()
    fields.gu_proxy_features.zero_()
    fields.uav_assoc_uav_cost.zero_()
    fields.sat_cost_norm.zero_()
    fields.access_gain_matrix.zero_()
    fields.visible_ids.copy_(torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long, device=device))
    fields.visible_mask.fill_(True)
    fields.visible_flag_all.fill_(1.0)
    fields.elevation_matrix.zero_()
    fields.uav_ecef_all.zero_()
    fields.uav_vel_ecef_all.zero_()
    fields.active_sat_ids.copy_(kwargs["active_sat_ids_t"])
    fields.sat_pos_active.copy_(kwargs["sat_pos_active_t"])
    fields.sat_vel_active.copy_(kwargs["sat_vel_active_t"])
    fields.sat_queue_active.copy_(kwargs["sat_queue_active_t"])
    fields.sat_load_active.copy_(kwargs["sat_load_active_t"])
    fields.sat_cost_norm_active.zero_()
    fields.us_rel_pos_active.copy_(kwargs["rel_pos_active_t"])
    fields.us_rel_vel_active.copy_(kwargs["rel_vel_active_t"])
    fields.us_gain_active.copy_(kwargs["gain_active_t"])
    fields.us_nu_eff_active.copy_(kwargs["nu_eff_active_t"])
    fields.visible_flag_active.copy_(kwargs["visible_flag_active_t"])
    fields.us_valid_flag_active.copy_(kwargs["valid_flag_active_t"])

    bw_world = _build_world_from_stage_fields_direct_tensor_impl(
        local_obs_params=domains.local_obs,
        fields_obj=fields,
        stage_id=schema.CRITIC_STAGE_BW,
        access_rate_static_params=domains.access_rate,
        device=device,
    )
    assert torch.all(bw_world.uav_sat_edges[..., schema.US_PREFIX_SELECTED_KNOWN][bw_world.uav_sat_mask] == 1.0)
    assert torch.all(bw_world.gu_nodes[..., schema.GU_PREFIX_COST_KNOWN] == 1.0)
    assert bw_world.global_scalars[0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN].item() == 1.0
    assert torch.all(bw_world.uav_gu_edges[..., schema.UG_ACCESS_SE_REF] == 0.0)
    direct_sat_flow_ref = 600.0 / float(domains.local_obs.sat_active_ref_count)
    np.testing.assert_allclose(
        bw_world.sat_nodes[0, :, schema.SAT_LAST_DROP_STEPS].numpy(),
        np.asarray([12.0, 13.0], dtype=np.float32) / direct_sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        bw_world.sat_nodes[0, :, schema.SAT_LAST_PROCESSED_STEPS].numpy(),
        np.asarray([18.0, 19.0], dtype=np.float32) / direct_sat_flow_ref,
        atol=1.0e-6,
    )

    sat_world = _build_world_from_stage_fields_direct_tensor_impl(
        local_obs_params=domains.local_obs,
        fields_obj=fields,
        stage_id=schema.CRITIC_STAGE_SAT,
        access_rate_static_params=domains.access_rate,
        device=device,
    )
    assert torch.all(sat_world.uav_gu_edges[..., schema.UG_PREFIX_BW_VALID_KNOWN] == 1.0)
    assert torch.all(sat_world.uav_sat_edges[..., schema.US_PREFIX_SELECTED_KNOWN] == 0.0)
    assert torch.all(sat_world.gu_nodes[..., schema.GU_PREFIX_COST_KNOWN] == 0.0)
    assert sat_world.global_scalars[0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN].item() == 0.0


def test_direct_stage_fields_world_builder_tracks_full_sat_residuals() -> None:
    device = torch.device("cpu")
    cfg, kwargs = _world_inputs(device)
    domains = _native_typed_domains_from_cfg(cfg, num_envs=1)
    fields = _allocate_native_main_kernel_stage_fields(
        cfg=cfg,
        local_obs_params=domains.local_obs,
        batch_size=1,
        active_width=2,
        max_keep=2,
        select_k=1,
        device=device,
    )
    fields.stage_id.fill_(schema.CRITIC_STAGE_SAT)
    fields.effective_b_backhaul_per_sat.copy_(kwargs["effective_b_backhaul_per_sat_t"])
    fields.uav_pos.copy_(kwargs["uav_pos_t"])
    fields.uav_vel.copy_(kwargs["uav_vel_t"])
    fields.uav_energy.copy_(kwargs["uav_energy_t"])
    fields.uav_queue.copy_(kwargs["uav_queue_t"])
    fields.gu_pos.copy_(kwargs["gu_pos_t"])
    fields.gu_queue.copy_(kwargs["gu_queue_t"])
    fields.arrival_ref_bits_per_step.copy_(kwargs["arrival_ref_t"])
    fields.expected_arrival_rate_vec.copy_(kwargs["expected_arrival_rate_vec_t"])
    fields.gu_ema.copy_(kwargs["gu_ema_t"])
    fields.uav_ema.copy_(kwargs["uav_ema_t"])
    fields.gu_drop.copy_(kwargs["gu_drop_t"])
    fields.uav_drop.copy_(kwargs["uav_drop_t"])
    fields.last_gu_arrival.copy_(kwargs["last_gu_arrival_t"])
    fields.last_gu_outflow.copy_(kwargs["last_gu_outflow_t"])
    fields.last_gu_to_uav_inflow_by_uav.copy_(kwargs["last_gu_to_uav_inflow_by_uav_t"])
    fields.last_uav_to_sat_outflow_matrix.copy_(torch.tensor([[[16.0, 0.0], [0.0, 17.0]]], dtype=torch.float32, device=device))
    fields.last_bw_fraction_by_uav_gu.copy_(kwargs["last_bw_fraction_by_uav_gu_t"])
    fields.last_access_interference_by_uav.copy_(kwargs["last_access_interference_by_uav_t"])
    fields.sat_queue.copy_(torch.tensor([[300.0, 900.0]], dtype=torch.float32, device=device))
    fields.sat_loads.copy_(kwargs["sat_load_active_t"])
    fields.sat_ema.copy_(torch.tensor([[100.0, 1000.0]], dtype=torch.float32, device=device))
    fields.sat_drop.copy_(torch.tensor([[12.0, 13.0]], dtype=torch.float32, device=device))
    fields.last_sat_processed.copy_(torch.tensor([[18.0, 19.0]], dtype=torch.float32, device=device))
    fields.last_selected_mask_by_uav_sat.copy_(kwargs["last_selected_flag_active_t"])
    fields.sat_pos.copy_(kwargs["sat_pos_active_t"])
    fields.sat_vel.copy_(kwargs["sat_vel_active_t"])
    fields.assoc.copy_(kwargs["assoc_t"])
    fields.prev_association.copy_(torch.tensor([[0, 1, 1]], dtype=torch.long, device=device))
    fields.candidate_indices.copy_(torch.tensor([[[0, 1, 2], [0, 1, 2]]], dtype=torch.long, device=device))
    fields.candidate_mask.fill_(True)
    fields.bw_valid_mask.copy_(kwargs["bw_valid_flag_t"])
    fields.sat_selection_matrix.copy_(torch.tensor([[[0], [0]]], dtype=torch.long, device=device))
    fields.candidate_flag.copy_(kwargs["candidate_flag_t"])
    fields.bw_valid_flag.copy_(kwargs["bw_valid_flag_t"])
    fields.prev_assoc_flag.copy_(kwargs["prev_assoc_flag_t"])
    fields.eta_ref_feature.copy_(kwargs["eta_ref_feature_t"])
    fields.eta_slots.zero_()
    fields.gu_proxy_features.zero_()
    fields.uav_assoc_uav_cost.zero_()
    fields.sat_cost_norm.zero_()
    fields.access_gain_matrix.zero_()
    fields.visible_ids.copy_(torch.tensor([[[0, -1], [0, -1]]], dtype=torch.long, device=device))
    fields.visible_mask.copy_(torch.tensor([[[True, False], [True, False]]], dtype=torch.bool, device=device))
    fields.visible_flag_all.zero_()
    fields.visible_flag_all[:, :, 0] = 1.0
    fields.elevation_matrix.zero_()
    fields.uav_ecef_all.zero_()
    fields.uav_vel_ecef_all.zero_()
    fields.active_sat_ids.copy_(torch.tensor([[0, -1]], dtype=torch.long, device=device))
    fields.sat_pos_active.copy_(torch.tensor([[[7.0e6, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32, device=device))
    fields.sat_vel_active.copy_(torch.tensor([[[0.0, 7500.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32, device=device))
    fields.sat_queue_active.copy_(torch.tensor([[300.0, 0.0]], dtype=torch.float32, device=device))
    fields.sat_load_active.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device))
    fields.sat_cost_norm_active.zero_()
    fields.us_rel_pos_active.copy_(kwargs["rel_pos_active_t"])
    fields.us_rel_pos_active[:, :, 1, :].zero_()
    fields.us_rel_vel_active.copy_(kwargs["rel_vel_active_t"])
    fields.us_rel_vel_active[:, :, 1, :].zero_()
    fields.us_gain_active.copy_(kwargs["gain_active_t"])
    fields.us_gain_active[:, :, 1].zero_()
    fields.us_nu_eff_active.zero_()
    fields.visible_flag_active.zero_()
    fields.visible_flag_active[:, :, 0] = 1.0
    fields.us_valid_flag_active.zero_()
    fields.us_valid_flag_active[:, :, 0] = 1.0

    world = _build_world_from_stage_fields_direct_tensor_impl(
        local_obs_params=domains.local_obs,
        fields_obj=fields,
        stage_id=schema.CRITIC_STAGE_SAT,
        access_rate_static_params=domains.access_rate,
        device=device,
    )
    sat_flow_ref = 600.0 / float(domains.local_obs.sat_active_ref_count)
    assert world.sat_ids[0, 0].item() == 0
    assert world.sat_ids[0, 1].item() == -1
    np.testing.assert_allclose(
        world.global_scalars[0, schema.GLOBAL_TOTAL_SAT_DROP_STEPS].item(),
        (12.0 + 13.0) / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world.global_scalars[0, schema.GLOBAL_NON_TOKEN_SAT_DROP_STEPS].item(),
        13.0 / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world.global_scalars[0, schema.GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS].item(),
        19.0 / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world.global_scalars[0, schema.GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS].item(),
        (1.0 / 1000.0) * 900.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world.global_scalars[0, schema.GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS].item(),
        (1.0 / 1000.0) * 13.0,
        atol=1.0e-6,
    )


def test_tensor_native_world_builder_uses_route_specific_last_and_prefix_costs() -> None:
    device = torch.device("cpu")
    cfg, kwargs = _world_inputs(device)
    kwargs["gu_ema_t"] = torch.tensor([[200.0, 400.0, 800.0]], dtype=torch.float32, device=device)
    kwargs["uav_ema_t"] = torch.tensor([[300.0, 600.0]], dtype=torch.float32, device=device)
    kwargs["sat_ema_active_t"] = torch.tensor([[100.0, 1000.0]], dtype=torch.float32, device=device)
    kwargs["last_selected_flag_active_t"] = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    kwargs["current_sel_flag_active_t"] = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    world = _build_world_from_packed_specs_tensor_impl(**kwargs)
    gu_flow_ref = 600.0 / float(cfg.num_gu)
    uav_flow_ref = 600.0 / float(cfg.num_uav)
    sat_flow_ref = 600.0 / 2.0
    gu_total_cost_ref = (1.0 / gu_flow_ref) + (1.0 / uav_flow_ref) + (1.0 / sat_flow_ref)
    uav_total_cost_ref = (1.0 / uav_flow_ref) + (1.0 / sat_flow_ref)

    sat_cost = np.asarray([1.0 / 100.0, 1.0 / 1000.0], dtype=np.float32)
    local_uav = np.asarray([1.0 / 300.0, 1.0 / 600.0], dtype=np.float32)
    local_gu = np.asarray([1.0 / 200.0, 1.0 / 400.0, 1.0 / 800.0], dtype=np.float32)
    last_uav_cost = local_uav + np.asarray([sat_cost[1], sat_cost[0]], dtype=np.float32)
    prefix_uav_cost = local_uav + np.asarray([sat_cost[0], sat_cost[1]], dtype=np.float32)
    last_gu_cost = local_gu + np.asarray([last_uav_cost[0], last_uav_cost[1], last_uav_cost[1]], dtype=np.float32)
    prefix_gu_cost = local_gu + np.asarray([prefix_uav_cost[0], prefix_uav_cost[0], prefix_uav_cost[1]], dtype=np.float32)

    np.testing.assert_allclose(
        world["uav_nodes"][0, :, schema.UAV_LAST_TOTAL_COST_LOG_RATIO].numpy(),
        np.log(last_uav_cost / uav_total_cost_ref),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["uav_nodes"][0, :, schema.UAV_PREFIX_TOTAL_COST_LOG_RATIO].numpy(),
        np.log(prefix_uav_cost / uav_total_cost_ref),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["gu_nodes"][0, :, schema.GU_LAST_TOTAL_COST_LOG_RATIO].numpy(),
        np.log(last_gu_cost / gu_total_cost_ref),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["gu_nodes"][0, :, schema.GU_PREFIX_TOTAL_COST_LOG_RATIO].numpy(),
        np.log(prefix_gu_cost / gu_total_cost_ref),
        atol=1.0e-6,
    )

    gu_queue = kwargs["gu_queue_t"].cpu().numpy()[0]
    uav_queue = kwargs["uav_queue_t"].cpu().numpy()[0]
    sat_queue = kwargs["sat_queue_active_t"].cpu().numpy()[0]
    expected_last_workload = float(
        np.sum(last_gu_cost * gu_queue)
        + np.sum(last_uav_cost * uav_queue)
        + np.sum(sat_cost * sat_queue)
    )
    expected_prefix_workload = float(
        np.sum(prefix_gu_cost * gu_queue)
        + np.sum(prefix_uav_cost * uav_queue)
        + np.sum(sat_cost * sat_queue)
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_TOTAL_LAST_WEIGHTED_WORKLOAD_STEPS].item(),
        expected_last_workload,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS].item(),
        expected_prefix_workload,
        atol=1.0e-5,
    )


def test_tensor_native_world_builder_tracks_non_token_satellite_residuals_when_full_state_is_available() -> None:
    device = torch.device("cpu")
    cfg, kwargs = _world_inputs(device)
    kwargs["stage_ids_t"] = torch.tensor([schema.CRITIC_STAGE_SAT], dtype=torch.long, device=device)
    kwargs["active_sat_ids_t"] = torch.tensor([[0, -1]], dtype=torch.long, device=device)
    kwargs["sat_active_mask_t"] = torch.tensor([[True, False]], dtype=torch.bool, device=device)
    kwargs["sat_queue_active_t"] = torch.tensor([[300.0, 0.0]], dtype=torch.float32, device=device)
    kwargs["sat_ema_active_t"] = torch.tensor([[100.0, 1.0]], dtype=torch.float32, device=device)
    kwargs["sat_drop_active_t"] = torch.tensor([[12.0, 0.0]], dtype=torch.float32, device=device)
    kwargs["last_sat_processed_active_t"] = torch.tensor([[18.0, 0.0]], dtype=torch.float32, device=device)
    kwargs["sat_queue_full_t"] = torch.tensor([[300.0, 900.0]], dtype=torch.float32, device=device)
    kwargs["sat_ema_full_t"] = torch.tensor([[100.0, 1000.0]], dtype=torch.float32, device=device)
    kwargs["sat_drop_full_t"] = torch.tensor([[12.0, 13.0]], dtype=torch.float32, device=device)
    kwargs["last_sat_processed_full_t"] = torch.tensor([[18.0, 19.0]], dtype=torch.float32, device=device)

    world = _build_world_from_packed_specs_tensor_impl(**kwargs)
    sat_flow_ref = 600.0 / 2.0
    assert world["sat_ids"][0, 0].item() == 0
    assert world["sat_ids"][0, 1].item() == -1
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_TOTAL_SAT_QUEUE_STEPS].item(),
        (300.0 + 900.0) / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_COUNT_FRAC].item(),
        1.0 / float(cfg.num_sat),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS].item(),
        900.0 / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_DROP_STEPS].item(),
        13.0 / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS].item(),
        19.0 / sat_flow_ref,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS].item(),
        (1.0 / 1000.0) * 900.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world["global_scalars"][0, schema.GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS].item(),
        (1.0 / 1000.0) * 13.0,
        atol=1.0e-6,
    )
