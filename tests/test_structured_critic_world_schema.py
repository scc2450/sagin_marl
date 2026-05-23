from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl import structured_critic_schema as schema


def _cfg() -> SaginConfig:
    cfg = SaginConfig(
        seed=20260429,
        num_uav=2,
        num_gu=5,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        visible_sats_max=4,
        nbrs_obs_max=1,
        sat_num_select=1,
    )
    cfg.fading_enabled = False
    cfg.doppler_enabled = False
    cfg.doppler_observed = False
    cfg.doppler_atten_enabled = False
    return cfg


def _first_visible_selection(driver: StructuredControlDriver) -> list[list[int]]:
    assert driver._stage_visible is not None
    out: list[list[int]] = []
    for visible in driver._stage_visible:
        out.append([int(visible[0])] if visible else [])
    return out


def test_python_critic_world_schema_shapes_and_stage_flags() -> None:
    cfg = _cfg()
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)

    accel_world = driver.begin_step()
    sat_world = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    bw_world = driver._build_world_state(
        stage_id=driver.STAGE_BW,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=_first_visible_selection(driver),
    )

    token_max = schema.critic_sat_token_max_from_cfg(cfg)
    for world in (accel_world, sat_world, bw_world):
        assert world.uav_nodes.shape == (1, cfg.num_uav, schema.CRITIC_UAV_NODE_DIM)
        assert world.gu_nodes.shape == (1, cfg.num_gu, schema.CRITIC_GU_NODE_DIM)
        assert world.sat_nodes.shape == (1, token_max, schema.CRITIC_SAT_NODE_DIM)
        assert world.sat_ids.shape == world.sat_mask.shape == (1, token_max)
        assert world.uav_gu_edges.shape == (1, cfg.num_uav, cfg.num_gu, schema.CRITIC_UAV_GU_EDGE_DIM)
        assert world.uav_sat_edges.shape == (1, cfg.num_uav, token_max, schema.CRITIC_UAV_SAT_EDGE_DIM)
        assert world.uav_uav_edges.shape == (1, cfg.num_uav, cfg.num_uav, schema.CRITIC_UAV_UAV_EDGE_DIM)
        assert world.global_scalars.shape == (1, schema.CRITIC_GLOBAL_SCALAR_DIM)

    assert np.all(accel_world.uav_gu_edges[..., schema.UG_PREFIX_BW_VALID_KNOWN] == 0.0)
    assert np.all(sat_world.uav_gu_edges[..., schema.UG_PREFIX_BW_VALID_KNOWN] == 1.0)
    assert np.all(bw_world.uav_gu_edges[..., schema.UG_PREFIX_BW_VALID_KNOWN] == 1.0)

    assert np.all(accel_world.uav_sat_edges[..., schema.US_PREFIX_SELECTED_KNOWN] == 0.0)
    assert np.all(sat_world.uav_sat_edges[..., schema.US_PREFIX_SELECTED_KNOWN] == 0.0)
    assert np.all(bw_world.uav_sat_edges[..., schema.US_PREFIX_SELECTED_KNOWN][bw_world.uav_sat_mask] == 1.0)

    assert np.all(accel_world.gu_nodes[..., schema.GU_PREFIX_COST_KNOWN] == 0.0)
    assert np.all(sat_world.gu_nodes[..., schema.GU_PREFIX_COST_KNOWN] == 0.0)
    assert np.all(bw_world.gu_nodes[..., schema.GU_PREFIX_COST_KNOWN] == 1.0)
    assert accel_world.global_scalars[0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN] == 0.0
    assert sat_world.global_scalars[0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN] == 0.0
    assert bw_world.global_scalars[0, schema.GLOBAL_PREFIX_WORKLOAD_KNOWN] == 1.0


def test_python_critic_sat_ids_and_backhaul_prefix_contract() -> None:
    cfg = _cfg()
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)
    driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))

    visible_union: list[int] = []
    seen: set[int] = set()
    for visible in driver._stage_visible:
        for sat_idx in visible:
            sat_idx_i = int(sat_idx)
            if sat_idx_i not in seen:
                seen.add(sat_idx_i)
                visible_union.append(sat_idx_i)
    world_a = driver._build_world_state(
        stage_id=driver.STAGE_BW,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=_first_visible_selection(driver),
    )
    world_b = driver._build_world_state(
        stage_id=driver.STAGE_BW,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=[[int(visible[-1])] if visible else [] for visible in driver._stage_visible],
    )
    np.testing.assert_array_equal(world_a.sat_ids[0, world_a.sat_mask[0]], np.asarray(visible_union, dtype=np.int64))
    assert np.all(world_a.sat_ids[0, ~world_a.sat_mask[0]] == -1)
    np.testing.assert_allclose(
        world_a.uav_sat_edges[..., schema.US_BACKHAUL_SE_REF],
        world_b.uav_sat_edges[..., schema.US_BACKHAUL_SE_REF],
        atol=1.0e-6,
    )

    sat_world = driver._build_world_state(
        stage_id=driver.STAGE_SAT,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=None,
    )
    assert np.all(sat_world.uav_sat_edges[..., schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS] == 0.0)
    selected = world_a.uav_sat_edges[..., schema.US_PREFIX_SELECTED_FLAG] > 0.5
    assert np.all(world_a.uav_sat_edges[..., schema.US_PREFIX_BACKHAUL_CAPACITY_STEPS][~selected] == 0.0)


def test_python_critic_world_uses_persistent_last_fields() -> None:
    cfg = _cfg()
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)
    driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))

    env.last_bw_fraction_by_uav_gu = np.linspace(0.0, 1.0, cfg.num_uav * cfg.num_gu, dtype=np.float32).reshape(cfg.num_uav, cfg.num_gu)
    env.last_access_interference_by_uav = np.asarray([2.0, 5.0], dtype=np.float32)
    env.last_gu_to_uav_inflow_by_uav = np.asarray([100.0, 300.0], dtype=np.float32)
    env.last_uav_to_sat_outflow_matrix = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
    env.last_selected_mask_by_uav_sat = np.zeros((cfg.num_uav, cfg.num_sat), dtype=np.float32)
    env.last_association = np.asarray([0, 1, 1, 0, 1], dtype=np.int32)

    world = driver._build_world_state(
        stage_id=driver.STAGE_BW,
        assoc=driver._stage_assoc,
        candidates=driver._stage_candidates,
        sat_pos=driver._stage_sat_pos,
        sat_vel=driver._stage_sat_vel,
        visible=driver._stage_visible,
        sat_selection=_first_visible_selection(driver),
    )
    np.testing.assert_allclose(
        world.uav_gu_edges[0, :, :, schema.UG_LAST_BW_FRACTION],
        env.last_bw_fraction_by_uav_gu,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        world.uav_gu_edges[0, :, :, schema.UG_LAST_SERVED_FLAG],
        env.last_bw_fraction_by_uav_gu,
        atol=1.0e-6,
    )
    arrival_ref = env._arrival_ref()
    uav_flow_ref = arrival_ref / float(cfg.num_uav)
    np.testing.assert_allclose(
        world.uav_nodes[0, :, schema.UAV_LAST_INFLOW_STEPS],
        env.last_gu_to_uav_inflow_by_uav / uav_flow_ref,
        atol=1.0e-6,
    )
    assert np.all(world.uav_nodes[0, :, schema.UAV_LAST_ACCESS_INTERFERENCE_LOG1P] > 0.0)
