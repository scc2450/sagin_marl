from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.env.structured_batch_env_core import native_module_shape_spec_from_config
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl import structured_accel_actor_schema as schema
from sagin_marl.rl.structured_actor import AccelPolicy
from sagin_marl.rl.structured_stage_builders import build_batched_local_accel_states_from_spec


def _schema_cfg(**overrides) -> SaginConfig:
    base = dict(
        seed=123,
        num_uav=3,
        num_gu=5,
        num_sat=7,
        users_obs_max=5,
        sats_obs_max=4,
        visible_sats_max=4,
        per_uav_visible_sat_token_max=4,
        nbrs_obs_max=2,
        sat_num_select=2,
        candidate_mode="nearest",
        candidate_k=5,
        access_fading_mode="large_scale",
        fading_enabled=False,
    )
    base.update(overrides)
    return SaginConfig(**base)


def _stage_spec_for_env(env: SaginParallelEnv) -> dict:
    env._invalidate_step_caches()
    driver = StructuredControlDriver(env)
    driver.begin_step()
    assert driver._accel_stage_spec_cache is not None
    return driver._accel_stage_spec_cache


def test_accel_schema_shapes_owner_and_public_fields_are_fixed() -> None:
    cfg = _schema_cfg()
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    env.uav_pos = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], dtype=np.float32)
    env.uav_vel = np.array([[2.0, 0.0], [0.0, -3.0], [-4.0, 1.0]], dtype=np.float32)
    env.gu_pos = np.array([[10.0, 0.0], [90.0, 0.0], [150.0, 0.0], [190.0, 0.0], [100.0, 40.0]], dtype=np.float32)
    env.gu_queue = np.array([4.0, 8.0, 16.0, 32.0, 64.0], dtype=np.float32)
    env.last_association = np.array([0, 1, 2, 1, -1], dtype=np.int32)
    env.last_bw_fraction_by_uav_gu = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.75, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env.gu_drop = np.array([0.0, 1.0, 2.0, 0.0, 3.0], dtype=np.float32)

    spec = _stage_spec_for_env(env)
    state = build_batched_local_accel_states_from_spec(spec)
    expected_owner = np.argmin(
        np.sum((env.gu_pos[:, None, :] - env.uav_pos[None, :, :]) ** 2, axis=-1),
        axis=1,
    ).astype(np.int32)

    assert np.array_equal(spec["assoc"], expected_owner)
    assert state.ego_features.shape == (cfg.num_uav, schema.ACCEL_EGO_DIM)
    assert state.ego_cell.shape == (cfg.num_uav, schema.ACCEL_CELL_DIM)
    assert state.gu_tokens.shape == (cfg.num_uav, cfg.num_gu, schema.ACCEL_GU_TOKEN_DIM)
    assert state.peer_tokens.shape == (cfg.num_uav, cfg.num_uav - 1, schema.ACCEL_PEER_TOKEN_DIM)
    assert state.sat_tokens.shape == (cfg.num_uav, cfg.per_uav_visible_sat_token_max, schema.ACCEL_SAT_TOKEN_DIM)
    assert torch.all(state.gu_mask)

    for ego in range(cfg.num_uav):
        owner_flags = state.gu_tokens[ego, :, schema.GU_PRE_OWNER_IS_EGO].numpy()
        np.testing.assert_array_equal(owner_flags, (expected_owner == ego).astype(np.float32))
        np.testing.assert_allclose(
            state.ego_cell[ego, schema.CELL_GU_COUNT_FRAC].item(),
            np.sum(expected_owner == ego) / float(cfg.num_gu),
            atol=1.0e-6,
        )

    peer_row_for_uav1 = state.peer_tokens[1]
    np.testing.assert_allclose(
        peer_row_for_uav1[0, schema.PEER_REL_X : schema.PEER_REL_Y + 1].numpy(),
        (env.uav_pos[1] - env.uav_pos[0]) / float(cfg.map_size),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        peer_row_for_uav1[1, schema.PEER_REL_X : schema.PEER_REL_Y + 1].numpy(),
        (env.uav_pos[1] - env.uav_pos[2]) / float(cfg.map_size),
        atol=1.0e-6,
    )

    access_gain = np.asarray(spec["access_gain_matrix"], dtype=np.float32)
    access_noise_ref = (
        float(cfg.noise_density)
        * float(cfg.b_acc)
        * float(10.0 ** (float(getattr(cfg, "access_noise_figure_db", 0.0)) / 10.0))
    )
    bw_sum = np.where(
        (env.last_association >= 0) & (env.last_association < cfg.num_uav),
        np.maximum(np.sum(env.last_bw_fraction_by_uav_gu, axis=0), 0.0),
        0.0,
    ).astype(np.float32)
    ego = 0
    expected_nonself = np.log1p(
        np.maximum(float(cfg.gu_tx_power) * access_gain[:, ego] * bw_sum * (env.last_association != ego) / access_noise_ref, 0.0)
    )
    np.testing.assert_allclose(
        state.gu_tokens[ego, :, schema.GU_LAST_BW_SUM].numpy(),
        bw_sum,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        state.gu_tokens[ego, :, schema.GU_LAST_NONSELF_INTERFERENCE_LOG1P].numpy(),
        expected_nonself,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_accel_schema_allows_single_uav_without_peer_tokens() -> None:
    cfg = _schema_cfg(num_uav=1, num_gu=3, users_obs_max=3, nbrs_obs_max=0)
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed + 1)
    spec = _stage_spec_for_env(env)
    state = build_batched_local_accel_states_from_spec(spec)

    assert state.peer_tokens.shape == (1, 0, schema.ACCEL_PEER_TOKEN_DIM)
    assert state.peer_mask.shape == (1, 0)
    assert torch.all(state.gu_tokens[0, :, schema.GU_PRE_OWNER_IS_EGO] == 1.0)
    assert torch.all(state.gu_tokens[0, :, schema.GU_PARTITION_BOUNDARY_WEIGHT] == 0.0)
    assert torch.all(state.gu_tokens[0, :, schema.GU_OWNER_STABILITY_MARGIN] == 1.0)


def test_accel_policy_consumes_schema_with_global_log_std() -> None:
    cfg = _schema_cfg()
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed + 2)
    state = build_batched_local_accel_states_from_spec(_stage_spec_for_env(env))
    policy = AccelPolicy(
        ego_dim=schema.ACCEL_EGO_DIM,
        cell_dim=schema.ACCEL_CELL_DIM,
        gu_token_dim=schema.ACCEL_GU_TOKEN_DIM,
        peer_token_dim=schema.ACCEL_PEER_TOKEN_DIM,
        sat_token_dim=schema.ACCEL_SAT_TOKEN_DIM,
        hidden_dim=32,
        embed_dim=16,
    )

    assert tuple(policy.log_std.shape) == (2,)
    assert policy.gu_query_count == schema.ACCEL_GU_QUERY_COUNT
    assert policy.peer_query_count == schema.ACCEL_PEER_QUERY_COUNT
    assert policy.sat_query_count == schema.ACCEL_SAT_QUERY_COUNT
    output = policy(state, deterministic=True)
    assert output.action.shape == (cfg.num_uav, 2)
    assert torch.isfinite(output.action).all()


def test_accel_cuda_schema_enums_match_python_schema() -> None:
    kernels = Path("sagin_marl/env/native_cuda/kernels.cu").read_text(encoding="utf-8")

    def enum_value(name: str) -> int:
        match = re.search(rf"\b{name}\s*=\s*(\d+)", kernels)
        assert match is not None, name
        return int(match.group(1))

    assert enum_value("kAccelEgoDim") == schema.ACCEL_EGO_DIM
    assert enum_value("kAccelCellDim") == schema.ACCEL_CELL_DIM
    assert enum_value("kAccelGuTokenDim") == schema.ACCEL_GU_TOKEN_DIM
    assert enum_value("kAccelPeerTokenDim") == schema.ACCEL_PEER_TOKEN_DIM
    assert enum_value("kAccelSatTokenDim") == schema.ACCEL_SAT_TOKEN_DIM
    for py_name in dir(schema):
        if not py_name.startswith(("EGO_", "CELL_", "GU_", "PEER_", "SAT_")):
            continue
        value = getattr(schema, py_name)
        if not isinstance(value, int):
            continue
        parts = ["Log1p" if part == "log1p" else part.title() for part in py_name.lower().split("_")]
        cuda_name = "kAccel" + "".join(parts)
        assert enum_value(cuda_name) == value


def test_accel_sat_width_is_finalized_visible_sat_width() -> None:
    cfg = _schema_cfg(per_uav_visible_sat_token_max=3, visible_sats_max=4, sats_obs_max=4)
    env = SaginParallelEnv(cfg)
    assert int(env.cfg.per_uav_visible_sat_token_max) == 4
    assert native_module_shape_spec_from_config(env.cfg).accel_sat_width == 4

    bad_cfg = _schema_cfg(per_uav_visible_sat_token_max=0, visible_sats_max=4, sats_obs_max=4)
    with pytest.raises(ValueError, match="per_uav_visible_sat_token_max"):
        native_module_shape_spec_from_config(bad_cfg)


def test_direct_env_constructor_finalizes_accel_actor_config() -> None:
    cfg = _schema_cfg(per_uav_visible_sat_token_max=None)
    env = SaginParallelEnv(cfg)
    assert int(env.cfg.per_uav_visible_sat_token_max) == min(int(cfg.num_sat), int(cfg.visible_sats_max))

    compatible_cfg = _schema_cfg(num_gu=6, users_obs_max=5, candidate_k=5)
    compatible_env = SaginParallelEnv(compatible_cfg)
    assert int(compatible_env.cfg.users_obs_max) == 6
    assert int(compatible_env.cfg.candidate_k) == 6
