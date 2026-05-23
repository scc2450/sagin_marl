from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl.structured_actor import AccelPolicy, BwPolicy, SatSubsetPolicy, StructuredActor
from sagin_marl.rl.structured_critic import StructuredCritic
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import StructuredMAPPO


def close_if_possible(value) -> None:
    close_fn = getattr(value, "close", None)
    if callable(close_fn):
        close_fn()


def make_structured_test_cfg(**overrides) -> SaginConfig:
    base = dict(
        seed=7,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        T_steps=8,
        danger_imitation_enabled=True,
        danger_imitation_coef=0.1,
    )
    base.update(overrides)
    return SaginConfig(**base)


def make_structured_test_learner(cfg: SaginConfig) -> tuple[StructuredMAPPO, object, object]:
    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16)
    actor = bundle.actor
    critic = bundle.critic
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=2,
        num_mini_batch=1,
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
    )
    return learner, actor, critic


def default_sat_subset_action(driver: StructuredControlDriver, sat_states):
    subset_indices = []
    for state in sat_states:
        valid = state.subset_mask[0].nonzero(as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    return driver.decode_sat_subset_actions(sat_states, subset_indices)


def materialize_stage_obs_baseline(driver: StructuredControlDriver):
    env = driver.env
    env._cached_assoc = driver._stage_assoc.copy()
    env._cached_candidates = [list(c) for c in driver._stage_candidates]
    if driver._stage_bw_valid_mask is not None:
        env._cached_bw_valid_mask = driver._stage_bw_valid_mask.copy()
    _, env._cached_eta = env._compute_access_rates(
        driver._stage_assoc,
        driver._stage_candidates,
        driver._zero_bw_action_matrix(),
        record_exec=False,
    )
    saved_last_sat_selection = [list(sel) for sel in getattr(env, "last_sat_selection", [])]
    saved_last_sat_connection_counts = np.asarray(env.last_sat_connection_counts, dtype=np.float32).copy()
    try:
        if driver._stage_sat_selection is not None:
            stage_sat_connection_counts = driver._compute_stage_sat_selection_counts(
                driver._stage_sat_selection
            ).astype(np.float32, copy=False)
            env.last_sat_connection_counts = stage_sat_connection_counts
            env.last_sat_selection = [list(sel) for sel in driver._stage_sat_selection]
        env._cache_sat_obs(driver._stage_sat_pos, driver._stage_sat_vel, driver._stage_visible)
        obs_many = [env._get_obs(i) for i in range(len(env.agents))]
        return (
            obs_many,
            np.asarray(env._cached_sat_obs, dtype=np.float32).copy(),
            np.asarray(env._cached_sat_mask, dtype=np.float32).copy(),
            np.asarray(env._cached_sat_valid_mask, dtype=np.float32).copy(),
        )
    finally:
        env.last_sat_selection = saved_last_sat_selection
        env.last_sat_connection_counts = saved_last_sat_connection_counts


def assert_world_state_close(left, right) -> None:
    for field_name in (
        "uav_nodes",
        "gu_nodes",
        "sat_nodes",
        "uav_gu_edges",
        "uav_sat_edges",
        "uav_uav_edges",
        "gu_mask",
        "sat_mask",
        "uav_gu_mask",
        "uav_sat_mask",
        "uav_uav_mask",
        "stage_id",
    ):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        left_tensor = left_value if torch.is_tensor(left_value) else torch.as_tensor(left_value)
        right_tensor = right_value if torch.is_tensor(right_value) else torch.as_tensor(right_value)
        torch.testing.assert_close(left_tensor, right_tensor)


def assert_local_state_group_close(left_group, right_group) -> None:
    assert len(left_group) == len(right_group)
    for left_state, right_state in zip(left_group, right_group):
        for field in fields(type(left_state)):
            left_value = getattr(left_state, field.name)
            right_value = getattr(right_state, field.name)
            left_tensor = left_value if torch.is_tensor(left_value) else torch.as_tensor(left_value)
            right_tensor = right_value if torch.is_tensor(right_value) else torch.as_tensor(right_value)
            if torch.is_tensor(left_tensor):
                left_tensor = left_tensor.detach().to(device=torch.device("cpu"))
            if torch.is_tensor(right_tensor):
                right_tensor = right_tensor.detach().to(device=torch.device("cpu"))
            torch.testing.assert_close(left_tensor, right_tensor)


def assert_runtime_state_matches_env(state, env) -> None:
    np.testing.assert_allclose(np.asarray(state["uav_pos"], dtype=np.float32), np.asarray(env.uav_pos, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["uav_vel"], dtype=np.float32), np.asarray(env.uav_vel, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["uav_energy"], dtype=np.float32), np.asarray(env.uav_energy, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["gu_queue"], dtype=np.float32), np.asarray(env.gu_queue, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["uav_queue"], dtype=np.float32), np.asarray(env.uav_queue, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["sat_queue"], dtype=np.float32), np.asarray(env.sat_queue, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(state["last_association"], dtype=np.int32),
        np.asarray(env.last_association, dtype=np.int32),
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(state["last_sat_connection_counts"], dtype=np.float32),
        np.asarray(env.last_sat_connection_counts, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state["last_gu_service_gap"], dtype=np.float32),
        np.asarray(env.last_gu_service_gap, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state["last_gu_deadline_age"], dtype=np.float32),
        np.asarray(env.last_gu_deadline_age, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state["_doppler_residual_state_hz"], dtype=np.float32),
        np.asarray(env._doppler_residual_state_hz, dtype=np.float32),
        atol=1e-6,
    )
    assert int(state["t"]) == int(env.t)
    assert int(state["global_step"]) == int(env.global_step)


def build_structured_modules_from_probe_driver(
    cfg: SaginConfig,
    driver: StructuredControlDriver,
) -> tuple[StructuredActor, StructuredCritic]:
    z0 = driver.begin_step()
    accel_states = driver.build_local_accel_states(z0)
    z1 = driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    sat_states = driver.build_local_sat_states(z1)
    sat_action = default_sat_subset_action(driver, sat_states)
    z2 = driver.run_sat_stage(sat_action)
    bw_states = driver.build_bw_valid_context(z2)

    accel_policy = AccelPolicy(
        ego_dim=accel_states[0].ego_features.shape[-1],
        cell_dim=accel_states[0].ego_cell.shape[-1],
        gu_token_dim=accel_states[0].gu_tokens.shape[-1],
        peer_token_dim=accel_states[0].peer_tokens.shape[-1],
        sat_token_dim=accel_states[0].sat_tokens.shape[-1],
        hidden_dim=32,
        embed_dim=16,
    )
    sat_policy = SatSubsetPolicy(
        hidden_dim=32,
        embed_dim=16,
        sat_action_select_k=int(getattr(cfg, "sat_action_select_k", cfg.sat_num_select)),
        per_uav_visible_sat_token_max=sat_states[0].sat_tokens.shape[-2],
        sat_competition_layers=1,
        sat_attention_heads=4,
    )
    bw_policy = BwPolicy(
        ego_dim=bw_states[0].ego_features.shape[-1],
        sat_token_dim=bw_states[0].selected_sat_tokens.shape[-1],
        gu_token_dim=bw_states[0].gu_tokens.shape[-1],
        hidden_dim=32,
        embed_dim=16,
        down_query_count=2,
        num_competition_layers=1,
        num_heads=4,
    )
    actor = StructuredActor(accel_policy=accel_policy, sat_subset_policy=sat_policy, bw_policy=bw_policy)
    critic = StructuredCritic(
        uav_dim=z0.uav_nodes.shape[-1],
        gu_dim=z0.gu_nodes.shape[-1],
        sat_dim=z0.sat_nodes.shape[-1],
        uav_gu_edge_dim=z0.uav_gu_edges.shape[-1],
        uav_sat_edge_dim=z0.uav_sat_edges.shape[-1],
        uav_uav_edge_dim=z0.uav_uav_edges.shape[-1],
        hidden_dim=32,
        embed_dim=16,
    )
    return actor, critic


