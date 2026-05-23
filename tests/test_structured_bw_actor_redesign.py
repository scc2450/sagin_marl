from __future__ import annotations

import numpy as np
import pytest
import torch

from sagin_marl.env import native_cuda
from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv, compute_access_interference_beta_continuous
from sagin_marl.env.structured_driver import StructuredControlDriver
from sagin_marl.rl import structured_bw_actor_schema as bw_schema
from sagin_marl.rl.native_actor_cuda import build_native_actor_cuda_binding
from sagin_marl.rl.structured_actor import BwPolicy
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import _collate_dataclass
from sagin_marl.rl.structured_stage_builders import build_local_bw_states_from_snapshot
from sagin_marl.rl.structured_train import close_structured_env_group, make_structured_env_group
from sagin_marl.rl.structured_types import LocalBwState


def test_bw_actor_schema_constants_are_final() -> None:
    assert bw_schema.BW_EGO_DIM == len(bw_schema.BW_EGO_FIELDS) == 10
    assert bw_schema.BW_SAT_TOKEN_DIM == len(bw_schema.BW_SAT_TOKEN_FIELDS) == 9
    assert bw_schema.BW_GU_TOKEN_DIM == len(bw_schema.BW_GU_TOKEN_FIELDS) == 13
    assert bw_schema.BW_OBJECTIVE_NORM == "per_latent_dim"


def test_bw_policy_outputs_full_g_masked_simplex() -> None:
    torch.manual_seed(1)
    state = LocalBwState(
        ego_features=torch.randn(3, bw_schema.BW_EGO_DIM),
        selected_sat_tokens=torch.randn(3, 2, bw_schema.BW_SAT_TOKEN_DIM),
        selected_sat_mask=torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.bool),
        gu_tokens=torch.randn(3, 5, bw_schema.BW_GU_TOKEN_DIM),
        gu_mask=torch.ones(3, 5, dtype=torch.bool),
        bw_valid_mask=torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
    )
    policy = BwPolicy(hidden_dim=32, embed_dim=16, down_query_count=2, num_competition_layers=1, num_heads=4)

    det = policy(state, deterministic=True)
    assert det.action.shape == (3, 5)
    torch.testing.assert_close(det.action[0], torch.zeros(5))
    torch.testing.assert_close(det.action[1], state.bw_valid_mask[1].float())
    torch.testing.assert_close(det.action[2, ~state.bw_valid_mask[2]], torch.zeros(2))
    torch.testing.assert_close(det.action[2, state.bw_valid_mask[2]].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(det.logprob[:2], torch.zeros(2))
    torch.testing.assert_close(det.entropy[:2], torch.zeros(2))
    assert det.valid_count.tolist() == [0, 1, 3]
    assert det.latent_count.tolist() == [0, 0, 2]

    sample = policy(state, deterministic=False)
    torch.testing.assert_close(sample.action[0], torch.zeros(5))
    torch.testing.assert_close(sample.action[1], state.bw_valid_mask[1].float())
    torch.testing.assert_close(sample.action[2, ~state.bw_valid_mask[2]], torch.zeros(2))
    torch.testing.assert_close(sample.action[2, state.bw_valid_mask[2]].sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_access_interference_is_beta_continuous() -> None:
    assoc = np.asarray([0, 1], dtype=np.int32)
    gain = np.ones((2, 2), dtype=np.float32)
    none = compute_access_interference_beta_continuous(
        assoc,
        gain,
        np.asarray([0.0, 0.0], dtype=np.float32),
        gu_tx_power=2.0,
    )
    small = compute_access_interference_beta_continuous(
        assoc,
        gain,
        np.asarray([0.01, 1.0], dtype=np.float32),
        gu_tx_power=2.0,
    )
    full = compute_access_interference_beta_continuous(
        assoc,
        gain,
        np.asarray([1.0, 1.0], dtype=np.float32),
        gu_tx_power=2.0,
    )
    np.testing.assert_allclose(none, np.zeros(2, dtype=np.float32))
    assert 0.0 < float(small[1]) < float(full[1])
    np.testing.assert_allclose(small[0], full[0])


def test_single_env_bw_dict_action_is_full_g_and_not_renormalized() -> None:
    cfg = SaginConfig(
        seed=4,
        num_uav=1,
        num_gu=3,
        num_sat=2,
        users_obs_max=5,
        sats_obs_max=2,
        nbrs_obs_max=1,
        T_steps=2,
        enable_bw_action=True,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    assert env.action_space(env.agents[0])["bw_alloc"].shape == (cfg.num_gu,)

    assoc = np.zeros((cfg.num_gu,), dtype=np.int32)
    candidates = [list(range(cfg.num_gu))]
    valid_action = {
        env.agents[0]: {
            "accel": np.zeros((2,), dtype=np.float32),
            "bw_alloc": np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
            "sat_select_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        }
    }
    _rates, _eta = env._compute_access_rates(assoc, candidates, valid_action, record_exec=True)
    np.testing.assert_allclose(env.last_bw_fraction_by_uav_gu[0], valid_action[env.agents[0]]["bw_alloc"], rtol=1e-6, atol=1e-6)

    invalid_action = {
        env.agents[0]: {
            "accel": np.zeros((2,), dtype=np.float32),
            "bw_alloc": np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
            "sat_select_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        }
    }
    try:
        env._compute_access_rates(assoc, candidates, invalid_action, record_exec=True)
    except ValueError as exc:
        assert "valid sum" in str(exc)
    else:
        raise AssertionError("BW dict action with valid mass != 1 should not be renormalized silently.")


def test_driver_builds_full_g_bw_local_state() -> None:
    cfg = SaginConfig(
        seed=3,
        num_uav=2,
        num_gu=4,
        num_sat=8,
        users_obs_max=4,
        sats_obs_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        per_uav_visible_sat_token_max=4,
        T_steps=3,
    )
    env = SaginParallelEnv(cfg)
    env.reset(seed=cfg.seed)
    driver = StructuredControlDriver(env)
    driver.begin_step()
    driver.run_accel_stage(np.zeros((cfg.num_uav, 2), dtype=np.float32))
    sat_states = driver.build_local_sat_states()
    subset_indices = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if valid.numel() > 0 else -1)
    sat_actions = driver.decode_sat_subset_actions(sat_states, subset_indices)
    driver.run_sat_stage(sat_actions)

    local = _collate_dataclass(driver.build_local_bw_states(), torch.device("cpu"))
    assert local.ego_features.shape == (cfg.num_uav, bw_schema.BW_EGO_DIM)
    assert local.selected_sat_tokens.shape == (cfg.num_uav, sat_actions.shape[-1], bw_schema.BW_SAT_TOKEN_DIM)
    assert local.gu_tokens.shape == (cfg.num_uav, cfg.num_gu, bw_schema.BW_GU_TOKEN_DIM)
    assoc = np.asarray(driver._stage_assoc, dtype=np.int32)
    expected_valid = torch.as_tensor(np.stack([assoc == u for u in range(cfg.num_uav)]), dtype=torch.bool)
    assert torch.equal(local.bw_valid_mask, expected_valid)
    torch.testing.assert_close(local.gu_tokens[~local.bw_valid_mask], torch.zeros_like(local.gu_tokens[~local.bw_valid_mask]))

    snapshot_rows = build_local_bw_states_from_snapshot(driver.build_bw_stage_snapshot())
    snapshot_local = _collate_dataclass(snapshot_rows, torch.device("cpu"))
    torch.testing.assert_close(snapshot_local.ego_features, local.ego_features)
    torch.testing.assert_close(snapshot_local.selected_sat_tokens, local.selected_sat_tokens)
    torch.testing.assert_close(snapshot_local.gu_tokens, local.gu_tokens)
    torch.testing.assert_close(snapshot_local.bw_valid_mask, local.bw_valid_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native BW actor parity")
def test_native_bw_actor_deterministic_matches_torch_policy() -> None:
    cfg = SaginConfig(
        seed=777,
        num_uav=2,
        num_gu=4,
        num_sat=6,
        users_obs_max=4,
        sats_obs_max=4,
        visible_sats_max=4,
        per_uav_visible_sat_token_max=4,
        nbrs_obs_max=1,
        sat_num_select=2,
        candidate_mode="nearest",
        candidate_k=4,
        access_fading_mode="large_scale",
        fading_enabled=False,
        fixed_satellite_strategy=False,
        structured_env_backend="native",
        structured_env_tensor_backend="cuda",
        T_steps=4,
        accel_gu_query_count=4,
        accel_peer_query_count=2,
        accel_sat_query_count=2,
    )
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    try:
        env_group.reset_many([int(cfg.seed)])
        env_group.set_tensor_device(torch.device("cuda"))
        env_group.begin_native_main_kernel_rollout(capacity=2, num_envs=1)
        runtime = env_group.native_rollout_runtime
        abi = runtime.main.native_cuda_abi

        torch.manual_seed(4321)
        actor = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16, build_critic=False).actor.to("cuda")
        actor.eval()
        binding = build_native_actor_cuda_binding(actor, device="cuda")

        native_cuda.prepare_initial_accel_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )
        native_cuda.actor_accel_live(abi, binding.abi, active_idx=0, deterministic=True, rng_step=1)
        native_cuda.accel_to_sat_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )
        native_cuda.actor_sat_live(abi, binding.abi, deterministic=True, rng_step=2)
        native_cuda.sat_to_bw_live(
            abi,
            slot=0,
            active_idx=0,
            accel_source_mode=native_cuda.SOURCE_POLICY,
            sat_source_mode=native_cuda.SOURCE_POLICY,
            bw_source_mode=native_cuda.SOURCE_POLICY,
        )

        with torch.no_grad():
            expected = actor.bw_policy(runtime.main.live_bw_obs, deterministic=True)
        native_cuda.actor_bw_live(abi, binding.abi, deterministic=True, rng_step=3)

        torch.testing.assert_close(runtime.main.live_bw_action.reshape_as(expected.action), expected.action, atol=5.0e-4, rtol=5.0e-4)
        torch.testing.assert_close(runtime.main.live_bw_ref_action.reshape_as(expected.det_mean), expected.det_mean, atol=5.0e-4, rtol=5.0e-4)
        torch.testing.assert_close(runtime.main.live_bw_old_logprobs_per_agent.reshape_as(expected.logprob), expected.logprob, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_entropy_per_agent.reshape_as(expected.entropy), expected.entropy, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_logprob_raw_per_agent.reshape_as(expected.logprob_raw), expected.logprob_raw, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_entropy_raw_per_agent.reshape_as(expected.entropy_raw), expected.entropy_raw, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_valid_count.reshape_as(expected.valid_count), expected.valid_count)
        torch.testing.assert_close(runtime.main.live_bw_latent_count.reshape_as(expected.latent_count), expected.latent_count)

        native_cuda.actor_bw_live_fused(abi, binding.abi, deterministic=True, rng_step=3)
        torch.testing.assert_close(runtime.main.live_bw_action.reshape_as(expected.action), expected.action, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_ref_action.reshape_as(expected.det_mean), expected.det_mean, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_old_logprobs_per_agent.reshape_as(expected.logprob), expected.logprob, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_entropy_per_agent.reshape_as(expected.entropy), expected.entropy, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_logprob_raw_per_agent.reshape_as(expected.logprob_raw), expected.logprob_raw, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_entropy_raw_per_agent.reshape_as(expected.entropy_raw), expected.entropy_raw, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(runtime.main.live_bw_valid_count.reshape_as(expected.valid_count), expected.valid_count)
        torch.testing.assert_close(runtime.main.live_bw_latent_count.reshape_as(expected.latent_count), expected.latent_count)
    finally:
        close_structured_env_group(env_group)


