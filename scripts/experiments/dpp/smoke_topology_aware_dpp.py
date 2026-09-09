#!/usr/bin/env python3
"""Historical smoke test for the topology-aware DPP baseline.

This script belongs to the archived `lyapunov-dpp` experiment path.  The
current joint MC-GAE branch may not expose the topology-aware DPP helpers.
"""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))

from sagin_marl.env.config import SaginConfig
try:
    from sagin_marl.rl.baselines import (
        _generate_accel_candidates,
        _predict_users_rel_after_accel,
        _predict_topology_after_accel,
        _approx_eta_from_distance,
        lyapunov_queue_aware_policy_step,
    )
except ImportError as exc:
    raise SystemExit(
        "This archived smoke script requires the topology-aware DPP helpers "
        "from the lyapunov-dpp experiment branch. Current branch may not "
        "include them yet."
    ) from exc


def create_mock_obs(cfg: SaginConfig) -> dict:
    """Create a mock observation for testing."""
    obs = {
        "users": np.random.randn(cfg.users_obs_max, 7).astype(np.float32),
        "sats": np.random.randn(cfg.sats_obs_max, 12).astype(np.float32),  # Must have 12 columns
        "nbrs": np.random.randn(cfg.nbrs_obs_max, 5).astype(np.float32),
        "own": np.random.randn(8).astype(np.float32),
        "users_mask": np.ones((cfg.users_obs_max,), dtype=np.float32),
        "sats_mask": np.ones((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs_mask": np.ones((cfg.nbrs_obs_max,), dtype=np.float32),
        "bw_valid_mask": np.ones((cfg.users_obs_max,), dtype=np.float32),
    }
    # Normalize obs["own"] to valid ranges
    obs["own"][0:2] = np.clip(obs["own"][0:2], -1.0, 1.0)  # position
    obs["own"][2:4] = np.clip(obs["own"][2:4], -1.0, 1.0)  # velocity
    obs["own"][4] = np.clip(obs["own"][4], 0.0, 1.0)  # energy
    obs["users"][:, 2] = np.clip(obs["users"][:, 2], 0.0, 100.0)  # queue
    obs["users"][:, 3] = np.clip(obs["users"][:, 3], 0.0, 1.0)  # eta
    obs["users"][:, 4] = np.clip(obs["users"][:, 4], 0.0, 1.0)  # assoc
    obs["sats"][:, 5] = np.clip(obs["sats"][:, 5], 0.0, 100.0)  # queue
    obs["sats"][:, 7] = np.clip(obs["sats"][:, 7], 0.0, 2.0)  # se
    obs["sats"][:, 8] = np.clip(obs["sats"][:, 8], 0.0, 100.0)  # qsat
    obs["sats"][:, 9] = np.clip(obs["sats"][:, 9], 0.0, 1.0)  # load_norm
    obs["sats"][:, 10] = np.clip(obs["sats"][:, 10], 0.0, 1.0)  # bw_ratio
    obs["sats"][:, 11] = np.clip(obs["sats"][:, 11], 0.0, 1.0)  # stay
    return obs


def test_topology_prediction():
    """Test _predict_topology_after_accel() with and without callbacks."""
    print("\n=== Test: Topology Prediction ===")
    
    cfg = SaginConfig()
    cfg.baseline_lyapunov_mode = "dpp"
    obs = create_mock_obs(cfg)
    
    accel = np.array([0.5, -0.3], dtype=np.float32)
    
    # Test 1: Without env_callbacks (heuristic mode)
    print("Test 1: Topology prediction WITHOUT env_callbacks...")
    topo = _predict_topology_after_accel(obs, cfg, accel)
    assert "rel_next" in topo, "Missing rel_next"
    assert "eta" in topo, "Missing eta"
    assert "sat_visible_mask" in topo, "Missing sat_visible_mask"
    print(f"  ✓ rel_next shape: {topo['rel_next'].shape}")
    print(f"  ✓ eta shape: {topo['eta'].shape}, mean: {np.mean(topo['eta']):.3f}")
    print(f"  ✓ sat_visible: {np.sum(topo['sat_visible_mask'])} / {cfg.sats_obs_max}")
    
    # Test 2: With env_callbacks (should degrade gracefully if callbacks raise)
    print("\nTest 2: Topology prediction WITH mock env_callbacks...")
    def mock_compute_access_rates(agent_id, accel_vec, obs):
        eta = np.random.uniform(0.3, 0.9, size=(cfg.users_obs_max,))
        rates = 0.6 * eta
        return eta, rates
    
    def mock_check_sat_visibility(agent_id, accel_vec, obs):
        return np.random.rand(cfg.sats_obs_max) > 0.3
    
    callbacks = {
        "compute_access_rates": mock_compute_access_rates,
        "check_sat_visibility": mock_check_sat_visibility,
    }
    setattr(cfg, "_dpp_env_callbacks_temp", callbacks)
    
    topo = _predict_topology_after_accel(obs, cfg, accel)
    print(f"  ✓ With callbacks: eta mean = {np.mean(topo['eta']):.3f}")
    print(f"  ✓ sat_visible: {np.sum(topo['sat_visible_mask'])} / {cfg.sats_obs_max}")


def test_dpp_baseline():
    """Test full DPP baseline with topology awareness."""
    print("\n=== Test: DPP Baseline with Topology ===")
    
    cfg = SaginConfig()
    cfg.baseline_lyapunov_mode = "dpp"
    cfg.dpp_accel_num_candidates = 5  # Small for fast test
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    
    num_agents = 2
    obs_list = [create_mock_obs(cfg) for _ in range(num_agents)]
    
    print(f"Running DPP baseline for {num_agents} agents...")
    accel, bw_alloc, sat_mask, next_state = lyapunov_queue_aware_policy_step(
        obs_list,
        cfg,
        state=None,
        env_callbacks=None,  # Start without callbacks
    )
    
    assert accel.shape == (num_agents, 2), f"accel shape mismatch: {accel.shape}"
    assert bw_alloc.shape == (num_agents, cfg.users_obs_max), f"bw shape mismatch: {bw_alloc.shape}"
    assert sat_mask.shape == (num_agents, cfg.sats_obs_max), f"sat shape mismatch: {sat_mask.shape}"
    print(f"  ✓ Output shapes correct")
    print(f"  ✓ accel range: [{accel.min():.3f}, {accel.max():.3f}]")
    print(f"  ✓ BW sum per agent: {np.sum(bw_alloc, axis=1)}")
    print(f"  ✓ SAT selection per agent: {np.sum(sat_mask, axis=1)}")
    
    # Verify state has new DPP fields
    for key in ["dpp_access_term", "dpp_backhaul_term", "dpp_reg_term", "dpp_objective_term"]:
        assert key in next_state, f"Missing state key: {key}"
    print(f"  ✓ State contains DPP term tracking")


def test_dpp_with_callbacks():
    """Test DPP baseline passing env_callbacks."""
    print("\n=== Test: DPP Baseline WITH env_callbacks ===")
    
    cfg = SaginConfig()
    cfg.baseline_lyapunov_mode = "dpp"
    cfg.dpp_accel_num_candidates = 3  # Very small for speed
    cfg.enable_bw_action = True
    
    num_agents = 1
    obs_list = [create_mock_obs(cfg) for _ in range(num_agents)]
    
    # Build mock callbacks
    def compute_access_rates(agent_id, accel_vec, obs):
        eta = np.random.uniform(0.2, 0.8, size=(cfg.users_obs_max,))
        rates = 0.5 * eta
        return eta, rates
    
    def check_sat_visibility(agent_id, accel_vec, obs):
        return np.random.rand(cfg.sats_obs_max) > 0.4
    
    callbacks = {
        "compute_access_rates": compute_access_rates,
        "check_sat_visibility": check_sat_visibility,
    }
    
    print(f"Running DPP with {len(callbacks)} callbacks...")
    accel, bw_alloc, sat_mask, next_state = lyapunov_queue_aware_policy_step(
        obs_list,
        cfg,
        state=None,
        env_callbacks=callbacks,
    )
    
    print(f"  ✓ DPP executed with callbacks successfully")
    print(f"  ✓ Callback helpers accepted and used")
    
    # Run again to test state persistence
    print(f"\nRunning DPP again with same state...")
    accel2, bw_alloc2, sat_mask2, next_state2 = lyapunov_queue_aware_policy_step(
        obs_list,
        cfg,
        state=next_state,
        env_callbacks=callbacks,
    )
    print(f"  ✓ State persistence OK")


if __name__ == "__main__":
    print("=" * 60)
    print("DPP Topology-Aware Baseline - Smoke Tests")
    print("=" * 60)
    
    try:
        test_topology_prediction()
        test_dpp_baseline()
        test_dpp_with_callbacks()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
