from pathlib import Path

import pytest

from scripts.experiments.more_gus.run_fixed_baselines import (
    METHODS, METRICS, aggregate, check_protocol, command_for, point_config,
)


def base():
    return dict(num_uav=3, num_gu=100, gu_init_num_clusters=22, T_steps=250,
        access_bw_decision_interval=1, sat_decision_interval=1,
        safety_shield_enabled=True, safety_shield_solver="NATIVE_CUDA",
        avoidance_enabled=False, fixed_satellite_strategy=False,
        resource_scale_enabled=False, b_acc=4e6, b_backhaul_per_sat=1e7,
        sat_cpu_freq=5e10, task_arrival_rate=4e5, reward_mode="unchanged", seed=45211)


def test_scan_axes_scale_current_training_nominals_only():
    cfg = base()
    assert point_config(cfg, "nominal", 1) == cfg
    assert point_config(cfg, "load", 2) == dict(cfg, task_arrival_rate=8e5)
    assert point_config(cfg, "resource", 0.5) == dict(cfg,
        b_acc=2e6, b_backhaul_per_sat=5e6, sat_cpu_freq=2.5e10)
    assert cfg == base()


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf")])
def test_invalid_scan_point_rejected(multiplier):
    with pytest.raises(ValueError):
        point_config(base(), "load", multiplier)


def test_protocol_guard_rejects_old_safety_and_batching():
    check_protocol(base(), 32, 32, [1980000, 1981000])
    for changed in (dict(avoidance_enabled=True), dict(safety_shield_enabled=False),
                    dict(access_bw_decision_interval=5), dict(num_gu=20)):
        with pytest.raises(ValueError, match="protocol mismatch"):
            check_protocol(dict(base(), **changed), 32, 32, [1980000, 1981000])
    with pytest.raises(ValueError, match="num_envs"):
        check_protocol(base(), 8, 8, [1980000])
    with pytest.raises(ValueError, match="overlap"):
        check_protocol(base(), 64, 32, [1980000, 1980032])


def test_no_cost_c_uses_official_cli_and_does_not_change_other_methods():
    for method in METHODS:
        command = command_for(Path("/source"), Path("/config"), Path("/out"), method, 1980000, 32, 32)
        assert command[1] == "/source/scripts/evaluation/evaluate_structured_fixed_policy.py"
        assert ("--dq_movement_weight" in command) == (method == "distributed_queue_c")
        if method == "distributed_queue_c":
            assert command[-4:] == ["--dq_movement_weight", "0", "--dq_switch_weight", "0"]


def test_aggregate_uses_all_episode_rows_and_labels_episode_sd():
    rows = [dict(axis="nominal", multiplier=1, method="distributed_queue_c",
                 **{key: value for key in METRICS}) for value in (1, 3, 5)]
    result, = aggregate(rows)
    assert result["episodes"] == 3
    assert result["reward_sum_mean"] == 3
    assert result["reward_sum_episode_std"] == 2

def test_paper_roster_replaces_qccs_with_dqs():
    from scripts.experiments.more_gus.run_fixed_baselines import PAPER_LABELS
    assert METHODS == ("distributed_queue_c", "maxweight_lyapunov", "queue_aware_bw", "static_uniform")
    assert "cluster_center_queue_aware" not in METHODS
    assert PAPER_LABELS["distributed_queue_c"] == "DQS"
