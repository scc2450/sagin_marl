from __future__ import annotations

import pytest

from sagin_marl.env.config import SaginConfig, ablation_flag, load_config, update_config


def test_load_config_coerces_numeric_strings(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "noise_density: 4e-21\n"
        "sat_cpu_freq: 1e10\n"
        "early_stop_enabled: true\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg.noise_density, float)
    assert isinstance(cfg.sat_cpu_freq, float)
    assert cfg.early_stop_enabled is True


def test_load_config_nested_ablation_flags(tmp_path):
    cfg_path = tmp_path / "cfg_ablation.yaml"
    cfg_path.write_text(
        "ablation:\n"
        "  use_imitation_loss: true\n"
        "  use_curriculum_spawn: false\n"
        "ablation_flags:\n"
        "  use_heuristic_mask: true\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.ablation.use_imitation_loss is True
    assert cfg.ablation.use_curriculum_spawn is False
    assert cfg.ablation.use_heuristic_mask is True


def test_load_config_redesigned_bw_actor_knobs(tmp_path):
    cfg_path = tmp_path / "cfg_bw_actor.yaml"
    cfg_path.write_text(
        "bw_down_query_count: 3\n"
        "bw_competition_layers: 1\n"
        "bw_attention_heads: 2\n"
        "bw_tau_min: 0.4\n"
        "bw_tau_max: 1.6\n"
        "bw_kappa_min: 0.75\n"
        "bw_kappa_max: 24.0\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.bw_down_query_count == 3
    assert cfg.bw_competition_layers == 1
    assert cfg.bw_attention_heads == 2
    assert cfg.bw_tau_min == 0.4
    assert cfg.bw_tau_max == 1.6
    assert cfg.bw_kappa_min == 0.75
    assert cfg.bw_kappa_max == 24.0


def test_load_config_rejects_removed_bw_actor_knobs(tmp_path):
    cfg_path = tmp_path / "cfg_bw_removed.yaml"
    cfg_path.write_text(
        "structured_bw_parameterization: score_support_kappa_dirichlet\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="structured_bw_parameterization"):
        load_config(str(cfg_path))


def test_load_config_bw_flow_proxy_aux_flags(tmp_path):
    cfg_path = tmp_path / "cfg_bw_flow_proxy_aux.yaml"
    cfg_path.write_text(
        "bw_flow_proxy_aux_enabled: true\n"
        "bw_flow_proxy_aux_coef: 0.03\n"
        "bw_flow_proxy_aux_delta: 0.07\n"
        "bw_flow_proxy_aux_min_gap: 1e-3\n"
        "bw_flow_proxy_aux_regression_coef: 0.4\n"
        "bw_flow_proxy_grad_diagnostics_enabled: true\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.bw_flow_proxy_aux_enabled is True
    assert cfg.bw_flow_proxy_aux_coef == 0.03
    assert cfg.bw_flow_proxy_aux_delta == 0.07
    assert cfg.bw_flow_proxy_aux_min_gap == 1e-3
    assert cfg.bw_flow_proxy_aux_regression_coef == 0.4
    assert cfg.bw_flow_proxy_grad_diagnostics_enabled is True


def test_load_config_train_trace_flags(tmp_path):
    cfg_path = tmp_path / "cfg_train_trace.yaml"
    cfg_path.write_text(
        "train_trace_enabled: false\n"
        "train_trace_rollout_interval: 17\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.train_trace_enabled is False
    assert cfg.train_trace_rollout_interval == 17


def test_load_config_update_direction_probe_flags(tmp_path):
    cfg_path = tmp_path / "cfg_update_direction_probe.yaml"
    cfg_path.write_text(
        "update_direction_probe_enabled: true\n"
        "update_direction_probe_interval_updates: 3\n"
        "update_direction_probe_start_update: 2\n"
        "update_direction_probe_panel_episodes: 5\n"
        "update_direction_probe_panel_states: 11\n"
        "update_direction_probe_panel_seed: 777\n"
        "update_direction_probe_k_steps: 9\n"
        "update_direction_probe_bw_sample_limit: 13\n"
        "update_direction_probe_actor_policy_mode: stochastic\n"
        "update_direction_probe_actor_policy_samples: 7\n"
        "update_direction_probe_true_mc_enabled: true\n"
        "update_direction_probe_true_mc_samples: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.update_direction_probe_enabled is True
    assert cfg.update_direction_probe_interval_updates == 3
    assert cfg.update_direction_probe_start_update == 2
    assert cfg.update_direction_probe_panel_episodes == 5
    assert cfg.update_direction_probe_panel_states == 11
    assert cfg.update_direction_probe_panel_seed == 777
    assert cfg.update_direction_probe_k_steps == 9
    assert cfg.update_direction_probe_bw_sample_limit == 13
    assert cfg.update_direction_probe_actor_policy_mode == "stochastic"
    assert cfg.update_direction_probe_actor_policy_samples == 7
    assert cfg.update_direction_probe_true_mc_enabled is True
    assert cfg.update_direction_probe_true_mc_samples == 5


def test_ablation_flag_legacy_fallback_compatibility():
    cfg = SaginConfig()
    cfg.imitation_enabled = True
    cfg.ablation.use_imitation_loss = False
    assert ablation_flag(cfg, "use_imitation_loss", fallback_attr="imitation_enabled", default=False) is True


def test_ablation_flag_false_when_both_disabled():
    cfg = SaginConfig()
    cfg.imitation_enabled = False
    cfg.ablation.use_imitation_loss = False
    assert ablation_flag(cfg, "use_imitation_loss", fallback_attr="imitation_enabled", default=False) is False


def test_load_config_queue_init_abs_and_steps(tmp_path):
    cfg_path = tmp_path / "cfg_queue_init.yaml"
    cfg_path.write_text(
        "queue_init_gu_abs: 1234\n"
        "queue_init_uav_steps: 0.5\n"
        "queue_init_sat_steps: 2\n"
        "queue_ref_uav_per_step: 456.5\n"
        "queue_ref_sat_per_step: 789\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.queue_init_gu_abs == 1234
    assert cfg.queue_init_uav_steps == 0.5
    assert cfg.queue_init_sat_steps == 2
    assert cfg.queue_ref_uav_per_step == 456.5
    assert cfg.queue_ref_sat_per_step == 789


def test_load_config_queue_max_steps_uses_layer_refs(tmp_path):
    cfg_path = tmp_path / "cfg_queue_max_steps.yaml"
    cfg_path.write_text(
        "num_gu: 20\n"
        "num_uav: 3\n"
        "num_sat: 144\n"
        "queue_max_gu_steps: 40\n"
        "queue_max_uav_steps: 80\n"
        "queue_max_sat_steps: 120\n"
        "queue_ref_gu_per_step: 2.4e7\n"
        "queue_ref_uav_per_step: 2.4e7\n"
        "queue_ref_sat_per_step: 2.3e7\n"
        "queue_ref_sat_active_count: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.queue_max_gu == 4.8e7
    assert cfg.queue_max_uav == 6.4e8
    assert cfg.queue_max_sat == 6.9e8


def test_update_config_queue_max_steps_falls_back_to_total_sat_count():
    cfg = update_config(
        SaginConfig(),
        {
            "num_sat": 144,
            "queue_max_sat_steps": 120,
            "queue_ref_sat_per_step": 2.16e7,
        },
    )
    assert cfg.queue_max_sat == 1.8e7


def test_load_config_resource_scaling_uses_active_sat_counts_not_total_sat_count(tmp_path):
    cfg_path = tmp_path / "cfg_resource_scale.yaml"
    cfg_path.write_text(
        "num_uav: 1\n"
        "num_gu: 2\n"
        "num_sat: 144\n"
        "task_arrival_rate: 8.0e5\n"
        "sat_num_select: 2\n"
        "queue_ref_sat_active_count: 2\n"
        "resource_scale_enabled: true\n"
        "resource_scale_ref_num_uav: 3\n"
        "resource_scale_ref_num_gu: 20\n"
        "resource_scale_ref_task_arrival_rate: 4.9e5\n"
        "resource_scale_ref_sat_active_count: 3\n"
        "resource_scale_b_acc_multiplier: 0.45\n"
        "b_acc: 1.0e7\n"
        "b_sat_total: 1.5e7\n"
        "sat_cpu_freq: 1.8e10\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    acc_scale = (8.0e5 * 2.0 / 1.0) / (4.9e5 * 20.0 / 3.0)
    sat_scale = (8.0e5 * 2.0 / 2.0) / (4.9e5 * 20.0 / 3.0)
    assert cfg.b_acc == pytest.approx(1.0e7 * acc_scale * 0.45)
    assert cfg.b_backhaul_per_sat == pytest.approx(1.5e7 * sat_scale)
    assert cfg.b_sat_total == pytest.approx(cfg.b_backhaul_per_sat)
    assert cfg.sat_cpu_freq == pytest.approx(1.8e10 * sat_scale)


def test_update_config_resource_scaling_falls_back_to_projected_sat_usage():
    cfg = update_config(
        SaginConfig(),
        {
            "num_uav": 1,
            "num_gu": 2,
            "num_sat": 144,
            "task_arrival_rate": 8.0e5,
            "sat_num_select": 2,
            "resource_scale_enabled": True,
            "resource_scale_ref_num_uav": 3,
            "resource_scale_ref_num_gu": 20,
            "resource_scale_ref_task_arrival_rate": 4.9e5,
            "resource_scale_ref_sat_active_count": 3,
            "b_sat_total": 1.5e7,
            "sat_cpu_freq": 1.8e10,
        },
    )
    sat_scale = (8.0e5 * 2.0 / 2.0) / (4.9e5 * 20.0 / 3.0)
    assert cfg.b_backhaul_per_sat == pytest.approx(1.5e7 * sat_scale)
    assert cfg.b_sat_total == pytest.approx(cfg.b_backhaul_per_sat)
    assert cfg.sat_cpu_freq == pytest.approx(1.8e10 * sat_scale)
