from scripts.experiments.more_gus.run_gc_training import ablation_config
import pytest


def test_gc_changes_only_critic_and_requested_seed():
    cfg = dict(seed=45211, critic_value_mode="relational", actor_lr=1e-4,
               safety_shield_enabled=True, checkpoint_eval_reward_patience=4,
               train_sat=True, fixed_satellite_strategy=False)
    assert ablation_config(cfg, 45211) == dict(cfg, critic_value_mode="global_only")
    assert ablation_config(cfg, 61723) == dict(cfg, seed=61723, critic_value_mode="global_only")
    assert cfg["critic_value_mode"] == "relational"


def test_gc_rejects_nonrelational_reference():
    with pytest.raises(ValueError, match="relational"):
        ablation_config(dict(seed=45211, critic_value_mode="flat_mlp"), 45211)
