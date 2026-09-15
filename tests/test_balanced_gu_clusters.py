from pathlib import Path

import numpy as np
import pytest
import torch

from sagin_marl.env.config import SaginConfig, load_config
from sagin_marl.env.topology import thomas_cluster_process
from sagin_marl.env.structured_batch_env_core import StructuredBatchEnvCore


@pytest.mark.parametrize("num_gu,num_clusters", [(100, 22), (100, 20), (100, 25), (3, 5)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_balanced_cluster_population_and_seed(num_gu, num_clusters, backend):
    def sample():
        if backend == "numpy":
            return thomas_cluster_process(
                num_gu, 1500.0, num_clusters=num_clusters, balanced=True,
                rng=np.random.default_rng(45211), return_metadata=True,
            )
        cfg = SaginConfig(num_gu=num_gu, gu_init_num_clusters=num_clusters,
                          gu_init_cluster_balanced=True)
        result = StructuredBatchEnvCore._sample_native_reset_gu_tape(
            None, cfg, generator=torch.Generator().manual_seed(45211),
            device=torch.device("cpu"),
        )
        return tuple(t.numpy() for t in result)

    points, centers, counts = sample()
    assert points.shape == (num_gu, 2)
    assert centers.shape == (num_clusters, 2)
    assert counts.sum() == num_gu
    assert counts.max() - counts.min() <= 1
    assert np.all((points >= 0) & (points <= 1500))
    for actual, repeated in zip((points, centers, counts), sample()):
        np.testing.assert_array_equal(actual, repeated)


def test_default_cluster_sampling_unchanged():
    def sample(**kwargs):
        return thomas_cluster_process(20, 1500.0, rng=np.random.default_rng(42),
                                      return_metadata=True, **kwargs)
    for default, explicit in zip(sample(), sample(balanced=False)):
        np.testing.assert_array_equal(default, explicit)


def test_more_gus_config_uses_full_gu_width_and_policy_satellite():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "configs/experiments/more_gus/structured_joint_mcgae_3uav100gu_22clusters_t250.yaml"))
    assert (cfg.num_gu, cfg.gu_init_num_clusters) == (100, 22)
    assert cfg.gu_init_cluster_balanced
    assert cfg.users_obs_max == cfg.candidate_k == 100
    assert not cfg.fixed_satellite_strategy
    assert cfg.exec_sat_source == "policy"
