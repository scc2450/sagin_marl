from types import SimpleNamespace
import pytest
import torch
from scripts.render_structured_episode import _state_from_full_accel_observation


def fixture():
    cfg = SimpleNamespace(num_uav=2, num_gu=2, map_size=1000, queue_max_gu=1e6, T_steps=250)
    ego = torch.zeros(2, 28)
    ego[:, :2] = torch.tensor([[0.2, 0.3], [0.8, 0.7]])
    ego[:, 27] = 1
    gu = torch.zeros(2, 2, 27)
    gu[:, :, :2] = torch.tensor([[0.1, 0.2], [0.6, 0.5]])
    gu[:, :, 3] = torch.tensor([0.2, 0.4])
    gu[0, 0, 16] = 1
    gu[1, 1, 16] = 1
    return cfg, SimpleNamespace(ego_features=ego, gu_tokens=gu,
        gu_mask=torch.ones(2, 2, dtype=torch.bool))


@pytest.mark.parametrize("t", [0, 125, 249])
def test_decode_native_snapshot(t):
    cfg, obs = fixture()
    obs.ego_features[:, 27] = (249-t)/249
    state = _state_from_full_accel_observation(cfg, obs)
    assert state["t"] == t
    assert state["last_association"] == [0, 1]
    assert state["uav_pos"][0] == pytest.approx([200, 300])
    assert state["gu_queue"] == pytest.approx([200000, 400000])


def test_partial_observation_is_not_silently_rendered_as_complete():
    cfg, obs = fixture()
    obs.gu_mask[0, 1] = False
    with pytest.raises(ValueError, match="all GU"):
        _state_from_full_accel_observation(cfg, obs)


def test_conflicting_associations_rejected():
    cfg, obs = fixture()
    obs.gu_tokens[1, 0, 16] = 1
    with pytest.raises(ValueError, match="Conflicting"):
        _state_from_full_accel_observation(cfg, obs)
