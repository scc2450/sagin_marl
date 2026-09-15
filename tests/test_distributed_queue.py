from types import SimpleNamespace
import pytest
import torch
from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.distributed_queue import DQSettings, distributed_queue_action


def fixture(device="cpu"):
    cfg = SaginConfig(num_uav=3, num_gu=4, map_size=1500, task_arrival_rate=4e5)
    ego = torch.zeros(2, 28, device=device)
    ego[:, :2] = 0.5
    gu = torch.zeros(2, 4, 27, device=device)
    gu[..., 3] = 0.5
    gu[..., 4] = 0.693147
    gu[..., 12] = 0.15
    gu[..., 15] = 3.0
    obs = SimpleNamespace(ego_features=ego, gu_tokens=gu,
        gu_mask=torch.ones(2, 4, device=device, dtype=torch.bool),
        peer_tokens=torch.zeros(2, 2, 15, device=device),
        peer_mask=torch.zeros(2, 2, device=device, dtype=torch.bool))
    return cfg, obs


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_actions_are_finite_bounded_and_row_independent(variant):
    cfg, obs = fixture()
    a, _ = distributed_queue_action(obs, cfg, variant)
    assert torch.isfinite(a).all()
    assert (a.norm(dim=-1) <= 1.000001).all()
    obs.gu_tokens[1] *= 50
    obs.ego_features[1, :2] = 0.1
    changed, _ = distributed_queue_action(obs, cfg, variant)
    torch.testing.assert_close(a[0], changed[0])


def test_masked_private_values_do_not_affect_output():
    cfg, obs = fixture()
    obs.gu_mask[:, 1:] = False
    a, _ = distributed_queue_action(obs, cfg)
    obs.gu_tokens[:, 1:] = float("nan")
    obs.peer_tokens[:] = float("nan")
    b, _ = distributed_queue_action(obs, cfg)
    torch.testing.assert_close(a, b)


def test_no_users_brakes_instead_of_accelerating():
    cfg, obs = fixture()
    obs.gu_mask[:] = False
    action, _ = distributed_queue_action(obs, cfg)
    torch.testing.assert_close(action, torch.zeros_like(action))


def test_local_service_direction_changes_with_observation():
    cfg, obs = fixture()
    cfg.queue_max_gu = 1e8
    cfg.b_acc = 1e6
    # Isolate the service prediction from movement/switch regularization.
    settings = DQSettings(movement_weight=0, switch_weight=0)
    right, _ = distributed_queue_action(obs, cfg, "a", settings)
    obs.gu_tokens[..., 12] *= -1
    left, _ = distributed_queue_action(obs, cfg, "a", settings)
    assert (right[:, 0] > 0).all()
    assert (left[:, 0] < 0).all()


def test_head_on_forecast_chooses_feasible_candidate():
    cfg, obs = fixture()
    obs.ego_features[:, 2] = 10 / cfg.v_max
    obs.peer_mask[:, 0] = True
    obs.peer_tokens[:, 0, 0] = 70 / cfg.map_size
    obs.peer_tokens[:, 0, 2] = -20 / cfg.v_max
    _, diag = distributed_queue_action(obs, cfg, "c")
    assert (~diag["fallback"]).all()
    assert (diag["predicted_clearance"] >= cfg.d_safe + 5).all()


def test_crowded_fallback_remains_finite():
    cfg, obs = fixture()
    obs.peer_mask[:] = True
    action, diag = distributed_queue_action(obs, cfg, "c")
    assert diag["fallback"].all()
    assert torch.isfinite(action).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_cpu_same_decision():
    cfg, obs = fixture()
    cpu, _ = distributed_queue_action(obs, cfg)
    cfg, obs = fixture("cuda")
    gpu, _ = distributed_queue_action(obs, cfg)
    torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-5, rtol=1e-5)
