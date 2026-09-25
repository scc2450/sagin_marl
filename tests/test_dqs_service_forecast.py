from dataclasses import replace

import pytest
import torch

from sagin_marl.rl.distributed_queue import (
    DQSettings, _predicted_access_service, service_consistent_queue_action, distributed_queue_action,
)
from tests.test_distributed_queue import fixture


def test_forecast_uses_owner_mask_and_prearrival_queue():
    cfg, _ = fixture()
    cfg.b_acc = 1e6
    q = torch.tensor([[[100., 0., 20.]]])
    snr = torch.full((1, 1, 3, 2), 10.)
    owners = torch.tensor([[[0, 0, 1]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    available = q + 100
    served, allocation = _predicted_access_service(q, snr, owners, owners, valid, cfg, available)
    torch.testing.assert_close(allocation, torch.tensor([[[[1., 0.], [0., 0.], [0., 1.]]]]))
    torch.testing.assert_close(served, torch.tensor([[[200., 0., 120.]]]))


def test_forecast_interference_reduces_service():
    cfg, _ = fixture()
    cfg.b_acc = 1e3
    q = torch.full((1, 1, 2), 1e8)
    snr = torch.full((1, 1, 2, 2), 10.)
    owners = torch.tensor([[[0, 1]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    cfg.interference_enabled = False
    clear, _ = _predicted_access_service(q, snr, owners, owners, valid, cfg)
    cfg.interference_enabled = True
    reuse, _ = _predicted_access_service(q, snr, owners, owners, valid, cfg)
    assert (reuse < clear).all()


def test_service_planner_independent_and_masked():
    cfg, obs = fixture()
    cfg.b_acc = 1e5
    cfg.queue_max_gu = 1e8
    settings = DQSettings(movement_weight=0, switch_weight=0)
    obs.gu_mask[:, 1:] = False
    a, diag = service_consistent_queue_action(obs, cfg, settings=settings)
    assert (a[:, 0] > 0).all()
    assert (a.norm(dim=-1) <= 1.000001).all()
    obs.gu_tokens[:, 1:] = float("nan")
    obs.peer_tokens[:] = float("nan")
    b, _ = service_consistent_queue_action(obs, cfg, settings=settings)
    torch.testing.assert_close(a, b)
    obs.gu_tokens[1, 0, 12] *= -1
    c, _ = service_consistent_queue_action(obs, cfg, settings=settings)
    torch.testing.assert_close(a[0], c[0])
    assert c[1, 0] < 0
    expected = obs.ego_features[:, None, :2] * cfg.map_size + diag["candidate_actions"] * cfg.a_max * cfg.tau0**2
    torch.testing.assert_close(diag["first_positions"], expected)


def test_service_planner_brakes_with_no_demand():
    cfg, obs = fixture()
    obs.gu_mask[:] = False
    obs.ego_features[:, 2] = 10 / cfg.v_max
    action, _ = service_consistent_queue_action(obs, cfg, settings=DQSettings(movement_weight=0, switch_weight=0))
    assert (action[:, 0] < 0).all()
    torch.testing.assert_close(action[:, 1], torch.zeros(2))


def test_current_dqs_uses_forecast_and_pressure_resources():
    from sagin_marl.rl.structured_eval import _fixed_policy_exec_sources
    from sagin_marl.rl.distributed_queue import DQS_REVISION
    cfg, obs = fixture()
    cfg.b_acc = 1e5
    actual, _ = distributed_queue_action(obs, cfg)
    expected, _ = service_consistent_queue_action(obs, cfg)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert _fixed_policy_exec_sources("distributed_queue_c") == ("distributed_queue_c", "lyapunov", "lyapunov")
    assert DQS_REVISION == "service-forecast-pressure-v1"


@pytest.mark.parametrize("distance,speed,relative_speed", [(70, 10, 20), (23, 0, 0)])
def test_current_dqs_respects_robust_first_step_screen(distance, speed, relative_speed):
    cfg, obs = fixture()
    obs.ego_features[:, 2] = speed / cfg.v_max
    obs.peer_mask[:, 0] = True
    obs.peer_tokens[:, 0, 0] = -distance / cfg.map_size
    obs.peer_tokens[:, 0, 2] = relative_speed / cfg.v_max
    _, diag = distributed_queue_action(obs, cfg)
    assert (~diag["fallback"]).all()
    chosen = diag["robust_first_clearance"].gather(1, diag["selected"][:, None])
    assert (chosen >= cfg.d_safe).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_service_planner_cpu_cuda_agree():
    cfg, cpu = fixture()
    cfg.b_acc = 1e5
    _, gpu = fixture("cuda")
    settings = replace(DQSettings(), movement_weight=0, switch_weight=0)
    a, _ = service_consistent_queue_action(cpu, cfg, settings=settings)
    b, _ = service_consistent_queue_action(gpu, cfg, settings=settings)
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
